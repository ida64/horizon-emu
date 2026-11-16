//! XUSB UPHY pads shared with PCIe.
//!
//! Horizon 23.5.0 `pcie`, text `0x18c50`, initializes +0x360..+0x38c,
//! then waits for CAL_DONE (+0x364 bit 1) to follow CAL_EN (bit 0),
//! LOCKDET (+0x360 bit 15) after ENABLE (bit 3), and RCAL_DONE
//! (+0x37c bit 31) to follow RCAL_EN (bit 12), including deassertion.
//! Calibration completes synchronously: there is no analog settling to simulate.
//! Lane controls +0x464 through +0x5e4 and +0x964, mux +0x28 and
//! clamp +0x24 are the same routine's retained configuration. Initial
//! configuration is assumed zero; unknown registers refuse access.
//! USB text `0x5b00` programs bias tracking at +0x284/+0x288 and
//! waits fixed settling intervals (no status polling). `0x5dc0` programs
//! USB2 mux, capability and OTG pad controls; `0x63c0` changes VBUS/ID
//! override +0xc60. These registers retain software configuration.

use super::regs::{restore_words, save_words};
use crate::{
    bus::MmioDevice,
    snapshot::{Snapshot, SnapshotError},
};
use std::sync::Mutex;

pub const BASE: u64 = 0x7009_f000;
pub const SIZE: u64 = 0x1000;

pub struct XusbPadctl {
    registers: Mutex<[u32; 1024]>,
}

impl Default for XusbPadctl {
    fn default() -> Self {
        Self {
            registers: Mutex::new([0; 1024]),
        }
    }
}

fn index(offset: u64, size: u8) -> Option<usize> {
    if size != 4 || offset & 3 != 0 {
        return None;
    }
    match offset {
        0x004
        | 0x008
        | 0x010
        | 0x014
        | 0x018
        | 0x024
        | 0x028
        | 0x080..=0x08c
        | 0x280..=0x288
        | 0x360..=0x38c
        | 0x464
        | 0x4a4
        | 0x4e4
        | 0x524
        | 0x564
        | 0x5a4
        | 0x5e4
        | 0x964
        | 0xc60 => Some(offset as usize / 4),
        _ => None,
    }
}

impl MmioDevice for XusbPadctl {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        Some(self.registers.lock().unwrap()[index(offset, size)?] as u64)
    }
    fn write(&self, _core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        let index = index(offset, size)?;
        let mut value = value as u32;
        let handshake = match offset {
            0x360 => Some((3, 15)),
            0x364 => Some((0, 1)),
            0x37c => Some((12, 31)),
            _ => None,
        };
        if let Some((enable, done)) = handshake {
            value = (value & !(1 << done)) | (((value >> enable) & 1) << done);
        }
        self.registers.lock().unwrap()[index] = value;
        Some(())
    }

    fn snapshot(&self) -> Option<&dyn Snapshot> {
        Some(self)
    }
}

/// Saves the pad configuration, calibration status included.
impl Snapshot for XusbPadctl {
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        save_words(&self.registers)
    }

    fn restore(&self, state: &[u8]) -> Result<(), SnapshotError> {
        restore_words("XUSB pads", &self.registers, state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn calibration_and_lock_follow_enable_not_written_status() {
        let pads = XusbPadctl::default();
        for (offset, enable, done) in [(0x360, 3, 15), (0x364, 0, 1), (0x37c, 12, 31)] {
            pads.write(0, offset, 4, 1 << enable).unwrap();
            assert_eq!(pads.read(0, offset, 4), Some((1 << enable) | (1 << done)));
            pads.write(0, offset, 4, 1 << done).unwrap();
            assert_eq!(pads.read(0, offset, 4), Some(0));
        }
        assert_eq!(pads.read(0, 0x390, 4), None);
    }
}
