//! XUSB device controller configuration and FPCI windows.
//!
//! Horizon usb `0x2f4d0` checks FPCI vendor/device 10de:0fad before
//! programming BARs, IPFS routing and pad timing. The identity is required
//! by that exact comparison, not guessed from a different Tegra device.
//! Only its observed configuration registers are retained; all other
//! accesses refuse until their semantics are established.
//! Public register semantics: Linux `drivers/usb/gadget/udc/tegra-xudc.c`.
//! No cable means PORTSC has no connection, speed, or change bits and
//! no event TRBs are produced. Endpoint reload completes synchronously;
//! pause changes raise EP_STCHG bits, which software acknowledges W1C.
use super::regs::{restore_words, save_words};
use crate::{
    bus::MmioDevice,
    snapshot::{Snapshot, SnapshotError},
};
use std::sync::Mutex;
pub const BASE: u64 = 0x700d_0000;
pub const SIZE: u64 = 0xa000;
pub struct XusbDevice {
    registers: Mutex<[u32; 0x2800]>,
}
impl Default for XusbDevice {
    fn default() -> Self {
        Self {
            registers: Mutex::new([0; 0x2800]),
        }
    }
}
fn configuration(offset: u64) -> bool {
    matches!(offset, 0x000 | 0x008 | 0x010..=0x02c | 0x030..=0x044 |
        0x050..=0x05c | 0x06c | 0x750 | 0x610 | 0x688 | 0x690 |
        0x858 | 0x85c | 0x8004 | 0x8010 | 0x8014 |
        0x9000 | 0x9040 | 0x9080 | 0x9180 | 0x9188)
}
impl MmioDevice for XusbDevice {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        if size != 4 || offset & 3 != 0 {
            return None;
        }
        if offset == 0x8000 {
            return Some(0x0fad_10de);
        }
        if !configuration(offset) {
            return None;
        }
        Some(self.registers.lock().unwrap()[offset as usize / 4] as u64)
    }
    fn write(&self, _core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        if size == 8 && matches!(offset, 0x10 | 0x18 | 0x20 | 0x28 | 0x40) {
            let mut registers = self.registers.lock().unwrap();
            registers[offset as usize / 4] = value as u32;
            registers[offset as usize / 4 + 1] = (value >> 32) as u32;
            return Some(());
        }
        if size != 4 || offset & 3 != 0 || !configuration(offset) {
            return None;
        }
        let mut registers = self.registers.lock().unwrap();
        let index = offset as usize / 4;
        match offset {
            0x34 | 0x5c => registers[index] &= !(value as u32),
            0x3c => {}
            0x58 => registers[index] = 0,
            0x54 => {
                registers[0x5c / 4] |= registers[index] ^ value as u32;
                registers[index] = value as u32;
            }
            _ => registers[index] = value as u32,
        }
        Some(())
    }

    fn snapshot(&self) -> Option<&dyn Snapshot> {
        Some(self)
    }
}

/// Saves the configuration registers and the endpoint state-change bits.
impl Snapshot for XusbDevice {
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        save_words(&self.registers)
    }

    fn restore(&self, state: &[u8]) -> Result<(), SnapshotError> {
        restore_words("XUSB device", &self.registers, state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn endpoint_pause_reload_and_disconnected_port() {
        let usb = XusbDevice::default();
        usb.write(0, 0x54, 4, 3).unwrap();
        assert_eq!(usb.read(0, 0x5c, 4), Some(3));
        usb.write(0, 0x5c, 4, 1).unwrap();
        assert_eq!(usb.read(0, 0x5c, 4), Some(2));
        usb.write(0, 0x58, 4, 3).unwrap();
        assert_eq!(usb.read(0, 0x58, 4), Some(0));
        usb.write(0, 0x3c, 4, u32::MAX.into()).unwrap();
        assert_eq!(usb.read(0, 0x3c, 4), Some(0));
        usb.write(0, 0x10, 8, 0x1234_5678_9abc_def0).unwrap();
        assert_eq!(usb.read(0, 0x14, 4), Some(0x1234_5678));
    }
}
