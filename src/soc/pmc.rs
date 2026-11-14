//! Power management controller: pad voltages, deep power down requests,
//! wake sources, scratch registers and the power partitions. Reached only
//! through the secure monitor's register call.
//!
//! Most of it is configuration storage. `boot` sets up the wake pins by
//! reading a register whole and writing it back whole, so what the block
//! starts with only survives in bits `boot` leaves alone; registers start
//! at zero, which stands for whatever the earlier boot stages programmed.
//!
//! The power partitions behave: the firmware's clock driver turns a
//! partition on or off by writing its ID with the start bit to
//! `PWRGATE_TOGGLE`, waits for the start bit to clear ("accept"), then for
//! the partition's bit in `PWRGATE_STATUS` (`VccPgSetPowerEnabled`, pcv
//! text `0x3d730`: up to 100 polls each, and it fails the request
//! otherwise). Here a partition switches at once. Register and partition
//! names are as in Linux (`drivers/soc/tegra/pmc.c`, Tegra210 table),
//! recalled.

use std::sync::Mutex;

use super::regs::{RegisterFile, word};
use crate::bus::MmioDevice;

pub const BASE: u64 = 0x7000_E400;
pub const SIZE: u64 = 0xC00;

const PWRGATE_TOGGLE: u64 = 0x30;
const TOGGLE_START: u32 = 1 << 8;
const TOGGLE_PARTITION: u32 = 0x1F;
const PWRGATE_STATUS: u64 = 0x38;

/// Partitions powered at kernel entry: the CPU rail (CRAIL, 0), the
/// non-CPU part of the fast cluster (C0NC, 15) and its first core (CE0, 14)
/// — fusee `fusee_cpu.cpp:141-143` powers these before starting the CPU —
/// and the other three cores (CE1-3, 9-11), which the secure monitor
/// powers on when the kernel starts them (the kernel runs on four cores).
/// Every other partition is assumed off: no public stage of the boot chain
/// powers one, and the firmware's clock driver turns on what it needs.
const POWERED_AT_ENTRY: u32 = 1 << 0 | 1 << 9 | 1 << 10 | 1 << 11 | 1 << 14 | 1 << 15;

/// The GPU rail's gate: bit 0 set clamps the rail off. The firmware's
/// clock driver reads it when it starts, to count the rail as in use if
/// it finds it open, and writes 0 to open and 1 to close it (pcv text
/// `0x3db30`, `0x3d9c0`); with the rail counted twice it refuses the power
/// request by which nvservices ends its start-up GPU cycle. The rail is
/// closed at entry: the GPU's regulator is off (see `max77621`), and no
/// boot stage uses the GPU.
const GPU_RG_CNTRL: u64 = 0x2D4;
const GPU_RAIL_CLAMPED: u32 = 1 << 0;

pub struct Pmc {
    registers: RegisterFile,
    powered: Mutex<u32>,
}

impl Pmc {
    pub fn new() -> Self {
        Self {
            registers: RegisterFile::zeroed([(GPU_RG_CNTRL, GPU_RAIL_CLAMPED)]),
            powered: Mutex::new(POWERED_AT_ENTRY),
        }
    }
}

impl Default for Pmc {
    fn default() -> Self {
        Self::new()
    }
}

impl MmioDevice for Pmc {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        let value = match word(offset, size)? {
            PWRGATE_STATUS => *self.powered.lock().unwrap(),
            // A toggle is accepted at once, so the start bit never reads set.
            PWRGATE_TOGGLE => self.registers.get(PWRGATE_TOGGLE)? & !TOGGLE_START,
            register => self.registers.get(register)?,
        };
        Some(value.into())
    }

    fn write(&self, _core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        let register = word(offset, size)?;
        let value = value as u32;
        match register {
            // Read-only.
            PWRGATE_STATUS => return None,
            PWRGATE_TOGGLE if value & TOGGLE_START != 0 => {
                *self.powered.lock().unwrap() ^= 1 << (value & TOGGLE_PARTITION);
            }
            _ => {}
        }
        self.registers.set(register, value);
        Some(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_toggle_switches_one_partition_and_is_accepted_at_once() {
        let pmc = Pmc::new();
        let status = |pmc: &Pmc| pmc.read(0, PWRGATE_STATUS, 4).unwrap() as u32;
        assert_eq!(status(&pmc) & 1 << 3, 0);
        pmc.write(0, PWRGATE_TOGGLE, 4, u64::from(TOGGLE_START | 3))
            .unwrap();
        assert_eq!(pmc.read(0, PWRGATE_TOGGLE, 4), Some(3));
        assert_eq!(status(&pmc), POWERED_AT_ENTRY | 1 << 3);
        pmc.write(0, PWRGATE_TOGGLE, 4, u64::from(TOGGLE_START | 3))
            .unwrap();
        assert_eq!(status(&pmc), POWERED_AT_ENTRY);
        // Without the start bit nothing switches.
        pmc.write(0, PWRGATE_TOGGLE, 4, 3).unwrap();
        assert_eq!(status(&pmc), POWERED_AT_ENTRY);
    }
}
