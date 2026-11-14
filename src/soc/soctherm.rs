//! Tegra SOCTHERM overcurrent/throttle controller at rest.
//!
//! ptm `0x2ad60` initializes clock, throttle priority, two overcurrent
//! filters and pulse-skip controls. `0x2af50` polls OC_INTR_STATUS;
//! `0x2b030/0x2b090/0x2b0f0` read CPU/GPU pulse-skip and throttle status.
//! There is no overcurrent or thermal breach at rest, so those status
//! registers remain clear. Writes to OC_INTR_STATUS acknowledge W1C;
//! OC_INTR_ENABLE/DISABLE are set/clear aliases. Public register semantics:
//! <https://github.com/torvalds/linux/blob/master/drivers/thermal/tegra/soctherm.c>.
//! The observed firmware path uses TMP451 temperatures instead of raw die
//! sensors and does not read SOCTHERM calibration fuses. Unknown sensor
//! registers refuse rather than inventing conversion/calibration data.
use crate::bus::MmioDevice;
use std::sync::Mutex;
pub const BASE: u64 = 0x700e_2000;
pub const SIZE: u64 = 0x1000;
pub struct Soctherm {
    registers: Mutex<[u32; 0x160]>,
}
impl Default for Soctherm {
    fn default() -> Self {
        Self {
            registers: Mutex::new([0; 0x160]),
        }
    }
}
fn configuration(offset: u64) -> bool {
    matches!(offset, 0x1dc | 0x2f0 | 0x310 | 0x318 | 0x320 | 0x324 | 0x32c |
        0x334 | 0x338 | 0x34c | 0x360 | 0x3c4 | 0x400 | 0x424 |
        0x490..=0x4ac | 0x4c0..=0x4dc | 0x4f0 | 0x4f8 | 0x520 | 0x528 | 0x550 | 0x558)
}
impl MmioDevice for Soctherm {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        // ptm `0x2b640` tests single OC_INTR_STATUS bits with byte loads
        // (`LDRB [base, #0x39c]`); a narrow read returns its lanes of the word.
        let word = offset & !3;
        let shift = (offset & 3) * 8;
        if !(size == 4 && shift == 0 || size == 1 && word == 0x39c) {
            return None;
        }
        let value = match word {
            0x39c | 0x418 | 0x41c | 0x428 => 0,
            0x3a0 | 0x3a4 => self.registers.lock().unwrap()[0x3a0 / 4],
            _ if configuration(word) => self.registers.lock().unwrap()[word as usize / 4],
            _ => return None,
        };
        Some(u64::from(value >> shift) & (u64::MAX >> (64 - 8 * u64::from(size))))
    }
    fn write(&self, _core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        if size != 4 || offset & 3 != 0 {
            return None;
        }
        let mut registers = self.registers.lock().unwrap();
        match offset {
            0x39c => {}
            0x3a0 => registers[0x3a0 / 4] |= value as u32,
            0x3a4 => registers[0x3a0 / 4] &= !(value as u32),
            _ if configuration(offset) => registers[offset as usize / 4] = value as u32,
            _ => return None,
        }
        Some(())
    }
}
