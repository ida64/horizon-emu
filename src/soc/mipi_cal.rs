//! MIPI pad calibration: the block that trims the drive of the DSI and CSI
//! pads.
//!
//! `boot` writes its configuration while bringing up the display link and
//! never reads it back. nvservices runs a calibration after putting the DSI
//! link in video mode (text `0x3cca0`): it programs the pads, writes the
//! control register and then sets its start bit by read-modify-write, and
//! polls the status register until the calibration is done and no longer
//! active, `(status & 0x10001) == 0x10000`, for 2 ms (a timeout is recorded
//! as an error, not fatal). Bit names are those of Linux's `host1x/mipi.c`
//! (public), which waits for the same condition.
//!
//! There are no pads to measure here: a calibration is over as soon as it
//! is started. Everything else is configuration without effect. Some of it
//! nvservices changes by read-modify-write before ever writing it (`+0x5c`,
//! text `0x3cca0`), so registers read as zero until written; that stands
//! for whatever earlier boot stages left, an assumption about bits the
//! firmware does not look at again.

use super::regs::{RegisterFile, word};
use crate::{bus::MmioDevice, snapshot::Snapshot};

pub const BASE: u64 = 0x700E_3000;
/// nvservices maps 0x100 bytes (text `0x22fe8`).
pub const SIZE: u64 = 0x100;

const CTRL: u64 = 0x00;
const STATUS: u64 = 0x08;

/// Starts a calibration.
const CTRL_START: u32 = 1 << 0;
/// A calibration has finished; `STATUS_ACTIVE` would say one is running.
const STATUS_DONE: u32 = 1 << 16;

pub struct MipiCal {
    registers: RegisterFile,
}

impl Default for MipiCal {
    fn default() -> Self {
        Self {
            registers: RegisterFile::zeroed([]),
        }
    }
}

impl MmioDevice for MipiCal {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        let register = word(offset, size)?;
        if register == STATUS {
            // Its other bits are per-pad results nothing reads; before any
            // calibration nothing has been measured to report.
            return self
                .registers
                .get(STATUS)
                .filter(|&status| status != 0)
                .map(u64::from);
        }
        self.registers.get(register).map(u64::from)
    }

    fn write(&self, _core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        let register = word(offset, size)?;
        let value = value as u32;
        match register {
            // Write one to clear (nvservices writes 0xf1f00000 first, text
            // 0x3af30): of the bits the model sets, it leaves the result of
            // the last calibration alone.
            STATUS => {
                self.registers.update(STATUS, |status| status & !value)?;
            }
            // The calibration starts and is over; the start bit, a trigger,
            // reads back clear.
            CTRL if value & CTRL_START != 0 => {
                self.registers.set(CTRL, value & !CTRL_START);
                self.registers.set(STATUS, STATUS_DONE);
            }
            _ => self.registers.set(register, value),
        }
        Some(())
    }

    /// All there is: the registers, the last calibration's result included.
    fn snapshot(&self) -> Option<&dyn Snapshot> {
        Some(&self.registers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibration_is_done_once_started_and_its_start_bit_reads_clear() {
        let cal = MipiCal::default();
        assert_eq!(cal.read(0, STATUS, 4), None, "nothing calibrated yet");
        // nvservices' sequence (text 0x3af30, 0x3cca0).
        cal.write(0, STATUS, 4, 0xF1F0_0000).unwrap();
        assert_eq!(cal.read(0, 0x5C, 4), Some(0));
        cal.write(0, CTRL, 4, 0x2A00_0000).unwrap();
        let ctrl = cal.read(0, CTRL, 4).unwrap();
        cal.write(0, CTRL, 4, ctrl | 1).unwrap();
        let status = cal.read(0, STATUS, 4).unwrap() as u32;
        assert_eq!(status & 0x10001, 0x10000);
        assert_eq!(cal.read(0, CTRL, 4), Some(0x2A00_0000));
        // Clearing the pad bits again leaves the result.
        cal.write(0, STATUS, 4, 0xF1F0_0000).unwrap();
        assert_eq!(cal.read(0, STATUS, 4), Some(0x10000));
    }
}
