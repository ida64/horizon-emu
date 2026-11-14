//! DPAUX1: the auxiliary channel and hot-plug detection of the external
//! DisplayPort output, as far as nvservices uses it on a console that is
//! not docked.
//!
//! nvservices brings the external head up at start-up (text `0x460b0`,
//! `0x424e0`): it clears and enables the channel's interrupts, pulses a bit
//! of the control register by read-modify-write (the first access to it is
//! a read), configures the pads and hot-plug timing, then polls the
//! hot-plug status bit for about 80 ms. With it low the output counts as
//! disconnected and nothing else happens on the channel; a thread waits on
//! INTID 43 for a plug event. Register and bit names are those of Linux's
//! `tegra/dpaux.c` (public).
//!
//! Nothing is plugged in here, ever. The status register reports no sink
//! and no transaction; the interrupt status has nothing pending, and the
//! interrupt line, left unconnected, stays low. A transaction request,
//! which only follows a plug, is refused. The rest is configuration;
//! registers read as zero until written, an assumption about bits the
//! firmware does not use.

use super::regs::{RegisterFile, word};
use crate::bus::MmioDevice;

pub const BASE: u64 = 0x5404_0000;
/// Up to the last register nvservices uses, the hybrid pad spare register
/// at `0x134`.
pub const SIZE: u64 = 0x200;

const INTR_AUX: u64 = 0x14;
const DP_AUXCTL: u64 = 0xB4;
const DP_AUXSTAT: u64 = 0xC4;

/// Starts an AUX transaction.
const AUXCTL_TRANSACTREQ: u32 = 1 << 16;

pub struct Dpaux {
    registers: RegisterFile,
}

impl Default for Dpaux {
    fn default() -> Self {
        Self {
            registers: RegisterFile::zeroed([]),
        }
    }
}

impl MmioDevice for Dpaux {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        let value = match word(offset, size)? {
            // No sink (bit 28, HPD_STATUS, clear), no reply, no error.
            DP_AUXSTAT => 0,
            // Plug, unplug and transaction events never happen.
            INTR_AUX => 0,
            register => self.registers.get(register)?,
        };
        Some(value.into())
    }

    fn write(&self, _core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        let value = value as u32;
        match word(offset, size)? {
            // Write one to clear, of nothing.
            INTR_AUX => {}
            DP_AUXCTL if value & AUXCTL_TRANSACTREQ != 0 => return None,
            DP_AUXSTAT => return None,
            register => self.registers.set(register, value),
        }
        Some(())
    }
}
