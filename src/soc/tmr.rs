//! The timer block's microsecond counter.
//!
//! `TIMERUS_CNTR_1US` counts microseconds from power-on, and the firmware's
//! clock driver spins on it for its short delays (pcv maps the block and
//! reads offset `0x10` in a loop). The block's timers, watchdogs and the
//! counter's divider setting are not modeled.

use std::time::Duration;

use super::regs::word;
use crate::bus::MmioDevice;

pub const BASE: u64 = 0x6000_5000;
pub const SIZE: u64 = 0x400;

const TIMERUS_CNTR_1US: u64 = 0x10;

pub struct Timers {
    /// Time since the machine was powered on.
    elapsed: Box<dyn Fn() -> Duration + Send + Sync>,
}

impl Timers {
    pub fn new(elapsed: Box<dyn Fn() -> Duration + Send + Sync>) -> Self {
        Self { elapsed }
    }
}

impl MmioDevice for Timers {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        match word(offset, size)? {
            // The counter is 32 bits wide and wraps.
            TIMERUS_CNTR_1US => Some(u64::from((self.elapsed)().as_micros() as u32)),
            _ => None,
        }
    }

    fn write(&self, _core: usize, _offset: u64, _size: u8, _value: u64) -> Option<()> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counter_wraps_at_32_bits() {
        let timers = Timers::new(Box::new(|| Duration::from_micros((1 << 32) + 7)));
        assert_eq!(timers.read(0, TIMERUS_CNTR_1US, 4), Some(7));
        assert_eq!(timers.read(0, TIMERUS_CNTR_1US, 1), None);
        assert_eq!(timers.read(0, 0, 4), None);
    }
}
