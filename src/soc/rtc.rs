//! The SoC's real-time clock (`0x7000_E000`), in the always-on domain
//! beside the PMC: a seconds counter with milliseconds. Horizon keeps time
//! with the PMIC's clock instead; hekate times its delays and timeouts on
//! this one (`timer.c`: "RTC_MILLI_SECONDS (0x10) -> RTC_SHADOW_SECONDS
//! (0xC)", in that order).
//!
//! Registers (names and offsets as hekate `t210.h` and Linux `rtc-tegra.c`
//! have them, reference for facts):
//!
//! * `SECONDS` (`0x08`): whole seconds counted.
//! * `MILLI_SECONDS` (`0x10`): the milliseconds of the current second, 0 to
//!   999. Reading it copies the seconds into the shadow register, so that
//!   a reader gets the two halves of one instant (Linux: "RTC hardware
//!   copies seconds to shadow seconds when a read of milliseconds occurs").
//! * `SHADOW_SECONDS` (`0x0C`): the seconds as of the last read of
//!   `MILLI_SECONDS`; 0 before the first.
//!
//! The count is the machine's guest time: \[assumption\] it starts at 0
//! when the machine powers on. On a console it has run since the battery
//! was connected; software takes differences. Setting the clock, its
//! alarms and its interrupt are not modelled: every other access is
//! refused.

use std::{sync::Mutex, time::Duration};

use super::regs::word;
use crate::{
    bus::MmioDevice,
    snapshot::{Snapshot, SnapshotError, decode, encode},
};

pub const BASE: u64 = 0x7000_E000;
pub const SIZE: u64 = 0x100;

const SECONDS: u64 = 0x08;
const SHADOW_SECONDS: u64 = 0x0C;
const MILLI_SECONDS: u64 = 0x10;

pub struct Rtc {
    /// Time since the machine was powered on.
    elapsed: Box<dyn Fn() -> Duration + Send + Sync>,
    shadow_seconds: Mutex<u32>,
}

impl Rtc {
    pub fn new(elapsed: Box<dyn Fn() -> Duration + Send + Sync>) -> Self {
        Self {
            elapsed,
            shadow_seconds: Mutex::new(0),
        }
    }
}

impl MmioDevice for Rtc {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        let now = (self.elapsed)();
        // The counter is 32 bits wide and wraps.
        let seconds = now.as_secs() as u32;
        let value = match word(offset, size)? {
            SECONDS => seconds,
            SHADOW_SECONDS => *self.shadow_seconds.lock().unwrap(),
            MILLI_SECONDS => {
                *self.shadow_seconds.lock().unwrap() = seconds;
                now.subsec_millis()
            }
            _ => return None,
        };
        Some(u64::from(value))
    }

    fn write(&self, _core: usize, _offset: u64, _size: u8, _value: u64) -> Option<()> {
        None
    }

    fn snapshot(&self) -> Option<&dyn Snapshot> {
        Some(self)
    }
}

/// The shadow register; the count is guest time, which the machine's clock
/// saves.
impl Snapshot for Rtc {
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        encode(&*self.shadow_seconds.lock().unwrap())
    }

    /// Nothing saved is a snapshot from before the block was modelled.
    fn restore(&self, state: &[u8]) -> Result<(), SnapshotError> {
        if !state.is_empty() {
            *self.shadow_seconds.lock().unwrap() = decode("RTC", state)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn the_shadow_holds_the_seconds_of_the_last_milliseconds_read() {
        let now = Arc::new(Mutex::new(Duration::from_millis(2_750)));
        let clock = Arc::clone(&now);
        let rtc = Rtc::new(Box::new(move || *clock.lock().unwrap()));
        assert_eq!(rtc.read(4, SHADOW_SECONDS, 4), Some(0));
        assert_eq!(rtc.read(4, MILLI_SECONDS, 4), Some(750));
        *now.lock().unwrap() = Duration::from_millis(3_100);
        assert_eq!(rtc.read(4, SHADOW_SECONDS, 4), Some(2));
        assert_eq!(rtc.read(4, SECONDS, 4), Some(3));
        assert_eq!(rtc.read(4, MILLI_SECONDS, 4), Some(100));
        assert_eq!(rtc.read(4, SHADOW_SECONDS, 4), Some(3));
        assert_eq!(rtc.write(4, SECONDS, 4, 0), None);
    }
}
