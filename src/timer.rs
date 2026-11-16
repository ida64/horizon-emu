//! ARM generic timer: the system counter and each core's EL1 physical timer.
//!
//! The kernel drives everything time-related from `CNTPCT_EL0` and
//! `CNTP_CTL_EL0`/`CNTP_CVAL_EL0` (`0x800ceea0`), and assumes the console's
//! 19.2 MHz counter without ever reading `CNTFRQ_EL0`. The host counter runs
//! at another rate, so the counter is synthesized from guest time
//! ([`GuestClock`]) instead of exposing the host's.

use std::{
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use serde::{Deserialize, Serialize};

use crate::{
    boot::CORE_COUNT,
    clock::GuestClock,
    snapshot::{Snapshot, SnapshotError, decode, encode},
};

/// System counter frequency of the console.
pub const COUNTER_HZ: u64 = 19_200_000;

/// Interrupt ID of the non-secure EL1 physical timer (PPI 14), as the kernel
/// binds it (`0x800ceed4`).
pub const PHYSICAL_TIMER_INTID: u32 = 30;

const CONTROL_ENABLE: u64 = 1 << 0;
const CONTROL_IMASK: u64 = 1 << 1;
const CONTROL_ISTATUS: u64 = 1 << 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimerRegister {
    /// `CNTPCT_EL0`.
    Count,
    /// `CNTP_CTL_EL0`.
    Control,
    /// `CNTP_CVAL_EL0`.
    CompareValue,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
struct PhysicalTimer {
    /// ENABLE and IMASK only; ISTATUS is derived.
    control: u64,
    compare_value: u64,
    /// Level last driven onto the interrupt line.
    output: bool,
}

impl PhysicalTimer {
    fn condition_met(self, count: u64) -> bool {
        self.control & CONTROL_ENABLE != 0 && count >= self.compare_value
    }

    /// Level of the timer's interrupt output.
    fn line(self, count: u64) -> bool {
        self.condition_met(count) && self.control & CONTROL_IMASK == 0
    }
}

/// Called with `(core, level)` to drive the timer interrupt line of a core.
pub type TimerLine = Box<dyn Fn(usize, bool) + Send + Sync>;

pub struct Timers {
    clock: Arc<GuestClock>,
    line: TimerLine,
    /// Held across every `line` call so that the levels reach the interrupt
    /// controller in the order they were computed.
    timers: Mutex<[PhysicalTimer; CORE_COUNT]>,
    changed: Condvar,
}

impl Timers {
    /// The counter counts guest time ([`GuestClock`]): zero when the machine
    /// is built.
    pub fn new(clock: Arc<GuestClock>, line: TimerLine) -> Self {
        Self {
            clock,
            line,
            timers: Mutex::new([PhysicalTimer::default(); CORE_COUNT]),
            changed: Condvar::new(),
        }
    }

    pub fn count(&self) -> u64 {
        ticks(self.clock.now())
    }

    /// Reads a register as `core` sees it.
    pub fn read(&self, core: usize, register: TimerRegister) -> u64 {
        let count = self.count();
        let timer = self.timers.lock().unwrap()[core];
        match register {
            TimerRegister::Count => count,
            TimerRegister::Control if timer.condition_met(count) => timer.control | CONTROL_ISTATUS,
            TimerRegister::Control => timer.control,
            TimerRegister::CompareValue => timer.compare_value,
        }
    }

    /// Writes a register of `core`'s timer; the counter itself is read-only.
    pub fn write(&self, core: usize, register: TimerRegister, value: u64) -> Option<()> {
        let mut timers = self.timers.lock().unwrap();
        match register {
            TimerRegister::Count => return None,
            TimerRegister::Control => {
                timers[core].control = value & (CONTROL_ENABLE | CONTROL_IMASK)
            }
            TimerRegister::CompareValue => timers[core].compare_value = value,
        }
        self.drive(&mut timers[core], core, self.count());
        self.changed.notify_all();
        Some(())
    }

    /// Brings `timer`'s interrupt line to its level at `count`.
    fn drive(&self, timer: &mut PhysicalTimer, core: usize, count: u64) {
        let level = timer.line(count);
        if level != timer.output {
            timer.output = level;
            (self.line)(core, level);
        }
    }

    /// Raises timer interrupts as their deadlines pass, until `stop` is set
    /// and [`Timers::wake`] is called. Run on a dedicated thread.
    pub fn run(&self, stop: &AtomicBool) {
        let mut timers = self.timers.lock().unwrap();
        while !stop.load(Ordering::SeqCst) {
            let count = self.count();
            for (core, timer) in timers.iter_mut().enumerate() {
                self.drive(timer, core, count);
            }
            let next_deadline = timers
                .iter()
                .filter(|timer| timer.control == CONTROL_ENABLE && timer.compare_value > count)
                .map(|timer| timer.compare_value)
                .min();
            timers = match next_deadline {
                Some(compare_value) => {
                    let wait = self.clock.until(duration(compare_value));
                    self.changed.wait_timeout(timers, wait).unwrap().0
                }
                None => self.changed.wait(timers).unwrap(),
            };
        }
    }

    /// Makes [`Timers::run`] re-evaluate, including its stop flag.
    pub fn wake(&self) {
        // Taking the lock orders this after `run`'s check of the flag, so the
        // notification cannot fall between that check and the wait.
        let _timers = self.timers.lock().unwrap();
        self.changed.notify_all();
    }
}

impl Snapshot for Timers {
    /// Raises what is due by the paused guest time, which the timer thread
    /// might otherwise do in the middle of the save.
    fn quiesce(&self) -> Result<(), SnapshotError> {
        let count = self.count();
        let mut timers = self.timers.lock().unwrap();
        for (core, timer) in timers.iter_mut().enumerate() {
            self.drive(timer, core, count);
        }
        Ok(())
    }

    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        encode(&*self.timers.lock().unwrap())
    }

    fn restore(&self, state: &[u8]) -> Result<(), SnapshotError> {
        *self.timers.lock().unwrap() = decode("CNTP timers", state)?;
        self.changed.notify_all();
        Ok(())
    }
}

fn ticks(elapsed: Duration) -> u64 {
    (elapsed.as_nanos() * u128::from(COUNTER_HZ) / 1_000_000_000) as u64
}

/// Shortest duration that spans at least `ticks` counter ticks.
fn duration(ticks: u64) -> Duration {
    let nanos = (u128::from(ticks) * 1_000_000_000).div_ceil(u128::from(COUNTER_HZ));
    Duration::from_nanos(u64::try_from(nanos).unwrap_or(u64::MAX))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_requires_enable_unmasked_and_elapsed_compare_value() {
        let timer = |control, compare_value| PhysicalTimer {
            control,
            compare_value,
            output: false,
        };
        assert!(timer(CONTROL_ENABLE, 100).line(100));
        assert!(!timer(CONTROL_ENABLE, 100).line(99));
        assert!(!timer(CONTROL_ENABLE | CONTROL_IMASK, 100).line(200));
        assert!(!timer(0, 0).line(200));
        // Masked timers still report the condition through ISTATUS.
        assert!(timer(CONTROL_ENABLE | CONTROL_IMASK, 100).condition_met(200));
    }

    #[test]
    fn tick_conversions_round_so_a_sleep_never_ends_early() {
        assert_eq!(ticks(Duration::from_secs(10)), 192_000_000);
        for wanted in [1, 2, 7, 19_199_999, 19_200_001] {
            assert!(ticks(duration(wanted)) >= wanted, "{wanted}");
        }
        // The kernel parks idle timers at the maximum compare value.
        assert!(duration(u64::MAX) > Duration::from_secs(1 << 30));
    }
}
