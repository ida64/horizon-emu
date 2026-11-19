//! The timer block (`TMR`, `0x6000_5000`): the microsecond counter, and the
//! timers' and watchdogs' registers as they read while nothing uses them.
//!
//! `TIMERUS_CNTR_1US` counts microseconds from power-on, and the firmware's
//! clock driver spins on it for its short delays (pcv maps the block and
//! reads offset `0x10` in a loop); package1ldr and the NX bootloader time
//! every delay and timeout on it.
//!
//! Offsets (names as hekate `t210.h` and Linux `timer-tegra.c` have them,
//! reference only): `TMR1`/`TMR2` at `0x000`/`0x008`, `TMR3`..`TMR10` at
//! `0x050` + 8·n, four more timers at `0x090`..`0x0A8` that the NX
//! bootloader's check also reads, each a `PTV` (enable bit 31, periodic
//! bit 30, count 28:0) and a `PCR` (present count; write bit 30 to clear
//! the interrupt); the counter's `USEC_CFG` (`0x014`) and
//! `CNTR_FREEZE` (`0x03C`); watchdogs `WDT0`..`WDT4` at `0x100` + 0x20·n,
//! `CONFIG` (+0), `STATUS` (+4; bit 0 says the watchdog runs, and the boot
//! ROM asks it of `WDT4`, `0x104758`), `COMMAND` (+8), `UNLOCK_PATTERN`
//! (+0xC).
//!
//! Power-on values: the NX bootloader's hardware-state check reads all of
//! those as zero in the bits it looks at (`nxbl_hwcheck.csv`, the check
//! `0x40002A7C`, bootloader text `0x4000C928`..`0x4000CBD6`), as a block
//! nothing has started would; here they read zero altogether
//! (\[assumption\] for the bits the check skips), and so they do at kernel
//! entry, since the boot chain starts none of them.
//!
//! `USEC_CFG` (dividend + 1 in bits 15:8, divisor + 1 in 7:0, applied to the
//! timers' clock, CLK_M) holds `0x45F` at kernel entry, what the NX
//! bootloader writes (`0x40015250`, research §2.1): 19.2 MHz · 5 / 96, one
//! tick per microsecond, CLK_M being half the 38.4 MHz oscillator by then.
//! The boot ROM writes `0x4BF` (`0x10023A`, by the oscillator it measured):
//! 38.4 MHz · 5 / 192, CLK_M still being the oscillator itself. Its reset
//! value is not public: at power-on it refuses to be read until written.
//! Only those two settings are accepted, each taken to go with the CLK_M of
//! its writer; any other would make the counter count something other than
//! microseconds, which it does not.
//!
//! # Watchdogs
//!
//! In a machine that runs the boot processor ([`Timers::power_on`]), a
//! watchdog can be armed the way Linux's `tegra_wdt.c` and hekate's
//! `timer.c` do it (reference for facts):
//!
//! * its source timer started periodic (`PTV` with bits 31 and 30; the
//!   count is the period in microseconds, less one as recalled from the
//!   Tegra manuals' "n+1 scheme");
//! * `CONFIG`: the source timer's number in bits 3:0 (`TMR9` is 9), the
//!   timer expirations per watchdog period in bits 11:4, and what its
//!   expirations do: an interrupt at the first (bit 12), a fast interrupt
//!   at the second (bit 13), a system reset at the third (bit 14), a reset
//!   through the PMC at the fourth (bit 15) (Linux: "the watchdog system
//!   reset actually occurs on the 4th expiration of this counter"; hekate
//!   arms 2.5 s with the fast interrupt and calls it 5 seconds);
//! * `COMMAND` bit 0 starts the count, or starts it over; bit 1 stops the
//!   watchdog if `UNLOCK_PATTERN` holds `0xC45A`, which that takes away
//!   again \[assumption\]. `STATUS` bit 0 reads 1 while it is armed.
//!
//! Nothing here delivers what an expiration does: no interrupt goes to the
//! interrupt controller and the SoC is not reset. Instead
//! [`Timers::watchdog_expired`] says when the first enabled one is due,
//! \[assumption\] counted from the start command, and the boot processor's
//! machine stops there with that said: a watchdog that fires means its
//! software hung or panicked (package1ldr, the bootloader and hekate arm
//! `WDT4` to reset the console after a panic). A watchdog whose source
//! timer is stopped does not count.
//!
//! Refused, because they are not modelled: starting a timer any other way
//! (one shot, for its own interrupt), anything written to a watchdog in a
//! machine with no boot processor to stop, a watchdog configured while it
//! is armed or with no source or period, and writes of `CNTR_FREEZE`. A
//! `PTV` write that leaves the timer stopped is stored; a `PCR` write
//! clears an interrupt that never fires.

use std::{
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use super::regs::{RegisterFile, word};
use crate::bus::MmioDevice;
use crate::snapshot::{Snapshot, SnapshotError, decode, encode};

pub const BASE: u64 = 0x6000_5000;
pub const SIZE: u64 = 0x400;

const TIMERUS_CNTR_1US: u64 = 0x10;
const TIMERUS_USEC_CFG: u64 = 0x14;
const TIMERUS_CNTR_FREEZE: u64 = 0x3C;
/// One tick per microsecond from a CLK_M of 19.2 MHz, and of 38.4 MHz.
const USEC_CFG_19_2_MHZ: u32 = 0x045F;
const USEC_CFG_38_4_MHZ: u32 = 0x04BF;

/// Each timer's `PTV`, `TMR1` first; its `PCR` follows.
const TIMERS: [u64; 14] = [
    0x000, 0x008, 0x050, 0x058, 0x060, 0x068, 0x070, 0x078, 0x080, 0x088, 0x090, 0x098, 0x0A0,
    0x0A8,
];
const PTV_ENABLE: u32 = 1 << 31;
const PTV_PERIODIC: u32 = 1 << 30;
const PTV_COUNT: u32 = 0x1FFF_FFFF;
const PCR: u64 = 0x4;
/// Each watchdog's registers start here.
const WATCHDOGS: [u64; 5] = [0x100, 0x120, 0x140, 0x160, 0x180];
const WDT_CONFIG: u64 = 0x0;
const WDT_STATUS: u64 = 0x4;
const WDT_COMMAND: u64 = 0x8;
const WDT_UNLOCK_PATTERN: u64 = 0xC;
const WDT_REGISTERS: [u64; 4] = [WDT_CONFIG, WDT_STATUS, WDT_COMMAND, WDT_UNLOCK_PATTERN];
const CONFIG_SOURCE: u32 = 0xF;
const CONFIG_PERIOD_SHIFT: u32 = 4;
const CONFIG_PERIOD: u32 = 0xFF;
/// What the first, second, third and fourth expiration does, by its
/// enable bit.
const CONFIG_EVENTS: [(u32, &str); 4] = [
    (1 << 12, "its interrupt"),
    (1 << 13, "its fast interrupt"),
    (1 << 14, "a system reset"),
    (1 << 15, "a reset of the SoC through the PMC"),
];
const CONFIG_FIELDS: u32 = CONFIG_SOURCE | CONFIG_PERIOD << CONFIG_PERIOD_SHIFT | 0xF << 12;
const STATUS_ENABLED: u32 = 1 << 0;
const COMMAND_START: u32 = 1 << 0;
const COMMAND_DISABLE: u32 = 1 << 1;
const UNLOCK_PATTERN: u32 = 0xC45A;

pub struct Timers {
    /// Time since the machine was powered on.
    elapsed: Box<dyn Fn() -> Duration + Send + Sync>,
    registers: RegisterFile,
    /// Whether a boot processor's machine asks [`Timers::watchdog_expired`]:
    /// only then can a watchdog be armed.
    watched: bool,
    /// When each armed watchdog was started.
    armed: Mutex<[Option<Duration>; WATCHDOGS.len()]>,
    /// The first moment an armed watchdog's expiration does something, in
    /// nanoseconds of the machine's time; `u64::MAX` for never.
    due: AtomicU64,
}

impl Timers {
    /// The block as the boot chain leaves it for the kernel.
    pub fn new(elapsed: Box<dyn Fn() -> Duration + Send + Sync>) -> Self {
        let mut timers = Self::power_on(elapsed);
        timers.watched = false;
        timers.registers.set(TIMERUS_USEC_CFG, USEC_CFG_19_2_MHZ);
        timers
    }

    /// The block as the SoC comes out of reset, in a machine whose boot
    /// processor stops at an expired watchdog.
    pub fn power_on(elapsed: Box<dyn Fn() -> Duration + Send + Sync>) -> Self {
        let timers = TIMERS.into_iter().flat_map(|ptv| [ptv, ptv + PCR]);
        let watchdogs = WATCHDOGS
            .into_iter()
            .flat_map(|base| WDT_REGISTERS.map(|register| base + register));
        Self {
            elapsed,
            registers: RegisterFile::strict(
                timers
                    .chain(watchdogs)
                    .chain([TIMERUS_CNTR_FREEZE])
                    .map(|register| (register, 0)),
            ),
            watched: true,
            armed: Mutex::default(),
            due: AtomicU64::new(u64::MAX),
        }
    }

    /// What an armed watchdog has come to, if the first expiration that
    /// does anything is due: which watchdog, and what it would do.
    pub fn watchdog_expired(&self) -> Option<String> {
        let due = self.due.load(Ordering::Acquire);
        if due == u64::MAX || (self.elapsed)().as_nanos() < u128::from(due) {
            return None;
        }
        let armed = self.armed.lock().unwrap();
        let (index, _, what) = (0..WATCHDOGS.len())
            .filter_map(|index| {
                let (after, what) = self.first_event(index)?;
                Some((index, armed[index]? + after, what))
            })
            .min_by_key(|&(_, at, _)| at)?;
        Some(format!(
            "watchdog WDT{index} expired: {what} is due, which is not modelled (the software \
             that armed it hung, or panicked and asked for the reset)"
        ))
    }

    /// How long after its start watchdog `index` first does something, and
    /// what; `None` if its source timer is stopped or no expiration is
    /// enabled.
    fn first_event(&self, index: usize) -> Option<(Duration, &'static str)> {
        let config = self.registers.get(WATCHDOGS[index] + WDT_CONFIG)?;
        let source = (config & CONFIG_SOURCE).checked_sub(1)? as usize;
        let ptv = self.registers.get(*TIMERS.get(source)?)?;
        if ptv & (PTV_ENABLE | PTV_PERIODIC) != PTV_ENABLE | PTV_PERIODIC {
            return None;
        }
        let period = u64::from(ptv & PTV_COUNT) + 1;
        let expirations = u64::from(config >> CONFIG_PERIOD_SHIFT & CONFIG_PERIOD);
        let (nth, what) = (1u64..)
            .zip(CONFIG_EVENTS)
            .find_map(|(nth, (enable, what))| (config & enable != 0).then_some((nth, what)))?;
        Some((Duration::from_micros(nth * expirations * period), what))
    }

    /// Works [`Self::due`] out again, after a write that may have moved it.
    fn reschedule(&self) {
        let armed = self.armed.lock().unwrap();
        let due = (0..WATCHDOGS.len())
            .filter_map(|index| Some(armed[index]? + self.first_event(index)?.0))
            .min()
            .map_or(u64::MAX, |at| {
                u64::try_from(at.as_nanos()).unwrap_or(u64::MAX - 1)
            });
        self.due.store(due, Ordering::Release);
    }

    /// A write to register `register` of watchdog `index`.
    fn write_watchdog(&self, index: usize, register: u64, value: u32) -> Option<()> {
        if !self.watched {
            return None;
        }
        let base = WATCHDOGS[index];
        let mut armed = self.armed.lock().unwrap();
        match register {
            WDT_CONFIG => {
                let source = value & CONFIG_SOURCE;
                let expirations = value >> CONFIG_PERIOD_SHIFT & CONFIG_PERIOD;
                let valid = value & !CONFIG_FIELDS == 0
                    && (1..=TIMERS.len() as u32).contains(&source)
                    && expirations != 0;
                if armed[index].is_some() || !valid {
                    return None;
                }
                self.registers.set(base + WDT_CONFIG, value);
            }
            WDT_COMMAND if value == COMMAND_START => armed[index] = Some((self.elapsed)()),
            WDT_COMMAND if value == COMMAND_DISABLE => {
                if self.registers.get(base + WDT_UNLOCK_PATTERN)? != UNLOCK_PATTERN {
                    return None;
                }
                self.registers.set(base + WDT_UNLOCK_PATTERN, 0);
                armed[index] = None;
            }
            WDT_UNLOCK_PATTERN => self.registers.set(base + WDT_UNLOCK_PATTERN, value),
            _ => return None,
        }
        Some(())
    }
}

impl MmioDevice for Timers {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        let register = word(offset, size)?;
        let status = WATCHDOGS
            .iter()
            .position(|base| base + WDT_STATUS == register);
        match (register, status) {
            // The counter is 32 bits wide and wraps.
            (TIMERUS_CNTR_1US, _) => Some(u64::from((self.elapsed)().as_micros() as u32)),
            (_, Some(index)) => {
                let armed = self.armed.lock().unwrap()[index].is_some();
                Some(u64::from(if armed { STATUS_ENABLED } else { 0 }))
            }
            _ => self.registers.get(register).map(u64::from),
        }
    }

    fn write(&self, _core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        let register = word(offset, size)?;
        let value = value as u32;
        let watchdog = WATCHDOGS
            .iter()
            .position(|base| (*base..base + 0x10).contains(&register));
        match (register, watchdog) {
            (_, Some(index)) => {
                self.write_watchdog(index, register - WATCHDOGS[index], value)?;
            }
            (TIMERUS_USEC_CFG, _) if matches!(value, USEC_CFG_19_2_MHZ | USEC_CFG_38_4_MHZ) => {
                self.registers.set(register, value)
            }
            // Stopped, or running periodic as a watchdog's source.
            (ptv, _)
                if TIMERS.contains(&ptv)
                    && (value & PTV_ENABLE == 0 || self.watched && value & PTV_PERIODIC != 0) =>
            {
                self.registers.set(ptv, value)
            }
            // The interrupt never fires: nothing to clear, and the count of
            // a stopped timer stays zero.
            (pcr, _) if pcr >= PCR && TIMERS.contains(&(pcr - PCR)) => {}
            _ => return None,
        }
        self.reschedule();
        Some(())
    }

    fn snapshot(&self) -> Option<&dyn Snapshot> {
        Some(self)
    }
}

/// Saves the registers and when each armed watchdog was started; the
/// counter is guest time, which the machine's clock saves.
impl Snapshot for Timers {
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        encode(&(self.registers.save()?, *self.armed.lock().unwrap()))
    }

    /// Nothing saved is a snapshot from when the block held no state: the
    /// registers stay as built.
    fn restore(&self, state: &[u8]) -> Result<(), SnapshotError> {
        if state.is_empty() {
            return Ok(());
        }
        let (registers, armed): (Vec<u8>, [Option<Duration>; WATCHDOGS.len()]) =
            decode("timers", state)?;
        self.registers.restore(&registers)?;
        *self.armed.lock().unwrap() = armed;
        self.reschedule();
        Ok(())
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
        assert_eq!(timers.read(0, 0x0B0, 4), None);
    }

    #[test]
    fn idle_timers_read_zero_and_cannot_start() {
        let timers = Timers::power_on(Box::new(|| Duration::ZERO));
        assert_eq!(timers.read(0, 0x000, 4), Some(0));
        assert_eq!(timers.read(0, 0x0AC, 4), Some(0));
        assert_eq!(timers.read(0, 0x18C, 4), Some(0));
        assert_eq!(timers.read(0, TIMERUS_USEC_CFG, 4), None);
        assert_eq!(timers.write(0, TIMERUS_USEC_CFG, 4, 0x000C), None);
        assert_eq!(timers.write(0, TIMERUS_USEC_CFG, 4, 0x045F), Some(()));
        assert_eq!(timers.read(0, TIMERUS_USEC_CFG, 4), Some(0x045F));
        assert_eq!(timers.write(0, 0x080, 4, 0x8000_0010), None, "one shot");
        assert_eq!(timers.write(0, 0x080, 4, 0x4000_0010), Some(()));
        assert_eq!(timers.read(0, 0x080, 4), Some(0x4000_0010));
        assert_eq!(timers.write(0, 0x084, 4, 1 << 30), Some(()));
        // No boot processor watches the kernel's machine.
        let unwatched = Timers::new(Box::new(|| Duration::ZERO));
        assert_eq!(unwatched.write(0, 0x18C, 4, 0xC45A), None);
        assert_eq!(unwatched.write(0, 0x080, 4, 0xC000_0010), None);
    }

    #[test]
    fn an_armed_watchdog_is_due_at_its_first_enabled_expiration() {
        use std::sync::Arc;
        let now = Arc::new(Mutex::new(Duration::from_secs(10)));
        let clock = Arc::clone(&now);
        let timers = Timers::power_on(Box::new(move || *clock.lock().unwrap()));
        let (ptv9, wdt4) = (0x080, 0x180);
        // hekate's: TMR9 every 2.5 s, the fast interrupt, so 5 s.
        timers
            .write(4, wdt4 + WDT_UNLOCK_PATTERN, 4, 0xC45A)
            .unwrap();
        timers.write(4, ptv9, 4, 0xC000_0000 | 2_499_999).unwrap();
        timers
            .write(4, wdt4 + WDT_CONFIG, 4, 9 | 1 << 4 | 1 << 13)
            .unwrap();
        assert_eq!(timers.read(4, wdt4 + WDT_STATUS, 4), Some(0));
        timers.write(4, wdt4 + WDT_COMMAND, 4, 1).unwrap();
        assert_eq!(timers.read(4, wdt4 + WDT_STATUS, 4), Some(1));
        *now.lock().unwrap() = Duration::from_millis(14_999);
        assert_eq!(timers.watchdog_expired(), None);
        *now.lock().unwrap() = Duration::from_secs(15);
        assert!(timers.watchdog_expired().unwrap().contains("WDT4"));
        // Started over, it has its time again; disabled, it never fires.
        timers.write(4, wdt4 + WDT_COMMAND, 4, 1).unwrap();
        assert_eq!(timers.watchdog_expired(), None);
        timers.write(4, wdt4 + WDT_COMMAND, 4, 2).unwrap();
        assert_eq!(timers.read(4, wdt4 + WDT_STATUS, 4), Some(0));
        *now.lock().unwrap() = Duration::from_secs(100);
        assert_eq!(timers.watchdog_expired(), None);
        // The pattern was used up.
        timers.write(4, wdt4 + WDT_COMMAND, 4, 1).unwrap();
        assert_eq!(timers.write(4, wdt4 + WDT_COMMAND, 4, 2), None);
        // A stopped source timer stops the count.
        timers.write(4, ptv9, 4, 0).unwrap();
        assert_eq!(timers.watchdog_expired(), None);
    }
}
