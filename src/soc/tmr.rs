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
//! `CONFIG` (+0), `COMMAND` (+8), `UNLOCK_PATTERN` (+0xC).
//!
//! Power-on values: the NX bootloader's hardware-state check reads all of
//! those as zero in the bits it looks at (`nxbl_hwcheck.csv`, the check
//! `0x40002A7C`, bootloader text `0x4000C928`..`0x4000CBD6`), as a block
//! nothing has started would; here they read zero altogether
//! (\[assumption\] for the bits the check skips), and so they do at kernel
//! entry, since the boot chain starts none of them.
//!
//! `USEC_CFG` (dividend + 1 in bits 15:8, divisor + 1 in 7:0, applied to the
//! oscillator) holds `0x45F` at kernel entry, what the NX bootloader writes
//! (`0x40015250`, research §2.1): 19.2 MHz · 5 / 96, one tick per
//! microsecond. Its reset value is not public: at power-on it refuses to be
//! read until written. Only that setting is accepted; any other would make
//! the counter count something other than microseconds, which it does not.
//!
//! Refused, because they are not modelled: starting a timer (a `PTV` write
//! with bit 31 set), anything written to a watchdog (package1ldr and the
//! bootloader arm `WDT4` only to reset the console after a panic, research
//! §2.12), and writes of `CNTR_FREEZE`. A `PTV` write that leaves the timer
//! stopped is stored; a `PCR` write clears an interrupt that never fires.

use std::time::Duration;

use super::regs::{RegisterFile, word};
use crate::bus::MmioDevice;
use crate::snapshot::{Snapshot, SnapshotError};

pub const BASE: u64 = 0x6000_5000;
pub const SIZE: u64 = 0x400;

const TIMERUS_CNTR_1US: u64 = 0x10;
const TIMERUS_USEC_CFG: u64 = 0x14;
const TIMERUS_CNTR_FREEZE: u64 = 0x3C;
/// One tick per microsecond from the 19.2 MHz oscillator.
const USEC_CFG_19_2_MHZ: u32 = 0x045F;

/// Each timer's `PTV`; its `PCR` follows.
const TIMERS: [u64; 14] = [
    0x000, 0x008, 0x050, 0x058, 0x060, 0x068, 0x070, 0x078, 0x080, 0x088, 0x090, 0x098, 0x0A0,
    0x0A8,
];
const PTV_ENABLE: u32 = 1 << 31;
const PCR: u64 = 0x4;
/// Each watchdog's registers start here.
const WATCHDOGS: [u64; 5] = [0x100, 0x120, 0x140, 0x160, 0x180];
const WDT_REGISTERS: [u64; 3] = [0x0, 0x8, 0xC];

pub struct Timers {
    /// Time since the machine was powered on.
    elapsed: Box<dyn Fn() -> Duration + Send + Sync>,
    registers: RegisterFile,
}

impl Timers {
    /// The block as the boot chain leaves it for the kernel.
    pub fn new(elapsed: Box<dyn Fn() -> Duration + Send + Sync>) -> Self {
        let timers = Self::power_on(elapsed);
        timers.registers.set(TIMERUS_USEC_CFG, USEC_CFG_19_2_MHZ);
        timers
    }

    /// The block as the SoC comes out of reset.
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
        }
    }
}

impl MmioDevice for Timers {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        match word(offset, size)? {
            // The counter is 32 bits wide and wraps.
            TIMERUS_CNTR_1US => Some(u64::from((self.elapsed)().as_micros() as u32)),
            register => self.registers.get(register).map(u64::from),
        }
    }

    fn write(&self, _core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        let register = word(offset, size)?;
        let value = value as u32;
        match register {
            TIMERUS_USEC_CFG if value == USEC_CFG_19_2_MHZ => self.registers.set(register, value),
            ptv if TIMERS.contains(&ptv) && value & PTV_ENABLE == 0 => {
                self.registers.set(ptv, value)
            }
            // The interrupt never fires: nothing to clear, and the count of
            // a stopped timer stays zero.
            pcr if pcr >= PCR && TIMERS.contains(&(pcr - PCR)) => {}
            _ => return None,
        }
        Some(())
    }

    fn snapshot(&self) -> Option<&dyn Snapshot> {
        Some(self)
    }
}

/// Saves the registers; the counter is guest time, which the machine's
/// clock saves.
impl Snapshot for Timers {
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        self.registers.save()
    }

    /// Nothing saved is a snapshot from when the block held no state: the
    /// registers stay as built.
    fn restore(&self, state: &[u8]) -> Result<(), SnapshotError> {
        if state.is_empty() {
            return Ok(());
        }
        self.registers.restore(state)
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
        assert_eq!(timers.write(0, 0x080, 4, 0xC000_0000), None);
        assert_eq!(timers.write(0, 0x080, 4, 0x4000_0010), Some(()));
        assert_eq!(timers.read(0, 0x080, 4), Some(0x4000_0010));
        assert_eq!(timers.write(0, 0x084, 4, 1 << 30), Some(()));
        assert_eq!(timers.write(0, 0x18C, 4, 0xC45A), None);
    }
}
