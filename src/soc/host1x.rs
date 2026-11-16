//! Host1x synchronization block: the syncpoints, counters that hardware
//! events increment and software waits on, and the threshold interrupts
//! that tell software a wait is over.
//!
//! The block is the part of host1x at offset `0x2100` (nvservices text
//! `0x81a8c`). Register offsets here count from there; they are the ones
//! Linux's `hw_host1x04_sync.h` lists for this generation of host1x (public;
//! names recalled from there), and the ones nvservices uses (its HAL,
//! `0xe72f8`-`0xe7400`).
//!
//! `boot` reads one syncpoint directly. Before it takes the display down it
//! spins until syncpoint 9, which it had the display controller count
//! frames on, has advanced by five (text `0x2ecc`). The devices that count
//! on syncpoints are asked to bring them up to date whenever one is read.
//!
//! nvservices waits on syncpoints through thresholds. For a waiter it
//! writes the threshold, then enables the syncpoint's threshold interrupt
//! (text `0x5e950`, with no read of the value after enabling). Its
//! interrupt thread on INTID 97 reads the status words, disables and
//! acknowledges each set bit, then reads the syncpoint and completes the
//! waiters it has passed (`0x82220`). So a status bit has to be set as soon
//! as an enabled syncpoint is at or past its threshold, including at the
//! moment it is enabled, and the interrupt is a level that lasts while any
//! status bit is set (the kernel event is created for a level interrupt,
//! nvservices `0x10f5c`).

use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use super::{
    dc::Dc,
    regs::{RegisterFile, word},
};
use crate::{
    bus::MmioDevice,
    snapshot::{Snapshot, SnapshotError, decode, encode},
};

pub const SYNC_BASE: u64 = 0x5000_2100;
/// Up to the last register nvservices uses, `SYNCPT_PROT_EN` at
/// `0x500038dc` (text `0x82110`).
pub const SYNC_SIZE: u64 = 0x17E0;

/// Level of an interrupt line into the interrupt controller.
pub type InterruptLine = Box<dyn Fn(bool) + Send + Sync>;

/// Syncpoints of the Tegra X1; nvservices configures 192 (text `0x81a40`).
const SYNCPOINT_COUNT: usize = 192;
/// Bitmaps over the syncpoints are this many words.
const BITMAP_WORDS: usize = SYNCPOINT_COUNT / 32;

/// Register offsets from the start of the block.
mod register {
    /// Configuration nvservices writes and never reads back: the interrupt
    /// masks, the microsecond clock, timeouts and register protection
    /// (text `0x81fc0`, `0x81e30`, `0x81ee0`, `0x82110`).
    pub const CONFIGURATION: [u64; 8] = [0x004, 0x008, 0x024, 0x02C, 0x1A4, 0x1A8, 0x1BC, 0x17DC];
    /// General interrupt status and its extension: errors of host1x
    /// itself, write one to clear.
    pub const HINTSTATUS: u64 = 0x020;
    pub const HINTSTATUS_EXT: u64 = 0x028;
    /// Threshold interrupt status, routed to CPU0 and CPU1: one bit per
    /// syncpoint, write one to clear.
    pub const THRESH_CPU0_INT_STATUS: u64 = 0xE80;
    pub const THRESH_CPU1_INT_STATUS: u64 = 0xEA0;
    /// Writing a one disables that syncpoint's threshold interrupt.
    pub const THRESH_INT_DISABLE: u64 = 0xF00;
    /// Writing a one enables it towards CPU0.
    pub const THRESH_INT_ENABLE_CPU0: u64 = 0xF20;
    /// Writing a one increments that syncpoint.
    pub const CPU_INCR: u64 = 0xF60;
    /// Syncpoint values, a word each.
    pub const SYNCPT: u64 = 0xF80;
    /// Thresholds, a word each.
    pub const SYNCPT_INT_THRESH: u64 = 0x1380;
}

/// Which bitmap register, and which of its words, `offset` is.
fn bitmap_word(offset: u64, bitmap: u64) -> Option<usize> {
    let index = usize::try_from(offset.checked_sub(bitmap)? / 4).ok()?;
    (index < BITMAP_WORDS).then_some(index)
}

/// Which syncpoint's register in an array starting at `array` is at
/// `offset`.
fn syncpoint_at(offset: u64, array: u64) -> Option<usize> {
    let index = usize::try_from(offset.checked_sub(array)? / 4).ok()?;
    (index < SYNCPOINT_COUNT).then_some(index)
}

#[derive(Serialize, Deserialize)]
struct State {
    #[serde(with = "crate::snapshot::array")]
    values: [u32; SYNCPOINT_COUNT],
    #[serde(with = "crate::snapshot::array")]
    thresholds: [u32; SYNCPOINT_COUNT],
    /// Threshold interrupts enabled towards CPU0.
    enabled: [u32; BITMAP_WORDS],
    /// CPU0 threshold interrupt status.
    status: [u32; BITMAP_WORDS],
    /// Level last driven onto the interrupt line.
    asserted: bool,
}

impl State {
    /// Whether syncpoint `index` is at or past its threshold. Values wrap,
    /// and so does the comparison, as nvservices' own (text `0x5e2d4`).
    fn reached(&self, index: usize) -> bool {
        self.values[index].wrapping_sub(self.thresholds[index]) as i32 >= 0
    }

    /// Sets the status bit of syncpoint `index` if its interrupt is enabled
    /// and its threshold reached.
    fn compare(&mut self, index: usize) {
        let bit = 1 << (index % 32);
        if self.enabled[index / 32] & bit != 0 && self.reached(index) {
            self.status[index / 32] |= bit;
        }
    }
}

/// The syncpoint counters, shared with the devices that increment them,
/// and the threshold interrupt they raise. Shared by the display
/// controllers, the GPU, the VIC and the host1x channels; [`Host1xSync`]'s
/// [`Snapshot`] saves them, once.
pub struct Syncpoints {
    state: Mutex<State>,
    /// INTID 97, nvservices' syncpoint interrupt (text `0x5a454`); none in
    /// tests that do not look at it.
    line: Option<InterruptLine>,
}

impl Default for Syncpoints {
    fn default() -> Self {
        Self::with_line(None)
    }
}

impl Syncpoints {
    /// Syncpoints whose threshold interrupt drives `line`.
    pub fn new(line: InterruptLine) -> Self {
        Self::with_line(Some(line))
    }

    fn with_line(line: Option<InterruptLine>) -> Self {
        Self {
            state: Mutex::new(State {
                values: [0; SYNCPOINT_COUNT],
                thresholds: [0; SYNCPOINT_COUNT],
                enabled: [0; BITMAP_WORDS],
                status: [0; BITMAP_WORDS],
                asserted: false,
            }),
            line,
        }
    }

    /// Counts `events` on syncpoint `index`, raising its threshold
    /// interrupt if that makes it reach the threshold; one that does not
    /// exist counts nothing.
    pub fn add(&self, index: usize, events: u32) {
        if index >= SYNCPOINT_COUNT || events == 0 {
            return;
        }
        self.change(|state| {
            state.values[index] = state.values[index].wrapping_add(events);
            state.compare(index);
        });
    }

    pub fn get(&self, index: usize) -> Option<u32> {
        self.state.lock().unwrap().values.get(index).copied()
    }

    /// Applies `change` and brings the interrupt line to the resulting
    /// level. The lock is held across the line update so that levels reach
    /// the interrupt controller in the order they were computed.
    fn change<R>(&self, change: impl FnOnce(&mut State) -> R) -> R {
        let mut state = self.state.lock().unwrap();
        let result = change(&mut state);
        let level = state.status.iter().any(|&word| word != 0);
        if level != state.asserted {
            state.asserted = level;
            if let Some(line) = &self.line {
                line(level);
            }
        }
        result
    }

    /// Applies `update` to every syncpoint whose bit is set in `bits`, word
    /// `word` of a bitmap.
    fn each(state: &mut State, word: usize, bits: u32, mut update: impl FnMut(&mut State, usize)) {
        (0..32)
            .filter(|bit| bits >> bit & 1 != 0)
            .for_each(|bit| update(state, word * 32 + bit));
    }
}

pub struct Host1xSync {
    syncpoints: Arc<Syncpoints>,
    /// The display controllers, which count frames and window updates on
    /// syncpoints.
    displays: Vec<Arc<Dc>>,
    configuration: RegisterFile,
}

impl Host1xSync {
    pub fn new(syncpoints: Arc<Syncpoints>, displays: Vec<Arc<Dc>>) -> Self {
        Self {
            syncpoints,
            displays,
            configuration: RegisterFile::strict([]),
        }
    }

    /// Brings the counts of the devices counting by the clock up to date.
    fn settle(&self) {
        self.displays
            .iter()
            .for_each(|display| display.count_frames());
    }
}

impl MmioDevice for Host1xSync {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        use register::*;
        let offset = word(offset, size)?;
        if CONFIGURATION.contains(&offset) {
            return self.configuration.get(offset).map(u64::from);
        }
        let value = match offset {
            // Their sources are host1x's own errors, of channels and
            // register accesses not modeled here, so nothing is ever
            // pending; INTID 99 stays low.
            HINTSTATUS | HINTSTATUS_EXT => 0,
            _ => {
                if let Some(index) = syncpoint_at(offset, SYNCPT) {
                    self.settle();
                    self.syncpoints.get(index)?
                } else {
                    let word = bitmap_word(offset, THRESH_CPU0_INT_STATUS)?;
                    self.settle();
                    self.syncpoints.state.lock().unwrap().status[word]
                }
            }
        };
        Some(value.into())
    }

    fn write(&self, _core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        use register::*;
        let offset = word(offset, size)?;
        let value = value as u32;
        if CONFIGURATION.contains(&offset) {
            self.configuration.set(offset, value);
            return Some(());
        }
        if matches!(offset, HINTSTATUS | HINTSTATUS_EXT) {
            // Write one to clear, of bits that are never set.
            return Some(());
        }
        if bitmap_word(offset, THRESH_CPU1_INT_STATUS).is_some() {
            // Nothing routes threshold interrupts to CPU1 (its enable
            // register is not modeled), so there is no status to clear.
            return Some(());
        }
        if let Some(index) = syncpoint_at(offset, SYNCPT) {
            self.settle();
            self.syncpoints.change(|state| {
                state.values[index] = value;
                state.compare(index);
            });
            return Some(());
        }
        if let Some(index) = syncpoint_at(offset, SYNCPT_INT_THRESH) {
            self.syncpoints.change(|state| {
                state.thresholds[index] = value;
                state.compare(index);
            });
            return Some(());
        }
        if let Some(word) = bitmap_word(offset, THRESH_CPU0_INT_STATUS) {
            self.syncpoints.change(|state| state.status[word] &= !value);
        } else if let Some(word) = bitmap_word(offset, THRESH_INT_DISABLE) {
            self.syncpoints
                .change(|state| state.enabled[word] &= !value);
        } else if let Some(word) = bitmap_word(offset, THRESH_INT_ENABLE_CPU0) {
            self.syncpoints.change(|state| {
                state.enabled[word] |= value;
                Syncpoints::each(state, word, value, State::compare);
            });
        } else {
            let word = bitmap_word(offset, CPU_INCR)?;
            // Increments count on top of whatever the devices have counted.
            self.settle();
            self.syncpoints.change(|state| {
                Syncpoints::each(state, word, value, |state, index| {
                    state.values[index] = state.values[index].wrapping_add(1);
                    state.compare(index);
                });
            });
        }
        Some(())
    }

    fn snapshot(&self) -> Option<&dyn Snapshot> {
        Some(self)
    }
}

/// Saves the syncpoints for every device that counts on them (values,
/// thresholds, enabled threshold interrupts, their status and the level
/// last driven onto the interrupt line), and the configuration registers.
/// The line is not driven again: the interrupt controller saves its own
/// level.
impl Snapshot for Host1xSync {
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        let syncpoints = encode(&*self.syncpoints.state.lock().unwrap())?;
        encode(&(syncpoints, self.configuration.save()?))
    }

    fn restore(&self, state: &[u8]) -> Result<(), SnapshotError> {
        let (syncpoints, configuration): (Vec<u8>, Vec<u8>) = decode("host1x", state)?;
        let syncpoints = decode("host1x syncpoints", &syncpoints)?;
        self.configuration.restore(&configuration)?;
        *self.syncpoints.state.lock().unwrap() = syncpoints;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::atomic::AtomicBool, time::Duration};

    use super::*;

    fn host1x() -> (Arc<Syncpoints>, Host1xSync) {
        let syncpoints: Arc<Syncpoints> = Arc::default();
        let dc = Arc::new(Dc::new(
            Arc::clone(&syncpoints),
            Box::new(Duration::default),
            Box::new(|_| {}),
        ));
        let host1x = Host1xSync::new(Arc::clone(&syncpoints), vec![dc]);
        (syncpoints, host1x)
    }

    fn write(host1x: &Host1xSync, offset: u64, value: u32) {
        host1x.write(0, offset, 4, value.into()).unwrap();
    }

    #[test]
    fn syncpoint_values_follow_the_fixed_registers_and_nothing_else_is_there() {
        let (syncpoints, host1x) = host1x();
        syncpoints.add(9, 5);
        // The register `boot` polls, at 0x500030a4.
        assert_eq!(host1x.read(0, 0xFA4, 4), Some(5));
        assert_eq!(host1x.read(0, 0xF80, 4), Some(0));
        write(&host1x, 0xF80 + 191 * 4, 7);
        assert_eq!(syncpoints.get(191), Some(7));
        assert_eq!(host1x.read(0, 0xF80 + 192 * 4, 4), None);
        assert_eq!(host1x.read(0, 0xFA4, 2), None);
        // Write-only as far as anybody uses them.
        assert_eq!(host1x.read(0, 0x1380, 4), None);
        assert_eq!(host1x.read(0, 0xF20, 4), None);
        // Configuration holds what was written, and only that.
        assert_eq!(host1x.read(0, 0x1A4, 4), None);
        write(&host1x, 0x1A4, 81);
        assert_eq!(host1x.read(0, 0x1A4, 4), Some(81));
    }

    /// Host1x whose threshold interrupt line is watched.
    fn watched() -> (Arc<Syncpoints>, Host1xSync, Arc<AtomicBool>) {
        let level = Arc::new(AtomicBool::new(false));
        let line = Arc::clone(&level);
        let syncpoints = Arc::new(Syncpoints::new(Box::new(move |asserted| {
            line.store(asserted, std::sync::atomic::Ordering::SeqCst)
        })));
        let host1x = Host1xSync::new(Arc::clone(&syncpoints), Vec::new());
        (syncpoints, host1x, level)
    }

    #[test]
    fn threshold_interrupt_rises_when_an_enabled_syncpoint_reaches_it_and_falls_when_acknowledged()
    {
        use std::sync::atomic::Ordering::SeqCst;
        let (syncpoints, host1x, level) = watched();
        // The start-up sequence: everything disabled and acknowledged.
        for word in 0..6 {
            write(&host1x, 0xF00 + word * 4, !0);
            write(&host1x, 0xE80 + word * 4, !0);
            write(&host1x, 0xEA0 + word * 4, !0);
        }
        // Syncpoint 40 is waited on for value 2.
        write(&host1x, 0x1380 + 40 * 4, 2);
        write(&host1x, 0xF24, 1 << 8);
        syncpoints.add(40, 1);
        assert!(!level.load(SeqCst));
        assert_eq!(host1x.read(0, 0xE84, 4), Some(0));
        syncpoints.add(40, 1);
        assert!(level.load(SeqCst));
        assert_eq!(host1x.read(0, 0xE84, 4), Some(1 << 8));

        // The interrupt thread disables, then acknowledges.
        write(&host1x, 0xF04, 1 << 8);
        assert!(level.load(SeqCst), "disabling leaves the status");
        write(&host1x, 0xE84, 1 << 8);
        assert!(!level.load(SeqCst));
        // Disabled, passing the threshold again sets nothing.
        syncpoints.add(40, 1);
        assert_eq!(host1x.read(0, 0xE84, 4), Some(0));

        // Enabling a syncpoint already past its threshold raises at once,
        // through the comparison that wraps: 1 is past 0xFFFF_FFFF.
        write(&host1x, 0xF80 + 3 * 4, 1);
        write(&host1x, 0x1380 + 3 * 4, 0xFFFF_FFFF);
        write(&host1x, 0xF20, 1 << 3);
        assert!(level.load(SeqCst));
        assert_eq!(host1x.read(0, 0xE80, 4), Some(1 << 3));
        write(&host1x, 0xF00, 1 << 3);
        write(&host1x, 0xE80, 1 << 3);
        assert!(!level.load(SeqCst));

        // A threshold ahead of the value: the CPU increment gets it there.
        write(&host1x, 0x1380 + 3 * 4, 3);
        write(&host1x, 0xF20, 1 << 3);
        assert!(!level.load(SeqCst));
        write(&host1x, 0xF60, 1 << 3);
        assert!(!level.load(SeqCst));
        write(&host1x, 0xF60, 1 << 3);
        assert_eq!(syncpoints.get(3), Some(3));
        assert!(level.load(SeqCst));
    }
}
