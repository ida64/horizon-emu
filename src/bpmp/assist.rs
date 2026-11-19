//! What runs alongside a core without being part of its state: a cache of
//! fetched instructions, and a trace of the last steps and bus accesses.
//!
//! Neither changes what the core does. Both live in an [`Assist`] the
//! machine owns and hands to [`Arm7::run_assisted`] or
//! [`Arm7::step_assisted`]; snapshots save the core alone, and a restored
//! core runs on with the same `Assist` (the cache sees the restore as a
//! write to its memory, through the generation counter).

use std::{
    collections::VecDeque,
    fmt,
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use super::{Arm7, BpmpBus, BusError, Exception, Fault, Step};

/// The cache and the trace a core runs with; both off by default.
#[derive(Default)]
pub struct Assist {
    cache: Option<CodeCache>,
    trace: Option<Trace>,
}

impl Assist {
    pub fn new() -> Self {
        Self::default()
    }

    /// Caches the instructions fetched from `region`, which must be plain
    /// memory: a fetch there that succeeded once succeeds again, and
    /// returns what was last written. The core's own stores there
    /// invalidate what they overwrite; every other writer of the region
    /// must increment `generation` once its write is visible (as
    /// [`crate::soc::iram::Iram`] does for writes other than the BPMP's),
    /// which makes the core drop the whole cache before its next fetch.
    /// `region` is aligned to words; a cache of 8 bytes per halfword of it
    /// is allocated.
    pub fn set_code_cache(&mut self, region: Range<u32>, generation: Arc<AtomicU64>) {
        self.cache = Some(CodeCache::new(region, generation));
    }

    /// Caches the instructions fetched from `region` as well, another
    /// plain memory with its own `generation` (see
    /// [`Assist::set_code_cache`]); the regions must not overlap.
    pub fn add_code_cache(&mut self, region: Range<u32>, generation: Arc<AtomicU64>) {
        match &mut self.cache {
            Some(cache) => cache.regions.push(Region::new(region, generation)),
            None => self.set_code_cache(region, generation),
        }
    }

    /// Stops caching instructions.
    pub fn clear_code_cache(&mut self) {
        self.cache = None;
    }

    /// Keeps the last `steps` steps and the last `accesses` data accesses
    /// whose address `keep` accepts (fetches are not accesses here: the
    /// steps show them); replaces any trace kept so far.
    pub fn set_trace(&mut self, steps: usize, accesses: usize, keep: fn(u32) -> bool) {
        self.trace = (steps > 0 || accesses > 0).then(|| Trace {
            steps: VecDeque::with_capacity(steps),
            step_capacity: steps,
            accesses: VecDeque::with_capacity(accesses),
            access_capacity: accesses,
            keep,
        });
    }

    /// The trace kept so far, if tracing is on.
    pub fn trace(&self) -> Option<&Trace> {
        self.trace.as_ref()
    }
}

impl Arm7 {
    /// [`Arm7::step`] with `assist`'s cache and trace.
    pub fn step_assisted(
        &mut self,
        bus: &mut dyn BpmpBus,
        assist: &mut Assist,
    ) -> Result<Step, Fault> {
        let Assist { cache, trace } = assist;
        let Some(trace) = trace else {
            return self.step_inner(bus, cache.as_mut(), None);
        };
        let (pc, thumb) = (self.pc(), self.thumb());
        let accesses = (trace.access_capacity > 0).then_some(&mut *trace);
        let result = self.step_inner(bus, cache.as_mut(), accesses);
        trace.push_step(TracedStep {
            pc,
            thumb,
            outcome: match &result {
                Ok(Step::Executed) => Outcome::Executed,
                Ok(Step::Exception(exception)) => Outcome::Exception(*exception),
                Err(_) => Outcome::Fault,
            },
        });
        result
    }

    /// [`Arm7::run`] with `assist`'s cache and trace.
    pub fn run_assisted(
        &mut self,
        bus: &mut dyn BpmpBus,
        assist: &mut Assist,
        budget: u64,
    ) -> Result<u64, Fault> {
        if assist.trace.is_some() {
            let mut done = 0;
            while done < budget {
                self.step_assisted(bus, assist)?;
                done += 1;
                if bus.should_pause() {
                    break;
                }
            }
            return Ok(done);
        }
        self.run_inner(bus, assist.cache.as_mut(), budget)
    }
}

/// Instructions by address, valid while nothing has written them.
///
/// One entry per halfword of each region holds the instruction fetched
/// there and a tag: the state it was fetched in and the epoch it was
/// fetched in. A write from elsewhere (the region's generation moved)
/// starts a new epoch, which invalidates every entry of the region at
/// once; the core's own stores clear the entries they overlap.
pub(super) struct CodeCache {
    regions: Vec<Region>,
}

struct Region {
    base: u32,
    entries: Vec<Entry>,
    generation: Arc<AtomicU64>,
    seen: u64,
    /// The current epoch, from 1; entries of other epochs are stale.
    epoch: u32,
}

#[derive(Clone, Copy, Default)]
struct Entry {
    instruction: u32,
    /// `epoch << 1 | thumb`; 0 is never valid.
    tag: u32,
}

impl Region {
    fn new(region: Range<u32>, generation: Arc<AtomicU64>) -> Self {
        let base = region.start & !3;
        let end = region.end.max(base).wrapping_add(3) & !3;
        let seen = generation.load(Ordering::Acquire);
        Self {
            base,
            entries: vec![Entry::default(); (end.wrapping_sub(base) / 2) as usize],
            generation,
            seen,
            epoch: 1,
        }
    }

    fn new_epoch(&mut self) {
        self.epoch += 1;
        if self.epoch >= 1 << 31 {
            self.entries.fill(Entry::default());
            self.epoch = 1;
        }
    }

    /// Drops the entries a write of `len` bytes at `address` overlaps: the
    /// word-aligned ARM entry and the halfword Thumb entries.
    fn invalidate(&mut self, address: u32, len: u32) {
        let start = u64::from(address & !3);
        let end = u64::from(address) + u64::from(len);
        let base = u64::from(self.base);
        let (low, high) = (
            start.max(base),
            end.min(base + 2 * self.entries.len() as u64),
        );
        if low < high {
            let (first, last) = ((low - base) / 2, (high - 1 - base) / 2);
            for entry in &mut self.entries[first as usize..=last as usize] {
                entry.tag = 0;
            }
        }
        if end > 1 << 32 {
            // The write wraps past the top of the address space.
            self.invalidate(0, (end - (1 << 32)) as u32);
        }
    }
}

impl CodeCache {
    fn new(region: Range<u32>, generation: Arc<AtomicU64>) -> Self {
        Self {
            regions: vec![Region::new(region, generation)],
        }
    }

    /// The instruction at `pc` (an aligned address of `width` bytes),
    /// from the cache or the bus.
    #[inline]
    pub(super) fn fetch(
        &mut self,
        bus: &mut dyn BpmpBus,
        pc: u32,
        width: u8,
    ) -> Result<u32, BusError> {
        for region in &mut self.regions {
            let slot = (pc.wrapping_sub(region.base) / 2) as usize;
            if slot >= region.entries.len() {
                continue;
            }
            let generation = region.generation.load(Ordering::Acquire);
            if generation != region.seen {
                region.seen = generation;
                region.new_epoch();
            }
            let tag = region.epoch << 1 | u32::from(width == 2);
            let entry = &mut region.entries[slot];
            if entry.tag == tag {
                return Ok(entry.instruction);
            }
            let instruction = bus.read(pc, width)?;
            *entry = Entry { instruction, tag };
            return Ok(instruction);
        }
        bus.read(pc, width)
    }

    /// Drops the entries a write of `len` bytes at `address` overlaps.
    fn invalidate(&mut self, address: u32, len: u32) {
        for region in &mut self.regions {
            region.invalidate(address, len);
        }
    }
}

/// The bus the core executes an instruction through while it caches
/// code: the core's stores clear the cache entries they overwrite.
pub(super) struct Invalidating<'a> {
    pub(super) inner: &'a mut dyn BpmpBus,
    pub(super) cache: &'a mut CodeCache,
}

impl BpmpBus for Invalidating<'_> {
    fn read(&mut self, address: u32, size: u8) -> Result<u32, BusError> {
        self.inner.read(address, size)
    }

    fn write(&mut self, address: u32, size: u8, value: u32) -> Result<(), BusError> {
        self.cache.invalidate(address, u32::from(size));
        self.inner.write(address, size, value)
    }

    fn read_words(&mut self, address: u32, words: &mut [u32]) -> Result<(), (usize, BusError)> {
        self.inner.read_words(address, words)
    }

    fn write_words(&mut self, address: u32, words: &[u32]) -> Result<(), (usize, BusError)> {
        self.cache.invalidate(address, 4 * words.len() as u32);
        self.inner.write_words(address, words)
    }

    fn exception_vector(&mut self, exception: Exception) -> u32 {
        self.inner.exception_vector(exception)
    }

    fn should_pause(&mut self) -> bool {
        self.inner.should_pause()
    }
}

/// The last steps and data accesses of a core.
pub struct Trace {
    steps: VecDeque<TracedStep>,
    step_capacity: usize,
    accesses: VecDeque<TracedAccess>,
    access_capacity: usize,
    keep: fn(u32) -> bool,
}

/// One step: where the core was and what came of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TracedStep {
    /// The instruction's address, or where the interrupt was taken.
    pub pc: u32,
    pub thumb: bool,
    pub outcome: Outcome,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Executed,
    Exception(Exception),
    /// The core stopped with a [`Fault`].
    Fault,
}

/// One data access and how the bus answered it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TracedAccess {
    /// The instruction that made it.
    pub pc: u32,
    pub address: u32,
    pub size: u8,
    pub write: bool,
    /// The value written, or read if the read succeeded.
    pub value: Option<u32>,
    pub result: Result<(), BusError>,
}

impl Trace {
    /// The steps kept, oldest first.
    pub fn steps(&self) -> impl Iterator<Item = &TracedStep> {
        self.steps.iter()
    }

    /// The accesses kept, oldest first.
    pub fn accesses(&self) -> impl Iterator<Item = &TracedAccess> {
        self.accesses.iter()
    }

    /// Both, one line each: steps, then accesses.
    pub fn lines(&self) -> Vec<String> {
        let steps = self.steps().map(ToString::to_string);
        steps
            .chain(self.accesses().map(ToString::to_string))
            .collect()
    }

    pub fn clear(&mut self) {
        self.steps.clear();
        self.accesses.clear();
    }

    fn push_step(&mut self, step: TracedStep) {
        if self.step_capacity == 0 {
            return;
        }
        if self.steps.len() == self.step_capacity {
            self.steps.pop_front();
        }
        self.steps.push_back(step);
    }

    fn push_access(&mut self, access: TracedAccess) {
        if !(self.keep)(access.address) {
            return;
        }
        if self.accesses.len() == self.access_capacity {
            self.accesses.pop_front();
        }
        self.accesses.push_back(access);
    }
}

impl fmt::Display for TracedStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = if self.thumb { "Thumb" } else { "ARM" };
        write!(f, "at {:#010x} ({state})", self.pc)?;
        match self.outcome {
            Outcome::Executed => Ok(()),
            Outcome::Exception(exception) => write!(f, ": {exception:?} exception"),
            Outcome::Fault => write!(f, ": stopped"),
        }
    }
}

impl fmt::Display for TracedAccess {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = if self.write { "W" } else { "R" };
        write!(
            f,
            "{kind} {:#010x}/{} from {:#010x}: ",
            self.address, self.size, self.pc
        )?;
        match (&self.result, self.value) {
            (Ok(()), Some(value)) => write!(f, "{value:#x}"),
            (Ok(()), None) => write!(f, "done"),
            (Err(BusError::Abort), _) => write!(f, "aborted"),
            (Err(BusError::Unmodelled(reason)), _) => write!(f, "refused: {reason}"),
        }
    }
}

/// The bus an instruction runs through while its accesses are traced.
pub(super) struct TracingBus<'a> {
    pub(super) inner: &'a mut dyn BpmpBus,
    pub(super) trace: &'a mut Trace,
    pub(super) pc: u32,
}

impl TracingBus<'_> {
    fn record(
        &mut self,
        address: u32,
        size: u8,
        write: bool,
        value: Option<u32>,
        result: Result<(), BusError>,
    ) {
        self.trace.push_access(TracedAccess {
            pc: self.pc,
            address,
            size,
            write,
            value,
            result,
        });
    }

    /// Records a burst: the words up to the one that failed, and it.
    fn record_words(
        &mut self,
        address: u32,
        words: &[u32],
        write: bool,
        result: &Result<(), (usize, BusError)>,
    ) {
        let done = match result {
            Ok(()) => words.len(),
            Err((index, _)) => *index,
        };
        for (i, &word) in words[..done].iter().enumerate() {
            self.record(
                address.wrapping_add(4 * i as u32),
                4,
                write,
                Some(word),
                Ok(()),
            );
        }
        if let Err((index, error)) = result {
            let value = write.then(|| words[*index]);
            let at = address.wrapping_add(4 * *index as u32);
            self.record(at, 4, write, value, Err(error.clone()));
        }
    }
}

impl BpmpBus for TracingBus<'_> {
    fn read(&mut self, address: u32, size: u8) -> Result<u32, BusError> {
        let result = self.inner.read(address, size);
        let value = result.as_ref().ok().copied();
        self.record(
            address,
            size,
            false,
            value,
            result.as_ref().map(|_| ()).map_err(Clone::clone),
        );
        result
    }

    fn write(&mut self, address: u32, size: u8, value: u32) -> Result<(), BusError> {
        let result = self.inner.write(address, size, value);
        self.record(address, size, true, Some(value), result.clone());
        result
    }

    fn read_words(&mut self, address: u32, words: &mut [u32]) -> Result<(), (usize, BusError)> {
        let result = self.inner.read_words(address, words);
        self.record_words(address, words, false, &result);
        result
    }

    fn write_words(&mut self, address: u32, words: &[u32]) -> Result<(), (usize, BusError)> {
        let result = self.inner.write_words(address, words);
        self.record_words(address, words, true, &result);
        result
    }

    fn exception_vector(&mut self, exception: Exception) -> u32 {
        self.inner.exception_vector(exception)
    }

    fn should_pause(&mut self) -> bool {
        self.inner.should_pause()
    }
}
