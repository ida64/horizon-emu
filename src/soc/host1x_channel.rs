//! Tegra X1 host1x CDMA execution; offsets passed to engines are words.
//!
//! Encoding evidence: Linux `drivers/gpu/host1x/hw/` [opcodes.h],
//! [hw_host1x04_channel.h], [hw_host1x04_uclass.h], and [channel_hw.c].
//! The latter emits the 32-bit wait as payload at 0x4e followed by ID at 0x50.
//! nvservices initializes GET from PUT with DMACTRL=7, then runs with zero.
//!
//! A stream runs on the DMAPUT write until it reaches PUT or a host
//! syncpoint wait whose threshold has not been reached. Such a wait
//! stalls the channel where it stands (ring position and gather nesting are
//! kept), as the hardware's command processor does; it continues once the
//! syncpoint gets there: a thread of its own looks again at every change of
//! the syncpoints ([`Syncpoints::wait_change`]), so at the guest time the
//! syncpoint reached the threshold. The wait is channel state: a channel
//! restored from a snapshot standing at one starts its thread again.
//!
//! The channel's registers are never held up by an engine's work, as on
//! the chip, where the CPU's accesses to a channel do not wait for the
//! engines it feeds: the thread running the stream lets go of the channel
//! while an engine method runs (a VIC EXECUTE composes a whole picture),
//! and a DMAPUT written meanwhile is taken up by that thread when the
//! method is done, the write returning at once. One thread runs a stream
//! at a time.
//!
//! Unimplemented methods/conditions fail the stream and never manufacture a
//! fence completion; the DMAPUT write that ran into them is refused, or, for
//! a stream that failed while continuing after a wait, the next one. Shared
//! wait-base registers are not modeled; a host increment of any of the four
//! conditions counts when the command processor reaches it (see
//! `Inner::method`), a reserved condition refuses. A
//! failed stream stays faulted until DMA GET reset, so retrying a failed
//! submission cannot replay its effects.
//!
//! [opcodes.h]: https://github.com/torvalds/linux/blob/master/drivers/gpu/host1x/hw/opcodes.h
//! [hw_host1x04_channel.h]: https://github.com/torvalds/linux/blob/master/drivers/gpu/host1x/hw/hw_host1x04_channel.h
//! [hw_host1x04_uclass.h]: https://github.com/torvalds/linux/blob/master/drivers/gpu/host1x/hw/hw_host1x04_uclass.h
//! [channel_hw.c]: https://github.com/torvalds/linux/blob/master/drivers/gpu/host1x/hw/channel_hw.c

use std::{
    sync::{Arc, Condvar, Mutex, MutexGuard, Weak},
    time::Duration,
};

use serde::{Deserialize, Serialize};

use super::{Notes, host1x::Syncpoints, mc::DeviceMemory};
use crate::{
    bus::MmioDevice,
    clock::GuestClock,
    snapshot::{self, Snapshot, SnapshotError, encode},
};

type Engine = dyn Fn(u32, u32, u32) -> Option<()> + Send + Sync;

/// How often the waiting thread looks at the clock while it stands still,
/// and whether its channel is still there, when no syncpoint changes.
const PAUSED_POLL: Duration = Duration::from_millis(20);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Opcode {
    SetClass {
        class: u32,
        offset: u32,
        mask: u16,
    },
    Incr {
        offset: u32,
        count: u16,
    },
    NonIncr {
        offset: u32,
        count: u16,
    },
    Mask {
        offset: u32,
        mask: u16,
    },
    Imm {
        offset: u32,
        value: u32,
    },
    Restart(u32),
    Gather {
        offset: u32,
        count: u16,
        insert: bool,
        increment: bool,
    },
}

/// Pure decoding: unsupported opcode families are not silently treated as NOPs.
fn decode(word: u32) -> Option<Opcode> {
    let offset = (word >> 16) & 0xfff;
    let count = word as u16;
    Some(match word >> 28 {
        0 => Opcode::SetClass {
            class: (word >> 6) & 0x3ff,
            offset,
            mask: (word & 0x3f) as u16,
        },
        1 => Opcode::Incr { offset, count },
        2 => Opcode::NonIncr { offset, count },
        3 => Opcode::Mask {
            offset,
            mask: count,
        },
        4 => Opcode::Imm {
            offset,
            value: word & 0xffff,
        },
        5 => Opcode::Restart((word & 0x0fff_ffff) << 4),
        6 => Opcode::Gather {
            offset,
            count: count & 0x3fff,
            insert: word & (1 << 15) != 0,
            increment: word & (1 << 14) != 0,
        },
        _ => return None,
    })
}

#[derive(Clone, Copy, Serialize, Deserialize)]
enum Pending {
    None,
    Sequence {
        offset: u32,
        count: u16,
        increment: bool,
    },
    Mask {
        offset: u32,
        mask: u16,
    },
    Gather {
        offset: u32,
        count: u16,
        insert: bool,
        increment: bool,
    },
}

#[derive(Clone, Copy, Serialize, Deserialize)]
struct Frame {
    get: u64,
    end: u64,
    // Insert gathers contain raw method data, not opcodes.
    insert: Option<(u32, bool)>,
}

/// A host syncpoint wait the stream stands at: syncpoint `index` must reach
/// `threshold`, compared in the low `bits` bits (the legacy wait has 24).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Wait {
    index: usize,
    threshold: u32,
    bits: u32,
}

impl Wait {
    fn met(&self, syncpoints: &Syncpoints) -> bool {
        let current = syncpoints.get(self.index).unwrap_or(0);
        (current.wrapping_sub(self.threshold) << (32 - self.bits)) as i32 >= 0
    }
}

const FRAME_DEPTH: usize = 32;

/// The channel's registers and where its stream stands; what its
/// [`Snapshot`] saves.
#[derive(Serialize, Deserialize)]
struct State {
    start: u32,
    put: u32,
    get: u32,
    end: u32,
    ctrl: u32,
    class: u32,
    payload: Option<u32>,
    pending: Pending,
    /// Gathers being executed, innermost last (`depth` of them).
    frames: [Frame; FRAME_DEPTH],
    depth: usize,
    wait: Option<Wait>,
    /// Whether a thread is watching `wait`. Not saved: it says whether
    /// this machine has such a thread.
    #[serde(skip)]
    polling: bool,
    /// Whether a thread is running the stream (see [`run`]). Not saved: a
    /// snapshot is taken between runs.
    #[serde(skip)]
    running: bool,
    fault: bool,
}

/// An engine method the stream reached: class, method offset, data.
type Call = (u32, u32, u32);

impl State {
    fn idle(&self) -> bool {
        !self.fault
            && !self.running
            && self.get == self.put
            && self.depth == 0
            && self.wait.is_none()
            && matches!(self.pending, Pending::None)
    }
}

pub struct Channel {
    inner: Arc<Inner>,
}

struct Inner {
    memory: DeviceMemory,
    syncpoints: Arc<Syncpoints>,
    /// Guest time: a stalled channel does not go on while it stands still.
    clock: Arc<GuestClock>,
    engine: Arc<Engine>,
    notes: Arc<Notes>,
    state: Mutex<State>,
    /// Signalled when a run of the stream ends.
    ran: Condvar,
}

impl Channel {
    pub fn new(
        memory: DeviceMemory,
        syncpoints: Arc<Syncpoints>,
        clock: Arc<GuestClock>,
        engine: Arc<Engine>,
        notes: Arc<Notes>,
    ) -> Self {
        let frame = Frame {
            get: 0,
            end: 0,
            insert: None,
        };
        Self {
            inner: Arc::new(Inner {
                memory,
                syncpoints,
                clock,
                engine,
                notes,
                state: Mutex::new(State {
                    start: 0,
                    put: 0,
                    get: 0,
                    end: 0,
                    ctrl: 1,
                    class: 0,
                    payload: None,
                    pending: Pending::None,
                    frames: [frame; FRAME_DEPTH],
                    depth: 0,
                    wait: None,
                    polling: false,
                    running: false,
                    fault: false,
                }),
                ran: Condvar::new(),
            }),
        }
    }

    /// The channel's CBSTAT and CBREAD, as the host1x sync block shows
    /// them: the class (bits 25:16) and method offset (bits 15:0) its
    /// command processor stands at, and that method's data (Linux
    /// `hw_host1x04_sync.h` `CBSTAT_CBCLASS`/`CBOFFSET`; `debug_hw_1x01.c`
    /// reads class 1 offset `WAIT_SYNCPT` as "waiting on syncpt
    /// CBREAD >> 24 val CBREAD & 0xffffff"). nvservices' CDMA timeout
    /// thread (`0x828A0`) reads both and lets a channel standing at a wait
    /// for one of its job's syncpoints run on (offset 8: id in CBREAD bits
    /// 31:24; offset 0x50: id in CBREAD bits 7:0).
    ///
    /// Known while the stream stands at a host syncpoint wait, the one
    /// method that stays in the command processor here. A read while an
    /// engine method runs waits for the run to end (the method in the
    /// processor is not kept), so it sees what it did when every method
    /// completed within the DMAPUT write that reached it. Otherwise, idle
    /// or faulted, the method the processor last held is not kept, and
    /// this refuses with a note.
    pub fn command_status(&self) -> Option<(u32, u32)> {
        let state = self.inner.between_runs();
        let Some(wait) = state.wait else {
            self.inner.notes.add(format!(
                "host1x channel: CBSTAT/CBREAD read while the stream stands at no \
                 syncpoint wait (faulted: {}); its last method is not kept",
                state.fault
            ));
            return None;
        };
        // Class 1 (host), at WAIT_SYNCPT (0x8, the legacy 24-bit wait) or
        // WAIT_SYNCPT_32 (0x50, data the id; its threshold came by 0x4e).
        let index = wait.index as u32;
        Some(if wait.bits == 24 {
            (1 << 16 | 0x8, index << 24 | wait.threshold)
        } else {
            (1 << 16 | 0x50, index)
        })
    }
}

impl Inner {
    /// Does a host class method; an engine method is handed back to be
    /// run without the channel held.
    fn method(
        &self,
        state: &mut State,
        offset: u32,
        value: u32,
    ) -> Result<Option<Call>, &'static str> {
        if offset > 0xfff {
            return Err("method offset exceeds 12-bit range");
        }
        if state.class != 1 {
            return Ok(Some((state.class, offset, value)));
        }
        match offset {
            0 => {
                // INCR_SYNCPT: index 7:0, condition 15:8 (Linux
                // `hw_host1x04_uclass.h`: 0 IMMEDIATE, 1 OP_DONE, 2 RD_DONE,
                // 3 REG_WR_SAFE). The host class's own operations are its
                // methods, each done when the command processor goes past
                // it: by the time this one is reached every earlier one in
                // the stream is done, read and written, so all four hold
                // at once [inference: the conditions name the class's own
                // unit, which for class 1 is the command processor]. An
                // engine's work started earlier is not waited for: that is
                // the engine's own increment's to say (a client waits for an
                // engine with a WAIT_SYNCPT on that increment).
                let (index, condition) = ((value & 0xff) as usize, value >> 8);
                if condition > 3 {
                    self.notes.add(format!(
                        "host1x channel: host syncpoint increment {value:#x} (condition {condition})"
                    ));
                    return Err("reserved host syncpoint increment condition");
                }
                self.syncpoints.get(index).ok_or("invalid syncpoint ID")?;
                self.syncpoints.add(index, 1);
            }
            8 => {
                let index = (value >> 24) as usize;
                self.syncpoints
                    .get(index)
                    .ok_or("invalid wait syncpoint ID")?;
                state.wait = Some(Wait {
                    index,
                    threshold: value & 0xffffff,
                    bits: 24,
                });
            }
            0x4e => state.payload = Some(value),
            0x50 => {
                let threshold = state.payload.ok_or("32-bit wait has no loaded payload")?;
                self.syncpoints
                    .get(value as usize)
                    .ok_or("invalid wait syncpoint ID")?;
                state.wait = Some(Wait {
                    index: value as usize,
                    threshold,
                    bits: 32,
                });
            }
            9 | 0xb => return Err("shared syncpoint wait-base methods are unsupported"),
            _ => return Err("unknown host1x class method"),
        }
        Ok(None)
    }

    fn word(&self, address: u64) -> Result<u32, &'static str> {
        if address & 3 != 0 {
            return Err("unaligned CDMA address");
        }
        let mut bytes = [0; 4];
        self.memory
            .read(address, &mut bytes)
            .ok_or("unmapped CDMA command memory")?;
        Ok(u32::from_le_bytes(bytes))
    }

    /// The channel locked when no thread is running its stream: the state
    /// a snapshot takes, between engine methods' effects and not inside
    /// one.
    fn between_runs(&self) -> MutexGuard<'_, State> {
        let state = self.state.lock().unwrap();
        self.ran.wait_while(state, |state| state.running).unwrap()
    }

    /// Runs the stream until it reaches PUT, stands at an unmet wait
    /// (`None`) or reaches an engine method, which it hands back, the
    /// stream standing past it. `budget` limits the words fetched in one
    /// [`run`].
    fn execute(&self, state: &mut State, budget: &mut usize) -> Result<Option<Call>, &'static str> {
        if state.fault {
            return Err("channel remains faulted; DMA GET reset required");
        }
        loop {
            if let Some(wait) = state.wait {
                if !wait.met(&self.syncpoints) {
                    return Ok(None);
                }
                state.wait = None;
            }
            let depth = state.depth;
            if depth != 0 && state.frames[depth - 1].get == state.frames[depth - 1].end {
                if !matches!(state.pending, Pending::None) {
                    return Err("gather ends inside a command packet");
                }
                state.depth -= 1;
                continue;
            }
            if depth == 0 && state.get == state.put {
                return Ok(None);
            }
            if *budget == 0 {
                return Err("CDMA execution limit reached (possible restart/gather cycle)");
            }
            *budget -= 1;
            let address = if depth == 0 {
                if state.get < state.start || state.get >= state.end {
                    return Err("DMA GET outside START/END");
                }
                u64::from(state.get)
            } else {
                state.frames[depth - 1].get
            };
            let value = self.word(address)?;
            if depth == 0 {
                state.get = state.get.checked_add(4).ok_or("DMA GET overflow")?;
            } else {
                let frame = &mut state.frames[depth - 1];
                frame.get += 4;
                if let Some((offset, increment)) = frame.insert {
                    frame.insert = Some((offset + u32::from(increment), increment));
                    match self.method(state, offset, value)? {
                        Some(call) => return Ok(Some(call)),
                        None => continue,
                    }
                }
            }
            match std::mem::replace(&mut state.pending, Pending::None) {
                Pending::Sequence {
                    offset,
                    count,
                    increment,
                } => {
                    if count > 1 {
                        state.pending = Pending::Sequence {
                            offset: offset + u32::from(increment),
                            count: count - 1,
                            increment,
                        };
                    }
                    match self.method(state, offset, value)? {
                        Some(call) => return Ok(Some(call)),
                        None => continue,
                    }
                }
                Pending::Mask { offset, mask } => {
                    let bit = mask.trailing_zeros();
                    let rest = mask & (mask - 1);
                    if rest != 0 {
                        state.pending = Pending::Mask { offset, mask: rest };
                    }
                    match self.method(state, offset + bit, value)? {
                        Some(call) => return Ok(Some(call)),
                        None => continue,
                    }
                }
                Pending::Gather {
                    offset,
                    count,
                    insert,
                    increment,
                } => {
                    if value & 3 != 0 {
                        return Err("unaligned gather address");
                    }
                    if count != 0 {
                        if depth == FRAME_DEPTH {
                            return Err("nested gather depth exceeded");
                        }
                        let end = u64::from(value) + u64::from(count) * 4;
                        if end > 1u64 << 32 {
                            return Err("gather crosses 32-bit address space");
                        }
                        state.frames[depth] = Frame {
                            get: value.into(),
                            end,
                            insert: insert.then_some((offset, increment)),
                        };
                        state.depth += 1;
                    }
                    continue;
                }
                Pending::None => {}
            }
            match decode(value).ok_or("unsupported CDMA opcode")? {
                Opcode::SetClass {
                    class,
                    offset,
                    mask,
                } => {
                    state.class = class;
                    if mask != 0 {
                        state.pending = Pending::Mask { offset, mask };
                    }
                }
                Opcode::Incr { offset, count } | Opcode::NonIncr { offset, count } => {
                    if count != 0 {
                        state.pending = Pending::Sequence {
                            offset,
                            count,
                            increment: value >> 28 == 1,
                        };
                    }
                }
                Opcode::Mask { offset, mask } => {
                    if mask != 0 {
                        state.pending = Pending::Mask { offset, mask };
                    }
                }
                Opcode::Imm { offset, value } => {
                    if let Some(call) = self.method(state, offset, value)? {
                        return Ok(Some(call));
                    }
                }
                Opcode::Restart(target) => {
                    if depth != 0 {
                        // Restart changes the DMA ring GET, not the gather cursor.
                        // This combination is not used by the evidenced submit path.
                        return Err("RESTART inside gather is unsupported");
                    }
                    if target < state.start || target >= state.end {
                        return Err("restart target outside START/END");
                    }
                    state.get = target;
                }
                Opcode::Gather {
                    offset,
                    count,
                    insert,
                    increment,
                } => {
                    if !insert && (offset != 0 || increment) {
                        return Err("reserved gather header fields");
                    }
                    state.pending = Pending::Gather {
                        offset,
                        count,
                        insert,
                        increment,
                    };
                }
            }
        }
    }
}

/// Runs the stream from `state` (the channel locked) unless another thread
/// already does, letting go of the channel while each engine method runs;
/// then has a thread watch the wait it stands at, if any. On failure faults
/// the channel, leaves a note and returns `None`.
fn run<'a>(
    inner: &'a Arc<Inner>,
    mut state: MutexGuard<'a, State>,
) -> (MutexGuard<'a, State>, Option<()>) {
    if state.running {
        return (state, Some(()));
    }
    state.running = true;
    // Limits refuse loops rather than claiming that work completed.
    let mut budget = 1_048_576usize;
    let outcome = loop {
        match inner.execute(&mut state, &mut budget) {
            Ok(None) => break Ok(()),
            Ok(Some((class, offset, value))) => {
                drop(state);
                let accepted = (inner.engine)(class, offset, value);
                state = inner.state.lock().unwrap();
                if accepted.is_none() {
                    break Err("engine refused method");
                }
            }
            Err(reason) => break Err(reason),
        }
    };
    state.running = false;
    inner.ran.notify_all();
    if let Err(reason) = outcome {
        state.fault = true;
        state.wait = None;
        inner.notes.add(format!(
            "host1x channel: {reason}; GET={:#010x}, PUT={:#010x}, class={:#x}",
            state.get, state.put, state.class
        ));
        return (state, None);
    }
    watch(inner, &mut state);
    (state, Some(()))
}

/// Watches the wait `state` stands at from a thread of its own, unless one
/// already does.
fn watch(inner: &Arc<Inner>, state: &mut State) {
    if state.wait.is_some() && !state.polling {
        state.polling = true;
        let (weak, clock) = (Arc::downgrade(inner), Arc::clone(&inner.clock));
        std::thread::Builder::new()
            .name("host1x channel wait".into())
            .spawn(move || poll(weak, clock))
            .expect("host1x channel wait thread");
    }
}

/// Continues `inner`'s stream whenever its wait is met, looking at every
/// change of the syncpoints, until it no longer stands at one or the
/// channel is gone. Nothing continues while the clock stands still: the
/// machine is paused, perhaps being saved.
fn poll(inner: Weak<Inner>, clock: Arc<GuestClock>) {
    loop {
        let syncpoints = {
            let Some(inner) = inner.upgrade() else {
                return;
            };
            // Read before the wait is looked at: a change after it wakes
            // the wait below.
            let seen = inner.syncpoints.changes();
            let mut state = inner.state.lock().unwrap();
            if !clock.is_paused() && state.wait.is_some_and(|wait| wait.met(&inner.syncpoints)) {
                // A failure faults the channel; its note and the refusal of
                // the next DMAPUT write report it.
                state = run(&inner, state).0;
            }
            if state.wait.is_none() {
                state.polling = false;
                return;
            }
            drop(state);
            (Arc::clone(&inner.syncpoints), seen)
        };
        syncpoints.0.wait_change(syncpoints.1, PAUSED_POLL);
    }
}

impl MmioDevice for Channel {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        if size != 4 || offset & 3 != 0 {
            return None;
        }
        let state = self.inner.state.lock().unwrap();
        let value = match offset {
            0 => u32::from(state.idle()) << 11,
            0x14 => state.start,
            0x18 => state.put,
            0x1c => state.get,
            0x20 => state.end,
            0x24 => state.ctrl,
            _ => return None,
        };
        Some(value.into())
    }

    fn write(&self, _core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        if size != 4 || offset & 3 != 0 {
            return None;
        }
        let inner = &self.inner;
        let value = value as u32;
        let mut state = inner.state.lock().unwrap();
        match offset {
            0x14 | 0x18 | 0x20 if value & 3 != 0 => {
                inner.notes.add(format!(
                    "host1x channel: unaligned DMA register {offset:#x}={value:#x}"
                ));
                return None;
            }
            0x14 => state.start = value,
            0x18 => state.put = value,
            0x20 => state.end = value,
            0x24 => {
                if value & !7 != 0 {
                    inner
                        .notes
                        .add(format!("host1x channel: unsupported DMACTRL {value:#x}"));
                    return None;
                }
                state.ctrl = value;
                if value & 2 != 0 {
                    state.get = if value & 4 != 0 {
                        state.put
                    } else {
                        state.start
                    };
                    state.pending = Pending::None;
                    state.depth = 0;
                    state.wait = None;
                    state.fault = false;
                }
            }
            _ => {
                inner.notes.add(format!(
                    "host1x channel: unsupported register write {offset:#x}={value:#x}"
                ));
                return None;
            }
        }
        if matches!(offset, 0x18 | 0x24) && state.ctrl & 1 == 0 {
            run(inner, state).1?;
        }
        Some(())
    }

    fn snapshot(&self) -> Option<&dyn Snapshot> {
        Some(self)
    }
}

/// Saves the DMA registers, the class, the loaded payload, the command
/// packet in progress, the gathers being executed, the syncpoint wait the
/// stream stands at and whether it has faulted. A channel restored standing
/// at a wait has a thread watching it again.
impl Snapshot for Channel {
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        encode(&*self.inner.between_runs())
    }

    fn restore(&self, saved: &[u8]) -> Result<(), SnapshotError> {
        let restored: State = snapshot::decode("host1x channel", saved)?;
        if restored.depth > FRAME_DEPTH {
            return Err(SnapshotError::Corrupt(format!(
                "host1x channel: {} gathers deep",
                restored.depth
            )));
        }
        let mut state = self.inner.between_runs();
        // A thread already watching goes on watching the restored wait,
        // or ends when there is none.
        let polling = state.polling;
        *state = restored;
        state.polling = polling;
        watch(&self.inner, &mut state);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        boot::DramSize,
        memory::GuestRam,
        soc::mc::{Mc, SMMU_HOST1X_ASID},
    };
    use std::time::Instant;

    const RAM_BASE: u64 = 0x8000_0000;
    /// Device address 0 maps here (one 4 MiB SMMU section).
    const WINDOW: u64 = RAM_BASE + 0x40_0000;
    const RING: u32 = 0x1000;
    const GATHER: u32 = 0x8000;

    /// Whether the test engine lets its methods finish, and its signal.
    type Gate = Arc<(Mutex<bool>, Condvar)>;

    struct Bench {
        ram: Arc<GuestRam>,
        syncpoints: Arc<Syncpoints>,
        notes: Arc<Notes>,
        calls: Arc<Mutex<Vec<(u32, u32, u32)>>>,
        /// Open at first: each engine method records itself, then stands
        /// while the gate is closed.
        gate: Gate,
        channel: Arc<Channel>,
    }

    impl Bench {
        fn new() -> Self {
            let ram = Arc::new(GuestRam::new(RAM_BASE, 0x80_0000).unwrap());
            let mc = Arc::new(Mc::new(DramSize::Gb4));
            mc.write(0, 0x1c, 4, 1).unwrap();
            mc.write(0, 0x20, 4, 0xe008_0000).unwrap();
            mc.write(0, SMMU_HOST1X_ASID, 4, 0x8000_0001).unwrap();
            ram.write(RAM_BASE, &0xc008_0400u32.to_le_bytes()).unwrap();
            let syncpoints = Arc::new(Syncpoints::default());
            let notes: Arc<Notes> = Arc::default();
            let calls = Arc::new(Mutex::new(Vec::new()));
            let gate: Gate = Arc::new((Mutex::new(true), Condvar::new()));
            let (record, held) = (Arc::clone(&calls), Arc::clone(&gate));
            let channel = Arc::new(Channel::new(
                DeviceMemory::new(&mc, &ram, SMMU_HOST1X_ASID),
                Arc::clone(&syncpoints),
                Arc::new(GuestClock::new()),
                Arc::new(move |class, offset, value| {
                    record.lock().unwrap().push((class, offset, value));
                    let (open, signal) = &*held;
                    drop(
                        signal
                            .wait_while(open.lock().unwrap(), |open| !*open)
                            .unwrap(),
                    );
                    Some(())
                }),
                Arc::clone(&notes),
            ));
            for (register, value) in [(0x14, RING), (0x20, RING + 0x1000), (0x18, RING)] {
                channel.write(0, register, 4, value.into()).unwrap();
            }
            channel.write(0, 0x24, 4, 7).unwrap();
            channel.write(0, 0x24, 4, 0).unwrap();
            Bench {
                ram,
                syncpoints,
                notes,
                calls,
                gate,
                channel,
            }
        }

        fn poke(&self, address: u32, words: &[u32]) {
            let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            self.ram.write(WINDOW + u64::from(address), &bytes).unwrap();
        }

        /// Puts `words` in the ring and moves PUT past them.
        fn submit(&self, words: &[u32]) -> Option<()> {
            self.poke(RING, words);
            self.channel
                .write(0, 0x18, 4, u64::from(RING) + 4 * words.len() as u64)
        }

        fn idle(&self) -> bool {
            self.channel.read(0, 0, 4) == Some(1 << 11)
        }

        /// Waits (up to a second) for the engine to have seen `count` methods.
        fn engine_calls(&self, count: usize) -> Vec<(u32, u32, u32)> {
            let deadline = Instant::now() + Duration::from_secs(1);
            while self.calls.lock().unwrap().len() < count && Instant::now() < deadline {
                std::thread::sleep(Duration::from_micros(500));
            }
            self.calls.lock().unwrap().clone()
        }
    }

    /// `SETCLASS class, offset, mask`.
    fn setclass(class: u32, offset: u32, mask: u32) -> u32 {
        offset << 16 | class << 6 | mask
    }

    /// The 32-bit wait nvservices puts before a job (Linux `channel_hw.c`):
    /// payload at host method 0x4e, syncpoint at 0x50.
    fn wait32(index: u32, threshold: u32) -> [u32; 4] {
        [setclass(1, 0x4e, 1), threshold, setclass(1, 0x50, 1), index]
    }

    #[test]
    fn an_unmet_wait_stalls_the_channel_until_its_syncpoint_gets_there() {
        let bench = Bench::new();
        bench.syncpoints.add(7, 4);
        let mut words = wait32(7, 5).to_vec();
        words.extend([setclass(0x5d, 0, 0), 0x1010_0002, 0xc0, 0x100]);
        bench.submit(&words).unwrap();
        assert!(bench.calls.lock().unwrap().is_empty());
        assert!(!bench.idle());
        // The stream stands right after the wait.
        assert_eq!(bench.channel.read(0, 0x1c, 4), Some(u64::from(RING) + 16));
        bench.syncpoints.add(7, 1);
        assert_eq!(
            bench.engine_calls(2),
            [(0x5d, 0x10, 0xc0), (0x5d, 0x11, 0x100)]
        );
        assert!(bench.idle());
        assert!(bench.notes.take().is_empty());
    }

    /// While an engine method runs (a VIC EXECUTE composing), the channel's
    /// registers answer at once: GET and the status are read, a DMAPUT is
    /// written and returns, its words then run on the thread that runs the
    /// stream once the method is done; a snapshot taken meanwhile waits for
    /// the run to end and holds where it ended.
    #[test]
    fn register_access_while_an_engine_method_runs_does_not_wait_for_it() {
        let bench = Arc::new(Bench::new());
        *bench.gate.0.lock().unwrap() = false;
        let first = [setclass(0x5d, 0, 0), 0x1010_0002, 0xc0, 0x100];
        let feeder = {
            let bench = Arc::clone(&bench);
            std::thread::spawn(move || bench.submit(&first))
        };
        assert_eq!(bench.engine_calls(1), [(0x5d, 0x10, 0xc0)]);
        let (done, finished) = std::sync::mpsc::channel();
        {
            let bench = Arc::clone(&bench);
            std::thread::spawn(move || {
                let get = bench.channel.read(0, 0x1c, 4);
                let idle = bench.idle();
                // Two more methods after the first four words.
                bench.poke(RING + 16, &[0x1010_0002, 0xc4, 0x200]);
                let put = bench.channel.write(0, 0x18, 4, u64::from(RING) + 28);
                done.send((get, idle, put)).unwrap();
            });
        }
        let (get, idle, put) = finished
            .recv_timeout(Duration::from_secs(2))
            .expect("register access waited for the engine method");
        // GET stands past the method running; the channel is busy.
        assert_eq!(get, Some(u64::from(RING) + 12));
        assert!(!idle);
        assert_eq!(put, Some(()));
        let saver = {
            let bench = Arc::clone(&bench);
            std::thread::spawn(move || bench.channel.save().unwrap())
        };
        std::thread::sleep(Duration::from_millis(50));
        assert!(
            !saver.is_finished(),
            "a snapshot taken inside an engine method"
        );
        assert_eq!(bench.calls.lock().unwrap().len(), 1);
        *bench.gate.0.lock().unwrap() = true;
        bench.gate.1.notify_all();
        assert_eq!(feeder.join().unwrap(), Some(()));
        assert_eq!(
            bench.engine_calls(4),
            [
                (0x5d, 0x10, 0xc0),
                (0x5d, 0x11, 0x100),
                (0x5d, 0x10, 0xc4),
                (0x5d, 0x11, 0x200)
            ]
        );
        assert!(bench.idle());
        let saved: State = snapshot::decode("host1x channel", &saver.join().unwrap()).unwrap();
        assert!(
            saved.idle(),
            "the snapshot holds the stream where its run ended"
        );
        assert!(bench.notes.take().is_empty());
    }

    /// CBSTAT and CBREAD of channel 0, read through the sync block as
    /// nvservices' CDMA timeout thread reads them: the wait the stream
    /// stands at, either kind; nothing for a channel that stands at none.
    #[test]
    fn the_sync_block_shows_the_wait_a_channel_stands_at() {
        use crate::soc::host1x::Host1xSync;
        let sync = |bench: &Bench| {
            Host1xSync::new(
                Arc::clone(&bench.syncpoints),
                Vec::new(),
                vec![Arc::clone(&bench.channel)],
            )
        };
        let bench = Bench::new();
        let host1x = sync(&bench);
        assert_eq!(host1x.read(0, 0xCC0, 4), None);
        assert!(
            bench.notes.take()[0].contains("stands at no syncpoint wait"),
            "an idle channel's last method is not kept"
        );
        bench.syncpoints.add(7, 4);
        bench.submit(&wait32(7, 5)).unwrap();
        // Class 1, WAIT_SYNCPT_32; CBREAD the syncpoint id.
        assert_eq!(host1x.read(0, 0xCC0, 4), Some(0x1_0050));
        assert_eq!(host1x.read(0, 0xC80, 4), Some(7));
        // Channel 1 is not there.
        assert_eq!(host1x.read(0, 0xCC4, 4), None);

        let bench = Bench::new();
        let host1x = sync(&bench);
        bench.submit(&[setclass(1, 0x8, 1), 3 << 24 | 9]).unwrap();
        // Class 1, WAIT_SYNCPT; CBREAD id << 24 | threshold.
        assert_eq!(host1x.read(0, 0xCC0, 4), Some(0x1_0008));
        assert_eq!(host1x.read(0, 0xC80, 4), Some(0x0300_0009));
        assert!(bench.notes.take().is_empty());
    }

    #[test]
    fn a_restored_channel_standing_at_a_wait_continues_once_it_is_met() {
        let saved = Bench::new();
        saved.syncpoints.add(7, 4);
        let mut words = wait32(7, 5).to_vec();
        words.extend([setclass(0x5d, 0, 0), 0x1010_0002, 0xc0, 0x100]);
        saved.submit(&words).unwrap();
        let state = saved.channel.save().unwrap();

        let restored = Bench::new();
        // Memory and syncpoints are their owners' to restore.
        restored.poke(RING, &words);
        restored.syncpoints.add(7, 4);
        restored.channel.restore(&state).unwrap();
        assert!(!restored.idle());
        restored.syncpoints.add(7, 1);
        assert_eq!(
            restored.engine_calls(2),
            [(0x5d, 0x10, 0xc0), (0x5d, 0x11, 0x100)]
        );
        assert!(restored.idle());
    }

    #[test]
    fn a_met_wait_runs_through_and_a_wait_inside_a_gather_resumes_inside_it() {
        let bench = Bench::new();
        bench.syncpoints.add(3, 2);
        // Met at once: the legacy 24-bit form, id in bits 31:24.
        bench
            .submit(&[setclass(1, 8, 1), 3 << 24 | 2, setclass(0x5d, 0x10, 1), 1])
            .unwrap();
        assert_eq!(bench.calls.lock().unwrap().len(), 1);
        let mut gather = wait32(3, 3).to_vec();
        gather.extend([setclass(0x5d, 0x11, 1), 2]);
        bench.poke(GATHER, &gather);
        // The ring continues after the gather; the method after it waits too.
        bench.poke(RING + 16, &[0x6000_0000 | gather.len() as u32, GATHER]);
        bench.poke(RING + 24, &[setclass(0x5d, 0x12, 1), 3]);
        bench
            .channel
            .write(0, 0x18, 4, u64::from(RING) + 32)
            .unwrap();
        assert_eq!(bench.calls.lock().unwrap().len(), 1);
        bench.syncpoints.add(3, 1);
        assert_eq!(
            bench.engine_calls(3),
            [(0x5d, 0x10, 1), (0x5d, 0x11, 2), (0x5d, 0x12, 3)]
        );
        assert!(bench.idle());
    }

    #[test]
    fn a_get_reset_abandons_a_stalled_wait() {
        let bench = Bench::new();
        bench.submit(&wait32(9, 1)).unwrap();
        assert!(!bench.idle());
        bench.channel.write(0, 0x24, 4, 7).unwrap();
        bench.channel.write(0, 0x24, 4, 0).unwrap();
        assert!(bench.idle());
    }

    #[test]
    fn class_and_mask_fields() {
        assert_eq!(
            decode((0x4e << 16) | (1 << 6) | 5),
            Some(Opcode::SetClass {
                class: 1,
                offset: 0x4e,
                mask: 5
            })
        );
        assert_eq!(
            decode(0x3abc8001),
            Some(Opcode::Mask {
                offset: 0xabc,
                mask: 0x8001
            })
        );
        assert_eq!(
            decode(0x0fffffff),
            Some(Opcode::SetClass {
                class: 0x3ff,
                offset: 0xfff,
                mask: 0x3f
            })
        );
    }

    #[test]
    fn sequential_and_immediate_packets() {
        assert_eq!(
            decode(0x11230002),
            Some(Opcode::Incr {
                offset: 0x123,
                count: 2
            })
        );
        assert_eq!(
            decode(0x2123ffff),
            Some(Opcode::NonIncr {
                offset: 0x123,
                count: 0xffff
            })
        );
        assert_eq!(
            decode(0x20000000),
            Some(Opcode::NonIncr {
                offset: 0,
                count: 0
            })
        );
        assert_eq!(
            decode(0x4123abcd),
            Some(Opcode::Imm {
                offset: 0x123,
                value: 0xabcd
            })
        );
    }

    #[test]
    fn restart_and_gather_fields() {
        assert_eq!(decode(0x51234567), Some(Opcode::Restart(0x12345670)));
        assert_eq!(
            decode(0x60003fff),
            Some(Opcode::Gather {
                offset: 0,
                count: 0x3fff,
                insert: false,
                increment: false
            })
        );
        assert_eq!(
            decode(0x61238005),
            Some(Opcode::Gather {
                offset: 0x123,
                count: 5,
                insert: true,
                increment: false
            })
        );
        assert_eq!(
            decode(0x6123c005),
            Some(Opcode::Gather {
                offset: 0x123,
                count: 5,
                insert: true,
                increment: true
            })
        );
    }

    #[test]
    fn unknown_opcodes_refuse() {
        for opcode in [7, 8, 9, 10, 11, 12, 13, 14, 15] {
            assert_eq!(decode(opcode << 28), None);
        }
    }

    /// Host-class increments. With Minecraft running and video capture on,
    /// the channel that carries the VIC's submissions sends host increment
    /// 0x10C (syncpoint 12, condition 1, OP_DONE). Each of the four
    /// conditions counts when the processor reaches it; a reserved one
    /// refuses and names the value.
    #[test]
    fn host_increments_of_every_condition_count_at_once() {
        let bench = Bench::new();
        let mut words = Vec::new();
        for condition in 0..4 {
            words.extend([setclass(1, 0, 1), condition << 8 | 9]);
        }
        bench.submit(&words).unwrap();
        assert_eq!(bench.syncpoints.get(9), Some(4));
        assert!(bench.idle());
        let bench = Bench::new();
        assert_eq!(bench.submit(&[setclass(1, 0, 1), 4 << 8 | 9]), None);
        assert_eq!(bench.syncpoints.get(9), Some(0));
        let notes = bench.notes.take();
        assert!(
            notes[0].contains("increment 0x409 (condition 4)"),
            "{notes:?}"
        );
    }
}
