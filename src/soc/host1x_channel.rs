//! Tegra X1 host1x CDMA execution; offsets passed to engines are words.
//!
//! Encoding evidence: Linux `drivers/gpu/host1x/hw/` [opcodes.h],
//! [hw_host1x04_channel.h], [hw_host1x04_uclass.h], and [channel_hw.c].
//! The latter emits the 32-bit wait as payload at 0x4e followed by ID at 0x50.
//! nvservices initializes GET from PUT with DMACTRL=7, then runs with zero.
//!
//! A stream runs synchronously on the DMAPUT write until it reaches PUT or a
//! host syncpoint wait whose threshold has not been reached. Such a wait
//! stalls the channel where it stands (ring position and gather nesting are
//! kept), as the hardware's command processor does; it continues once the
//! syncpoint gets there. Nothing announces syncpoint changes, so a stalled
//! channel looks again every `WAIT_POLL` from a thread of its own, like
//! the GPU's channel waits.
//!
//! Unimplemented methods/conditions fail the stream and never manufacture a
//! fence completion; the DMAPUT write that ran into them is refused, or, for
//! a stream that failed while continuing after a wait, the next one. Shared
//! wait-base registers and conditional host increments are not modeled. A
//! failed stream stays faulted until DMA GET reset, so retrying a failed
//! submission cannot replay its effects.
//!
//! [opcodes.h]: https://github.com/torvalds/linux/blob/master/drivers/gpu/host1x/hw/opcodes.h
//! [hw_host1x04_channel.h]: https://github.com/torvalds/linux/blob/master/drivers/gpu/host1x/hw/hw_host1x04_channel.h
//! [hw_host1x04_uclass.h]: https://github.com/torvalds/linux/blob/master/drivers/gpu/host1x/hw/hw_host1x04_uclass.h
//! [channel_hw.c]: https://github.com/torvalds/linux/blob/master/drivers/gpu/host1x/hw/channel_hw.c

use std::{
    sync::{Arc, Mutex, Weak},
    time::Duration,
};

use super::{Notes, host1x::Syncpoints, mc::DeviceMemory};
use crate::bus::MmioDevice;

type Engine = dyn Fn(u32, u32, u32) -> Option<()> + Send + Sync;

/// How often a channel stalled on a syncpoint wait looks again (the GPU's
/// channels poll at the same rate).
const WAIT_POLL: Duration = Duration::from_micros(500);

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

#[derive(Clone, Copy)]
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

#[derive(Clone, Copy)]
struct Frame {
    get: u64,
    end: u64,
    // Insert gathers contain raw method data, not opcodes.
    insert: Option<(u32, bool)>,
}

/// A host syncpoint wait the stream stands at: syncpoint `index` must reach
/// `threshold`, compared in the low `bits` bits (the legacy wait has 24).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
    /// Whether a thread is watching `wait`.
    polling: bool,
    fault: bool,
}

impl State {
    fn idle(&self) -> bool {
        !self.fault
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
    engine: Arc<Engine>,
    notes: Arc<Notes>,
    state: Mutex<State>,
}

impl Channel {
    pub fn new(
        memory: DeviceMemory,
        syncpoints: Arc<Syncpoints>,
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
                    fault: false,
                }),
            }),
        }
    }
}

impl Inner {
    fn method(&self, state: &mut State, offset: u32, value: u32) -> Result<(), &'static str> {
        if offset > 0xfff {
            return Err("method offset exceeds 12-bit range");
        }
        if state.class != 1 {
            return (self.engine)(state.class, offset, value).ok_or("engine refused method");
        }
        match offset {
            0 => {
                if value & !0xff != 0 {
                    return Err("conditional/reserved host syncpoint increment is unsupported");
                }
                self.syncpoints
                    .get(value as usize)
                    .ok_or("invalid syncpoint ID")?;
                self.syncpoints.add(value as usize, 1);
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
        Ok(())
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

    /// Runs the stream until it reaches PUT or stands at an unmet wait.
    fn execute(&self, state: &mut State) -> Result<(), &'static str> {
        if state.fault {
            return Err("channel remains faulted; DMA GET reset required");
        }
        // Limits refuse loops rather than claiming that work completed.
        let mut budget = 1_048_576usize;
        loop {
            if let Some(wait) = state.wait {
                if !wait.met(&self.syncpoints) {
                    return Ok(());
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
                return Ok(());
            }
            if budget == 0 {
                return Err("CDMA execution limit reached (possible restart/gather cycle)");
            }
            budget -= 1;
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
                    self.method(state, offset, value)?;
                    continue;
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
                    self.method(state, offset, value)?;
                    continue;
                }
                Pending::Mask { offset, mask } => {
                    let bit = mask.trailing_zeros();
                    let rest = mask & (mask - 1);
                    if rest != 0 {
                        state.pending = Pending::Mask { offset, mask: rest };
                    }
                    self.method(state, offset + bit, value)?;
                    continue;
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
                Opcode::Imm { offset, value } => self.method(state, offset, value)?,
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

    /// Runs the stream; on failure faults the channel and leaves a note.
    fn run(&self, state: &mut State) -> Option<()> {
        if let Err(reason) = self.execute(state) {
            state.fault = true;
            state.wait = None;
            self.notes.add(format!(
                "host1x channel: {reason}; GET={:#010x}, PUT={:#010x}, class={:#x}",
                state.get, state.put, state.class
            ));
            return None;
        }
        Some(())
    }
}

/// Continues `inner`'s stream whenever its wait is met, until it no longer
/// stands at one or the channel is gone.
fn poll(inner: Weak<Inner>) {
    loop {
        std::thread::sleep(WAIT_POLL);
        let Some(inner) = inner.upgrade() else {
            return;
        };
        let mut state = inner.state.lock().unwrap();
        if state.wait.is_some_and(|wait| wait.met(&inner.syncpoints)) {
            // A failure faults the channel; its note and the refusal of the
            // next DMAPUT write report it.
            let _ = inner.run(&mut state);
        }
        if state.wait.is_none() {
            state.polling = false;
            return;
        }
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
            inner.run(&mut state)?;
            if state.wait.is_some() && !state.polling {
                state.polling = true;
                let weak = Arc::downgrade(inner);
                std::thread::Builder::new()
                    .name("host1x channel wait".into())
                    .spawn(move || poll(weak))
                    .expect("host1x channel wait thread");
            }
        }
        Some(())
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

    struct Bench {
        ram: Arc<GuestRam>,
        syncpoints: Arc<Syncpoints>,
        notes: Arc<Notes>,
        calls: Arc<Mutex<Vec<(u32, u32, u32)>>>,
        channel: Channel,
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
            let record = Arc::clone(&calls);
            let channel = Channel::new(
                DeviceMemory::new(&mc, &ram, SMMU_HOST1X_ASID),
                Arc::clone(&syncpoints),
                Arc::new(move |class, offset, value| {
                    record.lock().unwrap().push((class, offset, value));
                    Some(())
                }),
                Arc::clone(&notes),
            );
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
                std::thread::sleep(WAIT_POLL);
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
}
