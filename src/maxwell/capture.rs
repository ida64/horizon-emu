//! Capture and replay of the work GPU channels do, for measuring and
//! comparing renderers on a real command stream without booting.
//!
//! A [`Capture`] wraps each channel's [`Engines`]: it records the
//! channel's method-init list, every method call (with the time it was
//! made) and the guest memory the engines read. Memory is recorded by
//! difference: per channel, the capture keeps a shadow of every byte the
//! engines have read or written, and logs a read only where guest memory
//! no longer matches the shadow (the first read of a range, or bytes the
//! CPU or another engine changed since). The engines' own writes are not
//! logged, since replay performs them again.
//!
//! [`Replay`] reads a capture back. Replaying applies each logged read to
//! a [`SparseMemory`] before the call it was made in, then runs the call;
//! a read of a byte the capture never saw is refused, so a replay that
//! diverges from the captured run stops instead of inventing memory.
//!
//! File format (little endian), after the magic `b"MXCAP001"`, a sequence
//! of records:
//! - `I` channel `u32`, count `u32`, then `count` × (index `u32`, value
//!   `u32`): the channel's method-init list, before its first call;
//! - `C` channel `u32`, subchannel `u8`, method `u32`, argument `u32`,
//!   microseconds since the capture started `u64`; subchannel [`FLUSH`]
//!   records a call of [`Engines::flush`] instead;
//! - `R` channel `u32`, address `u64`, length `u32`, then the bytes: a read
//!   made during the preceding call of that channel.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufWriter, Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::{Engines, GpuMemory, Unmodeled};

const MAGIC: &[u8; 8] = b"MXCAP001";
const PAGE: u64 = 0x1000;

/// The subchannel of a call record that stands for [`Engines::flush`].
pub const FLUSH: u8 = 0xFF;

/// A capture file being written; shared by the channels it records.
pub struct Capture {
    out: Mutex<BufWriter<File>>,
    failed: std::sync::atomic::AtomicBool,
    next_channel: AtomicU32,
    start: Instant,
}

impl Capture {
    pub fn create(path: &Path) -> io::Result<Arc<Capture>> {
        let mut out = BufWriter::new(File::create(path)?);
        out.write_all(MAGIC)?;
        Ok(Arc::new(Capture {
            out: Mutex::new(out),
            failed: Default::default(),
            next_channel: AtomicU32::new(0),
            start: Instant::now(),
        }))
    }

    /// Wraps the engines of a new channel built from `method_init`.
    pub fn record(
        self: &Arc<Self>,
        method_init: &[(u32, u32)],
        engines: Box<dyn Engines>,
    ) -> Box<dyn Engines> {
        let channel = self.next_channel.fetch_add(1, Ordering::Relaxed);
        let mut record = vec![b'I'];
        record.extend_from_slice(&channel.to_le_bytes());
        record.extend_from_slice(&(method_init.len() as u32).to_le_bytes());
        for &(index, value) in method_init {
            record.extend_from_slice(&index.to_le_bytes());
            record.extend_from_slice(&value.to_le_bytes());
        }
        self.write(&record);
        Box::new(Recorder {
            capture: Arc::clone(self),
            channel,
            inner: engines,
            shadow: Mutex::new(Shadow::default()),
        })
    }

    /// Appends one record. A capture that cannot be written is useless,
    /// and the machine it observes must not be disturbed: the error
    /// is reported once and the capture stops growing.
    fn write(&self, record: &[u8]) {
        let mut out = self.out.lock().unwrap_or_else(|e| e.into_inner());
        if self.failed.load(Ordering::Relaxed) {
            return;
        }
        // Everything before a call is flushed with it, so a machine that
        // stops without unwinding leaves a capture complete up to its
        // last call.
        let result = out.write_all(record).and_then(|()| {
            if record[0] == b'C' {
                out.flush()
            } else {
                Ok(())
            }
        });
        if let Err(error) = result {
            eprintln!("GPU capture: {error}; capture stopped");
            self.failed.store(true, Ordering::Relaxed);
        }
    }
}

/// Bytes known to a capture or replay, by 4 KiB page.
#[derive(Default)]
struct Shadow {
    pages: HashMap<u64, Box<Page>>,
}

struct Page {
    bytes: [u8; PAGE as usize],
    known: [u64; PAGE as usize / 64],
}

impl Shadow {
    /// Whether `data` at `address` is what the shadow holds, every byte
    /// known.
    fn matches(&self, address: u64, data: &[u8]) -> bool {
        spans(address, data.len()).all(|(page, offset, range)| {
            self.pages.get(&page).is_some_and(|p| {
                (offset..offset + range.len()).all(|i| p.known[i / 64] >> (i % 64) & 1 != 0)
                    && p.bytes[offset..offset + range.len()] == data[range]
            })
        })
    }

    fn store(&mut self, address: u64, data: &[u8]) {
        for (page, offset, range) in spans(address, data.len()) {
            let p = self.pages.entry(page).or_insert_with(|| {
                Box::new(Page {
                    bytes: [0; PAGE as usize],
                    known: [0; PAGE as usize / 64],
                })
            });
            for i in offset..offset + range.len() {
                p.known[i / 64] |= 1 << (i % 64);
            }
            p.bytes[offset..offset + range.len()].copy_from_slice(&data[range]);
        }
    }

    /// `buffer` filled from the shadow, or `None` if a byte is unknown.
    fn load(&self, address: u64, buffer: &mut [u8]) -> Option<()> {
        for (page, offset, range) in spans(address, buffer.len()) {
            let p = self.pages.get(&page)?;
            if !(offset..offset + range.len()).all(|i| p.known[i / 64] >> (i % 64) & 1 != 0) {
                return None;
            }
            buffer[range.clone()].copy_from_slice(&p.bytes[offset..offset + range.len()]);
        }
        Some(())
    }
}

/// The pieces of `length` bytes at `address` by page: page number, offset
/// in the page, range in the buffer.
fn spans(
    address: u64,
    length: usize,
) -> impl Iterator<Item = (u64, usize, std::ops::Range<usize>)> {
    let mut done = 0;
    std::iter::from_fn(move || {
        if done == length {
            return None;
        }
        let at = address + done as u64;
        let offset = (at % PAGE) as usize;
        let count = (PAGE as usize - offset).min(length - done);
        let item = (at / PAGE, offset, done..done + count);
        done += count;
        Some(item)
    })
}

struct Recorder {
    capture: Arc<Capture>,
    channel: u32,
    inner: Box<dyn Engines>,
    shadow: Mutex<Shadow>,
}

impl Engines for Recorder {
    fn call(
        &mut self,
        subchannel: u8,
        method: u32,
        argument: u32,
        memory: &dyn GpuMemory,
    ) -> Result<(), Unmodeled> {
        self.call_record(subchannel, method, argument);
        let observed = Observed {
            memory,
            capture: &self.capture,
            channel: self.channel,
            shadow: &self.shadow,
        };
        self.inner.call(subchannel, method, argument, &observed)
    }

    fn flush(&mut self, memory: &dyn GpuMemory) -> Result<(), Unmodeled> {
        self.call_record(FLUSH, 0, 0);
        let observed = Observed {
            memory,
            capture: &self.capture,
            channel: self.channel,
            shadow: &self.shadow,
        };
        self.inner.flush(&observed)
    }
}

impl Recorder {
    fn call_record(&self, subchannel: u8, method: u32, argument: u32) {
        let mut record = Vec::with_capacity(22);
        record.push(b'C');
        record.extend_from_slice(&self.channel.to_le_bytes());
        record.push(subchannel);
        record.extend_from_slice(&method.to_le_bytes());
        record.extend_from_slice(&argument.to_le_bytes());
        let time = self.capture.start.elapsed().as_micros() as u64;
        record.extend_from_slice(&time.to_le_bytes());
        self.capture.write(&record);
    }
}

/// Guest memory as a recorded channel sees it.
struct Observed<'a> {
    memory: &'a dyn GpuMemory,
    capture: &'a Capture,
    channel: u32,
    shadow: &'a Mutex<Shadow>,
}

impl GpuMemory for Observed<'_> {
    fn read(&self, address: u64, buffer: &mut [u8]) -> Option<()> {
        self.memory.read(address, buffer)?;
        let mut shadow = self.shadow.lock().unwrap_or_else(|e| e.into_inner());
        if !shadow.matches(address, buffer) {
            shadow.store(address, buffer);
            let mut record = Vec::with_capacity(17 + buffer.len());
            record.push(b'R');
            record.extend_from_slice(&self.channel.to_le_bytes());
            record.extend_from_slice(&address.to_le_bytes());
            record.extend_from_slice(&(buffer.len() as u32).to_le_bytes());
            record.extend_from_slice(buffer);
            self.capture.write(&record);
        }
        Some(())
    }

    fn write(&self, address: u64, data: &[u8]) -> Option<()> {
        self.memory.write(address, data)?;
        self.shadow
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .store(address, data);
        Some(())
    }
}

/// One record of a capture.
#[derive(Debug, Clone)]
pub enum Record {
    Init {
        channel: u32,
        method_init: Vec<(u32, u32)>,
    },
    Call {
        channel: u32,
        subchannel: u8,
        method: u32,
        argument: u32,
        micros: u64,
    },
    Read {
        channel: u32,
        address: u64,
        data: Vec<u8>,
    },
}

/// A capture read back.
pub struct Replay {
    pub records: Vec<Record>,
}

impl Replay {
    pub fn load(path: &Path) -> io::Result<Replay> {
        let mut bytes = Vec::new();
        File::open(path)?.read_to_end(&mut bytes)?;
        let invalid = |what: &str| io::Error::new(io::ErrorKind::InvalidData, what.to_string());
        if bytes.get(..8) != Some(MAGIC) {
            return Err(invalid("not a GPU capture"));
        }
        let mut at = 8;
        let mut take = |n: usize| -> io::Result<&[u8]> {
            let slice = bytes
                .get(at..at + n)
                .ok_or_else(|| invalid("truncated record"))?;
            at += n;
            Ok(slice)
        };
        let u32_ = |b: &[u8]| u32::from_le_bytes(b.try_into().unwrap());
        let u64_ = |b: &[u8]| u64::from_le_bytes(b.try_into().unwrap());
        let mut records = Vec::new();
        while let Ok(tag) = take(1) {
            let tag = tag[0];
            let record = match tag {
                b'I' => {
                    let channel = u32_(take(4)?);
                    let count = u32_(take(4)?) as usize;
                    let mut method_init = Vec::with_capacity(count);
                    for _ in 0..count {
                        let pair = take(8)?;
                        method_init.push((u32_(&pair[..4]), u32_(&pair[4..])));
                    }
                    Record::Init {
                        channel,
                        method_init,
                    }
                }
                b'C' => {
                    let b = take(21)?;
                    Record::Call {
                        channel: u32_(&b[..4]),
                        subchannel: b[4],
                        method: u32_(&b[5..9]),
                        argument: u32_(&b[9..13]),
                        micros: u64_(&b[13..21]),
                    }
                }
                b'R' => {
                    let b = take(16)?;
                    let (channel, address, length) =
                        (u32_(&b[..4]), u64_(&b[4..12]), u32_(&b[12..16]) as usize);
                    Record::Read {
                        channel,
                        address,
                        data: take(length)?.to_vec(),
                    }
                }
                _ => return Err(invalid("unknown record")),
            };
            records.push(record);
        }
        Ok(Replay { records })
    }

    /// The calls of the capture in order, each with the reads made during
    /// it (which replay applies to memory first).
    pub fn calls(&self) -> Vec<ReplayCall<'_>> {
        let mut calls: Vec<ReplayCall<'_>> = Vec::new();
        let mut last: HashMap<u32, usize> = HashMap::new();
        for record in &self.records {
            match record {
                Record::Call {
                    channel,
                    subchannel,
                    method,
                    argument,
                    micros,
                } => {
                    last.insert(*channel, calls.len());
                    calls.push(ReplayCall {
                        channel: *channel,
                        subchannel: *subchannel,
                        method: *method,
                        argument: *argument,
                        micros: *micros,
                        reads: Vec::new(),
                    });
                }
                Record::Read {
                    channel,
                    address,
                    data,
                } => {
                    if let Some(&index) = last.get(channel) {
                        calls[index].reads.push((*address, data.as_slice()));
                    }
                }
                Record::Init { .. } => {}
            }
        }
        calls
    }

    /// The method-init list of `channel`.
    pub fn method_init(&self, channel: u32) -> Option<&[(u32, u32)]> {
        self.records.iter().find_map(|record| match record {
            Record::Init {
                channel: c,
                method_init,
            } if *c == channel => Some(method_init.as_slice()),
            _ => None,
        })
    }
}

/// One call to replay.
pub struct ReplayCall<'a> {
    pub channel: u32,
    pub subchannel: u8,
    pub method: u32,
    pub argument: u32,
    pub micros: u64,
    pub reads: Vec<(u64, &'a [u8])>,
}

/// Guest memory rebuilt from a capture: what was read and written so far.
#[derive(Default)]
pub struct SparseMemory(Mutex<Shadow>);

impl SparseMemory {
    pub fn apply(&self, address: u64, data: &[u8]) {
        self.0.lock().unwrap().store(address, data);
    }
}

impl GpuMemory for SparseMemory {
    fn read(&self, address: u64, buffer: &mut [u8]) -> Option<()> {
        self.0.lock().unwrap().load(address, buffer)
    }

    fn write(&self, address: u64, data: &[u8]) -> Option<()> {
        self.0.lock().unwrap().store(address, data);
        Some(())
    }
}

/// A channel of a replay, with the memory it sees.
pub struct ReplayChannel {
    pub channel: super::Channel,
    pub memory: SparseMemory,
}

/// Builds the engines of captured channel from its method-init list the
/// way the GM20B model does (`soc::gm20b::maxwell_engines`), with a timer
/// that reads the capture's time of the call being replayed.
pub fn replay_channel(
    method_init: &[(u32, u32)],
    clock: Arc<std::sync::atomic::AtomicU64>,
) -> Result<super::Channel, Unmodeled> {
    let mut channel = super::Channel::new();
    for &(index, value) in method_init {
        channel.set_initial_state(index as u16, index >> 16, value)?;
    }
    channel.set_timer(Box::new(move || clock.load(Ordering::Relaxed)));
    channel.set_syncpoint_increment(Box::new(|_| {}));
    Ok(channel)
}

/// Replays every call of `replay` in order; `after` sees each call with
/// its channel once it ran and how long it took. Stops at the first call
/// the engines refuse, returning its index.
pub fn run(
    replay: &Replay,
    mut after: impl FnMut(&ReplayCall<'_>, &mut ReplayChannel, std::time::Duration),
) -> Result<(), (usize, Unmodeled)> {
    let clock = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let mut channels: HashMap<u32, ReplayChannel> = HashMap::new();
    for (index, call) in replay.calls().iter().enumerate() {
        let state = match channels.entry(call.channel) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                let init = replay.method_init(call.channel).unwrap_or(&[]);
                let channel = replay_channel(init, Arc::clone(&clock)).map_err(|e| (index, e))?;
                entry.insert(ReplayChannel {
                    channel,
                    memory: SparseMemory::default(),
                })
            }
        };
        for &(address, data) in &call.reads {
            state.memory.apply(address, data);
        }
        clock.store(call.micros * 1000, Ordering::Relaxed);
        let start = Instant::now();
        let result = if call.subchannel == FLUSH {
            state.channel.flush(&state.memory)
        } else {
            state
                .channel
                .call(call.subchannel, call.method, call.argument, &state.memory)
        };
        result.map_err(|e| (index, e))?;
        after(call, state, start.elapsed());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shadow_spans_pages_and_tracks_known_bytes() {
        let mut shadow = Shadow::default();
        let data: Vec<u8> = (0..0x1800u32).map(|i| i as u8).collect();
        assert!(!shadow.matches(0xF00, &data));
        shadow.store(0xF00, &data);
        assert!(shadow.matches(0xF00, &data));
        assert!(shadow.matches(0x1000, &data[0x100..0x200]));
        let mut buffer = [0u8; 0x10];
        assert!(shadow.load(0xEF8, &mut buffer).is_none());
        shadow.load(0x2000, &mut buffer).unwrap();
        assert_eq!(buffer[0], 0x00);
        let mut changed = data.clone();
        changed[0x17FF] ^= 1;
        assert!(!shadow.matches(0xF00, &changed));
    }

    /// `MAXWELL_REPLAY=<capture> [MAXWELL_DUMP=<dir>] cargo test --release
    /// --lib maxwell::capture::tests::replay -- --ignored --nocapture`:
    /// replays a capture, prints where the time went by method and the
    /// time between 3D syncpoint increments per channel, and optionally
    /// writes render target 0 at each of them.
    #[test]
    #[ignore]
    fn replay() {
        let path = std::env::var("MAXWELL_REPLAY").expect("MAXWELL_REPLAY");
        let dump = std::env::var_os("MAXWELL_DUMP").map(std::path::PathBuf::from);
        let replay = Replay::load(Path::new(&path)).unwrap();
        let mut by_method: HashMap<(u8, u32), (u32, std::time::Duration)> = HashMap::new();
        let mut fences: HashMap<u32, Vec<std::time::Duration>> = HashMap::new();
        let mut since: HashMap<u32, std::time::Duration> = HashMap::new();
        let mut count = 0;
        let result = run(&replay, |call, state, took| {
            let key = if call.method >= 0xE00 {
                (call.subchannel, 0xE00 + (call.method - 0xE00) / 2 * 2)
            } else {
                (call.subchannel, call.method)
            };
            let entry = by_method.entry(key).or_default();
            entry.0 += 1;
            entry.1 += took;
            *since.entry(call.channel).or_default() += took;
            if call.subchannel == 0 && call.method == 0x2C8 / 4 {
                fences
                    .entry(call.channel)
                    .or_default()
                    .push(std::mem::take(since.get_mut(&call.channel).unwrap()));
                if let Some(dir) = &dump
                    && let Some(image) = state.channel.render_target(0, &state.memory)
                {
                    image
                        .write_png(&dir.join(format!("{count:04}-{}.png", call.channel)))
                        .unwrap();
                    count += 1;
                }
            }
        });
        let mut methods: Vec<_> = by_method.into_iter().collect();
        methods.sort_by_key(|(_, (_, time))| std::cmp::Reverse(*time));
        for ((subchannel, method), (calls, time)) in methods.iter().take(12) {
            println!(
                "subchannel {subchannel} method {:#06x}: {calls} calls, {time:?}",
                method * 4
            );
        }
        for (channel, times) in &fences {
            println!("channel {channel}: work between syncpoint increments {times:?}");
        }
        if let Err((index, error)) = result {
            panic!("call {index}: {}", error.0);
        }
    }
}
