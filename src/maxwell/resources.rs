//! What draws read from memory besides vertices: shader programs and
//! textures, cached by content.
//!
//! Every cache entry keeps the raw bytes it was built from. A program is
//! reused only after the bytes in memory compare equal, at every draw. A
//! texture is compared at most once per *epoch* of its channel ([`Writes`]):
//! once its bytes compared equal, it is reused without reading memory
//! again until the channel signals (a semaphore release, report or
//! syncpoint increment) or one of the channel's own engines writes memory
//! overlapping it. The hardware's own caches (`INVALIDATE_SHADER_CACHES`,
//! `INVALIDATE_TEXTURE_HEADER_CACHE`, ...) therefore have no counterpart
//! here, and their invalidation methods have no effect.
//!
//! Both caches are bounded ([`PROGRAM_BUDGET`], [`TEXTURE_BUDGET`]):
//! beyond its budget a cache drops the entries used least recently
//! (`maxwell::lru`), which are decoded again when next used.
//!
//! # The staleness an epoch allows
//!
//! Within an epoch, a write to a validated texture by anything but the
//! channel's own engines (the guest CPU, another channel) is not seen
//! until the next signal. No correct guest makes one: the texture was
//! validated because a draw of this channel read it, and memory a
//! submitted draw reads may only be overwritten once that draw is known
//! to be complete. NVN learns that only from a sync object, a semaphore
//! or syncpoint the channel releases after the draw (`nvnQueueFenceSync`,
//! `nvnSyncWait`, `nvnQueueFinish` [public: NVN API as documented by the
//! Switch homebrew headers]), and every such release starts a new epoch.
//! On hardware the same race reads stale texture cache lines.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::lru::Lru;
use super::sampling::{FetchRequest, GatherRequest, Query, SampleRequest, Target, TextureUnit};
use super::shader::Program;
use super::texture::{Texture, Tic, Tsc};
use super::{GpuMemory, Unmodeled};

/// The guest memory a channel's engines wrote in the current epoch: since
/// the channel's last signal ([`Writes::signal`]). Caches of guest memory
/// stamp what they last compared ([`Writes::stamp`]) and need not read
/// memory again while [`Writes::unchanged`] holds.
#[derive(Default)]
pub struct Writes(Mutex<WriteLog>);

#[derive(Default)]
struct WriteLog {
    epoch: u64,
    /// Written ranges, `start..end`, in order of writing.
    ranges: Vec<(u64, u64)>,
    /// Whether a stamp counts the last range: it may not grow then.
    stamped: bool,
}

/// When a copy of guest memory was last known to be equal to it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stamp {
    epoch: u64,
    writes: usize,
}

/// Ranges an epoch holds before it is ended early (everything is compared
/// again then, which is always allowed): keeps [`Writes::unchanged`]
/// short.
const MOST_RANGES: usize = 1024;

impl Writes {
    /// Notes a write of `length` bytes at `address`. Consecutive writes
    /// (constant buffer uploads, a surface stored band by band) extend one
    /// range.
    pub fn record(&self, address: u64, length: u64) {
        let mut log = self.0.lock().unwrap();
        let end = address.saturating_add(length);
        if !log.stamped
            && let Some(last) = log.ranges.last_mut()
            && address <= last.1
            && end >= last.0
        {
            *last = (last.0.min(address), last.1.max(end));
            return;
        }
        if log.ranges.len() == MOST_RANGES {
            log.epoch += 1;
            log.ranges.clear();
        }
        log.ranges.push((address, end));
        log.stamped = false;
    }

    /// The channel signaled: a new epoch starts, in which every cached
    /// copy is compared with memory again before it is used.
    pub fn signal(&self) {
        let mut log = self.0.lock().unwrap();
        log.epoch += 1;
        log.ranges.clear();
        log.stamped = false;
    }

    /// Now, for a copy that was just compared with (or written to) memory.
    pub fn stamp(&self) -> Stamp {
        let mut log = self.0.lock().unwrap();
        log.stamped = true;
        Stamp {
            epoch: log.epoch,
            writes: log.ranges.len(),
        }
    }

    /// Whether `start..end` may be taken as unchanged since `stamp`: same
    /// epoch, and no write of the channel's engines overlapped it since.
    pub fn unchanged(&self, stamp: Stamp, start: u64, end: u64) -> bool {
        let log = self.0.lock().unwrap();
        log.epoch == stamp.epoch
            && log.ranges[stamp.writes..]
                .iter()
                .all(|&(a, b)| b <= start || end <= a)
    }
}

/// Engine memory that notes every write in [`Writes`].
pub struct Tracked<'a> {
    pub memory: &'a dyn GpuMemory,
    pub writes: &'a Writes,
}

impl GpuMemory for Tracked<'_> {
    fn read(&self, address: u64, buffer: &mut [u8]) -> Option<()> {
        self.memory.read(address, buffer)
    }

    fn write(&self, address: u64, data: &[u8]) -> Option<()> {
        self.writes.record(address, data.len() as u64);
        self.memory.write(address, data)
    }

    fn probe(&self, address: u64, buffer: &mut [u8]) -> Option<()> {
        self.memory.probe(address, buffer)
    }
}

fn read(memory: &dyn GpuMemory, address: u64, length: usize) -> Result<Vec<u8>, Unmodeled> {
    let mut bytes = vec![0; length];
    memory
        .read(address, &mut bytes)
        .ok_or_else(|| unmapped(address, length))?;
    Ok(bytes)
}

fn unmapped(address: u64, length: usize) -> Unmodeled {
    Unmodeled(format!(
        "read of {length:#x} bytes at unmapped GPU address {address:#x}"
    ))
}

/// Whether memory at `address` holds `expected`, compared a piece at a
/// time without a copy of the whole; refuses as [`read`] of the whole
/// range would (pieces are read in order, so the first one that fails
/// holds the first unmapped byte).
fn equals(memory: &dyn GpuMemory, address: u64, expected: &[u8]) -> Result<bool, Unmodeled> {
    let mut buffer = [0u8; 4096];
    let mut at = address;
    for chunk in expected.chunks(buffer.len()) {
        let piece = &mut buffer[..chunk.len()];
        memory
            .read(at, piece)
            .ok_or_else(|| unmapped(address, expected.len()))?;
        if piece != chunk {
            return Ok(false);
        }
        at += chunk.len() as u64;
    }
    Ok(true)
}

/// `None` if memory at `address` holds `expected`, else the `length` bytes
/// it holds, read once: compared a piece at a time while equal, the rest
/// read after the first difference. Refuses as [`read`] would.
fn read_unless_equal(
    memory: &dyn GpuMemory,
    address: u64,
    length: usize,
    expected: Option<&[u8]>,
) -> Result<Option<Vec<u8>>, Unmodeled> {
    let Some(expected) = expected.filter(|expected| expected.len() == length) else {
        return read(memory, address, length).map(Some);
    };
    let mut buffer = [0u8; 4096];
    let mut at = 0;
    for chunk in expected.chunks(buffer.len()) {
        let piece = &mut buffer[..chunk.len()];
        memory
            .read(address + at as u64, piece)
            .ok_or_else(|| unmapped(address, length))?;
        if piece != chunk {
            let mut bytes = Vec::with_capacity(length);
            bytes.extend_from_slice(&expected[..at]);
            bytes.extend_from_slice(piece);
            let done = bytes.len();
            bytes.resize(length, 0);
            if done < length {
                memory
                    .read(address + done as u64, &mut bytes[done..])
                    .ok_or_else(|| unmapped(address, length))?;
            }
            return Ok(Some(bytes));
        }
        at += chunk.len();
    }
    Ok(None)
}

/// Host memory the decoded programs of one channel may take, with the
/// bytes they were decoded from \[assumption: sized so that every
/// program a session uses at once stays (a program is at most 64 KiB of
/// code, `shader::MAX_PROGRAM`, decoded to a few times that); a dropped
/// program is decoded again when drawn with\].
pub const PROGRAM_BUDGET: usize = 64 << 20;

/// Host memory the decoded textures of one channel may take, with the
/// bytes they were decoded from \[assumption: decoded texels are four
/// `f32` each, so a 1280x720 texture takes 14 MiB; this holds a few
/// dozen screen-sized images, more than a frame of the HOME menu or the
/// browser samples; a dropped texture is decoded again when sampled\].
pub const TEXTURE_BUDGET: usize = 512 << 20;

/// Host memory the textures of one channel kept as their guest bytes
/// may take ([`RawTexture`]) \[assumption: sized so that what an
/// application samples in a frame stays; a dropped texture is read
/// again when sampled\].
pub const RAW_TEXTURE_BUDGET: usize = 768 << 20;

/// A texture as guest memory holds it: its header and the bytes of its
/// footprint, for a renderer that samples the format itself.
pub struct RawTexture {
    pub tic: Tic,
    pub bytes: Box<[u8]>,
}

/// Decoded shader programs by address, the least recently used dropped
/// beyond [`PROGRAM_BUDGET`].
pub struct ProgramCache {
    entries: Lru<u64, (Vec<u8>, Arc<Program>)>,
}

impl Default for ProgramCache {
    fn default() -> Self {
        Self::with_budget(PROGRAM_BUDGET)
    }
}

impl ProgramCache {
    pub fn with_budget(budget: usize) -> Self {
        ProgramCache {
            entries: Lru::new(budget),
        }
    }

    /// The program whose header is at `address`, as memory holds it now.
    pub fn get(&mut self, memory: &dyn GpuMemory, address: u64) -> Result<Arc<Program>, Unmodeled> {
        if let Some((raw, program)) = self.entries.get(&address)
            && equals(memory, address, raw)?
        {
            return Ok(program.clone());
        }
        let program = Program::decode(memory, address)?;
        let raw = read(memory, address, program.len())?;
        let program = Arc::new(program);
        let cost = raw.len() + program.bytes();
        self.entries.insert(address, (raw, program.clone()), cost);
        Ok(program)
    }

    /// Programs held, and the host memory they take.
    pub fn usage(&self) -> (usize, usize) {
        (self.entries.len(), self.entries.cost())
    }
}

/// A texture decoded from `raw`, the bytes of its footprint.
struct Decoded {
    raw: Box<[u8]>,
    texture: Arc<Texture>,
    /// Tells this decode from any other kept under the same key before or
    /// after it.
    serial: u64,
}

/// The decoded texture a TIC's memory last compared equal to: the one
/// under `(words, hash)` with serial `serial`, compared at `stamp`.
#[derive(Clone, Copy)]
struct Binding {
    hash: u64,
    serial: u64,
    stamp: Stamp,
}

/// TICs whose binding is kept \[assumption: far more than a frame binds;
/// a dropped binding costs a hash of the memory at the next use\].
const BINDING_LIMIT: usize = 1 << 16;

struct Entries {
    bindings: Lru<[u32; 8], Binding>,
    /// Decoded textures by TIC contents and [`content_hash`] of their
    /// bytes.
    decoded: Lru<([u32; 8], u64), Arc<Decoded>>,
    serials: u64,
    /// Textures kept as their bytes, by TIC contents, with when memory
    /// was last known to hold those bytes.
    raw: Lru<[u32; 8], (Arc<RawTexture>, Stamp)>,
}

/// Decoded textures by TIC contents and memory contents, kept across
/// draws; the least recently used are dropped beyond [`TEXTURE_BUDGET`].
/// A draw keeps the textures it resolved ([`DrawTextures`]), so none is
/// dropped while in use.
///
/// Each TIC is bound to the decode its memory last compared equal to.
/// When the memory changed, the decode of the new contents is looked up
/// by their hash and taken only if its bytes are equal (a texture that a
/// guest alternates between a few contents, as the Mii editor's render
/// targets do, is decoded once per contents); otherwise the new contents
/// are decoded.
pub struct TextureCache {
    entries: Mutex<Entries>,
}

impl Default for TextureCache {
    fn default() -> Self {
        Self::with_budget(TEXTURE_BUDGET)
    }
}

impl TextureCache {
    pub fn with_budget(budget: usize) -> Self {
        TextureCache {
            entries: Mutex::new(Entries {
                bindings: Lru::new(BINDING_LIMIT),
                decoded: Lru::new(budget),
                serials: 0,
                raw: Lru::new(RAW_TEXTURE_BUDGET),
            }),
        }
    }

    /// Textures held, and the host memory they take.
    pub fn usage(&self) -> (usize, usize) {
        let entries = self.entries.lock().unwrap();
        (entries.decoded.len(), entries.decoded.cost())
    }

    /// The texture `words` describe whose memory holds `raw`, now bound to
    /// it: kept if decoded before, decoded otherwise.
    fn contents(
        &self,
        words: [u32; 8],
        tic: &Tic,
        raw: Vec<u8>,
        writes: &Writes,
    ) -> Result<Arc<Texture>, Unmodeled> {
        let hash = content_hash(&raw);
        let kept = self
            .entries
            .lock()
            .unwrap()
            .decoded
            .get(&(words, hash))
            .cloned();
        let decoded = match kept {
            Some(decoded) if *decoded.raw == *raw => decoded,
            _ => {
                let texture = Arc::new(Texture::decode(&raw, tic)?);
                let cost = raw.len() + texture.bytes();
                let mut entries = self.entries.lock().unwrap();
                entries.serials += 1;
                let decoded = Arc::new(Decoded {
                    raw: raw.into(),
                    texture,
                    serial: entries.serials,
                });
                let dropped = entries.decoded.insert((words, hash), decoded.clone(), cost);
                drop(entries);
                // Dropped textures are freed outside the lock.
                drop(dropped);
                decoded
            }
        };
        let binding = Binding {
            hash,
            serial: decoded.serial,
            stamp: writes.stamp(),
        };
        self.entries
            .lock()
            .unwrap()
            .bindings
            .insert(words, binding, 1);
        Ok(decoded.texture.clone())
    }
}

/// A hash of `bytes` for [`TextureCache`]'s lookup by contents, which
/// compares the bytes too: fast rather than strong (four independent
/// multiply-rotate lanes, eight bytes each).
fn content_hash(bytes: &[u8]) -> u64 {
    const K: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut lanes = [1u64, 2, 3, 4];
    let (chunks, remainder) = bytes.as_chunks::<32>();
    for chunk in chunks {
        let (words, _) = chunk.as_chunks::<8>();
        for (lane, word) in lanes.iter_mut().zip(words) {
            *lane = (*lane ^ u64::from_le_bytes(*word))
                .wrapping_mul(K)
                .rotate_left(29);
        }
    }
    let mut hash = bytes.len() as u64;
    for lane in lanes {
        hash = (hash ^ lane).wrapping_mul(K).rotate_left(31);
    }
    for &byte in remainder {
        hash = (hash ^ u64::from(byte)).wrapping_mul(K).rotate_left(23);
    }
    hash ^ (hash >> 32)
}

/// Where a draw's texture handles point.
#[derive(Debug, Clone, Copy)]
pub struct Pools {
    pub header_pool: u64,
    /// Largest valid TIC index (`SET_TEX_HEADER_POOL_C`).
    pub header_maximum: u32,
    pub sampler_pool: u64,
    /// Largest valid TSC index (`SET_TEX_SAMPLER_POOL_C`).
    pub sampler_maximum: u32,
    /// `SET_SAMPLER_BINDING` `VIA_HEADER_BINDING`: the sampler index is
    /// the texture index.
    pub sampler_from_header: bool,
}

/// The textures of one draw, shared by its worker threads.
pub struct DrawTextures<'a> {
    memory: &'a dyn GpuMemory,
    pools: Pools,
    cache: &'a TextureCache,
    writes: &'a Writes,
    /// Textures resolved during this draw, by TIC index.
    resolved: Mutex<HashMap<u32, Arc<Texture>>>,
}

impl<'a> DrawTextures<'a> {
    pub fn new(
        memory: &'a dyn GpuMemory,
        pools: Pools,
        cache: &'a TextureCache,
        writes: &'a Writes,
    ) -> Self {
        Self {
            memory,
            pools,
            cache,
            writes,
            resolved: Default::default(),
        }
    }

    fn words(&self, pool: u64, index: u32) -> Result<[u32; 8], Unmodeled> {
        let bytes = read(self.memory, pool + u64::from(index) * 32, 32)?;
        Ok(std::array::from_fn(|i| {
            u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().expect("4 bytes"))
        }))
    }

    /// The header at TIC index `index`.
    fn tic(&self, index: u32) -> Result<Tic, Unmodeled> {
        if index > self.pools.header_maximum {
            return Err(Unmodeled(format!(
                "texture: TIC index {index:#x} beyond the pool maximum {:#x}",
                self.pools.header_maximum
            )));
        }
        Tic::parse(self.words(self.pools.header_pool, index)?)
    }

    fn texture(&self, index: u32) -> Result<Arc<Texture>, Unmodeled> {
        if let Some(texture) = self.resolved.lock().unwrap().get(&index) {
            return Ok(texture.clone());
        }
        let tic = self.tic(index)?;
        let words = tic.words;
        let (address, length) = Texture::footprint(&tic)?;
        let end = address.saturating_add(length);
        // The decode this TIC's memory last compared equal to, if kept.
        let bound = {
            let mut entries = self.cache.entries.lock().unwrap();
            let Entries {
                bindings, decoded, ..
            } = &mut *entries;
            bindings.get(&words).copied().and_then(|binding| {
                decoded
                    .get(&(words, binding.hash))
                    .filter(|decoded| decoded.serial == binding.serial)
                    .map(|decoded| (binding, decoded.clone()))
            })
        };
        let texture = match bound {
            Some((binding, decoded)) if self.writes.unchanged(binding.stamp, address, end) => {
                decoded.texture.clone()
            }
            bound => {
                let expected = bound.as_ref().map(|(_, decoded)| &*decoded.raw);
                match read_unless_equal(self.memory, address, length as usize, expected)? {
                    None => {
                        let (binding, decoded) = bound.expect("compared with a kept decode");
                        let stamp = self.writes.stamp();
                        if let Some(kept) =
                            self.cache.entries.lock().unwrap().bindings.get_mut(&words)
                            && kept.serial == binding.serial
                        {
                            kept.stamp = stamp;
                        }
                        decoded.texture.clone()
                    }
                    Some(raw) => self.cache.contents(words, &tic, raw, self.writes)?,
                }
            }
        };
        self.resolved.lock().unwrap().insert(index, texture.clone());
        Ok(texture)
    }

    /// The texture of a bound-texture handle as its bytes: kept while
    /// memory holds them (taken as unchanged while no write of the
    /// channel touched them in this epoch, compared otherwise), read
    /// anew when it does not.
    pub fn raw(&self, handle: u32) -> Result<Arc<RawTexture>, Unmodeled> {
        let tic = self.tic(self.indices(handle).0)?;
        let words = tic.words;
        let (address, length) = Texture::footprint(&tic)?;
        let end = address.saturating_add(length);
        let kept = self.cache.entries.lock().unwrap().raw.get(&words).cloned();
        if let Some((raw, stamp)) = &kept
            && self.writes.unchanged(*stamp, address, end)
        {
            return Ok(raw.clone());
        }
        let expected = kept.as_ref().map(|(raw, _)| &*raw.bytes);
        let raw = match read_unless_equal(self.memory, address, length as usize, expected)? {
            None => kept.expect("compared with kept bytes").0,
            Some(bytes) => Arc::new(RawTexture {
                tic,
                bytes: bytes.into(),
            }),
        };
        let stamp = self.writes.stamp();
        let dropped = self.cache.entries.lock().unwrap().raw.insert(
            words,
            (raw.clone(), stamp),
            raw.bytes.len(),
        );
        // Dropped textures are freed outside the lock.
        drop(dropped);
        Ok(raw)
    }

    fn sampler(&self, index: u32) -> Result<Tsc, Unmodeled> {
        if index > self.pools.sampler_maximum {
            return Err(Unmodeled(format!(
                "texture: TSC index {index:#x} beyond the pool maximum {:#x}",
                self.pools.sampler_maximum
            )));
        }
        Tsc::parse(self.words(self.pools.sampler_pool, index)?)
    }

    /// The texture and sampler of a bound-texture handle: TIC index in
    /// bits 19:0, TSC index in bits 31:20 (see [`ThreadTextures`]).
    pub fn resolve(&self, handle: u32) -> Result<(Arc<Texture>, Tsc), Unmodeled> {
        let (tic, tsc) = self.indices(handle);
        Ok((self.texture(tic)?, self.sampler(tsc)?))
    }

    /// The TIC and TSC indices of a bound-texture handle.
    fn indices(&self, handle: u32) -> (u32, u32) {
        let tic = handle & 0xF_FFFF;
        let tsc = if self.pools.sampler_from_header {
            tic
        } else {
            handle >> 20
        };
        (tic, tsc)
    }

    /// The texture header and sampler of a bound-texture handle, without
    /// the texture's contents (no read of its memory): what
    /// [`resolve`](Self::resolve) decodes the texture by. Refuses as
    /// `resolve` does for a handle that names no header or sampler.
    pub fn header(&self, handle: u32) -> Result<(Tic, Tsc), Unmodeled> {
        let (tic, tsc) = self.indices(handle);
        Ok((self.tic(tic)?, self.sampler(tsc)?))
    }

    /// A per-thread view with its own small lookup cache.
    pub fn unit(&'a self) -> ThreadTextures<'a> {
        ThreadTextures {
            draw: self,
            recent: RefCell::new(Vec::new()),
        }
    }
}

/// The texture unit one worker thread hands to the shader interpreter.
pub struct ThreadTextures<'a> {
    draw: &'a DrawTextures<'a>,
    recent: RefCell<Vec<(u32, Arc<Texture>, Tsc)>>,
}

impl ThreadTextures<'_> {
    /// Runs `f` on the texture and sampler of a handle: TIC index in bits
    /// 19:0, TSC index in bits 31:20. Recently used handles are
    /// served from this thread's list without touching shared state.
    fn with<R>(&self, handle: u32, f: impl FnOnce(&Texture, &Tsc) -> R) -> Result<R, Unmodeled> {
        {
            let recent = self.recent.borrow();
            if let Some((_, texture, sampler)) = recent.iter().find(|(h, _, _)| *h == handle) {
                return Ok(f(texture, sampler));
            }
        }
        let tic = handle & 0xF_FFFF;
        let tsc = if self.draw.pools.sampler_from_header {
            tic
        } else {
            handle >> 20
        };
        let texture = self.draw.texture(tic)?;
        let sampler = self.draw.sampler(tsc)?;
        let result = f(&texture, &sampler);
        let mut recent = self.recent.borrow_mut();
        if recent.len() >= 16 {
            recent.remove(0);
        }
        recent.push((handle, texture, sampler));
        Ok(result)
    }
}

impl TextureUnit for ThreadTextures<'_> {
    fn sample(&self, handle: u32, request: &SampleRequest) -> Result<[f32; 4], Unmodeled> {
        self.with(handle, |texture, sampler| texture.sample(sampler, request))?
    }

    fn fetch(&self, handle: u32, request: &FetchRequest) -> Result<[f32; 4], Unmodeled> {
        self.with(handle, |texture, _| texture.fetch(request))?
    }

    fn gather(&self, handle: u32, request: &GatherRequest) -> Result<[f32; 4], Unmodeled> {
        self.with(handle, |texture, sampler| texture.gather(sampler, request))?
    }

    /// From the header: a query reads nothing of the texture.
    fn query(&self, handle: u32, query: Query) -> Result<[u32; 4], Unmodeled> {
        let (tic, _) = self.draw.header(handle)?;
        super::texture::query_header(&tic, query)
    }

    fn lod(
        &self,
        handle: u32,
        target: Target,
        ddx: [f32; 3],
        ddy: [f32; 3],
    ) -> Result<[f32; 2], Unmodeled> {
        self.with(handle, |texture, sampler| {
            texture.lod(sampler, target, ddx, ddy)
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{
        Decoded, DrawTextures, Pools, ProgramCache, Texture, TextureCache, Tracked, Writes,
        content_hash, read,
    };
    use crate::maxwell::GpuMemory;
    use crate::maxwell::tests::Ram;

    /// A pixel program at `address` whose one instruction moves `value`
    /// into R0 (layout as in `shader::tests`), then the end group.
    fn put_program(ram: &Ram, address: u64, value: u32) {
        let mut bytes = vec![0; 0x50];
        bytes[..4].copy_from_slice(&0x1402u32.to_le_bytes());
        bytes[0x48] = 15;
        bytes.extend_from_slice(&0u64.to_le_bytes());
        let mov = 0x010000000007f000u64 | (u64::from(value) << 20);
        for word in [mov, 0x50b0000000070f00, 0x50b0000000070f00] {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        bytes.extend_from_slice(&[0; 32]);
        ram.put(address, &bytes);
    }

    #[test]
    fn program_cache_drops_least_recently_used_and_decodes_again() {
        let ram = Ram::new(0x10000);
        for i in 0..8u32 {
            put_program(&ram, u64::from(i) * 0x1000, i + 1);
        }
        let mut sizing = ProgramCache::default();
        sizing.get(&ram, 0).unwrap();
        let one = sizing.usage().1;
        // Room for four programs.
        let mut cache = ProgramCache::with_budget(4 * one);
        let mut held = Vec::new();
        for i in 0..8u64 {
            held.push(cache.get(&ram, i * 0x1000).unwrap());
            let (count, cost) = cache.usage();
            assert!(
                cost <= 4 * one && count <= 4,
                "{count} programs, {cost} bytes"
            );
        }
        // The most recent stays and is the same program.
        assert!(Arc::ptr_eq(&cache.get(&ram, 7 * 0x1000).unwrap(), &held[7]));
        // The first was dropped: decoded again, equal in content.
        let again = cache.get(&ram, 0).unwrap();
        assert!(!Arc::ptr_eq(&again, &held[0]));
        assert_eq!(
            again.mnemonics().collect::<Vec<_>>(),
            held[0].mnemonics().collect::<Vec<_>>()
        );
    }

    /// A 2x2 A8B8G8R8 block-linear TIC (as in `texture::tests`) at
    /// `address`.
    fn tic(address: u32) -> [u32; 8] {
        [
            8 | (2 << 7)
                | (2 << 10)
                | (2 << 13)
                | (2 << 16)
                | (2 << 19)
                | (3 << 22)
                | (4 << 25)
                | (5 << 28),
            address,
            3 << 21,
            0,
            1 | (1 << 23) | (7 << 29),
            1 | (1 << 31),
            0,
            0,
        ]
    }

    #[test]
    fn texture_cache_stays_within_budget_and_redecodes() {
        const HEADERS: u64 = 0;
        const SAMPLERS: u64 = 0x800;
        let ram = Ram::new(0x20000);
        let sampler: [u32; 3] = [2 | (2 << 3) | (2 << 6), 2 | (2 << 4) | (1 << 6), 15 << 20];
        for (i, word) in sampler.iter().enumerate() {
            ram.put(SAMPLERS + i as u64 * 4, &word.to_le_bytes());
        }
        for i in 0..8u32 {
            let address = 0x10000 + i * 0x1000;
            for (k, word) in tic(address).iter().enumerate() {
                ram.put(
                    HEADERS + u64::from(i) * 32 + k as u64 * 4,
                    &word.to_le_bytes(),
                );
            }
            ram.put(u64::from(address), &[i as u8 * 30; 512]);
        }
        let pools = Pools {
            header_pool: HEADERS,
            header_maximum: 7,
            sampler_pool: SAMPLERS,
            sampler_maximum: 0,
            sampler_from_header: false,
        };
        let writes = Writes::default();
        let sizing = TextureCache::default();
        DrawTextures::new(&ram, pools, &sizing, &writes)
            .resolve(0)
            .unwrap();
        let one = sizing.usage().1;
        let cache = TextureCache::with_budget(3 * one);
        // One draw per texture, then the first again.
        for i in (0..8u32).chain([0]) {
            let draw = DrawTextures::new(&ram, pools, &cache, &writes);
            let (texture, _) = draw.resolve(i).unwrap();
            let value = f32::from(i as u8 * 30) / 255.;
            assert_eq!(texture.level(0).unwrap().3[0], [value; 4], "texture {i}");
            let (count, cost) = cache.usage();
            assert!(
                cost <= 3 * one && count <= 3,
                "{count} textures, {cost} bytes"
            );
        }
    }

    /// Every texture the cache hands out equals a decode of memory as it
    /// stands: a texture whose memory changed is decoded again, contents
    /// seen before are taken from the cache, and a kept decode whose hash
    /// matches but whose bytes differ is not taken.
    #[test]
    fn texture_cache_follows_memory_contents() {
        const SAMPLERS: u64 = 0x800;
        const TEXELS: u64 = 0x10000;
        let ram = Ram::new(0x20000);
        let sampler: [u32; 3] = [2 | (2 << 3) | (2 << 6), 2 | (2 << 4) | (1 << 6), 15 << 20];
        for (i, word) in sampler.iter().enumerate() {
            ram.put(SAMPLERS + i as u64 * 4, &word.to_le_bytes());
        }
        let words = tic(TEXELS as u32);
        for (k, word) in words.iter().enumerate() {
            ram.put(k as u64 * 4, &word.to_le_bytes());
        }
        let pools = Pools {
            header_pool: 0,
            header_maximum: 0,
            sampler_pool: SAMPLERS,
            sampler_maximum: 0,
            sampler_from_header: false,
        };
        let writes = Writes::default();
        let cache = TextureCache::default();
        let resolve = || {
            let texture = DrawTextures::new(&ram, pools, &cache, &writes)
                .resolve(0)
                .unwrap()
                .0;
            // Equal, bit for bit, to a decode of memory now.
            let raw = read(&ram, TEXELS, 512).unwrap();
            let fresh = Texture::decode(&raw, texture.tic()).unwrap();
            let (ours, theirs) = (texture.level(0).unwrap().3, fresh.level(0).unwrap().3);
            assert_eq!(
                ours.iter()
                    .flatten()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                theirs
                    .iter()
                    .flatten()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
            texture
        };
        let a: Vec<u8> = (0..512).map(|i| (i * 7) as u8).collect();
        let mut b = a.clone();
        b[0] ^= 0xff;
        ram.put(TEXELS, &a);
        let first = resolve();
        // Unchanged within the epoch, and compared equal in the next.
        assert!(Arc::ptr_eq(&resolve(), &first));
        writes.signal();
        assert!(Arc::ptr_eq(&resolve(), &first));
        // Changed: decoded again.
        ram.put(TEXELS, &b);
        writes.signal();
        let second = resolve();
        assert!(!Arc::ptr_eq(&second, &first));
        assert_ne!(second.level(0).unwrap().3[0], first.level(0).unwrap().3[0]);
        // The first contents again (an engine's write this time): the
        // first decode, not a new one.
        Tracked {
            memory: &ram,
            writes: &writes,
        }
        .write(TEXELS, &a[..4])
        .unwrap();
        assert!(Arc::ptr_eq(&resolve(), &first));
        assert_eq!(cache.usage().0, 2);
        // A kept decode under the hash of new contents, of other bytes (as
        // a hash collision would leave): decoded again all the same.
        let mut c = a.clone();
        c[1] ^= 0xff;
        {
            let mut entries = cache.entries.lock().unwrap();
            let impostor = Arc::new(Decoded {
                raw: a.clone().into(),
                texture: first.clone(),
                serial: u64::MAX,
            });
            entries
                .decoded
                .insert((words, content_hash(&c)), impostor, 1);
        }
        ram.put(TEXELS, &c);
        writes.signal();
        let third = resolve();
        assert!(!Arc::ptr_eq(&third, &first) && !Arc::ptr_eq(&third, &second));
    }
}
