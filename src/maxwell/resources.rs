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
}

fn read(memory: &dyn GpuMemory, address: u64, length: usize) -> Result<Vec<u8>, Unmodeled> {
    let mut bytes = vec![0; length];
    memory.read(address, &mut bytes).ok_or_else(|| {
        Unmodeled(format!(
            "read of {length:#x} bytes at unmapped GPU address {address:#x}"
        ))
    })?;
    Ok(bytes)
}

/// Decoded shader programs by address.
#[derive(Default)]
pub struct ProgramCache {
    entries: HashMap<u64, (Vec<u8>, Arc<Program>)>,
}

impl ProgramCache {
    /// The program whose header is at `address`, as memory holds it now.
    pub fn get(&mut self, memory: &dyn GpuMemory, address: u64) -> Result<Arc<Program>, Unmodeled> {
        if let Some((raw, program)) = self.entries.get(&address)
            && read(memory, address, raw.len())? == *raw
        {
            return Ok(program.clone());
        }
        let program = Program::decode(memory, address)?;
        let raw = read(memory, address, program.len())?;
        let program = Arc::new(program);
        self.entries.insert(address, (raw, program.clone()));
        Ok(program)
    }
}

/// A decoded texture, the raw bytes it was decoded from, and when they
/// last compared equal to memory.
type CachedTexture = (Vec<u8>, Arc<Texture>, Stamp);

/// Decoded textures by TIC contents, kept across draws.
#[derive(Default)]
pub struct TextureCache {
    entries: Mutex<HashMap<[u32; 8], CachedTexture>>,
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

    fn texture(&self, index: u32) -> Result<Arc<Texture>, Unmodeled> {
        if let Some(texture) = self.resolved.lock().unwrap().get(&index) {
            return Ok(texture.clone());
        }
        if index > self.pools.header_maximum {
            return Err(Unmodeled(format!(
                "texture: TIC index {index:#x} beyond the pool maximum {:#x}",
                self.pools.header_maximum
            )));
        }
        let words = self.words(self.pools.header_pool, index)?;
        let tic = Tic::parse(words)?;
        let (address, length) = Texture::footprint(&tic)?;
        let end = address.saturating_add(length);
        let fresh = self
            .cache
            .entries
            .lock()
            .unwrap()
            .get(&words)
            .filter(|(_, _, stamp)| self.writes.unchanged(*stamp, address, end))
            .map(|(_, texture, _)| texture.clone());
        let texture = match fresh {
            Some(texture) => texture,
            None => {
                let raw = read(self.memory, address, length as usize)?;
                let mut entries = self.cache.entries.lock().unwrap();
                match entries.get_mut(&words) {
                    Some((cached_raw, texture, stamp)) if *cached_raw == raw => {
                        *stamp = self.writes.stamp();
                        texture.clone()
                    }
                    _ => {
                        drop(entries);
                        let texture = Arc::new(Texture::decode(self.memory, &tic)?);
                        self.cache
                            .entries
                            .lock()
                            .unwrap()
                            .insert(words, (raw, texture.clone(), self.writes.stamp()));
                        texture
                    }
                }
            }
        };
        self.resolved.lock().unwrap().insert(index, texture.clone());
        Ok(texture)
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
        let tic = handle & 0xF_FFFF;
        let tsc = if self.pools.sampler_from_header {
            tic
        } else {
            handle >> 20
        };
        Ok((self.texture(tic)?, self.sampler(tsc)?))
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
    /// 19:0, TSC index in bits 31:20 [public: Ryujinx
    /// `Ryujinx.Graphics.Shader/TextureHandle.cs`
    /// `UnpackTextureId`/`UnpackSamplerId`]. Recently used handles are
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

    fn query(&self, handle: u32, query: Query) -> Result<[u32; 4], Unmodeled> {
        self.with(handle, |texture, _| texture.query(query))?
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
