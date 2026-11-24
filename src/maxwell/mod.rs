//! The engines of a GM20B (Maxwell) GPU channel, rendered in software.
//!
//! The GPU front end (FIFO, pushbuffer decoding, host methods) lives with
//! the GM20B device model; it hands every engine method of a channel to
//! [`Channel`] through [`Engines::call`]. A channel binds objects of the
//! classes the firmware's driver accepts (nvservices
//! `NV_IsSupportedGm20bClass_876C0`: 0x902D, 0xA0B5, 0xB0B5, 0xB197,
//! 0xB1C0; the characteristics also report 0xA140) to its eight
//! subchannels with method 0, and executes:
//!
//! * [`three_d`]: the Maxwell B 3D class with its [`pipeline`] (vertex
//!   fetch, [`shader`] interpretation, clipping, [`raster`]ization in 2x2
//!   quads, depth test, [`blend`]ing) into render targets in guest memory
//!   ([`surface`]), reading [`texture`]s through [`sampling`];
//! * [`mme`]: the macro method expander the driver programs through the
//!   3D class;
//! * [`two_d`]: Fermi 2D blits; [`copy`]: the DMA copy engine;
//!   [`inline`]: inline-to-memory, shared by 3D, compute and 0xA140;
//! * [`compute`]: the compute class: kernel launches run by the
//!   [`shader`] interpreter, inline-to-memory and semaphores.
//!
//! Everything runs synchronously inside [`Engines::call`]: when a method
//! returns, its effects are in memory. Whatever is not modeled returns
//! [`Unmodeled`] naming class, method and value; it is never skipped.
//!
//! Sources: the method interface is NVIDIA's own [public: open-gpu-doc
//! `classes/`]; behaviors the class headers do not define follow the
//! Direct3D/OpenGL
//! rules the hardware implements, and are labelled so where they are
//! used. Values pinned by the firmware are cited with program and address.

pub mod blend;
pub mod capture;
pub mod compute;
pub mod copy;
pub mod inline;
mod lru;
pub mod metal;
pub mod mme;
pub mod multisample;
pub mod pipeline;
pub mod raster;
pub mod resources;
pub mod sampling;
pub mod shader;
pub mod surface;
pub mod texture;
pub mod three_d;
pub mod two_d;

#[cfg(test)]
mod tests;

use serde::{Deserialize, Serialize};

use crate::snapshot::{SnapshotError, decode, encode};

/// Memory as a GPU channel addresses it: its page tables applied.
///
/// A failed [`read`](Self::read) or [`write`](Self::write) may leave its
/// reason behind for the refusal it causes (the pushbuffer executor's
/// `ChannelMemory` does). A memory that wraps another forwards all three
/// methods to it.
pub trait GpuMemory: Sync {
    fn read(&self, address: u64, buffer: &mut [u8]) -> Option<()>;
    fn write(&self, address: u64, data: &[u8]) -> Option<()>;
    /// A read whose failure the caller answers itself (it takes what is
    /// mapped, or rechecks a cache): no refusal follows from it, so it
    /// leaves no reason behind for an unrelated later one to carry.
    fn probe(&self, address: u64, buffer: &mut [u8]) -> Option<()> {
        self.read(address, buffer)
    }
    /// Whether the bytes at `address` are `expected` (a [`read`]
    /// compared): in place where the memory allows, with no copy.
    ///
    /// [`read`]: GpuMemory::read
    fn equals(&self, address: u64, expected: &[u8]) -> Option<bool> {
        let mut held = vec![0; expected.len()];
        self.read(address, &mut held)?;
        Some(held == expected)
    }
}

/// What a method did not model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unmodeled(pub String);

/// The engines of one channel: objects are bound to subchannels 0-7 by
/// method 0 (class ID in the low 16 bits of its argument), which the
/// pushbuffer executor passes through like any other method.
pub trait Engines: Send {
    /// One method call: `method` is the method's word index (register
    /// offset / 4) as it stands in the pushbuffer header, on `subchannel`.
    fn call(
        &mut self,
        subchannel: u8,
        method: u32,
        argument: u32,
        memory: &dyn GpuMemory,
    ) -> Result<(), Unmodeled>;

    /// Makes everything earlier calls wrote visible in guest memory, as
    /// the channel is about to signal. Engines that write memory as they
    /// go have nothing to do.
    fn flush(&mut self, _memory: &dyn GpuMemory) -> Result<(), Unmodeled> {
        Ok(())
    }

    /// The pushbuffer executor's own semaphore release or syncpoint
    /// increment, the host work that lets other agents observe the
    /// channel's progress: made once everything earlier calls did is
    /// visible in guest memory. Returned to be made now, or `None` when
    /// the engines make it themselves once their work before it is done
    /// ([`idle`](Self::idle) waits for that).
    fn release(
        &mut self,
        memory: &dyn GpuMemory,
        release: Release,
    ) -> Result<Option<Release>, Unmodeled> {
        self.flush(memory)?;
        Ok(Some(release))
    }

    /// The channel has nothing to run (it is idle, or waits for a
    /// semaphore or syncpoint): the releases the engines hold back are
    /// made, once the work before each is done.
    fn idle(&mut self, _memory: &dyn GpuMemory) -> Result<(), Unmodeled> {
        Ok(())
    }

    /// The channel synchronized with another agent (it fetched work
    /// submitted after what it had, or a semaphore acquire or syncpoint
    /// wait let it go on): what others wrote to memory before is to be
    /// seen from here on (`resources::Writes::synchronize`).
    fn synchronize(&mut self, _memory: &dyn GpuMemory) -> Result<(), Unmodeled> {
        Ok(())
    }

    /// Makes guest memory hold all of the channel's work, including what
    /// [`flush`](Self::flush) may leave on the host for the guest has no
    /// use reading it (the Metal renderer's targets held on demand, see
    /// [`metal`]): before a machine snapshot.
    fn settle(&mut self, memory: &dyn GpuMemory) -> Result<(), Unmodeled> {
        self.flush(memory)
    }

    /// The engines' state for a machine snapshot: everything that decides
    /// what later methods do (register files, shadow RAM, bindings, macro
    /// code, transfers and macro calls in flight), never a cache of guest
    /// memory or anything held on the host GPU. Taken after
    /// [`settle`](Self::settle), so nothing rendered is left only on the
    /// host.
    fn save(&self) -> Result<Vec<u8>, SnapshotError>;

    /// Takes back what [`save`](Self::save) returned, into engines made for
    /// the channel as for a new one (with the same hooks); caches start
    /// empty.
    fn restore(&mut self, state: &[u8]) -> Result<(), SnapshotError>;
}

/// Connections to the rest of the GPU a channel's engines need for some
/// methods; each method that uses one refuses to run without it.
#[derive(Default)]
pub struct Hooks {
    /// GPU timer (PTIMER) in nanoseconds, for the timestamp of four-word
    /// semaphore releases.
    pub timer: Option<Box<dyn Fn() -> u64 + Send>>,
    /// Increments a host1x syncpoint by one (3D `INCREMENT_SYNC_POINT`).
    pub syncpoint_increment: Option<std::sync::Arc<dyn Fn(u32) + Send + Sync>>,
    /// The Metal renderer of the channel, if it renders with Metal: every
    /// completion the channel signals (semaphore release, report,
    /// syncpoint increment) is made once its pending work is back in
    /// guest memory ([`metal::Renderer::release`]), but for the targets it
    /// holds on demand (depth buffers and multisampled targets, see
    /// [`metal`]).
    pub metal: Option<std::sync::Arc<metal::Renderer>>,
    /// What the engines wrote since the channel last synchronized.
    pub writes: resources::Writes,
}

/// A completion a channel signals: what others observe of its progress.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Release {
    /// Bytes written at a GPU virtual address (a semaphore release or a
    /// report).
    Write { address: u64, bytes: Vec<u8> },
    /// A host1x syncpoint incremented by one.
    Increment(u32),
}

impl Hooks {
    /// The channel is about to signal (release a semaphore or report,
    /// increment a syncpoint): makes all rendering so far visible in guest
    /// memory.
    pub fn signal(&self, memory: &dyn GpuMemory) -> Result<(), Unmodeled> {
        if let Some(renderer) = &self.metal {
            renderer.flush(memory, &self.writes)?;
        }
        Ok(())
    }

    /// Makes `release` once all rendering so far is visible in guest
    /// memory: now, or when the Metal renderer has done the work before
    /// it.
    pub fn release(&self, memory: &dyn GpuMemory, release: Release) -> Result<(), Unmodeled> {
        let now = match &self.metal {
            Some(renderer) => renderer.release(memory, &self.writes, release)?,
            None => Some(release),
        };
        match now {
            Some(release) => self.make(memory, release),
            None => Ok(()),
        }
    }

    fn make(&self, memory: &dyn GpuMemory, release: Release) -> Result<(), Unmodeled> {
        match release {
            Release::Write { address, bytes } => memory.write(address, &bytes).ok_or_else(|| {
                Unmodeled(format!(
                    "semaphore write to unmapped GPU address {address:#x}"
                ))
            }),
            Release::Increment(index) => {
                let increment = self.syncpoint_increment.as_ref().ok_or_else(|| {
                    Unmodeled("syncpoint increment without host1x connected".into())
                })?;
                increment(index);
                Ok(())
            }
        }
    }
}

/// Which renderer new channels draw with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RendererKind {
    /// The software renderer: the reference.
    Software,
    /// Metal on the host GPU ([`metal`]), falling back to the software
    /// renderer per draw for what it cannot express.
    Metal,
}

static RENDERER: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Selects the renderer of channels created from now on. Metal needs a
/// Metal device ([`metal::available`]).
pub fn select_renderer(kind: RendererKind) {
    RENDERER.store(
        kind == RendererKind::Metal,
        std::sync::atomic::Ordering::Relaxed,
    );
}

/// Class IDs.
pub const THREE_D: u16 = 0xB197;
pub const COMPUTE: u16 = 0xB1C0;
pub const TWO_D: u16 = 0x902D;
pub const COPY: u16 = 0xB0B5;
pub const COPY_A: u16 = 0xA0B5;
pub const INLINE_TO_MEMORY: u16 = 0xA140;

/// The engine state of one channel. Saved in machine snapshots without
/// its hooks, which the channel restored into has already (see
/// [`Engines::restore`]).
#[derive(Serialize, Deserialize)]
pub struct Channel {
    subchannels: [Option<u16>; 8],
    three_d: three_d::ThreeD,
    two_d: two_d::TwoD,
    copy: copy::Copy,
    compute: compute::Compute,
    inline: inline::InlineToMemory,
    mme: mme::Mme,
    #[serde(skip)]
    hooks: Hooks,
}

impl Default for Channel {
    fn default() -> Self {
        Self::new()
    }
}

/// The current color target as an image, for debugging.
#[derive(Debug, Clone, PartialEq)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    /// Row-major RGBA, 8 bits per component.
    pub rgba: Vec<u8>,
}

impl Image {
    /// Writes the image as an 8-bit RGBA PNG (debug aid).
    pub fn write_png(&self, path: &std::path::Path) -> std::io::Result<()> {
        let file = std::io::BufWriter::new(std::fs::File::create(path)?);
        let mut encoder = png::Encoder::new(file, self.width, self.height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder
            .write_header()
            .and_then(|mut writer| writer.write_image_data(&self.rgba))
            .map_err(std::io::Error::other)
    }
}

impl Channel {
    /// Draws what the 3D engine holds back (`ThreeD::draw_held`), with
    /// memory as its methods see it.
    fn draw_held(&mut self, memory: &dyn GpuMemory) -> Result<(), Unmodeled> {
        let tracked = resources::Tracked {
            memory,
            writes: &self.hooks.writes,
        };
        match &self.hooks.metal {
            Some(renderer) => self.three_d.draw_held(
                &metal::Coherent {
                    memory: &tracked,
                    renderer,
                    writes: &self.hooks.writes,
                },
                &self.hooks,
            ),
            None => self.three_d.draw_held(&tracked, &self.hooks),
        }
    }

    /// A channel drawing with the renderer chosen by [`select_renderer`].
    pub fn new() -> Channel {
        Self::with_renderer(if RENDERER.load(std::sync::atomic::Ordering::Relaxed) {
            RendererKind::Metal
        } else {
            RendererKind::Software
        })
    }

    /// A channel drawing with `kind` (software if Metal is asked for and
    /// no Metal device exists).
    pub fn with_renderer(kind: RendererKind) -> Channel {
        Channel {
            subchannels: [None; 8],
            three_d: Default::default(),
            two_d: Default::default(),
            copy: Default::default(),
            compute: Default::default(),
            inline: Default::default(),
            mme: Default::default(),
            hooks: Hooks {
                metal: if kind == RendererKind::Metal {
                    metal::Renderer::new().map(std::sync::Arc::new)
                } else {
                    None
                },
                ..Hooks::default()
            },
        }
    }

    /// The initial value of a class method, as the driver loads it into
    /// the MME shadow RAM when it builds the graphics context: nvservices'
    /// netlist (`NETB_img.bin`, ELF 0x11DD8C, region 7 `SW_METHOD_INIT`,
    /// 1350 pairs: `method << 16 | class`, value) holds 1333 for 0xB197 and
    /// 17 for 0x902D [region and register names RECALLED from nvgpu].
    pub fn set_initial_state(
        &mut self,
        class: u16,
        method: u32,
        value: u32,
    ) -> Result<(), Unmodeled> {
        match class {
            THREE_D => self.three_d.set_initial(method, value),
            TWO_D => self.two_d.set_initial(method, value),
            _ => Err(Unmodeled(format!(
                "initial state for class {class:#06x} method {:#x} = {value:#x}",
                method * 4
            ))),
        }
    }

    /// Connects the GPU timer (PTIMER nanoseconds).
    pub fn set_timer(&mut self, timer: Box<dyn Fn() -> u64 + Send>) {
        self.hooks.timer = Some(timer);
    }

    /// Connects host1x syncpoint increments.
    pub fn set_syncpoint_increment(&mut self, increment: Box<dyn Fn(u32) + Send + Sync>) {
        let increment: std::sync::Arc<dyn Fn(u32) + Send + Sync> = increment.into();
        if let Some(renderer) = &self.hooks.metal {
            renderer.set_increment(increment.clone());
        }
        self.hooks.syncpoint_increment = Some(increment);
    }

    /// The 3D engine's render target `index` as it is in memory now,
    /// converted to RGBA8 (debug aid; `None` if disabled or unreadable).
    pub fn render_target(&self, index: u32, memory: &dyn GpuMemory) -> Option<Image> {
        let registers = self.three_d.registers();
        let (surface, format) = pipeline::color_target(registers, index).ok()??;
        surface_image(surface, format, memory)
    }
}

/// `surface`, holding `format` pixels, as it is in memory now, converted to
/// RGBA8 (debug aid; `None` if unreadable).
pub fn surface_image(
    surface: surface::Surface,
    format: texture::format::ColorFormat,
    memory: &dyn GpuMemory,
) -> Option<Image> {
    let rect = surface::Rect {
        x0: 0,
        y0: 0,
        x1: surface.width,
        y1: surface.height,
    };
    let (width, height) = (surface.width, surface.height);
    let mut window = surface::Window::load(memory, surface, rect).ok()?;
    let mut rgba = vec![0; (width * height * 4) as usize];
    for band in window.bands_mut() {
        for y in band.y0..(band.y0 + surface::BAND_ROWS).min(height) {
            for x in 0..width {
                let value = format.read(band.pixel(x, y));
                let at = ((y * width + x) * 4) as usize;
                for (k, v) in value.iter().enumerate() {
                    rgba[at + k] = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
                }
            }
        }
    }
    Some(Image {
        width,
        height,
        rgba,
    })
}

impl Engines for Channel {
    fn call(
        &mut self,
        subchannel: u8,
        method: u32,
        argument: u32,
        memory: &dyn GpuMemory,
    ) -> Result<(), Unmodeled> {
        let tracked = resources::Tracked {
            memory,
            writes: &self.hooks.writes,
        };
        let coherent;
        let memory: &dyn GpuMemory = match &self.hooks.metal {
            Some(renderer) => {
                coherent = metal::Coherent {
                    memory: &tracked,
                    renderer,
                    writes: &self.hooks.writes,
                };
                &coherent
            }
            None => &tracked,
        };
        // Signals whose work the GPU has done since are made.
        if let Some(renderer) = &self.hooks.metal {
            renderer.complete(&tracked, &self.hooks.writes, false)?;
        }
        let slot = self
            .subchannels
            .get_mut(subchannel as usize)
            .ok_or_else(|| Unmodeled(format!("subchannel {subchannel} does not exist")))?;
        if method == 0 {
            let class = argument as u16;
            if !matches!(
                class,
                THREE_D | COMPUTE | TWO_D | COPY | COPY_A | INLINE_TO_MEMORY
            ) {
                return Err(Unmodeled(format!(
                    "SET_OBJECT {argument:#x} on subchannel {subchannel}: class not modeled"
                )));
            }
            *slot = Some(class);
            return Ok(());
        }
        let class = slot.ok_or_else(|| {
            Unmodeled(format!(
                "method {:#x} = {argument:#x} on subchannel {subchannel} without an object",
                method * 4
            ))
        })?;
        if method < 0x40 {
            return Err(Unmodeled(format!(
                "host method {:#x} = {argument:#x} reached the engines",
                method * 4
            )));
        }
        // What another engine does comes after the draws before it.
        if class != THREE_D {
            self.three_d.draw_held(memory, &self.hooks)?;
        }
        match class {
            THREE_D => self
                .three_d
                .method(method, argument, memory, &mut self.mme, &self.hooks),
            TWO_D => self.two_d.method(method, argument, memory, &self.hooks),
            COPY | COPY_A => self.copy.method(method, argument, memory, &self.hooks),
            COMPUTE => self.compute.method(method, argument, memory, &self.hooks),
            _ => match method {
                inline::FIRST_METHOD..=inline::LAST_METHOD => {
                    match self.inline.method(method, argument, memory)? {
                        inline::Completion::None => Ok(()),
                        inline::Completion::Semaphore {
                            address,
                            payload,
                            four_words,
                        } => three_d::release_semaphore(
                            memory,
                            &self.hooks,
                            address,
                            payload,
                            four_words,
                        ),
                    }
                }
                // NO_OPERATION, WAIT_FOR_IDLE: synchronous execution.
                0x40 | 0x44 => Ok(()),
                _ => Err(Unmodeled(format!(
                    "inline-to-memory (0xA140): method {:#x} = {argument:#x}",
                    method * 4
                ))),
            },
        }
    }

    fn flush(&mut self, memory: &dyn GpuMemory) -> Result<(), Unmodeled> {
        self.draw_held(memory)?;
        let tracked = resources::Tracked {
            memory,
            writes: &self.hooks.writes,
        };
        self.hooks.signal(&tracked)
    }

    fn release(
        &mut self,
        memory: &dyn GpuMemory,
        release: Release,
    ) -> Result<Option<Release>, Unmodeled> {
        self.draw_held(memory)?;
        let tracked = resources::Tracked {
            memory,
            writes: &self.hooks.writes,
        };
        match &self.hooks.metal {
            Some(renderer) => renderer.release(&tracked, &self.hooks.writes, release),
            None => Ok(Some(release)),
        }
    }

    fn idle(&mut self, memory: &dyn GpuMemory) -> Result<(), Unmodeled> {
        self.draw_held(memory)?;
        if let Some(renderer) = &self.hooks.metal {
            let tracked = resources::Tracked {
                memory,
                writes: &self.hooks.writes,
            };
            renderer.complete(&tracked, &self.hooks.writes, true)?;
        }
        Ok(())
    }

    /// Draws held back belong to the work before: they are drawn first.
    fn synchronize(&mut self, memory: &dyn GpuMemory) -> Result<(), Unmodeled> {
        self.draw_held(memory)?;
        self.hooks.writes.synchronize();
        Ok(())
    }

    fn settle(&mut self, memory: &dyn GpuMemory) -> Result<(), Unmodeled> {
        self.draw_held(memory)?;
        let tracked = resources::Tracked {
            memory,
            writes: &self.hooks.writes,
        };
        if let Some(renderer) = &self.hooks.metal {
            renderer.settle(&tracked, &self.hooks.writes)?;
        }
        self.hooks.writes.synchronize();
        Ok(())
    }

    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        encode(self)
    }

    /// Every engine's state is replaced; the hooks (timer, syncpoints,
    /// renderer) stay. Every cache starts empty: the 3D engine's programs
    /// and textures are not saved, the Metal renderer drops what it holds
    /// on the host GPU, and a new epoch of [`resources::Writes`] begins.
    fn restore(&mut self, state: &[u8]) -> Result<(), SnapshotError> {
        let restored: Channel = decode("GPU channel engines", state)?;
        if !(restored.three_d.fits()
            && restored.two_d.fits()
            && restored.copy.fits()
            && restored.compute.fits())
        {
            return Err(SnapshotError::Corrupt(
                "GPU channel engines: register files of another size".into(),
            ));
        }
        let hooks = std::mem::take(&mut self.hooks);
        *self = Channel { hooks, ..restored };
        if let Some(renderer) = &self.hooks.metal {
            renderer.drop_caches();
        }
        self.hooks.writes.synchronize();
        Ok(())
    }
}
