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
//! * the compute class only as far as inline-to-memory and semaphores:
//!   kernel launches are refused.
//!
//! Everything runs synchronously inside [`Engines::call`]: when a method
//! returns, its effects are in memory. Whatever is not modeled returns
//! [`Unmodeled`] naming class, method and value; it is never skipped.
//!
//! Sources: the method interface is NVIDIA's own [public: open-gpu-doc
//! `classes/`]; behaviors the class headers do not define follow the
//! open-source Switch emulators (eden, Ryujinx) or the Direct3D/OpenGL
//! rules the hardware implements, and are labelled so where they are
//! used. Values pinned by the firmware are cited with program and address.

pub mod blend;
pub mod capture;
pub mod copy;
pub mod inline;
pub mod metal;
pub mod mme;
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

/// Memory as a GPU channel addresses it: its page tables applied.
pub trait GpuMemory: Sync {
    fn read(&self, address: u64, buffer: &mut [u8]) -> Option<()>;
    fn write(&self, address: u64, data: &[u8]) -> Option<()>;
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
    /// the channel is about to signal: the pushbuffer executor calls it
    /// before its own semaphore releases and syncpoint increments, the
    /// host work that lets other agents observe the channel's progress.
    /// Engines that write memory as they go have nothing to do.
    fn flush(&mut self, _memory: &dyn GpuMemory) -> Result<(), Unmodeled> {
        Ok(())
    }
}

/// Connections to the rest of the GPU a channel's engines need for some
/// methods; each method that uses one refuses to run without it.
#[derive(Default)]
pub struct Hooks {
    /// GPU timer (PTIMER) in nanoseconds, for the timestamp of four-word
    /// semaphore releases.
    pub timer: Option<Box<dyn Fn() -> u64 + Send>>,
    /// Increments a host1x syncpoint by one (3D `INCREMENT_SYNC_POINT`).
    pub syncpoint_increment: Option<Box<dyn Fn(u32) + Send>>,
    /// The Metal renderer of the channel, if it renders with Metal: every
    /// completion the channel signals (semaphore release, report,
    /// syncpoint increment) first writes its pending work back to guest
    /// memory ([`metal::Renderer::flush`]).
    pub metal: Option<std::sync::Arc<metal::Renderer>>,
    /// What the engines wrote since the channel last signaled.
    pub writes: resources::Writes,
}

impl Hooks {
    /// The channel is about to signal (release a semaphore or report,
    /// increment a syncpoint): makes all rendering so far visible in guest
    /// memory and starts a new epoch of [`resources::Writes`].
    pub fn signal(&self, memory: &dyn GpuMemory) -> Result<(), Unmodeled> {
        if let Some(renderer) = &self.metal {
            renderer.flush(memory, &self.writes)?;
        }
        self.writes.signal();
        Ok(())
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

/// The engine state of one channel.
pub struct Channel {
    subchannels: [Option<u16>; 8],
    three_d: three_d::ThreeD,
    two_d: two_d::TwoD,
    copy: copy::Copy,
    compute: Compute,
    inline: inline::InlineToMemory,
    mme: mme::Mme,
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
    pub fn set_syncpoint_increment(&mut self, increment: Box<dyn Fn(u32) + Send>) {
        self.hooks.syncpoint_increment = Some(increment);
    }

    /// The 3D engine's render target `index` as it is in memory now,
    /// converted to RGBA8 (debug aid; `None` if disabled or unreadable).
    pub fn render_target(&self, index: u32, memory: &dyn GpuMemory) -> Option<Image> {
        let registers = self.three_d.registers();
        let (surface, format) = pipeline::color_target(registers, index).ok()??;
        let rect = surface::Rect {
            x0: 0,
            y0: 0,
            x1: surface.width,
            y1: surface.height,
        };
        let mut window = surface::Window::load(memory, surface, rect).ok()?;
        let mut rgba = vec![0; (surface.width * surface.height * 4) as usize];
        for band in window.bands_mut() {
            for y in band.y0..(band.y0 + surface::BAND_ROWS).min(surface.height) {
                for x in 0..surface.width {
                    let value = format.read(band.pixel(x, y));
                    let at = ((y * surface.width + x) * 4) as usize;
                    for (k, v) in value.iter().enumerate() {
                        rgba[at + k] = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
                    }
                }
            }
        }
        Some(Image {
            width: surface.width,
            height: surface.height,
            rgba,
        })
    }
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
        match class {
            THREE_D => self
                .three_d
                .method(method, argument, memory, &mut self.mme, &self.hooks),
            TWO_D => self.two_d.method(method, argument, memory),
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
        let tracked = resources::Tracked {
            memory,
            writes: &self.hooks.writes,
        };
        self.hooks.signal(&tracked)
    }
}

/// The compute class (0xB1C0), as far as modeled: its inline-to-memory
/// block and semaphore releases. Kernel launches are refused.
#[derive(Default)]
struct Compute {
    registers: Vec<u32>,
    inline: inline::InlineToMemory,
}

impl Compute {
    fn method(
        &mut self,
        method: u32,
        argument: u32,
        memory: &dyn GpuMemory,
        hooks: &Hooks,
    ) -> Result<(), Unmodeled> {
        if self.registers.is_empty() {
            self.registers = vec![0; 0xE00];
        }
        let index = method as usize;
        if index >= self.registers.len() {
            return Err(Unmodeled(format!(
                "compute (0xB1C0): method {:#x} = {argument:#x}",
                method * 4
            )));
        }
        self.registers[index] = argument;
        match method {
            inline::FIRST_METHOD..=inline::LAST_METHOD => {
                match self.inline.method(method, argument, memory)? {
                    inline::Completion::None => Ok(()),
                    inline::Completion::Semaphore {
                        address,
                        payload,
                        four_words,
                    } => three_d::release_semaphore(memory, hooks, address, payload, four_words),
                }
            }
            // NO_OPERATION, WAIT_FOR_IDLE, cache invalidations (0x1330,
            // 0x1334, 0x1338, 0x120c, 0x1210, 0x1288, 0x1424, 0x1698): no
            // effect with synchronous execution and no caches.
            0x40 | 0x44 | 0x4CC | 0x4CD | 0x4CE | 0x483 | 0x484 | 0x4A2 | 0x509 | 0x5A6 => Ok(()),
            // SET_REPORT_SEMAPHORE_D: release of the payload only.
            0x6C3 => {
                let address = u64::from(self.registers[0x6C0] & 0xFF) << 32
                    | u64::from(self.registers[0x6C1]);
                let payload = self.registers[0x6C2];
                if argument & 3 != 0 || argument & !(3 | (1 << 28)) != 0 {
                    return Err(Unmodeled(format!(
                        "compute (0xB1C0): SET_REPORT_SEMAPHORE_D {argument:#x}"
                    )));
                }
                three_d::release_semaphore(
                    memory,
                    hooks,
                    address,
                    payload,
                    argument & (1 << 28) == 0,
                )
            }
            // Actions: notify, launches (SEND_PCAS_A/B, SEND_SIGNALING_PCAS_B),
            // MME loads.
            0x43 | 0x45..=0x48 | 0xAD..=0xAF => Err(Unmodeled(format!(
                "compute (0xB1C0): action method {:#x} = {argument:#x} not modeled",
                method * 4
            ))),
            _ => Ok(()),
        }
    }
}
