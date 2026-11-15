//! The Metal renderer: 3D draws and clears of a channel run on the host
//! GPU through Metal, with the software renderer as the reference and the
//! fallback.
//!
//! # What runs where
//!
//! Everything up to rasterization stays in the software pipeline
//! ([`super::pipeline`]): vertex fetch, the vertex program (interpreted),
//! clipping, the viewport transform and triangle setup. Metal receives
//! the set-up triangles (snapped window positions) with each triangle's
//! interpolation planes, rasterizes them, runs the pixel program
//! translated to MSL ([`super::shader::msl`]), which evaluates the planes
//! exactly as the interpreter does, samples textures decoded by
//! [`super::texture`], and blends into render targets held as Metal
//! textures. A draw or clear the Metal path cannot express (an
//! untranslated instruction, a format, blend factor, sampler mode or
//! pipeline state it does not map) runs in the software renderer instead,
//! after the render targets it touches are written back; the reason is
//! counted ([`Renderer::fallbacks`]).
//!
//! # Guest memory stays truthful where the guest can look
//!
//! A render target used by Metal is *resident*: a Metal texture holding
//! the surface, plus a mirror of the surface's bytes as guest memory had
//! them when the two were last made equal. Rendering makes the resident
//! *dirty* (the Metal texture is newer than guest memory, over a known
//! rectangle). Draws and clears are recorded into one command buffer that
//! stays open until the next write-back, which ends it, waits for the GPU
//! and stores the dirty pixels in guest memory, in the surface's own
//! layout. That happens:
//!
//! - when the channel signals: semaphore releases, reports and syncpoint
//!   increments of its engines and of the pushbuffer executor
//!   ([`super::Hooks::signal`], [`super::Engines::flush`]);
//! - before any other access of the channel's engines to memory that
//!   overlaps a dirty target: all engine memory accesses go through
//!   [`Coherent`], which writes back first. That covers copies, 2D blits,
//!   inline uploads, texture sampling of a render target, and constant
//!   and vertex data;
//! - before a draw or clear Metal cannot express runs in the software
//!   renderer.
//!
//! Between signals the dirty pixels exist only on the host GPU. Nothing
//! outside the channel can tell: the guest CPU, other channels and the
//! display learn that rendering finished only from a signal (NVN fences
//! are syncpoint increments or semaphore releases the CPU, nvservices or
//! another channel waits on; the compositor flips a buffer once its fence
//! passed), and on hardware too, memory written by work not yet followed
//! by a signal may or may not hold the result. Ends of a channel's turn,
//! wait-for-idle, `FB_FLUSH` and L2 flushes are not signals: they let no
//! one observe anything until a signal follows.
//!
//! Before a resident is used, its mirror is compared with guest memory
//! (and the Metal texture reloaded if they differ) at most once per epoch
//! of [`Writes`]: after each signal of the channel, and after the
//! channel's own engines wrote memory overlapping it. Anything else (the
//! CPU, another channel) may only write a surface the channel rendered or
//! read once a signal told it the GPU is done with it; between, such a
//! write races with the GPU, as on hardware.

mod format;

use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::{Arc, LazyLock, Mutex};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBlendFactor, MTLBlendOperation, MTLBuffer, MTLColorWriteMask, MTLCommandBuffer,
    MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions, MTLCreateSystemDefaultDevice,
    MTLDepthClipMode, MTLDevice, MTLLibrary, MTLLoadAction, MTLMathMode, MTLOrigin, MTLPixelFormat,
    MTLPrimitiveType, MTLRegion, MTLRenderCommandEncoder, MTLRenderPassDescriptor,
    MTLRenderPipelineDescriptor, MTLRenderPipelineState, MTLResourceOptions, MTLSamplerAddressMode,
    MTLSamplerBorderColor, MTLSamplerDescriptor, MTLSamplerMinMagFilter, MTLSamplerMipFilter,
    MTLSamplerState, MTLScissorRect, MTLSize, MTLStorageMode, MTLStoreAction, MTLTexture,
    MTLTextureDescriptor, MTLTextureUsage, MTLViewport,
};

use super::blend::{Blend, Factor, Operation};
use super::raster::{ColorOutput, Compare, Triangle};
use super::resources::{DrawTextures, Stamp, Writes};
use super::shader::Program;
use super::shader::msl::{self, Translated};
use super::surface::{Rect, Surface, Window};
use super::texture::format::ColorFormat;
use super::texture::{Texture, Tsc};
use super::{GpuMemory, Unmodeled};

type Device = ProtocolObject<dyn MTLDevice>;
type Queue = ProtocolObject<dyn MTLCommandQueue>;
type MetalTexture = ProtocolObject<dyn MTLTexture>;
type MetalBuffer = ProtocolObject<dyn MTLBuffer>;
type Pipeline = ProtocolObject<dyn MTLRenderPipelineState>;
type Sampler = ProtocolObject<dyn MTLSamplerState>;
type CommandBuffer = ProtocolObject<dyn MTLCommandBuffer>;
type Encoder = ProtocolObject<dyn MTLRenderCommandEncoder>;
type Library = ProtocolObject<dyn MTLLibrary>;
/// A draw's texture fault words, with its unbound slots and the refusal
/// an executed sample of each is.
type Faults = (Retained<MetalBuffer>, Vec<(usize, String)>);

/// The host GPU, shared by all channels.
struct Gpu {
    device: Retained<Device>,
    queue: Retained<Queue>,
}

// SAFETY: Metal devices and command queues are thread-safe [public: Apple
// Metal Programming Guide, "Transient and Non-transient Objects in
// Metal": `MTLDevice`, `MTLCommandQueue` may be used from any thread].
unsafe impl Send for Gpu {}
unsafe impl Sync for Gpu {}

static GPU: LazyLock<Option<Gpu>> = LazyLock::new(|| {
    let device = MTLCreateSystemDefaultDevice()?;
    let queue = device.newCommandQueue()?;
    Some(Gpu { device, queue })
});

/// Whether a Metal device is present.
pub fn available() -> bool {
    GPU.is_some()
}

/// A render target held on the GPU.
struct Resident {
    surface: Surface,
    format: ColorFormat,
    texture: Retained<MetalTexture>,
    /// The whole surface as guest memory held it when it and `texture`
    /// were last equal.
    mirror: Window,
    /// Pixels rendered since the last write-back.
    dirty: Option<Rect>,
    /// When `mirror` was last known to equal guest memory.
    stamp: Option<Stamp>,
}

/// The attachments of a render pass: resident index and whether the sRGB
/// view is used.
type Attachments = Vec<(usize, bool)>;

/// Command buffer being recorded.
struct Frame {
    command: Retained<CommandBuffer>,
    /// The open render pass: its encoder and attachments.
    pass: Option<(Retained<Encoder>, Attachments)>,
}

impl Drop for Frame {
    /// A frame dropped unfinished (its channel stopped, or the machine
    /// did, between signals) ends its open pass: Metal aborts the process
    /// when an encoder is released without `endEncoding`. The command
    /// buffer itself is dropped uncommitted; its work was never signaled.
    fn drop(&mut self) {
        if let Some((encoder, _)) = self.pass.take() {
            encoder.endEncoding();
        }
    }
}

struct Program2 {
    /// Keeps the program alive so its address is not reused as a key.
    _program: Arc<Program>,
    translated: Result<Arc<Translated>, String>,
}

#[derive(Default)]
struct State {
    residents: Vec<Resident>,
    frame: Option<Frame>,
    programs: HashMap<usize, Program2>,
    /// Compiled MSL by source: programs drawn with different attachment
    /// formats or blending share their functions.
    libraries: HashMap<String, Retained<Library>>,
    pipelines: HashMap<PipelineKey, Retained<Pipeline>>,
    clear_pipelines: HashMap<(usize, u8), Retained<Pipeline>>,
    /// Uploaded textures by decoded texture (kept alive with it while the
    /// texture cache holds it too).
    textures: HashMap<usize, (Arc<Texture>, Retained<MetalTexture>)>,
    samplers: HashMap<[u32; 8], Retained<Sampler>>,
    fallbacks: HashMap<String, u64>,
    draws: u64,
    /// Texture fault words of the draws recorded since the last flush
    /// that bound placeholders, with each unbound slot and the refusal a
    /// sample of it is (see `shader::msl` `texture.rs`).
    faults: Vec<Faults>,
    /// What an unbound texture slot is bound to: a 1x1 texture and a
    /// sampler, never read into a result that reaches guest memory.
    placeholder: Option<(Retained<MetalTexture>, Retained<Sampler>)>,
}

// SAFETY: the Metal objects of a channel (textures, buffers, pipeline
// states, command buffers, encoders) are only used under the renderer's
// lock, by one thread at a time; pipeline and sampler states are
// thread-safe themselves [public: Apple Metal Programming Guide].
unsafe impl Send for State {}

#[derive(Clone, PartialEq, Eq, Hash)]
struct PipelineKey {
    program: usize,
    /// The constant buffer 1 a translation with BRX was made for.
    jump_tables: Option<Vec<u8>>,
    /// Per attachment: pixel format, blend, write mask, program output.
    attachments: Vec<(usize, Option<Blend>, u8, usize)>,
    /// Whether the function tests alpha (its function and reference are
    /// draw-time values).
    alpha_test: bool,
}

/// The Metal renderer of one channel.
pub struct Renderer {
    state: Mutex<State>,
    /// Set while [`Renderer::flush`] writes back: its own writes pass
    /// [`Coherent`] unchecked (the state is locked by the flush).
    writing_back: std::sync::atomic::AtomicBool,
}

/// What a draw hands the Metal renderer once the software pipeline has
/// set it up.
pub struct Draw<'a> {
    pub program: &'a Arc<Program>,
    pub triangles: &'a [Triangle],
    /// Color outputs, and the surface of each (by `window` index).
    pub colors: &'a [ColorOutput],
    pub surfaces: &'a [Surface],
    pub bounds: Rect,
    pub blend_constant: [f32; 4],
    /// Constant buffer contents of the pixel stage and their bound sizes.
    pub constants: &'a [(Vec<u8>, usize)],
    pub textures: &'a DrawTextures<'a>,
    pub texture_buffer_index: u32,
    /// The alpha test (see `raster::Backend::alpha_test`).
    pub alpha_test: Option<(Compare, f32)>,
}

impl Renderer {
    /// A renderer, or `None` without a Metal device.
    pub fn new() -> Option<Renderer> {
        GPU.as_ref()?;
        Some(Renderer {
            state: Mutex::new(State::default()),
            writing_back: Default::default(),
        })
    }

    fn gpu() -> &'static Gpu {
        GPU.as_ref().expect("a renderer exists only with a device")
    }

    /// Counts of draws and clears left to the software renderer, by
    /// reason, and of draws Metal ran.
    pub fn fallbacks(&self) -> (u64, Vec<(String, u64)>) {
        let state = self.state.lock().unwrap();
        let mut list: Vec<_> = state
            .fallbacks
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        list.sort();
        (state.draws, list)
    }

    /// Counts a draw left to the software renderer; each reason is
    /// reported once on stderr when first seen.
    fn fall_back(&self, reason: String) -> Result<bool, Unmodeled> {
        let mut state = self.state.lock().unwrap();
        let count = state.fallbacks.entry(reason.clone()).or_default();
        if *count == 0 {
            eprintln!("Metal renderer: left to the software renderer: {reason}");
        }
        *count += 1;
        Ok(false)
    }

    /// Counts a draw left to the software renderer by its caller.
    pub fn note_fallback(&self, reason: &str) {
        let _ = self.fall_back(reason.to_string());
    }

    /// Ends the command buffer, waits for the GPU and writes every dirty
    /// target back to guest memory (through `memory`, which notes the
    /// writes in `writes`).
    pub fn flush(&self, memory: &dyn GpuMemory, writes: &Writes) -> Result<(), Unmodeled> {
        use std::sync::atomic::Ordering;
        let mut state = self.state.lock().unwrap();
        state.finish();
        // An executed sample of an unbound texture is the interpreter's
        // refusal; nothing of the work is written back.
        for (buffer, slots) in std::mem::take(&mut state.faults) {
            let words = buffer.contents().cast::<u32>();
            for (slot, refusal) in slots {
                // SAFETY: the buffer holds two words per texture slot of
                // the draw; the GPU finished.
                if unsafe { words.add(2 * slot + 1).read() } != 0 {
                    return Err(Unmodeled(refusal));
                }
            }
        }
        // Residents whose mirror still equals memory do after their own
        // write-back too (residents do not overlap).
        let current: Vec<bool> = state
            .residents
            .iter()
            .map(|r| {
                let (start, end) = r.extent();
                r.stamp.is_some_and(|s| writes.unchanged(s, start, end))
            })
            .collect();
        self.writing_back.store(true, Ordering::Relaxed);
        let mut result = Ok(());
        for resident in &mut state.residents {
            if let Some(rect) = resident.dirty.take()
                && result.is_ok()
            {
                result = resident.write_back(rect, memory);
            }
        }
        self.writing_back.store(false, Ordering::Relaxed);
        let stamp = writes.stamp();
        for (resident, current) in state.residents.iter_mut().zip(current) {
            resident.stamp = current.then_some(stamp);
        }
        // Uploads of textures the texture cache dropped are dropped too.
        state
            .textures
            .retain(|_, (texture, _)| Arc::strong_count(texture) > 1);
        result
    }

    /// Flushes if `length` bytes at `address` overlap a dirty target.
    fn before_access(
        &self,
        memory: &dyn GpuMemory,
        writes: &Writes,
        address: u64,
        length: usize,
    ) -> Result<(), Unmodeled> {
        if self.writing_back.load(std::sync::atomic::Ordering::Relaxed) {
            return Ok(());
        }
        let overlaps = {
            let state = self.state.lock().unwrap();
            state.residents.iter().any(|r| {
                r.dirty.is_some() && overlap(r.extent(), (address, address + length as u64))
            })
        };
        if overlaps {
            self.flush(memory, writes)?;
        }
        Ok(())
    }

    /// A draw on Metal; `Ok(false)` if Metal cannot do it (nothing was
    /// changed then).
    pub fn draw(
        &self,
        memory: &dyn GpuMemory,
        writes: &Writes,
        draw: &Draw<'_>,
    ) -> Result<bool, Unmodeled> {
        // Inputs that come from memory are gathered before the state is
        // locked: reading them may flush.
        // Constant buffer 1, where BRX jump tables live (a translation with
        // indirect branches is valid only for the content it was made for).
        let constant_buffer_1 = draw
            .constants
            .get(1)
            .map_or(&[][..], |(data, size)| &data[..data.len().min(*size)]);
        let translated = {
            let mut state = self.state.lock().unwrap();
            let key = Arc::as_ptr(draw.program) as usize;
            let entry = state.programs.entry(key).or_insert_with(|| Program2 {
                _program: Arc::clone(draw.program),
                translated: msl::translate(draw.program, constant_buffer_1).map(Arc::new),
            });
            if let Ok(t) = &entry.translated
                && t.constant_buffer_1
                    .as_deref()
                    .is_some_and(|table| table != constant_buffer_1)
            {
                entry.translated = msl::translate(draw.program, constant_buffer_1).map(Arc::new);
            }
            entry.translated.clone()
        };
        let translated = match translated {
            Ok(t) => t,
            Err(reason) => return self.fall_back(format!("pixel program: {reason}")),
        };
        if draw.program.header().ps_writes_depth() {
            return self.fall_back("pixel program writes depth".into());
        }
        // Slots read at register offsets (LDC): their readable part of the
        // bound range, in words, and its size in bytes (see
        // `Translated::source`). A bound range can reach past the memory
        // its owner mapped (see `pipeline::read_mapped`); an indexed read
        // there, which the interpreter refuses (hardware would fault),
        // reads 0 on Metal. No statically known read may go there.
        let mut constants: HashMap<u32, Vec<u8>> = HashMap::new();
        let mut sizes = [0u32; 18];
        for &slot in &translated.indirect {
            let Some((data, size)) = draw.constants.get(slot as usize) else {
                return self.fall_back(format!("constant buffer slot {slot}"));
            };
            let readable = data.len().min(*size);
            let mut buffer = data[..readable].to_vec();
            buffer.resize(readable.next_multiple_of(4).max(4), 0);
            constants.insert(slot, buffer);
            sizes[slot as usize] = readable as u32;
        }
        // Constant buffer words read statically must be readable.
        for &(slot, offset) in &translated.constants {
            let indexed = translated.indirect.contains(&slot);
            let Some((data, size)) = draw.constants.get(slot as usize) else {
                return self.fall_back(format!("constant buffer slot {slot}"));
            };
            let end = offset as usize + 4;
            if end > data.len() && end <= *size {
                return self.fall_back("constant read past the mapped part".into());
            }
            if indexed {
                // Bounds-checked by the program against `sizes`.
                continue;
            }
            let buffer = constants.entry(slot).or_insert_with(|| data.clone());
            if buffer.len() < end {
                buffer.resize(end, 0);
            }
        }
        // Textures and samplers. A handle that names no texture is bound to
        // a placeholder and marked: the program records an executed sample
        // of it, which `flush` refuses as the interpreter does (see
        // `shader::msl` `texture.rs`). netConnect's uber-shader names slots
        // its draws leave unset (TIC index 0xfffff) on branches they do not
        // take.
        let mut bound = Vec::new();
        let mut unbound = Vec::new();
        for (slot, &handle_offset) in translated.textures.iter().enumerate() {
            let Some((data, _)) = draw.constants.get(draw.texture_buffer_index as usize) else {
                return self.fall_back("texture handle buffer unbound".into());
            };
            let Some(bytes) = data.get(handle_offset as usize..handle_offset as usize + 4) else {
                return self.fall_back("texture handle outside the constant buffer".into());
            };
            let handle = u32::from_le_bytes(bytes.try_into().expect("4 bytes"));
            let (texture, tsc) = match draw.textures.resolve(handle) {
                Ok(resolved) => resolved,
                Err(refusal) => {
                    unbound.push((slot, refusal.0));
                    bound.push(None);
                    continue;
                }
            };
            if let Err(reason) = format::check_texture(&texture, &tsc) {
                return self.fall_back(reason);
            }
            let bias = if translated.implicit_lod[slot] {
                tsc.lod_bias
            } else {
                0.0
            };
            bound.push(Some((texture, tsc, bias)));
        }
        // Targets.
        let mut attachments = Vec::new();
        for color in draw.colors {
            let Some(pixel_format) = format::pixel_format(color.format) else {
                return self.fall_back(format!("color format {:?}", color.format));
            };
            if let Some(blend) = &color.blend
                && let Err(reason) = format::check_blend(blend)
            {
                return self.fall_back(reason);
            }
            let surface = draw.surfaces[color.window];
            attachments.push((surface, color, pixel_format));
        }
        if attachments.is_empty() {
            return self.fall_back("draw without color targets".into());
        }
        let residents = self.residents(
            memory,
            writes,
            &attachments
                .iter()
                .map(|a| (a.0, a.1.format))
                .collect::<Vec<_>>(),
        )?;

        let mut state = self.state.lock().unwrap();
        let gpu = Self::gpu();
        // Pipeline.
        let key = PipelineKey {
            program: Arc::as_ptr(draw.program) as usize,
            jump_tables: translated.constant_buffer_1.clone(),
            attachments: attachments
                .iter()
                .map(|(_, color, pixel_format)| {
                    let format = if color.srgb {
                        format::srgb(*pixel_format)
                    } else {
                        *pixel_format
                    };
                    let mask = color
                        .write
                        .iter()
                        .enumerate()
                        .fold(0u8, |m, (i, &w)| m | (u8::from(w) << i));
                    (format.0, color.blend, mask, color.output)
                })
                .collect(),
            alpha_test: draw.alpha_test.is_some(),
        };
        let pipeline = match state.pipelines.get(&key) {
            Some(p) => p.clone(),
            None => {
                let outputs: Vec<usize> = key.attachments.iter().map(|a| a.3).collect();
                let source = translated.source(draw.program, &outputs, key.alpha_test);
                let library = state.library(gpu, source)?;
                let pipeline = build_pipeline(gpu, &library, "fragment_main", &key.attachments)?;
                state.pipelines.insert(key.clone(), pipeline.clone());
                pipeline
            }
        };
        // Textures and samplers on the GPU.
        let mut gpu_textures = Vec::new();
        for entry in &bound {
            let Some((texture, tsc, _)) = entry else {
                gpu_textures.push(state.placeholder(gpu)?);
                continue;
            };
            let id = Arc::as_ptr(texture) as usize;
            let metal = match state.textures.get(&id) {
                Some((_, t)) => t.clone(),
                None => {
                    let t = upload_texture(gpu, texture)?;
                    state.textures.insert(id, (Arc::clone(texture), t.clone()));
                    t
                }
            };
            let sampler = match state.samplers.get(&tsc.words) {
                Some(s) => s.clone(),
                None => {
                    let s = build_sampler(gpu, texture, tsc);
                    state.samplers.insert(tsc.words, s.clone());
                    s
                }
            };
            gpu_textures.push((metal, sampler));
        }
        // Geometry: window positions to normalized device coordinates of
        // a viewport covering the first target.
        let width = attachments[0].0.width as f32;
        let height = attachments[0].0.height as f32;
        let mut positions = Vec::with_capacity(draw.triangles.len() * 12);
        let mut planes =
            Vec::with_capacity(draw.triangles.len() * translated.ipa_words.len().max(1) * 3);
        for triangle in draw.triangles {
            for [x, y] in triangle.vertices() {
                let depth = triangle.depth.base + triangle.depth.dx * x + triangle.depth.dy * y;
                positions.extend_from_slice(&[
                    x / width * 2.0 - 1.0,
                    1.0 - y / height * 2.0,
                    depth.clamp(0.0, 1.0),
                    1.0,
                ]);
            }
            if translated.ipa_words.is_empty() {
                planes.extend_from_slice(&[0.0; 3]);
            }
            for &word in &translated.ipa_words {
                let plane = triangle.planes[word as usize];
                planes.extend_from_slice(&[plane.dx, plane.dy, plane.base]);
            }
        }
        let biases: Vec<f32> = bound
            .iter()
            .map(|b| b.as_ref().map_or(0.0, |b| b.2))
            .chain([0.0])
            .collect();
        // Texture fault words: word 2i marks slot i unbound.
        let mut fault_words = vec![0u32; 2 * translated.textures.len()];
        for &(slot, _) in &unbound {
            fault_words[2 * slot] = 1;
        }
        let encoder = state.encoder(
            &residents,
            &attachments.iter().map(|a| a.1.srgb).collect::<Vec<_>>(),
        )?;
        encoder.setRenderPipelineState(&pipeline);
        encoder.setViewport(MTLViewport {
            originX: 0.0,
            originY: 0.0,
            width: f64::from(width),
            height: f64::from(height),
            znear: 0.0,
            zfar: 1.0,
        });
        encoder.setDepthClipMode(MTLDepthClipMode::Clamp);
        let scissor = draw.bounds.intersect(&Rect {
            x0: 0,
            y0: 0,
            x1: attachments.iter().map(|a| a.0.width).min().unwrap_or(0),
            y1: attachments.iter().map(|a| a.0.height).min().unwrap_or(0),
        });
        if scissor.is_empty() {
            return Ok(true);
        }
        encoder.setScissorRect(MTLScissorRect {
            x: scissor.x0 as usize,
            y: scissor.y0 as usize,
            width: (scissor.x1 - scissor.x0) as usize,
            height: (scissor.y1 - scissor.y0) as usize,
        });
        let [r, g, b, a] = draw.blend_constant;
        encoder.setBlendColorRed_green_blue_alpha(r, g, b, a);
        // SAFETY: buffers and textures outlive the command buffer (Metal
        // retains them, and copies bytes), indices are within the
        // function's bindings.
        unsafe {
            bind(gpu, &encoder, Stage::Vertex, bytes_of(&positions), 0)?;
            bind(gpu, &encoder, Stage::Fragment, bytes_of(&planes), 0)?;
            bind(gpu, &encoder, Stage::Fragment, bytes_of(&biases), 1)?;
            for (slot, data) in &constants {
                bind(gpu, &encoder, Stage::Fragment, data, 2 + *slot as usize)?;
            }
            if !translated.indirect.is_empty() {
                bind(gpu, &encoder, Stage::Fragment, bytes_of_u32(&sizes), 20)?;
            }
            if let Some((compare, reference)) = draw.alpha_test {
                // The comparison numbering of `Translated::source`.
                let function = match compare {
                    Compare::Never => 0u32,
                    Compare::Less => 1,
                    Compare::Equal => 2,
                    Compare::LessEqual => 3,
                    Compare::Greater => 4,
                    Compare::NotEqual => 5,
                    Compare::GreaterEqual => 6,
                    Compare::Always => 7,
                };
                let words = [function, reference.to_bits()];
                bind(gpu, &encoder, Stage::Fragment, bytes_of_u32(&words), 22)?;
            }
            for (i, (texture, sampler)) in gpu_textures.iter().enumerate() {
                encoder.setFragmentTexture_atIndex(Some(texture), i);
                encoder.setFragmentSamplerState_atIndex(Some(sampler), i);
            }
            if !unbound.is_empty() {
                let faults = buffer(gpu, bytes_of_u32(&fault_words))?;
                encoder.setFragmentBuffer_offset_atIndex(Some(&faults), 0, 21);
                state.faults.push((faults, unbound));
            } else if !fault_words.is_empty() {
                bind(gpu, &encoder, Stage::Fragment, bytes_of_u32(&fault_words), 21)?;
            }
            encoder.drawPrimitives_vertexStart_vertexCount(
                MTLPrimitiveType::Triangle,
                0,
                draw.triangles.len() * 3,
            );
        }
        for &index in &residents {
            state.rendered(index, scissor, writes);
        }
        state.draws += 1;
        Ok(true)
    }

    /// A color clear on Metal: `components` (bit i: component i) of
    /// `rect` of the target set to `color`, encoded as the format stores
    /// it (no sRGB conversion, as `CLEAR_SURFACE` writes the value raw).
    pub fn clear(
        &self,
        memory: &dyn GpuMemory,
        writes: &Writes,
        (surface, format): (Surface, ColorFormat),
        rect: Rect,
        color: [f32; 4],
        components: u8,
    ) -> Result<bool, Unmodeled> {
        let Some(pixel_format) = format::pixel_format(format) else {
            return self.fall_back(format!("clear of color format {format:?}"));
        };
        let rect = rect.intersect(&Rect {
            x0: 0,
            y0: 0,
            x1: surface.width,
            y1: surface.height,
        });
        if rect.is_empty() {
            return Ok(true);
        }
        let residents = self.residents(memory, writes, &[(surface, format)])?;
        let mut state = self.state.lock().unwrap();
        let gpu = Self::gpu();
        let key = (pixel_format.0, components);
        let pipeline = match state.clear_pipelines.get(&key) {
            Some(p) => p.clone(),
            None => {
                let library = state.library(gpu, CLEAR_SOURCE.to_string())?;
                let p = build_pipeline(
                    gpu,
                    &library,
                    "clear_main",
                    &[(pixel_format.0, None, components, 0)],
                )?;
                state.clear_pipelines.insert(key, p.clone());
                p
            }
        };
        let encoder = state.encoder(&residents, &[false])?;
        encoder.setRenderPipelineState(&pipeline);
        encoder.setViewport(MTLViewport {
            originX: 0.0,
            originY: 0.0,
            width: f64::from(surface.width),
            height: f64::from(surface.height),
            znear: 0.0,
            zfar: 1.0,
        });
        encoder.setScissorRect(MTLScissorRect {
            x: rect.x0 as usize,
            y: rect.y0 as usize,
            width: (rect.x1 - rect.x0) as usize,
            height: (rect.y1 - rect.y0) as usize,
        });
        // SAFETY: as in `draw`.
        unsafe {
            bind(gpu, &encoder, Stage::Fragment, bytes_of(&color), 0)?;
            encoder.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3);
        }
        state.rendered(residents[0], rect, writes);
        Ok(true)
    }

    /// The residents for `targets`, created, checked against guest memory
    /// or reloaded as needed; their indices.
    fn residents(
        &self,
        memory: &dyn GpuMemory,
        writes: &Writes,
        targets: &[(Surface, ColorFormat)],
    ) -> Result<Vec<usize>, Unmodeled> {
        // A resident of other dimensions or format at overlapping memory is
        // written back and dropped, before any index is taken.
        for &(surface, format) in targets {
            let stale: bool = {
                let state = self.state.lock().unwrap();
                state.residents.iter().any(|r| {
                    (r.surface != surface || r.format != format)
                        && overlap(r.extent(), extent(&surface))
                })
            };
            if stale {
                self.flush(memory, writes)?;
                let mut state = self.state.lock().unwrap();
                state.residents.retain(|r| {
                    !((r.surface != surface || r.format != format)
                        && overlap(r.extent(), extent(&surface)))
                });
            }
        }
        let mut indices = Vec::new();
        for &(surface, format) in targets {
            // A dirty resident is current: anything else of the channel
            // that wrote its memory wrote it back first ([`Coherent`]),
            // and a signal writes it back too. A clean one has no work
            // pending on the GPU, so it may be reloaded.
            let found = {
                let state = self.state.lock().unwrap();
                state
                    .residents
                    .iter()
                    .position(|r| r.surface == surface && r.format == format)
                    .map(|i| {
                        let r = &state.residents[i];
                        let (start, end) = r.extent();
                        let current = r.dirty.is_some()
                            || r.stamp.is_some_and(|s| writes.unchanged(s, start, end));
                        (i, current)
                    })
            };
            let full = Rect {
                x0: 0,
                y0: 0,
                x1: surface.width,
                y1: surface.height,
            };
            match found {
                Some((index, true)) => indices.push(index),
                Some((index, false)) => {
                    // Compare with guest memory; reload what changed.
                    let mut window = Window::load(memory, surface, full)?;
                    let stamp = writes.stamp();
                    let mut state = self.state.lock().unwrap();
                    let resident = &mut state.residents[index];
                    if window.data() != resident.mirror.data() {
                        upload_surface(&resident.texture, &mut window, full);
                        resident.mirror = window;
                    }
                    resident.stamp = Some(stamp);
                    indices.push(index);
                }
                None => {
                    let pixel_format = format::pixel_format(format).expect("checked by the caller");
                    let mut window = Window::load(memory, surface, full)?;
                    let stamp = writes.stamp();
                    let texture =
                        new_target(Self::gpu(), pixel_format, surface.width, surface.height)?;
                    upload_surface(&texture, &mut window, full);
                    let mut state = self.state.lock().unwrap();
                    state.residents.push(Resident {
                        surface,
                        format,
                        texture,
                        mirror: window,
                        dirty: None,
                        stamp: Some(stamp),
                    });
                    indices.push(state.residents.len() - 1);
                }
            }
        }
        Ok(indices)
    }
}

impl State {
    /// Ends recording and waits for the GPU.
    fn finish(&mut self) {
        if let Some(mut frame) = self.frame.take() {
            if let Some((encoder, _)) = frame.pass.take() {
                encoder.endEncoding();
            }
            frame.command.commit();
            frame.command.waitUntilCompleted();
        }
    }

    /// Resident `index` got `rect` rendered by the GPU. The first draw
    /// after a write-back notes the whole surface as written, so that
    /// caches of its memory (a texture sampling it) look again, through
    /// [`Coherent`], which writes it back; the resident itself stays
    /// current.
    fn rendered(&mut self, index: usize, rect: Rect, writes: &Writes) {
        let resident = &mut self.residents[index];
        match resident.dirty {
            Some(dirty) => resident.dirty = Some(union(dirty, rect)),
            None => {
                let (start, end) = resident.extent();
                writes.record(start, end - start);
                resident.dirty = Some(rect);
                resident.stamp = Some(writes.stamp());
            }
        }
    }

    /// What an unbound texture slot is bound to (made once).
    fn placeholder(
        &mut self,
        gpu: &Gpu,
    ) -> Result<(Retained<MetalTexture>, Retained<Sampler>), Unmodeled> {
        if let Some(placeholder) = &self.placeholder {
            return Ok(placeholder.clone());
        }
        // SAFETY: plain descriptor construction.
        let descriptor = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat::RGBA32Float,
                1,
                1,
                false,
            )
        };
        descriptor.setStorageMode(MTLStorageMode::Shared);
        descriptor.setUsage(MTLTextureUsage::ShaderRead);
        let texture = gpu
            .device
            .newTextureWithDescriptor(&descriptor)
            .ok_or_else(|| Unmodeled("Metal: texture allocation failed".into()))?;
        let sampler = gpu
            .device
            .newSamplerStateWithDescriptor(&MTLSamplerDescriptor::new())
            .expect("sampler state");
        self.placeholder = Some((texture.clone(), sampler.clone()));
        Ok((texture, sampler))
    }

    /// The compiled library of `source`, compiled once.
    fn library(&mut self, gpu: &Gpu, source: String) -> Result<Retained<Library>, Unmodeled> {
        if let Some(library) = self.libraries.get(&source) {
            return Ok(library.clone());
        }
        let library = compile_library(gpu, &source)?;
        self.libraries.insert(source, library.clone());
        Ok(library)
    }

    /// The render pass encoder for `residents` as attachments (in order,
    /// with sRGB views where asked), opening one if needed.
    fn encoder(
        &mut self,
        residents: &[usize],
        srgb: &[bool],
    ) -> Result<Retained<Encoder>, Unmodeled> {
        let wanted: Vec<(usize, bool)> = residents
            .iter()
            .copied()
            .zip(srgb.iter().copied())
            .collect();
        if self.frame.is_none() {
            let command = Renderer::gpu()
                .queue
                .commandBuffer()
                .ok_or_else(|| Unmodeled("Metal: no command buffer".into()))?;
            self.frame = Some(Frame {
                command,
                pass: None,
            });
        }
        let frame = self.frame.as_mut().expect("just made");
        if let Some((encoder, attachments)) = &frame.pass
            && *attachments == wanted
        {
            return Ok(encoder.clone());
        }
        if let Some((encoder, _)) = frame.pass.take() {
            encoder.endEncoding();
        }
        let descriptor = MTLRenderPassDescriptor::new();
        for (i, &(index, srgb)) in wanted.iter().enumerate() {
            let texture = &self.residents[index].texture;
            let view;
            let texture = if srgb {
                view = texture
                    .newTextureViewWithPixelFormat(format::srgb(texture.pixelFormat()))
                    .ok_or_else(|| Unmodeled("Metal: sRGB view".into()))?;
                &view
            } else {
                texture
            };
            // SAFETY: `i` < 8 attachments.
            let attachment = unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(i) };
            attachment.setTexture(Some(texture));
            attachment.setLoadAction(MTLLoadAction::Load);
            attachment.setStoreAction(MTLStoreAction::Store);
        }
        let encoder = frame
            .command
            .renderCommandEncoderWithDescriptor(&descriptor)
            .ok_or_else(|| Unmodeled("Metal: no render encoder".into()))?;
        frame.pass = Some((encoder.clone(), wanted));
        Ok(encoder)
    }
}

impl Resident {
    fn extent(&self) -> (u64, u64) {
        extent(&self.surface)
    }

    /// Writes `rect` of the Metal texture to guest memory.
    fn write_back(&mut self, rect: Rect, memory: &dyn GpuMemory) -> Result<(), Unmodeled> {
        let bpp = self.surface.bytes_per_pixel as usize;
        let row = (rect.x1 - rect.x0) as usize * bpp;
        let mut pixels = vec![0u8; row * (rect.y1 - rect.y0) as usize];
        // SAFETY: the buffer holds the region's rows; the texture is
        // shared storage and the GPU finished writing it.
        unsafe {
            self.texture.getBytes_bytesPerRow_fromRegion_mipmapLevel(
                NonNull::new(pixels.as_mut_ptr().cast()).expect("non-null"),
                row,
                region(rect),
                0,
            );
        }
        self.mirror.write_pixels(rect, &pixels);
        self.mirror.store_rect(memory, rect)
    }
}

const CLEAR_SOURCE: &str = r#"#include <metal_stdlib>
using namespace metal;
struct Vertex { float4 position [[position]]; };
vertex Vertex vertex_main(uint id [[vertex_id]]) {
    // One triangle covering the viewport; the scissor limits it.
    float2 p = float2(id == 1 ? 3.0 : -1.0, id == 2 ? -3.0 : 1.0);
    Vertex out;
    out.position = float4(p, 0.0, 1.0);
    return out;
}
fragment float4 clear_main(const device float4* color [[buffer(0)]]) { return color[0]; }
"#;

fn compile_library(gpu: &Gpu, source: &str) -> Result<Retained<Library>, Unmodeled> {
    let options = MTLCompileOptions::new();
    options.setMathMode(MTLMathMode::Safe);
    gpu.device
        .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&options))
        .map_err(|e| Unmodeled(format!("Metal: shader compilation failed: {e}\n{source}")))
}

fn build_pipeline(
    gpu: &Gpu,
    library: &Library,
    fragment: &str,
    attachments: &[(usize, Option<Blend>, u8, usize)],
) -> Result<Retained<Pipeline>, Unmodeled> {
    let vertex = library
        .newFunctionWithName(&NSString::from_str("vertex_main"))
        .ok_or_else(|| Unmodeled("Metal: no vertex function".into()))?;
    let fragment = library
        .newFunctionWithName(&NSString::from_str(fragment))
        .ok_or_else(|| Unmodeled("Metal: no fragment function".into()))?;
    let descriptor = MTLRenderPipelineDescriptor::new();
    descriptor.setVertexFunction(Some(&vertex));
    descriptor.setFragmentFunction(Some(&fragment));
    for (i, &(pixel_format, blend, mask, _)) in attachments.iter().enumerate() {
        // SAFETY: `i` < 8 attachments.
        let attachment = unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(i) };
        attachment.setPixelFormat(MTLPixelFormat(pixel_format));
        let mut write = MTLColorWriteMask::None;
        for (bit, m) in [
            MTLColorWriteMask::Red,
            MTLColorWriteMask::Green,
            MTLColorWriteMask::Blue,
            MTLColorWriteMask::Alpha,
        ]
        .into_iter()
        .enumerate()
        {
            if mask & (1 << bit) != 0 {
                write |= m;
            }
        }
        attachment.setWriteMask(write);
        if let Some(blend) = blend {
            attachment.setBlendingEnabled(true);
            attachment.setRgbBlendOperation(operation(blend.color_op));
            attachment.setAlphaBlendOperation(operation(blend.alpha_op));
            attachment.setSourceRGBBlendFactor(factor(blend.color_source));
            attachment.setDestinationRGBBlendFactor(factor(blend.color_destination));
            attachment.setSourceAlphaBlendFactor(factor(blend.alpha_source));
            attachment.setDestinationAlphaBlendFactor(factor(blend.alpha_destination));
        }
    }
    gpu.device
        .newRenderPipelineStateWithDescriptor_error(&descriptor)
        .map_err(|e| Unmodeled(format!("Metal: pipeline creation failed: {e}")))
}

fn operation(op: Operation) -> MTLBlendOperation {
    match op {
        Operation::Add => MTLBlendOperation::Add,
        Operation::Subtract => MTLBlendOperation::Subtract,
        Operation::ReverseSubtract => MTLBlendOperation::ReverseSubtract,
        Operation::Min => MTLBlendOperation::Min,
        Operation::Max => MTLBlendOperation::Max,
    }
}

/// Blend factors; dual-source factors are refused by
/// [`format::check_blend`].
fn factor(factor: Factor) -> MTLBlendFactor {
    match factor {
        Factor::Zero => MTLBlendFactor::Zero,
        Factor::One => MTLBlendFactor::One,
        Factor::SourceColor => MTLBlendFactor::SourceColor,
        Factor::OneMinusSourceColor => MTLBlendFactor::OneMinusSourceColor,
        Factor::SourceAlpha => MTLBlendFactor::SourceAlpha,
        Factor::OneMinusSourceAlpha => MTLBlendFactor::OneMinusSourceAlpha,
        Factor::DestinationAlpha => MTLBlendFactor::DestinationAlpha,
        Factor::OneMinusDestinationAlpha => MTLBlendFactor::OneMinusDestinationAlpha,
        Factor::DestinationColor => MTLBlendFactor::DestinationColor,
        Factor::OneMinusDestinationColor => MTLBlendFactor::OneMinusDestinationColor,
        Factor::SourceAlphaSaturate => MTLBlendFactor::SourceAlphaSaturated,
        Factor::ConstantColor => MTLBlendFactor::BlendColor,
        Factor::OneMinusConstantColor => MTLBlendFactor::OneMinusBlendColor,
        Factor::ConstantAlpha => MTLBlendFactor::BlendAlpha,
        Factor::OneMinusConstantAlpha => MTLBlendFactor::OneMinusBlendAlpha,
        Factor::Source1Color => MTLBlendFactor::Source1Color,
        Factor::OneMinusSource1Color => MTLBlendFactor::OneMinusSource1Color,
        Factor::Source1Alpha => MTLBlendFactor::Source1Alpha,
        Factor::OneMinusSource1Alpha => MTLBlendFactor::OneMinusSource1Alpha,
    }
}

fn new_target(
    gpu: &Gpu,
    pixel_format: MTLPixelFormat,
    width: u32,
    height: u32,
) -> Result<Retained<MetalTexture>, Unmodeled> {
    // SAFETY: plain descriptor construction.
    let descriptor = unsafe {
        MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
            pixel_format,
            width as usize,
            height as usize,
            false,
        )
    };
    descriptor.setStorageMode(MTLStorageMode::Shared);
    descriptor.setUsage(
        MTLTextureUsage::RenderTarget
            | MTLTextureUsage::ShaderRead
            | MTLTextureUsage::PixelFormatView,
    );
    gpu.device
        .newTextureWithDescriptor(&descriptor)
        .ok_or_else(|| Unmodeled("Metal: render target allocation failed".into()))
}

/// Loads `rect` of `window` into the texture.
fn upload_surface(texture: &MetalTexture, window: &mut Window, rect: Rect) {
    let bpp = window.surface().bytes_per_pixel as usize;
    let row = (rect.x1 - rect.x0) as usize * bpp;
    let mut pixels = vec![0u8; row * (rect.y1 - rect.y0) as usize];
    window.read_pixels(rect, &mut pixels);
    // SAFETY: the buffer holds the region's rows.
    unsafe {
        texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
            region(rect),
            0,
            NonNull::new(pixels.as_mut_ptr().cast()).expect("non-null"),
            row,
        );
    }
}

/// A sampled texture: the decoded levels from the base level to the
/// maximum level as RGBA32Float.
fn upload_texture(gpu: &Gpu, texture: &Texture) -> Result<Retained<MetalTexture>, Unmodeled> {
    let tic = texture.tic();
    let levels = tic.max_level - tic.base_level + 1;
    let (width, height, _, _) = texture
        .level(tic.base_level)
        .ok_or_else(|| Unmodeled("Metal: texture base level".into()))?;
    // SAFETY: plain descriptor construction.
    let descriptor = unsafe {
        MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
            MTLPixelFormat::RGBA32Float,
            width as usize,
            height as usize,
            levels > 1,
        )
    };
    // SAFETY: the level count fits the base level's dimensions (the TIC
    // was validated by the decoder).
    unsafe { descriptor.setMipmapLevelCount(levels as usize) };
    descriptor.setStorageMode(MTLStorageMode::Shared);
    descriptor.setUsage(MTLTextureUsage::ShaderRead);
    let metal = gpu
        .device
        .newTextureWithDescriptor(&descriptor)
        .ok_or_else(|| Unmodeled("Metal: texture allocation failed".into()))?;
    for level in 0..levels {
        let (w, h, _, texels) = texture
            .level(tic.base_level + level)
            .ok_or_else(|| Unmodeled("Metal: texture level".into()))?;
        // SAFETY: `texels` holds the level's w*h RGBA32F texels (2D, one
        // layer: checked by `format::check_texture`).
        unsafe {
            metal.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                region(Rect {
                    x0: 0,
                    y0: 0,
                    x1: w,
                    y1: h,
                }),
                level as usize,
                NonNull::new(texels.as_ptr() as *mut _).expect("non-null"),
                w as usize * 16,
            );
        }
    }
    Ok(metal)
}

fn build_sampler(gpu: &Gpu, texture: &Texture, tsc: &Tsc) -> Retained<Sampler> {
    let descriptor = MTLSamplerDescriptor::new();
    let filter = |linear: bool| {
        if linear {
            MTLSamplerMinMagFilter::Linear
        } else {
            MTLSamplerMinMagFilter::Nearest
        }
    };
    descriptor.setMagFilter(filter(tsc.mag_linear));
    descriptor.setMinFilter(filter(tsc.min_filter != 1));
    descriptor.setMipFilter(match tsc.mip_filter {
        1 => MTLSamplerMipFilter::NotMipmapped,
        2 => MTLSamplerMipFilter::Nearest,
        _ => MTLSamplerMipFilter::Linear,
    });
    let address = |wrap: u32| match wrap {
        0 => MTLSamplerAddressMode::Repeat,
        1 => MTLSamplerAddressMode::MirrorRepeat,
        2 => MTLSamplerAddressMode::ClampToEdge,
        3 => MTLSamplerAddressMode::ClampToBorderColor,
        _ => MTLSamplerAddressMode::MirrorClampToEdge,
    };
    descriptor.setSAddressMode(address(tsc.wrap[0]));
    descriptor.setTAddressMode(address(tsc.wrap[1]));
    descriptor.setBorderColor(
        format::border_color(tsc).unwrap_or(MTLSamplerBorderColor::TransparentBlack),
    );
    let (min, max) = texture.lod_range(tsc);
    descriptor.setLodMinClamp(min);
    descriptor.setLodMaxClamp(max);
    gpu.device
        .newSamplerStateWithDescriptor(&descriptor)
        .expect("sampler state")
}

fn buffer(gpu: &Gpu, data: &[u8]) -> Result<Retained<MetalBuffer>, Unmodeled> {
    // Metal refuses empty buffers.
    let padded;
    let data = if data.is_empty() {
        padded = [0u8; 16];
        &padded[..]
    } else {
        data
    };
    // SAFETY: `data` is valid for its length; Metal copies it.
    unsafe {
        gpu.device.newBufferWithBytes_length_options(
            NonNull::new(data.as_ptr() as *mut _).expect("non-null"),
            data.len(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or_else(|| Unmodeled("Metal: buffer allocation failed".into()))
}

/// A shader stage to bind data for.
#[derive(Clone, Copy)]
enum Stage {
    Vertex,
    Fragment,
}

/// Binds `data` at buffer `index` of `stage`: copied into the command
/// stream when small (no allocation, as Metal advises below 4 KiB), as a
/// buffer of its own otherwise.
///
/// # Safety
///
/// `index` is within the stage function's buffer bindings.
unsafe fn bind(
    gpu: &Gpu,
    encoder: &Encoder,
    stage: Stage,
    data: &[u8],
    index: usize,
) -> Result<(), Unmodeled> {
    const INLINE: usize = 4096;
    if !data.is_empty() && data.len() <= INLINE {
        let bytes = NonNull::new(data.as_ptr() as *mut _).expect("non-null");
        // SAFETY: `data` is valid for its length; Metal copies it.
        unsafe {
            match stage {
                Stage::Vertex => encoder.setVertexBytes_length_atIndex(bytes, data.len(), index),
                Stage::Fragment => {
                    encoder.setFragmentBytes_length_atIndex(bytes, data.len(), index)
                }
            }
        }
        return Ok(());
    }
    let buffer = buffer(gpu, data)?;
    // SAFETY: Metal retains the buffer; `index` per the caller.
    unsafe {
        match stage {
            Stage::Vertex => encoder.setVertexBuffer_offset_atIndex(Some(&buffer), 0, index),
            Stage::Fragment => encoder.setFragmentBuffer_offset_atIndex(Some(&buffer), 0, index),
        }
    }
    Ok(())
}

fn bytes_of(values: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding and any byte is a valid u8.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

fn bytes_of_u32(values: &[u32]) -> &[u8] {
    // SAFETY: u32 has no padding and any byte is a valid u8.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

fn region(rect: Rect) -> MTLRegion {
    MTLRegion {
        origin: MTLOrigin {
            x: rect.x0 as usize,
            y: rect.y0 as usize,
            z: 0,
        },
        size: MTLSize {
            width: (rect.x1 - rect.x0) as usize,
            height: (rect.y1 - rect.y0) as usize,
            depth: 1,
        },
    }
}

fn union(a: Rect, b: Rect) -> Rect {
    Rect {
        x0: a.x0.min(b.x0),
        y0: a.y0.min(b.y0),
        x1: a.x1.max(b.x1),
        y1: a.y1.max(b.y1),
    }
}

/// The guest memory a surface may occupy: from its address to the end of
/// its last row (or block of rows).
fn extent(surface: &Surface) -> (u64, u64) {
    use super::surface::Layout;
    let size = match surface.layout {
        Layout::Pitch { pitch } => u64::from(pitch) * u64::from(surface.height),
        Layout::BlockLinear(block) => super::texture::layout::block_linear_size(
            surface.width * surface.bytes_per_pixel,
            surface.height,
            1,
            block,
        ),
    };
    (surface.address, surface.address + size)
}

fn overlap(a: (u64, u64), b: (u64, u64)) -> bool {
    a.0 < b.1 && b.0 < a.1
}

/// Engine memory for a channel with a Metal renderer: flushes dirty
/// render targets before any access that overlaps them.
pub struct Coherent<'a> {
    pub memory: &'a dyn GpuMemory,
    pub renderer: &'a Renderer,
    pub writes: &'a Writes,
}

impl GpuMemory for Coherent<'_> {
    fn read(&self, address: u64, buffer: &mut [u8]) -> Option<()> {
        self.renderer
            .before_access(self.memory, self.writes, address, buffer.len())
            .ok()?;
        self.memory.read(address, buffer)
    }

    fn write(&self, address: u64, data: &[u8]) -> Option<()> {
        self.renderer
            .before_access(self.memory, self.writes, address, data.len())
            .ok()?;
        self.memory.write(address, data)
    }
}
