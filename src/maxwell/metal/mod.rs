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
//! # Guest memory stays truthful
//!
//! A render target used by Metal is *resident*: a Metal texture holding
//! the surface, plus a mirror of the surface's bytes as guest memory had
//! them when the two were last made equal. Rendering makes the resident
//! *dirty* (the Metal texture is newer than guest memory, over a known
//! rectangle). Dirty pixels are written back to guest memory, in the
//! surface's own layout:
//!
//! - before anything the channel does can let another agent observe its
//!   progress: semaphore releases, reports and syncpoint increments of the
//!   engines (they call [`Renderer::flush`]), and every host point the
//!   pushbuffer executor calls [`super::Engines::flush`] for;
//! - before any other access of the channel's engines to memory that
//!   overlaps a dirty target: all engine memory accesses go through
//!   [`Coherent`], which flushes first. That covers the software
//!   renderer's own reads, copies, 2D blits, inline uploads, texture
//!   sampling of a render target, and constant and vertex data.
//!
//! Before a resident is rendered to again after a write-back, its mirror
//! is compared with guest memory; if anything (the CPU, another channel,
//! an engine of this one) changed the surface, the Metal texture is
//! reloaded. Between write-backs, a guest CPU writing to a surface the
//! GPU is rendering to races with the GPU, as on hardware.

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
use super::raster::{ColorOutput, Triangle};
use super::resources::DrawTextures;
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
    /// Whether `mirror` was compared with guest memory since the last
    /// write-back.
    checked: bool,
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
    pipelines: HashMap<PipelineKey, Retained<Pipeline>>,
    clear_pipelines: HashMap<(usize, u8), Retained<Pipeline>>,
    /// Uploaded textures by decoded texture (kept alive with it).
    textures: HashMap<usize, (Arc<Texture>, Retained<MetalTexture>)>,
    samplers: HashMap<[u32; 8], Retained<Sampler>>,
    fallbacks: HashMap<String, u64>,
    draws: u64,
}

// SAFETY: the Metal objects of a channel (textures, buffers, pipeline
// states, command buffers, encoders) are only used under the renderer's
// lock, by one thread at a time; pipeline and sampler states are
// thread-safe themselves [public: Apple Metal Programming Guide].
unsafe impl Send for State {}

#[derive(Clone, PartialEq, Eq, Hash)]
struct PipelineKey {
    program: usize,
    /// Per attachment: pixel format, blend, write mask, program output.
    attachments: Vec<(usize, Option<Blend>, u8, usize)>,
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

    /// Writes every dirty target back to guest memory.
    pub fn flush(&self, memory: &dyn GpuMemory) -> Result<(), Unmodeled> {
        use std::sync::atomic::Ordering;
        let mut state = self.state.lock().unwrap();
        state.finish();
        self.writing_back.store(true, Ordering::Relaxed);
        let mut result = Ok(());
        for resident in &mut state.residents {
            if let Some(rect) = resident.dirty.take()
                && result.is_ok()
            {
                result = resident.write_back(rect, memory);
            }
            resident.checked = false;
        }
        self.writing_back.store(false, Ordering::Relaxed);
        result
    }

    /// Flushes if `length` bytes at `address` overlap a dirty target.
    fn before_access(
        &self,
        memory: &dyn GpuMemory,
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
            self.flush(memory)?;
        }
        Ok(())
    }

    /// A draw on Metal; `Ok(false)` if Metal cannot do it (nothing was
    /// changed then).
    pub fn draw(&self, memory: &dyn GpuMemory, draw: &Draw<'_>) -> Result<bool, Unmodeled> {
        // Inputs that come from memory are gathered before the state is
        // locked: reading them may flush.
        let translated = {
            let mut state = self.state.lock().unwrap();
            let key = Arc::as_ptr(draw.program) as usize;
            state
                .programs
                .entry(key)
                .or_insert_with(|| Program2 {
                    _program: Arc::clone(draw.program),
                    translated: msl::translate(draw.program).map(Arc::new),
                })
                .translated
                .clone()
        };
        let translated = match translated {
            Ok(t) => t,
            Err(reason) => return self.fall_back(format!("pixel program: {reason}")),
        };
        // Constant buffer words read statically must be readable.
        let mut constants: HashMap<u32, Vec<u8>> = HashMap::new();
        for &(slot, offset) in &translated.constants {
            let Some((data, size)) = draw.constants.get(slot as usize) else {
                return self.fall_back(format!("constant buffer slot {slot}"));
            };
            let end = offset as usize + 4;
            if end > data.len() && end <= *size {
                return self.fall_back("constant read past the mapped part".into());
            }
            let buffer = constants.entry(slot).or_insert_with(|| data.clone());
            if buffer.len() < end {
                buffer.resize(end, 0);
            }
        }
        // Textures and samplers.
        let mut bound = Vec::new();
        for (slot, &handle_offset) in translated.textures.iter().enumerate() {
            let Some((data, _)) = draw.constants.get(draw.texture_buffer_index as usize) else {
                return self.fall_back("texture handle buffer unbound".into());
            };
            let Some(bytes) = data.get(handle_offset as usize..handle_offset as usize + 4) else {
                return self.fall_back("texture handle outside the constant buffer".into());
            };
            let handle = u32::from_le_bytes(bytes.try_into().expect("4 bytes"));
            let (texture, tsc) = draw.textures.resolve(handle)?;
            if let Err(reason) = format::check_texture(&texture, &tsc) {
                return self.fall_back(reason);
            }
            let bias = if translated.implicit_lod[slot] {
                tsc.lod_bias
            } else {
                0.0
            };
            bound.push((texture, tsc, bias));
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
        };
        let pipeline = match state.pipelines.get(&key) {
            Some(p) => p.clone(),
            None => {
                let outputs: Vec<usize> = key.attachments.iter().map(|a| a.3).collect();
                let source = translated.source(draw.program, &outputs);
                let pipeline = build_pipeline(gpu, &source, "fragment_main", &key.attachments)?;
                state.pipelines.insert(key.clone(), pipeline.clone());
                pipeline
            }
        };
        // Textures and samplers on the GPU.
        let mut gpu_textures = Vec::new();
        for (texture, tsc, _) in &bound {
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
        let biases: Vec<f32> = bound.iter().map(|b| b.2).chain([0.0]).collect();
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
        let vertex_buffer = buffer(gpu, bytes_of(&positions))?;
        let plane_buffer = buffer(gpu, bytes_of(&planes))?;
        let bias_buffer = buffer(gpu, bytes_of(&biases))?;
        // SAFETY: buffers and textures outlive the command buffer (Metal
        // retains them), indices are within the function's bindings.
        unsafe {
            encoder.setVertexBuffer_offset_atIndex(Some(&vertex_buffer), 0, 0);
            encoder.setFragmentBuffer_offset_atIndex(Some(&plane_buffer), 0, 0);
            encoder.setFragmentBuffer_offset_atIndex(Some(&bias_buffer), 0, 1);
            for (slot, data) in &constants {
                let b = buffer(gpu, data)?;
                encoder.setFragmentBuffer_offset_atIndex(Some(&b), 0, 2 + *slot as usize);
            }
            for (i, (texture, sampler)) in gpu_textures.iter().enumerate() {
                encoder.setFragmentTexture_atIndex(Some(texture), i);
                encoder.setFragmentSamplerState_atIndex(Some(sampler), i);
            }
            encoder.drawPrimitives_vertexStart_vertexCount(
                MTLPrimitiveType::Triangle,
                0,
                draw.triangles.len() * 3,
            );
        }
        for &index in &residents {
            let resident = &mut state.residents[index];
            resident.dirty = Some(match resident.dirty {
                Some(d) => union(d, scissor),
                None => scissor,
            });
        }
        state.draws += 1;
        Ok(true)
    }

    /// A color clear on Metal: `components` (bit i: component i) of
    /// `rect` set to `color`, encoded as the format stores it (no sRGB
    /// conversion, as `CLEAR_SURFACE` writes the value raw).
    pub fn clear(
        &self,
        memory: &dyn GpuMemory,
        surface: Surface,
        format: ColorFormat,
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
        let residents = self.residents(memory, &[(surface, format)])?;
        let mut state = self.state.lock().unwrap();
        let gpu = Self::gpu();
        let key = (pixel_format.0, components);
        let pipeline = match state.clear_pipelines.get(&key) {
            Some(p) => p.clone(),
            None => {
                let p = build_pipeline(
                    gpu,
                    CLEAR_SOURCE,
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
        let color_buffer = buffer(gpu, bytes_of(&color))?;
        // SAFETY: as in `draw`.
        unsafe {
            encoder.setFragmentBuffer_offset_atIndex(Some(&color_buffer), 0, 0);
            encoder.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3);
        }
        let resident = &mut state.residents[residents[0]];
        resident.dirty = Some(match resident.dirty {
            Some(d) => union(d, rect),
            None => rect,
        });
        Ok(true)
    }

    /// The residents for `targets`, created, checked against guest memory
    /// or reloaded as needed; their indices.
    fn residents(
        &self,
        memory: &dyn GpuMemory,
        targets: &[(Surface, ColorFormat)],
    ) -> Result<Vec<usize>, Unmodeled> {
        let mut indices = Vec::new();
        for &(surface, format) in targets {
            // A resident of other dimensions or format at overlapping
            // memory is written back and dropped.
            let stale: bool = {
                let state = self.state.lock().unwrap();
                state.residents.iter().any(|r| {
                    (r.surface != surface || r.format != format)
                        && overlap(r.extent(), extent(&surface))
                })
            };
            if stale {
                self.flush(memory)?;
                let mut state = self.state.lock().unwrap();
                state.residents.retain(|r| {
                    !((r.surface != surface || r.format != format)
                        && overlap(r.extent(), extent(&surface)))
                });
            }
            let found = {
                let state = self.state.lock().unwrap();
                state
                    .residents
                    .iter()
                    .position(|r| r.surface == surface && r.format == format)
                    .map(|i| {
                        (
                            i,
                            state.residents[i].checked || state.residents[i].dirty.is_some(),
                        )
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
                    let mut state = self.state.lock().unwrap();
                    let resident = &mut state.residents[index];
                    if window.data() != resident.mirror.data() {
                        upload_surface(&resident.texture, &mut window, full);
                        resident.mirror = window;
                    }
                    resident.checked = true;
                    indices.push(index);
                }
                None => {
                    let pixel_format = format::pixel_format(format).expect("checked by the caller");
                    let mut window = Window::load(memory, surface, full)?;
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
                        checked: true,
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
        if let Some(frame) = self.frame.take() {
            if let Some((encoder, _)) = frame.pass {
                encoder.endEncoding();
            }
            frame.command.commit();
            frame.command.waitUntilCompleted();
        }
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

fn build_pipeline(
    gpu: &Gpu,
    source: &str,
    fragment: &str,
    attachments: &[(usize, Option<Blend>, u8, usize)],
) -> Result<Retained<Pipeline>, Unmodeled> {
    let options = MTLCompileOptions::new();
    options.setMathMode(MTLMathMode::Safe);
    let library = gpu
        .device
        .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&options))
        .map_err(|e| Unmodeled(format!("Metal: shader compilation failed: {e}\n{source}")))?;
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

/// Compiles `source` into a pipeline with `outputs` RGBA8 attachments, for
/// tests of the translation.
#[cfg(test)]
pub(crate) fn compile(source: &str, outputs: usize) -> Result<(), String> {
    let gpu = GPU.as_ref().ok_or("no Metal device")?;
    let attachments: Vec<_> = (0..outputs)
        .map(|i| (MTLPixelFormat::RGBA8Unorm.0, None, 15, i))
        .collect();
    build_pipeline(gpu, source, "fragment_main", &attachments)
        .map(drop)
        .map_err(|e| e.0)
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

fn bytes_of(values: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding and any byte is a valid u8.
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
}

impl GpuMemory for Coherent<'_> {
    fn read(&self, address: u64, buffer: &mut [u8]) -> Option<()> {
        self.renderer
            .before_access(self.memory, address, buffer.len())
            .ok()?;
        self.memory.read(address, buffer)
    }

    fn write(&self, address: u64, data: &[u8]) -> Option<()> {
        self.renderer
            .before_access(self.memory, address, data.len())
            .ok()?;
        self.memory.write(address, data)
    }
}
