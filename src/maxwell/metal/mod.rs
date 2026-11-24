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
//! interpolation planes, draws each over its pixel bounding box, keeps
//! the pixels the software rasterizer's exact edge test keeps, runs the
//! pixel program translated to MSL ([`super::shader::msl`]), which
//! evaluates the planes exactly as the interpreter does, samples textures
//! decoded by [`super::texture`], tests depth against the depth buffer
//! held as its raw words (read and written in the fragment function with
//! the format's own encoding), and blends into render targets held as
//! Metal textures. A draw or clear the Metal path cannot express (an
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
//! rectangle). A write-back waits for the GPU and stores the dirty pixels
//! in guest memory, in the surface's own layout. That happens:
//!
//! - when the channel signals: semaphore releases, reports and syncpoint
//!   increments of its engines and of the pushbuffer executor
//!   ([`super::Hooks::signal`], [`super::Engines::flush`]), except for
//!   the targets held on demand (below);
//! - before any other access of the channel's engines to memory that
//!   overlaps a dirty target: all engine memory accesses go through
//!   [`Coherent`], which writes back first. That covers copies, 2D blits
//!   the GPU does not do ([`blit`]), inline uploads, texture sampling of
//!   a render target that is not sampled on the GPU (below), and constant
//!   and vertex data;
//! - before a draw or clear Metal cannot express runs in the software
//!   renderer, and before a machine snapshot ([`Renderer::settle`]).
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
//!
//! # The GPU works while the frame is recorded
//!
//! Draws and clears are recorded into command buffers that are committed
//! as recording goes: whenever a render pass ends (other attachments, a
//! compute pass, or [`PASS_DRAWS`] draws in it) the buffer is committed and
//! the next pass starts another, so the GPU runs the frame's first passes
//! while the CPU records its later ones. Nothing waits at a commit. A
//! write-back commits what is recorded and waits for the buffer committed
//! last: buffers of one queue run in commit order, so everything recorded
//! is done then, and most of it was done already. Draw data lives in the
//! arena until such a wait ([`Arena`]); a signal with nothing to write
//! back waits only once the arena has grown ([`ARENA_UNWAITED`]) or a
//! draw's texture faults are to be read.
//!
//! # Targets held on demand
//!
//! Depth buffers and multisampled targets ([`Resident::on_demand`]) are
//! not written back at signals: only when an engine of the channel
//! accesses their memory ([`Coherent`]: a 2D blit or copy the GPU does
//! not do, texture sampling, the software renderer), before a snapshot,
//! and when a surface of another shape is rendered at their memory.
//! \[inference: the guest CPU has no use for these bytes. GM20B keeps
//! depth and multisampled color compressed, with the compression state
//! outside the surface (nvgpu's comptags and compression bit cache,
//! RECALLED), which is why the class has `DECOMPRESS_SURFACE` and
//! `DECOMPRESS_ZETA_SURFACE` (NVIDIA open-gpu-doc `clb197.h`; NVN issues
//! the former on its present path, see `three_d`): what a CPU read of
//! such a surface returns between those is not the pixels. No guest seen
//! reads them; every use goes through an engine.\]
//!
//! What the guest may do with the memory is give it another use (NVN
//! sub-allocates memory pools; no unmap or remap tells). So a target held
//! on demand is written back only while guest memory still holds what the
//! mirror does, bit for bit: nothing wrote there since the two were last
//! equal. Otherwise (other contents, or the memory is no longer mapped)
//! the memory has a new owner whose data stands, and what was rendered is
//! forgotten: the resident is left stale and loaded from memory when next
//! drawn to. The comparison is skipped while the resident was made equal
//! to memory in the current epoch of [`Writes`]: until the channel
//! signals, the guest may not touch memory its submitted work uses.
//!
//! # Sampling render targets
//!
//! A texture whose levels (some or all) are current residents of a color
//! format with the header's bytes is sampled from a [`Composite`]: a
//! Metal texture whose levels are filled from those residents on the GPU
//! each time they were rendered to, and from guest memory for the other
//! levels. Guest memory is not involved: no write-back, decode or upload
//! of the rendered levels (a guest may draw level 0 of a texture atlas
//! every frame). A draw that renders to a level it samples reads guest
//! memory instead, after the write-back of what was rendered before it.

//!
//! # Multisampled targets
//!
//! Guest memory holds a multisampled target's samples as the texels of a
//! larger surface ([`super::multisample`]). Its resident is a 2D array
//! texture of one layer per sample at the size in pixels: a draw runs
//! its fragment function once per sample into that sample's layer, each
//! run evaluating the pixel as the software back end does once
//! ([`Translated::source`]), so quads, derivatives and the program's
//! values are those of whole pixels while coverage, depth, stencil and
//! blending are per sample. Write-back and loading interleave the layers'
//! pixels into the texels holding them and back.
//!
//! A draw is issued once per layer. Rendering into a Metal multisample
//! texture with the guest's sample positions
//! (`MTLRenderPassDescriptor::setSamplePositions`) would issue it once
//! and run the vertex stage and the rasterizer once; that is not done.
//! \[measurement, a guest drawing with 2X1_D3D targets on an M5, `sample`
//! of the channel's thread: with the frame committed pass by pass, the
//! thread waits for the GPU at the frame's last signal only, 0.9 ms a
//! frame at 47 fps on a loaded host and 2.7 ms at the guest's 60 fps,
//! where it is idle half of the time; the draw calls of both layers
//! together are about 1% of its samples (61 of 5165). A multisample
//! texture takes no `replaceRegion`, `getBytes` or write from a compute
//! kernel (RECALLED from Apple's `MTLTexture` documentation, not tried
//! here), so loading, the packing kernel ([`State::pack`]) and the blit
//! kernel ([`blit`]) would each need a second form, and both fragment
//! generators a per-sample one.\]
pub mod blit;
mod format;

use std::borrow::Cow;
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::{Arc, LazyLock, Mutex};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLBlendFactor, MTLBlendOperation, MTLBuffer, MTLColorWriteMask, MTLCommandBuffer,
    MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLCullMode, MTLDepthClipMode,
    MTLDevice, MTLGPUFamily, MTLIndexType, MTLLibrary, MTLLoadAction, MTLMathMode, MTLOrigin,
    MTLPixelFormat, MTLPrimitiveTopologyClass, MTLPrimitiveType, MTLRegion,
    MTLRenderCommandEncoder, MTLRenderPassDescriptor, MTLRenderPipelineDescriptor,
    MTLRenderPipelineState, MTLResourceOptions, MTLSamplerAddressMode,
    MTLSamplerBorderColor, MTLSamplerDescriptor, MTLSamplerMinMagFilter, MTLSamplerMipFilter,
    MTLSamplerState, MTLScissorRect, MTLSize, MTLStorageMode, MTLStoreAction, MTLTexture,
    MTLTextureDescriptor, MTLTextureSwizzle, MTLTextureSwizzleChannels, MTLTextureType,
    MTLTextureUsage, MTLViewport, MTLWinding,
};

use super::blend::{Blend, Factor, Operation};
use super::lru::Lru;
use super::multisample::Samples;
use super::raster::{ColorOutput, Compare, DepthOutput, StencilFace, StencilOp, Triangle};
use super::resources::{DrawTextures, RawTexture, Stamp, Writes};
use super::shader::Program;
use super::shader::msl::{self, TextureKind, Translated};
use super::surface::{Rect, Surface, Window};
use super::sampling::Target;
use super::texture::format::{ColorFormat, ComponentType, DepthFormat};
use super::texture::{Texture, Tic, Tsc};
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
type ComputePipeline = ProtocolObject<dyn MTLComputePipelineState>;
/// A draw's texture fault words, with its unbound slots and the refusal
/// an executed sample of each is.
type Faults = (Retained<MetalBuffer>, Vec<(usize, String)>);

/// The host GPU, shared by all channels.
struct Gpu {
    device: Retained<Device>,
    queue: Retained<Queue>,
    /// Whether fragment functions can read color attachments (Apple GPU
    /// families: programmable blending [public: Apple, Metal feature set
    /// tables, "Programmable blending": Apple1 and later]), which the depth
    /// test is made of (see [`Translated::source`]).
    fetch: bool,
}

// SAFETY: Metal devices and command queues are thread-safe [public: Apple
// Metal Programming Guide, "Transient and Non-transient Objects in
// Metal": `MTLDevice`, `MTLCommandQueue` may be used from any thread].
unsafe impl Send for Gpu {}
unsafe impl Sync for Gpu {}

static GPU: LazyLock<Option<Gpu>> = LazyLock::new(|| {
    let device = MTLCreateSystemDefaultDevice()?;
    let queue = device.newCommandQueue()?;
    let fetch = device.supportsFamily(MTLGPUFamily::Apple1);
    Some(Gpu {
        device,
        queue,
        fetch,
    })
});

/// Whether a Metal device is present.
pub fn available() -> bool {
    GPU.is_some()
}

/// What a resident holds: a color target, or a depth buffer as its raw
/// words (see [`format::depth_pixel_format`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Content {
    Color(ColorFormat),
    Depth(DepthFormat),
}

impl Content {
    /// The Metal pixel format of the resident's texture, if Metal has one
    /// with the surface's bytes.
    fn pixel_format(self) -> Option<MTLPixelFormat> {
        match self {
            Content::Color(format) => format::pixel_format(format),
            Content::Depth(format) => Some(format::depth_pixel_format(format)),
        }
    }
}

/// A render target or depth buffer held on the GPU.
struct Resident {
    /// The surface in guest memory, in texels (see `multisample`).
    surface: Surface,
    format: Content,
    /// log2 of its samples per pixel in x and y. With more than one,
    /// `texture` is a 2D array of one layer per sample at the size in
    /// pixels (see `Translated::source`); dirty rectangles are pixels.
    log2: [u32; 2],
    texture: Retained<MetalTexture>,
    /// The whole surface as guest memory held it when it and `texture`
    /// were last equal.
    mirror: Window,
    /// Pixels rendered since the last write-back.
    dirty: Option<Rect>,
    /// When `mirror` was last known to equal guest memory.
    stamp: Option<Stamp>,
    /// [`State::clock`] when a draw or clear last used it.
    used: u64,
    /// Tells the texture's content from any it held before: a count
    /// ([`State::versions`]) taken whenever it is rendered to or loaded.
    version: u64,
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
    /// Keeps the program alive so its address is not reused as a key;
    /// once this is the last reference (the program cache dropped it),
    /// the translation and its pipelines are dropped ([`State::trim`]).
    program: Arc<Program>,
    translated: Result<Arc<Translated>, String>,
}

/// A vertex program's translation, kept as [`Program2`].
struct VertexProgram {
    program: Arc<Program>,
    translated: Result<Arc<msl::TranslatedVertex>, String>,
}

/// Compiled MSL libraries kept \[assumption: a library is one translated
/// pixel program, tens to hundreds of KiB of host memory; this is more
/// than the programs a HOME menu session draws with\].
const LIBRARY_LIMIT: usize = 512;
/// Render pipeline states kept (programs by attachment formats and
/// blending) \[assumption: as [`LIBRARY_LIMIT`]\].
const PIPELINE_LIMIT: usize = 1024;
/// Sampler states kept \[assumption: a sampler state is a few hundred
/// bytes; guests use a handful of distinct samplers\].
const SAMPLER_LIMIT: usize = 512;
/// Host memory the render targets held on the GPU may take (texture and
/// mirror) before the least recently used clean ones are dropped
/// \[assumption: sized as the texture cache, `resources::TEXTURE_BUDGET`;
/// a dropped target is loaded from guest memory when next drawn to\].
const RESIDENT_BUDGET: usize = 512 << 20;

/// Draws and clears one render pass takes before it is ended and
/// committed, to be continued in another with the same attachments: the
/// GPU starts on a pass only once it is committed, so a frame drawn in
/// one long pass would leave it idle until the frame is recorded, while
/// each pass boundary costs it a store and a load of the attachments.
/// \[measurement, a guest drawing at its 60 fps on an M5,
/// share of the channel's thread spent waiting for the GPU in `sample`
/// runs of 8 s: 52% at 16 draws a pass, 33-37% at 32, 29-32% at 64 (three
/// runs), 28-44% at 128, 42-55% at 256, 45% with passes never split\]
const PASS_DRAWS: u32 = 64;

/// The renderer's host objects. Every cache is bounded: by
/// least-recent use (`maxwell::lru`) for libraries, pipelines and
/// samplers, by the software caches' own bounds for programs and
/// textures (an entry here goes once its program or texture is dropped
/// there), and by [`RESIDENT_BUDGET`] for render targets.
/// `clear_pipelines` holds at most one pipeline per pixel format, write
/// mask (mask `u8::MAX`: the depth clear) and layering (multisampled
/// targets).
struct State {
    residents: Vec<Resident>,
    frame: Option<Frame>,
    /// The command buffer committed last, until it is known to have
    /// completed ([`State::wait`]).
    submitted: Option<Retained<CommandBuffer>>,
    /// Draws and clears recorded in the open pass ([`PASS_DRAWS`]).
    pass_draws: u32,
    programs: HashMap<usize, Program2>,
    vertex_programs: HashMap<usize, VertexProgram>,
    /// Reasons draws shaded their vertices on the CPU (`pipeline::native`),
    /// counted.
    cpu_vertices: HashMap<String, u64>,
    /// Compiled MSL by source: programs drawn with different attachment
    /// formats or blending share their functions.
    libraries: Lru<String, Retained<Library>>,
    pipelines: Lru<PipelineKey, Retained<Pipeline>>,
    clear_pipelines: HashMap<(usize, u8, bool), Retained<Pipeline>>,
    /// Uploaded textures by decoded texture and kind (kept alive with
    /// the texture while the texture cache holds it too).
    textures: HashMap<(usize, TextureKind), (Arc<Texture>, Retained<MetalTexture>)>,
    /// Textures uploaded in their own format, by the bytes they were
    /// made of and kind (kept as the decoded ones are).
    native: HashMap<(usize, TextureKind), (Arc<RawTexture>, Retained<MetalTexture>)>,
    /// Bytes of guest textures uploaded and not known to be dropped
    /// (`trim_textures` counts them anew).
    texture_bytes: usize,
    /// By TSC words and the level-of-detail range the texture's header
    /// leaves ([`Tic::lod_range`]).
    samplers: Lru<([u32; 8], [u32; 2]), Retained<Sampler>>,
    /// Textures with levels held as residents, by header words.
    composites: HashMap<[u32; 8], Composite>,
    /// Counts changes of residents' textures ([`Resident::version`]).
    versions: u64,
    convert_pipeline: Option<Retained<ComputePipeline>>,
    /// Counts uses of residents (`Resident::used`).
    clock: u64,
    fallbacks: HashMap<String, u64>,
    draws: u64,
    /// Texture fault words of the draws recorded since the last flush
    /// that bound placeholders, with each unbound slot and the refusal a
    /// sample of it is (see `shader::msl` `texture.rs`).
    faults: Vec<Faults>,
    /// What an unbound texture slot of each kind is bound to: a 1x1
    /// texture and a sampler, never read into a result that reaches
    /// guest memory.
    placeholders: HashMap<TextureKind, (Retained<MetalTexture>, Retained<Sampler>)>,
    /// Buffer space for the frame's draw data.
    arena: Arena,
    /// Global memory bound for pixel programs, by the regions it holds
    /// ([`Renderer::global_memory`]).
    globals: Lru<Vec<(u64, u32)>, GlobalMemory>,
    /// Bytes of global memory uploaded since the last wait: buffers
    /// replaced meanwhile live on until the work using them is done.
    globals_pending: usize,
    /// The word a global memory load outside the bound memory sets (made
    /// once, looked at after every wait).
    global_fault: Option<Retained<MetalBuffer>>,
    /// The packing kernels (`State::pack`), plain and layered.
    pack_pipelines: HashMap<bool, Retained<ComputePipeline>>,
    /// The 2D blit kernel (`blit`), made once.
    blit_pipeline: Option<Retained<ComputePipeline>>,
}

impl Default for State {
    fn default() -> Self {
        State {
            residents: Vec::new(),
            frame: None,
            submitted: None,
            pass_draws: 0,
            programs: HashMap::new(),
            vertex_programs: HashMap::new(),
            cpu_vertices: HashMap::new(),
            libraries: Lru::new(LIBRARY_LIMIT),
            pipelines: Lru::new(PIPELINE_LIMIT),
            clear_pipelines: HashMap::new(),
            textures: HashMap::new(),
            samplers: Lru::new(SAMPLER_LIMIT),
            clock: 0,
            fallbacks: HashMap::new(),
            draws: 0,
            faults: Vec::new(),
            placeholders: HashMap::new(),
            native: HashMap::new(),
            texture_bytes: 0,
            arena: Arena::default(),
            globals: Lru::new(GLOBAL_BUDGET),
            globals_pending: 0,
            global_fault: None,
            pack_pipelines: HashMap::new(),
            composites: HashMap::new(),
            versions: 0,
            convert_pipeline: None,
            blit_pipeline: None,
        }
    }
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
    /// Per color attachment: pixel format, blend, write mask, program
    /// output.
    attachments: Vec<(usize, Option<Blend>, u8, usize)>,
    /// Whether the function tests alpha (its function and reference are
    /// draw-time values).
    alpha_test: bool,
    /// The depth buffer's format, whether depth is written and whether
    /// the stencil test runs (comparisons, operations, masks and the
    /// clamp range are draw-time values).
    depth: Option<msl::DepthStage>,
    /// Multisampled targets (layered: one layer per sample).
    multisample: Option<msl::MultisampleStage>,
    /// A native draw's vertex function.
    native: Option<NativeKey>,
}

/// What a native draw's vertex function is made from: its program, the
/// constant buffer 1 its translation was made for, the attributes it
/// fetches and the output words of its clip distances.
type NativeKey = (usize, Option<Vec<u8>>, Vec<msl::Fetch>, Vec<u32>);

/// Where a draw's fragments come from.
pub enum Geometry<'a> {
    /// The software pipeline's set-up triangles, each drawn over its pixel
    /// bounding box and covered by its exact edge test.
    Triangles(&'a [Triangle]),
    /// Vertices shaded on the host GPU and rasterized by Metal.
    Native(&'a NativeDraw<'a>),
}

/// The vertex stage of a native draw (see `pipeline::native`).
pub struct NativeDraw<'a> {
    /// The vertex program.
    pub program: &'a Arc<Program>,
    /// Constant buffer contents of the vertex stage and their bound sizes.
    pub constants: &'a [(Vec<u8>, usize)],
    /// Attributes whose `SET_VERTEX_ATTRIBUTE_A` SOURCE is INACTIVE (bit
    /// per attribute): a program that reads one runs on the CPU, which
    /// refuses the read.
    pub inactive: u32,
    pub fetches: Vec<msl::Fetch>,
    /// By stream slot.
    pub streams: Vec<NativeStream>,
    /// How Metal's vertex ID names a vertex.
    pub vertices: NativeVertices,
    /// The triangles' vertices as Metal's indices, the provoking vertex
    /// first.
    pub indices: NativeIndices<'a>,
    /// The viewport words of `msl::NativeVertex::new` (without the layer).
    pub view: [f32; 9],
    /// The instance ID and the instance per-instance streams step by.
    pub instance: u32,
    pub fetch_instance: u32,
    /// Output words of the user clip distances.
    pub clip_words: Vec<u32>,
    /// Culled faces: 0 none, 1 front, 2 back.
    pub cull: u8,
    /// Front faces wind clockwise in Metal's framebuffer.
    pub front_clockwise: bool,
    /// Clip to the depth range (else clamp).
    pub depth_clip: bool,
    /// The polygon offset words of `native_depth_bias` (MSL prelude).
    pub depth_bias: [f32; 4],
}

/// How Metal's vertex ID gives a vertex's fetch index and vertex ID.
pub enum NativeVertices {
    /// It is the batch position: (fetch index, vertex ID) pairs by it.
    Listed(Vec<u32>),
    /// It is an index of the guest's index buffer: the fetch index is it
    /// plus `base_vertex`, the vertex ID it plus `id_base` (modulo 2^32).
    Indexed { base_vertex: u32, id_base: u32 },
}

/// The Metal indices of a native draw's triangles.
pub enum NativeIndices<'a> {
    /// Batch positions.
    Positions(Vec<u32>),
    /// Indices of the guest's index buffer as it holds them: two bytes
    /// each, or four (`wide`), little-endian.
    Buffer { bytes: Cow<'a, [u8]>, wide: bool },
}

/// A vertex stream's bytes from its first fetched element on.
pub struct NativeStream {
    pub bytes: Vec<u8>,
    pub first: u32,
    pub stride: u32,
    pub divisor: u32,
}

/// Which dirty residents a write-back takes.
#[derive(Clone, Copy)]
enum Scope {
    /// The channel signals: those the guest may read once it has
    /// (all but the ones held on demand, [`Resident::on_demand`]).
    Signal,
    /// Memory `start..end` is about to be accessed: those overlapping it.
    Range(u64, u64),
    /// All of them.
    All,
}

/// Arena chunks a channel's recorded work may fill without anything
/// having waited for the GPU \[assumption: a frame's draw data is a chunk
/// or two; work whose results nothing reads is still held to the GPU's
/// pace\].
const ARENA_UNWAITED: usize = 4;

/// The Metal renderer of one channel.
pub struct Renderer {
    state: Mutex<State>,
    /// Set while [`Renderer::flush`] writes back: its own writes pass
    /// [`Coherent`] unchecked (the state is locked by the flush).
    writing_back: std::sync::atomic::AtomicBool,
    /// Whether a resident may be dirty: set by every draw and clear, clear
    /// after a flush wrote them all back. While clear, an access needs no
    /// look at the residents ([`Renderer::before_access`]).
    any_dirty: std::sync::atomic::AtomicBool,
}

/// What a draw hands the Metal renderer once the software pipeline has
/// set it up.
pub struct Draw<'a> {
    pub program: &'a Arc<Program>,
    pub geometry: Geometry<'a>,
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
    /// The depth buffer and test (see `raster::Backend::depth`).
    pub depth: Option<(Surface, DepthOutput)>,
    /// Samples per pixel of the targets (see `pipeline::multisample`).
    pub multisample: super::pipeline::Multisample,
}

impl Renderer {
    /// A renderer, or `None` without a Metal device.
    pub fn new() -> Option<Renderer> {
        GPU.as_ref()?;
        Some(Renderer {
            state: Mutex::new(State::default()),
            writing_back: Default::default(),
            any_dirty: Default::default(),
        })
    }

    /// Drops everything held on the host GPU (render targets, uploaded
    /// textures, programs, pipelines, the frame being recorded), as a
    /// restored machine starts with: it is rebuilt from guest memory as
    /// draws need it. Only the fallback counts stay.
    pub fn drop_caches(&self) {
        let mut state = self.state.lock().unwrap();
        let State {
            fallbacks, draws, ..
        } = std::mem::take(&mut *state);
        *state = State {
            fallbacks,
            draws,
            ..State::default()
        };
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
    /// reported on stderr when first seen and again at every tenfold
    /// count, so the reasons that cost most stand out.
    fn fall_back(&self, reason: String) -> Result<bool, Unmodeled> {
        let mut state = self.state.lock().unwrap();
        let count = state.fallbacks.entry(reason.clone()).or_default();
        *count += 1;
        if *count == 1 {
            eprintln!("Metal renderer: left to the software renderer: {reason}");
        } else if count.is_multiple_of(10) && count.ilog10() == (*count - 1).ilog10() + 1 {
            eprintln!("Metal renderer: left to the software renderer {count} times: {reason}");
        }
        Ok(false)
    }

    /// Counts a draw left to the software renderer by its caller.
    pub fn note_fallback(&self, reason: &str) {
        let _ = self.fall_back(reason.to_string());
    }

    /// Counts a draw whose vertices the CPU shades rather than Metal
    /// (`pipeline::native`); each reason is reported once on stderr.
    pub fn note_cpu_vertices(&self, reason: &str) {
        let mut state = self.state.lock().unwrap();
        let count = state.cpu_vertices.entry(reason.to_string()).or_default();
        if *count == 0 {
            eprintln!("Metal renderer: vertices shaded on the CPU: {reason}");
        }
        *count += 1;
    }

    /// Counts of draws whose vertices the CPU shaded, by reason
    /// ([`Renderer::note_cpu_vertices`]).
    pub fn cpu_vertices(&self) -> Vec<(String, u64)> {
        let state = self.state.lock().unwrap();
        let mut list: Vec<_> = state
            .cpu_vertices
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        list.sort();
        list
    }

    /// Textures being sampled from the targets that are their levels
    /// ([`Composite`]).
    #[cfg(test)]
    pub(crate) fn composites(&self) -> usize {
        self.state.lock().unwrap().composites.len()
    }

    /// The channel is about to signal: writes every dirty target the
    /// guest may read from then on back to guest memory (through
    /// `memory`, which notes the writes in `writes`), after waiting for
    /// the GPU. Depth buffers and multisampled targets stay on the GPU
    /// ([`Resident::on_demand`]).
    pub fn flush(&self, memory: &dyn GpuMemory, writes: &Writes) -> Result<(), Unmodeled> {
        self.write_back(memory, writes, Scope::Signal)
    }

    /// Writes every dirty target back to guest memory, those held on
    /// demand too: afterwards guest memory holds all of the channel's
    /// work (before a machine snapshot, and before the software renderer
    /// reads memory from several threads).
    pub fn settle(&self, memory: &dyn GpuMemory, writes: &Writes) -> Result<(), Unmodeled> {
        self.write_back(memory, writes, Scope::All)
    }

    /// Waits for the GPU and writes the dirty targets of `scope` back.
    /// The others stay dirty: their textures are final for the work so
    /// far, and later work continues in a new command buffer. A target
    /// held on demand is written only where guest memory still holds what
    /// it held when the two were last equal; otherwise the memory was
    /// reused and what was rendered is forgotten (see the module
    /// documentation).
    fn write_back(
        &self,
        memory: &dyn GpuMemory,
        writes: &Writes,
        scope: Scope,
    ) -> Result<(), Unmodeled> {
        use std::sync::atomic::Ordering;
        let mut state = self.state.lock().unwrap();
        let mut wanted: Vec<bool> = state
            .residents
            .iter()
            .map(|r| {
                r.dirty.is_some()
                    && match scope {
                        Scope::Signal => !r.on_demand(),
                        Scope::Range(start, end) => overlap(r.extent(), (start, end)),
                        Scope::All => true,
                    }
            })
            .collect();
        if !wanted.contains(&true) {
            // Nothing for guest memory: the GPU goes on with what is
            // recorded, and nothing waits for it, unless a draw's texture
            // faults are to be looked at or the recorded draws' data has
            // piled up.
            if matches!(scope, Scope::Range(..)) {
                return Ok(());
            }
            if state.faults.is_empty() && state.arena.chunks_in_use() <= ARENA_UNWAITED {
                state.submit();
                state.trim();
                return Ok(());
            }
        }
        // Residents whose mirror is known to equal memory still (nothing
        // of the channel wrote there since they were compared, in this
        // epoch): they do after their own write-back too (residents do
        // not overlap).
        let mut current: Vec<bool> = state
            .residents
            .iter()
            .map(|r| {
                let (start, end) = r.extent();
                r.stamp.is_some_and(|s| writes.unchanged(s, start, end))
            })
            .collect();
        // Reads and writes of this function pass `Coherent` unchecked.
        self.writing_back.store(true, Ordering::Relaxed);
        // On-demand targets: has the guest reused the memory?
        for (index, resident) in state.residents.iter_mut().enumerate() {
            if !(wanted[index] && resident.on_demand()) || current[index] {
                continue;
            }
            if matches!(resident.mirror.matches(memory), Ok(true)) {
                current[index] = true;
            } else {
                // Forgotten: the texture is stale, and loaded from memory
                // when next drawn to.
                wanted[index] = false;
                resident.dirty = None;
                resident.stamp = None;
            }
        }
        // The wanted residents packed into their guest bytes on the GPU,
        // after the work rendering them (see `State::pack`); `None` where
        // the CPU converts instead.
        let mut packs = Vec::with_capacity(state.residents.len());
        for (index, &wanted) in wanted.iter().enumerate() {
            packs.push(if wanted {
                match state.pack(index) {
                    Ok(pack) => pack,
                    Err(refusal) => {
                        self.writing_back.store(false, Ordering::Relaxed);
                        return Err(refusal);
                    }
                }
            } else {
                None
            });
        }
        state.wait();
        // An executed sample of an unbound texture is the interpreter's
        // refusal; nothing of the work is written back.
        for (buffer, slots) in std::mem::take(&mut state.faults) {
            let words = buffer.contents().cast::<u32>();
            for (slot, refusal) in slots {
                // SAFETY: the buffer holds two words per texture slot of
                // the draw; the GPU finished.
                if unsafe { words.add(2 * slot + 1).read() } != 0 {
                    self.writing_back.store(false, Ordering::Relaxed);
                    return Err(Unmodeled(refusal));
                }
            }
        }
        if let Some(fault) = &state.global_fault {
            let word = fault.contents().cast::<u32>();
            // SAFETY: the buffer holds the word; the GPU finished.
            if unsafe { word.read() } != 0 {
                // SAFETY: as above.
                unsafe { word.write(0) };
                self.writing_back.store(false, Ordering::Relaxed);
                return Err(Unmodeled(
                    "pixel program: global memory load outside the buffers its constants name"
                        .into(),
                ));
            }
        }
        let mut result = Ok(());
        for ((resident, &wanted), pack) in state.residents.iter_mut().zip(&wanted).zip(packs) {
            if !wanted {
                continue;
            }
            if let Some(rect) = resident.dirty.take()
                && result.is_ok()
            {
                result = match pack {
                    Some(pack) => resident.store_packed(pack, memory),
                    None => resident.write_back(rect, memory),
                };
            }
        }
        self.writing_back.store(false, Ordering::Relaxed);
        let any_dirty = state.residents.iter().any(|r| r.dirty.is_some());
        self.any_dirty.store(any_dirty, Ordering::Relaxed);
        let stamp = writes.stamp();
        for ((resident, current), wanted) in state.residents.iter_mut().zip(current).zip(wanted) {
            if wanted {
                resident.stamp = current.then_some(stamp);
            }
        }
        state.trim();
        result
    }

    /// Writes back the dirty targets that overlap `length` bytes at
    /// `address`, if any.
    fn before_access(
        &self,
        memory: &dyn GpuMemory,
        writes: &Writes,
        address: u64,
        length: usize,
    ) -> Result<(), Unmodeled> {
        use std::sync::atomic::Ordering;
        if self.writing_back.load(Ordering::Relaxed) || !self.any_dirty.load(Ordering::Relaxed) {
            return Ok(());
        }
        let range = (address, address + length as u64);
        let overlaps = {
            let state = self.state.lock().unwrap();
            state
                .residents
                .iter()
                .any(|r| r.dirty.is_some() && overlap(r.extent(), range))
        };
        if overlaps {
            self.write_back(memory, writes, Scope::Range(range.0, range.1))?;
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
                program: Arc::clone(draw.program),
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
        if draw.depth.is_some() && !Self::gpu().fetch {
            return self.fall_back("depth test without framebuffer fetch".into());
        }
        // A native draw's vertex program, translated once per content of
        // its constant buffer 1 (as the pixel program).
        let vertex = match &draw.geometry {
            Geometry::Triangles(_) => None,
            Geometry::Native(native) => {
                let buffer_1 = native
                    .constants
                    .get(1)
                    .map_or(&[][..], |(data, size)| &data[..data.len().min(*size)]);
                let translated = {
                    let mut state = self.state.lock().unwrap();
                    let key = Arc::as_ptr(native.program) as usize;
                    let entry = state
                        .vertex_programs
                        .entry(key)
                        .or_insert_with(|| VertexProgram {
                            program: Arc::clone(native.program),
                            translated: msl::translate_vertex(native.program, buffer_1)
                                .map(Arc::new),
                        });
                    if let Ok(t) = &entry.translated
                        && t.constant_buffer_1
                            .as_deref()
                            .is_some_and(|table| table != buffer_1)
                    {
                        entry.translated =
                            msl::translate_vertex(native.program, buffer_1).map(Arc::new);
                    }
                    entry.translated.clone()
                };
                let translated = match translated {
                    Ok(t) => t,
                    Err(reason) => {
                        self.note_cpu_vertices(&format!("vertex program: {reason}"));
                        return Ok(false);
                    }
                };
                // A read of an inactive attribute is the CPU stage's
                // refusal when it executes.
                if translated.inputs.iter().any(|&word| {
                    (0x20..0xA0).contains(&word)
                        && native.inactive & (1 << ((word - 0x20) / 4)) != 0
                }) {
                    self.note_cpu_vertices("vertex program reads an inactive attribute");
                    return Ok(false);
                }
                let (constants, sizes) = match stage_constants(
                    &translated.indirect,
                    &translated.constants,
                    native.constants,
                ) {
                    Ok(bound) => bound,
                    Err(reason) => {
                        self.note_cpu_vertices(&format!("vertex stage: {reason}"));
                        return Ok(false);
                    }
                };
                Some((native, translated, constants, sizes))
            }
        };
        // Per triangle: the exact coverage test and the depth plane (see
        // `Translated::source`).
        let triangles = match draw.geometry {
            Geometry::Triangles(triangles) => triangles,
            Geometry::Native(_) => &[],
        };
        let mut triangle_words = Vec::with_capacity(triangles.len() * msl::TRIANGLE_WORDS);
        for triangle in triangles {
            let (vertices, outside) = triangle.edges();
            for coordinate in vertices.into_iter().flatten() {
                let Ok(coordinate) = i32::try_from(coordinate) else {
                    return self.fall_back("vertex beyond 2^23 pixels".into());
                };
                triangle_words.push(coordinate as u32);
            }
            triangle_words.push(
                outside
                    .iter()
                    .enumerate()
                    .fold(0, |bits, (edge, &o)| bits | (u32::from(o) << edge))
                    | u32::from(!triangle.front_facing) << 3,
            );
            let depth = triangle.depth;
            triangle_words.extend([depth.dx, depth.dy, depth.base].map(f32::to_bits));
        }
        let (constants, sizes) =
            match stage_constants(&translated.indirect, &translated.constants, draw.constants) {
                Ok(bound) => bound,
                Err(reason) => return self.fall_back(reason),
            };
        // The texture handles the program samples through.
        let mut handles = Vec::with_capacity(translated.textures.len());
        for &handle_offset in &translated.textures {
            let Some((data, _)) = draw.constants.get(draw.texture_buffer_index as usize) else {
                return self.fall_back("texture handle buffer unbound".into());
            };
            let Some(bytes) = data.get(handle_offset as usize..handle_offset as usize + 4) else {
                return self.fall_back("texture handle outside the constant buffer".into());
            };
            handles.push(u32::from_le_bytes(bytes.try_into().expect("4 bytes")));
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
        if attachments.len() == 8 && draw.depth.is_some() {
            return self.fall_back("depth test with eight color targets".into());
        }
        if attachments.is_empty() && draw.depth.is_none() {
            return self.fall_back("draw without color or depth targets".into());
        }
        // A fragment failing the stencil or depth test is discarded with
        // its colors; its stencil update survives only without them.
        if !attachments.is_empty()
            && let Some((_, depth)) = draw.depth
            && let Some(stencil) = depth.stencil
        {
            let depth_can_fail = depth.compare != Compare::Always;
            if stencil.front.writes_on_failure(depth_can_fail)
                || stencil.back.writes_on_failure(depth_can_fail)
            {
                return self.fall_back("stencil update of failing fragments with color".into());
            }
        }
        let targets: Vec<(Surface, Content)> = attachments
            .iter()
            .map(|a| (a.0, Content::Color(a.1.format)))
            .chain(
                draw.depth
                    .map(|(surface, depth)| (surface, Content::Depth(depth.format))),
            )
            .collect();
        let samples = draw.multisample.samples;
        // Before the targets' residents are taken: resolving a texture may
        // drop residents (see [`Renderer::composite`]).
        // Textures and samplers. A handle that names no texture is bound to
        // a placeholder and marked: the program records an executed sample
        // of it, which `flush` refuses as the interpreter does (see
        // `shader::msl` `texture.rs`). netConnect's uber-shader names slots
        // its draws leave unset (TIC index 0xfffff) on branches they do not
        // take. A texture with levels held as render targets is sampled
        // from them ([`Composite`]); any other is decoded from memory.
        let mut bound = Vec::new();
        let mut unbound = Vec::new();
        for (slot, &handle) in handles.iter().enumerate() {
            let (tic, tsc) = match draw.textures.header(handle) {
                Ok(header) => header,
                Err(refusal) => {
                    unbound.push((slot, refusal.0));
                    bound.push(None);
                    continue;
                }
            };
            let kind = translated.kinds[slot];
            let sampled = match self.composite(memory, writes, &tic, &targets)? {
                Some(texture) => Ok(Sampled::Composite(texture)),
                None => match format::native_format(&tic, kind) {
                    Some(pixel_format) => draw
                        .textures
                        .raw(handle)
                        .map(|raw| Sampled::Native(raw, pixel_format)),
                    None => draw
                        .textures
                        .resolve(handle)
                        .map(|(texture, _)| Sampled::Decoded(texture)),
                },
            };
            let sampled = match sampled {
                Ok(sampled) => sampled,
                Err(refusal) => {
                    unbound.push((slot, refusal.0));
                    bound.push(None);
                    continue;
                }
            };
            if let Err(reason) = format::check_texture(&tic, &tsc, kind) {
                return self.fall_back(reason);
            }
            if matches!(sampled, Sampled::Composite(_)) && kind != TextureKind::Flat {
                return self.fall_back(format!("rendered texture in a {kind:?} slot"));
            }
            let bias = if translated.implicit_lod[slot] {
                tsc.lod_bias
            } else {
                0.0
            };
            bound.push(Some((sampled, tic, tsc, bias, kind)));
        }
        let global_memory = if translated.globals.is_empty() {
            None
        } else {
            Some(self.global_memory(memory, writes, draw.constants, &translated.globals)?)
        };
        let residents = self.residents(memory, writes, &targets, samples.log2, false)?;

        let mut state = self.state.lock().unwrap();
        let gpu = Self::gpu();
        // Draw data is held until the GPU has done the work using it: a
        // frame of more than this waits for the GPU here.
        if state.arena.chunks_in_use() > ARENA_LIMIT {
            state.wait();
        }
        // Uploads live as long as the caches hold what they were made of,
        // which is looked at when a frame ends: within a frame that
        // uploads more than this, here too.
        if state.texture_bytes > UPLOAD_LIMIT {
            state.wait();
            state.trim_textures();
        }
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
            depth: draw.depth.map(|(_, depth)| msl::DepthStage {
                format: depth.format,
                write: depth.write,
                stencil: depth.stencil.is_some(),
            }),
            multisample: (!samples.is_single()).then_some(msl::MultisampleStage {
                samples: samples.count() as u32,
                alpha_to_coverage: draw.multisample.alpha_to_coverage,
                alpha_to_one: draw.multisample.alpha_to_one,
                center_depth: draw.multisample.center_depth,
            }),
            native: vertex.as_ref().map(|(native, translated, ..)| {
                (
                    Arc::as_ptr(native.program) as usize,
                    translated.constant_buffer_1.clone(),
                    native.fetches.clone(),
                    native.clip_words.clone(),
                )
            }),
        };
        let layered = key.multisample.is_some();
        let pipeline = match state.pipelines.get(&key) {
            Some(p) => p.clone(),
            None => {
                let outputs: Vec<usize> = key.attachments.iter().map(|a| a.3).collect();
                let native_vertex = match &vertex {
                    None => None,
                    Some((native, translated_vertex, ..)) => {
                        let varyings: Vec<(u32, super::shader::Interpolation)> = translated
                            .ipa_words
                            .iter()
                            .map(|&word| {
                                (word, draw.program.header().ps_interpolation(word as usize))
                            })
                            .collect();
                        match msl::NativeVertex::new(
                            translated_vertex,
                            &native.fetches,
                            &varyings,
                            &native.clip_words,
                            layered,
                            false,
                        ) {
                            Ok(source) => Some(source),
                            Err(reason) => {
                                drop(state);
                                self.note_cpu_vertices(&format!("vertex fetch: {reason}"));
                                return Ok(false);
                            }
                        }
                    }
                };
                let source = translated.source(
                    draw.program,
                    &outputs,
                    key.alpha_test,
                    key.depth,
                    key.multisample,
                    native_vertex.as_ref(),
                );
                let library = match state.library(gpu, source) {
                    Ok(library) => library,
                    Err(error) if native_vertex.is_some() => {
                        drop(state);
                        let first = error.0.lines().next().unwrap_or_default().to_string();
                        self.note_cpu_vertices(&first);
                        return Ok(false);
                    }
                    Err(error) => return Err(error),
                };
                // With the stencil test the function stores the whole
                // word (unchanged where nothing changed it).
                let depth = key.depth.map(|stage| {
                    (
                        format::depth_pixel_format(stage.format),
                        stage.write || stage.stencil,
                    )
                });
                let vertex_function = match (&native_vertex, layered) {
                    (Some(_), _) => "vertex_native",
                    (None, true) => "vertex_layered",
                    (None, false) => "vertex_main",
                };
                let pipeline = build_pipeline(
                    gpu,
                    &library,
                    vertex_function,
                    "fragment_main",
                    &key.attachments,
                    depth,
                    layered,
                )?;
                state.pipelines.insert(key.clone(), pipeline.clone(), 1);
                pipeline
            }
        };
        // Textures and samplers on the GPU.
        let mut gpu_textures = Vec::new();
        for (slot, entry) in bound.iter().enumerate() {
            let Some((sampled, tic, tsc, _, kind)) = entry else {
                gpu_textures.push(state.placeholder(gpu, translated.kinds[slot])?);
                continue;
            };
            let metal = match sampled {
                Sampled::Composite(texture) => texture.clone(),
                Sampled::Decoded(texture) => {
                    let id = (Arc::as_ptr(texture) as usize, *kind);
                    match state.textures.get(&id) {
                        Some((_, t)) => t.clone(),
                        None => {
                            let t = upload_texture(gpu, texture, *kind)?;
                            state.texture_bytes += texture.bytes();
                            state.textures.insert(id, (Arc::clone(texture), t.clone()));
                            t
                        }
                    }
                }
                Sampled::Native(raw, pixel_format) => {
                    let id = (Arc::as_ptr(raw) as usize, *kind);
                    match state.native.get(&id) {
                        Some((_, t)) => t.clone(),
                        None => {
                            let t = upload_native(gpu, raw, *kind, *pixel_format)?;
                            state.texture_bytes += raw.bytes.len();
                            state.native.insert(id, (Arc::clone(raw), t.clone()));
                            t
                        }
                    }
                }
            };
            let (min, max) = tic.lod_range(tsc);
            let key = (tsc.words, [min.to_bits(), max.to_bits()]);
            let sampler = match state.samplers.get(&key) {
                Some(s) => s.clone(),
                None => {
                    let s = build_sampler(gpu, tsc, (min, max));
                    state.samplers.insert(key, s.clone(), 1);
                    s
                }
            };
            gpu_textures.push((metal, sampler));
        }
        // Plane geometry: each triangle's pixel bounding box as two
        // triangles, in normalized device coordinates of a viewport
        // covering the first target; the fragment function keeps the
        // pixels the triangle covers. The box's edges lie on pixel edges,
        // so how Metal rounds them does not matter, and Metal's fill rule
        // gives each pixel of the box to one half. Multisampled targets are
        // held as one layer per sample at the size in pixels.
        let pixels = |surface: &Surface| samples.pixels(surface);
        let (width, height) = {
            let p = pixels(&targets[0].0);
            (p.x1 as f32, p.y1 as f32)
        };
        let mut positions = Vec::with_capacity(triangles.len() * 24);
        let mut planes =
            Vec::with_capacity(triangles.len() * translated.ipa_words.len().max(1) * 3);
        for triangle in triangles {
            let b = triangle.bounds();
            let x = |x: u32| x as f32 / width * 2.0 - 1.0;
            let y = |y: u32| 1.0 - y as f32 / height * 2.0;
            let (x0, y0, x1, y1) = (x(b.x0), y(b.y0), x(b.x1), y(b.y1));
            for [x, y] in [[x0, y0], [x1, y0], [x0, y1], [x1, y0], [x1, y1], [x0, y1]] {
                positions.extend_from_slice(&[x, y, 0.0, 1.0]);
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
            .map(|b| b.as_ref().map_or(0.0, |b| b.3))
            .chain([0.0])
            .collect();
        // Texture fault words: word 2i marks slot i unbound.
        let mut fault_words = vec![0u32; 2 * translated.textures.len()];
        for &(slot, _) in &unbound {
            fault_words[2 * slot] = 1;
        }
        let srgb: Vec<bool> = attachments
            .iter()
            .map(|a| a.1.srgb)
            .chain(draw.depth.map(|_| false))
            .collect();
        let encoder = state.encoder(&residents, &srgb)?;
        encoder.setRenderPipelineState(&pipeline);
        let viewport = |dx: f64, dy: f64| MTLViewport {
            originX: dx,
            originY: dy,
            width: f64::from(width),
            height: f64::from(height),
            znear: 0.0,
            zfar: 1.0,
        };
        encoder.setViewport(viewport(0.0, 0.0));
        // Per-draw rasterizer state (the pass's draws share the encoder).
        match &vertex {
            Some((native, ..)) => {
                encoder.setCullMode(match native.cull {
                    1 => MTLCullMode::Front,
                    2 => MTLCullMode::Back,
                    _ => MTLCullMode::None,
                });
                encoder.setFrontFacingWinding(if native.front_clockwise {
                    MTLWinding::Clockwise
                } else {
                    MTLWinding::CounterClockwise
                });
                encoder.setDepthClipMode(if native.depth_clip {
                    MTLDepthClipMode::Clip
                } else {
                    MTLDepthClipMode::Clamp
                });
            }
            None => {
                encoder.setCullMode(MTLCullMode::None);
                encoder.setDepthClipMode(MTLDepthClipMode::Clip);
            }
        }
        let scissor = draw.bounds.intersect(&Rect {
            x0: 0,
            y0: 0,
            x1: targets.iter().map(|t| pixels(&t.0).x1).min().unwrap_or(0),
            y1: targets.iter().map(|t| pixels(&t.0).y1).min().unwrap_or(0),
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
            match &vertex {
                None => {
                    bind(
                        &mut state.arena,
                        &encoder,
                        Stage::Vertex,
                        bytes_of(&positions),
                        0,
                    )?;
                    bind(
                        &mut state.arena,
                        &encoder,
                        Stage::Fragment,
                        bytes_of(&planes),
                        0,
                    )?;
                    bind(
                        &mut state.arena,
                        &encoder,
                        Stage::Fragment,
                        bytes_of_u32(&triangle_words),
                        24,
                    )?;
                }
                Some((native, _, vertex_constants, vertex_sizes)) => {
                    let bases;
                    let elements = match &native.vertices {
                        NativeVertices::Listed(elements) => &elements[..],
                        NativeVertices::Indexed {
                            base_vertex,
                            id_base,
                        } => {
                            bases = [*base_vertex, *id_base];
                            &bases[..]
                        }
                    };
                    bind(
                        &mut state.arena,
                        &encoder,
                        Stage::Vertex,
                        bytes_of_u32(elements),
                        0,
                    )?;
                    for (slot, data) in vertex_constants.iter().enumerate() {
                        if let Some(data) = data {
                            bind(&mut state.arena, &encoder, Stage::Vertex, data, 2 + slot)?;
                        }
                    }
                    bind(
                        &mut state.arena,
                        &encoder,
                        Stage::Vertex,
                        bytes_of_u32(vertex_sizes),
                        20,
                    )?;
                    let mut words = vec![0; msl::INDEXED_WORD + 1];
                    words[0] = native.instance;
                    words[1] = native.fetch_instance;
                    for (slot, stream) in native.streams.iter().enumerate() {
                        words[2 + 3 * slot..][..3].copy_from_slice(&[
                            stream.first,
                            stream.stride,
                            stream.divisor,
                        ]);
                    }
                    words[msl::INDEXED_WORD] =
                        u32::from(matches!(native.vertices, NativeVertices::Indexed { .. }));
                    bind(
                        &mut state.arena,
                        &encoder,
                        Stage::Vertex,
                        bytes_of_u32(&words),
                        29,
                    )?;
                    for (slot, stream) in native.streams.iter().enumerate() {
                        bind(
                            &mut state.arena,
                            &encoder,
                            Stage::Vertex,
                            &stream.bytes,
                            msl::STREAM_BUFFER + slot,
                        )?;
                    }
                    if draw.depth.is_some() {
                        bind(
                            &mut state.arena,
                            &encoder,
                            Stage::Fragment,
                            bytes_of(&native.depth_bias),
                            27,
                        )?;
                    }
                }
            }
            bind(
                &mut state.arena,
                &encoder,
                Stage::Fragment,
                bytes_of(&biases),
                1,
            )?;
            for (slot, data) in constants.iter().enumerate() {
                if let Some(data) = data {
                    bind(&mut state.arena, &encoder, Stage::Fragment, data, 2 + slot)?;
                }
            }
            if !translated.indirect.is_empty() {
                bind(
                    &mut state.arena,
                    &encoder,
                    Stage::Fragment,
                    bytes_of_u32(&sizes),
                    20,
                )?;
            }
            if let Some((compare, reference)) = draw.alpha_test {
                let words = [compare_number(compare), reference.to_bits()];
                bind(
                    &mut state.arena,
                    &encoder,
                    Stage::Fragment,
                    bytes_of_u32(&words),
                    22,
                )?;
            }
            if let Some((_, depth)) = draw.depth {
                let words = [
                    compare_number(depth.compare),
                    depth.clamp.0.to_bits(),
                    depth.clamp.1.to_bits(),
                ];
                bind(
                    &mut state.arena,
                    &encoder,
                    Stage::Fragment,
                    bytes_of_u32(&words),
                    23,
                )?;
            }
            if let Some(stencil) = draw.depth.and_then(|(_, depth)| depth.stencil) {
                let mut words = Vec::with_capacity(14);
                for face in [stencil.front, stencil.back] {
                    words.extend(stencil_words(&face));
                }
                bind(
                    &mut state.arena,
                    &encoder,
                    Stage::Fragment,
                    bytes_of_u32(&words),
                    25,
                )?;
            }
            for (i, (texture, sampler)) in gpu_textures.iter().enumerate() {
                encoder.setFragmentTexture_atIndex(Some(texture), i);
                encoder.setFragmentSamplerState_atIndex(Some(sampler), i);
            }
            if let Some(global_memory) = &global_memory {
                let fault = match &state.global_fault {
                    Some(fault) => fault.clone(),
                    None => {
                        let fault = buffer(gpu, &[0; 4])?;
                        state.global_fault = Some(fault.clone());
                        fault
                    }
                };
                encoder.setFragmentBuffer_offset_atIndex(
                    Some(global_memory),
                    0,
                    msl::GLOBAL_BUFFER,
                );
                encoder.setFragmentBuffer_offset_atIndex(
                    Some(&fault),
                    0,
                    msl::GLOBAL_BUFFER + 1,
                );
            }
            if !unbound.is_empty() {
                let faults = buffer(gpu, bytes_of_u32(&fault_words))?;
                encoder.setFragmentBuffer_offset_atIndex(Some(&faults), 0, 21);
                state.faults.push((faults, unbound));
            } else if !fault_words.is_empty() {
                bind(
                    &mut state.arena,
                    &encoder,
                    Stage::Fragment,
                    bytes_of_u32(&fault_words),
                    21,
                )?;
            }
            if !samples.is_single() && vertex.is_none() {
                let sample_words: Vec<u32> = (0..samples.count())
                    .map(|s| {
                        let [x, y] = samples.positions[s];
                        u32::from(x) | u32::from(y) << 8
                    })
                    .collect();
                bind(
                    &mut state.arena,
                    &encoder,
                    Stage::Fragment,
                    bytes_of_u32(&sample_words),
                    26,
                )?;
            }
            let index_buffer = match &vertex {
                Some((native, ..)) => Some(match &native.indices {
                    NativeIndices::Positions(positions) => (
                        state.arena.alloc(bytes_of_u32(positions))?,
                        positions.len(),
                        MTLIndexType::UInt32,
                    ),
                    NativeIndices::Buffer { bytes, wide: true } => {
                        (state.arena.alloc(bytes)?, bytes.len() / 4, MTLIndexType::UInt32)
                    }
                    NativeIndices::Buffer { bytes, wide: false } => {
                        (state.arena.alloc(bytes)?, bytes.len() / 2, MTLIndexType::UInt16)
                    }
                }),
                None => None,
            };
            // One draw per sample of multisampled targets, each into its
            // layer (see `Translated::source`).
            for layer in 0..samples.count() as u32 {
                match (&vertex, &index_buffer) {
                    (Some((native, ..)), Some(((indices, offset), count, index_type))) => {
                        // Metal samples each pixel at its center: shifted so
                        // that it lands on the layer's sample.
                        let [dx, dy] = if samples.is_single() {
                            [0.0, 0.0]
                        } else {
                            samples.offset(layer as usize).map(|o| 0.5 - f64::from(o))
                        };
                        encoder.setViewport(viewport(dx, dy));
                        let mut view = [0u32; 10];
                        for (word, value) in view.iter_mut().zip(native.view) {
                            *word = value.to_bits();
                        }
                        view[9] = layer;
                        bind(
                            &mut state.arena,
                            &encoder,
                            Stage::Vertex,
                            bytes_of_u32(&view),
                            1,
                        )?;
                        encoder
                            .drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset(
                                MTLPrimitiveType::Triangle,
                                *count,
                                *index_type,
                                indices,
                                *offset,
                            );
                    }
                    _ => {
                        if !samples.is_single() {
                            bind(
                                &mut state.arena,
                                &encoder,
                                Stage::Vertex,
                                bytes_of_u32(&[layer]),
                                1,
                            )?;
                        }
                        encoder.drawPrimitives_vertexStart_vertexCount(
                            MTLPrimitiveType::Triangle,
                            0,
                            triangles.len() * 6,
                        );
                    }
                }
            }
        }
        // A depth buffer the draw does not write stays clean.
        for (&index, (_, content)) in residents.iter().zip(&targets) {
            if matches!(content, Content::Depth(_)) && draw.depth.is_some_and(|d| !d.1.writes()) {
                continue;
            }
            state.rendered(index, scissor, writes);
            self.any_dirty
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }
        state.draws += 1;
        Ok(true)
    }

    /// A color clear on Metal: `components` (bit i: component i) of
    /// `rect` of the target set to `color`, encoded as the format stores
    /// it (no sRGB conversion, as `CLEAR_SURFACE` writes the value raw).
    #[allow(clippy::too_many_arguments)]
    pub fn clear(
        &self,
        memory: &dyn GpuMemory,
        writes: &Writes,
        (surface, format): (Surface, ColorFormat),
        samples: Samples,
        rect: Rect,
        color: [f32; 4],
        components: u8,
    ) -> Result<bool, Unmodeled> {
        let Some(pixel_format) = format::pixel_format(format) else {
            return self.fall_back(format!("clear of color format {format:?}"));
        };
        let size = samples.pixels(&surface);
        let rect = rect.intersect(&size);
        if rect.is_empty() {
            return Ok(true);
        }
        let residents = self.residents(
            memory,
            writes,
            &[(surface, Content::Color(format))],
            samples.log2,
            rect == size && components == 0xF,
        )?;
        let mut state = self.state.lock().unwrap();
        let gpu = Self::gpu();
        let layered = !samples.is_single();
        let key = (pixel_format.0, components, layered);
        let pipeline = match state.clear_pipelines.get(&key) {
            Some(p) => p.clone(),
            None => {
                let library = state.library(gpu, CLEAR_SOURCE.to_string())?;
                let p = build_pipeline(
                    gpu,
                    &library,
                    if layered {
                        "vertex_layered"
                    } else {
                        "vertex_main"
                    },
                    "clear_main",
                    &[(pixel_format.0, None, components, 0)],
                    None,
                    layered,
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
            width: f64::from(size.x1),
            height: f64::from(size.y1),
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
            bind(
                &mut state.arena,
                &encoder,
                Stage::Fragment,
                bytes_of(&color),
                0,
            )?;
            draw_clear(&mut state.arena, &encoder, samples)?;
        }
        state.rendered(residents[0], rect, writes);
        self.any_dirty
            .store(true, std::sync::atomic::Ordering::Relaxed);
        Ok(true)
    }

    /// A depth/stencil clear on Metal: each texel of `rect` of the depth
    /// buffer becomes `apply` of it. `apply` must set each bit to a
    /// constant or leave it (as `DepthFormat::write_depth` and
    /// `write_stencil` do); the clear is then the same bitwise operation
    /// on the buffer's words, found by applying it to texels of all zeros
    /// and all ones, and checked on two more.
    pub fn clear_depth(
        &self,
        memory: &dyn GpuMemory,
        writes: &Writes,
        (surface, format): (Surface, DepthFormat),
        samples: Samples,
        rect: Rect,
        apply: impl Fn(&mut [u8]),
    ) -> Result<bool, Unmodeled> {
        if !Self::gpu().fetch {
            return self.fall_back("depth clear without framebuffer fetch".into());
        }
        let bytes = format.bytes_per_pixel() as usize;
        let probe = |fill: u8| {
            let mut texel = vec![fill; bytes];
            apply(&mut texel);
            texel
        };
        let (zeros, ones) = (probe(0), probe(0xFF));
        let keep: Vec<u8> = zeros.iter().zip(&ones).map(|(a, b)| a ^ b).collect();
        for fill in [0x55, 0xA6] {
            let expected: Vec<u8> = (0..bytes).map(|i| (fill & keep[i]) | zeros[i]).collect();
            if probe(fill) != expected {
                return self.fall_back("depth clear that is not bitwise".into());
            }
        }
        // Keep and set masks per word (a 16-bit texel in the low half).
        let word = |bytes: &[u8], i: usize| {
            let mut w = [0u8; 4];
            for (k, b) in w.iter_mut().enumerate() {
                *b = bytes.get(4 * i + k).copied().unwrap_or(0);
            }
            u32::from_le_bytes(w)
        };
        let masks = [
            word(&keep, 0),
            word(&zeros, 0),
            word(&keep, 1),
            word(&zeros, 1),
        ];
        let size = samples.pixels(&surface);
        let rect = rect.intersect(&size);
        if rect.is_empty() {
            return Ok(true);
        }
        // Every bit of every texel set: the buffer's content is replaced.
        let overwrite = rect == size && masks[0] == 0 && masks[2] == 0;
        let residents = self.residents(
            memory,
            writes,
            &[(surface, Content::Depth(format))],
            samples.log2,
            overwrite,
        )?;
        let mut state = self.state.lock().unwrap();
        let gpu = Self::gpu();
        let pixel_format = format::depth_pixel_format(format);
        let layered = !samples.is_single();
        let key = (pixel_format.0, u8::MAX, layered);
        let pipeline = match state.clear_pipelines.get(&key) {
            Some(p) => p.clone(),
            None => {
                let library =
                    state.library(gpu, format!("{CLEAR_SOURCE}{DEPTH_CLEAR_FUNCTIONS}"))?;
                let function = if pixel_format == MTLPixelFormat::RG32Uint {
                    "clear_words"
                } else {
                    "clear_word"
                };
                let p = build_pipeline(
                    gpu,
                    &library,
                    if layered {
                        "vertex_layered"
                    } else {
                        "vertex_main"
                    },
                    function,
                    &[],
                    Some((pixel_format, true)),
                    layered,
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
            width: f64::from(size.x1),
            height: f64::from(size.y1),
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
            bind(
                &mut state.arena,
                &encoder,
                Stage::Fragment,
                bytes_of_u32(&masks),
                0,
            )?;
            draw_clear(&mut state.arena, &encoder, samples)?;
        }
        state.rendered(residents[0], rect, writes);
        self.any_dirty
            .store(true, std::sync::atomic::Ordering::Relaxed);
        Ok(true)
    }

    /// The memory a pixel program's constants name, as its global memory
    /// buffer (`shader::msl` `memory.rs`): the table of `places` (where
    /// `constants` hold descriptors) and the bytes of each. A place that
    /// is no descriptor of mapped memory, or whose bytes would take the
    /// buffer past [`GLOBAL_LIMIT`], binds nothing. The buffer is made
    /// once and used until memory there changes: kept by its regions,
    /// taken as it is while no write of the channel touched them in this
    /// epoch, compared with memory otherwise.
    fn global_memory(
        &self,
        memory: &dyn GpuMemory,
        writes: &Writes,
        constants: &[(Vec<u8>, usize)],
        places: &[(u32, u32)],
    ) -> Result<Retained<MetalBuffer>, Unmodeled> {
        let mut total = 0;
        let regions: Vec<(u64, u32)> = places
            .iter()
            .map(|&(slot, offset)| {
                let descriptor = constants
                    .get(slot as usize)
                    .and_then(|(data, _)| data.get(offset as usize..offset as usize + 12))?;
                let address = u64::from_le_bytes(descriptor[..8].try_into().expect("8 bytes"));
                let size = u32::from_le_bytes(descriptor[8..].try_into().expect("4 bytes"));
                let fits = address >> 40 == 0
                    && address.is_multiple_of(4)
                    && size != 0
                    && total + size as usize <= GLOBAL_LIMIT;
                fits.then(|| {
                    total += (size as usize).next_multiple_of(4);
                    (address, size)
                })
            })
            .map(|region| region.unwrap_or((0, 0)))
            .collect();
        let unchanged = |stamp: Stamp| {
            regions
                .iter()
                .all(|&(address, size)| writes.unchanged(stamp, address, address + u64::from(size)))
        };
        if let Some(held) = self.state.lock().unwrap().globals.get(&regions)
            && unchanged(held.stamp)
        {
            return Ok(held.buffer.clone());
        }
        // Read outside the lock: a read may write residents back.
        let table = 4 * regions.len();
        let mut words = vec![0u32; table];
        let mut bytes: Vec<u8> = Vec::with_capacity(total);
        for (index, &(address, size)) in regions.iter().enumerate() {
            if size == 0 {
                continue;
            }
            let start = bytes.len();
            bytes.resize(start + (size as usize).next_multiple_of(4), 0);
            if memory
                .probe(address, &mut bytes[start..start + size as usize])
                .is_none()
            {
                bytes.truncate(start);
                continue;
            }
            words[4 * index..4 * index + 4].copy_from_slice(&[
                address as u32,
                (address >> 32) as u32,
                size,
                (table + start / 4) as u32,
            ]);
        }
        let mut content = bytes_of_u32(&words).to_vec();
        content.extend_from_slice(&bytes);
        let stamp = writes.stamp();
        let mut state = self.state.lock().unwrap();
        if let Some(held) = state.globals.get_mut(&regions)
            && held.content == content
        {
            held.stamp = stamp;
            return Ok(held.buffer.clone());
        }
        // What this replaces lives on until the work using it is done.
        if state.globals_pending + content.len() > GLOBAL_PENDING {
            state.wait();
        }
        state.globals_pending += content.len();
        let buffer = buffer(Self::gpu(), &content)?;
        let cost = content.len();
        state.globals.insert(
            regions,
            GlobalMemory {
                content,
                stamp,
                buffer: buffer.clone(),
            },
            cost,
        );
        Ok(buffer)
    }

    /// The residents for `targets` of samples `log2` (see
    /// [`Resident::log2`]), created, checked against guest memory or
    /// reloaded as needed; their indices.
    fn residents(
        &self,
        memory: &dyn GpuMemory,
        writes: &Writes,
        targets: &[(Surface, Content)],
        log2: [u32; 2],
        overwrite: bool,
    ) -> Result<Vec<usize>, Unmodeled> {
        {
            let mut state = self.state.lock().unwrap();
            // Over budget with nothing clean to drop: targets held on
            // demand are written back, to be dropped too.
            if state.trim_residents(targets) && state.residents.iter().any(|r| r.dirty.is_some())
            {
                drop(state);
                self.settle(memory, writes)?;
                self.state.lock().unwrap().trim_residents(targets);
            }
        }
        let other = |r: &Resident, surface: Surface, format: Content| {
            r.surface != surface || r.format != format || r.log2 != log2
        };
        // A resident of other dimensions, format or samples at overlapping
        // memory is written back and dropped, before any index is taken
        // (and with no pass open that names one).
        for &(surface, format) in targets {
            let stale: bool = {
                let state = self.state.lock().unwrap();
                state
                    .residents
                    .iter()
                    .any(|r| other(r, surface, format) && overlap(r.extent(), extent(&surface)))
            };
            if stale {
                let (start, end) = extent(&surface);
                self.write_back(memory, writes, Scope::Range(start, end))?;
                let mut state = self.state.lock().unwrap();
                state.submit();
                state.residents.retain(|r| {
                    !(other(r, surface, format) && overlap(r.extent(), extent(&surface)))
                });
            }
        }
        let layout = Samples {
            log2,
            ..Samples::SINGLE
        };
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
                    .position(|r| !other(r, surface, format))
                    .map(|i| {
                        let r = &state.residents[i];
                        let (start, end) = r.extent();
                        let current = r.dirty.is_some()
                            || r.stamp.is_some_and(|s| writes.unchanged(s, start, end));
                        (i, current)
                    })
            };
            // The whole surface: its pixels, and the texels holding them.
            let full = layout.pixels(&surface);
            let texels = Rect {
                x0: 0,
                y0: 0,
                x1: surface.width,
                y1: surface.height,
            };
            match found {
                Some((index, true)) => indices.push(index),
                // A target the caller replaces whole needs no look at
                // memory: what it held does not matter. (One held on demand
                // keeps a mirror equal to memory, which tells a reuse of
                // the memory later.)
                Some((index, false))
                    if overwrite && !self.state.lock().unwrap().residents[index].on_demand() =>
                {
                    self.state.lock().unwrap().residents[index].stamp = Some(writes.stamp());
                    indices.push(index);
                }
                Some((index, false)) => {
                    // Compare with guest memory, without holding the
                    // state (reads may flush); reload if it changed.
                    let mirror = std::mem::replace(
                        &mut self.state.lock().unwrap().residents[index].mirror,
                        Window::empty(surface),
                    );
                    let reloaded = match mirror.matches(memory) {
                        Ok(true) => None,
                        Ok(false) => Some(Window::load(memory, surface, texels)),
                        Err(refusal) => Some(Err(refusal)),
                    };
                    let stamp = writes.stamp();
                    let mut state = self.state.lock().unwrap();
                    state.residents[index].mirror = mirror;
                    match reloaded {
                        None => {}
                        Some(Ok(mut window)) => {
                            // Work recorded or running may read the
                            // texture (a depth buffer tested, not written):
                            // it sees what the texture held when recorded.
                            state.wait();
                            let resident = &mut state.residents[index];
                            upload_surface(&resident.texture, &mut window, full, log2);
                            state.versions += 1;
                            let version = state.versions;
                            let resident = &mut state.residents[index];
                            resident.version = version;
                            resident.mirror = window;
                        }
                        Some(Err(refusal)) => return Err(refusal),
                    }
                    state.residents[index].stamp = Some(stamp);
                    indices.push(index);
                }
                None => {
                    let pixel_format = format.pixel_format().expect("checked by the caller");
                    let mut window = Window::load(memory, surface, texels)?;
                    let stamp = writes.stamp();
                    let texture =
                        new_target(Self::gpu(), pixel_format, full.x1, full.y1, layout.count())?;
                    upload_surface(&texture, &mut window, full, log2);
                    let mut state = self.state.lock().unwrap();
                    state.residents.push(Resident {
                        surface,
                        format,
                        log2,
                        texture,
                        mirror: window,
                        dirty: None,
                        stamp: Some(stamp),
                        used: 0,
                        version: 0,
                    });
                    indices.push(state.residents.len() - 1);
                }
            }
        }
        let mut state = self.state.lock().unwrap();
        state.clock += 1;
        let now = state.clock;
        for &index in &indices {
            state.residents[index].used = now;
        }
        Ok(indices)
    }

    /// The Metal texture to sample for the texture `tic` describes when
    /// levels of it are render targets held on the GPU and current
    /// ([`Composite`]); `None` when none is (or the draw renders to one:
    /// it then samples what memory holds once the target is written
    /// back), and the texture is decoded from memory as any other.
    fn composite(
        &self,
        memory: &dyn GpuMemory,
        writes: &Writes,
        tic: &Tic,
        targets: &[(Surface, Content)],
    ) -> Result<Option<Retained<MetalTexture>>, Unmodeled> {
        // Most textures have no level at a resident's address.
        {
            let state = self.state.lock().unwrap();
            if !state
                .residents
                .iter()
                .any(|r| r.surface.address >= tic.address && r.log2 == [0, 0])
            {
                return Ok(None);
            }
        }
        let Some(levels) = level_surfaces(tic) else {
            return Ok(None);
        };
        // Which levels are residents: their content, how Metal's
        // components of each give the header's, and whether they are
        // known to hold what the channel's work so far makes of them.
        let candidates: Vec<Option<(Content, [u32; 4], bool)>> = {
            let state = self.state.lock().unwrap();
            levels
                .iter()
                .map(|(surface, _)| {
                    let resident = state
                        .residents
                        .iter()
                        .find(|r| r.surface == *surface && r.log2 == [0, 0])?;
                    let Content::Color(format) = resident.format else {
                        return None;
                    };
                    let components = shared_components(tic, format)?;
                    Some((resident.format, components, resident.current(writes)))
                })
                .collect()
        };
        let Some((_, components, _)) = candidates.iter().flatten().next().copied() else {
            return Ok(None);
        };
        if candidates.iter().flatten().any(|c| c.1 != components)
            || levels
                .iter()
                .zip(&candidates)
                .any(|((surface, _), c)| c.is_some() && targets.iter().any(|t| t.0 == *surface))
        {
            return Ok(None);
        }
        // A resident not rendered to since the channel last signaled is
        // compared with memory first, as before a draw to it.
        let unchecked: Vec<(Surface, Content)> = levels
            .iter()
            .zip(&candidates)
            .filter_map(|((surface, _), c)| match c {
                Some((content, _, false)) => Some((*surface, *content)),
                _ => None,
            })
            .collect();
        if !unchecked.is_empty() {
            self.residents(memory, writes, &unchecked, [0, 0], false)?;
        }
        // Whether a composite's levels come from where they do now.
        let same_sources = |composite: &Composite| {
            composite
                .levels
                .iter()
                .zip(&candidates)
                .all(|(level, c)| level.resident.is_some() == c.is_some())
                && composite.levels.len() == candidates.len()
        };
        {
            let mut state = self.state.lock().unwrap();
            state.clock += 1;
            let now = state.clock;
            // The composite kept for the header, if its levels come from
            // where they do now and those from memory are known to be
            // unchanged.
            let ready = state.composites.get_mut(&tic.words).is_some_and(|composite| {
                composite.used = now;
                same_sources(composite)
                    && composite.levels.iter().all(|level| {
                        level.resident.is_some()
                            || writes.unchanged(
                                composite.stamp,
                                tic.address + level.range.start as u64,
                                tic.address + level.range.end as u64,
                            )
                    })
            });
            if ready {
                state.convert(&tic.words)?;
                return Ok(state
                    .composites
                    .get(&tic.words)
                    .map(|composite| composite.texture.clone()));
            }
        }
        // The levels memory holds, read without the state held (a read
        // may write targets back). What cannot be read or decoded is the
        // decoder's refusal.
        let Ok((_, footprint)) = Texture::footprint(tic) else {
            return Ok(None);
        };
        let mut raw = vec![0u8; footprint as usize];
        let memory_levels = || {
            levels
                .iter()
                .zip(&candidates)
                .filter(|(_, c)| c.is_none())
                .map(|((_, range), _)| range.clone())
        };
        for range in memory_levels() {
            let address = tic.address + range.start as u64;
            if memory.probe(address, &mut raw[range]).is_none() {
                return Ok(None);
            }
        }
        let stamp = writes.stamp();
        let mut state = self.state.lock().unwrap();
        let unchanged = state.composites.get(&tic.words).is_some_and(|composite| {
            same_sources(composite)
                && memory_levels().all(|range| composite.raw[range.clone()] == raw[range])
        });
        if unchanged {
            let composite = state.composites.get_mut(&tic.words).expect("found above");
            composite.stamp = stamp;
        } else {
            let decoded = if candidates.iter().any(Option::is_none) {
                match Texture::decode(&raw, tic) {
                    Ok(decoded) => Some(decoded),
                    Err(_) => return Ok(None),
                }
            } else {
                None
            };
            let residents: Vec<bool> = candidates.iter().map(Option::is_some).collect();
            let composite = Composite::new(
                Self::gpu(),
                tic,
                &levels,
                &residents,
                components,
                decoded.as_ref(),
                raw,
                stamp,
            )?;
            state.clock += 1;
            let composite = Composite {
                used: state.clock,
                ..composite
            };
            state.composites.insert(tic.words, composite);
        }
        state.convert(&tic.words)?;
        Ok(state
            .composites
            .get(&tic.words)
            .map(|composite| composite.texture.clone()))
    }
}

impl State {
    /// Commits the command buffer being recorded, if any: the GPU starts
    /// on it while later work is recorded into a new one. Command buffers
    /// of one queue run in the order they were committed [public: Apple,
    /// `MTLCommandQueue`: "command buffers are executed in the order in
    /// which they are committed"], so the last one committed completes
    /// last.
    fn submit(&mut self) {
        if let Some(mut frame) = self.frame.take() {
            if let Some((encoder, _)) = frame.pass.take() {
                encoder.endEncoding();
            }
            frame.command.commit();
            self.submitted = Some(frame.command.clone());
        }
        self.pass_draws = 0;
    }

    /// Commits what is recorded and waits until the GPU has done all the
    /// work committed so far.
    fn wait(&mut self) {
        self.submit();
        if let Some(command) = self.submitted.take() {
            command.waitUntilCompleted();
        }
        // Nothing recorded or running uses the arena's space any more.
        self.arena.reset();
        self.globals_pending = 0;
    }

    /// Encodes, after the work recorded so far, a compute pass that writes
    /// resident `index`'s dirty texels as the guest stores them (samples
    /// interleaved as `multisample` lays them out, block-linear GOBs or
    /// pitch rows) over a copy of the mirror's bands that hold them, so
    /// that the write-back is a copy of whole bands (see
    /// [`Resident::store_packed`]). `None` (nothing encoded) for a resident
    /// that is not dirty, of a texel size without a raw view, or whose
    /// mirror lays texels out otherwise than [`guest_offset`] says.
    fn pack(&mut self, index: usize) -> Result<Option<Packed>, Unmodeled> {
        let resident = &self.residents[index];
        let Some(rect) = resident.dirty else {
            return Ok(None);
        };
        let bpp = resident.surface.bytes_per_pixel;
        let raw_format = match bpp {
            1 => MTLPixelFormat::R8Uint,
            2 => MTLPixelFormat::R16Uint,
            4 => MTLPixelFormat::R32Uint,
            8 => MTLPixelFormat::RG32Uint,
            16 => MTLPixelFormat::RGBA32Uint,
            _ => return Ok(None),
        };
        let layout = Samples {
            log2: resident.log2,
            ..Samples::SINGLE
        };
        let texels = layout.texels(rect);
        if texels.is_empty() {
            return Ok(None);
        }
        let (Some(first), Some(last)) = (
            resident.mirror.band_range(texels.y0),
            resident.mirror.band_range(texels.y1 - 1),
        ) else {
            return Ok(None);
        };
        // A pitch surface's mirror holds its rows without the padding
        // between them (`surface::Window`): 8 rows a band.
        let pitch = match resident.surface.layout {
            super::surface::Layout::Pitch { .. } => (first.len() / 8) as u32,
            super::surface::Layout::BlockLinear(_) => 0,
        };
        let words = PackWords {
            x0: texels.x0,
            y0: texels.y0,
            width: texels.x1 - texels.x0,
            height: texels.y1 - texels.y0,
            bpp,
            lx: resident.log2[0],
            ly: resident.log2[1],
            pitch,
            band_bytes: first.len() as u32,
            start: first.start as u32,
        };
        // The mirror must lay texels out as the kernel does.
        let corners = [
            (texels.x0, texels.y0),
            (texels.x1 - 1, texels.y0),
            (texels.x0, texels.y1 - 1),
            (texels.x1 - 1, texels.y1 - 1),
        ];
        if corners.iter().any(|&(x, y)| {
            first.start + guest_offset(&words, x, y) as usize != resident.mirror.texel_offset(x, y)
        }) {
            self.note("write-back on the CPU: mirror layout");
            return Ok(None);
        }
        let Some(view) = resident.texture.newTextureViewWithPixelFormat(raw_format) else {
            self.note("write-back on the CPU: no raw view");
            return Ok(None);
        };
        let (start, end) = (first.start, last.end);
        let layered = resident.log2 != [0, 0];
        let (buffer, offset) = self.arena.alloc(&resident.mirror.data()[start..end])?;
        let pipeline = self.pack_pipeline(layered)?;
        let command = self.command()?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| Unmodeled("Metal: no compute encoder".into()))?;
        encoder.setComputePipelineState(&pipeline);
        // SAFETY: the kernel's bindings: the view, the staging range and
        // the words; the grid covers the dirty texels.
        unsafe {
            encoder.setTexture_atIndex(Some(&view), 0);
            encoder.setBuffer_offset_atIndex(Some(&buffer), offset, 0);
            let bytes =
                NonNull::new((&words as *const PackWords).cast_mut().cast()).expect("non-null");
            encoder.setBytes_length_atIndex(bytes, std::mem::size_of::<PackWords>(), 1);
        }
        let group = MTLSize {
            width: 16,
            height: 16,
            depth: 1,
        };
        encoder.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: words.width as usize,
                height: words.height as usize,
                depth: 1,
            },
            group,
        );
        encoder.endEncoding();
        Ok(Some(Packed {
            buffer,
            offset,
            bands: start..end,
            texels,
        }))
    }

    /// Counts a reason as [`Renderer::fall_back`] does.
    fn note(&mut self, reason: &str) {
        let count = self.fallbacks.entry(reason.to_string()).or_default();
        if *count == 0 {
            eprintln!("Metal renderer: {reason}");
        }
        *count += 1;
    }

    /// The command buffer being recorded (made if there is none), with no
    /// render pass open.
    fn command(&mut self) -> Result<Retained<CommandBuffer>, Unmodeled> {
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
        if let Some((encoder, _)) = frame.pass.take() {
            encoder.endEncoding();
        }
        Ok(frame.command.clone())
    }

    /// The packing kernel for layered (one layer per sample) or plain
    /// residents (made once).
    fn pack_pipeline(&mut self, layered: bool) -> Result<Retained<ComputePipeline>, Unmodeled> {
        if let Some(pipeline) = self.pack_pipelines.get(&layered) {
            return Ok(pipeline.clone());
        }
        let gpu = Renderer::gpu();
        let library = self.library(gpu, PACK_SOURCE.to_string())?;
        let name = if layered { "pack_layers" } else { "pack_plain" };
        let function = library
            .newFunctionWithName(&NSString::from_str(name))
            .ok_or_else(|| Unmodeled("Metal: no packing function".into()))?;
        let pipeline = gpu
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| Unmodeled(format!("Metal: packing pipeline: {e}")))?;
        self.pack_pipelines.insert(layered, pipeline.clone());
        Ok(pipeline)
    }

    /// Drops the uploads of textures the texture cache no longer holds
    /// (and no draw: a draw holds what it samples), and counts the
    /// bytes of the others.
    fn trim_textures(&mut self) {
        self.textures
            .retain(|_, (texture, _)| Arc::strong_count(texture) > 1);
        self.native.retain(|_, (raw, _)| Arc::strong_count(raw) > 1);
        self.texture_bytes = self
            .textures
            .values()
            .map(|(texture, _)| texture.bytes())
            .chain(self.native.values().map(|(raw, _)| raw.bytes.len()))
            .sum();
    }

    /// Drops what the software caches no longer hold: uploads of textures
    /// the texture cache dropped, and translations of programs the program
    /// cache dropped, with every pipeline made from them (a pipeline is
    /// keyed by the program's address, which may be reused once the
    /// program is freed). Run after a flush, when no draw holds either.
    fn trim(&mut self) {
        self.trim_textures();
        // Composites none of whose levels is a resident any more, and the
        // least recently used beyond the limit.
        let residents = &self.residents;
        self.composites.retain(|_, composite| {
            composite.levels.iter().any(|level| {
                level.resident.is_some() && residents.iter().any(|r| r.surface == level.surface)
            })
        });
        while self.composites.len() > COMPOSITE_LIMIT {
            let oldest = self
                .composites
                .iter()
                .min_by_key(|(_, composite)| composite.used)
                .map(|(key, _)| *key)
                .expect("not empty");
            self.composites.remove(&oldest);
        }
        let before = self.programs.len() + self.vertex_programs.len();
        self.programs
            .retain(|_, entry| Arc::strong_count(&entry.program) > 1);
        self.vertex_programs
            .retain(|_, entry| Arc::strong_count(&entry.program) > 1);
        if self.programs.len() + self.vertex_programs.len() != before {
            let (programs, vertex_programs) = (&self.programs, &self.vertex_programs);
            self.pipelines.retain(|key, _| {
                programs.contains_key(&key.program)
                    && key
                        .native
                        .as_ref()
                        .is_none_or(|native| vertex_programs.contains_key(&native.0))
            });
        }
    }

    /// With no frame recording (so no resident is an attachment of an
    /// open pass, and no index is held), drops the least recently used
    /// clean residents other than `targets` while all of them take more
    /// than [`RESIDENT_BUDGET`] (guest memory holds everything they
    /// hold). Whether they still take more.
    fn trim_residents(&mut self, targets: &[(Surface, Content)]) -> bool {
        if self.frame.is_some() {
            return false;
        }
        // The texture takes about what the mirror does: the surface.
        let size = |r: &Resident| 2 * r.mirror.data().len();
        let mut total: usize = self.residents.iter().map(size).sum();
        if total <= RESIDENT_BUDGET {
            return false;
        }
        let mut order: Vec<(u64, usize)> = self
            .residents
            .iter()
            .enumerate()
            .filter(|(_, r)| r.dirty.is_none() && !targets.contains(&(r.surface, r.format)))
            .map(|(i, r)| (r.used, i))
            .collect();
        order.sort_unstable();
        let mut drop = vec![false; self.residents.len()];
        for (_, index) in order {
            if total <= RESIDENT_BUDGET {
                break;
            }
            total -= size(&self.residents[index]);
            drop[index] = true;
        }
        let mut index = 0;
        self.residents.retain(|_| {
            index += 1;
            !drop[index - 1]
        });
        total > RESIDENT_BUDGET
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
        self.versions += 1;
        self.residents[index].version = self.versions;
    }

    /// The kernel filling a composite's level from a resident
    /// ([`CONVERT_SOURCE`]), made once.
    fn convert_pipeline(&mut self) -> Result<Retained<ComputePipeline>, Unmodeled> {
        if let Some(pipeline) = &self.convert_pipeline {
            return Ok(pipeline.clone());
        }
        let gpu = Renderer::gpu();
        let library = self.library(gpu, CONVERT_SOURCE.to_string())?;
        let function = library
            .newFunctionWithName(&NSString::from_str("convert_level"))
            .ok_or_else(|| Unmodeled("Metal: no conversion function".into()))?;
        let pipeline = gpu
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| Unmodeled(format!("Metal: conversion pipeline: {e}")))?;
        self.convert_pipeline = Some(pipeline.clone());
        Ok(pipeline)
    }

    /// Fills the levels of composite `key` that are residents rendered
    /// since they were last converted, after the work recorded so far.
    fn convert(&mut self, key: &[u32; 8]) -> Result<(), Unmodeled> {
        let Some(composite) = self.composites.get(key) else {
            return Ok(());
        };
        let mut stale = Vec::new();
        for (index, level) in composite.levels.iter().enumerate() {
            let Some((view, converted)) = &level.resident else {
                continue;
            };
            let Some(resident) = self.residents.iter().find(|r| r.surface == level.surface)
            else {
                continue;
            };
            if *converted != Some(resident.version) {
                stale.push((
                    index,
                    resident.texture.clone(),
                    view.clone(),
                    resident.version,
                    level.surface,
                ));
            }
        }
        if stale.is_empty() {
            return Ok(());
        }
        let select = composite.select;
        let pipeline = self.convert_pipeline()?;
        let command = self.command()?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| Unmodeled("Metal: no compute encoder".into()))?;
        encoder.setComputePipelineState(&pipeline);
        for (_, source, view, _, surface) in &stale {
            // SAFETY: the kernel's bindings: the resident, the level and
            // the component selection; the grid covers the level.
            unsafe {
                encoder.setTexture_atIndex(Some(source), 0);
                encoder.setTexture_atIndex(Some(view), 1);
                let bytes = NonNull::new(select.as_ptr().cast_mut().cast()).expect("non-null");
                encoder.setBytes_length_atIndex(bytes, std::mem::size_of_val(&select), 0);
            }
            encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: surface.width as usize,
                    height: surface.height as usize,
                    depth: 1,
                },
                MTLSize {
                    width: 16,
                    height: 16,
                    depth: 1,
                },
            );
        }
        encoder.endEncoding();
        let composite = self.composites.get_mut(key).expect("found above");
        for (index, _, _, version, _) in stale {
            if let Some((_, converted)) = &mut composite.levels[index].resident {
                *converted = Some(version);
            }
        }
        Ok(())
    }

    /// What an unbound texture slot of `kind` is bound to (made once).
    fn placeholder(
        &mut self,
        gpu: &Gpu,
        kind: TextureKind,
    ) -> Result<(Retained<MetalTexture>, Retained<Sampler>), Unmodeled> {
        if let Some(placeholder) = self.placeholders.get(&kind) {
            return Ok(placeholder.clone());
        }
        let descriptor = texture_descriptor(kind, [1, 1, 1], 1, 1);
        let texture = gpu
            .device
            .newTextureWithDescriptor(&descriptor)
            .ok_or_else(|| Unmodeled("Metal: texture allocation failed".into()))?;
        let sampler = gpu
            .device
            .newSamplerStateWithDescriptor(&MTLSamplerDescriptor::new())
            .expect("sampler state");
        self.placeholders
            .insert(kind, (texture.clone(), sampler.clone()));
        Ok((texture, sampler))
    }

    /// The compiled library of `source`, compiled once while kept
    /// ([`LIBRARY_LIMIT`]).
    fn library(&mut self, gpu: &Gpu, source: String) -> Result<Retained<Library>, Unmodeled> {
        if let Some(library) = self.libraries.get(&source) {
            return Ok(library.clone());
        }
        let library = compile_library(gpu, &source)?;
        self.libraries.insert(source, library.clone(), 1);
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
        if let Some(frame) = &self.frame
            && let Some((encoder, attachments)) = &frame.pass
            && *attachments == wanted
            && self.pass_draws < PASS_DRAWS
        {
            self.pass_draws += 1;
            return Ok(encoder.clone());
        }
        // Another pass: the GPU starts on the work recorded so far while
        // this one is recorded.
        self.submit();
        self.pass_draws = 1;
        let command = Renderer::gpu()
            .queue
            .commandBuffer()
            .ok_or_else(|| Unmodeled("Metal: no command buffer".into()))?;
        let frame = self.frame.insert(Frame {
            command,
            pass: None,
        });
        let descriptor = MTLRenderPassDescriptor::new();
        // Multisampled targets: their layers, one per sample.
        if let Some(&(first, _)) = wanted.first() {
            let [lx, ly] = self.residents[first].log2;
            if lx + ly > 0 {
                descriptor.setRenderTargetArrayLength(1 << (lx + ly));
            }
        }
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

    /// Whether the target is written back only on demand rather than at
    /// every signal: depth buffers and multisampled targets (see the
    /// module documentation).
    fn on_demand(&self) -> bool {
        matches!(self.format, Content::Depth(_)) || self.log2 != [0, 0]
    }

    /// Whether the texture holds what the channel's work so far makes of
    /// the surface: rendered to and not written back yet, or equal to
    /// guest memory when last compared with nothing of the channel having
    /// written there since.
    fn current(&self, writes: &Writes) -> bool {
        let (start, end) = self.extent();
        self.dirty.is_some()
            || self
                .stamp
                .is_some_and(|stamp| writes.unchanged(stamp, start, end))
    }

    /// Stores the texels [`State::pack`] packed, the GPU being done: its
    /// bands into the mirror, and the texels from there to guest memory.
    fn store_packed(&mut self, packed: Packed, memory: &dyn GpuMemory) -> Result<(), Unmodeled> {
        let length = packed.bands.len();
        // SAFETY: the staging range holds `length` bytes the finished GPU
        // work wrote; nothing else uses it until the arena hands it out
        // again.
        let staged = unsafe {
            std::slice::from_raw_parts(
                packed
                    .buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(packed.offset),
                length,
            )
        };
        self.mirror.data_mut()[packed.bands].copy_from_slice(staged);
        self.mirror.store_rect(memory, packed.texels)
    }

    /// Writes `rect` (pixels) of the Metal texture to guest memory.
    fn write_back(&mut self, rect: Rect, memory: &dyn GpuMemory) -> Result<(), Unmodeled> {
        let bpp = self.surface.bytes_per_pixel as usize;
        let row = (rect.x1 - rect.x0) as usize * bpp;
        let layers = 1usize << (self.log2[0] + self.log2[1]);
        let planes: Vec<Vec<u8>> = (0..layers)
            .map(|layer| {
                let mut pixels = vec![0u8; row * (rect.y1 - rect.y0) as usize];
                // SAFETY: the buffer holds the region's rows; the texture
                // is shared storage and the GPU finished writing it.
                unsafe {
                    self.texture
                        .getBytes_bytesPerRow_bytesPerImage_fromRegion_mipmapLevel_slice(
                            NonNull::new(pixels.as_mut_ptr().cast()).expect("non-null"),
                            row,
                            0,
                            region(rect),
                            0,
                            layer,
                        );
                }
                pixels
            })
            .collect();
        let layout = Samples {
            log2: self.log2,
            ..Samples::SINGLE
        };
        let texels = layout.texels(rect);
        self.mirror
            .write_pixels(texels, &interleave(planes, rect, layout, bpp));
        self.mirror.store_rect(memory, texels)
    }
}

/// The texels of pixels `rect` of a surface whose samples are `layout`
/// (see `multisample`), row by row, from `planes`, the pixels of each
/// sample, row by row: the guest's layout from the layers of a resident.
fn interleave(mut planes: Vec<Vec<u8>>, rect: Rect, layout: Samples, bpp: usize) -> Vec<u8> {
    if planes.len() == 1 {
        return planes.pop().expect("one plane");
    }
    let texels = layout.texels(rect);
    let [lx, ly] = layout.log2;
    let (width, height) = ((rect.x1 - rect.x0) as usize, (rect.y1 - rect.y0) as usize);
    let texel_row = (texels.x1 - texels.x0) as usize * bpp;
    let mut out = vec![0u8; texel_row * (texels.y1 - texels.y0) as usize];
    for (sample, plane) in planes.iter().enumerate() {
        let [cx, cy] = layout.cell(sample).map(|c| c as usize);
        for y in 0..height {
            for x in 0..width {
                let to = ((y << ly | cy) * texel_row) + (x << lx | cx) * bpp;
                let from = (y * width + x) * bpp;
                out[to..to + bpp].copy_from_slice(&plane[from..from + bpp]);
            }
        }
    }
    out
}

/// The inverse of [`interleave`]: the pixels of each sample of `rect`
/// from the texels holding them.
fn deinterleave(texels: Vec<u8>, rect: Rect, layout: Samples, bpp: usize) -> Vec<Vec<u8>> {
    if layout.is_single() {
        return vec![texels];
    }
    let [lx, ly] = layout.log2;
    let (width, height) = ((rect.x1 - rect.x0) as usize, (rect.y1 - rect.y0) as usize);
    let texel_row = (width << lx) * bpp;
    (0..layout.count())
        .map(|sample| {
            let [cx, cy] = layout.cell(sample).map(|c| c as usize);
            let mut plane = vec![0u8; width * height * bpp];
            for y in 0..height {
                for x in 0..width {
                    let from = ((y << ly | cy) * texel_row) + (x << lx | cx) * bpp;
                    let to = (y * width + x) * bpp;
                    plane[to..to + bpp].copy_from_slice(&texels[from..from + bpp]);
                }
            }
            plane
        })
        .collect()
}

/// Draws a clear's triangle (see [`CLEAR_SOURCE`]) into every layer of
/// targets with `samples`.
///
/// # Safety
///
/// As [`bind`]: the encoder's pipeline is a clear pipeline, layered when
/// `samples` has more than one.
unsafe fn draw_clear(
    arena: &mut Arena,
    encoder: &Encoder,
    samples: Samples,
) -> Result<(), Unmodeled> {
    if samples.is_single() {
        // SAFETY: per the caller.
        unsafe { encoder.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3) };
        return Ok(());
    }
    for layer in 0..samples.count() as u32 {
        // SAFETY: per the caller.
        unsafe {
            bind(arena, encoder, Stage::Vertex, bytes_of_u32(&[layer]), 1)?;
            encoder.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3);
        }
    }
    Ok(())
}

/// What [`State::pack`] encoded: where the packed bands are staged, the
/// range of the mirror's data they replace and the texels written.
struct Packed {
    buffer: Retained<MetalBuffer>,
    offset: usize,
    bands: std::ops::Range<usize>,
    texels: Rect,
}

/// The words of the packing kernel ([`PACK_SOURCE`] `PackWords`): the
/// texel rectangle, texel size, samples (log2 x, y), the bytes of a row of
/// a pitch surface as the mirror holds it (0: block linear), the bytes
/// per band of 8 rows and the mirror data offset of the first band packed.
#[repr(C)]
struct PackWords {
    x0: u32,
    y0: u32,
    width: u32,
    height: u32,
    bpp: u32,
    lx: u32,
    ly: u32,
    pitch: u32,
    band_bytes: u32,
    start: u32,
}

/// Offset of texel (`x`, `y`) from the first band packed, as
/// [`PACK_SOURCE`] `guest_offset` computes it: bands of 8 rows one after
/// the other; in a block-linear band the GOBs left to right, 512 bytes
/// each, a 16-byte unit of byte column c and row r at ((c & 0x20) << 3) |
/// ((r & 6) << 5) | ((c & 0x10) << 1) | ((r & 1) << 4) (as
/// `surface::Window`); in a pitch band 8 rows of
/// `pitch` bytes (the mirror's rows, without the surface's padding).
fn guest_offset(p: &PackWords, x: u32, y: u32) -> u32 {
    let band = (y >> 3) * p.band_bytes;
    let column = x * p.bpp;
    let within = if p.pitch != 0 {
        (y & 7) * p.pitch + column
    } else {
        let (c, r) = (column & 63, y & 7);
        (column >> 6) * 512
            + (((c & 0x20) << 3) | ((r & 6) << 5) | ((c & 0x10) << 1) | ((r & 1) << 4))
            + (c & 0xF)
    };
    band + within - p.start
}

/// The packing kernels (`State::pack`): one thread per texel reads it (the
/// layer of its sample for multisampled residents, the sample from the
/// texel's place in its pixel's block as `multisample::Samples::cell`
/// places them) through a view of the texture as unsigned words and
/// stores its bytes at its guest offset (`guest_offset`).
const PACK_SOURCE: &str = r#"#include <metal_stdlib>
using namespace metal;
struct PackWords { uint x0, y0, width, height, bpp, lx, ly, pitch, band_bytes, start; };
static inline uint guest_offset(constant PackWords& p, uint x, uint y) {
    const uint band = (y >> 3) * p.band_bytes;
    const uint column = x * p.bpp;
    uint within;
    if (p.pitch != 0u) {
        within = (y & 7u) * p.pitch + column;
    } else {
        const uint c = column & 63u, r = y & 7u;
        within = (column >> 6) * 512u + (((c & 0x20u) << 3) | ((r & 6u) << 5) | ((c & 0x10u) << 1) | ((r & 1u) << 4)) + (c & 0xFu);
    }
    return band + within - p.start;
}
static inline void store(device uchar* out, uint4 v, uint bpp) {
    for (uint i = 0u; i < bpp; i++) {
        out[i] = uchar(v[min(i >> 2, 3u)] >> (8u * (i & 3u)));
    }
}
kernel void pack_plain(texture2d<uint, access::read> t [[texture(0)]], device uchar* out [[buffer(0)]], constant PackWords& p [[buffer(1)]], uint2 g [[thread_position_in_grid]]) {
    if (g.x >= p.width || g.y >= p.height) { return; }
    const uint x = p.x0 + g.x, y = p.y0 + g.y;
    store(out + guest_offset(p, x, y), t.read(uint2(x, y)), p.bpp);
}
kernel void pack_layers(texture2d_array<uint, access::read> t [[texture(0)]], device uchar* out [[buffer(0)]], constant PackWords& p [[buffer(1)]], uint2 g [[thread_position_in_grid]]) {
    if (g.x >= p.width || g.y >= p.height) { return; }
    const uint x = p.x0 + g.x, y = p.y0 + g.y;
    const uint cx = x & ((1u << p.lx) - 1u), cy = y & ((1u << p.ly) - 1u);
    const uint sample = (cx & 1u) | ((cy & 1u) << 1) | ((cx & 2u) << 1) | ((cy & 2u) << 2);
    store(out + guest_offset(p, x, y), t.read(uint2(x >> p.lx, y >> p.ly), sample), p.bpp);
}
"#;

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
/// The same triangle into layer `layer` (sample `layer` of a
/// multisampled target).
struct LayeredVertex { float4 position [[position]]; uint layer [[render_target_array_index]]; };
vertex LayeredVertex vertex_layered(uint id [[vertex_id]], constant uint& layer [[buffer(1)]]) {
    float2 p = float2(id == 1 ? 3.0 : -1.0, id == 2 ? -3.0 : 1.0);
    LayeredVertex out;
    out.position = float4(p, 0.0, 1.0);
    out.layer = layer;
    return out;
}
fragment float4 clear_main(const device float4* color [[buffer(0)]]) { return color[0]; }
"#;

/// The depth clear's fragment functions (after [`CLEAR_SOURCE`]): the
/// texel's words kept where `masks` (keep, set per word) keep them, set
/// elsewhere (see [`Renderer::clear_depth`]).
const DEPTH_CLEAR_FUNCTIONS: &str = r#"
fragment uint clear_word(uint old [[color(0)]], constant uint* masks [[buffer(0)]]) {
    return (old & masks[0]) | masks[1];
}
fragment uint2 clear_words(uint2 old [[color(0)]], constant uint* masks [[buffer(0)]]) {
    return (old & uint2(masks[0], masks[2])) | uint2(masks[1], masks[3]);
}
"#;

/// The comparison numbering of [`Translated::source`].
fn compare_number(compare: Compare) -> u32 {
    match compare {
        Compare::Never => 0,
        Compare::Less => 1,
        Compare::Equal => 2,
        Compare::LessEqual => 3,
        Compare::Greater => 4,
        Compare::NotEqual => 5,
        Compare::GreaterEqual => 6,
        Compare::Always => 7,
    }
}

/// A stencil face as the seven words of [`Translated::source`].
fn stencil_words(face: &StencilFace) -> [u32; 7] {
    let op = |op: StencilOp| match op {
        StencilOp::Keep => 0,
        StencilOp::Zero => 1,
        StencilOp::Replace => 2,
        StencilOp::IncrementClamp => 3,
        StencilOp::DecrementClamp => 4,
        StencilOp::Invert => 5,
        StencilOp::IncrementWrap => 6,
        StencilOp::DecrementWrap => 7,
    };
    [
        op(face.fail),
        op(face.depth_fail),
        op(face.pass),
        compare_number(face.compare),
        u32::from(face.reference),
        u32::from(face.compare_mask),
        u32::from(face.write_mask),
    ]
}

fn compile_library(gpu: &Gpu, source: &str) -> Result<Retained<Library>, Unmodeled> {
    let options = MTLCompileOptions::new();
    options.setMathMode(MTLMathMode::Safe);
    gpu.device
        .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&options))
        .map_err(|e| Unmodeled(format!("Metal: shader compilation failed: {e}\n{source}")))
}

/// A pipeline of vertex function `vertex` and `fragment` with color
/// `attachments` (pixel format, blend, write mask, program output) and,
/// after them, the depth buffer's words `depth` (pixel format, written),
/// never blended; `layered` when the vertex function picks the layer of
/// targets of one layer per sample.
fn build_pipeline(
    gpu: &Gpu,
    library: &Library,
    vertex: &str,
    fragment: &str,
    attachments: &[(usize, Option<Blend>, u8, usize)],
    depth: Option<(MTLPixelFormat, bool)>,
    layered: bool,
) -> Result<Retained<Pipeline>, Unmodeled> {
    let vertex = library
        .newFunctionWithName(&NSString::from_str(vertex))
        .ok_or_else(|| Unmodeled("Metal: no vertex function".into()))?;
    let fragment = library
        .newFunctionWithName(&NSString::from_str(fragment))
        .ok_or_else(|| Unmodeled("Metal: no fragment function".into()))?;
    let descriptor = MTLRenderPipelineDescriptor::new();
    if layered {
        // Layered rendering needs the primitive class [public: Apple,
        // "Rendering to multiple texture slices in a draw command"].
        // SAFETY: a valid topology class.
        unsafe { descriptor.setInputPrimitiveTopology(MTLPrimitiveTopologyClass::Triangle) };
    }
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
    if let Some((pixel_format, write)) = depth {
        // SAFETY: fewer than 8 attachments with the colors.
        let attachment = unsafe {
            descriptor
                .colorAttachments()
                .objectAtIndexedSubscript(attachments.len())
        };
        attachment.setPixelFormat(pixel_format);
        attachment.setWriteMask(if write {
            MTLColorWriteMask::All
        } else {
            MTLColorWriteMask::None
        });
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

/// A render target texture: 2D, or with more than one `layers` a 2D
/// array of that many.
fn new_target(
    gpu: &Gpu,
    pixel_format: MTLPixelFormat,
    width: u32,
    height: u32,
    layers: usize,
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
    if layers > 1 {
        descriptor.setTextureType(MTLTextureType::Type2DArray);
        // SAFETY: at most 16 layers, within Metal's array limit.
        unsafe { descriptor.setArrayLength(layers) };
    }
    descriptor.setStorageMode(MTLStorageMode::Shared);
    descriptor.setUsage(
        MTLTextureUsage::RenderTarget
            | MTLTextureUsage::ShaderRead
            | MTLTextureUsage::ShaderWrite
            | MTLTextureUsage::PixelFormatView,
    );
    gpu.device
        .newTextureWithDescriptor(&descriptor)
        .ok_or_else(|| Unmodeled("Metal: render target allocation failed".into()))
}

/// Loads pixels `rect` of `window` (whose surface holds samples `log2`)
/// into the texture: its layer per sample.
fn upload_surface(texture: &MetalTexture, window: &mut Window, rect: Rect, log2: [u32; 2]) {
    let bpp = window.surface().bytes_per_pixel as usize;
    let layout = Samples {
        log2,
        ..Samples::SINGLE
    };
    let texels = layout.texels(rect);
    let mut bytes =
        vec![0u8; (texels.x1 - texels.x0) as usize * bpp * (texels.y1 - texels.y0) as usize];
    window.read_pixels(texels, &mut bytes);
    let row = (rect.x1 - rect.x0) as usize * bpp;
    for (layer, mut pixels) in deinterleave(bytes, rect, layout, bpp)
        .into_iter()
        .enumerate()
    {
        // SAFETY: the buffer holds the region's rows.
        unsafe {
            texture.replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage(
                region(rect),
                0,
                layer,
                NonNull::new(pixels.as_mut_ptr().cast()).expect("non-null"),
                row,
                0,
            );
        }
    }
}

/// The descriptor of a sampled RGBA32Float texture of `kind`: `size`
/// texels (width, height and, for a volume, depth) at its first level,
/// `layers` layers (an array's; a cube has its six faces).
fn texture_descriptor(
    kind: TextureKind,
    size: [u32; 3],
    layers: u32,
    levels: u32,
) -> Retained<MTLTextureDescriptor> {
    // SAFETY: plain descriptor construction.
    let descriptor = unsafe {
        MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
            MTLPixelFormat::RGBA32Float,
            size[0] as usize,
            size[1] as usize,
            levels > 1,
        )
    };
    match kind {
        TextureKind::Flat => {}
        TextureKind::Array => {
            descriptor.setTextureType(MTLTextureType::Type2DArray);
            // SAFETY: a TIC's layer count, within Metal's array limit.
            unsafe { descriptor.setArrayLength(layers as usize) };
        }
        TextureKind::Cube => descriptor.setTextureType(MTLTextureType::TypeCube),
        TextureKind::Volume => {
            descriptor.setTextureType(MTLTextureType::Type3D);
            // SAFETY: a TIC's depth, within Metal's limit.
            unsafe { descriptor.setDepth(size[2] as usize) };
        }
    }
    // SAFETY: the level count fits the first level's dimensions (the
    // TIC was validated by the decoder).
    unsafe { descriptor.setMipmapLevelCount(levels as usize) };
    descriptor.setStorageMode(MTLStorageMode::Shared);
    descriptor.setUsage(MTLTextureUsage::ShaderRead);
    descriptor
}

/// A sampled texture: the decoded levels from the base level to the
/// maximum level as RGBA32Float, as a Metal texture of `kind` (which
/// `format::check_texture` found the texture to fit).
fn upload_texture(
    gpu: &Gpu,
    texture: &Texture,
    kind: TextureKind,
) -> Result<Retained<MetalTexture>, Unmodeled> {
    let tic = texture.tic();
    let levels = tic.max_level - tic.base_level + 1;
    let (width, height, depth, _) = texture
        .level(tic.base_level)
        .ok_or_else(|| Unmodeled("Metal: texture base level".into()))?;
    let layers = tic.shape.layers;
    let descriptor = texture_descriptor(kind, [width, height, depth], layers, levels);
    let metal = gpu
        .device
        .newTextureWithDescriptor(&descriptor)
        .ok_or_else(|| Unmodeled("Metal: texture allocation failed".into()))?;
    for level in 0..levels {
        let (w, h, d, texels) = texture
            .level(tic.base_level + level)
            .ok_or_else(|| Unmodeled("Metal: texture level".into()))?;
        let image = w as usize * h as usize;
        let whole = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width: w as usize,
                height: h as usize,
                depth: if kind == TextureKind::Volume {
                    d as usize
                } else {
                    1
                },
            },
        };
        // A volume's slices go in one image block; an array's layers and
        // a cube's faces are slices of the texture.
        let (slices, bytes_per_image) = if kind == TextureKind::Volume {
            (1, image * 16)
        } else {
            (layers as usize, 0)
        };
        for slice in 0..slices {
            let from = &texels[slice * image..];
            // SAFETY: `texels` holds the level's RGBA32F texels, layers
            // then slices, rows and columns (`Texture::level`): `image`
            // texels a layer or slice from `from` on.
            unsafe {
                metal.replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage(
                    whole,
                    level as usize,
                    slice,
                    NonNull::new(from.as_ptr() as *mut _).expect("non-null"),
                    w as usize * 16,
                    bytes_per_image,
                );
            }
        }
    }
    Ok(metal)
}

/// What a texture slot of a draw samples.
enum Sampled {
    /// A texture decoded from guest memory (`resources::TextureCache`).
    Decoded(Arc<Texture>),
    /// A texture with levels held as render targets ([`Composite`]).
    Composite(Retained<MetalTexture>),
    /// A texture in its own format, made of its bytes
    /// (`resources::RawTexture`).
    Native(Arc<RawTexture>, MTLPixelFormat),
}

/// Bytes of guest textures uploaded before a draw waits for the GPU and
/// drops the uploads of textures the caches no longer hold.
const UPLOAD_LIMIT: usize = 1 << 30;

/// A sampled texture in its own format `pixel_format`
/// (`format::native_format`): the levels from the base level to the
/// maximum level, the header's swizzle as the texture's.
fn upload_native(
    gpu: &Gpu,
    raw: &RawTexture,
    kind: TextureKind,
    pixel_format: MTLPixelFormat,
) -> Result<Retained<MetalTexture>, Unmodeled> {
    let tic = &raw.tic;
    let levels = Texture::linear(&raw.bytes, tic)?;
    let levels = levels
        .get(tic.base_level as usize..=tic.max_level as usize)
        .ok_or_else(|| Unmodeled("Metal: texture levels".into()))?;
    let base = &levels[0];
    let layers = tic.shape.layers;
    let descriptor = texture_descriptor(
        kind,
        [base.width, base.height, base.depth],
        layers,
        levels.len() as u32,
    );
    descriptor.setPixelFormat(pixel_format);
    let source = |s: u8| match s {
        0 => MTLTextureSwizzle::Zero,
        2 => MTLTextureSwizzle::Red,
        3 => MTLTextureSwizzle::Green,
        4 => MTLTextureSwizzle::Blue,
        5 => MTLTextureSwizzle::Alpha,
        _ => MTLTextureSwizzle::One,
    };
    descriptor.setSwizzle(MTLTextureSwizzleChannels {
        red: source(tic.swizzle[0]),
        green: source(tic.swizzle[1]),
        blue: source(tic.swizzle[2]),
        alpha: source(tic.swizzle[3]),
    });
    let metal = gpu
        .device
        .newTextureWithDescriptor(&descriptor)
        .ok_or_else(|| Unmodeled("Metal: texture allocation failed".into()))?;
    for (index, level) in levels.iter().enumerate() {
        let volume = kind == TextureKind::Volume;
        let whole = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width: level.width as usize,
                height: level.height as usize,
                depth: if volume { level.depth as usize } else { 1 },
            },
        };
        let image = level.row_bytes * level.rows;
        let (slices, bytes_per_image) = if volume {
            (1, image)
        } else {
            (layers as usize, 0)
        };
        for slice in 0..slices {
            let from = &level.bytes[slice * image..];
            // SAFETY: `level.bytes` holds the level's blocks, layers
            // then slices and rows (`Texture::linear`): `image` bytes a
            // layer or slice from `from` on.
            unsafe {
                metal.replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage(
                    whole,
                    index,
                    slice,
                    NonNull::new(from.as_ptr() as *mut _).expect("non-null"),
                    level.row_bytes,
                    bytes_per_image,
                );
            }
        }
    }
    Ok(metal)
}

/// Composites kept \[assumption: a guest renders into a handful of the
/// textures it samples (a texture atlas); a dropped one is made
/// again when next sampled\].
const COMPOSITE_LIMIT: usize = 16;

/// The most global memory bound for one pixel program \[assumption: what
/// a constant names with a larger size is no buffer descriptor\].
const GLOBAL_LIMIT: usize = 16 << 20;

/// Bytes of global memory buffers kept for reuse.
const GLOBAL_BUDGET: usize = 128 << 20;

/// Bytes of global memory uploaded before the renderer waits for the
/// GPU, which frees the buffers those replaced.
const GLOBAL_PENDING: usize = 128 << 20;

/// Arena chunks a frame's recorded work may fill before a draw waits
/// for the GPU, which frees them all: bounds the draw data held at once
/// (16 MiB a chunk).
const ARENA_LIMIT: usize = 16;

/// A pixel program's global memory as bound ([`Renderer::global_memory`]).
struct GlobalMemory {
    /// The buffer's bytes: the table, then the regions.
    content: Vec<u8>,
    /// When the regions were last known to equal guest memory.
    stamp: Stamp,
    buffer: Retained<MetalBuffer>,
}

/// A texture some of whose levels are render targets held on the GPU
/// (a guest may draw level 0 of a four-level atlas every frame and
/// samples it right away): a Metal texture as [`upload_texture`] makes,
/// whose levels that are residents are filled from them on the GPU
/// ([`State::convert`], whenever they were rendered to since) and the
/// others from guest memory. Sampling it needs no write-back, decode or
/// upload of what was rendered; guest memory gets the rendered levels as
/// any render target's, at the next write-back.
struct Composite {
    /// RGBA32Float, the levels from the header's base level to its
    /// maximum level.
    texture: Retained<MetalTexture>,
    /// The same levels.
    levels: Vec<CompositeLevel>,
    /// For each component the header's swizzle yields, which of the
    /// resident's red, green, blue and alpha (0-3) as Metal reads them,
    /// zero (4) or one (5) it is.
    select: [u32; 4],
    /// The bytes of the texture's footprint the levels from memory were
    /// decoded from (zeros where a resident's level lies), and when they
    /// were last known to equal memory.
    raw: Vec<u8>,
    stamp: Stamp,
    /// [`State::clock`] when last sampled.
    used: u64,
}

struct CompositeLevel {
    /// The level as a surface, and its bytes in the footprint.
    surface: Surface,
    range: std::ops::Range<usize>,
    /// For a level that is a resident: the view of the level the kernel
    /// writes, and the resident's version it last converted.
    resident: Option<(Retained<MetalTexture>, Option<u64>)>,
}

impl Composite {
    /// The composite of `tic`, whose `levels` ([`level_surfaces`]) are
    /// residents where `residents` says and otherwise taken from
    /// `decoded` (a decode of `raw`); resident levels are filled by the
    /// next [`State::convert`].
    #[allow(clippy::too_many_arguments)]
    fn new(
        gpu: &Gpu,
        tic: &Tic,
        levels: &[(Surface, std::ops::Range<usize>)],
        residents: &[bool],
        components: [u32; 4],
        decoded: Option<&Texture>,
        raw: Vec<u8>,
        stamp: Stamp,
    ) -> Result<Composite, Unmodeled> {
        let (width, height) = (levels[0].0.width, levels[0].0.height);
        // SAFETY: plain descriptor construction.
        let descriptor = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat::RGBA32Float,
                width as usize,
                height as usize,
                levels.len() > 1,
            )
        };
        // SAFETY: the level count fits the base level's dimensions (the
        // header was validated by `level_surfaces`).
        unsafe { descriptor.setMipmapLevelCount(levels.len()) };
        descriptor.setStorageMode(MTLStorageMode::Shared);
        descriptor.setUsage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);
        let texture = gpu
            .device
            .newTextureWithDescriptor(&descriptor)
            .ok_or_else(|| Unmodeled("Metal: texture allocation failed".into()))?;
        let mut made = Vec::with_capacity(levels.len());
        for (index, ((surface, range), &resident)) in levels.iter().zip(residents).enumerate() {
            let resident = if resident {
                // SAFETY: one level and the one slice of the texture.
                let view = unsafe {
                    texture.newTextureViewWithPixelFormat_textureType_levels_slices(
                        MTLPixelFormat::RGBA32Float,
                        MTLTextureType::Type2D,
                        NSRange::new(index, 1),
                        NSRange::new(0, 1),
                    )
                }
                .ok_or_else(|| Unmodeled("Metal: level view".into()))?;
                Some((view, None))
            } else {
                let (w, h, _, texels) = decoded
                    .and_then(|decoded| decoded.level(tic.base_level + index as u32))
                    .ok_or_else(|| Unmodeled("Metal: texture level".into()))?;
                // SAFETY: `texels` holds the level's w*h RGBA32F texels.
                unsafe {
                    texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                        region(Rect {
                            x0: 0,
                            y0: 0,
                            x1: w,
                            y1: h,
                        }),
                        index,
                        NonNull::new(texels.as_ptr() as *mut _).expect("non-null"),
                        w as usize * 16,
                    );
                }
                None
            };
            made.push(CompositeLevel {
                surface: *surface,
                range: range.clone(),
                resident,
            });
        }
        // As `Texture::decode` applies the swizzle.
        let select = tic.swizzle.map(|s| match s {
            2..=5 => components[usize::from(s - 2)],
            6 | 7 => 5,
            _ => 4,
        });
        Ok(Composite {
            texture,
            levels: made,
            select,
            raw,
            stamp,
            used: 0,
        })
    }
}

/// The levels of `tic` from its base level to its maximum level as the
/// surfaces a render target at each would be, with the bytes of each in
/// the texture's footprint; `None` for a texture that is not a plain 2D
/// one of one layer and uncompressed one-texel blocks without sRGB
/// decoding.
fn level_surfaces(tic: &Tic) -> Option<Vec<(Surface, std::ops::Range<usize>)>> {
    use super::surface::Layout;
    use super::texture::format::Encoding;
    let format = tic.format;
    if tic.target != Target::Texture2D
        || tic.array
        || tic.shape.layers != 1
        || tic.shape.depth != 1
        || tic.srgb
        || !matches!(format.encoding, Encoding::Packed(_))
        || (format.block_width, format.block_height) != (1, 1)
        || tic.max_level < tic.base_level
    {
        return None;
    }
    let bytes_per_pixel = format.bytes_per_block;
    if let Some(pitch) = tic.pitch {
        return Some(vec![(
            Surface {
                address: tic.address,
                width: tic.shape.width,
                height: tic.shape.height,
                bytes_per_pixel,
                layout: Layout::Pitch { pitch },
            },
            0..pitch as usize * tic.shape.height as usize,
        )]);
    }
    let image = super::texture::layout::block_linear_image(&tic.shape, tic.block).ok()?;
    let levels = image
        .levels
        .get(tic.base_level as usize..=tic.max_level as usize)?;
    Some(
        levels
            .iter()
            .map(|level| {
                (
                    Surface {
                        address: tic.address + level.offset,
                        width: level.width,
                        height: level.height,
                        bytes_per_pixel,
                        layout: Layout::BlockLinear(level.surface.block()),
                    },
                    level.offset as usize..(level.offset + level.size) as usize,
                )
            })
            .collect(),
    )
}

/// For each of the red, green, blue and alpha a texture header `tic`
/// decodes from a texel's bytes, which of the red, green, blue and alpha
/// (0-3) Metal reads from a resident of color format `format` holding the
/// same bytes is the same value; `None` unless that holds for every
/// texel. A8B8G8R8 UNORM headers over the four 8-bit UNORM target
/// layouts: both sides are the byte / 255 (`texture::format::unorm_to_f32`;
/// Metal's conversion of normalized formats [public: Apple, Metal Shading
/// Language Specification, "Conversion rules for normalized integer
/// pixel data types"]). An sRGB target's resident holds its bytes in the
/// linear format ([`format::pixel_format`]), so a header without sRGB
/// decoding reads them raw there too.
fn shared_components(tic: &Tic, format: ColorFormat) -> Option<[u32; 4]> {
    if tic.format.raw != 0x08 || tic.types != [ComponentType::Unorm; 4] || tic.srgb {
        return None;
    }
    match format {
        ColorFormat::Rgba8Unorm | ColorFormat::Rgba8Srgb => Some([0, 1, 2, 3]),
        ColorFormat::Bgra8Unorm | ColorFormat::Bgra8Srgb => Some([2, 1, 0, 3]),
        _ => None,
    }
}

/// The kernel filling a level of a [`Composite`] from the resident that
/// is the level: each texel as Metal reads it from the resident, its
/// components selected by `select` ([`Composite::select`]).
const CONVERT_SOURCE: &str = r#"#include <metal_stdlib>
using namespace metal;
kernel void convert_level(texture2d<float, access::read> source [[texture(0)]], texture2d<float, access::write> level [[texture(1)]], constant uint* select [[buffer(0)]], uint2 g [[thread_position_in_grid]]) {
    if (g.x >= level.get_width() || g.y >= level.get_height()) { return; }
    const float4 c = source.read(g);
    const float v[6] = { c.x, c.y, c.z, c.w, 0.0f, 1.0f };
    level.write(float4(v[select[0]], v[select[1]], v[select[2]], v[select[3]]), g);
}
"#;

fn build_sampler(gpu: &Gpu, tsc: &Tsc, (min, max): (f32, f32)) -> Retained<Sampler> {
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
    descriptor.setRAddressMode(address(tsc.wrap[2]));
    if tsc.min_filter == 3 {
        descriptor.setMaxAnisotropy(tsc.anisotropy as usize);
    }
    descriptor.setBorderColor(
        format::border_color(tsc).unwrap_or(MTLSamplerBorderColor::TransparentBlack),
    );
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
/// stream when small (no allocation, as Metal advises below 4 KiB), into
/// the frame's arena otherwise.
///
/// # Safety
///
/// `index` is within the stage function's buffer bindings.
unsafe fn bind(
    arena: &mut Arena,
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
    let (buffer, offset) = arena.alloc(data)?;
    // SAFETY: Metal retains the buffer; `index` per the caller.
    unsafe {
        match stage {
            Stage::Vertex => encoder.setVertexBuffer_offset_atIndex(Some(&buffer), offset, index),
            Stage::Fragment => {
                encoder.setFragmentBuffer_offset_atIndex(Some(&buffer), offset, index)
            }
        }
    }
    Ok(())
}

/// Shared buffer space for the data of the draws one command buffer
/// records (vertex streams, indices, large bindings), handed out in
/// order and taken back whole once the command buffer completed
/// ([`State::finish`]): allocating a Metal buffer per binding costs a
/// kernel call each.
#[derive(Default)]
struct Arena {
    chunks: Vec<Retained<MetalBuffer>>,
    /// The chunk being filled and the bytes used of it.
    current: usize,
    used: usize,
}

impl Arena {
    /// Bytes per chunk \[assumption: a frame's draw data of a game at a
    /// few chunks; larger data gets a chunk of its own size\].
    const CHUNK: usize = 16 << 20;
    /// Offsets are aligned for any buffer binding [public: Apple, Metal
    /// feature set tables, "Minimum constant buffer offset alignment": 256
    /// bytes on macOS GPUs].
    const ALIGN: usize = 256;

    /// `data` copied into the arena: its buffer and offset.
    fn alloc(&mut self, data: &[u8]) -> Result<(Retained<MetalBuffer>, usize), Unmodeled> {
        let length = data.len().max(4);
        loop {
            if let Some(chunk) = self.chunks.get(self.current) {
                let offset = self.used.next_multiple_of(Self::ALIGN);
                if offset + length <= chunk.length() {
                    // SAFETY: the range is inside the shared buffer, which
                    // no recorded GPU work uses (see `reset`).
                    unsafe {
                        let base = chunk.contents().as_ptr().cast::<u8>().add(offset);
                        std::ptr::copy_nonoverlapping(data.as_ptr(), base, data.len());
                    }
                    self.used = offset + length;
                    return Ok((chunk.clone(), offset));
                }
                self.current += 1;
                self.used = 0;
                continue;
            }
            let size = length.max(Self::CHUNK);
            let chunk = Renderer::gpu()
                .device
                .newBufferWithLength_options(size, MTLResourceOptions::StorageModeShared)
                .ok_or_else(|| Unmodeled("Metal: buffer allocation failed".into()))?;
            // A chunk past the usual size is not kept for later frames.
            self.chunks.insert(self.current, chunk);
        }
    }

    /// Every chunk free again; an oversized one is dropped.
    fn reset(&mut self) {
        self.chunks.retain(|chunk| chunk.length() <= Self::CHUNK);
        self.current = 0;
        self.used = 0;
    }

    /// Chunks holding data of work recorded since the last reset.
    fn chunks_in_use(&self) -> usize {
        if self.chunks.is_empty() {
            0
        } else {
            self.current + 1
        }
    }
}

/// A stage's constant buffers as its function binds them (see
/// `Translated::source`), by slot, with the sizes of the readable parts
/// of indexed slots; or why they cannot be bound. Slots read at register
/// offsets (LDC) are their readable part of the bound range, in words; a
/// bound range can reach past the memory its owner mapped (see
/// `pipeline::read_mapped`); an indexed read there, which the interpreter
/// refuses (hardware would fault), reads 0 on Metal. No statically known
/// read may go there. Borrowed from `supplied` (one per slot) unless
/// padded.
#[allow(clippy::type_complexity)]
fn stage_constants<'c>(
    indirect: &std::collections::BTreeSet<u32>,
    statics: &std::collections::BTreeSet<(u32, u32)>,
    supplied: &'c [(Vec<u8>, usize)],
) -> Result<([Option<Cow<'c, [u8]>>; 18], [u32; 18]), String> {
    let mut constants: [Option<Cow<'c, [u8]>>; 18] = Default::default();
    let mut sizes = [0u32; 18];
    for &slot in indirect {
        let Some((data, size)) = supplied.get(slot as usize) else {
            return Err(format!("constant buffer slot {slot}"));
        };
        let readable = data.len().min(*size);
        let padded = readable.next_multiple_of(4).max(4);
        constants[slot as usize] = Some(if padded == readable {
            Cow::Borrowed(&data[..readable])
        } else {
            let mut buffer = data[..readable].to_vec();
            buffer.resize(padded, 0);
            Cow::Owned(buffer)
        });
        sizes[slot as usize] = readable as u32;
    }
    for &(slot, offset) in statics {
        let Some((data, size)) = supplied.get(slot as usize) else {
            return Err(format!("constant buffer slot {slot}"));
        };
        let end = offset as usize + 4;
        if end > data.len() && end <= *size {
            return Err("constant read past the mapped part".into());
        }
        if indirect.contains(&slot) {
            // Bounds-checked by the program against the sizes.
            continue;
        }
        let buffer = constants[slot as usize].get_or_insert(Cow::Borrowed(data.as_slice()));
        if buffer.len() < end {
            buffer.to_mut().resize(end, 0);
        }
    }
    Ok((constants, sizes))
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

    fn probe(&self, address: u64, buffer: &mut [u8]) -> Option<()> {
        self.renderer
            .before_access(self.memory, self.writes, address, buffer.len())
            .ok()?;
        self.memory.probe(address, buffer)
    }
}
