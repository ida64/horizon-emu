//! The Metal renderer: 3D draws and clears, and compute launches
//! ([`compute`]), of a channel run on the host GPU through Metal, with the
//! software renderer and the interpreter as the reference and the
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
//! rectangle). A write-back stores the dirty pixels in guest memory, in
//! the surface's own layout. That happens:
//!
//! - when the channel signals: semaphore releases, reports and syncpoint
//!   increments of its engines and of the pushbuffer executor
//!   ([`super::Hooks::release`], [`super::Engines::release`]), except
//!   for the targets held on demand (below). The write-back is recorded
//!   after the work so far (targets packed into their guest bytes on the
//!   GPU, global memory regions copied) and finished once the GPU has
//!   done it, and the release is made then ([`Renderer::release`],
//!   [`Renderer::complete`]), the channel going on with later work
//!   meanwhile, as hardware does. The channel finishes them as the GPU
//!   gets done, between methods, and waits for them when it has nothing
//!   to run ([`super::Engines::idle`]);
//! - before any other access of the channel's engines to memory that
//!   overlaps a dirty target or an unfinished write-back: all engine
//!   memory accesses go through [`Coherent`], which writes back first.
//!   That covers copies, 2D blits the GPU does not do ([`blit`]), inline
//!   uploads, texture sampling of a render target that is not sampled on
//!   the GPU (below), and constant and vertex data;
//! - before a draw or clear Metal cannot express runs in the software
//!   renderer, and before a machine snapshot ([`Renderer::settle`]).
//!
//! Until its release is made, nothing outside the channel can tell
//! whether a write-back happened: the guest CPU, other channels and the
//! display learn that rendering finished only from a release (NVN
//! fences are syncpoint increments or semaphore releases the CPU,
//! nvservices or another channel waits on; the compositor flips a buffer
//! once its fence passed), and on hardware too, memory written by work
//! not yet followed by a release may or may not hold the result. Ends of
//! a channel's turn, wait-for-idle, `FB_FLUSH` and L2 flushes are not
//! signals: they let no one observe anything until a signal follows.
//!
//! Before a resident is used, its mirror is compared with guest memory
//! (and the Metal texture reloaded, into a new texture so that recorded
//! work keeps the old one, if they differ) at most once per epoch of
//! [`Writes`]: after each synchronization of the channel with another
//! agent (new work submitted, a semaphore acquire or syncpoint wait
//! passed), and after the channel's own engines wrote memory overlapping
//! it. A resident whose write-back is unfinished holds what memory is
//! to hold. Anything else (the CPU, another channel) may only write a
//! surface the channel rendered or read once a release told it the GPU
//! is done with it; between, such a write races with the GPU, as on
//! hardware.
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
//! A texture whose levels (some or all, of any of its array layers or
//! cube faces) are current residents whose Metal texture gives the
//! header's values ([`shared_components`]: a color target of the format
//! that samples the header's bytes, read through sRGB decoding where the
//! header asks, or a ZF32 or Z16 depth buffer; the target may be wider or
//! taller than the level, as a GOB-aligned target is, [`holds_level`]) is
//! sampled from a [`Composite`]: a Metal texture whose levels are filled
//! from those residents on the GPU each time they were rendered to, and
//! from guest memory for the other levels. Guest memory is not involved:
//! no write-back, decode or upload of the rendered levels (a guest draws
//! the stages of its post-processing from one another's targets every
//! frame). A draw that renders to a level it samples samples what the
//! target held before the draw.
//!
//! A target drawn to in another color format of the same texel size as
//! the resident at its memory (the same surface) takes the resident's
//! texels as unsigned words on the GPU; a target replacing a surface
//! whole drops the residents inside its memory unwritten.
//!
//! # Global memory
//!
//! The memory programs address with `LDG`, `STG`, `RED` and `ATOM` lives
//! in one Metal buffer ([`Globals`]) that every draw's and launch's
//! program is given, with a table of where the regions its constants name
//! are in it: what a program stores, a later program loads without the
//! bytes leaving the GPU. A region programs may have stored to is written
//! back to guest memory as a render target is: at a signal, or before an
//! engine of the channel accesses it ([`Coherent`]).

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
pub mod compute;
mod format;
mod relayout;

use std::borrow::Cow;
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::{Arc, LazyLock, Mutex};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLBlendFactor, MTLBlendOperation, MTLBlitCommandEncoder, MTLBuffer, MTLColorWriteMask,
    MTLCommandBuffer,
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
use super::texture::format::{ColorFormat, DepthFormat};
use super::texture::{Texture, Tic, Tsc};
use super::{GpuMemory, Release, Unmodeled, WatchMark};

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

/// What [`flushes_subnormals`] runs: products, sums and a fused
/// multiply-add with subnormal operands (`v[2]`, `v[6]`) or a subnormal
/// result (`v[0] * v[1]`), and comparisons of subnormals.
const SUBNORMAL_PROBE: &str = r#"#include <metal_stdlib>
using namespace metal;
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
kernel void probe(const device float* v [[buffer(0)]], device uint* out [[buffer(1)]], uint g [[thread_position_in_grid]]) {
    out[0] = as_type<uint>(v[0] * v[1]);
    out[1] = as_type<uint>(v[2] * v[3]);
    out[2] = as_type<uint>(v[2] + v[4]);
    out[3] = as_type<uint>(fma(v[2], v[5], v[4]));
    out[4] = as_type<uint>((-v[0]) * v[1]);
    out[5] = as_type<uint>(v[6] * v[3]);
    out[6] = as_type<uint>(v[6] + v[6]);
    out[7] = v[2] == v[4] ? 1u : 0u;
    out[8] = v[2] > v[4] ? 1u : 0u;
    out[9] = v[6] < v[4] ? 1u : 0u;
    out[10] = v[2] != v[6] ? 1u : 0u;
    out[11] = as_type<uint>(v[2] * v[5]);
    out[12] = as_type<uint>(v[0] * v[1] + v[5]);
}
"#;

/// Whether the host GPU's float arithmetic flushes to zero as an FTZ
/// instruction of GM20B does (`shader::exec` `alu::flush`): a subnormal
/// operand of a product, a sum or a fused multiply-add counts as the
/// zero of its sign, a subnormal result is returned as the zero of its
/// sign, and comparisons take subnormals as zeros. Measured once, by
/// running [`SUBNORMAL_PROBE`] \[observed: an Apple M5 does; the Metal
/// Shading Language leaves it to the implementation\]. `false` where
/// the probe cannot run.
pub fn flushes_subnormals() -> bool {
    static FLUSHES: LazyLock<bool> = LazyLock::new(|| {
        let Some(gpu) = GPU.as_ref() else {
            return false;
        };
        let subnormal = f32::from_bits(0x0001_16c2);
        let inputs = [1e-30f32, 1e-10, subnormal, 1e10, 0.0, 1.0, -subnormal];
        let expected = [
            0,
            0,
            0,
            0,
            0x8000_0000,
            0x8000_0000,
            0x8000_0000,
            1,
            0,
            0,
            0,
            0,
            1.0f32.to_bits(),
        ];
        let run = || -> Option<Vec<u32>> {
            let library = compile_library(gpu, SUBNORMAL_PROBE).ok()?;
            let function = library.newFunctionWithName(&NSString::from_str("probe"))?;
            let pipeline = gpu
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .ok()?;
            let values = buffer(gpu, bytes_of(&inputs)).ok()?;
            let results = buffer(gpu, &[0xFF; 4 * 13]).ok()?;
            let command = gpu.queue.commandBuffer()?;
            let encoder = command.computeCommandEncoder()?;
            encoder.setComputePipelineState(&pipeline);
            // SAFETY: the kernel's two buffers, each of the size it
            // reads or writes.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&values), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&results), 0, 1);
            }
            let one = MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            };
            encoder.dispatchThreads_threadsPerThreadgroup(one, one);
            encoder.endEncoding();
            command.commit();
            command.waitUntilCompleted();
            let words = results.contents().cast::<u32>();
            // SAFETY: the buffer holds 13 words; the GPU finished.
            Some((0..13).map(|i| unsafe { words.add(i).read() }).collect())
        };
        run().is_some_and(|results| results == expected)
    });
    *FLUSHES
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
    /// Written back at signals whose write-backs have yet to reach guest
    /// memory ([`Pending`]), how many: the texture is current, the mirror
    /// and memory are not yet.
    pending: u32,
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
    /// Kernels of compute launches, by the address of the compute class's
    /// [`Kernel`](super::shader::Kernel) (`compute::Renderer::dispatch`).
    kernels: HashMap<usize, compute::KernelEntry>,
    /// Reasons draws shaded their vertices on the CPU (`pipeline::native`),
    /// counted.
    cpu_vertices: HashMap<String, u64>,
    /// The reason of the last draw left to the software renderer, and
    /// the time that renderer took by reason
    /// ([`Renderer::software_time`]).
    last_fallback: Option<String>,
    software_time: HashMap<String, std::time::Duration>,
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
    /// The kernels filling composites' levels, from color and from depth
    /// residents.
    convert_pipelines: [Option<Retained<ComputePipeline>>; 2],
    /// Counts uses of residents (`Resident::used`).
    clock: u64,
    fallbacks: HashMap<String, u64>,
    draws: u64,
    /// Texture fault words of the draws recorded since the last flush
    /// that bound placeholders, with each unbound slot and the refusal a
    /// sample of it is (see `shader::msl` `texture.rs`).
    faults: Vec<Faults>,
    /// Signals whose write-backs wait for the GPU, oldest first
    /// ([`Renderer::release`]).
    pending: std::collections::VecDeque<Pending>,
    /// What fragment functions tell the renderer ([`Feedback`]), made
    /// when first needed.
    feedback: Option<Feedback>,
    /// What an unbound texture slot of each kind is bound to: a 1x1
    /// texture and a sampler, never read into a result that reaches
    /// guest memory.
    placeholders: HashMap<TextureKind, (Retained<MetalTexture>, Retained<Sampler>)>,
    /// Buffer space for the frame's draw data.
    arena: Arena,
    /// The memory programs address through their constants
    /// ([`Renderer::global_tables`]), made when first needed.
    globals: Option<Globals>,
    /// The extents of the dirty residents and global memory regions,
    /// sorted and merged ([`Renderer::before_access`]); rebuilt when
    /// something became dirty since ([`Renderer::dirtied`]). It may hold
    /// extents written back since, which costs a needless look only.
    dirty_extents: Vec<(u64, u64)>,
    /// The packing kernels (`State::pack`), plain and layered.
    pack_pipelines: HashMap<bool, Retained<ComputePipeline>>,
    /// The relayout kernel ([`relayout`]), made when first needed.
    relayout_pipeline: Option<Retained<ComputePipeline>>,
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
            kernels: HashMap::new(),
            cpu_vertices: HashMap::new(),
            libraries: Lru::new(LIBRARY_LIMIT),
            pipelines: Lru::new(PIPELINE_LIMIT),
            clear_pipelines: HashMap::new(),
            textures: HashMap::new(),
            samplers: Lru::new(SAMPLER_LIMIT),
            clock: 0,
            fallbacks: HashMap::new(),
            last_fallback: None,
            software_time: HashMap::new(),
            draws: 0,
            faults: Vec::new(),
            pending: Default::default(),
            feedback: None,
            placeholders: HashMap::new(),
            native: HashMap::new(),
            texture_bytes: 0,
            arena: Arena::default(),
            globals: None,
            dirty_extents: Vec::new(),
            pack_pipelines: HashMap::new(),
            relayout_pipeline: None,
            composites: HashMap::new(),
            versions: 0,
            convert_pipelines: [None, None],
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
    /// The function counts the samples that pass ([`Draw::count_passed`]).
    count_passed: bool,
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
    /// The bytes of all streams, each at its offset.
    pub stream_bytes: Vec<u8>,
    /// How Metal's vertex ID names a vertex.
    pub vertices: NativeVertices,
    /// The triangles' vertices as Metal's indices, the provoking vertex
    /// first.
    pub indices: NativeIndices<'a>,
    /// The viewport words of `msl::NativeVertex::new` (without the layer).
    pub view: [f32; 9],
    /// The instance ID of the first instance, the instance per-instance
    /// streams step by for it, and how many instances are drawn (each
    /// the one after the last).
    pub instance: u32,
    pub fetch_instance: u32,
    pub instances: u32,
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

/// A vertex stream's bytes from its first fetched element on: where they
/// are in [`NativeDraw::stream_bytes`].
pub struct NativeStream {
    pub offset: u32,
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

/// Arena chunks the channel's work not yet done on the GPU may fill
/// before a release waits for the GPU \[assumption: a frame's draw data
/// is a chunk or two; work whose results nothing reads is still held to
/// the GPU's pace\].
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
    /// Something became dirty since `State::dirty_extents` was built.
    extents_stale: std::sync::atomic::AtomicBool,
    /// How many signals wait in `State::pending`: none, and a poll need
    /// not lock the state.
    pending_count: std::sync::atomic::AtomicUsize,
    /// Increments a host1x syncpoint, for the releases the renderer
    /// makes itself.
    increment: Mutex<Option<Arc<dyn Fn(u32) + Send + Sync>>>,
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
    /// `SET_ZPASS_PIXEL_COUNT` is enabled: the samples that pass the
    /// stencil and depth tests are counted ([`Renderer::report_passed`]).
    pub count_passed: bool,
}

impl Renderer {
    /// `CLEAR_REPORT_VALUE` of ZPASS_PIXEL_CNT: the draws from here on
    /// count from zero.
    pub fn reset_passed(&self, memory: &dyn GpuMemory, writes: &Writes) -> Result<(), Unmodeled> {
        self.next_slot(memory, writes)?;
        let mut state = self.state.lock().unwrap();
        let feedback = state.feedback()?;
        feedback.span = feedback.slot;
        feedback.base = 0;
        Ok(())
    }

    /// A report of ZPASS_PIXEL_CNT at `address`: `counted` (what the
    /// software renderer counted since the reset) and what the draws
    /// recorded since the reset count on the GPU, as one word or as a
    /// 64-bit value followed by `time`. The GPU has yet to draw them, so
    /// the report is written when a write-back next waits for it: at
    /// the channel's next signal, or before an engine of the channel
    /// accesses the report's memory ([`Coherent`]).
    pub fn report_passed(
        &self,
        memory: &dyn GpuMemory,
        writes: &Writes,
        address: u64,
        one_word: bool,
        time: u64,
        counted: u64,
    ) -> Result<(), Unmodeled> {
        {
            let mut state = self.state.lock().unwrap();
            let feedback = state.feedback()?;
            feedback.pending.push(Report {
                address,
                one_word,
                time,
                base: feedback.base + counted,
                slots: feedback.span..feedback.slot + 1,
            });
        }
        self.any_dirty
            .store(true, std::sync::atomic::Ordering::Relaxed);
        self.next_slot(memory, writes)
    }

    /// Gives the draws from here on the next slot of the feedback
    /// buffer; a buffer whose slots are all used is read first
    /// ([`resolve_feedback`]), which starts it over.
    fn next_slot(&self, memory: &dyn GpuMemory, writes: &Writes) -> Result<(), Unmodeled> {
        {
            let mut state = self.state.lock().unwrap();
            let feedback = state.feedback()?;
            if feedback.slot + 1 < FEEDBACK_SLOTS {
                feedback.slot += 1;
                return Ok(());
            }
        }
        self.write_back(memory, writes, Scope::All)
    }
}

impl Renderer {
    /// A renderer, or `None` without a Metal device.
    pub fn new() -> Option<Renderer> {
        GPU.as_ref()?;
        Some(Renderer {
            state: Mutex::new(State::default()),
            writing_back: Default::default(),
            any_dirty: Default::default(),
            extents_stale: Default::default(),
            pending_count: Default::default(),
            increment: Mutex::new(None),
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
        self.pending_count
            .store(0, std::sync::atomic::Ordering::Relaxed);
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
        state.last_fallback = Some(reason);
        Ok(false)
    }

    /// Adds the time the software renderer took for the draw last left
    /// to it; a reason is reported on stderr each time its total doubles
    /// (from one second on).
    pub fn software_time(&self, elapsed: std::time::Duration) {
        let mut state = self.state.lock().unwrap();
        let Some(reason) = state.last_fallback.take() else {
            return;
        };
        let total = state.software_time.entry(reason.clone()).or_default();
        let before = total.as_secs();
        *total += elapsed;
        let after = total.as_secs();
        if after >= 1 && (before == 0 || after.ilog2() > before.ilog2()) {
            eprintln!("Metal renderer: {after} s on the software renderer: {reason}");
        }
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

    /// Connects host1x syncpoint increments, for the releases the
    /// renderer makes itself ([`Renderer::release`]).
    pub fn set_increment(&self, increment: Arc<dyn Fn(u32) + Send + Sync>) {
        *self.increment.lock().unwrap() = Some(increment);
    }

    /// The channel signals `release`: it is to be made once every dirty
    /// target the guest may read from then on (all but those held on
    /// demand, [`Resident::on_demand`]), every global memory region
    /// programs stored to and every report is in guest memory. Those
    /// write-backs are recorded after the work so far (targets packed on
    /// the GPU, regions copied) and finished, with the release, once the
    /// GPU has done it ([`Renderer::complete`]); the channel goes on
    /// meanwhile, as hardware does, its later work in later command
    /// buffers. Returns the release to be made now when there is nothing
    /// to wait for, or what is to be written cannot be recorded so (then
    /// it is written back now, after waiting for the GPU).
    ///
    /// Until it is finished, a write-back holds back any access of the
    /// channel's engines to its memory ([`Coherent`]), and any later
    /// write-back of the same memory: memory is written in the order of
    /// the work.
    pub fn release(
        &self,
        memory: &dyn GpuMemory,
        writes: &Writes,
        release: Release,
    ) -> Result<Option<Release>, Unmodeled> {
        use std::sync::atomic::Ordering;
        let increment = self.increment.lock().unwrap().is_some();
        {
            let mut state = self.state.lock().unwrap();
            let wanted: Vec<usize> = (0..state.residents.len())
                .filter(|&i| state.residents[i].dirty.is_some() && !state.residents[i].on_demand())
                .collect();
            let keys: Vec<(u64, u32)> = state
                .globals
                .iter()
                .flat_map(|globals| &globals.regions)
                .filter(|(_, region)| region.dirty)
                .map(|(key, _)| *key)
                .collect();
            let reports = state
                .feedback
                .as_ref()
                .map_or(0, |feedback| feedback.pending.len());
            let nothing = wanted.is_empty() && keys.is_empty() && reports == 0;
            if (nothing && state.pending.is_empty())
                || (matches!(release, Release::Increment(_)) && !increment)
                || state.arena.chunks_in_use() > ARENA_UNWAITED
                || state
                    .feedback
                    .as_ref()
                    .is_some_and(|feedback| feedback.slot + 1 >= FEEDBACK_SLOTS || feedback.trapped)
            {
                drop(state);
                self.write_back(memory, writes, Scope::Signal)?;
                return Ok(Some(release));
            }
            let mut extents: Vec<(u64, u64)> = wanted
                .iter()
                .map(|&i| state.residents[i].extent())
                .chain(keys.iter().map(|&key| global_extent(key)))
                .collect();
            if let Some(feedback) = &state.feedback {
                extents.extend(feedback.pending.iter().map(Report::extent));
            }
            let mut packs = Vec::with_capacity(wanted.len());
            for &index in &wanted {
                match state.pack(index)? {
                    Some(packed) => packs.push((index, packed)),
                    None => {
                        // The CPU reads the texture: once the GPU is done.
                        drop(state);
                        self.write_back(memory, writes, Scope::Signal)?;
                        return Ok(Some(release));
                    }
                }
            }
            let mut residents = Vec::with_capacity(packs.len());
            for (index, packed) in packs {
                let resident = &mut state.residents[index];
                residents.push(PendingResident {
                    surface: resident.surface,
                    log2: resident.log2,
                    packed,
                });
                resident.dirty = None;
                resident.pending += 1;
            }
            let mut regions = Vec::with_capacity(keys.len());
            if !keys.is_empty() {
                let command = state.command()?;
                let blit = command
                    .blitCommandEncoder()
                    .ok_or_else(|| Unmodeled("Metal: no blit encoder".into()))?;
                let State { globals, arena, .. } = &mut *state;
                let globals = globals.as_mut().expect("regions are in it");
                for key in keys {
                    let region = globals.regions.get_mut(&key).expect("listed");
                    let size = key.1 as usize;
                    let (staging, offset) = arena.alloc(&vec![0; size])?;
                    // SAFETY: both ranges are inside their buffers.
                    unsafe {
                        blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                            &globals.buffer,
                            region.offset,
                            &staging,
                            offset,
                            size,
                        );
                    }
                    region.dirty = false;
                    region.pending += 1;
                    regions.push((key, staging, offset));
                }
                blit.endEncoding();
            }
            let reports = state
                .feedback
                .as_mut()
                .map(|feedback| std::mem::take(&mut feedback.pending))
                .unwrap_or_default();
            let faults = std::mem::take(&mut state.faults);
            let bound = state
                .globals
                .as_mut()
                .is_some_and(|globals| std::mem::take(&mut globals.bound));
            let command = if nothing && faults.is_empty() && !bound {
                // Nothing of its own: it follows the earlier ones.
                None
            } else {
                state.submit();
                state.submitted.clone()
            };
            state.pending.push_back(Pending {
                command,
                residents,
                regions,
                reports,
                faults,
                bound,
                extents,
                release: Some(release),
            });
            self.pending_count.store(state.pending.len(), Ordering::Relaxed);
            self.any_dirty.store(true, Ordering::Relaxed);
            self.extents_stale.store(true, Ordering::Relaxed);
            return Ok(None);
        }
    }

    /// Writes residents `indices` (those dirty) back as a signal does
    /// ([`Renderer::release`]), with no release: the write-back is
    /// finished once the GPU has done the work so far, and the residents
    /// may be dropped meanwhile. One held on demand is written only where
    /// memory still holds its mirror, as `write_back` does. `false` if
    /// a resident cannot be packed on the GPU (nothing was changed then).
    fn write_back_later(
        &self,
        memory: &dyn GpuMemory,
        writes: &Writes,
        indices: &[usize],
    ) -> Result<bool, Unmodeled> {
        use std::sync::atomic::Ordering;
        // On-demand targets: has the guest reused the memory? (Compared
        // without the state held: a read may write back.)
        let mut wanted = Vec::with_capacity(indices.len());
        for &index in indices {
            let check = {
                let state = self.state.lock().unwrap();
                let resident = &state.residents[index];
                if resident.dirty.is_none() {
                    continue;
                }
                resident
                    .on_demand()
                    .then_some((resident.surface, resident.stamp, resident.extent()))
            };
            // Unchanged in this epoch, or unwritten since its watch mark.
            let check = check.filter(|&(_, stamp, (start, end))| {
                self.writing_back.store(true, Ordering::Relaxed);
                let kept = stamp.and_then(|stamp| writes.kept(memory, stamp, start, end));
                self.writing_back.store(false, Ordering::Relaxed);
                kept.is_none()
            });
            if let Some((surface, ..)) = check {
                let mirror = std::mem::replace(
                    &mut self.state.lock().unwrap().residents[index].mirror,
                    Window::empty(surface),
                );
                // Memory as it is, not as this resident's own write-back
                // would make it ([`Coherent`] would write it back first).
                self.writing_back.store(true, Ordering::Relaxed);
                let same = matches!(mirror.matches(memory), Ok(true));
                self.writing_back.store(false, Ordering::Relaxed);
                let mut state = self.state.lock().unwrap();
                state.residents[index].mirror = mirror;
                if !same {
                    // Forgotten: the memory has a new owner.
                    let resident = &mut state.residents[index];
                    resident.dirty = None;
                    resident.stamp = None;
                    continue;
                }
            }
            wanted.push(index);
        }
        if wanted.is_empty() {
            return Ok(true);
        }
        let mut state = self.state.lock().unwrap();
        let mut packs = Vec::with_capacity(wanted.len());
        for &index in &wanted {
            match state.pack(index)? {
                Some(packed) => packs.push((index, packed)),
                None => return Ok(false),
            }
        }
        let mut residents = Vec::with_capacity(packs.len());
        let mut extents = Vec::with_capacity(packs.len());
        for (index, packed) in packs {
            let resident = &mut state.residents[index];
            extents.push(resident.extent());
            residents.push(PendingResident {
                surface: resident.surface,
                log2: resident.log2,
                packed,
            });
            resident.dirty = None;
            resident.pending += 1;
        }
        state.submit();
        let command = state.submitted.clone();
        state.pending.push_back(Pending {
            command,
            residents,
            regions: Vec::new(),
            reports: Vec::new(),
            faults: Vec::new(),
            bound: false,
            extents,
            release: None,
        });
        self.pending_count.store(state.pending.len(), Ordering::Relaxed);
        self.any_dirty.store(true, Ordering::Relaxed);
        Ok(true)
    }

    /// Finishes the write-backs of signals the GPU has done the work
    /// before (all of them, waiting for the GPU, with `wait`), in order,
    /// and makes their releases ([`Renderer::release`]).
    pub fn complete(
        &self,
        memory: &dyn GpuMemory,
        writes: &Writes,
        wait: bool,
    ) -> Result<(), Unmodeled> {
        self.complete_over(memory, writes, wait, None)
    }

    /// [`Renderer::complete`]; with `range`, waiting for those up to the
    /// last one that writes memory in it (none if none does).
    fn complete_over(
        &self,
        memory: &dyn GpuMemory,
        writes: &Writes,
        wait: bool,
        range: Option<(u64, u64)>,
    ) -> Result<(), Unmodeled> {
        use objc2_metal::MTLCommandBufferStatus;
        use std::sync::atomic::Ordering;
        if self.pending_count.load(Ordering::Relaxed) == 0 {
            return Ok(());
        }
        let mut remaining = match range {
            Some(range) => {
                let state = self.state.lock().unwrap();
                state
                    .pending
                    .iter()
                    .rposition(|pending| {
                        pending.extents.iter().any(|&extent| overlap(extent, range))
                    })
                    .map_or(0, |last| last + 1)
            }
            None => usize::MAX,
        };
        loop {
            if remaining == 0 {
                return Ok(());
            }
            remaining -= 1;
            let pending = {
                let mut state = self.state.lock().unwrap();
                let Some(front) = state.pending.front() else {
                    return Ok(());
                };
                if let Some(command) = &front.command {
                    let status = command.status();
                    if !wait
                        && status != MTLCommandBufferStatus::Completed
                        && status != MTLCommandBufferStatus::Error
                    {
                        return Ok(());
                    }
                    command.waitUntilCompleted();
                }
                let pending = state.pending.pop_front().expect("looked at");
                self.pending_count.store(state.pending.len(), Ordering::Relaxed);
                self.writing_back.store(true, Ordering::Relaxed);
                let mut pending = pending;
                let result = finish(&mut state, &mut pending, memory, writes);
                self.writing_back.store(false, Ordering::Relaxed);
                if state.pending.is_empty() {
                    self.any_dirty.store(state.anything_dirty(), Ordering::Relaxed);
                }
                result?;
                pending
            };
            match pending.release {
                Some(Release::Write { address, bytes }) => {
                    memory.write(address, &bytes).ok_or_else(|| {
                        Unmodeled(format!(
                            "semaphore write to unmapped GPU address {address:#x}"
                        ))
                    })?;
                }
                Some(Release::Increment(index)) => {
                    let increment = self.increment.lock().unwrap().clone();
                    increment.expect("checked when held")(index);
                }
                None => {}
            }
        }
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
        // Earlier signals' write-backs come first.
        let range = match scope {
            Scope::Range(start, end) => Some((start, end)),
            _ => None,
        };
        self.complete_over(memory, writes, true, range)?;
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
        let wanted_globals: Vec<(u64, u32)> = state
            .globals
            .iter()
            .flat_map(|globals| &globals.regions)
            .filter(|(key, region)| {
                region.dirty
                    && match scope {
                        Scope::Range(start, end) => overlap(global_extent(**key), (start, end)),
                        _ => true,
                    }
            })
            .map(|(key, _)| *key)
            .collect();
        // Reports that wait for their counts, and a feedback buffer with
        // no slot left, are read once the GPU is done.
        let reports = state.feedback.as_ref().is_some_and(|feedback| {
            feedback.slot + 1 >= FEEDBACK_SLOTS
                || match scope {
                    Scope::Range(start, end) => feedback
                        .pending
                        .iter()
                        .any(|report| overlap(report.extent(), (start, end))),
                    _ => !feedback.pending.is_empty(),
                }
        });
        if !wanted.contains(&true) && wanted_globals.is_empty() && !reports {
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
        // Reads and writes of this function pass `Coherent` unchecked.
        self.writing_back.store(true, Ordering::Relaxed);
        // Residents whose mirror is known to equal memory still (nothing
        // of the channel wrote there since they were compared, in this
        // epoch, or nothing at all since their watch mark): they do after
        // their own write-back too (residents do not overlap).
        let mut current: Vec<bool> = state
            .residents
            .iter()
            .map(|r| {
                let (start, end) = r.extent();
                r.stamp
                    .is_some_and(|s| writes.kept(memory, s, start, end).is_some())
            })
            .collect();
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
        // An executed instruction neither renderer models is the
        // interpreter's refusal; the reports get their counts.
        if let Err(refusal) = resolve_feedback(&mut state, memory) {
            self.writing_back.store(false, Ordering::Relaxed);
            return Err(refusal);
        }
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
        if let Some(globals) = &mut state.globals {
            if std::mem::take(&mut globals.bound) {
                let word = globals.buffer.contents().cast::<u32>();
                // SAFETY: the buffer starts with the fault word; the GPU
                // finished.
                if unsafe { word.read() } != 0 {
                    // SAFETY: as above.
                    unsafe { word.write(0) };
                    self.writing_back.store(false, Ordering::Relaxed);
                    return Err(Unmodeled(
                        "shader: global memory access outside the buffers the program's \
                         constants name"
                            .into(),
                    ));
                }
            }
            // Regions programs may have stored to: what changed goes to
            // guest memory, and the buffer's bytes are the copy of it
            // from now on.
            for key in &wanted_globals {
                let (address, size) = *key;
                let Some(region) = globals.regions.get_mut(key) else {
                    continue;
                };
                // SAFETY: the region's bytes are in the buffer; the GPU
                // finished.
                let now = unsafe {
                    std::slice::from_raw_parts(
                        globals.buffer.contents().as_ptr().cast::<u8>().add(region.offset),
                        size as usize,
                    )
                };
                if now != region.content {
                    if memory.write(address, now).is_none() {
                        self.writing_back.store(false, Ordering::Relaxed);
                        return Err(Unmodeled(format!(
                            "Metal: global memory write-back to unmapped {address:#x}"
                        )));
                    }
                    region.content.copy_from_slice(now);
                }
                region.dirty = false;
                region.stamp = writes.stamp();
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
        self.any_dirty.store(state.anything_dirty(), Ordering::Relaxed);
        let stamp = writes.stamp();
        for ((resident, current), wanted) in state.residents.iter_mut().zip(current).zip(wanted) {
            if wanted {
                resident.stamp = current.then_some(stamp);
            }
        }
        state.trim();
        result
    }

    /// Something may be dirty now; `newly` if it was clean before (a
    /// resident rendered to, a global memory region stored to).
    fn dirtied(&self, newly: bool) {
        use std::sync::atomic::Ordering;
        self.any_dirty.store(true, Ordering::Relaxed);
        if newly {
            self.extents_stale.store(true, Ordering::Relaxed);
        }
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
        // Memory a signal's write-back has yet to write.
        self.complete_over(memory, writes, true, Some(range))?;
        let overlaps = {
            let mut state = self.state.lock().unwrap();
            if self.extents_stale.swap(false, Ordering::Relaxed) {
                let mut extents: Vec<(u64, u64)> = state
                    .residents
                    .iter()
                    .filter(|r| r.dirty.is_some())
                    .map(|r| r.extent())
                    .chain(state.globals.iter().flat_map(|globals| {
                        globals
                            .regions
                            .iter()
                            .filter(|(_, region)| region.dirty)
                            .map(|(key, _)| global_extent(*key))
                    }))
                    .collect();
                extents.sort_unstable();
                let mut merged: Vec<(u64, u64)> = Vec::with_capacity(extents.len());
                for (start, end) in extents {
                    match merged.last_mut() {
                        Some(last) if start <= last.1 => last.1 = last.1.max(end),
                        _ => merged.push((start, end)),
                    }
                }
                state.dirty_extents = merged;
            }
            let first = state.dirty_extents.partition_point(|&(_, end)| end <= range.0);
            state
                .dirty_extents
                .get(first)
                .is_some_and(|&(start, _)| start < range.1)
                || state.feedback.iter().any(|feedback| {
                    feedback
                        .pending
                        .iter()
                        .any(|report| overlap(report.extent(), range))
                })
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
        // Tests mandated to come first (`raster::DepthOutput::early`) the
        // fragment function cannot make as the software renderer does:
        // the stencil test, the samples of a pixel of the plane path, or
        // a fragment dropped after its depth was written into blended
        // targets (the function holds such a fragment, writing the
        // colors the targets hold, `Translated::source`, which blending
        // would change) \[assumption: reading a target in the fragment
        // function and writing what was read stores the same bits, for
        // the formats targets have here\].
        if let Some((_, depth)) = draw.depth
            && depth.early
        {
            if depth.stencil.is_some() {
                return self.fall_back("mandated early tests with the stencil test".into());
            }
            if !draw.multisample.samples.is_single() && vertex.is_none() {
                return self.fall_back("mandated early tests of several samples".into());
            }
            let drops = draw.alpha_test.is_some()
                || draw.multisample.alpha_to_coverage
                || draw.program.mnemonics().any(|mnemonic| mnemonic == "KIL");
            if drops && depth.write && draw.colors.iter().any(|color| color.blend.is_some()) {
                return self.fall_back(
                    "mandated early tests of fragments dropped afterwards into blended targets"
                        .into(),
                );
            }
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
        let target_surfaces: Vec<Surface> = targets.iter().map(|&(surface, _)| surface).collect();
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
            let kind = translated.kinds[slot];
            let (sampled, tic, tsc) =
                match self.sampled(memory, writes, draw.textures, handle, kind, &target_surfaces)? {
                    Ok(sampled) => sampled,
                    Err(refusal) => {
                        unbound.push((slot, refusal));
                        bound.push(None);
                        continue;
                    }
                };
            if let Err(reason) = format::check_texture(&tic, &tsc, kind) {
                return self.fall_back(reason);
            }
            if matches!(sampled, Sampled::Composite(_)) && kind != composite_kind(&tic) {
                return self.fall_back(format!("rendered texture in a {kind:?} slot"));
            }
            if translated.compare[slot] {
                // The translated comparison has one filter for both
                // magnification and minification (`msl` `compare_sample`).
                if !tsc.compare {
                    return self.fall_back("depth comparison without a compare sampler".into());
                }
                if tsc.mag_linear != (tsc.min_filter != 1) {
                    return self.fall_back("depth comparison with two filters".into());
                }
            }
            let bias = if translated.implicit_lod[slot] {
                tsc.lod_bias
            } else {
                0.0
            };
            bound.push(Some((sampled, tic, tsc, bias, kind)));
        }
        // A vertex program's textures. It has no fault words: a slot
        // naming no texture leaves the vertices to the CPU, which refuses
        // the sample when it executes.
        let mut vertex_bound = Vec::new();
        if let Some((native, vertex, ..)) = &vertex {
            for (slot, &offset) in vertex.textures.iter().enumerate() {
                let kind = vertex.kinds[slot];
                let handle = native
                    .constants
                    .get(draw.texture_buffer_index as usize)
                    .and_then(|(data, _)| data.get(offset as usize..offset as usize + 4))
                    .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("4 bytes")));
                let sampled = match handle {
                    Some(handle) => {
                        self.sampled(memory, writes, draw.textures, handle, kind, &target_surfaces)?
                    }
                    None => Err("texture handle outside the constant buffer".into()),
                };
                let reason = match sampled {
                    Ok((sampled, tic, tsc)) => match format::check_texture(&tic, &tsc, kind) {
                        Ok(()) if matches!(sampled, Sampled::Composite(_))
                            && kind != TextureKind::Flat =>
                        {
                            format!("rendered texture in a {kind:?} slot")
                        }
                        Ok(()) => {
                            vertex_bound.push((sampled, tic, tsc, kind));
                            continue;
                        }
                        Err(reason) => reason,
                    },
                    Err(_) => "a slot names no texture".into(),
                };
                self.note_cpu_vertices(&format!("vertex texture: {reason}"));
                return Ok(false);
            }
        }
        // The answers to the programs' texture queries. A vertex program
        // that asks about a handle naming no texture runs on the CPU,
        // which refuses the query when it executes.
        let queries = query_words(
            draw.textures,
            draw.constants,
            draw.texture_buffer_index,
            &translated.queries,
        );
        let vertex_queries = match &vertex {
            Some((native, vertex, ..)) => query_words(
                draw.textures,
                native.constants,
                draw.texture_buffer_index,
                &vertex.queries,
            ),
            None => Vec::new(),
        };
        if vertex_queries.chunks(5).any(|answer| answer[4] == 0) {
            self.note_cpu_vertices("vertex texture query: a slot names no texture");
            return Ok(false);
        }
        if translated.atomics && !samples.is_single() {
            return self.fall_back("atomic update of global memory with several samples".into());
        }
        let [global_table, vertex_table] = self.global_tables(
            memory,
            writes,
            [
                (draw.constants, &translated.globals, translated.stores),
                match &vertex {
                    Some((native, vertex, ..)) => {
                        (native.constants, &vertex.globals, vertex.stores)
                    }
                    None => (&[], &[], false),
                },
            ],
        )?;
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
                early: depth.early,
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
            count_passed: draw.count_passed,
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
                    key.count_passed,
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
            gpu_textures.push(match entry {
                Some((sampled, tic, tsc, _, kind)) => {
                    state.gpu_texture(gpu, sampled, tic, tsc, *kind)?
                }
                None => state.placeholder(gpu, translated.kinds[slot])?,
            });
        }
        let mut vertex_textures = Vec::with_capacity(vertex_bound.len());
        for (sampled, tic, tsc, kind) in &vertex_bound {
            vertex_textures.push(state.gpu_texture(gpu, sampled, tic, tsc, *kind)?);
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
        // The LOD bias of each slot and a spare one; then, for depth
        // comparisons, each slot's function, filter (bit 3 linear) and
        // mip filter (bits 5:4) as the bits of a float, each slot's
        // least level of detail, and each slot's greatest.
        let of = |value: &dyn Fn(&Tic, &Tsc) -> f32| -> Vec<f32> {
            bound
                .iter()
                .map(|b| b.as_ref().map_or(0.0, |(_, tic, tsc, ..)| value(tic, tsc)))
                .collect()
        };
        let mut biases: Vec<f32> = bound
            .iter()
            .map(|b| b.as_ref().map_or(0.0, |b| b.3))
            .chain([0.0])
            .chain(of(&|_, tsc| {
                f32::from_bits(
                    tsc.compare_func | u32::from(tsc.mag_linear) << 3 | (tsc.mip_filter & 3) << 4,
                )
            }))
            .chain(of(&|tic, tsc| tic.lod_range(tsc).0))
            .chain(of(&|tic, tsc| tic.lod_range(tsc).1))
            .collect();
        // A program of more slots than a function has samplers is given
        // the distinct samplers of its slots, and after the words above
        // each slot's index among them (`Translated::shared_samplers`).
        let mut shared: Vec<Retained<Sampler>> = Vec::new();
        if translated.shared_samplers() {
            for (_, sampler) in &gpu_textures {
                let known = shared
                    .iter()
                    .position(|s| Retained::as_ptr(s) == Retained::as_ptr(sampler));
                let index = known.unwrap_or_else(|| {
                    shared.push(sampler.clone());
                    shared.len() - 1
                });
                biases.push(f32::from_bits(index as u32));
            }
            if shared.len() > msl::SAMPLERS {
                drop(state);
                return self.fall_back(format!("{} distinct samplers", shared.len()));
            }
        }
        // Then the answers to its texture queries.
        biases.extend(queries.iter().map(|&word| f32::from_bits(word)));
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
                        words[2 + 4 * slot..][..4].copy_from_slice(&[
                            stream.first,
                            stream.stride,
                            stream.divisor,
                            stream.offset,
                        ]);
                    }
                    words[msl::INDEXED_WORD] =
                        u32::from(matches!(native.vertices, NativeVertices::Indexed { .. }));
                    words.extend_from_slice(&vertex_table);
                    words.extend_from_slice(&vertex_queries);
                    bind(
                        &mut state.arena,
                        &encoder,
                        Stage::Vertex,
                        bytes_of_u32(&words),
                        29,
                    )?;
                    if !native.streams.is_empty() {
                        bind(
                            &mut state.arena,
                            &encoder,
                            Stage::Vertex,
                            &native.stream_bytes,
                            msl::STREAM_BUFFER,
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
            if translated.traps || draw.count_passed {
                let feedback = state.feedback()?;
                feedback.trapped |= translated.traps;
                let (words, offset) = (feedback.buffer.clone(), 8 * feedback.slot);
                encoder.setFragmentBuffer_offset_atIndex(Some(&words), offset, msl::TRAP_BUFFER);
            }
            for (i, (texture, sampler)) in vertex_textures.iter().enumerate() {
                encoder.setVertexTexture_atIndex(Some(texture), i);
                encoder.setVertexSamplerState_atIndex(Some(sampler), i);
            }
            for (i, (texture, sampler)) in gpu_textures.iter().enumerate() {
                encoder.setFragmentTexture_atIndex(Some(texture), i);
                if shared.is_empty() {
                    encoder.setFragmentSamplerState_atIndex(Some(sampler), i);
                }
            }
            // Every sampler of the array is one Metal knows.
            if let Some(last) = shared.last() {
                for i in 0..msl::SAMPLERS {
                    encoder.setFragmentSamplerState_atIndex(Some(shared.get(i).unwrap_or(last)), i);
                }
            }
            if !global_table.is_empty() || !vertex_table.is_empty() {
                let globals = state.globals.as_mut().expect("made for the tables");
                globals.bound = true;
                let buffer = globals.buffer.clone();
                if !global_table.is_empty() {
                    encoder.setFragmentBuffer_offset_atIndex(
                        Some(&buffer),
                        0,
                        msl::GLOBAL_BUFFER,
                    );
                    bind(
                        &mut state.arena,
                        &encoder,
                        Stage::Fragment,
                        bytes_of_u32(&global_table),
                        msl::GLOBAL_BUFFER + 1,
                    )?;
                }
                if !vertex_table.is_empty() {
                    encoder.setVertexBuffer_offset_atIndex(
                        Some(&buffer),
                        0,
                        msl::VERTEX_GLOBAL_BUFFER,
                    );
                }
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
                            .drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount(
                                MTLPrimitiveType::Triangle,
                                *count,
                                *index_type,
                                indices,
                                *offset,
                                native.instances as usize,
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
            let newly = state.rendered(index, scissor, writes);
            self.dirtied(newly);
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
        let newly = state.rendered(residents[0], rect, writes);
        self.dirtied(newly);
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
        let newly = state.rendered(residents[0], rect, writes);
        self.dirtied(newly);
        Ok(true)
    }

    /// The global memory tables of a draw's programs (`shader::msl`
    /// `memory.rs`): for each request (a stage's constants, the places
    /// of them its program takes descriptors from, and whether the
    /// program stores) four words per place, saying where in the global
    /// memory buffer the memory the place names is; zeros for a place
    /// that is no descriptor of mapped memory of at most
    /// [`GLOBAL_LIMIT`] bytes.
    fn global_tables<const N: usize>(
        &self,
        memory: &dyn GpuMemory,
        writes: &Writes,
        requests: [(&[(Vec<u8>, usize)], &[(u32, u32)], bool); N],
    ) -> Result<[Vec<u32>; N], Unmodeled> {
        if requests.iter().all(|(_, places, _)| places.is_empty()) {
            return Ok(std::array::from_fn(|_| Vec::new()));
        }
        // A buffer that fills up is emptied, which forgets the places
        // found before: they are found once more then.
        for _ in 0..2 {
            let generation = self.state.lock().unwrap().globals()?.generation;
            let mut tables: [Vec<u32>; N] = std::array::from_fn(|_| Vec::new());
            for (table, &(constants, places, store)) in tables.iter_mut().zip(&requests) {
                for &(slot, offset) in places {
                    let key = constants
                        .get(slot as usize)
                        .and_then(|(data, _)| data.get(offset as usize..offset as usize + 12))
                        .map(|descriptor| {
                            (
                                u64::from_le_bytes(descriptor[..8].try_into().expect("8 bytes")),
                                u32::from_le_bytes(descriptor[8..].try_into().expect("4 bytes")),
                            )
                        })
                        .filter(|&(address, size)| {
                            address >> 40 == 0
                                && address.is_multiple_of(4)
                                && size != 0
                                && size as usize <= GLOBAL_LIMIT
                        });
                    let at = match key {
                        Some(key) => self.global_region(memory, writes, key, store)?,
                        None => None,
                    };
                    table.extend_from_slice(&match (key, at) {
                        (Some((address, size)), Some(at)) => {
                            [address as u32, (address >> 32) as u32, size, (at / 4) as u32]
                        }
                        _ => [0; 4],
                    });
                }
            }
            if self.state.lock().unwrap().globals()?.generation == generation {
                return Ok(tables);
            }
        }
        Err(Unmodeled(
            "Metal: the global memory of a draw does not fit the buffer".into(),
        ))
    }

    /// Where in the global memory buffer the bytes of guest memory region
    /// `key` (address, size) are, in bytes; `None` for memory that is not
    /// mapped. A region is copied in once and used until memory there
    /// changes: taken as it is while no write of the channel touched it
    /// in this epoch, compared with memory otherwise, and copied in anew
    /// when memory differs (work recorded earlier goes on reading the
    /// copy it was given). With `store` the program may store to it:
    /// the copy is then what memory is to hold, until it is written back
    /// ([`Renderer::write_back`]). A region overlapping another that
    /// programs may have stored to waits for that one's write-back.
    fn global_region(
        &self,
        memory: &dyn GpuMemory,
        writes: &Writes,
        key: (u64, u32),
        store: bool,
    ) -> Result<Option<usize>, Unmodeled> {
        let extent = global_extent(key);
        loop {
            {
                let mut state = self.state.lock().unwrap();
                let globals = state.globals()?;
                // Bytes inside a region programs may have stored to are
                // read (and stored) in that region's copy, which holds what
                // memory is to hold.
                let inside = globals
                    .regions
                    .iter()
                    .find(|(other, region)| {
                        region.dirty && **other != key && {
                            let outer = global_extent(**other);
                            outer.0 <= extent.0 && extent.1 <= outer.1
                        }
                    })
                    .map(|(other, region)| region.offset + (key.0 - other.0) as usize);
                if let Some(offset) = inside {
                    return Ok(Some(offset));
                }
                let conflict = globals.regions.iter().any(|(other, region)| {
                    region.dirty && *other != key && overlap(global_extent(*other), extent)
                });
                if !conflict {
                    if let Some(region) = globals.regions.get_mut(&key)
                        && (region.dirty
                            || region.pending != 0
                            || writes.unchanged(region.stamp, extent.0, extent.1))
                    {
                        if store {
                            self.dirtied(!region.dirty);
                            region.dirty = true;
                        }
                        return Ok(Some(region.offset));
                    }
                    break;
                }
            }
            self.write_back(memory, writes, Scope::Range(extent.0, extent.1))?;
        }
        // Read outside the lock: a read may write residents back.
        let mut bytes = vec![0u8; key.1 as usize];
        if memory.probe(key.0, &mut bytes).is_none() {
            return Ok(None);
        }
        let stamp = writes.stamp();
        let mut state = self.state.lock().unwrap();
        let globals = state.globals()?;
        if let Some(region) = globals.regions.get_mut(&key)
            && region.content == bytes
        {
            region.stamp = stamp;
            if store {
                self.dirtied(!region.dirty);
                region.dirty = true;
            }
            return Ok(Some(region.offset));
        }
        let length = bytes.len().next_multiple_of(16);
        if globals.next + length > GLOBAL_ARENA {
            // Full: once guest memory holds what programs stored and the
            // GPU is done with every copy, the buffer starts over.
            drop(state);
            self.settle(memory, writes)?;
            state = self.state.lock().unwrap();
            state.wait();
            let globals = state.globals()?;
            globals.regions.clear();
            globals.next = GLOBAL_FIRST;
            globals.generation += 1;
        }
        let globals = state.globals()?;
        let offset = globals.next;
        globals.next += length;
        // SAFETY: the bytes fit the buffer at `offset` (checked above), and
        // no work recorded or running was given that part of it.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                globals.buffer.contents().as_ptr().cast::<u8>().add(offset),
                bytes.len(),
            );
        }
        globals.regions.insert(
            key,
            GlobalRegion {
                offset,
                content: bytes,
                stamp,
                dirty: store,
                pending: 0,
            },
        );
        if store {
            self.dirtied(true);
        }
        Ok(Some(offset))
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
        // (and with no pass open that names one). One inside the memory of
        // a target the caller replaces whole is dropped unwritten: every
        // byte it would write is overwritten.
        for &(surface, format) in targets {
            let target = extent(&surface);
            // The one resident there being the same surface in another
            // color format of the same texel size: its texels are the
            // same bytes, copied on the GPU as unsigned words into a
            // texture of the new format.
            if let Content::Color(_) = format
                && !overwrite
            {
                let mut state = self.state.lock().unwrap();
                let overlapping: Vec<usize> = (0..state.residents.len())
                    .filter(|&i| {
                        let r = &state.residents[i];
                        other(r, surface, format) && overlap(r.extent(), target)
                    })
                    .collect();
                if let [index] = overlapping[..]
                    && let r = &state.residents[index]
                    && r.surface == surface
                    && r.log2 == log2
                    && matches!(r.format, Content::Color(_))
                    && let Some(raw) = raw_view_format(surface.bytes_per_pixel)
                    && let Some(pixel_format) = format.pixel_format()
                {
                    let layout = Samples {
                        log2,
                        ..Samples::SINGLE
                    };
                    let full = layout.pixels(&surface);
                    let texture =
                        new_target(Self::gpu(), pixel_format, full.x1, full.y1, layout.count())?;
                    let from = r.texture.newTextureViewWithPixelFormat(raw);
                    let to = texture.newTextureViewWithPixelFormat(raw);
                    if let (Some(from), Some(to)) = (from, to) {
                        let command = state.command()?;
                        let blit = command
                            .blitCommandEncoder()
                            .ok_or_else(|| Unmodeled("Metal: no blit encoder".into()))?;
                        // SAFETY: two views of one size and format.
                        unsafe { blit.copyFromTexture_toTexture(&from, &to) };
                        blit.endEncoding();
                        state.versions += 1;
                        let version = state.versions;
                        let resident = &mut state.residents[index];
                        resident.format = format;
                        resident.texture = texture;
                        resident.version = version;
                        // Composites made of it read it as before.
                        state.composites.retain(|_, composite| {
                            !composite.levels.iter().any(|level| {
                                level.resident.is_some() && holds_level(&surface, &level.surface)
                            })
                        });
                        continue;
                    }
                }
            }
            let (stale, covered) = {
                let state = self.state.lock().unwrap();
                let mut stale = state
                    .residents
                    .iter()
                    .filter(|r| other(r, surface, format) && overlap(r.extent(), target))
                    .peekable();
                let any = stale.peek().is_some();
                (any, any && stale.all(|r| {
                    let (start, end) = r.extent();
                    overwrite && target.0 <= start && end <= target.1
                }))
            };
            if stale {
                let indices: Vec<usize> = {
                    let state = self.state.lock().unwrap();
                    (0..state.residents.len())
                        .filter(|&i| {
                            let r = &state.residents[i];
                            other(r, surface, format) && overlap(r.extent(), target)
                        })
                        .collect()
                };
                // A color target drawn to takes the residents' texels on
                // the GPU ([`relayout`]).
                if !covered
                    && !overwrite
                    && matches!(format, Content::Color(_))
                    && log2 == [0, 0]
                    && self.take_over(memory, writes, surface, format, &indices)?
                {
                    continue;
                }
                // Under a target replaced whole, what the residents hold
                // beyond it reaches memory once the GPU is done, with no
                // wait now; otherwise the target is loaded from memory
                // holding it.
                let later = !covered && overwrite && self.write_back_later(memory, writes, &indices)?;
                if !covered && !later {
                    self.write_back(memory, writes, Scope::Range(target.0, target.1))?;
                }
                let mut state = self.state.lock().unwrap();
                state.submit();
                state
                    .residents
                    .retain(|r| !(other(r, surface, format) && overlap(r.extent(), target)));
                self.extents_stale.store(true, std::sync::atomic::Ordering::Relaxed);
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
                    .map(|i| (i, state.residents[i].current(writes), state.residents[i].stamp))
            };
            // One the channel's epoch no longer vouches for may still be
            // unwritten since its watch mark: current then, with a stamp
            // of this epoch. (Asked without the state, as reads may flush.)
            let found = found.map(|(index, current, stamp)| {
                if current {
                    return (index, true);
                }
                let (start, end) = extent(&surface);
                match stamp.and_then(|stamp| writes.kept(memory, stamp, start, end)) {
                    Some(now) => {
                        let mut state = self.state.lock().unwrap();
                        let resident = &mut state.residents[index];
                        let same = resident.surface == surface && resident.stamp == stamp;
                        if same {
                            resident.stamp = Some(now);
                        }
                        (index, same)
                    }
                    None => (index, false),
                }
            });
            // The whole surface: its pixels, and the texels holding them.
            let full = layout.pixels(&surface);
            let texels = Rect {
                x0: 0,
                y0: 0,
                x1: surface.width,
                y1: surface.height,
            };
            // A target replaced whole, watched for the guest's reuse of its
            // memory from here on.
            let watched = overwrite
                .then(|| {
                    let (start, end) = extent(&surface);
                    memory.watch(start, (end - start) as usize)
                })
                .flatten();
            match found {
                Some((index, true)) => indices.push(index),
                // A target the caller replaces whole needs no look at
                // memory: what it held does not matter. (One held on demand
                // tells a reuse of the memory later by the watch, or else
                // keeps a mirror equal to memory.)
                Some((index, false))
                    if overwrite
                        && (watched.is_some()
                            || !self.state.lock().unwrap().residents[index].on_demand()) =>
                {
                    self.state.lock().unwrap().residents[index].stamp =
                        Some(writes.stamp().watched(watched));
                    indices.push(index);
                }
                Some((index, false)) => {
                    // Compare with guest memory, without holding the
                    // state (reads may flush); reload if it changed.
                    let mirror = std::mem::replace(
                        &mut self.state.lock().unwrap().residents[index].mirror,
                        Window::empty(surface),
                    );
                    let (start, end) = extent(&surface);
                    let mark = memory.watch(start, (end - start) as usize);
                    let reloaded = match mirror.matches(memory) {
                        Ok(true) => None,
                        Ok(false) => Some(Window::load(memory, surface, texels)),
                        Err(refusal) => Some(Err(refusal)),
                    };
                    let stamp = writes.stamp().watched(mark);
                    let mut state = self.state.lock().unwrap();
                    state.residents[index].mirror = mirror;
                    match reloaded {
                        None => {}
                        Some(Ok(mut window)) => {
                            // Work recorded or running may read the
                            // texture (a depth buffer tested, not written):
                            // it keeps the old one, as it was when recorded,
                            // and the resident takes a new one. An open
                            // pass may name it: it ends.
                            state.command()?;
                            let old = &state.residents[index].texture;
                            let texture = new_target(
                                Self::gpu(),
                                old.pixelFormat(),
                                full.x1,
                                full.y1,
                                layout.count(),
                            )?;
                            upload_surface(&texture, &mut window, full, log2);
                            state.versions += 1;
                            let version = state.versions;
                            let resident = &mut state.residents[index];
                            resident.texture = texture;
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
                    let on_demand = matches!(format, Content::Depth(_)) || log2 != [0, 0];
                    let texture =
                        new_target(Self::gpu(), pixel_format, full.x1, full.y1, layout.count())?;
                    // A target the caller replaces whole is neither read
                    // nor uploaded: its mirror holds zeros, which a
                    // write-back stores where its GOBs (pitch rows) reach
                    // past the surface \[assumption: those bytes are the
                    // surface's padding, which nothing reads\]. (One held
                    // on demand needs the watch, as above, or a mirror
                    // equal to memory.)
                    let (window, stamp) = if overwrite && (!on_demand || watched.is_some()) {
                        let stamp = on_demand.then(|| writes.stamp().watched(watched));
                        (Window::blank(surface, texels), stamp)
                    } else {
                        let (start, end) = extent(&surface);
                        let mark = memory.watch(start, (end - start) as usize);
                        let mut window = Window::load(memory, surface, texels)?;
                        let stamp = writes.stamp().watched(mark);
                        upload_surface(&texture, &mut window, full, log2);
                        (window, Some(stamp))
                    };
                    let mut state = self.state.lock().unwrap();
                    state.residents.push(Resident {
                        surface,
                        format,
                        log2,
                        texture,
                        mirror: window,
                        dirty: None,
                        pending: 0,
                        stamp,
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

    /// What the texture `handle` names is sampled from in a slot of
    /// `kind`, with its headers: the targets that are its levels
    /// ([`Renderer::composite`]), else its bytes in its own format, else
    /// what they decode to. The inner error is why the handle names no
    /// texture.
    #[allow(clippy::type_complexity)]
    fn sampled(
        &self,
        memory: &dyn GpuMemory,
        writes: &Writes,
        textures: &DrawTextures<'_>,
        handle: u32,
        kind: TextureKind,
        targets: &[Surface],
    ) -> Result<Result<(Sampled, Tic, Tsc), String>, Unmodeled> {
        let (tic, tsc) = match textures.header(handle) {
            Ok(header) => header,
            Err(refusal) => return Ok(Err(refusal.0)),
        };
        let composite = self.composite(memory, writes, &tic, targets)?;
        let sampled = match composite {
            Some(texture) => Ok(Sampled::Composite(texture)),
            None => match format::native_format(&tic, kind) {
                Some(pixel_format) => textures
                    .raw(handle)
                    .map(|raw| Sampled::Native(raw, pixel_format)),
                None => textures
                    .resolve(handle)
                    .map(|(texture, _)| Sampled::Decoded(texture)),
            },
        };
        Ok(match sampled {
            Ok(sampled) => Ok((sampled, tic, tsc)),
            Err(refusal) => Err(refusal.0),
        })
    }

    /// The Metal texture to sample for the texture `tic` describes when
    /// levels of it are render targets held on the GPU and current
    /// ([`Composite`]); `None` when none is, and the texture is taken
    /// from memory as any other. A level the draw itself renders to (one
    /// of `targets`) is sampled as the target was before the draw.
    ///
    /// A texture of one level that is a resident of the same size no
    /// target of the draw is samples the resident itself, through a view
    /// of it that reads the header's components (the resident's format,
    /// or its sRGB form, a depth buffer's words as floats or UNORM, and
    /// the header's swizzle): no copy is made. Metal filters the
    /// resident's own format then, where a composite's is 32-bit floats
    /// \[known difference: filtered samples of 8- and 16-bit formats may
    /// differ in the last bits\].
    fn composite(
        &self,
        memory: &dyn GpuMemory,
        writes: &Writes,
        tic: &Tic,
        targets: &[Surface],
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
        // Which levels are residents: the resident's surface and content,
        // how Metal's read of each gives the header's components, and
        // whether they are known to hold what the channel's work so far
        // makes of them.
        let candidates: Vec<Option<(Surface, Content, Reading, bool)>> = {
            let state = self.state.lock().unwrap();
            levels
                .iter()
                .map(|(surface, _)| {
                    let resident = state
                        .residents
                        .iter()
                        .find(|r| holds_level(&r.surface, surface) && r.log2 == [0, 0])?;
                    let reading = shared_components(tic, resident.format)?;
                    Some((
                        resident.surface,
                        resident.format,
                        reading,
                        resident.current(writes),
                    ))
                })
                .collect()
        };
        let Some((_, _, components, _)) = candidates.iter().flatten().next().copied() else {
            return Ok(None);
        };
        if candidates.iter().flatten().any(|c| c.2 != components) {
            return Ok(None);
        }
        // A resident not rendered to since the channel last signaled is
        // compared with memory first, as before a draw to it.
        let unchecked: Vec<(Surface, Content)> = candidates
            .iter()
            .filter_map(|c| match c {
                Some((surface, content, _, false)) => Some((*surface, *content)),
                _ => None,
            })
            .collect();
        if !unchecked.is_empty() {
            self.residents(memory, writes, &unchecked, [0, 0], false)?;
        }
        if let [(level, _)] = &levels[..]
            && let [Some((surface, content, (components, read), _))] = candidates[..]
            && surface == *level
            && !targets.contains(&surface)
        {
            let state = self.state.lock().unwrap();
            if let Some(resident) = state
                .residents
                .iter()
                .find(|r| r.surface == surface && r.format == content && r.log2 == [0, 0])
            {
                let held = resident.texture.pixelFormat();
                let view_format = match read {
                    READ_PLAIN => Some(held),
                    READ_SRGB => Some(format::srgb(held)),
                    READ_FLOAT_BITS => Some(MTLPixelFormat::R32Float),
                    READ_UNORM16 => Some(MTLPixelFormat::R16Unorm),
                    _ => None,
                };
                let channel = |s: u8| match s {
                    2..=5 => match components[usize::from(s - 2)] {
                        0 => MTLTextureSwizzle::Red,
                        1 => MTLTextureSwizzle::Green,
                        2 => MTLTextureSwizzle::Blue,
                        3 => MTLTextureSwizzle::Alpha,
                        4 => MTLTextureSwizzle::Zero,
                        _ => MTLTextureSwizzle::One,
                    },
                    6 | 7 => MTLTextureSwizzle::One,
                    _ => MTLTextureSwizzle::Zero,
                };
                let swizzle = MTLTextureSwizzleChannels {
                    red: channel(tic.swizzle[0]),
                    green: channel(tic.swizzle[1]),
                    blue: channel(tic.swizzle[2]),
                    alpha: channel(tic.swizzle[3]),
                };
                if let Some(view_format) = view_format {
                    // SAFETY: one level and slice of a 2D texture, in a
                    // format of its size class.
                    let view = unsafe {
                        resident
                            .texture
                            .newTextureViewWithPixelFormat_textureType_levels_slices_swizzle(
                                view_format,
                                MTLTextureType::Type2D,
                                NSRange::new(0, 1),
                                NSRange::new(0, 1),
                                swizzle,
                            )
                    };
                    if view.is_some() {
                        return Ok(view);
                    }
                }
            }
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
                // Only the levels memory holds (`Composite::new`'s order:
                // layer after layer, each's levels).
                let count = (levels.len() / composite_layers(tic) as usize).max(1);
                let memory_level = |level: u32| {
                    candidates.iter().enumerate().any(|(index, c)| {
                        c.is_none() && tic.base_level + (index % count) as u32 == level
                    })
                };
                match Texture::decode_levels(&raw, tic, memory_level) {
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
    /// Whether anything is to reach guest memory still: a dirty target or
    /// global memory region, a report, a signal's write-back.
    fn anything_dirty(&self) -> bool {
        self.residents.iter().any(|r| r.dirty.is_some())
            || self
                .globals
                .iter()
                .any(|globals| globals.regions.values().any(|region| region.dirty))
            || self
                .feedback
                .iter()
                .any(|feedback| !feedback.pending.is_empty())
            || !self.pending.is_empty()
    }

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
            self.arena.committed(frame.command.clone());
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
        let Some(raw_format) = raw_view_format(bpp) else {
            return Ok(None);
        };
        let layout = Samples {
            log2: resident.log2,
            ..Samples::SINGLE
        };
        let texels = layout.texels(rect);
        if texels.is_empty() {
            return Ok(None);
        }
        // Whole bands: what is staged comes from the texture alone, not
        // from a mirror an earlier write-back has yet to bring up to date
        // ([`Renderer::release`]).
        let band = super::surface::BAND_ROWS;
        let texels = Rect {
            x0: 0,
            y0: texels.y0 / band * band,
            x1: resident.surface.width,
            y1: texels.y1.next_multiple_of(band).min(resident.surface.height),
        };
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
                level.resident.is_some()
                    && residents
                        .iter()
                        .any(|r| r.log2 == [0, 0] && holds_level(&r.surface, &level.surface))
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
        self.kernels
            .retain(|_, entry| Arc::strong_count(&entry.kernel) > 1);
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
    /// current. Whether the resident was clean before (it became dirty).
    fn rendered(&mut self, index: usize, rect: Rect, writes: &Writes) -> bool {
        let resident = &mut self.residents[index];
        let newly = resident.dirty.is_none();
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
        newly
    }

    /// The kernel filling a composite's level from a color resident, or
    /// from a depth one ([`CONVERT_SOURCE`]), made once.
    fn convert_pipeline(&mut self, depth: bool) -> Result<Retained<ComputePipeline>, Unmodeled> {
        if let Some(pipeline) = &self.convert_pipelines[usize::from(depth)] {
            return Ok(pipeline.clone());
        }
        let gpu = Renderer::gpu();
        let library = self.library(gpu, CONVERT_SOURCE.to_string())?;
        let name = if depth { "convert_depth" } else { "convert_level" };
        let function = library
            .newFunctionWithName(&NSString::from_str(name))
            .ok_or_else(|| Unmodeled("Metal: no conversion function".into()))?;
        let pipeline = gpu
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| Unmodeled(format!("Metal: conversion pipeline: {e}")))?;
        self.convert_pipelines[usize::from(depth)] = Some(pipeline.clone());
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
            let Some(resident) = self
                .residents
                .iter()
                .find(|r| r.log2 == [0, 0] && holds_level(&r.surface, &level.surface))
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
        let pipeline = self.convert_pipeline(select[4] >= READ_FLOAT_BITS)?;
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

    /// The Metal texture and sampler of what a slot samples, uploaded and
    /// built once while kept.
    fn gpu_texture(
        &mut self,
        gpu: &Gpu,
        sampled: &Sampled,
        tic: &Tic,
        tsc: &Tsc,
        kind: TextureKind,
    ) -> Result<(Retained<MetalTexture>, Retained<Sampler>), Unmodeled> {
        let metal = match sampled {
            Sampled::Composite(texture) => texture.clone(),
            Sampled::Decoded(texture) => {
                let id = (Arc::as_ptr(texture) as usize, kind);
                match self.textures.get(&id) {
                    Some((_, t)) => t.clone(),
                    None => {
                        let t = upload_texture(gpu, texture, kind)?;
                        self.texture_bytes += texture.bytes();
                        self.textures.insert(id, (Arc::clone(texture), t.clone()));
                        t
                    }
                }
            }
            Sampled::Native(raw, pixel_format) => {
                let id = (Arc::as_ptr(raw) as usize, kind);
                match self.native.get(&id) {
                    Some((_, t)) => t.clone(),
                    None => {
                        let t = upload_native(gpu, raw, kind, *pixel_format)?;
                        self.texture_bytes += raw.bytes.len();
                        self.native.insert(id, (Arc::clone(raw), t.clone()));
                        t
                    }
                }
            }
        };
        let (min, max) = tic.lod_range(tsc);
        let key = (tsc.words, [min.to_bits(), max.to_bits()]);
        let sampler = match self.samplers.get(&key) {
            Some(s) => s.clone(),
            None => {
                let s = build_sampler(gpu, tsc, (min, max));
                self.samplers.insert(key, s.clone(), 1);
                s
            }
        };
        Ok((metal, sampler))
    }

    /// The feedback buffer, made when first asked for.
    fn feedback(&mut self) -> Result<&mut Feedback, Unmodeled> {
        if self.feedback.is_none() {
            self.feedback = Some(Feedback {
                buffer: buffer(Renderer::gpu(), &[0; 8 * FEEDBACK_SLOTS])?,
                slot: 0,
                trapped: false,
                span: 0,
                base: 0,
                pending: Vec::new(),
            });
        }
        Ok(self.feedback.as_mut().expect("made above"))
    }

    /// The global memory buffer, made when first asked for.
    fn globals(&mut self) -> Result<&mut Globals, Unmodeled> {
        if self.globals.is_none() {
            let buffer = Renderer::gpu()
                .device
                .newBufferWithLength_options(GLOBAL_ARENA, MTLResourceOptions::StorageModeShared)
                .ok_or_else(|| Unmodeled("Metal: buffer allocation failed".into()))?;
            self.globals = Some(Globals {
                buffer,
                next: GLOBAL_FIRST,
                generation: 0,
                regions: HashMap::new(),
                bound: false,
            });
        }
        Ok(self.globals.as_mut().expect("made above"))
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
            || self.pending != 0
            || self
                .stamp
                .is_some_and(|stamp| writes.unchanged(stamp, start, end))
    }

    /// Stores the texels [`State::pack`] packed, the GPU being done: its
    /// bands into the mirror, and the texels from there to guest memory.
    fn store_packed(&mut self, packed: Packed, memory: &dyn GpuMemory) -> Result<(), Unmodeled> {
        store_staged(&mut self.mirror, packed, memory)
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

/// Composites kept \[assumption: a guest samples a few dozen of the
/// targets it renders (the stages of its post-processing, an atlas); a
/// dropped one is made again when next sampled\].
const COMPOSITE_LIMIT: usize = 48;

/// How Metal's read of a resident gives the components a texture header
/// decodes from the same bytes: which of the read's red, green, blue and
/// alpha (0-3) each is, and what is done to the read first (`READ_*`).
type Reading = ([u32; 4], u32);

/// The read as it is.
const READ_PLAIN: u32 = 0;
/// Red, green and blue decoded from sRGB.
const READ_SRGB: u32 = 1;
/// A depth resident's word as the bits of a float.
const READ_FLOAT_BITS: u32 = 2;
/// A depth resident's 16-bit word as a UNORM.
const READ_UNORM16: u32 = 3;
/// A depth resident's word's bits 8-31 as a UNORM (Z24S8).
const READ_UNORM24_HIGH: u32 = 4;
/// A depth resident's word's bits 0-23 as a UNORM (S8Z24, X8Z24).
const READ_UNORM24_LOW: u32 = 5;

/// The largest region of global memory a program is given \[assumption:
/// what a constant names with a larger size is no buffer descriptor\].
const GLOBAL_LIMIT: usize = 16 << 20;

/// Bytes of the global memory buffer ([`Globals`]): four of the largest
/// regions.
const GLOBAL_ARENA: usize = 4 * GLOBAL_LIMIT;

/// Where the regions of the global memory buffer start: after the fault
/// word, at a multiple of 16 bytes.
const GLOBAL_FIRST: usize = 16;

/// Arena chunks a frame's recorded work may fill before a draw waits
/// for the GPU, which frees them all: bounds the draw data held at once
/// (16 MiB a chunk).
const ARENA_LIMIT: usize = 16;

/// The memory programs address through their constants, on the GPU
/// (`shader::msl` `memory.rs`): one buffer holding the fault word and a
/// copy of every region a draw's program was given. Every program
/// addresses the same copy of a region, so what one stores the next
/// loads, in the order of the draws, without a write-back between them.
struct Globals {
    buffer: Retained<MetalBuffer>,
    /// Bytes of the buffer handed out. Space is handed out once: a region
    /// whose memory changed gets new space, and work recorded earlier
    /// reads the old. A full buffer starts over
    /// ([`Renderer::global_region`]).
    next: usize,
    /// Counts the times the buffer started over.
    generation: u64,
    /// The regions held, by address and size.
    regions: HashMap<(u64, u32), GlobalRegion>,
    /// A draw was given the buffer since its fault word was last looked
    /// at.
    bound: bool,
}

struct GlobalRegion {
    /// Where its bytes are in the buffer.
    offset: usize,
    /// The bytes as guest memory held them when it and the copy were
    /// last equal, and when that was last known to hold.
    content: Vec<u8>,
    stamp: Stamp,
    /// A program that may store to the copy was given it since: the copy
    /// is what memory is to hold, and is written back.
    dirty: bool,
    /// Written back at signals whose write-backs have yet to reach guest
    /// memory ([`Pending`]), how many: the copy is current, memory is not
    /// yet.
    pending: u32,
}

/// The addresses of a global memory region (address, size).
fn global_extent((address, size): (u64, u32)) -> (u64, u64) {
    (address, address + u64::from(size))
}

/// Slots of the feedback buffer ([`Feedback`]): the reports of
/// ZPASS_PIXEL_CNT a channel can make between two write-backs that wait
/// for the GPU; one more makes such a write-back.
const FEEDBACK_SLOTS: usize = 2048;

/// What fragment functions tell the renderer, in a buffer of
/// [`FEEDBACK_SLOTS`] slots of two words (`msl::TRAP_BUFFER`): the trap
/// word (`msl::Translated::traps`) and the count of samples that passed
/// ([`Draw::count_passed`]). A draw is given the slot current when it is
/// recorded; a report of the count ends the slot, so that the counts of
/// the draws before and after it stay apart until the GPU has drawn
/// them and the report can be written ([`resolve_feedback`]).
struct Feedback {
    buffer: Retained<MetalBuffer>,
    /// The slot draws are given now.
    slot: usize,
    /// A draw that may trap was given a slot since the trap words were
    /// last looked at.
    trapped: bool,
    /// The first slot of the draws recorded since the counter was last
    /// reset, and what the slots before it counted since then (slots are
    /// used again once read).
    span: usize,
    base: u64,
    /// The reports that wait for their counts.
    pending: Vec<Report>,
}

/// A signal whose write-backs wait for the GPU ([`Renderer::release`]).
struct Pending {
    /// The command buffer whose work they wait for (`None`: nothing of
    /// their own, only the signals before).
    command: Option<Retained<CommandBuffer>>,
    /// Targets packed, as [`State::pack`] packs them.
    residents: Vec<PendingResident>,
    /// Global memory regions, and where their bytes were copied to.
    regions: Vec<((u64, u32), Retained<MetalBuffer>, usize)>,
    reports: Vec<Report>,
    /// Texture fault words of the draws before it.
    faults: Vec<Faults>,
    /// The global memory buffer's fault word is to be looked at.
    bound: bool,
    /// The memory written.
    extents: Vec<(u64, u64)>,
    /// The release made once they are done (none for a write-back of
    /// targets being dropped, [`Renderer::write_back_later`]).
    release: Option<Release>,
}

struct PendingResident {
    surface: Surface,
    log2: [u32; 2],
    packed: Packed,
}

/// The packed bands into `window` (the mirror of their resident, as
/// [`State::pack`] read it), and the packed texels from there to guest
/// memory.
fn store_staged(window: &mut Window, packed: Packed, memory: &dyn GpuMemory) -> Result<(), Unmodeled> {
    let length = packed.bands.len();
    // SAFETY: the staging range holds `length` bytes the finished GPU work
    // wrote; nothing else uses it until the arena hands it out again.
    let staged = unsafe {
        std::slice::from_raw_parts(
            packed.buffer.contents().as_ptr().cast::<u8>().add(packed.offset),
            length,
        )
    };
    window.data_mut()[packed.bands].copy_from_slice(staged);
    window.store_rect(memory, packed.texels)
}

/// Writes what `pending` waited for to guest memory, the GPU having done
/// the work before it; refuses as a waiting write-back would (an
/// executed sample of an unbound texture, a global memory access outside
/// the buffers).
fn finish(
    state: &mut State,
    pending: &mut Pending,
    memory: &dyn GpuMemory,
    writes: &Writes,
) -> Result<(), Unmodeled> {
    for (buffer, slots) in &pending.faults {
        let words = buffer.contents().cast::<u32>();
        for (slot, refusal) in slots {
            // SAFETY: two words per texture slot of the draw; the GPU
            // finished it.
            if unsafe { words.add(2 * slot + 1).read() } != 0 {
                return Err(Unmodeled(refusal.clone()));
            }
        }
    }
    if pending.bound
        && let Some(globals) = &state.globals
    {
        let word = globals.buffer.contents().cast::<u32>();
        // SAFETY: the buffer starts with the fault word.
        if unsafe { word.read_volatile() } != 0 {
            // SAFETY: as above.
            unsafe { word.write_volatile(0) };
            return Err(Unmodeled(
                "shader: global memory access outside the buffers the program's constants name"
                    .into(),
            ));
        }
    }
    for resident in pending.residents.drain(..) {
        // Its bytes, whatever format the resident there has now.
        let found = state
            .residents
            .iter_mut()
            .find(|r| r.surface == resident.surface && r.log2 == resident.log2);
        match found {
            Some(r) if !r.mirror.data().is_empty() => {
                r.store_packed(resident.packed, memory)?;
                r.pending = r.pending.saturating_sub(1);
                // Memory now holds the mirror: watched from here on
                // \[assumption: no other agent writes the surface while its
                // write-back lands, so none slips in before the watch\].
                let (start, end) = r.extent();
                let mark = memory.watch(start, (end - start) as usize);
                r.stamp = Some(writes.stamp().watched(mark));
            }
            // Dropped since, or its mirror out for a comparison with
            // memory (`Renderer::residents`, which finds memory changed
            // and reloads): its bands as the pack staged them.
            found => {
                if let Some(r) = found {
                    r.pending = r.pending.saturating_sub(1);
                }
                let surface = resident.surface;
                let full = Rect {
                    x0: 0,
                    y0: 0,
                    x1: surface.width,
                    y1: surface.height,
                };
                store_staged(&mut Window::blank(surface, full), resident.packed, memory)?;
            }
        }
    }
    for (key, staging, offset) in &pending.regions {
        // SAFETY: the copy of the region's bytes the finished GPU work
        // made.
        let staged = unsafe {
            std::slice::from_raw_parts(
                staging.contents().as_ptr().cast::<u8>().add(*offset),
                key.1 as usize,
            )
        };
        let region = state
            .globals
            .as_mut()
            .and_then(|globals| globals.regions.get_mut(key));
        if region.as_ref().is_none_or(|region| region.content != staged) {
            memory.write(key.0, staged).ok_or_else(|| {
                Unmodeled(format!(
                    "Metal: global memory write-back to unmapped {:#x}",
                    key.0
                ))
            })?;
        }
        if let Some(region) = region {
            region.content.copy_from_slice(staged);
            region.stamp = writes.stamp();
            region.pending = region.pending.saturating_sub(1);
        }
    }
    if let Some(feedback) = &state.feedback {
        let words = feedback.buffer.contents().cast::<u32>();
        for report in &pending.reports {
            // SAFETY: two words per slot; the GPU finished the draws
            // counted in the report's slots.
            let counted: u64 = report
                .slots
                .clone()
                .map(|slot| u64::from(unsafe { words.add(2 * slot + 1).read_volatile() }))
                .sum();
            let count = report.base + counted;
            let written = if report.one_word {
                memory.write(report.address, &(count as u32).to_le_bytes())
            } else {
                let mut bytes = [0; 16];
                bytes[..8].copy_from_slice(&count.to_le_bytes());
                bytes[8..].copy_from_slice(&report.time.to_le_bytes());
                memory.write(report.address, &bytes)
            };
            if written.is_none() {
                return Err(Unmodeled(format!(
                    "report write to unmapped GPU address {:#x}",
                    report.address
                )));
            }
        }
    }
    Ok(())
}

/// A report of ZPASS_PIXEL_CNT to be written ([`Renderer::report_passed`]).
struct Report {
    address: u64,
    one_word: bool,
    time: u64,
    /// What is counted apart from `slots`.
    base: u64,
    /// The slots whose counts it adds up.
    slots: std::ops::Range<usize>,
}

impl Report {
    /// The addresses it is written to.
    fn extent(&self) -> (u64, u64) {
        (self.address, self.address + if self.one_word { 4 } else { 16 })
    }
}

/// Reads the feedback buffer once the GPU has done every draw recorded:
/// refuses a trap (the address of an instruction neither renderer models
/// that was executed, or bit 30 with the address of a texture query of a
/// handle that names no texture), writes the reports that waited for
/// their counts to guest memory, and frees the slots.
fn resolve_feedback(state: &mut State, memory: &dyn GpuMemory) -> Result<(), Unmodeled> {
    let Some(feedback) = &mut state.feedback else {
        return Ok(());
    };
    // SAFETY: the buffer holds two words per slot; the GPU finished, and
    // nothing else reads or writes it meanwhile (the state is held).
    let words = unsafe {
        std::slice::from_raw_parts_mut(
            feedback.buffer.contents().cast::<u32>().as_ptr(),
            2 * (feedback.slot + 1),
        )
    };
    if std::mem::take(&mut feedback.trapped)
        && let Some(trapped) = words.iter().step_by(2).copied().find(|&word| word != 0)
    {
        words.iter_mut().step_by(2).for_each(|word| *word = 0);
        let what = if trapped & 0x4000_0000 != 0 {
            "a query of a texture slot that names no texture"
        } else {
            "an instruction that is not modeled"
        };
        return Err(Unmodeled(format!(
            "pixel program: {what} was executed at {:#x}",
            trapped & 0x3fff_ffff
        )));
    }
    let counted = |slots: std::ops::Range<usize>| -> u64 {
        slots.map(|slot| u64::from(words[2 * slot + 1])).sum()
    };
    for report in feedback.pending.drain(..) {
        let count = report.base + counted(report.slots.clone());
        let low = (count as u32).to_le_bytes();
        let written = if report.one_word {
            memory.write(report.address, &low)
        } else {
            let mut bytes = [0; 16];
            bytes[..8].copy_from_slice(&count.to_le_bytes());
            bytes[8..].copy_from_slice(&report.time.to_le_bytes());
            memory.write(report.address, &bytes)
        };
        if written.is_none() {
            return Err(Unmodeled(format!(
                "report write to unmapped GPU address {:#x}",
                report.address
            )));
        }
    }
    feedback.base += counted(feedback.span..feedback.slot + 1);
    words.fill(0);
    feedback.span = 0;
    feedback.slot = 0;
    Ok(())
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
    /// resident's red, green, blue and alpha (0-3) as read, zero (4) or
    /// one (5) it is; then how the resident is read (`READ_*`).
    select: [u32; 5],
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
        (components, read): Reading,
        decoded: Option<&Texture>,
        raw: Vec<u8>,
        stamp: Stamp,
    ) -> Result<Composite, Unmodeled> {
        let (width, height) = (levels[0].0.width, levels[0].0.height);
        let layers = composite_layers(tic) as usize;
        let count = levels.len() / layers;
        let descriptor = texture_descriptor(
            composite_kind(tic),
            [width, height, 1],
            layers as u32,
            count as u32,
        );
        descriptor.setUsage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);
        let texture = gpu
            .device
            .newTextureWithDescriptor(&descriptor)
            .ok_or_else(|| Unmodeled("Metal: texture allocation failed".into()))?;
        let mut made = Vec::with_capacity(levels.len());
        // Layer after layer, each's levels (`level_surfaces`).
        for (index, ((surface, range), &resident)) in levels.iter().zip(residents).enumerate() {
            let (slice, level) = (index / count, index % count);
            let resident = if resident {
                // SAFETY: one level of one slice of the texture.
                let view = unsafe {
                    texture.newTextureViewWithPixelFormat_textureType_levels_slices(
                        MTLPixelFormat::RGBA32Float,
                        MTLTextureType::Type2D,
                        NSRange::new(level, 1),
                        NSRange::new(slice, 1),
                    )
                }
                .ok_or_else(|| Unmodeled("Metal: level view".into()))?;
                Some((view, None))
            } else {
                let (w, h, _, texels) = decoded
                    .and_then(|decoded| decoded.level(tic.base_level + level as u32))
                    .ok_or_else(|| Unmodeled("Metal: texture level".into()))?;
                // The level's texels, layer after layer (`Texture::level`).
                let image = w as usize * h as usize;
                let texels = texels
                    .get(slice * image..(slice + 1) * image)
                    .ok_or_else(|| Unmodeled("Metal: texture layer".into()))?;
                // SAFETY: `texels` holds the layer's w*h RGBA32F texels.
                unsafe {
                    texture.replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage(
                        region(Rect {
                            x0: 0,
                            y0: 0,
                            x1: w,
                            y1: h,
                        }),
                        level,
                        slice,
                        NonNull::new(texels.as_ptr() as *mut _).expect("non-null"),
                        w as usize * 16,
                        0,
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
        let swizzled = tic.swizzle.map(|s| match s {
            2..=5 => components[usize::from(s - 2)],
            6 | 7 => 5,
            _ => 4,
        });
        let select = [swizzled[0], swizzled[1], swizzled[2], swizzled[3], read];
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

/// The unsigned-word format a view of a texture of `bytes` bytes a texel
/// reads and writes its bits as.
fn raw_view_format(bytes: u32) -> Option<MTLPixelFormat> {
    Some(match bytes {
        1 => MTLPixelFormat::R8Uint,
        2 => MTLPixelFormat::R16Uint,
        4 => MTLPixelFormat::R32Uint,
        8 => MTLPixelFormat::RG32Uint,
        16 => MTLPixelFormat::RGBA32Uint,
        _ => return None,
    })
}

/// The texture type a composite of `tic` is, as the slots that sample it
/// declare it ([`level_surfaces`] takes no other).
pub(super) fn composite_kind(tic: &Tic) -> TextureKind {
    if tic.target == Target::Cube {
        TextureKind::Cube
    } else if tic.array {
        TextureKind::Array
    } else {
        TextureKind::Flat
    }
}

/// The layers (array layers or cube faces) of `tic` a composite has.
fn composite_layers(tic: &Tic) -> u32 {
    if composite_kind(tic) == TextureKind::Flat {
        1
    } else {
        tic.shape.layers
    }
}

/// The levels of each layer of `tic` (array layer or cube face), from its
/// base level to its maximum level, as the surfaces a render target at
/// each would be, with the bytes of each in the texture's footprint:
/// layer after layer, each's levels in order. `None` for a texture that
/// is not a 2D one (of any layers) or a cube (not an array of cubes) of
/// uncompressed one-texel blocks.
fn level_surfaces(tic: &Tic) -> Option<Vec<(Surface, std::ops::Range<usize>)>> {
    use super::surface::Layout;
    use super::texture::format::Encoding;
    let format = tic.format;
    let shape_fits = match tic.target {
        Target::Texture2D => tic.array || tic.shape.layers == 1,
        Target::Cube => !tic.array && tic.shape.layers == 6,
        _ => false,
    };
    if !shape_fits
        || tic.shape.depth != 1
        || (tic.srgb && format.raw != 0x08)
        || !matches!(format.encoding, Encoding::Packed(_))
        || (format.block_width, format.block_height) != (1, 1)
        || tic.max_level < tic.base_level
    {
        return None;
    }
    let bytes_per_pixel = format.bytes_per_block;
    if let Some(pitch) = tic.pitch {
        if composite_kind(tic) != TextureKind::Flat {
            return None;
        }
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
    let mut surfaces = Vec::new();
    for layer in 0..u64::from(composite_layers(tic)) {
        for level in levels {
            let offset = layer * image.layer_stride + level.offset;
            surfaces.push((
                Surface {
                    address: tic.address + offset,
                    width: level.width,
                    height: level.height,
                    bytes_per_pixel,
                    layout: Layout::BlockLinear(level.surface.block()),
                },
                offset as usize..(offset + level.size) as usize,
            ));
        }
    }
    Some(surfaces)
}

/// Whether every texel of `level` is the texel of `resident` at the same
/// coordinates: the same memory laid out alike (a target may be wider
/// or taller than a texture of its memory, as a GOB-aligned target
/// width is), the resident at least as large.
fn holds_level(resident: &Surface, level: &Surface) -> bool {
    use super::surface::Layout;
    let gobs = |s: &Surface| (s.width * s.bytes_per_pixel).div_ceil(64);
    resident.address == level.address
        && resident.bytes_per_pixel == level.bytes_per_pixel
        && resident.width >= level.width
        && resident.height >= level.height
        && match (resident.layout, level.layout) {
            (Layout::Pitch { pitch: a }, Layout::Pitch { pitch: b }) => a == b,
            (Layout::BlockLinear(a), Layout::BlockLinear(b)) => {
                a == b && gobs(resident) == gobs(level)
            }
            _ => false,
        }
}

/// How Metal's read of a resident holding `content` gives what the
/// texture header `tic` decodes from the same bytes ([`Reading`]); `None`
/// unless there is a way that holds for every texel. A color target
/// whose Metal format is the one that samples the header's bytes as the
/// header says ([`format::native_format`]) reads the same values; an
/// A8B8G8R8 header reads a B8G8R8A8 target's with red and blue exchanged
/// (both sides are the byte / 255: `texture::format::unorm_to_f32`;
/// Metal's conversion of normalized formats [public: Apple, Metal Shading
/// Language Specification, "Conversion rules for normalized integer
/// pixel data types"]). An sRGB target's resident holds its bytes in the
/// linear format ([`format::pixel_format`]), so a header reads them raw
/// there, and one with sRGB decoding decodes them. A depth buffer's
/// resident holds its words ([`format::depth_pixel_format`]): a ZF32
/// header reads Z32Float's as floats, a Z16 header Z16Unorm's as UNORM,
/// and a header of a 24-bit depth format the depth bits of Z24S8's,
/// S8Z24's or X8Z24's words laid out as it says (`texture::format`
/// 0x29-0x2b) as UNORM, where it selects no stencil bits and declares the
/// depth UNORM (absent components 0, alpha 1, as `Texture::decode`).
fn shared_components(tic: &Tic, content: Content) -> Option<Reading> {
    use super::texture::format::ComponentType;
    if let Content::Depth(depth) = content {
        // (read, the header's depth component, its stencil component).
        let (read, depth_component, stencil) = match (depth, tic.format.raw) {
            (DepthFormat::Z24S8, 0x29) => (READ_UNORM24_HIGH, 1, Some(0)),
            (DepthFormat::S8Z24 | DepthFormat::X8Z24, 0x2a) => (READ_UNORM24_LOW, 0, None),
            (DepthFormat::S8Z24 | DepthFormat::X8Z24, 0x2b) => (READ_UNORM24_LOW, 0, Some(1)),
            _ => (0, 0, None),
        };
        if read != 0 {
            let stencil_selected = stencil.is_some_and(|s| tic.swizzle.contains(&(2 + s)));
            if tic.srgb || stencil_selected || tic.types[depth_component] != ComponentType::Unorm {
                return None;
            }
            // The depth in the read's red; the stencil never selected;
            // absent components as constants.
            let mut components = [4, 4, 4, 5];
            components[depth_component] = 0;
            return Some((components, read));
        }
    }
    let linear = Tic {
        srgb: false,
        ..tic.clone()
    };
    let native = format::native_format(&linear, TextureKind::Flat)?;
    match content {
        Content::Color(format) => {
            let held = format::pixel_format(format)?;
            let components = if native == held {
                [0, 1, 2, 3]
            } else if (native, held) == (MTLPixelFormat::RGBA8Unorm, MTLPixelFormat::BGRA8Unorm) {
                [2, 1, 0, 3]
            } else {
                return None;
            };
            Some((components, if tic.srgb { READ_SRGB } else { READ_PLAIN }))
        }
        Content::Depth(DepthFormat::Z32Float)
            if native == MTLPixelFormat::R32Float && !tic.srgb =>
        {
            Some(([0, 1, 2, 3], READ_FLOAT_BITS))
        }
        Content::Depth(DepthFormat::Z16Unorm)
            if native == MTLPixelFormat::R16Unorm && !tic.srgb =>
        {
            Some(([0, 1, 2, 3], READ_UNORM16))
        }
        Content::Depth(_) => None,
    }
}

/// The kernels filling a level of a [`Composite`] from the resident that
/// is the level: each texel as Metal reads it from the resident (a
/// depth resident's as the red of a texel whose green and blue are 0
/// and alpha 1), read as `select[4]` says and its components selected
/// by the rest of `select` ([`Composite::select`]). sRGB decoding is the
/// transfer function's inverse computed in floats, which can differ
/// from `texture::format`'s in the last bit [known difference].
const CONVERT_SOURCE: &str = r#"#include <metal_stdlib>
using namespace metal;
static float4 selected(float4 c, constant uint* how) {
    const float v[6] = { c.x, c.y, c.z, c.w, 0.0f, 1.0f };
    return float4(v[how[0]], v[how[1]], v[how[2]], v[how[3]]);
}
kernel void convert_level(texture2d<float, access::read> source [[texture(0)]], texture2d<float, access::write> level [[texture(1)]], constant uint* how [[buffer(0)]], uint2 g [[thread_position_in_grid]]) {
    if (g.x >= level.get_width() || g.y >= level.get_height()) { return; }
    float4 c = source.read(g);
    if (how[4] == 1u) {
        const float3 low = c.xyz / 12.92f;
        const float3 high = pow((c.xyz + 0.055f) / 1.055f, float3(2.4f));
        c.xyz = select(high, low, c.xyz <= 0.04045f);
    }
    level.write(selected(c, how), g);
}
kernel void convert_depth(texture2d<uint, access::read> source [[texture(0)]], texture2d<float, access::write> level [[texture(1)]], constant uint* how [[buffer(0)]], uint2 g [[thread_position_in_grid]]) {
    if (g.x >= level.get_width() || g.y >= level.get_height()) { return; }
    const uint word = source.read(g).x;
    const uint read = how[4];
    const float depth = read == 2u ? as_type<float>(word)
        : read == 3u ? float(word) / 65535.0f
        : read == 4u ? float(word >> 8) / 16777215.0f
        : float(word & 0xffffffu) / 16777215.0f;
    level.write(selected(float4(depth, 0.0f, 0.0f, 1.0f), how), g);
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

/// Shared buffer space for the data of the draws command buffers record
/// (vertex streams, indices, large bindings, staging), handed out in
/// order a chunk at a time: allocating a Metal buffer per binding costs a
/// kernel call each. A chunk is filled again once every command buffer
/// that may read it completed, which the arena learns from the command
/// buffers committed ([`Arena::committed`]); all of them at once after a
/// wait ([`Arena::reset`]).
#[derive(Default)]
struct Arena {
    /// Chunks with the serial of the last command buffer that may read
    /// them (see `committed`).
    chunks: Vec<(Retained<MetalBuffer>, u64)>,
    /// The chunk being filled and the bytes used of it.
    current: usize,
    used: usize,
    /// Command buffers committed so far: the one being recorded gets
    /// serial `committed + 1`.
    committed: u64,
    /// Committed command buffers not known to have completed, by serial.
    flight: std::collections::VecDeque<(u64, Retained<CommandBuffer>)>,
    /// Every command buffer up to this serial completed.
    done: u64,
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
        let serial = self.committed + 1;
        loop {
            if let Some((chunk, last)) = self.chunks.get_mut(self.current) {
                let offset = self.used.next_multiple_of(Self::ALIGN);
                if offset + length <= chunk.length() {
                    // SAFETY: the range is inside the shared buffer, which
                    // no command buffer still to complete was given (see
                    // `next_chunk`).
                    unsafe {
                        let base = chunk.contents().as_ptr().cast::<u8>().add(offset);
                        std::ptr::copy_nonoverlapping(data.as_ptr(), base, data.len());
                    }
                    self.used = offset + length;
                    *last = serial;
                    return Ok((chunk.clone(), offset));
                }
            }
            self.next_chunk(length)?;
        }
    }

    /// Moves on to a chunk of at least `length` bytes no command buffer
    /// still to complete reads: a free one, or a new one.
    fn next_chunk(&mut self, length: usize) -> Result<(), Unmodeled> {
        self.poll();
        let done = self.done;
        // An oversized chunk is not kept once done.
        let mut index = 0;
        while index < self.chunks.len() {
            let (chunk, last) = &self.chunks[index];
            if *last <= done && chunk.length() > Self::CHUNK && index != self.current {
                self.chunks.remove(index);
                if index < self.current {
                    self.current -= 1;
                }
            } else {
                index += 1;
            }
        }
        let count = self.chunks.len();
        let free = (1..=count)
            .map(|step| (self.current + step) % count.max(1))
            .find(|&i| {
                i != self.current
                    && self.chunks[i].1 <= done
                    && self.chunks[i].0.length() >= length
            });
        self.current = match free {
            Some(index) => index,
            None => {
                let chunk = Renderer::gpu()
                    .device
                    .newBufferWithLength_options(
                        length.max(Self::CHUNK),
                        MTLResourceOptions::StorageModeShared,
                    )
                    .ok_or_else(|| Unmodeled("Metal: buffer allocation failed".into()))?;
                let at = (self.current + 1).min(count);
                self.chunks.insert(at, (chunk, 0));
                at
            }
        };
        self.used = 0;
        Ok(())
    }

    /// `command`, the command buffer recorded since the last commit, was
    /// committed.
    fn committed(&mut self, command: Retained<CommandBuffer>) {
        self.committed += 1;
        self.flight.push_back((self.committed, command));
    }

    /// Notes the command buffers that completed, in commit order.
    fn poll(&mut self) {
        use objc2_metal::MTLCommandBufferStatus;
        while let Some((serial, command)) = self.flight.front() {
            let status = command.status();
            if status != MTLCommandBufferStatus::Completed && status != MTLCommandBufferStatus::Error
            {
                break;
            }
            self.done = *serial;
            self.flight.pop_front();
        }
    }

    /// Every command buffer committed completed: every chunk is free;
    /// an oversized one is dropped.
    fn reset(&mut self) {
        self.flight.clear();
        self.done = self.committed;
        self.chunks.retain(|(chunk, _)| chunk.length() <= Self::CHUNK);
        self.current = 0;
        self.used = 0;
    }

    /// Chunks holding data of work recorded or running: of command buffers
    /// not known to have completed.
    fn chunks_in_use(&mut self) -> usize {
        self.poll();
        let done = self.done;
        self.chunks.iter().filter(|(_, last)| *last > done).count()
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

impl Coherent<'_> {
    /// The write-back an access needs. An access can only fail as
    /// unmapped, so what a failed write-back refused is said here.
    fn before(&self, address: u64, length: usize) -> Option<()> {
        self.renderer
            .before_access(self.memory, self.writes, address, length)
            .map_err(|Unmodeled(reason)| {
                eprintln!(
                    "Metal renderer: write-back before an access of {address:#x} failed: {reason}"
                );
            })
            .ok()
    }
}

impl GpuMemory for Coherent<'_> {
    fn read(&self, address: u64, buffer: &mut [u8]) -> Option<()> {
        self.before(address, buffer.len())?;
        self.memory.read(address, buffer)
    }

    fn write(&self, address: u64, data: &[u8]) -> Option<()> {
        self.before(address, data.len())?;
        self.memory.write(address, data)
    }

    fn probe(&self, address: u64, buffer: &mut [u8]) -> Option<()> {
        self.before(address, buffer.len())?;
        self.memory.probe(address, buffer)
    }

    fn equals(&self, address: u64, expected: &[u8]) -> Option<bool> {
        self.before(address, expected.len())?;
        self.memory.equals(address, expected)
    }

    fn watch(&self, address: u64, len: usize) -> Option<WatchMark> {
        self.memory.watch(address, len)
    }

    /// A write-back over the bytes still to come lands first, and counts
    /// as a write.
    fn unwritten_since(&self, address: u64, len: usize, mark: WatchMark) -> bool {
        self.before(address, len).is_some() && self.memory.unwritten_since(address, len, mark)
    }
}

/// The answers to a program's texture queries (`shader::msl`
/// `texture.rs` `txq`), five words per handle of `queries` (offsets in
/// constant buffer `buffer`, the bound-texture handles): what
/// `texture::query_header` takes from the header, for the program to
/// take a level's from. All zeros for a handle that names no header.
fn query_words(
    textures: &DrawTextures<'_>,
    constants: &[(Vec<u8>, usize)],
    buffer: u32,
    queries: &[u32],
) -> Vec<u32> {
    let mut words = Vec::with_capacity(5 * queries.len());
    for &offset in queries {
        let header = constants
            .get(buffer as usize)
            .and_then(|(data, _)| data.get(offset as usize..offset as usize + 4))
            .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("4 bytes")))
            .and_then(|handle| textures.header(handle).ok());
        words.extend(match header {
            Some((tic, _)) => {
                let base = |size: u32| size.checked_shr(tic.base_level).unwrap_or(0);
                let (third, flags) = if tic.target == Target::Cube {
                    (tic.shape.layers / 6, 1)
                } else if tic.array {
                    (tic.shape.layers, 1)
                } else {
                    (base(tic.shape.depth), 3)
                };
                [
                    base(tic.shape.width),
                    base(tic.shape.height),
                    third,
                    tic.max_level.saturating_sub(tic.base_level) + 1,
                    flags,
                ]
            }
            None => [0; 5],
        });
    }
    words
}
