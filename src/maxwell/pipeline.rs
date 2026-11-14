//! A draw of the 3D engine: the state the registers hold at the draw,
//! vertex fetch, vertex shading, primitive assembly, clipping, the
//! viewport transform, and hand-off to the rasterizer.
//!
//! Register offsets, fields and enumerants are those of [public: NVIDIA
//! open-gpu-doc `classes/3d/clb197.h`] (names in the rustdoc are its method
//! names without the `NVB197_` prefix). State the UI path is not known to
//! use and that would change the result is refused with `Unmodeled` naming
//! the method and value rather than ignored.

use std::sync::Arc;

use super::blend::{Blend, Factor, Operation};
use super::raster::{self, Backend, BandSet, ColorOutput, Compare, DepthOutput, Triangle};
use super::resources::{DrawTextures, Pools, ProgramCache, TextureCache};
use super::shader::{self, Attributes, Env, Interpolation, Plane, Program};
use super::surface::{Layout, Rect, Surface, Window};
use super::texture::format::{ColorFormat, DepthFormat};
use super::texture::layout::BlockLinear;
use super::{GpuMemory, Unmodeled};

/// The 3D engine's method registers, by byte offset.
#[derive(Clone, Copy)]
pub struct Registers<'a>(pub &'a [u32]);

impl Registers<'_> {
    #[inline]
    pub fn get(&self, offset: u32) -> u32 {
        self.0[(offset / 4) as usize]
    }

    fn float(&self, offset: u32) -> f32 {
        f32::from_bits(self.get(offset))
    }

    fn address(&self, upper: u32, lower: u32) -> u64 {
        u64::from(self.get(upper) & 0xFF) << 32 | u64::from(self.get(lower))
    }

    fn enabled(&self, offset: u32) -> bool {
        self.get(offset) & 1 != 0
    }
}

fn refuse(what: &str, offset: u32, value: u32) -> Unmodeled {
    Unmodeled(format!("3D: {what} ({offset:#06x} = {value:#x})"))
}

/// Requires `register & mask == expected`.
fn require(
    r: Registers<'_>,
    what: &str,
    offset: u32,
    mask: u32,
    expected: u32,
) -> Result<(), Unmodeled> {
    let value = r.get(offset);
    if value & mask == expected {
        Ok(())
    } else {
        Err(refuse(what, offset, value))
    }
}

/// Primitive topologies (`BEGIN_OP`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Topology {
    Triangles,
    TriangleStrip,
    TriangleFan,
    Quads,
    QuadStrip,
    Polygon,
}

impl Topology {
    pub fn from_raw(raw: u32) -> Result<Self, Unmodeled> {
        Ok(match raw {
            4 => Topology::Triangles,
            5 => Topology::TriangleStrip,
            6 => Topology::TriangleFan,
            7 => Topology::Quads,
            8 => Topology::QuadStrip,
            9 => Topology::Polygon,
            _ => {
                return Err(Unmodeled(format!(
                    "3D: primitive topology {raw:#x} (points, lines, adjacency and patches are not modeled)"
                )));
            }
        })
    }
}

/// One vertex of a draw: where to fetch it from, or a primitive restart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Element {
    Vertex {
        /// Vertex fetch index: index buffer value plus
        /// `SET_GLOBAL_BASE_VERTEX_INDEX`, or array position.
        fetch: u32,
        /// What the shader reads as the vertex ID (a\[0x2fc\]).
        id: u32,
    },
    Restart,
}

/// A batch of vertices between `BEGIN` and `END`.
#[derive(Debug, Clone)]
pub struct Batch {
    pub topology: Topology,
    /// Instance ID the shader reads (a\[0x2f8\]).
    pub instance: u32,
    /// `SET_GLOBAL_BASE_INSTANCE_INDEX`, added to the instance for
    /// per-instance vertex fetch.
    pub base_instance: u32,
    pub elements: Vec<Element>,
}

/// A constant buffer binding: address and size in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstantBuffer {
    pub address: u64,
    pub size: u32,
}

/// Bind groups (`BIND_GROUP_CONSTANT_BUFFER(j)`, j < 5) by 18 slots.
pub type Bindings = [[Option<ConstantBuffer>; 18]; 5];

/// Persistent caches of the engine.
#[derive(Default)]
pub struct Caches {
    pub programs: ProgramCache,
    pub textures: TextureCache,
}

/// Engine state outside the register file a draw needs.
pub struct Context<'a> {
    pub registers: Registers<'a>,
    pub bindings: &'a Bindings,
    pub memory: &'a dyn GpuMemory,
    /// The channel's Metal renderer, which draws what it can express.
    pub metal: Option<&'a super::metal::Renderer>,
}

const SET_COLOR_TARGET_A: u32 = 0x0800;
const SET_VIEWPORT_SCALE_X: u32 = 0x0a00;
const SET_VIEWPORT_COORDINATE_SWIZZLE: u32 = 0x0a18;
const SET_VIEWPORT_CLIP_MIN_Z: u32 = 0x0c08;
const SET_VIEWPORT_CLIP_MAX_Z: u32 = 0x0c0c;
const SET_WINDOW_OFFSET_X: u32 = 0x0df8;
const SET_WINDOW_OFFSET_Y: u32 = 0x0dfc;
const SET_SCISSOR_ENABLE: u32 = 0x0e00;
const SET_SCISSOR_HORIZONTAL: u32 = 0x0e04;
const SET_SCISSOR_VERTICAL: u32 = 0x0e08;
const SET_VIEWPORT_Z_CLIP: u32 = 0x0d7c;
const SET_FRONT_POLYGON_MODE: u32 = 0x0dac;
const SET_BACK_POLYGON_MODE: u32 = 0x0db0;
const SET_POLY_OFFSET_FILL: u32 = 0x0dc8;
const SET_SINGLE_CT_WRITE_CONTROL: u32 = 0x0f90;
const SET_CT_MRT_ENABLE: u32 = 0x0fac;
const SET_ZT_A: u32 = 0x0fe0;
const SET_ZT_B: u32 = 0x0fe4;
const SET_ZT_FORMAT: u32 = 0x0fe8;
const SET_ZT_BLOCK_SIZE: u32 = 0x0fec;
const SET_SURFACE_CLIP_HORIZONTAL: u32 = 0x0ff4;
const SET_SURFACE_CLIP_VERTICAL: u32 = 0x0ff8;
const SET_VERTEX_ATTRIBUTE_A: u32 = 0x1160;
const SET_CT_SELECT: u32 = 0x121c;
const SET_ZT_SIZE_A: u32 = 0x1228;
const SET_ZT_SIZE_B: u32 = 0x122c;
const SET_ZT_SIZE_C: u32 = 0x1230;
const SET_SAMPLER_BINDING: u32 = 0x1234;
const SET_DEPTH_TEST: u32 = 0x12cc;
const SET_BLEND_STATE_PER_TARGET: u32 = 0x12e4;
const SET_DEPTH_WRITE: u32 = 0x12e8;
const SET_ALPHA_TEST: u32 = 0x12ec;
const SET_DEPTH_FUNC: u32 = 0x130c;
const SET_BLEND_CONST_RED: u32 = 0x131c;
const SET_BLEND_SEPARATE_FOR_ALPHA: u32 = 0x133c;
const SET_BLEND_COLOR_OP: u32 = 0x1340;
const SET_BLEND: u32 = 0x1360;
const SET_STENCIL_TEST: u32 = 0x1380;
const SET_PS_SATURATE: u32 = 0x13a8;
const SET_WINDOW_ORIGIN: u32 = 0x13ac;
const SET_USER_CLIP_ENABLE: u32 = 0x1510;
const SET_ZT_SELECT: u32 = 0x1538;
const SET_TEX_SAMPLER_POOL_A: u32 = 0x155c;
const SET_TEX_SAMPLER_POOL_B: u32 = 0x1560;
const SET_TEX_SAMPLER_POOL_C: u32 = 0x1564;
const SET_TEX_HEADER_POOL_A: u32 = 0x1574;
const SET_TEX_HEADER_POOL_B: u32 = 0x1578;
const SET_TEX_HEADER_POOL_C: u32 = 0x157c;
const SET_SRGB_WRITE: u32 = 0x15b8;
const SET_RT_LAYER: u32 = 0x15cc;
const SET_ANTI_ALIAS: u32 = 0x15d0;
const SET_PROGRAM_REGION_A: u32 = 0x1608;
const SET_PROGRAM_REGION_B: u32 = 0x160c;
const SET_PROVOKING_VERTEX: u32 = 0x1684;
const SET_ZT_LAYER: u32 = 0x179c;
const SET_VERTEX_STREAM_INSTANCE_A: u32 = 0x1880;
const OGL_SET_CULL: u32 = 0x1918;
const OGL_SET_FRONT_FACE: u32 = 0x191c;
const OGL_SET_CULL_FACE: u32 = 0x1920;
const SET_VIEWPORT_PIXEL: u32 = 0x1924;
const SET_VIEWPORT_SCALE_OFFSET: u32 = 0x192c;
const SET_VIEWPORT_CLIP_CONTROL: u32 = 0x193c;
const SET_WINDOW_CLIP_ENABLE: u32 = 0x194c;
const SET_DEPTH_BOUNDS_TEST: u32 = 0x19bc;
const SET_LOGIC_OP: u32 = 0x19c4;
const SET_CT_WRITE: u32 = 0x1a00;
const SET_VERTEX_STREAM_A_FORMAT: u32 = 0x1c00;
const SET_VERTEX_STREAM_LIMIT_A_A: u32 = 0x1f00;
const SET_BLEND_PER_TARGET: u32 = 0x1e00;
const SET_PIPELINE_SHADER: u32 = 0x2000;
const SET_PIPELINE_PROGRAM: u32 = 0x2004;
const SET_PIPELINE_BINDING: u32 = 0x2010;
const SET_BINDLESS_TEXTURE: u32 = 0x2608;
const SET_RASTER_ENABLE: u32 = 0x037c;
const SET_STREAM_OUTPUT: u32 = 0x0744;

/// Pipeline stages (`SET_PIPELINE_*(j)`): 0 vertex A (cull before fetch),
/// 1 vertex B, 2 tessellation init, 3 tessellation, 4 geometry, 5 pixel.
const PIPELINE_STRIDE: u32 = 64;
const VERTEX_PIPELINE: u32 = 1;
const PIXEL_PIPELINE: u32 = 5;

/// The geometry of a render target or depth buffer.
fn layout(
    what: &str,
    offset: u32,
    memory_field: u32,
    pitch: bool,
    pitch_bytes: u32,
) -> Result<Layout, Unmodeled> {
    if pitch {
        return Ok(Layout::Pitch { pitch: pitch_bytes });
    }
    if memory_field & 0xF != 0 {
        return Err(refuse(&format!("{what} block width"), offset, memory_field));
    }
    Ok(Layout::BlockLinear(BlockLinear {
        block_height_log2: (memory_field >> 4) & 0xF,
        block_depth_log2: (memory_field >> 8) & 0xF,
    }))
}

/// Render target `index` (`SET_COLOR_TARGET_*(j)`), if its format is not
/// `DISABLED`.
pub fn color_target(
    r: Registers<'_>,
    index: u32,
) -> Result<Option<(Surface, ColorFormat)>, Unmodeled> {
    let base = SET_COLOR_TARGET_A + index * 64;
    let raw_format = r.get(base + 0x10) & 0xFF;
    if raw_format == 0 {
        return Ok(None);
    }
    let format = ColorFormat::from_raw(raw_format)
        .ok_or_else(|| refuse("color target format", base + 0x10, raw_format))?;
    let memory = r.get(base + 0x14);
    let pitch = memory & (1 << 12) != 0;
    // Pitch-linear targets give their pitch in bytes in place of the
    // width [public: Ryujinx `TextureCache.FindOrCreateTexture`,
    // `WidthOrStride`].
    let width_field = r.get(base + 0x08) & 0x0FFF_FFFF;
    let bpp = format.bytes_per_pixel();
    let width = if pitch {
        width_field / bpp
    } else {
        width_field
    };
    let layer = r.get(base + 0x20) & 0xFFFF;
    if layer != 0 {
        return Err(refuse("color target layer offset", base + 0x20, layer));
    }
    let layout = layout("color target", base + 0x14, memory, pitch, width_field)?;
    Ok(Some((
        Surface {
            address: r.address(base, base + 0x04),
            width,
            height: r.get(base + 0x0c) & 0x1_FFFF,
            bytes_per_pixel: bpp,
            layout,
        },
        format,
    )))
}

/// The depth buffer (`SET_ZT_*`), if `SET_ZT_SELECT` enables it.
pub fn depth_target(r: Registers<'_>) -> Result<Option<(Surface, DepthFormat)>, Unmodeled> {
    if r.get(SET_ZT_SELECT) & 1 == 0 {
        return Ok(None);
    }
    let raw_format = r.get(SET_ZT_FORMAT) & 0x1F;
    let format = DepthFormat::from_raw(raw_format)
        .ok_or_else(|| refuse("depth target format", SET_ZT_FORMAT, raw_format))?;
    let layer = r.get(SET_ZT_LAYER) & 0xFFFF;
    if layer != 0 {
        return Err(refuse("depth target layer", SET_ZT_LAYER, layer));
    }
    let layout = layout(
        "depth target",
        SET_ZT_BLOCK_SIZE,
        r.get(SET_ZT_BLOCK_SIZE),
        false,
        0,
    )?;
    let _ = r.get(SET_ZT_SIZE_C);
    Ok(Some((
        Surface {
            address: r.address(SET_ZT_A, SET_ZT_B),
            width: r.get(SET_ZT_SIZE_A) & 0x0FFF_FFFF,
            height: r.get(SET_ZT_SIZE_B) & 0x1_FFFF,
            bytes_per_pixel: format.bytes_per_pixel(),
            layout,
        },
        format,
    )))
}

/// The rectangle draws and clears may touch: the surface clip
/// (`SET_SURFACE_CLIP_*`), in physical coordinates.
pub fn surface_clip(r: Registers<'_>) -> Rect {
    let horizontal = r.get(SET_SURFACE_CLIP_HORIZONTAL);
    let vertical = r.get(SET_SURFACE_CLIP_VERTICAL);
    let x = horizontal & 0xFFFF;
    let y = vertical & 0xFFFF;
    Rect {
        x0: x,
        y0: y,
        x1: x + (horizontal >> 16),
        y1: y + (vertical >> 16),
    }
}

/// Scissor 0 (`SET_SCISSOR_*(0)`): `[min, max)` in each axis.
pub fn scissor(r: Registers<'_>) -> Option<Rect> {
    if !r.enabled(SET_SCISSOR_ENABLE) {
        return None;
    }
    let horizontal = r.get(SET_SCISSOR_HORIZONTAL);
    let vertical = r.get(SET_SCISSOR_VERTICAL);
    Some(Rect {
        x0: horizontal & 0xFFFF,
        y0: vertical & 0xFFFF,
        x1: horizontal >> 16,
        y1: vertical >> 16,
    })
}

/// State this model refuses rather than ignore: features whose effect is
/// not implemented. Checked at every draw.
fn check_unmodeled_state(r: Registers<'_>) -> Result<(), Unmodeled> {
    // SET_WINDOW_ORIGIN MODE: only UPPER_LEFT. With LOWER_LEFT the public
    // reimplementations flip y about different axes (Ryujinx flips scissors
    // about the surface clip height but viewports about their own center),
    // so the hardware rule is not established.
    require(r, "window origin lower-left", SET_WINDOW_ORIGIN, 1, 0)?;
    require(r, "window offset x", SET_WINDOW_OFFSET_X, !0, 0)?;
    require(r, "window offset y", SET_WINDOW_OFFSET_Y, !0, 0)?;
    require(r, "window clip", SET_WINDOW_CLIP_ENABLE, 1, 0)?;
    require(r, "stencil test", SET_STENCIL_TEST, 1, 0)?;
    require(r, "depth bounds test", SET_DEPTH_BOUNDS_TEST, 1, 0)?;
    require(r, "alpha test", SET_ALPHA_TEST, 1, 0)?;
    require(r, "logic operation", SET_LOGIC_OP, 1, 0)?;
    // SET_USER_CLIP_ENABLE: applied in `setup` to the clip distances the
    // vertex shader writes. NVN's init blast enables all eight.
    require(r, "polygon offset (fill)", SET_POLY_OFFSET_FILL, 1, 0)?;
    require(r, "stream output", SET_STREAM_OUTPUT, 1, 0)?;
    // SET_ANTI_ALIAS SAMPLES: only 1X1 (0).
    require(r, "multisampling", SET_ANTI_ALIAS, 0xF, 0)?;
    // SET_FRONT/BACK_POLYGON_MODE: only FILL (0x1B02).
    require(r, "front polygon mode", SET_FRONT_POLYGON_MODE, !0, 0x1B02)?;
    require(r, "back polygon mode", SET_BACK_POLYGON_MODE, !0, 0x1B02)?;
    require(r, "single render target broadcast", SET_CT_MRT_ENABLE, 1, 1)?;
    require(r, "pixel shader output saturation", SET_PS_SATURATE, !0, 0)?;
    // SET_VIEWPORT_PIXEL CENTER: only AT_HALF_INTEGERS (0).
    require(r, "pixel centers at integers", SET_VIEWPORT_PIXEL, 1, 0)?;
    require(
        r,
        "viewport transform disabled",
        SET_VIEWPORT_SCALE_OFFSET,
        1,
        1,
    )?;
    // SET_RT_LAYER: V 15:0 must select layer 0. CONTROL (bit 16) =
    // GEOMETRY_SHADER_SELECTS_LAYER, as NVN's init blast sets it, is
    // checked against the vertex shader's outputs in `draw`.
    require(r, "render target layer", SET_RT_LAYER, 0xFFFF, 0)?;
    // SET_VIEWPORT_COORDINATE_SWIZZLE(0): only the identity, the
    // netlist's initial value.
    require(
        r,
        "viewport swizzle",
        SET_VIEWPORT_COORDINATE_SWIZZLE,
        0x7777,
        0x6420,
    )?;
    for pipeline in [0, 2, 3, 4] {
        let offset = SET_PIPELINE_SHADER + pipeline * PIPELINE_STRIDE;
        require(
            r,
            "shader stage other than vertex B and pixel",
            offset,
            1,
            0,
        )?;
    }
    Ok(())
}

/// The program of pipeline `index`, if enabled.
fn program(
    context: &Context<'_>,
    caches: &mut Caches,
    index: u32,
) -> Result<Option<Arc<Program>>, Unmodeled> {
    let r = context.registers;
    let control = r.get(SET_PIPELINE_SHADER + index * PIPELINE_STRIDE);
    if control & 1 == 0 {
        return Ok(None);
    }
    let base = r.address(SET_PROGRAM_REGION_A, SET_PROGRAM_REGION_B);
    let offset = r.get(SET_PIPELINE_PROGRAM + index * PIPELINE_STRIDE);
    caches
        .programs
        .get(context.memory, base + u64::from(offset))
        .map(Some)
}

/// The contents of the constant buffers pipeline `index` sees, as far as
/// `program` can read them (the shader's own bound: reads past a slice
/// return 0 like reads past the bound size), with each slot's bound size.
fn constant_buffers(
    context: &Context<'_>,
    index: u32,
    program: &Program,
) -> Result<Vec<(Vec<u8>, usize)>, Unmodeled> {
    let r = context.registers;
    let group = r.get(SET_PIPELINE_BINDING + index * PIPELINE_STRIDE) & 7;
    let slots = context.bindings.get(group as usize).ok_or_else(|| {
        refuse(
            "pipeline binding group",
            SET_PIPELINE_BINDING + index * PIPELINE_STRIDE,
            group,
        )
    })?;
    let texture_slot = (r.get(SET_BINDLESS_TEXTURE) & 0x1F) as usize;
    slots
        .iter()
        .enumerate()
        .map(|(slot, binding)| match binding {
            None => Ok((Vec::new(), 0)),
            Some(buffer) => {
                let size = buffer.size as usize;
                let mut extent = program.constant_buffer_extent(slot);
                if slot == texture_slot {
                    extent = extent.map(|e| e.max(program.texture_handle_extent()));
                }
                let length = extent.map_or(size, |e| e.min(size));
                Ok((read_mapped(context.memory, buffer.address, length), size))
            }
        })
        .collect()
}

/// The leading bytes of `length` at `address` the GPU can read: reading
/// stops at the first 4 KiB page (the SMMU's page size) that does not
/// translate. A bound constant buffer may reach past the memory its owner
/// mapped (NVN binds 0xE900 bytes of which nvservices mapped 0x7900, seen
/// from starter); a shader reading there is refused when it does.
fn read_mapped(memory: &dyn GpuMemory, address: u64, length: usize) -> Vec<u8> {
    const PAGE: u64 = 0x1000;
    let mut bytes = vec![0; length];
    let mut done = 0;
    while done < length {
        let at = address + done as u64;
        let chunk = ((PAGE - at % PAGE) as usize).min(length - done);
        if memory.read(at, &mut bytes[done..done + chunk]).is_none() {
            break;
        }
        done += chunk;
    }
    bytes.truncate(done);
    bytes
}

fn env<'a>(
    buffers: &'a [(Vec<u8>, usize)],
    textures: &'a dyn super::sampling::TextureUnit,
    r: Registers<'_>,
    memory: &'a dyn GpuMemory,
) -> Env<'a> {
    Env {
        const_buffers: std::array::from_fn(|slot| {
            let (data, size) = &buffers[slot];
            shader::ConstantBuffer { data, size: *size }
        }),
        textures,
        texture_buffer_index: r.get(SET_BINDLESS_TEXTURE) & 0x1F,
        memory,
        // Upper-left origin only (see `check_unmodeled_state`).
        y_direction: 1.0,
    }
}

/// Vertex attribute formats (`SET_VERTEX_ATTRIBUTE_A` `COMPONENT_BIT_WIDTHS`):
/// component sizes in bits, in memory order.
fn component_widths(code: u32) -> Option<&'static [u32]> {
    Some(match code {
        0x01 => &[32, 32, 32, 32],
        0x02 => &[32, 32, 32],
        0x03 => &[16, 16, 16, 16],
        0x04 => &[32, 32],
        0x05 => &[16, 16, 16],
        // A8B8G8R8 and G8R8 are packed little-endian words with R lowest,
        // the same bytes as R8_G8_B8_A8 / R8_G8.
        0x0A | 0x2F => &[8, 8, 8, 8],
        0x0F => &[16, 16],
        0x12 => &[32],
        0x13 => &[8, 8, 8],
        0x18 | 0x32 => &[8, 8],
        0x1B => &[16],
        0x1D => &[8],
        _ => return None,
    })
}

/// Fetches the attribute `attribute` of a vertex: four words as the
/// vertex shader reads them (floats, or integers for SINT/UINT).
fn fetch_attribute(raw: u32, bytes: &[u8]) -> Result<[u32; 4], Unmodeled> {
    let code = (raw >> 21) & 0x3F;
    let kind = (raw >> 27) & 7;
    let widths = component_widths(code)
        .ok_or_else(|| Unmodeled(format!("3D: vertex attribute component sizes {code:#x}")))?;
    let integer = matches!(kind, 3 | 4);
    // Missing components read as (0, 0, 0, 1).
    let one = if integer { 1 } else { 1.0f32.to_bits() };
    let mut out = [0, 0, 0, one];
    let mut offset = 0;
    for (component, &width) in widths.iter().enumerate() {
        let size = (width / 8) as usize;
        let mut word = [0u8; 4];
        word[..size].copy_from_slice(&bytes[offset..offset + size]);
        offset += size;
        let value = u32::from_le_bytes(word);
        let signed = |value: u32| ((value << (32 - width)) as i32) >> (32 - width);
        let max_signed = ((1u64 << (width - 1)) - 1) as f32;
        let max_unsigned = ((1u64 << width) - 1) as f64;
        out[component] = match kind {
            // SNORM: c / (2^(n-1) - 1), clamped to -1 (OpenGL 4.2+ rule).
            1 => (signed(value) as f32 / max_signed).max(-1.0).to_bits(),
            2 => ((f64::from(value) / max_unsigned) as f32).to_bits(),
            3 => signed(value) as u32,
            4 => value,
            5 => (value as f32).to_bits(),
            6 => (signed(value) as f32).to_bits(),
            7 => match width {
                32 => value,
                16 => f16_to_f32(value as u16).to_bits(),
                _ => {
                    return Err(Unmodeled(format!("3D: vertex attribute {width}-bit float")));
                }
            },
            _ => {
                return Err(Unmodeled(format!(
                    "3D: vertex attribute numerical type {kind}"
                )));
            }
        };
    }
    if raw & (1 << 31) != 0 {
        out.swap(0, 2);
    }
    Ok(out)
}

/// IEEE half to single precision.
pub fn f16_to_f32(half: u16) -> f32 {
    let sign = u32::from(half >> 15) << 31;
    let exponent = u32::from((half >> 10) & 0x1F);
    let mantissa = u32::from(half & 0x3FF);
    let bits = match (exponent, mantissa) {
        (0, 0) => sign,
        (0, _) => {
            // Subnormal: normalize.
            let shift = mantissa.leading_zeros() - 21;
            let mantissa = (mantissa << shift) & 0x3FF;
            sign | ((113 - shift) << 23) | (mantissa << 13)
        }
        (0x1F, _) => sign | 0x7F80_0000 | (mantissa << 13),
        _ => sign | ((exponent + 112) << 23) | (mantissa << 13),
    };
    f32::from_bits(bits)
}

/// Vertex fetch: what the vertex shader's inputs are made of.
struct VertexFetch {
    /// (attribute, raw `SET_VERTEX_ATTRIBUTE_A`, stream) for each input the
    /// shader reads.
    attributes: Vec<(usize, u32, usize)>,
    streams: Vec<Stream>,
}

#[derive(Clone, Copy, Default)]
struct Stream {
    address: u64,
    limit: u64,
    stride: u32,
    /// Instance divisor, or 0 for per-vertex streams.
    divisor: u32,
}

impl VertexFetch {
    fn new(r: Registers<'_>, vertex: &Program) -> Result<VertexFetch, Unmodeled> {
        let mut attributes = Vec::new();
        for attribute in 0..32 {
            let word = 0x80 / 4 + attribute * 4;
            if !(0..4).any(|c| vertex.header().vtg_input_used(word + c)) {
                continue;
            }
            let raw = r.get(SET_VERTEX_ATTRIBUTE_A + attribute as u32 * 4);
            if raw & (1 << 6) != 0 {
                // SOURCE INACTIVE: the default value applies, which this
                // model does not establish.
                return Err(refuse(
                    "inactive vertex attribute read by the shader",
                    SET_VERTEX_ATTRIBUTE_A + attribute as u32 * 4,
                    raw,
                ));
            }
            attributes.push((attribute, raw, (raw & 0x1F) as usize));
        }
        let mut streams = vec![Stream::default(); 32];
        for &(_, _, index) in &attributes {
            let base = SET_VERTEX_STREAM_A_FORMAT + index as u32 * 16;
            let format = r.get(base);
            if format & (1 << 12) == 0 {
                return Err(refuse(
                    "vertex attribute on a disabled stream",
                    base,
                    format,
                ));
            }
            let instanced = r.get(SET_VERTEX_STREAM_INSTANCE_A + index as u32 * 4) & 1 != 0;
            let limit_base = SET_VERTEX_STREAM_LIMIT_A_A + index as u32 * 8;
            streams[index] = Stream {
                address: r.address(base + 4, base + 8),
                limit: r.address(limit_base, limit_base + 4),
                stride: format & 0xFFF,
                divisor: if instanced { r.get(base + 0xC) } else { 0 },
            };
        }
        Ok(VertexFetch {
            attributes,
            streams,
        })
    }

    fn fetch(
        &self,
        memory: &dyn GpuMemory,
        index: u32,
        instance: u32,
        inputs: &mut Attributes,
    ) -> Result<(), Unmodeled> {
        for &(attribute, raw, stream) in &self.attributes {
            let stream = self.streams[stream];
            // Per-instance streams step every `divisor` instances.
            let element = instance.checked_div(stream.divisor).unwrap_or(index);
            let offset = (raw >> 7) & 0x3FFF;
            let address =
                stream.address + u64::from(element) * u64::from(stream.stride) + u64::from(offset);
            let widths = component_widths((raw >> 21) & 0x3F)
                .ok_or_else(|| Unmodeled(format!("3D: vertex attribute format {raw:#x}")))?;
            let size: u32 = widths.iter().sum::<u32>() / 8;
            // SET_VERTEX_STREAM_LIMIT is the last valid byte address.
            if address + u64::from(size) - 1 > stream.limit {
                return Err(Unmodeled(format!(
                    "3D: vertex fetch at {address:#x} beyond stream limit {:#x}",
                    stream.limit
                )));
            }
            let mut bytes = [0u8; 16];
            memory
                .read(address, &mut bytes[..size as usize])
                .ok_or_else(|| Unmodeled(format!("3D: vertex fetch at unmapped {address:#x}")))?;
            let words = fetch_attribute(raw, &bytes)?;
            let base = 0x80 / 4 + attribute * 4;
            inputs.0[base..base + 4].copy_from_slice(&words);
        }
        Ok(())
    }
}

/// A vertex after the vertex shader.
#[derive(Clone)]
struct Shaded {
    clip: [f32; 4],
    attributes: Arc<Attributes>,
}

/// A clipped vertex: clip position and the attribute words the fragment
/// shader reads, interpolated.
#[derive(Clone)]
struct ClipVertex {
    clip: [f32; 4],
    words: Vec<f32>,
    /// User clip distances of the active planes.
    distances: Vec<f32>,
}

/// A clip plane: `dot(coefficients, clip) + bias >= 0`, or user clip
/// distance `n >= 0`.
#[derive(Clone, Copy)]
enum ClipPlane {
    Homogeneous([f32; 4], f32),
    User(usize),
}

impl ClipPlane {
    fn distance(&self, v: &ClipVertex) -> f32 {
        match *self {
            ClipPlane::Homogeneous(plane, bias) => {
                (0..4).map(|i| plane[i] * v.clip[i]).sum::<f32>() + bias
            }
            ClipPlane::User(index) => v.distances[index],
        }
    }
}

/// Triangles of a batch, by vertex positions in the batch.
fn assemble(topology: Topology, elements: &[Element]) -> Vec<[usize; 3]> {
    let mut triangles = Vec::new();
    // Primitive restart splits the batch into independent runs.
    let mut start = 0;
    for end in
        (0..=elements.len()).filter(|&i| i == elements.len() || elements[i] == Element::Restart)
    {
        let run: Vec<usize> = (start..end).collect();
        start = end + 1;
        let n = run.len();
        match topology {
            Topology::Triangles => {
                for t in run.as_chunks::<3>().0 {
                    triangles.push(*t);
                }
            }
            // Odd triangles swap their first two vertices to keep the
            // winding (OpenGL 4.6, 10.1.6).
            Topology::TriangleStrip => {
                for i in 2..n {
                    triangles.push(if i % 2 == 0 {
                        [run[i - 2], run[i - 1], run[i]]
                    } else {
                        [run[i - 1], run[i - 2], run[i]]
                    });
                }
            }
            Topology::TriangleFan | Topology::Polygon => {
                for i in 2..n {
                    triangles.push([run[0], run[i - 1], run[i]]);
                }
            }
            Topology::Quads => {
                for q in run.as_chunks::<4>().0 {
                    triangles.push([q[0], q[1], q[2]]);
                    triangles.push([q[0], q[2], q[3]]);
                }
            }
            Topology::QuadStrip => {
                for i in (3..n).step_by(2) {
                    triangles.push([run[i - 3], run[i - 2], run[i]]);
                    triangles.push([run[i - 3], run[i], run[i - 1]]);
                }
            }
        }
    }
    triangles
}

/// Sutherland-Hodgman clipping of a polygon against `plane . clip >= 0`.
fn clip_polygon(polygon: Vec<ClipVertex>, plane: ClipPlane) -> Vec<ClipVertex> {
    let distance = |v: &ClipVertex| plane.distance(v);
    let mut out = Vec::with_capacity(polygon.len() + 2);
    for i in 0..polygon.len() {
        let a = &polygon[i];
        let b = &polygon[(i + 1) % polygon.len()];
        let da = distance(a);
        let db = distance(b);
        if da >= 0.0 {
            out.push(a.clone());
        }
        if (da >= 0.0) != (db >= 0.0) {
            let t = da / (da - db);
            let lerp = |x: f32, y: f32| x + (y - x) * t;
            out.push(ClipVertex {
                clip: std::array::from_fn(|k| lerp(a.clip[k], b.clip[k])),
                words: a
                    .words
                    .iter()
                    .zip(&b.words)
                    .map(|(&x, &y)| lerp(x, y))
                    .collect(),
                distances: a
                    .distances
                    .iter()
                    .zip(&b.distances)
                    .map(|(&x, &y)| lerp(x, y))
                    .collect(),
            });
        }
    }
    out
}

/// Runs a batch through the pipeline into the bound targets.
pub fn draw(context: &Context<'_>, caches: &mut Caches, batch: &Batch) -> Result<(), Unmodeled> {
    let r = context.registers;
    check_unmodeled_state(r)?;
    let vertex = program(context, caches, VERTEX_PIPELINE)?.ok_or_else(|| {
        refuse(
            "draw without a vertex shader",
            SET_PIPELINE_SHADER + PIPELINE_STRIDE,
            0,
        )
    })?;
    let fetch = VertexFetch::new(r, &vertex)?;
    if r.get(SET_RT_LAYER) & 0x1_0000 != 0 && vertex.header().vtg_output_used(0x64 / 4) {
        return Err(refuse(
            "render target layer written by the vertex shader",
            SET_RT_LAYER,
            r.get(SET_RT_LAYER),
        ));
    }
    let pixel = program(context, caches, PIXEL_PIPELINE)?;

    let pools = Pools {
        header_pool: r.address(SET_TEX_HEADER_POOL_A, SET_TEX_HEADER_POOL_B),
        header_maximum: r.get(SET_TEX_HEADER_POOL_C) & 0x3F_FFFF,
        sampler_pool: r.address(SET_TEX_SAMPLER_POOL_A, SET_TEX_SAMPLER_POOL_B),
        sampler_maximum: r.get(SET_TEX_SAMPLER_POOL_C) & 0xF_FFFF,
        sampler_from_header: r.get(SET_SAMPLER_BINDING) & 1 != 0,
    };
    let textures = DrawTextures::new(context.memory, pools, &caches.textures);

    // Vertex shading, once per distinct vertex of the batch.
    let vertex_buffers = constant_buffers(context, VERTEX_PIPELINE, &vertex)?;
    let vertex_unit = textures.unit();
    let vertex_env = env(&vertex_buffers, &vertex_unit, r, context.memory);
    let mut shaded: std::collections::HashMap<(u32, u32), Shaded> = Default::default();
    let mut vertices = Vec::with_capacity(batch.elements.len());
    for element in &batch.elements {
        let Element::Vertex { fetch: index, id } = *element else {
            vertices.push(None);
            continue;
        };
        if let Some(done) = shaded.get(&(index, id)) {
            vertices.push(Some(done.clone()));
            continue;
        }
        let mut inputs = Attributes([0; 256]);
        // Per-instance attributes count from the base instance [public:
        // eden `DrawManager` `base_instance`, OpenGL `baseinstance`].
        let fetch_instance = batch.instance.wrapping_add(batch.base_instance);
        fetch.fetch(context.memory, index, fetch_instance, &mut inputs)?;
        inputs.0[0x2F8 / 4] = batch.instance;
        inputs.0[0x2FC / 4] = id;
        let mut outputs = Attributes([0; 256]);
        vertex.run_vertex(&vertex_env, &inputs, &mut outputs)?;
        let clip = std::array::from_fn(|i| f32::from_bits(outputs.0[0x70 / 4 + i]));
        let done = Shaded {
            clip,
            attributes: Arc::new(outputs),
        };
        shaded.insert((index, id), done.clone());
        vertices.push(Some(done));
    }

    if r.get(SET_RASTER_ENABLE) & 1 == 0 {
        return Ok(());
    }
    let Some(pixel) = pixel else {
        return Err(refuse(
            "draw without a pixel shader",
            SET_PIPELINE_SHADER + PIXEL_PIPELINE * PIPELINE_STRIDE,
            0,
        ));
    };
    rasterize(context, &vertex, &pixel, &textures, batch, &vertices)
}

/// Attribute words the pixel shader interpolates, with their mode.
fn varyings(pixel: &Program) -> Vec<(usize, Interpolation)> {
    (0x80 / 4..0x280 / 4)
        .filter_map(|word| match pixel.header().ps_interpolation(word) {
            Interpolation::Unused => None,
            mode => Some((word, mode)),
        })
        .collect()
}

fn rasterize(
    context: &Context<'_>,
    vertex: &Program,
    pixel: &Arc<Program>,
    textures: &DrawTextures<'_>,
    batch: &Batch,
    vertices: &[Option<Shaded>],
) -> Result<(), Unmodeled> {
    let r = context.registers;
    let varyings = varyings(pixel);
    for &(word, _) in &varyings {
        if !(0x80 / 4..0x280 / 4).contains(&word) {
            return Err(Unmodeled(format!(
                "3D: pixel shader reads attribute {:#x}",
                word * 4
            )));
        }
    }

    // Outputs and targets.
    let select = r.get(SET_CT_SELECT);
    let count = (select & 0xF).min(8);
    let mut windows = Vec::new();
    let mut colors = Vec::new();
    let mut clip = surface_clip(r);
    if let Some(scissor) = scissor(r) {
        clip = clip.intersect(&scissor);
    }
    let per_target = r.enabled(SET_BLEND_STATE_PER_TARGET);
    let single_write = r.enabled(SET_SINGLE_CT_WRITE_CONTROL);
    let srgb_write = r.enabled(SET_SRGB_WRITE);
    let mut targets = Vec::new();
    for output in 0..count {
        let target = (select >> (4 + 3 * output)) & 7;
        let mask = pixel.header().ps_color_mask(output as usize);
        let Some((surface, format)) = color_target(r, target)? else {
            continue;
        };
        let write_raw = r.get(SET_CT_WRITE + 4 * if single_write { 0 } else { target });
        let write = [0, 4, 8, 12].map(|bit| write_raw & (1 << bit) != 0);
        if mask == 0 || !write.iter().any(|&w| w) {
            continue;
        }
        let blend = if r.get(SET_BLEND + 4 * target) & 1 != 0 {
            Some(blend_state(r, target, per_target)?)
        } else {
            None
        };
        targets.push((surface, output));
        colors.push(ColorOutput {
            output: output as usize,
            window: colors.len(),
            format,
            blend,
            write,
            srgb: srgb_write && format.is_srgb(),
        });
    }
    let blend_constant = std::array::from_fn(|i| r.float(SET_BLEND_CONST_RED + 4 * i as u32));
    let depth = if r.enabled(SET_DEPTH_TEST) {
        let Some((surface, format)) = depth_target(r)? else {
            return Err(refuse(
                "depth test without a depth target",
                SET_ZT_SELECT,
                r.get(SET_ZT_SELECT),
            ));
        };
        let control = r.get(SET_VIEWPORT_CLIP_CONTROL);
        // PIXEL_MIN_Z / PIXEL_MAX_Z: CLAMP (1) only.
        if control & 0x18 != 0x18 {
            return Err(refuse(
                "per-pixel depth clipping",
                SET_VIEWPORT_CLIP_CONTROL,
                control,
            ));
        }
        let clamp = if control & 1 != 0 {
            (0.0, 1.0)
        } else {
            (
                r.float(SET_VIEWPORT_CLIP_MIN_Z),
                r.float(SET_VIEWPORT_CLIP_MAX_Z),
            )
        };
        Some((
            surface,
            DepthOutput {
                format,
                compare: Compare::from_raw(r.get(SET_DEPTH_FUNC))?,
                write: r.enabled(SET_DEPTH_WRITE),
                clamp,
            },
        ))
    } else {
        None
    };

    // Clipping, viewport transform and triangle setup.
    let triangles = setup(r, batch, vertices, &varyings, vertex)?;
    if triangles.is_empty() || (colors.is_empty() && depth.is_none()) {
        return Ok(());
    }
    let mut bounds = triangles
        .iter()
        .fold(None, |acc: Option<Rect>, t| {
            let b = t.bounds();
            Some(match acc {
                None => b,
                Some(a) => Rect {
                    x0: a.x0.min(b.x0),
                    y0: a.y0.min(b.y0),
                    x1: a.x1.max(b.x1),
                    y1: a.y1.max(b.y1),
                },
            })
        })
        .unwrap_or(clip);
    bounds = bounds.intersect(&clip);
    for (surface, _) in &targets {
        bounds = bounds.intersect(&Rect {
            x0: 0,
            y0: 0,
            x1: surface.width,
            y1: surface.height,
        });
    }
    if let Some((surface, _)) = &depth {
        bounds = bounds.intersect(&Rect {
            x0: 0,
            y0: 0,
            x1: surface.width,
            y1: surface.height,
        });
    }
    if bounds.is_empty() {
        return Ok(());
    }
    let pixel_buffers = constant_buffers(context, PIXEL_PIPELINE, pixel)?;
    if let Some(metal) = context.metal {
        if depth.is_some() {
            metal.note_fallback("depth test");
        } else {
            let surfaces: Vec<Surface> = targets.iter().map(|(surface, _)| *surface).collect();
            let draw = super::metal::Draw {
                program: pixel,
                triangles: &triangles,
                colors: &colors,
                surfaces: &surfaces,
                bounds,
                blend_constant,
                constants: &pixel_buffers,
                textures,
                texture_buffer_index: r.get(SET_BINDLESS_TEXTURE) & 0x1F,
            };
            if metal.draw(context.memory, &draw)? {
                return Ok(());
            }
        }
        // The software renderer reads memory from several threads; all
        // of Metal's work is made visible before it starts.
        metal.flush(context.memory)?;
    }
    for (surface, _) in &targets {
        windows.push(Window::load(context.memory, *surface, bounds)?);
    }
    let mut depth_window = match &depth {
        Some((surface, _)) => Some(Window::load(context.memory, *surface, bounds)?),
        None => None,
    };

    let backend = Backend {
        program: pixel,
        colors: &colors,
        depth: depth.map(|(_, d)| d),
        blend_constant,
        // SET_BLEND_FLOAT_OPTION (0x19c0) ZERO_TIMES_ANYTHING_IS_ZERO.
        blend_zero_rule: r.get(0x19c0) & 1 != 0,
        clip: bounds,
    };

    // Bands of all windows, grouped by row.
    let mut color_bands: Vec<_> = windows.iter_mut().map(|w| w.bands_mut()).collect();
    let mut depth_bands = depth_window.as_mut().map(|w| w.bands_mut());
    let mut groups = Vec::new();
    loop {
        let colors: Vec<_> = color_bands.iter_mut().filter_map(|b| b.next()).collect();
        let depth = depth_bands.as_mut().and_then(|b| b.next());
        if colors.len() < color_bands.len() || (colors.is_empty() && depth.is_none()) {
            break;
        }
        groups.push(BandSet { colors, depth });
    }

    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(groups.len().max(1));
    let pixels = u64::from(bounds.x1 - bounds.x0) * u64::from(bounds.y1 - bounds.y0);
    let run = |groups: &mut [BandSet<'_>]| -> Result<(), Unmodeled> {
        let unit = textures.unit();
        let env = env(&pixel_buffers, &unit, r, context.memory);
        for group in groups.iter_mut() {
            raster::rasterize(&backend, &env, &triangles, group)?;
        }
        Ok(())
    };
    // Small draws run on the calling thread; spawning costs more than it
    // saves below a few thousand pixels.
    if threads <= 1 || pixels < 16 * 1024 {
        run(&mut groups)?;
    } else {
        let chunk = groups.len().div_ceil(threads * 4).max(1);
        let work = std::sync::Mutex::new(groups.chunks_mut(chunk));
        let results: Vec<Result<(), Unmodeled>> = std::thread::scope(|scope| {
            let (tx, rx) = std::sync::mpsc::channel();
            for _ in 0..threads {
                let tx = tx.clone();
                let work = &work;
                let run = &run;
                scope.spawn(move || {
                    loop {
                        let Some(slice) = work.lock().unwrap().next() else {
                            break;
                        };
                        if let Err(error) = run(slice) {
                            let _ = tx.send(Err(error));
                            break;
                        }
                    }
                });
            }
            drop(tx);
            rx.into_iter().collect()
        });
        if let Some(error) = results.into_iter().find_map(|r| r.err()) {
            return Err(error);
        }
    }
    drop(groups);
    drop(color_bands);
    drop(depth_bands);
    for window in &mut windows {
        window.store(context.memory)?;
    }
    if let Some(window) = &mut depth_window {
        window.store(context.memory)?;
    }
    Ok(())
}

/// The blend state of render target `target`.
fn blend_state(r: Registers<'_>, target: u32, per_target: bool) -> Result<Blend, Unmodeled> {
    // Per target: SEPARATE_FOR_ALPHA, COLOR_OP, COLOR_SOURCE, COLOR_DEST,
    // ALPHA_OP, ALPHA_SOURCE, ALPHA_DEST at +0..+0x18. Common:
    // SEPARATE_FOR_ALPHA 0x133c, COLOR_OP 0x1340, COLOR_SOURCE 0x1344,
    // COLOR_DEST 0x1348, ALPHA_OP 0x134c, ALPHA_SOURCE 0x1350,
    // GLOBAL_COLOR_KEY 0x1354, ALPHA_DEST 0x1358.
    let (separate, color, alpha_op, alpha_source, alpha_destination) = if per_target {
        let base = SET_BLEND_PER_TARGET + 32 * target;
        (r.get(base), base + 4, base + 0x10, base + 0x14, base + 0x18)
    } else {
        (
            r.get(SET_BLEND_SEPARATE_FOR_ALPHA),
            SET_BLEND_COLOR_OP,
            0x134c,
            0x1350,
            0x1358,
        )
    };
    if !per_target && r.get(0x1354) & 1 != 0 {
        return Err(refuse("global color key", 0x1354, r.get(0x1354)));
    }
    let color_op = Operation::from_raw(r.get(color))?;
    let color_source = Factor::from_raw(r.get(color + 4))?;
    let color_destination = Factor::from_raw(r.get(color + 8))?;
    let blend = if separate & 1 != 0 {
        Blend {
            color_op,
            color_source,
            color_destination,
            alpha_op: Operation::from_raw(r.get(alpha_op))?,
            alpha_source: Factor::from_raw(r.get(alpha_source))?,
            alpha_destination: Factor::from_raw(r.get(alpha_destination))?,
        }
    } else {
        Blend {
            color_op,
            color_source,
            color_destination,
            alpha_op: color_op,
            alpha_source: color_source,
            alpha_destination: color_destination,
        }
    };
    if blend.dual_source() && target != 0 {
        return Err(refuse(
            "dual-source blending on a target other than 0",
            SET_BLEND + 4 * target,
            1,
        ));
    }
    Ok(blend)
}

/// Clips, projects, culls and sets up the triangles of a batch.
fn setup(
    r: Registers<'_>,
    batch: &Batch,
    vertices: &[Option<Shaded>],
    varyings: &[(usize, Interpolation)],
    vertex: &Program,
) -> Result<Vec<Triangle>, Unmodeled> {
    let control = r.get(SET_VIEWPORT_CLIP_CONTROL);
    // GEOMETRY_CLIP: FRUSTUM_XY_CLIP (2), FRUSTUM_XYZ_CLIP (3) or
    // FRUSTUM_Z_CLIP (5, the netlist's initial value). Without x/y
    // clipping the rasterizer's guard band applies: primitives are cut only
    // by the scissor and surface clip.
    let mode = (control >> 11) & 7;
    let (clip_xy, clip_z) = match mode {
        2 => (true, false),
        3 => (true, true),
        5 => (false, true),
        _ => {
            return Err(refuse(
                "geometry clip mode",
                SET_VIEWPORT_CLIP_CONTROL,
                control,
            ));
        }
    };
    // SET_VIEWPORT_Z_CLIP RANGE: -w..w (0) or 0..w (1).
    let zero_to_w = r.get(SET_VIEWPORT_Z_CLIP) & 1 != 0;
    let mut planes: Vec<ClipPlane> = Vec::new();
    let homogeneous = |plane: [f32; 4]| ClipPlane::Homogeneous(plane, 0.0);
    if clip_z {
        planes.push(homogeneous(if zero_to_w {
            [0.0, 0.0, 1.0, 0.0]
        } else {
            [0.0, 0.0, 1.0, 1.0]
        }));
        planes.push(homogeneous([0.0, 0.0, -1.0, 1.0]));
    }
    if clip_xy {
        planes.extend(
            [
                [1.0, 0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, -1.0, 0.0, 1.0],
            ]
            .map(homogeneous),
        );
    }
    // User clip planes (SET_USER_CLIP_ENABLE bit i) whose clip distance
    // a[0x2c0 + 4i] the vertex shader writes [public: Ryujinx
    // `AttributeConsts.ClipDistance0`]. A plane whose distance is not
    // written has nothing to clip against; it is left out.
    let user_enable = r.get(SET_USER_CLIP_ENABLE) & 0xFF;
    let user: Vec<usize> = (0..8)
        .filter(|&i| user_enable & (1 << i) != 0 && vertex.header().vtg_output_used(0x2C0 / 4 + i))
        .collect();
    planes.extend((0..user.len()).map(ClipPlane::User));
    // Guard band: keeps snapped coordinates far inside i64 range; it cuts
    // nothing that could reach a surface (at most 65536 pixels wide).
    const GUARD: f32 = 1048576.0;
    let scale = [0, 4, 8].map(|o| r.float(SET_VIEWPORT_SCALE_X + o));
    let offset = [0xc, 0x10, 0x14].map(|o| r.float(SET_VIEWPORT_SCALE_X + o));
    let guard_x = GUARD / scale[0].abs().max(1.0);
    let guard_y = GUARD / scale[1].abs().max(1.0);
    planes.extend(
        [
            [1.0, 0.0, 0.0, guard_x],
            [-1.0, 0.0, 0.0, guard_x],
            [0.0, 1.0, 0.0, guard_y],
            [0.0, -1.0, 0.0, guard_y],
        ]
        .map(homogeneous),
    );
    // w above a small positive bound, so the perspective divide is finite.
    planes.push(ClipPlane::Homogeneous([0.0, 0.0, 0.0, 1.0], -1e-6));

    let cull = r.enabled(OGL_SET_CULL);
    let cull_face = r.get(OGL_SET_CULL_FACE);
    let front_ccw = match r.get(OGL_SET_FRONT_FACE) {
        0x900 => false,
        0x901 => true,
        other => return Err(refuse("front face", OGL_SET_FRONT_FACE, other)),
    };
    // SET_WINDOW_ORIGIN FLIP_Y flips the triangle orientation [public:
    // Ryujinx `StateUpdater.UpdateFrontFace`: front face inverted unless
    // `TriangleRastFlip`].
    let flip = r.get(SET_WINDOW_ORIGIN) & 0x10 != 0;
    let provoking_last = r.get(SET_PROVOKING_VERTEX) & 1 != 0;

    let mut triangles = Vec::new();
    for indices in assemble(batch.topology, &batch.elements) {
        let shaded: Vec<&Shaded> = indices
            .iter()
            .map(|&i| vertices[i].as_ref().expect("assembly skips restarts"))
            .collect();
        let provoking = if provoking_last { shaded[2] } else { shaded[0] };
        let polygon: Vec<ClipVertex> = shaded
            .iter()
            .map(|v| ClipVertex {
                clip: v.clip,
                words: varyings
                    .iter()
                    .map(|&(word, _)| f32::from_bits(v.attributes.0[word]))
                    .collect(),
                distances: user
                    .iter()
                    .map(|&i| f32::from_bits(v.attributes.0[0x2C0 / 4 + i]))
                    .collect(),
            })
            .collect();
        let mut polygon = polygon;
        for &plane in &planes {
            if polygon.len() < 3 {
                break;
            }
            if polygon.iter().all(|v| plane.distance(v) >= 0.0) {
                continue;
            }
            polygon = clip_polygon(polygon, plane);
        }
        if polygon.len() < 3 {
            continue;
        }
        // Window coordinates.
        let window: Vec<[f32; 4]> = polygon
            .iter()
            .map(|v| {
                let inv_w = 1.0 / v.clip[3];
                [
                    v.clip[0] * inv_w * scale[0] + offset[0],
                    v.clip[1] * inv_w * scale[1] + offset[1],
                    v.clip[2] * inv_w * scale[2] + offset[2],
                    inv_w,
                ]
            })
            .collect();
        // Facing from the signed area of the whole polygon, in window
        // coordinates with y down.
        let area: f32 = (0..window.len())
            .map(|i| {
                let a = window[i];
                let b = window[(i + 1) % window.len()];
                a[0] * b[1] - b[0] * a[1]
            })
            .sum();
        // With y down, a positive area is clockwise on screen. OpenGL
        // window coordinates have y up, where the same triangle is
        // counter-clockwise; the orientation flag inverts that.
        let ccw_in_gl = (area > 0.0) != flip;
        let front_facing = ccw_in_gl == front_ccw;
        if cull {
            let culled = match cull_face {
                0x404 => front_facing,
                0x405 => !front_facing,
                0x408 => true,
                other => return Err(refuse("cull face", OGL_SET_CULL_FACE, other)),
            };
            if culled {
                continue;
            }
        }
        for i in 1..window.len() - 1 {
            let corners = [0, i, i + 1];
            let points = corners.map(|k| [window[k][0], window[k][1]]);
            let mut planes_out = Box::new([Plane::default(); 256]);
            for (slot, &(word, mode)) in varyings.iter().enumerate() {
                let values = corners.map(|k| match mode {
                    Interpolation::Perspective => polygon[k].words[slot] * window[k][3],
                    _ => polygon[k].words[slot],
                });
                planes_out[word] = match mode {
                    Interpolation::Flat => Plane {
                        dx: 0.0,
                        dy: 0.0,
                        base: f32::from_bits(provoking.attributes.0[word]),
                    },
                    _ => raster::plane(points, values),
                };
            }
            // Fragment coordinates: x, y at the pixel center, z, 1/w.
            planes_out[0x70 / 4] = Plane {
                dx: 1.0,
                dy: 0.0,
                base: 0.0,
            };
            planes_out[0x74 / 4] = Plane {
                dx: 0.0,
                dy: 1.0,
                base: 0.0,
            };
            let depth = raster::plane(points, corners.map(|k| window[k][2]));
            planes_out[0x78 / 4] = depth;
            planes_out[0x7C / 4] = raster::plane(points, corners.map(|k| window[k][3]));
            if let Some(triangle) = Triangle::new(points, planes_out, depth, front_facing) {
                triangles.push(triangle);
            }
        }
    }
    Ok(triangles)
}

#[cfg(test)]
mod tests {
    use super::super::tests::Ram;

    /// A buffer bound past the end of memory yields the pages that exist,
    /// split at page boundaries relative to the GPU address.
    #[test]
    fn constant_buffer_snapshot_stops_at_unmapped_page() {
        let ram = Ram::new(0x3000);
        let bytes = super::read_mapped(&ram, 0x1700, 0xE900);
        assert_eq!(bytes.len(), 0x1900);
        assert!(super::read_mapped(&ram, 0x3000, 0x100).is_empty());
        assert_eq!(super::read_mapped(&ram, 0x1700, 0x80).len(), 0x80);
    }
}
