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

use serde::{Deserialize, Serialize};

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

/// Primitive topologies (`BEGIN_OP`). The point and line variants come
/// last so that snapshots taken before they existed name the same
/// variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Topology {
    Triangles,
    TriangleStrip,
    TriangleFan,
    Quads,
    QuadStrip,
    Polygon,
    Points,
    Lines,
    LineLoop,
    LineStrip,
}

impl Topology {
    pub fn from_raw(raw: u32) -> Result<Self, Unmodeled> {
        Ok(match raw {
            0 => Topology::Points,
            1 => Topology::Lines,
            2 => Topology::LineLoop,
            3 => Topology::LineStrip,
            4 => Topology::Triangles,
            5 => Topology::TriangleStrip,
            6 => Topology::TriangleFan,
            7 => Topology::Quads,
            8 => Topology::QuadStrip,
            9 => Topology::Polygon,
            0xA..=0xD => {
                return Err(Unmodeled(format!(
                    "3D: primitive topology {raw:#x} (adjacency topologies are not modeled)"
                )));
            }
            0xE => {
                return Err(Unmodeled(
                    "3D: primitive topology 0xe (patches are not modeled)".into(),
                ));
            }
            _ => return Err(Unmodeled(format!("3D: primitive topology {raw:#x}"))),
        })
    }
}

/// One vertex of a draw: where to fetch it from, or a primitive restart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
    /// What the channel's engines wrote since it last signaled.
    pub writes: &'a super::resources::Writes,
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
const SET_ALPHA_REF: u32 = 0x1310;
const SET_ALPHA_FUNC: u32 = 0x1314;
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
const SET_ALIASED_LINE_WIDTH_ENABLE: u32 = 0x020c;
const SET_SNAP_GRID_LINE: u32 = 0x0318;
const SET_LINE_WIDTH_FLOAT: u32 = 0x13b0;
const SET_ALIASED_LINE_WIDTH_FLOAT: u32 = 0x13b4;
const SET_LINE_MULTISAMPLE_OVERRIDE: u32 = 0x1418;
const SET_POINT_SIZE: u32 = 0x1518;
const SET_POINT_SPRITE: u32 = 0x1520;
const SET_ANTI_ALIAS_ENABLE: u32 = 0x1534;
const SET_ANTI_ALIASED_LINE: u32 = 0x1570;
const SET_POINT_SPRITE_SELECT: u32 = 0x1604;
const SET_ANTI_ALIASED_POINT: u32 = 0x1658;
const SET_POINT_CENTER_MODE: u32 = 0x165c;
const SET_LINE_STIPPLE: u32 = 0x166c;
const SET_ATTRIBUTE_POINT_SIZE: u32 = 0x1910;

/// Pipeline stages (`SET_PIPELINE_*(j)`): 0 vertex A (cull before fetch),
/// 1 vertex B, 2 tessellation init, 3 tessellation, 4 geometry, 5 pixel.
const PIPELINE_STRIDE: u32 = 64;
const VERTEX_PIPELINE: u32 = 1;
const GEOMETRY_PIPELINE: u32 = 4;
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
    for pipeline in [0, 2, 3] {
        let offset = SET_PIPELINE_SHADER + pipeline * PIPELINE_STRIDE;
        require(
            r,
            "shader stage other than vertex B, geometry and pixel",
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
pub(super) fn read_mapped(memory: &dyn GpuMemory, address: u64, length: usize) -> Vec<u8> {
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
/// component sizes in bits, in memory order: component 0 in the lowest
/// bits of the little-endian attribute.
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
        // A2B10G10R10: one little-endian word, R in bits 9:0, G 19:10, B
        // 29:20, A 31:30, named most significant first as A8B8G8R8 is
        // [public: open-gpu-doc `clb197.h`; the OpenGL
        // `GL_INT_2_10_10_10_REV` / Direct3D `DXGI_FORMAT_R10G10B10A2`
        // packing it serves]. The Settings Mii editor fetches SNORM ones.
        0x30 => &[10, 10, 10, 2],
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
    let mut packed = [0u8; 16];
    let length = bytes.len().min(16);
    packed[..length].copy_from_slice(&bytes[..length]);
    let packed = u128::from_le_bytes(packed);
    let mut shift = 0;
    for (component, &width) in widths.iter().enumerate() {
        let value = ((packed >> shift) as u32) & (u32::MAX >> (32 - width));
        shift += width;
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
///
/// An attribute whose `SET_VERTEX_ATTRIBUTE_A` SOURCE is INACTIVE is not
/// fetched: the hardware supplies a constant for it, the per-attribute
/// value of the immediate-mode methods (envytools `gf100_3d.xml`:
/// `VERTEX_ATTRIB_FORMAT` bit 6 `CONST`, values from `VTX_ATTR_DEFINE`
/// 0x114c / `VTX_ATTR_DATA` 0x1150). NVN writes neither \[inference: the
/// Settings Mii editor's channel never calls them, nor does its
/// method-init list set them, in a capture of the session\], and what
/// they hold then is not established. So such an attribute is marked
/// ([`VertexFetch::inactive`]) and only a read the vertex program
/// executes refuses. NVN marks every attribute its vertex state leaves
/// unused inactive (`0x3a400040`, R32 float), and uber-shaders name
/// attributes in their header (`vtg_input_used`) that branches on
/// constant buffer values skip: the Mii editor's vertex program reads
/// attribute 2 (a skinning index) only when its constant buffer asks for
/// skinning.
struct VertexFetch {
    /// (attribute, raw `SET_VERTEX_ATTRIBUTE_A`, stream) for each active
    /// input the shader names.
    attributes: Vec<(usize, u32, usize)>,
    /// Bit `a`: the shader names attribute `a`, which is inactive.
    inactive: u32,
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
        let mut inactive = 0;
        for attribute in 0..32 {
            let word = 0x80 / 4 + attribute * 4;
            if !(0..4).any(|c| vertex.header().vtg_input_used(word + c)) {
                continue;
            }
            let raw = r.get(SET_VERTEX_ATTRIBUTE_A + attribute as u32 * 4);
            if raw & (1 << 6) != 0 {
                inactive |= 1 << attribute;
                continue;
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
            inactive,
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

/// The primitives of a batch, by vertex positions in the batch.
#[derive(Debug, PartialEq, Eq)]
enum Primitives {
    Points(Vec<usize>),
    /// Segments from their first to their second vertex.
    Lines(Vec<[usize; 2]>),
    Triangles(Vec<[usize; 3]>),
}

/// Primitive assembly [public: OpenGL 4.6 core, 10.1 "Primitive Types"].
fn assemble(topology: Topology, elements: &[Element]) -> Primitives {
    let mut points = Vec::new();
    let mut lines = Vec::new();
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
            Topology::Points => points.extend_from_slice(&run),
            Topology::Lines => lines.extend_from_slice(run.as_chunks::<2>().0),
            Topology::LineStrip => lines.extend(run.windows(2).map(|s| [s[0], s[1]])),
            // A loop closes with a segment from its last vertex to its
            // first; a restart ends (and closes) the loop.
            Topology::LineLoop => {
                lines.extend(run.windows(2).map(|s| [s[0], s[1]]));
                if n >= 2 {
                    lines.push([run[n - 1], run[0]]);
                }
            }
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
    match topology {
        Topology::Points => Primitives::Points(points),
        Topology::Lines | Topology::LineStrip | Topology::LineLoop => Primitives::Lines(lines),
        _ => Primitives::Triangles(triangles),
    }
}

impl ClipVertex {
    /// The point a fraction `t` of the way from `a` to `b`.
    fn lerp(a: &ClipVertex, b: &ClipVertex, t: f32) -> ClipVertex {
        let lerp = |x: f32, y: f32| x + (y - x) * t;
        ClipVertex {
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
        }
    }
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
            out.push(ClipVertex::lerp(a, b, da / (da - db)));
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
    let geometry = program(context, caches, GEOMETRY_PIPELINE)?;
    // The last stage before the rasterizer writes the layer.
    let last_header = geometry.as_ref().unwrap_or(&vertex).header();
    if r.get(SET_RT_LAYER) & 0x1_0000 != 0 && last_header.vtg_output_used(0x64 / 4) {
        return Err(refuse(
            "render target layer written by the last vertex-pipeline shader",
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
    let textures = DrawTextures::new(context.memory, pools, &caches.textures, context.writes);

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
        vertex.run_vertex(&vertex_env, &inputs, fetch.inactive, &mut outputs)?;
        let clip = std::array::from_fn(|i| f32::from_bits(outputs.0[0x70 / 4 + i]));
        let done = Shaded {
            clip,
            attributes: Arc::new(outputs),
        };
        shaded.insert((index, id), done.clone());
        vertices.push(Some(done));
    }

    let geometry_output;
    let (batch, vertices, last) = match &geometry {
        Some(program) => {
            geometry_output = geometry_stage(context, program, &textures, batch, &vertices)?;
            (&geometry_output.0, &geometry_output.1[..], &**program)
        }
        None => (batch, &vertices[..], &*vertex),
    };

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
    rasterize(context, last, &pixel, &textures, batch, vertices)
}

/// The geometry shader stage: one thread per input primitive of the
/// shaded batch, the vertices of their output strips (or points) gathered
/// into a new batch of the header's output topology (a restart after each
/// thread, and at each `OUT.CUT`), which clipping and rasterization take
/// like a draw's own.
fn geometry_stage(
    context: &Context<'_>,
    geometry: &Program,
    textures: &DrawTextures<'_>,
    batch: &Batch,
    vertices: &[Option<Shaded>],
) -> Result<(Batch, Vec<Option<Shaded>>), Unmodeled> {
    let r = context.registers;
    let header = geometry.header();
    let control = SET_PIPELINE_SHADER + GEOMETRY_PIPELINE * PIPELINE_STRIDE;
    // Input primitives: the triangles of lists, strips and fans, with
    // their vertices in the order `assemble` gives (odd strip triangles
    // swap their first two vertices) [public: OpenGL 4.6, 11.3.1]. OpenGL
    // has no geometry shader input for quads and polygons; what the
    // hardware does with them is not established.
    if !matches!(
        batch.topology,
        Topology::Triangles | Topology::TriangleStrip | Topology::TriangleFan
    ) {
        return Err(Unmodeled(format!(
            "3D: geometry shader with {:?} input",
            batch.topology
        )));
    }
    // CommonWord3.OutputTopology: POINTLIST (1), LINESTRIP (6),
    // TRIANGLESTRIP (7) [public: NVIDIA open-gpu-doc Shader-Program-Header].
    let output = match header.output_topology() {
        1 => Topology::Points,
        6 => Topology::LineStrip,
        7 => Topology::TriangleStrip,
        other => {
            return Err(Unmodeled(format!(
                "3D: geometry shader output topology {other}"
            )));
        }
    };
    // Several threads per primitive tell themselves apart with
    // SR_INVOCATION_ID, which is not modeled.
    let threads = header.threads_per_input_primitive();
    if threads != 1 {
        return Err(refuse(
            "geometry shader threads per input primitive",
            control,
            threads,
        ));
    }
    let buffers = constant_buffers(context, GEOMETRY_PIPELINE, geometry)?;
    let unit = textures.unit();
    let env = env(&buffers, &unit, r, context.memory);
    let mut emitted = Vec::new();
    let Primitives::Triangles(primitives) = assemble(batch.topology, &batch.elements) else {
        unreachable!("the input topology is checked to be triangles");
    };
    for indices in primitives {
        let inputs = indices.map(|i| {
            &*vertices[i]
                .as_ref()
                .expect("assembly skips restarts")
                .attributes
        });
        geometry.run_geometry(&env, &inputs, &mut emitted)?;
        emitted.push(None);
    }
    let mut elements = Vec::with_capacity(emitted.len());
    let mut shaded = Vec::with_capacity(emitted.len());
    for vertex in emitted {
        match vertex {
            Some(attributes) => {
                let index = shaded.len() as u32;
                elements.push(Element::Vertex {
                    fetch: index,
                    id: index,
                });
                shaded.push(Some(Shaded {
                    clip: std::array::from_fn(|i| f32::from_bits(attributes.0[0x70 / 4 + i])),
                    attributes: Arc::new(attributes),
                }));
            }
            None => {
                elements.push(Element::Restart);
                shaded.push(None);
            }
        }
    }
    let batch = Batch {
        topology: output,
        instance: batch.instance,
        base_instance: batch.base_instance,
        elements,
    };
    Ok((batch, shaded))
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
    // Without a depth target (SET_ZT_SELECT TARGET_COUNT 0) the depth
    // test and depth write are bypassed and every fragment passes: the rule
    // of the APIs NVN implements [public: OpenGL 4.6 core, 17.3.4 Depth
    // Buffer Test (with no depth buffer the test behaves as if it always
    // passes and nothing is written); Vulkan, Fragment Operations, Depth
    // Test (no depth attachment: the test is skipped)]. eden and Ryujinx
    // reach the same result by binding no depth attachment and leaving the
    // test enabled (eden `fixed_pipeline_state.cpp`: `depth_enabled` from
    // `zeta_enable`, `depth_test_enable` passed through). NVIDIA documents
    // nothing for the combination [assumption: the hardware follows the
    // APIs].
    let depth_target = if r.enabled(SET_DEPTH_TEST) {
        depth_target(r)?
    } else {
        None
    };
    let depth = if let Some((surface, format)) = depth_target {
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

    // SET_ALPHA_TEST, SET_ALPHA_REF (0x1310, a float), SET_ALPHA_FUNC
    // (0x1314, OpenGL or Direct3D comparison encodings) [public: NVIDIA
    // open-gpu-doc `clb197.h`]: the fragment's output 0 alpha, as the pixel
    // program leaves it, is compared with the reference and the fragment
    // is discarded where the comparison fails, before the depth test and
    // blending (the OpenGL compatibility-profile alpha test NVN exposes;
    // eden `emit_spirv_special.cpp` `AlphaTest` and Ryujinx
    // `EmitterContext.PrepareForReturn` emit it at the end of the pixel
    // program the same way). A program that does not write output 0 alpha
    // is tested against nothing: Ryujinx skips the test then, eden reads
    // an undefined output; it is refused unless the function is NEVER or
    // ALWAYS, which do not read it.
    let alpha_test = if r.enabled(SET_ALPHA_TEST) {
        let compare = Compare::from_raw(r.get(SET_ALPHA_FUNC))?;
        if !matches!(compare, Compare::Never | Compare::Always)
            && pixel.header().ps_color_mask(0) & 8 == 0
        {
            return Err(refuse(
                "alpha test of a pixel program that writes no output 0 alpha",
                SET_ALPHA_FUNC,
                r.get(SET_ALPHA_FUNC),
            ));
        }
        (compare != Compare::Always).then(|| (compare, r.float(SET_ALPHA_REF)))
    } else {
        None
    };
    // Clipping, viewport transform and primitive setup.
    let triangles = setup(r, batch, vertices, &varyings, vertex, pixel, clip)?;
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
                alpha_test,
            };
            if metal.draw(context.memory, context.writes, &draw)? {
                return Ok(());
            }
        }
        // The software renderer reads memory from several threads; all
        // of Metal's work is made visible before it starts.
        metal.flush(context.memory, context.writes)?;
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
        alpha_test,
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

/// Clips, projects and sets up the primitives of a batch as the triangles
/// the rasterizer covers: triangles as they are (after culling), line
/// segments and points as the quads their rules cover (see `line_rule`
/// and `point_rule`). `area` is the scissor and surface clip.
fn setup(
    r: Registers<'_>,
    batch: &Batch,
    vertices: &[Option<Shaded>],
    varyings: &[(usize, Interpolation)],
    last: &Program,
    pixel: &Program,
    area: Rect,
) -> Result<Vec<Triangle>, Unmodeled> {
    let clip = Clip::new(r, last)?;
    match assemble(batch.topology, &batch.elements) {
        Primitives::Triangles(list) => setup_triangles(r, &clip, &list, vertices, varyings),
        Primitives::Lines(list) => setup_lines(r, &clip, &list, vertices, varyings, area),
        Primitives::Points(list) => setup_points(r, &clip, &list, vertices, varyings, last, pixel),
    }
}

/// Clipping and the viewport transform of a draw.
struct Clip {
    planes: Vec<ClipPlane>,
    /// The user clip distances in use (`ClipPlane::User(i)` is the `i`th).
    user: Vec<usize>,
    scale: [f32; 3],
    offset: [f32; 3],
}

impl Clip {
    fn new(r: Registers<'_>, last: &Program) -> Result<Clip, Unmodeled> {
        let control = r.get(SET_VIEWPORT_CLIP_CONTROL);
        // GEOMETRY_CLIP: FRUSTUM_XY_CLIP (2), FRUSTUM_XYZ_CLIP (3) or
        // FRUSTUM_Z_CLIP (5, the netlist's initial value). Without x/y
        // clipping the rasterizer's guard band applies: primitives are cut
        // only by the scissor and surface clip.
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
        // a[0x2c0 + 4i] the last vertex-pipeline shader writes [public:
        // Ryujinx `AttributeConsts.ClipDistance0`]. A plane whose distance
        // is not written has nothing to clip against; it is left out.
        let user_enable = r.get(SET_USER_CLIP_ENABLE) & 0xFF;
        let user: Vec<usize> = (0..8)
            .filter(|&i| {
                user_enable & (1 << i) != 0 && last.header().vtg_output_used(0x2C0 / 4 + i)
            })
            .collect();
        planes.extend((0..user.len()).map(ClipPlane::User));
        // Guard band: keeps snapped coordinates far inside i64 range; it
        // cuts nothing that could reach a surface (at most 65536 pixels
        // wide).
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
        // w above a small positive bound, so the perspective divide is
        // finite.
        planes.push(ClipPlane::Homogeneous([0.0, 0.0, 0.0, 1.0], -1e-6));
        Ok(Clip {
            planes,
            user,
            scale,
            offset,
        })
    }

    /// A shaded vertex as clipping sees it.
    fn vertex(&self, v: &Shaded, varyings: &[(usize, Interpolation)]) -> ClipVertex {
        ClipVertex {
            clip: v.clip,
            words: varyings
                .iter()
                .map(|&(word, _)| f32::from_bits(v.attributes.0[word]))
                .collect(),
            distances: self
                .user
                .iter()
                .map(|&i| f32::from_bits(v.attributes.0[0x2C0 / 4 + i]))
                .collect(),
        }
    }

    /// Window coordinates: physical x, y, depth and 1/w.
    fn window(&self, v: &ClipVertex) -> [f32; 4] {
        let inv_w = 1.0 / v.clip[3];
        [
            v.clip[0] * inv_w * self.scale[0] + self.offset[0],
            v.clip[1] * inv_w * self.scale[1] + self.offset[1],
            v.clip[2] * inv_w * self.scale[2] + self.offset[2],
            inv_w,
        ]
    }

    /// The part of segment `a`-`b` inside every plane, in the same
    /// direction; `None` if no part is.
    fn segment(&self, mut a: ClipVertex, mut b: ClipVertex) -> Option<(ClipVertex, ClipVertex)> {
        for plane in &self.planes {
            let da = plane.distance(&a);
            let db = plane.distance(&b);
            match (da >= 0.0, db >= 0.0) {
                (true, true) => {}
                (false, false) => return None,
                (true, false) => b = ClipVertex::lerp(&a, &b, da / (da - db)),
                (false, true) => a = ClipVertex::lerp(&a, &b, da / (da - db)),
            }
        }
        Some((a, b))
    }
}

/// The interpolation planes of a primitive through three window
/// positions, each given with the clip vertex whose attribute words hold
/// there and that vertex's window coordinates (for depth and 1/w). Flat
/// inputs take the provoking vertex's words. Returns the planes and the
/// depth plane.
fn primitive_planes(
    varyings: &[(usize, Interpolation)],
    points: [[f32; 2]; 3],
    corners: [(&ClipVertex, [f32; 4]); 3],
    provoking: &Shaded,
) -> (Box<[Plane; 256]>, Plane) {
    let mut planes = Box::new([Plane::default(); 256]);
    for (slot, &(word, mode)) in varyings.iter().enumerate() {
        planes[word] = match mode {
            Interpolation::Flat => Plane {
                dx: 0.0,
                dy: 0.0,
                base: f32::from_bits(provoking.attributes.0[word]),
            },
            Interpolation::Perspective => {
                raster::plane(points, corners.map(|(v, w)| v.words[slot] * w[3]))
            }
            _ => raster::plane(points, corners.map(|(v, _)| v.words[slot])),
        };
    }
    // Fragment coordinates: x, y at the pixel center, z, 1/w.
    planes[0x70 / 4] = Plane {
        dx: 1.0,
        dy: 0.0,
        base: 0.0,
    };
    planes[0x74 / 4] = Plane {
        dx: 0.0,
        dy: 1.0,
        base: 0.0,
    };
    let depth = raster::plane(points, corners.map(|(_, w)| w[2]));
    planes[0x78 / 4] = depth;
    planes[0x7C / 4] = raster::plane(points, corners.map(|(_, w)| w[3]));
    (planes, depth)
}

/// Clips, projects, culls and sets up the triangles of a batch.
fn setup_triangles(
    r: Registers<'_>,
    clip: &Clip,
    list: &[[usize; 3]],
    vertices: &[Option<Shaded>],
    varyings: &[(usize, Interpolation)],
) -> Result<Vec<Triangle>, Unmodeled> {
    let cull = r.enabled(OGL_SET_CULL);
    let cull_face = r.get(OGL_SET_CULL_FACE);
    let front_ccw = match r.get(OGL_SET_FRONT_FACE) {
        0x900 => false,
        0x901 => true,
        other => return Err(refuse("front face", OGL_SET_FRONT_FACE, other)),
    };
    // SET_WINDOW_ORIGIN FLIP_Y says which way the rows count when the
    // orientation is taken: with it they count up, as OpenGL window
    // coordinates do, without it down [public: Ryujinx
    // `StateUpdater.UpdateFrontFace` keeps the front face on a host with
    // rows up (OpenGL's lower-left clip origin) only with
    // `TriangleRastFlip` and inverts it otherwise] [inference, from the
    // firmware: under the opposite rule every triangle drawn with culling
    // on was culled, in Settings (FLIP_Y clear) as in the Mii editor's
    // full-screen composition strip (FLIP_Y set), which left the editor's
    // frame buffers unwritten].
    let flip = r.get(SET_WINDOW_ORIGIN) & 0x10 != 0;
    let provoking_last = r.get(SET_PROVOKING_VERTEX) & 1 != 0;

    let mut triangles = Vec::new();
    for indices in list {
        let shaded: Vec<&Shaded> = indices
            .iter()
            .map(|&i| vertices[i].as_ref().expect("assembly skips restarts"))
            .collect();
        let provoking = if provoking_last { shaded[2] } else { shaded[0] };
        let mut polygon: Vec<ClipVertex> =
            shaded.iter().map(|v| clip.vertex(v, varyings)).collect();
        for &plane in &clip.planes {
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
        let window: Vec<[f32; 4]> = polygon.iter().map(|v| clip.window(v)).collect();
        // Facing from the signed area of the whole polygon in window
        // coordinates: positive is counter-clockwise with rows counted up
        // and clockwise with rows counted down.
        let area: f32 = (0..window.len())
            .map(|i| {
                let a = window[i];
                let b = window[(i + 1) % window.len()];
                a[0] * b[1] - b[0] * a[1]
            })
            .sum();
        let counter_clockwise = (area > 0.0) == flip;
        let front_facing = counter_clockwise == front_ccw;
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
            let (planes, depth) = primitive_planes(
                varyings,
                points,
                corners.map(|k| (&polygon[k], window[k])),
                provoking,
            );
            if let Some(triangle) = Triangle::new(points, planes, depth, front_facing) {
                triangles.push(triangle);
            }
        }
    }
    Ok(triangles)
}

/// How line segments rasterize.
#[derive(Debug, Clone, Copy, PartialEq)]
enum LineRule {
    /// A rectangle `width` wide centered on the segment, with its ends
    /// through the endpoints perpendicular to it, covering the pixels
    /// whose centers are inside.
    Rectangle { width: f32 },
    /// The non-antialiased line, `width` pixels wide
    /// ([`raster::aliased_line`]).
    Aliased { width: u32 },
}

/// The line rule the draw's state selects.
///
/// The rasterizer has three line modes. Mesa's NVK (nouveau's Vulkan
/// driver) selects them as follows [public: Mesa
/// `src/nouveau/vulkan/nvk_cmd_draw.c`]. It enables multisample
/// rasterization (`SET_ANTI_ALIAS_ENABLE`) even for one sample ("this
/// way we get strict lines and rectangular line support"). Rectangular
/// lines are `SET_LINE_MULTISAMPLE_OVERRIDE` and `SET_ANTI_ALIASED_LINE`
/// both off. Bresenham lines are the override on. Smooth rectangular
/// lines are both on.
///
/// - Rectangular lines (multisample rasterization on, override and
///   smooth off) cover the sample points inside a rectangle of the line
///   width centered on the segment [public: Vulkan 1.3, "Basic Line
///   Segment Rasterization"]. `SET_ANTI_ALIAS` 1X1 (checked at every
///   draw) puts the one sample at the pixel center. Samples exactly on
///   the rectangle's edges go by the top-left rule of the two triangles
///   the rectangle is drawn as \[assumption: the hardware's tie rule is
///   not public\]. The width is `SET_LINE_WIDTH_FLOAT`: nouveau's OpenGL
///   driver writes that register for smooth or multisampled lines and
///   `SET_ALIASED_LINE_WIDTH_FLOAT` for the others, with
///   `SET_ALIASED_LINE_WIDTH_ENABLE` set at init [public: Mesa
///   `nvc0_state.c` `nvc0_rasterizer_state_create`, `nvc0_screen.c`].
/// - Non-antialiased lines (multisample rasterization off, or the
///   override on; smooth off) follow OpenGL's diamond-exit rule, the
///   width rounded to the nearest integer, 0 counting as 1 (see
///   [`raster::aliased_line`]). The width is
///   `SET_ALIASED_LINE_WIDTH_FLOAT` with `SET_ALIASED_LINE_WIDTH_ENABLE`,
///   else `SET_LINE_WIDTH_FLOAT` \[inference: envytools `gf100_3d.xml`
///   names 0x20c `LINE_WIDTH_SEPARATE`\]. The firmware's touch screen
///   test (Settings, Test Input Devices) draws its strokes this way:
///   multisample rasterization off, width 1.
///
/// Widths beyond 10, the widest line nouveau's OpenGL driver advertises
/// (`max_line_width`), are refused: the hardware's limit and what it
/// does past it are not established. Endpoints are snapped to
/// `SET_SNAP_GRID_LINE`, here only the 1/256 pixel grid with RTNE
/// rounding that triangles use.
///
/// Refused: stippled lines, and smooth lines (their coverage comes from
/// `SET_LINE_SMOOTH_PARAMETERS` and `SET_LINE_SMOOTH_EDGE_TABLE`, whose
/// meaning is not public).
fn line_rule(r: Registers<'_>) -> Result<LineRule, Unmodeled> {
    require(r, "line stipple", SET_LINE_STIPPLE, 1, 0)?;
    require(r, "smooth lines", SET_ANTI_ALIASED_LINE, 1, 0)?;
    require(
        r,
        "line snap grid other than 256x256 RTNE",
        SET_SNAP_GRID_LINE,
        0x10F,
        8,
    )?;
    let rectangular = r.enabled(SET_ANTI_ALIAS_ENABLE) && !r.enabled(SET_LINE_MULTISAMPLE_OVERRIDE);
    let register = if !rectangular && r.enabled(SET_ALIASED_LINE_WIDTH_ENABLE) {
        SET_ALIASED_LINE_WIDTH_FLOAT
    } else {
        SET_LINE_WIDTH_FLOAT
    };
    let width = r.float(register);
    if !(width > 0.0 && width <= 10.0) {
        return Err(refuse("line width", register, r.get(register)));
    }
    Ok(if rectangular {
        LineRule::Rectangle { width }
    } else {
        LineRule::Aliased {
            width: (width.round() as u32).max(1),
        }
    })
}

/// Clips and sets up line segments as the pixels `line_rule` covers: the
/// rectangle of a rectangular line, the runs of whole pixels of a
/// non-antialiased one, two triangles each. Attributes vary along the
/// segment only: the planes are those through the corners of the
/// segment's rectangle with the values of the endpoint each corner lies
/// beside, which is the interpolation of a line fragment at its
/// projection onto the segment that OpenGL and Vulkan define. The
/// provoking vertex of a segment is its second vertex, or its first with
/// `SET_PROVOKING_VERTEX` FIRST; lines are front-facing and not culled
/// [public: OpenGL 4.6 core, "Flatshading" and the fragment shader input
/// `gl_FrontFacing`]. Only aliased-line pixels in `area` (scissor and
/// surface clip) are set up.
fn setup_lines(
    r: Registers<'_>,
    clip: &Clip,
    list: &[[usize; 2]],
    vertices: &[Option<Shaded>],
    varyings: &[(usize, Interpolation)],
    area: Rect,
) -> Result<Vec<Triangle>, Unmodeled> {
    let rule = line_rule(r)?;
    let provoking_last = r.get(SET_PROVOKING_VERTEX) & 1 != 0;
    let mut triangles = Vec::new();
    let mut push = |points: [[f32; 2]; 3], planes: &[Plane; 256], depth: Plane| {
        if let Some(triangle) = Triangle::new(points, Box::new(*planes), depth, true) {
            triangles.push(triangle);
        }
    };
    for &[a, b] in list {
        let shaded = [a, b].map(|i| vertices[i].as_ref().expect("assembly skips restarts"));
        let provoking = shaded[usize::from(provoking_last)];
        let Some((a, b)) = clip.segment(
            clip.vertex(shaded[0], varyings),
            clip.vertex(shaded[1], varyings),
        ) else {
            continue;
        };
        let [wa, wb] = [&a, &b].map(|v| {
            let [x, y, z, inv_w] = clip.window(v);
            [raster::snap_to_grid(x), raster::snap_to_grid(y), z, inv_w]
        });
        let d = [wb[0] - wa[0], wb[1] - wa[1]];
        let length = d[0].hypot(d[1]);
        if length == 0.0 {
            continue;
        }
        let half = match rule {
            LineRule::Rectangle { width } => width / 2.0,
            LineRule::Aliased { .. } => 0.5,
        } / length;
        let normal = [-d[1] * half, d[0] * half];
        let at = |w: [f32; 4], side: f32| [w[0] + side * normal[0], w[1] + side * normal[1]];
        let corners = [at(wa, 1.0), at(wb, 1.0), at(wb, -1.0), at(wa, -1.0)];
        let (planes, depth) = primitive_planes(
            varyings,
            [corners[0], corners[1], corners[3]],
            [(&a, wa), (&b, wb), (&a, wa)],
            provoking,
        );
        match rule {
            LineRule::Rectangle { .. } => {
                push([corners[0], corners[1], corners[2]], &planes, depth);
                push([corners[0], corners[2], corners[3]], &planes, depth);
            }
            LineRule::Aliased { width } => {
                let runs = raster::aliased_line([wa[0], wa[1]], [wb[0], wb[1]], width, area);
                for [x0, y0, x1, y1] in runs {
                    let [x0, y0, x1, y1] = [x0, y0, x1, y1].map(|v| v as f32);
                    push([[x0, y0], [x1, y0], [x1, y1]], &planes, depth);
                    push([[x0, y0], [x1, y1], [x0, y1]], &planes, depth);
                }
            }
        }
    }
    Ok(triangles)
}

/// The square a point covers: axis-aligned, of side `size`, centered at
/// `center`, covering the pixels whose centers are inside.
#[derive(Debug, Clone, Copy, PartialEq)]
struct PointSquare {
    center: [f32; 2],
    size: f32,
}

/// The square a point at window position (`x`, `y`) with point size
/// `size` covers.
///
/// With `SET_POINT_SPRITE` enabled it is centered at the point, its side
/// the point size [public: OpenGL 4.6 compatibility profile, "Basic
/// Point Rasterization" (point sprites) and "Point Multisample
/// Rasterization" (a square with point sprites)]. Without point sprites
/// and without multisample rasterization (`SET_ANTI_ALIAS_ENABLE`) it is
/// OpenGL's non-antialiased point: the size rounded to the nearest
/// integer (at least 1), centered at the center of the pixel holding the
/// point for odd sizes and at the pixel corner nearest it for even ones
/// (same section). Without point sprites but with multisample
/// rasterization the point is a circle there; that is refused, as are
/// smooth points (`SET_ANTI_ALIASED_POINT`) and `SET_POINT_CENTER_MODE`
/// D3D, whose effect is not documented. Sizes beyond 63, the largest
/// point nouveau's OpenGL driver advertises (`max_point_size`, Mesa
/// `nvc0_screen.c`), are refused like wide lines.
fn point_rule(r: Registers<'_>, x: f32, y: f32, size: f32) -> Result<PointSquare, Unmodeled> {
    require(r, "smooth points", SET_ANTI_ALIASED_POINT, 1, 0)?;
    require(r, "point center mode D3D", SET_POINT_CENTER_MODE, !0, 0)?;
    if !(size > 0.0 && size <= 63.0) {
        return Err(Unmodeled(format!("3D: point size {size}")));
    }
    let sprite = r.enabled(SET_POINT_SPRITE);
    let multisample = r.enabled(SET_ANTI_ALIAS_ENABLE);
    Ok(match (sprite, multisample) {
        (true, _) => PointSquare {
            center: [raster::snap_to_grid(x), raster::snap_to_grid(y)],
            size,
        },
        (false, false) => {
            let size = size.round().max(1.0);
            let center = |v: f32| {
                if size % 2.0 == 1.0 {
                    v.floor() + 0.5
                } else {
                    (v + 0.5).floor()
                }
            };
            PointSquare {
                center: [center(x), center(y)],
                size,
            }
        }
        (false, true) => {
            return Err(refuse(
                "round (multisample, non-sprite) points",
                SET_POINT_SPRITE,
                r.get(SET_POINT_SPRITE),
            ));
        }
    })
}

/// The size of a point: the last vertex-pipeline shader's point size
/// output a\[0x6c\] with `SET_ATTRIBUTE_POINT_SIZE` ENABLE, else
/// `SET_POINT_SIZE` [public: envytools `gf100_3d.xml` `VP_POINT_SIZE`;
/// NVK enables it with SLOT 0 for shaders that write the point size].
/// Other SLOT values, and the attribute enabled for a shader that does
/// not write it, are refused: what either selects is not established.
fn point_size(r: Registers<'_>, last: &Program, vertex: &Shaded) -> Result<f32, Unmodeled> {
    let control = r.get(SET_ATTRIBUTE_POINT_SIZE);
    if control & 1 == 0 {
        return Ok(r.float(SET_POINT_SIZE));
    }
    if control >> 4 & 0xFF != 0 || !last.header().vtg_output_used(0x6C / 4) {
        return Err(refuse(
            "point size attribute slot, or a shader that writes no point size",
            SET_ATTRIBUTE_POINT_SIZE,
            control,
        ));
    }
    Ok(f32::from_bits(vertex.attributes.0[0x6C / 4]))
}

/// Clips and sets up points as the squares `point_rule` covers, two
/// triangles each, with every input constant at the point's values. A
/// point is discarded when its vertex is outside a clip plane [public:
/// OpenGL 4.6 core, "Primitive Clipping"]. Point sprite coordinates
/// (a\[0x2e0\], a\[0x2e4\]) and the coordinates point sprites replace
/// (`SET_POINT_SPRITE_SELECT` GENERATE) are refused.
fn setup_points(
    r: Registers<'_>,
    clip: &Clip,
    list: &[usize],
    vertices: &[Option<Shaded>],
    varyings: &[(usize, Interpolation)],
    last: &Program,
    pixel: &Program,
) -> Result<Vec<Triangle>, Unmodeled> {
    if [0x2E0 / 4, 0x2E4 / 4]
        .iter()
        .any(|&word| pixel.header().ps_interpolation(word) != Interpolation::Unused)
    {
        return Err(Unmodeled(
            "3D: pixel shader reads point sprite coordinates".into(),
        ));
    }
    let select = r.get(SET_POINT_SPRITE_SELECT);
    if r.enabled(SET_POINT_SPRITE) && select & 0x1FF8 != 0 {
        return Err(refuse(
            "point sprite texture coordinate generation",
            SET_POINT_SPRITE_SELECT,
            select,
        ));
    }
    let mut triangles = Vec::new();
    for &i in list {
        let shaded = vertices[i].as_ref().expect("assembly skips restarts");
        let v = clip.vertex(shaded, varyings);
        if clip.planes.iter().any(|plane| plane.distance(&v) < 0.0) {
            continue;
        }
        let window = clip.window(&v);
        let size = point_size(r, last, shaded)?;
        let square = point_rule(r, window[0], window[1], size)?;
        let half = square.size / 2.0;
        let [cx, cy] = square.center;
        let corners = [
            [cx - half, cy - half],
            [cx + half, cy - half],
            [cx + half, cy + half],
            [cx - half, cy + half],
        ];
        let (planes, depth) = primitive_planes(
            varyings,
            [corners[0], corners[1], corners[3]],
            [(&v, window); 3],
            shaded,
        );
        for points in [
            [corners[0], corners[1], corners[2]],
            [corners[0], corners[2], corners[3]],
        ] {
            if let Some(triangle) = Triangle::new(points, planes.clone(), depth, true) {
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

    /// A2B10G10R10 unpacks bit fields, R lowest; SNORM clamps the most
    /// negative code to -1 (the 2-bit alpha too); byte-sized formats read
    /// as before.
    #[test]
    fn packed_and_byte_vertex_attributes() {
        let attribute = |widths: u32, kind: u32| (kind << 27) | (widths << 21);
        let float = |words: [u32; 4]| words.map(f32::from_bits);
        // R = 511, G = -511, B = -512, A = -2 (binary 10).
        let packed: u32 = 511 | ((1024 - 511) << 10) | (512 << 20) | (2 << 30);
        let words = super::fetch_attribute(attribute(0x30, 1), &packed.to_le_bytes()).unwrap();
        assert_eq!(float(words), [1.0, -1.0, -1.0, -1.0]);
        // UNORM: 0, 1023, 512, 3.
        let packed: u32 = (1023 << 10) | (512 << 20) | (3 << 30);
        let words = super::fetch_attribute(attribute(0x30, 2), &packed.to_le_bytes()).unwrap();
        assert_eq!(float(words), [0.0, 1.0, (512.0f64 / 1023.0) as f32, 1.0]);
        // UINT, with R and B swapped.
        let words = super::fetch_attribute(
            attribute(0x30, 4) | (1 << 31),
            &(5u32 | (6 << 10) | (7 << 20) | (1 << 30)).to_le_bytes(),
        )
        .unwrap();
        assert_eq!(words, [7, 6, 5, 1]);
        // R16_G16_B16 SINT reads three halves, alpha 1.
        let bytes = [0xff, 0xff, 2, 0, 0, 0x80];
        let words = super::fetch_attribute(attribute(0x05, 3), &bytes).unwrap();
        assert_eq!(words, [-1i32 as u32, 2, -32768i32 as u32, 1]);
    }
}
