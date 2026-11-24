//! CPU texture storage and Maxwell texture descriptors.
//!
//! Descriptor fields follow [public: NVIDIA open-gpu-doc classes/3d/clb197tex.h,
//! TEXHEAD_BL/TEXHEAD_PITCH and TEXSAMP; Ryujinx Image/TextureDescriptor.cs].
//! Cache coherency hashes and sector promotion only control memory transactions,
//! not texel values, and have no observable effect on this coherent CPU decoder.

//! Filtering quality hints in words 3 and 6 are inactive when
//! USE_HEADER_OPT_CONTROL is clear; otherwise parsing refuses the override.
//! GOBS_PER_BLOCK_WIDTH is ignored by TEX (assumed one GOB)
//! [public: envytools rnndb/graph/gm200_texture.xml].
mod astc;
mod bc;
mod filtering;
pub mod format;
pub mod layout;

use super::Unmodeled;
use super::sampling::Target;
use format::{ComponentType, Encoding, ImageFormat};
use layout::{BlockLinear, ImageShape};

/// Parsed Maxwell v3 texture descriptor, retaining every original field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tic {
    pub words: [u32; 8],
    pub address: u64,
    pub format: ImageFormat,
    pub types: [ComponentType; 4],
    pub swizzle: [u8; 4],
    pub shape: ImageShape,
    pub block: BlockLinear,
    pub pitch: Option<u32>,
    pub target: Target,
    pub array: bool,
    pub normalized: bool,
    pub srgb: bool,
    pub base_level: u32,
    pub max_level: u32,
    pub border_size: u32,
}

fn unsupported(field: &str, value: impl std::fmt::Display) -> Unmodeled {
    Unmodeled(format!("{field}={value}"))
}

impl Tic {
    /// The level-of-detail range sampling may select, as the interpreter
    /// clamps it (`clamp_lod`): (minimum, maximum) relative to the base
    /// level.
    pub fn lod_range(&self, tsc: &Tsc) -> (f32, f32) {
        let resident = ((self.words[7] >> 12) & 4095) as f32 / 256.;
        let top = (self.max_level - self.base_level) as f32;
        (
            tsc.min_lod.max(resident).clamp(0., top),
            tsc.max_lod.max(resident).clamp(0., top),
        )
    }
}

impl Tic {
    /// Parses the block-linear and pitch headers. Color-key, sparse, multisample,
    /// packed-component, and header-overridden filtering modes are refused.
    /// Pitch textures have no mipmaps [public: Ryujinx Image/TexturePool.cs GetInfo].
    pub fn parse(words: [u32; 8]) -> Result<Self, Unmodeled> {
        let version = (words[2] >> 21) & 7;
        if !matches!(version, 2 | 3) {
            return Err(unsupported("TIC HEADER_VERSION", version));
        }
        let format = ImageFormat::from_tic(words[0] & 0x7f)?;
        if words[0] >> 31 != 0 {
            return Err(unsupported("TIC PACK_COMPONENTS", 1));
        }
        let mut types = [ComponentType::Unorm; 4];
        for (i, t) in types.iter_mut().enumerate() {
            let raw = (words[0] >> (7 + i * 3)) & 7;
            *t = ComponentType::from_raw(raw).ok_or_else(|| unsupported("TIC DATA_TYPE", raw))?;
        }
        if !matches!(
            format.encoding,
            Encoding::Packed(_) | Encoding::SharedExponent
        ) {
            for (i, present) in format.present().into_iter().enumerate() {
                if present
                    && types[i] != ComponentType::Unorm
                    && !(matches!(format.encoding, Encoding::Bc4 | Encoding::Bc5)
                        && types[i] == ComponentType::Snorm)
                {
                    return Err(unsupported(
                        "TIC compressed DATA_TYPE",
                        format_args!("component {i}: {:?}", types[i]),
                    ));
                }
            }
            if format.encoding == Encoding::Bc5 && types[0] != types[1] {
                return Err(unsupported("TIC BC5 mixed DATA_TYPE", words[0]));
            }
        }
        // Word 6 bit 0 is COLOR_KEY_OP, read only by the colour-key header
        // versions (1, 4) refused above; the block-linear and pitch headers
        // name it RESERVED6Y [public: envytools rnndb/graph/gm200_texture.xml
        // TIC word 6 `COLOR_KEY_OP`, "Used if (HEADER_VERSION ==
        // PITCH_COLORKEY || HEADER_VERSION == BLOCKLINEAR_COLORKEY)"; NVIDIA
        // clb197tex.h `TEXHEAD_BL_RESERVED6Y` MW(192:192)]. The Mii editor's
        // textures arrive with it set.
        for (index, mask) in [
            (2, 0xe11f0000u32),
            (4, 0x00070000),
            (5, 0x40000000),
            (7, 0xff000000),
        ] {
            if words[index] & mask != 0 {
                return Err(unsupported(
                    "TIC reserved field",
                    format_args!("word{index}={:#x}", words[index] & mask),
                ));
            }
        }
        let swizzle = std::array::from_fn(|i| ((words[0] >> (19 + 3 * i)) & 7) as u8);
        if swizzle.contains(&1) {
            return Err(unsupported("TIC SOURCE", 1));
        }
        // Integer stencil components are allowed when the swizzle never selects them.
        for &s in &swizzle {
            if (2..=5).contains(&s)
                && matches!(
                    types[(s - 2) as usize],
                    ComponentType::Sint | ComponentType::Uint
                )
            {
                return Err(unsupported("TIC selected integer DATA_TYPE", s));
            }
        }
        if words[3] & (1 << 26) != 0 {
            return Err(unsupported("TIC USE_HEADER_OPT_CONTROL", 1));
        }
        if words[7] & 0xf00 != 0 {
            return Err(unsupported("TIC MULTI_SAMPLE_COUNT", (words[7] >> 8) & 15));
        }
        let kind = (words[4] >> 23) & 15;
        let (target, array) = match kind {
            0 => (Target::Texture1D, false),
            1 | 7 => (Target::Texture2D, false),
            2 => (Target::Texture3D, false),
            3 => (Target::Cube, false),
            4 => (Target::Texture1D, true),
            5 => (Target::Texture2D, true),
            8 => (Target::Cube, true),
            _ => return Err(unsupported("TIC TEXTURE_TYPE", kind)),
        };
        let width = (words[4] & 0xffff) + 1;
        let height = if target == Target::Texture1D {
            1
        } else {
            (words[5] & 0xffff) + 1
        };
        let d = ((words[5] >> 16) & 0x3fff) + 1;
        let depth = if target == Target::Texture3D { d } else { 1 };
        let layers = if target == Target::Cube {
            if array { d * 6 } else { 6 }
        } else if array {
            d
        } else {
            1
        };
        let levels = (words[3] >> 28) + 1;
        let base_level = words[7] & 15;
        let max_level = (words[7] >> 4) & 15;
        if base_level > max_level || max_level >= levels {
            return Err(unsupported("TIC mip view", words[7] & 255));
        }
        let block = BlockLinear {
            block_height_log2: (words[3] >> 3) & 7,
            block_depth_log2: (words[3] >> 6) & 7,
        };
        let pitch = if version == 2 {
            if levels != 1 {
                return Err(unsupported("TIC pitch mip levels", levels));
            }
            let pitch = (words[3] & 0xffff) << 5;
            if pitch < width.div_ceil(format.block_width) * format.bytes_per_block {
                return Err(unsupported("TIC PITCH", pitch));
            }
            Some(pitch)
        } else {
            if block.block_height_log2 > 5 || block.block_depth_log2 > 5 {
                return Err(unsupported("TIC GOBS_PER_BLOCK", words[3] & 511));
            }
            if words[3] & 0x3e00 != 0 {
                return Err(unsupported("TIC sparse/GOB3D", words[3] & 0x3e00));
            }
            if words[1] & 0x1ff != 0 {
                return Err(unsupported(
                    "TIC GOB_DEPTH_OFFSET/reserved address",
                    words[1] & 511,
                ));
            }
            None
        };
        Ok(Self {
            words,
            address: u64::from(words[1]) | (u64::from(words[2] & 0xffff) << 32),
            format,
            types,
            swizzle,
            shape: ImageShape {
                width,
                height,
                depth,
                layers,
                levels,
                block_width: format.block_width,
                block_height: format.block_height,
                bytes_per_block: format.bytes_per_block,
            },
            block,
            pitch,
            target,
            array,
            normalized: words[5] >> 31 != 0,
            srgb: words[4] & (1 << 22) != 0,
            base_level,
            max_level,
            border_size: words[4] >> 29,
        })
    }
}

/// Parsed TEXSAMP state [public: NVIDIA clb197tex.h TEXSAMP0..7].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tsc {
    pub words: [u32; 8],
    pub wrap: [u32; 3],
    pub mag_linear: bool,
    pub min_filter: u32,
    pub mip_filter: u32,
    pub compare: bool,
    pub compare_func: u32,
    pub anisotropy: u32,
    pub lod_bias: f32,
    pub min_lod: f32,
    pub max_lod: f32,
    pub border: [f32; 4],
    /// S_R_G_B_CONVERSION (word 0 bit 13). Texel decoding follows the TIC
    /// sRGB bit when this is set [public: eden decodes by the TIC bit
    /// alone; Ryujinx `TexturePool.GetForBinding` decodes when both bits
    /// are set]. With it set, an sRGB texture's border is the 8-bit sRGB
    /// border color of words 2-3 (eden `TSCEntry::SrgbBorderColor`), which
    /// Ryujinx ignores; that border is refused when sampled.
    pub srgb_conversion: bool,
    pub force_unnormalized: bool,
    pub cube_filter: u32,
}
impl Tsc {
    /// Rejects reduction filters and nonzero trilinear optimization: their
    /// approximation must not silently alter the exact filtering implemented here.
    ///
    /// Word 0 bits 16:14 and 19:17 (RESERVED0A/0B in NVIDIA clb197tex.h)
    /// are FONT_FILTER_WIDTH/HEIGHT, the box filter of BITMAP_8X8 (one bit
    /// per texel) textures only [public: envytools
    /// rnndb/graph/g80_texture.xml TSC]. The one-bit TIC component format,
    /// R1 (0x1f), is refused by [`ImageFormat::from_tic`], so the fields
    /// cannot affect any texture this model samples. NVN's samplers set
    /// them: 0x24000 (width 2, height 3) arrives from the system applets.
    pub fn parse(words: [u32; 8]) -> Result<Self, Unmodeled> {
        for (index, mask) in [(0, 0xff800000u32), (1, 0x80000000), (3, 0xf0000fff)] {
            if words[index] & mask != 0 {
                return Err(unsupported(
                    "TSC reserved field",
                    format_args!("word{index}={:#x}", words[index] & mask),
                ));
            }
        }
        let mag = words[1] & 7;
        let min_filter = (words[1] >> 4) & 3;
        let mip_filter = (words[1] >> 6) & 3;
        if !matches!(mag, 1 | 2) {
            return Err(unsupported("TSC MAG_FILTER", mag));
        }
        if min_filter == 0 {
            return Err(unsupported("TSC MIN_FILTER", min_filter));
        }
        if mip_filter == 0 {
            return Err(unsupported("TSC MIP_FILTER", mip_filter));
        }
        if words[1] & 0xc00 != 0 {
            return Err(unsupported("TSC REDUCTION_FILTER", (words[1] >> 10) & 3));
        }
        if words[1] & 0x7c000000 != 0 {
            return Err(unsupported("TSC TRILIN_OPT", (words[1] >> 26) & 31));
        }
        if words[1] & 8 != 0 {
            return Err(unsupported("TSC MIN_LOD_CLAMP_BEHAVIOR", 1));
        }
        let min_lod = (words[2] & 4095) as f32 / 256.;
        let max_lod = ((words[2] >> 12) & 4095) as f32 / 256.;
        if min_lod > max_lod {
            return Err(unsupported("TSC MIN_LOD exceeds MAX_LOD", min_lod));
        }
        Ok(Self {
            words,
            wrap: std::array::from_fn(|i| (words[0] >> (3 * i)) & 7),
            mag_linear: mag == 2,
            min_filter,
            mip_filter,
            compare: words[0] & 512 != 0,
            compare_func: (words[0] >> 10) & 7,
            anisotropy: [1, 2, 4, 6, 8, 10, 12, 16][((words[0] >> 20) & 7) as usize],
            lod_bias: (((words[1] >> 12) << 19) as i32 >> 19) as f32 / 256.,
            min_lod,
            max_lod,
            border: std::array::from_fn(|i| f32::from_bits(words[4 + i])),
            srgb_conversion: words[0] & (1 << 13) != 0,
            force_unnormalized: words[1] & (1 << 25) != 0,
            cube_filter: (words[1] >> 8) & 3,
        })
    }
}

struct Level {
    width: u32,
    height: u32,
    depth: u32,
    texels: Vec<[f32; 4]>,
}

/// Immutable decoded image; shareable across render workers without locking.
/// Each mip stores layers consecutively, then depth slices, rows, and columns.
pub struct Texture {
    tic: Tic,
    levels: Vec<Level>,
}

impl Texture {
    /// The descriptor the texture was decoded from.
    pub fn tic(&self) -> &Tic {
        &self.tic
    }

    /// Decoded level `level`: width, height, depth and the texels (layers,
    /// then slices, rows and columns), component swizzle and sRGB decoding
    /// applied.
    pub fn level(&self, level: u32) -> Option<(u32, u32, u32, &[[f32; 4]])> {
        self.levels
            .get(level as usize)
            .map(|m| (m.width, m.height, m.depth, m.texels.as_slice()))
    }

    /// Host memory the decoded texels take.
    pub fn bytes(&self) -> usize {
        self.levels
            .iter()
            .map(|m| std::mem::size_of_val(m.texels.as_slice()))
            .sum()
    }

    /// Byte extent of all allocated levels and layers (including layout padding).
    pub fn footprint(tic: &Tic) -> Result<(u64, u64), Unmodeled> {
        let size = if let Some(pitch) = tic.pitch {
            u64::from(pitch)
                * u64::from(tic.shape.height.div_ceil(tic.format.block_height))
                * u64::from(tic.shape.depth)
                * u64::from(tic.shape.layers)
        } else {
            layout::block_linear_image(&tic.shape, tic.block)?.size
        };
        tic.address
            .checked_add(size)
            .ok_or_else(|| unsupported("TIC address extent overflow", tic.address))?;
        Ok((tic.address, size))
    }

    /// Decodes the texture `tic` describes from `raw`, the bytes of its
    /// footprint ([`Texture::footprint`]), directly into the final mip
    /// buffers. There is no per-texel allocation. sRGB conversion precedes
    /// component swizzle and filtering [public: OpenGL 4.6 §8.24; NVIDIA
    /// clb197tex.h S_R_G_B_CONVERSION].
    ///
    /// The work per texel is kept small: a block's offset is its row's
    /// offset plus its column's, both computed once (block-linear
    /// addressing adds the two, see [`layout::gob_offset`]), and each
    /// component of a packed format is converted once per code (by
    /// table).
    pub fn decode(raw: &[u8], tic: &Tic) -> Result<Self, Unmodeled> {
        let (_, size) = Self::footprint(tic)?;
        let raw = usize::try_from(size)
            .ok()
            .and_then(|size| raw.get(..size))
            .ok_or_else(|| {
                unsupported(
                    "texture bytes",
                    format_args!("{:#x} of {size:#x}", raw.len()),
                )
            })?;
        let tiled = if tic.pitch.is_none() {
            Some(layout::block_linear_image(&tic.shape, tic.block)?)
        } else {
            None
        };
        let f = tic.format;
        let bytes_per_block = f.bytes_per_block as usize;
        // Bytes of a block straddle 16-byte sectors (and are gathered one
        // at a time) only in a block-linear layout of a size that is not a
        // power of two.
        let gather = tiled.is_some() && !f.bytes_per_block.is_power_of_two();
        let packed = match f.encoding {
            Encoding::Packed(fields) => Some(Packed::new(tic, fields)?),
            _ => None,
        };
        // Index of each output component in [R, G, B, A, 0, 1].
        let swizzle = tic.swizzle.map(|s| match s {
            0 => 4,
            2..=5 => usize::from(s - 2),
            6 | 7 => 5,
            _ => unreachable!(),
        });
        let srgb8 = matches!(
            f.encoding,
            Encoding::Bc1 | Encoding::Bc2 | Encoding::Bc3 | Encoding::Bc7
        );
        // Fields of up to 16 bits are converted by table too when the
        // texture has many more texels than a table has entries.
        let level_0 = u64::from(tic.shape.width)
            * u64::from(tic.shape.height)
            * u64::from(tic.shape.depth)
            * u64::from(tic.shape.layers);
        let lookups = packed.as_ref().filter(|_| !gather).and_then(|packed| {
            packed.lookups(
                f.bytes_per_block,
                swizzle,
                if level_0 >= 4 << 16 { 16 } else { 8 },
            )
        });
        let mut levels = Vec::with_capacity(tic.shape.levels as usize);
        for level in 0..tic.shape.levels {
            let width = (tic.shape.width >> level).max(1);
            let height = (tic.shape.height >> level).max(1);
            let depth = (tic.shape.depth >> level).max(1);
            let count = u64::from(width)
                * u64::from(height)
                * u64::from(depth)
                * u64::from(tic.shape.layers);
            let mut texels = Vec::new();
            texels
                .try_reserve_exact(
                    usize::try_from(count)
                        .map_err(|_| unsupported("texture texel extent", count))?,
                )
                .map_err(|e| unsupported("decoded texture allocation", e))?;
            if lookups.is_none() {
                texels.resize(count as usize, [0.; 4]);
            }
            let blocks_wide = width.div_ceil(f.block_width);
            let blocks_high = height.div_ceil(f.block_height);
            let surface = tiled.as_ref().map(|image| &image.levels[level as usize]);
            let blocks = Blocks {
                raw,
                // Offset of each block of a row from the row's offset.
                columns: (0..blocks_wide)
                    .map(|bx| match surface {
                        Some(m) => m.surface.offset(bx * f.bytes_per_block, 0, 0) as usize,
                        None => bx as usize * bytes_per_block,
                    })
                    .collect(),
                gather: surface.filter(|_| gather).map(|m| &m.surface),
                bytes_per_block: f.bytes_per_block,
            };
            let mut split = [0u8; 16];
            let mut block = [[0.; 4]; 144];
            for layer in 0..tic.shape.layers {
                for z in 0..depth {
                    let plane = (layer as usize * depth as usize + z as usize) * height as usize;
                    for by in 0..blocks_high {
                        let row = match (&tiled, surface) {
                            (Some(image), Some(m)) => {
                                u64::from(layer) * image.layer_stride
                                    + m.offset
                                    + m.surface.offset(0, by, z)
                            }
                            _ => {
                                u64::from(tic.pitch.unwrap())
                                    * (u64::from(layer * depth + z) * u64::from(blocks_high)
                                        + u64::from(by))
                            }
                        } as usize;
                        if let Some(lookups) = &lookups {
                            // One texel per block, rows in order.
                            let row_of = match bytes_per_block {
                                1 => lookup_row::<1>,
                                2 => lookup_row::<2>,
                                4 => lookup_row::<4>,
                                _ => lookup_row::<8>,
                            };
                            row_of(raw, row, &blocks.columns, lookups, &mut texels)?;
                            continue;
                        }
                        if let Some(packed) = &packed {
                            // One texel per block.
                            let start = (plane + by as usize) * width as usize;
                            for (bx, texel) in
                                texels[start..start + width as usize].iter_mut().enumerate()
                            {
                                let c = packed.texel(blocks.get(row, bx as u32, &mut split)?);
                                let c = [c[0], c[1], c[2], c[3], 0., 1.];
                                *texel = swizzle.map(|i| c[i]);
                            }
                            continue;
                        }
                        for bx in 0..blocks_wide {
                            decode_block(tic, blocks.get(row, bx, &mut split)?, &mut block)?;
                            for y in 0..f.block_height.min(height - by * f.block_height) {
                                let start = (plane + (by * f.block_height + y) as usize)
                                    * width as usize
                                    + (bx * f.block_width) as usize;
                                for x in 0..f.block_width.min(width - bx * f.block_width) {
                                    let mut c = block[(y * f.block_width + x) as usize];
                                    if tic.srgb {
                                        for v in &mut c[..3] {
                                            *v = srgb_decode(*v, srgb8);
                                        }
                                    }
                                    let c = [c[0], c[1], c[2], c[3], 0., 1.];
                                    texels[start + x as usize] = swizzle.map(|i| c[i]);
                                }
                            }
                        }
                    }
                }
            }
            levels.push(Level {
                width,
                height,
                depth,
                texels,
            });
        }
        Ok(Self { tic: *tic, levels })
    }
}

/// The blocks of one level in the footprint's bytes.
struct Blocks<'a> {
    raw: &'a [u8],
    /// Offset of each block of a row from the row's offset.
    columns: Vec<usize>,
    /// The surface whose blocks straddle 16-byte sectors, gathered a byte
    /// at a time.
    gather: Option<&'a layout::BlockLinearSurface>,
    bytes_per_block: u32,
}

impl Blocks<'_> {
    /// The bytes of block `bx` of the row at offset `row` (in `split` if
    /// gathered).
    #[inline]
    fn get<'s>(
        &'s self,
        row: usize,
        bx: u32,
        split: &'s mut [u8; 16],
    ) -> Result<&'s [u8], Unmodeled> {
        let length = self.bytes_per_block as usize;
        if let Some(surface) = self.gather {
            for byte in 0..self.bytes_per_block {
                let address = row + surface.offset(bx * self.bytes_per_block + byte, 0, 0) as usize;
                split[byte as usize] = *self
                    .raw
                    .get(address)
                    .ok_or_else(|| unsupported("texture byte offset", address))?;
            }
            return Ok(&split[..length]);
        }
        let offset = row + self.columns[bx as usize];
        self.raw
            .get(offset..offset + length)
            .ok_or_else(|| unsupported("texture block offset", offset))
    }
}

/// An sRGB-encoded component to linear: through the table of the 256
/// 8-bit codes (rounding to the nearest code) for formats of 8-bit
/// components, by the curve otherwise.
fn srgb_decode(value: f32, srgb8: bool) -> f32 {
    if srgb8 {
        format::SRGB8_TO_LINEAR[(value * 255.).round().clamp(0., 255.) as usize]
    } else {
        format::srgb_to_linear(value)
    }
}

/// How a packed component's code becomes a number.
#[derive(Clone, Copy)]
enum Conversion {
    Unorm,
    Snorm,
    Uf10,
    Uf11,
    F16,
    F32,
    /// Integer components: never selected (`Tic::parse`), so their value
    /// cannot affect the result.
    Zero,
}

/// One component of a packed format as sampling sees it: its field of the
/// texel word, converted by its data type, rounded through binary16 for
/// the FORCE_FP16 types, then sRGB-decoded for R, G and B of an sRGB
/// view.
#[derive(Clone, Copy)]
struct Component {
    shift: u32,
    bits: u32,
    conversion: Conversion,
    force_fp16: bool,
    /// `Some(srgb8)` when sRGB-decoded (see [`srgb_decode`]).
    srgb: Option<bool>,
}

impl Component {
    /// The value of `code`.
    #[inline]
    fn value(&self, code: u32) -> f32 {
        let mut c = match self.conversion {
            Conversion::Unorm => format::unorm_to_f32(code, self.bits),
            Conversion::Snorm => format::snorm_to_f32(code, self.bits),
            Conversion::Uf10 => format::uf10_to_f32(code),
            Conversion::Uf11 => format::uf11_to_f32(code),
            Conversion::F16 => format::f16_to_f32(code as u16),
            Conversion::F32 => f32::from_bits(code),
            Conversion::Zero => 0.,
        };
        if self.force_fp16 {
            c = format::f16_to_f32(format::f32_to_f16(c));
        }
        if let Some(srgb8) = self.srgb {
            c = srgb_decode(c, srgb8);
        }
        c
    }
}

/// A component of a packed texel, prepared once per texture.
enum Channel {
    /// Absent from the format: 0 for R, G and B (sRGB-decoded if the view
    /// is sRGB), 1 for A.
    Constant(f32),
    /// At most 8 bits: the value of every code, from [`Component::value`].
    Table {
        shift: u32,
        mask: u128,
        values: Box<[f32]>,
    },
    /// Wider: [`Component::value`] per texel.
    Computed(Component),
}

/// The components of a packed format.
struct Packed {
    channels: [Channel; 4],
}

impl Packed {
    fn new(tic: &Tic, fields: [(u8, u8); 4]) -> Result<Self, Unmodeled> {
        let srgb8 = fields[..3].iter().all(|(_, bits)| matches!(bits, 0 | 8));
        let mut channels = Vec::with_capacity(4);
        for (i, (shift, bits)) in fields.into_iter().enumerate() {
            let srgb = (tic.srgb && i < 3).then_some(srgb8);
            if bits == 0 {
                let default = if i == 3 { 1. } else { 0. };
                channels.push(Channel::Constant(match srgb {
                    Some(srgb8) => srgb_decode(default, srgb8),
                    None => default,
                }));
                continue;
            }
            let conversion = match tic.types[i] {
                ComponentType::Unorm | ComponentType::UnormForceFp16 => Conversion::Unorm,
                ComponentType::Snorm | ComponentType::SnormForceFp16 => Conversion::Snorm,
                ComponentType::Float => match bits {
                    10 => Conversion::Uf10,
                    11 => Conversion::Uf11,
                    16 => Conversion::F16,
                    32 => Conversion::F32,
                    _ => return Err(unsupported("TIC FLOAT component bits", bits)),
                },
                ComponentType::Uint | ComponentType::Sint => Conversion::Zero,
            };
            let component = Component {
                shift: shift.into(),
                bits: bits.into(),
                conversion,
                force_fp16: matches!(
                    tic.types[i],
                    ComponentType::UnormForceFp16 | ComponentType::SnormForceFp16
                ),
                srgb,
            };
            channels.push(if bits <= 8 {
                Channel::Table {
                    shift: shift.into(),
                    mask: (1 << bits) - 1,
                    values: (0..1u32 << bits)
                        .map(|code| component.value(code))
                        .collect(),
                }
            } else {
                Channel::Computed(component)
            });
        }
        Ok(Packed {
            channels: channels.try_into().ok().expect("four components"),
        })
    }

    /// R, G, B and A of the texel stored in `bytes`, before the swizzle.
    #[inline]
    fn texel(&self, bytes: &[u8]) -> [f32; 4] {
        let word = match *bytes {
            [a] => u128::from(a),
            [a, b] => u16::from_le_bytes([a, b]).into(),
            [a, b, c, d] => u32::from_le_bytes([a, b, c, d]).into(),
            _ => bytes
                .iter()
                .enumerate()
                .fold(0, |word, (i, &b)| word | u128::from(b) << (i * 8)),
        };
        self.channels.each_ref().map(|channel| match channel {
            Channel::Constant(value) => *value,
            Channel::Table {
                shift,
                mask,
                values,
            } => values[((word >> shift) & mask) as usize],
            Channel::Computed(c) => c.value(((word >> c.shift) & ((1 << c.bits) - 1)) as u32),
        })
    }

    /// Per output component (after `swizzle`, indices into [R, G, B, A,
    /// 0, 1]): the value of every code of the field it selects. Only for
    /// texels of 1, 2, 4 or 8 bytes whose selected fields have at most
    /// `most_bits` bits.
    fn lookups(
        &self,
        bytes_per_block: u32,
        swizzle: [usize; 4],
        most_bits: u32,
    ) -> Option<[Lookup; 4]> {
        if !matches!(bytes_per_block, 1 | 2 | 4 | 8) {
            return None;
        }
        let constant = |value: f32| Lookup {
            shift: 0,
            mask: 0,
            values: Box::new([value]),
        };
        let mut lookups = Vec::with_capacity(4);
        for source in swizzle {
            lookups.push(match self.channels.get(source) {
                None => constant(if source == 4 { 0. } else { 1. }),
                Some(Channel::Constant(value)) => constant(*value),
                Some(Channel::Table {
                    shift,
                    mask,
                    values,
                }) => Lookup {
                    shift: *shift,
                    mask: *mask as u64,
                    values: values.clone(),
                },
                Some(Channel::Computed(c)) if c.bits <= most_bits => Lookup {
                    shift: c.shift,
                    mask: (1 << c.bits) - 1,
                    values: (0..1u32 << c.bits).map(|code| c.value(code)).collect(),
                },
                Some(Channel::Computed(_)) => return None,
            });
        }
        lookups.try_into().ok()
    }
}

/// One output component of a packed texel of at most 8 bytes: its value
/// for every code of the field at `shift` (`mask` wide; a constant has a
/// field of no bits).
struct Lookup {
    shift: u32,
    mask: u64,
    values: Box<[f32]>,
}

/// Appends the texels of the row at offset `row` of a packed format of
/// `N`-byte texels (each `columns[x]` past the row's offset), through
/// `lookups`.
fn lookup_row<const N: usize>(
    raw: &[u8],
    row: usize,
    columns: &[usize],
    lookups: &[Lookup; 4],
    texels: &mut Vec<[f32; 4]>,
) -> Result<(), Unmodeled> {
    for &column in columns {
        let offset = row + column;
        let bytes = raw
            .get(offset..offset + N)
            .ok_or_else(|| unsupported("texture block offset", offset))?;
        let mut word = [0u8; 8];
        word[..N].copy_from_slice(bytes);
        let word = u64::from_le_bytes(word);
        texels.push(
            lookups
                .each_ref()
                .map(|l| l.values[((word >> l.shift) & l.mask) as usize]),
        );
    }
    Ok(())
}

/// Decodes one block of a compressed or shared-exponent format (packed
/// formats are decoded per component, [`Packed`]).
fn decode_block(tic: &Tic, bytes: &[u8], out: &mut [[f32; 4]; 144]) -> Result<(), Unmodeled> {
    let signed = matches!(
        tic.types[0],
        ComponentType::Snorm | ComponentType::SnormForceFp16
    );
    match tic.format.encoding {
        Encoding::Packed(_) => unreachable!("packed formats are decoded per component"),
        Encoding::SharedExponent => {
            let v = u32::from_le_bytes(bytes.try_into().unwrap());
            let scale = 2f32.powi((v >> 27) as i32 - 24);
            out[0] = [
                (v & 511) as f32 * scale,
                ((v >> 9) & 511) as f32 * scale,
                ((v >> 18) & 511) as f32 * scale,
                1.,
            ];
        }
        Encoding::Astc => astc::decode(
            bytes,
            tic.format.block_width as usize,
            tic.format.block_height as usize,
            tic.srgb,
            out,
        )?,
        encoding => {
            let block: &mut [[f32; 4]; 16] = (&mut out[..16]).try_into().unwrap();
            match encoding {
                Encoding::Bc1 => bc::decode_bc1(bytes, block),
                Encoding::Bc2 => bc::decode_bc2(bytes, block),
                Encoding::Bc3 => bc::decode_bc3(bytes, block),
                Encoding::Bc4 => bc::decode_bc4(bytes, signed, block),
                Encoding::Bc5 => bc::decode_bc5(bytes, signed, block),
                Encoding::Bc7 => bc::decode_bc7(bytes, block),
                _ => unreachable!(),
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
