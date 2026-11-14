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

use super::sampling::Target;
use super::{GpuMemory, Unmodeled};
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
        for (index, mask) in [
            (2, 0xe11f0000u32),
            (4, 0x00070000),
            (5, 0x40000000),
            (6, 1),
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

    /// The level-of-detail range sampling may select, as the interpreter
    /// clamps it (`clamp_lod`): (minimum, maximum) relative to the base
    /// level.
    pub fn lod_range(&self, tsc: &Tsc) -> (f32, f32) {
        let resident = ((self.tic.words[7] >> 12) & 4095) as f32 / 256.;
        let top = (self.tic.max_level - self.tic.base_level) as f32;
        (
            tsc.min_lod.max(resident).clamp(0., top),
            tsc.max_lod.max(resident).clamp(0., top),
        )
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

    /// Decodes compressed blocks directly into the final mip buffer. There is no
    /// per-texel allocation. sRGB conversion precedes component swizzle and filtering
    /// [public: OpenGL 4.6 §8.24; NVIDIA clb197tex.h S_R_G_B_CONVERSION].
    pub fn decode(memory: &dyn GpuMemory, tic: &Tic) -> Result<Self, Unmodeled> {
        let (_, size) = Self::footprint(tic)?;
        let mut raw = Vec::new();
        raw.try_reserve_exact(
            usize::try_from(size).map_err(|_| unsupported("texture extent", size))?,
        )
        .map_err(|e| unsupported("texture allocation", e))?;
        raw.resize(size as usize, 0);
        memory.read(tic.address, &mut raw).ok_or_else(|| {
            unsupported(
                "texture memory read",
                format_args!("{:#x}+{size:#x}", tic.address),
            )
        })?;
        let tiled = if tic.pitch.is_none() {
            Some(layout::block_linear_image(&tic.shape, tic.block)?)
        } else {
            None
        };
        let mut levels = Vec::with_capacity(tic.shape.levels as usize);
        let f = tic.format;
        let srgb8 = matches!(
            f.encoding,
            Encoding::Bc1 | Encoding::Bc2 | Encoding::Bc3 | Encoding::Bc7
        ) || matches!(f.encoding, Encoding::Packed(fields) if fields[..3].iter().all(|(_,bits)| matches!(bits,0|8)));
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
            texels.resize(count as usize, [0.; 4]);
            let mut block = [[0.; 4]; 144];
            for layer in 0..tic.shape.layers {
                for z in 0..depth {
                    for by in 0..height.div_ceil(f.block_height) {
                        for bx in 0..width.div_ceil(f.block_width) {
                            let offset = if let Some(image) = &tiled {
                                let m = &image.levels[level as usize];
                                u64::from(layer) * image.layer_stride
                                    + m.offset
                                    + m.surface.offset(bx * f.bytes_per_block, by, z)
                            } else {
                                u64::from(tic.pitch.unwrap())
                                    * (u64::from(layer * depth + z)
                                        * u64::from(height.div_ceil(f.block_height))
                                        + u64::from(by))
                                    + u64::from(bx * f.bytes_per_block)
                            } as usize;
                            let mut split = [0u8; 16];
                            let bytes = if let Some(image) = &tiled
                                && !f.bytes_per_block.is_power_of_two()
                            {
                                let m = &image.levels[level as usize];
                                for byte in 0..f.bytes_per_block {
                                    let address = u64::from(layer) * image.layer_stride
                                        + m.offset
                                        + m.surface.offset(bx * f.bytes_per_block + byte, by, z);
                                    split[byte as usize] =
                                        *raw.get(address as usize).ok_or_else(|| {
                                            unsupported("texture byte offset", address)
                                        })?;
                                }
                                &split[..f.bytes_per_block as usize]
                            } else {
                                raw.get(offset..offset + f.bytes_per_block as usize)
                                    .ok_or_else(|| unsupported("texture block offset", offset))?
                            };
                            decode_block(tic, bytes, &mut block)?;
                            for y in 0..f.block_height.min(height - by * f.block_height) {
                                for x in 0..f.block_width.min(width - bx * f.block_width) {
                                    let mut c = block[(y * f.block_width + x) as usize];
                                    if tic.srgb {
                                        for v in &mut c[..3] {
                                            *v = if srgb8 {
                                                format::SRGB8_TO_LINEAR
                                                    [(*v * 255.).round().clamp(0., 255.) as usize]
                                            } else {
                                                format::srgb_to_linear(*v)
                                            };
                                        }
                                    }
                                    let c = tic.swizzle.map(|s| match s {
                                        0 => 0.,
                                        2..=5 => c[(s - 2) as usize],
                                        6 | 7 => 1.,
                                        _ => unreachable!(),
                                    });
                                    let index = (((layer as usize * depth as usize + z as usize)
                                        * height as usize
                                        + (by * f.block_height + y) as usize)
                                        * width as usize)
                                        + (bx * f.block_width + x) as usize;
                                    texels[index] = c;
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

fn decode_block(tic: &Tic, bytes: &[u8], out: &mut [[f32; 4]; 144]) -> Result<(), Unmodeled> {
    let signed = matches!(
        tic.types[0],
        ComponentType::Snorm | ComponentType::SnormForceFp16
    );
    match tic.format.encoding {
        Encoding::Packed(fields) => {
            let mut word = 0u128;
            for (i, &b) in bytes.iter().enumerate() {
                word |= u128::from(b) << (i * 8);
            }
            let mut c = [0., 0., 0., 1.];
            for (i, (shift, bits)) in fields.into_iter().enumerate() {
                if bits == 0 {
                    continue;
                }
                let v = ((word >> shift) & ((1u128 << bits) - 1)) as u32;
                c[i] = match tic.types[i] {
                    ComponentType::Unorm | ComponentType::UnormForceFp16 => {
                        format::unorm_to_f32(v, bits.into())
                    }
                    ComponentType::Snorm | ComponentType::SnormForceFp16 => {
                        format::snorm_to_f32(v, bits.into())
                    }
                    ComponentType::Float => match bits {
                        10 => format::uf10_to_f32(v),
                        11 => format::uf11_to_f32(v),
                        16 => format::f16_to_f32(v as u16),
                        32 => f32::from_bits(v),
                        _ => return Err(unsupported("TIC FLOAT component bits", bits)),
                    },
                    // Unselected stencil/integer channels cannot affect the result.
                    ComponentType::Uint | ComponentType::Sint => 0.,
                };
                if matches!(
                    tic.types[i],
                    ComponentType::UnormForceFp16 | ComponentType::SnormForceFp16
                ) {
                    c[i] = format::f16_to_f32(format::f32_to_f16(c[i]));
                }
            }
            out[0] = c;
        }
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
