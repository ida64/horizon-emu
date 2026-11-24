//! What the Metal renderer can express: render target formats, blend
//! states, textures and samplers. Anything outside is left to the
//! software renderer.

use objc2_metal::{MTLPixelFormat, MTLSamplerBorderColor};

use super::super::blend::{Blend, Factor};
use super::super::sampling::Target;
use super::super::shader::msl::TextureKind;
use super::super::texture::format::{ColorFormat, DepthFormat};
use super::super::texture::{Tic, Tsc};

/// The Metal pixel format with the same bytes per pixel and component
/// layout as `format` (the linear form for sRGB formats; see [`srgb`]).
/// Formats with an unused component (X8, X16, X32) and A1R5G5B5 are not
/// mapped: Metal has no format writing their filler the way the guest
/// format defines it.
pub fn pixel_format(format: ColorFormat) -> Option<MTLPixelFormat> {
    use ColorFormat::*;
    Some(match format {
        Rgba32Float => MTLPixelFormat::RGBA32Float,
        Rgba16Unorm => MTLPixelFormat::RGBA16Unorm,
        Rgba16Snorm => MTLPixelFormat::RGBA16Snorm,
        Rgba16Float => MTLPixelFormat::RGBA16Float,
        Rg32Float => MTLPixelFormat::RG32Float,
        Bgra8Unorm | Bgra8Srgb => MTLPixelFormat::BGRA8Unorm,
        Rgb10A2Unorm => MTLPixelFormat::RGB10A2Unorm,
        Rgba8Unorm | Rgba8Srgb => MTLPixelFormat::RGBA8Unorm,
        Rgba8Snorm => MTLPixelFormat::RGBA8Snorm,
        Rg16Unorm => MTLPixelFormat::RG16Unorm,
        Rg16Snorm => MTLPixelFormat::RG16Snorm,
        Rg16Float => MTLPixelFormat::RG16Float,
        Bgr10A2Unorm => MTLPixelFormat::BGR10A2Unorm,
        R11G11B10Float => MTLPixelFormat::RG11B10Float,
        R32Float => MTLPixelFormat::R32Float,
        B5G6R5Unorm => MTLPixelFormat::B5G6R5Unorm,
        Rg8Unorm => MTLPixelFormat::RG8Unorm,
        Rg8Snorm => MTLPixelFormat::RG8Snorm,
        R16Unorm => MTLPixelFormat::R16Unorm,
        R16Snorm => MTLPixelFormat::R16Snorm,
        R16Float => MTLPixelFormat::R16Float,
        R8Unorm => MTLPixelFormat::R8Unorm,
        R8Snorm => MTLPixelFormat::R8Snorm,
        _ => return None,
    })
}

/// The Metal pixel format holding a depth buffer of `format` as its raw
/// words, one texel per pixel with the surface's bytes: the depth test
/// reads and writes them in the fragment function with the format's own
/// encoding (`DepthFormat::read_depth`, `write_depth`), so stencil bits
/// and the 24-bit UNORM layouts (which Apple GPUs have no depth format
/// for) are kept exactly.
pub fn depth_pixel_format(format: DepthFormat) -> MTLPixelFormat {
    match format {
        DepthFormat::Z16Unorm => MTLPixelFormat::R16Uint,
        DepthFormat::Z32FloatX24S8 => MTLPixelFormat::RG32Uint,
        DepthFormat::Z32Float | DepthFormat::Z24S8 | DepthFormat::X8Z24 | DepthFormat::S8Z24 => {
            MTLPixelFormat::R32Uint
        }
    }
}

/// The sRGB view format of a linear 8-bit format (Metal converts linear
/// shader output to sRGB on write and back for blending, as a target of
/// an sRGB format does); other formats unchanged.
pub fn srgb(format: MTLPixelFormat) -> MTLPixelFormat {
    match format {
        MTLPixelFormat::RGBA8Unorm => MTLPixelFormat::RGBA8Unorm_sRGB,
        MTLPixelFormat::BGRA8Unorm => MTLPixelFormat::BGRA8Unorm_sRGB,
        other => other,
    }
}

/// Dual-source blending needs a second fragment output, which the
/// translation does not produce.
pub fn check_blend(blend: &Blend) -> Result<(), String> {
    for factor in [
        blend.color_source,
        blend.color_destination,
        blend.alpha_source,
        blend.alpha_destination,
    ] {
        if matches!(
            factor,
            Factor::Source1Color
                | Factor::OneMinusSource1Color
                | Factor::Source1Alpha
                | Factor::OneMinusSource1Alpha
        ) {
            return Err(format!("blend factor {factor:?}"));
        }
    }
    Ok(())
}

/// The Metal border color equal to the sampler's, if there is one
/// (Metal offers only transparent black, opaque black and opaque white).
pub fn border_color(tsc: &Tsc) -> Option<MTLSamplerBorderColor> {
    match tsc.border {
        [0.0, 0.0, 0.0, 0.0] => Some(MTLSamplerBorderColor::TransparentBlack),
        [0.0, 0.0, 0.0, 1.0] => Some(MTLSamplerBorderColor::OpaqueBlack),
        [1.0, 1.0, 1.0, 1.0] => Some(MTLSamplerBorderColor::OpaqueWhite),
        _ => None,
    }
}

/// Whether a texture and sampler map onto a Metal texture of the kind a
/// slot declares and a sampler state with the interpreter's results:
/// normalized textures of the slot's target (one layer for a 2D slot,
/// six for a cube); wrap, mirror, clamp-to-edge, border (with a border
/// color Metal has) and mirror-once-clamp-to-edge addressing. A depth
/// comparison is the translated program's own (`compare_sample`), so the
/// sampler state has none. Anisotropic filtering is Metal's, with the
/// sampler's limit: GM20B's tap placement is not documented.
pub fn check_texture(tic: &Tic, tsc: &Tsc, kind: TextureKind) -> Result<(), String> {
    let fits = match kind {
        // A 2D sample reads layer 0, of an array texture too (as the
        // interpreter's `texture::filtering` reads it).
        TextureKind::Flat => tic.target == Target::Texture2D && tic.shape.layers == 1,
        TextureKind::Array => tic.target == Target::Texture2D,
        TextureKind::Cube => tic.target == Target::Cube && tic.shape.layers == 6,
        TextureKind::Volume => tic.target == Target::Texture3D && tic.shape.layers == 1,
    };
    if !fits {
        return Err(format!(
            "texture target {:?} array {} layers {} in a {kind:?} slot",
            tic.target, tic.array, tic.shape.layers
        ));
    }
    if tic.integer() {
        return Err("integer texture".into());
    }
    if !tic.normalized || tsc.force_unnormalized {
        return Err("unnormalized texture coordinates".into());
    }
    let axes = match kind {
        TextureKind::Volume => 3,
        // Metal filters cubes across their edges whatever the sampler
        // says.
        TextureKind::Cube => 0,
        _ => 2,
    };
    for &wrap in &tsc.wrap[..axes] {
        match wrap {
            0..=2 | 5 => {}
            3 => {
                if border_color(tsc).is_none() {
                    return Err(format!("border color {:?}", tsc.border));
                }
                if tsc.srgb_conversion && tic.srgb {
                    return Err("sRGB border color".into());
                }
            }
            other => return Err(format!("address mode {other}")),
        }
    }
    Ok(())
}

/// The Metal pixel format that samples a texture's bytes as its header
/// says (the header's swizzle is the Metal texture's), if there is one:
/// the block-compressed formats and the packed ones Metal has, with
/// UNORM, SNORM or float components. Metal decodes BC1-BC3 and ASTC
/// itself, which can differ from GM20B's decoding in the last bit of a
/// component [known difference]. Volumes are taken uncompressed only.
pub fn native_format(tic: &Tic, kind: TextureKind) -> Option<MTLPixelFormat> {
    use super::super::texture::format::{ComponentType, Encoding};
    use MTLPixelFormat as F;
    let format = tic.format;
    let present = format.present();
    let mut types = (0..4).filter(|&i| present[i]).map(|i| tic.types[i]);
    let first = types.next()?;
    if types.any(|t| t != first) {
        return None;
    }
    let compressed = !matches!(format.encoding, Encoding::Packed(_) | Encoding::SharedExponent);
    if compressed && kind == TextureKind::Volume {
        return None;
    }
    let srgb = tic.srgb;
    let pick = |linear: F, encoded: F| if srgb { encoded } else { linear };
    let unorm = first == ComponentType::Unorm;
    Some(match (format.encoding, first) {
        (Encoding::Bc1, _) => pick(F::BC1_RGBA, F::BC1_RGBA_sRGB),
        (Encoding::Bc2, _) => pick(F::BC2_RGBA, F::BC2_RGBA_sRGB),
        (Encoding::Bc3, _) => pick(F::BC3_RGBA, F::BC3_RGBA_sRGB),
        (Encoding::Bc7, _) => pick(F::BC7_RGBAUnorm, F::BC7_RGBAUnorm_sRGB),
        (Encoding::Bc4, _) if srgb => return None,
        (Encoding::Bc5, _) if srgb => return None,
        (Encoding::Bc4, ComponentType::Snorm) => F::BC4_RSnorm,
        (Encoding::Bc4, _) => F::BC4_RUnorm,
        (Encoding::Bc5, ComponentType::Snorm) => F::BC5_RGSnorm,
        (Encoding::Bc5, _) => F::BC5_RGUnorm,
        (Encoding::Astc, _) => match (format.block_width, format.block_height) {
            (4, 4) => pick(F::ASTC_4x4_LDR, F::ASTC_4x4_sRGB),
            (5, 4) => pick(F::ASTC_5x4_LDR, F::ASTC_5x4_sRGB),
            (5, 5) => pick(F::ASTC_5x5_LDR, F::ASTC_5x5_sRGB),
            (6, 5) => pick(F::ASTC_6x5_LDR, F::ASTC_6x5_sRGB),
            (6, 6) => pick(F::ASTC_6x6_LDR, F::ASTC_6x6_sRGB),
            (8, 5) => pick(F::ASTC_8x5_LDR, F::ASTC_8x5_sRGB),
            (8, 6) => pick(F::ASTC_8x6_LDR, F::ASTC_8x6_sRGB),
            (8, 8) => pick(F::ASTC_8x8_LDR, F::ASTC_8x8_sRGB),
            (10, 5) => pick(F::ASTC_10x5_LDR, F::ASTC_10x5_sRGB),
            (10, 6) => pick(F::ASTC_10x6_LDR, F::ASTC_10x6_sRGB),
            (10, 8) => pick(F::ASTC_10x8_LDR, F::ASTC_10x8_sRGB),
            (10, 10) => pick(F::ASTC_10x10_LDR, F::ASTC_10x10_sRGB),
            (12, 10) => pick(F::ASTC_12x10_LDR, F::ASTC_12x10_sRGB),
            (12, 12) => pick(F::ASTC_12x12_LDR, F::ASTC_12x12_sRGB),
            _ => return None,
        },
        (Encoding::SharedExponent, ComponentType::Float) if !srgb => F::RGB9E5Float,
        (Encoding::Packed(_), _) => match (format.raw, first) {
            (0x08, ComponentType::Unorm) => pick(F::RGBA8Unorm, F::RGBA8Unorm_sRGB),
            // Depth buffers sampled as textures: ZF32 and Z16.
            (0x2f, ComponentType::Float) if !srgb => F::R32Float,
            (0x3a, ComponentType::Unorm) if !srgb => F::R16Unorm,
            _ if srgb => return None,
            (0x01, ComponentType::Float) => F::RGBA32Float,
            (0x03, ComponentType::Unorm) => F::RGBA16Unorm,
            (0x03, ComponentType::Snorm) => F::RGBA16Snorm,
            (0x03, ComponentType::Float) => F::RGBA16Float,
            (0x04, ComponentType::Float) => F::RG32Float,
            (0x08, ComponentType::Snorm) => F::RGBA8Snorm,
            (0x09, ComponentType::Unorm) => F::RGB10A2Unorm,
            (0x0c, ComponentType::Unorm) => F::RG16Unorm,
            (0x0c, ComponentType::Snorm) => F::RG16Snorm,
            (0x0c, ComponentType::Float) => F::RG16Float,
            (0x0f, ComponentType::Float) => F::R32Float,
            (0x18, ComponentType::Unorm) => F::RG8Unorm,
            (0x18, ComponentType::Snorm) => F::RG8Snorm,
            (0x1b, ComponentType::Unorm) => F::R16Unorm,
            (0x1b, ComponentType::Snorm) => F::R16Snorm,
            (0x1b, ComponentType::Float) => F::R16Float,
            (0x1d, ComponentType::Unorm) => F::R8Unorm,
            (0x1d, ComponentType::Snorm) => F::R8Snorm,
            (0x21, ComponentType::Float) => F::RG11B10Float,
            _ => return None,
        },
        _ => return None,
    })
    .filter(|_| unorm || !compressed || matches!(format.encoding, Encoding::Bc4 | Encoding::Bc5))
}
