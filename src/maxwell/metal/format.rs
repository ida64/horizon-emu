//! What the Metal renderer can express: render target formats, blend
//! states, textures and samplers. Anything outside is left to the
//! software renderer.

use objc2_metal::{MTLPixelFormat, MTLSamplerBorderColor};

use super::super::blend::{Blend, Factor};
use super::super::sampling::Target;
use super::super::texture::format::{ColorFormat, DepthFormat};
use super::super::texture::{Texture, Tsc};

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

/// Whether a texture and sampler map onto a Metal 2D texture and sampler
/// state with the interpreter's results: one-layer normalized 2D
/// textures; wrap, mirror, clamp-to-edge, border (with a border color
/// Metal has) and mirror-once-clamp-to-edge addressing; no depth compare,
/// no anisotropy.
pub fn check_texture(texture: &Texture, tsc: &Tsc) -> Result<(), String> {
    let tic = texture.tic();
    if tic.target != Target::Texture2D || tic.array || tic.shape.layers != 1 {
        return Err(format!(
            "texture target {:?} array {} layers {}",
            tic.target, tic.array, tic.shape.layers
        ));
    }
    if !tic.normalized || tsc.force_unnormalized {
        return Err("unnormalized texture coordinates".into());
    }
    if tsc.compare {
        return Err("depth compare sampler".into());
    }
    if tsc.anisotropy > 1 && tsc.min_filter == 3 {
        return Err(format!("anisotropy {}", tsc.anisotropy));
    }
    for &wrap in &tsc.wrap[..2] {
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
