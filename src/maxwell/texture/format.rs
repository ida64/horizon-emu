//! Texel formats: the component layouts a TIC names, the render-target
//! color formats of the 3D/2D engines, and the depth/stencil ("zeta")
//! formats.
//!
//! Number conversions follow the D3D11.3 functional specification, which
//! Maxwell-class hardware implements [public: D3D11.3 functional spec
//! §3.2 "Data Conversion"]:
//! * UNORM -> float: `v / (2^n - 1)` (§3.2.3.5); SNORM -> float:
//!   `v / (2^(n-1) - 1)`, the most negative code also giving -1 (§3.2.3.3).
//! * float -> UNORM: NaN -> 0, clamp to [0, 1], `floor(c * (2^n - 1) + 0.5)`
//!   (round half up, §3.2.3.6); float -> SNORM: NaN -> 0, clamp to [-1, 1],
//!   scale by `2^(n-1) - 1`, add +-0.5 away from zero and truncate (§3.2.3.4).
//!   The spec allows 0.6 ULP of slack; the ideal conversion is used.
//! * float32 -> smaller floats (fp16, the unsigned 11/10-bit floats):
//!   round toward zero; finite values beyond the range give the largest
//!   finite value, infinities stay infinite, NaN stays NaN, denormals are
//!   kept; the unsigned floats clamp negatives to 0 (§3.2.2, §3.1 "11-bit
//!   and 10-bit Floating Point").
//! * sRGB: the ideal curves of §3.2.3.7 (sRGB -> float) and §3.2.3.8
//!   (float -> sRGB). Hardware uses tables within the spec's tolerance; the
//!   exact table of GM20B is not public.

use std::sync::LazyLock;

use crate::maxwell::Unmodeled;

/// Decodes an IEEE binary16 value.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exponent = u32::from((h >> 10) & 0x1f);
    let mantissa = u32::from(h & 0x3ff);
    let bits = match exponent {
        0 if mantissa == 0 => sign,
        // Denormal: value = mantissa * 2^-24, exact in f32.
        0 => sign | ((mantissa as f32) * (1.0 / 16_777_216.0)).to_bits(),
        0x1f => sign | 0x7f80_0000 | (mantissa << 13),
        _ => sign | ((exponent + 127 - 15) << 23) | (mantissa << 13),
    };
    f32::from_bits(bits)
}

/// Encodes an f32 as binary16, rounding toward zero (D3D11.3 §3.2.2).
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;
    if exponent == 0xff {
        return if mantissa != 0 {
            sign | 0x7e00
        } else {
            sign | 0x7c00
        };
    }
    let unbiased = exponent - 127;
    if unbiased > 15 {
        return sign | 0x7bff;
    }
    if unbiased >= -14 {
        return sign | (((unbiased + 15) as u16) << 10) | (mantissa >> 13) as u16;
    }
    // Denormal result (or zero): mantissa with its hidden bit, shifted.
    if exponent == 0 || unbiased < -25 {
        return sign;
    }
    let full = mantissa | 0x80_0000;
    sign | (full >> (13 + (-14 - unbiased))) as u16
}

/// Decodes an unsigned float with a 5-bit exponent and `mantissa_bits` of
/// mantissa (the 11-bit and 10-bit components of `R11G11B10F`).
fn small_float_to_f32(value: u32, mantissa_bits: u32) -> f32 {
    let exponent = value >> mantissa_bits;
    let mantissa = value & ((1 << mantissa_bits) - 1);
    match exponent {
        0 => mantissa as f32 * 2f32.powi(-14 - mantissa_bits as i32),
        31 if mantissa == 0 => f32::INFINITY,
        31 => f32::NAN,
        _ => f32::from_bits(((exponent + 127 - 15) << 23) | (mantissa << (23 - mantissa_bits))),
    }
}

/// Encodes into an unsigned float with a 5-bit exponent, rounding toward
/// zero (D3D11.3 §3.1 "11-bit and 10-bit Floating Point", §3.2.2).
fn f32_to_small_float(value: f32, mantissa_bits: u32) -> u32 {
    if value.is_nan() {
        return (31 << mantissa_bits) | 1;
    }
    if value <= 0.0 {
        return 0;
    }
    if value == f32::INFINITY {
        return 31 << mantissa_bits;
    }
    let bits = value.to_bits();
    let unbiased = ((bits >> 23) & 0xff) as i32 - 127;
    let mantissa = bits & 0x7f_ffff;
    if unbiased > 15 {
        return (30 << mantissa_bits) | ((1 << mantissa_bits) - 1);
    }
    if unbiased >= -14 {
        return (((unbiased + 15) as u32) << mantissa_bits) | (mantissa >> (23 - mantissa_bits));
    }
    if unbiased < -14 - mantissa_bits as i32 - 1 {
        return 0;
    }
    (mantissa | 0x80_0000) >> (23 - mantissa_bits + (-14 - unbiased) as u32)
}

/// Decodes the 11-bit unsigned float of `R11G11B10F`.
pub fn uf11_to_f32(value: u32) -> f32 {
    small_float_to_f32(value & 0x7ff, 6)
}

/// Decodes the 10-bit unsigned float of `R11G11B10F`.
pub fn uf10_to_f32(value: u32) -> f32 {
    small_float_to_f32(value & 0x3ff, 5)
}

/// Encodes an 11-bit unsigned float (round toward zero).
pub fn f32_to_uf11(value: f32) -> u32 {
    f32_to_small_float(value, 6)
}

/// Encodes a 10-bit unsigned float (round toward zero).
pub fn f32_to_uf10(value: f32) -> u32 {
    f32_to_small_float(value, 5)
}

/// sRGB-encoded value to linear (D3D11.3 §3.2.3.7).
pub fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

/// Linear value to sRGB encoding, before quantization (D3D11.3 §3.2.3.8:
/// NaN gives 0, the input is clamped to [0, 1]).
pub fn linear_to_srgb(value: f32) -> f32 {
    if value.is_nan() {
        return 0.0;
    }
    let value = value.clamp(0.0, 1.0);
    if value <= 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

/// [`srgb_to_linear`] of the 256 8-bit codes.
pub(crate) static SRGB8_TO_LINEAR: LazyLock<[f32; 256]> =
    LazyLock::new(|| std::array::from_fn(|code| srgb_to_linear(code as f32 / 255.0)));

/// UNORM code of `bits` bits to float (D3D11.3 §3.2.3.5).
#[inline]
pub fn unorm_to_f32(value: u32, bits: u32) -> f32 {
    (f64::from(value) / ((1u64 << bits) - 1) as f64) as f32
}

/// SNORM code of `bits` bits (two's complement in the low bits) to float
/// (D3D11.3 §3.2.3.3).
#[inline]
pub fn snorm_to_f32(value: u32, bits: u32) -> f32 {
    let signed = ((value << (32 - bits)) as i32) >> (32 - bits);
    let max = ((1u32 << (bits - 1)) - 1) as f64;
    (f64::from(signed) / max).max(-1.0) as f32
}

/// Float to UNORM code of `bits` bits (D3D11.3 §3.2.3.6).
#[inline]
pub fn f32_to_unorm(value: f32, bits: u32) -> u32 {
    if value.is_nan() {
        return 0;
    }
    let max = ((1u64 << bits) - 1) as f64;
    (f64::from(value.clamp(0.0, 1.0)) * max + 0.5).floor() as u32
}

/// Float to SNORM code of `bits` bits, in the low bits (D3D11.3 §3.2.3.4).
#[inline]
pub fn f32_to_snorm(value: f32, bits: u32) -> u32 {
    if value.is_nan() {
        return 0;
    }
    let max = ((1u64 << (bits - 1)) - 1) as f64;
    let scaled = f64::from(value.clamp(-1.0, 1.0)) * max;
    let rounded = if scaled >= 0.0 {
        scaled + 0.5
    } else {
        scaled - 0.5
    }
    .trunc() as i32;
    (rounded as u32) & ((1u64 << bits) - 1) as u32
}

/// How a render-target component is stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Numeric {
    Unorm,
    Snorm,
    Float,
    /// Unsigned integers: a component is its bits, carried in the `f32`
    /// of a color unchanged ([`ColorFormat::read`], [`ColorFormat::write`]).
    Uint,
    /// Signed integers, the same way, sign-extended to 32 bits.
    Sint,
}

/// What the unused bits of an `X` format receive on write.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Filler {
    /// No `X` bits.
    None,
    /// `Xn` formats: the hardware leaves the bits unspecified; the alpha
    /// value is written there \[assumption\].
    Alpha,
    /// `Zn` formats: zeros.
    Zero,
    /// `On` formats: ones.
    One,
}

/// Render-target color formats as the 3D engine's `SET_COLOR_TARGET_FORMAT`
/// and the 2D engine's `SET_SRC_FORMAT`/`SET_DST_FORMAT` registers encode
/// them [public: NVIDIA open-gpu-doc `classes/3d/clb197.h`
/// `NVB197_SET_COLOR_TARGET_FORMAT_V_*`, `classes/twod/cl902d.h`
/// `NV902D_SET_DST_FORMAT_V_*` (same codes)].
///
/// Names list components in memory order, lowest address first (the NVIDIA
/// names list a packed word from its most significant bits, so NVIDIA
/// `A8B8G8R8` is [`ColorFormat::Rgba8Unorm`] here). The alpha-only/VCAA
/// variants (0x40-0x47, `R32` 0xFF) and `AU2BU10GU10RU10` (0xD2) are not
/// modeled: [`ColorFormat::from_raw`] returns `None` for them.
///
/// Integer formats (`RSn`, `RUn`): a fragment program's output register
/// holds the integer, and it is stored clamped to the component's range
/// \[recalled: D3D11.3 functional spec §3.2.4 "Integer Conversion", a
/// wider integer converted to a narrower one is clamped\].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorFormat {
    /// `RF32_GF32_BF32_AF32` (0xC0).
    Rgba32Float,
    /// `RF32_GF32_BF32_X32` (0xC3).
    Rgbx32Float,
    /// `R16_G16_B16_A16` (0xC6).
    Rgba16Unorm,
    /// `RN16_GN16_BN16_AN16` (0xC7).
    Rgba16Snorm,
    /// `RF16_GF16_BF16_AF16` (0xCA).
    Rgba16Float,
    /// `RF32_GF32` (0xCB).
    Rg32Float,
    /// `RF16_GF16_BF16_X16` (0xCE).
    Rgbx16Float,
    /// `A8R8G8B8` (0xCF).
    Bgra8Unorm,
    /// `A8RL8GL8BL8` (0xD0).
    Bgra8Srgb,
    /// `A2B10G10R10` (0xD1).
    Rgb10A2Unorm,
    /// `A8B8G8R8` (0xD5).
    Rgba8Unorm,
    /// `A8BL8GL8RL8` (0xD6).
    Rgba8Srgb,
    /// `AN8BN8GN8RN8` (0xD7).
    Rgba8Snorm,
    /// `R16_G16` (0xDA).
    Rg16Unorm,
    /// `RN16_GN16` (0xDB).
    Rg16Snorm,
    /// `RF16_GF16` (0xDE).
    Rg16Float,
    /// `A2R10G10B10` (0xDF).
    Bgr10A2Unorm,
    /// `BF10GF11RF11` (0xE0).
    R11G11B10Float,
    /// `RF32` (0xE5).
    R32Float,
    /// `X8R8G8B8` (0xE6).
    Bgrx8Unorm,
    /// `X8RL8GL8BL8` (0xE7).
    Bgrx8Srgb,
    /// `R5G6B5` (0xE8): blue in bits 0-4, green 5-10, red 11-15.
    B5G6R5Unorm,
    /// `A1R5G5B5` (0xE9).
    Bgr5A1Unorm,
    /// `G8R8` (0xEA).
    Rg8Unorm,
    /// `GN8RN8` (0xEB).
    Rg8Snorm,
    /// `R16` (0xEE).
    R16Unorm,
    /// `RN16` (0xEF).
    R16Snorm,
    /// `RF16` (0xF2).
    R16Float,
    /// `R8` (0xF3).
    R8Unorm,
    /// `RN8` (0xF4).
    R8Snorm,
    /// `A8` (0xF7): alpha only.
    A8Unorm,
    /// `X1R5G5B5` (0xF8).
    Bgr5X1Unorm,
    /// `X8B8G8R8` (0xF9).
    Rgbx8Unorm,
    /// `X8BL8GL8RL8` (0xFA).
    Rgbx8Srgb,
    /// `Z1R5G5B5` (0xFB): the X bit written as 0.
    Bgr5Z1Unorm,
    /// `O1R5G5B5` (0xFC): the X bit written as 1.
    Bgr5O1Unorm,
    /// `Z8R8G8B8` (0xFD): the X byte written as 0.
    Bgrz8Unorm,
    /// `O8R8G8B8` (0xFE): the X byte written as 0xFF.
    Bgro8Unorm,
    /// `RS32_GS32_BS32_AS32` (0xC1).
    Rgba32Sint,
    /// `RU32_GU32_BU32_AU32` (0xC2).
    Rgba32Uint,
    /// `RS32_GS32_BS32_X32` (0xC4).
    Rgbx32Sint,
    /// `RU32_GU32_BU32_X32` (0xC5).
    Rgbx32Uint,
    /// `RS16_GS16_BS16_AS16` (0xC8).
    Rgba16Sint,
    /// `RU16_GU16_BU16_AU16` (0xC9).
    Rgba16Uint,
    /// `RS32_GS32` (0xCC).
    Rg32Sint,
    /// `RU32_GU32` (0xCD).
    Rg32Uint,
    /// `AS8BS8GS8RS8` (0xD8).
    Rgba8Sint,
    /// `AU8BU8GU8RU8` (0xD9).
    Rgba8Uint,
    /// `RS16_GS16` (0xDC).
    Rg16Sint,
    /// `RU16_GU16` (0xDD).
    Rg16Uint,
    /// `RS32` (0xE3).
    R32Sint,
    /// `RU32` (0xE4).
    R32Uint,
    /// `GS8RS8` (0xEC).
    Rg8Sint,
    /// `GU8RU8` (0xED).
    Rg8Uint,
    /// `RS16` (0xF0).
    R16Sint,
    /// `RU16` (0xF1).
    R16Uint,
    /// `RS8` (0xF5).
    R8Sint,
    /// `RU8` (0xF6).
    R8Uint,
}

/// Bit position and width of R, G, B, A in a texel's little-endian word
/// (width 0: absent).
type Fields = [(u8, u8); 4];

struct ColorLayout {
    bytes: u32,
    numeric: Numeric,
    fields: Fields,
    /// Bit position and width of the `X` bits.
    filler: (Filler, u8, u8),
    srgb: bool,
}

const NO_FILL: (Filler, u8, u8) = (Filler::None, 0, 0);

impl ColorFormat {
    /// Decodes a `SET_COLOR_TARGET_FORMAT`/2D `FORMAT` value.
    pub fn from_raw(raw: u32) -> Option<ColorFormat> {
        use ColorFormat::*;
        Some(match raw {
            0xc0 => Rgba32Float,
            0xc3 => Rgbx32Float,
            0xc6 => Rgba16Unorm,
            0xc7 => Rgba16Snorm,
            0xca => Rgba16Float,
            0xcb => Rg32Float,
            0xce => Rgbx16Float,
            0xcf => Bgra8Unorm,
            0xd0 => Bgra8Srgb,
            0xd1 => Rgb10A2Unorm,
            0xd5 => Rgba8Unorm,
            0xd6 => Rgba8Srgb,
            0xd7 => Rgba8Snorm,
            0xda => Rg16Unorm,
            0xdb => Rg16Snorm,
            0xde => Rg16Float,
            0xdf => Bgr10A2Unorm,
            0xe0 => R11G11B10Float,
            0xe5 => R32Float,
            0xe6 => Bgrx8Unorm,
            0xe7 => Bgrx8Srgb,
            0xe8 => B5G6R5Unorm,
            0xe9 => Bgr5A1Unorm,
            0xea => Rg8Unorm,
            0xeb => Rg8Snorm,
            0xee => R16Unorm,
            0xef => R16Snorm,
            0xf2 => R16Float,
            0xf3 => R8Unorm,
            0xf4 => R8Snorm,
            0xf7 => A8Unorm,
            0xf8 => Bgr5X1Unorm,
            0xf9 => Rgbx8Unorm,
            0xfa => Rgbx8Srgb,
            0xfb => Bgr5Z1Unorm,
            0xfc => Bgr5O1Unorm,
            0xfd => Bgrz8Unorm,
            0xfe => Bgro8Unorm,
            0xc1 => Rgba32Sint,
            0xc2 => Rgba32Uint,
            0xc4 => Rgbx32Sint,
            0xc5 => Rgbx32Uint,
            0xc8 => Rgba16Sint,
            0xc9 => Rgba16Uint,
            0xcc => Rg32Sint,
            0xcd => Rg32Uint,
            0xd8 => Rgba8Sint,
            0xd9 => Rgba8Uint,
            0xdc => Rg16Sint,
            0xdd => Rg16Uint,
            0xe3 => R32Sint,
            0xe4 => R32Uint,
            0xec => Rg8Sint,
            0xed => Rg8Uint,
            0xf0 => R16Sint,
            0xf1 => R16Uint,
            0xf5 => R8Sint,
            0xf6 => R8Uint,
            _ => return None,
        })
    }

    fn layout(self) -> ColorLayout {
        use ColorFormat::*;
        use Numeric::*;
        let (bytes, numeric, fields, filler, srgb) = match self {
            Rgba32Float => (
                16,
                Float,
                [(0, 32), (32, 32), (64, 32), (96, 32)],
                NO_FILL,
                false,
            ),
            Rgbx32Float => (
                16,
                Float,
                [(0, 32), (32, 32), (64, 32), (0, 0)],
                (Filler::Alpha, 96, 32),
                false,
            ),
            Rgba16Unorm => (
                8,
                Unorm,
                [(0, 16), (16, 16), (32, 16), (48, 16)],
                NO_FILL,
                false,
            ),
            Rgba16Snorm => (
                8,
                Snorm,
                [(0, 16), (16, 16), (32, 16), (48, 16)],
                NO_FILL,
                false,
            ),
            Rgba16Float => (
                8,
                Float,
                [(0, 16), (16, 16), (32, 16), (48, 16)],
                NO_FILL,
                false,
            ),
            Rg32Float => (
                8,
                Float,
                [(0, 32), (32, 32), (0, 0), (0, 0)],
                NO_FILL,
                false,
            ),
            Rgbx16Float => (
                8,
                Float,
                [(0, 16), (16, 16), (32, 16), (0, 0)],
                (Filler::Alpha, 48, 16),
                false,
            ),
            Bgra8Unorm => (4, Unorm, [(16, 8), (8, 8), (0, 8), (24, 8)], NO_FILL, false),
            Bgra8Srgb => (4, Unorm, [(16, 8), (8, 8), (0, 8), (24, 8)], NO_FILL, true),
            Rgb10A2Unorm => (
                4,
                Unorm,
                [(0, 10), (10, 10), (20, 10), (30, 2)],
                NO_FILL,
                false,
            ),
            Rgba8Unorm => (4, Unorm, [(0, 8), (8, 8), (16, 8), (24, 8)], NO_FILL, false),
            Rgba8Srgb => (4, Unorm, [(0, 8), (8, 8), (16, 8), (24, 8)], NO_FILL, true),
            Rgba8Snorm => (4, Snorm, [(0, 8), (8, 8), (16, 8), (24, 8)], NO_FILL, false),
            Rg16Unorm => (
                4,
                Unorm,
                [(0, 16), (16, 16), (0, 0), (0, 0)],
                NO_FILL,
                false,
            ),
            Rg16Snorm => (
                4,
                Snorm,
                [(0, 16), (16, 16), (0, 0), (0, 0)],
                NO_FILL,
                false,
            ),
            Rg16Float => (
                4,
                Float,
                [(0, 16), (16, 16), (0, 0), (0, 0)],
                NO_FILL,
                false,
            ),
            Bgr10A2Unorm => (
                4,
                Unorm,
                [(20, 10), (10, 10), (0, 10), (30, 2)],
                NO_FILL,
                false,
            ),
            R11G11B10Float => (
                4,
                Float,
                [(0, 11), (11, 11), (22, 10), (0, 0)],
                NO_FILL,
                false,
            ),
            R32Float => (4, Float, [(0, 32), (0, 0), (0, 0), (0, 0)], NO_FILL, false),
            Bgrx8Unorm => (
                4,
                Unorm,
                [(16, 8), (8, 8), (0, 8), (0, 0)],
                (Filler::Alpha, 24, 8),
                false,
            ),
            Bgrx8Srgb => (
                4,
                Unorm,
                [(16, 8), (8, 8), (0, 8), (0, 0)],
                (Filler::Alpha, 24, 8),
                true,
            ),
            B5G6R5Unorm => (2, Unorm, [(11, 5), (5, 6), (0, 5), (0, 0)], NO_FILL, false),
            Bgr5A1Unorm => (2, Unorm, [(10, 5), (5, 5), (0, 5), (15, 1)], NO_FILL, false),
            Rg8Unorm => (2, Unorm, [(0, 8), (8, 8), (0, 0), (0, 0)], NO_FILL, false),
            Rg8Snorm => (2, Snorm, [(0, 8), (8, 8), (0, 0), (0, 0)], NO_FILL, false),
            R16Unorm => (2, Unorm, [(0, 16), (0, 0), (0, 0), (0, 0)], NO_FILL, false),
            R16Snorm => (2, Snorm, [(0, 16), (0, 0), (0, 0), (0, 0)], NO_FILL, false),
            R16Float => (2, Float, [(0, 16), (0, 0), (0, 0), (0, 0)], NO_FILL, false),
            R8Unorm => (1, Unorm, [(0, 8), (0, 0), (0, 0), (0, 0)], NO_FILL, false),
            R8Snorm => (1, Snorm, [(0, 8), (0, 0), (0, 0), (0, 0)], NO_FILL, false),
            A8Unorm => (1, Unorm, [(0, 0), (0, 0), (0, 0), (0, 8)], NO_FILL, false),
            Bgr5X1Unorm => (
                2,
                Unorm,
                [(10, 5), (5, 5), (0, 5), (0, 0)],
                (Filler::Alpha, 15, 1),
                false,
            ),
            Rgbx8Unorm => (
                4,
                Unorm,
                [(0, 8), (8, 8), (16, 8), (0, 0)],
                (Filler::Alpha, 24, 8),
                false,
            ),
            Rgbx8Srgb => (
                4,
                Unorm,
                [(0, 8), (8, 8), (16, 8), (0, 0)],
                (Filler::Alpha, 24, 8),
                true,
            ),
            Bgr5Z1Unorm => (
                2,
                Unorm,
                [(10, 5), (5, 5), (0, 5), (0, 0)],
                (Filler::Zero, 15, 1),
                false,
            ),
            Bgr5O1Unorm => (
                2,
                Unorm,
                [(10, 5), (5, 5), (0, 5), (0, 0)],
                (Filler::One, 15, 1),
                false,
            ),
            Bgrz8Unorm => (
                4,
                Unorm,
                [(16, 8), (8, 8), (0, 8), (0, 0)],
                (Filler::Zero, 24, 8),
                false,
            ),
            Bgro8Unorm => (
                4,
                Unorm,
                [(16, 8), (8, 8), (0, 8), (0, 0)],
                (Filler::One, 24, 8),
                false,
            ),
            Rgba32Sint | Rgba32Uint | Rgbx32Sint | Rgbx32Uint | Rgba16Sint | Rgba16Uint
            | Rg32Sint | Rg32Uint | Rgba8Sint | Rgba8Uint | Rg16Sint | Rg16Uint | R32Sint
            | R32Uint | Rg8Sint | Rg8Uint | R16Sint | R16Uint | R8Sint | R8Uint => {
                let (bits, count, signed, filler): (u8, u8, bool, bool) = match self {
                    Rgba32Sint => (32, 4, true, false),
                    Rgba32Uint => (32, 4, false, false),
                    Rgbx32Sint => (32, 3, true, true),
                    Rgbx32Uint => (32, 3, false, true),
                    Rgba16Sint => (16, 4, true, false),
                    Rgba16Uint => (16, 4, false, false),
                    Rg32Sint => (32, 2, true, false),
                    Rg32Uint => (32, 2, false, false),
                    Rgba8Sint => (8, 4, true, false),
                    Rgba8Uint => (8, 4, false, false),
                    Rg16Sint => (16, 2, true, false),
                    Rg16Uint => (16, 2, false, false),
                    R32Sint => (32, 1, true, false),
                    R32Uint => (32, 1, false, false),
                    Rg8Sint => (8, 2, true, false),
                    Rg8Uint => (8, 2, false, false),
                    R16Sint => (16, 1, true, false),
                    R16Uint => (16, 1, false, false),
                    R8Sint => (8, 1, true, false),
                    _ => (8, 1, false, false),
                };
                let mut fields = [(0, 0); 4];
                for (index, field) in fields.iter_mut().enumerate().take(count.into()) {
                    *field = (index as u8 * bits, bits);
                }
                (
                    u32::from(bits / 8) * if filler { 4 } else { u32::from(count) },
                    if signed { Sint } else { Uint },
                    fields,
                    if filler {
                        (Filler::Alpha, 3 * bits, bits)
                    } else {
                        NO_FILL
                    },
                    false,
                )
            }
        };
        ColorLayout {
            bytes,
            numeric,
            fields,
            filler,
            srgb,
        }
    }

    /// Bytes per pixel.
    pub fn bytes_per_pixel(self) -> u32 {
        self.layout().bytes
    }

    /// Whether the format is an sRGB (`*L8*`) format. [`ColorFormat::read`]
    /// and [`ColorFormat::write`] move raw UNORM values for these; the 3D
    /// engine applies the transfer function itself where its sRGB-write
    /// state asks for it.
    pub fn is_srgb(self) -> bool {
        self.layout().srgb
    }

    /// Whether the components are integers: neither blended nor
    /// converted, see the type's documentation.
    pub fn is_integer(self) -> bool {
        matches!(self.layout().numeric, Numeric::Uint | Numeric::Sint)
    }

    /// Reads one pixel. Components the format lacks read as 0 (RGB) and
    /// 1 (alpha; the integer 1 for an integer format), D3D's defaults for
    /// missing components [public: D3D11.3 functional spec §19.1.3.3
    /// "Defaults for Missing Components"].
    pub fn read(self, texel: &[u8]) -> [f32; 4] {
        let layout = self.layout();
        let word = load_word(&texel[..layout.bytes as usize]);
        let mut out = [0.0, 0.0, 0.0, 1.0];
        if self.is_integer() {
            out[3] = f32::from_bits(1);
        }
        for (slot, &(shift, bits)) in out.iter_mut().zip(&layout.fields) {
            if bits == 0 {
                continue;
            }
            let raw = ((word >> shift) & ((1u128 << bits) - 1)) as u32;
            *slot = match layout.numeric {
                Numeric::Unorm => unorm_to_f32(raw, u32::from(bits)),
                Numeric::Snorm => snorm_to_f32(raw, u32::from(bits)),
                Numeric::Float => decode_float(raw, bits),
                Numeric::Uint => f32::from_bits(raw),
                Numeric::Sint => {
                    f32::from_bits(((raw << (32 - bits)) as i32 >> (32 - bits)) as u32)
                }
            };
        }
        out
    }

    /// Writes one pixel (D3D11.3 conversions, see the module docs).
    pub fn write(self, value: [f32; 4], texel: &mut [u8]) {
        let layout = self.layout();
        let mut word = 0u128;
        let mut put = |raw: u32, shift: u8, bits: u8| {
            word |= u128::from(raw & ((1u64 << bits) - 1) as u32) << shift;
        };
        for (&v, &(shift, bits)) in value.iter().zip(&layout.fields) {
            if bits == 0 {
                continue;
            }
            let raw = match layout.numeric {
                Numeric::Unorm => f32_to_unorm(v, u32::from(bits)),
                Numeric::Snorm => f32_to_snorm(v, u32::from(bits)),
                Numeric::Float => encode_float(v, bits),
                Numeric::Uint => encode_integer(v, bits, false),
                Numeric::Sint => encode_integer(v, bits, true),
            };
            put(raw, shift, bits);
        }
        let (filler, shift, bits) = layout.filler;
        let raw = match filler {
            Filler::None | Filler::Zero => 0,
            Filler::One => u32::MAX,
            Filler::Alpha => match layout.numeric {
                Numeric::Float => encode_float(value[3], bits),
                Numeric::Uint => encode_integer(value[3], bits, false),
                Numeric::Sint => encode_integer(value[3], bits, true),
                _ => f32_to_unorm(value[3], u32::from(bits)),
            },
        };
        if filler != Filler::None {
            put(raw, shift, bits);
        }
        texel[..layout.bytes as usize]
            .copy_from_slice(&word.to_le_bytes()[..layout.bytes as usize]);
    }
}

fn load_word(bytes: &[u8]) -> u128 {
    let mut buffer = [0u8; 16];
    buffer[..bytes.len()].copy_from_slice(bytes);
    u128::from_le_bytes(buffer)
}

fn decode_float(raw: u32, bits: u8) -> f32 {
    match bits {
        32 => f32::from_bits(raw),
        16 => f16_to_f32(raw as u16),
        11 => uf11_to_f32(raw),
        _ => uf10_to_f32(raw),
    }
}

fn encode_float(value: f32, bits: u8) -> u32 {
    match bits {
        32 => value.to_bits(),
        16 => u32::from(f32_to_f16(value)),
        11 => f32_to_uf11(value),
        _ => f32_to_uf10(value),
    }
}

/// The integer whose bits `value` carries, clamped to a component of
/// `bits` bits.
fn encode_integer(value: f32, bits: u8, signed: bool) -> u32 {
    let raw = value.to_bits();
    if bits == 32 {
        raw
    } else if signed {
        let limit = 1i32 << (bits - 1);
        (raw as i32).clamp(-limit, limit - 1) as u32
    } else {
        raw.min((1u32 << bits) - 1)
    }
}

/// Depth/stencil ("zeta") formats as the 3D engine's `SET_ZT_FORMAT`
/// encodes them [public: NVIDIA open-gpu-doc `classes/3d/clb197.h`
/// `NVB197_SET_ZT_FORMAT_V_*`].
///
/// Packed names list the 32-bit word from its most significant bits, as
/// for the color formats: `Z24S8` keeps stencil in bits 0-7 and depth in
/// bits 8-31; `S8Z24` keeps depth in bits 0-23 and stencil in 24-31. The
/// TIC component types of the same-named texture formats agree:
/// `Z24S8` has a UINT R (stencil) and UNORM G (depth), `S8Z24` a UNORM R
/// and UINT G. `ZF32_X24S8` is a float32 depth word followed by a word
/// with stencil in bits 0-7. The stencil-only `S8` and the VCAA formats
/// are not modeled ([`DepthFormat::from_raw`] returns `None`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DepthFormat {
    /// `ZF32` (0x0A).
    Z32Float,
    /// `Z16` (0x13).
    Z16Unorm,
    /// `Z24S8` (0x14).
    Z24S8,
    /// `X8Z24` (0x15).
    X8Z24,
    /// `S8Z24` (0x16).
    S8Z24,
    /// `ZF32_X24S8` (0x19).
    Z32FloatX24S8,
}

impl DepthFormat {
    /// Decodes a `SET_ZT_FORMAT` value.
    pub fn from_raw(raw: u32) -> Option<DepthFormat> {
        Some(match raw {
            0x0a => DepthFormat::Z32Float,
            0x13 => DepthFormat::Z16Unorm,
            0x14 => DepthFormat::Z24S8,
            0x15 => DepthFormat::X8Z24,
            0x16 => DepthFormat::S8Z24,
            0x19 => DepthFormat::Z32FloatX24S8,
            _ => return None,
        })
    }

    /// Bytes per pixel.
    pub fn bytes_per_pixel(self) -> u32 {
        match self {
            DepthFormat::Z16Unorm => 2,
            DepthFormat::Z32FloatX24S8 => 8,
            _ => 4,
        }
    }

    /// Whether the format stores stencil.
    pub fn has_stencil(self) -> bool {
        matches!(
            self,
            DepthFormat::Z24S8 | DepthFormat::S8Z24 | DepthFormat::Z32FloatX24S8
        )
    }

    /// Whether depth is stored as UNORM (so values clamp to [0, 1]).
    pub fn is_unorm(self) -> bool {
        !matches!(self, DepthFormat::Z32Float | DepthFormat::Z32FloatX24S8)
    }

    fn word(texel: &[u8]) -> u32 {
        u32::from_le_bytes([texel[0], texel[1], texel[2], texel[3]])
    }

    /// Reads the depth of one pixel.
    pub fn read_depth(self, texel: &[u8]) -> f32 {
        match self {
            DepthFormat::Z32Float | DepthFormat::Z32FloatX24S8 => f32::from_bits(Self::word(texel)),
            DepthFormat::Z16Unorm => {
                unorm_to_f32(u32::from(u16::from_le_bytes([texel[0], texel[1]])), 16)
            }
            DepthFormat::Z24S8 => unorm_to_f32(Self::word(texel) >> 8, 24),
            DepthFormat::X8Z24 | DepthFormat::S8Z24 => {
                unorm_to_f32(Self::word(texel) & 0xff_ffff, 24)
            }
        }
    }

    /// Writes the depth of one pixel, leaving its stencil bits alone.
    /// UNORM formats convert per D3D11.3 §3.2.3.6; float depth is stored
    /// as given.
    pub fn write_depth(self, depth: f32, texel: &mut [u8]) {
        match self {
            DepthFormat::Z32Float | DepthFormat::Z32FloatX24S8 => {
                texel[..4].copy_from_slice(&depth.to_le_bytes())
            }
            DepthFormat::Z16Unorm => {
                texel[..2].copy_from_slice(&(f32_to_unorm(depth, 16) as u16).to_le_bytes())
            }
            DepthFormat::Z24S8 => {
                let word = (Self::word(texel) & 0xff) | (f32_to_unorm(depth, 24) << 8);
                texel[..4].copy_from_slice(&word.to_le_bytes());
            }
            DepthFormat::X8Z24 | DepthFormat::S8Z24 => {
                let word = (Self::word(texel) & 0xff00_0000) | f32_to_unorm(depth, 24);
                texel[..4].copy_from_slice(&word.to_le_bytes());
            }
        }
    }

    fn stencil_byte(self) -> Option<usize> {
        match self {
            DepthFormat::Z24S8 => Some(0),
            DepthFormat::S8Z24 => Some(3),
            DepthFormat::Z32FloatX24S8 => Some(4),
            _ => None,
        }
    }

    /// Reads the stencil of one pixel (`None` for formats without stencil).
    pub fn read_stencil(self, texel: &[u8]) -> Option<u8> {
        self.stencil_byte().map(|byte| texel[byte])
    }

    /// Writes the stencil of one pixel; formats without stencil ignore it.
    pub fn write_stencil(self, stencil: u8, texel: &mut [u8]) {
        if let Some(byte) = self.stencil_byte() {
            texel[byte] = stencil;
        }
    }
}

/// Data type of one TIC component (`R_DATA_TYPE` .. `A_DATA_TYPE`)
/// [public: NVIDIA open-gpu-doc `classes/3d/clb197tex.h`
/// `NVB197_TEXHEAD_BL_R_DATA_TYPE_NUM_*`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComponentType {
    Snorm,
    Unorm,
    Sint,
    Uint,
    SnormForceFp16,
    UnormForceFp16,
    Float,
}

impl ComponentType {
    pub(crate) fn from_raw(raw: u32) -> Option<ComponentType> {
        Some(match raw {
            1 => ComponentType::Snorm,
            2 => ComponentType::Unorm,
            3 => ComponentType::Sint,
            4 => ComponentType::Uint,
            5 => ComponentType::SnormForceFp16,
            6 => ComponentType::UnormForceFp16,
            7 => ComponentType::Float,
            _ => return None,
        })
    }
}

/// How the texels of a TIC component layout are stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Encoding {
    /// Components packed into a little-endian word: bit position and width
    /// of R, G, B, A (width 0: the format lacks the component).
    Packed([(u8, u8); 4]),
    /// `E5B9G9R9_SHAREDEXP`.
    SharedExponent,
    /// `DXT1` (BC1).
    Bc1,
    /// `DXT23` (BC2).
    Bc2,
    /// `DXT45` (BC3).
    Bc3,
    /// `DXN1` (BC4).
    Bc4,
    /// `DXN2` (BC5).
    Bc5,
    /// `BC7U`.
    Bc7,
    /// `ASTC_2D_<w>X<h>`.
    Astc,
}

/// A TIC `COMPONENTS` value: the texel layout of a texture
/// [public: NVIDIA open-gpu-doc `classes/3d/clb197tex.h`
/// `NVB197_TEXHEAD_BL_COMPONENTS_SIZES_*`].
///
/// Packed layouts name the word from its most significant component, so the
/// last-named component sits in the lowest bits and is R (`A8B8G8R8`: R in
/// bits 0-7); `R32_G32_B32_A32`-style names (underscore-separated) list
/// whole components in memory order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageFormat {
    /// The raw `COMPONENTS` value.
    pub raw: u32,
    pub encoding: Encoding,
    /// Texels per block (1x1 for uncompressed formats).
    pub block_width: u32,
    pub block_height: u32,
    pub bytes_per_block: u32,
}

impl ImageFormat {
    /// Decodes a TIC `COMPONENTS` value. Layouts not modeled are refused:
    /// `R1`, `Y8_VIDEO`, the subsampled `G8B8G8R8`/`B8G8R8G8`, ETC2/EAC,
    /// BC6H (HDR) and the VCAA depth formats.
    pub fn from_tic(raw: u32) -> Result<ImageFormat, Unmodeled> {
        let packed = |bytes: u32, fields: [(u8, u8); 4]| (Encoding::Packed(fields), 1, 1, bytes);
        let (encoding, block_width, block_height, bytes_per_block) = match raw {
            0x01 => packed(16, [(0, 32), (32, 32), (64, 32), (96, 32)]),
            0x02 => packed(12, [(0, 32), (32, 32), (64, 32), (0, 0)]),
            0x03 => packed(8, [(0, 16), (16, 16), (32, 16), (48, 16)]),
            0x04 => packed(8, [(0, 32), (32, 32), (0, 0), (0, 0)]),
            // R32_B24G8: a 32-bit R word, then G in bits 0-7 and B in bits
            // 8-31 of the second word.
            0x05 => packed(8, [(0, 32), (32, 8), (40, 24), (0, 0)]),
            0x07 => packed(4, [(0, 8), (8, 8), (16, 8), (0, 0)]),
            0x08 => packed(4, [(0, 8), (8, 8), (16, 8), (24, 8)]),
            0x09 => packed(4, [(0, 10), (10, 10), (20, 10), (30, 2)]),
            0x0c => packed(4, [(0, 16), (16, 16), (0, 0), (0, 0)]),
            0x0d => packed(4, [(0, 24), (24, 8), (0, 0), (0, 0)]),
            0x0e => packed(4, [(0, 8), (8, 24), (0, 0), (0, 0)]),
            0x0f => packed(4, [(0, 32), (0, 0), (0, 0), (0, 0)]),
            0x12 => packed(2, [(0, 4), (4, 4), (8, 4), (12, 4)]),
            0x13 => packed(2, [(0, 1), (1, 5), (6, 5), (11, 5)]),
            0x14 => packed(2, [(0, 5), (5, 5), (10, 5), (15, 1)]),
            0x15 => packed(2, [(0, 5), (5, 6), (11, 5), (0, 0)]),
            0x16 => packed(2, [(0, 5), (5, 5), (10, 6), (0, 0)]),
            0x18 => packed(2, [(0, 8), (8, 8), (0, 0), (0, 0)]),
            0x1b => packed(2, [(0, 16), (0, 0), (0, 0), (0, 0)]),
            0x1d => packed(1, [(0, 8), (0, 0), (0, 0), (0, 0)]),
            0x1e => packed(1, [(0, 4), (4, 4), (0, 0), (0, 0)]),
            0x20 => (Encoding::SharedExponent, 1, 1, 4),
            0x21 => packed(4, [(0, 11), (11, 11), (22, 10), (0, 0)]),
            0x24 => (Encoding::Bc1, 4, 4, 8),
            0x25 => (Encoding::Bc2, 4, 4, 16),
            0x26 => (Encoding::Bc3, 4, 4, 16),
            0x27 => (Encoding::Bc4, 4, 4, 8),
            0x28 => (Encoding::Bc5, 4, 4, 16),
            0x17 => (Encoding::Bc7, 4, 4, 16),
            // Z24S8: stencil (R) in bits 0-7, depth (G) in bits 8-31.
            0x29 => packed(4, [(0, 8), (8, 24), (0, 0), (0, 0)]),
            // X8Z24: depth (R) in bits 0-23.
            0x2a => packed(4, [(0, 24), (0, 0), (0, 0), (0, 0)]),
            // S8Z24: depth (R) in bits 0-23, stencil (G) in bits 24-31.
            0x2b => packed(4, [(0, 24), (24, 8), (0, 0), (0, 0)]),
            0x2f => packed(4, [(0, 32), (0, 0), (0, 0), (0, 0)]),
            // ZF32_X24S8: float depth (R), then stencil (G) in bits 0-7 of
            // the second word.
            0x30 => packed(8, [(0, 32), (32, 8), (0, 0), (0, 0)]),
            0x3a => packed(2, [(0, 16), (0, 0), (0, 0), (0, 0)]),
            0x40 => (Encoding::Astc, 4, 4, 16),
            0x50 => (Encoding::Astc, 5, 4, 16),
            0x41 => (Encoding::Astc, 5, 5, 16),
            0x51 => (Encoding::Astc, 6, 5, 16),
            0x42 => (Encoding::Astc, 6, 6, 16),
            0x55 => (Encoding::Astc, 8, 5, 16),
            0x52 => (Encoding::Astc, 8, 6, 16),
            0x44 => (Encoding::Astc, 8, 8, 16),
            0x56 => (Encoding::Astc, 10, 5, 16),
            0x57 => (Encoding::Astc, 10, 6, 16),
            0x53 => (Encoding::Astc, 10, 8, 16),
            0x45 => (Encoding::Astc, 10, 10, 16),
            0x54 => (Encoding::Astc, 12, 10, 16),
            0x46 => (Encoding::Astc, 12, 12, 16),
            _ => return Err(Unmodeled(format!("TIC COMPONENTS {raw:#04x}"))),
        };
        Ok(ImageFormat {
            raw,
            encoding,
            block_width,
            block_height,
            bytes_per_block,
        })
    }

    /// Which of R, G, B, A the layout stores.
    pub fn present(&self) -> [bool; 4] {
        match self.encoding {
            Encoding::Packed(fields) => fields.map(|(_, bits)| bits != 0),
            Encoding::SharedExponent => [true, true, true, false],
            Encoding::Bc4 => [true, false, false, false],
            Encoding::Bc5 => [true, true, false, false],
            _ => [true; 4],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unorm_round_trip_and_rounding() {
        for code in 0..=255u32 {
            assert_eq!(f32_to_unorm(unorm_to_f32(code, 8), 8), code);
        }
        // Exact halfway rounds up (D3D: +0.5 then truncate), not to even.
        assert_eq!(f32_to_unorm(0.5 / 255.0, 8), 1);
        assert_eq!(f32_to_unorm(2.5 / 255.0, 8), 3);
        assert_eq!(f32_to_unorm(-1.0, 8), 0);
        assert_eq!(f32_to_unorm(f32::NAN, 8), 0);
        assert_eq!(f32_to_unorm(2.0, 5), 31);
        for code in 0..(1u32 << 24) {
            if code % 4099 == 0 {
                assert_eq!(f32_to_unorm(unorm_to_f32(code, 24), 24), code);
            }
        }
    }

    #[test]
    fn snorm_conversions() {
        assert_eq!(snorm_to_f32(0x80, 8), -1.0);
        assert_eq!(snorm_to_f32(0x81, 8), -1.0);
        assert_eq!(snorm_to_f32(0x7f, 8), 1.0);
        assert_eq!(f32_to_snorm(-1.0, 8), 0x81);
        // 0.5/127 is rounded slightly below the exact midpoint as f32.
        // Test both representable neighbors rather than assuming a tie.
        let below = 0.5f32 / 127.0;
        let above = f32::from_bits(below.to_bits() + 1);
        assert_eq!(f32_to_snorm(-below, 8), 0);
        assert_eq!(f32_to_snorm(below, 8), 0);
        assert_eq!(f32_to_snorm(-above, 8), 0xff);
        assert_eq!(f32_to_snorm(above, 8), 1);
        for code in 0..=255u32 {
            if code != 0x80 {
                assert_eq!(f32_to_snorm(snorm_to_f32(code, 8), 8), code);
            }
        }
    }

    #[test]
    fn half_conversions() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x8001), -(2f32.powi(-24)));
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert!(f16_to_f32(0x7e00).is_nan());
        assert_eq!(f16_to_f32(0xfc00), f32::NEG_INFINITY);
        for h in 0..=0xffffu16 {
            let f = f16_to_f32(h);
            if !f.is_nan() {
                assert_eq!(f32_to_f16(f), h, "{h:#06x}");
            }
        }
        // Round toward zero; overflow gives the largest finite value.
        assert_eq!(f32_to_f16(1.0 + 1.5 * 2f32.powi(-10)), 0x3c01);
        assert_eq!(f32_to_f16(-(1.0 + 1.9 * 2f32.powi(-10))), 0xbc01);
        assert_eq!(f32_to_f16(1.0e6), 0x7bff);
        assert_eq!(f32_to_f16(f32::INFINITY), 0x7c00);
        assert_eq!(f32_to_f16(3.0 * 2f32.powi(-25)), 0x0001);
    }

    #[test]
    fn small_float_conversions() {
        assert_eq!(uf11_to_f32(0x3c0), 1.0);
        assert_eq!(uf10_to_f32(0x1e0), 1.0);
        assert_eq!(uf11_to_f32(0x7bf), 65024.0);
        assert_eq!(uf11_to_f32(1), 2f32.powi(-20));
        assert_eq!(uf10_to_f32(1), 2f32.powi(-19));
        for v in 0..0x7c0u32 {
            assert_eq!(f32_to_uf11(uf11_to_f32(v)), v);
        }
        for v in 0..0x3e0u32 {
            assert_eq!(f32_to_uf10(uf10_to_f32(v)), v);
        }
        assert_eq!(f32_to_uf11(-3.0), 0);
        assert_eq!(f32_to_uf11(1.0e9), 0x7bf);
        assert_eq!(f32_to_uf10(1.99), 0x1ff);
    }

    #[test]
    fn r11g11b10_round_trip() {
        let format = ColorFormat::R11G11B10Float;
        let mut texel = [0u8; 4];
        format.write([1.0, 0.5, 2.0, 0.25], &mut texel);
        // R 1.0 = 0x3c0, G 0.5 = 0x380, B 2.0 = 0x200.
        let word = u32::from_le_bytes(texel);
        assert_eq!(word, 0x3c0 | (0x380 << 11) | (0x200 << 22));
        assert_eq!(format.read(&texel), [1.0, 0.5, 2.0, 1.0]);
    }

    #[test]
    fn color_format_layouts() {
        let mut texel = [0u8; 4];
        ColorFormat::Bgra8Unorm.write([1.0, 0.5, 0.0, 0.2], &mut texel);
        assert_eq!(texel, [0, 128, 255, 51]);
        ColorFormat::Rgba8Srgb.write([1.0, 0.5, 0.0, 0.2], &mut texel);
        assert_eq!(texel, [255, 128, 0, 51], "sRGB formats store raw UNORM");
        assert!(ColorFormat::Rgba8Srgb.is_srgb());
        let mut texel = [0u8; 2];
        ColorFormat::B5G6R5Unorm.write([1.0, 0.0, 1.0, 0.0], &mut texel);
        assert_eq!(u16::from_le_bytes(texel), 0xf81f);
        assert_eq!(
            ColorFormat::B5G6R5Unorm.read(&[0x00, 0xf8]),
            [1.0, 0.0, 0.0, 1.0]
        );
        ColorFormat::Bgr5O1Unorm.write([0.0; 4], &mut texel);
        assert_eq!(u16::from_le_bytes(texel), 0x8000);
        let mut texel = [0u8; 4];
        ColorFormat::Rgb10A2Unorm.write([1.0, 0.0, 0.0, 1.0], &mut texel);
        assert_eq!(u32::from_le_bytes(texel), 0xc000_03ff);
        assert_eq!(ColorFormat::Bgrx8Unorm.read(&[0, 0, 0, 0])[3], 1.0);
        let mut texel = [0u8; 8];
        ColorFormat::Rgba16Float.write([1.0, -2.0, 0.5, 65504.0], &mut texel);
        assert_eq!(
            ColorFormat::Rgba16Float.read(&texel),
            [1.0, -2.0, 0.5, 65504.0]
        );
    }

    #[test]
    fn depth_layouts() {
        let mut texel = [0x5a, 0, 0, 0];
        DepthFormat::Z24S8.write_depth(1.0, &mut texel);
        assert_eq!(u32::from_le_bytes(texel), 0xffff_ff5a);
        assert_eq!(DepthFormat::Z24S8.read_stencil(&texel), Some(0x5a));
        let mut texel = [0, 0, 0, 0x77];
        DepthFormat::S8Z24.write_depth(0.5, &mut texel);
        assert_eq!(u32::from_le_bytes(texel), 0x7780_0000);
        assert_eq!(
            DepthFormat::S8Z24.read_depth(&texel),
            unorm_to_f32(0x80_0000, 24)
        );
        let mut texel = [0u8; 8];
        DepthFormat::Z32FloatX24S8.write_depth(0.25, &mut texel);
        DepthFormat::Z32FloatX24S8.write_stencil(9, &mut texel);
        assert_eq!(texel, [0, 0, 0x80, 0x3e, 9, 0, 0, 0]);
    }
}
