//! The shader program header (SPH): the first 0x50 bytes of every program.
//!
//! Layout [public: NVIDIA open-gpu-doc Shader-Program-Header, tables 1-24]:
//!
//! * `CommonWord0` (byte 0x00): `SphType` bits 0-4 (1 = VTG, 2 = PS),
//!   `Version` 5-9, `ShaderType` 10-13 (1 vertex, 2 tessellation init,
//!   3 tessellation, 4 geometry, 5 pixel), `MrtEnable` 14, `KillsPixels`
//!   15, `DoesGlobalStore` 16, `SassVersion` 17-20, `DoesLoadOrStore` 26,
//!   `DoesFp64` 27, `StreamOutMask` 28-31.
//! * `CommonWord1`-`CommonWord4` (bytes 0x04-0x13): local memory sizes,
//!   geometry output topology and vertex count, store-request range.
//! * The input map (Imap) from byte 0x14, one entry per attribute word in
//!   attribute-address order: system values A/B (words 0x00-0x1f, i.e.
//!   a\[0x000\]-a\[0x07c\]), 32 generic vectors (a\[0x080\]-a\[0x27c\]),
//!   colors (a\[0x280\]-), system values C (a\[0x2c0\]-a\[0x2fc\]), ten
//!   fixed-function texture coordinates (a\[0x300\]-a\[0x39c\]).
//!   * VTG (type 1): one bit per word for words 0x00-0xef (30 bytes), then
//!     the output map (Omap) in the same layout from byte 0x32.
//!   * PS (type 2): system values keep one bit per word; generic vectors,
//!     the eight colors (diffuse RGBA, specular RGBA) and the texture
//!     coordinates take two bits per component (`PixelImap`: 0 unused,
//!     1 constant, 2 perspective, 3 screen-linear). The PS Omap at byte 0x48
//!     is `OmapTarget[8]` (four bits RGBA per render target), then bit 32
//!     `OmapSampleMask` and bit 33 `OmapDepth`.

use super::super::Unmodeled;

/// Size of the shader program header in bytes.
pub const SPH_SIZE: usize = 0x50;

/// The pipeline stage a program is for (`CommonWord0.ShaderType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderType {
    Vertex,
    TessellationInit,
    Tessellation,
    Geometry,
    Pixel,
}

/// How a pixel shader input word is interpolated, from the PS Imap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interpolation {
    /// Not read by the program.
    Unused,
    /// `PixelImap` 2: perspective-correct.
    Perspective,
    /// `PixelImap` 3: screen-linear.
    Linear,
    /// `PixelImap` 1: constant across the primitive.
    Flat,
    /// A one-bit system-value entry (position, primitive ID, clip
    /// distances, point sprite S/T, ...): the SPH only declares that the
    /// program reads it, the value comes from fixed function (for
    /// a\[0x070\]-a\[0x07c\] the window x/y, depth and 1/w the rasterizer
    /// prepares).
    SystemValue,
}

/// A parsed shader program header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Header {
    bytes: [u8; SPH_SIZE],
    shader_type: ShaderType,
}

impl Header {
    /// Parses the 0x50-byte SPH. Only the two documented `SphType`s with a
    /// matching `ShaderType` are accepted.
    pub fn parse(bytes: &[u8; SPH_SIZE]) -> Result<Header, Unmodeled> {
        let word0 = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let sph_type = word0 & 0x1f;
        let shader_type = match (sph_type, (word0 >> 10) & 0xf) {
            (1, 1) => ShaderType::Vertex,
            (1, 2) => ShaderType::TessellationInit,
            (1, 3) => ShaderType::Tessellation,
            (1, 4) => ShaderType::Geometry,
            (2, 5) => ShaderType::Pixel,
            (s, t) => {
                return Err(Unmodeled(format!(
                    "shader: SPH type {s} with shader type {t} (word0 {word0:#010x})"
                )));
            }
        };
        Ok(Header {
            bytes: *bytes,
            shader_type,
        })
    }

    /// The raw header bytes.
    pub fn bytes(&self) -> &[u8; SPH_SIZE] {
        &self.bytes
    }

    pub fn shader_type(&self) -> ShaderType {
        self.shader_type
    }

    fn bit(&self, index: usize) -> bool {
        (self.bytes[index / 8] >> (index % 8)) & 1 != 0
    }

    fn is_pixel(&self) -> bool {
        self.shader_type == ShaderType::Pixel
    }

    /// `CommonWord0.MrtEnable`: the pixel shader writes one color per
    /// render target (otherwise output 0 goes to every enabled target).
    pub fn mrt_enable(&self) -> bool {
        self.bit(14)
    }

    /// `CommonWord0.KillsPixels`: the pixel shader may execute `KIL`.
    pub fn ps_uses_discard(&self) -> bool {
        self.is_pixel() && self.bit(15)
    }

    /// `CommonWord0.DoesGlobalStore`.
    pub fn does_global_store(&self) -> bool {
        self.bit(16)
    }

    /// `CommonWord2.ThreadsPerInputPrimitive` (bits 24-31 of byte 0x08):
    /// for a geometry shader, the threads run per input primitive.
    pub fn threads_per_input_primitive(&self) -> u32 {
        u32::from(self.bytes[0x0b])
    }

    /// `CommonWord3.OutputTopology` (bits 24-27 of byte 0x0c): for a
    /// geometry shader, POINTLIST (1), LINESTRIP (6) or TRIANGLESTRIP (7).
    pub fn output_topology(&self) -> u32 {
        u32::from(self.bytes[0x0f] & 0xf)
    }

    /// `CommonWord4.MaxOutputVertexCount` (bits 0-11 of byte 0x10): for a
    /// geometry shader, the vertices one thread may output; `OUT`
    /// instructions beyond it are no-ops.
    pub fn max_output_vertex_count(&self) -> u32 {
        u32::from(u16::from_le_bytes([self.bytes[0x10], self.bytes[0x11]]) & 0xfff)
    }

    /// How a pixel shader reads attribute word `attribute_word`
    /// (attribute byte address / 4). [`Interpolation::Unused`] for words the
    /// PS Imap has no entry for, and for a VTG header.
    pub fn ps_interpolation(&self, attribute_word: usize) -> Interpolation {
        if !self.is_pixel() {
            return Interpolation::Unused;
        }
        let two_bit = |entry: usize| -> Interpolation {
            let bit = entry * 2;
            match (self.bytes[bit / 8] >> (bit % 8)) & 3 {
                0 => Interpolation::Unused,
                1 => Interpolation::Flat,
                2 => Interpolation::Perspective,
                _ => Interpolation::Linear,
            }
        };
        let one_bit = |bit: usize| -> Interpolation {
            if self.bit(bit) {
                Interpolation::SystemValue
            } else {
                Interpolation::Unused
            }
        };
        match attribute_word {
            // ImapSystemValuesA/B: bits 160.. (byte 0x14).
            0x00..=0x1f => one_bit(0x14 * 8 + attribute_word),
            // ImapGenericVector[32] (ImapPixelVector): two bits per word from byte 0x18.
            0x20..=0x9f => two_bit(0x18 * 4 + (attribute_word - 0x20)),
            // ImapPixelColor: diffuse RGBA, specular RGBA from byte 0x38.
            0xa0..=0xa7 => two_bit(0x38 * 4 + (attribute_word - 0xa0)),
            // ImapSystemValuesC: one bit per word (a[0x2c0]-a[0x2fc]) from byte 0x3a.
            0xb0..=0xbf => one_bit(0x3a * 8 + (attribute_word - 0xb0)),
            // ImapFixedFncTexture[10] (ImapPixelTexture) from byte 0x3c.
            0xc0..=0xe7 => two_bit(0x3c * 4 + (attribute_word - 0xc0)),
            _ => Interpolation::Unused,
        }
    }

    /// The RGBA component mask (bit 0 red .. bit 3 alpha) render target
    /// `target` (0-7) receives from a pixel shader (`OmapTarget[target]`);
    /// 0 for a VTG header.
    pub fn ps_color_mask(&self, target: usize) -> u8 {
        if !self.is_pixel() || target >= 8 {
            return 0;
        }
        let bit = 0x48 * 8 + target * 4;
        (self.bytes[bit / 8] >> (bit % 8)) & 0xf
    }

    /// `OmapSampleMask`: the pixel shader writes the coverage mask.
    pub fn ps_writes_sample_mask(&self) -> bool {
        self.is_pixel() && self.bit(0x4c * 8)
    }

    /// `OmapDepth`: the pixel shader writes depth.
    pub fn ps_writes_depth(&self) -> bool {
        self.is_pixel() && self.bit(0x4c * 8 + 1)
    }

    /// Whether a vertex/tessellation/geometry program reads attribute word
    /// `attribute_word` (VTG Imap); false for a PS header.
    pub fn vtg_input_used(&self, attribute_word: usize) -> bool {
        !self.is_pixel() && attribute_word < 0xf0 && self.bit(0x14 * 8 + attribute_word)
    }

    /// Whether a vertex/tessellation/geometry program writes attribute word
    /// `attribute_word` (VTG Omap); false for a PS header.
    pub fn vtg_output_used(&self, attribute_word: usize) -> bool {
        !self.is_pixel() && attribute_word < 0xf0 && self.bit(0x32 * 8 + attribute_word)
    }
}
