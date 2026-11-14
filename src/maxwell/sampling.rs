//! What a shader's texture instructions ask of the texture unit.
//!
//! The shader interpreter ([`super::shader`]) turns `TEX`/`TEXS`/`TLD`/
//! `TLDS`/`TLD4`/`TLD4S`/`TXQ`/`TMML`/`TXD` into these requests; the draw
//! pipeline answers them through [`TextureUnit`], resolving the 32-bit
//! texture handle to a texture image control (TIC) and texture sampler
//! control (TSC) entry and asking [`super::texture`] for the texels.
//!
//! Texel values are `f32` throughout: every texture format the firmware's
//! UI uses is a normalized or floating-point format. Integer formats are
//! refused by the texture module.

use super::Unmodeled;

/// The dimensionality a texture instruction declares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    Texture1D,
    Texture2D,
    Texture3D,
    Cube,
}

/// How the level of detail of a sample is chosen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Lod {
    /// From screen-space derivatives of the coordinates (fragment shaders,
    /// computed across the 2x2 quad), plus a bias (`LB`, 0 otherwise).
    Derivatives {
        ddx: [f32; 3],
        ddy: [f32; 3],
        bias: f32,
    },
    /// An explicit level (`LL`; `LZ` is level 0). Also what a stage without
    /// quads (vertex) uses when the instruction asks for implicit LOD:
    /// hardware samples level 0 there.
    Level(f32),
}

/// A filtered sample (`TEX`, `TEXS`, `TXD`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampleRequest {
    pub target: Target,
    pub array: bool,
    /// Normalized coordinates (s, t, r); unused ones are 0. For cube maps
    /// the direction vector.
    pub coords: [f32; 3],
    /// Array layer, already converted to an integer the way the
    /// instruction defines it.
    pub layer: u32,
    pub lod: Lod,
    /// Texel offsets (`AOFFI`), 0 when absent.
    pub offset: [i32; 3],
    /// Depth-compare reference (`DC`).
    pub compare: Option<f32>,
}

/// An unfiltered texel fetch (`TLD`, `TLDS`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FetchRequest {
    pub target: Target,
    pub array: bool,
    /// Integer texel coordinates.
    pub coords: [i32; 3],
    pub layer: u32,
    pub level: i32,
    pub offset: [i32; 3],
}

/// A four-texel gather of one component (`TLD4`, `TLD4S`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GatherRequest {
    pub target: Target,
    pub array: bool,
    pub coords: [f32; 3],
    pub layer: u32,
    pub offset: [i32; 3],
    /// Which component (0-3) to gather.
    pub component: u8,
    pub compare: Option<f32>,
}

/// What `TXQ` asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Query {
    /// Width, height, depth/layers and level count of `level`.
    Dimension { level: u32 },
    /// Texture type, as `TXQ` mode 1 reports it.
    TextureType,
    /// Sample position (multisample).
    SamplePosition,
}

/// The texture unit as a shader sees it. `handle` is the 32-bit texture
/// handle the instruction resolved (from the bound-texture constant buffer
/// slot, or a register for bindless instructions); the implementation maps
/// it to TIC/TSC entries.
pub trait TextureUnit {
    fn sample(&self, handle: u32, request: &SampleRequest) -> Result<[f32; 4], Unmodeled>;
    fn fetch(&self, handle: u32, request: &FetchRequest) -> Result<[f32; 4], Unmodeled>;
    fn gather(&self, handle: u32, request: &GatherRequest) -> Result<[f32; 4], Unmodeled>;
    fn query(&self, handle: u32, query: Query) -> Result<[u32; 4], Unmodeled>;
    /// `TMML`: the level of detail the hardware would compute, as
    /// (clamped LOD, unclamped LOD).
    fn lod(
        &self,
        handle: u32,
        target: Target,
        ddx: [f32; 3],
        ddy: [f32; 3],
    ) -> Result<[f32; 2], Unmodeled>;
}
