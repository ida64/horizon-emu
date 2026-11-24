//! Triangle rasterization and the fragment back end: coverage with the
//! top-left fill rule, fragment shading in 2x2 quads, depth test, blending
//! and render target writes. Line segments and points reach it as the
//! triangles of the quads they cover (`pipeline::setup`).
//!
//! Positions are snapped to 1/256 pixel, the precision the netlist's
//! initial `SET_SNAP_GRID_NON_LINE` value 8 selects (`LOCATIONS_PER_PIXEL
//! 256X256`, rounding `RTNE`; nvservices netlist region 7, field values
//! [public: NVIDIA open-gpu-doc `classes/3d/clb197.h`]). Coverage is tested
//! at the sample positions (the pixel center with one sample per pixel; see
//! `multisample` and `pipeline::multisample`) with integer edge functions
//! and the top-left rule of the Direct3D 10+ rasterization rules [public:
//! Direct3D 11 functional specification, 3.4.1 "Triangle Rasterization
//! Rules"], which NVIDIA hardware implements; it is not pinned by the
//! firmware. The pixel program runs once per pixel with any sample
//! covered; depth, stencil and blending run per covered sample, into the
//! texel holding it. Coordinates here are physical: x to the right, y down
//! in memory row order.

use super::Unmodeled;
use super::blend::{Blend, Inputs};
use super::multisample::Samples;
use super::pipeline::Multisample;
use super::shader::{Env, FragmentOutput, Plane, Program, Quad};
use super::surface::{BAND_ROWS, Band, Rect};
use super::texture::format::{ColorFormat, DepthFormat, linear_to_srgb, srgb_to_linear};

/// Sub-pixel bits of snapped positions.
pub const SUBPIXEL_BITS: u32 = 8;
const SUBPIXEL: f64 = (1 << SUBPIXEL_BITS) as f64;

/// A depth comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compare {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}

impl Compare {
    /// `SET_DEPTH_FUNC` values, OpenGL (0x200-0x207) and Direct3D (1-8)
    /// encodings [public: NVIDIA open-gpu-doc `clb197.h`].
    pub fn from_raw(raw: u32) -> Result<Self, Unmodeled> {
        Ok(match raw {
            0x200 | 1 => Compare::Never,
            0x201 | 2 => Compare::Less,
            0x202 | 3 => Compare::Equal,
            0x203 | 4 => Compare::LessEqual,
            0x204 | 5 => Compare::Greater,
            0x205 | 6 => Compare::NotEqual,
            0x206 | 7 => Compare::GreaterEqual,
            0x207 | 8 => Compare::Always,
            _ => return Err(Unmodeled(format!("3D: depth function {raw:#x}"))),
        })
    }

    #[inline]
    pub fn test(self, incoming: f32, stored: f32) -> bool {
        match self {
            Compare::Never => false,
            Compare::Less => incoming < stored,
            Compare::Equal => incoming == stored,
            Compare::LessEqual => incoming <= stored,
            Compare::Greater => incoming > stored,
            Compare::NotEqual => incoming != stored,
            Compare::GreaterEqual => incoming >= stored,
            Compare::Always => true,
        }
    }
}

/// How one fragment shader color output reaches its render target.
#[derive(Debug, Clone, Copy)]
pub struct ColorOutput {
    /// Shader output slot (0-7).
    pub output: usize,
    /// Index into the band set of the color window.
    pub window: usize,
    pub format: ColorFormat,
    pub blend: Option<Blend>,
    /// R, G, B, A write enables.
    pub write: [bool; 4],
    /// Convert linear shader output to sRGB on write (and back for
    /// blending).
    pub srgb: bool,
}

/// A stencil operation (`SET_STENCIL_OP_FAIL` / `ZFAIL` / `ZPASS` and
/// their `SET_BACK_STENCIL_OP_*` twins).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StencilOp {
    Keep,
    Zero,
    Replace,
    /// Increment, saturating at 255.
    IncrementClamp,
    /// Decrement, saturating at 0.
    DecrementClamp,
    Invert,
    /// Increment, wrapping.
    IncrementWrap,
    /// Decrement, wrapping.
    DecrementWrap,
}

impl StencilOp {
    /// The OpenGL (`GL_KEEP` .. `GL_DECR_WRAP`) and Direct3D (1-8)
    /// encodings [public: NVIDIA open-gpu-doc `clb197.h`,
    /// `NVB197_SET_STENCIL_OP_FAIL_V_*`: OGL INCRSAT 0x1E02 and INCR
    /// 0x8507 are OpenGL's `GL_INCR` (saturating) and `GL_INCR_WRAP`;
    /// D3D INCRSAT 4, INCR 7 are `D3D11_STENCIL_OP_INCR_SAT`, `_INCR`].
    pub fn from_raw(raw: u32) -> Option<Self> {
        Some(match raw {
            0x1E00 | 1 => StencilOp::Keep,
            0x0000 | 2 => StencilOp::Zero,
            0x1E01 | 3 => StencilOp::Replace,
            0x1E02 | 4 => StencilOp::IncrementClamp,
            0x1E03 | 5 => StencilOp::DecrementClamp,
            0x150A | 6 => StencilOp::Invert,
            0x8507 | 7 => StencilOp::IncrementWrap,
            0x8508 | 8 => StencilOp::DecrementWrap,
            _ => return None,
        })
    }

    /// The new stencil value from the stored one, before the write mask.
    #[inline]
    pub fn apply(self, stored: u8, reference: u8) -> u8 {
        match self {
            StencilOp::Keep => stored,
            StencilOp::Zero => 0,
            StencilOp::Replace => reference,
            StencilOp::IncrementClamp => stored.saturating_add(1),
            StencilOp::DecrementClamp => stored.saturating_sub(1),
            StencilOp::Invert => !stored,
            StencilOp::IncrementWrap => stored.wrapping_add(1),
            StencilOp::DecrementWrap => stored.wrapping_sub(1),
        }
    }
}

/// The stencil state of one face (OpenGL's `glStencilFuncSeparate`,
/// `glStencilOpSeparate`, `glStencilMaskSeparate`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StencilFace {
    /// Applied where the stencil test fails.
    pub fail: StencilOp,
    /// Applied where the stencil test passes and the depth test fails.
    pub depth_fail: StencilOp,
    /// Applied where both pass.
    pub pass: StencilOp,
    pub compare: Compare,
    pub reference: u8,
    /// `FUNC_MASK`: ANDed with the reference and the stored value before
    /// the comparison.
    pub compare_mask: u8,
    /// `MASK`: the stencil bits operations may change.
    pub write_mask: u8,
}

impl StencilFace {
    /// Whether the test passes for a stored value: (reference & mask)
    /// `compare` (stored & mask), the reference on the left as in
    /// OpenGL 4.6 core, 17.3.3 "Stencil Test" and Direct3D 11.3,
    /// 17.12 "Stencil Test" [public].
    #[inline]
    pub fn test(&self, stored: u8) -> bool {
        self.compare.test(
            f32::from(self.reference & self.compare_mask),
            f32::from(stored & self.compare_mask),
        )
    }

    /// The stored value after `op`, with only the write mask's bits
    /// changed.
    #[inline]
    pub fn update(&self, op: StencilOp, stored: u8) -> u8 {
        let value = op.apply(stored, self.reference);
        stored & !self.write_mask | value & self.write_mask
    }

    /// Whether a fragment failing the stencil test, or (with `depth_can_fail`)
    /// the depth test, may change the stencil buffer.
    pub fn writes_on_failure(&self, depth_can_fail: bool) -> bool {
        self.write_mask != 0
            && (self.fail != StencilOp::Keep
                || (depth_can_fail && self.depth_fail != StencilOp::Keep))
    }

    /// Whether any fragment may change the stencil buffer.
    pub fn writes(&self) -> bool {
        self.write_mask != 0
            && [self.fail, self.depth_fail, self.pass]
                .iter()
                .any(|&op| op != StencilOp::Keep)
    }
}

/// The stencil test: front faces use `front`, back faces `back` (equal to
/// `front` unless `SET_TWO_SIDED_STENCIL_TEST` is enabled). Points and
/// lines are front-facing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stencil {
    pub front: StencilFace,
    pub back: StencilFace,
}

impl Stencil {
    #[inline]
    pub fn face(&self, front_facing: bool) -> &StencilFace {
        if front_facing {
            &self.front
        } else {
            &self.back
        }
    }
}

/// The depth/stencil stage. With the depth test disabled, `compare` is
/// `Always` and `write` false (OpenGL 4.6 core, 17.3.4: a disabled depth
/// test always passes and writes nothing).
#[derive(Debug, Clone, Copy)]
pub struct DepthOutput {
    pub format: DepthFormat,
    pub compare: Compare,
    pub write: bool,
    /// Range the fragment depth is clamped to before the test.
    pub clamp: (f32, f32),
    /// The stencil test, when enabled and the format has stencil (without
    /// a stencil buffer the test passes and nothing is written: OpenGL 4.6
    /// core, 17.3.3).
    pub stencil: Option<Stencil>,
}

impl DepthOutput {
    /// Whether a draw may change the buffer.
    pub fn writes(&self) -> bool {
        self.write
            || self
                .stencil
                .is_some_and(|s| s.front.writes() || s.back.writes())
    }
}

/// Everything fragment processing of one draw needs besides the bands.
pub struct Backend<'a> {
    pub program: &'a Program,
    pub colors: &'a [ColorOutput],
    pub depth: Option<DepthOutput>,
    /// The alpha test: output 0 alpha must pass `compare` against the
    /// reference (`None`: no test).
    pub alpha_test: Option<(Compare, f32)>,
    pub blend_constant: [f32; 4],
    /// `SET_BLEND_FLOAT_OPTION` `ZERO_TIMES_ANYTHING_IS_ZERO`.
    pub blend_zero_rule: bool,
    /// Pixels that may be written (scissor, surface clip, targets).
    pub clip: Rect,
    /// Samples per pixel, where they are and how they are stored.
    pub multisample: Multisample,
    /// Where the samples that pass the stencil and depth tests are
    /// counted (`SET_ZPASS_PIXEL_COUNT`, ZPASS_PIXEL_CNT), while that
    /// counter is enabled.
    pub passed: Option<&'a std::sync::atomic::AtomicU64>,
}

/// A triangle in physical window space, ready to rasterize.
pub struct Triangle {
    /// Snapped vertex positions, counter-clockwise in y-down order, in
    /// 1/256 pixels.
    vertices: [[i64; 2]; 3],
    /// Bias per edge: 0 for top/left edges, -1 otherwise.
    bias: [i64; 3],
    /// Pixel bounding box (exclusive end).
    bounds: Rect,
    pub planes: Box<[Plane; 256]>,
    /// Window-space depth at the pixel center.
    pub depth: Plane,
    pub front_facing: bool,
}

/// `orient2d(a, b, c)`: twice the signed area, positive when `c` is to
/// the right of `a -> b` with y down.
#[inline]
fn orient(a: [i64; 2], b: [i64; 2], c: [i64; 2]) -> i64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// Snaps a window coordinate to the sub-pixel grid, rounding to nearest
/// even.
fn snap(value: f32) -> i64 {
    (f64::from(value) * SUBPIXEL).round_ties_even() as i64
}

/// A window coordinate snapped to the sub-pixel grid, in pixels.
pub fn snap_to_grid(value: f32) -> f32 {
    (snap(value) as f64 / SUBPIXEL) as f32
}

/// The pixels a non-antialiased line segment from `a` to `b` (window
/// positions, snapped here) produces, `width` pixels wide, as runs of
/// whole pixels `[x0, y0, x1, y1]` (end exclusive). Only pixels whose
/// major-axis coordinate lies in `area` are produced (the rest is cut by
/// the scissor anyway).
///
/// The rule is the diamond exit of OpenGL and of Vulkan's Bresenham lines
/// [public: OpenGL 4.6 core, "Basic Line Segment Rasterization"; Vulkan
/// 1.3, "Bresenham Line Segment Rasterization"]. A pixel is produced
/// when the segment, with both endpoints perturbed by (-ε, -ε²),
/// intersects the open diamond |x - x_c| + |y - y_c| < 1/2 around the
/// pixel center and its perturbed end point is not inside it. A wide
/// line is the segment offset by -(w - 1)/2 in the minor axis (y for an
/// x-major segment, |Δx| >= |Δy|), each of its pixels widened to a run
/// of `width` in the minor axis towards +y or +x (the same sections).
/// Vulkan states the rule in framebuffer coordinates, whose y runs down
/// as here; these lines are selected with `SET_LINE_MULTISAMPLE_OVERRIDE`
/// (see `pipeline::line_rule`). \[assumption: the hardware implements
/// this exact rule. Both APIs allow rules within one pixel of it, and
/// NVIDIA's is not public. Whether `SET_WINDOW_ORIGIN` FLIP_Y, which
/// makes facing count rows up, also turns the -ε² towards +y is not
/// established either; it decides only lines and endpoints exactly on a
/// pixel edge or diamond boundary in y.\]
///
/// For a segment with |slope| <= 1 in the major axis, a diamond the line
/// enters is crossed on its minor-axis diagonal, so each column (or row)
/// has one candidate: the pixel holding the perturbed line at the
/// column's center. That pixel is produced when the end point is outside
/// its diamond and past the center in the direction of travel, and the
/// start point is inside the diamond or not yet past the center.
pub fn aliased_line(a: [f32; 2], b: [f32; 2], width: u32, area: Rect) -> Vec<[i64; 4]> {
    let unit = 1i64 << SUBPIXEL_BITS;
    let half = unit / 2;
    let [mut a, mut b] = [a, b].map(|p| p.map(snap));
    let x_major = (b[0] - a[0]).abs() >= (b[1] - a[1]).abs();
    let (major, minor) = if x_major { (0, 1) } else { (1, 0) };
    let shift = i64::from(width.max(1) - 1) * half;
    a[minor] -= shift;
    b[minor] -= shift;
    let dm = i128::from(b[major] - a[major]);
    let dn = i128::from(b[minor] - a[minor]);
    if dm == 0 {
        return Vec::new();
    }
    // A perturbed line exactly on a pixel edge in the minor axis at a
    // column's center leaves it by the perturbation: for an x-major
    // segment by s ε - ε² in y (towards +y only with a positive slope),
    // for a y-major one by s ε² - ε in x (towards -x).
    let tie_after = x_major && dn != 0 && (dn > 0) == (dm > 0);
    // A perturbed point inside the open diamond around `center`: on its
    // boundary, the -ε in x moves points right of center inside.
    let inside = |p: [i64; 2], center: [i64; 2]| {
        let u = p[0] - center[0];
        let distance = u.abs() + (p[1] - center[1]).abs();
        distance < half || (distance == half && u > 0)
    };
    let (area_first, area_end) = if x_major {
        (area.x0, area.x1)
    } else {
        (area.y0, area.y1)
    };
    let first = (a[major].min(b[major]).div_euclid(unit) - 1).max(i64::from(area_first));
    let last = (a[major].max(b[major]).div_euclid(unit) + 1).min(i64::from(area_end) - 1);
    let forward = dm > 0;
    // Runs as (major start, major end, minor), merged while the minor
    // index stays the same.
    let mut runs: Vec<(i64, i64, i64)> = Vec::new();
    for column in first..=last {
        let center_major = column * unit + half;
        // The minor coordinate of the line at the column's center is
        // `num / den` (den > 0).
        let num = i128::from(a[minor]) * dm + i128::from(center_major - a[major]) * dn;
        let (num, den) = if dm < 0 { (-num, -dm) } else { (num, dm) };
        let cell = den * i128::from(unit);
        let mut row = num.div_euclid(cell);
        if num.rem_euclid(cell) == 0 && !tie_after {
            row -= 1;
        }
        let row = row as i64;
        let mut center = [0; 2];
        center[major] = center_major;
        center[minor] = row * unit + half;
        let produced = !inside(b, center)
            && if forward {
                b[major] > center_major && (a[major] <= center_major || inside(a, center))
            } else {
                b[major] <= center_major && (a[major] > center_major || inside(a, center))
            };
        if !produced {
            continue;
        }
        match runs.last_mut() {
            Some(run) if run.1 == column && run.2 == row => run.1 = column + 1,
            _ => runs.push((column, column + 1, row)),
        }
    }
    let width = i64::from(width.max(1));
    runs.into_iter()
        .map(|(start, end, row)| {
            if x_major {
                [start, row, end, row + width]
            } else {
                [row, start, row + width, end]
            }
        })
        .collect()
}

impl Triangle {
    /// Sets up a triangle from window-space positions (`x`, `y` physical
    /// pixels). Returns `None` for a triangle that covers no area after
    /// snapping. `planes` and `depth` are evaluated at physical pixel
    /// centers.
    pub fn new(
        positions: [[f32; 2]; 3],
        planes: Box<[Plane; 256]>,
        depth: Plane,
        front_facing: bool,
    ) -> Option<Triangle> {
        let mut vertices = positions.map(|[x, y]| [snap(x), snap(y)]);
        let area = orient(vertices[0], vertices[1], vertices[2]);
        if area == 0 {
            return None;
        }
        if area < 0 {
            vertices.swap(1, 2);
        }
        let mut bias = [0; 3];
        for (edge, bias) in bias.iter_mut().enumerate() {
            let a = vertices[edge];
            let b = vertices[(edge + 1) % 3];
            let dx = b[0] - a[0];
            let dy = b[1] - a[1];
            // With this orientation an edge going up (dy < 0) is a left
            // edge and a horizontal edge going right is a top edge.
            let top_left = dy < 0 || (dy == 0 && dx > 0);
            *bias = if top_left { 0 } else { -1 };
        }
        let min = |axis: usize| vertices.iter().map(|v| v[axis]).min().unwrap_or(0);
        let max = |axis: usize| vertices.iter().map(|v| v[axis]).max().unwrap_or(0);
        // Candidates are the pixels the triangle's extent touches: with
        // several samples per pixel a pixel whose center is outside may
        // still have a sample inside (`Triangle::coverage`).
        let first = |value: i64| (value as f64 / SUBPIXEL).floor();
        let x0 = first(min(0)).max(0.0);
        let y0 = first(min(1)).max(0.0);
        let x1 = first(max(0)) + 1.0;
        let y1 = first(max(1)) + 1.0;
        if x1 <= x0 || y1 <= y0 {
            return None;
        }
        let clamp = |value: f64| value.clamp(0.0, f64::from(u32::MAX)) as u32;
        Some(Triangle {
            vertices,
            bias,
            bounds: Rect {
                x0: clamp(x0),
                y0: clamp(y0),
                x1: clamp(x1),
                y1: clamp(y1),
            },
            planes,
            depth,
            front_facing,
        })
    }

    pub fn bounds(&self) -> Rect {
        self.bounds
    }

    /// The coverage test of [`Triangle::covers`] as data: the snapped
    /// vertices in 1/256 pixels, in the order the edges run, and per edge
    /// whether it is not top-left (so a pixel center exactly on it is
    /// outside).
    pub fn edges(&self) -> ([[i64; 2]; 3], [bool; 3]) {
        (self.vertices, self.bias.map(|b| b != 0))
    }

    /// Whether the center of pixel (`x`, `y`) is inside.
    #[inline]
    pub fn covers(&self, x: u32, y: u32) -> bool {
        self.covers_at(x, y, [8, 8])
    }

    /// Whether the point `position` (in 1/16 pixel from its corner) of
    /// pixel (`x`, `y`) is inside.
    #[inline]
    pub fn covers_at(&self, x: u32, y: u32, position: [u8; 2]) -> bool {
        let p = [
            (i64::from(x) << SUBPIXEL_BITS) + (i64::from(position[0]) << (SUBPIXEL_BITS - 4)),
            (i64::from(y) << SUBPIXEL_BITS) + (i64::from(position[1]) << (SUBPIXEL_BITS - 4)),
        ];
        (0..3).all(|edge| {
            orient(self.vertices[edge], self.vertices[(edge + 1) % 3], p) + self.bias[edge] >= 0
        })
    }

    /// The samples of pixel (`x`, `y`) inside: bit `s` for sample `s`.
    #[inline]
    pub fn coverage(&self, x: u32, y: u32, samples: &Samples) -> u16 {
        (0..samples.count()).fold(0, |mask, s| {
            mask | u16::from(self.covers_at(x, y, samples.positions[s])) << s
        })
    }

    /// The largest magnitude of the depth plane at the snapped vertices.
    pub fn max_depth(&self) -> f32 {
        self.vertices
            .iter()
            .map(|&[x, y]| evaluate(&self.depth, x as f32 / 256.0, y as f32 / 256.0).abs())
            .fold(0.0, f32::max)
    }
}

/// The samples alpha to coverage leaves of a fragment whose output 0
/// alpha is `alpha`: the first round(alpha × n) of the n samples (alpha
/// clamped to [0, 1], NaN 0) [public: OpenGL 4.6 core, 17.3.1: a mask
/// whose share of set bits is proportional to alpha, its pattern
/// implementation-dependent. \[assumption: NVIDIA's pattern, and the
/// dithering over 2x2 pixels `SET_ALPHA_TO_COVERAGE_DITHER_CONTROL`
/// selects, are not public; no dithering is modeled\]].
pub fn alpha_coverage(samples: &Samples, alpha: f32) -> u16 {
    let n = samples.count() as f32;
    let alpha = if alpha.is_nan() {
        0.0
    } else {
        alpha.clamp(0.0, 1.0)
    };
    let k = (alpha * n).round_ties_even() as u32;
    ((1u32 << k) - 1) as u16
}

/// Where `IPA` CENTROID evaluates in a pixel whose samples `mask` (bit
/// `s`: sample `s`) a primitive covers, as an offset from the pixel's
/// corner: the pixel center when all are covered (or none, a helper
/// lane), else the position of the covered sample of lowest index.
/// [public: OpenGL 4.6 core, 15.2.2: a centroid-sampled input is
/// evaluated at "any location within the pixel and the primitive",
/// the pixel center when the pixel is fully covered; which location the
/// hardware picks is not public. \[assumption: a covered sample, the
/// first; Direct3D 11's functional specification, 16.3.2, likewise
/// allows any covered sample position\]]
pub fn centroid(samples: &Samples, mask: u16) -> [f32; 2] {
    let all = (1u32 << samples.count()) - 1;
    if mask == 0 || u32::from(mask) == all {
        return [0.5, 0.5];
    }
    samples.offset(mask.trailing_zeros() as usize)
}

/// A plane through three points: value(x, y) = base + dx x + dy y.
pub fn plane(points: [[f32; 2]; 3], values: [f32; 3]) -> Plane {
    let [p0, p1, p2] = points.map(|[x, y]| [f64::from(x), f64::from(y)]);
    let [v0, v1, v2] = values.map(f64::from);
    let det = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]);
    if det == 0.0 {
        return Plane {
            dx: 0.0,
            dy: 0.0,
            base: values[0],
        };
    }
    let dx = ((v1 - v0) * (p2[1] - p0[1]) - (v2 - v0) * (p1[1] - p0[1])) / det;
    let dy = ((v2 - v0) * (p1[0] - p0[0]) - (v1 - v0) * (p2[0] - p0[0])) / det;
    Plane {
        dx: dx as f32,
        dy: dy as f32,
        base: (v0 - dx * p0[0] - dy * p0[1]) as f32,
    }
}

#[inline]
fn evaluate(plane: &Plane, x: f32, y: f32) -> f32 {
    plane.base + plane.dx * x + plane.dy * y
}

/// The bands of one row range: one per color window, plus depth.
pub struct BandSet<'a> {
    pub colors: Vec<Band<'a>>,
    pub depth: Option<Band<'a>>,
}

/// Rasterizes `triangles`, in order, into the rows of `bands`. Bands
/// are rows of texels; with several samples per pixel a band holds the
/// samples of `BAND_ROWS >> ly` rows of pixels (see `multisample`).
pub fn rasterize(
    backend: &Backend<'_>,
    env: &Env<'_>,
    triangles: &[Triangle],
    bands: &mut BandSet<'_>,
) -> Result<(), Unmodeled> {
    let Some(y0) = bands
        .colors
        .first()
        .map(|band| band.y0)
        .or(bands.depth.as_ref().map(|band| band.y0))
    else {
        return Ok(());
    };
    let samples = &backend.multisample.samples;
    let ly = samples.log2[1];
    let rows = Rect {
        x0: 0,
        y0: y0 >> ly,
        x1: u32::MAX,
        y1: (y0 + BAND_ROWS) >> ly,
    };
    for triangle in triangles {
        let area = triangle.bounds.intersect(&backend.clip).intersect(&rows);
        if area.is_empty() {
            continue;
        }
        // Quads are aligned to even coordinates.
        let mut qy = area.y0 & !1;
        while qy < area.y1 {
            let mut qx = area.x0 & !1;
            while qx < area.x1 {
                let mut coverage = [0u16; 4];
                for (lane, mask) in coverage.iter_mut().enumerate() {
                    let x = qx + (lane as u32 & 1);
                    let y = qy + (lane as u32 >> 1);
                    if x >= area.x0 && x < area.x1 && y >= area.y0 && y < area.y1 {
                        *mask = triangle.coverage(x, y, samples);
                    }
                }
                if coverage.iter().any(|&c| c != 0) {
                    shade_quad(backend, env, triangle, qx, qy, coverage, bands)?;
                }
                qx += 2;
            }
            qy += 2;
        }
    }
    Ok(())
}

/// Shades the quad at (`x`, `y`) whose lanes have samples `coverage`
/// inside the triangle, and runs each covered sample of each lane
/// through the stencil and depth tests and blending into its texel.
fn shade_quad(
    backend: &Backend<'_>,
    env: &Env<'_>,
    triangle: &Triangle,
    x: u32,
    y: u32,
    coverage: [u16; 4],
    bands: &mut BandSet<'_>,
) -> Result<(), Unmodeled> {
    let samples = &backend.multisample.samples;
    let mut quad = Quad {
        planes: &triangle.planes,
        x,
        y,
        covered: coverage.map(|mask| mask != 0),
        centroids: coverage.map(|mask| centroid(samples, mask)),
        front_facing: triangle.front_facing,
        outputs: [FragmentOutput::default(); 4],
    };
    backend.program.run_fragment(env, &mut quad)?;
    let [lx, ly] = samples.log2;
    let multisample = &backend.multisample;
    for (lane, output) in quad.outputs.iter().enumerate() {
        if coverage[lane] == 0 || output.discarded {
            continue;
        }
        if let Some((compare, reference)) = backend.alpha_test
            && !compare.test(output.colors[0][3], reference)
        {
            continue;
        }
        let mut mask = coverage[lane];
        if multisample.alpha_to_coverage {
            mask &= alpha_coverage(samples, output.colors[0][3]);
        }
        let mut output = *output;
        if multisample.alpha_to_one {
            for color in &mut output.colors {
                color[3] = 1.0;
            }
        }
        let output = &output;
        let px = x + (lane as u32 & 1);
        let py = y + (lane as u32 >> 1);
        for sample in (0..samples.count()).filter(|&s| mask & (1 << s) != 0) {
            let [cx, cy] = samples.cell(sample);
            let (tx, ty) = (px << lx | cx, py << ly | cy);
            // The stencil test, then the depth test; the stencil buffer is
            // updated by the operation the outcome selects, a fragment
            // failing either test writes no color (OpenGL 4.6 core,
            // 17.3.3-17.3.4).
            if let (Some(depth), Some(band)) = (backend.depth, bands.depth.as_mut()) {
                let [ox, oy] = if multisample.center_depth {
                    [0.5, 0.5]
                } else {
                    samples.offset(sample)
                };
                let incoming = output
                    .depth
                    .unwrap_or_else(|| evaluate(&triangle.depth, px as f32 + ox, py as f32 + oy));
                let incoming = incoming.clamp(depth.clamp.0, depth.clamp.1);
                let texel = band.pixel_mut(tx, ty);
                let stencil = depth.stencil.map(|stencil| {
                    let face = *stencil.face(triangle.front_facing);
                    let stored = depth.format.read_stencil(texel).unwrap_or(0);
                    (face, stored, face.test(stored))
                });
                let stencil_passed = stencil.is_none_or(|(_, _, passed)| passed);
                let depth_passed =
                    stencil_passed && depth.compare.test(incoming, depth.format.read_depth(texel));
                if let Some((face, stored, _)) = stencil {
                    let op = if !stencil_passed {
                        face.fail
                    } else if !depth_passed {
                        face.depth_fail
                    } else {
                        face.pass
                    };
                    depth.format.write_stencil(face.update(op, stored), texel);
                }
                if !depth_passed {
                    continue;
                }
                if depth.write {
                    depth.format.write_depth(incoming, texel);
                }
            }
            if let Some(passed) = backend.passed {
                passed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            for color in backend.colors {
                let band = &mut bands.colors[color.window];
                write_color(color, backend, output, band, tx, ty);
            }
        }
    }
    Ok(())
}

fn write_color(
    color: &ColorOutput,
    backend: &Backend<'_>,
    output: &FragmentOutput,
    band: &mut Band<'_>,
    x: u32,
    y: u32,
) {
    if !color.write.iter().any(|&w| w) {
        return;
    }
    let texel = band.pixel_mut(x, y);
    let stored = color.format.read(texel);
    let mut value = output.colors[color.output];
    // Integer targets are neither blended nor converted: blending is
    // defined on numbers the formats' conversions give [public: OpenGL
    // 4.6 §17.3.6.1, "Blending applies only if the color buffer has a
    // fixed-point or floating-point format"].
    let integer = color.format.is_integer();
    if let Some(blend) = color.blend.as_ref().filter(|_| !integer) {
        let mut destination = stored;
        if color.srgb {
            for component in &mut destination[..3] {
                *component = srgb_to_linear(*component);
            }
        }
        value = blend.apply(&Inputs {
            source: value,
            source1: output.colors[1],
            destination,
            constant: backend.blend_constant,
            zero_times_anything_is_zero: backend.blend_zero_rule,
        });
    }
    if color.srgb && !integer {
        for component in &mut value[..3] {
            *component = linear_to_srgb(*component);
        }
    }
    for component in 0..4 {
        if !color.write[component] {
            value[component] = stored[component];
        }
    }
    color.format.write(value, texel);
}
