//! Triangle rasterization and the fragment back end: coverage with the
//! top-left fill rule, fragment shading in 2x2 quads, depth test, blending
//! and render target writes.
//!
//! Positions are snapped to 1/256 pixel, the precision the netlist's
//! initial `SET_SNAP_GRID_NON_LINE` value 8 selects (`LOCATIONS_PER_PIXEL
//! 256X256`, rounding `RTNE`; nvservices netlist region 7, field values
//! [public: NVIDIA open-gpu-doc `classes/3d/clb197.h`]). Coverage is tested
//! at pixel centers with integer edge functions and the top-left rule of the
//! Direct3D 10+ rasterization rules [public: Direct3D 11 functional
//! specification, 3.4.1 "Triangle Rasterization Rules"], which NVIDIA
//! hardware implements; it is not pinned by the firmware. Coordinates here
//! are physical: x to the right, y down in memory row order.

use super::Unmodeled;
use super::blend::{Blend, Inputs};
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

/// The depth stage.
#[derive(Debug, Clone, Copy)]
pub struct DepthOutput {
    pub format: DepthFormat,
    pub compare: Compare,
    pub write: bool,
    /// Range the fragment depth is clamped to before the test.
    pub clamp: (f32, f32),
}

/// Everything fragment processing of one draw needs besides the bands.
pub struct Backend<'a> {
    pub program: &'a Program,
    pub colors: &'a [ColorOutput],
    pub depth: Option<DepthOutput>,
    pub blend_constant: [f32; 4],
    /// `SET_BLEND_FLOAT_OPTION` `ZERO_TIMES_ANYTHING_IS_ZERO`.
    pub blend_zero_rule: bool,
    /// Pixels that may be written (scissor, surface clip, targets).
    pub clip: Rect,
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
        // Pixel p is sampled at p + 0.5: the first candidate is the pixel
        // whose center is at or after the minimum.
        let first = |value: i64| ((value - (1 << (SUBPIXEL_BITS - 1))) as f64 / SUBPIXEL).ceil();
        let last = |value: i64| ((value - (1 << (SUBPIXEL_BITS - 1))) as f64 / SUBPIXEL).floor();
        let x0 = first(min(0)).max(0.0);
        let y0 = first(min(1)).max(0.0);
        let x1 = last(max(0)) + 1.0;
        let y1 = last(max(1)) + 1.0;
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

    /// Whether the center of pixel (`x`, `y`) is inside.
    #[inline]
    pub fn covers(&self, x: u32, y: u32) -> bool {
        let half = 1i64 << (SUBPIXEL_BITS - 1);
        let p = [
            (i64::from(x) << SUBPIXEL_BITS) + half,
            (i64::from(y) << SUBPIXEL_BITS) + half,
        ];
        (0..3).all(|edge| {
            orient(self.vertices[edge], self.vertices[(edge + 1) % 3], p) + self.bias[edge] >= 0
        })
    }
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

/// Rasterizes `triangles`, in order, into the rows of `bands`.
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
    let rows = Rect {
        x0: 0,
        y0,
        x1: u32::MAX,
        y1: y0 + BAND_ROWS,
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
                let mut covered = [false; 4];
                for (lane, covered) in covered.iter_mut().enumerate() {
                    let x = qx + (lane as u32 & 1);
                    let y = qy + (lane as u32 >> 1);
                    *covered = x >= area.x0
                        && x < area.x1
                        && y >= area.y0
                        && y < area.y1
                        && triangle.covers(x, y);
                }
                if covered.iter().any(|&c| c) {
                    shade_quad(backend, env, triangle, qx, qy, covered, bands)?;
                }
                qx += 2;
            }
            qy += 2;
        }
    }
    Ok(())
}

fn shade_quad(
    backend: &Backend<'_>,
    env: &Env<'_>,
    triangle: &Triangle,
    x: u32,
    y: u32,
    covered: [bool; 4],
    bands: &mut BandSet<'_>,
) -> Result<(), Unmodeled> {
    let mut quad = Quad {
        planes: &triangle.planes,
        x,
        y,
        covered,
        front_facing: triangle.front_facing,
        outputs: [FragmentOutput::default(); 4],
    };
    backend.program.run_fragment(env, &mut quad)?;
    for (lane, output) in quad.outputs.iter().enumerate() {
        if !covered[lane] || output.discarded {
            continue;
        }
        let px = x + (lane as u32 & 1);
        let py = y + (lane as u32 >> 1);
        if let (Some(depth), Some(band)) = (backend.depth, bands.depth.as_mut()) {
            let incoming = output
                .depth
                .unwrap_or_else(|| evaluate(&triangle.depth, px as f32 + 0.5, py as f32 + 0.5));
            let incoming = incoming.clamp(depth.clamp.0, depth.clamp.1);
            let texel = band.pixel_mut(px, py);
            let stored = depth.format.read_depth(texel);
            if !depth.compare.test(incoming, stored) {
                continue;
            }
            if depth.write {
                depth.format.write_depth(incoming, texel);
            }
        }
        for color in backend.colors {
            let band = &mut bands.colors[color.window];
            write_color(color, backend, output, band, px, py);
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
    if let Some(blend) = &color.blend {
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
    if color.srgb {
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
