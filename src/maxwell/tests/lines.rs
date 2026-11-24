//! Point and line rasterization: the diamond-exit rule of non-antialiased
//! lines against hand-worked cases and a direct test of its definition,
//! rectangular lines and points through the whole pipeline, and the
//! software and Metal renderers drawing the same lines.

use std::collections::BTreeSet;

use super::super::Engines;
use super::super::RendererKind::{self, Metal, Software};
use super::super::pipeline::Topology;
use super::super::raster::{SUBPIXEL_BITS, aliased_line};
use super::super::surface::Rect;
use super::{Frame, SIZE, STARTER, floats, starter_program, ui2d_vertex_constants};

const AREA: Rect = Rect {
    x0: 0,
    y0: 0,
    x1: 64,
    y1: 64,
};

/// The pixels of a width-1 non-antialiased line.
fn pixels(a: [f32; 2], b: [f32; 2], width: u32) -> BTreeSet<(i64, i64)> {
    let mut set = BTreeSet::new();
    for [x0, y0, x1, y1] in aliased_line(a, b, width, AREA) {
        for y in y0..y1 {
            for x in x0..x1 {
                assert!(set.insert((x, y)), "pixel {x},{y} twice");
            }
        }
    }
    set
}

fn set(pixels: &[(i64, i64)]) -> BTreeSet<(i64, i64)> {
    pixels.iter().copied().collect()
}

fn row(xs: std::ops::Range<i64>, y: i64) -> BTreeSet<(i64, i64)> {
    xs.map(|x| (x, y)).collect()
}

/// Cases worked by hand from the definition, the perturbation by
/// (-ε, -ε²) deciding endpoints and lines on pixel corners and edges.
#[test]
fn aliased_lines_follow_the_diamond_exit_rule() {
    // Between pixel centers: the start pixel is drawn, the end pixel is
    // not ("half-open"), whichever way the line runs.
    assert_eq!(pixels([8.5, 8.5], [12.5, 8.5], 1), row(8..12, 8));
    assert_eq!(pixels([12.5, 8.5], [8.5, 8.5], 1), row(9..13, 8));
    // On a pixel edge: -ε² puts a horizontal line in the row above, -ε a
    // vertical one in the column to the left.
    assert_eq!(pixels([8.5, 9.0], [12.5, 9.0], 1), row(8..12, 8));
    assert_eq!(
        pixels([9.0, 8.5], [9.0, 12.5], 1),
        set(&[(8, 8), (8, 9), (8, 10), (8, 11)])
    );
    // Corner to corner: one pixel per column, through the centers.
    assert_eq!(
        pixels([8.0, 8.0], [12.0, 12.0], 1),
        set(&[(8, 8), (9, 9), (10, 10), (11, 11)])
    );
    // A start on a pixel corner is moved by -ε into the diamond of the
    // pixel to its left, an end likewise: the run shifts left by one.
    assert_eq!(pixels([8.0, 8.5], [10.0, 8.5], 1), row(7..9, 8));
    // Through a pixel edge at a column's center: s ε - ε² takes a rising
    // line to the row below the edge, a falling one to the row above.
    assert_eq!(
        pixels([8.5, 8.0], [12.5, 10.0], 1),
        set(&[(8, 8), (9, 8), (10, 9), (11, 9)])
    );
    assert_eq!(
        pixels([8.5, 10.0], [12.5, 8.0], 1),
        set(&[(8, 9), (9, 9), (10, 8), (11, 8)])
    );
    // Wide lines: the segment moved up by (w - 1) / 2, each pixel a
    // column of w.
    let wide = |rows: std::ops::Range<i64>| -> BTreeSet<_> {
        rows.flat_map(|y| (8..11).map(move |x| (x, y))).collect()
    };
    assert_eq!(pixels([8.5, 10.5], [11.5, 10.5], 3), wide(9..12));
    assert_eq!(pixels([8.5, 10.5], [11.5, 10.5], 2), wide(9..11));
    // Zero length: nothing.
    assert!(pixels([8.3, 8.3], [8.3, 8.3], 1).is_empty());
    // Only columns inside the area are produced.
    let clipped: Vec<_> = aliased_line(
        [8.5, 8.5],
        [12.5, 8.5],
        1,
        Rect {
            x0: 10,
            y0: 0,
            x1: 11,
            y1: 64,
        },
    );
    assert_eq!(clipped, [[10, 8, 11, 9]]);
}

/// Connected x-major segments run the same way neither repeat nor skip
/// a column (a rule both APIs state for the diamond exit).
#[test]
fn aliased_strips_share_endpoints_once() {
    let points = [[8.5, 8.5], [12.25, 10.75], [16.5, 9.5], [21.0, 9.0]];
    let mut columns = Vec::new();
    for pair in points.windows(2) {
        let segment = pixels(pair[0], pair[1], 1);
        columns.extend(segment.iter().map(|&(x, _)| x));
    }
    columns.sort_unstable();
    assert_eq!(columns, (8..21).collect::<Vec<_>>());
}

/// The diamond test of one pixel, straight from the definition in 1/256
/// pixel units: whether the segment meets the open diamond around
/// `center` and its end is outside it. `None` when an endpoint or the
/// segment only touches the diamond's boundary, where the perturbation
/// decides.
fn diamond_reference(a: [i64; 2], b: [i64; 2], center: [i64; 2]) -> Option<bool> {
    let half = 1i128 << (SUBPIXEL_BITS - 1);
    let [ax, ay, bx, by, cx, cy] = [a[0], a[1], b[0], b[1], center[0], center[1]].map(i128::from);
    let h = |x: i128, y: i128| (x - cx).abs() + (y - cy).abs();
    // Minimum of the convex |x - cx| + |y - cy| over the segment: at an
    // endpoint, or where x = cx or y = cy. Values are kept as fractions
    // (numerator, denominator > 0).
    let mut candidates = vec![(h(ax, ay), 1), (h(bx, by), 1)];
    let (dx, dy) = (bx - ax, by - ay);
    if dx != 0 && (ax.min(bx)..=ax.max(bx)).contains(&cx) {
        let numerator = ay * dx + (cx - ax) * dy - cy * dx;
        candidates.push((numerator.abs(), dx.abs()));
    }
    if dy != 0 && (ay.min(by)..=ay.max(by)).contains(&cy) {
        let numerator = ax * dy + (cy - ay) * dx - cx * dy;
        candidates.push((numerator.abs(), dy.abs()));
    }
    let below = |(n, d): (i128, i128)| (n < half * d, n == half * d);
    let mut meets = false;
    for candidate in candidates {
        let (less, equal) = below(candidate);
        if equal {
            return None;
        }
        meets |= less;
    }
    let (end_inside, end_on) = below((h(bx, by), 1));
    if end_on {
        return None;
    }
    Some(meets && !end_inside)
}

/// Random segments whose endpoints and diamonds do not touch: the pixels
/// produced are exactly those the definition gives.
#[test]
fn aliased_lines_match_the_definition() {
    let unit = 1i64 << SUBPIXEL_BITS;
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut random = |range: i64| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % range as u64) as i64
    };
    let mut checked = 0;
    for _ in 0..2000 {
        let a = [random(12 * unit) + unit, random(12 * unit) + unit];
        let b = [random(12 * unit) + unit, random(12 * unit) + unit];
        let mut expected = BTreeSet::new();
        let mut decided = true;
        for y in 0..16 {
            for x in 0..16 {
                let center = [x * unit + unit / 2, y * unit + unit / 2];
                match diamond_reference(a, b, center) {
                    Some(true) => {
                        expected.insert((x, y));
                    }
                    Some(false) => {}
                    None => decided = false,
                }
            }
        }
        if !decided {
            continue;
        }
        let to_window = |p: [i64; 2]| p.map(|v| v as f32 / unit as f32);
        assert_eq!(
            pixels(to_window(a), to_window(b), 1),
            expected,
            "{a:?} -> {b:?}"
        );
        checked += 1;
    }
    assert!(checked > 1500, "only {checked} segments without ties");
}

/// A 64x64 frame with the ui2d solid-color programs (color (0.25, 0.5,
/// 1, 1)) and the line state the touch screen test leaves: line snap grid
/// 256x256 (the netlist's), multisample rasterization off, the aliased
/// width separate, both widths 1.
fn line_frame(kind: RendererKind) -> Frame {
    let elf = std::fs::read(STARTER).expect("starter ELF (fw_elf) must be present");
    let mut frame = Frame::sized_with(SIZE, SIZE, kind);
    frame.program(1, 0, starter_program(&elf, 0x4c0e00));
    frame.program(5, 0x800, starter_program(&elf, 0x4be400));
    frame.constants(0, 3, 0x30_0000, &ui2d_vertex_constants(&[]));
    frame.constants(4, 3, 0x30_1000, &floats(&[0.25, 0.5, 1.0, 1.0]));
    for (offset, value) in [
        (0x0318, 8),
        (0x020c, 1),
        (0x13b0, 1f32.to_bits()),
        (0x13b4, 1f32.to_bits()),
    ] {
        frame.method(offset, value);
    }
    frame
}

/// Draws `points` (window positions) with `topology` and writes back.
fn draw(frame: &mut Frame, topology: u32, points: &[[f32; 2]]) {
    let half = SIZE as f32 / 2.0;
    let data: Vec<f32> = points
        .iter()
        .flat_map(|&[x, y]| [(x - half) / half, (y - half) / half, 0.0])
        .collect();
    frame.vertices(0x40_0000, 12, &data, &[(0, 0x02)]);
    frame.method(0x1618, topology);
    frame.method(0x0d74, 0);
    frame.method(0x0d78, points.len() as u32);
    frame.method(0x1614, 0);
    frame.channel.flush(&frame.ram).unwrap();
}

/// The pixels drawn (not the cleared black).
fn drawn(frame: &Frame) -> BTreeSet<(i64, i64)> {
    let mut set = BTreeSet::new();
    for y in 0..SIZE {
        for x in 0..SIZE {
            let pixel = frame.pixel(x, y);
            if pixel != [0, 0, 0, 255] {
                assert_eq!(pixel, [64, 128, 255, 255], "{x},{y}");
                set.insert((i64::from(x), i64::from(y)));
            }
        }
    }
    set
}

/// A line loop through the pipeline draws the diamond-exit pixels of
/// each segment, the closing one included.
#[test]
fn line_loop_draws_its_segments() {
    let points = [[10.5, 10.5], [40.25, 14.75], [30.0, 50.0]];
    let mut frame = line_frame(Software);
    draw(&mut frame, 2, &points);
    let mut expected = BTreeSet::new();
    for i in 0..points.len() {
        expected.extend(pixels(points[i], points[(i + 1) % points.len()], 1));
    }
    assert_eq!(drawn(&frame), expected);
    // As a strip, the closing segment is not drawn; as separate lines,
    // the third point has no partner.
    let mut frame = line_frame(Software);
    draw(&mut frame, 3, &points);
    let strip: BTreeSet<_> = pixels(points[0], points[1], 1)
        .union(&pixels(points[1], points[2], 1))
        .copied()
        .collect();
    assert_eq!(drawn(&frame), strip);
    let mut frame = line_frame(Software);
    draw(&mut frame, 1, &points);
    assert_eq!(drawn(&frame), pixels(points[0], points[1], 1));
}

/// With multisample rasterization on, lines are rectangles of
/// `SET_LINE_WIDTH_FLOAT` sampled at pixel centers.
#[test]
fn rectangular_lines_cover_the_centers_inside() {
    let mut frame = line_frame(Software);
    frame.method(0x1534, 1); // SET_ANTI_ALIAS_ENABLE
    frame.method(0x13b0, 3f32.to_bits());
    draw(&mut frame, 1, &[[5.0, 30.25], [25.0, 30.25]]);
    let expected: BTreeSet<_> = (29..32)
        .flat_map(|y| (5..25).map(move |x| (x, y)))
        .collect();
    assert_eq!(drawn(&frame), expected);
    // A slanted one: the centers within half the width of the segment,
    // between its ends (none within 1/64 pixel of an edge).
    let (a, b, width) = ([40.1f64, 5.3], [60.7f64, 25.9], 2.0);
    let mut frame = line_frame(Software);
    frame.method(0x1534, 1);
    frame.method(0x13b0, (width as f32).to_bits());
    draw(&mut frame, 1, &[a.map(|v| v as f32), b.map(|v| v as f32)]);
    let d = [b[0] - a[0], b[1] - a[1]];
    let length = d[0].hypot(d[1]);
    let mut expected = BTreeSet::new();
    for y in 0..SIZE {
        for x in 0..SIZE {
            let p = [f64::from(x) + 0.5 - a[0], f64::from(y) + 0.5 - a[1]];
            let across = (p[0] * d[1] - p[1] * d[0]).abs() / length;
            let along = (p[0] * d[0] + p[1] * d[1]) / length;
            let margins = [width / 2.0 - across, along, length - along];
            assert!(margins.iter().all(|m| m.abs() > 1.0 / 64.0), "{x},{y}");
            if margins.iter().all(|&m| m > 0.0) {
                expected.insert((i64::from(x), i64::from(y)));
            }
        }
    }
    assert_eq!(drawn(&frame), expected);
}

/// Points: without sprites or multisampling, the size rounded and the
/// square put on the pixel (odd) or the nearest corner (even); with
/// sprites, a square of the exact size around the point.
#[test]
fn points_cover_their_squares() {
    let square = |x: std::ops::Range<i64>, y: std::ops::Range<i64>| -> BTreeSet<_> {
        y.flat_map(|y| x.clone().map(move |x| (x, y))).collect()
    };
    let mut frame = line_frame(Software);
    frame.method(0x1518, 3.2f32.to_bits()); // SET_POINT_SIZE
    draw(&mut frame, 0, &[[10.3, 20.7]]);
    assert_eq!(drawn(&frame), square(9..12, 19..22));
    let mut frame = line_frame(Software);
    frame.method(0x1518, 1.6f32.to_bits());
    draw(&mut frame, 0, &[[30.3, 40.7]]);
    assert_eq!(drawn(&frame), square(29..31, 40..42));
    let mut frame = line_frame(Software);
    frame.method(0x1520, 1); // SET_POINT_SPRITE
    frame.method(0x1518, 2.5f32.to_bits());
    draw(&mut frame, 0, &[[50.2, 10.0]]);
    assert_eq!(drawn(&frame), square(49..51, 9..11));
    // Round points (multisampling without sprites) are refused.
    let mut frame = line_frame(Software);
    frame.method(0x1534, 1);
    frame.method(0x1518, 2f32.to_bits());
    frame.vertices(0x40_0000, 12, &[0.0, 0.0, 0.0], &[(0, 0x02)]);
    frame.method(0x1618, 0);
    frame.method(0x0d74, 0);
    frame.method(0x0d78, 1);
    let end = frame
        .channel
        .call(0, 0x1614 / 4, 0, &frame.ram)
        .and_then(|()| frame.channel.draw_held(&frame.ram));
    assert!(end.unwrap_err().0.contains("round"));
}

/// What stays refused: adjacency, patches, smooth and stippled lines.
#[test]
fn unmodeled_line_state_is_refused() {
    assert!(Topology::from_raw(0xA).unwrap_err().0.contains("adjacency"));
    assert!(Topology::from_raw(0xE).unwrap_err().0.contains("patches"));
    for (offset, what) in [(0x1570, "smooth lines"), (0x166c, "line stipple")] {
        let mut frame = line_frame(Software);
        frame.method(offset, 1);
        frame.vertices(0x40_0000, 12, &[0.0, 0.0, 0.0, 0.5, 0.5, 0.0], &[(0, 0x02)]);
        frame.method(0x1618, 1);
        frame.method(0x0d74, 0);
        frame.method(0x0d78, 2);
        let end = frame
            .channel
            .call(0, 0x1614 / 4, 0, &frame.ram)
            .and_then(|()| frame.channel.draw_held(&frame.ram));
        assert!(end.unwrap_err().0.contains(what), "{what}");
    }
}

/// Metal draws the same pixels as the software renderer, for a
/// non-antialiased line loop and a wide rectangular line strip, without
/// falling back.
#[test]
fn software_and_metal_lines_agree() {
    let image = |kind: RendererKind, rectangular: bool| {
        let mut frame = line_frame(kind);
        let points: &[[f32; 2]] = if rectangular {
            frame.method(0x1534, 1);
            frame.method(0x13b0, 2.5f32.to_bits());
            &[[3.3, 60.1], [20.7, 33.2], [44.0, 58.9], [61.5, 4.25]]
        } else {
            &[
                [2.5, 3.5],
                [61.0, 9.0],
                [50.25, 62.75],
                [5.0, 40.0],
                [33.3, 20.6],
            ]
        };
        draw(&mut frame, if rectangular { 3 } else { 2 }, points);
        if let Some(metal) = &frame.channel.hooks.metal {
            assert_eq!(metal.fallbacks(), (1, vec![]));
        }
        drawn(&frame)
    };
    for rectangular in [false, true] {
        let software = image(Software, rectangular);
        assert!(software.len() > 100);
        assert_eq!(
            image(Metal, rectangular),
            software,
            "rectangular {rectangular}"
        );
    }
}

/// Points of odd and even sizes, and point sprites, at positions on and
/// next to pixel centers and edges (sprites of size 1 and 2 centered on
/// pixel corners and centers put pixel centers on their edges, which the
/// top-left rule decides): Metal draws them with its vertex stage (no
/// CPU vertices) and covers the pixels the software renderer does.
#[test]
fn software_and_metal_points_agree() {
    let points: &[[f32; 2]] = &[
        [10.3, 20.7],
        [30.5, 40.5],
        [50.0, 10.0],
        [5.49, 55.51],
        [44.75, 30.25],
        [20.0, 50.5],
    ];
    for (size, sprite) in [
        (1.0f32, false),
        (1.6, false),
        (3.2, false),
        (4.0, false),
        (1.0, true),
        (2.0, true),
        (2.5, true),
        (3.3, true),
    ] {
        let image = |kind: RendererKind| {
            let mut frame = line_frame(kind);
            frame.method(0x1518, size.to_bits()); // SET_POINT_SIZE
            frame.method(0x1520, u32::from(sprite)); // SET_POINT_SPRITE
            draw(&mut frame, 0, points);
            if let Some(metal) = &frame.channel.hooks.metal {
                assert_eq!(metal.fallbacks(), (1, vec![]), "size {size}");
                assert_eq!(metal.cpu_vertices(), vec![], "size {size}");
            }
            drawn(&frame)
        };
        let software = image(Software);
        assert!(software.len() >= points.len(), "size {size} sprite {sprite}");
        assert_eq!(image(Metal), software, "size {size} sprite {sprite}");
    }
}
