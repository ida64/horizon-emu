//! Tests across the engines: rasterization rules, surface layouts, the
//! copy, 2D and inline-to-memory engines, clears, and an end-to-end frame
//! drawn with real firmware shaders.

use std::sync::Mutex;

use super::raster::Triangle;
use super::shader::Plane;
use super::surface::{Layout, Rect, Surface, Window};
use super::texture::layout::BlockLinear;
use super::{Channel, Engines, GpuMemory};

/// Flat memory at GPU address 0.
pub struct Ram(pub Mutex<Vec<u8>>);

impl Ram {
    pub fn new(size: usize) -> Ram {
        Ram(Mutex::new(vec![0; size]))
    }

    pub fn bytes(&self, address: u64, length: usize) -> Vec<u8> {
        self.0.lock().unwrap()[address as usize..][..length].to_vec()
    }

    pub fn put(&self, address: u64, data: &[u8]) {
        self.0.lock().unwrap()[address as usize..][..data.len()].copy_from_slice(data);
    }

    pub fn word(&self, address: u64) -> u32 {
        u32::from_le_bytes(self.bytes(address, 4).try_into().unwrap())
    }
}

impl GpuMemory for Ram {
    fn read(&self, address: u64, buffer: &mut [u8]) -> Option<()> {
        let ram = self.0.lock().unwrap();
        let start = usize::try_from(address).ok()?;
        buffer.copy_from_slice(ram.get(start..start.checked_add(buffer.len())?)?);
        Some(())
    }

    fn write(&self, address: u64, data: &[u8]) -> Option<()> {
        let mut ram = self.0.lock().unwrap();
        let start = usize::try_from(address).ok()?;
        ram.get_mut(start..start.checked_add(data.len())?)?
            .copy_from_slice(data);
        Some(())
    }
}

/// Block-linear offset written from the Tegra GOB description, independent
/// of `texture::layout`: GOBs of 64 bytes x 8 rows, bytes inside a GOB in
/// 16-byte rows paired two by two, blocks of `2^block_height` GOBs stacked
/// vertically, blocks in row-major order.
fn reference_offset(x: u32, y: u32, width_bytes: u32, block_height: u32) -> u64 {
    let gobs_per_row = width_bytes.div_ceil(64) as u64;
    let block_rows = 8 << block_height;
    let block = (y / block_rows) as u64 * gobs_per_row + (x / 64) as u64;
    let gob_in_block = ((y % block_rows) / 8) as u64;
    let (gx, gy) = ((x % 64) as u64, (y % 8) as u64);
    let inner = (gx / 32) * 256 + (gy / 2) * 64 + ((gx % 32) / 16) * 32 + (gy % 2) * 16 + gx % 16;
    block * (512 << block_height) + gob_in_block * 512 + inner
}

#[test]
fn block_linear_windows_match_the_reference_layout() {
    for (block_height, width, height, bpp) in [
        (0, 40, 20, 4),
        (2, 33, 70, 4),
        (4, 300, 140, 1),
        (1, 17, 9, 16),
    ] {
        let width_bytes = width * bpp;
        let size = reference_offset(width_bytes - 1, height - 1, width_bytes, block_height)
            as usize
            + (512 << block_height);
        let ram = Ram::new(0x1000 + size + 0x1000);
        let base = 0x1000;
        // Every byte holds a hash of its (x_byte, y) position.
        for y in 0..height {
            for x in 0..width_bytes {
                let value = (x.wrapping_mul(31) ^ y.wrapping_mul(17)) as u8;
                ram.put(
                    base + reference_offset(x, y, width_bytes, block_height),
                    &[value],
                );
            }
        }
        let surface = Surface {
            address: base,
            width,
            height,
            bytes_per_pixel: bpp,
            layout: Layout::BlockLinear(BlockLinear {
                block_height_log2: block_height,
                block_depth_log2: 0,
            }),
        };
        let rect = Rect {
            x0: 3,
            y0: 5,
            x1: width,
            y1: height,
        };
        let mut window = Window::load(&ram, surface, rect).unwrap();
        for band in window.bands_mut() {
            for y in band.y0.max(rect.y0)..(band.y0 + 8).min(height) {
                for x in rect.x0..width {
                    let pixel = band.pixel(x, y);
                    for (k, &byte) in pixel.iter().enumerate() {
                        let xb = x * bpp + k as u32;
                        assert_eq!(
                            byte,
                            (xb.wrapping_mul(31) ^ y.wrapping_mul(17)) as u8,
                            "{x},{y}"
                        );
                    }
                }
            }
        }
    }
}

fn no_planes() -> Box<[Plane; 256]> {
    Box::new([Plane::default(); 256])
}

fn triangle(points: [[f32; 2]; 3]) -> Option<Triangle> {
    Triangle::new(points, no_planes(), Plane::default(), true)
}

fn coverage(triangles: &[Triangle], size: u32) -> Vec<u32> {
    let mut count = vec![0; (size * size) as usize];
    for t in triangles {
        for y in 0..size {
            for x in 0..size {
                if t.covers(x, y) {
                    count[(y * size + x) as usize] += 1;
                }
            }
        }
    }
    count
}

#[test]
fn shared_edges_cover_every_pixel_exactly_once() {
    // A fan around an interior point, with vertices on pixel centers,
    // pixel corners and arbitrary positions: every pixel center inside the
    // outline is covered once, none twice.
    let center = [8.5, 8.5];
    let outline = [
        [1.0, 1.0],
        [8.5, 0.5],
        [15.0, 1.0],
        [15.5, 8.5],
        [15.0, 15.0],
        [7.25, 15.75],
        [1.0, 15.0],
        [0.5, 8.0],
    ];
    let triangles: Vec<Triangle> = (0..outline.len())
        .filter_map(|i| triangle([center, outline[i], outline[(i + 1) % outline.len()]]))
        .collect();
    let count = coverage(&triangles, 17);
    assert!(count.iter().all(|&c| c <= 1));
    // The same fan with the opposite winding covers the same pixels.
    let reversed: Vec<Triangle> = (0..outline.len())
        .filter_map(|i| triangle([center, outline[(i + 1) % outline.len()], outline[i]]))
        .collect();
    assert_eq!(coverage(&reversed, 17), count);
    // A quad split along either diagonal covers the same pixels.
    let quad = [[2.0, 2.0], [10.0, 2.0], [10.0, 10.0], [2.0, 10.0]];
    let a = [
        triangle([quad[0], quad[1], quad[2]]).unwrap(),
        triangle([quad[0], quad[2], quad[3]]).unwrap(),
    ];
    let b = [
        triangle([quad[0], quad[1], quad[3]]).unwrap(),
        triangle([quad[1], quad[2], quad[3]]).unwrap(),
    ];
    let ca = coverage(&a, 12);
    assert_eq!(ca, coverage(&b, 12));
    assert!(ca.iter().all(|&c| c <= 1));
    // Pixels 2..10 in each axis: the 8x8 interior, centers at 2.5..9.5.
    assert_eq!(ca.iter().sum::<u32>(), 64);
}

#[test]
fn top_left_rule_on_pixel_centers() {
    // An axis-aligned square whose edges pass exactly through pixel
    // centers: the top and left edges include them, bottom and right do
    // not.
    let square = [[1.5, 1.5], [4.5, 1.5], [4.5, 4.5], [1.5, 4.5]];
    let triangles = [
        triangle([square[0], square[1], square[2]]).unwrap(),
        triangle([square[0], square[2], square[3]]).unwrap(),
    ];
    let count = coverage(&triangles, 6);
    for y in 0..6 {
        for x in 0..6 {
            let inside = (1..4).contains(&x) && (1..4).contains(&y);
            assert_eq!(count[(y * 6 + x) as usize], inside as u32, "{x},{y}");
        }
    }
    // Degenerate and sub-pixel triangles cover nothing.
    assert!(triangle([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]).is_none());
    let tiny = triangle([[2.6, 2.6], [2.9, 2.6], [2.6, 2.9]]);
    assert!(tiny.is_none_or(|t| coverage(&[t], 4).iter().all(|&c| c == 0)));
}

/// Writes method `offset` (byte offset) on subchannel `subchannel`.
pub fn call(
    channel: &mut Channel,
    memory: &dyn GpuMemory,
    subchannel: u8,
    offset: u32,
    value: u32,
) {
    channel
        .call(subchannel, offset / 4, value, memory)
        .unwrap_or_else(|e| panic!("method {offset:#x} = {value:#x}: {}", e.0));
}

#[test]
fn inline_to_memory_pitch_and_block_linear() {
    let ram = Ram::new(0x10000);
    let mut channel = Channel::new();
    call(&mut channel, &ram, 2, 0, 0xA140);
    // Pitch: 2 lines of 8 bytes, pitch 0x100, at 0x1000; then a one-word
    // semaphore release of 0xCAFE at 0x3000.
    for (method, value) in [
        (0x180, 8),
        (0x184, 2),
        (0x188, 0),
        (0x18c, 0x1000),
        (0x190, 0x100),
        (0x1dc, 0),
        (0x1e0, 0x3000),
        (0x1e4, 0xCAFE),
    ] {
        call(&mut channel, &ram, 2, method, value);
    }
    call(&mut channel, &ram, 2, 0x1b0, 1 | 2 << 4 | 1 << 12);
    for word in [0x0302_0100u32, 0x0706_0504, 0x1312_1110, 0x1716_1514] {
        call(&mut channel, &ram, 2, 0x1b4, word);
    }
    assert_eq!(ram.bytes(0x1000, 8), (0..8).collect::<Vec<u8>>());
    assert_eq!(ram.bytes(0x1100, 8), (0x10..0x18).collect::<Vec<u8>>());
    assert_eq!(ram.word(0x3000), 0xCAFE);
    // Block linear: 16 bytes at (x 64, y 3) of a 128-byte-wide surface,
    // block height 1 GOB.
    for (method, value) in [
        (0x180, 16),
        (0x184, 1),
        (0x18c, 0x4000),
        (0x194, 0),
        (0x198, 128),
        (0x19c, 8),
        (0x1a0, 1),
        (0x1a4, 0),
        (0x1a8, 64),
        (0x1ac, 3),
    ] {
        call(&mut channel, &ram, 2, method, value);
    }
    call(&mut channel, &ram, 2, 0x1b0, 0);
    for word in [0x0302_0100u32, 0x0706_0504, 0x0B0A_0908, 0x0F0E_0D0C] {
        call(&mut channel, &ram, 2, 0x1b4, word);
    }
    for x in 0..16 {
        let at = 0x4000 + reference_offset(64 + x, 3, 128, 0);
        assert_eq!(ram.bytes(at, 1)[0], x as u8);
    }
}

#[test]
fn copy_engine_round_trips_through_block_linear_with_remap() {
    let ram = Ram::new(0x40000);
    let mut channel = Channel::new();
    call(&mut channel, &ram, 4, 0, 0xB0B5);
    // A 20x10 RGBA8 image, pitch 96 at 0x1000.
    let (width, height, pitch) = (20u32, 10u32, 96u32);
    for y in 0..height {
        for x in 0..width {
            ram.put(
                0x1000 + (y * pitch + x * 4) as u64,
                &[x as u8, y as u8, (x ^ y) as u8, 0xFF],
            );
        }
    }
    let common = |channel: &mut Channel| {
        call(channel, &ram, 4, 0x400, 0);
        call(channel, &ram, 4, 0x408, 0);
        call(channel, &ram, 4, 0x418, width);
        call(channel, &ram, 4, 0x41c, height);
        // Remap: 4 components of 1 byte; destination x <- source z (2),
        // y <- y (1), z <- x (0), w <- w (3).
        call(
            channel,
            &ram,
            4,
            0x708,
            2 | 1 << 4 | 3 << 12 | 3 << 20 | 3 << 24,
        );
    };
    // Pitch -> block linear at 0x10000 (block height 2 GOBs, FERMI_8),
    // origin (4, 3) of a 32x16 surface.
    common(&mut channel);
    for (method, value) in [
        (0x404, 0x1000),
        (0x40c, 0x10000),
        (0x410, pitch),
        (0x70c, 1 << 4 | 1 << 12),
        (0x710, 32),
        (0x714, 16),
        (0x718, 1),
        (0x71c, 0),
        (0x720, 4 | 3 << 16),
    ] {
        call(&mut channel, &ram, 4, method, value);
    }
    // MULTI_LINE, REMAP, source pitch, destination block linear, pipelined.
    call(&mut channel, &ram, 4, 0x300, 1 | 1 << 7 | 1 << 9 | 1 << 10);
    for y in 0..height {
        for x in 0..width {
            let at = 0x10000 + reference_offset((x + 4) * 4, y + 3, 32 * 4, 1);
            assert_eq!(
                ram.bytes(at, 4),
                vec![(x ^ y) as u8, y as u8, x as u8, 0xFF]
            );
        }
    }
    // Block linear -> pitch at 0x20000 with the same swap undoes it; with
    // a one-word semaphore release.
    common(&mut channel);
    for (method, value) in [
        (0x404, 0x10000),
        (0x40c, 0x20000),
        (0x414, pitch),
        (0x728, 1 << 4 | 1 << 12),
        (0x72c, 32),
        (0x730, 16),
        (0x734, 1),
        (0x738, 0),
        (0x73c, 4 | 3 << 16),
        (0x240, 0),
        (0x244, 0x30000),
        (0x248, 0x1234),
    ] {
        call(&mut channel, &ram, 4, method, value);
    }
    call(
        &mut channel,
        &ram,
        4,
        0x300,
        1 | 1 << 3 | 1 << 8 | 1 << 9 | 1 << 10,
    );
    for y in 0..height {
        let row = (y * pitch) as u64;
        assert_eq!(
            ram.bytes(0x20000 + row, (width * 4) as usize),
            ram.bytes(0x1000 + row, (width * 4) as usize)
        );
    }
    assert_eq!(ram.word(0x30000), 0x1234);
}

#[test]
fn two_d_blit_converts_and_scales() {
    let ram = Ram::new(0x20000);
    let mut channel = Channel::new();
    call(&mut channel, &ram, 3, 0, 0x902D);
    // Source: 8x8 A8B8G8R8 (0xD5) pitch 32 at 0x1000; pixel = (x*32, y*32, 7, 255).
    for y in 0..8u32 {
        for x in 0..8u32 {
            ram.put(
                0x1000 + (y * 32 + x * 4) as u64,
                &[(x * 32) as u8, (y * 32) as u8, 7, 255],
            );
        }
    }
    let surface = |channel: &mut Channel, base: u32, format: u32, address: u32, size: u32| {
        for (offset, value) in [
            (0x00, format),
            (0x04, 1),
            (0x0c, 1),
            (0x14, size * 4),
            (0x18, size),
            (0x1c, size),
            (0x20, 0),
            (0x24, address),
        ] {
            call(channel, &ram, 3, base + offset, value);
        }
    };
    surface(&mut channel, 0x230, 0xD5, 0x1000, 8);
    // Destination: A8R8G8B8 (0xCF, BGRA byte order) 4x4 at 0x8000.
    surface(&mut channel, 0x200, 0xCF, 0x8000, 4);
    call(&mut channel, &ram, 3, 0x2ac, 3);
    // 2:1 point-sampled downscale from (0, 0), origin center.
    for (method, value) in [
        (0x88c, 0),
        (0x8b0, 0),
        (0x8b4, 0),
        (0x8b8, 4),
        (0x8bc, 4),
        (0x8c0, 0),
        (0x8c4, 2),
        (0x8c8, 0),
        (0x8cc, 2),
        (0x8d0, 0),
        (0x8d4, 0),
        (0x8d8, 0),
    ] {
        call(&mut channel, &ram, 3, method, value);
    }
    call(&mut channel, &ram, 3, 0x8dc, 0);
    // Destination pixel i samples source 2i + 1.
    for y in 0..4u32 {
        for x in 0..4u32 {
            let sx = 2 * x + 1;
            let sy = 2 * y + 1;
            assert_eq!(
                ram.bytes(0x8000 + (y * 16 + x * 4) as u64, 4),
                vec![7, (sy * 32) as u8, (sx * 32) as u8, 255]
            );
        }
    }
}

/// Programs render target 0 as a pitch-linear RGBA8 surface.
pub fn bind_color_target(
    channel: &mut Channel,
    memory: &dyn GpuMemory,
    address: u64,
    width: u32,
    height: u32,
) {
    for (offset, value) in [
        (0x800, (address >> 32) as u32),
        (0x804, address as u32),
        (0x808, width * 4),
        (0x80c, height),
        (0x810, 0xD5),
        (0x814, 1 << 12),
        (0x818, 1),
        (0x820, 0),
        (0x121c, 1),
        (0x0ff4, width << 16),
        (0x0ff8, height << 16),
    ] {
        call(channel, memory, 0, offset, value);
    }
}

/// The 3D state the netlist gives a new channel (nvservices netlist region
/// 7, the non-zero 0xB197 values that matter for drawing and clearing).
pub fn initial_state(channel: &mut Channel) {
    for (offset, value) in [
        (0x0d6c, 0xFFFF_0000u32),
        (0x0d70, 0xFFFF_0000),
        (0x10f8, 0x1010),
        (0x0e04, 0xFFFF_0000),
        (0x0e08, 0xFFFF_0000),
        (0x1a00, 0x1111),
        (0x0fac, 1),
        (0x037c, 1),
        (0x193c, 0x2c1c),
        (0x192c, 1),
        (0x0a18, 0x6420),
        (0x0dac, 0x1b02),
        (0x0db0, 0x1b02),
        (0x191c, 0x900),
        (0x1920, 0x405),
        (0x0c0c, 0x3f80_0000),
        (0x157c, 0xfffff),
        (0x1564, 0xfff),
    ] {
        channel
            .set_initial_state(0xB197, offset / 4, value)
            .unwrap();
    }
}

#[test]
fn clears_respect_rectangles_and_component_masks() {
    let ram = Ram::new(0x20000);
    let mut channel = Channel::new();
    call(&mut channel, &ram, 0, 0, 0xB197);
    initial_state(&mut channel);
    bind_color_target(&mut channel, &ram, 0x1000, 16, 8);
    // The viewport clip 0 limits clears (USE_VIEWPORT_CLIP0 in the initial
    // SET_CLEAR_SURFACE_CONTROL 0x1010): x 2.., width 10; y 1.., height 5.
    call(&mut channel, &ram, 0, 0x0c00, 2 | 10 << 16);
    call(&mut channel, &ram, 0, 0x0c04, 1 | 5 << 16);
    for (i, value) in [1.0f32, 0.5, 0.0, 1.0].iter().enumerate() {
        call(
            &mut channel,
            &ram,
            0,
            0x0d80 + 4 * i as u32,
            value.to_bits(),
        );
    }
    // R, G, B, A of target slot 0.
    call(&mut channel, &ram, 0, 0x19d0, 0xF << 2);
    // Then only green, with value 1.0, in a clear rect 4..6 x 2..3.
    call(&mut channel, &ram, 0, 0x0d84, 1.0f32.to_bits());
    call(&mut channel, &ram, 0, 0x0d6c, 4 | 6 << 16);
    call(&mut channel, &ram, 0, 0x0d70, 2 | 3 << 16);
    call(&mut channel, &ram, 0, 0x19d0, 1 << 3);
    for y in 0..8u32 {
        for x in 0..16u32 {
            let pixel = ram.bytes(0x1000 + (y * 64 + x * 4) as u64, 4);
            let inside = (2..12).contains(&x) && (1..6).contains(&y);
            let green = (4..6).contains(&x) && y == 2;
            let expected = match (inside, green) {
                (false, _) => vec![0, 0, 0, 0],
                (true, false) => vec![255, 128, 0, 255],
                (true, true) => vec![255, 255, 0, 255],
            };
            assert_eq!(pixel, expected, "{x},{y}");
        }
    }
}

/// Depth and stencil clears, in both renderers (Metal clears its copy of
/// the depth buffer and writes it back).
#[test]
fn depth_and_stencil_clear_together_or_apart() {
    for kind in [super::RendererKind::Software, super::RendererKind::Metal] {
        depth_and_stencil_clears(kind);
    }
}

fn depth_and_stencil_clears(kind: super::RendererKind) {
    let ram = Ram::new(0x20000);
    let mut channel = Channel::with_renderer(kind);
    call(&mut channel, &ram, 0, 0, 0xB197);
    initial_state(&mut channel);
    bind_color_target(&mut channel, &ram, 0x1000, 16, 8);
    // A 16x8 Z24S8 depth target at 0x4000 (one block of block-linear
    // memory), everything within the viewport clip and the clear rect.
    for (method, value) in [
        (0x0c00, 16 << 16),
        (0x0c04, 8 << 16),
        (0x0d6c, 16 << 16),
        (0x0d70, 8 << 16),
        (0x0fe0, 0),
        (0x0fe4, 0x4000),
        (0x0fe8, 0x14),
        (0x0fec, 0),
        (0x1228, 16),
        (0x122c, 8),
        (0x1230, 1),
        (0x1538, 1),
        (0x0d90, 1.0f32.to_bits()),
        (0x0da0, 0x7F),
    ] {
        call(&mut channel, &ram, 0, method, value);
    }
    let words = |channel: &mut Channel, ram: &Ram| -> Vec<u32> {
        channel.flush(ram).unwrap();
        ram.bytes(0x4000, 16 * 8 * 4)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect()
    };
    // Depth and stencil, as deko3d clears them.
    call(&mut channel, &ram, 0, 0x19d0, 3);
    assert!(
        words(&mut channel, &ram)
            .iter()
            .all(|&word| word == 0xFFFF_FF7F),
        "{kind:?}"
    );
    // Stencil alone leaves the depth.
    call(&mut channel, &ram, 0, 0x0da0, 1);
    call(&mut channel, &ram, 0, 0x19d0, 2);
    assert!(
        words(&mut channel, &ram)
            .iter()
            .all(|&word| word == 0xFFFF_FF01),
        "{kind:?}"
    );
    // Depth alone leaves the stencil.
    call(&mut channel, &ram, 0, 0x0d90, 0.0f32.to_bits());
    call(&mut channel, &ram, 0, 0x19d0, 1);
    assert!(
        words(&mut channel, &ram)
            .iter()
            .all(|&word| word == 0x0000_0001),
        "{kind:?}"
    );
    // Under the stencil mask (control bit 0, with the viewport clip and
    // clear rect bits as before) only the mask's bits change.
    call(&mut channel, &ram, 0, 0x10f8, 0x1011);
    call(&mut channel, &ram, 0, 0x139c, 0xF0);
    call(&mut channel, &ram, 0, 0x0da0, 0xAA);
    call(&mut channel, &ram, 0, 0x19d0, 2);
    assert!(
        words(&mut channel, &ram)
            .iter()
            .all(|&word| word == 0x0000_00A1),
        "{kind:?}"
    );
    if let Some(metal) = &channel.hooks.metal {
        assert_eq!(metal.fallbacks().1, vec![], "Metal cleared");
    }
}

/// The starter applet's ELF, which embeds the nn::ui2d built-in shaders
/// (BNSH "ui2d_BuildinShader_Nvn_Binary" at file offset 0x4a0000).
const STARTER: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../fw_elf/0100000000001012/ca6e167d711b924ba111ecb24f6a4ae0/section0_exefs/main.elf"
);

/// A shader stage of the ui2d built-in BNSH: the glslc data section at ELF
/// file offset `data` (0x30-byte header, SPH at +0x30, code at +0x80);
/// NVN points the hardware at the SPH (local nvn-methods §3).
fn starter_program(elf: &[u8], data: usize) -> &[u8] {
    assert_eq!(&elf[data..data + 4], &0x1234_5678u32.to_le_bytes());
    &elf[data + 0x30..data + 0x30 + 0x800]
}

/// A channel set up the way NVN leaves it after its init blast (starter
/// 0x25ce30) as far as drawing depends on it, with an RGBA8 pitch render
/// target at 0x80_0000 (64x64 unless sized) cleared to opaque black and a
/// full-target viewport.
struct Frame {
    ram: Ram,
    channel: Channel,
    width: u32,
}

const TARGET: u64 = 0x80_0000;
const PROGRAMS: u64 = 0x20_0000;
const SIZE: u32 = 64;

impl Frame {
    fn new() -> Frame {
        Frame::sized(SIZE, SIZE)
    }

    /// A `width` x `height` target with a full-target viewport.
    fn sized(width: u32, height: u32) -> Frame {
        Frame::sized_with(width, height, super::RendererKind::Software)
    }

    /// A `width` x `height` target drawn by renderer `kind`.
    fn sized_with(width: u32, height: u32, kind: super::RendererKind) -> Frame {
        let ram = Ram::new(0x80_0000 + (width * height * 4) as usize);
        let mut channel = Channel::with_renderer(kind);
        call(&mut channel, &ram, 0, 0, 0xB197);
        initial_state(&mut channel);
        // Netlist initial pipeline bindings: vertex B group 0, pixel group 4.
        for (offset, value) in [(0x2040u32, 0x11u32), (0x2150, 4), (0x2140, 0x51)] {
            channel
                .set_initial_state(0xB197, offset / 4, value)
                .unwrap();
        }
        let mut frame = Frame {
            ram,
            channel,
            width,
        };
        for (offset, value) in [
            (0x13ac, 0x10),   // SET_WINDOW_ORIGIN upper left, FLIP_Y
            (0x193c, 0x181d), // SET_VIEWPORT_CLIP_CONTROL as NVN
            (0x2608, 2),      // SET_BINDLESS_TEXTURE slot 2
            (0x19c0, 1),      // SET_BLEND_FLOAT_OPTION
            (0x10f8, 0x101),  // SET_CLEAR_SURFACE_CONTROL as NVN
            (0x12e4, 1),      // SET_BLEND_STATE_PER_TARGET
            (0x1608, 0),
            (0x160c, PROGRAMS as u32),
        ] {
            frame.method(offset, value);
        }
        bind_color_target(&mut frame.channel, &frame.ram, TARGET, width, height);
        // Viewport 0: NDC -> the target, y down; depth 0..1.
        let (half_x, half_y) = ((width / 2) as f32, (height / 2) as f32);
        for (offset, value) in [
            (0x0a00, half_x),
            (0x0a04, half_y),
            (0x0a08, 0.5),
            (0x0a0c, half_x),
            (0x0a10, half_y),
            (0x0a14, 0.5),
        ] {
            frame.method(offset, value.to_bits());
        }
        for i in 0..4 {
            frame.method(0x0d80 + 4 * i, if i == 3 { 1.0f32 } else { 0.0 }.to_bits());
        }
        frame.method(0x19d0, 0xF << 2);
        frame
    }

    fn method(&mut self, offset: u32, value: u32) {
        call(&mut self.channel, &self.ram, 0, offset, value);
    }

    /// Places a program at `PROGRAMS + offset` and enables it in pipeline
    /// `pipeline` (1 vertex B, 5 pixel).
    fn program(&mut self, pipeline: u32, offset: u32, code: &[u8]) {
        self.ram.put(PROGRAMS + u64::from(offset), code);
        self.method(
            0x2000 + 64 * pipeline,
            if pipeline == 1 { 0x11 } else { 0x51 },
        );
        self.method(0x2004 + 64 * pipeline, offset);
    }

    /// Binds `data` as constant buffer `slot` of bind group `group`.
    fn constants(&mut self, group: u32, slot: u32, address: u64, data: &[u8]) {
        self.ram.put(address, data);
        self.method(0x2380, data.len().next_multiple_of(256) as u32);
        self.method(0x2384, (address >> 32) as u32);
        self.method(0x2388, address as u32);
        self.method(0x2410 + 32 * group, 1 | slot << 4);
    }

    /// Vertex stream 0 at `address` with `stride`, attribute 0 = 3 floats.
    fn vertices(&mut self, address: u64, stride: u32, data: &[f32], attributes: &[(u32, u32)]) {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.ram.put(address, &bytes);
        self.method(0x1c00, 1 << 12 | stride);
        self.method(0x1c04, (address >> 32) as u32);
        self.method(0x1c08, address as u32);
        self.method(0x1f00, 0);
        self.method(0x1f04, (address + bytes.len() as u64 - 1) as u32);
        // SET_VERTEX_ATTRIBUTE_A(i): stream 0, offset, widths, FLOAT.
        for (i, &(offset, widths)) in attributes.iter().enumerate() {
            self.method(0x1160 + 4 * i as u32, offset << 7 | widths << 21 | 7 << 27);
        }
    }

    fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
        self.ram
            .bytes(TARGET + u64::from((y * self.width + x) * 4), 4)
            .try_into()
            .unwrap()
    }
}

fn floats(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// ui2d constants for the vertex shaders at 0x4c0e00 / 0x4be500 (c3, from
/// their disassembly): rows of the projection at 0x00-0x3c, rows of the
/// 3x4 model-view at 0x40-0x6c, vertex position scale at 0x210/0x214 and
/// offset at 0x218/0x21c.
fn ui2d_vertex_constants(extra: &[(usize, f32)]) -> Vec<u8> {
    let mut c = vec![0f32; 0x240 / 4];
    for i in 0..4 {
        c[i * 4 + i] = 1.0; // projection = identity
    }
    for i in 0..3 {
        c[0x40 / 4 + i * 4 + i] = 1.0; // model-view = identity
    }
    c[0x210 / 4] = 1.0;
    c[0x214 / 4] = 1.0;
    for &(offset, value) in extra {
        c[offset / 4] = value;
    }
    floats(&c)
}

#[test]
fn firmware_ui2d_solid_quad() {
    let elf = std::fs::read(STARTER).expect("starter ELF (fw_elf) must be present");
    let mut frame = Frame::new();
    // ui2d vertex shader 0x4c0e00: position = projection * model-view *
    // (a[0x80].xy * c3[0x210/0x214], a[0x80].z, 1).
    frame.program(1, 0, starter_program(&elf, 0x4c0e00));
    // ui2d pixel shader 0x4be400: color = c3[0x00..0x0c], alpha saturated.
    frame.program(5, 0x800, starter_program(&elf, 0x4be400));
    frame.constants(0, 3, 0x30_0000, &ui2d_vertex_constants(&[]));
    frame.constants(4, 3, 0x30_1000, &floats(&[0.25, 0.5, 1.0, 1.0]));
    // A quad from NDC (-0.5, -0.5) to (0.25, 0.75): pixels 16..40 x 16..56.
    let quad = [
        -0.5, -0.5, 0.0, 0.25, -0.5, 0.0, 0.25, 0.75, 0.0, -0.5, 0.75, 0.0,
    ];
    frame.vertices(0x40_0000, 12, &quad, &[(0, 0x02)]);
    frame.method(0x1618, 7); // BEGIN quads
    frame.method(0x0d74, 0);
    frame.method(0x0d78, 4);
    frame.method(0x1614, 0);
    for y in 0..SIZE {
        for x in 0..SIZE {
            let inside = (16..40).contains(&x) && (16..56).contains(&y);
            let expected = if inside {
                [64, 128, 255, 255]
            } else {
                [0, 0, 0, 255]
            };
            assert_eq!(frame.pixel(x, y), expected, "{x},{y}");
        }
    }
    // The debug view of render target 0 shows the same pixels.
    let image = frame.channel.render_target(0, &frame.ram).unwrap();
    assert_eq!((image.width, image.height), (SIZE, SIZE));
    let at = ((20 * SIZE + 20) * 4) as usize;
    assert_eq!(image.rgba[at..at + 4], frame.pixel(20, 20));
}

/// With the depth test on but no depth target (`SET_ZT_SELECT` 0) the test
/// is bypassed: a draw whose depth function is NEVER still writes color.
#[test]
fn depth_test_without_depth_target_passes() {
    let elf = std::fs::read(STARTER).expect("starter ELF (fw_elf) must be present");
    let mut frame = Frame::new();
    frame.program(1, 0, starter_program(&elf, 0x4c0e00));
    frame.program(5, 0x800, starter_program(&elf, 0x4be400));
    frame.constants(0, 3, 0x30_0000, &ui2d_vertex_constants(&[]));
    frame.constants(4, 3, 0x30_1000, &floats(&[0.25, 0.5, 1.0, 1.0]));
    frame.method(0x12cc, 1); // SET_DEPTH_TEST enable
    frame.method(0x130c, 0x200); // SET_DEPTH_FUNC NEVER
    frame.method(0x12e8, 1); // SET_DEPTH_WRITE enable
    frame.method(0x1538, 0); // SET_ZT_SELECT no target
    let quad = [
        -0.5, -0.5, 0.0, 0.25, -0.5, 0.0, 0.25, 0.75, 0.0, -0.5, 0.75, 0.0,
    ];
    frame.vertices(0x40_0000, 12, &quad, &[(0, 0x02)]);
    frame.method(0x1618, 7);
    frame.method(0x0d74, 0);
    frame.method(0x0d78, 4);
    frame.method(0x1614, 0);
    assert_eq!(frame.pixel(20, 20), [64, 128, 255, 255]);
    assert_eq!(frame.pixel(4, 4), [0, 0, 0, 255]);
}

/// The ui2d solid quad (color c3[0..0x10] = (0.25, 0.5, 1, 0.5)) drawn by
/// `kind` with the alpha test enabled with `function` and `reference`;
/// the pixel at the quad's inside after a write-back.
fn alpha_tested_quad(kind: super::RendererKind, function: u32, reference: f32) -> [u8; 4] {
    let elf = std::fs::read(STARTER).expect("starter ELF (fw_elf) must be present");
    let mut frame = Frame::sized_with(SIZE, SIZE, kind);
    frame.program(1, 0, starter_program(&elf, 0x4c0e00));
    frame.program(5, 0x800, starter_program(&elf, 0x4be400));
    frame.constants(0, 3, 0x30_0000, &ui2d_vertex_constants(&[]));
    frame.constants(4, 3, 0x30_1000, &floats(&[0.25, 0.5, 1.0, 0.5]));
    frame.method(0x12ec, 1); // SET_ALPHA_TEST
    frame.method(0x1310, reference.to_bits()); // SET_ALPHA_REF
    frame.method(0x1314, function); // SET_ALPHA_FUNC
    let quad = [
        -0.5, -0.5, 0.0, 0.25, -0.5, 0.0, 0.25, 0.75, 0.0, -0.5, 0.75, 0.0,
    ];
    frame.vertices(0x40_0000, 12, &quad, &[(0, 0x02)]);
    frame.method(0x1618, 7);
    frame.method(0x0d74, 0);
    frame.method(0x0d78, 4);
    frame.method(0x1614, 0);
    frame.channel.flush(&frame.ram).unwrap();
    if let Some(metal) = &frame.channel.hooks.metal {
        assert_eq!(metal.fallbacks(), (1, vec![]));
    }
    frame.pixel(20, 20)
}

/// The alpha test compares output 0 alpha with the reference and drops
/// failing fragments, in both renderers, for OpenGL and Direct3D
/// function encodings.
#[test]
fn alpha_test_discards_failing_fragments() {
    use super::RendererKind::{Metal, Software};
    let drawn = [64, 128, 255, 128];
    let kept = [0, 0, 0, 255];
    for kind in [Software, Metal] {
        // OGL_GREATER: 0.5 > 0.25 passes, 0.5 > 0.75 fails.
        assert_eq!(alpha_tested_quad(kind, 0x204, 0.25), drawn, "{kind:?}");
        assert_eq!(alpha_tested_quad(kind, 0x204, 0.75), kept, "{kind:?}");
        // D3D_LESSEQUAL (4): 0.5 <= 0.5 passes; D3D_NEVER (1) fails.
        assert_eq!(alpha_tested_quad(kind, 4, 0.5), drawn, "{kind:?}");
        assert_eq!(alpha_tested_quad(kind, 1, 0.0), kept, "{kind:?}");
    }
}

/// The ui2d solid quad (pixels 16..40 x 16..56, its corners counter-
/// clockwise with the target's rows counted as y up) drawn by `kind` with
/// back faces culled, `SET_WINDOW_ORIGIN` = `origin` and `OGL_SET_FRONT_FACE`
/// = `front_face`; the pixel at the quad's inside after a write-back.
fn back_culled_quad(kind: super::RendererKind, origin: u32, front_face: u32) -> [u8; 4] {
    let elf = std::fs::read(STARTER).expect("starter ELF (fw_elf) must be present");
    let mut frame = Frame::sized_with(SIZE, SIZE, kind);
    frame.program(1, 0, starter_program(&elf, 0x4c0e00));
    frame.program(5, 0x800, starter_program(&elf, 0x4be400));
    frame.constants(0, 3, 0x30_0000, &ui2d_vertex_constants(&[]));
    frame.constants(4, 3, 0x30_1000, &floats(&[0.25, 0.5, 1.0, 1.0]));
    frame.method(0x13ac, origin); // SET_WINDOW_ORIGIN
    frame.method(0x1918, 1); // OGL_SET_CULL
    frame.method(0x1920, 0x405); // OGL_SET_CULL_FACE BACK
    frame.method(0x191c, front_face); // OGL_SET_FRONT_FACE
    let quad = [
        -0.5, -0.5, 0.0, 0.25, -0.5, 0.0, 0.25, 0.75, 0.0, -0.5, 0.75, 0.0,
    ];
    frame.vertices(0x40_0000, 12, &quad, &[(0, 0x02)]);
    frame.method(0x1618, 7);
    frame.method(0x0d74, 0);
    frame.method(0x0d78, 4);
    frame.method(0x1614, 0);
    frame.channel.flush(&frame.ram).unwrap();
    frame.pixel(20, 20)
}

/// Facing follows `SET_WINDOW_ORIGIN` FLIP_Y: with it the target's rows
/// count as y up (OpenGL window coordinates, NVN's setting), without it as
/// y down. The Mii editor composes its frame with a full-screen strip
/// counter-clockwise with rows up, FLIP_Y, front face CCW and back faces
/// culled, and Settings draws with rows down and FLIP_Y clear; both must
/// keep their triangles.
#[test]
fn window_origin_flip_sets_which_side_faces_front() {
    use super::RendererKind::{Metal, Software};
    let drawn = [64, 128, 255, 255];
    let kept = [0, 0, 0, 255];
    for kind in [Software, Metal] {
        // OGL_CCW (0x901) with FLIP_Y: front, drawn.
        assert_eq!(back_culled_quad(kind, 0x10, 0x901), drawn, "{kind:?}");
        // OGL_CW (0x900) with FLIP_Y: back, culled.
        assert_eq!(back_culled_quad(kind, 0x10, 0x900), kept, "{kind:?}");
        // Without FLIP_Y the rows count down and the order reverses.
        assert_eq!(back_culled_quad(kind, 0, 0x901), kept, "{kind:?}");
        assert_eq!(back_culled_quad(kind, 0, 0x900), drawn, "{kind:?}");
    }
}

/// A program of `words` after a program header whose first word is
/// `kind` (with `header` bytes set), followed by the padding the decoder
/// reads past the end.
fn program_bytes(kind: u32, header: &[(usize, u8)], words: &[u64]) -> Vec<u8> {
    let mut bytes = vec![0u8; 0x50];
    bytes[..4].copy_from_slice(&kind.to_le_bytes());
    for &(at, value) in header {
        bytes[at] = value;
    }
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes.extend_from_slice(&[0; 32]);
    bytes
}

/// A quad whose vertex program, like the Mii editor's face-texture
/// programs, stores only x, y and z of the position (ALD attribute 0 xyz,
/// AST a[0x70..0x7c)): w starts at 1, so the quad is drawn; with the
/// pixel program's color (1, 0.5, 0.25, 1).
fn position_xyz_quad(kind: super::RendererKind) -> [u8; 4] {
    let mut frame = Frame::sized_with(SIZE, SIZE, kind);
    let vs = program_bytes(
        0x0401,
        // Imap: attribute 0 x, y, z; Omap: position x, y, z.
        &[(0x18, 0x07), (0x35, 0x70)],
        &[
            0,
            0xefd8_7f80_0807_ff00, // ALD R0, a[0x80]
            0xefd8_7f80_0847_ff01, // ALD R1, a[0x84]
            0xefd8_7f80_0887_ff02, // ALD R2, a[0x88]
            0,
            0xeff0_7f80_0707_ff00, // AST a[0x70], R0
            0xeff0_7f80_0747_ff01, // AST a[0x74], R1
            0xeff0_7f80_0787_ff02, // AST a[0x78], R2
            0,
            0xe300_0000_0007_000f, // EXIT
            0xe240_0fff_ff07_000f, // BRA to itself
            0x50b0_0000_0007_0f00, // NOP
        ],
    );
    let mov = |rd: u64, v: f32| 0x010000000007f000u64 | (u64::from(v.to_bits()) << 20) | rd;
    let ps = program_bytes(
        0x1402,
        &[(0x48, 15)],
        &[
            0,
            mov(0, 1.0),
            mov(1, 0.5),
            mov(2, 0.25),
            0,
            mov(3, 1.0),
            0xe300_0000_0007_000f, // EXIT
            0xe240_0fff_ff07_000f, // BRA to itself
        ],
    );
    frame.program(1, 0, &vs);
    frame.program(5, 0x800, &ps);
    let quad = [
        -0.5, -0.5, 0.0, 0.25, -0.5, 0.0, 0.25, 0.75, 0.0, -0.5, 0.75, 0.0,
    ];
    frame.vertices(0x40_0000, 12, &quad, &[(0, 0x02)]);
    frame.method(0x1618, 7);
    frame.method(0x0d74, 0);
    frame.method(0x0d78, 4);
    frame.method(0x1614, 0);
    frame.channel.flush(&frame.ram).unwrap();
    frame.pixel(20, 20)
}

/// A target of an sRGB format (`A8BL8GL8RL8`, 0xD6) encodes the program's
/// linear color with `SET_SRGB_WRITE` (0x15b8) left 0, as the Mii editor
/// draws its Mii, in both renderers.
#[test]
fn srgb_target_encodes_without_srgb_write() {
    use super::texture::format::linear_to_srgb;
    let color = [0.25f32, 0.5, 1.0, 1.0];
    let expected: Vec<u8> = (0..4)
        .map(|c| {
            let v = if c < 3 {
                linear_to_srgb(color[c])
            } else {
                color[c]
            };
            (v * 255.0).round() as u8
        })
        .collect();
    for kind in [super::RendererKind::Software, super::RendererKind::Metal] {
        let elf = std::fs::read(STARTER).expect("starter ELF (fw_elf) must be present");
        let mut frame = Frame::sized_with(SIZE, SIZE, kind);
        frame.method(0x810, 0xD6); // SET_COLOR_TARGET_FORMAT(0) A8BL8GL8RL8
        frame.method(0x15b8, 0); // SET_SRGB_WRITE
        frame.program(1, 0, starter_program(&elf, 0x4c0e00));
        frame.program(5, 0x800, starter_program(&elf, 0x4be400));
        frame.constants(0, 3, 0x30_0000, &ui2d_vertex_constants(&[]));
        frame.constants(4, 3, 0x30_1000, &floats(&color));
        let quad = [
            -0.5, -0.5, 0.0, 0.25, -0.5, 0.0, 0.25, 0.75, 0.0, -0.5, 0.75, 0.0,
        ];
        frame.vertices(0x40_0000, 12, &quad, &[(0, 0x02)]);
        frame.method(0x1618, 7);
        frame.method(0x0d74, 0);
        frame.method(0x0d78, 4);
        frame.method(0x1614, 0);
        frame.channel.flush(&frame.ram).unwrap();
        assert_eq!(frame.pixel(20, 20).to_vec(), expected, "{kind:?}");
    }
}

#[test]
fn unstored_position_w_is_one() {
    for kind in [super::RendererKind::Software, super::RendererKind::Metal] {
        assert_eq!(position_xyz_quad(kind), [255, 128, 64, 255], "{kind:?}");
    }
}

/// A small xorshift generator for test data.
struct Random(u64);

impl Random {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }

    /// A float in [0, 1) with 24 random bits.
    fn unit(&mut self) -> f32 {
        (self.next() >> 8) as f32 / (1 << 24) as f32
    }
}

/// Where `depth_tested_frame` puts its depth buffer.
const DEPTH: u64 = 0x60_0000;

/// The 64x64 frame drawn by `kind` with a block-linear depth buffer of
/// `SET_ZT_FORMAT` `raw` holding `fill`, the depth test `function`
/// (`SET_DEPTH_FUNC`) and depth writes `write`: two solid ui2d quads
/// across the target, each with its own color and its four corners at
/// different depths (so each of its triangles has its own depth plane).
/// The color target and the depth buffer after a write-back.
fn depth_tested_frame(
    kind: super::RendererKind,
    raw: u32,
    function: u32,
    write: bool,
    fill: &[u8],
) -> (Vec<u8>, Vec<u8>) {
    let elf = std::fs::read(STARTER).expect("starter ELF (fw_elf) must be present");
    let mut frame = Frame::sized_with(SIZE, SIZE, kind);
    frame.program(1, 0, starter_program(&elf, 0x4c0e00));
    frame.program(5, 0x800, starter_program(&elf, 0x4be400));
    frame.constants(0, 3, 0x30_0000, &ui2d_vertex_constants(&[]));
    frame.ram.put(DEPTH, fill);
    for (offset, value) in [
        (0x0fe0, (DEPTH >> 32) as u32),
        (0x0fe4, DEPTH as u32),
        (0x0fe8, raw),
        (0x0fec, 0),
        (0x1228, SIZE),
        (0x122c, SIZE),
        (0x1230, 1),
        (0x1538, 1),
        (0x12cc, 1),
        (0x130c, function),
        (0x12e8, u32::from(write)),
    ] {
        frame.method(offset, value);
    }
    for (i, (color, depths)) in [
        ([0.25, 0.5, 1.0, 1.0], [-0.9, 0.7, 0.1, -0.2]),
        ([1.0, 0.25, 0.0, 1.0], [0.8, -0.6, -0.1, 0.3]),
    ]
    .into_iter()
    .enumerate()
    {
        frame.constants(4, 3, 0x30_1000 + 0x100 * i as u64, &floats(&color));
        let corners = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
        let quad: Vec<f32> = corners
            .iter()
            .zip(depths)
            .flat_map(|(&[x, y], z)| [x, y, z])
            .collect();
        frame.vertices(0x40_0000 + 0x100 * i as u64, 12, &quad, &[(0, 0x02)]);
        frame.method(0x1618, 7);
        frame.method(0x0d74, 0);
        frame.method(0x0d78, 4);
        frame.method(0x1614, 0);
    }
    frame.channel.flush(&frame.ram).unwrap();
    if let Some(metal) = &frame.channel.hooks.metal {
        assert_eq!(metal.fallbacks(), (2, vec![]), "Metal drew both quads");
    }
    (
        frame.ram.bytes(TARGET, (SIZE * SIZE * 4) as usize),
        frame.ram.bytes(DEPTH, fill.len()),
    )
}

/// The depth test on Metal gives the software renderer's colors and
/// depth buffer, bit for bit, for every depth format, several functions,
/// with and without depth writes, over a buffer of random stored depths
/// (and stencil bits, which stay).
#[test]
fn depth_test_matches_software() {
    use super::RendererKind::{Metal, Software};
    use super::texture::format::DepthFormat;
    let mut random = Random(0x2545_f491_4f6c_dd1d);
    for raw in [0x0a, 0x13, 0x14, 0x15, 0x16, 0x19] {
        let format = DepthFormat::from_raw(raw).unwrap();
        let bytes = format.bytes_per_pixel() as usize;
        let mut fill = vec![0u8; (SIZE * SIZE) as usize * bytes];
        for texel in fill.chunks_mut(bytes) {
            for byte in texel.iter_mut() {
                *byte = random.next() as u8;
            }
            if !format.is_unorm() {
                // Normal floats in [0, 1).
                texel[..4].copy_from_slice(&random.unit().max(1e-30).to_le_bytes());
            }
        }
        // OGL_LESS, OGL_LEQUAL, D3D_GREATER, D3D_GREATEREQUAL.
        for function in [0x201, 0x203, 5, 7] {
            for write in [true, false] {
                let software = depth_tested_frame(Software, raw, function, write, &fill);
                let metal = depth_tested_frame(Metal, raw, function, write, &fill);
                let what = format!("{format:?} function {function:#x} write {write}");
                // Both quads and the clear color show: the test both
                // passed and failed.
                let mut colors: Vec<&[u8]> = software.0.chunks(4).collect();
                colors.sort();
                colors.dedup();
                assert!(colors.len() >= 2, "{what}: {colors:?}");
                assert!(software.0 == metal.0, "{what}: colors differ");
                assert!(software.1 == metal.1, "{what}: depth buffers differ");
                assert_eq!(software.1 != fill, write, "{what}: depth written");
            }
        }
    }
}

/// The 64x64 frame drawn by `kind` with a block-linear depth buffer of
/// `SET_ZT_FORMAT` `raw` holding `fill`, the depth test `depth_function`
/// (with depth writes) or none, the stencil test enabled with the
/// registers `stencil`, and color writes `color`: two full-target ui2d
/// quads as in `depth_tested_frame`, the second one wound the other way
/// (so it is back-facing). The color target and the depth buffer after a
/// write-back, and the Metal renderer's counts.
fn stencil_tested_frame(
    kind: super::RendererKind,
    raw: u32,
    depth_function: Option<u32>,
    stencil: &[(u32, u32)],
    color: bool,
    fill: &[u8],
) -> (Vec<u8>, Vec<u8>, Option<(u64, Vec<(String, u64)>)>) {
    let elf = std::fs::read(STARTER).expect("starter ELF (fw_elf) must be present");
    let mut frame = Frame::sized_with(SIZE, SIZE, kind);
    frame.program(1, 0, starter_program(&elf, 0x4c0e00));
    frame.program(5, 0x800, starter_program(&elf, 0x4be400));
    frame.constants(0, 3, 0x30_0000, &ui2d_vertex_constants(&[]));
    frame.ram.put(DEPTH, fill);
    for (offset, value) in [
        (0x0fe0, (DEPTH >> 32) as u32),
        (0x0fe4, DEPTH as u32),
        (0x0fe8, raw),
        (0x0fec, 0),
        (0x1228, SIZE),
        (0x122c, SIZE),
        (0x1230, 1),
        (0x1538, 1),
        (0x12cc, u32::from(depth_function.is_some())),
        (0x130c, depth_function.unwrap_or(0x207)),
        (0x12e8, 1),
        (0x1380, 1), // SET_STENCIL_TEST
        (0x1a00, if color { 0x1111 } else { 0 }),
    ] {
        frame.method(offset, value);
    }
    for &(offset, value) in stencil {
        frame.method(offset, value);
    }
    for (i, (color, depths)) in [
        ([0.25, 0.5, 1.0, 1.0], [-0.9, 0.7, 0.1, -0.2]),
        ([1.0, 0.25, 0.0, 1.0], [0.8, -0.6, -0.1, 0.3]),
    ]
    .into_iter()
    .enumerate()
    {
        frame.constants(4, 3, 0x30_1000 + 0x100 * i as u64, &floats(&color));
        let mut corners = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
        if i == 1 {
            corners.reverse();
        }
        let quad: Vec<f32> = corners
            .iter()
            .zip(depths)
            .flat_map(|(&[x, y], z)| [x, y, z])
            .collect();
        frame.vertices(0x40_0000 + 0x100 * i as u64, 12, &quad, &[(0, 0x02)]);
        frame.method(0x1618, 7);
        frame.method(0x0d74, 0);
        frame.method(0x0d78, 4);
        frame.method(0x1614, 0);
    }
    frame.channel.flush(&frame.ram).unwrap();
    let counts = frame.channel.hooks.metal.as_ref().map(|m| m.fallbacks());
    (
        frame.ram.bytes(TARGET, (SIZE * SIZE * 4) as usize),
        frame.ram.bytes(DEPTH, fill.len()),
        counts,
    )
}

/// Stencil operations: saturating and wrapping steps, the write mask and
/// the masked comparison with the reference on the left.
#[test]
fn stencil_operations_and_masks() {
    use super::raster::{Compare, StencilFace, StencilOp};
    assert_eq!(StencilOp::IncrementClamp.apply(255, 0), 255);
    assert_eq!(StencilOp::IncrementWrap.apply(255, 0), 0);
    assert_eq!(StencilOp::DecrementClamp.apply(0, 0), 0);
    assert_eq!(StencilOp::DecrementWrap.apply(0, 0), 255);
    assert_eq!(StencilOp::Invert.apply(0x5a, 0), 0xa5);
    assert_eq!(StencilOp::Replace.apply(0x12, 0x34), 0x34);
    // OpenGL and Direct3D encodings.
    assert_eq!(StencilOp::from_raw(0x1e02), Some(StencilOp::IncrementClamp));
    assert_eq!(StencilOp::from_raw(0x8507), Some(StencilOp::IncrementWrap));
    assert_eq!(StencilOp::from_raw(0), Some(StencilOp::Zero));
    assert_eq!(StencilOp::from_raw(1), Some(StencilOp::Keep));
    assert_eq!(StencilOp::from_raw(8), Some(StencilOp::DecrementWrap));
    assert_eq!(StencilOp::from_raw(9), None);
    let face = StencilFace {
        fail: StencilOp::Keep,
        depth_fail: StencilOp::Keep,
        pass: StencilOp::Replace,
        compare: Compare::Less,
        reference: 0x13,
        compare_mask: 0x0f,
        write_mask: 0xf0,
    };
    // 0x3 < 0x4 (stored 0x24 masked); 0x3 < 0x3 fails.
    assert!(face.test(0x24));
    assert!(!face.test(0xf3));
    // Only the upper nibble of the reference lands.
    assert_eq!(face.update(StencilOp::Replace, 0xab), 0x1b);
}

/// The stencil test on Metal gives the software renderer's colors and
/// depth/stencil buffer, bit for bit, for every format with stencil, one
/// and two-sided, with and without the depth test, over random stored
/// stencil bits; failing fragments that update the stencil buffer while
/// colors are written are left to the software renderer.
#[test]
fn stencil_test_matches_software() {
    use super::RendererKind::{Metal, Software};
    use super::texture::format::DepthFormat;
    let mut random = Random(0x1405_6f3c_9d2b_77e1);
    // Front: fail, zfail, zpass, func, ref, func mask, mask (0x1384..).
    let front = |ops: [u32; 3], func: u32, reference: u32, compare_mask: u32, mask: u32| {
        let mut registers: Vec<(u32, u32)> = ops
            .iter()
            .enumerate()
            .map(|(i, &op)| (0x1384 + 4 * i as u32, op))
            .collect();
        registers.extend([
            (0x1390, func),
            (0x1394, reference),
            (0x1398, compare_mask),
            (0x139c, mask),
        ]);
        registers
    };
    // Back: two-sided, ops and func (0x1598..), ref, mask, func mask
    // (0x0f54..).
    let back = |ops: [u32; 3], func: u32, reference: u32, compare_mask: u32, mask: u32| {
        let mut registers = vec![(0x1594, 1)];
        registers.extend(
            ops.iter()
                .enumerate()
                .map(|(i, &op)| (0x1598 + 4 * i as u32, op)),
        );
        registers.extend([
            (0x15a4, func),
            (0x0f54, reference),
            (0x0f58, mask),
            (0x0f5c, compare_mask),
        ]);
        registers
    };
    // OGL KEEP, INCR (wrap), INVERT, REPLACE, D3D DECRSAT, INCRSAT, ZERO.
    let (keep, incr_wrap, invert, replace, decr_sat, incr_sat, zero) =
        (0x1e00, 0x8507, 0x150a, 0x1e01, 5, 4, 0);
    struct Case {
        what: &'static str,
        depth: Option<u32>,
        registers: Vec<(u32, u32)>,
        color: bool,
        /// Whether Metal draws it.
        metal: bool,
    }
    let cases = [
        Case {
            what: "one-sided, keep on failure",
            depth: Some(0x201),
            registers: front([keep, keep, incr_wrap], 0x203, 0x80, 0xff, 0xff),
            color: true,
            metal: true,
        },
        Case {
            what: "two-sided, masked",
            depth: Some(7),
            registers: [
                front([keep, keep, replace], 0x202, 0x35, 0x0f, 0x3c),
                back([keep, keep, invert], 6, 0x40, 0xc0, 0xf0),
            ]
            .concat(),
            color: true,
            metal: true,
        },
        Case {
            what: "no depth test, failure ops",
            depth: None,
            registers: [
                front([decr_sat, zero, incr_sat], 0x204, 0x60, 0xff, 0xff),
                back([replace, keep, incr_wrap], 2, 0x90, 0x7f, 0xff),
            ]
            .concat(),
            color: true,
            metal: false,
        },
        Case {
            what: "stencil only, every op",
            depth: Some(0x203),
            registers: [
                front([decr_sat, zero, incr_sat], 0x206, 0x70, 0xfe, 0xff),
                back([replace, invert, incr_wrap], 5, 0x90, 0xff, 0x7f),
            ]
            .concat(),
            color: false,
            metal: true,
        },
    ];
    for raw in [0x14, 0x16, 0x19] {
        let format = DepthFormat::from_raw(raw).unwrap();
        let bytes = format.bytes_per_pixel() as usize;
        let mut fill = vec![0u8; (SIZE * SIZE) as usize * bytes];
        for texel in fill.chunks_mut(bytes) {
            for byte in texel.iter_mut() {
                *byte = random.next() as u8;
            }
            if !format.is_unorm() {
                texel[..4].copy_from_slice(&random.unit().max(1e-30).to_le_bytes());
            }
        }
        for case in &cases {
            let what = format!("{format:?} {}", case.what);
            let run = |kind| {
                stencil_tested_frame(kind, raw, case.depth, &case.registers, case.color, &fill)
            };
            let software = run(Software);
            let metal = run(Metal);
            let stencil = |buffer: &[u8]| -> Vec<u8> {
                buffer
                    .chunks(bytes)
                    .map(|texel| format.read_stencil(texel).unwrap())
                    .collect()
            };
            // The stencil test both passed and failed, and stencil bits
            // changed.
            if case.color {
                let mut colors: Vec<&[u8]> = software.0.chunks(4).collect();
                colors.sort();
                colors.dedup();
                assert!(colors.len() >= 2, "{what}: {colors:?}");
            }
            assert_ne!(stencil(&software.1), stencil(&fill), "{what}: stencil kept");
            assert!(software.0 == metal.0, "{what}: colors differ");
            assert!(
                software.1 == metal.1,
                "{what}: depth/stencil buffers differ"
            );
            let (draws, fallbacks) = metal.2.expect("Metal renderer");
            if case.metal {
                assert_eq!((draws, fallbacks), (2, vec![]), "{what}: Metal drew both");
            } else {
                assert_eq!(draws, 0, "{what}: left to the software renderer");
            }
        }
    }
}

/// Triangles with vertices on pixel centers and pixel edges of a target
/// whose size is no power of two (so window positions are not exact in
/// Metal's normalized coordinates) cover the same pixels on Metal as in
/// the software rasterizer (8-bit sub-pixel snapping, top-left rule).
#[test]
fn exact_edges_match_software() {
    use super::RendererKind::{Metal, Software};
    let (width, height) = (60u32, 36u32);
    let image = |kind| {
        let elf = std::fs::read(STARTER).expect("starter ELF (fw_elf) must be present");
        let mut frame = Frame::sized_with(width, height, kind);
        frame.program(1, 0, starter_program(&elf, 0x4c0e00));
        frame.program(5, 0x800, starter_program(&elf, 0x4be400));
        frame.constants(0, 3, 0x30_0000, &ui2d_vertex_constants(&[]));
        let mut random = Random(0x9e37_79b9_7f4a_7c15);
        let draws = 48u64;
        for i in 0..draws {
            let color = [random.unit(), random.unit(), random.unit(), 1.0];
            frame.constants(4, 3, 0x30_1000 + 0x100 * i, &floats(&color));
            // Corners on the half-pixel grid, in normalized coordinates.
            let triangle: Vec<f32> = (0..3)
                .flat_map(|_| {
                    let x = (random.next() % (2 * width + 1)) as f32 / 2.0;
                    let y = (random.next() % (2 * height + 1)) as f32 / 2.0;
                    [
                        x / width as f32 * 2.0 - 1.0,
                        y / height as f32 * 2.0 - 1.0,
                        0.0,
                    ]
                })
                .collect();
            frame.vertices(0x40_0000 + 0x100 * i, 12, &triangle, &[(0, 0x02)]);
            frame.method(0x1618, 4); // BEGIN triangles
            frame.method(0x0d74, 0);
            frame.method(0x0d78, 3);
            frame.method(0x1614, 0);
        }
        frame.channel.flush(&frame.ram).unwrap();
        if let Some(metal) = &frame.channel.hooks.metal {
            assert_eq!(metal.fallbacks().1, vec![], "Metal drew every triangle");
        }
        frame.ram.bytes(TARGET, (width * height * 4) as usize)
    };
    let software = image(Software);
    let metal = image(Metal);
    let differing: Vec<(u32, u32)> = (0..width * height)
        .filter(|&i| software[4 * i as usize..][..4] != metal[4 * i as usize..][..4])
        .map(|i| (i % width, i / width))
        .collect();
    assert!(differing.is_empty(), "pixels differ: {differing:?}");
}

/// A draw whose pixel program names a texture slot the draw leaves unset
/// (handle TIC index 0xfffff, as netConnect's uber-shader leaves on
/// branches its draws do not take): P0 = c3[0] != 0; @P0 BRA over a TEXS
/// of bound texture 8 (handle c2[0x20]); color = (1, 0.5, 0.25, 1). The
/// branch depends on a constant, so the program names the texture.
/// Returns the frame with the draw recorded up to `END`, and `END`'s
/// result.
fn unset_texture_draw(kind: super::RendererKind, skip_sample: bool) -> (Frame, Result<(), String>) {
    let elf = std::fs::read(STARTER).expect("starter ELF (fw_elf) must be present");
    let mut frame = Frame::sized_with(SIZE, SIZE, kind);
    frame.program(1, 0, starter_program(&elf, 0x4c0e00));
    // PS header: shader type 5, target 0 RGBA.
    let mut ps = vec![0u8; 0x50];
    ps[..4].copy_from_slice(&0x1402u32.to_le_bytes());
    ps[0x48] = 15;
    let mov = |rd: u64, v: f32| 0x010000000007f000u64 | (u64::from(v.to_bits()) << 20) | rd;
    for word in [
        0,
        0x4c98_078c_0007_0004, // MOV R4, c3[0x0]
        0x5b6b_0380_0ff7_0407, // ISETP.NE.AND P0, PT, R4, RZ, PT
        0xe240_0000_0100_000f, // @P0 BRA 0x30
        0,
        0xd832_0080_4057_040a, // TEXS (from netConnect's shader), texture 8
        mov(0, 1.0),
        mov(1, 0.5),
        0,
        mov(2, 0.25),
        mov(3, 1.0),
        0xe300_0000_0007_000f, // EXIT
        0,
        0xe240_0fff_ff07_000f, // BRA to itself
        0x50b0_0000_0007_0f00, // NOP
        0x50b0_0000_0007_0f00, // NOP
    ] {
        ps.extend_from_slice(&u64::to_le_bytes(word));
    }
    ps.extend_from_slice(&[0; 32]);
    frame.program(5, 0x800, &ps);
    frame.constants(0, 3, 0x30_0000, &ui2d_vertex_constants(&[]));
    frame.constants(4, 2, 0x30_1000, &[0xff; 0x40]);
    frame.constants(4, 3, 0x30_2000, &u32::from(skip_sample).to_le_bytes());
    // SET_TEX_HEADER_POOL_A..C: a pool of 0x11 headers at 0x50_0000.
    frame.method(0x1574, 0);
    frame.method(0x1578, 0x50_0000);
    frame.method(0x157c, 0x10);
    let quad = [
        -0.5, -0.5, 0.0, 0.25, -0.5, 0.0, 0.25, 0.75, 0.0, -0.5, 0.75, 0.0,
    ];
    frame.vertices(0x40_0000, 12, &quad, &[(0, 0x02)]);
    frame.method(0x1618, 7);
    frame.method(0x0d74, 0);
    frame.method(0x0d78, 4);
    let end = frame
        .channel
        .call(0, 0x1614 / 4, 0, &frame.ram)
        .map_err(|e| e.0);
    (frame, end)
}

/// An unset texture slot whose sample does not execute draws (on Metal,
/// without falling back); one whose sample executes is refused, by the
/// software renderer at the draw and by Metal at the next write-back,
/// before any of its pixels reach memory.
#[test]
fn unset_texture_slot_refused_only_where_sampled() {
    use super::RendererKind::{Metal, Software};
    for kind in [Software, Metal] {
        let (mut frame, end) = unset_texture_draw(kind, true);
        end.unwrap();
        frame.channel.flush(&frame.ram).unwrap();
        assert_eq!(frame.pixel(20, 20), [255, 128, 64, 255], "{kind:?}");
    }
    let (frame, _) = unset_texture_draw(Metal, true);
    let metal = frame.channel.hooks.metal.as_ref().expect("Metal device");
    let (draws, fallbacks) = metal.fallbacks();
    assert_eq!((draws, fallbacks), (1, vec![]));

    let (_, end) = unset_texture_draw(Software, false);
    let refusal = end.unwrap_err();
    assert!(refusal.contains("TIC index 0xfffff"), "{refusal}");
    let (mut frame, end) = unset_texture_draw(Metal, false);
    end.unwrap();
    let refusal = frame.channel.flush(&frame.ram).unwrap_err().0;
    assert!(refusal.contains("TIC index 0xfffff"), "{refusal}");
    // Nothing recorded since the last write-back reaches memory.
    assert_ne!(frame.pixel(20, 20), [255, 128, 64, 255]);
}

/// Binds the ui2d textured shader pair of the starter ELF and an `n` x `n`
/// A8B8G8R8 block-linear texture whose texel (x, y) is `texel(x, y)`, and
/// draws the unit corner square, which the vertex shader maps to NDC
/// `(u * scale[0] + offset[0], -v * scale[1] + offset[1])`.
///
/// Shader behaviour, from their disassembly: vertex shader 0x4be500 with
/// flags c3\[0x22c\] = 0 and c3\[0x220\] = 0 computes position =
/// projection * model-view * (x * c3\[0x210\] + c3\[0x218\], -y *
/// c3\[0x214\] + c3\[0x21c\], 0, 1) for a\[0x80\] = (x, y) in {0, 1}, and
/// takes the texture coordinate (u, v) = c3\[0xe0 + 16 * (x + 2y)\],
/// transformed by the rows c3\[0x70\] / c3\[0x80\] into a\[0x90\] = (s, 1 -
/// t). Pixel shader 0x4beb00: color = texture(handle c2\[0x20\], a\[0x90\])
/// * c3\[0x00..0x0c\] + c3\[0x10..0x1c\], alpha saturated.
fn ui2d_textured(
    frame: &mut Frame,
    elf: &[u8],
    n: u32,
    block_height_log2: u32,
    scale: [f32; 2],
    offset: [f32; 2],
    texel: impl Fn(u32, u32) -> [u8; 4],
) {
    frame.program(1, 0, starter_program(elf, 0x4be500));
    frame.program(5, 0x800, starter_program(elf, 0x4beb00));
    let mut extra = vec![
        (0x210, scale[0]),
        (0x214, scale[1]),
        (0x218, offset[0]),
        (0x21c, offset[1]),
        (0x70, 1.0),
        (0x84, 1.0),
    ];
    for corner in 0..4usize {
        extra.push((0xe0 + 16 * corner, (corner & 1) as f32));
        extra.push((0xe4 + 16 * corner, (corner >> 1) as f32));
    }
    frame.constants(0, 3, 0x30_0000, &ui2d_vertex_constants(&extra));
    frame.constants(
        4,
        3,
        0x30_1000,
        &floats(&[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    );
    // The shaders also read their compiled constants (slot 1); bound and
    // zero here, as no flag they test is set.
    frame.constants(0, 1, 0x30_2000, &[0; 256]);
    frame.constants(4, 1, 0x30_3000, &[0; 256]);
    // Texture handles live in slot 2 at 0x20 + 8n (local nvn-methods §3):
    // TIC 3, TSC 5.
    let mut handles = vec![0u8; 0x40];
    handles[0x20..0x24].copy_from_slice(&(3u32 | 5 << 20).to_le_bytes());
    frame.constants(4, 2, 0x30_4000, &handles);
    let texture = 0x50_0000u64;
    for y in 0..n {
        for x in 0..n {
            frame.ram.put(
                texture + reference_offset(x * 4, y, n * 4, block_height_log2),
                &texel(x, y),
            );
        }
    }
    // TIC: format A8B8G8R8 (8), UNORM components, swizzle RGBA, version 3
    // (block linear), 2D, normalized coordinates [public: NVIDIA
    // clb197tex.h TEXHEAD_BL].
    let tic: [u32; 8] = [
        8 | 2 << 7 | 2 << 10 | 2 << 13 | 2 << 16 | 2 << 19 | 3 << 22 | 4 << 25 | 5 << 28,
        texture as u32,
        (texture >> 32) as u32 | 3 << 21,
        block_height_log2 << 3,
        (n - 1) | 1 << 23 | 7 << 29,
        (n - 1) | 1 << 31,
        0,
        0,
    ];
    // TSC: clamp to edge, nearest magnification/minification, no mips.
    let tsc: [u32; 8] = [
        2 | 2 << 3 | 2 << 6,
        1 | 1 << 4 | 1 << 6,
        15 << 20,
        0,
        0,
        0,
        0,
        0,
    ];
    let pools = 0x60_0000u64;
    frame.ram.put(
        pools + 3 * 32,
        &tic.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<_>>(),
    );
    frame.ram.put(
        pools + 0x1000 + 5 * 32,
        &tsc.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<_>>(),
    );
    for (offset, value) in [
        (0x1574, 0),
        (0x1578, pools as u32),
        (0x157c, 15),
        (0x155c, 0),
        (0x1560, pools as u32 + 0x1000),
        (0x1564, 15),
    ] {
        frame.method(offset, value);
    }
    // Corners as ui2d draws them: a triangle strip over (0,0) (1,0)
    // (0,1) (1,1).
    frame.vertices(
        0x40_0000,
        8,
        &[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        &[(0, 0x04)],
    );
}

/// BEGIN triangle strip, 4 vertices, END.
fn draw_strip(frame: &mut Frame) {
    frame.method(0x1618, 5);
    frame.method(0x0d74, 0);
    frame.method(0x0d78, 4);
    frame.method(0x1614, 0);
}

#[test]
fn firmware_ui2d_textured_quad() {
    let elf = std::fs::read(STARTER).expect("starter ELF (fw_elf) must be present");
    let mut frame = Frame::new();
    let texel = |x: u32, y: u32| {
        [
            (60 * x) as u8,
            (60 * y) as u8,
            (255 - 40 * (x + y)) as u8,
            255,
        ]
    };
    ui2d_textured(&mut frame, &elf, 4, 0, [1.0, 1.0], [-0.5, 0.5], texel);
    draw_strip(&mut frame);
    // Pixels 16..48 in both axes; vertex y = 1 is at the top (window y
    // 16), where t = 1 - v = 0: texture row 0.
    for y in 0..SIZE {
        for x in 0..SIZE {
            let expected = if (16..48).contains(&x) && (16..48).contains(&y) {
                texel((x - 16) / 8, (y - 16) / 8)
            } else {
                [0, 0, 0, 255]
            };
            assert_eq!(frame.pixel(x, y), expected, "{x},{y}");
        }
    }
}

/// Throughput of a full-screen 1280x720 frame of the firmware's textured
/// ui2d shader pair. A measurement, not a check: run with
/// `cargo test --release --lib maxwell::tests::full_screen -- --ignored --nocapture`.
#[test]
#[ignore]
fn full_screen_textured_frame_timing() {
    let elf = std::fs::read(STARTER).expect("starter ELF (fw_elf) must be present");
    let mut frame = Frame::sized(1280, 720);
    ui2d_textured(&mut frame, &elf, 256, 4, [2.0, 2.0], [-1.0, 1.0], |x, y| {
        [x as u8, y as u8, (x ^ y) as u8, 255]
    });
    let frames = 10;
    let start = std::time::Instant::now();
    for _ in 0..frames {
        draw_strip(&mut frame);
    }
    let each = start.elapsed() / frames;
    println!("full-screen textured 1280x720 frame: {each:?}");
    // Texel (128, 64) lands at pixel (640, 180).
    assert_eq!(frame.pixel(640, 180), [128, 64, 192, 255]);
}

#[test]
fn empty_inline_transfer_completes_at_launch() {
    // NVN's ClearMemory macro launches a pitch transfer of 0 bytes with a
    // one-word flush-only completion and sends no data; the next method
    // must be accepted.
    let ram = Ram::new(0x10000);
    let mut channel = Channel::new();
    call(&mut channel, &ram, 0, 0, 0xB197);
    for (method, value) in [(0x180, 0), (0x184, 1), (0x188, 0), (0x18c, 0x1000)] {
        call(&mut channel, &ram, 0, method, value);
    }
    call(&mut channel, &ram, 0, 0x1b0, 0x1011);
    call(&mut channel, &ram, 0, 0x2380, 0x7000);
}

/// The SIZE x SIZE frame of `SET_ANTI_ALIAS` `mode` drawn by `kind`:
/// color target and depth buffer (`Z24S8`, holding `fill`) hold the
/// samples as texels (`multisample`), the surface clip and viewport
/// count pixels, samples at `positions` (`SET_ANTI_ALIAS_SAMPLE_POSITIONS`),
/// `SET_ANTI_ALIAS_ALPHA_CONTROL` `alpha_control`. Each of `quads`
/// (corners in normalized device coordinates with their depths, and an
/// RGBA color) is drawn with the depth test LESS and depth writes. The
/// color texels and the depth buffer after a write-back.
fn multisampled_frame(
    kind: super::RendererKind,
    mode: u32,
    positions: u32,
    alpha_control: u32,
    fill: &[u8],
    quads: &[([[f32; 3]; 4], [f32; 4])],
) -> (Vec<u8>, Vec<u8>) {
    let elf = std::fs::read(STARTER).expect("starter ELF (fw_elf) must be present");
    let samples = super::multisample::Samples::from_mode(mode).unwrap();
    let [lx, ly] = samples.log2;
    let mut frame = Frame::sized_with(SIZE << lx, SIZE << ly, kind);
    frame.program(1, 0, starter_program(&elf, 0x4c0e00));
    frame.program(5, 0x800, starter_program(&elf, 0x4be400));
    frame.constants(0, 3, 0x30_0000, &ui2d_vertex_constants(&[]));
    frame.ram.put(DEPTH, fill);
    let half = (SIZE / 2) as f32;
    for (offset, value) in [
        (0x15d0, mode),
        (0x1534, 1),
        (0x11e0, positions),
        (0x11e4, positions),
        (0x11e8, positions),
        (0x11ec, positions),
        (0x153c, alpha_control),
        (0x0754, 1),
        (0x0ff4, SIZE << 16),
        (0x0ff8, SIZE << 16),
        (0x0a00, half.to_bits()),
        (0x0a04, half.to_bits()),
        (0x0a0c, half.to_bits()),
        (0x0a10, half.to_bits()),
        (0x0fe0, (DEPTH >> 32) as u32),
        (0x0fe4, DEPTH as u32),
        (0x0fe8, 0x14),
        (0x0fec, 0),
        (0x1228, SIZE << lx),
        (0x122c, SIZE << ly),
        (0x1230, 1),
        (0x1538, 1),
        (0x12cc, 1),
        (0x130c, 0x201),
        (0x12e8, 1),
    ] {
        frame.method(offset, value);
    }
    for (i, (corners, color)) in quads.iter().enumerate() {
        frame.constants(4, 3, 0x30_1000 + 0x100 * i as u64, &floats(color));
        let quad: Vec<f32> = corners.iter().flatten().copied().collect();
        frame.vertices(0x40_0000 + 0x100 * i as u64, 12, &quad, &[(0, 0x02)]);
        frame.method(0x1618, 7);
        frame.method(0x0d74, 0);
        frame.method(0x0d78, 4);
        frame.method(0x1614, 0);
    }
    frame.channel.flush(&frame.ram).unwrap();
    if let Some(metal) = &frame.channel.hooks.metal {
        assert_eq!(
            metal.fallbacks(),
            (quads.len() as u64, vec![]),
            "Metal drew every quad"
        );
    }
    (
        frame
            .ram
            .bytes(TARGET, ((SIZE * SIZE) << (lx + ly)) as usize * 4),
        frame.ram.bytes(DEPTH, fill.len()),
    )
}

/// Minecraft's 2X1_D3D sample positions: sample 0 at (12/16, 12/16),
/// sample 1 at (4/16, 4/16).
const D3D_2X1: u32 = 0x44cc_44cc;

/// A 2X1 target holds pixel (x, y)'s sample s in texel (2x + s, y), and
/// coverage is taken at the sample positions: a quad whose left edge
/// runs between the two samples of pixel column 10 (at x = 10.5) covers
/// sample 0 (x = 10.75) there, not sample 1 (x = 10.25); columns right
/// of it are covered whole, columns left of it not at all.
#[test]
fn multisampled_target_holds_samples_side_by_side() {
    let edge = 10.5 / (SIZE / 2) as f32 - 1.0;
    let corners = [
        [edge, -1.0, 0.5],
        [1.0, -1.0, 0.5],
        [1.0, 1.0, 0.5],
        [edge, 1.0, 0.5],
    ];
    let fill = vec![0xFFu8; (2 * SIZE * SIZE * 4) as usize];
    let (colors, _) = multisampled_frame(
        super::RendererKind::Software,
        5,
        D3D_2X1,
        0,
        &fill,
        &[(corners, [1.0, 0.0, 0.0, 1.0])],
    );
    let texel = |x: u32, y: u32| &colors[((y * 2 * SIZE + x) * 4) as usize..][..4];
    for y in 0..SIZE {
        for x in 0..SIZE {
            let expected = [x >= 11 || x == 10, x >= 11];
            for (sample, covered) in expected.into_iter().enumerate() {
                // Uncovered samples keep what the frame starts with.
                let want: &[u8] = if covered {
                    &[255, 0, 0, 255]
                } else {
                    &[0, 0, 0, 255]
                };
                assert_eq!(
                    texel(2 * x + sample as u32, y),
                    want,
                    "{x},{y} sample {sample}"
                );
            }
        }
    }
}

/// Multisampled drawing on Metal gives the software renderer's samples
/// and depth buffer bit for bit, for 2X1_D3D and 2X2, with and without
/// alpha to coverage: two overlapping quads with corners off the pixel
/// grid, depth-tested against a buffer of random depths, leave pixels
/// whose samples differ.
#[test]
fn multisampling_matches_software() {
    use super::RendererKind::{Metal, Software};
    let mut random = Random(0x9e37_79b9_7f4a_7c15);
    let quads = [
        (
            [
                [-0.83, -0.71, -0.9],
                [0.77, -0.93, 0.6],
                [0.91, 0.8, 0.1],
                [-0.6, 0.66, -0.3],
            ],
            [0.25, 0.5, 1.0, 0.6],
        ),
        (
            [
                [-0.3, -0.97, 0.7],
                [0.95, -0.2, -0.6],
                [0.4, 0.93, -0.1],
                [-0.97, 0.13, 0.4],
            ],
            [1.0, 0.25, 0.0, 0.3],
        ),
    ];
    // 2X2 at Direct3D's standard 4x positions: (6, 2), (14, 6), (2, 10),
    // (10, 14).
    for (mode, positions) in [(5, D3D_2X1), (2, 0xeaa2_6e26_u32)] {
        let samples = super::multisample::Samples::from_mode(mode).unwrap();
        let texels = (SIZE * SIZE) as usize * samples.count();
        let mut fill = vec![0u8; texels * 4];
        for texel in fill.chunks_mut(4) {
            texel.copy_from_slice(&random.next().to_le_bytes());
        }
        for alpha_control in [0, 1] {
            let run =
                |kind| multisampled_frame(kind, mode, positions, alpha_control, &fill, &quads);
            let (software, metal) = (run(Software), run(Metal));
            let what = format!("mode {mode} alpha control {alpha_control}");
            assert!(software.0 == metal.0, "{what}: samples differ");
            assert!(software.1 == metal.1, "{what}: depth buffers differ");
            // Some pixel has samples of different colors.
            let [lx, _] = samples.log2;
            let row = (SIZE << lx) as usize * 4;
            let split = (0..SIZE as usize).any(|y| {
                (0..SIZE as usize).any(|x| {
                    let at = |s: usize| &software.0[y * row + ((x << lx) + s) * 4..][..4];
                    at(0) != at(1)
                })
            });
            assert!(split, "{what}: no partly covered pixel");
        }
    }
}

mod lines;
