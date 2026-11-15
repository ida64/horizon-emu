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
        let ram = Ram::new(0x80_0000 + (width * height * 4) as usize);
        let mut channel = Channel::new();
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
