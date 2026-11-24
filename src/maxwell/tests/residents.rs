//! Render targets the Metal renderer keeps on the host GPU (`metal`):
//! a texture sampled from the targets that are its levels, and depth
//! buffers held on demand, against the software renderer; and draws
//! taken from the index buffer's bytes against the same vertices listed.

use super::super::Engines;
use super::super::RendererKind::{self, Metal, Software};
use super::{
    Frame, SIZE, STARTER, TARGET, bind_color_target, call, draw_strip, floats, reference_offset,
    starter_program, ui2d_vertex_constants,
};

/// The sampled texture: 16x16 A8B8G8R8, block linear in blocks of two
/// GOBs, two levels.
const TEXTURE: u64 = 0x50_0000;
const TEXELS: u32 = 16;
/// Level 1 (8x8, one GOB) follows level 0's one block of 1024 bytes.
const LEVEL_1: u64 = TEXTURE + 1024;
const POOLS: u64 = 0x68_0000;

/// [`TEXTURE`]'s header: A8B8G8R8 UNORM, RGBA, block linear in blocks of
/// two GOBs, both levels in view.
const TEXTURE_TIC: [u32; 8] = [
    8 | 2 << 7 | 2 << 10 | 2 << 13 | 2 << 16 | 2 << 19 | 3 << 22 | 4 << 25 | 5 << 28,
    TEXTURE as u32,
    (TEXTURE >> 32) as u32 | 3 << 21,
    1 << 3 | 1 << 28,
    (TEXELS - 1) | 1 << 23 | 7 << 29,
    (TEXELS - 1) | 1 << 31,
    0,
    1 << 4,
];

fn level_1_texel(x: u32, y: u32) -> [u8; 4] {
    [(30 * x) as u8, (30 * y) as u8, 99, 255]
}

/// Binds the firmware's textured ui2d shader pair with texture slot 0 on
/// TIC 3 (by default [`TEXTURE_TIC`]: [`TEXTURE`] with both levels in
/// view; TSC 5: nearest texels of the nearest level, clamped to the edge)
/// and its quad's corners.
fn bind_textured(frame: &mut Frame, elf: &[u8], tic: [u32; 8]) {
    frame.program(1, 0, starter_program(elf, 0x4be500));
    frame.program(5, 0x800, starter_program(elf, 0x4beb00));
    frame.constants(
        4,
        3,
        0x30_1000,
        &floats(&[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    );
    frame.constants(0, 1, 0x30_2000, &[0; 256]);
    frame.constants(4, 1, 0x30_3000, &[0; 256]);
    let mut handles = vec![0u8; 0x40];
    handles[0x20..0x24].copy_from_slice(&(3u32 | 5 << 20).to_le_bytes());
    frame.constants(4, 2, 0x30_4000, &handles);
    let tsc: [u32; 8] = [
        2 | 2 << 3 | 2 << 6,
        1 | 1 << 4 | 2 << 6,
        15 << 20,
        0,
        0,
        0,
        0,
        0,
    ];
    frame.ram.put(
        POOLS + 3 * 32,
        &tic.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<_>>(),
    );
    frame.ram.put(
        POOLS + 0x1000 + 5 * 32,
        &tsc.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<_>>(),
    );
    for (offset, value) in [
        (0x1574, 0),
        (0x1578, POOLS as u32),
        (0x157c, 15),
        (0x155c, 0),
        (0x1560, POOLS as u32 + 0x1000),
        (0x1564, 15),
    ] {
        frame.method(offset, value);
    }
    frame.vertices(
        0x40_0000,
        8,
        &[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        &[(0, 0x04)],
    );
}

/// The textured quad over `scale` of the target's half extent from
/// `offset` (see `ui2d_vertex_constants`).
fn textured_quad(frame: &mut Frame, scale: [f32; 2], offset: [f32; 2]) {
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
    draw_strip(frame);
}

/// Clears `x0..x1` by `y0..y1` of the bound target to `color`.
fn clear_rect(frame: &mut Frame, [x0, x1, y0, y1]: [u32; 4], color: [f32; 4]) {
    for (i, value) in color.into_iter().enumerate() {
        frame.method(0x0d80 + 4 * i as u32, value.to_bits());
    }
    frame.method(0x0e00, 1);
    frame.method(0x0e04, x0 | x1 << 16);
    frame.method(0x0e08, y0 | y1 << 16);
    frame.method(0x19d0, 0xF << 2);
    frame.method(0x0e00, 0);
}

/// A frame that renders level 0 of [`TEXTURE`] `rounds` times (other
/// contents each time), each time followed by two quads sampling the
/// texture into the 64x64 target: one magnified (level 0) and one at half
/// size (level 1, which only memory holds). The target and the texture's
/// memory after the channel signaled, and how many textures the Metal
/// renderer sampled from its targets.
fn sampled_target_frame(kind: RendererKind, rounds: u32) -> (Vec<u8>, Vec<u8>, usize) {
    let elf = std::fs::read(STARTER).expect("starter ELF (fw_elf) must be present");
    let mut frame = Frame::sized_with(SIZE, SIZE, kind);
    bind_textured(&mut frame, &elf, TEXTURE_TIC);
    for y in 0..TEXELS / 2 {
        for x in 0..TEXELS / 2 {
            frame.ram.put(
                LEVEL_1 + reference_offset(x * 4, y, TEXELS * 2, 0),
                &level_1_texel(x, y),
            );
        }
    }
    for round in 0..rounds {
        // Level 0 as render target 0: block linear, two GOBs a block.
        for (offset, value) in [
            (0x800, (TEXTURE >> 32) as u32),
            (0x804, TEXTURE as u32),
            (0x808, TEXELS),
            (0x80c, TEXELS),
            (0x810, 0xD5),
            (0x814, 1 << 4),
            (0x818, 1),
            (0x820, 0),
            (0x0ff4, TEXELS << 16),
            (0x0ff8, TEXELS << 16),
        ] {
            frame.method(offset, value);
        }
        let shade = 0.2 * round as f32;
        clear_rect(&mut frame, [0, 16, 0, 16], [1.0, shade, 0.0, 1.0]);
        clear_rect(&mut frame, [0, 8, 0, 16], [0.0, 1.0, shade, 1.0]);
        clear_rect(&mut frame, [4, 12, 4 + round, 12], [0.2, 0.4, 0.6, 0.8]);
        bind_color_target(&mut frame.channel, &frame.ram, TARGET, SIZE, SIZE);
        // 32 pixels across 16 texels, then 8 pixels across them.
        textured_quad(&mut frame, [1.0, 1.0], [-0.5, 0.5 - 0.25 * round as f32]);
        textured_quad(&mut frame, [0.25, 0.25], [-1.0 + 0.25 * round as f32, 1.0]);
    }
    frame.channel.flush(&frame.ram).unwrap();
    let composites = match &frame.channel.hooks.metal {
        Some(metal) => {
            assert_eq!(metal.fallbacks().1, vec![], "Metal drew everything");
            metal.composites()
        }
        None => 0,
    };
    (
        frame.ram.bytes(TARGET, (SIZE * SIZE * 4) as usize),
        frame.ram.bytes(TEXTURE, 1024 + 512),
        composites,
    )
}

/// A texture whose level 0 is a render target the Metal renderer holds is
/// sampled from that target on the GPU (its other level from memory),
/// with the software renderer's result bit for bit, also when the target
/// is rendered again between draws sampling it; and guest memory gets the
/// rendered level at the signal.
#[test]
fn texture_sampled_from_its_render_target_matches_software() {
    for rounds in [1, 3] {
        let (software, software_texture, _) = sampled_target_frame(Software, rounds);
        let (metal, metal_texture, composites) = sampled_target_frame(Metal, rounds);
        // The quads drew texels of both levels.
        let pixel = |image: &[u8], x: u32, y: u32| -> [u8; 4] {
            image[((y * SIZE + x) * 4) as usize..][..4]
                .try_into()
                .unwrap()
        };
        let drawn = (0..SIZE * SIZE)
            .filter(|i| pixel(&software, i % SIZE, i / SIZE) != [0, 0, 0, 255])
            .count();
        assert!(drawn >= 32 * 32, "{rounds} rounds: {drawn} pixels drawn");
        assert!(
            software.chunks(4).any(|p| p[2] == 99),
            "{rounds} rounds: level 1 sampled"
        );
        assert!(software == metal, "{rounds} rounds: targets differ");
        assert!(
            software_texture == metal_texture,
            "{rounds} rounds: texture memory differs"
        );
        if super::super::metal::available() {
            assert_eq!(composites, 1, "{rounds} rounds: sampled from the target");
        }
    }
}

/// Where the depth buffer of [`depth_frame`] is, and where
/// [`copy_depth`] copies it.
const DEPTH: u64 = 0x60_0000;
const COPY: u64 = 0x70_0000;
const DEPTH_BYTES: usize = (SIZE * SIZE * 4) as usize;

/// What the depth buffer holds before the frame: depth 1, stencil 0x5a
/// (Z24S8).
fn depth_fill() -> Vec<u8> {
    [0x5a, 0xff, 0xff, 0xff].repeat((SIZE * SIZE) as usize)
}

/// A 64x64 frame drawn by `kind` with a block-linear Z24S8 depth buffer
/// at [`DEPTH`]: two solid quads across the target with depth writes
/// (corners at different depths), after which the channel signals.
fn depth_frame(kind: RendererKind) -> Frame {
    let elf = std::fs::read(STARTER).expect("starter ELF (fw_elf) must be present");
    let mut frame = Frame::sized_with(SIZE, SIZE, kind);
    frame.program(1, 0, starter_program(&elf, 0x4c0e00));
    frame.program(5, 0x800, starter_program(&elf, 0x4be400));
    frame.constants(0, 3, 0x30_0000, &ui2d_vertex_constants(&[]));
    frame.ram.put(DEPTH, &depth_fill());
    for (offset, value) in [
        (0x0fe0, (DEPTH >> 32) as u32),
        (0x0fe4, DEPTH as u32),
        (0x0fe8, 0x14),
        (0x0fec, 0),
        (0x1228, SIZE),
        (0x122c, SIZE),
        (0x1230, 1),
        (0x1538, 1),
        (0x12cc, 1),
        (0x130c, 0x201),
        (0x12e8, 1),
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
    frame
}

/// Copies the depth buffer's texels with the 2D engine (as a 4-byte
/// color surface, texel for texel) to a pitch surface at [`COPY`]; its
/// bytes.
fn copy_depth(frame: &mut Frame) -> Vec<u8> {
    call(&mut frame.channel, &frame.ram, 3, 0, 0x902D);
    let mut method = |offset: u32, value: u32| {
        call(&mut frame.channel, &frame.ram, 3, offset, value);
    };
    // Source: block linear, one GOB a block. Destination: pitch.
    for (base, layout, address) in [(0x230, 0, DEPTH), (0x200, 1, COPY)] {
        for (offset, value) in [
            (0x00, 0xD5),
            (0x04, layout),
            (0x08, 0),
            (0x0c, 1),
            (0x14, SIZE * 4),
            (0x18, SIZE),
            (0x1c, SIZE),
            (0x20, 0),
            (0x24, address as u32),
        ] {
            method(base + offset, value);
        }
    }
    method(0x2ac, 3);
    for (offset, value) in [
        (0x88c, 0),
        (0x8b0, 0),
        (0x8b4, 0),
        (0x8b8, SIZE),
        (0x8bc, SIZE),
        (0x8c0, 0),
        (0x8c4, 1),
        (0x8c8, 0),
        (0x8cc, 1),
        (0x8d0, 0),
        (0x8d4, 0),
        (0x8d8, 0),
        (0x8dc, 0),
    ] {
        method(offset, value);
    }
    frame.ram.bytes(COPY, DEPTH_BYTES)
}

/// The Metal renderer holds a depth buffer on demand: a signal leaves
/// guest memory as it was, and an engine reading the buffer's memory (a
/// 2D blit of it) sees what was rendered, as the software renderer left
/// it in memory.
#[test]
fn depth_held_on_demand_is_written_back_for_an_engine_read() {
    let mut software = depth_frame(Software);
    let rendered = software.ram.bytes(DEPTH, DEPTH_BYTES);
    assert!(rendered != depth_fill(), "the quads wrote depth");
    let expected = copy_depth(&mut software);
    assert!(expected != depth_fill());

    let mut metal = depth_frame(Metal);
    if metal.channel.hooks.metal.is_some() {
        assert!(
            metal.ram.bytes(DEPTH, DEPTH_BYTES) == depth_fill(),
            "the signal left the depth buffer on the GPU"
        );
    }
    assert!(copy_depth(&mut metal) == expected, "the blit read stale depth");
    assert!(
        metal.ram.bytes(DEPTH, DEPTH_BYTES) == rendered,
        "the depth buffer was written back for the blit"
    );
    // A snapshot's write-back changes nothing more.
    metal.channel.settle(&metal.ram).unwrap();
    assert!(metal.ram.bytes(DEPTH, DEPTH_BYTES) == rendered);
}

/// Memory of a depth buffer held on demand that the guest gave another
/// use (other bytes are there now, written before the work that follows
/// was submitted) is not written: an engine reading it, and a snapshot,
/// find the guest's bytes.
#[test]
fn depth_held_on_demand_is_dropped_when_its_memory_is_reused() {
    let reused: Vec<u8> = (0..DEPTH_BYTES).map(|i| (i * 7 + i / 256) as u8).collect();
    // What a blit of those bytes gives.
    let mut plain = Frame::sized_with(SIZE, SIZE, Software);
    plain.ram.put(DEPTH, &reused);
    let expected = copy_depth(&mut plain);

    let mut metal = depth_frame(Metal);
    metal.ram.put(DEPTH, &reused);
    metal.channel.synchronize(&metal.ram).unwrap();
    assert!(copy_depth(&mut metal) == expected, "rendered depth written");
    assert!(metal.ram.bytes(DEPTH, DEPTH_BYTES) == reused);
    metal.channel.settle(&metal.ram).unwrap();
    assert!(metal.ram.bytes(DEPTH, DEPTH_BYTES) == reused);

    // The same with the snapshot's write-back first.
    let mut metal = depth_frame(Metal);
    metal.ram.put(DEPTH, &reused);
    metal.channel.synchronize(&metal.ram).unwrap();
    metal.channel.settle(&metal.ram).unwrap();
    assert!(metal.ram.bytes(DEPTH, DEPTH_BYTES) == reused);
    assert!(copy_depth(&mut metal) == expected);
}

/// The Z24S8 depth buffer of [`depth_frame`], held on the GPU, sampled as
/// a texture of the same bytes (header format 0x29: stencil R, depth G,
/// here G in red, green and blue) by a quad over the whole target: read
/// on the GPU from the held buffer, with the software renderer's result
/// bit for bit.
#[test]
fn held_depth_buffer_sampled_as_a_texture_matches_software() {
    let elf = std::fs::read(STARTER).expect("starter ELF (fw_elf) must be present");
    let draw = |kind: RendererKind| {
        let mut frame = depth_frame(kind);
        // Depth tests and writes off: the buffer is only sampled.
        frame.method(0x12cc, 0);
        frame.method(0x12e8, 0);
        let tic: [u32; 8] = [
            0x29 | 4 << 7 | 2 << 10 | 2 << 13 | 2 << 16 | 3 << 19 | 3 << 22 | 3 << 25 | 7 << 28,
            DEPTH as u32,
            (DEPTH >> 32) as u32 | 3 << 21,
            0,
            (SIZE - 1) | 1 << 23 | 7 << 29,
            (SIZE - 1) | 1 << 31,
            0,
            0,
        ];
        bind_textured(&mut frame, &elf, tic);
        textured_quad(&mut frame, [2.0, 2.0], [-1.0, 1.0]);
        frame.channel.flush(&frame.ram).unwrap();
        let composites = frame.channel.hooks.metal.as_ref().map_or(0, |metal| {
            assert_eq!(metal.fallbacks().1, vec![], "Metal drew everything");
            metal.composites()
        });
        (frame.ram.bytes(TARGET, (SIZE * SIZE * 4) as usize), composites)
    };
    let (software, _) = draw(Software);
    let (metal, composites) = draw(Metal);
    let shades: std::collections::HashSet<u8> = software.chunks(4).map(|p| p[0]).collect();
    assert!(shades.len() > 8, "the depths sampled vary: {shades:?}");
    assert!(software == metal, "targets differ");
    if super::super::metal::available() {
        assert_eq!(composites, 1, "sampled from the held buffer");
    }
}

/// A mesh of 8x8 quads over the target (each corner at its own depth
/// value, unused: no depth buffer) drawn by `kind` with the firmware's
/// solid ui2d shaders: from the index buffer with indices of `size`
/// bytes over 256 vertices that follow `base` others in the stream
/// (`SET_GLOBAL_BASE_VERTEX_INDEX`), with primitive restart enabled if
/// `restart` (no index is the restart index); or, with `size` 0, as the
/// same triangles' vertices in order. The target.
fn mesh_frame(kind: RendererKind, size: u32, base: u32, restart: bool) -> Vec<u8> {
    let elf = std::fs::read(STARTER).expect("starter ELF (fw_elf) must be present");
    let mut frame = Frame::sized_with(SIZE, SIZE, kind);
    frame.program(1, 0, starter_program(&elf, 0x4c0e00));
    frame.program(5, 0x800, starter_program(&elf, 0x4be400));
    frame.constants(0, 3, 0x30_0000, &ui2d_vertex_constants(&[]));
    frame.constants(4, 3, 0x30_1000, &floats(&[1.0, 0.25, 0.0, 1.0]));
    let mut corners: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    for qy in 0..8u32 {
        for qx in 0..8u32 {
            let (x0, y0) = (qx as f32 / 4.0 - 1.0, qy as f32 / 4.0 - 1.0);
            let (x1, y1) = (x0 + 0.25, y0 + 0.25);
            let first = corners.len() as u32;
            corners.extend([[x0, y0, 0.1], [x1, y0, 0.2], [x1, y1, 0.3], [x0, y1, 0.4]]);
            // Every other quad wound the other way, and left out of some.
            let order: [u32; 6] = if (qx + qy) % 2 == 0 {
                [0, 1, 2, 0, 2, 3]
            } else {
                [0, 2, 1, 0, 3, 2]
            };
            if (qx + 2 * qy) % 5 != 0 {
                indices.extend(order.map(|i| first + i));
            }
        }
    }
    frame.method(0x1618, 4); // BEGIN TRIANGLES
    if size == 0 {
        let vertices: Vec<f32> = indices
            .iter()
            .flat_map(|&i| corners[i as usize])
            .collect();
        frame.vertices(0x40_0000, 12, &vertices, &[(0, 0x02)]);
        frame.method(0x0d74, 0);
        frame.method(0x0d78, indices.len() as u32);
    } else {
        let vertices: Vec<f32> = std::iter::repeat_n([9.0, 9.0, 9.0], base as usize)
            .chain(corners.iter().copied())
            .flatten()
            .collect();
        frame.vertices(0x40_0000, 12, &vertices, &[(0, 0x02)]);
        // The run starts at the buffer's third index.
        let bytes: Vec<u8> = [0xdead, 0xbeef]
            .iter()
            .chain(&indices)
            .flat_map(|i| i.to_le_bytes()[..size as usize].to_vec())
            .collect();
        const INDICES: u64 = 0x48_0000;
        frame.ram.put(INDICES, &bytes);
        for (offset, value) in [
            (0x17c8, 0),
            (0x17cc, INDICES as u32),
            (0x17d0, 0),
            (0x17d4, INDICES as u32 + bytes.len() as u32 - 1),
            (0x17d8, size.trailing_zeros()),
            (0x17dc, 2),
            (0x1434, base),
            (0x1644, u32::from(restart)),
            (0x1648, 0xffff_ffff),
            (0x17e0, indices.len() as u32),
        ] {
            frame.method(offset, value);
        }
    }
    frame.method(0x1614, 0);
    frame.channel.flush(&frame.ram).unwrap();
    if let Some(metal) = &frame.channel.hooks.metal {
        assert_eq!(metal.fallbacks(), (1, vec![]), "Metal drew the mesh");
        assert_eq!(metal.cpu_vertices(), vec![], "on the native path");
    }
    frame.ram.bytes(TARGET, (SIZE * SIZE * 4) as usize)
}

/// A native draw taken straight from the index buffer's bytes
/// (`pipeline::indexed_geometry`: 16- and 32-bit indices, with a base
/// vertex) covers what the draw of the same triangles' vertices listed in
/// order does, bit for bit; so does one whose elements are listed from
/// the run (primitive restart enabled). The software renderer agrees with
/// itself likewise.
#[test]
fn indexed_draws_match_listed_ones() {
    for kind in [Software, Metal] {
        let listed = mesh_frame(kind, 0, 0, false);
        assert!(
            listed.chunks(4).filter(|p| p[0] == 255).count() > 1500,
            "{kind:?}: the mesh was drawn"
        );
        for (size, base, restart) in [
            (2, 0, false),
            (2, 5, false),
            (4, 0, false),
            (4, 3, false),
            (2, 0, true),
            (2, 7, true),
        ] {
            assert!(
                mesh_frame(kind, size, base, restart) == listed,
                "{kind:?}: {size}-byte indices, base {base}, restart {restart}"
            );
        }
    }
}
