//! Block-compressed formats: BC1-BC5 (DXT1, DXT2/3, DXT4/5, DXN1, DXN2)
//! and BC7. Each decoder turns one 4x4 block into 16 RGBA texels, row by
//! row.
//!
//! * BC1-BC3 decode to 8-bit UNORM colors. The interpolated colors follow
//!   NVIDIA GPUs'
//!   integer formulas as measured by rgbcx [public: richgel999/bc7enc
//!   `rgbcx.h` `interp_5_nv`, `interp_6_nv`, `interp_half_5_nv`,
//!   `interp_half_6_nv`, `unpack_bc1` mode `cBC1NVidia`].
//! * BC4/BC5 ("DXN") decode to more than 8 bits (16-bit components); the
//!   exact precision of
//!   GM20B is not public, so the interpolation is computed exactly
//!   (`((7 - i) * e0 + i * e1) / 7` and the 6-value variant) [public:
//!   D3D11.3 functional spec §19.5 "Block Compression" BC4/BC5].
//! * BC7 is decoded per the BC7 specification [public: D3D11.3 functional
//!   spec §19.5 "BC7"].

use super::format::snorm_to_f32;

/// One decoded 4x4 block: RGBA of texel `x + 4 * y`.
pub type Block4x4 = [[f32; 4]; 16];

fn expand5(v: u32) -> u32 {
    (v << 3) | (v >> 2)
}

fn expand6(v: u32) -> u32 {
    (v << 2) | (v >> 4)
}

/// The four palette entries of a BC1-style color block, as 8-bit RGBA.
/// `allow_three_color` selects the BC1 rule (`color0 <= color1` gives the
/// three-color + transparent black palette); BC2/BC3 always use four colors
/// (D3D11.3 §19.5.7, §19.5.8).
fn color_palette(block: &[u8], allow_three_color: bool) -> [[u8; 4]; 4] {
    let c0 = u32::from(u16::from_le_bytes([block[0], block[1]]));
    let c1 = u32::from(u16::from_le_bytes([block[2], block[3]]));
    let (r0, g0, b0) = ((c0 >> 11) & 31, (c0 >> 5) & 63, c0 & 31);
    let (r1, g1, b1) = ((c1 >> 11) & 31, (c1 >> 5) & 63, c1 & 31);
    let (g0e, g1e) = (expand6(g0) as i32, expand6(g1) as i32);
    let color = |r: u32, g: i32, b: u32| [r as u8, g as u8, b as u8, 255];
    let p0 = color(expand5(r0), g0e, expand5(b0));
    let p1 = color(expand5(r1), g1e, expand5(b1));
    // rgbcx `interp_5_nv`/`interp_6_nv` and their "half" variants; integer
    // division truncates toward zero as in C.
    let third5 = |a: u32, b: u32| (2 * a + b) * 22 / 8;
    let third6 = |a: i32, b: i32| {
        let diff = b - a;
        (256 * a + diff / 4 + 128 + diff * 80) / 256
    };
    let half5 = |a: u32, b: u32| (a + b) * 33 / 8;
    let half6 = |a: i32, b: i32| {
        let diff = b - a;
        (256 * a + diff / 4 + 128 + diff * 128) / 256
    };
    if c0 > c1 || !allow_three_color {
        [
            p0,
            p1,
            color(third5(r0, r1), third6(g0e, g1e), third5(b0, b1)),
            color(third5(r1, r0), third6(g1e, g0e), third5(b1, b0)),
        ]
    } else {
        [
            p0,
            p1,
            color(half5(r0, r1), half6(g0e, g1e), half5(b0, b1)),
            [0, 0, 0, 0],
        ]
    }
}

fn color_block(block: &[u8], allow_three_color: bool, out: &mut Block4x4) {
    let palette = color_palette(block, allow_three_color);
    let selectors = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
    for (i, texel) in out.iter_mut().enumerate() {
        let entry = palette[((selectors >> (2 * i)) & 3) as usize];
        *texel = entry.map(|c| f32::from(c) / 255.0);
    }
}

/// BC1 (`DXT1`): 8 bytes.
pub fn decode_bc1(block: &[u8], out: &mut Block4x4) {
    color_block(block, true, out);
}

/// BC2 (`DXT23`): 16 bytes, explicit 4-bit alpha then a four-color block.
pub fn decode_bc2(block: &[u8], out: &mut Block4x4) {
    color_block(&block[8..16], false, out);
    let alpha = u64::from_le_bytes(block[..8].try_into().unwrap());
    for (i, texel) in out.iter_mut().enumerate() {
        texel[3] = ((alpha >> (4 * i)) & 15) as f32 / 15.0;
    }
}

/// BC3 (`DXT45`): 16 bytes, a BC4 alpha block then a four-color block.
pub fn decode_bc3(block: &[u8], out: &mut Block4x4) {
    color_block(&block[8..16], false, out);
    let mut alpha = [0.0; 16];
    decode_bc4_channel(&block[..8], false, &mut alpha);
    for (texel, a) in out.iter_mut().zip(alpha) {
        texel[3] = a;
    }
}

/// One BC4 channel (8 bytes) into 16 values (D3D11.3 §19.5.9, §19.5.10).
pub fn decode_bc4_channel(block: &[u8], signed: bool, out: &mut [f32; 16]) {
    let (e0, e1, eight) = if signed {
        let (a, b) = (block[0] as i8, block[1] as i8);
        (
            snorm_to_f32(u32::from(block[0]), 8),
            snorm_to_f32(u32::from(block[1]), 8),
            a > b,
        )
    } else {
        (
            f32::from(block[0]) / 255.0,
            f32::from(block[1]) / 255.0,
            block[0] > block[1],
        )
    };
    let (e0, e1) = (f64::from(e0), f64::from(e1));
    let mut palette = [0.0f32; 8];
    palette[0] = e0 as f32;
    palette[1] = e1 as f32;
    if eight {
        for i in 1..7 {
            palette[i + 1] = (((7 - i) as f64 * e0 + i as f64 * e1) / 7.0) as f32;
        }
    } else {
        for i in 1..5 {
            palette[i + 1] = (((5 - i) as f64 * e0 + i as f64 * e1) / 5.0) as f32;
        }
        palette[6] = if signed { -1.0 } else { 0.0 };
        palette[7] = 1.0;
    }
    let mut bits = [0u8; 8];
    bits[..6].copy_from_slice(&block[2..8]);
    let indices = u64::from_le_bytes(bits);
    for (i, value) in out.iter_mut().enumerate() {
        *value = palette[((indices >> (3 * i)) & 7) as usize];
    }
}

/// BC4 (`DXN1`): 8 bytes, red only (G, B 0, A 1).
pub fn decode_bc4(block: &[u8], signed: bool, out: &mut Block4x4) {
    let mut red = [0.0; 16];
    decode_bc4_channel(block, signed, &mut red);
    for (texel, r) in out.iter_mut().zip(red) {
        *texel = [r, 0.0, 0.0, 1.0];
    }
}

/// BC5 (`DXN2`): 16 bytes, red then green (B 0, A 1).
pub fn decode_bc5(block: &[u8], signed: bool, out: &mut Block4x4) {
    let mut red = [0.0; 16];
    let mut green = [0.0; 16];
    decode_bc4_channel(&block[..8], signed, &mut red);
    decode_bc4_channel(&block[8..16], signed, &mut green);
    for ((texel, r), g) in out.iter_mut().zip(red).zip(green) {
        *texel = [r, g, 0.0, 1.0];
    }
}

/// Reads a 128-bit block LSB first.
struct Bits {
    value: u128,
    position: u32,
}

impl Bits {
    fn take(&mut self, count: u32) -> u32 {
        if count == 0 {
            return 0;
        }
        let value = ((self.value >> self.position) & ((1u128 << count) - 1)) as u32;
        self.position += count;
        value
    }
}

/// Per-mode layout of a BC7 block (D3D11.3 §19.5.14.4).
struct Bc7Mode {
    subsets: usize,
    partition_bits: u32,
    rotation_bits: u32,
    index_selection_bits: u32,
    color_bits: u32,
    alpha_bits: u32,
    endpoint_pbits: bool,
    shared_pbits: bool,
    index_bits: u32,
    index2_bits: u32,
}

const BC7_MODES: [Bc7Mode; 8] = [
    Bc7Mode {
        subsets: 3,
        partition_bits: 4,
        rotation_bits: 0,
        index_selection_bits: 0,
        color_bits: 4,
        alpha_bits: 0,
        endpoint_pbits: true,
        shared_pbits: false,
        index_bits: 3,
        index2_bits: 0,
    },
    Bc7Mode {
        subsets: 2,
        partition_bits: 6,
        rotation_bits: 0,
        index_selection_bits: 0,
        color_bits: 6,
        alpha_bits: 0,
        endpoint_pbits: false,
        shared_pbits: true,
        index_bits: 3,
        index2_bits: 0,
    },
    Bc7Mode {
        subsets: 3,
        partition_bits: 6,
        rotation_bits: 0,
        index_selection_bits: 0,
        color_bits: 5,
        alpha_bits: 0,
        endpoint_pbits: false,
        shared_pbits: false,
        index_bits: 2,
        index2_bits: 0,
    },
    Bc7Mode {
        subsets: 2,
        partition_bits: 6,
        rotation_bits: 0,
        index_selection_bits: 0,
        color_bits: 7,
        alpha_bits: 0,
        endpoint_pbits: true,
        shared_pbits: false,
        index_bits: 2,
        index2_bits: 0,
    },
    Bc7Mode {
        subsets: 1,
        partition_bits: 0,
        rotation_bits: 2,
        index_selection_bits: 1,
        color_bits: 5,
        alpha_bits: 6,
        endpoint_pbits: false,
        shared_pbits: false,
        index_bits: 2,
        index2_bits: 3,
    },
    Bc7Mode {
        subsets: 1,
        partition_bits: 0,
        rotation_bits: 2,
        index_selection_bits: 0,
        color_bits: 7,
        alpha_bits: 8,
        endpoint_pbits: false,
        shared_pbits: false,
        index_bits: 2,
        index2_bits: 2,
    },
    Bc7Mode {
        subsets: 1,
        partition_bits: 0,
        rotation_bits: 0,
        index_selection_bits: 0,
        color_bits: 7,
        alpha_bits: 7,
        endpoint_pbits: true,
        shared_pbits: false,
        index_bits: 4,
        index2_bits: 0,
    },
    Bc7Mode {
        subsets: 2,
        partition_bits: 6,
        rotation_bits: 0,
        index_selection_bits: 0,
        color_bits: 5,
        alpha_bits: 5,
        endpoint_pbits: true,
        shared_pbits: false,
        index_bits: 2,
        index2_bits: 0,
    },
];

fn bc7_weights(bits: u32) -> &'static [u32] {
    match bits {
        2 => &[0, 21, 43, 64],
        3 => &[0, 9, 18, 27, 37, 46, 55, 64],
        _ => &[0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64],
    }
}

/// BC7 (`BC7U`): 16 bytes (D3D11.3 §19.5.14). The reserved mode 8 decodes
/// to all zeros, as that section requires.
pub fn decode_bc7(block: &[u8], out: &mut Block4x4) {
    let value = u128::from_le_bytes(block[..16].try_into().unwrap());
    let mode_index = (value as u8).trailing_zeros();
    if mode_index >= 8 {
        *out = [[0.0; 4]; 16];
        return;
    }
    let mode = &BC7_MODES[mode_index as usize];
    let mut bits = Bits {
        value,
        position: mode_index + 1,
    };
    let partition = bits.take(mode.partition_bits) as usize;
    let rotation = bits.take(mode.rotation_bits);
    let index_selection = bits.take(mode.index_selection_bits);

    // Endpoints: [subset * 2 + endpoint][channel].
    let mut endpoints = [[0u32; 4]; 6];
    let count = mode.subsets * 2;
    for channel in 0..3 {
        for endpoint in endpoints.iter_mut().take(count) {
            endpoint[channel] = bits.take(mode.color_bits);
        }
    }
    for endpoint in endpoints.iter_mut().take(count) {
        endpoint[3] = bits.take(mode.alpha_bits);
    }
    let (mut color_precision, mut alpha_precision) = (mode.color_bits, mode.alpha_bits);
    if mode.endpoint_pbits || mode.shared_pbits {
        let mut pbits = [0u32; 6];
        if mode.endpoint_pbits {
            for pbit in pbits.iter_mut().take(count) {
                *pbit = bits.take(1);
            }
        } else {
            for subset in 0..mode.subsets {
                let pbit = bits.take(1);
                pbits[subset * 2] = pbit;
                pbits[subset * 2 + 1] = pbit;
            }
        }
        for (endpoint, pbit) in endpoints.iter_mut().zip(pbits).take(count) {
            for channel in endpoint.iter_mut() {
                *channel = (*channel << 1) | pbit;
            }
        }
        color_precision += 1;
        if alpha_precision > 0 {
            alpha_precision += 1;
        }
    }
    for endpoint in endpoints.iter_mut().take(count) {
        for (channel, value) in endpoint.iter_mut().enumerate() {
            let precision = if channel < 3 {
                color_precision
            } else {
                alpha_precision
            };
            *value = if precision == 0 {
                255
            } else {
                let shifted = *value << (8 - precision);
                shifted | (shifted >> precision)
            };
        }
    }

    let subset_of = |texel: usize| -> usize {
        match mode.subsets {
            1 => 0,
            2 => ((PARTITIONS2[partition] >> (2 * texel)) & 3) as usize,
            _ => ((PARTITIONS3[partition] >> (2 * texel)) & 3) as usize,
        }
    };
    let is_anchor = |texel: usize| -> bool {
        texel == 0
            || match mode.subsets {
                2 => texel == usize::from(ANCHORS2[partition]),
                3 => ANCHORS3[partition].contains(&(texel as u8)),
                _ => false,
            }
    };
    let mut primary = [0u32; 16];
    for (texel, index) in primary.iter_mut().enumerate() {
        *index = bits.take(mode.index_bits - u32::from(is_anchor(texel)));
    }
    let mut secondary = [0u32; 16];
    if mode.index2_bits > 0 {
        for (texel, index) in secondary.iter_mut().enumerate() {
            *index = bits.take(mode.index2_bits - u32::from(texel == 0));
        }
    }

    for (texel, out_texel) in out.iter_mut().enumerate() {
        let subset = subset_of(texel);
        let (e0, e1) = (endpoints[subset * 2], endpoints[subset * 2 + 1]);
        let (color_index, color_bits, alpha_index, alpha_bits) = if mode.index2_bits == 0 {
            (
                primary[texel],
                mode.index_bits,
                primary[texel],
                mode.index_bits,
            )
        } else if index_selection == 0 {
            (
                primary[texel],
                mode.index_bits,
                secondary[texel],
                mode.index2_bits,
            )
        } else {
            (
                secondary[texel],
                mode.index2_bits,
                primary[texel],
                mode.index_bits,
            )
        };
        let interpolate = |channel: usize, index: u32, bits: u32| {
            let w = bc7_weights(bits)[index as usize];
            ((64 - w) * e0[channel] + w * e1[channel] + 32) >> 6
        };
        let mut rgba = [
            interpolate(0, color_index, color_bits),
            interpolate(1, color_index, color_bits),
            interpolate(2, color_index, color_bits),
            interpolate(3, alpha_index, alpha_bits),
        ];
        match rotation {
            1 => rgba.swap(0, 3),
            2 => rgba.swap(1, 3),
            3 => rgba.swap(2, 3),
            _ => {}
        }
        *out_texel = rgba.map(|c| c as f32 / 255.0);
    }
}

// The BC7 specification's partition and fix-up index tables, packed two
// bits per texel.
/// Two-subset partitions: subset of texel `i` in bits `2i..2i+1`.
const PARTITIONS2: [u32; 64] = [
    0x50505050, 0x40404040, 0x54545454, 0x54505040, 0x50404000, 0x55545450, 0x55545040, 0x54504000,
    0x50400000, 0x55555450, 0x55544000, 0x54400000, 0x55555440, 0x55550000, 0x55555500, 0x55000000,
    0x55150100, 0x00004054, 0x15010000, 0x00405054, 0x00004050, 0x15050100, 0x05010000, 0x40505054,
    0x00404050, 0x05010100, 0x14141414, 0x05141450, 0x01155440, 0x00555500, 0x15014054, 0x05414150,
    0x44444444, 0x55005500, 0x11441144, 0x05055050, 0x05500550, 0x11114444, 0x41144114, 0x44111144,
    0x15055054, 0x01055040, 0x05041050, 0x05455150, 0x14414114, 0x50050550, 0x41411414, 0x00141400,
    0x00041504, 0x00105410, 0x10541000, 0x04150400, 0x50410514, 0x41051450, 0x05415014, 0x14054150,
    0x41050514, 0x41505014, 0x40011554, 0x54150140, 0x50505500, 0x00555050, 0x15151010, 0x54540404,
];
/// Three-subset partitions: subset of texel `i` in bits `2i..2i+1`.
const PARTITIONS3: [u32; 64] = [
    0xaa685050, 0x6a5a5040, 0x5a5a4200, 0x5450a0a8, 0xa5a50000, 0xa0a05050, 0x5555a0a0, 0x5a5a5050,
    0xaa550000, 0xaa555500, 0xaaaa5500, 0x90909090, 0x94949494, 0xa4a4a4a4, 0xa9a59450, 0x2a0a4250,
    0xa5945040, 0x0a425054, 0xa5a5a500, 0x55a0a0a0, 0xa8a85454, 0x6a6a4040, 0xa4a45000, 0x1a1a0500,
    0x0050a4a4, 0xaaa59090, 0x14696914, 0x69691400, 0xa08585a0, 0xaa821414, 0x50a4a450, 0x6a5a0200,
    0xa9a58000, 0x5090a0a8, 0xa8a09050, 0x24242424, 0x00aa5500, 0x24924924, 0x24499224, 0x50a50a50,
    0x500aa550, 0xaaaa4444, 0x66660000, 0xa5a0a5a0, 0x50a050a0, 0x69286928, 0x44aaaa44, 0x66666600,
    0xaa444444, 0x54a854a8, 0x95809580, 0x96969600, 0xa85454a8, 0x80959580, 0xaa141414, 0x96960000,
    0xaaaa1414, 0xa05050a0, 0xa0a5a5a0, 0x96000000, 0x40804080, 0xa9a8a9a8, 0xaaaaaa44, 0x2a4a5254,
];
/// Anchor texel of subset 1 in the two-subset partitions.
const ANCHORS2: [u8; 64] = [
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 2, 8, 2, 2, 8, 8, 15, 2, 8,
    2, 2, 8, 8, 2, 2, 15, 15, 6, 8, 2, 8, 15, 15, 2, 8, 2, 2, 2, 15, 15, 6, 6, 2, 6, 8, 15, 15, 2,
    2, 15, 15, 15, 15, 15, 2, 2, 15,
];
/// Anchor texels of subsets 1 and 2 in the three-subset partitions.
const ANCHORS3: [[u8; 2]; 64] = [
    [3, 15],
    [3, 8],
    [15, 8],
    [15, 3],
    [8, 15],
    [3, 15],
    [15, 3],
    [15, 8],
    [8, 15],
    [8, 15],
    [6, 15],
    [6, 15],
    [6, 15],
    [5, 15],
    [3, 15],
    [3, 8],
    [3, 15],
    [3, 8],
    [8, 15],
    [15, 3],
    [3, 15],
    [3, 8],
    [6, 15],
    [10, 8],
    [5, 3],
    [8, 15],
    [8, 6],
    [6, 10],
    [8, 15],
    [5, 15],
    [15, 10],
    [15, 8],
    [8, 15],
    [15, 3],
    [3, 15],
    [5, 10],
    [6, 10],
    [10, 8],
    [8, 9],
    [15, 10],
    [15, 6],
    [3, 15],
    [15, 8],
    [5, 15],
    [15, 3],
    [15, 6],
    [15, 6],
    [15, 8],
    [3, 15],
    [15, 3],
    [5, 15],
    [5, 15],
    [5, 15],
    [8, 15],
    [5, 15],
    [10, 15],
    [5, 15],
    [10, 15],
    [8, 15],
    [13, 15],
    [15, 3],
    [12, 15],
    [3, 15],
    [3, 8],
];
