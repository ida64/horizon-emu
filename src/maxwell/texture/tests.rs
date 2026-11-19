use super::*;
use crate::maxwell::sampling::{FetchRequest, GatherRequest, Lod, Query, SampleRequest};

struct Memory(Vec<u8>);
impl GpuMemory for Memory {
    fn read(&self, address: u64, buffer: &mut [u8]) -> Option<()> {
        buffer.copy_from_slice(
            self.0
                .get(address as usize..address as usize + buffer.len())?,
        );
        Some(())
    }
    fn write(&self, _: u64, _: &[u8]) -> Option<()> {
        None
    }
}
fn descriptor(format: u32, width: u32, height: u32) -> [u32; 8] {
    [
        format
            | (2 << 7)
            | (2 << 10)
            | (2 << 13)
            | (2 << 16)
            | (2 << 19)
            | (3 << 22)
            | (4 << 25)
            | (5 << 28),
        0,
        3 << 21,
        0,
        (width - 1) | (1 << 23) | (7 << 29),
        (height - 1) | (1 << 31),
        0,
        0,
    ]
}
fn sampler() -> Tsc {
    Tsc::parse([
        2 | (2 << 3) | (2 << 6),
        2 | (2 << 4) | (1 << 6),
        15 << 20,
        0,
        0,
        0,
        0,
        0,
    ])
    .unwrap()
}
fn request(coords: [f32; 3]) -> SampleRequest {
    SampleRequest {
        target: Target::Texture2D,
        array: false,
        coords,
        layer: 0,
        lod: Lod::Level(0.),
        offset: [0; 3],
        compare: None,
    }
}
fn image() -> Texture {
    let tic = Tic::parse(descriptor(8, 2, 2)).unwrap();
    Texture {
        tic,
        levels: vec![Level {
            width: 2,
            height: 2,
            depth: 1,
            texels: vec![
                [0., 0., 0., 1.],
                [1., 0., 0., 1.],
                [0., 1., 0., 1.],
                [1., 1., 0., 1.],
            ],
        }],
    }
}
#[test]
fn filtering_wrap_border_and_gather() {
    let texture = image();
    let mut tsc = sampler();
    assert_eq!(
        texture.sample(&tsc, &request([0.25, 0.25, 0.])).unwrap(),
        [0., 0., 0., 1.]
    );
    assert_eq!(
        texture.sample(&tsc, &request([0.5, 0.5, 0.])).unwrap(),
        [0.5, 0.5, 0., 1.]
    );
    tsc.border = [0.8, 0.6, 0.4, 0.2];
    // At x=-0.25, repeat selects rightmost, mirror and mirror-once select
    // leftmost, edge clamps leftmost, border selects border; CLAMP blends.
    for (mode, red) in [
        (0, 1.),
        (1, 0.),
        (2, 0.),
        (3, 0.8),
        (4, 0.4),
        (5, 0.),
        (6, 0.),
        (7, 0.),
    ] {
        tsc.wrap[0] = mode;
        let c = texture.sample(&tsc, &request([-0.25, 0.25, 0.])).unwrap();
        assert!((c[0] - red).abs() < 1e-6, "mode={mode} {c:?}");
    }
    tsc.wrap = [2; 3];
    assert_eq!(
        texture
            .gather(
                &tsc,
                &GatherRequest {
                    target: Target::Texture2D,
                    array: false,
                    coords: [0.5, 0.5, 0.],
                    layer: 0,
                    offset: [0; 3],
                    component: 0,
                    compare: None
                }
            )
            .unwrap(),
        [0., 1., 1., 0.]
    );
    assert_eq!(
        texture
            .fetch(&FetchRequest {
                target: Target::Texture2D,
                array: false,
                coords: [-1, 0, 0],
                layer: 0,
                level: 0,
                offset: [0; 3]
            })
            .unwrap(),
        [0.; 4]
    );
}
#[test]
fn derivative_mip_selection() {
    let mut texture = image();
    texture.tic.shape.levels = 2;
    texture.tic.max_level = 1;
    texture.levels.push(Level {
        width: 1,
        height: 1,
        depth: 1,
        texels: vec![[0.75; 4]],
    });
    let mut tsc = sampler();
    tsc.mip_filter = 2;
    let mut r = request([0.25, 0.25, 0.]);
    r.lod = Lod::Derivatives {
        ddx: [1., 0., 0.],
        ddy: [0., 1., 0.],
        bias: 0.,
    };
    assert_eq!(texture.sample(&tsc, &r).unwrap(), [0.75; 4]);
    assert_eq!(
        texture.lod(&tsc, Target::Texture2D, [1., 0., 0.], [0., 1., 0.]),
        [1., 1.]
    );
    assert_eq!(
        texture.query(Query::Dimension { level: 1 }).unwrap(),
        [1, 1, 1, 2]
    );
}
#[test]
fn pitch_srgb_and_swizzle() {
    let mut words = descriptor(8, 1, 1);
    words[2] = 2 << 21;
    words[3] = 1;
    words[4] |= 1 << 22;
    let tic = Tic::parse(words).unwrap();
    let mut data = vec![0; 32];
    data[..4].copy_from_slice(&[128, 64, 255, 128]);
    let texture = Texture::decode(&Memory(data), &tic).unwrap();
    let c = texture.levels[0].texels[0];
    assert!((c[0] - 0.2158605).abs() < 1e-6);
    assert!((c[1] - 0.05126946).abs() < 1e-6);
    assert_eq!(c[2], 1.);
    assert_eq!(c[3], 128. / 255.);
    fn send_sync<T: Send + Sync>() {}
    send_sync::<Texture>();
}
/// Word 6 bit 0 (COLOR_KEY_OP) only means something to the colour-key
/// header versions, which stay refused; a block-linear header with it set
/// (as the Mii editor's arrive) parses as without it.
#[test]
fn color_key_op_outside_color_key_headers() {
    let plain = descriptor(8, 2, 2);
    let mut keyed = plain;
    keyed[6] = 1;
    let tic = Tic::parse(keyed).unwrap();
    assert_eq!(tic.shape, Tic::parse(plain).unwrap().shape);
    keyed[2] = 4 << 21; // BLOCKLINEAR_COLORKEY
    assert!(Tic::parse(keyed).is_err());
}
#[test]
fn bc_known_vectors() {
    // D3D11.3 §19.5: BC1 c0=black <= c1=white, index 3 is transparent.
    let mut out = [[0.; 4]; 16];
    bc::decode_bc1(&[0, 0, 255, 255, 255, 255, 255, 255], &mut out);
    assert_eq!(out, [[0.; 4]; 16]);
    // BC3 alpha endpoints 255,0; index0; red RGB565 color0, index0.
    let mut b = [0; 16];
    b[0] = 255;
    b[8..10].copy_from_slice(&0xf800u16.to_le_bytes());
    bc::decode_bc3(&b, &mut out);
    assert_eq!(out, [[1., 0., 0., 1.]; 16]);
    b[0] = 0;
    b[1] = 255;
    b[2..8].fill(255);
    bc::decode_bc4(&b[..8], false, &mut out);
    assert_eq!(out, [[1., 0., 0., 1.]; 16]);
    b[0] = 128;
    b[1] = 127;
    b[2..8].fill(0);
    b[8] = 127;
    b[9] = 128;
    b[10..].fill(0);
    bc::decode_bc5(&b, true, &mut out);
    assert_eq!(out, [[-1., 1., 0., 1.]; 16]);
    // BC7 mode6: 7-bit endpoints plus per-endpoint p-bit. All endpoint
    // components 127 with p-bit1 -> 255; zero indices select endpoint0.
    let mut bits = 1u128 << 6;
    let mut pos = 7;
    for _ in 0..8 {
        bits |= 127u128 << pos;
        pos += 7;
    }
    bits |= 3u128 << pos;
    bc::decode_bc7(&bits.to_le_bytes(), &mut out);
    assert_eq!(out, [[1.; 4]; 16]);
}
#[test]
fn astc_void_and_normal_block() {
    // Khronos ASTC §C.2.22: void-extent all-one bounds, four UNORM16 values.
    let mut b = [255; 16];
    b[0] = 0xfc;
    b[1] = 0xfd;
    for (i, v) in [0u16, 32768, 65535, 16384].into_iter().enumerate() {
        b[8 + 2 * i..10 + 2 * i].copy_from_slice(&v.to_le_bytes());
    }
    let mut out = [[0.; 4]; 144];
    astc::decode(&b, 4, 4, false, &mut out).unwrap();
    assert_eq!(out[0], [0., 32768. / 65535., 1., 16384. / 65535.]);
    // Independent bit construction from ASTC §C.2: mode0x42 -> 4x4 grid,
    // two-bit weights (32 bits). One partition, CEM8 direct RGB endpoints,
    // eight-bit endpoint ISE. Endpoint0 black, endpoint1 white. Reverse-order
    // weights: first 0, remaining 3 -> exactly 0 and 64 after unquantization.
    let mut bits = 0x42u128 | (8 << 13);
    for (i, v) in [0u128, 255, 0, 255, 0, 255].into_iter().enumerate() {
        bits |= v << (17 + i * 8);
    }
    bits |= 0x3fffffffu128 << 96;
    astc::decode(&bits.to_le_bytes(), 4, 4, false, &mut out).unwrap();
    assert_eq!(out[0], [0., 0., 0., 1.]);
    assert_eq!(out[1..16], [[1.; 4]; 15]);
}

fn u16le(d: &[u8], o: usize) -> usize {
    u16::from_le_bytes(d[o..o + 2].try_into().unwrap()) as usize
}
fn u32le(d: &[u8], o: usize) -> usize {
    u32::from_le_bytes(d[o..o + 4].try_into().unwrap()) as usize
}
fn u64le(d: &[u8], o: usize) -> usize {
    u64::from_le_bytes(d[o..o + 8].try_into().unwrap()) as usize
}
fn yaz0(d: &[u8]) -> Vec<u8> {
    assert_eq!(&d[..4], b"Yaz0");
    let size = u32::from_be_bytes(d[4..8].try_into().unwrap()) as usize;
    let mut out = Vec::with_capacity(size);
    let mut p = 16;
    while out.len() < size {
        let code = d[p];
        p += 1;
        for bit in 0..8 {
            if out.len() == size {
                break;
            }
            if code & (128 >> bit) != 0 {
                out.push(d[p]);
                p += 1;
            } else {
                let a = d[p] as usize;
                let b = d[p + 1] as usize;
                p += 2;
                let distance = ((a & 15) << 8 | b) + 1;
                let mut count = a >> 4;
                if count == 0 {
                    count = d[p] as usize + 18;
                    p += 1;
                } else {
                    count += 2;
                }
                for _ in 0..count {
                    out.push(out[out.len() - distance]);
                }
            }
        }
    }
    out
}
/// Firmware evidence: starter common.szs first BNTX, BRTI +0x1060 describes
/// BtnDirA_mono^s: 32x32 BC4 UNORM, one mip, block height0; payload +0xf000.
#[test]
fn starter_bntx_decode() {
    let data=std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"),"/../fw_dump/0100000000001012/ca6e167d711b924ba111ecb24f6a4ae0/section1_romfs/lyt/common.szs")).expect("starter firmware common.szs fixture required");
    let data = yaz0(&data);
    assert_eq!(&data[..4], b"SARC");
    let header = u16le(&data, 4);
    let start = u32le(&data, 12);
    let count = u16le(&data, header + 6);
    let bntx = (0..count)
        .find_map(|i| {
            let p = header + 12 + 16 * i;
            let d = &data[start + u32le(&data, p + 8)..start + u32le(&data, p + 12)];
            d.starts_with(b"BNTX").then_some(d)
        })
        .expect("BNTX member");
    let o = bntx.windows(4).position(|w| w == b"BRTI").unwrap();
    assert_eq!(o, 0x1060);
    assert_eq!(u32le(bntx, o + 0x1c), 0x1d01);
    assert_eq!(u16le(bntx, o + 0x12), 0);
    assert_eq!(u16le(bntx, o + 0x16), 1);
    let width = u32le(bntx, o + 0x24) as u32;
    let height = u32le(bntx, o + 0x28) as u32;
    let size = u32le(bntx, o + 0x50);
    let ptr = u64le(bntx, o + 0x70);
    let offset = u64le(bntx, ptr);
    assert_eq!(offset, 0xf000);
    let mut words = descriptor(0x27, width, height);
    words[3] = (u32le(bntx, o + 0x34) as u32 & 7) << 3;
    let tic = Tic::parse(words).unwrap();
    assert_eq!(Texture::footprint(&tic).unwrap(), (0, size as u64));
    let texture = Texture::decode(&Memory(bntx[offset..offset + size].to_vec()), &tic).unwrap();
    let pixels = &texture.levels[0].texels;
    assert_eq!(pixels.len(), 1024);
    assert!(pixels.iter().all(|p| p[0] >= 0.
        && p[0] <= 1.
        && p[1] == 0.
        && p[2] == 0.
        && p[3] == 1.));
    assert!(pixels.iter().any(|p| p[0] > 0.9));
    assert!(pixels.iter().any(|p| p[0] < 0.1));
}
