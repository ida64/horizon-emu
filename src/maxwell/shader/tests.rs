//! Firmware fixtures are required, not skipped. Stage containers are located
//! by NVN's magic and zero reserved header, as observed in starter main.elf.
use super::*;
use std::collections::{BTreeMap, HashSet};
pub(super) const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../");
pub(super) const TITLE: &str = "0100000000001012/ca6e167d711b924ba111ecb24f6a4ae0/";
pub(super) fn elf() -> Vec<u8> {
    std::fs::read(format!("{ROOT}fw_elf/{TITLE}section0_exefs/main.elf"))
        .expect("required starter ELF")
}
pub(super) fn stages(data: &[u8]) -> Vec<(usize, Program)> {
    let mut result = Vec::new();
    for offset in 0..data.len().saturating_sub(0x80) {
        if data[offset..offset + 4] != [0x78, 0x56, 0x34, 0x12]
            || data[offset + 4..offset + 0x30].iter().any(|&b| b != 0)
        {
            continue;
        }
        if !matches!(data[offset + 0x30] & 31, 1 | 2) {
            continue;
        }
        let program = Program::from_bytes(&data[offset + 0x30..])
            .unwrap_or_else(|e| panic!("stage at {offset:#x}: {e:?}"));
        result.push((offset, program));
    }
    result
}
pub(super) fn yaz0(data: &[u8]) -> Vec<u8> {
    if &data[..4] != b"Yaz0" {
        return data.to_vec();
    }
    let size = u32::from_be_bytes(data[4..8].try_into().unwrap()) as usize;
    let mut out = Vec::with_capacity(size);
    let mut p = 16;
    while out.len() < size {
        let code = data[p];
        p += 1;
        for bit in 0..8 {
            if out.len() == size {
                break;
            }
            if code & (0x80 >> bit) != 0 {
                out.push(data[p]);
                p += 1;
            } else {
                let a = data[p] as usize;
                let b = data[p + 1] as usize;
                p += 2;
                let distance = ((a & 15) << 8 | b) + 1;
                let length = if a >> 4 == 0 {
                    let n = data[p] as usize + 18;
                    p += 1;
                    n
                } else {
                    (a >> 4) + 2
                };
                for _ in 0..length {
                    out.push(out[out.len() - distance]);
                }
            }
        }
    }
    out
}
pub(super) fn members(data: &[u8]) -> Vec<&[u8]> {
    assert_eq!(&data[..4], b"SARC");
    let le = data[6..8] == [0xff, 0xfe];
    let u16at = |o| {
        let a = data[o..o + 2].try_into().unwrap();
        if le {
            u16::from_le_bytes(a)
        } else {
            u16::from_be_bytes(a)
        }
    };
    let u32at = |o| {
        let a = data[o..o + 4].try_into().unwrap();
        if le {
            u32::from_le_bytes(a)
        } else {
            u32::from_be_bytes(a)
        }
    };
    let h = u16at(4) as usize;
    assert_eq!(&data[h..h + 4], b"SFAT");
    let base = u32at(12) as usize;
    (0..u16at(h + 6) as usize)
        .map(|i| {
            let p = h + 12 + 16 * i;
            &data[base + u32at(p + 8) as usize..base + u32at(p + 12) as usize]
        })
        .collect()
}
#[test]
fn firmware_corpus_decodes() {
    let mut seen = HashSet::new();
    let mut histogram = BTreeMap::new();
    let mut add = |data: &[u8]| {
        for (offset, p) in stages(data) {
            let code = data[offset + 0x80..offset + 0x30 + p.len()].to_vec();
            if seen.insert(code) {
                for mnemonic in p.mnemonics() {
                    *histogram.entry(mnemonic).or_insert(0usize) += 1;
                }
            }
        }
    };
    add(&elf());
    for name in [
        "shader/VarietyOceanShader_Nx.arc.szs",
        "agl/agl_resource_min.Nin_Nx_NVN.release.sarc.szs",
    ] {
        let data = std::fs::read(format!("{ROOT}fw_dump/{TITLE}section1_romfs/common/{name}"))
            .expect("required shader archive");
        let data = yaz0(&data);
        for member in members(&data) {
            add(member);
        }
    }
    add(&std::fs::read(format!("{ROOT}fw_dump/{TITLE}section1_romfs/common/sead/primitive_renderer/primitive_drawer_nvn_shader.bin")).expect("required primitive shader"));
    println!("{} unique programs: {histogram:?}", seen.len());
    assert_eq!(seen.len(), 102);
}
#[test]
fn unknown_instruction_is_not_skipped() {
    let mut bytes = vec![0; 0x70];
    bytes[..4].copy_from_slice(&0x401u32.to_le_bytes());
    bytes[0x58..0x60].copy_from_slice(&u64::MAX.to_le_bytes());
    let err = Program::from_bytes(&bytes).unwrap_err();
    assert!(
        err.0
            .contains("unknown instruction 0xffffffffffffffff at 0x8")
    );
}

use super::super::sampling::*;
use std::cell::RefCell;
#[derive(Default)]
struct Textures {
    samples: RefCell<Vec<(u32, SampleRequest)>>,
}
impl TextureUnit for Textures {
    fn sample(&self, h: u32, r: &SampleRequest) -> Result<[f32; 4], Unmodeled> {
        self.samples.borrow_mut().push((h, *r));
        Ok([0.25, 0.5, 0.75, 1.0])
    }
    fn fetch(&self, _: u32, _: &FetchRequest) -> Result<[f32; 4], Unmodeled> {
        Err(Unmodeled("test unexpected fetch".into()))
    }
    fn gather(&self, _: u32, _: &GatherRequest) -> Result<[f32; 4], Unmodeled> {
        Err(Unmodeled("test unexpected gather".into()))
    }
    fn query(&self, _: u32, _: Query) -> Result<[u32; 4], Unmodeled> {
        Err(Unmodeled("test unexpected query".into()))
    }
    fn lod(&self, _: u32, _: Target, _: [f32; 3], _: [f32; 3]) -> Result<[f32; 2], Unmodeled> {
        Err(Unmodeled("test unexpected lod".into()))
    }
}
struct Memory;
impl GpuMemory for Memory {
    fn read(&self, _: u64, _: &mut [u8]) -> Option<()> {
        None
    }
    fn write(&self, _: u64, _: &[u8]) -> Option<()> {
        None
    }
}
fn environment(textures: &Textures) -> Env<'_> {
    Env {
        const_buffers: [ConstantBuffer::default(); 18],
        textures,
        texture_buffer_index: 2,
        memory: &Memory,
        y_direction: 1.0,
    }
}
/// Programs from instruction encodings.
/// The scheduling word is deliberately zero: it does not affect values.
fn assembled(words: &[u64]) -> Program {
    let mut header = [0; 0x50];
    header[..4].copy_from_slice(&0x1402u32.to_le_bytes());
    header[0x48] = 15;
    with_header(&header, words)
}
fn with_header(header: &[u8; 0x50], words: &[u64]) -> Program {
    let mut bytes = header.to_vec();
    for chunk in words.chunks(3) {
        bytes.extend_from_slice(&0u64.to_le_bytes());
        for i in 0..3 {
            bytes.extend_from_slice(
                &chunk
                    .get(i)
                    .copied()
                    .unwrap_or(0x50b0000000070f00)
                    .to_le_bytes(),
            );
        }
    }
    bytes.extend_from_slice(&[0; 32]);
    Program::from_bytes(&bytes).unwrap()
}
fn mov(rd: u64, v: u32) -> u64 {
    0x010000000007f000 | (u64::from(v) << 20) | rd
}
fn fragment(program: &Program, env: &Env<'_>, planes: &[Plane; 256]) -> Quad<'static> {
    let mut q = Quad {
        planes,
        x: 0,
        y: 0,
        covered: [true; 4],
        centroids: [[0.5; 2]; 4],
        front_facing: true,
        outputs: [FragmentOutput::default(); 4],
    };
    program.run_fragment(env, &mut q).unwrap();
    // Return only owned outputs; no test relies on retaining the input planes.
    static EMPTY: [Plane; 256] = [Plane {
        dx: 0.0,
        dy: 0.0,
        base: 0.0,
    }; 256];
    Quad {
        planes: &EMPTY,
        x: q.x,
        y: q.y,
        covered: q.covered,
        centroids: q.centroids,
        front_facing: q.front_facing,
        outputs: q.outputs,
    }
}
#[test]
fn float_modifiers_mufu_and_omap() {
    let p = assembled(&[
        mov(4, (-2.0f32).to_bits()),
        mov(5, 3f32.to_bits()),
        mov(6, 4f32.to_bits()),
        0x5c58000000070000 | (4 << 8) | (5 << 20) | (1 << 46),
        0x5c68000000070001 | (4 << 8) | (5 << 20) | (1 << 48),
        0x5980000000070002 | (4 << 8) | (5 << 20) | (6 << 39),
        0x5080000000070003 | (6 << 8) | (8 << 20),
        0xe30000000007000f,
    ]);
    let textures = Textures::default();
    let q = fragment(&p, &environment(&textures), &[Plane::default(); 256]);
    assert_eq!(q.outputs[0].colors[0], [5.0, 6.0, -2.0, 2.0]);
}
#[test]
fn carry_chain_and_bitfields() {
    let p = assembled(&[
        mov(4, u32::MAX),
        mov(5, 1),
        mov(6, 0),
        0x5c10000000070000 | (4 << 8) | (5 << 20) | (1 << 47),
        0x5c10000000070001 | (6 << 8) | (6 << 20) | (1 << 43),
        mov(7, 0x0808),
        mov(8, 0x12345678),
        0x5c00000000070002 | (8 << 8) | (7 << 20),
        0x5bf0000000070003 | (5 << 8) | (7 << 20) | (8 << 39),
        0xe30000000007000f,
    ]);
    let t = Textures::default();
    let q = fragment(&p, &environment(&t), &[Plane::default(); 256]);
    assert_eq!(
        q.outputs[0].colors[0].map(f32::to_bits),
        [0, 1, 0x56, 0x12340178]
    );
}
#[test]
fn texs_quad_derivatives_and_helpers() {
    let p = assembled(&[
        0xe003ff890ff7ff04,
        0xe003ff894ff7ff05,
        0xd832008020570400,
        0xe30000000007000f,
    ]);
    let t = Textures::default();
    let mut env = environment(&t);
    let mut handles = [0; 36];
    handles[32..36].copy_from_slice(&0x12345678u32.to_le_bytes());
    env.const_buffers[2] = ConstantBuffer::whole(&handles);
    let mut planes = [Plane::default(); 256];
    planes[0x90 / 4] = Plane {
        base: 0.0,
        dx: 0.1,
        dy: 0.0,
    };
    planes[0x94 / 4] = Plane {
        base: 0.0,
        dx: 0.0,
        dy: 0.2,
    };
    let q = fragment(&p, &env, &planes);
    assert_eq!(q.outputs[0].colors[0], [0.25, 0.5, 0.75, 1.0]);
    let samples = t.samples.borrow();
    assert_eq!(samples.len(), 4);
    assert_eq!(samples[0].0, 0x12345678);
    assert_eq!(samples[0].1.coords, [0.05, 0.1, 0.0]);
    let Lod::Derivatives { ddx, ddy, bias } = samples[0].1.lod else {
        panic!("implicit LOD")
    };
    assert!((ddx[0] - 0.1).abs() < 1e-7);
    assert!((ddy[1] - 0.2).abs() < 1e-7);
    assert_eq!(bias, 0.0);
}
/// Real ui2d stage pair: NVN containers at ELF offsets 0x4be500 (VS)
/// and 0x4beb00 (PS). Layout inferred from their disassembly: VS c3
/// 0x00..0x40 projection rows, 0x40..0x70 model rows, 0x70..0x90
/// texture transform rows, 0x210/214 pane scale, 0x218/21c pane bias,
/// 0x220 texture-generation flag, 0x22c feature bits (2 keeps incoming UV).
/// PS c3 0x00..0x10 RGBA multiplier and 0x10..0x20 RGBA bias.
/// Texture handle is c[texture_buffer_index][0x20]. These offsets are
/// observations of starter main.elf, not a general NVN uniform ABI.
#[test]
fn real_ui2d_vertex_and_fragment() {
    let bytes = elf();
    let vs = Program::from_bytes(&bytes[0x4be530..]).unwrap();
    let fs = Program::from_bytes(&bytes[0x4beb30..]).unwrap();
    let t = Textures::default();
    let mut env = environment(&t);
    let mut cb = vec![0u8; 0x230];
    for offset in [
        0, 0x14, 0x28, 0x3c, 0x40, 0x54, 0x68, 0x70, 0x84, 0x210, 0x214,
    ] {
        cb[offset..offset + 4].copy_from_slice(&1f32.to_le_bytes());
    }
    cb[0x22c..0x230].copy_from_slice(&2u32.to_le_bytes());
    env.const_buffers[3] = ConstantBuffer::whole(&cb);
    let mut input = Attributes::default();
    input.0[0x80 / 4] = 0.25f32.to_bits();
    input.0[0x84 / 4] = 0.5f32.to_bits();
    let mut output = Attributes::default();
    vs.run_vertex(&env, &input, 0, &mut output).unwrap();
    assert_eq!(f32::from_bits(output.0[0x7c / 4]), 1.0);
    let color = [1f32.to_le_bytes(); 4].concat();
    env.const_buffers[3] = ConstantBuffer::whole(&color);
    let mut planes = [Plane::default(); 256];
    for (i, plane) in planes.iter_mut().enumerate() {
        plane.base = f32::from_bits(output.0[i]);
    }
    planes[0x7c / 4].base = 1.0;
    let q = fragment(&fs, &env, &planes);
    assert_eq!(q.outputs[0].colors[0], [0.25, 0.5, 0.75, 1.0]);
    assert_eq!(t.samples.borrow().len(), 4);
}

fn pc(index: usize) -> i32 {
    ((index / 3) * 32 + (index % 3 + 1) * 8) as i32
}
fn branch(base: u64, from: usize, to: usize) -> u64 {
    base | ((u64::from((pc(to) - pc(from) - 8) as u32) & 0xffffff) << 20)
}
#[test]
fn reconvergence_break_and_indirect_branch() {
    let words = vec![
        0xe003ff870ff7ff04,
        mov(5, 1f32.to_bits()),
        0x5bb1000000070007 | (4 << 8) | (5 << 20) | (7 << 39),
        branch(0xe290000000070000, 3, 8),
        branch(0xe24000000000000f, 4, 7),
        mov(0, 2f32.to_bits()),
        0xf0f800000007000f,
        mov(0, 3f32.to_bits()),
        branch(0xe2a0000000070000, 8, 11),
        0xe34000000007000f,
        mov(0, 99f32.to_bits()),
        mov(6, 0),
        branch(0xe25000000007000f | (6 << 8), 12, 14),
        mov(0, 99f32.to_bits()),
        0xe30000000007000f,
    ];
    let p = assembled(&words);
    let t = Textures::default();
    let mut planes = [Plane::default(); 256];
    planes[0x70 / 4] = Plane {
        base: 0.0,
        dx: 1.0,
        dy: 0.0,
    };
    let q = fragment(&p, &environment(&t), &planes);
    assert_eq!(q.outputs.map(|o| o.colors[0][0]), [3.0, 2.0, 3.0, 2.0]);
}
#[test]
fn predicate_kill() {
    let p = assembled(&[
        mov(4, 1f32.to_bits()),
        mov(5, 2f32.to_bits()),
        0x5bb1000000070007 | (4 << 8) | (5 << 20) | (7 << 39),
        0xe33000000000000f,
        mov(0, 7f32.to_bits()),
        0xe30000000007000f,
    ]);
    let t = Textures::default();
    let q = fragment(&p, &environment(&t), &[Plane::default(); 256]);
    assert!(q.outputs.iter().all(|o| o.discarded));
}
#[test]
fn conversions_round_and_saturate() {
    let p = assembled(&[
        mov(4, 2.5f32.to_bits()),
        mov(5, (-3.8f32).to_bits()),
        mov(6, u32::MAX),
        0x5cb0000000071a00 | (4 << 20),
        0x5cb0000000071a01 | (5 << 20) | (3 << 39),
        0x5cb8000000070a02 | (6 << 20),
        0x5cb0000000070a03 | (5 << 20),
        0xe30000000007000f,
    ]);
    let t = Textures::default();
    let q = fragment(&p, &environment(&t), &[Plane::default(); 256]);
    assert_eq!(
        q.outputs[0].colors[0].map(f32::to_bits),
        [2, (-3i32) as u32, (u32::MAX as f32).to_bits(), 0]
    );
}

#[test]
fn snapshot_extents_track_static_dynamic_and_bound_handles() {
    let bytes = elf();
    let vs = Program::from_bytes(&bytes[0x4be530..]).unwrap();
    let fs = Program::from_bytes(&bytes[0x4beb30..]).unwrap();
    assert_eq!(vs.constant_buffer_extent(0), Some(0));
    assert_eq!(vs.constant_buffer_extent(1), Some(4));
    assert_eq!(vs.constant_buffer_extent(3), None);
    assert_eq!(fs.constant_buffer_extent(3), Some(0x20));
    assert_eq!(fs.texture_handle_extent(), 0x24);
    assert_eq!(vs.texture_handle_extent(), 0);
    let p = assembled(&[0xef9500300e07ff00, 0xe30000000007000f]);
    assert_eq!(p.constant_buffer_extent(3), Some(0xe8));
}

#[test]
fn multiply_add_half_modes_and_lut() {
    let p = assembled(&[
        mov(4, 0x12340003),
        mov(5, 0x56780005),
        mov(6, 7),
        0x5b00000000070000 | (4 << 8) | (5 << 20) | (6 << 39),
        0x5b00000000070001 | (4 << 8) | (5 << 20) | (6 << 39) | (1 << 37),
        0x5cc0000000070002 | (4 << 8) | (5 << 20) | (6 << 39) | (1 << 35) | (1 << 33),
        0x5be0000000070003 | (4 << 8) | (5 << 20) | (6 << 39) | (0x96 << 28),
        0xe30000000007000f,
    ]);
    let t = Textures::default();
    let q = fragment(&p, &environment(&t), &[Plane::default(); 256]);
    assert_eq!(
        q.outputs[0].colors[0].map(f32::to_bits),
        [22, 0x00050016, 15, 0x12340003 ^ 0x56780005 ^ 7]
    );
}

#[test]
fn ipa_multiply_sat_and_helper_output() {
    let p = assembled(&[
        mov(9, 2f32.to_bits()),
        0xe043ff880097ff00,
        0xe04bff884097ff01,
        0xe30000000007000f,
    ]);
    let t = Textures::default();
    let mut planes = [Plane::default(); 256];
    planes[0x80 / 4] = Plane {
        base: 0.25,
        dx: 0.5,
        dy: 0.0,
    };
    planes[0x84 / 4] = Plane {
        base: 0.75,
        dx: 0.0,
        dy: 0.0,
    };
    let mut q = Quad {
        planes: &planes,
        x: 0,
        y: 0,
        covered: [true, false, true, false],
        centroids: [[0.5; 2]; 4],
        front_facing: true,
        outputs: [FragmentOutput::default(); 4],
    };
    p.run_fragment(&environment(&t), &mut q).unwrap();
    assert_eq!(q.outputs[0].colors[0], [1.0, 1.0, 0.0, 0.0]);
    assert_eq!(q.outputs[1], FragmentOutput::default());
}

#[test]
fn counted_loop_breaks_at_predicate() {
    let p = assembled(&[
        mov(4, 0),
        branch(0xe2a0000000070000, 1, 7),
        0x1c00000000170404,
        0x366c038000370407,
        0xe34000000000000f,
        branch(0xe24000000007000f, 5, 2),
        mov(4, 999),
        0x5c98078000470000,
        0xe30000000007000f,
    ]);
    let t = Textures::default();
    let q = fragment(&p, &environment(&t), &[Plane::default(); 256]);
    assert_eq!(q.outputs[0].colors[0][0].to_bits(), 3);
}

#[test]
fn unordered_predicate_combining_and_sparse_omap_depth() {
    // FSETP.NAN.AND P0,PT: true; FSETP.LT.XOR P1,P0: false XOR true.
    // Conditional MOVs expose both predicates in separate render targets.
    let mut p = assembled(&[
        mov(4, f32::NAN.to_bits()),
        mov(5, 1f32.to_bits()),
        0x5bb8038000570407,
        0x5bb140000057040f,
        mov(1, 7f32.to_bits()) & !(15 << 16),
        (mov(6, 9f32.to_bits()) & !(15 << 16)) | (1 << 16),
        mov(9, 0.5f32.to_bits()),
        0xe30000000007000f,
    ]);
    let mut sph = *p.header.bytes();
    sph[0x48] = 0x02; // target 0 G at R1, still reserves R0..R3.
    sph[0x49] = 0x04; // target 2 B at R6, reserves R4..R7.
    sph[0x4c] = 2; // depth at R9 (R8 is the sample-mask slot).
    p.header = Header::parse(&sph).unwrap();
    let t = Textures::default();
    let q = fragment(&p, &environment(&t), &[Plane::default(); 256]);
    assert_eq!(q.outputs[0].colors[0], [0.0, 7.0, 0.0, 0.0]);
    assert_eq!(q.outputs[0].colors[2], [0.0, 0.0, 9.0, 0.0]);
    assert_eq!(q.outputs[0].depth, Some(0.5));
}
/// A geometry shader header: VTG, shader type 4, one thread per
/// primitive, TRIANGLESTRIP output of at most `max` vertices.
fn geometry_header(max: u8) -> [u8; 0x50] {
    let mut header = [0; 0x50];
    header[..4].copy_from_slice(&0x1001u32.to_le_bytes());
    header[0x0b] = 1;
    header[0x0f] = 7;
    header[0x10] = max;
    header
}
/// `ALD a[offset]` / `AST a[offset]` of one word with vertex register
/// `vertex` (encodings as the NVN compiler emits them; see the ocean
/// archive's geometry shaders).
fn ald(rd: u64, offset: u64, vertex: u64) -> u64 {
    0xefd8_0000_0007_ff00 | (offset << 20) | (vertex << 39) | rd
}
fn ast(rs: u64, offset: u64, vertex: u64) -> u64 {
    0xeff0_0000_0007_ff00 | (offset << 20) | (vertex << 39) | rs
}
/// `OUT` (register form, stream RZ): `kind` 1 EMIT, 2 CUT, 3 both.
fn out(rd: u64, ra: u64, kind: u64) -> u64 {
    0xfbe0_0000_0ff7_0000 | (kind << 39) | (ra << 8) | rd
}
/// The input-handle and output-handle protocol of geometry shaders:
/// `SR_INVOCATION_INFO` carries the vertices per primitive, `ISBERD`
/// turns an ISBE entry into the handle `ALD` takes, `OUT` returns the
/// next output handle, emits past `MaxOutputVertexCount` are dropped, and
/// a handle the program was not given is refused.
#[test]
fn geometry_handles_emit_and_cut() {
    const S2R_INVOCATION_INFO: u64 = 0xf0c8_0000_01d7_0000;
    const ISBERD: u64 = 0xefd0_0000_0007_0000;
    let p = with_header(
        &geometry_header(2),
        &[
            S2R_INVOCATION_INFO,
            mov(2, 2),
            ISBERD | (2 << 8) | 1,
            ald(4, 0x80, 1),
            ast(4, 0x80, 5),
            ast(0, 0x84, 5),
            out(5, 5, 1),
            ISBERD | (0xff << 8) | 1,
            ald(4, 0x80, 1),
            ast(4, 0x80, 5),
            out(5, 5, 3),
            out(5, 5, 1),
            ast(5, 0x88, 5),
            0xe30000000007000f,
        ],
    );
    let inputs: Vec<Attributes> = (0..3)
        .map(|i| {
            let mut a = Attributes::default();
            a.0[0x80 / 4] = 10 + i;
            a
        })
        .collect();
    let refs: Vec<&Attributes> = inputs.iter().collect();
    let t = Textures::default();
    let env = environment(&t);
    let mut emitted = Vec::new();
    p.run_geometry(&env, &refs, &mut emitted).unwrap();
    assert_eq!(emitted.len(), 3);
    let first = emitted[0].as_ref().unwrap();
    assert_eq!(first.0[0x80 / 4], 12);
    assert_eq!(first.0[0x84 / 4], 3 << 16);
    let second = emitted[1].as_ref().unwrap();
    assert_eq!(second.0[0x80 / 4], 10);
    // Written before the emit only: an emitted vertex starts the next.
    assert_eq!(second.0[0x84 / 4], 0);
    assert!(emitted[2].is_none());

    // A stale output handle.
    let p = with_header(
        &geometry_header(4),
        &[out(5, 5, 1), ast(5, 0x80, 6), 0xe30000000007000f],
    );
    let err = p.run_geometry(&env, &refs, &mut Vec::new()).unwrap_err();
    assert!(err.0.contains("output vertex handle 0x0"), "{}", err.0);
    // An ISBE entry past the primitive.
    let p = with_header(
        &geometry_header(4),
        &[mov(2, 3), ISBERD | (2 << 8) | 1, 0xe30000000007000f],
    );
    let err = p.run_geometry(&env, &refs, &mut Vec::new()).unwrap_err();
    assert!(err.0.contains("ISBE entry 3"), "{}", err.0);
}

/// An executed `ALD` of an inactive vertex attribute refuses; one the
/// program does not execute (here predicated on `!PT`), or of another
/// attribute, reads the inputs.
#[test]
fn vertex_reads_of_inactive_attributes_refuse_when_executed() {
    let mut header = [0; 0x50];
    header[..4].copy_from_slice(&0x0401u32.to_le_bytes());
    const EXIT: u64 = 0xe30000000007000f;
    const NEVER: u64 = 8 << 16;
    let mut input = Attributes::default();
    input.0[0xa0 / 4] = 42;
    input.0[0x9c / 4] = 7;
    let t = Textures::default();
    let env = environment(&t);
    let run = |words: &[u64], inactive: u32| {
        let mut output = Attributes::default();
        with_header(&header, words)
            .run_vertex(&env, &input, inactive, &mut output)
            .map(|()| output)
    };
    let store = ast(0, 0x80, 0xff);
    let err = run(&[ald(0, 0xa0, 0xff), store, EXIT], 1 << 2).unwrap_err();
    assert!(
        err.0.contains("vertex attribute 2, which is inactive"),
        "{}",
        err.0
    );
    // Two words from 0x9c: the second is attribute 2's first.
    let pair = ald(0, 0x9c, 0xff) | (1 << 47);
    assert!(run(&[pair, store, EXIT], 1 << 2).is_err());
    let output = run(&[ald(0, 0xa0, 0xff), store, EXIT], 1 << 3).unwrap();
    assert_eq!(output.0[0x80 / 4], 42);
    let output = run(&[ald(0, 0x9c, 0xff), store, EXIT], 1 << 2).unwrap();
    assert_eq!(output.0[0x80 / 4], 7);
    run(&[ald(0, 0xa0, 0xff) | NEVER, EXIT], 1 << 2).unwrap();
}
