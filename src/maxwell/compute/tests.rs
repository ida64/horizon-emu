//! QMD decoding, and kernels launched through the compute class and run
//! by the interpreter. QMD bit positions:
//! [public: NVIDIA open-gpu-doc `classes/compute/clb1c0qmd.h`].

use super::super::tests::Ram;
use super::super::{Channel, Engines, RendererKind};
use super::*;

/// Sets `MW(high:low)` of a QMD.
fn set(words: &mut [u32; QMD_SIZE / 4], high: u32, low: u32, value: u32) {
    for bit in low..=high {
        let word = &mut words[(bit / 32) as usize];
        let mask = 1 << (bit % 32);
        if value >> (bit - low) & 1 != 0 {
            *word |= mask;
        } else {
            *word &= !mask;
        }
    }
}

fn bytes(words: &[u32; QMD_SIZE / 4]) -> [u8; QMD_SIZE] {
    let mut out = [0; QMD_SIZE];
    for (i, word) in words.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    out
}

/// A version 01_07 QMD: `grid` blocks of `block` threads, `shared` bytes
/// of shared memory, one barrier, constant buffer 0 at `cbuf0` (0x100
/// bytes), a one-word release of `payload` at `release`.
fn qmd(
    offset: u32,
    grid: [u32; 3],
    block: [u32; 3],
    shared: u32,
    cbuf0: u64,
    release: Option<(u64, u32)>,
) -> [u32; QMD_SIZE / 4] {
    let mut w = [0; QMD_SIZE / 4];
    set(&mut w, 583, 580, 1);
    set(&mut w, 579, 576, 7);
    set(&mut w, 287, 256, offset);
    set(&mut w, 415, 384, grid[0]);
    set(&mut w, 431, 416, grid[1]);
    set(&mut w, 447, 432, grid[2]);
    set(&mut w, 607, 592, block[0]);
    set(&mut w, 623, 608, block[1]);
    set(&mut w, 639, 624, block[2]);
    set(&mut w, 561, 544, shared);
    set(&mut w, 1471, 1467, 1);
    set(&mut w, 640, 640, 1);
    set(&mut w, 959, 928, cbuf0 as u32);
    set(&mut w, 967, 960, (cbuf0 >> 32) as u32);
    set(&mut w, 991, 975, 0x100);
    if let Some((address, payload)) = release {
        set(&mut w, 202, 202, 1);
        set(&mut w, 767, 736, address as u32);
        set(&mut w, 775, 768, (address >> 32) as u32);
        set(&mut w, 799, 799, 1);
        set(&mut w, 831, 800, payload);
    }
    w
}

#[test]
fn qmd_fields_decode() {
    let mut w = qmd(
        0x1230,
        [7, 3, 2],
        [32, 4, 1],
        0x400,
        0x12_3456_7800,
        Some((0xAB_CDEF_0010, 0x55)),
    );
    // Constant buffer 5, the sampler from the header, release 1 with the
    // four-word structure.
    set(&mut w, 645, 645, 1);
    set(&mut w, 959 + 5 * 64, 928 + 5 * 64, 0x9000);
    set(&mut w, 991 + 5 * 64, 975 + 5 * 64, 0x10000);
    set(&mut w, 382, 382, 1);
    set(&mut w, 203, 203, 1);
    set(&mut w, 863, 832, 0x2000);
    set(&mut w, 927, 896, 9);
    let q = Qmd::parse(&bytes(&w)).unwrap();
    assert_eq!(q.program_offset, 0x1230);
    assert_eq!(
        q.launch,
        Launch {
            grid: [7, 3, 2],
            block: [32, 4, 1],
            shared: 0x400,
            barriers: 1,
        }
    );
    assert_eq!(q.constant_buffers[0], Some((0x12_3456_7800, 0x100)));
    assert_eq!(q.constant_buffers[5], Some((0x9000, 0x10000)));
    assert!(q.constant_buffers[1].is_none());
    assert!(q.sampler_from_header);
    assert_eq!(
        q.releases,
        [
            Some(Release {
                address: 0xAB_CDEF_0010,
                payload: 0x55,
                four_words: false,
            }),
            Some(Release {
                address: 0x2000,
                payload: 9,
                four_words: true,
            }),
        ]
    );
}

#[test]
fn qmd_states_not_modeled_refuse() {
    let base = qmd(0, [1; 3], [1; 3], 0, 0, None);
    for (name, bit) in [
        ("IS_QUEUE", 200),
        ("DEPENDENT_QMD_SCHEDULE_ENABLE", 205),
        ("FP32_F2I_NAN_BEHAVIOR", 377),
        ("RELEASE0_REDUCTION_ENABLE", 794),
        ("CTA_RASTER_WIDTH_RESUME", 448),
    ] {
        let mut w = base;
        set(&mut w, bit, bit, 1);
        let error = Qmd::parse(&bytes(&w)).unwrap_err();
        assert!(error.0.contains(name), "{name}: {error:?}");
    }
    let mut w = base;
    set(&mut w, 583, 580, 2);
    set(&mut w, 579, 576, 1);
    assert!(Qmd::parse(&bytes(&w)).unwrap_err().0.contains("02_01"));
}

// Instructions, guard PT unless `on` is used.
const PT: u64 = 7 << 16;
const EXIT: u64 = 0xE300_0000_0007_000F;
const BAR_SYNC: u64 = 0xF0A8_1B80_0007_0000;
const RZ: u64 = 255;
const TID_X: u64 = 33;
const CTAID_X: u64 = 37;

fn s2r(rd: u64, sr: u64) -> u64 {
    0xF0C8_0000_0000_0000 | sr << 20 | PT | rd
}
/// `rd = (ra << shift) + rb`.
fn iscadd(rd: u64, ra: u64, rb: u64, shift: u64) -> u64 {
    0x5C18_0000_0000_0000 | shift << 39 | rb << 20 | PT | ra << 8 | rd
}
/// `rd = (ra << shift) + c[0][offset]`.
fn iscadd_c0(rd: u64, ra: u64, offset: u64, shift: u64) -> u64 {
    0x4C18_0000_0000_0000 | shift << 39 | (offset / 4) << 20 | PT | ra << 8 | rd
}
fn iadd(rd: u64, ra: u64, rb: u64) -> u64 {
    0x5C10_0000_0000_0000 | rb << 20 | PT | ra << 8 | rd
}
/// `LDG.32 rd, [ra]` (32-bit address).
fn ldg(rd: u64, ra: u64) -> u64 {
    0xEED4_0000_0000_0000 | PT | ra << 8 | rd
}
fn stg(ra: u64, rs: u64) -> u64 {
    0xEEDC_0000_0000_0000 | PT | ra << 8 | rs
}
fn lds(rd: u64, ra: u64, offset: u64) -> u64 {
    0xEF4C_0000_0000_0000 | offset << 20 | PT | ra << 8 | rd
}
fn sts(ra: u64, offset: u64, rs: u64) -> u64 {
    0xEF5C_0000_0000_0000 | offset << 20 | PT | ra << 8 | rs
}
/// `ISETP.<cmp>.U32.AND pd, PT, ra, imm, PT` (cmp 1 LT, 2 EQ, 6 GE).
fn isetp(pd: u64, ra: u64, imm: u64, cmp: u64) -> u64 {
    0x3660_0000_0000_0000 | cmp << 49 | 7 << 39 | imm << 20 | PT | ra << 8 | pd << 3 | 7
}
/// `inst` guarded by `@Pp`.
fn on(p: u64, inst: u64) -> u64 {
    inst & !(0xF << 16) | p << 16
}

/// Code in groups of a (zero) scheduling word and three instructions,
/// padded with NOPs and closed by an all-zero group.
fn code(words: &[u64]) -> Vec<u8> {
    let mut out = Vec::new();
    for chunk in words.chunks(3) {
        out.extend_from_slice(&0u64.to_le_bytes());
        for i in 0..3 {
            let word = chunk.get(i).copied().unwrap_or(0x50B0_0000_0007_0F00);
            out.extend_from_slice(&word.to_le_bytes());
        }
    }
    out.extend_from_slice(&[0; 32]);
    out
}

const REGION: u64 = 0x800;
const CODE: u64 = 0x1000;
const QMD: u64 = 0x3000;
const CBUF: u64 = 0x3200;

fn words(ram: &Ram, address: u64, count: usize) -> Vec<u32> {
    (0..count)
        .map(|i| ram.word(address + i as u64 * 4))
        .collect()
}

/// Launches `kernel` with `qmd` through the compute class, as the
/// driver does: program region, QMD address, signaling PCAS with
/// SCHEDULE.
fn launch(ram: &Ram, kernel: &[u64], qmd: &[u32; QMD_SIZE / 4]) -> Result<(), Unmodeled> {
    ram.put(CODE, &code(kernel));
    ram.put(QMD, &bytes(qmd));
    let mut channel = Channel::with_renderer(RendererKind::Software);
    let mut call = |method: u32, argument: u32| channel.call(1, method / 4, argument, ram);
    call(0, 0xB1C0)?;
    call(0x1608, (REGION >> 32) as u32)?;
    call(0x160c, REGION as u32)?;
    call(0x2b4, (QMD >> 8) as u32)?;
    call(0x2bc, 3)
}

#[test]
fn vector_add_writes_global_memory_and_releases() {
    let ram = Ram::new(0x8000);
    let (a, b, c) = (0x4000u32, 0x5000u32, 0x6000u32);
    for i in 0..64u32 {
        ram.put(u64::from(a + i * 4), &(i * 3).to_le_bytes());
        ram.put(u64::from(b + i * 4), &(1000 + i).to_le_bytes());
    }
    for (i, base) in [a, b, c].into_iter().enumerate() {
        ram.put(CBUF + i as u64 * 4, &base.to_le_bytes());
    }
    let kernel = [
        s2r(0, TID_X),
        s2r(1, CTAID_X),
        iscadd(0, 1, 0, 5),
        iscadd_c0(2, 0, 0, 2),
        iscadd_c0(4, 0, 4, 2),
        iscadd_c0(6, 0, 8, 2),
        ldg(8, 2),
        ldg(9, 4),
        iadd(8, 8, 9),
        stg(6, 8),
        EXIT,
    ];
    let q = qmd(
        (CODE - REGION) as u32,
        [2, 1, 1],
        [32, 1, 1],
        0,
        CBUF,
        Some((0x7000, 0xC0FFEE)),
    );
    launch(&ram, &kernel, &q).unwrap();
    let expected: Vec<u32> = (0..64).map(|i| i * 3 + 1000 + i).collect();
    assert_eq!(words(&ram, u64::from(c), 64), expected);
    assert_eq!(ram.word(0x7000), 0xC0FFEE);
}

/// Sums 64 inputs per block in shared memory, halving the active threads
/// each step with a barrier between steps, across the block's two warps.
fn reduction(first_barrier: bool) -> Vec<u64> {
    let mut kernel = vec![
        s2r(0, TID_X),
        s2r(2, CTAID_X),
        iscadd(3, 2, 0, 6),
        iscadd_c0(4, 3, 0, 2),
        ldg(5, 4),
        iscadd(1, 0, RZ, 2),
        sts(1, 0, 5),
    ];
    if first_barrier {
        kernel.push(BAR_SYNC);
    }
    for s in [32u64, 16, 8, 4, 2, 1] {
        kernel.extend([
            isetp(0, 0, s, 1),
            on(0, lds(6, 1, s * 4)),
            on(0, lds(7, 1, 0)),
            on(0, iadd(7, 7, 6)),
            on(0, sts(1, 0, 7)),
            BAR_SYNC,
        ]);
    }
    kernel.extend([
        isetp(1, 0, 0, 2),
        on(1, lds(7, 1, 0)),
        on(1, iscadd_c0(8, 2, 4, 2)),
        on(1, stg(8, 7)),
        EXIT,
    ]);
    kernel
}

fn reduction_inputs(ram: &Ram) {
    for i in 0..128u32 {
        ram.put(0x4000 + u64::from(i) * 4, &(i * i + 7).to_le_bytes());
    }
    ram.put(CBUF, &0x4000u32.to_le_bytes());
    ram.put(CBUF + 4, &0x5000u32.to_le_bytes());
}

#[test]
fn shared_memory_reduction_with_barriers() {
    let ram = Ram::new(0x8000);
    reduction_inputs(&ram);
    let q = qmd(
        (CODE - REGION) as u32,
        [2, 1, 1],
        [64, 1, 1],
        0x100,
        CBUF,
        None,
    );
    launch(&ram, &reduction(true), &q).unwrap();
    let sum = |range: std::ops::Range<u32>| range.map(|i| i * i + 7).sum::<u32>();
    assert_eq!(words(&ram, 0x5000, 2), [sum(0..64), sum(64..128)]);
}

#[test]
fn shared_memory_read_before_any_write_refuses() {
    // Without the first barrier, warp 0 reads what warp 1 has not stored
    // yet: on hardware the outcome depends on scheduling.
    let ram = Ram::new(0x8000);
    reduction_inputs(&ram);
    let q = qmd(
        (CODE - REGION) as u32,
        [1, 1, 1],
        [64, 1, 1],
        0x100,
        CBUF,
        None,
    );
    let error = launch(&ram, &reduction(false), &q).unwrap_err();
    assert!(
        error.0.contains("no thread of the block wrote"),
        "{error:?}"
    );
}

/// Threads that exited do not hold back a barrier the others reach.
#[test]
fn barrier_releases_past_exited_threads() {
    let ram = Ram::new(0x8000);
    let kernel = [
        s2r(0, TID_X),
        isetp(0, 0, 32, 6),
        on(0, EXIT),
        BAR_SYNC,
        EXIT,
    ];
    let q = qmd((CODE - REGION) as u32, [1, 1, 1], [40, 1, 1], 0, CBUF, None);
    launch(&ram, &kernel, &q).unwrap();
}

#[test]
fn thread_ids_of_a_three_dimensional_block() {
    // Each thread stores SR_TID at out[linear thread index]: x + 3 y + 6 z
    // for a 3x2x2 block, the layout warps are filled in.
    let ram = Ram::new(0x8000);
    ram.put(CBUF, &0x4000u32.to_le_bytes());
    let kernel = [
        s2r(0, 32),
        s2r(1, 34),
        s2r(2, 35),
        s2r(3, TID_X),
        // linear = x + 3 y + 6 z = x + (y << 1) + y + (z << 2) + (z << 1)
        iscadd(4, 1, 3, 1),
        iadd(4, 4, 1),
        iscadd(4, 2, 4, 2),
        iscadd(4, 2, 4, 1),
        iscadd_c0(5, 4, 0, 2),
        stg(5, 0),
        EXIT,
    ];
    let q = qmd((CODE - REGION) as u32, [1, 1, 1], [3, 2, 2], 0, CBUF, None);
    launch(&ram, &kernel, &q).unwrap();
    let expected: Vec<u32> = (0..12)
        .map(|t| (t % 3) | (t / 3 % 2) << 16 | (t / 6) << 26)
        .collect();
    assert_eq!(words(&ram, 0x4000, 12), expected);
}
