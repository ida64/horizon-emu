use super::*;
use crate::{
    boot::DramSize,
    bus::MmioDevice,
    memory::GuestRam,
    soc::{
        host1x_channel::Channel,
        mc::{Mc, SMMU_HOST1X_ASID, SMMU_VIC_ASID},
    },
};

const RAM_BASE: u64 = 0x8000_0000;
/// Device address 0 maps here (one 4 MiB SMMU section).
const WINDOW: u64 = RAM_BASE + 0x40_0000;
const CONFIG: u64 = 0x1000;
const OUTPUT: u64 = 0x1_0000;
const INPUT: u64 = 0x8_0000;
const SYNCPOINT: usize = 12;

struct Bench {
    ram: Arc<GuestRam>,
    mc: Arc<Mc>,
    syncpoints: Arc<Syncpoints>,
    notes: Arc<Notes>,
    vic: Arc<Vic>,
    config: [u64; CONFIG_SIZE / 8],
}

impl Bench {
    fn new() -> Self {
        let ram = Arc::new(GuestRam::new(RAM_BASE, 0x80_0000).unwrap());
        let mc = Arc::new(Mc::new(DramSize::Gb4));
        mc.write(0, 0x1c, 4, 1).unwrap();
        mc.write(0, 0x20, 4, 0xe008_0000).unwrap();
        for client in [SMMU_VIC_ASID, SMMU_HOST1X_ASID] {
            mc.write(0, client, 4, 0x8000_0001).unwrap();
        }
        // Directory entry 0: a section (no next-level bit) at WINDOW.
        ram.write(RAM_BASE, &0xc008_0400u32.to_le_bytes()).unwrap();
        let syncpoints = Arc::new(Syncpoints::default());
        let notes: Arc<Notes> = Arc::default();
        let vic = Arc::new(Vic::new(
            DeviceMemory::new(&mc, &ram, SMMU_VIC_ASID),
            Arc::clone(&syncpoints),
            Arc::clone(&notes),
        ));
        Bench {
            ram,
            mc,
            syncpoints,
            notes,
            vic,
            config: [0; CONFIG_SIZE / 8],
        }
    }

    fn poke(&self, address: u64, bytes: &[u8]) {
        self.ram.write(WINDOW + address, bytes).unwrap();
    }

    fn peek(&self, address: u64) -> [u8; 4] {
        let mut bytes = [0; 4];
        self.ram.read(WINDOW + address, &mut bytes).unwrap();
        bytes
    }

    fn set(&mut self, offset: usize, value: u64) {
        self.config[offset / 8] = value;
    }

    /// Output surface: `format`, pitch-linear (`block` None) or block-linear
    /// with 2^`block` GOB-high blocks, `width`x`height`, and the target
    /// rectangle (inclusive).
    fn output(&mut self, format: u64, block: Option<u64>, size: (u64, u64), target: [u64; 4]) {
        self.set(0x18, rect(target));
        self.set(0x20, surface_word(format, block, size));
        self.set(0x28, luma_word(size));
    }

    /// Slot `slot`: a 1:1 copy of `source` (inclusive) to `destination`
    /// (inclusive) from a surface of `size`, opaque.
    fn slot(
        &mut self,
        slot: usize,
        format: u64,
        block: Option<u64>,
        size: (u64, u64),
        source: [u64; 4],
        destination: [u64; 4],
    ) {
        let base = SLOT_BASE + slot * SLOT_SIZE;
        self.set(base, 1 | 1 << 8);
        // Soft clamp 0..1023, planar alpha 1023, constant alpha.
        self.set(base + 0x10, 1023 << 10 | 1023 << 32 | 1 << 42);
        let [left, top, right, bottom] = source;
        self.set(base + 0x20, left << 16 | (right << 16) << 32);
        self.set(base + 0x28, top << 16 | (bottom << 16) << 32);
        self.set(base + 0x30, rect(destination));
        self.set(base + 0x40, surface_word(format, block, size));
        self.set(base + 0x48, luma_word(size));
        // Source colour × K1 (1.0) + destination × 0.
        self.set(base + 0xA0, 1023 | 5 << 36 | 2 << 44);
    }

    /// Sends the methods nvnflinger sends, directly to the THI.
    fn execute(&self, input: Option<u64>) -> Option<()> {
        self.poke(CONFIG, &self.config.map(u64::to_le_bytes).concat());
        for (method, value) in methods(input) {
            self.vic.thi_write(THI_METHOD_OFFSET, method >> 2)?;
            self.vic.thi_write(THI_METHOD_DATA, value)?;
        }
        Some(())
    }
}

/// nvnflinger's method sequence (`0xAE1A0`), one input in slot 0.
fn methods(input: Option<u64>) -> Vec<(u32, u32)> {
    let mut methods = vec![
        (0x200, 1),
        (0x704, 0x610010),
        (0x708, (CONFIG >> 8) as u32),
        (0x730, 0),
        (0x720, (OUTPUT >> 8) as u32),
    ];
    methods.extend(input.map(|address| (0x400, (address >> 8) as u32)));
    methods.push((0x300, 0x100));
    methods
}

fn rect([left, top, right, bottom]: [u64; 4]) -> u64 {
    left | right << 16 | top << 32 | bottom << 48
}

fn surface_word(format: u64, block: Option<u64>, (width, height): (u64, u64)) -> u64 {
    let kind = block.map_or(0, |log2| 1 << 11 | log2 << 15);
    format | kind | (width - 1) << 32 | (height - 1) << 46
}

fn luma_word((width, height): (u64, u64)) -> u64 {
    (width - 1) | (height - 1) << 14 | 16383 << 32 | 16383 << 46
}

#[test]
fn a_configuration_without_slots_fills_the_target_rectangle_with_the_background() {
    let mut bench = Bench::new();
    // 4x2 A8R8G8B8 pitch-linear, target the middle two columns.
    bench.output(32, None, (4, 2), [1, 0, 2, 1]);
    // Alpha fill "background", background (R, G, B, A) = (1023, 0, 1023, 512).
    bench.set(0x10, 1 | 512 << 6 | 1023 << 16 | 1023 << 36);
    bench.poke(OUTPUT, &[0x11; 32]);
    bench.execute(None).unwrap();
    for y in 0..2 {
        assert_eq!(bench.peek(OUTPUT + y * 16), [0x11; 4]);
        // B, G, R, A in memory.
        assert_eq!(bench.peek(OUTPUT + y * 16 + 4), [0xff, 0, 0xff, 0x80]);
        assert_eq!(bench.peek(OUTPUT + y * 16 + 8), [0xff, 0, 0xff, 0x80]);
        assert_eq!(bench.peek(OUTPUT + y * 16 + 12), [0x11; 4]);
    }
}

#[test]
fn an_opaque_slot_is_copied_from_pitch_linear_into_block_linear_with_components_reordered() {
    let mut bench = Bench::new();
    // 16x16 output, one-GOB-high blocks; input 8x4 A8B8G8R8 pitch-linear.
    bench.output(32, Some(0), (16, 16), [0, 0, 15, 15]);
    bench.slot(0, 31, None, (8, 4), [2, 1, 5, 2], [10, 13, 13, 14]);
    let pixels: Vec<u8> = (0..32u8)
        .flat_map(|i| [i, 0x40 + i, 0x80 + i, 0x7f])
        .collect();
    bench.poke(INPUT, &pixels);
    bench.execute(Some(INPUT)).unwrap();
    let gob_high = BlockLinear {
        block_height_log2: 0,
        block_depth_log2: 0,
    };
    let block = BlockLinearSurface::new(64, 16, gob_high);
    let at = |x: u32, y: u32| OUTPUT + block.offset(x * 4, y, 0);
    // Source (2, 1) is pixel 10: R 10, G 0x4a, B 0x8a; written B, G, R, A.
    // Constant alpha 1023 and alpha fill "opaque" (0).
    assert_eq!(bench.peek(at(10, 13)), [0x8a, 0x4a, 10, 0xff]);
    // Source (5, 2) is pixel 21.
    assert_eq!(bench.peek(at(13, 14)), [0x95, 0x55, 21, 0xff]);
    // Outside the destination: the black background.
    assert_eq!(bench.peek(at(9, 13)), [0, 0, 0, 0xff]);
}

#[test]
fn a_translucent_slot_is_blended_over_the_background_and_the_composited_alpha_kept() {
    let mut bench = Bench::new();
    bench.output(31, None, (1, 1), [0, 0, 0, 0]);
    // Background opaque blue; alpha fill "composited" (4).
    bench.set(0x10, 4 | 1023 << 6 | 1023 << 36);
    bench.slot(0, 31, None, (1, 1), [0, 0, 0, 0], [0, 0, 0, 0]);
    let base = SLOT_BASE;
    // Per-pixel alpha (no constant alpha); source × (K1·As),
    // destination × (1 − K1·As); alpha: source × K1 + destination × (1 − K1·As).
    bench.set(base + 0x10, 1023 << 10 | 1023 << 32);
    bench.set(base + 0xA0, 1023 | 3 << 32 | 4 << 36 | 1 << 44);
    // Red at alpha 0x66 (0.4).
    bench.poke(INPUT, &[0xff, 0, 0, 0x66]);
    bench.execute(Some(INPUT)).unwrap();
    // R = 0.4, B = 0.6, A = 0.4 + 1 × 0.6 = 1.
    assert_eq!(bench.peek(OUTPUT), [0x66, 0, 0x99, 0xff]);
}

#[test]
fn the_slot_colour_matrix_override_and_mask_apply() {
    let mut bench = Bench::new();
    bench.output(31, None, (1, 1), [0, 0, 0, 0]);
    bench.slot(0, 31, None, (1, 1), [0, 0, 0, 0], [0, 0, 0, 0]);
    let base = SLOT_BASE;
    // Matrix swapping red and blue, coefficients 1.0 = 256 << 0 (shift 0,
    // final >> 8): row r reads B, row b reads R, row g reads G.
    let one = 256u64;
    bench.set(base + 0x60, one << 40);
    bench.set(base + 0x68, one << 20 | 1 << 63);
    bench.set(base + 0x70, one);
    bench.set(base + 0x78, 0);
    // Override green with 1023; mask alpha (keep the destination's).
    bench.set(base + 0xA8, 1023 << 12 | 1 << 45 | 1 << 51);
    bench.set(0x10, 4 | 256 << 6);
    bench.poke(INPUT, &[0x10, 0x20, 0x30, 0xff]);
    bench.execute(Some(INPUT)).unwrap();
    // 0x30 → 10-bit 192.8 → 193 → back to 8 bits 48.1 → 0x30.
    assert_eq!(bench.peek(OUTPUT), [0x30, 0xff, 0x10, 0x40]);
}

#[test]
fn scaling_refuses_and_names_itself() {
    let mut bench = Bench::new();
    bench.output(31, None, (4, 4), [0, 0, 3, 3]);
    bench.slot(0, 31, None, (4, 4), [0, 0, 1, 1], [0, 0, 3, 3]);
    assert_eq!(bench.execute(Some(INPUT)), None);
    let notes = bench.notes.take();
    assert!(notes[0].contains("scaling"), "{notes:?}");
}

/// A host1x CDMA stream as nvnflinger's `DdkVicBlit` builds it, executed
/// through a channel: the composition happens and the THI increment with
/// condition 1 (operation done) advances the syncpoint.
#[test]
fn a_channel_submission_composes_and_signals_its_syncpoint() {
    let mut bench = Bench::new();
    bench.output(31, None, (2, 1), [0, 0, 1, 0]);
    bench.slot(0, 31, None, (2, 1), [0, 0, 1, 0], [0, 0, 1, 0]);
    bench.poke(INPUT, &[1, 2, 3, 4, 5, 6, 7, 8]);
    bench.poke(CONFIG, &bench.config.map(u64::to_le_bytes).concat());
    let vic = Arc::clone(&bench.vic);
    let channel = Channel::new(
        DeviceMemory::new(&bench.mc, &bench.ram, SMMU_HOST1X_ASID),
        Arc::clone(&bench.syncpoints),
        Arc::new(move |class, offset, value| {
            (class == CLASS).then_some(())?;
            vic.thi_write(offset, value)
        }),
        Arc::clone(&bench.notes),
    );
    // SETCLASS 0x5d; per method INCR THI 0x10 count 2; NONINCR THI 0
    // count 1 with `id | 1 << 8`.
    let mut words = vec![CLASS << 6];
    for (method, value) in methods(Some(INPUT)) {
        words.extend([0x1010_0002, method >> 2, value]);
    }
    words.extend([0x2000_0001, SYNCPOINT as u32 | 1 << 8]);
    let push = 0x4_0000u32;
    bench.poke(
        push.into(),
        &words
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let end = push + 4 * words.len() as u32;
    for (register, value) in [
        (0x14, push),
        (0x20, push + 0x1000),
        (0x18, push),
        (0x24, 7),
        (0x24, 0),
    ] {
        channel.write(0, register, 4, value.into()).unwrap();
    }
    assert_eq!(bench.syncpoints.get(SYNCPOINT), Some(0));
    channel.write(0, 0x18, 4, end.into()).unwrap();
    assert_eq!(bench.syncpoints.get(SYNCPOINT), Some(1));
    assert_eq!(channel.read(0, 0x1c, 4), Some(end.into()));
    // Constant alpha 1023, alpha fill opaque.
    assert_eq!(bench.peek(OUTPUT), [1, 2, 3, 0xff]);
    assert_eq!(bench.peek(OUTPUT + 4), [5, 6, 7, 0xff]);
    assert!(bench.notes.take().is_empty());
}
