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
        self.send(methods(input))
    }

    /// Writes the configuration and sends `methods` to the THI.
    fn send(&self, methods: Vec<(u32, u32)>) -> Option<()> {
        self.poke(CONFIG, &self.config.map(u64::to_le_bytes).concat());
        for (method, value) in methods {
            self.vic.thi_write(THI_METHOD_OFFSET, method >> 2)?;
            self.vic.thi_write(THI_METHOD_DATA, value)?;
        }
        Some(())
    }

    fn peek_bytes(&self, address: u64, count: usize) -> Vec<u8> {
        let mut bytes = vec![0; count];
        self.ram.read(WINDOW + address, &mut bytes).unwrap();
        bytes
    }
}

/// nvnflinger's method sequence (`0xAE1A0`), one input in slot 0.
fn methods(input: Option<u64>) -> Vec<(u32, u32)> {
    let mut methods = vec![
        (0x200, 1),
        (0x704, 0x610110),
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

/// A semi-planar 4:2:0 slot (format 68, U first) through an identity
/// colour matrix: R, G, B are the pixel's Y, U, V, chroma interpolated
/// between the samples sited with the even luma samples.
#[test]
fn a_semi_planar_slot_upsamples_chroma_between_its_samples() {
    let mut bench = Bench::new();
    bench.output(31, None, (4, 2), [0, 0, 3, 1]);
    bench.slot(0, 68, None, (4, 2), [0, 0, 3, 1], [0, 0, 3, 1]);
    let base = SLOT_BASE;
    // Luma 4x2, chroma 2x1 pairs (pitches 4 and 2 × 2 bytes).
    bench.set(base + 0x48, 3 | 1 << 14 | 1 << 32);
    let one = 256u64;
    bench.set(base + 0x60, one);
    bench.set(base + 0x68, one << 20 | 1 << 63);
    bench.set(base + 0x70, one << 40);
    bench.set(base + 0x78, 0);
    const CHROMA: u64 = INPUT + 0x1000;
    bench.poke(INPUT, &[10, 20, 30, 40, 50, 60, 70, 80]);
    bench.poke(CHROMA, &[100, 200, 110, 210]);
    let mut methods = methods(Some(INPUT));
    methods.insert(methods.len() - 1, (0x404, (CHROMA >> 8) as u32));
    bench.send(methods).unwrap();
    let pixel = |x: u64, y: u64| bench.peek(OUTPUT + 16 * y + 4 * x);
    assert_eq!(pixel(0, 0), [10, 100, 200, 0xff]);
    assert_eq!(pixel(1, 0), [20, 105, 205, 0xff]);
    assert_eq!(pixel(2, 0), [30, 110, 210, 0xff]);
    // The right edge sample is repeated.
    assert_eq!(pixel(3, 0), [40, 110, 210, 0xff]);
    assert_eq!(pixel(1, 1), [60, 105, 205, 0xff]);
}

#[test]
fn a_semi_planar_slot_needs_its_chroma_plane_and_a_colour_matrix() {
    let mut bench = Bench::new();
    bench.output(31, None, (2, 2), [0, 0, 1, 1]);
    bench.slot(0, 68, None, (2, 2), [0, 0, 1, 1], [0, 0, 1, 1]);
    bench.set(SLOT_BASE + 0x48, 1 | 1 << 14);
    assert_eq!(bench.execute(Some(INPUT)), None);
    let notes = bench.notes.take();
    assert!(notes[0].contains("2-plane YUV surface"), "{notes:?}");
    let mut methods = methods(Some(INPUT));
    methods.insert(methods.len() - 1, (0x404, (INPUT >> 8) as u32 + 1));
    assert_eq!(bench.send(methods), None);
    let notes = bench.notes.take();
    assert!(notes[0].contains("colour matrix"), "{notes:?}");
}

/// Slot 0 as [`Bench::slot`] sets it, with filter length `length` on both
/// axes.
fn filter_length(bench: &mut Bench, length: u64) {
    bench.set(SLOT_BASE, 1 | 1 << 8 | length << 20 | length << 22);
}

/// Red of the first `count` pixels of a pitch-linear A8B8G8R8 output row.
fn reds(bench: &Bench, count: usize) -> Vec<u8> {
    (bench.peek_bytes(OUTPUT, 4 * count).chunks(4))
        .map(|pixel| pixel[0])
        .collect()
}

/// One source row of reds `values`, scaled to `width` with filter length
/// `length`; the reds written.
fn scale_row(values: &[u8], width: u64, length: u64) -> Vec<u8> {
    let mut bench = Bench::new();
    let count = values.len() as u64;
    bench.output(31, None, (width, 1), [0, 0, width - 1, 0]);
    bench.slot(
        0,
        31,
        None,
        (count, 1),
        [0, 0, count - 1, 0],
        [0, 0, width - 1, 0],
    );
    filter_length(&mut bench, length);
    let pixels: Vec<u8> = values.iter().flat_map(|&red| [red, 0, 0, 0xff]).collect();
    bench.poke(INPUT, &pixels);
    bench.execute(Some(INPUT)).unwrap();
    assert!(bench.notes.take().is_empty());
    reds(&bench, width as usize)
}

/// Reducing with the long filters averages each destination sample's
/// footprint, source samples weighted by the part covered.
#[test]
fn a_reducing_slot_averages_the_footprint_of_each_destination_sample() {
    assert_eq!(scale_row(&[30, 60, 90, 120, 150, 180], 2, 3), [60, 150]);
    // Ratio 1.5: [0, 1.5) and [1.5, 3).
    assert_eq!(scale_row(&[30, 60, 90], 2, 2), [40, 80]);
}

/// Enlarging with the long filters, and the two-tap filter either way,
/// interpolate linearly at each destination centre; past the source
/// rectangle the edge samples repeat.
#[test]
fn bilinear_scaling_interpolates_at_destination_centres() {
    // Centres −0.25, 0.25, 0.75, 1.25.
    assert_eq!(scale_row(&[40, 200], 4, 3), [40, 80, 160, 200]);
    assert_eq!(scale_row(&[40, 200], 4, 1), [40, 80, 160, 200]);
    // Centres 0.5 and 2.5.
    assert_eq!(scale_row(&[30, 60, 90, 120], 2, 1), [45, 105]);
}

/// The one-tap filter takes the sample nearest the destination centre.
#[test]
fn nearest_scaling_takes_the_sample_at_each_destination_centre() {
    // Ratio 3: centres 1 and 4.
    assert_eq!(scale_row(&[30, 60, 90, 120, 150, 180], 2, 0), [60, 150]);
}

#[test]
fn scaling_with_noise_or_detail_filters_refuses_and_names_them() {
    let mut bench = Bench::new();
    bench.output(31, None, (4, 4), [0, 0, 3, 3]);
    bench.slot(0, 31, None, (4, 4), [0, 0, 1, 1], [0, 0, 3, 3]);
    // FilterDetail 0x100.
    bench.set(SLOT_BASE + 8, 0x100 << 10);
    assert_eq!(bench.execute(Some(INPUT)), None);
    let notes = bench.notes.take();
    assert!(notes[0].contains("noise/detail"), "{notes:?}");
}

/// A single-plane slot and output use only their luma plane addresses
/// (`vic_ceb6_types.h`: "For single plane pixel format(ARGB), only use
/// Luma surface"): chroma addresses left from an earlier configuration are
/// neither read nor written.
#[test]
fn single_plane_surfaces_ignore_their_chroma_plane_addresses() {
    let mut bench = Bench::new();
    bench.output(31, None, (2, 1), [0, 0, 1, 0]);
    bench.slot(0, 31, None, (2, 1), [0, 0, 1, 0], [0, 0, 1, 0]);
    bench.poke(INPUT, &[1, 2, 3, 4, 5, 6, 7, 8]);
    bench.poke(CHROMA_U, &[0x22; 8]);
    bench.poke(CHROMA_V, &[0x33; 8]);
    let mut methods = planar_methods();
    let execute = methods.pop().unwrap();
    methods.extend([
        (0x404, (CHROMA_U >> 8) as u32),
        (0x408, (CHROMA_V >> 8) as u32),
    ]);
    methods.push(execute);
    bench.send(methods).unwrap();
    assert!(bench.notes.take().is_empty());
    // Constant alpha 1023 makes the output opaque.
    assert_eq!(bench.peek_bytes(OUTPUT, 8), [1, 2, 3, 0xff, 5, 6, 7, 0xff]);
    assert_eq!(bench.peek_bytes(CHROMA_U, 8), [0x22; 8]);
    assert_eq!(bench.peek_bytes(CHROMA_V, 8), [0x33; 8]);
}

/// A fully planar `T_Y8___U8___V8_N422R` slot (format 71) through an
/// identity colour matrix: chroma at full horizontal resolution, sited
/// between each pair of rows (vertical siting 1) and interpolated between
/// them, the edge rows repeated.
#[test]
fn a_planar_n422r_slot_upsamples_chroma_vertically() {
    let mut bench = Bench::new();
    bench.output(31, None, (2, 4), [0, 0, 1, 3]);
    bench.slot(0, 71, None, (2, 4), [0, 0, 1, 3], [0, 0, 1, 3]);
    let base = SLOT_BASE;
    let (w0, w1) = planar_words(71, [0, 1], (2, 4), (2, 4), (2, 2));
    bench.set(base + 0x40, w0);
    bench.set(base + 0x48, w1);
    let one = 256u64;
    bench.set(base + 0x60, one);
    bench.set(base + 0x68, one << 20 | 1 << 63);
    bench.set(base + 0x70, one << 40);
    bench.set(base + 0x78, 0);
    bench.poke(INPUT, &[10, 20, 30, 40, 50, 60, 70, 80]);
    bench.poke(CHROMA_U, &[100, 120, 200, 220]);
    bench.poke(CHROMA_V, &[40, 44, 80, 84]);
    let mut methods = methods(Some(INPUT));
    let execute = methods.pop().unwrap();
    methods.extend([
        (0x404, (CHROMA_U >> 8) as u32),
        (0x408, (CHROMA_V >> 8) as u32),
        execute,
    ]);
    bench.send(methods).unwrap();
    assert!(bench.notes.take().is_empty());
    let pixel = |x: u64, y: u64| bench.peek(OUTPUT + 8 * y + 4 * x);
    // Rows sit at chroma positions −0.25, 0.25, 0.75 and 1.25.
    assert_eq!(pixel(0, 0), [10, 100, 40, 0xff]);
    assert_eq!(pixel(0, 1), [30, 125, 50, 0xff]);
    assert_eq!(pixel(1, 1), [40, 145, 54, 0xff]);
    assert_eq!(pixel(0, 2), [50, 175, 70, 0xff]);
    assert_eq!(pixel(1, 3), [80, 220, 84, 0xff]);
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
        Arc::default(),
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

/// BT.601 full-range RGB to YCbCr (JFIF), as the output matrix of the
/// EXECUTE seen when News opens: coefficients over 2^19 (shift 11), chroma
/// offsets 131458 / 256 / 1023 = 128/255.
const YCBCR_MATRIX: [(usize, u64); 4] = [
    (0x30, 0xb40000ea66f2645a),
    (0x38, 0x8ca688d59934b22d),
    (0x40, 0x0f597a400000e979),
    (0x48, 0x0201822018200000),
];

/// Planar output surface configuration: `format`, vertical chroma siting
/// `vertical` and horizontal `horizontal`, pitch-linear, `size`, the luma
/// and chroma plane extents.
fn planar_words(
    format: u64,
    [horizontal, vertical]: [u64; 2],
    (width, height): (u64, u64),
    (luma_width, luma_height): (u64, u64),
    (chroma_width, chroma_height): (u64, u64),
) -> (u64, u64) {
    (
        format | horizontal << 7 | vertical << 9 | (width - 1) << 32 | (height - 1) << 46,
        (luma_width - 1)
            | (luma_height - 1) << 14
            | (chroma_width - 1) << 32
            | (chroma_height - 1) << 46,
    )
}

const CHROMA_U: u64 = 0x1_1000;
const CHROMA_V: u64 = 0x1_2000;

/// nvnflinger's methods with the three output planes.
fn planar_methods() -> Vec<(u32, u32)> {
    let mut methods = methods(Some(INPUT));
    let execute = methods.pop().unwrap();
    methods.extend([
        (0x724, (CHROMA_U >> 8) as u32),
        (0x728, (CHROMA_V >> 8) as u32),
    ]);
    methods.push(execute);
    methods
}

/// A 4x4 `T_Y8___U8___V8_N422R` output with padded planes (luma pitch 8,
/// chroma pitch 6) and the target its left two columns: each luma sample
/// is Y of its pixel, each chroma sample the mean of the U (V) of the two
/// pixels above each other it is sited between, every value rounded.
#[test]
fn a_planar_n422r_output_has_luma_per_pixel_and_chroma_per_vertical_pair() {
    let mut bench = Bench::new();
    bench.set(0x18, rect([0, 0, 1, 3]));
    let (w0, w1) = planar_words(71, [0, 1], (4, 4), (8, 4), (6, 2));
    bench.set(0x20, w0);
    bench.set(0x28, w1);
    for (offset, word) in YCBCR_MATRIX {
        bench.set(offset, word);
    }
    bench.slot(0, 31, None, (2, 4), [0, 0, 1, 3], [0, 0, 1, 3]);
    // R, G, B, A rows of two: red, white; green, black; blue, brown twice.
    let brown = [0x80, 0x40, 0x20, 0xff];
    let blue = [0, 0, 0xff, 0xff];
    let pixels = [
        [0xff, 0, 0, 0xff],
        [0xff; 4],
        [0, 0xff, 0, 0xff],
        [0, 0, 0, 0xff],
        blue,
        brown,
        blue,
        brown,
    ];
    bench.poke(INPUT, &pixels.concat());
    bench.poke(OUTPUT, &[0x11; 32]);
    bench.poke(CHROMA_U, &[0x22; 12]);
    bench.poke(CHROMA_V, &[0x33; 12]);
    bench.send(planar_methods()).unwrap();
    assert!(bench.notes.take().is_empty());
    // Y = 0.299 R + 0.587 G + 0.114 B: red 76.2, white 255, green 149.7,
    // black 0, blue 29.1, brown 79.5.
    let luma = [[76, 255], [150, 0], [29, 79], [29, 79]];
    let luma_rows = bench.peek_bytes(OUTPUT, 32);
    for (y, row) in luma.iter().enumerate() {
        let mut expected = [0x11; 8];
        expected[..2].copy_from_slice(row);
        assert_eq!(luma_rows[y * 8..y * 8 + 8], expected, "luma row {y}");
    }
    // U = 128 − 0.1687 R − 0.3313 G + 0.5 B, V = 128 + 0.5 R − 0.4187 G −
    // 0.0813 B, each clamped: red (85.0, 255), green (43.5, 21.2) → (64.3,
    // 138.1); white and black (128, 128); blue (255, 107.3) twice; brown
    // (101.2, 162.6) twice.
    for (address, rows) in [
        (CHROMA_U, [[64, 128], [255, 101]]),
        (CHROMA_V, [[138, 128], [107, 163]]),
    ] {
        let fill = if address == CHROMA_U { 0x22 } else { 0x33 };
        let plane = bench.peek_bytes(address, 12);
        for (y, row) in rows.iter().enumerate() {
            let mut expected = [fill; 6];
            expected[..2].copy_from_slice(row);
            assert_eq!(plane[y * 6..y * 6 + 6], expected, "{address:#x} row {y}");
        }
    }
}

/// 4:2:0 with both axes subsampled: the chroma sample of each 2x2 block is
/// the mean of the four.
#[test]
fn a_planar_n420_output_averages_each_two_by_two_block() {
    let mut bench = Bench::new();
    bench.set(0x18, rect([0, 0, 1, 1]));
    let (w0, w1) = planar_words(72, [1, 1], (2, 2), (2, 2), (1, 1));
    bench.set(0x20, w0);
    bench.set(0x28, w1);
    for (offset, word) in YCBCR_MATRIX {
        bench.set(offset, word);
    }
    bench.slot(0, 31, None, (2, 2), [0, 0, 1, 1], [0, 0, 1, 1]);
    // Red, green, blue, black.
    let pixels = [
        [0xff, 0, 0, 0xff],
        [0, 0xff, 0, 0xff],
        [0, 0, 0xff, 0xff],
        [0, 0, 0, 0xff],
    ];
    bench.poke(INPUT, &pixels.concat());
    bench.send(planar_methods()).unwrap();
    assert_eq!(bench.peek(OUTPUT), [76, 150, 29, 0]);
    // U: (85.0 + 43.5 + 255 + 128) / 4 = 127.9; V: (255 + 21.2 + 107.3 +
    // 128) / 4 = 127.9.
    assert_eq!(bench.peek_bytes(CHROMA_U, 1), [128]);
    assert_eq!(bench.peek_bytes(CHROMA_V, 1), [128]);
}

/// grc's encoder input: NV12 (68), block-linear, chroma co-sited with the
/// even luma column and between the two rows. Each chroma pair is the
/// mean of the U (V) of the even column's two pixels; the odd columns'
/// chroma has no weight.
#[test]
fn a_block_linear_nv12_output_takes_chroma_from_the_even_column() {
    let mut bench = Bench::new();
    bench.set(0x18, rect([0, 0, 3, 1]));
    let (w0, w1) = planar_words(68, [0, 1], (4, 2), (4, 2), (2, 1));
    bench.set(0x20, w0 | 1 << 11);
    bench.set(0x28, w1);
    for (offset, word) in YCBCR_MATRIX {
        bench.set(offset, word);
    }
    bench.slot(0, 31, None, (4, 2), [0, 0, 3, 1], [0, 0, 3, 1]);
    let (red, green, blue) = ([0xff, 0, 0, 0xff], [0, 0xff, 0, 0xff], [0, 0, 0xff, 0xff]);
    let (white, black, brown) = ([0xff; 4], [0, 0, 0, 0xff], [0x80, 0x40, 0x20, 0xff]);
    let pixels = [red, black, blue, white, green, white, brown, black];
    bench.poke(INPUT, &pixels.concat());
    bench.send(planar_methods()).unwrap();
    assert!(bench.notes.take().is_empty());
    let gob = |row_bytes| {
        BlockLinearSurface::new(
            row_bytes,
            2,
            BlockLinear {
                block_height_log2: 0,
                block_depth_log2: 0,
            },
        )
    };
    let luma = bench.peek_bytes(OUTPUT, 512);
    let at = |y| gob(4).offset(0, y, 0) as usize;
    assert_eq!(luma[at(0)..at(0) + 4], [76, 0, 29, 255]);
    assert_eq!(luma[at(1)..at(1) + 4], [150, 255, 79, 0]);
    // U, V: red and green (85.0, 255; 43.5, 21.2) → (64.3, 138.1); blue
    // and brown (255, 107.3; 101.2, 162.6) → (178.1, 135.0).
    let chroma = bench.peek_bytes(CHROMA_U, 512);
    assert_eq!(chroma[..4], [64, 138, 178, 135]);
}

#[test]
fn planar_outputs_the_model_does_not_establish_refuse_and_name_themselves() {
    let cases = [
        // Siting 3 is none of even, between and odd.
        (72, [3, 1], [0, 0, 3, 3], true, "horizontal chroma siting 3"),
        // A target whose top row shares its chroma with the row above.
        (71, [0, 1], [0, 1, 3, 2], true, "splits"),
        // No output matrix: what would be Y, U and V is not said.
        (
            71,
            [0, 1],
            [0, 0, 3, 3],
            false,
            "without an output colour matrix",
        ),
        // Chroma sited off a full-resolution axis.
        (69, [1, 0], [0, 0, 3, 3], true, "horizontal chroma siting 1"),
    ];
    for (format, siting, target, matrix, reason) in cases {
        let mut bench = Bench::new();
        bench.set(0x18, rect(target));
        let (w0, w1) = planar_words(format, siting, (4, 4), (4, 4), (4, 4));
        bench.set(0x20, w0);
        bench.set(0x28, w1);
        if matrix {
            for (offset, word) in YCBCR_MATRIX {
                bench.set(offset, word);
            }
        }
        bench.slot(0, 31, None, (4, 4), [0, 0, 3, 3], [0, 0, 3, 3]);
        bench.poke(OUTPUT, &[0x11; 16]);
        assert_eq!(bench.send(planar_methods()), None, "{reason}");
        let notes = bench.notes.take();
        assert!(notes[0].contains(reason), "{notes:?}");
        assert_eq!(bench.peek_bytes(OUTPUT, 16), [0x11; 16], "{reason}");
    }
}

/// RAM behind 4 MiB SMMU sections of device address space, and a VIC
/// behind a host1x channel, to replay an EXECUTE from a refusal note
/// (methods and every non-zero configuration word, real device addresses).
struct Capture {
    ram: Arc<GuestRam>,
    mc: Arc<Mc>,
    syncpoints: Arc<Syncpoints>,
    notes: Arc<Notes>,
}

/// The block-linear layout (16-GOB blocks) of the 1280x720 A8B8G8R8 frames
/// both captures read.
fn frame_layout() -> BlockLinearSurface {
    BlockLinearSurface::new(
        1280 * 4,
        720,
        BlockLinear {
            block_height_log2: 4,
            block_depth_log2: 0,
        },
    )
}

/// R, G, B, A of the frame at (`x`, `y`).
fn pattern(x: u32, y: u32) -> [u8; 4] {
    [x as u8, y as u8, (x ^ y) as u8, 0x40]
}

impl Capture {
    /// `sections` are the device sections the submission uses.
    fn new(sections: &[u64]) -> Capture {
        let ram = Arc::new(GuestRam::new(RAM_BASE, (sections.len() + 1) << 22).unwrap());
        let mc = Arc::new(Mc::new(DramSize::Gb4));
        mc.write(0, 0x1c, 4, 1).unwrap();
        mc.write(0, 0x20, 4, 0xe008_0000).unwrap();
        for client in [SMMU_VIC_ASID, SMMU_HOST1X_ASID] {
            mc.write(0, client, 4, 0x8000_0001).unwrap();
        }
        for (n, section) in sections.iter().enumerate() {
            let frame = (RAM_BASE + ((n as u64 + 1) << 22)) >> 12;
            let entry = 0xc000_0000 | frame as u32;
            ram.write(RAM_BASE + section * 4, &entry.to_le_bytes())
                .unwrap();
        }
        Capture {
            ram,
            mc,
            syncpoints: Arc::default(),
            notes: Arc::default(),
        }
    }

    fn memory(&self) -> DeviceMemory {
        DeviceMemory::new(&self.mc, &self.ram, SMMU_VIC_ASID)
    }

    /// Writes the [`pattern`] frame at `address`.
    fn frame(&self, address: u64) {
        let layout = frame_layout();
        let mut input = vec![0u8; layout.size(1) as usize];
        for y in 0..720 {
            for x in 0..1280 {
                let at = layout.offset(x * 4, y, 0) as usize;
                input[at..at + 4].copy_from_slice(&pattern(x, y));
            }
        }
        self.memory().write(address, &input).unwrap();
    }

    /// Writes the configuration at `config`, then sends `methods`, EXECUTE
    /// and a THI increment with condition 1 as a CDMA stream.
    fn submit(&self, config: u64, words: &[(u64, u64)], methods: &[(u32, u32)]) {
        let mut bytes = [0u8; CONFIG_SIZE];
        for &(offset, word) in words {
            bytes[offset as usize..offset as usize + 8].copy_from_slice(&word.to_le_bytes());
        }
        self.memory().write(config, &bytes).unwrap();
        let vic = Arc::new(Vic::new(
            self.memory(),
            Arc::clone(&self.syncpoints),
            Arc::clone(&self.notes),
        ));
        let channel = Channel::new(
            DeviceMemory::new(&self.mc, &self.ram, SMMU_HOST1X_ASID),
            Arc::clone(&self.syncpoints),
            Arc::default(),
            Arc::new(move |class, offset, value| {
                (class == CLASS).then_some(())?;
                vic.thi_write(offset, value)
            }),
            Arc::clone(&self.notes),
        );
        let mut stream = vec![CLASS << 6];
        for &(method, value) in methods.iter().chain(&[(0x300, 0x100)]) {
            stream.extend([0x1010_0002, method >> 2, value]);
        }
        stream.extend([0x2000_0001, SYNCPOINT as u32 | 1 << 8]);
        let push = 0xff00_0000u32;
        let bytes: Vec<u8> = stream.iter().flat_map(|w| w.to_le_bytes()).collect();
        DeviceMemory::new(&self.mc, &self.ram, SMMU_HOST1X_ASID)
            .write(push.into(), &bytes)
            .unwrap();
        for (register, value) in [
            (0x14, 0),
            (0x20, push + 0x1000),
            (0x18, push),
            (0x24, 7),
            (0x24, 0),
        ] {
            channel.write(0, register, 4, value.into()).unwrap();
        }
        channel
            .write(0, 0x18, 4, (push + bytes.len() as u32).into())
            .unwrap();
        assert!(self.notes.take().is_empty());
        assert_eq!(self.syncpoints.get(SYNCPOINT), Some(1));
    }
}

/// The EXECUTE nvnflinger sends when starter launches netConnect: an
/// opaque 1:1 copy of a 1280x720 block-linear A8B8G8R8 frame through sRGB
/// degamma/regamma and an identity output matrix.
#[test]
fn the_netconnect_launch_capture_copies_the_frame_and_signals() {
    let capture = Capture::new(&[0x380, 0x381, 0x389, 0x38a, 0x38b, 0x3fc]);
    capture.frame(0xe26e_0000);
    let size = frame_layout().size(1) as usize;
    capture
        .memory()
        .write(0xe016_0000, &vec![0x5a; size])
        .unwrap();
    capture.submit(
        0xe2aa_0000,
        &[
            (0x010, 0x0000400000000004),
            (0x018, 0x02cf000004ff0000),
            (0x020, 0x00b3c4ff00020a1f),
            (0x028, 0x0fffffff00b3c4ff),
            (0x030, 0xa000000000040000),
            (0x038, 0x8000004000000000),
            (0x040, 0x0400000000000000),
            (0x090, 0x0000000000000101),
            (0x0a0, 0x008003ff000ffc00),
            (0x0b0, 0x04ff000000000000),
            (0x0b8, 0x02cf000000000000),
            (0x0c0, 0x02cf000004ff0000),
            (0x0d0, 0x00b3c4ff000a0a1f),
            (0x0d8, 0x0fffffff00b3c4ff),
            (0x130, 0x00002050000003ff),
            (0x138, 0x000083ff00000000),
        ],
        &[
            (0x200, 0x1),
            (0x704, 0x610110),
            (0x708, 0xe2aa00),
            (0x70c, 0xe2ac00),
            (0x71c, 0x3c0),
            (0x72c, 0xe0043d),
            (0x720, 0xe01600),
            (0x400, 0xe26e00),
        ],
    );
    let layout = frame_layout();
    let mut output = vec![0u8; size];
    capture.memory().read(0xe016_0000, &mut output).unwrap();
    for y in 0..720 {
        for x in 0..1280 {
            let at = layout.offset(x * 4, y, 0) as usize;
            let [r, g, b, _] = pattern(x, y);
            assert_eq!(output[at..at + 4], [r, g, b, 0xff], "pixel ({x}, {y})");
        }
    }
}

/// The EXECUTE seen when News opens from HOME (`/tmp/gapgpu/run2.log`):
/// the same slot copied into a pitch-linear 1280x720
/// `T_Y8___U8___V8_N422R` output (chroma planes 1280x360) through the
/// BT.601 full-range matrix. Checked against the textbook JFIF formulas
/// in f64, within one step of rounding.
#[test]
fn the_news_capture_converts_the_frame_to_planar_ycbcr_422r() {
    let capture = Capture::new(&[0x380, 0x384, 0x385, 0x388, 0x389, 0x3fc]);
    capture.frame(0xe106_0000);
    let mut words = vec![
        (0x010, 0x0000400000000004),
        (0x018, 0x02cf000004ff0000),
        (0x020, 0x00b3c4ff00000247),
        (0x028, 0x0059c4ff00b3c4ff),
        (0x090, 0x0000000000000101),
        (0x0a0, 0x008003ff000ffc00),
        (0x0b0, 0x04ff000000000000),
        (0x0b8, 0x02cf000000000000),
        (0x0c0, 0x02cf000004ff0000),
        (0x0d0, 0x00b3c4ff000a0a1f),
        (0x0d8, 0x0fffffff00b3c4ff),
        (0x130, 0x00002050000003ff),
        (0x138, 0x000083ff00000000),
    ];
    words.extend(YCBCR_MATRIX.map(|(offset, word)| (offset as u64, word)));
    capture.submit(
        0xe238_0000,
        &words,
        &[
            (0x200, 0x1),
            (0x400, 0xe10600),
            (0x704, 0x610110),
            (0x708, 0xe23800),
            (0x70c, 0xe23a00),
            (0x71c, 0x3c0),
            (0x720, 0xe23c00),
            (0x724, 0xe24c00),
            (0x728, 0xe25600),
            (0x72c, 0xe0043d),
        ],
    );
    let read = |address: u64, size: usize| {
        let mut bytes = vec![0u8; size];
        capture.memory().read(address, &mut bytes).unwrap();
        bytes
    };
    let luma = read(0xe23c_0000, 1280 * 720);
    let chroma = [read(0xe24c_0000, 1280 * 360), read(0xe256_0000, 1280 * 360)];
    let ycbcr = |x: u32, y: u32| {
        let [r, g, b, _] = pattern(x, y).map(f64::from);
        [
            0.299 * r + 0.587 * g + 0.114 * b,
            128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b,
            128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b,
        ]
        .map(|value| value.clamp(0.0, 255.0))
    };
    let close = |actual: u8, expected: f64, what: &str| {
        assert!(
            (f64::from(actual) - expected).abs() <= 1.0,
            "{what}: {actual} for {expected}"
        );
    };
    for y in 0..720 {
        for x in 0..1280 {
            let at = (y * 1280 + x) as usize;
            close(luma[at], ycbcr(x, y)[0], &format!("Y ({x}, {y})"));
        }
    }
    for y in 0..360 {
        for x in 0..1280 {
            let at = (y * 1280 + x) as usize;
            for (plane, component) in chroma.iter().zip(1..) {
                let mean = (ycbcr(x, 2 * y)[component] + ycbcr(x, 2 * y + 1)[component]) / 2.0;
                close(plane[at], mean, &format!("chroma {component} ({x}, {y})"));
            }
        }
    }
}

/// The EXECUTE the capture button on HOME sends after the News-style
/// conversion (GapExplorer3 `runs/b1`): the 1280x720
/// `T_Y8___U8___V8_N422R` picture (luma 0xe2360000, chroma 0xe2460000 and
/// 0xe2500000, vertical siting 1) reduced with ten-tap filters to a 96x54
/// pitch-linear A8B8G8R8 thumbnail (pitch 128 pixels) through the BT.601
/// full-range YCbCr-to-RGB matrix, the output's chroma addresses still
/// those of the input. With neutral chroma each thumbnail pixel is grey at
/// the mean luma of its footprint, here the luma ramp's value at the
/// footprint's centre.
#[test]
fn the_capture_thumbnail_reduces_the_planar_picture_to_rgb() {
    let capture = Capture::new(&[0x388, 0x389, 0x3fc]);
    let ramp = |x: f64, y: f64| x / 10.0 + y / 6.0;
    let luma: Vec<u8> = (0..720)
        .flat_map(|y| (0..1280).map(move |x| ramp(f64::from(x), f64::from(y)).round() as u8))
        .collect();
    capture.memory().write(0xe236_0000, &luma).unwrap();
    for plane in [0xe246_0000, 0xe250_0000] {
        capture
            .memory()
            .write(plane, &vec![128; 1280 * 360])
            .unwrap();
    }
    capture
        .memory()
        .write(0xe25a_0000, &vec![0x5a; 512 * 54])
        .unwrap();
    capture.submit(
        0xe272_0000,
        &[
            (0x000, 0x0000000000040004),
            (0x010, 0x000000000000ffc0),
            (0x018, 0x00350000005f0000),
            (0x020, 0x000d405f0000021f),
            (0x028, 0x0fffffff000d407f),
            (0x090, 0x0000000000f00101),
            (0x0a0, 0x000007ff000ffc00),
            (0x0b0, 0x04ff000000000000),
            (0x0b8, 0x02cf000000000000),
            (0x0c0, 0x00350000005f0000),
            (0x0d0, 0x00b3c4ff00100247),
            (0x0d8, 0x0059c4ff00b3c4ff),
            (0x0f0, 0xa400004000040000),
            (0x0f8, 0x871687e9f9c00000),
            (0x100, 0x000000d24ba59ba6),
            (0x108, 0x0c721221f6ed3012),
            (0x130, 0x00002050000003ff),
            (0x138, 0x0000800000000000),
        ],
        &[
            (0x200, 0x1),
            (0x400, 0xe23600),
            (0x404, 0xe24600),
            (0x408, 0xe25000),
            (0x704, 0x610110),
            (0x708, 0xe27200),
            (0x70c, 0xe27400),
            (0x71c, 0x3c0),
            (0x720, 0xe25a00),
            (0x724, 0xe24600),
            (0x728, 0xe25000),
            (0x72c, 0xe0043d),
        ],
    );
    let mut output = vec![0u8; 512 * 54];
    capture.memory().read(0xe25a_0000, &mut output).unwrap();
    let (rx, ry) = (1280.0 / 96.0, 720.0 / 54.0);
    for y in 0..54 {
        for x in 0..128 {
            let at = (y * 128 + x) * 4;
            let pixel = &output[at..at + 4];
            if x >= 96 {
                assert_eq!(pixel, [0x5a; 4], "outside the target ({x}, {y})");
                continue;
            }
            let grey = ramp((x as f64 + 0.5) * rx - 0.5, (y as f64 + 0.5) * ry - 0.5);
            for &component in &pixel[..3] {
                assert!(
                    (f64::from(component) - grey).abs() <= 1.0,
                    "({x}, {y}): {pixel:?} for {grey}"
                );
            }
            assert_eq!(pixel[3], 0xff);
        }
    }
    // The input's chroma planes, named again as the output's, are unchanged.
    let mut chroma = vec![0u8; 1280 * 360];
    capture.memory().read(0xe246_0000, &mut chroma).unwrap();
    assert!(chroma.iter().all(|&value| value == 128));
}

/// The gamma memo returns the curve's own value for every argument, also
/// for far more distinct arguments than it holds and for arguments that
/// share a set, in any order.
#[test]
fn the_gamma_memo_returns_the_curves_own_values() {
    let mut degamma = Memo::new(srgb_to_linear);
    let mut regamma = Memo::new(clamped_linear_to_srgb);
    let arguments: Vec<f32> = (0..20_000u32)
        .map(|n| (n.wrapping_mul(7919) % 65_536) as f32 / 65_535.0)
        .chain((0..=255).map(|byte| f32::from(byte as u8) / 255.0))
        .chain([-0.5, 1.5, -0.0])
        .collect();
    for _ in 0..2 {
        for &argument in &arguments {
            let expected = srgb_to_linear(argument);
            assert_eq!(
                degamma.get(argument).to_bits(),
                expected.to_bits(),
                "{argument}"
            );
            let expected = clamped_linear_to_srgb(argument);
            assert_eq!(
                regamma.get(argument).to_bits(),
                expected.to_bits(),
                "{argument}"
            );
        }
    }
}

/// Banding hands every row, the shorter last one included, to exactly one
/// band, with the right first row, whether or not it splits the work.
#[test]
fn bands_cover_every_row_once() {
    for (len, width, work) in [(10, 3, 0), (10, 3, BANDED_WORK), (1000, 7, BANDED_WORK)] {
        let mut items = vec![(0usize, 0usize); len];
        in_bands(&mut items, width, work, |first, band| {
            for (row, items) in (first..).zip(band.chunks_mut(width)) {
                for item in items {
                    item.0 += 1;
                    item.1 = row;
                }
            }
        });
        for (at, &(count, row)) in items.iter().enumerate() {
            assert_eq!((count, row), (1, at / width), "item {at} of {len}");
        }
    }
}

/// xorshift64*.
struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    fn chance(&mut self, one_in: u64) -> bool {
        self.below(one_in) == 0
    }

    fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len() as u64) as usize]
    }
}

const SOURCE_AT: u64 = 0x10_0000;
/// Output planes: luma, chroma U (or the pairs), chroma V.
const PLANES_AT: [u64; 3] = [0x20_0000, 0x28_0000, 0x30_0000];

/// A random enabled colour matrix (coefficients around 0..2, offsets
/// around ±0.25).
fn random_matrix(random: &mut Random) -> [u64; 4] {
    let mut words = [0u64; 4];
    words[0] = 2 << 60;
    words[1] = 1 << 63;
    for row in 0..3 {
        for (column, word) in words.iter_mut().enumerate() {
            let value = if column == 3 {
                random.below(1 << 18) as i64 - (1 << 17)
            } else {
                random.below(2560) as i64 - 512
            };
            *word |= ((value as u64) & 0xF_FFFF) << (row * 20);
        }
    }
    words
}

/// A random output surface configuration and the size of its picture:
/// 32-bit RGB (31-38) or 8-bit YUV (67-72), pitch- or block-linear.
fn random_output(random: &mut Random) -> ([u64; 2], (u32, u32)) {
    let planar = random.chance(2);
    let format = if planar {
        random.pick(&[67, 68, 69, 70, 71, 72])
    } else {
        31 + random.below(8)
    };
    let subsampling = yuv_input(format as u32).map_or([1, 1], |(_, s)| s);
    let siting = subsampling.map(|factor| if factor == 1 { 0 } else { random.below(3) });
    let block = random.chance(2).then(|| random.below(5));
    let kind = block.map_or(0, |log2| 1 << 11 | log2 << 15);
    let width = 2 + 2 * random.below(40) as u32;
    let height = 2 + 2 * random.below(30) as u32;
    let w0 = format
        | siting[0] << 7
        | siting[1] << 9
        | kind
        | u64::from(width - 1) << 32
        | u64::from(height - 1) << 46;
    let pad = |random: &mut Random, size: u32| u64::from(size + random.below(3) as u32 * 4);
    let chroma = [
        width.div_ceil(subsampling[0] as u32),
        height.div_ceil(subsampling[1] as u32),
    ];
    let w1 = (pad(random, width) - 1)
        | (u64::from(height) - 1) << 14
        | (pad(random, chroma[0]) - 1) << 32
        | (u64::from(chroma[1]) - 1) << 46;
    ([w0, w1], (width, height))
}

/// An inclusive rectangle word within `width`x`height`, sometimes larger
/// than it, aligned to `align` both ways.
fn random_rect(random: &mut Random, (width, height): (u32, u32), align: u64) -> u64 {
    let mut axis = |size: u32| {
        let start = random.below(u64::from(size) / 2 + 1) / align * align;
        let end = (start + align * (1 + random.below(u64::from(size) / align + 2))).max(start + 1);
        (start, end - 1)
    };
    let ((left, right), (top, bottom)) = (axis(width), axis(height));
    left | right << 16 | top << 32 | bottom << 48
}

/// A random configuration: an output and one slot (sometimes two, a
/// scaling one, one with a matrix), most of them the copies [`Fused`]
/// takes; and the methods naming their surfaces.
fn random_composition(random: &mut Random) -> (Config, [u32; method::COUNT]) {
    let mut config = Config([0; CONFIG_SIZE / 8]);
    let mut set = |offset: usize, value: u64| config.0[offset / 8] = value;
    let background = (0..4).fold(0u64, |word, n| {
        word | random.below(1024) << [16, 26, 36, 6][n]
    });
    let output =
        random.pick(&[0u64, 1, 4]) | background | random.below(2) << 46 | random.below(4) << 48;
    let (words, size) = random_output(random);
    let planar = yuv_input(field(words[0], 0, 7)).is_some();
    let transpose = !planar && field(words[0], 11, 4) == 0 && random.chance(6);
    set(0x10, output | u64::from(transpose) << 50);
    set(0x18, random_rect(random, size, 2));
    set(0x20, words[0]);
    set(0x28, words[1]);
    if planar || random.chance(2) {
        for (offset, word) in [0x30, 0x38, 0x40, 0x48]
            .into_iter()
            .zip(random_matrix(random))
        {
            set(offset, word);
        }
    }
    let slots = if random.chance(6) { 2 } else { 1 };
    for slot in 0..slots {
        let base = SLOT_BASE + slot * SLOT_SIZE;
        let filter = if random.chance(4) {
            random.below(16) << 20
        } else {
            0
        };
        set(base, 1 | 1 << 8 | filter);
        let clamp_low = if random.chance(2) {
            0
        } else {
            random.below(300)
        };
        let clamp_high = if random.chance(2) {
            1023
        } else {
            700 + random.below(324)
        };
        set(
            base + 0x10,
            clamp_low
                | clamp_high << 10
                | random.below(1024) << 32
                | random.below(2) << 42
                | random.below(2) << 55,
        );
        let surface_size = (8 + random.below(80) as u32, 8 + random.below(60) as u32);
        let source = random_rect(random, surface_size, 1);
        let (left, right) = (
            field(source, 0, 14),
            field(source, 16, 14).min(surface_size.0 - 1),
        );
        let (top, bottom) = (
            field(source, 32, 14),
            field(source, 48, 14).min(surface_size.1 - 1),
        );
        let destination = if random.chance(5) {
            random_rect(random, size, 1)
        } else {
            // The same size, somewhere.
            let (x, y) = (random.below(16), random.below(16));
            let (w, h) = (u64::from(right - left), u64::from(bottom - top));
            x | (x + w) << 16 | y << 32 | (y + h) << 48
        };
        set(base + 0x20, u64::from(left) << 16 | u64::from(right) << 48);
        set(base + 0x28, u64::from(top) << 16 | u64::from(bottom) << 48);
        set(base + 0x30, destination);
        let block = random.chance(2).then(|| random.below(5));
        let format = 31 + random.below(8);
        set(
            base + 0x40,
            surface_word(
                format,
                block,
                (u64::from(surface_size.0), u64::from(surface_size.1)),
            ),
        );
        set(
            base + 0x48,
            u64::from(surface_size.0 - 1) | u64::from(surface_size.1 - 1) << 14,
        );
        if random.chance(5) {
            for (offset, word) in [0x60, 0x68, 0x70, 0x78]
                .into_iter()
                .zip(random_matrix(random))
            {
                set(base + offset, word);
            }
        }
        let selections = random.below(5) << 32
            | random.below(7) << 36
            | random.below(4) << 40
            | random.below(4) << 44;
        let constants = random.below(1024) | random.below(1024) << 16;
        let overrides = (0..4).fold(0u64, |word, channel| {
            word | random.below(1024) << (2 + 10 * channel)
                | u64::from(random.chance(4)) << (44 + channel)
                | u64::from(random.chance(8)) << (48 + channel)
        });
        set(base + 0xA0, constants | selections);
        set(base + 0xA8, overrides);
    }
    let mut methods = [0; method::COUNT];
    for slot in 0..slots {
        methods[method::SURFACES.start + slot * 24] =
            ((SOURCE_AT + slot as u64 * 0x4_0000) >> 8) as u32;
    }
    for (method, address) in method::SET_OUTPUT_SURFACE.zip(PLANES_AT) {
        methods[method] = (address >> 8) as u32;
    }
    (config, methods)
}

/// Fills device addresses 0..4 MiB of `bench` with random bytes.
fn scramble(bench: &Bench, random: &mut Random) {
    let bytes: Vec<u8> = (0..0x40_0000).map(|_| random.next() as u8).collect();
    bench.poke(0, &bytes);
}

/// The composition through [`Fused`] writes the same bytes as the
/// per-pixel path, and refuses the same compositions, for random copies
/// and the compositions around them.
#[test]
fn fused_compositions_match_the_per_pixel_path() {
    let mut random = Random(0x5D5D_5D5D);
    let (mut composed, mut differing) = (0, 0);
    for case in 0..400 {
        let (config, methods) = random_composition(&mut random);
        let fused = Bench::new();
        scramble(&fused, &mut random);
        let general = Bench::new();
        general.poke(0, &fused.peek_bytes(0, 0x40_0000));
        let memory = |bench: &Bench| DeviceMemory::new(&bench.mc, &bench.ram, SMMU_VIC_ASID);
        let outcome = compose_config(&memory(&fused), &methods, &config, true);
        let expected = compose_config(&memory(&general), &methods, &config, false);
        assert_eq!(outcome, expected, "case {case}");
        let (after, reference) = (
            fused.peek_bytes(0, 0x40_0000),
            general.peek_bytes(0, 0x40_0000),
        );
        assert!(
            after == reference,
            "case {case}: {}",
            dump(&methods, &config)
        );
        composed += usize::from(outcome.is_ok());
        differing += usize::from(outcome.is_ok() && after != general.peek_bytes(0, 0));
    }
    assert!(composed > 150, "{composed} compositions ran");
    let _ = differing;
}

/// The writers before row-wise output: every pixel stored on its own.
fn reference_write(
    output: &Output,
    memory: &DeviceMemory,
    area: Rect,
    colors: &[[f32; 4]],
) -> Result<(), String> {
    let surface = match output {
        Output::Rgb(surface) => surface,
        Output::Planar(planar) => return reference_planar_write(planar, memory, area, colors),
    };
    let mut rows = surface.read(memory, area.top, area.bottom)?;
    let width = area.width() as usize;
    for y in area.top..area.bottom {
        for x in area.left..area.right {
            let color = colors[(y - area.top) as usize * width + (x - area.left) as usize];
            let at = (surface.offset(x, y) - rows.start) as usize;
            for (channel, index) in surface.format.rgb.into_iter().enumerate() {
                rows.bytes[at + index] = unorm8(color[channel]);
            }
            rows.bytes[at + surface.format.alpha] = unorm8(color[3]);
        }
    }
    memory
        .write(surface.address + rows.start, &rows.bytes)
        .ok_or_else(|| "output not mapped".to_string())
}

fn reference_planar_write(
    planar: &Planar,
    memory: &DeviceMemory,
    area: Rect,
    colors: &[[f32; 4]],
) -> Result<(), String> {
    let [sx, sy] = planar.subsampling;
    if [area.left, area.right].iter().any(|x| x % sx != 0)
        || [area.top, area.bottom].iter().any(|y| y % sy != 0)
    {
        return Err(format!(
            "target {area:?} splits the {sx}x{sy} chroma samples"
        ));
    }
    let chroma_area = Rect {
        left: area.left / sx,
        top: area.top / sy,
        right: area.right / sx,
        bottom: area.bottom / sy,
    };
    let width = area.width() as usize;
    let at = |x: u32, y: u32| colors[(y - area.top) as usize * width + (x - area.left) as usize];
    let taps = |axis: usize, k: u32| -> [(u32, f32); 2] {
        if planar.subsampling[axis] == 1 {
            return [(k, 1.0), (k, 0.0)];
        }
        let odd = planar.siting[axis] as f32 / 2.0;
        [(2 * k, 1.0 - odd), (2 * k + 1, odd)]
    };
    let chroma = |cx: u32, cy: u32, component: usize| {
        let mut sum = 0.0;
        for (y, wy) in taps(1, cy) {
            for (x, wx) in taps(0, cx) {
                if wx * wy != 0.0 {
                    sum += wx * wy * at(x, y)[component].clamp(0.0, 1.0);
                }
            }
        }
        sum
    };
    let fill = |rect: Rect, samples: u32, value: &dyn Fn(u32, u32) -> u8| {
        let row = (rect.width() * samples) as usize;
        let mut bytes = vec![0u8; row * rect.height() as usize];
        for (y, line) in (rect.top..).zip(bytes.chunks_mut(row)) {
            for (x, byte) in (rect.left * samples..).zip(line) {
                *byte = value(x, y);
            }
        }
        bytes
    };
    let mut planes = vec![(
        planar.luma,
        area,
        fill(area, 1, &|x, y| unorm8(at(x, y)[0])),
    )];
    match planar.chroma {
        ChromaPlanes::Interleaved { plane, u_first } => {
            let value = move |x: u32, y: u32| {
                let byte = (x % 2) as usize;
                let component = if u_first { 1 + byte } else { 2 - byte };
                unorm8(chroma(x / 2, y, component))
            };
            planes.push((plane, chroma_area, fill(chroma_area, 2, &value)));
        }
        ChromaPlanes::Separate(separate) => {
            for (index, plane) in separate.into_iter().enumerate() {
                let value = |x: u32, y: u32| unorm8(chroma(x, y, 1 + index));
                planes.push((plane, chroma_area, fill(chroma_area, 1, &value)));
            }
        }
    }
    let mut stored = Vec::with_capacity(planes.len());
    for (plane, rect, _) in &planes {
        stored.push(plane.read(memory, rect.top, rect.bottom)?);
    }
    for ((plane, rect, bytes), mut rows) in planes.into_iter().zip(stored) {
        let row = bytes.len() / rect.height() as usize;
        let left = rect.left * row as u32 / rect.width();
        for (y, line) in (rect.top..).zip(bytes.chunks(row)) {
            for (x, &byte) in (left..).zip(line) {
                let at = (plane.offset(x, y) - rows.start) as usize;
                rows.bytes[at] = byte;
            }
        }
        memory
            .write(plane.address + rows.start, &rows.bytes)
            .ok_or_else(|| format!("plane at {:#x} is not mapped", plane.address))?;
    }
    Ok(())
}

/// The row-wise output writers store what the per-pixel ones did, also
/// for components outside 0..1, in every output format and layout.
#[test]
fn row_wise_writers_match_the_per_pixel_writers() {
    let mut random = Random(0x0A7_u64 << 20 | 0x5D);
    let specials = [-0.5, -0.0, 0.0, 0.5 / 255.0, 0.5, 1.0, 1.5, f32::NAN];
    let mut written = 0;
    for case in 0..300 {
        let (words, size) = random_output(&mut random);
        let addresses = PLANES_AT.map(|address| (address >> 8) as u32);
        let output = Output::new(words, &addresses).unwrap();
        let area = Rect::inclusive(random_rect(&mut random, size, 2)).intersect(Rect {
            left: 0,
            top: 0,
            right: size.0,
            bottom: size.1,
        });
        if area.width() == 0 || area.height() == 0 {
            continue;
        }
        let colors: Vec<[f32; 4]> = (0..area.width() * area.height())
            .map(|_| {
                std::array::from_fn(|_| {
                    if random.chance(8) {
                        random.pick(&specials)
                    } else {
                        random.below(1 << 16) as f32 / 50_000.0 - 0.1
                    }
                })
            })
            .collect();
        let fast = Bench::new();
        scramble(&fast, &mut random);
        let reference = Bench::new();
        reference.poke(0, &fast.peek_bytes(0, 0x40_0000));
        let memory = |bench: &Bench| DeviceMemory::new(&bench.mc, &bench.ram, SMMU_VIC_ASID);
        let outcome = output.write(&memory(&fast), area, &colors);
        let expected = reference_write(&output, &memory(&reference), area, &colors);
        assert_eq!(outcome, expected, "case {case}");
        assert!(
            fast.peek_bytes(0, 0x40_0000) == reference.peek_bytes(0, 0x40_0000),
            "case {case}: {words:x?} {area:?}"
        );
        written += usize::from(outcome.is_ok());
    }
    assert!(written > 100, "{written}");
}
