use super::*;
use crate::{
    boot::DramSize,
    bus::MmioDevice,
    memory::GuestRam,
    soc::mc::{Mc, SMMU_NVENC_ASID},
};

/// The picture setup of grc's first EXECUTE with an application running (IDR
/// picture 0, 1280×720), its first 512 bytes as words.
const GRC_SETUP: [u32; 128] = [
    0xd0b70006, 0x02cf04ff, 0x05000500, 0x00000000, 0x00000000, 0x00000000, 0x00000e10, 0x00000000,
    0x00000080, 0x02cf04ff, 0x05000500, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000002, 0x02cf04ff, 0x05000500, 0x00000000, 0x00000000, 0x00000000, 0x00000e10, 0x00000000,
    0x00000080, 0x14492064, 0x00000100, 0x000000c0, 0x181c1801, 0x33000000, 0x03063333, 0x00000000,
    0x00000000, 0x00000000, 0x00001e00, 0x004c4b40, 0x00000000, 0x00000000, 0x004c4b40, 0x004c4b40,
    0x0000000f, 0x0000000e, 0x00000115, 0x00000000, 0x0000002e, 0x00000000, 0x00010000, 0x00010200,
    0x00000000, 0x00000000, 0xfefefefe, 0xfefefefe, 0xfefefefe, 0xfefefefe, 0xffffffff, 0xffffffff,
    0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
    0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
    0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
    0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
    0xffffffff, 0xffffffff, 0x00000000, 0x00000000, 0x00000e10, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000300, 0x00000600, 0x00000400, 0x00000500, 0x000ac800,
    0x00151800, 0x00000000, 0x0000009c, 0x00080000, 0x00000000, 0x00000080, 0x001000b0, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
];

fn setup_bytes(words: &[u32; 128]) -> [u8; SETUP_SIZE] {
    let mut bytes = [0; SETUP_SIZE];
    for (chunk, word) in bytes.chunks_exact_mut(4).zip(words) {
        chunk.copy_from_slice(&word.to_le_bytes());
    }
    bytes
}

#[test]
fn grcs_setup_reads_as_an_idr_picture_of_its_parameter_sets() {
    let setup = Setup::parse(&setup_bytes(&GRC_SETUP)).unwrap();
    assert_eq!((setup.input.width, setup.input.height), (1280, 720));
    assert_eq!(
        (setup.input.pitch, setup.input.pitch_chroma),
        (0x500, 0x500)
    );
    // Block-linear, 2-GOB blocks, semi-planar NV12.
    assert_eq!(setup.input.layout, 2);
    assert_eq!((setup.pic_struct(), setup.pic_type()), (0, 3));
    assert!(setup.reference() && setup.four_byte_start_codes());
    assert_eq!((setup.bitstream_size, setup.bitstream_start), (0x151800, 0));
    assert_eq!(
        setup.format(),
        Format {
            width: 1280,
            height: 720,
            cabac: true,
        }
    );
    assert_eq!(
        setup.rate().unwrap(),
        Rate {
            bit_rate: 5_000_000,
            frame_rate: Some(30.0),
        }
    );
    let driver = setup.driver();
    assert_eq!(driver.sps.log2_max_frame_num, 8);
    assert_eq!(driver.sps.poc_type, 2);
    assert_eq!(driver.sps.chroma_format_idc, 1);
    assert!(driver.sps.frame_mbs_only);
    assert!(driver.pps.cabac && driver.pps.transform_8x8_mode);
    assert!(driver.pps.deblocking_filter_control_present);
    assert_eq!(driver.pps.num_ref_idx_default, [1, 1]);
    assert_eq!((driver.pps.pic_init_qp, driver.pps_id), (26, 0));
    assert!(driver.idr && driver.reference);
}

const RAM_BASE: u64 = 0x8000_0000;
/// Device address 0 maps here (one 4 MiB SMMU section).
const WINDOW: u64 = RAM_BASE + 0x40_0000;
const SETUP: u64 = 0x1000;
const STATUS: u64 = 0x2000;
const BITSTREAM: u64 = 0x10_0000;
const LUMA: u64 = 0x20_0000;
const CHROMA: u64 = 0x30_0000;
const SYNCPOINT: usize = 14;
const WIDTH: u32 = 320;
const HEIGHT: u32 = 192;

struct Bench {
    ram: Arc<GuestRam>,
    syncpoints: Arc<Syncpoints>,
    notes: Arc<Notes>,
    nvenc: Nvenc,
}

impl Bench {
    fn new() -> Self {
        let ram = Arc::new(GuestRam::new(RAM_BASE, 0x80_0000).unwrap());
        let mc = Arc::new(Mc::new(DramSize::Gb4));
        mc.write(0, 0x1c, 4, 1).unwrap();
        mc.write(0, 0x20, 4, 0xe008_0000).unwrap();
        mc.write(0, SMMU_NVENC_ASID, 4, 0x8000_0001).unwrap();
        // Directory entry 0: a section (no next-level bit) at WINDOW.
        ram.write(RAM_BASE, &0xc008_0400u32.to_le_bytes()).unwrap();
        let syncpoints = Arc::new(Syncpoints::default());
        let notes: Arc<Notes> = Arc::default();
        let nvenc = Nvenc::new(
            DeviceMemory::new(&mc, &ram, SMMU_NVENC_ASID),
            Arc::clone(&syncpoints),
            Arc::clone(&notes),
        );
        nvenc.shared.state.lock().unwrap().started = true;
        Bench {
            ram,
            syncpoints,
            notes,
            nvenc,
        }
    }

    fn poke(&self, address: u64, bytes: &[u8]) {
        self.ram.write(WINDOW + address, bytes).unwrap();
    }

    fn peek(&self, address: u64, count: usize) -> Vec<u8> {
        let mut bytes = vec![0; count];
        self.ram.read(WINDOW + address, &mut bytes).unwrap();
        bytes
    }

    fn word(&self, address: u64) -> u32 {
        u32::from_le_bytes(self.peek(address, 4).try_into().unwrap())
    }

    /// Writes picture `index` (a gradient with a square moving across it)
    /// block-linear, 2-GOB blocks, and its setup: an IDR picture first,
    /// then P pictures, as grc sends them.
    fn picture(&self, index: u32) -> (Vec<u8>, Vec<u8>) {
        let luma: Vec<u8> = (0..WIDTH * HEIGHT)
            .map(|i| {
                let (x, y) = (i % WIDTH, i / WIDTH);
                let inside = (x.wrapping_sub(8 * index)) < 48 && (40..120).contains(&y);
                if inside { 235 } else { (x / 2 + y / 3) as u8 }
            })
            .collect();
        let chroma: Vec<u8> = (0..WIDTH * HEIGHT / 2)
            .map(|i| {
                if i % 2 == 0 {
                    (i % WIDTH / 3) as u8 + 40
                } else {
                    160
                }
            })
            .collect();
        let block = BlockLinear {
            block_height_log2: 1,
            block_depth_log2: 0,
        };
        for (address, plane, rows) in [(LUMA, &luma, HEIGHT), (CHROMA, &chroma, HEIGHT / 2)] {
            let surface = BlockLinearSurface::new(WIDTH, rows, block);
            let mut image = vec![0; surface.size(1) as usize];
            for (y, row) in plane.chunks(WIDTH as usize).enumerate() {
                for (x, run) in (0..WIDTH).step_by(16).zip(row.chunks(16)) {
                    let at = surface.offset(x, y as u32, 0) as usize;
                    image[at..at + 16].copy_from_slice(run);
                }
            }
            self.poke(address, &image);
        }
        let mut words = GRC_SETUP;
        // input_cfg: size and pitches.
        words[9] = (HEIGHT - 1) << 16 | (WIDTH - 1);
        words[10] = WIDTH << 16 | WIDTH;
        let pc = PIC_CONTROL / 4;
        // frame_num, pic_order_cnt_lsb (unused: POC type 2), idr_pic_id 0.
        words[pc + 45] = (2 * index) << 16 | index;
        // References all: an IDR picture, an I picture every 15 (grc's
        // `gop_length`), P pictures between.
        words[pc + 56] = pic_type(index) << 2 | 0x90;
        // grc's measured frame rate (8.8): 30, then 15, then none, as
        // when pictures stop coming during world creation.
        words[0x88 / 4] = match index {
            0..15 => 0x1e00,
            15..25 => 0xf00,
            _ => 0,
        };
        self.poke(SETUP, &setup_bytes(&words));
        (luma, chroma)
    }

    fn methods(&self, index: u32) -> Option<()> {
        let methods = [
            (0x700, 0x1200_1103),
            (0x704, index),
            (0x200, APPLICATION_H264),
            (0x710, (SETUP >> 8) as u32),
            (0x718, (STATUS >> 8) as u32),
            (0x71c, (BITSTREAM >> 8) as u32),
            (0x734, (LUMA >> 8) as u32),
            (0x740, (CHROMA >> 8) as u32),
            (0x300, 0x100),
        ];
        self.nvenc.thi_write(THI_CONT_SYNCPT_L1, 0)?;
        for (method, value) in methods {
            self.nvenc.thi_write(THI_METHOD_OFFSET, method >> 2)?;
            self.nvenc.thi_write(THI_METHOD_DATA, value)?;
        }
        self.nvenc
            .thi_write(THI_INCR_SYNCPT, 1 << 8 | SYNCPOINT as u32)
    }

    /// The bitstream EXECUTE wrote, by the status record's bit count.
    fn stream(&self) -> Vec<u8> {
        let bits = self.word(STATUS + status::TOTAL_BITS as u64);
        assert_eq!(bits % 8, 0);
        self.peek(BITSTREAM, bits as usize / 8)
    }
}

/// The picture type of picture `index`: 3 IDR, 2 I, 0 P.
fn pic_type(index: u32) -> u32 {
    match index {
        0 => 3,
        _ if index % 15 == 0 => 2,
        _ => 0,
    }
}

/// The driver's sequence and picture parameter sets as the setup
/// describes them (what grc writes, `0x211540`): High profile, level 3.2,
/// 8-bit frame_num, POC type 2, one reference frame; CABAC, 8x8 transform,
/// deblocking control present.
fn driver_sets() -> (Vec<u8>, Vec<u8>) {
    let mut sps = h264::Writer::default();
    sps.u(8, 100);
    sps.u(8, 0);
    sps.u(8, 32);
    sps.ue(0);
    sps.ue(1); // chroma_format_idc
    sps.ue(0);
    sps.ue(0);
    sps.u(2, 0); // qpprime, seq_scaling_matrix_present
    sps.ue(4); // log2_max_frame_num_minus4
    sps.ue(2); // pic_order_cnt_type
    sps.ue(1); // max_num_ref_frames
    sps.u(1, 0);
    sps.ue(WIDTH / 16 - 1);
    sps.ue(HEIGHT / 16 - 1);
    sps.u(4, 0b1100); // frame_mbs_only, direct_8x8_inference, no crop, no VUI
    let mut pps = h264::Writer::default();
    pps.ue(0);
    pps.ue(0);
    pps.u(2, 0b10); // CABAC, no bottom field POC
    pps.ue(0);
    pps.ue(0);
    pps.ue(0);
    pps.u(3, 0); // weighted prediction
    pps.se(0);
    pps.se(0);
    pps.se(0);
    pps.u(3, 0b100); // deblocking control, no constrained intra, no redundant
    pps.u(2, 0b10); // 8x8 transform, no scaling matrices
    pps.se(0);
    (
        h264::nal_unit(0x67, &sps.trailing()),
        h264::nal_unit(0x68, &pps.trailing()),
    )
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let error: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
        .sum::<f64>()
        / a.len() as f64;
    10.0 * (255.0f64.powi(2) / error.max(1e-9)).log10()
}

/// Pictures encoded the way grc submits them, decoded with the driver's
/// own parameter sets: the slices are valid under them, show the input,
/// and complete with the syncpoint and status record grc waits for.
#[test]
fn encoded_pictures_decode_under_the_drivers_parameter_sets() {
    let bench = Bench::new();
    let (sps, pps) = driver_sets();
    let mut decoder = crate::soc::nvdec::videotoolbox::Session::new(&sps, &[&pps]).unwrap();
    for index in 0..40 {
        let (luma, chroma) = bench.picture(index);
        bench.methods(index).unwrap();
        bench.nvenc.drain();
        assert_eq!(bench.notes.take(), Vec::<String>::new());
        assert_eq!(bench.syncpoints.get(SYNCPOINT), Some(index + 1));
        assert_eq!(bench.word(STATUS + status::PICTURE_INDEX as u64), index);
        let stream = bench.stream();
        assert_eq!(stream[..4], [0, 0, 0, 1]);
        // IDR first (nal_unit_type 5), then non-IDR slices, references.
        let kind = if index == 0 { 5 } else { 1 };
        assert_eq!(stream[4] & 0x1F, kind, "picture {index}");
        assert_ne!(stream[4] & 0x60, 0);
        let pic_type = u16::from_le_bytes(bench.peek(STATUS + 16, 2).try_into().unwrap());
        assert_eq!(u32::from(pic_type), self::pic_type(index));
        // As VideoToolbox takes it: each NAL unit after its length.
        let mut sample = Vec::new();
        let units: Vec<&[u8]> = split_start_codes(&stream);
        // The stream goes on through the frame rate changes: P pictures
        // stay P pictures (slice_type % 5 == 0 after first_mb_in_slice).
        let rbsp = h264::rbsp(&units[0][1..]);
        let mut slice = h264::Reader::new(&rbsp);
        slice.ue().unwrap();
        let slice_type = slice.ue().unwrap() % 5;
        assert_eq!(
            slice_type,
            if index % 15 == 0 { 2 } else { 0 },
            "picture {index}"
        );
        assert_eq!(
            units.len(),
            usize::from(u16::from_le_bytes(
                bench.peek(STATUS + 18, 2).try_into().unwrap()
            ))
        );
        for unit in units {
            sample.extend_from_slice(&(unit.len() as u32).to_be_bytes());
            sample.extend_from_slice(unit);
        }
        let frame = decoder.decode(&sample).unwrap();
        assert_eq!((frame.width, frame.height), (WIDTH, HEIGHT));
        let (y, c) = (psnr(&frame.luma, &luma), psnr(&frame.chroma, &chroma));
        assert!(
            y > 30.0 && c > 30.0,
            "picture {index}: PSNR {y:.1}/{c:.1} dB"
        );
    }
}

fn split_start_codes(stream: &[u8]) -> Vec<&[u8]> {
    let mut starts = Vec::new();
    let mut at = 0;
    while at + 4 <= stream.len() {
        if stream[at..at + 4] == [0, 0, 0, 1] {
            starts.push(at + 4);
            at += 4;
        } else {
            at += 1;
        }
    }
    starts
        .iter()
        .enumerate()
        .map(|(i, &start)| {
            let end = starts.get(i + 1).map_or(stream.len(), |next| next - 4);
            &stream[start..end]
        })
        .collect()
}

#[test]
fn only_the_known_firmware_starts_it_and_methods_wait_for_it() {
    let bench = Bench::new();
    bench.nvenc.shared.state.lock().unwrap().started = false;
    bench.nvenc.thi_write(THI_METHOD_OFFSET, 0x200 / 4).unwrap();
    assert_eq!(bench.nvenc.thi_write(THI_METHOD_DATA, 1), None);
    let zero = [0u8; 0x10000];
    assert!(
        bench
            .nvenc
            .start(&zero, &zero)
            .unwrap_err()
            .contains("unknown firmware")
    );
    bench.nvenc.shared.state.lock().unwrap().started = true;
    bench.nvenc.thi_write(THI_METHOD_DATA, 1).unwrap();
    assert_eq!(bench.nvenc.method_offset(), 0x200 / 4);
    // A method outside the class, and continuous increments.
    bench.nvenc.thi_write(THI_METHOD_OFFSET, 0x240 / 4).unwrap();
    assert_eq!(bench.nvenc.thi_write(THI_METHOD_DATA, 1), None);
    assert_eq!(bench.nvenc.thi_write(THI_CONT_SYNCPT_L1, 0x10e), None);
}

/// A picture the model cannot read stops the engine: the increment
/// queued behind it never counts, and the next THI access refuses.
#[test]
fn a_failed_picture_stops_the_engine_without_completing_its_fence() {
    let bench = Bench::new();
    bench.picture(0);
    // A good picture, then one whose input luma plane lies past the end
    // of the mapped window.
    bench.methods(0).unwrap();
    bench.nvenc.thi_write(THI_METHOD_OFFSET, 0x734 / 4).unwrap();
    bench.nvenc.thi_write(THI_METHOD_DATA, 0x4_0000).unwrap();
    bench.nvenc.thi_write(THI_METHOD_OFFSET, 0x300 / 4).unwrap();
    bench.nvenc.thi_write(THI_METHOD_DATA, 0x100).unwrap();
    bench
        .nvenc
        .thi_write(THI_INCR_SYNCPT, 1 << 8 | SYNCPOINT as u32)
        .unwrap();
    bench.nvenc.drain();
    assert_eq!(bench.syncpoints.get(SYNCPOINT), Some(1));
    let notes = bench.notes.take();
    assert!(notes[0].contains("input luma"), "{notes:?}");
    assert_eq!(bench.nvenc.thi_write(THI_METHOD_OFFSET, 0), None);
    assert!(bench.notes.take()[0].contains("stopped earlier"));
}

#[test]
fn a_setup_of_another_class_version_is_refused_at_execute() {
    let bench = Bench::new();
    bench.picture(0);
    let mut words = GRC_SETUP;
    words[0] = 0xc1b7_0006;
    bench.poke(SETUP, &setup_bytes(&words));
    assert_eq!(bench.methods(0), None);
    assert!(bench.notes.take()[0].contains("magic"));
}

#[test]
fn the_state_survives_a_snapshot_and_the_encoder_starts_over() {
    let bench = Bench::new();
    bench.picture(0);
    bench.methods(0).unwrap();
    let saved = bench.nvenc.save().unwrap();
    assert_eq!(bench.syncpoints.get(SYNCPOINT), Some(1));
    let other = Bench::new();
    other.nvenc.restore(&saved).unwrap();
    assert_eq!(other.nvenc.method_offset(), bench.nvenc.method_offset());
    // A P picture first after the restore: written as an I picture.
    other.picture(1);
    other.methods(1).unwrap();
    other.nvenc.drain();
    assert_eq!(other.notes.take(), Vec::<String>::new());
    let stream = other.stream();
    assert_eq!(stream[4] & 0x1F, 1);
}
