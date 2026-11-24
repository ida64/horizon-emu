//! Video encoder (NVENC 5.0, host1x class 0x21): its THI registers, its
//! methods, the firmware nvservices boots it with, and H.264 encoding by
//! the host's encoder.
//!
//! # Firmware
//!
//! nvservices 23.5.0's engine configuration for NVENC (`0xee170`: aperture
//! 0x544C0000, 0x40000 bytes, class 0x21, firmware `nvhost_nvenc050.fw`)
//! names the host falcon routines the VIC's uses: `0x80bf0` looks the
//! firmware up in the embedded table (`0x942d0`, entry `0x199040`:
//! 0x1E000 bytes at `0xf4000`), checks the bin header (magic 0x10DE,
//! version 1) and pins it; `0x80990` boots it with `0x809e0`, the transfer
//! the [`HostFalcon`](super::falcon_host::HostFalcon) models (the image's
//! data segment, 0x200 bytes at 0x1CD00 past the OS data at 0x200, to DMEM
//! 0; the first 256 bytes of its code segment to IMEM 0). Only that
//! firmware is known: the start is refused unless IMEM holds exactly what
//! it puts there ([`IMEM_SHA256`]; its data segment is zeros, so DMEM is
//! all zero, [`DMEM_SHA256`]). No Falcon instruction is executed; the
//! encoder it would run is the model below.
//!
//! # Methods
//!
//! grc (`0100000000000035`, 23.5.0) records video for applications that
//! allow capture: nvnflinger's frames go through the VIC to
//! NV12 and then, 30 times a second, to NVENC. Its H.264 submission
//! (`0x214740`) sends through THI words 0x10/0x11 (Linux
//! `hw_host1x04_uclass.h`, as for the VIC and NVDEC):
//! SET_CONTROL_PARAMS 0x700 = 0x12001103 (codec 3, H.264; FORCE_OUT_PIC;
//! GPTIMER_ON; RC mode 0x12), SET_PICTURE_INDEX 0x704, SET_APPLICATION_ID
//! 0x200 = 1, SET_IN_DRV_PIC_SETUP 0x710, SET_OUT_ENC_STATUS 0x718,
//! SET_OUT_BITSTREAM 0x71C, SET_IOHISTORY 0x720, SET_IO_RC_PROCESS 0x724,
//! SET_OUT_REF_PIC 0x730, SET_IN_CUR_PIC 0x734, SET_IN_CUR_PIC_CHROMA_U
//! 0x740, after the first picture SET_IN_REF_PIC0 0x400 and the motion
//! estimation predictor buffers 0x738/0x73C, then EXECUTE 0x300 = 0x100
//! and a THI syncpoint increment with condition 1 (operation done). It
//! first writes THI word 0xB = 0, `THI_CONT_SYNCPT_L1` by the THI's
//! register list (word 0x0A `CONT_SYNCPT_EOF`) \[inference: recalled
//! from the Tegra THI register names\]: no continuous syncpoint
//! increments, the only value accepted. Method names and offsets are
//! NVIDIA's public `open-gpu-doc` `classes/video/clc5b7.h`
//! (`NVC5B7_VIDEO_ENCODER`, a later NVENC whose method layout is taken to
//! be NVENC 5.0's \[assumption\], the values seen agreeing with it).
//! Buffer addresses are IOVA >> 8 in NVENC's SMMU address space.
//!
//! # Picture setup
//!
//! `nvenc_h264_drv_pic_setup_s` of `open-gpu-doc` `nvenc_drv.h`, whose
//! magic for NVENC 5.0 (`NV_NVENC_5_0`) is 0xD0B70006, what grc's version
//! table (`0x1fb940`, `qword_30B304`) gives this chip and its pictures
//! carry. Read: the input surface configuration (1280×720, pitch 0x500,
//! block-linear with 2-GOB blocks, NV12), the sequence and picture
//! parameter set data (High profile, POC type 2, 8-bit frame_num; CABAC,
//! 8x8 transform, deblocking control present), the rate control's frame
//! rate (8.8 fixed point, 0x1E00 = 30) and bit rate (VCL 5 Mbit/s), and of
//! the picture control the frame number, POC LSB, IDR picture ID, the
//! bitstream buffer's size and start position, picture structure and type
//! (IDR, then P pictures), reference flag and 4-byte start codes.
//!
//! # Encoding
//!
//! EXECUTE reads the input picture (block-linear NV12, the block height
//! of the input configuration in GOBs: the VIC's output of the same
//! submission chain, 2-GOB blocks) and encodes it with VideoToolbox
//! (`videotoolbox`). The driver writes the sequence and picture parameter
//! sets itself (grc `0x211540` puts them before the picture's data); NVENC
//! writes the slices, so the host encoder's slices get headers written
//! again for the driver's sets (`h264`). Into the bitstream buffer
//! (SET_OUT_BITSTREAM, from the setup's start position) go the slice NAL
//! units, each after a start code. The reconstructed picture
//! (SET_OUT_REF_PIC), the history, rate control and motion estimation
//! buffers are NVENC's own between pictures and are not written: the host
//! encoder keeps its own reference picture.
//!
//! # Rate control
//!
//! The rate control's frame rate is not fixed: grc's encoder measures it
//! from the input pictures' timestamps (`CalculateDynamicFrameRate`,
//! `0x1f8b90`: `(u32)(10000000.0 / average_delta + 0.5)`, timestamps in
//! 100 ns) and hands each new value to the H.264 encoder, so the setup's
//! frame rate follows the application's: 30 at first, 15 later, and 0 once
//! pictures come more than 2 s apart (a long load),
//! with the per-picture rate `R` (+0x40 of `nvenc_h264_rc_s`) then
//! 0x7FFFFFFF. Every picture is still encoded. The bit rate and frame
//! rate are set on the running host session when they change (a new
//! session would restart the stream at a key picture); a frame rate of 0
//! gives the rate control no per-picture budget, which here leaves the
//! session's last frame rate in place \[inference: VideoToolbox's frame
//! rate is a hint, the bit rate target stays\].
//!
//! # Status
//!
//! The status record (SET_OUT_ENC_STATUS, `nvenc_pic_stat_s`, 128 bytes,
//! then `nvenc_slice_stat_s` a slice). grc reads it after the syncpoint
//! (`0x20b100`): `total_bit_count` (+8) is the size of what it copies from
//! the bitstream buffer (`>> 3` bytes), `pic_type` (+16) whether the
//! picture is a key frame, `avgQP` (+22), the QP range (+68, +70), and the
//! picture index (+0) for the first picture's parameter sets. Written:
//! picture index, no error, the bits written (start codes included, as
//! grc copies them), the bits without start codes (`type1_bit_count`),
//! picture type, slices, the slice QP as average, minimum and maximum,
//! the start position and last byte written, and per slice its offset,
//! size and header bits. Activity, macroblock counts, costs, motion
//! averages, cycles and the HRD fullness are zero: not measured.
//!
//! # Time
//!
//! The encoding runs on a thread of its own, not on the processor that
//! wrote the method: EXECUTE queues the picture (the method state and the
//! picture setup as they are then), and the syncpoint increments that
//! wait for the operation (conditions 1-3) go into the same queue after
//! it; an immediate one (condition 0) counts at once. So the increment
//! comes when the picture is in the bitstream buffer, as on the chip, and
//! the game's processors never wait for the host encoder. The queue holds
//! [`QUEUE`] items; a submission beyond that waits in the method write
//! as for a full method FIFO.
//!
//! A picture the model cannot encode stops the engine: the note says
//! why, its increments are never counted, and the next THI access
//! refuses. No fence completes for work not done.

mod h264;
mod videotoolbox;

use std::sync::{Arc, Condvar, Mutex, mpsc};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{Notes, falcon_host::Engine, host1x::Syncpoints, mc::DeviceMemory};
use crate::{
    maxwell::texture::layout::{BlockLinear, BlockLinearSurface},
    snapshot::{Snapshot, SnapshotError, decode, encode},
};
use videotoolbox::{Encoder, Format, Rate};

/// Host1x class of NVENC (nvservices' engine configuration `0xee170`;
/// Linux `include/linux/host1x.h` `HOST1X_CLASS_NVENC`).
pub const CLASS: u32 = 0x21;

/// SHA-256 of the 64 KiB IMEM once nvservices 23.5.0 has loaded
/// `nvhost_nvenc050.fw` (its first 256 code bytes at 0, zero after).
pub const IMEM_SHA256: &str = "be7a7355f248f9f08879f8481075f223fd9cfe4cd928f87616ad864da27a3d2a";
/// SHA-256 of the 64 KiB DMEM then: the image's 0x200-byte data segment
/// is zeros, so this is the digest of 64 KiB of zeros.
pub const DMEM_SHA256: &str = "de2f256064a0af797747c2b97505dc0b9f3df0de4f489eac731c23ae9ca9cc31";

/// THI registers, word offsets.
const THI_INCR_SYNCPT: u32 = 0x00;
const THI_CONT_SYNCPT_L1: u32 = 0x0B;
const THI_METHOD_OFFSET: u32 = 0x10;
const THI_METHOD_DATA: u32 = 0x11;

/// Conditions of a THI syncpoint increment (Linux `hw_host1x04_uclass.h`):
/// immediate, operation done, read done, register write safe.
const INCREMENT_CONDITIONS: u32 = 4;

/// Items the encoding thread's queue holds.
const QUEUE: usize = 8;

/// Method word indices (byte offset / 4), `clc5b7.h` names.
mod method {
    pub const SET_APPLICATION_ID: usize = 0x200 / 4;
    pub const SET_WATCHDOG_TIMER: usize = 0x204 / 4;
    pub const EXECUTE: usize = 0x300 / 4;
    /// `SET_IN_REF_PIC0_LUMA` 0x400 to `SET_IN_REF_PIC15_CHROMA` 0x47C.
    pub const REFERENCES: std::ops::RangeInclusive<usize> = 0x400 / 4..=0x47C / 4;
    /// `SET_CONTROL_PARAMS` 0x700 to `SET_IN_MV_HINTS_TASK_STATUS` 0x764.
    pub const SETUP: std::ops::RangeInclusive<usize> = 0x700 / 4..=0x764 / 4;
    pub const SET_CONTROL_PARAMS: usize = 0x700 / 4;
    pub const SET_PICTURE_INDEX: usize = 0x704 / 4;
    pub const SET_IN_DRV_PIC_SETUP: usize = 0x710 / 4;
    pub const SET_OUT_ENC_STATUS: usize = 0x718 / 4;
    pub const SET_OUT_BITSTREAM: usize = 0x71C / 4;
    pub const SET_IN_CUR_PIC: usize = 0x734 / 4;
    pub const SET_IN_CUR_PIC_CHROMA_U: usize = 0x740 / 4;
    pub const COUNT: usize = 0x800 / 4;
}

/// `SET_APPLICATION_ID` of H.264 (`NVC5B7_SET_APPLICATION_ID_ID_NVENC_H264`)
/// and `SET_CONTROL_PARAMS` `CODEC_TYPE` of H.264.
const APPLICATION_H264: u32 = 1;
const CODEC_H264: u32 = 3;
/// `SET_CONTROL_PARAMS` bits that change nothing here: the codec type,
/// FORCE_OUT_PIC (8: the reconstructed picture, not written, see above),
/// GPTIMER_ON (12, the watchdog) and the RC mode (31:24, the rate control
/// is the host encoder's).
const CONTROL_KNOWN: u32 = 0xF | 1 << 8 | 1 << 12 | 0xFF << 24;

/// `nvenc_h264_drv_pic_setup_s`.
const SETUP_SIZE: usize = 512;
const MAGIC: u32 = 0xD0B7_0006;

/// The picture setup, the parts read (offsets of `nvenc_drv.h`).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Setup {
    /// `input_cfg`.
    input: SurfaceConfig,
    sps: u32,
    pps: [u32; 2],
    hrd_type: u8,
    frame_rate: i32,
    bit_rates: [i32; 2],
    delta_poc_bottom: i32,
    frame_num: u16,
    poc_lsb: u16,
    idr_pic_id: u16,
    bitstream_size: u32,
    bitstream_start: u32,
    /// `pic_struct`, `pic_type`, `ref_pic_flag`...
    flags: u32,
    /// `codec`, `e4byteStartCode`...
    codec: u8,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct SurfaceConfig {
    width: u32,
    height: u32,
    pitch: u32,
    pitch_chroma: u32,
    transform: u16,
    luma_offset: u32,
    chroma_offset: u32,
    /// `block_height`, `tiled_16x16`, `memory_mode`, `nv21_enable`,
    /// `input_bl_mode`.
    layout: u32,
}

/// `pic_control` within the setup.
const PIC_CONTROL: usize = 0xC8;

impl Setup {
    fn parse(bytes: &[u8; SETUP_SIZE]) -> Result<Setup, String> {
        let u32_at = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        let u16_at = |at: usize| u16::from_le_bytes(bytes[at..at + 2].try_into().unwrap());
        let magic = u32_at(0);
        if magic != MAGIC {
            return Err(format!(
                "picture setup magic {magic:#x}, not NVENC 5.0's {MAGIC:#x}"
            ));
        }
        let surface = |at: usize| SurfaceConfig {
            width: u32::from(u16_at(at)) + 1,
            height: u32::from(u16_at(at + 2)) + 1,
            pitch: u16_at(at + 4).into(),
            pitch_chroma: u16_at(at + 6).into(),
            transform: u16_at(at + 8),
            luma_offset: u32_at(at + 0xC),
            chroma_offset: u32_at(at + 0x14),
            layout: u32_at(at + 0x1C),
        };
        let pc = PIC_CONTROL;
        Ok(Setup {
            input: surface(0x24),
            sps: u32_at(0x64),
            pps: [u32_at(0x68), u32_at(0x6C)],
            hrd_type: bytes[0x70],
            frame_rate: u32_at(0x70 + 24) as i32,
            bit_rates: [u32_at(0x70 + 44) as i32, u32_at(0x70 + 36) as i32],
            delta_poc_bottom: u32_at(pc + 176) as i32,
            frame_num: u16_at(pc + 180),
            poc_lsb: u16_at(pc + 182),
            idr_pic_id: u16_at(pc + 184),
            bitstream_size: u32_at(pc + 216),
            bitstream_start: u32_at(pc + 220),
            flags: u32_at(pc + 224),
            codec: bytes[pc + 230],
        })
    }

    fn pic_struct(&self) -> u32 {
        self.flags & 3
    }

    /// 0 P, 1 B, 2 I, 3 IDR.
    fn pic_type(&self) -> u32 {
        self.flags >> 2 & 3
    }

    fn reference(&self) -> bool {
        self.flags >> 4 & 1 != 0
    }

    fn four_byte_start_codes(&self) -> bool {
        self.codec >> 3 & 1 != 0
    }

    /// The driver's sequence and picture parameter sets as far as they
    /// are given (`nvenc_h264_sps_data_s`, `nvenc_h264_pps_data_s`), and
    /// this picture's header values.
    fn driver(&self) -> h264::Driver {
        let field = |word: u32, low: u32, width: u32| word >> low & ((1 << width) - 1);
        let signed = |word: u32, low: u32, width: u32| {
            let value = field(word, low, width) as i32;
            value << (32 - width) >> (32 - width)
        };
        let (s, [p0, p1]) = (self.sps, self.pps);
        let qp_offset = signed(p0, 27, 5);
        h264::Driver {
            sps: h264::Sps {
                chroma_format_idc: field(s, 16, 2),
                bit_depth: [8, 8],
                log2_max_frame_num: field(s, 20, 4) + 4,
                poc_type: field(s, 18, 2),
                log2_max_poc_lsb: field(s, 24, 4) + 4,
                delta_pic_order_always_zero: false,
                frame_mbs_only: field(s, 28, 1) != 0,
            },
            pps: h264::Pps {
                cabac: field(p0, 8, 1) != 0,
                bottom_field_pic_order_present: field(p1, 8, 1) != 0,
                num_ref_idx_default: [field(p0, 9, 5) + 1, field(p0, 14, 5) + 1],
                weighted_pred: field(p1, 9, 1) != 0,
                weighted_bipred_idc: field(p0, 19, 2),
                pic_init_qp: 26 + signed(p0, 21, 6),
                chroma_qp_index_offset: [qp_offset, signed(p1, 0, 5)],
                deblocking_filter_control_present: field(p1, 6, 1) != 0,
                constrained_intra_pred: field(p1, 5, 1) != 0,
                // Not in `nvenc_h264_pps_data_s`: NVENC writes no
                // redundant pictures [inference].
                redundant_pic_cnt_present: false,
                transform_8x8_mode: field(p1, 7, 1) != 0,
            },
            pps_id: field(p0, 0, 8),
            idr: self.pic_type() == 3,
            reference: self.reference(),
            frame_num: self.frame_num.into(),
            poc_lsb: self.poc_lsb.into(),
            delta_poc_bottom: self.delta_poc_bottom,
            idr_pic_id: self.idr_pic_id.into(),
        }
    }

    /// What the host encoder is made for: the picture in whole
    /// macroblocks (NVENC encodes the padded input, `nvenc_drv.h` 2.1)
    /// and the driver's entropy coder.
    fn format(&self) -> Format {
        Format {
            width: self.input.width.next_multiple_of(16),
            height: self.input.height.next_multiple_of(16),
            cabac: self.pps[0] >> 8 & 1 != 0,
        }
    }

    /// The rate control's bit rate and frame rate. A frame rate of 0 is
    /// none: see "Rate control" in the module.
    fn rate(&self) -> Result<Rate, String> {
        if self.frame_rate < 0 {
            return Err(format!("frame rate {:#x}", self.frame_rate));
        }
        // `hrd_type`: VCL (1) or NAL (2) HRD, whose bit rate it is.
        let bit_rate = match self.hrd_type {
            1 => self.bit_rates[0],
            2 => self.bit_rates[1],
            other => return Err(format!("HRD type {other}")),
        };
        if bit_rate <= 0 {
            return Err(format!("bit rate {bit_rate}"));
        }
        Ok(Rate {
            bit_rate: bit_rate as u32,
            frame_rate: (self.frame_rate > 0).then(|| f64::from(self.frame_rate) / 256.0),
        })
    }
}

/// Words of `nvenc_pic_stat_s` written, by byte offset.
mod status {
    pub const PICTURE_INDEX: usize = 0;
    pub const ERROR: usize = 4;
    pub const TOTAL_BITS: usize = 8;
    pub const TYPE1_BITS: usize = 12;
    pub const PIC_TYPE: usize = 16;
    pub const SLICES: usize = 18;
    pub const AVERAGE_QP: usize = 22;
    pub const START: usize = 32;
    pub const LAST_VALID: usize = 36;
    pub const MIN_QP: usize = 68;
    pub const MAX_QP: usize = 70;
    pub const SIZE: usize = 128;
    /// `nvenc_slice_stat_s`: offset, size, header bits, 4 reserved.
    pub const SLICE_SIZE: usize = 16;
}

/// The NVENC engine behind the THI.
pub struct Nvenc {
    shared: Arc<Shared>,
    queue: mpsc::SyncSender<Job>,
}

struct Shared {
    memory: DeviceMemory,
    syncpoints: Arc<Syncpoints>,
    notes: Arc<Notes>,
    state: Mutex<State>,
    /// Items queued and not yet done.
    pending: Mutex<usize>,
    done: Condvar,
}

#[derive(Serialize, Deserialize)]
struct State {
    offset: u32,
    #[serde(with = "crate::snapshot::array")]
    methods: [u32; method::COUNT],
    /// Whether the firmware was started.
    started: bool,
    /// Why the engine stopped, if it did.
    failure: Option<String>,
}

enum Job {
    Execute {
        methods: Box<[u32; method::COUNT]>,
        setup: Setup,
    },
    Increment(usize),
    /// A restored machine: the host encoder's pictures are not the
    /// stream's any more.
    Reset,
}

impl Nvenc {
    /// `memory` is memory as NVENC's SMMU client reaches it.
    pub fn new(memory: DeviceMemory, syncpoints: Arc<Syncpoints>, notes: Arc<Notes>) -> Self {
        let shared = Arc::new(Shared {
            memory,
            syncpoints,
            notes,
            state: Mutex::new(State {
                offset: 0,
                methods: [0; method::COUNT],
                started: false,
                failure: None,
            }),
            pending: Mutex::new(0),
            done: Condvar::new(),
        });
        let (queue, jobs) = mpsc::sync_channel(QUEUE);
        let worker = Arc::clone(&shared);
        std::thread::Builder::new()
            .name("NVENC".into())
            .spawn(move || worker.run(jobs))
            .expect("NVENC encoding thread");
        Self { shared, queue }
    }

    fn submit(&self, job: Job) {
        *self.shared.pending.lock().unwrap() += 1;
        // The thread ends only when this sender is dropped.
        self.queue.send(job).expect("NVENC encoding thread");
    }

    /// Waits until everything queued is done.
    fn drain(&self) {
        let pending = self.shared.pending.lock().unwrap();
        drop(
            self.shared
                .done
                .wait_while(pending, |pending| *pending > 0)
                .unwrap(),
        );
    }

    fn method(&self, value: u32) -> Option<()> {
        let mut state = self.shared.state.lock().unwrap();
        if !state.started {
            return self.refuse(format!(
                "method {:#x} = {value:#x} before the firmware was started",
                state.offset * 4
            ));
        }
        let index = state.offset as usize;
        if index == method::EXECUTE {
            let methods = Box::new(state.methods);
            drop(state);
            let setup = match self.setup(&methods, value) {
                Ok(setup) => setup,
                Err(reason) => {
                    return self
                        .refuse(format!("EXECUTE {value:#x}: {reason}; {}", dump(&methods)));
                }
            };
            self.submit(Job::Execute { methods, setup });
            return Some(());
        }
        let accepted = matches!(
            index,
            method::SET_APPLICATION_ID | method::SET_WATCHDOG_TIMER
        ) || method::REFERENCES.contains(&index)
            || method::SETUP.contains(&index);
        if !accepted {
            return self.refuse(format!("method {:#x} = {value:#x}", index * 4));
        }
        state.methods[index] = value;
        Some(())
    }

    /// What EXECUTE takes from the processor's side: the methods it
    /// encodes with and the picture setup as it is now.
    fn setup(&self, methods: &[u32; method::COUNT], execute: u32) -> Result<Setup, String> {
        // EXECUTE's NOTIFY and AWAKEN bits (0, 8) concern the falcon's
        // interrupt to a driver that waits on it; the syncpoint
        // increment that follows is what grc waits on.
        if execute & !0x101 != 0 {
            return Err("EXECUTE flags".into());
        }
        let application = methods[method::SET_APPLICATION_ID];
        let control = methods[method::SET_CONTROL_PARAMS];
        if application != APPLICATION_H264 || control & 0xF != CODEC_H264 {
            return Err(format!(
                "application {application}, control {control:#x}: only H.264 is modeled"
            ));
        }
        if control & !CONTROL_KNOWN != 0 {
            return Err(format!("control parameters {control:#x}"));
        }
        let address = u64::from(methods[method::SET_IN_DRV_PIC_SETUP]) << 8;
        let mut bytes = [0; SETUP_SIZE];
        self.shared
            .memory
            .read(address, &mut bytes)
            .ok_or_else(|| format!("picture setup at {address:#x} is not mapped"))?;
        let setup = Setup::parse(&bytes)?;
        if setup.pic_struct() != 0 || setup.pic_type() == 1 {
            return Err(format!(
                "picture structure {} type {}: only frames, I and P pictures are modeled",
                setup.pic_struct(),
                setup.pic_type()
            ));
        }
        Ok(setup)
    }

    fn refuse(&self, reason: String) -> Option<()> {
        self.shared.notes.add(format!("NVENC: {reason}"));
        None
    }
}

impl Shared {
    /// The encoding thread: runs the queue until the engine goes.
    fn run(&self, jobs: mpsc::Receiver<Job>) {
        let mut encoder: Option<Encoder> = None;
        for job in jobs {
            let stopped = self.state.lock().unwrap().failure.is_some();
            match job {
                Job::Reset => encoder = None,
                _ if stopped => {}
                Job::Increment(index) => self.syncpoints.add(index, 1),
                Job::Execute { methods, setup } => {
                    if let Err(reason) = self.execute(&mut encoder, &methods, &setup) {
                        let reason = format!("EXECUTE: {reason}; {}", dump(&methods));
                        self.notes.add(format!("NVENC: {reason}"));
                        self.state.lock().unwrap().failure = Some(reason);
                        encoder = None;
                    }
                }
            }
            let mut pending = self.pending.lock().unwrap();
            *pending -= 1;
            if *pending == 0 {
                self.done.notify_all();
            }
        }
    }

    /// Encodes the picture `methods` and `setup` describe.
    fn execute(
        &self,
        encoder: &mut Option<Encoder>,
        methods: &[u32; method::COUNT],
        setup: &Setup,
    ) -> Result<(), String> {
        let (format, rate) = (setup.format(), setup.rate()?);
        let (luma, chroma) = self.picture(methods, setup, &format)?;
        if encoder.as_ref().is_none_or(|e| e.format() != format) {
            *encoder = Some(Encoder::new(format)?);
        }
        let encoder = encoder.as_mut().unwrap();
        encoder.set_rate(rate)?;
        let driver = setup.driver();
        let encoded = encoder.encode(&luma, &chroma, setup.pic_type() >= 2)?;
        let sps = h264::parse_sps(&h264::rbsp(encoded.sps.get(1..).unwrap_or_default()))?;
        let pps = h264::parse_pps(&h264::rbsp(encoded.pps.get(1..).unwrap_or_default()))?;
        h264::check((&sps, &pps), &driver)?;
        let start_code: &[u8] = if setup.four_byte_start_codes() {
            &[0, 0, 0, 1]
        } else {
            &[0, 0, 1]
        };
        let mut stream = Vec::new();
        let mut slices = Vec::new();
        for nal in &encoded.nal_units {
            match nal.first().map(|header| header & 0x1F) {
                Some(h264::NAL_SLICE | h264::NAL_IDR_SLICE) => {}
                // Parameter sets, SEI, delimiters: the driver writes its
                // own.
                _ => continue,
            }
            let slice = h264::rewrite(nal, &sps, &pps, &driver)?;
            stream.extend_from_slice(start_code);
            let offset = stream.len() - start_code.len();
            stream.extend_from_slice(&slice.nal_unit);
            slices.push((offset, stream.len() - offset, slice));
        }
        if slices.is_empty() {
            return Err("the host encoder gave no slice".into());
        }
        let room = setup.bitstream_size.saturating_sub(setup.bitstream_start) as usize;
        if stream.len() > room {
            return Err(format!(
                "{} bytes for a bitstream buffer of {room}",
                stream.len()
            ));
        }
        let bitstream =
            (u64::from(methods[method::SET_OUT_BITSTREAM]) << 8) + u64::from(setup.bitstream_start);
        self.write(bitstream, &stream, "bitstream")?;
        let mut record = vec![0u8; status::SIZE + status::SLICE_SIZE * slices.len()];
        let mut put = |at: usize, bytes: &[u8]| record[at..at + bytes.len()].copy_from_slice(bytes);
        let qps = slices.iter().map(|(_, _, slice)| slice.qp);
        let (low, high) = (qps.clone().min().unwrap(), qps.clone().max().unwrap());
        let average = qps.sum::<i32>() / slices.len() as i32;
        let start_codes = start_code.len() * slices.len();
        put(
            status::PICTURE_INDEX,
            &methods[method::SET_PICTURE_INDEX].to_le_bytes(),
        );
        put(status::ERROR, &0u32.to_le_bytes());
        put(status::TOTAL_BITS, &(8 * stream.len() as u32).to_le_bytes());
        put(
            status::TYPE1_BITS,
            &(8 * (stream.len() - start_codes) as u32).to_le_bytes(),
        );
        put(status::PIC_TYPE, &(setup.pic_type() as u16).to_le_bytes());
        put(status::SLICES, &(slices.len() as u16).to_le_bytes());
        put(status::AVERAGE_QP, &(average as u16).to_le_bytes());
        put(status::START, &setup.bitstream_start.to_le_bytes());
        let last = setup.bitstream_start + stream.len() as u32 - 1;
        put(status::LAST_VALID, &last.to_le_bytes());
        put(status::MIN_QP, &(low as u16).to_le_bytes());
        put(status::MAX_QP, &(high as u16).to_le_bytes());
        for (index, (offset, size, slice)) in slices.iter().enumerate() {
            let at = status::SIZE + status::SLICE_SIZE * index;
            put(at, &(*offset as u32).to_le_bytes());
            put(at + 4, &(*size as u32).to_le_bytes());
            put(at + 8, &slice.header_bits.to_le_bytes());
        }
        let status = u64::from(methods[method::SET_OUT_ENC_STATUS]) << 8;
        self.write(status, &record, "status record")
    }

    /// The input picture as the host encoder takes it: `format`'s size,
    /// luma rows, then rows of Cb/Cr pairs.
    fn picture(
        &self,
        methods: &[u32; method::COUNT],
        setup: &Setup,
        format: &Format,
    ) -> Result<(Vec<u8>, Vec<u8>), String> {
        let input = &setup.input;
        let layout = input.layout;
        let gobs = layout & 0x7F;
        let (tiled, memory_mode, nv21, bl_mode) = (
            layout >> 7 & 1,
            layout >> 8 & 3,
            layout >> 10 & 1,
            layout >> 11 & 3,
        );
        if tiled != 0 || memory_mode != 0 || bl_mode != 0 || !gobs.is_power_of_two() {
            return Err(format!(
                "input surface layout {layout:#x}: only semi-planar block-linear (GPU \
                 blocks) is modeled"
            ));
        }
        if input.transform != 0 {
            return Err(format!("input surface transform {}", input.transform));
        }
        let (width, height) = (format.width, format.height);
        if input.pitch < width || input.pitch_chroma < width {
            return Err(format!(
                "input pitches {}/{} for {width} bytes a row",
                input.pitch, input.pitch_chroma
            ));
        }
        let block = BlockLinear {
            block_height_log2: gobs.trailing_zeros(),
            block_depth_log2: 0,
        };
        let plane = |method: usize, offset: u32, pitch: u32, rows: u32, what: &str| {
            let address = (u64::from(methods[method]) + u64::from(offset)) << 8;
            let surface = BlockLinearSurface::new(pitch, rows, block);
            let mut image = vec![0u8; surface.size(1) as usize];
            self.read(address, &mut image, what)?;
            let mut out = vec![0u8; (width * rows) as usize];
            for (y, row) in out.chunks_exact_mut(width as usize).enumerate() {
                for (x, run) in (0..width).step_by(16).zip(row.chunks_mut(16)) {
                    let at = surface.offset(x, y as u32, 0) as usize;
                    run.copy_from_slice(&image[at..at + run.len()]);
                }
            }
            Ok::<_, String>(out)
        };
        let luma = plane(
            method::SET_IN_CUR_PIC,
            input.luma_offset,
            input.pitch,
            height,
            "input luma",
        )?;
        let mut chroma = plane(
            method::SET_IN_CUR_PIC_CHROMA_U,
            input.chroma_offset,
            input.pitch_chroma,
            height / 2,
            "input chroma",
        )?;
        if nv21 != 0 {
            for pair in chroma.as_chunks_mut::<2>().0 {
                pair.swap(0, 1);
            }
        }
        Ok((luma, chroma))
    }

    fn read(&self, address: u64, buffer: &mut [u8], what: &str) -> Result<(), String> {
        self.memory
            .read(address, buffer)
            .ok_or_else(|| format!("{what} at {address:#x} is not mapped"))
    }

    fn write(&self, address: u64, data: &[u8], what: &str) -> Result<(), String> {
        self.memory
            .write(address, data)
            .ok_or_else(|| format!("{what} at {address:#x} is not mapped"))
    }
}

impl Engine for Nvenc {
    const NAME: &'static str = "NVENC";
    const CLASS: u32 = CLASS;

    fn thi_write(&self, offset: u32, value: u32) -> Option<()> {
        if let Some(failure) = &self.shared.state.lock().unwrap().failure {
            return self.refuse(format!("stopped earlier: {failure}"));
        }
        match offset {
            THI_INCR_SYNCPT => {
                let (index, condition) = ((value & 0xFF) as usize, value >> 8);
                if condition >= INCREMENT_CONDITIONS {
                    return self.refuse(format!("syncpoint increment {value:#x}"));
                }
                if self.shared.syncpoints.get(index).is_none() {
                    return self.refuse(format!("increment of syncpoint {index}"));
                }
                if condition == 0 {
                    self.shared.syncpoints.add(index, 1);
                } else {
                    self.submit(Job::Increment(index));
                }
                Some(())
            }
            THI_CONT_SYNCPT_L1 if value == 0 => Some(()),
            THI_METHOD_OFFSET => {
                self.shared.state.lock().unwrap().offset = value;
                Some(())
            }
            THI_METHOD_DATA => self.method(value),
            _ => self.refuse(format!("THI word {offset:#x} = {value:#x}")),
        }
    }

    fn method_offset(&self) -> u32 {
        self.shared.state.lock().unwrap().offset
    }

    fn start(&self, imem: &[u8], dmem: &[u8]) -> Result<(), String> {
        let digest = |bytes: &[u8]| {
            Sha256::digest(bytes)
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        };
        let (code, data) = (digest(imem), digest(dmem));
        if code != IMEM_SHA256 || data != DMEM_SHA256 {
            return Err(format!(
                "unknown firmware (IMEM SHA-256 {code}, DMEM SHA-256 {data}); only \
                 nvservices 23.5.0's nvhost_nvenc050.fw is modeled"
            ));
        }
        self.shared.state.lock().unwrap().started = true;
        Ok(())
    }
}

/// Saves the THI method offset, the methods set, whether the firmware
/// runs and why the engine stopped, once every queued picture is encoded
/// and every queued increment counted. The host encoder's reference
/// picture is not saved: after a restore its first picture is a key
/// picture, written as an I picture where the driver asked for a P one
/// (`h264::rewrite`), which the driver's decoder takes as it is. Saved by
/// its host, the [`HostFalcon`](super::falcon_host::HostFalcon).
impl Snapshot for Nvenc {
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        self.drain();
        encode(&*self.shared.state.lock().unwrap())
    }

    fn restore(&self, state: &[u8]) -> Result<(), SnapshotError> {
        self.drain();
        *self.shared.state.lock().unwrap() = decode("NVENC", state)?;
        self.submit(Job::Reset);
        self.drain();
        Ok(())
    }
}

/// The non-zero methods, for a refusal note.
fn dump(methods: &[u32; method::COUNT]) -> String {
    let set: Vec<String> = (methods.iter().enumerate())
        .filter(|(_, value)| **value != 0)
        .map(|(i, value)| format!("{:03x}={value:x}", i * 4))
        .collect();
    format!("methods {}", set.join(" "))
}

#[cfg(test)]
mod tests;
