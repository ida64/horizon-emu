//! Video decoder (NVDEC 2.0, host1x class 0xF0): its THI registers, its
//! methods, the firmware nvservices boots it with, and H.264 decoding by
//! the host's decoder.
//!
//! # Firmware
//!
//! nvservices 23.5.0 boots the engine when a client first opens it
//! (`0x835e0`, the same host falcon interface as the VIC's,
//! [`HostFalcon`](super::falcon_host::HostFalcon)): it waits for DMACTL
//! bits 1-2 (memory scrubbing) to clear, points DMATRFBASE at its pinned
//! copy of the embedded `nvhost_nvdec020_ns.fw` (looked up at `0x83820`;
//! table entry `0x199180`: 0x20000 bytes at `0x142000`), transfers the
//! image's data segment (0x400 bytes) to DMEM 0 and the first 256 bytes of
//! its code segment to IMEM 0, unmasks interrupts, starts CPUCTL and waits
//! for IDLESTATE to read 0. The image has the falcon "bin header" Linux's
//! `drivers/gpu/host1x`-era `falcon.c` parses (magic 0x10DE, version 1,
//! OS header at 0x100, OS data at 0x200: code 0..0x300, data 0x1F600..
//! 0x1FA00). Only that firmware is known: the start is refused unless IMEM
//! and DMEM hold exactly what it puts there ([`IMEM_SHA256`],
//! [`DMEM_SHA256`]). No Falcon instruction is executed; the decoder it
//! would run is the model below.
//!
//! # Methods
//!
//! Method offsets and the codec numbers are NVIDIA's public
//! `open-gpu-doc` `classes/video/clc5b0.h` (`NVC5B0_VIDEO_DECODER`, a
//! later NVDEC whose method layout is taken to be NVDEC 2.0's
//! \[assumption\]). They arrive as on the VIC: THI word 0x10 (method offset
//! in words) and 0x11 (data), and THI word 0 increments a syncpoint (Linux
//! `hw_host1x04_uclass.h`). Buffer addresses are IOVA >> 8 in the NVDEC's
//! SMMU address space.
//!
//! The browser (openWeb, playing the parental controls video) sends per
//! picture: SET_APPLICATION_ID 3 (H.264), SET_CONTROL_PARAMS 0x53 (H.264,
//! GPTIMER_ON, ERR_CONCEAL_ON), the picture setup (0x404), bitstream
//! (0x408), slice offsets, co-located data, history, histogram and status
//! buffers, the MB history buffer (0x500), SET_PICTURE_INDEX counting
//! pictures, SET_DISPLAY_BUF_* naming the previous picture, and the 17
//! picture surfaces (0x430 luma, 0x474 chroma), then EXECUTE 0x100.
//!
//! # Decoding
//!
//! EXECUTE decodes one H.264 picture (module `h264` has the picture setup
//! and how the stream is rebuilt from it) with VideoToolbox (`videotoolbox`)
//! and writes it where NVDEC writes it: NV12 at the luma and chroma
//! surfaces of `CurrPicIdx` (plus the frame offsets of the picture setup),
//! block linear with blocks `2 << gob_height` GOBs high and rows of the
//! setup's pitches — what the VIC is then told it reads (the browser's
//! slot: format 68 `T_Y8___V8U8_N420`, block kind 1, block height 2 GOBs,
//! for `gob_height` 0). Rows a block row pads are left as they were.
//!
//! The host decoder keeps its own reference pictures: the surfaces of the
//! DPB entries are not read, the pictures are fed to it in decoding order
//! from the IDR picture on, and the stream since that picture is kept so
//! the decoder can be brought back to the same point (after a snapshot is
//! restored, or when the rebuilt sequence parameter set changes).
//! `max_num_ref_frames` is not in the picture setup: it is the largest
//! number of reference frames a picture's DPB has held since the IDR
//! picture. The decoder drops a reference only when that number is
//! reached, so until a DPB shows one more than assumed it holds the same
//! pictures as the stream's own decoder; when one does, the sequence is
//! rebuilt with the new number and the stream decoded again up to the
//! current picture, before which the two decoders' DPBs could not have
//! differed in a way a decoded picture depends on.
//!
//! The status record (SET_NVDEC_STATUS_OFFSET, `nvdec_status_s` of
//! `nvdec_drv.h`; the browser's TVMR reads words 0-3 as decoded MBs,
//! concealed MBs, cycles and error flags, openWeb `0x53BB30`) gets every
//! macroblock decoded, none concealed, no error and no slice header error.
//! The cycle count (word 2) is left as it is: no NVDEC cycles are spent,
//! and the time the host takes is no measure of them. The browser scales
//! its NVDEC clock request by it (`VideoDFS_PID_NVDEC`).
//!
//! Anything else refuses: other codecs, control bits, a picture setup
//! `h264::PictureInfo::check` refuses, NAL units other than slices, end
//! of stream and filler data, slices naming different picture parameter
//! sets, a non-IDR picture first, sequence or picture parameter sets
//! changing without an IDR picture, a picture the decoder fails on.
//!
//! VP9 in particular: NVDEC hands the driver the frame's symbol counts
//! (`VP9_SET_CTX_COUNTER_BUF_OFFSET`, `nvdec_vp9EntropyCounts_t` in
//! `nvdec_drv.h`), from which the driver adapts the probabilities of the
//! next frame \[inference from those names\]; a decoder that only returns
//! pictures cannot give them.

mod h264;
pub(crate) mod videotoolbox;

use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{Notes, falcon_host::Engine, host1x::Syncpoints, mc::DeviceMemory};
use crate::{
    maxwell::texture::layout::{BlockLinear, BlockLinearSurface},
    snapshot::{Snapshot, SnapshotError, decode, encode},
};
use h264::PictureInfo;

/// Host1x class of the NVDEC (Linux `include/linux/host1x.h`
/// `HOST1X_CLASS_NVDEC`).
pub const CLASS: u32 = 0xF0;

/// SHA-256 of the 64 KiB IMEM once nvservices 23.5.0 has loaded
/// `nvhost_nvdec020_ns.fw` (its first 256 code bytes at 0, zero after).
pub const IMEM_SHA256: &str = "7684223bc3a0a63c02c9e98c87ee1ad0f782a503e71095660576404c8b56aef9";
/// SHA-256 of the 64 KiB DMEM then (the 0x400-byte data segment at 0,
/// zero after).
pub const DMEM_SHA256: &str = "de2f256064a0af797747c2b97505dc0b9f3df0de4f489eac731c23ae9ca9cc31";

/// THI registers, word offsets (as the VIC's, libdrm `vic.h`).
const THI_INCR_SYNCPT: u32 = 0x00;
const THI_METHOD_OFFSET: u32 = 0x10;
const THI_METHOD_DATA: u32 = 0x11;

/// Conditions of a THI syncpoint increment (Linux `hw_host1x04_uclass.h`):
/// immediate, operation done, read done, register write safe. EXECUTE
/// finishes within the method that starts it, so each holds at once.
const INCREMENT_CONDITIONS: u32 = 4;

/// Method word indices (byte offset / 4), `clc5b0.h` names.
mod method {
    pub const SET_APPLICATION_ID: usize = 0x200 / 4;
    pub const SET_WATCHDOG_TIMER: usize = 0x204 / 4;
    pub const EXECUTE: usize = 0x300 / 4;
    /// `SET_CONTROL_PARAMS` 0x400 to `SET_EXTERNAL_MVBUFFER_OFFSET` 0x4BC:
    /// the codec, the picture setup, bitstream and output buffers.
    pub const SETUP: std::ops::RangeInclusive<usize> = 0x400 / 4..=0x4BC / 4;
    pub const SET_CONTROL_PARAMS: usize = 0x400 / 4;
    pub const SET_DRV_PIC_SETUP_OFFSET: usize = 0x404 / 4;
    pub const SET_IN_BUF_BASE_OFFSET: usize = 0x408 / 4;
    pub const SET_NVDEC_STATUS_OFFSET: usize = 0x424 / 4;
    /// `SET_PICTURE_LUMA_OFFSET0..16`.
    pub const SET_PICTURE_LUMA_OFFSET: usize = 0x430 / 4;
    /// `SET_PICTURE_CHROMA_OFFSET0..16`.
    pub const SET_PICTURE_CHROMA_OFFSET: usize = 0x474 / 4;
    pub const PICTURES: u32 = 17;
    pub const H264_SET_MBHIST_BUF_OFFSET: usize = 0x500 / 4;
    /// `VP9_SET_PROB_TAB_BUF_OFFSET` 0x5C0 to
    /// `VP9_SET_FILTER_BUFFER_OFFSET` 0x5DC.
    pub const VP9: std::ops::RangeInclusive<usize> = 0x5C0 / 4..=0x5DC / 4;
    pub const COUNT: usize = 0x600 / 4;
}

/// `SET_APPLICATION_ID` and `SET_CONTROL_PARAMS` `CODEC_TYPE` of H.264.
const CODEC_H264: u32 = 3;
/// `SET_CONTROL_PARAMS` bits besides the codec type: GPTIMER_ON (4),
/// ERR_CONCEAL_ON (6) and EC_INTRA_FRAME_USING_PSLC (14, set by the
/// browser on some pictures), which change nothing for a picture that
/// decodes without error.
const CONTROL_KNOWN: u32 = 0xF | 1 << 4 | 1 << 6 | 1 << 14;

/// Words of `nvdec_status_s` written: `mbs_correctly_decoded`,
/// `mbs_in_error`, `error_status` and `slice_header_error_code` (after the
/// nine words of the HEVC/VP9 union).
const STATUS_DECODED: u64 = 0;
const STATUS_IN_ERROR: u64 = 4;
const STATUS_ERROR: u64 = 0xC;
const STATUS_SLICE_HEADER_ERROR: u64 = 0x34;

/// The NVDEC engine behind the THI.
pub struct Nvdec {
    memory: DeviceMemory,
    syncpoints: Arc<Syncpoints>,
    notes: Arc<Notes>,
    state: Mutex<State>,
    /// The host decoder, at the end of `State::stream`'s history; none
    /// until a picture needs it. Not saved: it is rebuilt from the
    /// history.
    decoder: Mutex<Option<Decoder>>,
}

#[derive(Serialize, Deserialize)]
struct State {
    offset: u32,
    #[serde(with = "crate::snapshot::array")]
    methods: [u32; method::COUNT],
    /// Whether the firmware was started.
    started: bool,
    stream: Stream,
}

/// The H.264 stream since its last IDR picture.
#[derive(Default, Serialize, Deserialize)]
struct Stream {
    /// `max_num_ref_frames` of the rebuilt sequence parameter set.
    references: u32,
    sps: Vec<u8>,
    /// Picture parameter sets by ID.
    pps: BTreeMap<u32, Vec<u8>>,
    /// The pictures decoded (slice NAL units with 4-byte lengths), the IDR
    /// one first.
    history: Vec<Vec<u8>>,
}

/// A decompression session and the parameter sets it was made for.
struct Decoder {
    sps: Vec<u8>,
    pps: BTreeMap<u32, Vec<u8>>,
    session: videotoolbox::Session,
}

impl Nvdec {
    /// `memory` is memory as the NVDEC's SMMU client reaches it.
    pub fn new(memory: DeviceMemory, syncpoints: Arc<Syncpoints>, notes: Arc<Notes>) -> Self {
        Self {
            memory,
            syncpoints,
            notes,
            state: Mutex::new(State {
                offset: 0,
                methods: [0; method::COUNT],
                started: false,
                stream: Stream::default(),
            }),
            decoder: Mutex::new(None),
        }
    }

    fn method(&self, value: u32) -> Option<()> {
        let mut state = self.state.lock().unwrap();
        if !state.started {
            return self.refuse(format!(
                "method {:#x} = {value:#x} before the firmware was started",
                state.offset * 4
            ));
        }
        let index = state.offset as usize;
        if index == method::EXECUTE {
            let methods = state.methods;
            return match self.execute(&mut state.stream, &methods) {
                Ok(()) => Some(()),
                Err(reason) => {
                    drop(state);
                    self.refuse(format!("EXECUTE {value:#x}: {reason}; {}", dump(&methods)))
                }
            };
        }
        let accepted = matches!(
            index,
            method::SET_APPLICATION_ID | method::SET_WATCHDOG_TIMER
        ) || method::SETUP.contains(&index)
            || index == method::H264_SET_MBHIST_BUF_OFFSET
            || method::VP9.contains(&index);
        if !accepted {
            return self.refuse(format!("method {:#x} = {value:#x}", index * 4));
        }
        state.methods[index] = value;
        Some(())
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

    /// Decodes the picture the methods describe.
    fn execute(&self, stream: &mut Stream, methods: &[u32; method::COUNT]) -> Result<(), String> {
        let application = methods[method::SET_APPLICATION_ID];
        let control = methods[method::SET_CONTROL_PARAMS];
        if application != CODEC_H264 || control & 0xF != CODEC_H264 {
            return Err(format!(
                "codec {application} (control {control:#x}): only H.264 is modeled"
            ));
        }
        if control & !CONTROL_KNOWN != 0 {
            return Err(format!("control parameters {control:#x}"));
        }
        let address = |index: usize| u64::from(methods[index]) << 8;
        let mut setup = [0; h264::PICTURE_INFO_SIZE];
        self.read(
            address(method::SET_DRV_PIC_SETUP_OFFSET),
            &mut setup,
            "picture setup",
        )?;
        let info = PictureInfo::parse(&setup);
        info.check()?;
        let mut bitstream = vec![0; info.stream_len as usize];
        self.read(
            address(method::SET_IN_BUF_BASE_OFFSET),
            &mut bitstream,
            "bitstream",
        )?;
        let sample = stream.add(&info, &bitstream)?;
        let frame = self.decode(stream, &sample)?;
        stream.history.push(sample);
        self.store(methods, &info, &frame)?;
        let status = address(method::SET_NVDEC_STATUS_OFFSET);
        if status != 0 {
            let macroblocks = info.width_mbs * info.height_mbs;
            for (offset, value) in [
                (STATUS_DECODED, macroblocks),
                (STATUS_IN_ERROR, 0),
                (STATUS_ERROR, 0),
                (STATUS_SLICE_HEADER_ERROR, 0),
            ] {
                self.write(status + offset, &value.to_le_bytes(), "status record")?;
            }
        }
        Ok(())
    }

    /// Decodes `sample`, the stream's next picture, first bringing the
    /// decoder to the end of the stream's history if it is not there.
    fn decode(&self, stream: &Stream, sample: &[u8]) -> Result<videotoolbox::Frame, String> {
        let mut decoder = self.decoder.lock().unwrap();
        let current = decoder
            .as_ref()
            .is_some_and(|d| d.sps == stream.sps && d.pps == stream.pps);
        if !current || stream.history.is_empty() {
            let pps: Vec<&[u8]> = stream.pps.values().map(Vec::as_slice).collect();
            let mut session = videotoolbox::Session::new(&stream.sps, &pps)?;
            for earlier in &stream.history {
                session.decode(earlier)?;
            }
            *decoder = Some(Decoder {
                sps: stream.sps.clone(),
                pps: stream.pps.clone(),
                session,
            });
        }
        decoder.as_mut().unwrap().session.decode(sample)
    }

    /// Writes `frame` to the surfaces of the picture `info` describes.
    fn store(
        &self,
        methods: &[u32; method::COUNT],
        info: &PictureInfo,
        frame: &videotoolbox::Frame,
    ) -> Result<(), String> {
        let (width, height) = (16 * info.width_mbs, 16 * info.height_mbs);
        if (frame.width, frame.height) != (width, height) {
            return Err(format!(
                "decoded {}x{} for a {width}x{height} picture",
                frame.width, frame.height
            ));
        }
        let index = info.curr_pic_idx;
        if index >= method::PICTURES {
            return Err(format!("picture index {index}"));
        }
        let block = BlockLinear {
            block_height_log2: info.gob_height + 1,
            block_depth_log2: 0,
        };
        let plane = |surface: usize, offset: u32| {
            (u64::from(methods[surface + index as usize]) + u64::from(offset)) << 8
        };
        let luma = plane(method::SET_PICTURE_LUMA_OFFSET, info.luma_frame_offset);
        let chroma = plane(method::SET_PICTURE_CHROMA_OFFSET, info.chroma_frame_offset);
        for (address, pitch, rows, bytes, what) in [
            (luma, info.pitch_luma, height, &frame.luma, "luma surface"),
            (
                chroma,
                info.pitch_chroma,
                height / 2,
                &frame.chroma,
                "chroma surface",
            ),
        ] {
            if pitch < width || pitch % 64 != 0 {
                return Err(format!("{what} pitch {pitch} for {width} bytes a row"));
            }
            let surface = BlockLinearSurface::new(pitch, rows, block);
            let mut image = vec![0; surface.size(1) as usize];
            self.read(address, &mut image, what)?;
            swizzle(&surface, bytes, width, rows, &mut image);
            self.write(address, &image, what)?;
        }
        Ok(())
    }

    fn refuse(&self, reason: String) -> Option<()> {
        self.notes.add(format!("NVDEC: {reason}"));
        None
    }
}

impl Stream {
    /// Takes in the picture of `info` and `bitstream`: the parameter sets
    /// it needs and its sample for the decoder.
    fn add(&mut self, info: &PictureInfo, bitstream: &[u8]) -> Result<Vec<u8>, String> {
        let slices = h264::slices(bitstream)?;
        let ids: Vec<Option<u32>> = slices.iter().map(|s| h264::slice_pps_id(s)).collect();
        let Some(id) = ids[0].filter(|_| ids.iter().all(|other| *other == ids[0])) else {
            return Err(format!("slices name picture parameter sets {ids:?}"));
        };
        let idr = slices[0][0] & 0x1F == h264::IDR_SLICE;
        if idr {
            *self = Stream::default();
        } else if self.history.is_empty() {
            return Err("a picture other than IDR starts the stream".into());
        }
        let references = self.references.max(info.reference_frames()).max(1);
        let sps = h264::sequence_parameter_set(info, references);
        if !self.sps.is_empty()
            && sps != self.sps
            && h264::sequence_parameter_set(info, self.references) != self.sps
        {
            return Err("the sequence changed without an IDR picture".into());
        }
        let pps = h264::picture_parameter_set(info, id);
        if self.pps.get(&id).is_some_and(|known| *known != pps) {
            return Err(format!(
                "picture parameter set {id} changed without an IDR picture"
            ));
        }
        self.references = references;
        self.sps = sps;
        self.pps.insert(id, pps);
        let mut sample = Vec::with_capacity(bitstream.len() + 4 * slices.len());
        for slice in slices {
            sample.extend_from_slice(&(slice.len() as u32).to_be_bytes());
            sample.extend_from_slice(slice);
        }
        Ok(sample)
    }
}

/// Writes the `rows` rows of `width` bytes of the pitch-linear `source`
/// into `image`, the bytes of block-linear `surface`. A GOB keeps 16-byte
/// runs of a row together (`gob_offset`), so the copy is by such runs.
fn swizzle(surface: &BlockLinearSurface, source: &[u8], width: u32, rows: u32, image: &mut [u8]) {
    for y in 0..rows {
        let row = &source[(y * width) as usize..((y + 1) * width) as usize];
        for (x, run) in (0..width).step_by(16).zip(row.as_chunks::<16>().0) {
            let at = surface.offset(x, y, 0) as usize;
            image[at..at + 16].copy_from_slice(run);
        }
    }
}

impl Engine for Nvdec {
    const NAME: &'static str = "NVDEC";
    const CLASS: u32 = CLASS;

    fn thi_write(&self, offset: u32, value: u32) -> Option<()> {
        match offset {
            THI_INCR_SYNCPT => {
                let (index, condition) = ((value & 0xFF) as usize, value >> 8);
                if condition >= INCREMENT_CONDITIONS {
                    return self.refuse(format!("syncpoint increment {value:#x}"));
                }
                if self.syncpoints.get(index).is_none() {
                    return self.refuse(format!("increment of syncpoint {index}"));
                }
                self.syncpoints.add(index, 1);
                Some(())
            }
            THI_METHOD_OFFSET => {
                self.state.lock().unwrap().offset = value;
                Some(())
            }
            THI_METHOD_DATA => self.method(value),
            _ => self.refuse(format!("THI word {offset:#x} = {value:#x}")),
        }
    }

    fn method_offset(&self) -> u32 {
        self.state.lock().unwrap().offset
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
                 nvservices 23.5.0's nvhost_nvdec020_ns.fw is modeled"
            ));
        }
        self.state.lock().unwrap().started = true;
        Ok(())
    }
}

/// Saves the THI method offset, the methods set, whether the firmware
/// runs, and the H.264 stream since its IDR picture, from which the host
/// decoder is rebuilt when the next picture comes. Saved by its host, the
/// [`HostFalcon`](super::falcon_host::HostFalcon).
impl Snapshot for Nvdec {
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        encode(&*self.state.lock().unwrap())
    }

    fn restore(&self, state: &[u8]) -> Result<(), SnapshotError> {
        *self.state.lock().unwrap() = decode("NVDEC", state)?;
        *self.decoder.lock().unwrap() = None;
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
mod tests {
    use super::*;
    use crate::{boot::DramSize, memory::GuestRam, soc::mc::Mc};

    fn nvdec() -> (Nvdec, Arc<Notes>) {
        let ram = Arc::new(GuestRam::new(0x8000_0000, 0x1000).unwrap());
        let mc = Arc::new(Mc::new(DramSize::Gb4));
        let notes: Arc<Notes> = Arc::default();
        let memory = DeviceMemory::new(&mc, &ram, crate::soc::mc::SMMU_NVDEC_ASID);
        let syncpoints = Arc::new(Syncpoints::default());
        (Nvdec::new(memory, syncpoints, Arc::clone(&notes)), notes)
    }

    #[test]
    fn only_the_known_firmware_starts_it_and_methods_wait_for_it() {
        let (nvdec, _) = nvdec();
        nvdec.thi_write(THI_METHOD_OFFSET, 0x200 / 4).unwrap();
        assert_eq!(nvdec.thi_write(THI_METHOD_DATA, 3), None);
        let zero = [0u8; 0x10000];
        assert!(
            nvdec
                .start(&zero, &zero)
                .unwrap_err()
                .contains("unknown firmware")
        );
        nvdec.state.lock().unwrap().started = true;
        nvdec.thi_write(THI_METHOD_DATA, 3).unwrap();
        assert_eq!(nvdec.method_offset(), 0x200 / 4);
        nvdec.thi_write(THI_METHOD_OFFSET, 0x240 / 4).unwrap();
        assert_eq!(nvdec.thi_write(THI_METHOD_DATA, 1), None);
    }

    #[test]
    fn swizzled_runs_land_in_gob_order() {
        // 128 bytes wide (two GOBs), 16 rows, blocks 2 GOBs high: GOB 0
        // holds x 0..64 of rows 0..8, GOB 1 rows 8..16, GOB 2 x 64..128.
        let block = BlockLinear {
            block_height_log2: 1,
            block_depth_log2: 0,
        };
        let surface = BlockLinearSurface::new(128, 16, block);
        let source: Vec<u8> = (0..128 * 16).map(|i| (i / 16) as u8).collect();
        let mut image = vec![0xEE; surface.size(1) as usize];
        swizzle(&surface, &source, 128, 16, &mut image);
        // Row 1, x 0..16 follows row 0's x 0..16 in the GOB, and row 0's
        // x 16..32 comes after both (`gob_offset`).
        assert_eq!(image[0], 0);
        assert_eq!(image[16], 8);
        assert_eq!(image[32], 1);
        // Row 8 starts GOB 1 (512 bytes in), x 64 starts GOB 2.
        assert_eq!(image[512], 64);
        assert_eq!(image[1024], 4);
        assert!(!image.contains(&0xEE));
    }

    #[test]
    fn the_stream_starts_at_an_idr_picture_and_grows_its_references() {
        let mut setup = [0; h264::PICTURE_INFO_SIZE];
        for (at, value) in [
            (0x60, 1u32),
            (0x64, 0x50),
            (0x68, 0x2d),
            (0x8c, 0x500),
            (0x90, 0x500),
            (0xb0, 0x1216),
        ] {
            setup[at..at + 4].copy_from_slice(&value.to_le_bytes());
        }
        setup[0x1C0..0x2A0].fill(16);
        let info = PictureInfo::parse(&setup);
        let p_slice = [0, 0, 1, 0x21, 0x9a, 0x08];
        let mut stream = Stream::default();
        assert!(stream.add(&info, &p_slice).is_err());
        let idr = stream.add(&info, &[0, 0, 1, 0x25, 0x88, 0x80]).unwrap();
        assert_eq!(idr, [0, 0, 0, 3, 0x25, 0x88, 0x80]);
        assert_eq!(stream.references, 1);
        stream.history.push(idr);
        // A DPB of two reference frames raises max_num_ref_frames.
        for entry in [0xC0, 0xD0] {
            setup[entry..entry + 4].copy_from_slice(&0x3000u32.to_le_bytes());
        }
        stream.add(&PictureInfo::parse(&setup), &p_slice).unwrap();
        assert_eq!(stream.references, 2);
        // A different sequence without an IDR picture is refused.
        setup[0x64] = 0x28;
        assert!(stream.add(&PictureInfo::parse(&setup), &p_slice).is_err());
    }
}
