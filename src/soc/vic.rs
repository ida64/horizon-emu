//! Video Image Compositor (VIC 4.0, host1x class 0x5D): its THI registers,
//! its methods and the composition an EXECUTE performs, done in software
//! through the VIC's SMMU address space.
//!
//! # Method stream
//!
//! nvnflinger 23.5.0 (`DdkVicBlit` `0xAE1A0`, VIC 4.0 function table
//! `0xFB8E0`) writes every VIC method as host1x `INCR` of THI word 0x10
//! (method offset in words) and 0x11 (data) (`0xAE040`):
//! SET_APPLICATION_ID 0x200 = 1 (`0xAFBB0`), SET_CONTROL_PARAMS 0x704 =
//! 0x610110 (`0xAFBD0`: `MOV W2, #0x610110`), SET_CONFIG_STRUCT_OFFSET 0x708 (`0xAFBF0`), method
//! 0x730 = 0 (`0xAFC20`, ordinary path), SET_FILTER_STRUCT_OFFSET 0x70C
//! (`0xAFC60`), the output planes 0x720 + 4·plane (`0xAFD40`), the input
//! planes 0x400 + 0x60·slot + 4·(3·surface + plane) (`0xAFD60`), EXECUTE
//! 0x300 = 0x100 (`0xAFDC0`), then a THI syncpoint increment with condition
//! 1 (`0xAFE90` → `0x85010`: `NONINCR` word 0, `id | 1 << 8`). Addresses
//! are IOVA >> 8 (`0x85250`, shift 8). nvservices writes 0x200 = 1, 0x71C
//! and 0x72C through the same THI pair in MMIO at boot (spec §5.2 step 8).
//! The method names and the THI word offsets are public: libdrm
//! `tests/tegra/vic.h`, `vic40.h`, `vic40.c` (MIT, © NVIDIA).
//!
//! # Configuration structure
//!
//! 0x610 bytes, layout from libdrm `vic40.h` (public) and checked against
//! nvnflinger's serializer `0xAEA60`, which fills it field by field: slot
//! source rectangles in 16.16 fixed point with inclusive right/bottom
//! (`vcvts_n_u32_f32(…, 16)` of `right − 1.0`, low 12 bits cleared), every
//! other rectangle and size inclusive (`− 1`), 10-bit colors and constants
//! (`× 1023`), planar alpha 0x3FF with or without CONSTANT_ALPHA, soft clamp
//! 0..0x3FF, SLOT w0 bit 8 (current field) always set.
//!
//! # What is composed
//!
//! Inside the output target rectangle the picture starts as the background
//! colour (what libdrm's `vic40_fill` relies on: a configuration with no
//! slot fills the target rectangle); each enabled slot is then blended over
//! it in slot order at its destination rectangle; outside the target
//! rectangle the output surface is left as it was. Implemented: the 32-bit
//! RGB formats 31-38, pitch-linear and block-linear (16Bx2 GOB) surfaces,
//! the 8-bit YUV input formats 67-72 (semi-planar 4:2:0, NVDEC's pictures,
//! and fully planar; see `Yuv`), scaling (see `taps`), the slot and output
//! colour matrices (Ryujinx `Blender.cs` formula, MIT), soft clamp, sRGB
//! degamma/regamma (mode 1), planar/constant alpha, the blending factor
//! selections (libdrm `vic.h` enum names), override and mask, alpha fill
//! modes opaque, background and composited (DXVA-HD names in libdrm
//! `vic.h`), the output flips, transposition of a pitch-linear RGB output
//! (`Surface::transposed`), and as output the 8-bit YUV formats 67-72
//! (semi-planar 4:2:0, grc's NV12 encoder input, and fully planar;
//! pitch-linear or block-linear; see `Planar`). Everything else
//! (fractional source rectangles, noise and detail filters, other YUV
//! input, 10-bit YUV, deinterlacing and denoise, luma key, gamut
//! matrix, gamma modes 2-3, clear rectangles, decompression,
//! downsampling) refuses with the field named.
//!
//! # The submission seen
//!
//! When starter launches the netConnect library applet, nvnflinger sends one
//! EXECUTE (nvhost-vic SUBMIT `0xC09C0024`): slot 0 copies a 1280×720
//! A8B8G8R8 block-linear surface (16-GOB blocks) 1:1 into another of the same
//! kind, degamma and regamma 1, identity output matrix, blending "source ×
//! K1, destination × 0", alpha overridden to 1023, alpha fill composited —
//! an opaque copy of the caller's last frame.
//!
//! Opening News from HOME sends an EXECUTE with the same slot (a 1280×720
//! A8B8G8R8 block-linear frame, copied 1:1 through sRGB degamma/regamma)
//! into a pitch-linear 1280×720 `T_Y8___U8___V8_N422R` output (format 71,
//! chroma planes 1280×360, vertical siting 1) through the BT.601 full-range
//! RGB-to-YCbCr matrix (Y = 0.299 R + 0.587 G + 0.114 B, chroma offsets
//! 128/255): a JPEG-style YCbCr picture of the frame.
//!
//! The capture button on HOME sends an EXECUTE that reads a 1280×720
//! picture of that planar kind (slot format 71, through the inverse BT.601
//! full-range matrix) and reduces it with ten-tap filters to a 96×54
//! pitch-linear A8B8G8R8 thumbnail, the output's chroma addresses still
//! those of the planar picture (see `Surface`).
//!
//! # Time
//!
//! An EXECUTE composes on the thread that runs the host1x channel's
//! stream when it reaches the EXECUTE: the DMAPUT write that reaches it, or
//! the channel's wait thread (see `host1x_channel`), so the host time it
//! takes passes in guest time too, and the THI syncpoint increment that
//! follows comes that much later. The channel's registers stay free
//! meanwhile, and other cores run on. Stopping guest time for the duration
//! would freeze the clock those cores read while they go on running, which
//! no console does, so the model keeps guest time running and makes the
//! host work short instead.
//!
//! That matters to the capture button. vi (`010000000000002d`, 23.5.0)
//! asks nvnflinger for the next composed frame and waits for it with a
//! timeout of 100 ms (`0x123C50` passes 0x5F5E100 ns to `0x1319C8`, which
//! waits on the frame event at `0x131AEC`); without it vi fails the
//! capture with 0xDCCE (`0x123E54`) and HOME says "Unable to take a
//! screenshot". nvnflinger's frame is the planar YCbCr EXECUTE above, so
//! that EXECUTE, the wait for the next composition and the GPU's part all
//! fall inside those 100 ms. Composing one pixel at a time on one host
//! thread took 53 ms for that EXECUTE (54 ms for the netConnect copy, 29
//! ms for the 96×54 thumbnail; the replays in `tests.rs`, release build,
//! Apple M-series host), and with the emulator's other threads busy the
//! capture missed the 100 ms often. Now each EXECUTE splits its rows among
//! up to `BAND_THREADS` host threads (`in_bands`) and the sRGB curves are
//! memoized exactly (`Memo`): 8, 10 and 7 ms for the same replays, and the
//! picture written is the same bit for bit. A lone slot copying 8-bit RGB
//! 1:1 then went through tables of its composed values (`Fused`) and the
//! outputs were written a row at a time: grc's 30-a-second NV12 capture of
//! Minecraft's frames went from about 10 ms an EXECUTE to about 5 ms on a
//! busy host (median, title menu), the same bit for bit (the tests compare
//! both paths). The VIC's own time for such a picture is not public for
//! VIC 4.0; at one pixel a clock and several hundred MHz it would be a few
//! milliseconds \[assumption\], the order of what the model now takes.
//!
//! # Without the firmware
//!
//! hekate's Nyx turns its picture for the panel with the VIC and loads no
//! Falcon firmware for it: it writes the VIC's own registers through the
//! Falcon's private register window (`vic.c`, reference for facts; the
//! window is the host interface's, [`super::falcon_host`]). In its order:
//! the fetch control engine's microcode, word by word (`FCE_UCODE_ADDR`,
//! `FCE_UCODE_INST`), and its start (`FCE_CTRL` 1); the configuration
//! structure's address and size (`SC_PRAMBASE`, `SC_PRAMSIZE`: the same
//! structure the firmware takes, 0x610 bytes); the slot map; the first
//! slot's surface and the output (`SC_SFC0_BASE_LUMA`, `BL_TARGET_BASADR`,
//! addresses >> 8 as in the methods); `BL_CONFIG`, which has the blending
//! output take the structure; then `FC_COMPOSE` 1 for every frame.
//!
//! Here `FC_COMPOSE` runs the same composition an EXECUTE does, on that
//! structure and those two surfaces, at once (the Falcon's idle state
//! reads idle). The microcode is not run: only hekate's is accepted, by
//! its digest, as the one known to make the engine fetch what the
//! structure says. \[assumption\] The structure is read when a frame is
//! composed; on the chip the surface cache parses it when its address is
//! written, and hekate never changes it afterwards. Only what hekate
//! writes is taken (one slot, its one `BL_CONFIG`); private registers
//! cannot be read.

use std::sync::{Arc, Mutex};

use sha2::Digest as _;

use serde::{Deserialize, Serialize};

use super::{Notes, host1x::Syncpoints, mc::DeviceMemory};
use crate::{
    maxwell::texture::layout::{BlockLinear, BlockLinearSurface, GOB_HEIGHT},
    snapshot::{Snapshot, SnapshotError, decode, encode},
};

/// Host1x class of the VIC (nvnflinger `0x84F60(…, 106, 93)`).
pub const CLASS: u32 = 0x5D;

/// THI registers, word offsets (libdrm `vic.h` `VIC_UCLASS_*`).
const THI_INCR_SYNCPT: u32 = 0x00;
const THI_METHOD_OFFSET: u32 = 0x10;
const THI_METHOD_DATA: u32 = 0x11;

/// Conditions of a THI syncpoint increment (Linux `hw_host1x04_uclass.h`,
/// public): immediate, operation done, read done, register write safe. The
/// methods before an increment have all completed when it arrives (EXECUTE
/// composes synchronously), so every one of them holds at once.
const INCREMENT_CONDITIONS: u32 = 4;

/// Method word indices (byte offset / 4).
mod method {
    pub const SET_APPLICATION_ID: usize = 0x200 / 4;
    pub const EXECUTE: usize = 0x300 / 4;
    /// 8 slots × 8 surfaces × {luma, chroma U, chroma V}.
    pub const SURFACES: std::ops::Range<usize> = 0x400 / 4..0x700 / 4;
    pub const SET_CONTROL_PARAMS: usize = 0x704 / 4;
    pub const SET_CONFIG_STRUCT_OFFSET: usize = 0x708 / 4;
    pub const SET_FILTER_STRUCT_OFFSET: usize = 0x70C / 4;
    pub const SET_FCE_UCODE_SIZE: usize = 0x71C / 4;
    pub const SET_OUTPUT_SURFACE: std::ops::Range<usize> = 0x720 / 4..0x72C / 4;
    pub const SET_FCE_UCODE_OFFSET: usize = 0x72C / 4;
    /// Written 0 by nvnflinger on its ordinary path (`0xAFC20`); a buffer
    /// address there only on its other path (`0xAFC60`, with 0x734).
    pub const OFFSET_730: usize = 0x730 / 4;
    pub const COUNT: usize = 0x800 / 4;
}

/// EXECUTE value both nvnflinger (`0xAFDC0`) and libdrm (`1 << 8`) write.
const EXECUTE_VALUE: u32 = 0x100;
/// SET_CONTROL_PARAMS: configuration size in 16-byte units in bits 31:16
/// (libdrm `vic40.c`); nvnflinger also sets bits 8 and 4 (`0x610110`, seen
/// on the channel at the netConnect launch), libdrm leaves them clear, and
/// no output depends on them in either.
const CONTROL_SIZE_SHIFT: u32 = 16;
const CONTROL_KNOWN_LOW: [u32; 2] = [0, 0x110];

const CONFIG_SIZE: usize = 0x610;
const SLOT_COUNT: usize = 8;
const SLOT_BASE: usize = 0x90;
const SLOT_SIZE: usize = 0xB0;

/// The VIC engine behind the THI: method state and the compositor.
pub struct Vic {
    memory: DeviceMemory,
    syncpoints: Arc<Syncpoints>,
    notes: Arc<Notes>,
    state: Mutex<State>,
}

#[derive(Serialize, Deserialize)]
struct State {
    offset: u32,
    #[serde(with = "crate::snapshot::array")]
    methods: [u32; method::COUNT],
    direct: Direct,
}

/// What was written through the private registers.
#[derive(Default, Serialize, Deserialize)]
struct Direct {
    ucode_address: u32,
    /// The fetch control engine's microcode, by word.
    ucode: Vec<u32>,
    /// The engine was started with the known microcode.
    started: bool,
    slot_map: u32,
    /// The configuration structure's address and size, the first slot's
    /// first surface and the output, as written (addresses >> 8, the size
    /// >> 6).
    config: u32,
    config_size: u32,
    source: u32,
    target: u32,
    /// The blending output was told to take the configuration structure.
    configured: bool,
}

/// The VIC's own registers hekate writes (`vic.c`; its names without
/// `VIC_FC_`, `VIC_SC_` and `VIC_BL_`).
mod private {
    pub const COMPOSE: u32 = 0x10000;
    pub const SLOT_MAP: u32 = 0x10C00;
    pub const FCE_CTRL: u32 = 0x11000;
    pub const FCE_UCODE_ADDR: u32 = 0x11200;
    pub const FCE_UCODE_INST: u32 = 0x11300;
    pub const PRAMBASE: u32 = 0x14000;
    pub const PRAMSIZE: u32 = 0x14100;
    pub const SFC0_BASE_LUMA: u32 = 0x14300;
    pub const TARGET_BASADR: u32 = 0x22000;
    pub const BL_CONFIG: u32 = 0x22800;
}

const COMPOSE_START: u32 = 1;
const FCE_START: u32 = 1;
/// Slot 0 takes surface 0, the other seven none.
const SLOT_MAP_FIRST_ONLY: u32 = 0xFFFF_FFF0;
/// Slot mask `0x1F`, the configuration structure trigger, subpartition
/// mode: the one value hekate writes.
const BL_CONFIG_ONE_PASS: u32 = 0x1F << 8 | 1 << 2 | 1 << 0;
/// Microcode addresses are bytes below this.
const UCODE_LIMIT: u32 = 0x1000;
/// SHA-256 of the fetch control engine microcode hekate loads ("Dumped
/// from L4T r33", 241 words).
const FCE_UCODE_SHA256: [u8; 32] = [
    0xAC, 0x41, 0xB5, 0xEA, 0x51, 0x29, 0x18, 0xE2, 0xF1, 0xB1, 0xCE, 0x34, 0x78, 0xF9, 0xA3, 0x73,
    0xCB, 0x56, 0x60, 0xFA, 0x1F, 0xF8, 0xE5, 0xA8, 0xFB, 0xA9, 0x6B, 0xB8, 0x54, 0x6B, 0x6F, 0x3C,
];

impl Vic {
    /// `memory` is memory as the VIC's SMMU client reaches it.
    pub fn new(memory: DeviceMemory, syncpoints: Arc<Syncpoints>, notes: Arc<Notes>) -> Self {
        Self {
            memory,
            syncpoints,
            notes,
            state: Mutex::new(State {
                offset: 0,
                methods: [0; method::COUNT],
                direct: Direct::default(),
            }),
        }
    }

    /// A write of THI register `offset` (in words), from a host1x channel
    /// or from the MMIO window (byte offset `4 * offset`).
    pub fn thi_write(&self, offset: u32, value: u32) -> Option<()> {
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

    /// Value of the THI method offset register (MMIO +0x40).
    pub fn method_offset(&self) -> u32 {
        self.state.lock().unwrap().offset
    }

    fn method(&self, value: u32) -> Option<()> {
        let mut state = self.state.lock().unwrap();
        let index = state.offset as usize;
        let accepted = match index {
            method::SET_APPLICATION_ID => value == 1,
            method::SET_CONTROL_PARAMS => {
                value >> CONTROL_SIZE_SHIFT >= (CONFIG_SIZE / 16) as u32
                    && CONTROL_KNOWN_LOW.contains(&(value & 0xFFFF))
            }
            method::SET_CONFIG_STRUCT_OFFSET
            | method::SET_FILTER_STRUCT_OFFSET
            | method::SET_FCE_UCODE_SIZE
            | method::SET_FCE_UCODE_OFFSET => true,
            method::OFFSET_730 => value == 0,
            _ if method::SURFACES.contains(&index)
                || method::SET_OUTPUT_SURFACE.contains(&index) =>
            {
                true
            }
            method::EXECUTE if value == EXECUTE_VALUE => {
                let methods = state.methods;
                drop(state);
                return match compose(&self.memory, &methods) {
                    Ok(()) => Some(()),
                    Err(reason) => self.refuse(format!("EXECUTE: {reason}")),
                };
            }
            _ => false,
        };
        if !accepted {
            return self.refuse(format!("method {:#x} = {value:#x}", index * 4));
        }
        state.methods[index] = value;
        Some(())
    }

    fn refuse(&self, reason: String) -> Option<()> {
        self.notes.add(format!("VIC: {reason}"));
        None
    }

    /// A write of the VIC's own register at `address`, reached through the
    /// Falcon's private register window: see "Without the firmware" in the
    /// module.
    pub fn private_write(&self, address: u32, value: u32) -> Option<()> {
        let mut state = self.state.lock().unwrap();
        let direct = &mut state.direct;
        match address {
            private::FCE_UCODE_ADDR if value.is_multiple_of(4) && value < UCODE_LIMIT => {
                direct.ucode_address = value
            }
            private::FCE_UCODE_INST => {
                let index = direct.ucode_address as usize / 4;
                if direct.ucode.len() <= index {
                    direct.ucode.resize(index + 1, 0);
                }
                direct.ucode[index] = value;
                direct.started = false;
            }
            private::FCE_CTRL if value == FCE_START => {
                let bytes: Vec<u8> = direct
                    .ucode
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect();
                let digest = sha2::Sha256::digest(&bytes);
                if digest[..] != FCE_UCODE_SHA256 {
                    drop(state);
                    return self.refuse(
                        "fetch control engine started with a microcode other than the one \
                         hekate loads: what it would fetch is not known"
                            .into(),
                    );
                }
                direct.started = true;
            }
            private::SLOT_MAP => direct.slot_map = value,
            private::PRAMBASE => direct.config = value,
            private::PRAMSIZE => direct.config_size = value,
            private::SFC0_BASE_LUMA => direct.source = value,
            private::TARGET_BASADR => direct.target = value,
            private::BL_CONFIG if value == BL_CONFIG_ONE_PASS => direct.configured = true,
            private::COMPOSE if value == COMPOSE_START => {
                let ready = direct.started
                    && direct.configured
                    && direct.slot_map == SLOT_MAP_FIRST_ONLY
                    && (direct.config_size as usize) << 6 >= CONFIG_SIZE & !0x3F;
                if !ready {
                    drop(state);
                    return self.refuse(
                        "composition started before the engine was set up the way hekate \
                         sets it up"
                            .into(),
                    );
                }
                let mut methods = [0; method::COUNT];
                methods[method::SET_CONFIG_STRUCT_OFFSET] = direct.config;
                methods[method::SURFACES.start] = direct.source;
                methods[method::SET_OUTPUT_SURFACE.start] = direct.target;
                drop(state);
                return match compose(&self.memory, &methods) {
                    Ok(()) => Some(()),
                    Err(reason) => self.refuse(format!("composition: {reason}")),
                };
            }
            _ => {
                drop(state);
                return self.refuse(format!("private register {address:#x} = {value:#x}"));
            }
        }
        Some(())
    }
}

/// Saves the THI method offset and the method state the next EXECUTE
/// composes with. An EXECUTE completes within the method that starts it,
/// so nothing is ever in flight. Saved by its host, the
/// [`HostFalcon`](super::falcon_host::HostFalcon).
impl Snapshot for Vic {
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        encode(&*self.state.lock().unwrap())
    }

    fn restore(&self, state: &[u8]) -> Result<(), SnapshotError> {
        *self.state.lock().unwrap() = decode("VIC", state)?;
        Ok(())
    }
}

/// `width` bits of `word` from bit `low`.
fn field(word: u64, low: u32, width: u32) -> u32 {
    ((word >> low) & ((1 << width) - 1)) as u32
}

fn flag(word: u64, bit: u32) -> bool {
    word >> bit & 1 != 0
}

/// A rectangle, right and bottom exclusive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Rect {
    left: u32,
    top: u32,
    right: u32,
    bottom: u32,
}

impl Rect {
    /// The 14-bit inclusive layout every non-source rectangle uses.
    fn inclusive(word: u64) -> Rect {
        Rect {
            left: field(word, 0, 14),
            right: field(word, 16, 14) + 1,
            top: field(word, 32, 14),
            bottom: field(word, 48, 14) + 1,
        }
    }

    fn intersect(self, other: Rect) -> Rect {
        let left = self.left.max(other.left);
        let top = self.top.max(other.top);
        Rect {
            left,
            top,
            right: self.right.min(other.right).max(left),
            bottom: self.bottom.min(other.bottom).max(top),
        }
    }

    fn width(self) -> u32 {
        self.right.saturating_sub(self.left)
    }

    fn height(self) -> u32 {
        self.bottom.saturating_sub(self.top)
    }
}

/// Byte positions of red, green, blue and alpha in a 32-bit pixel, and
/// whether alpha is stored (X formats read as opaque). VIC format names list
/// components from the most significant byte of the little-endian word:
/// A8B8G8R8 (31) is stored R, G, B, A and A8R8G8B8 (32) B, G, R, A (Ryujinx
/// `SurfaceWriter.cs`, public); the other six apply the same naming
/// [INFERENCE]. Codes from libdrm `vic.h` / Ryujinx `PixelFormat.cs`.
#[derive(Clone, Copy, Debug)]
struct Format {
    rgb: [usize; 3],
    alpha: usize,
    stored_alpha: bool,
}

fn format(code: u32) -> Result<Format, String> {
    let rgb = match code {
        31 | 35 => [0, 1, 2],
        32 | 36 => [2, 1, 0],
        33 | 37 => [1, 2, 3],
        34 | 38 => [3, 2, 1],
        _ => return Err(format!("pixel format {code}")),
    };
    Ok(Format {
        rgb,
        // The remaining byte of 0..=3.
        alpha: 6 - rgb.iter().sum::<usize>(),
        stored_alpha: code <= 34,
    })
}

const BYTES_PER_PIXEL: u32 = 4;

#[derive(Clone, Copy, Debug)]
enum Layout {
    Pitch(u64),
    /// The surface and the rows of one of its blocks.
    Block(BlockLinearSurface, u32),
}

/// One single-plane surface: the slot surface or the output surface
/// configuration (same field layout) with its plane address.
///
/// A single-plane (pixel-interleaved) format uses only the luma plane
/// address; the chroma U and V addresses beside it are not read. NVIDIA
/// open-gpu-doc `classes/video/vic_ceb6_types.h` says so of the slot
/// surface ("For single plane pixel format(ARGB), only use Luma surface";
/// the chroma padded sizes are "not required for pixel interleaved
/// surfaces such as ARGB"), and libdrm `tests/tegra/vic40.c` programs only
/// `SET_OUTPUT_SURFACE_LUMA_OFFSET` and `SET_SURFACE0_SLOT0_LUMA_OFFSET`
/// for its A8B8G8R8 surfaces. The output surface configuration has the
/// same luma/chroma pairs and the same pixel formats, so the rule is taken
/// to hold for it too [inference]. nvnflinger leaves the chroma addresses
/// of an earlier configuration in place: the capture button's thumbnail
/// EXECUTE writes an A8B8G8R8 output with 0x724/0x728 still naming the
/// chroma planes of the planar YUV picture it reads.
///
/// Device address 0 is a plane address like any other: grc's capture
/// EXECUTEs (Minecraft running, video capture on) read the game's frames,
/// A8R8G8B8 block-linear 1280×720, from VIC addresses 0, 0x3C0000 and
/// 0x780000 in turn — three buffers of 0x3C0000 bytes from the bottom of
/// the VIC's address space \[inference: the game's swapchain, mapped
/// there\]. An unmapped address refuses when the surface is read.
#[derive(Clone, Copy, Debug)]
struct Surface {
    address: u64,
    format: Format,
    layout: Layout,
    width: u32,
    height: u32,
}

impl Surface {
    /// `words` are the two 64-bit words of a surface configuration, `name`
    /// says which in diagnostics.
    fn new(words: [u64; 2], address: u32, name: &str) -> Result<Surface, String> {
        let [w0, w1] = words;
        let format = format(field(w0, 0, 7)).map_err(|e| format!("{name}: {e}"))?;
        // Chroma siting and the slot cache width (w0 bits 7-10, 19-21) only
        // concern subsampled formats and fetch efficiency.
        let kind = field(w0, 11, 4);
        let block_height_log2 = field(w0, 15, 4);
        let width = field(w0, 32, 14) + 1;
        let height = field(w0, 46, 14) + 1;
        // The luma plane extent: libdrm sets it to the row length in pixels
        // (stride) and the height; it must cover the surface.
        let luma_width = field(w1, 0, 14) + 1;
        let luma_height = field(w1, 14, 14) + 1;
        if luma_width < width || luma_height < height {
            return Err(format!(
                "{name}: luma plane {luma_width}x{luma_height} smaller than surface {width}x{height}"
            ));
        }
        let row_bytes = luma_width * BYTES_PER_PIXEL;
        let layout = plane_layout(kind, block_height_log2, row_bytes, luma_height)
            .map_err(|e| format!("{name}: {e}"))?;
        Ok(Surface {
            address: u64::from(address) << 8,
            format,
            layout,
            width,
            height,
        })
    }

    fn offset(&self, x: u32, y: u32) -> u64 {
        match self.layout {
            Layout::Pitch(pitch) => u64::from(y) * pitch + u64::from(x * BYTES_PER_PIXEL),
            Layout::Block(surface, _) => surface.offset(x * BYTES_PER_PIXEL, y, 0),
        }
    }

    /// Bytes from the surface start that hold rows `top..bottom`, as
    /// a range whole blocks wide for block-linear surfaces.
    fn span(&self, top: u32, bottom: u32) -> std::ops::Range<u64> {
        match self.layout {
            Layout::Pitch(pitch) => {
                u64::from(top) * pitch
                    ..u64::from(bottom - 1) * pitch + u64::from(self.width * BYTES_PER_PIXEL)
            }
            Layout::Block(surface, lines) => {
                // Row of blocks `n` starts at the offset of its first line.
                let row = surface.offset(0, lines, 0);
                u64::from(top / lines) * row..u64::from((bottom - 1) / lines + 1) * row
            }
        }
    }

    /// Reads rows `top..bottom`.
    fn read(&self, memory: &DeviceMemory, top: u32, bottom: u32) -> Result<Rows, String> {
        let span = self.span(top, bottom);
        let mut bytes = vec![0; (span.end - span.start) as usize];
        memory
            .read(self.address + span.start, &mut bytes)
            .ok_or_else(|| format!("surface at {:#x} is not mapped", self.address + span.start))?;
        Ok(Rows {
            start: span.start,
            bytes,
        })
    }

    fn pixel(&self, rows: &Rows, x: u32, y: u32) -> [f32; 4] {
        let at = (self.offset(x, y) - rows.start) as usize;
        let byte = |index: usize| f32::from(rows.bytes[at + index]) / 255.0;
        let [r, g, b] = self.format.rgb.map(byte);
        let alpha = if self.format.stored_alpha {
            byte(self.format.alpha)
        } else {
            1.0
        };
        [r, g, b, alpha]
    }

    /// Copies `out.len()` bytes of row `y` from pixel `left` out of `rows`.
    fn read_row(&self, rows: &Rows, left: u32, y: u32, out: &mut [u8]) {
        row_runs(self.layout, left * BYTES_PER_PIXEL, y, out.len(), |at, run| {
            let at = (at - rows.start) as usize;
            out[run.clone()].copy_from_slice(&rows.bytes[at..at + run.len()]);
        });
    }

    /// Stores `line` (pixels as stored) at pixel `left` of row `y` in
    /// `bytes`, which hold the surface from byte `start`.
    fn write_row(&self, bytes: &mut [u8], start: u64, left: u32, y: u32, line: &[u8]) {
        row_runs(self.layout, left * BYTES_PER_PIXEL, y, line.len(), |at, run| {
            let at = (at - start) as usize;
            bytes[at..at + run.len()].copy_from_slice(&line[run]);
        });
    }

    /// `colors` as stored, a pixel after another. X formats store the
    /// alpha too (Ryujinx writes 35 as 31).
    fn encode_row(&self, colors: &[[f32; 4]], line: &mut [u8]) {
        let ([r, g, b], a) = (self.format.rgb, self.format.alpha);
        for (color, pixel) in colors.iter().zip(line.chunks_exact_mut(BYTES_PER_PIXEL as usize)) {
            pixel[r] = unorm8(color[0]);
            pixel[g] = unorm8(color[1]);
            pixel[b] = unorm8(color[2]);
            pixel[a] = unorm8(color[3]);
        }
    }

    /// How [`Surface::read`] holds rows from `top`: in runs of bytes, each
    /// all of a row (pitch-linear) or of a row of blocks (block-linear),
    /// as (bytes a run, rows a run, the first run's first row).
    fn runs(&self, top: u32) -> (usize, u32, u32) {
        match self.layout {
            Layout::Pitch(pitch) => (pitch as usize, 1, top),
            Layout::Block(surface, lines) => (
                surface.offset(0, lines, 0) as usize,
                lines,
                top / lines * lines,
            ),
        }
    }

    /// The surface a picture of this output configuration is written to
    /// with `OutputTranspose`: its rows are the picture's columns.
    ///
    /// hekate's Nyx turns its 1280×720 drawing for the 720×1280 panel this
    /// way (`vic.c`, "rotation 270": transpose and flip X). It declares the
    /// output surface with the picture's size, 1280×720, and the display
    /// controller then scans the buffer 720 pixels to the row: so the
    /// memory holds the transposed picture with rows as long as the
    /// declared luma plane is high \[inference from those two\]. That the
    /// flips come before the transposition, in the picture's own axes, is
    /// \[inference\] from the result: Nyx then stands upright on the
    /// screen, and flipped after it would stand on its head.
    fn transposed(self, [_, w1]: [u64; 2]) -> Result<Surface, String> {
        let Layout::Pitch(_) = self.layout else {
            return Err("transposed block-linear output".into());
        };
        let luma_height = field(w1, 14, 14) + 1;
        Ok(Surface {
            layout: Layout::Pitch(u64::from(luma_height * BYTES_PER_PIXEL)),
            width: self.height,
            height: self.width,
            ..self
        })
    }
}

/// The layout of a plane `row_bytes` wide and `rows` high of block kind
/// `kind`.
fn plane_layout(
    kind: u32,
    block_height_log2: u32,
    row_bytes: u32,
    rows: u32,
) -> Result<Layout, String> {
    match kind {
        // libdrm `VIC_BLK_KIND_PITCH`; pitch = stride × bytes per pixel.
        0 => Ok(Layout::Pitch(u64::from(row_bytes))),
        // `VIC_BLK_KIND_GENERIC_16Bx2`: the GPU's GOB layout.
        1 => Ok(Layout::Block(
            BlockLinearSurface::new(
                row_bytes,
                rows,
                BlockLinear {
                    block_height_log2,
                    block_depth_log2: 0,
                },
            ),
            GOB_HEIGHT << block_height_log2,
        )),
        _ => Err(format!("block kind {kind}")),
    }
}

/// The runs of `count` bytes of row `y` from byte column `x_bytes` in
/// `layout` that lie together in memory, `run(offset of the run from the
/// plane's start, its bytes in the row)`: the whole of it pitch-linear, the
/// 16-byte runs a GOB keeps together block-linear.
#[inline]
fn row_runs(
    layout: Layout,
    x_bytes: u32,
    y: u32,
    count: usize,
    mut run: impl FnMut(u64, std::ops::Range<usize>),
) {
    match layout {
        Layout::Pitch(pitch) => run(u64::from(y) * pitch + u64::from(x_bytes), 0..count),
        Layout::Block(surface, _) => {
            let mut done = 0;
            while done < count {
                let x = x_bytes + done as u32;
                let length = ((16 - x % 16) as usize).min(count - done);
                run(surface.offset(x, y, 0), done..done + length);
                done += length;
            }
        }
    }
}

/// How a YUV slot format stores chroma.
#[derive(Clone, Copy, Debug)]
enum ChromaLayout {
    /// U/V pairs in one plane; whether U is the lower byte.
    Interleaved { u_first: bool },
    /// U and V each in a plane of its own.
    Separate,
}

/// The 8-bit YUV slot input formats, their chroma layout and subsampling
/// (horizontal, vertical): the semi-planar 4:2:0 formats
/// (`vic_ceb6_types.h` `PIXELFORMAT`, the same codes as libdrm `vic.h`'s
/// for VIC 4.0 [assumption, as for [`planar_subsampling`]]) 67
/// `T_Y8___U8V8_N420` and 68 `T_Y8___V8U8_N420`, luma in one plane, Cb/Cr
/// pairs at half the resolution both ways in a second, and the fully planar
/// formats 69-72 of [`planar_subsampling`]. By the naming rule of
/// [`format`] the component named last is the lower byte, so 68 stores U
/// first (NV12, what NVDEC writes and the browser hands the VIC as 68) and
/// 67 V first.
fn yuv_input(code: u32) -> Option<(ChromaLayout, [u32; 2])> {
    match code {
        67 => Some((ChromaLayout::Interleaved { u_first: false }, [2, 2])),
        68 => Some((ChromaLayout::Interleaved { u_first: true }, [2, 2])),
        _ => planar_subsampling(code).map(|subsampling| (ChromaLayout::Separate, subsampling)),
    }
}

/// One plane of 8-bit samples, `row_bytes` of them a row.
#[derive(Clone, Copy, Debug)]
struct SamplePlane {
    address: u64,
    layout: Layout,
    row_bytes: u32,
}

impl SamplePlane {
    fn offset(&self, x_bytes: u32, y: u32) -> u64 {
        match self.layout {
            Layout::Pitch(pitch) => u64::from(y) * pitch + u64::from(x_bytes),
            Layout::Block(surface, _) => surface.offset(x_bytes, y, 0),
        }
    }

    /// Reads rows `top..bottom`, whole block rows of a block-linear plane.
    fn read(&self, memory: &DeviceMemory, top: u32, bottom: u32) -> Result<Rows, String> {
        let span = match self.layout {
            Layout::Pitch(pitch) => {
                u64::from(top) * pitch..u64::from(bottom - 1) * pitch + u64::from(self.row_bytes)
            }
            Layout::Block(surface, lines) => {
                let row = surface.offset(0, lines, 0);
                u64::from(top / lines) * row..u64::from((bottom - 1) / lines + 1) * row
            }
        };
        let mut bytes = vec![0; (span.end - span.start) as usize];
        memory
            .read(self.address + span.start, &mut bytes)
            .ok_or_else(|| format!("plane at {:#x} is not mapped", self.address + span.start))?;
        Ok(Rows {
            start: span.start,
            bytes,
        })
    }

    fn sample(&self, rows: &Rows, x_bytes: u32, y: u32) -> f32 {
        f32::from(rows.bytes[(self.offset(x_bytes, y) - rows.start) as usize]) / 255.0
    }

    /// Stores `line` at byte `x_bytes` of row `y` in `rows` (read by
    /// [`SamplePlane::read`] to hold that row).
    fn place(&self, rows: &mut Rows, x_bytes: u32, y: u32, line: &[u8]) {
        row_runs(self.layout, x_bytes, y, line.len(), |at, run| {
            let at = (at - rows.start) as usize;
            rows.bytes[at..at + run.len()].copy_from_slice(&line[run]);
        });
    }
}

/// The chroma planes of a [`Yuv`] slot surface.
#[derive(Clone, Copy, Debug)]
enum ChromaPlanes {
    Interleaved {
        plane: SamplePlane,
        u_first: bool,
    },
    /// U, V.
    Separate([SamplePlane; 2]),
}

/// A YUV slot surface ([`yuv_input`]).
///
/// Chroma siting (w0 bits 7-8 horizontal, 9-10 vertical; 0 with the even
/// luma sample, 1 between the even and the odd, 2 with the odd, as for
/// the output in [`Planar`]) places chroma sample `k` of a subsampled axis
/// at luma position `2k + siting / 2`; a luma sample's chroma is
/// interpolated linearly between the two chroma samples around it, the
/// edge ones repeated [inference: the VIC's upsampling filter is not
/// public]. On a full-resolution axis chroma sample `k` is luma sample
/// `k`'s, and any siting but 0 refuses.
#[derive(Clone, Copy, Debug)]
struct Yuv {
    luma: SamplePlane,
    chroma: ChromaPlanes,
    subsampling: [u32; 2],
    siting: [u32; 2],
    width: u32,
    height: u32,
    /// Chroma samples a row and rows.
    chroma_size: [u32; 2],
}

impl Yuv {
    fn new(
        [w0, w1]: [u64; 2],
        (layout, subsampling): (ChromaLayout, [u32; 2]),
        planes: [u32; 3],
        name: &str,
    ) -> Result<Yuv, String> {
        let siting = [field(w0, 7, 2), field(w0, 9, 2)];
        for (axis, (factor, siting)) in ["horizontal", "vertical"]
            .into_iter()
            .zip(subsampling.into_iter().zip(siting))
        {
            if siting == 3 || (factor == 1 && siting != 0) {
                return Err(format!(
                    "{name}: {axis} chroma siting {siting} with {factor}:1 subsampling"
                ));
            }
        }
        let kind = field(w0, 11, 4);
        let block_height_log2 = field(w0, 15, 4);
        let width = field(w0, 32, 14) + 1;
        let height = field(w0, 46, 14) + 1;
        let luma = [field(w1, 0, 14) + 1, field(w1, 14, 14) + 1];
        let chroma = [field(w1, 32, 14) + 1, field(w1, 46, 14) + 1];
        let chroma_size = [
            width.div_ceil(subsampling[0]),
            height.div_ceil(subsampling[1]),
        ];
        if luma[0] < width
            || luma[1] < height
            || chroma[0] < chroma_size[0]
            || chroma[1] < chroma_size[1]
        {
            return Err(format!(
                "{name}: planes {}x{} and {}x{} smaller than surface {width}x{height}",
                luma[0], luma[1], chroma[0], chroma[1]
            ));
        }
        let used = match layout {
            ChromaLayout::Interleaved { .. } => 2,
            ChromaLayout::Separate => 3,
        };
        if planes[..used].contains(&0) || planes[used..].iter().any(|&plane| plane != 0) {
            return Err(format!(
                "{name}: plane addresses {planes:#x?} of a {used}-plane YUV surface"
            ));
        }
        let plane = |address: u32, row_bytes: u32, rows: u32| {
            Ok::<_, String>(SamplePlane {
                address: u64::from(address) << 8,
                layout: plane_layout(kind, block_height_log2, row_bytes, rows)
                    .map_err(|e| format!("{name}: {e}"))?,
                row_bytes,
            })
        };
        let chroma_planes = match layout {
            ChromaLayout::Interleaved { u_first } => ChromaPlanes::Interleaved {
                plane: plane(planes[1], 2 * chroma[0], chroma[1])?,
                u_first,
            },
            ChromaLayout::Separate => ChromaPlanes::Separate([
                plane(planes[1], chroma[0], chroma[1])?,
                plane(planes[2], chroma[0], chroma[1])?,
            ]),
        };
        Ok(Yuv {
            luma: plane(planes[0], luma[0], luma[1])?,
            chroma: chroma_planes,
            subsampling,
            siting,
            width,
            height,
            chroma_size,
        })
    }

    /// The chroma rows luma rows `top..bottom` interpolate from.
    fn chroma_rows(&self, top: u32, bottom: u32) -> (u32, u32) {
        if self.subsampling[1] == 1 {
            return (top, bottom);
        }
        let last = self.chroma_size[1] - 1;
        (
            (top / 2).saturating_sub(1),
            (bottom.div_ceil(2) + 1).min(last + 1),
        )
    }

    /// Luma rows, then the rows of each chroma plane.
    fn read(&self, memory: &DeviceMemory, top: u32, bottom: u32) -> Result<Vec<Rows>, String> {
        let (chroma_top, chroma_bottom) = self.chroma_rows(top, bottom);
        let mut rows = vec![self.luma.read(memory, top, bottom)?];
        match &self.chroma {
            ChromaPlanes::Interleaved { plane, .. } => {
                rows.push(plane.read(memory, chroma_top, chroma_bottom)?);
            }
            ChromaPlanes::Separate(planes) => {
                for plane in planes {
                    rows.push(plane.read(memory, chroma_top, chroma_bottom)?);
                }
            }
        }
        Ok(rows)
    }

    /// Y, U, V of a pixel, normalized, and opaque alpha.
    fn pixel(&self, rows: &[Rows], x: u32, y: u32) -> [f32; 4] {
        // The two chroma samples around luma position `p` on an axis and
        // the weight of the second.
        let around = |p: u32, axis: usize| {
            let factor = self.subsampling[axis] as f32;
            let position = (p as f32 - self.siting[axis] as f32 / 2.0) / factor;
            let last = self.chroma_size[axis] - 1;
            let first = position.floor().max(0.0);
            let k = (first as u32).min(last);
            (k, (k + 1).min(last), (position - first).clamp(0.0, 1.0))
        };
        let (x0, x1, tx) = around(x, 0);
        let (y0, y1, ty) = around(y, 1);
        let mut uv = [0.0; 2];
        for (component, value) in uv.iter_mut().enumerate() {
            let at = |cx: u32, cy: u32| match &self.chroma {
                ChromaPlanes::Interleaved { plane, u_first } => {
                    let byte = if *u_first { component } else { 1 - component } as u32;
                    plane.sample(&rows[1], 2 * cx + byte, cy)
                }
                ChromaPlanes::Separate(planes) => {
                    planes[component].sample(&rows[1 + component], cx, cy)
                }
            };
            let top = at(x0, y0) * (1.0 - tx) + at(x1, y0) * tx;
            let bottom = at(x0, y1) * (1.0 - tx) + at(x1, y1) * tx;
            *value = top * (1.0 - ty) + bottom * ty;
        }
        [self.luma.sample(&rows[0], x, y), uv[0], uv[1], 1.0]
    }
}

/// A slot's input surface.
#[derive(Clone, Copy, Debug)]
enum Source {
    Rgb(Surface),
    Yuv(Yuv),
}

/// Rows of a slot's input surface held in host memory.
enum SourceRows {
    Rgb(Rows),
    /// Luma rows, then each chroma plane's.
    Yuv(Vec<Rows>),
}

impl Source {
    /// `words` are the slot surface configuration, `planes` the luma,
    /// chroma U and chroma V plane addresses.
    fn new(words: [u64; 2], planes: [u32; 3], name: &str) -> Result<Source, String> {
        if let Some(format) = yuv_input(field(words[0], 0, 7)) {
            return Yuv::new(words, format, planes, name).map(Source::Yuv);
        }
        // Chroma plane addresses are ignored, see [`Surface`].
        Surface::new(words, planes[0], name).map(Source::Rgb)
    }

    fn size(&self) -> (u32, u32) {
        match self {
            Source::Rgb(surface) => (surface.width, surface.height),
            Source::Yuv(surface) => (surface.width, surface.height),
        }
    }

    fn read(&self, memory: &DeviceMemory, top: u32, bottom: u32) -> Result<SourceRows, String> {
        Ok(match self {
            Source::Rgb(surface) => SourceRows::Rgb(surface.read(memory, top, bottom)?),
            Source::Yuv(surface) => SourceRows::Yuv(surface.read(memory, top, bottom)?),
        })
    }

    /// The pixel's components as stored (R, G, B or Y, U, V) and alpha.
    fn pixel(&self, rows: &SourceRows, x: u32, y: u32) -> [f32; 4] {
        match (self, rows) {
            (Source::Rgb(surface), SourceRows::Rgb(rows)) => surface.pixel(rows, x, y),
            (Source::Yuv(surface), SourceRows::Yuv(rows)) => surface.pixel(rows, x, y),
            _ => unreachable!("rows are read from the source they are used with"),
        }
    }
}

/// Rows of a surface held in host memory.
struct Rows {
    start: u64,
    bytes: Vec<u8>,
}

/// A normalized value as an 8-bit sample, rounded to nearest.
fn unorm8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Chroma subsampling (horizontal, vertical) of the fully planar 8-bit
/// YUV formats: Y, U and V each in a plane of its own, a byte a sample.
/// Codes from NVIDIA open-gpu-doc `classes/video/vic_ceb6_types.h`
/// `PIXELFORMAT` (VIC 5.1; its codes 31-38 and 67-68 are libdrm `vic.h`'s
/// for VIC 4.0, the enumeration is taken to be shared [assumption]):
/// 69 `T_Y8___U8___V8_N444`, 70 `N422`, 71 `N422R`, 72 `N420`. "422R" is
/// 4:2:2 rotated, chroma shared by two vertically adjacent luma samples
/// (NVIDIA Orin TRM, pixel formats: "422R means vertical down sampling").
fn planar_subsampling(code: u32) -> Option<[u32; 2]> {
    match code {
        69 => Some([1, 1]),
        70 => Some([2, 1]),
        71 => Some([1, 2]),
        72 => Some([2, 2]),
        _ => None,
    }
}

/// An 8-bit YUV output surface: the fully planar formats
/// ([`planar_subsampling`]) and the semi-planar 4:2:0 ones, 67 and 68
/// ([`yuv_input`]: 68 is NV12, U the lower byte of a pair), pitch-linear
/// or block-linear (16Bx2 GOBs, the block kind and height of w0 as for a
/// slot surface).
///
/// The output surface configuration (libdrm `vic40.h`
/// `OutputSurfaceConfig`) gives the padded luma and chroma plane widths in
/// samples; as for the RGB formats, where libdrm sets the luma width to its
/// stride (`vic.c`: pitch = stride × bytes per pixel), each is the plane's
/// pitch, here in samples (a U/V pair of an interleaved plane being one
/// sample of two bytes). The output colour matrix gives Y, U and V as its
/// rows 0, 1 and 2 (`vic_ceb6_types.h` names the output clamps "R/Y",
/// "G/U", "B/V").
///
/// Chroma siting (`OutChromaLocHoriz`/`Vert`, `vic_ceb6_types.h`: 0 with
/// the even luma sample, 1 between the even and the odd, 2 with the odd):
/// on a subsampled axis chroma sample `k` sits at luma position
/// `2k + siting / 2`, and its value is the linear interpolation there of
/// the two luma positions' chroma, each clamped to 0..1 first, then
/// quantized: the mean of the two for siting 1, the even one's own for 0,
/// the odd one's for 2 [inference: the value at that point by linear
/// interpolation, after the clamp that ends the per-pixel pipeline; the
/// VIC's decimation filter is not public]. Siting 3, and any but 0 on a
/// full-resolution axis, refuse. grc's encoder input (Minecraft, video
/// capture on) is NV12 block-linear (2-GOB blocks), 1280×720, chroma 0
/// horizontally and 1 vertically: the MPEG-2/H.264 default siting.
#[derive(Clone, Copy, Debug)]
struct Planar {
    luma: SamplePlane,
    chroma: ChromaPlanes,
    subsampling: [u32; 2],
    siting: [u32; 2],
    width: u32,
    height: u32,
}

impl Planar {
    fn new(
        [w0, w1]: [u64; 2],
        addresses: &[u32],
        (layout, subsampling): (ChromaLayout, [u32; 2]),
    ) -> Result<Planar, String> {
        let siting = [field(w0, 7, 2), field(w0, 9, 2)];
        for (axis, (factor, siting)) in ["horizontal", "vertical"]
            .into_iter()
            .zip(subsampling.into_iter().zip(siting))
        {
            if siting == 3 || (factor == 1 && siting != 0) {
                return Err(format!(
                    "output: {axis} chroma siting {siting} with {factor}:1 subsampling"
                ));
            }
        }
        let kind = field(w0, 11, 4);
        let block_height_log2 = field(w0, 15, 4);
        let width = field(w0, 32, 14) + 1;
        let height = field(w0, 46, 14) + 1;
        let luma = [field(w1, 0, 14) + 1, field(w1, 14, 14) + 1];
        let chroma = [field(w1, 32, 14) + 1, field(w1, 46, 14) + 1];
        let [sx, sy] = subsampling;
        if luma[0] < width
            || luma[1] < height
            || chroma[0] < width.div_ceil(sx)
            || chroma[1] < height.div_ceil(sy)
        {
            return Err(format!(
                "output: planes {}x{} and {}x{} smaller than surface {width}x{height}",
                luma[0], luma[1], chroma[0], chroma[1]
            ));
        }
        let used = match layout {
            ChromaLayout::Interleaved { .. } => 2,
            ChromaLayout::Separate => 3,
        };
        if addresses[..used].contains(&0) {
            return Err(format!("output: plane addresses {addresses:#x?}"));
        }
        let plane = |address: u32, row_bytes: u32, rows: u32| {
            Ok::<_, String>(SamplePlane {
                address: u64::from(address) << 8,
                layout: plane_layout(kind, block_height_log2, row_bytes, rows)
                    .map_err(|e| format!("output: {e}"))?,
                row_bytes,
            })
        };
        let chroma_planes = match layout {
            ChromaLayout::Interleaved { u_first } => ChromaPlanes::Interleaved {
                plane: plane(addresses[1], 2 * chroma[0], chroma[1])?,
                u_first,
            },
            ChromaLayout::Separate => ChromaPlanes::Separate([
                plane(addresses[1], chroma[0], chroma[1])?,
                plane(addresses[2], chroma[0], chroma[1])?,
            ]),
        };
        Ok(Planar {
            luma: plane(addresses[0], luma[0], luma[1])?,
            chroma: chroma_planes,
            subsampling,
            siting,
            width,
            height,
        })
    }

    /// Writes `colors` (Y, U, V, alpha; `area`'s rows, top first) into
    /// `area`, which must not split a chroma sample.
    fn write(&self, memory: &DeviceMemory, area: Rect, colors: &[[f32; 4]]) -> Result<(), String> {
        let [sx, sy] = self.subsampling;
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
        let at =
            |x: u32, y: u32| colors[(y - area.top) as usize * width + (x - area.left) as usize];
        // The luma positions a chroma sample takes its value from on each
        // axis, with their weights (see the siting above).
        let taps = |axis: usize, k: u32| -> [(u32, f32); 2] {
            if self.subsampling[axis] == 1 {
                return [(k, 1.0), (k, 0.0)];
            }
            let odd = self.siting[axis] as f32 / 2.0;
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
        // Each plane's bytes of the area, a row of them a line, computed
        // in bands of rows by `line(row, bytes)`; then placed in the
        // plane's layout.
        let fill = |rect: Rect, samples: u32, line: &(dyn Fn(u32, &mut [u8]) + Sync)| {
            let row = (rect.width() * samples) as usize;
            let mut bytes = vec![0u8; row * rect.height() as usize];
            let work = bytes.len();
            in_bands(&mut bytes, row, work, |first, band| {
                for (y, out) in (rect.top + first as u32..).zip(band.chunks_mut(row)) {
                    line(y, out);
                }
            });
            bytes
        };
        let luma = |y: u32, out: &mut [u8]| {
            let row = &colors[(y - area.top) as usize * width..][..width];
            for (byte, color) in out.iter_mut().zip(row) {
                *byte = unorm8(color[0]);
            }
        };
        let mut planes = vec![(self.luma, area, fill(area, 1, &luma))];
        match self.chroma {
            ChromaPlanes::Interleaved { plane, u_first } => {
                let line = |cy: u32, out: &mut [u8]| {
                    for (cx, pair) in (chroma_area.left..).zip(out.chunks_exact_mut(2)) {
                        for (byte, value) in pair.iter_mut().enumerate() {
                            let component = if u_first { 1 + byte } else { 2 - byte };
                            *value = unorm8(chroma(cx, cy, component));
                        }
                    }
                };
                planes.push((plane, chroma_area, fill(chroma_area, 2, &line)));
            }
            ChromaPlanes::Separate(separate) => {
                for (index, plane) in separate.into_iter().enumerate() {
                    let line = |cy: u32, out: &mut [u8]| {
                        for (cx, value) in (chroma_area.left..).zip(out) {
                            *value = unorm8(chroma(cx, cy, 1 + index));
                        }
                    };
                    planes.push((plane, chroma_area, fill(chroma_area, 1, &line)));
                }
            }
        }
        // Every plane is read before any is written: an unmapped one leaves
        // the others as they were.
        let mut stored = Vec::with_capacity(planes.len());
        for (plane, rect, _) in &planes {
            stored.push(plane.read(memory, rect.top, rect.bottom)?);
        }
        for ((plane, rect, bytes), mut rows) in planes.into_iter().zip(stored) {
            let row = bytes.len() / rect.height() as usize;
            let left = rect.left * row as u32 / rect.width();
            for (y, line) in (rect.top..).zip(bytes.chunks(row)) {
                plane.place(&mut rows, left, y, line);
            }
            memory
                .write(plane.address + rows.start, &rows.bytes)
                .ok_or_else(|| format!("plane at {:#x} is not mapped", plane.address))?;
        }
        Ok(())
    }
}

/// The output surface.
enum Output {
    /// A 32-bit RGB surface, one plane.
    Rgb(Surface),
    Planar(Planar),
}

impl Output {
    /// `words` are the output surface configuration, `planes` the luma,
    /// chroma U and chroma V plane addresses.
    fn new(words: [u64; 2], planes: &[u32]) -> Result<Output, String> {
        if let Some(format) = yuv_input(field(words[0], 0, 7)) {
            return Planar::new(words, planes, format).map(Output::Planar);
        }
        // Chroma plane addresses are ignored, see [`Surface`].
        Surface::new(words, planes[0], "output").map(Output::Rgb)
    }

    fn size(&self) -> (u32, u32) {
        match self {
            Output::Rgb(surface) => (surface.width, surface.height),
            Output::Planar(planar) => (planar.width, planar.height),
        }
    }

    /// Writes `colors` (`area`'s rows, top first) into `area`.
    fn write(&self, memory: &DeviceMemory, area: Rect, colors: &[[f32; 4]]) -> Result<(), String> {
        let surface = match self {
            Output::Rgb(surface) => surface,
            Output::Planar(planar) => return planar.write(memory, area, colors),
        };
        let mut rows = surface.read(memory, area.top, area.bottom)?;
        let (run, lines, first_line) = surface.runs(area.top);
        let start = rows.start;
        let width = area.width() as usize;
        in_bands(&mut rows.bytes, run, colors.len(), |first, band| {
            let top = first_line + first as u32 * lines;
            let bottom = top + band.len().div_ceil(run) as u32 * lines;
            let base = start + (first * run) as u64;
            let mut line = vec![0u8; width * BYTES_PER_PIXEL as usize];
            for y in top.max(area.top)..bottom.min(area.bottom) {
                let row = &colors[(y - area.top) as usize * width..][..width];
                surface.encode_row(row, &mut line);
                surface.write_row(band, base, area.left, y, &line);
            }
        });
        memory
            .write(surface.address + rows.start, &rows.bytes)
            .ok_or_else(|| {
                format!(
                    "output at {:#x} is not mapped",
                    surface.address + rows.start
                )
            })
    }
}

/// The configuration structure as 64-bit words.
struct Config([u64; CONFIG_SIZE / 8]);

impl Config {
    fn word(&self, offset: usize) -> u64 {
        self.0[offset / 8]
    }

    fn pair(&self, offset: usize) -> [u64; 2] {
        [self.word(offset), self.word(offset + 8)]
    }
}

/// An enabled colour matrix (libdrm `MatrixStruct`): 20-bit signed
/// coefficients `C[row][column]`, row = output component. Ryujinx
/// `Blender.cs` (MIT) applies it to 10-bit components as
/// `((Σ in·C >> RShift) + C3) >> 8`; the same arithmetic is done here on
/// normalized values without the intermediate truncations (the datapath's
/// precision is not modeled), which keeps linear-light values intact.
#[derive(Clone, Copy, Debug)]
struct Matrix([[f32; 4]; 3]);

impl Matrix {
    fn new(words: [u64; 4]) -> Option<Matrix> {
        let [w0, w1, w2, w3] = words;
        flag(w1, 63).then(|| {
            let scale = (1u64 << (field(w0, 60, 4) + 8)) as f32;
            let signed = |word: u64, row: u32| {
                let value = i64::from(field(word, row * 20, 20));
                (value - ((value & 0x80000) << 1)) as f32
            };
            Matrix(std::array::from_fn(|row| {
                let row = row as u32;
                [
                    signed(w0, row) / scale,
                    signed(w1, row) / scale,
                    signed(w2, row) / scale,
                    // The offset is added after the shift, in 10-bit units.
                    signed(w3, row) / 256.0 / 1023.0,
                ]
            }))
        })
    }

    fn apply(&self, color: [f32; 4]) -> [f32; 4] {
        let mut out = color;
        for (channel, row) in self.0.iter().enumerate() {
            out[channel] = color[0] * row[0] + color[1] * row[1] + color[2] * row[2] + row[3];
        }
        out
    }
}

/// Degamma/regamma mode 1: the sRGB transfer function (IEC 61966-2-1),
/// decoding to linear light in the slot and encoding again at the output.
/// nvnflinger's serializer (`0xAEA60`) sets mode 1 at both ends on its
/// linear-blending path for its colour spaces 0 and 1 and mode 2 for the
/// others [INFERENCE: mode 1 = sRGB; for the 1:1 replace it issues at the
/// netConnect launch the pair cancels whatever the exact curve].
const GAMMA_SRGB: u32 = 1;

fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(value: f32) -> f32 {
    if value <= 0.0031308 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

/// The output's regamma of a composed value, clamped to 0..1 first.
fn clamped_linear_to_srgb(value: f32) -> f32 {
    linear_to_srgb(value.clamp(0.0, 1.0))
}

/// log2 of the sets of a [`Memo`].
const MEMO_SET_BITS: u32 = 10;
/// Entries a [`Memo`] set holds.
const MEMO_WAYS: usize = 4;

/// A memo of a function of one `f32` (the gamma curves, each a `powf`,
/// which would otherwise be much of an EXECUTE's host time): a set
/// associative table of arguments, by their bits, and the function's
/// value for each. A value is taken from it only for the very argument it
/// was computed from, so what it returns is the function's own result.
/// Pictures hold few distinct values per component (8-bit sources), so
/// nearly every call finds its argument.
struct Memo {
    function: fn(f32) -> f32,
    sets: Vec<[(u32, f32); MEMO_WAYS]>,
    /// The way the next miss replaces, in turn.
    next: usize,
}

impl Memo {
    fn new(function: fn(f32) -> f32) -> Memo {
        // Every entry starts as 0.0 and its value.
        Memo {
            function,
            sets: vec![[(0, function(0.0)); MEMO_WAYS]; 1 << MEMO_SET_BITS],
            next: 0,
        }
    }

    fn get(&mut self, argument: f32) -> f32 {
        let bits = argument.to_bits();
        // MurmurHash3's finalizer: every argument bit moves the set.
        let mut hash = bits ^ bits >> 16;
        hash = hash.wrapping_mul(0x85eb_ca6b);
        hash ^= hash >> 13;
        hash = hash.wrapping_mul(0xc2b2_ae35);
        hash ^= hash >> 16;
        let set = &mut self.sets[(hash >> (32 - MEMO_SET_BITS)) as usize];
        if let Some(&(_, value)) = set.iter().find(|entry| entry.0 == bits) {
            return value;
        }
        let value = (self.function)(argument);
        set[self.next] = (bits, value);
        self.next = (self.next + 1) % MEMO_WAYS;
        value
    }
}

/// Blending factor selections (libdrm `vic.h` `VIC_BLEND_*`): source and
/// destination colour, source and destination alpha.
#[derive(Clone, Copy)]
struct Blend {
    k1: f32,
    k2: f32,
    source_color: u32,
    destination_color: u32,
    source_alpha: u32,
    destination_alpha: u32,
    overrides: [Option<f32>; 4],
    masks: [bool; 4],
}

impl Blend {
    fn new([w0, w1]: [u64; 2]) -> Result<Blend, String> {
        let mut blend = Blend {
            k1: field(w0, 0, 10) as f32 / 1023.0,
            k2: field(w0, 16, 10) as f32 / 1023.0,
            source_color: field(w0, 32, 3),
            destination_color: field(w0, 36, 3),
            source_alpha: field(w0, 40, 3),
            destination_alpha: field(w0, 44, 3),
            overrides: std::array::from_fn(|channel| {
                flag(w1, 44 + channel as u32)
                    .then(|| field(w1, 2 + 10 * channel as u32, 10) as f32 / 1023.0)
            }),
            masks: std::array::from_fn(|channel| flag(w1, 48 + channel as u32)),
        };
        // A blending structure left all zero copies the slot over what is
        // below it [inference: hekate leaves it so ("No need to configure.
        // Reset to zeros", `vic.c`) and its picture arrives; by the
        // selections' numbering zero would be "source × K1" with K1 0,
        // nothing at all].
        if w0 == 0 && w1 == 0 {
            (blend.k1, blend.destination_color, blend.destination_alpha) = (1.0, 5, 2);
        }
        if blend.source_color > 4
            || blend.destination_color > 6
            || blend.source_alpha > 3
            || blend.destination_alpha > 3
        {
            return Err(format!("blending selection {:#x}", field(w0, 32, 15)));
        }
        Ok(blend)
    }

    /// `source` and `destination` are straight colours with alpha.
    fn apply(&self, source: [f32; 4], destination: [f32; 4]) -> [f32; 4] {
        let (k1, k2) = (self.k1, self.k2);
        let (source_alpha, destination_alpha) = (source[3], destination[3]);
        let source_color = match self.source_color {
            0 => k1,
            1 => k1 * destination_alpha,
            2 => 1.0 - k1 * destination_alpha,
            3 => k1 * source_alpha,
            _ => 0.0,
        };
        let destination_color = match self.destination_color {
            0 => k1,
            1 => k2,
            2 => k1 * destination_alpha,
            3 => 1.0 - k1 * destination_alpha,
            4 => 1.0 - k1 * source_alpha,
            5 => 0.0,
            _ => 1.0,
        };
        let source_alpha_factor = match self.source_alpha {
            0 => k1,
            1 => k2,
            2 => 1.0 - k1 * destination_alpha,
            _ => 0.0,
        };
        let destination_alpha_factor = match self.destination_alpha {
            0 => k2,
            1 => 1.0 - k1 * source_alpha,
            2 => 0.0,
            _ => 1.0,
        };
        let mut out = [0.0; 4];
        for channel in 0..3 {
            out[channel] =
                source[channel] * source_color + destination[channel] * destination_color;
        }
        out[3] = source_alpha * source_alpha_factor + destination_alpha * destination_alpha_factor;
        for channel in 0..4 {
            if self.masks[channel] {
                out[channel] = destination[channel];
            }
        }
        out
    }
}

/// The scaler on one axis (`vic_ceb6_types.h`: "SourceRect -> DestRect
/// Shape Transform"): the source
/// samples, with their weights, that destination sample `d` (counted from
/// the destination rectangle's start) is filtered from. `start` and
/// `source_size` are the source rectangle on this axis.
///
/// `length` is the slot's filter length (libdrm `vic40.h` `SlotConfig`
/// `FilterLengthX`/`Y`, w0 bits 22-23 and 20-21; `vic_ceb6_types.h`
/// `FILTER_LENGTH`: 0 one tap "(Nearst)", 1 two taps "(Bilinear)", 2 five
/// taps, 3 ten taps). Destination sample `d`'s centre falls on source
/// position `start + (d + ½)·ratio − ½` (sample centres on integers, ratio
/// = source size / destination size) [inference: the rectangles' edges
/// coincide]. One tap takes the sample nearest it; two interpolate
/// linearly between the two around it. Positions past the rectangle take
/// its edge samples (`vic_ceb6_types.h` `SrcMaskPixelFillMode` 0: "padding
/// with nearest boundary pixels").
///
/// The five- and ten-tap filters read their coefficients from the filter
/// structure at SET_FILTER_STRUCT_OFFSET, whose layout is not public
/// (libdrm `vic30.c` and `vic41.c` hand the VIC a zeroed one; nvnflinger
/// uploads a 0x2080-byte table from its read-only data at `0xFB988`,
/// `0xAE960`). They are taken as interpolating linearly when enlarging and
/// as the mean over each destination sample's footprint, each source
/// sample weighted by the part of it covered, when reducing [inference: a
/// low-pass filter as wide as the ratio, the response such a table is
/// built to approach; the coefficients themselves are not decoded]. With
/// equal sizes every filter copies [inference: at phase 0 an interpolating
/// filter passes the sample through].
fn taps(
    length: u32,
    start: u32,
    source_size: u32,
    destination_size: u32,
    d: u32,
) -> Vec<(u32, f32)> {
    if source_size == destination_size {
        return vec![(start + d, 1.0)];
    }
    let ratio = f64::from(source_size) / f64::from(destination_size);
    let (first, last) = (f64::from(start), f64::from(start + source_size - 1));
    let sample = |position: f64| position.clamp(first, last) as u32;
    let centre = first + (f64::from(d) + 0.5) * ratio - 0.5;
    if length == 0 {
        return vec![(sample(centre.round()), 1.0)];
    }
    if length == 1 || ratio < 1.0 {
        let below = centre.floor();
        let weight = (centre - below) as f32;
        return vec![(sample(below), 1.0 - weight), (sample(below + 1.0), weight)];
    }
    let low = first + f64::from(d) * ratio;
    let high = low + ratio;
    let mut taps = Vec::with_capacity(ratio.ceil() as usize + 1);
    let mut edge = low.floor();
    while edge < high {
        let covered = high.min(edge + 1.0) - low.max(edge);
        if covered > 0.0 {
            taps.push((sample(edge), (covered / ratio) as f32));
        }
        edge += 1.0;
    }
    taps
}

/// One enabled slot, ready to draw.
struct Slot {
    surface: Source,
    source: Rect,
    destination: Rect,
    /// Filter lengths, horizontal and vertical ([`taps`]).
    filter: [u32; 2],
    matrix: Option<Matrix>,
    /// sRGB decoding (degamma mode 1) after the colour matrix.
    degamma: bool,
    clamp: (f32, f32),
    planar_alpha: f32,
    constant_alpha: bool,
    blend: Blend,
}

impl Slot {
    /// `Ok(None)` for a disabled slot.
    fn new(
        config: &Config,
        methods: &[u32; method::COUNT],
        index: usize,
    ) -> Result<Option<Slot>, String> {
        let base = SLOT_BASE + index * SLOT_SIZE;
        let w0 = config.word(base);
        if !flag(w0, 0) {
            return Ok(None);
        }
        let name = format!("slot {index}");
        let unmodeled = |what: &str| Err(format!("{name}: {what}"));
        // DeNoise, AdvancedDenoise, CadenceDetect, MotionMap, MMapCombine;
        // IsEven/ChromaEven only select fields of interlaced frames.
        if field(w0, 1, 5) != 0 {
            return unmodeled(&format!(
                "denoise/cadence/motion bits {:#x}",
                field(w0, 1, 5)
            ));
        }
        // The current field alone, or none named at all: hekate leaves all
        // eight clear for its progressive frame and the frame is fetched
        // all the same [inference: its picture arrives].
        if field(w0, 9, 7) != 0 {
            return unmodeled(&format!("field enables {:#x}", field(w0, 8, 8)));
        }
        if field(w0, 16, 4) != 0 {
            return unmodeled(&format!("frame format {}", field(w0, 16, 4)));
        }
        if field(w0, 24, 12) != 0 {
            return unmodeled(&format!("panoramic {:#x}", field(w0, 24, 12)));
        }
        // The noise and detail filter strengths of w1 act with the scaler
        // (refused below when it resamples); its deinterlace mode only with
        // interlaced frame formats, refused above.
        let w2 = config.word(base + 0x10);
        let clamp = (
            field(w2, 0, 10) as f32 / 1023.0,
            field(w2, 10, 10) as f32 / 1023.0,
        );
        if field(w2, 43, 3) != 0 {
            return unmodeled(&format!("stereo interleave {}", field(w2, 43, 3)));
        }
        if flag(w2, 46) {
            return unmodeled("clip enable");
        }
        if field(w2, 47, 8) != 0 {
            return unmodeled(&format!("clear rectangle mask {:#x}", field(w2, 47, 8)));
        }
        let degamma = field(w2, 55, 2);
        if !matches!(degamma, 0 | GAMMA_SRGB) {
            return unmodeled(&format!("degamma mode {degamma}"));
        }
        if flag(w2, 58) {
            return unmodeled("decompression");
        }
        let [w4, w5] = config.pair(base + 0x20);
        // 16.16 fixed point, right/bottom inclusive.
        let fixed = |word: u64, low: u32| field(word, low, 30);
        let edges = [fixed(w4, 0), fixed(w5, 0), fixed(w4, 32), fixed(w5, 32)];
        if edges.iter().any(|edge| edge & 0xFFFF != 0) {
            return unmodeled(&format!("fractional source rectangle {edges:#x?}"));
        }
        let [left, top, right, bottom] = edges.map(|edge| edge >> 16);
        let source = Rect {
            left,
            top,
            right: right + 1,
            bottom: bottom + 1,
        };
        let destination = Rect::inclusive(config.word(base + 0x30));
        if source.width() == 0 || source.height() == 0 {
            return unmodeled(&format!("empty source rectangle {source:?}"));
        }
        let filter = [field(w0, 22, 2), field(w0, 20, 2)];
        // FilterNoise, FilterDetail, ChromaNoise, ChromaDetail: w1 bits 0-39.
        let strengths = config.word(base + 8) & ((1 << 40) - 1);
        let scaling =
            source.width() != destination.width() || source.height() != destination.height();
        if scaling && strengths != 0 {
            return unmodeled(&format!(
                "noise/detail filter strengths {strengths:#x} while scaling"
            ));
        }
        let plane = |p: usize| methods[method::SURFACES.start + index * 24 + p];
        let surface = Source::new(
            config.pair(base + 0x40),
            [plane(0), plane(1), plane(2)],
            &name,
        )?;
        let (width, height) = surface.size();
        if source.right > width || source.bottom > height {
            return unmodeled(&format!(
                "source {source:?} outside surface {width}x{height}"
            ));
        }
        if flag(config.word(base + 0x58), 40) {
            return unmodeled("luma key");
        }
        let matrix_at =
            |offset: usize| [0, 8, 16, 24].map(|word| config.word(base + offset + word));
        if Matrix::new(matrix_at(0x80)).is_some() {
            return unmodeled("gamut matrix");
        }
        let matrix = Matrix::new(matrix_at(0x60));
        if matches!(surface, Source::Yuv(_)) && matrix.is_none() {
            return unmodeled("YUV input without a colour matrix");
        }
        Ok(Some(Slot {
            surface,
            source,
            destination,
            filter,
            matrix,
            degamma: degamma == GAMMA_SRGB,
            clamp,
            planar_alpha: field(w2, 32, 10) as f32 / 1023.0,
            constant_alpha: flag(w2, 42),
            blend: Blend::new(config.pair(base + 0xA0)).map_err(|e| format!("{name}: {e}"))?,
        }))
    }

    /// The source samples and weights of destination column (`axis` 0) or
    /// row (1) `at`, which lies in the destination rectangle.
    fn taps(&self, axis: usize, at: u32) -> Vec<(u32, f32)> {
        let (s, d) = (self.source, self.destination);
        let (start, source, destination, origin) = if axis == 0 {
            (s.left, s.width(), d.width(), d.left)
        } else {
            (s.top, s.height(), d.height(), d.top)
        };
        taps(self.filter[axis], start, source, destination, at - origin)
    }

    /// The scaled source pixel of `columns` × `lines` taps, its components
    /// as stored: the scaler works ahead of the colour matrix [inference:
    /// `vic_ceb6_types.h` lists the slot's scaler parameters before its
    /// colour conversion].
    fn sample(&self, rows: &SourceRows, columns: &[(u32, f32)], lines: &[(u32, f32)]) -> [f32; 4] {
        // A copy's single tap of weight 1 is the pixel itself (the sum
        // below would add it, times 1, to zeros).
        if let ([(x, column_weight)], [(y, row_weight)]) = (columns, lines)
            && *column_weight == 1.0
            && *row_weight == 1.0
        {
            return self.surface.pixel(rows, *x, *y);
        }
        let mut pixel = [0.0; 4];
        for &(y, row_weight) in lines {
            for &(x, column_weight) in columns {
                let weight = row_weight * column_weight;
                let value = self.surface.pixel(rows, x, y);
                for (sum, value) in pixel.iter_mut().zip(value) {
                    *sum += value * weight;
                }
            }
        }
        pixel
    }

    /// The slot's colour of a source pixel, before blending: colour matrix
    /// (converts the source's encoding), soft clamp, then degamma to linear
    /// light for blending [INFERENCE: the order each stage's input implies;
    /// the observed configuration has no slot matrix]. `degamma` is the
    /// composing thread's memo of [`srgb_to_linear`].
    fn color(&self, pixel: [f32; 4], degamma: &mut Memo) -> [f32; 4] {
        let mut color = self.matrix.map_or(pixel, |matrix| matrix.apply(pixel));
        for value in &mut color[..3] {
            *value = value.clamp(self.clamp.0, self.clamp.1);
            if self.degamma {
                *value = degamma.get(*value);
            }
        }
        color[3] = if self.constant_alpha {
            self.planar_alpha
        } else {
            pixel[3] * self.planar_alpha
        };
        for (channel, value) in self.blend.overrides.iter().enumerate() {
            if let Some(value) = value {
                color[channel] = *value;
            }
        }
        color
    }
}

/// Runs one EXECUTE with the method state `methods`.
fn compose(memory: &DeviceMemory, methods: &[u32; method::COUNT]) -> Result<(), String> {
    let config_address = u64::from(methods[method::SET_CONFIG_STRUCT_OFFSET]) << 8;
    let mut bytes = [0; CONFIG_SIZE];
    memory
        .read(config_address, &mut bytes)
        .ok_or_else(|| format!("configuration at {config_address:#x} is not mapped"))?;
    let config = Config(std::array::from_fn(|i| {
        u64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap())
    }));
    compose_config(memory, methods, &config, true)
        .map_err(|reason| format!("{reason}; {}", dump(methods, &config)))
}

/// The non-zero methods and configuration words, for a refusal note: with
/// the surfaces they name, a replayable description of the EXECUTE.
fn dump(methods: &[u32; method::COUNT], config: &Config) -> String {
    let methods: Vec<String> = (methods.iter().enumerate())
        .filter(|(_, value)| **value != 0)
        .map(|(i, value)| format!("{:03x}={value:x}", i * 4))
        .collect();
    let words: Vec<String> = (config.0.iter().enumerate())
        .filter(|(_, word)| **word != 0)
        .map(|(i, word)| format!("{:03x}:{word:016x}", i * 8))
        .collect();
    format!("methods {} config {}", methods.join(" "), words.join(" "))
}

/// Composes the EXECUTE of `methods` and `config`; `fused` lets a lone
/// copying slot take [`Fused`] (the tests compare it with the general
/// path).
fn compose_config(
    memory: &DeviceMemory,
    methods: &[u32; method::COUNT],
    config: &Config,
    fused: bool,
) -> Result<(), String> {
    let pipe = config.word(0);
    // nvnflinger writes 4 (`0xAEA60`: `| 0x40004`) or 0, libdrm 0; both
    // leave the picture at full size [INFERENCE: 4 is 1.0 in 2-bit fixed].
    for (name, value) in [
        ("horizontal", field(pipe, 0, 11)),
        ("vertical", field(pipe, 16, 11)),
    ] {
        if !matches!(value, 0 | 4) {
            return Err(format!("{name} downsampling {value}"));
        }
    }
    let [output, target] = config.pair(0x10);
    let alpha_fill = field(output, 0, 3);
    let background = [
        field(output, 16, 10),
        field(output, 26, 10),
        field(output, 36, 10),
        field(output, 6, 10),
    ]
    .map(|value| value as f32 / 1023.0);
    let regamma_mode = field(output, 46, 2);
    if !matches!(regamma_mode, 0 | GAMMA_SRGB) {
        return Err(format!("regamma mode {regamma_mode}"));
    }
    // OutputFlipX, OutputFlipY, OutputTranspose (libdrm `vic.h`
    // `OutputConfig`). The flips mirror the composed picture within the
    // target rectangle [inference: the rectangle is what is written];
    // the browser's frames come bottom-up and are flipped vertically.
    // Transposition then writes the picture's rows as the output's
    // columns: see `Surface::transposed`.
    let (flip_x, flip_y) = (flag(output, 48), flag(output, 49));
    let transpose = flag(output, 50);
    if !matches!(alpha_fill, 0 | 1 | 4) {
        return Err(format!("alpha fill mode {alpha_fill}"));
    }
    let output = Output::new(config.pair(0x20), &methods[method::SET_OUTPUT_SURFACE])?;
    // Identity in the configuration nvnflinger sends at the netConnect
    // launch (coefficients 0x40000, shift 10); BT.601 full-range RGB to
    // YCbCr (JFIF) in the EXECUTE with the planar YUV output seen when
    // News opens.
    let output_matrix = Matrix::new([0x30, 0x38, 0x40, 0x48].map(|offset| config.word(offset)));
    if matches!(output, Output::Planar(_)) && output_matrix.is_none() {
        return Err("planar YUV output without an output colour matrix".into());
    }
    let (surface_width, surface_height) = output.size();
    let area = Rect::inclusive(target).intersect(Rect {
        left: 0,
        top: 0,
        right: surface_width,
        bottom: surface_height,
    });
    let mut slots = Vec::new();
    for index in 0..SLOT_COUNT {
        slots.extend(Slot::new(config, methods, index)?);
    }
    if area.width() == 0 || area.height() == 0 {
        return Ok(());
    }
    let mut drawn = Vec::new();
    for slot in &slots {
        let rect = slot.destination.intersect(area);
        if rect.width() == 0 || rect.height() == 0 {
            continue;
        }
        let columns: Vec<_> = (rect.left..rect.right).map(|x| slot.taps(0, x)).collect();
        let lines: Vec<_> = (rect.top..rect.bottom).map(|y| slot.taps(1, y)).collect();
        let used = lines.iter().flatten().map(|&(y, _)| y);
        let (top, bottom) = (used.clone().min().unwrap(), used.max().unwrap() + 1);
        let rows = slot.surface.read(memory, top, bottom)?;
        drawn.push(Drawn {
            slot,
            rect,
            columns,
            lines,
            rows,
        });
    }
    let (width, height) = (area.width() as usize, area.height() as usize);
    // The output's colour matrix and alpha fill, after regamma.
    let encode = |mut color: [f32; 4]| {
        // Converts to the output's encoding after regamma [INFERENCE: a
        // YUV output is formed from encoded RGB].
        if let Some(matrix) = output_matrix {
            color = matrix.apply(color);
        }
        color[3] = match alpha_fill {
            0 => 1.0,
            1 => background[3],
            _ => color[3],
        };
        color
    };
    let mut colors = vec![[0.0; 4]; width * height];
    if let Some(lone) = fused.then(|| Fused::new(&drawn, background, regamma_mode)).flatten() {
        let drawn = &drawn[0];
        let outside = encode(lone.outside);
        in_bands(&mut colors, width, width * height, |first, band| {
            let mut line = vec![0u8; drawn.columns.len() * BYTES_PER_PIXEL as usize];
            for (out_y, out_row) in (first..).zip(band.chunks_mut(width)) {
                let y = if flip_y { height - 1 - out_y } else { out_y };
                let row = area.top + y as u32;
                let mut put =
                    |x: usize, color| out_row[if flip_x { width - 1 - x } else { x }] = color;
                if !(drawn.rect.top..drawn.rect.bottom).contains(&row) {
                    (0..width).for_each(|x| put(x, outside));
                    continue;
                }
                let start = (drawn.rect.left - area.left) as usize;
                let end = start + drawn.columns.len();
                (0..start).chain(end..width).for_each(|x| put(x, outside));
                let source_row = drawn.lines[(row - drawn.rect.top) as usize][0].0;
                let left = drawn.columns[0][0].0;
                lone.surface.read_row(lone.rows, left, source_row, &mut line);
                for (x, pixel) in (start..).zip(line.chunks_exact(BYTES_PER_PIXEL as usize)) {
                    put(x, encode(lone.color(pixel)));
                }
            }
        });
    } else {
        // Source samples read: each pixel of a slot takes its column's taps
        // times its line's, plus the pass that finishes every pixel.
        let taps = |all: &[Vec<(u32, f32)>]| all.iter().map(Vec::len).sum::<usize>();
        let work = width * height
            + (drawn.iter())
                .map(|drawn| taps(&drawn.columns) * taps(&drawn.lines))
                .sum::<usize>();
        in_bands(&mut colors, width, work, |first, band| {
            let mut degamma = Memo::new(srgb_to_linear);
            let mut regamma = Memo::new(clamped_linear_to_srgb);
            let mut line = vec![background; width];
            for (out_y, out_row) in (first..).zip(band.chunks_mut(width)) {
                let y = if flip_y { height - 1 - out_y } else { out_y };
                let row = area.top + y as u32;
                line.fill(background);
                for drawn in &drawn {
                    if !(drawn.rect.top..drawn.rect.bottom).contains(&row) {
                        continue;
                    }
                    let (slot, taps) = (drawn.slot, &drawn.lines[(row - drawn.rect.top) as usize]);
                    let start = (drawn.rect.left - area.left) as usize;
                    for (at, column) in line[start..].iter_mut().zip(&drawn.columns) {
                        let color =
                            slot.color(slot.sample(&drawn.rows, column, taps), &mut degamma);
                        *at = slot.blend.apply(color, *at);
                    }
                }
                for (x, &(mut color)) in line.iter().enumerate() {
                    if regamma_mode == GAMMA_SRGB {
                        for value in &mut color[..3] {
                            *value = regamma.get(*value);
                        }
                    }
                    out_row[if flip_x { width - 1 - x } else { x }] = encode(color);
                }
            }
        });
    }
    if !transpose {
        return output.write(memory, area, &colors);
    }
    let Output::Rgb(surface) = output else {
        return Err("transposed planar output".into());
    };
    let turned = surface.transposed(config.pair(0x20))?;
    let mut turned_colors = vec![[0.0; 4]; colors.len()];
    for (y, row) in colors.chunks(width).enumerate() {
        for (x, color) in row.iter().enumerate() {
            turned_colors[x * height + y] = *color;
        }
    }
    let turned_area = Rect {
        left: area.top,
        top: area.left,
        right: area.bottom,
        bottom: area.right,
    };
    Output::Rgb(turned).write(memory, turned_area, &turned_colors)
}

/// A slot's part of the picture: the rectangle it draws, the source
/// samples ([`taps`]) of each of its columns and lines, and the source
/// rows those read.
struct Drawn<'a> {
    slot: &'a Slot,
    rect: Rect,
    columns: Vec<Vec<(u32, f32)>>,
    lines: Vec<Vec<(u32, f32)>>,
    rows: SourceRows,
}

/// The composition of a lone slot that copies 8-bit RGB pixels 1:1, takes
/// no colour matrix and has a constant alpha: grc's video capture of the
/// game's frames (30 an second), nvnflinger's News and netConnect frames.
/// Then each composed and regamma'd colour component is a function of the
/// source pixel's byte for it alone: the slot's clamp and degamma act on
/// each component by itself, and the blending factors depend only on the
/// alphas, the source's constant and the background's. So each such
/// function is tabulated over the 256 codes, by running the slot's and the
/// output's own steps ([`Slot::color`], [`Blend::apply`], the regamma)
/// on each code: a pixel looked up gets the values those steps give it.
struct Fused<'a> {
    surface: &'a Surface,
    rows: &'a Rows,
    /// R, G, B after blending and regamma, by the code of each.
    tables: [[f32; 256]; 3],
    alpha: f32,
    /// The regamma'd background, around the slot.
    outside: [f32; 4],
}

impl<'a> Fused<'a> {
    fn new(drawn: &'a [Drawn], background: [f32; 4], regamma_mode: u32) -> Option<Fused<'a>> {
        let [drawn] = drawn else {
            return None;
        };
        let slot = drawn.slot;
        let (Source::Rgb(surface), SourceRows::Rgb(rows)) = (&slot.surface, &drawn.rows) else {
            return None;
        };
        // One tap of weight 1 for each column and line, on consecutive
        // source columns and lines.
        let copies = |taps: &[Vec<(u32, f32)>]| {
            let first = taps.first().map_or(0, |tap| tap[0].0);
            (taps.iter().zip(first..))
                .all(|(tap, next)| matches!(tap[..], [(at, weight)] if at == next && weight == 1.0))
        };
        let alpha_constant = slot.blend.overrides[3].is_some()
            || slot.constant_alpha
            || !surface.format.stored_alpha;
        if slot.matrix.is_some()
            || !alpha_constant
            || !copies(&drawn.columns)
            || !copies(&drawn.lines)
        {
            return None;
        }
        let regamma = |value: f32| {
            if regamma_mode == GAMMA_SRGB {
                clamped_linear_to_srgb(value)
            } else {
                value
            }
        };
        let mut degamma = Memo::new(srgb_to_linear);
        let mut tables = [[0.0; 256]; 3];
        let mut alpha = 0.0;
        for code in 0..=255u8 {
            // As `Surface::pixel` reads the code in each component.
            let value = f32::from(code) / 255.0;
            let stored_alpha = if surface.format.stored_alpha { value } else { 1.0 };
            let color = slot.color([value, value, value, stored_alpha], &mut degamma);
            let blended = slot.blend.apply(color, background);
            for (table, component) in tables.iter_mut().zip(blended) {
                table[usize::from(code)] = regamma(component);
            }
            alpha = blended[3];
        }
        let [r, g, b, a] = background;
        Some(Fused {
            surface,
            rows,
            tables,
            alpha,
            outside: [regamma(r), regamma(g), regamma(b), a],
        })
    }

    /// The composed, regamma'd colour of a source pixel's bytes.
    #[inline]
    fn color(&self, pixel: &[u8]) -> [f32; 4] {
        let [r, g, b] = self.surface.format.rgb;
        let [tr, tg, tb] = &self.tables;
        [
            tr[usize::from(pixel[r])],
            tg[usize::from(pixel[g])],
            tb[usize::from(pixel[b])],
            self.alpha,
        ]
    }
}

/// EXECUTEs that read fewer source samples than this (with a pass per
/// pixel counted as one) are composed on the thread that runs them; larger
/// ones are split among host threads.
const BANDED_WORK: usize = 1 << 16;
/// The most host threads one EXECUTE composes on.
const BAND_THREADS: usize = 8;

/// Runs `fill(first, band)` over `items`, rows of `width` items (the last
/// may be shorter), in bands of whole rows, `first` being the band's first
/// row: for `work` of [`BANDED_WORK`] or more on up to [`BAND_THREADS`]
/// host threads at once. Every row is computed the same way whichever band
/// it falls in, so the result does not depend on the split.
fn in_bands<T: Send>(
    items: &mut [T],
    width: usize,
    work: usize,
    fill: impl Fn(usize, &mut [T]) + Sync,
) {
    let rows = items.len().div_ceil(width);
    let threads = if work < BANDED_WORK {
        1
    } else {
        std::thread::available_parallelism().map_or(1, |n| n.get().min(BAND_THREADS))
    };
    if threads <= 1 {
        return fill(0, items);
    }
    let band = rows.div_ceil(threads);
    std::thread::scope(|scope| {
        for (index, items) in items.chunks_mut(band * width).enumerate() {
            let fill = &fill;
            scope.spawn(move || fill(index * band, items));
        }
    });
}

#[cfg(test)]
mod tests;
