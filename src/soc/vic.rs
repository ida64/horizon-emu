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
//! 0x610010 (`0xAFBD0`), SET_CONFIG_STRUCT_OFFSET 0x708 (`0xAFBF0`), method
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
//! 1:1 source-to-destination copies, the slot colour matrix (Ryujinx
//! `Blender.cs` formula, MIT), soft clamp, planar/constant alpha, the
//! blending factor selections (libdrm `vic.h` enum names), override and
//! mask, alpha fill modes opaque, background and composited (DXVA-HD names
//! in libdrm `vic.h`). Everything else (scaling and its filters, YUV
//! formats, deinterlacing and denoise, luma key, gamut and output matrices,
//! degamma/regamma, clear rectangles, decompression, flips and transposition,
//! downsampling) refuses with the field named.

use std::sync::{Arc, Mutex};

use super::{Notes, host1x::Syncpoints, mc::DeviceMemory};
use crate::maxwell::texture::layout::{BlockLinear, BlockLinearSurface, GOB_HEIGHT};

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
/// (libdrm `vic40.c`); nvnflinger also sets bit 4 (`0x610010`), libdrm
/// leaves it clear, and no output depends on it in either.
const CONTROL_SIZE_SHIFT: u32 = 16;
const CONTROL_KNOWN_LOW: [u32; 2] = [0, 0x10];

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

struct State {
    offset: u32,
    methods: [u32; method::COUNT],
}

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
        let layout = match kind {
            // libdrm `VIC_BLK_KIND_PITCH`; pitch = stride × bytes per pixel.
            0 => Layout::Pitch(u64::from(row_bytes)),
            // `VIC_BLK_KIND_GENERIC_16Bx2`: the GPU's GOB layout.
            1 => Layout::Block(
                BlockLinearSurface::new(
                    row_bytes,
                    luma_height,
                    BlockLinear {
                        block_height_log2,
                        block_depth_log2: 0,
                    },
                ),
                GOB_HEIGHT << block_height_log2,
            ),
            _ => return Err(format!("{name}: block kind {kind}")),
        };
        if address == 0 {
            return Err(format!("{name}: no plane address"));
        }
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

    fn put(&self, rows: &mut Rows, x: u32, y: u32, color: [f32; 4]) {
        let at = (self.offset(x, y) - rows.start) as usize;
        let byte = |value: f32| (value.clamp(0.0, 1.0) * 255.0).round() as u8;
        for (channel, index) in self.format.rgb.into_iter().enumerate() {
            rows.bytes[at + index] = byte(color[channel]);
        }
        // X formats store the alpha too (Ryujinx writes 35 as 31).
        rows.bytes[at + self.format.alpha] = byte(color[3]);
    }
}

/// Rows of a surface held in host memory.
struct Rows {
    start: u64,
    bytes: Vec<u8>,
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

/// Coefficients of a colour matrix (libdrm `MatrixStruct`) if enabled.
fn matrix(words: [u64; 4]) -> Option<([[i64; 4]; 3], u32)> {
    let [w0, w1, w2, w3] = words;
    flag(w1, 63).then(|| {
        let signed = |word: u64, row: u32| {
            let value = field(word, row * 20, 20) as i64;
            value - ((value & 0x80000) << 1)
        };
        let row = |r: u32| [signed(w0, r), signed(w1, r), signed(w2, r), signed(w3, r)];
        ([row(0), row(1), row(2)], field(w0, 60, 4))
    })
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
        let blend = Blend {
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

/// One enabled slot, ready to draw.
struct Slot {
    surface: Surface,
    source: Rect,
    destination: Rect,
    matrix: Option<([[i64; 4]; 3], u32)>,
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
        if !flag(w0, 8) || field(w0, 9, 7) != 0 {
            return unmodeled(&format!("field enables {:#x}", field(w0, 8, 8)));
        }
        if field(w0, 16, 4) != 0 {
            return unmodeled(&format!("frame format {}", field(w0, 16, 4)));
        }
        if field(w0, 24, 12) != 0 {
            return unmodeled(&format!("panoramic {:#x}", field(w0, 24, 12)));
        }
        // The w0 filter lengths and all of w1 (noise/detail filter
        // strengths, deinterlace mode) act only when scaling, denoising or
        // deinterlacing, all refused here.
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
        if field(w2, 55, 2) != 0 {
            return unmodeled(&format!("degamma mode {}", field(w2, 55, 2)));
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
        if source.width() != destination.width() || source.height() != destination.height() {
            return unmodeled(&format!("scaling {source:?} to {destination:?}"));
        }
        let plane = |p: usize| methods[method::SURFACES.start + index * 24 + p];
        if plane(1) != 0 || plane(2) != 0 {
            return unmodeled("chroma planes");
        }
        let surface = Surface::new(config.pair(base + 0x40), plane(0), &name)?;
        if source.right > surface.width || source.bottom > surface.height {
            return unmodeled(&format!(
                "source {source:?} outside surface {}x{}",
                surface.width, surface.height
            ));
        }
        if flag(config.word(base + 0x58), 40) {
            return unmodeled("luma key");
        }
        let matrix_at =
            |offset: usize| [0, 8, 16, 24].map(|word| config.word(base + offset + word));
        if matrix(matrix_at(0x80)).is_some() {
            return unmodeled("gamut matrix");
        }
        Ok(Some(Slot {
            surface,
            source,
            destination,
            matrix: matrix(matrix_at(0x60)),
            clamp,
            planar_alpha: field(w2, 32, 10) as f32 / 1023.0,
            constant_alpha: flag(w2, 42),
            blend: Blend::new(config.pair(base + 0xA0)).map_err(|e| format!("{name}: {e}"))?,
        }))
    }

    /// The slot's colour of a source pixel, before blending.
    fn color(&self, pixel: [f32; 4]) -> [f32; 4] {
        let mut color = pixel;
        if let Some((rows, shift)) = self.matrix {
            // Ryujinx `Blender.cs`: 10-bit components, arithmetic shifts.
            let ten = pixel.map(|value| (value * 1023.0).round() as i64);
            for (channel, row) in rows.iter().enumerate() {
                let sum = ten[0] * row[0] + ten[1] * row[1] + ten[2] * row[2];
                color[channel] = (((sum >> shift) + row[3]) >> 8) as f32 / 1023.0;
            }
        }
        for value in &mut color[..3] {
            *value = value.clamp(self.clamp.0, self.clamp.1);
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
    compose_config(memory, methods, &config)
        .map_err(|reason| format!("{reason}; {}", dump(&config)))
}

/// The non-zero words of `config`, for a refusal note.
fn dump(config: &Config) -> String {
    let words: Vec<String> = (config.0.iter().enumerate())
        .filter(|(_, word)| **word != 0)
        .map(|(i, word)| format!("{:03x}:{word:016x}", i * 8))
        .collect();
    format!("config {}", words.join(" "))
}

fn compose_config(
    memory: &DeviceMemory,
    methods: &[u32; method::COUNT],
    config: &Config,
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
    if field(output, 46, 2) != 0 {
        return Err(format!("regamma mode {}", field(output, 46, 2)));
    }
    if field(output, 48, 3) != 0 {
        return Err(format!("output flip/transpose {:#x}", field(output, 48, 3)));
    }
    if !matches!(alpha_fill, 0 | 1 | 4) {
        return Err(format!("alpha fill mode {alpha_fill}"));
    }
    let planes = &methods[method::SET_OUTPUT_SURFACE];
    if planes[1] != 0 || planes[2] != 0 {
        return Err("output chroma planes".into());
    }
    let surface = Surface::new(config.pair(0x20), planes[0], "output")?;
    if matrix([0x30, 0x38, 0x40, 0x48].map(|offset| config.word(offset))).is_some() {
        return Err("output colour matrix".into());
    }
    let area = Rect::inclusive(target).intersect(Rect {
        left: 0,
        top: 0,
        right: surface.width,
        bottom: surface.height,
    });
    let mut slots = Vec::new();
    for index in 0..SLOT_COUNT {
        slots.extend(Slot::new(config, methods, index)?);
    }
    if area.width() == 0 || area.height() == 0 {
        return Ok(());
    }
    let (width, height) = (area.width() as usize, area.height() as usize);
    let mut picture = vec![background; width * height];
    for slot in &slots {
        let drawn = slot.destination.intersect(area);
        if drawn.width() == 0 || drawn.height() == 0 {
            continue;
        }
        let top = slot.source.top + (drawn.top - slot.destination.top);
        let rows = slot.surface.read(memory, top, top + drawn.height())?;
        for y in drawn.top..drawn.bottom {
            let source_y = slot.source.top + (y - slot.destination.top);
            for x in drawn.left..drawn.right {
                let source_x = slot.source.left + (x - slot.destination.left);
                let color = slot.color(slot.surface.pixel(&rows, source_x, source_y));
                let at = (y - area.top) as usize * width + (x - area.left) as usize;
                picture[at] = slot.blend.apply(color, picture[at]);
            }
        }
    }
    let mut rows = surface.read(memory, area.top, area.bottom)?;
    for y in 0..height {
        for x in 0..width {
            let mut color = picture[y * width + x];
            color[3] = match alpha_fill {
                0 => 1.0,
                1 => background[3],
                _ => color[3],
            };
            surface.put(&mut rows, area.left + x as u32, area.top + y as u32, color);
        }
    }
    memory
        .write(surface.address + rows.start, &rows.bytes)
        .ok_or_else(|| {
            format!(
                "output at {:#x} is not mapped",
                surface.address + rows.start
            )
        })
}

#[cfg(test)]
mod tests;
