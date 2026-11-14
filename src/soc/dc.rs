//! Display controller: what it is programmed to show, the picture that
//! results, and the events it signals as frames go by.
//!
//! The chip has two, one per display head: DC0 drives the panel through
//! DSI, DC1 the external output, where nothing is attached. Both run the
//! same code in nvservices (head base `0x54200000 + head << 18`).
//!
//! # Registers
//!
//! The controller is a large block of registers, numbered by word. The
//! command registers at the start of the block take effect when written.
//! The others are double-buffered: software writes an assembly copy, and
//! the controller works from an active copy. The state control register
//! moves one into the other, per part: the general registers, and each
//! window's. Linux's driver for this controller (`tegra_dc_commit` and the
//! comment above it, public) describes the protocol: the `*_UPDATE` bits
//! arm the assembly copy, the `*_ACT_REQ` bits make the armed copy active,
//! at once while the controller is stopped and at the next frame boundary
//! while it sends frames. Linux, `boot` and nvservices (annex §3.4) all
//! write the update bits immediately before the request bits. Which
//! registers of this chip are double- and which triple-buffered is not
//! known here, so a request arms what it activates as well: a request
//! alone, as nvservices' HDMI path writes one, takes the assembly copy too.
//! The state access register lets software read and
//! write the active copy directly (nvservices writes 5 around such writes,
//! text `0x2a0a0`).
//!
//! The registers of the windows share their numbers: the window header
//! register selects which windows an access reaches. A register that was
//! never written refuses to be read.
//!
//! # Frames
//!
//! Frames pass at the panel's pace by the clock. The model brings itself up
//! to date whenever it is accessed, and [`keep_time`] does so at every frame
//! boundary for whoever waits on its interrupt. At a boundary the pending
//! activations take effect, syncpoint increments waiting for them happen,
//! the frame-end and vertical-blank interrupt status bits are set, and the
//! frame is counted on the syncpoint `CONT_SYNCPT_VSYNC` names.
//!
//! Nothing is scanned out continuously. [`Dc::frame`] produces the picture
//! the controller would be sending at that moment from the active copy:
//! the enabled windows' frame buffers, fetched through the SMMU like the
//! hardware fetches them, layered on the active area.

use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Duration,
};

use super::{
    host1x::{InterruptLine, Syncpoints},
    mc::DeviceMemory,
    regs::word,
};
use crate::bus::MmioDevice;

/// DC0, the panel's controller.
pub const BASE: u64 = 0x5420_0000;
/// DC1, the external output's.
pub const DCB_BASE: u64 = 0x5424_0000;
/// Up to the last window's underflow status, `0x3f28` (nvservices text
/// `0x246c0`).
pub const SIZE: u64 = 0x4000;

/// Time since the machine was built.
pub type Clock = Box<dyn Fn() -> Duration + Send + Sync>;

/// Register numbers; a register's offset is four times its number. Names
/// are those of Linux's `dc.h` (public).
mod register {
    /// The syncpoint increment registers of the general part and of
    /// windows A, B and C.
    pub const INCR_SYNCPT: [u64; 4] = [0x000, 0x008, 0x010, 0x018];
    pub const CONT_SYNCPT_VSYNC: u64 = 0x028;
    pub const DISPLAY_COMMAND: u64 = 0x032;
    pub const INT_STATUS: u64 = 0x037;
    pub const INT_MASK: u64 = 0x038;
    pub const INT_ENABLE: u64 = 0x039;
    pub const STATE_ACCESS: u64 = 0x040;
    pub const STATE_CONTROL: u64 = 0x041;
    pub const DISPLAY_WINDOW_HEADER: u64 = 0x042;
    /// The command registers end here; double-buffered ones follow.
    pub const COMMAND_END: u64 = 0x044;
    /// Every read nvservices makes of it discards the value: in its clock
    /// change sequence while the head's clock is reported on (text
    /// `0x22d20`, `0x22c4c`, `0x3b580` and others) and before it
    /// (`0x27790`).
    pub const DISP_TIMING_OPTIONS: u64 = 0x405;
    pub const DISP_ACTIVE: u64 = 0x409;
    pub const BLEND_BACKGROUND_COLOR: u64 = 0x4E4;
    /// Registers from here up to [`WINDOW_END`] exist once per window.
    pub const WINDOW_START: u64 = 0x500;
    pub const WINDOW_END: u64 = 0x900;
    pub const WIN_OPTIONS: u64 = 0x700;
    pub const WIN_BYTE_SWAP: u64 = 0x701;
    pub const WIN_COLOR_DEPTH: u64 = 0x703;
    pub const WIN_POSITION: u64 = 0x704;
    pub const WIN_SIZE: u64 = 0x705;
    pub const WIN_PRESCALED_SIZE: u64 = 0x706;
    pub const WIN_H_INITIAL_DDA: u64 = 0x707;
    pub const WIN_V_INITIAL_DDA: u64 = 0x708;
    pub const WIN_DDA_INC: u64 = 0x709;
    pub const WIN_LINE_STRIDE: u64 = 0x70A;
    pub const WIN_BLEND_LAYER_CONTROL: u64 = 0x716;
    pub const WIN_BLEND_MATCH_SELECT: u64 = 0x717;
    pub const WINBUF_START_ADDR: u64 = 0x800;
    pub const WINBUF_ADDR_H_OFFSET: u64 = 0x806;
    pub const WINBUF_ADDR_V_OFFSET: u64 = 0x808;
    pub const WINBUF_SURFACE_KIND: u64 = 0x80B;
}

/// Window header: one select bit per window, A first. There are three
/// windows, A to C, as Linux's `dc.h` names them and nvservices programs
/// them (annex §3.4).
const WINDOW_SELECT_SHIFT: u32 = 4;
const WINDOW_COUNT: usize = 3;

/// Parts of the double-buffered registers: the general ones, then one per
/// window. In the state control register, part `p`'s activation request is
/// bit `p` and its update bit `8 + p`.
const PART_COUNT: usize = 1 + WINDOW_COUNT;
const STATE_CONTROL_ACT_REQ: u32 = (1 << PART_COUNT) - 1;
const STATE_CONTROL_UPDATE: u32 = STATE_CONTROL_ACT_REQ << 8;

/// State access: reads, and writes, reach the active copy.
const READ_MUX: u32 = 1 << 0;
const WRITE_MUX: u32 = 1 << 2;

/// Display command: the controller sends frames continuously.
const DISPLAY_MODE_CONTINUOUS: u32 = 1 << 5;
const WIN_ENABLE: u32 = 1 << 30;
/// Window options (Linux `dc.h`) a window can be shown with: scanned right
/// to left, bottom up, and small pixels expanded to 32 bits.
const H_DIRECTION: u32 = 1 << 0;
const V_DIRECTION: u32 = 1 << 2;
const COLOR_EXPAND: u32 = 1 << 6;
/// Scans the surface by columns: line `n` of the window is column `n` of
/// the surface, its pixels taken down the surface's lines. The direction
/// bits keep their meaning in the surface's own axes, which is how
/// nvservices programs the offsets for them (text `0x279d0`: with
/// horizontal reversal the horizontal offset is the last byte of a
/// surface row, `0x13ff` for 1280 pixels). Not in Linux's `dc.h`;
/// nvservices sets it for a flip with its swap flag (text `0x28090`),
/// which nvnflinger sends for a 90° transform (its text `0x6f260`).
///
/// That the bit transposes, rather than turns, is established by the first
/// guest frame shown through it: omm's start-up logo, flipped by
/// nvnflinger as `WIN_OPTIONS` `0xc0000511` (scan columns, horizontal
/// reversal) over a 1280x720 surface stored top row first, is upright on
/// the console only if the bit transposes; turning it as well showed the
/// logo upside down. [INFERENCE from that frame; no documentation at hand
/// describes the bit.]
const SCAN_COLUMN: u32 = 1 << 4;
/// Filtering between source pixels when a window is scaled, horizontally
/// and vertically (Linux `dc.h`). nvservices sets bit 31 exactly together
/// with the horizontal one (text `0x28b90`: `0x80000100` when the source
/// and destination widths differ, `0x400` for the heights), so it belongs
/// to horizontal filtering [INFERENCE: a mode bit of that filter; not named
/// in public sources at hand].
const H_FILTER: u32 = 1 << 8;
const V_FILTER: u32 = 1 << 10;
const H_FILTER_MODE: u32 = 1 << 31;
/// One source pixel in the digital differential analyzer's 4.12 fixed
/// point: the identity step nvservices writes for unscaled windows
/// (`0x10001000`, text `0x28090`).
const DDA_ONE: u32 = 0x1000;
/// Byte order within each 32-bit pixel (Linux `dc.h`): bytes swapped in
/// pairs, or all four reversed. nvservices programs the latter for two of
/// its 32-bit formats (text `0x286e0`).
const BYTE_SWAP_SWAP2: u32 = 1;
const BYTE_SWAP_SWAP4: u32 = 2;

/// Frame counting: enabled, and on which syncpoint. `boot` programs 0x109,
/// syncpoint 9; nvservices `9 | 0x100` (text `0x23b74`).
const VSYNC_SYNCPOINT_ENABLE: u32 = 1 << 8;
const SYNCPOINT_INDEX: u32 = 0xFF;

/// Syncpoint increment conditions, bits 15:8 of an increment register
/// (names recalled from host1x documentation). nvservices writes condition
/// 3 for its flips (text `0x28018`): the increment happens once the
/// register writes before it are safe, which is when the activation they
/// were committed with has taken effect.
const CONDITION_IMMEDIATE: u32 = 0;
const CONDITION_REG_WR_SAFE: u32 = 3;

/// Interrupt status bits nvservices services (text `0x246c0`); Linux names
/// them the same.
const FRAME_END_INT: u32 = 1 << 1;
const VBLANK_INT: u32 = 1 << 2;

/// Frames per second the panel is driven at.
const REFRESH_RATE: u128 = 60;
const NANOS_PER_SECOND: u128 = 1_000_000_000;

/// Color depths (Linux `dc.h`): 32 bits per pixel with the named component
/// in the lowest byte.
const COLOR_DEPTH_B8G8R8A8: u32 = 12;
const COLOR_DEPTH_R8G8B8A8: u32 = 13;

/// Surface kinds (Linux `dc.h`): pitch-linear, or the GPU's block-linear
/// layout with the block height in GOBs, as a power of two, in bits 6:4.
const SURFACE_KIND_MASK: u32 = 3;
const SURFACE_KIND_PITCH: u32 = 0;
const SURFACE_KIND_BLOCK: u32 = 2;
const SURFACE_KIND_BLOCK_HEIGHT_SHIFT: u32 = 4;
const SURFACE_KIND_BLOCK_HEIGHT: u32 = 7;

/// A GOB, the unit of the block-linear layout: 64 bytes across, 8 lines
/// down, 512 bytes in memory.
const GOB_WIDTH: u64 = 64;
const GOB_HEIGHT: u64 = 8;
const GOB_SIZE: u64 = GOB_WIDTH * GOB_HEIGHT;

/// Blending of a window over what lies below (Linux `dc.h`): the window
/// layer depth in bits 7:0, larger further back (Linux writes `255 -
/// zpos`); the constants K1 in bits 15:8 and K2 in bits 23:16; and the
/// bypass of blending in bit 24.
const BLEND_DEPTH: u32 = 0xFF;
const BLEND_BYPASS: u32 = 1 << 24;

/// Largest active area [`Dc::frame`] renders; the panel's is 720x1280.
const MAX_DIMENSION: u32 = 4096;

/// A picture, eight bits each of red, green and blue per pixel, rows from
/// the top.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
}

impl Frame {
    /// The picture turned a quarter turn clockwise.
    pub fn rotated_clockwise(&self) -> Self {
        let (width, height) = (self.width as usize, self.height as usize);
        let mut rgb = Vec::with_capacity(self.rgb.len());
        // Each row of the result is a column of the source, read upward.
        for column in 0..width {
            for row in (0..height).rev() {
                let pixel = (row * width + column) * 3;
                rgb.extend_from_slice(&self.rgb[pixel..pixel + 3]);
            }
        }
        Self {
            width: self.height,
            height: self.width,
            rgb,
        }
    }
}

/// Why there is no picture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NoFrame {
    /// The controller has not been set up to send frames.
    NotRunning,
    /// It is set up in a way [`Dc::frame`] does not render.
    Unsupported(String),
    /// The window's frame buffer is not memory the controller can reach.
    BufferUnmapped { address: u64 },
}

impl std::fmt::Display for NoFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotRunning => write!(f, "the display controller is not sending frames"),
            Self::Unsupported(what) => write!(
                f,
                "the display controller is set up with {what}, which is not rendered"
            ),
            Self::BufferUnmapped { address } => {
                write!(
                    f,
                    "the frame buffer at device address {address:#x} is not mapped to memory"
                )
            }
        }
    }
}

/// One copy of the double-buffered registers.
#[derive(Default, Clone)]
struct Copy {
    general: HashMap<u64, u32>,
    windows: [HashMap<u64, u32>; WINDOW_COUNT],
}

impl Copy {
    fn part(&self, part: usize) -> &HashMap<u64, u32> {
        match part {
            0 => &self.general,
            window => &self.windows[window - 1],
        }
    }

    fn part_mut(&mut self, part: usize) -> &mut HashMap<u64, u32> {
        match part {
            0 => &mut self.general,
            window => &mut self.windows[window - 1],
        }
    }

    fn general(&self, number: u64) -> u32 {
        self.general.get(&number).copied().unwrap_or(0)
    }
}

#[derive(Default)]
struct Registers {
    /// The command registers, which have one copy.
    command: HashMap<u64, u32>,
    assembly: Copy,
    armed: Copy,
    active: Copy,
    /// Parts whose armed copy becomes active at the next frame boundary,
    /// one bit each as in the state control register.
    pending: u32,
    /// Syncpoints to increment once the pending activations have taken
    /// effect.
    waiting_increments: Vec<usize>,
    /// Interrupt status, bits set by events and cleared by writing ones.
    status: u32,
    /// Level last driven onto the interrupt line.
    asserted: bool,
    /// Since when, and how many frames have been sent, while the
    /// controller sends frames.
    running: Option<Running>,
}

struct Running {
    since: Duration,
    frames: u128,
}

impl Registers {
    fn command(&self, number: u64) -> u32 {
        self.command.get(&number).copied().unwrap_or(0)
    }

    /// Windows the header register selects.
    fn selected_windows(&self) -> impl Iterator<Item = usize> {
        let header = self.command(register::DISPLAY_WINDOW_HEADER);
        (0..WINDOW_COUNT).filter(move |window| header >> WINDOW_SELECT_SHIFT >> window & 1 != 0)
    }

    /// Makes the armed copy of the parts in `parts` active.
    fn activate(&mut self, parts: u32) {
        for part in (0..PART_COUNT).filter(|part| parts >> part & 1 != 0) {
            *self.active.part_mut(part) = self.armed.part(part).clone();
        }
    }

    fn sending(&self) -> bool {
        self.active.general(register::DISPLAY_COMMAND) & DISPLAY_MODE_CONTINUOUS != 0
    }

    /// Whether `status` has a bit both enabled and unmasked.
    fn interrupt(&self) -> bool {
        self.status & self.command(register::INT_ENABLE) & self.command(register::INT_MASK) != 0
    }
}

pub struct Dc {
    registers: Mutex<Registers>,
    syncpoints: Arc<Syncpoints>,
    clock: Clock,
    /// INTID 105 for DC0, 106 for DC1 (nvservices text `0x23634`).
    line: InterruptLine,
}

impl Dc {
    /// `syncpoints` is where frames and updates are counted, `clock` what
    /// paces frames, `line` the controller's interrupt.
    pub fn new(syncpoints: Arc<Syncpoints>, clock: Clock, line: InterruptLine) -> Self {
        Self {
            registers: Mutex::default(),
            syncpoints,
            clock,
            line,
        }
    }

    /// Brings the controller up to date with the frames sent since the last
    /// look.
    pub fn count_frames(&self) {
        self.settle(&mut self.registers.lock().unwrap());
    }

    /// Puts the controller in its reset state, as holding it in reset does.
    /// nvservices pulses each head's reset before it programs the head
    /// (text `0x22a94`, `0x22b0c`), and `boot` leaves the panel's
    /// controller programmed. The reset values of most registers are not
    /// known here: they hold nothing again, refusing to be read until
    /// written. The controller stops, nothing is pending, the interrupt
    /// status is clear and the line falls.
    pub fn reset(&self) {
        let mut registers = self.registers.lock().unwrap();
        *registers = Registers {
            asserted: registers.asserted,
            ..Registers::default()
        };
        self.settle(&mut registers);
    }

    /// Time until the next frame boundary, if the controller sends frames.
    fn next_frame(&self) -> Option<Duration> {
        let registers = self.registers.lock().unwrap();
        let running = registers.running.as_ref()?;
        let boundary = (running.frames + 1) * NANOS_PER_SECOND / REFRESH_RATE;
        let elapsed = (self.clock)().saturating_sub(running.since).as_nanos();
        Some(Duration::from_nanos(
            boundary
                .saturating_sub(elapsed)
                .try_into()
                .unwrap_or(u64::MAX),
        ))
    }

    /// Handles the frame boundaries passed since the last look, then starts
    /// or stops sending as the active copy now says, and drives the
    /// interrupt line.
    fn settle(&self, registers: &mut Registers) {
        let now = (self.clock)();
        if let Some(running) = &mut registers.running {
            let frames =
                now.saturating_sub(running.since).as_nanos() * REFRESH_RATE / NANOS_PER_SECOND;
            let passed = frames - running.frames;
            running.frames = frames;
            if passed > 0 {
                // The first boundary activates what is pending, which may
                // stop the controller; frames after that were only sent if
                // it still runs.
                let pending = std::mem::take(&mut registers.pending);
                registers.activate(pending);
                self.release_increments(registers);
                registers.status |= FRAME_END_INT | VBLANK_INT;
                let sent = if registers.sending() { passed } else { 1 };
                let vsync = registers.command(register::CONT_SYNCPT_VSYNC);
                if vsync & VSYNC_SYNCPOINT_ENABLE != 0 {
                    self.syncpoints.add(
                        (vsync & SYNCPOINT_INDEX) as usize,
                        u32::try_from(sent).unwrap_or(u32::MAX),
                    );
                }
            }
        }
        match (registers.sending(), &registers.running) {
            (true, None) => {
                registers.running = Some(Running {
                    since: now,
                    frames: 0,
                })
            }
            (false, Some(_)) => registers.running = None,
            _ => {}
        }
        let level = registers.interrupt();
        if level != registers.asserted {
            registers.asserted = level;
            (self.line)(level);
        }
    }

    /// Write of the state control register.
    fn state_control(&self, registers: &mut Registers, value: u32) -> Option<()> {
        (value & !(STATE_CONTROL_UPDATE | STATE_CONTROL_ACT_REQ) == 0).then_some(())?;
        let requests = value & STATE_CONTROL_ACT_REQ;
        let arm = (value & STATE_CONTROL_UPDATE) >> 8 | requests;
        for part in (0..PART_COUNT).filter(|part| arm >> part & 1 != 0) {
            *registers.armed.part_mut(part) = registers.assembly.part(part).clone();
        }
        if registers.running.is_some() {
            registers.pending |= requests;
        } else {
            // Stopped, nothing is ever pending.
            registers.activate(requests);
            self.release_increments(registers);
        }
        Some(())
    }

    fn release_increments(&self, registers: &mut Registers) {
        for syncpoint in std::mem::take(&mut registers.waiting_increments) {
            self.syncpoints.add(syncpoint, 1);
        }
    }

    /// Write of a syncpoint increment register.
    fn increment(&self, registers: &mut Registers, value: u32) -> Option<()> {
        let syncpoint = (value & SYNCPOINT_INDEX) as usize;
        match value >> 8 {
            CONDITION_IMMEDIATE => self.syncpoints.add(syncpoint, 1),
            CONDITION_REG_WR_SAFE if registers.pending == 0 => self.syncpoints.add(syncpoint, 1),
            CONDITION_REG_WR_SAFE => registers.waiting_increments.push(syncpoint),
            _ => return None,
        }
        Some(())
    }

    /// The picture being sent to the panel, with the frame buffers read
    /// through `memory`.
    pub fn frame(&self, memory: &DeviceMemory) -> Result<Frame, NoFrame> {
        let mut registers = self.registers.lock().unwrap();
        self.settle(&mut registers);
        if !registers.sending() {
            return Err(NoFrame::NotRunning);
        }
        let active = &registers.active;
        let halves = |value: u32| (value & 0xFFFF, value >> 16);
        let (width, height) = halves(active.general(register::DISP_ACTIVE));
        if width == 0 || height == 0 || width > MAX_DIMENSION || height > MAX_DIMENSION {
            return Err(NoFrame::Unsupported(format!(
                "an active area of {width}x{height}"
            )));
        }
        if active.general(register::BLEND_BACKGROUND_COLOR) != 0 {
            return Err(NoFrame::Unsupported("a background color".into()));
        }
        // Back to front.
        let mut windows: Vec<Window> = active
            .windows
            .iter()
            .filter(|window| {
                window
                    .get(&register::WIN_OPTIONS)
                    .is_some_and(|options| options & WIN_ENABLE != 0)
            })
            .map(|registers| Window { registers })
            .collect();
        if windows.len() > 1 {
            if windows.iter().any(|window| window.blending().is_none()) {
                return Err(NoFrame::Unsupported(
                    "several windows without blending set up".into(),
                ));
            }
            windows.sort_by_key(|window| {
                std::cmp::Reverse(window.blending().map(|control| control & BLEND_DEPTH))
            });
        }
        // The background is black, and opaque.
        let mut picture = vec![[0.0f32; 3]; (width * height) as usize];
        for window in windows {
            window.draw(memory, width, height, &mut picture)?;
        }
        let rgb = picture
            .iter()
            .flat_map(|pixel| {
                pixel.map(|channel| (channel * 255.0).round().clamp(0.0, 255.0) as u8)
            })
            .collect();
        Ok(Frame { width, height, rgb })
    }
}

/// An enabled window's active registers.
struct Window<'a> {
    registers: &'a HashMap<u64, u32>,
}

impl Window<'_> {
    fn value(&self, number: u64) -> u32 {
        self.registers.get(&number).copied().unwrap_or(0)
    }

    fn blending(&self) -> Option<u32> {
        self.registers
            .get(&register::WIN_BLEND_LAYER_CONTROL)
            .copied()
    }

    /// Fetches the window's pixels, red, green, blue and alpha each.
    fn fetch(&self, memory: &DeviceMemory) -> Result<(u32, u32, Vec<[u8; 4]>), NoFrame> {
        let halves = |value: u32| (value & 0xFFFF, value >> 16);
        let red_first = match self.value(register::WIN_COLOR_DEPTH) {
            COLOR_DEPTH_B8G8R8A8 => false,
            COLOR_DEPTH_R8G8B8A8 => true,
            depth => return Err(NoFrame::Unsupported(format!("color depth {depth}"))),
        };
        let byte_swap = self.value(register::WIN_BYTE_SWAP);
        if byte_swap > BYTE_SWAP_SWAP4 {
            return Err(NoFrame::Unsupported(format!("byte swap {byte_swap}")));
        }
        let options = self.value(register::WIN_OPTIONS);
        // Color expansion only concerns pixels under 32 bits.
        let unmodeled = options
            & !(WIN_ENABLE
                | H_DIRECTION
                | V_DIRECTION
                | SCAN_COLUMN
                | COLOR_EXPAND
                | H_FILTER
                | H_FILTER_MODE
                | V_FILTER);
        if unmodeled != 0 {
            return Err(NoFrame::Unsupported(format!(
                "window options {unmodeled:#x}"
            )));
        }
        let (width, height) = halves(self.value(register::WIN_SIZE));
        // The source size counts bytes across and lines down, as the window
        // shows them.
        let (source_bytes, source_lines) = halves(self.value(register::WIN_PRESCALED_SIZE));
        if !source_bytes.is_multiple_of(4) || source_bytes == 0 || source_lines == 0 {
            return Err(NoFrame::Unsupported(format!(
                "a source of {source_bytes} bytes by {source_lines} lines"
            )));
        }
        let (across, down) = (source_bytes / 4, source_lines);
        let column = options & SCAN_COLUMN != 0;
        // The rectangle of the surface the window shows, in the surface's
        // own orientation.
        let (source_width, source_height) = if column {
            (down, across)
        } else {
            (across, down)
        };
        let stride = u64::from(self.value(register::WIN_LINE_STRIDE) & 0xFFFF);
        let start = u64::from(self.value(register::WINBUF_START_ADDR));
        let row_bytes = u64::from(source_width) * 4;
        // The horizontal offset counts bytes, the vertical one lines; a
        // window scanned right to left or bottom up starts from its last
        // byte or line (nvservices text 0x279d0, Linux likewise).
        let mut left = u64::from(self.value(register::WINBUF_ADDR_H_OFFSET));
        let mut top = u64::from(self.value(register::WINBUF_ADDR_V_OFFSET));
        if options & H_DIRECTION != 0 {
            left = (left + 1)
                .checked_sub(row_bytes)
                .ok_or_else(|| NoFrame::Unsupported("a window left of its surface".into()))?;
        }
        if options & V_DIRECTION != 0 {
            top = (top + 1)
                .checked_sub(source_height.into())
                .ok_or_else(|| NoFrame::Unsupported("a window above its surface".into()))?;
        }
        let kind = self.value(register::WINBUF_SURFACE_KIND);
        let mut bytes = vec![0; (row_bytes * u64::from(source_height)) as usize];
        match kind & SURFACE_KIND_MASK {
            SURFACE_KIND_PITCH => {
                for (row, line) in bytes.chunks_mut(row_bytes as usize).enumerate() {
                    let address = start + (top + row as u64) * stride + left;
                    memory
                        .read(address, line)
                        .ok_or(NoFrame::BufferUnmapped { address })?;
                }
            }
            SURFACE_KIND_BLOCK => {
                let block_height =
                    1 << (kind >> SURFACE_KIND_BLOCK_HEIGHT_SHIFT & SURFACE_KIND_BLOCK_HEIGHT);
                let layout = BlockLinear {
                    start,
                    stride,
                    block_height,
                };
                layout.read(memory, left, top, row_bytes, &mut bytes)?;
            }
            _ => return Err(NoFrame::Unsupported(format!("surface kind {kind:#x}"))),
        }
        let mut pixels: Vec<[u8; 4]> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&word| {
                let [first, green, third, alpha] = match byte_swap {
                    BYTE_SWAP_SWAP2 => [word[1], word[0], word[3], word[2]],
                    BYTE_SWAP_SWAP4 => [word[3], word[2], word[1], word[0]],
                    _ => word,
                };
                if red_first {
                    [first, green, third, alpha]
                } else {
                    [third, green, first, alpha]
                }
            })
            .collect();
        if options & H_DIRECTION != 0 {
            pixels
                .chunks_mut(source_width as usize)
                .for_each(|row| row.reverse());
        }
        if options & V_DIRECTION != 0 {
            let rows: Vec<&[[u8; 4]]> = pixels.chunks(source_width as usize).rev().collect();
            pixels = rows.concat();
        }
        if column {
            // Each line of the window is a column of the surface, taken from
            // the top: the surface transposed.
            let (across, down) = (source_width as usize, source_height as usize);
            pixels = (0..across)
                .flat_map(|column| (0..down).map(move |row| (row, column)))
                .map(|(row, column)| pixels[row * across + column])
                .collect();
        }
        let scaling = Scaling {
            increments: self.value(register::WIN_DDA_INC),
            initial: [
                self.value(register::WIN_H_INITIAL_DDA),
                self.value(register::WIN_V_INITIAL_DDA),
            ],
            filters: [options & H_FILTER != 0, options & V_FILTER != 0],
        };
        let pixels = scaling.apply(&pixels, [across, down], [width, height]);
        Ok((width, height, pixels))
    }

    /// Lays the window over `picture`, the active area so far.
    fn draw(
        &self,
        memory: &DeviceMemory,
        area_width: u32,
        area_height: u32,
        picture: &mut [[f32; 3]],
    ) -> Result<(), NoFrame> {
        let (left, top) = {
            let position = self.value(register::WIN_POSITION);
            (position & 0xFFFF, position >> 16)
        };
        let (width, height, pixels) = self.fetch(memory)?;
        if left + width > area_width || top + height > area_height {
            return Err(NoFrame::Unsupported(
                "a window reaching outside the active area".into(),
            ));
        }
        let blend = Blend::of(self)?;
        for row in 0..height {
            let source = &pixels[(row * width) as usize..][..width as usize];
            let offset = ((top + row) * area_width + left) as usize;
            for (below, &pixel) in picture[offset..].iter_mut().zip(source) {
                *below = blend.apply(pixel, *below);
            }
        }
        Ok(())
    }
}

/// How a window's source, as the window scans it, becomes its size on the
/// screen. The digital differential analyzer steps through the source by
/// a fixed-point increment per destination pixel, horizontally in the low
/// half of `DDA_INC` and vertically in the high half, from the initial
/// positions (nvservices text `0x28090`: increments `src/dst`, or `(src −
/// 1)/(dst − 1)` when shrinking to half or less, in 4.12; initial
/// positions the fractional source origin).
///
/// With a direction's filter off, each destination pixel takes the source
/// pixel the analyzer is on. With it on, the hardware weighs neighboring
/// source pixels with multi-tap kernels whose coefficients nvservices loads
/// per window (annex §2.1, `0x1804`-`0x18e0`); that is modeled as linear
/// interpolation between the two nearest source pixels, an assumption. An
/// unscaled window comes out pixel for pixel either way.
struct Scaling {
    increments: u32,
    initial: [u32; 2],
    filters: [bool; 2],
}

impl Scaling {
    /// Scales `pixels`, `source` wide and high, to `destination`.
    fn apply(&self, pixels: &[[u8; 4]], source: [u32; 2], destination: [u32; 2]) -> Vec<[u8; 4]> {
        let steps = [self.increments & 0xFFFF, self.increments >> 16];
        if source == destination && steps == [DDA_ONE; 2] && self.initial == [0; 2] {
            return pixels.to_vec();
        }
        // Per direction and destination position: the two source pixels
        // and the weight of the second.
        let taps = |axis: usize| -> Vec<(usize, usize, f32)> {
            let last = source[axis].saturating_sub(1) as usize;
            (0..u64::from(destination[axis]))
                .map(|position| {
                    let at = u64::from(self.initial[axis]) + position * u64::from(steps[axis]);
                    let whole = ((at >> 12) as usize).min(last);
                    let weight = if self.filters[axis] && whole < last {
                        (at & 0xFFF) as f32 / 4096.0
                    } else {
                        0.0
                    };
                    (whole, (whole + 1).min(last), weight)
                })
                .collect()
        };
        let (columns, rows) = (taps(0), taps(1));
        let width = source[0] as usize;
        let mix = |a: [u8; 4], b: [u8; 4], weight: f32| -> [f32; 4] {
            std::array::from_fn(|channel| {
                f32::from(a[channel]) * (1.0 - weight) + f32::from(b[channel]) * weight
            })
        };
        rows.iter()
            .flat_map(|&(top, bottom, down)| {
                columns.iter().map(move |&(left, right, across)| {
                    let at = |row: usize, column: usize| pixels[row * width + column];
                    let upper = mix(at(top, left), at(top, right), across);
                    let lower = mix(at(bottom, left), at(bottom, right), across);
                    std::array::from_fn(|channel| {
                        (upper[channel] * (1.0 - down) + lower[channel] * down).round() as u8
                    })
                })
            })
            .collect()
    }
}

/// How a window's pixels combine with what lies below.
enum Blend {
    /// The window replaces what is below.
    Opaque,
    /// `result = source * source_factor + below * below_factor`, the
    /// factors selected by the window's blend selection register from its
    /// pixels' alpha and the constants K1 and K2. The background is opaque
    /// and everything is blended onto it, so the alpha below is one.
    Factors { select: u32, k1: f32, k2: f32 },
}

impl Blend {
    fn of(window: &Window) -> Result<Self, NoFrame> {
        let Some(control) = window.blending() else {
            return Ok(Self::Opaque);
        };
        if control & BLEND_BYPASS != 0 {
            return Ok(Self::Opaque);
        }
        // nvservices (text 0x26630) writes the same selection to both for
        // straight alpha (0x265: source times K1·alpha over below times one
        // minus that), and for premultiplied alpha 0x262 to the match
        // register but 0x224 to the other, which mirrors it: the window's
        // color weighted by what is not covered by the other, the other's
        // by K1. So the match selection is the one for a window lying over
        // the other, as every window here does over the layers behind it.
        // [INFERENCE: register roles not documented anywhere at hand.]
        let select = window.value(register::WIN_BLEND_MATCH_SELECT);
        if select & 7 > 5 {
            return Err(NoFrame::Unsupported(format!("blend selection {select:#x}")));
        }
        let constant = |shift: u32| (control >> shift & 0xFF) as f32 / 255.0;
        Ok(Self::Factors {
            select,
            k1: constant(8),
            k2: constant(16),
        })
    }

    fn apply(&self, [red, green, blue, alpha]: [u8; 4], below: [f32; 3]) -> [f32; 3] {
        let source = [red, green, blue].map(|channel| f32::from(channel) / 255.0);
        let Self::Factors { select, k1, k2 } = *self else {
            return source;
        };
        let alpha = f32::from(alpha) / 255.0;
        // Linux's `BLEND_FACTOR_*_COLOR_*` encodings, where "DST" is the
        // alpha below, one, and "SRC" the window's.
        let source_factor = match select & 7 {
            0 => 0.0,
            1 => 1.0,
            2 | 3 => k1,
            4 => 1.0 - k1,
            _ => k1 * alpha,
        };
        let below_factor = match select >> 4 & 7 {
            0 => 0.0,
            1 => 1.0,
            2 | 4 => k1,
            3 => k2,
            5 | 7 => 1.0 - k1,
            _ => 1.0 - k1 * alpha,
        };
        std::array::from_fn(|channel| {
            source[channel] * source_factor + below[channel] * below_factor
        })
    }
}

/// Where the GPU's block-linear layout puts the bytes of a surface: rows of
/// blocks, each block one GOB wide and `block_height` GOBs high, the GOBs
/// of a block one after the other, the blocks of a row one after the other.
/// Inside a GOB, 16-byte pieces of two lines are interleaved. This is the
/// layout open-source emulators of this GPU generation decode (yuzu's and
/// Ryujinx's swizzle tables), recalled from them, not checked against the
/// hardware; nvservices hands the controller such surfaces with the block
/// height in GOBs as a power of two (text `0x28620`).
struct BlockLinear {
    start: u64,
    /// Bytes across the surface, a whole number of GOBs.
    stride: u64,
    block_height: u64,
}

impl BlockLinear {
    /// Offset of the byte `x` bytes across and `y` lines down.
    fn offset(&self, x: u64, y: u64) -> u64 {
        let block_lines = GOB_HEIGHT * self.block_height;
        let block_size = GOB_SIZE * self.block_height;
        let blocks_across = self.stride / GOB_WIDTH;
        let in_gob =
            (x % 64 / 32) * 256 + (y % 8 / 2) * 64 + (x % 32 / 16) * 32 + (y % 2) * 16 + x % 16;
        (y / block_lines) * blocks_across * block_size
            + (x / GOB_WIDTH) * block_size
            + (y % block_lines / GOB_HEIGHT) * GOB_SIZE
            + in_gob
    }

    /// Reads `row_bytes` across of each line from (`left`, `top`) on into
    /// `bytes`, a block row at a time.
    fn read(
        &self,
        memory: &DeviceMemory,
        left: u64,
        top: u64,
        row_bytes: u64,
        bytes: &mut [u8],
    ) -> Result<(), NoFrame> {
        if !self.stride.is_multiple_of(GOB_WIDTH) || left + row_bytes > self.stride {
            return Err(NoFrame::Unsupported(format!(
                "a block-linear surface {} bytes across",
                self.stride
            )));
        }
        let block_lines = GOB_HEIGHT * self.block_height;
        let block_row_size = (self.stride / GOB_WIDTH) * GOB_SIZE * self.block_height;
        let lines = bytes.len() as u64 / row_bytes;
        let mut block_row = vec![0; block_row_size as usize];
        let mut loaded = None;
        for line in 0..lines {
            let y = top + line;
            let block = y / block_lines;
            if loaded != Some(block) {
                let address = self.start + block * block_row_size;
                memory
                    .read(address, &mut block_row)
                    .ok_or(NoFrame::BufferUnmapped { address })?;
                loaded = Some(block);
            }
            let base = block * block_row_size;
            let out = &mut bytes[(line * row_bytes) as usize..][..row_bytes as usize];
            // Sixteen bytes at a time stay together.
            for (piece, chunk) in out.chunks_mut(16).enumerate() {
                let x = left + piece as u64 * 16;
                let at = (self.offset(x, y) - base) as usize;
                chunk.copy_from_slice(&block_row[at..at + chunk.len()]);
            }
        }
        Ok(())
    }
}

impl MmioDevice for Dc {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        let number = word(offset, size)? / 4;
        let mut registers = self.registers.lock().unwrap();
        self.settle(&mut registers);
        if number == register::INT_STATUS {
            return Some(registers.status.into());
        }
        if number < register::COMMAND_END && number != register::DISPLAY_COMMAND {
            return registers.command.get(&number).map(|&value| value.into());
        }
        let copy = if registers.command(register::STATE_ACCESS) & READ_MUX != 0 {
            &registers.active
        } else {
            &registers.assembly
        };
        let value = if (register::WINDOW_START..register::WINDOW_END).contains(&number) {
            let window = registers.selected_windows().next()?;
            copy.windows[window].get(&number)
        } else {
            copy.general.get(&number).or_else(|| {
                // Not known before it is written; nobody can tell.
                (number == register::DISP_TIMING_OPTIONS).then_some(&0)
            })
        };
        value.map(|&value| value.into())
    }

    fn write(&self, _core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        let number = word(offset, size)? / 4;
        let value = value as u32;
        let mut registers = self.registers.lock().unwrap();
        self.settle(&mut registers);
        match number {
            register::INT_STATUS => registers.status &= !value,
            register::STATE_CONTROL => self.state_control(&mut registers, value)?,
            number if register::INCR_SYNCPT.contains(&number) => {
                self.increment(&mut registers, value)?
            }
            number if number < register::COMMAND_END && number != register::DISPLAY_COMMAND => {
                registers.command.insert(number, value);
            }
            number => {
                let direct = registers.command(register::STATE_ACCESS) & WRITE_MUX != 0;
                let selected: Vec<usize> = registers.selected_windows().collect();
                let copy = if direct {
                    &mut registers.active
                } else {
                    &mut registers.assembly
                };
                if (register::WINDOW_START..register::WINDOW_END).contains(&number) {
                    // A window register with no window selected has no home.
                    (!selected.is_empty()).then_some(())?;
                    for window in selected {
                        copy.windows[window].insert(number, value);
                    }
                } else {
                    copy.general.insert(number, value);
                }
            }
        }
        self.settle(&mut registers);
        Some(())
    }
}

/// Keeps the controllers' frames going by the clock, so that whoever waits
/// on their interrupts or syncpoints sees them, until `stop` is set. Run on
/// a dedicated thread; it notices `stop` within a frame.
pub fn keep_time(controllers: &[&Dc], stop: &AtomicBool) {
    let frame = Duration::from_nanos((NANOS_PER_SECOND / REFRESH_RATE) as u64);
    while !stop.load(Ordering::SeqCst) {
        controllers.iter().for_each(|dc| dc.count_frames());
        let next = controllers
            .iter()
            .filter_map(|dc| dc.next_frame())
            .min()
            .unwrap_or(frame)
            .min(frame);
        thread::sleep(next.max(Duration::from_micros(100)));
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, atomic::AtomicBool};

    use super::*;
    use crate::{
        boot::DramSize,
        memory::GuestRam,
        soc::mc::{self, Mc},
    };

    /// A controller whose clock stands still.
    fn controller() -> Dc {
        Dc::new(
            Arc::default(),
            Box::new(Duration::default),
            Box::new(|_| {}),
        )
    }

    /// A controller with a clock the test moves, and its interrupt level.
    struct Bench {
        dc: Dc,
        syncpoints: Arc<Syncpoints>,
        now: Arc<Mutex<Duration>>,
        level: Arc<AtomicBool>,
    }

    impl Bench {
        fn new() -> Self {
            let now = Arc::new(Mutex::new(Duration::ZERO));
            let syncpoints = Arc::new(Syncpoints::default());
            let level = Arc::new(AtomicBool::new(false));
            let (clock, line) = (Arc::clone(&now), Arc::clone(&level));
            let dc = Dc::new(
                Arc::clone(&syncpoints),
                Box::new(move || *clock.lock().unwrap()),
                Box::new(move |asserted| line.store(asserted, Ordering::SeqCst)),
            );
            Self {
                dc,
                syncpoints,
                now,
                level,
            }
        }

        fn advance(&self, milliseconds: u64) {
            *self.now.lock().unwrap() += Duration::from_millis(milliseconds);
            self.dc.count_frames();
        }

        fn count(&self, index: usize) -> u32 {
            self.dc.count_frames();
            self.syncpoints.get(index).unwrap()
        }
    }

    fn write(dc: &Dc, number: u64, value: u32) {
        dc.write(0, number * 4, 4, value.into()).unwrap();
    }

    fn read(dc: &Dc, number: u64) -> Option<u32> {
        dc.read(0, number * 4, 4).map(|value| value as u32)
    }

    /// Commits the general part as Linux and nvservices do.
    fn commit(dc: &Dc) {
        write(dc, register::STATE_CONTROL, 0x100);
        write(dc, register::STATE_CONTROL, 0x001);
    }

    #[test]
    fn frames_are_counted_at_the_refresh_rate_while_sending_with_a_syncpoint() {
        let bench = Bench::new();
        let dc = &bench.dc;
        write(dc, register::CONT_SYNCPT_VSYNC, 0x109);
        bench.advance(100);
        assert_eq!(bench.count(9), 0, "the controller is not sending yet");
        write(dc, register::DISPLAY_COMMAND, DISPLAY_MODE_CONTINUOUS);
        bench.advance(10);
        assert_eq!(bench.count(9), 0, "the command is not active yet");
        commit(dc);
        bench.advance(50);
        assert_eq!(bench.count(9), 3);
        bench.advance(50);
        assert_eq!(bench.count(9), 6);

        // Counting moves to another syncpoint, then stops with the display
        // at the frame boundary after the commit.
        bench.advance(20);
        write(dc, register::CONT_SYNCPT_VSYNC, 0x10A);
        bench.advance(1000);
        write(dc, register::DISPLAY_COMMAND, 0);
        commit(dc);
        bench.advance(1000);
        assert_eq!((bench.count(9), bench.count(10)), (7, 61));
    }

    #[test]
    fn frame_events_set_status_and_the_line_follows_enable_and_mask() {
        let bench = Bench::new();
        let dc = &bench.dc;
        write(dc, register::DISPLAY_COMMAND, DISPLAY_MODE_CONTINUOUS);
        commit(dc);
        // nvservices' interrupt thread start (text 0x246e8).
        write(dc, register::INT_STATUS, !0);
        write(dc, register::INT_MASK, 6);
        write(dc, register::INT_ENABLE, 6);
        assert_eq!(read(dc, register::INT_STATUS), Some(0));
        bench.advance(10);
        assert!(!bench.level.load(Ordering::SeqCst));
        bench.advance(10);
        assert!(bench.level.load(Ordering::SeqCst));
        let status = read(dc, register::INT_STATUS).unwrap();
        assert_eq!(status, FRAME_END_INT | VBLANK_INT);
        write(dc, register::INT_STATUS, status);
        assert!(!bench.level.load(Ordering::SeqCst));
        // Masked: the status is set, the line stays low.
        write(dc, register::INT_MASK, 0);
        bench.advance(17);
        assert_eq!(read(dc, register::INT_STATUS), Some(6));
        assert!(!bench.level.load(Ordering::SeqCst));
        write(dc, register::INT_MASK, 2);
        assert!(bench.level.load(Ordering::SeqCst));

        // A reset stops the controller and forgets everything.
        bench.dc.reset();
        assert!(!bench.level.load(Ordering::SeqCst));
        assert_eq!(read(dc, register::INT_STATUS), Some(0));
        assert_eq!(read(dc, register::INT_MASK), None);
        assert_eq!(read(dc, register::DISPLAY_COMMAND), None);
        bench.advance(100);
        assert_eq!(read(dc, register::INT_STATUS), Some(0));
    }

    #[test]
    fn a_flip_increments_its_syncpoint_when_the_update_takes_effect() {
        let bench = Bench::new();
        let dc = &bench.dc;
        // Stopped: the activation, and so the increment, happen at once.
        write(dc, register::DISPLAY_WINDOW_HEADER, 1 << 4);
        write(dc, register::WINBUF_START_ADDR, 0x1000);
        write(dc, register::STATE_CONTROL, 0x300);
        write(dc, register::STATE_CONTROL, 0x003);
        write(dc, register::INCR_SYNCPT[1], 3 | 0x300);
        assert_eq!(bench.syncpoints.get(3), Some(1));

        write(dc, register::DISPLAY_COMMAND, DISPLAY_MODE_CONTINUOUS);
        commit(dc);
        bench.advance(5);
        // Running: the new address and the increment wait for the frame
        // boundary.
        write(dc, register::WINBUF_START_ADDR, 0x2000);
        write(dc, register::STATE_CONTROL, 0x300);
        write(dc, register::STATE_CONTROL, 0x003);
        write(dc, register::INCR_SYNCPT[1], 3 | 0x300);
        write(dc, register::STATE_ACCESS, READ_MUX);
        assert_eq!(read(dc, register::WINBUF_START_ADDR), Some(0x1000));
        assert_eq!(bench.syncpoints.get(3), Some(1));
        bench.advance(12);
        assert_eq!(read(dc, register::WINBUF_START_ADDR), Some(0x2000));
        assert_eq!(bench.syncpoints.get(3), Some(2));

        // Nothing pending: register writes are safe right away. The
        // immediate condition never waits; others are not modeled.
        write(dc, register::INCR_SYNCPT[0], 1 | 0x300);
        write(dc, register::INCR_SYNCPT[0], 1);
        assert_eq!(bench.syncpoints.get(1), Some(2));
        assert_eq!(dc.write(0, 0, 4, 0x101), None);
    }

    #[test]
    fn window_registers_are_banked_by_the_header_and_latched_by_state_control() {
        let dc = controller();
        write(&dc, register::DISP_ACTIVE, 3 << 16 | 4);
        write(&dc, register::DISPLAY_WINDOW_HEADER, 1 << 4);
        write(&dc, register::WIN_SIZE, 2 << 16 | 2);
        assert_eq!(read(&dc, register::WIN_SIZE), Some(2 << 16 | 2));
        // Window B has its own, so far unwritten.
        write(&dc, register::DISPLAY_WINDOW_HEADER, 1 << 5);
        assert_eq!(read(&dc, register::WIN_SIZE), None);
        write(&dc, register::WIN_SIZE, 5);
        write(&dc, register::DISPLAY_WINDOW_HEADER, 1 << 4);
        assert_eq!(read(&dc, register::WIN_SIZE), Some(2 << 16 | 2));
        // With no window selected there is nowhere to write.
        write(&dc, register::DISPLAY_WINDOW_HEADER, 0);
        assert_eq!(dc.write(0, register::WIN_SIZE * 4, 4, 0), None);
        assert_eq!(read(&dc, register::DISP_ACTIVE), Some(3 << 16 | 4));
        assert_eq!(read(&dc, 0x300), None);

        // The active copy has nothing until the window's part is activated.
        write(&dc, register::DISPLAY_WINDOW_HEADER, 1 << 4);
        write(&dc, register::STATE_ACCESS, READ_MUX);
        assert_eq!(read(&dc, register::WIN_SIZE), None);
        // Activating window B's part leaves A's alone.
        write(&dc, register::STATE_CONTROL, 0x400);
        write(&dc, register::STATE_CONTROL, 0x004);
        assert_eq!(read(&dc, register::WIN_SIZE), None);
        write(&dc, register::DISPLAY_WINDOW_HEADER, 1 << 5);
        assert_eq!(read(&dc, register::WIN_SIZE), Some(5));
        // Writes to the active copy go around the assembly copy.
        write(&dc, register::STATE_ACCESS, READ_MUX | WRITE_MUX);
        write(&dc, register::WIN_SIZE, 6);
        assert_eq!(read(&dc, register::WIN_SIZE), Some(6));
        write(&dc, register::STATE_ACCESS, 0);
        assert_eq!(read(&dc, register::WIN_SIZE), Some(5));
        // Unknown state control bits are refused.
        assert_eq!(dc.write(0, register::STATE_CONTROL * 4, 4, 1 << 24), None);
    }

    const RAM_BASE: u64 = 0x8000_0000;
    /// Device address of the frame buffer and where it is in DRAM.
    const BUFFER: u64 = 0xC000_0000;
    const BUFFER_PHYSICAL: u64 = RAM_BASE + 0x4000;

    /// DRAM with four pages of frame buffer mapped for the controller.
    fn memory() -> (Arc<GuestRam>, DeviceMemory) {
        let ram = Arc::new(GuestRam::new(RAM_BASE, 0x1_0000).unwrap());
        let controller = Arc::new(Mc::new(DramSize::Gb4));
        let set =
            |register: u64, value: u32| controller.write(0, register, 4, value.into()).unwrap();
        set(0x1C, 7);
        set(0x20, 0xE000_0000 | (RAM_BASE >> 12) as u32);
        set(mc::SMMU_DC_ASID, 0x8404_0407);
        let table = RAM_BASE + 0x1000;
        let entry =
            |address: u64, value: u64| ram.write(address, &(value as u32).to_le_bytes()).unwrap();
        entry(RAM_BASE + (BUFFER >> 22) * 4, 0xF000_0000 | table >> 12);
        for page in 0..4 {
            entry(
                table + page * 4,
                0xE000_0000 | ((BUFFER_PHYSICAL >> 12) + page),
            );
        }
        let memory = DeviceMemory::new(&controller, &ram, mc::SMMU_DC_ASID);
        (ram, memory)
    }

    /// A 4x3 active area with a 2x2 window at (1, 1), eight bytes of stride,
    /// all of it active.
    fn configured() -> Dc {
        let dc = controller();
        write(&dc, register::DISP_ACTIVE, 3 << 16 | 4);
        write(&dc, register::DISPLAY_WINDOW_HEADER, 1 << 4);
        write(&dc, register::WIN_COLOR_DEPTH, COLOR_DEPTH_B8G8R8A8);
        write(&dc, register::WIN_POSITION, 1 << 16 | 1);
        write(&dc, register::WIN_SIZE, 2 << 16 | 2);
        write(&dc, register::WIN_PRESCALED_SIZE, 2 << 16 | 8);
        write(&dc, register::WIN_DDA_INC, DDA_ONE << 16 | DDA_ONE);
        write(&dc, register::WIN_LINE_STRIDE, 0x0004_0008);
        write(&dc, register::WINBUF_START_ADDR, BUFFER as u32);
        write(&dc, register::WIN_OPTIONS, WIN_ENABLE);
        write(&dc, register::STATE_CONTROL, 0x300);
        write(&dc, register::STATE_CONTROL, 0x003);
        dc
    }

    #[test]
    fn clockwise_rotation_puts_the_bottom_left_pixel_at_the_top_left() {
        // Two wide, three high: pixels numbered by row, then column.
        let rgb = (0..6).flat_map(|pixel| [pixel, pixel, pixel]).collect();
        let rotated = Frame {
            width: 2,
            height: 3,
            rgb,
        }
        .rotated_clockwise();
        assert_eq!((rotated.width, rotated.height), (3, 2));
        let pixels: Vec<u8> = rotated.rgb.chunks(3).map(|pixel| pixel[0]).collect();
        assert_eq!(pixels, [4, 2, 0, 5, 3, 1]);
    }

    #[test]
    fn frame_places_a_pitch_linear_window_on_a_black_active_area_with_colors_in_order() {
        let (ram, memory) = memory();
        let dc = configured();
        assert_eq!(dc.frame(&memory), Err(NoFrame::NotRunning));
        write(&dc, register::DISPLAY_COMMAND, DISPLAY_MODE_CONTINUOUS);
        commit(&dc);

        // Blue, green, red, alpha in memory.
        let pixels: [[u8; 4]; 4] = [[1, 2, 3, 0], [4, 5, 6, 0], [7, 8, 9, 0], [10, 11, 12, 0]];
        ram.write(BUFFER_PHYSICAL, pixels.as_flattened()).unwrap();
        let frame = dc.frame(&memory).unwrap();
        assert_eq!((frame.width, frame.height), (4, 3));
        #[rustfmt::skip]
        let expected = [
            0, 0, 0,  0, 0, 0,   0, 0, 0,    0, 0, 0,
            0, 0, 0,  3, 2, 1,   6, 5, 4,    0, 0, 0,
            0, 0, 0,  9, 8, 7,   12, 11, 10, 0, 0, 0,
        ];
        assert_eq!(frame.rgb, expected);

        // What is in the assembly copy is not shown until activated.
        write(&dc, register::WIN_COLOR_DEPTH, COLOR_DEPTH_R8G8B8A8);
        assert_eq!(dc.frame(&memory).unwrap().rgb[15..18], [3, 2, 1]);
        write(&dc, register::STATE_CONTROL, 0x200);
        write(&dc, register::STATE_CONTROL, 0x002);
        // The controller is running and its clock stands still: the
        // activation waits for a frame boundary that never comes.
        assert_eq!(dc.frame(&memory).unwrap().rgb[15..18], [3, 2, 1]);

        // Stopped, activations take effect at once. The horizontal offset
        // counts bytes: one pixel to the right, red first.
        let stopped = configured();
        write(&stopped, register::WIN_COLOR_DEPTH, COLOR_DEPTH_R8G8B8A8);
        write(&stopped, register::WINBUF_ADDR_H_OFFSET, 4);
        write(&stopped, register::DISPLAY_COMMAND, DISPLAY_MODE_CONTINUOUS);
        write(&stopped, register::STATE_CONTROL, 0x003);
        assert_eq!(stopped.frame(&memory).unwrap().rgb[15..18], [4, 5, 6]);

        let unmapped = configured();
        write(&unmapped, register::WINBUF_START_ADDR, 0x1000);
        write(
            &unmapped,
            register::DISPLAY_COMMAND,
            DISPLAY_MODE_CONTINUOUS,
        );
        write(&unmapped, register::STATE_CONTROL, 0x003);
        assert_eq!(
            unmapped.frame(&memory),
            Err(NoFrame::BufferUnmapped { address: 0x1000 })
        );
    }

    #[test]
    fn a_landscape_surface_scanned_by_columns_as_nvnflinger_asks_is_upright_on_the_console() {
        let (ram, memory) = memory();
        let dc = configured();
        // A 4x2 landscape surface, pixel n has blue n, on a 2x4 panel, as
        // nvnflinger flipped omm's logo: options 0x11 (columns, horizontal
        // reversal), the horizontal offset the last byte of a surface row,
        // the vertical one the first line (nvservices text 0x279d0).
        write(&dc, register::DISP_ACTIVE, 4 << 16 | 2);
        write(&dc, register::WIN_POSITION, 0);
        write(&dc, register::WIN_SIZE, 4 << 16 | 2);
        write(&dc, register::WIN_PRESCALED_SIZE, 4 << 16 | 8);
        write(&dc, register::WIN_LINE_STRIDE, 16);
        write(&dc, register::WINBUF_ADDR_H_OFFSET, 4 * 4 - 1);
        write(&dc, register::WINBUF_ADDR_V_OFFSET, 0);
        write(
            &dc,
            register::WIN_OPTIONS,
            WIN_ENABLE | SCAN_COLUMN | H_DIRECTION,
        );
        write(&dc, register::DISPLAY_COMMAND, DISPLAY_MODE_CONTINUOUS);
        write(&dc, register::STATE_CONTROL, 0x003);
        let surface: Vec<u8> = (0..8).flat_map(|pixel| [pixel, 0, 0, 0xFF]).collect();
        ram.write(BUFFER_PHYSICAL, &surface).unwrap();
        let seen = dc.frame(&memory).unwrap().rotated_clockwise();
        assert_eq!((seen.width, seen.height), (4, 2));
        let blues: Vec<u8> = seen.rgb.chunks(3).map(|pixel| pixel[2]).collect();
        assert_eq!(blues, [0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn scaling_steps_through_the_source_and_filters_only_when_enabled() {
        let black_white = [[0, 0, 0, 255], [255, 255, 255, 255]];
        let blues = |filters| {
            let scaling = Scaling {
                // Half a source pixel per destination pixel across, one down.
                increments: (DDA_ONE << 16) | (DDA_ONE / 2),
                initial: [0; 2],
                filters,
            };
            let scaled = scaling.apply(&black_white, [2, 1], [4, 1]);
            scaled.iter().map(|pixel| pixel[2]).collect::<Vec<_>>()
        };
        assert_eq!(blues([false, false]), [0, 0, 255, 255]);
        assert_eq!(blues([true, false]), [0, 128, 255, 255]);
        // nvservices' blank fill: 2x2 pixels stretched over the panel.
        let scaling = Scaling {
            increments: 0x0006_000B,
            initial: [0; 2],
            filters: [true; 2],
        };
        let scaled = scaling.apply(&[[9, 9, 9, 9]; 4], [2, 2], [720, 1280]);
        assert_eq!(scaled.len(), 720 * 1280);
        assert!(scaled.iter().all(|&pixel| pixel == [9, 9, 9, 9]));
    }

    #[test]
    fn block_linear_offsets_follow_the_gob_layout() {
        let layout = BlockLinear {
            start: 0,
            stride: 128,
            block_height: 2,
        };
        // Inside the first GOB: 16-byte pieces of line pairs interleave.
        assert_eq!(layout.offset(0, 0), 0);
        assert_eq!(layout.offset(16, 0), 32);
        assert_eq!(layout.offset(0, 1), 16);
        assert_eq!(layout.offset(32, 0), 256);
        assert_eq!(layout.offset(0, 2), 64);
        // The second GOB of the block is below the first, the next block to
        // the right, the next block row after both blocks.
        assert_eq!(layout.offset(0, 8), 512);
        assert_eq!(layout.offset(64, 0), 1024);
        assert_eq!(layout.offset(0, 16), 2048);
    }

    #[test]
    fn frame_reads_a_block_linear_window_through_the_smmu() {
        let (ram, memory) = memory();
        let dc = configured();
        // A 16x16 window of a 16-pixel-wide surface, one-GOB-high blocks.
        write(&dc, register::DISP_ACTIVE, 16 << 16 | 16);
        write(&dc, register::WIN_POSITION, 0);
        write(&dc, register::WIN_SIZE, 16 << 16 | 16);
        write(&dc, register::WIN_PRESCALED_SIZE, 16 << 16 | 64);
        write(&dc, register::WIN_LINE_STRIDE, 64);
        write(&dc, register::WINBUF_SURFACE_KIND, SURFACE_KIND_BLOCK);
        write(&dc, register::DISPLAY_COMMAND, DISPLAY_MODE_CONTINUOUS);
        write(&dc, register::STATE_CONTROL, 0x003);
        let layout = BlockLinear {
            start: 0,
            stride: 64,
            block_height: 1,
        };
        // Pixel (x, y) is blue x, green y.
        for y in 0..16 {
            for x in 0..16 {
                let at = BUFFER_PHYSICAL + layout.offset(x * 4, y);
                ram.write(at, &[x as u8, y as u8, 0, 0xFF]).unwrap();
            }
        }
        let frame = dc.frame(&memory).unwrap();
        for (y, x) in [(0, 0), (1, 4), (7, 15), (8, 3), (15, 15)] {
            let pixel = (y * 16 + x) * 3;
            assert_eq!(frame.rgb[pixel..pixel + 3], [0, y as u8, x as u8]);
        }
    }

    #[test]
    fn windows_are_layered_by_depth_and_blended_by_alpha() {
        let (ram, memory) = memory();
        let dc = configured();
        // Window A opaque at the back, B half transparent in front,
        // overlapping at (2, 1) and (2, 2).
        write(&dc, register::WIN_BLEND_LAYER_CONTROL, BLEND_BYPASS | 2);
        write(&dc, register::DISPLAY_WINDOW_HEADER, 1 << 5);
        for (number, value) in [
            (register::WIN_COLOR_DEPTH, COLOR_DEPTH_R8G8B8A8),
            (register::WIN_POSITION, 1 << 16 | 2),
            (register::WIN_SIZE, 2 << 16 | 1),
            (register::WIN_PRESCALED_SIZE, 2 << 16 | 4),
            (register::WIN_DDA_INC, DDA_ONE << 16 | DDA_ONE),
            (register::WIN_LINE_STRIDE, 4),
            (register::WINBUF_START_ADDR, (BUFFER + 0x100) as u32),
            (register::WIN_OPTIONS, WIN_ENABLE),
            // K1 = 255, non-premultiplied alpha.
            (register::WIN_BLEND_LAYER_CONTROL, 0xFF_FF00 | 1),
            (register::WIN_BLEND_MATCH_SELECT, 0x265),
        ] {
            write(&dc, number, value);
        }
        write(&dc, register::DISPLAY_COMMAND, DISPLAY_MODE_CONTINUOUS);
        write(&dc, register::STATE_CONTROL, 0x007);
        ram.write(BUFFER_PHYSICAL, &[200; 16]).unwrap();
        ram.write(BUFFER_PHYSICAL + 0x100, &[0, 0, 100, 0x80, 0, 0, 0, 0])
            .unwrap();
        let frame = dc.frame(&memory).unwrap();
        let pixel = |x: usize, y: usize| &frame.rgb[(y * 4 + x) * 3..][..3];
        assert_eq!(pixel(1, 1), [200, 200, 200]);
        // 200 * (1 - 128/255) + 100 * 128/255 = 149.8 for blue.
        assert_eq!(pixel(2, 1), [100, 100, 150]);
        // Fully transparent.
        assert_eq!(pixel(3, 1), [0, 0, 0]);
        assert_eq!(pixel(2, 2), [200, 200, 200]);
    }
}
