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
//! Frames pass at the panel's pace by guest time. The model brings itself up
//! to date whenever it is accessed, and [`keep_time`] does so at every frame
//! boundary for whoever waits on its interrupt. At a boundary the pending
//! activations take effect, syncpoint increments waiting for them happen,
//! the frame-end and vertical-blank interrupt status bits are set, and the
//! frame is counted on the syncpoint `CONT_SYNCPT_VSYNC` names. Frames are
//! counted from when the controller started sending, so a controller
//! restored from a snapshot ([`Dc`]'s [`Snapshot`]) with guest time set to
//! the saved time reaches its next boundary when the saved one would have.
//!
//! Nothing is scanned out continuously. [`Dc::frame_turned_clockwise`]
//! produces the picture the controller sends in the frame under way, as
//! the screen shows it, from the active copy: the enabled windows' frame
//! buffers, fetched through the SMMU like the hardware fetches them,
//! layered on the active area. It is put together at the first look in a
//! frame; later looks in the same frame get the same picture unless the
//! active copy changed.

use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread,
    time::Duration,
};

use serde::{Deserialize, Serialize};

use super::{
    host1x::{InterruptLine, Syncpoints},
    mc::DeviceMemory,
    regs::word,
};
use crate::{
    bus::MmioDevice,
    clock::GuestClock,
    snapshot::{Snapshot, SnapshotError, decode, encode},
};

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
    /// `DC_COM_DSC_TOP_CTL`. package1ldr's memory built-in self-test
    /// workaround sets bit 2 by read-modify-write while the head is fresh
    /// out of reset (its text `0x910`) and never reads it again; nothing
    /// else touches it. \[assumption\] It reads zero until written: its
    /// reset value is not public, and nothing ever looks at the other bits.
    pub const DSC_TOP_CTL: u64 = 0x33E;
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

/// Window header: one select bit per window, A first. There are four
/// windows: A to C, as Linux's `dc.h` names them and nvservices programs
/// them (annex §3.4), and D, which this chip adds and hekate uses (its
/// `di.h`: select bit 7, activation request bit 4, update bit 12, in line
/// with the others). \[assumption\] D is modelled as a window like the
/// others; the chip's is said to be simpler (no scaling), and hekate asks
/// nothing of it the others lack.
const WINDOW_SELECT_SHIFT: u32 = 4;
const WINDOW_COUNT: usize = 4;

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
/// 16 bits per pixel, named from the lowest bits up (Linux `dc.h`): blue in
/// bits 4:0, green in 10:5, red in 15:11. libnx's text console draws in
/// this format (`PIXEL_FORMAT_RGB_565`); seen for `hello-world.nro` run
/// from hbmenu.
const COLOR_DEPTH_B5G6R5: u32 = 6;
/// 16 bits per pixel, four each: red in bits 3:0, green in 7:4, blue in
/// 11:8, alpha in 15:12 (Linux `dc.c` gives this depth to
/// `DRM_FORMAT_ABGR4444`). The `fatal` program draws its error screen in
/// this format.
const COLOR_DEPTH_R4G4B4A4: u32 = 27;

/// The 16-bit pixel formats: where each of red, green, blue and alpha is
/// in the pixel (lowest bit, width); no alpha is opaque.
#[derive(Clone, Copy)]
struct Narrow([(u32, u32); 3], Option<(u32, u32)>);

impl Narrow {
    const B5G6R5: Self = Self([(11, 5), (5, 6), (0, 5)], None);
    const R4G4B4A4: Self = Self([(0, 4), (4, 4), (8, 4)], Some((12, 4)));
}

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

/// A picture, rows from the top, a 32-bit word per pixel: `0x00RRGGBB`,
/// eight bits each of red, green and blue, which is what the screen
/// window draws.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u32>,
}

impl Frame {
    /// The picture as bytes, red, green and blue per pixel.
    pub fn rgb(&self) -> Vec<u8> {
        self.pixels
            .iter()
            .flat_map(|pixel| {
                let [_, red, green, blue] = pixel.to_be_bytes();
                [red, green, blue]
            })
            .collect()
    }

    /// The picture turned a quarter turn clockwise: the reference
    /// [`Dc::frame_turned_clockwise`] is checked against.
    #[cfg(test)]
    pub fn rotated_clockwise(&self) -> Self {
        let (width, height) = (self.width as usize, self.height as usize);
        // Each row of the result is a column of the source, read upward.
        let pixels = (0..width)
            .flat_map(|column| (0..height).rev().map(move |row| row * width + column))
            .map(|at| self.pixels[at])
            .collect();
        Self {
            width: self.height,
            height: self.width,
            pixels,
        }
    }
}

/// One point of [`Dc::frame_turned_clockwise`]'s picture, and the
/// picture's size; see [`Dc::pixel_turned_clockwise`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pixel {
    pub width: u32,
    pub height: u32,
    /// `0x00RRGGBB`, as in [`Frame`]; `None` for a point outside the
    /// picture.
    pub value: Option<u32>,
}

impl Pixel {
    /// Point `(x, y)` of `frame`.
    fn of(frame: &Frame, x: u32, y: u32) -> Self {
        Self {
            width: frame.width,
            height: frame.height,
            value: (x < frame.width && y < frame.height)
                .then(|| frame.pixels[(y * frame.width + x) as usize]),
        }
    }
}

/// Why there is no picture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NoFrame {
    /// The controller has not been set up to send frames.
    NotRunning,
    /// It is set up in a way [`Dc::frame_turned_clockwise`] does not render.
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
#[derive(Default, Clone, PartialEq, Serialize, Deserialize)]
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

/// Everything the controller holds; what its [`Snapshot`] saves.
#[derive(Default, Serialize, Deserialize)]
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

#[derive(Serialize, Deserialize)]
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

    /// Makes the armed copy of the parts in `parts` active. Returns how
    /// many windows that gave another buffer (`WINBUF_START_ADDR`): the
    /// pictures a client presented.
    fn activate(&mut self, parts: u32) -> u64 {
        let mut flips = 0;
        for part in (0..PART_COUNT).filter(|part| parts >> part & 1 != 0) {
            let before = self.active.part(part).get(&register::WINBUF_START_ADDR).copied();
            *self.active.part_mut(part) = self.armed.part(part).clone();
            let after = self.active.part(part).get(&register::WINBUF_START_ADDR).copied();
            flips += u64::from(part > 0 && after.is_some() && before != after);
        }
        flips
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
    /// Guest time in nanoseconds before which [`Dc::settle`] finds nothing
    /// to do unless a register is accessed: the next frame boundary while
    /// the controller sends frames, [`u64::MAX`] while it does not. Lets
    /// [`Dc::count_frames`], which host1x calls on every syncpoint read,
    /// skip the register lock between boundaries.
    settled_until: AtomicU64,
    syncpoints: Arc<Syncpoints>,
    clock: Clock,
    /// INTID 105 for DC0, 106 for DC1 (nvservices text `0x23634`).
    line: InterruptLine,
    /// Buffers the picture is put together in, kept between pictures so
    /// that showing one 60 times a second does not allocate them anew.
    scratch: Mutex<Scratch>,
    /// The picture last put together, given again to every look within the
    /// same frame while the active copy stays as it was.
    shown: Mutex<Option<Shown>>,
    /// Pictures presented since the machine was built ([`Dc::flips`]).
    flips: AtomicU64,
}

/// See [`Dc::shown`].
struct Shown {
    /// The frame it was put together in: when the controller started
    /// sending and how many frames it had sent.
    frame: (Duration, u128),
    turned: bool,
    /// The active copy it was put together from.
    active: Copy,
    picture: Result<Arc<Frame>, NoFrame>,
}

/// See [`Dc::scratch`].
#[derive(Default)]
struct Scratch {
    /// A window's pixels as fetched from its surface.
    fetched: Vec<[u8; 4]>,
    /// A scaled window's pixels.
    scaled: Vec<[u8; 4]>,
    /// One horizontally scaled source line per source line.
    lines: Vec<[u32; 4]>,
    /// A block row of a block-linear surface.
    block_row: Vec<u8>,
    /// A 16-bit window's pixels as fetched, before they are widened.
    narrow: Vec<u8>,
}

impl Dc {
    /// `syncpoints` is where frames and updates are counted, `clock` what
    /// paces frames, `line` the controller's interrupt.
    pub fn new(syncpoints: Arc<Syncpoints>, clock: Clock, line: InterruptLine) -> Self {
        Self {
            registers: Mutex::default(),
            settled_until: AtomicU64::new(0),
            syncpoints,
            clock,
            line,
            scratch: Mutex::default(),
            shown: Mutex::default(),
            flips: AtomicU64::new(0),
        }
    }

    /// How many times a window has been given a new buffer to show since
    /// the machine was built: frames presented, whatever the refresh.
    pub fn flips(&self) -> u64 {
        self.flips.load(Ordering::Relaxed)
    }

    /// Brings the controller up to date with the frames sent since the last
    /// look.
    pub fn count_frames(&self) {
        let until = self.settled_until.load(Ordering::Acquire);
        if ((self.clock)().as_nanos()) < u128::from(until) {
            return;
        }
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

    /// Guest time of the next frame boundary, if the controller sends
    /// frames.
    fn next_frame(&self) -> Option<Duration> {
        let registers = self.registers.lock().unwrap();
        let running = registers.running.as_ref()?;
        let boundary = (running.frames + 1) * NANOS_PER_SECOND / REFRESH_RATE;
        Some(running.since.saturating_add(Duration::from_nanos(
            boundary.try_into().unwrap_or(u64::MAX),
        )))
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
                let flips = registers.activate(pending);
                self.flips.fetch_add(flips, Ordering::Relaxed);
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
        let until = registers.running.as_ref().map_or(u64::MAX, |running| {
            let boundary = (running.frames + 1) * NANOS_PER_SECOND / REFRESH_RATE;
            u64::try_from(running.since.as_nanos() + boundary).unwrap_or(u64::MAX)
        });
        self.settled_until.store(until, Ordering::Release);
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
            let flips = registers.activate(requests);
            self.flips.fetch_add(flips, Ordering::Relaxed);
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
    /// through `memory`, unturned: what the tests compare against the
    /// controller's own coordinates.
    #[cfg(test)]
    pub fn frame(&self, memory: &DeviceMemory) -> Result<Arc<Frame>, NoFrame> {
        self.render(memory, false)
    }

    /// The same picture turned a quarter turn clockwise, as the console's
    /// screen shows the panel, put together in one pass.
    pub fn frame_turned_clockwise(&self, memory: &DeviceMemory) -> Result<Arc<Frame>, NoFrame> {
        self.render(memory, true)
    }

    /// The picture from the active copy as it stands. It is put together
    /// once per frame the controller sends: a look within the same frame,
    /// with the active copy unchanged, is given the picture put together
    /// at the first. The register lock is held only for taking a copy of
    /// the active registers, so the guest's accesses to the controller
    /// (and host1x's syncpoint reads, which count its frames) do not wait
    /// for the frame buffers to be read.
    fn render(&self, memory: &DeviceMemory, turned: bool) -> Result<Arc<Frame>, NoFrame> {
        let (frame, active) = self.looked_at()?;
        let mut shown = self.shown.lock().unwrap();
        if let Some(last) = &*shown
            && last.frame == frame
            && last.turned == turned
            && last.active == active
        {
            return last.picture.clone();
        }
        // The last picture's pixels, unless someone still holds it.
        let reused = shown
            .take()
            .and_then(|last| Arc::into_inner(last.picture.ok()?))
            .map(|last| last.pixels)
            .unwrap_or_default();
        let picture = self.compose(memory, &active, turned, reused).map(Arc::new);
        *shown = Some(Shown {
            frame,
            turned,
            active,
            picture: picture.clone(),
        });
        picture
    }

    /// The frame under way (when the controller started sending and how
    /// many frames it has sent) and a copy of the active registers, the
    /// controller brought up to date first.
    fn looked_at(&self) -> Result<((Duration, u128), Copy), NoFrame> {
        let mut registers = self.registers.lock().unwrap();
        self.settle(&mut registers);
        let Some(running) = &registers.running else {
            return Err(NoFrame::NotRunning);
        };
        Ok(((running.since, running.frames), registers.active.clone()))
    }

    /// Point `(x, y)` of [`Dc::frame_turned_clockwise`]'s picture, worked
    /// out alone: the picture of this frame if it has been put together
    /// already, otherwise only the window pixels the point is made of are
    /// read, the windows refused where the picture refuses them. A buffer
    /// is found unmapped only where the point reads from it. What a wait
    /// for a color looks at a few hundred times a second.
    pub fn pixel_turned_clockwise(
        &self,
        memory: &DeviceMemory,
        x: u32,
        y: u32,
    ) -> Result<Pixel, NoFrame> {
        let (frame, active) = self.looked_at()?;
        if let Some(last) = &*self.shown.lock().unwrap()
            && last.frame == frame
            && last.turned
            && last.active == active
        {
            return match &last.picture {
                Ok(picture) => Ok(Pixel::of(picture, x, y)),
                Err(reason) => Err(reason.clone()),
            };
        }
        Self::compose_pixel(memory, &active, x, y)
    }

    /// [`Dc::pixel_turned_clockwise`] from `active`, the picture not put
    /// together.
    fn compose_pixel(
        memory: &DeviceMemory,
        active: &Copy,
        x: u32,
        y: u32,
    ) -> Result<Pixel, NoFrame> {
        let ([width, height], windows) = layers(active)?;
        // Point (x, y) of the turned picture is (y, height - 1 - x) of the
        // active area.
        let at = (x < height && y < width).then(|| [y, height - 1 - x]);
        // The background is black, and opaque.
        let mut value = 0;
        for window in windows {
            value = window.pixel(memory, [width, height], at, value)?;
        }
        Ok(Pixel {
            width: height,
            height: width,
            value: at.map(|_| value),
        })
    }

    /// Puts the picture together from `active` into `pixels`, whatever
    /// they held before.
    fn compose(
        &self,
        memory: &DeviceMemory,
        active: &Copy,
        turned: bool,
        mut pixels: Vec<u32>,
    ) -> Result<Frame, NoFrame> {
        let ([width, height], windows) = layers(active)?;
        // The background is black, and opaque.
        pixels.clear();
        pixels.resize((width * height) as usize, 0);
        let mut canvas = Canvas {
            pixels,
            area: [width, height],
            turned,
        };
        let scratch = &mut *self.scratch.lock().unwrap();
        for window in windows {
            window.draw(memory, scratch, &mut canvas)?;
        }
        let (width, height) = if turned {
            (height, width)
        } else {
            (width, height)
        };
        Ok(Frame {
            width,
            height,
            pixels: canvas.pixels,
        })
    }
}

/// The size of the active area `active` sets up and its enabled windows,
/// back to front, or why they make no picture.
fn layers(active: &Copy) -> Result<([u32; 2], Vec<Window<'_>>), NoFrame> {
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
    Ok(([width, height], windows))
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

    /// Fetches the window's pixels, red, green, blue and alpha each, into
    /// `scratch.fetched`, and says where each pixel of the window is there.
    fn fetch(&self, memory: &DeviceMemory, scratch: &mut Scratch) -> Result<Layout, NoFrame> {
        self.source()?.fetch(memory, scratch)
    }

    /// Where the window's pixels come from and how they are laid out:
    /// everything [`Window::fetch`] works out from the registers before it
    /// reads the surface, refused where it refuses.
    fn source(&self) -> Result<Source, NoFrame> {
        let halves = |value: u32| (value & 0xFFFF, value >> 16);
        // Red first among the 32-bit formats, or the layout of 16 bits.
        let depth = match self.value(register::WIN_COLOR_DEPTH) {
            COLOR_DEPTH_B8G8R8A8 => Ok(false),
            COLOR_DEPTH_R8G8B8A8 => Ok(true),
            COLOR_DEPTH_B5G6R5 => Err(Narrow::B5G6R5),
            COLOR_DEPTH_R4G4B4A4 => Err(Narrow::R4G4B4A4),
            depth => return Err(NoFrame::Unsupported(format!("color depth {depth}"))),
        };
        let pixel_bytes: u32 = if depth.is_ok() { 4 } else { 2 };
        let byte_swap = self.value(register::WIN_BYTE_SWAP);
        if byte_swap > BYTE_SWAP_SWAP4 || (pixel_bytes == 2 && byte_swap > BYTE_SWAP_SWAP2) {
            return Err(NoFrame::Unsupported(format!(
                "byte swap {byte_swap} of {pixel_bytes}-byte pixels"
            )));
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
        if !source_bytes.is_multiple_of(pixel_bytes) || source_bytes == 0 || source_lines == 0 {
            return Err(NoFrame::Unsupported(format!(
                "a source of {source_bytes} bytes by {source_lines} lines"
            )));
        }
        let (across, down) = (source_bytes / pixel_bytes, source_lines);
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
        let row_bytes = u64::from(source_width * pixel_bytes);
        // The horizontal offset counts bytes, the vertical one lines; a
        // window scanned right to left or bottom up starts from its last
        // byte or line (nvservices text 0x279d0, Linux likewise).
        let mut left = u64::from(self.value(register::WINBUF_ADDR_H_OFFSET));
        let mut top = u64::from(self.value(register::WINBUF_ADDR_V_OFFSET));
        let (reversed_across, reversed_down) =
            (options & H_DIRECTION != 0, options & V_DIRECTION != 0);
        if reversed_across {
            left = (left + 1)
                .checked_sub(row_bytes)
                .ok_or_else(|| NoFrame::Unsupported("a window left of its surface".into()))?;
        }
        if reversed_down {
            top = (top + 1)
                .checked_sub(source_height.into())
                .ok_or_else(|| NoFrame::Unsupported("a window above its surface".into()))?;
        }
        let kind = self.value(register::WINBUF_SURFACE_KIND);
        let surface = match kind & SURFACE_KIND_MASK {
            SURFACE_KIND_PITCH => Surface::Pitch { start, stride },
            SURFACE_KIND_BLOCK => {
                let block_height =
                    1 << (kind >> SURFACE_KIND_BLOCK_HEIGHT_SHIFT & SURFACE_KIND_BLOCK_HEIGHT);
                let layout = BlockLinear {
                    start,
                    stride,
                    block_height,
                };
                layout.check(left, row_bytes)?;
                Surface::Block(layout)
            }
            _ => return Err(NoFrame::Unsupported(format!("surface kind {kind:#x}"))),
        };
        // Which byte of a pixel as fetched is red, green, blue and alpha.
        let swapped = match byte_swap {
            BYTE_SWAP_SWAP2 => [1, 0, 3, 2],
            BYTE_SWAP_SWAP4 => [3, 2, 1, 0],
            _ => [0, 1, 2, 3],
        };
        let order = match depth {
            Ok(true) => swapped,
            Ok(false) => [swapped[2], swapped[1], swapped[0], swapped[3]],
            // Widened into red, green, blue and alpha.
            Err(_) => [0, 1, 2, 3],
        };
        // Where pixel (x, y) of the source, as the window scans it, is in
        // the surface's rectangle: the directions reverse the surface's
        // axes, and scanning by columns swaps them.
        let (surface_width, surface_height) = (source_width as isize, source_height as isize);
        let (step_x, step_y) = (
            if reversed_across { -1 } else { 1 },
            if reversed_down {
                -surface_width
            } else {
                surface_width
            },
        );
        let origin = if reversed_across {
            surface_width - 1
        } else {
            0
        } + if reversed_down {
            (surface_height - 1) * surface_width
        } else {
            0
        };
        Ok(Source {
            depth,
            swapped_halves: byte_swap == BYTE_SWAP_SWAP2,
            expand: options & COLOR_EXPAND != 0,
            pixel_bytes,
            surface,
            corner: (left, top),
            row_bytes,
            rectangle: [source_width, source_height],
            layout: Layout {
                width: across,
                height: down,
                origin: origin as usize,
                steps: if column {
                    [step_y, step_x]
                } else {
                    [step_x, step_y]
                },
                order,
            },
            scaling: Scaling {
                increments: self.value(register::WIN_DDA_INC),
                initial: [
                    self.value(register::WIN_H_INITIAL_DDA),
                    self.value(register::WIN_V_INITIAL_DDA),
                ],
                filters: [options & H_FILTER != 0, options & V_FILTER != 0],
            },
            size: [width, height],
        })
    }

    /// What the window makes of point `at` of the active area, which is
    /// `area` in size, over `below` there: the window refused where
    /// [`Window::draw`] refuses it, but only what that point takes read
    /// from its surface, and nothing if the window does not cover it. The
    /// surface is read where the point takes its pixels from, so a buffer
    /// unmapped only elsewhere does not keep the point from being worked
    /// out, as it keeps the whole picture from being put together.
    fn pixel(
        &self,
        memory: &DeviceMemory,
        area: [u32; 2],
        at: Option<[u32; 2]>,
        below: u32,
    ) -> Result<u32, NoFrame> {
        let (left, top) = {
            let position = self.value(register::WIN_POSITION);
            (position & 0xFFFF, position >> 16)
        };
        let source = self.source()?;
        let [width, height] = source.size;
        if left + width > area[0] || top + height > area[1] {
            return Err(NoFrame::Unsupported(
                "a window reaching outside the active area".into(),
            ));
        }
        let blend = Blend::of(self)?;
        let Some([x, y]) = at.filter(|&[x, y]| {
            (left..left + width).contains(&x) && (top..top + height).contains(&y)
        }) else {
            return Ok(below);
        };
        let pixel = source.pixel(memory, [x - left, y - top])?;
        let [red, green, blue, alpha] = source.layout.order.map(|byte| pixel[byte]);
        Ok(match blend {
            Blend::Opaque => u32::from_be_bytes([0, red, green, blue]),
            Blend::Factors { select, k1, k2 } => {
                BlendTable::new(select, k1, k2).apply([red, green, blue, alpha], below)
            }
        })
    }

    /// Lays the window over `canvas`, the active area so far.
    fn draw(
        &self,
        memory: &DeviceMemory,
        scratch: &mut Scratch,
        canvas: &mut Canvas,
    ) -> Result<(), NoFrame> {
        let (left, top) = {
            let position = self.value(register::WIN_POSITION);
            (position & 0xFFFF, position >> 16)
        };
        let layout = self.fetch(memory, scratch)?;
        let [area_width, area_height] = canvas.area;
        if left + layout.width > area_width || top + layout.height > area_height {
            return Err(NoFrame::Unsupported(
                "a window reaching outside the active area".into(),
            ));
        }
        let blend = Blend::of(self)?;
        let pixels = &scratch.fetched;
        let (width, height) = (layout.width as usize, layout.height as usize);
        if width == 0 || height == 0 {
            return Ok(());
        }
        let (left, top) = (left as usize, top as usize);
        let (area_width, area_height) = (area_width as usize, area_height as usize);
        // Visit the window in the order the canvas is laid out: lines of
        // the canvas one after the other, and the window's pixels along a
        // canvas line as a start and a step in `pixels`.
        let lines = if canvas.turned {
            // Canvas line `left + x` holds window column `x`, bottom up.
            Lines {
                count: width,
                length: height,
                canvas: left * area_height + (area_height - top - height),
                canvas_stride: area_height,
                start: layout.at(0, height - 1),
                start_stride: layout.steps[0],
                step: -layout.steps[1],
            }
        } else {
            Lines {
                count: height,
                length: width,
                canvas: top * area_width + left,
                canvas_stride: area_width,
                start: layout.at(0, 0),
                start_stride: layout.steps[1],
                step: layout.steps[0],
            }
        };
        let canvas = &mut canvas.pixels[..];
        match blend {
            // The two orders the firmware's 32-bit formats come in get a
            // copy of their own, which the compiler can make tight: red
            // first (`R8G8B8A8`) needs the word's bytes reversed, blue first
            // only alpha cleared.
            Blend::Opaque => match layout.order {
                [0, 1, 2, _] => lay(canvas, pixels, lines, |pixel, _| {
                    u32::from_le_bytes(*pixel).swap_bytes() >> 8
                }),
                [2, 1, 0, _] => lay(canvas, pixels, lines, |pixel, _| {
                    u32::from_le_bytes(*pixel) & 0x00FF_FFFF
                }),
                [red, green, blue, _] => lay(canvas, pixels, lines, |pixel, _| {
                    u32::from_be_bytes([0, pixel[red], pixel[green], pixel[blue]])
                }),
            },
            Blend::Factors { select, k1, k2 } => {
                let table = BlendTable::new(select, k1, k2);
                lay(canvas, pixels, lines, |pixel, below| {
                    table.apply(layout.order.map(|byte| pixel[byte]), below)
                })
            }
        }
        Ok(())
    }
}

/// What [`Window::source`] works out: how the window's surface is read
/// and how its pixels are laid out as read.
struct Source {
    /// Red first among the 32-bit formats, or the layout of 16 bits.
    depth: Result<bool, Narrow>,
    /// 16-bit pixels with their bytes swapped.
    swapped_halves: bool,
    /// The window's `COLOR_EXPAND` option; see [`widen`].
    expand: bool,
    pixel_bytes: u32,
    surface: Surface,
    /// Where the rectangle the window shows starts on the surface: bytes
    /// across and lines down.
    corner: (u64, u64),
    /// Bytes across the rectangle.
    row_bytes: u64,
    /// The rectangle's pixels across and lines down, in the surface's own
    /// orientation: how the pixels are laid out as read.
    rectangle: [u32; 2],
    /// Where each pixel of the source, as the window scans it, is among
    /// the pixels read.
    layout: Layout,
    scaling: Scaling,
    /// The window's size on the screen.
    size: [u32; 2],
}

enum Surface {
    Pitch { start: u64, stride: u64 },
    Block(BlockLinear),
}

impl Source {
    /// [`Window::fetch`] once the registers are checked.
    fn fetch(&self, memory: &DeviceMemory, scratch: &mut Scratch) -> Result<Layout, NoFrame> {
        let Scratch {
            fetched,
            block_row,
            narrow,
            ..
        } = &mut *scratch;
        let [source_width, source_height] = self.rectangle;
        // Every byte is read over, so what was there stays until it is.
        fetched.resize((source_width * source_height) as usize, [0; 4]);
        let bytes: &mut [u8] = if self.depth.is_ok() {
            fetched.as_flattened_mut()
        } else {
            narrow.resize(fetched.len() * 2, 0);
            &mut narrow[..]
        };
        let (left, top) = self.corner;
        match &self.surface {
            Surface::Pitch { start, stride } => {
                for (row, line) in bytes.chunks_mut(self.row_bytes as usize).enumerate() {
                    let address = start + (top + row as u64) * stride + left;
                    memory
                        .read(address, line)
                        .ok_or(NoFrame::BufferUnmapped { address })?;
                }
            }
            Surface::Block(layout) => {
                layout.read(memory, (left, top), self.row_bytes, bytes, block_row)?
            }
        }
        if let Err(format) = self.depth {
            widen(narrow, fetched, format, self.swapped_halves, self.expand);
        }
        let source = self.layout;
        let [width, height] = self.size;
        if self
            .scaling
            .is_identity([source.width, source.height], self.size)
        {
            return Ok(source);
        }
        self.scaling.apply(
            &scratch.fetched,
            &source,
            self.size,
            &mut scratch.lines,
            &mut scratch.scaled,
        );
        std::mem::swap(&mut scratch.fetched, &mut scratch.scaled);
        // Scaling works on each byte alike, so the order stays.
        Ok(Layout {
            width,
            height,
            origin: 0,
            steps: [1, width as isize],
            order: source.order,
        })
    }

    /// Pixel `(x, y)` of the window, as [`Source::fetch`] leaves it at
    /// `(x, y)` of the layout it says, from only the source pixels it is
    /// made of.
    fn pixel(&self, memory: &DeviceMemory, [x, y]: [u32; 2]) -> Result<[u8; 4], NoFrame> {
        const ONE: u32 = 1 << 12;
        let source = &self.layout;
        let read = |x: usize, y: usize| self.read(memory, source.at(x, y) as usize);
        if self
            .scaling
            .is_identity([source.width, source.height], self.size)
        {
            return read(x as usize, y as usize);
        }
        // [`Scaling::apply`] for one destination pixel.
        let size = [source.width, source.height];
        let (left, right, across) = self.scaling.tap(0, size, x.into());
        let (top, bottom, down) = self.scaling.tap(1, size, y.into());
        let line = |y: usize| -> Result<[u32; 4], NoFrame> {
            let (a, b) = (read(left, y)?, read(right, y)?);
            Ok(std::array::from_fn(|channel| {
                u32::from(a[channel]) * (ONE - across) + u32::from(b[channel]) * across
            }))
        };
        let (upper, lower) = (line(top)?, line(bottom)?);
        Ok(std::array::from_fn(|channel| {
            ((upper[channel] * (ONE - down) + lower[channel] * down + (1 << 23)) >> 24) as u8
        }))
    }

    /// Pixel `index` of those read, read alone, widened if 16-bit. An
    /// unmapped buffer is reported at the address [`Source::fetch`] would
    /// have failed to read: the line's (pitch) or its block row's.
    fn read(&self, memory: &DeviceMemory, index: usize) -> Result<[u8; 4], NoFrame> {
        let source_width = self.rectangle[0] as usize;
        let (row, column) = (index / source_width, index % source_width);
        let pixel_bytes = self.pixel_bytes as usize;
        let (left, top) = self.corner;
        let y = top + row as u64;
        let mut bytes = [0; 4];
        for (offset, byte) in bytes[..pixel_bytes].iter_mut().enumerate() {
            // The byte's place in the line as fetched.
            let across = (column * pixel_bytes + offset) as u64;
            let (unmapped, address) = match &self.surface {
                Surface::Pitch { start, stride } => {
                    let line = start + y * stride + left;
                    (line, line + across)
                }
                Surface::Block(layout) => layout.byte(left, y, across),
            };
            memory
                .read(address, std::slice::from_mut(byte))
                .ok_or(NoFrame::BufferUnmapped { address: unmapped })?;
        }
        Ok(match self.depth {
            Ok(_) => bytes,
            Err(format) => {
                let mut pixel = [[0; 4]];
                widen(
                    &bytes[..2],
                    &mut pixel,
                    format,
                    self.swapped_halves,
                    self.expand,
                );
                pixel[0]
            }
        })
    }
}

/// The canvas lines a window covers and where their pixels are in the
/// window's: line `k` starts at `canvas + k * canvas_stride` and takes its
/// `length` pixels from `start + k * start_stride` on, `step` apart.
#[derive(Clone, Copy)]
struct Lines {
    count: usize,
    length: usize,
    canvas: usize,
    canvas_stride: usize,
    start: isize,
    start_stride: isize,
    step: isize,
}

/// How many canvas lines are filled side by side when a line's pixels lie
/// apart in the window's (a window turned by the canvas): each position
/// along them then takes neighbouring pixels of the window, not one from
/// each of as many window lines.
const LINES_TOGETHER: usize = 16;

/// Lays `lines` of `pixels` onto `canvas`, each canvas pixel becoming what
/// `put` makes of the window's pixel and the canvas pixel below it.
fn lay(canvas: &mut [u32], pixels: &[[u8; 4]], lines: Lines, put: impl Fn(&[u8; 4], u32) -> u32) {
    let Lines {
        count,
        length,
        canvas: first_line,
        canvas_stride,
        start,
        start_stride,
        step,
    } = lines;
    let line_start = |line: usize| start + line as isize * start_stride;
    if step.abs() == 1 {
        // Pixels next to each other, forwards or backwards: a straight copy.
        for line in 0..count {
            let out = &mut canvas[first_line + line * canvas_stride..][..length];
            let from = line_start(line) as usize;
            if step == 1 {
                for (out, pixel) in out.iter_mut().zip(&pixels[from..][..length]) {
                    *out = put(pixel, *out);
                }
            } else {
                let source = &pixels[from + 1 - length..][..length];
                for (out, pixel) in out.iter_mut().zip(source.iter().rev()) {
                    *out = put(pixel, *out);
                }
            }
        }
        return;
    }
    for group in (0..count).step_by(LINES_TOGETHER) {
        let group = group..count.min(group + LINES_TOGETHER);
        for along in 0..length {
            let offset = along as isize * step;
            for line in group.clone() {
                let out = &mut canvas[first_line + line * canvas_stride + along];
                *out = put(&pixels[(line_start(line) + offset) as usize], *out);
            }
        }
    }
}

/// Widens 16-bit pixels into red, green, blue and alpha bytes. With
/// `expand` (the window's `COLOR_EXPAND` option) each component's top bits
/// are repeated into the low bits it lacks, so white stays white; without
/// it the low bits are zero. That is what the option's name and Linux's
/// use of it (always set for 16-bit windows) suggest [assumption: the
/// TRM's description of the option is not at hand].
fn widen(narrow: &[u8], pixels: &mut [[u8; 4]], format: Narrow, swapped: bool, expand: bool) {
    let Narrow(colors, alpha) = format;
    let grow = |value: u16, (low, bits): (u32, u32)| {
        let value = u32::from(value) >> low & ((1 << bits) - 1);
        let high = value << (8 - bits);
        (if expand { high | high >> bits } else { high }) as u8
    };
    for (pixel, two) in pixels.iter_mut().zip(narrow.as_chunks::<2>().0) {
        let value = if swapped {
            u16::from_be_bytes(*two)
        } else {
            u16::from_le_bytes(*two)
        };
        *pixel = [
            grow(value, colors[0]),
            grow(value, colors[1]),
            grow(value, colors[2]),
            alpha.map_or(0xFF, |alpha| grow(value, alpha)),
        ];
    }
}

/// The picture being put together, a line after the other, the active
/// area as it is or turned a quarter turn clockwise; see [`Frame`].
struct Canvas {
    pixels: Vec<u32>,
    /// The active area's width and height, unturned.
    area: [u32; 2],
    turned: bool,
}

/// Where a window's pixels are in a buffer: pixel `(x, y)` is at
/// `origin + x * steps[0] + y * steps[1]`, its red, green, blue and alpha
/// the bytes `order` names.
#[derive(Clone, Copy)]
struct Layout {
    width: u32,
    height: u32,
    origin: usize,
    steps: [isize; 2],
    order: [usize; 4],
}

impl Layout {
    fn at(&self, x: usize, y: usize) -> isize {
        self.origin as isize + x as isize * self.steps[0] + y as isize * self.steps[1]
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
    /// The steps across and down.
    fn steps(&self) -> [u32; 2] {
        [self.increments & 0xFFFF, self.increments >> 16]
    }

    /// Whether `source` comes out at `destination` pixel for pixel.
    fn is_identity(&self, source: [u32; 2], destination: [u32; 2]) -> bool {
        source == destination && self.steps() == [DDA_ONE; 2] && self.initial == [0; 2]
    }

    /// Per direction (`axis`) and destination position: the two pixels of
    /// a source `size` across and down it takes, and the weight of the
    /// second.
    fn tap(&self, axis: usize, size: [u32; 2], position: u64) -> (usize, usize, u32) {
        let last = size[axis].saturating_sub(1) as usize;
        let at = u64::from(self.initial[axis]) + position * u64::from(self.steps()[axis]);
        let whole = ((at >> 12) as usize).min(last);
        let weight = if self.filters[axis] && whole < last {
            (at & 0xFFF) as u32
        } else {
            0
        };
        (whole, (whole + 1).min(last), weight)
    }

    /// Scales the window's source, laid out in `pixels` as `source` says,
    /// to `destination` in `out`, a line after the other. `lines` holds the
    /// source's lines scaled across on the way. Weights have the
    /// analyzer's 12 fractional bits; results are rounded.
    fn apply(
        &self,
        pixels: &[[u8; 4]],
        source: &Layout,
        destination: [u32; 2],
        lines: &mut Vec<[u32; 4]>,
        out: &mut Vec<[u8; 4]>,
    ) {
        const ONE: u32 = 1 << 12;
        let size = [source.width, source.height];
        let taps = |axis: usize| -> Vec<(usize, usize, u32)> {
            (0..u64::from(destination[axis]))
                .map(|position| self.tap(axis, size, position))
                .collect()
        };
        let (columns, rows) = (taps(0), taps(1));
        let width = destination[0] as usize;
        // Every source line scaled across, in 12 fractional bits.
        lines.clear();
        for y in 0..source.height as usize {
            lines.extend(columns.iter().map(|&(left, right, weight)| {
                let (a, b) = (
                    pixels[source.at(left, y) as usize],
                    pixels[source.at(right, y) as usize],
                );
                std::array::from_fn(|channel| {
                    u32::from(a[channel]) * (ONE - weight) + u32::from(b[channel]) * weight
                })
            }));
        }
        out.clear();
        for &(top, bottom, weight) in &rows {
            let (upper, lower) = (
                &lines[top * width..][..width],
                &lines[bottom * width..][..width],
            );
            out.extend(upper.iter().zip(lower).map(|(a, b)| {
                std::array::from_fn(|channel| {
                    ((a[channel] * (ONE - weight) + b[channel] * weight + (1 << 23)) >> 24) as u8
                })
            }));
        }
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

    /// `below` and the result are `0x00RRGGBB` words. What [`BlendTable`]
    /// works out with less arithmetic per pixel.
    #[cfg(test)]
    fn apply(&self, [red, green, blue, alpha]: [u8; 4], below: u32) -> u32 {
        let Self::Factors { select, k1, k2 } = *self else {
            return u32::from_be_bytes([0, red, green, blue]);
        };
        let unit = |channel: u8| f32::from(channel) / 255.0;
        let source = [red, green, blue].map(unit);
        let [_, below @ ..] = below.to_be_bytes();
        let below = below.map(unit);
        let (source_factor, below_factor) = Self::factors(select, k1, k2, unit(alpha));
        let [red, green, blue]: [u8; 3] = std::array::from_fn(|channel| {
            let value = source[channel] * source_factor + below[channel] * below_factor;
            (value * 255.0).round().clamp(0.0, 255.0) as u8
        });
        u32::from_be_bytes([0, red, green, blue])
    }

    /// The factors of the window's color and of the color below for a
    /// window pixel of `alpha`, from zero to one.
    fn factors(select: u32, k1: f32, k2: f32, alpha: f32) -> (f32, f32) {
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
        (source_factor, below_factor)
    }
}

/// [`Blend::Factors`] for a whole window: each byte value as a fraction of
/// one and the factors for each alpha, worked out once instead of per
/// pixel, with the same arithmetic, so the results are the same.
struct BlendTable {
    units: [f32; 256],
    factors: [(f32, f32); 256],
}

impl BlendTable {
    fn new(select: u32, k1: f32, k2: f32) -> Self {
        let units = std::array::from_fn(|byte| f32::from(byte as u8) / 255.0);
        Self {
            units,
            factors: std::array::from_fn(|alpha| Blend::factors(select, k1, k2, units[alpha])),
        }
    }

    /// `below` and the result are `0x00RRGGBB` words.
    fn apply(&self, [red, green, blue, alpha]: [u8; 4], below: u32) -> u32 {
        let (source_factor, below_factor) = self.factors[usize::from(alpha)];
        let [_, below @ ..] = below.to_be_bytes();
        let source = [red, green, blue];
        let [red, green, blue]: [u8; 3] = std::array::from_fn(|channel| {
            let value = self.units[usize::from(source[channel])] * source_factor
                + self.units[usize::from(below[channel])] * below_factor;
            (value * 255.0).round().clamp(0.0, 255.0) as u8
        });
        u32::from_be_bytes([0, red, green, blue])
    }
}

/// Where the GPU's block-linear layout puts the bytes of a surface: rows of
/// blocks, each block one GOB wide and `block_height` GOBs high, the GOBs
/// of a block one after the other, the blocks of a row one after the other.
/// Inside a GOB, 16-byte pieces of two lines are interleaved \[recalled,
/// not checked against the hardware\]; nvservices hands the controller
/// such surfaces with the block
/// height in GOBs as a power of two (text `0x28620`).
struct BlockLinear {
    start: u64,
    /// Bytes across the surface, a whole number of GOBs.
    stride: u64,
    block_height: u64,
}

impl BlockLinear {
    /// Offset of the byte `x` bytes across and `y` lines down; what
    /// [`Self::read`] works out piece by piece.
    #[cfg(test)]
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

    /// Refuses a surface [`Self::read`] cannot read `row_bytes` of from
    /// byte `left` on.
    fn check(&self, left: u64, row_bytes: u64) -> Result<(), NoFrame> {
        if !self.stride.is_multiple_of(GOB_WIDTH) || left + row_bytes > self.stride {
            return Err(NoFrame::Unsupported(format!(
                "a block-linear surface {} bytes across",
                self.stride
            )));
        }
        Ok(())
    }

    fn block_lines(&self) -> u64 {
        GOB_HEIGHT * self.block_height
    }

    fn block_row_size(&self) -> u64 {
        (self.stride / GOB_WIDTH) * GOB_SIZE * self.block_height
    }

    /// Where in its block row [`Self::read`] finds the sixteen bytes of a
    /// line from `x` on, `in_line` being what depends on the line.
    fn piece(&self, x: u64, in_line: u64) -> usize {
        (x / GOB_WIDTH * GOB_SIZE * self.block_height
            + (x % 64 / 32) * 256
            + (x % 32 / 16) * 32
            + x % 16
            + in_line) as usize
    }

    /// The part of [`Self::offset`] in its block row that depends on the
    /// line.
    fn in_line(&self, y: u64) -> u64 {
        (y % self.block_lines() / GOB_HEIGHT) * GOB_SIZE + (y % 8 / 2) * 64 + (y % 2) * 16
    }

    /// Reads `row_bytes` across of each line from (`left`, `top`) on into
    /// `bytes`, a block row at a time through `block_row`.
    fn read(
        &self,
        memory: &DeviceMemory,
        (left, top): (u64, u64),
        row_bytes: u64,
        bytes: &mut [u8],
        block_row: &mut Vec<u8>,
    ) -> Result<(), NoFrame> {
        self.check(left, row_bytes)?;
        let block_lines = self.block_lines();
        let block_row_size = self.block_row_size();
        let lines = bytes.len() as u64 / row_bytes;
        block_row.resize(block_row_size as usize, 0);
        let mut loaded = None;
        for line in 0..lines {
            let y = top + line;
            let block = y / block_lines;
            if loaded != Some(block) {
                let address = self.start + block * block_row_size;
                memory
                    .read(address, block_row)
                    .ok_or(NoFrame::BufferUnmapped { address })?;
                loaded = Some(block);
            }
            let in_line = self.in_line(y);
            let out = &mut bytes[(line * row_bytes) as usize..][..row_bytes as usize];
            // Sixteen bytes at a time stay together.
            for (piece, chunk) in out.chunks_mut(16).enumerate() {
                let at = self.piece(left + piece as u64 * 16, in_line);
                chunk.copy_from_slice(&block_row[at..at + chunk.len()]);
            }
        }
        Ok(())
    }

    /// Where [`Self::read`] takes byte `across` of line `y` from, reading
    /// from byte `left` of each line on: the address of the line's block
    /// row, and of the byte.
    fn byte(&self, left: u64, y: u64, across: u64) -> (u64, u64) {
        let block_row = self.start + y / self.block_lines() * self.block_row_size();
        let at = self.piece(left + across / 16 * 16, self.in_line(y)) as u64 + across % 16;
        (block_row, block_row + at)
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
                // Not known before they are written; nobody can tell.
                matches!(
                    number,
                    register::DISP_TIMING_OPTIONS | register::DSC_TOP_CTL
                )
                .then_some(&0)
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
                let selected = registers
                    .selected_windows()
                    .fold(0u32, |selected, window| selected | 1 << window);
                let copy = if direct {
                    &mut registers.active
                } else {
                    &mut registers.assembly
                };
                if (register::WINDOW_START..register::WINDOW_END).contains(&number) {
                    // A window register with no window selected has no home.
                    (selected != 0).then_some(())?;
                    for (window, registers) in copy.windows.iter_mut().enumerate() {
                        if selected >> window & 1 != 0 {
                            registers.insert(number, value);
                        }
                    }
                } else {
                    copy.general.insert(number, value);
                }
            }
        }
        self.settle(&mut registers);
        Some(())
    }

    fn snapshot(&self) -> Option<&dyn Snapshot> {
        Some(self)
    }
}

/// Saves the command and the assembly, armed and active registers, the
/// activations pending for the next frame boundary and the syncpoint
/// increments waiting for them, the interrupt status and the level last
/// driven onto the line, and, while it sends frames, since when (guest
/// time) and how many it has sent. The picture buffers are host scratch,
/// dropped on restore. The line is not driven again: the interrupt
/// controller saves its own level.
impl Snapshot for Dc {
    /// Counts the frames sent by the paused guest time, which the frame
    /// thread might otherwise do in the middle of the save.
    fn quiesce(&self) -> Result<(), SnapshotError> {
        self.count_frames();
        Ok(())
    }

    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        encode(&*self.registers.lock().unwrap())
    }

    fn restore(&self, state: &[u8]) -> Result<(), SnapshotError> {
        let registers = decode("display controller", state)?;
        *self.registers.lock().unwrap() = registers;
        self.settled_until.store(0, Ordering::Release);
        *self.scratch.lock().unwrap() = Scratch::default();
        *self.shown.lock().unwrap() = None;
        Ok(())
    }
}

/// Keeps the controllers' frames going by guest time, so that whoever
/// waits on their interrupts or syncpoints sees them, until `stop` is set.
/// `clock` is the one the controllers count frames by. Run on a dedicated
/// thread; it notices `stop` within a frame of guest time, or a short poll
/// while the clock stands still, when no frame passes.
pub fn keep_time(controllers: &[&Dc], clock: &GuestClock, stop: &AtomicBool) {
    let frame = Duration::from_nanos((NANOS_PER_SECOND / REFRESH_RATE) as u64);
    while !stop.load(Ordering::SeqCst) {
        controllers.iter().for_each(|dc| dc.count_frames());
        let latest = clock.now() + frame;
        let next = controllers
            .iter()
            .filter_map(|dc| dc.next_frame())
            .min()
            .map_or(latest, |next| next.min(latest));
        thread::sleep(clock.until(next).max(Duration::from_micros(100)));
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

    /// DRAM with 8 MiB mapped for the controller at [`BUFFER`] in two 4 MiB
    /// sections, for full-screen windows.
    fn screen_memory() -> (Arc<GuestRam>, DeviceMemory) {
        let ram = Arc::new(GuestRam::new(RAM_BASE, 0xC0_0000).unwrap());
        let controller = Arc::new(Mc::new(DramSize::Gb4));
        let set =
            |register: u64, value: u32| controller.write(0, register, 4, value.into()).unwrap();
        set(0x1C, 7);
        set(0x20, 0xE000_0000 | (RAM_BASE >> 12) as u32);
        set(mc::SMMU_DC_ASID, 0x8404_0407);
        for section in 0..2 {
            let physical = RAM_BASE + 0x40_0000 * (section + 1);
            let entry = 0xC000_0000 | (physical >> 12) as u32;
            ram.write(
                RAM_BASE + ((BUFFER >> 22) + section) * 4,
                &entry.to_le_bytes(),
            )
            .unwrap();
        }
        let memory = DeviceMemory::new(&controller, &ram, mc::SMMU_DC_ASID);
        (ram, memory)
    }

    /// Programs `window` of the panel's 720x1280 head as nvnflinger's flips
    /// do (`/tmp/main/h5.log` at 15.98 s and 16.06 s): a 1280x720 surface
    /// shown through scan columns with horizontal reversal, filters on.
    fn full_screen_window(dc: &Dc, window: u32, source: (u32, u32), surface: (u32, u32)) {
        let (across, down) = source;
        let (kind, stride) = surface;
        write(dc, register::DISPLAY_WINDOW_HEADER, 0x10 << window);
        for (number, value) in [
            (register::WIN_COLOR_DEPTH, COLOR_DEPTH_R8G8B8A8),
            (register::WIN_POSITION, 0),
            (register::WIN_SIZE, 1280 << 16 | 720),
            (register::WIN_PRESCALED_SIZE, (down << 16) | (across * 4)),
            (
                register::WIN_DDA_INC,
                (((down << 12) / 1280) << 16) | ((across << 12) / 720),
            ),
            (register::WIN_LINE_STRIDE, stride),
            (register::WINBUF_SURFACE_KIND, kind),
            (register::WINBUF_START_ADDR, BUFFER as u32),
            (register::WINBUF_ADDR_H_OFFSET, down * 4 - 1),
            (register::WIN_OPTIONS, 0xC000_0511),
        ] {
            write(dc, number, value);
        }
    }

    /// The two windows HOME shows (a `--verbose` run at the lock screen): a
    /// 2x2 fill scaled up and blended with premultiplied alpha over a
    /// block-linear picture behind it.
    fn home_windows(dc: &Dc) {
        full_screen_window(dc, 2, (720, 1280), (0x42, 0x1400));
        write(dc, register::WIN_BLEND_LAYER_CONTROL, 0x0100_00FF);
        full_screen_window(dc, 1, (2, 2), (0, 8));
        write(dc, register::WIN_BLEND_LAYER_CONTROL, 0x00FF_FF80);
        write(dc, register::WIN_BLEND_MATCH_SELECT, 0x262);
    }

    /// Time `Soc::screen` spends on the panel's picture, per kind of window
    /// nvnflinger flips. A measurement, not a check: run with
    /// `cargo test --release --lib soc::dc::tests::screen_timing -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn screen_timing() {
        let (ram, memory) = screen_memory();
        let pattern: Vec<u8> = (0..0x80_0000u32)
            .map(|byte| ((byte * 7) >> 3) as u8)
            .collect();
        ram.write(BUFFER_PHYSICAL - 0x4000 + 0x40_0000, &pattern)
            .unwrap();
        let single = |source, surface| move |dc: &Dc| full_screen_window(dc, 1, source, surface);
        type SetUp<'a> = &'a dyn Fn(&Dc);
        let cases: [(&str, SetUp); 4] = [
            ("pitch-linear 1280x720", &single((720, 1280), (0, 0x1400))),
            (
                "block-linear 1280x720",
                &single((720, 1280), (0x42, 0x1400)),
            ),
            ("2x2 blank fill scaled up", &single((2, 2), (0, 8))),
            ("HOME: fill blended over block-linear", &home_windows),
        ];
        for (name, set_up) in cases {
            let dc = controller();
            write(&dc, register::DISP_ACTIVE, 1280 << 16 | 720);
            set_up(&dc);
            write(&dc, register::DISPLAY_COMMAND, DISPLAY_MODE_CONTINUOUS);
            write(&dc, register::STATE_CONTROL, 0x0F);
            // The best and the median of many, the machine being shared.
            // Each is put together anew, as in a frame of its own.
            let mut times: Vec<_> = (0..50)
                .map(|_| {
                    if let Some(shown) = &mut *dc.shown.lock().unwrap() {
                        shown.frame.1 += 1;
                    }
                    let start = std::time::Instant::now();
                    let frame = dc.frame_turned_clockwise(&memory).unwrap();
                    assert_eq!((frame.width, frame.height), (1280, 720));
                    start.elapsed()
                })
                .collect();
            times.sort();
            // A point looked at alone, as a wait for a color does: the
            // picture not put together in the frame.
            let (_, active) = dc.looked_at().unwrap();
            let start = std::time::Instant::now();
            for point in 0..1000 {
                let pixel = Dc::compose_pixel(&memory, &active, point % 1280, point % 720);
                assert!(pixel.unwrap().value.is_some());
            }
            println!(
                "{name}: best {:?}, median {:?} per picture, {:?} per point alone",
                times[0],
                times[25],
                start.elapsed() / 1000
            );
        }
    }

    #[test]
    fn clockwise_rotation_puts_the_bottom_left_pixel_at_the_top_left() {
        // Two wide, three high: pixels numbered by row, then column.
        let rotated = Frame {
            width: 2,
            height: 3,
            pixels: (0..6).collect(),
        }
        .rotated_clockwise();
        assert_eq!((rotated.width, rotated.height), (3, 2));
        assert_eq!(rotated.pixels, [4, 2, 0, 5, 3, 1]);
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
        assert_eq!(frame.rgb(), expected);

        // What is in the assembly copy is not shown until activated.
        write(&dc, register::WIN_COLOR_DEPTH, COLOR_DEPTH_R8G8B8A8);
        assert_eq!(dc.frame(&memory).unwrap().rgb()[15..18], [3, 2, 1]);
        write(&dc, register::STATE_CONTROL, 0x200);
        write(&dc, register::STATE_CONTROL, 0x002);
        // The controller is running and its clock stands still: the
        // activation waits for a frame boundary that never comes.
        assert_eq!(dc.frame(&memory).unwrap().rgb()[15..18], [3, 2, 1]);

        // Stopped, activations take effect at once. The horizontal offset
        // counts bytes: one pixel to the right, red first.
        let stopped = configured();
        write(&stopped, register::WIN_COLOR_DEPTH, COLOR_DEPTH_R8G8B8A8);
        write(&stopped, register::WINBUF_ADDR_H_OFFSET, 4);
        write(&stopped, register::DISPLAY_COMMAND, DISPLAY_MODE_CONTINUOUS);
        write(&stopped, register::STATE_CONTROL, 0x003);
        assert_eq!(stopped.frame(&memory).unwrap().rgb()[15..18], [4, 5, 6]);

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
    fn sixteen_bit_pixels_widen_with_color_expansion_and_pad_with_zeros_without() {
        let (ram, memory) = memory();
        // White, red; green, blue: two lines of two `B5G6R5` pixels, eight
        // bytes apart.
        for (line, pair) in [[0xFFFF_u16, 0xF800], [0x07E0, 0x001F]].iter().enumerate() {
            let bytes: Vec<u8> = pair.iter().flat_map(|pixel| pixel.to_le_bytes()).collect();
            ram.write(BUFFER_PHYSICAL + 8 * line as u64, &bytes)
                .unwrap();
        }
        let shown = |options: u32| {
            let dc = configured();
            write(&dc, register::WIN_COLOR_DEPTH, COLOR_DEPTH_B5G6R5);
            write(&dc, register::WIN_PRESCALED_SIZE, 2 << 16 | 4);
            write(&dc, register::WIN_OPTIONS, WIN_ENABLE | options);
            write(&dc, register::DISPLAY_COMMAND, DISPLAY_MODE_CONTINUOUS);
            write(&dc, register::STATE_CONTROL, 0x003);
            let rgb = dc.frame(&memory).unwrap().rgb();
            [&rgb[15..21], &rgb[27..33]].concat()
        };
        assert_eq!(
            shown(COLOR_EXPAND),
            [255, 255, 255, 255, 0, 0, 0, 255, 0, 0, 0, 255]
        );
        // The same four as `R4G4B4A4`: 0xFFFF is white, 0xF800 blue 8 with
        // full alpha, 0x07E0 green 14 and blue 7, 0x001F red 15 and green 1.
        let dc = configured();
        write(&dc, register::WIN_COLOR_DEPTH, COLOR_DEPTH_R4G4B4A4);
        write(&dc, register::WIN_PRESCALED_SIZE, 2 << 16 | 4);
        write(&dc, register::WIN_OPTIONS, WIN_ENABLE | COLOR_EXPAND);
        write(&dc, register::DISPLAY_COMMAND, DISPLAY_MODE_CONTINUOUS);
        write(&dc, register::STATE_CONTROL, 0x003);
        let rgb = dc.frame(&memory).unwrap().rgb();
        assert_eq!(rgb[15..21], [255, 255, 255, 0, 0, 0x88]);
        assert_eq!(rgb[27..33], [0, 0xEE, 0x77, 0xFF, 0x11, 0]);
        assert_eq!(shown(0), [248, 252, 248, 248, 0, 0, 0, 252, 0, 0, 0, 248]);
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
        let blues: Vec<u8> = seen.rgb().chunks(3).map(|pixel| pixel[2]).collect();
        assert_eq!(blues, [0, 1, 2, 3, 4, 5, 6, 7]);
    }

    /// `pixels`, `size` wide and high a line after the other, scaled.
    fn scaled(scaling: &Scaling, pixels: &[[u8; 4]], size: [u32; 2], to: [u32; 2]) -> Vec<[u8; 4]> {
        let layout = Layout {
            width: size[0],
            height: size[1],
            origin: 0,
            steps: [1, size[0] as isize],
            order: [0, 1, 2, 3],
        };
        let mut out = Vec::new();
        scaling.apply(pixels, &layout, to, &mut Vec::new(), &mut out);
        out
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
            let out = scaled(&scaling, &black_white, [2, 1], [4, 1]);
            out.iter().map(|pixel| pixel[2]).collect::<Vec<_>>()
        };
        assert_eq!(blues([false, false]), [0, 0, 255, 255]);
        assert_eq!(blues([true, false]), [0, 128, 255, 255]);
        // nvservices' blank fill: 2x2 pixels stretched over the panel.
        let scaling = Scaling {
            increments: 0x0006_000B,
            initial: [0; 2],
            filters: [true; 2],
        };
        let out = scaled(&scaling, &[[9, 9, 9, 9]; 4], [2, 2], [720, 1280]);
        assert_eq!(out.len(), 720 * 1280);
        assert!(out.iter().all(|&pixel| pixel == [9, 9, 9, 9]));
    }

    #[test]
    fn the_direct_path_matches_the_general_one_and_the_turned_picture_the_rotation() {
        let (ram, memory) = memory();
        let dc = configured();
        // A 5x3 surface, pixel n blue n, shown through columns with both
        // directions reversed: a 3x5 window at (1, 0) of a 4x5 area.
        let surface: Vec<u8> = (0..15).flat_map(|n| [n, 2 * n, 3 * n, 0xFF]).collect();
        ram.write(BUFFER_PHYSICAL, &surface).unwrap();
        for (number, value) in [
            (register::WIN_POSITION, 1),
            (register::WIN_SIZE, 5 << 16 | 3),
            (register::WIN_PRESCALED_SIZE, 5 << 16 | 12),
            (register::WIN_LINE_STRIDE, 20),
            (register::WINBUF_ADDR_H_OFFSET, 19),
            (register::WINBUF_ADDR_V_OFFSET, 2),
            (
                register::WIN_OPTIONS,
                WIN_ENABLE | SCAN_COLUMN | H_DIRECTION | V_DIRECTION,
            ),
        ] {
            write(&dc, number, value);
        }
        write(&dc, register::DISP_ACTIVE, 5 << 16 | 4);
        write(&dc, register::DISPLAY_COMMAND, DISPLAY_MODE_CONTINUOUS);
        write(&dc, register::STATE_CONTROL, 0x003);
        let portrait = dc.frame(&memory).unwrap();
        // Window pixel (x, y) is surface pixel (4 − y, 2 − x).
        for y in 0..5 {
            for x in 0..3 {
                let at = (y * 4 + 1 + x) * 3;
                let n = ((2 - x) * 5 + 4 - y) as u8;
                assert_eq!(portrait.rgb()[at..at + 3], [3 * n, 2 * n, n], "{x},{y}");
            }
        }
        assert_eq!(
            *dc.frame_turned_clockwise(&memory).unwrap(),
            portrait.rotated_clockwise()
        );

        // The general path, given the same layout and no scaling, takes the
        // same pixels.
        let pixels: Vec<[u8; 4]> = (0..15).map(|n| [n, 0, 0, 0]).collect();
        let layout = Layout {
            width: 3,
            height: 5,
            origin: 14,
            steps: [-5, -1],
            order: [0, 1, 2, 3],
        };
        let identity = Scaling {
            increments: DDA_ONE << 16 | DDA_ONE,
            initial: [0; 2],
            filters: [true; 2],
        };
        let mut out = Vec::new();
        identity.apply(&pixels, &layout, [3, 5], &mut Vec::new(), &mut out);
        for y in 0..5 {
            for x in 0..3 {
                assert_eq!(out[y * 3 + x], pixels[layout.at(x, y) as usize]);
            }
        }
    }

    /// [`lay`] fills canvas lines side by side when their pixels lie apart;
    /// it must come out as taking each line's pixels one after the other.
    #[test]
    fn laying_lines_side_by_side_matches_laying_them_one_after_the_other() {
        let (across, down) = (37usize, 23usize);
        let pixels: Vec<[u8; 4]> = (0..across * down)
            .map(|n| (n as u32 * 0x0101_0007).to_le_bytes())
            .collect();
        let put = |pixel: &[u8; 4], below: u32| below.rotate_left(5) ^ u32::from_le_bytes(*pixel);
        let (a, d) = (across as isize, down as isize);
        // Rows forwards and backwards, and columns (a turned canvas) up and
        // down, from either end.
        let cases = [
            (down, across, 0, a, 1),
            (down, across, a - 1, a, -1),
            (across, down, (d - 1) * a, 1, -a),
            (across, down, (d - 1) * a + a - 1, -1, -a),
            (across, down, 0, 1, a),
            (across, down, a - 1, -1, a),
        ];
        for (count, length, start, start_stride, step) in cases {
            let lines = Lines {
                count,
                length,
                canvas: 5,
                canvas_stride: length + 3,
                start,
                start_stride,
                step,
            };
            let below: Vec<u32> = (0..5 + count * (length + 3) + 7)
                .map(|n| n as u32 * 2_654_435_761)
                .collect();
            let mut laid = below.clone();
            lay(&mut laid, &pixels, lines, put);
            let mut expected = below;
            for line in 0..count {
                let mut at = start + line as isize * start_stride;
                for along in 0..length {
                    let out = &mut expected[5 + line * (length + 3) + along];
                    *out = put(&pixels[at as usize], *out);
                    at += step;
                }
            }
            assert_eq!(laid, expected, "{count} lines from {start}, step {step}");
        }
    }

    /// A scaled window, whose pixels a turned canvas takes a column at a
    /// time, and a block-linear one: the turned picture is the rotation.
    #[test]
    fn turned_pictures_of_full_screen_windows_are_the_rotations() {
        let (ram, memory) = screen_memory();
        let pattern: Vec<u8> = (0..0x80_0000u32)
            .map(|byte| (byte.wrapping_mul(2_654_435_761) >> 24) as u8)
            .collect();
        ram.write(BUFFER_PHYSICAL - 0x4000 + 0x40_0000, &pattern)
            .unwrap();
        let cases = [
            ((720, 1280), (0x42, 0x1400)),
            ((5, 7), (0, 0x40)),
            ((360, 640), (0, 0x1400)),
        ];
        for (source, surface) in cases {
            let dc = controller();
            write(&dc, register::DISP_ACTIVE, 1280 << 16 | 720);
            full_screen_window(&dc, 1, source, surface);
            write(&dc, register::DISPLAY_COMMAND, DISPLAY_MODE_CONTINUOUS);
            write(&dc, register::STATE_CONTROL, 0x0F);
            let unturned = dc.frame(&memory).unwrap();
            assert_eq!(
                *dc.frame_turned_clockwise(&memory).unwrap(),
                unturned.rotated_clockwise(),
                "{source:?}"
            );
        }
        let dc = controller();
        write(&dc, register::DISP_ACTIVE, 1280 << 16 | 720);
        home_windows(&dc);
        write(&dc, register::DISPLAY_COMMAND, DISPLAY_MODE_CONTINUOUS);
        write(&dc, register::STATE_CONTROL, 0x0F);
        let unturned = dc.frame(&memory).unwrap();
        assert_eq!(
            *dc.frame_turned_clockwise(&memory).unwrap(),
            unturned.rotated_clockwise()
        );
    }

    /// The blend table's arithmetic is the per-pixel one's, for every
    /// selection, alpha and a spread of colors and constants.
    #[test]
    fn blending_through_the_table_gives_what_blending_each_pixel_gives() {
        let colors = [0u8, 1, 37, 127, 128, 200, 254, 255];
        for select in (0..=5).flat_map(|low| (0..8).map(move |high| high << 4 | low)) {
            for (k1, k2) in [(255u32, 0u32), (128, 255), (1, 77), (0, 0)] {
                let (k1, k2) = (k1 as f32 / 255.0, k2 as f32 / 255.0);
                let blend = Blend::Factors { select, k1, k2 };
                let table = BlendTable::new(select, k1, k2);
                for alpha in 0..=255u8 {
                    for (&source, &below) in colors.iter().zip(colors.iter().rev()) {
                        let pixel = [source, below, source / 2, alpha];
                        let below = u32::from_be_bytes([0, below, source, 99]);
                        assert_eq!(
                            table.apply(pixel, below),
                            blend.apply(pixel, below),
                            "select {select:#x}, alpha {alpha}"
                        );
                    }
                }
            }
        }
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
            assert_eq!(frame.rgb()[pixel..pixel + 3], [0, y as u8, x as u8]);
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
        let rgb = dc.frame(&memory).unwrap().rgb();
        let pixel = |x: usize, y: usize| &rgb[(y * 4 + x) * 3..][..3];
        assert_eq!(pixel(1, 1), [200, 200, 200]);
        // 200 * (1 - 128/255) + 100 * 128/255 = 149.8 for blue.
        assert_eq!(pixel(2, 1), [100, 100, 150]);
        // Fully transparent.
        assert_eq!(pixel(3, 1), [0, 0, 0]);
        assert_eq!(pixel(2, 2), [200, 200, 200]);
    }

    #[test]
    fn a_restored_controller_reaches_its_next_boundary_at_the_saved_guest_time() {
        let saved = Bench::new();
        let dc = &saved.dc;
        write(dc, register::CONT_SYNCPT_VSYNC, 0x109);
        write(dc, register::DISPLAY_COMMAND, DISPLAY_MODE_CONTINUOUS);
        commit(dc);
        saved.advance(20);
        assert_eq!(saved.count(9), 1);
        // A stop committed for the next boundary, and an increment waiting
        // for it.
        write(dc, register::DISPLAY_COMMAND, 0);
        commit(dc);
        write(
            dc,
            register::INCR_SYNCPT[0],
            CONDITION_REG_WR_SAFE << 8 | 12,
        );
        write(dc, register::INT_STATUS, !0);
        let state = dc.save().unwrap();

        let restored = Bench::new();
        *restored.now.lock().unwrap() = Duration::from_millis(20);
        // The syncpoints are host1x's to restore.
        restored.syncpoints.add(9, 1);
        restored.dc.restore(&state).unwrap();
        for bench in [&saved, &restored] {
            assert_eq!(
                bench.dc.next_frame(),
                Some(Duration::from_nanos(33_333_333))
            );
            bench.advance(13);
            assert_eq!((bench.count(9), bench.count(12)), (1, 0));
            assert_eq!(read(&bench.dc, register::INT_STATUS), Some(0));
            bench.advance(1);
            assert_eq!((bench.count(9), bench.count(12)), (2, 1));
            assert_eq!(
                read(&bench.dc, register::INT_STATUS),
                Some(FRAME_END_INT | VBLANK_INT)
            );
            bench.advance(100);
            assert_eq!(bench.count(9), 2, "the committed stop took effect");
        }
    }

    /// Points of the turned picture worked out alone are the picture's:
    /// every `step`th across and down and the last of each, and points
    /// just outside it are none. A controller that makes no picture
    /// refuses a point for the same reason.
    fn assert_points_are_the_picture(dc: &Dc, memory: &DeviceMemory, step: usize) {
        let (_, active) = dc.looked_at().unwrap();
        let picture = dc.frame_turned_clockwise(memory);
        let frame = match &picture {
            Ok(frame) => frame,
            Err(reason) => {
                assert_eq!(
                    Dc::compose_pixel(memory, &active, 0, 0).as_ref(),
                    Err(reason)
                );
                return;
            }
        };
        let points = |size: u32| (0..size).step_by(step).chain([size - 1, size, size + 9]);
        for y in points(frame.height) {
            for x in points(frame.width) {
                assert_eq!(
                    Dc::compose_pixel(memory, &active, x, y),
                    Ok(Pixel::of(frame, x, y)),
                    "{x},{y}"
                );
            }
        }
        // Within the frame, the picture put together is looked at instead.
        let (x, y) = (frame.width / 2, frame.height / 3);
        assert_eq!(
            dc.pixel_turned_clockwise(memory, x, y),
            Ok(Pixel::of(frame, x, y))
        );
    }

    #[test]
    fn points_of_full_screen_and_home_windows_are_the_pictures() {
        let (ram, memory) = screen_memory();
        let pattern: Vec<u8> = (0..0x80_0000u32)
            .map(|byte| (byte.wrapping_mul(2_654_435_761) >> 24) as u8)
            .collect();
        ram.write(BUFFER_PHYSICAL - 0x4000 + 0x40_0000, &pattern)
            .unwrap();
        let set_ups: [&dyn Fn(&Dc); 4] = [
            &|dc| full_screen_window(dc, 1, (720, 1280), (0x42, 0x1400)),
            &|dc| full_screen_window(dc, 1, (5, 7), (0, 0x40)),
            &|dc| full_screen_window(dc, 1, (360, 640), (0, 0x1400)),
            &home_windows,
        ];
        for set_up in set_ups {
            let dc = controller();
            write(&dc, register::DISP_ACTIVE, 1280 << 16 | 720);
            set_up(&dc);
            write(&dc, register::DISPLAY_COMMAND, DISPLAY_MODE_CONTINUOUS);
            write(&dc, register::STATE_CONTROL, 0x0F);
            assert!(dc.frame_turned_clockwise(&memory).is_ok());
            assert_points_are_the_picture(&dc, &memory, 3);
        }
    }

    /// Windows of every kind the compositor renders, made up at random:
    /// each color depth and byte swap, both directions, scanned by rows
    /// and columns, pitch and block-linear surfaces at odd offsets, scaled
    /// with and without filters, layered and blended.
    #[test]
    fn points_of_made_up_windows_are_the_pictures() {
        let (ram, memory) = screen_memory();
        let pattern: Vec<u8> = (0..0x40_0000u32)
            .map(|byte| (byte.wrapping_mul(2_246_822_519) >> 23) as u8)
            .collect();
        ram.write(BUFFER_PHYSICAL - 0x4000 + 0x40_0000, &pattern)
            .unwrap();
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut random = |below: u32| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % u64::from(below)) as u32
        };
        let mut pictured = 0;
        for _ in 0..400 {
            let dc = controller();
            let (area_width, area_height) = (8 + random(41), 8 + random(41));
            write(&dc, register::DISP_ACTIVE, area_height << 16 | area_width);
            let windows = 1 + random(3);
            for window in 0..windows {
                write(&dc, register::DISPLAY_WINDOW_HEADER, 0x10 << window);
                let (depth, pixel_bytes) = [
                    (COLOR_DEPTH_B8G8R8A8, 4),
                    (COLOR_DEPTH_R8G8B8A8, 4),
                    (COLOR_DEPTH_B5G6R5, 2),
                    (COLOR_DEPTH_R4G4B4A4, 2),
                ][random(4) as usize];
                let byte_swap = random(if pixel_bytes == 4 { 3 } else { 2 });
                let options = [
                    H_DIRECTION,
                    V_DIRECTION,
                    SCAN_COLUMN,
                    COLOR_EXPAND,
                    H_FILTER,
                    V_FILTER,
                ]
                .into_iter()
                .filter(|_| random(2) == 1)
                .fold(WIN_ENABLE, |options, option| options | option);
                let (width, height) = (1 + random(area_width), 1 + random(area_height));
                let position =
                    random(area_height - height + 1) << 16 | random(area_width - width + 1);
                let (across, down) = (1 + random(40), 1 + random(40));
                let (rectangle_width, rectangle_height) = if options & SCAN_COLUMN != 0 {
                    (down, across)
                } else {
                    (across, down)
                };
                let row_bytes = rectangle_width * pixel_bytes;
                let (mut left, mut top) = (random(70), random(9));
                let block = random(2) == 1;
                let stride = if block {
                    (left + row_bytes).div_ceil(64) * 64 + 64 * random(3)
                } else {
                    left + row_bytes + random(50)
                };
                if options & H_DIRECTION != 0 {
                    left += row_bytes - 1;
                }
                if options & V_DIRECTION != 0 {
                    top += rectangle_height - 1;
                }
                let scaled = random(3) != 0;
                let increment =
                    |source: u32, shown: u32, random: &mut dyn FnMut(u32) -> u32| match scaled {
                        true => (((source << 12) / shown).max(1) + random(0x200)).min(0xFFFF),
                        false => DDA_ONE,
                    };
                let increments = increment(down, height, &mut random) << 16
                    | increment(across, width, &mut random);
                let (initial_across, initial_down) = match scaled {
                    true => (random(0x1000), random(0x1000)),
                    false => (0, 0),
                };
                let kind = if block {
                    SURFACE_KIND_BLOCK | random(6) << SURFACE_KIND_BLOCK_HEIGHT_SHIFT
                } else {
                    SURFACE_KIND_PITCH
                };
                for (number, value) in [
                    (register::WIN_COLOR_DEPTH, depth),
                    (register::WIN_BYTE_SWAP, byte_swap),
                    (register::WIN_OPTIONS, options),
                    (register::WIN_POSITION, position),
                    (register::WIN_SIZE, height << 16 | width),
                    (
                        register::WIN_PRESCALED_SIZE,
                        (down << 16) | (across * pixel_bytes),
                    ),
                    (register::WIN_DDA_INC, increments),
                    (register::WIN_H_INITIAL_DDA, initial_across),
                    (register::WIN_V_INITIAL_DDA, initial_down),
                    (register::WIN_LINE_STRIDE, stride),
                    (register::WINBUF_SURFACE_KIND, kind),
                    (
                        register::WINBUF_START_ADDR,
                        BUFFER as u32 + random(0x1_0000),
                    ),
                    (register::WINBUF_ADDR_H_OFFSET, left),
                    (register::WINBUF_ADDR_V_OFFSET, top),
                ] {
                    write(&dc, number, value);
                }
                if windows > 1 || random(2) == 1 {
                    let bypass = if random(4) == 0 { BLEND_BYPASS } else { 0 };
                    let control = bypass | random(1 << 16) << 8 | random(256);
                    write(&dc, register::WIN_BLEND_LAYER_CONTROL, control);
                    let select = random(8) << 4 | random(6);
                    write(&dc, register::WIN_BLEND_MATCH_SELECT, select);
                }
            }
            write(&dc, register::DISPLAY_COMMAND, DISPLAY_MODE_CONTINUOUS);
            write(&dc, register::STATE_CONTROL, 0x0F);
            if dc.frame_turned_clockwise(&memory).is_ok() {
                pictured += 1;
            }
            assert_points_are_the_picture(&dc, &memory, 1);
        }
        assert!(pictured > 350, "only {pictured} of 400 made a picture");
    }

    /// A controller that makes no picture refuses every point for the
    /// reason it refuses the picture.
    #[test]
    fn points_are_refused_as_the_picture_is() {
        let (_ram, memory) = memory();
        let stopped = configured();
        assert_eq!(
            stopped.pixel_turned_clockwise(&memory, 0, 0),
            Err(NoFrame::NotRunning)
        );
        let refusals: [(u64, u32); 4] = [
            (register::BLEND_BACKGROUND_COLOR, 1),
            (register::WIN_COLOR_DEPTH, 3),
            (register::WIN_POSITION, 3),
            (register::WIN_OPTIONS, WIN_ENABLE | 1 << 12),
        ];
        for (number, value) in refusals {
            let dc = configured();
            write(&dc, number, value);
            write(&dc, register::DISPLAY_COMMAND, DISPLAY_MODE_CONTINUOUS);
            write(&dc, register::STATE_CONTROL, 0x0F);
            assert!(dc.frame_turned_clockwise(&memory).is_err(), "{number:#x}");
            assert_points_are_the_picture(&dc, &memory, 1);
        }
    }
}
