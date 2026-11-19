//! The console's two controller rails and the Joy-Con that slide onto them.
//!
//! A Joy-Con on a rail talks to the console over that rail's UART (the
//! right one over UART-B, the left one over UART-C) and pulls the rail's
//! detect pin low. Its firmware is not run here; what it says on the wire
//! is modeled, from public descriptions of real sessions and of the
//! controller's commands, used for the facts:
//!
//! - the wired handshake, its checksums and the status request, with
//!   captured exchanges:
//!   <https://github.com/dekuNukem/Nintendo_Switch_Reverse_Engineering>
//!   (README, "Joy-Con to Console Communication");
//! - input reports, subcommands and the layout of the controller's flash
//!   memory: the same repository's `bluetooth_hid_notes.md`,
//!   `bluetooth_hid_subcommands_notes.md` and `spi_flash_notes.md`;
//! - names of the wired commands, which controller is on which UART and
//!   which pins detect them: the Joy-Con driver of an open-source
//!   bootloader (hekate, `bdk/input/joycon.c`);
//! - the right Joy-Con's NFC and infrared processor: see `joycon/mcu.rs`.
//!
//! The console's side of all of it is the firmware's own: `hid` sends the
//! wake bytes and the handshake at 1 000 000 baud on both rails as soon as
//! it sees a detect pin low (seen in an IPC trace of its `uart` session).
//!
//! What these two Joy-Con are is this model's choice, since there is no
//! real pair to copy: the constants of `Identity`. A request nothing here
//! answers is refused with a note, so the machine stops rather than go on
//! with a controller that says something a real one would not.

mod mcu;

use std::{
    sync::{
        Arc, Mutex, MutexGuard, PoisonError,
        atomic::{AtomicBool, AtomicU32, Ordering},
    },
    time::Duration,
};

use serde::{Deserialize, Serialize};

use super::{ntag::Ntag215, serial::Peer};
use crate::{
    clock::GuestClock,
    snapshot::{Snapshot, SnapshotError, decode, encode},
};

/// UART-B, which the right rail's serial lines lead to.
pub const RIGHT_UART_BASE: u64 = 0x7000_6040;
/// UART-C, likewise for the left rail.
pub const LEFT_UART_BASE: u64 = 0x7000_6200;
/// Interrupts of UART-B and UART-C (public Tegra210 device tree, SPI 37
/// and SPI 46).
pub const RIGHT_UART_INTID: u32 = 32 + 37;
pub const LEFT_UART_INTID: u32 = 32 + 46;
/// GPIO port and pin a Joy-Con on the rail pulls low: H6 for the right
/// rail, E6 for the left.
pub const RIGHT_DETECT: (usize, u8) = (7, 6);
pub const LEFT_DETECT: (usize, u8) = (4, 6);
/// The rails' CTS lines as GPIO pins: G3 for UART-B, D4 for UART-C
/// (Atmosphère's pad map, `ExtUart2Cts` and `ExtUart3Cts`). When the
/// console goes to sleep, `hid` puts each Joy-Con on a rail to sleep
/// (subcommand 6, state 0) and the GPIO driver makes both pins inputs with
/// any-edge interrupts, then masks every interrupt of their ports just
/// before the console sleeps (a 23.5.0 lock screen's auto-sleep, writes
/// to ports D and G, `0x6000d00c` and `0x6000d108` and their rows). What
/// wakes the console is then the PMC's wake event of each pad, enabled
/// with the level "auto" ([`super::pmc::WAKE_EVENT_UART2_CTS`]): a change
/// of the line.
///
/// The line is low while the Joy-Con asserts it, which here is whenever
/// one is on the rail, and pulled high on an empty rail \[assumption\].
/// A Joy-Con the console has put to sleep lets its line go high while
/// its HOME button is held, and only then ([`Controls::cts_high`]):
/// \[inference\] the line is the rail's only wake pad, HOME on a
/// controller is what wakes a sleeping console (Nintendo's support
/// pages), and its other buttons do not. How long a real Joy-Con holds
/// the line, and whether it pulses it, is not known.
pub const RIGHT_CTS: (usize, u8) = (6, 3);
pub const LEFT_CTS: (usize, u8) = (3, 4);

/// Which controller of the family this is: a Joy-Con of either rail, or
/// the Pro Controller, which is on no rail and speaks the same input
/// reports and subcommands over Bluetooth (`bluetooth_hid_notes.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Left,
    Right,
    Pro,
}

/// A button of the pair, as a bit of the three button bytes of a standard
/// input report taken as one little-endian number (right, shared, left).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Button(u32);

impl Button {
    pub const Y: Self = Self(1 << 0);
    pub const X: Self = Self(1 << 1);
    pub const B: Self = Self(1 << 2);
    pub const A: Self = Self(1 << 3);
    pub const R: Self = Self(1 << 6);
    pub const ZR: Self = Self(1 << 7);
    pub const MINUS: Self = Self(1 << 8);
    pub const PLUS: Self = Self(1 << 9);
    pub const RIGHT_STICK: Self = Self(1 << 10);
    pub const LEFT_STICK: Self = Self(1 << 11);
    pub const HOME: Self = Self(1 << 12);
    pub const CAPTURE: Self = Self(1 << 13);
    pub const DOWN: Self = Self(1 << 16);
    pub const UP: Self = Self(1 << 17);
    pub const RIGHT: Self = Self(1 << 18);
    pub const LEFT: Self = Self(1 << 19);
    pub const L: Self = Self(1 << 22);
    pub const ZL: Self = Self(1 << 23);

    /// Its bit (bits, for buttons held together) of a report's three
    /// button bytes, as [`Controls::held`] gives them.
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// This button and `other`, held together.
    pub const fn and(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// The button called `name` on the command line, in any case.
    pub fn named(name: &str) -> Option<Self> {
        Some(match name.to_ascii_lowercase().as_str() {
            "a" => Self::A,
            "b" => Self::B,
            "x" => Self::X,
            "y" => Self::Y,
            "l" => Self::L,
            "r" => Self::R,
            "zl" => Self::ZL,
            "zr" => Self::ZR,
            "plus" => Self::PLUS,
            "minus" => Self::MINUS,
            "home" => Self::HOME,
            "capture" => Self::CAPTURE,
            "up" => Self::UP,
            "down" => Self::DOWN,
            "left" => Self::LEFT,
            "right" => Self::RIGHT,
            "lstick" => Self::LEFT_STICK,
            "rstick" => Self::RIGHT_STICK,
            _ => return None,
        })
    }
}

/// The buttons each controller of the pair reports: its own byte and its
/// share of the middle one.
const LEFT_BUTTONS: u32 = 0xFF_0000 | Button::MINUS.0 | Button::LEFT_STICK.0 | Button::CAPTURE.0;
const RIGHT_BUTTONS: u32 = 0x00_00FF | Button::PLUS.0 | Button::RIGHT_STICK.0 | Button::HOME.0;
/// The Pro Controller has all of them but the rail buttons SL and SR.
const PRO_BUTTONS: u32 = (LEFT_BUTTONS | RIGHT_BUTTONS) & !0x30_0030;

/// What is being done to the pair of Joy-Con: where they are, which
/// buttons are held and which NFC tag is held over the right one's reader;
/// and which of them the console has put to sleep, which with the buttons
/// decides the level of each rail's CTS line ([`Controls::cts_high`]).
/// Shared between the rails and whoever is at the controls.
#[derive(Default)]
pub struct Controls {
    seated: AtomicBool,
    buttons: AtomicU32,
    /// Per controller ([`Side`] as an index): put to sleep by the console
    /// (subcommand 6, state 0); a handshake wakes it.
    asleep: [AtomicBool; 3],
    /// The right Joy-Con, asleep, has had HOME pressed since the console
    /// last asked why it woke (wired command 7).
    home_woke: AtomicBool,
    tag: Mutex<Option<Ntag215>>,
    /// Told the level of each rail's CTS line whenever it may have
    /// changed: the board wiring of those lines.
    cts: Mutex<Option<CtsLines>>,
}

/// The rails' CTS lines: the rail, and whether its line is high.
type CtsLines = Box<dyn Fn(Side, bool) + Send + Sync>;

impl Controls {
    /// Whether both Joy-Con are on their rails.
    pub fn seated(&self) -> bool {
        self.seated.load(Ordering::SeqCst)
    }

    pub(super) fn set_seated(&self, seated: bool) {
        if self.seated.swap(seated, Ordering::SeqCst) != seated {
            self.drive_cts();
        }
    }

    /// Holds or lets go of `button`.
    pub fn press(&self, button: Button, held: bool) {
        let before = match held {
            true => self.buttons.fetch_or(button.0, Ordering::SeqCst),
            false => self.buttons.fetch_and(!button.0, Ordering::SeqCst),
        };
        if (before & button.0 != 0) != held {
            if held && button.0 & Button::HOME.0 != 0 && self.asleep(Side::Right) {
                self.home_woke.store(true, Ordering::SeqCst);
            }
            self.drive_cts();
        }
    }

    /// The buttons held, as a mask of [`Button`] bits.
    pub fn held(&self) -> u32 {
        self.buttons.load(Ordering::SeqCst)
    }

    /// Holds `tag` over the right Joy-Con's NFC reader, or takes the one
    /// there away (`None`).
    pub fn place_tag(&self, tag: Option<Ntag215>) {
        *self.tag.lock().unwrap_or_else(PoisonError::into_inner) = tag;
    }

    fn tag(&self) -> MutexGuard<'_, Option<Ntag215>> {
        self.tag.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Whether the console has put the controller of `side` to sleep.
    fn asleep(&self, side: Side) -> bool {
        self.asleep[side.index()].load(Ordering::SeqCst)
    }

    fn set_asleep(&self, side: Side, asleep: bool) {
        if self.asleep[side.index()].swap(asleep, Ordering::SeqCst) != asleep {
            self.drive_cts();
        }
    }

    /// Wires the rails' CTS lines: `lines` is told each one's level now
    /// and whenever it may have changed.
    pub(super) fn connect_cts(&self, lines: CtsLines) {
        let mut cts = self.cts.lock().unwrap_or_else(PoisonError::into_inner);
        for side in [Side::Left, Side::Right] {
            lines(side, self.cts_high(side));
        }
        *cts = Some(lines);
    }

    /// Whether the CTS line of `side`'s rail is high: on an empty rail,
    /// and while the right Joy-Con, asleep, has its HOME button held (see
    /// [`RIGHT_CTS`]). The left Joy-Con has no HOME button.
    pub fn cts_high(&self, side: Side) -> bool {
        !self.seated()
            || side == Side::Right && self.asleep(side) && self.held() & Button::HOME.0 != 0
    }

    fn drive_cts(&self) {
        let cts = self.cts.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(lines) = cts.as_ref() {
            for side in [Side::Left, Side::Right] {
                lines(side, self.cts_high(side));
            }
        }
    }
}

/// Saved by the SoC ([`super::Soc::snapshot_parts`]): whether the pair is
/// on the rails, which buttons are held, which controllers sleep and the
/// tag on the reader, with what has been written to it, so that a restored
/// console has them where the saved one had them. The detect and CTS pins
/// they drive are GPIO inputs, which the GPIO saves, so a restore drives
/// nothing.
impl Snapshot for Controls {
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        let asleep = [Side::Left, Side::Right, Side::Pro].map(|side| self.asleep(side));
        let woke = self.home_woke.load(Ordering::SeqCst);
        encode(&(self.seated(), self.held(), asleep, woke, &*self.tag()))
    }

    fn restore(&self, state: &[u8]) -> Result<(), SnapshotError> {
        let (seated, buttons, asleep, woke, tag): (bool, u32, [bool; 3], bool, Option<Ntag215>) =
            decode("Joy-Con controls", state)?;
        self.seated.store(seated, Ordering::SeqCst);
        self.buttons.store(buttons, Ordering::SeqCst);
        for (flag, asleep) in self.asleep.iter().zip(asleep) {
            flag.store(asleep, Ordering::SeqCst);
        }
        self.home_woke.store(woke, Ordering::SeqCst);
        self.place_tag(tag);
        Ok(())
    }
}

/// What makes these two Joy-Con the ones they are. None of it is copied
/// from a real pair: it is what a pair fresh from the box is documented to
/// hold, with the per-unit values chosen here.
struct Identity;

impl Identity {
    /// Bluetooth addresses, most significant byte first: locally
    /// administered, next to the console's own virtual addresses.
    const ADDRESS: [[u8; 6]; 3] = [
        [0x02, 0x68, 0x45, 0x6D, 0x75, 0x10],
        [0x02, 0x68, 0x45, 0x6D, 0x75, 0x11],
        [0x02, 0x68, 0x45, 0x6D, 0x75, 0x12],
    ];
    /// Firmware version reported by "request device info": 4.32 for the
    /// Joy-Con and 4.33 for the Pro Controller, the ones this console's
    /// system software carries (`FirmwareInfo.csv` of system data
    /// 0100000000000822: `JoyLeft,BT,0420`, `FullKey,BT,0421`), so that
    /// there is nothing to update.
    const FIRMWARE: [[u8; 2]; 3] = [[0x04, 0x20], [0x04, 0x20], [0x04, 0x21]];
    /// Version of the right Joy-Con's NFC and infrared processor's
    /// firmware, likewise (`JoyRight,MCU,0920`).
    const PROCESSOR_FIRMWARE: [u16; 2] = [0x09, 0x20];
    /// The Pro Controller's NFC processor's firmware (`FullKey,MCU,0403`).
    const PRO_PROCESSOR_FIRMWARE: [u16; 2] = [0x04, 0x03];
    /// Bluetooth class of device (peripheral, gamepad) and names, as the
    /// controllers give them when pairing.
    const CLASS: [u8; 3] = [0x08, 0x25, 0x00];
    const NAME: [&'static str; 3] = ["Joy-Con (L)", "Joy-Con (R)", "Pro Controller"];
    /// The link keys the pair would use over Bluetooth once off the
    /// rails. Anything will do for a key no radio ever uses; these say
    /// what they are.
    const LINK_KEY: [&'static [u8; 16]; 2] = [b"virtual Joy-ConL", b"virtual Joy-ConR"];
    /// Battery level nibble: full, not charging.
    const BATTERY: u8 = 8;
    /// Body and button colours of the grey Joy-Con.
    const BODY: [u8; 3] = [0x82, 0x82, 0x82];
    const KEYS: [u8; 3] = [0x0F, 0x0F, 0x0F];
    /// Stick calibration: centre of the 12-bit range, and a travel either
    /// way a little above what the notes report the console insists on
    /// for a Pro Controller (`0x542`).
    const STICK_CENTER: u16 = 0x800;
    const STICK_TRAVEL: u16 = 0x600;
    /// Motion sensor calibration: no offset, and the sensitivity
    /// coefficients every documented unit has.
    const ACCELEROMETER_SCALE: i16 = 0x4000;
    const GYROSCOPE_SCALE: i16 = 0x343B;
    /// What the accelerometer reads with the controller lying flat: the
    /// documented default "horizontal offsets" of each kind, which is
    /// where the console expects a Joy-Con at rest on its rail to be.
    const AT_REST: [[i16; 3]; 3] = [[350, 0, 4081], [350, 0, -4081], [-688, 0, 4038]];
    /// Stick device parameters, the documented sample (dead zone `0xAE`).
    const STICK_PARAMETERS: [u16; 12] = [
        0x019, 0x4CD, 0x0AE, 0xE14, 0x2EE, 0x2EE, 0x2EE, 0x2EE, 0xAB4, 0xAB4, 0x496, 0x496,
    ];
}

/// Packs 12-bit numbers two to three bytes, as stick data is everywhere
/// in the protocol.
fn pack12(values: &[u16]) -> Vec<u8> {
    values
        .chunks(2)
        .flat_map(|pair| {
            let (a, b) = (pair[0], pair[1]);
            [
                a as u8,
                (a >> 8) as u8 & 0xF | (b as u8) << 4,
                (b >> 4) as u8,
            ]
        })
        .collect()
}

fn words(values: &[i16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

impl Side {
    fn index(self) -> usize {
        self as usize
    }

    /// Device type: 1 for a left Joy-Con, 2 for a right one, 3 for a Pro
    /// Controller.
    fn kind(self) -> u8 {
        self as u8 + 1
    }

    /// The factory-programmed parts of the controller's flash memory, by
    /// address (`spi_flash_notes.md`).
    fn flash(self) -> Vec<(u32, Vec<u8>)> {
        let (center, travel) = (Identity::STICK_CENTER, Identity::STICK_TRAVEL);
        // Each Joy-Con has one stick; the other's calibration is not
        // programmed. The Pro Controller has both.
        let stick = |values: [u16; 6], present: bool| match present {
            true => pack12(&values),
            false => vec![0xFF; 9],
        };
        let mut second = stick(
            [travel, travel, center, center, travel, travel],
            self != Side::Right,
        );
        second.extend(stick(
            [center, center, travel, travel, travel, travel],
            self != Side::Left,
        ));
        second.push(0xFF);
        second.extend(Identity::BODY);
        second.extend(Identity::KEYS);
        // Grip colours: a Pro Controller's.
        second.extend([0xFF; 6]);

        let scale = |scale: i16| [0, 0, 0, scale, scale, scale];
        let mut first = words(&scale(Identity::ACCELEROMETER_SCALE));
        first.extend(words(&scale(Identity::GYROSCOPE_SCALE)));

        let mut parameters = words(&Identity::AT_REST[self.index()]);
        parameters.extend(pack12(&Identity::STICK_PARAMETERS));
        parameters.extend(pack12(&Identity::STICK_PARAMETERS));

        // No serial number: a first byte of 0x80 or more says so.
        let mut header = vec![0xFF; 0x20];
        header[0x12] = self.kind();
        header[0x13] = 0xA0;
        // Colours are programmed.
        header[0x1B] = 0x01;
        vec![
            (0x6000, header),
            (0x6020, first),
            (0x603D, second),
            (0x6080, parameters),
        ]
    }

    /// `size` bytes of the flash memory at `address`. Outside the
    /// configuration sections, which read as erased where nothing is
    /// programmed, nothing is modeled.
    fn read_flash(self, address: u32, size: u8) -> Option<Vec<u8>> {
        // Pairing information, shipment flag, factory configuration and
        // user calibration.
        const SECTIONS: std::ops::Range<u32> = 0x2000..0x1_0000;
        let end = address.checked_add(size.into())?;
        if address < SECTIONS.start || end > SECTIONS.end {
            return None;
        }
        let programmed = self.flash();
        Some(
            (address..end)
                .map(|at| {
                    programmed
                        .iter()
                        .find_map(|(start, bytes)| bytes.get(at.checked_sub(*start)? as usize))
                        .copied()
                        .unwrap_or(0xFF)
                })
                .collect(),
        )
    }
}

/// CRC-8 with polynomial `0x8D`, as the wired protocol checks its headers
/// and payloads.
fn crc8(data: &[u8]) -> u8 {
    data.iter().fold(0, |crc, byte| {
        (0..8).fold(crc ^ byte, |crc, _| match crc & 0x80 {
            0 => crc << 1,
            _ => crc << 1 ^ 0x8D,
        })
    })
}

/// First bytes of a packet from the console, and of one to it.
const TO_JOYCON: [u8; 3] = [0x19, 0x01, 0x03];
const TO_CONSOLE: [u8; 3] = [0x19, 0x81, 0x03];
/// The packet that starts a wired session, after the wake bytes, and the
/// answer to it (captured exchange).
const HANDSHAKE: [u8; 7] = [0xA5, 0x02, 0x01, 0x7E, 0x00, 0x00, 0x00];
const HANDSHAKE_ANSWER: [u8; 7] = [0xA5, 0x02, 0x02, 0x7D, 0x00, 0x00, 0x64];
/// Wired commands, their answers, and the envelope of controller reports.
const COMMAND: u8 = 0x91;
const COMMAND_ANSWER: u8 = 0x94;
const REPORT: u8 = 0x92;
/// Bytes of a packet before its payload: command, subcommand, payload
/// size, status and the two checksums.
const ENVELOPE: usize = 7;
/// A controller report in a wired packet: `0x31` bytes; one of kind
/// `0x31` carries the NFC and infrared processor's answer after that.
const REPORT_SIZE: usize = 49;
const MCU_REPORT_SIZE: usize = REPORT_SIZE + mcu::ANSWER_SIZE;
/// How many answered packets a Joy-Con remembers for its refusals.
const ANSWERED_KEPT: usize = 4;

/// A packet to the console.
fn packet(command: u8, subcommand: u8, status: u8, payload: &[u8]) -> Vec<u8> {
    let size = (payload.len() as u16).to_le_bytes();
    let mut envelope = vec![command, subcommand, size[0], size[1], status, crc8(payload)];
    envelope.push(crc8(&envelope));
    let total = ((envelope.len() + payload.len()) as u16).to_le_bytes();
    let mut packet = TO_CONSOLE.to_vec();
    packet.extend(total);
    packet.extend(envelope);
    packet.extend(payload);
    packet
}

/// One Joy-Con, as far as its rail connector shows; or the Pro Controller,
/// whose input reports and subcommands are the same, carried over
/// Bluetooth (`super::bluetooth_uart`).
pub(super) struct JoyCon {
    side: Side,
    controls: Arc<Controls>,
    clock: Arc<GuestClock>,
    session: Session,
    /// What a refusal of the packet being answered is to add.
    why: Option<String>,
}

/// What the Joy-Con holds of its session with the console: everything a
/// snapshot saves of it.
#[derive(Serialize, Deserialize)]
struct Session {
    /// The packet being received.
    incoming: Vec<u8>,
    /// The last few packets answered, a run of the same one counted once:
    /// what a refusal says the refused packet came after.
    answered: Vec<Vec<u8>>,
    /// The console has opened the wired session for controller reports.
    reporting: bool,
    motion: bool,
    vibration: bool,
    /// Input reports of kind `0x31` (subcommand 3), which carry the NFC
    /// and infrared processor's answers, rather than `0x30`.
    processor_reports: bool,
    /// The right Joy-Con's NFC and infrared processor.
    processor: mcu::Mcu,
    /// Guest time at which it was powered, which its report timer counts
    /// from.
    powered: Duration,
}

impl Session {
    fn new(now: Duration) -> Self {
        Self {
            incoming: Vec::new(),
            answered: Vec::new(),
            reporting: false,
            motion: false,
            vibration: false,
            processor_reports: false,
            processor: mcu::Mcu::new(),
            powered: now,
        }
    }
}

impl JoyCon {
    fn new(side: Side, controls: Arc<Controls>, clock: Arc<GuestClock>) -> Self {
        Self {
            side,
            controls,
            session: Session::new(clock.now()),
            clock,
            why: None,
        }
    }

    /// The Pro Controller, whose buttons are held on `controls` and whose
    /// report timer counts guest time from `clock`.
    pub(super) fn pro(controls: Arc<Controls>, clock: Arc<GuestClock>) -> Self {
        Self::new(Side::Pro, controls, clock)
    }

    /// The session ends when the Joy-Con leaves the rail, or when the Pro
    /// Controller's link drops; and it no longer sleeps.
    pub(super) fn unplug(&mut self) {
        self.session = Session::new(self.clock.now());
        self.controls.set_asleep(self.side, false);
    }

    /// The session, encoded: what a snapshot saves of the controller.
    pub(super) fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        encode(&self.session)
    }

    pub(super) fn restore(&mut self, state: &[u8]) -> Result<(), SnapshotError> {
        self.session = decode("Joy-Con", state)?;
        Ok(())
    }

    fn refuse<T>(&self, what: impl std::fmt::Display) -> Result<T, String> {
        Err(format!("{:?} Joy-Con: {what} is not modeled", self.side))
    }

    fn byte(&mut self, byte: u8) -> Result<Vec<u8>, String> {
        // Between packets only the start of one means anything: the wake
        // bytes, `A1 A2 A3 A4`, are not part of any.
        if self.session.incoming.is_empty() && byte != TO_JOYCON[0] {
            return Ok(Vec::new());
        }
        self.session.incoming.push(byte);
        let Some(header) = self.session.incoming.first_chunk::<5>() else {
            return Ok(Vec::new());
        };
        let length = usize::from(u16::from_le_bytes([header[3], header[4]]));
        if self.session.incoming.len() < 5 + length {
            return Ok(Vec::new());
        }
        let received = std::mem::take(&mut self.session.incoming);
        let refusal = |joycon: &mut Self| {
            let why = joycon.why.take().map(|why| format!(" ({why})"));
            joycon.refuse(format_args!(
                "the packet {received:02x?}, after {:02x?},{}",
                joycon.session.answered,
                why.unwrap_or_default(),
            ))
        };
        if received[..3] != TO_JOYCON || length < ENVELOPE {
            return refusal(self);
        }
        let answered = std::mem::take(&mut self.session.answered);
        let answer = self.packet(&received[5..]).or_else(|()| {
            self.session.answered = answered.clone();
            refusal(self)
        })?;
        // The list outlasts a handshake that starts the Joy-Con over.
        self.session.answered = answered;
        let kept = &mut self.session.answered;
        if kept.last() != Some(&received) {
            kept.push(received);
            let excess = kept.len().saturating_sub(ANSWERED_KEPT);
            kept.drain(..excess);
        }
        Ok(answer)
    }

    /// Answers the packet whose bytes after the header are `body`. `Err`
    /// if it is one nothing is known about.
    fn packet(&mut self, body: &[u8]) -> Result<Vec<u8>, ()> {
        let (envelope, payload) = body.split_at(ENVELOPE);
        if envelope == HANDSHAKE && payload.is_empty() {
            // A sleeping Joy-Con starts over, as one just slid on.
            if self.controls.asleep(self.side) {
                self.unplug();
            }
            let mut answer = TO_CONSOLE.to_vec();
            answer.extend([ENVELOPE as u8, 0]);
            answer.extend(HANDSHAKE_ANSWER);
            return Ok(answer);
        }
        // Asleep, it listens for nothing else [assumption].
        if self.controls.asleep(self.side) {
            return Ok(Vec::new());
        }
        let [
            command,
            subcommand,
            size_low,
            size_high,
            status,
            payload_crc,
            crc,
        ] = *envelope.first_chunk().ok_or(())?;
        // hekate sends its reports to a Joy-Con with both checksums zero
        // and notes that nothing checks them (`joycon.c`
        // `_jc_packet_add_uart_hdr`; only its packets to a HORI pad carry
        // them), and its driver works on consoles. \[inference: a Joy-Con
        // takes a report whose checksums are zero; what it does with other
        // wrong ones is not known, so those are refused\]
        let unchecked = command == REPORT && payload_crc == 0 && crc == 0;
        let checked = payload_crc == crc8(payload) && crc == crc8(&envelope[..ENVELOPE - 1]);
        let sound = usize::from(u16::from_le_bytes([size_low, size_high])) == payload.len()
            && status == 0
            && (checked || unchecked);
        if !sound {
            return Err(());
        }
        match (command, subcommand, payload) {
            // Who are you: kind and address, least significant byte first.
            (COMMAND, 0x01, []) => {
                let mut info = vec![self.side.kind()];
                info.extend(Identity::ADDRESS[self.side.index()].iter().rev());
                info.push(0);
                Ok(packet(COMMAND_ANSWER, 0x01, 0, &info))
            }
            // Change of baud rate, to 3 000 000: the line here has no
            // rate, so there is nothing to change.
            (COMMAND, 0x20, [0xC0, 0xC6, 0x2D, 0, 0, 0, 0, 0]) => {
                Ok(packet(COMMAND_ANSWER, 0x20, 0, &[]))
            }
            // End of controller reports. The captured answer has status
            // 0x0F.
            (COMMAND, 0x11, []) => {
                self.session.reporting = false;
                Ok(packet(COMMAND_ANSWER, 0x11, 0x0F, &[]))
            }
            (COMMAND, 0x10, []) => {
                self.session.reporting = true;
                Ok(packet(COMMAND_ANSWER, 0x10, 0, &[]))
            }
            // Interval of reports in milliseconds; they are sent when asked
            // for. The bootloader's driver knows of 5, 10 and 15.
            (COMMAND, 0x12, [5 | 10 | 15, 0, 0, 0]) => Ok(packet(COMMAND_ANSWER, 0x12, 0, &[])),
            // Why the Joy-Con woke the console (hekate `joycon.c` names
            // command 7 "wake reason"). `omm` asks through `hid` after a
            // wake by the right rail's CTS pad, and only payload byte 0
            // equal to 1 turns the screen on; anything else sends the
            // console back to sleep. `hid` reads 4 payload bytes and takes
            // bytes 2 and 3 as a further pair it passes on, ignoring byte 1
            // and the status (hid 23.5.0 text `0x173200`..`0x17322c`; omm
            // `0x7FBB0`, `0x80CE0`). \[inference\] 1 is a HOME press, which
            // is how this model wakes the console ([`RIGHT_CTS`]); 0 says
            // the Joy-Con did not; the other bytes zero, the value `hid`
            // takes as nothing to report.
            (COMMAND, 0x07, []) if self.side == Side::Right => {
                let home = self.controls.home_woke.swap(false, Ordering::SeqCst);
                Ok(packet(COMMAND_ANSWER, 0x07, 0, &[u8::from(home), 0, 0, 0]))
            }
            // How the Joy-Con is to charge from the rail: the supply and
            // current settings the bootloader's driver names. `hid` sends
            // `0x14` after learning who is there and waits for an answer
            // with this subcommand, of which it reads nothing (text
            // `0x172F80`); that the answer is an empty one with status 0
            // is an assumption.
            (COMMAND, 0x02, [0x00 | 0x04 | 0x10 | 0x14, 0, 0, 0]) => {
                Ok(packet(COMMAND_ANSWER, 0x02, 0, &[]))
            }
            (REPORT, 0x00, output) if self.session.reporting => {
                let report = self.report(output)?;
                Ok(packet(REPORT, 0x00, 0, &report))
            }
            // With no session there is nobody to report to. `hid` ends
            // the session (`91 11`) when the console goes back to sleep
            // after a wake that never lit the screen, and what polls the
            // Joy-Con still sends a request for its state after that.
            // That a Joy-Con says nothing to one, or to vibration data, is
            // an assumption; a subcommand without a session is refused.
            (REPORT, 0x00, [0x1F] | [0x10, ..]) => Ok(Vec::new()),
            _ => Err(()),
        }
    }

    /// The input report that answers output report `output`.
    pub(super) fn report(&mut self, output: &[u8]) -> Result<Vec<u8>, ()> {
        let side = self.side.index();
        let own = [LEFT_BUTTONS, RIGHT_BUTTONS, PRO_BUTTONS][side];
        let buttons = self.controls.buttons.load(Ordering::SeqCst) & own;
        // The stick is at its centre; the other side's is not there.
        let stick = pack12(&[Identity::STICK_CENTER, Identity::STICK_CENTER]);
        let sticks = match self.side {
            Side::Left => [stick, vec![0; 3]],
            Side::Right => [vec![0; 3], stick],
            Side::Pro => [stick.clone(), stick],
        };
        // Per side: vibration enabled, and room for more of it. Seen in
        // the notes' captures: 7 while off, 0xC while on. The Pro
        // Controller's is taken to be in the right Joy-Con's place
        // \[assumption\].
        let vibrator = match self.session.vibration {
            true => 0xC,
            false => 0x7,
        } << [4, 0, 0][side];
        // In units of 5 ms of guest time.
        let powered_for = self.clock.now().saturating_sub(self.session.powered);
        let timer = (powered_for.as_millis() / 5) as u8;
        let mut report = vec![0, timer, Identity::BATTERY << 4];
        report.extend(&buttons.to_le_bytes()[..3]);
        report.extend(sticks.concat());
        report.push(vibrator);
        match output {
            // A subcommand, after the packet counter and the vibration
            // data.
            [0x01, _, _, _, _, _, _, _, _, _, subcommand, arguments @ ..] => {
                report[0] = 0x21;
                let (acknowledge, data) = self.subcommand(*subcommand, arguments)?;
                report.extend([acknowledge, *subcommand]);
                report.extend(data);
            }
            // Vibration data only, the bare request for the state, and a
            // request to the NFC and infrared processor after the
            // vibration data.
            [0x10, _, _, _, _, _, _, _, _, _]
            | [0x1F]
            | [0x11, _, _, _, _, _, _, _, _, _, _, ..] => {
                report[0] = 0x30;
                // Three samples of the motion sensor: at rest.
                let sample = match self.session.motion {
                    true => {
                        let mut sample = words(&Identity::AT_REST[side]);
                        sample.extend([0; 6]);
                        sample
                    }
                    false => vec![0; 12],
                };
                report.extend(sample.repeat(3));
                if let [0x11, _, _, _, _, _, _, _, _, _, request @ ..] = output {
                    if self.side != Side::Right || !self.session.processor_reports {
                        return Err(());
                    }
                    let mut tag = self.controls.tag();
                    let now = self.clock.now();
                    if let Err(why) = self.session.processor.request(request, &mut tag, now) {
                        self.why = Some(why);
                        return Err(());
                    }
                }
                if self.session.processor_reports {
                    report[0] = 0x31;
                    let tag = self.controls.tag();
                    let answer = self.session.processor.answer(
                        tag.as_ref(),
                        Identity::PROCESSOR_FIRMWARE,
                        self.clock.now(),
                    );
                    report.extend(answer);
                    debug_assert_eq!(report.len(), MCU_REPORT_SIZE);
                    return Ok(report);
                }
            }
            _ => return Err(()),
        }
        if report.len() > REPORT_SIZE {
            return Err(());
        }
        report.resize(REPORT_SIZE, 0);
        Ok(report)
    }

    /// Carries out a subcommand: the acknowledgement byte and the data of
    /// the answer (`bluetooth_hid_subcommands_notes.md`).
    fn subcommand(&mut self, subcommand: u8, arguments: &[u8]) -> Result<(u8, Vec<u8>), ()> {
        // Arguments come in a field of fixed size; what follows the ones
        // a subcommand has is padding.
        let argument = |index: usize| arguments.get(index).copied().ok_or(());
        const DONE: (u8, Vec<u8>) = (0x80, Vec::new());
        Ok(match subcommand {
            // Pairing over the rail, in three steps, so that the Joy-Con
            // and the console know each other over Bluetooth afterwards
            // (the Pro Controller, on no rail, pairs over the air).
            // `hid` (text `0x14F480`) takes from the first answer the
            // address, three bytes and 21 more, which are the class and
            // name it sent of itself in the same places; from the second
            // the link key, each byte XORed with 0xAA.
            //
            // A console that already has this Joy-Con in its list opens
            // with step 4 instead of 1 (`0x14C360`: for a known address
            // and firmware above 3.3), which asks whether the pairing
            // still stands. This Joy-Con keeps nothing from one run of
            // the machine to the next, the console's storage does: it has
            // no pairing to confirm. What a real one says then is not
            // documented; `hid` takes an answer of kind 1 to either
            // opening as the start of a fresh pairing and goes on with
            // steps 2 and 3, and that is the answer given, which is an
            // assumption.
            0x01 if self.side != Side::Pro => {
                let step = argument(0)?;
                let side = self.side.index();
                let data = match step {
                    1 | 4 => {
                        let mut data = vec![1];
                        data.extend(Identity::ADDRESS[side].iter().rev());
                        data.extend(Identity::CLASS);
                        data.extend(Identity::NAME[side].bytes());
                        data
                    }
                    2 => {
                        let mut data = vec![2];
                        data.extend(Identity::LINK_KEY[side].map(|byte| byte ^ 0xAA));
                        data
                    }
                    3 => vec![3],
                    _ => return Err(()),
                };
                (0x81, data)
            }
            // Device information.
            0x02 => {
                let mut info = Identity::FIRMWARE[self.side.index()].to_vec();
                info.extend([self.side.kind(), 0x02]);
                info.extend(Identity::ADDRESS[self.side.index()]);
                // Colours come from the flash memory.
                info.extend([0x01, 0x01]);
                (0x82, info)
            }
            // Input report mode: standard full reports, or (on the right
            // Joy-Con) those with the NFC and infrared processor's answers.
            0x03 if argument(0)? == 0x30 => {
                self.session.processor_reports = false;
                DONE
            }
            0x03 if argument(0)? == 0x31 && self.side == Side::Right => {
                self.session.processor_reports = true;
                DONE
            }
            // How long the trigger buttons have been held: not at all.
            0x04 => (0x83, vec![0; 14]),
            // Leave the low power state of shipment.
            0x08 if argument(0)? == 0 => DONE,
            // Power state (`bluetooth_hid_subcommands_notes.md`, "Set HCI
            // state"): 0 is sleep, which `hid` sends each Joy-Con on the
            // rails when the console goes to sleep. The bootloader's
            // driver reads a packet after sending it (hekate `joycon.c`),
            // so it is acknowledged first; that the acknowledgement is the
            // plain one, and that only a new handshake wakes the Joy-Con,
            // are assumptions. Asleep, the right one raises its rail's CTS
            // line while HOME is held, which wakes the console
            // ([`RIGHT_CTS`]). The other states (reboot into reconnecting
            // or pairing over Bluetooth) are not modeled.
            0x06 if argument(0)? == 0 => {
                self.controls.set_asleep(self.side, true);
                DONE
            }
            // Flash memory read: address, size.
            0x10 => {
                let request = arguments.first_chunk::<5>().ok_or(())?;
                let address = u32::from_le_bytes([request[0], request[1], request[2], request[3]]);
                let size = request[4];
                if size > 0x1D {
                    return Err(());
                }
                let mut data = request.to_vec();
                data.extend(self.side.read_flash(address, size).ok_or(())?);
                (0x90, data)
            }
            // The right Joy-Con's NFC and infrared processor: suspend (0),
            // resume (1).
            0x22 if self.side == Side::Right && argument(0)? <= 1 => {
                self.session.processor.power(argument(0)? == 1);
                DONE
            }
            // A configuration request for that processor, answered with
            // its 34-byte state (`hid` checks it at `0x153C60`). The Pro
            // Controller has the same processor for NFC (`FullKey,MCU` in
            // `FirmwareInfo.csv`); `hid` asks it for its state without
            // resuming it first, so it is taken to be in standby from the
            // start \[assumption\].
            0x21 if self.side != Side::Left => {
                let firmware = match self.side {
                    Side::Pro => {
                        if !self.session.processor.powered() {
                            self.session.processor.power(true);
                        }
                        Identity::PRO_PROCESSOR_FIRMWARE
                    }
                    _ => Identity::PROCESSOR_FIRMWARE,
                };
                let state = self
                    .session
                    .processor
                    .configure(arguments, firmware)
                    .ok_or(())?;
                (0xA0, state)
            }
            // Player lights; the HOME button's light on the right Joy-Con
            // and the Pro Controller.
            0x30 => DONE,
            0x38 if self.side != Side::Left => DONE,
            // Motion sensor on or off.
            0x40 => {
                self.session.motion = argument(0)? != 0;
                DONE
            }
            // Motion sensor sensitivity and filters: the samples here are
            // of a Joy-Con at rest whatever they are.
            0x41 => DONE,
            // Vibration on or off.
            0x48 => {
                self.session.vibration = argument(0)? != 0;
                DONE
            }
            // Battery voltage, in units of 2.5 mV: 4.0 V, within the range
            // the notes give for the "full" the reports claim.
            0x50 => (0xD0, 1600u16.to_le_bytes().to_vec()),
            // How the Joy-Con charges from the rail, with the settings of
            // wired command 2 (hekate `joycon.c` "charge set"; the notes
            // call it two output pins). hekate waits for an answer with
            // this subcommand and reads no more of it; that the
            // acknowledgement is the plain one is an assumption.
            0x51 if matches!(argument(0)?, 0x00 | 0x04 | 0x10 | 0x14) => DONE,
            _ => return Err(()),
        })
    }
}

/// The serial lines of one rail, with the Joy-Con that goes on it.
pub struct Rail {
    joycon: JoyCon,
}

impl Rail {
    /// The Joy-Con's report timer counts guest time from `clock`.
    pub fn new(side: Side, controls: Arc<Controls>, clock: Arc<GuestClock>) -> Self {
        Self {
            joycon: JoyCon::new(side, controls, clock),
        }
    }
}

impl Peer for Rail {
    fn byte(&mut self, byte: u8) -> Result<Vec<u8>, String> {
        match self.joycon.controls.seated() {
            true => self.joycon.byte(byte),
            // With nothing on the rail the transmit line leads nowhere.
            false => {
                self.joycon.unplug();
                Ok(Vec::new())
            }
        }
    }

    /// The Joy-Con asserts CTS while its line is low ([`Controls::cts_high`]).
    fn clear_to_send(&self) -> bool {
        !self.joycon.controls.cts_high(self.joycon.side)
    }

    /// The Joy-Con's session; whether it is on the rail, and whether it
    /// sleeps, is the [`Controls`]', which the SoC saves.
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        encode(&self.joycon.session)
    }

    fn restore(&mut self, state: &[u8]) -> Result<(), SnapshotError> {
        self.joycon.session = decode("Joy-Con", state)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seated(side: Side) -> (Rail, Arc<Controls>) {
        seated_on(side, Arc::default())
    }

    fn seated_on(side: Side, clock: Arc<GuestClock>) -> (Rail, Arc<Controls>) {
        let controls = Arc::new(Controls::default());
        controls.set_seated(true);
        (Rail::new(side, Arc::clone(&controls), clock), controls)
    }

    fn send(rail: &mut Rail, bytes: &[u8]) -> Vec<u8> {
        bytes
            .iter()
            .flat_map(|&byte| rail.byte(byte).unwrap())
            .collect()
    }

    /// The exchanges captured from a real left Joy-Con, as published; the
    /// address in the second answer is that unit's.
    #[test]
    fn the_published_session_checks_out() {
        // Checksums of the published packets, console's and Joy-Con's.
        assert_eq!(crc8(&[0x91, 0x01, 0, 0, 0, 0]), 0x24);
        assert_eq!(crc8(&[0xC0, 0xC6, 0x2D, 0, 0, 0, 0, 0]), 0xBD);
        assert_eq!(crc8(&[0x91, 0x20, 0x08, 0, 0, 0xBD]), 0xB1);
        assert_eq!(
            packet(
                COMMAND_ANSWER,
                0x01,
                0,
                &[0x01, 0x31, 0x67, 0x9C, 0x8A, 0xBB, 0x7C, 0x00]
            ),
            [
                0x19, 0x81, 0x03, 0x0F, 0x00, 0x94, 0x01, 0x08, 0x00, 0x00, 0xFA, 0xE8, 0x01, 0x31,
                0x67, 0x9C, 0x8A, 0xBB, 0x7C, 0x00
            ]
        );

        let (mut rail, _) = seated(Side::Left);
        let handshake = [
            0xA1, 0xA2, 0xA3, 0xA4, 0x19, 0x01, 0x03, 0x07, 0x00, 0xA5, 0x02, 0x01, 0x7E, 0x00,
            0x00, 0x00,
        ];
        assert_eq!(
            send(&mut rail, &handshake),
            [
                0x19, 0x81, 0x03, 0x07, 0x00, 0xA5, 0x02, 0x02, 0x7D, 0x00, 0x00, 0x64
            ]
        );
        let identity = send(
            &mut rail,
            &[
                0x19, 0x01, 0x03, 0x07, 0x00, 0x91, 0x01, 0x00, 0x00, 0x00, 0x00, 0x24,
            ],
        );
        assert_eq!(
            identity[..10],
            [0x19, 0x81, 0x03, 0x0F, 0x00, 0x94, 0x01, 0x08, 0x00, 0x00]
        );
        assert_eq!(
            identity[12..],
            [0x01, 0x10, 0x75, 0x6D, 0x45, 0x68, 0x02, 0x00]
        );
        let exchanges: [(&[u8], &[u8]); 4] = [
            (
                &[
                    0x19, 0x01, 0x03, 0x0F, 0x00, 0x91, 0x20, 0x08, 0x00, 0x00, 0xBD, 0xB1, 0xC0,
                    0xC6, 0x2D, 0x00, 0x00, 0x00, 0x00, 0x00,
                ],
                &[
                    0x19, 0x81, 0x03, 0x07, 0x00, 0x94, 0x20, 0x00, 0x00, 0x00, 0x00, 0xA8,
                ],
            ),
            (
                &[
                    0x19, 0x01, 0x03, 0x07, 0x00, 0x91, 0x11, 0x00, 0x00, 0x00, 0x00, 0x0E,
                ],
                &[
                    0x19, 0x81, 0x03, 0x07, 0x00, 0x94, 0x11, 0x00, 0x00, 0x0F, 0x00, 0x33,
                ],
            ),
            (
                &[
                    0x19, 0x01, 0x03, 0x07, 0x00, 0x91, 0x10, 0x00, 0x00, 0x00, 0x00, 0x3D,
                ],
                &[
                    0x19, 0x81, 0x03, 0x07, 0x00, 0x94, 0x10, 0x00, 0x00, 0x00, 0x00, 0xD6,
                ],
            ),
            (
                &[
                    0x19, 0x01, 0x03, 0x0B, 0x00, 0x91, 0x12, 0x04, 0x00, 0x00, 0x12, 0xA6, 0x0F,
                    0x00, 0x00, 0x00,
                ],
                &[
                    0x19, 0x81, 0x03, 0x07, 0x00, 0x94, 0x12, 0x00, 0x00, 0x00, 0x00, 0xB0,
                ],
            ),
        ];
        for (request, answer) in exchanges {
            assert_eq!(send(&mut rail, request), answer);
        }
    }

    fn connected(side: Side) -> (Rail, Arc<Controls>) {
        let (mut rail, controls) = seated(side);
        open_session(&mut rail);
        (rail, controls)
    }

    fn open_session(rail: &mut Rail) {
        send(
            rail,
            &[
                0x19, 0x01, 0x03, 0x07, 0x00, 0x91, 0x10, 0x00, 0x00, 0x00, 0x00, 0x3D,
            ],
        );
    }

    /// A restored Joy-Con is in the session the saved one was in, and its
    /// report timer reads the same at the same guest time.
    #[test]
    fn a_restored_session_reports_with_the_timer_of_the_saved_one() {
        let paused_clock = || {
            let clock = Arc::new(GuestClock::new());
            clock.pause();
            clock.set(Duration::ZERO);
            clock
        };
        let (saved_clock, restored_clock) = (paused_clock(), paused_clock());
        let (mut saved, _) = seated_on(Side::Right, Arc::clone(&saved_clock));
        open_session(&mut saved);
        saved_clock.set(Duration::from_secs(1));
        restored_clock.set(Duration::from_secs(1));
        let (mut restored, _) = seated_on(Side::Right, Arc::clone(&restored_clock));
        restored.restore(&saved.save().unwrap()).unwrap();
        for at in [1_000, 1_500] {
            saved_clock.set(Duration::from_millis(at));
            restored_clock.set(Duration::from_millis(at));
            let answer = send(&mut restored, &STATUS);
            assert_eq!(answer[12 + 1], (at / 5) as u8, "the timer counts 5 ms");
            assert_eq!(answer, send(&mut saved, &STATUS));
        }
    }

    /// The published status request, and the report that answers it.
    const STATUS: [u8; 13] = [
        0x19, 0x01, 0x03, 0x08, 0x00, 0x92, 0x00, 0x01, 0x00, 0x00, 0x69, 0x2D, 0x1F,
    ];

    #[test]
    fn each_side_reports_its_own_buttons_and_stick() {
        for (side, own, other, stick) in [
            (Side::Left, Button::UP, Button::A, 6),
            (Side::Right, Button::A, Button::UP, 9),
        ] {
            let (mut rail, controls) = connected(side);
            controls.press(own, true);
            controls.press(other, true);
            let answer = send(&mut rail, &STATUS);
            assert_eq!(
                answer[..10],
                [0x19, 0x81, 0x03, 0x38, 0x00, 0x92, 0x00, 0x31, 0x00, 0x00]
            );
            let report = &answer[12..];
            assert_eq!(report.len(), REPORT_SIZE);
            assert_eq!(report[0], 0x30);
            assert_eq!(report[2], 0x80);
            assert_eq!(report[3..6], own.0.to_le_bytes()[..3]);
            // Centre: x = y = 0x800.
            assert_eq!(report[stick..stick + 3], [0x00, 0x08, 0x80]);
            assert_eq!(report[15 - stick..18 - stick], [0, 0, 0]);
        }
    }

    fn subcommand(rail: &mut Rail, subcommand: u8, arguments: &[u8]) -> Result<Vec<u8>, String> {
        let mut output = vec![0x01, 0, 0, 1, 0x40, 0x40, 0, 1, 0x40, 0x40, subcommand];
        output.extend(arguments);
        let size = (output.len() as u16).to_le_bytes();
        let mut envelope = vec![REPORT, 0, size[0], size[1], 0, crc8(&output)];
        envelope.push(crc8(&envelope));
        let mut request = TO_JOYCON.to_vec();
        request.extend(((envelope.len() + output.len()) as u16).to_le_bytes());
        request.extend(envelope);
        request.extend(output);
        let mut answer = Vec::new();
        for byte in request {
            answer.extend(rail.byte(byte)?);
        }
        Ok(answer.split_off(12))
    }

    #[test]
    fn flash_reads_give_the_factory_data_and_erased_bytes_around_it() {
        let (mut rail, _) = connected(Side::Right);
        // Device type and its neighbours.
        let report = subcommand(&mut rail, 0x10, &[0x10, 0x60, 0, 0, 4]).unwrap();
        assert_eq!(report[0], 0x21);
        assert_eq!(report[13..15], [0x90, 0x10]);
        assert_eq!(report[15..20], [0x10, 0x60, 0, 0, 4]);
        assert_eq!(report[20..24], [0xFF, 0xFF, 0x02, 0xA0]);
        // The right stick's calibration: centre, then travel down and up.
        let report = subcommand(&mut rail, 0x10, &[0x46, 0x60, 0, 0, 9]).unwrap();
        assert_eq!(
            report[20..29],
            [0x00, 0x08, 0x80, 0x00, 0x06, 0x60, 0x00, 0x06, 0x60]
        );
        // User calibration: none.
        let report = subcommand(&mut rail, 0x10, &[0x10, 0x80, 0, 0, 2]).unwrap();
        assert_eq!(report[20..22], [0xFF, 0xFF]);
        // The firmware's part of the memory is not modeled.
        assert!(subcommand(&mut rail, 0x10, &[0, 0, 1, 0, 4]).is_err());
    }

    #[test]
    fn asleep_a_joycon_answers_only_the_handshake_and_starts_over() {
        let (mut rail, _) = connected(Side::Left);
        let report = subcommand(&mut rail, 0x06, &[0]).unwrap();
        assert_eq!(report[13..15], [0x80, 0x06]);
        assert!(send(&mut rail, &STATUS).is_empty());
        let handshake = [
            0xA1, 0xA2, 0xA3, 0xA4, 0x19, 0x01, 0x03, 0x07, 0x00, 0xA5, 0x02, 0x01, 0x7E, 0x00,
            0x00, 0x00,
        ];
        assert_eq!(send(&mut rail, &handshake).len(), 12);
        // Awake, with no session for reports yet: it answers commands,
        // not a request for its state.
        assert!(send(&mut rail, &STATUS).is_empty());
        let who = [
            0x19, 0x01, 0x03, 0x07, 0x00, 0x91, 0x01, 0x00, 0x00, 0x00, 0x00, 0x24,
        ];
        assert!(!send(&mut rail, &who).is_empty());
        // The reboots into Bluetooth modes are not modeled.
        let (mut rail, _) = connected(Side::Left);
        assert!(subcommand(&mut rail, 0x06, &[2]).is_err());
    }

    #[test]
    fn asleep_the_right_joycon_raises_its_cts_line_while_home_is_held() {
        let (mut right, controls) = connected(Side::Right);
        let mut left = Rail::new(Side::Left, Arc::clone(&controls), Arc::default());
        open_session(&mut left);
        let lines = Arc::new(Mutex::new(Vec::new()));
        let wired = Arc::clone(&lines);
        controls.connect_cts(Box::new(move |side, high| {
            wired.lock().unwrap().push((side, high));
        }));
        let seen = |expected: &[(Side, bool)]| {
            assert_eq!(std::mem::take(&mut *lines.lock().unwrap()), expected);
        };
        // Wiring tells the levels as they are: asserted on both rails.
        seen(&[(Side::Left, false), (Side::Right, false)]);
        // Awake, HOME is reported, not signalled on the line.
        controls.press(Button::HOME, true);
        assert!(right.clear_to_send());
        controls.press(Button::HOME, false);
        seen(&[(Side::Left, false), (Side::Right, false)].repeat(2));
        subcommand(&mut right, 0x06, &[0]).unwrap();
        subcommand(&mut left, 0x06, &[0]).unwrap();
        lines.lock().unwrap().clear();
        // Asleep, A does nothing to the line; HOME raises it until let go.
        controls.press(Button::A, true);
        controls.press(Button::A, false);
        assert!(right.clear_to_send() && left.clear_to_send());
        controls.press(Button::HOME, true);
        assert!(!right.clear_to_send() && left.clear_to_send());
        seen(&[
            (Side::Left, false),
            (Side::Right, false),
            (Side::Left, false),
            (Side::Right, false),
            (Side::Left, false),
            (Side::Right, true),
        ]);
        controls.press(Button::HOME, false);
        assert!(right.clear_to_send());
        // The console's handshake wakes it: HOME is reported again.
        let handshake = [
            0xA1, 0xA2, 0xA3, 0xA4, 0x19, 0x01, 0x03, 0x07, 0x00, 0xA5, 0x02, 0x01, 0x7E, 0x00,
            0x00, 0x00,
        ];
        assert_eq!(send(&mut right, &handshake).len(), 12);
        // Asked why it woke the console (as `hid` does then): HOME, once.
        let why = [
            0x19, 0x01, 0x03, 0x07, 0x00, 0x91, 0x07, 0x00, 0x00, 0x00, 0x00, 0x8E,
        ];
        let answer = send(&mut right, &why);
        assert_eq!(answer[5..7], [COMMAND_ANSWER, 0x07]);
        assert_eq!(answer[12..], [1, 0, 0, 0]);
        assert_eq!(send(&mut right, &why)[12..], [0, 0, 0, 0]);
        controls.press(Button::HOME, true);
        assert!(right.clear_to_send());
        // Snapshots keep which controllers sleep, without driving lines.
        let saved = controls.save().unwrap();
        let restored = Controls::default();
        restored.restore(&saved).unwrap();
        assert!(!restored.cts_high(Side::Right), "awake");
        assert!(restored.asleep(Side::Left));
        // An empty rail's line is high.
        controls.set_seated(false);
        assert!(controls.cts_high(Side::Left) && controls.cts_high(Side::Right));
    }

    #[test]
    fn pairing_gives_the_address_backwards_then_the_masked_key() {
        let (mut rail, _) = connected(Side::Left);
        let report = subcommand(&mut rail, 0x01, &[1, 0x02, 0x75, 0x6D, 0x45, 0x68, 0x02]).unwrap();
        assert_eq!(report[13..16], [0x81, 0x01, 1]);
        assert_eq!(report[16..22], [0x10, 0x75, 0x6D, 0x45, 0x68, 0x02]);
        assert_eq!(report[22..25], [0x08, 0x25, 0x00]);
        assert_eq!(report[25..36], *b"Joy-Con (L)");
        let report = subcommand(&mut rail, 0x01, &[2]).unwrap();
        let key: Vec<u8> = report[16..32].iter().map(|byte| byte ^ 0xAA).collect();
        assert_eq!(key, b"virtual Joy-ConL");
        assert_eq!(
            subcommand(&mut rail, 0x01, &[3]).unwrap()[13..16],
            [0x81, 0x01, 3]
        );
        // A console that knows the Joy-Con asks with step 4; having kept
        // nothing, the Joy-Con starts over.
        let report = subcommand(&mut rail, 0x01, &[4, 0x02, 0x75, 0x6D, 0x45, 0x68, 0x02]).unwrap();
        assert_eq!(report[13..16], [0x81, 0x01, 1]);
        assert_eq!(report[16..22], [0x10, 0x75, 0x6D, 0x45, 0x68, 0x02]);
        assert!(subcommand(&mut rail, 0x01, &[5]).is_err());
    }

    /// The processor's configuration requests fill the 38 bytes of the
    /// arguments: `21`, the request, its checksum at the end.
    fn processor_config(request: &[u8]) -> Vec<u8> {
        let mut arguments = vec![0x21];
        arguments.extend(request);
        arguments.resize(mcu::REQUEST_SIZE - 1, 0);
        arguments.push(mcu::crc8(&arguments[1..]));
        arguments
    }

    #[test]
    fn the_processor_reports_only_while_resumed_and_only_on_the_right() {
        let query = processor_config(&[0, 0]);
        let (mut left, _) = connected(Side::Left);
        assert!(subcommand(&mut left, 0x22, &[1]).is_err());
        assert!(subcommand(&mut left, 0x03, &[0x31]).is_err());
        let (mut rail, _) = connected(Side::Right);
        assert!(subcommand(&mut rail, 0x21, &query).is_err());
        subcommand(&mut rail, 0x22, &[1]).unwrap();
        let report = subcommand(&mut rail, 0x21, &query).unwrap();
        assert_eq!(report[13..15], [0xA0, 0x21]);
        let status = &report[15..49];
        // Kind, firmware 9.32, standby.
        assert_eq!(status[..8], [0x01, 0, 0, 0x00, 0x09, 0x00, 0x20, 0x01]);
        // A checksum of polynomial 7 over the rest leaves nothing when it
        // is run over its own result too.
        assert_eq!(mcu::crc8(status), 0);
        subcommand(&mut rail, 0x22, &[0]).unwrap();
        assert!(subcommand(&mut rail, 0x21, &query).is_err());
    }

    /// An output report of kind `0x11` over the wire: the packet counter,
    /// vibration data, then the request to the processor. The answer is
    /// the input report, without the wired envelope.
    fn processor_request(rail: &mut Rail, request: &[u8]) -> Result<Vec<u8>, String> {
        let mut output = vec![0x11, 0, 0, 1, 0x40, 0x40, 0, 1, 0x40, 0x40];
        output.extend(request);
        let size = (output.len() as u16).to_le_bytes();
        let mut envelope = vec![REPORT, 0, size[0], size[1], 0, crc8(&output)];
        envelope.push(crc8(&envelope));
        let mut packet = TO_JOYCON.to_vec();
        packet.extend(((envelope.len() + output.len()) as u16).to_le_bytes());
        packet.extend(envelope);
        packet.extend(output);
        let mut answer = Vec::new();
        for byte in packet {
            answer.extend(rail.byte(byte)?);
        }
        // The wired envelope says how long the report is.
        assert_eq!(
            usize::from(u16::from_le_bytes([answer[3], answer[4]])),
            answer.len() - 5
        );
        assert_eq!(crc8(&answer[12..]), answer[10]);
        Ok(answer.split_off(12))
    }

    /// An NFC request: kind 2, command, packet numbers, the last-packet
    /// flag, the data and its checksum.
    fn nfc_request(command: u8, data: &[u8]) -> Vec<u8> {
        let mut request = vec![0x02, command, 0, 0, 0x08, data.len() as u8];
        request.extend(data);
        request.resize(mcu::REQUEST_SIZE - 1, 0);
        request.push(mcu::crc8(&request[1..]));
        request
    }

    #[test]
    fn the_processor_answers_in_reports_of_kind_0x31_and_sees_the_tag() {
        let (mut rail, controls) = connected(Side::Right);
        // Requests to a processor not asked to report are refused.
        assert!(processor_request(&mut rail, &nfc_request(0x04, &[])).is_err());
        subcommand(&mut rail, 0x03, &[0x31]).unwrap();
        subcommand(&mut rail, 0x22, &[1]).unwrap();
        // In standby it answers a request for its state with that state.
        let mut state_request = vec![0x01];
        state_request.resize(mcu::REQUEST_SIZE, 0);
        let report = processor_request(&mut rail, &state_request).unwrap();
        assert_eq!(report.len(), MCU_REPORT_SIZE);
        assert_eq!(report[0], 0x31);
        let answer = &report[REPORT_SIZE..];
        assert_eq!(answer[..8], [0x01, 0, 0, 0x00, 0x09, 0x00, 0x20, 0x01]);
        assert_eq!(mcu::crc8(answer), 0);
        let nfc = subcommand(&mut rail, 0x21, &processor_config(&[0x00, 0x04])).unwrap();
        assert_eq!(nfc[15 + 7], 0x04);
        let report =
            processor_request(&mut rail, &nfc_request(0x01, &[0, 0, 0, 0x2C, 0x01])).unwrap();
        assert_eq!(
            report[REPORT_SIZE..REPORT_SIZE + 8],
            [0x2A, 0, 5, 0, 0, 0x09, 0x31, 1]
        );
        let uid = [0x04, 1, 2, 3, 4, 5, 6];
        controls.place_tag(Some(crate::soc::ntag::tests::blank(uid)));
        let report = processor_request(&mut rail, &nfc_request(0x04, &[])).unwrap();
        let answer = &report[REPORT_SIZE..];
        assert_eq!(mcu::crc8(answer), 0);
        assert_eq!(answer[7], 9);
        assert_eq!(answer[16..23], uid);
        controls.place_tag(None);
        let report = processor_request(&mut rail, &nfc_request(0x04, &[])).unwrap();
        assert_eq!(report[REPORT_SIZE + 7], 8);
        // A command the model does not know stops the machine with a note.
        let refusal = processor_request(&mut rail, &nfc_request(0x0F, &[])).unwrap_err();
        assert!(refusal.contains("NFC command 0x0f"), "{refusal}");
    }

    #[test]
    fn nothing_answers_on_an_empty_rail_and_the_session_does_not_survive_it() {
        let (mut rail, controls) = connected(Side::Left);
        controls.set_seated(false);
        assert_eq!(send(&mut rail, &STATUS), Vec::<u8>::new());
        assert!(!rail.clear_to_send());
        // Back on the rail, reports have to be asked for again.
        controls.set_seated(true);
        assert!(rail.clear_to_send());
        assert_eq!(send(&mut rail, &STATUS), Vec::<u8>::new());
    }
}
