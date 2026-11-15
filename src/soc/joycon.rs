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
//!   bootloader (hekate, `bdk/input/joycon.c`).
//!
//! The console's side of all of it is the firmware's own: `hid` sends the
//! wake bytes and the handshake at 1 000 000 baud on both rails as soon as
//! it sees a detect pin low (seen in an IPC trace of its `uart` session).
//!
//! What these two Joy-Con are is this model's choice, since there is no
//! real pair to copy: the constants of `Identity`. A request nothing here
//! answers is refused with a note, so the machine stops rather than go on
//! with a controller that says something a real one would not.

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU32, Ordering},
    },
    time::Instant,
};

use super::serial::Peer;

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
/// (Atmosphère's pad map, `ExtUart2Cts` and `ExtUart3Cts`). `hid` turns
/// them into any-edge wake interrupts when the console goes to sleep, for
/// which the GPIO driver samples them. The line is low while the Joy-Con
/// asserts it, which here is whenever one is on the rail
/// ([`Rail::clear_to_send`]), and pulled high on an empty rail
/// \[assumption\]. What a sleeping Joy-Con does to the line when a button
/// is pressed, which is how one wakes a console, is not modeled.
pub const RIGHT_CTS: (usize, u8) = (6, 3);
pub const LEFT_CTS: (usize, u8) = (3, 4);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Left,
    Right,
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

/// What is being done to the pair of Joy-Con: where they are and which
/// buttons are held. Shared between the rails and whoever is at the
/// controls.
#[derive(Default)]
pub struct Controls {
    seated: AtomicBool,
    buttons: AtomicU32,
}

impl Controls {
    /// Whether both Joy-Con are on their rails.
    pub fn seated(&self) -> bool {
        self.seated.load(Ordering::SeqCst)
    }

    pub(super) fn set_seated(&self, seated: bool) {
        self.seated.store(seated, Ordering::SeqCst);
    }

    /// Holds or lets go of `button`.
    pub fn press(&self, button: Button, held: bool) {
        match held {
            true => self.buttons.fetch_or(button.0, Ordering::SeqCst),
            false => self.buttons.fetch_and(!button.0, Ordering::SeqCst),
        };
    }
}

/// What makes these two Joy-Con the ones they are. None of it is copied
/// from a real pair: it is what a pair fresh from the box is documented to
/// hold, with the per-unit values chosen here.
struct Identity;

impl Identity {
    /// Bluetooth addresses, most significant byte first: locally
    /// administered, next to the console's own virtual addresses.
    const ADDRESS: [[u8; 6]; 2] = [
        [0x02, 0x68, 0x45, 0x6D, 0x75, 0x10],
        [0x02, 0x68, 0x45, 0x6D, 0x75, 0x11],
    ];
    /// Firmware version reported by "request device info": 4.32, the one
    /// this console's system software carries for Joy-Con
    /// (`FirmwareInfo.csv` of system data 0100000000000822: `JoyLeft,BT,
    /// 0420`), so that it has nothing to update.
    const FIRMWARE: [u8; 2] = [0x04, 0x20];
    /// Version of the right Joy-Con's NFC and infrared processor's
    /// firmware, likewise (`JoyRight,MCU,0920`).
    const PROCESSOR_FIRMWARE: [u16; 2] = [0x09, 0x20];
    /// Bluetooth class of device (peripheral, gamepad) and names, as the
    /// Joy-Con give them when pairing.
    const CLASS: [u8; 3] = [0x08, 0x25, 0x00];
    const NAME: [&'static str; 2] = ["Joy-Con (L)", "Joy-Con (R)"];
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
    /// What the accelerometer reads with the Joy-Con lying flat: the
    /// documented default "horizontal offsets" of each side, which is
    /// where the console expects a Joy-Con at rest on its rail to be.
    const AT_REST: [[i16; 3]; 2] = [[350, 0, 4081], [350, 0, -4081]];
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

    /// Device type: 1 for a left Joy-Con, 2 for a right one.
    fn kind(self) -> u8 {
        self as u8 + 1
    }

    /// The factory-programmed parts of the controller's flash memory, by
    /// address (`spi_flash_notes.md`).
    fn flash(self) -> Vec<(u32, Vec<u8>)> {
        let (center, travel) = (Identity::STICK_CENTER, Identity::STICK_TRAVEL);
        // Each side has one stick; the other's calibration is not
        // programmed.
        let stick = |values: [u16; 6], present: bool| match present {
            true => pack12(&values),
            false => vec![0xFF; 9],
        };
        let mut second = stick(
            [travel, travel, center, center, travel, travel],
            self == Side::Left,
        );
        second.extend(stick(
            [center, center, travel, travel, travel, travel],
            self == Side::Right,
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
/// A controller report in a wired packet: `0x31` bytes.
const REPORT_SIZE: usize = 49;
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

/// One Joy-Con, as far as its rail connector shows.
struct JoyCon {
    side: Side,
    controls: Arc<Controls>,
    /// The packet being received.
    incoming: Vec<u8>,
    /// The last few packets answered, a run of the same one counted once:
    /// what a refusal says the refused packet came after.
    answered: Vec<Vec<u8>>,
    /// The console has opened the wired session for controller reports.
    reporting: bool,
    motion: bool,
    vibration: bool,
    /// The right Joy-Con's NFC and infrared processor is running.
    processor: bool,
    /// Put to sleep by the console (subcommand 6); the handshake wakes it.
    asleep: bool,
    powered: Instant,
}

impl JoyCon {
    fn new(side: Side, controls: Arc<Controls>) -> Self {
        Self {
            side,
            controls,
            incoming: Vec::new(),
            answered: Vec::new(),
            reporting: false,
            motion: false,
            vibration: false,
            processor: false,
            asleep: false,
            powered: Instant::now(),
        }
    }

    /// The session ends when the Joy-Con leaves the rail.
    fn unplug(&mut self) {
        *self = Self::new(self.side, Arc::clone(&self.controls));
    }

    fn refuse<T>(&self, what: impl std::fmt::Display) -> Result<T, String> {
        Err(format!("{:?} Joy-Con: {what} is not modeled", self.side))
    }

    fn byte(&mut self, byte: u8) -> Result<Vec<u8>, String> {
        // Between packets only the start of one means anything: the wake
        // bytes, `A1 A2 A3 A4`, are not part of any.
        if self.incoming.is_empty() && byte != TO_JOYCON[0] {
            return Ok(Vec::new());
        }
        self.incoming.push(byte);
        let Some(header) = self.incoming.first_chunk::<5>() else {
            return Ok(Vec::new());
        };
        let length = usize::from(u16::from_le_bytes([header[3], header[4]]));
        if self.incoming.len() < 5 + length {
            return Ok(Vec::new());
        }
        let received = std::mem::take(&mut self.incoming);
        let refusal = |joycon: &Self| {
            joycon.refuse(format_args!(
                "the packet {received:02x?}, after {:02x?},",
                joycon.answered
            ))
        };
        if received[..3] != TO_JOYCON || length < ENVELOPE {
            return refusal(self);
        }
        let answered = std::mem::take(&mut self.answered);
        let answer = self.packet(&received[5..]).or_else(|()| {
            self.answered = answered.clone();
            refusal(self)
        })?;
        // The list outlasts a handshake that starts the Joy-Con over.
        self.answered = answered;
        if self.answered.last() != Some(&received) {
            self.answered.push(received);
            let excess = self.answered.len().saturating_sub(ANSWERED_KEPT);
            self.answered.drain(..excess);
        }
        Ok(answer)
    }

    /// Answers the packet whose bytes after the header are `body`. `Err`
    /// if it is one nothing is known about.
    fn packet(&mut self, body: &[u8]) -> Result<Vec<u8>, ()> {
        let (envelope, payload) = body.split_at(ENVELOPE);
        if envelope == HANDSHAKE && payload.is_empty() {
            // A sleeping Joy-Con starts over, as one just slid on.
            if self.asleep {
                self.unplug();
            }
            let mut answer = TO_CONSOLE.to_vec();
            answer.extend([ENVELOPE as u8, 0]);
            answer.extend(HANDSHAKE_ANSWER);
            return Ok(answer);
        }
        // Asleep, it listens for nothing else [assumption].
        if self.asleep {
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
        let sound = usize::from(u16::from_le_bytes([size_low, size_high])) == payload.len()
            && status == 0
            && payload_crc == crc8(payload)
            && crc == crc8(&envelope[..ENVELOPE - 1]);
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
                self.reporting = false;
                Ok(packet(COMMAND_ANSWER, 0x11, 0x0F, &[]))
            }
            (COMMAND, 0x10, []) => {
                self.reporting = true;
                Ok(packet(COMMAND_ANSWER, 0x10, 0, &[]))
            }
            // Interval of reports in milliseconds; they are sent when asked
            // for. The bootloader's driver knows of 5, 10 and 15.
            (COMMAND, 0x12, [5 | 10 | 15, 0, 0, 0]) => Ok(packet(COMMAND_ANSWER, 0x12, 0, &[])),
            // How the Joy-Con is to charge from the rail: the supply and
            // current settings the bootloader's driver names. `hid` sends
            // `0x14` after learning who is there and waits for an answer
            // with this subcommand, of which it reads nothing (text
            // `0x172F80`); that the answer is an empty one with status 0
            // is an assumption.
            (COMMAND, 0x02, [0x00 | 0x04 | 0x10 | 0x14, 0, 0, 0]) => {
                Ok(packet(COMMAND_ANSWER, 0x02, 0, &[]))
            }
            (REPORT, 0x00, output) if self.reporting => {
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
    fn report(&mut self, output: &[u8]) -> Result<Vec<u8>, ()> {
        let side = self.side.index();
        let own = [LEFT_BUTTONS, RIGHT_BUTTONS][side];
        let buttons = self.controls.buttons.load(Ordering::SeqCst) & own;
        // The stick is at its centre; the other side's is not there.
        let stick = pack12(&[Identity::STICK_CENTER, Identity::STICK_CENTER]);
        let sticks = match self.side {
            Side::Left => [stick, vec![0; 3]],
            Side::Right => [vec![0; 3], stick],
        };
        // Per side: vibration enabled, and room for more of it. Seen in
        // the notes' captures: 7 while off, 0xC while on.
        let vibrator = match self.vibration {
            true => 0xC,
            false => 0x7,
        } << [4, 0][side];
        // In units of 5 ms.
        let timer = (self.powered.elapsed().as_millis() / 5) as u8;
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
            // Vibration data only, and the bare request for the state.
            [0x10, _, _, _, _, _, _, _, _, _] | [0x1F] => {
                report[0] = 0x30;
                // Three samples of the motion sensor: at rest.
                let sample = match self.motion {
                    true => {
                        let mut sample = words(&Identity::AT_REST[side]);
                        sample.extend([0; 6]);
                        sample
                    }
                    false => vec![0; 12],
                };
                report.extend(sample.repeat(3));
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
            // and the console know each other over Bluetooth afterwards.
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
            0x01 => {
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
                let mut info = Identity::FIRMWARE.to_vec();
                info.extend([self.side.kind(), 0x02]);
                info.extend(Identity::ADDRESS[self.side.index()]);
                // Colours come from the flash memory.
                info.extend([0x01, 0x01]);
                (0x82, info)
            }
            // Input report mode: standard full reports.
            0x03 if argument(0)? == 0x30 => DONE,
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
            // are assumptions. The other states (reboot into reconnecting
            // or pairing over Bluetooth) are not modeled.
            0x06 if argument(0)? == 0 => {
                self.asleep = true;
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
                self.processor = argument(0)? == 1;
                DONE
            }
            // A command for that processor. Only the one asking how it is
            // is modeled, which `hid` sends once it has resumed it (38
            // bytes, built at text `0x153A10`: `0x21`, zeros, checksum).
            // The answer is 34 bytes (checked at `0x153C60`, read at
            // `0x158DD0`): a report of kind 1, the firmware version as two
            // big-endian words at 3 and 5, the state at 7, and a checksum
            // (CRC-8, polynomial 7) of the rest. State 1 is standby in
            // public descriptions of the protocol; that the processor is
            // there at once after being resumed is an assumption.
            0x21 if self.processor && arguments.first_chunk() == Some(&[0x21, 0, 0]) => {
                let [major, minor] = Identity::PROCESSOR_FIRMWARE.map(u16::to_be_bytes);
                let mut status = vec![0x01, 0, 0, major[0], major[1], minor[0], minor[1], 0x01];
                status.resize(33, 0);
                let checksum = status.iter().fold(0, |crc, byte| {
                    (0..8).fold(crc ^ byte, |crc: u8, _| match crc & 0x80 {
                        0 => crc << 1,
                        _ => crc << 1 ^ 0x07,
                    })
                });
                status.push(checksum);
                (0xA0, status)
            }
            // Player lights; the HOME button's light on the right one.
            0x30 => DONE,
            0x38 if self.side == Side::Right => DONE,
            // Motion sensor on or off.
            0x40 => {
                self.motion = argument(0)? != 0;
                DONE
            }
            // Motion sensor sensitivity and filters: the samples here are
            // of a Joy-Con at rest whatever they are.
            0x41 => DONE,
            // Vibration on or off.
            0x48 => {
                self.vibration = argument(0)? != 0;
                DONE
            }
            // Battery voltage, in units of 2.5 mV: 4.0 V, within the range
            // the notes give for the "full" the reports claim.
            0x50 => (0xD0, 1600u16.to_le_bytes().to_vec()),
            _ => return Err(()),
        })
    }
}

/// The serial lines of one rail, with the Joy-Con that goes on it.
pub struct Rail {
    joycon: JoyCon,
}

impl Rail {
    pub fn new(side: Side, controls: Arc<Controls>) -> Self {
        Self {
            joycon: JoyCon::new(side, controls),
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

    /// A Joy-Con on the rail lets the console send.
    fn clear_to_send(&self) -> bool {
        self.joycon.controls.seated()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seated(side: Side) -> (Rail, Arc<Controls>) {
        let controls = Arc::new(Controls::default());
        controls.set_seated(true);
        (Rail::new(side, Arc::clone(&controls)), controls)
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
        send(
            &mut rail,
            &[
                0x19, 0x01, 0x03, 0x07, 0x00, 0x91, 0x10, 0x00, 0x00, 0x00, 0x00, 0x3D,
            ],
        );
        (rail, controls)
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

    #[test]
    fn the_processor_reports_only_while_resumed_and_only_on_the_right() {
        let query = [0x21, 0, 0];
        let (mut left, _) = connected(Side::Left);
        assert!(subcommand(&mut left, 0x22, &[1]).is_err());
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
        let remainder = status.iter().fold(0u8, |crc, byte| {
            (0..8).fold(crc ^ byte, |crc, _| {
                (crc << 1) ^ if crc & 0x80 != 0 { 7 } else { 0 }
            })
        });
        assert_eq!(remainder, 0);
        subcommand(&mut rail, 0x22, &[0]).unwrap();
        assert!(subcommand(&mut rail, 0x21, &query).is_err());
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
