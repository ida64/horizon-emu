//! A Pro Controller in range of the console's Bluetooth radio: the device
//! on the far end of the air, from its baseband identity up to the HID
//! reports it carries.
//!
//! What it says is public:
//! - L2CAP signalling and channels: Bluetooth Core Specification Vol 3
//!   Part A (see [`super::l2cap`]);
//! - HID over L2CAP: the Bluetooth HID Profile, transaction headers as
//!   public Linux `net/bluetooth/hidp/hidp.h` names them; the control
//!   channel is PSM 0x11 and the interrupt channel PSM 0x13 (Bluetooth
//!   Assigned Numbers);
//! - the reports and subcommands themselves: dekuNukem's
//!   `bluetooth_hid_notes.md` and `bluetooth_hid_subcommands_notes.md`,
//!   which [`JoyCon`] models for the Joy-Con on the rails and, as
//!   [`joycon::Side::Pro`], for this controller;
//! - name, class of device and the rate of its full reports: the same
//!   notes, and public projects that present a computer as a Pro Controller
//!   to a Switch (joycontrol: "Pro Controller", class 0x002508, reports
//!   every 15 ms; facts only).
//!
//! It answers what the console's stack asks of it; anything else the
//! console asks is refused with what was asked.

use std::{path::PathBuf, sync::Arc, time::Duration};

use serde::{Deserialize, Serialize};

use super::{
    l2cap::{self, Signal},
    sdp,
};
use crate::{
    clock::GuestClock,
    snapshot::{SnapshotError, decode, encode},
    soc::joycon::{Controls, JoyCon},
};

/// PSMs (Bluetooth Assigned Numbers).
const SDP: u16 = 0x0001;
const HID_CONTROL: u16 = 0x0011;
const HID_INTERRUPT: u16 = 0x0013;

/// HIDP transaction headers (HID Profile §7.3; public Linux `hidp.h`):
/// DATA with report type Input, and with Output.
const DATA_INPUT: u8 = 0xa1;
const DATA_OUTPUT: u8 = 0xa2;

/// Output reports (`bluetooth_hid_notes.md`): vibration and a subcommand,
/// vibration only.
const SUBCOMMAND: u8 = 0x01;
const VIBRATION: u8 = 0x10;
/// Subcommands this side looks at: the input report mode, and the power
/// state ("Set HCI state") whose argument 0 sends the controller to sleep.
const SET_MODE: u8 = 0x03;
const SET_POWER: u8 = 0x06;

/// How often the controller sends its state once the console asked for
/// full reports: 15 ms, as joycontrol measured of a real one.
const REPORT_PERIOD: Duration = Duration::from_millis(15);
/// How long a woken controller tries to reach the console it is paired
/// with before it gives up and sleeps again \[assumption: the notes give
/// no time; a real one blinks for some seconds\].
const RECONNECT_FOR: Duration = Duration::from_secs(30);

/// Its identity on the air. The address is the one [`JoyCon`]'s identity
/// gives it (most significant byte first there); class of device 0x002508
/// (peripheral, gamepad, limited discoverable), the one its rail siblings
/// give.
const ADDRESS: [u8; 6] = [0x02, 0x68, 0x45, 0x6D, 0x75, 0x12];
pub(super) const CLASS: [u8; 3] = [0x08, 0x25, 0x00];
pub(super) const NAME: &str = "Pro Controller";
/// LMP features page 0 (Vol 2 Part C §3.3): the controller of a Bluetooth
/// 3.0 gamepad as this model takes it \[assumption: no public record of
/// the controller's chip\]: 3- and 5-slot packets, encryption, slot
/// offset, timing accuracy, role switch, sniff mode; power control
/// requests, CQDDR, RSSI; EDR 2 and 3 Mb/s; AFH; sniff subrating, pause
/// encryption; Secure Simple Pairing, encapsulated PDU, non-flushable
/// packet boundary flag, inquiry TX power; extended features. Not:
/// extended inquiry response (so the console asks for the name), LE.
pub(super) const FEATURES: [u8; 8] = [0xbf, 0x06, 0x06, 0x08, 0x18, 0x06, 0x48, 0x81];
/// Page 1, the host's features: Secure Simple Pairing (host support).
pub(super) const HOST_FEATURES: [u8; 8] = [0x01, 0, 0, 0, 0, 0, 0, 0];
/// Version information: LMP version 5 (Bluetooth 3.0), Broadcom (15),
/// subversion 0 \[assumption\].
pub(super) const VERSION: (u8, u16, u16) = (5, 15, 0);
/// Secure Simple Pairing capabilities (IO Capability Response): no input
/// and no output, as a gamepad has no display or keypad; no out-of-band
/// data; general bonding without MITM protection \[assumption: a
/// controller that bonds with its console and has no way to confirm a
/// number\].
pub(super) const IO_CAPABILITY: [u8; 3] = [0x03, 0x00, 0x04];

/// What the controller remembers of the console it is paired with: in its
/// flash memory on a real one (`spi_flash_notes.md`, pairing information
/// at 0x2000); here in a file, so that it outlasts the run as the
/// console's own record does.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Pairing {
    /// The console's address, in HCI byte order (least significant first).
    console: [u8; 6],
    link_key: [u8; 16],
}

#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum Power {
    /// Asleep, until a button wakes it.
    Off,
    /// Holding SYNC: discoverable and connectable, until a console
    /// connects.
    Pairing,
    /// Woken by a button: paging the console it is paired with since then.
    Reconnecting { since: Duration },
    /// Linked to a console.
    Connected,
}

/// One L2CAP channel: the PSM, the controller's own channel identifier
/// and the console's (0 until the console answers a connection the
/// controller asked for), and how far configuration has come.
#[derive(Clone, Serialize, Deserialize)]
struct Channel {
    psm: u16,
    local: u16,
    remote: u16,
    /// The console's configuration request was accepted.
    configured_in: bool,
    /// The controller's configuration request was accepted.
    configured_out: bool,
}

impl Channel {
    fn open(&self) -> bool {
        self.configured_in && self.configured_out
    }
}

/// Everything the controller holds that decides what it does next: what a
/// snapshot saves of it, with the HID session of [`JoyCon`].
#[derive(Clone, Serialize, Deserialize)]
struct State {
    power: Power,
    pairing: Option<Pairing>,
    channels: Vec<Channel>,
    /// Identifier of the controller's next signalling request.
    next_id: u8,
    /// The controller connected to the console (reconnection), and so
    /// opens the HID channels itself.
    initiator: bool,
    /// Full input reports are being sent, the next at `next_report`.
    streaming: bool,
    next_report: Duration,
    /// Asked to sleep: the link is dropped at the next look.
    leaving: bool,
    /// The SDP server, with an answer it is part way through.
    sdp: sdp::Server,
    /// The buttons the last simple HID report (0x3F) said were held; `None`
    /// before the first of the link.
    simple: Option<u32>,
    /// The link is authenticated and encrypted.
    secured: bool,
}

/// What the controller has for its link at a given moment.
#[derive(Default)]
pub(super) struct Due {
    /// L2CAP PDUs for the console.
    pub pdus: Vec<Vec<u8>>,
    /// It pages its console.
    pub page: bool,
    /// It drops the link (power off, HCI reason 0x15).
    pub disconnect: bool,
    /// When it next has something.
    pub next: Option<Duration>,
}

pub(super) struct ProController {
    hid: JoyCon,
    controls: Arc<Controls>,
    clock: Arc<GuestClock>,
    /// The file that holds [`Pairing`].
    memory: PathBuf,
    state: State,
}

impl ProController {
    /// The controller whose buttons are held on `controls`, remembering its
    /// pairing in `memory` (absent: it is paired with nothing).
    pub(super) fn new(
        controls: Arc<Controls>,
        clock: Arc<GuestClock>,
        memory: PathBuf,
    ) -> Result<Self, String> {
        let pairing = match std::fs::read(&memory) {
            Ok(bytes) => Some(
                serde_json::from_slice(&bytes)
                    .map_err(|error| format!("{}: {error}", memory.display()))?,
            ),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(format!("{}: {error}", memory.display())),
        };
        Ok(Self {
            hid: JoyCon::pro(Arc::clone(&controls), Arc::clone(&clock)),
            controls,
            clock,
            memory,
            state: State {
                power: Power::Off,
                pairing,
                channels: Vec::new(),
                next_id: 1,
                initiator: false,
                streaming: false,
                next_report: Duration::ZERO,
                leaving: false,
                sdp: sdp::Server::default(),
                simple: None,
                secured: false,
            },
        })
    }

    /// Its address in HCI byte order.
    pub(super) fn address(&self) -> [u8; 6] {
        let mut address = ADDRESS;
        address.reverse();
        address
    }

    /// Answers inquiries.
    pub(super) fn discoverable(&self) -> bool {
        self.state.power == Power::Pairing
    }

    /// Answers pages.
    pub(super) fn connectable(&self) -> bool {
        self.state.power == Power::Pairing
    }

    /// The SYNC button: drops the link it has and becomes discoverable.
    pub(super) fn sync(&mut self) -> bool {
        let linked = self.state.power == Power::Connected;
        self.reset_link();
        self.state.power = Power::Pairing;
        linked
    }

    /// The link key it holds for `console`.
    pub(super) fn link_key(&self, console: [u8; 6]) -> Option<[u8; 16]> {
        let pairing = self.state.pairing.as_ref()?;
        (pairing.console == console).then_some(pairing.link_key)
    }

    /// Pairing with `console` has produced `link_key`: kept, in its memory
    /// too, in place of any earlier pairing.
    pub(super) fn bond(&mut self, console: [u8; 6], link_key: [u8; 16]) -> Result<(), String> {
        let pairing = Pairing { console, link_key };
        let json = serde_json::to_vec(&pairing).map_err(|error| error.to_string())?;
        std::fs::write(&self.memory, json)
            .map_err(|error| format!("Pro Controller memory {}: {error}", self.memory.display()))?;
        self.state.pairing = Some(pairing);
        Ok(())
    }

    /// The link to `console` is up; `paged` if the controller paged it.
    pub(super) fn connected(&mut self, paged: bool) {
        self.reset_link();
        self.state.power = Power::Connected;
        self.state.initiator = paged;
        self.hid.unplug();
    }

    /// The link is authenticated and encrypted.
    pub(super) fn secured(&mut self) {
        self.state.secured = true;
    }

    /// The link is gone, or its page failed: the controller sleeps.
    pub(super) fn disconnected(&mut self) {
        self.reset_link();
        self.state.power = Power::Off;
    }

    fn reset_link(&mut self) {
        let state = &mut self.state;
        state.channels.clear();
        state.initiator = false;
        state.streaming = false;
        state.leaving = false;
        state.simple = None;
        state.secured = false;
    }

    fn refuse<T>(&self, what: impl std::fmt::Display) -> Result<T, String> {
        Err(format!("Pro Controller: {what} is not modeled"))
    }

    fn request_id(&mut self) -> u8 {
        let id = self.state.next_id;
        self.state.next_id = id.checked_add(1).unwrap_or(1);
        id
    }

    fn allocate(&mut self, psm: u16, remote: u16) -> u16 {
        let local = (l2cap::FIRST_DYNAMIC..)
            .find(|cid| self.state.channels.iter().all(|c| c.local != *cid))
            .expect("a free channel identifier");
        self.state.channels.push(Channel {
            psm,
            local,
            remote,
            configured_in: false,
            configured_out: false,
        });
        local
    }

    /// Its configuration request for `channel`: no options, the defaults
    /// (MTU 672, Vol 3 Part A §5.1) being enough for its reports.
    fn configure(&mut self, remote: u16) -> Vec<u8> {
        let id = self.request_id();
        l2cap::signal(
            l2cap::CONFIGURATION_REQUEST,
            id,
            &l2cap::words(&[remote, 0]),
        )
    }

    /// An L2CAP PDU from the console; what the controller sends back.
    pub(super) fn receive(&mut self, pdu: &[u8]) -> Result<Vec<Vec<u8>>, String> {
        let Some((channel, payload)) = l2cap::parse_frame(pdu) else {
            return self.refuse(format_args!("the L2CAP PDU {pdu:02x?}"));
        };
        if channel == l2cap::SIGNALLING {
            let Some(signals) = l2cap::signals(payload) else {
                return self.refuse(format_args!("the signalling frame {payload:02x?}"));
            };
            let mut answers = Vec::new();
            for signal in signals {
                answers.extend(self.signal(&signal)?);
            }
            return Ok(answers);
        }
        let Some(open) = self
            .state
            .channels
            .iter()
            .find(|c| c.local == channel && c.open())
            .cloned()
        else {
            return self.refuse(format_args!(
                "data {payload:02x?} on channel {channel:#06x}, which is not open,"
            ));
        };
        match (open.psm, payload) {
            (HID_INTERRUPT, [DATA_OUTPUT, output @ ..]) => {
                let answer = self.output(output)?;
                Ok(answer
                    .map(|report| self.input(open.remote, &report))
                    .into_iter()
                    .collect())
            }
            (SDP, _) => match self.state.sdp.answer(payload) {
                Some(answer) => Ok(vec![l2cap::frame(open.remote, &answer)]),
                None => self.refuse(format_args!("the SDP request {payload:02x?}")),
            },
            _ => self.refuse(format_args!(
                "data {payload:02x?} on the channel for PSM {:#06x}",
                open.psm
            )),
        }
    }

    /// A DATA input transaction on the console's channel `remote`.
    fn input(&self, remote: u16, report: &[u8]) -> Vec<u8> {
        let mut data = vec![DATA_INPUT];
        data.extend(report);
        l2cap::frame(remote, &data)
    }

    /// An output report; the input report that answers it, if any.
    fn output(&mut self, output: &[u8]) -> Result<Option<Vec<u8>>, String> {
        match output {
            // Vibration alone has no answer over the air: the state goes
            // out with the next full report.
            [VIBRATION, ..] => Ok(None),
            [
                SUBCOMMAND,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                subcommand,
                arguments @ ..,
            ] => {
                let Ok(report) = self.hid.report(output) else {
                    return self.refuse(format_args!("the output report {output:02x?}"));
                };
                match (*subcommand, arguments.first()) {
                    (SET_MODE, Some(0x30 | 0x31)) => {
                        self.state.streaming = true;
                        self.state.next_report = self.clock.now() + REPORT_PERIOD;
                    }
                    (SET_POWER, Some(0)) => self.state.leaving = true,
                    _ => {}
                }
                Ok(Some(report))
            }
            _ => match self.hid.report(output) {
                Ok(report) => Ok(Some(report)),
                Err(()) => self.refuse(format_args!("the output report {output:02x?}")),
            },
        }
    }

    fn signal(&mut self, signal: &Signal) -> Result<Vec<Vec<u8>>, String> {
        let word = |index| l2cap::word(signal.data, index);
        let refusal = |this: &Self| {
            this.refuse(format_args!(
                "the L2CAP signal {:#04x} (identifier {}) {:02x?}",
                signal.code, signal.id, signal.data
            ))
        };
        match signal.code {
            l2cap::CONNECTION_REQUEST if signal.data.len() == 4 => {
                let (psm, source) = (word(0).unwrap(), word(1).unwrap());
                if ![SDP, HID_CONTROL, HID_INTERRUPT].contains(&psm) {
                    return refusal(self);
                }
                let local = self.allocate(psm, source);
                Ok(vec![
                    l2cap::signal(
                        l2cap::CONNECTION_RESPONSE,
                        signal.id,
                        &l2cap::words(&[local, source, l2cap::CONNECTION_SUCCESSFUL, 0]),
                    ),
                    self.configure(source),
                ])
            }
            l2cap::CONNECTION_RESPONSE if signal.data.len() == 8 => {
                let (destination, source) = (word(0).unwrap(), word(1).unwrap());
                let result = word(2).unwrap();
                let Some(at) = self.state.channels.iter().position(|c| c.local == source) else {
                    return refusal(self);
                };
                match result {
                    l2cap::CONNECTION_PENDING => Ok(Vec::new()),
                    l2cap::CONNECTION_SUCCESSFUL => {
                        self.state.channels[at].remote = destination;
                        Ok(vec![self.configure(destination)])
                    }
                    _ => refusal(self),
                }
            }
            l2cap::CONFIGURATION_REQUEST if signal.data.len() >= 4 => {
                let (destination, flags) = (word(0).unwrap(), word(1).unwrap());
                let Some(at) = self
                    .state
                    .channels
                    .iter()
                    .position(|c| c.local == destination)
                else {
                    return refusal(self);
                };
                // Continued requests are not modeled; of the options, the
                // MTU, the flush timeout and QoS are accepted as given
                // (Vol 3 Part A §5); anything else, such as a mode other
                // than basic, is refused.
                if flags != 0 || !accepted_options(&signal.data[4..]) {
                    return refusal(self);
                }
                let channel = &mut self.state.channels[at];
                channel.configured_in = true;
                let remote = channel.remote;
                Ok(vec![l2cap::signal(
                    l2cap::CONFIGURATION_RESPONSE,
                    signal.id,
                    &l2cap::words(&[remote, 0, 0]),
                )])
            }
            l2cap::CONFIGURATION_RESPONSE if signal.data.len() >= 6 => {
                let (source, flags, result) =
                    (word(0).unwrap(), word(1).unwrap(), word(2).unwrap());
                let Some(at) = self.state.channels.iter().position(|c| c.local == source) else {
                    return refusal(self);
                };
                if flags != 0 || result != 0 {
                    return refusal(self);
                }
                self.state.channels[at].configured_out = true;
                Ok(Vec::new())
            }
            l2cap::DISCONNECTION_REQUEST if signal.data.len() == 4 => {
                let (destination, source) = (word(0).unwrap(), word(1).unwrap());
                let Some(at) = self
                    .state
                    .channels
                    .iter()
                    .position(|c| c.local == destination && c.remote == source)
                else {
                    return refusal(self);
                };
                let channel = self.state.channels.remove(at);
                if channel.psm == HID_INTERRUPT {
                    self.state.streaming = false;
                }
                Ok(vec![l2cap::signal(
                    l2cap::DISCONNECTION_RESPONSE,
                    signal.id,
                    signal.data,
                )])
            }
            // The extended features mask (type 2): none (§4.12)
            // \[assumption: a device that needs none of them\].
            l2cap::INFORMATION_REQUEST if signal.data == [2, 0] => Ok(vec![l2cap::signal(
                l2cap::INFORMATION_RESPONSE,
                signal.id,
                &[2, 0, 0, 0, 0, 0, 0, 0],
            )]),
            l2cap::ECHO_REQUEST => Ok(vec![l2cap::signal(
                l2cap::ECHO_RESPONSE,
                signal.id,
                signal.data,
            )]),
            _ => refusal(self),
        }
    }

    /// What the controller does by guest time `now` of its own accord,
    /// linked to `console` or looking for it.
    pub(super) fn due(&mut self, now: Duration, console: [u8; 6]) -> Result<Due, String> {
        let mut due = Due::default();
        match self.state.power {
            Power::Off => {
                // A button wakes a paired controller, which then looks for
                // its console.
                if self.controls.held() != 0 && self.state.pairing.is_some() {
                    self.state.power = Power::Reconnecting { since: now };
                    due.page = true;
                }
            }
            Power::Reconnecting { since } => {
                if now.saturating_sub(since) >= RECONNECT_FOR {
                    self.state.power = Power::Off;
                } else {
                    due.page = self.link_key(console).is_some();
                }
            }
            Power::Pairing => {}
            Power::Connected => {
                if self.state.leaving {
                    due.disconnect = true;
                    return Ok(due);
                }
                self.open_channels(&mut due);
                let interrupt = self
                    .state
                    .channels
                    .iter()
                    .find(|c| c.psm == HID_INTERRUPT && c.open())
                    .map(|c| c.remote);
                // In its default mode the controller sends the simple HID
                // report 0x3F (`bluetooth_hid_notes.md`: "pushed to the host
                // when a button is pressed or released"), and one once the
                // interrupt channel is open: the console sends its first
                // subcommand only after hearing from the controller, as
                // joycontrol found ("send some empty input reports until
                // the Switch decides to reply").
                let held = self.controls.held();
                if let Some(remote) =
                    interrupt.filter(|_| !self.state.streaming && self.state.simple != Some(held))
                {
                    self.state.simple = Some(held);
                    due.pdus.push(self.input(remote, &simple_report(held)));
                }
                if let Some(remote) = interrupt.filter(|_| self.state.streaming) {
                    if now >= self.state.next_report {
                        let Ok(report) = self.hid.report(&[0x1F]) else {
                            return self.refuse("the full input report of the mode set");
                        };
                        due.pdus.push(self.input(remote, &report));
                        // Behind by more than a period: the reports that
                        // would have gone out meanwhile are not made up.
                        self.state.next_report =
                            (self.state.next_report + REPORT_PERIOD).max(now + REPORT_PERIOD / 2);
                    }
                    due.next = Some(self.state.next_report);
                }
            }
        }
        Ok(due)
    }

    /// Having paged the console, the controller opens the HID control
    /// channel and, once that is configured, the interrupt channel (HID
    /// Profile: the device that reconnects opens both).
    fn open_channels(&mut self, due: &mut Due) {
        // It opens them once the link it paged is secure.
        if !self.state.initiator || !self.state.secured {
            return;
        }
        let has = |state: &State, psm| state.channels.iter().find(|c| c.psm == psm).cloned();
        let next = match (
            has(&self.state, HID_CONTROL),
            has(&self.state, HID_INTERRUPT),
        ) {
            (None, _) => HID_CONTROL,
            (Some(control), None) if control.open() => HID_INTERRUPT,
            _ => return,
        };
        let local = self.allocate(next, 0);
        let id = self.request_id();
        due.pdus.push(l2cap::signal(
            l2cap::CONNECTION_REQUEST,
            id,
            &l2cap::words(&[next, local]),
        ));
    }

    /// The controller's state, encoded.
    pub(super) fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        encode(&(&self.state, self.hid.save()?))
    }

    pub(super) fn restore(&mut self, bytes: &[u8]) -> Result<(), SnapshotError> {
        let (state, hid): (State, Vec<u8>) = decode("Pro Controller", bytes)?;
        self.hid.restore(&hid)?;
        self.state = state;
        Ok(())
    }
}

/// The simple HID report 0x3F of a Pro Controller with `held` ([`Button`]
/// bits) down and both sticks centred: two button bytes, the d-pad as a
/// hat (0 up, clockwise to 7, 8 centred), then each stick's two axes as
/// 16-bit values (`bluetooth_hid_notes.md`). Which button is which bit of
/// a Pro Controller's two bytes (B A Y X L R ZL ZR; minus, plus, the
/// stick buttons, HOME, capture) follows SDL's public HIDAPI Switch driver
/// \[inference: the notes give the Joy-Con's layout only\].
///
/// [`Button`]: crate::soc::joycon::Button
fn simple_report(held: u32) -> [u8; 12] {
    use crate::soc::joycon::Button;
    let is = |button: Button| held & button.bits() != 0;
    let bits = |buttons: &[Button]| {
        buttons
            .iter()
            .enumerate()
            .fold(0u8, |byte, (bit, &b)| byte | u8::from(is(b)) << bit)
    };
    let first = bits(&[
        Button::B,
        Button::A,
        Button::Y,
        Button::X,
        Button::L,
        Button::R,
        Button::ZL,
        Button::ZR,
    ]);
    let second = bits(&[
        Button::MINUS,
        Button::PLUS,
        Button::LEFT_STICK,
        Button::RIGHT_STICK,
        Button::HOME,
        Button::CAPTURE,
    ]);
    let hat = match (
        is(Button::UP),
        is(Button::RIGHT),
        is(Button::DOWN),
        is(Button::LEFT),
    ) {
        (true, false, _, false) => 0,
        (true, true, _, _) => 1,
        (false, true, false, _) => 2,
        (_, true, true, _) => 3,
        (false, false, true, false) => 4,
        (_, _, true, true) => 5,
        (false, _, false, true) => 6,
        (true, _, _, true) => 7,
        _ => 8,
    };
    [0x3f, first, second, hat, 0, 0x80, 0, 0x80, 0, 0x80, 0, 0x80]
}

/// Whether every option of a configuration request is one the controller
/// takes as given: MTU (1), flush timeout (2), QoS (3), or a hint (type
/// bit 7) it may skip (Vol 3 Part A §5).
fn accepted_options(mut options: &[u8]) -> bool {
    while let [kind, length, rest @ ..] = options {
        let Some(rest) = rest.get(usize::from(*length)..) else {
            return false;
        };
        if !matches!(kind & 0x7f, 1..=3) && kind & 0x80 == 0 {
            return false;
        }
        options = rest;
    }
    options.is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn controller() -> (ProController, Arc<Controls>, PathBuf) {
        let memory = std::env::temp_dir().join(format!(
            "horizon-emu-pro-{}-{:?}.json",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_file(&memory);
        let controls = Arc::new(Controls::default());
        let pro = ProController::new(
            Arc::clone(&controls),
            Arc::new(GuestClock::new()),
            memory.clone(),
        )
        .unwrap();
        (pro, controls, memory)
    }

    /// Opens `psm` from the console's channel `source`: the controller's
    /// channel identifier.
    fn open(pro: &mut ProController, psm: u16, source: u16) -> u16 {
        let answers = pro
            .receive(&l2cap::signal(
                l2cap::CONNECTION_REQUEST,
                7,
                &l2cap::words(&[psm, source]),
            ))
            .unwrap();
        assert_eq!(answers.len(), 2);
        let (_, response) = l2cap::parse_frame(&answers[0]).unwrap();
        let response = &l2cap::signals(response).unwrap()[0];
        assert_eq!(
            (response.code, response.id),
            (l2cap::CONNECTION_RESPONSE, 7)
        );
        let local = l2cap::word(response.data, 0).unwrap();
        assert_eq!(&response.data[2..], &l2cap::words(&[source, 0, 0])[..]);
        // Its own configuration request, to the console's channel.
        let (_, request) = l2cap::parse_frame(&answers[1]).unwrap();
        let request = &l2cap::signals(request).unwrap()[0];
        assert_eq!(request.code, l2cap::CONFIGURATION_REQUEST);
        assert_eq!(request.data, l2cap::words(&[source, 0]));
        // The console's request, with an MTU, is accepted; then the
        // console accepts the controller's.
        let configure = [l2cap::words(&[local, 0]), vec![1, 2, 0x80, 0x02]].concat();
        let answers = pro
            .receive(&l2cap::signal(l2cap::CONFIGURATION_REQUEST, 8, &configure))
            .unwrap();
        assert_eq!(
            answers,
            [l2cap::signal(
                l2cap::CONFIGURATION_RESPONSE,
                8,
                &l2cap::words(&[source, 0, 0])
            )]
        );
        let accepted = l2cap::words(&[local, 0, 0]);
        let answers = pro
            .receive(&l2cap::signal(
                l2cap::CONFIGURATION_RESPONSE,
                request.id,
                &accepted,
            ))
            .unwrap();
        assert!(answers.is_empty());
        local
    }

    #[test]
    fn subcommands_are_answered_on_the_interrupt_channel_and_full_reports_follow() {
        let (mut pro, controls, _) = controller();
        pro.sync();
        pro.connected(false);
        open(&mut pro, HID_CONTROL, 0x0040);
        let interrupt = open(&mut pro, HID_INTERRUPT, 0x0041);
        // Request device info (subcommand 2).
        let mut output = vec![DATA_OUTPUT, SUBCOMMAND, 0];
        output.extend([0; 8]);
        output.push(0x02);
        let answers = pro.receive(&l2cap::frame(interrupt, &output)).unwrap();
        assert_eq!(answers.len(), 1);
        let (channel, input) = l2cap::parse_frame(&answers[0]).unwrap();
        assert_eq!(channel, 0x0041);
        assert_eq!(input[..2], [DATA_INPUT, 0x21]);
        // Acknowledged with data, subcommand 2: firmware 4.33, a Pro
        // Controller (3).
        assert_eq!(input[14..19], [0x82, 0x02, 0x04, 0x21, 0x03]);
        // Until the console sets the full report mode, one simple report
        // (0x3F, nothing held, hat centred) says the controller is there,
        // and nothing more while nothing changes.
        let first = pro.due(Duration::MAX / 2, [0; 6]).unwrap();
        let (_, simple) = l2cap::parse_frame(&first.pdus[0]).unwrap();
        assert_eq!(simple[..5], [DATA_INPUT, 0x3f, 0, 0, 0x08]);
        assert!(pro.due(Duration::MAX / 2, [0; 6]).unwrap().pdus.is_empty());
        output[11] = SET_MODE;
        output.push(0x30);
        pro.receive(&l2cap::frame(interrupt, &output)).unwrap();
        controls.press(crate::soc::joycon::Button::LEFT, true);
        let due = pro.due(Duration::MAX / 2, [0; 6]).unwrap();
        let (_, report) = l2cap::parse_frame(&due.pdus[0]).unwrap();
        assert_eq!(report[..2], [DATA_INPUT, 0x30]);
        // The left button of the d-pad, in the left byte.
        assert_eq!(report[6], 0x08);
        assert!(due.next.is_some());
        // Vibration alone has no answer; data on a closed channel is
        // refused.
        let vibration = [&[DATA_OUTPUT, VIBRATION, 1][..], &[0; 8]].concat();
        assert!(
            pro.receive(&l2cap::frame(interrupt, &vibration))
                .unwrap()
                .is_empty()
        );
        assert!(pro.receive(&l2cap::frame(0x0077, &vibration)).is_err());
    }

    #[test]
    fn a_paired_controller_woken_by_a_button_reconnects_and_opens_both_channels() {
        let (mut pro, controls, memory) = controller();
        let console = [1, 2, 3, 4, 5, 6];
        pro.bond(console, [9; 16]).unwrap();
        // Its memory outlasts it.
        let again = ProController::new(
            Arc::clone(&controls),
            Arc::new(GuestClock::new()),
            memory.clone(),
        )
        .unwrap();
        assert_eq!(again.link_key(console), Some([9; 16]));
        assert_eq!(again.link_key([0; 6]), None);
        std::fs::remove_file(&memory).unwrap();

        assert!(!pro.due(Duration::ZERO, console).unwrap().page);
        controls.press(crate::soc::joycon::Button::A, true);
        assert!(pro.due(Duration::ZERO, console).unwrap().page);
        pro.connected(true);
        // Nothing until the link is secure.
        assert!(pro.due(Duration::ZERO, console).unwrap().pdus.is_empty());
        pro.secured();
        let due = pro.due(Duration::ZERO, console).unwrap();
        let (_, request) = l2cap::parse_frame(&due.pdus[0]).unwrap();
        let request = l2cap::signals(request).unwrap()[0].clone();
        assert_eq!(request.code, l2cap::CONNECTION_REQUEST);
        assert_eq!(l2cap::word(request.data, 0), Some(HID_CONTROL));
        let local = l2cap::word(request.data, 1).unwrap();
        // Pending, then accepted: the controller configures the channel.
        let pending = l2cap::words(&[0, local, l2cap::CONNECTION_PENDING, 0]);
        let answers = pro
            .receive(&l2cap::signal(
                l2cap::CONNECTION_RESPONSE,
                request.id,
                &pending,
            ))
            .unwrap();
        assert!(answers.is_empty());
        let accepted = l2cap::words(&[0x0050, local, 0, 0]);
        let answers = pro
            .receive(&l2cap::signal(
                l2cap::CONNECTION_RESPONSE,
                request.id,
                &accepted,
            ))
            .unwrap();
        let (_, configure) = l2cap::parse_frame(&answers[0]).unwrap();
        let configure = l2cap::signals(configure).unwrap()[0].clone();
        assert_eq!(configure.data, l2cap::words(&[0x0050, 0]));
        // Nothing more until the control channel is configured both ways.
        assert!(pro.due(Duration::ZERO, console).unwrap().pdus.is_empty());
        pro.receive(&l2cap::signal(
            l2cap::CONFIGURATION_REQUEST,
            3,
            &l2cap::words(&[local, 0]),
        ))
        .unwrap();
        pro.receive(&l2cap::signal(
            l2cap::CONFIGURATION_RESPONSE,
            configure.id,
            &l2cap::words(&[local, 0, 0]),
        ))
        .unwrap();
        let due = pro.due(Duration::ZERO, console).unwrap();
        let (_, request) = l2cap::parse_frame(&due.pdus[0]).unwrap();
        let request = &l2cap::signals(request).unwrap()[0];
        assert_eq!(l2cap::word(request.data, 0), Some(HID_INTERRUPT));
    }

    #[test]
    fn configuration_options_other_than_mtu_flush_and_qos_are_refused() {
        assert!(accepted_options(&[1, 2, 0x80, 0x02, 2, 2, 0xff, 0xff]));
        // A hint it does not know may be skipped.
        assert!(accepted_options(&[0x85, 1, 0]));
        // Retransmission and flow control (4), and a cut-short option.
        assert!(!accepted_options(&[4, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0]));
        assert!(!accepted_options(&[1, 2, 0x80]));
    }
}
