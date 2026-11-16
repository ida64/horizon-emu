//! DSI host controller and the LCD panel at the other end of its link.
//!
//! `boot` drives the link in command mode (text `0x23a8` onward): it queues
//! packets in the write FIFO, sends them with the host trigger, and to read
//! the panel's identity turns the bus around and takes three words from the
//! read FIFO, of which it uses the third. Transmissions and turnarounds are
//! over before the write that starts them returns, so the trigger and the
//! turnaround request read back clear. Until the controller is enabled its
//! FIFOs take nothing: `boot` first writes zero to every register, the
//! write FIFO's included.
//!
//! What a turnaround leaves in the read FIFO is: a word saying how the
//! panel answered (`0x87`, low-power data follows, or `0x84`, a bare
//! acknowledge), then the response packet as it came off the link: header
//! with its ECC, payload and checksum. The first word's values are those
//! other drivers for this controller test for; they are not established by
//! anything the guest does with them.
//!
//! The panel is a JDI LPM062M326A, the 6.2" 720x1280 LCD of the original
//! console. It answers the identity read, follows the standard sleep and
//! display on/off commands, and accepts every other write as configuration
//! of its drive electronics, which has no effect here.
//!
//! nvservices drives the same link (text `0x3b350` onward) and checks more.
//! It writes the power control register until it reads back what was
//! written, with no timeout (`0x3b720`). Around the identity read it
//! requires the status register to report an empty read FIFO before the
//! turnaround, the number of words after it and an empty FIFO again once
//! it has taken them, with neither overflow nor underflow (`0x3a1d0`); it
//! tests nothing else there. Two pad control registers it changes by
//! read-modify-write before ever writing them (`0x12c` at `0x3b708`,
//! `0x160` at `0x3af68`): they read as zero until written, which stands for
//! whatever earlier boot stages left, an assumption about bits it does not
//! look at again.

use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex},
};

use serde::{Deserialize, Serialize};

use super::{Notes, regs::word};
use crate::{
    bus::MmioDevice,
    snapshot::{Snapshot, SnapshotError, decode, encode},
};

pub const BASE: u64 = 0x5430_0000;
pub const SIZE: u64 = 0x1000;

const RD_DATA: u64 = 0x24;
const WR_DATA: u64 = 0x28;
const POWER_CONTROL: u64 = 0x2C;
const HOST_CONTROL: u64 = 0x3C;
const TRIGGER: u64 = 0x4C;
const STATUS: u64 = 0x54;
const PAD_CONTROL_0: u64 = 0x12C;
/// Not named in public sources; next to the pad registers.
const PAD_CONTROL_160: u64 = 0x160;

/// Status: words in the read FIFO. The overflow and underflow bits above
/// them never get set: nothing is lost and nothing is read from an empty
/// FIFO without being refused.
const STATUS_READ_FIFO_COUNT: u32 = 0x1F;

/// The controller is enabled.
const POWER_CONTROL_ENABLE: u32 = 1 << 0;
/// Requests an immediate bus turnaround.
const HOST_CONTROL_IMM_BTA: u32 = 1 << 3;
/// Sends what is in the write FIFO.
const TRIGGER_HOST: u32 = 1 << 1;

/// Read FIFO words that describe the turnaround itself.
const TURNAROUND_ACKNOWLEDGE: u32 = 0x84;
const TURNAROUND_DATA: u32 = 0x87;

/// Data types of packets from the host.
mod host_packet {
    pub const GENERIC_SHORT_WRITE_0: u8 = 0x03;
    pub const DCS_SHORT_WRITE: u8 = 0x05;
    pub const DCS_READ: u8 = 0x06;
    pub const GENERIC_SHORT_WRITE_1: u8 = 0x13;
    pub const DCS_SHORT_WRITE_PARAM: u8 = 0x15;
    pub const GENERIC_SHORT_WRITE_2: u8 = 0x23;
    pub const GENERIC_LONG_WRITE: u8 = 0x29;
    pub const SET_MAXIMUM_RETURN_PACKET_SIZE: u8 = 0x37;
    pub const DCS_LONG_WRITE: u8 = 0x39;
}

/// Data types of packets from the panel.
mod panel_packet {
    pub const DCS_LONG_READ_RESPONSE: u8 = 0x1C;
    pub const DCS_SHORT_READ_RESPONSE_1: u8 = 0x21;
    pub const DCS_SHORT_READ_RESPONSE_2: u8 = 0x22;
}

/// Display command set commands the panel acts on.
mod dcs {
    pub const GET_DISPLAY_ID: u8 = 0x04;
    pub const ENTER_SLEEP_MODE: u8 = 0x10;
    pub const EXIT_SLEEP_MODE: u8 = 0x11;
    pub const SET_DISPLAY_OFF: u8 = 0x28;
    pub const SET_DISPLAY_ON: u8 = 0x29;
}

/// Identity of the LPM062M326A: manufacturer 0x10 (JDI), then revision and
/// model bytes as tools that identify console panels list them.
const DISPLAY_ID: [u8; 3] = [0x10, 0x81, 0x26];

/// The LCD module.
#[derive(Serialize, Deserialize)]
pub struct Panel {
    max_return_size: usize,
    /// Answer to the last read request, sent at the next turnaround.
    response: Option<Vec<u8>>,
    sleeping: bool,
    display_on: bool,
}

impl Panel {
    fn new() -> Self {
        // A panel comes out of reset asleep and blanked, returning one byte
        // per read until told otherwise.
        Self {
            max_return_size: 1,
            response: None,
            sleeping: true,
            display_on: false,
        }
    }

    /// Whether the panel shows what it is sent: awake and not blanked.
    pub fn is_showing(&self) -> bool {
        !self.sleeping && self.display_on
    }

    /// Takes one packet from the host. `None` if the panel model does not
    /// know what to do with it.
    fn receive(&mut self, data_type: u8, payload: &[u8]) -> Option<()> {
        use host_packet::*;
        match data_type {
            SET_MAXIMUM_RETURN_PACKET_SIZE => {
                self.max_return_size =
                    u16::from_le_bytes([*payload.first()?, *payload.get(1)?]).into();
            }
            DCS_READ => match *payload.first()? {
                dcs::GET_DISPLAY_ID => {
                    self.response =
                        Some(DISPLAY_ID[..DISPLAY_ID.len().min(self.max_return_size)].to_vec());
                }
                _ => return None,
            },
            DCS_SHORT_WRITE | DCS_SHORT_WRITE_PARAM | DCS_LONG_WRITE => match *payload.first()? {
                dcs::ENTER_SLEEP_MODE => self.sleeping = true,
                dcs::EXIT_SLEEP_MODE => self.sleeping = false,
                dcs::SET_DISPLAY_OFF => self.display_on = false,
                dcs::SET_DISPLAY_ON => self.display_on = true,
                // Drive electronics configuration.
                _ => {}
            },
            // The manufacturer's own commands.
            GENERIC_SHORT_WRITE_0
            | GENERIC_SHORT_WRITE_1
            | GENERIC_SHORT_WRITE_2
            | GENERIC_LONG_WRITE => {}
            _ => return None,
        }
        Some(())
    }
}

/// Error correction code of a packet header's first three bytes.
fn header_ecc(header: [u8; 3]) -> u8 {
    /// Header bits each parity bit covers.
    const PARITY: [u32; 6] = [
        0xF1_2CB7, 0xF2_555B, 0x74_9A6D, 0xB8_E38E, 0xDF_03F0, 0xEF_FC00,
    ];
    let bits = u32::from_le_bytes([header[0], header[1], header[2], 0]);
    PARITY.iter().enumerate().fold(0, |ecc, (index, mask)| {
        ecc | ((bits & mask).count_ones() as u8 & 1) << index
    })
}

/// Checksum of a long packet's payload.
fn payload_crc(payload: &[u8]) -> u16 {
    payload.iter().fold(0xFFFF, |crc, &byte| {
        (0..8).fold(crc ^ u16::from(byte), |crc, _| {
            crc >> 1 ^ (0x8408 & (crc & 1).wrapping_neg())
        })
    })
}

/// The packet a panel sends in answer to a read, as bytes on the link.
fn read_response(data: &[u8]) -> Vec<u8> {
    let (data_type, body) = match *data {
        [byte] => (panel_packet::DCS_SHORT_READ_RESPONSE_1, [byte, 0]),
        [first, second] => (panel_packet::DCS_SHORT_READ_RESPONSE_2, [first, second]),
        _ => (
            panel_packet::DCS_LONG_READ_RESPONSE,
            (data.len() as u16).to_le_bytes(),
        ),
    };
    let header = [data_type, body[0], body[1]];
    let mut packet = header.to_vec();
    packet.push(header_ecc(header));
    if data_type == panel_packet::DCS_LONG_READ_RESPONSE {
        packet.extend_from_slice(data);
        packet.extend_from_slice(&payload_crc(data).to_le_bytes());
    }
    packet
}

pub struct Dsi {
    state: Mutex<State>,
    notes: Arc<Notes>,
}

#[derive(Serialize, Deserialize)]
struct State {
    /// Registers without behavior of their own, as last written.
    registers: HashMap<u64, u32>,
    write_fifo: Vec<u32>,
    read_fifo: VecDeque<u32>,
    panel: Panel,
}

impl Dsi {
    pub fn new(notes: Arc<Notes>) -> Self {
        let state = State {
            registers: HashMap::from([(PAD_CONTROL_0, 0), (PAD_CONTROL_160, 0)]),
            write_fifo: Vec::new(),
            read_fifo: VecDeque::new(),
            panel: Panel::new(),
        };
        Self {
            state: Mutex::new(state),
            notes,
        }
    }

    /// Whether the panel is awake and showing the picture it receives.
    pub fn panel_is_showing(&self) -> bool {
        self.state.lock().unwrap().panel.is_showing()
    }

    /// Sends the packets queued in the write FIFO to the panel.
    fn transmit(&self, state: &mut State) -> Option<()> {
        let words = std::mem::take(&mut state.write_fifo);
        let mut words = words.iter();
        while let Some(&header) = words.next() {
            let [data_type, first, second, _] = header.to_le_bytes();
            let data_type = data_type & 0x3F;
            let long = matches!(
                data_type,
                host_packet::GENERIC_LONG_WRITE | host_packet::DCS_LONG_WRITE
            );
            let payload = if long {
                let length = usize::from(u16::from_le_bytes([first, second]));
                let mut payload: Vec<u8> = words
                    .by_ref()
                    .take(length.div_ceil(4))
                    .flat_map(|word| word.to_le_bytes())
                    .collect();
                if payload.len() < length {
                    self.notes.add(format!(
                        "DSI: long packet {header:#x} is missing part of its payload"
                    ));
                    return None;
                }
                payload.truncate(length);
                payload
            } else {
                vec![first, second]
            };
            if state.panel.receive(data_type, &payload).is_none() {
                self.notes.add(format!(
                    "DSI: the panel does not model packet type {data_type:#04x} with {payload:02x?}"
                ));
                return None;
            }
        }
        Some(())
    }
}

impl State {
    /// Hands the bus to the panel and collects what it sends.
    fn turn_around(&mut self) {
        let Some(data) = self.panel.response.take() else {
            self.read_fifo.push_back(TURNAROUND_ACKNOWLEDGE);
            return;
        };
        self.read_fifo.push_back(TURNAROUND_DATA);
        for chunk in read_response(&data).chunks(4) {
            let mut word = [0; 4];
            word[..chunk.len()].copy_from_slice(chunk);
            self.read_fifo.push_back(u32::from_le_bytes(word));
        }
    }
}

impl MmioDevice for Dsi {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        let register = word(offset, size)?;
        let mut state = self.state.lock().unwrap();
        let value = match register {
            RD_DATA => state.read_fifo.pop_front().or_else(|| {
                self.notes.add("DSI: read from the empty read FIFO".into());
                None
            })?,
            // Whatever was to be sent has been.
            TRIGGER => 0,
            STATUS => (state.read_fifo.len() as u32).min(STATUS_READ_FIFO_COUNT),
            _ => *state.registers.get(&register)?,
        };
        Some(value.into())
    }

    fn write(&self, _core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        let register = word(offset, size)?;
        let value = value as u32;
        let mut state = self.state.lock().unwrap();
        match register {
            RD_DATA | STATUS => return None,
            WR_DATA => {
                let enabled = state
                    .registers
                    .get(&POWER_CONTROL)
                    .is_some_and(|power| power & POWER_CONTROL_ENABLE != 0);
                if enabled {
                    state.write_fifo.push(value);
                }
            }
            POWER_CONTROL => {
                if value & POWER_CONTROL_ENABLE == 0 {
                    state.write_fifo.clear();
                    state.read_fifo.clear();
                }
                state.registers.insert(register, value);
            }
            TRIGGER if value & TRIGGER_HOST != 0 => self.transmit(&mut state)?,
            HOST_CONTROL => {
                if value & HOST_CONTROL_IMM_BTA != 0 {
                    state.turn_around();
                }
                state
                    .registers
                    .insert(register, value & !HOST_CONTROL_IMM_BTA);
            }
            _ => {
                state.registers.insert(register, value);
            }
        }
        Some(())
    }

    fn snapshot(&self) -> Option<&dyn Snapshot> {
        Some(self)
    }
}

/// Saves the registers, both FIFOs and the panel: its return size, the
/// answer it holds for the next turnaround, and whether it sleeps and
/// shows.
impl Snapshot for Dsi {
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        encode(&*self.state.lock().unwrap())
    }

    fn restore(&self, state: &[u8]) -> Result<(), SnapshotError> {
        *self.state.lock().unwrap() = decode("DSI", state)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dsi: &Dsi, register: u64, value: u32) -> Option<()> {
        dsi.write(0, register, 4, value.into())
    }

    fn read(dsi: &Dsi, register: u64) -> Option<u32> {
        dsi.read(0, register, 4).map(|value| value as u32)
    }

    fn send(dsi: &Dsi, words: &[u32]) {
        for &word in words {
            write(dsi, WR_DATA, word).unwrap();
        }
        write(dsi, TRIGGER, TRIGGER_HOST).unwrap();
        assert_eq!(read(dsi, TRIGGER), Some(0));
    }

    #[test]
    fn header_ecc_and_payload_crc_match_known_packets() {
        // Display on, exit sleep, display off and enter sleep short packets.
        assert_eq!(header_ecc([0x05, 0x29, 0x00]), 0x1C);
        assert_eq!(header_ecc([0x05, 0x11, 0x00]), 0x36);
        assert_eq!(header_ecc([0x05, 0x28, 0x00]), 0x06);
        assert_eq!(header_ecc([0x05, 0x10, 0x00]), 0x2C);
        // CRC-16/MCRF4XX check value.
        assert_eq!(payload_crc(b"123456789"), 0x6F91);
    }

    #[test]
    fn identity_read_follows_the_sequence_boot_uses() {
        let dsi = Dsi::new(Arc::default());
        // The register reset `boot` starts with leaves nothing queued.
        write(&dsi, WR_DATA, 0).unwrap();
        write(&dsi, POWER_CONTROL, POWER_CONTROL_ENABLE).unwrap();
        // Maximum return size 3, then read the display ID.
        send(&dsi, &[0x337]);
        send(&dsi, &[0x406]);
        // nvservices' checks around the turnaround (text 0x3a1d0).
        assert_eq!(read(&dsi, STATUS), Some(0));
        write(&dsi, HOST_CONTROL, 0x200B).unwrap();
        assert_eq!(read(&dsi, HOST_CONTROL), Some(0x2003));
        assert_eq!(read(&dsi, STATUS), Some(4));

        assert_eq!(read(&dsi, RD_DATA), Some(TURNAROUND_DATA));
        let header = read(&dsi, RD_DATA).unwrap();
        assert_eq!(
            header & 0xFF_FFFF,
            0x00_031C,
            "long read response of three bytes"
        );
        let payload = read(&dsi, RD_DATA).unwrap();
        assert_eq!(payload & 0xFF_FFFF, 0x26_8110);
        // The checksum straddles the last two words.
        let crc = payload_crc(&DISPLAY_ID);
        assert_eq!(payload >> 24, u32::from(crc & 0xFF));
        assert_eq!(read(&dsi, RD_DATA), Some(u32::from(crc >> 8)));
        assert_eq!(read(&dsi, STATUS), Some(0));
        assert_eq!(read(&dsi, RD_DATA), None);
        // Never written, yet read-modify-written by nvservices.
        assert_eq!(read(&dsi, PAD_CONTROL_0), Some(0));

        // Nothing to read: the panel only acknowledges.
        write(&dsi, HOST_CONTROL, 0x200B).unwrap();
        assert_eq!(read(&dsi, RD_DATA), Some(TURNAROUND_ACKNOWLEDGE));
    }

    #[test]
    fn panel_shows_only_awake_and_unblanked_and_long_packets_carry_their_length() {
        let dsi = Dsi::new(Arc::default());
        write(&dsi, POWER_CONTROL, POWER_CONTROL_ENABLE).unwrap();
        assert!(!dsi.panel_is_showing());
        // A seven-byte manufacturer command, then exit sleep and display on.
        send(&dsi, &[0x739, 0x7515_48B1, 0x0014_3209, 0x1105]);
        assert!(!dsi.panel_is_showing());
        send(&dsi, &[0x2905]);
        assert!(dsi.panel_is_showing());
        send(&dsi, &[0x2805]);
        assert!(!dsi.panel_is_showing());

        // A long packet cut short, and a packet type the panel has no use for.
        write(&dsi, WR_DATA, 0x739).unwrap();
        assert_eq!(write(&dsi, TRIGGER, TRIGGER_HOST), None);
        write(&dsi, WR_DATA, 0x0000_0032).unwrap();
        assert_eq!(write(&dsi, TRIGGER, TRIGGER_HOST), None);
    }
}
