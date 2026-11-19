//! The right Joy-Con's NFC and infrared processor (its "MCU"), in the mode
//! that reads and writes NFC tags.
//!
//! The console talks to the processor through the controller: subcommand
//! `0x22` powers it, subcommand `0x21` configures it and answers with a
//! 34-byte report of its state, and once the console has asked for input
//! reports of kind `0x31` every output report of kind `0x11` carries a
//! 38-byte request to the processor and every input report the 313 bytes
//! of its answer. Public descriptions of the protocol name the parts used
//! here (dekuNukem's `bluetooth_hid_subcommands_notes.md` for the
//! subcommands and report kinds; the Joy-Con NFC driver of an open-source
//! emulator, yuzu's `joycon_protocol/nfc.cpp`, for the NFC commands, their
//! arguments and the processor's NFC states). Layouts are what `hid` builds
//! and checks; text offsets below are in its `main` (system module
//! 0100000000000013):
//!
//! - a request: request kind (`0x01` state, `0x02` NFC), then for NFC the
//!   command, the console's packet number, the number of the processor's
//!   last packet it took, flags (`0x08`: the last packet) with the top
//!   three bits of the length, the rest of the length, up to 31 bytes of
//!   data, and a CRC-8 (polynomial 7) of all but the first byte (built at
//!   `0x157420`);
//! - an answer: report kind, then for NFC a result code (0: none), what it
//!   carries (5: the NFC state, 7: data read from a tag), the processor's
//!   packet number (0 for a state, counting from 1 for data, checked at
//!   `0x156E40`), the number of the console's last packet it took, flags
//!   and length as above, the data, and in its last byte a CRC-8 of all
//!   the bytes before it (checked at `0x153C60`).
//!
//! What the processor's own firmware does between those reports is not
//! run: the tag commands it sends are chosen here, from what the console
//! asked for, and given to the tag model ([`crate::soc::ntag`]).

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::soc::ntag::Ntag215;

/// Bytes of a request (the output report's MCU part) and of an answer
/// (the input report's), and of the answer to subcommand `0x21`.
pub(super) const REQUEST_SIZE: usize = 38;
pub(super) const ANSWER_SIZE: usize = 313;
const CONFIG_ANSWER_SIZE: usize = 34;
/// Bytes of an NFC answer before its data, and the most data one holds.
const HEADER: usize = 7;
const MAX_DATA: usize = ANSWER_SIZE - HEADER - 1;

/// Kinds of answer (the first byte): the processor's state, an empty one,
/// the NFC state and NFC data.
const STATE_REPORT: u8 = 0x01;
const EMPTY: u8 = 0xFF;
const NFC_STATE: u8 = 0x2A;
const NFC_DATA: u8 = 0x3A;
/// What an NFC answer carries.
const CARRIES_STATE: u8 = 0x05;
const CARRIES_READ_DATA: u8 = 0x07;
/// The flag of the last packet of a transfer.
const LAST: u8 = 0x08;

/// The processor's modes: standby once powered, and NFC.
const STANDBY: u8 = 1;
const NFC: u8 = 4;

/// NFC states, as the processor reports them. `hid` raises its "tag
/// found" event when the state becomes 9 and its "tag lost" event when it
/// becomes 8 (`0x156680`); a read ends with state 4 and a write with state
/// 5 (`0x156DB0`, `0x1567E0`). 0 is idle and 1 polling with nothing found
/// (yuzu's names).
const IDLE: u8 = 0;
const POLLING: u8 = 1;
const READ_DONE: u8 = 4;
const WRITE_DONE: u8 = 5;
const TAG_LOST: u8 = 8;
const TAG_FOUND: u8 = 9;

/// NFC commands (yuzu's `NFCCommand`).
const CANCEL: u8 = 0x00;
const START_POLLING: u8 = 0x01;
const STOP_POLLING: u8 = 0x02;
const NEXT: u8 = 0x04;
const READ_NTAG: u8 = 0x06;
const WRITE_NTAG: u8 = 0x08;

/// How the processor describes a tag: an NFC Forum Type 2 tag over NFC-A,
/// as `hid` maps them to the console's tag kinds (`0x158490`: protocol 0,
/// type 2).
const TYPE_2: u8 = 0x02;
const NFC_A: u8 = 0x00;

/// CRC-8 with polynomial 7, as the processor's reports and requests carry.
pub(super) fn crc8(data: &[u8]) -> u8 {
    data.iter().fold(0, |crc, byte| {
        (0..8).fold(crc ^ byte, |crc: u8, _| match crc & 0x80 {
            0 => crc << 1,
            _ => crc << 1 ^ 0x07,
        })
    })
}

/// A transfer of data to the console in numbered packets, and the state
/// the processor is in once the console has taken the last one.
#[derive(Serialize, Deserialize)]
struct Outgoing {
    packets: Vec<Vec<u8>>,
    then: u8,
}

/// The processor as far as the console can see it. Saved with the
/// Joy-Con's session.
#[derive(Serialize, Deserialize)]
pub(super) struct Mcu {
    powered: bool,
    mode: u8,
    /// The console's last request asked for the processor's state rather
    /// than for NFC: the next answer is the state report.
    state_asked: bool,
    nfc_state: u8,
    polling: bool,
    /// The guest time a poll with a time limit ends at if it finds
    /// nothing.
    polling_ends: Option<Duration>,
    /// The serial number of the tag the state reports describe.
    found: Option<[u8; 7]>,
    /// What the console has still to take, and the number of the packet
    /// it took last.
    outgoing: Option<Outgoing>,
    taken: u8,
    /// A write arriving in several packets: its data so far, and the
    /// number of the console's last packet.
    incoming: Option<Vec<u8>>,
    received: u8,
}

impl Mcu {
    pub(super) fn new() -> Self {
        Self {
            powered: false,
            mode: STANDBY,
            state_asked: true,
            nfc_state: IDLE,
            polling: false,
            polling_ends: None,
            found: None,
            outgoing: None,
            taken: 0,
            incoming: None,
            received: 0,
        }
    }

    /// Whether it is resumed.
    pub(super) fn powered(&self) -> bool {
        self.powered
    }

    /// Subcommand `0x22`: resume (1) or suspend (0). Resumed, it is in
    /// standby, which is an assumption: a real one reports itself busy for
    /// a while first.
    pub(super) fn power(&mut self, on: bool) {
        *self = Self {
            powered: on,
            ..Self::new()
        };
    }

    /// The 34-byte report of the processor's state: kind 1, the firmware
    /// version as two big-endian words at 3 and 5, the mode at 7, and a
    /// checksum of the rest (`hid` reads it at `0x158DD0`).
    fn state_report(&self, size: usize, firmware: [u16; 2]) -> Vec<u8> {
        let [major, minor] = firmware.map(u16::to_be_bytes);
        let mut report = vec![STATE_REPORT, 0, 0, major[0], major[1], minor[0], minor[1]];
        report.push(self.mode);
        report.resize(size - 1, 0);
        report.push(crc8(&report));
        report
    }

    /// Subcommand `0x21`, a configuration request of 38 bytes: `21`, what
    /// to do, its argument, and a checksum of all but the first byte at
    /// the end. `21 00 MODE` sets the mode (`hid` builds it at
    /// `0x1542A0`, the mode from its table at `0x1B9608`), of which
    /// standby and NFC are modeled; `21 00 00`, which `hid` sends to learn
    /// the state (`0x153A10`), changes nothing. `None` for anything else.
    pub(super) fn configure(&mut self, request: &[u8], firmware: [u16; 2]) -> Option<Vec<u8>> {
        let request = request.get(..REQUEST_SIZE)?;
        if !self.powered || request[REQUEST_SIZE - 1] != crc8(&request[1..REQUEST_SIZE - 1]) {
            return None;
        }
        match request[..3] {
            [0x21, 0x00, 0x00] => {}
            [0x21, 0x00, mode @ (STANDBY | NFC)] if self.mode != mode => {
                *self = Self {
                    powered: true,
                    mode,
                    ..Self::new()
                };
            }
            [0x21, 0x00, mode] if self.mode == mode => {}
            _ => return None,
        }
        Some(self.state_report(CONFIG_ANSWER_SIZE, firmware))
    }

    /// A request in an output report of kind `0x11`. `Err` with what is
    /// not modeled.
    pub(super) fn request(
        &mut self,
        request: &[u8],
        tag: &mut Option<Ntag215>,
        now: Duration,
    ) -> Result<(), String> {
        let request: &[u8; REQUEST_SIZE] = request
            .get(..REQUEST_SIZE)
            .and_then(|bytes| bytes.try_into().ok())
            .ok_or("an MCU request of the wrong size")?;
        // The answer follows the kind of request: the processor's state,
        // which public descriptions of the protocol have the console ask
        // for until the mode it set is reported, or NFC.
        match request[0] {
            0x01 if request[1..] == [0; REQUEST_SIZE - 1] => {
                self.state_asked = true;
                return Ok(());
            }
            0x02 if self.powered && self.mode == NFC => self.state_asked = false,
            _ => return Err("an MCU request of this kind, in this mode,".into()),
        }
        if request[REQUEST_SIZE - 1] != crc8(&request[1..REQUEST_SIZE - 1]) {
            return Err("an NFC request with a bad checksum".into());
        }
        let &[_, command, packet, taken, flags, length_low, ..] = request;
        let length = usize::from(flags & 7) << 8 | usize::from(length_low);
        let data = request
            .get(6..6 + length)
            .filter(|_| length <= REQUEST_SIZE - 7)
            .ok_or("an NFC request longer than its packet")?;
        let last = flags & LAST != 0;
        self.taken = taken;
        if let Some(outgoing) = &self.outgoing
            && usize::from(taken) >= outgoing.packets.len()
        {
            self.nfc_state = outgoing.then;
            self.outgoing = None;
        }
        match command {
            NEXT => {}
            CANCEL | STOP_POLLING if last => {
                self.polling = false;
                self.nfc_state = IDLE;
                self.found = None;
                self.outgoing = None;
                self.incoming = None;
            }
            // StartPolling's data (`hid` builds it at `0x157630`): a byte,
            // then two 16-bit times in milliseconds, `03e8` (1 s) and
            // `012c`, of which the first is 0 when `hid` starts looking
            // for a tag and 1000 when it polls again to see whether the
            // one it found is still there.
            //
            // \[inference\] The first is how long the poll goes on without
            // finding a tag (0: until stopped), and one that runs out ends
            // in state 8: `hid` learns that a tag has gone from nothing
            // but that state, and it starts these polls again only after
            // each "found".
            START_POLLING if last => {
                let limit = data
                    .get(1..3)
                    .map_or(0, |ms| u16::from_le_bytes([ms[0], ms[1]]));
                self.polling = true;
                self.polling_ends = (limit != 0).then(|| now + Duration::from_millis(limit.into()));
                self.nfc_state = POLLING;
                self.found = None;
            }
            READ_NTAG if last => {
                let tag = tag
                    .as_mut()
                    .ok_or("reading a tag with no tag on the reader")?;
                self.read(data, tag)?;
            }
            WRITE_NTAG => {
                // Packets count from 1; the first one starts the command.
                // The same packet again has been taken already.
                if self.incoming.is_some() && packet == self.received {
                    return Ok(());
                }
                if packet == 1 {
                    self.incoming = Some(Vec::new());
                } else if self.incoming.is_none() || packet != self.received.wrapping_add(1) {
                    return Err(format!("NFC write packet {packet} out of order"));
                }
                let assembled = self.incoming.as_mut().expect("started above");
                assembled.extend_from_slice(data);
                self.received = packet;
                if last {
                    let command = self.incoming.take().expect("started above");
                    let tag = tag
                        .as_mut()
                        .ok_or("writing a tag with no tag on the reader")?;
                    self.write(&command, tag)?;
                }
            }
            _ => return Err(format!("the NFC command {command:#04x}")),
        }
        Ok(())
    }

    /// The tag on the reader as the state reports describe it (61 bytes
    /// of state that `hid` keeps, `0x1562F0`): the state, at 4 whether a
    /// tag is there, at 6 its type, at 7 its protocol, at 8 the length of
    /// its serial number and the number from 9.
    fn state_data(&self) -> Vec<u8> {
        let mut data = vec![0; 61];
        data[0] = self.nfc_state;
        if let Some(uid) = self.found {
            data[4] = 1;
            data[6] = TYPE_2;
            data[7] = NFC_A;
            data[8] = uid.len() as u8;
            data[9..9 + uid.len()].copy_from_slice(&uid);
        }
        data
    }

    /// ReadNtag: `d0 07`, the serial number the tag must have (all zeros:
    /// any), which tags (yuzu: 0 any, 1 NTAG215), how many runs of pages
    /// (up to 4) and each one's first and last page. The answer, sent in
    /// packets of NFC data (`hid` puts them together at `0x1562F0` and
    /// takes them apart at `0x158810`), is the tag's description as in
    /// the state reports from byte 1, then 32 bytes at 19 that `hid` hands
    /// on, the runs at 51, and the pages from 60 on.
    ///
    /// \[inference\] The 32 bytes are the tag's originality signature
    /// (READ_SIG answers 32 bytes; nothing else of an NTAG215 is that
    /// size), and byte 18, which `hid` hands on too, is left 0: what the
    /// processor puts there is not established.
    fn read(&mut self, data: &[u8], tag: &mut Ntag215) -> Result<(), String> {
        let [0xD0, 0x07, ref uid @ .., kind, runs, a, b, c, d, e, f, g, h] = *data else {
            return Err(format!("an NTAG read of {data:02x?}"));
        };
        if uid != [0; 7] && uid != tag.uid() || kind > 1 || !(1..=4).contains(&runs) {
            return Err(format!("an NTAG read of {data:02x?}"));
        }
        let mut answer = self.state_data()[1..19].to_vec();
        answer.push(0);
        let signature = tag
            .command(&[0x3C, 0x00])
            .map_err(|nak| format!("the tag refused READ_SIG: {nak:?}"))?;
        answer.extend(signature);
        answer.push(runs);
        let pairs = [a, b, c, d, e, f, g, h];
        answer.extend(pairs);
        for run in pairs.chunks(2).take(runs.into()) {
            let pages = tag.command(&[0x3A, run[0], run[1]]).map_err(|nak| {
                format!(
                    "the tag refused FAST_READ {:#04x}..{:#04x}: {nak:?}",
                    run[0], run[1]
                )
            })?;
            answer.extend(pages);
        }
        let count = answer.len().div_ceil(MAX_DATA);
        let packets = answer
            .chunks(MAX_DATA)
            .enumerate()
            .map(|(index, chunk)| {
                let flags = if index + 1 == count { LAST } else { 0 };
                nfc_answer(
                    NFC_DATA,
                    CARRIES_READ_DATA,
                    index as u8 + 1,
                    0,
                    flags,
                    chunk,
                )
            })
            .collect();
        self.outgoing = Some(Outgoing {
            packets,
            then: READ_DONE,
        });
        self.taken = 0;
        Ok(())
    }

    /// WriteNtag, put together from its packets: `d0 07`, the serial
    /// number, which tags, 7 bytes yuzu sends as `00 01 04 ff ff ff ff`,
    /// the 4 bytes for page 4 (yuzu: amiibo magic, write counter,
    /// version), how many runs, and each run's first page, length in bytes
    /// and bytes (`hid` builds it at `0x157A40`).
    ///
    /// \[assumption\] The processor writes the runs page by page with
    /// WRITE and then page 4; that it first authenticates with the
    /// password amiibo writers derive from the serial number when the tag
    /// guards those pages, and nothing else about what its firmware does,
    /// is taken from public amiibo tools. The 7 bytes are not interpreted.
    fn write(&mut self, command: &[u8], tag: &mut Ntag215) -> Result<(), String> {
        let refuse = || format!("an NTAG write of {command:02x?}");
        let (header, mut runs) = command.split_at_checked(22).ok_or_else(refuse)?;
        let [
            0xD0,
            0x07,
            ref uid @ ..,
            kind,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            p0,
            p1,
            p2,
            p3,
            count,
        ] = *header
        else {
            return Err(refuse());
        };
        if uid != tag.uid() || kind > 1 || count > 4 {
            return Err(refuse());
        }
        let mut writes = Vec::new();
        for _ in 0..count {
            let [page, length, ref rest @ ..] = *runs else {
                return Err(refuse());
            };
            let (bytes, after) = rest.split_at_checked(length.into()).ok_or_else(refuse)?;
            if bytes.len() % 4 != 0 {
                return Err(refuse());
            }
            for (n, page_bytes) in bytes.chunks(4).enumerate() {
                writes.push((usize::from(page) + n, page_bytes.to_vec()));
            }
            runs = after;
        }
        writes.push((4, vec![p0, p1, p2, p3]));
        let auth0 = tag.dump()[0x83 * 4 + 3];
        if writes.iter().any(|(page, _)| *page >= usize::from(auth0)) {
            let uid = tag.uid();
            let password = [
                0xAA ^ uid[1] ^ uid[3],
                0x55 ^ uid[2] ^ uid[4],
                0xAA ^ uid[3] ^ uid[5],
                0x55 ^ uid[4] ^ uid[6],
            ];
            let mut frame = vec![0x1B];
            frame.extend(password);
            tag.command(&frame)
                .map_err(|nak| format!("the tag refused the amiibo password: {nak:?}"))?;
        }
        for (page, bytes) in writes {
            let page = u8::try_from(page).map_err(|_| refuse())?;
            let mut frame = vec![0xA2, page];
            frame.extend(bytes);
            tag.command(&frame)
                .map_err(|nak| format!("the tag refused WRITE {page:#04x}: {nak:?}"))?;
        }
        tag.save_to_file()?;
        self.nfc_state = WRITE_DONE;
        Ok(())
    }

    /// The MCU part of the next input report of kind `0x31`, with `tag` on
    /// the reader (or none), at guest time `now`.
    pub(super) fn answer(
        &mut self,
        tag: Option<&Ntag215>,
        firmware: [u16; 2],
        now: Duration,
    ) -> Vec<u8> {
        if !self.powered {
            let mut answer = vec![0; ANSWER_SIZE];
            answer[0] = EMPTY;
            return answer;
        }
        if self.mode != NFC || self.state_asked {
            return self.state_report(ANSWER_SIZE, firmware);
        }
        if let Some(outgoing) = &self.outgoing
            && let Some(packet) = outgoing.packets.get(usize::from(self.taken))
        {
            return packet.clone();
        }
        // A tag coming into the field of a polling reader is found; one
        // leaving it, or another taking its place, is lost first. A poll
        // with a time limit that finds nothing ends as lost too.
        let uid = tag.map(Ntag215::uid);
        if self.found.is_some() && self.found != uid {
            self.found = None;
            self.nfc_state = TAG_LOST;
        } else if self.polling
            && self.found.is_none()
            && uid.is_some()
            && matches!(self.nfc_state, POLLING | TAG_LOST)
        {
            self.found = uid;
            self.nfc_state = TAG_FOUND;
        } else if self.polling
            && self.found.is_none()
            && self.polling_ends.is_some_and(|end| now >= end)
        {
            self.polling = false;
            self.polling_ends = None;
            self.nfc_state = TAG_LOST;
        }
        nfc_answer(
            NFC_STATE,
            CARRIES_STATE,
            0,
            self.received,
            LAST,
            &self.state_data(),
        )
    }
}

/// An NFC answer: `data` in a packet numbered `packet`, with the console's
/// last packet taken, its flags, and the checksum. A state report gives
/// its length as the whole room (yuzu checks the `0x31` of `09 31`).
fn nfc_answer(kind: u8, carries: u8, packet: u8, taken: u8, flags: u8, data: &[u8]) -> Vec<u8> {
    let length = match carries {
        CARRIES_STATE => MAX_DATA,
        _ => data.len(),
    };
    let mut answer = vec![
        kind,
        0,
        carries,
        packet,
        taken,
        flags | (length >> 8) as u8,
        length as u8,
    ];
    answer.extend(data);
    answer.resize(ANSWER_SIZE - 1, 0);
    answer.push(crc8(&answer));
    answer
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::soc::ntag;

    const FIRMWARE: [u16; 2] = [0x09, 0x20];
    const UID: [u8; 7] = [0x04, 0xA1, 0xB2, 0xC3, 0xD4, 0xE5, 0xF6];

    /// A request with a correct checksum.
    fn request(kind: u8, command: u8, packet: u8, taken: u8, flags: u8, data: &[u8]) -> Vec<u8> {
        let mut request = vec![kind, command, packet, taken, flags, data.len() as u8];
        request.extend(data);
        request.resize(REQUEST_SIZE - 1, 0);
        request.push(crc8(&request[1..]));
        request
    }

    fn nfc(command: u8, packet: u8, taken: u8, flags: u8, data: &[u8]) -> Vec<u8> {
        request(0x02, command, packet, taken, flags, data)
    }

    fn config(arguments: &[u8]) -> Vec<u8> {
        let mut request = vec![0x21];
        request.extend(arguments);
        request.resize(REQUEST_SIZE - 1, 0);
        request.push(crc8(&request[1..]));
        request
    }

    fn nfc_mode() -> Mcu {
        let mut mcu = Mcu::new();
        mcu.power(true);
        mcu.configure(&config(&[0x00, NFC]), FIRMWARE).unwrap();
        mcu
    }

    /// An answer's checksum covers every byte before it, so running the
    /// CRC over the whole answer leaves nothing.
    fn checks_out(answer: &[u8]) -> bool {
        answer.len() == ANSWER_SIZE && crc8(answer) == 0
    }

    #[test]
    fn the_checksum_is_crc8_with_polynomial_7() {
        // CRC-8/SMBUS check value.
        assert_eq!(crc8(b"123456789"), 0xF4);
        assert_eq!(crc8(&[0; 33]), 0);
    }

    #[test]
    fn configuration_reports_version_and_mode_and_refuses_the_unknown() {
        let mut mcu = Mcu::new();
        assert!(mcu.configure(&config(&[0, 0]), FIRMWARE).is_none());
        mcu.power(true);
        let state = mcu.configure(&config(&[0, 0]), FIRMWARE).unwrap();
        assert_eq!(state.len(), CONFIG_ANSWER_SIZE);
        assert_eq!(state[..8], [0x01, 0, 0, 0x00, 0x09, 0x00, 0x20, STANDBY]);
        assert_eq!(crc8(&state), 0);
        let state = mcu.configure(&config(&[0x00, NFC]), FIRMWARE).unwrap();
        assert_eq!(state[7], NFC);
        // A bad checksum, and the infrared mode, are refused.
        let mut bad = config(&[0, 0]);
        bad[REQUEST_SIZE - 1] ^= 1;
        assert!(mcu.configure(&bad, FIRMWARE).is_none());
        assert!(mcu.configure(&config(&[0x00, 0x05]), FIRMWARE).is_none());
    }

    #[test]
    fn polling_finds_a_tag_and_notices_it_leave() {
        let mut mcu = nfc_mode();
        let tag = ntag::tests::blank(UID);
        mcu.request(
            &nfc(START_POLLING, 0, 0, LAST, &[0, 0, 0, 0x2C, 0x01]),
            &mut None,
            Duration::ZERO,
        )
        .unwrap();
        let answer = mcu.answer(None, FIRMWARE, Duration::ZERO);
        assert!(checks_out(&answer));
        assert_eq!(answer[..8], [NFC_STATE, 0, 5, 0, 0, 0x09, 0x31, POLLING]);
        let answer = mcu.answer(Some(&tag), FIRMWARE, Duration::ZERO);
        assert!(checks_out(&answer));
        assert_eq!(answer[7], TAG_FOUND);
        // Present, type 2, NFC-A, 7 bytes of serial number.
        assert_eq!(answer[11], 1);
        assert_eq!(answer[13..16], [TYPE_2, NFC_A, 7]);
        assert_eq!(answer[16..23], UID);
        let answer = mcu.answer(None, FIRMWARE, Duration::ZERO);
        assert_eq!(answer[7], TAG_LOST);
        assert_eq!(answer[11], 0);
        let answer = mcu.answer(Some(&tag), FIRMWARE, Duration::ZERO);
        assert_eq!(answer[7], TAG_FOUND);
        mcu.request(
            &nfc(STOP_POLLING, 0, 0, LAST, &[]),
            &mut None,
            Duration::ZERO,
        )
        .unwrap();
        let answer = mcu.answer(Some(&tag), FIRMWARE, Duration::ZERO);
        assert_eq!(answer[7], IDLE);
    }

    /// The presence check `hid` makes after a find: poll for 1 s again;
    /// with the tag gone that poll ends in the lost state.
    #[test]
    fn a_poll_with_a_time_limit_that_finds_nothing_ends_lost() {
        let mut mcu = nfc_mode();
        let tag = ntag::tests::blank(UID);
        let check = nfc(START_POLLING, 0, 0, LAST, &[0, 0xE8, 0x03, 0x2C, 0x01]);
        let start = Duration::from_secs(5);
        mcu.request(&check, &mut None, start).unwrap();
        assert_eq!(mcu.answer(Some(&tag), FIRMWARE, start)[7], TAG_FOUND);
        mcu.request(&check, &mut None, start).unwrap();
        let before = start + Duration::from_millis(999);
        assert_eq!(mcu.answer(None, FIRMWARE, before)[7], POLLING);
        let after = start + Duration::from_secs(1);
        assert_eq!(mcu.answer(None, FIRMWARE, after)[7], TAG_LOST);
        // Without a limit it polls on.
        mcu.request(&nfc(START_POLLING, 0, 0, LAST, &[0; 5]), &mut None, start)
            .unwrap();
        assert_eq!(mcu.answer(None, FIRMWARE, after * 10)[7], POLLING);
    }

    #[test]
    fn a_read_comes_in_numbered_packets_then_reports_done() {
        let mut mcu = nfc_mode();
        let mut tag = Some(ntag::tests::blank(UID));
        mcu.request(
            &nfc(START_POLLING, 0, 0, LAST, &[0; 5]),
            &mut tag,
            Duration::ZERO,
        )
        .unwrap();
        let answer = mcu.answer(tag.as_ref(), FIRMWARE, Duration::ZERO);
        assert_eq!(answer[7], TAG_FOUND);
        // The three runs of pages amiibo are read in.
        let read = [
            0xD0, 0x07, 0, 0, 0, 0, 0, 0, 0, 0x01, 3, 0x00, 0x3B, 0x3C, 0x77, 0x78, 0x86, 0, 0,
        ];
        mcu.request(&nfc(READ_NTAG, 0, 0, LAST, &read), &mut tag, Duration::ZERO)
            .unwrap();
        let mut received: Vec<u8> = Vec::new();
        for taken in 0..2 {
            let packet = mcu.answer(tag.as_ref(), FIRMWARE, Duration::ZERO);
            assert!(checks_out(&packet));
            // Sent again until taken.
            let again = mcu.answer(tag.as_ref(), FIRMWARE, Duration::ZERO);
            assert_eq!(packet, again);
            assert_eq!(packet[..5], [NFC_DATA, 0, CARRIES_READ_DATA, taken + 1, 0]);
            let length = usize::from(packet[5] & 7) << 8 | usize::from(packet[6]);
            assert_eq!(packet[5] & LAST != 0, taken == 1);
            received.extend(&packet[7..7 + length]);
            mcu.request(
                &nfc(NEXT, 0, taken + 1, LAST, &[]),
                &mut tag,
                Duration::ZERO,
            )
            .unwrap();
        }
        assert_eq!(received.len(), 60 + ntag::SIZE);
        assert_eq!(
            received[5..15],
            [TYPE_2, NFC_A, 7, 0x04, 0xA1, 0xB2, 0xC3, 0xD4, 0xE5, 0xF6]
        );
        assert_eq!(received[51..58], [3, 0x00, 0x3B, 0x3C, 0x77, 0x78, 0x86]);
        let pages = &received[60..];
        assert_eq!(pages[..0x85 * 4], tag.as_ref().unwrap().dump()[..0x85 * 4]);
        let answer = mcu.answer(tag.as_ref(), FIRMWARE, Duration::ZERO);
        assert_eq!(answer[7], READ_DONE);
    }

    #[test]
    fn a_write_is_taken_packet_by_packet_and_ends_written() {
        let mut mcu = nfc_mode();
        let mut tag = Some(ntag::tests::blank(UID));
        mcu.request(
            &nfc(START_POLLING, 0, 0, LAST, &[0; 5]),
            &mut tag,
            Duration::ZERO,
        )
        .unwrap();
        mcu.answer(tag.as_ref(), FIRMWARE, Duration::ZERO);
        let mut command = vec![0xD0, 0x07];
        command.extend(UID);
        command.extend([0x01, 0x00, 0x01, 0x04, 0xFF, 0xFF, 0xFF, 0xFF]);
        command.extend([0xA5, 0x00, 0x01, 0x00, 1]);
        command.extend([0x20, 0x08]);
        command.extend(1..=8);
        for (index, chunk) in command.chunks(31).enumerate() {
            let last = (index + 1) * 31 >= command.len();
            let flags = if last { LAST } else { 0 };
            mcu.request(
                &nfc(WRITE_NTAG, index as u8 + 1, 0, flags, chunk),
                &mut tag,
                Duration::ZERO,
            )
            .unwrap();
            let answer = mcu.answer(tag.as_ref(), FIRMWARE, Duration::ZERO);
            // The state report says which packet it took last.
            assert_eq!(answer[4], index as u8 + 1);
            if last {
                assert_eq!(answer[7], WRITE_DONE);
            }
        }
        let memory = tag.unwrap().dump().to_vec();
        assert_eq!(memory[16..20], [0xA5, 0x00, 0x01, 0x00]);
        assert_eq!(memory[0x80..0x88], [1, 2, 3, 4, 5, 6, 7, 8]);
        // Out of order is refused.
        let mut tag = Some(ntag::tests::blank(UID));
        let packet = nfc(WRITE_NTAG, 3, 0, 0, &[0; 31]);
        assert!(mcu.request(&packet, &mut tag, Duration::ZERO).is_err());
    }
}
