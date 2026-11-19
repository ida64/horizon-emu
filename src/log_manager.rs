//! What processes log, read off the requests they send the log manager.
//!
//! A process that logs opens a logger: command 0 (`OpenLogger`) of the `lm`
//! service returns an `ILogger`, whose command 0 (`Log`) hands over one log
//! packet in a buffer. The log manager is the real `lm` module; this module
//! only reads the packets from the client's memory, as the supervisor-call
//! trace sees each request, and turns them into lines.
//!
//! # The packet
//!
//! From Switchbrew's "Log services" page (revision 15405), which this module
//! follows:
//!
//! * A header of 0x18 bytes: process ID (8 bytes), thread ID (8), flags (1:
//!   head 1, tail 2, little-endian 4), a reserved byte, severity (1: trace,
//!   info, warn, error, fatal), verbosity (1), payload size (4).
//! * The payload: chunks of a key and a size, both LEB128, and that many
//!   bytes of data. The keys: 0 session begin, 1 session end, 2 text, 3 line
//!   number, 4 file name, 5 function name, 6 module name, 7 thread name, 8
//!   count of dropped packets, 9 user system clock, 10 process name.
//! * A message longer than one packet holds is sent as several: the first
//!   flagged head, the last flagged tail, one packet both. \[inference\] Their
//!   text chunks are parts of one text, in order, and the other fields come
//!   once; packets of different threads are told apart by process and
//!   thread ID.
//!
//! What firmware 23.5.0 itself shows: every system module opens a logger
//! (the trace of a boot to HOME has nvnflinger, nvservices, ns, ptm, audio
//! and the others send `OpenLogger` with their process ID and get an
//! object; am also sends `ILogger` command 11), and `lm` answers
//! `QueryPointerBufferSize` with 0x400, so a packet in a pointer buffer is
//! at most that long. None of them sends `Log`, through a boot, HOME,
//! System Settings and the eShop applet: their logging is compiled out.
//! The log manager is `LogManager.Prod`, whose only service names are `lm`
//! (which it registers) and `psc:m`: it has no session to the filesystem
//! or to settings, so no setting makes it keep logs on the microSD card.
//! The packets of the tests are homebrew's, which builds them as above and
//! which `lm` accepts.
//!
//! \[assumption\] Numbers in chunks (line number, drop count, clock) are in
//! the byte order the little-endian flag names, and the header is
//! little-endian regardless (it is the processor's own order).

use std::collections::HashMap;

use crate::ipc::{Action, BufferKind, Contents, Message, Request};

/// What [`crate::ipc::Tracker`] calls an `ILogger`: the object or session
/// that command 0 (`OpenLogger`) of `lm` returns.
pub const LOGGER: &str = "lm#0";

/// `ILogger` command 0: `Log`, the packet in a buffer.
const LOG: u32 = 0;

/// Size of the packet header.
pub const HEADER_SIZE: usize = 0x18;

/// Flags of the packet header.
pub const HEAD: u8 = 1;
pub const TAIL: u8 = 2;
pub const LITTLE_ENDIAN: u8 = 4;

/// Keys of the payload's chunks.
pub const SESSION_BEGIN: u64 = 0;
pub const SESSION_END: u64 = 1;
pub const TEXT: u64 = 2;
pub const LINE_NUMBER: u64 = 3;
pub const FILE_NAME: u64 = 4;
pub const FUNCTION_NAME: u64 = 5;
pub const MODULE_NAME: u64 = 6;
pub const THREAD_NAME: u64 = 7;
pub const DROP_COUNT: u64 = 8;
pub const USER_SYSTEM_CLOCK: u64 = 9;
pub const PROCESS_NAME: u64 = 10;

/// The packet that a request a process sends to [`LOGGER`] carries, when
/// the request is `Log`: the start of the one buffer it hands over, as far
/// as `contents` has it. `data` is the message.
pub fn logged_packet<'c>(data: &[u8], contents: &'c [Contents]) -> Option<&'c [u8]> {
    let message = Message::parse(data)?;
    if Request::parse(&message)?.action != Action::Request(LOG) {
        return None;
    }
    // The buffer is of the kind the client picks (pointer or mapped send),
    // the other one of the pair left empty.
    contents
        .iter()
        .find(|contents| {
            contents.buffer.size != 0
                && matches!(
                    contents.buffer.kind,
                    BufferKind::Pointer { .. } | BufferKind::Send
                )
        })
        .map(|contents| contents.bytes.as_slice())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    pub process_id: u64,
    pub thread_id: u64,
    pub flags: u8,
    pub severity: u8,
    pub verbosity: u8,
    pub payload_size: u32,
}

/// One key and its data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Chunk<'a> {
    pub key: u64,
    pub data: &'a [u8],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// Fewer bytes than a header.
    NoHeader,
    /// The payload the header announces goes beyond the bytes there are.
    PayloadBeyondPacket { payload_size: u32, available: usize },
    /// A chunk's key, size or data goes beyond the payload.
    ChunkBeyondPayload { offset: usize },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoHeader => write!(f, "shorter than a packet header"),
            Self::PayloadBeyondPacket {
                payload_size,
                available,
            } => write!(
                f,
                "payload of {payload_size:#x} bytes, only {available:#x} there"
            ),
            Self::ChunkBeyondPayload { offset } => {
                write!(f, "chunk at payload offset {offset:#x} runs beyond it")
            }
        }
    }
}

/// A decoded packet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Packet<'a> {
    pub header: Header,
    pub chunks: Vec<Chunk<'a>>,
}

/// Reads a LEB128 number at `*offset`, moving past it; `None` if it runs
/// beyond `bytes` or beyond 64 bits.
fn leb128(bytes: &[u8], offset: &mut usize) -> Option<u64> {
    let mut value = 0u64;
    for shift in (0..64).step_by(7) {
        let byte = *bytes.get(*offset)?;
        *offset += 1;
        value |= u64::from(byte & 0x7F) << shift;
        if byte & 0x80 == 0 {
            return Some(value);
        }
    }
    None
}

impl<'a> Packet<'a> {
    /// Decodes the packet at the start of `bytes`, which may go on beyond
    /// it (the buffer it came in).
    pub fn parse(bytes: &'a [u8]) -> Result<Self, Error> {
        let header = bytes.first_chunk::<HEADER_SIZE>().ok_or(Error::NoHeader)?;
        let u64_at = |at: usize| u64::from_le_bytes(header[at..at + 8].try_into().unwrap());
        let header = Header {
            process_id: u64_at(0),
            thread_id: u64_at(8),
            flags: header[0x10],
            severity: header[0x12],
            verbosity: header[0x13],
            payload_size: u32::from_le_bytes(header[0x14..0x18].try_into().unwrap()),
        };
        let rest = &bytes[HEADER_SIZE..];
        let payload =
            rest.get(..header.payload_size as usize)
                .ok_or(Error::PayloadBeyondPacket {
                    payload_size: header.payload_size,
                    available: rest.len(),
                })?;
        let mut chunks = Vec::new();
        let mut offset = 0;
        while offset < payload.len() {
            let start = offset;
            let beyond = Error::ChunkBeyondPayload { offset: start };
            let key = leb128(payload, &mut offset).ok_or(beyond)?;
            let size = leb128(payload, &mut offset).ok_or(beyond)?;
            let end = usize::try_from(size)
                .ok()
                .and_then(|size| offset.checked_add(size))
                .filter(|&end| end <= payload.len())
                .ok_or(beyond)?;
            chunks.push(Chunk {
                key,
                data: &payload[offset..end],
            });
            offset = end;
        }
        Ok(Self { header, chunks })
    }
}

/// A message, from the packets that make it up.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct LogMessage {
    pub process_id: u64,
    pub thread_id: u64,
    pub severity: u8,
    pub verbosity: u8,
    pub text: Vec<u8>,
    pub line_number: Option<u64>,
    pub file: Option<String>,
    pub function: Option<String>,
    pub module: Option<String>,
    pub thread: Option<String>,
    pub process: Option<String>,
    pub drop_count: Option<u64>,
    pub user_system_clock: Option<u64>,
    pub session_begin: bool,
    pub session_end: bool,
    /// Chunks of keys this module does not know, by key.
    pub unknown: Vec<u64>,
    /// The first packet seen was not flagged head: the start was missed.
    pub start_missed: bool,
}

/// A number in a chunk, `None` if the chunk is not one to eight bytes.
fn number(data: &[u8], little_endian: bool) -> Option<u64> {
    if data.is_empty() || data.len() > 8 {
        return None;
    }
    let mut bytes = [0; 8];
    Some(match little_endian {
        true => {
            bytes[..data.len()].copy_from_slice(data);
            u64::from_le_bytes(bytes)
        }
        false => {
            bytes[8 - data.len()..].copy_from_slice(data);
            u64::from_be_bytes(bytes)
        }
    })
}

/// A name in a chunk, without the NUL the sender may end it with.
fn name(data: &[u8]) -> String {
    let end = data
        .iter()
        .position(|&byte| byte == 0)
        .unwrap_or(data.len());
    String::from_utf8_lossy(&data[..end]).into_owned()
}

impl LogMessage {
    fn add(&mut self, packet: &Packet) {
        let little_endian = packet.header.flags & LITTLE_ENDIAN != 0;
        for chunk in &packet.chunks {
            let data = chunk.data;
            match chunk.key {
                SESSION_BEGIN => self.session_begin = true,
                SESSION_END => self.session_end = true,
                TEXT => {
                    let end = data
                        .iter()
                        .position(|&byte| byte == 0)
                        .unwrap_or(data.len());
                    self.text.extend_from_slice(&data[..end]);
                }
                LINE_NUMBER => self.line_number = number(data, little_endian),
                FILE_NAME => self.file = Some(name(data)),
                FUNCTION_NAME => self.function = Some(name(data)),
                MODULE_NAME => self.module = Some(name(data)),
                THREAD_NAME => self.thread = Some(name(data)),
                DROP_COUNT => self.drop_count = number(data, little_endian),
                USER_SYSTEM_CLOCK => self.user_system_clock = number(data, little_endian),
                PROCESS_NAME => self.process = Some(name(data)),
                other => self.unknown.push(other),
            }
        }
    }

    /// The trace lines of this message: `log: PROCESS SEVERITY [MODULE]
    /// FILE:LINE FUNCTION: TEXT`, one per line of text, leaving out what
    /// the message does not have. `caller` names the process where the
    /// message does not.
    pub fn lines(&self, caller: &str) -> Vec<String> {
        let mut prefix = format!(
            "log: {} {}",
            self.process.as_deref().unwrap_or(caller),
            severity_name(self.severity)
        );
        if let Some(module) = &self.module {
            prefix += &format!(" [{module}]");
        }
        match (&self.file, self.line_number) {
            (Some(file), Some(line)) => prefix += &format!(" {file}:{line}"),
            (Some(file), None) => prefix += &format!(" {file}"),
            (None, Some(line)) => prefix += &format!(" line {line}"),
            (None, None) => {}
        }
        if let Some(function) = &self.function {
            prefix += &format!(" {function}");
        }
        let mut notes = Vec::new();
        if let Some(thread) = &self.thread {
            notes.push(format!("thread {thread}"));
        }
        if self.session_begin {
            notes.push("logging begins".into());
        }
        if self.session_end {
            notes.push("logging ends".into());
        }
        if let Some(count) = self.drop_count {
            notes.push(format!("{count} packets dropped"));
        }
        if let Some(clock) = self.user_system_clock {
            notes.push(format!("user system clock {clock}"));
        }
        if !self.unknown.is_empty() {
            notes.push(format!("chunks of unknown keys {:?}", self.unknown));
        }
        if self.start_missed {
            notes.push("start missed".into());
        }
        if !notes.is_empty() {
            prefix += &format!(" ({})", notes.join(", "));
        }
        let text = String::from_utf8_lossy(&self.text);
        let text = text.trim_end_matches(['\n', '\r']);
        if text.is_empty() {
            return vec![prefix];
        }
        text.lines()
            .map(|line| format!("{prefix}: {line}"))
            .collect()
    }
}

/// Name of a severity, as the trace lines show it.
pub fn severity_name(severity: u8) -> String {
    match severity {
        0 => "trace".into(),
        1 => "info".into(),
        2 => "warn".into(),
        3 => "error".into(),
        4 => "fatal".into(),
        other => format!("severity {other}"),
    }
}

/// Puts messages back together from the packets they were sent in.
#[derive(Debug, Default)]
pub struct Reassembler {
    /// Messages begun and not ended, by process and thread ID.
    partial: HashMap<(u64, u64), LogMessage>,
}

impl Reassembler {
    /// Takes the next packet; the messages it completes: its own if it is
    /// flagged tail. A head packet also ends what an earlier packet of the
    /// same thread began and never ended: that message comes first, as far
    /// as it got, so nothing logged goes unseen.
    pub fn push(&mut self, packet: &Packet) -> Vec<LogMessage> {
        let header = packet.header;
        let key = (header.process_id, header.thread_id);
        let mut done = Vec::new();
        if header.flags & HEAD != 0
            && let Some(unended) = self.partial.remove(&key)
        {
            done.push(unended);
        }
        let message = self.partial.entry(key).or_insert_with(|| LogMessage {
            process_id: header.process_id,
            thread_id: header.thread_id,
            severity: header.severity,
            verbosity: header.verbosity,
            start_missed: header.flags & HEAD == 0,
            ..LogMessage::default()
        });
        message.add(packet);
        if header.flags & TAIL != 0
            && let Some(message) = self.partial.remove(&key)
        {
            done.push(message);
        }
        done
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipc::Buffer;

    /// The packets homebrew (`lmtest`, process ID 0x83, thread ID 0x236)
    /// sent `ILogger::Log` on firmware 23.5.0, as `--log` read them from
    /// its memory: one message in one packet, then one in two.
    const ONE: &[u8] = &[
        0x83, 0, 0, 0, 0, 0, 0, 0, 0x36, 0x02, 0, 0, 0, 0, 0, 0, 0x07, 0x00, 0x01, 0x00, 0x4b,
        0x00, 0x00, 0x00, // header
        0x03, 0x04, 0x40, 0x00, 0x00, 0x00, // line 64
        0x04, 0x13, b'/', b'w', b'o', b'r', b'k', b'/', b's', b'o', b'u', b'r', b'c', b'e', b'/',
        b'm', b'a', b'i', b'n', b'.', b'c', // file
        0x05, 0x04, b'm', b'a', b'i', b'n', // function
        0x06, 0x06, b'l', b'm', b't', b'e', b's', b't', // module
        0x07, 0x04, b'm', b'a', b'i', b'n', // thread
        0x02, 0x1a, b'o', b'n', b'e', b' ', b'p', b'a', b'c', b'k', b'e', b't', b',', b' ', b's',
        b'e', b'v', b'e', b'r', b'i', b't', b'y', b' ', b'i', b'n', b'f', b'o', b'\n', // text
    ];
    const FIRST: &[u8] = &[
        0x83, 0, 0, 0, 0, 0, 0, 0, 0x36, 0x02, 0, 0, 0, 0, 0, 0, 0x05, 0x00, 0x02, 0x00, 0x3c,
        0x00, 0x00, 0x00, // header
        0x03, 0x04, 0x4c, 0x00, 0x00, 0x00, // line 76
        0x04, 0x13, b'/', b'w', b'o', b'r', b'k', b'/', b's', b'o', b'u', b'r', b'c', b'e', b'/',
        b'm', b'a', b'i', b'n', b'.', b'c', // file
        0x05, 0x04, b'm', b'a', b'i', b'n', // function
        0x02, 0x19, b'f', b'i', b'r', b's', b't', b' ', b'h', b'a', b'l', b'f', b' ', b'o', b'f',
        b' ', b'a', b' ', b'w', b'a', b'r', b'n', b'i', b'n', b'g', b',', b' ', // text
    ];
    const LAST: &[u8] = &[
        0x83, 0, 0, 0, 0, 0, 0, 0, 0x36, 0x02, 0, 0, 0, 0, 0, 0, 0x06, 0x00, 0x02, 0x00, 0x20,
        0x00, 0x00, 0x00, // header
        0x02, 0x1e, b's', b'e', b'c', b'o', b'n', b'd', b' ', b'h', b'a', b'l', b'f', b'\n', b'a',
        b'n', b'd', b' ', b'a', b' ', b's', b'e', b'c', b'o', b'n', b'd', b' ', b'l', b'i', b'n',
        b'e', b'\n', // text
    ];

    /// `packet` as another thread's.
    fn of_thread(packet: &[u8], thread: u8) -> Vec<u8> {
        let mut packet = packet.to_vec();
        packet[8] = thread;
        packet
    }

    #[test]
    fn one_packet_is_one_message() {
        let packet = Packet::parse(ONE).unwrap();
        assert_eq!(
            packet.header,
            Header {
                process_id: 0x83,
                thread_id: 0x236,
                flags: HEAD | TAIL | LITTLE_ENDIAN,
                severity: 1,
                verbosity: 0,
                payload_size: 0x4b,
            }
        );
        let keys: Vec<u64> = packet.chunks.iter().map(|chunk| chunk.key).collect();
        assert_eq!(
            keys,
            [
                LINE_NUMBER,
                FILE_NAME,
                FUNCTION_NAME,
                MODULE_NAME,
                THREAD_NAME,
                TEXT
            ]
        );
        let messages = Reassembler::default().push(&packet);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].line_number, Some(64));
        assert_eq!(
            messages[0].lines("hbloader"),
            [
                "log: hbloader info [lmtest] /work/source/main.c:64 main (thread main): \
              one packet, severity info"
            ]
        );
    }

    #[test]
    fn head_and_tail_make_one_message_per_thread() {
        let mut packets = Reassembler::default();
        assert!(packets.push(&Packet::parse(FIRST).unwrap()).is_empty());
        // Another thread's message in between belongs to that thread.
        let other = of_thread(ONE, 0x40);
        let messages = packets.push(&Packet::parse(&other).unwrap());
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].thread_id, 0x240);
        let messages = packets.push(&Packet::parse(LAST).unwrap());
        assert_eq!(messages.len(), 1);
        assert_eq!(
            messages[0].lines("hbloader"),
            [
                "log: hbloader warn /work/source/main.c:76 main: first half of a warning, \
                 second half",
                "log: hbloader warn /work/source/main.c:76 main: and a second line",
            ]
        );
        // Nothing is left over.
        assert!(packets.partial.is_empty());
    }

    #[test]
    fn unended_and_unbegun_messages_are_shown_as_far_as_they_go() {
        let mut packets = Reassembler::default();
        packets.push(&Packet::parse(FIRST).unwrap());
        // A new head on the same thread: the first message never ended.
        let messages = packets.push(&Packet::parse(FIRST).unwrap());
        assert_eq!(
            messages[0].lines("hbloader"),
            ["log: hbloader warn /work/source/main.c:76 main: first half of a warning, "]
        );
        // A tail whose head was missed.
        let messages = Reassembler::default().push(&Packet::parse(LAST).unwrap());
        assert_eq!(
            messages[0].lines("hbloader")[0],
            "log: hbloader warn (start missed): second half"
        );
    }

    #[test]
    fn packets_cut_short_are_refused() {
        assert_eq!(Packet::parse(&ONE[..0x17]), Err(Error::NoHeader));
        assert_eq!(
            Packet::parse(&ONE[..ONE.len() - 1]),
            Err(Error::PayloadBeyondPacket {
                payload_size: 0x4b,
                available: 0x4a
            })
        );
        // The header says less than the chunks take.
        let mut short = ONE.to_vec();
        short[0x14] = 0x4a;
        assert_eq!(
            Packet::parse(&short),
            Err(Error::ChunkBeyondPayload { offset: 0x2f })
        );
        // A buffer longer than the packet is fine.
        let mut long = ONE.to_vec();
        long.extend_from_slice(&[0xAA; 8]);
        assert_eq!(Packet::parse(&long), Packet::parse(ONE));
    }

    #[test]
    fn the_packet_is_the_buffer_of_a_log_request() {
        // Type 4 with a pointer buffer (index 0) of the packet's size and an
        // empty send buffer: the pair a HipcAutoSelect buffer makes.
        let words = [
            4 | 1 << 16 | 1 << 20,
            8,
            (ONE.len() as u32) << 16,
            0x1000,
            0,
            0,
            0,
            0,
            u32::from_le_bytes(*b"SFCI"),
            1,
            LOG,
            0,
            0,
            0,
            0,
        ];
        let message: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        let buffer = Buffer {
            kind: BufferKind::Pointer { index: 0 },
            address: 0x1000,
            size: ONE.len() as u64,
        };
        let contents = [Contents {
            buffer,
            bytes: ONE.to_vec(),
        }];
        assert_eq!(logged_packet(&message, &contents), Some(ONE));
        // Another command carries no packet.
        let mut other = message.clone();
        other[40] = 1;
        assert_eq!(logged_packet(&other, &contents), None);
    }
}
