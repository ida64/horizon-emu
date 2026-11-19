//! Messages between processes, for traces.
//!
//! A process talks to a service by putting a message in a buffer and naming a
//! session handle to the kernel (`SendSyncRequest` and its variants); when
//! the server has answered, the reply is in the same buffer. This module
//! reads such buffers, in three layers:
//!
//! * [`Message`] is the envelope, which the kernel itself interprets: the
//!   handles and buffers that travel with the message, and where its raw
//!   data is.
//! * [`Request`] and [`Reply`] are what the two ends put in the raw data. The
//!   firmware has two conventions. CMIF starts the data with an `SFCI`
//!   (request) or `SFCO` (reply) header, optionally preceded by a domain
//!   header that selects one of several objects behind the session. TIPC has
//!   no headers: the command is the envelope's type, the result the first
//!   data word.
//! * [`Tracker`] follows each process's handles through the calls it makes,
//!   so a trace can say which service a request goes to.
//!
//! # Where the layouts come from
//!
//! Firmware 23.5.0, by image address (kernel) or text offset (initial
//! processes):
//!
//! * Envelope: the kernel's server receive path at 0x8009fdb0 (the count
//!   fields of both header words, the special header, how descriptors pack
//!   addresses and sizes, the receive list). sm computes the same offsets to
//!   find its reply's data at 0x1c0.
//! * CMIF without a domain: ProcessMana 0xa060 builds a type 6 message whose
//!   data, aligned to 16 bytes from the start of the message, is `SFCI`, a
//!   version, the command and a token, then the arguments; it expects `SFCO`
//!   at the same place in the reply with the result 8 bytes on and the
//!   outputs 16 bytes on.
//! * CMIF with a domain: ProcessMana 0x91b0 and 0x7ed0 put 16 bytes before
//!   that header (a byte 1, a byte 0, the 16-bit size of header plus
//!   arguments, the object ID) and expect `SFCO` 16 bytes into the reply.
//! * TIPC: sm's dispatchers at 0xf80 and 0x1240 switch on the envelope type
//!   (16 is command 0), read arguments from the first data word on and write
//!   the result there; 0x9c30 treats type 15 as closing the session.
//! * sm's commands: 0xf80. Command 1 takes eight bytes of name and answers
//!   with a moved handle, the session to that service; command 2 takes a
//!   name, a session limit and a flag and answers with a moved handle;
//!   command 3 takes a name. The same commands come in CMIF from homebrew
//!   (libnx on hbloader: command 0 with the process ID, then command 1 with
//!   a name, answered with a moved handle), and sm answers them.
//!
//! From public documentation of the formats rather than from these binaries:
//! the CMIF envelope types other than 6 (1 and 4 request, 3, 5 and 7
//! control, 2 close), the control commands (0 makes the session a domain and
//! returns the ID of its object, 1 returns a session for a domain object, 2
//! and 4 return another session to the same object), the count of objects
//! passed in a domain request (second byte of the domain header, their IDs
//! after the arguments), domain header type 2 (release the object), and the
//! objects a domain reply returns (count in the first word of the 16 bytes
//! before `SFCO`). The first seconds of a boot agree with what they exercise
//! of this: boot's control 0 to fsp-srv returns 1 and its requests then name
//! object 1; the ID fsp-srv's command 12 returns is the object of boot's next
//! request and, once released, is returned again; the handle NCM gets from
//! control 2 takes requests for the object of the session it came from.
//!
//! Inferred: how long the arguments or outputs are where no header says. A
//! sender reserves 16 bytes of raw data for the alignment and uses the front
//! of them, so the contents end where the unused rest begins. The requests
//! ProcessMana builds have exactly that size, as do the answers inspected
//! from spl and FS. The IDs of returned objects are taken to be the last
//! words of the contents (FS's answer to command 12 has one, and nothing
//! else, there).

use std::{
    collections::HashMap,
    fmt::{self, Write as _},
};

use crate::svc;

const SFCI: u32 = u32::from_le_bytes(*b"SFCI");
const SFCO: u32 = u32::from_le_bytes(*b"SFCO");

/// CMIF aligns its headers to this, counted from the start of the message,
/// and reserves as many bytes of raw data for doing so.
const CMIF_ALIGNMENT: usize = 16;
/// Size of the `SFCI`/`SFCO` header and of each domain header.
const CMIF_HEADER_SIZE: usize = 16;

const CMIF_CLOSE: u16 = 2;
const TIPC_CLOSE: u16 = 15;
/// Envelope type of TIPC command 0.
const TIPC_FIRST_COMMAND: u16 = 16;

const CONTROL_CONVERT_TO_DOMAIN: u32 = 0;
const CONTROL_COPY_FROM_DOMAIN: u32 = 1;
const CONTROL_CLONE: u32 = 2;
const CONTROL_CLONE_EX: u32 = 4;

/// Name of the port sm serves its lookups on.
const SM_PORT: &str = "sm:";
const SM_GET_SERVICE: u32 = 1;
const SM_REGISTER_SERVICE: u32 = 2;
const SM_UNREGISTER_SERVICE: u32 = 3;

/// How many data words and buffers a trace line shows before eliding.
const SHOWN_WORDS: usize = 12;
const SHOWN_BUFFERS: usize = 8;

/// A result code as services and the kernel return it: zero for success,
/// else a module (low 9 bits) and a description (next 13 bits), displayed in
/// the `2mmm-dddd` form error screens use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResultCode(pub u32);

impl fmt::Display for ResultCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 == 0 {
            return f.write_str("ok");
        }
        write!(f, "2{:03}-{:04}", self.0 & 0x1FF, self.0 >> 9 & 0x1FFF)
    }
}

/// How a buffer named by a message reaches the other side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferKind {
    /// Copied into a buffer the receiver offers in its receive list; `index`
    /// says which.
    Pointer { index: u8 },
    /// Mapped into the receiver, to read.
    Send,
    /// Mapped into the receiver, to fill.
    Receive,
    /// Mapped into the receiver, to read and fill.
    Exchange,
    /// Offered for the pointer data of the answer.
    ReceiveList,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Buffer {
    pub kind: BufferKind,
    pub address: u64,
    pub size: u64,
}

impl fmt::Display for Buffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            BufferKind::Pointer { index } => write!(f, "pointer#{index}")?,
            BufferKind::Send => f.write_str("send")?,
            BufferKind::Receive => f.write_str("receive")?,
            BufferKind::Exchange => f.write_str("exchange")?,
            BufferKind::ReceiveList => f.write_str("receive-list")?,
        }
        write!(f, " {:#x}+{:#x}", self.address, self.size)
    }
}

/// How much of a buffer a trace line shows, from its start.
pub const SHOWN_BUFFER_BYTES: u64 = 32;

/// The start of a buffer a message names, as it is in the client's memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Contents {
    pub buffer: Buffer,
    pub bytes: Vec<u8>,
}

/// The buffers the message in `data` names that there is something to see
/// in: first those whose contents are there when the message is (what a
/// request hands over, and the pointer data of an answer), then those of a
/// request that its answer fills.
pub fn carried(data: &[u8]) -> (Vec<Buffer>, Vec<Buffer>) {
    let buffers = Message::parse(data).map_or(Vec::new(), |message| message.buffers);
    let of = |kinds: fn(BufferKind) -> bool| {
        buffers
            .iter()
            .copied()
            .filter(|buffer| buffer.size != 0 && kinds(buffer.kind))
            .collect()
    };
    (
        of(|kind| {
            matches!(
                kind,
                BufferKind::Pointer { .. } | BufferKind::Send | BufferKind::Exchange
            )
        }),
        of(|kind| matches!(kind, BufferKind::Receive | BufferKind::Exchange)),
    )
}

/// The envelope of a message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message<'a> {
    /// What the message is, in the convention of the session's server.
    pub kind: u16,
    /// The sender asks the kernel to tell the receiver its process ID.
    pub sends_process_id: bool,
    /// Handles the receiver gets a copy of.
    pub copy_handles: Vec<u32>,
    /// Handles the sender gives away.
    pub move_handles: Vec<u32>,
    pub buffers: Vec<Buffer>,
    /// Offset of the raw data from the start of the message.
    pub raw_offset: usize,
    pub raw: &'a [u8],
    /// The header describes more than the given bytes hold; what is here is
    /// the part that could be read.
    pub truncated: bool,
}

/// Little-endian words of a message, read in order.
struct Words<'a> {
    bytes: &'a [u8],
    index: usize,
}

impl Words<'_> {
    fn take(&mut self) -> Option<u32> {
        let word = word_at(self.bytes, self.index * 4)?;
        self.index += 1;
        Some(word)
    }
}

fn word_at(bytes: &[u8], offset: usize) -> Option<u32> {
    let word = bytes.get(offset..)?.first_chunk::<4>()?;
    Some(u32::from_le_bytes(*word))
}

impl<'a> Message<'a> {
    /// Decodes the envelope at the start of `bytes`, which may extend beyond
    /// the message. `None` if not even the two header words are there.
    pub fn parse(bytes: &'a [u8]) -> Option<Self> {
        let mut words = Words { bytes, index: 0 };
        let (first, second) = (words.take()?, words.take()?);
        let mut message = Self {
            kind: first as u16,
            sends_process_id: false,
            copy_handles: Vec::new(),
            move_handles: Vec::new(),
            buffers: Vec::new(),
            raw_offset: 8,
            raw: &[],
            truncated: false,
        };
        message.truncated = message.parse_body(&mut words, first, second).is_none();
        Some(message)
    }

    /// Everything after the two header words; `None` when the bytes run out.
    fn parse_body(&mut self, words: &mut Words<'a>, first: u32, second: u32) -> Option<()> {
        if second >> 31 != 0 {
            let special = words.take()?;
            self.sends_process_id = special & 1 != 0;
            if self.sends_process_id {
                // Room for the ID, which the kernel fills in for the receiver.
                words.take()?;
                words.take()?;
            }
            for _ in 0..special >> 1 & 0xF {
                self.copy_handles.push(words.take()?);
            }
            for _ in 0..special >> 5 & 0xF {
                self.move_handles.push(words.take()?);
            }
        }
        for _ in 0..first >> 16 & 0xF {
            let (packed, low) = (words.take()?, words.take()?);
            self.buffers.push(Buffer {
                kind: BufferKind::Pointer {
                    index: (packed & 0xF) as u8,
                },
                address: u64::from(low)
                    | u64::from(packed >> 12 & 0xF) << 32
                    | u64::from(packed >> 6 & 0x3F) << 36,
                size: u64::from(packed >> 16),
            });
        }
        let mapped = [
            (20, BufferKind::Send),
            (24, BufferKind::Receive),
            (28, BufferKind::Exchange),
        ];
        for (shift, kind) in mapped {
            for _ in 0..first >> shift & 0xF {
                let (size, low, packed) = (words.take()?, words.take()?, words.take()?);
                self.buffers.push(Buffer {
                    kind,
                    address: u64::from(low)
                        | u64::from(packed >> 28) << 32
                        | u64::from(packed >> 2 & 0x3F) << 36,
                    size: u64::from(size) | u64::from(packed >> 24 & 0xF) << 32,
                });
            }
        }

        let raw_size = (second & 0x3FF) as usize * 4;
        self.raw_offset = words.index * 4;
        let rest = words.bytes.get(self.raw_offset..)?;
        self.raw = &rest[..raw_size.min(rest.len())];

        // Mode 0 offers nothing and mode 1 the message buffer itself; from 2
        // on there are entries, after the raw data unless an offset says
        // where.
        let mode = second >> 10 & 0xF;
        if mode >= 2 {
            let offset = (second >> 20 & 0x7FF) as usize;
            words.index = if offset != 0 {
                offset
            } else {
                (self.raw_offset + raw_size) / 4
            };
            for _ in 0..(mode - 2).max(1) {
                let (low, packed) = (words.take()?, words.take()?);
                self.buffers.push(Buffer {
                    kind: BufferKind::ReceiveList,
                    address: u64::from(low) | u64::from(packed & 0x3FF) << 32,
                    size: u64::from(packed >> 16),
                });
            }
        }
        (self.raw.len() == raw_size).then_some(())
    }

    /// The raw data from where CMIF, which aligns to the start of the
    /// message, puts its headers; and how much of that its contents take.
    /// The sender reserved [`CMIF_ALIGNMENT`] bytes of raw data for
    /// alignment: what it did not need in front is left over behind.
    fn cmif_body(&self) -> Option<(&'a [u8], usize)> {
        let padding = self.raw_offset.wrapping_neg() % CMIF_ALIGNMENT;
        let body = self.raw.get(padding..)?;
        Some((body, body.len().saturating_sub(CMIF_ALIGNMENT - padding)))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Protocol {
    Cmif,
    Tipc,
}

impl fmt::Display for Protocol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Cmif => "cmif",
            Self::Tipc => "tipc",
        })
    }
}

/// What a request asks of the other side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// A command of the service, or of the object the request names.
    Request(u32),
    /// A command to the server's session handling rather than to the service.
    Control(u32),
    /// The client is done with the session.
    Close,
    /// The client is done with the object the request names.
    ReleaseObject,
}

impl fmt::Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Request(command) => write!(f, "request {command}"),
            Self::Control(command) => write!(f, "control {command}"),
            Self::Close => f.write_str("close"),
            Self::ReleaseObject => f.write_str("release"),
        }
    }
}

/// The raw data of a message sent to a server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request<'a> {
    pub protocol: Protocol,
    pub action: Action,
    /// The object the request is for, when the session is a domain.
    pub object: Option<u32>,
    /// Objects of the domain passed as arguments.
    pub in_objects: Vec<u32>,
    /// The arguments, to the next multiple of four bytes.
    pub data: &'a [u8],
}

impl<'a> Request<'a> {
    /// `None` when the message is in neither convention.
    pub fn parse(message: &Message<'a>) -> Option<Self> {
        let plain = |protocol, action, data| Self {
            protocol,
            action,
            object: None,
            in_objects: Vec::new(),
            data,
        };
        let control = match message.kind {
            TIPC_CLOSE => return Some(plain(Protocol::Tipc, Action::Close, &[])),
            TIPC_FIRST_COMMAND.. => {
                let command = u32::from(message.kind - TIPC_FIRST_COMMAND);
                return Some(plain(Protocol::Tipc, Action::Request(command), message.raw));
            }
            CMIF_CLOSE => return Some(plain(Protocol::Cmif, Action::Close, &[])),
            1 | 4 | 6 => false,
            3 | 5 | 7 => true,
            _ => return None,
        };
        let action = |command| {
            if control {
                Action::Control(command)
            } else {
                Action::Request(command)
            }
        };

        let (body, end) = message.cmif_body()?;
        let first = word_at(body, 0)?;
        if first == SFCI {
            let data = body.get(CMIF_HEADER_SIZE..end.max(CMIF_HEADER_SIZE))?;
            return Some(plain(Protocol::Cmif, action(word_at(body, 8)?), data));
        }
        let object = Some(word_at(body, 4)?);
        match first & 0xFF {
            1 if word_at(body, CMIF_HEADER_SIZE) == Some(SFCI) => {
                // The size covers the `SFCI` header and the arguments.
                let end = CMIF_HEADER_SIZE + (first >> 16) as usize;
                let data = body.get(2 * CMIF_HEADER_SIZE..end.min(body.len()))?;
                Some(Self {
                    protocol: Protocol::Cmif,
                    action: action(word_at(body, CMIF_HEADER_SIZE + 8)?),
                    object,
                    in_objects: (0..(first >> 8 & 0xFF) as usize)
                        .map_while(|index| word_at(body, end + index * 4))
                        .collect(),
                    data,
                })
            }
            2 => Some(Self {
                object,
                ..plain(Protocol::Cmif, Action::ReleaseObject, &[])
            }),
            _ => None,
        }
    }
}

/// The raw data of a server's answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reply<'a> {
    pub result: ResultCode,
    /// The outputs, to the next multiple of four bytes.
    pub data: &'a [u8],
    /// Objects of the domain the answer returns.
    pub objects: Vec<u32>,
}

impl<'a> Reply<'a> {
    /// Reads the answer to a request made in `protocol`. `None` when it is
    /// not in that convention.
    pub fn parse(message: &Message<'a>, protocol: Protocol) -> Option<Self> {
        if protocol == Protocol::Tipc {
            return Some(Self {
                result: ResultCode(word_at(message.raw, 0)?),
                data: &message.raw[4..],
                objects: Vec::new(),
            });
        }
        let (body, end) = message.cmif_body()?;
        if word_at(body, 0)? == SFCO {
            return Some(Self {
                result: ResultCode(word_at(body, 8)?),
                data: body.get(CMIF_HEADER_SIZE..end.max(CMIF_HEADER_SIZE))?,
                objects: Vec::new(),
            });
        }
        if word_at(body, CMIF_HEADER_SIZE)? != SFCO {
            return None;
        }
        let outputs = 2 * CMIF_HEADER_SIZE;
        // The IDs of the returned objects are the last of the contents.
        let ids = (word_at(body, 0)? as usize)
            .checked_mul(4)
            .and_then(|size| end.checked_sub(size))
            .filter(|&ids| ids >= outputs)
            .unwrap_or(end.max(outputs));
        Some(Self {
            result: ResultCode(word_at(body, CMIF_HEADER_SIZE + 8)?),
            data: body.get(outputs..ids).unwrap_or(&[]),
            objects: body
                .get(ids..end.max(ids))
                .unwrap_or(&[])
                .as_chunks()
                .0
                .iter()
                .map(|id| u32::from_le_bytes(*id))
                .collect(),
        })
    }
}

/// The thread a call is made on. Any two values work that are unique to the
/// process and to the thread within it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Client {
    pub process: u64,
    pub thread: u64,
}

/// What a handle of some process is a session to.
#[derive(Debug)]
struct Session {
    name: String,
    /// Handles that are this session; it is forgotten with the last.
    handles: usize,
    /// Names of the objects behind the session, once it is a domain.
    objects: Option<HashMap<u32, String>>,
}

/// A request whose answer is outstanding.
#[derive(Debug)]
struct Sent {
    session: Option<usize>,
    /// The name the request line gave its target.
    target: String,
    /// `None` when the request was in neither convention.
    request: Option<(Protocol, Action)>,
    /// First data word: the object control command 1 asks a session for.
    argument: Option<u32>,
    /// Service an sm lookup names.
    service: Option<String>,
}

#[derive(Debug)]
enum Pending {
    /// `ConnectToNamedPort` with this name.
    Connect(String),
    Request(Sent),
}

/// Follows what the session handles of every process are, from the calls
/// that create, use and close them, and words a trace line for each request
/// and each answer.
///
/// A handle gets its name from how the process came by it: the port name
/// for `ConnectToNamedPort`, the service name for a handle sm returns for a
/// lookup, and `parent#command` for a handle or domain object that a command
/// to `parent` returns. Handles obtained any other way, or before the
/// tracker was listening, are shown by number.
#[derive(Debug, Default)]
pub struct Tracker {
    handles: HashMap<(u64, u32), usize>,
    sessions: HashMap<usize, Session>,
    next_session: usize,
    pending: HashMap<Client, Pending>,
}

impl Tracker {
    /// Takes note of supervisor call `id` with `X0`-`X7` in `arguments` and
    /// `data` being the memory [`svc::input`] names for it. For a call that
    /// sends a request, returns the words describing the request, with
    /// what `contents` has of its buffers.
    pub fn call(
        &mut self,
        client: Client,
        id: u16,
        arguments: &[u64; 8],
        data: &[u8],
        contents: &[Contents],
    ) -> Option<String> {
        // A thread has one call in flight; anything older never returned.
        self.pending.remove(&client);
        match id {
            svc::CLOSE_HANDLE => {
                self.forget(client.process, arguments[0] as u32);
                None
            }
            svc::CONNECT_TO_NAMED_PORT => {
                let name = svc::text(data);
                self.pending.insert(client, Pending::Connect(name));
                None
            }
            _ => {
                let handle = svc::session_handle(id, arguments)?;
                let (line, sent) = self.request(client.process, handle, data, contents);
                if id != svc::SEND_ASYNC_REQUEST_WITH_USER_BUFFER {
                    self.pending.insert(client, Pending::Request(sent));
                    Some(line)
                } else {
                    Some(line + " (answer arrives through an event)")
                }
            }
        }
    }

    /// Takes note of the return of the call last passed to [`Tracker::call`]
    /// for `client`, with `X0`-`X7` in `results` and `data` being the memory
    /// [`svc::output`] names. For a call that sent a request, returns the
    /// words describing the answer, with what `contents` has of its buffers
    /// and of those the request had filled.
    pub fn returned(
        &mut self,
        client: Client,
        id: u16,
        results: &[u64; 8],
        data: &[u8],
        contents: &[Contents],
    ) -> Option<String> {
        let result = ResultCode(results[0] as u32);
        match self.pending.remove(&client)? {
            Pending::Connect(name) if id == svc::CONNECT_TO_NAMED_PORT => {
                if result.0 == 0 {
                    self.name(client.process, results[1] as u32, name);
                }
                None
            }
            Pending::Request(sent)
                if matches!(
                    id,
                    svc::SEND_SYNC_REQUEST | svc::SEND_SYNC_REQUEST_WITH_USER_BUFFER
                ) =>
            {
                Some(self.reply(client.process, sent, result, data, contents))
            }
            _ => None,
        }
    }

    /// What a request that supervisor call `id`, with `X0`-`X7` in
    /// `arguments` and `data` as for [`Tracker::call`], sends goes to, named
    /// as its trace line names it; `None` for a call that sends none. Asked
    /// before the call is passed to [`Tracker::call`].
    pub fn addressee(
        &self,
        process: u64,
        id: u16,
        arguments: &[u64; 8],
        data: &[u8],
    ) -> Option<String> {
        let handle = svc::session_handle(id, arguments)?;
        let session = self.handles.get(&(process, handle)).copied();
        let object = Message::parse(data)
            .as_ref()
            .and_then(Request::parse)
            .and_then(|request| request.object);
        Some(self.target(session, handle, object))
    }

    fn request(
        &mut self,
        process: u64,
        handle: u32,
        data: &[u8],
        contents: &[Contents],
    ) -> (String, Sent) {
        let session = self.handles.get(&(process, handle)).copied();
        let message = Message::parse(data);
        let request = message.as_ref().and_then(Request::parse);
        let object = request.as_ref().and_then(|request| request.object);
        let mut sent = Sent {
            session,
            target: self.target(session, handle, object),
            request: request
                .as_ref()
                .map(|request| (request.protocol, request.action)),
            argument: request
                .as_ref()
                .and_then(|request| word_at(request.data, 0)),
            service: None,
        };

        let mut line = format!("-> {} ", sent.target);
        let Some(message) = message else {
            line.push_str("message buffer is not readable");
            return (line, sent);
        };
        match &request {
            Some(request) => {
                let _ = write!(line, "{} {}", request.protocol, request.action);
                if let Some(object) = request.object {
                    let _ = write!(line, " object {object}");
                }
                if self.is_sm(session)
                    && let Action::Request(
                        SM_GET_SERVICE | SM_REGISTER_SERVICE | SM_UNREGISTER_SERVICE,
                    ) = request.action
                    && let Some(name) = request.data.get(..8)
                {
                    let name = svc::text(name);
                    let _ = write!(line, " name {name:?}");
                    sent.service = Some(name);
                }
                write_words(&mut line, " data", request.data);
                if !request.in_objects.is_empty() {
                    let _ = write!(line, " objects {:?}", request.in_objects);
                }
                if request.action == Action::ReleaseObject
                    && let Some(objects) = self.objects(session)
                    && let Some(object) = request.object
                {
                    objects.remove(&object);
                }
            }
            None => {
                let _ = write!(line, "message type {}", message.kind);
                write_words(&mut line, " data", message.raw);
            }
        }
        if message.sends_process_id {
            line.push_str(" pid");
        }
        let contents = shown(&sent.target, contents);
        write_envelope(&mut line, &message, true, contents, |_, _| None);
        (line, sent)
    }

    fn reply(
        &mut self,
        process: u64,
        sent: Sent,
        result: ResultCode,
        data: &[u8],
        contents: &[Contents],
    ) -> String {
        let contents = shown(&sent.target, contents);
        let mut line = format!("<- {} ", sent.target);
        match sent.request {
            Some((protocol, action)) => {
                let _ = write!(line, "{protocol} {action}");
            }
            None => line.push_str("message"),
        }
        if result.0 != 0 {
            let _ = write!(line, " not answered, the call failed with {result}");
            return line;
        }
        let Some(message) = Message::parse(data) else {
            line.push_str(" answer is not readable");
            return line;
        };
        let protocol = match sent.request {
            Some((protocol, Action::Request(_) | Action::Control(_))) => protocol,
            // The server has nothing to say to these: the buffer comes back
            // as it was sent (seen between boot and FS).
            Some((_, Action::Close | Action::ReleaseObject)) => {
                line.push_str(" done");
                return line;
            }
            None => {
                line.push_str(" answered");
                write_words(&mut line, " data", message.raw);
                write_envelope(&mut line, &message, false, contents, |_, _| None);
                return line;
            }
        };
        let Some(reply) = Reply::parse(&message, protocol) else {
            line.push_str(" answer is not in that convention");
            write_words(&mut line, " data", message.raw);
            write_envelope(&mut line, &message, false, contents, |_, _| None);
            return line;
        };

        let _ = write!(line, " result {}", reply.result);
        write_words(&mut line, " data", reply.data);
        let mut moved = vec![None; message.move_handles.len()];
        let mut objects = vec![None; reply.objects.len()];
        if reply.result.0 == 0 {
            self.learn(process, &sent, &message, &reply, &mut moved, &mut objects);
        }
        if !reply.objects.is_empty() {
            line.push_str(" objects [");
            for (index, (id, name)) in reply.objects.iter().zip(&objects).enumerate() {
                let separator = if index == 0 { "" } else { ", " };
                let _ = write!(line, "{separator}{id}");
                if let Some(name) = name {
                    let _ = write!(line, " = {name}");
                }
            }
            line.push(']');
        }
        write_envelope(&mut line, &message, false, contents, |is_move, index| {
            moved.get(index).filter(|_| is_move)?.as_deref()
        });
        let mut filled = contents
            .iter()
            .filter(|contents| {
                matches!(
                    contents.buffer.kind,
                    BufferKind::Receive | BufferKind::Exchange
                )
            })
            .peekable();
        if filled.peek().is_some() {
            line.push_str(" filled [");
            for (index, contents) in filled.enumerate() {
                let separator = if index == 0 { "" } else { ", " };
                let _ = write!(line, "{separator}{}", contents.buffer);
                write_contents(&mut line, &contents.bytes);
            }
            line.push(']');
        }
        line
    }

    /// Names what a successful answer hands the client: `moved` and `objects`
    /// receive the names given to the answer's moved handles and returned
    /// objects, in order.
    fn learn(
        &mut self,
        process: u64,
        sent: &Sent,
        message: &Message,
        reply: &Reply,
        moved: &mut [Option<String>],
        objects: &mut [Option<String>],
    ) {
        let Some((_, action)) = sent.request else {
            return;
        };
        let first_handle = message.move_handles.first().copied();
        match action {
            Action::Request(command) => {
                if command == SM_GET_SERVICE
                    && self.is_sm(sent.session)
                    && let (Some(service), Some(handle)) = (&sent.service, first_handle)
                {
                    self.name(process, handle, service.clone());
                    moved[0] = Some(service.clone());
                    return;
                }
                // What sm returns for its other commands is not a session.
                if self.is_sm(sent.session) {
                    return;
                }
                let derived = format!("{}#{command}", sent.target);
                for (index, &handle) in message.move_handles.iter().enumerate() {
                    let name = match index {
                        0 => derived.clone(),
                        _ => format!("{derived}.{index}"),
                    };
                    self.name(process, handle, name.clone());
                    moved[index] = Some(name);
                }
                if let Some(known) = self.objects(sent.session) {
                    for (&id, slot) in reply.objects.iter().zip(objects) {
                        known.insert(id, derived.clone());
                        *slot = Some(derived.clone());
                    }
                }
            }
            Action::Control(CONTROL_CONVERT_TO_DOMAIN) => {
                let session = sent.session.and_then(|index| self.sessions.get_mut(&index));
                if let (Some(session), Some(id)) = (session, word_at(reply.data, 0)) {
                    session.objects = Some(HashMap::from([(id, session.name.clone())]));
                }
            }
            Action::Control(CONTROL_COPY_FROM_DOMAIN) => {
                if let (Some(handle), Some(object)) = (first_handle, sent.argument) {
                    let known = self.objects(sent.session);
                    let name = known
                        .and_then(|known| known.get(&object).cloned())
                        .unwrap_or_else(|| format!("{}@{object}", sent.target));
                    self.name(process, handle, name.clone());
                    moved[0] = Some(name);
                }
            }
            Action::Control(CONTROL_CLONE | CONTROL_CLONE_EX) => {
                if let (Some(handle), Some(index)) = (first_handle, sent.session) {
                    self.forget(process, handle);
                    if let Some(session) = self.sessions.get_mut(&index) {
                        session.handles += 1;
                        self.handles.insert((process, handle), index);
                        moved[0] = Some(session.name.clone());
                    }
                }
            }
            Action::Control(_) | Action::Close | Action::ReleaseObject => {}
        }
    }

    fn is_sm(&self, session: Option<usize>) -> bool {
        session
            .and_then(|index| self.sessions.get(&index))
            .is_some_and(|session| session.name == SM_PORT)
    }

    fn objects(&mut self, session: Option<usize>) -> Option<&mut HashMap<u32, String>> {
        self.sessions.get_mut(&session?)?.objects.as_mut()
    }

    /// What to call the thing a request on `handle` addresses.
    fn target(&self, session: Option<usize>, handle: u32, object: Option<u32>) -> String {
        let session = session.and_then(|index| self.sessions.get(&index));
        let name = match session {
            Some(session) => session.name.clone(),
            None => format!("handle {handle:#x}"),
        };
        let Some(object) = object else {
            return name;
        };
        session
            .and_then(|session| session.objects.as_ref()?.get(&object).cloned())
            .unwrap_or_else(|| format!("{name}@{object}"))
    }

    /// Records that `handle` of `process` is a new session called `name`.
    fn name(&mut self, process: u64, handle: u32, name: String) {
        self.forget(process, handle);
        let session = Session {
            name,
            handles: 1,
            objects: None,
        };
        self.sessions.insert(self.next_session, session);
        self.handles.insert((process, handle), self.next_session);
        self.next_session += 1;
    }

    fn forget(&mut self, process: u64, handle: u32) {
        let Some(index) = self.handles.remove(&(process, handle)) else {
            return;
        };
        let session = self.sessions.get_mut(&index).expect("a handle's session");
        session.handles -= 1;
        if session.handles == 0 {
            self.sessions.remove(&index);
        }
    }
}

/// What is shown of the buffers of a request to `target`: nothing where
/// they carry keys or the random numbers keys are made of.
fn shown<'c>(target: &str, contents: &'c [Contents]) -> &'c [Contents] {
    match ["spl", "csrng"].iter().any(|name| target.starts_with(name)) {
        true => &[],
        false => contents,
    }
}

/// Appends ` = ` and the start of a buffer: as text if that is what it is,
/// as little-endian words otherwise.
fn write_contents(line: &mut String, bytes: &[u8]) {
    let bytes = &bytes[..bytes.len().min(SHOWN_BUFFER_BYTES as usize)];
    let text = bytes.split(|&byte| byte == 0).next().unwrap_or_default();
    let is_text = text.len() >= 4
        && text
            .iter()
            .all(|byte| byte.is_ascii_graphic() || *byte == b' ')
        && bytes[text.len()..].iter().all(|&byte| byte == 0);
    match is_text {
        true => {
            let _ = write!(line, " = {:?}", String::from_utf8_lossy(text));
        }
        false => write_words(line, " =", bytes),
    }
}

/// Appends `label [words]` for the little-endian words of `data`; nothing if
/// there is none.
fn write_words(line: &mut String, label: &str, data: &[u8]) {
    let words: Vec<u32> = data
        .chunks(4)
        .map(|chunk| {
            let mut word = [0; 4];
            word[..chunk.len()].copy_from_slice(chunk);
            u32::from_le_bytes(word)
        })
        .collect();
    if words.is_empty() {
        return;
    }
    let _ = write!(line, "{label} [");
    for (index, word) in words.iter().take(SHOWN_WORDS).enumerate() {
        let separator = if index == 0 { "" } else { ", " };
        let _ = write!(line, "{separator}{word:x}");
    }
    if let Some(more) = words
        .len()
        .checked_sub(SHOWN_WORDS)
        .filter(|&more| more != 0)
    {
        let _ = write!(line, ", ... {more} more");
    }
    line.push(']');
}

/// Appends the handles and buffers of `message`. `handle_name` is asked for
/// the name of each handle, by whether it is moved and its position.
///
/// The receive list is left out of an answer: what the header says there is
/// the server's own list, for the next request it receives in the buffer it
/// answers from, and the kernel does not copy the entries (every CMIF answer
/// in the boot trace names one entry at the end of the buffer, where the
/// client has zeros).
fn write_envelope<'n>(
    line: &mut String,
    message: &Message,
    is_request: bool,
    contents: &[Contents],
    handle_name: impl Fn(bool, usize) -> Option<&'n str>,
) {
    let handles = [
        (" copy", false, &message.copy_handles),
        (" move", true, &message.move_handles),
    ];
    for (label, is_move, handles) in handles {
        if handles.is_empty() {
            continue;
        }
        let _ = write!(line, "{label} [");
        for (index, handle) in handles.iter().enumerate() {
            let separator = if index == 0 { "" } else { ", " };
            let _ = write!(line, "{separator}{handle:#x}");
            if let Some(name) = handle_name(is_move, index) {
                let _ = write!(line, " = {name}");
            }
        }
        line.push(']');
    }
    let buffers: Vec<&Buffer> = message
        .buffers
        .iter()
        .filter(|buffer| is_request || buffer.kind != BufferKind::ReceiveList)
        .collect();
    if !buffers.is_empty() {
        line.push_str(" buffers [");
        for (index, buffer) in buffers.iter().take(SHOWN_BUFFERS).enumerate() {
            let separator = if index == 0 { "" } else { ", " };
            let _ = write!(line, "{separator}{buffer}");
            if let Some(contents) = contents.iter().find(|contents| contents.buffer == **buffer) {
                write_contents(line, &contents.bytes);
            }
        }
        if let Some(more) = buffers.len().checked_sub(SHOWN_BUFFERS)
            && more != 0
        {
            let _ = write!(line, ", ... {more} more");
        }
        line.push(']');
    }
    if message.truncated {
        line.push_str(" (message continues beyond what was read)");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|word| word.to_le_bytes()).collect()
    }

    const CLIENT: Client = Client {
        process: 7,
        thread: 0x1000,
    };

    /// `SendSyncRequest` on `handle`, then its successful return with `reply`
    /// in the buffer; the request and answer lines.
    fn exchange(
        tracker: &mut Tracker,
        handle: u32,
        request: &[u32],
        reply: &[u32],
    ) -> (String, String) {
        let arguments = [u64::from(handle), 0, 0, 0, 0, 0, 0, 0];
        let request = bytes(request);
        let sent = tracker.call(CLIENT, svc::SEND_SYNC_REQUEST, &arguments, &request, &[]);
        let reply = bytes(reply);
        let answered = tracker.returned(CLIENT, svc::SEND_SYNC_REQUEST, &[0; 8], &reply, &[]);
        (sent.unwrap(), answered.unwrap())
    }

    /// Connects `handle` to sm and has sm return `service_handle` for the
    /// service named by the two words of `name`.
    fn look_up(tracker: &mut Tracker, handle: u32, name: [u32; 2], service_handle: u32) {
        let port = [0, 0x2000, 0, 0, 0, 0, 0, 0];
        tracker.call(CLIENT, svc::CONNECT_TO_NAMED_PORT, &port, b"sm:\0", &[]);
        let connected = [0, u64::from(handle), 0, 0, 0, 0, 0, 0];
        tracker.returned(CLIENT, svc::CONNECT_TO_NAMED_PORT, &connected, &[], &[]);
        exchange(
            tracker,
            handle,
            &[0x11, 2, name[0], name[1]],
            &[0x11, 0x8000_0001, 1 << 5, service_handle, 0],
        );
    }

    const FSP_SRV: [u32; 2] = [0x2D70_7366, 0x0076_7273];

    #[test]
    fn domain_request_with_handles_and_buffers() {
        // Type 6; one pointer, one send and one receive buffer; a special
        // header with the process ID, one copied and one moved handle; a
        // receive list of one entry.
        let message = bytes(&[
            6 | 1 << 16 | 1 << 20 | 1 << 24,
            0x8000_0000 | 2 << 10 | 14,
            1 | 1 << 1 | 1 << 5,
            0,
            0,
            0xC0FE,
            0xBEEF,
            // Pointer #3: 0x40 bytes at 0x12_3456_7000.
            3 | 1 << 6 | 2 << 12 | 0x40 << 16,
            0x3456_7000,
            // Send: 0x1_0000_0100 bytes at 0x21_0000_2000.
            0x100,
            0x2000,
            1 | 2 << 2 | 1 << 24 | 1 << 28,
            // Receive: 0x80 bytes at 0x3000.
            0x80,
            0x3000,
            0,
            // Raw data starts at word 15: one word of alignment.
            0,
            // Domain: send, one object passed, 16 + 8 bytes, to object 5.
            1 | 1 << 8 | 0x18 << 16,
            5,
            0,
            0,
            SFCI,
            1,
            7,
            0,
            0x1111,
            0x2222,
            9,
            0,
            0,
            // Receive list: 0x500 bytes at 0x2_0000_4000.
            0x4000,
            2 | 0x500 << 16,
        ]);
        let message = Message::parse(&message).unwrap();
        assert!(!message.truncated);
        assert!(message.sends_process_id);
        assert_eq!(message.copy_handles, [0xC0FE]);
        assert_eq!(message.move_handles, [0xBEEF]);
        let buffer = |kind, address, size| Buffer {
            kind,
            address,
            size,
        };
        assert_eq!(
            message.buffers,
            [
                buffer(BufferKind::Pointer { index: 3 }, 0x12_3456_7000, 0x40),
                buffer(BufferKind::Send, 0x21_0000_2000, 0x1_0000_0100),
                buffer(BufferKind::Receive, 0x3000, 0x80),
                buffer(BufferKind::ReceiveList, 0x2_0000_4000, 0x500),
            ]
        );
        let request = Request::parse(&message).unwrap();
        assert_eq!(request.protocol, Protocol::Cmif);
        assert_eq!(request.action, Action::Request(7));
        assert_eq!(request.object, Some(5));
        assert_eq!(request.in_objects, [9]);
        assert_eq!(request.data, bytes(&[0x1111, 0x2222]));
    }

    #[test]
    fn control_request_and_object_release() {
        let control = bytes(&[5, 8, 0, 0, SFCI, 0, 3, 0, 0, 0]);
        let request = Request::parse(&Message::parse(&control).unwrap()).unwrap();
        assert_eq!(request.action, Action::Control(3));
        assert_eq!(request.object, None);

        let release = bytes(&[4, 8, 0, 0, 2, 11, 0, 0, 0, 0]);
        let request = Request::parse(&Message::parse(&release).unwrap()).unwrap();
        assert_eq!(request.action, Action::ReleaseObject);
        assert_eq!(request.object, Some(11));
    }

    #[test]
    fn reply_with_moved_handle_and_failure_code() {
        // Special header with one moved handle puts the raw data at word 4,
        // which is aligned already.
        let reply = bytes(&[
            0,
            0x8000_0000 | 10,
            1 << 5,
            0x1_8003,
            SFCO,
            1,
            0,
            0,
            0xAB,
            0,
            0,
            0,
            0,
            0,
        ]);
        let message = Message::parse(&reply).unwrap();
        assert_eq!(message.move_handles, [0x1_8003]);
        let reply = Reply::parse(&message, Protocol::Cmif).unwrap();
        assert_eq!(reply.result, ResultCode(0));
        assert_eq!(word_at(reply.data, 0), Some(0xAB));

        // Module 2, description 1001.
        let failed = bytes(&[0, 10, 0, 0, SFCO, 1, 0x7D202, 0, 0, 0, 0, 0]);
        let reply = Reply::parse(&Message::parse(&failed).unwrap(), Protocol::Cmif).unwrap();
        assert_eq!(reply.result.to_string(), "2002-1001");
    }

    #[test]
    fn domain_reply_returns_objects_after_outputs() {
        // Raw data at word 2: two words of alignment in front, two behind.
        let reply = bytes(&[0, 15, 0, 0, 2, 0, 0, 0, SFCO, 1, 0, 0, 0x55, 21, 22, 0, 0]);
        let reply = Reply::parse(&Message::parse(&reply).unwrap(), Protocol::Cmif).unwrap();
        assert_eq!(reply.objects, [21, 22]);
        assert_eq!(reply.data, bytes(&[0x55]));
    }

    #[test]
    fn message_cut_short_keeps_what_was_read() {
        let message = bytes(&[0x11, 0x8000_0004, 2 << 1, 0x8001]);
        let message = Message::parse(&message).unwrap();
        assert!(message.truncated);
        assert_eq!(message.copy_handles, [0x8001]);
        assert_eq!(Message::parse(&[0x11, 0, 0, 0, 2]), None);
    }

    #[test]
    fn handle_from_sm_is_named_after_the_service() {
        let mut tracker = Tracker::default();
        let port = [0, 0x2000, 0, 0, 0, 0, 0, 0];
        tracker.call(CLIENT, svc::CONNECT_TO_NAMED_PORT, &port, b"sm:\0junk", &[]);
        let connected = [0, 0x8001, 0, 0, 0, 0, 0, 0];
        tracker.returned(CLIENT, svc::CONNECT_TO_NAMED_PORT, &connected, &[], &[]);

        // sm command 1 with the name, answered with a moved handle.
        let (request, reply) = exchange(
            &mut tracker,
            0x8001,
            &[0x11, 2, FSP_SRV[0], FSP_SRV[1]],
            &[0x11, 0x8000_0001, 1 << 5, 0x8002, 0],
        );
        assert_eq!(
            request,
            "-> sm: tipc request 1 name \"fsp-srv\" data [2d707366, 767273]"
        );
        assert_eq!(
            reply,
            "<- sm: tipc request 1 result ok move [0x8002 = fsp-srv]"
        );

        // A failing command on the new session, then one returning a session.
        let (request, reply) = exchange(
            &mut tracker,
            0x8002,
            &[4, 9, 0, 0, SFCI, 1, 200, 0, 0x33, 0, 0],
            &[0, 6, 0, 0, SFCO, 1, 0x7D202, 0],
        );
        assert_eq!(request, "-> fsp-srv cmif request 200 data [33]");
        assert_eq!(reply, "<- fsp-srv cmif request 200 result 2002-1001");
        exchange(
            &mut tracker,
            0x8002,
            &[4, 6, 0, 0, SFCI, 1, 18, 0],
            &[0, 0x8000_0004, 1 << 5, 0x8003, SFCO, 1, 0, 0],
        );
        let (request, _) = exchange(&mut tracker, 0x8003, &[2, 0], &[0, 0]);
        assert_eq!(request, "-> fsp-srv#18 cmif close");

        // Closed handles are forgotten, and so are other processes' handles.
        tracker.call(
            CLIENT,
            svc::CLOSE_HANDLE,
            &[0x8002, 0, 0, 0, 0, 0, 0, 0],
            &[],
            &[],
        );
        let (request, _) = exchange(&mut tracker, 0x8002, &[4, 6, 0, 0, SFCI, 1, 1, 0], &[0, 0]);
        assert_eq!(request, "-> handle 0x8002 cmif request 1");
    }

    #[test]
    fn cmif_lookup_names_the_service_and_what_it_returns() {
        // libnx on hbloader asks sm in CMIF, and lm returns its logger as a
        // moved handle when the session is no domain.
        let mut tracker = Tracker::default();
        let port = [0, 0x2000, 0, 0, 0, 0, 0, 0];
        tracker.call(CLIENT, svc::CONNECT_TO_NAMED_PORT, &port, b"sm:\0", &[]);
        let connected = [0, 0x8001, 0, 0, 0, 0, 0, 0];
        tracker.returned(CLIENT, svc::CONNECT_TO_NAMED_PORT, &connected, &[], &[]);
        let (request, reply) = exchange(
            &mut tracker,
            0x8001,
            &[4, 10, 0, 0, SFCI, 1, 1, 0, 0x6D6C, 0, 0, 0],
            &[0, 0x8000_0004, 1 << 5, 0x8002, SFCO, 1, 0, 0],
        );
        assert_eq!(request, "-> sm: cmif request 1 name \"lm\" data [6d6c, 0]");
        assert_eq!(reply, "<- sm: cmif request 1 result ok move [0x8002 = lm]");
        exchange(
            &mut tracker,
            0x8002,
            &[4, 10, 0, 0, SFCI, 1, 0, 0, 0, 0, 0, 0],
            &[0, 0x8000_0004, 1 << 5, 0x8003, SFCO, 1, 0, 0],
        );
        let log = bytes(&[4, 8, 0, 0, SFCI, 1, 0, 0]);
        let arguments = [0x8003, 0, 0, 0, 0, 0, 0, 0];
        let addressee = tracker.addressee(CLIENT.process, svc::SEND_SYNC_REQUEST, &arguments, &log);
        assert_eq!(addressee.as_deref(), Some("lm#0"));
        // Not a request-sending call.
        let closing = tracker.addressee(CLIENT.process, svc::CLOSE_HANDLE, &arguments, &[]);
        assert_eq!(closing, None);
    }

    #[test]
    fn failed_lookup_and_failed_call_name_nothing() {
        let mut tracker = Tracker::default();
        look_up(&mut tracker, 0x8001, FSP_SRV, 0x8002);

        // sm answers with an error: whatever is in the handle slot is not a
        // session.
        let (_, reply) = exchange(
            &mut tracker,
            0x8001,
            &[0x11, 2, FSP_SRV[0], FSP_SRV[1]],
            &[0x11, 0x8000_0001, 1 << 5, 0x8005, 0xE02],
        );
        assert_eq!(
            reply,
            "<- sm: tipc request 1 result 2002-0007 move [0x8005]"
        );

        // The kernel refuses the call: the buffer still holds the request.
        let request = bytes(&[4, 6, 0, 0, SFCI, 1, 1, 0]);
        let arguments = [0x8002, 0, 0, 0, 0, 0, 0, 0];
        tracker.call(CLIENT, svc::SEND_SYNC_REQUEST, &arguments, &request, &[]);
        let failed = [0xE401, 0, 0, 0, 0, 0, 0, 0];
        let reply = tracker.returned(CLIENT, svc::SEND_SYNC_REQUEST, &failed, &request, &[]);
        assert_eq!(
            reply.unwrap(),
            "<- fsp-srv cmif request 1 not answered, the call failed with 2001-0114"
        );
    }

    #[test]
    fn domain_objects_are_named_after_the_command_that_returned_them() {
        let mut tracker = Tracker::default();
        look_up(&mut tracker, 0x8001, FSP_SRV, 0x8002);

        // Control 0 turns the session into a domain whose own object is 1.
        exchange(
            &mut tracker,
            0x8002,
            &[5, 6, 0, 0, SFCI, 1, 0, 0],
            &[0, 9, 0, 0, SFCO, 1, 0, 0, 1, 0, 0],
        );
        // Command 8 on object 1 returns object 2.
        let domain_request = |object, command| {
            [
                4,
                10,
                0,
                0,
                1 | 0x10 << 16,
                object,
                0,
                0,
                SFCI,
                1,
                command,
                0,
            ]
        };
        let (request, reply) = exchange(
            &mut tracker,
            0x8002,
            &domain_request(1, 8),
            &[0, 13, 0, 0, 1, 0, 0, 0, SFCO, 1, 0, 0, 2, 0, 0],
        );
        assert_eq!(request, "-> fsp-srv cmif request 8 object 1");
        assert_eq!(
            reply,
            "<- fsp-srv cmif request 8 result ok objects [2 = fsp-srv#8]"
        );
        let (request, _) = exchange(&mut tracker, 0x8002, &domain_request(2, 0), &[0, 0]);
        assert_eq!(request, "-> fsp-srv#8 cmif request 0 object 2");

        // A clone of the session shares its objects; an unknown object is
        // shown by number.
        exchange(
            &mut tracker,
            0x8002,
            &[5, 6, 0, 0, SFCI, 1, 2, 0],
            &[0, 0x8000_0004, 1 << 5, 0x8009, SFCO, 1, 0, 0],
        );
        let (request, _) = exchange(&mut tracker, 0x8009, &domain_request(2, 1), &[0, 0]);
        assert_eq!(request, "-> fsp-srv#8 cmif request 1 object 2");
        let (request, _) = exchange(&mut tracker, 0x8009, &domain_request(40, 1), &[0, 0]);
        assert_eq!(request, "-> fsp-srv@40 cmif request 1 object 40");

        // A released object's name is forgotten.
        let release = [4, 8, 0, 0, 2, 2, 0, 0, 0, 0];
        let (request, reply) = exchange(&mut tracker, 0x8002, &release, &release);
        assert_eq!(request, "-> fsp-srv#8 cmif release object 2");
        assert_eq!(reply, "<- fsp-srv#8 cmif release done");
        let (request, _) = exchange(&mut tracker, 0x8002, &domain_request(2, 0), &[0, 0]);
        assert_eq!(request, "-> fsp-srv@2 cmif request 0 object 2");
    }
}
