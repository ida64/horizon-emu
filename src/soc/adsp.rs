//! The audio DSP (ADSP) of the audio processing engine and its host
//! interface: the miscellaneous block (AMISC) with the hardware mailboxes,
//! the exception vectors the DSP boots from, and the protocol the DSP's
//! firmware speaks to `audio`.
//!
//! The DSP is a 32-bit ARM core inside the SoC. `audio` carries its
//! firmware, NVIDIA's "ADSP OS" with three applications (`gmix`,
//! `AudioRenderer`, `OpusDecoder`), as an image in its read-only data (text
//! `0x9c0`: `0x82DC8` bytes at `0xC5460`, linked at DSP address
//! `0x80500000`) plus its exception vector table (`0x40` bytes at
//! `0x148228`). The host cannot run 32-bit ARM code, so that firmware is not
//! executed. This module models what `audio` observes of it, as established
//! from both sides of the protocol: `audio`'s driver and the DSP image
//! (offsets below are DSP addresses in that image). **No sound is produced
//! and nothing is captured**: render commands are acknowledged without
//! being executed, and the mixer's streams carry nothing.
//!
//! # Boot (`audio` text `0x17ab0`, `0x17e20`)
//!
//! `audio` holds the DSP in reset, copies the image into a 4 MiB buffer it
//! maps for the engine at device address `0x80500000`, clears a `0x7000`
//! byte buffer shared with the DSP at `0x80300000`, writes the vectors to
//! [`EVP_BASE`], releases the reset (pcv: `RST_DEV_Y` bit 7, through
//! `RST_DEV_Y_CLR`) and waits at most five seconds for message 0 on mailbox
//! channel 2; without it the driver fails and `audio` aborts with
//! 2153-1542 (text `0x6abc8`). The firmware, once let go
//! (`bootstrap2`, `0x805243B8`), initializes two message queues in the
//! shared buffer (`0x80524A00`), queues one descriptor per application it
//! carries (`0x80524984`), and sends the boot-complete message (`0x80520D2C`).
//!
//! # Mailboxes (`audio` text `0x16540`, `0x168d0`, `0x16a00`)
//!
//! AMISC has four hardware mailbox registers (`0x58` + 4n). A message is
//! `1 << 31` (full) | channel << 20 | 20-bit payload. The host sends in
//! mailbox 1 (`0x5C`) and receives in mailbox 0 (`0x58`). The receiver
//! takes a message by writing the register back with the full bit clear and
//! a 20-bit status (0, or a negative error) in the low bits. A message
//! raises the mailbox's "full" interrupt (AGIC line 32 + n), a taken one
//! its "empty" interrupt (36 + n); `audio` handles line 32 (receive) and 37
//! (its message was taken), configured edge-triggered, and routes 33 and 36
//! to the DSP. Channels on the DSP side: 1 application service, 2
//! host communication, 3 clock scaling, `gmix`'s (allocated: the first free
//! dynamic one, 32), 50 and 51 the renderer's, 70 the Opus decoder's.
//!
//! # Message queues (`audio` text `0x19e40`, `0x19f30`)
//!
//! A queue is `{capacity, write index, read index, words[capacity]}`; a
//! record is a word count followed by that many words, and wraps around.
//! The shared buffer holds the host-to-DSP queue at `+0` and the DSP-to-host
//! queue at `+0x1000`, each of 1021 words (`0x80524A00`).
//!
//! # Applications (`audio` text `0x19430`, `0x19660`, `0x19750`)
//!
//! The DSP describes each application it carries in a 48-word record:
//! word 0 zero, words 1..33 its name, words 33..43 five 64-bit memory sizes
//! (DRAM, shared DRAM, write-combined shared DRAM, two ARAM sizes), then its
//! descriptor's address (the handle the host launches it by), its flags (bit
//! 0: started by the OS itself), and three more descriptor words; an
//! all-zero record ends the list (`0x805245C0`). To launch one, the host
//! allocates its memory in a 1 MiB area at `0x80800000`, queues a 152-word
//! request (handle, the host's instance pointer, memory addresses and
//! sizes, command 1 in the last word) and sends any message on channel 1;
//! the DSP dequeues one request per message, starts the application and
//! answers with a 5-word status record (4 = started, the host's pointer,
//! its own instance handle, result) and message 4 on channel 1
//! (`0x80524874`).
//!
//! * `gmix`, the mixer between the engine's audio hub and everyone else,
//!   initializes its shared memory (`0x80500DB4`): it clears it, sets up a
//!   command queue at `+0` and a notification queue at `+0x800` (509 words
//!   each), records three buffer addresses `audio` reads back, and queues a
//!   record holding the mailbox channel it opened. `audio` polls for that
//!   record (text `0x12580`) and then sends commands: a record in the
//!   command queue and its opcode as message (text `0x124f0`). No command
//!   needs an answer; what the commands configure (routes, stream buffers)
//!   carries no sound here, so they are taken without effect.
//! * `AudioRenderer` executes the command lists `audio`'s renderers build.
//!   On channel 50 it answers 22 with 22 and then 32, each 42 (render) with
//!   32 once the frame is done, and 52 (stop) with 52; on channel 51 the
//!   memory requests 1/3/5/7 with 2/4/6/8 (`0x80504140`, `0x80504430`).
//!   Its shared memory holds one 40-byte descriptor per renderer session:
//!   command list address and size at words 0 and 1, the DSP's count of
//!   commands left at word 3 and its render time in microseconds at word 8.
//!   Here a frame takes 5 ms (240 samples at 48 kHz, the frame the DSP
//!   renders at `0x80504140`) and leaves no command undone in no time; the
//!   commands themselves are not executed, so voices play nothing and
//!   their state is not updated.
//! * `OpusDecoder` (started by the OS, channel 70) decodes for `hwopus`.
//!   `audio` starts it at start-up (text `0x95b60`): message 1 (start),
//!   then the DSP address of a 4096-byte buffer it shares, high and low 16
//!   bits; the decoder clears the buffer's word at `0xFC0` (its dispatcher
//!   started) and answers 11 (`0x8051FB00`). `audio` then asks twice for
//!   the work buffer size of a decoder (message 21 and a channel count, 1
//!   or 2): the decoder leaves it at `0xFC0` and answers 41
//!   (`0x8051FD08`). The size is libopus's `opus_decoder_get_size` as built
//!   into the image (`0x8052A32C`: the SILK decoder's `0x2168` bytes, the
//!   CELT decoder's `(0x2030 + 4 * 120) * channels + 16 * 21 + 0x58` for
//!   its 48 kHz mode, `0x50` for the Opus state) plus the dispatcher's
//!   `0x14` (`0x80520518`). Decoding itself (messages 22 onwards) is not
//!   modeled and refuses.
use std::{
    collections::VecDeque,
    sync::{Arc, Condvar, Mutex, MutexGuard, Weak},
    thread,
    time::{Duration, Instant},
};

use super::{
    Notes,
    agic::Agic,
    mc::DeviceMemory,
    regs::{RegisterFile, word},
};
use crate::bus::MmioDevice;

/// AMISC: `audio` maps `0x2000` bytes here for the mailboxes (text
/// `0x16b80`) and the activity monitor (text `0x17940`).
pub const AMISC_BASE: u64 = 0x702E_C000;
pub const AMISC_SIZE: u64 = 0x2000;
/// The DSP's exception vectors: `audio` maps 64 bytes here and copies the
/// image's vector table in (text `0x17940`, `0x17ab0`).
pub const EVP_BASE: u64 = 0x702E_F700;
pub const EVP_SIZE: u64 = 0x40;
/// The engine's SMMU client (`ape`, from the public Linux `tegra210.c`
/// swgroup table): the DSP reaches memory at the device addresses `audio`
/// maps for it.
pub const SMMU_ASID: u64 = 0xAB8;
/// `RST_DEVICES_Y` bit 7 holds the DSP core in reset: the last bit `pcv`
/// releases (through `RST_DEV_Y_CLR`) when `audio` asks it to deassert the
/// reset of module `0x40000037`, after the DSP's other reset bits (21-26).
pub const RESET_REGISTER: u64 = 0x2A4;
pub const RESET_BIT: u32 = 7;

/// A mailbox register's full bit.
const FULL: u32 = 1 << 31;
/// Mailbox registers as the host uses them.
const TO_HOST: u64 = 0x58;
const TO_DSP: u64 = 0x5C;
/// AGIC lines: mailbox 0 full (a message for the host), mailbox 1 empty
/// (the host's message was taken).
const TO_HOST_FULL: u32 = 32;
const TO_DSP_EMPTY: u32 = 37;

/// The buffer shared with the OS and its two queues.
const SHARED: u32 = 0x8030_0000;
const HOST_QUEUE: u32 = SHARED;
const DSP_QUEUE: u32 = SHARED + 0x1000;
const OS_QUEUE_WORDS: u32 = 1021;

const APP_SERVICE: u32 = 1;
const HOST_COM: u32 = 2;
/// `gmix` opens its channel without a number and gets the first free one
/// above the 32 the OS keeps for fixed channels (`0x805225DC`); the
/// renderer's (50, 51) and the Opus decoder's (70) are fixed and opened
/// later, if at all, so that is 32.
const GMIX: u32 = 32;
const RENDERER: u32 = 50;
const RENDERER_MEMORY: u32 = 51;
const OPUS: u32 = 70;

/// The applications the image carries: its table at `0x80552C28`, 26 words
/// per application. Only the words the descriptor records carry are kept.
struct App {
    name: &'static str,
    /// The table entry's DSP address.
    handle: u32,
    /// DRAM, shared DRAM, write-combined shared DRAM, ARAM, ARAM-X.
    sizes: [u64; 5],
    flags: u32,
    /// Entry words 3 to 5.
    extra: [u32; 3],
}

const GMIX_APP: u32 = 0x8055_2C28;
const RENDERER_APP: u32 = 0x8055_2C90;
const APPS: [App; 3] = [
    App {
        name: "gmix",
        handle: GMIX_APP,
        sizes: [0, 0x2_EDC8, 0, 0, 0],
        flags: 0,
        extra: [0, 0x8055_4000, 0],
    },
    App {
        name: "AudioRenderer",
        handle: RENDERER_APP,
        sizes: [0, 0x70, 0, 0, 0],
        flags: 0,
        extra: [0, 0x807A_DBB0, 0],
    },
    App {
        name: "OpusDecoder",
        handle: 0x8055_2CF8,
        sizes: [0; 5],
        flags: 1,
        extra: [0; 3],
    },
];

/// A launch request: words of its record (after the count).
const LAUNCH_WORDS: usize = 152;
const LAUNCH_COMMAND: usize = 151;
const LAUNCH_RUN: u32 = 1;
/// Word of the launch request with the shared DRAM's DSP address, the
/// only memory `gmix` and the renderer have.
const LAUNCH_SHARED_DRAM: usize = 5;
const STATUS_STARTED: u32 = 4;

/// `gmix`'s shared memory (`0x80500DB4`).
const GMIX_SIZE: usize = 0x2_EDC8;
const GMIX_QUEUE_WORDS: u32 = 509;
const GMIX_NOTIFICATIONS: u32 = 0x800;
/// Words where `gmix` leaves three buffer addresses `audio` reads back
/// (text `0x12580`): `(base + offset) & !(alignment - 1)` rounded up, as
/// `0x80500DB4` computes them.
const GMIX_BUFFERS: [(u32, u32, u32); 3] = [
    (0x2_BBB8, 0x2_B2B7, 0x100),
    (0x2_D3D0, 0x2_BBCF, 0x1000),
    (0x2_ECE0, 0x2_D3DF, 0x100),
];

/// Opus decoder messages on channel 70 (`0x8051FB00`, `0x8051FD08`).
const OPUS_START: u32 = 1;
const OPUS_STARTED: u32 = 11;
const OPUS_WORK_SIZE: u32 = 21;
const OPUS_WORK_SIZE_DONE: u32 = 41;
/// Where in the shared buffer the decoder leaves its results.
const OPUS_RESULT: u32 = 0xFC0;

/// Work buffer bytes of an Opus decoder of `channels` channels, as the
/// image computes it (see the module documentation).
fn opus_work_size(channels: u32) -> u32 {
    let silk = 0x2168;
    let celt = (0x2030 + 4 * 120) * channels + 16 * 21 + 0x58;
    silk + celt + 0x50 + 0x14
}

/// Where the Opus decoder's conversation with `audio` stands.
#[derive(Clone, Copy)]
enum Opus {
    /// Waiting for the start message.
    Idle,
    /// Started; waiting for the shared buffer's address, high half first.
    Address(Option<u32>),
    /// Serving requests in the shared buffer at this DSP address.
    Ready(u32),
    /// Got a work buffer size request; waiting for the channel count.
    WorkSize(u32),
}

/// Renderer messages on channel 50 (`0x80504140`).
const RENDER_INITIALIZE: u32 = 22;
const RENDER_DONE: u32 = 32;
const RENDER: u32 = 42;
const RENDER_STOP: u32 = 52;
/// One renderer frame: 240 samples at 48 kHz.
const FRAME: Duration = Duration::from_millis(5);
/// Renderer session descriptors in the renderer's shared memory: 40 bytes
/// each; command list address at word 0, commands left at 3, render time
/// at 8.
const SESSIONS: u32 = 2;
const SESSION_BYTES: u32 = 40;

struct State {
    /// Out of reset.
    running: bool,
    mailboxes: [u32; 4],
    /// Messages for the host waiting for mailbox 0 to be free.
    outbox: VecDeque<u32>,
    /// AMISC `0x54`: `audio` sets bit 31 before it starts the activity
    /// monitor (text `0x19d20`).
    control: u32,
    /// `gmix`'s shared memory, once launched.
    gmix: Option<u32>,
    /// The renderer's shared memory, once launched.
    renderer: Option<u32>,
    opus: Opus,
    /// When the renderer's frame in progress is done.
    frame_due: Option<Instant>,
    last_frame: Option<Instant>,
}

struct Inner {
    state: Mutex<State>,
    wake: Condvar,
    memory: DeviceMemory,
    agic: Arc<Agic>,
    /// The activity monitor (AMISC `0x800`...).
    monitor: RegisterFile,
    notes: Arc<Notes>,
}

/// The DSP and its AMISC block, as mapped at [`AMISC_BASE`].
pub struct Adsp(Arc<Inner>);

impl Adsp {
    /// The activity monitor's registers as `audio` programs them (text
    /// `0x19d20`): control at `0x800`, an interrupt status at `0x820` it
    /// clears by writing ones, two settings at `0x824` and `0x828`, and the
    /// busy count it reads when the monitor interrupts (`0x818`, text
    /// `0x19be0`). The DSP's firmware does not run here, so the monitor
    /// counts no busy cycles: the count and the status stay zero and the
    /// monitor never interrupts. The control and settings start at zero,
    /// which is an assumption; `audio` sets the bits it needs.
    pub fn new(memory: DeviceMemory, agic: Arc<Agic>, notes: Arc<Notes>) -> Self {
        let inner = Arc::new(Inner {
            state: Mutex::new(State {
                running: false,
                mailboxes: [0; 4],
                outbox: VecDeque::new(),
                control: 0,
                gmix: None,
                renderer: None,
                opus: Opus::Idle,
                frame_due: None,
                last_frame: None,
            }),
            wake: Condvar::new(),
            memory,
            agic,
            monitor: RegisterFile::strict([
                (0x800, 0),
                (0x818, 0),
                (0x820, 0),
                (0x824, 0),
                (0x828, 0),
            ]),
            notes,
        });
        let weak = Arc::downgrade(&inner);
        thread::Builder::new()
            .name("ADSP".into())
            .spawn(move || keep_time(weak))
            .expect("cannot start the ADSP's thread");
        Self(inner)
    }

    /// The DSP's reset was released: its OS boots.
    pub fn release(&self) {
        let mut state = self.0.state.lock().unwrap();
        if let Err(what) = self.0.boot(&mut state) {
            self.0.notes.add(format!("ADSP: cannot boot: {what}"));
        }
    }

    /// The DSP's reset was asserted: it stops and forgets its applications.
    pub fn reset(&self) {
        let mut state = self.0.state.lock().unwrap();
        state.running = false;
        state.outbox.clear();
        state.gmix = None;
        state.renderer = None;
        state.opus = Opus::Idle;
        state.frame_due = None;
        state.last_frame = None;
    }
}

/// Completes renderer frames when they are due.
fn keep_time(inner: Weak<Inner>) {
    while let Some(inner) = inner.upgrade() {
        let mut state = inner.state.lock().unwrap();
        let now = Instant::now();
        if let Some(due) = state.frame_due.filter(|&due| due <= now) {
            state.frame_due = None;
            state.last_frame = Some(due);
            if let Err(what) = inner.finish_frame(&mut state) {
                inner.notes.add(format!("ADSP: renderer: {what}"));
            }
        }
        let wait = state
            .frame_due
            .map_or(Duration::from_millis(100), |due| {
                due.saturating_duration_since(now)
            })
            .min(Duration::from_millis(100));
        drop(inner.wake.wait_timeout(state, wait).unwrap());
    }
}

type Result<T, E = String> = std::result::Result<T, E>;

impl Inner {
    fn read_word(&self, address: u32) -> Result<u32> {
        let mut bytes = [0; 4];
        self.memory
            .read(u64::from(address), &mut bytes)
            .ok_or_else(|| format!("DSP address {address:#x} is not mapped"))?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn write_word(&self, address: u32, value: u32) -> Result<()> {
        self.memory
            .write(u64::from(address), &value.to_le_bytes())
            .ok_or_else(|| format!("DSP address {address:#x} is not mapped"))
    }

    fn write_words(&self, address: u32, words: &[u32]) -> Result<()> {
        let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        self.memory
            .write(u64::from(address), &bytes)
            .ok_or_else(|| format!("DSP addresses from {address:#x} are not mapped"))
    }

    /// `msgq_init`: an empty queue of `capacity` words.
    fn queue_init(&self, queue: u32, capacity: u32) -> Result<()> {
        self.write_words(queue, &[capacity, 0, 0])
    }

    /// Queues `record` (its word count first), wrapping around the end.
    fn enqueue(&self, queue: u32, record: &[u32]) -> Result<()> {
        let [capacity, write, read] = [0, 4, 8].map(|at| self.read_word(queue + at));
        let (capacity, write, read) = (capacity?, write?, read?);
        let words = record.len() as u32;
        let (to_end, free) = if read <= write {
            (capacity - write, read + capacity - write)
        } else {
            (read - write, read - write)
        };
        if write >= capacity || words >= free {
            return Err(format!("queue at {queue:#x} is full"));
        }
        let data = queue + 12;
        if words >= to_end {
            let (head, tail) = record.split_at(to_end as usize);
            self.write_words(data + write * 4, head)?;
            self.write_words(data, tail)?;
            self.write_word(queue + 4, write + words - capacity)
        } else {
            self.write_words(data + write * 4, record)?;
            self.write_word(queue + 4, write + words)
        }
    }

    /// Takes the next record's words (without the count), if any.
    fn dequeue(&self, queue: u32) -> Result<Option<Vec<u32>>> {
        let [capacity, write, read] = [0, 4, 8].map(|at| self.read_word(queue + at));
        let (capacity, write, read) = (capacity?, write?, read?);
        if read == write {
            return Ok(None);
        }
        if read >= capacity {
            return Err(format!("queue at {queue:#x} is corrupt"));
        }
        let data = queue + 12;
        let count = self.read_word(data + read * 4)?;
        if count >= capacity {
            return Err(format!(
                "queue at {queue:#x} holds a record of {count} words"
            ));
        }
        let mut record = Vec::with_capacity(count as usize);
        let mut at = read;
        for _ in 0..count {
            at = (at + 1) % capacity;
            record.push(self.read_word(data + at * 4)?);
        }
        self.write_word(queue + 8, (at + 1) % capacity)?;
        Ok(Some(record))
    }

    fn boot(&self, state: &mut State) -> Result<()> {
        state.running = true;
        self.queue_init(HOST_QUEUE, OS_QUEUE_WORDS)?;
        self.queue_init(DSP_QUEUE, OS_QUEUE_WORDS)?;
        for app in &APPS {
            self.enqueue(DSP_QUEUE, &descriptor(Some(app)))?;
        }
        self.enqueue(DSP_QUEUE, &descriptor(None))?;
        self.send(state, HOST_COM, 0);
        Ok(())
    }

    /// Sends `payload` to the host on `channel`, now or once mailbox 0 is
    /// free.
    fn send(&self, state: &mut State, channel: u32, payload: u32) {
        let message = FULL | channel << 20 | payload;
        if state.mailboxes[0] & FULL == 0 {
            state.mailboxes[0] = message;
            self.agic.pulse(TO_HOST_FULL);
        } else {
            state.outbox.push_back(message);
        }
    }

    /// The host took (or, at start-up, cleared) mailbox 0.
    fn host_took(&self, state: &mut State, value: u32) -> Result<()> {
        if value & FULL != 0 {
            return Err(format!("the host put {value:#x} in the DSP's mailbox"));
        }
        state.mailboxes[0] = value;
        if value & 0xF_FFFF != 0 {
            return Err(format!("the host refused a message with status {value:#x}"));
        }
        if let Some(next) = state.outbox.pop_front() {
            state.mailboxes[0] = next;
            self.agic.pulse(TO_HOST_FULL);
        }
        Ok(())
    }

    /// A message from the host in mailbox 1. The DSP takes it with status 0
    /// once it has handled it.
    fn host_sent(&self, state: &mut State, message: u32) -> Result<()> {
        state.mailboxes[1] = message;
        if message & FULL == 0 {
            return Err(format!("the host wrote {message:#x} without the full bit"));
        }
        if !state.running {
            // Held in reset: nobody takes it.
            return Ok(());
        }
        let (channel, payload) = (message >> 20 & 0x3FF, message & 0xF_FFFF);
        match channel {
            APP_SERVICE => self.app_service(state)?,
            RENDERER if state.renderer.is_some() => self.renderer(state, payload)?,
            RENDERER_MEMORY if state.renderer.is_some() && matches!(payload, 1 | 3 | 5 | 7) => {
                self.send(state, RENDERER_MEMORY, payload + 1)
            }
            GMIX if state.gmix.is_some() => self.gmix(state, payload)?,
            OPUS => self.opus(state, payload)?,
            _ => {
                return Err(format!(
                    "message {payload:#x} on channel {channel} is not modeled"
                ));
            }
        }
        state.mailboxes[1] = 0;
        self.agic.pulse(TO_DSP_EMPTY);
        Ok(())
    }

    /// One launch request (`0x80524874`).
    fn app_service(&self, state: &mut State) -> Result<()> {
        let request = self
            .dequeue(HOST_QUEUE)?
            .ok_or("an application service message without a request")?;
        if request.len() != LAUNCH_WORDS || request[LAUNCH_COMMAND] != LAUNCH_RUN {
            return Err(format!(
                "application request of {} words, command {:?}, is not modeled",
                request.len(),
                request.get(LAUNCH_COMMAND)
            ));
        }
        let (handle, shared) = (request[0], request[LAUNCH_SHARED_DRAM]);
        match handle {
            GMIX_APP => {
                self.start_gmix(shared)?;
                state.gmix = Some(shared);
            }
            RENDERER_APP => state.renderer = Some(shared),
            _ => {
                return Err(format!(
                    "launching the application at {handle:#x} is not modeled"
                ));
            }
        }
        // The OS's instance handle is the address of its record of the
        // running application, which the host only hands back; the
        // application's descriptor address stands in for it.
        let status = [5, STATUS_STARTED, request[1], request[2], handle, 0];
        self.enqueue(DSP_QUEUE, &status)?;
        self.send(state, APP_SERVICE, STATUS_STARTED);
        Ok(())
    }

    /// `gmix`'s start-up (`0x80500DB4`).
    fn start_gmix(&self, base: u32) -> Result<()> {
        self.memory
            .write(u64::from(base), &vec![0; GMIX_SIZE])
            .ok_or_else(|| format!("gmix's memory at {base:#x} is not mapped"))?;
        self.queue_init(base, GMIX_QUEUE_WORDS)?;
        self.queue_init(base + GMIX_NOTIFICATIONS, GMIX_QUEUE_WORDS)?;
        for (at, from, alignment) in GMIX_BUFFERS {
            self.write_word(base + at, ((base + from) & !(alignment - 1)) + alignment)?;
        }
        self.enqueue(base + GMIX_NOTIFICATIONS, &[2, 0, GMIX])
    }

    /// A `gmix` command: the opcode as message, its record in the command
    /// queue. Taken without effect (see the module documentation).
    fn gmix(&self, state: &mut State, opcode: u32) -> Result<()> {
        let base = state.gmix.ok_or("gmix is not running")?;
        let command = self
            .dequeue(base)?
            .ok_or_else(|| format!("gmix command {opcode} without a record"))?;
        if command.first() != Some(&opcode) {
            return Err(format!("gmix command {opcode} with record {command:x?}"));
        }
        Ok(())
    }

    /// One message to the Opus decoder (see the module documentation).
    fn opus(&self, state: &mut State, message: u32) -> Result<()> {
        state.opus = match (state.opus, message) {
            (Opus::Idle, OPUS_START) => Opus::Address(None),
            (Opus::Address(None), high) => Opus::Address(Some(high << 16)),
            (Opus::Address(Some(high)), low) => {
                let buffer = high | low;
                self.write_word(buffer + OPUS_RESULT, 0)?;
                self.send(state, OPUS, OPUS_STARTED);
                Opus::Ready(buffer)
            }
            (Opus::Ready(buffer), OPUS_WORK_SIZE) => Opus::WorkSize(buffer),
            (Opus::WorkSize(buffer), channels @ (1 | 2)) => {
                self.write_word(buffer + OPUS_RESULT, opus_work_size(channels))?;
                self.send(state, OPUS, OPUS_WORK_SIZE_DONE);
                Opus::Ready(buffer)
            }
            _ => return Err(format!("Opus decoder message {message} is not modeled")),
        };
        Ok(())
    }

    fn renderer(&self, state: &mut State, message: u32) -> Result<()> {
        match message {
            RENDER_INITIALIZE => {
                self.send(state, RENDERER, RENDER_INITIALIZE);
                self.send(state, RENDERER, RENDER_DONE);
            }
            RENDER if state.frame_due.is_none() => {
                let now = Instant::now();
                let due = state.last_frame.map_or(now, |last| (last + FRAME).max(now));
                state.frame_due = Some(due);
                self.wake.notify_all();
            }
            RENDER_STOP => {
                state.renderer = None;
                self.send(state, RENDERER, RENDER_STOP);
            }
            _ => return Err(format!("renderer message {message} is not modeled")),
        }
        Ok(())
    }

    /// A frame is done: every session's command list is finished, in no
    /// time (words 3 and 8 of its descriptor), and the host is told.
    fn finish_frame(&self, state: &mut State) -> Result<()> {
        let Some(base) = state.renderer else {
            return Ok(());
        };
        for session in 0..SESSIONS {
            let descriptor = base + session * SESSION_BYTES;
            if self.read_word(descriptor)? != 0 {
                self.write_word(descriptor + 3 * 4, 0)?;
                self.write_word(descriptor + 8 * 4, 0)?;
            }
        }
        self.send(state, RENDERER, RENDER_DONE);
        Ok(())
    }
}

/// An application's descriptor record, or the empty one that ends the list
/// (`0x805245C0`).
fn descriptor(app: Option<&App>) -> [u32; 49] {
    let mut record = [0; 49];
    record[0] = 48;
    if let Some(app) = app {
        let words = &mut record[1..];
        for (index, chunk) in app.name.as_bytes().chunks(4).enumerate() {
            let mut bytes = [0; 4];
            bytes[..chunk.len()].copy_from_slice(chunk);
            words[1 + index] = u32::from_le_bytes(bytes);
        }
        for (index, size) in app.sizes.iter().enumerate() {
            words[33 + 2 * index] = *size as u32;
            words[34 + 2 * index] = (*size >> 32) as u32;
        }
        words[43] = app.handle;
        words[44] = app.flags;
        words[45..48].copy_from_slice(&app.extra);
    }
    record
}

impl Inner {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap()
    }

    fn refuse(&self, what: String) -> Option<()> {
        self.notes.add(format!("ADSP: {what}"));
        None
    }
}

impl MmioDevice for Adsp {
    fn read(&self, core: usize, offset: u64, size: u8) -> Option<u64> {
        let offset = word(offset, size)?;
        let state = self.0.lock();
        let value = match offset {
            0x54 => state.control,
            0x58..=0x64 => state.mailboxes[((offset - 0x58) / 4) as usize],
            0x800.. => return self.0.monitor.read(core, offset, size),
            _ => return None,
        };
        Some(u64::from(value))
    }

    fn write(&self, core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        let offset = word(offset, size)?;
        let value = value as u32;
        let mut state = self.0.lock();
        let done = match offset {
            0x54 => {
                state.control = value;
                Ok(())
            }
            TO_HOST => self.0.host_took(&mut state, value),
            TO_DSP => self.0.host_sent(&mut state, value),
            // The status bits clear when written with ones; none is ever set.
            0x820 => Ok(()),
            0x800 | 0x824 | 0x828 => {
                return self.0.monitor.write(core, offset, size, u64::from(value));
            }
            _ => return None,
        };
        match done {
            Ok(()) => Some(()),
            Err(what) => self.0.refuse(what),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{boot::DramSize, memory::GuestRam, soc::mc::Mc};
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn descriptor_records_match_the_image_table() {
        let record = descriptor(Some(&APPS[1]));
        assert_eq!(record[0], 48);
        assert_eq!(
            &record[2..6],
            &[0x6964_7541, 0x6E65_526F, 0x6572_6564, 0x72]
        );
        assert_eq!(record[1 + 35], 0x70);
        assert_eq!(record[1 + 43], 0x8055_2C90);
        assert_eq!(record[1 + 46], 0x807A_DBB0);
        assert_eq!(descriptor(None)[1..], [0; 48]);
    }

    /// A DSP whose memory is DRAM seen one to one: the test's SMMU maps the
    /// engine's addresses `0x8000_0000..` onto DRAM at the same address.
    fn dsp() -> (Adsp, Arc<GuestRam>, Arc<Agic>, Arc<AtomicBool>) {
        let ram = Arc::new(GuestRam::new(0x8000_0000, 0x200_0000).unwrap());
        let mc = Arc::new(Mc::new(DramSize::Gb4));
        // Address space 1: a directory at 0x81000000 of 4 MiB sections.
        for index in 0x200..0x208u32 {
            let entry = 0xC000_0000 | index << 10;
            ram.write(0x8100_0000 + u64::from(index) * 4, &entry.to_le_bytes())
                .unwrap();
        }
        for (register, value) in [(0x1C, 1), (0x20, 0xC008_1000), (SMMU_ASID, 0x8000_0001)] {
            mc.write(0, register, 4, value).unwrap();
        }
        let line = Arc::new(AtomicBool::new(false));
        let seen = Arc::clone(&line);
        let agic = Arc::new(Agic::new(Box::new(move |high| {
            seen.store(high, Ordering::Relaxed)
        })));
        let (gicd, gicc) = (agic.distributor(), agic.cpu_interface());
        for (offset, value) in [
            (0x104, 0xFFFF_FFFF),
            (0x820, 0x0101_0101),
            (0x824, 0x0101_0101),
        ] {
            gicd.write(0, offset, 4, value).unwrap();
        }
        gicd.write(0, 0x80, 4, 0).unwrap();
        gicd.write(0, 0, 4, 3).unwrap();
        gicc.write(0, 4, 4, 0xFF).unwrap();
        gicc.write(0, 0, 4, 0x1F).unwrap();
        let memory = DeviceMemory::new(&mc, &ram, SMMU_ASID);
        let notes = Arc::new(Notes::default());
        (Adsp::new(memory, Arc::clone(&agic), notes), ram, agic, line)
    }

    fn word_at(ram: &GuestRam, address: u32) -> u32 {
        let mut bytes = [0; 4];
        ram.read(u64::from(address), &mut bytes).unwrap();
        u32::from_le_bytes(bytes)
    }

    /// The host side of a queue, as `audio` reads it (text `0x19f30`).
    fn host_dequeue(ram: &GuestRam, queue: u32) -> Option<Vec<u32>> {
        let (capacity, write, read) = (
            word_at(ram, queue),
            word_at(ram, queue + 4),
            word_at(ram, queue + 8),
        );
        if read == write {
            return None;
        }
        let at = |index: u32| word_at(ram, queue + 12 + (index % capacity) * 4);
        let count = at(read);
        let record = (1..=count).map(|index| at(read + index)).collect();
        ram.write(
            u64::from(queue + 8),
            &((read + count + 1) % capacity).to_le_bytes(),
        )
        .unwrap();
        Some(record)
    }

    /// Takes the DSP's message as `audio`'s interrupt handler does.
    fn receive(adsp: &Adsp, agic: &Arc<Agic>) -> u32 {
        let gicc = agic.cpu_interface();
        assert_eq!(gicc.read(0, 0x0C, 4), Some(u64::from(TO_HOST_FULL)));
        let message = adsp.read(0, TO_HOST, 4).unwrap() as u32;
        adsp.write(0, TO_HOST, 4, 0).unwrap();
        gicc.write(0, 0x10, 4, u64::from(TO_HOST_FULL)).unwrap();
        message
    }

    /// Sends a message as `audio` does and checks that the DSP took it.
    fn send(adsp: &Adsp, agic: &Arc<Agic>, channel: u32, payload: u32) {
        let gicd = agic.distributor();
        adsp.write(0, TO_DSP, 4, u64::from(FULL | channel << 20 | payload))
            .unwrap();
        assert_eq!(adsp.read(0, TO_DSP, 4), Some(0), "the DSP took the message");
        let bit = 1 << (TO_DSP_EMPTY - 32);
        assert_ne!(gicd.read(0, 0x204, 4).unwrap() & bit, 0, "and said so");
        gicd.write(0, 0x284, 4, bit).unwrap();
    }

    #[test]
    fn boot_handshake_and_application_launch() {
        let (adsp, ram, agic, line) = dsp();
        adsp.write(0, TO_HOST, 4, 0).unwrap();
        adsp.release();
        assert!(line.load(Ordering::Relaxed));
        assert_eq!(receive(&adsp, &agic), FULL | HOST_COM << 20);
        let names: Vec<u32> = std::iter::from_fn(|| host_dequeue(&ram, DSP_QUEUE))
            .map(|record| record[43])
            .collect();
        assert_eq!(names, [0x8055_2C28, 0x8055_2C90, 0x8055_2CF8, 0]);
        // Launch gmix with its shared memory at 0x80800000.
        let mut request = [0u32; 1 + LAUNCH_WORDS];
        request[0] = LAUNCH_WORDS as u32;
        request[1] = GMIX_APP;
        request[2] = 0x1234_5678;
        request[1 + LAUNCH_SHARED_DRAM] = 0x8080_0000;
        request[1 + LAUNCH_COMMAND] = LAUNCH_RUN;
        adsp.0.enqueue(HOST_QUEUE, &request).unwrap();
        send(&adsp, &agic, APP_SERVICE, 0);
        assert_eq!(
            receive(&adsp, &agic),
            FULL | APP_SERVICE << 20 | STATUS_STARTED
        );
        assert_eq!(
            host_dequeue(&ram, DSP_QUEUE),
            Some(vec![STATUS_STARTED, 0x1234_5678, 0, GMIX_APP, 0])
        );
        assert_eq!(
            host_dequeue(&ram, 0x8080_0000 + GMIX_NOTIFICATIONS),
            Some(vec![0, GMIX])
        );
        assert_eq!(word_at(&ram, 0x8080_0000 + 0x2_BBB8), 0x8082_B300);
        // A gmix command is taken.
        adsp.0.enqueue(0x8080_0000, &[2, 1, 7]).unwrap();
        send(&adsp, &agic, GMIX, 1);
        // Unknown channels refuse.
        assert_eq!(
            adsp.write(0, TO_DSP, 4, u64::from(FULL | 3 << 20 | 1)),
            None
        );
    }

    #[test]
    fn opus_start_and_work_sizes() {
        let (adsp, ram, agic, _) = dsp();
        adsp.release();
        receive(&adsp, &agic);
        let buffer = 0x8091_0000;
        ram.write(u64::from(buffer + OPUS_RESULT), &[0xFF; 4])
            .unwrap();
        for word in [OPUS_START, buffer >> 16, buffer & 0xFFFF] {
            send(&adsp, &agic, OPUS, word);
        }
        assert_eq!(receive(&adsp, &agic), FULL | OPUS << 20 | OPUS_STARTED);
        assert_eq!(word_at(&ram, buffer + OPUS_RESULT), 0);
        for (channels, size) in [(1, 0x4584), (2, 0x6794)] {
            send(&adsp, &agic, OPUS, OPUS_WORK_SIZE);
            send(&adsp, &agic, OPUS, channels);
            assert_eq!(
                receive(&adsp, &agic),
                FULL | OPUS << 20 | OPUS_WORK_SIZE_DONE
            );
            assert_eq!(word_at(&ram, buffer + OPUS_RESULT), size);
        }
        assert_eq!(
            adsp.write(0, TO_DSP, 4, u64::from(FULL | OPUS << 20 | 24)),
            None
        );
    }

    #[test]
    fn renderer_handshake_and_frames() {
        let (adsp, ram, agic, _) = dsp();
        adsp.release();
        receive(&adsp, &agic);
        while host_dequeue(&ram, DSP_QUEUE).is_some() {}
        let mut request = [0u32; 1 + LAUNCH_WORDS];
        request[0] = LAUNCH_WORDS as u32;
        request[1] = RENDERER_APP;
        request[1 + LAUNCH_SHARED_DRAM] = 0x8090_0000;
        request[1 + LAUNCH_COMMAND] = LAUNCH_RUN;
        adsp.0.enqueue(HOST_QUEUE, &request).unwrap();
        send(&adsp, &agic, APP_SERVICE, 0);
        receive(&adsp, &agic);
        send(&adsp, &agic, RENDERER, RENDER_INITIALIZE);
        assert_eq!(
            receive(&adsp, &agic),
            FULL | RENDERER << 20 | RENDER_INITIALIZE
        );
        assert_eq!(receive(&adsp, &agic), FULL | RENDERER << 20 | RENDER_DONE);
        send(&adsp, &agic, RENDERER_MEMORY, 1);
        assert_eq!(receive(&adsp, &agic), FULL | RENDERER_MEMORY << 20 | 2);
        // A frame with a command list in session 0.
        ram.write(0x8090_0000, &0x8100_0000u32.to_le_bytes())
            .unwrap();
        ram.write(0x8090_000C, &5u32.to_le_bytes()).unwrap();
        let started = Instant::now();
        for _ in 0..3 {
            send(&adsp, &agic, RENDERER, RENDER);
            while agic.cpu_interface().read(0, 0x18, 4) != Some(u64::from(TO_HOST_FULL)) {
                thread::sleep(Duration::from_millis(1));
            }
            assert_eq!(receive(&adsp, &agic), FULL | RENDERER << 20 | RENDER_DONE);
        }
        assert!(started.elapsed() >= 2 * FRAME, "frames come in real time");
        assert_eq!(word_at(&ram, 0x8090_000C), 0, "no commands left");
    }
}
