//! `gmix`'s audio output sessions: the part of the DSP's mixer that plays
//! the buffers `audio`'s audio output service (`audout`) hands it, as far
//! as `audio` observes it. See [`super::adsp`] for the DSP and how `gmix`
//! is launched; offsets below are DSP addresses in its firmware image.
//!
//! # What `audio` relies on
//!
//! `audio` keeps one output queue per audio output session (target 2, 12
//! sessions; `gmix`'s `GetSession`, `0x80501524`) and talks to it with
//! commands in `gmix`'s command queue (`audio` text `0x14700` onwards; the
//! DSP's dispatcher `0x80500B0C`): open with the session's routing and
//! channel count (8), close (9), start and stop (10, 11), pause and resume
//! (16, 17), append a buffer (15, its device address and size), release
//! everything at the next buffer boundary (18), and gains (12 to 14).
//!
//! What comes back is a 24-byte descriptor per session in `gmix`'s shared
//! memory (`0x2B060` + 24 × session; `0x80500DB4`): the number of buffers
//! appended (word 0, written on each append, `0x8050328E`), the number
//! played and released (word 1), the read position in the current buffer
//! in bytes (word 2) and the frames played so far (words 4 and 5,
//! `0x805033E8`). `audio` releases a client's buffers up to word 1 (text
//! `0x14d10`), and when it closes a session it waits, polling every 5 ms
//! without a time limit, until word 1 equals the number it appended (text
//! `0x14c30`) — while holding its audio output manager's lock, so every
//! audio service request that needs that lock waits as well. The DSP
//! releases them because closing first plays out every queued buffer
//! (`0x80503158`, which consumes each whole buffer, `0x8050317C`).
//!
//! # Devices
//!
//! `audio` opens, starts and closes `gmix`'s two output devices (target 7,
//! index 0 the speaker, index 1 the second device) with the same commands.
//! Each device has an area in `gmix`'s shared memory, `0x2B2B8` and
//! `0x2BBD0` (`0x80500DB4`; `audio` text `0x135bc` and `0x136bc` keep the
//! same addresses), whose word at `0x910` is the device's state: `Open`
//! (`0x80503828`) sets it to 2 and `Close` (`0x805038B0`) to 1, after
//! counting the close (`0x80502C46`, which asserts the device is open).
//! When the system goes to sleep, `audio`'s power-state handler closes the
//! speaker (text `0x15680`, on psc's request for minimum awake) and polls
//! every 5 ms, without a time limit, until that word is 1; psc does not
//! let the sleep go on before `audio` acknowledges.
//!
//! # Playing
//!
//! `gmix` mixes the sessions into the speaker output (target 7, index 0,
//! the device `audio` opens and starts with the same commands) each time
//! that device's DMA has consumed a period, as many frames as it consumed
//! (`0x80502984`, `0x805019C0`, `0x80501F34`): a session is played if it is
//! open, started and not paused, has a buffer, and is routed to the
//! speaker (its routing is 7, or 0 while the default device set by
//! command 1 is 7; `0x80501ED6`). A buffer whose last frame is played is
//! released, and a release-all request releases the rest with it. After
//! each period `gmix` tells `audio` with a notification record
//! (`[1, 0, 1, 0, 0]`: output sessions played) and message 1 on its
//! channel; `audio` then refreshes its output queues (text `0x138b0`).
//!
//! Here the speaker's DMA is not modeled: a period is 5 ms, 240 frames at
//! 48 kHz, so sessions advance in real time. The samples are not read and
//! no sound is produced. If `audio` has not taken the previous
//! notification yet, a period passes without one; its record would only
//! repeat that sessions played.
//!
//! Commands for other targets (renderer, capture, near-voice and game
//! recording sessions) are taken without effect, as before; no client of
//! those is modeled.

/// `gmix`'s shared memory, as the DSP reaches it.
pub trait Shared {
    fn read(&self, offset: u32) -> Result<u32, String>;
    fn write(&mut self, offset: u32, value: u32) -> Result<(), String>;
}

const OUTPUT_SESSIONS: usize = 12;
const OUTPUT_TARGET: u32 = 2;
const DEVICE_TARGET: u32 = 7;
/// The DSP's queue of buffers per session (`0x8050328E`: four slots).
const QUEUE_SLOTS: usize = 4;
/// Descriptor of output session `n`: `0x2B060 + 24 n` (176224 in
/// `0x80500DB4`).
const DESCRIPTORS: u32 = 0x2_B060;
/// Each device's area in `gmix`'s shared memory (`0x80500DB4`), and the
/// offset of its state word in it.
const DEVICE_AREAS: [u32; 2] = [0x2_B2B8, 0x2_BBD0];
const DEVICE_STATE: u32 = 0x910;
const DEVICE_OPEN: u32 = 2;
const DEVICE_CLOSED: u32 = 1;
/// One period of the speaker's DMA, in frames.
pub const PERIOD_FRAMES: u32 = 240;
/// The notification record of a period in which output sessions played.
pub const NOTIFICATION: [u32; 6] = [5, 1, 0, 1, 0, 0];

const OPEN: u32 = 8;
const CLOSE: u32 = 9;
const START: u32 = 10;
const STOP: u32 = 11;
const GAIN: u32 = 12;
const GAIN_RAMP: u32 = 13;
const GAIN_RAMP_2: u32 = 14;
const APPEND: u32 = 15;
const PAUSE: u32 = 16;
const RESUME: u32 = 17;
const RELEASE_ALL: u32 = 18;
const DEFAULT_DEVICE: u32 = 1;

/// Open, start and pause counts as the DSP keeps them for a session or a
/// device (object offsets 4, 72 and 76).
#[derive(Default, Clone, Copy)]
struct Counts {
    open: u32,
    started: u32,
    paused: u32,
}

impl Counts {
    fn playing(&self) -> bool {
        self.open > 0 && self.started > 0 && self.paused == 0
    }

    /// Closing the last reference also clears the start and pause counts
    /// (`0x80502C46`).
    fn close(&mut self) -> Result<(), String> {
        self.open = self.open.checked_sub(1).ok_or("closing what is not open")?;
        if self.open == 0 {
            self.started = 0;
            self.paused = 0;
        }
        Ok(())
    }

    fn count(&mut self, command: u32) -> Result<(), String> {
        let (count, up) = match command {
            START => (&mut self.started, true),
            STOP => (&mut self.started, false),
            PAUSE => (&mut self.paused, true),
            _ => (&mut self.paused, false),
        };
        *count = if up {
            *count + 1
        } else {
            count
                .checked_sub(1)
                .ok_or("stopping or resuming too often")?
        };
        Ok(())
    }
}

#[derive(Default, Clone)]
struct Session {
    counts: Counts,
    /// Routing: 0 (the default device), 7 or 8.
    routing: u32,
    channels: u32,
    /// Sizes of the queued buffers, the one playing first.
    queue: Vec<u32>,
    /// Bytes of the first buffer played.
    read: u32,
    release_all: bool,
}

#[derive(Default)]
pub struct Streams {
    default_device: u32,
    sessions: [Session; OUTPUT_SESSIONS],
    /// The devices' counts; index 0 is the speaker.
    devices: [Counts; 2],
}

impl Streams {
    /// Whether the speaker plays, so that periods pass.
    pub fn playing(&self) -> bool {
        self.devices[0].playing()
    }

    /// One command record (opcode first, without the word count).
    pub fn command(&mut self, record: &[u32], shared: &mut impl Shared) -> Result<(), String> {
        let word = |index: usize| {
            record
                .get(index)
                .copied()
                .ok_or_else(|| format!("gmix command {record:x?} is too short"))
        };
        let opcode = word(0)?;
        if opcode == DEFAULT_DEVICE {
            self.default_device = word(1)?;
            return Ok(());
        }
        let (target, index) = match opcode {
            OPEN..=RELEASE_ALL => (word(1)?, word(2)? as usize),
            _ => return Ok(()),
        };
        if target == DEVICE_TARGET {
            let counts = self
                .devices
                .get_mut(index)
                .ok_or_else(|| format!("gmix has no device {index}"))?;
            let state = DEVICE_AREAS[index] + DEVICE_STATE;
            return match opcode {
                OPEN => {
                    counts.open += 1;
                    shared.write(state, DEVICE_OPEN)
                }
                CLOSE => {
                    counts.close()?;
                    shared.write(state, DEVICE_CLOSED)
                }
                START | STOP | PAUSE | RESUME => counts.count(opcode),
                // Gains and the mute setting (`0x80502CFA`): no sound.
                GAIN | GAIN_RAMP | GAIN_RAMP_2 => Ok(()),
                _ => Err(format!(
                    "gmix command {opcode} for device {index} is not modeled"
                )),
            };
        }
        if target != OUTPUT_TARGET {
            return Ok(());
        }
        let session = self
            .sessions
            .get_mut(index)
            .ok_or_else(|| format!("gmix has no output session {index}"))?;
        let descriptor = DESCRIPTORS + 24 * index as u32;
        match opcode {
            // `Open` (`0x80503082`) asserts the session is not open.
            OPEN if session.counts.open == 0 => {
                let channels = word(4)?;
                if channels == 0 {
                    return Err("an output session of no channels".into());
                }
                *session = Session {
                    counts: Counts {
                        open: 1,
                        ..Counts::default()
                    },
                    routing: word(3)?,
                    channels,
                    ..Session::default()
                };
            }
            OPEN => return Err(format!("output session {index} is already open")),
            CLOSE => {
                let appended = shared.read(descriptor)?;
                shared.write(descriptor + 4, appended)?;
                shared.write(descriptor + 8, 0)?;
                session.queue.clear();
                session.read = 0;
                session.counts.close()?;
            }
            START | STOP | PAUSE | RESUME => session.counts.count(opcode)?,
            GAIN | GAIN_RAMP | GAIN_RAMP_2 => {}
            APPEND => {
                let size = word(4)?;
                // `AppendBuffer` asserts whole frames.
                if session.counts.open == 0
                    || session.queue.len() == QUEUE_SLOTS
                    || size == 0
                    || size % (2 * session.channels) != 0
                {
                    return Err(format!(
                        "appending {size} bytes to output session {index} is not modeled"
                    ));
                }
                session.queue.push(size);
                let appended = shared.read(descriptor)?;
                shared.write(descriptor, appended.wrapping_add(1))?;
            }
            _ => session.release_all = true,
        }
        Ok(())
    }

    /// One period of the speaker: plays every session routed to it.
    pub fn period(&mut self, shared: &mut impl Shared) -> Result<(), String> {
        let default_device = self.default_device;
        for (index, session) in self.sessions.iter_mut().enumerate() {
            let routed = session.routing == DEVICE_TARGET
                || session.routing == 0 && default_device == DEVICE_TARGET;
            if !routed || !session.counts.playing() {
                continue;
            }
            let descriptor = DESCRIPTORS + 24 * index as u32;
            let frame = 2 * session.channels;
            let mut frames = PERIOD_FRAMES;
            let mut played = 0;
            while frames > 0 && !session.queue.is_empty() {
                let left = (session.queue[0] - session.read) / frame;
                let now = frames.min(left);
                frames -= now;
                played += now;
                session.read += now * frame;
                if session.read == session.queue[0] {
                    let released = if session.release_all {
                        session.queue.len()
                    } else {
                        1
                    };
                    session.queue.drain(..released);
                    session.read = 0;
                    session.release_all = false;
                    let count = shared.read(descriptor + 4)?;
                    shared.write(descriptor + 4, count.wrapping_add(released as u32))?;
                }
            }
            shared.write(descriptor + 8, session.read)?;
            let total = u64::from(shared.read(descriptor + 16)?)
                | u64::from(shared.read(descriptor + 20)?) << 32;
            let total = total.wrapping_add(u64::from(played));
            shared.write(descriptor + 16, total as u32)?;
            shared.write(descriptor + 20, (total >> 32) as u32)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[derive(Default)]
    struct Memory(HashMap<u32, u32>);

    impl Shared for Memory {
        fn read(&self, offset: u32) -> Result<u32, String> {
            Ok(self.0.get(&offset).copied().unwrap_or(0))
        }

        fn write(&mut self, offset: u32, value: u32) -> Result<(), String> {
            self.0.insert(offset, value);
            Ok(())
        }
    }

    const D: u32 = DESCRIPTORS + 24 * 3;

    #[test]
    fn closing_plays_out_every_queued_buffer() {
        let (mut streams, mut memory) = (Streams::default(), Memory::default());
        streams
            .command(&[OPEN, 2, 3, 0, 2, 0], &mut memory)
            .unwrap();
        for _ in 0..3 {
            streams
                .command(&[APPEND, 2, 3, 0x8000_0000, 0x1000], &mut memory)
                .unwrap();
        }
        assert_eq!(
            (memory.read(D).unwrap(), memory.read(D + 4).unwrap()),
            (3, 0)
        );
        // Nothing plays without the speaker: closing still releases all.
        streams.command(&[START, 2, 3], &mut memory).unwrap();
        streams.period(&mut memory).unwrap();
        streams.command(&[CLOSE, 2, 3], &mut memory).unwrap();
        assert_eq!(memory.read(D + 4).unwrap(), 3);
        assert!(
            streams
                .command(&[APPEND, 2, 3, 0, 0x1000], &mut memory)
                .is_err()
        );
        streams
            .command(&[OPEN, 2, 3, 7, 2, 0], &mut memory)
            .unwrap();
    }

    #[test]
    fn the_speaker_plays_routed_sessions_in_periods() {
        let (mut streams, mut memory) = (Streams::default(), Memory::default());
        streams.command(&[DEFAULT_DEVICE, 7], &mut memory).unwrap();
        streams.command(&[OPEN, 7, 0], &mut memory).unwrap();
        streams.command(&[START, 7, 0], &mut memory).unwrap();
        assert!(streams.playing());
        streams
            .command(&[OPEN, 2, 3, 0, 2, 0], &mut memory)
            .unwrap();
        streams.command(&[START, 2, 3], &mut memory).unwrap();
        // 480 stereo frames, then 240.
        streams
            .command(&[APPEND, 2, 3, 0x8000_0000, 1920], &mut memory)
            .unwrap();
        streams
            .command(&[APPEND, 2, 3, 0x8000_1000, 960], &mut memory)
            .unwrap();
        streams.period(&mut memory).unwrap();
        assert_eq!(
            (memory.read(D + 4).unwrap(), memory.read(D + 8).unwrap()),
            (0, 960)
        );
        streams.command(&[PAUSE, 2, 3], &mut memory).unwrap();
        streams.period(&mut memory).unwrap();
        assert_eq!(memory.read(D + 8).unwrap(), 960, "paused");
        streams.command(&[RESUME, 2, 3], &mut memory).unwrap();
        streams.period(&mut memory).unwrap();
        assert_eq!(
            (memory.read(D + 4).unwrap(), memory.read(D + 8).unwrap()),
            (1, 0)
        );
        streams.period(&mut memory).unwrap();
        assert_eq!(memory.read(D + 4).unwrap(), 2);
        assert_eq!(memory.read(D + 16).unwrap(), 720);
        // Underrun: nothing to play, nothing changes.
        streams.period(&mut memory).unwrap();
        assert_eq!(memory.read(D + 16).unwrap(), 720);
        streams.command(&[STOP, 7, 0], &mut memory).unwrap();
        assert!(!streams.playing());
    }

    #[test]
    fn release_all_takes_the_rest_at_the_buffer_boundary() {
        let (mut streams, mut memory) = (Streams::default(), Memory::default());
        streams
            .command(&[OPEN, 2, 3, 7, 1, 0], &mut memory)
            .unwrap();
        streams.command(&[START, 2, 3], &mut memory).unwrap();
        for _ in 0..3 {
            streams
                .command(&[APPEND, 2, 3, 0, 480], &mut memory)
                .unwrap();
        }
        streams.command(&[RELEASE_ALL, 2, 3], &mut memory).unwrap();
        streams.period(&mut memory).unwrap();
        assert_eq!(memory.read(D + 4).unwrap(), 3);
    }

    /// `audio` waits without a time limit for a closed device's state word
    /// before it lets the system sleep.
    #[test]
    fn devices_report_open_and_closed_in_their_state_word() {
        let (mut streams, mut memory) = (Streams::default(), Memory::default());
        let speaker = 0x2_B2B8 + 0x910;
        streams.command(&[OPEN, 7, 0], &mut memory).unwrap();
        assert_eq!(memory.read(speaker).unwrap(), 2);
        streams.command(&[START, 7, 0], &mut memory).unwrap();
        streams.command(&[CLOSE, 7, 0], &mut memory).unwrap();
        assert_eq!(memory.read(speaker).unwrap(), 1);
        assert!(!streams.playing(), "closing the last reference stops it");
        streams.command(&[OPEN, 7, 1], &mut memory).unwrap();
        assert_eq!(memory.read(0x2_BBD0 + 0x910).unwrap(), 2);
        assert!(streams.command(&[CLOSE, 7, 0], &mut memory).is_err());
        assert!(streams.command(&[OPEN, 7, 2], &mut memory).is_err());
    }
}
