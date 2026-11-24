//! What the `AudioRenderer` application's command lists do to the voices
//! they play, without the sound: each frame the DSP walks every voice's
//! wave buffers as far as the frame's samples take it and leaves the
//! voice's state (samples played, the buffer and offset it is at, the
//! buffers consumed) for `audio` to report to the game, which refills and
//! paces its streams by it. Addresses are DSP addresses in the image (see
//! [`super`]).
//!
//! # Command lists (`ProcessCommands`, `0x8050F760`; set-up `0x8050EFD4`)
//!
//! A list starts with a 24-byte header: its size, its command count, the
//! mix buffers' address and count, the samples a frame renders and their
//! rate (words 0 to 5). Commands follow, each `{u8 flags, u8 type, u16
//! size, ...}`; one runs when bit 0 of its flags is set (`DispatchCommand`,
//! `0x8050F03C`, checks each type's size).
//!
//! # Wave buffer data sources (types 2, 4 and 6, `0x8050C3B6`,
//! `0x8050C5E8`, `0x8050C81C`)
//!
//! A 208-byte command: at +16 the resampler's quality (byte), +22 flags
//! (bit 0: a looping buffer that restarts resets the samples played; bit
//! 1: the voice skips pitch and sample rate conversion), +24 the source
//! sample rate, +28 the pitch (Q15), +40 four 40-byte wave buffers, +200
//! the voice's state. Type 2 plays PCM16, 4 float and 6 ADPCM (with the
//! coefficients' address at +204).
//!
//! A wave buffer: data address (+0), ADPCM context address (+4), data size
//! (+8), start and end sample (+16, +20), loop start and end (+24, +28),
//! loop count (+32, negative: forever), and bytes for looping (+36) and
//! end of stream (+37).
//!
//! The voice state: samples played (u64, +0), the offset in the current
//! buffer (+8), the current buffer (+12), the resampler's fraction (Q15,
//! +32), the ADPCM context (+36, 6 bytes), each buffer's valid byte (+52),
//! the buffers consumed (+56) and the current buffer's loops done (+240).
//!
//! `DecodeFromWaveBuffersImpl` (`0x8050F848`) renders the frame in chunks
//! of at most as many output samples as 16128 source samples hold, each
//! needing `(fraction + chunk * step) >> 15` source samples at `step =
//! (int)(source rate / output rate * pitch)` (floats), taken from the
//! buffers in turn; the resampler (`0x8051DD22`) leaves the fraction at
//! `(fraction + chunk * step) & 0x7FFF`, a voice that skips conversion
//! leaves it as it was. Its decoders (`0x805102A2`, `0x80510360`,
//! `0x80510584`) give `min(end - start - offset, wanted)` samples of a
//! buffer, none of one without data or with its end before its start
//! (ADPCM: nor of one whose data is shorter than its end sample needs).
//! Only the counts are modeled here: no sample is decoded or mixed.

/// The renderer's view of memory: DSP addresses.
pub(super) trait DspMemory {
    fn read(&self, address: u32, bytes: &mut [u8]) -> Result<(), String>;
    fn write(&self, address: u32, bytes: &[u8]) -> Result<(), String>;
}

const HEADER_BYTES: usize = 24;
const WAVE_BUFFERS: u32 = 4;
const WAVE_BUFFER_BYTES: u32 = 40;
/// The most source samples a chunk takes (`0x1F800000 >> 15`).
const CHUNK_SOURCE_SAMPLES: i32 = 16128;

/// Sample formats as the decoders take them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Format {
    Pcm16,
    Float,
    Adpcm,
}

/// Runs the voices of the command list of `size` bytes at `list`; one
/// shorter than its header runs nothing.
pub(super) fn run(memory: &impl DspMemory, list: u32, size: u32) -> Result<(), String> {
    if (size as usize) < HEADER_BYTES {
        return Ok(());
    }
    let mut bytes = vec![0; size as usize];
    memory.read(list, &mut bytes)?;
    let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().expect("4 bytes"));
    let count = word(4);
    let frame = Frame {
        samples: word(16) as i32,
        rate: word(20),
    };
    let mut at = HEADER_BYTES;
    for _ in 0..count {
        let Some(header) = bytes.get(at..at + 4) else {
            return Err(format!("command at {at:#x} past the list's {size} bytes"));
        };
        let (flags, kind) = (header[0], header[1]);
        let length = usize::from(u16::from_le_bytes([header[2], header[3]]));
        if length == 0 || at + length > bytes.len() {
            return Err(format!("command at {at:#x} of {length} bytes"));
        }
        let command = &bytes[at..at + length];
        if flags & 1 != 0 {
            let format = match (kind, length) {
                (2, 208) => Some(Format::Pcm16),
                (4, 208) => Some(Format::Float),
                (6, 208) => Some(Format::Adpcm),
                _ => None,
            };
            if let Some(format) = format {
                let source = Source::parse(format, command, list + at as u32);
                decode_from_wave_buffers(memory, &source, &frame)?;
            }
        }
        at += length;
    }
    Ok(())
}

/// The frame a list renders.
struct Frame {
    samples: i32,
    rate: u32,
}

/// A data source command's operands.
struct Source {
    format: Format,
    quality: u8,
    reset_played_at_loop: bool,
    skip_conversion: bool,
    source_rate: u32,
    pitch: u32,
    wave_buffers: u32,
    state: u32,
}

impl Source {
    fn parse(format: Format, command: &[u8], address: u32) -> Self {
        let word = |at: usize| u32::from_le_bytes(command[at..at + 4].try_into().expect("4 bytes"));
        Self {
            format,
            quality: command[16],
            reset_played_at_loop: command[22] & 1 != 0,
            skip_conversion: command[22] & 2 != 0,
            source_rate: word(24),
            pitch: word(28),
            wave_buffers: address + 40,
            state: word(200),
        }
    }
}

/// A wave buffer's words.
struct WaveBuffer {
    data: u32,
    context: u32,
    size: u32,
    start: i32,
    end: i32,
    loop_start: i32,
    loop_end: i32,
    loop_count: i32,
    looping: bool,
    end_of_stream: bool,
}

impl WaveBuffer {
    fn read(memory: &impl DspMemory, address: u32) -> Result<Self, String> {
        let mut b = [0; WAVE_BUFFER_BYTES as usize];
        memory.read(address, &mut b)?;
        let word = |at: usize| u32::from_le_bytes(b[at..at + 4].try_into().expect("4 bytes"));
        Ok(Self {
            data: word(0),
            context: word(4),
            size: word(8),
            start: word(16) as i32,
            end: word(20) as i32,
            loop_start: word(24) as i32,
            loop_end: word(28) as i32,
            loop_count: word(32) as i32,
            looping: b[36] == 1,
            end_of_stream: b[37] != 0,
        })
    }

    /// The samples its decoder gives at `offset` from `start` of those
    /// `wanted`.
    fn decoded(&self, format: Format, start: i32, end: i32, offset: i32, wanted: i32) -> i32 {
        if self.data == 0 || self.size == 0 || end < start {
            return 0;
        }
        if format == Format::Adpcm {
            // Bytes to the end sample: 8-byte frames of 14 samples, each
            // after a header byte pair (`0x80510584`).
            let nibbles = end % 14 + 16 * (end / 14) + if end % 14 == 0 { 1 } else { 3 };
            if (self.size as i32) < nibbles / 2 {
                return 0;
            }
        }
        (end - (start + offset)).min(wanted)
    }
}

/// `DecodeFromWaveBuffersImpl`'s walk of the buffers, for the counts.
fn decode_from_wave_buffers(memory: &impl DspMemory, source: &Source, frame: &Frame) -> Result<(), String> {
    let state = source.state;
    let mut head = [0u8; 60];
    memory.read(state, &mut head)?;
    let word = |at: usize| i32::from_le_bytes(head[at..at + 4].try_into().expect("4 bytes"));
    let mut played = u64::from_le_bytes(head[0..8].try_into().expect("8 bytes"));
    let mut offset = word(8);
    let mut index = word(12);
    let mut fraction = word(32);
    let mut consumed = word(56);
    let mut valid = [head[52], head[53], head[54], head[55]];
    let mut loops = {
        let mut b = [0; 4];
        memory.read(state + 240, &mut b)?;
        i32::from_le_bytes(b)
    };

    let step = (source.source_rate as f32 / frame.rate as f32 * source.pitch as f32) as i32;
    let taps = if source.quality == 1 { 8 } else { 4 };
    let total = fraction.wrapping_add(frame.samples.wrapping_mul(step));
    if total >> 15 < 0 || taps + (total >> 15) > CHUNK_SOURCE_SAMPLES {
        return Ok(());
    }
    // A pitch of 0 takes no source samples \[assumption: the DSP's divide
    // by zero is never reached by a voice that plays\].
    let per_chunk = match step {
        0 => frame.samples,
        _ => ((CHUNK_SOURCE_SAMPLES << 15) - fraction) / step,
    }
    .min(frame.samples);
    let mut remaining = frame.samples;
    let mut stopped = false;
    while remaining > 0 && !stopped {
        let chunk = remaining.min(per_chunk);
        if chunk <= 0 {
            break;
        }
        let needed = (fraction + chunk * step) >> 15;
        let mut decoded = 0;
        while decoded < needed {
            if !(0..WAVE_BUFFERS as i32).contains(&index) {
                played = 0;
                stopped = true;
                offset = 0;
                index = 0;
                valid = [0; 4];
                consumed = WAVE_BUFFERS as i32;
                break;
            }
            if valid[index as usize] == 0 {
                stopped = true;
                break;
            }
            let buffer = WaveBuffer::read(memory, source.wave_buffers + WAVE_BUFFER_BYTES * index as u32)?;
            if offset == 0 && source.format == Format::Adpcm && buffer.context != 0 {
                let mut context = [0; 6];
                memory.read(buffer.context, &mut context)?;
                memory.write(state + 36, &context)?;
            }
            let (start, end) = if buffer.looping && loops >= 1 && buffer.loop_end >= buffer.loop_start {
                (buffer.loop_start, buffer.loop_end)
            } else {
                (buffer.start, buffer.end)
            };
            let n = buffer.decoded(source.format, start, end, offset, needed - decoded);
            decoded += n;
            played = played.wrapping_add_signed(i64::from(n));
            if n != 0 && n + offset < end - start {
                // Inside the buffer still.
                offset += n;
                continue;
            }
            // The buffer is done, or gave nothing.
            if !buffer.looping {
                offset = 0;
                consume(&mut loops, &mut valid, &mut index, &mut consumed, &mut played, &buffer);
                continue;
            }
            let done = loops;
            loops += 1;
            if buffer.loop_count >= 0 && (n == 0 || done >= buffer.loop_count) {
                consume(&mut loops, &mut valid, &mut index, &mut consumed, &mut played, &buffer);
            }
            offset = 0;
            if n == 0 {
                stopped = true;
                break;
            }
            if source.reset_played_at_loop {
                played = 0;
            }
        }
        if !source.skip_conversion {
            fraction = (fraction + chunk * step) & 0x7FFF;
        }
        remaining -= chunk;
    }

    let mut out = [0u8; 16];
    out[0..8].copy_from_slice(&played.to_le_bytes());
    out[8..12].copy_from_slice(&offset.to_le_bytes());
    out[12..16].copy_from_slice(&index.to_le_bytes());
    memory.write(state, &out)?;
    memory.write(state + 32, &fraction.to_le_bytes())?;
    memory.write(state + 56, &consumed.to_le_bytes())?;
    // Only the valid bytes the walk cleared: `audio` sets those of the
    // buffers it appends, while the DSP runs.
    for (slot, &byte) in valid.iter().enumerate() {
        if byte == 0 && head[52 + slot] != 0 {
            memory.write(state + 52 + slot as u32, &[0])?;
        }
    }
    memory.write(state + 240, &loops.to_le_bytes())
}

/// A buffer played out: the next one is current, and the end of a stream
/// starts the count of samples played over.
fn consume(
    loops: &mut i32,
    valid: &mut [u8; 4],
    index: &mut i32,
    consumed: &mut i32,
    played: &mut u64,
    buffer: &WaveBuffer,
) {
    *loops = 0;
    valid[*index as usize] = 0;
    *index = (*index + 1) % WAVE_BUFFERS as i32;
    if buffer.end_of_stream {
        *played = 0;
    }
    *consumed += 1;
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::HashMap;

    use super::*;

    /// DSP memory as bytes by address.
    #[derive(Default)]
    struct Memory(RefCell<HashMap<u32, u8>>);

    impl DspMemory for Memory {
        fn read(&self, address: u32, bytes: &mut [u8]) -> Result<(), String> {
            let memory = self.0.borrow();
            for (i, byte) in bytes.iter_mut().enumerate() {
                *byte = *memory.get(&(address + i as u32)).unwrap_or(&0);
            }
            Ok(())
        }

        fn write(&self, address: u32, bytes: &[u8]) -> Result<(), String> {
            let mut memory = self.0.borrow_mut();
            for (i, &byte) in bytes.iter().enumerate() {
                memory.insert(address + i as u32, byte);
            }
            Ok(())
        }
    }

    impl Memory {
        fn words(&self, address: u32, words: &[u32]) {
            let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            self.write(address, &bytes).unwrap();
        }

        fn word(&self, address: u32) -> u32 {
            let mut b = [0; 4];
            self.read(address, &mut b).unwrap();
            u32::from_le_bytes(b)
        }
    }

    const LIST: u32 = 0x1000;
    const STATE: u32 = 0x8000;

    /// A list of one ADPCM voice (type 6) at `pitch` (Q15) with `buffers`
    /// (samples, looping, loop count, end of stream), each 0x2000 bytes of
    /// data, all valid.
    fn voice(memory: &Memory, pitch: u32, buffers: &[(i32, bool, i32, bool)]) {
        let mut command = vec![0u32; 52];
        command[0] = 1 | 6 << 8 | 208 << 16;
        command[6] = 48000;
        command[7] = pitch;
        for (i, &(samples, looping, count, end)) in buffers.iter().enumerate() {
            let wb = 10 + 10 * i;
            command[wb] = 0x10_0000 + 0x2000 * i as u32;
            command[wb + 2] = 0x2000;
            command[wb + 5] = samples as u32;
            command[wb + 7] = samples as u32;
            command[wb + 8] = count as u32;
            command[wb + 9] = u32::from(looping) | u32::from(end) << 8;
        }
        command[50] = STATE;
        let mut list = vec![24 + 208, 1, 0, 0, 240, 48000];
        list.extend(command);
        memory.words(LIST, &list);
        let valid: Vec<u8> = (0..4).map(|i| u8::from(i < buffers.len())).collect();
        memory.write(STATE + 52, &valid).unwrap();
    }

    fn frames(memory: &Memory, count: u32) {
        for _ in 0..count {
            run(memory, LIST, 24 + 208).unwrap();
        }
    }

    /// (samples played, offset, buffer, consumed, valid bytes).
    fn state(memory: &Memory) -> (u32, u32, u32, u32, u32) {
        (
            memory.word(STATE),
            memory.word(STATE + 8),
            memory.word(STATE + 12),
            memory.word(STATE + 56),
            memory.word(STATE + 52),
        )
    }

    /// At pitch 1 a frame takes its 240 samples: two 0x3800-sample buffers
    /// play out in turn, the voice stops when none is valid, and an end of
    /// stream restarts the count of samples played.
    #[test]
    fn frames_consume_wave_buffers_in_turn() {
        let memory = Memory::default();
        voice(&memory, 0x8000, &[(0x3800, false, 0, false), (0x3800, false, 0, true)]);
        frames(&memory, 10);
        assert_eq!(state(&memory), (2400, 2400, 0, 0, 0x0101));
        // 0x3800 = 14336 samples: the first buffer is done in frame 60.
        frames(&memory, 50);
        assert_eq!(state(&memory), (14400, 64, 1, 1, 0x0100));
        // The second ends the stream at 28672 samples: played starts over,
        // the third buffer is current, and with it invalid the voice runs
        // dry (its last frame takes what is left).
        frames(&memory, 60);
        assert_eq!(state(&memory), (0, 0, 2, 2, 0));
    }

    /// Half pitch takes 120 samples a frame; a buffer looping twice plays
    /// three times before it is consumed.
    #[test]
    fn pitch_and_loops_set_the_pace() {
        let memory = Memory::default();
        voice(&memory, 0x4000, &[(2400, true, 2, false)]);
        frames(&memory, 10);
        assert_eq!(state(&memory).0, 1200);
        frames(&memory, 50);
        // 3 * 2400 samples = 60 frames of 120.
        assert_eq!(state(&memory), (7200, 0, 1, 1, 0));
    }
}
