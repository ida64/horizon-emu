//! The `OpusDecoder` application's decoding requests (`0x8051FD08`), served
//! by libopus. `audio` (for `hwopus`) leaves each request's operands in
//! the 4096-byte buffer it shares with the decoder (see [`super`]) and
//! sends its number; the decoder leaves its result at `0xFC0` and answers
//! with the number plus 20.
//!
//! | Message | Request | Operands from `0xF80` |
//! |---|---|---|
//! | 22 | initialize a decoder | work buffer, its size, sample rate, channels |
//! | 23 | finalize it | work buffer |
//! | 24 | decode a packet | work buffer, packet, its size, output, frame size, final range, reset |
//! | 25, 26 | clean, invalidate the work buffer's cache lines | work buffer, its size |
//! | 27 | work buffer size of a multistream decoder | streams, coupled streams |
//! | 28 | initialize a multistream decoder | work buffer, its size, sample rate, channels, streams, coupled streams, mapping (bytes) |
//! | 29 | finalize it | work buffer |
//! | 30 | decode a packet | as 24 |
//!
//! A decode answers the samples per channel at `0xFC4` and its time in
//! microseconds at `0xFC8` (none here). The image's libopus
//! (`opus_decode` at `0x8052B4B4`, `opus_multistream_decode` at
//! `0x80544B0E`) takes the packet's bytes and the frame size as given, and
//! a reset request (`OPUS_RESET_STATE`) comes before the decode. When the
//! request names a final range, the decoder's (`OPUS_GET_FINAL_RANGE`)
//! must equal it, else the result is -4 (`0x8051FD08`).
//!
//! The work buffer starts with the dispatcher's 20 bytes
//! (`0x80520524`/`0x80520552`): `0xDEADBEEF`, an initialized byte, a byte
//! set when the buffer is the one initialized (else requests on it fail
//! with -6), the buffer's own address, its decoder's, and the final range
//! of the last packet. The image's libopus state follows; here instead the
//! decoder's parameters do (kind, sample rate, channels, streams, coupled
//! streams, then the mapping), so that a decoder lost with the host (a
//! restored machine) is made again from them. A decoder made again starts
//! from a reset state; a packet's final range does not depend on it.

use std::{collections::HashMap, ffi::c_int, ptr::NonNull};

use audiopus_sys as ffi;

use super::{DspMemory, Result};

const MAGIC: u32 = 0xDEAD_BEEF;
/// The dispatcher's header in a work buffer.
const HEADER: u32 = 20;
/// The most samples per channel a packet holds (120 ms at 48 kHz).
const MOST_SAMPLES: usize = 5760;
/// `OPUS_RESET_STATE`, `OPUS_GET_FINAL_RANGE_REQUEST`.
const RESET_STATE: c_int = 4028;
const GET_FINAL_RANGE: c_int = 4031;
/// `OPUS_INTERNAL_ERROR`: a final range not the one expected.
const RANGE_MISMATCH: i32 = -4;
/// The dispatcher's answer to requests on a buffer not initialized as it.
const NOT_READY: i32 = -6;

/// Operands in the shared buffer.
const OPERANDS: u32 = 0xF80;
pub(super) const RESULT: u32 = 0xFC0;
const SAMPLES: u32 = 0xFC4;
const TIME: u32 = 0xFC8;

/// libopus decoders by work buffer.
#[derive(Default)]
pub(super) struct Decoders(HashMap<u32, Decoder>);

/// A libopus decoder.
enum Decoder {
    Single(NonNull<ffi::OpusDecoder>),
    Multi(NonNull<ffi::OpusMSDecoder>),
}

// SAFETY: a libopus decoder is plain memory, used by one thread at a time
// (the map's owner holds a lock).
unsafe impl Send for Decoder {}

impl Drop for Decoder {
    fn drop(&mut self) {
        // SAFETY: made by the matching create, dropped once.
        unsafe {
            match self {
                Self::Single(decoder) => ffi::opus_decoder_destroy(decoder.as_ptr()),
                Self::Multi(decoder) => ffi::opus_multistream_decoder_destroy(decoder.as_ptr()),
            }
        }
    }
}

/// A decoder's parameters, as kept in its work buffer.
#[derive(Clone, PartialEq, Eq, Debug)]
struct Parameters {
    rate: u32,
    channels: u32,
    /// Streams, coupled streams and mapping of a multistream decoder.
    streams: Option<(u32, u32, Vec<u8>)>,
}

impl Parameters {
    fn store(&self, memory: &impl DspMemory, work: u32) -> Result<()> {
        let (streams, coupled, mapping) = match &self.streams {
            Some((streams, coupled, mapping)) => (*streams, *coupled, mapping.as_slice()),
            None => (0, 0, &[][..]),
        };
        let kind = u32::from(self.streams.is_some());
        memory.write_words(work + HEADER, &[kind, self.rate, self.channels, streams, coupled])?;
        memory.write(work + HEADER + 20, mapping)
    }

    fn load(memory: &impl DspMemory, work: u32) -> Result<Self> {
        let word = |at| memory.read_word(work + HEADER + at);
        let (kind, rate, channels) = (word(0)?, word(4)?, word(8)?);
        let streams = match kind {
            0 => None,
            _ => {
                let mut mapping = vec![0; channels.min(255) as usize];
                memory.read(work + HEADER + 20, &mut mapping)?;
                Some((word(12)?, word(16)?, mapping))
            }
        };
        Ok(Self { rate, channels, streams })
    }

    fn create(&self) -> std::result::Result<Decoder, i32> {
        let mut error = 0;
        // SAFETY: plain values; the mapping holds `channels` bytes.
        let decoder = unsafe {
            match &self.streams {
                None => NonNull::new(ffi::opus_decoder_create(self.rate as i32, self.channels as c_int, &mut error))
                    .map(Decoder::Single),
                Some((streams, coupled, mapping)) => {
                    if mapping.len() < self.channels as usize {
                        return Err(-1);
                    }
                    NonNull::new(ffi::opus_multistream_decoder_create(
                        self.rate as i32,
                        self.channels as c_int,
                        *streams as c_int,
                        *coupled as c_int,
                        mapping.as_ptr(),
                        &mut error,
                    ))
                    .map(Decoder::Multi)
                }
            }
        };
        match decoder {
            Some(decoder) if error == 0 => Ok(decoder),
            _ => Err(if error == 0 { -7 } else { error }),
        }
    }
}

/// Work buffer bytes of a decoder of `channels` channels, as the image
/// computes it (see [`super`]): libopus's, plus the dispatcher's header.
pub(super) fn work_size(channels: u32) -> u32 {
    decoder_size(channels) + HEADER
}

/// `opus_decoder_get_size` as built into the image (`0x8052A32C`).
fn decoder_size(channels: u32) -> u32 {
    let silk = 0x2168;
    let celt = (0x2030 + 4 * 120) * channels + 16 * 21 + 0x58;
    silk + celt + 0x50
}

/// `opus_multistream_decoder_get_size` as built into the image
/// (`0x80544520`, each decoder 4-byte aligned after 268 bytes of its own),
/// plus the header; 0 for counts the dispatcher refuses (`0x80520648`).
fn multistream_work_size(streams: u32, coupled: u32) -> u32 {
    if !(1..=255).contains(&streams) || coupled > streams {
        return 0;
    }
    let align = |size: u32| (size + 3) & !3;
    (streams - coupled) * align(decoder_size(1)) + coupled * align(decoder_size(2)) + 268 + HEADER
}

impl Decoders {
    /// Serves request `message` (22 to 30) in the shared buffer at
    /// `buffer`.
    pub(super) fn serve(&mut self, memory: &impl DspMemory, buffer: u32, message: u32) -> Result<()> {
        let operand = |index: u32| memory.read_word(buffer + OPERANDS + 4 * index);
        let result = match message {
            22 | 28 => {
                let (work, rate, channels) = (operand(0)?, operand(2)?, operand(3)?);
                let streams = match message {
                    28 => {
                        let mut mapping = vec![0; channels.min(255) as usize];
                        memory.read(buffer + OPERANDS + 24, &mut mapping)?;
                        Some((operand(4)?, operand(5)?, mapping))
                    }
                    _ => None,
                };
                self.initialize(memory, work, Parameters { rate, channels, streams })?
            }
            23 | 29 => self.finalize(memory, operand(0)?)?,
            24 | 30 => {
                let request = Decode {
                    work: operand(0)?,
                    packet: operand(1)?,
                    packet_size: operand(2)?,
                    output: operand(3)?,
                    frame_size: operand(4)?,
                    final_range: operand(5)?,
                    reset: operand(6)? != 0,
                };
                let (result, samples) = self.decode(memory, &request)?;
                memory.write_word(buffer + SAMPLES, samples)?;
                memory.write_word(buffer + TIME, 0)?;
                result
            }
            25 | 26 => 0,
            27 => multistream_work_size(operand(0)?, operand(1)?) as i32,
            _ => return Err(format!("Opus decoder message {message} is not modeled")),
        };
        memory.write_word(buffer + RESULT, result as u32)
    }

    /// `0x80520524` then `0x80520552`: a buffer initialized as itself
    /// stays as it is.
    fn initialize(&mut self, memory: &impl DspMemory, work: u32, parameters: Parameters) -> Result<i32> {
        if memory.read_word(work)? == MAGIC
            && memory.read_word(work + 4)? & 0xFF == 1
            && memory.read_word(work + 8)? == work
        {
            return Ok(0);
        }
        let decoder = match parameters.create() {
            Ok(decoder) => decoder,
            Err(error) => {
                memory.write(work + 4, &[0, 1])?;
                return Ok(error);
            }
        };
        self.0.insert(work, decoder);
        parameters.store(memory, work)?;
        memory.write_words(work, &[MAGIC, 0x0101, work, work + HEADER, 0])?;
        Ok(0)
    }

    /// `0x80520598`.
    fn finalize(&mut self, memory: &impl DspMemory, work: u32) -> Result<i32> {
        if memory.read_word(work)? != MAGIC {
            return Ok(0);
        }
        let flags = memory.read_word(work + 4)?;
        if flags & 0xFF == 1 && memory.read_word(work + 8)? != work {
            return Ok(NOT_READY);
        }
        self.0.remove(&work);
        if flags & 0xFF == 1 {
            memory.write_words(work, &[0; 5])?;
        }
        Ok(0)
    }

    /// `0x805205C0` (`0x8052071E` for multistream): the result and the
    /// samples per channel.
    fn decode(&mut self, memory: &impl DspMemory, request: &Decode) -> Result<(i32, u32)> {
        let work = request.work;
        if memory.read_word(work)? != MAGIC || memory.read_word(work + 8)? != work {
            return Ok((NOT_READY, 0));
        }
        let parameters = Parameters::load(memory, work)?;
        if !self.0.contains_key(&work) {
            match parameters.create() {
                Ok(decoder) => self.0.insert(work, decoder),
                Err(error) => return Ok((error, 0)),
            };
        }
        let decoder = &self.0[&work];
        let mut packet = vec![0; request.packet_size as usize];
        memory.read(request.packet, &mut packet)?;
        let channels = parameters.channels as usize;
        let frame_size = (request.frame_size as usize).min(MOST_SAMPLES);
        let mut pcm = vec![0i16; frame_size * channels];
        let mut range = 0u32;
        // SAFETY: `pcm` holds `frame_size` samples of each channel; the
        // range is written through a valid pointer.
        let decoded = unsafe {
            match decoder {
                Decoder::Single(decoder) => {
                    if request.reset {
                        ffi::opus_decoder_ctl(decoder.as_ptr(), RESET_STATE);
                    }
                    let decoded = ffi::opus_decode(
                        decoder.as_ptr(),
                        packet.as_ptr(),
                        packet.len() as i32,
                        pcm.as_mut_ptr(),
                        frame_size as c_int,
                        0,
                    );
                    ffi::opus_decoder_ctl(decoder.as_ptr(), GET_FINAL_RANGE, &mut range as *mut u32);
                    decoded
                }
                Decoder::Multi(decoder) => {
                    if request.reset {
                        ffi::opus_multistream_decoder_ctl(decoder.as_ptr(), RESET_STATE);
                    }
                    let decoded = ffi::opus_multistream_decode(
                        decoder.as_ptr(),
                        packet.as_ptr(),
                        packet.len() as i32,
                        pcm.as_mut_ptr(),
                        frame_size as c_int,
                        0,
                    );
                    ffi::opus_multistream_decoder_ctl(decoder.as_ptr(), GET_FINAL_RANGE, &mut range as *mut u32);
                    decoded
                }
            }
        };
        if decoded < 0 {
            return Ok((decoded, 0));
        }
        let samples: Vec<u8> = pcm[..decoded as usize * channels].iter().flat_map(|s| s.to_le_bytes()).collect();
        memory.write(request.output, &samples)?;
        memory.write_word(work + 16, range)?;
        let result = match request.final_range {
            0 => 0,
            expected if expected == range => 0,
            _ => RANGE_MISMATCH,
        };
        Ok((result, decoded as u32))
    }
}

/// A decode request's operands.
struct Decode {
    work: u32,
    packet: u32,
    packet_size: u32,
    output: u32,
    frame_size: u32,
    final_range: u32,
    reset: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::soc::adsp::ByteMemory;

    const BUFFER: u32 = 0x1000;
    const WORK: u32 = 0x1_0000;
    const PACKET: u32 = 0x2_0000;
    const OUTPUT: u32 = 0x3_0000;

    /// A 20 ms packet of a 440 Hz tone at 48 kHz, mono, and the encoder's
    /// final range for it.
    fn packet() -> (Vec<u8>, u32) {
        let pcm: Vec<i16> = (0..960)
            .map(|i| ((i as f32 * 440.0 * std::f32::consts::TAU / 48000.0).sin() * 8000.0) as i16)
            .collect();
        let mut data = vec![0u8; 1500];
        let mut range = 0u32;
        // SAFETY: buffers sized as passed; the encoder is destroyed once.
        let size = unsafe {
            let mut error = 0;
            let encoder = ffi::opus_encoder_create(48000, 1, ffi::OPUS_APPLICATION_AUDIO, &mut error);
            assert_eq!(error, 0);
            let size = ffi::opus_encode(encoder, pcm.as_ptr(), 960, data.as_mut_ptr(), data.len() as i32);
            ffi::opus_encoder_ctl(encoder, GET_FINAL_RANGE, &mut range as *mut u32);
            ffi::opus_encoder_destroy(encoder);
            size
        };
        assert!(size > 0);
        data.truncate(size as usize);
        (data, range)
    }

    /// Sends `message` with `operands`: (result, samples).
    fn request(decoders: &mut Decoders, memory: &ByteMemory, message: u32, operands: &[u32]) -> (i32, u32) {
        memory.write_words(BUFFER + OPERANDS, operands).unwrap();
        decoders.serve(memory, BUFFER, message).unwrap();
        (
            memory.read_word(BUFFER + RESULT).unwrap() as i32,
            memory.read_word(BUFFER + SAMPLES).unwrap(),
        )
    }

    /// A packet decodes to its samples, its final range is checked when
    /// the request names one, and a decoder lost with the host is made
    /// again from its work buffer.
    #[test]
    fn decodes_and_checks_the_final_range() {
        let memory = ByteMemory::default();
        let mut decoders = Decoders::default();
        let (data, range) = packet();
        memory.write(PACKET, &data).unwrap();
        let size = work_size(1);
        assert_eq!(request(&mut decoders, &memory, 22, &[WORK, size, 48000, 1]).0, 0);
        assert_eq!(memory.read_word(WORK).unwrap(), MAGIC);

        let decode = |range| [WORK, PACKET, data.len() as u32, OUTPUT, 960, range, 0];
        assert_eq!(request(&mut decoders, &memory, 24, &decode(0)), (0, 960));
        let mut pcm = vec![0; 960 * 2];
        memory.read(OUTPUT, &mut pcm).unwrap();
        assert!(pcm.iter().any(|&b| b != 0), "a tone decodes to sound");
        assert_eq!(memory.read_word(WORK + 16).unwrap(), range);
        assert_eq!(request(&mut decoders, &memory, 24, &decode(range)), (0, 960));
        assert_eq!(request(&mut decoders, &memory, 24, &decode(range ^ 1)).0, RANGE_MISMATCH);

        // A restored machine: the work buffer is all there is.
        let mut restored = Decoders::default();
        assert_eq!(request(&mut restored, &memory, 24, &decode(range)), (0, 960));

        assert_eq!(request(&mut restored, &memory, 23, &[WORK]).0, 0);
        assert_eq!(request(&mut restored, &memory, 24, &decode(0)).0, NOT_READY);
    }

    /// The image refuses stream counts out of 1 to 255 and more coupled
    /// streams than streams.
    #[test]
    fn multistream_work_sizes_refuse_bad_counts() {
        assert_eq!(multistream_work_size(0, 0), 0);
        assert_eq!(multistream_work_size(256, 0), 0);
        assert_eq!(multistream_work_size(2, 3), 0);
        assert_ne!(multistream_work_size(2, 2), 0);
    }
}
