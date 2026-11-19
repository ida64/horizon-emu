//! IRAM: the SoC's internal SRAM.
//!
//! 256 KiB at `0x4000_0000`, in four 64 KiB banks (IRAM-A to IRAM-D),
//! reachable from the BPMP and the CCPLEX (Switchbrew "Memory map" and
//! "IRAM"; the T210 TRM's address map). The boot ROM copies the BCT and the
//! bootloader there, and package1ldr and the NX bootloader run from it on
//! the BPMP.
//!
//! Its contents at power-on are not documented; here they start as zeroes
//! \[assumption: the boot ROM writes everything firmware later reads from
//! it, the BIT and BCT included\]. The bank clocks and resets of `CLK_RST`
//! are not consulted.
//!
//! IRAM counts the writes made by every master but the BPMP
//! ([`Iram::code_generation`]): the BPMP core caches the instructions it
//! fetches from IRAM ([`crate::bpmp::Assist::set_code_cache`]) and sees
//! its own stores itself; the BPMP's bus writes through
//! [`Iram::bpmp_write`] and [`Iram::bpmp_write_words`], which do not count.
//!
//! # Unestablished bytes
//!
//! What the boot ROM leaves in IRAM outside what it wrote is not known, nor
//! what IRAM holds after it lost power in SC7. Such bytes are marked
//! unestablished ([`Iram::set_unestablished`], [`Iram::power_off`]) with a
//! note saying why, and the BPMP's reads of them are refused
//! ([`Iram::bpmp_read`]) until a master writes them: the BPMP itself, the
//! CPU cluster over the bus, [`Iram::load`] (the TSEC's and the security
//! engine's DMA, the machine's own loads), any write path here.

use std::{
    ops::Range,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use serde::{Deserialize, Serialize};

use crate::{
    bus::MmioDevice,
    snapshot::{Snapshot, SnapshotError, decode, encode},
};

pub const BASE: u64 = 0x4000_0000;
pub const SIZE: u64 = 0x4_0000;

/// IRAM's bus addresses.
const ADDRESSES: Range<u32> = BASE as u32..(BASE + SIZE) as u32;

pub struct Iram {
    contents: Mutex<Contents>,
    /// Writes by masters other than the BPMP, counted once each is visible.
    generation: Arc<AtomicU64>,
}

/// The bytes and which of them are unestablished, under one lock: a BPMP
/// access checks and reads in one go.
struct Contents {
    bytes: Box<[u8]>,
    unestablished: Unestablished,
}

/// Which bytes hold nothing established, and why.
#[derive(Clone, Serialize, Deserialize)]
struct Unestablished {
    /// A bit per byte, set while it is unestablished.
    bits: Vec<u64>,
    /// How many bits are set: with none, accesses skip the bit work.
    count: usize,
    /// Why, by bus address range; the first range holding a byte says it.
    notes: Vec<(Range<u32>, String)>,
}

/// Why [`Iram::bpmp_read`] gives no value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    /// The access runs past the end of IRAM.
    Outside,
    /// A byte of it is unestablished: its bus address and the note.
    Unestablished { address: u32, note: String },
}

impl Unestablished {
    fn none() -> Self {
        Self {
            bits: vec![0; (SIZE / 64) as usize],
            count: 0,
            notes: Vec::new(),
        }
    }

    /// The bit words and masks `span` (byte offsets) covers.
    fn words(span: Range<usize>) -> impl Iterator<Item = (usize, u64)> {
        let mut at = span.start;
        std::iter::from_fn(move || {
            (at < span.end).then(|| {
                let bit = at % 64;
                let n = (64 - bit).min(span.end - at);
                let mask = if n == 64 {
                    u64::MAX
                } else {
                    ((1 << n) - 1) << bit
                };
                let word = at / 64;
                at += n;
                (word, mask)
            })
        })
    }

    /// Marks the bytes of `span` established.
    fn establish(&mut self, span: Range<usize>) {
        if self.count == 0 {
            return;
        }
        for (word, mask) in Self::words(span) {
            self.count -= (self.bits[word] & mask).count_ones() as usize;
            self.bits[word] &= !mask;
        }
    }

    /// The first unestablished byte of `span`, if there is one.
    fn first(&self, span: Range<usize>) -> Option<Refusal> {
        if self.count == 0 {
            return None;
        }
        let (word, set) = Self::words(span)
            .map(|(word, mask)| (word, self.bits[word] & mask))
            .find(|&(_, set)| set != 0)?;
        let address = ADDRESSES.start + (word * 64) as u32 + set.trailing_zeros();
        let note = self
            .notes
            .iter()
            .find(|(range, _)| range.contains(&address))
            .map_or_else(|| "unestablished".to_string(), |(_, note)| note.clone());
        Some(Refusal::Unestablished { address, note })
    }
}

impl Default for Iram {
    fn default() -> Self {
        Self::new()
    }
}

impl Iram {
    pub fn new() -> Self {
        Self {
            contents: Mutex::new(Contents {
                bytes: vec![0; SIZE as usize].into_boxed_slice(),
                unestablished: Unestablished::none(),
            }),
            generation: Arc::new(AtomicU64::new(0)),
        }
    }

    /// The count of writes by masters other than the BPMP, for the BPMP's
    /// code cache.
    pub fn code_generation(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.generation)
    }

    fn written(&self) {
        self.generation.fetch_add(1, Ordering::Release);
    }

    /// The byte range `len` bytes at `offset` covers, if it lies inside.
    fn span(offset: u64, len: usize) -> Option<Range<usize>> {
        let start = usize::try_from(offset).ok()?;
        let end = start.checked_add(len)?;
        (end <= SIZE as usize).then_some(start..end)
    }

    /// Makes exactly the bytes of `ranges` (bus addresses; outside IRAM is
    /// ignored) unestablished, each with its note, and every other byte
    /// established. The BPMP's code cache is invalidated: a fetch from a
    /// byte made unestablished must be refused again.
    pub fn set_unestablished(&self, ranges: &[(Range<u32>, &str)]) {
        let mut unestablished = Unestablished::none();
        for (range, note) in ranges {
            let start = range.start.max(ADDRESSES.start);
            let end = range.end.min(ADDRESSES.end);
            if start >= end {
                continue;
            }
            let span = (start - ADDRESSES.start) as usize..(end - ADDRESSES.start) as usize;
            for (word, mask) in Unestablished::words(span) {
                unestablished.bits[word] |= mask;
            }
            unestablished.notes.push((start..end, (*note).to_string()));
        }
        unestablished.count = unestablished
            .bits
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum();
        self.contents.lock().unwrap().unestablished = unestablished;
        self.written();
    }

    /// IRAM lost power: every byte reads zero and is unestablished, with
    /// `note`.
    pub fn power_off(&self, note: &str) {
        self.contents.lock().unwrap().bytes.fill(0);
        self.set_unestablished(&[(ADDRESSES, note)]);
    }

    /// The note of the first unestablished byte of `size` at `offset`.
    pub fn unestablished(&self, offset: u32, size: u8) -> Option<Refusal> {
        let span = Self::span(u64::from(offset), usize::from(size))?;
        self.contents.lock().unwrap().unestablished.first(span)
    }

    /// Reads `size` (1, 2 or 4) bytes at `offset`, little-endian.
    pub fn read(&self, offset: u32, size: u8) -> Option<u32> {
        if !matches!(size, 1 | 2 | 4) {
            return None;
        }
        self.read_le(u64::from(offset), size)
            .map(|value| value as u32)
    }

    /// [`Iram::read`] for the BPMP: refused if a byte is unestablished.
    pub fn bpmp_read(&self, offset: u32, size: u8) -> Result<u32, Refusal> {
        let span = Self::span(u64::from(offset), usize::from(size))
            .filter(|_| matches!(size, 1 | 2 | 4))
            .ok_or(Refusal::Outside)?;
        let contents = self.contents.lock().unwrap();
        if let Some(refusal) = contents.unestablished.first(span.clone()) {
            return Err(refusal);
        }
        let mut word = [0; 4];
        word[..usize::from(size)].copy_from_slice(&contents.bytes[span]);
        Ok(u32::from_le_bytes(word))
    }

    /// Writes the low `size` (1, 2 or 4) bytes of `value` at `offset`.
    pub fn write(&self, offset: u32, size: u8, value: u32) -> Option<()> {
        self.bpmp_write(offset, size, value)?;
        self.written();
        Some(())
    }

    /// [`Iram::write`] for the BPMP's own stores, which its code cache
    /// sees without the generation counter. No other master may use it.
    pub fn bpmp_write(&self, offset: u32, size: u8, value: u32) -> Option<()> {
        if !matches!(size, 1 | 2 | 4) {
            return None;
        }
        self.write_le(u64::from(offset), size, u64::from(value))
    }

    /// Reads consecutive little-endian words at `offset` into `words`, all
    /// or none.
    pub fn read_words(&self, offset: u32, words: &mut [u32]) -> Option<()> {
        let span = Self::span(u64::from(offset), 4 * words.len())?;
        let contents = self.contents.lock().unwrap();
        for (word, chunk) in words
            .iter_mut()
            .zip(contents.bytes[span].as_chunks::<4>().0)
        {
            *word = u32::from_le_bytes(*chunk);
        }
        Some(())
    }

    /// [`Iram::read_words`] for the BPMP: refused, reading nothing, if a
    /// byte is unestablished.
    pub fn bpmp_read_words(&self, offset: u32, words: &mut [u32]) -> Result<(), Refusal> {
        let span = Self::span(u64::from(offset), 4 * words.len()).ok_or(Refusal::Outside)?;
        let contents = self.contents.lock().unwrap();
        if let Some(refusal) = contents.unestablished.first(span.clone()) {
            return Err(refusal);
        }
        for (word, chunk) in words
            .iter_mut()
            .zip(contents.bytes[span].as_chunks::<4>().0)
        {
            *word = u32::from_le_bytes(*chunk);
        }
        Ok(())
    }

    /// Writes `words` at `offset`, all or none: the BPMP's block stores,
    /// as [`Iram::bpmp_write`].
    pub fn bpmp_write_words(&self, offset: u32, words: &[u32]) -> Option<()> {
        let span = Self::span(u64::from(offset), 4 * words.len())?;
        let mut contents = self.contents.lock().unwrap();
        for (word, chunk) in words
            .iter()
            .zip(contents.bytes[span.clone()].as_chunks_mut::<4>().0)
        {
            *chunk = word.to_le_bytes();
        }
        contents.unestablished.establish(span);
        Some(())
    }

    /// Copies `data` in at `offset`.
    pub fn load(&self, offset: u32, data: &[u8]) -> Option<()> {
        let span = Self::span(u64::from(offset), data.len())?;
        let mut contents = self.contents.lock().unwrap();
        contents.bytes[span.clone()].copy_from_slice(data);
        contents.unestablished.establish(span);
        drop(contents);
        self.written();
        Some(())
    }

    /// Copies `out.len()` bytes at `offset` out.
    pub fn copy_out(&self, offset: u32, out: &mut [u8]) -> Option<()> {
        let span = Self::span(u64::from(offset), out.len())?;
        out.copy_from_slice(&self.contents.lock().unwrap().bytes[span]);
        Some(())
    }

    fn read_le(&self, offset: u64, size: u8) -> Option<u64> {
        let span = Self::span(offset, usize::from(size))?;
        let contents = self.contents.lock().unwrap();
        let bytes = &contents.bytes[span];
        Some(match size {
            1 => u64::from(bytes[0]),
            2 => u64::from(u16::from_le_bytes([bytes[0], bytes[1]])),
            4 => u64::from(u32::from_le_bytes(bytes.try_into().unwrap())),
            _ => u64::from_le_bytes(bytes.try_into().unwrap()),
        })
    }

    fn write_le(&self, offset: u64, size: u8, value: u64) -> Option<()> {
        let span = Self::span(offset, usize::from(size))?;
        let mut contents = self.contents.lock().unwrap();
        let bytes = &mut contents.bytes[span.clone()];
        match size {
            1 => bytes[0] = value as u8,
            2 => bytes.copy_from_slice(&(value as u16).to_le_bytes()),
            4 => bytes.copy_from_slice(&(value as u32).to_le_bytes()),
            _ => bytes.copy_from_slice(&value.to_le_bytes()),
        }
        contents.unestablished.establish(span);
        Some(())
    }
}

/// SRAM: any width the bus carries, at any alignment inside it.
impl MmioDevice for Iram {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        if !matches!(size, 1 | 2 | 4 | 8) {
            return None;
        }
        self.read_le(offset, size)
    }

    fn write(&self, _core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        if !matches!(size, 1 | 2 | 4 | 8) {
            return None;
        }
        self.write_le(offset, size, value)?;
        self.written();
        Some(())
    }

    fn snapshot(&self) -> Option<&dyn Snapshot> {
        Some(self)
    }
}

/// The contents and which bytes are unestablished.
impl Snapshot for Iram {
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        let contents = self.contents.lock().unwrap();
        encode(&(contents.bytes.to_vec(), &contents.unestablished))
    }

    fn restore(&self, state: &[u8]) -> Result<(), SnapshotError> {
        let (bytes, unestablished): (Vec<u8>, Unestablished) = decode("IRAM", state)?;
        if bytes.len() != SIZE as usize {
            return Err(SnapshotError::Corrupt(format!(
                "IRAM: {} bytes saved, {SIZE} expected",
                bytes.len()
            )));
        }
        let counted: usize = unestablished
            .bits
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum();
        if unestablished.bits.len() != (SIZE / 64) as usize || counted != unestablished.count {
            return Err(SnapshotError::Corrupt("IRAM: unestablished map".into()));
        }
        *self.contents.lock().unwrap() = Contents {
            bytes: bytes.into_boxed_slice(),
            unestablished,
        };
        self.written();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn little_endian_accesses_stay_inside_the_256_kib() {
        let iram = Iram::new();
        iram.write(0x10, 4, 0x4433_2211).unwrap();
        assert_eq!(iram.read(0x10, 1), Some(0x11));
        assert_eq!(iram.read(0x11, 2), Some(0x3322));
        assert_eq!(MmioDevice::read(&iram, 4, 0x10, 8), Some(0x4433_2211));
        assert_eq!(iram.read(0x3_FFFC, 4), Some(0));
        assert_eq!(iram.read(0x3_FFFD, 4), None);
        assert_eq!(iram.write(0x4_0000, 1, 0), None);
        assert_eq!(iram.read(0, 3), None);
        iram.load(0x3_FFFE, &[1, 2]).unwrap();
        assert_eq!(iram.load(0x3_FFFF, &[1, 2]), None);
        let mut out = [0; 2];
        iram.copy_out(0x3_FFFE, &mut out).unwrap();
        assert_eq!(out, [1, 2]);
    }

    #[test]
    fn every_master_writing_a_byte_establishes_it_for_the_bpmp() {
        let iram = Iram::new();
        let generation = iram.code_generation();
        iram.set_unestablished(&[(0x4000_0000..0x4000_0100, "the ROM's work area")]);
        assert_eq!(
            generation.load(Ordering::Acquire),
            1,
            "the code cache drops"
        );
        assert_eq!(
            iram.bpmp_read(0x7E, 4),
            Err(Refusal::Unestablished {
                address: 0x4000_007E,
                note: "the ROM's work area".into()
            })
        );
        assert_eq!(iram.bpmp_read(0x100, 4), Ok(0));
        // The CPU cluster over the bus, a DMA engine's load, the BPMP.
        MmioDevice::write(&iram, 0, 0x40, 8, 0x1122_3344_5566_7788).unwrap();
        iram.load(0x48, &[0xAA; 8]).unwrap();
        iram.bpmp_write(0x50, 2, 0xBEEF).unwrap();
        iram.bpmp_write_words(0x54, &[1, 2, 3]).unwrap();
        assert_eq!(iram.bpmp_read(0x44, 4), Ok(0x1122_3344));
        let mut words = [0; 6];
        assert!(
            iram.bpmp_read_words(0x48, &mut words).is_err(),
            "0x52 unwritten"
        );
        iram.write(0x52, 2, 0).unwrap();
        iram.bpmp_read_words(0x48, &mut words).unwrap();
        assert_eq!(words, [0xAAAA_AAAA, 0xAAAA_AAAA, 0xBEEF, 1, 2, 3]);
        assert!(iram.unestablished(0x60, 1).is_some());
        // Power lost: everything reads zero and is refused again.
        iram.power_off("lost in SC7");
        assert_eq!(iram.read(0x44, 4), Some(0));
        assert!(matches!(
            iram.bpmp_read(0x3_FFFC, 4),
            Err(Refusal::Unestablished { note, .. }) if note == "lost in SC7"
        ));
        assert_eq!(iram.bpmp_read(0x3_FFFE, 4), Err(Refusal::Outside));
    }

    #[test]
    fn snapshot_restores_the_contents_and_the_unestablished_bytes() {
        let iram = Iram::new();
        iram.load(0x100, b"BCT").unwrap();
        iram.set_unestablished(&[(0x4000_0200..0x4000_0204, "x")]);
        let saved = iram.save().unwrap();
        let other = Iram::new();
        other.restore(&saved).unwrap();
        let mut out = [0; 3];
        other.copy_out(0x100, &mut out).unwrap();
        assert_eq!(&out, b"BCT");
        assert!(other.unestablished(0x203, 1).is_some());
        assert!(other.unestablished(0x204, 1).is_none());
        assert!(other.restore(&encode(&vec![0u8; 4]).unwrap()).is_err());
    }
}
