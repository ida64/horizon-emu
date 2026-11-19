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

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};

use crate::{
    bus::MmioDevice,
    snapshot::{Snapshot, SnapshotError, decode, encode},
};

pub const BASE: u64 = 0x4000_0000;
pub const SIZE: u64 = 0x4_0000;

pub struct Iram {
    bytes: Mutex<Box<[u8]>>,
    /// Writes by masters other than the BPMP, counted once each is visible.
    generation: Arc<AtomicU64>,
}

impl Default for Iram {
    fn default() -> Self {
        Self::new()
    }
}

impl Iram {
    pub fn new() -> Self {
        Self {
            bytes: Mutex::new(vec![0; SIZE as usize].into_boxed_slice()),
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
    fn span(offset: u64, len: usize) -> Option<std::ops::Range<usize>> {
        let start = usize::try_from(offset).ok()?;
        let end = start.checked_add(len)?;
        (end <= SIZE as usize).then_some(start..end)
    }

    /// Reads `size` (1, 2 or 4) bytes at `offset`, little-endian.
    pub fn read(&self, offset: u32, size: u8) -> Option<u32> {
        if !matches!(size, 1 | 2 | 4) {
            return None;
        }
        self.read_le(u64::from(offset), size)
            .map(|value| value as u32)
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
        let bytes = self.bytes.lock().unwrap();
        for (word, chunk) in words.iter_mut().zip(bytes[span].as_chunks::<4>().0) {
            *word = u32::from_le_bytes(*chunk);
        }
        Some(())
    }

    /// Writes `words` at `offset`, all or none: the BPMP's block stores,
    /// as [`Iram::bpmp_write`].
    pub fn bpmp_write_words(&self, offset: u32, words: &[u32]) -> Option<()> {
        let span = Self::span(u64::from(offset), 4 * words.len())?;
        let mut bytes = self.bytes.lock().unwrap();
        for (word, chunk) in words.iter().zip(bytes[span].as_chunks_mut::<4>().0) {
            *chunk = word.to_le_bytes();
        }
        Some(())
    }

    /// Copies `data` in at `offset`.
    pub fn load(&self, offset: u32, data: &[u8]) -> Option<()> {
        let span = Self::span(u64::from(offset), data.len())?;
        self.bytes.lock().unwrap()[span].copy_from_slice(data);
        self.written();
        Some(())
    }

    /// Copies `out.len()` bytes at `offset` out.
    pub fn copy_out(&self, offset: u32, out: &mut [u8]) -> Option<()> {
        let span = Self::span(u64::from(offset), out.len())?;
        out.copy_from_slice(&self.bytes.lock().unwrap()[span]);
        Some(())
    }

    fn read_le(&self, offset: u64, size: u8) -> Option<u64> {
        let span = Self::span(offset, usize::from(size))?;
        let bytes = self.bytes.lock().unwrap();
        let bytes = &bytes[span];
        Some(match size {
            1 => u64::from(bytes[0]),
            2 => u64::from(u16::from_le_bytes([bytes[0], bytes[1]])),
            4 => u64::from(u32::from_le_bytes(bytes.try_into().unwrap())),
            _ => u64::from_le_bytes(bytes.try_into().unwrap()),
        })
    }

    fn write_le(&self, offset: u64, size: u8, value: u64) -> Option<()> {
        let span = Self::span(offset, usize::from(size))?;
        let mut bytes = self.bytes.lock().unwrap();
        let bytes = &mut bytes[span];
        match size {
            1 => bytes[0] = value as u8,
            2 => bytes.copy_from_slice(&(value as u16).to_le_bytes()),
            4 => bytes.copy_from_slice(&(value as u32).to_le_bytes()),
            _ => bytes.copy_from_slice(&value.to_le_bytes()),
        }
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

impl Snapshot for Iram {
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        encode(&self.bytes.lock().unwrap().to_vec())
    }

    fn restore(&self, state: &[u8]) -> Result<(), SnapshotError> {
        let bytes: Vec<u8> = decode("IRAM", state)?;
        if bytes.len() != SIZE as usize {
            return Err(SnapshotError::Corrupt(format!(
                "IRAM: {} bytes saved, {SIZE} expected",
                bytes.len()
            )));
        }
        *self.bytes.lock().unwrap() = bytes.into_boxed_slice();
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
    fn snapshot_restores_the_contents() {
        let iram = Iram::new();
        iram.load(0x100, b"BCT").unwrap();
        let saved = iram.save().unwrap();
        let other = Iram::new();
        other.restore(&saved).unwrap();
        let mut out = [0; 3];
        other.copy_out(0x100, &mut out).unwrap();
        assert_eq!(&out, b"BCT");
        assert!(other.restore(&encode(&vec![0u8; 4]).unwrap()).is_err());
    }
}
