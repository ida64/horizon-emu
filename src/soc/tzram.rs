//! TZRAM: the SoC's TrustZone SRAM, 64 KiB at `0x7C01_0000`.
//!
//! The secure monitor lives there: its boot stage decompresses its resident
//! image to `0x7C01_0800` and runs from it, and before SC7 it encrypts
//! `0x7C01_2000..0x7C02_0000` into DRAM for the warmboot firmware to
//! decrypt back (TZRAM loses power in SC7 on the original console,
//! `warmboot-research` §4). The CPU cluster's non-secure world cannot reach
//! it ([`super::apb_security`]); the security engine and the BPMP can.
//!
//! What the high-level emulated monitor keeps here is documented in
//! [`crate::secmon::cold_boot`]. Contents at power-on are not documented;
//! here they start as zeroes \[assumption: everything read is written
//! first\].

use std::sync::Mutex;

use crate::{
    bus::MmioDevice,
    snapshot::{Snapshot, SnapshotError, decode, encode},
};

pub const BASE: u64 = 0x7C01_0000;
pub const SIZE: u64 = 0x1_0000;

pub struct Tzram {
    bytes: Mutex<Box<[u8]>>,
}

impl Default for Tzram {
    fn default() -> Self {
        Self::new()
    }
}

impl Tzram {
    pub fn new() -> Self {
        Self {
            bytes: Mutex::new(vec![0; SIZE as usize].into_boxed_slice()),
        }
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
        if !matches!(size, 1 | 2 | 4) {
            return None;
        }
        self.write_le(u64::from(offset), size, u64::from(value))
    }

    /// Copies `data` in at `offset`.
    pub fn load(&self, offset: u32, data: &[u8]) -> Option<()> {
        let span = Self::span(u64::from(offset), data.len())?;
        self.bytes.lock().unwrap()[span].copy_from_slice(data);
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
        let mut value = [0u8; 8];
        value[..usize::from(size)].copy_from_slice(&bytes[span]);
        Some(u64::from_le_bytes(value))
    }

    fn write_le(&self, offset: u64, size: u8, value: u64) -> Option<()> {
        let span = Self::span(offset, usize::from(size))?;
        self.bytes.lock().unwrap()[span].copy_from_slice(&value.to_le_bytes()[..usize::from(size)]);
        Some(())
    }
}

/// SRAM: any width the bus carries, at any alignment inside it.
impl MmioDevice for Tzram {
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
        self.write_le(offset, size, value)
    }

    fn snapshot(&self) -> Option<&dyn Snapshot> {
        Some(self)
    }
}

impl Snapshot for Tzram {
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        encode(&self.bytes.lock().unwrap().to_vec())
    }

    fn restore(&self, state: &[u8]) -> Result<(), SnapshotError> {
        let bytes: Vec<u8> = decode("TZRAM", state)?;
        if bytes.len() != SIZE as usize {
            return Err(SnapshotError::Corrupt(format!(
                "TZRAM: {} bytes saved, {SIZE} expected",
                bytes.len()
            )));
        }
        *self.bytes.lock().unwrap() = bytes.into_boxed_slice();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn holds_64_kib_and_round_trips_through_a_snapshot() {
        let tzram = Tzram::new();
        tzram.write(0xFFFC, 4, 0x4433_2211).unwrap();
        assert_eq!(tzram.read(0xFFFC, 4), Some(0x4433_2211));
        assert_eq!(tzram.read(0xFFFD, 4), None);
        tzram.load(0x2000, &[1, 2, 3]).unwrap();
        let mut out = [0; 3];
        tzram.copy_out(0x2000, &mut out).unwrap();
        assert_eq!(out, [1, 2, 3]);
        assert_eq!(MmioDevice::read(&tzram, 0, 0x2000, 8), Some(0x03_0201));

        let saved = tzram.save().unwrap();
        let restored = Tzram::new();
        restored.restore(&saved).unwrap();
        assert_eq!(restored.read(0xFFFC, 4), Some(0x4433_2211));
        assert!(restored.restore(&encode(&vec![0u8; 4]).unwrap()).is_err());
    }
}
