//! Exception vector registers (EVP): where the processors take their
//! exceptions.
//!
//! The BPMP (ARM7TDMI) has no vector table in memory: each of its
//! exceptions fetches from the address held in its `EVP_COP_*_VECTOR`
//! register, in the order of the architected vectors (hekate
//! `bdk/soc/t210.h`: reset `0x200`, undefined `0x204`, SWI `0x208`,
//! prefetch abort `0x20c`, data abort `0x210`, reserved `0x214`, IRQ
//! `0x218`, FIQ `0x21c`, i.e. `0x200` plus the architected vector
//! offset). `EVP_CPU_RESET_VECTOR` (`0x100`) is where a CPU-cluster core
//! starting in AArch32 begins; the cores of this SoC start in AArch64 at
//! the secure boot block's reset vector instead ([`super::sb`]) once that
//! one selects AArch64, and the public bootloaders write zero here
//! (hekate `ccplex_boot_cpu0`).
//!
//! What the registers hold before they are written is what the boot ROM
//! left, and no public source fixes it (the ROM is not available here), so
//! an unwritten register refuses to be read; package1ldr writes all eight
//! BPMP vectors before anything else. Other offsets of the block (the
//! CPU's other AArch32 vectors, `EVP_COP_IRQ_STS` at `0x220`) are refused.

use std::sync::Mutex;

use super::regs::word;
use crate::{
    bus::MmioDevice,
    snapshot::{Snapshot, SnapshotError, decode, encode},
};

pub const BASE: u64 = 0x6000_F000;
pub const SIZE: u64 = 0x1000;

pub const CPU_RESET_VECTOR: u64 = 0x100;
/// `EVP_COP_RESET_VECTOR`; the other seven follow, a word each.
pub const COP_VECTORS: u64 = 0x200;

#[derive(Default)]
pub struct Evp {
    /// `EVP_CPU_RESET_VECTOR`, then the BPMP's eight vectors.
    vectors: Mutex<[Option<u32>; 9]>,
}

impl Evp {
    pub fn new() -> Self {
        Self::default()
    }

    /// The register at `offset`, as an index into `vectors`.
    fn index(offset: u64) -> Option<usize> {
        match offset {
            CPU_RESET_VECTOR => Some(0),
            COP_VECTORS..=0x21C => Some(1 + ((offset - COP_VECTORS) / 4) as usize),
            _ => None,
        }
    }

    /// The address the BPMP's exception with architected vector offset
    /// `vector` (0 reset, 4 undefined ... 0x1c FIQ) is taken at; `None`
    /// while the register is unwritten.
    pub fn cop_vector(&self, vector: u32) -> Option<u32> {
        if vector > 0x1C || !vector.is_multiple_of(4) {
            return None;
        }
        self.vectors.lock().unwrap()[1 + (vector / 4) as usize]
    }

    /// `EVP_CPU_RESET_VECTOR`, `None` while unwritten.
    pub fn cpu_reset_vector(&self) -> Option<u32> {
        self.vectors.lock().unwrap()[0]
    }
}

impl MmioDevice for Evp {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        let index = Self::index(word(offset, size)?)?;
        self.vectors.lock().unwrap()[index].map(u64::from)
    }

    fn write(&self, _core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        let index = Self::index(word(offset, size)?)?;
        self.vectors.lock().unwrap()[index] = Some(value as u32);
        Some(())
    }

    fn snapshot(&self) -> Option<&dyn Snapshot> {
        Some(self)
    }
}

impl Snapshot for Evp {
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        encode(&*self.vectors.lock().unwrap())
    }

    fn restore(&self, state: &[u8]) -> Result<(), SnapshotError> {
        *self.vectors.lock().unwrap() = decode("exception vectors", state)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vectors_refuse_until_written() {
        let evp = Evp::new();
        assert_eq!(evp.read(0, 0x218, 4), None);
        assert_eq!(evp.cop_vector(0x18), None);
        evp.write(4, 0x218, 4, 0x4001_0123).unwrap();
        assert_eq!(evp.read(0, 0x218, 4), Some(0x4001_0123));
        assert_eq!(evp.cop_vector(0x18), Some(0x4001_0123));
        assert_eq!(evp.cop_vector(0x1C), None);
        evp.write(4, CPU_RESET_VECTOR, 4, 0).unwrap();
        assert_eq!(evp.cpu_reset_vector(), Some(0));
        assert_eq!(evp.cop_vector(0x20), None);
    }

    #[test]
    fn refuses_other_registers() {
        let evp = Evp::new();
        assert_eq!(evp.write(0, 0x220, 4, 0), None);
        assert_eq!(evp.write(0, 0x104, 4, 0), None);
        assert_eq!(evp.write(0, 0x200, 2, 0), None);
    }

    #[test]
    fn snapshot_round_trip() {
        let evp = Evp::new();
        evp.write(0, 0x204, 4, 7).unwrap();
        let other = Evp::new();
        other.restore(&evp.save().unwrap()).unwrap();
        assert_eq!(other.cop_vector(4), Some(7));
        assert_eq!(other.cop_vector(0), None);
    }
}
