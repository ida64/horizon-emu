//! Storage for blocks of 32-bit configuration registers.

use std::{collections::HashMap, sync::Mutex};

use crate::{
    bus::MmioDevice,
    snapshot::{Snapshot, SnapshotError, decode, encode},
};

/// Offset of the register a device access addresses, for blocks that only
/// support whole, aligned 32-bit accesses.
pub fn word(offset: u64, size: u8) -> Option<u64> {
    (size == 4 && offset.is_multiple_of(4)).then_some(offset)
}

/// A block of 32-bit registers that hold what was last written to them.
///
/// What a register reads as before the guest writes it is the block's
/// choice:
///
/// * [`RegisterFile::strict`] refuses such reads unless the register was
///   given an initial value, so nothing is ever invented.
/// * [`RegisterFile::zeroed`] reads as zero. That is for pure configuration
///   blocks the guest programs by read-modify-write, where the only effect
///   of the initial contents is on bits the guest never looks at again. It
///   is an assumption about those bits, not a model of what the boot chain
///   left in them.
pub struct RegisterFile {
    registers: Mutex<HashMap<u64, u32>>,
    unwritten: Option<u32>,
}

impl RegisterFile {
    pub fn strict(initial: impl IntoIterator<Item = (u64, u32)>) -> Self {
        Self {
            registers: Mutex::new(initial.into_iter().collect()),
            unwritten: None,
        }
    }

    pub fn zeroed(initial: impl IntoIterator<Item = (u64, u32)>) -> Self {
        Self {
            registers: Mutex::new(initial.into_iter().collect()),
            unwritten: Some(0),
        }
    }

    pub fn get(&self, offset: u64) -> Option<u32> {
        self.registers
            .lock()
            .unwrap()
            .get(&offset)
            .copied()
            .or(self.unwritten)
    }

    pub fn set(&self, offset: u64, value: u32) {
        self.registers.lock().unwrap().insert(offset, value);
    }

    /// Replaces the register's value with `update(value)`; `None` if it has
    /// no value to update.
    pub fn update(&self, offset: u64, update: impl FnOnce(u32) -> u32) -> Option<()> {
        let mut registers = self.registers.lock().unwrap();
        let value = registers.get(&offset).copied().or(self.unwritten)?;
        registers.insert(offset, update(value));
        Some(())
    }
}

impl MmioDevice for RegisterFile {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        self.get(word(offset, size)?).map(u64::from)
    }

    fn write(&self, _core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        self.set(word(offset, size)?, value as u32);
        Some(())
    }

    fn snapshot(&self) -> Option<&dyn Snapshot> {
        Some(self)
    }
}

/// Saves every register that holds a value, written or initial. What an
/// unwritten register reads as is the block's construction, not state.
impl Snapshot for RegisterFile {
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        encode(&*self.registers.lock().unwrap())
    }

    fn restore(&self, state: &[u8]) -> Result<(), SnapshotError> {
        *self.registers.lock().unwrap() = decode("register file", state)?;
        Ok(())
    }
}

/// Saves a block kept as a plain array of registers; see [`restore_words`].
pub fn save_words<const N: usize>(words: &Mutex<[u32; N]>) -> Result<Vec<u8>, SnapshotError> {
    encode(&&words.lock().unwrap()[..])
}

/// Takes back what [`save_words`] saved of a block of the same size;
/// `what` names the block for the error.
pub fn restore_words<const N: usize>(
    what: &str,
    words: &Mutex<[u32; N]>,
    state: &[u8],
) -> Result<(), SnapshotError> {
    let saved: Vec<u32> = decode(what, state)?;
    let len = saved.len();
    *words.lock().unwrap() = saved
        .try_into()
        .map_err(|_| SnapshotError::Corrupt(format!("{what}: {len} registers saved, {N} here")))?;
    Ok(())
}
