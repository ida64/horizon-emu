//! The BPMP's cache and memory protection controller, `0x1000` bytes at
//! `0x5004_0000`.
//!
//! No cache is emulated: the BPMP reads and writes memory directly, so
//! there is never a line to clean or invalidate, and nothing here changes
//! what the BPMP sees. What its software does with the block still has to
//! come out as it expects. Register names, offsets and bits as hekate's
//! `bpmp.c` has them (reference for facts); Atmosphère's
//! `tegra_avp_cache.hpp` agrees on the ones it names.
//!
//! * `CONFIG` (`0x00`): storage. \[assumption\] 0 (cache disabled) until
//!   written: its power-on value is not public, and the boot chain's first
//!   use is the secure monitor's SC7 stub, which sets the buffer-disable
//!   bits 10 and 11 by read-modify-write (text `0x40003464`).
//! * Maintenance: a write of `MAINT_REQ` (`0x28`, the operation in its
//!   low bits, the ways in bits 11:8) is done at once and sets
//!   `MAINT_DONE` (bit 0) in `INT_RAW_EVENT` (`0x48`), which hekate polls
//!   after each request ("This is a blocking operation"). `INT_CLEAR`
//!   (`0x44`) clears the raw event bits written as ones; `INT_STATUS`
//!   (`0x4C`) is the raw events under `INT_MASK` (`0x40`). The block's
//!   interrupt is not wired to the interrupt controller: `CONFIG` bit 7,
//!   which would enable it, is refused.
//! * The memory protection unit: 32 shadow entries from `0x400` and 32
//!   main entries from `0x800`, 16 bytes each (start, end, attributes, and
//!   a fourth word hekate's structure has and never writes). `MMU_CMD`
//!   (`0xB0`) 2 copies the shadow entries whose bit is set in
//!   `MMU_SHADOW_COPY_MASK` (`0xA4`) to the main ones; 1 initialises the
//!   unit, taken here to clear both sets and the mask \[assumption: hekate
//!   issues it first and then writes everything it uses\]. The entries
//!   protect nothing here: no access of the BPMP is checked against them.
//!   So a fallback entry (`MMU_FALLBACK_ENTRY`, `0xA0`) that denies
//!   reading, writing or executing outside the entries is refused, as it
//!   would make accesses fault that go through here; hekate's allows all
//!   three.
//! * Every other register keeps what is written and refuses to be read
//!   until then: their power-on values are not public.

use super::regs::{RegisterFile, word};
use crate::{bus::MmioDevice, snapshot::Snapshot};

pub const BASE: u64 = 0x5004_0000;
pub const SIZE: u64 = 0x1000;

const CONFIG: u64 = 0x00;
const CONFIG_ENABLE_INTERRUPT: u32 = 1 << 7;
const MAINT_REQ: u64 = 0x28;
const INT_MASK: u64 = 0x40;
const INT_CLEAR: u64 = 0x44;
const INT_RAW_EVENT: u64 = 0x48;
const INT_STATUS: u64 = 0x4C;
const INT_MAINT_DONE: u32 = 1 << 0;
const MMU_FALLBACK_ENTRY: u64 = 0xA0;
/// Read, write and execute allowed (bits 2, 3 and 1).
const MMU_ALLOW_ALL: u32 = 0b1110;
const MMU_SHADOW_COPY_MASK: u64 = 0xA4;
const MMU_CMD: u64 = 0xB0;
const MMU_CMD_NOP: u32 = 0;
const MMU_CMD_INIT: u32 = 1;
const MMU_CMD_COPY_SHADOW: u32 = 2;
const SHADOW_ENTRIES: u64 = 0x400;
const MAIN_ENTRIES: u64 = 0x800;
const ENTRY_COUNT: u64 = 32;
const ENTRY_SIZE: u64 = 0x10;

pub struct BpmpCache {
    registers: RegisterFile,
}

impl BpmpCache {
    pub fn new() -> Self {
        Self {
            registers: RegisterFile::strict([(CONFIG, 0), (INT_MASK, 0), (INT_RAW_EVENT, 0)]),
        }
    }

    fn entry_words(base: u64, entry: u64) -> impl Iterator<Item = u64> {
        (0..ENTRY_SIZE / 4).map(move |word| base + entry * ENTRY_SIZE + 4 * word)
    }
}

impl Default for BpmpCache {
    fn default() -> Self {
        Self::new()
    }
}

impl MmioDevice for BpmpCache {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        let value = match word(offset, size)? {
            INT_STATUS => self.registers.get(INT_RAW_EVENT)? & self.registers.get(INT_MASK)?,
            register => self.registers.get(register)?,
        };
        Some(u64::from(value))
    }

    fn write(&self, _core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        let register = word(offset, size)?;
        let value = value as u32;
        match register {
            CONFIG if value & CONFIG_ENABLE_INTERRUPT != 0 => return None,
            MAINT_REQ => {
                self.registers.set(register, value);
                self.registers
                    .update(INT_RAW_EVENT, |raw| raw | INT_MAINT_DONE)?;
            }
            INT_CLEAR => self.registers.update(INT_RAW_EVENT, |raw| raw & !value)?,
            INT_RAW_EVENT | INT_STATUS => return None,
            MMU_FALLBACK_ENTRY if value & MMU_ALLOW_ALL != MMU_ALLOW_ALL => return None,
            MMU_CMD => match value {
                MMU_CMD_NOP => {}
                MMU_CMD_INIT => {
                    for entry in 0..ENTRY_COUNT {
                        for base in [SHADOW_ENTRIES, MAIN_ENTRIES] {
                            for word in Self::entry_words(base, entry) {
                                self.registers.set(word, 0);
                            }
                        }
                    }
                    self.registers.set(MMU_SHADOW_COPY_MASK, 0);
                }
                MMU_CMD_COPY_SHADOW => {
                    let mask = self.registers.get(MMU_SHADOW_COPY_MASK)?;
                    for entry in (0..ENTRY_COUNT).filter(|entry| mask >> entry & 1 != 0) {
                        let words = Self::entry_words(SHADOW_ENTRIES, entry)
                            .zip(Self::entry_words(MAIN_ENTRIES, entry));
                        for (shadow, main) in words {
                            self.registers.set(main, self.registers.get(shadow)?);
                        }
                    }
                }
                _ => return None,
            },
            _ => self.registers.set(register, value),
        }
        Some(())
    }

    fn snapshot(&self) -> Option<&dyn Snapshot> {
        Some(&self.registers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maintenance_is_done_at_once_and_the_shadow_entries_copy_by_mask() {
        let cache = BpmpCache::new();
        assert_eq!(cache.read(4, INT_RAW_EVENT, 4), Some(0));
        cache.write(4, MAINT_REQ, 4, 0xF << 8 | 18).unwrap();
        assert_eq!(cache.read(4, INT_RAW_EVENT, 4), Some(1));
        assert_eq!(cache.read(4, INT_STATUS, 4), Some(0), "masked");
        cache.write(4, INT_CLEAR, 4, 1).unwrap();
        assert_eq!(cache.read(4, INT_RAW_EVENT, 4), Some(0));

        cache.write(4, MMU_CMD, 4, MMU_CMD_INIT.into()).unwrap();
        for (entry, start) in [(0, 0x8000_0000u64), (1, 0x4000_0000)] {
            cache
                .write(4, SHADOW_ENTRIES + entry * ENTRY_SIZE, 4, start)
                .unwrap();
        }
        cache.write(4, MMU_SHADOW_COPY_MASK, 4, 0b10).unwrap();
        cache
            .write(4, MMU_CMD, 4, MMU_CMD_COPY_SHADOW.into())
            .unwrap();
        assert_eq!(cache.read(4, MAIN_ENTRIES, 4), Some(0), "not in the mask");
        assert_eq!(
            cache.read(4, MAIN_ENTRIES + ENTRY_SIZE, 4),
            Some(0x4000_0000)
        );
        // A unit that would deny accesses is not modelled.
        assert_eq!(cache.write(4, MMU_FALLBACK_ENTRY, 4, 0b0100), None);
        assert_eq!(cache.write(4, MMU_FALLBACK_ENTRY, 4, 0b1110), Some(()));
    }
}
