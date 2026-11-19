//! The fuse block at `0x7000_F800`: the driver registers, the cache of
//! sensed fuses, and what of the fuse array the boot ROM reads.
//!
//! Register names, offsets and fields as switchbrew's "Fuses" page lists
//! them (the Erista column).
//!
//! # What is modelled
//!
//! * The cache registers (`0x100`..`0x3FC`) and the driver's plain
//!   registers hold a value given at construction or written since; one
//!   that holds none refuses to be read, so no fuse is ever invented.
//!   Writes are kept (the cache is writable by software while
//!   `FUSE_WRITE_ACCESS_SW` allows, which the boot chain leaves so; what a
//!   write to a read-only register does is not public, and none is seen).
//! * `FUSE_PRIVATEKEYDISABLE` (`0x28`): once bit 0 is set,
//!   `FUSE_PRIVATE_KEY0..4` read all ones "until the next reset"
//!   (switchbrew); bits written can only be set \[assumption: the name and
//!   "until the next reset"; no code is seen to clear them\].
//! * Reads of the fuse array through the driver, as the boot ROM does them
//!   for its patches (`0x100590`): the word address in `FUSE_FUSEADDR`,
//!   command 1 (`READ`) in `FUSE_FUSECTRL` bits 1:0, the word in
//!   `FUSE_FUSERDATA`. The command is done as it is written:
//!   `FUSE_FUSECTRL`'s state field (bits 20:16) always reads idle (4), what
//!   the ROM and package1ldr wait for, and its command field idle (0)
//!   \[inference: the ROM leaves `READ` there after its last word and
//!   never writes the register again, and the NX bootloader's
//!   hardware-state check requires bits 1:0 clear\]. Only array words given at
//!   construction can be read; any other address, and the commands that
//!   program or sense fuses, are refused.
//! * [`Fuse::power_off`]: the block as constructed again, for a SoC that
//!   lost power in SC7 (the private key fuses readable again, the driver
//!   registers at their reset values).
//!
//! # The patch records
//!
//! [`patch_records`] is the Switch's boot ROM patches as fuse array words,
//! for a machine that runs the ROM. A console's own array is not public;
//! what is public is its decoded content (switchbrew "Fuses",
//! `irom_patch`: the twelve patch unit entries of an unpatched ("old")
//! Erista and the disassembly of the handler code the ROM copies to IRAM)
//! and the record format (hexkyz's "ipatch decoder", which the ROM's own
//! decoder at `0x100590`, `0x1003C0` and `0x1006E0` matches). The records
//! here encode that content:
//!
//! * \[assumption\] as one record of 78 words. A console's fuses hold the
//!   same entries and code in some split of records burnt over time; the
//!   ROM appends each record's entries and code in order, so the split
//!   changes nothing it does after decoding.
//! * \[inference\] with the handler code assembled from the published
//!   disassembly. Every instruction and literal there has one encoding but
//!   the five constants the listing shows as single `MOVS Rd, #imm32`
//!   (`0xA0000000`, `0xE0000000`, `0x70000000`, `0x4000`, `0xF000000`), each
//!   a `MOVS Rd, #imm8` and an `LSLS` in ARMv4T Thumb: they are encoded
//!   with an odd `imm8`, as every such pair in the ROM itself is. Another
//!   split of a constant computes the same value and flags.

use std::{collections::HashMap, sync::Mutex};

use serde::{Deserialize, Serialize};

use super::regs::word;
use crate::{
    bus::MmioDevice,
    snapshot::{Snapshot, SnapshotError, decode, encode},
};

pub const BASE: u64 = 0x7000_F800;
pub const SIZE: u64 = 0x400;

const FUSECTRL: u64 = 0x00;
const FUSEADDR: u64 = 0x04;
const FUSERDATA: u64 = 0x08;
pub const PRIVATEKEYDISABLE: u64 = 0x28;
/// `FUSE_FIRST_BOOTROM_PATCH_SIZE`: the first patch record's words (7 bits).
pub const FIRST_BOOTROM_PATCH_SIZE: u64 = 0x19C;
pub const PRIVATE_KEY0: u64 = 0x1A4;
const PRIVATE_KEY4: u64 = 0x1B4;

const CTRL_CMD: u32 = 0b11;
const CMD_IDLE: u32 = 0;
const CMD_READ: u32 = 1;
const CTRL_STATE: u32 = 0x1F << 16;
const STATE_IDLE: u32 = 4 << 16;

/// Where the ROM starts reading patch records, going down (the last word
/// of an array of 6 kbit; the decoder's `FUSE_PATCH_START_ADDRESS`).
const PATCH_START_ADDRESS: u32 = 191;

/// The patch unit entries of an unpatched Erista (switchbrew, "ipatches
/// (old)"): bits 31:16 the halfword's index in the ROM, bits 15:0 its
/// replacement.
const PATCH_ENTRIES: [u32; 12] = [
    0x0B57_DF00,
    0x1820_DF22,
    0x3797_DF26,
    0x7D9E_2000,
    0x042B_DF2C,
    0x37AA_DF42,
    0x0972_DF4B,
    0x2293_DF54,
    0x21FA_DF5D,
    0xBBA2_AC57,
    0xBBAC_3D19,
    0x1E95_2001,
];

/// The handler code the ROM appends to its dispatcher in IRAM
/// (switchbrew's listing from `0x48` to `0x14C`, assembled; see the
/// module): the handlers of `svc` 0x00, 0x22, 0x26, 0x2C, 0x42, 0x4B, 0x54
/// and 0x5D, their literals, and the common return.
const PATCH_CODE: [u32; 65] = [
    0x43D2_2200,
    0x630A_4930,
    0x4930_638A,
    0x604A_600A,
    0x600A_492F,
    0x3180_618A,
    0x600A_311C,
    0x610A_608A,
    0x600A_3180,
    0x492B_604A,
    0x2205_600A,
    0x492A_0752,
    0x3138_600A,
    0x2207_600A,
    0x4928_0752,
    0x2100_600A,
    0xE05D_200E,
    0x2201_6988,
    0xE059_4310,
    0x1C92_6D22,
    0x3980_6522,
    0xE053_6361,
    0x0700_2007,
    0x4A21_4E20,
    0x2A00_6812,
    0x6832_D002,
    0xE003_6082,
    0x0C80_6882,
    0x6032_4302,
    0x6830_4E1C,
    0x0392_2201,
    0x6032_4302,
    0xE03D_6928,
    0x0612_220F,
    0x6121_4391,
    0x2207_6D21,
    0x6521_4391,
    0x4A15_E034,
    0x0A12_6812,
    0x6821_0052,
    0x6021_4391,
    0xE02B_2800,
    0x6802_4811,
    0x4811_9200,
    0xD025_0C52,
    0x6004_4810,
    0x4A10_E7FE,
    0x6992_9200,
    0x61C2_30FC,
    0x0000_E01C,
    0x6000_6410,
    0x6000_60F8,
    0x6000_6284,
    0x6000_6554,
    0x6000_6148,
    0x6000_66A0,
    0x7000_EF14,
    0x7000_E5B4,
    0x7000_E9C0,
    0x7000_FA9C,
    0x4000_49F0,
    0x4001_0000,
    0x7000_E400,
    0x4001_0220,
    0x46F7_BC04,
];

/// A patch record's header: the count of patch unit entries in bits
/// 19:16, a Hamming code over the record in bits 14:0 with its parity in
/// bit 15, and in bits 31:20 the next record's size with a code of its
/// own (zero here: no next record).
const HEADER_ENTRY_COUNT_SHIFT: u32 = 16;
const HEADER_PARITY: u32 = 1 << 15;
/// Bit `j` of word `i` of a record counts as this plus `32 * i + j` in the
/// record's Hamming syndrome (the decoder's `H16_START_OFFSET`).
const SYNDROME_START: u32 = 1 << 14;

/// The XOR of the syndrome positions of every set bit of `record`.
fn syndrome(record: &[u32]) -> u32 {
    let mut syndrome = 0;
    for (i, &word) in record.iter().enumerate() {
        for j in (0..32).filter(|j| word >> j & 1 != 0) {
            syndrome ^= SYNDROME_START + 32 * i as u32 + j;
        }
    }
    syndrome
}

/// The Switch's patch record (see the module): its header, the patch unit
/// entries, the handler code.
fn patch_record() -> Vec<u32> {
    let mut record = vec![(PATCH_ENTRIES.len() as u32) << HEADER_ENTRY_COUNT_SHIFT];
    record.extend(PATCH_ENTRIES);
    record.extend(PATCH_CODE);
    // The decoder computes the syndrome with only the entry count left in
    // the header, and the parity over the record as stored.
    record[0] |= syndrome(&record);
    if record
        .iter()
        .fold(0, |parity, word| parity ^ word)
        .count_ones()
        % 2
        != 0
    {
        record[0] |= HEADER_PARITY;
    }
    record
}

/// What a machine that runs the boot ROM needs of the fuses for the ROM's
/// patches: the fuse array words of the Switch's record, by address, and
/// the value of `FUSE_FIRST_BOOTROM_PATCH_SIZE` that goes with them.
pub fn patch_records() -> (Vec<(u32, u32)>, u32) {
    let record = patch_record();
    let size = record.len() as u32;
    let words = record
        .into_iter()
        .enumerate()
        .map(|(i, word)| (PATCH_START_ADDRESS - i as u32, word))
        .collect();
    (words, size)
}

#[derive(Clone, Default, Serialize, Deserialize)]
struct State {
    registers: HashMap<u64, u32>,
}

/// The fuse block.
pub struct Fuse {
    state: Mutex<State>,
    /// The registers as constructed, for [`Fuse::power_off`].
    initial: HashMap<u64, u32>,
    /// The fuse array words that can be read, by word address.
    array: HashMap<u32, u32>,
}

impl Fuse {
    /// A block whose registers hold `registers` (offset and value) and
    /// whose array holds `array` (word address and value).
    pub fn new(
        registers: impl IntoIterator<Item = (u64, u32)>,
        array: impl IntoIterator<Item = (u32, u32)>,
    ) -> Self {
        let initial: HashMap<u64, u32> = registers.into_iter().collect();
        Self {
            state: Mutex::new(State {
                registers: initial.clone(),
            }),
            initial,
            array: array.into_iter().collect(),
        }
    }

    /// The block as constructed again: what a reset of the SoC leaves.
    pub fn power_off(&self) {
        self.state.lock().unwrap().registers = self.initial.clone();
    }
}

impl MmioDevice for Fuse {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        let offset = word(offset, size)?;
        let state = self.state.lock().unwrap();
        let hidden = state
            .registers
            .get(&PRIVATEKEYDISABLE)
            .is_some_and(|disable| disable & 1 != 0);
        let value = match offset {
            PRIVATE_KEY0..=PRIVATE_KEY4 if hidden => u32::MAX,
            FUSECTRL => state.registers.get(&offset).copied()? & !CTRL_STATE | STATE_IDLE,
            _ => state.registers.get(&offset).copied()?,
        };
        Some(u64::from(value))
    }

    fn write(&self, _core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        let offset = word(offset, size)?;
        let value = value as u32;
        let mut state = self.state.lock().unwrap();
        match offset {
            FUSECTRL => {
                match value & CTRL_CMD {
                    CMD_IDLE => {}
                    CMD_READ => {
                        let address = state.registers.get(&FUSEADDR).copied()?;
                        let data = self.array.get(&address).copied()?;
                        state.registers.insert(FUSERDATA, data);
                    }
                    _ => return None,
                }
                // Done: the command field is back at idle.
                state.registers.insert(offset, value & !CTRL_CMD);
            }
            PRIVATEKEYDISABLE => {
                *state.registers.get_mut(&offset)? |= value;
            }
            _ => {
                state.registers.insert(offset, value);
            }
        }
        Some(())
    }

    fn snapshot(&self) -> Option<&dyn Snapshot> {
        Some(self)
    }
}

/// The registers that hold a value; the array is the machine's
/// construction.
impl Snapshot for Fuse {
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        encode(&*self.state.lock().unwrap())
    }

    fn restore(&self, state: &[u8]) -> Result<(), SnapshotError> {
        *self.state.lock().unwrap() = decode("fuses", state)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CORE: usize = 0;

    fn block() -> Fuse {
        Fuse::new(
            [
                (FUSECTRL, 0),
                (PRIVATEKEYDISABLE, 0),
                (PRIVATE_KEY0, 0x1111_1111),
                (PRIVATE_KEY4, 0x5555_5555),
                (0x110, 0x83),
            ],
            [(191, 0xAABB_CCDD)],
        )
    }

    #[test]
    fn registers_hold_what_they_were_given_and_refuse_the_rest() {
        let fuse = block();
        assert_eq!(fuse.read(CORE, 0x110, 4), Some(0x83));
        assert_eq!(fuse.read(CORE, 0x114, 4), None);
        assert_eq!(fuse.read(CORE, 0x110, 2), None);
        fuse.write(CORE, 0x114, 4, 7).unwrap();
        assert_eq!(fuse.read(CORE, 0x114, 4), Some(7));
        assert_eq!(fuse.read(CORE, FUSECTRL, 4), Some(u64::from(STATE_IDLE)));
    }

    #[test]
    fn the_private_key_reads_all_ones_once_disabled_until_power_is_lost() {
        let fuse = block();
        assert_eq!(fuse.read(CORE, PRIVATE_KEY0, 4), Some(0x1111_1111));
        fuse.write(CORE, PRIVATEKEYDISABLE, 4, 1).unwrap();
        assert_eq!(fuse.read(CORE, PRIVATE_KEY0, 4), Some(0xFFFF_FFFF));
        assert_eq!(fuse.read(CORE, PRIVATE_KEY4, 4), Some(0xFFFF_FFFF));
        assert_eq!(fuse.read(CORE, 0x110, 4), Some(0x83));
        // The bit cannot be taken back.
        fuse.write(CORE, PRIVATEKEYDISABLE, 4, 0x10).unwrap();
        assert_eq!(fuse.read(CORE, PRIVATEKEYDISABLE, 4), Some(0x11));
        fuse.power_off();
        assert_eq!(fuse.read(CORE, PRIVATEKEYDISABLE, 4), Some(0));
        assert_eq!(fuse.read(CORE, PRIVATE_KEY0, 4), Some(0x1111_1111));
    }

    #[test]
    fn the_array_is_read_through_the_driver() {
        let fuse = block();
        fuse.write(CORE, FUSEADDR, 4, 191).unwrap();
        fuse.write(CORE, FUSECTRL, 4, u64::from(CMD_READ)).unwrap();
        assert_eq!(fuse.read(CORE, FUSERDATA, 4), Some(0xAABB_CCDD));
        assert_eq!(
            fuse.read(CORE, FUSECTRL, 4),
            Some(u64::from(STATE_IDLE | CMD_IDLE))
        );
        // A word nothing establishes, and programming.
        fuse.write(CORE, FUSEADDR, 4, 190).unwrap();
        assert_eq!(fuse.write(CORE, FUSECTRL, 4, u64::from(CMD_READ)), None);
        assert_eq!(fuse.write(CORE, FUSECTRL, 4, 2), None);
        assert_eq!(fuse.write(CORE, FUSECTRL, 4, 3), None);
    }

    /// The public decoder's check of a record (hexkyz's "ipatch decoder",
    /// `hamming16_decode`): no error means the parity over the stored
    /// words is even and the syndrome of the record, with only the entry
    /// count left in its header, is the stored one.
    #[test]
    fn the_patch_record_passes_the_decoder_and_holds_the_entries_and_code() {
        let (words, size) = patch_records();
        assert_eq!(size, 78);
        assert_eq!(words.len(), 78);
        assert_eq!(words[0].0, 191);
        assert_eq!(words[77].0, 114);
        let mut record: Vec<u32> = words.iter().map(|&(_, word)| word).collect();
        let parity = record.iter().fold(0, |parity, word| parity ^ word);
        assert_eq!(parity.count_ones() % 2, 0);
        let header = record[0];
        assert_eq!(header >> 20, 0, "no next record");
        record[0] &= 0x000F_0000;
        assert_eq!(syndrome(&record), header & 0x7FFF);
        assert_eq!(record[0] >> HEADER_ENTRY_COUNT_SHIFT, 12);
        assert_eq!(record[1..13], PATCH_ENTRIES);
        assert_eq!(record[13..], PATCH_CODE);
        // The entries name the halfwords switchbrew lists, e.g. the first
        // at 0x1016AE and the key patches in the ROM's data.
        assert_eq!((PATCH_ENTRIES[0] >> 16) * 2, 0x16AE);
        assert_eq!((PATCH_ENTRIES[9] >> 16) * 2, 0x1_7744);
    }

    #[test]
    fn snapshot_round_trip() {
        let fuse = block();
        fuse.write(CORE, PRIVATEKEYDISABLE, 4, 1).unwrap();
        fuse.write(CORE, 0x200, 4, 9).unwrap();
        let other = block();
        other.restore(&fuse.save().unwrap()).unwrap();
        assert_eq!(other.read(CORE, PRIVATE_KEY0, 4), Some(0xFFFF_FFFF));
        assert_eq!(other.read(CORE, 0x200, 4), Some(9));
    }
}
