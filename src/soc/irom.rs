//! IROM: the boot ROM's memory at `0x0010_0000`, and the patch unit
//! (IPATCH) at `0x6001_DC00` that overrides words of it.
//!
//! The ROM itself is NVIDIA's code and is not part of this program: a
//! machine has an IROM only when it is given a dump ([`RomImage`]). Without
//! one the boot ROM is high-level emulated ([`crate::bootrom`]) and the
//! BPMP's accesses here are refused.
//!
//! # The patch unit
//!
//! The ROM corrects itself from fuses: early after reset (ROM
//! `0x100590`) it reads patch records from the fuse array, writes up to
//! twelve entries into the patch unit and enables them (`0x100492`, the
//! unit's reset at `0x1006AC`). As the ROM programs it, and as the public
//! decoder of the fuse records has it (hexkyz's "ipatch decoder",
//! switchbrew "Fuses", `irom_patch`):
//!
//! | offset | register | here |
//! |---|---|---|
//! | `0x00` | enables, one bit per entry | storage; bits 11:0 |
//! | `0x04` + 4·n | entry n: bits 31:16 the halfword's index in the ROM (its offset / 2), bits 15:0 the halfword read there instead | storage |
//!
//! An enabled entry replaces the halfword at its offset for every read of
//! the ROM by the BPMP, fetches and data alike (switchbrew: "overriding
//! data returned for code and data fetches done by BPMP"; two of the
//! Switch's twelve entries patch a key in the ROM's data). \[assumption\]
//! When two enabled entries name the same halfword the lower-numbered one
//! wins; the ROM never programs such a pair (it updates an entry that
//! names the same halfword instead, `0x100492`). \[assumption\] Both
//! registers read back what was written and reset to zero: the ROM reads
//! the enables back to find the next free entry. Other offsets are refused.
//!
//! Most patches are `svc` instructions: the ROM copies handler code from
//! the same fuse records into IRAM and points the BPMP's software
//! interrupt vector at it. A dump read on a console already shows that
//! console's patches in place, since the reads went through its patch
//! unit; one read on another Tegra X1 device shows that device's. The
//! Switch's entries, once the ROM has programmed them, cover whatever the
//! dump has at the same halfwords. Anywhere else the dump has the other
//! device's patch and not the ROM's instruction. Where that instruction
//! has been worked out for a dump known by its digest, it is put back
//! when the dump is loaded ([`RomImage::restored`]). An `svc` left
//! elsewhere is another device's patch, whose handler the Switch's fuses
//! do not hold and whose original instruction the dump does not:
//! [`Irom::foreign_svc`] finds it, and the BPMP's machine refuses to run
//! it.
//!
//! # The lock-out
//!
//! `SB_CSR` bit 4 (`PIROM_DISABLE`) hides the ROM from `SB_PIROM_START`
//! on ([`super::sb`]): the ROM sets both before it starts the bootloader
//! (Fusée Gelée write-up: the ROM "protects" itself; the NX bootloader
//! requires `SB_PIROM_START` 0x1000 with the bit set). What a read of the
//! hidden part gives is not public: it is refused. The part below
//! `SB_PIROM_START` stays readable; the ROM's last instructions run there.

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{regs::word, sb::Sb};
use crate::{
    bus::MmioDevice,
    snapshot::{Snapshot, SnapshotError, decode, encode},
};

/// Where the BPMP sees the ROM, and its first instruction (the reset
/// vector, an ARM branch).
pub const BASE: u32 = 0x0010_0000;
/// The T210's ROM is 96 KiB (the ROM's own bound on patch offsets,
/// `0x100492`: `2 * index < 0x18000`).
pub const SIZE: usize = 0x1_8000;

pub const IPATCH_BASE: u64 = 0x6001_DC00;
pub const IPATCH_SIZE: u64 = 0x100;
const ENABLES: u64 = 0x00;
const ENTRIES: u64 = 0x04;
const ENTRY_COUNT: usize = 12;

/// The identification string every T210 boot ROM carries ("NV Boot T210
/// WXYZ.HIJK", as the public ipatch decoder quotes it).
const IDENTIFICATION: &[u8] = b"NV Boot T210";

/// A halfword of a dump that shows another device's patch, and the
/// instruction the ROM has there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Restored {
    /// Offset of the halfword in the ROM.
    pub offset: u32,
    /// What the dump has: the other device's patch.
    pub dumped: u16,
    /// What the ROM has.
    pub original: u16,
}

/// Dumps known to show another device's patches at halfwords the Switch's
/// own patches do not cover, by SHA-256, with what is known of the
/// instructions underneath.
///
/// The one entry is the dump this was developed against. At `0x286` it
/// has `b 0x100290` (`0xE003`) as the first instruction of the function
/// both mode predicates call (`0x10033A`, `0x100358`), which then always
/// answers 1: the device it was read on is thereby never in "ODM
/// production" mode and its ROM checks no signature. The seven halfwords
/// that follow are the function's untouched body: `ldr r0, [r0, #0x2c]`,
/// bit 1 of that, as 0 or 1. \[inference\] The instruction replaced is
/// `ldr r0, =0x7000FAC0` (`0x483B`): the body needs a register block's
/// address in `r0`; the function just before (`0x100280`) is `ldr r0,
/// =0x7000FAC0` (`0x483C`, the literal at `0x100374`) and the same `ldr
/// r0, [r0, #0x2c]`, which is `FUSE_SECURE_PROVISION_INFO`; `0x483B` is
/// that literal from two bytes further on; and of the literals in reach
/// it is the only one whose `+0x2c` is a register a mode predicate could
/// ask (the others give `FUSE_PUBLIC_KEY2` and a clock register).
///
/// The two `svc` patches of that device which the Switch's do not cover
/// (`0x1D4A`, `0x30C6`) are not restored, the instructions under them
/// being unknown: [`Irom::foreign_svc`] has them refused.
const KNOWN_DUMPS: &[([u8; 32], &[Restored])] = &[(
    [
        0x34, 0x68, 0xF3, 0x44, 0xF0, 0xDD, 0x47, 0xE1, 0xA5, 0xCE, 0x56, 0x09, 0xEC, 0xA8, 0x47,
        0x1D, 0x9D, 0x75, 0x13, 0xF3, 0xC0, 0x04, 0x1E, 0xAD, 0xAC, 0xE1, 0xAE, 0xA2, 0x08, 0xF1,
        0x79, 0x23,
    ],
    &[Restored {
        offset: 0x286,
        dumped: 0xE003,
        original: 0x483B,
    }],
)];

/// A dump of the T210 boot ROM.
#[derive(Clone)]
pub struct RomImage {
    bytes: Arc<[u8]>,
    digest: [u8; 32],
    restored: &'static [Restored],
}

impl RomImage {
    /// Takes `bytes` as the ROM: [`SIZE`] bytes that start with the ARM
    /// reset branch and carry NVIDIA's identification of the T210 ROM.
    ///
    /// A dump listed in `KNOWN_DUMPS` has the halfwords listed there put
    /// back ([`Self::restored`]).
    pub fn new(mut bytes: Vec<u8>) -> Result<Self, String> {
        if bytes.len() != SIZE {
            return Err(format!(
                "{} bytes; the T210 boot ROM is {SIZE:#x}",
                bytes.len()
            ));
        }
        // An ARM `b` with the always condition: what the reset vector of
        // an ARM7TDMI image is.
        if bytes[3] != 0xEA {
            return Err("its first word is not an ARM branch (the reset vector)".into());
        }
        if !bytes
            .windows(IDENTIFICATION.len())
            .any(|window| window == IDENTIFICATION)
        {
            return Err("it does not identify itself as the T210 boot ROM".into());
        }
        let digest: [u8; 32] = Sha256::digest(&bytes).into();
        let restored = KNOWN_DUMPS
            .iter()
            .find(|(known, _)| *known == digest)
            .map_or(&[][..], |(_, restored)| restored);
        for halfword in restored {
            let at = halfword.offset as usize;
            debug_assert_eq!(bytes[at..at + 2], halfword.dumped.to_le_bytes());
            bytes[at..at + 2].copy_from_slice(&halfword.original.to_le_bytes());
        }
        Ok(Self {
            bytes: bytes.into(),
            digest,
            restored,
        })
    }

    /// The SHA-256 of the dump as it was given, to tell dumps apart in
    /// reports.
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }

    /// The halfwords of the dump that were put back: it was read through
    /// another device's patch unit and showed that device's patches there.
    pub fn restored(&self) -> &'static [Restored] {
        self.restored
    }

    /// The ROM: the dump with [`Self::restored`] applied.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// The patch unit's registers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
struct Patches {
    enables: u32,
    entries: [u32; ENTRY_COUNT],
}

impl Patches {
    /// The halfword an enabled entry puts at halfword `index` of the ROM.
    fn at(&self, index: u32) -> Option<u16> {
        (0..ENTRY_COUNT)
            .filter(|&entry| self.enables >> entry & 1 != 0)
            .map(|entry| self.entries[entry])
            .find(|&entry| entry >> 16 == index)
            .map(|entry| entry as u16)
    }
}

/// Why a read of the ROM was not served.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refusal {
    /// Past the end of the ROM, or not aligned to its size.
    Outside,
    /// In the part the ROM's lock-out hides.
    Hidden,
}

/// The ROM as the BPMP reads it: the dump, the patch unit's entries over
/// it, and the lock-out.
pub struct Irom {
    image: RomImage,
    patches: Mutex<Patches>,
    sb: Arc<Sb>,
    /// Counts what changes the ROM's contents as read: writes to the patch
    /// unit here, the lock-out registers in [`Sb`] (which holds the same
    /// counter). A cache of instructions fetched from the ROM is valid
    /// while it stands.
    generation: Arc<AtomicU64>,
}

impl Irom {
    pub fn new(image: RomImage, sb: Arc<Sb>) -> Self {
        let generation = sb.irom_generation();
        Self {
            image,
            patches: Mutex::default(),
            sb,
            generation,
        }
    }

    pub fn image(&self) -> &RomImage {
        &self.image
    }

    /// See the field: for [`crate::bpmp::Assist::add_code_cache`].
    pub fn code_generation(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.generation)
    }

    /// The patch unit loses its entries with the SoC's power (SC7); the
    /// ROM programs it again.
    pub fn power_off(&self) {
        *self.patches.lock().unwrap() = Patches::default();
        self.generation.fetch_add(1, Ordering::AcqRel);
    }

    /// Reads `size` bytes (1, 2 or 4, aligned) at `offset` of the ROM.
    pub fn read(&self, offset: u32, size: u8) -> Result<u32, Refusal> {
        let (start, len) = (offset as usize, usize::from(size));
        if !matches!(size, 1 | 2 | 4) || !start.is_multiple_of(len) || start + len > SIZE {
            return Err(Refusal::Outside);
        }
        if self
            .sb
            .irom_hidden_from()
            .is_some_and(|hidden| u64::from(offset) + u64::from(size) > u64::from(hidden))
        {
            return Err(Refusal::Hidden);
        }
        let mut bytes = [0; 4];
        bytes[..len].copy_from_slice(&self.image.bytes[start..start + len]);
        let patches = *self.patches.lock().unwrap();
        if patches.enables != 0 {
            let first = start & !1;
            for halfword in (first..start + len).step_by(2) {
                let Some(patch) = patches.at((halfword / 2) as u32) else {
                    continue;
                };
                for (at, byte) in (halfword..).zip(patch.to_le_bytes()) {
                    if (start..start + len).contains(&at) {
                        bytes[at - start] = byte;
                    }
                }
            }
        }
        Ok(u32::from_le_bytes(bytes))
    }

    /// Whether the Thumb instruction at `offset`, an `svc` the BPMP has
    /// just executed, is one the dump holds that no enabled entry of the
    /// patch unit put there: another device's patch (see the module).
    pub fn foreign_svc(&self, offset: u32) -> bool {
        let patches = *self.patches.lock().unwrap();
        (offset as usize) < SIZE && offset.is_multiple_of(2) && patches.at(offset / 2).is_none()
    }
}

/// The patch unit's registers on the bus.
pub struct Ipatch(pub Arc<Irom>);

impl Ipatch {
    fn entry(offset: u64) -> Option<usize> {
        let entry = usize::try_from(offset.checked_sub(ENTRIES)? / 4).ok()?;
        (entry < ENTRY_COUNT).then_some(entry)
    }
}

impl MmioDevice for Ipatch {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        let offset = word(offset, size)?;
        let patches = *self.0.patches.lock().unwrap();
        let value = match offset {
            ENABLES => patches.enables,
            _ => patches.entries[Self::entry(offset)?],
        };
        Some(u64::from(value))
    }

    fn write(&self, _core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        let offset = word(offset, size)?;
        let value = value as u32;
        let mut patches = self.0.patches.lock().unwrap();
        match offset {
            ENABLES if value >> ENTRY_COUNT == 0 => patches.enables = value,
            ENABLES => return None,
            _ => patches.entries[Self::entry(offset)?] = value,
        }
        self.0.generation.fetch_add(1, Ordering::AcqRel);
        Some(())
    }

    fn snapshot(&self) -> Option<&dyn Snapshot> {
        Some(self)
    }
}

/// The patch unit's registers; the dump is the machine's construction.
impl Snapshot for Ipatch {
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        encode(&*self.0.patches.lock().unwrap())
    }

    fn restore(&self, state: &[u8]) -> Result<(), SnapshotError> {
        *self.0.patches.lock().unwrap() = decode("IPATCH", state)?;
        self.0.generation.fetch_add(1, Ordering::AcqRel);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::soc::sb;

    const BPMP: usize = crate::boot::CORE_COUNT;

    /// A stand-in of the right size and shape, with a marker at 0x2000.
    fn image() -> RomImage {
        let mut bytes = vec![0; SIZE];
        bytes[..4].copy_from_slice(&0xEA00_000Eu32.to_le_bytes());
        bytes[0x40..0x40 + IDENTIFICATION.len()].copy_from_slice(IDENTIFICATION);
        bytes[0x2000..0x2008].copy_from_slice(&[0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]);
        RomImage::new(bytes).unwrap()
    }

    fn irom() -> (Arc<Irom>, Arc<Sb>) {
        let sb = Arc::new(Sb::new());
        (Arc::new(Irom::new(image(), Arc::clone(&sb))), sb)
    }

    #[test]
    fn a_dump_is_the_right_size_starts_with_a_branch_and_names_itself() {
        assert!(RomImage::new(vec![0; SIZE - 1]).is_err());
        assert!(RomImage::new(vec![0; SIZE]).is_err());
        let mut unnamed = vec![0; SIZE];
        unnamed[3] = 0xEA;
        assert!(RomImage::new(unnamed).is_err());
        assert_eq!(image().bytes().len(), SIZE);
    }

    #[test]
    fn enabled_entries_replace_their_halfword_for_every_width() {
        let (irom, _) = irom();
        let unit = Ipatch(Arc::clone(&irom));
        let generation = irom.code_generation();
        let before = generation.load(Ordering::Acquire);
        assert_eq!(irom.read(0x2000, 4), Ok(0x4433_2211));
        // Entry 3 names the halfword at 0x2002; not enabled yet.
        unit.write(BPMP, ENTRIES + 12, 4, 0x1001_BEEF).unwrap();
        assert_eq!(irom.read(0x2000, 4), Ok(0x4433_2211));
        unit.write(BPMP, ENABLES, 4, 1 << 3).unwrap();
        assert!(generation.load(Ordering::Acquire) > before);
        assert_eq!(irom.read(0x2000, 4), Ok(0xBEEF_2211));
        assert_eq!(irom.read(0x2002, 2), Ok(0xBEEF));
        assert_eq!(irom.read(0x2003, 1), Ok(0xBE));
        assert_eq!(irom.read(0x2004, 4), Ok(0x8877_6655));
        assert_eq!(unit.read(BPMP, ENABLES, 4), Some(8));
        assert_eq!(unit.read(BPMP, ENTRIES + 12, 4), Some(0x1001_BEEF));
        irom.power_off();
        assert_eq!(irom.read(0x2000, 4), Ok(0x4433_2211));
    }

    #[test]
    fn an_svc_no_entry_put_there_is_foreign() {
        let (irom, _) = irom();
        let unit = Ipatch(Arc::clone(&irom));
        assert!(irom.foreign_svc(0x2002));
        unit.write(BPMP, ENTRIES, 4, 0x1001_DF22).unwrap();
        unit.write(BPMP, ENABLES, 4, 1).unwrap();
        assert!(!irom.foreign_svc(0x2002));
        assert!(irom.foreign_svc(0x2004));
    }

    #[test]
    fn the_lock_out_hides_the_rom_from_pirom_start() {
        let (irom, sb) = irom();
        assert_eq!(irom.read(0x2000, 1), Ok(0x11));
        sb.write(BPMP, 0x00, 4, u64::from(sb::CSR_PIROM_DISABLE))
            .unwrap();
        assert_eq!(irom.read(0x2000, 1), Err(Refusal::Hidden));
        assert_eq!(irom.read(0xFFC, 4), Ok(0));
        assert_eq!(irom.read(0xFFE, 4), Err(Refusal::Outside));
        assert_eq!(irom.read(0x1000, 4), Err(Refusal::Hidden));
        assert_eq!(irom.read(SIZE as u32, 1), Err(Refusal::Outside));
    }

    #[test]
    fn refuses_other_registers_and_enables() {
        let (irom, _) = irom();
        let unit = Ipatch(irom);
        assert_eq!(unit.write(BPMP, ENABLES, 4, 1 << 12), None);
        assert_eq!(unit.write(BPMP, ENTRIES + 4 * 12, 4, 0), None);
        assert_eq!(unit.read(BPMP, ENTRIES, 2), None);
    }

    #[test]
    fn snapshot_round_trip() {
        let (irom, _) = irom();
        let unit = Ipatch(Arc::clone(&irom));
        unit.write(BPMP, ENTRIES, 4, 0x1001_DF22).unwrap();
        unit.write(BPMP, ENABLES, 4, 1).unwrap();
        let (other, _) = self::irom();
        Ipatch(Arc::clone(&other))
            .restore(&unit.save().unwrap())
            .unwrap();
        assert_eq!(other.read(0x2002, 2), Ok(0xDF22));
    }
}
