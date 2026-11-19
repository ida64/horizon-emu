//! The miscellaneous APB block (`APB_MISC`, `0x7000_0000`..`0x7000_1000`)
//! as a machine that runs the boot chain has it: the pad and strap
//! registers (`PP_*`), the general registers (`GP_*`) with the chip ID, and
//! the pad control groups. A machine started at kernel entry maps only the
//! chip ID and the pad groups ([`super::Soc::new`]).
//!
//! Power-on values: what the NX bootloader's hardware-state check reads
//! before it writes anything (`nxbl_hwcheck.csv`, the check `0x40002A7C`,
//! these rows at bootloader text `0x40002A8A`..`0x40003480`): every row's
//! expected value under its mask. \[inference\] They are what a console's
//! block holds when the bootloader starts, since the check halts it on any
//! difference; whether the hardware's reset or the boot ROM put them
//! there is not public. The pad groups of SDMMC2 (`0xA98`..`0xAA4`), a port
//! the ROM does not boot from, hold the same values as those of SDMMC4
//! (`0xAB0`..`0xABC`), the eMMC's, which points to reset values. Bits
//! outside the masks are \[assumption\] zero.
//!
//! Registers the check does not read refuse to be read before they are
//! written, except in the pad groups, which read zero like a machine at
//! kernel entry has them (a pure configuration block `boot`'s pin driver
//! reads and rewrites, text `0x2eaac`). Writes are stored: the bootloader
//! writes `PP_PINMUX_GLOBAL` (`0x040` = 0, in its SoC init `0x400162D0`)
//! and the eMMC pad group (`0xAB4`, `0xABC`, `0x40016864` and
//! `0x400168A8`) and reads nothing back but those.

use super::regs::{RegisterFile, word};
use crate::{
    bus::MmioDevice,
    snapshot::{Snapshot, SnapshotError, decode, encode},
};

pub const BASE: u64 = 0x7000_0000;
pub const SIZE: u64 = 0x1000;

/// `APB_MISC_GP_HIDREV`: which chip this is. The firmware's clock driver
/// tells the original console's SoC from its successor by the major
/// revision in bits 7:4, where 2 is the successor (pcv text `0x50cc0`).
/// Here: chip ID 0x21 in bits 15:8 and major revision 1, as publicly
/// documented for this SoC. The minor revision (bits 19:16) and the family
/// (bits 3:0) of a console's chip are not known here and read zero, which
/// is an assumption.
pub const HIDREV: u64 = 0x804;
pub const HIDREV_VALUE: u32 = 0x21 << 8 | 1 << 4;

/// Pad control groups: drive strength and slew rate per group of pins,
/// the part of the block `boot`'s pin driver reads into its tables (text
/// `0x2eaac`) before it programs 47 of the groups. Configuration that
/// starts at zero, like the pin multiplexer, but for the bits the
/// bootloader's check finds set (`PAD_POWER_ON_STATE`).
pub const PAD_GROUPS: std::ops::Range<u64> = 0x8E4..0xB70;

/// The registers outside the pad groups the check reads, with their
/// values (offset from [`BASE`]): `PP_STRAPPING_OPT_A` (`0x008`, bits 9:8
/// and 0), `PP_PINMUX_GLOBAL` (`0x040`), and others the check names only
/// by offset.
const POWER_ON_STATE: [(u64, u32); 24] = [
    (0x008, 0x0000_0000),
    (0x024, 0x0000_0000),
    (0x040, 0x0000_0000),
    (0x07C, 0x0000_0000),
    (0x428, 0x0000_0000),
    (0x810, 0x0000_0000),
    (0x864, 0x0000_0000),
    (0x868, 0x0000_0000),
    (0x880, 0x0000_0000),
    (0x884, 0x0000_0000),
    (0x888, 0x0000_0000),
    (0x88C, 0x0000_0000),
    (0x890, 0x0000_0000),
    (0x8D0, 0xFFFF_0000),
    (0x8D4, 0x0000_0001),
    (0x8D8, 0x0000_0001),
    (0x8DC, 0x0000_FF37),
    (0x8E0, 0x0000_FF37),
    (0xB70, 0x0000_0002),
    (0xB74, 0x0000_000F),
    (0xB78, 0x0000_0001),
    (0xC00, 0x0000_0000),
    (0xC04, 0x0000_0000),
    (0xC08, 0x0000_0000),
];

/// The pad groups the check finds bits set in; the other 145 groups it
/// reads have zero in the bits it looks at.
const PAD_POWER_ON_STATE: [(u64, u32); 10] = [
    (0xA78, 0x0080_8000),
    (0xA90, 0x5000_0000),
    (0xA98, 0x0080_8000),
    (0xA9C, 0x07FF_C310),
    (0xAA0, 0x0015_5555),
    (0xAA4, 0x026A_AAA6),
    (0xAB0, 0x0080_8000),
    (0xAB4, 0x07FF_C310),
    (0xAB8, 0x0015_5555),
    (0xABC, 0x026A_AAA6),
];

pub struct ApbMisc {
    /// Everything outside the pad groups.
    registers: RegisterFile,
    /// The pad groups, by offset from [`BASE`].
    pads: RegisterFile,
}

impl ApbMisc {
    /// The block as the SoC comes out of reset and the boot ROM leaves it.
    pub fn power_on() -> Self {
        Self {
            registers: RegisterFile::strict(
                POWER_ON_STATE.into_iter().chain([(HIDREV, HIDREV_VALUE)]),
            ),
            pads: RegisterFile::zeroed(PAD_POWER_ON_STATE),
        }
    }

    fn file(&self, offset: u64) -> &RegisterFile {
        if PAD_GROUPS.contains(&offset) {
            &self.pads
        } else {
            &self.registers
        }
    }
}

impl MmioDevice for ApbMisc {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        let offset = word(offset, size)?;
        self.file(offset).get(offset).map(u64::from)
    }

    fn write(&self, _core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        let offset = word(offset, size)?;
        // The chip ID is read only.
        (offset != HIDREV).then_some(())?;
        self.file(offset).set(offset, value as u32);
        Some(())
    }

    fn snapshot(&self) -> Option<&dyn Snapshot> {
        Some(self)
    }
}

/// Saves both register files.
impl Snapshot for ApbMisc {
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        encode(&(self.registers.save()?, self.pads.save()?))
    }

    fn restore(&self, state: &[u8]) -> Result<(), SnapshotError> {
        let (registers, pads): (Vec<u8>, Vec<u8>) = decode("APB_MISC", state)?;
        self.registers.restore(&registers)?;
        self.pads.restore(&pads)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn power_on_state_and_storage() {
        let misc = ApbMisc::power_on();
        assert_eq!(misc.read(0, 0x8D0, 4), Some(0xFFFF_0000));
        assert_eq!(misc.read(0, HIDREV, 4), Some(HIDREV_VALUE.into()));
        // A pad group the check reads as zero, and one it does not read.
        assert_eq!(misc.read(0, 0x8E4, 4), Some(0));
        assert_eq!(misc.read(0, 0xA7C, 4), Some(0));
        // Outside the pad groups, unread registers refuse.
        assert_eq!(misc.read(0, 0x000, 4), None);
        assert_eq!(misc.write(0, 0xAB4, 4, 0x07FF_C311), Some(()));
        assert_eq!(misc.read(0, 0xAB4, 4), Some(0x07FF_C311));
        assert_eq!(misc.write(0, HIDREV, 4, 0), None);
    }
}
