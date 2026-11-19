//! AHB arbitration and the AHB "gizmos" (`0x6000_C000`..`0x6000_C200`):
//! the AHB masters' priorities, the bridges' settings and the memory
//! prefetchers, as far as a machine that runs the boot chain reaches
//! them. The secure boot registers follow at `0x6000_C200` ([`super::sb`]).
//!
//! Pure configuration here: nothing in the emulator arbitrates a bus or
//! prefetches. Power-on values: what the NX bootloader's hardware-state
//! check reads (`nxbl_hwcheck.csv`, the check `0x40002A7C`, these rows at
//! bootloader text `0x4000C6D8`..`0x4000C878`), each row's expected value
//! under its mask. \[inference\] What a console's block holds when the
//! bootloader starts, the check halting it on any difference; whether the
//! hardware's reset or the boot ROM put them there is not public. Bits
//! outside the masks are \[assumption\] zero. Registers the check does not
//! read refuse to be read before they are written.
//!
//! Writes are stored. The bootloader writes one register:
//! `AHB_ARBITRATION_XBAR_CTRL` (`0x0E0`) bit 16 from its SDRAM parameters
//! (read-modify-write in `0x40015348`, research §2.5). The secure monitor
//! (high-level emulated) would write `0x004`, `0x008`, `0x018`, `0x038`
//! and `0x054`.

use super::regs::RegisterFile;

pub const BASE: u64 = 0x6000_C000;
pub const SIZE: u64 = 0x200;

/// Offsets as the check reads them: `0x004` `AHB_ARBITRATION_DISABLE`,
/// `0x008` `_PRIORITY_CTRL`, `0x0CC`..`0x0F4` the memory prefetchers'
/// configuration, `0x0E0` `XBAR_CTRL` (names as hekate `t210.h` has them,
/// reference only).
const POWER_ON_STATE: [(u64, u32); 33] = [
    (0x004, 0x0000_0000),
    (0x008, 0xE000_0000),
    (0x00C, 0x0000_0000),
    (0x010, 0x0002_0385),
    (0x014, 0x0002_0000),
    (0x018, 0x0000_0000),
    (0x020, 0x0002_0087),
    (0x024, 0x0000_008D),
    (0x028, 0x0006_0000),
    (0x02C, 0x0006_0000),
    (0x030, 0x0000_0008),
    (0x038, 0x0000_0000),
    (0x050, 0x0002_0000),
    (0x054, 0x0000_0081),
    (0x07C, 0x0002_0087),
    (0x090, 0x0002_0000),
    (0x098, 0x0006_0000),
    (0x0C0, 0x0000_0000),
    (0x0CC, 0x1480_0800),
    (0x0D0, 0x1880_0800),
    (0x0D4, 0x1C80_0800),
    (0x0D8, 0x2080_0800),
    (0x0DC, 0x0000_0000),
    (0x0E0, 0x0000_0000),
    (0x0E4, 0x0C80_0800),
    (0x0E8, 0x1080_0800),
    (0x0F0, 0x0480_0800),
    (0x0F4, 0x0880_0800),
    (0x110, 0x0000_0000),
    (0x114, 0x0000_0000),
    (0x120, 0x0000_0000),
    (0x124, 0x0000_0032),
    (0x128, 0x0000_0000),
];

/// The block as the SoC comes out of reset and the boot ROM leaves it.
pub fn power_on() -> RegisterFile {
    RegisterFile::strict(POWER_ON_STATE)
}
