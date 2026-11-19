//! The system counter's control frame (`SYSCTR0`, `0x700F_0000`), as far as
//! a machine that runs the boot chain reaches it: the frequency the NX
//! bootloader records and the registers the secure monitor checks before
//! it starts the counter.
//!
//! * `CNTCR` (`0x000`): zero, the counter not enabled, as the ARM generic
//!   timer architecture has it out of reset; the secure monitor requires
//!   that (TZPROG `0x1F0146C80`) before it enables the counter itself.
//! * `CNTFID0` (`0x020`): the bootloader writes 19 200 000, the
//!   oscillator's rate (`0x40015250`, research §2.1); the monitor requires
//!   that value. Reads before the write refuse: the reset value is not
//!   public.
//! * `0xFD0`..`0xFFC`, the identification registers: \[inference\] zero,
//!   since the monitor requires all twelve to read zero (same check).
//!
//! Starting the counter (`CNTCR` writes) and setting its count (`CNTCV`)
//! are refused: the CPU cluster's counter is the host's, which runs
//! regardless.

use super::regs::RegisterFile;

pub const BASE: u64 = 0x700F_0000;
pub const SIZE: u64 = 0x1000;

const CNTCR: u64 = 0x000;
const CNTFID0: u64 = 0x020;
const ID_REGISTERS: [u64; 12] = [
    0xFD0, 0xFD4, 0xFD8, 0xFDC, 0xFE0, 0xFE4, 0xFE8, 0xFEC, 0xFF0, 0xFF4, 0xFF8, 0xFFC,
];

pub struct Sysctr0 {
    registers: RegisterFile,
}

impl Sysctr0 {
    /// The frame as the SoC comes out of reset.
    pub fn power_on() -> Self {
        Self {
            registers: RegisterFile::strict(
                [(CNTCR, 0)]
                    .into_iter()
                    .chain(ID_REGISTERS.map(|register| (register, 0))),
            ),
        }
    }
}

impl crate::bus::MmioDevice for Sysctr0 {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        let offset = super::regs::word(offset, size)?;
        self.registers.get(offset).map(u64::from)
    }

    fn write(&self, _core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        match super::regs::word(offset, size)? {
            CNTFID0 => self.registers.set(CNTFID0, value as u32),
            _ => return None,
        }
        Some(())
    }

    fn snapshot(&self) -> Option<&dyn crate::snapshot::Snapshot> {
        Some(&self.registers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bus::MmioDevice;

    #[test]
    fn frequency_is_stored_and_the_rest_refused() {
        let counter = Sysctr0::power_on();
        assert_eq!(counter.read(0, CNTCR, 4), Some(0));
        assert_eq!(counter.read(0, 0xFFC, 4), Some(0));
        assert_eq!(counter.read(0, CNTFID0, 4), None);
        assert_eq!(counter.write(0, CNTFID0, 4, 19_200_000), Some(()));
        assert_eq!(counter.read(0, CNTFID0, 4), Some(19_200_000));
        assert_eq!(counter.write(0, CNTCR, 4, 3), None);
    }
}
