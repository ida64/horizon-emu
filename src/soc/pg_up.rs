//! The processor tag at `0x6000_0000` (`PG_UP_TAG`): a read tells the
//! reading processor which one it is. U-Boot's Tegra `ap.h` names the
//! register and both values: `PG_UP_TAG_0_PID_CPU` 0x55555555 for the CPU
//! cluster, `PG_UP_TAG_0_PID_COP` 0xAAAAAAAA for the BPMP (the "COP", AVP).
//! The warmboot firmware requires the BPMP's value before it goes on
//! (`0x4001056C`, `warmboot-research` §7 step 3). The register has no state;
//! the rest of the page (the arbitration semaphores in U-Boot's map) is not
//! modelled and refuses.

use super::regs::word;
use crate::{boot::CORE_COUNT, bus::MmioDevice};

pub const BASE: u64 = 0x6000_0000;
pub const SIZE: u64 = 0x1000;

const TAG: u64 = 0x000;
const TAG_CPU: u32 = 0x5555_5555;
const TAG_COP: u32 = 0xAAAA_AAAA;

pub struct PgUp;

impl MmioDevice for PgUp {
    fn read(&self, core: usize, offset: u64, size: u8) -> Option<u64> {
        match word(offset, size)? {
            TAG if core < CORE_COUNT => Some(TAG_CPU.into()),
            TAG => Some(TAG_COP.into()),
            _ => None,
        }
    }

    fn write(&self, _core: usize, _offset: u64, _size: u8, _value: u64) -> Option<()> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_processor_reads_its_own_tag() {
        assert_eq!(PgUp.read(0, TAG, 4), Some(0x5555_5555));
        assert_eq!(PgUp.read(3, TAG, 4), Some(0x5555_5555));
        assert_eq!(PgUp.read(CORE_COUNT, TAG, 4), Some(0xAAAA_AAAA));
        assert_eq!(PgUp.read(CORE_COUNT, 4, 4), None);
        assert_eq!(PgUp.write(CORE_COUNT, TAG, 4, 0), None);
    }
}
