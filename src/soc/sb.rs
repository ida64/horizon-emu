//! Secure boot registers (SB): the CPU cluster's AArch64 reset vector and
//! the lock-outs that protect it and the boot ROM.
//!
//! Offsets from `0x6000_C200` and fields as in hekate `bdk/soc/t210.h`
//! (`Secure boot registers`); the other names as Atmosphère's register
//! headers have them (reference only):
//!
//! | offset | register | here |
//! |---|---|---|
//! | `0x00` | `SB_CSR` | bit 1 `NS_RST_VEC_WR_DIS`, bit 4 `PIROM_DISABLE`; sticky |
//! | `0x04` | `SB_PIROM_START` | read only here |
//! | `0x08` | `SB_PFCFG` | storage: the debug-authentication enables (hekate `SB_PFCFG`) the warmboot firmware sets on a production unit (`0x40010500`, `warmboot-research` §7 step 2); nothing here acts on them |
//! | `0x0c`..`0x28` | `SB_SECURE_SPAREREG_0..7` | read only here |
//! | `0x30` | `SB_AA64_RESET_LOW` | reset vector bits 31:2; bit 0 `AARCH64_MODE_EN` |
//! | `0x34` | `SB_AA64_RESET_HIGH` | reset vector bits 63:32 |
//!
//! The NX bootloader's way of starting the CPU cluster
//! (`nxbootloader-research` §2.9, hekate `ccplex_boot_cpu0`): write the
//! secure monitor's entry with bit 0 set to `SB_AA64_RESET_LOW`, zero to
//! `SB_AA64_RESET_HIGH`, set `NS_RST_VEC_WR_DIS` (`SB_CSR = 2`), then
//! release core 0's reset in the clock and reset controller.
//! [`Sb::aa64_reset_vector`] is where the machine starts core 0.
//!
//! Power-on values: what the NX bootloader's hardware-state check requires
//! before it writes the block (`nxbl_hwcheck.csv`, text `0x40002A7C`):
//! `SB_PFCFG`, the spare registers and both reset vector halves zero, and
//! `SB_CSR` with only `PIROM_DISABLE` set and `SB_PIROM_START` 0x1000 —
//! those two the boot ROM's lock-outs ([`crate::bootrom::ROM_LOCKOUTS`],
//! applied with [`Sb::apply_rom`]); before them `SB_CSR` is
//! \[assumption\] zero and `SB_PIROM_START` refuses to be read. Bits the
//! check does not look at are \[assumption\] zero.
//!
//! \[assumption\] Once `NS_RST_VEC_WR_DIS` is set, writes of the reset
//! vector by non-secure masters are dropped, as the name says; the BPMP
//! counts as secure (it sets the lock, and the secure monitor rewrites the
//! vector later), the CPU cluster's cores here run only non-secure
//! software (the monitor is high-level emulated), so their writes are the
//! ones dropped. \[assumption\] The lock bits are sticky until reset:
//! writing zero does not clear them. Writes that set other `SB_CSR` bits,
//! and writes of the read-only registers above, are refused.

use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use super::regs::word;
use crate::{
    boot::CORE_COUNT,
    bus::MmioDevice,
    snapshot::{Snapshot, SnapshotError, decode, encode},
};

pub const BASE: u64 = 0x6000_C200;
pub const SIZE: u64 = 0x100;

const CSR: u64 = 0x00;
const PIROM_START: u64 = 0x04;
const PFCFG: u64 = 0x08;
const SPARE: u64 = 0x0C;
const SPARE_COUNT: usize = 8;
const AA64_RESET_LOW: u64 = 0x30;
const AA64_RESET_HIGH: u64 = 0x34;

pub const CSR_NS_RST_VEC_WR_DIS: u32 = 1 << 1;
pub const CSR_PIROM_DISABLE: u32 = 1 << 4;
const AARCH64_MODE_EN: u32 = 1 << 0;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
struct State {
    csr: u32,
    pirom_start: Option<u32>,
    pfcfg: u32,
    spare: [u32; SPARE_COUNT],
    reset_low: u32,
    reset_high: u32,
}

/// Where a CPU-cluster core starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResetVector {
    pub address: u64,
    /// `AARCH64_MODE_EN`: the core starts in AArch64 at `address`; else
    /// in AArch32 at the exception vector block's CPU reset vector.
    pub aarch64: bool,
}

#[derive(Default)]
pub struct Sb {
    state: Mutex<State>,
}

impl Sb {
    pub fn new() -> Self {
        Self::default()
    }

    /// Back to the state after reset, lock-outs and reset vector cleared,
    /// as the SoC powers up again after SC7 (the boot ROM then applies its
    /// lock-outs again, and the warmboot firmware writes the vector).
    pub fn reset(&self) {
        *self.state.lock().unwrap() = State::default();
    }

    /// The AArch64 reset vector.
    pub fn aa64_reset_vector(&self) -> ResetVector {
        let state = *self.state.lock().unwrap();
        ResetVector {
            address: u64::from(state.reset_high) << 32 | u64::from(state.reset_low & !3),
            aarch64: state.reset_low & AARCH64_MODE_EN != 0,
        }
    }

    /// What the boot ROM leaves in the register at `offset`: the bits in
    /// `mask` are `value`. `None` for a register the ROM's lock-outs do not
    /// name (`SB_CSR`, `SB_PIROM_START`).
    pub fn apply_rom(&self, offset: u64, mask: u32, value: u32) -> Option<()> {
        let mut state = self.state.lock().unwrap();
        let register = match offset {
            CSR => &mut state.csr,
            PIROM_START => state.pirom_start.get_or_insert(0),
            _ => return None,
        };
        *register = *register & !mask | value & mask;
        Some(())
    }
}

impl MmioDevice for Sb {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        let state = *self.state.lock().unwrap();
        let offset = word(offset, size)?;
        let value = match offset {
            CSR => state.csr,
            PIROM_START => state.pirom_start?,
            PFCFG => state.pfcfg,
            AA64_RESET_LOW => state.reset_low,
            AA64_RESET_HIGH => state.reset_high,
            _ => *state
                .spare
                .get(usize::try_from(offset.checked_sub(SPARE)? / 4).ok()?)?,
        };
        Some(u64::from(value))
    }

    fn write(&self, core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        let value = value as u32;
        let mut state = self.state.lock().unwrap();
        let offset = word(offset, size)?;
        let non_secure = core < CORE_COUNT;
        let locked = state.csr & CSR_NS_RST_VEC_WR_DIS != 0;
        match offset {
            CSR if value & !(CSR_NS_RST_VEC_WR_DIS | CSR_PIROM_DISABLE) == 0 => state.csr |= value,
            AA64_RESET_LOW | AA64_RESET_HIGH if non_secure && locked => {}
            AA64_RESET_LOW => state.reset_low = value,
            AA64_RESET_HIGH => state.reset_high = value,
            PFCFG => state.pfcfg = value,
            _ => return None,
        }
        Some(())
    }

    fn snapshot(&self) -> Option<&dyn Snapshot> {
        Some(self)
    }
}

impl Snapshot for Sb {
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        encode(&*self.state.lock().unwrap())
    }

    fn restore(&self, state: &[u8]) -> Result<(), SnapshotError> {
        *self.state.lock().unwrap() = decode("secure boot registers", state)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BPMP: usize = CORE_COUNT;

    fn after_rom() -> Sb {
        let sb = Sb::new();
        for lockout in crate::bootrom::ROM_LOCKOUTS {
            sb.apply_rom(
                u64::from(lockout.address) - BASE,
                lockout.mask,
                lockout.value,
            )
            .unwrap();
        }
        sb
    }

    #[test]
    fn reads_what_the_bootloader_checks() {
        assert_eq!(Sb::new().read(BPMP, PIROM_START, 4), None);
        let sb = after_rom();
        assert_eq!(sb.read(BPMP, CSR, 4), Some(0x10));
        assert_eq!(sb.read(BPMP, PIROM_START, 4), Some(0x1000));
        for offset in [PFCFG, 0x0C, 0x28, AA64_RESET_LOW, AA64_RESET_HIGH] {
            assert_eq!(sb.read(BPMP, offset, 4), Some(0), "{offset:#x}");
        }
        assert_eq!(sb.read(BPMP, 0x2C, 4), None);
        assert_eq!(sb.apply_rom(PFCFG, 1, 1), None);
    }

    #[test]
    fn reset_vector_as_the_bootloader_writes_it() {
        let sb = after_rom();
        sb.write(BPMP, AA64_RESET_LOW, 4, 0x4003_0000 | 1).unwrap();
        sb.write(BPMP, AA64_RESET_HIGH, 4, 0).unwrap();
        assert_eq!(
            sb.aa64_reset_vector(),
            ResetVector {
                address: 0x4003_0000,
                aarch64: true
            }
        );
        assert_eq!(sb.read(0, AA64_RESET_LOW, 4), Some(0x4003_0001));
        sb.write(BPMP, CSR, 4, u64::from(CSR_NS_RST_VEC_WR_DIS))
            .unwrap();
        assert_eq!(sb.read(BPMP, CSR, 4), Some(0x12));
    }

    #[test]
    fn lock_drops_non_secure_writes_and_is_sticky() {
        let sb = after_rom();
        sb.write(BPMP, AA64_RESET_LOW, 4, 0x4003_0001).unwrap();
        // Unlocked: the CPU cluster may still write.
        sb.write(0, AA64_RESET_HIGH, 4, 0).unwrap();
        sb.write(BPMP, CSR, 4, u64::from(CSR_NS_RST_VEC_WR_DIS))
            .unwrap();
        sb.write(1, AA64_RESET_LOW, 4, 0x8000_0001).unwrap();
        assert_eq!(sb.aa64_reset_vector().address, 0x4003_0000);
        sb.write(BPMP, CSR, 4, 0).unwrap();
        assert_eq!(sb.read(0, CSR, 4), Some(0x12));
        // The secure side still may.
        sb.write(BPMP, AA64_RESET_LOW, 4, 0x4004_0001).unwrap();
        assert_eq!(sb.aa64_reset_vector().address, 0x4004_0000);
    }

    #[test]
    fn refuses_the_rest() {
        let sb = after_rom();
        assert_eq!(sb.write(BPMP, CSR, 4, 1), None);
        assert_eq!(sb.write(BPMP, PIROM_START, 4, 0), None);
        assert_eq!(sb.write(BPMP, 0x0C, 4, 0), None);
        assert_eq!(sb.write(BPMP, AA64_RESET_LOW, 2, 0), None);
    }

    #[test]
    fn snapshot_round_trip() {
        let sb = after_rom();
        sb.write(BPMP, AA64_RESET_LOW, 4, 0x4003_0001).unwrap();
        let other = Sb::new();
        other.restore(&sb.save().unwrap()).unwrap();
        assert_eq!(other.aa64_reset_vector(), sb.aa64_reset_vector());
        assert_eq!(other.read(0, PIROM_START, 4), Some(0x1000));
    }
}
