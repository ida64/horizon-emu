//! MSELECT (`0x5006_0000`): the bridge between the CPU cluster and the
//! SoC's buses, as far as a machine that runs the boot chain reaches it.
//!
//! It comes out of the boot ROM held in reset: `RST_DEVICES_V` bit 3 set
//! with its clock on (`CLK_OUT_ENB_V` bit 3), as the NX bootloader's
//! hardware-state check requires of the clock and reset controller. The
//! same check then reads twelve of its registers (`nxbl_hwcheck.csv`, the
//! check `0x40002A7C`, bootloader text `0x4000BBD8`..`0x4000BC84`) and
//! expects one value of all of them under their masks, `0xEAFFFFFE`, the
//! literal at `0x4000BC64`. \[inference\] That is what a read of the block
//! returns while it is held in reset, not twelve registers' reset values:
//! the twelve differ in purpose but not in value. Here a block held in
//! reset answers those twelve offsets with that word, refuses other reads,
//! and refuses writes (what they do to a block in reset is not known).
//!
//! The bootloader lets it out of reset (`RST_DEVICES_V &= ~8`, its
//! `0x400164A0`) and does not touch it again; the secure monitor then
//! read-modify-writes `MSELECT_CONFIG` (offset 0). Out of reset, the
//! registers' reset values are not public: reads refuse until a register
//! is written, and written registers hold what was written.

use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use super::regs::{RegisterFile, word};
use crate::{
    bus::MmioDevice,
    snapshot::{Snapshot, SnapshotError, decode, encode},
};

pub const BASE: u64 = 0x5006_0000;
pub const SIZE: u64 = 0x1000;

/// `RST_DEVICES_V` and its MSELECT bit (hekate `clock.h`).
pub const RESET: (u64, u32) = (0x358, 3);

/// The offsets the check reads while the block is held in reset.
const READ_IN_RESET: [u64; 12] = [
    0x00, 0x08, 0x0C, 0x4C, 0x50, 0x54, 0x58, 0x5C, 0x60, 0x64, 0x68, 0x6C,
];
/// What they read then.
const IN_RESET: u32 = 0xEAFF_FFFE;

#[derive(Serialize, Deserialize)]
struct State {
    held: bool,
}

pub struct Mselect {
    state: Mutex<State>,
    registers: RegisterFile,
}

impl Mselect {
    /// The block as the boot ROM leaves it: held in reset.
    pub fn power_on() -> Self {
        Self {
            state: Mutex::new(State { held: true }),
            registers: RegisterFile::strict([]),
        }
    }

    /// The clock and reset controller put the block in reset: what was
    /// written is gone.
    pub fn hold(&self) {
        self.state.lock().unwrap().held = true;
        self.registers.clear();
    }

    /// The clock and reset controller let the block out of reset.
    pub fn release(&self) {
        self.state.lock().unwrap().held = false;
    }
}

impl MmioDevice for Mselect {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        let offset = word(offset, size)?;
        if self.state.lock().unwrap().held {
            return READ_IN_RESET.contains(&offset).then_some(IN_RESET.into());
        }
        self.registers.get(offset).map(u64::from)
    }

    fn write(&self, _core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        let offset = word(offset, size)?;
        (!self.state.lock().unwrap().held).then_some(())?;
        self.registers.set(offset, value as u32);
        Some(())
    }

    fn snapshot(&self) -> Option<&dyn Snapshot> {
        Some(self)
    }
}

/// Saves whether the block is held in reset and what was written to it.
impl Snapshot for Mselect {
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        encode(&(&*self.state.lock().unwrap(), self.registers.save()?))
    }

    fn restore(&self, state: &[u8]) -> Result<(), SnapshotError> {
        let (saved, registers): (State, Vec<u8>) = decode("MSELECT", state)?;
        self.registers.restore(&registers)?;
        *self.state.lock().unwrap() = saved;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_reset_word_until_released() {
        let mselect = Mselect::power_on();
        assert_eq!(mselect.read(0, 0x00, 4), Some(IN_RESET.into()));
        assert_eq!(mselect.read(0, 0x6C, 4), Some(IN_RESET.into()));
        assert_eq!(mselect.read(0, 0x04, 4), None);
        assert_eq!(mselect.write(0, 0x00, 4, 0x3800_0000), None);
        mselect.release();
        assert_eq!(mselect.read(0, 0x00, 4), None);
        assert_eq!(mselect.write(0, 0x00, 4, 0x3800_0000), Some(()));
        assert_eq!(mselect.read(0, 0x00, 4), Some(0x3800_0000));
        mselect.hold();
        assert_eq!(mselect.read(0, 0x00, 4), Some(IN_RESET.into()));
        mselect.release();
        assert_eq!(mselect.read(0, 0x00, 4), None);
    }
}
