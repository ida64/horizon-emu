//! Power management controller: pad voltages, deep power down requests,
//! wake sources, scratch registers and the power partitions. Reached only
//! through the secure monitor's register call.
//!
//! Most of it is configuration storage. `boot` sets up the wake pins by
//! reading a register whole and writing it back whole, so what the block
//! starts with only survives in bits `boot` leaves alone; registers start
//! at zero, which stands for whatever the earlier boot stages programmed.
//!
//! The power partitions behave: the firmware's clock driver turns a
//! partition on or off by writing its ID with the start bit to
//! `PWRGATE_TOGGLE`, waits for the start bit to clear ("accept"), then for
//! the partition's bit in `PWRGATE_STATUS` (`VccPgSetPowerEnabled`, pcv
//! text `0x3d730`: up to 100 polls each, and it fails the request
//! otherwise). Here a partition switches at once. Register and partition
//! names are as in Linux (`drivers/soc/tegra/pmc.c`, Tegra210 table),
//! recalled.
//!
//! So do the wake events, as far as something here raises one: an event
//! that is enabled and at its wake level is recorded in the status
//! registers, which is where the firmware looks, one event at a time,
//! for why the console woke. See [`Pmc::wake`].

use std::sync::Mutex;

use super::regs::{RegisterFile, word};
use crate::{
    bus::MmioDevice,
    snapshot::{Snapshot, SnapshotError, decode, encode},
};

pub const BASE: u64 = 0x7000_E400;
pub const SIZE: u64 = 0xC00;

const PWRGATE_TOGGLE: u64 = 0x30;
const TOGGLE_START: u32 = 1 << 8;
const TOGGLE_PARTITION: u32 = 0x1F;
const PWRGATE_STATUS: u64 = 0x38;

/// The 64 wake events, 32 to a register: which may wake the console
/// (`WAKE_MASK`, `WAKE2_MASK`), at which level of their input, high if
/// set (`WAKE_LVL`, `WAKE2_LVL`: the firmware sets the one event its
/// table has as active high, the radio's, and leaves the active low ones
/// clear), and which have (`WAKE_STATUS`, `WAKE2_STATUS`). The firmware
/// clears the status by writing all ones before the console sleeps and
/// reads it 64 times after (Bus through the monitor), so a write clears
/// the bits it has set.
const WAKE_MASK: [u64; 2] = [0x0C, 0x160];
const WAKE_LVL: [u64; 2] = [0x10, 0x164];
const WAKE_STATUS: [u64; 2] = [0x14, 0x168];

/// The wake event of the power management IC's interrupt output, active
/// low (`PwrIntN` in Atmosphère's list of this board's wake events,
/// `wec_wake_event.board.nintendo_nx.hpp`; the firmware enables it, bit
/// 19 of `WAKE2_MASK`).
pub const WAKE_EVENT_PMIC: u8 = 51;

/// Partitions powered at kernel entry: the CPU rail (CRAIL, 0), the
/// non-CPU part of the fast cluster (C0NC, 15) and its first core (CE0, 14)
/// — fusee `fusee_cpu.cpp:141-143` powers these before starting the CPU.
/// The other three cores (CE1-3, 9-11) are the secure monitor's to switch,
/// when the kernel starts and stops them. Every other partition is assumed
/// off: no public stage of the boot chain powers one, and the firmware's
/// clock driver turns on what it needs.
const POWERED_AT_ENTRY: u32 = 1 << 0 | 1 << 14 | 1 << 15;

/// The GPU rail's gate: bit 0 set clamps the rail off. The firmware's
/// clock driver reads it when it starts, to count the rail as in use if
/// it finds it open, and writes 0 to open and 1 to close it (pcv text
/// `0x3db30`, `0x3d9c0`); with the rail counted twice it refuses the power
/// request by which nvservices ends its start-up GPU cycle. The rail is
/// closed at entry: the GPU's regulator is off (see `max77621`), and no
/// boot stage uses the GPU.
const GPU_RG_CNTRL: u64 = 0x2D4;
const GPU_RAIL_CLAMPED: u32 = 1 << 0;

pub struct Pmc {
    registers: RegisterFile,
    powered: Mutex<u32>,
}

impl Pmc {
    pub fn new() -> Self {
        Self {
            registers: RegisterFile::zeroed([(GPU_RG_CNTRL, GPU_RAIL_CLAMPED)]),
            powered: Mutex::new(POWERED_AT_ENTRY),
        }
    }

    /// The input of wake event `event` is at this level. If the event is
    /// enabled and that is its wake level, it is recorded and the answer
    /// is `true`: the console wakes.
    pub fn wake(&self, event: u8, high: bool) -> bool {
        let (index, bit) = (usize::from(event / 32), 1 << (event % 32));
        let is_set = |register| self.registers.get(register).unwrap_or(0) & bit != 0;
        let wakes = is_set(WAKE_MASK[index]) && is_set(WAKE_LVL[index]) == high;
        if wakes {
            let status = self.registers.get(WAKE_STATUS[index]).unwrap_or(0);
            self.registers.set(WAKE_STATUS[index], status | bit);
        }
        wakes
    }
}

impl Default for Pmc {
    fn default() -> Self {
        Self::new()
    }
}

impl MmioDevice for Pmc {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        let value = match word(offset, size)? {
            PWRGATE_STATUS => *self.powered.lock().unwrap(),
            // A toggle is accepted at once, so the start bit never reads set.
            PWRGATE_TOGGLE => self.registers.get(PWRGATE_TOGGLE)? & !TOGGLE_START,
            register => self.registers.get(register)?,
        };
        Some(value.into())
    }

    fn write(&self, _core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        let register = word(offset, size)?;
        let value = value as u32;
        match register {
            // Read-only.
            PWRGATE_STATUS => return None,
            PWRGATE_TOGGLE if value & TOGGLE_START != 0 => {
                *self.powered.lock().unwrap() ^= 1 << (value & TOGGLE_PARTITION);
            }
            _ => {}
        }
        let value = if WAKE_STATUS.contains(&register) {
            self.registers.get(register)? & !value
        } else {
            value
        };
        self.registers.set(register, value);
        Some(())
    }

    fn snapshot(&self) -> Option<&dyn Snapshot> {
        Some(self)
    }
}

/// Saves the registers, wake status included, and which partitions are
/// powered.
impl Snapshot for Pmc {
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        encode(&(self.registers.save()?, *self.powered.lock().unwrap()))
    }

    fn restore(&self, state: &[u8]) -> Result<(), SnapshotError> {
        let (registers, powered): (Vec<u8>, u32) = decode("PMC", state)?;
        self.registers.restore(&registers)?;
        *self.powered.lock().unwrap() = powered;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_toggle_switches_one_partition_and_is_accepted_at_once() {
        let pmc = Pmc::new();
        let status = |pmc: &Pmc| pmc.read(0, PWRGATE_STATUS, 4).unwrap() as u32;
        assert_eq!(status(&pmc) & 1 << 3, 0);
        pmc.write(0, PWRGATE_TOGGLE, 4, u64::from(TOGGLE_START | 3))
            .unwrap();
        assert_eq!(pmc.read(0, PWRGATE_TOGGLE, 4), Some(3));
        assert_eq!(status(&pmc), POWERED_AT_ENTRY | 1 << 3);
        pmc.write(0, PWRGATE_TOGGLE, 4, u64::from(TOGGLE_START | 3))
            .unwrap();
        assert_eq!(status(&pmc), POWERED_AT_ENTRY);
        // Without the start bit nothing switches.
        pmc.write(0, PWRGATE_TOGGLE, 4, 3).unwrap();
        assert_eq!(status(&pmc), POWERED_AT_ENTRY);
    }

    #[test]
    fn an_enabled_wake_event_at_its_level_is_recorded_until_cleared() {
        let pmc = Pmc::new();
        let status = |pmc: &Pmc| pmc.read(0, WAKE_STATUS[1], 4).unwrap();
        assert!(!pmc.wake(WAKE_EVENT_PMIC, false), "not enabled");
        pmc.write(0, WAKE_MASK[1], 4, 1 << 19).unwrap();
        assert!(!pmc.wake(WAKE_EVENT_PMIC, true), "its wake level is low");
        assert_eq!(status(&pmc), 0);
        assert!(pmc.wake(WAKE_EVENT_PMIC, false));
        assert_eq!(status(&pmc), 1 << 19);
        pmc.write(0, WAKE_STATUS[1], 4, 0xFFFF_FFFF).unwrap();
        assert_eq!(status(&pmc), 0);
    }
}
