//! CPU closed-loop DVFS controller (CL-DVFS): the DFLL that clocks the CPU
//! cluster from a digitally controlled oscillator (DVCO) and sets the CPU
//! regulator's voltage over its own I2C link.
//!
//! The firmware's clock driver (`pcv`) finds it disabled at start-up
//! (`ClDvfsInit`, text `0x547f0`, reads `CTRL`: "CLDVFS enabled on boot"
//! otherwise), programs it (`ClDvfsInitOutIf` `0x558c0`,
//! `ClDvfsInitCntrlLogic` `0x55e30`), and when it first sets the CPU rate
//! closes the loop (`ClDvfsLock` `0x524d0`) and calibrates the DVCO's rate
//! at the minimum voltage (`ClDvfsCalibrate` `0x52db0`): it selects the
//! frequency monitor, waits — without a timeout — for a new sample,
//! checks two samples agree, and reads the output index the controller
//! last sent to the regulator. Register names and the rate units are those
//! of Linux `drivers/clk/tegra/clk-dfll.c` (`DVCO rate = MULT · ref / 2`,
//! `monitored rate = sample · ref / 4`), recalled; `pcv` uses the same
//! units (calibration sums `sample · ref / 4`).
//!
//! Model: the loop always reaches its request. With the loop closed and a
//! valid frequency request the DVCO runs at the requested rate, so the
//! frequency monitor reads `2 · MULT`; a sample completes every four
//! reference cycles, so a new one is always ready. The output is the
//! forced value when forcing is enabled and otherwise the configured
//! minimum: no chip voltage-frequency curve is modelled, so the CPU is
//! taken to reach every requested rate at the lowest voltage the
//! controller may use. Nothing is sent to the regulator. The open-loop
//! DVCO rate depends on the silicon and refuses to be read.

use std::sync::Mutex;

use super::regs::{RegisterFile, word};
use crate::{
    bus::MmioDevice,
    snapshot::{Snapshot, SnapshotError, decode, encode},
};

pub const BASE: u64 = 0x7011_0000;
pub const SIZE: u64 = 0x400;

const CTRL: u64 = 0x00;
const MODE_MASK: u32 = 3;
const MODE_DISABLED: u32 = 0;
const MODE_CLOSED_LOOP: u32 = 2;
const FREQ_REQ: u64 = 0x14;
const FREQ_MULT: u32 = 0x7F;
const FREQ_VALID: u32 = 1 << 7;
const OUTPUT_CFG: u64 = 0x20;
const OUTPUT_FORCE: u64 = 0x24;
const FORCE_ENABLE: u32 = 1 << 6;
const OUTPUT_MASK: u32 = 0x3F;
const MONITOR_CTRL: u64 = 0x28;
const MONITOR_DATA: u64 = 0x2C;
const MONITOR_NEW: u32 = 1 << 16;
/// Monitor selects: the DVCO frequency (Linux `DFLL_MONITOR_CTRL_FREQ`)
/// and the output index (inferred: `pcv`'s calibration compares the
/// reading to the minimum output index, text `0x52db0`).
const MONITOR_FREQ: u32 = 6;
const MONITOR_OUTPUT: u32 = 5;
const I2C_CFG: u64 = 0x40;
/// Bit 0: a request is pending; bits 6:1: the last output sent.
const I2C_STS: u64 = 0x48;
const INTR_STS: u64 = 0x5C;

/// What the controller holds at kernel entry, as far as `pcv` reads it
/// before writing.
///
/// * `CTRL` 0, disabled: the CPU runs from PLLX at entry (the clock
///   controller's CPU burst policy), and no stage of the boot chain
///   programs the DFLL (fusee and hekate only enable its clock); `pcv`
///   restarts the CPU rail otherwise.
/// * `I2C_CFG` 0 (assumed): `pcv` reads it only to make sure its writes
///   have landed and discards the value (text `0x55bb4`, `LDR WZR`).
/// * `FREQ_REQ` 0 (assumed): `pcv` keeps its scale field when it first
///   rewrites the register (`0x55e30`) and later requests rates whole.
const ENTRY_STATE: [(u64, u32); 3] = [(CTRL, 0), (I2C_CFG, 0), (FREQ_REQ, 0)];

pub struct ClDvfs {
    registers: RegisterFile,
    /// Output index last sent to the regulator.
    last_output: Mutex<u32>,
}

impl ClDvfs {
    pub fn new() -> Self {
        Self {
            registers: RegisterFile::strict(ENTRY_STATE),
            last_output: Mutex::new(0),
        }
    }

    fn mode(&self) -> Option<u32> {
        Some(self.registers.get(CTRL)? & MODE_MASK)
    }

    /// The output index the controller drives, if it drives one.
    fn output(&self) -> Option<u32> {
        let force = self.registers.get(OUTPUT_FORCE).unwrap_or(0);
        if force & FORCE_ENABLE != 0 {
            return Some(force & OUTPUT_MASK);
        }
        if self.mode()? == MODE_CLOSED_LOOP {
            return Some(self.registers.get(OUTPUT_CFG)? >> 8 & OUTPUT_MASK);
        }
        None
    }

    fn monitor(&self) -> Option<u32> {
        let sample = match self.registers.get(MONITOR_CTRL)? {
            MONITOR_FREQ => match self.mode()? {
                MODE_DISABLED => 0,
                MODE_CLOSED_LOOP => {
                    let request = self.registers.get(FREQ_REQ)?;
                    if request & FREQ_VALID == 0 {
                        return None;
                    }
                    2 * (request & FREQ_MULT)
                }
                _ => return None,
            },
            MONITOR_OUTPUT => self.output()?,
            _ => return None,
        };
        Some(MONITOR_NEW | sample)
    }
}

impl Default for ClDvfs {
    fn default() -> Self {
        Self::new()
    }
}

impl MmioDevice for ClDvfs {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        let value = match word(offset, size)? {
            MONITOR_DATA => self.monitor()?,
            I2C_STS => {
                let mut last = self.last_output.lock().unwrap();
                if let Some(output) = self.output() {
                    *last = output;
                }
                *last << 1
            }
            register => self.registers.get(register)?,
        };
        Some(value.into())
    }

    fn write(&self, _core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        let register = word(offset, size)?;
        let value = value as u32;
        match register {
            // Write one to clear; the controller raises nothing.
            INTR_STS => {
                let pending = self.registers.get(INTR_STS).unwrap_or(0);
                self.registers.set(INTR_STS, pending & !value);
            }
            MONITOR_DATA | I2C_STS => return None,
            _ => self.registers.set(register, value),
        }
        Some(())
    }

    fn snapshot(&self) -> Option<&dyn Snapshot> {
        Some(self)
    }
}

/// Saves the registers and the output index last sent to the regulator.
impl Snapshot for ClDvfs {
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        encode(&(self.registers.save()?, *self.last_output.lock().unwrap()))
    }

    fn restore(&self, state: &[u8]) -> Result<(), SnapshotError> {
        let (registers, last_output): (Vec<u8>, u32) = decode("CL-DVFS", state)?;
        self.registers.restore(&registers)?;
        *self.last_output.lock().unwrap() = last_output;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dfll: &ClDvfs, register: u64, value: u32) {
        dfll.write(0, register, 4, value.into()).unwrap();
    }

    #[test]
    fn a_closed_loop_reaches_the_requested_rate() {
        let dfll = ClDvfs::new();
        write(&dfll, MONITOR_CTRL, MONITOR_FREQ);
        assert_eq!(dfll.read(0, MONITOR_DATA, 4), Some(u64::from(MONITOR_NEW)));
        write(&dfll, OUTPUT_CFG, 0x0C19_0B00);
        write(&dfll, FREQ_REQ, FREQ_VALID | 40);
        write(&dfll, CTRL, MODE_CLOSED_LOOP);
        assert_eq!(
            dfll.read(0, MONITOR_DATA, 4),
            Some(u64::from(MONITOR_NEW | 80))
        );
        write(&dfll, MONITOR_CTRL, MONITOR_OUTPUT);
        assert_eq!(
            dfll.read(0, MONITOR_DATA, 4),
            Some(u64::from(MONITOR_NEW | 0x0B))
        );
        assert_eq!(dfll.read(0, I2C_STS, 4), Some(0x0B << 1));
        write(&dfll, OUTPUT_FORCE, FORCE_ENABLE | 0x10);
        assert_eq!(
            dfll.read(0, MONITOR_DATA, 4),
            Some(u64::from(MONITOR_NEW | 0x10))
        );
        // Open loop depends on the silicon.
        write(&dfll, CTRL, 1);
        write(&dfll, MONITOR_CTRL, MONITOR_FREQ);
        assert_eq!(dfll.read(0, MONITOR_DATA, 4), None);
    }
}
