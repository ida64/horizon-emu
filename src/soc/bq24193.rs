//! BQ24193 battery charger, on I2C1 at address 0x6B.
//!
//! Eleven byte registers. The configuration registers start at the values
//! the part's datasheet gives for power-on and hold what is written; the
//! status registers describe a console running from a healthy battery with
//! nothing plugged into its USB port.

use super::i2c::RegisterMap;
use crate::snapshot::{SnapshotError, decode, encode};

pub const ADDRESS: u8 = 0x6B;

/// Power-on values of the eight configuration registers, 0x00-0x07.
const CONFIGURATION_AT_POWER_ON: [u8; 8] = [0x30, 0x1B, 0x60, 0x11, 0xB2, 0x9A, 0x03, 0x4B];

/// Self-clearing bit of the power-on configuration register: restores every
/// register to its power-on value.
const POWER_ON_CONFIGURATION: u8 = 0x01;
const REGISTER_RESET: u8 = 1 << 7;
/// Self-clearing bit of the same register: restarts the I2C watchdog.
const WATCHDOG_RESET: u8 = 1 << 6;

/// No input source, not charging, battery above the minimum system voltage.
const SYSTEM_STATUS: u8 = 0x08;
const SYSTEM_STATUS_VALUE: u8 = 0x00;
/// No watchdog, charge, battery or thermistor fault.
const FAULT: u8 = 0x09;
const FAULT_VALUE: u8 = 0x00;
/// Part number and revision of the BQ24193.
const VENDOR_PART: u8 = 0x0A;
const VENDOR_PART_VALUE: u8 = 0x2F;

pub struct Bq24193 {
    configuration: [u8; 8],
}

impl Bq24193 {
    pub fn new() -> Self {
        Self {
            configuration: CONFIGURATION_AT_POWER_ON,
        }
    }
}

impl Default for Bq24193 {
    fn default() -> Self {
        Self::new()
    }
}

impl RegisterMap for Bq24193 {
    fn read(&mut self, register: u8) -> Option<u8> {
        match register {
            SYSTEM_STATUS => Some(SYSTEM_STATUS_VALUE),
            FAULT => Some(FAULT_VALUE),
            VENDOR_PART => Some(VENDOR_PART_VALUE),
            _ => self.configuration.get(usize::from(register)).copied(),
        }
    }

    fn write(&mut self, register: u8, value: u8) -> Option<()> {
        match register {
            POWER_ON_CONFIGURATION if value & REGISTER_RESET != 0 => {
                self.configuration = CONFIGURATION_AT_POWER_ON
            }
            POWER_ON_CONFIGURATION => self.configuration[1] = value & !WATCHDOG_RESET,
            // The status registers ignore writes.
            SYSTEM_STATUS | FAULT | VENDOR_PART => {}
            _ => *self.configuration.get_mut(usize::from(register))? = value,
        }
        Some(())
    }

    /// The configuration registers; the status registers are constant.
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        encode(&self.configuration)
    }

    fn restore(&mut self, state: &[u8]) -> Result<(), SnapshotError> {
        self.configuration = decode("BQ24193", state)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configuration_holds_writes_and_self_clearing_bits_never_read_back() {
        let mut charger = Bq24193::new();
        charger.write(0x00, 0x07).unwrap();
        charger
            .write(POWER_ON_CONFIGURATION, 0x1B | WATCHDOG_RESET)
            .unwrap();
        assert_eq!(
            (charger.read(0x00), charger.read(0x01)),
            (Some(0x07), Some(0x1B))
        );

        charger
            .write(POWER_ON_CONFIGURATION, REGISTER_RESET)
            .unwrap();
        assert_eq!(
            (charger.read(0x00), charger.read(0x01)),
            (Some(0x30), Some(0x1B))
        );

        charger.write(SYSTEM_STATUS, 0xFF).unwrap();
        assert_eq!(charger.read(SYSTEM_STATUS), Some(0));
        assert_eq!((charger.read(0x0B), charger.write(0x0B, 0)), (None, None));
    }
}
