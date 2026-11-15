//! MAX17050 battery fuel gauge, on I2C1 at address 0x36.
//!
//! Sixteen-bit registers, low byte first. This is a gauge whose battery
//! has just been connected: the status register reports a power-on reset,
//! nothing is configured yet, and the boot process answers by loading its
//! battery model and parameters (`boot` text `0x43590`), exactly as on a
//! console leaving the factory.
//!
//! The gauge here measures; it does not estimate. Voltage, current,
//! temperature and the voltage-derived state of charge come from the
//! [`Battery`] it is attached to. The registers the driver programs hold
//! what was written, and the ones the gauge's own algorithm would keep
//! refining stay at what the driver seeded them with. A register that was
//! never written and is not a measurement refuses to be read.

use std::{collections::BTreeMap, sync::Arc};

use super::{Notes, i2c::I2cDevice};

pub const ADDRESS: u8 = 0x36;

/// What the gauge is attached to. The console has no charger connected and
/// runs from this battery.
#[derive(Debug, Clone, Copy)]
pub struct Battery {
    pub millivolts: u32,
    /// Negative while the battery discharges.
    pub milliamps: i32,
    pub celsius: i32,
    /// State of charge.
    pub percent: u32,
}

impl Battery {
    /// A full battery at room temperature under the load of a console
    /// that is starting up.
    pub const FULL: Self = Self {
        millivolts: 4200,
        milliamps: -500,
        celsius: 25,
        percent: 100,
    };
}

mod register {
    pub const STATUS: u8 = 0x00;
    /// Age, in 1/256 percent: public hekate `bdk/power/max17050.h`.
    pub const AGE: u8 = 0x07;
    pub const TEMPERATURE: u8 = 0x08;
    pub const VCELL: u8 = 0x09;
    pub const CURRENT: u8 = 0x0A;
    pub const AVERAGE_CURRENT: u8 = 0x0B;
    pub const AVERAGE_TEMPERATURE: u8 = 0x16;
    pub const AVERAGE_VCELL: u8 = 0x19;
    /// Highest and lowest temperature, voltage and current measured since
    /// the register was last written: maximum in the upper byte, minimum
    /// in the lower (hekate `MinMaxTemp`, `MinMaxVolt`, `MinMaxCurr`).
    pub const MIN_MAX_TEMPERATURE: u8 = 0x1A;
    pub const MIN_MAX_VOLTAGE: u8 = 0x1B;
    pub const MIN_MAX_CURRENT: u8 = 0x1C;
    /// Charge counted since power-on.
    pub const QH: u8 = 0x4D;
    /// The two keys that unlock the battery model table.
    pub const MODEL_ACCESS_1: u8 = 0x62;
    pub const MODEL_ACCESS_2: u8 = 0x63;
    pub const MODEL_TABLE: std::ops::Range<u8> = 0x80..0xB0;
    /// State of charge by the voltage fuel gauge alone.
    pub const SOCVF: u8 = 0xFF;
}

/// Power-on reset: the gauge lost its configuration.
const STATUS_POWER_ON_RESET: u16 = 1 << 1;

/// Registers the driver reads before it has written them, with their
/// power-on values from the datasheet. Alert thresholds initially span
/// their representable range (assumed unconfigured board); ptm changes
/// them by read-modify-write. Register names: public hekate
/// <https://github.com/CTCaer/hekate/blob/master/bdk/power/max17050.h>.
const POWER_ON_VALUES: [(u8, u16); 9] = [
    (register::STATUS, STATUS_POWER_ON_RESET),
    (0x01, 0xff00),
    (0x02, 0x7f80),
    (0x03, 0xff00),
    // The min/max registers start out empty: each maximum at the bottom
    // of its range and each minimum at the top, so that the first
    // measurement becomes both. Datasheet power-on values as remembered,
    // not checked against the document [assumption]; only their being
    // empty matters here, since every read includes a measurement.
    (register::MIN_MAX_TEMPERATURE, 0x807F),
    (register::MIN_MAX_VOLTAGE, 0x00FF),
    (register::MIN_MAX_CURRENT, 0x807F),
    // MiscCFG.
    (0x2B, 0x0870),
    // COFF, the current measurement's offset: none.
    (0x2F, 0x0000),
];

const MODEL_KEYS: (u16, u16) = (0x0059, 0x00C4);

/// Registers the driver may program that are plain storage here, by the
/// datasheet's register map: alert thresholds, capacities and the
/// application's seeds for them, configuration, and characterization data.
const PROGRAMMABLE: [u8; 47] = [
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x0D, 0x0E, 0x0F, 0x10, 0x12, 0x13, 0x17, 0x18,
    0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x22, 0x23, 0x24, 0x25, 0x28, 0x29, 0x2A, 0x2B, 0x2C, 0x2D,
    0x2E, 0x2F, 0x32, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x3F, 0x42, 0x45, 0x46, 0x48, 0x4C, 0x60,
];

/// A min/max register as the gauge keeps it: what was written (or the
/// power-on value), widened by `measured`, which is in the register's
/// units. The battery here holds still, so its one measurement is all the
/// gauge has seen since.
fn min_max(stored: u16, measured: i32, signed: bool) -> u16 {
    let [max, min] = stored.to_be_bytes();
    let (max, min, range) = if signed {
        (i32::from(max as i8), i32::from(min as i8), -128..=127)
    } else {
        (i32::from(max), i32::from(min), 0..=255)
    };
    let measured = measured.clamp(*range.start(), *range.end());
    // Truncating to a byte keeps two's complement for the signed ones.
    u16::from_be_bytes([max.max(measured) as u8, min.min(measured) as u8])
}

pub struct Max17050 {
    battery: Battery,
    pointer: u8,
    /// Everything that has been written, and the status register.
    registers: BTreeMap<u8, u16>,
    notes: Arc<Notes>,
}

impl Max17050 {
    pub fn new(battery: Battery, notes: Arc<Notes>) -> Self {
        let registers = BTreeMap::from(POWER_ON_VALUES);
        Self {
            battery,
            pointer: 0,
            registers,
            notes,
        }
    }

    fn refuse(&self, what: &str) -> Option<bool> {
        self.notes.add(format!(
            "fuel gauge: {what} register {:#04x} is not modeled",
            self.pointer
        ));
        None
    }

    fn model_unlocked(&self) -> bool {
        let key = |register| self.registers.get(&register).copied();
        (key(register::MODEL_ACCESS_1), key(register::MODEL_ACCESS_2))
            == (Some(MODEL_KEYS.0), Some(MODEL_KEYS.1))
    }

    fn read_register(&self, register: u8) -> Option<u16> {
        let battery = &self.battery;
        Some(match register {
            // A new battery retains 100% of its design capacity.
            register::AGE => 100 * 256,
            // 78.125 uV per unit, in the upper 13 bits.
            register::VCELL | register::AVERAGE_VCELL => ((battery.millivolts * 8 / 5) << 3) as u16,
            // 1.5625 uV across the 5 mOhm sense resistor per unit.
            register::CURRENT | register::AVERAGE_CURRENT => (battery.milliamps * 16 / 5) as u16,
            // 1/256 degree per unit.
            register::TEMPERATURE | register::AVERAGE_TEMPERATURE => (battery.celsius * 256) as u16,
            // 1 degree per unit.
            register::MIN_MAX_TEMPERATURE => {
                min_max(*self.registers.get(&register)?, battery.celsius, true)
            }
            // 20 mV per unit.
            register::MIN_MAX_VOLTAGE => min_max(
                *self.registers.get(&register)?,
                (battery.millivolts / 20) as i32,
                false,
            ),
            // 0.4 mV across the 5 mOhm sense resistor, 80 mA, per unit.
            register::MIN_MAX_CURRENT => min_max(
                *self.registers.get(&register)?,
                battery.milliamps / 80,
                true,
            ),
            // 1/256 percent per unit.
            register::SOCVF => (battery.percent * 256) as u16,
            // Nothing has been counted yet.
            register::QH => 0,
            // The table reads as zeros while it is locked.
            _ if register::MODEL_TABLE.contains(&register) && !self.model_unlocked() => 0,
            _ => *self.registers.get(&register)?,
        })
    }

    fn write_register(&mut self, register: u8, value: u16) -> Option<()> {
        let table = register::MODEL_TABLE.contains(&register);
        let keys = matches!(
            register,
            register::MODEL_ACCESS_1 | register::MODEL_ACCESS_2
        );
        if !(PROGRAMMABLE.contains(&register) || keys || table) {
            return None;
        }
        // Writes to the locked table are lost.
        if !table || self.model_unlocked() {
            self.registers.insert(register, value);
        }
        Some(())
    }
}

impl I2cDevice for Max17050 {
    fn write(&mut self, data: &[u8]) -> Option<bool> {
        let (&pointer, values) = data.split_first()?;
        self.pointer = pointer;
        let (words, []) = values.as_chunks::<2>() else {
            return self.refuse("half a write to");
        };
        for word in words {
            if self
                .write_register(self.pointer, u16::from_le_bytes(*word))
                .is_none()
            {
                return self.refuse("write to");
            }
            self.pointer = self.pointer.wrapping_add(1);
        }
        Some(true)
    }

    fn read(&mut self, data: &mut [u8]) -> Option<bool> {
        let (words, []) = data.as_chunks_mut::<2>() else {
            return self.refuse("half a read of");
        };
        for word in words {
            let Some(value) = self.read_register(self.pointer) else {
                return self.refuse("read of");
            };
            *word = value.to_le_bytes();
            self.pointer = self.pointer.wrapping_add(1);
        }
        Some(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(gauge: &mut Max17050, register: u8) -> Option<u16> {
        let mut word = [0; 2];
        gauge.write(&[register])?;
        gauge.read(&mut word)?;
        Some(u16::from_le_bytes(word))
    }

    fn write(gauge: &mut Max17050, register: u8, value: u16) -> Option<bool> {
        let [low, high] = value.to_le_bytes();
        gauge.write(&[register, low, high])
    }

    #[test]
    fn measurements_are_in_the_units_of_the_datasheet() {
        let battery = Battery {
            millivolts: 3700,
            milliamps: -1000,
            celsius: -10,
            percent: 55,
        };
        let mut gauge = Max17050::new(battery, Arc::default());
        // 3.7 V in 0.625 mV steps above three unused bits.
        assert_eq!(read(&mut gauge, register::VCELL), Some(5920 << 3));
        // 1 A discharging in 312.5 uA steps.
        assert_eq!(read(&mut gauge, register::CURRENT), Some(-3200i16 as u16));
        assert_eq!(read(&mut gauge, register::TEMPERATURE), Some(0xF600));
        assert_eq!(read(&mut gauge, register::SOCVF), Some(55 << 8));
        assert_eq!(
            read(&mut gauge, register::STATUS),
            Some(STATUS_POWER_ON_RESET)
        );
    }

    #[test]
    fn model_table_is_only_there_between_unlock_and_lock() {
        let mut gauge = Max17050::new(Battery::FULL, Arc::default());
        write(&mut gauge, 0x80, 0x1234).unwrap();
        write(&mut gauge, register::MODEL_ACCESS_1, MODEL_KEYS.0).unwrap();
        write(&mut gauge, register::MODEL_ACCESS_2, MODEL_KEYS.1).unwrap();
        assert_eq!(read(&mut gauge, 0x80), None, "the locked write was lost");
        write(&mut gauge, 0x80, 0xBEEF).unwrap();
        assert_eq!(read(&mut gauge, 0x80), Some(0xBEEF));
        write(&mut gauge, register::MODEL_ACCESS_1, 0).unwrap();
        write(&mut gauge, register::MODEL_ACCESS_2, 0).unwrap();
        assert_eq!(read(&mut gauge, 0x80), Some(0));
    }

    #[test]
    fn min_max_registers_hold_what_was_measured_since_they_were_written() {
        let battery = Battery {
            millivolts: 3700,
            milliamps: -1000,
            celsius: -10,
            percent: 55,
        };
        let mut gauge = Max17050::new(battery, Arc::default());
        // From power-on, the one measurement is both maximum and minimum:
        // -10 degrees, 3.7 V in 20 mV, -1 A in 80 mA (truncated).
        assert_eq!(
            read(&mut gauge, register::MIN_MAX_TEMPERATURE),
            Some(0xF6F6)
        );
        assert_eq!(read(&mut gauge, register::MIN_MAX_VOLTAGE), Some(0xB9B9));
        assert_eq!(read(&mut gauge, register::MIN_MAX_CURRENT), Some(0xF4F4));
        // A written history is widened by the measurement, not replaced.
        write(&mut gauge, register::MIN_MAX_TEMPERATURE, 0x2814).unwrap();
        assert_eq!(
            read(&mut gauge, register::MIN_MAX_TEMPERATURE),
            Some(0x28F6)
        );
        write(&mut gauge, register::MIN_MAX_VOLTAGE, 0xC8B0).unwrap();
        assert_eq!(read(&mut gauge, register::MIN_MAX_VOLTAGE), Some(0xC8B0));
    }

    #[test]
    fn unwritten_configuration_and_unknown_registers_refuse() {
        let mut gauge = Max17050::new(Battery::FULL, Arc::default());
        assert_eq!(read(&mut gauge, 0x1D), None);
        write(&mut gauge, 0x1D, 0x2210).unwrap();
        assert_eq!(read(&mut gauge, 0x1D), Some(0x2210));
        assert_eq!(
            write(&mut gauge, 0x09, 0),
            None,
            "measurements cannot be written"
        );
        assert_eq!(gauge.write(&[0x1D, 0x10]), None);
    }
}
