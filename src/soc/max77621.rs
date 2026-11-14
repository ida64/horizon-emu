//! MAX77621 step-down regulators on I2C5: one feeds the processor cores,
//! one the GPU.
//!
//! The firmware's power driver programs both control registers of each
//! before it reads anything (pcv, on the bus as `[02, b0]`, `[03, c2]`),
//! then changes the output register a bit at a time. Its regulator table
//! gives both a default of 1.0 V in steps of 6.25 mV (pcv data `0x9e350`,
//! entries 13 and 14); the base of the scale, 606.25 mV, is recalled from
//! open-source bootloaders. The processor's regulator is on, or nothing
//! would be running; the GPU's is taken to be off, which is how those
//! bootloaders hand the console over.

use super::i2c::RegisterMap;

pub const CPU_ADDRESS: u8 = 0x1B;
pub const GPU_ADDRESS: u8 = 0x1C;

/// Output voltage in bits 6:0, and [`VOUT_ENABLE`].
const VOUT: u8 = 0x00;
const VOUT_ENABLE: u8 = 1 << 7;
/// The output voltage used while the chip's voltage-select pin is high.
const VOUT_DVS: u8 = 0x01;
const CONTROL1: u8 = 0x02;
const CONTROL2: u8 = 0x03;

/// 1.0 V: `(1_000_000 - 606_250) / 6_250`.
const VOUT_1_VOLT: u8 = 63;

/// One of the two regulators. The chip identification registers are not
/// modeled.
pub struct Max77621 {
    registers: [u8; 4],
}

impl Max77621 {
    /// The regulator of the processor cores: on, at its default voltage.
    pub fn cpu() -> Self {
        Self::new(VOUT_ENABLE)
    }

    /// The regulator of the GPU: off.
    pub fn gpu() -> Self {
        Self::new(0)
    }

    fn new(enable: u8) -> Self {
        Self {
            registers: [VOUT_1_VOLT | enable, VOUT_1_VOLT, 0, 0],
        }
    }
}

impl RegisterMap for Max77621 {
    fn read(&mut self, register: u8) -> Option<u8> {
        self.registers.get(usize::from(register)).copied()
    }

    fn write(&mut self, register: u8, value: u8) -> Option<()> {
        debug_assert!([VOUT, VOUT_DVS, CONTROL1, CONTROL2].len() == self.registers.len());
        *self.registers.get_mut(usize::from(register))? = value;
        Some(())
    }
}
