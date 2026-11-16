//! TI TMP451 local/remote temperature sensor on I2C1.
//!
//! Public register/address evidence: <https://github.com/CTCaer/hekate/blob/master/bdk/thermal/tmp451.h>.
//! Horizon 23.5.0 ptm `0x1d460` opens device 0x3e000001; `0x1e5f0`
//! reads manufacturer ID 0xfe, `0x1e410` writes configuration/limits,
//! `0x1e140` checks configuration readback at 3 after writing alias 9.
//! `0x1dd60` masks ALERT and `0x1de40` sets conversion rate.
//! The physical situation is a board at 32 C and an idle SoC at 35 C;
//! integer temperatures have zero fractional parts. No thermal alarm.
use super::i2c::RegisterMap;
use crate::snapshot::{SnapshotError, decode, encode};
pub const ADDRESS: u8 = 0x4c;
/// TI manufacturer identity (TMP451 datasheet); ptm reads but does not compare it.
const MANUFACTURER: u8 = 0x55;
pub struct Tmp451 {
    registers: [u8; 0x25],
}
impl Default for Tmp451 {
    fn default() -> Self {
        Self {
            registers: [0; 0x25],
        }
    }
}
impl RegisterMap for Tmp451 {
    fn read(&mut self, register: u8) -> Option<u8> {
        match register {
            0 | 1 => Some(
                (if register == 0 { 32 } else { 35 })
                    + if self.registers[3] & 4 != 0 { 64 } else { 0 },
            ),
            2 | 0x10 | 0x15 => Some(0),
            0xfe => Some(MANUFACTURER),
            3..=8 | 0x11..=0x14 | 0x19 | 0x20..=0x24 => Some(self.registers[register as usize]),
            _ => None,
        }
    }
    fn write(&mut self, register: u8, value: u8) -> Option<()> {
        let target = match register {
            9..=0xe => register - 6,
            0x11..=0x14 | 0x19 | 0x20..=0x24 => register,
            _ => return None,
        };
        self.registers[target as usize] = value;
        Some(())
    }
    /// The configuration and limit registers; the temperatures are constant.
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        encode(&self.registers.as_slice())
    }
    fn restore(&mut self, state: &[u8]) -> Result<(), SnapshotError> {
        let registers: Vec<u8> = decode("TMP451", state)?;
        self.registers = registers.try_into().map_err(|registers: Vec<u8>| {
            SnapshotError::Corrupt(format!("TMP451: {} registers saved", registers.len()))
        })?;
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn configuration_alias_and_extended_range() {
        let mut sensor = Tmp451::default();
        sensor.write(9, 0xc4).unwrap();
        assert_eq!(sensor.read(3), Some(0xc4));
        assert_eq!(sensor.read(0), Some(96));
        sensor.write(9, 0x80).unwrap();
        assert_eq!(sensor.read(1), Some(35));
    }
}
