//! Console LSM6DS3H register protocol, behind SPI4 chip-select zero.
//!
//! hid 23.5.0 text 0xe7ae0 reads WHO_AM_I with opcode 0x8f, accepting
//! 0x69, 0x6a or 0x6c. 0xe7d80/0xe7e80 configure CTRL1_XL/CTRL2_G;
//! 0xe7f80 programs the filters. Public register definitions consulted:
//! <https://github.com/STMicroelectronics/lsm6ds3-pid/blob/master/lsm6ds3_reg.h>.
//! Samples are read as one 12-byte burst from OUTX_L_G (0x22-0x2d): hid
//! 0xe7950 has Bus repeat that read periodically (sasbus periodic receive)
//! while a client uses the console's six-axis sensor.
//! The console rests screen-up: +Z gravity, no angular velocity. Ideal
//! scaling matches `nand::calibration`; no factory trim is invented.

/// WHO_AM_I of the original LSM6DS3 family in ST's public header, accepted
/// by hid 0xe7ae0 for the calibration module type LSM6DS3H.
const ID: u8 = 0x69;

/// The registers and the SPI command in progress: what the SPI controller
/// saves of the sensor in a snapshot.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Lsm6ds3 {
    registers: [u8; 0x20],
    command: Option<u8>,
}
/// hid probes alternate ICM parts at WHO_AM_I 0x75 before the ST part.
/// This address is reserved on LSM6DS3: assume zero for this one probe,
/// not an ICM identity. No documented readable silicon value is available.
const ALTERNATE_ID_PROBE: u8 = 0;
impl Default for Lsm6ds3 {
    fn default() -> Self {
        let mut registers = [0; 0x20];
        // CTRL3_C reset: IF_INC enables consecutive-register transfers.
        registers[0x12] = 4;
        Self {
            registers,
            command: None,
        }
    }
}
impl Lsm6ds3 {
    pub fn deselect(&mut self) {
        self.command = None;
    }
    pub fn exchange(&mut self, byte: u8) -> Option<u8> {
        let Some(command) = self.command else {
            self.command = Some(byte);
            return Some(0);
        };
        let address = command & 0x7f;
        let result = if command & 0x80 != 0 {
            self.read(address)?
        } else {
            self.write(address, byte)?;
            0
        };
        if self.registers[0x12] & 4 != 0 {
            self.command = Some((command & 0x80) | address.wrapping_add(1));
        }
        Some(result)
    }
    pub fn read(&self, register: u8) -> Option<u8> {
        match register {
            0x0f => Some(ID),
            0x01 | 0x04 | 0x06..=0x0b | 0x0d..=0x0e | 0x10..=0x19 => {
                Some(self.registers[register as usize])
            }
            // No wake-up, free-fall, taps or orientation events at rest.
            0x1b..=0x1d => Some(0),
            0x1e => Some(
                u8::from(self.registers[0x10] & 0xf0 != 0)
                    | (u8::from(self.registers[0x11] & 0xf0 != 0) << 1),
            ),
            0x75 => Some(ALTERNATE_ID_PROBE),
            // Temperature offset zero means 25 C; rotation and X/Y gravity zero.
            0x20..=0x2b => Some(0),
            0x2c..=0x2d => {
                let scale = match (self.registers[0x10] >> 2) & 3 {
                    0 => 16384i16,
                    1 => 2048,
                    2 => 8192,
                    _ => 4096,
                };
                Some(scale.to_le_bytes()[(register - 0x2c) as usize])
            }
            // FIFO disabled: zero unread words, empty flag set.
            0x3a | 0x3c | 0x3d => Some(0),
            0x3b => Some(0x10),
            _ => None,
        }
    }
    pub fn write(&mut self, register: u8, value: u8) -> Option<()> {
        match register {
            0x12 if value & 1 != 0 => {
                let command = self.command;
                *self = Self::default();
                self.command = command;
            }
            0x12 => self.registers[0x12] = value & !0x80,
            0x01 if value & 0x80 != 0 => return None, // Embedded-function bank not modeled.
            0x0a if value & 7 != 0 => return None,    // FIFO mode needs timed production.
            0x01 | 0x04 | 0x06..=0x0b | 0x0d..=0x0e | 0x10..=0x11 | 0x13..=0x19 => {
                self.registers[register as usize] = value
            }
            _ => return None,
        }
        Some(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn spi_command_and_autoincrement() {
        let mut s = Lsm6ds3::default();
        assert_eq!(s.exchange(0x8f), Some(0));
        assert_eq!(s.exchange(0), Some(ID));
        s.deselect();
        s.exchange(0x10);
        s.exchange(0x38);
        s.exchange(0x40);
        assert_eq!(s.read(0x10), Some(0x38));
        assert_eq!(s.read(0x11), Some(0x40));
        assert_eq!(s.read(0x1e), Some(3));
        assert_eq!(s.read(0x2d), Some(0x20));
        s.write(0x12, 1);
        assert_eq!(s.read(0x10), Some(0));
        assert_eq!(s.read(0x12), Some(4));
        assert_eq!(s.read(0x7f), None);
    }
    /// hid samples by starting sasbus periodic receive of one 12-byte burst
    /// from OUTX_L_G (opcode 0xa2, text 0xe7950 via 0x195fe0, with transfer
    /// memory), which Bus then repeats at the interval hid asks for.
    #[test]
    fn burst_samples_rest_flat() {
        let mut s = Lsm6ds3::default();
        // CTRL1_XL: 104 Hz, +-8 g; CTRL2_G: 104 Hz.
        for (register, value) in [(0x10, 0x4c), (0x11, 0x40)] {
            s.exchange(register);
            s.exchange(value);
            s.deselect();
        }
        for _ in 0..3 {
            s.exchange(0xa2);
            let sample: Vec<u8> = (0..12).map(|_| s.exchange(0).unwrap()).collect();
            s.deselect();
            let words: Vec<i16> = sample
                .chunks(2)
                .map(|word| i16::from_le_bytes([word[0], word[1]]))
                .collect();
            assert_eq!(words, [0, 0, 0, 0, 0, 4096]);
        }
    }
}
