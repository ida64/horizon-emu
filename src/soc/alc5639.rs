//! Realtek ALC5639 audio codec on I2C1, address 0x1C (bus and address as
//! in Atmosphère's I2C device table, `AudioCodec`).
//!
//! The codec has 16-bit registers behind 8-bit addresses: a write transfer
//! is the address and the value, most significant byte first; a read is a
//! write of the address and a two-byte read (`audio` text `0x1e370`,
//! `0x1b9c0`). Its private registers are reached through an index register
//! (`0x6A`) and a data register (`0x6C`) (`audio` text `0x1b9c0`, which
//! writes them for entries of kind 2 of its start-up table at `0x148a20`).
//!
//! `audio` resets the codec (writing register 0), then programs it from
//! that table and read-modify-writes power and mixer registers as it
//! routes and powers its outputs (text `0x1b490` onwards); it reads the
//! interrupt control register (`0xBE`, bit 3) and the GPIO and internal
//! status register (`0xBF`, bits 0 and 3) for faults. Register addresses,
//! names and reset values are those of the ALC5639's public Linux driver
//! (`rt5640.c`, `rt5640_reg`, which drives the RT5639/ALC5639 as well).
//!
//! The console's speakers are silent and nothing is wrong with them: the
//! status bits for over-current and over-temperature (`0xBE` bits 2-3,
//! `0xBF` bits 0 and 3) read clear, as do the jack and GPIO status bits of
//! `0xBF`. Configuration registers hold what was written. No sound comes
//! out. Registers the public driver does not list refuse, and so does its
//! device ID register (`0xFF`, listed with the RT5640's ID; the ALC5639's
//! own is not known here and `audio` never reads it). Three registers the
//! driver names (`rt5640.h`) but gives no reset value for, and that
//! `audio`'s table writes, are taken: `OUT_L2_MIXER` (`0x4E`) and the two
//! general control registers (`0xFA`, `0xFB`); they refuse to be read
//! before written.
use std::{collections::HashMap, sync::Arc};

use super::{Notes, i2c::I2cDevice};
use crate::snapshot::{SnapshotError, decode, encode};

pub const ADDRESS: u8 = 0x1C;

const RESET: u8 = 0x00;
const PRIVATE_INDEX: u8 = 0x6A;
const PRIVATE_DATA: u8 = 0x6C;
const IRQ_CONTROL_2: u8 = 0xBE;
/// `0xBE` bits 3 and 2: microphone bias over-current.
const OVER_CURRENT: u16 = 0b1100;
const INTERNAL_STATUS: u8 = 0xBF;
/// See the module documentation.
const NO_RESET_VALUE: [u8; 3] = [0x4E, 0xFA, 0xFB];

/// Reset values, from the public Linux `rt5640.c` (`rt5640_reg`).
const DEFAULTS: [(u8, u16); 102] = [
    (0x00, 0x000e),
    (0x01, 0xc8c8),
    (0x02, 0xc8c8),
    (0x03, 0xc8c8),
    (0x04, 0x8000),
    (0x0d, 0x0000),
    (0x0e, 0x0000),
    (0x0f, 0x0808),
    (0x19, 0xafaf),
    (0x1a, 0xafaf),
    (0x1b, 0x0000),
    (0x1c, 0x2f2f),
    (0x1d, 0x2f2f),
    (0x1e, 0x0000),
    (0x27, 0x7060),
    (0x28, 0x7070),
    (0x29, 0x8080),
    (0x2a, 0x5454),
    (0x2b, 0x5454),
    (0x2c, 0xaa00),
    (0x2d, 0x0000),
    (0x2e, 0xa000),
    (0x2f, 0x0000),
    (0x3b, 0x0000),
    (0x3c, 0x007f),
    (0x3d, 0x0000),
    (0x3e, 0x007f),
    (0x45, 0xe000),
    (0x46, 0x003e),
    (0x47, 0x003e),
    (0x48, 0xf800),
    (0x49, 0x3800),
    (0x4a, 0x0004),
    (0x4c, 0xfc00),
    (0x4d, 0x0000),
    (0x4f, 0x01ff),
    (0x50, 0x0000),
    (0x51, 0x0000),
    (0x52, 0x01ff),
    (0x53, 0xf000),
    (0x61, 0x0000),
    (0x62, 0x0000),
    (0x63, 0x00c0),
    (0x64, 0x0000),
    (0x65, 0x0000),
    (0x66, 0x0000),
    (0x6a, 0x0000),
    (0x6c, 0x0000),
    (0x70, 0x8000),
    (0x71, 0x8000),
    (0x72, 0x8000),
    (0x73, 0x1114),
    (0x74, 0x0c00),
    (0x75, 0x1d00),
    (0x80, 0x0000),
    (0x81, 0x0000),
    (0x82, 0x0000),
    (0x83, 0x0000),
    (0x84, 0x0000),
    (0x85, 0x0008),
    (0x89, 0x0000),
    (0x8a, 0x0000),
    (0x8b, 0x0600),
    (0x8c, 0x0228),
    (0x8d, 0xa000),
    (0x8e, 0x0004),
    (0x8f, 0x1100),
    (0x90, 0x0646),
    (0x91, 0x0c00),
    (0x92, 0x0000),
    (0x93, 0x3000),
    (0xb0, 0x2080),
    (0xb1, 0x0000),
    (0xb4, 0x2206),
    (0xb5, 0x1f00),
    (0xb6, 0x0000),
    (0xb8, 0x034b),
    (0xb9, 0x0066),
    (0xba, 0x000b),
    (0xbb, 0x0000),
    (0xbc, 0x0000),
    (0xbd, 0x0000),
    (0xbe, 0x0000),
    (0xbf, 0x0000),
    (0xc0, 0x0400),
    (0xc2, 0x0000),
    (0xc4, 0x0000),
    (0xc5, 0x0000),
    (0xc6, 0x2000),
    (0xc8, 0x0000),
    (0xc9, 0x0000),
    (0xca, 0x0000),
    (0xcb, 0x0000),
    (0xcc, 0x0000),
    (0xcf, 0x0013),
    (0xd0, 0x0680),
    (0xd1, 0x1c17),
    (0xd2, 0x8c00),
    (0xd3, 0xaa20),
    (0xd6, 0x0400),
    (0xd9, 0x0809),
    (0xfe, 0x10ec),
];

pub struct Alc5639 {
    registers: HashMap<u8, u16>,
    /// Private registers written so far: their reset values are not public.
    private: HashMap<u16, u16>,
    /// The register a read transfer returns.
    pointer: u8,
    notes: Arc<Notes>,
}

impl Alc5639 {
    pub fn new(notes: Arc<Notes>) -> Self {
        Self {
            registers: DEFAULTS.into_iter().collect(),
            private: HashMap::new(),
            pointer: 0,
            notes,
        }
    }

    fn refuse(&self, what: String) -> Option<bool> {
        self.notes.add(format!("ALC5639 audio codec: {what}"));
        None
    }

    fn register(&self, register: u8) -> Option<u16> {
        match register {
            INTERNAL_STATUS => Some(0),
            IRQ_CONTROL_2 => Some(self.registers[&register] & !OVER_CURRENT),
            PRIVATE_DATA => self.private.get(&self.registers[&PRIVATE_INDEX]).copied(),
            _ => self.registers.get(&register).copied(),
        }
    }
}

impl I2cDevice for Alc5639 {
    fn write(&mut self, data: &[u8]) -> Option<bool> {
        let [register, rest @ ..] = data else {
            return self.refuse("an empty write".into());
        };
        self.pointer = *register;
        let value = match rest {
            [] => return Some(true),
            [high, low] => u16::from_be_bytes([*high, *low]),
            _ => return self.refuse(format!("a write of {} bytes", data.len())),
        };
        match *register {
            RESET => {
                self.registers = DEFAULTS.into_iter().collect();
                self.private.clear();
            }
            PRIVATE_DATA => {
                self.private.insert(self.registers[&PRIVATE_INDEX], value);
            }
            INTERNAL_STATUS => return self.refuse("a write of the status register".into()),
            register
                if self.registers.contains_key(&register) || NO_RESET_VALUE.contains(&register) =>
            {
                self.registers.insert(register, value);
            }
            register => return self.refuse(format!("register {register:#04x} is not modeled")),
        }
        Some(true)
    }

    fn read(&mut self, data: &mut [u8]) -> Option<bool> {
        let Some(value) = self.register(self.pointer) else {
            return self.refuse(format!("register {:#04x} has no known value", self.pointer));
        };
        match data {
            [high, low] => [*high, *low] = value.to_be_bytes(),
            _ => return self.refuse(format!("a read of {} bytes", data.len())),
        }
        Some(true)
    }

    /// The registers, the private registers written and the pointer.
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        encode(&(&self.registers, &self.private, self.pointer))
    }

    fn restore(&mut self, state: &[u8]) -> Result<(), SnapshotError> {
        (self.registers, self.private, self.pointer) = decode("ALC5639", state)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(codec: &mut Alc5639, register: u8) -> Option<u16> {
        codec.write(&[register])?;
        let mut bytes = [0; 2];
        codec.read(&mut bytes)?;
        Some(u16::from_be_bytes(bytes))
    }

    #[test]
    fn identifies_and_follows_audio_start_up() {
        let mut codec = Alc5639::new(Arc::new(Notes::default()));
        assert_eq!(read(&mut codec, 0xFE), Some(0x10EC), "Realtek's vendor ID");
        codec.write(&[RESET, 0, 0]).unwrap();
        // Private register 0x3D through the index and data pair.
        codec.write(&[PRIVATE_INDEX, 0x00, 0x3D]).unwrap();
        codec.write(&[PRIVATE_DATA, 0x36, 0x00]).unwrap();
        assert_eq!(read(&mut codec, PRIVATE_DATA), Some(0x3600));
        // Power: read-modify-write of PWR_ANLG1, as text 0x1b490 does.
        let power = read(&mut codec, 0x63).unwrap();
        codec
            .write(&[0x63, ((power | 0x4008) >> 8) as u8, power as u8 | 8])
            .unwrap();
        assert_eq!(read(&mut codec, 0x63), Some(0x40C8));
        // No faults.
        codec.write(&[IRQ_CONTROL_2, 0x80, 0x0C]).unwrap();
        assert_eq!(read(&mut codec, IRQ_CONTROL_2), Some(0x8000));
        assert_eq!(read(&mut codec, INTERNAL_STATUS), Some(0));
        assert_eq!(read(&mut codec, 0xFA), None, "unknown until written");
        codec.write(&[0xFA, 0x3E, 0x01]).unwrap();
        assert_eq!(read(&mut codec, 0xFA), Some(0x3E01));
        assert_eq!(codec.write(&[0x42, 0, 0]), None);
    }
}
