//! Rohm BH1730 ambient light sensor on I2C2, address 0x29.
//!
//! Public references consulted: hekate `bdk/input/als.{c,h}` for board
//! wiring, command bits and optical-window coefficients; Zephyr
//! `drivers/sensor/rohm/bh1730/bh1730.c` for identity and integration time.
//! <https://github.com/zephyrproject-rtos/zephyr/blob/main/drivers/sensor/rohm/bh1730/bh1730.c>
//! The room is a chosen dim 30 lux. Counts scale with programmed gain and
//! integration cycles; infrared is one quarter of the visible channel.
use super::{Notes, i2c::I2cDevice};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
pub const ADDRESS: u8 = 0x29;
/// Public Zephyr identity: part 7, revision 1 (revision is assumed).
const ID: u8 = 0x71;
const ROOM_LUX: u32 = 30;
pub struct Bh1730 {
    pointer: u8,
    control: u8,
    timing: u8,
    gain: u8,
    started: Instant,
    notes: Arc<Notes>,
}
impl Bh1730 {
    pub fn new(notes: Arc<Notes>) -> Self {
        Self {
            pointer: 0,
            control: 0,
            timing: 0xda,
            gain: 0,
            started: Instant::now(),
            notes,
        }
    }
    fn ready(&self) -> bool {
        self.control & 3 == 3
            && self.started.elapsed() >= Duration::from_micros((256 - self.timing as u64) * 2700)
    }
    fn register(&self, reg: u8) -> Option<u8> {
        match reg {
            0 => Some(self.control | if self.ready() { 0x10 } else { 0 }),
            1 => Some(self.timing),
            7 => Some(self.gain),
            0x12 => Some(ID),
            0x14..=0x17 => {
                // Invert the Icosa optical-window equation for IR/VIS=1/4:
                // lux ~= 416.8667 * visible / (gain * integration_cycles).
                let gain = [1, 2, 64, 128][self.gain as usize];
                let visible = if self.ready() {
                    (ROOM_LUX * gain * (256 - self.timing as u32) * 10 / 4169) as u16
                } else {
                    0
                };
                let count = if reg < 0x16 { visible } else { visible / 4 };
                Some(count.to_le_bytes()[(reg & 1) as usize])
            }
            _ => None,
        }
    }
}
impl I2cDevice for Bh1730 {
    fn write(&mut self, data: &[u8]) -> Option<bool> {
        if data == [0xe4] || data == [0xe4, 0] {
            let notes = self.notes.clone();
            *self = Self::new(notes);
            return Some(true);
        }
        let (&cmd, values) = data.split_first()?;
        if cmd & 0xe0 != 0x80 {
            self.notes
                .add(format!("BH1730: command {data:02x?} unsupported"));
            return None;
        }
        self.pointer = cmd & 31;
        for &value in values {
            match self.pointer {
                0 if value & !0x0f == 0 => {
                    self.control = value;
                    self.started = Instant::now();
                }
                1 => {
                    self.timing = value;
                    self.started = Instant::now();
                }
                7 if value < 4 => {
                    self.gain = value;
                    self.started = Instant::now();
                }
                _ => {
                    self.notes.add(format!(
                        "BH1730: write {:02x}={value:02x} unsupported",
                        self.pointer
                    ));
                    return None;
                }
            }
            self.pointer = self.pointer.wrapping_add(1);
        }
        Some(true)
    }
    fn read(&mut self, data: &mut [u8]) -> Option<bool> {
        for byte in data {
            *byte = match self.register(self.pointer) {
                Some(v) => v,
                None => {
                    self.notes
                        .add(format!("BH1730: read {:02x} unsupported", self.pointer));
                    return None;
                }
            };
            self.pointer = self.pointer.wrapping_add(1);
        }
        Some(true)
    }
}
