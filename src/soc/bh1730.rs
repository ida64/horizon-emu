//! Rohm BH1730 ambient light sensor on I2C2, address 0x29.
//!
//! Public references consulted: hekate `bdk/input/als.{c,h}` for board
//! wiring, command bits and optical-window coefficients; Zephyr
//! `drivers/sensor/rohm/bh1730/bh1730.c` for identity and integration time.
//! <https://github.com/zephyrproject-rtos/zephyr/blob/main/drivers/sensor/rohm/bh1730/bh1730.c>
//! The room is a chosen dim 30 lux. Counts scale with programmed gain and
//! integration cycles; infrared is one quarter of the visible channel.
use super::{Notes, i2c::I2cDevice};
use crate::{
    clock::GuestClock,
    snapshot::{SnapshotError, decode, encode},
};
use serde::{Deserialize, Serialize};
use std::{sync::Arc, time::Duration};
pub const ADDRESS: u8 = 0x29;
/// Public Zephyr identity: part 7, revision 1 (revision is assumed).
const ID: u8 = 0x71;
const ROOM_LUX: u32 = 30;
pub struct Bh1730 {
    state: State,
    clock: Arc<GuestClock>,
    notes: Arc<Notes>,
}
/// What a snapshot saves: the registers and when the measurement they
/// started began, in guest time.
#[derive(Serialize, Deserialize)]
struct State {
    pointer: u8,
    control: u8,
    timing: u8,
    gain: u8,
    started: Duration,
}
impl State {
    fn new(now: Duration) -> Self {
        Self {
            pointer: 0,
            control: 0,
            timing: 0xda,
            gain: 0,
            started: now,
        }
    }
}
impl Bh1730 {
    /// Measurements take guest time from `clock`.
    pub fn new(clock: Arc<GuestClock>, notes: Arc<Notes>) -> Self {
        Self {
            state: State::new(clock.now()),
            clock,
            notes,
        }
    }
    fn ready(&self) -> bool {
        let s = &self.state;
        s.control & 3 == 3
            && self.clock.now().saturating_sub(s.started)
                >= Duration::from_micros((256 - s.timing as u64) * 2700)
    }
    fn register(&self, reg: u8) -> Option<u8> {
        let s = &self.state;
        match reg {
            0 => Some(s.control | if self.ready() { 0x10 } else { 0 }),
            1 => Some(s.timing),
            7 => Some(s.gain),
            0x12 => Some(ID),
            0x14..=0x17 => {
                // Invert the Icosa optical-window equation for IR/VIS=1/4:
                // lux ~= 416.8667 * visible / (gain * integration_cycles).
                let gain = [1, 2, 64, 128][s.gain as usize];
                let visible = if self.ready() {
                    (ROOM_LUX * gain * (256 - s.timing as u32) * 10 / 4169) as u16
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
        let now = self.clock.now();
        if data == [0xe4] || data == [0xe4, 0] {
            self.state = State::new(now);
            return Some(true);
        }
        let (&cmd, values) = data.split_first()?;
        if cmd & 0xe0 != 0x80 {
            self.notes
                .add(format!("BH1730: command {data:02x?} unsupported"));
            return None;
        }
        let s = &mut self.state;
        s.pointer = cmd & 31;
        for &value in values {
            match s.pointer {
                0 if value & !0x0f == 0 => {
                    s.control = value;
                    s.started = now;
                }
                1 => {
                    s.timing = value;
                    s.started = now;
                }
                7 if value < 4 => {
                    s.gain = value;
                    s.started = now;
                }
                _ => {
                    self.notes.add(format!(
                        "BH1730: write {:02x}={value:02x} unsupported",
                        s.pointer
                    ));
                    return None;
                }
            }
            s.pointer = s.pointer.wrapping_add(1);
        }
        Some(true)
    }
    fn read(&mut self, data: &mut [u8]) -> Option<bool> {
        for byte in data {
            *byte = match self.register(self.state.pointer) {
                Some(v) => v,
                None => {
                    self.notes.add(format!(
                        "BH1730: read {:02x} unsupported",
                        self.state.pointer
                    ));
                    return None;
                }
            };
            self.state.pointer = self.state.pointer.wrapping_add(1);
        }
        Some(true)
    }
    /// The registers and the guest time the measurement began.
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        encode(&self.state)
    }
    fn restore(&mut self, state: &[u8]) -> Result<(), SnapshotError> {
        self.state = decode("BH1730", state)?;
        Ok(())
    }
}
