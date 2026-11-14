//! USB-C PD controller, I2C1:0x18 (usb text `0x567b0`).
//!
//! usb `0x54ce0` uses little-endian words and length-prefixed long reads.
//! `0x555e0` requires nonzero revision 0x4e; assumed revision 1 is not a
//! physical chip dump. `0x519a0` tests status bit 7 for attachment: clear
//! with no cable. `0x556c0` pulses reset then waits for active-low IRQ;
//! observed GPIO traffic identifies reset V5 and IRQ K4. `0x55670`
//! issues command 0x0d0d; `0x554d0` requires accepted bit 13 of status 3
//! and `0x54830` waits for command-complete interrupt bit 2. Reading
//! interrupt status 2 acknowledges it (`0x51250`). No PD negotiation is
//! possible without a cable; unknown commands/registers refuse.
use super::{gpio::Gpio, i2c::I2cDevice};
use std::sync::{Arc, Mutex};
pub const ADDRESS: u8 = 0x18;
#[derive(Default)]
struct State {
    pointer: u8,
    interrupt: u16,
    status: u16,
    interrupt_mask: u16,
    configuration: u16,
    mode: u16,
}
pub struct M92t36 {
    state: Mutex<State>,
    gpio: std::sync::Weak<Gpio>,
    notes: Arc<super::Notes>,
}
impl M92t36 {
    pub fn new(gpio: &Arc<Gpio>, notes: Arc<super::Notes>) -> Arc<Self> {
        let device = Arc::new(Self {
            state: Mutex::new(State::default()),
            gpio: Arc::downgrade(gpio),
            notes,
        });
        let weak = Arc::downgrade(&device);
        gpio.observe_output(Box::new(move |port, pin, high| {
            if port == 21 && pin == 5 && !high
                && let Some(device) = weak.upgrade()
            {
                let mut state = device.state.lock().unwrap();
                state.interrupt = 4;
                state.status = 0;
                device.irq(true);
            }
        }));
        device
    }
    fn irq(&self, asserted: bool) {
        if let Some(gpio) = self.gpio.upgrade() {
            gpio.set_input(10, 4, !asserted);
        }
    }
}
impl I2cDevice for Arc<M92t36> {
    fn write(&mut self, data: &[u8]) -> Option<bool> {
        let mut state = self.state.lock().unwrap();
        state.pointer = *data.first()?;
        if data.len() == 1 {
            return Some(true);
        }
        if data.len() != 3 {
            return None;
        }
        let value = u16::from_le_bytes([data[1], data[2]]);
        match state.pointer {
            5 if value == 0x0d0d => {
                state.status = 1 << 13;
                state.interrupt |= 4;
                self.irq(true);
            }
            // usb 0x538c0 retains mode bits 15:14 through register 6.
            6 => state.mode = value,
            0x19 => state.interrupt_mask = value,
            0x1a => state.configuration = value,
            _ => return None,
        }
        Some(true)
    }
    fn read(&mut self, data: &mut [u8]) -> Option<bool> {
        let mut state = self.state.lock().unwrap();
        let value: u16 = match state.pointer {
            2 if data.len() == 2 => {
                let value = state.interrupt;
                state.interrupt = 0;
                self.irq(false);
                value
            }
            3 => state.status,
            6 => state.mode,
            4 | 0x18 => 0,
            0x19 => state.interrupt_mask,
            0x1a => state.configuration,
            0x4b..=0x4e => 1,
            0x28 | 0x2b if data.len() == 5 => {
                data.copy_from_slice(&[4, 0, 0, 0, 0]);
                return Some(true);
            }
            _ => {
                self.notes.add(format!(
                    "M92T36: read register {:#x}, length {}",
                    state.pointer,
                    data.len()
                ));
                return None;
            }
        };
        if data.len() != 2 {
            return None;
        }
        data.copy_from_slice(&value.to_le_bytes());
        Some(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bus::MmioDevice;
    #[test]
    fn command_interrupt_is_acknowledged_by_status_read() {
        let gpio = Arc::new(Gpio::new([(10, 0x10)]));
        let mut device = M92t36::new(&gpio, Arc::default());
        device.write(&[5, 0x0d, 0x0d]).unwrap();
        assert_eq!(gpio.read(0, 0x238, 4), Some(0));
        let mut status = [0; 2];
        device.write(&[3]).unwrap();
        device.read(&mut status).unwrap();
        assert_eq!(u16::from_le_bytes(status), 1 << 13);
        device.write(&[2]).unwrap();
        device.read(&mut status).unwrap();
        assert_eq!(u16::from_le_bytes(status), 4);
        assert_eq!(gpio.read(0, 0x238, 4), Some(0x10));
        device.read(&mut status).unwrap();
        assert_eq!(status, [0; 2]);
        assert_eq!(device.write(&[5, 0xff, 0xff]), None);
    }
}
