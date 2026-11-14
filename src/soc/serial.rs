//! A Tegra UART in FIFO mode, and what is on the other end of its lines.
//!
//! The registers are the 16550's at a 4-byte stride, with Tegra's
//! additions, as the firmware's driver uses them: Bus text `0x1d850`
//! programs them; `0x1dd60` tests LSR bit 8 (transmit FIFO full),
//! `0x1dee0`/`0x1e050` bit 9 (receive FIFO empty), and `0x1db70` waits for
//! TEMT, bit 6. The receiver (`0x1e050`) takes IIR 4 as "serve the FIFO or
//! the DMA" and IIR 8 as "end of received data", which sets the
//! notification flag at driver object +63; clearing IER bit 5 acknowledges
//! that. Reporting only a timeout (IIR `0xc`) delivers bytes but does not
//! notify a waiting client promptly.
//!
//! Bytes cross the line in no time, at whatever rate the divisor says: the
//! [`Peer`] has each byte as it is written and its answer is in the receive
//! FIFO when the write returns. The end of an answer latches "end of
//! received data", after FIFO service in priority.

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use super::Notes;
use crate::bus::MmioDevice;

/// Register block of one port.
pub const SIZE: u64 = 0x40;

/// What a UART's lines lead to.
pub trait Peer: Send {
    /// A byte arrives from the console. `Ok` is what the peer sends back
    /// because of it, often nothing; `Err` says what it was given that is
    /// not modeled.
    fn byte(&mut self, byte: u8) -> Result<Vec<u8>, String>;

    /// Whether the peer holds the console's CTS input asserted.
    fn clear_to_send(&self) -> bool;
}

struct State {
    regs: [u32; 11],
    divisor: [u8; 2],
    receive: VecDeque<u8>,
    /// End of received data, until acknowledged.
    eord: bool,
    peer: Box<dyn Peer>,
}

impl State {
    fn iir(&self) -> u32 {
        let fifo = if self.regs[2] & 1 != 0 { 0xc0 } else { 0 };
        let threshold = [1, 4, 8, 14][((self.regs[2] >> 6) & 3) as usize];
        fifo | if self.regs[1] & 1 != 0 && self.receive.len() >= threshold {
            4
        } else if self.regs[1] & 0x20 != 0 && self.eord {
            8
        } else if self.regs[1] & 0x11 != 0 && !self.receive.is_empty() {
            0xc
        } else if self.regs[1] & 2 != 0 {
            2
        } else {
            1
        }
    }
}

pub struct Serial {
    state: Mutex<State>,
    interrupt: Box<dyn Fn(bool) + Send + Sync>,
    notes: Arc<Notes>,
}

impl Serial {
    pub fn new(
        peer: Box<dyn Peer>,
        interrupt: Box<dyn Fn(bool) + Send + Sync>,
        notes: Arc<Notes>,
    ) -> Self {
        Self {
            state: Mutex::new(State {
                regs: [0; 11],
                divisor: [0; 2],
                receive: VecDeque::with_capacity(1024),
                eord: false,
                peer,
            }),
            interrupt,
            notes,
        }
    }

    /// Bytes waiting in the receive FIFO.
    pub(crate) fn receive_len(&self) -> usize {
        self.state.lock().unwrap().receive.len()
    }

    fn update(&self, state: &State) {
        (self.interrupt)(state.iir() & 1 == 0);
    }
}

impl MmioDevice for Serial {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        if size != 4 || !offset.is_multiple_of(4) {
            return None;
        }
        let mut s = self.state.lock().unwrap();
        let value = match offset {
            0 | 4 if s.regs[3] & 0x80 != 0 => u32::from(s.divisor[(offset / 4) as usize]),
            0 => u32::from(s.receive.pop_front()?),
            4 | 12 | 16 | 28 | 32 | 40 => s.regs[(offset / 4) as usize],
            8 => s.iir(),
            20 => 0x60 | if s.receive.is_empty() { 0x200 } else { 1 },
            24 => u32::from(s.peer.clear_to_send()) << 4,
            _ => return None,
        };
        self.update(&s);
        Some(u64::from(value))
    }

    fn write(&self, _core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        if size != 4 || !offset.is_multiple_of(4) {
            return None;
        }
        let mut s = self.state.lock().unwrap();
        match offset {
            0 | 4 if s.regs[3] & 0x80 != 0 => s.divisor[(offset / 4) as usize] = value as u8,
            0 => match s.peer.byte(value as u8) {
                Ok(answer) => {
                    s.eord |= !answer.is_empty();
                    s.receive.extend(answer);
                }
                Err(what) => {
                    self.notes.add(what);
                    return None;
                }
            },
            8 => {
                s.regs[2] = value as u32 & !6;
                if value & 2 != 0 {
                    s.receive.clear();
                    s.eord = false;
                }
            }
            4 => {
                if value & 0x20 == 0 {
                    s.eord = false;
                }
                s.regs[1] = value as u32;
            }
            12 | 16 | 28 | 32 | 40 => s.regs[(offset / 4) as usize] = value as u32,
            _ => return None,
        }
        self.update(&s);
        Some(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Answers every byte with itself and the next number.
    pub(crate) struct Echo;

    impl Peer for Echo {
        fn byte(&mut self, byte: u8) -> Result<Vec<u8>, String> {
            match byte {
                0xFF => Err("the echo was given 0xff".into()),
                byte => Ok(vec![byte, byte + 1]),
            }
        }

        fn clear_to_send(&self) -> bool {
            true
        }
    }

    fn port() -> Serial {
        Serial::new(Box::new(Echo), Box::new(|_| {}), Arc::default())
    }

    #[test]
    fn divisor_access_does_not_transmit() {
        let uart = port();
        uart.write(0, 12, 4, 0x80).unwrap();
        uart.write(0, 0, 4, 1).unwrap();
        assert_eq!(uart.read(0, 0, 4), Some(1));
        uart.write(0, 12, 4, 3).unwrap();
        assert_eq!(uart.read(0, 20, 4), Some(0x260));
    }

    #[test]
    fn an_answer_is_served_from_the_fifo_and_then_ends() {
        let uart = port();
        uart.write(0, 8, 4, 0x41).unwrap();
        uart.write(0, 4, 4, 0x31).unwrap();
        for byte in [1, 3] {
            uart.write(0, 0, 4, byte).unwrap();
        }
        // Four bytes: the FIFO threshold first.
        assert_eq!(uart.read(0, 8, 4), Some(0xc4));
        assert_eq!(uart.read(0, 24, 4), Some(0x10));
        for expected in [1, 2, 3, 4] {
            assert_eq!(uart.read(0, 0, 4), Some(expected));
        }
        // Then the end of the data, until it is acknowledged.
        assert_eq!(uart.read(0, 8, 4), Some(0xc8));
        uart.write(0, 4, 4, 0x10).unwrap();
        uart.write(0, 4, 4, 0x30).unwrap();
        assert_eq!(uart.read(0, 8, 4), Some(0xc1));
    }

    #[test]
    fn what_the_peer_does_not_take_is_refused_with_its_words() {
        let notes = Arc::new(Notes::default());
        let uart = Serial::new(Box::new(Echo), Box::new(|_| {}), Arc::clone(&notes));
        assert_eq!(uart.write(0, 0, 4, 0xFF), None);
        assert_eq!(uart.receive_len(), 0);
    }
}
