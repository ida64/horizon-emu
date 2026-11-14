//! The on-board Bluetooth controller, on UART-D's lines.
//!
//! The port itself is a [`Serial`](super::serial::Serial); this is its
//! peer. H4 framing and the Command Complete layout follow public Linux
//! hci.h:
//! <https://github.com/torvalds/linux/blob/master/include/net/bluetooth/hci.h>.
//! The commands themselves are executed by the `hci` controller model; an
//! HCI command it does not know refuses the command's final byte rather
//! than claim success.

mod hci;

use super::serial::Peer;

pub const BASE: u64 = 0x7000_6300;
/// UARTD interrupt, public Tegra210 device tree SPI 90.
pub const INTID: u32 = 32 + 90;

/// The controller's end of the line: collects a command, has it executed
/// and answers with its completion event.
pub struct Line {
    command: [u8; 259],
    used: usize,
    controller: hci::Controller,
}

impl Default for Line {
    fn default() -> Self {
        Self {
            command: [0; 259],
            used: 0,
            controller: hci::Controller::default(),
        }
    }
}

impl Line {
    fn command_byte(&mut self, value: u8) -> Option<Vec<u8>> {
        if self.used == self.command.len() || (self.used == 0 && value != 1) {
            return None;
        }
        self.command[self.used] = value;
        self.used += 1;
        if self.used < 4 || self.used != 4 + usize::from(self.command[3]) {
            return Some(Vec::new());
        }
        let op = u16::from_le_bytes([self.command[1], self.command[2]]);
        let mut reply = [0u8; 252];
        let len = self
            .controller
            .execute(op, &self.command[4..self.used], &mut reply)?;
        // Command Complete, one command credit (Vol 4 Part E §7.7.14).
        let mut event = vec![4, 0x0e, (len + 3) as u8, 1, op as u8, (op >> 8) as u8];
        event.extend(&reply[..len]);
        self.used = 0;
        Some(event)
    }
}

impl Peer for Line {
    fn byte(&mut self, byte: u8) -> Result<Vec<u8>, String> {
        self.command_byte(byte).ok_or_else(|| {
            format!(
                "Bluetooth H4 command not modeled: {:02x?}",
                &self.command[..self.used]
            )
        })
    }

    /// CTS is asserted by the powered, present controller.
    fn clear_to_send(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{bus::MmioDevice, soc::serial::Serial};

    #[test]
    fn fragmented_reset_returns_exact_command_complete() {
        let mut line = Line::default();
        for byte in [1, 3, 12] {
            assert_eq!(line.byte(byte), Ok(Vec::new()));
        }
        assert_eq!(line.byte(0), Ok(vec![4, 14, 4, 1, 3, 12, 0]));
    }

    #[test]
    fn download_requires_minidriver_and_launch_exits_it() {
        let mut line = Line::default();
        for command in [
            &[1, 0x2e, 0xfc, 0][..],
            &[1, 0x4c, 0xfc, 5, 0, 0, 0, 0, 0xaa],
            &[1, 0x4e, 0xfc, 4, 0xff, 0xff, 0xff, 0xff],
        ] {
            let answer: Vec<u8> = command
                .iter()
                .flat_map(|&byte| line.byte(byte).unwrap())
                .collect();
            assert_eq!(answer.len(), 7);
        }
        for byte in [1, 0x4c, 0xfc, 5, 0, 0, 0, 0] {
            assert_eq!(line.byte(byte), Ok(Vec::new()));
        }
        assert!(line.byte(0).is_err());
    }

    #[test]
    fn fifo_dma_tail_then_eord_notifies_once() {
        let uart = Serial::new(Box::new(Line::default()), Box::new(|_| {}), Arc::default());
        uart.write(0, 8, 4, 0x41).unwrap();
        uart.write(0, 4, 4, 0x31).unwrap();
        for byte in [1, 3, 0x0c, 0] {
            uart.write(0, 0, 4, byte).unwrap();
        }
        assert_eq!(uart.read(0, 8, 4), Some(0xc4));
        uart.write(0, 4, 4, 0x30).unwrap();
        for _ in 0..4 {
            uart.read(0, 0, 4).unwrap();
        }
        assert_eq!(uart.read(0, 8, 4), Some(0xc8));
        uart.write(0, 4, 4, 0x10).unwrap();
        uart.write(0, 4, 4, 0x30).unwrap();
        assert_eq!(uart.read(0, 8, 4), Some(0xcc));
        for _ in 0..3 {
            uart.read(0, 0, 4).unwrap();
        }
        assert_eq!(uart.read(0, 8, 4), Some(0xc1));
    }
}
