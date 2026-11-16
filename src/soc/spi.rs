//! Tegra210 SPI4 with the board's console motion sensor on CS0.
//!
//! Public register meanings and FIFO depth from Linux `spi-tegra114.c`:
//! <https://github.com/torvalds/linux/blob/master/drivers/spi/spi-tegra114.c>.
//! Transfers complete synchronously; RDY and FIFO occupancy reflect actual
//! transferred bytes, never a fabricated all-ready register value.
use super::{Notes, lsm6ds3::Lsm6ds3};
use crate::{
    bus::MmioDevice,
    snapshot::{Snapshot, SnapshotError, decode, encode},
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
pub const BASE: u64 = 0x7000_da00;
pub const SIZE: u64 = 0x200;
/// Linux Tegra210 DTS SPI4 shared interrupt 93.
pub const INTID: u32 = 32 + 93;

pub struct Spi {
    state: Mutex<State>,
    interrupt: Box<dyn Fn(bool) + Send + Sync>,
    notes: Arc<Notes>,
}
/// Everything a snapshot saves: registers, both FIFOs and the sensor.
#[derive(Serialize, Deserialize)]
struct State {
    configuration: [u32; 10],
    #[serde(with = "crate::snapshot::array")]
    tx: [u32; 64],
    tx_len: usize,
    #[serde(with = "crate::snapshot::array")]
    rx: [u32; 64],
    rx_len: usize,
    rx_pos: usize,
    ready: bool,
    interrupt_mask: u32,
    sensor: Lsm6ds3,
}
impl Spi {
    pub fn new(interrupt: Box<dyn Fn(bool) + Send + Sync>, notes: Arc<Notes>) -> Self {
        Self {
            state: Mutex::new(State {
                configuration: [0; 10],
                tx: [0; 64],
                tx_len: 0,
                rx: [0; 64],
                rx_len: 0,
                rx_pos: 0,
                ready: false,
                interrupt_mask: 0,
                sensor: Lsm6ds3::default(),
            }),
            interrupt,
            notes,
        }
    }
    fn update(&self, s: &State) {
        (self.interrupt)(s.ready && s.configuration[8] & 0x3000_0000 != 0);
    }
    fn transfer(&self, s: &mut State) -> Option<()> {
        let command = s.configuration[0];
        let bits = (command & 31) + 1;
        let count = (s.configuration[9] & 0xffff) as usize + 1;
        let packed = command & 32 != 0;
        let transmit = command & (1 << 11) != 0;
        let receive = command & (1 << 12) != 0;
        if bits != 8 || count > 256 || command & (3 << 26) != 0 {
            return None;
        }
        let words = if packed { count.div_ceil(4) } else { count };
        if words > 64 || (transmit && words > s.tx_len) {
            return None;
        }
        s.rx_len = 0;
        s.rx_pos = 0;
        s.rx.fill(0);
        for index in 0..count {
            let slot = if packed { index / 4 } else { index };
            let shift = if packed { 8 * (index % 4) } else { 0 };
            let byte = if transmit {
                (s.tx[slot] >> shift) as u8
            } else {
                0
            };
            let result = s.sensor.exchange(byte)?;
            if receive {
                s.rx[slot] |= (result as u32) << shift;
            }
        }
        s.tx_len = 0;
        if receive {
            s.rx_len = words;
        }
        s.ready = true;
        s.configuration[0] &= !(1 << 31);
        // Hardware CS deasserts at the end; software CS can span transfers.
        if command & (1 << 21) == 0 {
            s.sensor.deselect();
        }
        Some(())
    }
}
impl MmioDevice for Spi {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        if size != 4 {
            return None;
        }
        let mut s = self.state.lock().unwrap();
        let value = match offset {
            0 | 4 | 8 | 12 | 0x18 | 0x20 | 0x24 => s.configuration[(offset / 4) as usize],
            0x10 => {
                if s.ready {
                    1 << 30
                } else {
                    0
                }
            }
            0x14 => {
                let rx = s.rx_len - s.rx_pos;
                u32::from(rx == 0)
                    | (u32::from(rx == 64) << 1)
                    | (u32::from(s.tx_len == 0) << 2)
                    | (u32::from(s.tx_len == 64) << 3)
                    | (((64 - s.tx_len) as u32) << 16)
                    | ((rx as u32) << 23)
            }
            0x188 | 0x1c => {
                if s.rx_pos == s.rx_len {
                    return None;
                }
                let word = s.rx[s.rx_pos];
                s.rx_pos += 1;
                word
            }
            0x18c => s.interrupt_mask,
            _ => return None,
        };
        Some(value as u64)
    }
    fn write(&self, _core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        if size != 4 {
            return None;
        }
        let value = value as u32;
        let mut s = self.state.lock().unwrap();
        match offset {
            0 => {
                s.configuration[0] = value;
                if value & (1 << 20) != 0 || value & ((1 << 21) | (1 << 31)) == 0 {
                    s.sensor.deselect();
                }
                if value & (1 << 31) != 0 && self.transfer(&mut s).is_none() {
                    self.notes.add(format!(
                        "SPI4: unsupported transfer command {value:#010x}, block {}, TX {:08x?}",
                        s.configuration[9],
                        &s.tx[..s.tx_len]
                    ));
                    return None;
                }
            }
            4 | 8 | 12 | 0x18 | 0x24 => s.configuration[(offset / 4) as usize] = value,
            0x20 => {
                if value & (1 << 31) != 0 {
                    return None;
                }
                s.configuration[8] = value;
            }
            0x10 => {
                if value & (1 << 30) != 0 {
                    s.ready = false;
                }
            }
            0x14 => {
                if value & (1 << 14) != 0 {
                    s.tx_len = 0;
                }
                if value & (1 << 15) != 0 {
                    s.rx_len = 0;
                    s.rx_pos = 0;
                }
            }
            0x108 => {
                if s.tx_len == 64 {
                    return None;
                }
                let n = s.tx_len;
                s.tx[n] = value;
                s.tx_len += 1;
            }
            0x18c => s.interrupt_mask = value,
            _ => return None,
        }
        self.update(&s);
        Some(())
    }
    fn snapshot(&self) -> Option<&dyn Snapshot> {
        Some(self)
    }
}
/// The interrupt output follows from the registers at the next access and
/// is not driven on restore.
impl Snapshot for Spi {
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        encode(&*self.state.lock().unwrap())
    }
    fn restore(&self, state: &[u8]) -> Result<(), SnapshotError> {
        let saved: State = decode("SPI4", state)?;
        if saved.tx_len > saved.tx.len()
            || saved.rx_pos > saved.rx_len
            || saved.rx_len > saved.rx.len()
        {
            return Err(SnapshotError::Corrupt(
                "SPI4: FIFO counts out of range".into(),
            ));
        }
        *self.state.lock().unwrap() = saved;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn split_opcode_read_and_chip_select_reset() {
        let irq = Arc::new(AtomicBool::new(false));
        let line = Arc::clone(&irq);
        let spi = Spi::new(
            Box::new(move |v| line.store(v, Ordering::SeqCst)),
            Arc::default(),
        );
        for opcode in [0xf5, 0x8f] {
            // Bus configures hardware-CS idle, then holds CS0 low across
            // the one-byte opcode and the receive-only PIO transfer.
            spi.write(0, 0, 4, 0x4000_8000).unwrap();
            spi.write(0, 0, 4, 0x4060_0007).unwrap();
            spi.write(0, 0x108, 4, opcode).unwrap();
            spi.write(0, 0x24, 4, 0).unwrap();
            spi.write(0, 0x20, 4, 0x1000_0000).unwrap();
            spi.write(0, 0, 4, 0xc060_0807).unwrap();
            assert!(irq.load(Ordering::SeqCst));
            assert_eq!(spi.read(0, 0x10, 4), Some(1 << 30));
            spi.write(0, 0x10, 4, 1 << 30).unwrap();
            assert!(!irq.load(Ordering::SeqCst));
            spi.write(0, 0x20, 4, 0x2000_0000).unwrap();
            spi.write(0, 0, 4, 0xc060_1007).unwrap();
            assert_eq!(
                spi.read(0, 0x188, 4),
                Some(if opcode == 0x8f { 0x69 } else { 0 })
            );
            assert_eq!(spi.read(0, 0x14, 4), Some(0x0040_0005));
            spi.write(0, 0x10, 4, 1 << 30).unwrap();
        }
    }
}
