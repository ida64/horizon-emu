//! Tegra APB DMA, for the one-shot transfers of the UARTs.
//!
//! Bus text 0x20d50 programs global enable and IRQ masks, 0x20e50 creates
//! channels at +0x1000 + n*0x40, 0x211f0 starts them, 0x21110 stores byte
//! count minus four, 0x21330 tests EOC bit30. Public register definitions:
//! <https://github.com/torvalds/linux/blob/master/drivers/dma/tegra20-apb-dma.c>.
//! Only one-shot transfers to and from the UARTs given are supported, and
//! completion follows actual data movement. Other peripheral addresses or
//! modes refuse, never complete.
use super::{Notes, mc::DeviceMemory, regs::word, serial::Serial};
use crate::{
    bus::MmioDevice,
    snapshot::{Snapshot, SnapshotError, decode, encode},
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
pub const BASE: u64 = 0x6002_0000;
pub const SIZE: u64 = 0x2000;
/// Public tegra210.c: PPCS SMMU client shared by AHB DMA.
pub const SMMU_ASID: u64 = 0x270;
/// Bus 0x20f30's table at 0x44644, plus the SPI base of 32.
pub fn intid(channel: usize) -> u32 {
    32 + if channel < 16 {
        104 + channel as u32
    } else {
        112 + channel as u32
    }
}
#[derive(Serialize, Deserialize)]
struct State {
    general: u32,
    mask: u32,
    channels: [[u32; 10]; 32],
}
pub struct ApbDma {
    state: Mutex<State>,
    memory: DeviceMemory,
    /// The UARTs, by the address of their data register.
    uarts: Vec<(u64, Arc<Serial>)>,
    interrupts: Vec<Box<dyn Fn(bool) + Send + Sync>>,
    notes: Arc<Notes>,
}
impl ApbDma {
    pub fn new(
        memory: DeviceMemory,
        uarts: Vec<(u64, Arc<Serial>)>,
        interrupts: Vec<Box<dyn Fn(bool) + Send + Sync>>,
        notes: Arc<Notes>,
    ) -> Self {
        Self {
            state: Mutex::new(State {
                general: 0,
                mask: 0,
                channels: [[0; 10]; 32],
            }),
            memory,
            uarts,
            interrupts,
            notes,
        }
    }
    fn pump(&self, s: &mut State, channel: usize) -> Option<()> {
        let c = &mut s.channels[channel];
        if s.general & (1 << 31) == 0 || c[0] & (1 << 31) == 0 || c[3] & (1 << 31) != 0 {
            return Some(());
        }
        let uart = self
            .uarts
            .iter()
            .find(|(base, _)| *base == u64::from(c[6]))
            .filter(|_| c[0] & (1 << 27) != 0);
        let Some((_, uart)) = uart else {
            self.notes.add(format!(
                "APB DMA channel {channel}: unsupported transfer {c:08x?}"
            ));
            return None;
        };
        let count = (c[8] & 0xfffc) + 4;
        let transmit = c[0] & (1 << 28) != 0;
        while c[9] < count {
            let address = u64::from(c[4]) + u64::from(c[9]);
            if transmit {
                let mut byte = [0];
                if self.memory.read(address, &mut byte).is_none() {
                    self.notes
                        .add(format!("APB DMA unmapped source {address:#x}"));
                    return None;
                }
                uart.write(0, 0, 4, u64::from(byte[0]))?;
            } else {
                if c[9].is_multiple_of(4) && uart.receive_len() < 4 {
                    break;
                }
                let byte = uart.read(0, 0, 4)? as u8;
                self.memory.write(address, &[byte])?;
            }
            c[9] += 1;
            // Bus 0x211e0 reads the channel byte-transfer counter at +8.
            c[2] = c[9];
        }
        if c[9] == count {
            c[0] &= !(1 << 31);
            c[1] = 1 << 30;
        } else {
            c[1] = 1 << 31;
        }
        (self.interrupts[channel])(c[1] & c[0] & (1 << 30) != 0 && s.mask & (1 << channel) != 0);
        Some(())
    }
}
impl MmioDevice for ApbDma {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        let offset = word(offset, size)?;
        let mut s = self.state.lock().unwrap();
        let value = match offset {
            0 => s.general,
            0x1c => s.mask,
            0x3c | 0x54 => s
                .channels
                .iter()
                .enumerate()
                .fold(0, |bits, (i, c)| bits | (((c[1] >> 30) & 1) << i)),
            0x1000..=0x17fc => {
                let channel = ((offset - 0x1000) / 0x40) as usize;
                let reg = ((offset & 0x3f) / 4) as usize;
                if reg > 9 {
                    return None;
                }
                self.pump(&mut s, channel)?;
                s.channels[channel][reg]
            }
            _ => return None,
        };
        Some(u64::from(value))
    }
    fn write(&self, _core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        let offset = word(offset, size)?;
        let value = value as u32;
        let mut s = self.state.lock().unwrap();
        match offset {
            0 => s.general = value,
            0x1c => s.mask = value,
            0x20 => s.mask |= value,
            0x24 => s.mask &= !value,
            0x1000..=0x17fc => {
                let channel = ((offset - 0x1000) / 0x40) as usize;
                let reg = ((offset & 0x3f) / 4) as usize;
                let c = &mut s.channels[channel];
                match reg {
                    1 => c[1] &= !(value & (1 << 30)),
                    0 => {
                        if value & (1 << 31) != 0 && c[0] & (1 << 31) == 0 {
                            c[9] = 0;
                            c[2] = 0;
                        }
                        c[0] = value;
                    }
                    3..=8 => c[reg] = value,
                    _ => return None,
                }
                self.pump(&mut s, channel)?;
                let c = &s.channels[channel];
                (self.interrupts[channel])(
                    c[1] & c[0] & (1 << 30) != 0 && s.mask & (1 << channel) != 0,
                );
            }
            _ => return None,
        }
        Some(())
    }

    fn snapshot(&self) -> Option<&dyn Snapshot> {
        Some(self)
    }
}

/// Saves the global enable, the interrupt mask and every channel's
/// registers, including how far a transfer has come: a transfer waiting
/// for received bytes continues from there. The UARTs save their own
/// FIFOs; the channel interrupt lines are levels the interrupt controller
/// saves.
impl Snapshot for ApbDma {
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        encode(&*self.state.lock().unwrap())
    }

    fn restore(&self, state: &[u8]) -> Result<(), SnapshotError> {
        *self.state.lock().unwrap() = decode("APB DMA", state)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        boot::DramSize,
        memory::GuestRam,
        soc::{bluetooth_uart, mc::Mc},
    };
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn dma_moves_reset_command_before_raising_completion() {
        let ram = Arc::new(GuestRam::new(0x8000_0000, 0x4000).unwrap());
        let mc = Arc::new(Mc::new(DramSize::Gb4));
        ram.write(0x8000_1800, &0xc008_0000u32.to_le_bytes())
            .unwrap();
        ram.write(0x8000_2000, &[1, 3, 12, 0]).unwrap();
        mc.write(0, 0x1c, 4, 1).unwrap();
        mc.write(0, 0x20, 4, 0xe008_0001).unwrap();
        mc.write(0, SMMU_ASID, 4, 0x8000_0001).unwrap();
        let uart = Arc::new(Serial::new(
            Box::new(bluetooth_uart::Line::default()),
            Box::new(|_| {}),
            Arc::default(),
        ));
        let raised = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&raised);
        let mut interrupts: Vec<Box<dyn Fn(bool) + Send + Sync>> = (0..32)
            .map(|_| Box::new(|_| {}) as Box<dyn Fn(bool) + Send + Sync>)
            .collect();
        interrupts[0] = Box::new(move |value| flag.store(value, Ordering::Relaxed));
        let dma = ApbDma::new(
            DeviceMemory::new(&mc, &ram, SMMU_ASID),
            vec![(bluetooth_uart::BASE, Arc::clone(&uart))],
            interrupts,
            Arc::default(),
        );
        for (reg, value) in [
            (0, 1 << 31),
            (0x20, 1),
            (0x1010, 0x8000_2000),
            (0x1018, bluetooth_uart::BASE as u32),
            (0x1020, 0),
            (0x1000, 0xd833_0000),
        ] {
            dma.write(0, reg, 4, u64::from(value)).unwrap();
        }
        assert!(raised.load(Ordering::Relaxed));
        assert_eq!(dma.read(0, 0x1008, 4), Some(4));
        let reply: Vec<_> = (0..7).map(|_| uart.read(0, 0, 4).unwrap()).collect();
        assert_eq!(reply, [4, 14, 4, 1, 3, 12, 0]);
        dma.write(0, 0x1004, 4, 1 << 30).unwrap();
        assert!(!raised.load(Ordering::Relaxed));
    }
}
