//! Host-engine Falcon boot interface, not a Falcon instruction emulator.
//!
//! nvservices 23.5.0 `0x809e0` (VIC) and `0x835e0` (NVDEC) load 256-byte
//! DMEM/IMEM blocks through 0x1110..0x111c, start CPUCTL, and require
//! IDLESTATE == 0 (spec §5.2). Firmware bytes are fetched through the
//! engine SMMU and retained; the [`Engine`] is shown them when CPUCTL
//! starts the processor and may refuse firmware it does not know.
//! Completing boot means the modeled service is ready, not that firmware
//! was executed. pcv `0x39450`, partition table `0xb1b18`, workaround table
//! `0xab40c`, sets then clears VIC SLCG override 0x8c mask 0x010000dd for
//! MBIST.
//!
//! The THI method port at +0x40/+0x44 is THI words 0x10/0x11, the registers
//! host1x channels reach as class methods; both go to the [`Engine`].
use super::{Notes, mc::DeviceMemory};
use crate::{
    bus::MmioDevice,
    snapshot::{Snapshot, SnapshotError, decode, encode},
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

pub const VIC_BASE: u64 = 0x5434_0000;
pub const NVDEC_BASE: u64 = 0x5448_0000;
pub const SIZE: u64 = 0x40000;

/// The engine a host falcon runs: what its THI registers and class methods
/// reach.
pub trait Engine: Snapshot {
    /// Its name in notes and snapshot errors.
    const NAME: &'static str;
    /// Its host1x class.
    const CLASS: u32;
    /// A write of THI register `offset` (in words), from a host1x channel
    /// or from the MMIO window (byte offset `4 * offset`).
    fn thi_write(&self, offset: u32, value: u32) -> Option<()>;
    /// Value of the THI method offset register (MMIO +0x40).
    fn method_offset(&self) -> u32;
    /// CPUCTL starts the processor with these memories; an error refuses
    /// the start.
    fn start(&self, imem: &[u8], dmem: &[u8]) -> Result<(), String>;
    /// A write of the engine's own register at `address`, through the
    /// private register window. Refused unless the engine models it.
    fn private_write(&self, _address: u32, _value: u32) -> Option<()> {
        None
    }
}

/// The Falcon's window on its engine's own registers: the word at
/// `0x1000 + (address >> 6)` is the engine's register `address`, the low
/// address bits coming from the register at `0x10AC` (hekate `vic.c`
/// `_vic_write_priv`, `PVIC_FALCON_PA_OFFSET` and `PVIC_FALCON_ADDR`,
/// reference for facts). The engines' registers start at `0x10000`, past
/// the Falcon's own. A register of the window this model keeps itself
/// (`0x1604`, which nvservices writes) is not passed to the engine.
const PRIVATE_BASE: u64 = 0x1000;
const PRIVATE_SHIFT: u32 = 6;
const PRIVATE_WINDOW: std::ops::Range<u64> = 0x1400..0x2000;

/// The VIC's firmware is not checked: any firmware starts it (the
/// compositor is modeled whatever runs on its Falcon).
impl Engine for super::vic::Vic {
    const NAME: &'static str = "VIC";
    const CLASS: u32 = super::vic::CLASS;

    fn thi_write(&self, offset: u32, value: u32) -> Option<()> {
        super::vic::Vic::thi_write(self, offset, value)
    }

    fn method_offset(&self) -> u32 {
        super::vic::Vic::method_offset(self)
    }

    fn start(&self, _imem: &[u8], _dmem: &[u8]) -> Result<(), String> {
        Ok(())
    }

    fn private_write(&self, address: u32, value: u32) -> Option<()> {
        super::vic::Vic::private_write(self, address, value)
    }
}

#[derive(Serialize, Deserialize)]
struct State {
    #[serde(with = "crate::snapshot::array")]
    regs: [u32; 0x1610 / 4],
    #[serde(with = "crate::snapshot::array")]
    imem: [u8; 0x10000],
    #[serde(with = "crate::snapshot::array")]
    dmem: [u8; 0x10000],
}
impl Default for State {
    fn default() -> Self {
        Self {
            regs: [0; 0x1610 / 4],
            imem: [0; 0x10000],
            dmem: [0; 0x10000],
        }
    }
}

pub struct HostFalcon<E> {
    state: Mutex<State>,
    memory: DeviceMemory,
    engine: E,
    notes: Arc<Notes>,
}
impl<E: Engine> HostFalcon<E> {
    /// `memory` is memory as the engine's SMMU client reaches it.
    pub fn new(memory: DeviceMemory, engine: E, notes: Arc<Notes>) -> Self {
        Self {
            state: Mutex::new(State::default()),
            memory,
            engine,
            notes,
        }
    }
    /// A host1x channel method: THI register `offset` (in words) of `class`.
    pub fn method(&self, class: u32, offset: u32, value: u32) -> Option<()> {
        if class != E::CLASS {
            return self.refuse(format!("class {class:#x} method {offset:#x}={value:#x}"));
        }
        self.engine.thi_write(offset, value)
    }
    fn known(offset: u64) -> bool {
        matches!(
            offset,
            0x8c | 0x1010
                | 0x101c
                | 0x1048
                | 0x104c
                | 0x1100
                | 0x1104
                | 0x110c
                | 0x1110
                | 0x1114
                | 0x1118
                | 0x111c
                | 0x1604
        )
    }
    fn refuse(&self, reason: String) -> Option<()> {
        self.notes.add(format!("{}: {reason}", E::NAME));
        None
    }
}
impl<E: Engine> MmioDevice for HostFalcon<E> {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        if size == 4 && offset == 0x40 {
            return Some(self.engine.method_offset().into());
        }
        if size != 4 || !Self::known(offset) {
            self.refuse(format!("unmodeled read +{offset:#x}/{size}"))?;
        }
        Some(u64::from(
            self.state.lock().unwrap().regs[offset as usize / 4],
        ))
    }
    fn write(&self, _core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        // nvservices writes CPUCTL and the adjacent zero BOOTVEC together.
        if offset == 0x1100 && size == 8 {
            self.write(_core, 0x1104, 4, value >> 32)?;
            return self.write(_core, 0x1100, 4, value as u32 as u64);
        }
        if size == 4 && matches!(offset, 0x40 | 0x44) {
            return self.engine.thi_write(offset as u32 / 4, value as u32);
        }
        if size == 4 && PRIVATE_WINDOW.contains(&offset) && !Self::known(offset) {
            // The low address bits come from 0x10AC, which nothing here
            // lets be written: they are zero.
            let address = ((offset - PRIVATE_BASE) << PRIVATE_SHIFT) as u32;
            return self.engine.private_write(address, value as u32);
        }
        if size != 4 || !Self::known(offset) {
            return self.refuse(format!("unmodeled write +{offset:#x}/{size}={value:#x}"));
        }
        let mut s = self.state.lock().unwrap();
        let v = value as u32;
        match offset {
            0x1118 => {
                if !matches!(v, 0x600 | 0x610) {
                    return self.refuse(format!("DMA command {v:#x}"));
                }
                let address = (u64::from(s.regs[0x1110 / 4]) << 8) + u64::from(s.regs[0x111c / 4]);
                let at = s.regs[0x1114 / 4] as usize;
                let target = if v & 0x10 != 0 {
                    &mut s.imem
                } else {
                    &mut s.dmem
                };
                let Some(block) = target.get_mut(at..at.saturating_add(256)) else {
                    return self.refuse(format!("DMA local offset {at:#x}"));
                };
                if self.memory.read(address, block).is_none() {
                    return self.refuse(format!("DMA unmapped IOVA {address:#x}"));
                }
                s.regs[0x1118 / 4] = v | 2;
                return Some(());
            }
            0x1100 if v != 2 => return self.refuse(format!("CPUCTL {v:#x}")),
            0x1100 => {
                if let Err(reason) = self.engine.start(&s.imem, &s.dmem) {
                    return self.refuse(reason);
                }
            }
            _ => {}
        }
        s.regs[offset as usize / 4] = v;
        Some(())
    }

    fn snapshot(&self) -> Option<&dyn Snapshot> {
        Some(self)
    }
}

/// Saves the boot interface registers and the firmware loaded into IMEM
/// and DMEM, and with them the [`Engine`]'s own state.
impl<E: Engine> Snapshot for HostFalcon<E> {
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        let host = encode(&*self.state.lock().unwrap())?;
        encode(&(host, self.engine.save()?))
    }

    fn restore(&self, state: &[u8]) -> Result<(), SnapshotError> {
        let (host, engine): (Vec<u8>, Vec<u8>) = decode(E::NAME, state)?;
        let host: State = decode(E::NAME, &host)?;
        self.engine.restore(&engine)?;
        *self.state.lock().unwrap() = host;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        boot::DramSize,
        memory::GuestRam,
        soc::{
            mc::{Mc, SMMU_VIC_ASID},
            vic::Vic,
        },
    };

    #[test]
    fn firmware_dma_and_start_handshake() {
        let ram = Arc::new(GuestRam::new(0x8000_0000, 0x4000).unwrap());
        let mc = Arc::new(Mc::new(DramSize::Gb4));
        mc.write(0, 0x1c, 4, 1).unwrap();
        mc.write(0, 0x20, 4, 0xe0080000).unwrap();
        mc.write(0, SMMU_VIC_ASID, 4, 0x80000001).unwrap();
        ram.write(0x8000_0000, &0xd0080001u32.to_le_bytes())
            .unwrap();
        ram.write(0x8000_1000, &0xc0080002u32.to_le_bytes())
            .unwrap();
        ram.write(0x8000_2000, &[0xa5; 256]).unwrap();
        let memory = || DeviceMemory::new(&mc, &ram, SMMU_VIC_ASID);
        let vic = Vic::new(memory(), Arc::default(), Arc::default());
        let engine = HostFalcon::new(memory(), vic, Arc::default());
        assert_eq!(engine.read(0, 0x110c, 4), Some(0));
        engine.write(0, 0x1110, 4, 0).unwrap();
        engine.write(0, 0x1114, 4, 0).unwrap();
        engine.write(0, 0x111c, 4, 0).unwrap();
        engine.write(0, 0x1118, 4, 0x610).unwrap();
        assert_eq!(engine.read(0, 0x1118, 4), Some(0x612));
        assert_eq!(&engine.state.lock().unwrap().imem[..256], &[0xa5; 256]);
        engine.write(0, 0x1100, 8, 2).unwrap();
        assert_eq!(engine.read(0, 0x104c, 4), Some(0));
        assert_eq!(engine.write(0, 0x1118, 4, 0xdead), None);
        assert_eq!(engine.read(0, 0x1234, 4), None);
        engine.write(0, 0x8c, 4, 0x010000dd).unwrap();
        assert_eq!(engine.read(0, 0x8c, 4), Some(0x010000dd));
        engine.write(0, 0x8c, 4, 0).unwrap();
        // nvservices' write inside the private register window stays the
        // host's own; the window's other words are the engine's to refuse.
        engine.write(0, 0x1604, 4, 3).unwrap();
        assert_eq!(engine.read(0, 0x1604, 4), Some(3));
        assert_eq!(engine.write(0, 0x1608, 4, 3), None);
        // The boot methods nvservices sends through the THI port (survey
        // run: 0x80 = 1, 0x1c7 = 0x3c0, 0x1cb = 0xe0043d).
        for (method, value) in [(0x80, 1), (0x1c7, 0x3c0), (0x1cb, 0xe0043d)] {
            engine.write(0, 0x40, 4, method).unwrap();
            engine.write(0, 0x44, 4, value).unwrap();
        }
        assert_eq!(engine.read(0, 0x40, 4), Some(0x1cb));
        engine.write(0, 0x40, 4, 0x80).unwrap();
        assert_eq!(engine.write(0, 0x44, 4, 2), None);
    }
}
