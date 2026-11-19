//! Legacy interrupt controller (LIC, "ICTLR"): the boot processor's
//! interrupt controller.
//!
//! The BPMP is not connected to the CPU cluster's GIC; its IRQ and FIQ
//! inputs come from this block, whose six controllers each take 32 of the
//! SoC's shared interrupts (interrupt `n` is bit `n % 32` of controller
//! `n / 32`, the numbering of the GIC's shared interrupts: Linux
//! `drivers/irqchip/irq-tegra.c`, `tegra210.dtsi` `interrupt-controller@60004000`
//! with controllers at `0x60004000 + 0x100 * i`).
//!
//! Registers of each controller (hekate `bdk/soc/t210.h` `PRI_ICTLR_*`,
//! Linux `ICTLR_*`; the four virtual status registers at `0x00`-`0x0c` as
//! named in the T124/T210 TRM):
//!
//! | offset | register | access |
//! |---|---|---|
//! | `0x00` / `0x04` | `VIRQ_CPU` / `VIRQ_COP` | read: pending, enabled for that processor, class IRQ |
//! | `0x08` / `0x0c` | `VFIQ_CPU` / `VFIQ_COP` | read: likewise, class FIQ |
//! | `0x10` | `ISR` | read: levels of the interrupt inputs |
//! | `0x14` / `0x18` / `0x1c` | `FIR` / `FIR_SET` / `FIR_CLR` | forced (software) interrupts: read / set bits / clear bits |
//! | `0x20` / `0x24` / `0x28` | `CPU_IER` / `_SET` / `_CLR` | enables for the CPU cluster |
//! | `0x2c` | `CPU_IEP_CLASS` | read/write: bit set = FIQ |
//! | `0x30` / `0x34` / `0x38` | `COP_IER` / `_SET` / `_CLR` | enables for the BPMP |
//! | `0x3c` | `COP_IEP_CLASS` | read/write |
//!
//! An interrupt is pending when its input is asserted or it is forced.
//! \[assumption\] `ISR` shows the inputs only, not the forced bits.
//! A set or clear register reads as the register it acts on
//! (\[inference\]: the NX bootloader's hardware-state check reads all six
//! of them per controller and requires zero, the reset value of their
//! targets; `nxbl_hwcheck.csv`). Writing a read-only register is refused.
//! Reset values: all enables, classes and forced bits zero (\[assumption\],
//! the state in which nothing is delivered, which Linux's driver also
//! establishes before use).
//!
//! The CPU side is storage here: the CPU cluster takes its interrupts
//! from the GIC ([`crate::gic`]), and the LIC's part on that path (gating
//! wake-ups) is not modelled. The BPMP's two lines are
//! [`Ictlr::cop_lines`]. Inputs arrive through [`Ictlr::set_line`]; no
//! SoC device drives them yet (each drives its GIC line only), which is
//! what the boot chain's BPMP code is expected to need: it polls
//! \[inference: the public reimplementations run the BPMP with
//! interrupts masked\].

use std::sync::{
    Mutex,
    atomic::{AtomicBool, Ordering},
};

use serde::{Deserialize, Serialize};

use super::regs::word;
use crate::{
    bus::MmioDevice,
    snapshot::{Snapshot, SnapshotError, decode, encode},
};

pub const BASE: u64 = 0x6000_4000;
pub const CONTROLLERS: usize = 6;
pub const SIZE: u64 = 0x100 * CONTROLLERS as u64;
/// Interrupt inputs: 32 per controller.
pub const INTERRUPTS: u32 = 32 * CONTROLLERS as u32;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
struct Controller {
    inputs: u32,
    forced: u32,
    cpu_enabled: u32,
    cpu_fiq: u32,
    cop_enabled: u32,
    cop_fiq: u32,
}

impl Controller {
    fn pending(&self) -> u32 {
        self.inputs | self.forced
    }

    fn cop_irq(&self) -> u32 {
        self.pending() & self.cop_enabled & !self.cop_fiq
    }

    fn cop_fiq(&self) -> u32 {
        self.pending() & self.cop_enabled & self.cop_fiq
    }
}

#[derive(Default)]
pub struct Ictlr {
    controllers: Mutex<[Controller; CONTROLLERS]>,
    cop_irq: AtomicBool,
    cop_fiq: AtomicBool,
}

impl Ictlr {
    pub fn new() -> Self {
        Self::default()
    }

    /// Interrupt input `interrupt` (a shared interrupt's number, GIC ID
    /// minus 32) is at `level`.
    ///
    /// # Panics
    /// If `interrupt` is not below [`INTERRUPTS`]: wiring is fixed when
    /// the machine is built.
    pub fn set_line(&self, interrupt: u32, level: bool) {
        assert!(interrupt < INTERRUPTS, "LIC has no input {interrupt}");
        let mut controllers = self.controllers.lock().unwrap();
        let controller = &mut controllers[(interrupt / 32) as usize];
        let bit = 1 << (interrupt % 32);
        if level {
            controller.inputs |= bit;
        } else {
            controller.inputs &= !bit;
        }
        self.update(&controllers);
    }

    /// The BPMP's IRQ and FIQ inputs.
    pub fn cop_lines(&self) -> (bool, bool) {
        (
            self.cop_irq.load(Ordering::Acquire),
            self.cop_fiq.load(Ordering::Acquire),
        )
    }

    fn update(&self, controllers: &[Controller; CONTROLLERS]) {
        let irq = controllers.iter().any(|c| c.cop_irq() != 0);
        let fiq = controllers.iter().any(|c| c.cop_fiq() != 0);
        self.cop_irq.store(irq, Ordering::Release);
        self.cop_fiq.store(fiq, Ordering::Release);
    }
}

impl MmioDevice for Ictlr {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        let offset = word(offset, size)?;
        let controllers = self.controllers.lock().unwrap();
        let c = controllers.get((offset / 0x100) as usize)?;
        let value = match offset % 0x100 {
            0x00 => c.pending() & c.cpu_enabled & !c.cpu_fiq,
            0x04 => c.cop_irq(),
            0x08 => c.pending() & c.cpu_enabled & c.cpu_fiq,
            0x0C => c.cop_fiq(),
            0x10 => c.inputs,
            0x14 | 0x18 | 0x1C => c.forced,
            0x20 | 0x24 | 0x28 => c.cpu_enabled,
            0x2C => c.cpu_fiq,
            0x30 | 0x34 | 0x38 => c.cop_enabled,
            0x3C => c.cop_fiq,
            _ => return None,
        };
        Some(u64::from(value))
    }

    fn write(&self, _core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        let offset = word(offset, size)?;
        let value = value as u32;
        let mut controllers = self.controllers.lock().unwrap();
        let c = controllers.get_mut((offset / 0x100) as usize)?;
        match offset % 0x100 {
            0x18 => c.forced |= value,
            0x1C => c.forced &= !value,
            0x24 => c.cpu_enabled |= value,
            0x28 => c.cpu_enabled &= !value,
            0x2C => c.cpu_fiq = value,
            0x34 => c.cop_enabled |= value,
            0x38 => c.cop_enabled &= !value,
            0x3C => c.cop_fiq = value,
            _ => return None,
        }
        self.update(&controllers);
        Some(())
    }

    fn snapshot(&self) -> Option<&dyn Snapshot> {
        Some(self)
    }
}

/// Saves every controller, input levels included (the devices driving
/// them do not drive them again on restore).
impl Snapshot for Ictlr {
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        encode(&*self.controllers.lock().unwrap())
    }

    fn restore(&self, state: &[u8]) -> Result<(), SnapshotError> {
        let mut controllers = self.controllers.lock().unwrap();
        *controllers = decode("legacy interrupt controller", state)?;
        self.update(&controllers);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(ictlr: &Ictlr, offset: u64) -> u32 {
        ictlr.read(0, offset, 4).unwrap() as u32
    }

    #[test]
    fn delivers_enabled_interrupts_to_the_bpmp_by_class() {
        let ictlr = Ictlr::new();
        // Interrupt 38 (I2C1): controller 1, bit 6.
        ictlr.set_line(38, true);
        assert_eq!(read(&ictlr, 0x110), 1 << 6);
        assert_eq!(ictlr.cop_lines(), (false, false));
        ictlr.write(0, 0x134, 4, 1 << 6).unwrap();
        assert_eq!(ictlr.cop_lines(), (true, false));
        assert_eq!(read(&ictlr, 0x104), 1 << 6);
        assert_eq!(read(&ictlr, 0x10C), 0);
        // Class FIQ.
        ictlr.write(0, 0x13C, 4, 1 << 6).unwrap();
        assert_eq!(ictlr.cop_lines(), (false, true));
        assert_eq!(read(&ictlr, 0x10C), 1 << 6);
        ictlr.set_line(38, false);
        assert_eq!(ictlr.cop_lines(), (false, false));
        // The CPU side does not reach the BPMP.
        ictlr.set_line(0, true);
        ictlr.write(0, 0x24, 4, 1).unwrap();
        assert_eq!(read(&ictlr, 0x00), 1);
        assert_eq!(ictlr.cop_lines(), (false, false));
    }

    #[test]
    fn forced_interrupts_and_enable_clear() {
        let ictlr = Ictlr::new();
        ictlr.write(0, 0x534, 4, 0x8000_0001).unwrap();
        assert_eq!(read(&ictlr, 0x530), 0x8000_0001);
        ictlr.write(0, 0x518, 4, 1).unwrap();
        assert_eq!(read(&ictlr, 0x514), 1);
        assert_eq!(read(&ictlr, 0x510), 0, "ISR shows inputs only");
        assert_eq!(ictlr.cop_lines(), (true, false));
        ictlr.write(0, 0x538, 4, 1).unwrap();
        assert_eq!(ictlr.cop_lines(), (false, false));
        ictlr.write(0, 0x51C, 4, 1).unwrap();
        assert_eq!(read(&ictlr, 0x514), 0);
    }

    #[test]
    fn refuses_wrong_direction_and_unknown_offsets() {
        let ictlr = Ictlr::new();
        assert_eq!(ictlr.read(0, 0x24, 4), Some(0));
        assert_eq!(ictlr.write(0, 0x10, 4, 0), None);
        assert_eq!(ictlr.write(0, 0x04, 4, 0), None);
        assert_eq!(ictlr.read(0, 0x40, 4), None);
        assert_eq!(ictlr.read(0, 0x600, 4), None);
        assert_eq!(ictlr.read(0, 0x30, 2), None);
    }

    #[test]
    fn snapshot_restores_lines() {
        let ictlr = Ictlr::new();
        ictlr.write(0, 0x34, 4, 1 << 3).unwrap();
        ictlr.set_line(3, true);
        let other = Ictlr::new();
        other.restore(&ictlr.save().unwrap()).unwrap();
        assert_eq!(other.cop_lines(), (true, false));
    }
}
