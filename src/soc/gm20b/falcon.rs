//! The three falcon microcontrollers the driver starts at every power-on:
//! the PMU, and FECS and GPCCS, the context-switch controllers of the
//! graphics engine.
//!
//! None of their code runs here. The PMU is started once to run NVIDIA's
//! signed ACR (secure boot) application, which the falcon boot ROM
//! authenticates with a hardware key and which therefore cannot run
//! outside a real chip; FECS and GPCCS run firmware whose job, saving and
//! restoring graphics contexts, has no counterpart in a model without a
//! hardware graphics pipeline. What the driver observes of each is
//! modeled instead (spec §4.4, §4.5, annex boot §5-6):
//!
//! * ACR: after the driver loads the PMU bootloader and starts the PMU
//!   (`CPUCTL` ← 2, nvservices text `0x86908`), the application has run:
//!   the PMU halts (`CPUCTL` bit 4, polled at `0x864D0`), `MAILBOX0` reads
//!   0 (the driver's success test; anything else fails power-on with −11)
//!   and the halt interrupt is pending in `IRQSTAT` bit 4 until the driver
//!   clears it through `IRQSCLR`. On a console the application has also
//!   copied the LS firmware of FECS and of the PMU from the driver's blob
//!   into their falcons; that is what lets the driver start FECS later.
//!   The blob is not parsed or checked here.
//! * LS PMU: started through `CPUCTL_ALIAS` (`0x86A78`). The driver waits
//!   for nothing from it (spec §4.4 item 4); it is left running without
//!   ever posting a message, which the driver tolerates: no power gating
//!   and no hardware mutex.
//! * GPCCS is started by `CPUCTL` ← 2 after its bootloader is loaded, FECS
//!   by `CPUCTL_ALIAS` ← 2 (`NV_BootstrapSecureGraphicsFalcons_87D50`).
//!   Once both run and FECS holds the LS image the ACR placed, FECS
//!   reports ready: mailbox 0 reads 1 (polled at `0x665C0`, `[RECALLED]`
//!   nvgpu's "init complete" handshake value).
//! * FECS methods (`0x409500` data, `0x409504` method) answer in the
//!   mailboxes the way `NV_ExecuteRegisterCommandAndWait_62170` expects
//!   (table at `Fecs::method`).
//!
//! Scrubbing of the falcons' memories at reset is over at once: `DMACTL`
//! bits 2:1 read 0. They are idle (`IDLESTATE` 0) because no code runs.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::identity;
use crate::maxwell::Unmodeled;

/// Registers common to the falcons, relative to each one's base.
const IRQSCLR: u32 = 0x004;
const IRQSTAT: u32 = 0x008;
const IRQMSET: u32 = 0x010;
const IRQMCLR: u32 = 0x014;
const IRQMASK: u32 = 0x018;
const MAILBOX0: u32 = 0x040;
const MAILBOX1: u32 = 0x044;
const IDLESTATE: u32 = 0x04C;
const CPUCTL: u32 = 0x100;
const DMACTL: u32 = 0x10C;
const DMATRFCMD: u32 = 0x118;
const CPUCTL_ALIAS: u32 = 0x130;
const IMEMC: u32 = 0x180;
const IMEMD: u32 = 0x184;
const IMEMT: u32 = 0x188;
const DMEMC: u32 = 0x1C0;
const DMEMD: u32 = 0x1C4;

const CPUCTL_STARTCPU: u32 = 1 << 1;
const CPUCTL_HALTED: u32 = 1 << 4;
const IRQ_HALT: u32 = 1 << 4;
/// `DMACTL` bits that read set while instruction and data memory are being
/// scrubbed.
const DMACTL_SCRUBBING: u32 = 3 << 1;
/// `DMATRFCMD` bit that reads set when the DMA engine is idle (the bit the
/// driver polls for the host1x falcons, spec §5.2 step 4).
const DMATRFCMD_IDLE: u32 = 1 << 1;
/// Memory port control: byte offset in bits 15:2, auto-increment on write
/// (bit 24) and on read (bit 25) `[RECALLED]` (the driver writes
/// `0x1000000` before writing descriptors word by word, spec §4.5).
const PORT_OFFSET: u32 = 0xFFFC;
const PORT_AUTO_INCREMENT_WRITE: u32 = 1 << 24;
const PORT_AUTO_INCREMENT_READ: u32 = 1 << 25;

/// What one falcon's processor is doing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum Cpu {
    /// Out of reset, never started.
    Stopped,
    Running,
    Halted,
}

/// One of a falcon's memories as its programmed-I/O port sees it.
#[derive(Default, Serialize, Deserialize)]
struct Port {
    control: u32,
    words: HashMap<u32, u32>,
}

impl Port {
    fn write_data(&mut self, value: u32) {
        let offset = self.control & PORT_OFFSET;
        self.words.insert(offset, value);
        if self.control & PORT_AUTO_INCREMENT_WRITE != 0 {
            self.step();
        }
    }

    fn read_data(&mut self) -> Option<u32> {
        let value = self.words.get(&(self.control & PORT_OFFSET)).copied();
        if self.control & PORT_AUTO_INCREMENT_READ != 0 {
            self.step();
        }
        value
    }

    fn step(&mut self) {
        let offset = (self.control & PORT_OFFSET) + 4;
        self.control = self.control & !PORT_OFFSET | offset & PORT_OFFSET;
    }
}

/// State every falcon has: interrupts, processor, memory ports.
#[derive(Serialize, Deserialize)]
struct Core {
    cpu: Cpu,
    irq_status: u32,
    irq_mask: u32,
    dma_control: u32,
    imem: Port,
    imem_tag: u32,
    dmem: Port,
}

impl Core {
    fn new() -> Self {
        Self {
            cpu: Cpu::Stopped,
            irq_status: 0,
            irq_mask: 0,
            dma_control: 0,
            imem: Port::default(),
            imem_tag: 0,
            dmem: Port::default(),
        }
    }

    fn cpuctl(&self) -> u32 {
        if self.cpu == Cpu::Halted {
            CPUCTL_HALTED
        } else {
            0
        }
    }

    /// The registers whose behavior is the same on every falcon; `None`
    /// for the others.
    fn read(&mut self, offset: u32) -> Option<Result<u32, Unmodeled>> {
        Some(Ok(match offset {
            // Write-only. Its only reader ORs in the halt bit and writes
            // the result back (`NV_BootstrapPmuFirmware_864D0`), at moments
            // when the model has no other interrupt pending; nothing says
            // what the register reads as, and zero clears exactly the bit
            // the driver names.
            IRQSCLR => 0,
            IRQSTAT => self.irq_status,
            IRQMASK => self.irq_mask,
            IDLESTATE => 0,
            CPUCTL => self.cpuctl(),
            DMACTL => self.dma_control & !DMACTL_SCRUBBING,
            IMEMT => self.imem_tag,
            IMEMC => self.imem.control,
            DMEMC => self.dmem.control,
            DMEMD => match self.dmem.read_data() {
                Some(value) => value,
                None => {
                    return Some(Err(Unmodeled(format!(
                        "falcon data memory at {:#x} read before it was written",
                        self.dmem.control & PORT_OFFSET
                    ))));
                }
            },
            _ => return None,
        }))
    }

    fn write(&mut self, offset: u32, value: u32) -> Option<()> {
        match offset {
            IRQSCLR => self.irq_status &= !value,
            IRQMSET => self.irq_mask |= value,
            IRQMCLR => self.irq_mask &= !value,
            DMACTL => self.dma_control = value,
            IMEMC => self.imem.control = value,
            IMEMD => self.imem.write_data(value),
            IMEMT => self.imem_tag = value,
            DMEMC => self.dmem.control = value,
            DMEMD => self.dmem.write_data(value),
            _ => return None,
        }
        Some(())
    }
}

/// The power management unit, at BAR0 `0x10A000`. Saved whole in machine
/// snapshots: processor state, memory words, mailboxes.
#[derive(Serialize, Deserialize)]
pub struct Pmu {
    core: Core,
    mailbox: [u32; 2],
    /// The ACR application has run since the last reset of the PMU.
    secure_boot_done: bool,
}

impl Pmu {
    pub(super) fn new() -> Self {
        Self {
            core: Core::new(),
            mailbox: [0; 2],
            secure_boot_done: false,
        }
    }

    /// Whether the PMU asks for the host's attention: a pending interrupt
    /// it has unmasked (`[RECALLED]` `IRQDEST` routing is not modeled; the
    /// driver routes everything it unmasks to the host).
    pub fn interrupt(&self) -> bool {
        self.core.irq_status & self.core.irq_mask != 0
    }

    /// `None`: not a register with behavior; the caller stores it.
    pub fn read(&mut self, offset: u32) -> Option<Result<u32, Unmodeled>> {
        match offset {
            MAILBOX0 => Some(Ok(self.mailbox[0])),
            MAILBOX1 => Some(Ok(self.mailbox[1])),
            _ => self.core.read(offset),
        }
    }

    /// Whether the access was one with behavior (`Ok(false)`: the caller
    /// stores it).
    pub fn write(&mut self, offset: u32, value: u32, fecs: &mut Fecs) -> Result<bool, Unmodeled> {
        match offset {
            MAILBOX0 => self.mailbox[0] = value,
            MAILBOX1 => self.mailbox[1] = value,
            CPUCTL if value & CPUCTL_STARTCPU != 0 => {
                if self.core.cpu != Cpu::Stopped || self.core.imem.words.is_empty() {
                    return Err(Unmodeled(format!(
                        "PMU started ({value:#x}) while {:?} or with nothing loaded; only the \
                         one secure-boot start after reset is modeled",
                        self.core.cpu
                    )));
                }
                // The ACR application ran to completion.
                self.core.cpu = Cpu::Halted;
                self.core.irq_status |= IRQ_HALT;
                self.mailbox[0] = 0;
                self.secure_boot_done = true;
                fecs.ls_firmware = true;
            }
            CPUCTL if value & !CPUCTL_STARTCPU == 0 => {}
            CPUCTL_ALIAS if value == CPUCTL_STARTCPU => {
                if !self.secure_boot_done || self.core.cpu != Cpu::Halted {
                    return Err(Unmodeled(
                        "LS PMU started without a completed secure boot".into(),
                    ));
                }
                // The LS PMU firmware runs; it is not executed.
                self.core.cpu = Cpu::Running;
            }
            CPUCTL | CPUCTL_ALIAS => {
                return Err(Unmodeled(format!(
                    "PMU CPUCTL{} write {value:#x} is not modeled",
                    if offset == CPUCTL { "" } else { "_ALIAS" }
                )));
            }
            DMATRFCMD => {
                return Err(Unmodeled(
                    "PMU DMA transfers (the non-secure boot path) are not modeled".into(),
                ));
            }
            _ => return Ok(self.core.write(offset, value).is_some()),
        }
        Ok(true)
    }
}

/// FECS-specific registers, relative to `0x409000`.
const FECS_CTXSW_STATUS_1: u32 = 0x400;
const FECS_METHOD_DATA: u32 = 0x500;
const FECS_METHOD_PUSH: u32 = 0x504;
const FECS_MAILBOX: u32 = 0x800;
const FECS_MAILBOX_CLEAR: u32 = 0x840;
const FECS_ARB_CTX_CMD: u32 = 0xA10;
/// `gr_fecs_ctxsw_mailbox__size_1` `[RECALLED]`: 16 mailboxes, each with
/// a write-one-to-clear twin 0x40 above it.
const FECS_MAILBOXES: u32 = 16;
/// The arbiter command field, which reads 0 once the command is done.
const ARB_CTX_CMD: u32 = 0x1F;

/// The FECS controller, at BAR0 `0x409000`. Saved whole in machine
/// snapshots.
#[derive(Serialize, Deserialize)]
pub struct Fecs {
    core: Core,
    mailbox: [u32; FECS_MAILBOXES as usize],
    method_data: u32,
    arb_command: u32,
    /// The LS firmware the ACR places in FECS is there.
    ls_firmware: bool,
}

impl Fecs {
    pub(super) fn new() -> Self {
        Self {
            core: Core::new(),
            mailbox: [0; FECS_MAILBOXES as usize],
            method_data: 0,
            arb_command: 0,
            ls_firmware: false,
        }
    }

    /// Back to the state after reset: the falcon loses its firmware.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    fn running(&self) -> bool {
        self.core.cpu == Cpu::Running
    }

    pub fn read(&mut self, offset: u32) -> Option<Result<u32, Unmodeled>> {
        Some(Ok(match offset {
            // `ctxsw_status_1`, polled for its arbiter-busy bit 12: the
            // arbiter finishes every command at once.
            FECS_CTXSW_STATUS_1 => 0,
            FECS_MAILBOX..0x840 if offset.is_multiple_of(4) => {
                self.mailbox[((offset - FECS_MAILBOX) / 4) as usize]
            }
            FECS_ARB_CTX_CMD => self.arb_command & !ARB_CTX_CMD,
            _ => return self.core.read(offset),
        }))
    }

    pub fn write(&mut self, offset: u32, value: u32, gpccs: &Gpccs) -> Result<bool, Unmodeled> {
        match offset {
            FECS_METHOD_DATA => self.method_data = value,
            FECS_METHOD_PUSH => self.method(value)?,
            FECS_MAILBOX..0x840 if offset.is_multiple_of(4) => {
                self.mailbox[((offset - FECS_MAILBOX) / 4) as usize] = value
            }
            FECS_MAILBOX_CLEAR..0x880 if offset.is_multiple_of(4) => {
                self.mailbox[((offset - FECS_MAILBOX_CLEAR) / 4) as usize] &= !value
            }
            FECS_ARB_CTX_CMD => self.arb_command = value,
            CPUCTL_ALIAS if value == CPUCTL_STARTCPU => {
                if !self.ls_firmware || self.core.cpu != Cpu::Stopped {
                    return Err(Unmodeled(format!(
                        "FECS started while {:?}{}",
                        self.core.cpu,
                        if self.ls_firmware {
                            ""
                        } else {
                            ", without the firmware secure boot places in it"
                        }
                    )));
                }
                self.core.cpu = Cpu::Running;
                self.report_ready(gpccs);
            }
            CPUCTL | CPUCTL_ALIAS | DMATRFCMD => {
                return Err(Unmodeled(format!(
                    "FECS register {offset:#x} write {value:#x} (non-secure boot path) \
                     is not modeled"
                )));
            }
            _ => return Ok(self.core.write(offset, value).is_some()),
        }
        Ok(true)
    }

    /// The firmware signals it is up once it and GPCCS both run.
    fn report_ready(&mut self, gpccs: &Gpccs) {
        if self.running() && gpccs.core.cpu == Cpu::Running {
            self.mailbox[0] = 1;
        }
    }

    /// One method, as the driver uses them (spec §4.5, annex boot §6.2):
    ///
    /// | method | answer |
    /// |---|---|
    /// | 0x21 (watchdog timeout) | none awaited |
    /// | 0x10, 0x16, 0x25, 0x30 (image sizes) | the size in mailbox 0 |
    /// | 0x31, 0x32 (register list instance / address) | mailbox 0 = 1 |
    /// | 0x03 (bind context) | mailbox 0 bit 4 |
    /// | 0x09 (save golden context) | mailbox 0 bit 0 |
    /// | 0x38, 0x39, 0x04 (stop / start context switching, halt pipeline) | mailbox 1 = 1 |
    ///
    /// The golden-context save leaves the context buffer as it is: the
    /// driver copies whatever is there and checks none of it.
    fn method(&mut self, method: u32) -> Result<(), Unmodeled> {
        if !self.running() {
            return Err(Unmodeled(format!(
                "FECS method {method:#x} sent while FECS is not running"
            )));
        }
        match method & 0xFFF {
            0x21 => {}
            0x10 => self.mailbox[0] = identity::FECS_GOLDEN_IMAGE_SIZE,
            0x16 => self.mailbox[0] = identity::FECS_ZCULL_IMAGE_SIZE,
            0x25 => self.mailbox[0] = identity::FECS_PM_IMAGE_SIZE,
            0x30 => self.mailbox[0] = identity::FECS_REGLIST_IMAGE_SIZE,
            0x31 | 0x32 => self.mailbox[0] = 1,
            0x03 => self.mailbox[0] |= 1 << 4,
            0x09 => self.mailbox[0] |= 1 << 0,
            0x38 | 0x39 | 0x04 => self.mailbox[1] = 1,
            other => {
                return Err(Unmodeled(format!(
                    "FECS method {other:#x} (data {:#x}) is not modeled",
                    self.method_data
                )));
            }
        }
        Ok(())
    }
}

/// The GPCCS controller, at BAR0 `0x41A000`. Saved whole in machine
/// snapshots.
#[derive(Serialize, Deserialize)]
pub struct Gpccs {
    core: Core,
    dma_command: u32,
}

impl Gpccs {
    pub(super) fn new() -> Self {
        Self {
            core: Core::new(),
            dma_command: 0,
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }

    pub fn read(&mut self, offset: u32) -> Option<Result<u32, Unmodeled>> {
        match offset {
            // Transfers of the bootloader into instruction memory are over
            // at once; nothing executes what they would have copied.
            DMATRFCMD => Some(Ok(self.dma_command | DMATRFCMD_IDLE)),
            _ => self.core.read(offset),
        }
    }

    pub fn write(&mut self, offset: u32, value: u32, fecs: &mut Fecs) -> Result<bool, Unmodeled> {
        match offset {
            DMATRFCMD => self.dma_command = value & !DMATRFCMD_IDLE,
            CPUCTL if value == CPUCTL_STARTCPU => {
                if self.core.cpu != Cpu::Stopped {
                    return Err(Unmodeled(format!(
                        "GPCCS started while {:?}",
                        self.core.cpu
                    )));
                }
                // Its bootloader fetches and starts the GPCCS firmware.
                self.core.cpu = Cpu::Running;
                fecs.report_ready(self);
            }
            CPUCTL | CPUCTL_ALIAS => {
                return Err(Unmodeled(format!(
                    "GPCCS register {offset:#x} write {value:#x} is not modeled"
                )));
            }
            _ => return Ok(self.core.write(offset, value).is_some()),
        }
        Ok(true)
    }
}

impl Pmu {
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The secure start as `NV_BootstrapPmuFirmware_864D0` performs it:
    /// bootloader into instruction memory, start, wait for the halt, read
    /// mailbox 0, clear the halt interrupt, start the LS PMU.
    #[test]
    fn secure_boot_halts_with_mailbox_zero_and_loads_fecs() {
        let mut pmu = Pmu::new();
        let mut fecs = Fecs::new();
        let read = |pmu: &mut Pmu, offset| pmu.read(offset).unwrap().unwrap();
        assert_eq!(read(&mut pmu, DMACTL) & 6, 0);
        pmu.write(MAILBOX0, 0xDEAD, &mut fecs).unwrap();
        pmu.write(DMEMC, 0x0100_0000, &mut fecs).unwrap();
        pmu.write(DMEMD, 0x11, &mut fecs).unwrap();
        pmu.write(DMEMD, 0x22, &mut fecs).unwrap();
        pmu.write(IMEMC, 0x0100_0000, &mut fecs).unwrap();
        pmu.write(IMEMD, 0x1234, &mut fecs).unwrap();
        assert!(!fecs.ls_firmware);
        pmu.write(CPUCTL, 2, &mut fecs).unwrap();
        assert_ne!(read(&mut pmu, CPUCTL) & CPUCTL_HALTED, 0);
        assert_eq!(read(&mut pmu, MAILBOX0), 0);
        assert!(fecs.ls_firmware);
        // The halt interrupt clears by IRQSCLR read-modify-write.
        assert_eq!(read(&mut pmu, IRQSTAT) & IRQ_HALT, IRQ_HALT);
        let clear = read(&mut pmu, IRQSCLR) | 0x10;
        pmu.write(IRQSCLR, clear, &mut fecs).unwrap();
        assert_eq!(read(&mut pmu, IRQSTAT), 0);
        pmu.write(IRQMSET, 0xF3, &mut fecs).unwrap();
        assert!(!pmu.interrupt());
        pmu.write(CPUCTL_ALIAS, 2, &mut fecs).unwrap();
        assert_eq!(read(&mut pmu, CPUCTL), 0);
        // The descriptor words went where the port said.
        pmu.write(DMEMC, 0x0200_0004, &mut fecs).unwrap();
        assert_eq!(read(&mut pmu, DMEMD), 0x22);
        // A second start without a reset is not modeled.
        assert!(pmu.write(CPUCTL, 2, &mut fecs).is_err());
    }

    #[test]
    fn fecs_reports_ready_once_both_controllers_run() {
        let mut pmu = Pmu::new();
        let mut fecs = Fecs::new();
        let mut gpccs = Gpccs::new();
        // FECS cannot start without the secure boot.
        assert!(fecs.write(CPUCTL_ALIAS, 2, &gpccs).is_err());
        pmu.write(IMEMD, 1, &mut fecs).unwrap();
        pmu.write(CPUCTL, 2, &mut fecs).unwrap();
        // NV_BootstrapSecureGraphicsFalcons_87D50, step 6.
        fecs.write(FECS_MAILBOX_CLEAR, !0, &gpccs).unwrap();
        fecs.write(FECS_MAILBOX + 4, 1, &gpccs).unwrap();
        fecs.write(FECS_MAILBOX_CLEAR + 0x18, !0, &gpccs).unwrap();
        assert!(gpccs.write(DMATRFCMD, 0x610, &mut fecs).unwrap());
        assert_eq!(gpccs.read(DMATRFCMD).unwrap().unwrap() & 2, 2);
        gpccs.write(CPUCTL, 2, &mut fecs).unwrap();
        assert_eq!(fecs.read(FECS_MAILBOX).unwrap().unwrap(), 0);
        fecs.write(CPUCTL_ALIAS, 2, &gpccs).unwrap();
        assert_eq!(fecs.read(FECS_MAILBOX).unwrap().unwrap(), 1);
        assert_eq!(fecs.read(FECS_MAILBOX + 4).unwrap().unwrap(), 1);
    }

    #[test]
    fn fecs_methods_answer_in_the_mailboxes() {
        let mut pmu = Pmu::new();
        let mut fecs = Fecs::new();
        let mut gpccs = Gpccs::new();
        pmu.write(IMEMD, 1, &mut fecs).unwrap();
        pmu.write(CPUCTL, 2, &mut fecs).unwrap();
        gpccs.write(CPUCTL, 2, &mut fecs).unwrap();
        fecs.write(CPUCTL_ALIAS, 2, &gpccs).unwrap();
        let call = |fecs: &mut Fecs, method, clear| {
            fecs.write(FECS_MAILBOX_CLEAR, clear, &gpccs).unwrap();
            fecs.write(FECS_METHOD_DATA, 0, &gpccs).unwrap();
            fecs.write(FECS_METHOD_PUSH, method, &gpccs)
        };
        call(&mut fecs, 0x10, !0).unwrap();
        assert_ne!(fecs.read(FECS_MAILBOX).unwrap().unwrap(), 0);
        call(&mut fecs, 0x31, !0).unwrap();
        assert_eq!(fecs.read(FECS_MAILBOX).unwrap().unwrap(), 1);
        call(&mut fecs, 0x03, 0x30).unwrap();
        assert_eq!(fecs.read(FECS_MAILBOX).unwrap().unwrap() & 0x30, 0x10);
        call(&mut fecs, 0x09, 3).unwrap();
        assert_eq!(fecs.read(FECS_MAILBOX).unwrap().unwrap() & 3, 1);
        fecs.write(FECS_MAILBOX + 4, !0, &gpccs).unwrap();
        call(&mut fecs, 0x38, !0).unwrap();
        assert_eq!(fecs.read(FECS_MAILBOX + 4).unwrap().unwrap(), 1);
        assert!(call(&mut fecs, 0x77, !0).is_err());
        // The arbiter finishes its commands at once.
        fecs.write(FECS_ARB_CTX_CMD, 7, &gpccs).unwrap();
        assert_eq!(fecs.read(FECS_ARB_CTX_CMD).unwrap().unwrap() & 0x1F, 0);
    }
}
