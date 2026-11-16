//! Interrupt controller: the Tegra X1's ARM CoreLink GIC-400, a GICv2 with the
//! Security Extensions, as its registers appear to the kernel's Non-secure
//! accesses.
//!
//! Behavior follows the *ARM Generic Interrupt Controller Architecture
//! Specification, version 2.0* (ARM IHI 0048B.b, "the specification" below)
//! and, where that leaves a choice to the implementation, the *CoreLink
//! GIC-400 Technical Reference Manual* (ARM DDI 0471B, "the TRM"). The Tegra X1
//! instance has four CPU interfaces and 224 interrupt IDs.
//!
//! # What the kernel sees of the Security Extensions
//!
//! The kernel runs Non-secure, and the secure monitor hands it a controller in
//! which every interrupt the kernel may own is Group 1. This model is that
//! Non-secure view with all implemented interrupts in Group 1:
//!
//! - `GICD_IGROUPRn` is RAZ/WI. The kernel writes zero to it
//!   (`Gic_InitializeCoreInterface`, `0x800d4f94`), which on this hardware
//!   changes nothing.
//! - `GICD_CTLR` and `GICC_CTLR` are the Non-secure copies: bit 0 enables
//!   Group 1, and `GICC_CTLR` further holds only the two bypass disables and
//!   `EOImodeNS`.
//! - Priorities use the Non-secure view (specification 3.5.1): the GIC-400
//!   stores 5 priority bits, Non-secure accesses see the lower half of that
//!   range shifted left by one, so bits `[7:4]` of every priority field,
//!   of `GICC_PMR` and of `GICC_RPR` are implemented and bits `[3:0]` are
//!   RAZ/WI. The kernel's priority bytes `0x3f`/`0x7f`/`0xbf`/`0xff`
//!   (`KInterruptManager_BindInterrupt`, `0x800ca864`) read back as
//!   `0x30`/`0x70`/`0xb0`/`0xf0`, and its mask `0xff` as `0xf0`, so priority
//!   `0xff` is never signaled.
//! - The Non-secure `GICC_BPR` has a minimum of 3. A value `b` makes bits
//!   `[7:b+1]` of the Non-secure view the group priority; the kernel's 7
//!   (`0x800d4e60`) leaves no group priority bits, so nothing preempts.
//! - `GICC_APR0` shows one bit per Non-secure preemption level in bits
//!   `[15:0]` (specification table 4-47, row for a secure minimum binary point
//!   of 2).
//! - `GICC_ABPR`, `GICC_AIAR`, `GICC_AEOIR`, `GICC_AHPPIR` and `GICC_NSAPR0`
//!   are Secure registers, RAZ/WI here (TRM table 3-6 note d, TRM 3.2.1).
//!
//! Writes to `GICC_PMR` are only honored by the hardware while the Secure view
//! of the mask is `0x80` or above, and interrupts left in Group 0 are RAZ/WI.
//! Both depend on state only the secure world can set; the model assumes the
//! secure world left the mask in the Non-secure range and kept no interrupt in
//! Group 0.
//!
//! # GIC-400 implementation choices (TRM)
//!
//! - SGIs are permanently enabled (table 3-2 notes f, g).
//! - Only PPIs 25-31 exist; IDs 16-24 are unimplemented (table 3-2 note d).
//!   PPIs are level-sensitive and not configurable (`GICD_ICFGR1` is read-only
//!   `0x55540000`). SPIs are level-sensitive or rising-edge-triggered, and the
//!   even bit of each of their `GICD_ICFGRn` fields reads as one (3.4.2).
//! - While the distributor is disabled an edge cannot make an interrupt
//!   pending, `GICD_SGIR` has no effect, and the pending registers do not
//!   show level-sensitive inputs (2.8).
//! - Among SGIs of one ID the lowest source CPU is acknowledged first, and a
//!   priority write does not affect an interrupt that is already active (2.6).
//! - `GICC_HPPIR` ignores `GICC_CTLR` (table 3-6 note c).
//! - `GICD_PPISR` and `GICD_SPISRn` expose the raw input lines (3.4.3, 3.4.4).
//!
//! # What is reported as not modeled
//!
//! Accesses return `None`, for the machine to stop on, when they have no
//! architectural answer or when the kernel has no business making them:
//! reserved offsets (the hardware treats these as RAZ/WI, but the kernel
//! touches none), reads of write-only registers, access widths the
//! specification does not permit (4.1.4: words everywhere, bytes and halfwords
//! only for `GICD_IPRIORITYRn`, `GICD_ITARGETSRn`, `GICD_CPENDSGIRn` and
//! `GICD_SPENDSGIRn`), the reserved `GICD_SGIR` target filter, and the writes
//! to `GICC_EOIR` and `GICC_DIR` the specification calls UNPREDICTABLE.
//!
//! The legacy interrupt bypass is not modeled: the bypass disable bits of
//! `GICC_CTLR` are stored but a disabled CPU interface drives no interrupt.

use std::sync::{
    Mutex,
    atomic::{AtomicBool, Ordering},
};

use serde::{Deserialize, Serialize};

use crate::snapshot::{Snapshot, SnapshotError, decode, encode};

/// Interrupt IDs the Tegra X1 distributor implements (`GICD_TYPER.ITLinesNumber` = 6).
pub const INTERRUPT_COUNT: u32 = 224;
/// ID `GICC_IAR` and `GICC_HPPIR` return when there is no interrupt to report.
pub const SPURIOUS_INTID: u32 = 1023;

const SGI_COUNT: usize = 16;
/// The GIC-400 has no PPI inputs below this ID.
const FIRST_PPI: usize = 25;
/// IDs below this are banked per CPU interface.
const PRIVATE_COUNT: usize = 32;
const INTID_LIMIT: usize = INTERRUPT_COUNT as usize;
const SPI_COUNT: usize = INTID_LIMIT - PRIVATE_COUNT;
const MAX_CORES: usize = 8;

/// Implemented bits of a priority value in the Non-secure view.
const PRIORITY_BITS: u8 = 0xf0;
/// Position of the preemption level within a Non-secure priority value.
const PRIORITY_SHIFT: u32 = 4;
const MIN_BINARY_POINT: u8 = 3;

const CONTROL_ENABLE_GROUP1: u32 = 1 << 0;
const CONTROL_FIQ_BYPASS_DISABLE_GROUP1: u32 = 1 << 5;
const CONTROL_IRQ_BYPASS_DISABLE_GROUP1: u32 = 1 << 6;
const CONTROL_EOI_MODE_NS: u32 = 1 << 9;
const CONTROL_BITS: u32 = CONTROL_ENABLE_GROUP1
    | CONTROL_FIQ_BYPASS_DISABLE_GROUP1
    | CONTROL_IRQ_BYPASS_DISABLE_GROUP1
    | CONTROL_EOI_MODE_NS;

/// Interrupt ID field of `GICC_IAR`, `GICC_EOIR`, `GICC_DIR` and `GICC_HPPIR`.
const INTID_FIELD: u32 = 0x3ff;
/// Position of the SGI source CPU in the same registers.
const SOURCE_SHIFT: u32 = 10;
/// IDs from here up are the special IDs, never a real interrupt.
const FIRST_SPECIAL_INTID: usize = 1020;

/// `GICD_TYPER` fields that do not depend on the configuration (TRM table 3-2 note d).
const TYPER_LOCKABLE_SPIS: u32 = 31 << 11;
const TYPER_SECURITY_EXTENSIONS: u32 = 1 << 10;

const DISTRIBUTOR_IIDR: u32 = 0x0200_143b;
const CPU_INTERFACE_IIDR: u32 = 0x0202_143b;
/// `GICD_PIDR4`-`GICD_PIDR7`, `GICD_PIDR0`-`GICD_PIDR3`, `GICD_CIDR0`-`GICD_CIDR3` (TRM table 3-2).
const IDENTIFICATION: [u32; 12] = [
    0x04, 0x00, 0x00, 0x00, 0x90, 0xb4, 0x2b, 0x00, 0x0d, 0xf0, 0x05, 0xb1,
];

/// Distributor register offsets.
mod gicd {
    pub const CTLR: usize = 0x000;
    pub const TYPER: usize = 0x004;
    pub const IIDR: usize = 0x008;
    pub const IGROUPR: usize = 0x080;
    pub const ISENABLER: usize = 0x100;
    pub const ICENABLER: usize = 0x180;
    pub const ISPENDR: usize = 0x200;
    pub const ICPENDR: usize = 0x280;
    pub const ISACTIVER: usize = 0x300;
    pub const ICACTIVER: usize = 0x380;
    pub const IPRIORITYR: usize = 0x400;
    pub const IPRIORITYR_END: usize = IPRIORITYR + super::FIRST_SPECIAL_INTID;
    pub const ITARGETSR: usize = 0x800;
    pub const ITARGETSR_END: usize = ITARGETSR + super::FIRST_SPECIAL_INTID;
    pub const ICFGR: usize = 0xc00;
    pub const PPISR: usize = 0xd00;
    pub const SPISR: usize = 0xd04;
    pub const SPISR_END: usize = 0xd40;
    pub const SGIR: usize = 0xf00;
    pub const CPENDSGIR: usize = 0xf10;
    pub const SPENDSGIR: usize = 0xf20;
    pub const SPENDSGIR_END: usize = 0xf30;
    pub const IDENTIFICATION: usize = 0xfd0;
    pub const END: usize = 0x1000;
}

/// CPU interface register offsets.
mod gicc {
    pub const CTLR: usize = 0x00;
    pub const PMR: usize = 0x04;
    pub const BPR: usize = 0x08;
    pub const IAR: usize = 0x0c;
    pub const EOIR: usize = 0x10;
    pub const RPR: usize = 0x14;
    pub const HPPIR: usize = 0x18;
    pub const ABPR: usize = 0x1c;
    pub const AIAR: usize = 0x20;
    pub const AEOIR: usize = 0x24;
    pub const AHPPIR: usize = 0x28;
    pub const APR0: usize = 0xd0;
    pub const NSAPR0: usize = 0xe0;
    pub const IIDR: usize = 0xfc;
    pub const DIR: usize = 0x1000;
}

/// Distributor state of one peripheral interrupt, or of one SGI on one CPU
/// interface (which uses only `enabled`, `active` and `priority`).
#[derive(Clone, Copy, Default, Serialize, Deserialize)]
struct Interrupt {
    enabled: bool,
    /// Pending state that outlives the input: a captured edge or a
    /// `GICD_ISPENDRn` write. Cleared by acknowledgement and `GICD_ICPENDRn`.
    latched: bool,
    active: bool,
    /// Level of the input from the device.
    line: bool,
    edge_triggered: bool,
    /// Non-secure view of the priority.
    priority: u8,
    /// CPU interfaces an SPI is forwarded to.
    targets: u8,
}

impl Interrupt {
    /// Pending status while the distributor forwards interrupts: the latch, or
    /// the input itself when level-sensitive (specification figure 4-10).
    fn pending(&self) -> bool {
        self.latched || (self.line && !self.edge_triggered)
    }
}

/// One CPU interface and the distributor state banked for its processor.
#[derive(Serialize, Deserialize)]
struct Cpu {
    /// Non-secure `GICC_CTLR`.
    control: u32,
    priority_mask: u8,
    binary_point: u8,
    /// `GICC_APR0`: bit `n` is set while an interrupt acknowledged at group
    /// priority `n << PRIORITY_SHIFT` has not had its priority dropped.
    active_priorities: u16,
    /// Interrupts 0-31; entries 16-24 do not exist and stay disabled.
    private: [Interrupt; PRIVATE_COUNT],
    /// Per SGI, the CPUs whose request for it is pending here.
    sgi_sources: [u8; SGI_COUNT],
}

impl Cpu {
    fn new() -> Self {
        let mut private = [Interrupt::default(); PRIVATE_COUNT];
        for sgi in &mut private[..SGI_COUNT] {
            sgi.enabled = true;
        }
        Self {
            control: 0,
            priority_mask: 0,
            binary_point: MIN_BINARY_POINT,
            active_priorities: 0,
            private,
            sgi_sources: [0; SGI_COUNT],
        }
    }

    /// The part of `priority` that decides preemption.
    fn group_priority(&self, priority: u8) -> u8 {
        priority & !(0xff >> (7 - self.binary_point))
    }

    /// Group priority of the highest-priority interrupt being handled, `None` when idle.
    fn running_priority(&self) -> Option<u8> {
        // Bit `n` stands for priority `n << PRIORITY_SHIFT`. With no bit set
        // the count is 16, whose priority does not fit the byte.
        let level = self.active_priorities.trailing_zeros();
        u8::try_from(level << PRIORITY_SHIFT).ok()
    }

    /// Whether an interrupt of `priority` passes the mask and preempts what is running.
    fn permits(&self, priority: u8) -> bool {
        priority < self.priority_mask
            && self
                .running_priority()
                .is_none_or(|running| self.group_priority(priority) < running)
    }
}

/// A field of one of the byte-accessible distributor registers.
#[derive(Clone, Copy)]
enum ByteField {
    Priority(usize),
    Targets(usize),
    ClearSgiPending(usize),
    SetSgiPending(usize),
}

impl ByteField {
    fn at(offset: usize) -> Option<Self> {
        Some(match offset {
            gicd::IPRIORITYR..gicd::IPRIORITYR_END => Self::Priority(offset - gicd::IPRIORITYR),
            gicd::ITARGETSR..gicd::ITARGETSR_END => Self::Targets(offset - gicd::ITARGETSR),
            gicd::CPENDSGIR..gicd::SPENDSGIR => Self::ClearSgiPending(offset - gicd::CPENDSGIR),
            gicd::SPENDSGIR..gicd::SPENDSGIR_END => Self::SetSgiPending(offset - gicd::SPENDSGIR),
            _ => return None,
        })
    }
}

/// Builds a register value from a predicate over its low `width` bit positions.
fn pack_bits(width: usize, bit: impl Fn(usize) -> bool) -> u32 {
    (0..width)
        .filter(|&index| bit(index))
        .fold(0, |value, index| value | 1 << index)
}

/// First of the 32 interrupt IDs described by the word at `offset` of an
/// enable, pending or active register; each of those blocks spans 0x80 bytes.
fn bitmap_first_intid(offset: usize) -> usize {
    offset % 0x80 * 8
}

#[derive(Serialize, Deserialize)]
struct State {
    /// Non-secure `GICD_CTLR.EnableGrp1`.
    forwarding: bool,
    cpus: Box<[Cpu]>,
    #[serde(with = "crate::snapshot::array")]
    spis: [Interrupt; SPI_COUNT],
}

impl State {
    fn new(core_count: usize) -> Self {
        Self {
            forwarding: false,
            cpus: (0..core_count).map(|_| Cpu::new()).collect(),
            spis: [Interrupt::default(); SPI_COUNT],
        }
    }

    /// One bit per implemented CPU interface.
    fn cpu_mask(&self) -> u8 {
        u8::MAX >> (MAX_CORES - self.cpus.len())
    }

    /// `core`'s view of interrupt `intid`, `None` if the ID is not implemented.
    fn interrupt(&self, core: usize, intid: usize) -> Option<&Interrupt> {
        match intid {
            0..SGI_COUNT | FIRST_PPI..PRIVATE_COUNT => Some(&self.cpus[core].private[intid]),
            PRIVATE_COUNT..INTID_LIMIT => Some(&self.spis[intid - PRIVATE_COUNT]),
            _ => None,
        }
    }

    fn interrupt_mut(&mut self, core: usize, intid: usize) -> Option<&mut Interrupt> {
        match intid {
            0..SGI_COUNT | FIRST_PPI..PRIVATE_COUNT => Some(&mut self.cpus[core].private[intid]),
            PRIVATE_COUNT..INTID_LIMIT => Some(&mut self.spis[intid - PRIVATE_COUNT]),
            _ => None,
        }
    }

    fn spi_mut(&mut self, intid: usize) -> Option<&mut Interrupt> {
        self.spis.get_mut(intid.checked_sub(PRIVATE_COUNT)?)
    }

    /// Pending status as the pending registers report it, which includes
    /// interrupts that are also active.
    fn includes_pending(&self, core: usize, intid: usize) -> bool {
        match self.cpus[core].sgi_sources.get(intid) {
            Some(&sources) => sources != 0,
            // TRM 2.8: a disabled distributor hides level-sensitive inputs here.
            None => self.interrupt(core, intid).is_some_and(|interrupt| {
                interrupt.latched || (self.forwarding && interrupt.pending())
            }),
        }
    }

    /// Priority and ID of the interrupt the distributor forwards to `core`:
    /// among those enabled, pending, not active and targeted at it, the one
    /// with the lowest priority value, then the lowest ID.
    fn highest_pending(&self, core: usize) -> Option<(u8, usize)> {
        if !self.forwarding {
            return None;
        }
        let cpu = &self.cpus[core];
        let private = cpu
            .private
            .iter()
            .enumerate()
            .filter(|&(intid, interrupt)| match cpu.sgi_sources.get(intid) {
                Some(&sources) => sources != 0,
                None => interrupt.pending(),
            });
        let shared = self
            .spis
            .iter()
            .enumerate()
            .filter(|(_, spi)| spi.targets & (1 << core) != 0 && spi.pending())
            .map(|(index, spi)| (PRIVATE_COUNT + index, spi));
        private
            .chain(shared)
            .filter(|(_, interrupt)| interrupt.enabled && !interrupt.active)
            .map(|(intid, interrupt)| (interrupt.priority, intid))
            .min()
    }

    /// The interrupt `core`'s CPU interface is signaling to its processor.
    fn signaled(&self, core: usize) -> Option<(u8, usize)> {
        let cpu = &self.cpus[core];
        if cpu.control & CONTROL_ENABLE_GROUP1 == 0 {
            return None;
        }
        self.highest_pending(core)
            .filter(|&(priority, _)| cpu.permits(priority))
    }

    /// `intid` as `GICC_IAR` and `GICC_HPPIR` present it to `core`: an SGI
    /// carries the lowest-numbered CPU whose request is pending.
    fn identify(&self, core: usize, intid: usize) -> u32 {
        let source = self.cpus[core]
            .sgi_sources
            .get(intid)
            .map_or(0, |sources| sources.trailing_zeros());
        intid as u32 | source << SOURCE_SHIFT
    }

    /// Read of `GICC_IAR`: the signaled interrupt becomes active and loses the
    /// pending state this acknowledgement accounts for.
    fn acknowledge(&mut self, core: usize) -> u32 {
        let Some((priority, intid)) = self.signaled(core) else {
            return SPURIOUS_INTID;
        };
        let value = self.identify(core, intid);
        let cpu = &mut self.cpus[core];
        cpu.active_priorities |= 1 << (cpu.group_priority(priority) >> PRIORITY_SHIFT);
        if let Some(sources) = cpu.sgi_sources.get_mut(intid) {
            *sources &= *sources - 1;
        }
        let interrupt = self
            .interrupt_mut(core, intid)
            .expect("only implemented interrupts are forwarded");
        interrupt.latched = false;
        interrupt.active = true;
        value
    }

    /// The active interrupt a `GICC_EOIR` or `GICC_DIR` write of `intid`
    /// completes. `None` is a write the specification makes UNPREDICTABLE:
    /// the interrupt does not exist or is not active.
    fn completed_interrupt(&mut self, core: usize, intid: usize) -> Option<&mut Interrupt> {
        self.interrupt_mut(core, intid)
            .filter(|interrupt| interrupt.active)
    }

    /// Write of `GICC_EOIR`: drops the running priority to that of the
    /// interrupt preempted by the one being completed, and deactivates the
    /// completed interrupt unless `EOImodeNS` leaves that to `GICC_DIR`.
    fn end_of_interrupt(&mut self, core: usize, value: u32) -> Option<()> {
        let intid = (value & INTID_FIELD) as usize;
        // The controller ignores the special IDs.
        if intid >= FIRST_SPECIAL_INTID {
            return Some(());
        }
        let cpu = &self.cpus[core];
        let split = cpu.control & CONTROL_EOI_MODE_NS != 0;
        // With no acknowledged interrupt outstanding the write is UNPREDICTABLE.
        let remaining = cpu.active_priorities & cpu.active_priorities.checked_sub(1)?;
        let interrupt = self.completed_interrupt(core, intid)?;
        if !split {
            interrupt.active = false;
        }
        self.cpus[core].active_priorities = remaining;
        Some(())
    }

    /// Write of `GICC_DIR`, UNPREDICTABLE unless `EOImodeNS` is set.
    fn deactivate(&mut self, core: usize, value: u32) -> Option<()> {
        if self.cpus[core].control & CONTROL_EOI_MODE_NS == 0 {
            return None;
        }
        let intid = (value & INTID_FIELD) as usize;
        if intid < FIRST_SPECIAL_INTID {
            self.completed_interrupt(core, intid)?.active = false;
        }
        Some(())
    }

    /// Write of `GICD_SGIR` by `core`.
    fn generate_sgi(&mut self, core: usize, value: u32) -> Option<()> {
        let requester = 1 << core;
        let targets = match (value >> 24) & 0b11 {
            0b00 => (value >> 16) as u8 & self.cpu_mask(),
            0b01 => self.cpu_mask() & !requester,
            0b10 => requester,
            _ => return None,
        };
        // TRM 2.8: SGIs cannot be made pending while the distributor is disabled.
        if self.forwarding {
            let sgi = (value & 0xf) as usize;
            for (index, cpu) in self.cpus.iter_mut().enumerate() {
                if targets & (1 << index) != 0 {
                    cpu.sgi_sources[sgi] |= requester;
                }
            }
        }
        Some(())
    }

    fn set_line(&mut self, core: usize, intid: usize, asserted: bool) {
        let forwarding = self.forwarding;
        let interrupt = match intid {
            FIRST_PPI..PRIVATE_COUNT => &mut self.cpus[core].private[intid],
            PRIVATE_COUNT..INTID_LIMIT => &mut self.spis[intid - PRIVATE_COUNT],
            _ => panic!("interrupt {intid} has no input line"),
        };
        // TRM 2.8: an edge cannot set the pending state while the distributor is disabled.
        if asserted && !interrupt.line && interrupt.edge_triggered && forwarding {
            interrupt.latched = true;
        }
        interrupt.line = asserted;
    }

    fn read_byte_field(&self, core: usize, field: ByteField) -> u8 {
        match field {
            ByteField::Priority(intid) => self
                .interrupt(core, intid)
                .map_or(0, |interrupt| interrupt.priority),
            ByteField::Targets(intid) => match self.interrupt(core, intid) {
                Some(spi) if intid >= PRIVATE_COUNT => spi.targets,
                // A private interrupt targets exactly the processor reading it.
                Some(_) => 1 << core,
                None => 0,
            },
            ByteField::ClearSgiPending(sgi) | ByteField::SetSgiPending(sgi) => {
                self.cpus[core].sgi_sources[sgi]
            }
        }
    }

    fn write_byte_field(&mut self, core: usize, field: ByteField, value: u8) {
        let cpu_mask = self.cpu_mask();
        match field {
            ByteField::Priority(intid) => {
                if let Some(interrupt) = self.interrupt_mut(core, intid) {
                    interrupt.priority = value & PRIORITY_BITS;
                }
            }
            // The targets of private interrupts are read-only.
            ByteField::Targets(intid) => {
                if let Some(spi) = self.spi_mut(intid) {
                    spi.targets = value & cpu_mask;
                }
            }
            ByteField::ClearSgiPending(sgi) => self.cpus[core].sgi_sources[sgi] &= !value,
            ByteField::SetSgiPending(sgi) => self.cpus[core].sgi_sources[sgi] |= value & cpu_mask,
        }
    }

    /// Applies `update` to every implemented interrupt from `first_writable`
    /// up that `value` selects in the bitmap register word at `offset`.
    fn update_selected(
        &mut self,
        core: usize,
        offset: usize,
        value: u32,
        first_writable: usize,
        update: impl Fn(&mut Interrupt),
    ) {
        let first = bitmap_first_intid(offset);
        for intid in (0..32)
            .filter(|index| value & (1 << index) != 0)
            .map(|index| first + index)
        {
            if intid >= first_writable
                && let Some(interrupt) = self.interrupt_mut(core, intid)
            {
                update(interrupt);
            }
        }
    }

    /// `GICD_ICFGRn`, two bits per interrupt: bit 1 selects edge-triggered,
    /// bit 0 is the read-only legacy bit that reads as one for peripherals.
    fn read_configuration(&self, core: usize, register: usize) -> u32 {
        (0..16).fold(0, |value, field| {
            let intid = register * 16 + field;
            let config = match self.interrupt(core, intid) {
                Some(_) if intid < SGI_COUNT => 0b10,
                Some(interrupt) => 0b01 | u32::from(interrupt.edge_triggered) << 1,
                None => 0b00,
            };
            value | config << (2 * field)
        })
    }

    fn write_configuration(&mut self, register: usize, value: u32) {
        for field in 0..16 {
            if let Some(spi) = self.spi_mut(register * 16 + field) {
                spi.edge_triggered = value & (0b10 << (2 * field)) != 0;
            }
        }
    }

    fn distributor_read(&self, core: usize, offset: usize, size: usize) -> Option<u32> {
        if !matches!(size, 1 | 2 | 4) || !offset.is_multiple_of(size) {
            return None;
        }
        if ByteField::at(offset).is_some() {
            // An aligned access stays inside its block: the blocks start and end on word boundaries.
            return (0..size).try_fold(0, |value, index| {
                let byte = self.read_byte_field(core, ByteField::at(offset + index)?);
                Some(value | u32::from(byte) << (8 * index))
            });
        }
        if size != 4 {
            return None;
        }
        let bitmap = |bit: &dyn Fn(usize) -> bool| {
            let first = bitmap_first_intid(offset);
            pack_bits(32, |index| bit(first + index))
        };
        Some(match offset {
            gicd::CTLR => u32::from(self.forwarding),
            gicd::TYPER => {
                let cpu_number = self.cpus.len() as u32 - 1;
                TYPER_LOCKABLE_SPIS
                    | TYPER_SECURITY_EXTENSIONS
                    | cpu_number << 5
                    | (INTERRUPT_COUNT / 32 - 1)
            }
            gicd::IIDR => DISTRIBUTOR_IIDR,
            gicd::IGROUPR..gicd::ISENABLER => 0,
            gicd::ISENABLER..gicd::ISPENDR => bitmap(&|intid| {
                self.interrupt(core, intid)
                    .is_some_and(|interrupt| interrupt.enabled)
            }),
            gicd::ISPENDR..gicd::ISACTIVER => bitmap(&|intid| self.includes_pending(core, intid)),
            gicd::ISACTIVER..gicd::IPRIORITYR => bitmap(&|intid| {
                self.interrupt(core, intid)
                    .is_some_and(|interrupt| interrupt.active)
            }),
            gicd::ICFGR..gicd::PPISR => self.read_configuration(core, (offset - gicd::ICFGR) / 4),
            gicd::PPISR => {
                let private = &self.cpus[core].private;
                pack_bits(SGI_COUNT, |bit| private[SGI_COUNT + bit].line)
            }
            gicd::SPISR..gicd::SPISR_END => {
                let first = (offset - gicd::SPISR) * 8;
                pack_bits(32, |bit| {
                    self.spis.get(first + bit).is_some_and(|spi| spi.line)
                })
            }
            gicd::IDENTIFICATION..gicd::END => IDENTIFICATION[(offset - gicd::IDENTIFICATION) / 4],
            _ => return None,
        })
    }

    fn distributor_write(
        &mut self,
        core: usize,
        offset: usize,
        size: usize,
        value: u32,
    ) -> Option<()> {
        if !matches!(size, 1 | 2 | 4) || !offset.is_multiple_of(size) {
            return None;
        }
        if ByteField::at(offset).is_some() {
            for index in 0..size {
                let byte = value.to_le_bytes()[index];
                self.write_byte_field(core, ByteField::at(offset + index)?, byte);
            }
            return Some(());
        }
        if size != 4 {
            return None;
        }
        match offset {
            gicd::CTLR => self.forwarding = value & 1 != 0,
            // Read-only registers, and the group registers that only Secure accesses reach.
            gicd::TYPER
            | gicd::IIDR
            | gicd::IGROUPR..gicd::ISENABLER
            | gicd::PPISR..gicd::SPISR_END
            | gicd::IDENTIFICATION..gicd::END => {}
            // The enable and pending bits of SGIs ignore writes; SGIs have
            // their own pending registers.
            gicd::ISENABLER..gicd::ICENABLER => {
                self.update_selected(core, offset, value, SGI_COUNT, |interrupt| {
                    interrupt.enabled = true;
                });
            }
            gicd::ICENABLER..gicd::ISPENDR => {
                self.update_selected(core, offset, value, SGI_COUNT, |interrupt| {
                    interrupt.enabled = false;
                });
            }
            gicd::ISPENDR..gicd::ICPENDR => {
                self.update_selected(core, offset, value, SGI_COUNT, |interrupt| {
                    interrupt.latched = true;
                });
            }
            gicd::ICPENDR..gicd::ISACTIVER => {
                self.update_selected(core, offset, value, SGI_COUNT, |interrupt| {
                    interrupt.latched = false;
                });
            }
            gicd::ISACTIVER..gicd::ICACTIVER => {
                self.update_selected(core, offset, value, 0, |interrupt| interrupt.active = true);
            }
            gicd::ICACTIVER..gicd::IPRIORITYR => {
                self.update_selected(core, offset, value, 0, |interrupt| interrupt.active = false);
            }
            gicd::ICFGR..gicd::PPISR => {
                self.write_configuration((offset - gicd::ICFGR) / 4, value);
            }
            gicd::SGIR => self.generate_sgi(core, value)?,
            _ => return None,
        }
        Some(())
    }

    fn cpu_interface_read(&mut self, core: usize, offset: usize, size: usize) -> Option<u32> {
        if size != 4 {
            return None;
        }
        let cpu = &self.cpus[core];
        Some(match offset {
            gicc::CTLR => cpu.control,
            gicc::PMR => u32::from(cpu.priority_mask),
            gicc::BPR => u32::from(cpu.binary_point),
            gicc::IAR => self.acknowledge(core),
            // Idle reads as the lowest priority the Non-secure view can express.
            gicc::RPR => u32::from(cpu.running_priority().unwrap_or(PRIORITY_BITS)),
            gicc::HPPIR => self
                .highest_pending(core)
                .map_or(SPURIOUS_INTID, |(_, intid)| self.identify(core, intid)),
            gicc::ABPR | gicc::AIAR | gicc::AEOIR | gicc::AHPPIR | gicc::NSAPR0 => 0,
            gicc::APR0 => u32::from(cpu.active_priorities),
            gicc::IIDR => CPU_INTERFACE_IIDR,
            _ => return None,
        })
    }

    fn cpu_interface_write(
        &mut self,
        core: usize,
        offset: usize,
        size: usize,
        value: u32,
    ) -> Option<()> {
        if size != 4 {
            return None;
        }
        let cpu = &mut self.cpus[core];
        match offset {
            gicc::CTLR => cpu.control = value & CONTROL_BITS,
            gicc::PMR => cpu.priority_mask = value as u8 & PRIORITY_BITS,
            gicc::BPR => cpu.binary_point = (value as u8 & 0b111).max(MIN_BINARY_POINT),
            gicc::EOIR => self.end_of_interrupt(core, value)?,
            // Read-only registers, and the aliases that only Secure accesses reach.
            gicc::IAR
            | gicc::RPR
            | gicc::HPPIR
            | gicc::ABPR
            | gicc::AIAR
            | gicc::AEOIR
            | gicc::AHPPIR
            | gicc::NSAPR0
            | gicc::IIDR => {}
            gicc::APR0 => cpu.active_priorities = value as u16,
            gicc::DIR => self.deactivate(core, value)?,
            _ => return None,
        }
        Some(())
    }
}

/// The interrupt controller. Shared between the vCPU threads, which access its
/// registers, and the device models, which drive its input lines.
///
/// Every method that takes a `core` panics if it is not below the core count.
pub struct Gic {
    state: Mutex<State>,
    /// IRQ output per core. Written with `state` locked, so the values and the
    /// order of `irq_raised` calls follow the order of state changes; read
    /// without the lock because every vCPU entry samples it.
    outputs: Box<[AtomicBool]>,
    irq_raised: Box<dyn Fn(usize) + Send + Sync>,
}

impl Gic {
    /// Creates the controller as it is before the kernel programs it: nothing
    /// enabled, pending or active, and every interrupt in Group 1.
    ///
    /// `irq_raised(core)` is invoked, with no internal lock held, every time
    /// the IRQ output to `core` has gone from deasserted to asserted.
    ///
    /// # Panics
    /// If `core_count` is not between 1 and 8, the range GICv2 supports.
    pub fn new(core_count: usize, irq_raised: Box<dyn Fn(usize) + Send + Sync>) -> Self {
        assert!(
            (1..=MAX_CORES).contains(&core_count),
            "a GICv2 has 1 to 8 CPU interfaces, not {core_count}"
        );
        Self {
            state: Mutex::new(State::new(core_count)),
            outputs: (0..core_count).map(|_| AtomicBool::new(false)).collect(),
            irq_raised,
        }
    }

    /// Runs `access` on the state, then brings the IRQ outputs up to date and
    /// reports the ones that rose.
    fn update<R>(&self, access: impl FnOnce(&mut State) -> R) -> R {
        let mut raised = 0u8;
        let result = {
            let mut state = self.state.lock().unwrap();
            let result = access(&mut state);
            for (core, output) in self.outputs.iter().enumerate() {
                let level = state.signaled(core).is_some();
                if !output.swap(level, Ordering::SeqCst) && level {
                    raised |= 1 << core;
                }
            }
            result
        };
        (0..self.outputs.len())
            .filter(|core| raised & (1 << core) != 0)
            .for_each(|core| (self.irq_raised)(core));
        result
    }

    /// MMIO read of the distributor (`GICD_*`) register at `offset` by `core`,
    /// which selects the banked registers of interrupts 0-31. `size` is the
    /// access width in bytes. `None` means the access is not modeled.
    pub fn distributor_read(&self, core: usize, offset: u64, size: u8) -> Option<u32> {
        let offset = usize::try_from(offset).ok()?;
        self.state
            .lock()
            .unwrap()
            .distributor_read(core, offset, usize::from(size))
    }

    /// MMIO write to the distributor; see [`Self::distributor_read`].
    pub fn distributor_write(&self, core: usize, offset: u64, size: u8, value: u32) -> Option<()> {
        let offset = usize::try_from(offset).ok()?;
        self.update(|state| state.distributor_write(core, offset, usize::from(size), value))
    }

    /// MMIO read of `core`'s CPU interface (`GICC_*`) register at `offset`.
    /// Reading `GICC_IAR` acknowledges the signaled interrupt. `None` means
    /// the access is not modeled.
    pub fn cpu_interface_read(&self, core: usize, offset: u64, size: u8) -> Option<u32> {
        let offset = usize::try_from(offset).ok()?;
        self.update(|state| state.cpu_interface_read(core, offset, usize::from(size)))
    }

    /// MMIO write to `core`'s CPU interface; see [`Self::cpu_interface_read`].
    pub fn cpu_interface_write(
        &self,
        core: usize,
        offset: u64,
        size: u8,
        value: u32,
    ) -> Option<()> {
        let offset = usize::try_from(offset).ok()?;
        self.update(|state| state.cpu_interface_write(core, offset, usize::from(size), value))
    }

    /// Drives the input line of interrupt `intid` from a device model. PPIs
    /// (25..=31) are per `core`; for SPIs (32..[`INTERRUPT_COUNT`]) `core` is
    /// ignored. A level-sensitive interrupt is pending while its line is
    /// asserted; an edge-triggered one becomes pending when the line goes
    /// from deasserted to asserted.
    ///
    /// # Panics
    /// If `intid` has no input: SGIs, the PPIs the GIC-400 lacks (16..=24)
    /// and IDs from [`INTERRUPT_COUNT`] up.
    pub fn set_line(&self, core: usize, intid: u32, asserted: bool) {
        self.update(|state| state.set_line(core, intid as usize, asserted));
    }

    /// Current level of the IRQ signal to `core`'s processor.
    pub fn irq_asserted(&self, core: usize) -> bool {
        self.outputs[core].load(Ordering::SeqCst)
    }
}

/// Saves the distributor and every CPU interface (each interrupt's enable,
/// latched pending, active, priority, targets and trigger mode, and the level
/// of its input line), and the IRQ level last driven to each core.
///
/// The IRQ outputs are put back as they were, without reporting them through
/// `irq_raised`: the restored cores sample them on entry, as the saved ones
/// did.
impl Snapshot for Gic {
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        let state = self.state.lock().unwrap();
        let outputs = self
            .outputs
            .iter()
            .map(|output| output.load(Ordering::SeqCst))
            .collect::<Vec<_>>();
        encode(&(&*state, outputs))
    }

    fn restore(&self, bytes: &[u8]) -> Result<(), SnapshotError> {
        let (saved, outputs): (State, Vec<bool>) = decode("GIC", bytes)?;
        if saved.cpus.len() != self.outputs.len() || outputs.len() != self.outputs.len() {
            return Err(SnapshotError::Corrupt(format!(
                "GIC: {} CPU interfaces saved, {} here",
                saved.cpus.len(),
                self.outputs.len()
            )));
        }
        let mut state = self.state.lock().unwrap();
        *state = saved;
        for (output, level) in self.outputs.iter().zip(outputs) {
            output.store(level, Ordering::SeqCst);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    const GICD_CTLR: u64 = 0x000;
    const GICD_TYPER: u64 = 0x004;
    const GICD_IGROUPR: u64 = 0x080;
    const GICD_ISENABLER: u64 = 0x100;
    const GICD_ICENABLER: u64 = 0x180;
    const GICD_ISPENDR: u64 = 0x200;
    const GICD_ICPENDR: u64 = 0x280;
    const GICD_ISACTIVER: u64 = 0x300;
    const GICD_ICACTIVER: u64 = 0x380;
    const GICD_IPRIORITYR: u64 = 0x400;
    const GICD_ITARGETSR: u64 = 0x800;
    const GICD_ICFGR: u64 = 0xc00;
    const GICD_PPISR: u64 = 0xd00;
    const GICD_SPISR: u64 = 0xd04;
    const GICD_SGIR: u64 = 0xf00;
    const GICD_CPENDSGIR: u64 = 0xf10;
    const GICD_SPENDSGIR: u64 = 0xf20;

    const GICC_CTLR: u64 = 0x00;
    const GICC_PMR: u64 = 0x04;
    const GICC_BPR: u64 = 0x08;
    const GICC_IAR: u64 = 0x0c;
    const GICC_EOIR: u64 = 0x10;
    const GICC_RPR: u64 = 0x14;
    const GICC_HPPIR: u64 = 0x18;
    const GICC_APR0: u64 = 0xd0;
    const GICC_DIR: u64 = 0x1000;

    const TIMER_PPI: u32 = 30;
    const ALL: u32 = 0xffff_ffff;

    const _: fn() = || {
        fn shared_between_threads<T: Send + Sync>() {}
        shared_between_threads::<Gic>();
    };

    struct Bench {
        gic: Gic,
        raised: Arc<Mutex<Vec<usize>>>,
    }

    impl Bench {
        /// Four cores, nothing programmed.
        fn reset() -> Self {
            let raised = Arc::new(Mutex::new(Vec::new()));
            let gic = Gic::new(4, {
                let raised = Arc::clone(&raised);
                Box::new(move |core| raised.lock().unwrap().push(core))
            });
            Self { gic, raised }
        }

        /// Distributor and all CPU interfaces enabled with the mask fully
        /// open, as `Gic_InitializeCoreInterface` leaves them (`0x800d50b4`-`0x800d50e8`).
        fn enabled() -> Self {
            let bench = Self::reset();
            bench.d_write(0, GICD_CTLR, 1);
            for core in 0..4 {
                bench.c_write(core, GICC_CTLR, 1);
                bench.c_write(core, GICC_PMR, 0xff);
            }
            bench
        }

        fn d_read(&self, core: usize, offset: u64) -> u32 {
            self.gic.distributor_read(core, offset, 4).unwrap()
        }

        fn d_write(&self, core: usize, offset: u64, value: u32) {
            self.gic.distributor_write(core, offset, 4, value).unwrap();
        }

        fn c_read(&self, core: usize, offset: u64) -> u32 {
            self.gic.cpu_interface_read(core, offset, 4).unwrap()
        }

        fn c_write(&self, core: usize, offset: u64, value: u32) {
            self.gic
                .cpu_interface_write(core, offset, 4, value)
                .unwrap();
        }

        fn set_priority(&self, core: usize, intid: u32, priority: u8) {
            let offset = GICD_IPRIORITYR + u64::from(intid);
            self.gic
                .distributor_write(core, offset, 1, priority.into())
                .unwrap();
        }

        fn enable(&self, core: usize, intid: u32) {
            self.d_write(
                core,
                GICD_ISENABLER + u64::from(intid / 32) * 4,
                1 << (intid % 32),
            );
        }

        /// Gives an SPI a priority and targets, and enables it.
        fn route_spi(&self, intid: u32, priority: u8, targets: u8) {
            self.set_priority(0, intid, priority);
            let offset = GICD_ITARGETSR + u64::from(intid);
            self.gic
                .distributor_write(0, offset, 1, targets.into())
                .unwrap();
            self.enable(0, intid);
        }

        fn make_edge_triggered(&self, intid: u32) {
            self.d_write(
                0,
                GICD_ICFGR + u64::from(intid / 16) * 4,
                0b10 << (intid % 16 * 2),
            );
        }

        fn acknowledge(&self, core: usize) -> u32 {
            self.c_read(core, GICC_IAR)
        }

        fn end(&self, core: usize, value: u32) {
            self.c_write(core, GICC_EOIR, value);
        }

        fn pending_bit(&self, core: usize, intid: u32) -> bool {
            let word = self.d_read(core, GICD_ISPENDR + u64::from(intid / 32) * 4);
            word & (1 << (intid % 32)) != 0
        }

        fn asserted(&self) -> [bool; 4] {
            [0, 1, 2, 3].map(|core| self.gic.irq_asserted(core))
        }

        /// Cores reported through `irq_raised` since the last call.
        fn take_raised(&self) -> Vec<usize> {
            let mut raised = std::mem::take(&mut *self.raised.lock().unwrap());
            raised.sort_unstable();
            raised
        }
    }

    #[test]
    fn type_register_describes_tegra_x1() {
        // 31 lockable SPIs, Security Extensions, 4 CPUs, 7 * 32 interrupt IDs.
        assert_eq!(Bench::reset().d_read(0, GICD_TYPER), 0xfc66);
    }

    #[test]
    fn non_secure_view_implements_four_priority_bits() {
        let bench = Bench::enabled();
        bench.set_priority(0, 40, 0x3f);
        assert_eq!(
            bench.gic.distributor_read(0, GICD_IPRIORITYR + 40, 1),
            Some(0x30)
        );
        bench.d_write(0, GICD_IPRIORITYR + 44, ALL);
        assert_eq!(bench.d_read(0, GICD_IPRIORITYR + 44), 0xf0f0_f0f0);
        assert_eq!(bench.c_read(0, GICC_PMR), 0xf0);
        bench.c_write(0, GICC_BPR, 0);
        assert_eq!(bench.c_read(0, GICC_BPR), 3);
        assert_eq!(bench.c_read(0, GICC_RPR), 0xf0);

        // The lowest priority does not pass even the fully open mask.
        bench.route_spi(40, 0xff, 0b0001);
        bench.gic.set_line(0, 40, true);
        assert!(bench.pending_bit(0, 40));
        assert!(!bench.gic.irq_asserted(0));
        assert_eq!(bench.acknowledge(0), SPURIOUS_INTID);
    }

    #[test]
    fn private_targets_identify_the_reading_core() {
        let bench = Bench::reset();
        // How the kernel learns its CPU interface number (0x800d50bc).
        assert_eq!(bench.gic.distributor_read(3, GICD_ITARGETSR, 1), Some(0x08));
        assert_eq!(bench.d_read(2, GICD_ITARGETSR), 0x0404_0404);
        // IDs 16-24 do not exist.
        assert_eq!(bench.d_read(2, GICD_ITARGETSR + 16), 0);
        assert_eq!(bench.d_read(2, GICD_ITARGETSR + 24), 0x0404_0400);
        bench.d_write(2, GICD_ITARGETSR + 28, 0);
        assert_eq!(bench.d_read(2, GICD_ITARGETSR + 28), 0x0404_0404);
        // SPI targets are limited to the CPU interfaces that exist.
        bench.d_write(0, GICD_ITARGETSR + 32, 0xff);
        assert_eq!(bench.d_read(1, GICD_ITARGETSR + 32), 0x0f);
    }

    #[test]
    fn lowest_priority_value_wins_then_lowest_id() {
        let bench = Bench::enabled();
        bench.route_spi(50, 0x80, 0b0001);
        bench.route_spi(40, 0x80, 0b0001);
        bench.route_spi(60, 0x40, 0b0001);
        for intid in [40, 50, 60] {
            bench.gic.set_line(0, intid, true);
        }
        assert_eq!(bench.c_read(0, GICC_HPPIR), 60);
        for expected in [60, 40, 50] {
            assert_eq!(bench.acknowledge(0), expected);
            bench.gic.set_line(0, expected, false);
            bench.end(0, expected);
        }
        assert_eq!(bench.acknowledge(0), SPURIOUS_INTID);
    }

    #[test]
    fn running_priority_blocks_all_but_higher_groups() {
        let bench = Bench::enabled();
        bench.route_spi(40, 0x80, 0b0001);
        bench.route_spi(41, 0x90, 0b0001);
        bench.route_spi(42, 0x40, 0b0001);

        bench.gic.set_line(0, 40, true);
        assert_eq!(bench.acknowledge(0), 40);
        bench.gic.set_line(0, 40, false);
        assert_eq!(bench.c_read(0, GICC_RPR), 0x80);

        bench.gic.set_line(0, 41, true);
        assert!(!bench.gic.irq_asserted(0));
        assert_eq!(bench.acknowledge(0), SPURIOUS_INTID);
        assert_eq!(bench.c_read(0, GICC_HPPIR), 41);

        bench.gic.set_line(0, 42, true);
        assert!(bench.gic.irq_asserted(0));
        assert_eq!(bench.acknowledge(0), 42);
        bench.gic.set_line(0, 42, false);
        assert_eq!(bench.c_read(0, GICC_RPR), 0x40);
        assert_eq!(bench.c_read(0, GICC_APR0), 1 << 4 | 1 << 8);

        // Completing the preempting interrupt returns to the preempted
        // priority, which still blocks the lower one.
        bench.end(0, 42);
        assert_eq!(bench.c_read(0, GICC_RPR), 0x80);
        assert!(!bench.gic.irq_asserted(0));
        bench.end(0, 40);
        assert_eq!(bench.c_read(0, GICC_APR0), 0);
        assert_eq!(bench.acknowledge(0), 41);
    }

    #[test]
    fn priority_mask_gates_signaling() {
        let bench = Bench::enabled();
        bench.route_spi(40, 0x70, 0b0001);
        bench.gic.set_line(0, 40, true);
        assert!(bench.gic.irq_asserted(0));
        // An interrupt must be strictly higher than the mask.
        bench.c_write(0, GICC_PMR, 0x70);
        assert!(!bench.gic.irq_asserted(0));
        assert_eq!(bench.acknowledge(0), SPURIOUS_INTID);
        bench.c_write(0, GICC_PMR, 0x80);
        assert_eq!(bench.acknowledge(0), 40);
    }

    #[test]
    fn binary_point_selects_the_preempting_bits() {
        // Priority 0x80 arriving while 0x90 is being handled.
        let preempts = |binary_point| {
            let bench = Bench::enabled();
            bench.c_write(0, GICC_BPR, binary_point);
            bench.route_spi(40, 0x90, 0b0001);
            bench.route_spi(41, 0x80, 0b0001);
            bench.gic.set_line(0, 40, true);
            assert_eq!(bench.acknowledge(0), 40);
            bench.gic.set_line(0, 41, true);
            bench.gic.irq_asserted(0)
        };
        // Group priority is bits [7:4]: 0x80 against 0x90.
        assert!(preempts(3));
        // Group priority is bits [7:5]: both are 0x80.
        assert!(!preempts(4));

        // The kernel's setting leaves no group bits, so even the highest
        // priority waits for the end of the lowest.
        let bench = Bench::enabled();
        bench.c_write(0, GICC_BPR, 7);
        bench.route_spi(40, 0xb0, 0b0001);
        bench.route_spi(41, 0x00, 0b0001);
        bench.gic.set_line(0, 40, true);
        assert_eq!(bench.acknowledge(0), 40);
        bench.gic.set_line(0, 40, false);
        bench.gic.set_line(0, 41, true);
        assert!(!bench.gic.irq_asserted(0));
        bench.end(0, 40);
        assert_eq!(bench.acknowledge(0), 41);
    }

    #[test]
    fn level_sensitive_interrupt_follows_its_line() {
        let bench = Bench::enabled();
        bench.route_spi(40, 0x80, 0b0001);
        bench.gic.set_line(0, 40, true);
        assert_eq!(bench.acknowledge(0), 40);
        // Active and pending is not signaled again until deactivated.
        assert!(bench.pending_bit(0, 40));
        assert!(!bench.gic.irq_asserted(0));
        bench.end(0, 40);
        assert!(bench.gic.irq_asserted(0));
        assert_eq!(bench.acknowledge(0), 40);
        bench.gic.set_line(0, 40, false);
        bench.end(0, 40);
        assert!(!bench.gic.irq_asserted(0));

        // Withdrawn before it is acknowledged.
        bench.gic.set_line(0, 40, true);
        assert!(bench.gic.irq_asserted(0));
        bench.gic.set_line(0, 40, false);
        assert!(!bench.gic.irq_asserted(0));
        assert_eq!(bench.acknowledge(0), SPURIOUS_INTID);

        // Software-set pending state outlives the line, until acknowledged.
        bench.d_write(0, GICD_ISPENDR + 4, 1 << 8);
        assert_eq!(bench.acknowledge(0), 40);
        bench.end(0, 40);
        assert!(!bench.pending_bit(0, 40));
    }

    #[test]
    fn edge_triggered_interrupt_latches_rising_edges() {
        let bench = Bench::enabled();
        bench.make_edge_triggered(40);
        bench.route_spi(40, 0x80, 0b0001);
        bench.gic.set_line(0, 40, true);
        bench.gic.set_line(0, 40, false);
        assert!(bench.gic.irq_asserted(0));
        assert_eq!(bench.acknowledge(0), 40);
        assert!(!bench.pending_bit(0, 40));

        // An edge during handling is kept for after the end of interrupt.
        bench.gic.set_line(0, 40, true);
        assert!(!bench.gic.irq_asserted(0));
        bench.end(0, 40);
        assert_eq!(bench.acknowledge(0), 40);
        bench.end(0, 40);
        // The line is still high, which is not another edge.
        bench.gic.set_line(0, 40, true);
        assert!(!bench.gic.irq_asserted(0));

        bench.gic.set_line(0, 40, false);
        bench.gic.set_line(0, 40, true);
        bench.d_write(0, GICD_ICPENDR + 4, 1 << 8);
        assert!(!bench.gic.irq_asserted(0));
    }

    #[test]
    fn sgi_requests_are_pending_per_source() {
        let bench = Bench::enabled();
        // The form the kernel uses: ID | target list << 16 (0x800c9d6c).
        bench.d_write(0, GICD_SGIR, 3 | 0b1110 << 16);
        assert_eq!(bench.asserted(), [false, true, true, true]);
        assert_eq!(bench.acknowledge(1), 3);
        bench.end(1, 3);

        bench.d_write(3, GICD_SGIR, 3 | 0b0010 << 16);
        bench.d_write(2, GICD_SGIR, 3 | 0b0010 << 16);
        // SGI 3 is the fourth byte of the first register, banked per target.
        assert_eq!(
            bench.gic.distributor_read(1, GICD_SPENDSGIR + 3, 1),
            Some(0b1100)
        );
        assert_eq!(bench.d_read(0, GICD_SPENDSGIR), 0);
        let first = bench.acknowledge(1);
        assert_eq!(first, 3 | 2 << 10);
        assert!(!bench.gic.irq_asserted(1));
        bench.end(1, first);
        let second = bench.acknowledge(1);
        assert_eq!(second, 3 | 3 << 10);
        bench.end(1, second);
        assert!(!bench.gic.irq_asserted(1));
    }

    #[test]
    fn sgi_target_filters_and_pending_registers() {
        let bench = Bench::enabled();
        bench.d_write(2, GICD_SGIR, 0b01 << 24 | 5);
        assert_eq!(
            [0, 1, 2, 3].map(|core| bench.pending_bit(core, 5)),
            [true, true, false, true]
        );
        bench.d_write(2, GICD_SGIR, 0b10 << 24 | 6);
        assert_eq!(
            [0, 1, 2, 3].map(|core| bench.pending_bit(core, 6)),
            [false, false, true, false]
        );
        assert_eq!(
            bench.gic.distributor_write(2, GICD_SGIR, 4, 0b11 << 24),
            None
        );

        // SGIs are always enabled and are not cleared through GICD_ICPENDR0.
        bench.d_write(0, GICD_ICENABLER, ALL);
        bench.d_write(0, GICD_ICPENDR, ALL);
        assert_eq!(bench.d_read(0, GICD_ISENABLER), 0x0000_ffff);
        assert!(bench.gic.irq_asserted(0));
        assert_eq!(bench.c_read(0, GICC_HPPIR), 5 | 2 << 10);

        bench
            .gic
            .distributor_write(0, GICD_CPENDSGIR + 5, 1, 0xff)
            .unwrap();
        assert!(!bench.gic.irq_asserted(0));
        // Set-pending names the source and reaches only the writing core.
        bench.d_write(0, GICD_SPENDSGIR + 4, 0b1000 << 8);
        assert_eq!(bench.asserted(), [true, true, true, true]);
        assert_eq!(bench.acknowledge(0), 5 | 3 << 10);
    }

    #[test]
    fn spi_is_offered_to_every_target_and_acknowledged_once() {
        let bench = Bench::enabled();
        bench.route_spi(40, 0x80, 0b0110);
        bench.gic.set_line(0, 40, true);
        assert_eq!(bench.asserted(), [false, true, true, false]);
        assert_eq!(bench.acknowledge(2), 40);
        assert_eq!(bench.asserted(), [false; 4]);
        assert_eq!(bench.acknowledge(1), SPURIOUS_INTID);

        // Retargeting a pending interrupt moves where it is pending.
        bench.gic.set_line(0, 40, false);
        bench.end(2, 40);
        bench.gic.set_line(0, 40, true);
        bench
            .gic
            .distributor_write(0, GICD_ITARGETSR + 40, 1, 0b1000)
            .unwrap();
        assert_eq!(bench.asserted(), [false, false, false, true]);
    }

    #[test]
    fn private_interrupts_are_banked_per_core() {
        let bench = Bench::enabled();
        bench.set_priority(1, TIMER_PPI, 0x70);
        bench.enable(1, TIMER_PPI);
        assert_eq!(
            bench
                .gic
                .distributor_read(0, GICD_IPRIORITYR + u64::from(TIMER_PPI), 1),
            Some(0)
        );
        assert_eq!(bench.d_read(0, GICD_ISENABLER), 0x0000_ffff);
        assert_eq!(bench.d_read(1, GICD_ISENABLER), 0x4000_ffff);

        // Core 2's line is asserted but its copy is not enabled.
        bench.gic.set_line(1, TIMER_PPI, true);
        bench.gic.set_line(2, TIMER_PPI, true);
        assert_eq!(bench.asserted(), [false, true, false, false]);
        assert_eq!(
            [0, 1, 2].map(|core| bench.d_read(core, GICD_PPISR)),
            [0, 1 << 14, 1 << 14]
        );
        assert_eq!(bench.acknowledge(1), TIMER_PPI);
        assert_eq!(bench.d_read(1, GICD_ISACTIVER), 1 << TIMER_PPI);
        assert_eq!(bench.d_read(2, GICD_ISACTIVER), 0);

        bench.gic.set_line(0, 32 + 37, true);
        assert_eq!(bench.d_read(3, GICD_SPISR + 4), 1 << 5);
    }

    #[test]
    fn irq_raised_reports_every_output_that_rises() {
        let bench = Bench::enabled();
        bench.route_spi(40, 0x80, 0b1010);
        bench.route_spi(41, 0xa0, 0b0010);
        assert!(bench.take_raised().is_empty());
        bench.gic.set_line(0, 40, true);
        assert_eq!(bench.take_raised(), [1, 3]);

        // Completion unblocks a lower-priority interrupt on the same core.
        assert_eq!(bench.acknowledge(1), 40);
        bench.gic.set_line(0, 40, false);
        bench.gic.set_line(0, 41, true);
        assert!(bench.take_raised().is_empty());
        bench.end(1, 40);
        assert_eq!(bench.take_raised(), [1]);
        bench.gic.set_line(0, 41, false);

        // Enabling an interrupt whose line is already asserted.
        bench.gic.set_line(2, TIMER_PPI, true);
        assert!(bench.take_raised().is_empty());
        bench.enable(2, TIMER_PPI);
        assert_eq!(bench.take_raised(), [2]);

        bench.d_write(2, GICD_SGIR, 0b01 << 24 | 1);
        assert_eq!(bench.take_raised(), [0, 1, 3]);

        // Opening the mask, the CPU interface and the distributor.
        bench.c_write(0, GICC_PMR, 0);
        bench.c_write(0, GICC_PMR, 0xff);
        assert_eq!(bench.take_raised(), [0]);
        bench.c_write(3, GICC_CTLR, 0);
        bench.c_write(3, GICC_CTLR, 1);
        assert_eq!(bench.take_raised(), [3]);
        bench.d_write(0, GICD_CTLR, 0);
        assert_eq!(bench.asserted(), [false; 4]);
        bench.d_write(0, GICD_CTLR, 1);
        assert_eq!(bench.take_raised(), [0, 1, 2, 3]);
    }

    #[test]
    fn eoi_mode_separates_priority_drop_from_deactivation() {
        let bench = Bench::enabled();
        bench.c_write(0, GICC_CTLR, 1 | 1 << 9);
        bench.route_spi(40, 0x80, 0b0001);
        bench.route_spi(41, 0x90, 0b0001);
        bench.gic.set_line(0, 40, true);
        assert_eq!(bench.acknowledge(0), 40);

        // The priority drop lets other interrupts through, but the still
        // asserted interrupt stays active and is not signaled again.
        bench.end(0, 40);
        assert_eq!(bench.c_read(0, GICC_RPR), 0xf0);
        assert_eq!(bench.d_read(0, GICD_ISACTIVER + 4), 1 << 8);
        assert!(!bench.gic.irq_asserted(0));
        bench.gic.set_line(0, 41, true);
        assert_eq!(bench.acknowledge(0), 41);
        bench.gic.set_line(0, 41, false);
        bench.end(0, 41);

        bench.c_write(0, GICC_DIR, 40);
        assert_eq!(bench.acknowledge(0), 40);
        // Special IDs are ignored; an interrupt that is not active is not.
        bench.c_write(0, GICC_DIR, SPURIOUS_INTID);
        assert_eq!(bench.gic.cpu_interface_write(0, GICC_DIR, 4, 42), None);
    }

    #[test]
    fn unpredictable_completions_are_reported() {
        let bench = Bench::enabled();
        bench.route_spi(40, 0x80, 0b0001);
        bench.c_write(0, GICC_EOIR, SPURIOUS_INTID);
        // Nothing was acknowledged.
        assert_eq!(bench.gic.cpu_interface_write(0, GICC_EOIR, 4, 40), None);
        bench.gic.set_line(0, 40, true);
        assert_eq!(bench.acknowledge(0), 40);
        // GICC_DIR needs EOImodeNS.
        assert_eq!(bench.gic.cpu_interface_write(0, GICC_DIR, 4, 40), None);
        assert_eq!(bench.gic.cpu_interface_write(0, GICC_EOIR, 4, 500), None);
        bench.end(0, 40);
    }

    #[test]
    fn disabled_distributor_holds_back_new_pending_state() {
        let bench = Bench::enabled();
        bench.d_write(0, GICD_CTLR, 0);
        bench.make_edge_triggered(40);
        for intid in [40, 41, 42] {
            bench.route_spi(intid, 0x80, 0b0001);
        }

        bench.d_write(0, GICD_SGIR, 3 | 0b0010 << 16);
        bench.gic.set_line(0, 40, true);
        bench.gic.set_line(0, 41, true);
        bench.d_write(0, GICD_ISPENDR + 4, 1 << 10);
        assert_eq!(bench.d_read(0, GICD_ISPENDR + 4), 1 << 10);
        assert_eq!(bench.asserted(), [false; 4]);
        assert_eq!(bench.acknowledge(0), SPURIOUS_INTID);

        // The SGI and the edge are lost; the level and the written state are not.
        bench.d_write(0, GICD_CTLR, 1);
        assert_eq!(bench.d_read(1, GICD_SPENDSGIR), 0);
        assert_eq!(bench.d_read(0, GICD_ISPENDR + 4), 1 << 9 | 1 << 10);
        assert_eq!(bench.asserted(), [true, false, false, false]);
    }

    #[test]
    fn only_architected_access_widths_are_modeled() {
        let bench = Bench::reset();
        let gic = &bench.gic;
        assert_eq!(gic.distributor_read(0, GICD_ISENABLER, 1), None);
        assert_eq!(gic.distributor_read(0, GICD_ISENABLER, 2), None);
        assert_eq!(gic.distributor_read(0, GICD_ISENABLER + 2, 4), None);
        assert_eq!(gic.distributor_write(0, GICD_CTLR, 1, 1), None);
        assert_eq!(gic.distributor_read(0, GICD_TYPER, 8), None);
        assert_eq!(gic.cpu_interface_read(0, GICC_IAR, 1), None);
        assert_eq!(gic.cpu_interface_write(0, GICC_PMR, 1, 0xff), None);

        // Byte-accessible registers also take aligned halfwords.
        gic.distributor_write(0, GICD_IPRIORITYR + 40, 2, 0x7f3f)
            .unwrap();
        assert_eq!(
            gic.distributor_read(0, GICD_IPRIORITYR + 40, 4),
            Some(0x7030)
        );
        assert_eq!(gic.distributor_read(0, GICD_IPRIORITYR + 41, 1), Some(0x70));
        assert_eq!(gic.distributor_read(0, GICD_IPRIORITYR + 41, 2), None);

        // Reserved offsets and the write-only register.
        assert_eq!(gic.distributor_read(0, 0x00c, 4), None);
        assert_eq!(gic.distributor_read(0, 0x7fc, 4), None);
        assert_eq!(gic.distributor_write(0, 0xbfc, 4, 0), None);
        assert_eq!(gic.distributor_read(0, GICD_SGIR, 4), None);
        assert_eq!(gic.cpu_interface_read(0, 0x2c, 4), None);
        assert_eq!(gic.cpu_interface_read(0, GICC_EOIR, 4), None);
        assert_eq!(gic.cpu_interface_read(0, GICC_DIR, 4), None);
    }

    #[test]
    fn unimplemented_interrupts_are_raz_wi() {
        let bench = Bench::reset();
        // The kernel's suspend path walks all 1020 IDs (0x800d5210, 0x800d5700).
        for offset in [
            GICD_ISENABLER + 0x1c,
            GICD_ISENABLER + 0x78,
            GICD_ISPENDR + 0x1c,
            GICD_ISACTIVER + 0x7c,
            GICD_IPRIORITYR + 224,
            GICD_IPRIORITYR + 1016,
            GICD_ITARGETSR + 224,
            GICD_ITARGETSR + 1016,
            GICD_ICFGR + 0x38,
            GICD_ICFGR + 0xf8,
            GICD_IGROUPR,
        ] {
            bench.d_write(0, offset, ALL);
            assert_eq!(bench.d_read(0, offset), 0, "offset {offset:#x}");
        }
        assert_eq!(bench.d_read(0, GICD_SPISR + 0x18), 0);

        // The GIC-400 has no PPIs 16-24, and its SGIs cannot be disabled.
        bench.d_write(0, GICD_ISENABLER, ALL);
        assert_eq!(bench.d_read(0, GICD_ISENABLER), 0xfe00_ffff);
        bench.d_write(0, GICD_ICENABLER, ALL);
        assert_eq!(bench.d_read(0, GICD_ICENABLER), 0x0000_ffff);
        bench.d_write(0, GICD_ISACTIVER, ALL);
        assert_eq!(bench.d_read(0, GICD_ICACTIVER), 0xfe00_ffff);

        // The last implemented register words are fully populated.
        bench.d_write(0, GICD_ISENABLER + 0x18, ALL);
        assert_eq!(bench.d_read(0, GICD_ISENABLER + 0x18), ALL);
        bench.d_write(0, GICD_IPRIORITYR + 220, ALL);
        assert_eq!(bench.d_read(0, GICD_IPRIORITYR + 220), 0xf0f0_f0f0);
    }

    #[test]
    fn configuration_register_layout() {
        let bench = Bench::reset();
        for offset in [GICD_ICFGR, GICD_ICFGR + 4, GICD_ICFGR + 8] {
            bench.d_write(0, offset, 0);
        }
        // SGIs read as edge-triggered, the PPIs 25-31 are level-sensitive.
        assert_eq!(bench.d_read(0, GICD_ICFGR), 0xaaaa_aaaa);
        assert_eq!(bench.d_read(0, GICD_ICFGR + 4), 0x5554_0000);
        assert_eq!(bench.d_read(0, GICD_ICFGR + 8), 0x5555_5555);
        for offset in [GICD_ICFGR, GICD_ICFGR + 4, GICD_ICFGR + 8] {
            bench.d_write(0, offset, ALL);
        }
        assert_eq!(bench.d_read(0, GICD_ICFGR), 0xaaaa_aaaa);
        assert_eq!(bench.d_read(0, GICD_ICFGR + 4), 0x5554_0000);
        assert_eq!(bench.d_read(0, GICD_ICFGR + 8), ALL);
    }

    #[test]
    fn kernel_initialization_and_masked_timer_interrupt() {
        let bench = Bench::reset();
        let byte_write = |offset, value| bench.gic.distributor_write(0, offset, 1, value).unwrap();

        // Gic_InitializeCoreInterface on core 0 (0x800d4f44-0x800d50e8).
        bench.d_write(0, GICD_CTLR, 0);
        bench.c_write(0, GICC_PMR, 0);
        bench.c_write(0, GICC_BPR, 7);
        let interrupts = u64::from((bench.d_read(0, GICD_TYPER) & 0x1f) * 32 + 32);
        assert_eq!(interrupts, 224);
        for word in (0..interrupts / 32).map(|index| index * 4) {
            bench.d_write(0, GICD_ICENABLER + word, ALL);
            bench.d_write(0, GICD_ICPENDR + word, ALL);
            bench.d_write(0, GICD_ICACTIVER + word, ALL);
            bench.d_write(0, GICD_IGROUPR + word, 0);
        }
        for intid in 0..interrupts {
            byte_write(GICD_IPRIORITYR + intid, 0xff);
            byte_write(GICD_ITARGETSR + intid, 0);
        }
        for word in (0..interrupts / 16).map(|index| index * 4) {
            bench.d_write(0, GICD_ICFGR + word, 0);
        }
        bench.d_write(0, GICD_CTLR, 1);
        bench.c_write(0, GICC_CTLR, 1);
        bench.c_write(0, GICC_PMR, 0xff);

        // KInterruptManager_BindInterrupt for the timer at priority 1 (0x800ca860-0x800ca870).
        bench.d_write(0, GICD_ICPENDR, 1 << TIMER_PPI);
        byte_write(GICD_IPRIORITYR + u64::from(TIMER_PPI), 0x7f);
        bench.d_write(0, GICD_ISENABLER, 1 << TIMER_PPI);
        assert!(!bench.gic.irq_asserted(0));

        // KInterruptManager_Dispatch (0x800cad5c-0x800cadd0) disables the
        // interrupt for its handler and ends it while the timer still fires.
        bench.gic.set_line(0, TIMER_PPI, true);
        assert_eq!(bench.take_raised(), [0]);
        let acknowledged = bench.acknowledge(0);
        assert_eq!(acknowledged, TIMER_PPI);
        bench.d_write(0, GICD_ICENABLER, 1 << TIMER_PPI);
        bench.end(0, acknowledged);
        assert!(!bench.gic.irq_asserted(0));

        // KInterruptManager_ClearInterrupt (0x800ca688) enables it again.
        bench.d_write(0, GICD_ISENABLER, 1 << TIMER_PPI);
        assert_eq!(bench.take_raised(), [0]);
        assert_eq!(bench.acknowledge(0), TIMER_PPI);
    }

    #[test]
    fn restored_controller_keeps_pending_and_active_interrupts_on_their_cores() {
        let saved = Bench::enabled();
        // SPI 40: level-sensitive, held asserted, routed to core 2.
        saved.route_spi(40, 0x80, 1 << 2);
        saved.gic.set_line(0, 40, true);
        // SPI 41: an edge on core 1, acknowledged and not yet ended.
        saved.make_edge_triggered(41);
        saved.route_spi(41, 0x40, 1 << 1);
        saved.gic.set_line(0, 41, true);
        saved.gic.set_line(0, 41, false);
        assert_eq!(saved.acknowledge(1), 41);
        // SPI 42: an edge latched behind 41 on core 1, at a priority 41's
        // handling does not let through.
        saved.make_edge_triggered(42);
        saved.route_spi(42, 0x60, 1 << 1);
        saved.gic.set_line(0, 42, true);
        saved.gic.set_line(0, 42, false);
        assert_eq!(saved.asserted(), [false, false, true, false]);

        let restored = Bench::reset();
        restored.gic.restore(&saved.gic.save().unwrap()).unwrap();
        assert_eq!(restored.asserted(), [false, false, true, false]);
        assert!(restored.take_raised().is_empty());
        let active = restored.d_read(1, GICD_ISACTIVER + 4);
        assert_eq!(active, 1 << (41 - 32));
        assert!(restored.pending_bit(1, 42));

        // Ending 41 lets the latched 42 through to core 1.
        restored.end(1, 41);
        assert_eq!(restored.take_raised(), [1]);
        assert_eq!(restored.acknowledge(1), 42);
        // 40's line is still held: after it is handled it is pending again.
        assert_eq!(restored.acknowledge(2), 40);
        restored.end(2, 40);
        assert_eq!(restored.take_raised(), [2]);
        restored.gic.set_line(0, 40, false);
        assert_eq!(restored.asserted(), [false, false, false, false]);
    }
}
