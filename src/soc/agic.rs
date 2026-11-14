//! The audio processing engine's interrupt controller (AGIC): an ARM
//! GICv2-style distributor and CPU interface that gather the engine's
//! interrupts (its hardware mailboxes, its activity monitor, its DMA) and
//! hand them to two processors: interface 0 is the CPU complex, reached
//! through one line of the main interrupt controller, interface 1 is the
//! audio DSP.
//!
//! # What `audio` does with it
//!
//! `audio` maps the distributor at `0x702F9000` and the CPU interface at
//! `0x702FA000` (text `0x17610`) and then (text `0x17460`):
//!
//! * disables the distributor, reads `GICD_TYPER` and, for every bank of 32
//!   lines it reports, puts all lines in group 1 (`GICD_IGROUPR`) and
//!   disables them (`GICD_ICENABLER`); it routes every shared line to
//!   interface 0 (`GICD_ITARGETSR` = `0x01010101` from line 32 on) and
//!   enables the distributor (`GICD_CTLR` = 3);
//! * programs its interface: `GICC_PMR` = `0xFF`, `GICC_BPR` = 7,
//!   `GICC_CTLR` = `0x1F`;
//! * waits on interrupt 134 of the main controller (shared line 102), and on
//!   each wake-up reads `GICC_IAR`, runs the handler registered for the line
//!   unless it read the spurious ID 1023, and writes the ID back to
//!   `GICC_EOIR` (text `0x17130`).
//!
//! Lines it handles itself are configured edge-triggered (`GICD_ICFGR`) and
//! enabled one by one (`GICD_ISENABLER`, text `0x171f0`); two lines, the DSP's
//! own mailbox interrupts, it routes to interface 1 instead (text `0x167d0`
//! through `0x17050`). It accepts lines 32 to 87 only.
//!
//! Register layout and behaviour follow the public ARM GICv2 architecture
//! specification. Only interface 0 is modeled: interrupts routed to the DSP
//! are the business of the DSP's model ([`super::adsp`]), which reacts to
//! what raises them directly. Security extensions, software-generated and
//! private interrupts are not modeled; their registers refuse, except the
//! group and enable bits of bank 0, which the start-up loop above writes.
use std::sync::{Arc, Mutex};

use crate::bus::MmioDevice;

pub const DISTRIBUTOR_BASE: u64 = 0x702F_9000;
pub const DISTRIBUTOR_SIZE: u64 = 0x1000;
pub const CPU_INTERFACE_BASE: u64 = 0x702F_A000;
pub const CPU_INTERFACE_SIZE: u64 = 0x2000;
/// Interface 0's line into the main interrupt controller: `audio` creates
/// its interrupt event for interrupt 134 (text `0x17460`).
pub const INTID: u32 = 134;

/// Interrupt lines: `audio` accepts 32 to 87 (text `0x16f30`), so the
/// controller has at least 88; `GICD_TYPER` reports lines in banks of 32,
/// which makes 96 the smallest count it can report. That the engine has no
/// more is an assumption; `audio` never looks past 87.
const LINES: usize = 96;
const BANKS: usize = LINES / 32;
/// `GICD_TYPER`: `ITLinesNumber` (banks minus one) and `CPUNumber` (two
/// interfaces, minus one, in bits 7:5).
const TYPER: u32 = (BANKS as u32 - 1) | 1 << 5;
const SPURIOUS: u32 = 1023;
/// Interface 0's bit in a target byte.
const INTERFACE_0: u8 = 1;

struct State {
    distributor_control: u32,
    group: [u32; BANKS],
    enabled: [u32; BANKS],
    pending: [u32; BANKS],
    active: [u32; BANKS],
    priority: [u8; LINES],
    targets: [u8; LINES],
    config: [u32; BANKS * 2],
    cpu_control: u32,
    priority_mask: u32,
    binary_point: u32,
}

impl State {
    fn bit(bits: &[u32; BANKS], line: usize) -> bool {
        bits[line / 32] & 1 << (line % 32) != 0
    }

    /// Whether group `group` is enabled at both the distributor and
    /// interface 0.
    fn group_enabled(&self, group: u32) -> bool {
        self.distributor_control & 1 << group != 0 && self.cpu_control & 1 << group != 0
    }

    /// The priority interface 0 runs at: that of its highest-priority active
    /// interrupt, or idle (above every priority) when none is active.
    fn running_priority(&self) -> u32 {
        (32..LINES)
            .filter(|&line| Self::bit(&self.active, line))
            .map(|line| u32::from(self.priority[line]))
            .min()
            .unwrap_or(0x100)
    }

    /// The interrupt interface 0 would acknowledge now, if any: the
    /// highest-priority (then lowest-numbered) pending, enabled shared
    /// interrupt routed to it whose priority beats both the mask and the
    /// running priority.
    fn deliverable(&self) -> Option<usize> {
        let ceiling = self.running_priority().min(self.priority_mask);
        (32..LINES)
            .filter(|&line| {
                Self::bit(&self.pending, line)
                    && Self::bit(&self.enabled, line)
                    && self.targets[line] & INTERFACE_0 != 0
                    && self.group_enabled(u32::from(Self::bit(&self.group, line)))
                    && u32::from(self.priority[line]) < ceiling
            })
            .min_by_key(|&line| self.priority[line])
    }
}

pub struct Agic {
    state: Mutex<State>,
    /// Interface 0's interrupt line into the main controller.
    output: Box<dyn Fn(bool) + Send + Sync>,
}

impl Agic {
    /// At reset the distributor and interface are disabled, every line is
    /// in group 0, disabled, idle, at priority 0 and routed nowhere, as the
    /// GICv2 specification gives the reset values.
    pub fn new(output: Box<dyn Fn(bool) + Send + Sync>) -> Self {
        Self {
            state: Mutex::new(State {
                distributor_control: 0,
                group: [0; BANKS],
                enabled: [0; BANKS],
                pending: [0; BANKS],
                active: [0; BANKS],
                priority: [0; LINES],
                targets: [0; LINES],
                config: [0; BANKS * 2],
                cpu_control: 0,
                priority_mask: 0,
                binary_point: 0,
            }),
            output,
        }
    }

    /// An edge on shared line `line`: the interrupt becomes pending.
    pub fn pulse(&self, line: u32) {
        let mut state = self.state.lock().unwrap();
        let line = line as usize;
        if (32..LINES).contains(&line) {
            state.pending[line / 32] |= 1 << (line % 32);
        }
        self.update(&state);
    }

    pub fn distributor(self: &Arc<Self>) -> Distributor {
        Distributor(Arc::clone(self))
    }

    pub fn cpu_interface(self: &Arc<Self>) -> CpuInterface {
        CpuInterface(Arc::clone(self))
    }

    fn update(&self, state: &State) {
        (self.output)(state.deliverable().is_some());
    }
}

/// The distributor's registers.
pub struct Distributor(Arc<Agic>);

/// Bank index of a bit-per-line register at `offset` from its `base`, for
/// the banks that exist.
fn bank(offset: u64, base: u64) -> Option<usize> {
    let bank = ((offset - base) / 4) as usize;
    (bank < BANKS).then_some(bank)
}

impl MmioDevice for Distributor {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        let state = self.0.state.lock().unwrap();
        if let 0x400..0x460 | 0x800..0x860 = offset {
            let bytes = if offset < 0x800 {
                &state.priority
            } else {
                &state.targets
            };
            let first = (offset & 0x3FF) as usize;
            return match size {
                1 => Some(u64::from(bytes[first])),
                4 if first.is_multiple_of(4) => Some(u64::from(u32::from_le_bytes(
                    bytes[first..first + 4].try_into().unwrap(),
                ))),
                _ => None,
            };
        }
        if size != 4 || !offset.is_multiple_of(4) {
            return None;
        }
        let value = match offset {
            0x000 => state.distributor_control,
            0x004 => TYPER,
            0x080..0x08C => state.group[bank(offset, 0x080)?],
            0x100..0x10C => state.enabled[bank(offset, 0x100)?],
            0x180..0x18C => state.enabled[bank(offset, 0x180)?],
            0x200..0x20C => state.pending[bank(offset, 0x200)?],
            0x280..0x28C => state.pending[bank(offset, 0x280)?],
            0x300..0x30C => state.active[bank(offset, 0x300)?],
            0x380..0x38C => state.active[bank(offset, 0x380)?],
            0xC00..0xC18 => state.config[((offset - 0xC00) / 4) as usize],
            _ => return None,
        };
        Some(u64::from(value))
    }

    fn write(&self, _core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        let mut state = self.0.state.lock().unwrap();
        let shared = |line: usize| line >= 32;
        match (offset, size) {
            (0x400..0x460 | 0x800..0x860, 1 | 4) if offset.is_multiple_of(u64::from(size)) => {
                let first = (offset & 0x3FF) as usize;
                for (index, byte) in value.to_le_bytes()[..usize::from(size)].iter().enumerate() {
                    let line = first + index;
                    // Software-generated and private lines are routed by
                    // the hardware and their bytes are read-only.
                    if offset < 0x800 {
                        state.priority[line] = *byte;
                    } else if shared(line) {
                        state.targets[line] = *byte;
                    }
                }
            }
            (_, 4) if offset.is_multiple_of(4) => {
                let value = value as u32;
                // Bank 0 holds the software-generated and private lines,
                // which are not modeled: of its bits only those of the
                // group and enable registers are taken, as storage.
                match offset {
                    0x000 => state.distributor_control = value & 3,
                    0x080..0x08C => state.group[bank(offset, 0x080)?] = value,
                    0x100..0x10C => state.enabled[bank(offset, 0x100)?] |= value,
                    0x180..0x18C => state.enabled[bank(offset, 0x180)?] &= !value,
                    0x200..0x20C if bank(offset, 0x200)? > 0 => {
                        state.pending[bank(offset, 0x200)?] |= value
                    }
                    0x280..0x28C if bank(offset, 0x280)? > 0 => {
                        state.pending[bank(offset, 0x280)?] &= !value
                    }
                    0x300..0x30C if bank(offset, 0x300)? > 0 => {
                        state.active[bank(offset, 0x300)?] |= value
                    }
                    0x380..0x38C if bank(offset, 0x380)? > 0 => {
                        state.active[bank(offset, 0x380)?] &= !value
                    }
                    // The software-generated and private lines' half of
                    // the configuration is read-only.
                    0xC08..0xC18 => state.config[((offset - 0xC00) / 4) as usize] = value,
                    _ => return None,
                }
            }
            _ => return None,
        }
        self.0.update(&state);
        Some(())
    }
}

/// CPU interface 0's registers.
pub struct CpuInterface(Arc<Agic>);

impl MmioDevice for CpuInterface {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        if size != 4 {
            return None;
        }
        let mut state = self.0.state.lock().unwrap();
        let value = match offset {
            0x00 => state.cpu_control,
            0x04 => state.priority_mask,
            0x08 => state.binary_point,
            // GICC_EOIR is write-only in the specification, which leaves
            // what a read returns open. `audio`'s register writer reads
            // every register back after writing it and drops the value
            // (after the write at text `0x17130`): it reads as zero here.
            0x10 => 0,
            // GICC_IAR: acknowledging makes the interrupt active and no
            // longer pending (edge-triggered lines are all there are).
            0x0C => match state.deliverable() {
                Some(line) => {
                    state.pending[line / 32] &= !(1 << (line % 32));
                    state.active[line / 32] |= 1 << (line % 32);
                    self.0.update(&state);
                    line as u32
                }
                None => SPURIOUS,
            },
            0x14 => state.running_priority().min(0xFF),
            0x18 => state.deliverable().map_or(SPURIOUS, |line| line as u32),
            _ => return None,
        };
        Some(u64::from(value))
    }

    fn write(&self, _core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        if size != 4 {
            return None;
        }
        let mut state = self.0.state.lock().unwrap();
        let value = value as u32;
        match offset {
            0x00 => state.cpu_control = value & 0x1F,
            0x04 => state.priority_mask = value & 0xFF,
            0x08 => state.binary_point = value & 7,
            // GICC_EOIR: the interrupt is done.
            0x10 => {
                let line = (value & 0x3FF) as usize;
                if (32..LINES).contains(&line) {
                    state.active[line / 32] &= !(1 << (line % 32));
                }
            }
            _ => return None,
        }
        self.0.update(&state);
        Some(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn audio_start_up_then_one_mailbox_interrupt() {
        let level = Arc::new(AtomicBool::new(false));
        let seen = Arc::clone(&level);
        let agic = Arc::new(Agic::new(Box::new(move |high| {
            seen.store(high, Ordering::Relaxed)
        })));
        let (gicd, gicc) = (agic.distributor(), agic.cpu_interface());
        gicd.write(0, 0, 4, 0).unwrap();
        let banks = (gicd.read(0, 4, 4).unwrap() & 0x1F) + 1;
        for bank in 0..banks {
            gicd.write(0, 0x80 + bank * 4, 4, 0xFFFF_FFFF).unwrap();
            gicd.write(0, 0x180 + bank * 4, 4, 0xFFFF_FFFF).unwrap();
        }
        for word in (0x820..0x800 + 32 * banks).step_by(4) {
            gicd.write(0, word, 4, 0x0101_0101).unwrap();
        }
        gicd.write(0, 0, 4, 3).unwrap();
        gicc.write(0, 4, 4, 0xFF).unwrap();
        gicc.write(0, 8, 4, 7).unwrap();
        gicc.write(0, 0, 4, 0x1F).unwrap();
        // Line 32, edge-triggered, then enabled.
        gicd.write(0, 0xC08, 4, 2).unwrap();
        agic.pulse(32);
        assert!(!level.load(Ordering::Relaxed), "disabled lines stay quiet");
        gicd.write(0, 0x104, 4, 1).unwrap();
        assert!(level.load(Ordering::Relaxed));
        assert_eq!(gicc.read(0, 0x0C, 4), Some(32));
        assert!(!level.load(Ordering::Relaxed));
        assert_eq!(gicc.read(0, 0x0C, 4), Some(u64::from(SPURIOUS)));
        // A second edge while the first is being handled waits for EOI.
        agic.pulse(32);
        assert!(!level.load(Ordering::Relaxed));
        gicc.write(0, 0x10, 4, 32).unwrap();
        assert!(level.load(Ordering::Relaxed));
        // Lines routed to the DSP's interface never reach this one.
        gicd.write(0, 0x821, 1, 2).unwrap();
        gicd.write(0, 0x104, 4, 2).unwrap();
        agic.pulse(33);
        assert_eq!(gicc.read(0, 0x0C, 4), Some(32));
        gicc.write(0, 0x10, 4, 32).unwrap();
        assert!(!level.load(Ordering::Relaxed));
    }
}
