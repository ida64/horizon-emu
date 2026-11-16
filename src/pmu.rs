//! The performance monitors of a core, as far as the firmware touches
//! them: registers that hold their configuration, with nothing counting.
//!
//! Hypervisor.framework has no performance monitor for a guest, so every
//! access traps here. The kernel does not use the unit; it saves each
//! core's registers before the console sleeps and writes them back when it
//! wakes (mesosphere `kern_k_sleep_manager.cpp`, `SavedSystemRegisters`,
//! which the kernel's code matches: the first access is the read of
//! `PMCR_EL0` on all four cores). That needs registers that read back
//! what was written, and `PMCR_EL0.N` to say how many event counters
//! there are to save.
//!
//! Counting is not modeled. A write that would start a counter (the
//! enable in `PMCR_EL0` together with any counter's enable bit) is
//! refused, as is the software increment.
//!
//! What the registers hold when the kernel first reads them is an
//! assumption: the architecture leaves most of them unknown out of reset,
//! and no stage of the boot chain is known to use the unit, so they read
//! zero, with `PMCR_EL0` at the Cortex-A57's reset value.

use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::{
    arm64::SysReg,
    snapshot::{Snapshot, SnapshotError, decode, encode},
};

/// `PMCR_EL0` out of reset on a Cortex-A57 (its Technical Reference
/// Manual, `0x41013000`): implemented by Arm, ID code 1, six event
/// counters, everything off.
const PMCR_AT_RESET: u64 = 0x4101_3000;
const PMCR_ENABLE: u64 = 1 << 0;
/// Write-only: zero the event counters.
const PMCR_RESET_EVENT_COUNTERS: u64 = 1 << 1;
/// Write-only: zero the cycle counter.
const PMCR_RESET_CYCLE_COUNTER: u64 = 1 << 2;
/// The bits a write changes: enable, clock divider, export, disable in
/// prohibited regions, long cycle counter.
const PMCR_WRITABLE: u64 = PMCR_ENABLE | 0b1111 << 3;

const EVENT_COUNTERS: usize = 6;
/// The counters as the set and clear registers number them: the event
/// counters from bit 0, the cycle counter in bit 31.
const COUNTERS: u64 = ((1 << EVENT_COUNTERS) - 1) | (1 << 31);

const PMCR_EL0: SysReg = SysReg::new(3, 3, 9, 12, 0);
const PMCNTENSET_EL0: SysReg = SysReg::new(3, 3, 9, 12, 1);
const PMCNTENCLR_EL0: SysReg = SysReg::new(3, 3, 9, 12, 2);
const PMOVSCLR_EL0: SysReg = SysReg::new(3, 3, 9, 12, 3);
const PMSELR_EL0: SysReg = SysReg::new(3, 3, 9, 12, 5);
const PMCCNTR_EL0: SysReg = SysReg::new(3, 3, 9, 13, 0);
const PMUSERENR_EL0: SysReg = SysReg::new(3, 3, 9, 14, 0);
const PMOVSSET_EL0: SysReg = SysReg::new(3, 3, 9, 14, 3);
const PMINTENSET_EL1: SysReg = SysReg::new(3, 0, 9, 14, 1);
const PMINTENCLR_EL1: SysReg = SysReg::new(3, 0, 9, 14, 2);
const PMCCFILTR_EL0: SysReg = SysReg::new(3, 3, 14, 15, 7);

/// Where a register of the unit is kept.
#[derive(Clone, Copy)]
enum Slot {
    Control,
    /// A set of counters with a register that sets bits and one that
    /// clears them; `true` for the one that clears.
    Enabled(bool),
    Interrupts(bool),
    Overflowed(bool),
    Selected,
    UserAccess,
    CycleCounter,
    CycleFilter,
    EventCounter(usize),
    EventType(usize),
}

fn slot(register: SysReg) -> Option<Slot> {
    Some(match register {
        PMCR_EL0 => Slot::Control,
        PMCNTENSET_EL0 => Slot::Enabled(false),
        PMCNTENCLR_EL0 => Slot::Enabled(true),
        PMINTENSET_EL1 => Slot::Interrupts(false),
        PMINTENCLR_EL1 => Slot::Interrupts(true),
        PMOVSSET_EL0 => Slot::Overflowed(false),
        PMOVSCLR_EL0 => Slot::Overflowed(true),
        PMSELR_EL0 => Slot::Selected,
        PMUSERENR_EL0 => Slot::UserAccess,
        PMCCNTR_EL0 => Slot::CycleCounter,
        PMCCFILTR_EL0 => Slot::CycleFilter,
        _ => {
            // PMEVCNTR<n>_EL0 and PMEVTYPER<n>_EL0: S3_3_C14, CRm 8 to 11
            // and 12 to 15, n = CRm<1:0>:op2.
            if (register.op0(), register.op1(), register.crn()) != (3, 3, 14) {
                return None;
            }
            let index = usize::from((register.crm() & 3) << 3 | register.op2());
            match register.crm() >> 2 {
                2 if index < EVENT_COUNTERS => Slot::EventCounter(index),
                3 if index < EVENT_COUNTERS => Slot::EventType(index),
                _ => return None,
            }
        }
    })
}

/// One core's performance monitor registers.
#[derive(Default, Serialize, Deserialize)]
pub struct Pmu {
    control: u64,
    enabled: u64,
    interrupts: u64,
    overflowed: u64,
    selected: u64,
    user_access: u64,
    cycle_counter: u64,
    cycle_filter: u64,
    event_counters: [u64; EVENT_COUNTERS],
    event_types: [u64; EVENT_COUNTERS],
}

impl Pmu {
    /// What a trapped read of `register` gets, or `None` if it is not a
    /// register kept here.
    pub fn read(&self, register: SysReg) -> Option<u64> {
        Some(match slot(register)? {
            Slot::Control => PMCR_AT_RESET | self.control,
            Slot::Enabled(_) => self.enabled,
            Slot::Interrupts(_) => self.interrupts,
            Slot::Overflowed(_) => self.overflowed,
            Slot::Selected => self.selected,
            Slot::UserAccess => self.user_access,
            Slot::CycleCounter => self.cycle_counter,
            Slot::CycleFilter => self.cycle_filter,
            Slot::EventCounter(index) => self.event_counters[index],
            Slot::EventType(index) => self.event_types[index],
        })
    }

    /// Applies a trapped write, or returns `None` if the register is not
    /// kept here or the write would start a counter.
    pub fn write(&mut self, register: SysReg, value: u64) -> Option<()> {
        let change = |set: u64, clears: bool| {
            let bits = value & COUNTERS;
            if clears { set & !bits } else { set | bits }
        };
        let (mut control, mut enabled) = (self.control, self.enabled);
        match slot(register)? {
            Slot::Control => control = value & PMCR_WRITABLE,
            Slot::Enabled(clears) => enabled = change(enabled, clears),
            _ => {}
        }
        if control & PMCR_ENABLE != 0 && enabled != 0 {
            return None;
        }
        (self.control, self.enabled) = (control, enabled);
        match slot(register)? {
            Slot::Control => {
                if value & PMCR_RESET_EVENT_COUNTERS != 0 {
                    self.event_counters = [0; EVENT_COUNTERS];
                }
                if value & PMCR_RESET_CYCLE_COUNTER != 0 {
                    self.cycle_counter = 0;
                }
            }
            Slot::Enabled(_) => {}
            Slot::Interrupts(clears) => self.interrupts = change(self.interrupts, clears),
            Slot::Overflowed(clears) => self.overflowed = change(self.overflowed, clears),
            Slot::Selected => self.selected = value & 0x1F,
            Slot::UserAccess => self.user_access = value & 0xF,
            Slot::CycleCounter => self.cycle_counter = value,
            Slot::CycleFilter => self.cycle_filter = value & 0xFFFF_FFFF,
            Slot::EventCounter(index) => self.event_counters[index] = value & 0xFFFF_FFFF,
            Slot::EventType(index) => self.event_types[index] = value & 0xFFFF_FFFF,
        }
        Some(())
    }
}

/// Saves every register of the core's unit; nothing counts, so the values
/// are all there is.
impl Snapshot for Mutex<Pmu> {
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        encode(&*self.lock().unwrap())
    }

    fn restore(&self, state: &[u8]) -> Result<(), SnapshotError> {
        *self.lock().unwrap() = decode("performance monitors", state)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registers_keep_what_is_written_and_nothing_starts_counting() {
        let mut pmu = Pmu::default();
        assert_eq!(pmu.read(PMCR_EL0), Some(PMCR_AT_RESET));
        let counter = |index| SysReg::new(3, 3, 14, 8, index);
        let kind = |index| SysReg::new(3, 3, 14, 12, index);

        pmu.write(counter(5), 0x1234).unwrap();
        pmu.write(kind(5), 0x11).unwrap();
        pmu.write(PMCCNTR_EL0, 77).unwrap();
        assert_eq!(pmu.read(counter(5)), Some(0x1234));
        assert_eq!(pmu.read(kind(5)), Some(0x11));
        // The core has six event counters.
        assert_eq!(pmu.read(counter(6)), None);
        assert_eq!(pmu.write(kind(6), 0), None);

        // The reset bits act and do not stay.
        pmu.write(PMCR_EL0, PMCR_RESET_EVENT_COUNTERS).unwrap();
        assert_eq!(pmu.read(counter(5)), Some(0));
        assert_eq!(pmu.read(PMCCNTR_EL0), Some(77));
        pmu.write(PMCR_EL0, PMCR_RESET_CYCLE_COUNTER).unwrap();
        assert_eq!(pmu.read(PMCCNTR_EL0), Some(0));
        assert_eq!(pmu.read(PMCR_EL0), Some(PMCR_AT_RESET));

        // Set and clear registers show one set of bits.
        pmu.write(PMINTENSET_EL1, u64::MAX).unwrap();
        assert_eq!(pmu.read(PMINTENCLR_EL1), Some(COUNTERS));
        pmu.write(PMINTENCLR_EL1, 1 << 31).unwrap();
        assert_eq!(pmu.read(PMINTENSET_EL1), Some(COUNTERS & !(1 << 31)));

        // Enabling a counter is fine until the unit is switched on, in
        // either order.
        pmu.write(PMCNTENSET_EL0, 1).unwrap();
        assert_eq!(pmu.write(PMCR_EL0, PMCR_ENABLE), None);
        assert_eq!(pmu.read(PMCR_EL0), Some(PMCR_AT_RESET));
        pmu.write(PMCNTENCLR_EL0, 1).unwrap();
        pmu.write(PMCR_EL0, PMCR_ENABLE).unwrap();
        assert_eq!(pmu.write(PMCNTENSET_EL0, 1 << 31), None);
        assert_eq!(pmu.read(PMCNTENSET_EL0), Some(0));
    }
}
