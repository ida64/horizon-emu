//! Host-controlled breakpoints and single-step for one guest vCPU.
//!
//! # Mechanism
//!
//! Everything is per-vCPU hardware state; guest memory is never patched, so
//! cores executing the same code do not see each other's breakpoints.
//!
//! * A breakpoint is one of the CPU's `DBGBVRn_EL1`/`DBGBCRn_EL1` pairs,
//!   programmed for an unlinked address match at EL1 and EL0. It compares the
//!   virtual address of the instruction about to execute, so it works with the
//!   guest MMU off (virtual = physical) and on.
//! * A single-step is the architectural software step: `MDSCR_EL1.SS` with
//!   `PSTATE.SS` set for one guest entry.
//! * `hv_vcpu_set_trap_debug_exceptions` routes the resulting debug exceptions
//!   to the host instead of the guest's EL1 vectors. Routed this way they are
//!   taken from a lower exception level, so neither `PSTATE.D` nor
//!   `MDSCR_EL1.KDE` nor the guest's OS Lock can mask them. It is enabled only
//!   while a breakpoint is armed or a step is in flight.
//! * `hv_vcpu_set_trap_debug_reg_accesses` is always on: the guest's accesses
//!   to the debug registers exit to the host and are served from a shadow by
//!   [`GuestDebug::read_register`] and [`GuestDebug::write_register`], so the
//!   guest reads back what it wrote and cannot see or disturb the host's use
//!   of the real registers.
//!
//! # What the guest can and cannot observe
//!
//! * The guest's own hardware breakpoints, watchpoints and software step do
//!   not function: its `DBGBVR`/`DBGBCR`/`DBGWVR`/`DBGWCR` values and the
//!   `SS`, `KDE` and `MDE` bits of its `MDSCR_EL1` live only in the shadow, so
//!   they never raise an exception. `MDSCR_EL1.TDCC` does reach the hardware,
//!   which keeps EL0 accesses to the debug communication channel trapping to
//!   the guest's EL1 as the guest asked.
//! * A guest `BRK` executes natively while nothing is armed. While debug
//!   exceptions are routed to the host it exits instead ([`DebugExit::Serviced`]),
//!   and on the next entry the CPU itself is made to take an exception to the
//!   guest's EL1 vector at that instruction; the guest observes the same
//!   `ESR_EL1`, `ELR_EL1`, `SPSR_EL1`, `FAR_EL1` and `PSTATE` as natively.
//! * `PSTATE.SS` is never left visible. When a stepped instruction enters a
//!   guest exception handler instead of completing, the CPU saves the host's
//!   step bit into `SPSR_EL1`; it is cleared again before the handler runs.
//!
//! # Steps and guest exceptions
//!
//! A step executes exactly one instruction, or takes exactly one exception:
//! if the stepped instruction is an `SVC`, faults, or is preempted by an
//! interrupt, the step ends on the first instruction of the guest's handler
//! ([`DebugExit::Step`] with `exception_entry`), with `ELR_EL1` telling
//! whether the instruction ran. An instruction that exits to the host (`SMC`,
//! `HVC`, a trapped register or device access) is reported as that exit
//! first; the step completes on the following [`GuestDebug::run`] once the
//! host has emulated it, without entering the guest.
//!
//! # Limitations
//!
//! * A breakpoint is reported once per arrival at its address. If the
//!   instruction under it is preempted by an interrupt, or faults and is
//!   retried, the guest arrives again when its handler returns and the
//!   breakpoint is reported again. [`GuestDebug::wants_interrupts_deferred`]
//!   lets the caller rule out the interrupt case.
//! * If the guest returns from an exception with `SPSR_EL1.SS` set while the
//!   host is stepping that `ERET`, the step also covers the instruction
//!   returned to.

use super::{Exit, HvError, Vcpu, check, sys};
use crate::arm64::{SysReg, exception_class};

/// Illegal Execution state, in `ESR_EL1`.
const EC_ILLEGAL_STATE: u32 = 0x0E;
/// Breakpoint taken from a lower exception level (the guest, seen from the host).
const EC_BREAKPOINT_LOWER: u32 = 0x30;
/// Software Step taken from a lower exception level.
const EC_SOFTWARE_STEP_LOWER: u32 = 0x32;
/// `BRK` executed in AArch64 state.
const EC_BRK64: u32 = 0x3C;
/// Software Step syndrome: an instruction was stepped. Clear when the step
/// exception follows an exception entry instead.
const STEP_ISV: u64 = 1 << 24;

const PSTATE_IL: u64 = 1 << 20;
const PSTATE_SS: u64 = 1 << 21;

const MDSCR_SS: u64 = 1 << 0;
const MDSCR_TDCC: u64 = 1 << 12;
const MDSCR_MDE: u64 = 1 << 15;
/// `SS`, `TDCC`, `KDE`, `HDE` and `MDE`: writable whatever the OS Lock state.
const MDSCR_CONTROL: u64 = 0xF001;
/// `ERR`, `SC2`, `TDA`, `INTdis`, `TXU`, `RXO`, `TXfull` and `RXfull`: state
/// the architecture makes writable only while the OS Lock is set, so that an
/// OS can save and restore it; read-only otherwise.
const MDSCR_SAVE_RESTORE: u64 = 0x6CE8_0040;

/// `OSLSR_EL1.OSLM`: the OS Lock is implemented and controlled by `OSLAR_EL1`.
const OSLSR_OSLM: u64 = 1 << 3;
const OSLSR_OSLK: u64 = 1 << 1;

/// `DBGBCRn_EL1` for a host breakpoint: enabled, EL1 and EL0 (`PMC` = 0b11),
/// all four instruction bytes (`BAS` = 0b1111), unlinked address match.
const BREAKPOINT_CONTROL: u64 = 0x1E7;

const MDSCR_EL1: SysReg = SysReg::new(2, 0, 0, 2, 2);
const OSLSR_EL1: SysReg = SysReg::new(2, 0, 1, 1, 4);

/// `DBGBVRn`, `DBGBCRn`, `DBGWVRn` and `DBGWCRn_EL1` are `S2_0_C0_Cn_{4..=7}`.
const BREAKPOINT_VALUE: usize = 0;
const BREAKPOINT_CONTROL_KIND: usize = 1;
const WATCHPOINT_CONTROL_KIND: usize = 3;

const fn debug_register(kind: usize, index: usize) -> SysReg {
    SysReg::new(2, 0, 0, index as u16, 4 + kind as u16)
}

/// The architecture allows up to 16 breakpoints and 16 watchpoints.
const MAX_REGISTERS: usize = 16;

/// Why [`GuestDebug::run`] returned.
#[derive(Debug, Clone, Copy)]
pub enum DebugExit {
    /// The guest is about to execute the instruction at an armed breakpoint.
    /// `pc` is that address; nothing of the instruction has executed and all
    /// guest state is as the guest left it.
    Breakpoint { pc: u64 },
    /// A step requested with [`GuestDebug::request_step`] finished; `pc` is
    /// the next instruction. With `exception_entry` the stepped instruction
    /// entered a guest exception handler (it was an `SVC`, it faulted, or an
    /// interrupt preempted it) and `pc` is the handler's first instruction.
    Step { pc: u64, exception_entry: bool },
    /// This module handled the exit itself (it stepped the guest off a
    /// breakpoint, or handed a guest `BRK` to the guest). Nothing to service:
    /// call [`GuestDebug::run`] again.
    Serviced,
    /// Not a debug event; service it as if it came from [`Vcpu::run`].
    Guest(Exit),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakpointError {
    /// A64 instructions are word-aligned; the address can never match.
    Unaligned,
    /// Every breakpoint register of the CPU is in use.
    NoFreeRegister,
}

impl std::fmt::Display for BreakpointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unaligned => write!(f, "breakpoint address is not a multiple of 4"),
            Self::NoFreeRegister => write!(f, "all hardware breakpoints are in use"),
        }
    }
}

impl std::error::Error for BreakpointError {}

/// A single-step guest entry that exited to the host for an unrelated reason
/// before its step exception.
#[derive(Debug, Clone, Copy)]
struct InterruptedStep {
    pc: u64,
    /// The CPU had already cleared `PSTATE.SS`: the instruction completed
    /// (`HVC` exits this way) and only the step exception was outstanding.
    retired: bool,
}

/// A guest `BRK` that exited to the host and still has to reach the guest.
///
/// The exception entry is not emulated, because what the CPU does to
/// `PSTATE` on entry depends on features the guest's ID registers do not
/// reveal. Instead the instruction is re-entered with `PSTATE.IL` and a
/// single-step forced: the CPU takes an Illegal Execution state exception to
/// the guest's EL1, which has the same vector, return address and `PSTATE`
/// handling and leaves `FAR_EL1` alone, and the step returns control on the
/// handler's first instruction, where `ESR_EL1` is replaced with `syndrome`
/// and the forced bits are removed from `SPSR_EL1`.
#[derive(Debug, Clone, Copy)]
struct PendingBrk {
    pc: u64,
    /// `ESR_EL2` of the exit, which is also the `ESR_EL1` the guest expects.
    syndrome: u64,
}

/// What the next guest entry has to do, given the PC it resumes at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Plan {
    /// The instruction of an interrupted step has been executed by the CPU
    /// or emulated by the host; the step is over without entering the guest.
    Completed,
    Enter {
        /// Bit n: breakpoint n is on the resume PC and was already reported.
        suppressed: u16,
        /// `PSTATE` bits to force for this entry; nonzero means single-step.
        forced: u64,
    },
}

/// What the guest last wrote to the registers it no longer owns: the part
/// of a core's state that [`GuestDebug`] holds, which a snapshot saves.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GuestRegisters {
    mdscr: u64,
    /// `OSLSR_EL1.OSLK`.
    os_lock: bool,
    /// Indexed by register kind (`op2 - 4`), then by `n`.
    debug: [[u64; MAX_REGISTERS]; 4],
}

/// What is currently programmed into the vCPU, to skip redundant updates.
#[derive(Debug, Clone, Copy)]
struct Hardware {
    /// Bit n: `DBGBCRn_EL1` holds [`BREAKPOINT_CONTROL`] rather than zero.
    enabled: u16,
    mdscr: u64,
    trap_exceptions: bool,
}

/// Debug state of one vCPU: the host's breakpoints and step requests, and
/// the guest's view of the debug registers.
///
/// Create it right after the [`Vcpu`], enter the guest only through
/// [`GuestDebug::run`], and route trapped accesses to debug registers to
/// [`GuestDebug::read_register`] and [`GuestDebug::write_register`].
#[derive(Debug)]
pub struct GuestDebug {
    guest: GuestRegisters,
    breakpoint_count: usize,
    watchpoint_count: usize,
    /// Host breakpoint address held by each `DBGBVRn_EL1`.
    breakpoints: [Option<u64>; MAX_REGISTERS],
    /// Bit n: `breakpoints[n]` changed since `DBGBVRn_EL1` was written.
    unprogrammed: u16,
    hardware: Hardware,
    step: bool,
    /// PC of the last reported stop while the guest has not moved on;
    /// breakpoints there do not fire again for the first instruction.
    stopped_at: Option<u64>,
    interrupted: Option<InterruptedStep>,
    brk: Option<PendingBrk>,
}

impl GuestDebug {
    /// Takes the debug registers of `vcpu` away from the guest. Their
    /// current values become what the guest reads back; the guest's OS Lock
    /// starts out set, its architectural state after a cold reset.
    pub fn attach(vcpu: &Vcpu) -> Result<Self, HvError> {
        let features = host_debug_features()?;
        let mut debug = Self::new(
            (features >> 12 & 0xF) as usize + 1,
            (features >> 20 & 0xF) as usize + 1,
        );
        debug.guest.mdscr = vcpu.sys_reg(MDSCR_EL1)?;
        for kind in 0..4 {
            for index in 0..debug.register_count(kind) {
                let register = debug_register(kind, index);
                debug.guest.debug[kind][index] = vcpu.sys_reg(register)?;
                if kind == BREAKPOINT_CONTROL_KIND || kind == WATCHPOINT_CONTROL_KIND {
                    vcpu.set_sys_reg(register, 0)?;
                }
            }
        }
        debug.hardware.mdscr = debug.guest.mdscr & MDSCR_TDCC;
        vcpu.set_sys_reg(MDSCR_EL1, debug.hardware.mdscr)?;
        set_trap_debug_exceptions(vcpu, false)?;
        // SAFETY: the vCPU is live and owned by this thread.
        check("hv_vcpu_set_trap_debug_reg_accesses", unsafe {
            sys::hv_vcpu_set_trap_debug_reg_accesses(vcpu.id, true)
        })?;
        Ok(debug)
    }

    fn new(breakpoint_count: usize, watchpoint_count: usize) -> Self {
        Self {
            guest: GuestRegisters {
                mdscr: 0,
                os_lock: true,
                debug: [[0; MAX_REGISTERS]; 4],
            },
            breakpoint_count,
            watchpoint_count,
            breakpoints: [None; MAX_REGISTERS],
            unprogrammed: 0,
            hardware: Hardware {
                enabled: 0,
                mdscr: 0,
                trap_exceptions: false,
            },
            step: false,
            stopped_at: None,
            interrupted: None,
            brk: None,
        }
    }

    /// Number of registers of `kind` the CPU has.
    fn register_count(&self, kind: usize) -> usize {
        if kind <= BREAKPOINT_CONTROL_KIND {
            self.breakpoint_count
        } else {
            self.watchpoint_count
        }
    }

    /// How many breakpoints can be armed at once on this CPU.
    pub fn breakpoint_capacity(&self) -> usize {
        self.breakpoint_count
    }

    /// Addresses of the armed breakpoints.
    pub fn breakpoints(&self) -> impl Iterator<Item = u64> + '_ {
        self.breakpoints.iter().flatten().copied()
    }

    /// Arms a breakpoint on the guest virtual address `address`, for EL1 and
    /// EL0. It takes effect at the next [`GuestDebug::run`]. Arming an
    /// address twice is the same as arming it once.
    pub fn add_breakpoint(&mut self, address: u64) -> Result<(), BreakpointError> {
        if !address.is_multiple_of(4) {
            return Err(BreakpointError::Unaligned);
        }
        let slots = &mut self.breakpoints[..self.breakpoint_count];
        if slots.contains(&Some(address)) {
            return Ok(());
        }
        let index = slots
            .iter()
            .position(Option::is_none)
            .ok_or(BreakpointError::NoFreeRegister)?;
        slots[index] = Some(address);
        self.unprogrammed |= 1 << index;
        Ok(())
    }

    /// Disarms the breakpoint on `address`; `false` if there was none.
    pub fn remove_breakpoint(&mut self, address: u64) -> bool {
        match self
            .breakpoints
            .iter_mut()
            .find(|slot| **slot == Some(address))
        {
            Some(slot) => {
                *slot = None;
                true
            }
            None => false,
        }
    }

    /// Makes [`GuestDebug::run`] execute a single instruction and report
    /// [`DebugExit::Step`]. The request stays in force across other exits
    /// (including a [`DebugExit::Breakpoint`] on the instruction to step)
    /// until the step is reported or cancelled.
    pub fn request_step(&mut self) {
        self.step = true;
    }

    /// Withdraws a step request that has not been reported yet.
    pub fn cancel_step(&mut self) {
        self.step = false;
    }

    /// Whether the next [`GuestDebug::run`] silently steps the guest off a
    /// breakpoint it already reported. An interrupt taken on that entry
    /// preempts the instruction, and the breakpoint is reported a second time
    /// when the handler returns to it; a caller that wants one report per
    /// executed instruction holds its pending interrupts back for that run.
    pub fn wants_interrupts_deferred(&self, vcpu: &Vcpu) -> Result<bool, HvError> {
        if self.step || self.stopped_at.is_none() {
            return Ok(false);
        }
        Ok(matches!(
            self.plan(vcpu.pc()?),
            Plan::Enter { suppressed, .. } if suppressed != 0
        ))
    }

    /// Bit n: breakpoint n is armed on `address`.
    fn breakpoints_at(&self, address: u64) -> u16 {
        self.breakpoints
            .iter()
            .enumerate()
            .filter(|(_, slot)| **slot == Some(address))
            .fold(0, |mask, (index, _)| mask | 1 << index)
    }

    fn plan(&self, pc: u64) -> Plan {
        if self
            .interrupted
            .is_some_and(|step| step.retired || step.pc != pc)
        {
            return Plan::Completed;
        }
        let suppressed = if self.stopped_at == Some(pc) {
            self.breakpoints_at(pc)
        } else {
            0
        };
        let forced = if self.brk.is_some_and(|brk| brk.pc == pc) {
            PSTATE_SS | PSTATE_IL
        } else if self.step || suppressed != 0 {
            PSTATE_SS
        } else {
            0
        };
        Plan::Enter { suppressed, forced }
    }

    /// Ends the step in flight with the guest at `pc`.
    fn step_completed(&mut self, pc: u64, exception_entry: bool) -> DebugExit {
        self.stopped_at = None;
        if !self.step {
            return DebugExit::Serviced;
        }
        self.step = false;
        self.stopped_at = Some(pc);
        DebugExit::Step {
            pc,
            exception_entry,
        }
    }

    /// Enters the guest once, like [`Vcpu::run`], with the breakpoints armed
    /// and a requested step applied.
    ///
    /// Resuming from a reported [`DebugExit::Breakpoint`] or
    /// [`DebugExit::Step`] never reports a breakpoint on the instruction the
    /// guest stopped at: that instruction is stepped with its breakpoint off
    /// (reported as [`DebugExit::Serviced`] unless a step was requested), and
    /// the breakpoint is live again from the next instruction on, including
    /// for a branch back to the same address.
    pub fn run(&mut self, vcpu: &Vcpu) -> Result<DebugExit, HvError> {
        // With nothing carried over from earlier exits the PC does not matter.
        let idle = !self.step
            && self.stopped_at.is_none()
            && self.interrupted.is_none()
            && self.brk.is_none();
        let (suppressed, forced) = if idle {
            (0, 0)
        } else {
            let pc = vcpu.pc()?;
            let plan = self.plan(pc);
            self.interrupted = None;
            match plan {
                Plan::Completed => return Ok(self.step_completed(pc, false)),
                Plan::Enter { suppressed, forced } => (suppressed, forced),
            }
        };
        if forced & PSTATE_IL == 0 {
            // The guest is no longer on the BRK; it traps again if re-executed.
            self.brk = None;
        }
        if forced == 0 {
            self.stopped_at = None;
        }

        self.program(vcpu, suppressed, forced != 0)?;
        if forced != 0 {
            vcpu.set_pstate(vcpu.pstate()? | forced)?;
        }
        let exit = vcpu.run()?;
        let mut retired = false;
        if forced != 0 {
            // Exits other than the step exception can leave the forced bits
            // in PSTATE; they must not outlive the entry.
            let pstate = vcpu.pstate()?;
            retired = pstate & PSTATE_SS == 0;
            if pstate & forced != 0 {
                vcpu.set_pstate(pstate & !forced)?;
            }
        }

        let debug_class = match exit {
            Exit::Exception { syndrome, .. } => Some((exception_class(syndrome), syndrome)),
            _ => None,
        };
        match debug_class {
            Some((EC_SOFTWARE_STEP_LOWER, syndrome)) => {
                let pc = vcpu.pc()?;
                let exception_entry = syndrome & STEP_ISV == 0;
                if exception_entry {
                    self.finish_exception_entry(vcpu, pc, forced)?;
                }
                self.brk = None;
                Ok(self.step_completed(pc, exception_entry))
            }
            Some((EC_BREAKPOINT_LOWER, _)) => {
                let pc = vcpu.pc()?;
                self.stopped_at = Some(pc);
                Ok(DebugExit::Breakpoint { pc })
            }
            Some((EC_BRK64, syndrome)) => {
                self.brk = Some(PendingBrk {
                    pc: vcpu.pc()?,
                    syndrome,
                });
                Ok(DebugExit::Serviced)
            }
            _ => {
                if forced != 0 {
                    self.interrupted = Some(InterruptedStep {
                        pc: vcpu.pc()?,
                        retired,
                    });
                }
                Ok(DebugExit::Guest(exit))
            }
        }
    }

    /// Brings the vCPU's debug registers in line with the armed breakpoints
    /// minus `suppressed`, and with whether this entry single-steps.
    fn program(&mut self, vcpu: &Vcpu, suppressed: u16, single_step: bool) -> Result<(), HvError> {
        let mut enabled = 0;
        for (index, slot) in self.breakpoints.iter().enumerate() {
            let Some(address) = *slot else { continue };
            if self.unprogrammed & 1 << index != 0 {
                vcpu.set_sys_reg(debug_register(BREAKPOINT_VALUE, index), address)?;
            }
            enabled |= 1 << index & !suppressed;
        }
        self.unprogrammed = 0;
        for index in 0..self.breakpoint_count {
            let bit = 1 << index;
            if (enabled ^ self.hardware.enabled) & bit != 0 {
                let control = if enabled & bit != 0 {
                    BREAKPOINT_CONTROL
                } else {
                    0
                };
                vcpu.set_sys_reg(debug_register(BREAKPOINT_CONTROL_KIND, index), control)?;
            }
        }
        self.hardware.enabled = enabled;

        let trap_exceptions = enabled != 0 || single_step;
        if trap_exceptions != self.hardware.trap_exceptions {
            set_trap_debug_exceptions(vcpu, trap_exceptions)?;
            self.hardware.trap_exceptions = trap_exceptions;
        }

        let mut mdscr = self.guest.mdscr & MDSCR_TDCC;
        if enabled != 0 {
            mdscr |= MDSCR_MDE;
        }
        if single_step {
            mdscr |= MDSCR_SS;
        }
        if mdscr != self.hardware.mdscr {
            vcpu.set_sys_reg(MDSCR_EL1, mdscr)?;
            self.hardware.mdscr = mdscr;
        }
        Ok(())
    }

    /// Repairs the guest's exception registers after a stepped entry took an
    /// exception to the guest's EL1: `SPSR_EL1` captured the `forced` bits,
    /// and if the exception is the one forced to deliver a `BRK`, `ESR_EL1`
    /// says Illegal Execution state.
    fn finish_exception_entry(&self, vcpu: &Vcpu, pc: u64, forced: u64) -> Result<(), HvError> {
        let spsr = vcpu.sys_reg(SysReg::SPSR_EL1)?;
        if spsr & forced != 0 {
            vcpu.set_sys_reg(SysReg::SPSR_EL1, spsr & !forced)?;
        }
        let Some(brk) = self.brk.filter(|_| forced & PSTATE_IL != 0) else {
            return Ok(());
        };
        // An interrupt or an instruction abort can win over the forced
        // exception; then ESR_EL1 is not ours to touch and the BRK simply
        // traps again when the handler returns to it. Synchronous vectors are
        // the first of the four 0x80-spaced slots of each group.
        let synchronous = pc.wrapping_sub(vcpu.sys_reg(SysReg::VBAR_EL1)?) & 0x180 == 0;
        if synchronous && exception_class(vcpu.sys_reg(SysReg::ESR_EL1)?) == EC_ILLEGAL_STATE {
            vcpu.set_sys_reg(SysReg::ESR_EL1, brk.syndrome)?;
        }
        Ok(())
    }

    /// The shadow slot behind `DBGBVRn`/`DBGBCRn`/`DBGWVRn`/`DBGWCRn_EL1`.
    fn debug_slot(&self, register: SysReg) -> Option<(usize, usize)> {
        if register.op0() != 2 || register.op1() != 0 || register.crn() != 0 || register.op2() < 4 {
            return None;
        }
        let kind = usize::from(register.op2()) - 4;
        let index = usize::from(register.crm());
        (index < self.register_count(kind)).then_some((kind, index))
    }

    /// Value a trapped guest `MRS` of a debug register reads, or `None` if
    /// the register is not one of `MDSCR_EL1`, `OSLSR_EL1` and the
    /// breakpoint and watchpoint registers the CPU has.
    pub fn read_register(&self, register: SysReg) -> Option<u64> {
        match register {
            MDSCR_EL1 => Some(self.guest.mdscr),
            OSLSR_EL1 => Some(if self.guest.os_lock {
                OSLSR_OSLM | OSLSR_OSLK
            } else {
                OSLSR_OSLM
            }),
            _ => {
                let (kind, index) = self.debug_slot(register)?;
                Some(self.guest.debug[kind][index])
            }
        }
    }

    /// Applies a trapped guest `MSR` to a debug register, or returns `None`
    /// if the register is not one of `MDSCR_EL1`, `OSLAR_EL1` and the
    /// breakpoint and watchpoint registers the CPU has.
    pub fn write_register(&mut self, register: SysReg, value: u64) -> Option<()> {
        match register {
            MDSCR_EL1 => {
                let writable = if self.guest.os_lock {
                    MDSCR_CONTROL | MDSCR_SAVE_RESTORE
                } else {
                    MDSCR_CONTROL
                };
                self.guest.mdscr = self.guest.mdscr & !writable | value & writable;
            }
            SysReg::OSLAR_EL1 => self.guest.os_lock = value & 1 != 0,
            _ => {
                let (kind, index) = self.debug_slot(register)?;
                self.guest.debug[kind][index] = value;
            }
        }
        Some(())
    }

    /// The guest's view of the debug registers.
    pub fn guest_registers(&self) -> GuestRegisters {
        self.guest.clone()
    }

    /// Gives the guest back a view [`GuestDebug::guest_registers`] took;
    /// the hardware follows on the next entry.
    pub fn set_guest_registers(&mut self, guest: GuestRegisters) {
        self.guest = guest;
    }
}

fn set_trap_debug_exceptions(vcpu: &Vcpu, trap: bool) -> Result<(), HvError> {
    // SAFETY: the vCPU is live and owned by this thread.
    check("hv_vcpu_set_trap_debug_exceptions", unsafe {
        sys::hv_vcpu_set_trap_debug_exceptions(vcpu.id, trap)
    })
}

/// `ID_AA64DFR0_EL1` of the CPU the guest runs on. The vCPU's own copy of the
/// register is not used because the embedder may change what the guest sees.
fn host_debug_features() -> Result<u64, HvError> {
    let mut value = 0;
    // SAFETY: the configuration object is owned here and released once; the
    // out-pointer is valid.
    let code = unsafe {
        let config = sys::hv_vcpu_config_create();
        let code = sys::hv_vcpu_config_get_feature_reg(
            config,
            sys::HV_FEATURE_REG_ID_AA64DFR0_EL1,
            &mut value,
        );
        if !config.is_null() {
            sys::os_release(config);
        }
        code
    };
    check("hv_vcpu_config_get_feature_reg", code)?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    const DBGBVR5_EL1: SysReg = SysReg::new(2, 0, 0, 5, 4);
    const DBGBCR6_EL1: SysReg = SysReg::new(2, 0, 0, 6, 5);
    const DBGWVR3_EL1: SysReg = SysReg::new(2, 0, 0, 3, 6);
    const DBGWCR4_EL1: SysReg = SysReg::new(2, 0, 0, 4, 7);
    const MDCCINT_EL1: SysReg = SysReg::new(2, 0, 0, 2, 0);

    /// Six breakpoints and four watchpoints, as on Cortex-A57 and Apple cores.
    fn debug() -> GuestDebug {
        GuestDebug::new(6, 4)
    }

    #[test]
    fn mdscr_status_fields_are_writable_only_under_the_os_lock() {
        let mut debug = debug();
        assert_eq!(debug.read_register(OSLSR_EL1), Some(0b1010));
        // Locked (reset state): a restore of saved state sticks, RES0 does not.
        debug.write_register(MDSCR_EL1, u64::MAX).unwrap();
        assert_eq!(debug.read_register(MDSCR_EL1), Some(0x6CE8_F041));

        debug.write_register(SysReg::OSLAR_EL1, 0).unwrap();
        assert_eq!(debug.read_register(OSLSR_EL1), Some(0b1000));
        // Unlocked: the kernel's 0x9000 replaces the control bits only.
        debug.write_register(MDSCR_EL1, 0x9000).unwrap();
        assert_eq!(debug.read_register(MDSCR_EL1), Some(0x6CE8_9040));

        debug.write_register(SysReg::OSLAR_EL1, 1).unwrap();
        debug.write_register(MDSCR_EL1, 0).unwrap();
        assert_eq!(debug.read_register(MDSCR_EL1), Some(0));
    }

    #[test]
    fn only_registers_the_cpu_has_are_shadowed() {
        let mut debug = debug();
        debug
            .write_register(DBGBVR5_EL1, 0xFFFF_FFB4_3960_0001)
            .unwrap();
        debug.write_register(DBGWVR3_EL1, 7).unwrap();
        assert_eq!(
            debug.read_register(DBGBVR5_EL1),
            Some(0xFFFF_FFB4_3960_0001)
        );
        assert_eq!(debug.read_register(DBGWVR3_EL1), Some(7));
        // Same n, other kinds: untouched.
        assert_eq!(debug.read_register(SysReg::new(2, 0, 0, 5, 5)), Some(0));
        assert_eq!(debug.read_register(SysReg::new(2, 0, 0, 3, 7)), Some(0));

        assert_eq!(debug.write_register(DBGBCR6_EL1, 0), None);
        assert_eq!(debug.read_register(DBGWCR4_EL1), None);
        assert_eq!(debug.read_register(MDCCINT_EL1), None);
        assert_eq!(debug.write_register(MDCCINT_EL1, 0), None);
        // Read-only and write-only registers have no opposite direction.
        assert_eq!(debug.write_register(OSLSR_EL1, 0), None);
        assert_eq!(debug.read_register(SysReg::OSLAR_EL1), None);
        // Same CRn/CRm/op2 as DBGBVR5_EL1 under another op1.
        assert_eq!(debug.read_register(SysReg::new(2, 3, 0, 5, 4)), None);
    }

    #[test]
    fn breakpoint_registers_are_allocated_and_reused() {
        let mut debug = debug();
        assert_eq!(
            debug.add_breakpoint(0x8000_0002),
            Err(BreakpointError::Unaligned)
        );
        for index in 0..6 {
            debug.add_breakpoint(0x8000_0000 + 4 * index).unwrap();
        }
        debug.add_breakpoint(0x8000_0008).unwrap();
        assert_eq!(
            debug.add_breakpoint(0x9000_0000),
            Err(BreakpointError::NoFreeRegister)
        );

        assert!(debug.remove_breakpoint(0x8000_0008));
        assert!(!debug.remove_breakpoint(0x8000_0008));
        debug.add_breakpoint(0x9000_0000).unwrap();
        assert_eq!(debug.breakpoints_at(0x9000_0000), 1 << 2);
        assert_eq!(debug.unprogrammed & 1 << 2, 1 << 2);
        assert_eq!(debug.breakpoints().count(), 6);
    }

    #[test]
    fn resuming_suppresses_only_a_breakpoint_already_reported_at_the_pc() {
        const A: u64 = 0x8000_0100;
        let free = Plan::Enter {
            suppressed: 0,
            forced: 0,
        };
        let mut debug = debug();
        debug.add_breakpoint(A + 4).unwrap();
        debug.add_breakpoint(A).unwrap();

        // Never stopped at A: its breakpoint must fire.
        assert_eq!(debug.plan(A), free);
        debug.stopped_at = Some(A);
        assert_eq!(
            debug.plan(A),
            Plan::Enter {
                suppressed: 1 << 1,
                forced: PSTATE_SS
            }
        );
        // The host moved the PC to the other breakpoint: nothing reported there.
        assert_eq!(debug.plan(A + 4), free);
        // Stopped somewhere without a breakpoint: nothing to step over.
        debug.stopped_at = Some(A + 8);
        assert_eq!(debug.plan(A + 8), free);
    }

    #[test]
    fn an_interrupted_step_completes_once_its_instruction_is_accounted_for() {
        const A: u64 = 0x8000_0100;
        let mut debug = debug();
        debug.request_step();
        let stepping = Plan::Enter {
            suppressed: 0,
            forced: PSTATE_SS,
        };

        // Kicked out before the instruction ran: step it again.
        debug.interrupted = Some(InterruptedStep {
            pc: A,
            retired: false,
        });
        assert_eq!(debug.plan(A), stepping);
        // The host emulated the trapped instruction and advanced the PC.
        assert_eq!(debug.plan(A + 4), Plan::Completed);
        // The CPU retired it (HVC), wherever the PC is now.
        debug.interrupted = Some(InterruptedStep {
            pc: A,
            retired: true,
        });
        assert_eq!(debug.plan(A), Plan::Completed);

        assert!(matches!(
            debug.step_completed(A + 4, false),
            DebugExit::Step { pc, exception_entry: false } if pc == A + 4
        ));
        // The request is consumed; a later hidden step reports nothing.
        assert!(matches!(
            debug.step_completed(A + 8, false),
            DebugExit::Serviced
        ));
    }

    #[test]
    fn a_pending_brk_is_delivered_only_at_its_own_pc() {
        const A: u64 = 0x8000_0100;
        let mut debug = debug();
        debug.brk = Some(PendingBrk {
            pc: A,
            syndrome: 0xF200_0000,
        });
        assert_eq!(
            debug.plan(A),
            Plan::Enter {
                suppressed: 0,
                forced: PSTATE_SS | PSTATE_IL
            }
        );
        assert_eq!(
            debug.plan(A + 4),
            Plan::Enter {
                suppressed: 0,
                forced: 0
            }
        );
    }
}
