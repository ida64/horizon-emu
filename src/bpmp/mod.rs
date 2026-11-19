//! The BPMP: the Tegra X1's boot and power-management processor.
//!
//! The BPMP (also called the AVP or COP) is an ARM7TDMI, an ARMv4T core
//! with the ARM and Thumb instruction sets and no MMU, caches or
//! coprocessors of its own (Switchbrew "BPMP"; Linux `tegra210.dtsi`). The
//! boot ROM, package1ldr and the NX bootloader run on it.
//! Hypervisor.framework only runs AArch64, so [`Arm7`] interprets it, one
//! instruction at a time, as the ARM Architecture Reference Manual
//! (ARM DDI 0100E, "ARM ARM", covering ARMv4T) specifies; where the manual
//! leaves a choice to the implementation, the ARM7TDMI Technical Reference
//! Manual (ARM DDI 0029E, "ARM7TDMI TRM") decides.
//!
//! Everything outside the core reaches it through a [`BpmpBus`]: memory,
//! devices, and where the exception vectors lie. Encodings the manual calls
//! UNPREDICTABLE are not guessed at: the core stops with a [`Fault`] that
//! names the instruction, as it does for an access the bus does not model.
//!
//! The core keeps no host time; one [`Arm7::step`] is one instruction (or
//! one exception entry), and whoever runs it decides how that maps onto
//! guest time.
//!
//! An [`Assist`] beside the core caches the instructions it fetches from
//! memory that announces its other writers, and keeps a trace of the last
//! steps and bus accesses for diagnosing where firmware stopped; neither
//! changes what the core does (`bench.rs` measures the difference).

use std::{fmt, sync::Mutex};

use serde::{Deserialize, Serialize};

use crate::snapshot::{Snapshot, SnapshotError, decode, encode};

mod arm;
mod assist;
#[cfg(test)]
mod bench;
mod thumb;

pub use assist::{Assist, Outcome, Trace, TracedAccess, TracedStep};
use assist::{CodeCache, Invalidating, TracingBus};

#[cfg(test)]
mod tests;

/// Why the bus did not complete an access.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BusError {
    /// The system answered the access with an error, as hardware does (an
    /// address nothing decodes, a block the BPMP may not reach): the core
    /// takes a prefetch abort for an instruction fetch and a data abort for
    /// a load or store (ARM ARM A2, "Prefetch Abort", "Data Abort").
    Abort,
    /// The access reached something the emulator does not model: the core
    /// stops with [`FaultKind::Unmodelled`] instead of inventing a value.
    Unmodelled(String),
}

/// What the core sees of the rest of the system.
///
/// Accesses are little-endian and naturally aligned (the core aligns word
/// and block transfers itself and refuses unaligned halfwords); `size` is
/// 1, 2 or 4 and a read returns the value zero-extended.
pub trait BpmpBus {
    /// Reads `size` bytes at `address`.
    fn read(&mut self, address: u32, size: u8) -> Result<u32, BusError>;
    /// Writes the low `size` bytes of `value` at `address`.
    fn write(&mut self, address: u32, size: u8, value: u32) -> Result<(), BusError>;
    /// Reads the instruction of `size` bytes at `address`. The default is
    /// [`BpmpBus::read`]; a bus overrides it where data may be read and
    /// nothing run.
    fn fetch(&mut self, address: u32, size: u8) -> Result<u32, BusError> {
        self.read(address, size)
    }
    /// Reads consecutive words from `address` (aligned) into `words`, as
    /// that many 4-byte [`BpmpBus::read`]s in ascending order would; at
    /// the first that fails, stops and returns its index and error. `LDM`
    /// and `POP` transfer through it, so a bus can serve a block in one go.
    fn read_words(&mut self, address: u32, words: &mut [u32]) -> Result<(), (usize, BusError)> {
        for (i, word) in words.iter_mut().enumerate() {
            *word = self
                .read(address.wrapping_add(4 * i as u32), 4)
                .map_err(|error| (i, error))?;
        }
        Ok(())
    }
    /// Writes `words` to consecutive words from `address`, as
    /// [`BpmpBus::read_words`] reads them (`STM`, `PUSH`).
    fn write_words(&mut self, address: u32, words: &[u32]) -> Result<(), (usize, BusError)> {
        for (i, &word) in words.iter().enumerate() {
            self.write(address.wrapping_add(4 * i as u32), 4, word)
                .map_err(|error| (i, error))?;
        }
        Ok(())
    }
    /// Where the core continues on taking `exception`, after it has
    /// switched mode and saved the return address and CPSR.
    ///
    /// The default is the architected low vector, `exception.offset()`
    /// (ARM ARM A2, "Exceptions"). The T210 does not use it: its exception vector
    /// registers (`EVP_COP_*_VECTOR`, `0x6000_F200` + offset, one word per
    /// vector in the architected order) hold the address of each handler,
    /// and the BPMP continues there \[inference: U-Boot and Linux program
    /// the CCPLEX's `EVP_CPU_RESET_VECTOR` at `0x6000_F100` with the entry
    /// address itself, and the COP's registers are laid out the same way\].
    /// A machine that maps those registers answers from them here. The
    /// returned address is aligned to the state the core enters (ARM).
    fn exception_vector(&mut self, exception: Exception) -> u32 {
        exception.offset()
    }
    /// Asked after every instruction [`Arm7::run`] executes; `true` ends
    /// the run early. ARMv4T has no wait-for-interrupt instruction: the
    /// BPMP halts itself through the flow controller, a write the bus sees,
    /// and the bus answers `true` here once it has.
    fn should_pause(&mut self) -> bool {
        false
    }
}

/// An exception, with its architected vector and mode (ARM ARM A2,
/// "Exceptions", table of exception vectors).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Exception {
    Reset,
    Undefined,
    SoftwareInterrupt,
    PrefetchAbort,
    DataAbort,
    Irq,
    Fiq,
}

impl Exception {
    /// The architected low vector address.
    pub fn offset(self) -> u32 {
        match self {
            Exception::Reset => 0x00,
            Exception::Undefined => 0x04,
            Exception::SoftwareInterrupt => 0x08,
            Exception::PrefetchAbort => 0x0C,
            Exception::DataAbort => 0x10,
            Exception::Irq => 0x18,
            Exception::Fiq => 0x1C,
        }
    }

    /// The mode the core enters it in.
    pub fn mode(self) -> Mode {
        match self {
            Exception::Reset | Exception::SoftwareInterrupt => Mode::Supervisor,
            Exception::Undefined => Mode::Undefined,
            Exception::PrefetchAbort | Exception::DataAbort => Mode::Abort,
            Exception::Irq => Mode::Irq,
            Exception::Fiq => Mode::Fiq,
        }
    }
}

/// A processor mode, by its CPSR mode bits (ARM ARM A2, "Processor modes").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Mode {
    User = 0x10,
    Fiq = 0x11,
    Irq = 0x12,
    Supervisor = 0x13,
    Abort = 0x17,
    Undefined = 0x1B,
    System = 0x1F,
}

impl Mode {
    /// The mode `bits` (CPSR\[4:0\]) select; `None` for the reserved
    /// encodings, whose effect is UNPREDICTABLE.
    pub fn from_bits(bits: u32) -> Option<Mode> {
        Some(match bits & 0x1F {
            0x10 => Mode::User,
            0x11 => Mode::Fiq,
            0x12 => Mode::Irq,
            0x13 => Mode::Supervisor,
            0x17 => Mode::Abort,
            0x1B => Mode::Undefined,
            0x1F => Mode::System,
            _ => return None,
        })
    }

    /// Which set of banked R13/R14 (and SPSR) the mode uses: User and
    /// System share set 0 (ARM ARM A2, "Registers", register organization).
    fn bank(self) -> usize {
        match self {
            Mode::User | Mode::System => 0,
            Mode::Fiq => 1,
            Mode::Irq => 2,
            Mode::Supervisor => 3,
            Mode::Abort => 4,
            Mode::Undefined => 5,
        }
    }

    /// Whether the mode has an SPSR: all but User and System.
    pub fn has_spsr(self) -> bool {
        self.bank() != 0
    }
}

/// What one [`Arm7::step`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    /// One instruction executed (or its condition failed).
    Executed,
    /// The core entered the exception: raised by the instruction it
    /// executed (undefined instruction, SWI, data abort) or taken instead
    /// of executing one (IRQ, FIQ, prefetch abort).
    Exception(Exception),
}

/// Why the core stopped. The instruction did not complete: PC is back at
/// it and no register it would write was written, but stores a block
/// transfer made before the failing one stay made.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fault {
    /// Address of the instruction.
    pub pc: u32,
    /// Its encoding (a Thumb instruction in the low half); `None` if the
    /// fetch itself failed.
    pub instruction: Option<u32>,
    /// Whether the core was in Thumb state.
    pub thumb: bool,
    pub kind: FaultKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FaultKind {
    /// The bus does not model the access.
    Unmodelled {
        address: u32,
        size: u8,
        write: bool,
        reason: String,
    },
    /// The architecture leaves the instruction's effect UNPREDICTABLE (or
    /// the encoding undefined in a way the core will not guess at).
    Unpredictable(&'static str),
}

impl fmt::Display for Fault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = if self.thumb { "Thumb" } else { "ARM" };
        write!(f, "BPMP stopped at {:#010x} ({state}", self.pc)?;
        match self.instruction {
            Some(insn) if self.thumb => write!(f, " {insn:04x}): ")?,
            Some(insn) => write!(f, " {insn:08x}): ")?,
            None => write!(f, ", fetch): ")?,
        }
        match &self.kind {
            FaultKind::Unmodelled {
                address,
                size,
                write,
                reason,
            } => {
                let access = if *write { "write" } else { "read" };
                write!(f, "{size}-byte {access} at {address:#010x}: {reason}")
            }
            FaultKind::Unpredictable(what) => write!(f, "UNPREDICTABLE: {what}"),
        }
    }
}

impl std::error::Error for Fault {}

/// How an instruction ended other than normally; [`Arm7::step`] turns it
/// into an exception entry or a [`Fault`]. Small, so that the result of
/// every instruction stays in registers.
#[derive(Debug)]
enum Stop {
    Exception(Exception),
    /// A [`FaultKind::Unmodelled`].
    Unmodelled(Box<FaultKind>),
    Unpredictable(&'static str),
}

type Exec = Result<(), Stop>;

/// CPSR/SPSR bits (ARM ARM A2, "Program status registers").
pub const PSR_N: u32 = 1 << 31;
pub const PSR_Z: u32 = 1 << 30;
pub const PSR_C: u32 = 1 << 29;
pub const PSR_V: u32 = 1 << 28;
pub const PSR_I: u32 = 1 << 7;
pub const PSR_F: u32 = 1 << 6;
pub const PSR_T: u32 = 1 << 5;
const PSR_MODE: u32 = 0x1F;
/// The bits ARMv4T defines; the rest of a PSR is reserved and reads as
/// zero here \[assumption: ARM ARM A2 only requires software to preserve
/// them; what the ARM7TDMI stores in them is not documented\].
const PSR_DEFINED: u32 = 0xF000_00FF;

/// An ARMv4T core (ARM7TDMI).
///
/// `reg(15)` is the address of the next instruction to execute; an
/// instruction that reads R15 sees it plus 8 (ARM) or 4 (Thumb), as the
/// pipeline makes it (ARM ARM A2, "General-purpose registers", R15).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Arm7 {
    /// The registers of the current mode.
    r: [u32; 16],
    cpsr: u32,
    /// R13 and R14 of each bank (see [`Mode::bank`]); the entry of the
    /// current mode's bank is stale, its values are in `r`.
    r13_r14: [[u32; 2]; 6],
    /// R8 to R12 outside FIQ mode (0) and of FIQ mode (1); the current
    /// one is stale.
    r8_r12: [[u32; 5]; 2],
    /// SPSR of each bank; entry 0 (User/System) is unused.
    spsr: [u32; 6],
    irq: bool,
    fiq: bool,
}

impl Default for Arm7 {
    fn default() -> Self {
        Self::new()
    }
}

impl Arm7 {
    /// A core just out of reset at address 0.
    pub fn new() -> Self {
        let mut core = Self {
            r: [0; 16],
            cpsr: Mode::Supervisor as u32 | PSR_I | PSR_F,
            r13_r14: [[0; 2]; 6],
            r8_r12: [[0; 5]; 2],
            spsr: [0; 6],
            irq: false,
            fiq: false,
        };
        core.reset(0);
        core
    }

    /// Takes the reset exception and starts at `pc` (ARM ARM A2, "Reset"):
    /// Supervisor mode, ARM state, IRQ and FIQ masked. The architecture
    /// leaves every register's value UNPREDICTABLE after reset; they are
    /// zero here \[assumption: firmware sets each before use; what the
    /// boot ROM leaves is set by whoever applies its handoff\]. The IRQ
    /// and FIQ lines are inputs and keep their level.
    pub fn reset(&mut self, pc: u32) {
        self.r = [0; 16];
        self.r13_r14 = [[0; 2]; 6];
        self.r8_r12 = [[0; 5]; 2];
        self.spsr = [0; 6];
        self.cpsr = Mode::Supervisor as u32 | PSR_I | PSR_F;
        self.r[15] = pc & !3;
    }

    /// Register `n` of the current mode; R15 is the address of the next
    /// instruction.
    pub fn reg(&self, n: usize) -> u32 {
        self.r[n]
    }

    /// Sets register `n` of the current mode; R15 is aligned to the
    /// current state.
    pub fn set_reg(&mut self, n: usize, value: u32) {
        if n == 15 {
            self.write_pc(value);
        } else {
            self.r[n] = value;
        }
    }

    /// Register `n` as `mode` sees it, whatever the current mode.
    pub fn banked_reg(&self, mode: Mode, n: usize) -> u32 {
        let current = self.mode();
        match n {
            8..=12 if (mode == Mode::Fiq) != (current == Mode::Fiq) => {
                self.r8_r12[usize::from(mode == Mode::Fiq)][n - 8]
            }
            13 | 14 if mode.bank() != current.bank() => self.r13_r14[mode.bank()][n - 13],
            _ => self.r[n],
        }
    }

    /// Address of the next instruction.
    pub fn pc(&self) -> u32 {
        self.r[15]
    }

    pub fn cpsr(&self) -> u32 {
        self.cpsr
    }

    /// Sets the CPSR, switching register banks if the mode changes.
    ///
    /// # Panics
    /// If the mode bits are reserved.
    pub fn set_cpsr(&mut self, value: u32) {
        assert!(
            Mode::from_bits(value).is_some(),
            "reserved mode bits in CPSR {value:#x}"
        );
        self.write_cpsr(value);
    }

    /// SPSR of the current mode; `None` in User and System mode.
    pub fn spsr(&self) -> Option<u32> {
        let mode = self.mode();
        mode.has_spsr().then(|| self.spsr[mode.bank()])
    }

    /// SPSR of `mode`; `None` for User and System.
    pub fn banked_spsr(&self, mode: Mode) -> Option<u32> {
        mode.has_spsr().then(|| self.spsr[mode.bank()])
    }

    pub fn mode(&self) -> Mode {
        Mode::from_bits(self.cpsr).expect("the CPSR always holds a valid mode")
    }

    pub fn thumb(&self) -> bool {
        self.cpsr & PSR_T != 0
    }

    /// Drives the nIRQ input: `true` requests an interrupt, which the core
    /// takes before its next instruction while CPSR.I is clear.
    pub fn set_irq(&mut self, asserted: bool) {
        self.irq = asserted;
    }

    /// Drives the nFIQ input, as [`Arm7::set_irq`] with CPSR.F.
    pub fn set_fiq(&mut self, asserted: bool) {
        self.fiq = asserted;
    }

    /// Executes one instruction, or takes a pending FIQ or IRQ instead
    /// (FIQ first, ARM ARM A2, "Exception priorities").
    pub fn step(&mut self, bus: &mut dyn BpmpBus) -> Result<Step, Fault> {
        self.step_inner(bus, None, None)
    }

    /// Steps until `budget` steps are done or the bus asks to pause
    /// ([`BpmpBus::should_pause`]); returns the steps done.
    pub fn run(&mut self, bus: &mut dyn BpmpBus, budget: u64) -> Result<u64, Fault> {
        self.run_inner(bus, None, budget)
    }

    fn run_inner(
        &mut self,
        bus: &mut dyn BpmpBus,
        mut cache: Option<&mut CodeCache>,
        budget: u64,
    ) -> Result<u64, Fault> {
        let mut done = 0;
        while done < budget {
            self.step_inner(bus, cache.as_deref_mut(), None)?;
            done += 1;
            if bus.should_pause() {
                break;
            }
        }
        Ok(done)
    }

    /// [`Arm7::step`], fetching through `cache` if there is one, and
    /// keeping the instruction's data accesses in `accesses` if asked.
    #[inline]
    fn step_inner(
        &mut self,
        bus: &mut dyn BpmpBus,
        mut cache: Option<&mut CodeCache>,
        accesses: Option<&mut Trace>,
    ) -> Result<Step, Fault> {
        // An interrupt returns to the instruction it was taken before:
        // LR = that address + 4, so SUBS PC, LR, #4 resumes it (A2, "Interrupt
        // request (IRQ) exception").
        if self.fiq && self.cpsr & PSR_F == 0 {
            let link = self.r[15].wrapping_add(4);
            self.enter_exception(bus, Exception::Fiq, link);
            return Ok(Step::Exception(Exception::Fiq));
        }
        if self.irq && self.cpsr & PSR_I == 0 {
            let link = self.r[15].wrapping_add(4);
            self.enter_exception(bus, Exception::Irq, link);
            return Ok(Step::Exception(Exception::Irq));
        }
        let pc = self.r[15];
        let thumb = self.thumb();
        let width = if thumb { 2 } else { 4 };
        let fetched = match &mut cache {
            Some(cache) => cache.fetch(bus, pc, width),
            None => bus.fetch(pc, width),
        };
        let insn = match fetched {
            Ok(insn) => insn,
            // A2, "Prefetch Abort": LR = aborted instruction + 4 in either state.
            Err(BusError::Abort) => {
                self.enter_exception(bus, Exception::PrefetchAbort, pc.wrapping_add(4));
                return Ok(Step::Exception(Exception::PrefetchAbort));
            }
            Err(BusError::Unmodelled(reason)) => {
                return Err(Fault {
                    pc,
                    instruction: None,
                    thumb,
                    kind: FaultKind::Unmodelled {
                        address: pc,
                        size: width,
                        write: false,
                        reason,
                    },
                });
            }
        };
        self.r[15] = pc.wrapping_add(u32::from(width));
        let result = match (cache, accesses) {
            (None, None) => self.execute(bus, thumb, insn),
            (Some(cache), None) => {
                let mut view = Invalidating { inner: bus, cache };
                self.execute(&mut view, thumb, insn)
            }
            (None, Some(trace)) => {
                let mut view = TracingBus {
                    inner: bus,
                    trace,
                    pc,
                };
                self.execute(&mut view, thumb, insn)
            }
            (Some(cache), Some(trace)) => {
                let mut inner = Invalidating { inner: bus, cache };
                let mut view = TracingBus {
                    inner: &mut inner,
                    trace,
                    pc,
                };
                self.execute(&mut view, thumb, insn)
            }
        };
        let kind = match result {
            Ok(()) => return Ok(Step::Executed),
            Err(Stop::Exception(exception)) => {
                let link = match exception {
                    // A2, "Data Abort": LR = aborting instruction + 8 in either state.
                    Exception::DataAbort => pc.wrapping_add(8),
                    // A2, "Undefined Instruction", "Software Interrupt": LR = the
                    // next instruction.
                    _ => pc.wrapping_add(u32::from(width)),
                };
                self.enter_exception(bus, exception, link);
                return Ok(Step::Exception(exception));
            }
            Err(Stop::Unmodelled(kind)) => *kind,
            Err(Stop::Unpredictable(what)) => FaultKind::Unpredictable(what),
        };
        self.r[15] = pc;
        Err(Fault {
            pc,
            instruction: Some(if thumb { insn & 0xFFFF } else { insn }),
            thumb,
            kind,
        })
    }

    fn execute(&mut self, bus: &mut dyn BpmpBus, thumb: bool, insn: u32) -> Exec {
        if thumb {
            self.execute_thumb(bus, insn as u16)
        } else {
            self.execute_arm(bus, insn)
        }
    }

    /// Exception entry (ARM ARM A2, "Exceptions"): R14 and SPSR of the exception's
    /// mode get the return address and the old CPSR; the core enters the
    /// mode in ARM state with IRQ masked (and FIQ for FIQ and reset).
    fn enter_exception(&mut self, bus: &mut dyn BpmpBus, exception: Exception, link: u32) {
        let old = self.cpsr;
        let mode = exception.mode();
        let mut cpsr = (old & !(PSR_MODE | PSR_T)) | mode as u32 | PSR_I;
        if matches!(exception, Exception::Fiq | Exception::Reset) {
            cpsr |= PSR_F;
        }
        self.write_cpsr(cpsr);
        self.spsr[mode.bank()] = old;
        self.r[14] = link;
        self.r[15] = bus.exception_vector(exception) & !3;
    }

    /// Sets the CPSR to `value`, whose mode bits are valid, swapping the
    /// banked registers if the mode changes.
    fn write_cpsr(&mut self, value: u32) {
        let value = value & PSR_DEFINED;
        let old = self.mode();
        let new = Mode::from_bits(value).expect("mode bits checked by the caller");
        if old.bank() != new.bank() {
            self.r13_r14[old.bank()] = [self.r[13], self.r[14]];
            [self.r[13], self.r[14]] = self.r13_r14[new.bank()];
        }
        let (old_fiq, new_fiq) = (old == Mode::Fiq, new == Mode::Fiq);
        if old_fiq != new_fiq {
            self.r8_r12[usize::from(old_fiq)].copy_from_slice(&self.r[8..13]);
            self.r[8..13].copy_from_slice(&self.r8_r12[usize::from(new_fiq)]);
        }
        self.cpsr = value;
    }

    /// The SPSR an exception return copies into the CPSR (`MOVS PC`,
    /// `LDM ... {PC}^`): UNPREDICTABLE in User and System mode, which have
    /// none, and for reserved mode bits in it (A4, data-processing pages and
    /// "LDM (3)").
    fn return_spsr(&self) -> Result<u32, Stop> {
        let spsr = self.spsr().ok_or(Stop::Unpredictable(
            "exception return in a mode without SPSR",
        ))?;
        Mode::from_bits(spsr).ok_or(Stop::Unpredictable("SPSR holds reserved mode bits"))?;
        Ok(spsr)
    }

    /// Branches to `target`, aligned to the current state: an ARMv4T core
    /// ignores bit 0 of a Thumb address and bits 1:0 of an ARM one
    /// \[inference: ARM ARM A2 makes unaligned ARM-state writes
    /// UNPREDICTABLE; the ARM7TDMI TRM's address bus never drives them on
    /// an ARM fetch\].
    fn write_pc(&mut self, target: u32) {
        self.r[15] = if self.thumb() {
            target & !1
        } else {
            target & !3
        };
    }

    fn flag(&self, bit: u32) -> bool {
        self.cpsr & bit != 0
    }

    fn set_flag(&mut self, bit: u32, value: bool) {
        if value {
            self.cpsr |= bit;
        } else {
            self.cpsr &= !bit;
        }
    }

    fn set_nz(&mut self, result: u32) {
        self.set_flag(PSR_N, result >> 31 != 0);
        self.set_flag(PSR_Z, result == 0);
    }

    /// Whether condition `cond` (instruction bits 31:28) passes
    /// (ARM ARM A3, "The condition field"). NV is UNPREDICTABLE on ARMv4.
    fn condition(&self, cond: u32) -> Result<bool, Stop> {
        let (n, z, c, v) = (
            self.flag(PSR_N),
            self.flag(PSR_Z),
            self.flag(PSR_C),
            self.flag(PSR_V),
        );
        Ok(match cond {
            0x0 => z,
            0x1 => !z,
            0x2 => c,
            0x3 => !c,
            0x4 => n,
            0x5 => !n,
            0x6 => v,
            0x7 => !v,
            0x8 => c && !z,
            0x9 => !c || z,
            0xA => n == v,
            0xB => n != v,
            0xC => !z && n == v,
            0xD => z || n != v,
            0xE => true,
            _ => return Err(Stop::Unpredictable("condition code NV")),
        })
    }

    /// Register `n` of the User bank, for `LDM`/`STM` with `^` (A4, "LDM
    /// (2)", "STM (2)").
    fn user_reg(&self, n: usize) -> u32 {
        self.banked_reg(Mode::User, n)
    }

    fn set_user_reg(&mut self, n: usize, value: u32) {
        let current = self.mode();
        match n {
            8..=12 if current == Mode::Fiq => self.r8_r12[0][n - 8] = value,
            13 | 14 if current.bank() != 0 => self.r13_r14[0][n - 13] = value,
            _ => self.r[n] = value,
        }
    }

    // Memory. ARM ARM A2, "Unaligned access": words are aligned by the core; an unaligned
    // word load returns the aligned word rotated so the addressed byte is
    // the least significant (A4, "LDR"); unaligned halfwords are
    // UNPREDICTABLE on ARMv4.

    fn bus_read(&mut self, bus: &mut dyn BpmpBus, address: u32, size: u8) -> Result<u32, Stop> {
        bus.read(address, size)
            .map_err(|error| bus_stop(error, address, size, false))
    }

    fn bus_write(&mut self, bus: &mut dyn BpmpBus, address: u32, size: u8, value: u32) -> Exec {
        bus.write(address, size, value)
            .map_err(|error| bus_stop(error, address, size, true))
    }

    /// `LDR`: the aligned word, rotated by the address's low bits.
    fn load_word(&mut self, bus: &mut dyn BpmpBus, address: u32) -> Result<u32, Stop> {
        let word = self.bus_read(bus, address & !3, 4)?;
        Ok(word.rotate_right((address & 3) * 8))
    }

    fn load_half(&mut self, bus: &mut dyn BpmpBus, address: u32) -> Result<u32, Stop> {
        if address & 1 != 0 {
            return Err(Stop::Unpredictable("unaligned halfword load"));
        }
        Ok(self.bus_read(bus, address, 2)? & 0xFFFF)
    }

    fn load_byte(&mut self, bus: &mut dyn BpmpBus, address: u32) -> Result<u32, Stop> {
        Ok(self.bus_read(bus, address, 1)? & 0xFF)
    }

    /// `STR`: the core drives the aligned address (ARM7TDMI TRM, memory interface).
    fn store_word(&mut self, bus: &mut dyn BpmpBus, address: u32, value: u32) -> Exec {
        self.bus_write(bus, address & !3, 4, value)
    }

    fn store_half(&mut self, bus: &mut dyn BpmpBus, address: u32, value: u32) -> Exec {
        if address & 1 != 0 {
            return Err(Stop::Unpredictable("unaligned halfword store"));
        }
        self.bus_write(bus, address, 2, value & 0xFFFF)
    }

    fn store_byte(&mut self, bus: &mut dyn BpmpBus, address: u32, value: u32) -> Exec {
        self.bus_write(bus, address, 1, value & 0xFF)
    }

    /// Loads the registers of `list` from consecutive words at `start`
    /// (lowest register at the lowest address; bits 1:0 of the address
    /// ignored), then writes `base` back if asked. `user` loads the User
    /// bank; `restore` is the SPSR to copy into the CPSR after loading PC.
    ///
    /// On a data abort the ARM7TDMI's "base updated" model applies: the
    /// base is written back and no loaded register is changed (ARM7TDMI
    /// TRM, "Data aborts" and "Block data transfer"); its loaded value
    /// overrides the written-back one when the base is in the list (the
    /// same "Block data transfer" section).
    #[allow(clippy::too_many_arguments)]
    fn load_multiple(
        &mut self,
        bus: &mut dyn BpmpBus,
        list: u16,
        start: u32,
        base: usize,
        writeback: Option<u32>,
        user: bool,
        restore: Option<u32>,
    ) -> Exec {
        let mut words = [0u32; 16];
        let count = list.count_ones() as usize;
        let start = start & !3;
        let mut aborted = None;
        if let Err((index, error)) = bus.read_words(start, &mut words[..count]) {
            let address = start.wrapping_add(4 * index as u32);
            match bus_stop(error, address, 4, false) {
                stop @ Stop::Exception(_) => aborted = Some(stop),
                stop => return Err(stop),
            }
        }
        if let Some(new_base) = writeback {
            self.r[base] = new_base;
        }
        if let Some(stop) = aborted {
            return Err(stop);
        }
        let mut values = [0u32; 16];
        for (n, word) in registers(list).zip(words) {
            values[n] = word;
        }
        for n in registers(list) {
            match n {
                15 => {}
                _ if user => self.set_user_reg(n, values[n]),
                _ => self.r[n] = values[n],
            }
        }
        if list & 0x8000 != 0 {
            if let Some(spsr) = restore {
                self.write_cpsr(spsr);
            }
            self.write_pc(values[15]);
        }
        Ok(())
    }

    /// Stores the registers of `list` to consecutive words at `start`,
    /// then writes `base` back if asked. `pc` is the value stored for R15
    /// (the ARM7TDMI stores the instruction's address + 12, ARM7TDMI TRM,
    /// "Block data transfer"). A base in the list stores its original value
    /// if it is the lowest register listed, the written-back one otherwise
    /// (same section). The transfer stops at the first data abort, base
    /// written back \[assumption: the ARM7TDMI goes on driving the
    /// remaining stores and leaves it to the memory system to refuse them
    /// (same section); here they
    /// are not made\].
    #[allow(clippy::too_many_arguments)]
    fn store_multiple(
        &mut self,
        bus: &mut dyn BpmpBus,
        list: u16,
        start: u32,
        base: usize,
        writeback: Option<u32>,
        user: bool,
        pc: u32,
    ) -> Exec {
        let start = start & !3;
        let lowest = list.trailing_zeros() as usize;
        let mut words = [0u32; 16];
        let count = list.count_ones() as usize;
        for (n, word) in registers(list).zip(&mut words) {
            *word = match (n, writeback) {
                (15, _) => pc,
                (n, Some(new_base)) if n == base && n != lowest => new_base,
                (n, _) if user => self.user_reg(n),
                (n, _) => self.r[n],
            };
        }
        let mut result = Ok(());
        if let Err((index, error)) = bus.write_words(start, &words[..count]) {
            let address = start.wrapping_add(4 * index as u32);
            match bus_stop(error, address, 4, true) {
                stop @ Stop::Exception(_) => result = Err(stop),
                stop => return Err(stop),
            }
        }
        if let Some(new_base) = writeback {
            self.r[base] = new_base;
        }
        result
    }
}

/// The registers in `list`, lowest first.
fn registers(list: u16) -> impl Iterator<Item = usize> {
    let mut left = list;
    std::iter::from_fn(move || {
        (left != 0).then(|| {
            let n = left.trailing_zeros() as usize;
            left &= left - 1;
            n
        })
    })
}

fn bus_stop(error: BusError, address: u32, size: u8, write: bool) -> Stop {
    match error {
        BusError::Abort => Stop::Exception(Exception::DataAbort),
        BusError::Unmodelled(reason) => Stop::Unmodelled(Box::new(FaultKind::Unmodelled {
            address,
            size,
            write,
            reason,
        })),
    }
}

/// `a + b + carry`, with the carry out and signed overflow (ARM ARM
/// A2, "Program status registers": C is the unsigned carry, V the signed overflow; subtraction is
/// `a + !b + 1`, so C is NOT borrow).
fn add_with_carry(a: u32, b: u32, carry: bool) -> (u32, bool, bool) {
    let sum = u64::from(a) + u64::from(b) + u64::from(carry);
    let result = sum as u32;
    let overflow = (a ^ result) & (b ^ result) & 0x8000_0000 != 0;
    (result, sum >> 32 != 0, overflow)
}

/// The barrel shifter's shift types (ARM ARM A5, "Addressing Mode 1").
#[derive(Debug, Clone, Copy)]
enum Shift {
    Lsl,
    Lsr,
    Asr,
    Ror,
}

impl Shift {
    fn from_bits(bits: u32) -> Shift {
        match bits & 3 {
            0 => Shift::Lsl,
            1 => Shift::Lsr,
            2 => Shift::Asr,
            _ => Shift::Ror,
        }
    }
}

/// Shift by an immediate `amount` (0 to 31), with the shifter carry out
/// (ARM ARM A5, the immediate shift operands): `LSR #0` and `ASR #0` encode a shift by
/// 32, `ROR #0` encodes RRX, and `LSL #0` leaves the value and C alone.
fn shift_immediate(kind: Shift, value: u32, amount: u32, carry: bool) -> (u32, bool) {
    match (kind, amount) {
        (Shift::Lsl, 0) => (value, carry),
        (Shift::Lsl, n) => (value << n, value >> (32 - n) & 1 != 0),
        (Shift::Lsr, 0) => (0, value >> 31 != 0),
        (Shift::Lsr, n) => (value >> n, value >> (n - 1) & 1 != 0),
        (Shift::Asr, 0) => (((value as i32) >> 31) as u32, value >> 31 != 0),
        (Shift::Asr, n) => (((value as i32) >> n) as u32, value >> (n - 1) & 1 != 0),
        (Shift::Ror, 0) => ((u32::from(carry) << 31) | value >> 1, value & 1 != 0),
        (Shift::Ror, n) => (value.rotate_right(n), value >> (n - 1) & 1 != 0),
    }
}

/// Shift by the bottom byte of a register (ARM ARM A5, the register shift operands):
/// 0 leaves the value and C alone; shifts of 32 or more fill.
fn shift_register(kind: Shift, value: u32, amount: u32, carry: bool) -> (u32, bool) {
    let amount = amount & 0xFF;
    if amount == 0 {
        return (value, carry);
    }
    match kind {
        Shift::Lsl if amount < 32 => (value << amount, value >> (32 - amount) & 1 != 0),
        Shift::Lsl if amount == 32 => (0, value & 1 != 0),
        Shift::Lsl => (0, false),
        Shift::Lsr if amount < 32 => (value >> amount, value >> (amount - 1) & 1 != 0),
        Shift::Lsr if amount == 32 => (0, value >> 31 != 0),
        Shift::Lsr => (0, false),
        Shift::Asr if amount < 32 => (
            ((value as i32) >> amount) as u32,
            value >> (amount - 1) & 1 != 0,
        ),
        Shift::Asr => (((value as i32) >> 31) as u32, value >> 31 != 0),
        Shift::Ror if amount & 31 == 0 => (value, value >> 31 != 0),
        Shift::Ror => {
            let n = amount & 31;
            (value.rotate_right(n), value >> (n - 1) & 1 != 0)
        }
    }
}

/// The core's state is all architectural; a machine keeps it behind a
/// mutex shared with whatever drives the interrupt lines.
impl Snapshot for Mutex<Arm7> {
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        encode(&*self.lock().unwrap())
    }

    fn restore(&self, state: &[u8]) -> Result<(), SnapshotError> {
        let core: Arm7 = decode("BPMP", state)?;
        if Mode::from_bits(core.cpsr).is_none() {
            return Err(SnapshotError::Corrupt(format!(
                "BPMP: CPSR {:#x} holds reserved mode bits",
                core.cpsr
            )));
        }
        *self.lock().unwrap() = core;
        Ok(())
    }
}
