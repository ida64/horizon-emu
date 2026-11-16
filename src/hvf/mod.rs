//! CPU backend on Apple's Hypervisor.framework: the guest kernel runs natively
//! at EL1 on the host core, and everything that would reach EL2/EL3 or leave
//! mapped memory comes back as an [`Exit`].

pub mod debug;
mod sys;

use std::sync::OnceLock;
use std::time::Duration;
use std::{fmt, ptr};

use serde::{Deserialize, Serialize};

use crate::arm64::SysReg;

/// A core's state as [`Vcpu::save_state`] takes it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct VcpuState {
    pub x: [u64; 31],
    pub pc: u64,
    pub cpsr: u64,
    pub fpcr: u64,
    pub fpsr: u64,
    #[serde(with = "crate::snapshot::array")]
    pub q: [u128; 32],
    /// [`SAVED_SYS_REGS`] by encoding, with their values.
    pub sys: Vec<(u16, u64)>,
}

/// The EL0/EL1 system registers Hypervisor.framework holds for the guest
/// that a snapshot saves (`hv_vcpu_types.h`, `HV_SYS_REG_*`): translation,
/// exception, thread, context and virtual timer state. Identification
/// registers are set when a core is made; debug registers are shadowed.
pub const SAVED_SYS_REGS: [SysReg; 25] = [
    SysReg::new(3, 0, 1, 0, 0),  // SCTLR_EL1
    SysReg::new(3, 0, 1, 0, 2),  // CPACR_EL1
    SysReg::new(3, 0, 2, 0, 0),  // TTBR0_EL1
    SysReg::new(3, 0, 2, 0, 1),  // TTBR1_EL1
    SysReg::new(3, 0, 2, 0, 2),  // TCR_EL1
    SysReg::new(3, 0, 4, 0, 0),  // SPSR_EL1
    SysReg::new(3, 0, 4, 0, 1),  // ELR_EL1
    SysReg::new(3, 0, 4, 1, 0),  // SP_EL0
    SysReg::new(3, 0, 5, 1, 0),  // AFSR0_EL1
    SysReg::new(3, 0, 5, 1, 1),  // AFSR1_EL1
    SysReg::new(3, 0, 5, 2, 0),  // ESR_EL1
    SysReg::new(3, 0, 6, 0, 0),  // FAR_EL1
    SysReg::new(3, 0, 7, 4, 0),  // PAR_EL1
    SysReg::new(3, 0, 10, 2, 0), // MAIR_EL1
    SysReg::new(3, 0, 10, 3, 0), // AMAIR_EL1
    SysReg::new(3, 0, 12, 0, 0), // VBAR_EL1
    SysReg::new(3, 0, 13, 0, 1), // CONTEXTIDR_EL1
    SysReg::new(3, 0, 13, 0, 4), // TPIDR_EL1
    SysReg::new(3, 0, 14, 1, 0), // CNTKCTL_EL1
    SysReg::new(3, 2, 0, 0, 0),  // CSSELR_EL1
    SysReg::new(3, 3, 13, 0, 2), // TPIDR_EL0
    SysReg::new(3, 3, 13, 0, 3), // TPIDRRO_EL0
    SysReg::new(3, 3, 14, 3, 1), // CNTV_CTL_EL0
    SysReg::new(3, 3, 14, 3, 2), // CNTV_CVAL_EL0
    SysReg::new(3, 4, 4, 1, 0),  // SP_EL1
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HvError {
    call: &'static str,
    code: sys::hv_return_t,
}

impl fmt::Display for HvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} failed with {:#010x}", self.call, self.code as u32)?;
        if self.code == sys::HV_DENIED {
            write!(
                f,
                " (binary lacks the com.apple.security.hypervisor entitlement; run through cargo)"
            )?;
        }
        Ok(())
    }
}

impl std::error::Error for HvError {}

fn check(call: &'static str, code: sys::hv_return_t) -> Result<(), HvError> {
    if code == sys::HV_SUCCESS {
        Ok(())
    } else {
        Err(HvError { call, code })
    }
}

/// The process-wide virtual machine. Hypervisor.framework allows one per process.
pub struct Vm(());

impl Vm {
    pub fn new() -> Result<Self, HvError> {
        // SAFETY: a null config requests the default VM configuration.
        check("hv_vm_create", unsafe {
            sys::hv_vm_create(ptr::null_mut())
        })?;
        Ok(Self(()))
    }

    /// Maps host memory read/write/execute at guest-physical `ipa`.
    ///
    /// # Safety
    /// `host..host + size` must be a live mapping, aligned to the host page
    /// size, that outlives this `Vm`.
    pub unsafe fn map(&self, host: *mut u8, ipa: u64, size: usize) -> Result<(), HvError> {
        let flags = sys::HV_MEMORY_READ | sys::HV_MEMORY_WRITE | sys::HV_MEMORY_EXEC;
        // SAFETY: the caller guarantees the host range.
        check("hv_vm_map", unsafe {
            sys::hv_vm_map(host.cast(), ipa, size, flags)
        })
    }
}

impl Drop for Vm {
    fn drop(&mut self) {
        // SAFETY: the VM exists; all vCPUs borrow it and are already gone.
        unsafe { sys::hv_vm_destroy() };
    }
}

/// Identifies a vCPU from any thread, for [`VcpuHandle::kick`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VcpuHandle(sys::hv_vcpu_t);

impl VcpuHandle {
    /// Forces the vCPU out of [`Vcpu::run`] with [`Exit::Canceled`]. If it is
    /// not running, its next `run` returns at once instead, so a kick is
    /// never lost. The vCPU must still exist.
    pub fn kick(self) -> Result<(), HvError> {
        // SAFETY: one valid vCPU identifier is passed by pointer.
        check("hv_vcpus_exit", unsafe { sys::hv_vcpus_exit(&self.0, 1) })
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Exit {
    /// Another thread called [`VcpuHandle::kick`].
    Canceled,
    /// The guest took an exception to the hypervisor. `syndrome` is in
    /// `ESR_EL2` format; the addresses are meaningful for aborts only.
    Exception {
        syndrome: u64,
        virtual_address: u64,
        physical_address: u64,
    },
    /// The virtual timer fired; it stays masked until re-armed by the host.
    VtimerActivated,
    Unknown(u32),
}

/// A virtual CPU. Hypervisor.framework ties a vCPU to the thread that created
/// it, which the raw pointer field enforces by making this type `!Send`.
pub struct Vcpu<'vm> {
    id: sys::hv_vcpu_t,
    exit: *const sys::hv_vcpu_exit_t,
    _vm: &'vm Vm,
}

impl<'vm> Vcpu<'vm> {
    pub fn new(vm: &'vm Vm) -> Result<Self, HvError> {
        let mut id = 0;
        let mut exit = ptr::null();
        // SAFETY: both out-pointers are valid; a null config requests defaults.
        check("hv_vcpu_create", unsafe {
            sys::hv_vcpu_create(&mut id, &mut exit, ptr::null_mut())
        })?;
        Ok(Self { id, exit, _vm: vm })
    }

    /// Sets the level of the IRQ input for the next [`Vcpu::run`] only;
    /// Hypervisor.framework clears it every time the guest exits.
    pub fn set_pending_irq(&self, pending: bool) -> Result<(), HvError> {
        // SAFETY: the vCPU is live and owned by this thread.
        check("hv_vcpu_set_pending_interrupt", unsafe {
            sys::hv_vcpu_set_pending_interrupt(self.id, sys::HV_INTERRUPT_TYPE_IRQ, pending)
        })
    }

    pub fn handle(&self) -> VcpuHandle {
        VcpuHandle(self.id)
    }

    /// Reads `X<index>`; index 31 is XZR.
    pub fn gpr(&self, index: u8) -> Result<u64, HvError> {
        if index == 31 {
            Ok(0)
        } else {
            self.reg("hv_vcpu_get_reg", sys::HV_REG_X0 + u32::from(index))
        }
    }

    /// Writes `X<index>`; index 31 is XZR.
    pub fn set_gpr(&self, index: u8, value: u64) -> Result<(), HvError> {
        if index == 31 {
            Ok(())
        } else {
            self.set_reg(sys::HV_REG_X0 + u32::from(index), value)
        }
    }

    pub fn pc(&self) -> Result<u64, HvError> {
        self.reg("hv_vcpu_get_reg(PC)", sys::HV_REG_PC)
    }

    pub fn set_pc(&self, value: u64) -> Result<(), HvError> {
        self.set_reg(sys::HV_REG_PC, value)
    }

    pub fn pstate(&self) -> Result<u64, HvError> {
        self.reg("hv_vcpu_get_reg(CPSR)", sys::HV_REG_CPSR)
    }

    pub fn set_pstate(&self, value: u64) -> Result<(), HvError> {
        self.set_reg(sys::HV_REG_CPSR, value)
    }

    /// Takes the synchronous external abort a bus error gives the core at
    /// the instruction it is on, to EL1 in the guest's own vectors, as the
    /// CPU would: `ESR_EL1` (data abort from a lower or the same EL, fault
    /// status 0x10, `WnR` for a store), `FAR_EL1` = `address`, `ELR_EL1` =
    /// the instruction, `SPSR_EL1` = the current PSTATE; then EL1 on
    /// `SP_EL1` with all of `DAIF` masked at the synchronous vector (Arm
    /// ARM D1.10.2). On the console a store's error may arrive later as an
    /// SError instead, since stores to devices are posted; here both kinds
    /// of access abort at once, a simplification.
    pub fn take_external_abort(&self, address: u64, is_write: bool) -> Result<(), HvError> {
        const MODE_EL1H: u64 = 0b0101;
        const DFSC_EXTERNAL_ABORT: u64 = 0x10;
        let pstate = self.pstate()?;
        let (class, vector) = match pstate & 0xF {
            0 => (0x24, 0x400),
            _ if pstate & 1 != 0 => (0x25, 0x200),
            _ => (0x25, 0x000),
        };
        let write_not_read = if is_write { 1 << 6 } else { 0 };
        let syndrome = class << 26 | 1 << 25 | write_not_read | DFSC_EXTERNAL_ABORT;
        self.set_sys_reg(SysReg::ESR_EL1, syndrome)?;
        self.set_sys_reg(SysReg::FAR_EL1, address)?;
        self.set_sys_reg(SysReg::ELR_EL1, self.pc()?)?;
        self.set_sys_reg(SysReg::SPSR_EL1, pstate)?;
        self.set_pstate(0xF << 6 | MODE_EL1H)?;
        self.set_pc(self.sys_reg(SysReg::VBAR_EL1)?.wrapping_add(vector))
    }

    pub fn sys_reg(&self, register: SysReg) -> Result<u64, HvError> {
        let mut value = 0;
        // SAFETY: the vCPU is live and owned by this thread; `value` is valid.
        check("hv_vcpu_get_sys_reg", unsafe {
            sys::hv_vcpu_get_sys_reg(self.id, register.0, &mut value)
        })?;
        Ok(value)
    }

    pub fn set_sys_reg(&self, register: SysReg, value: u64) -> Result<(), HvError> {
        // SAFETY: the vCPU is live and owned by this thread.
        check("hv_vcpu_set_sys_reg", unsafe {
            sys::hv_vcpu_set_sys_reg(self.id, register.0, value)
        })
    }

    /// Everything of the core's architectural state that a running guest
    /// can have changed and Hypervisor.framework holds: the general and
    /// SIMD registers, PSTATE, FPCR/FPSR and [`SAVED_SYS_REGS`]. Not the
    /// debug registers, which [`debug::GuestDebug`] shadows, nor the
    /// physical timer, which the machine emulates.
    pub fn save_state(&self) -> Result<VcpuState, HvError> {
        let mut state = VcpuState::default();
        for (index, register) in (0u8..).zip(&mut state.x) {
            *register = self.gpr(index)?;
        }
        state.pc = self.pc()?;
        state.cpsr = self.pstate()?;
        state.fpcr = self.reg("hv_vcpu_get_reg(FPCR)", sys::HV_REG_FPCR)?;
        state.fpsr = self.reg("hv_vcpu_get_reg(FPSR)", sys::HV_REG_FPSR)?;
        for (index, register) in (0u32..).zip(&mut state.q) {
            let mut value = [0u8; 16];
            // SAFETY: the vCPU is live and owned by this thread; `value`
            // is a valid 16-byte buffer, the size of the register.
            check("hv_vcpu_get_simd_fp_reg", unsafe {
                sys::hv_vcpu_get_simd_fp_reg(self.id, index, value.as_mut_ptr().cast())
            })?;
            *register = u128::from_le_bytes(value);
        }
        state.sys = SAVED_SYS_REGS
            .iter()
            .map(|&register| Ok((register.0, self.sys_reg(register)?)))
            .collect::<Result<_, HvError>>()?;
        Ok(state)
    }

    /// Puts back what [`Vcpu::save_state`] returned.
    pub fn load_state(&self, state: &VcpuState) -> Result<(), HvError> {
        for (index, &register) in (0u8..).zip(&state.x) {
            self.set_gpr(index, register)?;
        }
        self.set_pc(state.pc)?;
        self.set_pstate(state.cpsr)?;
        self.set_reg(sys::HV_REG_FPCR, state.fpcr)?;
        self.set_reg(sys::HV_REG_FPSR, state.fpsr)?;
        for (index, &register) in (0u32..).zip(&state.q) {
            // SAFETY: the vCPU is live and owned by this thread.
            check("hv_vcpu_set_simd_fp_reg", unsafe {
                sys::hv_vcpu_set_simd_fp_reg_bytes(self.id, index, register.to_le_bytes())
            })?;
        }
        for &(register, value) in &state.sys {
            self.set_sys_reg(SysReg(register), value)?;
        }
        Ok(())
    }

    /// How long the guest's code has run on this vCPU since it was made,
    /// by the host kernel's count: without the time spent getting into
    /// and out of the guest, and without what is done for it in between.
    pub fn exec_time(&self) -> Result<Duration, HvError> {
        static NANOS_PER_TICK: OnceLock<(u64, u64)> = OnceLock::new();
        let &(numer, denom) = NANOS_PER_TICK.get_or_init(|| {
            let mut info = sys::mach_timebase_info_data_t { numer: 0, denom: 0 };
            // SAFETY: the out-pointer is valid. The call cannot fail for a
            // valid pointer; were it to, the ratio left is one to one.
            let code = unsafe { sys::mach_timebase_info(&mut info) };
            match (code, info.denom) {
                (0, 1..) => (u64::from(info.numer), u64::from(info.denom)),
                _ => (1, 1),
            }
        });
        let mut ticks = 0;
        // SAFETY: the vCPU is live and owned by this thread.
        check("hv_vcpu_get_exec_time", unsafe {
            sys::hv_vcpu_get_exec_time(self.id, &mut ticks)
        })?;
        let nanos = u128::from(ticks) * u128::from(numer) / u128::from(denom);
        Ok(Duration::from_nanos(nanos as u64))
    }

    /// Runs the guest until it exits to the hypervisor.
    pub fn run(&self) -> Result<Exit, HvError> {
        // SAFETY: the vCPU is live and owned by this thread.
        check("hv_vcpu_run", unsafe { sys::hv_vcpu_run(self.id) })?;
        // SAFETY: `exit` points at the framework-owned exit record, valid for
        // the vCPU's lifetime and only written during `hv_vcpu_run`.
        let exit = unsafe { &*self.exit };
        Ok(match exit.reason {
            sys::HV_EXIT_REASON_CANCELED => Exit::Canceled,
            sys::HV_EXIT_REASON_EXCEPTION => Exit::Exception {
                syndrome: exit.exception.syndrome,
                virtual_address: exit.exception.virtual_address,
                physical_address: exit.exception.physical_address,
            },
            sys::HV_EXIT_REASON_VTIMER_ACTIVATED => Exit::VtimerActivated,
            other => Exit::Unknown(other),
        })
    }

    fn reg(&self, call: &'static str, register: sys::hv_reg_t) -> Result<u64, HvError> {
        let mut value = 0;
        // SAFETY: the vCPU is live and owned by this thread; `value` is valid.
        check(call, unsafe {
            sys::hv_vcpu_get_reg(self.id, register, &mut value)
        })?;
        Ok(value)
    }

    fn set_reg(&self, register: sys::hv_reg_t, value: u64) -> Result<(), HvError> {
        // SAFETY: the vCPU is live and owned by this thread.
        check("hv_vcpu_set_reg", unsafe {
            sys::hv_vcpu_set_reg(self.id, register, value)
        })
    }
}

impl Drop for Vcpu<'_> {
    fn drop(&mut self) {
        // SAFETY: created by this thread (the type is `!Send`) and still live.
        unsafe { sys::hv_vcpu_destroy(self.id) };
    }
}
