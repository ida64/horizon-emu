//! CPU backend on Apple's Hypervisor.framework: the guest kernel runs natively
//! at EL1 on the host core, and everything that would reach EL2/EL3 or leave
//! mapped memory comes back as an [`Exit`].

pub mod debug;
mod sys;

use std::sync::OnceLock;
use std::time::Duration;
use std::{fmt, ptr};

use crate::arm64::SysReg;

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
