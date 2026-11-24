//! Raw Hypervisor.framework declarations (arm64), limited to what is used.
//! Transcribed from the macOS SDK headers `hv_vm.h`, `hv_vcpu.h`,
//! `hv_vcpu_types.h` and `hv_error.h`.

#![allow(non_camel_case_types)]

use std::ffi::c_void;

pub type hv_return_t = i32;
pub type hv_vcpu_t = u64;
pub type hv_ipa_t = u64;
pub type hv_memory_flags_t = u64;
pub type hv_reg_t = u32;
pub type hv_sys_reg_t = u16;

pub const HV_SUCCESS: hv_return_t = 0;
pub const HV_DENIED: hv_return_t = 0xfae9_4007_u32 as i32;

pub const HV_MEMORY_READ: hv_memory_flags_t = 1 << 0;
pub const HV_MEMORY_WRITE: hv_memory_flags_t = 1 << 1;
pub const HV_MEMORY_EXEC: hv_memory_flags_t = 1 << 2;

pub const HV_REG_X0: hv_reg_t = 0;
pub const HV_REG_PC: hv_reg_t = 31;
pub const HV_REG_FPCR: hv_reg_t = 32;
pub const HV_REG_FPSR: hv_reg_t = 33;
pub const HV_REG_CPSR: hv_reg_t = 34;

pub const HV_EXIT_REASON_CANCELED: u32 = 0;
pub const HV_EXIT_REASON_EXCEPTION: u32 = 1;
pub const HV_EXIT_REASON_VTIMER_ACTIVATED: u32 = 2;

pub const HV_INTERRUPT_TYPE_IRQ: u32 = 0;

pub const HV_FEATURE_REG_ID_AA64DFR0_EL1: u32 = 0;

#[repr(C)]
pub struct hv_vcpu_exit_exception_t {
    pub syndrome: u64,
    pub virtual_address: u64,
    pub physical_address: u64,
}

#[repr(C)]
pub struct hv_vcpu_exit_t {
    pub reason: u32,
    pub exception: hv_vcpu_exit_exception_t,
}

#[link(name = "Hypervisor", kind = "framework")]
unsafe extern "C" {
    pub fn hv_vm_create(config: *mut c_void) -> hv_return_t;
    pub fn hv_vm_destroy() -> hv_return_t;
    pub fn hv_vm_map(
        addr: *mut c_void,
        ipa: hv_ipa_t,
        size: usize,
        flags: hv_memory_flags_t,
    ) -> hv_return_t;
    pub fn hv_vm_protect(ipa: hv_ipa_t, size: usize, flags: hv_memory_flags_t) -> hv_return_t;

    pub fn hv_vcpu_create(
        vcpu: *mut hv_vcpu_t,
        exit: *mut *const hv_vcpu_exit_t,
        config: *mut c_void,
    ) -> hv_return_t;
    pub fn hv_vcpu_destroy(vcpu: hv_vcpu_t) -> hv_return_t;
    pub fn hv_vcpu_get_reg(vcpu: hv_vcpu_t, reg: hv_reg_t, value: *mut u64) -> hv_return_t;
    pub fn hv_vcpu_set_reg(vcpu: hv_vcpu_t, reg: hv_reg_t, value: u64) -> hv_return_t;
    pub fn hv_vcpu_get_sys_reg(vcpu: hv_vcpu_t, reg: hv_sys_reg_t, value: *mut u64) -> hv_return_t;
    pub fn hv_vcpu_set_sys_reg(vcpu: hv_vcpu_t, reg: hv_sys_reg_t, value: u64) -> hv_return_t;
    pub fn hv_vcpu_run(vcpu: hv_vcpu_t) -> hv_return_t;
    /// Cumulative, in units of `mach_absolute_time`.
    pub fn hv_vcpu_get_exec_time(vcpu: hv_vcpu_t, time: *mut u64) -> hv_return_t;
    pub fn hv_vcpus_exit(vcpus: *const hv_vcpu_t, vcpu_count: u32) -> hv_return_t;
    pub fn hv_vcpu_set_pending_interrupt(
        vcpu: hv_vcpu_t,
        interrupt_type: u32,
        pending: bool,
    ) -> hv_return_t;
    pub fn hv_vcpu_set_trap_debug_exceptions(vcpu: hv_vcpu_t, value: bool) -> hv_return_t;
    pub fn hv_vcpu_set_trap_debug_reg_accesses(vcpu: hv_vcpu_t, value: bool) -> hv_return_t;
    /// `value` points at the 16 bytes of `Q<reg>`.
    pub fn hv_vcpu_get_simd_fp_reg(vcpu: hv_vcpu_t, reg: u32, value: *mut c_void) -> hv_return_t;
    /// Takes the register's 16 bytes by value in a vector register, which
    /// stable Rust cannot pass over FFI: called through
    /// [`hv_vcpu_set_simd_fp_reg_bytes`]. Declared without its parameters
    /// only so that its address can be named.
    fn hv_vcpu_set_simd_fp_reg();

    /// Returns a retained `hv_vcpu_config_t`, to be released with [`os_release`].
    pub fn hv_vcpu_config_create() -> *mut c_void;
    pub fn hv_vcpu_config_get_feature_reg(
        config: *mut c_void,
        feature_reg: u32,
        value: *mut u64,
    ) -> hv_return_t;
    /// libSystem's release for `os_object` types such as `hv_vcpu_config_t`.
    pub fn os_release(object: *mut c_void);
}

/// `mach_timebase_info_data_t` of `mach/mach_time.h`: one unit of
/// `mach_absolute_time` is `numer / denom` nanoseconds.
#[repr(C)]
pub struct mach_timebase_info_data_t {
    pub numer: u32,
    pub denom: u32,
}

/// `hv_vcpu_set_simd_fp_reg(vcpu, reg, value)` with `value` the register's
/// 16 bytes in memory order, passed in `V0` as the C ABI passes a
/// `simd_uchar16`.
///
/// # Safety
/// As for any Hypervisor.framework call: `vcpu` is live and owned by the
/// calling thread.
pub unsafe fn hv_vcpu_set_simd_fp_reg_bytes(
    vcpu: hv_vcpu_t,
    reg: u32,
    value: [u8; 16],
) -> hv_return_t {
    let low = u64::from_le_bytes(value[..8].try_into().unwrap());
    let high = u64::from_le_bytes(value[8..].try_into().unwrap());
    let result: u64;
    // SAFETY: a call that follows the C ABI: arguments in X0, W1 and V0,
    // the result in W0; every register the ABI lets the callee change is
    // declared clobbered.
    unsafe {
        std::arch::asm!(
            "fmov d0, {low}",
            "mov v0.d[1], {high}",
            "bl {function}",
            low = in(reg) low,
            high = in(reg) high,
            function = sym hv_vcpu_set_simd_fp_reg,
            inout("x0") vcpu => result,
            in("x1") u64::from(reg),
            clobber_abi("C"),
        );
    }
    result as hv_return_t
}

unsafe extern "C" {
    /// libSystem; returns `KERN_SUCCESS` (0).
    pub fn mach_timebase_info(info: *mut mach_timebase_info_data_t) -> i32;
}
