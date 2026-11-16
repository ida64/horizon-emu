//! A core's state saved from one vCPU and loaded into another is the same
//! state, SIMD registers included (they go through a hand-written call).
//! Hypervisor.framework allows one VM per process, so this is its own test
//! binary.

use horizon_emu::{
    arm64::SysReg,
    hvf::{SAVED_SYS_REGS, Vcpu, VcpuState, Vm},
};

#[test]
fn state_round_trips_between_vcpus() {
    let vm = Vm::new().unwrap();
    let source = Vcpu::new(&vm).unwrap();
    let mut wanted = source.save_state().unwrap();
    for (index, register) in wanted.x.iter_mut().enumerate() {
        *register = 0x1111_0000_0000_0000 | index as u64;
    }
    for (index, register) in wanted.q.iter_mut().enumerate() {
        *register = u128::from_le_bytes(std::array::from_fn(|byte| (index * 16 + byte) as u8));
    }
    wanted.pc = 0x8000_1234;
    wanted.fpcr = 0x0300_0000;
    wanted.fpsr = 0x0800_0000;
    for (register, value) in &mut wanted.sys {
        match SysReg(*register) {
            SysReg::TTBR0_EL1 => *value = 0x8123_4000,
            SysReg::VBAR_EL1 => *value = 0xFFFF_FF80_0008_0000,
            // TPIDRRO_EL0
            register if register == SysReg::new(3, 3, 13, 0, 3) => *value = 0x55_6666_7000,
            _ => {}
        }
    }
    source.load_state(&wanted).unwrap();
    let saved: VcpuState = source.save_state().unwrap();
    assert_eq!(saved, wanted);
    assert_eq!(saved.sys.len(), SAVED_SYS_REGS.len());
    drop(source);

    let target = Vcpu::new(&vm).unwrap();
    target.load_state(&saved).unwrap();
    assert_eq!(target.save_state().unwrap(), wanted);
}
