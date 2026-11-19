//! Host breakpoints and single-step against real guest code under
//! Hypervisor.framework. Hypervisor.framework allows one VM per process, so
//! every scenario runs from the single test at the bottom, each on a fresh
//! vCPU.

use std::cell::Cell;

use horizon_emu::{
    arm64::{SysReg, SysRegAccess, exception_class},
    hvf::{
        Exit, Vcpu, Vm,
        debug::{BreakpointError, DebugExit, GuestDebug},
    },
    memory::GuestRam,
};

const RAM_BASE: u64 = 0x8000_0000;
const RAM_SIZE: usize = 0x40_0000;

/// EL1h with D, A, I and F masked, as at kernel entry.
const EL1H_MASKED: u64 = 0x3C5;
/// EL1h with only IRQs unmasked.
const EL1H_IRQS_ON: u64 = 0x345;
const EL0T: u64 = 0;
const PSTATE_SS: u64 = 1 << 21;
const NZCV: u64 = 0xF << 28;

const MDSCR_EL1: SysReg = SysReg::new(2, 0, 0, 2, 2);
const OSLSR_EL1: SysReg = SysReg::new(2, 0, 1, 1, 4);
const DBGBVR0_EL1: SysReg = SysReg::new(2, 0, 0, 0, 4);
const DBGBCR0_EL1: SysReg = SysReg::new(2, 0, 0, 0, 5);
const DBGWVR0_EL1: SysReg = SysReg::new(2, 0, 0, 0, 6);
const DBGWCR0_EL1: SysReg = SysReg::new(2, 0, 0, 0, 7);
const MAIR_EL1: SysReg = SysReg::new(3, 0, 10, 2, 0);

/// Offsets into the vector table.
const VECTOR_SYNC_EL1H: u64 = 0x200;
const VECTOR_IRQ_EL1H: u64 = 0x280;
const VECTOR_SYNC_EL0: u64 = 0x400;

/// A64 encodings of the few instructions the guest programs need.
mod asm {
    use horizon_emu::arm64::SysReg;

    pub const NOP: u32 = 0xD503_201F;
    pub const ERET: u32 = 0xD69F_03E0;
    pub const UDF: u32 = 0;

    pub fn hvc(immediate: u16) -> u32 {
        0xD400_0002 | u32::from(immediate) << 5
    }
    pub fn svc(immediate: u16) -> u32 {
        0xD400_0001 | u32::from(immediate) << 5
    }
    pub fn smc(immediate: u16) -> u32 {
        0xD400_0003 | u32::from(immediate) << 5
    }
    pub fn brk(immediate: u16) -> u32 {
        0xD420_0000 | u32::from(immediate) << 5
    }
    pub fn movz(rd: u32, immediate: u16) -> u32 {
        0xD280_0000 | u32::from(immediate) << 5 | rd
    }
    pub fn add(rd: u32, rn: u32, immediate: u32) -> u32 {
        0x9100_0000 | immediate << 10 | rn << 5 | rd
    }
    pub fn subs(rd: u32, rn: u32, immediate: u32) -> u32 {
        0xF100_0000 | immediate << 10 | rn << 5 | rd
    }
    /// `B.NE` by `instructions` relative to itself.
    pub fn b_ne(instructions: i32) -> u32 {
        0x5400_0001 | (instructions as u32 & 0x7_FFFF) << 5
    }
    /// `LDR Xt, [Xn]`.
    pub fn ldr(rt: u32, rn: u32) -> u32 {
        0xF940_0000 | rn << 5 | rt
    }
    pub fn msr(register: SysReg, rt: u32) -> u32 {
        0xD500_0000 | u32::from(register.0) << 5 | rt
    }
    pub fn mrs(rt: u32, register: SysReg) -> u32 {
        0xD520_0000 | u32::from(register.0) << 5 | rt
    }
}
use asm::{ERET, NOP, UDF, add, b_ne, brk, hvc, ldr, movz, mrs, msr, smc, subs, svc};

unsafe extern "C" {
    /// libSystem; the host-side counterpart of the guest's `IC IVAU`.
    fn sys_icache_invalidate(start: *mut std::ffi::c_void, len: usize);
}

/// Places guest code in RAM, each piece at a fresh address.
struct Loader<'a> {
    ram: &'a GuestRam,
    next: Cell<u64>,
}

impl Loader<'_> {
    fn write(&self, address: u64, code: &[u32]) {
        let bytes: Vec<u8> = code.iter().flat_map(|word| word.to_le_bytes()).collect();
        self.ram.write(address, &bytes).unwrap();
        // The instruction cache is shared with the guest and may hold what was
        // fetched (also speculatively) from these lines while they were zero.
        // SAFETY: the range was just bounds-checked by the write above.
        unsafe {
            sys_icache_invalidate(
                self.ram
                    .host_ptr()
                    .add((address - RAM_BASE) as usize)
                    .cast(),
                bytes.len(),
            )
        };
    }

    fn allocate(&self, size: u64, alignment: u64) -> u64 {
        let address = self.next.get().next_multiple_of(alignment);
        self.next.set(address + size);
        address
    }

    fn code(&self, code: &[u32]) -> u64 {
        let address = self.allocate(4 * code.len() as u64, 4);
        self.write(address, code);
        address
    }

    /// A vector table with `handlers` at the given offsets. Every other entry
    /// is `HVC #(0xF00 + entry)`, which no scenario expects.
    fn vectors(&self, handlers: &[(u64, &[u32])]) -> u64 {
        let table = self.allocate(0x800, 0x800);
        for entry in 0..16 {
            self.write(table + 0x80 * entry, &[hvc(0xF00 + entry as u16)]);
        }
        for &(offset, code) in handlers {
            self.write(table + offset, code);
        }
        table
    }
}

/// One guest exit, reduced to what the scenarios assert on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Event {
    Breakpoint(u64),
    Step(u64),
    /// A step that ended on the first instruction of a guest handler.
    StepIntoHandler(u64),
    Serviced,
    Hvc(u16),
    Smc(u16),
    /// A trapped debug register access, already served from the shadow.
    DebugRegister,
    Canceled,
    Other(u64),
}

/// A vCPU driven the way the machine's run loop drives one. A thread can
/// own one vCPU at a time, so a scenario drops its `Core` before making another.
struct Core<'a> {
    vcpu: Vcpu<'a>,
    debug: GuestDebug,
}

impl<'a> Core<'a> {
    fn new(vm: &'a Vm, pc: u64, vectors: u64) -> Self {
        let vcpu = Vcpu::new(vm).unwrap();
        vcpu.set_sys_reg(SysReg::SCTLR_EL1, 0x30D0_0800).unwrap();
        vcpu.set_sys_reg(SysReg::VBAR_EL1, vectors).unwrap();
        vcpu.set_pstate(EL1H_MASKED).unwrap();
        vcpu.set_pc(pc).unwrap();
        let debug = GuestDebug::attach(&vcpu).unwrap();
        Self { vcpu, debug }
    }

    /// Like [`Core::new`], but entered through an `ERET` to EL0 at `user`.
    fn new_el0(vm: &'a Vm, loader: &Loader, user: u64, vectors: u64) -> Self {
        let core = Self::new(vm, loader.code(&[ERET]), vectors);
        core.vcpu.set_sys_reg(SysReg::ELR_EL1, user).unwrap();
        core.vcpu.set_sys_reg(SysReg::SPSR_EL1, EL0T).unwrap();
        core
    }

    /// Enters the guest once.
    fn run(&mut self) -> Event {
        match self.debug.run(&self.vcpu).unwrap() {
            DebugExit::Breakpoint { pc } => {
                assert_eq!(pc, self.pc());
                Event::Breakpoint(pc)
            }
            DebugExit::Step {
                pc,
                exception_entry,
            } => {
                assert_eq!(pc, self.pc());
                if exception_entry {
                    Event::StepIntoHandler(pc)
                } else {
                    Event::Step(pc)
                }
            }
            DebugExit::Serviced => Event::Serviced,
            DebugExit::Guest(Exit::Canceled) => Event::Canceled,
            DebugExit::Guest(Exit::Exception { syndrome, .. }) => match exception_class(syndrome) {
                0x16 => Event::Hvc(syndrome as u16),
                0x17 => Event::Smc(syndrome as u16),
                0x18 => self.debug_register(SysRegAccess::from_syndrome(syndrome)),
                _ => Event::Other(syndrome),
            },
            DebugExit::Guest(other) => panic!("unexpected exit {other:?}"),
        }
    }

    /// What the machine's system register trap handler does for the
    /// registers this module owns.
    fn debug_register(&mut self, access: SysRegAccess) -> Event {
        if access.is_read {
            let value = self.debug.read_register(access.register).unwrap();
            self.vcpu.set_gpr(access.rt, value).unwrap();
        } else {
            let value = self.vcpu.gpr(access.rt).unwrap();
            self.debug.write_register(access.register, value).unwrap();
        }
        self.skip_instruction();
        Event::DebugRegister
    }

    /// Runs until something the host has to look at.
    fn resume(&mut self) -> Event {
        loop {
            match self.run() {
                Event::Serviced | Event::DebugRegister => {}
                event => return event,
            }
        }
    }

    fn step(&mut self) -> Event {
        self.debug.request_step();
        self.resume()
    }

    fn skip_instruction(&self) {
        self.vcpu.set_pc(self.pc() + 4).unwrap();
    }

    fn pc(&self) -> u64 {
        self.vcpu.pc().unwrap()
    }

    fn x(&self, index: u8) -> u64 {
        self.vcpu.gpr(index).unwrap()
    }

    fn pstate(&self) -> u64 {
        self.vcpu.pstate().unwrap()
    }

    fn reg(&self, register: SysReg) -> u64 {
        self.vcpu.sys_reg(register).unwrap()
    }

    /// The registers a guest exception handler finds on entry.
    fn exception_registers(&self) -> [u64; 5] {
        [
            self.pstate(),
            self.reg(SysReg::ELR_EL1),
            self.reg(SysReg::SPSR_EL1),
            self.reg(SysReg::ESR_EL1),
            self.reg(SysReg::FAR_EL1),
        ]
    }
}

/// `x0 = iterations; do { x1 += 1; x0 -= 1 } while x0 != 0; HVC #0`.
/// Returns the entry; the loop body starts one instruction later.
fn counting_loop(loader: &Loader, iterations: u16) -> u64 {
    loader.code(&[
        movz(0, iterations),
        add(1, 1, 1),
        subs(0, 0, 1),
        b_ne(-2),
        hvc(0),
    ])
}

fn breakpoint_fires_once_per_loop_iteration(vm: &Vm, loader: &Loader) {
    let entry = counting_loop(loader, 5);
    let body = entry + 4;
    let mut core = Core::new(vm, entry, loader.vectors(&[]));
    core.debug.add_breakpoint(body).unwrap();

    for iteration in 0..5 {
        assert_eq!(core.run(), Event::Breakpoint(body));
        // About to execute, not executed; debug exceptions masked by the
        // guest (PSTATE.D) do not matter; no step bit is left behind.
        assert_eq!(core.x(1), iteration);
        assert_eq!(core.x(0), 5 - iteration);
        assert_eq!(core.pstate() & !NZCV, EL1H_MASKED);
        // Resuming steps off the breakpoint, then runs freely.
        assert_eq!(core.run(), Event::Serviced);
        assert_eq!((core.pc(), core.x(1)), (body + 4, iteration + 1));
    }
    assert_eq!(core.run(), Event::Hvc(0));
    assert_eq!(core.x(1), 5);
}

fn adjacent_breakpoints_fire_in_order_and_can_be_removed(vm: &Vm, loader: &Loader) {
    let entry = counting_loop(loader, 3);
    let (first, second) = (entry + 4, entry + 8);
    let mut core = Core::new(vm, entry, loader.vectors(&[]));
    core.debug.add_breakpoint(second).unwrap();
    core.debug.add_breakpoint(first).unwrap();

    // Stepping off `first` lands on `second`, which was not reported yet.
    assert_eq!(core.resume(), Event::Breakpoint(first));
    assert_eq!(core.resume(), Event::Breakpoint(second));
    assert_eq!(core.resume(), Event::Breakpoint(first));
    assert!(core.debug.remove_breakpoint(first));
    assert_eq!(core.resume(), Event::Breakpoint(second));
    assert_eq!(core.resume(), Event::Breakpoint(second));
    assert_eq!(core.resume(), Event::Hvc(0));
    assert_eq!(core.x(1), 3);
}

fn every_breakpoint_register_works(vm: &Vm, loader: &Loader) {
    let mut core = Core::new(vm, 0, loader.vectors(&[]));
    let capacity = core.debug.breakpoint_capacity();
    assert!(capacity >= 4);
    let mut program = vec![add(1, 1, 1); capacity + 1];
    program.push(hvc(0));
    let entry = loader.code(&program);
    core.vcpu.set_pc(entry).unwrap();

    let address = |index: usize| entry + 4 * index as u64;
    for index in 0..capacity {
        core.debug.add_breakpoint(address(index)).unwrap();
    }
    assert_eq!(
        core.debug.add_breakpoint(address(capacity)),
        Err(BreakpointError::NoFreeRegister)
    );
    for index in 0..capacity {
        assert_eq!(core.resume(), Event::Breakpoint(address(index)));
        assert_eq!(core.x(1), index as u64);
    }
    assert_eq!(core.resume(), Event::Hvc(0));
}

fn single_step_visits_every_instruction(vm: &Vm, loader: &Loader) {
    let entry = counting_loop(loader, 2);
    let mut core = Core::new(vm, entry, loader.vectors(&[]));

    let expected = [4, 8, 12, 4, 8, 12, 16].map(|offset| entry + offset);
    for pc in expected {
        assert_eq!(core.step(), Event::Step(pc));
        assert_eq!(core.pstate() & PSTATE_SS, 0);
    }
    assert_eq!(core.x(1), 2);
    // HVC completes in the CPU but exits to the host before the step
    // exception; the step is reported once the host has serviced the call.
    core.debug.request_step();
    assert_eq!(core.run(), Event::Hvc(0));
    assert_eq!(core.pstate() & PSTATE_SS, 0);
    assert_eq!(core.run(), Event::Step(entry + 20));
}

fn step_off_a_breakpoint_then_continue(vm: &Vm, loader: &Loader) {
    let entry = counting_loop(loader, 2);
    let body = entry + 4;
    let mut core = Core::new(vm, entry, loader.vectors(&[]));
    core.debug.add_breakpoint(body).unwrap();

    assert_eq!(core.resume(), Event::Breakpoint(body));
    assert_eq!(core.step(), Event::Step(body + 4));
    assert_eq!(core.x(1), 1);
    // A breakpoint armed where a step stopped is not reported for that stop,
    // but is for the next arrival.
    core.debug.add_breakpoint(body + 8).unwrap();
    assert_eq!(core.step(), Event::Step(body + 8));
    assert_eq!(core.resume(), Event::Breakpoint(body));
    assert_eq!(core.x(1), 1);
    assert_eq!(core.resume(), Event::Breakpoint(body + 8));
    assert_eq!(core.resume(), Event::Hvc(0));
    assert_eq!(core.x(1), 2);
}

fn a_step_request_survives_a_breakpoint_on_its_instruction(vm: &Vm, loader: &Loader) {
    let entry = counting_loop(loader, 1);
    let mut core = Core::new(vm, entry, loader.vectors(&[]));
    core.debug.add_breakpoint(entry).unwrap();

    // Nothing was reported at the entry yet, so the breakpoint comes first.
    assert_eq!(core.step(), Event::Breakpoint(entry));
    assert_eq!(core.pstate() & PSTATE_SS, 0);
    assert_eq!(core.resume(), Event::Step(entry + 4));
    assert_eq!(core.x(0), 1);
}

fn a_step_resumes_after_the_host_emulates_or_interrupts_it(vm: &Vm, loader: &Loader) {
    let entry = loader.code(&[smc(1), add(1, 1, 1), add(1, 1, 1), hvc(0)]);
    let mut core = Core::new(vm, entry, loader.vectors(&[]));

    // The SMC traps instead of executing; emulating it is the step.
    core.debug.request_step();
    assert_eq!(core.run(), Event::Smc(1));
    assert_eq!((core.pc(), core.pstate()), (entry, EL1H_MASKED));
    core.skip_instruction();
    assert_eq!(core.run(), Event::Step(entry + 4));
    assert_eq!(core.x(1), 0);

    // Kicked out before the instruction ran: the step is still owed, once.
    core.debug.request_step();
    core.vcpu.handle().kick().unwrap();
    assert_eq!(core.run(), Event::Canceled);
    assert_eq!((core.pc(), core.pstate()), (entry + 4, EL1H_MASKED));
    assert_eq!(core.run(), Event::Step(entry + 8));
    assert_eq!(core.x(1), 1);
    assert_eq!(core.resume(), Event::Hvc(0));
}

fn guest_debug_registers_are_shadowed(vm: &Vm, loader: &Loader) {
    // The kernel's boot sequence in miniature (image 0x800c60a0), then a
    // guest breakpoint and watchpoint that would fire if they were real.
    let entry = loader.code(&[
        msr(MDSCR_EL1, 31),
        msr(SysReg::OSLAR_EL1, 31),
        msr(DBGBCR0_EL1, 31),
        msr(DBGBVR0_EL1, 31),
        msr(MDSCR_EL1, 9),
        msr(DBGBVR0_EL1, 10),
        msr(DBGBCR0_EL1, 11),
        msr(DBGWVR0_EL1, 13),
        msr(DBGWCR0_EL1, 12),
        mrs(1, MDSCR_EL1),
        mrs(2, DBGBVR0_EL1),
        mrs(3, DBGBCR0_EL1),
        mrs(4, OSLSR_EL1),
        mrs(5, DBGWCR0_EL1),
        ldr(6, 13),
        NOP,
        add(7, 7, 1),
        hvc(0),
    ]);
    let guest_breakpoint = entry + 4 * 15;
    let host_breakpoint = entry + 4 * 16;
    let mut core = Core::new(vm, entry, loader.vectors(&[]));
    core.vcpu.set_gpr(9, 0x9000).unwrap();
    core.vcpu.set_gpr(10, guest_breakpoint).unwrap();
    // Enabled, EL1 and EL0, all bytes.
    core.vcpu.set_gpr(11, 0x1E7).unwrap();
    // Enabled, loads and stores, EL1 and EL0, all eight bytes.
    core.vcpu.set_gpr(12, 0x1FFF).unwrap();
    // A doubleword the guest then loads (aligned: with the MMU off, data
    // accesses are to Device memory).
    let watched = loader.allocate(8, 8);
    core.vcpu.set_gpr(13, watched).unwrap();
    core.debug.add_breakpoint(host_breakpoint).unwrap();

    let mut accesses = 0;
    let event = loop {
        match core.run() {
            Event::DebugRegister => accesses += 1,
            event => break event,
        }
    };
    assert_eq!(event, Event::Breakpoint(host_breakpoint));
    assert_eq!(accesses, 14);
    assert_eq!(
        [1, 2, 3, 4, 5].map(|index| core.x(index)),
        [0x9000, guest_breakpoint, 0x1E7, 0b1000, 0x1FFF]
    );
    assert_eq!(core.x(7), 0);
    // The real registers hold the host's breakpoint, plus the one MDSCR bit
    // of the guest's that is not about debug exceptions (TDCC).
    assert_eq!(core.reg(DBGBVR0_EL1), host_breakpoint);
    assert_eq!(core.reg(DBGBCR0_EL1), 0x1E7);
    assert_eq!(core.reg(DBGWCR0_EL1), 0);
    assert_eq!(core.reg(MDSCR_EL1), 0x9000);
    assert_eq!(core.resume(), Event::Hvc(0));
}

fn breakpoint_and_steps_at_el0_across_an_svc(vm: &Vm, loader: &Loader) {
    let user = loader.code(&[add(1, 1, 1), add(1, 1, 1), svc(3), add(1, 1, 1), NOP]);
    let vectors = loader.vectors(&[(VECTOR_SYNC_EL0, &[add(2, 2, 1), ERET])]);
    let handler = vectors + VECTOR_SYNC_EL0;
    let mut core = Core::new_el0(vm, loader, user, vectors);
    core.debug.add_breakpoint(user + 4).unwrap();
    core.debug.add_breakpoint(user + 16).unwrap();

    assert_eq!(core.resume(), Event::Breakpoint(user + 4));
    assert_eq!((core.pstate(), core.x(1)), (EL0T, 1));
    assert_eq!(core.step(), Event::Step(user + 8));
    // The SVC executes and the step ends on the handler's first instruction.
    assert_eq!(core.step(), Event::StepIntoHandler(handler));
    assert_eq!(core.pstate() & 0xF, EL1H_MASKED & 0xF);
    assert_eq!(core.reg(SysReg::ELR_EL1), user + 12);
    assert_eq!(core.reg(SysReg::SPSR_EL1), EL0T);
    assert_eq!(core.reg(SysReg::ESR_EL1), 0x15 << 26 | 1 << 25 | 3);
    assert_eq!(core.step(), Event::Step(handler + 4));
    assert_eq!(core.step(), Event::Step(user + 12));
    assert_eq!((core.pstate(), core.x(1), core.x(2)), (EL0T, 2, 1));
    assert_eq!(core.resume(), Event::Breakpoint(user + 16));
    assert_eq!(core.x(1), 3);
}

fn a_step_into_a_fault_leaves_no_trace_in_the_guest(vm: &Vm, loader: &Loader) {
    let entry = loader.code(&[NOP, UDF, add(1, 1, 1), hvc(0)]);
    let fault = entry + 4;
    // Skip the faulting instruction and return.
    let vectors = loader.vectors(&[(
        VECTOR_SYNC_EL1H,
        &[
            mrs(9, SysReg::ELR_EL1),
            add(9, 9, 4),
            msr(SysReg::ELR_EL1, 9),
            ERET,
        ],
    )]);
    let handler = vectors + VECTOR_SYNC_EL1H;
    let mut core = Core::new(vm, entry, vectors);

    assert_eq!(core.step(), Event::Step(fault));
    assert_eq!(core.step(), Event::StepIntoHandler(handler));
    // The CPU saved the host's PSTATE.SS into SPSR_EL1; the guest must find
    // the PSTATE it had.
    assert_eq!(core.reg(SysReg::ELR_EL1), fault);
    assert_eq!(core.reg(SysReg::SPSR_EL1), EL1H_MASKED);
    for offset in [4, 8, 12] {
        assert_eq!(core.step(), Event::Step(handler + offset));
    }
    // Stepping the ERET stops on the instruction returned to, not after it.
    assert_eq!(core.step(), Event::Step(fault + 4));
    assert_eq!(core.x(1), 0);
    assert_eq!(core.resume(), Event::Hvc(0));
}

fn guest_brk_reaches_the_guest_unchanged(vm: &Vm, loader: &Loader) {
    const FAR_MARKER: u64 = 0x1234_5678;
    const ESR_MARKER: u64 = 0x9600_0045;
    let brk_syndrome = 0x3C << 26 | 1 << 25 | 0x42;

    for (pstate, vector) in [
        (EL1H_IRQS_ON | 0x9 << 28, VECTOR_SYNC_EL1H),
        (EL0T | 0x6 << 28, VECTOR_SYNC_EL0),
    ] {
        let program = [add(1, 1, 1), brk(0x42), add(1, 1, 1)];
        let handlers: [(u64, &[u32]); 2] = [
            (vector, &[hvc(0x77)]),
            (VECTOR_IRQ_EL1H, &[hvc(0x66), ERET]),
        ];
        let start = |core: &Core| {
            core.vcpu.set_pstate(pstate).unwrap();
            core.vcpu.set_sys_reg(SysReg::FAR_EL1, FAR_MARKER).unwrap();
            core.vcpu.set_sys_reg(SysReg::ESR_EL1, ESR_MARKER).unwrap();
        };

        // Nothing armed: the CPU delivers the BRK without the host seeing it.
        let entry = loader.code(&program);
        let mut core = Core::new(vm, entry, loader.vectors(&handlers));
        start(&core);
        assert_eq!(core.run(), Event::Hvc(0x77));
        let native_pstate = core.pstate();
        let expected = |entry: u64| [native_pstate, entry + 4, pstate, brk_syndrome, FAR_MARKER];
        assert_eq!(core.exception_registers(), expected(entry));

        drop(core);

        // A breakpoint elsewhere routes debug exceptions to the host: the
        // BRK exits, and is delivered on the following entry.
        let entry = loader.code(&program);
        let vectors = loader.vectors(&handlers);
        let mut core = Core::new(vm, entry, vectors);
        start(&core);
        core.debug.add_breakpoint(entry + 8).unwrap();
        assert_eq!(core.run(), Event::Serviced);
        assert_eq!((core.pc(), core.pstate()), (entry + 4, pstate));
        assert_eq!(core.run(), Event::Serviced);
        assert_eq!(core.pc(), vectors + vector);
        assert_eq!(core.run(), Event::Hvc(0x77));
        assert_eq!(core.exception_registers(), expected(entry));

        drop(core);

        // A step over the BRK ends in the handler like any other exception.
        let entry = loader.code(&program);
        let vectors = loader.vectors(&handlers);
        let mut core = Core::new(vm, entry, vectors);
        start(&core);
        assert_eq!(core.step(), Event::Step(entry + 4));
        assert_eq!(core.step(), Event::StepIntoHandler(vectors + vector));
        assert_eq!(core.exception_registers(), expected(entry));

        drop(core);

        if pstate & 0xF != EL1H_IRQS_ON & 0xF {
            continue;
        }
        // An interrupt that wins over the delivery sees the interrupted
        // context as it was; the BRK is delivered after its handler returns.
        let entry = loader.code(&program);
        let mut core = Core::new(vm, entry, loader.vectors(&handlers));
        start(&core);
        core.debug.add_breakpoint(entry + 8).unwrap();
        assert_eq!(core.run(), Event::Serviced);
        core.vcpu.set_pending_irq(true).unwrap();
        assert_eq!(core.run(), Event::Serviced);
        assert_eq!(core.run(), Event::Hvc(0x66));
        assert_eq!(
            core.exception_registers()[1..],
            [entry + 4, pstate, ESR_MARKER, FAR_MARKER]
        );
        assert_eq!(core.resume(), Event::Hvc(0x77));
        assert_eq!(core.exception_registers(), expected(entry));
    }
}

/// A load from an address with nothing behind it (the security engine's,
/// which the non-secure world may not reach) exits to the host, which
/// hands the guest the external abort the bus would: the guest's EL0
/// synchronous vector runs with the registers a CPU sets.
fn an_external_abort_reaches_the_guest_vectors(vm: &Vm, loader: &Loader) {
    const SECURE: u64 = 0x7001_2000;
    let user = loader.code(&[ldr(1, 2), NOP]);
    let vectors = loader.vectors(&[(VECTOR_SYNC_EL0, &[hvc(7)])]);
    let mut core = Core::new_el0(vm, loader, user, vectors);
    core.vcpu.set_gpr(2, SECURE).unwrap();

    let syndrome = match core.resume() {
        Event::Other(syndrome) => syndrome,
        other => panic!("expected the load to exit, got {other:?}"),
    };
    assert_eq!(exception_class(syndrome), exception_class::DATA_ABORT_LOWER);
    assert_eq!(core.pc(), user);
    core.vcpu.take_external_abort(SECURE, false).unwrap();
    assert_eq!(core.resume(), Event::Hvc(7));
    let [pstate, elr, spsr, esr, far] = core.exception_registers();
    assert_eq!(pstate & 0x3CF, EL1H_MASKED);
    assert_eq!((elr, spsr, far), (user, EL0T, SECURE));
    assert_eq!(esr, 0x24 << 26 | 1 << 25 | 0x10);
}

fn an_interrupt_can_preempt_the_step_off_a_breakpoint(vm: &Vm, loader: &Loader) {
    let handlers: [(u64, &[u32]); 1] = [(VECTOR_IRQ_EL1H, &[hvc(0x66), ERET])];

    // Interrupt asserted while resuming: the instruction under the
    // breakpoint is preempted and the breakpoint is reported again when the
    // handler returns to it.
    let entry = counting_loop(loader, 1);
    let body = entry + 4;
    let mut core = Core::new(vm, entry, loader.vectors(&handlers));
    core.vcpu.set_pstate(EL1H_IRQS_ON).unwrap();
    core.debug.add_breakpoint(body).unwrap();
    assert_eq!(core.resume(), Event::Breakpoint(body));
    assert!(core.debug.wants_interrupts_deferred(&core.vcpu).unwrap());
    core.vcpu.set_pending_irq(true).unwrap();
    assert_eq!(core.run(), Event::Serviced);
    assert_eq!(core.run(), Event::Hvc(0x66));
    assert_eq!(core.reg(SysReg::ELR_EL1), body);
    assert_eq!(core.reg(SysReg::SPSR_EL1), EL1H_IRQS_ON);
    assert_eq!(core.resume(), Event::Breakpoint(body));
    assert_eq!(core.x(1), 0);

    // Interrupt held back for the one entry that asks for it: exactly one
    // report per executed instruction.
    assert!(core.debug.wants_interrupts_deferred(&core.vcpu).unwrap());
    assert_eq!(core.run(), Event::Serviced);
    assert_eq!(core.x(1), 1);
    assert!(!core.debug.wants_interrupts_deferred(&core.vcpu).unwrap());
    core.vcpu.set_pending_irq(true).unwrap();
    assert_eq!(core.run(), Event::Hvc(0x66));
    assert_eq!(core.reg(SysReg::ELR_EL1), body + 4);
    assert_eq!(core.resume(), Event::Hvc(0));
}

/// Supervisor call tracing passes its breakpoints on the kernel's vector
/// (a `CLREX`) and return (`ERET`) by carrying the instructions out on the
/// host instead of stepping the core off them: the core goes through the
/// same breakpoints and ends in the same state either way.
fn passing_svc_breakpoints_on_the_host_ends_as_stepping(vm: &Vm, loader: &Loader) {
    const CLREX: u32 = 0xD503_3F5F;
    // Three calls, then a load from outside the VM's memory, an exit.
    let user = loader.code(&[
        movz(0, 3),
        add(1, 1, 1),
        svc(5),
        subs(0, 0, 1),
        b_ne(-3),
        movz(4, 0x1000),
        ldr(3, 4),
    ]);
    let vectors = loader.vectors(&[(VECTOR_SYNC_EL0, &[CLREX, add(2, 2, 1), ERET])]);
    let (vector, returns) = (vectors + VECTOR_SYNC_EL0, vectors + VECTOR_SYNC_EL0 + 8);
    let run = |passed: bool| {
        let mut core = Core::new_el0(vm, loader, user, vectors);
        core.debug.add_breakpoint(vector).unwrap();
        core.debug.add_breakpoint(returns).unwrap();
        let mut reached = Vec::new();
        let end = loop {
            match core.resume() {
                Event::Breakpoint(pc) => {
                    reached.push(pc);
                    if passed && pc == vector {
                        horizon_emu::machine::pass_clrex(&core.vcpu, pc).unwrap();
                    }
                    if passed && pc == returns {
                        assert!(horizon_emu::machine::pass_return_to_el0(&core.vcpu).unwrap());
                    }
                }
                event => break event,
            }
        };
        (reached, end, core.vcpu.save_state().unwrap())
    };
    let stepped = run(false);
    assert_eq!(stepped.0, [vector, returns].repeat(3));
    assert_eq!((stepped.2.x[1], stepped.2.x[2]), (3, 3));
    assert_eq!(run(true), stepped);

    // A return to EL1 is left to the core.
    let core = Core::new(vm, loader.code(&[ERET]), vectors);
    core.vcpu
        .set_sys_reg(SysReg::SPSR_EL1, EL1H_MASKED)
        .unwrap();
    core.vcpu.set_sys_reg(SysReg::ELR_EL1, user).unwrap();
    let before = core.vcpu.save_state().unwrap();
    assert!(!horizon_emu::machine::pass_return_to_el0(&core.vcpu).unwrap());
    assert_eq!(core.vcpu.save_state().unwrap(), before);
}

fn breakpoints_match_virtual_addresses_with_the_mmu_on(vm: &Vm, loader: &Loader) {
    /// The kernel's regime: 39-bit halves, 4 KiB granule.
    const TCR: u64 = 0x11_B519_3519;
    /// Block, access flag, inner shareable, EL1 read/write, attribute 0.
    const BLOCK: u64 = 1 << 10 | 3 << 8 | 1;
    const HIGH_HALF: u64 = 0xFFFF_FF80_0000_0000;

    // One 1 GiB block per half, both onto DRAM: identity in the low half,
    // and DRAM again at the bottom of the high half.
    let low_table = loader.allocate(0x1000, 0x1000);
    let high_table = loader.allocate(0x1000, 0x1000);
    let block = (RAM_BASE | BLOCK).to_le_bytes();
    loader
        .ram
        .write(low_table + 8 * (RAM_BASE >> 30), &block)
        .unwrap();
    loader.ram.write(high_table, &block).unwrap();

    let entry = counting_loop(loader, 2);
    let high = |address: u64| HIGH_HALF + (address - RAM_BASE);
    let mut core = Core::new(vm, entry, high(loader.vectors(&[])));
    core.vcpu.set_sys_reg(MAIR_EL1, 0xFF).unwrap();
    core.vcpu.set_sys_reg(SysReg::TCR_EL1, TCR).unwrap();
    core.vcpu.set_sys_reg(SysReg::TTBR0_EL1, low_table).unwrap();
    core.vcpu
        .set_sys_reg(SysReg::TTBR1_EL1, high_table)
        .unwrap();
    core.vcpu
        .set_sys_reg(SysReg::SCTLR_EL1, 0x30D0_0800 | 0x1005)
        .unwrap();
    core.debug.add_breakpoint(high(entry + 4)).unwrap();

    // Same instructions through the identity mapping: no match.
    assert_eq!(core.resume(), Event::Hvc(0));
    assert_eq!(core.x(1), 2);
    // Through the high mapping, as the kernel runs.
    core.vcpu.set_pc(high(entry)).unwrap();
    assert_eq!(core.resume(), Event::Breakpoint(high(entry + 4)));
    assert_eq!(core.step(), Event::Step(high(entry + 8)));
    assert_eq!(core.resume(), Event::Breakpoint(high(entry + 4)));
    assert_eq!(core.resume(), Event::Hvc(0));
    assert_eq!(core.x(1), 4);
}

/// An exit to the host and the entry back, an exception return, clear the
/// guest's exclusive monitor: a store-exclusive after one fails as after a
/// `CLREX`. Supervisor call tracing relies on that when it passes the
/// kernel's vector `CLREX` without the core executing it.
fn an_exit_clears_the_exclusive_monitor(vm: &Vm, loader: &Loader) {
    /// Block, access flag, inner shareable, EL1 read/write, attribute 0.
    const BLOCK: u64 = 1 << 10 | 3 << 8 | 1;
    const TCR: u64 = 0x11_B519_3519;
    let ldxr = |rt: u32, rn: u32| 0xC85F_7C00 | rn << 5 | rt;
    let stxr = |rs: u32, rt: u32, rn: u32| 0xC800_7C00 | rs << 16 | rn << 5 | rt;
    let table = loader.allocate(0x1000, 0x1000);
    loader
        .ram
        .write(
            table + 8 * (RAM_BASE >> 30),
            &(RAM_BASE | BLOCK).to_le_bytes(),
        )
        .unwrap();
    let word = loader.allocate(8, 64);
    let entry = loader.code(&[
        // A pair with nothing in between succeeds; one with an exit in
        // between fails.
        ldxr(5, 6),
        stxr(8, 5, 6),
        ldxr(5, 6),
        hvc(1),
        stxr(7, 5, 6),
        hvc(0),
    ]);
    let mut core = Core::new(vm, entry, loader.vectors(&[]));
    core.vcpu.set_sys_reg(MAIR_EL1, 0xFF).unwrap();
    core.vcpu.set_sys_reg(SysReg::TCR_EL1, TCR).unwrap();
    core.vcpu.set_sys_reg(SysReg::TTBR0_EL1, table).unwrap();
    core.vcpu
        .set_sys_reg(SysReg::SCTLR_EL1, 0x30D0_0800 | 0x1005)
        .unwrap();
    core.vcpu.set_gpr(6, word).unwrap();
    core.vcpu.set_gpr(7, 9).unwrap();
    core.vcpu.set_gpr(8, 9).unwrap();
    assert_eq!(core.resume(), Event::Hvc(1));
    assert_eq!(core.resume(), Event::Hvc(0));
    assert_eq!((core.x(8), core.x(7)), (0, 1));
}

#[test]
fn guest_debug() {
    let vm = Vm::new().unwrap();
    let ram = GuestRam::new(RAM_BASE, RAM_SIZE).unwrap();
    // SAFETY: `ram` is a page-aligned mapping that outlives `vm`'s use of it;
    // every vCPU is gone before either is dropped.
    unsafe { vm.map(ram.host_ptr(), ram.base(), ram.size()).unwrap() };
    let loader = Loader {
        ram: &ram,
        next: Cell::new(RAM_BASE),
    };

    breakpoint_fires_once_per_loop_iteration(&vm, &loader);
    adjacent_breakpoints_fire_in_order_and_can_be_removed(&vm, &loader);
    every_breakpoint_register_works(&vm, &loader);
    single_step_visits_every_instruction(&vm, &loader);
    step_off_a_breakpoint_then_continue(&vm, &loader);
    a_step_request_survives_a_breakpoint_on_its_instruction(&vm, &loader);
    a_step_resumes_after_the_host_emulates_or_interrupts_it(&vm, &loader);
    guest_debug_registers_are_shadowed(&vm, &loader);
    breakpoint_and_steps_at_el0_across_an_svc(&vm, &loader);
    a_step_into_a_fault_leaves_no_trace_in_the_guest(&vm, &loader);
    guest_brk_reaches_the_guest_unchanged(&vm, &loader);
    an_external_abort_reaches_the_guest_vectors(&vm, &loader);
    an_interrupt_can_preempt_the_step_off_a_breakpoint(&vm, &loader);
    passing_svc_breakpoints_on_the_host_ends_as_stepping(&vm, &loader);
    breakpoints_match_virtual_addresses_with_the_mmu_on(&vm, &loader);
    an_exit_clears_the_exclusive_monitor(&vm, &loader);
    drop(vm);
}
