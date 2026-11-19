//! Tests of the BPMP core. Encodings were produced with an ARMv4T
//! assembler (`clang --target=armv4t-none-eabi`) from the source in each
//! comment; the expected values of the data-processing tables were checked
//! against a second ARM implementation.

use std::{ops::Range, sync::Mutex};

use super::*;

const CODE: u32 = 0x1000;
const DATA: u32 = 0x8000;
const STACK: u32 = 0xC000;
/// Addresses the bus answers with an error.
const ABORTS: Range<u32> = 0xF000..0xF100;
/// Addresses the bus does not model.
const UNMODELLED: Range<u32> = 0xE000..0xE100;
const RAM_SIZE: usize = 0x1_0000;

/// 64 KiB of RAM at 0, with an aborting and an unmodelled hole.
struct Ram {
    bytes: Vec<u8>,
    /// Handler addresses by vector, as the T210's EVP registers give them.
    vectors: Option<[u32; 8]>,
    /// A write here asks the run to pause.
    halt: Option<u32>,
    halted: bool,
}

impl Ram {
    fn new() -> Self {
        Self {
            bytes: vec![0; RAM_SIZE],
            vectors: None,
            halt: None,
            halted: false,
        }
    }

    fn word(&self, address: u32) -> u32 {
        let at = address as usize;
        u32::from_le_bytes(self.bytes[at..at + 4].try_into().unwrap())
    }

    fn set_word(&mut self, address: u32, value: u32) {
        let at = address as usize;
        self.bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn half(&self, address: u32) -> u16 {
        let at = address as usize;
        u16::from_le_bytes(self.bytes[at..at + 2].try_into().unwrap())
    }

    fn arm(&mut self, address: u32, code: &[u32]) {
        for (i, &insn) in code.iter().enumerate() {
            self.set_word(address + 4 * i as u32, insn);
        }
    }

    fn thumb(&mut self, address: u32, code: &[u16]) {
        for (i, &insn) in code.iter().enumerate() {
            let at = address as usize + 2 * i;
            self.bytes[at..at + 2].copy_from_slice(&insn.to_le_bytes());
        }
    }

    fn check(&self, address: u32, size: u8) -> Result<usize, BusError> {
        assert!(
            matches!(size, 1 | 2 | 4) && address.is_multiple_of(u32::from(size)),
            "the core made a {size}-byte access at {address:#x}"
        );
        if ABORTS.contains(&address) {
            return Err(BusError::Abort);
        }
        if UNMODELLED.contains(&address) {
            return Err(BusError::Unmodelled("a hole in the test bus".into()));
        }
        let at = address as usize;
        if at + usize::from(size) > RAM_SIZE {
            return Err(BusError::Abort);
        }
        Ok(at)
    }
}

impl BpmpBus for Ram {
    fn read(&mut self, address: u32, size: u8) -> Result<u32, BusError> {
        let at = self.check(address, size)?;
        let mut value = [0; 4];
        value[..usize::from(size)].copy_from_slice(&self.bytes[at..at + usize::from(size)]);
        Ok(u32::from_le_bytes(value))
    }

    fn write(&mut self, address: u32, size: u8, value: u32) -> Result<(), BusError> {
        let at = self.check(address, size)?;
        self.bytes[at..at + usize::from(size)]
            .copy_from_slice(&value.to_le_bytes()[..usize::from(size)]);
        if self.halt == Some(address) {
            self.halted = true;
        }
        Ok(())
    }

    fn exception_vector(&mut self, exception: Exception) -> u32 {
        match self.vectors {
            Some(table) => table[(exception.offset() / 4) as usize],
            None => exception.offset(),
        }
    }

    fn should_pause(&mut self) -> bool {
        self.halted
    }
}

/// A core in Supervisor mode, ARM state, at `CODE`, which holds `code`.
fn arm_core(code: &[u32]) -> (Arm7, Ram) {
    let mut ram = Ram::new();
    ram.arm(CODE, code);
    let mut core = Arm7::new();
    core.reset(CODE);
    (core, ram)
}

/// The same in Thumb state.
fn thumb_core(code: &[u16]) -> (Arm7, Ram) {
    let mut ram = Ram::new();
    ram.thumb(CODE, code);
    let mut core = Arm7::new();
    core.reset(CODE);
    core.set_cpsr(core.cpsr() | PSR_T);
    (core, ram)
}

fn exec(core: &mut Arm7, ram: &mut Ram) {
    let pc = core.pc();
    assert_eq!(core.step(ram), Ok(Step::Executed), "at {pc:#x}");
}

fn exec_n(core: &mut Arm7, ram: &mut Ram, n: usize) {
    for _ in 0..n {
        exec(core, ram);
    }
}

fn nzcv(core: &Arm7) -> u32 {
    core.cpsr() >> 28
}

fn set_nzcv(core: &mut Arm7, flags: u32) {
    core.set_cpsr(core.cpsr() & 0x0FFF_FFFF | flags << 28);
}

fn unpredictable(result: Result<Step, Fault>) -> &'static str {
    match result {
        Err(Fault {
            kind: FaultKind::Unpredictable(what),
            ..
        }) => what,
        other => panic!("expected an UNPREDICTABLE fault, got {other:?}"),
    }
}

/// Register values the tables start from, so a write of 0 shows.
fn sentinel(n: usize) -> u32 {
    0xDEAD_0000 + n as u32
}

/// (encoding, source, inputs, NZCV before, registers changed, NZCV after)
type Row = (
    u32,
    &'static str,
    &'static [(usize, u32)],
    u32,
    &'static [(usize, u32)],
    u32,
);

fn check_rows(rows: &[Row], thumb: bool) {
    for &(insn, source, inputs, flags_in, outputs, flags_out) in rows {
        let (mut core, mut ram) = if thumb {
            thumb_core(&[insn as u16])
        } else {
            arm_core(&[insn])
        };
        for n in 0..13 {
            core.set_reg(n, sentinel(n));
        }
        for &(n, value) in inputs {
            core.set_reg(n, value);
        }
        set_nzcv(&mut core, flags_in);
        let mut expected: Vec<u32> = (0..15).map(|n| core.reg(n)).collect();
        for &(n, value) in outputs {
            expected[n] = value;
        }
        assert_eq!(core.step(&mut ram), Ok(Step::Executed), "{source}");
        for (n, &value) in expected.iter().enumerate() {
            assert_eq!(core.reg(n), value, "{source}: r{n}");
        }
        let width = if thumb { 2 } else { 4 };
        assert_eq!(core.pc(), CODE + width, "{source}: pc");
        assert_eq!(nzcv(&core), flags_out, "{source}: NZCV");
    }
}

#[test]
fn the_shifter_handles_every_edge_case() {
    use Shift::*;
    let v = 0x8000_0001;
    // By immediate: LSL #0 keeps C, LSR/ASR #0 mean 32, ROR #0 is RRX.
    assert_eq!(shift_immediate(Lsl, v, 0, true), (v, true));
    assert_eq!(shift_immediate(Lsl, v, 0, false), (v, false));
    assert_eq!(shift_immediate(Lsl, v, 1, false), (2, true));
    assert_eq!(shift_immediate(Lsl, 3, 31, false), (0x8000_0000, true));
    assert_eq!(shift_immediate(Lsr, v, 0, false), (0, true));
    assert_eq!(shift_immediate(Lsr, v, 1, false), (0x4000_0000, true));
    assert_eq!(shift_immediate(Asr, v, 0, false), (0xFFFF_FFFF, true));
    assert_eq!(shift_immediate(Asr, 0x4000_0000, 0, true), (0, false));
    assert_eq!(shift_immediate(Asr, v, 31, false), (0xFFFF_FFFF, false));
    assert_eq!(shift_immediate(Ror, v, 0, false), (0x4000_0000, true));
    assert_eq!(shift_immediate(Ror, 2, 0, true), (0x8000_0001, false));
    assert_eq!(shift_immediate(Ror, v, 8, false), (0x0180_0000, false));
    // By register: the bottom byte counts; 0 keeps C; 32 and more fill.
    assert_eq!(shift_register(Lsl, v, 0, true), (v, true));
    assert_eq!(shift_register(Lsl, v, 0x100, false), (v, false));
    assert_eq!(shift_register(Lsl, v, 32, false), (0, true));
    assert_eq!(shift_register(Lsl, v, 33, true), (0, false));
    assert_eq!(shift_register(Lsr, v, 32, false), (0, true));
    assert_eq!(shift_register(Lsr, v, 33, true), (0, false));
    assert_eq!(shift_register(Asr, v, 32, false), (0xFFFF_FFFF, true));
    assert_eq!(shift_register(Asr, 0x7FFF_FFFF, 200, true), (0, false));
    assert_eq!(shift_register(Ror, v, 32, false), (v, true));
    assert_eq!(shift_register(Ror, v, 33, false), (0xC000_0000, true));
    assert_eq!(shift_register(Ror, v, 4, true), (0x1800_0000, false));
}

#[test]
fn add_with_carry_gives_carry_and_overflow() {
    assert_eq!(
        add_with_carry(0x7FFF_FFFF, 1, false),
        (0x8000_0000, false, true)
    );
    assert_eq!(add_with_carry(0xFFFF_FFFF, 1, false), (0, true, false));
    assert_eq!(add_with_carry(0xFFFF_FFFF, 0, true), (0, true, false));
    // 5 - 5 as 5 + !5 + 1: no borrow, so C is set.
    assert_eq!(add_with_carry(5, !5, true), (0, true, false));
    assert_eq!(
        add_with_carry(0x8000_0000, !1, true),
        (0x7FFF_FFFF, true, true)
    );
}

#[test]
fn arm_data_processing_and_multiplies_set_flags_as_specified() {
    #[rustfmt::skip]
    const ROWS: &[Row] = &[
    (0xE0902001, "adds r2, r0, r1", &[(0, 0x7FFFFFFF), (1, 0x1)], 0b0000, &[(2, 0x80000000)], 0b1001),
    (0xE0902001, "adds r2, r0, r1", &[(0, 0xFFFFFFFF), (1, 0x1)], 0b0000, &[(2, 0x0)], 0b0110),
    (0xE0902F81, "adds r2, r0, r1, lsl #31", &[(0, 0x80000000), (1, 0x1)], 0b0000, &[(2, 0x0)], 0b0111),
    (0xE0802001, "add r2, r0, r1", &[(0, 0xFFFFFFFF), (1, 0x2)], 0b1111, &[(2, 0x1)], 0b1111),
    (0xE0502001, "subs r2, r0, r1", &[(0, 0x5), (1, 0x5)], 0b0000, &[(2, 0x0)], 0b0110),
    (0xE0502001, "subs r2, r0, r1", &[(0, 0x1), (1, 0x7FFFFFFF)], 0b0000, &[(2, 0x80000002)], 0b1000),
    (0xE0502001, "subs r2, r0, r1", &[(0, 0x80000000), (1, 0x1)], 0b0000, &[(2, 0x7FFFFFFF)], 0b0011),
    (0xE2702000, "rsbs r2, r0, #0", &[(0, 0x1)], 0b0000, &[(2, 0xFFFFFFFF)], 0b1000),
    (0xE2702000, "rsbs r2, r0, #0", &[(0, 0x0)], 0b0000, &[(2, 0x0)], 0b0110),
    (0xE0B02001, "adcs r2, r0, r1", &[(0, 0x7FFFFFFF), (1, 0x0)], 0b0010, &[(2, 0x80000000)], 0b1001),
    (0xE0B02001, "adcs r2, r0, r1", &[(0, 0xFFFFFFFF), (1, 0x0)], 0b0010, &[(2, 0x0)], 0b0110),
    (0xE0B02001, "adcs r2, r0, r1", &[(0, 0x1), (1, 0x2)], 0b0000, &[(2, 0x3)], 0b0000),
    (0xE0D02001, "sbcs r2, r0, r1", &[(0, 0x5), (1, 0x5)], 0b0000, &[(2, 0xFFFFFFFF)], 0b1000),
    (0xE0D02001, "sbcs r2, r0, r1", &[(0, 0x5), (1, 0x5)], 0b0010, &[(2, 0x0)], 0b0110),
    (0xE0D02001, "sbcs r2, r0, r1", &[(0, 0x80000000), (1, 0x0)], 0b0000, &[(2, 0x7FFFFFFF)], 0b0011),
    (0xE0F02001, "rscs r2, r0, r1", &[(0, 0x5), (1, 0x5)], 0b0000, &[(2, 0xFFFFFFFF)], 0b1000),
    (0xE0F02001, "rscs r2, r0, r1", &[(0, 0x1), (1, 0x80000000)], 0b0010, &[(2, 0x7FFFFFFF)], 0b0011),
    (0xE1500001, "cmp r0, r1", &[(0, 0x3), (1, 0x3)], 0b0000, &[], 0b0110),
    (0xE1500001, "cmp r0, r1", &[(0, 0x2), (1, 0x3)], 0b0000, &[], 0b1000),
    (0xE1500001, "cmp r0, r1", &[(0, 0x7FFFFFFF), (1, 0xFFFFFFFF)], 0b0000, &[], 0b1001),
    (0xE1700001, "cmn r0, r1", &[(0, 0xFFFFFFFF), (1, 0x1)], 0b0000, &[], 0b0110),
    (0xE1700001, "cmn r0, r1", &[(0, 0x40000000), (1, 0x40000000)], 0b0000, &[], 0b1001),
    (0xE3100102, "tst r0, #0x80000000", &[(0, 0x80000000)], 0b0000, &[], 0b1010),
    (0xE3100001, "tst r0, #1", &[(0, 0x2)], 0b0010, &[], 0b0110),
    (0xE1300001, "teq r0, r1", &[(0, 0x7), (1, 0x7)], 0b0001, &[], 0b0101),
    (0xE0102081, "ands r2, r0, r1, lsl #1", &[(0, 0xFFFFFFFF), (1, 0x80000001)], 0b0000, &[(2, 0x2)], 0b0010),
    (0xE03020A1, "eors r2, r0, r1, lsr #1", &[(0, 0x0), (1, 0x3)], 0b0000, &[(2, 0x1)], 0b0010),
    (0xE1902241, "orrs r2, r0, r1, asr #4", &[(0, 0x0), (1, 0x80000008)], 0b0000, &[(2, 0xF8000000)], 0b1010),
    (0xE1D02261, "bics r2, r0, r1, ror #4", &[(0, 0xFFFFFFFF), (1, 0xF)], 0b0000, &[(2, 0xFFFFFFF)], 0b0010),
    (0xE1B02061, "movs r2, r1, rrx", &[(1, 0x3)], 0b0000, &[(2, 0x1)], 0b0010),
    (0xE1B02061, "movs r2, r1, rrx", &[(1, 0x2)], 0b0010, &[(2, 0x80000001)], 0b1000),
    (0xE1F02F81, "mvns r2, r1, lsl #31", &[(1, 0x1)], 0b0010, &[(2, 0x7FFFFFFF)], 0b0000),
    (0xE1B02001, "movs r2, r1, lsl #0", &[(1, 0x0)], 0b0011, &[(2, 0x0)], 0b0111),
    (0xE1B02021, "movs r2, r1, lsr #32", &[(1, 0x80000000)], 0b0000, &[(2, 0x0)], 0b0110),
    (0xE1B02041, "movs r2, r1, asr #32", &[(1, 0x80000000)], 0b0000, &[(2, 0xFFFFFFFF)], 0b1010),
    (0xE1B02041, "movs r2, r1, asr #32", &[(1, 0x7FFFFFFF)], 0b0010, &[(2, 0x0)], 0b0100),
    (0xE3B02102, "movs r2, #0x80000000", &[], 0b0000, &[(2, 0x80000000)], 0b1010),
    (0xE3B020FF, "movs r2, #255", &[], 0b0010, &[(2, 0xFF)], 0b0010),
    (0xE21024FF, "ands r2, r0, #0xff000000", &[(0, 0xF0000000)], 0b0000, &[(2, 0xF0000000)], 0b1010),
    (0xE3802FFF, "orr r2, r0, #0x3fc", &[(0, 0x1)], 0b0000, &[(2, 0x3FD)], 0b0000),
    (0xE1B02311, "movs r2, r1, lsl r3", &[(1, 0x80000001), (3, 0x0)], 0b0010, &[(2, 0x80000001)], 0b1010),
    (0xE1B02331, "movs r2, r1, lsr r3", &[(1, 0x80000001), (3, 0x0)], 0b0010, &[(2, 0x80000001)], 0b1010),
    (0xE1B02351, "movs r2, r1, asr r3", &[(1, 0x80000001), (3, 0x0)], 0b0010, &[(2, 0x80000001)], 0b1010),
    (0xE1B02311, "movs r2, r1, lsl r3", &[(1, 0x80000001), (3, 0x1)], 0b0000, &[(2, 0x2)], 0b0010),
    (0xE1B02331, "movs r2, r1, lsr r3", &[(1, 0x80000001), (3, 0x1)], 0b0000, &[(2, 0x40000000)], 0b0010),
    (0xE1B02351, "movs r2, r1, asr r3", &[(1, 0x80000001), (3, 0x1)], 0b0000, &[(2, 0xC0000000)], 0b1010),
    (0xE1B02311, "movs r2, r1, lsl r3", &[(1, 0x80000001), (3, 0x1F)], 0b0000, &[(2, 0x80000000)], 0b1000),
    (0xE1B02331, "movs r2, r1, lsr r3", &[(1, 0x80000001), (3, 0x1F)], 0b0000, &[(2, 0x1)], 0b0000),
    (0xE1B02351, "movs r2, r1, asr r3", &[(1, 0x80000001), (3, 0x1F)], 0b0000, &[(2, 0xFFFFFFFF)], 0b1000),
    (0xE1B02311, "movs r2, r1, lsl r3", &[(1, 0x80000001), (3, 0x20)], 0b0000, &[(2, 0x0)], 0b0110),
    (0xE1B02331, "movs r2, r1, lsr r3", &[(1, 0x80000001), (3, 0x20)], 0b0000, &[(2, 0x0)], 0b0110),
    (0xE1B02351, "movs r2, r1, asr r3", &[(1, 0x80000001), (3, 0x20)], 0b0000, &[(2, 0xFFFFFFFF)], 0b1010),
    (0xE1B02311, "movs r2, r1, lsl r3", &[(1, 0x80000001), (3, 0x21)], 0b0000, &[(2, 0x0)], 0b0100),
    (0xE1B02331, "movs r2, r1, lsr r3", &[(1, 0x80000001), (3, 0x21)], 0b0000, &[(2, 0x0)], 0b0100),
    (0xE1B02351, "movs r2, r1, asr r3", &[(1, 0x80000001), (3, 0x21)], 0b0000, &[(2, 0xFFFFFFFF)], 0b1010),
    (0xE1B02311, "movs r2, r1, lsl r3", &[(1, 0x80000001), (3, 0x100)], 0b0000, &[(2, 0x80000001)], 0b1000),
    (0xE1B02331, "movs r2, r1, lsr r3", &[(1, 0x80000001), (3, 0x100)], 0b0000, &[(2, 0x80000001)], 0b1000),
    (0xE1B02351, "movs r2, r1, asr r3", &[(1, 0x80000001), (3, 0x100)], 0b0000, &[(2, 0x80000001)], 0b1000),
    (0xE1B02371, "movs r2, r1, ror r3", &[(1, 0x80000001), (3, 0x0)], 0b0000, &[(2, 0x80000001)], 0b1000),
    (0xE1B02371, "movs r2, r1, ror r3", &[(1, 0x80000001), (3, 0x4)], 0b0000, &[(2, 0x18000000)], 0b0000),
    (0xE1B02371, "movs r2, r1, ror r3", &[(1, 0x80000001), (3, 0x20)], 0b0000, &[(2, 0x80000001)], 0b1010),
    (0xE1B02371, "movs r2, r1, ror r3", &[(1, 0x80000001), (3, 0x24)], 0b0000, &[(2, 0x18000000)], 0b0000),
    (0xE0020190, "mul r2, r0, r1", &[(0, 0xFFFFFFFE), (1, 0x3)], 0b0000, &[(2, 0xFFFFFFFA)], 0b0000),
    (0xE0234190, "mla r3, r0, r1, r4", &[(0, 0xFFFFFFFE), (1, 0x3), (4, 0xA)], 0b0000, &[(3, 0x4)], 0b0000),
    (0xE0120190, "muls r2, r0, r1", &[(0, 0x10000), (1, 0x10000)], 0b0011, &[(2, 0x0)], 0b0111),
    (0xE0120190, "muls r2, r0, r1", &[(0, 0xFFFFFFFF), (1, 0x1)], 0b0000, &[(2, 0xFFFFFFFF)], 0b1000),
    (0xE0865190, "umull r5, r6, r0, r1", &[(0, 0xFFFFFFFE), (1, 0x3)], 0b0000, &[(5, 0xFFFFFFFA), (6, 0x2)], 0b0000),
    (0xE0A65190, "umlal r5, r6, r0, r1", &[(0, 0xFFFFFFFE), (1, 0x3), (5, 0xFFFFFFFF), (6, 0x1)], 0b0000, &[(5, 0xFFFFFFF9), (6, 0x4)], 0b0000),
    (0xE0C65190, "smull r5, r6, r0, r1", &[(0, 0xFFFFFFFE), (1, 0x3)], 0b0000, &[(5, 0xFFFFFFFA), (6, 0xFFFFFFFF)], 0b0000),
    (0xE0E65190, "smlal r5, r6, r0, r1", &[(0, 0xFFFFFFFE), (1, 0x3), (5, 0x6), (6, 0x0)], 0b0000, &[(5, 0x0)], 0b0000),
    (0xE0D65190, "smulls r5, r6, r0, r1", &[(0, 0xFFFFFFFE), (1, 0x3)], 0b0011, &[(5, 0xFFFFFFFA), (6, 0xFFFFFFFF)], 0b1011),
    (0xE0965190, "umulls r5, r6, r0, r1", &[(0, 0x0), (1, 0x3)], 0b0000, &[(5, 0x0), (6, 0x0)], 0b0100),
    ];
    check_rows(ROWS, false);
}

#[test]
fn thumb_shifts_arithmetic_alu_and_high_register_operations() {
    #[rustfmt::skip]
    const ROWS: &[Row] = &[
    (0x004A, "lsls r2, r1, #1", &[(1, 0x80000001)], 0b0000, &[(2, 0x2)], 0b0010),
    (0x000A, "lsls r2, r1, #0", &[(1, 0x80000001)], 0b0011, &[(2, 0x80000001)], 0b1011),
    (0x084A, "lsrs r2, r1, #1", &[(1, 0x80000001)], 0b0000, &[(2, 0x40000000)], 0b0010),
    (0x080A, "lsrs r2, r1, #32", &[(1, 0x80000001)], 0b0000, &[(2, 0x0)], 0b0110),
    (0x110A, "asrs r2, r1, #4", &[(1, 0x80000008)], 0b0000, &[(2, 0xF8000000)], 0b1010),
    (0x100A, "asrs r2, r1, #32", &[(1, 0x80000000)], 0b0000, &[(2, 0xFFFFFFFF)], 0b1010),
    (0x180A, "adds r2, r1, r0", &[(0, 0x7FFFFFFF), (1, 0x1)], 0b0000, &[(2, 0x80000000)], 0b1001),
    (0x1A0A, "subs r2, r1, r0", &[(0, 0x1), (1, 0x1)], 0b0000, &[(2, 0x0)], 0b0110),
    (0x1DCA, "adds r2, r1, #7", &[(1, 0xFFFFFFFA)], 0b0000, &[(2, 0x1)], 0b0010),
    (0x1FCA, "subs r2, r1, #7", &[(1, 0x6)], 0b0000, &[(2, 0xFFFFFFFF)], 0b1000),
    (0x2200, "movs r2, #0", &[], 0b0011, &[(2, 0x0)], 0b0111),
    (0x22C8, "movs r2, #200", &[], 0b1000, &[(2, 0xC8)], 0b0000),
    (0x29C8, "cmp r1, #200", &[(1, 0xC8)], 0b0000, &[], 0b0110),
    (0x31C8, "adds r1, #200", &[(1, 0x7FFFFFFF)], 0b0000, &[(1, 0x800000C7)], 0b1001),
    (0x3901, "subs r1, #1", &[(1, 0x0)], 0b0000, &[(1, 0xFFFFFFFF)], 0b1000),
    (0x4001, "ands r1, r0", &[(0, 0xF0), (1, 0x3C)], 0b0011, &[(1, 0x30)], 0b0011),
    (0x4041, "eors r1, r0", &[(0, 0xF0), (1, 0xF0)], 0b0000, &[(1, 0x0)], 0b0100),
    (0x4081, "lsls r1, r0", &[(0, 0x21), (1, 0x3)], 0b0000, &[(1, 0x0)], 0b0100),
    (0x4081, "lsls r1, r0", &[(0, 0x20), (1, 0x3)], 0b0000, &[(1, 0x0)], 0b0110),
    (0x4081, "lsls r1, r0", &[(0, 0x0), (1, 0x3)], 0b0010, &[], 0b0010),
    (0x40C1, "lsrs r1, r0", &[(0, 0x20), (1, 0x80000000)], 0b0000, &[(1, 0x0)], 0b0110),
    (0x40C1, "lsrs r1, r0", &[(0, 0x4), (1, 0x18)], 0b0000, &[(1, 0x1)], 0b0010),
    (0x4101, "asrs r1, r0", &[(0, 0x28), (1, 0x80000000)], 0b0000, &[(1, 0xFFFFFFFF)], 0b1010),
    (0x4101, "asrs r1, r0", &[(0, 0x1), (1, 0x80000001)], 0b0000, &[(1, 0xC0000000)], 0b1010),
    (0x4141, "adcs r1, r0", &[(0, 0x1), (1, 0x1)], 0b0010, &[(1, 0x3)], 0b0000),
    (0x4181, "sbcs r1, r0", &[(0, 0x1), (1, 0x1)], 0b0000, &[(1, 0xFFFFFFFF)], 0b1000),
    (0x41C1, "rors r1, r0", &[(0, 0x4), (1, 0x80000001)], 0b0000, &[(1, 0x18000000)], 0b0000),
    (0x41C1, "rors r1, r0", &[(0, 0x20), (1, 0x80000001)], 0b0000, &[], 0b1010),
    (0x41C1, "rors r1, r0", &[(0, 0x0), (1, 0x80000001)], 0b0000, &[], 0b1000),
    (0x4201, "tst r1, r0", &[(0, 0x2), (1, 0x1)], 0b0000, &[], 0b0100),
    (0x4241, "negs r1, r0", &[(0, 0x1)], 0b0000, &[(1, 0xFFFFFFFF)], 0b1000),
    (0x4241, "negs r1, r0", &[(0, 0x0)], 0b0000, &[(1, 0x0)], 0b0110),
    (0x4241, "negs r1, r0", &[(0, 0x80000000)], 0b0000, &[(1, 0x80000000)], 0b1001),
    (0x4281, "cmp r1, r0", &[(0, 0x1), (1, 0x0)], 0b0000, &[], 0b1000),
    (0x42C1, "cmn r1, r0", &[(0, 0x1), (1, 0xFFFFFFFF)], 0b0000, &[], 0b0110),
    (0x4301, "orrs r1, r0", &[(0, 0x80000000), (1, 0x1)], 0b0000, &[(1, 0x80000001)], 0b1000),
    (0x4341, "muls r1, r0", &[(0, 0xFFFFFFFF), (1, 0x5)], 0b0011, &[(1, 0xFFFFFFFB)], 0b1011),
    (0x4381, "bics r1, r0", &[(0, 0x1), (1, 0x1)], 0b0000, &[(1, 0x0)], 0b0100),
    (0x43C1, "mvns r1, r0", &[(0, 0x0)], 0b0010, &[(1, 0xFFFFFFFF)], 0b1010),
    (0x4449, "add r1, r9", &[(1, 0x1), (9, 0x2)], 0b1111, &[(1, 0x3)], 0b1111),
    (0x4489, "add r9, r1", &[(1, 0x1), (9, 0x2)], 0b0000, &[(9, 0x3)], 0b0000),
    (0x44DA, "add r10, r11", &[(10, 0xFFFFFFFF), (11, 0x1)], 0b0000, &[(10, 0x0)], 0b0000),
    (0x468C, "mov r12, r1", &[(1, 0x5)], 0b0000, &[(12, 0x5)], 0b0000),
    (0x4661, "mov r1, r12", &[(12, 0x5)], 0b0000, &[(1, 0x5)], 0b0000),
    (0x4551, "cmp r1, r10", &[(1, 0x5), (10, 0x5)], 0b0000, &[], 0b0110),
    (0x458A, "cmp r10, r1", &[(1, 0x6), (10, 0x5)], 0b0000, &[], 0b1000),
    (0x45D1, "cmp r9, r10", &[(9, 0x80000000), (10, 0x1)], 0b0000, &[], 0b0011),
    ];
    check_rows(ROWS, true);
}

#[test]
fn pc_reads_eight_ahead_or_twelve_with_a_register_shift() {
    let (mut core, mut ram) = arm_core(&[
        0xE28F8000, // add r8, pc, #0
        0xE08F9717, // add r9, pc, r7, lsl r7
    ]);
    core.set_reg(7, 0x100);
    exec_n(&mut core, &mut ram, 2);
    assert_eq!(core.reg(8), CODE + 8);
    assert_eq!(core.reg(9), CODE + 4 + 12 + 0x100);

    let (mut core, mut ram) = thumb_core(&[
        0x46C0, // nop
        0x467D, // mov r5, pc
        0xA302, // add r3, pc, #8
    ]);
    exec_n(&mut core, &mut ram, 3);
    assert_eq!(core.reg(5), CODE + 2 + 4);
    // ADD Rd, PC, #imm uses the address + 4 with bits 1:0 cleared.
    assert_eq!(core.reg(3), ((CODE + 4 + 4) & !3) + 8);
}

#[test]
fn single_transfers_with_every_addressing_mode_and_unaligned_rotation() {
    let (mut core, mut ram) = arm_core(&[
        0xE5A01004, // str r1, [r0, #4]!
        0xE4102004, // ldr r2, [r0], #-4
        0xE7903104, // ldr r3, [r0, r4, lsl #2]
        0xE7401004, // strb r1, [r0, -r4]
        0xE5D05001, // ldrb r5, [r0, #1]
        0xE5906001, // ldr r6, [r0, #1]
        0xE5907002, // ldr r7, [r0, #2]
        0xE5908003, // ldr r8, [r0, #3]
        0xE5804006, // str r4, [r0, #6]
        0xE580F008, // str pc, [r0, #8]
        0xE5909008, // ldr r9, [r0, #8]
        0xE710A024, // ldr r10, [r0, -r4, lsr #32]
        0xE6D0B084, // ldrb r11, [r0], r4, lsl #1
        0xE530C002, // ldr r12, [r0, #-2]!
    ]);
    ram.set_word(DATA, 0xAABB_CCDD);
    core.set_reg(0, DATA);
    core.set_reg(1, 0x1122_3344);
    core.set_reg(4, 1);
    exec(&mut core, &mut ram);
    assert_eq!((ram.word(DATA + 4), core.reg(0)), (0x1122_3344, DATA + 4));
    exec(&mut core, &mut ram);
    assert_eq!((core.reg(2), core.reg(0)), (0x1122_3344, DATA));
    exec(&mut core, &mut ram);
    assert_eq!(core.reg(3), 0x1122_3344);
    exec(&mut core, &mut ram);
    assert_eq!(ram.word(DATA - 4), 0x4400_0000);
    exec(&mut core, &mut ram);
    assert_eq!(core.reg(5), 0xCC);
    // An unaligned LDR rotates the aligned word (ARMv4).
    exec_n(&mut core, &mut ram, 3);
    assert_eq!(core.reg(6), 0xDDAA_BBCC);
    assert_eq!(core.reg(7), 0xCCDD_AABB);
    assert_eq!(core.reg(8), 0xBBCC_DDAA);
    // An unaligned STR stores to the aligned word.
    exec(&mut core, &mut ram);
    assert_eq!(ram.word(DATA + 4), 1);
    // STR PC stores the instruction's address + 12.
    exec_n(&mut core, &mut ram, 2);
    assert_eq!(ram.word(DATA + 8), CODE + 36 + 12);
    assert_eq!(core.reg(9), CODE + 48);
    exec(&mut core, &mut ram);
    assert_eq!(core.reg(10), 0xAABB_CCDD);
    exec(&mut core, &mut ram);
    assert_eq!((core.reg(11), core.reg(0)), (0xDD, DATA + 2));
    exec(&mut core, &mut ram);
    assert_eq!((core.reg(12), core.reg(0)), (0xAABB_CCDD, DATA));
}

#[test]
fn halfword_and_signed_transfers() {
    let (mut core, mut ram) = arm_core(&[
        0xE1C010B8, // strh r1, [r0, #8]
        0xE1D020B8, // ldrh r2, [r0, #8]
        0xE1D030F8, // ldrsh r3, [r0, #8]
        0xE1D050D2, // ldrsb r5, [r0, #2]
        0xE1D060D1, // ldrsb r6, [r0, #1]
        0xE1B070B4, // ldrh r7, [r0, r4]!
        0xE05080F2, // ldrsh r8, [r0], #-2
        0xE10010B4, // strh r1, [r0, -r4]
        0xE1D090B1, // ldrh r9, [r0, #1]
    ]);
    ram.set_word(DATA, 0x80FF_7F01);
    core.set_reg(0, DATA);
    core.set_reg(1, 0x8081_8283);
    core.set_reg(4, 2);
    exec(&mut core, &mut ram);
    assert_eq!(ram.word(DATA + 8), 0x8283);
    exec_n(&mut core, &mut ram, 4);
    assert_eq!(core.reg(2), 0x8283);
    assert_eq!(core.reg(3), 0xFFFF_8283);
    assert_eq!(core.reg(5), 0xFFFF_FFFF);
    assert_eq!(core.reg(6), 0x7F);
    exec(&mut core, &mut ram);
    assert_eq!((core.reg(7), core.reg(0)), (0x80FF, DATA + 2));
    exec(&mut core, &mut ram);
    assert_eq!((core.reg(8), core.reg(0)), (0xFFFF_80FF, DATA));
    exec(&mut core, &mut ram);
    assert_eq!(ram.half(DATA - 2), 0x8283);
    // An unaligned halfword is UNPREDICTABLE on ARMv4: the core stops at it.
    let fault = core.step(&mut ram).unwrap_err();
    assert_eq!(fault.pc, CODE + 32);
    assert_eq!(fault.instruction, Some(0xE1D090B1));
    assert_eq!(core.pc(), CODE + 32);
    assert_eq!(core.reg(9), 0);
}

#[test]
fn block_transfers_in_all_four_addressing_modes() {
    let (mut core, mut ram) = arm_core(&[
        0xE8A0000E, // stmia r0!, {r1-r3}
        0xE9300070, // ldmdb r0!, {r4-r6}
        0xE9800006, // stmib r0, {r1, r2}
        0xE8170300, // ldmda r7, {r8, r9}
        0xE92D400F, // stmdb sp!, {r0-r3, lr}
        0xE8BD1F00, // ldmia sp!, {r8-r12}
        0xE8270002, // stmda r7!, {r1}
    ]);
    for (n, value) in [
        (0, DATA),
        (1, 1),
        (2, 2),
        (3, 3),
        (7, DATA + 8),
        (13, STACK),
        (14, 0xABCD),
    ] {
        core.set_reg(n, value);
    }
    exec(&mut core, &mut ram);
    assert_eq!(
        [ram.word(DATA), ram.word(DATA + 4), ram.word(DATA + 8)],
        [1, 2, 3]
    );
    assert_eq!(core.reg(0), DATA + 12);
    exec(&mut core, &mut ram);
    assert_eq!(
        [core.reg(4), core.reg(5), core.reg(6), core.reg(0)],
        [1, 2, 3, DATA]
    );
    ram.set_word(DATA + 4, 0);
    ram.set_word(DATA + 8, 0);
    exec(&mut core, &mut ram);
    assert_eq!(
        [ram.word(DATA), ram.word(DATA + 4), ram.word(DATA + 8)],
        [1, 1, 2]
    );
    assert_eq!(core.reg(0), DATA);
    exec(&mut core, &mut ram);
    assert_eq!([core.reg(8), core.reg(9)], [1, 2]);
    exec(&mut core, &mut ram);
    assert_eq!(core.reg(13), STACK - 20);
    let pushed: Vec<u32> = (0..5).map(|i| ram.word(STACK - 20 + 4 * i)).collect();
    assert_eq!(pushed, [DATA, 1, 2, 3, 0xABCD]);
    exec(&mut core, &mut ram);
    let popped: Vec<u32> = (8..13).map(|n| core.reg(n)).collect();
    assert_eq!(popped, [DATA, 1, 2, 3, 0xABCD]);
    assert_eq!(core.reg(13), STACK);
    ram.set_word(DATA + 8, 0);
    exec(&mut core, &mut ram);
    assert_eq!((ram.word(DATA + 8), core.reg(7)), (1, DATA + 4));
}

#[test]
fn a_base_in_the_register_list_follows_the_arm7tdmi() {
    let (mut core, mut ram) = arm_core(&[
        0xE8A00005, // stmia r0!, {r0, r2}
        0xE8A10003, // stmia r1!, {r0, r1}
        0xE931000A, // ldmdb r1!, {r1, r3}
        0xE8838010, // stmia r3, {r4, pc}
    ]);
    for (n, value) in [(0, DATA), (1, DATA + 0x20), (2, 0x22), (4, 0x44)] {
        core.set_reg(n, value);
    }
    // Lowest in the list: the original base is stored.
    exec(&mut core, &mut ram);
    assert_eq!(
        [ram.word(DATA), ram.word(DATA + 4), core.reg(0)],
        [DATA, 0x22, DATA + 8]
    );
    // Not the lowest: the written-back base is stored.
    exec(&mut core, &mut ram);
    assert_eq!(ram.word(DATA + 0x24), DATA + 0x28);
    assert_eq!(core.reg(1), DATA + 0x28);
    // A loaded base overrides the writeback.
    exec(&mut core, &mut ram);
    assert_eq!([core.reg(1), core.reg(3)], [DATA + 8, DATA + 0x28]);
    // STM stores PC as the instruction's address + 12.
    exec(&mut core, &mut ram);
    assert_eq!(
        [ram.word(DATA + 0x28), ram.word(DATA + 0x2C)],
        [0x44, CODE + 12 + 12]
    );
}

#[test]
fn unpredictable_encodings_stop_the_core_where_they_are() {
    for (insn, what) in [
        (0xE8900000, "no registers"),
        (0xE89F0003, "based on R15"),
        (0xE8E06000, "with writeback"),
        (0xE00F0190, "multiply with R15"),
        (0xF0000000, "NV"),
        (0xE5B00004, "writeback to its destination"),
        (0xE321F0F3, "T bit"),
        (0xE321F0D4, "reserved mode"),
        (0xE10FF000, "MRS to R15"),
    ] {
        let (mut core, mut ram) = arm_core(&[insn]);
        let cpsr = core.cpsr();
        let reason = unpredictable(core.step(&mut ram));
        assert!(reason.contains(what), "{insn:08x}: {reason}");
        assert_eq!((core.pc(), core.cpsr()), (CODE, cpsr));
    }
    for (insn, what) in [
        (0x4608, "low registers"),      // mov r0, r1 with H1 = H2 = 0
        (0x4780, "H1"),                 // blx r0 (ARMv5T)
        (0xC800, "no registers"),       // ldmia r0!, {}
        (0xBC00, "no registers"),       // pop {}
        (0x8801, "unaligned halfword"), // ldrh r1, [r0, #0], r0 odd
    ] {
        let (mut core, mut ram) = thumb_core(&[insn]);
        core.set_reg(0, DATA + 1);
        let reason = unpredictable(core.step(&mut ram));
        assert!(reason.contains(what), "{insn:04x}: {reason}");
        assert_eq!(core.pc(), CODE);
    }
    // User and System mode have no SPSR to return with.
    for insn in [0xE1B0F00E, 0xE8D0000F, 0xE14F4000] {
        // movs pc, lr; ldmia r0, {r0-r3}^; mrs r4, spsr
        let (mut core, mut ram) = arm_core(&[insn]);
        core.set_cpsr(Mode::System as u32);
        core.set_reg(0, DATA);
        unpredictable(core.step(&mut ram));
        assert_eq!(core.reg(0), DATA);
    }
}

#[test]
fn swp_and_swpb_exchange_memory_and_register() {
    let (mut core, mut ram) = arm_core(&[
        0xE1002091, // swp r2, r1, [r0]
        0xE1403094, // swpb r3, r4, [r0]
        0xE1005095, // swp r5, r5, [r0]
    ]);
    ram.set_word(DATA, 0x1234_5678);
    for (n, value) in [(0, DATA), (1, 0xCAFE_BABE), (4, 0x11), (5, 7)] {
        core.set_reg(n, value);
    }
    exec(&mut core, &mut ram);
    assert_eq!((core.reg(2), ram.word(DATA)), (0x1234_5678, 0xCAFE_BABE));
    exec(&mut core, &mut ram);
    assert_eq!((core.reg(3), ram.word(DATA)), (0xBE, 0xCAFE_BA11));
    exec(&mut core, &mut ram);
    assert_eq!((core.reg(5), ram.word(DATA)), (0xCAFE_BA11, 7));
}

#[test]
fn branches_links_and_loads_to_pc() {
    let (mut core, mut ram) = arm_core(&[
        0xEA000000, // b 1f
        0xE3A00001, // mov r0, #1
        0xEB000002, // 1: bl .+0x10
        0xE2811001, // add r1, r1, #1
        0xE592F000, // ldr pc, [r2]
        0xE3A00001, // mov r0, #1
        0xE1A0300E, // mov r3, lr
        0xE12FFF1E, // bx lr
        0xE3A04004, // mov r4, #4
        0xEAFFFFF5, // b .-0x24
    ]);
    ram.set_word(DATA, CODE + 0x20);
    core.set_reg(2, DATA);
    exec(&mut core, &mut ram);
    assert_eq!(core.pc(), CODE + 8);
    exec(&mut core, &mut ram);
    assert_eq!((core.pc(), core.reg(14)), (CODE + 0x18, CODE + 0xC));
    exec_n(&mut core, &mut ram, 2);
    assert_eq!(
        (core.reg(3), core.pc(), core.thumb()),
        (CODE + 0xC, CODE + 0xC, false)
    );
    exec(&mut core, &mut ram);
    assert_eq!(core.reg(1), 1);
    exec(&mut core, &mut ram);
    assert_eq!(core.pc(), CODE + 0x20);
    exec(&mut core, &mut ram);
    assert_eq!(core.reg(4), 4);
    exec(&mut core, &mut ram);
    assert_eq!(core.pc(), CODE);
    assert_eq!(core.reg(0), 0, "skipped instructions did not run");

    // LDM loading PC: bits 1:0 ignored, no change of state on ARMv4T.
    let (mut core, mut ram) = arm_core(&[0xE8908010]); // ldmia r0, {r4, pc}
    ram.set_word(DATA, 0x44);
    ram.set_word(DATA + 4, CODE + 0x103);
    core.set_reg(0, DATA);
    exec(&mut core, &mut ram);
    assert_eq!(
        (core.reg(4), core.pc(), core.thumb()),
        (0x44, CODE + 0x100, false)
    );
}

#[test]
fn bx_interworks_between_arm_and_thumb() {
    let mut ram = Ram::new();
    ram.arm(
        CODE,
        &[
            0xE28F0001, // add r0, pc, #1
            0xE12FFF10, // bx r0
        ],
    );
    ram.thumb(
        CODE + 8,
        &[
            0x2105, // movs r1, #5
            0x46C0, // nop
            0x4778, // bx pc
            0x46C0, // nop
        ],
    );
    ram.arm(
        CODE + 0x10,
        &[
            0xE3A02007, // mov r2, #7
            0xE28F3001, // add r3, pc, #1
            0xE12FFF13, // bx r3
        ],
    );
    ram.thumb(
        CODE + 0x1C,
        &[
            0x2401, // movs r4, #1
            0x467D, // mov r5, pc
            0x4728, // bx r5
        ],
    );
    let mut core = Arm7::new();
    core.reset(CODE);
    exec_n(&mut core, &mut ram, 2);
    assert_eq!((core.pc(), core.thumb()), (CODE + 8, true));
    exec_n(&mut core, &mut ram, 3);
    assert_eq!(
        (core.reg(1), core.pc(), core.thumb()),
        (5, CODE + 0x10, false)
    );
    exec_n(&mut core, &mut ram, 3);
    assert_eq!(
        (core.reg(2), core.pc(), core.thumb()),
        (7, CODE + 0x1C, true)
    );
    exec_n(&mut core, &mut ram, 2);
    assert_eq!(core.reg(5), CODE + 0x22);
    // BX to an ARM address that is not word-aligned is UNPREDICTABLE.
    let reason = unpredictable(core.step(&mut ram));
    assert!(reason.contains("unaligned ARM address"));
    assert_eq!(core.pc(), CODE + 0x20);
}

#[test]
fn thumb_bl_pair_links_with_bit_0_and_reaches_4_mib() {
    let (mut core, mut ram) = thumb_core(&[
        0xF123, 0xFA29, // bl .+0x123456
    ]);
    exec(&mut core, &mut ram);
    assert_eq!((core.reg(14), core.pc()), (CODE + 4 + 0x12_3000, CODE + 2));
    exec(&mut core, &mut ram);
    assert_eq!((core.pc(), core.reg(14)), (CODE + 0x12_3456, CODE + 4 + 1));

    let (mut core, mut ram) = thumb_core(&[
        0xF5FF, 0xFFFE, // bl .-0x200000
    ]);
    exec_n(&mut core, &mut ram, 2);
    assert_eq!(core.pc(), CODE.wrapping_sub(0x20_0000));
    assert!(core.thumb());
}

#[test]
fn thumb_branches_calls_and_returns() {
    let code = [
        0xD001, // beq 1f
        0xD100, // bne 1f
        0x46C0, // nop
        0xE000, // 1: b 2f
        0x46C0, // nop
        0xF000, 0xF802, // 2: bl 3f
        0x2101, // movs r1, #1
        0x4728, // bx r5
        0x2009, // 3: movs r0, #9
        0x4770, // bx lr
    ];
    let (mut core, mut ram) = thumb_core(&code);
    set_nzcv(&mut core, 0b0100);
    exec(&mut core, &mut ram);
    assert_eq!(core.pc(), CODE + 6, "beq taken");
    exec(&mut core, &mut ram);
    assert_eq!(core.pc(), CODE + 0xA, "b");
    exec_n(&mut core, &mut ram, 2);
    assert_eq!((core.pc(), core.reg(14)), (CODE + 0x12, CODE + 0xF));
    exec_n(&mut core, &mut ram, 2);
    assert_eq!(
        (core.reg(0), core.pc(), core.thumb()),
        (9, CODE + 0xE, true)
    );
    core.set_reg(5, CODE + 0x40);
    exec_n(&mut core, &mut ram, 2);
    assert_eq!(
        (core.reg(1), core.pc(), core.thumb()),
        (1, CODE + 0x40, false)
    );

    let (mut core, mut ram) = thumb_core(&code);
    set_nzcv(&mut core, 0);
    exec(&mut core, &mut ram);
    assert_eq!(core.pc(), CODE + 2, "beq not taken");
    exec(&mut core, &mut ram);
    assert_eq!(core.pc(), CODE + 6, "bne taken");
}

#[test]
fn thumb_load_store_and_address_formats() {
    let (mut core, mut ram) = thumb_core(&[
        0x490C, // ldr r1, [pc, #48]
        0x5081, // str r1, [r0, r2]
        0x54C1, // strb r1, [r0, r3]
        0x58C4, // ldr r4, [r0, r3]
        0x5C85, // ldrb r5, [r0, r2]
        0x5281, // strh r1, [r0, r2]
        0x56C6, // ldrsb r6, [r0, r3]
        0x5A87, // ldrh r7, [r0, r2]
        0x5E84, // ldrsh r4, [r0, r2]
        0x67C1, // str r1, [r0, #124]
        0x6FC5, // ldr r5, [r0, #124]
        0x77C1, // strb r1, [r0, #31]
        0x7FC6, // ldrb r6, [r0, #31]
        0x87C1, // strh r1, [r0, #62]
        0x8FC7, // ldrh r7, [r0, #62]
        0x91FF, // str r1, [sp, #1020]
        0x9AFF, // ldr r2, [sp, #1020]
        0xA302, // add r3, pc, #8
        0xACFF, // add r4, sp, #1020
        0xB07F, // add sp, #508
        0xB0FF, // sub sp, #508
    ]);
    ram.set_word(CODE + 0x34, 0x8182_8384);
    for (n, value) in [(0, DATA), (2, 8), (3, 13), (13, STACK)] {
        core.set_reg(n, value);
    }
    exec(&mut core, &mut ram);
    assert_eq!(core.reg(1), 0x8182_8384, "PC-relative literal");
    exec_n(&mut core, &mut ram, 2);
    assert_eq!(ram.word(DATA + 8), 0x8182_8384);
    assert_eq!(ram.word(DATA + 12), 0x8400);
    exec_n(&mut core, &mut ram, 2);
    assert_eq!(core.reg(4), 0x84, "unaligned LDR rotates");
    assert_eq!(core.reg(5), 0x84);
    ram.set_word(DATA + 8, 0);
    exec(&mut core, &mut ram);
    assert_eq!(ram.word(DATA + 8), 0x8384);
    exec_n(&mut core, &mut ram, 3);
    assert_eq!(core.reg(6), 0xFFFF_FF84);
    assert_eq!(core.reg(7), 0x8384);
    assert_eq!(core.reg(4), 0xFFFF_8384);
    exec_n(&mut core, &mut ram, 2);
    assert_eq!(
        (ram.word(DATA + 124), core.reg(5)),
        (0x8182_8384, 0x8182_8384)
    );
    exec_n(&mut core, &mut ram, 2);
    assert_eq!((ram.word(DATA + 28), core.reg(6)), (0x8400_0000, 0x84));
    exec_n(&mut core, &mut ram, 2);
    assert_eq!((ram.word(DATA + 60), core.reg(7)), (0x8384_0000, 0x8384));
    exec_n(&mut core, &mut ram, 2);
    assert_eq!(
        (ram.word(STACK + 1020), core.reg(2)),
        (0x8182_8384, 0x8182_8384)
    );
    exec_n(&mut core, &mut ram, 2);
    assert_eq!(core.reg(3), ((CODE + 34 + 4) & !3) + 8);
    assert_eq!(core.reg(4), STACK + 1020);
    exec(&mut core, &mut ram);
    assert_eq!(core.reg(13), STACK + 508);
    exec(&mut core, &mut ram);
    assert_eq!(core.reg(13), STACK);
}

#[test]
fn thumb_push_pop_and_multiple_transfers() {
    let (mut core, mut ram) = thumb_core(&[
        0xB503, // push {r0, r1, lr}
        0xBC1C, // pop {r2, r3, r4}
        0xB420, // push {r5}
        0xC607, // stmia r6!, {r0, r1, r2}
        0xBD01, // pop {r0, pc}
    ]);
    for (n, value) in [(0, 1), (1, 2), (5, 5), (6, DATA), (13, STACK), (14, 0x77)] {
        core.set_reg(n, value);
    }
    exec(&mut core, &mut ram);
    assert_eq!(core.reg(13), STACK - 12);
    assert_eq!(
        [
            ram.word(STACK - 12),
            ram.word(STACK - 8),
            ram.word(STACK - 4)
        ],
        [1, 2, 0x77]
    );
    exec(&mut core, &mut ram);
    assert_eq!(
        [core.reg(2), core.reg(3), core.reg(4), core.reg(13)],
        [1, 2, 0x77, STACK]
    );
    exec(&mut core, &mut ram);
    assert_eq!((core.reg(13), ram.word(STACK - 4)), (STACK - 4, 5));
    exec(&mut core, &mut ram);
    assert_eq!(
        [ram.word(DATA), ram.word(DATA + 4), ram.word(DATA + 8)],
        [1, 2, 1]
    );
    assert_eq!(core.reg(6), DATA + 12);
    // POP {PC} stays in Thumb state on ARMv4T.
    ram.set_word(STACK, CODE + 0x41);
    exec(&mut core, &mut ram);
    assert_eq!(
        (core.reg(0), core.pc(), core.thumb()),
        (5, CODE + 0x40, true)
    );
    assert_eq!(core.reg(13), STACK + 4);

    // LDMIA with the base in the list: the loaded value wins.
    let (mut core, mut ram) = thumb_core(&[0xCF88]); // ldmia r7!, {r3, r7}
    ram.set_word(DATA, 1);
    ram.set_word(DATA + 4, 2);
    core.set_reg(7, DATA);
    exec(&mut core, &mut ram);
    assert_eq!([core.reg(3), core.reg(7)], [1, 2]);
}

#[test]
fn msr_and_mrs_reach_the_fields_the_mode_allows() {
    let (mut core, mut ram) = arm_core(&[
        0xE10F0000, // mrs r0, cpsr
        0xE321F0D2, // msr cpsr_c, #0xd2
        0xE3A0DC01, // mov sp, #0x100
        0xE321F0D3, // msr cpsr_c, #0xd3
        0xE16FF001, // msr spsr_fsxc, r1
        0xE14F2000, // mrs r2, spsr
        0xE328F20F, // msr cpsr_f, #0xf0000000
        0xE321F010, // msr cpsr_c, #0x10
        0xE321F013, // msr cpsr_c, #0x13
        0xE128F003, // msr cpsr_f, r3
        0xE14F4000, // mrs r4, spsr
    ]);
    core.set_reg(1, 0x6000_0010);
    exec(&mut core, &mut ram);
    assert_eq!(core.reg(0), 0xD3);
    exec(&mut core, &mut ram);
    assert_eq!(core.mode(), Mode::Irq);
    exec_n(&mut core, &mut ram, 2);
    assert_eq!((core.mode(), core.reg(13)), (Mode::Supervisor, 0));
    assert_eq!(core.banked_reg(Mode::Irq, 13), 0x100);
    exec_n(&mut core, &mut ram, 2);
    assert_eq!(core.spsr(), Some(0x6000_0010));
    assert_eq!(core.reg(2), 0x6000_0010);
    exec(&mut core, &mut ram);
    assert_eq!(core.cpsr(), 0xF000_00D3);
    exec(&mut core, &mut ram);
    assert_eq!(core.cpsr(), 0xF000_0010);
    // User mode writes only the flags.
    exec(&mut core, &mut ram);
    assert_eq!(core.cpsr(), 0xF000_0010);
    exec(&mut core, &mut ram);
    assert_eq!(core.cpsr(), 0x10);
    unpredictable(core.step(&mut ram));
}

#[test]
fn banked_registers_follow_mode_switches() {
    let (mut core, mut ram) = arm_core(&[
        0xE3A08008, // mov r8, #8
        0xE3A0D00D, // mov r13, #13
        0xE321F0D1, // msr cpsr_c, #0xd1
        0xE3A08088, // mov r8, #0x88
        0xE3A0D099, // mov r13, #0x99
        0xE321F0D3, // msr cpsr_c, #0xd3
        0xE321F0DF, // msr cpsr_c, #0xdf
        0xE8C06100, // stmia r0, {r8, r13, r14}^
    ]);
    core.set_reg(0, DATA);
    exec_n(&mut core, &mut ram, 3);
    assert_eq!((core.mode(), core.reg(8), core.reg(13)), (Mode::Fiq, 0, 0));
    exec_n(&mut core, &mut ram, 3);
    assert_eq!(
        (core.mode(), core.reg(8), core.reg(13)),
        (Mode::Supervisor, 8, 13)
    );
    assert_eq!(core.banked_reg(Mode::Fiq, 8), 0x88);
    assert_eq!(core.banked_reg(Mode::Fiq, 13), 0x99);
    assert_eq!(core.banked_reg(Mode::User, 8), 8);
    assert_eq!(core.banked_reg(Mode::User, 13), 0);
    exec(&mut core, &mut ram);
    assert_eq!(
        (core.mode(), core.reg(8), core.reg(13)),
        (Mode::System, 8, 0)
    );
    assert_eq!(core.banked_reg(Mode::Supervisor, 13), 13);
    core.set_reg(13, 0x5555);
    core.set_reg(14, 0x6666);
    core.set_cpsr(0xD1);
    assert_eq!((core.reg(8), core.reg(13)), (0x88, 0x99));
    // STM ^ in FIQ mode stores the User registers, R8 included.
    exec(&mut core, &mut ram);
    assert_eq!(
        [ram.word(DATA), ram.word(DATA + 4), ram.word(DATA + 8)],
        [8, 0x5555, 0x6666]
    );
}

#[test]
fn user_bank_transfers_with_caret() {
    let (mut core, mut ram) = arm_core(&[
        0xE8C06000, // stmia r0, {r13, r14}^
        0xE2800008, // add r0, r0, #8
        0xE8D06000, // ldmia r0, {r13, r14}^
    ]);
    core.set_cpsr(Mode::System as u32);
    core.set_reg(13, 0xAAAA);
    core.set_reg(14, 0xBBBB);
    core.set_cpsr(0xD2);
    core.set_reg(13, 0x1111);
    core.set_reg(14, 0x2222);
    core.set_reg(0, DATA);
    ram.set_word(DATA + 8, 0x5555);
    ram.set_word(DATA + 12, 0x6666);
    exec_n(&mut core, &mut ram, 3);
    assert_eq!([ram.word(DATA), ram.word(DATA + 4)], [0xAAAA, 0xBBBB]);
    assert_eq!([core.reg(13), core.reg(14)], [0x1111, 0x2222]);
    assert_eq!(core.banked_reg(Mode::User, 13), 0x5555);
    assert_eq!(core.banked_reg(Mode::User, 14), 0x6666);
}

#[test]
fn ldm_with_pc_and_caret_returns_from_an_exception() {
    let (mut core, mut ram) = arm_core(&[0xE8FD8001]); // ldmia sp!, {r0, pc}^
    core.set_cpsr(Mode::System as u32);
    core.set_reg(13, 0x4444);
    core.set_cpsr(0xD3);
    core.set_reg(13, STACK);
    // The SPSR returns to User mode in Thumb state with Z and C set.
    core.spsr[Mode::Supervisor.bank()] = 0x6000_0030;
    ram.set_word(STACK, 0x77);
    ram.set_word(STACK + 4, CODE + 0x81);
    exec(&mut core, &mut ram);
    assert_eq!(core.cpsr(), 0x6000_0030);
    assert_eq!(
        (core.reg(0), core.pc(), core.reg(13)),
        (0x77, CODE + 0x80, 0x4444)
    );
    assert_eq!(core.banked_reg(Mode::Supervisor, 13), STACK + 8);
}

#[test]
fn swi_enters_supervisor_mode_and_movs_pc_lr_returns() {
    let (mut core, mut ram) = arm_core(&[
        0xEF123456, // swi #0x123456
        0xE3A01001, // mov r1, #1
    ]);
    ram.arm(
        0x08,
        &[
            0xE51E0004, // ldr r0, [lr, #-4]
            0xE3C004FF, // bic r0, r0, #0xff000000
            0xE1B0F00E, // movs pc, lr
        ],
    );
    core.set_cpsr(0x8000_0010);
    assert_eq!(
        core.step(&mut ram),
        Ok(Step::Exception(Exception::SoftwareInterrupt))
    );
    assert_eq!(
        core.cpsr(),
        0x8000_0093,
        "Supervisor, IRQ masked, flags kept"
    );
    assert_eq!(
        (core.reg(14), core.spsr(), core.pc()),
        (CODE + 4, Some(0x8000_0010), 0x08)
    );
    exec_n(&mut core, &mut ram, 3);
    assert_eq!(
        core.reg(0),
        0x12_3456,
        "the handler reads the comment field"
    );
    assert_eq!((core.pc(), core.cpsr()), (CODE + 4, 0x8000_0010));
    exec(&mut core, &mut ram);
    assert_eq!(core.reg(1), 1);

    let (mut core, mut ram) = thumb_core(&[0xDF42]); // swi #0x42
    assert_eq!(
        core.step(&mut ram),
        Ok(Step::Exception(Exception::SoftwareInterrupt))
    );
    assert_eq!(
        (core.reg(14), core.pc(), core.thumb()),
        (CODE + 2, 0x08, false)
    );
    assert_eq!(core.spsr(), Some(0xF3));
}

#[test]
fn undefined_and_coprocessor_instructions_trap() {
    for insn in [
        0xEE010F10, // mcr p15, 0, r0, c1, c0, 0
        0xEE000100, // cdp p1, 0, c0, c0, c0, 0
        0xED900200, // ldc p2, c0, [r0]
        0xE7F000F0, // the architecturally undefined space
        0xE0400090, // a multiply encoding ARMv4T leaves undefined
        0xE1600010, // the miscellaneous space (CLZ from ARMv5T)
    ] {
        let (mut core, mut ram) = arm_core(&[insn]);
        assert_eq!(
            core.step(&mut ram),
            Ok(Step::Exception(Exception::Undefined)),
            "{insn:08x}"
        );
        assert_eq!(
            (core.mode(), core.reg(14), core.pc()),
            (Mode::Undefined, CODE + 4, 0x04)
        );
        assert_eq!(core.spsr(), Some(0xD3));
    }
    for insn in [0xDE00, 0xBE00, 0xE800] {
        // undefined conditional branch, BKPT and BLX suffix (ARMv5T)
        let (mut core, mut ram) = thumb_core(&[insn]);
        assert_eq!(
            core.step(&mut ram),
            Ok(Step::Exception(Exception::Undefined)),
            "{insn:04x}"
        );
        assert_eq!(
            (core.reg(14), core.pc(), core.thumb()),
            (CODE + 2, 0x04, false)
        );
    }
}

#[test]
fn data_aborts_update_the_base_and_return_with_subs_pc_lr_8() {
    let (mut core, mut ram) = arm_core(&[0xE5B01004]); // ldr r1, [r0, #4]!
    ram.arm(0x10, &[0xE25EF008]); // subs pc, lr, #8
    core.set_reg(0, ABORTS.start - 4);
    core.set_reg(1, 0x11);
    assert_eq!(
        core.step(&mut ram),
        Ok(Step::Exception(Exception::DataAbort))
    );
    assert_eq!((core.reg(0), core.reg(1)), (ABORTS.start, 0x11));
    assert_eq!(
        (core.mode(), core.reg(14), core.pc()),
        (Mode::Abort, CODE + 8, 0x10)
    );
    exec(&mut core, &mut ram);
    assert_eq!((core.mode(), core.pc()), (Mode::Supervisor, CODE));

    // LDM: the words before the abort are not loaded; the base is updated.
    let (mut core, mut ram) = arm_core(&[0xE8B20018]); // ldmia r2!, {r3, r4}
    ram.set_word(ABORTS.start - 4, 0x99);
    core.set_reg(2, ABORTS.start - 4);
    assert_eq!(
        core.step(&mut ram),
        Ok(Step::Exception(Exception::DataAbort))
    );
    assert_eq!((core.reg(2), core.reg(3)), (ABORTS.start + 4, 0));

    let (mut core, mut ram) = arm_core(&[0xE8A20018]); // stmia r2!, {r3, r4}
    core.set_reg(2, ABORTS.start - 4);
    core.set_reg(3, 0x33);
    assert_eq!(
        core.step(&mut ram),
        Ok(Step::Exception(Exception::DataAbort))
    );
    assert_eq!(
        (ram.word(ABORTS.start - 4), core.reg(2)),
        (0x33, ABORTS.start + 4)
    );

    // In Thumb state the return address is the instruction + 8 as well.
    let (mut core, mut ram) = thumb_core(&[0x6801]); // ldr r1, [r0, #0]
    core.set_reg(0, ABORTS.start);
    assert_eq!(
        core.step(&mut ram),
        Ok(Step::Exception(Exception::DataAbort))
    );
    assert_eq!(
        (core.reg(14), core.thumb(), core.spsr()),
        (CODE + 8, false, Some(0xF3))
    );
}

#[test]
fn prefetch_aborts_are_taken_at_the_instruction() {
    let (mut core, mut ram) = arm_core(&[0xE1A0F000]); // mov pc, r0
    core.set_reg(0, ABORTS.start);
    exec(&mut core, &mut ram);
    assert_eq!(
        core.step(&mut ram),
        Ok(Step::Exception(Exception::PrefetchAbort))
    );
    assert_eq!(
        (core.mode(), core.reg(14), core.pc()),
        (Mode::Abort, ABORTS.start + 4, 0x0C)
    );
}

#[test]
fn interrupts_are_taken_between_instructions_and_return_with_subs_pc_lr_4() {
    let (mut core, mut ram) = arm_core(&[
        0xE3A00001, // mov r0, #1
        0xE3A00002, // mov r0, #2
    ]);
    ram.arm(0x18, &[0xE25EF004]); // subs pc, lr, #4
    ram.arm(0x1C, &[0xE3A09001]); // mov r9, #1
    core.set_cpsr(0x13);
    exec(&mut core, &mut ram);
    core.set_irq(true);
    assert_eq!(core.step(&mut ram), Ok(Step::Exception(Exception::Irq)));
    assert_eq!(core.cpsr(), 0x92);
    assert_eq!(
        (core.reg(14), core.spsr(), core.pc()),
        (CODE + 8, Some(0x13), 0x18)
    );
    // IRQ is masked inside the handler.
    exec(&mut core, &mut ram);
    assert_eq!((core.pc(), core.cpsr()), (CODE + 4, 0x13));
    core.set_irq(false);
    exec(&mut core, &mut ram);
    assert_eq!(core.reg(0), 2);

    // Masked: nothing happens.
    let (mut core, mut ram) = arm_core(&[0xE3A00001]);
    core.set_irq(true);
    core.set_fiq(true);
    exec(&mut core, &mut ram);

    // FIQ goes first and masks both.
    let (mut core, mut ram) = arm_core(&[0xE3A00001]);
    ram.arm(0x1C, &[0xE3A09001]); // mov r9, #1
    core.set_cpsr(0x13);
    core.set_irq(true);
    core.set_fiq(true);
    assert_eq!(core.step(&mut ram), Ok(Step::Exception(Exception::Fiq)));
    assert_eq!(
        (core.cpsr(), core.reg(14), core.pc()),
        (0xD1, CODE + 4, 0x1C)
    );
    exec(&mut core, &mut ram);
    assert_eq!(core.reg(9), 1);
    assert_eq!(
        core.banked_reg(Mode::Supervisor, 9),
        0,
        "R9 is banked in FIQ mode"
    );

    // From Thumb state: LR = next instruction + 4, the return restores T.
    let (mut core, mut ram) = thumb_core(&[0x2001, 0x2002]); // movs r0, #1; movs r0, #2
    ram.arm(0x18, &[0xE25EF004]); // subs pc, lr, #4
    core.set_cpsr(0x33);
    exec(&mut core, &mut ram);
    core.set_irq(true);
    assert_eq!(core.step(&mut ram), Ok(Step::Exception(Exception::Irq)));
    assert_eq!(
        (core.reg(14), core.spsr(), core.thumb()),
        (CODE + 6, Some(0x33), false)
    );
    core.set_irq(false);
    exec(&mut core, &mut ram);
    assert_eq!((core.pc(), core.thumb()), (CODE + 2, true));
    exec(&mut core, &mut ram);
    assert_eq!(core.reg(0), 2);
}

#[test]
fn exception_vectors_come_from_the_bus() {
    let (mut core, mut ram) = arm_core(&[0xEF000000]); // swi #0
    let mut table = [0; 8];
    table[2] = 0x4000_0203;
    ram.vectors = Some(table);
    assert_eq!(
        core.step(&mut ram),
        Ok(Step::Exception(Exception::SoftwareInterrupt))
    );
    assert_eq!(core.pc(), 0x4000_0200);
}

#[test]
fn unmodelled_accesses_stop_the_core_without_changing_registers() {
    let (mut core, mut ram) = arm_core(&[0xE4901004]); // ldr r1, [r0], #4
    core.set_reg(0, UNMODELLED.start);
    let fault = core.step(&mut ram).unwrap_err();
    assert_eq!(fault.pc, CODE);
    assert_eq!(fault.instruction, Some(0xE4901004));
    assert!(!fault.thumb);
    assert_eq!(
        fault.kind,
        FaultKind::Unmodelled {
            address: UNMODELLED.start,
            size: 4,
            write: false,
            reason: "a hole in the test bus".into(),
        }
    );
    assert!(fault.to_string().contains("4-byte read at 0x0000e000"));
    assert_eq!(
        (core.pc(), core.reg(0), core.reg(1)),
        (CODE, UNMODELLED.start, 0)
    );

    // A block store made before the fault stays made; the base does not move.
    let (mut core, mut ram) = arm_core(&[0xE8A20018]); // stmia r2!, {r3, r4}
    core.set_reg(2, UNMODELLED.start - 4);
    core.set_reg(3, 0x33);
    let fault = core.step(&mut ram).unwrap_err();
    assert!(matches!(
        fault.kind,
        FaultKind::Unmodelled { write: true, .. }
    ));
    assert_eq!(
        (ram.word(UNMODELLED.start - 4), core.reg(2)),
        (0x33, UNMODELLED.start - 4)
    );

    let mut core = Arm7::new();
    core.reset(UNMODELLED.start);
    let fault = core.step(&mut Ram::new()).unwrap_err();
    assert_eq!((fault.pc, fault.instruction), (UNMODELLED.start, None));
}

/// A cache over all of the test RAM, and the counter its other writers
/// bump.
fn cached_assist() -> (Assist, std::sync::Arc<std::sync::atomic::AtomicU64>) {
    let generation = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let mut assist = Assist::new();
    assist.set_code_cache(0..RAM_SIZE as u32, generation.clone());
    (assist, generation)
}

#[test]
fn the_code_cache_sees_the_cores_stores_and_counted_writes() {
    // ARM code that rewrites its own first instruction with a word store.
    let (mut core, mut ram) = arm_core(&[
        0xE3A00001, // mov r0, #1
        0xE5823000, // str r3, [r2]
        0xEAFFFFFC, // b .-8
    ]);
    let (mut assist, generation) = cached_assist();
    core.set_reg(2, CODE);
    core.set_reg(3, 0xE3A00002); // mov r0, #2
    core.run_assisted(&mut ram, &mut assist, 4).unwrap();
    assert_eq!(core.reg(0), 2, "the store replaced the cached instruction");
    // Another writer changes the code: unseen until it counts the write.
    ram.set_word(CODE, 0xE3A00005); // mov r0, #5
    ram.set_word(CODE + 4, 0xE1A00000); // nop, over the str
    core.set_reg(3, 0xE3A00003); // mov r0, #3
    core.run_assisted(&mut ram, &mut assist, 1).unwrap();
    assert_eq!(ram.word(CODE), 0xE3A00003, "the cached str ran");
    ram.set_word(CODE, 0xE3A00004); // mov r0, #4
    generation.fetch_add(1, std::sync::atomic::Ordering::Release);
    core.run_assisted(&mut ram, &mut assist, 3).unwrap();
    assert_eq!(core.reg(0), 4, "counted, the write is seen");
    assert_eq!(ram.word(CODE), 0xE3A00004, "and the nop replaced the str");

    // A byte store into the middle of a cached ARM instruction.
    let (mut core, mut ram) = arm_core(&[
        0xE3A00001, // mov r0, #1
        0xE5C23000, // strb r3, [r2]
        0xEAFFFFFC, // b .-8
    ]);
    let (mut assist, _) = cached_assist();
    core.set_reg(2, CODE + 1);
    core.set_reg(3, 0x10); // turns the mov into mov r1, #1
    core.run_assisted(&mut ram, &mut assist, 4).unwrap();
    assert_eq!((core.reg(0), core.reg(1)), (1, 1));

    // A halfword store over a cached Thumb instruction.
    let (mut core, mut ram) = thumb_core(&[
        0x2001, // movs r0, #1
        0x8013, // strh r3, [r2]
        0xE7FC, // b .-4
    ]);
    let (mut assist, _) = cached_assist();
    core.set_reg(2, CODE);
    core.set_reg(3, 0x2002); // movs r0, #2
    core.run_assisted(&mut ram, &mut assist, 4).unwrap();
    assert_eq!(core.reg(0), 2);

    // The same address in the other state is another instruction.
    let (mut core, mut ram) = arm_core(&[0xE12FFF11]); // bx r1
    ram.thumb(CODE + 4, &[0x4708]); // bx r1
    let (mut assist, _) = cached_assist();
    core.set_reg(1, CODE + 4 + 1);
    core.run_assisted(&mut ram, &mut assist, 1).unwrap();
    core.set_reg(1, CODE);
    core.run_assisted(&mut ram, &mut assist, 1).unwrap();
    assert_eq!((core.pc(), core.thumb()), (CODE, false));
}

#[test]
fn the_trace_keeps_the_last_steps_and_the_accesses_asked_for() {
    let (mut core, mut ram) = arm_core(&[
        0xE5901000, // ldr r1, [r0]
        0xE5954000, // ldr r4, [r5]
        0xE5801004, // str r1, [r0, #4]
        0xEF000000, // swi #0
    ]);
    ram.set_word(DATA, 0x1234);
    core.set_reg(0, DATA);
    core.set_reg(5, CODE);
    let mut assist = Assist::new();
    assist.set_trace(3, 2, |address| (DATA..STACK).contains(&address));
    core.run_assisted(&mut ram, &mut assist, 4).unwrap();
    let trace = assist.trace().unwrap();
    let steps: Vec<_> = trace.steps().map(|step| (step.pc, step.outcome)).collect();
    assert_eq!(
        steps,
        [
            (CODE + 4, Outcome::Executed),
            (CODE + 8, Outcome::Executed),
            (CODE + 12, Outcome::Exception(Exception::SoftwareInterrupt)),
        ]
    );
    assert_eq!(
        trace.lines(),
        [
            "at 0x00001004 (ARM)",
            "at 0x00001008 (ARM)",
            "at 0x0000100c (ARM): SoftwareInterrupt exception",
            "R 0x00008000/4 from 0x00001000: 0x1234",
            "W 0x00008004/4 from 0x00001008: 0x1234",
        ]
    );

    // A fault is the last step, and the refused access is kept.
    let (mut core, mut ram) = arm_core(&[0xE8900003]); // ldmia r0, {r0, r1}
    core.set_reg(0, UNMODELLED.start - 4);
    let mut assist = Assist::new();
    assist.set_trace(4, 4, |_| true);
    assert!(core.step_assisted(&mut ram, &mut assist).is_err());
    let trace = assist.trace().unwrap();
    assert_eq!(trace.steps().last().unwrap().outcome, Outcome::Fault);
    let accesses: Vec<_> = trace
        .accesses()
        .map(|a| (a.address, a.result.is_ok()))
        .collect();
    assert_eq!(
        accesses,
        [(UNMODELLED.start - 4, true), (UNMODELLED.start, false)]
    );
}

/// ARM code calls a Thumb function through BX; the function fills a
/// buffer with Fibonacci numbers using a BL'd helper and returns their
/// sum; the ARM code stores it and halts through a write the bus sees.
#[test]
fn run_executes_an_arm_and_thumb_program_until_it_halts() {
    let mut ram = Ram::new();
    ram.arm(
        CODE,
        &[
            0xE3A0D903, // mov sp, #0xC000
            0xE3A00902, // mov r0, #0x8000
            0xE3A0100C, // mov r1, #12
            0xE28F202D, // add r2, pc, #0x2D     @ fib_fill + 1
            0xE1A0E00F, // mov lr, pc
            0xE12FFF12, // bx r2
            0xE3A03A09, // mov r3, #0x9000
            0xE5830000, // str r0, [r3]
            0xE3A04A0A, // mov r4, #0xA000
            0xE5840000, // str r0, [r4]         @ halt
            0xEAFFFFFE, // b .
        ],
    );
    ram.thumb(
        CODE + 0x40,
        &[
            0xB570, // fib_fill: push {r4-r6, lr}
            0x2400, // movs r4, #0
            0x2501, // movs r5, #1
            0x2600, // movs r6, #0
            0x6004, // loop: str r4, [r0]
            0x3004, // adds r0, #4
            0x1936, // adds r6, r6, r4
            0xF000, 0xF806, // bl next
            0x3901, // subs r1, #1
            0xD1F8, // bne loop
            0x0030, // movs r0, r6
            0xBC70, // pop {r4-r6}
            0xBC08, // pop {r3}
            0x4718, // bx r3
            0x1962, // next: adds r2, r4, r5
            0x002C, // movs r4, r5
            0x0015, // movs r5, r2
            0x4770, // bx lr
        ],
    );
    ram.halt = Some(0xA000);
    let mut core = Arm7::new();
    core.reset(CODE);
    let steps = core.run(&mut ram, 10_000).unwrap();
    assert!(ram.halted);
    assert_eq!(steps, 6 + 4 + 12 * 11 + 4 + 4);
    assert_eq!(ram.word(0x9000), 232);
    let fib: Vec<u32> = (0..12).map(|i| ram.word(DATA + 4 * i)).collect();
    assert_eq!(fib, [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]);
    assert_eq!(
        (core.pc(), core.thumb(), core.reg(13)),
        (CODE + 0x28, false, STACK)
    );
    // Without a halt the budget ends the run.
    assert_eq!(core.run(&mut Ram::new(), 5), Ok(5));
}

#[test]
fn snapshot_round_trips_the_core() {
    let (mut core, mut ram) = arm_core(&[0xE321F0D1, 0xE3A08088]); // msr cpsr_c, #0xd1; mov r8, #0x88
    exec_n(&mut core, &mut ram, 2);
    core.set_irq(true);
    let saved = Mutex::new(core.clone()).save().unwrap();
    let restored = Mutex::new(Arm7::new());
    restored.restore(&saved).unwrap();
    assert_eq!(*restored.lock().unwrap(), core);

    let mut corrupt = core;
    corrupt.cpsr = 0x15;
    let bytes = encode(&corrupt).unwrap();
    assert!(restored.restore(&bytes).is_err());
}
