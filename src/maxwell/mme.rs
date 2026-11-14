//! The macro method expander (MME): small programs that NVN uploads into
//! the channel's macro instruction RAM and then calls through the 3D
//! class's `CALL_MME_MACRO(j)`/`CALL_MME_DATA(j)` methods (byte offsets
//! 0x3800 + 8j / 0x3804 + 8j, [public: NVIDIA open-gpu-doc
//! `classes/3d/clb197.h`]). A macro reads engine registers and emits
//! method calls into the engine that invoked it, so its effects are part of
//! the method stream and must be interpreted, not guessed.
//!
//! Instruction semantics [public: eden `video_core/macro.cpp`
//! `MacroInterpreterImpl`, Ryujinx `Ryujinx.Graphics.Gpu/Engine/MME/
//! MacroInterpreter.cs`, envytools `envydis/macro.c`; the two emulators
//! agree on everything below]:
//!
//! * 32-bit instructions; 8 registers, `r0` reads as 0 and ignores
//!   writes. A call starts with every register 0 except `r1`, which holds
//!   the call's first parameter; further parameters are taken in order by
//!   the "fetch" result operations.
//! * bits 2:0 operation (0 ALU, 1 add-immediate, 2 bitfield replace,
//!   3 extract-shift-left-immediate, 4 extract-shift-left-register,
//!   5 read register, 7 branch); bits 6:4 result operation; bit 7 exit;
//!   bits 10:8 destination, 13:11 source A, 16:14 source B; bits 31:14 a
//!   signed immediate; bits 21:17 ALU operation / bitfield source bit,
//!   26:22 bitfield width, 31:27 bitfield destination bit.
//! * Branches (`bit 4`: on non-zero, else on zero; `bit 5`: annul the delay
//!   slot) and the exit flag both have one delay slot; an exit flag on the
//!   instruction in a delay slot is ignored.
//! * The method address register holds a method (bits 11:0) and an
//!   increment (bits 17:12) added after every send.
//!
//! Execution is resumable: the interpreter runs as far as the parameters
//! received so far allow and suspends before an instruction that fetches a
//! parameter that has not arrived yet. The real MME blocks on its parameter
//! FIFO the same way; this makes the result independent of how the
//! pushbuffer splits a call's parameters.

use super::Unmodeled;

/// Entries of the start address RAM: macro `j` of `CALL_MME_MACRO(j)` is
/// taken modulo this [public: Ryujinx `GPFifoProcessor.MacrosCount` 0x80,
/// eden `macro_positions` modulo its size].
pub const MACRO_COUNT: usize = 0x80;

/// Instruction RAM words accepted. The GM20B's macro RAM size is not
/// documented publicly; Ryujinx models 0x10000 words
/// (`GPFifoClass.MacroCodeSize`). Uploads beyond it are refused rather than
/// wrapped.
pub const CODE_WORDS: usize = 0x10000;

/// Instructions one call may execute between parameter arrivals. Real
/// macros finish in a few hundred; a macro still running after this many
/// is polling state (e.g. a shadow scratch register) that nothing in the
/// model will change, which on the hardware is a hang. It is reported
/// instead of looping forever.
pub const STEP_LIMIT: u32 = 1 << 20;

/// What a macro acts on: the engine whose `CALL_MME_MACRO` started it.
pub trait MacroTarget {
    /// The engine's current value of method register `method`.
    fn read_method(&self, method: u32) -> Result<u32, Unmodeled>;
    /// A method call emitted by the macro, processed like one from the
    /// pushbuffer.
    fn send_method(&mut self, method: u32, argument: u32) -> Result<(), Unmodeled>;
}

/// One in-flight macro call.
#[derive(Debug, Clone)]
struct Call {
    macro_index: usize,
    pc: usize,
    registers: [u32; 8],
    carry: bool,
    method: u32,
    increment: u32,
    /// Branch target to take after the delay-slot instruction at `pc`.
    delayed_target: Option<usize>,
    /// The instruction at `pc` is the delay slot of an exit.
    exiting: bool,
    parameters: std::collections::VecDeque<u32>,
}

/// The channel's macro unit: instruction RAM, start addresses, and the call
/// in progress.
pub struct Mme {
    code: Vec<u32>,
    code_pointer: u32,
    start_addresses: [u32; MACRO_COUNT],
    start_pointer: u32,
    call: Option<Call>,
}

impl Default for Mme {
    fn default() -> Self {
        Self {
            code: Vec::new(),
            code_pointer: 0,
            start_addresses: [0; MACRO_COUNT],
            start_pointer: 0,
            call: None,
        }
    }
}

/// Bits `[low, low + width)` of `value`.
fn bits(value: u32, low: u32, width: u32) -> u32 {
    (value >> low) & ((1u32 << width) - 1)
}

impl Mme {
    /// `LOAD_MME_INSTRUCTION_RAM_POINTER`.
    pub fn set_code_pointer(&mut self, pointer: u32) {
        self.code_pointer = pointer;
    }

    /// `LOAD_MME_INSTRUCTION_RAM`: stores one word and advances the pointer.
    pub fn load_code(&mut self, word: u32) -> Result<(), Unmodeled> {
        let index = self.code_pointer as usize;
        if index >= CODE_WORDS {
            return Err(Unmodeled(format!(
                "MME: instruction RAM write at word {index:#x}, beyond the {CODE_WORDS:#x} words modeled"
            )));
        }
        if self.code.len() <= index {
            self.code.resize(index + 1, 0);
        }
        self.code[index] = word;
        self.code_pointer += 1;
        Ok(())
    }

    /// `LOAD_MME_START_ADDRESS_RAM_POINTER`.
    pub fn set_start_pointer(&mut self, pointer: u32) {
        self.start_pointer = pointer;
    }

    /// `LOAD_MME_START_ADDRESS_RAM`: binds the next macro index to a code
    /// position.
    pub fn load_start_address(&mut self, address: u32) -> Result<(), Unmodeled> {
        let index = self.start_pointer as usize;
        let slot = self.start_addresses.get_mut(index).ok_or_else(|| {
            Unmodeled(format!(
                "MME: start address RAM write at entry {index:#x}, beyond the {MACRO_COUNT:#x} entries"
            ))
        })?;
        *slot = address;
        self.start_pointer += 1;
        Ok(())
    }

    /// Whether a call is waiting for more parameters.
    pub fn busy(&self) -> bool {
        self.call.is_some()
    }

    /// A write to a macro method (`method >= 0xE00`): even methods start
    /// macro `(method - 0xE00) / 2` with `argument` as its first parameter,
    /// odd ones pass a further parameter to the running call.
    pub fn method(
        &mut self,
        method: u32,
        argument: u32,
        target: &mut dyn MacroTarget,
    ) -> Result<(), Unmodeled> {
        let macro_index = ((method - 0xE00) >> 1) as usize % MACRO_COUNT;
        if method & 1 == 0 {
            if let Some(call) = &self.call {
                return Err(Unmodeled(format!(
                    "MME: macro {:#x} started while macro {:#x} still waits for parameters",
                    macro_index, call.macro_index
                )));
            }
            let mut registers = [0; 8];
            registers[1] = argument;
            self.call = Some(Call {
                macro_index,
                pc: self.start_addresses[macro_index] as usize,
                registers,
                carry: false,
                method: 0,
                increment: 0,
                delayed_target: None,
                exiting: false,
                parameters: Default::default(),
            });
        } else {
            let call = self.call.as_mut().ok_or_else(|| {
                Unmodeled(format!(
                    "MME: parameter {argument:#x} for macro {macro_index:#x}, which is not running"
                ))
            })?;
            if call.macro_index != macro_index {
                return Err(Unmodeled(format!(
                    "MME: parameter for macro {macro_index:#x} while macro {:#x} runs",
                    call.macro_index
                )));
            }
            call.parameters.push_back(argument);
        }
        self.run(target)
    }

    /// Runs the current call until it exits or needs a parameter it does
    /// not have.
    fn run(&mut self, target: &mut dyn MacroTarget) -> Result<(), Unmodeled> {
        let Some(mut call) = self.call.take() else {
            return Ok(());
        };
        let mut budget = STEP_LIMIT;
        loop {
            budget -= 1;
            if budget == 0 {
                return Err(Unmodeled(format!(
                    "MME: macro {:#x} still running after {STEP_LIMIT} instructions at word {:#x} \
                     (a wait on state nothing in this model changes)",
                    call.macro_index, call.pc
                )));
            }
            let word = *self.code.get(call.pc).ok_or_else(|| {
                Unmodeled(format!(
                    "MME: macro {:#x} runs past the uploaded code at word {:#x}",
                    call.macro_index, call.pc
                ))
            })?;
            let operation = word & 7;
            let result_operation = bits(word, 4, 3);
            let fetches = operation != 7 && matches!(result_operation, 0 | 3 | 5 | 6);
            if fetches && call.parameters.is_empty() {
                self.call = Some(call);
                return Ok(());
            }
            let in_delay_slot = call.delayed_target.is_some() || call.exiting;
            let next = call.delayed_target.take().unwrap_or(call.pc + 1);
            let pc = call.pc;
            call.pc = next;
            if operation == 7 {
                if in_delay_slot {
                    return Err(Unmodeled(format!(
                        "MME: branch {word:#010x} in a delay slot at word {pc:#x}"
                    )));
                }
                let value = call.registers[bits(word, 11, 3) as usize];
                let on_not_zero = word & 0x10 != 0;
                if (value != 0) == on_not_zero {
                    let target_pc = pc as i64 + i64::from((word as i32) >> 14);
                    let target_pc = usize::try_from(target_pc).map_err(|_| {
                        Unmodeled(format!("MME: branch to {target_pc} at word {pc:#x}"))
                    })?;
                    if word & 0x20 != 0 {
                        call.pc = target_pc;
                    } else {
                        call.delayed_target = Some(target_pc);
                    }
                    continue;
                }
            } else {
                let result = call.alu(word, target)?;
                call.store(word, result, target)?;
            }
            if call.exiting {
                if !call.parameters.is_empty() {
                    return Err(Unmodeled(format!(
                        "MME: macro {:#x} exited with {} parameters unread",
                        call.macro_index,
                        call.parameters.len()
                    )));
                }
                return Ok(());
            }
            if word & 0x80 != 0 && !in_delay_slot {
                call.exiting = true;
            }
        }
    }
}

impl Call {
    fn register(&self, index: u32) -> u32 {
        self.registers[index as usize]
    }

    fn set_register(&mut self, index: u32, value: u32) {
        if index != 0 {
            self.registers[index as usize] = value;
        }
    }

    fn fetch(&mut self) -> u32 {
        // `run` only executes a fetching instruction with a parameter queued.
        self.parameters.pop_front().unwrap_or_default()
    }

    fn alu(&mut self, word: u32, target: &mut dyn MacroTarget) -> Result<u32, Unmodeled> {
        let a = self.register(bits(word, 11, 3));
        let b = self.register(bits(word, 14, 3));
        let immediate = ((word as i32) >> 14) as u32;
        let source_bit = bits(word, 17, 5);
        let width = bits(word, 22, 5);
        let destination_bit = bits(word, 27, 5);
        let mask = (1u32 << width).wrapping_sub(1);
        Ok(match word & 7 {
            0 => match bits(word, 17, 5) {
                0 => {
                    let (sum, carry) = a.overflowing_add(b);
                    self.carry = carry;
                    sum
                }
                1 => {
                    let wide = u64::from(a) + u64::from(b) + u64::from(self.carry);
                    self.carry = wide > u64::from(u32::MAX);
                    wide as u32
                }
                2 => {
                    // Carry is "no borrow".
                    self.carry = a >= b;
                    a.wrapping_sub(b)
                }
                3 => {
                    let wide = i64::from(a) - i64::from(b) - i64::from(!self.carry);
                    self.carry = wide >= 0;
                    wide as u32
                }
                8 => a ^ b,
                9 => a | b,
                10 => a & b,
                11 => a & !b,
                12 => !(a & b),
                other => {
                    return Err(Unmodeled(format!(
                        "MME: ALU operation {other} in {word:#010x}"
                    )));
                }
            },
            1 => a.wrapping_add(immediate),
            2 => (a & !(mask << destination_bit)) | (((b >> source_bit) & mask) << destination_bit),
            // Register-supplied shift counts of 32 or more: the public
            // reimplementations disagree (C# masks the count, C++ leaves it
            // undefined), so they are refused.
            3 if a < 32 => ((b >> a) & mask) << destination_bit,
            4 if a < 32 => ((b >> source_bit) & mask) << a,
            3 | 4 => {
                return Err(Unmodeled(format!("MME: shift by {a} in {word:#010x}")));
            }
            5 => target.read_method(a.wrapping_add(immediate))?,
            other => {
                return Err(Unmodeled(format!("MME: operation {other} in {word:#010x}")));
            }
        })
    }

    fn set_method(&mut self, value: u32) {
        self.method = value & 0xFFF;
        self.increment = bits(value, 12, 6);
    }

    fn send(&mut self, value: u32, target: &mut dyn MacroTarget) -> Result<(), Unmodeled> {
        target.send_method(self.method, value)?;
        self.method = (self.method + self.increment) & 0xFFF;
        Ok(())
    }

    fn store(
        &mut self,
        word: u32,
        result: u32,
        target: &mut dyn MacroTarget,
    ) -> Result<(), Unmodeled> {
        let destination = bits(word, 8, 3);
        match bits(word, 4, 3) {
            0 => {
                let parameter = self.fetch();
                self.set_register(destination, parameter);
            }
            1 => self.set_register(destination, result),
            2 => {
                self.set_register(destination, result);
                self.set_method(result);
            }
            3 => {
                let parameter = self.fetch();
                self.set_register(destination, parameter);
                self.send(result, target)?;
            }
            4 => {
                self.set_register(destination, result);
                self.send(result, target)?;
            }
            5 => {
                let parameter = self.fetch();
                self.set_register(destination, parameter);
                self.set_method(result);
            }
            6 => {
                self.set_register(destination, result);
                self.set_method(result);
                let parameter = self.fetch();
                self.send(parameter, target)?;
            }
            _ => {
                self.set_register(destination, result);
                self.set_method(result);
                self.send(bits(result, 12, 6), target)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand assembly of the encoding described in the module docs.
    mod asm {
        pub const FETCH: u32 = 0; // IgnoreAndFetch
        pub const MOVE: u32 = 1;
        pub const MOVE_SET_METHOD: u32 = 2;
        pub const FETCH_SEND: u32 = 3;
        pub const MOVE_SEND: u32 = 4;
        pub const MOVE_SET_METHOD_FETCH_SEND: u32 = 6;
        pub const EXIT: u32 = 0x80;

        pub fn add_imm(result: u32, dst: u32, a: u32, imm: i32) -> u32 {
            1 | result << 4 | dst << 8 | a << 11 | ((imm as u32) << 14)
        }
        pub fn alu(op: u32, result: u32, dst: u32, a: u32, b: u32) -> u32 {
            result << 4 | dst << 8 | a << 11 | b << 14 | op << 17
        }
        pub fn branch(not_zero: bool, annul: bool, a: u32, offset: i32) -> u32 {
            7 | (not_zero as u32) << 4 | (annul as u32) << 5 | a << 11 | ((offset as u32) << 14)
        }
        pub fn read(dst: u32, a: u32, imm: i32) -> u32 {
            5 | super::asm::MOVE << 4 | dst << 8 | a << 11 | ((imm as u32) << 14)
        }
        #[allow(clippy::too_many_arguments)]
        pub fn bitfield(
            op: u32,
            result: u32,
            dst: u32,
            a: u32,
            b: u32,
            src_bit: u32,
            width: u32,
            dst_bit: u32,
        ) -> u32 {
            op | result << 4
                | dst << 8
                | a << 11
                | b << 14
                | src_bit << 17
                | width << 22
                | dst_bit << 27
        }
    }
    use asm::*;

    #[derive(Default)]
    struct Recorder {
        sent: Vec<(u32, u32)>,
        registers: std::collections::HashMap<u32, u32>,
    }

    impl MacroTarget for Recorder {
        fn read_method(&self, method: u32) -> Result<u32, Unmodeled> {
            Ok(self.registers.get(&method).copied().unwrap_or(0))
        }
        fn send_method(&mut self, method: u32, argument: u32) -> Result<(), Unmodeled> {
            self.sent.push((method, argument));
            Ok(())
        }
    }

    fn load(code: &[u32]) -> Mme {
        let mut mme = Mme::default();
        mme.set_code_pointer(0);
        for &word in code {
            mme.load_code(word).unwrap();
        }
        mme.set_start_pointer(3);
        mme.load_start_address(0).unwrap();
        mme
    }

    /// Calls macro 3 with `parameters`, delivered `split` at a time.
    fn call(
        code: &[u32],
        parameters: &[u32],
        split: usize,
        target: &mut Recorder,
    ) -> Result<(), Unmodeled> {
        let mut mme = load(code);
        let start = 0xE00 + 2 * 3;
        mme.method(start, parameters[0], target)?;
        for chunk in parameters[1..].chunks(split.max(1)) {
            for &p in chunk {
                mme.method(start + 1, p, target)?;
            }
        }
        if mme.busy() {
            return Err(Unmodeled("still running".into()));
        }
        Ok(())
    }

    #[test]
    fn sends_with_method_increment_and_parameter_fetch() {
        // r1 = 0x1234 | 1 << 12 (method 0x234, increment 1); then send two
        // fetched parameters with MoveAndSetMethodFetchAndSend semantics.
        let code = [
            add_imm(MOVE_SET_METHOD, 0, 1, 0),
            // r2 = fetch; send r2 + 0 (MOVE_SEND of r2).
            add_imm(FETCH, 2, 0, 0),
            add_imm(MOVE_SEND, 2, 2, 0),
            add_imm(FETCH, 3, 0, 0) | EXIT,
            add_imm(MOVE_SEND, 3, 3, 0x10),
        ];
        let mut target = Recorder::default();
        call(&code, &[0x1234, 7, 9], 1, &mut target).unwrap();
        // The exit flag's delay slot still runs.
        assert_eq!(target.sent, vec![(0x234, 7), (0x235, 9 + 0x10)]);
    }

    #[test]
    fn delay_slots_and_annulled_branches() {
        // r1 = count; loop: send r1 to method 0x10, decrement, branch back
        // while non-zero. The delay slot of the branch also executes.
        let code = [
            add_imm(MOVE_SET_METHOD, 0, 0, 0x10),
            add_imm(MOVE_SEND, 0, 1, 0),          // 1: send r1
            add_imm(MOVE, 1, 1, -1),              // 2: r1 -= 1
            branch(true, false, 1, -2),           // 3: if r1 != 0 goto 1 (delay slot 4)
            add_imm(MOVE_SET_METHOD, 0, 0, 0x10), // 4: delay slot: reset method
            add_imm(MOVE, 0, 0, 0) | EXIT,
            add_imm(MOVE_SEND, 0, 0, 0x55), // exit delay slot
        ];
        let mut target = Recorder::default();
        call(&code, &[3], 1, &mut target).unwrap();
        assert_eq!(
            target.sent,
            vec![(0x10, 3), (0x10, 2), (0x10, 1), (0x10, 0x55)]
        );

        // Annulled taken branch: the next instruction is skipped.
        let code = [
            add_imm(MOVE_SET_METHOD, 0, 0, 0x20),
            branch(false, true, 0, 2),   // r0 == 0: jump to 2, annul
            add_imm(MOVE_SEND, 0, 0, 1), // skipped
            add_imm(MOVE_SEND, 0, 0, 2) | EXIT,
            add_imm(MOVE, 0, 0, 0),
        ];
        let mut target = Recorder::default();
        call(&code, &[0], 1, &mut target).unwrap();
        assert_eq!(target.sent, vec![(0x20, 2)]);
    }

    #[test]
    fn carry_chain_and_logic() {
        // r2 = 0xFFFFFFFF + 1 (carry), r3 = 0 + 0 + carry = 1;
        // r4 = 0 - 1 (borrow: carry clear), r5 = 5 - 2 - borrow = 2.
        let code = [
            add_imm(MOVE, 6, 0, -1), // r6 = 0xFFFFFFFF
            add_imm(MOVE, 7, 0, 1),  // r7 = 1
            alu(0, MOVE, 2, 6, 7),   // add
            alu(1, MOVE, 3, 0, 0),   // addc
            alu(2, MOVE, 4, 0, 7),   // sub
            add_imm(MOVE, 6, 0, 5),
            add_imm(MOVE, 7, 0, 2),
            alu(3, MOVE, 5, 6, 7),                         // subb
            add_imm(MOVE_SET_METHOD, 0, 0, 0x1000 | 0x40), // method 0x40, increment 1
            add_imm(MOVE_SEND, 0, 2, 0),
            add_imm(MOVE_SEND, 0, 3, 0),
            add_imm(MOVE_SEND, 0, 4, 0),
            add_imm(MOVE_SEND, 0, 5, 0),
            alu(12, MOVE_SEND, 0, 6, 7) | EXIT, // nand
            alu(11, MOVE_SEND, 0, 6, 7),        // and-not
        ];
        let mut target = Recorder::default();
        call(&code, &[0], 1, &mut target).unwrap();
        let values: Vec<u32> = target.sent.iter().map(|s| s.1).collect();
        assert_eq!(values, vec![0, 1, 0xFFFF_FFFF, 2, !(5 & 2), 5 & !2]);
    }

    #[test]
    fn bitfields_and_register_reads() {
        let code = [
            add_imm(MOVE, 2, 0, 0x5A),
            // replace: r3 = r0 with bits 7:4 of r2 put at bit 8 (width 4).
            bitfield(2, MOVE, 3, 0, 2, 4, 4, 8),
            // extract-shift-left-immediate: ((r2 >> r4) & 0xF) << 12, r4 = 1.
            add_imm(MOVE, 4, 0, 1),
            bitfield(3, MOVE, 5, 4, 2, 0, 4, 12),
            // extract-shift-left-register: ((r2 >> 4) & 0x3) << r4.
            bitfield(4, MOVE, 6, 4, 2, 4, 2, 0),
            read(7, 0, 0x123),
            add_imm(MOVE_SET_METHOD, 0, 0, 0x1000 | 0x80),
            add_imm(MOVE_SEND, 0, 3, 0),
            add_imm(MOVE_SEND, 0, 5, 0),
            add_imm(MOVE_SEND, 0, 6, 0) | EXIT,
            add_imm(MOVE_SEND, 0, 7, 0),
        ];
        let mut target = Recorder::default();
        target.registers.insert(0x123, 0xBEEF);
        call(&code, &[0], 1, &mut target).unwrap();
        let values: Vec<u32> = target.sent.iter().map(|s| s.1).collect();
        assert_eq!(values, vec![0x500, 0xD000, 0x2, 0xBEEF]);
    }

    #[test]
    fn suspends_for_parameters_and_checks_their_count() {
        // Sends three fetched parameters; delivery granularity must not
        // matter.
        let code = [
            add_imm(MOVE_SET_METHOD, 0, 0, 0x1000 | 0x100),
            add_imm(FETCH_SEND, 2, 1, 0), // send r1, fetch r2
            add_imm(FETCH_SEND, 3, 2, 0), // send r2, fetch r3
            add_imm(MOVE_SEND, 0, 3, 0) | EXIT,
            add_imm(MOVE, 0, 0, 0),
        ];
        for split in [1, 2] {
            let mut target = Recorder::default();
            call(&code, &[1, 2, 3], split, &mut target).unwrap();
            assert_eq!(target.sent, vec![(0x100, 1), (0x101, 2), (0x102, 3)]);
        }
        // A parameter too many is refused, and so is a missing one.
        let mut target = Recorder::default();
        assert!(call(&code, &[1, 2, 3, 4], 1, &mut target).is_err());
        let mut target = Recorder::default();
        assert!(call(&code, &[1, 2], 1, &mut target).is_err());
        // MoveAndSetMethodFetchAndSend sends the fetched parameter.
        let code = [
            add_imm(MOVE_SET_METHOD_FETCH_SEND, 0, 0, 0x77) | EXIT,
            add_imm(MOVE, 0, 0, 0),
        ];
        let mut target = Recorder::default();
        call(&code, &[0, 42], 1, &mut target).unwrap();
        assert_eq!(target.sent, vec![(0x77, 42)]);
    }
}

/// NVN's own macros, taken from the starter applet, run through the
/// interpreter. The driver assembles its macro RAM image at run time
/// (uploader at starter 0x25ac30); some macros also exist whole in
/// read-only data, and those are used here (addresses are ELF virtual
/// addresses of `fw_elf/0100000000001012/.../main.elf`; identification in
/// the NVN method-stream analysis, macro slots 1, 16, 25 and 28).
#[cfg(test)]
mod firmware_tests {
    use super::*;

    const STARTER: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../fw_elf/0100000000001012/ca6e167d711b924ba111ecb24f6a4ae0/section0_exefs/main.elf"
    );

    /// `count` words at ELF virtual address `address`.
    fn words(elf: &[u8], address: u64, count: usize) -> Vec<u32> {
        let u16_at = |o: usize| u16::from_le_bytes(elf[o..o + 2].try_into().unwrap());
        let u32_at = |o: usize| u32::from_le_bytes(elf[o..o + 4].try_into().unwrap());
        let u64_at = |o: usize| u64::from_le_bytes(elf[o..o + 8].try_into().unwrap());
        let phoff = u64_at(0x20) as usize;
        for i in 0..u16_at(0x38) as usize {
            let header = phoff + 56 * i;
            let (offset, vaddr, size) =
                (u64_at(header + 8), u64_at(header + 16), u64_at(header + 32));
            if u32_at(header) == 1 && (vaddr..vaddr + size).contains(&address) {
                let start = (offset + address - vaddr) as usize;
                return (0..count).map(|k| u32_at(start + 4 * k)).collect();
            }
        }
        panic!("address {address:#x} not in a loadable segment");
    }

    #[derive(Default)]
    struct Recorder {
        sent: Vec<(u32, u32)>,
        registers: std::collections::HashMap<u32, u32>,
    }

    impl MacroTarget for Recorder {
        fn read_method(&self, method: u32) -> Result<u32, Unmodeled> {
            Ok(self.registers.get(&method).copied().unwrap_or(0))
        }
        fn send_method(&mut self, method: u32, argument: u32) -> Result<(), Unmodeled> {
            self.sent.push((method, argument));
            Ok(())
        }
    }

    fn run(code: &[u32], parameters: &[u32], target: &mut Recorder) {
        let mut mme = Mme::default();
        for &word in code {
            mme.load_code(word).unwrap();
        }
        mme.load_start_address(0).unwrap();
        mme.method(0xE00, parameters[0], target).unwrap();
        for &p in &parameters[1..] {
            mme.method(0xE01, p, target).unwrap();
        }
        assert!(!mme.busy());
    }

    #[test]
    fn nvn_draw_clear_and_constant_buffer_macros() {
        let elf = std::fs::read(STARTER).expect("starter ELF (fw_elf) must be present");

        // Macro 25 (DrawArrays): BEGIN topology, array start, count, END.
        let mut target = Recorder::default();
        run(&words(&elf, 0x3c8f20, 5), &[4, 10, 3], &mut target);
        assert_eq!(
            target.sent,
            vec![(0x586, 4), (0x35d, 10), (0x35e, 3), (0x585, 0)]
        );

        // Macro 28 (DrawElements): index buffer address, type, first 0,
        // count.
        let mut target = Recorder::default();
        run(
            &words(&elf, 0x3c8fdc, 8),
            &[5, 0x12, 0x3456_7800, 1, 6],
            &mut target,
        );
        assert_eq!(
            target.sent,
            vec![
                (0x586, 5),
                (0x5f2, 0x12),
                (0x5f3, 0x3456_7800),
                (0x5f6, 1),
                (0x5f7, 0),
                (0x5f8, 6),
                (0x585, 0)
            ]
        );

        // Macro 1 (ClearColor): one CLEAR_SURFACE per layer of the target
        // in MRT_SELECT, as SET_COLOR_TARGET_THIRD_DIMENSION(2) (0x206 +
        // 0x10 * 2) gives them, the layer field advancing by 0x400.
        let mut target = Recorder::default();
        target.registers.insert(0x206 + 0x20, 3);
        let argument = 2 << 6 | 0xF << 2;
        run(&words(&elf, 0x3c7b60, 10), &[argument], &mut target);
        assert_eq!(
            target.sent,
            vec![
                (0x674, argument),
                (0x674, argument + 0x400),
                (0x674, argument + 0x800)
            ]
        );

        // Macro 16: selects the 0x7000-byte support constant buffer whose
        // address >> 8 is in SET_MME_SHADOW_SCRATCH(24), offset = argument
        // * 4.
        let mut target = Recorder::default();
        target.registers.insert(0xd18, 0x0012_3456);
        run(&words(&elf, 0x3c7da4, 6), &[0x190], &mut target);
        assert_eq!(
            target.sent,
            vec![
                (0x8e0, 0x7000),
                (0x8e1, 0x00),
                (0x8e2, 0x1234_5600),
                (0x8e3, 0x640)
            ]
        );
    }
}
