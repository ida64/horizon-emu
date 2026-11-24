//! Falcon, NVIDIA's microcontroller core inside the Tegra's engines (the
//! TSEC's is version 5): an interpreter for its instructions.
//!
//! Sources, for the facts: public documentation of the core (formats and
//! subopcodes, arithmetic, data, branches, IO, interrupts, the code TLB
//! and the processor state, whose pseudocode the operations here follow)
//! and a public table of the encodings version 5
//! added or changed: the short `mov` of an immediate (`0x`, `4x`, `8x`,
//! `dx`), the two-byte `st`, `cmp`, `mov` and `ld` (`20`..`26`, `32`,
//! `3f`), `st` with an offset (`35`), three-operand `add` with a 16-bit
//! immediate (`38`), `call` (`f3`), `iowr` (`f6`, `f7`), `lbra` and
//! `lcall` (`3e`, `7e`).
//!
//! The core is the processor alone: registers, flags, the stack, traps
//! and interrupt entry. Its code and data memories, its IO space and its
//! interrupt lines are the engine's ([`Bus`]). Code is fetched by virtual
//! address through the engine's TLB (`vm.rst`); a fetch that finds no
//! page, or more than one, traps.
//!
//! `mpush $rN` and the `mpop` family, which the disassembler names
//! without their operation, save and restore `$r0` to `$rN`.
//! \[inference\] The count is from Nintendo's own TSEC boot code, whose
//! first function takes 0x94 bytes of stack, does `mpush $r0`, puts a
//! 0x84-byte buffer at `$sp + 0x14` rounded down to 16, and leaves with
//! `mpopaddret $r0, 0x94`: the buffer fits the frame only if exactly one
//! register was pushed, and the return address is found only if the pop
//! takes that one back before 0x94 is added. \[assumption\] `$r0` is
//! pushed first and popped last; only code that reads the saved
//! registers in place could tell.
//!
//! The compare-and-branch forms (`33`: a register's low bits against an
//! immediate, branch if equal or if not) are in the disassembler's table
//! as unverified. \[inference\] They decode as the table says: in
//! Nintendo's boot code their targets are the heads of its copy loops
//! (`sub b32 $r12 4` … `bra b32 $r12 0x0 ne`) and the instructions after
//! its `exit`s. \[assumption\] The immediate is zero-extended, as the
//! table prints it, and no flag changes: nothing after one of them in
//! that code reads a flag.
//!
//! Refused ([`Step::Refused`]), so that the engine stops with a note
//! rather than run on with something a Falcon would not have done:
//!
//! * the crypto coprocessor and everything of the secure modes (`ccmd`,
//!   `cxset`, the indirect crypto commands, `$cx` and `$cauth`, a jump
//!   into a secret page): they work with secrets inside the chip;
//! * transfers started by the processor (`xcld`, `xdld`, `xdst`, the two
//!   waits and `xdfence`);
//! * `iords`, whose difference from `iord` the documentation does not
//!   give;
//! * a write of `$pc`, any special register the table does not name, and
//!   an opcode the disassembler's table has no entry for (whether a
//!   version 5 core traps on each of those is not established);
//! * a data access outside the data memory \[assumption: nothing public
//!   says whether the address wraps\].
//!
//! \[assumption\] `$sp` keeps as many bits as the data memory needs, its
//! low two zero (`data.rst`); the other special registers keep all 32.

use serde::{Deserialize, Serialize};

const FLAG_C: u32 = 1 << 8;
const FLAG_O: u32 = 1 << 9;
const FLAG_S: u32 = 1 << 10;
const FLAG_Z: u32 = 1 << 11;
const FLAG_IE0: u32 = 1 << 16;
const FLAG_IE1: u32 = 1 << 17;
const FLAG_IS0: u32 = 1 << 20;
const FLAG_IS1: u32 = 1 << 21;
const FLAG_TA: u32 = 1 << 24;
/// The version 4 bits `intr.rst` moves with the interrupt enables without
/// naming them: 18 is saved in 22, 26 in 29.
const FLAG_UNK18: u32 = 1 << 18;
const FLAG_UNK22: u32 = 1 << 22;
const FLAG_UNK26: u32 = 1 << 26;
const FLAG_UNK29: u32 = 1 << 29;

/// Trap reasons (`intr.rst`).
const TRAP_NO_HIT: u32 = 0xA;
const TRAP_MULTI_HIT: u32 = 0xB;

/// What a fetch of one code byte finds (`vm.rst`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fetch {
    Byte(u8),
    /// No page is mapped at the address.
    NoHit,
    /// More than one is.
    MultiHit,
    /// The page holds secret code: entering it is a switch to an
    /// authenticated mode.
    Secret,
    /// The page is still being uploaded.
    Busy,
}

/// A TLB operation (`vm.rst`): look-ups return the entry in the format
/// given there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TlbCommand {
    /// Clears the entry of a physical page.
    Invalidate(u32),
    /// The entry of a physical page.
    Physical(u32),
    /// The entry that covers a virtual address.
    Virtual(u32),
}

/// The engine around the processor.
pub trait Bus {
    /// The code byte at virtual `address`.
    fn fetch(&mut self, address: u32) -> Fetch;
    /// The data memory.
    fn data(&self) -> &[u8];
    /// Writes `bytes` into the data memory at `address`; the core has
    /// checked that they fit.
    fn store(&mut self, address: u32, bytes: &[u8]);
    /// Reads the IO register at `address` of the processor's IO space.
    /// `Err` is what the engine does not model.
    fn io_read(&mut self, address: u32) -> Result<u32, String>;
    fn io_write(&mut self, address: u32, value: u32) -> Result<(), String>;
    fn tlb(&mut self, command: TlbCommand) -> u32;
    /// Whether a pending, enabled interrupt line is routed to vector 0 and
    /// to vector 1.
    fn interrupts(&mut self) -> [bool; 2];
}

/// What one [`Falcon::step`] did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Step {
    /// An instruction ran, or an interrupt or trap was entered.
    Ran,
    /// `exit`, or a trap inside a trap handler: the processor has stopped.
    Exit,
    /// `sleep` with its flag set: the processor waits for an interrupt,
    /// its `$pc` still at the instruction.
    Sleep,
    /// The fetch waits for a page that is being uploaded.
    Stalled,
    /// Something this model does not do; the processor has not moved.
    Refused(String),
}

/// The processor's registers.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Falcon {
    r: [u32; 16],
    iv0: u32,
    iv1: u32,
    tv: u32,
    sp: u32,
    pc: u32,
    xcbase: u32,
    xdbase: u32,
    flags: u32,
    xtargets: u32,
    tstatus: u32,
}

/// An instruction that could not complete.
enum Stop {
    Trap(u32),
    Refused(String),
    Stalled,
}

type Done<T = ()> = Result<T, Stop>;

fn refuse<T>(what: impl Into<String>) -> Done<T> {
    Err(Stop::Refused(what.into()))
}

fn mask(size: u32) -> u32 {
    if size == 32 { !0 } else { (1 << size) - 1 }
}

fn sign(size: u32, value: u32) -> bool {
    value >> (size - 1) & 1 != 0
}

fn sign_extend(bits: u32, value: u32) -> u32 {
    let shift = 32 - bits;
    ((value << shift) as i32 >> shift) as u32
}

/// `C(a, b, c)` of `arith.rst`: the carry out of an addition whose inputs
/// have sign bits `a` and `b` and whose result has `c`.
fn carry(a: bool, b: bool, c: bool) -> bool {
    (a && b) || ((a || b) && !c)
}

/// `O(a, b, c)`: its signed overflow.
fn overflow(a: bool, b: bool, c: bool) -> bool {
    a == b && a != c
}

impl Falcon {
    pub fn pc(&self) -> u32 {
        self.pc
    }

    /// Starts at `entry`, as the host's start through `CPUCTL` does. The
    /// other registers stay as they are.
    pub fn start(&mut self, entry: u32) {
        self.pc = entry;
    }

    fn flag(&self, bit: u32) -> bool {
        self.flags & bit != 0
    }

    fn set_flag(&mut self, bit: u32, on: bool) {
        self.flags = self.flags & !bit | if on { bit } else { 0 };
    }

    fn set_sz(&mut self, size: u32, value: u32) {
        self.set_flag(FLAG_S, sign(size, value));
        self.set_flag(FLAG_Z, value & mask(size) == 0);
    }

    fn set_sp(&mut self, bus: &dyn Bus, value: u32) {
        let span = bus.data().len().next_power_of_two() as u32;
        self.sp = value & (span - 1) & !3;
    }

    /// Writes the low `size` bits of a register; the rest stay.
    fn write(&mut self, register: usize, size: u32, value: u32) {
        let mask = mask(size);
        self.r[register] = self.r[register] & !mask | value & mask;
    }

    fn load(&self, bus: &dyn Bus, size: u32, address: u32) -> Done<u32> {
        let bytes = (size / 8) as usize;
        let address = (address & !(bytes as u32 - 1)) as usize;
        let Some(data) = bus.data().get(address..address + bytes) else {
            return refuse(format!(
                "data read of {bytes} bytes at {address:#x}, outside the data memory"
            ));
        };
        let mut word = [0; 4];
        word[..bytes].copy_from_slice(data);
        Ok(u32::from_le_bytes(word))
    }

    /// `ST` of `data.rst`, with what it does to an unaligned datum.
    fn store(&self, bus: &mut dyn Bus, size: u32, address: u32, value: u32) -> Done {
        let bytes = size / 8;
        let value = match (size, address & 3) {
            (32, 1 | 3) => (value & 0xFF) << ((address & 3) * 8),
            (32, 2) => (value & 0xFFFF) << 16,
            (16, 1 | 3) => (value & 0xFF) << 8,
            _ => value,
        };
        let address = address & !(bytes - 1);
        if address as usize + bytes as usize > bus.data().len() {
            return refuse(format!(
                "data write of {bytes} bytes at {address:#x}, outside the data memory"
            ));
        }
        bus.store(address, &value.to_le_bytes()[..bytes as usize]);
        Ok(())
    }

    fn push(&mut self, bus: &mut dyn Bus, value: u32) -> Done {
        let sp = self.sp.wrapping_sub(4);
        self.set_sp(bus, sp);
        self.store(bus, 32, self.sp, value)
    }

    fn pop(&mut self, bus: &mut dyn Bus) -> Done<u32> {
        let value = self.load(bus, 32, self.sp)?;
        let sp = self.sp.wrapping_add(4);
        self.set_sp(bus, sp);
        Ok(value)
    }

    /// Saves the interrupt enables and clears them, as entering an
    /// interrupt or a trap handler does (`intr.rst`, version 4 and later).
    fn save_enables(&mut self) {
        self.set_flag(FLAG_IS0, self.flag(FLAG_IE0));
        self.set_flag(FLAG_IS1, self.flag(FLAG_IE1));
        self.set_flag(FLAG_UNK22, self.flag(FLAG_UNK18));
        self.set_flag(FLAG_UNK29, self.flag(FLAG_UNK26));
        self.flags &= !(FLAG_IE0 | FLAG_IE1 | FLAG_UNK18);
    }

    /// Delivers a trap; `Exit` if one is already being handled.
    fn trap(&mut self, bus: &mut dyn Bus, reason: u32) -> Step {
        if self.flag(FLAG_TA) {
            return Step::Exit;
        }
        self.flags |= FLAG_TA;
        self.tstatus = self.pc & 0xF_FFFF | reason << 20;
        self.save_enables();
        match self.push(bus, self.pc) {
            Ok(()) => {
                self.pc = self.tv;
                Step::Ran
            }
            Err(Stop::Refused(why)) => Step::Refused(why),
            Err(_) => Step::Refused("trap entry could not push".into()),
        }
    }

    /// One instruction, or the entry of a pending interrupt.
    pub fn step(&mut self, bus: &mut dyn Bus) -> Step {
        let pending = bus.interrupts();
        for (vector, enable) in [FLAG_IE0, FLAG_IE1].into_iter().enumerate() {
            if pending[vector] && self.flag(enable) {
                let saved = self.clone();
                if let Err(stop) = self.push(bus, self.pc) {
                    *self = saved;
                    return match stop {
                        Stop::Refused(why) => Step::Refused(why),
                        _ => Step::Refused("interrupt entry could not push".into()),
                    };
                }
                self.save_enables();
                self.pc = if vector == 0 { self.iv0 } else { self.iv1 };
                return Step::Ran;
            }
        }
        let saved = self.clone();
        match self.execute(bus) {
            Ok(step) => step,
            Err(stop) => {
                *self = saved;
                match stop {
                    Stop::Trap(reason) => self.trap(bus, reason),
                    Stop::Refused(why) => {
                        Step::Refused(format!("Falcon at {:#06x}: {why}", self.pc))
                    }
                    Stop::Stalled => Step::Stalled,
                }
            }
        }
    }

    fn fetch(&self, bus: &mut dyn Bus, offset: u32) -> Done<u8> {
        match bus.fetch(self.pc.wrapping_add(offset)) {
            Fetch::Byte(byte) => Ok(byte),
            Fetch::NoHit => Err(Stop::Trap(TRAP_NO_HIT)),
            Fetch::MultiHit => Err(Stop::Trap(TRAP_MULTI_HIT)),
            Fetch::Busy => Err(Stop::Stalled),
            Fetch::Secret => refuse(
                "execution enters a secret code page: the authenticated modes work with \
                 secrets inside the chip",
            ),
        }
    }

    /// Whether branch condition `code` (`branch.rst`) holds.
    fn condition(&self, code: u32) -> Done<bool> {
        let (c, o, s, z) = (
            self.flag(FLAG_C),
            self.flag(FLAG_O),
            self.flag(FLAG_S),
            self.flag(FLAG_Z),
        );
        Ok(match code {
            0..=7 => self.flags >> code & 1 != 0,
            8 => c,
            9 => o,
            0xA => s,
            0xB => z,
            0xC => !c && !z,
            0xD => c || z,
            0xE => true,
            0x10..=0x17 => self.flags >> (code - 0x10) & 1 == 0,
            0x18 => !c,
            0x19 => !o,
            0x1A => !s,
            0x1B => !z,
            0x1C => o == s && !z,
            0x1D => o != s || z,
            0x1E => o != s,
            0x1F => o == s,
            _ => return refuse(format!("branch condition {code:#x}")),
        })
    }

    /// `add`, `adc`, `sub`, `sbb` and the shifts (`arith.rst`), by
    /// subopcode; all four flags are written.
    fn arithmetic(&mut self, op: u32, size: u32, a: u32, b: u32) -> Done<u32> {
        let mask = mask(size);
        let (a, b) = (a & mask, b & mask);
        let c = u32::from(self.flag(FLAG_C));
        let result = match op {
            0..=3 => {
                let result = match op {
                    0 => a.wrapping_add(b),
                    1 => a.wrapping_add(b).wrapping_add(c),
                    2 => a.wrapping_sub(b),
                    _ => a.wrapping_sub(b).wrapping_sub(c),
                } & mask;
                let (sa, sb, sr) = (sign(size, a), sign(size, b), sign(size, result));
                if op < 2 {
                    self.set_flag(FLAG_C, carry(sa, sb, sr));
                    self.set_flag(FLAG_O, overflow(sa, sb, sr));
                } else {
                    self.set_flag(FLAG_C, !carry(sa, !sb, sr));
                    self.set_flag(FLAG_O, overflow(sa, !sb, sr));
                }
                result
            }
            4 | 5 | 7 | 0xC | 0xD => {
                let count = b & (size - 1);
                let mut result;
                let out;
                if count == 0 {
                    result = a;
                    out = false;
                } else if op == 4 || op == 0xC {
                    result = a << count;
                    if op == 0xC {
                        result |= c << (count - 1);
                    }
                    out = a >> (size - count) & 1 != 0;
                } else {
                    result = a >> count;
                    if op == 0xD {
                        result |= c << (size - count);
                    }
                    if op == 7 && sign(size, a) {
                        result |= !0 << (size - count);
                    }
                    out = a >> (count - 1) & 1 != 0;
                }
                self.set_flag(FLAG_C, out);
                self.set_flag(FLAG_O, false);
                result & mask
            }
            _ => return refuse(format!("sized subopcode {op:#x}: not in the public table")),
        };
        self.set_sz(size, result);
        Ok(result)
    }

    /// The compares, by subopcode 4 (`cmpu`), 5 (`cmps`) and 6 (`cmp`).
    fn compare(&mut self, op: u32, size: u32, a: u32, b: u32) -> Done {
        let mask = mask(size);
        let (a, b) = (a & mask, b & mask);
        let difference = a.wrapping_sub(b) & mask;
        let (sa, sb, sd) = (sign(size, a), sign(size, b), sign(size, difference));
        self.set_flag(FLAG_Z, difference == 0);
        match op {
            4 => self.set_flag(FLAG_C, !carry(sa, !sb, sd)),
            5 => self.set_flag(FLAG_C, overflow(sa, !sb, sd) ^ sd),
            6 => {
                self.set_flag(FLAG_C, !carry(sa, !sb, sd));
                self.set_flag(FLAG_O, overflow(sa, !sb, sd));
                self.set_flag(FLAG_S, sd);
            }
            _ => {
                return refuse(format!(
                    "compare subopcode {op:#x}: not in the public table"
                ));
            }
        }
        Ok(())
    }

    /// `not`, `neg`, `mov` and `hswap` by subopcode.
    fn unary(&mut self, op: u32, size: u32, source: u32) -> Done<u32> {
        let mask = mask(size);
        let source = source & mask;
        let result = match op {
            0 => !source & mask,
            1 => source.wrapping_neg() & mask,
            2 => return Ok(source),
            3 => (source >> (size / 2) | source << (size / 2)) & mask,
            _ => return refuse(format!("unary subopcode {op:#x}: not in the public table")),
        };
        self.set_flag(FLAG_O, op == 1 && result == 1 << (size - 1));
        self.set_sz(size, result);
        Ok(result)
    }

    /// The unsized operations with a destination, by subopcode: `mulu`,
    /// `muls`, `sext`, `extrs`, `and`, `or`, `xor`, `extr`, `xbit`, `div`,
    /// `mod`. `ins` (0xB) and the IO reads are the caller's.
    fn unsized_op(&mut self, op: u32, a: u32, b: u32) -> Done<u32> {
        Ok(match op {
            0 => (a & 0xFFFF) * (b & 0xFFFF),
            1 => sign_extend(16, a & 0xFFFF).wrapping_mul(sign_extend(16, b & 0xFFFF)),
            2 => {
                let result = sign_extend((b & 0x1F) + 1, a);
                self.set_sz(32, result);
                result
            }
            3 | 7 => {
                let low = b & 0x1F;
                let size_less_one = b >> 5 & 0x1F;
                let field = (2u64 << size_less_one) - 1;
                let mut result = ((u64::from(a) >> low) & field) as u32;
                let fill = op == 3 && a >> ((low + size_less_one) & 0x1F) & 1 != 0;
                if fill {
                    result |= !(field as u32);
                }
                self.set_flag(FLAG_S, fill);
                self.set_flag(FLAG_Z, result == 0);
                result
            }
            4..=6 => {
                let result = match op {
                    4 => a & b,
                    5 => a | b,
                    _ => a ^ b,
                };
                self.flags &= !(FLAG_C | FLAG_O);
                self.set_sz(32, result);
                result
            }
            8 => {
                let result = a >> (b & 0x1F) & 1;
                self.set_flag(FLAG_S, false);
                self.set_flag(FLAG_Z, result == 0);
                result
            }
            0xC | 0xD => {
                let quotient = a.checked_div(b).unwrap_or(!0);
                if op == 0xC {
                    quotient
                } else {
                    a.wrapping_sub(quotient.wrapping_mul(b))
                }
            }
            _ => {
                return refuse(format!(
                    "unsized subopcode {op:#x}: not in the public table"
                ));
            }
        })
    }

    fn insert(destination: u32, source: u32, field: u32) -> u32 {
        let low = field & 0x1F;
        let size = (field >> 5 & 0x1F) + 1;
        if low + size > 32 {
            return destination;
        }
        let mask = ((1u64 << size) - 1) as u32;
        destination & !(mask << low) | (source & mask) << low
    }

    fn bit(op: u32, value: u32, bit: u32) -> u32 {
        let bit = 1 << (bit & 0x1F);
        match op {
            9 => value | bit,
            0xA => value & !bit,
            _ => value ^ bit,
        }
    }

    fn special(&self, index: usize) -> Done<u32> {
        Ok(match index {
            0 => self.iv0,
            1 => self.iv1,
            3 => self.tv,
            4 => self.sp,
            5 => self.pc,
            6 => self.xcbase,
            7 => self.xdbase,
            8 => self.flags,
            0xB => self.xtargets,
            0xC => self.tstatus,
            9 | 0xA => return refuse("read of a crypto special register ($cx, $cauth)"),
            _ => return refuse(format!("read of special register {index}")),
        })
    }

    fn set_special(&mut self, bus: &dyn Bus, index: usize, value: u32) -> Done {
        match index {
            0 => self.iv0 = value,
            1 => self.iv1 = value,
            3 => self.tv = value,
            4 => self.set_sp(bus, value),
            6 => self.xcbase = value,
            7 => self.xdbase = value,
            8 => self.flags = value,
            0xB => self.xtargets = value,
            0xC => self.tstatus = value,
            5 => return refuse("write of $pc"),
            9 | 0xA => return refuse("write of a crypto special register ($cx, $cauth)"),
            _ => return refuse(format!("write of special register {index}")),
        }
        Ok(())
    }

    fn execute(&mut self, bus: &mut dyn Bus) -> Done<Step> {
        let b0 = u32::from(self.fetch(bus, 0)?);
        let unknown = |what: &str| -> Done<Step> {
            refuse(format!(
                "{what} (opcode {b0:#04x}): not in the public table of version 5 opcodes"
            ))
        };
        // Further bytes, as the instruction's length calls for them.
        let mut bytes = [b0, 0, 0, 0, 0, 0];
        let mut fetched = 1;
        macro_rules! need {
            ($length:expr) => {
                while fetched < $length {
                    bytes[fetched] = u32::from(self.fetch(bus, fetched as u32)?);
                    fetched += 1;
                }
            };
        }
        let mut next = None;
        let length;
        // The short moves of an immediate take the low nibble as register.
        if b0 >> 6 != 3 && b0 & 0x30 == 0 {
            length = [2, 3, 4][(b0 >> 6) as usize];
            need!(length);
            let immediate = match length {
                2 => sign_extend(8, bytes[1]),
                3 => sign_extend(16, bytes[1] | bytes[2] << 8),
                _ => sign_extend(24, bytes[1] | bytes[2] << 8 | bytes[3] << 16),
            };
            self.r[(b0 & 0xF) as usize] = immediate;
        } else if b0 >> 6 != 3 {
            let size = 8 << (b0 >> 6);
            let scale = size / 8;
            let op = b0 & 0x3F;
            need!(2);
            let (hi, lo) = ((bytes[1] >> 4) as usize, (bytes[1] & 0xF) as usize);
            match op {
                0x10..=0x1F => {
                    length = 3;
                    need!(3);
                    let sub = op & 0xF;
                    if sub == 8 {
                        let address = self.r[hi].wrapping_add(bytes[2] * scale);
                        let value = self.load(bus, size, address)?;
                        self.write(lo, size, value);
                    } else {
                        let result = self.arithmetic(sub, size, self.r[hi], bytes[2])?;
                        self.write(lo, size, result);
                    }
                }
                0x20 => {
                    length = 2;
                    self.store(bus, size, self.r[hi], self.r[lo])?;
                }
                0x21 => {
                    length = 2;
                    let address = self.sp.wrapping_add(self.r[lo].wrapping_mul(scale));
                    self.store(bus, size, address, self.r[hi])?;
                }
                0x24..=0x26 => {
                    length = 2;
                    self.compare(op & 0xF, size, self.r[hi], self.r[lo])?;
                }
                0x30 | 0x31 => {
                    length = 3 + (op & 1) as usize;
                    need!(length);
                    let wide = op & 1 != 0;
                    let raw = bytes[2] | if wide { bytes[3] << 8 } else { 0 };
                    let bits = if wide { 16 } else { 8 };
                    match lo {
                        1 if !wide => {
                            let address = self.sp.wrapping_add(raw * scale);
                            self.store(bus, size, address, self.r[hi])?;
                        }
                        4 => self.compare(4, size, self.r[hi], raw)?,
                        5 | 6 => {
                            self.compare(lo as u32, size, self.r[hi], sign_extend(bits, raw))?
                        }
                        _ => return unknown("sized immediate operation"),
                    }
                }
                0x32 => {
                    length = 2;
                    self.write(lo, size, self.r[hi]);
                }
                0x33 => {
                    // Compare and branch: subopcode bit 2 branches on
                    // unequal; 9 and up have a 16-bit immediate (bit 1) or
                    // a 16-bit displacement (bit 0).
                    let sub = lo as u32;
                    let (wide_immediate, wide_target) = match sub {
                        0 | 4 => (false, false),
                        9 | 0xD => (false, true),
                        0xA | 0xE => (true, false),
                        0xB | 0xF => (true, true),
                        _ => return unknown("compare and branch"),
                    };
                    let at = 3 + usize::from(wide_immediate);
                    length = at + 1 + usize::from(wide_target);
                    need!(length);
                    let immediate = bytes[2] | if wide_immediate { bytes[3] << 8 } else { 0 };
                    let displacement = if wide_target {
                        sign_extend(16, bytes[at] | bytes[at + 1] << 8)
                    } else {
                        sign_extend(8, bytes[at])
                    };
                    let equal = (self.r[hi] ^ immediate) & mask(size) == 0;
                    if equal == matches!(sub, 0 | 9..=0xB) {
                        next = Some(self.pc.wrapping_add(displacement));
                    }
                }
                0x34 => {
                    length = 3;
                    need!(3);
                    if lo != 0 {
                        return unknown("stack load");
                    }
                    let value = self.load(bus, size, self.sp.wrapping_add(bytes[2] * scale))?;
                    self.write(hi, size, value);
                }
                0x35 => {
                    length = 3;
                    need!(3);
                    let address = self.r[hi].wrapping_add(bytes[2] * scale);
                    self.store(bus, size, address, self.r[lo])?;
                }
                0x36 | 0x37 => {
                    length = 3 + (op & 1) as usize;
                    need!(length);
                    let wide = op & 1 != 0;
                    let immediate = bytes[2] | if wide { bytes[3] << 8 } else { 0 };
                    let sub = lo as u32;
                    if wide && sub > 3 {
                        return unknown("shift by a 16-bit immediate");
                    }
                    let result = self.arithmetic(sub, size, self.r[hi], immediate)?;
                    self.write(hi, size, result);
                }
                0x38 => {
                    length = 5;
                    need!(5);
                    let sub = bytes[4] & 0xF;
                    if sub > 3 {
                        return unknown("three-operand immediate operation");
                    }
                    let immediate = bytes[2] | bytes[3] << 8;
                    let result = self.arithmetic(sub, size, self.r[hi], immediate)?;
                    self.write(lo, size, result);
                }
                0x39 => {
                    length = 3;
                    need!(3);
                    let sub = bytes[2] & 0xF;
                    if sub > 3 {
                        return unknown("unary operation");
                    }
                    let result = self.unary(sub, size, self.r[hi])?;
                    self.write(lo, size, result);
                }
                0x3A => {
                    length = 3;
                    need!(3);
                    if bytes[2] & 0xF != 0 {
                        return unknown("stack load");
                    }
                    let address = self.sp.wrapping_add(self.r[lo].wrapping_mul(scale));
                    let value = self.load(bus, size, address)?;
                    self.write(hi, size, value);
                }
                0x3B => {
                    length = 3;
                    need!(3);
                    let result = self.arithmetic(bytes[2] & 0xF, size, self.r[hi], self.r[lo])?;
                    self.write(hi, size, result);
                }
                0x3C => {
                    length = 3;
                    need!(3);
                    let third = (bytes[2] >> 4) as usize;
                    match bytes[2] & 0xF {
                        8 => {
                            let address = self.r[hi].wrapping_add(self.r[lo].wrapping_mul(scale));
                            let value = self.load(bus, size, address)?;
                            self.write(third, size, value);
                        }
                        9 => {
                            let address =
                                self.r[hi].wrapping_add(self.r[third].wrapping_mul(scale));
                            self.store(bus, size, address, self.r[lo])?;
                        }
                        sub => {
                            let result = self.arithmetic(sub, size, self.r[hi], self.r[lo])?;
                            self.write(third, size, result);
                        }
                    }
                }
                0x3D => {
                    length = 2;
                    match lo {
                        0..=3 => {
                            let result = self.unary(lo as u32, size, self.r[hi])?;
                            self.write(hi, size, result);
                        }
                        4 => self.write(hi, size, 0),
                        5 => {
                            self.set_flag(FLAG_O, false);
                            self.set_sz(size, self.r[hi]);
                        }
                        _ => return unknown("one-register operation"),
                    }
                }
                0x3E => {
                    // `lbra` (3e) and `lcall` (7e): a 24-bit absolute target.
                    length = 4;
                    need!(4);
                    let target = bytes[1] | bytes[2] << 8 | bytes[3] << 16;
                    match b0 {
                        0x3E => next = Some(target),
                        0x7E => {
                            self.push(bus, self.pc.wrapping_add(4))?;
                            next = Some(target);
                        }
                        _ => return unknown("long branch"),
                    }
                }
                0x3F => {
                    length = 2;
                    let value = self.load(bus, size, self.r[hi])?;
                    self.write(lo, size, value);
                }
                _ => return unknown("sized operation"),
            }
        } else {
            need!(2);
            let (hi, lo) = ((bytes[1] >> 4) as usize, (bytes[1] & 0xF) as usize);
            match b0 {
                0xC0..=0xCF | 0xE0..=0xEF => {
                    let wide = b0 >= 0xE0;
                    length = if wide { 4 } else { 3 };
                    need!(length);
                    let sub = b0 & 0xF;
                    let raw = bytes[2] | if wide { bytes[3] << 8 } else { 0 };
                    let source = self.r[hi];
                    match sub {
                        0xE => return refuse("iords: its difference from iord is not documented"),
                        0xF if !wide => {
                            self.r[lo] = bus
                                .io_read(source.wrapping_add(raw * 4))
                                .map_err(Stop::Refused)?;
                        }
                        0xB => self.r[lo] = Self::insert(self.r[lo], source, raw),
                        1 => {
                            let immediate = sign_extend(if wide { 16 } else { 8 }, raw);
                            self.r[lo] = self.unsized_op(1, source, immediate)?;
                        }
                        2 | 8 if wide => return unknown("unsized operation"),
                        0 | 2..=8 | 0xC | 0xD => self.r[lo] = self.unsized_op(sub, source, raw)?,
                        _ => return unknown("unsized operation"),
                    }
                }
                0xD0..=0xDF => {
                    length = 5;
                    need!(5);
                    self.r[(b0 & 0xF) as usize] =
                        bytes[1] | bytes[2] << 8 | bytes[3] << 16 | bytes[4] << 24;
                }
                0xF0 | 0xF1 => {
                    let wide = b0 == 0xF1;
                    length = if wide { 4 } else { 3 };
                    need!(length);
                    let raw = bytes[2] | if wide { bytes[3] << 8 } else { 0 };
                    let value = self.r[hi];
                    let sub = lo as u32;
                    self.r[hi] = match sub {
                        1 => {
                            self.unsized_op(1, value, sign_extend(if wide { 16 } else { 8 }, raw))?
                        }
                        2 | 9..=0xC if wide => return unknown("unsized immediate operation"),
                        0 | 2 | 4..=6 => self.unsized_op(sub, value, raw)?,
                        3 => value & 0xFFFF | raw << 16,
                        9..=0xB => Self::bit(sub, value, raw),
                        0xC => {
                            let result = self.flags >> (raw & 0x1F) & 1;
                            self.set_flag(FLAG_S, false);
                            self.set_flag(FLAG_Z, result == 0);
                            result
                        }
                        _ => return unknown("unsized immediate operation"),
                    };
                }
                0xF2 => {
                    length = 3;
                    need!(3);
                    match lo {
                        8 => {
                            let bit = bytes[2] & 0x1F;
                            self.flags = self.flags & !(1 << bit) | (self.r[hi] & 1) << bit;
                        }
                        0xC => return refuse("indirect crypto command"),
                        _ => return unknown("f2 operation"),
                    }
                }
                0xF3 => {
                    length = 3;
                    need!(3);
                    self.push(bus, self.pc.wrapping_add(3))?;
                    next = Some(bytes[1] | bytes[2] << 8);
                }
                0xF4 | 0xF5 => {
                    let wide = b0 == 0xF5;
                    length = if wide { 4 } else { 3 };
                    need!(length);
                    let raw = bytes[2] | if wide { bytes[3] << 8 } else { 0 };
                    let signed = sign_extend(if wide { 16 } else { 8 }, raw);
                    let sub = bytes[1] & 0x3F;
                    match sub {
                        0..=0x1F => {
                            if self.condition(sub)? {
                                next = Some(self.pc.wrapping_add(signed));
                            }
                        }
                        0x20 => next = Some(raw),
                        0x21 if !wide => {
                            self.push(bus, self.pc.wrapping_add(3))?;
                            next = Some(raw);
                        }
                        0x28 if !wide => {
                            if self.flags >> (raw & 0x1F) & 1 != 0 {
                                return Ok(Step::Sleep);
                            }
                        }
                        0x30 => {
                            let sp = self.sp.wrapping_add(signed);
                            self.set_sp(bus, sp);
                        }
                        0x31..=0x33 if !wide => {
                            self.flags = Self::bit(sub - 0x31 + 9, self.flags, raw);
                        }
                        0x3C => return refuse("crypto coprocessor command"),
                        _ => return unknown("f4/f5 operation"),
                    }
                }
                0xF6 | 0xF7 => {
                    length = 3;
                    need!(3);
                    bus.io_write(self.r[hi].wrapping_add(bytes[2] * 4), self.r[lo])
                        .map_err(Stop::Refused)?;
                }
                0xF8 => {
                    length = 2;
                    match bytes[1] & 0xF {
                        0 => next = Some(self.pop(bus)?),
                        1 => {
                            next = Some(self.pop(bus)?);
                            self.set_flag(FLAG_IE0, self.flag(FLAG_IS0));
                            self.set_flag(FLAG_IE1, self.flag(FLAG_IS1));
                            self.set_flag(FLAG_UNK18, self.flag(FLAG_UNK22));
                            self.set_flag(FLAG_UNK26, self.flag(FLAG_UNK29));
                        }
                        2 => {
                            self.pc = self.pc.wrapping_add(2);
                            return Ok(Step::Exit);
                        }
                        3 | 6 | 7 => return refuse("wait for a transfer the processor started"),
                        8..=0xB => {
                            // The return is to the instruction after.
                            self.pc = self.pc.wrapping_add(2);
                            return Ok(self.trap(bus, bytes[1] & 3));
                        }
                        _ => return unknown("f8 operation"),
                    }
                }
                0xF9 => {
                    length = 2;
                    match lo {
                        0 => self.push(bus, self.r[hi])?,
                        1 => {
                            let sp = self.sp.wrapping_add(self.r[hi]);
                            self.set_sp(bus, sp);
                        }
                        2 => {
                            for register in 0..=hi {
                                self.push(bus, self.r[register])?;
                            }
                        }
                        4 => next = Some(self.r[hi]),
                        5 => {
                            self.push(bus, self.pc.wrapping_add(2))?;
                            next = Some(self.r[hi]);
                        }
                        8 => {
                            bus.tlb(TlbCommand::Invalidate(self.r[hi] & 0xFF_FFFF));
                        }
                        9..=0xB => self.flags = Self::bit(lo as u32, self.flags, self.r[hi]),
                        _ => return unknown("f9 operation"),
                    }
                }
                0xFA => {
                    length = 3;
                    need!(3);
                    match bytes[2] & 0xF {
                        0 | 1 => bus
                            .io_write(self.r[hi], self.r[lo])
                            .map_err(Stop::Refused)?,
                        4..=6 => return refuse("transfer started by the processor"),
                        8 => {
                            let bit = self.r[lo] & 0x1F;
                            self.flags = self.flags & !(1 << bit) | (self.r[hi] & 1) << bit;
                        }
                        _ => return unknown("fa operation"),
                    }
                }
                0xFB => {
                    // `mpop` and its forms: bit 0 of the subopcode returns
                    // afterwards, bits 2:1 say whether an immediate is
                    // added to `$sp` and how wide it is.
                    let sub = bytes[1] & 7;
                    length = match sub {
                        0 | 1 => 2,
                        2 | 3 => 4,
                        4 | 5 => 3,
                        _ => return unknown("fb operation"),
                    };
                    need!(length);
                    for register in (0..=hi).rev() {
                        self.r[register] = self.pop(bus)?;
                    }
                    let adjust = match length {
                        4 => sign_extend(16, bytes[2] | bytes[3] << 8),
                        3 => sign_extend(8, bytes[2]),
                        _ => 0,
                    };
                    let sp = self.sp.wrapping_add(adjust);
                    self.set_sp(bus, sp);
                    if sub & 1 != 0 {
                        next = Some(self.pop(bus)?);
                    }
                }
                0xFC => {
                    length = 2;
                    if lo != 0 {
                        return unknown("fc operation");
                    }
                    self.r[hi] = self.pop(bus)?;
                }
                0xFD => {
                    length = 3;
                    need!(3);
                    let sub = bytes[2] & 0xF;
                    let (a, b) = (self.r[hi], self.r[lo]);
                    self.r[hi] = match sub {
                        0..=2 | 4..=6 => self.unsized_op(sub, a, b)?,
                        9..=0xB => Self::bit(sub, a, b),
                        _ => return unknown("fd operation"),
                    };
                }
                0xFE => {
                    length = 3;
                    need!(3);
                    match bytes[2] & 0xF {
                        0 => self.set_special(bus, lo, self.r[hi])?,
                        1 => self.r[lo] = self.special(hi)?,
                        2 => self.r[lo] = bus.tlb(TlbCommand::Physical(self.r[hi] & 0xFF_FFFF)),
                        3 => self.r[lo] = bus.tlb(TlbCommand::Virtual(self.r[hi] & 0xFF_FFFF)),
                        0xC => {
                            let result = self.flags >> (self.r[hi] & 0x1F) & 1;
                            self.set_flag(FLAG_S, false);
                            self.set_flag(FLAG_Z, result == 0);
                            self.r[lo] = result;
                        }
                        _ => return unknown("fe operation"),
                    }
                }
                0xFF => {
                    length = 3;
                    need!(3);
                    let third = (bytes[2] >> 4) as usize;
                    let (a, b) = (self.r[hi], self.r[lo]);
                    self.r[third] = match bytes[2] & 0xF {
                        0xE => return refuse("iords: its difference from iord is not documented"),
                        0xF => bus
                            .io_read(a.wrapping_add(b.wrapping_mul(4)))
                            .map_err(Stop::Refused)?,
                        sub @ (0..=8 | 0xC | 0xD) => self.unsized_op(sub, a, b)?,
                        _ => return unknown("ff operation"),
                    };
                }
                _ => return unknown("unsized operation"),
            }
        }
        self.pc = next.unwrap_or(self.pc.wrapping_add(length as u32));
        Ok(Step::Ran)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Flat code mapped one to one, 4 KiB of data, a few IO registers.
    struct Bench {
        code: Vec<u8>,
        data: Vec<u8>,
        io: std::collections::HashMap<u32, u32>,
        pending: [bool; 2],
    }

    impl Bench {
        fn new(code: &[u8]) -> Self {
            Self {
                code: code.to_vec(),
                data: vec![0; 0x1000],
                io: Default::default(),
                pending: [false; 2],
            }
        }

        /// Runs until the processor stops, for at most `limit` steps.
        fn run(&mut self, core: &mut Falcon, limit: usize) -> Step {
            for _ in 0..limit {
                match core.step(self) {
                    Step::Ran => {}
                    other => return other,
                }
            }
            Step::Ran
        }
    }

    impl Bus for Bench {
        fn fetch(&mut self, address: u32) -> Fetch {
            self.code
                .get(address as usize)
                .map_or(Fetch::NoHit, |&byte| Fetch::Byte(byte))
        }
        fn data(&self) -> &[u8] {
            &self.data
        }
        fn store(&mut self, address: u32, bytes: &[u8]) {
            let at = address as usize;
            self.data[at..at + bytes.len()].copy_from_slice(bytes);
        }
        fn io_read(&mut self, address: u32) -> Result<u32, String> {
            self.io
                .get(&address)
                .copied()
                .ok_or_else(|| format!("no register at {address:#x}"))
        }
        fn io_write(&mut self, address: u32, value: u32) -> Result<(), String> {
            self.io.insert(address, value);
            Ok(())
        }
        fn tlb(&mut self, _command: TlbCommand) -> u32 {
            0
        }
        fn interrupts(&mut self) -> [bool; 2] {
            self.pending
        }
    }

    /// A core whose stack starts at the top of the data memory, which is
    /// where `$sp` 0 puts the first push.
    fn started() -> Falcon {
        Falcon::default()
    }

    #[test]
    fn immediates_memory_and_io_as_the_encodings_say() {
        // mov $r1 0x2000; mov $r2 0xdead0004; st b32 D[$r1] $r2;
        // ld b8 $r3 D[$r1+2]; and $r3 $r3 0xff; mov $r9 0x1000;
        // iowr I[$r9] $r3; iord $r4 I[$r9]; exit
        let code = [
            0x41, 0x00, 0x02, 0xD2, 0x04, 0x00, 0xAD, 0xDE, 0xA0, 0x12, 0x18, 0x13, 0x02, 0xC4,
            0x33, 0xFF, 0x49, 0x00, 0x10, 0xF6, 0x93, 0x00, 0xCF, 0x94, 0x00, 0xF8, 0x02,
        ];
        let mut bench = Bench::new(&code);
        let mut core = started();
        assert_eq!(bench.run(&mut core, 100), Step::Exit);
        assert_eq!(bench.data[0x200..0x204], [0x04, 0x00, 0xAD, 0xDE]);
        assert_eq!(core.r[3], 0xAD);
        assert_eq!(bench.io[&0x1000], 0xAD);
        assert_eq!(core.r[4], 0xAD);
        assert_eq!(core.pc, code.len() as u32);
    }

    #[test]
    fn calls_branches_and_the_stack() {
        // 0: call 0x8; 3: mov $r0 0x7f (never); 5: exit (never, 2 bytes)
        // 7: (pad) 8: push $r5; pop $r6; cmp b32 $r6 $r5; bra e +5; exit;
        // 0x13: mov $r7 1; ret
        let code = [
            0xF3, 0x08, 0x00, // call 8
            0x00, 0x7F, // mov $r0 0x7f
            0xF8, 0x02, // exit
            0x00, // pad
            0xF9, 0x50, // push $r5
            0xFC, 0x60, // pop $r6
            0xA6, 0x65, // cmp b32 $r6 $r5
            0xF4, 0x0B, 0x05, // bra e +5 (to 0x13)
            0xF8, 0x02, // exit
            0x07, 0x01, // mov $r7 1
            0xF8, 0x00, // ret
        ];
        let mut bench = Bench::new(&code);
        let mut core = started();
        core.r[5] = 0x1234_5678;
        assert_eq!(bench.run(&mut core, 100), Step::Exit);
        assert_eq!(core.r[6], 0x1234_5678);
        assert_eq!(core.r[7], 1, "the branch on equal was taken");
        assert_eq!(core.r[0], 0x7F, "the call returned to the next instruction");
        assert_eq!(core.sp, 0, "the top of the data memory");
    }

    #[test]
    fn flags_of_arithmetic_compares_and_shifts() {
        let mut core = started();
        // 0x7f + 1 as bytes: signed overflow, no carry.
        assert_eq!(core.arithmetic(0, 8, 0x7F, 1).ok(), Some(0x80));
        assert_eq!(core.flags & 0xF00, FLAG_O | FLAG_S);
        // 0xff + 1: carry and zero.
        assert_eq!(core.arithmetic(0, 8, 0xFF, 1).ok(), Some(0));
        assert_eq!(core.flags & 0xF00, FLAG_C | FLAG_Z);
        // 0 - 1: borrow.
        assert_eq!(core.arithmetic(2, 32, 0, 1).ok(), Some(!0));
        assert_eq!(core.flags & 0xF00, FLAG_C | FLAG_S);
        // Shifts keep the last bit out; a count of zero clears carry.
        assert_eq!(core.arithmetic(5, 32, 0b101, 1).ok(), Some(0b10));
        assert!(core.flag(FLAG_C));
        assert_eq!(core.arithmetic(4, 32, 1, 0).ok(), Some(1));
        assert!(!core.flag(FLAG_C));
        assert_eq!(core.arithmetic(7, 8, 0x80, 3).ok(), Some(0xF0));
        // Signed compare through the unsigned conditions.
        core.compare(5, 32, !0, 1).ok().unwrap();
        assert!(core.flag(FLAG_C), "-1 is below 1 when compared signed");
        core.compare(4, 32, !0, 1).ok().unwrap();
        assert!(!core.flag(FLAG_C));
        // Sized writes keep the rest of the register.
        core.r[1] = 0xAABB_CCDD;
        core.write(1, 8, 0x11);
        assert_eq!(core.r[1], 0xAABB_CC11);
    }

    #[test]
    fn interrupts_traps_and_what_is_refused() {
        // 0: bset $flags ie0; 3: mov $r1 1; 5: exit; 8: handler: mov $r2 2; iret
        let code = [
            0xF4, 0x31, 0x10, // bset $flags 16
            0x01, 0x01, // mov $r1 1
            0xF8, 0x02, // exit
            0x00, // pad
            0x02, 0x02, // mov $r2 2
            0xF8, 0x01, // iret
        ];
        let mut bench = Bench::new(&code);
        let mut core = started();
        core.iv0 = 8;
        assert_eq!(core.step(&mut bench), Step::Ran);
        bench.pending[0] = true;
        assert_eq!(core.step(&mut bench), Step::Ran, "the interrupt is entered");
        assert_eq!(core.pc, 8);
        assert!(!core.flag(FLAG_IE0) && core.flag(FLAG_IS0));
        bench.pending[0] = false;
        assert_eq!(bench.run(&mut core, 10), Step::Exit);
        assert_eq!((core.r[1], core.r[2]), (1, 2));
        assert!(core.flag(FLAG_IE0), "iret restores the enable");

        // A fetch outside the mapped code traps to $tv; inside the
        // handler a second trap stops the processor.
        let mut bench = Bench::new(&[0xF8, 0x08]); // trap 0
        let mut core = started();
        core.tv = 0x100;
        assert_eq!(core.step(&mut bench), Step::Ran);
        assert_eq!((core.pc, core.tstatus), (0x100, 2));
        assert_eq!(
            bench.data[0xFFC..0x1000],
            [2, 0, 0, 0],
            "the return is after the trap"
        );
        assert_eq!(core.step(&mut bench), Step::Exit, "no code at the vector");

        // The crypto coprocessor is not here.
        let mut bench = Bench::new(&[0xF5, 0x3C, 0x00, 0x80]);
        let mut core = started();
        assert!(matches!(core.step(&mut bench), Step::Refused(_)));
        assert_eq!(core.pc, 0, "a refused instruction has not run");
    }
}
