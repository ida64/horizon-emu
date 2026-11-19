//! Thumb state: 16-bit instructions (ARM ARM chapters A6 "The Thumb
//! Instruction Set" and A7 "Thumb Instructions"; the format numbers are
//! those of the ARM7TDMI TRM's Thumb instruction set chapter).

use super::{
    Arm7, BpmpBus, Exception, Exec, PSR_C, PSR_T, PSR_V, Shift, Stop, add_with_carry,
    shift_immediate, shift_register,
};

impl Arm7 {
    /// Register `n` as a Thumb instruction's operand: R15 reads as the
    /// instruction's address + 4.
    fn thumb_operand(&self, n: usize) -> u32 {
        if n == 15 {
            self.r[15].wrapping_add(2)
        } else {
            self.r[n]
        }
    }

    fn set_nzcv(&mut self, (result, carry, overflow): (u32, bool, bool)) -> u32 {
        self.set_nz(result);
        self.set_flag(PSR_C, carry);
        self.set_flag(PSR_V, overflow);
        result
    }

    pub(super) fn execute_thumb(&mut self, bus: &mut dyn BpmpBus, insn: u16) -> Exec {
        let insn = u32::from(insn);
        let low = |shift: u32| ((insn >> shift) & 7) as usize;
        match insn >> 11 {
            // Format 1: LSL, LSR, ASR by immediate (A7, "LSL (1)", "LSR
            // (1)", "ASR (1)"); N, Z and C from the shifter.
            0b00000..=0b00010 => {
                let kind = Shift::from_bits(insn >> 11);
                let (result, carry) =
                    shift_immediate(kind, self.r[low(3)], (insn >> 6) & 31, self.flag(PSR_C));
                self.r[low(0)] = result;
                self.set_nz(result);
                self.set_flag(PSR_C, carry);
            }
            // Format 2: ADD, SUB with a register or a 3-bit immediate.
            0b00011 => {
                let a = self.r[low(3)];
                let b = if insn & (1 << 10) != 0 {
                    (insn >> 6) & 7
                } else {
                    self.r[low(6)]
                };
                let sum = if insn & (1 << 9) != 0 {
                    add_with_carry(a, !b, true)
                } else {
                    add_with_carry(a, b, false)
                };
                self.r[low(0)] = self.set_nzcv(sum);
            }
            // Format 3: MOV, CMP, ADD, SUB with an 8-bit immediate.
            0b00100..=0b00111 => {
                let rd = low(8);
                let imm = insn & 0xFF;
                let a = self.r[rd];
                match (insn >> 11) & 3 {
                    0 => {
                        self.r[rd] = imm;
                        self.set_nz(imm);
                    }
                    1 => {
                        self.set_nzcv(add_with_carry(a, !imm, true));
                    }
                    2 => self.r[rd] = self.set_nzcv(add_with_carry(a, imm, false)),
                    _ => self.r[rd] = self.set_nzcv(add_with_carry(a, !imm, true)),
                }
            }
            0b01000 if insn & (1 << 10) == 0 => self.thumb_alu(insn),
            0b01000 => return self.thumb_high_register(insn),
            // Format 6: LDR Rd, [PC, #imm8 * 4], PC word-aligned (A7, "LDR (3)").
            0b01001 => {
                let address = (self.thumb_operand(15) & !3).wrapping_add((insn & 0xFF) * 4);
                self.r[low(8)] = self.load_word(bus, address)?;
            }
            // Formats 7 and 8: register offset.
            0b01010 | 0b01011 => {
                let address = self.r[low(3)].wrapping_add(self.r[low(6)]);
                let rd = low(0);
                match (insn >> 9) & 7 {
                    0b000 => self.store_word(bus, address, self.r[rd])?,
                    0b010 => self.store_byte(bus, address, self.r[rd])?,
                    0b100 => self.r[rd] = self.load_word(bus, address)?,
                    0b110 => self.r[rd] = self.load_byte(bus, address)?,
                    0b001 => self.store_half(bus, address, self.r[rd])?,
                    0b011 => {
                        self.r[rd] = self.load_byte(bus, address)? as u8 as i8 as u32;
                    }
                    0b101 => self.r[rd] = self.load_half(bus, address)?,
                    _ => {
                        self.r[rd] = self.load_half(bus, address)? as u16 as i16 as u32;
                    }
                }
            }
            // Format 9: word and byte with a 5-bit immediate offset.
            0b01100..=0b01111 => {
                let byte = insn & (1 << 12) != 0;
                let offset = (insn >> 6) & 31;
                let base = self.r[low(3)];
                let rd = low(0);
                let address = base.wrapping_add(if byte { offset } else { offset * 4 });
                match (byte, insn & (1 << 11) != 0) {
                    (false, false) => self.store_word(bus, address, self.r[rd])?,
                    (false, true) => self.r[rd] = self.load_word(bus, address)?,
                    (true, false) => self.store_byte(bus, address, self.r[rd])?,
                    (true, true) => self.r[rd] = self.load_byte(bus, address)?,
                }
            }
            // Format 10: halfword with a 5-bit immediate offset * 2.
            0b10000 | 0b10001 => {
                let address = self.r[low(3)].wrapping_add(((insn >> 6) & 31) * 2);
                let rd = low(0);
                if insn & (1 << 11) != 0 {
                    self.r[rd] = self.load_half(bus, address)?;
                } else {
                    self.store_half(bus, address, self.r[rd])?;
                }
            }
            // Format 11: SP-relative word.
            0b10010 | 0b10011 => {
                let address = self.r[13].wrapping_add((insn & 0xFF) * 4);
                let rd = low(8);
                if insn & (1 << 11) != 0 {
                    self.r[rd] = self.load_word(bus, address)?;
                } else {
                    self.store_word(bus, address, self.r[rd])?;
                }
            }
            // Format 12: ADD Rd, PC or SP, #imm8 * 4 (A7, "ADD (5)", "ADD (6)").
            0b10100 | 0b10101 => {
                let base = if insn & (1 << 11) != 0 {
                    self.r[13]
                } else {
                    self.thumb_operand(15) & !3
                };
                self.r[low(8)] = base.wrapping_add((insn & 0xFF) * 4);
            }
            0b10110 | 0b10111 => return self.thumb_misc(bus, insn),
            // Format 15: LDMIA, STMIA Rb!, {list} (A7, "LDMIA", "STMIA").
            0b11000 | 0b11001 => {
                let rb = low(8);
                let list = (insn & 0xFF) as u16;
                if list == 0 {
                    return Err(Stop::Unpredictable("Thumb LDMIA/STMIA of no registers"));
                }
                let base = self.r[rb];
                let new_base = base.wrapping_add(list.count_ones() * 4);
                if insn & (1 << 11) != 0 {
                    self.load_multiple(bus, list, base, rb, Some(new_base), false, None)?;
                } else {
                    self.store_multiple(bus, list, base, rb, Some(new_base), false, 0)?;
                }
            }
            // Format 16 and 17: conditional branch, SWI (A7, "B (1)", "SWI").
            0b11010 | 0b11011 => {
                let cond = (insn >> 8) & 15;
                match cond {
                    0xE => return Err(Stop::Exception(Exception::Undefined)),
                    0xF => return Err(Stop::Exception(Exception::SoftwareInterrupt)),
                    _ => {
                        if self.condition(cond)? {
                            let offset = ((insn << 24) as i32 >> 23) as u32;
                            self.r[15] = self.thumb_operand(15).wrapping_add(offset);
                        }
                    }
                }
            }
            // Format 18: B (A7, "B (2)").
            0b11100 => {
                let offset = ((insn << 21) as i32 >> 20) as u32;
                self.r[15] = self.thumb_operand(15).wrapping_add(offset);
            }
            // Format 19: BL as two instructions (A7, "BL"). The first puts
            // the high part of the offset in LR; the second branches to LR
            // plus the low part and leaves the return address, bit 0 set,
            // in LR. Each is an instruction of its own: an interrupt may
            // come between them.
            0b11110 => {
                let high = ((insn << 21) as i32 >> 9) as u32;
                self.r[14] = self.thumb_operand(15).wrapping_add(high);
            }
            0b11111 => {
                let next = self.r[15];
                self.r[15] = self.r[14].wrapping_add((insn & 0x7FF) << 1) & !1;
                self.r[14] = next | 1;
            }
            // 0b11101: the BLX suffix of ARMv5T; undefined on ARMv4T.
            _ => return Err(Stop::Exception(Exception::Undefined)),
        }
        Ok(())
    }

    /// Format 4: ALU operations on low registers (A7, per mnemonic).
    /// Shifts by register set C from the shifter; the logical operations
    /// leave C and V; MUL leaves C (UNPREDICTABLE on ARMv4T, kept as for
    /// ARM MULS) and V.
    fn thumb_alu(&mut self, insn: u32) {
        let rd = (insn & 7) as usize;
        let rm = ((insn >> 3) & 7) as usize;
        let (a, b) = (self.r[rd], self.r[rm]);
        let carry = self.flag(PSR_C);
        let shift = |this: &mut Self, kind| {
            let (result, c) = shift_register(kind, a, b, carry);
            this.set_flag(PSR_C, c);
            Some(result)
        };
        let result = match (insn >> 6) & 15 {
            0x0 => Some(a & b),
            0x1 => Some(a ^ b),
            0x2 => shift(self, Shift::Lsl),
            0x3 => shift(self, Shift::Lsr),
            0x4 => shift(self, Shift::Asr),
            0x5 => {
                self.r[rd] = self.set_nzcv(add_with_carry(a, b, carry));
                return;
            }
            0x6 => {
                self.r[rd] = self.set_nzcv(add_with_carry(a, !b, carry));
                return;
            }
            0x7 => shift(self, Shift::Ror),
            0x8 => {
                self.set_nz(a & b);
                None
            }
            0x9 => {
                self.r[rd] = self.set_nzcv(add_with_carry(0, !b, true));
                return;
            }
            0xA => {
                self.set_nzcv(add_with_carry(a, !b, true));
                return;
            }
            0xB => {
                self.set_nzcv(add_with_carry(a, b, false));
                return;
            }
            0xC => Some(a | b),
            0xD => Some(a.wrapping_mul(b)),
            0xE => Some(a & !b),
            _ => Some(!b),
        };
        if let Some(result) = result {
            self.r[rd] = result;
            self.set_nz(result);
        }
    }

    /// Format 5: ADD, CMP, MOV with a high register, and BX (A7, "ADD
    /// (4)", "CMP (3)", "MOV (3)", "BX"). ADD and MOV set no flags; writing PC
    /// clears bit 0. With neither register high the first three are
    /// UNPREDICTABLE on ARMv4T, as is BX with H1 set.
    fn thumb_high_register(&mut self, insn: u32) -> Exec {
        let rd = (insn & 7 | (insn >> 4) & 8) as usize;
        let rm = ((insn >> 3) & 15) as usize;
        let op = (insn >> 8) & 3;
        if op != 3 && insn & 0xC0 == 0 {
            return Err(Stop::Unpredictable(
                "Thumb high-register operation on low registers",
            ));
        }
        let value = self.thumb_operand(rm);
        match op {
            0 | 2 => {
                let result = if op == 0 {
                    self.thumb_operand(rd).wrapping_add(value)
                } else {
                    value
                };
                if rd == 15 {
                    self.r[15] = result & !1;
                } else {
                    self.r[rd] = result;
                }
            }
            1 => {
                self.set_nzcv(add_with_carry(self.thumb_operand(rd), !value, true));
            }
            _ => {
                if insn & 0x80 != 0 {
                    return Err(Stop::Unpredictable("Thumb BX with H1 set"));
                }
                if value & 1 != 0 {
                    self.r[15] = value & !1;
                } else if value & 2 != 0 {
                    return Err(Stop::Unpredictable("BX to an unaligned ARM address"));
                } else {
                    self.cpsr &= !PSR_T;
                    self.r[15] = value;
                }
            }
        }
        Ok(())
    }

    /// Formats 13 and 14: adjust SP, PUSH, POP (A7, "ADD (7)", "SUB (4)",
    /// "PUSH", "POP"). POP {PC} stays in Thumb state on ARMv4T. The rest of the
    /// space is undefined on ARMv4T (BKPT is ARMv5T).
    fn thumb_misc(&mut self, bus: &mut dyn BpmpBus, insn: u32) -> Exec {
        if insn & 0xFF00 == 0xB000 {
            let offset = (insn & 0x7F) * 4;
            self.r[13] = if insn & 0x80 != 0 {
                self.r[13].wrapping_sub(offset)
            } else {
                self.r[13].wrapping_add(offset)
            };
            return Ok(());
        }
        if insn & 0xF600 != 0xB400 {
            return Err(Stop::Exception(Exception::Undefined));
        }
        let pop = insn & (1 << 11) != 0;
        let extra = if insn & 0x100 != 0 {
            if pop { 0x8000 } else { 0x4000 }
        } else {
            0
        };
        let list = (insn & 0xFF) as u16 | extra;
        if list == 0 {
            return Err(Stop::Unpredictable("PUSH/POP of no registers"));
        }
        let size = list.count_ones() * 4;
        let sp = self.r[13];
        if pop {
            self.load_multiple(bus, list, sp, 13, Some(sp.wrapping_add(size)), false, None)
        } else {
            let start = sp.wrapping_sub(size);
            self.store_multiple(bus, list, start, 13, Some(start), false, 0)
        }
    }
}
