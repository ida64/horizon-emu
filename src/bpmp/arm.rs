//! ARM state: 32-bit instructions (ARM ARM chapters A3 "The ARM
//! Instruction Set", A4 "ARM Instructions", A5 "ARM Addressing Modes").

use super::{
    Arm7, BpmpBus, Exception, Exec, Mode, PSR_C, PSR_DEFINED, PSR_N, PSR_T, PSR_V, PSR_Z, Shift,
    Stop, add_with_carry, shift_immediate, shift_register,
};

impl Arm7 {
    /// Register `n` as an ARM instruction's operand: R15 reads as the
    /// instruction's address + 8.
    fn operand(&self, n: usize) -> u32 {
        if n == 15 {
            self.r[15].wrapping_add(4)
        } else {
            self.r[n]
        }
    }

    pub(super) fn execute_arm(&mut self, bus: &mut dyn BpmpBus, insn: u32) -> Exec {
        if !self.condition(insn >> 28)? {
            return Ok(());
        }
        // ARM ARM A3, "Instruction set encoding".
        match (insn >> 25) & 7 {
            0b000 => {
                if insn & 0x0FFF_FFF0 == 0x012F_FF10 {
                    self.branch_exchange(insn)
                } else if insn & 0x90 == 0x90 {
                    self.multiply_or_extra(bus, insn)
                } else if insn & 0x0190_0000 == 0x0100_0000 {
                    self.status_register(insn)
                } else {
                    self.data_processing(insn)
                }
            }
            0b001 => {
                if insn & 0x0190_0000 == 0x0100_0000 {
                    self.status_register(insn)
                } else {
                    self.data_processing(insn)
                }
            }
            // Bit 4 set with a register offset is the architecturally
            // undefined space (A3, "Instruction set encoding").
            0b011 if insn & 0x10 != 0 => Err(Stop::Exception(Exception::Undefined)),
            0b010 | 0b011 => self.single_transfer(bus, insn),
            0b100 => self.block_transfer(bus, insn),
            0b101 => {
                // B, BL (A4, "B, BL").
                if insn & (1 << 24) != 0 {
                    self.r[14] = self.r[15];
                }
                let offset = (((insn << 8) as i32) >> 6) as u32;
                self.r[15] = self.operand(15).wrapping_add(offset);
                Ok(())
            }
            // SWI (A4, "SWI").
            0b111 if insn & (1 << 24) != 0 => Err(Stop::Exception(Exception::SoftwareInterrupt)),
            // LDC, STC, CDP, MCR, MRC: the BPMP has no coprocessor, so none
            // answers and the core takes the undefined instruction trap
            // (A2, "Undefined Instruction exception").
            _ => Err(Stop::Exception(Exception::Undefined)),
        }
    }

    /// BX (A4, "BX"): bit 0 of Rm selects Thumb state.
    fn branch_exchange(&mut self, insn: u32) -> Exec {
        let target = self.operand((insn & 15) as usize);
        if target & 1 != 0 {
            self.cpsr |= PSR_T;
            self.r[15] = target & !1;
        } else if target & 2 != 0 {
            return Err(Stop::Unpredictable("BX to an unaligned ARM address"));
        } else {
            self.r[15] = target;
        }
        Ok(())
    }

    /// Data processing (A3, "Data-processing instructions", and each
    /// opcode's page in A4; operands A5, "Addressing Mode 1").
    fn data_processing(&mut self, insn: u32) -> Exec {
        let opcode = (insn >> 21) & 15;
        let set_flags = insn & (1 << 20) != 0;
        let rn = ((insn >> 16) & 15) as usize;
        let rd = ((insn >> 12) & 15) as usize;
        let carry = self.flag(PSR_C);
        let (a, b, shifter_carry) = if insn & (1 << 25) != 0 {
            // An 8-bit immediate rotated right by twice a 4-bit
            // amount; C is bit 31 of the result unless the rotation is 0.
            let rotation = ((insn >> 8) & 15) * 2;
            let value = (insn & 0xFF).rotate_right(rotation);
            let c = if rotation == 0 {
                carry
            } else {
                value >> 31 != 0
            };
            (self.operand(rn), value, c)
        } else {
            let rm = (insn & 15) as usize;
            let kind = Shift::from_bits(insn >> 5);
            if insn & 0x10 != 0 {
                // A register-specified shift takes an extra cycle, and R15
                // read as an operand is then the address + 12 (ARM7TDMI TRM,
                // "Data processing", and ARM ARM A5: UNPREDICTABLE in the
                // architecture). Rs supplies only its bottom byte.
                let read = |n: usize| {
                    if n == 15 {
                        self.r[15].wrapping_add(8)
                    } else {
                        self.r[n]
                    }
                };
                let amount = read(((insn >> 8) & 15) as usize);
                let (value, c) = shift_register(kind, read(rm), amount, carry);
                (read(rn), value, c)
            } else {
                let (value, c) = shift_immediate(kind, self.operand(rm), (insn >> 7) & 31, carry);
                (self.operand(rn), value, c)
            }
        };
        // (result, writes Rd, arithmetic C and V)
        let (result, write, arithmetic) = match opcode {
            0x0 => (a & b, true, None),
            0x1 => (a ^ b, true, None),
            0x2 => arith(add_with_carry(a, !b, true), true),
            0x3 => arith(add_with_carry(b, !a, true), true),
            0x4 => arith(add_with_carry(a, b, false), true),
            0x5 => arith(add_with_carry(a, b, carry), true),
            0x6 => arith(add_with_carry(a, !b, carry), true),
            0x7 => arith(add_with_carry(b, !a, carry), true),
            0x8 => (a & b, false, None),
            0x9 => (a ^ b, false, None),
            0xA => arith(add_with_carry(a, !b, true), false),
            0xB => arith(add_with_carry(a, b, false), false),
            0xC => (a | b, true, None),
            0xD => (b, true, None),
            0xE => (a & !b, true, None),
            _ => (!b, true, None),
        };
        if write && rd == 15 {
            // With S, the CPSR is restored from the SPSR: an exception
            // return (A4, the data-processing pages, "Rd is R15 and S").
            if set_flags {
                let spsr = self.return_spsr()?;
                self.write_cpsr(spsr);
            }
            self.write_pc(result);
            return Ok(());
        }
        // Rd of TST/TEQ/CMP/CMN and Rn of MOV/MVN are "should be zero"
        // fields the core does not read.
        if write {
            self.r[rd] = result;
        }
        if set_flags {
            self.set_nz(result);
            match arithmetic {
                Some((c, v)) => {
                    self.set_flag(PSR_C, c);
                    self.set_flag(PSR_V, v);
                }
                None => self.set_flag(PSR_C, shifter_carry),
            }
        }
        Ok(())
    }

    /// MRS, MSR (A4, "MRS", "MSR"); everything else in the space between
    /// the compare instructions without S is undefined on ARMv4T.
    fn status_register(&mut self, insn: u32) -> Exec {
        let spsr = insn & (1 << 22) != 0;
        if insn & 0x0FBF_0FFF == 0x010F_0000 {
            let rd = ((insn >> 12) & 15) as usize;
            if rd == 15 {
                return Err(Stop::Unpredictable("MRS to R15"));
            }
            self.r[rd] = if spsr {
                self.spsr()
                    .ok_or(Stop::Unpredictable("MRS of SPSR in a mode without one"))?
            } else {
                self.cpsr
            };
            return Ok(());
        }
        let operand = if insn & 0x0FB0_FFF0 == 0x0120_F000 {
            let rm = (insn & 15) as usize;
            if rm == 15 {
                return Err(Stop::Unpredictable("MSR from R15"));
            }
            self.r[rm]
        } else if insn & 0x0FB0_F000 == 0x0320_F000 {
            (insn & 0xFF).rotate_right(((insn >> 8) & 15) * 2)
        } else {
            return Err(Stop::Exception(Exception::Undefined));
        };
        // Field mask bits 19:16 select bytes c, x, s, f.
        let mut mask = 0;
        for byte in 0..4 {
            if insn & (1 << (16 + byte)) != 0 {
                mask |= 0xFF << (8 * byte);
            }
        }
        let mode = self.mode();
        if spsr {
            if !mode.has_spsr() {
                return Err(Stop::Unpredictable("MSR to SPSR in a mode without one"));
            }
            let bank = mode.bank();
            self.spsr[bank] = (self.spsr[bank] & !mask | operand & mask) & PSR_DEFINED;
            return Ok(());
        }
        // In User mode only the flags byte can be written.
        if mode == Mode::User {
            mask &= 0xFF00_0000;
        }
        let value = (self.cpsr & !mask | operand & mask) & PSR_DEFINED;
        if (value ^ self.cpsr) & PSR_T != 0 {
            return Err(Stop::Unpredictable("MSR changing the T bit"));
        }
        if Mode::from_bits(value).is_none() {
            return Err(Stop::Unpredictable("MSR writing reserved mode bits"));
        }
        self.write_cpsr(value);
        Ok(())
    }

    /// Multiplies, SWP and the halfword and signed-byte transfers: the
    /// encodings with bits 7 and 4 set (A3, "Multiplies and extra
    /// load/store instructions").
    fn multiply_or_extra(&mut self, bus: &mut dyn BpmpBus, insn: u32) -> Exec {
        if (insn >> 5) & 3 != 0 {
            return self.halfword_transfer(bus, insn);
        }
        let rd = ((insn >> 16) & 15) as usize;
        let rn = ((insn >> 12) & 15) as usize;
        let rs = ((insn >> 8) & 15) as usize;
        let rm = (insn & 15) as usize;
        let set_flags = insn & (1 << 20) != 0;
        if (insn >> 22) & 0x3F == 0 {
            // MUL, MLA (A4, "MUL", "MLA"). C is UNPREDICTABLE after MULS
            // on ARMv4; it keeps its value \[assumption: ARMv5 defines
            // that, and the ARM7TDMI TRM calls the value "meaningless"\].
            // Rd = Rm is fine on the ARM7TDMI's multiplier
            // \[inference: the restriction dates from ARM2\].
            if [rd, rs, rm].contains(&15) || (insn & (1 << 21) != 0 && rn == 15) {
                return Err(Stop::Unpredictable("multiply with R15"));
            }
            let mut result = self.r[rm].wrapping_mul(self.r[rs]);
            if insn & (1 << 21) != 0 {
                result = result.wrapping_add(self.r[rn]);
            }
            self.r[rd] = result;
            if set_flags {
                self.set_nz(result);
            }
            return Ok(());
        }
        if (insn >> 23) & 0x1F == 0b00001 {
            // UMULL, UMLAL, SMULL, SMLAL (A4, per mnemonic): RdHi in bits
            // 19:16, RdLo in 15:12. C and V keep
            // their values, as after MULS.
            let (hi, lo) = (rd, rn);
            if [hi, lo, rs, rm].contains(&15) {
                return Err(Stop::Unpredictable("long multiply with R15"));
            }
            if hi == lo {
                return Err(Stop::Unpredictable("long multiply with RdHi = RdLo"));
            }
            let product = if insn & (1 << 22) != 0 {
                (i64::from(self.r[rm] as i32) * i64::from(self.r[rs] as i32)) as u64
            } else {
                u64::from(self.r[rm]) * u64::from(self.r[rs])
            };
            let accumulate = if insn & (1 << 21) != 0 {
                u64::from(self.r[hi]) << 32 | u64::from(self.r[lo])
            } else {
                0
            };
            let result = product.wrapping_add(accumulate);
            self.r[lo] = result as u32;
            self.r[hi] = (result >> 32) as u32;
            if set_flags {
                self.set_flag(PSR_N, result >> 63 != 0);
                self.set_flag(PSR_Z, result == 0);
            }
            return Ok(());
        }
        if insn & 0x0FB0_0FF0 == 0x0100_0090 {
            // SWP, SWPB (A4, "SWP", "SWPB"): Rn = base (19:16), Rd =
            // destination (15:12), Rm = source (3:0). The load and store
            // are locked together; nothing else runs between them here.
            let (base, dest) = (rd, rn);
            if [base, dest, rm].contains(&15) {
                return Err(Stop::Unpredictable("SWP with R15"));
            }
            let address = self.r[base];
            let source = self.r[rm];
            let loaded = if insn & (1 << 22) != 0 {
                let value = self.load_byte(bus, address)?;
                self.store_byte(bus, address, source)?;
                value
            } else {
                let value = self.load_word(bus, address)?;
                self.store_word(bus, address, source)?;
                value
            };
            self.r[dest] = loaded;
            return Ok(());
        }
        Err(Stop::Exception(Exception::Undefined))
    }

    /// LDRH, STRH, LDRSB, LDRSH (A4, per mnemonic; A5, "Addressing Mode
    /// 3").
    fn halfword_transfer(&mut self, bus: &mut dyn BpmpBus, insn: u32) -> Exec {
        let pre = insn & (1 << 24) != 0;
        let up = insn & (1 << 23) != 0;
        let immediate = insn & (1 << 22) != 0;
        let write_bit = insn & (1 << 21) != 0;
        let load = insn & (1 << 20) != 0;
        let rn = ((insn >> 16) & 15) as usize;
        let rd = ((insn >> 12) & 15) as usize;
        let kind = (insn >> 5) & 3;
        if !load && kind != 1 {
            return Err(Stop::Unpredictable(
                "signed store encoding (LDRD/STRD from ARMv5TE)",
            ));
        }
        if !pre && write_bit {
            return Err(Stop::Unpredictable(
                "post-indexed halfword transfer with W set",
            ));
        }
        if rd == 15 {
            return Err(Stop::Unpredictable("halfword or signed transfer of R15"));
        }
        let offset = if immediate {
            (insn >> 4) & 0xF0 | insn & 0xF
        } else {
            let rm = (insn & 15) as usize;
            if rm == 15 {
                return Err(Stop::Unpredictable("halfword transfer offset in R15"));
            }
            self.r[rm]
        };
        let writeback = !pre || write_bit;
        if writeback && rn == 15 {
            return Err(Stop::Unpredictable("writeback to R15"));
        }
        if writeback && load && rn == rd {
            return Err(Stop::Unpredictable(
                "load with writeback to its destination",
            ));
        }
        let base = self.operand(rn);
        let indexed = if up {
            base.wrapping_add(offset)
        } else {
            base.wrapping_sub(offset)
        };
        let address = if pre { indexed } else { base };
        if load {
            let loaded = match kind {
                1 => self.load_half(bus, address),
                2 => self
                    .load_byte(bus, address)
                    .map(|value| value as u8 as i8 as u32),
                _ => self
                    .load_half(bus, address)
                    .map(|value| value as u16 as i16 as u32),
            };
            self.commit_base_on_abort(&loaded, writeback, rn, indexed);
            self.r[rd] = loaded?;
        } else {
            let stored = self.store_half(bus, address, self.r[rd]);
            self.commit_base_on_abort(&stored, writeback, rn, indexed);
            stored?;
        }
        Ok(())
    }

    /// Writes the base back after a transfer that completed or took a data
    /// abort: the ARM7TDMI's "base updated" abort model (ARM7TDMI TRM,
    /// "Data aborts"). A transfer that faulted changes nothing.
    fn commit_base_on_abort<T>(
        &mut self,
        outcome: &Result<T, Stop>,
        writeback: bool,
        rn: usize,
        value: u32,
    ) {
        if writeback && matches!(outcome, Ok(_) | Err(Stop::Exception(_))) {
            self.r[rn] = value;
        }
    }

    /// LDR, LDRB, STR, STRB and their T forms (A4, per mnemonic; A5,
    /// "Addressing Mode 2"). The T forms (post-indexed with W set)
    /// ask for a User-mode access; the BPMP has no memory protection that
    /// tells the two apart, so they behave as the plain forms.
    fn single_transfer(&mut self, bus: &mut dyn BpmpBus, insn: u32) -> Exec {
        let register = insn & (1 << 25) != 0;
        let pre = insn & (1 << 24) != 0;
        let up = insn & (1 << 23) != 0;
        let byte = insn & (1 << 22) != 0;
        let write_bit = insn & (1 << 21) != 0;
        let load = insn & (1 << 20) != 0;
        let rn = ((insn >> 16) & 15) as usize;
        let rd = ((insn >> 12) & 15) as usize;
        let offset = if register {
            let rm = (insn & 15) as usize;
            if rm == 15 {
                return Err(Stop::Unpredictable("transfer offset in R15"));
            }
            let kind = Shift::from_bits(insn >> 5);
            shift_immediate(kind, self.r[rm], (insn >> 7) & 31, self.flag(PSR_C)).0
        } else {
            insn & 0xFFF
        };
        let writeback = !pre || write_bit;
        if writeback && rn == 15 {
            return Err(Stop::Unpredictable("writeback to R15"));
        }
        if writeback && load && rn == rd {
            return Err(Stop::Unpredictable(
                "load with writeback to its destination",
            ));
        }
        if byte && rd == 15 {
            return Err(Stop::Unpredictable("byte transfer of R15"));
        }
        let base = self.operand(rn);
        let indexed = if up {
            base.wrapping_add(offset)
        } else {
            base.wrapping_sub(offset)
        };
        let address = if pre { indexed } else { base };
        if load {
            let loaded = if byte {
                self.load_byte(bus, address)
            } else {
                self.load_word(bus, address)
            };
            self.commit_base_on_abort(&loaded, writeback, rn, indexed);
            let value = loaded?;
            // On ARMv4T a load to PC does not change state (A4, "LDR").
            if rd == 15 {
                self.write_pc(value);
            } else {
                self.r[rd] = value;
            }
        } else {
            // A stored R15 is the address + 12 on the ARM7TDMI (ARM7TDMI
            // TRM, "Single data transfer"; IMPLEMENTATION DEFINED in
            // A4, "STR"). A stored base that is written back stores its
            // original value: the data is read before the writeback.
            let value = if rd == 15 {
                self.r[15].wrapping_add(8)
            } else {
                self.r[rd]
            };
            let stored = if byte {
                self.store_byte(bus, address, value)
            } else {
                self.store_word(bus, address, value)
            };
            self.commit_base_on_abort(&stored, writeback, rn, indexed);
            stored?;
        }
        Ok(())
    }

    /// LDM, STM in their three forms (A4, "LDM (1)" to "LDM (3)", "STM
    /// (1)", "STM (2)"; A5, "Addressing Mode 4").
    fn block_transfer(&mut self, bus: &mut dyn BpmpBus, insn: u32) -> Exec {
        let pre = insn & (1 << 24) != 0;
        let up = insn & (1 << 23) != 0;
        let caret = insn & (1 << 22) != 0;
        let write_bit = insn & (1 << 21) != 0;
        let load = insn & (1 << 20) != 0;
        let rn = ((insn >> 16) & 15) as usize;
        let list = insn as u16;
        if rn == 15 {
            return Err(Stop::Unpredictable("block transfer based on R15"));
        }
        if list == 0 {
            return Err(Stop::Unpredictable("block transfer of no registers"));
        }
        let loads_pc = load && list & 0x8000 != 0;
        // ^ with PC loaded restores the CPSR (LDM (3)); otherwise it
        // transfers the User bank (LDM (2), STM (2)), where writeback is
        // UNPREDICTABLE. Either is UNPREDICTABLE in User and System mode.
        let user = caret && !loads_pc;
        if caret && !self.mode().has_spsr() {
            return Err(Stop::Unpredictable("LDM/STM with ^ in User or System mode"));
        }
        if user && write_bit {
            return Err(Stop::Unpredictable("User-bank LDM/STM with writeback"));
        }
        let restore = if caret && loads_pc {
            Some(self.return_spsr()?)
        } else {
            None
        };
        let size = list.count_ones() * 4;
        let base = self.r[rn];
        let (start, new_base) = match (pre, up) {
            (false, true) => (base, base.wrapping_add(size)),
            (true, true) => (base.wrapping_add(4), base.wrapping_add(size)),
            (false, false) => (
                base.wrapping_sub(size).wrapping_add(4),
                base.wrapping_sub(size),
            ),
            (true, false) => (base.wrapping_sub(size), base.wrapping_sub(size)),
        };
        let writeback = write_bit.then_some(new_base);
        if load {
            self.load_multiple(bus, list, start, rn, writeback, user, restore)
        } else {
            let pc = self.r[15].wrapping_add(8);
            self.store_multiple(bus, list, start, rn, writeback, user, pc)
        }
    }
}

fn arith(
    (result, carry, overflow): (u32, bool, bool),
    write: bool,
) -> (u32, bool, Option<(bool, bool)>) {
    (result, write, Some((carry, overflow)))
}
