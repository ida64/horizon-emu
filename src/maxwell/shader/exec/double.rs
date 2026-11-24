//! Double precision: `DADD`, `DMUL`, `DFMA`, `DMNMX`, `DSETP`, `DSET`,
//! and `F2F` with an F64 side. A double is a register pair, the low word
//! in the even register named and the high word in the next; RZ reads
//! as 0.0 and drops writes. Fields \[public: the Maxwell encodings as
//! nvdisasm prints them\]:
//!
//! * `DADD`: A 8-15, B 20-27 (or `c[34-38][20-33 * 4]`, 64 bits), rounding
//!   39-40, -B 45, |A| 46, CC 47, -A 48, |B| 49.
//! * `DMUL`: rounding 39-40, CC 47, -(A * B) 48.
//! * `DFMA`: C 39-46, CC 47, -(A * B) 48, -C 49, rounding 50-51.
//! * `DMNMX`, `DSETP`, `DSET`: as `FMNMX`, `FSETP`, `FSET` without their
//!   flush to zero (doubles keep their subnormals).
//! * `F2F`: destination format 8-9 and source format 10-11 (3: F64),
//!   rounding 39-40, round to an integer 42, FTZ 44 (of an F32 side), -B
//!   45, CC 47, |B| 49, SAT 50.
//!
//! The 20-bit immediate of `I` forms is the top 20 bits of the double,
//! as the 20-bit float immediate is of a float. Only rounding to nearest
//! even is modeled for the arithmetic; the narrowing `F2F` rounds in any
//! direction.
use super::alu::{flush, round_f64, saturate};
use super::*;

/// The double in registers `index`, `index + 1`.
fn pair<const L: usize>(m: &Machine<'_, L>, lane: usize, index: u32) -> f64 {
    if index >= RZ {
        return 0.0;
    }
    f64::from_bits(u64::from(m.r(lane, index)) | u64::from(m.r(lane, index + 1)) << 32)
}

fn modify(v: f64, abs: bool, neg: bool) -> f64 {
    let v = if abs { v.abs() } else { v };
    if neg { -v } else { v }
}

impl<const L: usize> Machine<'_, L> {
    /// Operand B as a double (`F2F` from an F64 included).
    fn b64(&self, lane: usize, form: Form) -> Result<f64, Unmodeled> {
        let w = self.word;
        Ok(match form {
            Form::R => pair(self, lane, bits(w, 20, 8)),
            Form::F | Form::I => f64::from_bits(u64::from(imm20(w) as u32) << 44),
            Form::C => {
                let (slot, offset) = (bits(w, 34, 5), bits(w, 20, 14) * 4);
                let low = self.cbuf(slot, offset)?;
                let high = self.cbuf(slot, offset + 4)?;
                f64::from_bits(u64::from(low) | u64::from(high) << 32)
            }
            Form::Rc => pair(self, lane, bits(w, 39, 8)),
            Form::Imm32 => return self.fail("32-bit immediate"),
        })
    }

    fn set_pair(&mut self, lane: usize, index: u32, value: f64) {
        let bits = value.to_bits();
        self.set(lane, index, bits as u32);
        self.set(lane, index + 1, (bits >> 32) as u32);
    }

    /// The instructions of the module documentation; false for others.
    pub(super) fn execute_double(&mut self, active: Mask) -> Result<bool, Unmodeled> {
        let (name, form) = match self.op {
            Op::DaddR => ("DADD", Form::R),
            Op::DaddI => ("DADD", Form::F),
            Op::DaddC => ("DADD", Form::C),
            Op::DmulR => ("DMUL", Form::R),
            Op::DmulI => ("DMUL", Form::F),
            Op::DmulC => ("DMUL", Form::C),
            Op::DfmaR => ("DFMA", Form::R),
            Op::DfmaI => ("DFMA", Form::F),
            Op::DfmaC => ("DFMA", Form::C),
            Op::DfmaRc => ("DFMA", Form::Rc),
            Op::DmnmxR => ("DMNMX", Form::R),
            Op::DmnmxI => ("DMNMX", Form::F),
            Op::DmnmxC => ("DMNMX", Form::C),
            Op::DsetpR => ("DSETP", Form::R),
            Op::DsetpI => ("DSETP", Form::F),
            Op::DsetpC => ("DSETP", Form::C),
            Op::DsetR => ("DSET", Form::R),
            Op::DsetI => ("DSET", Form::F),
            Op::DsetC => ("DSET", Form::C),
            Op::F2fR | Op::F2fI | Op::F2fC if bits(self.word, 8, 2) == 3 || bits(self.word, 10, 2) == 3 => {
                let form = match self.op {
                    Op::F2fR => Form::R,
                    Op::F2fI => Form::F,
                    _ => Form::C,
                };
                return self.f2f_double(active, form);
            }
            _ => return Ok(false),
        };
        let w = self.word;
        let rd = bits(w, 0, 8);
        let ra = bits(w, 8, 8);
        match name {
            "DADD" | "DMUL" => self.require(bits(w, 39, 2) == 0, "rounding mode")?,
            "DFMA" => self.require(bits(w, 50, 2) == 0, "rounding mode")?,
            "DSETP" | "DSET" => self.require(bits(w, 45, 2) < 3, "boolean operation")?,
            _ => {}
        }
        if !matches!(name, "DSETP" | "DSET") {
            self.require(rd >= RZ || rd % 2 == 0, "odd destination register pair")?;
        }
        self.each(active, |m, l| {
            let a = pair(m, l, ra);
            let b = m.b64(l, form)?;
            let value = match name {
                "DADD" => modify(a, bit(w, 46), bit(w, 48)) + modify(b, bit(w, 49), bit(w, 45)),
                "DMUL" => modify(a, false, bit(w, 48)) * b,
                "DFMA" => {
                    let c = match form {
                        Form::Rc => {
                            let (slot, offset) = (bits(w, 34, 5), bits(w, 20, 14) * 4);
                            let low = m.cbuf(slot, offset)?;
                            let high = m.cbuf(slot, offset + 4)?;
                            f64::from_bits(u64::from(low) | u64::from(high) << 32)
                        }
                        _ => pair(m, l, bits(w, 39, 8)),
                    };
                    modify(a, false, bit(w, 48)).mul_add(b, modify(c, false, bit(w, 49)))
                }
                "DMNMX" => {
                    let x = modify(a, bit(w, 46), bit(w, 48));
                    let y = modify(b, bit(w, 49), bit(w, 45));
                    if m.src_pred(l, 39) { x.min(y) } else { x.max(y) }
                }
                _ => {
                    let predicate = name == "DSETP";
                    let x = modify(a, bit(w, if predicate { 7 } else { 54 }), bit(w, 43));
                    let y = modify(b, bit(w, 44), bit(w, if predicate { 6 } else { 53 }));
                    let test = double_compare(x, y, bits(w, 48, 4));
                    let p = m.src_pred(l, 39);
                    let bop = bits(w, 45, 2);
                    let result = boolean(test, p, bop);
                    if predicate {
                        m.set_pred(l, bits(w, 3, 3), result);
                        m.set_pred(l, bits(w, 0, 3), boolean(!test, p, bop));
                        return Ok(());
                    }
                    let v = match (result, bit(w, 52)) {
                        (false, _) => 0,
                        (true, true) => 1f32.to_bits(),
                        (true, false) => u32::MAX,
                    };
                    m.set(l, rd, v);
                    if bit(w, 47) {
                        m.set_cc(l, v == 0, v >> 31 != 0);
                    }
                    return Ok(());
                }
            };
            m.set_pair(l, rd, value);
            if bit(w, 47) {
                m.set_cc(l, value == 0.0, value < 0.0);
            }
            Ok(())
        })?;
        Ok(true)
    }

    /// `F2F` with an F64 source or destination.
    fn f2f_double(&mut self, active: Mask, form: Form) -> Result<bool, Unmodeled> {
        let w = self.word;
        let (destination, source) = (bits(w, 8, 2), bits(w, 10, 2));
        self.require(matches!(destination, 2 | 3) && matches!(source, 2 | 3), "half precision side")?;
        self.require(!bit(w, 41), "high half")?;
        let rd = bits(w, 0, 8);
        if destination == 3 {
            self.require(rd >= RZ || rd % 2 == 0, "odd destination register pair")?;
        }
        let round = bits(w, 39, 2);
        let integral = bit(w, 42);
        let ftz = bit(w, 44);
        self.each(active, |m, l| {
            let x = if source == 3 {
                m.b64(l, form)?
            } else {
                let b = match form {
                    Form::F => (imm20(w) as u32) << 12,
                    _ => m.b(l, form)?,
                };
                f64::from(flush(f32::from_bits(b), ftz))
            };
            let x = modify(x, bit(w, 49), bit(w, 45));
            let x = if integral {
                match round {
                    0 => x.round_ties_even(),
                    1 => x.floor(),
                    2 => x.ceil(),
                    _ => x.trunc(),
                }
            } else {
                x
            };
            let (zero, negative) = if destination == 3 {
                let v = if bit(w, 50) { x.clamp(0.0, 1.0) } else { x };
                let v = if v.is_nan() && bit(w, 50) { 0.0 } else { v };
                m.set_pair(l, rd, v);
                (v == 0.0, v < 0.0)
            } else {
                // The integral value is exact in an F64; otherwise the
                // narrowing rounds in direction `round`.
                let v = flush(round_f64(x, if integral { 0 } else { round }), ftz);
                let v = if bit(w, 50) { saturate(v) } else { v };
                m.set(l, rd, v.to_bits());
                (v == 0.0, v < 0.0)
            };
            if bit(w, 47) {
                m.set_cc(l, zero, negative);
            }
            Ok(())
        })?;
        Ok(true)
    }

    /// The condition code's zero and sign flags, carry and overflow kept.
    fn set_cc(&mut self, lane: usize, zero: bool, negative: bool) {
        self.lanes[lane].cc =
            (self.lanes[lane].cc & (CC_C | CC_O)) | (u8::from(zero) * CC_Z) | (u8::from(negative) * CC_S);
    }
}

/// `alu::compare` on doubles.
fn double_compare(a: f64, b: f64, c: u32) -> bool {
    let nan = a.is_nan() || b.is_nan();
    match c {
        0 => false,
        1 => a < b,
        2 => a == b,
        3 => a <= b,
        4 => a > b,
        5 => !nan && a != b,
        6 => a >= b,
        7 => !nan,
        8 => nan,
        9 => nan || a < b,
        10 => nan || a == b,
        11 => nan || a <= b,
        12 => nan || a > b,
        13 => a != b,
        14 => nan || a >= b,
        _ => true,
    }
}

fn boolean(a: bool, b: bool, op: u32) -> bool {
    match op {
        0 => a && b,
        1 => a || b,
        _ => a ^ b,
    }
}
