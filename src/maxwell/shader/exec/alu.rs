//! Scalar arithmetic: field layouts and operations. Unsupported rounding
//! and extended modes fail explicitly rather than approximate them.
use super::*;

pub(super) fn saturate(v: f32) -> f32 {
    if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) }
}
pub(super) fn flush(v: f32, enabled: bool) -> f32 {
    if enabled && v.is_subnormal() {
        0.0f32.copysign(v)
    } else {
        v
    }
}
fn modify(v: f32, abs: bool, neg: bool) -> f32 {
    let v = if abs { v.abs() } else { v };
    if neg { -v } else { v }
}
/// A binary32 product is exact in binary64, so directed conversion can
/// compare the rounded result to the exact product without approximation.
fn round_f64(v: f64, mode: u32) -> f32 {
    let r = v as f32;
    if mode == 0 || v.is_nan() {
        return r;
    }
    let down = mode == 1 || (mode == 3 && v > 0.0);
    let up = mode == 2 || (mode == 3 && v < 0.0);
    if down && (r as f64) > v {
        r.next_down()
    } else if up && (r as f64) < v {
        r.next_up()
    } else {
        r
    }
}
pub(super) fn compare(a: f32, b: f32, c: u32) -> bool {
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
fn icmp(a: u32, b: u32, c: u32, signed: bool) -> bool {
    let (a, b) = if signed {
        (a as i32 as i64, b as i32 as i64)
    } else {
        (a as i64, b as i64)
    };
    match c {
        0 => false,
        1 => a < b,
        2 => a == b,
        3 => a <= b,
        4 => a > b,
        5 => a != b,
        6 => a >= b,
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
impl<const L: usize> Machine<'_, L> {
    pub(super) fn execute_alu(&mut self, active: Mask) -> Result<bool, Unmodeled> {
        let (name, form) = match self.op {
            Op::MovR => ("MOV", Form::R),
            Op::MovI => ("MOV", Form::I),
            Op::MovC => ("MOV", Form::C),
            Op::Mov32i => ("MOV32I", Form::Imm32),
            Op::FaddR => ("FADD", Form::R),
            Op::FaddI => ("FADD", Form::F),
            Op::FaddC => ("FADD", Form::C),
            Op::Fadd32i => ("FADD32I", Form::Imm32),
            Op::FmulR => ("FMUL", Form::R),
            Op::FmulI => ("FMUL", Form::F),
            Op::FmulC => ("FMUL", Form::C),
            Op::Fmul32i => ("FMUL32I", Form::Imm32),
            Op::FfmaR => ("FFMA", Form::R),
            Op::FfmaI => ("FFMA", Form::F),
            Op::FfmaC => ("FFMA", Form::C),
            Op::FfmaRc => ("FFMA", Form::Rc),
            Op::Ffma32i => ("FFMA32I", Form::Imm32),
            Op::FmnmxR => ("FMNMX", Form::R),
            Op::FmnmxI => ("FMNMX", Form::F),
            Op::FmnmxC => ("FMNMX", Form::C),
            Op::FsetR => ("FSET", Form::R),
            Op::FsetI => ("FSET", Form::F),
            Op::FsetC => ("FSET", Form::C),
            Op::FsetpR => ("FSETP", Form::R),
            Op::FsetpI => ("FSETP", Form::F),
            Op::FsetpC => ("FSETP", Form::C),
            Op::IsetpR => ("ISETP", Form::R),
            Op::IsetpI => ("ISETP", Form::I),
            Op::IsetpC => ("ISETP", Form::C),
            Op::IsetR => ("ISET", Form::R),
            Op::IsetI => ("ISET", Form::I),
            Op::IsetC => ("ISET", Form::C),
            Op::IaddR => ("IADD", Form::R),
            Op::IaddI => ("IADD", Form::I),
            Op::IaddC => ("IADD", Form::C),
            Op::Iadd32i => ("IADD32I", Form::Imm32),
            Op::LopR => ("LOP", Form::R),
            Op::LopI => ("LOP", Form::I),
            Op::LopC => ("LOP", Form::C),
            Op::Lop32i => ("LOP32I", Form::Imm32),
            Op::ShlR => ("SHL", Form::R),
            Op::ShlI => ("SHL", Form::I),
            Op::ShlC => ("SHL", Form::C),
            Op::ShrR => ("SHR", Form::R),
            Op::ShrI => ("SHR", Form::I),
            Op::ShrC => ("SHR", Form::C),
            Op::ShfLR => ("SHF<", Form::R),
            Op::ShfLI => ("SHF<", Form::I),
            Op::ShfRR => ("SHF>", Form::R),
            Op::ShfRI => ("SHF>", Form::I),
            Op::IscaddR => ("ISCADD", Form::R),
            Op::IscaddI => ("ISCADD", Form::I),
            Op::IscaddC => ("ISCADD", Form::C),
            Op::Iscadd32i => ("ISCADD32I", Form::Imm32),
            Op::SelR => ("SEL", Form::R),
            Op::SelI => ("SEL", Form::I),
            Op::SelC => ("SEL", Form::C),
            Op::F2iR => ("F2I", Form::R),
            Op::F2iI => ("F2I", Form::F),
            Op::F2iC => ("F2I", Form::C),
            Op::I2fR => ("I2F", Form::R),
            Op::I2fI => ("I2F", Form::I),
            Op::I2fC => ("I2F", Form::C),
            Op::I2iR => ("I2I", Form::R),
            Op::I2iI => ("I2I", Form::I),
            Op::I2iC => ("I2I", Form::C),
            Op::F2fR => ("F2F", Form::R),
            Op::F2fI => ("F2F", Form::F),
            Op::F2fC => ("F2F", Form::C),
            Op::RroR => ("RRO", Form::R),
            Op::RroI => ("RRO", Form::F),
            Op::RroC => ("RRO", Form::C),
            Op::Mufu => ("MUFU", Form::R),
            _ => return Ok(false),
        };
        let w = self.word;
        let rd = bits(w, 0, 8);
        let ra = bits(w, 8, 8);
        self.each(active, |m, l| {
            let a = m.r(l, ra);
            let b = m.b(l, form)?;
            let af = f32::from_bits(a);
            let bf = f32::from_bits(b);
            let mut cc = false;
            let value = match name {
                "MOV" | "MOV32I" => {
                    let mask = bits(w, if name == "MOV" { 39 } else { 12 }, 4);
                    m.require(mask == 15, "partial quad mask")?;
                    b
                }
                "FADD" | "FMUL" | "FFMA" | "FMNMX" | "FADD32I" | "FMUL32I" | "FFMA32I" => {
                    let wide = form == Form::Imm32;
                    let mut sat = bit(w, if wide { 55 } else { 50 });
                    cc = bit(w, if wide { 52 } else { 47 });
                    let v = match name {
                        "FADD" | "FADD32I" => {
                            if !wide {
                                m.require(bits(w, 39, 2) == 0, "rounding mode")?;
                            }
                            let ftz = bit(w, if wide { 55 } else { 44 });
                            sat = if wide { false } else { sat };
                            let x = flush(
                                modify(
                                    af,
                                    bit(w, if wide { 54 } else { 46 }),
                                    bit(w, if wide { 56 } else { 48 }),
                                ),
                                ftz,
                            );
                            let y = flush(
                                modify(
                                    bf,
                                    bit(w, if wide { 57 } else { 49 }),
                                    bit(w, if wide { 53 } else { 45 }),
                                ),
                                ftz,
                            );
                            flush(x + y, ftz)
                        }
                        "FMUL" | "FMUL32I" => {
                            let round = if wide { 0 } else { bits(w, 39, 2) };
                            let mode = bits(w, if wide { 53 } else { 44 }, 2);
                            m.require(mode <= 2, "FMZ encoding")?;
                            let scale = if wide { 0 } else { bits(w, 41, 3) };
                            m.require(scale < 7, "scale")?;
                            let x = flush(modify(af, false, !wide && bit(w, 48)), mode == 1);
                            let y = flush(bf, mode == 1);
                            let v = if mode == 2 && (x == 0.0 || y == 0.0) {
                                f64::from(f32::from_bits((x.to_bits() ^ y.to_bits()) & 0x80000000))
                            } else {
                                f64::from(x) * f64::from(y)
                            };
                            flush(
                                round_f64(
                                    v * [1.0, 0.5, 0.25, 0.125, 8.0, 4.0, 2.0][scale as usize],
                                    round,
                                ),
                                mode == 1,
                            )
                        }
                        "FFMA" | "FFMA32I" => {
                            if !wide {
                                m.require(bits(w, 51, 2) == 0, "rounding mode")?;
                            }
                            let mode = bits(w, 53, 2);
                            m.require(mode <= 2, "FMZ encoding")?;
                            let c = if wide {
                                m.rf(l, rd)
                            } else {
                                f32::from_bits(m.c(l, form)?)
                            };
                            let x = flush(
                                modify(af, false, bit(w, if wide { 56 } else { 48 })),
                                mode == 1,
                            );
                            let y = flush(bf, mode == 1);
                            let z = flush(
                                modify(c, false, bit(w, if wide { 57 } else { 49 })),
                                mode == 1,
                            );
                            flush(
                                if mode == 2 && (x == 0.0 || y == 0.0) {
                                    z
                                } else {
                                    x.mul_add(y, z)
                                },
                                mode == 1,
                            )
                        }
                        _ => {
                            sat = false;
                            let ftz = bit(w, 44);
                            let x = flush(modify(af, bit(w, 46), bit(w, 48)), ftz);
                            let y = flush(modify(bf, bit(w, 49), bit(w, 45)), ftz);
                            if m.src_pred(l, 39) {
                                x.min(y)
                            } else {
                                x.max(y)
                            }
                        }
                    };
                    (if sat { saturate(v) } else { v }).to_bits()
                }
                "MUFU" => {
                    let x = modify(af, bit(w, 46), bit(w, 48));
                    let v = match bits(w, 20, 4) {
                        0 => x.cos(),
                        1 => x.sin(),
                        2 => x.exp2(),
                        3 => x.log2(),
                        4 => x.recip(),
                        5 => x.sqrt().recip(),
                        8 => x.sqrt(),
                        v => return m.fail(&format!("suboperation {v}")),
                    };
                    (if bit(w, 50) { saturate(v) } else { v }).to_bits()
                }
                // RRO is a range-reduction hint: CPU transcendental operations perform
                // their own reduction.
                "RRO" => modify(bf, bit(w, 49), bit(w, 45)).to_bits(),
                "FSETP" | "FSET" | "ISETP" | "ISET" => {
                    let fp = name.starts_with('F');
                    let pred = name.ends_with('P');
                    let bop = bits(w, 45, 2);
                    m.require(bop < 3, "boolean operation")?;
                    let test = if fp {
                        let x = modify(af, bit(w, if pred { 7 } else { 54 }), bit(w, 43));
                        let y = modify(bf, bit(w, 44), bit(w, if pred { 6 } else { 53 }));
                        let ftz = bit(w, if pred { 47 } else { 55 });
                        compare(flush(x, ftz), flush(y, ftz), bits(w, 48, 4))
                    } else {
                        m.require(!bit(w, 43), "extended comparison")?;
                        icmp(a, b, bits(w, 49, 3), bit(w, 48))
                    };
                    let p = m.src_pred(l, 39);
                    let result = boolean(test, p, bop);
                    if pred {
                        m.set_pred(l, bits(w, 3, 3), result);
                        m.set_pred(l, bits(w, 0, 3), boolean(!test, p, bop));
                        return Ok(());
                    }
                    cc = bit(w, 47);
                    if result {
                        if bit(w, if fp { 52 } else { 44 }) {
                            1f32.to_bits()
                        } else {
                            u32::MAX
                        }
                    } else {
                        0
                    }
                }
                "IADD" | "IADD32I" | "ISCADD" | "ISCADD32I" => {
                    let wide = form == Form::Imm32;
                    let shift = name.starts_with("ISCADD");
                    let mut a = if shift {
                        a.wrapping_shl(bits(w, if wide { 53 } else { 39 }, 5))
                    } else {
                        a
                    };
                    let mut b = b;
                    let mode = if shift && wide {
                        0
                    } else {
                        bits(w, if wide { 55 } else { 48 }, 2)
                    };
                    let extended = !shift && bit(w, if wide { 53 } else { 43 });
                    let mut carry = if extended {
                        u32::from(m.lanes[l].cc & CC_C != 0)
                    } else {
                        0
                    };
                    match mode {
                        1 => {
                            b = !b;
                            carry += u32::from(!extended)
                        }
                        2 => {
                            a = !a;
                            carry += u32::from(!extended)
                        }
                        3 => carry += 1,
                        _ => {}
                    };
                    m.require(shift || !bit(w, if wide { 54 } else { 50 }), "SAT")?;
                    let sum = u64::from(a) + u64::from(b) + u64::from(carry);
                    let v = sum as u32;
                    let flags = (u8::from(v == 0 && (!extended || m.lanes[l].cc & CC_Z != 0))
                        * CC_Z)
                        | (u8::from(v >> 31 != 0) * CC_S)
                        | (u8::from(sum >> 32 != 0) * CC_C)
                        | (u8::from((!(a ^ b) & (a ^ v)) >> 31 != 0) * CC_O);
                    if bit(w, if wide { 52 } else { 47 }) {
                        m.lanes[l].cc = flags;
                    }
                    v
                }
                "LOP" | "LOP32I" => {
                    let wide = form == Form::Imm32;
                    m.require(!bit(w, if wide { 57 } else { 43 }), "X")?;
                    let a = if bit(w, if wide { 55 } else { 39 }) {
                        !a
                    } else {
                        a
                    };
                    let b = if bit(w, if wide { 56 } else { 40 }) {
                        !b
                    } else {
                        b
                    };
                    let v = match bits(w, if wide { 53 } else { 41 }, 2) {
                        0 => a & b,
                        1 => a | b,
                        2 => a ^ b,
                        _ => b,
                    };
                    if !wide {
                        let p = match bits(w, 44, 2) {
                            0 => false,
                            1 => true,
                            2 => v == 0,
                            _ => v != 0,
                        };
                        m.set_pred(l, bits(w, 48, 3), p);
                    }
                    cc = bit(w, if wide { 52 } else { 47 });
                    v
                }
                "SHL" | "SHR" => {
                    m.require(
                        bits(w, 43, if name == "SHL" { 1 } else { 2 }) == 0,
                        "extended shift",
                    )?;
                    let n = if bit(w, 39) { b & 31 } else { b.min(32) };
                    let a = if name == "SHR" && bit(w, 40) {
                        a.reverse_bits()
                    } else {
                        a
                    };
                    cc = bit(w, 47);
                    if name == "SHL" {
                        a.checked_shl(n).unwrap_or(0)
                    } else if bit(w, 48) {
                        ((a as i32) >> n.min(31)) as u32
                    } else {
                        a.checked_shr(n).unwrap_or(0)
                    }
                }
                // Funnel shift of the 64-bit value C:A (C register 39-46
                // the high word): left gives the high word of the shifted
                // value, right the low word. Shift count B (register
                // 20-27, or the 6-bit immediate at 20), maximum shift
                // bits 38:37 (0 32, 2 64 unsigned, 3 64 signed), wrap bit
                // 50: the count masked to the maximum minus one instead of
                // clamped to the maximum.
                "SHF<" | "SHF>" => {
                    m.require(bits(w, 48, 2) == 0, "extended funnel shift")?;
                    let maximum = bits(w, 37, 2);
                    m.require(maximum != 1, "maximum shift 1")?;
                    let limit = if maximum == 0 { 32 } else { 64 };
                    let count = if form == Form::I { bits(w, 20, 6) } else { b };
                    let n = if bit(w, 50) {
                        count & (limit - 1)
                    } else {
                        count.min(limit)
                    };
                    let high = m.r(l, bits(w, 39, 8));
                    let packed = u64::from(high) << 32 | u64::from(a);
                    cc = bit(w, 47);
                    if name == "SHF<" {
                        (packed.checked_shl(n).unwrap_or(0) >> 32) as u32
                    } else if maximum == 3 {
                        ((packed as i64) >> n.min(63)) as u32
                    } else {
                        packed.checked_shr(n).unwrap_or(0) as u32
                    }
                }
                "SEL" => {
                    if m.src_pred(l, 39) {
                        a
                    } else {
                        b
                    }
                }
                "F2I" | "F2F" => {
                    m.require(bits(w, 10, 2) == 2, "source format")?;
                    m.require(!bit(w, 41), "high half")?;
                    cc = bit(w, 47);
                    let x = flush(modify(bf, bit(w, 49), bit(w, 45)), bit(w, 44));
                    let round = bits(w, 39, 2);
                    let rounded = match round {
                        0 => x.round_ties_even(),
                        1 => x.floor(),
                        2 => x.ceil(),
                        _ => x.trunc(),
                    };
                    if name == "F2F" {
                        m.require(bits(w, 8, 2) == 2, "destination format")?;
                        let v = if bit(w, 42) { rounded } else { x };
                        (if bit(w, 50) { saturate(v) } else { v }).to_bits()
                    } else {
                        match bits(w, 8, 2) | (bits(w, 12, 1) << 2) {
                            1 => rounded as u16 as u32,
                            2 => rounded as u32,
                            5 => rounded as i16 as i32 as u32,
                            6 => rounded as i32 as u32,
                            v => return m.fail(&format!("destination format {v}")),
                        }
                    }
                }
                "I2F" => {
                    m.require(bits(w, 8, 2) == 2, "destination format")?;
                    let fmt = bits(w, 10, 2) | (bits(w, 13, 1) << 2);
                    let selection = bits(w, 41, 2);
                    m.require(
                        (fmt & 3) != 2 || selection == 0,
                        "byte selector with 32-bit source",
                    )?;
                    let value = b >> (selection * 8);
                    let mut x = match fmt {
                        0 => value as u8 as f64,
                        1 => value as u16 as f64,
                        2 => value as f64,
                        4 => value as i8 as f64,
                        5 => value as i16 as f64,
                        6 => value as i32 as f64,
                        v => return m.fail(&format!("source format {v}")),
                    };
                    cc = bit(w, 47);
                    if bit(w, 49) {
                        x = x.abs();
                    }
                    if bit(w, 45) {
                        x = -x;
                    }
                    round_f64(x, bits(w, 39, 2)).to_bits()
                }
                // Integer to integer: source format bits 11:10 with signed
                // bit 13, byte selector 42:41, |b| bit 49, -b bit 45, as
                // `I2F`; destination format bits 9:8 with signed bit 12,
                // saturate bit 50. The source is
                // extended per its format, then negated or made absolute
                // (wrapping, as 32-bit integer arithmetic does), then, with
                // SAT, clamped to the destination's range. Only 32-bit
                // destinations are modeled: how a narrower one is stored
                // (truncated or extended) is not established here.
                "I2I" => {
                    let destination = bits(w, 8, 2) | (bits(w, 12, 1) << 2);
                    m.require(destination & 3 == 2, "destination narrower than 32 bits")?;
                    // The condition code takes zero and sign of the
                    // result.
                    cc = bit(w, 47);
                    let fmt = bits(w, 10, 2) | (bits(w, 13, 1) << 2);
                    let selection = bits(w, 41, 2);
                    m.require(
                        (fmt & 3) != 2 || selection == 0,
                        "byte selector with 32-bit source",
                    )?;
                    let value = b >> (selection * 8);
                    let mut x: i64 = match fmt {
                        0 => value as u8 as i64,
                        1 => value as u16 as i64,
                        2 => value as i64,
                        4 => value as i8 as i64,
                        5 => value as i16 as i64,
                        6 => value as i32 as i64,
                        v => return m.fail(&format!("source format {v}")),
                    };
                    if bit(w, 49) {
                        x = x.abs();
                    }
                    if bit(w, 45) {
                        x = -x;
                    }
                    if bit(w, 50) {
                        x = if destination == 6 {
                            x.clamp(i64::from(i32::MIN), i64::from(i32::MAX))
                        } else {
                            x.clamp(0, i64::from(u32::MAX))
                        };
                    }
                    x as u32
                }
                _ => return m.fail("arithmetic operation"),
            };
            m.set(l, rd, value);
            if cc {
                let floating = matches!(
                    name,
                    "FADD"
                        | "FADD32I"
                        | "FMUL"
                        | "FMUL32I"
                        | "FFMA"
                        | "FFMA32I"
                        | "FMNMX"
                        | "F2F"
                        | "I2F"
                );
                let (zero, negative) = if floating {
                    let f = f32::from_bits(value);
                    (f == 0.0, f < 0.0)
                } else {
                    (value == 0, value >> 31 != 0)
                };
                m.lanes[l].cc = (m.lanes[l].cc & (CC_C | CC_O))
                    | (u8::from(zero) * CC_Z)
                    | (u8::from(negative) * CC_S);
            }
            Ok(())
        })?;
        Ok(true)
    }
}
