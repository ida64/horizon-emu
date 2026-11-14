//! Integer and predicate fields [public: Ryujinx Decoders/InstDecoders.cs].
//! XMAD modes [public: Instructions/InstEmitIntegerArithmetic.cs EmitXmad];
//! IADD3 halves and 33-bit shift [public: eden translate/impl/integer_add_three_input.cpp].
use super::*;
fn half(v: u32, high: bool, signed: bool) -> u32 {
    let v = if high { v >> 16 } else { v };
    if signed {
        v as i16 as i32 as u32
    } else {
        v & 0xffff
    }
}
fn cmp(a: u32, b: u32, c: u32, s: bool) -> bool {
    let (a, b) = if s {
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
    pub(super) fn execute_integer(&mut self, active: u8) -> Result<bool, Unmodeled> {
        let form = match self.op {
            Op::BfeR
            | Op::BfiR
            | Op::Iadd3R
            | Op::Lop3R
            | Op::XmadR
            | Op::ImnmxR
            | Op::IcmpR
            | Op::R2pR
            | Op::P2rR
            | Op::Psetp => Form::R,
            Op::BfeI
            | Op::BfiI
            | Op::Iadd3I
            | Op::Lop3I
            | Op::XmadI
            | Op::ImnmxI
            | Op::IcmpI
            | Op::R2pI
            | Op::P2rI => Form::I,
            Op::BfeC
            | Op::BfiC
            | Op::Iadd3C
            | Op::Lop3C
            | Op::XmadC
            | Op::ImnmxC
            | Op::IcmpC
            | Op::R2pC
            | Op::P2rC => Form::C,
            Op::BfiRc | Op::XmadRc | Op::IcmpRc => Form::Rc,
            _ => return Ok(false),
        };
        let w = self.word;
        let op = self.op;
        let rd = bits(w, 0, 8);
        self.each(active, |m, l| {
            let a = m.r(l, bits(w, 8, 8));
            let b = m.b(l, form)?;
            let c = m.c(l, form)?;
            let mut cc = bit(w, 47);
            let v = match op {
                Op::BfeR | Op::BfeI | Op::BfeC => {
                    let a = if bit(w, 40) { a.reverse_bits() } else { a };
                    let start = b & 255;
                    let width = ((b >> 8) & 255).min(32u32.saturating_sub(start));
                    if width == 0 {
                        0
                    } else {
                        let value = a >> start;
                        if bit(w, 48) {
                            sext(value, width) as u32
                        } else {
                            value & u32::MAX.wrapping_shr(32 - width)
                        }
                    }
                }
                Op::BfiR | Op::BfiI | Op::BfiC | Op::BfiRc => {
                    let start = b & 255;
                    let width = ((b >> 8) & 255).min(32u32.saturating_sub(start));
                    if width == 0 {
                        c
                    } else {
                        let mask = u32::MAX.wrapping_shr(32 - width) << start;
                        (c & !mask) | ((a << start) & mask)
                    }
                }
                Op::Lop3R | Op::Lop3I | Op::Lop3C => {
                    m.require(
                        !bit(
                            w,
                            match form {
                                Form::R => 38,
                                Form::I => 57,
                                _ => 56,
                            },
                        ),
                        "X",
                    )?;
                    let lut = bits(w, if form == Form::R { 28 } else { 48 }, 8);
                    let mut v = 0;
                    for i in 0..8 {
                        if lut & (1 << i) != 0 {
                            v |= (if i & 4 != 0 { a } else { !a })
                                & (if i & 2 != 0 { b } else { !b })
                                & (if i & 1 != 0 { c } else { !c });
                        }
                    }
                    if form == Form::R {
                        let p = match bits(w, 36, 2) {
                            0 => false,
                            1 => true,
                            2 => v == 0,
                            _ => v != 0,
                        };
                        m.set_pred(l, bits(w, 48, 3), p);
                    }
                    v
                }
                Op::Iadd3R | Op::Iadd3I | Op::Iadd3C => {
                    let select = |v: u32, sel: u32| match sel {
                        0 => v,
                        1 => v & 0xffff,
                        _ => v >> 16,
                    };
                    let mut values = [a, b, c];
                    if form == Form::R {
                        for (i, lo) in [35, 33, 31].into_iter().enumerate() {
                            let sel = bits(w, lo, 2);
                            m.require(sel < 3, "half selector")?;
                            values[i] = select(values[i], sel);
                        }
                    }
                    for (i, lo) in [51, 50, 49].into_iter().enumerate() {
                        if bit(w, lo) {
                            values[i] = values[i].wrapping_neg();
                        }
                    }
                    let shift = if form == Form::R { bits(w, 37, 2) } else { 0 };
                    m.require(shift < 3, "shift")?;
                    m.require(!(bit(w, 48) && (shift == 1 || cc)), "X with RS/CC")?;
                    let sum = values[0] as u64
                        + values[1] as u64
                        + u64::from(bit(w, 48) && m.lanes[l].cc & CC_C != 0);
                    let a = match shift {
                        1 => (sum >> 16) as u32,
                        2 => (sum as u32) << 16,
                        _ => sum as u32,
                    };
                    let v = a.wrapping_add(values[2]);
                    if cc {
                        m.lanes[l].cc = (u8::from(v == 0) * CC_Z)
                            | (u8::from(v >> 31 != 0) * CC_S)
                            | (u8::from(v < a) * CC_C)
                            | (u8::from(
                                ((!(a ^ values[2]) & (a ^ v)) >> 31 != 0) || sum >> 32 != 0,
                            ) * CC_O);
                        cc = false;
                    }
                    v
                }
                Op::XmadR | Op::XmadI | Op::XmadC | Op::XmadRc => {
                    let constant = matches!(form, Form::C | Form::Rc);
                    let b = if form == Form::I { bits(w, 20, 16) } else { b };
                    let x = half(a, bit(w, 53), bit(w, 48));
                    let y = half(
                        b,
                        form != Form::I && bit(w, if constant { 52 } else { 35 }),
                        bit(w, 49),
                    );
                    let mut product = x.wrapping_mul(y);
                    let shift = form != Form::Rc && bit(w, if constant { 55 } else { 36 });
                    if shift {
                        product <<= 16;
                    }
                    let mode = bits(w, 50, if constant { 2 } else { 3 });
                    let c = match mode {
                        0 => c,
                        1 => c & 0xffff,
                        2 => c >> 16,
                        3 => c.wrapping_sub(((x >> 31) + (y >> 31)) << 16),
                        4 => c.wrapping_add(b << 16),
                        _ => return m.fail("cop mode"),
                    };
                    let extended = bit(w, if constant { 54 } else { 38 });
                    m.require(!extended, "X")?;
                    let v = product.wrapping_add(c);
                    if cc {
                        m.lanes[l].cc = (u8::from(v == 0) * CC_Z)
                            | (u8::from(v >> 31 != 0) * CC_S)
                            | (u8::from(v < product) * CC_C)
                            | (u8::from((!(product ^ c) & (product ^ v)) >> 31 != 0) * CC_O);
                        cc = false;
                    }
                    if form != Form::Rc && bit(w, if constant { 56 } else { 37 }) {
                        (v & 0xffff) | (b << 16)
                    } else {
                        v
                    }
                }
                Op::ImnmxR | Op::ImnmxI | Op::ImnmxC => {
                    m.require(bits(w, 43, 2) == 0, "extended mode")?;
                    let less = cmp(a, b, 1, bit(w, 48));
                    if less == m.src_pred(l, 39) { a } else { b }
                }
                Op::IcmpR | Op::IcmpI | Op::IcmpC | Op::IcmpRc => {
                    cc = false;
                    if cmp(c, 0, bits(w, 49, 3), bit(w, 48)) {
                        a
                    } else {
                        b
                    }
                }
                Op::R2pR | Op::R2pI | Op::R2pC => {
                    let value = (a >> (bits(w, 41, 2) * 8)) as u8;
                    let mask = b as u8;
                    if bit(w, 40) {
                        m.lanes[l].cc = (m.lanes[l].cc & !mask) | (value & mask & 15);
                    } else {
                        m.lanes[l].p = ((m.lanes[l].p & !mask) | (value & mask)) | 0x80;
                    }
                    return Ok(());
                }
                Op::P2rR | Op::P2rI | Op::P2rC => {
                    cc = false;
                    let shift = bits(w, 41, 2) * 8;
                    let mask = (b & 255) << shift;
                    let value = if bit(w, 40) {
                        m.lanes[l].cc
                    } else {
                        m.lanes[l].p
                    };
                    (a & !mask) | (((value as u32) << shift) & mask)
                }
                Op::Psetp => {
                    let ab = bits(w, 24, 2);
                    let cop = bits(w, 45, 2);
                    m.require(ab < 3 && cop < 3, "boolean operation")?;
                    let test = boolean(m.src_pred(l, 12), m.src_pred(l, 29), ab);
                    let p = m.src_pred(l, 39);
                    m.set_pred(l, bits(w, 3, 3), boolean(test, p, cop));
                    m.set_pred(l, bits(w, 0, 3), boolean(!test, p, cop));
                    return Ok(());
                }
                _ => return m.fail("integer operation"),
            };
            m.set(l, rd, v);
            if cc {
                m.lanes[l].cc = (m.lanes[l].cc & (CC_C | CC_O))
                    | (u8::from(v == 0) * CC_Z)
                    | (u8::from(v >> 31 != 0) * CC_S);
            }
            Ok(())
        })?;
        Ok(true)
    }
}
