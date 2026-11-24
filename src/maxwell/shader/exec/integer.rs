//! Integer and predicate instructions: field layouts, XMAD modes, IADD3
//! halves and 33-bit shift.
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
    /// `CSET`, `CSETP` and `PSET`: a condition-code test (bits 12:8), or
    /// for `PSET` the boolean operation 25:24 of the predicates at 12 and
    /// 29, combined with the predicate at 39 by the operation 46:45. The
    /// outcome goes to a register as all ones, or 1.0 with BF (bit 44),
    /// and 0 otherwise (`CSET`, `PSET`), with the condition code's zero
    /// and (without BF) sign from it if bit 47 asks and carry and
    /// overflow reset; or, for `CSETP`, to the predicate at 3, and
    /// combined from the test's negation to the predicate at 0.
    fn execute_flag_set(&mut self, active: Mask) -> Result<bool, Unmodeled> {
        let (w, op) = (self.word, self.op);
        if !matches!(op, Op::Cset | Op::Csetp | Op::Pset) {
            return Ok(false);
        }
        let cop = bits(w, 45, 2);
        self.require(cop < 3, "boolean operation")?;
        self.each(active, |m, l| {
            let test = if matches!(op, Op::Pset) {
                let ab = bits(w, 24, 2);
                m.require(ab < 3, "boolean operation")?;
                boolean(m.src_pred(l, 12), m.src_pred(l, 29), ab)
            } else {
                m.condition(l, bits(w, 8, 5))?
            };
            let p = m.src_pred(l, 39);
            if matches!(op, Op::Csetp) {
                m.set_pred(l, bits(w, 3, 3), boolean(test, p, cop));
                m.set_pred(l, bits(w, 0, 3), boolean(!test, p, cop));
                return Ok(());
            }
            let float = bit(w, 44);
            let v = match (boolean(test, p, cop), float) {
                (false, _) => 0,
                (true, false) => u32::MAX,
                (true, true) => 1f32.to_bits(),
            };
            m.set(l, bits(w, 0, 8), v);
            if bit(w, 47) {
                m.lanes[l].cc = (u8::from(v == 0) * CC_Z) | (u8::from(v != 0 && !float) * CC_S);
            }
            Ok(())
        })?;
        Ok(true)
    }

    pub(super) fn execute_integer(&mut self, active: Mask) -> Result<bool, Unmodeled> {
        if self.execute_flag_set(active)? {
            return Ok(true);
        }
        let form = match self.op {
            Op::BfeR
            | Op::BfiR
            | Op::Iadd3R
            | Op::Lop3R
            | Op::XmadR
            | Op::ImnmxR
            | Op::IcmpR
            | Op::FcmpR
            | Op::PopcR
            | Op::FloR
            | Op::ImulR
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
            | Op::PopcI
            | Op::FloI
            | Op::ImulI
            | Op::R2pI
            | Op::P2rI => Form::I,
            Op::BfeC
            | Op::BfiC
            | Op::Iadd3C
            | Op::Lop3C
            | Op::XmadC
            | Op::ImnmxC
            | Op::IcmpC
            | Op::FcmpC
            | Op::PopcC
            | Op::FloC
            | Op::ImulC
            | Op::R2pC
            | Op::P2rC => Form::C,
            Op::BfiRc | Op::XmadRc | Op::IcmpRc | Op::FcmpRc | Op::LeaHiC => Form::Rc,
            Op::FcmpI => Form::F,
            Op::LeaR | Op::LeaHiR => Form::R,
            Op::LeaI => Form::I,
            Op::LeaC => Form::C,
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
                // `FCMP`: as `ICMP`, the comparison (bits 51:48) of float
                // C with 0.0.
                Op::FcmpR | Op::FcmpI | Op::FcmpC | Op::FcmpRc => {
                    m.require(!cc, "condition code write")?;
                    if super::alu::compare(f32::from_bits(c), 0.0, bits(w, 48, 4)) {
                        a
                    } else {
                        b
                    }
                }
                // `POPC`: the set bits of B, complemented first with bit
                // 40.
                Op::PopcR | Op::PopcI | Op::PopcC => (if bit(w, 40) { !b } else { b }).count_ones(),
                // `FLO`: the index of the most significant set bit of B
                // (complemented first with bit 40; signed with bit 48:
                // the most significant bit differing from the sign), all
                // ones if there is none; with SH (bit 41) counted from
                // bit 31 instead.
                Op::FloR | Op::FloI | Op::FloC => {
                    m.require(!cc, "condition code write")?;
                    let b = if bit(w, 40) { !b } else { b };
                    let b = if bit(w, 48) && b >> 31 != 0 { !b } else { b };
                    match b.leading_zeros() {
                        32 => u32::MAX,
                        zeros if bit(w, 41) => zeros,
                        zeros => 31 - zeros,
                    }
                }
                // `IMUL`: the low 32 bits of A times B, or with HI (bit
                // 39) the high 32 of the 64-bit product, signed with bit
                // 40.
                Op::ImulR | Op::ImulI | Op::ImulC => {
                    m.require(!cc, "condition code write")?;
                    if !bit(w, 39) {
                        a.wrapping_mul(b)
                    } else if bit(w, 40) {
                        ((i64::from(a as i32) * i64::from(b as i32)) >> 32) as u32
                    } else {
                        ((u64::from(a) * u64::from(b)) >> 32) as u32
                    }
                }
                // `LEA`: B plus A shifted left by the scale (bits 43:39),
                // A negated first with bit 45. `LEA.HI`: the base (B, or
                // the constant buffer word) plus the bits above the low
                // 32 of the 64-bit offset (A low, register 39 high;
                // negated first) shifted left by the scale (bits 32:28
                // with NEG 37 and X 38, or 55:51, 56 and 57 for the
                // constant buffer form). X (carry in), the predicate
                // result (bits 50:48) and CC are not modeled.
                Op::LeaR | Op::LeaI | Op::LeaC | Op::LeaHiR | Op::LeaHiC => {
                    m.require(!cc, "condition code write")?;
                    m.require(bits(w, 48, 3) == PT, "predicate result")?;
                    let (base, high, scale, negate, extended) = match op {
                        Op::LeaHiR => (b, Some(c), bits(w, 28, 5), bit(w, 37), bit(w, 38)),
                        Op::LeaHiC => (c, Some(b), bits(w, 51, 5), bit(w, 56), bit(w, 57)),
                        _ => (b, None, bits(w, 39, 5), bit(w, 45), bit(w, 46)),
                    };
                    m.require(!extended, "X")?;
                    match high {
                        None => {
                            let offset = if negate { a.wrapping_neg() } else { a };
                            base.wrapping_add(offset << scale)
                        }
                        Some(high) => {
                            let offset = u64::from(a) | u64::from(high) << 32;
                            let offset = if negate {
                                offset.wrapping_neg()
                            } else {
                                offset
                            };
                            base.wrapping_add((offset >> (32 - scale)) as u32)
                        }
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
