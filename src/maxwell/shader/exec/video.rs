//! Video operand selection, VMAD arithmetic, and VSETP, which uses the
//! same selectors.
use super::*;
fn select(value: u32, selector: u32) -> Option<i64> {
    let signed = selector & 8 != 0;
    let (v, width) = match selector & 7 {
        0..=3 => (value >> ((selector & 3) * 8), 8),
        4 | 5 => (value >> ((selector & 1) * 16), 16),
        6 => (value, 32),
        _ => return None,
    };
    Some(if signed {
        sext(v, width) as i64
    } else {
        (v as u64 & ((1u64 << width) - 1)) as i64
    })
}
fn compare(a: i64, b: i64, op: u32) -> bool {
    match op {
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
    /// `VMNMX`: the minimum (or with MX, bit 56, the maximum) of the
    /// selected A and B, compared signed if B is, combined with register
    /// 39 by the operation 53:51, of which MIN (5) and MAX (6) are
    /// modeled, signed with bit 54 (selectors as `VMAD`).
    /// CC and SAT are not modeled.
    fn execute_vmnmx(&mut self, active: Mask) -> Result<(), Unmodeled> {
        let w = self.word;
        let sa = bits(w, 36, 3) | (bits(w, 48, 1) << 3);
        let sb = bits(w, 28, 3) | (bits(w, 49, 1) << 3);
        let operation = bits(w, 51, 3);
        self.require(!bit(w, 47), "CC")?;
        self.require(!bit(w, 55), "SAT")?;
        self.require(
            matches!(operation, 5 | 6),
            format_args!("operation {operation}"),
        )?;
        let pick = |x: u32, y: u32, signed: bool, max: bool| {
            let less = if signed {
                (x as i32) < (y as i32)
            } else {
                x < y
            };
            if less != max { x } else { y }
        };
        self.each(active, |m, l| {
            let a = select(m.r(l, bits(w, 8, 8)), sa).ok_or_else(|| m.unmodeled("A selector"))?;
            let b = if bit(w, 50) {
                select(m.r(l, bits(w, 20, 8)), sb).ok_or_else(|| m.unmodeled("B selector"))?
            } else if sb & 8 != 0 {
                bits(w, 20, 16) as i16 as i64
            } else {
                bits(w, 20, 16) as i64
            };
            let first = pick(a as u32, b as u32, sb & 8 != 0, bit(w, 56));
            let c = m.r(l, bits(w, 39, 8));
            m.set(
                l,
                bits(w, 0, 8),
                pick(first, c, bit(w, 54), operation == 6),
            );
            Ok(())
        })
    }

    pub(super) fn execute_video(&mut self, active: Mask) -> Result<bool, Unmodeled> {
        if self.op == Op::Vmnmx {
            self.execute_vmnmx(active)?;
            return Ok(true);
        }
        if !matches!(self.op, Op::Vmad | Op::Vsetp) {
            return Ok(false);
        }
        let w = self.word;
        let pred = self.op == Op::Vsetp;
        let sa = bits(w, 36, 3) | (bits(w, 48, 1) << 3);
        let sb = bits(w, 28, 3) | (bits(w, 49, 1) << 3);
        if !pred {
            self.require(!bit(w, 47), "CC")?;
            self.require(bits(w, 51, 2) < 3, "scale")?;
        }
        self.each(active, |m, l| {
            let a = select(m.r(l, bits(w, 8, 8)), sa).ok_or_else(|| m.unmodeled("A selector"))?;
            let b = if bit(w, 50) {
                select(m.r(l, bits(w, 20, 8)), sb).ok_or_else(|| m.unmodeled("B selector"))?
            } else if sb & 8 != 0 {
                bits(w, 20, 16) as i16 as i64
            } else {
                bits(w, 20, 16) as i64
            };
            if pred {
                let op = bits(w, 45, 2);
                m.require(op < 3, "boolean operation")?;
                let test = compare(a, b, bits(w, 43, 2) | (bits(w, 47, 1) << 2));
                let p = m.src_pred(l, 39);
                m.set_pred(l, bits(w, 3, 3), boolean(test, p, op));
                m.set_pred(l, bits(w, 0, 3), boolean(!test, p, op));
            } else {
                let mode = bits(w, 53, 2);
                let product = (a as i128) * (b as i128);
                let c = m.r(l, bits(w, 39, 8));
                let c = if mode == 1 { c.wrapping_neg() } else { c };
                let v = (if mode == 2 { -product } else { product })
                    + c as i128
                    + i128::from(mode == 3);
                let shift = match bits(w, 51, 2) {
                    1 => 7,
                    2 => 15,
                    _ => 0,
                };
                let v = v >> shift;
                let signed = sa == 14 || sb == 14 || matches!(mode, 1 | 2);
                let v = if bit(w, 55) {
                    if signed {
                        v.clamp(i32::MIN as i128, i32::MAX as i128)
                    } else {
                        v.clamp(0, u32::MAX as i128)
                    }
                } else {
                    v
                };
                m.set(l, bits(w, 0, 8), v as u32);
            }
            Ok(())
        })?;
        Ok(true)
    }
}
