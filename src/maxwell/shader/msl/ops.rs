//! Arithmetic, logic, comparison and conversion instructions, written out
//! as the interpreter executes them (`exec/alu.rs`, `exec/integer.rs`).

use super::super::decode::Op;
use super::super::exec::{bit, bits, sext};
use super::{Form, Writer};

/// The sign-extended 20-bit immediate of `I` forms: bits 20-38 and the
/// sign at bit 56.
pub(super) fn imm20(word: u64) -> i32 {
    sext(bits(word, 20, 19) | (bits(word, 56, 1) << 19), 20)
}

fn modify(value: &str, abs: bool, neg: bool) -> String {
    let v = if abs {
        format!("fabs({value})")
    } else {
        value.to_string()
    };
    if neg { format!("(-{v})") } else { v }
}

fn flush(value: &str, enabled: bool) -> String {
    if enabled {
        format!("ftz({value})")
    } else {
        value.to_string()
    }
}

/// [`flush`] of an operand or the result of a product, a sum, a fused
/// multiply-add or a comparison: nothing where the host's operation
/// makes the flush itself (`super::host_flushes`).
fn flush_operation(value: &str, enabled: bool) -> String {
    flush(value, enabled && !super::host_flushes())
}

fn saturate(value: &str, enabled: bool) -> String {
    if enabled {
        format!("sat({value})")
    } else {
        value.to_string()
    }
}

/// `exec/alu.rs` `compare` on floats `x`, `y`.
fn float_compare(c: u32) -> &'static str {
    match c {
        0 => "false",
        1 => "(x < y)",
        2 => "(x == y)",
        3 => "(x <= y)",
        4 => "(x > y)",
        5 => "(!(isnan(x) || isnan(y)) && x != y)",
        6 => "(x >= y)",
        7 => "(!(isnan(x) || isnan(y)))",
        8 => "(isnan(x) || isnan(y))",
        9 => "(isnan(x) || isnan(y) || x < y)",
        10 => "(isnan(x) || isnan(y) || x == y)",
        11 => "(isnan(x) || isnan(y) || x <= y)",
        12 => "(isnan(x) || isnan(y) || x > y)",
        13 => "(x != y)",
        14 => "(isnan(x) || isnan(y) || x >= y)",
        _ => "true",
    }
}

/// Integer comparison `c` of `x`, `y` (already of the signedness wanted).
fn integer_compare(c: u32, x: &str, y: &str) -> String {
    let op = match c {
        0 => return "false".into(),
        1 => "<",
        2 => "==",
        3 => "<=",
        4 => ">",
        5 => "!=",
        6 => ">=",
        _ => return "true".into(),
    };
    format!("({x} {op} {y})")
}

fn boolean(a: &str, b: &str, op: u32) -> String {
    match op {
        0 => format!("({a} && {b})"),
        1 => format!("({a} || {b})"),
        _ => format!("({a} != {b})"),
    }
}

/// `exec/integer.rs` `half`.
fn half(v: &str, high: bool, signed: bool) -> String {
    let v = if high {
        format!("({v} >> 16)")
    } else {
        v.to_string()
    };
    if signed {
        format!("as_type<uint>(as_type<int>({v} << 16) >> 16)")
    } else {
        format!("({v} & 0xffffu)")
    }
}

impl Writer<'_> {
    /// The generic condition-code update after a value write: zero and
    /// sign of `v` (as a float for floating results), carry and overflow
    /// kept.
    fn write_cc(&mut self, floating: bool) {
        self.cc = true;
        if floating {
            self.line("cc = (cc & 12u) | (as_type<float>(v) == 0.0f ? 1u : 0u) | (as_type<float>(v) < 0.0f ? 2u : 0u);");
        } else {
            self.line("cc = (cc & 12u) | (v == 0u ? 1u : 0u) | ((v >> 31) != 0u ? 2u : 0u);");
        }
    }

    /// `exec/alu.rs` `execute_alu`; false if `op` is not one of its
    /// instructions.
    pub(super) fn alu(&mut self, op: Op) -> Result<bool, String> {
        let (name, form) = match op {
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
        let wide = form == Form::Imm32;
        // Checks that do not depend on values come first, so a refusal
        // leaves no half-written scope behind.
        match name {
            "MOV" | "MOV32I" => {
                let mask = bits(w, if name == "MOV" { 39 } else { 12 }, 4);
                self.require(mask == 15, "partial quad mask")?;
            }
            "FADD" if !wide => self.require(bits(w, 39, 2) == 0, "rounding mode")?,
            "FMUL" | "FMUL32I" => {
                let mode = bits(w, if wide { 53 } else { 44 }, 2);
                self.require(mode <= 2, "FMZ encoding")?;
                let scale = if wide { 0 } else { bits(w, 41, 3) };
                self.require(scale < 7, "scale")?;
            }
            "FFMA" | "FFMA32I" => {
                if !wide {
                    self.require(bits(w, 51, 2) == 0, "rounding mode")?;
                }
                self.require(bits(w, 53, 2) <= 2, "FMZ encoding")?;
            }
            "MUFU" => {
                let sub = bits(w, 20, 4);
                self.require(matches!(sub, 0..=5 | 8), &format!("suboperation {sub}"))?;
            }
            "FSETP" | "FSET" | "ISETP" | "ISET" => {
                self.require(bits(w, 45, 2) < 3, "boolean operation")?;
                if name.starts_with('I') {
                    self.require(!bit(w, 43), "extended comparison")?;
                }
            }
            "IADD" | "IADD32I" => {
                self.require(!bit(w, if wide { 54 } else { 50 }), "SAT")?;
            }
            "LOP" | "LOP32I" => self.require(!bit(w, if wide { 57 } else { 43 }), "X")?,
            "SHL" | "SHR" => self.require(
                bits(w, 43, if name == "SHL" { 1 } else { 2 }) == 0,
                "extended shift",
            )?,
            "SHF<" | "SHF>" => {
                self.require(bits(w, 48, 2) == 0, "extended funnel shift")?;
                self.require(bits(w, 37, 2) != 1, "maximum shift 1")?;
            }
            "F2I" | "F2F" => {
                self.require(bits(w, 10, 2) == 2, "source format")?;
                self.require(!bit(w, 41), "high half")?;
                if name == "F2F" {
                    self.require(bits(w, 8, 2) == 2, "destination format")?;
                } else {
                    let format = bits(w, 8, 2) | (bits(w, 12, 1) << 2);
                    self.require(
                        matches!(format, 1 | 2 | 5 | 6),
                        &format!("destination format {format}"),
                    )?;
                }
            }
            "I2F" => {
                self.require(bits(w, 8, 2) == 2, "destination format")?;
                let format = bits(w, 10, 2) | (bits(w, 13, 1) << 2);
                self.require(
                    (format & 3) != 2 || bits(w, 41, 2) == 0,
                    "byte selector with 32-bit source",
                )?;
                self.require(
                    matches!(format, 0 | 1 | 2 | 4 | 5 | 6),
                    &format!("source format {format}"),
                )?;
            }
            "I2I" => {
                let destination = bits(w, 8, 2) | (bits(w, 12, 1) << 2);
                self.require(destination & 3 == 2, "destination narrower than 32 bits")?;
                let format = bits(w, 10, 2) | (bits(w, 13, 1) << 2);
                self.require(
                    (format & 3) != 2 || bits(w, 41, 2) == 0,
                    "byte selector with 32-bit source",
                )?;
                self.require(
                    matches!(format, 0 | 1 | 2 | 4 | 5 | 6),
                    &format!("source format {format}"),
                )?;
            }
            _ => {}
        }
        let a = self.read(ra);
        let b = self.b(form)?;
        self.line(&format!(
            "{{ uint a = {a}; uint b = {b}; float af = as_type<float>(a); float bf = as_type<float>(b);"
        ));
        // `Some(floating)`: the instruction may update the condition code
        // from its value.
        let mut cc: Option<bool> = None;
        match name {
            "MOV" | "MOV32I" => self.line("uint v = b;"),
            "FADD" | "FADD32I" => {
                let ftz = bit(w, if wide { 55 } else { 44 });
                let sat = !wide && bit(w, 50);
                let x = flush_operation(
                    &modify(
                        "af",
                        bit(w, if wide { 54 } else { 46 }),
                        bit(w, if wide { 56 } else { 48 }),
                    ),
                    ftz,
                );
                let y = flush_operation(
                    &modify(
                        "bf",
                        bit(w, if wide { 57 } else { 49 }),
                        bit(w, if wide { 53 } else { 45 }),
                    ),
                    ftz,
                );
                self.line(&format!("float x = {x}; float y = {y};"));
                let v = saturate(&flush_operation("(x + y)", ftz), sat);
                self.line(&format!("uint v = as_type<uint>({v});"));
                cc = bit(w, if wide { 52 } else { 47 }).then_some(true);
            }
            "FMUL" | "FMUL32I" => {
                let round = if wide { 0 } else { bits(w, 39, 2) };
                let mode = bits(w, if wide { 53 } else { 44 }, 2);
                let scale = if wide { 0 } else { bits(w, 41, 3) };
                let sat = bit(w, if wide { 55 } else { 50 });
                // The scaled and directed products are more than one
                // operation: their flushes stay written out.
                let flush = if round == 0 && scale == 0 {
                    flush_operation
                } else {
                    flush
                };
                let x = flush(&modify("af", false, !wide && bit(w, 48)), mode == 1);
                let y = flush("bf", mode == 1);
                self.line(&format!("float x = {x}; float y = {y};"));
                let product = if round == 0 && scale == 0 {
                    "(x * y)".to_string()
                } else {
                    let factor =
                        ["1.0f", "0.5f", "0.25f", "0.125f", "8.0f", "4.0f", "2.0f"][scale as usize];
                    if round == 0 {
                        format!("((x * y) * {factor})")
                    } else {
                        format!("mul_round(x, y, {factor}, {round}u)")
                    }
                };
                // FMZ: a zero factor gives a zero of the product's sign.
                let product = if mode == 2 {
                    format!(
                        "((x == 0.0f || y == 0.0f) ? as_type<float>((as_type<uint>(x) ^ as_type<uint>(y)) & 0x80000000u) : {product})"
                    )
                } else {
                    product
                };
                let v = saturate(&flush(&product, mode == 1), sat);
                self.line(&format!("uint v = as_type<uint>({v});"));
                cc = bit(w, if wide { 52 } else { 47 }).then_some(true);
            }
            "FFMA" | "FFMA32I" => {
                let mode = bits(w, 53, 2);
                let sat = bit(w, if wide { 55 } else { 50 });
                let c = if wide { self.read(rd) } else { self.c(form)? };
                let x = flush_operation(
                    &modify("af", false, bit(w, if wide { 56 } else { 48 })),
                    mode == 1,
                );
                let y = flush_operation("bf", mode == 1);
                let z = flush_operation(
                    &modify(
                        &format!("as_type<float>({c})"),
                        false,
                        bit(w, if wide { 57 } else { 49 }),
                    ),
                    mode == 1,
                );
                self.line(&format!("float x = {x}; float y = {y}; float z = {z};"));
                let v = if mode == 2 {
                    "((x == 0.0f || y == 0.0f) ? z : fma(x, y, z))"
                } else {
                    "fma(x, y, z)"
                };
                let v = saturate(&flush_operation(v, mode == 1), sat);
                self.line(&format!("uint v = as_type<uint>({v});"));
                cc = bit(w, if wide { 52 } else { 47 }).then_some(true);
            }
            "FMNMX" => {
                let ftz = bit(w, 44);
                let x = flush(&modify("af", bit(w, 46), bit(w, 48)), ftz);
                let y = flush(&modify("bf", bit(w, 49), bit(w, 45)), ftz);
                let p = self.pred_operand(39);
                self.line(&format!(
                    "float x = {x}; float y = {y}; uint v = as_type<uint>({p} ? fmin(x, y) : fmax(x, y));"
                ));
                cc = bit(w, 47).then_some(true);
            }
            "MUFU" => {
                let x = modify("af", bit(w, 46), bit(w, 48));
                let v = match bits(w, 20, 4) {
                    0 => format!("precise::cos({x})"),
                    1 => format!("precise::sin({x})"),
                    2 => format!("precise::exp2({x})"),
                    3 => format!("precise::log2({x})"),
                    4 => format!("precise::divide(1.0f, {x})"),
                    5 => format!("precise::divide(1.0f, precise::sqrt({x}))"),
                    _ => format!("precise::sqrt({x})"),
                };
                let v = saturate(&v, bit(w, 50));
                self.line(&format!("uint v = as_type<uint>({v});"));
            }
            // RRO is a range-reduction hint (the interpreter's note).
            "RRO" => {
                let v = modify("bf", bit(w, 49), bit(w, 45));
                self.line(&format!("uint v = as_type<uint>({v});"));
            }
            "FSETP" | "FSET" | "ISETP" | "ISET" => {
                let fp = name.starts_with('F');
                let predicate = name.ends_with('P');
                let bop = bits(w, 45, 2);
                let test = if fp {
                    let x = modify("af", bit(w, if predicate { 7 } else { 54 }), bit(w, 43));
                    let y = modify("bf", bit(w, 44), bit(w, if predicate { 6 } else { 53 }));
                    let ftz = bit(w, if predicate { 47 } else { 55 });
                    self.line(&format!(
                        "float x = {}; float y = {};",
                        flush_operation(&x, ftz),
                        flush_operation(&y, ftz)
                    ));
                    float_compare(bits(w, 48, 4)).to_string()
                } else if bit(w, 48) {
                    integer_compare(bits(w, 49, 3), "as_type<int>(a)", "as_type<int>(b)")
                } else {
                    integer_compare(bits(w, 49, 3), "a", "b")
                };
                let p = self.pred_operand(39);
                self.line(&format!("bool t = {test}; bool q = {p};"));
                if predicate {
                    let first = boolean("t", "q", bop);
                    let second = boolean("!t", "q", bop);
                    self.set_pred(bits(w, 3, 3), &first);
                    self.set_pred(bits(w, 0, 3), &second);
                    self.line("}");
                    return Ok(true);
                }
                let one = if bit(w, if fp { 52 } else { 44 }) {
                    "0x3f800000u"
                } else {
                    "0xffffffffu"
                };
                let result = boolean("t", "q", bop);
                self.line(&format!("uint v = {result} ? {one} : 0u;"));
                cc = bit(w, 47).then_some(false);
            }
            "IADD" | "IADD32I" | "ISCADD" | "ISCADD32I" => {
                let shift = name.starts_with("ISCADD");
                let mut x = if shift {
                    format!("(a << {}u)", bits(w, if wide { 53 } else { 39 }, 5))
                } else {
                    "a".to_string()
                };
                let mut y = "b".to_string();
                let mode = if shift && wide {
                    0
                } else {
                    bits(w, if wide { 55 } else { 48 }, 2)
                };
                let extended = !shift && bit(w, if wide { 53 } else { 43 });
                if extended {
                    self.cc = true;
                }
                let mut carry = if extended {
                    "((cc >> 2) & 1u)".to_string()
                } else {
                    "0u".to_string()
                };
                match mode {
                    1 => {
                        y = format!("(~{y})");
                        if !extended {
                            carry = format!("({carry} + 1u)");
                        }
                    }
                    2 => {
                        x = format!("(~{x})");
                        if !extended {
                            carry = format!("({carry} + 1u)");
                        }
                    }
                    3 => carry = format!("({carry} + 1u)"),
                    _ => {}
                }
                self.line(&format!(
                    "uint x = {x}; uint y = {y}; ulong s = ulong(x) + ulong(y) + ulong({carry}); uint v = uint(s);"
                ));
                if bit(w, if wide { 52 } else { 47 }) {
                    self.cc = true;
                    let zero = if extended {
                        "(v == 0u && (cc & 1u) != 0u)"
                    } else {
                        "(v == 0u)"
                    };
                    self.line(&format!(
                        "cc = ({zero} ? 1u : 0u) | ((v >> 31) != 0u ? 2u : 0u) | ((s >> 32) != 0ul ? 4u : 0u) | ((((~(x ^ y)) & (x ^ v)) >> 31) != 0u ? 8u : 0u);"
                    ));
                }
            }
            "LOP" | "LOP32I" => {
                let x = if bit(w, if wide { 55 } else { 39 }) {
                    "(~a)"
                } else {
                    "a"
                };
                let y = if bit(w, if wide { 56 } else { 40 }) {
                    "(~b)"
                } else {
                    "b"
                };
                let v = match bits(w, if wide { 53 } else { 41 }, 2) {
                    0 => format!("({x} & {y})"),
                    1 => format!("({x} | {y})"),
                    2 => format!("({x} ^ {y})"),
                    _ => y.to_string(),
                };
                self.line(&format!("uint v = {v};"));
                if !wide {
                    let p = match bits(w, 44, 2) {
                        0 => "false",
                        1 => "true",
                        2 => "(v == 0u)",
                        _ => "(v != 0u)",
                    };
                    self.set_pred(bits(w, 48, 3), p);
                }
                cc = bit(w, if wide { 52 } else { 47 }).then_some(false);
            }
            "SHL" | "SHR" => {
                let n = if bit(w, 39) {
                    "(b & 31u)"
                } else {
                    "min(b, 32u)"
                };
                let x = if name == "SHR" && bit(w, 40) {
                    "reverse_bits(a)"
                } else {
                    "a"
                };
                self.line(&format!("uint n = {n}; uint x = {x};"));
                let v = if name == "SHL" {
                    "(n >= 32u ? 0u : x << n)"
                } else if bit(w, 48) {
                    "as_type<uint>(as_type<int>(x) >> min(n, 31u))"
                } else {
                    "(n >= 32u ? 0u : x >> n)"
                };
                self.line(&format!("uint v = {v};"));
                cc = bit(w, 47).then_some(false);
            }
            // `exec::alu` "SHF<" / "SHF>".
            "SHF<" | "SHF>" => {
                let maximum = bits(w, 37, 2);
                let limit = if maximum == 0 { 32 } else { 64 };
                let count = if form == Form::I {
                    format!("{}u", bits(w, 20, 6))
                } else {
                    "b".to_string()
                };
                let n = if bit(w, 50) {
                    format!("({count} & {}u)", limit - 1)
                } else {
                    format!("min({count}, {limit}u)")
                };
                let high = self.read(bits(w, 39, 8));
                self.line(&format!(
                    "uint n = {n}; ulong packed = (ulong({high}) << 32) | ulong(a);"
                ));
                let v = if name == "SHF<" {
                    "uint((n >= 64u ? 0ul : packed << n) >> 32)"
                } else if maximum == 3 {
                    "uint(as_type<ulong>(as_type<long>(packed) >> min(n, 63u)))"
                } else {
                    "uint(n >= 64u ? 0ul : packed >> n)"
                };
                self.line(&format!("uint v = {v};"));
                cc = bit(w, 47).then_some(false);
            }
            "SEL" => {
                let p = self.pred_operand(39);
                self.line(&format!("uint v = {p} ? a : b;"));
            }
            "F2I" | "F2F" => {
                let x = flush(&modify("bf", bit(w, 49), bit(w, 45)), bit(w, 44));
                let rounded =
                    ["rint(x)", "floor(x)", "ceil(x)", "trunc(x)"][bits(w, 39, 2) as usize];
                self.line(&format!("float x = {x}; float rounded = {rounded};"));
                if name == "F2F" {
                    let v = if bit(w, 42) { "rounded" } else { "x" };
                    let v = saturate(v, bit(w, 50));
                    self.line(&format!("uint v = as_type<uint>({v});"));
                    cc = bit(w, 47).then_some(true);
                } else {
                    let convert = match bits(w, 8, 2) | (bits(w, 12, 1) << 2) {
                        1 => "f2u16",
                        2 => "f2u32",
                        5 => "f2i16",
                        _ => "f2i32",
                    };
                    self.line(&format!("uint v = {convert}(rounded);"));
                    cc = bit(w, 47).then_some(false);
                }
            }
            "I2F" | "I2I" => {
                let format = bits(w, 10, 2) | (bits(w, 13, 1) << 2);
                let selection = bits(w, 41, 2);
                let value = format!("(b >> {}u)", selection * 8);
                let x = match format {
                    0 => format!("long({value} & 0xffu)"),
                    1 => format!("long({value} & 0xffffu)"),
                    2 => format!("long({value})"),
                    4 => format!("long(as_type<int>({value} << 24) >> 24)"),
                    5 => format!("long(as_type<int>({value} << 16) >> 16)"),
                    _ => format!("long(as_type<int>({value}))"),
                };
                self.line(&format!("long x = {x};"));
                if bit(w, 49) {
                    self.line("x = abs(x);");
                }
                if name == "I2I" {
                    // `exec/alu.rs` "I2I": wrapping negation of the
                    // extended value, SAT clamps to the destination's
                    // range, the low 32 bits are stored.
                    if bit(w, 45) {
                        self.line("x = -x;");
                    }
                    if bit(w, 50) {
                        let signed = (bits(w, 8, 2) | (bits(w, 12, 1) << 2)) == 6;
                        self.line(if signed {
                            "x = clamp(x, -2147483648l, 2147483647l);"
                        } else {
                            "x = clamp(x, 0l, 4294967295l);"
                        });
                    }
                    self.line("uint v = uint(ulong(x));");
                    cc = bit(w, 47).then_some(false);
                } else {
                    let mode = bits(w, 39, 2);
                    if bit(w, 45) {
                        // -(0.0) is -0.0 in the interpreter's binary64.
                        self.line(&format!(
                            "float f = x == 0 ? -0.0f : i2f(-x, {mode}u); uint v = as_type<uint>(f);"
                        ));
                    } else {
                        self.line(&format!("uint v = as_type<uint>(i2f(x, {mode}u));"));
                    }
                    cc = bit(w, 47).then_some(true);
                }
            }
            _ => unreachable!("every name is handled"),
        }
        self.set(rd, "v");
        if let Some(floating) = cc {
            self.write_cc(floating);
        }
        self.line("}");
        Ok(true)
    }

    /// `VMNMX` (`exec/video.rs` `execute_vmnmx`).
    pub(super) fn vmnmx(&mut self) -> Result<(), String> {
        let w = self.word;
        let sa = bits(w, 36, 3) | (bits(w, 48, 1) << 3);
        let sb = bits(w, 28, 3) | (bits(w, 49, 1) << 3);
        let operation = bits(w, 51, 3);
        self.require(!bit(w, 47), "CC")?;
        self.require(!bit(w, 55), "SAT")?;
        self.require(matches!(operation, 5 | 6), &format!("operation {operation}"))?;
        // `exec/video.rs` `select`: a byte, half or the word of the
        // register, sign- or zero-extended.
        let operand = |value: &str, selector: u32| {
            let (shift, width) = match selector & 7 {
                0..=3 => ((selector & 3) * 8, 8),
                4 | 5 => ((selector & 1) * 16, 16),
                6 => (0, 32),
                _ => return None,
            };
            Some(if width == 32 {
                value.to_string()
            } else if selector & 8 != 0 {
                format!(
                    "as_type<uint>(as_type<int>({value} << {}u) >> {}u)",
                    32 - width - shift,
                    32 - width
                )
            } else {
                format!("(({value} >> {shift}u) & {:#x}u)", (1u32 << width) - 1)
            })
        };
        let a = self.read(bits(w, 8, 8));
        let Some(a) = operand(&a, sa) else {
            return self.refuse("A selector");
        };
        let b = if bit(w, 50) {
            let b = self.read(bits(w, 20, 8));
            match operand(&b, sb) {
                Some(b) => b,
                None => return self.refuse("B selector"),
            }
        } else if sb & 8 != 0 {
            format!("{:#x}u", bits(w, 20, 16) as i16 as i32 as u32)
        } else {
            format!("{:#x}u", bits(w, 20, 16))
        };
        let c = self.read(bits(w, 39, 8));
        // The one of `x` and `y` that is less (or, `max`, not less).
        let pick = |x: &str, y: &str, signed: bool, max: bool| {
            let less = if signed {
                format!("(as_type<int>({x}) < as_type<int>({y}))")
            } else {
                format!("({x} < {y})")
            };
            format!("({less} != {max} ? {x} : {y})")
        };
        self.line(&format!(
            "{{ uint a = {a}; uint b = {b}; uint c = {c}; uint f = {}; uint v = {};",
            pick("a", "b", sb & 8 != 0, bit(w, 56)),
            pick("f", "c", bit(w, 54), operation == 6)
        ));
        self.set(bits(w, 0, 8), "v");
        self.line("}");
        Ok(())
    }

    /// `exec/integer.rs` `execute_flag_set`.
    fn flag_set(&mut self, op: Op) -> Result<bool, String> {
        if !matches!(op, Op::Cset | Op::Csetp | Op::Pset) {
            return Ok(false);
        }
        let w = self.word;
        let cop = bits(w, 45, 2);
        self.require(cop < 3, "boolean operation")?;
        let test = if op == Op::Pset {
            let ab = bits(w, 24, 2);
            self.require(ab < 3, "boolean operation")?;
            let x = self.pred_operand(12);
            let y = self.pred_operand(29);
            boolean(&x, &y, ab)
        } else {
            self.condition(bits(w, 8, 5))?
        };
        let p = self.pred_operand(39);
        self.line(&format!("{{ bool t = {test}; bool q = {p};"));
        if op == Op::Csetp {
            self.set_pred(bits(w, 3, 3), &boolean("t", "q", cop));
            self.set_pred(bits(w, 0, 3), &boolean("!t", "q", cop));
        } else {
            let float = bit(w, 44);
            let one = if float { "0x3f800000u" } else { "0xffffffffu" };
            self.line(&format!(
                "uint v = {} ? {one} : 0u;",
                boolean("t", "q", cop)
            ));
            self.set(bits(w, 0, 8), "v");
            if bit(w, 47) {
                self.cc = true;
                self.line(if float {
                    "cc = v == 0u ? 1u : 0u;"
                } else {
                    "cc = v == 0u ? 1u : 2u;"
                });
            }
        }
        self.line("}");
        Ok(true)
    }

    /// `exec/integer.rs` `execute_integer`; false if `op` is not one of its
    /// instructions.
    pub(super) fn integer(&mut self, op: Op) -> Result<bool, String> {
        if self.flag_set(op)? {
            return Ok(true);
        }
        let form = match op {
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
        let rd = bits(w, 0, 8);
        let constant = matches!(form, Form::C | Form::Rc);
        let mut cc = bit(w, 47);
        // Value-independent checks first.
        match op {
            Op::Lop3R | Op::Lop3I | Op::Lop3C => self.require(
                !bit(
                    w,
                    match form {
                        Form::R => 38,
                        Form::I => 57,
                        _ => 56,
                    },
                ),
                "X",
            )?,
            Op::Iadd3R | Op::Iadd3I | Op::Iadd3C => {
                if form == Form::R {
                    for low in [35, 33, 31] {
                        self.require(bits(w, low, 2) < 3, "half selector")?;
                    }
                }
                let shift = if form == Form::R { bits(w, 37, 2) } else { 0 };
                self.require(shift < 3, "shift")?;
                self.require(!(bit(w, 48) && (shift == 1 || cc)), "X with RS/CC")?;
            }
            Op::XmadR | Op::XmadI | Op::XmadC | Op::XmadRc => {
                let mode = bits(w, 50, if constant { 2 } else { 3 });
                self.require(mode <= 4, "cop mode")?;
                self.require(!bit(w, if constant { 54 } else { 38 }), "X")?;
            }
            Op::ImnmxR | Op::ImnmxI | Op::ImnmxC => {
                self.require(bits(w, 43, 2) == 0, "extended mode")?;
            }
            Op::Psetp => {
                self.require(
                    bits(w, 24, 2) < 3 && bits(w, 45, 2) < 3,
                    "boolean operation",
                )?;
            }
            Op::FloR | Op::FloI | Op::FloC | Op::ImulR | Op::ImulI | Op::ImulC => {
                self.require(!cc, "condition code write")?
            }
            Op::LeaR | Op::LeaI | Op::LeaC | Op::LeaHiR | Op::LeaHiC => {
                self.require(!cc, "condition code write")?;
                self.require(bits(w, 48, 3) == super::PT, "predicate result")?;
                let extended = match op {
                    Op::LeaHiR => 38,
                    Op::LeaHiC => 57,
                    _ => 46,
                };
                self.require(!bit(w, extended), "X")?;
            }
            _ => {}
        }
        let a = self.read(bits(w, 8, 8));
        let b = self.b(form)?;
        let c = self.c(form)?;
        self.line(&format!("{{ uint a = {a}; uint b = {b}; uint c = {c};"));
        match op {
            Op::BfeR | Op::BfeI | Op::BfeC => {
                let x = if bit(w, 40) { "reverse_bits(a)" } else { "a" };
                self.line(&format!(
                    "uint x = {x}; uint start = b & 255u; uint width = min((b >> 8) & 255u, 32u - min(start, 32u));"
                ));
                let extract = if bit(w, 48) {
                    "as_type<uint>(as_type<int>((x >> start) << (32u - width)) >> (32u - width))"
                } else {
                    "((x >> start) & (0xffffffffu >> (32u - width)))"
                };
                self.line(&format!("uint v = width == 0u ? 0u : {extract};"));
            }
            Op::BfiR | Op::BfiI | Op::BfiC | Op::BfiRc => {
                self.line("uint start = b & 255u; uint width = min((b >> 8) & 255u, 32u - min(start, 32u));");
                self.line(
                    "uint mask = width == 0u ? 0u : (0xffffffffu >> (32u - width)) << start;",
                );
                self.line("uint v = width == 0u ? c : ((c & ~mask) | ((a << start) & mask));");
            }
            Op::Lop3R | Op::Lop3I | Op::Lop3C => {
                let lut = bits(w, if form == Form::R { 28 } else { 48 }, 8);
                let terms: Vec<String> = (0..8)
                    .filter(|i| lut & (1 << i) != 0)
                    .map(|i| {
                        format!(
                            "({} & {} & {})",
                            if i & 4 != 0 { "a" } else { "~a" },
                            if i & 2 != 0 { "b" } else { "~b" },
                            if i & 1 != 0 { "c" } else { "~c" }
                        )
                    })
                    .collect();
                let v = if terms.is_empty() {
                    "0u".to_string()
                } else {
                    terms.join(" | ")
                };
                self.line(&format!("uint v = {v};"));
                if form == Form::R {
                    let p = match bits(w, 36, 2) {
                        0 => "false",
                        1 => "true",
                        2 => "(v == 0u)",
                        _ => "(v != 0u)",
                    };
                    self.set_pred(bits(w, 48, 3), p);
                }
            }
            Op::Iadd3R | Op::Iadd3I | Op::Iadd3C => {
                let mut values = ["a".to_string(), "b".to_string(), "c".to_string()];
                if form == Form::R {
                    for (value, low) in values.iter_mut().zip([35, 33, 31]) {
                        match bits(w, low, 2) {
                            1 => *value = format!("({value} & 0xffffu)"),
                            2 => *value = format!("({value} >> 16)"),
                            _ => {}
                        }
                    }
                }
                for (value, low) in values.iter_mut().zip([51, 50, 49]) {
                    if bit(w, low) {
                        *value = format!("(0u - {value})");
                    }
                }
                let [x, y, z] = values;
                let carry = if bit(w, 48) {
                    self.cc = true;
                    "ulong((cc >> 2) & 1u)"
                } else {
                    "0ul"
                };
                self.line(&format!(
                    "uint x = {x}; uint y = {y}; uint z = {z}; ulong s = ulong(x) + ulong(y) + {carry};"
                ));
                let shifted = match if form == Form::R { bits(w, 37, 2) } else { 0 } {
                    1 => "uint(s >> 16)",
                    2 => "(uint(s) << 16)",
                    _ => "uint(s)",
                };
                self.line(&format!("uint h = {shifted}; uint v = h + z;"));
                if cc {
                    self.cc = true;
                    self.line("cc = (v == 0u ? 1u : 0u) | ((v >> 31) != 0u ? 2u : 0u) | (v < h ? 4u : 0u) | (((((~(h ^ z)) & (h ^ v)) >> 31) != 0u || (s >> 32) != 0ul) ? 8u : 0u);");
                    cc = false;
                }
            }
            Op::XmadR | Op::XmadI | Op::XmadC | Op::XmadRc => {
                let y_source = if form == Form::I {
                    format!("{}u", bits(w, 20, 16))
                } else {
                    "b".to_string()
                };
                let x = half("a", bit(w, 53), bit(w, 48));
                let y = half(
                    "bb",
                    form != Form::I && bit(w, if constant { 52 } else { 35 }),
                    bit(w, 49),
                );
                self.line(&format!(
                    "uint bb = {y_source}; uint x = {x}; uint y = {y}; uint product = x * y;"
                ));
                if form != Form::Rc && bit(w, if constant { 55 } else { 36 }) {
                    self.line("product <<= 16;");
                }
                let addend = match bits(w, 50, if constant { 2 } else { 3 }) {
                    0 => "c",
                    1 => "(c & 0xffffu)",
                    2 => "(c >> 16)",
                    3 => "(c - (((x >> 31) + (y >> 31)) << 16))",
                    _ => "(c + (bb << 16))",
                };
                self.line(&format!("uint k = {addend}; uint v = product + k;"));
                if cc {
                    self.cc = true;
                    self.line("cc = (v == 0u ? 1u : 0u) | ((v >> 31) != 0u ? 2u : 0u) | (v < product ? 4u : 0u) | ((((~(product ^ k)) & (product ^ v)) >> 31) != 0u ? 8u : 0u);");
                    cc = false;
                }
                if form != Form::Rc && bit(w, if constant { 56 } else { 37 }) {
                    self.line("v = (v & 0xffffu) | (bb << 16);");
                }
            }
            Op::ImnmxR | Op::ImnmxI | Op::ImnmxC => {
                let less = if bit(w, 48) {
                    "(as_type<int>(a) < as_type<int>(b))"
                } else {
                    "(a < b)"
                };
                let p = self.pred_operand(39);
                self.line(&format!("uint v = ({less} == {p}) ? a : b;"));
            }
            Op::IcmpR | Op::IcmpI | Op::IcmpC | Op::IcmpRc => {
                cc = false;
                let test = if bit(w, 48) {
                    integer_compare(bits(w, 49, 3), "as_type<int>(c)", "0")
                } else {
                    integer_compare(bits(w, 49, 3), "c", "0u")
                };
                self.line(&format!("uint v = {test} ? a : b;"));
            }
            Op::FcmpR | Op::FcmpI | Op::FcmpC | Op::FcmpRc => {
                // Bit 47 is FTZ here, not a condition code write.
                let x = flush_operation("as_type<float>(c)", cc);
                cc = false;
                self.line(&format!(
                    "float x = {x}; float y = 0.0f; uint v = {} ? a : b;",
                    float_compare(bits(w, 48, 4))
                ));
            }
            Op::PopcR | Op::PopcI | Op::PopcC => {
                let x = if bit(w, 40) { "~b" } else { "b" };
                self.line(&format!("uint v = popcount({x});"));
            }
            Op::FloR | Op::FloI | Op::FloC => {
                let x = if bit(w, 40) { "~b" } else { "b" };
                self.line(&format!("uint x = {x};"));
                if bit(w, 48) {
                    self.line("x = (x >> 31) != 0u ? ~x : x;");
                }
                let index = if bit(w, 41) { "z" } else { "31u - z" };
                self.line(&format!(
                    "uint z = clz(x); uint v = z == 32u ? 0xffffffffu : {index};"
                ));
            }
            Op::ImulR | Op::ImulI | Op::ImulC => {
                self.line(if !bit(w, 39) {
                    "uint v = a * b;"
                } else if bit(w, 40) {
                    "uint v = as_type<uint>(mulhi(as_type<int>(a), as_type<int>(b)));"
                } else {
                    "uint v = mulhi(a, b);"
                });
            }
            Op::LeaR | Op::LeaI | Op::LeaC => {
                let offset = if bit(w, 45) { "(0u - a)" } else { "a" };
                self.line(&format!("uint v = b + ({offset} << {}u);", bits(w, 39, 5)));
            }
            Op::LeaHiR | Op::LeaHiC => {
                let (base, high, scale, negate) = if op == Op::LeaHiR {
                    ("b", "c", bits(w, 28, 5), bit(w, 37))
                } else {
                    ("c", "b", bits(w, 51, 5), bit(w, 56))
                };
                self.line(&format!("ulong o = ulong(a) | (ulong({high}) << 32);"));
                if negate {
                    self.line("o = 0ul - o;");
                }
                self.line(&format!("uint v = {base} + uint(o >> {}u);", 32 - scale));
            }
            Op::R2pR | Op::R2pI | Op::R2pC => {
                self.line(&format!(
                    "uint value = (a >> {}u) & 0xffu; uint mask = b & 0xffu;",
                    bits(w, 41, 2) * 8
                ));
                if bit(w, 40) {
                    self.cc = true;
                    self.line("cc = (cc & ~mask & 15u) | (value & mask & 15u);");
                } else {
                    for i in 0..7 {
                        self.predicates.insert(i);
                        self.line(&format!(
                            "if (((mask >> {i}) & 1u) != 0u) {{ p{i} = ((value >> {i}) & 1u) != 0u; }}"
                        ));
                    }
                }
                self.line("}");
                return Ok(true);
            }
            Op::P2rR | Op::P2rI | Op::P2rC => {
                cc = false;
                let value = if bit(w, 40) {
                    self.cc = true;
                    "cc".to_string()
                } else {
                    let mut terms = vec!["0x80u".to_string()];
                    for i in 0..7 {
                        self.predicates.insert(i);
                        terms.push(format!("(p{i} ? {}u : 0u)", 1 << i));
                    }
                    terms.join(" | ")
                };
                let shift = bits(w, 41, 2) * 8;
                self.line(&format!(
                    "uint mask = (b & 255u) << {shift}u; uint v = (a & ~mask) | ((({value}) << {shift}u) & mask);"
                ));
            }
            Op::Psetp => {
                let x = self.pred_operand(12);
                let y = self.pred_operand(29);
                let test = boolean(&x, &y, bits(w, 24, 2));
                let p = self.pred_operand(39);
                self.line(&format!("bool t = {test}; bool q = {p};"));
                let cop = bits(w, 45, 2);
                self.set_pred(bits(w, 3, 3), &boolean("t", "q", cop));
                self.set_pred(bits(w, 0, 3), &boolean("!t", "q", cop));
                self.line("}");
                return Ok(true);
            }
            _ => unreachable!("every op is handled"),
        }
        self.set(rd, "v");
        if cc {
            self.write_cc(false);
        }
        self.line("}");
        Ok(true)
    }
}
