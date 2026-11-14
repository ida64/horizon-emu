//! Translation of pixel programs to Metal Shading Language, for the Metal
//! renderer ([`crate::maxwell::metal`]).
//!
//! The translation mirrors the interpreter (`shader::exec`) instruction by
//! instruction: each register is a `uint` variable, each predicate a
//! `bool`, and every operation is written out with the same operand
//! modifiers, flushes and special cases as the interpreter applies, so that
//! both renderers compute the same values (up to the rounding of
//! transcendental functions and of texture filtering, which the Metal GPU
//! does its own way). Contraction of multiplies and adds into fused
//! operations is disabled; `FFMA` uses `fma`.
//!
//! What is translated is a subset: straight-line programs (no branches
//! other than the compiler's terminating branch-to-self after the final
//! `EXIT`), without condition codes, using the instructions and forms the
//! system applets' pixel programs use. Anything else is refused with the
//! reason, and the draw is left to the interpreter.

use std::collections::BTreeSet;
use std::fmt::Write;

use super::Program;
use super::decode::Op;
use super::exec::{bit, bits, branch_target, is_branch_to_self, sext};
use super::header::ShaderType;

const RZ: u32 = 255;
const PT: u32 = 7;

/// A pixel program in MSL, and what a draw must supply for it.
#[derive(Debug, Clone)]
pub struct Translated {
    /// The statements of the program body, then the output gathering, as
    /// the inside of the fragment function (see [`Translated::source`]).
    body: String,
    /// Attribute words `IPA` reads, in plane order: plane `k` of a
    /// primitive is the interpolation plane of word `ipa_words[k]`.
    pub ipa_words: Vec<u32>,
    /// Bound-texture handle offsets (bytes in the texture-handle constant
    /// buffer) the program samples, in texture/sampler slot order.
    pub textures: Vec<u32>,
    /// Constant buffer words read with static offsets: (slot, byte offset).
    pub constants: BTreeSet<(u32, u32)>,
    /// Which texture slots use implicit level of detail (and so the
    /// sampler's bias).
    pub implicit_lod: Vec<bool>,
}

/// Register value expressions.
fn r(index: u32) -> String {
    if index >= RZ {
        "0u".into()
    } else {
        format!("r{index}")
    }
}

fn f(index: u32) -> String {
    format!("as_type<float>({})", r(index))
}

/// A predicate operand: index at `low..low + 3`, negation at `low + 3`.
fn pred(word: u64, low: u32) -> String {
    let index = bits(word, low, 3);
    let negate = bit(word, low + 3);
    let value = if index == PT {
        "true".to_string()
    } else {
        format!("p{index}")
    };
    if negate { format!("(!{value})") } else { value }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Form {
    R,
    I,
    F,
    C,
    Rc,
    Imm32,
}

struct Writer<'a> {
    program: &'a Program,
    out: String,
    registers: BTreeSet<u32>,
    predicates: BTreeSet<u32>,
    ipa_words: Vec<u32>,
    textures: Vec<u32>,
    implicit_lod: Vec<bool>,
    constants: BTreeSet<(u32, u32)>,
    word: u64,
    pc: u32,
}

impl Writer<'_> {
    fn refuse<T>(&self, what: &str) -> Result<T, String> {
        let op = self
            .program
            .code
            .get((self.pc / 8) as usize)
            .copied()
            .flatten()
            .map_or("?", |inst| inst.op.mnemonic());
        Err(format!("{op} {what} {:#018x} at {:#x}", self.word, self.pc))
    }

    fn require(&self, ok: bool, what: &str) -> Result<(), String> {
        if ok { Ok(()) } else { self.refuse(what) }
    }

    fn read(&mut self, index: u32) -> String {
        if index < RZ {
            self.registers.insert(index);
        }
        r(index)
    }

    fn readf(&mut self, index: u32) -> String {
        if index < RZ {
            self.registers.insert(index);
        }
        f(index)
    }

    /// `rd = value;` for a `uint` expression.
    fn set(&mut self, index: u32, value: &str) {
        if index < RZ {
            self.registers.insert(index);
            let _ = writeln!(self.out, "    r{index} = {value};");
        } else {
            let _ = writeln!(self.out, "    (void)({value});");
        }
    }

    fn set_pred(&mut self, index: u32, value: &str) {
        if index != PT {
            self.predicates.insert(index);
            let _ = writeln!(self.out, "    p{index} = {value};");
        }
    }

    fn constant(&mut self) -> String {
        let slot = bits(self.word, 34, 5);
        let offset = bits(self.word, 20, 14) * 4;
        self.constants.insert((slot, offset));
        format!("cb{slot}[{}]", offset / 4)
    }

    /// Operand B as a `uint` expression.
    fn b(&mut self, form: Form) -> String {
        let w = self.word;
        match form {
            Form::R => self.read(bits(w, 20, 8)),
            Form::I => format!("{:#x}u", imm20(w) as u32),
            Form::F => format!("{:#x}u", (imm20(w) as u32) << 12),
            Form::C => self.constant(),
            Form::Rc => self.read(bits(w, 39, 8)),
            Form::Imm32 => format!("{:#x}u", (w >> 20) as u32),
        }
    }

    /// Operand C as a `uint` expression.
    fn c(&mut self, form: Form) -> String {
        match form {
            Form::Rc => self.constant(),
            _ => self.read(bits(self.word, 39, 8)),
        }
    }

    fn instruction(&mut self, op: Op) -> Result<(), String> {
        let w = self.word;
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
            Op::LopR => ("LOP", Form::R),
            Op::LopI => ("LOP", Form::I),
            Op::LopC => ("LOP", Form::C),
            Op::Lop32i => ("LOP32I", Form::Imm32),
            Op::Mufu => ("MUFU", Form::R),
            Op::Ipa => return self.ipa(),
            Op::Texs => return self.texs(),
            Op::Kil => {
                self.require(bits(w, 0, 5) == 15, "condition code test")?;
                let _ = writeln!(self.out, "    discard_fragment();");
                return Ok(());
            }
            Op::Nop | Op::Depbar => return Ok(()),
            _ => return self.refuse("not translated"),
        };
        let rd = bits(w, 0, 8);
        let ra = bits(w, 8, 8);
        let wide = form == Form::Imm32;
        match name {
            "MOV" | "MOV32I" => {
                let mask = bits(w, if name == "MOV" { 39 } else { 12 }, 4);
                self.require(mask == 15, "partial quad mask")?;
                let b = self.b(form);
                self.set(rd, &b);
            }
            "FADD" | "FADD32I" => {
                if !wide {
                    self.require(bits(w, 39, 2) == 0, "rounding mode")?;
                }
                self.require(!bit(w, if wide { 52 } else { 47 }), "condition code")?;
                let ftz = bit(w, if wide { 55 } else { 44 });
                let sat = !wide && bit(w, 50);
                let a = self.readf(ra);
                let b = self.b(form);
                let x = flush(
                    &modify(
                        &a,
                        bit(w, if wide { 54 } else { 46 }),
                        bit(w, if wide { 56 } else { 48 }),
                    ),
                    ftz,
                );
                let y = flush(
                    &modify(
                        &format!("as_type<float>({b})"),
                        bit(w, if wide { 57 } else { 49 }),
                        bit(w, if wide { 53 } else { 45 }),
                    ),
                    ftz,
                );
                let v = saturate(&flush(&format!("({x} + {y})"), ftz), sat);
                self.set(rd, &format!("as_type<uint>({v})"));
            }
            "FMUL" | "FMUL32I" => {
                let round = if wide { 0 } else { bits(w, 39, 2) };
                let mode = bits(w, if wide { 53 } else { 44 }, 2);
                self.require(mode <= 2, "FMZ encoding")?;
                let scale = if wide { 0 } else { bits(w, 41, 3) };
                self.require(scale < 7, "scale")?;
                self.require(!bit(w, if wide { 52 } else { 47 }), "condition code")?;
                let sat = bit(w, if wide { 55 } else { 50 });
                let a = self.readf(ra);
                let b = self.b(form);
                let x = flush(&modify(&a, false, !wide && bit(w, 48)), mode == 1);
                let y = flush(&format!("as_type<float>({b})"), mode == 1);
                let _ = writeln!(self.out, "    {{ float x = {x}; float y = {y};");
                // The interpreter rounds the exact (binary64) product times
                // the scale once, in direction `round` (`exec/alu.rs`
                // `round_f64`); see `mul_round` in the prelude.
                let product = if round == 0 && scale == 0 {
                    "(x * y)".to_string()
                } else {
                    let factor = ["1.0f", "0.5f", "0.25f", "0.125f", "8.0f", "4.0f", "2.0f"]
                        [scale as usize];
                    if round == 0 {
                        format!("((x * y) * {factor})")
                    } else {
                        format!("mul_round(x, y, {factor}, {round}u)")
                    }
                };
                let product = if mode == 2 {
                    format!(
                        "((x == 0.0f || y == 0.0f) ? as_type<float>((as_type<uint>(x) ^ as_type<uint>(y)) & 0x80000000u) : {product})"
                    )
                } else {
                    product
                };
                let v = saturate(&flush(&product, mode == 1), sat);
                self.set(rd, &format!("as_type<uint>({v})"));
                let _ = writeln!(self.out, "    }}");
            }
            "FFMA" | "FFMA32I" => {
                if !wide {
                    self.require(bits(w, 51, 2) == 0, "rounding mode")?;
                }
                let mode = bits(w, 53, 2);
                self.require(mode <= 2, "FMZ encoding")?;
                self.require(!bit(w, if wide { 52 } else { 47 }), "condition code")?;
                let sat = bit(w, if wide { 55 } else { 50 });
                let a = self.readf(ra);
                let b = self.b(form);
                let c = if wide {
                    self.readf(rd)
                } else {
                    let c = self.c(form);
                    format!("as_type<float>({c})")
                };
                let x = flush(
                    &modify(&a, false, bit(w, if wide { 56 } else { 48 })),
                    mode == 1,
                );
                let y = flush(&format!("as_type<float>({b})"), mode == 1);
                let z = flush(
                    &modify(&c, false, bit(w, if wide { 57 } else { 49 })),
                    mode == 1,
                );
                let _ = writeln!(
                    self.out,
                    "    {{ float x = {x}; float y = {y}; float z = {z};"
                );
                let v = if mode == 2 {
                    "((x == 0.0f || y == 0.0f) ? z : fma(x, y, z))".to_string()
                } else {
                    "fma(x, y, z)".to_string()
                };
                let v = saturate(&flush(&v, mode == 1), sat);
                self.set(rd, &format!("as_type<uint>({v})"));
                let _ = writeln!(self.out, "    }}");
            }
            "FMNMX" => {
                self.require(!bit(w, 47), "condition code")?;
                let ftz = bit(w, 44);
                let a = self.readf(ra);
                let b = self.b(form);
                let x = flush(&modify(&a, bit(w, 46), bit(w, 48)), ftz);
                let y = flush(
                    &modify(&format!("as_type<float>({b})"), bit(w, 49), bit(w, 45)),
                    ftz,
                );
                let p = self.pred_operand(39);
                self.set(
                    rd,
                    &format!("as_type<uint>({p} ? fmin({x}, {y}) : fmax({x}, {y}))"),
                );
            }
            "MUFU" => {
                let a = self.readf(ra);
                let x = modify(&a, bit(w, 46), bit(w, 48));
                let v = match bits(w, 20, 4) {
                    0 => format!("precise::cos({x})"),
                    1 => format!("precise::sin({x})"),
                    2 => format!("precise::exp2({x})"),
                    3 => format!("precise::log2({x})"),
                    4 => format!("(1.0f / {x})"),
                    5 => format!("(1.0f / precise::sqrt({x}))"),
                    8 => format!("precise::sqrt({x})"),
                    _ => return self.refuse("suboperation"),
                };
                let v = saturate(&v, bit(w, 50));
                self.set(rd, &format!("as_type<uint>({v})"));
            }
            "LOP" | "LOP32I" => {
                self.require(!bit(w, if wide { 57 } else { 43 }), "X")?;
                self.require(!bit(w, if wide { 52 } else { 47 }), "condition code")?;
                let a = self.read(ra);
                let b = self.b(form);
                let a = if bit(w, if wide { 55 } else { 39 }) {
                    format!("(~{a})")
                } else {
                    a
                };
                let b = if bit(w, if wide { 56 } else { 40 }) {
                    format!("(~{b})")
                } else {
                    b
                };
                let v = match bits(w, if wide { 53 } else { 41 }, 2) {
                    0 => format!("({a} & {b})"),
                    1 => format!("({a} | {b})"),
                    2 => format!("({a} ^ {b})"),
                    _ => b,
                };
                let _ = writeln!(self.out, "    {{ uint v = {v};");
                if !wide {
                    let p = match bits(w, 44, 2) {
                        0 => "false",
                        1 => "true",
                        2 => "(v == 0u)",
                        _ => "(v != 0u)",
                    };
                    self.set_pred(bits(w, 48, 3), p);
                }
                self.set(rd, "v");
                let _ = writeln!(self.out, "    }}");
            }
            _ => return self.refuse("not translated"),
        }
        Ok(())
    }

    /// A predicate source operand for a value use.
    fn pred_operand(&mut self, low: u32) -> String {
        let index = bits(self.word, low, 3);
        if index != PT {
            self.predicates.insert(index);
        }
        pred(self.word, low)
    }

    /// `IPA`, as the interpreter evaluates it: the attribute's plane at the
    /// pixel center, optionally multiplied by a register.
    fn ipa(&mut self) -> Result<(), String> {
        let w = self.word;
        let mode = bits(w, 54, 2);
        self.require(mode <= 1, "interpolation mode")?;
        self.require(!bit(w, 38), "IDX")?;
        self.require(bits(w, 52, 2) == 0, "sample mode")?;
        self.require(bits(w, 47, 4) == PT, "predicate operand")?;
        self.require(bits(w, 39, 8) == RZ, "sample-offset register")?;
        let attribute = bits(w, 28, 10);
        self.require(attribute.is_multiple_of(4), "unaligned attribute")?;
        self.require(attribute != 0x3fc, "front-facing attribute")?;
        let word = attribute / 4;
        let plane = match self.ipa_words.iter().position(|&x| x == word) {
            Some(index) => index,
            None => {
                self.ipa_words.push(word);
                self.ipa_words.len() - 1
            }
        };
        let (rd, rb) = (bits(w, 0, 8), bits(w, 20, 8));
        let mut value = format!("plane(planes, plane_base + {plane}u, px, py)");
        if mode == 1 {
            let b = self.readf(rb);
            value = format!("({value} * {b})");
        }
        let value = saturate(&value, bit(w, 51));
        self.set(rd, &format!("as_type<uint>({value})"));
        Ok(())
    }

    /// `TEXS` with a 2D target: implicit LOD (1), level zero (2) or an
    /// explicit level (3).
    fn texs(&mut self) -> Result<(), String> {
        let w = self.word;
        let encoding = bits(w, 53, 4);
        self.require(matches!(encoding, 1..=3), "target encoding")?;
        let a = bits(w, 8, 8);
        let b = bits(w, 20, 8);
        let d = bits(w, 0, 8);
        let e = bits(w, 28, 8);
        let mask_code = bits(w, 50, 3) as usize;
        let mask = if e == RZ {
            [1u8, 2, 4, 8, 3, 9, 10, 12][mask_code]
        } else {
            *[7u8, 11, 13, 14, 15]
                .get(mask_code)
                .ok_or_else(|| self.refuse::<()>("component mask").unwrap_err())?
        };
        if encoding >= 3 {
            self.require(a == RZ || a.is_multiple_of(2), "unaligned coordinate pair")?;
        }
        if mask.count_ones() > 1 {
            self.require(
                d == RZ || d.is_multiple_of(2),
                "unaligned first destination",
            )?;
        }
        if mask.count_ones() > 3 {
            self.require(
                e == RZ || e.is_multiple_of(2),
                "unaligned second destination",
            )?;
        }
        let handle = bits(w, 36, 13) * 4;
        let slot = match self.textures.iter().position(|&h| h == handle) {
            Some(slot) => slot,
            None => {
                self.textures.push(handle);
                self.implicit_lod.push(false);
                self.textures.len() - 1
            }
        };
        let (s, t) = match encoding {
            1 | 2 => (self.readf(a), self.readf(b)),
            _ => (self.readf(a), self.readf(a + 1)),
        };
        let sample = match encoding {
            1 => {
                self.implicit_lod[slot] = true;
                format!("t{slot}.sample(s{slot}, float2({s}, {t}), bias(lod_bias[{slot}]))")
            }
            2 => format!("t{slot}.sample(s{slot}, float2({s}, {t}), level(0.0f))"),
            _ => {
                let lod = self.readf(b);
                format!("t{slot}.sample(s{slot}, float2({s}, {t}), level({lod}))")
            }
        };
        let _ = writeln!(self.out, "    {{ float4 v = {sample};");
        let mut index = 0;
        for c in 0..4 {
            if mask & (1 << c) == 0 {
                continue;
            }
            let dest = if index < 2 {
                if d == RZ { RZ } else { d + index }
            } else if e == RZ {
                RZ
            } else {
                e + index - 2
            };
            let component = ["x", "y", "z", "w"][c];
            self.set(dest, &format!("as_type<uint>(v.{component})"));
            index += 1;
        }
        let _ = writeln!(self.out, "    }}");
        Ok(())
    }
}

fn imm20(word: u64) -> i32 {
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

fn saturate(value: &str, enabled: bool) -> String {
    if enabled {
        format!("sat({value})")
    } else {
        value.to_string()
    }
}

const PRELUDE: &str = r#"#include <metal_stdlib>
using namespace metal;
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

static inline float ftz(float v) {
    return (v != 0.0f && fabs(v) < 1.17549435e-38f) ? copysign(0.0f, v) : v;
}
static inline float sat(float v) { return isnan(v) ? 0.0f : clamp(v, 0.0f, 1.0f); }
/// x * y * scale (a power of two) rounded once in direction `mode` (1 down,
/// 2 up, 3 to zero), as the interpreter's `round_f64` of the exact
/// product: round to nearest, then step one ulp when the exact error
/// `fma(x, y, -p)` shows the nearest value lies on the wrong side. Exact
/// unless the product or its error is subnormal, or the scaled product
/// leaves the normal range [assumption: not met in UI programs].
static inline float mul_round(float x, float y, float scale, uint mode) {
    float p = x * y;
    float e = fma(x, y, -p);
    float v = p * scale;
    bool down = mode == 1u || (mode == 3u && v > 0.0f);
    bool up = mode == 2u || (mode == 3u && v < 0.0f);
    if (down && e < 0.0f) { v = nextafter(v, as_type<float>(0xff800000u)); }
    else if (up && e > 0.0f) { v = nextafter(v, as_type<float>(0x7f800000u)); }
    return v;
}
/// Plane `k` (dx, dy, base) at (x, y), summed in the interpreter's order.
static inline float plane(const device float* planes, uint k, float x, float y) {
    uint i = k * 3u;
    return (planes[i + 2] + planes[i] * x) + planes[i + 1] * y;
}

struct Vertex { float4 position [[position]]; };

vertex Vertex vertex_main(uint id [[vertex_id]], const device float4* positions [[buffer(0)]]) {
    Vertex out;
    out.position = positions[id];
    return out;
}
"#;

impl Translated {
    /// The complete MSL source: the vertex function `vertex_main` (window
    /// positions from buffer 0) and the fragment function
    /// `fragment_main`, whose color attachment `i` receives the pixel
    /// program's output target `outputs[i]` (unwritten components 0).
    ///
    /// Fragment bindings: buffer 0 the primitive planes (per primitive,
    /// `ipa_words.len()` planes of (dx, dy, base)), buffer 1 the texture
    /// LOD biases, buffer `2 + slot` constant buffer `slot`; texture and
    /// sampler `i` for `textures[i]`.
    pub fn source(&self, program: &Program, outputs: &[usize]) -> String {
        let mut s = String::from(PRELUDE);
        let _ = writeln!(s, "struct Output {{");
        for (attachment, _) in outputs.iter().enumerate() {
            let _ = writeln!(s, "    float4 c{attachment} [[color({attachment})]];");
        }
        let _ = writeln!(s, "}};");
        let mut parameters = vec![
            "float4 position [[position]]".to_string(),
            "uint primitive [[primitive_id]]".to_string(),
            "const device float* planes [[buffer(0)]]".to_string(),
            "const device float* lod_bias [[buffer(1)]]".to_string(),
        ];
        let slots: BTreeSet<u32> = self.constants.iter().map(|&(slot, _)| slot).collect();
        for slot in &slots {
            parameters.push(format!(
                "const device uint* cb{slot} [[buffer({})]]",
                2 + slot
            ));
        }
        for i in 0..self.textures.len() {
            parameters.push(format!("texture2d<float> t{i} [[texture({i})]]"));
            parameters.push(format!("sampler s{i} [[sampler({i})]]"));
        }
        let _ = writeln!(
            s,
            "fragment Output fragment_main({}) {{",
            parameters.join(", ")
        );
        let _ = writeln!(
            s,
            "    const float px = position.x; const float py = position.y;\n    const uint plane_base = primitive * {}u;",
            self.ipa_words.len().max(1)
        );
        s.push_str(&self.body);
        let _ = writeln!(s, "    Output out;");
        // Outputs as the interpreter gathers them: each target with a
        // component mask takes the next four registers from R0.
        let header = &program.header;
        let mut base = 0u32;
        let mut registers = [None; 8];
        for (target, slot) in registers.iter_mut().enumerate() {
            let mask = header.ps_color_mask(target);
            if mask == 0 {
                continue;
            }
            *slot = Some((base, mask));
            base += 4;
        }
        for (attachment, &target) in outputs.iter().enumerate() {
            let value = match registers.get(target).copied().flatten() {
                Some((base, mask)) => {
                    let components: Vec<String> = (0..4)
                        .map(|c| {
                            if mask & (1 << c) != 0 {
                                f(base + c)
                            } else {
                                "0.0f".into()
                            }
                        })
                        .collect();
                    format!("float4({})", components.join(", "))
                }
                None => "float4(0.0f)".into(),
            };
            let _ = writeln!(s, "    out.c{attachment} = {value};");
        }
        let _ = writeln!(s, "    return out;\n}}");
        s
    }
}

/// Translates a pixel program, or says why it cannot be.
pub fn translate(program: &Program) -> Result<Translated, String> {
    let header = &program.header;
    if header.shader_type() != ShaderType::Pixel {
        return Err("not a pixel program".into());
    }
    if header.ps_writes_sample_mask() {
        return Err("sample-mask output".into());
    }
    if header.ps_writes_depth() {
        return Err("depth output".into());
    }
    let mut writer = Writer {
        program,
        out: String::new(),
        registers: BTreeSet::new(),
        predicates: BTreeSet::new(),
        ipa_words: Vec::new(),
        textures: Vec::new(),
        implicit_lod: Vec::new(),
        constants: BTreeSet::new(),
        word: 0,
        pc: 8,
    };
    // Straight-line code from the entry to the first unconditional EXIT.
    let mut pc = 8u32;
    loop {
        let Some(inst) = program.code.get((pc / 8) as usize).copied().flatten() else {
            return Err(format!("execution reaches {pc:#x} without EXIT"));
        };
        writer.word = inst.word;
        writer.pc = pc;
        let w = inst.word;
        let guard_index = bits(w, 16, 3);
        let guard_negated = bit(w, 19);
        match inst.op {
            Op::Exit => {
                writer.require(!bit(w, 5), "KEEPREFCOUNT")?;
                writer.require(bits(w, 0, 5) == 15, "condition code test")?;
                if guard_index == PT && !guard_negated {
                    break;
                }
                if !(guard_index == PT && guard_negated) {
                    writer.predicates.insert(guard_index);
                    let _ = writeln!(writer.out, "    if ({}) {{ break; }}", pred(w, 16));
                }
            }
            Op::Bra => {
                if is_branch_to_self(w, pc) {
                    return Err(format!("terminating branch at {pc:#x} is reached"));
                }
                let _ = branch_target(w, pc);
                return writer.refuse("branch");
            }
            op => {
                if guard_index == PT && guard_negated {
                    // Never executes.
                } else if guard_index == PT {
                    writer.instruction(op)?;
                } else {
                    writer.predicates.insert(guard_index);
                    let _ = writeln!(writer.out, "    if ({}) {{", pred(w, 16));
                    writer.instruction(op)?;
                    let _ = writeln!(writer.out, "    }}");
                }
            }
        }
        pc += 8;
        if pc.is_multiple_of(32) {
            pc += 8;
        }
    }
    let mut body = String::new();
    // Registers and predicates start at zero, as in the interpreter.
    let declared: BTreeSet<u32> = writer
        .registers
        .iter()
        .copied()
        .chain(output_registers(program))
        .collect();
    for index in &declared {
        let _ = writeln!(body, "    uint r{index} = 0u;");
    }
    for index in &writer.predicates {
        let _ = writeln!(body, "    bool p{index} = false;");
    }
    let _ = writeln!(body, "    do {{");
    body.push_str(&writer.out);
    let _ = writeln!(body, "    }} while (false);");
    Ok(Translated {
        body,
        ipa_words: writer.ipa_words,
        textures: writer.textures,
        constants: writer.constants,
        implicit_lod: writer.implicit_lod,
    })
}

/// Registers the output gathering reads.
fn output_registers(program: &Program) -> impl Iterator<Item = u32> {
    let count = (0..8)
        .filter(|&target| program.header.ps_color_mask(target) != 0)
        .count() as u32;
    0..count * 4
}
