//! Translation of pixel programs to Metal Shading Language, for the Metal
//! renderer ([`crate::maxwell::metal`]).
//!
//! # Reference semantics
//!
//! The translation mirrors the interpreter (`exec`) instruction by
//! instruction: each register is a `uint` variable, each predicate a
//! `bool`, the condition code a `uint` of the interpreter's flag bits, and
//! every operation is written out with the same operand modifiers,
//! flushes, saturations and special cases as the interpreter applies, so
//! that both renderers compute the same values. Where the result depends on
//! the host's arithmetic:
//!
//! * The source is compiled with `math_mode(safe)` and `fp contract(off)`:
//!   no multiply-add contraction, no reassociation, NaN and signed-zero
//!   semantics kept. `FFMA` uses `fma` (fused, one rounding, like the
//!   interpreter's `mul_add`); `FMUL`/`FADD` are single IEEE operations.
//! * `FMUL` rounding modes and scales: the interpreter rounds the exact
//!   binary64 product once in the requested direction. The translation
//!   rounds to nearest and corrects by one ulp using the sign of the exact
//!   error `fma(x, y, -p)`. This is exact unless the product (or its error)
//!   is subnormal or the scaled product leaves the normal range
//!   [assumption: such operands do not occur in UI programs].
//! * `MUFU` uses the `precise::` functions; transcendental results may
//!   differ from the host libm by a few ulp (neither is GM20B's MUFU).
//!   Reciprocal and square root are correctly rounded on both sides.
//! * `I2F` with a directed rounding mode corrects the nearest conversion
//!   by one ulp after an exact integer comparison.
//! * Texture sampling is the GPU's own filtering (and implicit LOD from
//!   the GPU's quad derivatives), not [`crate::maxwell::texture`]'s.
//!
//! # Control flow
//!
//! A program without jumps (only `EXIT`/`KIL`, possibly guarded, leave
//! the straight line) is emitted as straight-line code. Any other program
//! keeps the interpreter's model: a program counter, a per-lane
//! reconvergence stack (`SSY`/`PBK`/`PCNT` push, `SYNC`/`BRK`/`CONT` pop)
//! and a dispatch loop over a `switch` on the program counter whose `case`
//! labels are the possible jump targets (`flow`); consecutive
//! instructions fall through from case to case, so one iteration runs a
//! lane from a jump target to the next jump. Each iteration the 2x2 quad
//! runs only its lanes at the smallest program counter (two
//! `quad_shuffle_xor` steps; `quad_min` made Metal's compiler take minutes
//! on the larger programs), as
//! the interpreter's lockstep schedule does, so lanes that diverged meet
//! again at their reconvergence point and implicit-LOD texture samples
//! there see the whole quad. (Within a run a lane does not wait for a
//! quad neighbour that jumped into the middle of it, which the
//! interpreter would; texture derivatives across such lanes are undefined
//! on hardware.) This rather than reconstructed `if`/`while` nesting:
//! it is correct for any graph, including the data-dependent targets of
//! `BRX` and lanes that leave an `SSY` region without its `SYNC`.
//!
//! The interpreter stops a program after `exec::STEP_LIMIT`
//! instructions; the dispatch loop stops a lane after `STEP_LIMIT`
//! iterations (a draw the interpreter would refuse; the GPU must not
//! hang).
//!
//! # What a draw supplies
//!
//! See [`Translated::source`] for the bindings. Anything the translation
//! does not reproduce exactly as the interpreter executes it (an
//! instruction, a modifier, a texture target) is refused with the reason,
//! and the draw is left to the interpreter.

mod flow;
mod ops;
#[cfg(test)]
mod tests;
mod texture;

use std::collections::BTreeSet;
use std::fmt::Write;

use super::Program;
use super::decode::Op;
use super::exec::{bit, bits, branch_target, is_branch_to_self, sext};
use super::header::ShaderType;

const RZ: u32 = 255;
/// Entries of a lane's reconvergence stack (the interpreter's
/// `STACK_DEPTH`).
const STACK_DEPTH: u32 = 32;
const PT: u32 = 7;

/// Constant buffer slots a stage has (`Env::const_buffers`).
const SLOTS: u32 = 18;

/// A pixel program in MSL, and what a draw must supply for it.
#[derive(Debug, Clone)]
pub struct Translated {
    /// Declarations, the program body and nothing else: the inside of the
    /// fragment function up to the output gathering (see
    /// [`Translated::source`]).
    body: String,
    /// Attribute words `IPA` reads, in plane order: plane `k` of a
    /// primitive is the interpolation plane of word `ipa_words[k]`.
    pub ipa_words: Vec<u32>,
    /// Bound-texture handle offsets (bytes in the texture-handle constant
    /// buffer, `TEXTURE_BUFFER_INDEX`) the program samples, in
    /// texture/sampler slot order. All are 2D textures.
    pub textures: Vec<u32>,
    /// Constant buffer words read with static offsets: (slot, byte offset).
    pub constants: BTreeSet<(u32, u32)>,
    /// Constant buffer slots read at register-dependent offsets (`LDC`):
    /// the draw binds the whole bound range of these.
    pub indirect: BTreeSet<u32>,
    /// Which texture slots are sampled with implicit level of detail (and
    /// so with the sampler's LOD bias).
    pub implicit_lod: Vec<bool>,
    /// The constant buffer 1 content the targets of `BRX` (indirect
    /// branches through a jump table there) were taken from; `None` if the
    /// program has none. The translation is valid only for draws whose
    /// constant buffer 1 has this content.
    pub constant_buffer_1: Option<Vec<u8>>,
}

/// Register value expressions.
fn r(index: u32) -> String {
    if index >= RZ {
        "0u".into()
    } else {
        format!("r{index}")
    }
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

/// Where operand B (and C) of an instruction come from, as the
/// interpreter's `Form`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Form {
    /// B register 20-27, C register 39-46.
    R,
    /// B the sign-extended 20-bit integer immediate.
    I,
    /// B the 20-bit float immediate (top bits of an f32).
    F,
    /// B `c[34-38][20-33 * 4]`.
    C,
    /// B register 39-46, C `c[34-38][20-33 * 4]`.
    Rc,
    /// B the 32-bit immediate at bits 20-51.
    Imm32,
}

struct Writer<'a> {
    program: &'a Program,
    out: String,
    registers: BTreeSet<u32>,
    predicates: BTreeSet<u32>,
    /// The program reads or writes the condition code.
    cc: bool,
    ipa_words: Vec<u32>,
    textures: Vec<u32>,
    implicit_lod: Vec<bool>,
    constants: BTreeSet<(u32, u32)>,
    indirect: BTreeSet<u32>,
    word: u64,
    pc: u32,
}

impl Writer<'_> {
    fn refuse<T>(&self, what: &str) -> Result<T, String> {
        Err(flow::describe(self.program, self.pc, what))
    }

    fn require(&self, ok: bool, what: &str) -> Result<(), String> {
        if ok { Ok(()) } else { self.refuse(what) }
    }

    fn line(&mut self, text: &str) {
        let _ = writeln!(self.out, "        {text}");
    }

    /// A register read, as a `uint` expression.
    fn read(&mut self, index: u32) -> String {
        if index < RZ {
            self.registers.insert(index);
        }
        r(index)
    }

    /// `rd = value;` for a `uint` expression (writes to RZ are dropped).
    fn set(&mut self, index: u32, value: &str) {
        if index < RZ {
            self.registers.insert(index);
            self.line(&format!("r{index} = {value};"));
        }
    }

    fn set_pred(&mut self, index: u32, value: &str) {
        if index != PT {
            self.predicates.insert(index);
            self.line(&format!("p{index} = {value};"));
        }
    }

    /// A predicate source operand.
    fn pred_operand(&mut self, low: u32) -> String {
        let index = bits(self.word, low, 3);
        if index != PT {
            self.predicates.insert(index);
        }
        pred(self.word, low)
    }

    /// The `c[slot][offset]` operand of `C`/`Rc` forms.
    fn constant(&mut self) -> Result<String, String> {
        let slot = bits(self.word, 34, 5);
        let offset = bits(self.word, 20, 14) * 4;
        self.require(slot < SLOTS, &format!("constant buffer slot {slot}"))?;
        self.constants.insert((slot, offset));
        Ok(if self.indirect.contains(&slot) {
            format!("cb_word(cb{slot}, cb_size[{slot}], {offset}u)")
        } else {
            format!("cb{slot}[{}]", offset / 4)
        })
    }

    /// Operand B as a `uint` expression.
    fn b(&mut self, form: Form) -> Result<String, String> {
        let w = self.word;
        Ok(match form {
            Form::R => self.read(bits(w, 20, 8)),
            Form::I => format!("{:#x}u", ops::imm20(w) as u32),
            Form::F => format!("{:#x}u", (ops::imm20(w) as u32) << 12),
            Form::C => self.constant()?,
            Form::Rc => self.read(bits(w, 39, 8)),
            Form::Imm32 => format!("{:#x}u", (w >> 20) as u32),
        })
    }

    /// Operand C as a `uint` expression.
    fn c(&mut self, form: Form) -> Result<String, String> {
        match form {
            Form::Rc => self.constant(),
            _ => Ok(self.read(bits(self.word, 39, 8))),
        }
    }

    /// The condition-code test `ccc` as a `bool` expression [public:
    /// Ryujinx `Instructions/InstEmitConditionCode.cs` `GetCondition`, as
    /// the interpreter's `condition`].
    fn condition(&mut self, ccc: u32) -> Result<String, String> {
        if ccc == 15 {
            return Ok("true".into());
        }
        if ccc == 0 {
            return Ok("false".into());
        }
        self.cc = true;
        let (z, n, c, v) = (
            "((cc & 1u) != 0u)",
            "((cc & 2u) != 0u)",
            "((cc & 4u) != 0u)",
            "((cc & 8u) != 0u)",
        );
        Ok(match ccc {
            1 => format!("(({n} && !{z}) != {v})"),
            2 => format!("(!{n} && {z})"),
            3 => format!("({n} != ({z} || {v}))"),
            4 => format!("(!(({n} != {v}) || {z}))"),
            5 => format!("(!{z})"),
            6 => format!("({n} == {v})"),
            7 => format!("(!({n} && {z}))"),
            8 => format!("({n} && {z})"),
            9 => format!("({n} != {v})"),
            10 => z.to_string(),
            11 => format!("(({n} != {v}) || {z})"),
            12 => format!("(!{n} != ({v} || {z}))"),
            13 => format!("({n} || !{z})"),
            14 => format!("((!{n} || {z}) != {v})"),
            16 => format!("(!{v})"),
            17 => format!("(!{c})"),
            18 => format!("(!{n})"),
            19 => format!("({z} || !{c})"),
            20 => format!("({c} && !{z})"),
            21 => n.to_string(),
            22 => c.to_string(),
            23 => v.to_string(),
            30 => format!("({n} || {z})"),
            31 => format!("(!({n} || {z}))"),
            other => return self.refuse(&format!("condition code test {other}")),
        })
    }

    /// Whether the guard (bits 16-19) and condition-code test (bits 0-4)
    /// of a control instruction hold, as a `bool` expression.
    fn control_condition(&mut self) -> Result<String, String> {
        let guard = self.pred_operand(16);
        let test = self.condition(bits(self.word, 0, 5))?;
        Ok(match (guard.as_str(), test.as_str()) {
            (g, "true") => g.to_string(),
            ("true", t) => t.to_string(),
            (g, t) => format!("({g} && {t})"),
        })
    }

    /// One instruction with its guard. Control instructions follow the
    /// interpreter's `step`/`execute_other` on the lane's own stack
    /// (`stack`/`depth`); a jump sets `pc` and leaves the `switch`. Where
    /// the interpreter stops with an error (stack overflow, no matching
    /// entry, the terminating branch to itself), the lane ends.
    fn step(&mut self, pc: u32, linear: bool) -> Result<(), String> {
        let inst = self.program.code[(pc / 8) as usize].expect("flow lists instructions");
        self.word = inst.word;
        self.pc = pc;
        let w = inst.word;
        let end = if linear {
            "break;".to_string()
        } else {
            format!("pc = {DONE}; break;")
        };
        let kind = |op: Op| match op {
            Op::Ssy | Op::Sync => 0u32,
            Op::Pbk | Op::Brk => 1,
            _ => 2,
        };
        let action = match inst.op {
            Op::Nop | Op::Depbar | Op::Membar | Op::Bar => return Ok(()),
            // No guard field (see `exec`'s `step`).
            Op::Ssy | Op::Pbk | Op::Pcnt => {
                let target = branch_target(w, pc).expect("checked by flow");
                self.line(&format!(
                    "if (depth == {STACK_DEPTH}u) {{ {end} }} stack[depth] = {:#x}u; depth++;",
                    (kind(inst.op) << 30) | target
                ));
                return Ok(());
            }
            Op::Bra | Op::Brx | Op::Sync | Op::Brk | Op::Cont | Op::Exit | Op::Kil
                if flow::inert(w) =>
            {
                return Ok(());
            }
            Op::Exit => end.clone(),
            Op::Kil => format!("killed = true; {end}"),
            Op::Bra if is_branch_to_self(w, pc) => end.clone(),
            Op::Bra => {
                let target = branch_target(w, pc).expect("checked by flow");
                format!("pc = {target}u; break;")
            }
            Op::Brx => {
                let base = (pc + 8).wrapping_add(sext(bits(w, 20, 24), 24) as u32);
                let offset = self.read(bits(w, 8, 8));
                format!(
                    "{{ uint t = {base:#x}u + {offset}; pc = (t & 31u) == 0u ? t + 8u : t; break; }}"
                )
            }
            Op::Sync => format!(
                "if (depth == 0u || (stack[depth - 1u] >> 30) != 0u) {{ {end} }} depth--; pc = stack[depth] & 0x3fffffffu; break;"
            ),
            Op::Brk | Op::Cont => format!(
                // BRK and CONT unwind inner SSY entries.
                "while (depth != 0u && (stack[depth - 1u] >> 30) == 0u) {{ depth--; }} if (depth == 0u || (stack[depth - 1u] >> 30) != {}u) {{ {end} }} depth--; pc = stack[depth] & 0x3fffffffu; break;",
                kind(inst.op)
            ),
            op => {
                let (guard, negated) = (bits(w, 16, 3), bit(w, 19));
                if guard == PT && negated {
                    return Ok(());
                }
                if guard == PT {
                    return self.instruction(op);
                }
                let condition = self.pred_operand(16);
                self.line(&format!("if ({condition}) {{"));
                self.instruction(op)?;
                self.line("}");
                return Ok(());
            }
        };
        if flow::always(w) {
            self.line(&action);
        } else {
            let condition = self.control_condition()?;
            self.line(&format!("if ({condition}) {{ {action} }}"));
        }
        Ok(())
    }

    fn instruction(&mut self, op: Op) -> Result<(), String> {
        if self.alu(op)? || self.integer(op)? {
            return Ok(());
        }
        match op {
            Op::Ipa => self.ipa(),
            Op::Ldc => self.ldc(),
            Op::Texs => self.texs(),
            Op::Tex => self.tex(),
            Op::TexB | Op::TldB | Op::Tld4B | Op::TmmlB | Op::TxdB | Op::TxqB => {
                self.refuse("bindless texture (handle in a register)")
            }
            _ => self.refuse("not translated"),
        }
    }
}

const PRELUDE: &str = r#"#include <metal_stdlib>
using namespace metal;
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

static inline float ftz(float v) {
    return (v != 0.0f && fabs(v) < 1.17549435e-38f) ? copysign(0.0f, v) : v;
}
/// As Rust's `clamp`: -0.0 stays -0.0, NaN becomes 0.
static inline float sat(float v) {
    return isnan(v) ? 0.0f : (v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v));
}
static inline float next_down(float v) { return nextafter(v, as_type<float>(0xff800000u)); }
static inline float next_up(float v) { return nextafter(v, as_type<float>(0x7f800000u)); }
/// x * y * scale rounded in direction `mode` (1 down, 2 up, 3 to zero).
static inline float mul_round(float x, float y, float scale, uint mode) {
    float p = x * y;
    float e = fma(x, y, -p);
    float v = p * scale;
    bool down = mode == 1u || (mode == 3u && v > 0.0f);
    bool up = mode == 2u || (mode == 3u && v < 0.0f);
    if (down && e < 0.0f) { v = next_down(v); }
    else if (up && e > 0.0f) { v = next_up(v); }
    return v;
}
/// An integer to f32 in direction `mode` (0 nearest even).
static inline float i2f(long v, uint mode) {
    float r = float(v);
    long q = long(r);
    bool down = mode == 1u || (mode == 3u && v > 0);
    bool up = mode == 2u || (mode == 3u && v < 0);
    if (down && q > v) { r = next_down(r); }
    else if (up && q < v) { r = next_up(r); }
    return r;
}
/// Saturating conversions of an integral float, as Rust's `as`.
static inline uint f2u32(float v) {
    return isnan(v) || v <= 0.0f ? 0u : (v >= 4294967296.0f ? 0xffffffffu : uint(v));
}
static inline uint f2i32(float v) {
    return isnan(v) ? 0u : (v <= -2147483648.0f ? 0x80000000u : (v >= 2147483648.0f ? 0x7fffffffu : as_type<uint>(int(v))));
}
static inline uint f2u16(float v) { return isnan(v) ? 0u : uint(clamp(v, 0.0f, 65535.0f)); }
static inline uint f2i16(float v) { return isnan(v) ? 0u : as_type<uint>(int(clamp(v, -32768.0f, 32767.0f))); }
/// Constant buffer reads at dynamic byte addresses: 0 past the readable
/// size (and for unaligned addresses, which the interpreter refuses).
static inline uint cb_word(const device uint* cb, uint size, uint address) {
    return ((address & 3u) == 0u && size >= 4u && address <= size - 4u) ? cb[address >> 2] : 0u;
}
static inline uint cb_bytes(const device uint* cb, uint size, uint address, uint width) {
    if ((address & (width - 1u)) != 0u || size < width || address > size - width) { return 0u; }
    return (cb[address >> 2] >> ((address & 3u) * 8u)) & (width == 1u ? 0xffu : 0xffffu);
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

/// The dispatch loop's "no block": lane finished.
const DONE: &str = "0xffffffffu";

impl Translated {
    /// The complete MSL source: the vertex function `vertex_main` (window
    /// positions from buffer 0) and the fragment function
    /// `fragment_main`, whose color attachment `i` receives the pixel
    /// program's output target `outputs[i]` (components the SPH does not
    /// declare are 0). With an SPH depth output, `Output` also has
    /// `depth [[depth(any)]]`.
    ///
    /// Fragment bindings: buffer 0 the primitive planes (per primitive,
    /// `ipa_words.len().max(1)` planes of (dx, dy, base), indexed by
    /// `[[primitive_id]]`), buffer 1 the texture LOD biases (one f32 per
    /// texture slot), buffer `2 + slot` constant buffer `slot` as words
    /// (for [`Translated::indirect`] slots the readable part of the bound
    /// range, padded to a multiple of 4 bytes), buffer 20 the sizes in
    /// bytes of those readable parts for the 18 slots (u32 each; bound only
    /// when `indirect` is not empty: indexed reads past it return 0, which
    /// is the interpreter's value past the bound size, and its refusal
    /// where the bound range is not mapped); texture and sampler `i` for
    /// `textures[i]`; buffer 21, when there are textures, the texture
    /// fault words: word `2i` non-zero marks texture `i` as unbound (its
    /// handle names no texture; a placeholder is bound), and a sample of
    /// such a texture sets word `2i + 1` (see `texture.rs`).
    pub fn source(&self, program: &Program, outputs: &[usize]) -> String {
        let header = &program.header;
        let mut s = String::from(PRELUDE);
        let _ = writeln!(s, "struct Output {{");
        for (attachment, _) in outputs.iter().enumerate() {
            let _ = writeln!(s, "    float4 c{attachment} [[color({attachment})]];");
        }
        if header.ps_writes_depth() {
            let _ = writeln!(s, "    float depth [[depth(any)]];");
        }
        let _ = writeln!(s, "}};");
        let mut parameters = vec![
            "float4 position [[position]]".to_string(),
            "uint primitive [[primitive_id]]".to_string(),
            "const device float* planes [[buffer(0)]]".to_string(),
            "const device float* lod_bias [[buffer(1)]]".to_string(),
        ];
        let slots: BTreeSet<u32> = self
            .constants
            .iter()
            .map(|&(slot, _)| slot)
            .chain(self.indirect.iter().copied())
            .collect();
        for slot in &slots {
            parameters.push(format!(
                "const device uint* cb{slot} [[buffer({})]]",
                2 + slot
            ));
        }
        if !self.indirect.is_empty() {
            parameters.push("const device uint* cb_size [[buffer(20)]]".to_string());
        }
        for i in 0..self.textures.len() {
            parameters.push(format!("texture2d<float> t{i} [[texture({i})]]"));
            parameters.push(format!("sampler s{i} [[sampler({i})]]"));
        }
        if !self.textures.is_empty() {
            parameters.push("device atomic_uint* texture_faults [[buffer(21)]]".to_string());
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
        let _ = writeln!(s, "    if (killed) {{ discard_fragment(); }}");
        let _ = writeln!(s, "    Output out;");
        // Outputs as the interpreter gathers them: each target with a
        // component mask takes the next four registers from R0; depth is
        // one register after the colors.
        let (registers, depth) = output_registers(program);
        for (attachment, &target) in outputs.iter().enumerate() {
            let value = match registers.get(target).copied().flatten() {
                Some((base, mask)) => {
                    let components: Vec<String> = (0..4)
                        .map(|c| {
                            if mask & (1 << c) != 0 {
                                format!("as_type<float>(r{})", base + c)
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
        if let Some(register) = depth {
            let _ = writeln!(s, "    out.depth = as_type<float>(r{register});");
        }
        let _ = writeln!(s, "    return out;\n}}");
        s
    }
}

/// Per render target, the first register and component mask of its
/// output, and the depth register.
fn output_registers(program: &Program) -> ([Option<(u32, u8)>; 8], Option<u32>) {
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
    let depth = header.ps_writes_depth().then_some(base + 1);
    (registers, depth)
}

/// Translates a pixel program, or says why it cannot be.
/// `constant_buffer_1`: the bound range of the stage's constant buffer 1
/// (only read for `BRX` jump tables; see [`Translated::constant_buffer_1`]).
pub fn translate(program: &Program, constant_buffer_1: &[u8]) -> Result<Translated, String> {
    let header = &program.header;
    if header.shader_type() != ShaderType::Pixel {
        return Err("not a pixel program".into());
    }
    if header.ps_writes_sample_mask() {
        return Err("sample-mask output".into());
    }
    let flow = flow::analyze(program, constant_buffer_1)?;
    let mut writer = Writer {
        program,
        out: String::new(),
        registers: BTreeSet::new(),
        predicates: BTreeSet::new(),
        cc: false,
        ipa_words: Vec::new(),
        textures: Vec::new(),
        implicit_lod: Vec::new(),
        constants: BTreeSet::new(),
        indirect: BTreeSet::new(),
        word: 0,
        pc: 8,
    };
    // Slots LDC reads: static reads of them are bounds-checked too.
    for &pc in &flow.pcs {
        let inst = program.code[(pc / 8) as usize].expect("instruction");
        if inst.op == Op::Ldc {
            writer.indirect.insert(bits(inst.word, 36, 5));
        }
    }
    for &pc in &flow.pcs {
        if !flow.linear && flow.labels.contains(&pc) {
            let _ = writeln!(writer.out, "    case {pc}u:");
        }
        writer.step(pc, flow.linear)?;
    }
    let mut body = String::new();
    // Registers, predicates and flags start at zero, as in the interpreter.
    let (outputs, depth) = output_registers(program);
    let declared: BTreeSet<u32> = writer
        .registers
        .iter()
        .copied()
        .chain(
            outputs
                .iter()
                .flatten()
                .flat_map(|&(base, _)| base..base + 4),
        )
        .chain(depth)
        .collect();
    for index in &declared {
        let _ = writeln!(body, "    uint r{index} = 0u;");
    }
    for index in &writer.predicates {
        let _ = writeln!(body, "    bool p{index} = false;");
    }
    if writer.cc {
        let _ = writeln!(body, "    uint cc = 0u;");
    }
    let _ = writeln!(body, "    bool killed = false;");
    if flow.linear {
        let _ = writeln!(body, "    do {{");
        body.push_str(&writer.out);
        let _ = writeln!(body, "    }} while (false);");
    } else {
        // Each iteration the quad runs the lanes at its smallest program
        // counter, as the interpreter's lockstep schedule.
        let _ = writeln!(
            body,
            "    uint pc = 8u; uint depth = 0u; uint stack[{STACK_DEPTH}];\n    for (uint steps = 0u; steps < {}u; steps++) {{\n    uint pair = min(pc, quad_shuffle_xor(pc, 1u));\n    uint current = min(pair, quad_shuffle_xor(pair, 2u));\n    if (current == {DONE}) {{ break; }}\n    if (pc != current) {{ continue; }}\n    switch (pc) {{",
            super::exec::STEP_LIMIT
        );
        body.push_str(&writer.out);
        let _ = writeln!(body, "    default: pc = {DONE}; break;\n    }}\n    }}");
    }
    Ok(Translated {
        body,
        ipa_words: writer.ipa_words,
        textures: writer.textures,
        constants: writer.constants,
        indirect: writer.indirect,
        implicit_lod: writer.implicit_lod,
        constant_buffer_1: flow.jump_tables.then(|| constant_buffer_1.to_vec()),
    })
}
