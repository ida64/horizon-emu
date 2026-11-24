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
mod memory;
mod ops;
#[cfg(test)]
mod tests;
mod texture;
mod vertex;

pub use vertex::{
    Fetch, INDEXED_WORD, MAX_STREAMS, NativeVertex, STREAM_BUFFER, TranslatedVertex,
    translate_vertex,
};

use std::collections::BTreeSet;
use std::fmt::Write;

use super::Program;
use super::decode::Op;
use super::exec::{bit, bits, branch_target, is_branch_to_self, sext};
use super::header::ShaderType;
use crate::maxwell::texture::format::DepthFormat;

const RZ: u32 = 255;
/// Entries of a lane's reconvergence stack (the interpreter's
/// `STACK_DEPTH`).
const STACK_DEPTH: u32 = 32;
const PT: u32 = 7;

/// Constant buffer slots a stage has (`Env::const_buffers`).
const SLOTS: u32 = 18;

/// Words per triangle of the triangle buffer (see [`Translated::source`]).
pub const TRIANGLE_WORDS: usize = 10;

/// The depth/stencil attachment a fragment function tests (see
/// [`Translated::source`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DepthStage {
    pub format: DepthFormat,
    /// Whether passing fragments store their depth.
    pub write: bool,
    /// Whether the stencil test runs (the format has stencil).
    pub stencil: bool,
}

/// How a fragment function draws into multisampled targets (see
/// [`Translated::source`]); the sample positions are a draw-time value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MultisampleStage {
    /// Samples per pixel, more than one.
    pub samples: u32,
    pub alpha_to_coverage: bool,
    pub alpha_to_one: bool,
    /// Depth at the pixel center rather than at each sample.
    pub center_depth: bool,
}

/// What a texture slot of a translated program is: the Metal texture
/// type the draw must bind there.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureKind {
    /// `texture2d`: one layer.
    Flat,
    /// `texture2d_array`: sampled with a layer.
    Array,
    /// `texturecube`.
    Cube,
    /// `texture3d`.
    Volume,
}

impl TextureKind {
    fn declaration(self) -> &'static str {
        match self {
            TextureKind::Flat => "texture2d<float>",
            TextureKind::Array => "texture2d_array<float>",
            TextureKind::Cube => "texturecube<float>",
            TextureKind::Volume => "texture3d<float>",
        }
    }
}

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
    /// texture/sampler slot order (a handle sampled as two kinds has two
    /// slots).
    pub textures: Vec<u32>,
    /// What each slot is.
    pub kinds: Vec<TextureKind>,
    /// Constant buffer words read with static offsets: (slot, byte offset).
    pub constants: BTreeSet<(u32, u32)>,
    /// Constant buffer slots read at register-dependent offsets (`LDC`):
    /// the draw binds the whole bound range of these.
    pub indirect: BTreeSet<u32>,
    /// Which texture slots are sampled with implicit level of detail (and
    /// so with the sampler's LOD bias).
    pub implicit_lod: Vec<bool>,
    /// Which texture slots are sampled with a depth comparison: the draw
    /// gives each one's function and filter (after the LOD biases, as
    /// the bits of a float: function in bits 2:0, bit 3 linear).
    pub compare: Vec<bool>,
    /// The constant buffer 1 content the targets of `BRX` (indirect
    /// branches through a jump table there) were taken from; `None` if the
    /// program has none. The translation is valid only for draws whose
    /// constant buffer 1 has this content.
    pub constant_buffer_1: Option<Vec<u8>>,
    /// Where the buffers `LDG`, `STG` and `RED` address may be named:
    /// (constant buffer slot, byte offset) of each 64-bit address the
    /// program loads into a register pair (see `memory.rs`). The draw
    /// binds the memory each names, in this order, at [`GLOBAL_BUFFER`].
    pub globals: Vec<(u32, u32)>,
    /// The program stores to that memory (`STG`, `RED`): the renderer
    /// writes it back to guest memory.
    pub stores: bool,
    /// The program updates that memory with `RED`, once per fragment:
    /// not to be run once per sample.
    pub atomics: bool,
}

/// The fragment buffer index of the global memory a program addresses
/// (see `memory.rs`).
pub const GLOBAL_BUFFER: usize = 28;
/// The vertex buffer index of the same for a vertex program.
pub const VERTEX_GLOBAL_BUFFER: usize = 30;
/// A word of the bound global memory, in a translated program's body.
const GLOBAL_WORD: &str = "#define GM(i) atomic_load_explicit(&gm[i], memory_order_relaxed)";

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

/// The program stage a [`Writer`] translates.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage {
    Pixel,
    Vertex,
}

struct Writer<'a> {
    program: &'a Program,
    stage: Stage,
    out: String,
    registers: BTreeSet<u32>,
    predicates: BTreeSet<u32>,
    /// The program reads or writes the condition code.
    cc: bool,
    ipa_words: Vec<u32>,
    textures: Vec<u32>,
    kinds: Vec<TextureKind>,
    implicit_lod: Vec<bool>,
    compare: Vec<bool>,
    constants: BTreeSet<(u32, u32)>,
    indirect: BTreeSet<u32>,
    globals: Vec<(u32, u32)>,
    /// Vertex programs: input attribute words `ALD` reads (`i{word}`).
    inputs: BTreeSet<u32>,
    /// Vertex programs: output attribute words `AST` writes or `ALD.O`
    /// reads (`o{word}`).
    outputs: BTreeSet<u32>,
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

    /// The condition-code test `ccc` as a `bool` expression (as the
    /// interpreter's `condition`).
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
        match (self.stage, op) {
            (Stage::Pixel, Op::Ipa) => self.ipa(),
            (Stage::Pixel, Op::Shfl) => self.shfl(),
            (Stage::Pixel, Op::Fswzadd) => self.fswzadd(),
            (Stage::Pixel, Op::Texs) => self.texs(),
            (Stage::Pixel, Op::Tex) => self.tex(),
            (Stage::Pixel, Op::Tld4) => self.tld4(),
            (Stage::Pixel, Op::Tld4s) => self.tld4s(),
            (Stage::Pixel, Op::Tlds) => self.tlds(),
            (Stage::Pixel, Op::Tld) => self.tld(),
            (_, Op::Ldg) => self.ldg(),
            (Stage::Pixel, Op::Stg) => self.stg(),
            (Stage::Pixel, Op::Red) => self.red(),
            (Stage::Pixel, Op::Ldl) => self.local(false),
            (Stage::Pixel, Op::Stl) => self.local(true),
            (Stage::Pixel, Op::Vmnmx) => self.vmnmx(),
            (Stage::Pixel, Op::Tmml) => self.tmml(),
            // SR_Y_DIRECTION: 1.0, as the interpreter's (`pipeline`'s
            // `env`).
            (Stage::Pixel, Op::S2r) if bits(self.word, 20, 8) == 18 => {
                self.set(bits(self.word, 0, 8), "0x3f800000u");
                Ok(())
            }
            // SR_THREAD_KILL (`exec::Machine::special_register`).
            (Stage::Pixel, Op::S2r) if bits(self.word, 20, 8) == 19 => {
                self.set(bits(self.word, 0, 8), "(GM_LIVE ? 0u : 0xffffffffu)");
                Ok(())
            }
            (Stage::Vertex, Op::Ald) => self.ald(),
            (Stage::Vertex, Op::Ast) => self.ast(),
            (_, Op::Ldc) => self.ldc(),
            (_, Op::TexB | Op::TldB | Op::Tld4B | Op::TmmlB | Op::TxdB | Op::TxqB) => {
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
/// The depth comparison `function` (NEVER, LESS, EQUAL, LEQUAL, GREATER,
/// NOTEQUAL, GEQUAL, ALWAYS) of the reference with each of `d`.
static inline float4 compare_taps(float4 d, float r, uint function) {
    switch (function & 7u) {
        case 0u: return float4(0.0f);
        case 1u: return float4(float4(r) < d);
        case 2u: return float4(float4(r) == d);
        case 3u: return float4(float4(r) <= d);
        case 4u: return float4(float4(r) > d);
        case 5u: return float4(float4(r) != d);
        case 6u: return float4(float4(r) >= d);
        default: return float4(1.0f);
    }
}
/// A depth comparison as the interpreter samples one
/// (`texture/filtering.rs` `sample`, `filtered`, `tap`): `cfg` holds the
/// function (bits 2:0), whether the filter is linear (bit 3) and the mip
/// filter (bits 5:4: 1 none, 2 nearest, 3 linear); `raw` is the level of
/// detail before the clamp to `lmin`..`lmax`. Each tap is one texel of
/// the level, taken at its center, compared, and then weighed.
#define COMMA ,
#define COMPARE_FUNCTIONS(TEX, LAYER_PARAMETER, LAYER) \
static inline float4 compare_level(TEX t, sampler s, float2 uv LAYER_PARAMETER, float r, uint cfg, float lv) { \
    if ((cfg & 8u) == 0u) { return compare_taps(float4(t.sample(s, uv LAYER, level(lv)).x), r, cfg); } \
    const float2 size = float2(t.get_width(uint(lv)), t.get_height(uint(lv))); \
    const float2 at = uv * size - 0.5f; \
    const float2 p = floor(at) + 0.5f; \
    const float2 f = at - floor(at); \
    const float4 d = float4(t.sample(s, p / size LAYER, level(lv)).x, t.sample(s, (p + float2(1.0f, 0.0f)) / size LAYER, level(lv)).x, t.sample(s, (p + float2(0.0f, 1.0f)) / size LAYER, level(lv)).x, t.sample(s, (p + 1.0f) / size LAYER, level(lv)).x); \
    const float4 c = compare_taps(d, r, cfg); \
    return float4(mix(mix(c.x, c.y, f.x), mix(c.z, c.w, f.x), f.y)); \
} \
static inline float4 compare_sample(TEX t, sampler s, float2 uv LAYER_PARAMETER, float r, uint cfg, float raw, float lmin, float lmax) { \
    const float lod = clamp(raw, lmin, lmax); \
    const uint mip = (cfg >> 4) & 3u; \
    const float lv = mip == 1u ? 0.0f : (mip == 2u ? round(lod) : floor(lod)); \
    const float4 a = compare_level(t, s, uv LAYER, r, cfg, lv); \
    if (mip != 3u || lod == lv) { return a; } \
    const float top = float(t.get_num_mip_levels()) - 1.0f; \
    return mix(a, compare_level(t, s, uv LAYER, r, cfg, min(lv + 1.0f, top)), lod - lv); \
}
COMPARE_FUNCTIONS(texture2d<float>, , )
COMPARE_FUNCTIONS(texture2d_array<float>, COMMA uint layer, COMMA layer)
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
/// `raster::Compare::test` by the numbering of `Translated::source`.
static inline bool compare(uint f, float a, float b) {
    return f == 1u ? a < b : f == 2u ? a == b : f == 3u ? a <= b : f == 4u ? a > b : f == 5u ? a != b : f == 6u ? a >= b : f == 7u;
}
/// `raster::StencilOp::apply` by the numbering of `Translated::source`.
static inline uint stencil_op(uint op, uint s, uint ref) {
    return op == 1u ? 0u : op == 2u ? ref : op == 3u ? min(s + 1u, 255u) : op == 4u ? (s == 0u ? 0u : s - 1u) : op == 5u ? (~s & 0xffu) : op == 6u ? ((s + 1u) & 0xffu) : op == 7u ? ((s - 1u) & 0xffu) : s;
}
/// `raster::Triangle::covers_at`: whether the point (sx, sy) / 16 from
/// the corner of the pixel whose center is the window position (x, y)
/// is inside triangle `t` (see `Translated::source`), with the
/// 1/256-pixel edge functions and the top-left rule of the software
/// rasterizer.
static inline bool covers_at(const device int* t, float x, float y, uint sx, uint sy) {
    const long cx = (long(x - 0.5f) << 8) + long(sx << 4);
    const long cy = (long(y - 0.5f) << 8) + long(sy << 4);
    bool inside = true;
    for (uint e = 0u; e < 3u; e++) {
        const uint n = e == 2u ? 0u : e + 1u;
        const long ax = t[2u * e], ay = t[2u * e + 1u], bx = t[2u * n], by = t[2u * n + 1u];
        const long o = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
        const long bias = ((uint(t[6]) >> e) & 1u) != 0u ? -1 : 0;
        inside = inside && o + bias >= 0;
    }
    return inside;
}
/// `raster::Triangle::covers`: the pixel center.
static inline bool covers(const device int* t, float x, float y) {
    return covers_at(t, x, y, 8u, 8u);
}
/// Polygon offset of a native draw (see `metal`): `b` holds the
/// constant term for a fixed least representable value, the slope
/// factor, the clamp (0: none) and the constant factor for float depth's
/// value-dependent one, 2^(exponent(z) - 23) (`pipeline::depth_bias`).
static inline float native_depth_bias(constant float* b, float z, float slope) {
    int e;
    const float m = frexp(fabs(z), e);
    const float r = m == 0.0f ? 0.0f : ldexp(1.0f, e - 24);
    const float bias = (b[0] + b[3] * r) + b[1] * slope;
    return b[2] > 0.0f ? min(bias, b[2]) : (b[2] < 0.0f ? max(bias, b[2]) : bias);
}
/// `raster::alpha_coverage` for `n` samples.
static inline uint alpha_coverage(float alpha, uint n) {
    const float a = isnan(alpha) ? 0.0f : clamp(alpha, 0.0f, 1.0f);
    return (1u << uint(rint(a * float(n)))) - 1u;
}
/// `unorm_to_f32(v, n)` for n = 16 or 24: v / (2^n - 1) correctly
/// rounded, which the interpreter's f64 quotient rounded to f32 is (a
/// double rounding would need 28 equal bits in a row of the quotient,
/// whose bits repeat v's n bits). Those bits give the 24 significant bits
/// from v's leading one and the rounding bit from two or three copies of
/// v, and the bits after them are never all zero (0 < v < 2^n - 1): no tie.
static inline float unorm_to_float(uint v, uint n) {
    const uint top = (1u << n) - 1u;
    if (v == 0u) { return 0.0f; }
    if (v >= top) { return 1.0f; }
    const uint copies = n == 16u ? 32u : 24u;
    const ulong w = n == 16u ? (ulong(v) << 32) | (ulong(v) << 16) | ulong(v) : (ulong(v) << 24) | ulong(v);
    const uint lead = 31u - clz(v);
    const uint p = copies + lead;
    const uint m = uint(w >> (p - 23u)) & 0xffffffu;
    const uint round = uint(w >> (p - 24u)) & 1u;
    const int e = int(lead) - 23 - int(n);
    return float(m + round) * as_type<float>(uint(127 + e) << 23);
}
/// `f32_to_unorm(v, n)` for n <= 24: floor(v (2^n - 1) + 1/2) of v clamped
/// to [0, 1] (NaN 0), exactly, as the interpreter's f64 arithmetic.
static inline uint float_to_unorm(float v, uint n) {
    if (isnan(v) || !(v > 0.0f)) { return 0u; }
    if (v >= 1.0f) { return (1u << n) - 1u; }
    const uint bits = as_type<uint>(v);
    const uint biased = bits >> 23;
    const ulong m = biased == 0u ? ulong(bits & 0x7fffffu) : ulong((bits & 0x7fffffu) | 0x800000u);
    // v = m 2^-s.
    const uint s = biased == 0u ? 149u : 150u - biased;
    if (s > 62u) { return 0u; }
    return uint((m * ulong((1u << n) - 1u) + (1ul << (s - 1u))) >> s);
}

struct Vertex { float4 position [[position]]; };

vertex Vertex vertex_main(uint id [[vertex_id]], const device float4* positions [[buffer(0)]]) {
    Vertex out;
    out.position = positions[id];
    return out;
}

/// Draws into layer `layer` (sample `layer` of multisampled targets).
struct LayeredVertex { float4 position [[position]]; uint layer [[render_target_array_index]]; };

vertex LayeredVertex vertex_layered(uint id [[vertex_id]], const device float4* positions [[buffer(0)]], constant uint& layer [[buffer(1)]]) {
    LayeredVertex out;
    out.position = positions[id];
    out.layer = layer;
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
    /// declare are 0).
    ///
    /// Each set-up triangle is drawn as two Metal triangles covering its
    /// pixel bounding box (primitives `2t` and `2t + 1` for triangle `t`),
    /// and a fragment is kept only where `raster::Triangle::covers` holds:
    /// pixels on an edge are decided by the software rasterizer's exact
    /// rule, not by Metal's (which sees the vertices as rounded floats).
    ///
    /// Fragment bindings: buffer 0 the triangle planes (per triangle,
    /// `ipa_words.len().max(1)` planes of (dx, dy, base)), buffer 1 the
    /// texture LOD biases (one f32 per texture slot), buffer `2 + slot`
    /// constant buffer `slot` as words (for [`Translated::indirect`] slots
    /// the readable part of the bound range, padded to a multiple of 4
    /// bytes), buffer 20 the sizes in bytes of those readable parts for the
    /// 18 slots (u32 each; bound only when `indirect` is not empty: indexed
    /// reads past it return 0, which is the interpreter's value past the
    /// bound size, and its refusal where the bound range is not mapped);
    /// texture and sampler `i` for `textures[i]`; buffer 21, when there are
    /// textures, the texture fault words: word `2i` non-zero marks texture
    /// `i` as unbound (its handle names no texture; a placeholder is
    /// bound), and a sample of such a texture sets word `2i + 1` (see
    /// `texture.rs`); buffer 24 [`TRIANGLE_WORDS`] words per triangle:
    /// the snapped vertices (x, y in 1/256 pixels, i32) in edge order,
    /// a word whose bit `e` marks edge `e` as not top-left
    /// (`raster::Triangle::edges`) and whose bit 3 marks a back-facing
    /// triangle, and the depth plane (dx, dy, base, f32).
    ///
    /// With `alpha_test`, buffer 22 holds the alpha test as two words: the
    /// comparison (0 never, 1 less, 2 equal, 3 less-or-equal, 4 greater, 5
    /// not equal, 6 greater-or-equal, 7 always) and the reference float;
    /// a fragment whose output 0 alpha fails it is discarded, after the
    /// program and before any output (the software back end's order). The
    /// comparison is a draw-time value, so one pipeline serves every
    /// function and reference.
    ///
    /// With `depth`, color attachment `outputs.len()` is the depth buffer
    /// of that format as its raw words (see `metal::depth_pixel_format`),
    /// read in the fragment function (Apple GPUs order such reads of one
    /// pixel by primitive), and buffer 23 holds the depth test as three
    /// words: the comparison (numbered as the alpha test's) and the clamp
    /// range (two f32). The fragment depth (the program's depth output if
    /// it has one, else the depth plane at the pixel center), clamped, is
    /// compared with the stored depth as `raster::shade_quad` does, after
    /// the alpha test; a failing fragment is discarded, a passing one
    /// stores its depth with the format's encoding when `write`, other
    /// bits kept.
    ///
    /// With `stencil`, buffer 25 holds the stencil test as two faces
    /// (front, then back) of seven words: the operations on stencil
    /// failure, depth failure and pass (0 keep, 1 zero, 2 replace, 3
    /// increment clamped, 4 decrement clamped, 5 invert, 6 increment
    /// wrapping, 7 decrement wrapping), the comparison, the reference, the
    /// compare mask and the write mask (8 bits each). The stencil test
    /// comes before the depth test and the stencil bits are updated by the
    /// operation the outcome selects, as `raster::shade_quad` does. A
    /// failing fragment is discarded when there are color outputs (the
    /// draw makes sure its stencil operations on failure keep the stored
    /// value then), and otherwise stores the updated word.
    ///
    /// With `multisample` (more than one sample per pixel), the targets
    /// are 2D arrays of one layer per sample at pixel size, drawn with
    /// `vertex_layered` once per sample (its buffer 1 the layer): the
    /// fragment function runs once per sample, for sample `layer`, each run
    /// computing everything for the pixel as the software back end does
    /// once (so quads and derivatives are those of whole pixels) and
    /// keeping its own sample. Buffer 26 holds the sample positions, one
    /// word per sample: x | y << 8 in 1/16 pixel. The pixel's coverage
    /// mask (`raster::Triangle::coverage`) gives the centroid `IPA`
    /// CENTROID reads (`raster::centroid`); with `alpha_to_coverage` it
    /// is limited by output 0 alpha (`raster::alpha_coverage`) after the
    /// alpha test; the depth plane is taken at the sample's position
    /// unless `center_depth`; with `alpha_to_one` every output's alpha is
    /// 1.
    pub fn source(
        &self,
        program: &Program,
        outputs: &[usize],
        alpha_test: bool,
        depth: Option<DepthStage>,
        multisample: Option<MultisampleStage>,
        native: Option<&NativeVertex>,
    ) -> String {
        let mut s = String::from(PRELUDE);
        if let Some(vertex) = native {
            s.push_str(&vertex.source);
        }
        // The depth attachment's words, how its depth is read (from `zin`)
        // and written (into `w`, the word being updated)
        // (`DepthFormat::read_depth`, `write_depth`), and how its stencil
        // is read and written (`s`) (`read_stencil`, `write_stencil`).
        let formats = depth.map(|stage| match stage.format {
            DepthFormat::Z32Float => ("uint", "as_type<float>(zin)", "as_type<uint>(z)", None),
            DepthFormat::Z32FloatX24S8 => (
                "uint2",
                "as_type<float>(zin.x)",
                "uint2(as_type<uint>(z), w.y)",
                Some(("(zin.y & 0xffu)", "uint2(w.x, (w.y & 0xffffff00u) | s)")),
            ),
            DepthFormat::Z16Unorm => (
                "uint",
                "unorm_to_float(zin & 0xffffu, 16u)",
                "float_to_unorm(z, 16u)",
                None,
            ),
            DepthFormat::Z24S8 => (
                "uint",
                "unorm_to_float(zin >> 8, 24u)",
                "(w & 0xffu) | (float_to_unorm(z, 24u) << 8)",
                Some(("(zin & 0xffu)", "(w & 0xffffff00u) | s")),
            ),
            DepthFormat::X8Z24 => (
                "uint",
                "unorm_to_float(zin & 0xffffffu, 24u)",
                "(w & 0xff000000u) | float_to_unorm(z, 24u)",
                None,
            ),
            DepthFormat::S8Z24 => (
                "uint",
                "unorm_to_float(zin & 0xffffffu, 24u)",
                "(w & 0xff000000u) | float_to_unorm(z, 24u)",
                Some(("(zin >> 24)", "(w & 0xffffffu) | (s << 24)")),
            ),
        });
        let depth = depth
            .zip(formats)
            .map(|(stage, (words, read, write, stencil))| {
                (
                    words,
                    read,
                    stage.write.then_some(write),
                    stencil.filter(|_| stage.stencil),
                )
            });
        let _ = writeln!(s, "struct Output {{");
        for (attachment, _) in outputs.iter().enumerate() {
            let _ = writeln!(s, "    float4 c{attachment} [[color({attachment})]];");
        }
        if let Some((words, ..)) = depth {
            let _ = writeln!(s, "    {words} z [[color({})]];", outputs.len());
        }
        let _ = writeln!(s, "}};");
        let mut parameters = if native.is_some() {
            vec![
                "Varyings in [[stage_in]]".to_string(),
                "bool front [[front_facing]]".to_string(),
                "const device float* lod_bias [[buffer(1)]]".to_string(),
            ]
        } else {
            vec![
                "float4 position [[position]]".to_string(),
                "uint primitive [[primitive_id]]".to_string(),
                "const device float* planes [[buffer(0)]]".to_string(),
                "const device float* lod_bias [[buffer(1)]]".to_string(),
                "const device int* triangles [[buffer(24)]]".to_string(),
            ]
        };
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
        for (i, kind) in self.kinds.iter().enumerate() {
            parameters.push(format!("{} t{i} [[texture({i})]]", kind.declaration()));
            parameters.push(format!("sampler s{i} [[sampler({i})]]"));
        }
        if !self.textures.is_empty() {
            parameters.push("device atomic_uint* texture_faults [[buffer(21)]]".to_string());
        }
        if !self.globals.is_empty() {
            parameters.push(format!(
                "device atomic_uint* gm [[buffer({GLOBAL_BUFFER})]]"
            ));
        }
        if alpha_test {
            parameters.push("constant uint* alpha_test [[buffer(22)]]".to_string());
        }
        if let Some((words, _, _, stencil)) = depth {
            parameters.push("constant uint* depth_test [[buffer(23)]]".to_string());
            parameters.push(format!("{words} zin [[color({})]]", outputs.len()));
            if stencil.is_some() {
                parameters.push("constant uint* stencil_test [[buffer(25)]]".to_string());
            }
            if native.is_some() {
                parameters.push("constant float* depth_bias [[buffer(27)]]".to_string());
            }
        }
        if multisample.is_some() && native.is_none() {
            parameters.push("uint layer [[render_target_array_index]]".to_string());
            parameters.push("constant uint* sample_positions [[buffer(26)]]".to_string());
        }
        let _ = writeln!(
            s,
            "fragment Output fragment_main({}) {{",
            parameters.join(", ")
        );
        if native.is_some() {
            let _ = writeln!(
                s,
                "    const float px = in.position.x; const float py = in.position.y;"
            );
            if multisample.is_some() {
                let _ = writeln!(s, "    const uint layer = in.layer;");
            }
        } else {
            let _ = writeln!(
                s,
                "    const float px = position.x; const float py = position.y;\n    const uint plane_base = (primitive >> 1u) * {}u;\n    const device int* tri = triangles + (primitive >> 1u) * {TRIANGLE_WORDS}u;",
                self.ipa_words.len().max(1)
            );
        }
        // The centroid (cpx, cpy) `IPA` CENTROID evaluates at. Native
        // draws interpolate where Metal samples the pixel (the sample's
        // position in a multisampled layer, see `metal`).
        match multisample {
            Some(stage) if native.is_none() => {
                let _ = writeln!(
                    s,
                    "    uint coverage_mask = 0u;\n    for (uint k = 0u; k < {n}u; k++) {{ const uint p = sample_positions[k]; coverage_mask |= covers_at(tri, px, py, p & 0xffu, p >> 8) ? 1u << k : 0u; }}\n    float cpx = px; float cpy = py;\n    if (coverage_mask != 0u && coverage_mask != {all:#x}u) {{ const uint p = sample_positions[ctz(coverage_mask)]; cpx = (px - 0.5f) + float(p & 0xffu) / 16.0f; cpy = (py - 0.5f) + float(p >> 8) / 16.0f; }}",
                    n = stage.samples,
                    all = (1u32 << stage.samples) - 1,
                );
            }
            _ => {
                let _ = writeln!(s, "    const float cpx = px; const float cpy = py;");
            }
        }
        // The depth slope polygon offset needs, taken while the whole
        // quad runs.
        if native.is_some() && depth.is_some() {
            let _ = writeln!(
                s,
                "    const float zslope = max(fabs(dfdx(in.depth)), fabs(dfdy(in.depth)));"
            );
        }
        // IPA{k} and IPAC{k}: plane k at the center and at the centroid.
        for (k, &word) in self.ipa_words.iter().enumerate() {
            let (center, centroid) = match native {
                Some(vertex) => {
                    let value = vertex.varying(word, k);
                    (value.clone(), value)
                }
                None => (
                    format!("plane(planes, plane_base + {k}u, px, py)"),
                    format!("plane(planes, plane_base + {k}u, cpx, cpy)"),
                ),
            };
            let _ = writeln!(s, "#define IPA{k} {center}\n#define IPAC{k} {centroid}");
        }
        // CMP_CFG(i), CMP_MIN(i), CMP_MAX(i): the depth comparison of
        // texture slot i and its range of levels.
        let n = self.textures.len();
        let _ = writeln!(
            s,
            "#define CMP_CFG(i) as_type<uint>(lod_bias[{}u + (i)])\n#define CMP_MIN(i) lod_bias[{}u + (i)]\n#define CMP_MAX(i) lod_bias[{}u + (i)]",
            n + 1,
            2 * n + 1,
            3 * n + 1
        );
        // GM_LIVE: whether this fragment is one of the primitive's (a
        // global memory access of any other reports and stores nothing);
        // GM(i): word i of the bound memory.
        let live = match (native, multisample) {
            (Some(_), _) => "!simd_is_helper_thread()",
            (None, Some(_)) => "(((coverage_mask >> layer) & 1u) != 0u)",
            (None, None) => "covers(tri, px, py)",
        };
        let _ = writeln!(s, "#define GM_LIVE {live}");
        if !self.globals.is_empty() {
            let _ = writeln!(s, "{GLOBAL_WORD}");
        }
        s.push_str(&self.body);
        for k in 0..self.ipa_words.len() {
            let _ = writeln!(s, "#undef IPA{k}\n#undef IPAC{k}");
        }
        let _ = writeln!(
            s,
            "#undef GM_LIVE\n#undef CMP_CFG\n#undef CMP_MIN\n#undef CMP_MAX"
        );
        if !self.globals.is_empty() {
            let _ = writeln!(s, "#undef GM");
        }
        let covered = match (native, multisample) {
            (Some(_), _) => "false",
            (None, Some(_)) => "((coverage_mask >> layer) & 1u) == 0u",
            (None, None) => "!covers(tri, px, py)",
        };
        let _ = writeln!(s, "    if (killed || {covered}) {{ discard_fragment(); }}");
        let (registers, depth_register) = output_registers(program);
        // Output 0 alpha (the draw refuses an alpha test or alpha to
        // coverage that needs it when the program does not write it).
        let alpha = match registers[0] {
            Some((base, mask)) if mask & 8 != 0 => format!("as_type<float>(r{})", base + 3),
            _ => "0.0f".into(),
        };
        if alpha_test {
            let _ = writeln!(
                s,
                "    if (!compare(alpha_test[0], {alpha}, as_type<float>(alpha_test[1]))) {{ discard_fragment(); }}"
            );
        }
        if let Some(stage) = multisample
            && stage.alpha_to_coverage
        {
            let _ = writeln!(
                s,
                "    if (((alpha_coverage({alpha}, {}u) >> layer) & 1u) == 0u) {{ discard_fragment(); }}",
                stage.samples
            );
        }
        let _ = writeln!(s, "    Output out;");
        if let Some((words, read, write, stencil)) = depth {
            let incoming = match (depth_register, multisample, native) {
                (Some(register), _, _) => format!("as_type<float>(r{register})"),
                // Window z where Metal samples, offset (see
                // `NativeVertex::depth_bias`).
                (None, _, Some(_)) => "in.depth + native_depth_bias(depth_bias, in.depth, zslope)".into(),
                (None, Some(stage), None) if !stage.center_depth => "(as_type<float>(tri[9]) + as_type<float>(tri[7]) * ((px - 0.5f) + float(sample_positions[layer] & 0xffu) / 16.0f)) + as_type<float>(tri[8]) * ((py - 0.5f) + float(sample_positions[layer] >> 8) / 16.0f)".into(),
                (None, _, None) => "(as_type<float>(tri[9]) + as_type<float>(tri[7]) * px) + as_type<float>(tri[8]) * py".into(),
            };
            let _ = writeln!(
                s,
                "    {{ float z = {incoming};\n      const float lo = as_type<float>(depth_test[1]); const float hi = as_type<float>(depth_test[2]);\n      z = z < lo ? lo : (z > hi ? hi : z);\n      {words} w = zin;"
            );
            match stencil {
                Some((read_stencil, write_stencil)) => {
                    let face = if native.is_some() {
                        "(front ? 0u : 7u)"
                    } else {
                        "(((uint(tri[6]) >> 3u) & 1u) != 0u ? 7u : 0u)"
                    };
                    let _ = writeln!(
                        s,
                        "      constant uint* st = stencil_test + {face};\n      const uint s0 = {read_stencil};\n      const bool stencil_passed = compare(st[3], float(st[4] & st[5]), float(s0 & st[5]));\n      const bool passed = stencil_passed && compare(depth_test[0], z, {read});\n      const uint s = (s0 & ~st[6]) | (stencil_op(!stencil_passed ? st[0] : (!passed ? st[1] : st[2]), s0, st[4]) & st[6]);\n      w = {write_stencil};"
                    );
                }
                None => {
                    let _ = writeln!(
                        s,
                        "      const bool passed = compare(depth_test[0], z, {read});"
                    );
                }
            }
            if let Some(write) = write {
                let _ = writeln!(s, "      if (passed) {{ w = {write}; }}");
            }
            let failed = if stencil.is_some() && outputs.is_empty() {
                "out.z = w; return out;"
            } else {
                "discard_fragment();"
            };
            let _ = writeln!(s, "      if (!passed) {{ {failed} }}\n      out.z = w; }}");
        }
        // Outputs as the interpreter gathers them: each target with a
        // component mask takes the next four registers from R0; depth is
        // one register after the colors.
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
            let value = if multisample.is_some_and(|stage| stage.alpha_to_one) {
                format!("float4({value}.xyz, 1.0f)")
            } else {
                value
            };
            let _ = writeln!(s, "    out.c{attachment} = {value};");
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
        stage: Stage::Pixel,
        inputs: BTreeSet::new(),
        outputs: BTreeSet::new(),
        out: String::new(),
        registers: BTreeSet::new(),
        predicates: BTreeSet::new(),
        cc: false,
        ipa_words: Vec::new(),
        textures: Vec::new(),
        kinds: Vec::new(),
        implicit_lod: Vec::new(),
        compare: Vec::new(),
        constants: BTreeSet::new(),
        indirect: BTreeSet::new(),
        globals: memory::candidates(program, &flow.pcs),
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
    let local = memory::local_words(program, &flow.pcs);
    if local != 0 {
        let _ = writeln!(
            body,
            "    const uint LM_WORDS = {local}u; uint lm[{local}] = {{}};"
        );
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
        kinds: writer.kinds,
        constants: writer.constants,
        indirect: writer.indirect,
        globals: writer.globals,
        stores: memory::stores(program, &flow.pcs),
        atomics: flow
            .pcs
            .iter()
            .any(|&pc| program.code[(pc / 8) as usize].expect("instruction").op == Op::Red),
        implicit_lod: writer.implicit_lod,
        compare: writer.compare,
        constant_buffer_1: flow.jump_tables.then(|| constant_buffer_1.to_vec()),
    })
}
