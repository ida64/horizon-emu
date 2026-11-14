//! The interpreter.
//!
//! # Model
//!
//! Each invocation is a *lane* with its own architectural state: 255
//! general registers R0-R254 plus RZ (R255, reads 0, writes discarded),
//! predicates P0-P6 plus PT (always true, writes discarded), the condition
//! code (zero, sign, carry, overflow flags read by `.X` forms and
//! CC-conditional branches), and a stack of pending `SSY`/`PBK`/`PCNT`
//! targets. Registers, predicates and flags start at zero (hardware leaves
//! them undefined; no firmware program reads one before writing it).
//!
//! A vertex, and a geometry shader thread, run as one lane. A fragment
//! quad runs its four lanes in
//! lockstep: each step takes the smallest program counter among lanes
//! still running and executes that instruction for every lane at it. For
//! structured control flow this reproduces the hardware's reconvergence
//! (lanes that branch ahead wait until the others reach them), and it lets
//! instructions that need the whole quad see their neighbours: implicit
//! level-of-detail texture sampling uses the coarse quad derivatives
//! `ddx = v(lane 1) - v(lane 0)`, `ddy = v(lane 2) - v(lane 0)` of the
//! coordinate registers. A lane that is not at the current instruction
//! contributes its registers as they are (on hardware such derivatives
//! are undefined).
//!
//! The stack operations are per lane: `SSY`/`PBK`/`PCNT` push their target
//! with its kind; `SYNC` pops an `SSY` entry and jumps to it; `BRK` pops
//! up to and including the innermost `PBK` entry (dropping inner `SSY`
//! entries the breaking lane leaves behind) and jumps to it; `CONT` does
//! the same for `PCNT`. That is the warp-level reconvergence stack of
//! [public: Ryujinx `Decoders/Decoder.cs` (`PushOpInfo`, `IsPopBranch`)]
//! seen from one lane. These three, like `SSY`/`PBK`/`PCNT`, may be
//! predicated or CC-conditional; a lane that does not take them falls
//! through.
//!
//! A program that runs more than [`STEP_LIMIT`] instructions in one
//! invocation, or reaches its terminating branch-to-self (it would never
//! end on hardware), stops with [`Unmodeled`].
//!
//! # Geometry shaders
//!
//! A geometry shader thread sees one input primitive and writes output
//! vertices through opaque handles [public: NVIDIA open-gpu-doc
//! Shader-Program-Header, `MaxOutputVertexCount`, "ALD.O is disallowed"
//! for geometry shaders; envytools `envydis/gm107.c` `isberd`, `out`;
//! eden `translate/impl/output_geometry.cpp`; Ryujinx
//! `Instructions/InstEmitAttribute.cs` `Isberd`]. As the NVN compiler
//! uses them (ocean archive of the corpus, and the language scene of
//! starter):
//!
//! * `S2R SR_INVOCATION_INFO` gives the vertices per input primitive in
//!   bits 16-23 and the primitive's position in the internal stage buffer
//!   (ISBE) batch in bits 0-7; the program forms `b0 * b2 + vertex` with
//!   `VMAD.U8` and reads that ISBE entry with `ISBERD`, getting the input
//!   vertex's handle, which `ALD`'s vertex register takes.
//! * Output vertices are built with `AST` whose vertex register holds the
//!   current output handle (the compiler starts it at 0); `OUT.EMIT Rd,
//!   Ra` emits the vertex of handle `Ra` and returns the next handle in
//!   `Rd`; `OUT.CUT` ends the strip.
//!
//! Handle values are the hardware's own; this model fills the ISBE with
//! one primitive at a time (position 0, other bits of
//! `SR_INVOCATION_INFO` 0, as Ryujinx returns), lets entry `i` hold input
//! handle `i` (the identity eden and Ryujinx use for `ISBERD`), and
//! numbers output handles by the vertices emitted so far. A program that
//! uses a handle it was not given (another ISBE entry, a made-up vertex)
//! is refused, so the choice of values cannot show. An emitted vertex
//! starts the next one from zeros: after `OUT.EMIT` the outputs are
//! undefined (the compiler stores every output again).
//!
//! # Instructions executed
//!
//! Float: `FADD`, `FADD32I`, `FMUL`, `FMUL32I`, `FFMA`, `FFMA32I`, `FMNMX`,
//! `FSET`, `FSETP`, `MUFU`, `RRO`, `F2I`, `I2F`, `F2F` (32-bit float forms).
//! Integer: `IADD`, `IADD32I`, `IADD3`, `ISCADD`, `ISCADD32I`, `IMNMX`,
//! `ISETP`, `ISET`, `ICMP`, `LOP`, `LOP32I`, `LOP3`, `SHL`, `SHR`, `BFE`,
//! `BFI`, `XMAD`, `SEL`, `MOV`, `MOV32I`, `VMAD`, `VSETP`.
//! Predicates: `PSETP`, `P2R`, `R2P`. System: `S2R SR_Y_DIRECTION`,
//! `S2R SR_INVOCATION_INFO` (geometry). Memory: `LDC`, `LDG`, `STG`,
//! `LD`, `ST` (global through [`super::Env::memory`]). Attributes: `ALD`,
//! `AST` (vertex, geometry), `ISBERD`, `OUT` (geometry), `IPA`
//! (fragment). Textures: `TEX`, `TEX.B`, `TEXS`, `TLD`, `TLD.B`, `TLDS`,
//! `TLD4`, `TLD4.B`, `TLD4S`, `TXQ`, `TXQ.B`, `TMML`, `TMML.B`, `TXD`,
//! `TXD.B`. Control: `BRA`, `BRX`, `SSY`, `SYNC`, `PBK`, `BRK`, `PCNT`,
//! `CONT`, `EXIT`, `KIL`, `NOP`, `DEPBAR`, `BAR`, `MEMBAR`.
//!
//! Every other decodable instruction, and every modifier of the above that
//! is not modeled (each is named at its check), returns [`Unmodeled`] when
//! executed, naming the mnemonic, what is unmodeled, the word and the pc.
//!
//! `DEPBAR` (wait for scoreboard), `BAR` (CTA barrier) and `MEMBAR`
//! (memory ordering) have no observable effect here: an invocation runs to
//! completion, each instruction's results are visible to the next one, and
//! memory accesses happen in program order. The `NOP` field `TRIG` and
//! the cache-policy fields of `LDG`/`STG`/`LD`/`ST` are hints that do not
//! change values in a model without caches.

mod alu;
mod integer;
mod tex;
mod texture_vector;
mod video;

use super::super::Unmodeled;
use super::decode::Op;
use super::{Attributes, Env, FragmentOutput, Plane, Program, Quad, ShaderType};

/// Instructions one invocation may execute before it is declared stuck.
pub const STEP_LIMIT: u64 = 1 << 20;

/// Depth of each lane's reconvergence stack.
const STACK_DEPTH: usize = 32;

const RZ: u32 = 255;
const PT: u32 = 7;

/// Condition code flags.
const CC_Z: u8 = 1;
const CC_S: u8 = 2;
const CC_C: u8 = 4;
const CC_O: u8 = 8;

/// One decoded instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Inst {
    pub op: Op,
    pub word: u64,
}

#[inline]
pub(super) fn bits(word: u64, low: u32, count: u32) -> u32 {
    ((word >> low) & ((1u64 << count) - 1)) as u32
}

#[inline]
pub(super) fn bit(word: u64, index: u32) -> bool {
    (word >> index) & 1 != 0
}

#[inline]
pub(super) fn sext(value: u32, width: u32) -> i32 {
    ((value << (32 - width)) as i32) >> (32 - width)
}

/// The byte offset a branch at `pc` with `word` lands on: `pc + 8 +
/// imm24` [public: Ryujinx `Decoders/InstOp.cs` `GetAbsoluteAddress`]; a
/// target on a scheduling word continues at the next instruction
/// [public: Ryujinx `Decoders/Decoder.cs` `FillBlock`].
pub(super) fn branch_target(word: u64, pc: u32) -> Option<u32> {
    let target = i64::from(pc) + 8 + i64::from(sext(bits(word, 20, 24), 24));
    let target = u32::try_from(target).ok()?;
    Some(if target % 32 == 0 { target + 8 } else { target })
}

/// Whether `word` (a `BRA`) at `pc` is the compiler's terminating
/// unconditional branch to itself: guard PT, condition code T (15), no
/// `CA`/`LMT`/`U` flags, target `pc`.
pub(super) fn is_branch_to_self(word: u64, pc: u32) -> bool {
    word & 0xf_00ff == 0x7_000f && branch_target(word, pc) == Some(pc)
}

fn next_pc(pc: u32) -> u32 {
    let next = pc + 8;
    if next.is_multiple_of(32) {
        next + 8
    } else {
        next
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Running,
    Exited,
    Killed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Sync,
    Break,
    Continue,
}

#[derive(Clone)]
struct Lane {
    r: [u32; 256],
    /// Bit i: Pi; bit 7 (PT) always set.
    p: u8,
    cc: u8,
    pc: u32,
    depth: usize,
    stack: [(Kind, u32); STACK_DEPTH],
    state: State,
}

impl Lane {
    fn new() -> Lane {
        Lane {
            r: [0; 256],
            p: 0x80,
            cc: 0,
            pc: 8,
            depth: 0,
            stack: [(Kind::Sync, 0); STACK_DEPTH],
            state: State::Running,
        }
    }
}

enum Io<'a> {
    Vertex {
        input: &'a Attributes,
        output: &'a mut Attributes,
    },
    Geometry(Geometry<'a>),
    Fragment {
        planes: &'a [Plane; 256],
        x: u32,
        y: u32,
        covered: [bool; 4],
    },
}

/// The inputs and outputs of a geometry shader thread.
struct Geometry<'a> {
    /// The input primitive's vertices; handle `i` is `inputs[i]`.
    inputs: &'a [&'a Attributes],
    /// The output vertex being written: the one of handle `handle`.
    current: &'a mut Attributes,
    /// Output handle: the vertices this thread has emitted.
    handle: u32,
    /// `MaxOutputVertexCount`.
    max: u32,
    emitted: &'a mut Vec<Option<Attributes>>,
}

struct Machine<'a, const L: usize> {
    program: &'a Program,
    env: &'a Env<'a>,
    lanes: [Lane; L],
    io: Io<'a>,
    /// The instruction being executed.
    op: Op,
    word: u64,
    pc: u32,
}

pub(super) fn run_vertex(
    program: &Program,
    env: &Env<'_>,
    input: &Attributes,
    output: &mut Attributes,
) -> Result<(), Unmodeled> {
    if program.header.shader_type() != ShaderType::Vertex {
        return Err(Unmodeled(format!(
            "shader: run_vertex on a {:?} program",
            program.header.shader_type()
        )));
    }
    let mut machine = Machine::<1> {
        program,
        env,
        lanes: [Lane::new()],
        io: Io::Vertex { input, output },
        op: Op::Invalid,
        word: 0,
        pc: 0,
    };
    machine.run()?;
    if machine.lanes[0].state == State::Killed {
        return Err(Unmodeled("shader: KIL in a vertex program".into()));
    }
    Ok(())
}

pub(super) fn run_geometry(
    program: &Program,
    env: &Env<'_>,
    inputs: &[&Attributes],
    emitted: &mut Vec<Option<Attributes>>,
) -> Result<(), Unmodeled> {
    let header = &program.header;
    if header.shader_type() != ShaderType::Geometry {
        return Err(Unmodeled(format!(
            "shader: run_geometry on a {:?} program",
            header.shader_type()
        )));
    }
    let mut current = Attributes::default();
    let mut machine = Machine::<1> {
        program,
        env,
        lanes: [Lane::new()],
        io: Io::Geometry(Geometry {
            inputs,
            current: &mut current,
            handle: 0,
            max: header.max_output_vertex_count(),
            emitted,
        }),
        op: Op::Invalid,
        word: 0,
        pc: 0,
    };
    machine.run()?;
    if machine.lanes[0].state == State::Killed {
        return Err(Unmodeled("shader: KIL in a geometry program".into()));
    }
    Ok(())
}

pub(super) fn run_fragment(
    program: &Program,
    env: &Env<'_>,
    quad: &mut Quad<'_>,
) -> Result<(), Unmodeled> {
    let header = &program.header;
    if header.shader_type() != ShaderType::Pixel {
        return Err(Unmodeled(format!(
            "shader: run_fragment on a {:?} program",
            header.shader_type()
        )));
    }
    if header.ps_writes_sample_mask() {
        return Err(Unmodeled("shader: SPH sample-mask output".into()));
    }
    let lanes = [Lane::new(), Lane::new(), Lane::new(), Lane::new()];
    let mut machine = Machine::<4> {
        program,
        env,
        lanes,
        io: Io::Fragment {
            planes: quad.planes,
            x: quad.x,
            y: quad.y,
            covered: quad.covered,
        },
        op: Op::Invalid,
        word: 0,
        pc: 0,
    };
    if !quad.front_facing {
        // Nothing in the modeled instruction set reads the facing of the
        // primitive (IPA of a[0x3fc] is refused), so it has no effect.
    }
    machine.run()?;
    // Outputs: each render target with any enabled component takes four
    // consecutive registers from R0 (component c at base + c), in target
    // order; the depth follows one register after the colors, after the
    // (unmodeled) sample-mask register [public: eden
    // `frontend/maxwell/translate/impl/exit_program.cpp` `ExitFragment`;
    // Ryujinx `Translation/EmitterContext.cs` `PrepareForReturn` agrees on
    // colors, its `TranslatorContext.GetDepthRegister` (popcount of the
    // whole Omap + 1) differs only for targets with partial masks].
    for (index, lane) in machine.lanes.iter().enumerate() {
        let out = &mut quad.outputs[index];
        *out = FragmentOutput::default();
        if !quad.covered[index] {
            continue;
        }
        if lane.state == State::Killed {
            out.discarded = true;
            continue;
        }
        let mut base = 0usize;
        for (target, color) in out.colors.iter_mut().enumerate() {
            let mask = header.ps_color_mask(target);
            if mask == 0 {
                continue;
            }
            for (component, value) in color.iter_mut().enumerate() {
                if mask & (1 << component) != 0 {
                    *value = f32::from_bits(lane.r[base + component]);
                }
            }
            base += 4;
        }
        if header.ps_writes_depth() {
            out.depth = Some(f32::from_bits(lane.r[base + 1]));
        }
    }
    Ok(())
}

impl<'a, const L: usize> Machine<'a, L> {
    fn run(&mut self) -> Result<(), Unmodeled> {
        let mut steps = 0u64;
        loop {
            let pc = self
                .lanes
                .iter()
                .filter(|lane| lane.state == State::Running)
                .map(|lane| lane.pc)
                .min();
            let Some(pc) = pc else {
                return Ok(());
            };
            let mut mask = 0u8;
            for (index, lane) in self.lanes.iter().enumerate() {
                if lane.state == State::Running && lane.pc == pc {
                    mask |= 1 << index;
                }
            }
            let inst = self
                .program
                .code
                .get((pc / 8) as usize)
                .copied()
                .flatten()
                .ok_or_else(|| {
                    Unmodeled(format!(
                        "shader: execution reached {pc:#x}, outside the decoded program"
                    ))
                })?;
            self.op = inst.op;
            self.word = inst.word;
            self.pc = pc;
            steps += 1;
            if steps > STEP_LIMIT {
                return Err(self.unmodeled(&format!("more than {STEP_LIMIT} steps")));
            }
            self.step(mask)?;
        }
    }

    /// Error for the instruction being executed.
    fn unmodeled(&self, what: &str) -> Unmodeled {
        Unmodeled(format!(
            "shader: {} {what} {:#018x} at {:#x}",
            self.op.mnemonic(),
            self.word,
            self.pc
        ))
    }

    fn fail<T>(&self, what: &str) -> Result<T, Unmodeled> {
        Err(self.unmodeled(what))
    }

    fn require(&self, ok: bool, what: impl std::fmt::Display) -> Result<(), Unmodeled> {
        if ok {
            Ok(())
        } else {
            self.fail(&what.to_string())
        }
    }

    fn is_fragment(&self) -> bool {
        matches!(self.io, Io::Fragment { .. })
    }

    // Register file access.

    #[inline]
    fn r(&self, lane: usize, index: u32) -> u32 {
        // Vector operands may run past R254; RZ and beyond read 0
        // [public: Ryujinx `Instructions/InstEmitTexture.cs` `Ra()`/`Rb()`].
        if index >= RZ {
            0
        } else {
            self.lanes[lane].r[index as usize]
        }
    }

    #[inline]
    fn rf(&self, lane: usize, index: u32) -> f32 {
        f32::from_bits(self.r(lane, index))
    }

    #[inline]
    fn set(&mut self, lane: usize, index: u32, value: u32) {
        if index < RZ {
            self.lanes[lane].r[index as usize] = value;
        }
    }

    #[inline]
    fn setf(&mut self, lane: usize, index: u32, value: f32) {
        self.set(lane, index, value.to_bits());
    }

    #[inline]
    fn pred(&self, lane: usize, index: u32) -> bool {
        (self.lanes[lane].p >> index) & 1 != 0
    }

    #[inline]
    fn set_pred(&mut self, lane: usize, index: u32, value: bool) {
        if index != PT {
            let p = &mut self.lanes[lane].p;
            *p = (*p & !(1 << index)) | (u8::from(value) << index);
        }
    }

    /// A predicate operand: index at bits `low..low + 3`, negation at
    /// `low + 3`.
    #[inline]
    fn src_pred(&self, lane: usize, low: u32) -> bool {
        self.pred(lane, bits(self.word, low, 3)) != bit(self.word, low + 3)
    }

    /// A constant buffer word; offsets past the bound size read 0.
    fn cbuf(&self, slot: u32, offset: u32) -> Result<u32, Unmodeled> {
        let mut bytes = [0u8; 4];
        self.cbuf_bytes(slot, offset, &mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn cbuf_bytes(&self, slot: u32, offset: u32, out: &mut [u8]) -> Result<(), Unmodeled> {
        let Some(buffer) = self.env.const_buffers.get(slot as usize) else {
            return self.fail(&format!("constant buffer slot {slot}"));
        };
        let start = offset as usize;
        let end = start + out.len();
        match buffer.data.get(start..end) {
            Some(data) => out.copy_from_slice(data),
            None if end > buffer.size => out.fill(0),
            None => {
                return self.fail(&format!(
                    "constant buffer slot {slot} read at {offset:#x}, past its mapped {:#x} bytes",
                    buffer.data.len()
                ));
            }
        }
        Ok(())
    }

    /// The `c[slot][offset]` operand of `C`/`Rc` forms: slot bits 34-38,
    /// word offset bits 20-33.
    fn cbuf_operand(&self) -> Result<u32, Unmodeled> {
        self.cbuf(bits(self.word, 34, 5), bits(self.word, 20, 14) * 4)
    }

    /// Operand B of the current form.
    #[inline]
    fn b(&self, lane: usize, form: Form) -> Result<u32, Unmodeled> {
        let w = self.word;
        Ok(match form {
            Form::R => self.r(lane, bits(w, 20, 8)),
            Form::I => imm20(w) as u32,
            Form::F => (imm20(w) as u32) << 12,
            Form::C => self.cbuf_operand()?,
            Form::Rc => self.r(lane, bits(w, 39, 8)),
            Form::Imm32 => (w >> 20) as u32,
        })
    }

    /// Operand C of the current form (register 39, or the constant buffer
    /// of `Rc` forms).
    #[inline]
    fn c(&self, lane: usize, form: Form) -> Result<u32, Unmodeled> {
        match form {
            Form::Rc => self.cbuf_operand(),
            _ => Ok(self.r(lane, bits(self.word, 39, 8))),
        }
    }

    /// The instruction's guard predicate (bits 16-18, negation bit 19).
    fn guard(&self, lane: usize) -> bool {
        self.src_pred(lane, 16)
    }

    /// Condition code test of field `Ccc` [public: Ryujinx
    /// `Instructions/InstEmitConditionCode.cs` `GetCondition`].
    fn condition(&self, lane: usize, ccc: u32) -> Result<bool, Unmodeled> {
        let cc = self.lanes[lane].cc;
        let z = cc & CC_Z != 0;
        let n = cc & CC_S != 0;
        let c = cc & CC_C != 0;
        let v = cc & CC_O != 0;
        Ok(match ccc {
            0 => false,
            1 => (n && !z) != v,
            2 => !n && z,
            3 => n != (z || v),
            4 => !((n != v) || z),
            5 => !z,
            6 => n == v,
            7 => !(n && z),
            8 => n && z,
            9 => n != v,
            10 => z,
            11 => (n != v) || z,
            12 => !n != (v || z),
            13 => n || !z,
            14 => (!n || z) != v,
            15 => true,
            16 => !v,
            17 => !c,
            18 => !n,
            19 => z || !c,
            20 => c && !z,
            21 => n,
            22 => c,
            23 => v,
            30 => n || z,
            31 => !(n || z),
            other => return self.fail(&format!("condition code test {other}")),
        })
    }

    fn step(&mut self, mask: u8) -> Result<(), Unmodeled> {
        let next = next_pc(self.pc);
        // SSY, PBK and PCNT have no guard predicate: their encoding is
        // the target (bits 20-43) and CA (bit 5) only, bits 16-19 are not
        // a field [public: Ryujinx `Decoders/InstDecoders.cs` `InstSsy`,
        // `InstPbk`, `InstPcnt`]. Compilers leave them as they please
        // (0, read as P0, in a scene of starter's).
        let unpredicated = matches!(self.op, Op::Ssy | Op::Pbk | Op::Pcnt);
        let mut active = 0u8;
        for lane in 0..L {
            if mask & (1 << lane) != 0 {
                self.lanes[lane].pc = next;
                if unpredicated || self.guard(lane) {
                    active |= 1 << lane;
                }
            }
        }
        if active == 0 {
            return Ok(());
        }
        if self.execute_alu(active)? {
            return Ok(());
        }
        if self.execute_integer(active)? {
            return Ok(());
        }
        if self.execute_texture(active)? {
            return Ok(());
        }
        if self.execute_texture_vector(active)? {
            return Ok(());
        }
        if self.execute_video(active)? {
            return Ok(());
        }
        self.execute_other(active)
    }

    /// Calls `f` for each lane in `active`.
    #[inline]
    fn each(
        &mut self,
        active: u8,
        mut f: impl FnMut(&mut Self, usize) -> Result<(), Unmodeled>,
    ) -> Result<(), Unmodeled> {
        for lane in 0..L {
            if active & (1 << lane) != 0 {
                f(self, lane)?;
            }
        }
        Ok(())
    }

    fn execute_other(&mut self, active: u8) -> Result<(), Unmodeled> {
        let w = self.word;
        match self.op {
            Op::Bra => {
                // CA: target from a constant buffer; LMT: limit the
                // reconvergence stack; neither is modeled. U marks a branch
                // the compiler knows to be warp-uniform: a scheduling hint.
                self.require(!bit(w, 5), "CA (constant-buffer target)")?;
                self.require(!bit(w, 6), "LMT")?;
                if is_branch_to_self(w, self.pc) {
                    return self.fail("terminating branch to itself (would never end)");
                }
                let target = self.target()?;
                self.each(active, |m, lane| {
                    if m.condition(lane, bits(w, 0, 5))? {
                        m.lanes[lane].pc = target;
                    }
                    Ok(())
                })
            }
            Op::Brx => {
                self.require(!bit(w, 5), "CA (constant-buffer target)")?;
                self.require(!bit(w, 6), "LMT")?;
                let base = i64::from(self.pc) + 8 + i64::from(sext(bits(w, 20, 24), 24));
                let ra = bits(w, 8, 8);
                self.each(active, |m, lane| {
                    if !m.condition(lane, bits(w, 0, 5))? {
                        return Ok(());
                    }
                    let target = base + i64::from(m.r(lane, ra) as i32);
                    let Ok(target) = u32::try_from(target) else {
                        return m.fail(&format!("target {target:#x}"));
                    };
                    let target = if target % 32 == 0 { target + 8 } else { target };
                    m.require(
                        target % 8 == 0,
                        format_args!("unaligned target {target:#x}"),
                    )?;
                    m.lanes[lane].pc = target;
                    Ok(())
                })
            }
            Op::Ssy | Op::Pbk | Op::Pcnt => {
                self.require(!bit(w, 5), "CA (constant-buffer target)")?;
                let target = self.target()?;
                let kind = match self.op {
                    Op::Ssy => Kind::Sync,
                    Op::Pbk => Kind::Break,
                    _ => Kind::Continue,
                };
                self.each(active, |m, lane| {
                    let l = &mut m.lanes[lane];
                    if l.depth == STACK_DEPTH {
                        return m.fail("reconvergence stack overflow");
                    }
                    l.stack[l.depth] = (kind, target);
                    l.depth += 1;
                    Ok(())
                })
            }
            Op::Sync | Op::Brk | Op::Cont => {
                let kind = match self.op {
                    Op::Sync => Kind::Sync,
                    Op::Brk => Kind::Break,
                    _ => Kind::Continue,
                };
                self.each(active, |m, lane| {
                    if !m.condition(lane, bits(w, 0, 5))? {
                        return Ok(());
                    }
                    let l = &mut m.lanes[lane];
                    loop {
                        if l.depth == 0 {
                            return m.fail("no matching stack entry");
                        }
                        l.depth -= 1;
                        let (entry, target) = l.stack[l.depth];
                        if entry == kind {
                            l.pc = target;
                            return Ok(());
                        }
                        // SYNC pops only its own SSY entry; BRK and CONT
                        // unwind inner SSY entries.
                        if kind == Kind::Sync || entry != Kind::Sync {
                            return m.fail(&format!("stack holds {entry:?}"));
                        }
                    }
                })
            }
            Op::Exit => {
                self.require(!bit(w, 5), "KEEPREFCOUNT")?;
                self.each(active, |m, lane| {
                    if m.condition(lane, bits(w, 0, 5))? {
                        m.lanes[lane].state = State::Exited;
                    }
                    Ok(())
                })
            }
            Op::Kil => {
                self.require(self.is_fragment(), "outside a fragment program")?;
                self.each(active, |m, lane| {
                    if m.condition(lane, bits(w, 0, 5))? {
                        m.lanes[lane].state = State::Killed;
                    }
                    Ok(())
                })
            }
            Op::Nop | Op::Depbar | Op::Membar | Op::Bar => Ok(()),
            Op::S2r => {
                let sr = bits(w, 20, 8);
                // SR_Y_DIRECTION (18), SR_INVOCATION_INFO (29) [public:
                // eden `translate/impl/move_special_register.cpp`]; see the
                // module documentation for the latter.
                let value = match (sr, &self.io) {
                    (18, _) => self.env.y_direction.to_bits(),
                    (29, Io::Geometry(geometry)) => (geometry.inputs.len() as u32) << 16,
                    _ => return self.fail(&format!("special register {sr}")),
                };
                let rd = bits(w, 0, 8);
                self.each(active, |m, lane| {
                    m.set(lane, rd, value);
                    Ok(())
                })
            }
            Op::Isberd => self.isberd(active),
            Op::OutR | Op::OutI | Op::OutC => self.out(active),
            Op::Ldc => self.ldc(active),
            Op::Ldg | Op::Ld => self.load_global(active),
            Op::Stg | Op::St => self.store_global(active),
            Op::Ald => self.ald(active),
            Op::Ast => self.ast(active),
            Op::Ipa => self.ipa(active),
            _ => self.fail("not modeled"),
        }
    }

    /// The target of `BRA`/`SSY`/`PBK`/`PCNT`.
    fn target(&self) -> Result<u32, Unmodeled> {
        match branch_target(self.word, self.pc) {
            Some(target) => Ok(target),
            None => self.fail("target before the program"),
        }
    }

    /// `LDC`: dest 0-7, address register 8-15, offset 20-35 (signed),
    /// slot 36-40, address mode 44-45, size 48-50 [public: Ryujinx
    /// `Decoders/InstDecoders.cs` `InstLdc`, `Instructions/InstEmitMemory.cs`
    /// `Ldc`].
    fn ldc(&mut self, active: u8) -> Result<(), Unmodeled> {
        let w = self.word;
        self.require(bits(w, 44, 2) == 0, "address mode IL/IS/ISL")?;
        let size = bits(w, 48, 3);
        let slot = bits(w, 36, 5);
        let offset = sext(bits(w, 20, 16), 16) as u32;
        let (rd, ra) = (bits(w, 0, 8), bits(w, 8, 8));
        self.require(size <= 5, format_args!("size {size}"))?;
        self.require(
            size != 5 || rd % 2 == 0 || rd == RZ,
            "B64 to an odd register",
        )?;
        self.each(active, |m, lane| {
            let address = m.r(lane, ra).wrapping_add(offset);
            let width = [1u32, 1, 2, 2, 4, 8][size as usize];
            m.require(
                address % width == 0,
                format_args!("unaligned address {address:#x}"),
            )?;
            let mut bytes = [0u8; 8];
            m.cbuf_bytes(slot, address, &mut bytes[..width as usize])?;
            let low = u32::from_le_bytes(bytes[..4].try_into().expect("4 bytes"));
            let value = match size {
                0 => low & 0xff,
                1 => low as u8 as i8 as i32 as u32,
                2 => low & 0xffff,
                3 => low as u16 as i16 as i32 as u32,
                _ => low,
            };
            m.set(lane, rd, value);
            if size == 5 && rd != RZ {
                let high = u32::from_le_bytes(bytes[4..].try_into().expect("4 bytes"));
                m.set(lane, rd + 1, high);
            }
            Ok(())
        })
    }

    /// Address, size code and register of `LDG`/`STG` (address register
    /// 8-15, offset 20-43 signed, `E` (64-bit address) bit 45, size 48-50)
    /// or `LD`/`ST` (offset 20-51, `E` bit 52, size 53-55, predicate
    /// 58-60) [public: Ryujinx `Decoders/InstDecoders.cs` `InstLdg`,
    /// `InstStg`, `InstLd`, `InstSt`].
    fn global_access(&self, lane: usize) -> Result<(u64, u32), Unmodeled> {
        let w = self.word;
        let generic = matches!(self.op, Op::Ld | Op::St);
        let (offset, e, size) = if generic {
            (
                i64::from((w >> 20) as u32 as i32),
                bit(w, 52),
                bits(w, 53, 3),
            )
        } else {
            (
                i64::from(sext(bits(w, 20, 24), 24)),
                bit(w, 45),
                bits(w, 48, 3),
            )
        };
        let ra = bits(w, 8, 8);
        let base = if e {
            u64::from(self.r(lane, ra)) | (u64::from(self.r(lane, ra + 1)) << 32)
        } else {
            u64::from(self.r(lane, ra))
        };
        Ok((base.wrapping_add(offset as u64), size))
    }

    fn check_global(&self) -> Result<(), Unmodeled> {
        let w = self.word;
        if matches!(self.op, Op::Ld | Op::St) {
            // LD/ST through the generic address space reach global memory
            // [public: Ryujinx `Instructions/InstEmitMemory.cs` `Ld`/`St`
            // (`MemoryRegion.Global`)]; a non-PT predicate operand is not
            // modeled.
            self.require(bits(w, 58, 3) == PT, "predicate operand")?;
        }
        let ra = bits(w, 8, 8);
        let e = if matches!(self.op, Op::Ld | Op::St) {
            bit(w, 52)
        } else {
            bit(w, 45)
        };
        self.require(
            !e || ra.is_multiple_of(2) || ra == RZ,
            "64-bit address in an odd register",
        )
    }

    fn load_global(&mut self, active: u8) -> Result<(), Unmodeled> {
        self.check_global()?;
        let rd = bits(self.word, 0, 8);
        self.each(active, |m, lane| {
            let (address, size) = m.global_access(lane)?;
            let width = global_width(size).ok_or_else(|| m.unmodeled(&format!("size {size}")))?;
            m.require(
                address % width as u64 == 0,
                format_args!("unaligned address {address:#x}"),
            )?;
            let count = width.div_ceil(4) as u32;
            m.require(
                rd == RZ || rd.is_multiple_of(count),
                "vector destination not aligned",
            )?;
            let mut bytes = [0u8; 16];
            if m.env.memory.read(address, &mut bytes[..width]).is_none() {
                return m.fail(&format!("load from unmapped {address:#x}"));
            }
            if rd == RZ {
                return Ok(());
            }
            let word =
                |i: usize| u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().expect("4"));
            match size {
                0 => m.set(lane, rd, u32::from(bytes[0])),
                1 => m.set(lane, rd, bytes[0] as i8 as i32 as u32),
                2 => m.set(
                    lane,
                    rd,
                    u32::from(u16::from_le_bytes([bytes[0], bytes[1]])),
                ),
                3 => m.set(
                    lane,
                    rd,
                    i16::from_le_bytes([bytes[0], bytes[1]]) as i32 as u32,
                ),
                _ => {
                    for i in 0..count {
                        m.set(lane, rd + i, word(i as usize));
                    }
                }
            }
            Ok(())
        })
    }

    fn store_global(&mut self, active: u8) -> Result<(), Unmodeled> {
        self.check_global()?;
        let rd = bits(self.word, 0, 8);
        let helpers = match self.io {
            Io::Fragment { covered, .. } => covered,
            Io::Vertex { .. } | Io::Geometry(_) => [true; 4],
        };
        self.each(active, |m, lane| {
            // Helper lanes (outside the primitive) do not write memory.
            if !helpers[lane] {
                return Ok(());
            }
            let (address, size) = m.global_access(lane)?;
            let width = global_width(size).ok_or_else(|| m.unmodeled(&format!("size {size}")))?;
            m.require(
                address % width as u64 == 0,
                format_args!("unaligned address {address:#x}"),
            )?;
            let count = width.div_ceil(4) as u32;
            m.require(
                rd == RZ || rd.is_multiple_of(count),
                "vector source not aligned",
            )?;
            let mut bytes = [0u8; 16];
            for i in 0..count {
                let value = m.r(lane, rd + i);
                bytes[i as usize * 4..i as usize * 4 + 4].copy_from_slice(&value.to_le_bytes());
            }
            if m.env.memory.write(address, &bytes[..width]).is_none() {
                return m.fail(&format!("store to unmapped {address:#x}"));
            }
            Ok(())
        })
    }

    /// `ALD`: dest 0-7, index register 8-15, offset 20-30, `P` (patch)
    /// bit 31, `O` (read outputs) bit 32, vertex register 39-46, size
    /// 47-48 (1-4 words) [public: eden
    /// `translate/impl/load_store_attribute.cpp` `ALD`]. In a geometry
    /// program the vertex register holds an input vertex handle.
    fn ald(&mut self, active: u8) -> Result<(), Unmodeled> {
        let w = self.word;
        self.require(!bit(w, 31), "P (patch)")?;
        self.require(bits(w, 8, 8) == RZ, "indexed address")?;
        let offset = bits(w, 20, 11);
        let count = bits(w, 47, 2) + 1;
        self.require(
            offset.is_multiple_of(4) && offset / 4 + count <= 256,
            "attribute address",
        )?;
        let (rd, vertex) = (bits(w, 0, 8), bits(w, 39, 8));
        let outputs = bit(w, 32);
        self.each(active, |m, lane| {
            let source = match &m.io {
                Io::Vertex { input, output } => {
                    m.require(vertex == RZ, "vertex register outside a geometry program")?;
                    if outputs { &output.0 } else { &input.0 }
                }
                Io::Geometry(geometry) => {
                    m.require(!outputs, "O (disallowed in a geometry program)")?;
                    let handle = m.r(lane, vertex);
                    match geometry.inputs.get(handle as usize) {
                        Some(input) => &input.0,
                        None => return m.fail(&format!("input vertex handle {handle:#x}")),
                    }
                }
                Io::Fragment { .. } => return m.fail("in a fragment program"),
            };
            let mut values = [0u32; 4];
            for (i, value) in values.iter_mut().take(count as usize).enumerate() {
                *value = source[(offset / 4) as usize + i];
            }
            for i in 0..count {
                m.set(lane, rd + i, values[i as usize]);
            }
            Ok(())
        })
    }

    /// `AST`: source 0-7, index register 8-15, offset 20-30, `P` bit 31,
    /// vertex register 39-46, size 47-48 [public: eden
    /// `translate/impl/load_store_attribute.cpp` `AST`]. In a geometry
    /// program the vertex register holds the current output handle.
    fn ast(&mut self, active: u8) -> Result<(), Unmodeled> {
        let w = self.word;
        self.require(!bit(w, 31), "P (patch)")?;
        self.require(bits(w, 8, 8) == RZ, "indexed address")?;
        let offset = bits(w, 20, 11);
        let count = bits(w, 47, 2) + 1;
        self.require(
            offset.is_multiple_of(4) && offset / 4 + count <= 256,
            "attribute address",
        )?;
        let rs = bits(w, 0, 8);
        // One lane in vertex and geometry programs.
        let values: [u32; 4] = std::array::from_fn(|i| self.r(0, rs + i as u32));
        if active & 1 == 0 {
            return Ok(());
        }
        let vertex = bits(w, 39, 8);
        let handle = self.r(0, vertex);
        match &self.io {
            Io::Vertex { .. } if vertex != RZ => {
                return self.fail("vertex register outside a geometry program");
            }
            Io::Geometry(geometry) if geometry.handle != handle => {
                return self.fail(&format!("output vertex handle {handle:#x}"));
            }
            Io::Fragment { .. } => return self.fail("in a fragment program"),
            _ => {}
        }
        let output = match &mut self.io {
            Io::Vertex { output, .. } => &mut **output,
            Io::Geometry(geometry) => &mut *geometry.current,
            Io::Fragment { .. } => unreachable!("refused above"),
        };
        for (i, value) in values.iter().take(count as usize).enumerate() {
            output.0[(offset / 4) as usize + i] = *value;
        }
        Ok(())
    }

    /// `ISBERD`: dest 0-7, address register 8-15, `SKEW` bit 31, `O` bit
    /// 32, mode 33-34 (default, patch, prim, attr), size 47-48 [public:
    /// envytools `envydis/gm107.c` `isberd`]. Only the default form, which
    /// reads the input vertex handle of an ISBE entry (see the module
    /// documentation), is modeled.
    fn isberd(&mut self, active: u8) -> Result<(), Unmodeled> {
        let w = self.word;
        let Io::Geometry(geometry) = &self.io else {
            return self.fail("outside a geometry program");
        };
        let vertices = geometry.inputs.len() as u32;
        // Bits 20-30 are no field in envytools; eden reads an offset
        // from 24-31.
        self.require(bits(w, 20, 11) == 0, "bits 20-30")?;
        self.require(bits(w, 31, 4) == 0, "SKEW, O or a mode")?;
        self.require(bits(w, 47, 2) == 0, "size")?;
        let (rd, ra) = (bits(w, 0, 8), bits(w, 8, 8));
        self.each(active, |m, lane| {
            let entry = m.r(lane, ra);
            m.require(
                entry < vertices,
                format_args!("ISBE entry {entry} of a {vertices}-vertex primitive"),
            )?;
            m.set(lane, rd, entry);
            Ok(())
        })
    }

    /// `OUT`: dest 0-7, handle register 8-15, stream operand B, `EMIT` bit
    /// 39, `CUT` bit 40 [public: envytools `envydis/gm107.c` `out`, eden
    /// `translate/impl/output_geometry.cpp`]. Emits beyond
    /// `MaxOutputVertexCount` are no-ops (Shader-Program-Header). Only
    /// stream 0 is modeled.
    fn out(&mut self, active: u8) -> Result<(), Unmodeled> {
        let w = self.word;
        let form = match self.op {
            Op::OutR => Form::R,
            Op::OutI => Form::I,
            _ => Form::C,
        };
        let (emit, cut) = (bit(w, 39), bit(w, 40));
        self.require(emit || cut, "neither EMIT nor CUT")?;
        let current = match &self.io {
            Io::Geometry(geometry) => geometry.handle,
            _ => return self.fail("outside a geometry program"),
        };
        if active & 1 == 0 {
            return Ok(());
        }
        let stream = self.b(0, form)?;
        self.require(stream == 0, format_args!("vertex stream {stream}"))?;
        let handle = self.r(0, bits(w, 8, 8));
        self.require(
            handle == current,
            format_args!("output vertex handle {handle:#x}"),
        )?;
        let Io::Geometry(geometry) = &mut self.io else {
            unreachable!("checked above");
        };
        if emit && geometry.handle < geometry.max {
            geometry
                .emitted
                .push(Some(std::mem::take(geometry.current)));
            geometry.handle += 1;
        }
        if cut {
            geometry.emitted.push(None);
        }
        let next = geometry.handle;
        self.set(0, bits(w, 0, 8), next);
        Ok(())
    }

    /// `IPA`: dest 0-7, index register 8-15 with `IDX` bit 38, multiplier
    /// 20-27, attribute 28-37, sample-mode register 39-46, predicate
    /// 47-50, `SAT` bit 51, sample mode 52-53, mode 54-55 (PASS, MULTIPLY,
    /// CONSTANT, SC) [public: Ryujinx `Decoders/InstDecoders.cs` `InstIpa`,
    /// eden `translate/impl/load_store_attribute.cpp` `IPA`].
    ///
    /// PASS yields the plane value at the pixel center, MULTIPLY that value
    /// times the multiplier register (the compiler passes w, from
    /// `MUFU.RCP` of `IPA.PASS a[0x7c]`, for perspective inputs). Neither
    /// public source defines CONSTANT or SC; they, indexed reads,
    /// non-default sample modes and the predicate operand are not modeled.
    fn ipa(&mut self, active: u8) -> Result<(), Unmodeled> {
        let w = self.word;
        let Io::Fragment { planes, x, y, .. } = self.io else {
            return self.fail("outside a fragment program");
        };
        let mode = bits(w, 54, 2);
        self.require(
            mode <= 1,
            format_args!("mode {}", ["", "", "CONSTANT", "SC"][mode as usize]),
        )?;
        self.require(!bit(w, 38), "IDX")?;
        self.require(bits(w, 52, 2) == 0, "sample mode")?;
        self.require(bits(w, 47, 4) == PT, "predicate operand")?;
        self.require(bits(w, 39, 8) == RZ, "sample-offset register")?;
        let attribute = bits(w, 28, 10);
        self.require(attribute.is_multiple_of(4), "unaligned attribute")?;
        self.require(attribute != 0x3fc, "front-facing attribute")?;
        let plane = planes[(attribute / 4) as usize];
        let (rd, rb) = (bits(w, 0, 8), bits(w, 20, 8));
        let sat = bit(w, 51);
        self.each(active, |m, lane| {
            let px = (x + (lane as u32 & 1)) as f32 + 0.5;
            let py = (y + (lane as u32 >> 1)) as f32 + 0.5;
            let mut value = plane.base + plane.dx * px + plane.dy * py;
            if mode == 1 {
                value *= m.rf(lane, rb);
            }
            if sat {
                value = alu::saturate(value);
            }
            m.setf(lane, rd, value);
            Ok(())
        })
    }
}

/// Byte width of an `LDG`/`STG`/`LD`/`ST` size code (U8, S8, U16, S16,
/// B32, B64, B128; 7 is not a valid size for all of them).
fn global_width(size: u32) -> Option<usize> {
    [1, 1, 2, 2, 4, 8, 16].get(size as usize).copied()
}

/// The sign-extended 20-bit immediate of `I` forms: bits 20-38 and the
/// sign at bit 56.
fn imm20(word: u64) -> i32 {
    sext(bits(word, 20, 19) | (bits(word, 56, 1) << 19), 20)
}

/// Where operand B (and C) of an instruction come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Form {
    /// B register 20-27, C register 39-46.
    R,
    /// B the sign-extended 20-bit integer immediate.
    I,
    /// B the 20-bit float immediate: the top 20 bits of an f32 (bits 20-38
    /// are bits 12-30, bit 56 the sign).
    F,
    /// B `c[34-38][20-33 * 4]`.
    C,
    /// B register 39-46, C `c[34-38][20-33 * 4]`.
    Rc,
    /// B the 32-bit immediate at bits 20-51.
    Imm32,
}
