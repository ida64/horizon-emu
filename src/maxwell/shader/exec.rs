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
//! the same for `PCNT`. That is the warp-level reconvergence stack seen
//! from one lane. These three, like `SSY`/`PBK`/`PCNT`, may be
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
//! for geometry shaders]. As the NVN compiler uses them (the language
//! scene of starter):
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
//! `SR_INVOCATION_INFO` 0), lets entry `i` hold input handle `i`, and
//! numbers output handles by the vertices emitted so far. A program that
//! uses a handle it was not given (another ISBE entry, a made-up vertex)
//! is refused, so the choice of values cannot show. An emitted vertex
//! starts the next one from zeros: after `OUT.EMIT` the outputs are
//! undefined (the compiler stores every output again).
//!
//! # Compute programs
//!
//! A compute thread block (CTA) runs as warps of [`WARP`] lanes, each
//! warp in the same lockstep as a fragment quad: thread `t` (linear index
//! `x + y * Dx + z * Dx * Dy` of its thread ID in a block of
//! `Dx * Dy * Dz` threads) is lane `t % 32` of warp `t / 32`, and a last warp the
//! block does not fill has absent lanes [public: NVIDIA CUDA C
//! Programming Guide, "SIMT Architecture": warps hold threads of
//! consecutive, increasing thread IDs, the first warp thread 0; "Thread
//! Hierarchy" for the linear thread ID]. A warp runs until every lane
//! has exited or waits at `BAR.SYNC`; once every warp of the block has,
//! the waiting lanes go on. A barrier that some threads of the block can
//! no longer reach (they exited), or lanes waiting at two different
//! barriers, are refused: the CUDA programming guide leaves such code's
//! behaviour undefined ("`__syncthreads()` is allowed in conditional code
//! but only if the conditional evaluates identically across the entire
//! thread block").
//!
//! Shared memory (`LDS`/`STS`) is one byte array per block, as large as
//! the launch declares; its contents at the start of a block are
//! undefined on hardware, so a read of a byte no thread of the block has
//! written is refused. Accesses outside it are refused (they trap on
//! hardware). The blocks of a grid run one after another in raster order
//! (x fastest) \[inference: hardware runs them concurrently in an order
//! it does not define, so one sequential order is one of the orders a
//! correct kernel must allow\]. Generic `LD`/`ST`, which reach shared or
//! local memory inside the windows `SET_SHADER_SHARED_MEMORY_WINDOW` and
//! `SET_SHADER_LOCAL_MEMORY_WINDOW` set, are refused in compute programs.
//!
//! # Instructions executed
//!
//! Float: `FADD`, `FADD32I`, `FMUL`, `FMUL32I`, `FFMA`, `FFMA32I`, `FMNMX`,
//! `FSET`, `FSETP`, `MUFU`, `RRO`, `F2I`, `I2F`, `F2F` (32-bit float forms).
//! Integer: `IADD`, `IADD32I`, `IADD3`, `ISCADD`, `ISCADD32I`, `IMNMX`,
//! `ISETP`, `ISET`, `ICMP`, `LOP`, `LOP32I`, `LOP3`, `SHL`, `SHR`, `BFE`,
//! `BFI`, `XMAD`, `SEL`, `MOV`, `MOV32I`, `VMAD`, `VSETP`.
//! Predicates: `PSETP`, `P2R`, `R2P`. System: `S2R SR_Y_DIRECTION`
//! (graphics), `S2R SR_INVOCATION_INFO` (geometry), `S2R SR_LANEID`,
//! `SR_TID`, `SR_TID.X/Y/Z`, `SR_CTAID.X/Y/Z` (compute). Memory: `LDC`,
//! `LDG`, `STG`, `LD`, `ST` (global through [`super::Env::memory`];
//! `LD`/`ST` outside compute programs), `LDS`, `STS` (compute).
//! Attributes: `ALD`, `AST` (vertex, geometry), `ISBERD`, `OUT`
//! (geometry), `IPA` (fragment). Textures: `TEX`, `TEX.B`, `TEXS`, `TLD`,
//! `TLD.B`, `TLDS`, `TLD4`, `TLD4.B`, `TLD4S`, `TXQ`, `TXQ.B`, `TMML`,
//! `TMML.B`, `TXD`, `TXD.B` (in compute programs with explicit levels of
//! detail only: there are no quad derivatives). Control: `BRA`, `BRX`,
//! `SSY`, `SYNC`, `PBK`, `BRK`, `PCNT`, `CONT`, `EXIT`, `KIL`, `NOP`,
//! `DEPBAR`, `BAR`, `MEMBAR`.
//!
//! Every other decodable instruction, and every modifier of the above that
//! is not modeled (each is named at its check), returns [`Unmodeled`] when
//! executed, naming the mnemonic, what is unmodeled, the word and the pc.
//!
//! `DEPBAR` (wait for scoreboard) and `MEMBAR` (memory ordering) have no
//! observable effect here: an invocation's instructions run in order,
//! each one's results visible to the next, and memory accesses happen in
//! program order. `BAR` has none outside compute programs, where one
//! invocation runs at a time; in compute programs `BAR.SYNC` is the
//! block-wide barrier described above. The `NOP` field `TRIG` and
//! the cache-policy fields of `LDG`/`STG`/`LD`/`ST` are hints that do not
//! change values in a model without caches.

mod alu;
mod double;
mod integer;
mod tex;
mod texture_vector;
mod video;

use std::cell::RefCell;

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

/// A set of lanes of a [`Machine`], bit `i` lane `i`.
pub(super) type Mask = u32;

/// Threads per warp [public: NVIDIA CUDA C Programming Guide, "SIMT
/// Architecture"; `SR_LANEID` counts 0-31].
pub const WARP: usize = 32;

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
/// imm24`; a target on a scheduling word continues at the next
/// instruction.
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
    /// Waiting at `BAR.SYNC` (compute).
    Barrier,
    /// A lane of a compute warp the thread block does not fill.
    Absent,
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
        /// Bit `a`: generic attribute `a` (words at 0x80 + 0x10 * a) has
        /// no value this model establishes; `ALD` of it refuses.
        inactive: u32,
        output: &'a mut Attributes,
    },
    Geometry(Geometry<'a>),
    Compute(Compute<'a>),
    Fragment {
        planes: &'a [Plane; 256],
        x: u32,
        y: u32,
        covered: [bool; 4],
        centroids: [[f32; 2]; 4],
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

/// The thread IDs and shared memory of a compute warp.
struct Compute<'a> {
    /// Linear thread index of lane 0 in the block.
    first: u32,
    /// Threads of the block per dimension.
    block: [u32; 3],
    /// The block's coordinates in the grid.
    cta: [u32; 3],
    shared: &'a RefCell<Shared>,
    /// Barriers the launch declares.
    barriers: u32,
}

/// The shared memory of a thread block, and which of its bytes threads
/// of the block have written.
struct Shared {
    bytes: Vec<u8>,
    written: Vec<bool>,
}

impl Shared {
    fn new(size: usize) -> Shared {
        Shared {
            bytes: vec![0; size],
            written: vec![false; size],
        }
    }

    /// A new block starts: nothing is written.
    fn reset(&mut self) {
        self.written.fill(false);
    }

    fn load(&self, address: usize, out: &mut [u8]) -> Result<(), String> {
        let range = self.range(address, out.len())?;
        if let Some(at) = self.written[range.clone()].iter().position(|w| !w) {
            return Err(format!(
                "reads shared byte {:#x}, which no thread of the block wrote",
                address + at
            ));
        }
        out.copy_from_slice(&self.bytes[range]);
        Ok(())
    }

    fn store(&mut self, address: usize, data: &[u8]) -> Result<(), String> {
        let range = self.range(address, data.len())?;
        self.written[range.clone()].fill(true);
        self.bytes[range].copy_from_slice(data);
        Ok(())
    }

    fn range(&self, address: usize, length: usize) -> Result<std::ops::Range<usize>, String> {
        match address.checked_add(length) {
            Some(end) if end <= self.bytes.len() => Ok(address..end),
            _ => Err(format!(
                "address {address:#x} outside the block's {:#x} bytes of shared memory",
                self.bytes.len()
            )),
        }
    }
}

/// A thread's local memory, grown as the thread stores to it.
#[derive(Default)]
struct Local {
    bytes: Vec<u8>,
}

/// The most local memory a program can declare: its size fields have 24
/// bits [public: NVIDIA open-gpu-doc Shader-Program-Header,
/// `ShaderLocalMemoryLowSize`, `ShaderLocalMemoryHighSize`].
const LOCAL_LIMIT: usize = 1 << 24;

struct Machine<'a, const L: usize> {
    code: &'a [Option<Inst>],
    env: &'a Env<'a>,
    lanes: [Lane; L],
    io: Io<'a>,
    /// The instruction being executed.
    op: Op,
    word: u64,
    pc: u32,
    /// Instructions executed so far.
    steps: u64,
    /// Each lane's local memory.
    local: [Local; L],
}

pub(super) fn run_vertex(
    program: &Program,
    env: &Env<'_>,
    input: &Attributes,
    inactive: u32,
    output: &mut Attributes,
) -> Result<(), Unmodeled> {
    if program.header.shader_type() != ShaderType::Vertex {
        return Err(Unmodeled(format!(
            "shader: run_vertex on a {:?} program",
            program.header.shader_type()
        )));
    }
    let mut machine = Machine::<1> {
        code: &program.code,
        env,
        lanes: [Lane::new()],
        io: Io::Vertex {
            input,
            inactive,
            output,
        },
        op: Op::Invalid,
        word: 0,
        pc: 0,
        steps: 0,
        local: Default::default(),
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
        code: &program.code,
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
        steps: 0,
        local: Default::default(),
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
        code: &program.code,
        env,
        lanes,
        io: Io::Fragment {
            planes: quad.planes,
            x: quad.x,
            y: quad.y,
            covered: quad.covered,
            centroids: quad.centroids,
        },
        op: Op::Invalid,
        word: 0,
        pc: 0,
        steps: 0,
        local: Default::default(),
    };
    if !quad.front_facing {
        // Nothing in the modeled instruction set reads the facing of the
        // primitive (IPA of a[0x3fc] is refused), so it has no effect.
    }
    machine.run()?;
    // Outputs: each render target with any enabled component takes four
    // consecutive registers from R0 (component c at base + c), in target
    // order; the depth follows one register after the colors, after the
    // (unmodeled) sample-mask register.
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

/// Runs every thread block of `launch` (see the module documentation).
pub(super) fn run_grid(
    code: &[Option<Inst>],
    env: &Env<'_>,
    launch: &super::Launch,
) -> Result<(), Unmodeled> {
    let [bx, by, bz] = launch.block;
    let threads = bx * by * bz;
    if threads == 0 || launch.grid.contains(&0) {
        return Ok(());
    }
    let shared = RefCell::new(Shared::new(launch.shared as usize));
    let warps = (threads as usize).div_ceil(WARP);
    let mut machines: Vec<Box<Machine<'_, WARP>>> = (0..warps)
        .map(|warp| {
            Box::new(Machine {
                code,
                env,
                lanes: std::array::from_fn(|_| Lane::new()),
                io: Io::Compute(Compute {
                    first: (warp * WARP) as u32,
                    block: launch.block,
                    cta: [0; 3],
                    shared: &shared,
                    barriers: launch.barriers,
                }),
                op: Op::Invalid,
                word: 0,
                pc: 0,
                steps: 0,
                local: std::array::from_fn(|_| Local::default()),
            })
        })
        .collect();
    for z in 0..launch.grid[2] {
        for y in 0..launch.grid[1] {
            for x in 0..launch.grid[0] {
                shared.borrow_mut().reset();
                for (warp, machine) in machines.iter_mut().enumerate() {
                    for (index, lane) in machine.lanes.iter_mut().enumerate() {
                        *lane = Lane::new();
                        if warp * WARP + index >= threads as usize {
                            lane.state = State::Absent;
                        }
                    }
                    machine.steps = 0;
                    for local in &mut machine.local {
                        local.bytes.clear();
                    }
                    if let Io::Compute(compute) = &mut machine.io {
                        compute.cta = [x, y, z];
                    }
                }
                run_block(&mut machines).map_err(|Unmodeled(what)| {
                    Unmodeled(format!("{what} (thread block {x}, {y}, {z})"))
                })?;
            }
        }
    }
    Ok(())
}

/// Runs the warps of one thread block to completion, releasing them from
/// each barrier once all that still run have reached it. Threads that
/// exited do not hold a barrier back \[inference: a barrier waiting for
/// them would never release, and shipped compute programs let all but
/// one thread of a block exit before a `BAR.SYNC` the last one reaches\].
fn run_block(machines: &mut [Box<Machine<'_, WARP>>]) -> Result<(), Unmodeled> {
    loop {
        for machine in machines.iter_mut() {
            machine.run()?;
        }
        // Every lane has exited, is absent, or waits at a barrier.
        let mut waiting = None;
        for lane in machines.iter().flat_map(|m| m.lanes.iter()) {
            match lane.state {
                State::Barrier => match waiting {
                    None => waiting = Some(lane.pc),
                    Some(pc) if pc != lane.pc => {
                        return Err(Unmodeled(format!(
                            "shader: threads of a block wait at two barriers (continuing at \
                             {pc:#x} and {:#x})",
                            lane.pc
                        )));
                    }
                    Some(_) => {}
                },
                State::Exited | State::Absent => {}
                State::Running | State::Killed => {
                    return Err(Unmodeled(format!(
                        "shader: compute lane stopped {:?}",
                        lane.state
                    )));
                }
            }
        }
        if waiting.is_none() {
            return Ok(());
        }
        for lane in machines.iter_mut().flat_map(|m| m.lanes.iter_mut()) {
            if lane.state == State::Barrier {
                lane.state = State::Running;
            }
        }
    }
}

impl<'a, const L: usize> Machine<'a, L> {
    /// Runs until no lane is running: all have exited or been killed, or
    /// (compute) wait at a barrier.
    fn run(&mut self) -> Result<(), Unmodeled> {
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
            let mut mask: Mask = 0;
            for (index, lane) in self.lanes.iter().enumerate() {
                if lane.state == State::Running && lane.pc == pc {
                    mask |= 1 << index;
                }
            }
            let inst = self
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
            self.steps += 1;
            if self.steps > STEP_LIMIT {
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

    fn is_compute(&self) -> bool {
        matches!(self.io, Io::Compute(_))
    }

    // Register file access.

    #[inline]
    fn r(&self, lane: usize, index: u32) -> u32 {
        // Vector operands may run past R254; RZ and beyond read 0.
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

    /// Condition code test of field `Ccc`.
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

    fn step(&mut self, mask: Mask) -> Result<(), Unmodeled> {
        let next = next_pc(self.pc);
        // SSY, PBK and PCNT have no guard predicate: their encoding is
        // the target (bits 20-43) and CA (bit 5) only, bits 16-19 are not
        // a field. Compilers leave them as they please (0, read as P0, in
        // a scene of starter's).
        let unpredicated = matches!(self.op, Op::Ssy | Op::Pbk | Op::Pcnt);
        let mut active: Mask = 0;
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
        if self.execute_double(active)? {
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
        active: Mask,
        mut f: impl FnMut(&mut Self, usize) -> Result<(), Unmodeled>,
    ) -> Result<(), Unmodeled> {
        for lane in 0..L {
            if active & (1 << lane) != 0 {
                f(self, lane)?;
            }
        }
        Ok(())
    }

    fn execute_other(&mut self, active: Mask) -> Result<(), Unmodeled> {
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
            Op::Nop | Op::Depbar | Op::Membar => Ok(()),
            Op::Bar if self.is_compute() => self.barrier(active),
            Op::Bar => Ok(()),
            Op::S2r => {
                let sr = bits(w, 20, 8);
                let rd = bits(w, 0, 8);
                self.each(active, |m, lane| {
                    let value = m.special_register(lane, sr)?;
                    m.set(lane, rd, value);
                    Ok(())
                })
            }
            Op::Vote if !self.is_compute() => self.vote(active),
            Op::Lds => self.shared_access(active, false),
            Op::Sts => self.shared_access(active, true),
            Op::Ldl => self.local_access(active, false),
            Op::Stl => self.local_access(active, true),
            Op::Isberd => self.isberd(active),
            Op::OutR | Op::OutI | Op::OutC => self.out(active),
            Op::Ldc => self.ldc(active),
            Op::Ld | Op::St if self.is_compute() => self.fail(
                "generic address in a compute program (the shared and local memory windows \
                 are not modeled)",
            ),
            Op::Ldg | Op::Ld => self.load_global(active),
            Op::Stg | Op::St => self.store_global(active),
            Op::Red | Op::Atom => self.atomic(active),
            Op::Ald => self.ald(active),
            Op::Ast => self.ast(active),
            Op::Ipa => self.ipa(active),
            Op::Shfl => self.shfl(active),
            Op::Fswzadd => self.fswzadd(active),
            _ => self.fail("not modeled"),
        }
    }

    /// `VOTE` outside a compute program: the register destination (0-7)
    /// takes the ballot, a bit per thread of the warp whose source
    /// predicate (39-41, negated by 42) holds \[observed: a pixel
    /// program of an application takes the highest set bit and the
    /// count of the ballot's bits (`FLO`, `POPC`), and the thread whose
    /// SR_LANEID equals the former adds the latter to a counter in
    /// global memory with `RED`: one atomic add for the warp instead of
    /// one per thread\]; the destination predicate (45-47) takes the
    /// vote, ALL, ANY or EQ (48-49) \[recalled\]. How GM20B gathers
    /// fragments or vertices into warps is not established, and what a
    /// program computes cannot depend on it: here every thread is a warp
    /// of its own \[assumption\], the ballot its predicate and its lane
    /// 0, which gives such a reduction the sum any other grouping gives.
    fn vote(&mut self, active: Mask) -> Result<(), Unmodeled> {
        let w = self.word;
        let mode = bits(w, 48, 2);
        self.require(mode < 3, "vote mode")?;
        self.each(active, |m, lane| {
            let holds = m.src_pred(lane, 39);
            m.set(lane, bits(w, 0, 8), u32::from(holds));
            m.set_pred(lane, bits(w, 45, 3), mode == 2 || holds);
            Ok(())
        })
    }

    /// `S2R`: dest 0-7, register 20-27: SR_LANEID (0: the thread's lane
    /// in its warp; outside a compute program every thread is a warp of
    /// its own, see [`Machine::vote`]), SR_Y_DIRECTION
    /// (18), SR_THREAD_KILL (19: all ones for a fragment outside the
    /// primitive, a helper of its quad, else 0 \[recalled\]),
    /// SR_INVOCATION_INFO (29, see the module documentation),
    /// SR_TID (32: x in bits 0-15, y from bit 16, z from bit 26),
    /// SR_TID.X/Y/Z (33-35), SR_CTAID.X/Y/Z (37-39).
    fn special_register(&self, lane: usize, sr: u32) -> Result<u32, Unmodeled> {
        Ok(match (&self.io, sr) {
            (Io::Compute(compute), 0 | 32..=35 | 37..=39) => {
                let [x, y, _] = compute.block;
                let thread = compute.first + lane as u32;
                let tid = [thread % x, thread / x % y, thread / (x * y)];
                match sr {
                    0 => lane as u32,
                    32 => tid[0] | tid[1] << 16 | tid[2] << 26,
                    33..=35 => tid[(sr - 33) as usize],
                    _ => compute.cta[(sr - 37) as usize],
                }
            }
            (Io::Compute(_), _) => return self.fail(&format!("special register {sr}")),
            (_, 0) => 0,
            (_, 18) => self.env.y_direction.to_bits(),
            (Io::Fragment { covered, .. }, 19) => {
                if covered[lane] {
                    0
                } else {
                    u32::MAX
                }
            }
            (Io::Geometry(geometry), 29) => (geometry.inputs.len() as u32) << 16,
            _ => return self.fail(&format!("special register {sr}")),
        })
    }

    /// `BAR` in a compute program: mode 32-34, reduction 35-36, predicate
    /// operand 39-41 negated by 42, `B` (thread count) immediate flag 43,
    /// `A` (barrier) immediate flag 44, barrier 8-15 or register A,
    /// thread count 20-31 or register B. Only `BAR.SYNC 0` with the
    /// default count (0: every thread of the block), which is what
    /// `__syncthreads()`/GLSL `barrier()` compile to, is modeled.
    fn barrier(&mut self, active: Mask) -> Result<(), Unmodeled> {
        let w = self.word;
        let mode = bits(w, 32, 3);
        self.require(mode == 0, format_args!("mode {mode} (only SYNC)"))?;
        self.require(bits(w, 35, 2) == 0, "reduction")?;
        self.require(bits(w, 39, 3) == PT && !bit(w, 42), "predicate operand")?;
        self.require(
            bit(w, 43) && bit(w, 44),
            "barrier or thread count in a register",
        )?;
        self.require(bits(w, 8, 8) == 0, "barrier other than 0")?;
        self.require(bits(w, 20, 12) == 0, "thread count")?;
        let Io::Compute(compute) = &self.io else {
            return self.fail("outside a compute program");
        };
        self.require(
            compute.barriers > 0,
            "barrier 0, which the launch does not declare (QMD BARRIER_COUNT 0)",
        )?;
        self.each(active, |m, lane| {
            m.lanes[lane].state = State::Barrier;
            Ok(())
        })
    }

    /// `LDS`/`STS`: data register 0-7, address register 8-15, offset 20-43
    /// (signed, unsigned when the address register is RZ), size 48-50
    /// (as `LDG`), `LDS.U` bit 44. `U` is taken to load the same bytes
    /// \[inference: shipped compute programs set it (0xef4d10000287ff0c)
    /// on plain loads of shared memory\].
    fn shared_access(&mut self, active: Mask, store: bool) -> Result<(), Unmodeled> {
        let w = self.word;
        let Io::Compute(compute) = &self.io else {
            return self.fail("outside a compute program");
        };
        let shared = compute.shared;
        let size = bits(w, 48, 3);
        let width = global_width(size).ok_or_else(|| self.unmodeled(&format!("size {size}")))?;
        let count = width.div_ceil(4) as u32;
        let (rd, ra) = (bits(w, 0, 8), bits(w, 8, 8));
        self.require(
            rd == RZ || rd.is_multiple_of(count),
            "vector register not aligned",
        )?;
        let immediate = bits(w, 20, 24);
        let offset = if ra == RZ {
            immediate
        } else {
            sext(immediate, 24) as u32
        };
        self.each(active, |m, lane| {
            let address = m.r(lane, ra).wrapping_add(offset);
            m.require(
                address.is_multiple_of(width as u32),
                format_args!("unaligned address {address:#x}"),
            )?;
            let mut bytes = [0u8; 16];
            if store {
                for i in 0..count {
                    let at = i as usize * 4;
                    bytes[at..at + 4].copy_from_slice(&m.r(lane, rd + i).to_le_bytes());
                }
                let stored = shared.borrow_mut().store(address as usize, &bytes[..width]);
                return stored.map_err(|what| m.unmodeled(&what));
            }
            let loaded = shared.borrow().load(address as usize, &mut bytes[..width]);
            loaded.map_err(|what| m.unmodeled(&what))?;
            m.write_loaded(lane, rd, size, &bytes);
            Ok(())
        })
    }

    /// `LDL`/`STL`: the fields of `LDS`/`STS`, on the thread's local
    /// memory, which is private to the thread. A byte the thread has not
    /// stored has no defined value and reads 0 here \[assumption: programs
    /// do not depend on such bytes; compilers load them as the padding of
    /// wider loads, as the fourth word of a three-word vector stored in
    /// a 16-byte slot\].
    fn local_access(&mut self, active: Mask, store: bool) -> Result<(), Unmodeled> {
        let w = self.word;
        let size = bits(w, 48, 3);
        let width = global_width(size).ok_or_else(|| self.unmodeled(&format!("size {size}")))?;
        let count = width.div_ceil(4) as u32;
        let (rd, ra) = (bits(w, 0, 8), bits(w, 8, 8));
        self.require(
            rd == RZ || rd.is_multiple_of(count),
            "vector register not aligned",
        )?;
        let immediate = bits(w, 20, 24);
        let offset = if ra == RZ {
            immediate
        } else {
            sext(immediate, 24) as u32
        };
        self.each(active, |m, lane| {
            let address = m.r(lane, ra).wrapping_add(offset) as usize;
            m.require(
                address.is_multiple_of(width),
                format_args!("unaligned address {address:#x}"),
            )?;
            let end = address + width;
            m.require(
                end <= LOCAL_LIMIT,
                format_args!("address {address:#x} outside local memory"),
            )?;
            let mut bytes = [0u8; 16];
            if store {
                for i in 0..count {
                    let at = i as usize * 4;
                    bytes[at..at + 4].copy_from_slice(&m.r(lane, rd + i).to_le_bytes());
                }
                let local = &mut m.local[lane].bytes;
                if local.len() < end {
                    local.resize(end, 0);
                }
                local[address..end].copy_from_slice(&bytes[..width]);
                return Ok(());
            }
            let local = &m.local[lane].bytes;
            let stored = local.len().min(end).saturating_sub(address);
            bytes[..stored].copy_from_slice(&local[address..address + stored]);
            m.write_loaded(lane, rd, size, &bytes);
            Ok(())
        })
    }

    /// Writes what a load of size code `size` read into `bytes` to the
    /// registers from `rd`: 8- and 16-bit sizes zero- or sign-extended.
    fn write_loaded(&mut self, lane: usize, rd: u32, size: u32, bytes: &[u8; 16]) {
        if rd == RZ {
            return;
        }
        let word = |i: usize| u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().expect("4"));
        match size {
            0 => self.set(lane, rd, u32::from(bytes[0])),
            1 => self.set(lane, rd, bytes[0] as i8 as i32 as u32),
            2 => self.set(
                lane,
                rd,
                u32::from(u16::from_le_bytes([bytes[0], bytes[1]])),
            ),
            3 => self.set(
                lane,
                rd,
                i16::from_le_bytes([bytes[0], bytes[1]]) as i32 as u32,
            ),
            _ => {
                let count = global_width(size).unwrap_or(4).div_ceil(4);
                for i in 0..count {
                    self.set(lane, rd + i as u32, word(i));
                }
            }
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
    /// slot 36-40, address mode 44-45, size 48-50.
    fn ldc(&mut self, active: Mask) -> Result<(), Unmodeled> {
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
    /// 58-60).
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
            // LD/ST through the generic address space reach global
            // memory; a non-PT predicate operand is not modeled.
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

    /// `LDG`/`LD`. Size code 7 (an unsigned form of B128) loads 128 bits
    /// like B128 \[inference: what the `U` adds is not documented; at a
    /// 16-byte aligned address, which is required here, a 128-bit load
    /// reads the same bytes whatever it means\].
    fn load_global(&mut self, active: Mask) -> Result<(), Unmodeled> {
        self.check_global()?;
        let rd = bits(self.word, 0, 8);
        self.each(active, |m, lane| {
            let (address, size) = m.global_access(lane)?;
            let size = if size == 7 { 6 } else { size };
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
            m.write_loaded(lane, rd, size, &bytes);
            Ok(())
        })
    }

    fn store_global(&mut self, active: Mask) -> Result<(), Unmodeled> {
        self.check_global()?;
        let rd = bits(self.word, 0, 8);
        let covered = match self.io {
            Io::Fragment { covered, .. } => Some(covered),
            Io::Vertex { .. } | Io::Geometry(_) | Io::Compute(_) => None,
        };
        self.each(active, |m, lane| {
            // Helper lanes (outside the primitive) do not write memory.
            if covered.is_some_and(|covered| !covered[lane]) {
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

    /// `RED` and `ATOM`: an atomic read-modify-write of a 32-bit word of
    /// global memory, `ATOM` returning what the word held. `RED`: source
    /// 0-7, address register 8-15, type 20-22, operation 23-25, offset
    /// 28-47 signed, `E` (64-bit address) bit 48. `ATOM`: dest 0-7,
    /// address register 8-15, source 20-27, offset 28-47, `E` bit 48,
    /// type 49-51, operation 52-55 \[`E` of `RED`: inference from a
    /// shipped program's 0xebf9000001070806, whose address register pair
    /// holds a 40-bit mapped address\]. Types U32
    /// (0) and S32 (1); operations ADD (0), MIN (1), MAX (2), AND (5),
    /// OR (6), XOR (7) and, for `ATOM`, EXCH (8). Lanes run one after
    /// another, and programs running on other threads are kept out by
    /// one lock, so each update is atomic.
    fn atomic(&mut self, active: Mask) -> Result<(), Unmodeled> {
        static UPDATE: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let w = self.word;
        let returns = matches!(self.op, Op::Atom);
        let (source, e, kind, operation) = if returns {
            (bits(w, 20, 8), bit(w, 48), bits(w, 49, 3), bits(w, 52, 4))
        } else {
            (bits(w, 0, 8), bit(w, 48), bits(w, 20, 3), bits(w, 23, 3))
        };
        let ra = bits(w, 8, 8);
        self.require(
            !e || ra.is_multiple_of(2) || ra == RZ,
            "64-bit address in an odd register",
        )?;
        self.require(kind < 2, format_args!("type {kind}"))?;
        self.require(
            matches!(operation, 0..=2 | 5..=7) || (returns && operation == 8),
            format_args!("operation {operation}"),
        )?;
        let offset = i64::from(sext(bits(w, 28, 20), 20));
        let covered = match self.io {
            Io::Fragment { covered, .. } => Some(covered),
            Io::Vertex { .. } | Io::Geometry(_) | Io::Compute(_) => None,
        };
        self.each(active, |m, lane| {
            // Helper lanes (outside the primitive) do not write memory.
            if covered.is_some_and(|covered| !covered[lane]) {
                return Ok(());
            }
            let base = if e {
                u64::from(m.r(lane, ra)) | (u64::from(m.r(lane, ra + 1)) << 32)
            } else {
                u64::from(m.r(lane, ra))
            };
            let address = base.wrapping_add(offset as u64);
            m.require(
                address.is_multiple_of(4),
                format_args!("unaligned address {address:#x}"),
            )?;
            let value = m.r(lane, source);
            let _update = UPDATE.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            let mut bytes = [0u8; 4];
            if m.env.memory.read(address, &mut bytes).is_none() {
                return m.fail(&format!("atomic update of unmapped {address:#x}"));
            }
            let old = u32::from_le_bytes(bytes);
            let signed = kind == 1;
            let new = match operation {
                0 => old.wrapping_add(value),
                1 if signed => (old as i32).min(value as i32) as u32,
                1 => old.min(value),
                2 if signed => (old as i32).max(value as i32) as u32,
                2 => old.max(value),
                5 => old & value,
                6 => old | value,
                7 => old ^ value,
                _ => value,
            };
            if m.env.memory.write(address, &new.to_le_bytes()).is_none() {
                return m.fail(&format!("atomic update of unmapped {address:#x}"));
            }
            if returns {
                m.set(lane, bits(w, 0, 8), old);
            }
            Ok(())
        })
    }

    /// `ALD`: dest 0-7, index register 8-15, offset 20-30, `P` (patch)
    /// bit 31, `O` (read outputs) bit 32, vertex register 39-46, size
    /// 47-48 (1-4 words). In a geometry
    /// program the vertex register holds an input vertex handle. An
    /// executed read of an inactive vertex attribute refuses (see
    /// `pipeline::VertexFetch`); a program that only names one in its
    /// header, on a path the draw does not take, runs.
    fn ald(&mut self, active: Mask) -> Result<(), Unmodeled> {
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
                Io::Vertex {
                    input,
                    inactive,
                    output,
                } => {
                    m.require(vertex == RZ, "vertex register outside a geometry program")?;
                    if outputs {
                        &output.0
                    } else {
                        // Generic attribute a: words 0x20 + 4a .. 0x24 + 4a.
                        let (first, last) = (offset / 4, offset / 4 + count - 1);
                        if let Some(a) = (0..32u32).find(|a| {
                            inactive & (1 << a) != 0 && first < 0x24 + 4 * a && 0x20 + 4 * a <= last
                        }) {
                            return m.fail(&format!(
                                "of vertex attribute {a}, which is inactive \
                                 (SET_VERTEX_ATTRIBUTE_A SOURCE INACTIVE): the value the \
                                 hardware supplies for it is not established"
                            ));
                        }
                        &input.0
                    }
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
                Io::Compute(_) => return m.fail("in a compute program"),
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
    /// vertex register 39-46, size 47-48. In a geometry
    /// program the vertex register holds the current output handle.
    fn ast(&mut self, active: Mask) -> Result<(), Unmodeled> {
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
            Io::Compute(_) => return self.fail("in a compute program"),
            _ => {}
        }
        let output = match &mut self.io {
            Io::Vertex { output, .. } => &mut **output,
            Io::Geometry(geometry) => &mut *geometry.current,
            Io::Fragment { .. } | Io::Compute(_) => unreachable!("refused above"),
        };
        for (i, value) in values.iter().take(count as usize).enumerate() {
            output.0[(offset / 4) as usize + i] = *value;
        }
        Ok(())
    }

    /// `ISBERD`: dest 0-7, address register 8-15, `SKEW` bit 31, `O` bit
    /// 32, mode 33-34 (default, patch, prim, attr), size 47-48. Only the
    /// default form, which
    /// reads the input vertex handle of an ISBE entry (see the module
    /// documentation), is modeled.
    fn isberd(&mut self, active: Mask) -> Result<(), Unmodeled> {
        let w = self.word;
        let Io::Geometry(geometry) = &self.io else {
            return self.fail("outside a geometry program");
        };
        let vertices = geometry.inputs.len() as u32;
        // Bits 20-30 are not a known field.
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
    /// 39, `CUT` bit 40. Emits beyond
    /// `MaxOutputVertexCount` are no-ops (Shader-Program-Header). Only
    /// stream 0 is modeled.
    fn out(&mut self, active: Mask) -> Result<(), Unmodeled> {
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
    /// CONSTANT, SC).
    ///
    /// PASS yields the plane value at the pixel center, MULTIPLY that value
    /// times the multiplier register (the compiler passes w, from
    /// `MUFU.RCP` of `IPA.PASS a[0x7c]`, for perspective inputs). SC,
    /// indexed reads, the OFFSET sample mode and the predicate operand
    /// are not modeled.
    /// CENTROID (sample mode 1) evaluates at the lane's centroid
    /// ([`Quad::centroids`](super::Quad::centroids)): the pixel center
    /// with one sample per pixel or all samples covered, else a covered
    /// sample's position (`raster::centroid`).
    fn ipa(&mut self, active: Mask) -> Result<(), Unmodeled> {
        let w = self.word;
        let Io::Fragment {
            planes,
            x,
            y,
            centroids,
            ..
        } = self.io
        else {
            return self.fail("outside a fragment program");
        };
        // CONSTANT reads an input that is constant across the primitive
        // (one the header declares so has a plane without slope, see
        // `pipeline`'s flat planes): the provoking vertex's word as it
        // is, the plane's base.
        let mode = bits(w, 54, 2);
        self.require(mode <= 2, "mode SC")?;
        self.require(!bit(w, 38), "IDX")?;
        self.require(bits(w, 52, 2) <= 1, "sample mode OFFSET")?;
        self.require(bits(w, 47, 4) == PT, "predicate operand")?;
        self.require(bits(w, 39, 8) == RZ, "sample-offset register")?;
        let attribute = bits(w, 28, 10);
        self.require(attribute.is_multiple_of(4), "unaligned attribute")?;
        self.require(attribute != 0x3fc, "front-facing attribute")?;
        let plane = planes[(attribute / 4) as usize];
        self.require(
            mode != 2 || (plane.dx == 0.0 && plane.dy == 0.0),
            "mode CONSTANT of an interpolated input",
        )?;
        let (rd, rb) = (bits(w, 0, 8), bits(w, 20, 8));
        let sat = bit(w, 51);
        let centroid = bits(w, 52, 2) == 1;
        self.each(active, |m, lane| {
            let [ox, oy] = if centroid {
                centroids[lane]
            } else {
                [0.5, 0.5]
            };
            let px = (x + (lane as u32 & 1)) as f32 + ox;
            let py = (y + (lane as u32 >> 1)) as f32 + oy;
            let mut value = if mode == 2 {
                plane.base
            } else {
                plane.base + plane.dx * px + plane.dy * py
            };
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

    /// `SHFL`: dest 0-7, A 8-15, B register 20-27 or (bit 28) immediate
    /// 20-24, C register 39-46 or (bit 29) immediate 34-46, mode 30-31
    /// (IDX, UP, DOWN, BFLY), destination predicate 48-50. Each
    /// lane reads A of the lane [`shuffle_lane`] picks, or its own where
    /// that lane is out of range, and sets the predicate to whether it was
    /// in range. Compute warps are the hardware's warps (lane `i` of a
    /// machine is lane `i` of its warp). A fragment quad is four lanes of
    /// some warp at an unknown position (lane `4q + i` for quad lane `i`):
    /// a shuffle is modeled when it stays inside the quad and picks the
    /// same quad lane wherever the quad sits ([`quad_shuffle_lane`]),
    /// which holds for the quad-wide segments compilers use for quad
    /// operations. Reading a lane that does not run the instruction is
    /// undefined on hardware; here it reads that lane's register.
    fn shfl(&mut self, active: Mask) -> Result<(), Unmodeled> {
        let w = self.word;
        self.require(
            self.is_compute() || self.is_fragment(),
            "outside a compute or fragment program",
        )?;
        let (rd, ra, pd, mode) = (bits(w, 0, 8), bits(w, 8, 8), bits(w, 48, 3), bits(w, 30, 2));
        let values: [u32; L] = std::array::from_fn(|lane| self.r(lane, ra));
        let fragment = self.is_fragment();
        self.each(active, |m, lane| {
            let b = if bit(w, 28) {
                bits(w, 20, 5)
            } else {
                m.r(lane, bits(w, 20, 8))
            };
            let c = if bit(w, 29) {
                bits(w, 34, 13)
            } else {
                m.r(lane, bits(w, 39, 8))
            };
            let picked = if fragment {
                quad_shuffle_lane(mode, lane as u32, b, c)
                    .ok_or_else(|| m.unmodeled("across the quads of a warp"))?
            } else {
                shuffle_lane(mode, lane as u32, b, c)
            };
            let value = picked.map_or(values[lane], |source| values[source as usize]);
            m.set(lane, rd, value);
            m.set_pred(lane, pd, picked.is_some());
            Ok(())
        })
    }

    /// `FSWZADD`: dest 0-7, A 8-15, B 20-27, per-lane operation 28-35 (two
    /// bits for each lane of the quad), NDV bit 38, rounding 39-40, FTZ
    /// bit 44, CC bit 47. Quad lane `i`
    /// computes `A * ka + B * kb` with (ka, kb) from its operation: (1, 1),
    /// (-1, 1), (1, -1), (0, 1), the two products added with one rounding
    /// (the products by ±1 and 0 are exact). Compilers pair it with
    /// `SHFL.BFLY`
    /// for fine derivatives (operations 0x99: the right lane minus the
    /// left one). Only round-to-nearest without NDV or CC is modeled.
    fn fswzadd(&mut self, active: Mask) -> Result<(), Unmodeled> {
        let w = self.word;
        self.require(self.is_fragment(), "outside a fragment program")?;
        self.require(!bit(w, 38), "NDV")?;
        self.require(bits(w, 39, 2) == 0, "rounding mode")?;
        self.require(!bit(w, 47), "condition code write")?;
        let ftz = bit(w, 44);
        let (rd, ra, rb, operations) =
            (bits(w, 0, 8), bits(w, 8, 8), bits(w, 20, 8), bits(w, 28, 8));
        self.each(active, |m, lane| {
            let a = alu::flush(m.rf(lane, ra), ftz);
            let b = alu::flush(m.rf(lane, rb), ftz);
            let (ka, kb) = fswzadd_factors((operations >> (2 * lane)) & 3);
            m.setf(lane, rd, alu::flush(a * ka + b * kb, ftz));
            Ok(())
        })
    }
}

/// Byte width of an `LDG`/`STG`/`LD`/`ST` size code (U8, S8, U16, S16,
/// B32, B64, B128; 7 is not a valid size for all of them).
fn global_width(size: u32) -> Option<usize> {
    [1, 1, 2, 2, 4, 8, 16].get(size as usize).copied()
}

/// The lane `SHFL` mode `mode` (0 IDX, 1 UP, 2 DOWN, 3 BFLY) makes warp
/// lane `lane` read with operands `b` (lane or offset) and `c` (clamp in
/// bits 0-4, segment mask in 8-12), or `None` where that lane is out of
/// range: the lane arithmetic of PTX `shfl.sync` [public: NVIDIA PTX ISA,
/// "shfl.sync"], `maxLane = (lane & segmask) | (clamp & ~segmask)`, with
/// UP testing the lane against it as a lower bound.
pub(super) fn shuffle_lane(mode: u32, lane: u32, b: u32, c: u32) -> Option<u32> {
    let (b, clamp, segment) = (b & 0x1f, c & 0x1f, (c >> 8) & 0x1f);
    let min = lane & segment;
    let max = min | (clamp & !segment);
    let (source, valid) = match mode {
        0 => {
            let source = min | (b & !segment);
            (source as i32, source <= max)
        }
        1 => {
            let source = lane as i32 - b as i32;
            (source, source >= max as i32)
        }
        2 => {
            let source = lane + b;
            (source as i32, source <= max)
        }
        _ => {
            let source = lane ^ b;
            (source as i32, source <= max)
        }
    };
    valid.then_some(source as u32)
}

/// [`shuffle_lane`] for quad lane `lane` (0-3) of a fragment quad at any
/// of the eight positions of a warp: the quad lane read (`Some(None)` for
/// out of range), if that is the same at every position and never a lane
/// of another quad (`None` otherwise).
pub(super) fn quad_shuffle_lane(mode: u32, lane: u32, b: u32, c: u32) -> Option<Option<u32>> {
    let mut picked = None;
    for quad in 0..(WARP as u32 / 4) {
        let base = 4 * quad;
        let local = match shuffle_lane(mode, base + lane, b, c) {
            None => None,
            Some(source) if (base..base + 4).contains(&source) => Some(source - base),
            Some(_) => return None,
        };
        match picked {
            None => picked = Some(local),
            Some(previous) if previous != local => return None,
            Some(_) => {}
        }
    }
    picked
}

/// `FSWZADD` operation `operation`'s factors of A and B.
pub(super) fn fswzadd_factors(operation: u32) -> (f32, f32) {
    [(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (0.0, 1.0)][operation as usize]
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
