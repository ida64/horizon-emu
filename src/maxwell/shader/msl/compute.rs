//! Compute programs (kernels) as Metal kernel functions, executing as the
//! interpreter's `run_grid` does (`exec.rs`, "Compute programs").
//!
//! A thread block is a threadgroup of the block's size; its threads are
//! numbered as the interpreter numbers them (`x + y * Dx + z * Dx * Dy`),
//! and the warp of the interpreter is the SIMD group of 32 consecutive
//! threads [public: Apple, Metal Shading Language Specification, "SIMD-
//! groups": threads of consecutive `thread_index_in_threadgroup`; Apple
//! GPUs execute 32 threads per SIMD group, as GM20B's warps]. Within a
//! SIMD group the threads run in the interpreter's lockstep: each step
//! the group runs its threads at the smallest program counter
//! (`simd_min`), so `SHFL` sees the threads that reach it together.
//!
//! `BAR.SYNC 0` parks a thread after the barrier. Once every thread of
//! the block is parked or has exited, all go on: the function runs its
//! threads phase by phase, each phase until every thread parks or exits,
//! then a `threadgroup_barrier` and a count of the parked threads in
//! threadgroup memory; a phase in which none parked ends the block.
//! Exited threads pass the later barriers, as the interpreter releases a
//! barrier past them.
//!
//! Shared memory (`LDS`/`STS`) is the threadgroup memory the dispatch
//! gives (its size the launch's); what no thread wrote reads as whatever
//! the GPU left there, where the interpreter refuses the read (undefined
//! on hardware either way). Blocks run concurrently in an order Metal
//! chooses, which a correct kernel must allow (the interpreter's
//! sequential order is one of them). Global memory is the renderer's one
//! buffer (`memory.rs`), with every thread live (`GM_LIVE`).

use std::collections::BTreeSet;
use std::fmt::Write as _;

use super::super::Program;
use super::super::decode::Op;
use super::super::exec::{bit, bits, sext};
use super::{
    DONE, GLOBAL_BUFFER, GLOBAL_WORD, PRELUDE, RZ, STACK_DEPTH, Stage, TextureKind, Writer,
};
use super::{flow, memory};

/// The buffer index of the answers to a kernel's texture queries.
pub const QUERY_BUFFER: usize = 1;

/// A kernel translated to a Metal kernel function (`compute_main`).
#[derive(Debug, Clone)]
pub struct TranslatedCompute {
    body: String,
    /// As [`super::Translated::textures`] and [`super::Translated::kinds`]
    /// (every sample is of a level the program names, or the base one).
    pub textures: Vec<u32>,
    pub kinds: Vec<TextureKind>,
    /// As [`super::Translated::constants`], [`super::Translated::indirect`].
    pub constants: BTreeSet<(u32, u32)>,
    pub indirect: BTreeSet<u32>,
    /// As [`super::Translated::globals`] and [`super::Translated::stores`].
    pub globals: Vec<(u32, u32)>,
    pub stores: bool,
    /// As [`super::Translated::queries`], answered at [`QUERY_BUFFER`].
    pub queries: Vec<u32>,
    /// As [`super::Translated::constant_buffer_1`].
    pub constant_buffer_1: Option<Vec<u8>>,
    /// The kernel waits at `BAR.SYNC`: its threads run in phases.
    pub barriers: bool,
}

/// Translates `program` (a kernel's code, see `shader::Kernel::program`),
/// or says why it cannot be.
pub fn translate_compute(
    program: &Program,
    constant_buffer_1: &[u8],
) -> Result<TranslatedCompute, String> {
    let mut flow = flow::analyze(program, constant_buffer_1)?;
    let mut writer = Writer {
        program,
        stage: Stage::Compute,
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
        queries: Vec::new(),
        inputs: BTreeSet::new(),
        outputs: BTreeSet::new(),
        word: 0,
        pc: 8,
    };
    // A thread that parks at a barrier goes on at the next instruction.
    let mut barriers = false;
    for &pc in &flow.pcs {
        let inst = program.code[(pc / 8) as usize].expect("instruction");
        match inst.op {
            Op::Ldc => {
                writer.indirect.insert(bits(inst.word, 36, 5));
            }
            // BAR has no condition-code test: only a guard of !PT keeps
            // it from executing.
            Op::Bar if !(bits(inst.word, 16, 3) == super::PT && bit(inst.word, 19)) => {
                barriers = true;
                flow.labels.insert(flow::next_pc(pc));
            }
            Op::Kil if !flow::inert(inst.word) => {
                writer.pc = pc;
                return writer.refuse("KIL in a compute program");
            }
            _ => {}
        }
    }
    for &pc in &flow.pcs {
        if flow.labels.contains(&pc) {
            let _ = writeln!(writer.out, "    case {pc}u:");
        }
        writer.step(pc, false)?;
    }
    let mut body = String::new();
    for index in &writer.registers {
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
    let _ = writeln!(
        body,
        "    bool killed = false; (void)killed; bool parked = false;\n    uint pc = 8u; uint depth = 0u; uint stack[{STACK_DEPTH}];"
    );
    if barriers {
        let _ = writeln!(
            body,
            "    if (tindex < 3u) {{ atomic_store_explicit(&parked_count[tindex], 0u, memory_order_relaxed); }}\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n    uint phase = 0u;"
        );
    }
    // Each step the SIMD group runs its threads at the smallest program
    // counter, as the interpreter's warp lockstep.
    let _ = writeln!(
        body,
        "    for (;;) {{\n    for (uint steps = 0u; steps < {}u; steps++) {{\n    const uint current = simd_min(parked ? {DONE} : pc);\n    if (current == {DONE}) {{ break; }}\n    if (parked || pc != current) {{ continue; }}\n    switch (pc) {{",
        super::super::exec::STEP_LIMIT
    );
    body.push_str(&writer.out);
    let _ = writeln!(body, "    default: pc = {DONE}; break;\n    }}\n    }}");
    if barriers {
        // The count of a phase is reset two phases later, after a barrier
        // every thread passed once it read that count.
        let _ = writeln!(
            body,
            "    if (parked) {{ atomic_fetch_add_explicit(&parked_count[phase % 3u], 1u, memory_order_relaxed); }}\n    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);\n    const uint waiting = atomic_load_explicit(&parked_count[phase % 3u], memory_order_relaxed);\n    if (tindex == 0u) {{ atomic_store_explicit(&parked_count[(phase + 2u) % 3u], 0u, memory_order_relaxed); }}\n    if (waiting == 0u) {{ break; }}\n    parked = false; phase++;\n    }}"
        );
    } else {
        let _ = writeln!(body, "    break;\n    }}");
    }
    Ok(TranslatedCompute {
        body,
        textures: writer.textures,
        kinds: writer.kinds,
        constants: writer.constants,
        indirect: writer.indirect,
        globals: writer.globals,
        stores: memory::stores(program, &flow.pcs)
            || flow
                .pcs
                .iter()
                .any(|&pc| program.code[(pc / 8) as usize].expect("instruction").op == Op::Atom),
        queries: writer.queries,
        constant_buffer_1: flow.jump_tables.then(|| constant_buffer_1.to_vec()),
        barriers,
    })
}

impl TranslatedCompute {
    /// The kernel function `compute_main`. Bindings: buffer
    /// [`QUERY_BUFFER`] the answers to its texture queries, `2 + slot` the
    /// constant buffers it reads and 20 their sizes (as a fragment
    /// function's), [`GLOBAL_BUFFER`] global memory and the one after it
    /// the kernel's table of it, textures and samplers by slot, and
    /// threadgroup memory 0 the block's shared memory.
    pub fn source(&self) -> String {
        let mut s = String::from(PRELUDE);
        let mut parameters = vec![
            "uint3 tid [[thread_position_in_threadgroup]]".to_string(),
            "uint3 cta [[threadgroup_position_in_grid]]".to_string(),
            "uint tindex [[thread_index_in_threadgroup]]".to_string(),
            format!("constant uint* tq [[buffer({QUERY_BUFFER})]]"),
            "threadgroup uint* sm [[threadgroup(0)]]".to_string(),
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
        for (i, kind) in self.kinds.iter().enumerate() {
            parameters.push(format!("{} t{i} [[texture({i})]]", kind.declaration()));
            parameters.push(format!("sampler s{i} [[sampler({i})]]"));
        }
        if !self.globals.is_empty() {
            parameters.push(format!(
                "device atomic_uint* gm [[buffer({GLOBAL_BUFFER})]]"
            ));
            parameters.push(format!(
                "const device uint* gt [[buffer({})]]",
                GLOBAL_BUFFER + 1
            ));
        }
        let _ = writeln!(
            s,
            "kernel void compute_main({}) {{\n    (void)tid; (void)cta; (void)tq; (void)sm;",
            parameters.join(", ")
        );
        if self.barriers {
            let _ = writeln!(s, "    threadgroup atomic_uint parked_count[3];");
        }
        let _ = writeln!(s, "#define GM_LIVE true\n#define TQ(i) tq[i]");
        if !self.globals.is_empty() {
            let _ = writeln!(s, "{GLOBAL_WORD}\n#define GT(i) gt[i]");
        }
        s.push_str(&self.body);
        let _ = writeln!(s, "#undef GM_LIVE\n#undef TQ");
        if !self.globals.is_empty() {
            let _ = writeln!(s, "#undef GM\n#undef GT");
        }
        let _ = writeln!(s, "}}");
        s
    }
}

impl Writer<'_> {
    /// `S2R` in a compute program (`exec::Machine::special_register`):
    /// SR_LANEID (0), SR_TID (32: x, y from bit 16, z from bit 26),
    /// SR_TID.X/Y/Z (33-35), SR_CTAID.X/Y/Z (37-39).
    pub(super) fn compute_register(&mut self) -> Result<(), String> {
        let value = match bits(self.word, 20, 8) {
            0 => "(tindex & 31u)",
            32 => "(tid.x | (tid.y << 16) | (tid.z << 26))",
            33 => "tid.x",
            34 => "tid.y",
            35 => "tid.z",
            37 => "cta.x",
            38 => "cta.y",
            39 => "cta.z",
            other => return self.refuse(&format!("special register {other}")),
        };
        self.set(bits(self.word, 0, 8), value);
        Ok(())
    }

    /// `SHFL` across the SIMD group (`exec::Machine::shfl`, lanes by
    /// `exec::shuffle_lane`): the lane or offset `b` and the clamp and
    /// segment mask `c` immediates or registers. A thread reads a lane
    /// that is not at the instruction with it as the GPU leaves it, where
    /// the interpreter reads that lane's register [known difference: the
    /// compilers shuffle in converged code].
    pub(super) fn simd_shuffle(&mut self) -> Result<(), String> {
        let w = self.word;
        let (rd, ra, pd, mode) = (bits(w, 0, 8), bits(w, 8, 8), bits(w, 48, 3), bits(w, 30, 2));
        let b = if bit(w, 28) {
            format!("{}u", bits(w, 20, 5))
        } else {
            self.read(bits(w, 20, 8))
        };
        let c = if bit(w, 29) {
            format!("{}u", bits(w, 34, 13))
        } else {
            self.read(bits(w, 39, 8))
        };
        let a = self.read(ra);
        let (source, valid) = match mode {
            0 => (
                "(shfl_min | (shfl_b & ~shfl_seg))",
                "(shfl_src <= shfl_max)",
            ),
            1 => (
                "uint(int(shfl_lane) - int(shfl_b))",
                "(int(shfl_lane) - int(shfl_b) >= int(shfl_max))",
            ),
            2 => ("(shfl_lane + shfl_b)", "(shfl_src <= shfl_max)"),
            _ => ("(shfl_lane ^ shfl_b)", "(shfl_src <= shfl_max)"),
        };
        self.line(&format!(
            "{{ const uint shfl_lane = tindex & 31u; const uint shfl_b = ({b}) & 31u; const uint shfl_c = {c}; const uint shfl_seg = (shfl_c >> 8) & 31u; const uint shfl_min = shfl_lane & shfl_seg; const uint shfl_max = shfl_min | ((shfl_c & 31u) & ~shfl_seg); const uint shfl_src = {source}; const bool shfl_ok = {valid}; const uint shfl_v = simd_shuffle({a}, ushort(shfl_ok ? shfl_src & 31u : shfl_lane));"
        ));
        self.set(rd, "shfl_v");
        self.set_pred(pd, "shfl_ok");
        self.line("}");
        Ok(())
    }

    /// `LDS`/`STS` (`exec::Machine::shared_access`): data register 0-7,
    /// address register 8-15, offset 20-43 (signed, unsigned with RZ),
    /// size 48-50. Loads of every size; stores of whole words (a narrower
    /// store would rewrite the bytes beside it, which other threads may
    /// store at once).
    pub(super) fn shared(&mut self, store: bool) -> Result<(), String> {
        let w = self.word;
        let size = bits(w, 48, 3);
        let width: u32 = match [1u32, 1, 2, 2, 4, 8, 16].get(size as usize) {
            Some(&width) => width,
            None => return self.refuse(&format!("size {size}")),
        };
        if store {
            self.require(width >= 4, "store narrower than a word")?;
        }
        let count = width.div_ceil(4);
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
        let a = self.read(ra);
        let values: Vec<String> = (0..count)
            .map(|i| self.read(if rd == RZ { RZ } else { rd + i }))
            .collect();
        self.line(&format!("{{ const uint sa = {a} + {offset:#x}u;"));
        if store {
            for (i, value) in values.iter().enumerate() {
                self.line(&format!("sm[(sa >> 2) + {i}u] = {value};"));
            }
        } else {
            self.line("const uint sw = sm[sa >> 2] >> ((sa & 3u) * 8u);");
            match size {
                0 => self.set(rd, "(sw & 0xffu)"),
                1 => self.set(rd, "as_type<uint>(as_type<int>(sw << 24) >> 24)"),
                2 => self.set(rd, "(sw & 0xffffu)"),
                3 => self.set(rd, "as_type<uint>(as_type<int>(sw << 16) >> 16)"),
                _ => {
                    if rd != RZ {
                        for i in 0..count {
                            self.set(rd + i, &format!("sm[(sa >> 2) + {i}u]"));
                        }
                    }
                }
            }
        }
        self.line("}");
        Ok(())
    }
}
