//! Vertex programs in MSL: the vertex function of a native Metal draw
//! (see `metal`), which fetches the draw's vertex attributes from the
//! guest's vertex streams, runs the program as the interpreter does
//! (`exec::run_vertex`) and hands the rasterizer the position and the
//! inputs the pixel program interpolates.
//!
//! The program body is the shared translation ([`super::Writer`]):
//! every instruction computes what the interpreter computes, bit for
//! bit. `ALD` reads input attribute words, `AST` writes output words;
//! vertex programs use no texture, quad or `IPA` instruction here (those
//! are refused: the draw runs its vertex stage on the CPU).
//!
//! Attribute fetch mirrors `pipeline::fetch_attribute`: the bytes at
//! (element - first element of the uploaded range) × stride + offset of
//! the attribute's stream, components unpacked least significant first,
//! converted by numerical type (UNORM and SNORM by a correctly rounded
//! division, which is the interpreter's: its f64 quotient of an integer
//! of at most 24 bits by 2^n - 1 never lies on an f32 rounding tie, the
//! quotient's bits repeating with period n), R and B swapped when asked,
//! missing components (0, 0, 0, 1).

use std::collections::BTreeSet;
use std::fmt::Write;

use super::super::Program;
use super::super::decode::Op;
use super::super::exec::{bit, bits};
use super::super::header::{Interpolation, ShaderType};
use super::{
    DONE, GLOBAL_WORD, RZ, STACK_DEPTH, Stage, VERTEX_GLOBAL_BUFFER, Writer, flow, memory,
};
use crate::maxwell::shader::Attributes;

/// Output word of clip-space x (a\[0x70\]); y, z, w follow.
const POSITION: u32 = 0x70 / 4;
/// Input words of the instance and vertex IDs.
const INSTANCE_ID: u32 = 0x2F8 / 4;
const VERTEX_ID: u32 = 0x2FC / 4;

/// A vertex program in MSL, and what a draw must supply for it.
#[derive(Debug, Clone)]
pub struct TranslatedVertex {
    /// Register declarations and the program (see [`NativeVertex::new`]).
    body: String,
    /// Input attribute words `ALD` reads.
    pub inputs: BTreeSet<u32>,
    /// Output attribute words `AST` writes or `ALD.O` reads.
    pub outputs: BTreeSet<u32>,
    /// Constant buffer words read with static offsets: (slot, byte offset).
    pub constants: BTreeSet<(u32, u32)>,
    /// Constant buffer slots read at register-dependent offsets (`LDC`).
    pub indirect: BTreeSet<u32>,
    /// As [`super::Translated::constant_buffer_1`].
    pub constant_buffer_1: Option<Vec<u8>>,
    /// As [`super::Translated::globals`].
    pub globals: Vec<(u32, u32)>,
    /// As [`super::Translated::textures`] and [`super::Translated::kinds`]
    /// (every sample is of a level the program names, or the base one).
    pub textures: Vec<u32>,
    pub kinds: Vec<super::TextureKind>,
    /// As [`super::Translated::queries`]; the answers follow the global
    /// memory table in the draw's words.
    pub queries: Vec<u32>,
}

/// Translates a vertex program, or says why it cannot be.
pub fn translate_vertex(
    program: &Program,
    constant_buffer_1: &[u8],
) -> Result<TranslatedVertex, String> {
    if program.header().shader_type() != ShaderType::Vertex {
        return Err("not a vertex program".into());
    }
    let flow = flow::analyze(program, constant_buffer_1)?;
    let mut writer = Writer {
        program,
        stage: Stage::Vertex,
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
    for &pc in &flow.pcs {
        let inst = program.code[(pc / 8) as usize].expect("instruction");
        match inst.op {
            Op::Ldc => {
                writer.indirect.insert(bits(inst.word, 36, 5));
            }
            // The interpreter refuses a vertex program that kills.
            Op::Kil if !flow::inert(inst.word) => {
                writer.pc = pc;
                return writer.refuse("KIL in a vertex program");
            }
            _ => {}
        }
    }
    for &pc in &flow.pcs {
        if !flow.linear && flow.labels.contains(&pc) {
            let _ = writeln!(writer.out, "    case {pc}u:");
        }
        writer.step(pc, flow.linear)?;
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
    let _ = writeln!(body, "    bool killed = false; (void)killed;");
    if flow.linear {
        let _ = writeln!(body, "    do {{");
        body.push_str(&writer.out);
        let _ = writeln!(body, "    }} while (false);");
    } else {
        // One thread per vertex: no lockstep to keep.
        let _ = writeln!(
            body,
            "    uint pc = 8u; uint depth = 0u; uint stack[{STACK_DEPTH}];\n    for (uint steps = 0u; steps < {}u; steps++) {{\n    if (pc == {DONE}) {{ break; }}\n    switch (pc) {{",
            super::super::exec::STEP_LIMIT
        );
        body.push_str(&writer.out);
        let _ = writeln!(body, "    default: pc = {DONE}; break;\n    }}\n    }}");
    }
    Ok(TranslatedVertex {
        body,
        inputs: writer.inputs,
        outputs: writer.outputs,
        constants: writer.constants,
        indirect: writer.indirect,
        constant_buffer_1: flow.jump_tables.then(|| constant_buffer_1.to_vec()),
        globals: writer.globals,
        textures: writer.textures,
        kinds: writer.kinds,
        queries: writer.queries,
    })
}

impl Writer<'_> {
    /// The attribute words `ALD`/`AST` address, as `exec`'s `ald`/`ast`
    /// check them.
    fn attribute_words(&self) -> Result<(u32, u32), String> {
        let w = self.word;
        self.require(!bit(w, 31), "P (patch)")?;
        self.require(bits(w, 8, 8) == RZ, "indexed address")?;
        self.require(bits(w, 39, 8) == RZ, "vertex register")?;
        let offset = bits(w, 20, 11);
        let count = bits(w, 47, 2) + 1;
        self.require(
            offset.is_multiple_of(4) && offset / 4 + count <= 256,
            "attribute address",
        )?;
        Ok((offset / 4, count))
    }

    /// `ALD` (`exec::Machine::ald`): input words, or output words with O.
    pub(super) fn ald(&mut self) -> Result<(), String> {
        let (first, count) = self.attribute_words()?;
        let rd = bits(self.word, 0, 8);
        let outputs = bit(self.word, 32);
        // All words are read before any register is written.
        let values: Vec<String> = (first..first + count)
            .map(|word| {
                if outputs {
                    self.outputs.insert(word);
                    format!("o{word}")
                } else {
                    self.inputs.insert(word);
                    format!("i{word}")
                }
            })
            .collect();
        for (i, value) in values.iter().enumerate() {
            self.set(rd + i as u32, value);
        }
        Ok(())
    }

    /// `AST` (`exec::Machine::ast`): output words from consecutive
    /// registers.
    pub(super) fn ast(&mut self) -> Result<(), String> {
        let (first, count) = self.attribute_words()?;
        let rs = bits(self.word, 0, 8);
        let values: Vec<String> = (0..count).map(|i| self.read(rs + i)).collect();
        for (i, value) in values.iter().enumerate() {
            let word = first + i as u32;
            self.outputs.insert(word);
            self.line(&format!("o{word} = {value};"));
        }
        Ok(())
    }
}

/// An attribute the vertex function fetches: its index, its raw
/// `SET_VERTEX_ATTRIBUTE_A` word and the binding slot of its stream (see
/// [`NativeVertex::new`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fetch {
    pub attribute: u32,
    pub raw: u32,
    pub slot: u32,
}

/// The vertex function of a native draw (with its varyings struct) as
/// MSL, to be compiled with the pixel program's fragment function
/// (`Translated::source`).
#[derive(Debug, Clone)]
pub struct NativeVertex {
    pub source: String,
    /// Per plane of the pixel program (`Translated::ipa_words`): the word
    /// and how it is interpolated.
    varyings: Vec<(u32, Interpolation)>,
}

/// Vertex stream binding slots: `MAX_STREAMS` buffers from
/// [`STREAM_BUFFER`].
pub const MAX_STREAMS: usize = 8;
pub const STREAM_BUFFER: usize = 21;
/// The word of the draw's words (buffer 29) that says what buffer 0
/// holds, after the words of [`MAX_STREAMS`] streams.
pub const INDEXED_WORD: usize = 2 + 3 * MAX_STREAMS;

/// The word of the draw's words where a program's global memory table
/// starts (`memory.rs`).
pub const GLOBAL_TABLE_WORD: usize = INDEXED_WORD + 1;

impl NativeVertex {
    /// The vertex function `vertex_native` of `vertex` for a draw that
    /// fetches `fetches` and whose pixel program interpolates `varyings`
    /// (per plane: word and mode), with the user clip distances in output
    /// words `clip_words`, drawing into one layer of layered targets when
    /// `layered`; or, with `kernel`, the compute function `vertex_kernel`
    /// that writes every output word of vertex `v` to `result[256 v +
    /// word]` (buffer 30), for tests.
    ///
    /// Bindings: buffer 0 the draw's elements as (fetch index, vertex ID)
    /// pairs, indexed by Metal's vertex ID, or, for a draw whose Metal
    /// vertex IDs are the guest's index buffer values (word
    /// [`INDEXED_WORD`] of buffer 29 not 0), what to add to one for the
    /// fetch index and for the vertex ID (`SET_GLOBAL_BASE_VERTEX_INDEX`,
    /// `SET_VERTEX_ID_BASE`; modulo 2^32 as `pipeline::indexed_element`);
    /// buffer 1 the viewport as ten
    /// floats (scale x, y, z, offset x, y, z, 2 / target width, 2 / target
    /// height, 1 for clipping z to -w..w rather than 0..w, and the layer as
    /// a `uint`); `2 + slot` the vertex stage's constant buffers and 20
    /// their sizes, as for the fragment function; buffer 29 the draw's
    /// words: the instance ID, the instance per-instance streams step by
    /// (with `SET_GLOBAL_BASE_INSTANCE_INDEX`), then per stream slot the
    /// first element of its uploaded range, its stride and its divisor (0:
    /// per vertex), and at [`INDEXED_WORD`] the elements' form;
    /// [`STREAM_BUFFER`] + slot the streams' bytes.
    ///
    /// The position is mapped so Metal's viewport (the whole target) puts
    /// the vertex where the draw's viewport transform does: window x =
    /// (x / w) scale + offset is Metal's x = (2 / width)(x scale + w
    /// offset) - w over w, and window y (rows down) likewise from Metal's
    /// y up. Metal's z is the clip z (or (z + w) / 2 for the -w..w
    /// range), so its clipping to 0..w is the draw's depth clip; the
    /// window depth (z / w) scale + offset, computed as
    /// `pipeline::Clip::window` does, is handed on in `depth`. A
    /// perspective input is handed on as its value times 1/w and a\[0x7c\]
    /// as 1/w, all interpolated linearly in the window, which is what
    /// `IPA` reads of them (the planes of `pipeline::primitive_planes`).
    pub fn new(
        vertex: &TranslatedVertex,
        fetches: &[Fetch],
        varyings: &[(u32, Interpolation)],
        clip_words: &[u32],
        layered: bool,
        kernel: bool,
    ) -> Result<NativeVertex, String> {
        let mut s = String::from(VERTEX_PRELUDE);
        // The varyings struct.
        let _ = writeln!(
            s,
            "struct Varyings {{\n    float4 position [[position]];\n    float inv_w [[center_no_perspective]];\n    float depth [[center_no_perspective]];"
        );
        for (k, &(word, mode)) in varyings.iter().enumerate() {
            let qualifier = match mode {
                Interpolation::Perspective | Interpolation::Linear => "center_no_perspective",
                Interpolation::Flat => "flat",
                _ => continue,
            };
            if is_generic(word) {
                let _ = writeln!(s, "    float v{k} [[{qualifier}]];");
            }
        }
        if !clip_words.is_empty() {
            let _ = writeln!(
                s,
                "    float clip [[clip_distance]] [{}];",
                clip_words.len()
            );
        }
        if layered {
            let _ = writeln!(s, "    uint layer [[render_target_array_index]];");
        }
        let _ = writeln!(s, "}};");
        // Parameters.
        let mut parameters = vec![
            "const device uint* elements [[buffer(0)]]".to_string(),
            "constant float* view [[buffer(1)]]".to_string(),
            "constant uint* draw [[buffer(29)]]".to_string(),
        ];
        // `iid`: the instance among the draw's (the test kernel runs one).
        if kernel {
            parameters.push("uint vid [[thread_position_in_grid]]".to_string());
        } else {
            parameters.push("uint vid [[vertex_id]]".to_string());
            parameters.push("uint iid [[instance_id]]".to_string());
        }
        let slots: BTreeSet<u32> = vertex
            .constants
            .iter()
            .map(|&(slot, _)| slot)
            .chain(vertex.indirect.iter().copied())
            .collect();
        for slot in &slots {
            parameters.push(format!(
                "const device uint* cb{slot} [[buffer({})]]",
                2 + slot
            ));
        }
        if !vertex.indirect.is_empty() {
            parameters.push("const device uint* cb_size [[buffer(20)]]".to_string());
        }
        let streams: BTreeSet<u32> = fetches.iter().map(|f| f.slot).collect();
        for &slot in &streams {
            if slot as usize >= MAX_STREAMS {
                return Err(format!("vertex stream slot {slot}"));
            }
            parameters.push(format!(
                "const device uchar* vs{slot} [[buffer({})]]",
                STREAM_BUFFER + slot as usize
            ));
        }
        for (i, kind) in vertex.kinds.iter().enumerate() {
            parameters.push(format!("{} t{i} [[texture({i})]]", kind.declaration()));
            parameters.push(format!("sampler s{i} [[sampler({i})]]"));
        }
        if !vertex.globals.is_empty() {
            if kernel {
                return Err("global memory in the test kernel".into());
            }
            parameters.push(format!(
                "device atomic_uint* gm [[buffer({VERTEX_GLOBAL_BUFFER})]]"
            ));
        }
        if kernel {
            parameters.push("device uint* result [[buffer(30)]]".to_string());
            let _ = writeln!(s, "kernel void vertex_kernel({}) {{", parameters.join(", "));
        } else {
            let _ = writeln!(
                s,
                "vertex Varyings vertex_native({}) {{",
                parameters.join(", ")
            );
        }
        if kernel {
            let _ = writeln!(s, "    const uint iid = 0u;");
        }
        let _ = writeln!(
            s,
            "    const bool indexed = draw[{INDEXED_WORD}u] != 0u;\n    const uint fetch = indexed ? vid + elements[0] : elements[2u * vid]; const uint vertex_id = indexed ? vid + elements[1] : elements[2u * vid + 1u]; (void)fetch; (void)vertex_id; (void)iid;"
        );
        // Inputs.
        let mut fetched = BTreeSet::new();
        for &word in &vertex.inputs {
            let value = match word {
                INSTANCE_ID => "(draw[0] + iid)".to_string(),
                VERTEX_ID => "vertex_id".to_string(),
                0x20..0xA0 => {
                    let attribute = (word - 0x20) / 4;
                    match fetches.iter().find(|f| f.attribute == attribute) {
                        Some(f) => {
                            if fetched.insert(attribute) {
                                s.push_str(&fetch_code(f)?);
                            }
                            format!("a{attribute}[{}]", (word - 0x20) % 4)
                        }
                        // Not fetched: the interpreter's inputs hold 0.
                        None => "0u".to_string(),
                    }
                }
                _ => "0u".to_string(),
            };
            let _ = writeln!(s, "    const uint i{word} = {value};");
        }
        // Outputs: what the program writes or reads, the position and what
        // is handed on, each starting at `Attributes::vertex_outputs`.
        let defaults = Attributes::vertex_outputs();
        let mut outputs: BTreeSet<u32> = vertex.outputs.clone();
        outputs.extend(POSITION..POSITION + 4);
        outputs.extend(
            varyings
                .iter()
                .map(|&(word, _)| word)
                .filter(|&word| is_generic(word)),
        );
        outputs.extend(clip_words.iter().copied());
        for &word in &outputs {
            let _ = writeln!(s, "    uint o{word} = {:#x}u;", defaults.0[word as usize]);
        }
        // Every vertex is the draw's (`memory.rs`).
        if !vertex.globals.is_empty() {
            let _ = writeln!(
                s,
                "#define GM_LIVE true\n{GLOBAL_WORD}\n#define GT(i) draw[{GLOBAL_TABLE_WORD}u + (i)]"
            );
        }
        let _ = writeln!(
            s,
            "#define TQ(i) draw[{}u + (i)]",
            GLOBAL_TABLE_WORD + 4 * vertex.globals.len()
        );
        s.push_str(&vertex.body);
        let _ = writeln!(s, "#undef TQ");
        if !vertex.globals.is_empty() {
            let _ = writeln!(s, "#undef GM_LIVE\n#undef GM\n#undef GT");
        }
        if kernel {
            for &word in &outputs {
                let _ = writeln!(s, "    result[256u * vid + {word}u] = o{word};");
            }
            let _ = writeln!(s, "}}");
        } else {
            let [x, y, z, w] = [0, 1, 2, 3].map(|i| format!("as_type<float>(o{})", POSITION + i));
            let _ = writeln!(
                s,
                "    Varyings out;\n    const float cw = {w}; const float cz = {z};\n    out.position = float4((({x} * view[0]) + cw * view[3]) * view[6] - cw, cw - (({y} * view[1]) + cw * view[4]) * view[7], view[8] != 0.0f ? (cz + cw) * 0.5f : cz, cw);\n    out.inv_w = 1.0f / cw;\n    out.depth = ((cz * out.inv_w) * view[2]) + view[5];"
            );
            if layered {
                let _ = writeln!(s, "    out.layer = as_type<uint>(view[9]);");
            }
            for (k, &(word, mode)) in varyings.iter().enumerate() {
                if !is_generic(word) {
                    continue;
                }
                match mode {
                    Interpolation::Perspective => {
                        let _ = writeln!(s, "    out.v{k} = as_type<float>(o{word}) * out.inv_w;");
                    }
                    Interpolation::Linear | Interpolation::Flat => {
                        let _ = writeln!(s, "    out.v{k} = as_type<float>(o{word});");
                    }
                    _ => {}
                }
            }
            for (i, word) in clip_words.iter().enumerate() {
                let _ = writeln!(s, "    out.clip[{i}] = as_type<float>(o{word});");
            }
            let _ = writeln!(s, "    return out;\n}}");
        }
        Ok(NativeVertex {
            source: s,
            varyings: varyings.to_vec(),
        })
    }

    /// What `IPA` of plane `k`, word `word`, reads in the fragment
    /// function: the window position for a\[0x70\]-a\[0x78\], 1/w for
    /// a\[0x7c\], a varying for an interpolated generic input, else 0 (an
    /// input nothing provides: the plane rasterization leaves 0 there).
    pub(super) fn varying(&self, word: u32, k: usize) -> String {
        match word {
            0x1C => "in.position.x".into(),
            0x1D => "in.position.y".into(),
            0x1E => "in.depth".into(),
            0x1F => "in.inv_w".into(),
            _ => match self.varyings.get(k) {
                Some(&(
                    w,
                    Interpolation::Perspective | Interpolation::Linear | Interpolation::Flat,
                )) if w == word && is_generic(word) => {
                    format!("in.v{k}")
                }
                _ => "0.0f".into(),
            },
        }
    }
}

/// A generic attribute word (a\[0x80\]-a\[0x27c\]).
fn is_generic(word: u32) -> bool {
    (0x20..0xA0).contains(&word)
}

/// `uint4 a{attribute}`: the attribute's four words as
/// `pipeline::fetch_attribute` gives them.
fn fetch_code(fetch: &Fetch) -> Result<String, String> {
    let raw = fetch.raw;
    let code = (raw >> 21) & 0x3F;
    let kind = (raw >> 27) & 7;
    let offset = (raw >> 7) & 0x3FFF;
    let widths: &[u32] = match code {
        0x01 => &[32, 32, 32, 32],
        0x02 => &[32, 32, 32],
        0x03 => &[16, 16, 16, 16],
        0x04 => &[32, 32],
        0x05 => &[16, 16, 16],
        0x0A | 0x2F => &[8, 8, 8, 8],
        0x0F => &[16, 16],
        0x12 => &[32],
        0x13 => &[8, 8, 8],
        0x18 | 0x32 => &[8, 8],
        0x1B => &[16],
        0x1D => &[8],
        0x30 => &[10, 10, 10, 2],
        _ => return Err(format!("vertex attribute component sizes {code:#x}")),
    };
    let size: u32 = widths.iter().sum::<u32>() / 8;
    let integer = matches!(kind, 3 | 4);
    let one = if integer { 1 } else { 1.0f32.to_bits() };
    let slot = fetch.slot;
    let a = fetch.attribute;
    let mut s = String::new();
    // The element and its bytes: per-instance streams step every
    // `divisor` instances.
    let _ = writeln!(
        s,
        "    uint4 a{a} = uint4(0u, 0u, 0u, {one:#x}u);\n    {{ const uint divisor = draw[4u + 3u * {slot}u]; const uint element = divisor != 0u ? (draw[1] + iid) / divisor : fetch;\n      const uint at = (element - draw[2u + 3u * {slot}u]) * draw[3u + 3u * {slot}u] + {offset}u;"
    );
    let words = size.div_ceil(4);
    for k in 0..words {
        let length = (size - 4 * k).min(4);
        let _ = writeln!(
            s,
            "      const uint w{k} = load_bytes(vs{slot}, at + {}u, {length}u);",
            4 * k
        );
    }
    let mut shift = 0;
    for (component, &width) in widths.iter().enumerate() {
        let word = shift / 32;
        let bit = shift % 32;
        shift += width;
        let mask = if width == 32 {
            "0xffffffffu".to_string()
        } else {
            format!("{:#x}u", (1u32 << width) - 1)
        };
        let value = format!("((w{word} >> {bit}u) & {mask})");
        let signed = format!("(int({value} << {}u) >> {}u)", 32 - width, 32 - width);
        let converted = match kind {
            // SNORM: c / (2^(n-1) - 1), clamped to -1.
            1 => format!(
                "as_type<uint>(max(precise::divide(float({signed}), {:.1}f), -1.0f))",
                ((1u64 << (width - 1)) - 1) as f32
            ),
            2 if width <= 24 => format!(
                "as_type<uint>(precise::divide(float({value}), {:.1}f))",
                ((1u64 << width) - 1) as f32
            ),
            3 => format!("as_type<uint>({signed})"),
            4 => value,
            5 => format!("as_type<uint>(float({value}))"),
            6 => format!("as_type<uint>(float({signed}))"),
            7 if width == 32 => value,
            7 if width == 16 => {
                format!("as_type<uint>(float(as_type<half>(ushort({value}))))")
            }
            _ => {
                return Err(format!(
                    "vertex attribute numerical type {kind} of {width} bits"
                ));
            }
        };
        let _ = writeln!(s, "      a{a}[{component}] = {converted};");
    }
    if raw & (1 << 31) != 0 {
        let _ = writeln!(s, "      a{a} = a{a}.zyxw;");
    }
    let _ = writeln!(s, "    }}");
    Ok(s)
}

/// Helpers of the vertex function (after the shared prelude).
const VERTEX_PRELUDE: &str = r#"
/// `n` (1-4) little-endian bytes at `p + at`.
static inline uint load_bytes(const device uchar* p, uint at, uint n) {
    uint v = 0u;
    for (uint i = 0u; i < n; i++) { v |= uint(p[at + i]) << (8u * i); }
    return v;
}
"#;
