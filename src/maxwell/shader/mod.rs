//! Maxwell (SM 5.3, GM20B) shader programs: decoder and interpreter.
//!
//! A program is the 0x50-byte shader program header ([`Header`]) followed
//! by code in 32-byte groups: one scheduling (control) word, then three
//! 64-bit instructions. Word `i` after the header is a scheduling word when
//! `i % 4 == 0`; it carries only latency/barrier hints for the hardware
//! scheduler, which an in-order interpreter has no use for. Program
//! counters in this module are byte offsets from the first word after the
//! header (the base [public: Ryujinx `Decoders/Decoder.cs`] uses, so that
//! branch targets are `pc + 8 + imm24` and a target on a scheduling word
//! continues at the next instruction).
//!
//! # End of program
//!
//! The NVN compiler ends every program with an unconditional branch to
//! itself after the last `EXIT` (0xE2400FFFFF87000F in slots 2 and 3 of a
//! group, 0xE2400FFFFF07000F in slot 1, whose target `pc - 8` is the
//! group's scheduling word and so continues at the branch itself). Every
//! one of the 102 programs of the corpus below ends that way. Decoding
//! stops after that branch, at an all-zero group, or fails after 64 KiB.
//!
//! # Firmware corpus
//!
//! The decoder is exercised on every NVN program of the starter applet
//! (0100000000001012): the ELF's embedded `ui2d_BuildinShader_Nvn_Binary`
//! and `font_BuildinShader_Nvn_Binary` (file offsets 0x498000-0x4c2000),
//! `common/shader/VarietyOceanShader_Nx.arc.szs`,
//! `common/agl/agl_resource_min.Nin_Nx_NVN.release.sarc.szs` and
//! `common/sead/primitive_renderer/primitive_drawer_nvn_shader.bin` from
//! its romfs. Each stage blob is a 0x30-byte header (magic 0x12345678, then
//! zeros) followed by the SPH. 102 distinct programs (40 ELF, 8 ocean,
//! 52 agl, 2 primitive renderer; qlaunch's ELF embeds the same 40 as
//! starter's). Instruction counts (ELF / ocean / agl / primitive):
//!
//! | mnemonic | count | ELF | ocean | agl | prim |
//! |---|---|---|---|---|---|
//! | FFMA | 2250 | 560 | 1002 | 660 | 28 |
//! | FADD | 1653 | 301 | 821 | 527 | 4 |
//! | FMUL | 1295 | 418 | 639 | 230 | 8 |
//! | MOV | 979 | 276 | 679 | 20 | 4 |
//! | IPA | 448 | 202 | 47 | 194 | 5 |
//! | AST | 432 | 38 | 176 | 210 | 8 |
//! | BRA | 400 | 144 | 202 | 52 | 2 |
//! | ISETP | 302 | 24 | 278 | 0 | 0 |
//! | MOV32I | 311 | 29 | 255 | 27 | 0 |
//! | FSETP | 340 | 16 | 324 | 0 | 0 |
//! | TEXS | 244 | 49 | 4 | 191 | 0 |
//! | MUFU | 241 | 52 | 146 | 42 | 1 |
//! | SYNC | 182 | 2 | 180 | 0 | 0 |
//! | LOP32I | 172 | 2 | 170 | 0 | 0 |
//! | ALD | 144 | 17 | 64 | 59 | 4 |
//! | SHR | 140 | 4 | 136 | 0 | 0 |
//! | IADD | 170 | 30 | 140 | 0 | 0 |
//! | IADD32I | 112 | 15 | 97 | 0 | 0 |
//! | FMNMX | 124 | 54 | 64 | 6 | 0 |
//! | FSET | 110 | 0 | 30 | 80 | 0 |
//! | EXIT | 109 | 45 | 10 | 52 | 2 |
//! | F2I | 97 | 21 | 76 | 0 | 0 |
//! | SSY | 92 | 2 | 90 | 0 | 0 |
//! | ISCADD | 85 | 13 | 72 | 0 | 0 |
//! | NOP | 72 | 28 | 8 | 34 | 2 |
//! | BRK | 70 | 8 | 62 | 0 | 0 |
//! | ISET | 74 | 0 | 74 | 0 | 0 |
//! | LOP | 65 | 55 | 10 | 0 | 0 |
//! | LDC | 64 | 31 | 33 | 0 | 0 |
//! | DEPBAR | 64 | 6 | 0 | 58 | 0 |
//! | I2F | 52 | 12 | 40 | 0 | 0 |
//! | SHL | 48 | 21 | 27 | 0 | 0 |
//! | BFI | 34 | 0 | 34 | 0 | 0 |
//! | FMUL32I | 32 | 8 | 0 | 24 | 0 |
//! | IMNMX | 30 | 7 | 23 | 0 | 0 |
//! | BRX | 30 | 7 | 23 | 0 | 0 |
//! | RRO | 22 | 0 | 18 | 4 | 0 |
//! | R2P | 22 | 18 | 4 | 0 | 0 |
//! | OUT | 16 | 0 | 16 | 0 | 0 |
//! | LDG | 14 | 14 | 0 | 0 | 0 |
//! | PBK | 12 | 2 | 10 | 0 | 0 |
//! | FADD32I | 10 | 6 | 4 | 0 | 0 |
//! | XMAD | 6 | 6 | 0 | 0 | 0 |
//! | VMAD | 6 | 0 | 6 | 0 | 0 |
//! | VSETP | 6 | 0 | 6 | 0 | 0 |
//! | ISBERD | 6 | 0 | 6 | 0 | 0 |
//! | ICMP | 6 | 0 | 6 | 0 | 0 |
//! | IADD3 | 6 | 0 | 6 | 0 | 0 |
//! | BFE | 4 | 3 | 1 | 0 | 0 |
//! | F2F | 4 | 0 | 0 | 4 | 0 |
//! | KIL | 3 | 3 | 0 | 0 | 0 |
//! | S2R | 2 | 0 | 2 | 0 | 0 |
//!
//! The ocean archive's geometry shaders (`OUT`, `ISBERD`, `S2R
//! SR_INVOCATION_INFO`) run as described in the interpreter's
//! documentation (`exec`); starter's language scene draws with one.
//! No program of the corpus uses `TEX`, `TLD`,
//! `TLDS`, `TLD4`, `TLD4S`, `TXQ`, `TMML`, `TXD`, `LD`, `ST`, `STG`,
//! `LOP3`, `SEL`, `PSETP`, `P2R`, `VADD`, `VSET`, `VABSDIFF`, `FSWZADD`,
//! `SHFL`, `PCNT` or `CONT`; those are implemented from the public sources
//! cited at each, except `VADD`/`VSET`/`VABSDIFF`/`FSWZADD`/`SHFL`,
//! which are outside the executed firmware subset and return [`Unmodeled`].
//!
//! # Execution
//!
//! Fragment lanes execute in minimum-PC order, preserving structured
//! divergence and exposing neighbouring register values to texture derivatives.

mod decode;
mod exec;
mod header;
mod metadata;
pub mod msl;
#[cfg(test)]
mod tests;

pub use exec::WARP;
pub use header::{Header, Interpolation, SPH_SIZE, ShaderType};

use super::sampling::TextureUnit;
use super::{GpuMemory, Unmodeled};
use decode::Op;

/// Largest program the decoder accepts, header included.
const MAX_PROGRAM: usize = 64 * 1024;

/// A decoded program.
#[derive(Debug, Clone)]
pub struct Program {
    header: Header,
    /// One entry per 64-bit word after the header; scheduling words are
    /// stored as `None`.
    code: Vec<Option<exec::Inst>>,
}

impl Program {
    /// `address`: GPU VA of the shader program header (SPH, 0x50 bytes);
    /// instructions start at +0x50. Reads 32-byte groups until the program
    /// ends (see the module documentation) and decodes every instruction
    /// word in that range.
    pub fn decode(memory: &dyn GpuMemory, address: u64) -> Result<Program, Unmodeled> {
        let mut sph = [0u8; SPH_SIZE];
        memory.read(address, &mut sph).ok_or_else(|| {
            Unmodeled(format!(
                "shader: program header at {address:#x} is not mapped"
            ))
        })?;
        let header = Header::parse(&sph)?;
        let mut builder = Builder::default();
        let mut group = [0u8; 32];
        loop {
            let offset = builder.code.len() * 8;
            let group_address = address + (SPH_SIZE + offset) as u64;
            memory.read(group_address, &mut group).ok_or_else(|| {
                Unmodeled(format!(
                    "shader: code group at {group_address:#x} is not mapped"
                ))
            })?;
            if builder.push_group(&group, SPH_SIZE)? {
                return Ok(Program {
                    header,
                    code: builder.code,
                });
            }
        }
    }

    /// Same as [`Program::decode`], from bytes starting at the SPH.
    pub fn from_bytes(bytes: &[u8]) -> Result<Program, Unmodeled> {
        let sph: &[u8; SPH_SIZE] = bytes
            .get(..SPH_SIZE)
            .and_then(|s| s.try_into().ok())
            .ok_or_else(|| Unmodeled("shader: shorter than a program header".into()))?;
        let header = Header::parse(sph)?;
        let code = Builder::from_bytes(&bytes[SPH_SIZE..], SPH_SIZE)?;
        Ok(Program { header, code })
    }

    pub fn header(&self) -> &Header {
        &self.header
    }

    /// Size in bytes from the SPH start to the end of the decoded code
    /// (the terminating self-branch included).
    pub fn len(&self) -> usize {
        SPH_SIZE + self.code.len() * 8
    }

    /// Host memory the decoded program takes.
    pub fn bytes(&self) -> usize {
        std::mem::size_of::<Self>() + std::mem::size_of_val(self.code.as_slice())
    }

    /// Always false: a program has at least its terminating branch.
    pub fn is_empty(&self) -> bool {
        self.code.is_empty()
    }

    /// Runs one vertex: `input` is the attribute space `ALD` reads,
    /// `output` the one `AST` writes (and `ALD.O` reads back). Bit `a` of
    /// `inactive` marks generic attribute `a` as having no established
    /// value: an executed `ALD` of it refuses.
    pub fn run_vertex(
        &self,
        env: &Env<'_>,
        input: &Attributes,
        inactive: u32,
        output: &mut Attributes,
    ) -> Result<(), Unmodeled> {
        exec::run_vertex(self, env, input, inactive, output)
    }

    /// Runs one geometry shader thread on an input primitive (`inputs`,
    /// the attribute spaces of its vertices as the previous stage wrote
    /// them) and appends what it outputs to `emitted`: each emitted
    /// vertex, and `None` where it ended a strip.
    pub fn run_geometry(
        &self,
        env: &Env<'_>,
        inputs: &[&Attributes],
        emitted: &mut Vec<Option<Attributes>>,
    ) -> Result<(), Unmodeled> {
        exec::run_geometry(self, env, inputs, emitted)
    }

    /// Runs one 2x2 quad of fragments and fills `quad.outputs`.
    pub fn run_fragment(&self, env: &Env<'_>, quad: &mut Quad<'_>) -> Result<(), Unmodeled> {
        exec::run_fragment(self, env, quad)
    }

    /// Mnemonics of all decoded instructions, in program order.
    pub fn mnemonics(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.code.iter().flatten().map(|inst| inst.op.mnemonic())
    }
}

/// A compute program (kernel). Unlike graphics programs it has no program
/// header: code starts at its address [public: Ryujinx
/// `Translation/Translator.cs`, which decodes compute programs from their
/// address and others from address + 0x50 after the `ShaderHeader`]. The
/// end of the code is found as for [`Program`].
#[derive(Debug, Clone)]
pub struct Kernel {
    code: Vec<Option<exec::Inst>>,
}

/// The thread blocks of one compute launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Launch {
    /// Thread blocks per dimension.
    pub grid: [u32; 3],
    /// Threads per block per dimension.
    pub block: [u32; 3],
    /// Bytes of shared memory per block.
    pub shared: u32,
    /// Hardware barriers the kernel may use (barrier `i < barriers`).
    pub barriers: u32,
}

impl Kernel {
    /// Decodes the kernel whose code starts at GPU VA `address`.
    pub fn decode(memory: &dyn GpuMemory, address: u64) -> Result<Kernel, Unmodeled> {
        let mut builder = Builder::default();
        let mut group = [0u8; 32];
        loop {
            let group_address = address + (builder.code.len() * 8) as u64;
            memory.read(group_address, &mut group).ok_or_else(|| {
                Unmodeled(format!(
                    "shader: kernel code group at {group_address:#x} is not mapped"
                ))
            })?;
            if builder.push_group(&group, 0)? {
                return Ok(Kernel { code: builder.code });
            }
        }
    }

    /// Same as [`Kernel::decode`], from bytes starting at the code.
    pub fn from_bytes(bytes: &[u8]) -> Result<Kernel, Unmodeled> {
        Ok(Kernel {
            code: Builder::from_bytes(bytes, 0)?,
        })
    }

    /// Size in bytes of the decoded code (the terminating self-branch
    /// included).
    pub fn len(&self) -> usize {
        self.code.len() * 8
    }

    /// Always false: a kernel has at least its terminating branch.
    pub fn is_empty(&self) -> bool {
        self.code.is_empty()
    }

    /// Host memory the decoded kernel takes.
    pub fn bytes(&self) -> usize {
        std::mem::size_of::<Self>() + std::mem::size_of_val(self.code.as_slice())
    }

    /// Leading bytes of constant buffer `slot` the kernel may read (see
    /// [`Program::constant_buffer_extent`]).
    pub fn constant_buffer_extent(&self, slot: usize) -> Option<usize> {
        metadata::constant_buffer_extent(&self.code, slot)
    }

    /// Leading bytes of the texture-handle buffer the kernel reads (see
    /// [`Program::texture_handle_extent`]).
    pub fn texture_handle_extent(&self) -> usize {
        metadata::texture_handle_extent(&self.code)
    }

    /// Runs every thread of `launch`; global memory, constant buffers and
    /// textures through `env`. See the interpreter's documentation
    /// (`exec`, "Compute programs") for the model of warps, barriers and
    /// shared memory.
    pub fn run(&self, env: &Env<'_>, launch: &Launch) -> Result<(), Unmodeled> {
        exec::run_grid(&self.code, env, launch)
    }

    /// Mnemonics of all decoded instructions, in program order.
    pub fn mnemonics(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.code.iter().flatten().map(|inst| inst.op.mnemonic())
    }
}

#[derive(Default)]
struct Builder {
    code: Vec<Option<exec::Inst>>,
}

impl Builder {
    /// Decodes `bytes` (code from its first group on) until the program
    /// ends; `start` is the code's offset from the program's start, which
    /// counts against [`MAX_PROGRAM`].
    fn from_bytes(bytes: &[u8], start: usize) -> Result<Vec<Option<exec::Inst>>, Unmodeled> {
        let mut builder = Builder::default();
        loop {
            let at = builder.code.len() * 8;
            let group = bytes.get(at..at + 32).ok_or_else(|| {
                Unmodeled(format!(
                    "shader: program runs past the end of its {} bytes",
                    bytes.len() + start
                ))
            })?;
            if builder.push_group(group.try_into().expect("32-byte group"), start)? {
                return Ok(builder.code);
            }
        }
    }

    /// Decodes one 32-byte group; true when the program ended in it.
    /// `start`: as for [`Builder::from_bytes`].
    fn push_group(&mut self, group: &[u8; 32], start: usize) -> Result<bool, Unmodeled> {
        if group.iter().all(|&b| b == 0) {
            if self.code.is_empty() {
                return Err(Unmodeled("shader: program has no code".into()));
            }
            return Ok(true);
        }
        if start + self.code.len() * 8 + 32 > MAX_PROGRAM {
            return Err(Unmodeled(format!(
                "shader: no end of program within {MAX_PROGRAM:#x} bytes"
            )));
        }
        for (slot, chunk) in group.as_chunks::<8>().0.iter().enumerate() {
            let word = u64::from_le_bytes(*chunk);
            if slot == 0 {
                self.code.push(None);
                continue;
            }
            let pc = (self.code.len() * 8) as u32;
            let op = Op::of(word);
            if op == Op::Invalid {
                return Err(Unmodeled(format!(
                    "shader: unknown instruction {word:#018x} at {pc:#x}"
                )));
            }
            self.code.push(Some(exec::Inst { op, word }));
            if op == Op::Bra && exec::is_branch_to_self(word, pc) {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

/// One constant buffer slot as a stage sees it.
#[derive(Clone, Copy, Debug, Default)]
pub struct ConstantBuffer<'a> {
    /// The leading bytes of the buffer that were read from memory.
    pub data: &'a [u8],
    /// The bound size (0: unbound). Reads beyond it return 0 [public: eden
    /// `shader_recompiler/backend/glsl/emit_glsl_context_get_set.cpp`
    /// `GetCbuf`, which returns 0 for offsets outside the 64 KiB buffer
    /// window; within the window the bound size is what the hardware's
    /// constant buffer bounds check uses]. A read within it but past
    /// `data` reaches memory the GPU could not read (the buffer's tail is
    /// not mapped) and is refused: on hardware it would fault.
    pub size: usize,
}

impl<'a> ConstantBuffer<'a> {
    /// A buffer whose bound size is all readable.
    pub fn whole(data: &'a [u8]) -> Self {
        Self {
            data,
            size: data.len(),
        }
    }
}

/// What a stage's instructions read besides registers and attributes.
pub struct Env<'a> {
    /// Bound constant buffers of this stage, slots 0-17.
    pub const_buffers: [ConstantBuffer<'a>; 18],
    pub textures: &'a dyn TextureUnit,
    /// Constant buffer slot holding bound-texture handles (the 3D engine's
    /// `TEXTURE_BUFFER_INDEX`); the handle of `TEX ... tex_index` is the
    /// u32 at byte offset `tex_index * 4` of that buffer.
    pub texture_buffer_index: u32,
    /// Global memory (`LDG`/`STG`/`LD`/`ST`).
    pub memory: &'a dyn GpuMemory,
    /// The value `S2R SR_Y_DIRECTION` reads (+1.0 or -1.0, from the 3D
    /// engine's window origin state).
    pub y_direction: f32,
}

/// The attribute space of one vertex: bytes 0..0x400 as 256 words.
/// a\[0x070..0x080) is the position, a\[0x080 + 16 * i\] generic attribute
/// `i`, a\[0x2f8\] the instance ID, a\[0x2fc\] the vertex ID
/// [public: Ryujinx `Translation/AttributeConsts.cs`; NVIDIA open-gpu-doc
/// Shader-Program-Header, Imap/Omap order].
#[derive(Clone, PartialEq, Eq)]
pub struct Attributes(pub [u32; 256]);

impl Default for Attributes {
    fn default() -> Attributes {
        Attributes([0; 256])
    }
}

impl std::fmt::Debug for Attributes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map()
            .entries(
                self.0
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| **v != 0)
                    .map(|(i, v)| (format!("a[{:#05x}]", i * 4), format!("{v:#010x}"))),
            )
            .finish()
    }
}

/// value(x, y) = base + dx * x + dy * y in window coordinates.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Plane {
    pub dx: f32,
    pub dy: f32,
    pub base: f32,
}

/// One 2x2 quad of fragments.
pub struct Quad<'a> {
    /// One plane per attribute word, prepared by the rasterizer for the
    /// SPH's interpolation mode of that word (perspective: attr/w; linear:
    /// attr; flat: constant; a\[0x070\]/a\[0x074\]: window x/y; a\[0x078\]:
    /// depth z; a\[0x07c\]: 1/w). `IPA` evaluates the plane at the lane's
    /// pixel center (x + 0.5, y + 0.5); MULTIPLY then multiplies by its operand.
    pub planes: &'a [Plane; 256],
    /// Window coordinates of lane 0 (both even). Lanes: 0 (x, y),
    /// 1 (x + 1, y), 2 (x, y + 1), 3 (x + 1, y + 1).
    pub x: u32,
    pub y: u32,
    /// Lanes inside the primitive. Uncovered lanes still execute as
    /// helpers (for derivatives) but produce no output and no memory
    /// writes.
    pub covered: [bool; 4],
    pub front_facing: bool,
    pub outputs: [FragmentOutput; 4],
}

/// What one fragment produced.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FragmentOutput {
    /// Per render target RGBA; components the SPH output map does not
    /// declare are 0.
    pub colors: [[f32; 4]; 8],
    /// Present when the SPH declares a depth output.
    pub depth: Option<f32>,
    /// The fragment executed `KIL`.
    pub discarded: bool,
}
