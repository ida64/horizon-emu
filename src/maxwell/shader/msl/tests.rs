//! Every pixel program of the firmware corpus (see the `shader` module
//! documentation) either translates to MSL that Metal compiles, or is
//! refused with a reason; and every translation computes, on the GPU, what
//! the interpreter computes from the same planes, constant buffers and
//! textures. Without a Metal device the GPU halves are skipped (the
//! translation itself still runs).

use std::collections::{BTreeMap, HashSet};
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLClearColor, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLLoadAction,
    MTLMathMode, MTLOrigin, MTLPixelFormat, MTLPrimitiveType, MTLRegion, MTLRenderCommandEncoder,
    MTLRenderPassDescriptor, MTLRenderPipelineDescriptor, MTLRenderPipelineState,
    MTLResourceOptions, MTLSamplerAddressMode, MTLSamplerDescriptor, MTLSamplerMinMagFilter,
    MTLSamplerState, MTLSize, MTLStorageMode, MTLStoreAction, MTLTexture, MTLTextureDescriptor,
    MTLTextureUsage,
};

use super::super::super::sampling::{
    FetchRequest, GatherRequest, Query, SampleRequest, Target as TextureTarget, TextureUnit,
};
use super::super::super::{GpuMemory, Unmodeled};
use super::super::tests::{ROOT, TITLE, elf, members, stages, yaz0};
use super::super::{ConstantBuffer, Env, FragmentOutput, Plane, Program, Quad, ShaderType};
use super::{Translated, translate};

/// The corpus: (name, program) for every distinct pixel program.
fn corpus() -> Vec<(String, Program)> {
    let mut seen = HashSet::new();
    let mut programs = Vec::new();
    let mut add = |source: &str, data: &[u8]| {
        for (offset, p) in stages(data) {
            let code = data[offset + 0x80..offset + 0x30 + p.len()].to_vec();
            if seen.insert(code) && p.header().shader_type() == ShaderType::Pixel {
                programs.push((format!("{source}@{offset:#x}"), p));
            }
        }
    };
    add("starter.elf", &elf());
    for name in [
        "shader/VarietyOceanShader_Nx.arc.szs",
        "agl/agl_resource_min.Nin_Nx_NVN.release.sarc.szs",
    ] {
        let data = std::fs::read(format!("{ROOT}fw_dump/{TITLE}section1_romfs/common/{name}"))
            .expect("required shader archive");
        let short = name.split('/').nth(1).expect("file name");
        for (index, member) in members(&yaz0(&data)).into_iter().enumerate() {
            add(&format!("{short}#{index}"), member);
        }
    }
    let name = "sead/primitive_renderer/primitive_drawer_nvn_shader.bin";
    let data = std::fs::read(format!("{ROOT}fw_dump/{TITLE}section1_romfs/common/{name}"))
        .expect("required primitive shader");
    add("primitive_drawer", &data);
    programs
}

type Device = ProtocolObject<dyn MTLDevice>;

/// The offscreen Metal device of the tests.
struct Gpu {
    device: Retained<Device>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

impl Gpu {
    fn new() -> Option<Gpu> {
        let device = MTLCreateSystemDefaultDevice()?;
        let queue = device.newCommandQueue()?;
        Some(Gpu { device, queue })
    }

    /// A render pipeline of `source` with `targets` RGBA32Float color
    /// attachments.
    fn pipeline(
        &self,
        source: &str,
        targets: usize,
    ) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
        let options = MTLCompileOptions::new();
        options.setMathMode(MTLMathMode::Safe);
        let library = self
            .device
            .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&options))
            .map_err(|e| format!("compilation failed: {e}"))?;
        let function = |name: &str| {
            library
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or_else(|| format!("no function {name}"))
        };
        let descriptor = MTLRenderPipelineDescriptor::new();
        let vertex = function("vertex_main")?;
        let fragment = function("fragment_main")?;
        descriptor.setVertexFunction(Some(&vertex));
        descriptor.setFragmentFunction(Some(&fragment));
        for i in 0..targets {
            // SAFETY: fewer than 8 attachments.
            let attachment = unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(i) };
            attachment.setPixelFormat(MTLPixelFormat::RGBA32Float);
        }
        self.device
            .newRenderPipelineStateWithDescriptor_error(&descriptor)
            .map_err(|e| format!("pipeline creation failed: {e}"))
    }

    fn buffer(&self, data: &[u8]) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        let padded = [0u8; 16];
        let data = if data.is_empty() { &padded[..] } else { data };
        // SAFETY: `data` is valid for its length; Metal copies it.
        unsafe {
            self.device.newBufferWithBytes_length_options(
                NonNull::new(data.as_ptr() as *mut _).expect("non-null"),
                data.len(),
                MTLResourceOptions::StorageModeShared,
            )
        }
        .expect("buffer")
    }

    fn texture(
        &self,
        width: usize,
        height: usize,
        usage: MTLTextureUsage,
    ) -> Retained<ProtocolObject<dyn MTLTexture>> {
        // SAFETY: plain descriptor construction.
        let descriptor = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat::RGBA32Float,
                width,
                height,
                false,
            )
        };
        descriptor.setStorageMode(MTLStorageMode::Shared);
        descriptor.setUsage(usage);
        self.device
            .newTextureWithDescriptor(&descriptor)
            .expect("texture")
    }
}

fn region(width: usize, height: usize) -> MTLRegion {
    MTLRegion {
        origin: MTLOrigin { x: 0, y: 0, z: 0 },
        size: MTLSize {
            width,
            height,
            depth: 1,
        },
    }
}

fn bytes<T: Copy>(values: &[T]) -> Vec<u8> {
    // SAFETY: only used with f32/u32, which have no padding.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
    .to_vec()
}

/// The color targets a program writes.
fn targets(program: &Program) -> Vec<usize> {
    (0..8)
        .filter(|&t| program.header().ps_color_mask(t) != 0)
        .collect()
}

/// A constant buffer 1 for `BRX` jump tables: for each `BRX`, 8 words at
/// the offset of the `LDC` that loads its register (found by scanning
/// back), whose targets are the 8 instructions after the branch (the shape
/// of a compiled `switch`; the real table is draw state, not part of the
/// program).
fn jump_table(program: &Program) -> Vec<u8> {
    let mut words = vec![0u32; 64];
    let instructions: Vec<(u32, super::super::exec::Inst)> = program
        .code
        .iter()
        .enumerate()
        .filter_map(|(i, inst)| inst.map(|inst| (i as u32 * 8, inst)))
        .collect();
    for (position, &(pc, inst)) in instructions.iter().enumerate() {
        if inst.op != super::Op::Brx {
            continue;
        }
        let reg = super::bits(inst.word, 8, 8);
        let Some(&(_, load)) = instructions[..position]
            .iter()
            .rev()
            .take(64)
            .find(|(_, i)| i.op == super::Op::Ldc && super::bits(i.word, 0, 8) == reg)
        else {
            continue;
        };
        let offset = super::sext(super::bits(load.word, 20, 16), 16) as u32;
        let base = (pc + 8).wrapping_add(super::sext(super::bits(inst.word, 20, 24), 24) as u32);
        let mut target = pc;
        for k in 0..8 {
            target = super::flow::next_pc(target);
            if let Some(word) = words.get_mut((offset / 4 + k) as usize) {
                *word = target.wrapping_sub(base);
            }
        }
    }
    words.into_iter().flat_map(u32::to_le_bytes).collect()
}

/// Every corpus pixel program translates or is refused with a reason;
/// every translation compiles.
#[test]
fn corpus_translates_and_compiles() {
    let gpu = Gpu::new();
    let programs = corpus();
    let mut refused: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut translated = Vec::new();
    let mut slowest = (std::time::Duration::ZERO, String::new());
    for (name, program) in &programs {
        match translate(program, &jump_table(program)) {
            Ok(t) => {
                if let Some(gpu) = &gpu {
                    let outputs = targets(program);
                    let source = t.source(program, &outputs, false, None, None);
                    let start = std::time::Instant::now();
                    if let Err(e) = gpu.pipeline(&source, outputs.len()) {
                        panic!("{name}: {e}\n{source}");
                    }
                    let time = start.elapsed();
                    if time > slowest.0 {
                        slowest = (time, name.clone());
                    }
                }
                translated.push(name.as_str());
                if !t.indirect.is_empty() {
                    println!("{name}: indirect constant buffer slots {:?}", t.indirect);
                }
                if program.header().ps_writes_depth() {
                    println!("{name}: depth output");
                }
            }
            Err(reason) => {
                // "{mnemonic} {what} {word} at {pc}" without the word.
                let key = match reason.rsplit_once(" at 0x") {
                    Some((head, pc)) => {
                        let what = head.rsplit_once(' ').map_or(head, |(what, _)| what);
                        format!("{what} at 0x{pc}")
                    }
                    None => reason.clone(),
                };
                refused.entry(key).or_default().push(name.clone());
            }
        }
    }
    println!(
        "{} of {} pixel programs translated{}",
        translated.len(),
        programs.len(),
        if gpu.is_some() {
            " and compiled"
        } else {
            " (no Metal device: not compiled)"
        }
    );
    println!("slowest compile: {} {:?}", slowest.1, slowest.0);
    for (reason, names) in &refused {
        println!("refused ({}): {reason}: {names:?}", names.len());
    }
    // The ui2d/font programs starter draws its first-boot logo and attach
    // animation with (captured from those draws).
    for offset in [0x4bb800, 0x4bba00, 0x4beb00, 0x4bf500] {
        let name = format!("starter.elf@{offset:#x}");
        assert!(translated.contains(&name.as_str()), "{name} not translated");
    }
}

/// A deterministic generator (xorshift64*).
struct Random(u64);

impl Random {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 32) as u32
    }

    /// Uniform in [low, high).
    fn float(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * (self.next() >> 8) as f32 / (1u32 << 24) as f32
    }

    /// A constant buffer word: mostly moderate floats, some small integers
    /// (indices, flags, counts) and some raw bits.
    fn word(&mut self) -> u32 {
        match self.next() % 8 {
            0..=4 => self.float(-2.0, 2.0).to_bits(),
            5 | 6 => self.next() % 8,
            _ => self.next(),
        }
    }
}

/// 1x1 textures: every sample of handle `h` is texel `texels[h - 1]`, as
/// with nearest filtering on the GPU whatever the coordinates and level.
struct Texels(Vec<[f32; 4]>);

impl TextureUnit for Texels {
    fn sample(&self, handle: u32, _: &SampleRequest) -> Result<[f32; 4], Unmodeled> {
        self.0
            .get(handle.wrapping_sub(1) as usize)
            .copied()
            .ok_or_else(|| Unmodeled(format!("test: handle {handle}")))
    }
    fn fetch(&self, _: u32, _: &FetchRequest) -> Result<[f32; 4], Unmodeled> {
        Err(Unmodeled("test: fetch".into()))
    }
    fn gather(&self, _: u32, _: &GatherRequest) -> Result<[f32; 4], Unmodeled> {
        Err(Unmodeled("test: gather".into()))
    }
    fn query(&self, _: u32, _: Query) -> Result<[u32; 4], Unmodeled> {
        Err(Unmodeled("test: query".into()))
    }
    fn lod(
        &self,
        _: u32,
        _: TextureTarget,
        _: [f32; 3],
        _: [f32; 3],
    ) -> Result<[f32; 2], Unmodeled> {
        Err(Unmodeled("test: lod".into()))
    }
}

struct NoMemory;

impl GpuMemory for NoMemory {
    fn read(&self, _: u64, _: &mut [u8]) -> Option<()> {
        None
    }
    fn write(&self, _: u64, _: &[u8]) -> Option<()> {
        None
    }
}

const WIDTH: usize = 8;
const HEIGHT: usize = 8;
/// A triangle (`Translated::source` buffer 24) covering the target: corners
/// 256 pixels out, counter-clockwise with y down.
const COVERING: [i32; super::TRIANGLE_WORDS] = [
    -0x1_0000, -0x1_0000, 0x4_0000, -0x1_0000, -0x1_0000, 0x4_0000, 0, 0, 0, 0,
];
/// The bound-texture constant buffer slot of the tests.
const TEXTURE_SLOT: u32 = 2;
/// What a discarded pixel keeps on the GPU.
const CLEARED: f32 = -1234.5;

/// The inputs of one differential run.
struct Inputs {
    planes: [Plane; 256],
    buffers: Vec<Vec<u8>>,
    texels: Vec<[f32; 4]>,
}

fn inputs(t: &Translated, seed: u64) -> Inputs {
    let mut random = Random(seed | 1);
    let mut planes = [Plane::default(); 256];
    for &word in &t.ipa_words {
        planes[word as usize] = Plane {
            dx: random.float(-0.1, 0.1),
            dy: random.float(-0.1, 0.1),
            base: random.float(-1.0, 2.0),
        };
    }
    let mut buffers = vec![Vec::new(); 18];
    let mut sizes = [0usize; 18];
    for &(slot, offset) in &t.constants {
        let size = &mut sizes[slot as usize];
        *size = (*size).max(offset as usize + 4);
    }
    for &slot in &t.indirect {
        sizes[slot as usize] = sizes[slot as usize].max(0x400);
    }
    let handles = t
        .textures
        .iter()
        .map(|&h| h as usize + 4)
        .max()
        .unwrap_or(0);
    sizes[TEXTURE_SLOT as usize] = sizes[TEXTURE_SLOT as usize].max(handles);
    for (buffer, &size) in buffers.iter_mut().zip(&sizes) {
        *buffer = (0..size.div_ceil(4))
            .flat_map(|_| random.word().to_le_bytes())
            .collect();
    }
    for (index, &handle) in t.textures.iter().enumerate() {
        let at = handle as usize;
        buffers[TEXTURE_SLOT as usize][at..at + 4]
            .copy_from_slice(&(index as u32 + 1).to_le_bytes());
    }
    // The jump tables the translation was made for.
    if let Some(table) = &t.constant_buffer_1 {
        buffers[1].clone_from(table);
    }
    let texels = t
        .textures
        .iter()
        .map(|_| std::array::from_fn(|_| random.float(0.0, 1.0)))
        .collect();
    Inputs {
        planes,
        buffers,
        texels,
    }
}

/// The interpreter's outputs of every pixel, row-major.
fn interpret(program: &Program, input: &Inputs) -> Result<Vec<FragmentOutput>, Unmodeled> {
    let unit = Texels(input.texels.clone());
    let mut const_buffers = [ConstantBuffer::default(); 18];
    for (slot, data) in input.buffers.iter().enumerate() {
        const_buffers[slot] = ConstantBuffer::whole(data);
    }
    let env = Env {
        const_buffers,
        textures: &unit,
        texture_buffer_index: TEXTURE_SLOT,
        memory: &NoMemory,
        y_direction: 1.0,
    };
    let mut pixels = vec![FragmentOutput::default(); WIDTH * HEIGHT];
    for y in (0..HEIGHT).step_by(2) {
        for x in (0..WIDTH).step_by(2) {
            let mut quad = Quad {
                planes: &input.planes,
                x: x as u32,
                y: y as u32,
                covered: [true; 4],
                centroids: [[0.5; 2]; 4],
                front_facing: true,
                outputs: [FragmentOutput::default(); 4],
            };
            program.run_fragment(&env, &mut quad)?;
            for (lane, output) in quad.outputs.iter().enumerate() {
                pixels[(y + lane / 2) * WIDTH + x + lane % 2] = *output;
            }
        }
    }
    Ok(pixels)
}

/// The GPU's colors of every pixel, per target, row-major.
fn render(
    gpu: &Gpu,
    program: &Program,
    t: &Translated,
    input: &Inputs,
) -> Result<Vec<Vec<[f32; 4]>>, String> {
    let outputs = targets(program);
    let pipeline = gpu.pipeline(
        &t.source(program, &outputs, false, None, None),
        outputs.len(),
    )?;
    let positions: [[f32; 4]; 6] = [
        [-1.0, -1.0, 0.0, 1.0],
        [1.0, -1.0, 0.0, 1.0],
        [-1.0, 1.0, 0.0, 1.0],
        [1.0, -1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
        [-1.0, 1.0, 0.0, 1.0],
    ];
    let mut planes = Vec::new();
    for _ in 0..2 {
        if t.ipa_words.is_empty() {
            planes.extend([0.0f32; 3]);
        }
        for &word in &t.ipa_words {
            let p = input.planes[word as usize];
            planes.extend([p.dx, p.dy, p.base]);
        }
    }
    let biases = vec![0.0f32; t.textures.len() + 1];
    let sizes: Vec<u32> = input.buffers.iter().map(|b| b.len() as u32).collect();
    let usage = MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead;
    let attachments: Vec<_> = outputs
        .iter()
        .map(|_| gpu.texture(WIDTH, HEIGHT, usage))
        .collect();
    let textures: Vec<_> = input
        .texels
        .iter()
        .map(|texel| {
            let texture = gpu.texture(1, 1, MTLTextureUsage::ShaderRead);
            // SAFETY: one RGBA32F texel.
            unsafe {
                texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                    region(1, 1),
                    0,
                    NonNull::new(texel.as_ptr() as *mut _).expect("non-null"),
                    16,
                );
            }
            texture
        })
        .collect();
    let sampler_descriptor = MTLSamplerDescriptor::new();
    sampler_descriptor.setMinFilter(MTLSamplerMinMagFilter::Nearest);
    sampler_descriptor.setMagFilter(MTLSamplerMinMagFilter::Nearest);
    sampler_descriptor.setSAddressMode(MTLSamplerAddressMode::Repeat);
    sampler_descriptor.setTAddressMode(MTLSamplerAddressMode::Repeat);
    let sampler: Retained<ProtocolObject<dyn MTLSamplerState>> = gpu
        .device
        .newSamplerStateWithDescriptor(&sampler_descriptor)
        .expect("sampler");
    let pass = MTLRenderPassDescriptor::new();
    for (i, texture) in attachments.iter().enumerate() {
        // SAFETY: fewer than 8 attachments.
        let attachment = unsafe { pass.colorAttachments().objectAtIndexedSubscript(i) };
        attachment.setTexture(Some(texture));
        attachment.setLoadAction(MTLLoadAction::Clear);
        attachment.setClearColor(MTLClearColor {
            red: f64::from(CLEARED),
            green: f64::from(CLEARED),
            blue: f64::from(CLEARED),
            alpha: f64::from(CLEARED),
        });
        attachment.setStoreAction(MTLStoreAction::Store);
    }
    let command = gpu.queue.commandBuffer().expect("command buffer");
    let encoder = command
        .renderCommandEncoderWithDescriptor(&pass)
        .expect("encoder");
    encoder.setRenderPipelineState(&pipeline);
    // Buffers must stay alive until the command buffer completes.
    let buffers: Vec<_> = [bytes(&positions), bytes(&planes), bytes(&biases)]
        .into_iter()
        .chain(input.buffers.iter().map(|data| {
            // Static reads past the bound size read zeros (see
            // `Translated::source`): pad.
            let mut data = data.clone();
            data.resize(data.len().max(0x10000), 0);
            data
        }))
        .chain([bytes(&sizes)])
        // Texture fault words: every texture bound.
        .chain([vec![0u8; 8 * t.textures.len().max(1)]])
        // The triangle (both halves of the quad are triangle 0): one
        // covering the target, no depth plane.
        .chain([bytes(&COVERING)])
        .map(|data| gpu.buffer(&data))
        .collect();
    // SAFETY: buffers and textures outlive the command buffer; indices are
    // the function's bindings.
    unsafe {
        encoder.setVertexBuffer_offset_atIndex(Some(&buffers[0]), 0, 0);
        encoder.setFragmentBuffer_offset_atIndex(Some(&buffers[1]), 0, 0);
        encoder.setFragmentBuffer_offset_atIndex(Some(&buffers[2]), 0, 1);
        for slot in 0..18 {
            encoder.setFragmentBuffer_offset_atIndex(Some(&buffers[3 + slot]), 0, 2 + slot);
        }
        encoder.setFragmentBuffer_offset_atIndex(Some(&buffers[21]), 0, 20);
        encoder.setFragmentBuffer_offset_atIndex(Some(&buffers[22]), 0, 21);
        encoder.setFragmentBuffer_offset_atIndex(Some(&buffers[23]), 0, 24);
        for (i, texture) in textures.iter().enumerate() {
            encoder.setFragmentTexture_atIndex(Some(texture), i);
            encoder.setFragmentSamplerState_atIndex(Some(&sampler), i);
        }
        encoder.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 6);
    }
    encoder.endEncoding();
    command.commit();
    command.waitUntilCompleted();
    Ok(attachments
        .iter()
        .map(|texture| {
            let mut pixels = vec![[0f32; 4]; WIDTH * HEIGHT];
            // SAFETY: the buffer holds the texture's rows; the GPU finished.
            unsafe {
                texture.getBytes_bytesPerRow_fromRegion_mipmapLevel(
                    NonNull::new(pixels.as_mut_ptr().cast()).expect("non-null"),
                    WIDTH * 16,
                    region(WIDTH, HEIGHT),
                    0,
                );
            }
            pixels
        })
        .collect())
}

/// Distance in units in the last place between two floats of one sign
/// (`u32::MAX` across signs or with one NaN).
fn ulps(a: f32, b: f32) -> u32 {
    if a.is_nan() || b.is_nan() || (a.is_sign_negative() != b.is_sign_negative() && a != b) {
        return u32::MAX;
    }
    a.to_bits().abs_diff(b.to_bits())
}

/// Every translated corpus program, on random inputs, produces on the GPU
/// what the interpreter produces: same discards, and colors bit-identical
/// except where the program uses transcendental `MUFU` operations.
#[test]
fn translations_match_interpreter() {
    let Some(gpu) = Gpu::new() else {
        println!("no Metal device: differential test skipped");
        return;
    };
    let mut report = Vec::new();
    let mut worst = (0f32, 0u32, String::new());
    for (name, program) in corpus() {
        let Ok(t) = translate(&program, &jump_table(&program)) else {
            continue;
        };
        let transcendental = program
            .code
            .iter()
            .flatten()
            .any(|inst| inst.op == super::Op::Mufu && super::bits(inst.word, 20, 4) <= 3);
        let outputs = targets(&program);
        let (mut runs, mut skipped, mut differing) = (0, 0, 0usize);
        let (mut max_abs, mut max_ulps) = (0f32, 0u32);
        for seed in 1..=4u64 {
            let input = inputs(&t, seed.wrapping_mul(0x9e37_79b9_7f4a_7c15));
            let Ok(expected) = interpret(&program, &input) else {
                skipped += 1;
                continue;
            };
            runs += 1;
            let colors =
                render(&gpu, &program, &t, &input).unwrap_or_else(|e| panic!("{name}: {e}"));
            for (pixel, output) in expected.iter().enumerate() {
                for (attachment, &target) in outputs.iter().enumerate() {
                    let got = colors[attachment][pixel];
                    if output.discarded {
                        assert_eq!(got, [CLEARED; 4], "{name}: pixel {pixel} not discarded");
                        continue;
                    }
                    assert_ne!(got, [CLEARED; 4], "{name}: pixel {pixel} discarded");
                    for (c, (&want, &have)) in output.colors[target].iter().zip(&got).enumerate() {
                        if want.to_bits() == have.to_bits() || (want.is_nan() && have.is_nan()) {
                            continue;
                        }
                        differing += 1;
                        let u = ulps(want, have);
                        if !transcendental {
                            panic!(
                                "{name}: pixel {pixel} target {target} component {c}: interpreter {want:e} ({:#010x}), Metal {have:e} ({:#010x})",
                                want.to_bits(),
                                have.to_bits()
                            );
                        }
                        max_ulps = max_ulps.max(u);
                        let d = (want - have).abs();
                        if d > max_abs || d.is_nan() {
                            max_abs = d;
                        }
                    }
                }
            }
        }
        if max_abs > worst.0 {
            worst = (max_abs, max_ulps, name.clone());
        }
        report.push(format!(
            "{name}: {runs} runs ({skipped} refused by the interpreter), {differing} differing components, max |diff| {max_abs:e}, max ulps {max_ulps}"
        ));
    }
    for line in &report {
        println!("{line}");
    }
    println!(
        "{} programs compared; worst: {} (|diff| {:e}, {} ulps)",
        report.len(),
        worst.2,
        worst.0,
        worst.1
    );
    // Only transcendental MUFU differs (checked above); its error stays
    // within a small fraction of the color range.
    assert!(worst.0 <= 1e-3, "worst difference {worst:?}");
}

/// Programs from instruction encodings [public: Ryujinx
/// `Decoders/InstDecoders.cs`], laid out as `shader::tests::assembled`.
fn assembled(words: &[u64]) -> Program {
    let mut bytes = vec![0; 0x50];
    bytes[..4].copy_from_slice(&0x1402u32.to_le_bytes());
    bytes[0x48] = 15;
    for chunk in words.chunks(3) {
        bytes.extend_from_slice(&0u64.to_le_bytes());
        for i in 0..3 {
            let word = chunk.get(i).copied().unwrap_or(0x50b0000000070f00);
            bytes.extend_from_slice(&word.to_le_bytes());
        }
    }
    bytes.extend_from_slice(&[0; 32]);
    Program::from_bytes(&bytes).expect("program")
}

fn mov(rd: u64, v: u32) -> u64 {
    0x010000000007f000 | (u64::from(v) << 20) | rd
}

/// A branch at word `from` to word `to` (`shader::tests::branch`).
fn branch(base: u64, from: usize, to: usize) -> u64 {
    let pc = |index: usize| ((index / 3) * 32 + (index % 3 + 1) * 8) as i32;
    base | ((u64::from((pc(to) - pc(from) - 8) as u32) & 0xffffff) << 20)
}

/// Divergent if/else (SSY/SYNC on one side only, so the other side's lane
/// reaches the PBK with the SSY entry still stacked and BRK unwinds it),
/// a counted loop with a predicated BRK, and a predicated KIL: the forms
/// of `shader::tests` `reconvergence_break_and_indirect_branch`,
/// `counted_loop_breaks_at_predicate` and `predicate_kill`.
fn control_flow_programs() -> Vec<Program> {
    let fsetp_lt_p0 = 0x5bb1000000070007 | (4 << 8) | (5 << 20) | (7 << 39);
    vec![
        assembled(&[
            0xe003ff870ff7ff04, // IPA R4, a[0x70] (window x)
            mov(5, 1f32.to_bits()),
            fsetp_lt_p0,
            branch(0xe290000000070000, 3, 8), // SSY
            branch(0xe24000000000000f, 4, 7), // @P0 BRA
            mov(0, 2f32.to_bits()),
            0xf0f800000007000f, // SYNC
            mov(0, 3f32.to_bits()),
            branch(0xe2a0000000070000, 8, 11), // PBK
            0xe34000000007000f,                // BRK
            mov(0, 99f32.to_bits()),
            0xe30000000007000f, // EXIT
        ]),
        assembled(&[
            mov(4, 0),
            branch(0xe2a0000000070000, 1, 7), // PBK
            0x1c00000000170404,               // IADD32I R4, R4, 1
            0x366c038000370407,               // ISETP P0
            0xe34000000000000f,               // @P0 BRK
            branch(0xe24000000007000f, 5, 2), // BRA
            mov(4, 999),
            0x5c98078000470000, // MOV R0, R4
            0xe30000000007000f,
        ]),
        assembled(&[
            0xe003ff870ff7ff04, // IPA R4, a[0x70]
            mov(5, 2f32.to_bits()),
            fsetp_lt_p0,
            0xe33000000000000f, // @P0 KIL
            mov(0, 7f32.to_bits()),
            0xe30000000007000f,
        ]),
    ]
}

#[test]
fn control_flow_matches_interpreter() {
    let gpu = Gpu::new();
    for (index, program) in control_flow_programs().iter().enumerate() {
        let t = translate(program, &[]).unwrap_or_else(|e| panic!("program {index}: {e}"));
        let Some(gpu) = &gpu else {
            continue;
        };
        for seed in 1..=2 {
            let mut input = inputs(&t, seed);
            // Window x for IPA a[0x70]: lanes diverge within quads.
            input.planes[0x70 / 4] = Plane {
                dx: 1.0,
                dy: 0.0,
                base: 0.0,
            };
            let expected = interpret(program, &input).expect("interprets");
            let colors = render(gpu, program, &t, &input).expect("renders");
            for (pixel, output) in expected.iter().enumerate() {
                let got = colors[0][pixel];
                if output.discarded {
                    assert_eq!(got, [CLEARED; 4], "program {index} pixel {pixel}");
                } else {
                    assert_eq!(
                        got.map(f32::to_bits),
                        output.colors[0].map(f32::to_bits),
                        "program {index} pixel {pixel}"
                    );
                }
            }
        }
    }
}

/// `I2I` R form [public: Ryujinx `InstI2iR`]: destination format `dst`
/// (bits 9:8, signed bit 12), source format `src` (bits 11:10, signed bit
/// 13), byte selector, -b, |b| and SAT.
#[allow(clippy::too_many_arguments)]
fn i2i(
    rd: u64,
    rb: u64,
    dst: (u64, bool),
    src: (u64, bool),
    sel: u64,
    neg: bool,
    abs: bool,
    sat: bool,
) -> u64 {
    0x5ce0_0000_0007_0000
        | rd
        | dst.0 << 8
        | src.0 << 10
        | u64::from(dst.1) << 12
        | u64::from(src.1) << 13
        | rb << 20
        | sel << 41
        | u64::from(neg) << 45
        | u64::from(abs) << 49
        | u64::from(sat) << 50
}

/// Integer conversions (the Mii editor's pixel programs take |x| of a
/// signed word with `I2I`) give the interpreter's results on Metal:
/// extension by source format and byte, wrapping |x| and -x, SAT.
#[test]
fn integer_conversions_match_interpreter() {
    let program = assembled(&[
        mov(4, 0x8000_0000),
        mov(5, 0xffff_fff3),
        mov(6, 0x89ab_cdef),
        // |s32 R4| wraps to i32::MIN.
        i2i(0, 4, (2, true), (2, true), 0, false, true, false),
        // -(s8 byte 2 of R6) = -(-0x55).
        i2i(1, 6, (2, true), (0, true), 2, true, false, false),
        // -(u16 half 1 of R6), SAT to u32: 0.
        i2i(2, 6, (2, false), (1, false), 2, true, false, true),
        // |s32 R5| = 13.
        i2i(3, 5, (2, true), (2, true), 0, false, true, false),
        0xe30000000007000f, // EXIT
    ]);
    let t = translate(&program, &[]).expect("translates");
    let input = inputs(&t, 1);
    let expected = interpret(&program, &input).expect("interprets");
    let words = [0x8000_0000u32, 0x55, 0, 13];
    assert_eq!(expected[0].colors[0].map(f32::to_bits), words);
    let Some(gpu) = Gpu::new() else {
        return;
    };
    let colors = render(&gpu, &program, &t, &input).expect("renders");
    for pixel in &colors[0] {
        assert_eq!(pixel.map(f32::to_bits), words);
    }
}

/// `SHFL` with immediate operands: mode, destination, source, lane or
/// offset `b`, clamp and segment mask `c`, destination predicate `pd`
/// [public: Ryujinx `InstShfl`].
fn shfl(mode: u64, rd: u64, ra: u64, b: u64, c: u64, pd: u64) -> u64 {
    0xef10_0000_0007_0000 | rd | ra << 8 | b << 20 | 3 << 28 | mode << 30 | c << 34 | pd << 48
}

/// `FSWZADD` R`rd` = R`ra` and R`rb` by the per-lane operations `ops`
/// [public: Ryujinx `InstFswzadd`].
fn fswzadd(rd: u64, ra: u64, rb: u64, ops: u64) -> u64 {
    0x50f8_0000_0007_0000 | rd | ra << 8 | rb << 20 | ops << 28
}

/// Quad shuffles (butterfly, indexed, up and down within the quad, and
/// the in-range predicate) read on Metal the lanes the interpreter reads:
/// `quad_shuffle`'s lane numbering is the interpreter's `(x & 1) | (y &
/// 1) << 1`. `FSWZADD` picks each lane's operation by the same numbering
/// (fine derivatives as compilers pair it with `SHFL.BFLY`). A shuffle
/// reaching another quad of the warp is refused by both.
#[test]
fn shfl_matches_interpreter() {
    const IPA_X: u64 = 0xe003ff870ff7ff04; // IPA R4, a[0x70] (window x)
    const IPA_Y: u64 = 0xe003ff874ff7ff05; // IPA R5, a[0x74] (window y)
    const QUAD: u64 = 0x1c03; // segments of four lanes, clamp 3
    let guarded_mov = |rd: u64, v: u32, p: u64| mov(rd, v) & !0x7_0000 | p << 16;
    let programs = [
        assembled(&[
            IPA_X,
            IPA_Y,
            shfl(3, 0, 4, 1, QUAD, 7),   // BFLY 1: x neighbour
            shfl(3, 1, 5, 2, QUAD, 7),   // BFLY 2: y neighbour
            shfl(0, 2, 4, 3, QUAD, 7),   // IDX 3
            shfl(1, 3, 5, 1, 0x1c00, 7), // UP 1
            0xe30000000007000f,          // EXIT
        ]),
        assembled(&[
            IPA_X,
            IPA_Y,
            shfl(2, 0, 4, 1, QUAD, 0), // DOWN 1, P0 in range
            mov(1, 0),
            guarded_mov(1, 1f32.to_bits(), 0),
            shfl(0, 2, 5, 1, 0x1e01, 1), // IDX 1 in segments of two
            mov(3, 0),
            guarded_mov(3, 1f32.to_bits(), 1),
            0xe30000000007000f,
        ]),
        assembled(&[
            IPA_X,
            IPA_Y,
            shfl(3, 6, 4, 1, QUAD, 7),
            fswzadd(0, 4, 6, 0x99), // dFdx(x)
            shfl(3, 7, 5, 2, QUAD, 7),
            fswzadd(1, 5, 7, 0xa5), // dFdy(y)
            fswzadd(2, 4, 5, 0x00), // x + y
            fswzadd(3, 4, 5, 0xff), // y
            0xe30000000007000f,
        ]),
    ];
    let gpu = Gpu::new();
    for (index, program) in programs.iter().enumerate() {
        let t = translate(program, &[]).unwrap_or_else(|e| panic!("program {index}: {e}"));
        let mut input = inputs(&t, 1);
        input.planes[0x70 / 4] = Plane {
            dx: 1.0,
            dy: 0.0,
            base: 0.0,
        };
        input.planes[0x74 / 4] = Plane {
            dx: 0.0,
            dy: 1.0,
            base: 0.0,
        };
        let expected = interpret(program, &input).expect("interprets");
        if index == 0 {
            // Pixel (0, 0): x of (1, 0), y of (0, 1), x of (1, 1), its own y.
            assert_eq!(expected[0].colors[0], [1.5, 1.5, 1.5, 0.5]);
            // Pixel (1, 1): UP 1 reads lane 2, (0, 1).
            assert_eq!(expected[WIDTH + 1].colors[0][3], 1.5);
        }
        if index == 2 {
            assert_eq!(expected[WIDTH + 1].colors[0], [1.0, 1.0, 3.0, 1.5]);
        }
        let Some(gpu) = &gpu else {
            continue;
        };
        let colors = render(gpu, program, &t, &input).expect("renders");
        for (pixel, output) in expected.iter().enumerate() {
            assert_eq!(
                colors[0][pixel].map(f32::to_bits),
                output.colors[0].map(f32::to_bits),
                "program {index} pixel {pixel}"
            );
        }
    }
    // BFLY 4 reads the next quad of the warp.
    let across = assembled(&[IPA_X, shfl(3, 0, 4, 4, 0x1c07, 7), 0xe30000000007000f]);
    assert!(translate(&across, &[]).is_err());
    let plain = assembled(&[IPA_X, 0xe30000000007000f]);
    let input = inputs(&translate(&plain, &[]).unwrap(), 1);
    assert!(interpret(&across, &input).is_err());
}
