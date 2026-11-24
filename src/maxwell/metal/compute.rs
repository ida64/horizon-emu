//! Compute launches on the GPU: a kernel translated to a Metal kernel
//! function (`shader::msl` `compute.rs`) dispatched in the channel's
//! command stream, after the work recorded before it and before the work
//! after it, as the compute class runs launches in method order.
//!
//! What the kernel reads and writes is what draws read and write: its
//! constant buffers, the textures its handles name (render targets held
//! on the GPU through [`Renderer::composite`]) and global memory in the
//! one buffer every program addresses (`Globals`), whose regions it
//! stores to are written back at the channel's next signal like a draw's.
//! A launch the translation or the bindings cannot express is the
//! interpreter's, as a draw is the software renderer's.

use std::sync::Arc;

use objc2::rc::Retained;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLDevice, MTLLibrary, MTLSize,
};

use super::super::resources::{DrawTextures, Writes};
use super::super::shader::msl;
use super::super::shader::{Kernel, Launch};
use super::super::{GpuMemory, Unmodeled};
use super::{
    ARENA_LIMIT, ComputePipeline, Renderer, Sampled, bytes_of_u32, composite_kind, format,
    query_words,
    stage_constants,
};

/// A kernel's translation and its pipeline, kept while the compute class
/// keeps the kernel (as [`super::Program2`]).
pub(super) struct KernelEntry {
    pub(super) kernel: Arc<Kernel>,
    translated: Result<Arc<msl::TranslatedCompute>, String>,
    pipeline: Option<Retained<ComputePipeline>>,
}

/// One compute launch.
pub struct Dispatch<'a> {
    pub kernel: &'a Arc<Kernel>,
    /// The launch's constant buffers and their bound sizes.
    pub constants: &'a [(Vec<u8>, usize)],
    pub textures: &'a DrawTextures<'a>,
    /// The constant buffer holding the bound-texture handles.
    pub texture_buffer_index: u32,
    pub launch: Launch,
}

impl Renderer {
    /// A compute launch on Metal; `Ok(false)` if Metal cannot run it
    /// (nothing was changed then).
    pub fn dispatch(
        &self,
        memory: &dyn GpuMemory,
        writes: &Writes,
        dispatch: &Dispatch<'_>,
    ) -> Result<bool, Unmodeled> {
        let launch = dispatch.launch;
        let constant_buffer_1 = dispatch
            .constants
            .get(1)
            .map_or(&[][..], |(data, size)| &data[..data.len().min(*size)]);
        let translated = {
            let mut state = self.state.lock().unwrap();
            let key = Arc::as_ptr(dispatch.kernel) as usize;
            let entry = state.kernels.entry(key).or_insert_with(|| KernelEntry {
                kernel: Arc::clone(dispatch.kernel),
                translated: msl::translate_compute(&dispatch.kernel.program(), constant_buffer_1)
                    .map(Arc::new),
                pipeline: None,
            });
            if let Ok(t) = &entry.translated
                && t.constant_buffer_1
                    .as_deref()
                    .is_some_and(|table| table != constant_buffer_1)
            {
                entry.translated =
                    msl::translate_compute(&dispatch.kernel.program(), constant_buffer_1)
                        .map(Arc::new);
                entry.pipeline = None;
            }
            entry.translated.clone()
        };
        let translated = match translated {
            Ok(t) => t,
            Err(reason) => return self.fall_back(format!("compute program: {reason}")),
        };
        // The interpreter refuses a barrier the launch does not declare.
        if translated.barriers && launch.barriers == 0 {
            return self.fall_back("compute barrier the launch does not declare".into());
        }
        let threads: u64 = launch.block.iter().map(|&n| u64::from(n)).product();
        let gpu = Self::gpu();
        if threads == 0
            || launch.grid.contains(&0)
            || launch.shared as usize > gpu.device.maxThreadgroupMemoryLength()
        {
            return self.fall_back("compute launch Metal cannot dispatch".into());
        }
        let key = Arc::as_ptr(dispatch.kernel) as usize;
        let pipeline = {
            let mut state = self.state.lock().unwrap();
            match state.kernels.get(&key).and_then(|e| e.pipeline.clone()) {
                Some(pipeline) => Ok(pipeline),
                None => {
                    let made = state
                        .library(gpu, translated.source())
                        .and_then(|library| {
                            library
                                .newFunctionWithName(&NSString::from_str("compute_main"))
                                .ok_or_else(|| Unmodeled("no compute function".into()))
                        })
                        .and_then(|function| {
                            gpu.device
                                .newComputePipelineStateWithFunction_error(&function)
                                .map_err(|e| Unmodeled(format!("compute pipeline: {e}")))
                        });
                    // A kernel Metal does not take stays the interpreter's.
                    if let Some(entry) = state.kernels.get_mut(&key) {
                        match &made {
                            Ok(pipeline) => entry.pipeline = Some(pipeline.clone()),
                            Err(Unmodeled(error)) => {
                                let first = error.lines().next().unwrap_or_default();
                                entry.translated = Err(format!("Metal: {first}"));
                            }
                        }
                    }
                    made
                }
            }
        };
        let pipeline = match pipeline {
            Ok(pipeline) => pipeline,
            Err(Unmodeled(error)) => {
                let first = error.lines().next().unwrap_or_default().to_string();
                return self.fall_back(format!("compute program: Metal: {first}"));
            }
        };
        if threads > pipeline.maxTotalThreadsPerThreadgroup() as u64 {
            return self.fall_back(format!("compute block of {threads} threads"));
        }
        let (constants, sizes) = match stage_constants(
            &translated.indirect,
            &translated.constants,
            dispatch.constants,
        ) {
            Ok(bound) => bound,
            Err(reason) => return self.fall_back(format!("compute launch: {reason}")),
        };
        // Textures: a slot that names none is the interpreter's refusal
        // when the kernel samples it, which only the interpreter tells.
        let mut bound = Vec::with_capacity(translated.textures.len());
        for (slot, &offset) in translated.textures.iter().enumerate() {
            let kind = translated.kinds[slot];
            let handle = dispatch
                .constants
                .get(dispatch.texture_buffer_index as usize)
                .and_then(|(data, _)| data.get(offset as usize..offset as usize + 4))
                .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("4 bytes")));
            let Some(handle) = handle else {
                return self.fall_back("compute texture handle outside its buffer".into());
            };
            let (sampled, tic, tsc) =
                match self.sampled(memory, writes, dispatch.textures, handle, kind, &[])? {
                    Ok(sampled) => sampled,
                    Err(_) => return self.fall_back("compute texture: a slot names none".into()),
                };
            if let Err(reason) = format::check_texture(&tic, &tsc, kind) {
                return self.fall_back(reason);
            }
            if matches!(sampled, Sampled::Composite(_)) && kind != composite_kind(&tic) {
                return self.fall_back(format!("rendered texture in a {kind:?} slot"));
            }
            bound.push((sampled, tic, tsc, kind));
        }
        let queries = query_words(
            dispatch.textures,
            dispatch.constants,
            dispatch.texture_buffer_index,
            &translated.queries,
        );
        if queries.chunks(5).any(|answer| answer[4] == 0) {
            return self.fall_back("compute texture query: a slot names none".into());
        }
        let [table] = self.global_tables(
            memory,
            writes,
            [(dispatch.constants, &translated.globals, translated.stores)],
        )?;

        let mut state = self.state.lock().unwrap();
        let mut textures = Vec::with_capacity(bound.len());
        for (sampled, tic, tsc, kind) in &bound {
            textures.push(state.gpu_texture(gpu, sampled, tic, tsc, *kind)?);
        }
        if state.arena.chunks_in_use() > ARENA_LIMIT {
            state.wait();
        }
        let command = state.command()?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| Unmodeled("Metal: no compute encoder".into()))?;
        encoder.setComputePipelineState(&pipeline);
        let mut words = queries;
        words.resize(words.len().max(4), 0);
        let globals = if table.is_empty() {
            None
        } else {
            let globals = state.globals.as_mut().expect("made for the table");
            globals.bound = true;
            Some(globals.buffer.clone())
        };
        let table = if table.is_empty() { vec![0; 4] } else { table };
        // SAFETY: indices are the function's bindings; Metal retains the
        // buffers and copies the bytes.
        unsafe {
            let mut set = |data: &[u8], index: usize| -> Result<(), Unmodeled> {
                let (buffer, offset) = state.arena.alloc(data)?;
                encoder.setBuffer_offset_atIndex(Some(&buffer), offset, index);
                Ok(())
            };
            set(bytes_of_u32(&words), msl::QUERY_BUFFER)?;
            for (slot, data) in constants.iter().enumerate() {
                if let Some(data) = data {
                    set(data, 2 + slot)?;
                }
            }
            set(bytes_of_u32(&sizes), 20)?;
            if let Some(globals) = &globals {
                encoder.setBuffer_offset_atIndex(Some(globals), 0, msl::GLOBAL_BUFFER);
                set(bytes_of_u32(&table), msl::GLOBAL_BUFFER + 1)?;
            }
            for (i, (texture, sampler)) in textures.iter().enumerate() {
                encoder.setTexture_atIndex(Some(texture), i);
                encoder.setSamplerState_atIndex(Some(sampler), i);
            }
            encoder.setThreadgroupMemoryLength_atIndex(
                (launch.shared as usize).next_multiple_of(16).max(16),
                0,
            );
        }
        let size = |v: [u32; 3]| MTLSize {
            width: v[0] as usize,
            height: v[1] as usize,
            depth: v[2] as usize,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(size(launch.grid), size(launch.block));
        encoder.endEncoding();
        state.draws += 1;
        Ok(true)
    }
}
