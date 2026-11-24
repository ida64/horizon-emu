//! 2D blits on the GPU: texels copied from one surface the renderer holds
//! to another, with no look at the guest memory of either.
//!
//! The 2D class ([`super::super::two_d`]) hands [`Renderer::blit`] the
//! blits its software path does by moving bytes: every destination texel
//! takes one source texel of the same format, whose bits a read and a
//! write keep. The 2D class works out which source column each
//! destination column takes and which row each row, with the arithmetic
//! of its software path, so both paths take the same texels; the kernel
//! here ([`BLIT_SOURCE`]) only moves them, as unsigned words, through
//! views of the two textures (a resident's texture holds its surface's
//! bytes: `format::pixel_format`).
//!
//! The 2D class addresses texels. A multisampled surface holds its
//! pixels' samples as texels ([`super::super::multisample`]), and its
//! resident holds each sample in a layer; the kernel reads and writes
//! the layer and pixel of a texel as the packing kernel does
//! (`PACK_SOURCE`). A sample-grid reduction (the blit that resolves a
//! multisampled target, taking every second texel of a 2X1 target) so
//! reads one layer.
//!
//! The source must be resident already, and current with no look at
//! guest memory (rendered to since its last write-back, or known to
//! equal memory): any other surface stays with the software path, which
//! reads only the texels the blit takes. The destination becomes
//! resident and dirty, written back as rendered targets are (see the
//! module documentation of [`super`]: at the channel's next signal, or
//! on demand if it is held with samples): the blit is work of the
//! channel like a draw, and no one learns that it is done before the
//! channel signals. The blit does not write the source back: a
//! multisampled target that only its reductions read stays on the GPU.

use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLDevice, MTLLibrary,
    MTLPixelFormat, MTLSize, MTLTexture, MTLTextureType,
};

use super::super::resources::Writes;
use super::super::surface::{Rect, Surface};
use super::super::texture::format::ColorFormat;
use super::super::{GpuMemory, Unmodeled};
use super::{
    ComputePipeline, Content, MetalTexture, Renderer, State, bytes_of_u32, extent, overlap,
};

/// A blit that moves texels: texel (`rect.x0 + i`, `rect.y0 + j`) of
/// `destination` becomes texel (`columns[i]`, `rows[j]`) of `source`.
pub struct Blit<'a> {
    pub source: Surface,
    pub destination: Surface,
    /// The format of both surfaces.
    pub format: ColorFormat,
    /// The destination texels written, inside the surface.
    pub rect: Rect,
    /// The source column of each column of `rect`, inside the surface.
    pub columns: &'a [u32],
    /// The source row of each row of `rect`.
    pub rows: &'a [u32],
}

/// The words of the kernel ([`BLIT_SOURCE`] `BlitWords`): the destination
/// rectangle in texels, and log2 of the samples per pixel in x and y of
/// the source and of the destination.
#[repr(C)]
struct BlitWords {
    x0: u32,
    y0: u32,
    width: u32,
    height: u32,
    source_log2: [u32; 2],
    destination_log2: [u32; 2],
}

/// The blit kernel: one thread per destination texel. A texel of a
/// surface with samples is a sample of a pixel: the sample from the
/// texel's place in its pixel's block, as `multisample::Samples::cell`
/// places them, in the layer of that sample.
const BLIT_SOURCE: &str = r#"#include <metal_stdlib>
using namespace metal;
struct BlitWords { uint x0, y0, width, height, slx, sly, dlx, dly; };
static inline uint layer(uint x, uint y, uint lx, uint ly) {
    const uint cx = x & ((1u << lx) - 1u), cy = y & ((1u << ly) - 1u);
    return (cx & 1u) | ((cy & 1u) << 1) | ((cx & 2u) << 1) | ((cy & 2u) << 2);
}
kernel void blit(texture2d_array<uint, access::read> source [[texture(0)]], texture2d_array<uint, access::write> destination [[texture(1)]], const device uint* columns [[buffer(0)]], const device uint* rows [[buffer(1)]], constant BlitWords& p [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
    if (g.x >= p.width || g.y >= p.height) { return; }
    const uint sx = columns[g.x], sy = rows[g.y];
    const uint dx = p.x0 + g.x, dy = p.y0 + g.y;
    const uint4 texel = source.read(uint2(sx >> p.slx, sy >> p.sly), layer(sx, sy, p.slx, p.sly));
    destination.write(texel, uint2(dx >> p.dlx, dy >> p.dly), layer(dx, dy, p.dlx, p.dly));
}
"#;

/// The texture's layers as an array of unsigned words of `raw`'s size:
/// its texels' bits.
fn words(texture: &MetalTexture, raw: MTLPixelFormat) -> Result<Retained<MetalTexture>, Unmodeled> {
    let (levels, layers) = (NSRange::new(0, 1), NSRange::new(0, texture.arrayLength()));
    // SAFETY: the first level and every layer of the texture.
    unsafe {
        texture.newTextureViewWithPixelFormat_textureType_levels_slices(
            raw,
            MTLTextureType::Type2DArray,
            levels,
            layers,
        )
    }
    .ok_or_else(|| Unmodeled("Metal: no raw view of a blit surface".into()))
}

/// Whether a surface with `log2` samples per pixel holds whole pixels,
/// as its resident's layers do.
fn whole_pixels(surface: &Surface, [lx, ly]: [u32; 2]) -> bool {
    surface.width.trailing_zeros() >= lx && surface.height.trailing_zeros() >= ly
}

impl Renderer {
    /// A blit on the GPU; `Ok(false)` if it is left to the software path
    /// (nothing was changed then): the source is not held on the GPU as a
    /// color target or may differ from guest memory unrendered, or Metal
    /// has no texture of the format.
    ///
    /// A resident serves whatever color format it was rendered with:
    /// formats of one texel size at one surface name the same bytes.
    pub fn blit(
        &self,
        memory: &dyn GpuMemory,
        writes: &Writes,
        blit: &Blit,
    ) -> Result<bool, Unmodeled> {
        let (source, destination) = (blit.source, blit.destination);
        let raw = match source.bytes_per_pixel {
            1 => MTLPixelFormat::R8Uint,
            2 => MTLPixelFormat::R16Uint,
            4 => MTLPixelFormat::R32Uint,
            8 => MTLPixelFormat::RG32Uint,
            16 => MTLPixelFormat::RGBA32Uint,
            _ => return Ok(false),
        };
        let made = Content::Color(blit.format);
        if destination.bytes_per_pixel != source.bytes_per_pixel
            || made.pixel_format().is_none()
            || blit.rect.is_empty()
            || overlap(extent(&source), extent(&destination))
        {
            return Ok(false);
        }
        // The resident of a surface (format, samples) and whether it is
        // current with no look at memory.
        let held = |surface: Surface| {
            let state = self.state.lock().unwrap();
            state
                .residents
                .iter()
                .find(|r| r.surface == surface && matches!(r.format, Content::Color(_)))
                .map(|r| ((r.format, r.log2), r.current(writes)))
        };
        let Some((from, true)) = held(source) else {
            return Ok(false);
        };
        let to = held(destination).map_or((made, [0, 0]), |(to, _)| to);
        if !whole_pixels(&source, from.1) || !whole_pixels(&destination, to.1) {
            return Ok(false);
        }
        // Both current on the GPU; the destination with no look at memory
        // where the blit replaces all of it.
        let whole = Rect {
            x0: 0,
            y0: 0,
            x1: destination.width,
            y1: destination.height,
        };
        self.residents(memory, writes, &[(destination, to.0)], to.1, blit.rect == whole)?;
        self.residents(memory, writes, &[(source, from.0)], from.1, false)?;
        let mut state = self.state.lock().unwrap();
        let index = |state: &State, surface: Surface, (format, log2): (Content, [u32; 2])| {
            state
                .residents
                .iter()
                .position(|r| r.surface == surface && r.format == format && r.log2 == log2)
        };
        // Making one current may have dropped the other (`trim_residents`).
        let (Some(from_index), Some(to_index)) = (
            index(&state, source, from),
            index(&state, destination, to),
        ) else {
            return Ok(false);
        };
        let source_view = words(&state.residents[from_index].texture, raw)?;
        let destination_view = words(&state.residents[to_index].texture, raw)?;
        let pipeline = state.blit_pipeline()?;
        let (columns, columns_offset) = state.arena.alloc(bytes_of_u32(blit.columns))?;
        let (rows, rows_offset) = state.arena.alloc(bytes_of_u32(blit.rows))?;
        let rect = blit.rect;
        let words = BlitWords {
            x0: rect.x0,
            y0: rect.y0,
            width: rect.x1 - rect.x0,
            height: rect.y1 - rect.y0,
            source_log2: from.1,
            destination_log2: to.1,
        };
        debug_assert!(blit.columns.len() == words.width as usize);
        debug_assert!(blit.rows.len() == words.height as usize);
        let command = state.command()?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| Unmodeled("Metal: no compute encoder".into()))?;
        encoder.setComputePipelineState(&pipeline);
        // SAFETY: the kernel's bindings: the two views, a word per column
        // and per row of the grid, and the words.
        unsafe {
            encoder.setTexture_atIndex(Some(&source_view), 0);
            encoder.setTexture_atIndex(Some(&destination_view), 1);
            encoder.setBuffer_offset_atIndex(Some(&columns), columns_offset, 0);
            encoder.setBuffer_offset_atIndex(Some(&rows), rows_offset, 1);
            let bytes =
                NonNull::new((&words as *const BlitWords).cast_mut().cast()).expect("non-null");
            encoder.setBytes_length_atIndex(bytes, std::mem::size_of::<BlitWords>(), 2);
        }
        encoder.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: words.width as usize,
                height: words.height as usize,
                depth: 1,
            },
            MTLSize {
                width: 16,
                height: 16,
                depth: 1,
            },
        );
        encoder.endEncoding();
        // The pixels holding the texels written.
        let [lx, ly] = to.1;
        let pixels = Rect {
            x0: rect.x0 >> lx,
            y0: rect.y0 >> ly,
            x1: rect.x1.div_ceil(1 << lx),
            y1: rect.y1.div_ceil(1 << ly),
        };
        state.rendered(to_index, pixels, writes);
        self.any_dirty
            .store(true, std::sync::atomic::Ordering::Relaxed);
        Ok(true)
    }

    /// Makes a color surface resident, as a draw to it does.
    #[cfg(test)]
    pub(in super::super) fn hold(
        &self,
        memory: &dyn GpuMemory,
        writes: &Writes,
        (surface, format): (Surface, ColorFormat),
        log2: [u32; 2],
    ) -> Result<bool, Unmodeled> {
        let content = Content::Color(format);
        if content.pixel_format().is_none() || !whole_pixels(&surface, log2) {
            return Ok(false);
        }
        self.residents(memory, writes, &[(surface, content)], log2, false)?;
        Ok(true)
    }
}

impl State {
    /// The blit kernel (made once).
    fn blit_pipeline(&mut self) -> Result<Retained<ComputePipeline>, Unmodeled> {
        if let Some(pipeline) = &self.blit_pipeline {
            return Ok(pipeline.clone());
        }
        let gpu = Renderer::gpu();
        let library = self.library(gpu, BLIT_SOURCE.to_string())?;
        let function = library
            .newFunctionWithName(&NSString::from_str("blit"))
            .ok_or_else(|| Unmodeled("Metal: no blit function".into()))?;
        let pipeline = gpu
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| Unmodeled(format!("Metal: blit pipeline: {e}")))?;
        self.blit_pipeline = Some(pipeline.clone());
        Ok(pipeline)
    }
}
