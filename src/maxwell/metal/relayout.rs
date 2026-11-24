//! A render target drawn to at memory where residents of another shape
//! lie (a guest reusing one memory pool for the transient targets of its
//! frame) takes their texels on the GPU: no write-back is waited for and
//! nothing is read back.
//!
//! Every texel of the new target is the same bytes of guest memory as a
//! texel of one of the residents there, where it lies inside one: the
//! kernel ([`RELAYOUT_SOURCE`]) finds the texel's byte offset in the new
//! surface's layout (block linear, of any block height, or pitch), the
//! resident's texel at the same address by the inverse of the resident's
//! layout, and copies its bits through views of both textures as unsigned
//! words (both of one texel size), from each resident newer than memory
//! (rendered to, or written back with the write-back not in memory yet;
//! memory holds what the others hold). The texels no such resident holds
//! are loaded from guest memory first, as any target's are; the
//! residents' dirty texels reach memory as a signal's write-backs do,
//! once the GPU has done the work so far
//! ([`Renderer::write_back_later`]), and the residents are dropped. The
//! new target is dirty whole: it holds what memory is to hold.

use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLDevice, MTLLibrary, MTLSize,
    MTLTexture,
};

use super::super::multisample::Samples;
use super::super::resources::Writes;
use super::super::surface::{Layout, Rect, Surface, Window};
use super::super::{GpuMemory, Unmodeled};
use super::{
    ComputePipeline, Content, Renderer, Resident, State, extent, new_target, overlap,
    raw_view_format, upload_surface,
};

/// A surface's layout as the kernel reads it: texels, bytes per texel,
/// the pitch (0: block linear), the width in GOBs and log2 of the block
/// height in GOBs.
#[repr(C)]
#[derive(Clone, Copy)]
struct LayoutWords {
    width: u32,
    height: u32,
    bpp: u32,
    pitch: u32,
    gobs: u32,
    log2: u32,
}

/// The kernel's words: the resident's layout, the new surface's, the new
/// surface's address minus the resident's, and the resident's size.
#[repr(C)]
#[derive(Clone, Copy)]
struct RelayoutWords {
    from: LayoutWords,
    to: LayoutWords,
    delta: [u32; 2],
    size: [u32; 2],
}

/// [`LayoutWords`] of `surface`; `None` for a block-linear one with blocks
/// deeper than one GOB.
fn layout_words(surface: &Surface) -> Option<LayoutWords> {
    let (pitch, log2) = match surface.layout {
        Layout::Pitch { pitch } => (pitch, 0),
        Layout::BlockLinear(block) if block.block_depth_log2 == 0 => (0, block.block_height_log2),
        Layout::BlockLinear(_) => return None,
    };
    Some(LayoutWords {
        width: surface.width,
        height: surface.height,
        bpp: surface.bytes_per_pixel,
        pitch,
        gobs: (surface.width * surface.bytes_per_pixel).div_ceil(64),
        log2,
    })
}

/// Texel (x, y) of a surface at its byte offset, and the inverse; a GOB
/// is 64 bytes by 8 rows, blocks one GOB wide and `1 << log2` high, row
/// after row of blocks (`surface::Window` lays them out alike).
pub(super) const RELAYOUT_SOURCE: &str = r#"#include <metal_stdlib>
using namespace metal;
struct L { uint width, height, bpp, pitch, gobs, log2; };
struct P { L from; L to; uint delta_lo, delta_hi, size_lo, size_hi; };
static inline ulong offset_of(constant L& l, uint x, uint y) {
    const uint column = x * l.bpp;
    if (l.pitch != 0u) { return ulong(y) * l.pitch + column; }
    const uint c = column & 63u, r = y & 7u;
    const uint gob_y = y >> 3;
    const ulong block = ulong(gob_y >> l.log2) * l.gobs + (column >> 6);
    const uint within = ((gob_y & ((1u << l.log2) - 1u)) << 9) | (((c & 0x20u) << 3) | ((r & 6u) << 5) | ((c & 0x10u) << 1) | ((r & 1u) << 4) | (c & 0xFu));
    return (block << (9u + l.log2)) + within;
}
static inline bool texel_at(constant L& l, ulong o, thread uint2& xy) {
    uint column, y;
    if (l.pitch != 0u) {
        y = uint(o / l.pitch); column = uint(o % l.pitch);
    } else {
        const ulong block = o >> (9u + l.log2);
        const uint within = uint(o & ((ulong(1) << (9u + l.log2)) - 1ul));
        const uint g = within & 511u;
        const uint c = (g & 0xFu) | ((g >> 1) & 0x10u) | ((g >> 3) & 0x20u);
        const uint r = ((g >> 4) & 1u) | ((g >> 5) & 6u);
        const uint gob_y = (uint(block / l.gobs) << l.log2) | (within >> 9);
        column = uint(block % l.gobs) * 64u + c;
        y = gob_y * 8u + r;
    }
    if (column % l.bpp != 0u) { return false; }
    xy = uint2(column / l.bpp, y);
    return xy.x < l.width && xy.y < l.height;
}
kernel void relayout(texture2d<uint, access::read> from [[texture(0)]], texture2d<uint, access::write> to [[texture(1)]], constant P& p [[buffer(0)]], uint2 g [[thread_position_in_grid]]) {
    if (g.x >= p.to.width || g.y >= p.to.height) { return; }
    const long delta = long((ulong(p.delta_hi) << 32) | ulong(p.delta_lo));
    const long a = long(offset_of(p.to, g.x, g.y)) + delta;
    const ulong size = (ulong(p.size_hi) << 32) | ulong(p.size_lo);
    if (a < 0 || ulong(a) >= size) { return; }
    uint2 xy;
    if (!texel_at(p.from, ulong(a), xy)) { return; }
    to.write(from.read(xy), g);
}
"#;

impl State {
    /// The relayout kernel, made once.
    fn relayout_pipeline(&mut self) -> Result<Retained<ComputePipeline>, Unmodeled> {
        if let Some(pipeline) = &self.relayout_pipeline {
            return Ok(pipeline.clone());
        }
        let gpu = Renderer::gpu();
        let library = self.library(gpu, RELAYOUT_SOURCE.to_string())?;
        let function = library
            .newFunctionWithName(&NSString::from_str("relayout"))
            .ok_or_else(|| Unmodeled("Metal: no relayout function".into()))?;
        let pipeline = gpu
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| Unmodeled(format!("Metal: relayout pipeline: {e}")))?;
        self.relayout_pipeline = Some(pipeline.clone());
        Ok(pipeline)
    }
}

impl Renderer {
    /// Makes `surface`, a color target of one sample per pixel drawn to
    /// (not replaced whole), resident from the residents of another shape
    /// at its memory, `stale` (see the module documentation). `false`
    /// when that cannot be (another texel size or layout, memory other
    /// work has yet to write there): nothing was changed then.
    pub(super) fn take_over(
        &self,
        memory: &dyn GpuMemory,
        writes: &Writes,
        surface: Surface,
        format: Content,
        stale: &[usize],
    ) -> Result<bool, Unmodeled> {
        use std::sync::atomic::Ordering;
        let target = extent(&surface);
        let (Some(to), Some(raw), Some(pixel_format)) = (
            layout_words(&surface),
            raw_view_format(surface.bytes_per_pixel),
            format.pixel_format(),
        ) else {
            return Ok(false);
        };
        // The residents newer than memory: rendered to, or written back
        // with the write-back not yet in memory. Memory holds what the
        // others hold, or what was written there since.
        let sources: Vec<(usize, Surface, LayoutWords)> = {
            let state = self.state.lock().unwrap();
            let mut sources = Vec::with_capacity(stale.len());
            for &index in stale {
                let r: &Resident = &state.residents[index];
                if r.dirty.is_none() && r.pending == 0 {
                    continue;
                }
                let Some(words) = layout_words(&r.surface) else {
                    return Ok(false);
                };
                if r.surface.bytes_per_pixel != surface.bytes_per_pixel {
                    return Ok(false);
                }
                sources.push((index, r.surface, words));
            }
            // Memory there other work has yet to write: global memory, or
            // write-backs of targets dropped before, unless a resident's
            // memory holds the whole target (every texel of it the kernel
            // does not take lies in the resident's padding then).
            let inside = sources.iter().any(|(_, s, _)| {
                let (start, end) = extent(s);
                start <= target.0 && target.1 <= end
            });
            let elsewhere = state.globals.iter().any(|globals| {
                globals.regions.iter().any(|(key, region)| {
                    (region.dirty || region.pending != 0)
                        && overlap(super::global_extent(*key), target)
                })
            }) || state.pending.iter().any(|pending| {
                (!inside
                    && pending.residents.iter().any(|resident| {
                        overlap(extent(&resident.surface), target)
                            && !sources.iter().any(|(_, s, _)| *s == resident.surface)
                    }))
                    || pending
                        .regions
                        .iter()
                        .any(|(key, _, _)| overlap(super::global_extent(*key), target))
                    || pending
                        .reports
                        .iter()
                        .any(|report| overlap(report.extent(), target))
            });
            if elsewhere {
                return Ok(false);
            }
            sources
        };
        if !self.write_back_later(memory, writes, stale)? {
            return Ok(false);
        }
        // A target held on demand whose memory has a new owner is
        // forgotten rather than written back (`write_back_later`): its
        // texture is not what memory is to hold, memory is.
        let sources: Vec<_> = {
            let state = self.state.lock().unwrap();
            sources
                .into_iter()
                .filter(|&(index, ..)| state.residents[index].pending != 0)
                .collect()
        };
        // The texels no resident holds, from memory as it is: what the
        // residents' write-backs are to change there the kernel takes
        // from them.
        let texels = Rect {
            x0: 0,
            y0: 0,
            x1: surface.width,
            y1: surface.height,
        };
        self.writing_back.store(true, Ordering::Relaxed);
        let window = Window::load(memory, surface, texels);
        self.writing_back.store(false, Ordering::Relaxed);
        let mut window = window?;
        let mut state = self.state.lock().unwrap();
        let full = Samples::SINGLE.pixels(&surface);
        let texture = new_target(Self::gpu(), pixel_format, full.x1, full.y1, 1)?;
        upload_surface(&texture, &mut window, full, [0, 0]);
        let pipeline = state.relayout_pipeline()?;
        let to_view = texture
            .newTextureViewWithPixelFormat(raw)
            .ok_or_else(|| Unmodeled("Metal: raw view".into()))?;
        let mut from_views = Vec::with_capacity(sources.len());
        for &(index, ..) in &sources {
            from_views.push(
                state.residents[index]
                    .texture
                    .newTextureViewWithPixelFormat(raw)
                    .ok_or_else(|| Unmodeled("Metal: raw view".into()))?,
            );
        }
        let command = state.command()?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| Unmodeled("Metal: no compute encoder".into()))?;
        encoder.setComputePipelineState(&pipeline);
        for ((_, from_surface, from), view) in sources.iter().zip(&from_views) {
            let (start, end) = extent(from_surface);
            let delta = surface.address.wrapping_sub(start);
            let size = end - start;
            let words = RelayoutWords {
                from: *from,
                to,
                delta: [delta as u32, (delta >> 32) as u32],
                size: [size as u32, (size >> 32) as u32],
            };
            // SAFETY: the kernel's bindings: both views and the words.
            unsafe {
                encoder.setTexture_atIndex(Some(view), 0);
                encoder.setTexture_atIndex(Some(&to_view), 1);
                encoder.setBytes_length_atIndex(
                    NonNull::from(&words).cast(),
                    std::mem::size_of::<RelayoutWords>(),
                    0,
                );
            }
            encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: surface.width as usize,
                    height: surface.height as usize,
                    depth: 1,
                },
                MTLSize {
                    width: 16,
                    height: 16,
                    depth: 1,
                },
            );
        }
        encoder.endEncoding();
        // The residents go; the new one holds what memory is to hold.
        let mut index = 0;
        state.residents.retain(|_| {
            index += 1;
            !stale.contains(&(index - 1))
        });
        state.versions += 1;
        let version = state.versions;
        state.residents.push(Resident {
            surface,
            format,
            log2: [0, 0],
            texture,
            mirror: window,
            dirty: Some(full),
            pending: 0,
            stamp: None,
            used: 0,
            version,
            guard: None,
        });
        let (start, end) = target;
        writes.record(start, end - start);
        self.dirtied(true);
        Ok(true)
    }
}
