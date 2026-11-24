//! The Fermi 2D class (0x902D): surface-to-surface blits with scaling,
//! filtering and format conversion ("pixels from memory").
//!
//! Methods and fields [public: NVIDIA open-gpu-doc
//! `classes/twod/cl902d.h`]. The blit is triggered by
//! `PIXELS_FROM_MEMORY_SRC_Y0_INT` (0x8DC), the last method of the block.
//! Destination pixel `i` samples
//! the source at `src_x0 + (i + 0.5) * du_dx` with `ORIGIN_CENTER` and at
//! `src_x0 + i * du_dx` with `ORIGIN_CORNER` (with CORNER the API adds
//! half a step itself); `src_x0`,
//! `du_dx` and their y counterparts are 32.32 fixed point.
//!
//! On a channel that renders with Metal, a blit that only moves texels'
//! bytes from a surface the renderer holds runs on the GPU
//! ([`super::metal::blit`]), from the columns and rows worked out here.

use std::sync::LazyLock;

use serde::{Deserialize, Serialize};

use super::metal::blit::Blit;
use super::surface::{Layout, Rect, Surface, Window};
use super::texture::format::{ColorFormat, f32_to_unorm, unorm_to_f32};
use super::texture::layout::BlockLinear;
use super::{GpuMemory, Hooks, Unmodeled};

/// Method registers of the class (below the macro methods).
const METHODS: usize = 0xE00;

const fn m(offset: u32) -> usize {
    (offset / 4) as usize
}

const SET_DST_FORMAT: u32 = 0x200;
const SET_SRC_FORMAT: u32 = 0x230;
const SET_CLIP_X0: usize = m(0x280);
const SET_CLIP_Y0: usize = m(0x284);
const SET_CLIP_WIDTH: usize = m(0x288);
const SET_CLIP_HEIGHT: usize = m(0x28c);
const SET_CLIP_ENABLE: usize = m(0x290);
const SET_COLOR_KEY_ENABLE: usize = m(0x29c);
const SET_OPERATION: usize = m(0x2ac);
const SET_DST_COLOR_RENDER_TO_ZETA_SURFACE: usize = m(0x2b8);
const SET_RENDER_ENABLE_C: usize = m(0x26c);
const SET_RENDER_ENABLE_OVERRIDE: usize = m(0x2dc);
const SET_PIXELS_FROM_MEMORY_SAMPLE_MODE: usize = m(0x88c);
const SET_PIXELS_FROM_MEMORY_DST_X0: usize = m(0x8b0);
const SET_PIXELS_FROM_MEMORY_DST_Y0: usize = m(0x8b4);
const SET_PIXELS_FROM_MEMORY_DST_WIDTH: usize = m(0x8b8);
const SET_PIXELS_FROM_MEMORY_DST_HEIGHT: usize = m(0x8bc);
const SET_PIXELS_FROM_MEMORY_DU_DX_FRAC: usize = m(0x8c0);
const SET_PIXELS_FROM_MEMORY_DV_DY_FRAC: usize = m(0x8c8);
const SET_PIXELS_FROM_MEMORY_SRC_X0_FRAC: usize = m(0x8d0);
const SET_PIXELS_FROM_MEMORY_SRC_Y0_FRAC: usize = m(0x8d8);
const PIXELS_FROM_MEMORY_SRC_Y0_INT: usize = m(0x8dc);

/// The 2D engine of a channel; saved whole in machine snapshots.
#[derive(Serialize, Deserialize)]
pub struct TwoD {
    registers: Vec<u32>,
    /// Render enable latched by `SET_RENDER_ENABLE_C`, as in the 3D class.
    render_enable: bool,
}

impl Default for TwoD {
    fn default() -> Self {
        Self {
            registers: vec![0; METHODS],
            render_enable: true,
        }
    }
}

fn refuse(method: usize, argument: u32, what: &str) -> Unmodeled {
    Unmodeled(format!(
        "2D (0x902D): {what}: method {:#06x} = {argument:#x}",
        method * 4
    ))
}

impl TwoD {
    /// Whether a restored register file has the size methods index.
    pub(super) fn fits(&self) -> bool {
        self.registers.len() == METHODS
    }

    /// Sets a register's initial value (netlist SW_METHOD_INIT).
    pub fn set_initial(&mut self, method: u32, value: u32) -> Result<(), Unmodeled> {
        let slot = self.registers.get_mut(method as usize).ok_or_else(|| {
            refuse(
                method as usize,
                value,
                "initial state outside the method registers",
            )
        })?;
        *slot = value;
        Ok(())
    }

    pub fn method(
        &mut self,
        method: u32,
        argument: u32,
        memory: &dyn GpuMemory,
        hooks: &Hooks,
    ) -> Result<(), Unmodeled> {
        let index = method as usize;
        if index >= METHODS {
            return Err(refuse(index, argument, "macro call on the 2D class"));
        }
        self.registers[index] = argument;
        match index {
            // NO_OPERATION, WAIT_FOR_IDLE, FLUSH_AND_INVALIDATE_ROP_MINI_CACHE,
            // TWOD_INVALIDATE_TEXTURE_DATA_CACHE: execution is synchronous
            // and nothing is cached.
            0x40 | 0x44 | 0x8a | 0x90 => Ok(()),
            // NOTIFY, MME loads, solid primitives, pixels from CPU.
            0x43 | 0x45..=0x48 | 0x160 | 0x178..=0x19f | 0x218 => {
                Err(refuse(index, argument, "action not modeled"))
            }
            SET_RENDER_ENABLE_C => {
                self.render_enable = match argument & 7 {
                    0 => false,
                    1 => true,
                    _ => return Err(refuse(index, argument, "conditional rendering")),
                };
                Ok(())
            }
            PIXELS_FROM_MEMORY_SRC_Y0_INT => self.blit(memory, hooks),
            _ => Ok(()),
        }
    }

    fn surface(&self, base: u32) -> Result<(Surface, ColorFormat), Unmodeled> {
        let r = |offset: u32| self.registers[m(base + offset)];
        let raw = r(0x00);
        // The class has no integer formats [public: NVIDIA open-gpu-doc
        // `classes/twod/cl902d.h` `NV902D_SET_DST_FORMAT_V_*`].
        let format = ColorFormat::from_raw(raw)
            .filter(|format| !format.is_integer())
            .ok_or_else(|| refuse(m(base), raw, "surface format"))?;
        let bpp = format.bytes_per_pixel();
        // SET_*_MEMORY_LAYOUT: 0 block linear, 1 pitch.
        let layout = if r(0x04) & 1 != 0 {
            Layout::Pitch { pitch: r(0x14) }
        } else {
            let block = r(0x08);
            Layout::BlockLinear(BlockLinear {
                block_height_log2: (block >> 4) & 7,
                block_depth_log2: (block >> 8) & 7,
            })
        };
        // DEPTH (0x0c), LAYER (0x10, destination only): single 2D slices.
        if r(0x0c) > 1 {
            return Err(refuse(m(base + 0x0c), r(0x0c), "surface depth"));
        }
        if base == SET_DST_FORMAT && r(0x10) != 0 {
            return Err(refuse(m(base + 0x10), r(0x10), "surface layer"));
        }
        Ok((
            Surface {
                address: u64::from(r(0x20) & 0xFF) << 32 | u64::from(r(0x24)),
                width: r(0x18),
                height: r(0x1c),
                bytes_per_pixel: bpp,
                layout,
            },
            format,
        ))
    }

    fn fixed(&self, frac: usize) -> i64 {
        (i64::from(self.registers[frac + 1] as i32) << 32) | i64::from(self.registers[frac])
    }

    fn blit(&mut self, memory: &dyn GpuMemory, hooks: &Hooks) -> Result<(), Unmodeled> {
        let r = &self.registers;
        let enabled = match r[SET_RENDER_ENABLE_OVERRIDE] & 3 {
            1 => true,
            2 => false,
            _ => self.render_enable,
        };
        if !enabled {
            return Ok(());
        }
        // SET_OPERATION: SRCCOPY (3) only.
        if r[SET_OPERATION] & 7 != 3 {
            return Err(refuse(SET_OPERATION, r[SET_OPERATION], "blit operation"));
        }
        if r[SET_COLOR_KEY_ENABLE] & 1 != 0 {
            return Err(refuse(SET_COLOR_KEY_ENABLE, 1, "color key"));
        }
        if r[SET_DST_COLOR_RENDER_TO_ZETA_SURFACE] & 1 != 0 {
            return Err(refuse(
                SET_DST_COLOR_RENDER_TO_ZETA_SURFACE,
                1,
                "render to depth surface",
            ));
        }
        let (source, source_format) = self.surface(SET_SRC_FORMAT)?;
        let (destination, destination_format) = self.surface(SET_DST_FORMAT)?;
        if source_format.is_srgb() != destination_format.is_srgb() {
            return Err(refuse(
                m(SET_SRC_FORMAT),
                r[m(SET_SRC_FORMAT)],
                "blit between sRGB and linear formats",
            ));
        }
        let mode = r[SET_PIXELS_FROM_MEMORY_SAMPLE_MODE];
        let corner = mode & 1 != 0;
        let bilinear = mode & 0x10 != 0;
        let x0 = r[SET_PIXELS_FROM_MEMORY_DST_X0];
        let y0 = r[SET_PIXELS_FROM_MEMORY_DST_Y0];
        let mut rect = Rect {
            x0,
            y0,
            x1: x0.saturating_add(r[SET_PIXELS_FROM_MEMORY_DST_WIDTH]),
            y1: y0.saturating_add(r[SET_PIXELS_FROM_MEMORY_DST_HEIGHT]),
        };
        if r[SET_CLIP_ENABLE] & 1 != 0 {
            let cx = r[SET_CLIP_X0];
            let cy = r[SET_CLIP_Y0];
            rect = rect.intersect(&Rect {
                x0: cx,
                y0: cy,
                x1: cx.saturating_add(r[SET_CLIP_WIDTH]),
                y1: cy.saturating_add(r[SET_CLIP_HEIGHT]),
            });
        }
        rect = rect.intersect(&Rect {
            x0: 0,
            y0: 0,
            x1: destination.width,
            y1: destination.height,
        });
        if rect.is_empty() {
            return Ok(());
        }
        let to_f64 = |fixed: i64| fixed as f64 / 4294967296.0;
        let du = to_f64(self.fixed(SET_PIXELS_FROM_MEMORY_DU_DX_FRAC));
        let dv = to_f64(self.fixed(SET_PIXELS_FROM_MEMORY_DV_DY_FRAC));
        let u0 = to_f64(self.fixed(SET_PIXELS_FROM_MEMORY_SRC_X0_FRAC));
        let v0 = to_f64(self.fixed(SET_PIXELS_FROM_MEMORY_SRC_Y0_FRAC));
        let offset = if corner { 0.0 } else { 0.5 };
        let position = |i: u32, start: u32, origin: f64, step: f64| {
            origin + (f64::from(i - start) + offset) * step
        };
        // Source rows and columns the blit can touch.
        let span = |a: f64, b: f64, limit: u32| {
            let low = (a.min(b) - 1.0).floor().max(0.0) as u32;
            let high = ((a.max(b) + 1.0).ceil().max(0.0) as u32).min(limit);
            (low, high)
        };
        let (sx0, sx1) = span(
            position(rect.x0, x0, u0, du),
            position(rect.x1, x0, u0, du),
            source.width,
        );
        let (sy0, sy1) = span(
            position(rect.y0, y0, v0, dv),
            position(rect.y1, y0, v0, dv),
            source.height,
        );
        if sx0 >= sx1 || sy0 >= sy1 {
            return Err(refuse(
                PIXELS_FROM_MEMORY_SRC_Y0_INT,
                0,
                "blit source outside the source surface",
            ));
        }
        let loaded = Rect {
            x0: sx0,
            y0: sy0,
            x1: sx1,
            y1: sy1,
        };
        let taps = |range: std::ops::Range<u32>, start: u32, origin: f64, step: f64, low, high| {
            range
                .map(|i| Tap::new(position(i, start, origin, step), bilinear, low, high))
                .collect()
        };
        let sampling = Sampling {
            columns: taps(rect.x0..rect.x1, x0, u0, du, sx0, sx1),
            rows: taps(rect.y0..rect.y1, y0, v0, dv, sy0, sy1),
            bilinear,
            span_width: (sx1 - sx0) as usize,
        };
        // A blit that moves bytes, from a surface Metal holds: on the GPU,
        // with no look at either surface's memory.
        if let Some(metal) = &hooks.metal
            && sampling.moves_bytes(source_format, destination_format)
        {
            let texels = |taps: &[Tap], low: u32| -> Vec<u32> {
                taps.iter().map(|tap| low + tap.first as u32).collect()
            };
            let blit = Blit {
                source,
                destination,
                format: source_format,
                rect,
                columns: &texels(&sampling.columns, sx0),
                rows: &texels(&sampling.rows, sy0),
            };
            if metal.blit(memory, &hooks.writes, &blit)? {
                return Ok(());
            }
        }
        // The loaded source texels, row-major, as stored.
        let source_bpp = source_format.bytes_per_pixel() as usize;
        let mut texels = vec![0u8; (sx1 - sx0) as usize * (sy1 - sy0) as usize * source_bpp];
        Window::read_rect(memory, source, loaded, &mut texels)?;
        let destination_bpp = destination_format.bytes_per_pixel() as usize;
        let mut pixels = vec![0u8; sampling.columns.len() * sampling.rows.len() * destination_bpp];
        resample(
            &texels,
            source_format,
            destination_format,
            &sampling,
            &mut pixels,
        );
        // Every pixel of `rect` is written: what it covers whole is not read.
        let mut destination_window = Window::load_to_overwrite(memory, destination, rect)?;
        destination_window.write_pixels(rect, &pixels);
        destination_window.store(memory)
    }
}

/// Where one destination column (or row) samples the loaded source: the
/// texel at or before the sample point and the one after it, each clamped
/// to the loaded texels and counted from their first, and the weight of the
/// second. Nearest sampling takes `first` alone.
#[derive(Clone, Copy, Debug)]
struct Tap {
    first: usize,
    second: usize,
    weight: f32,
}

impl Tap {
    /// The texels around source `position` (pixel centres at `n + 0.5`),
    /// the loaded ones being `low..high`.
    fn new(position: f64, bilinear: bool, low: u32, high: u32) -> Tap {
        let clamp =
            |i: i64| (i.clamp(i64::from(low), i64::from(high) - 1) - i64::from(low)) as usize;
        if bilinear {
            let below = position - 0.5;
            let first = below.floor();
            let weight = (below - first) as f32;
            let first = first as i64;
            Tap {
                first: clamp(first),
                second: clamp(first + 1),
                weight,
            }
        } else {
            let first = clamp(position.floor() as i64);
            Tap {
                first,
                second: first,
                weight: 0.0,
            }
        }
    }
}

/// The taps of every destination column and row of a blit.
struct Sampling {
    columns: Vec<Tap>,
    rows: Vec<Tap>,
    bilinear: bool,
    /// Texels in a row of the loaded source.
    span_width: usize,
}

impl Sampling {
    /// Whether every pixel is one texel's decoded value: nearest sampling,
    /// or a bilinear filter whose weights are all 0 on a source without
    /// floats (see [`resample`]).
    fn point(&self, from: ColorFormat) -> bool {
        let unweighted = |taps: &[Tap]| taps.iter().all(|tap| tap.weight == 0.0);
        !self.bilinear || (!is_float(from) && unweighted(&self.columns) && unweighted(&self.rows))
    }

    /// Whether the blit from `from` to `to` copies texels' bytes: point
    /// samples between texels of one format that decodes and encodes to
    /// the same bits (see [`keeps_bits`]).
    fn moves_bytes(&self, from: ColorFormat, to: ColorFormat) -> bool {
        from == to && keeps_bits(from) && self.point(from)
    }
}

/// Fills `pixels` (the destination rectangle, row-major, in the destination
/// format) from `texels` (the loaded source, row-major, in the source
/// format). Bilinear filtering interpolates in `f32` between the decoded
/// texels, first along x, then y. A filter whose weights are all 0 takes the
/// first texel's decoded value as it is, the interpolation adding exact
/// zeros, which holds for every finite value: so for the UNORM and SNORM
/// sources, not for float ones (an infinity or NaN beside it turns the sum
/// into NaN). A copy between texels of one format that decodes and encodes
/// to the same bits (see [`keeps_bits`]) moves the bytes themselves.
fn resample(
    texels: &[u8],
    from: ColorFormat,
    to: ColorFormat,
    sampling: &Sampling,
    pixels: &mut [u8],
) {
    let point = sampling.point(from);
    if sampling.moves_bytes(from, to) {
        match from.bytes_per_pixel() {
            1 => gather::<1>(texels, sampling, pixels),
            2 => gather::<2>(texels, sampling, pixels),
            4 => gather::<4>(texels, sampling, pixels),
            8 => gather::<8>(texels, sampling, pixels),
            _ => gather::<16>(texels, sampling, pixels),
        }
        return;
    }
    let (decode, encode) = (Codec::new(from), Codec::new(to));
    let source_bpp = from.bytes_per_pixel() as usize;
    let destination_bpp = to.bytes_per_pixel() as usize;
    let row_bytes = sampling.columns.len() * destination_bpp;
    let span_bytes = sampling.span_width * source_bpp;
    if point {
        in_rows(pixels, row_bytes, |first, band| {
            for (row, out) in sampling.rows[first..]
                .iter()
                .zip(band.chunks_exact_mut(row_bytes))
            {
                let line = &texels[row.first * span_bytes..][..span_bytes];
                let columns = sampling.columns.iter();
                for (column, out) in columns.zip(out.chunks_exact_mut(destination_bpp)) {
                    let texel = &line[column.first * source_bpp..][..source_bpp];
                    encode.write(decode.read(texel), out);
                }
            }
        });
        return;
    }
    let span_width = sampling.span_width;
    let mut decoded = vec![[0f32; 4]; texels.len() / source_bpp];
    in_rows(&mut decoded, span_width, |first, band| {
        let texels = &texels[first * span_bytes..];
        for (value, texel) in band.iter_mut().zip(texels.chunks_exact(source_bpp)) {
            *value = decode.read(texel);
        }
    });
    in_rows(pixels, row_bytes, |first, band| {
        for (row, out) in sampling.rows[first..]
            .iter()
            .zip(band.chunks_exact_mut(row_bytes))
        {
            let top = &decoded[row.first * span_width..][..span_width];
            let bottom = &decoded[row.second * span_width..][..span_width];
            let columns = sampling.columns.iter();
            for (column, out) in columns.zip(out.chunks_exact_mut(destination_bpp)) {
                let (a, b) = (top[column.first], top[column.second]);
                let (c, d) = (bottom[column.first], bottom[column.second]);
                let (wu, wv) = (column.weight, row.weight);
                let value = std::array::from_fn(|k| {
                    let top = a[k] + (b[k] - a[k]) * wu;
                    let bottom = c[k] + (d[k] - c[k]) * wu;
                    top + (bottom - top) * wv
                });
                encode.write(value, out);
            }
        }
    });
}

/// [`resample`] of point samples that keep their bits: texels of `N` bytes
/// copied, a row at once where its columns are consecutive texels.
fn gather<const N: usize>(texels: &[u8], sampling: &Sampling, pixels: &mut [u8]) {
    let columns = &sampling.columns;
    let row_bytes = columns.len() * N;
    let span_bytes = sampling.span_width * N;
    let consecutive = columns
        .windows(2)
        .all(|pair| pair[1].first == pair[0].first + 1);
    in_rows(pixels, row_bytes, |first, band| {
        for (row, out) in sampling.rows[first..]
            .iter()
            .zip(band.chunks_exact_mut(row_bytes))
        {
            let line = &texels[row.first * span_bytes..][..span_bytes];
            if consecutive {
                out.copy_from_slice(&line[columns[0].first * N..][..row_bytes]);
                continue;
            }
            for (column, out) in columns.iter().zip(out.chunks_exact_mut(N)) {
                let texel: &[u8; N] = line[column.first * N..][..N].try_into().unwrap();
                *<&mut [u8; N]>::try_from(out).unwrap() = *texel;
            }
        }
    });
}

/// Whether the format stores floats (whose infinities and NaNs make
/// bilinear filtering differ from its first texel even at weight 0).
fn is_float(format: ColorFormat) -> bool {
    use ColorFormat::*;
    matches!(
        format,
        Rgba32Float
            | Rgbx32Float
            | Rgba16Float
            | Rg32Float
            | Rgbx16Float
            | Rg16Float
            | R11G11B10Float
            | R32Float
            | R16Float
    )
}

/// Whether [`ColorFormat::write`] of [`ColorFormat::read`] gives back
/// every texel's own bits: the UNORM formats whose fields cover the whole
/// texel (no `X` bits, which a write fills; SNORM reads both -128 and -127
/// as -1.0). A UNORM code `k` of `n` bits reads as the nearest `f32` to
/// `k / (2^n - 1)` and writes back as `⌊v · (2^n - 1) + ½⌋`, which is `k`
/// again for `n` up to 16 (checked by `formats_that_keep_bits_round_trip`).
fn keeps_bits(format: ColorFormat) -> bool {
    use ColorFormat::*;
    matches!(
        format,
        Rgba16Unorm
            | Bgra8Unorm
            | Bgra8Srgb
            | Rgb10A2Unorm
            | Rgba8Unorm
            | Rgba8Srgb
            | Rg16Unorm
            | Bgr10A2Unorm
            | B5G6R5Unorm
            | Bgr5A1Unorm
            | Rg8Unorm
            | R16Unorm
            | R8Unorm
            | A8Unorm
    )
}

/// What the fourth byte of a [`Codec::Unorm8`] texel is.
#[derive(Clone, Copy, Debug)]
enum Fourth {
    Alpha,
    /// An `X8` byte: read as alpha 1.0, written from alpha.
    FromAlpha,
    /// A `Z8` byte, written as 0.
    Zeros,
    /// An `O8` byte, written as 0xFF.
    Ones,
}

/// [`ColorFormat::read`] and [`ColorFormat::write`], done directly for the
/// 32-bit formats of four 8-bit UNORM fields (R, G, B at the bytes of
/// `rgb`, then the fourth byte) with a table of the 256 codes' values and
/// the same conversion functions, and through the format for the others.
#[derive(Clone, Copy, Debug)]
enum Codec {
    Unorm8 { rgb: [usize; 3], fourth: Fourth },
    Other(ColorFormat),
}

/// The 8-bit UNORM codes' values.
static UNORM8: LazyLock<[f32; 256]> =
    LazyLock::new(|| std::array::from_fn(|code| unorm_to_f32(code as u32, 8)));

impl Codec {
    fn new(format: ColorFormat) -> Codec {
        use ColorFormat::*;
        const RGB: [usize; 3] = [0, 1, 2];
        const BGR: [usize; 3] = [2, 1, 0];
        let (rgb, fourth) = match format {
            Bgra8Unorm | Bgra8Srgb => (BGR, Fourth::Alpha),
            Rgba8Unorm | Rgba8Srgb => (RGB, Fourth::Alpha),
            Bgrx8Unorm | Bgrx8Srgb => (BGR, Fourth::FromAlpha),
            Rgbx8Unorm | Rgbx8Srgb => (RGB, Fourth::FromAlpha),
            Bgrz8Unorm => (BGR, Fourth::Zeros),
            Bgro8Unorm => (BGR, Fourth::Ones),
            _ => return Codec::Other(format),
        };
        Codec::Unorm8 { rgb, fourth }
    }

    #[inline]
    fn read(self, texel: &[u8]) -> [f32; 4] {
        match self {
            Codec::Unorm8 { rgb, fourth } => {
                let table = &*UNORM8;
                let alpha = match fourth {
                    Fourth::Alpha => table[usize::from(texel[3])],
                    _ => 1.0,
                };
                let [r, g, b] = rgb.map(|at| table[usize::from(texel[at])]);
                [r, g, b, alpha]
            }
            Codec::Other(format) => format.read(texel),
        }
    }

    #[inline]
    fn write(self, value: [f32; 4], texel: &mut [u8]) {
        match self {
            Codec::Unorm8 { rgb, fourth } => {
                for (channel, at) in rgb.into_iter().enumerate() {
                    texel[at] = f32_to_unorm(value[channel], 8) as u8;
                }
                texel[3] = match fourth {
                    Fourth::Alpha | Fourth::FromAlpha => f32_to_unorm(value[3], 8) as u8,
                    Fourth::Zeros => 0,
                    Fourth::Ones => 0xFF,
                };
            }
            Codec::Other(format) => format.write(value, texel),
        }
    }
}

/// Blits of fewer pixels than this (destination pixels, or decoded source
/// texels) run on the calling thread; larger ones are split among host
/// threads.
const BANDED_PIXELS: usize = 1 << 16;
/// The most host threads one blit runs on.
const BAND_THREADS: usize = 8;

/// Runs `fill(first, band)` over `items`, rows of `width` items, in bands of
/// whole rows, `first` being the band's first row: on up to
/// [`BAND_THREADS`] host threads at once for [`BANDED_PIXELS`] items or
/// more. Each row is computed alone, so the split does not change it.
fn in_rows<T: Send>(items: &mut [T], width: usize, fill: impl Fn(usize, &mut [T]) + Sync) {
    let width = width.max(1);
    let rows = items.len().div_ceil(width);
    let threads = if items.len() < BANDED_PIXELS {
        1
    } else {
        std::thread::available_parallelism().map_or(1, |n| n.get().min(BAND_THREADS))
    };
    if threads <= 1 || rows <= 1 {
        return fill(0, items);
    }
    let band = rows.div_ceil(threads);
    std::thread::scope(|scope| {
        for (index, items) in items.chunks_mut(band * width).enumerate() {
            let fill = &fill;
            // A helper of the channel's thread (see `pipeline::rasterize`).
            scope.spawn(move || {
                let _exempt = crate::memory::exempt();
                fill(index * band, items)
            });
        }
    });
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    /// Guest memory from address 0.
    struct Flat(Mutex<Vec<u8>>);

    impl GpuMemory for Flat {
        fn read(&self, address: u64, buffer: &mut [u8]) -> Option<()> {
            let bytes = self.0.lock().unwrap();
            let start = usize::try_from(address).ok()?;
            buffer.copy_from_slice(bytes.get(start..start.checked_add(buffer.len())?)?);
            Some(())
        }

        fn write(&self, address: u64, data: &[u8]) -> Option<()> {
            let mut bytes = self.0.lock().unwrap();
            let start = usize::try_from(address).ok()?;
            bytes
                .get_mut(start..start.checked_add(data.len())?)?
                .copy_from_slice(data);
            Some(())
        }
    }

    /// xorshift64*.
    struct Random(u64);

    impl Random {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }

        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }

        fn pick<T: Copy>(&mut self, items: &[T]) -> T {
            items[self.below(items.len() as u64) as usize]
        }

        fn bytes(&mut self, count: usize) -> Vec<u8> {
            (0..count).map(|_| self.next() as u8).collect()
        }
    }

    /// The `FORMAT` codes the class has among those
    /// [`ColorFormat::from_raw`] accepts.
    fn format_codes() -> Vec<u32> {
        (0xc0..=0xff)
            .filter(|&raw| ColorFormat::from_raw(raw).is_some_and(|format| !format.is_integer()))
            .collect()
    }

    /// The blit as it was before [`resample`]: every destination pixel
    /// sampled and converted on its own, through [`ColorFormat`].
    fn reference_blit(two_d: &TwoD, memory: &dyn GpuMemory) -> Result<(), Unmodeled> {
        let r = &two_d.registers;
        let enabled = match r[SET_RENDER_ENABLE_OVERRIDE] & 3 {
            1 => true,
            2 => false,
            _ => two_d.render_enable,
        };
        if !enabled {
            return Ok(());
        }
        if r[SET_OPERATION] & 7 != 3 {
            return Err(refuse(SET_OPERATION, r[SET_OPERATION], "blit operation"));
        }
        if r[SET_COLOR_KEY_ENABLE] & 1 != 0 {
            return Err(refuse(SET_COLOR_KEY_ENABLE, 1, "color key"));
        }
        if r[SET_DST_COLOR_RENDER_TO_ZETA_SURFACE] & 1 != 0 {
            return Err(refuse(
                SET_DST_COLOR_RENDER_TO_ZETA_SURFACE,
                1,
                "render to depth surface",
            ));
        }
        let (source, source_format) = two_d.surface(SET_SRC_FORMAT)?;
        let (destination, destination_format) = two_d.surface(SET_DST_FORMAT)?;
        if source_format.is_srgb() != destination_format.is_srgb() {
            return Err(refuse(
                m(SET_SRC_FORMAT),
                r[m(SET_SRC_FORMAT)],
                "blit between sRGB and linear formats",
            ));
        }
        let mode = r[SET_PIXELS_FROM_MEMORY_SAMPLE_MODE];
        let corner = mode & 1 != 0;
        let bilinear = mode & 0x10 != 0;
        let x0 = r[SET_PIXELS_FROM_MEMORY_DST_X0];
        let y0 = r[SET_PIXELS_FROM_MEMORY_DST_Y0];
        let mut rect = Rect {
            x0,
            y0,
            x1: x0.saturating_add(r[SET_PIXELS_FROM_MEMORY_DST_WIDTH]),
            y1: y0.saturating_add(r[SET_PIXELS_FROM_MEMORY_DST_HEIGHT]),
        };
        if r[SET_CLIP_ENABLE] & 1 != 0 {
            let cx = r[SET_CLIP_X0];
            let cy = r[SET_CLIP_Y0];
            rect = rect.intersect(&Rect {
                x0: cx,
                y0: cy,
                x1: cx.saturating_add(r[SET_CLIP_WIDTH]),
                y1: cy.saturating_add(r[SET_CLIP_HEIGHT]),
            });
        }
        rect = rect.intersect(&Rect {
            x0: 0,
            y0: 0,
            x1: destination.width,
            y1: destination.height,
        });
        if rect.is_empty() {
            return Ok(());
        }
        let to_f64 = |fixed: i64| fixed as f64 / 4294967296.0;
        let du = to_f64(two_d.fixed(SET_PIXELS_FROM_MEMORY_DU_DX_FRAC));
        let dv = to_f64(two_d.fixed(SET_PIXELS_FROM_MEMORY_DV_DY_FRAC));
        let u0 = to_f64(two_d.fixed(SET_PIXELS_FROM_MEMORY_SRC_X0_FRAC));
        let v0 = to_f64(two_d.fixed(SET_PIXELS_FROM_MEMORY_SRC_Y0_FRAC));
        let offset = if corner { 0.0 } else { 0.5 };
        let position = |i: u32, start: u32, origin: f64, step: f64| {
            origin + (f64::from(i - start) + offset) * step
        };
        let span = |a: f64, b: f64, limit: u32| {
            let low = (a.min(b) - 1.0).floor().max(0.0) as u32;
            let high = ((a.max(b) + 1.0).ceil().max(0.0) as u32).min(limit);
            (low, high)
        };
        let (sx0, sx1) = span(
            position(rect.x0, x0, u0, du),
            position(rect.x1, x0, u0, du),
            source.width,
        );
        let (sy0, sy1) = span(
            position(rect.y0, y0, v0, dv),
            position(rect.y1, y0, v0, dv),
            source.height,
        );
        if sx0 >= sx1 || sy0 >= sy1 {
            return Err(refuse(
                PIXELS_FROM_MEMORY_SRC_Y0_INT,
                0,
                "blit source outside the source surface",
            ));
        }
        let mut source_window = Window::load(
            memory,
            source,
            Rect {
                x0: sx0,
                y0: sy0,
                x1: sx1,
                y1: sy1,
            },
        )?;
        let width = (sx1 - sx0) as usize;
        let mut texels = vec![[0f32; 4]; width * (sy1 - sy0) as usize];
        for band in source_window.bands_mut() {
            for y in band.y0.max(sy0)..(band.y0 + 8).min(sy1) {
                for x in sx0..sx1 {
                    texels[(y - sy0) as usize * width + (x - sx0) as usize] =
                        source_format.read(band.pixel(x, y));
                }
            }
        }
        let texel = |x: i64, y: i64| {
            let x = x.clamp(i64::from(sx0), i64::from(sx1) - 1) - i64::from(sx0);
            let y = y.clamp(i64::from(sy0), i64::from(sy1) - 1) - i64::from(sy0);
            texels[y as usize * width + x as usize]
        };
        let mut destination_window = Window::load(memory, destination, rect)?;
        for mut band in destination_window.bands_mut() {
            for y in band.y0.max(rect.y0)..(band.y0 + 8).min(rect.y1) {
                let v = position(y, y0, v0, dv);
                for x in rect.x0..rect.x1 {
                    let u = position(x, x0, u0, du);
                    let value = if bilinear {
                        let fu = u - 0.5;
                        let fv = v - 0.5;
                        let (iu, iv) = (fu.floor(), fv.floor());
                        let (wu, wv) = ((fu - iu) as f32, (fv - iv) as f32);
                        let (iu, iv) = (iu as i64, iv as i64);
                        let a = texel(iu, iv);
                        let b = texel(iu + 1, iv);
                        let c = texel(iu, iv + 1);
                        let d = texel(iu + 1, iv + 1);
                        std::array::from_fn(|k| {
                            let top = a[k] + (b[k] - a[k]) * wu;
                            let bottom = c[k] + (d[k] - c[k]) * wu;
                            top + (bottom - top) * wv
                        })
                    } else {
                        texel(u.floor() as i64, v.floor() as i64)
                    };
                    destination_format.write(value, band.pixel_mut(x, y));
                }
            }
        }
        destination_window.store(memory)
    }

    const MEMORY: usize = 2 << 20;
    const SOURCE: u32 = 0;
    const DESTINATION: u32 = 1 << 20;

    /// Programs a surface at `address` of a random format among `codes`,
    /// layout and size; returns its format.
    fn surface(two_d: &mut TwoD, random: &mut Random, base: u32, address: u32, codes: &[u32]) {
        let code = random.pick(codes);
        let format = ColorFormat::from_raw(code).unwrap();
        let width = 1 + random.below(64) as u32;
        let height = 1 + random.below(40) as u32;
        let set = |two_d: &mut TwoD, offset: u32, value: u32| {
            two_d.registers[m(base + offset)] = value;
        };
        set(two_d, 0x00, code);
        let pitch = random.below(2) == 0;
        set(two_d, 0x04, u32::from(pitch));
        set(two_d, 0x08, (random.below(5) as u32) << 4);
        set(two_d, 0x0c, 1);
        set(two_d, 0x10, 0);
        let row = width * format.bytes_per_pixel();
        set(two_d, 0x14, row + 4 * random.below(3) as u32);
        set(two_d, 0x18, width);
        set(two_d, 0x1c, height);
        set(two_d, 0x20, 0);
        set(two_d, 0x24, address);
    }

    /// 32.32 fixed point into the register pair at `frac`.
    fn set_fixed(two_d: &mut TwoD, frac: usize, value: i64) {
        two_d.registers[frac] = value as u32;
        two_d.registers[frac + 1] = (value >> 32) as u32;
    }

    /// A random blit: the steps and origins drawn from the ones games use
    /// (1:1, 2:1 sample-grid reductions, halving and doubling, with
    /// whole, half and random fractions) and arbitrary ones.
    fn random_blit(random: &mut Random, codes: &[u32]) -> TwoD {
        let mut two_d = TwoD::default();
        two_d.registers[SET_OPERATION] = 3;
        surface(&mut two_d, random, SET_SRC_FORMAT, SOURCE, codes);
        surface(&mut two_d, random, SET_DST_FORMAT, DESTINATION, codes);
        two_d.registers[SET_PIXELS_FROM_MEMORY_SAMPLE_MODE] = random.pick(&[0, 1, 0x10, 0x11]);
        for register in [SET_PIXELS_FROM_MEMORY_DST_X0, SET_PIXELS_FROM_MEMORY_DST_Y0] {
            two_d.registers[register] = random.below(8) as u32;
        }
        for register in [
            SET_PIXELS_FROM_MEMORY_DST_WIDTH,
            SET_PIXELS_FROM_MEMORY_DST_HEIGHT,
        ] {
            two_d.registers[register] = random.below(70) as u32;
        }
        if random.below(4) == 0 {
            two_d.registers[SET_CLIP_ENABLE] = 1;
            for register in [SET_CLIP_X0, SET_CLIP_Y0, SET_CLIP_WIDTH, SET_CLIP_HEIGHT] {
                two_d.registers[register] = random.below(40) as u32;
            }
        }
        const ONE: i64 = 1 << 32;
        for frac in [
            SET_PIXELS_FROM_MEMORY_DU_DX_FRAC,
            SET_PIXELS_FROM_MEMORY_DV_DY_FRAC,
        ] {
            let step = match random.below(5) {
                0 => ONE,
                1 => 2 * ONE,
                2 => ONE / 2,
                3 => ONE + (random.next() as i64 >> 33),
                _ => (random.next() as i64) >> 30,
            };
            set_fixed(&mut two_d, frac, step);
        }
        for frac in [
            SET_PIXELS_FROM_MEMORY_SRC_X0_FRAC,
            SET_PIXELS_FROM_MEMORY_SRC_Y0_FRAC,
        ] {
            let origin = match random.below(4) {
                0 => 0,
                1 => ONE / 2,
                2 => (random.below(8) as i64) * ONE + ONE / 2,
                _ => (random.next() as i64) >> 28,
            };
            set_fixed(&mut two_d, frac, origin);
        }
        two_d
    }

    /// Runs `count` random blits among `codes` through [`TwoD::blit`] and
    /// [`reference_blit`] on the same random memory and compares the
    /// outcome and every byte; returns how many wrote something.
    fn compare(seed: u64, count: usize, codes: &[u32]) -> usize {
        let mut random = Random(seed);
        let mut written = 0;
        for case in 0..count {
            let mut two_d = random_blit(&mut random, codes);
            let bytes = random.bytes(MEMORY);
            let fast = Flat(Mutex::new(bytes.clone()));
            let reference = Flat(Mutex::new(bytes.clone()));
            let outcome = two_d.blit(&fast, &Hooks::default());
            let expected = reference_blit(&two_d, &reference);
            assert_eq!(outcome, expected, "case {case}");
            let (fast, reference) = (
                fast.0.into_inner().unwrap(),
                reference.0.into_inner().unwrap(),
            );
            if fast != reference {
                let at = fast
                    .iter()
                    .zip(&reference)
                    .position(|(a, b)| a != b)
                    .unwrap();
                panic!(
                    "case {case}: byte {at:#x} is {:#x}, the per-pixel blit wrote {:#x}; registers {:?}",
                    fast[at],
                    reference[at],
                    two_d
                        .registers
                        .iter()
                        .enumerate()
                        .filter(|(_, v)| **v != 0)
                        .collect::<Vec<_>>()
                );
            }
            written += usize::from(fast != bytes);
        }
        written
    }

    #[test]
    fn blits_match_the_per_pixel_blit_between_any_formats() {
        let written = compare(0x2D2D_2D2D, 1500, &format_codes());
        assert!(written > 300, "{written}");
    }

    /// Same-format blits, where the byte copies and the table conversions
    /// apply: the 8-bit four-component formats and a few others.
    #[test]
    fn same_format_blits_match_the_per_pixel_blit() {
        for code in format_codes() {
            let written = compare(0x5EED ^ u64::from(code), 60, &[code]);
            assert!(written > 0, "{code:#x}");
        }
        let eight_bit = [0xcf, 0xd0, 0xd5, 0xd6, 0xe6, 0xe7, 0xf9, 0xfa, 0xfd, 0xfe];
        compare(0xB17, 1500, &eight_bit);
    }

    /// The MSAA 2x1 sample-grid reduction NVN applications blit: a
    /// 2560x720 A8R8G8B8 block-linear surface at 0 to a 1280x720 one at
    /// 8 MiB, du 2, u0 0.5, bilinear, origin centre.
    fn sample_grid_reduction() -> TwoD {
        let mut two_d = TwoD::default();
        let r = &mut two_d.registers;
        r[SET_OPERATION] = 3;
        for (base, address, width) in [(SET_SRC_FORMAT, 0, 2560), (SET_DST_FORMAT, 8 << 20, 1280)] {
            for (offset, value) in [
                (0x00, 0xcf),
                (0x04, 0),
                (0x08, 4 << 4),
                (0x0c, 1),
                (0x18, width),
                (0x1c, 720),
                (0x24, address),
            ] {
                r[m(base + offset)] = value;
            }
        }
        r[SET_PIXELS_FROM_MEMORY_SAMPLE_MODE] = 0x10;
        r[SET_PIXELS_FROM_MEMORY_DST_WIDTH] = 1280;
        r[SET_PIXELS_FROM_MEMORY_DST_HEIGHT] = 720;
        set_fixed(&mut two_d, SET_PIXELS_FROM_MEMORY_DU_DX_FRAC, 2 << 32);
        set_fixed(&mut two_d, SET_PIXELS_FROM_MEMORY_DV_DY_FRAC, 1 << 32);
        set_fixed(&mut two_d, SET_PIXELS_FROM_MEMORY_SRC_X0_FRAC, 1 << 31);
        two_d
    }

    #[test]
    fn the_sample_grid_reduction_matches_the_per_pixel_blit() {
        let mut random = Random(0x1280);
        let mut two_d = sample_grid_reduction();
        let bytes = random.bytes(12 << 20);
        let fast = Flat(Mutex::new(bytes.clone()));
        let reference = Flat(Mutex::new(bytes.clone()));
        assert_eq!(two_d.blit(&fast, &Hooks::default()), Ok(()));
        assert_eq!(reference_blit(&two_d, &reference), Ok(()));
        let fast = fast.0.into_inner().unwrap();
        assert!(fast == reference.0.into_inner().unwrap());
        assert!(fast != bytes);
    }

    /// The hooks of a channel that renders with Metal, if there is a
    /// device.
    fn metal_hooks() -> Option<Hooks> {
        let renderer = super::super::metal::Renderer::new()?;
        Some(Hooks {
            metal: Some(std::sync::Arc::new(renderer)),
            ..Hooks::default()
        })
    }

    /// Runs `step` with engine memory over `flat` as a channel with
    /// `hooks` passes it to its engines (`Channel::call`).
    fn with_engine_memory<T>(
        flat: &Flat,
        hooks: &Hooks,
        step: impl FnOnce(&dyn GpuMemory) -> T,
    ) -> T {
        let tracked = super::super::resources::Tracked {
            memory: flat,
            writes: &hooks.writes,
        };
        step(&super::super::metal::Coherent {
            memory: &tracked,
            renderer: hooks.metal.as_ref().expect("a Metal channel"),
            writes: &hooks.writes,
        })
    }

    /// The channel signals (`Channel::flush`): what it did reaches `flat`,
    /// but for the multisampled surfaces the renderer holds, which reach
    /// it on demand; all of them with `settle` (`Channel::settle`).
    fn signal(flat: &Flat, hooks: &Hooks, settle: bool) {
        let tracked = super::super::resources::Tracked {
            memory: flat,
            writes: &hooks.writes,
        };
        assert_eq!(hooks.signal(&tracked), Ok(()));
        if settle {
            let metal = hooks.metal.as_ref().expect("a Metal channel");
            assert_eq!(metal.settle(&tracked, &hooks.writes), Ok(()));
        }
    }

    /// The sample-grid reduction on Metal as a frame has it: a 2X1 target
    /// rendered on the GPU (here by a 1:1 blit into it from another target
    /// the renderer holds as a draw leaves it), then reduced. Nothing
    /// reaches memory before the signal (the GPU did both blits). After
    /// it the destination's memory is the software path's, and the 2X1
    /// target's is still what it was: only the reduction read it, on the
    /// GPU, so it stays there until the channel settles.
    #[test]
    fn the_sample_grid_reduction_on_metal_matches_the_software_blit() {
        const FILL: u32 = 16 << 20;
        let Some(hooks) = metal_hooks() else {
            return;
        };
        let mut random = Random(0x1281);
        let mut two_d = sample_grid_reduction();
        // 1:1 from a 2560x720 surface at `FILL` to the reduction's source.
        let mut fill = sample_grid_reduction();
        fill.registers[m(SET_SRC_FORMAT + 0x24)] = FILL;
        fill.registers[m(SET_DST_FORMAT + 0x18)] = 2560;
        fill.registers[m(SET_DST_FORMAT + 0x24)] = 0;
        fill.registers[SET_PIXELS_FROM_MEMORY_SAMPLE_MODE] = 0;
        fill.registers[SET_PIXELS_FROM_MEMORY_DST_WIDTH] = 2560;
        set_fixed(&mut fill, SET_PIXELS_FROM_MEMORY_DU_DX_FRAC, 1 << 32);
        set_fixed(&mut fill, SET_PIXELS_FROM_MEMORY_SRC_X0_FRAC, 0);
        let bytes = random.bytes(24 << 20);
        let reference = Flat(Mutex::new(bytes.clone()));
        assert_eq!(fill.blit(&reference, &Hooks::default()), Ok(()));
        assert_eq!(two_d.blit(&reference, &Hooks::default()), Ok(()));
        let reference = reference.0.into_inner().unwrap();
        let flat = Flat(Mutex::new(bytes.clone()));
        let metal = hooks.metal.as_ref().unwrap();
        with_engine_memory(&flat, &hooks, |memory| {
            for surface in [fill.surface(SET_SRC_FORMAT), two_d.surface(SET_SRC_FORMAT)] {
                let held = metal.hold(memory, &hooks.writes, surface.unwrap(), [1, 0]);
                assert_eq!(held, Ok(true));
            }
            assert_eq!(fill.blit(memory, &hooks), Ok(()));
            assert_eq!(two_d.blit(memory, &hooks), Ok(()));
        });
        assert!(*flat.0.lock().unwrap() == bytes, "written before the signal");
        signal(&flat, &hooks, false);
        {
            let flat = flat.0.lock().unwrap();
            let (target, destination) = (0..8 << 20, 8 << 20..16 << 20);
            assert!(flat[destination.clone()] == reference[destination.clone()]);
            assert!(reference[destination.clone()] != bytes[destination]);
            assert!(flat[target.clone()] == bytes[target.clone()], "2X1 target written");
            assert!(reference[target.clone()] != bytes[target]);
        }
        signal(&flat, &hooks, true);
        assert!(flat.0.into_inner().unwrap() == reference);
    }

    /// Blits on a channel that renders with Metal leave the software
    /// path's bytes in memory once the channel signals. Each case is a
    /// random blit between surfaces of one format that keeps its bits,
    /// from a surface the renderer holds with a random number of samples
    /// per pixel (as a draw to it leaves it; the destination too in half
    /// the cases, which reaches memory on demand if it has samples), then
    /// a second random blit from the first one's destination, which only
    /// the GPU holds by then, to a third surface. Blits the GPU does not
    /// take (filtered ones, formats Metal has no texture for) run in
    /// software on the written-back surfaces. Every second case runs with
    /// the surfaces the case before left on the GPU, at the same
    /// addresses in other sizes and formats; in the others, the blits the
    /// GPU took are counted by their destination's memory staying as it
    /// was until the signal.
    #[test]
    fn blits_on_metal_match_the_software_blit() {
        const THIRD: u32 = 1 << 19;
        let Some(mut hooks) = metal_hooks() else {
            return;
        };
        let codes: Vec<u32> = format_codes()
            .into_iter()
            .filter(|&code| keeps_bits(ColorFormat::from_raw(code).unwrap()))
            .collect();
        let mut random = Random(0xB117_6B0);
        let mut on_gpu = [0, 0];
        for case in 0..400 {
            let fresh = case % 2 == 0;
            if fresh {
                hooks = metal_hooks().unwrap();
            }
            let metal = hooks.metal.as_ref().unwrap();
            let code = random.pick(&codes);
            let mut first = random_blit(&mut random, &[code]);
            let mut second = random_blit(&mut random, &[code]);
            // Half the cases take texels without filtering.
            for two_d in [&mut first, &mut second] {
                if random.below(2) == 0 {
                    two_d.registers[SET_PIXELS_FROM_MEMORY_SAMPLE_MODE] &= 1;
                }
            }
            // The second blit's source is the first one's destination.
            for word in 0..=m(0x24) {
                second.registers[m(SET_SRC_FORMAT) + word] =
                    first.registers[m(SET_DST_FORMAT) + word];
            }
            second.registers[m(SET_DST_FORMAT + 0x24)] = THIRD;
            let bytes = random.bytes(MEMORY);
            const SAMPLES: [[u32; 2]; 6] = [[0, 0], [0, 0], [1, 0], [1, 1], [2, 1], [2, 2]];
            let log2 = [SOURCE, DESTINATION].map(|_| random.pick(&SAMPLES));
            let held_destination = random.below(2) == 0;

            let reference = Flat(Mutex::new(bytes.clone()));
            let none = Hooks::default();
            let expected_first = first.blit(&reference, &none);
            let after_first = reference.0.lock().unwrap().clone();
            let expected_second = second.blit(&reference, &none);
            let reference = reference.0.into_inner().unwrap();

            let flat = Flat(Mutex::new(bytes.clone()));
            let (outcome_first, outcome_second) = with_engine_memory(&flat, &hooks, |memory| {
                if let Ok(source) = first.surface(SET_SRC_FORMAT) {
                    metal.hold(memory, &hooks.writes, source, log2[0]).unwrap();
                }
                if let Ok(destination) = first.surface(SET_DST_FORMAT)
                    && held_destination
                {
                    metal
                        .hold(memory, &hooks.writes, destination, log2[1])
                        .unwrap();
                }
                let outcome_first = first.blit(memory, &hooks);
                let held = fresh && *flat.0.lock().unwrap() == bytes;
                on_gpu[0] += usize::from(held && after_first != bytes);
                let outcome_second = second.blit(memory, &hooks);
                let held = held && *flat.0.lock().unwrap() == bytes;
                on_gpu[1] += usize::from(held && reference != after_first);
                (outcome_first, outcome_second)
            });
            assert_eq!(outcome_first, expected_first, "case {case}");
            assert_eq!(outcome_second, expected_second, "case {case}");
            signal(&flat, &hooks, held_destination && log2[1] != [0, 0]);
            let flat = flat.0.into_inner().unwrap();
            if flat != reference {
                let at = flat
                    .iter()
                    .zip(&reference)
                    .position(|(a, b)| a != b)
                    .unwrap();
                panic!(
                    "case {case} ({:?}, samples {log2:?}): byte {at:#x} is {:#x}, the software blits wrote {:#x}",
                    ColorFormat::from_raw(code).unwrap(),
                    flat[at],
                    reference[at],
                );
            }
        }
        assert!(on_gpu[0] > 30 && on_gpu[1] > 10, "{on_gpu:?}");
    }

    /// [`keeps_bits`]: every field value of every such format, the others
    /// random, comes back from a read and a write unchanged.
    #[test]
    fn formats_that_keep_bits_round_trip() {
        let mut random = Random(0xB175);
        for code in format_codes() {
            let format = ColorFormat::from_raw(code).unwrap();
            if !keeps_bits(format) {
                continue;
            }
            let bytes = format.bytes_per_pixel() as usize;
            let mut texels: Vec<Vec<u8>> = Vec::new();
            for _ in 0..20_000 {
                texels.push(random.bytes(bytes));
            }
            // Every 16-bit window of the texel at every byte offset.
            for at in 0..bytes.saturating_sub(1).max(1) {
                for value in 0..=u16::MAX {
                    let mut texel = random.bytes(bytes);
                    let [low, high] = value.to_le_bytes();
                    texel[at] = low;
                    if at + 1 < bytes {
                        texel[at + 1] = high;
                    }
                    texels.push(texel);
                }
            }
            for texel in texels {
                let mut out = vec![0u8; bytes];
                format.write(format.read(&texel), &mut out);
                assert_eq!(out, texel, "{format:?}");
            }
        }
    }

    /// [`Codec`] reads and writes the bits [`ColorFormat`] does.
    #[test]
    fn the_codec_converts_as_the_format_does() {
        let mut random = Random(0xC0DEC);
        let specials = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.5,
            0.5 / 255.0,
            1.5 / 255.0,
            2.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::MIN_POSITIVE,
            1e-30,
        ];
        for code in format_codes() {
            let format = ColorFormat::from_raw(code).unwrap();
            let codec = Codec::new(format);
            let bytes = format.bytes_per_pixel() as usize;
            for _ in 0..50_000 {
                let texel = random.bytes(bytes);
                let (read, expected) = (codec.read(&texel), format.read(&texel));
                assert_eq!(
                    read.map(f32::to_bits),
                    expected.map(f32::to_bits),
                    "{format:?}"
                );
                let value: [f32; 4] = std::array::from_fn(|_| match random.below(3) {
                    0 => random.pick(&specials),
                    1 => (random.below(1 << 20) as f32) / (1 << 20) as f32,
                    _ => f32::from_bits(random.next() as u32),
                });
                let mut out = random.bytes(bytes);
                let mut expected = out.clone();
                codec.write(value, &mut out);
                format.write(value, &mut expected);
                assert_eq!(out, expected, "{format:?} {value:?}");
            }
        }
    }
}
