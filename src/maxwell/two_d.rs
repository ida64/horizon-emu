//! The Fermi 2D class (0x902D): surface-to-surface blits with scaling,
//! filtering and format conversion ("pixels from memory").
//!
//! Methods and fields [public: NVIDIA open-gpu-doc
//! `classes/twod/cl902d.h`]. The blit is triggered by
//! `PIXELS_FROM_MEMORY_SRC_Y0_INT` (0x8DC), the last method of the block
//! [public: eden `Fermi2D::CallMethod`, Ryujinx
//! `TwodClass.PixelsFromMemorySrcY0Int`]. Destination pixel `i` samples
//! the source at `src_x0 + (i + 0.5) * du_dx` with `ORIGIN_CENTER` and at
//! `src_x0 + i * du_dx` with `ORIGIN_CORNER` (Ryujinx's reading of the
//! origin modes: with CORNER the API adds half a step itself); `src_x0`,
//! `du_dx` and their y counterparts are 32.32 fixed point.

use super::surface::{Layout, Rect, Surface, Window};
use super::texture::format::ColorFormat;
use super::texture::layout::BlockLinear;
use super::{GpuMemory, Unmodeled};

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

/// The 2D engine of a channel.
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
            PIXELS_FROM_MEMORY_SRC_Y0_INT => self.blit(memory),
            _ => Ok(()),
        }
    }

    fn surface(&self, base: u32) -> Result<(Surface, ColorFormat), Unmodeled> {
        let r = |offset: u32| self.registers[m(base + offset)];
        let raw = r(0x00);
        let format =
            ColorFormat::from_raw(raw).ok_or_else(|| refuse(m(base), raw, "surface format"))?;
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

    fn blit(&mut self, memory: &dyn GpuMemory) -> Result<(), Unmodeled> {
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
        // Source texels as f32, row-major over the loaded span.
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
}
