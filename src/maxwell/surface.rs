//! Render surfaces in guest memory, and the rectangular part of one a
//! draw, clear or blit works on, held locally while the operation runs.
//!
//! A window covers whole bands of 8 rows: for block-linear surfaces a band
//! is one row of GOBs (64 bytes x 8 rows, 512 contiguous bytes each
//! [public: Tegra X1 TRM "Block Linear" layout as modeled by Ryujinx
//! `BlockLinearLayout.cs`]), for pitch-linear surfaces 8 consecutive rows.
//! Bands are disjoint in memory, so they can be split across threads.

use std::ops::Range;

use super::texture::layout::{BlockLinear, block_linear_offset};
use super::{GpuMemory, Unmodeled};

/// Rows per band.
pub const BAND_ROWS: u32 = 8;
/// Bytes in a GOB.
const GOB_BYTES: usize = 512;
/// Byte width of a GOB.
const GOB_WIDTH: u32 = 64;

/// How a surface is laid out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// Rows `pitch` bytes apart.
    Pitch {
        pitch: u32,
    },
    BlockLinear(BlockLinear),
}

/// A 2D surface: one layer of a render target, depth buffer or 2D engine
/// surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Surface {
    pub address: u64,
    /// Width and height in pixels.
    pub width: u32,
    pub height: u32,
    pub bytes_per_pixel: u32,
    pub layout: Layout,
}

/// A pixel rectangle, `x0..x1` by `y0..y1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32,
}

impl Rect {
    pub fn is_empty(&self) -> bool {
        self.x0 >= self.x1 || self.y0 >= self.y1
    }

    pub fn intersect(&self, other: &Rect) -> Rect {
        Rect {
            x0: self.x0.max(other.x0),
            y0: self.y0.max(other.y0),
            x1: self.x1.min(other.x1),
            y1: self.y1.min(other.y1),
        }
    }
}

/// The bands of a surface that cover a rectangle, copied out of guest
/// memory.
pub struct Window {
    surface: Surface,
    /// First band and the pixel columns covered.
    band0: u32,
    bands: u32,
    /// For block-linear: first GOB column; for pitch: first byte column.
    column0: u32,
    /// Bytes per band in `data`.
    band_bytes: usize,
    /// Bytes per row in a pitch band.
    row_bytes: usize,
    data: Vec<u8>,
}

fn fault(what: &str, address: u64) -> Unmodeled {
    Unmodeled(format!(
        "{what}: access to unmapped GPU address {address:#x}"
    ))
}

impl Window {
    /// Reads the bands of `surface` covering `rect` (clipped to the
    /// surface).
    pub fn load(memory: &dyn GpuMemory, surface: Surface, rect: Rect) -> Result<Window, Unmodeled> {
        let rect = rect.intersect(&Rect {
            x0: 0,
            y0: 0,
            x1: surface.width,
            y1: surface.height,
        });
        let mut window = if rect.is_empty() {
            Window {
                surface,
                band0: 0,
                bands: 0,
                column0: 0,
                band_bytes: 0,
                row_bytes: 0,
                data: Vec::new(),
            }
        } else {
            let band0 = rect.y0 / BAND_ROWS;
            let bands = rect.y1.div_ceil(BAND_ROWS) - band0;
            let bpp = surface.bytes_per_pixel;
            match surface.layout {
                Layout::BlockLinear(_) => {
                    let column0 = rect.x0 * bpp / GOB_WIDTH;
                    let columns = (rect.x1 * bpp).div_ceil(GOB_WIDTH) - column0;
                    let band_bytes = columns as usize * GOB_BYTES;
                    Window {
                        surface,
                        band0,
                        bands,
                        column0,
                        band_bytes,
                        row_bytes: 0,
                        data: vec![0; band_bytes * bands as usize],
                    }
                }
                Layout::Pitch { .. } => {
                    let column0 = rect.x0 * bpp;
                    let row_bytes = ((rect.x1 - rect.x0) * bpp) as usize;
                    Window {
                        surface,
                        band0,
                        bands,
                        column0,
                        band_bytes: row_bytes * BAND_ROWS as usize,
                        row_bytes,
                        data: vec![0; row_bytes * (BAND_ROWS * bands) as usize],
                    }
                }
            }
        };
        window.transfer(memory, false, None)?;
        Ok(window)
    }

    /// Writes the window back to guest memory.
    pub fn store(&mut self, memory: &dyn GpuMemory) -> Result<(), Unmodeled> {
        self.transfer(memory, true, None)
    }

    /// Writes back only the bands and columns (GOBs, or pitch rows)
    /// covering `rect`.
    pub fn store_rect(&mut self, memory: &dyn GpuMemory, rect: Rect) -> Result<(), Unmodeled> {
        self.transfer(memory, true, Some(rect))
    }

    fn transfer(
        &mut self,
        memory: &dyn GpuMemory,
        write: bool,
        only: Option<Rect>,
    ) -> Result<(), Unmodeled> {
        let surface = self.surface;
        let bpp = surface.bytes_per_pixel;
        for band in 0..self.bands {
            let y = (self.band0 + band) * BAND_ROWS;
            if let Some(rect) = only
                && (y + BAND_ROWS <= rect.y0 || y >= rect.y1)
            {
                continue;
            }
            let band_data = &mut self.data[band as usize * self.band_bytes..][..self.band_bytes];
            match surface.layout {
                Layout::BlockLinear(block) => {
                    let width_bytes = surface.width * bpp;
                    for (index, gob) in band_data.chunks_mut(GOB_BYTES).enumerate() {
                        let x = (self.column0 + index as u32) * GOB_WIDTH;
                        if let Some(rect) = only
                            && (x + GOB_WIDTH <= rect.x0 * bpp || x >= rect.x1 * bpp)
                        {
                            continue;
                        }
                        let address = surface.address
                            + block_linear_offset(x, y, 0, width_bytes, surface.height, block);
                        if write {
                            memory.write(address, gob)
                        } else {
                            memory.read(address, gob)
                        }
                        .ok_or_else(|| fault("surface", address))?;
                    }
                }
                Layout::Pitch { pitch } => {
                    for (row, line) in band_data.chunks_mut(self.row_bytes).enumerate() {
                        let y = y + row as u32;
                        if y >= surface.height {
                            break;
                        }
                        if let Some(rect) = only
                            && (y < rect.y0 || y >= rect.y1)
                        {
                            continue;
                        }
                        let address = surface.address
                            + u64::from(y) * u64::from(pitch)
                            + u64::from(self.column0);
                        if write {
                            memory.write(address, line)
                        } else {
                            memory.read(address, line)
                        }
                        .ok_or_else(|| fault("surface", address))?;
                    }
                }
            }
        }
        Ok(())
    }

    pub fn surface(&self) -> &Surface {
        &self.surface
    }

    /// The window's bytes as they are in guest memory, band after band.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Copies the pixels of `rect` (inside the window) to `out`, row after
    /// row, `bytes_per_pixel * (x1 - x0)` bytes per row.
    pub fn read_pixels(&self, rect: Rect, out: &mut [u8]) {
        let data = &self.data;
        self.runs(rect, |window, linear| {
            out[linear].copy_from_slice(&data[window]);
        });
    }

    /// The inverse of [`Window::read_pixels`].
    pub fn write_pixels(&mut self, rect: Rect, pixels: &[u8]) {
        // `runs` looks only at the window's geometry.
        let mut data = std::mem::take(&mut self.data);
        self.runs(rect, |window, linear| {
            data[window].copy_from_slice(&pixels[linear]);
        });
        self.data = data;
    }

    /// Walks `rect` (inside the window) in runs of bytes contiguous both in
    /// the window's data and in a row-major buffer of the rectangle:
    /// `run(window range, buffer range)`. Within a GOB, 16-byte units
    /// are contiguous in x ([`Band::offset`]); pitch rows are contiguous.
    fn runs(&self, rect: Rect, mut run: impl FnMut(Range<usize>, Range<usize>)) {
        let bpp = self.surface.bytes_per_pixel as usize;
        let row = (rect.x1 - rect.x0) as usize * bpp;
        let (xb0, xb1) = (rect.x0 as usize * bpp, rect.x1 as usize * bpp);
        for y in rect.y0..rect.y1 {
            let Some(index) = (y / BAND_ROWS).checked_sub(self.band0) else {
                continue;
            };
            if index >= self.bands {
                continue;
            }
            let band = index as usize * self.band_bytes;
            let line = (y - rect.y0) as usize * row;
            let r = (y % BAND_ROWS) as usize;
            match self.surface.layout {
                Layout::BlockLinear(_) => {
                    let mut xb = xb0;
                    while xb < xb1 {
                        let gob = xb / GOB_WIDTH as usize - self.column0 as usize;
                        let x = xb % GOB_WIDTH as usize;
                        let n = (16 - x % 16).min(xb1 - xb);
                        let inner = ((x & 0x20) << 3)
                            | ((r & 0x6) << 5)
                            | ((x & 0x10) << 1)
                            | ((r & 1) << 4)
                            | (x & 0xF);
                        let at = band + gob * GOB_BYTES + inner;
                        let to = line + xb - xb0;
                        run(at..at + n, to..to + n);
                        xb += n;
                    }
                }
                Layout::Pitch { .. } => {
                    let at = band + r * self.row_bytes + xb0 - self.column0 as usize;
                    run(at..at + row, line..line + row);
                }
            }
        }
    }

    /// The bands as (first row, band) pairs, for processing in parallel.
    pub fn bands_mut(&mut self) -> impl Iterator<Item = Band<'_>> {
        let surface = self.surface;
        let column0 = self.column0;
        let row_bytes = self.row_bytes;
        let band0 = self.band0;
        let band_bytes = self.band_bytes.max(1);
        self.data
            .chunks_mut(band_bytes)
            .enumerate()
            .map(move |(index, data)| Band {
                surface,
                y0: (band0 + index as u32) * BAND_ROWS,
                column0,
                row_bytes,
                data,
            })
    }

    /// The band holding row `y`, if the window covers it.
    pub fn band_mut(&mut self, y: u32) -> Option<Band<'_>> {
        let index = (y / BAND_ROWS).checked_sub(self.band0)?;
        if index >= self.bands {
            return None;
        }
        let surface = self.surface;
        Some(Band {
            surface,
            y0: (self.band0 + index) * BAND_ROWS,
            column0: self.column0,
            row_bytes: self.row_bytes,
            data: &mut self.data[index as usize * self.band_bytes..][..self.band_bytes],
        })
    }
}

/// Eight rows of a window.
pub struct Band<'a> {
    surface: Surface,
    /// First row of the band.
    pub y0: u32,
    column0: u32,
    row_bytes: usize,
    data: &'a mut [u8],
}

impl Band<'_> {
    /// Byte offset of pixel (`x`, `y`) in the band; `y` must be in the band
    /// and `x` in the window.
    #[inline]
    fn offset(&self, x: u32, y: u32) -> usize {
        let bpp = self.surface.bytes_per_pixel;
        let row = y - self.y0;
        match self.surface.layout {
            Layout::BlockLinear(_) => {
                let x_bytes = x * bpp;
                let gob = (x_bytes / GOB_WIDTH - self.column0) as usize;
                // Inside a GOB [public: Ryujinx BlockLinearLayout.cs]: x
                // bits 3:0 -> 3:0, y bit 0 -> 4, x bit 4 -> 5, y bits 2:1 ->
                // 7:6, x bit 5 -> 8.
                let x = x_bytes % GOB_WIDTH;
                let inner = ((x & 0x20) << 3)
                    | ((row & 0x6) << 5)
                    | ((x & 0x10) << 1)
                    | ((row & 1) << 4)
                    | (x & 0xF);
                gob * GOB_BYTES + inner as usize
            }
            Layout::Pitch { .. } => {
                row as usize * self.row_bytes + (x * bpp - self.column0) as usize
            }
        }
    }

    /// The bytes of pixel (`x`, `y`).
    #[inline]
    pub fn pixel(&self, x: u32, y: u32) -> &[u8] {
        let offset = self.offset(x, y);
        &self.data[offset..offset + self.surface.bytes_per_pixel as usize]
    }

    #[inline]
    pub fn pixel_mut(&mut self, x: u32, y: u32) -> &mut [u8] {
        let offset = self.offset(x, y);
        let bpp = self.surface.bytes_per_pixel as usize;
        &mut self.data[offset..offset + bpp]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Memory that holds `byte(address)` everywhere.
    struct Pattern;

    fn byte(address: u64) -> u8 {
        (address.wrapping_mul(0x9E37_79B9) >> 13) as u8
    }

    impl GpuMemory for Pattern {
        fn read(&self, address: u64, buffer: &mut [u8]) -> Option<()> {
            for (i, b) in buffer.iter_mut().enumerate() {
                *b = byte(address + i as u64);
            }
            Some(())
        }

        fn write(&self, _: u64, _: &[u8]) -> Option<()> {
            None
        }
    }

    /// The run-wise copies agree with [`Band::pixel`] pixel by pixel, for
    /// both layouts, several pixel sizes and unaligned rectangles, and
    /// writing leaves the bytes outside the rectangle alone.
    #[test]
    fn pixel_copies_match_band_pixels() {
        let layouts = [
            Layout::Pitch { pitch: 0x400 },
            Layout::BlockLinear(BlockLinear {
                block_height_log2: 1,
                block_depth_log2: 0,
            }),
            Layout::BlockLinear(BlockLinear {
                block_height_log2: 4,
                block_depth_log2: 0,
            }),
        ];
        let rects = [
            Rect {
                x0: 0,
                y0: 0,
                x1: 100,
                y1: 70,
            },
            Rect {
                x0: 3,
                y0: 5,
                x1: 61,
                y1: 37,
            },
            Rect {
                x0: 17,
                y0: 9,
                x1: 18,
                y1: 10,
            },
            Rect {
                x0: 31,
                y0: 0,
                x1: 97,
                y1: 69,
            },
        ];
        for layout in layouts {
            for bpp in [1, 2, 4, 8, 16] {
                let surface = Surface {
                    address: 0x10000,
                    width: 100,
                    height: 70,
                    bytes_per_pixel: bpp,
                    layout,
                };
                for window_rect in rects {
                    let mut window = Window::load(&Pattern, surface, window_rect).unwrap();
                    for rect in rects.map(|r| r.intersect(&window_rect)) {
                        if rect.is_empty() {
                            continue;
                        }
                        let bpp = bpp as usize;
                        let row = (rect.x1 - rect.x0) as usize * bpp;
                        let mut out = vec![0; row * (rect.y1 - rect.y0) as usize];
                        window.read_pixels(rect, &mut out);
                        for y in rect.y0..rect.y1 {
                            let band = window.band_mut(y).unwrap();
                            for x in rect.x0..rect.x1 {
                                let at =
                                    (y - rect.y0) as usize * row + (x - rect.x0) as usize * bpp;
                                assert_eq!(&out[at..at + bpp], band.pixel(x, y), "{layout:?}");
                            }
                        }
                        let before = window.data().to_vec();
                        let pixels: Vec<u8> = (0..out.len()).map(|i| !out[i] ^ i as u8).collect();
                        window.write_pixels(rect, &pixels);
                        let mut expected = before;
                        let mut reference = Window {
                            data: std::mem::take(&mut expected),
                            ..window.clone_geometry()
                        };
                        for y in rect.y0..rect.y1 {
                            let mut band = reference.band_mut(y).unwrap();
                            for x in rect.x0..rect.x1 {
                                let at =
                                    (y - rect.y0) as usize * row + (x - rect.x0) as usize * bpp;
                                band.pixel_mut(x, y).copy_from_slice(&pixels[at..at + bpp]);
                            }
                        }
                        assert_eq!(window.data(), reference.data(), "{layout:?} {rect:?}");
                    }
                }
            }
        }
    }

    impl Window {
        /// A window of the same geometry without data.
        fn clone_geometry(&self) -> Window {
            Window {
                surface: self.surface,
                band0: self.band0,
                bands: self.bands,
                column0: self.column0,
                band_bytes: self.band_bytes,
                row_bytes: self.row_bytes,
                data: Vec::new(),
            }
        }
    }
}
