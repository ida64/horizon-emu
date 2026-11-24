//! Render surfaces in guest memory, and the rectangular part of one a
//! draw, clear or blit works on, held locally while the operation runs.
//!
//! A window covers whole bands of 8 rows: for block-linear surfaces a band
//! is one row of GOBs (64 bytes x 8 rows, 512 contiguous bytes each
//! [public: Tegra X1 TRM "Block Linear" layout]), for pitch-linear
//! surfaces 8 consecutive rows.
//! Bands are disjoint in memory, so they can be split across threads.

use std::ops::Range;

use super::texture::layout::{BlockLinear, BlockLinearSurface};
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

/// Copies one run of [`Window::runs`]: most are one 16-byte unit of a
/// GOB, copied as such rather than through a call per unit.
#[inline(always)]
fn copy_run(to: &mut [u8], from: &[u8]) {
    match (
        <&mut [u8; 16]>::try_from(&mut *to),
        <&[u8; 16]>::try_from(from),
    ) {
        (Ok(to), Ok(from)) => *to = *from,
        _ => to.copy_from_slice(from),
    }
}

fn fault(what: &str, address: u64) -> Unmodeled {
    Unmodeled(format!(
        "{what}: access to unmapped GPU address {address:#x}"
    ))
}

/// The most bytes one transfer of [`in_runs`] moves.
const RUN_BYTES: usize = 256 << 10;

/// Calls `run(address, pieces)` for each run of `pieces` (by address)
/// that lie end to end in memory, of up to [`RUN_BYTES`] (a longer
/// piece alone): one guest memory access, one translation a page, for
/// what would be hundreds of 512-byte GOBs. A block row of a block-linear
/// surface the window spans whole lies together in memory.
fn in_runs(
    pieces: &[(u64, Range<usize>)],
    mut run: impl FnMut(u64, &[(u64, Range<usize>)]) -> Result<(), Unmodeled>,
) -> Result<(), Unmodeled> {
    let mut start = 0;
    while start < pieces.len() {
        let (mut end, mut bytes) = (start + 1, pieces[start].1.len());
        while let Some((address, range)) = pieces.get(end)
            && *address == pieces[end - 1].0 + pieces[end - 1].1.len() as u64
            && bytes + range.len() <= RUN_BYTES
        {
            bytes += range.len();
            end += 1;
        }
        run(pieces[start].0, &pieces[start..end])?;
        start = end;
    }
    Ok(())
}

/// Offset in a GOB of the 16-byte unit at byte column `x` (a multiple of
/// 16 below 64) of row `row` (below 8), as in [`Band::offset`].
fn gob_unit(x: usize, row: usize) -> usize {
    ((x & 0x20) << 3) | ((row & 0x6) << 5) | ((x & 0x10) << 1) | ((row & 1) << 4)
}

impl Window {
    /// A window of `surface` holding nothing.
    pub fn empty(surface: Surface) -> Window {
        Window {
            surface,
            band0: 0,
            bands: 0,
            column0: 0,
            band_bytes: 0,
            row_bytes: 0,
            data: Vec::new(),
        }
    }

    /// The window of `surface` covering `rect` (clipped to the surface),
    /// without its bytes.
    fn shape(surface: Surface, rect: Rect) -> Window {
        let rect = rect.intersect(&Rect {
            x0: 0,
            y0: 0,
            x1: surface.width,
            y1: surface.height,
        });
        if rect.is_empty() {
            return Window::empty(surface);
        }
        let band0 = rect.y0 / BAND_ROWS;
        let bands = rect.y1.div_ceil(BAND_ROWS) - band0;
        let bpp = surface.bytes_per_pixel;
        match surface.layout {
            Layout::BlockLinear(_) => {
                let column0 = rect.x0 * bpp / GOB_WIDTH;
                let columns = (rect.x1 * bpp).div_ceil(GOB_WIDTH) - column0;
                Window {
                    surface,
                    band0,
                    bands,
                    column0,
                    band_bytes: columns as usize * GOB_BYTES,
                    row_bytes: 0,
                    data: Vec::new(),
                }
            }
            Layout::Pitch { .. } => {
                let row_bytes = ((rect.x1 - rect.x0) * bpp) as usize;
                Window {
                    surface,
                    band0,
                    bands,
                    column0: rect.x0 * bpp,
                    band_bytes: row_bytes * BAND_ROWS as usize,
                    row_bytes,
                    data: Vec::new(),
                }
            }
        }
    }

    /// The window [`Window::load`] makes of `rect`, holding zeros, for a
    /// caller that fills the bands it stores.
    pub fn blank(surface: Surface, rect: Rect) -> Window {
        let mut window = Window::shape(surface, rect);
        window.data = vec![0; window.band_bytes * window.bands as usize];
        window
    }

    /// Reads the bands of `surface` covering `rect` (clipped to the
    /// surface).
    pub fn load(memory: &dyn GpuMemory, surface: Surface, rect: Rect) -> Result<Window, Unmodeled> {
        let mut window = Window::shape(surface, rect);
        window.data = vec![0; window.band_bytes * window.bands as usize];
        window.transfer(memory, false, None, None)?;
        Ok(window)
    }

    /// [`Window::load`] for a caller that writes every pixel of `rect`
    /// before storing the window: the GOBs (pitch rows) `rect` covers
    /// whole are not read, and hold zeros until written.
    pub fn load_to_overwrite(
        memory: &dyn GpuMemory,
        surface: Surface,
        rect: Rect,
    ) -> Result<Window, Unmodeled> {
        let mut window = Window::shape(surface, rect);
        window.data = vec![0; window.band_bytes * window.bands as usize];
        window.transfer(memory, false, None, Some(rect))?;
        Ok(window)
    }

    /// The pixels of `rect` (inside the surface) read from guest memory
    /// into `out`, row after row as [`Window::read_pixels`] gives them,
    /// without a window: each run of memory goes straight to its rows.
    pub fn read_rect(
        memory: &dyn GpuMemory,
        surface: Surface,
        rect: Rect,
        out: &mut [u8],
    ) -> Result<(), Unmodeled> {
        let window = Window::shape(surface, rect);
        let bpp = surface.bytes_per_pixel as usize;
        let row = (rect.x1 - rect.x0) as usize * bpp;
        let (xb0, xb1) = (rect.x0 as usize * bpp, rect.x1 as usize * bpp);
        let mut scratch = Vec::new();
        in_runs(&window.pieces(None, None), |address, run| {
            let total = run.iter().map(|(_, range)| range.len()).sum();
            scratch.resize(total, 0);
            memory
                .read(address, &mut scratch)
                .ok_or_else(|| fault("surface", address))?;
            let mut at = 0;
            for (_, range) in run {
                let piece = &scratch[at..at + range.len()];
                at += range.len();
                let band = range.start / window.band_bytes;
                let within = range.start % window.band_bytes;
                let y0 = (window.band0 + band as u32) * BAND_ROWS;
                match surface.layout {
                    Layout::BlockLinear(_) => {
                        let gob =
                            (window.column0 as usize + within / GOB_BYTES) * GOB_WIDTH as usize;
                        for r in 0..BAND_ROWS as usize {
                            let y = y0 + r as u32;
                            if y < rect.y0 || y >= rect.y1 {
                                continue;
                            }
                            let line = (y - rect.y0) as usize * row;
                            for unit in (0..GOB_WIDTH as usize).step_by(16) {
                                let (lo, hi) = ((gob + unit).max(xb0), (gob + unit + 16).min(xb1));
                                if lo >= hi {
                                    continue;
                                }
                                let inner = gob_unit(unit, r) + (lo - gob - unit);
                                copy_run(
                                    &mut out[line + lo - xb0..line + hi - xb0],
                                    &piece[inner..inner + hi - lo],
                                );
                            }
                        }
                    }
                    Layout::Pitch { .. } => {
                        let y = y0 + (within / window.row_bytes) as u32;
                        if (rect.y0..rect.y1).contains(&y) {
                            let line = (y - rect.y0) as usize * row;
                            out[line..line + row].copy_from_slice(piece);
                        }
                    }
                }
            }
            Ok(())
        })
    }

    /// Writes the window back to guest memory.
    pub fn store(&mut self, memory: &dyn GpuMemory) -> Result<(), Unmodeled> {
        self.transfer(memory, true, None, None)
    }

    /// Writes back only the bands and columns (GOBs, or pitch rows)
    /// covering `rect`.
    pub fn store_rect(&mut self, memory: &dyn GpuMemory, rect: Rect) -> Result<(), Unmodeled> {
        self.transfer(memory, true, Some(rect), None)
    }

    /// Moves the pieces covering `only` (all without it) between guest
    /// memory and the window, except those `skip` covers whole, each run
    /// of them that lies together in memory at once.
    fn transfer(
        &mut self,
        memory: &dyn GpuMemory,
        write: bool,
        only: Option<Rect>,
        skip: Option<Rect>,
    ) -> Result<(), Unmodeled> {
        let pieces = self.pieces(only, skip);
        let data = &mut self.data;
        let mut scratch = Vec::new();
        in_runs(&pieces, |address, run| {
            let done = if let [(_, range)] = run {
                let piece = &mut data[range.clone()];
                if write {
                    memory.write(address, piece)
                } else {
                    memory.read(address, piece)
                }
            } else {
                scratch.clear();
                if write {
                    for (_, range) in run {
                        scratch.extend_from_slice(&data[range.clone()]);
                    }
                    memory.write(address, &scratch)
                } else {
                    scratch.resize(run.iter().map(|(_, range)| range.len()).sum(), 0);
                    let read = memory.read(address, &mut scratch);
                    let mut at = 0;
                    for (_, range) in run {
                        data[range.clone()].copy_from_slice(&scratch[at..at + range.len()]);
                        at += range.len();
                    }
                    read
                }
            };
            done.ok_or_else(|| fault("surface", address))
        })
    }

    /// Whether guest memory holds the window's bytes, compared run by run.
    /// Reads the same bytes as [`Window::load`], and refuses where it
    /// would.
    pub fn matches(&self, memory: &dyn GpuMemory) -> Result<bool, Unmodeled> {
        let mut scratch = Vec::new();
        let mut same = true;
        // A difference ends the walk early, through this marker.
        let differs = || Unmodeled(String::new());
        let result = in_runs(&self.pieces(None, None), |address, run| {
            // A run whose bytes lie together in the window too is
            // compared in place.
            let first = &run[0].1;
            let last = &run[run.len() - 1].1;
            if run.windows(2).all(|pair| pair[0].1.end == pair[1].1.start) {
                let held = memory
                    .equals(address, &self.data[first.start..last.end])
                    .ok_or_else(|| fault("surface", address))?;
                if !held {
                    same = false;
                    return Err(differs());
                }
                return Ok(());
            }
            scratch.resize(run.iter().map(|(_, range)| range.len()).sum(), 0);
            memory
                .read(address, &mut scratch)
                .ok_or_else(|| fault("surface", address))?;
            let mut at = 0;
            for (_, range) in run {
                if scratch[at..at + range.len()] != self.data[range.clone()] {
                    same = false;
                    return Err(differs());
                }
                at += range.len();
            }
            Ok(())
        });
        match result {
            Err(_) if !same => Ok(false),
            outcome => outcome.map(|()| true),
        }
    }

    /// The pieces of guest memory the window holds, contiguous there and
    /// in `data`, restricted to the bands and columns (GOBs, or pitch
    /// rows) covering `only`, without those `skip` covers whole: (address,
    /// data range), by address.
    fn pieces(&self, only: Option<Rect>, skip: Option<Rect>) -> Vec<(u64, Range<usize>)> {
        let surface = self.surface;
        let bpp = surface.bytes_per_pixel;
        // Whether `skip` covers bytes `x0..x1` of rows `y0..y1`.
        let skipped = |x0: u32, x1: u32, y0: u32, y1: u32| {
            skip.is_some_and(|rect| {
                x0 >= rect.x0 * bpp && x1 <= rect.x1 * bpp && y0 >= rect.y0 && y1 <= rect.y1
            })
        };
        let bands: Vec<u32> = (0..self.bands)
            .filter(|band| {
                let y = (self.band0 + band) * BAND_ROWS;
                only.is_none_or(|rect| y + BAND_ROWS > rect.y0 && y < rect.y1)
            })
            .collect();
        let mut pieces = Vec::new();
        match surface.layout {
            Layout::BlockLinear(block) => {
                let layout = BlockLinearSurface::new(surface.width * bpp, surface.height, block);
                // In memory order: by block row, then GOB column, then GOB
                // row within the block.
                let block_row = |band: &u32| (self.band0 + band) >> block.block_height_log2;
                for group in bands.chunk_by(|a, b| block_row(a) == block_row(b)) {
                    for index in 0..self.band_bytes / GOB_BYTES {
                        let x = (self.column0 + index as u32) * GOB_WIDTH;
                        if let Some(rect) = only
                            && (x + GOB_WIDTH <= rect.x0 * bpp || x >= rect.x1 * bpp)
                        {
                            continue;
                        }
                        for &band in group {
                            let y = (self.band0 + band) * BAND_ROWS;
                            if skipped(x, x + GOB_WIDTH, y, y + BAND_ROWS) {
                                continue;
                            }
                            let address = surface.address + layout.offset(x, y, 0);
                            let start = band as usize * self.band_bytes + index * GOB_BYTES;
                            pieces.push((address, start..start + GOB_BYTES));
                        }
                    }
                }
            }
            Layout::Pitch { pitch } => {
                for band in bands {
                    for row in 0..BAND_ROWS as usize {
                        let y = (self.band0 + band) * BAND_ROWS + row as u32;
                        if y >= surface.height {
                            break;
                        }
                        if let Some(rect) = only
                            && (y < rect.y0 || y >= rect.y1)
                        {
                            continue;
                        }
                        if skipped(self.column0, self.column0 + self.row_bytes as u32, y, y + 1) {
                            continue;
                        }
                        let address = surface.address
                            + u64::from(y) * u64::from(pitch)
                            + u64::from(self.column0);
                        let start = band as usize * self.band_bytes + row * self.row_bytes;
                        pieces.push((address, start..start + self.row_bytes));
                    }
                }
            }
        }
        debug_assert!(pieces.is_sorted_by_key(|&(address, _)| address));
        pieces
    }

    pub fn surface(&self) -> &Surface {
        &self.surface
    }

    /// The window's bytes as they are in guest memory, band after band.
    ///
    /// The layout, stable for code that fills it itself ([`Window::data_mut`]):
    /// band `n` (rows `8·(band0 + n)` on, see [`Window::band_range`]) is
    /// `band_bytes` long. Block-linear, a band is its GOBs (64 bytes × 8
    /// rows, 512 bytes each) left to right from the window's first GOB
    /// column, each GOB's bytes in the GPU's order (16-byte units, see
    /// [`Window::texel_offset`]); pitch-linear, its 8 rows one after the
    /// other, each the window's columns' bytes. Guest memory gets them at
    /// [`Window::store`]/[`Window::store_rect`].
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// [`Window::data`], to fill in that layout.
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Where in [`Window::data`] the band holding row `y` lies, if the
    /// window covers it.
    pub fn band_range(&self, y: u32) -> Option<Range<usize>> {
        let index = (y / BAND_ROWS).checked_sub(self.band0)?;
        (index < self.bands).then(|| {
            let start = index as usize * self.band_bytes;
            start..start + self.band_bytes
        })
    }

    /// Where in [`Window::data`] the bytes of pixel (`x`, `y`) start; the
    /// pixel must lie in the window. Block-linear: band start + GOB column
    /// (`x·bpp / 64` from the window's first) × 512 + the offset inside the
    /// GOB, `x` bits of the byte column 3:0 → 3:0, row bit 0 → 4, byte
    /// column bit 4 → 5, row bits 2:1 → 7:6, byte column bit 5 → 8 (row
    /// counted within the band).
    /// Pitch: band start + row within the band × row bytes + the byte
    /// column from the window's first.
    pub fn texel_offset(&self, x: u32, y: u32) -> usize {
        let index = (y / BAND_ROWS - self.band0) as usize;
        let band = Band {
            surface: self.surface,
            y0: y / BAND_ROWS * BAND_ROWS,
            column0: self.column0,
            row_bytes: self.row_bytes,
            data: &mut [],
        };
        index * self.band_bytes + band.offset(x, y)
    }

    /// Copies the pixels of `rect` (inside the window) to `out`, row after
    /// row, `bytes_per_pixel * (x1 - x0)` bytes per row.
    pub fn read_pixels(&self, rect: Rect, out: &mut [u8]) {
        let data = &self.data;
        self.runs(rect, |window, linear| {
            copy_run(&mut out[linear], &data[window]);
        });
    }

    /// The inverse of [`Window::read_pixels`].
    pub fn write_pixels(&mut self, rect: Rect, pixels: &[u8]) {
        // `runs` looks only at the window's geometry.
        let mut data = std::mem::take(&mut self.data);
        self.runs(rect, |window, linear| {
            copy_run(&mut data[window], &pixels[linear]);
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
                // Inside a GOB: x
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

    /// Byte-addressed memory over [`Pattern`], with the writes kept.
    #[derive(Default)]
    struct Written(std::sync::Mutex<std::collections::HashMap<u64, u8>>);

    impl GpuMemory for Written {
        fn read(&self, address: u64, buffer: &mut [u8]) -> Option<()> {
            let written = self.0.lock().unwrap();
            for (i, b) in buffer.iter_mut().enumerate() {
                let at = address + i as u64;
                *b = written.get(&at).copied().unwrap_or(byte(at));
            }
            Some(())
        }

        fn write(&self, address: u64, data: &[u8]) -> Option<()> {
            let mut written = self.0.lock().unwrap();
            for (i, b) in data.iter().enumerate() {
                written.insert(address + i as u64, *b);
            }
            Some(())
        }
    }

    /// A window matches memory exactly when loading it again gives the
    /// same bytes: after a load, after a store of changed bytes, and not
    /// after a change to one byte it holds; a change outside it does not
    /// count.
    #[test]
    fn matching_is_loading_and_comparing() {
        let layouts = [
            Layout::Pitch { pitch: 0x400 },
            Layout::BlockLinear(BlockLinear {
                block_height_log2: 2,
                block_depth_log2: 0,
            }),
        ];
        for layout in layouts {
            let surface = Surface {
                address: 0x10000,
                width: 200,
                height: 70,
                bytes_per_pixel: 4,
                layout,
            };
            let rect = Rect {
                x0: 20,
                y0: 9,
                x1: 150,
                y1: 50,
            };
            let memory = Written::default();
            let mut window = Window::load(&memory, surface, rect).unwrap();
            assert!(window.matches(&memory).unwrap());
            window.write_pixels(rect, &vec![0x5A; 130 * 41 * 4]);
            assert!(!window.matches(&memory).unwrap());
            window.store(&memory).unwrap();
            assert!(window.matches(&memory).unwrap());
            let mut touched = 0;
            for address in (0x10000..0x10000 + 0x400 * 70).step_by(97) {
                let before = Window::load(&memory, surface, rect).unwrap();
                let old = memory.0.lock().unwrap().get(&address).copied();
                memory.write(address, &[!old.unwrap_or(byte(address))]);
                let after = Window::load(&memory, surface, rect).unwrap();
                let held = before.data() != after.data();
                touched += usize::from(held);
                assert_eq!(
                    before.matches(&memory).unwrap(),
                    !held,
                    "{layout:?} {address:#x}"
                );
                assert!(after.matches(&memory).unwrap());
            }
            assert!(touched > 50, "{touched}");
        }
    }

    /// [`Written`], counting the accesses.
    #[derive(Default)]
    struct Counted {
        memory: Written,
        accesses: std::sync::atomic::AtomicUsize,
    }

    impl GpuMemory for Counted {
        fn read(&self, address: u64, buffer: &mut [u8]) -> Option<()> {
            self.accesses
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.memory.read(address, buffer)
        }

        fn write(&self, address: u64, data: &[u8]) -> Option<()> {
            self.accesses
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.memory.write(address, data)
        }
    }

    impl Counted {
        fn accesses(&self) -> usize {
            self.accesses.swap(0, std::sync::atomic::Ordering::Relaxed)
        }

        fn written(&self) -> std::collections::BTreeMap<u64, u8> {
            self.memory
                .0
                .lock()
                .unwrap()
                .iter()
                .map(|(a, b)| (*a, *b))
                .collect()
        }
    }

    /// Surfaces of both layouts, block heights 1 to 32 GOBs, several pixel
    /// sizes, and rectangles from one pixel to the whole surface.
    fn cases() -> Vec<(Surface, Rect)> {
        let mut cases = Vec::new();
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut below = move |n: u32| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % u64::from(n)) as u32
        };
        for case in 0..300 {
            let bpp = [1, 2, 4, 8, 16][case % 5];
            let (width, height) = (1 + below(300), 1 + below(150));
            let layout = if case % 3 == 0 {
                Layout::Pitch {
                    pitch: width * bpp + 4 * below(8),
                }
            } else {
                Layout::BlockLinear(BlockLinear {
                    block_height_log2: below(6),
                    block_depth_log2: 0,
                })
            };
            let surface = Surface {
                address: 0x10000,
                width,
                height,
                bytes_per_pixel: bpp,
                layout,
            };
            let rect = if case % 4 == 0 {
                Rect {
                    x0: 0,
                    y0: 0,
                    x1: width,
                    y1: height,
                }
            } else {
                let (x0, y0) = (below(width), below(height));
                Rect {
                    x0,
                    y0,
                    x1: x0 + 1 + below(width - x0),
                    y1: y0 + 1 + below(height - y0),
                }
            };
            cases.push((surface, rect));
        }
        cases
    }

    /// Loads, stores and partial stores in runs move the bytes the pieces
    /// moved one by one, with far fewer accesses for whole block rows.
    #[test]
    fn runs_move_what_the_pieces_move() {
        let mut merged = 0;
        for (surface, rect) in cases() {
            let memory = Counted::default();
            let mut window = Window::load(&memory, surface, rect).unwrap();
            let pieces = window.pieces(None, None);
            let mut expected = vec![0; window.data().len()];
            for (address, range) in &pieces {
                Pattern
                    .read(*address, &mut expected[range.clone()])
                    .unwrap();
            }
            assert!(window.data() == expected, "{surface:?} {rect:?}");
            merged += usize::from(memory.accesses() < pieces.len());
            // Changed bytes stored whole, and for a part.
            let changed: Vec<u8> = (0..window.data().len()).map(|i| (i * 7) as u8).collect();
            let part = Rect {
                x0: rect.x0 + (rect.x1 - rect.x0) / 3,
                y0: rect.y0 + (rect.y1 - rect.y0) / 2,
                ..rect
            };
            for only in [None, Some(part)] {
                let (runs, one_by_one) = (Counted::default(), Counted::default());
                window.data.copy_from_slice(&changed);
                match only {
                    None => window.store(&runs).unwrap(),
                    Some(part) => window.store_rect(&runs, part).unwrap(),
                }
                for (address, range) in window.pieces(only, None) {
                    one_by_one.write(address, &changed[range]).unwrap();
                }
                assert_eq!(runs.written(), one_by_one.written(), "{surface:?} {rect:?}");
            }
        }
        assert!(merged > 100, "{merged}");
    }

    /// [`Window::read_rect`] reads what a load and [`Window::read_pixels`]
    /// read.
    #[test]
    fn rect_reads_are_loads_and_pixel_reads() {
        for (surface, rect) in cases() {
            let row = (rect.x1 - rect.x0) as usize * surface.bytes_per_pixel as usize;
            let mut direct = vec![0; row * (rect.y1 - rect.y0) as usize];
            Window::read_rect(&Pattern, surface, rect, &mut direct).unwrap();
            let mut through = vec![0xA5; direct.len()];
            Window::load(&Pattern, surface, rect)
                .unwrap()
                .read_pixels(rect, &mut through);
            assert!(direct == through, "{surface:?} {rect:?}");
        }
    }

    /// A window loaded to be overwritten stores, once every pixel of its
    /// rectangle is written, what a loaded one stores; it reads less when
    /// the rectangle covers GOBs or rows whole.
    #[test]
    fn overwriting_loads_store_what_loads_store() {
        let mut skipped = 0;
        for (surface, rect) in cases() {
            let (loaded, overwritten) = (Counted::default(), Counted::default());
            let row = (rect.x1 - rect.x0) as usize * surface.bytes_per_pixel as usize;
            let pixels: Vec<u8> = (0..row * (rect.y1 - rect.y0) as usize)
                .map(|i| (i * 13 + 5) as u8)
                .collect();
            let mut window = Window::load(&loaded, surface, rect).unwrap();
            window.write_pixels(rect, &pixels);
            window.store(&loaded).unwrap();
            let mut window = Window::load_to_overwrite(&overwritten, surface, rect).unwrap();
            let reads = overwritten.accesses();
            window.write_pixels(rect, &pixels);
            window.store(&overwritten).unwrap();
            assert_eq!(
                loaded.written(),
                overwritten.written(),
                "{surface:?} {rect:?}"
            );
            skipped += usize::from(reads == 0);
        }
        assert!(skipped >= 20, "{skipped}");
    }

    /// [`Window::texel_offset`] and [`Window::band_range`] name the bytes
    /// [`Band::pixel`] holds.
    #[test]
    fn texel_offsets_are_band_pixels() {
        for (surface, rect) in cases() {
            let mut window = Window::load(&Pattern, surface, rect).unwrap();
            let bpp = surface.bytes_per_pixel as usize;
            let data = window.data().to_vec();
            for y in rect.y0..rect.y1 {
                let at = window.texel_offset(rect.x0, y);
                let range = window.band_range(y).unwrap();
                assert!(range.contains(&at));
                let offsets: Vec<usize> = (rect.x0..rect.x1)
                    .map(|x| window.texel_offset(x, y))
                    .collect();
                let band = window.band_mut(y).unwrap();
                for (x, at) in (rect.x0..rect.x1).zip(offsets) {
                    assert_eq!(
                        &data[at..at + bpp],
                        band.pixel(x, y),
                        "{surface:?} ({x}, {y})"
                    );
                }
            }
        }
    }
}
