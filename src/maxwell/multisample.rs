//! Multisampling: the sample modes of `SET_ANTI_ALIAS`, where a pixel's
//! samples lie, and how a multisampled surface holds them in memory.
//!
//! # Layout
//!
//! A multisampled render target, depth buffer or texture stores the
//! samples of a pixel as a block of `2^lx` by `2^ly` texels of an
//! ordinary surface `2^lx` times as wide and `2^ly` times as tall
//! (1X1: 1x1, 2X1: 2x1, 2X2: 2x2, 4X2: 4x2, 4X4: 4x4): pixel (x, y),
//! sample s is texel (x << lx | cx(s), y << ly | cy(s)). The render
//! target and depth buffer size registers (`SET_COLOR_TARGET_WIDTH`,
//! `SET_ZT_SIZE_A`) give the size of that surface, in texels; the
//! surface clip, scissors and viewports count pixels \[inference, from a
//! capture: a 2X1 target of width 2560 drawn with a 1280-pixel surface
//! clip and viewport\]. A texture header gives the size in pixels, with
//! the `MULTI_SAMPLE_COUNT` field.
//!
//! The block position of sample s takes the bits of s alternately for x
//! and y: s = 0..7 at (0,0), (1,0), (0,1), (1,1), (2,0), (3,0), (2,1),
//! (3,1). GM20B places every sample where
//! `SET_ANTI_ALIAS_SAMPLE_POSITIONS` says (see below), so storage by
//! index is taken to hold for all modes \[assumption; only which sample a
//! shader's fetch of sample index s reads depends on it\]. 4X4's samples
//! 8-15 continue the pattern \[inference\].
//!
//! # Sample positions
//!
//! GM20B's sample positions are programmable: `SET_ANTI_ALIAS_SAMPLE_POSITIONS(i)`
//! (0x11e0, four words) holds 16 slots of a 4-bit x and y in 1/16 pixel,
//! slot `p * n + s` being sample s of pixel p of a small pixel grid for n
//! samples [public: NVIDIA open-gpu-doc `clb197.h`]. NVN programs the
//! same positions for every pixel of the grid (2X1_D3D:
//! `0x44cc44cc`, sample 0 at (12/16, 12/16), sample 1 at (4/16, 4/16),
//! Direct3D's standard 2x pattern); positions that differ between pixels
//! are refused. One sample per pixel is at the pixel center (the 1X1 rules
//! of `raster`, which no register here changes).

use super::Unmodeled;
use super::surface::{Rect, Surface};

/// The samples of a pixel: how many, where they are stored and where
/// they are taken.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Samples {
    /// log2 of the samples per pixel in x and in y.
    pub log2: [u32; 2],
    /// Sample positions in the pixel, x and y in 1/16 pixel, by sample
    /// index (the first [`Samples::count`] are used).
    pub positions: [[u8; 2]; 16],
}

impl Samples {
    /// One sample, at the pixel center.
    pub const SINGLE: Samples = Samples {
        log2: [0, 0],
        positions: [[8, 8]; 16],
    };

    /// The mode of `SET_ANTI_ALIAS` SAMPLES `mode`, with every sample at
    /// the pixel center; `Err` names what is not modeled.
    pub fn from_mode(mode: u32) -> Result<Samples, &'static str> {
        let log2 = match mode {
            0 => [0, 0],
            // 2X1, 2X1_D3D.
            1 | 5 => [1, 0],
            // 2X2.
            2 => [1, 1],
            // 4X2, 4X2_D3D.
            3 | 4 => [2, 1],
            // 4X4.
            6 => [2, 2],
            // 2X2_VC_4, 2X2_VC_12, 4X2_VC_8, 4X2_VC_24: coverage samples
            // beyond the stored ones (virtual coverage), whose rules are
            // not public.
            8..=0xB => return Err("virtual coverage sample mode"),
            _ => return Err("sample mode"),
        };
        Ok(Samples {
            log2,
            positions: [[8, 8]; 16],
        })
    }

    /// Samples per pixel.
    pub fn count(&self) -> usize {
        1 << (self.log2[0] + self.log2[1])
    }

    pub fn is_single(&self) -> bool {
        self.log2 == [0, 0]
    }

    /// Takes the positions from the four `SET_ANTI_ALIAS_SAMPLE_POSITIONS`
    /// words; refused where they differ between the pixels of the grid.
    pub fn with_positions(mut self, words: [u32; 4]) -> Result<Samples, Unmodeled> {
        let slot = |i: usize| {
            let byte = (words[i / 4] >> (8 * (i % 4))) as u8;
            [byte & 0xF, byte >> 4]
        };
        let n = self.count();
        for s in 0..n {
            self.positions[s] = slot(s);
            if (n..16).step_by(n).any(|p| slot(p + s) != slot(s)) {
                return Err(Unmodeled(format!(
                    "3D: sample positions that differ between pixels ({words:#x?})"
                )));
            }
        }
        Ok(self)
    }

    /// Where sample `s` of a pixel is stored: its texel in the pixel's
    /// block.
    #[inline]
    pub fn cell(&self, s: usize) -> [u32; 2] {
        let s = s as u32;
        [(s & 1) | (s >> 1 & 2), (s >> 1 & 1) | (s >> 2 & 2)]
    }

    /// The texels of a pixel rectangle.
    pub fn texels(&self, pixels: Rect) -> Rect {
        let [lx, ly] = self.log2;
        Rect {
            x0: pixels.x0 << lx,
            y0: pixels.y0 << ly,
            x1: pixels.x1 << lx,
            y1: pixels.y1 << ly,
        }
    }

    /// The pixels of a surface of this mode (whose size is in texels).
    pub fn pixels(&self, surface: &Surface) -> Rect {
        Rect {
            x0: 0,
            y0: 0,
            x1: surface.width >> self.log2[0],
            y1: surface.height >> self.log2[1],
        }
    }

    /// The position of sample `s` as an offset from the pixel's corner,
    /// in pixels.
    pub fn offset(&self, s: usize) -> [f32; 2] {
        self.positions[s].map(|v| f32::from(v) / 16.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modes_cells_and_positions() {
        let two = Samples::from_mode(5).unwrap();
        assert_eq!((two.log2, two.count()), ([1, 0], 2));
        assert_eq!([two.cell(0), two.cell(1)], [[0, 0], [1, 0]]);
        let eight = Samples::from_mode(4).unwrap();
        let cells: Vec<[u32; 2]> = (0..8).map(|s| eight.cell(s)).collect();
        assert_eq!(
            cells,
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [1, 1],
                [2, 0],
                [3, 0],
                [2, 1],
                [3, 1]
            ]
        );
        // Every sample of 4X4 has its own texel of the 4x4 block.
        let sixteen = Samples::from_mode(6).unwrap();
        let mut seen = [[false; 4]; 4];
        for s in 0..16 {
            let [x, y] = sixteen.cell(s);
            assert!(!seen[y as usize][x as usize]);
            seen[y as usize][x as usize] = true;
        }
        assert!(Samples::from_mode(9).is_err());
        // The 2X1_D3D positions NVN programs.
        let two = two.with_positions([0x44cc44cc; 4]).unwrap();
        assert_eq!(two.offset(0), [0.75, 0.75]);
        assert_eq!(two.offset(1), [0.25, 0.25]);
        // Positions that change from pixel to pixel are refused.
        assert!(
            Samples::from_mode(5)
                .unwrap()
                .with_positions([0x44cc44cc, 0x44cc44cc, 0x44cc44cc, 0xcc44cc44])
                .is_err()
        );
        let surface = Surface {
            address: 0,
            width: 2560,
            height: 720,
            bytes_per_pixel: 4,
            layout: super::super::surface::Layout::Pitch { pitch: 10240 },
        };
        assert_eq!(two.pixels(&surface).x1, 1280);
        let texels = two.texels(Rect {
            x0: 3,
            y0: 4,
            x1: 5,
            y1: 6,
        });
        assert_eq!((texels.x0, texels.x1, texels.y0, texels.y1), (6, 10, 4, 6));
    }
}
