//! `TEXS`, `TLDS` and `TLD4S`: packed scalar operands and component maps.
//! NODEP affects scheduling only: synchronous CPU sampling has no
//! outstanding reads.
use super::super::super::sampling::{Lod, SampleRequest, Target};
use super::*;
impl<const L: usize> Machine<'_, L> {
    pub(super) fn execute_texture(&mut self, active: Mask) -> Result<bool, Unmodeled> {
        if self.op != Op::Texs {
            return Ok(false);
        }
        let w = self.word;
        let encoding = bits(w, 53, 4);
        self.require(encoding <= 13, "target encoding")?;
        let a = bits(w, 8, 8);
        let b = bits(w, 20, 8);
        let d = bits(w, 0, 8);
        let e = bits(w, 28, 8);
        let mask_code = bits(w, 50, 3) as usize;
        let mask = if e == RZ {
            [1u8, 2, 4, 8, 3, 9, 10, 12][mask_code]
        } else {
            *[7u8, 11, 13, 14, 15]
                .get(mask_code)
                .ok_or_else(|| self.unmodeled("component mask"))?
        };
        if encoding >= 3 {
            self.require(a == RZ || a.is_multiple_of(2), "unaligned coordinate pair")?;
        }
        if matches!(encoding, 5 | 9 | 13) {
            self.require(b == RZ || b.is_multiple_of(2), "unaligned operand pair")?;
        }
        if mask.count_ones() > 1 {
            self.require(
                d == RZ || d.is_multiple_of(2),
                "unaligned first destination",
            )?;
        }
        if mask.count_ones() > 3 {
            self.require(
                e == RZ || e.is_multiple_of(2),
                "unaligned second destination",
            )?;
        }
        let handle = self.cbuf(self.env.texture_buffer_index, bits(w, 36, 13) * 4)?;
        let target = match encoding {
            0 => Target::Texture1D,
            10 | 11 => Target::Texture3D,
            12 | 13 => Target::Cube,
            _ => Target::Texture2D,
        };
        let array = matches!(encoding, 7..=9);
        let coords: [[f32; 3]; L] = std::array::from_fn(|l| match encoding {
            0 => [self.rf(l, a), 0.0, 0.0],
            1 | 2 => [self.rf(l, a), self.rf(l, b), 0.0],
            3..=6 => [self.rf(l, a), self.rf(l, a + 1), 0.0],
            7..=9 => [self.rf(l, a + 1), self.rf(l, b), 0.0],
            _ => [self.rf(l, a), self.rf(l, a + 1), self.rf(l, b)],
        });
        // Implicit level of detail comes from the quad's derivatives in a
        // fragment program; other stages have none and sample the base
        // level \[recalled: GLSL 4.60 §8.9, implicit derivatives exist
        // only in fragment shaders, elsewhere the base level is used\].
        let implicit = matches!(encoding, 1 | 4 | 7 | 10 | 12);
        let implicit_lod = if L == 4 {
            Lod::Derivatives {
                ddx: std::array::from_fn(|i| coords[1][i] - coords[0][i]),
                ddy: std::array::from_fn(|i| coords[2][i] - coords[0][i]),
                bias: 0.0,
            }
        } else {
            Lod::Level(0.0)
        };
        self.each(active, |m, l| {
            let lod = if implicit {
                implicit_lod
            } else {
                Lod::Level(match encoding {
                    3 | 5 => m.rf(l, b),
                    13 => m.rf(l, b + 1),
                    _ => 0.0,
                })
            };
            let compare = match encoding {
                4 | 6 => Some(m.rf(l, b)),
                5 | 9 => Some(m.rf(l, b + 1)),
                _ => None,
            };
            let request = SampleRequest {
                target,
                array,
                coords: coords[l],
                layer: if array { m.r(l, a) & 0xffff } else { 0 },
                lod,
                offset: [0; 3],
                compare,
            };
            let sample = m.env.textures.sample(handle, &request)?;
            let mut index = 0;
            for (c, value) in sample.into_iter().enumerate() {
                if mask & (1 << c) != 0 {
                    let dest = if index < 2 {
                        if d == RZ { RZ } else { d + index }
                    } else if e == RZ {
                        RZ
                    } else {
                        e + index - 2
                    };
                    m.setf(l, dest, value);
                    index += 1;
                }
            }
            Ok(())
        })?;
        Ok(true)
    }
}
