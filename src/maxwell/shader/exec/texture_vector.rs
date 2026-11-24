//! `TEX`, `TLD4`, `TMML`, `TXD` and `TXQ`: operand order and offsets.
//! Unsupported multisample and per-texel gather offsets cannot be
//! represented by TextureUnit.
use super::super::super::sampling::{
    FetchRequest, GatherRequest, Lod, Query, SampleRequest, Target,
};
use super::*;
fn offsets(p: u32, stride: u32, width: u32, n: usize) -> [i32; 3] {
    std::array::from_fn(|i| {
        if i < n {
            sext((p >> (i as u32 * stride)) & ((1 << width) - 1), width)
        } else {
            0
        }
    })
}
impl<const L: usize> Machine<'_, L> {
    fn store_texture(&mut self, l: usize, values: [u32; 4], mask: u32, d: u32, e: Option<u32>) {
        let mut index = 0;
        for (c, v) in values.into_iter().enumerate() {
            if mask & (1 << c) == 0 {
                continue;
            }
            let reg = if let Some(e) = e {
                if index < 2 {
                    if d == RZ { RZ } else { d + index }
                } else if e == RZ {
                    RZ
                } else {
                    e + index - 2
                }
            } else {
                d + index
            };
            self.set(l, reg, v);
            index += 1;
        }
    }
    pub(super) fn execute_texture_vector(&mut self, active: Mask) -> Result<bool, Unmodeled> {
        let op = self.op;
        let w = self.word;
        if matches!(op, Op::Tlds | Op::Tld4s) {
            self.scalar_fetch(active)?;
            return Ok(true);
        }
        if !matches!(
            op,
            Op::Tex
                | Op::TexB
                | Op::Tld
                | Op::TldB
                | Op::Tld4
                | Op::Tld4B
                | Op::Tmml
                | Op::TmmlB
                | Op::Txd
                | Op::TxdB
                | Op::Txq
                | Op::TxqB
        ) {
            return Ok(false);
        }
        let bindless = matches!(
            op,
            Op::TexB | Op::TldB | Op::Tld4B | Op::TmmlB | Op::TxdB | Op::TxqB
        );
        let query = matches!(op, Op::Txq | Op::TxqB);
        let gradient = matches!(op, Op::Txd | Op::TxdB);
        let fetch = matches!(op, Op::Tld | Op::TldB);
        let gather = matches!(op, Op::Tld4 | Op::Tld4B);
        let tmml = matches!(op, Op::Tmml | Op::TmmlB);
        let a = bits(w, 8, 8);
        let b = bits(w, 20, 8);
        let d = bits(w, 0, 8);
        let mask = bits(w, 31, 4);
        let dim = bits(w, 28, 3);
        let array = dim & 1 != 0;
        let (target, n) = match dim >> 1 {
            0 => (Target::Texture1D, 1),
            1 => (Target::Texture2D, 2),
            2 => (Target::Texture3D, 3),
            _ => (Target::Cube, 3),
        };
        if !query {
            self.require(dim != 5, "3D array")?;
        }
        if !query && !tmml {
            self.require(bits(w, 51, 3) == PT, "residency predicate")?;
        }
        if fetch {
            self.require(!bit(w, 50), "multisample")?;
            self.require(!bit(w, 54), "CL")?;
        }
        if gradient {
            self.require(!bit(w, 50), "LOD clamp")?;
        }
        if !query && !tmml && !fetch && !gradient {
            self.require(!bit(w, if bindless { 40 } else { 58 }), "LOD clamp")?;
        }
        let lod_mode = if fetch {
            if bit(w, 55) { 3 } else { 1 }
        } else {
            bits(w, if bindless { 37 } else { 55 }, 3)
        };
        if !query && !tmml && !gather && !gradient {
            self.require(matches!(lod_mode, 0 | 1 | 2 | 3 | 6 | 7), "LOD mode")?;
        }
        let offset_mode = if gather {
            bits(w, if bindless { 36 } else { 54 }, 2)
        } else if fetch || gradient {
            u32::from(bit(w, 35))
        } else {
            u32::from(bit(w, if bindless { 36 } else { 54 }))
        };
        if gather {
            self.require(offset_mode <= 1, "per-texel gather offsets")?;
        }
        let coordinates: [[f32; 3]; L] = std::array::from_fn(|l| {
            std::array::from_fn(|i| {
                if i < n {
                    self.rf(
                        l,
                        a + u32::from(if gradient { bindless } else { array }) + i as u32,
                    )
                } else {
                    0.0
                }
            })
        });
        let ddx = if L == 4 {
            std::array::from_fn(|i| coordinates[1][i] - coordinates[0][i])
        } else {
            [0.0; 3]
        };
        let ddy = if L == 4 {
            std::array::from_fn(|i| coordinates[2][i] - coordinates[0][i])
        } else {
            [0.0; 3]
        };
        let bound = if bindless {
            0
        } else {
            self.cbuf(self.env.texture_buffer_index, bits(w, 36, 13) * 4)?
        };
        self.each(active, |m, l| {
            let mut rb = b;
            let handle = if bindless {
                if query || gradient {
                    m.r(l, a)
                } else {
                    rb += 1;
                    m.r(l, b)
                }
            } else {
                bound
            };
            if query {
                let level = m.r(l, a + u32::from(bindless));
                let q = match bits(w, 22, 6) {
                    1 => Query::Dimension { level },
                    2 => Query::TextureType,
                    5 => Query::SamplePosition,
                    v => return m.fail(&format!("query {v}")),
                };
                let result = m.env.textures.query(handle, q)?;
                m.store_texture(l, result, mask, d, None);
                return Ok(());
            }
            if tmml {
                m.require(mask & 12 == 0, "LOD components Z/W")?;
                m.require(
                    !m.is_compute(),
                    "in a compute program (no quad derivatives)",
                )?;
                let values = m.env.textures.lod(handle, target, ddx, ddy)?;
                m.store_texture(
                    l,
                    [
                        (values[0] * 256.0) as i32 as u32,
                        (values[1] * 256.0) as i32 as u32,
                        0,
                        0,
                    ],
                    mask,
                    d,
                    None,
                );
                return Ok(());
            }
            let params = if gradient {
                m.r(l, a + u32::from(bindless) + n as u32)
            } else {
                0
            };
            let layer = if array {
                if gradient { params & 0xffff } else { m.r(l, a) }
            } else {
                0
            };
            let mut lod_value = 0;
            if !gradient && !gather && lod_mode > 1 {
                lod_value = m.r(l, rb);
                rb += 1;
            }
            let packed = if gradient {
                params >> 16
            } else if offset_mode != 0 {
                let p = m.r(l, rb);
                rb += 1;
                p
            } else {
                0
            };
            let offset = offsets(
                packed,
                if gather { 8 } else { 4 },
                if gather { 6 } else { 4 },
                n,
            );
            let compare = if !fetch && !gradient && bit(w, 50) {
                Some(m.rf(l, rb))
            } else {
                None
            };
            let values = if fetch {
                let base = a + u32::from(array);
                m.env.textures.fetch(
                    handle,
                    &FetchRequest {
                        target,
                        array,
                        coords: std::array::from_fn(|i| {
                            if i < n {
                                m.r(l, base + i as u32) as i32
                            } else {
                                0
                            }
                        }),
                        layer,
                        level: lod_value as i32,
                        offset,
                    },
                )?
            } else if gather {
                m.env.textures.gather(
                    handle,
                    &GatherRequest {
                        target,
                        array,
                        coords: coordinates[l],
                        layer,
                        offset,
                        component: bits(w, if bindless { 38 } else { 56 }, 2) as u8,
                        compare,
                    },
                )?
            } else {
                let lod = if gradient {
                    Lod::Derivatives {
                        ddx: std::array::from_fn(
                            |i| if i < n { m.rf(l, b + i as u32) } else { 0.0 },
                        ),
                        ddy: std::array::from_fn(|i| {
                            if i < n {
                                m.rf(l, b + n as u32 + i as u32)
                            } else {
                                0.0
                            }
                        }),
                        bias: 0.0,
                    }
                } else if matches!(lod_mode, 1 | 3 | 7) {
                    Lod::Level(f32::from_bits(lod_value))
                } else if L == 4 {
                    Lod::Derivatives {
                        ddx,
                        ddy,
                        bias: if matches!(lod_mode, 2 | 6) {
                            f32::from_bits(lod_value)
                        } else {
                            0.0
                        },
                    }
                } else {
                    // No derivatives outside a fragment program: the
                    // base level (see `tex.rs`).
                    Lod::Level(0.0)
                };
                m.env.textures.sample(
                    handle,
                    &SampleRequest {
                        target,
                        array,
                        coords: coordinates[l],
                        layer,
                        lod,
                        offset,
                        compare,
                    },
                )?
            };
            m.store_texture(l, values.map(f32::to_bits), mask, d, None);
            Ok(())
        })?;
        Ok(true)
    }
    fn scalar_fetch(&mut self, active: Mask) -> Result<(), Unmodeled> {
        let w = self.word;
        let a = bits(w, 8, 8);
        let b = bits(w, 20, 8);
        let d = bits(w, 0, 8);
        let e = bits(w, 28, 8);
        let gather = self.op == Op::Tld4s;
        let encoding = bits(w, 53, 4);
        let code = bits(w, 50, 3) as usize;
        let mask = if gather {
            15
        } else if e == RZ {
            [1, 2, 4, 8, 3, 9, 10, 12][code]
        } else {
            *[7, 11, 13, 14, 15]
                .get(code)
                .ok_or_else(|| self.unmodeled("component mask"))?
        };
        let handle = self.cbuf(self.env.texture_buffer_index, bits(w, 36, 13) * 4)?;
        if !gather {
            self.require(
                matches!(encoding, 0 | 1 | 2 | 4 | 5 | 7 | 8 | 12),
                "scalar target (multisample unsupported)",
            )?;
        }
        self.each(active, |m, l| {
            let values = if gather {
                let offset = bit(w, 51);
                let dc = bit(w, 50);
                let coords = if offset || dc {
                    [m.rf(l, a), m.rf(l, a + 1), 0.0]
                } else {
                    [m.rf(l, a), m.rf(l, b), 0.0]
                };
                let packed = if offset { m.r(l, b) } else { 0 };
                m.env.textures.gather(
                    handle,
                    &GatherRequest {
                        target: Target::Texture2D,
                        array: false,
                        coords,
                        layer: 0,
                        offset: offsets(packed, 8, 6, 2),
                        component: bits(w, 52, 2) as u8,
                        compare: if dc {
                            Some(m.rf(l, b + u32::from(offset)))
                        } else {
                            None
                        },
                    },
                )?
            } else {
                let (target, coords, layer, level, packed) = match encoding {
                    0 | 1 => (
                        Target::Texture1D,
                        [m.r(l, a) as i32, 0, 0],
                        0,
                        if encoding == 1 { m.r(l, b) as i32 } else { 0 },
                        0,
                    ),
                    2 => (
                        Target::Texture2D,
                        [m.r(l, a) as i32, m.r(l, b) as i32, 0],
                        0,
                        0,
                        0,
                    ),
                    4 | 5 | 12 => (
                        Target::Texture2D,
                        [m.r(l, a) as i32, m.r(l, a + 1) as i32, 0],
                        0,
                        if encoding == 4 { 0 } else { m.r(l, b) as i32 },
                        if encoding == 4 {
                            m.r(l, b)
                        } else if encoding == 12 {
                            m.r(l, b + 1)
                        } else {
                            0
                        },
                    ),
                    7 => (
                        Target::Texture3D,
                        [m.r(l, a) as i32, m.r(l, a + 1) as i32, m.r(l, b) as i32],
                        0,
                        0,
                        0,
                    ),
                    _ => (
                        Target::Texture2D,
                        [m.r(l, b) as i32, m.r(l, b + 1) as i32, 0],
                        m.r(l, a),
                        0,
                        0,
                    ),
                };
                m.env.textures.fetch(
                    handle,
                    &FetchRequest {
                        target,
                        array: encoding == 8,
                        coords,
                        layer,
                        level,
                        offset: offsets(packed, 4, 4, 2),
                    },
                )?
            };
            m.store_texture(l, values.map(f32::to_bits), mask, d, Some(e));
            Ok(())
        })
    }
}
