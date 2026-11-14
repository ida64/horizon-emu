//! Filtering follows [public: OpenGL 4.6 §8.14-8.24; NVIDIA clb197tex.h TEXSAMP].
//! Depth comparison occurs at each tap before interpolation. Gather order is
//! (lower-left, lower-right, upper-right, upper-left) in increasing texture Y
//! [public: GLSL 4.60 textureGather].
use super::{Level, Texture, Tsc, Unmodeled, unsupported};
use crate::maxwell::sampling::{FetchRequest, GatherRequest, Lod, Query, SampleRequest, Target};

impl Texture {
    fn validate(&self, target: Target, array: bool) -> Result<(), Unmodeled> {
        if target != self.tic.target || array != self.tic.array {
            return Err(unsupported(
                "texture instruction target/array",
                format_args!(
                    "{target:?}/{array}, TIC {:?}/{}",
                    self.tic.target, self.tic.array
                ),
            ));
        }
        Ok(())
    }

    /// TLD is unfiltered; out-of-range coordinates, mip, or array layers return
    /// zero [public: D3D11.3 §5.9.2, ld instruction]. Offsets are texel units.
    pub fn fetch(&self, request: &FetchRequest) -> Result<[f32; 4], Unmodeled> {
        self.validate(request.target, request.array)?;
        if request.target == Target::Cube {
            return Err(unsupported("TLD cube target", 3));
        }
        if request.level < 0 {
            return Ok([0.; 4]);
        }
        let level = request.level as u32 + self.tic.base_level;
        if level > self.tic.max_level {
            return Ok([0.; 4]);
        }
        let m = &self.levels[level as usize];
        let p =
            std::array::from_fn(|i| i64::from(request.coords[i]) + i64::from(request.offset[i]));
        Ok(self.texel(m, p, request.layer).unwrap_or([0.; 4]))
    }

    fn texel(&self, m: &Level, p: [i64; 3], layer: u32) -> Option<[f32; 4]> {
        if layer >= self.tic.shape.layers
            || p[0] < 0
            || p[0] >= i64::from(m.width)
            || p[1] < 0
            || p[1] >= i64::from(m.height)
            || p[2] < 0
            || p[2] >= i64::from(m.depth)
        {
            return None;
        }
        Some(
            m.texels[(((layer as usize * m.depth as usize + p[2] as usize) * m.height as usize
                + p[1] as usize)
                * m.width as usize)
                + p[0] as usize],
        )
    }

    /// Returns clamped and unclamped isotropic LOD, respectively. The derivative
    /// footprint is max(length(dP/dx), length(dP/dy)) in base-level texels.
    /// Cube TMML requires projected derivatives; callers supply these, unlike
    /// sample requests whose direction derivatives are projected internally.
    pub fn lod(&self, tsc: &Tsc, target: Target, ddx: [f32; 3], ddy: [f32; 3]) -> [f32; 2] {
        let m = &self.levels[self.tic.base_level as usize];
        let dimensions = match target {
            Target::Texture1D => 1,
            Target::Texture3D => 3,
            _ => 2,
        };
        let scale = if self.tic.normalized && !tsc.force_unnormalized {
            [m.width as f32, m.height as f32, m.depth as f32]
        } else {
            [1.; 3]
        };
        let len = |d: [f32; 3]| {
            (0..dimensions)
                .map(|i| (d[i] * scale[i]).powi(2))
                .sum::<f32>()
                .sqrt()
        };
        let raw = len(ddx).max(len(ddy)).log2() + tsc.lod_bias;
        [self.clamp_lod(tsc, raw), raw]
    }

    /// Nearest, bilinear/trilinear spatial and mip filtering. Anisotropic min
    /// filtering reduces exactly to isotropic for explicit LOD, magnification,
    /// isotropic footprints, or anisotropy 1. Other footprints are refused rather
    /// than silently treated as isotropic: GM20B's tap placement is not documented.
    pub fn sample(&self, tsc: &Tsc, request: &SampleRequest) -> Result<[f32; 4], Unmodeled> {
        self.validate(request.target, request.array)?;
        let (coords, layer) = self.coordinates(request.coords, request.layer)?;
        let raw = match request.lod {
            Lod::Level(level) => level,
            Lod::Derivatives {
                mut ddx,
                mut ddy,
                bias,
            } => {
                if request.target == Target::Cube {
                    let (_, face) = cube(request.coords)?;
                    let (u, v, m) = face_coordinates(request.coords, face);
                    let derivative = |d: [f32; 3]| {
                        let (du, dv, dm) = face_coordinates(d, face);
                        [
                            0.5 * (du * m - u * dm) / (m * m),
                            0.5 * (dv * m - v * dm) / (m * m),
                            0.,
                        ]
                    };
                    ddx = derivative(ddx);
                    ddy = derivative(ddy);
                }
                let l = self.lod(tsc, request.target, ddx, ddy)[1] + bias;
                if tsc.min_filter == 3 && tsc.anisotropy > 1 && l > 0. {
                    let m = &self.levels[self.tic.base_level as usize];
                    let scale = [m.width as f32, m.height as f32, m.depth as f32];
                    let x = std::array::from_fn::<_, 3, _>(|i| ddx[i] * scale[i]);
                    let y = std::array::from_fn::<_, 3, _>(|i| ddy[i] * scale[i]);
                    let a: f32 = x.iter().map(|v| v * v).sum();
                    let b: f32 = y.iter().map(|v| v * v).sum();
                    let dot: f32 = x.iter().zip(y).map(|(x, y)| x * y).sum();
                    if (a - b).abs() > f32::EPSILON * a.max(b)
                        || dot.abs() > f32::EPSILON * a.max(b)
                    {
                        return Err(unsupported(
                            "TEX anisotropic non-isotropic footprint MAX_ANISOTROPY",
                            tsc.anisotropy,
                        ));
                    }
                }
                l
            }
        };
        if raw.is_nan() {
            return Err(unsupported("TEX LOD", raw));
        }
        let lod = self.clamp_lod(tsc, raw);
        let linear = if raw <= 0. {
            tsc.mag_linear
        } else {
            tsc.min_filter != 1
        };
        let level = if tsc.mip_filter == 1 {
            0
        } else if tsc.mip_filter == 2 {
            lod.round() as u32
        } else {
            lod.floor() as u32
        };
        let a = self.filtered(
            tsc,
            coords,
            layer,
            self.tic.base_level + level,
            request.offset,
            request.compare,
            linear,
        )?;
        if tsc.mip_filter != 3 || lod.fract() == 0. {
            return Ok(a);
        }
        let b = self.filtered(
            tsc,
            coords,
            layer,
            (self.tic.base_level + level + 1).min(self.tic.max_level),
            request.offset,
            request.compare,
            linear,
        )?;
        Ok(std::array::from_fn(|i| a[i] + (b[i] - a[i]) * lod.fract()))
    }

    /// TIC MIN_LOD_CLAMP is the minimum resident mip [public: envytools
    /// rnndb/graph/gm200_texture.xml]. It applies after sampler LOD clamps.
    fn clamp_lod(&self, tsc: &Tsc, lod: f32) -> f32 {
        let resident = ((self.tic.words[7] >> 12) & 4095) as f32 / 256.;
        lod.clamp(tsc.min_lod, tsc.max_lod)
            .max(resident)
            .clamp(0., (self.tic.max_level - self.tic.base_level) as f32)
    }

    fn coordinates(&self, p: [f32; 3], layer: u32) -> Result<([f32; 3], u32), Unmodeled> {
        if p.iter().any(|v| !v.is_finite()) {
            return Err(unsupported(
                "TEX nonfinite coordinates",
                format_args!("{p:?}"),
            ));
        }
        let layer = layer.min(if self.tic.target == Target::Cube {
            self.tic.shape.layers / 6 - 1
        } else {
            self.tic.shape.layers - 1
        });
        if self.tic.target == Target::Cube {
            let (p, face) = cube(p)?;
            Ok(([p[0], p[1], 0.], layer * 6 + face))
        } else {
            Ok((p, layer))
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn filtered(
        &self,
        tsc: &Tsc,
        coords: [f32; 3],
        layer: u32,
        level: u32,
        offset: [i32; 3],
        reference: Option<f32>,
        linear: bool,
    ) -> Result<[f32; 4], Unmodeled> {
        let m = &self.levels[level as usize];
        let dim = match self.tic.target {
            Target::Texture1D => 1,
            Target::Texture3D => 3,
            _ => 2,
        };
        let extents = [m.width, m.height, m.depth];
        let mut p = [0i64; 3];
        let mut frac = [0.; 3];
        for i in 0..dim {
            let scale = if self.tic.normalized && !tsc.force_unnormalized
                || self.tic.target == Target::Cube
            {
                extents[i] as f32
            } else {
                1.
            };
            let mut c = coords[i] * scale;
            if tsc.wrap[i] >= 5 {
                c = c.abs();
            }
            if matches!(tsc.wrap[i], 4 | 7) {
                c = c.clamp(0., extents[i] as f32);
            }
            c += offset[i] as f32 - if linear { 0.5 } else { 0. };
            p[i] = c.floor() as i64;
            frac[i] = c - c.floor();
        }
        if !linear {
            return self.tap(tsc, m, p, layer, reference);
        }
        let mut result = [0.; 4];
        for bits in 0..(1 << dim) {
            let mut q = p;
            let mut weight = 1.;
            for i in 0..dim {
                if bits & (1 << i) != 0 {
                    q[i] += 1;
                    weight *= frac[i];
                } else {
                    weight *= 1. - frac[i];
                }
            }
            if weight == 0. {
                continue;
            }
            let c = self.tap(tsc, m, q, layer, reference)?;
            for i in 0..4 {
                result[i] += weight * c[i];
            }
        }
        Ok(result)
    }

    fn tap(
        &self,
        tsc: &Tsc,
        m: &Level,
        mut p: [i64; 3],
        mut layer: u32,
        reference: Option<f32>,
    ) -> Result<[f32; 4], Unmodeled> {
        let mut border = false;
        if self.tic.target == Target::Cube
            && tsc.cube_filter >= 2
            && (p[0] < 0 || p[1] < 0 || p[0] >= i64::from(m.width) || p[1] >= i64::from(m.height))
        {
            let uv = [
                2. * (p[0] as f32 + 0.5) / m.width as f32 - 1.,
                2. * (p[1] as f32 + 0.5) / m.height as f32 - 1.,
            ];
            let dir = unproject(uv, layer % 6);
            let (uv, face) = cube(dir)?;
            layer = layer / 6 * 6 + face;
            p[0] = (uv[0] * m.width as f32).floor() as i64;
            p[1] = (uv[1] * m.height as f32).floor() as i64;
        }
        for (i, n) in [m.width, m.height, m.depth].into_iter().enumerate() {
            let mode = if self.tic.target == Target::Cube && tsc.cube_filter != 0 {
                2
            } else {
                tsc.wrap[i]
            };
            match wrap(p[i], i64::from(n), mode) {
                Some(v) => p[i] = v,
                None => border = true,
            }
        }
        let c = if border {
            if self.tic.border_size != 7 {
                return Err(unsupported(
                    "TIC texture-border BORDER_SIZE",
                    self.tic.border_size,
                ));
            }
            if tsc.srgb_conversion && self.tic.srgb {
                return Err(unsupported("TSC sRGB border color", 1));
            }
            tsc.border
        } else {
            self.texel(m, p, layer).unwrap_or([0.; 4])
        };
        if let Some(r) = reference {
            if !tsc.compare {
                return Err(unsupported("TEX DC with TSC DEPTH_COMPARE", 0));
            }
            let d = c[0];
            let pass = match tsc.compare_func {
                0 => false,
                1 => r < d,
                2 => r == d,
                3 => r <= d,
                4 => r > d,
                5 => r != d,
                6 => r >= d,
                7 => true,
                _ => unreachable!(),
            };
            Ok([if pass { 1. } else { 0. }; 4])
        } else {
            Ok(c)
        }
    }

    /// Four base-level taps, without interpolation, in GLSL gather order.
    pub fn gather(&self, tsc: &Tsc, request: &GatherRequest) -> Result<[f32; 4], Unmodeled> {
        self.validate(request.target, request.array)?;
        if !matches!(request.target, Target::Texture2D | Target::Cube) {
            return Err(unsupported(
                "TLD4 target",
                format_args!("{:?}", request.target),
            ));
        }
        if request.component > 3 {
            return Err(unsupported("TLD4 component", request.component));
        }
        let (coords, layer) = self.coordinates(request.coords, request.layer)?;
        let m = &self.levels[self.tic.base_level as usize];
        let normalized =
            self.tic.normalized && !tsc.force_unnormalized || request.target == Target::Cube;
        let mut p = [0i64; 3];
        for (i, n) in [m.width, m.height].into_iter().enumerate() {
            let mut c = coords[i] * if normalized { n as f32 } else { 1. };
            if tsc.wrap[i] >= 5 {
                c = c.abs();
            }
            if matches!(tsc.wrap[i], 4 | 7) {
                c = c.clamp(0., n as f32);
            }
            p[i] = (c - 0.5).floor() as i64 + i64::from(request.offset[i]);
        }
        let mut result = [0.; 4];
        for (i, (x, y)) in [(0, 1), (1, 1), (1, 0), (0, 0)].into_iter().enumerate() {
            result[i] = self.tap(tsc, m, [p[0] + x, p[1] + y, 0], layer, request.compare)?
                [request.component as usize];
        }
        Ok(result)
    }

    /// TXQ dimensions and visible mip count. Non-dimension modes are refused:
    /// the public translator documents only the sample-count component of type,
    /// not the other returned fields [public: Ryujinx InstEmitTexture.cs EmitTxq].
    pub fn query(&self, query: Query) -> Result<[u32; 4], Unmodeled> {
        match query {
            Query::Dimension { level } => {
                let count = self.tic.max_level - self.tic.base_level + 1;
                if level >= count {
                    return Ok([0, 0, 0, count]);
                }
                let m = &self.levels[(self.tic.base_level + level) as usize];
                let d = if self.tic.target == Target::Cube {
                    self.tic.shape.layers / 6
                } else if self.tic.array {
                    self.tic.shape.layers
                } else {
                    m.depth
                };
                Ok([m.width, m.height, d, count])
            }
            _ => Err(unsupported("TXQ mode", format_args!("{query:?}"))),
        }
    }
}

fn wrap(p: i64, n: i64, mode: u32) -> Option<i64> {
    match mode {
        0 => Some(p.rem_euclid(n)),
        1 => {
            let p = p.rem_euclid(2 * n);
            Some(if p < n { p } else { 2 * n - 1 - p })
        }
        2 | 5 => Some(p.clamp(0, n - 1)),
        3 | 4 | 6 | 7 => (0..n).contains(&p).then_some(p),
        _ => unreachable!(),
    }
}

/// Cube face equations [public: OpenGL 4.6 table 8.19].
fn cube(p: [f32; 3]) -> Result<([f32; 2], u32), Unmodeled> {
    let a = p.map(f32::abs);
    if a == [0.; 3] {
        return Err(unsupported("cube zero direction", 0));
    }
    let axis = if a[0] >= a[1] && a[0] >= a[2] {
        0
    } else if a[1] >= a[2] {
        1
    } else {
        2
    };
    let face = axis as u32 * 2 + u32::from(p[axis] < 0.);
    Ok((project(p, face), face))
}
fn face_coordinates(p: [f32; 3], face: u32) -> (f32, f32, f32) {
    match face {
        0 => (-p[2], -p[1], p[0]),
        1 => (p[2], -p[1], -p[0]),
        2 => (p[0], p[2], p[1]),
        3 => (p[0], -p[2], -p[1]),
        4 => (p[0], -p[1], p[2]),
        _ => (-p[0], -p[1], -p[2]),
    }
}
fn project(p: [f32; 3], face: u32) -> [f32; 2] {
    let (u, v, m) = face_coordinates(p, face);
    [0.5 * (u / m + 1.), 0.5 * (v / m + 1.)]
}
fn unproject(p: [f32; 2], face: u32) -> [f32; 3] {
    match face {
        0 => [1., -p[1], -p[0]],
        1 => [-1., -p[1], p[0]],
        2 => [p[0], 1., p[1]],
        3 => [p[0], -1., -p[1]],
        4 => [p[0], -p[1], 1.],
        _ => [-p[0], -p[1], -1.],
    }
}
