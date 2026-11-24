//! Inputs of a pixel program besides registers: interpolated attributes
//! (`IPA`), indexed constant buffer reads (`LDC`), texture samples
//! (`TEXS`, `TEX`) and the registers of the quad's other lanes (`SHFL`),
//! as the interpreter executes them (`exec.rs`, `exec/tex.rs`,
//! `exec/texture_vector.rs`).
//!
//! Textures are bound per handle offset: the instruction's 13-bit index
//! times 4 is the byte offset of the 32-bit handle in the bound-texture
//! constant buffer (`TEXTURE_BUFFER_INDEX`); the draw reads the handle
//! there, as the interpreter does, and binds the texture and sampler it
//! names at the slot [`Translated::textures`](super::Translated) lists the
//! offset in, of the kind the instruction samples
//! ([`TextureKind`](super::TextureKind): 2D, 2D array, 3D or cube; a
//! handle sampled as two kinds takes two slots). Depth-compare samples
//! are not translated. The renderer binds a texture of that type or
//! leaves the draw to the software renderer
//! (`metal::format::check_texture`).
//!
//! A cube is sampled as Metal samples cubes, filtering across the faces'
//! edges; the interpreter does so only for the sampler's seamless cube
//! filter and clamps to the face otherwise, so texels within half a
//! texel of an edge can differ [known difference].
//!
//! A handle that names no texture (an unset slot of an uber-shader, whose
//! sample sits on a branch the draw may not take) is not refused when the
//! draw is set up: the interpreter resolves a handle only when a sample
//! executes. The draw binds a placeholder and marks the slot in the fault
//! words (buffer 21, see [`Translated::source`](super::Translated::source));
//! every sample checks its slot's mark first and records an executed
//! sample of an unbound slot, which the renderer turns into the
//! interpreter's refusal before any result reaches guest memory. Known
//! difference: a sample executed only by helper lanes (outside the
//! primitive, which the interpreter refuses as well) is recorded only if
//! Metal performs the device-memory writes of helper threads. The Metal
//! Shading Language specification found here does not say; SPIRV-Cross
//! guards stores after `discard_fragment` with an explicit helper check
//! (`spirv_msl.cpp`, `needs_frag_discard_checks`), which suggests they are
//! performed [unverified].

use super::super::exec::{bit, bits, fswzadd_factors, quad_shuffle_lane, sext};
use super::{PT, RZ, SLOTS, TextureKind, Writer};

impl Writer<'_> {
    /// `IPA`: the attribute's value at the pixel center or, CENTROID, at
    /// the centroid (see `exec::Machine::ipa`), optionally multiplied by
    /// a register, optionally saturated. The value is the macro
    /// `IPA{k}` (`IPAC{k}` for CENTROID) of plane `k`, which
    /// `Translated::source` defines for the draw's way of interpolating.
    pub(super) fn ipa(&mut self) -> Result<(), String> {
        let w = self.word;
        let mode = bits(w, 54, 2);
        self.require(mode <= 2, "mode SC")?;
        if mode == 2 {
            // `exec::Machine::ipa`: a constant input's plane has no
            // slope, so its value anywhere is the provoking vertex's.
            self.require(
                self.program
                    .header
                    .ps_interpolation(bits(w, 28, 10) as usize / 4)
                    == super::super::Interpolation::Flat,
                "mode CONSTANT of an interpolated input",
            )?;
        }
        self.require(!bit(w, 38), "IDX")?;
        self.require(bits(w, 52, 2) <= 1, "sample mode OFFSET")?;
        self.require(bits(w, 47, 4) == PT, "predicate operand")?;
        self.require(bits(w, 39, 8) == RZ, "sample-offset register")?;
        let attribute = bits(w, 28, 10);
        self.require(attribute.is_multiple_of(4), "unaligned attribute")?;
        self.require(attribute != 0x3fc, "front-facing attribute")?;
        let word = attribute / 4;
        let plane = match self.ipa_words.iter().position(|&x| x == word) {
            Some(index) => index,
            None => {
                self.ipa_words.push(word);
                self.ipa_words.len() - 1
            }
        };
        let (rd, rb) = (bits(w, 0, 8), bits(w, 20, 8));
        let mut value = if bits(w, 52, 2) == 1 {
            format!("IPAC{plane}")
        } else {
            format!("IPA{plane}")
        };
        if mode == 1 {
            let b = self.read(rb);
            value = format!("({value} * as_type<float>({b}))");
        }
        if bit(w, 51) {
            value = format!("sat({value})");
        }
        self.set(rd, &format!("as_type<uint>({value})"));
        Ok(())
    }

    /// `SHFL` with immediate lane and mask operands (`exec::Machine::shfl`):
    /// the quad lane each lane reads is fixed by them
    /// (`exec::quad_shuffle_lane`), and read with `quad_shuffle`, whose
    /// lane `(x & 1) | (y & 1) << 1` of a fragment quad is the
    /// interpreter's lane order [the `shfl_matches_interpreter` test
    /// checks it on the GPU].
    pub(super) fn shfl(&mut self) -> Result<(), String> {
        let w = self.word;
        self.require(bit(w, 28) && bit(w, 29), "lane or mask in a register")?;
        let (mode, b, c) = (bits(w, 30, 2), bits(w, 20, 5), bits(w, 34, 13));
        let mut sources = Vec::with_capacity(4);
        for lane in 0..4 {
            let Some(picked) = quad_shuffle_lane(mode, lane, b, c) else {
                return self.refuse("across the quads of a warp");
            };
            // 4: out of range, the lane's own value.
            sources.push(format!("{}u", picked.unwrap_or(4)));
        }
        let a = self.read(bits(w, 8, 8));
        let (rd, pd) = (bits(w, 0, 8), bits(w, 48, 3));
        self.line(&format!(
            "{{ const uint shfl_q = (uint(px) & 1u) | ((uint(py) & 1u) << 1u); const uint shfl_s = shfl_q == 0u ? {} : shfl_q == 1u ? {} : shfl_q == 2u ? {} : {}; const uint shfl_v = quad_shuffle({a}, ushort(shfl_s < 4u ? shfl_s : shfl_q));",
            sources[0], sources[1], sources[2], sources[3]
        ));
        self.set(rd, "shfl_v");
        self.set_pred(pd, "shfl_s < 4u");
        self.line("}");
        Ok(())
    }

    /// `FSWZADD` (`exec::Machine::fswzadd`): each lane's factors by its
    /// quad lane, numbered as `shfl`'s.
    pub(super) fn fswzadd(&mut self) -> Result<(), String> {
        let w = self.word;
        self.require(!bit(w, 38), "NDV")?;
        self.require(bits(w, 39, 2) == 0, "rounding mode")?;
        self.require(!bit(w, 47), "condition code write")?;
        let ftz = |value: &str| {
            if bit(w, 44) {
                format!("ftz({value})")
            } else {
                value.to_string()
            }
        };
        let a = ftz(&format!("as_type<float>({})", self.read(bits(w, 8, 8))));
        let b = ftz(&format!("as_type<float>({})", self.read(bits(w, 20, 8))));
        let factors = |i: usize| {
            (0..4)
                .map(|lane| {
                    let pair = fswzadd_factors((bits(w, 28, 8) >> (2 * lane)) & 3);
                    format!("{:?}f", if i == 0 { pair.0 } else { pair.1 })
                })
                .collect::<Vec<_>>()
        };
        let (ka, kb) = (factors(0), factors(1));
        self.line(&format!(
            "{{ const uint fswz_q = (uint(px) & 1u) | ((uint(py) & 1u) << 1u); const float fswz_ka = fswz_q == 0u ? {} : fswz_q == 1u ? {} : fswz_q == 2u ? {} : {}; const float fswz_kb = fswz_q == 0u ? {} : fswz_q == 1u ? {} : fswz_q == 2u ? {} : {};",
            ka[0], ka[1], ka[2], ka[3], kb[0], kb[1], kb[2], kb[3]
        ));
        let value = ftz(&format!("{a} * fswz_ka + {b} * fswz_kb"));
        self.set(bits(w, 0, 8), &format!("as_type<uint>({value})"));
        self.line("}");
        Ok(())
    }

    /// `LDC`: dest 0-7, address register 8-15, offset 20-35 (signed),
    /// slot 36-40, address mode 44-45, size 48-50.
    pub(super) fn ldc(&mut self) -> Result<(), String> {
        let w = self.word;
        self.require(bits(w, 44, 2) == 0, "address mode IL/IS/ISL")?;
        let size = bits(w, 48, 3);
        let slot = bits(w, 36, 5);
        let (rd, ra) = (bits(w, 0, 8), bits(w, 8, 8));
        self.require(size <= 5, &format!("size {size}"))?;
        self.require(
            size != 5 || rd % 2 == 0 || rd == RZ,
            "B64 to an odd register",
        )?;
        self.require(slot < SLOTS, &format!("constant buffer slot {slot}"))?;
        self.indirect.insert(slot);
        let offset = sext(bits(w, 20, 16), 16) as u32;
        let a = self.read(ra);
        self.line(&format!(
            "{{ uint address = {a} + {offset:#x}u; const device uint* cb = cb{slot}; uint size = cb_size[{slot}];"
        ));
        match size {
            0 => self.line("uint v = cb_bytes(cb, size, address, 1u);"),
            1 => self.line(
                "uint v = as_type<uint>(as_type<int>(cb_bytes(cb, size, address, 1u) << 24) >> 24);",
            ),
            2 => self.line("uint v = cb_bytes(cb, size, address, 2u);"),
            3 => self.line(
                "uint v = as_type<uint>(as_type<int>(cb_bytes(cb, size, address, 2u) << 16) >> 16);",
            ),
            4 => self.line("uint v = cb_word(cb, size, address);"),
            _ => {
                self.line("bool ok = (address & 7u) == 0u && size >= 8u && address <= size - 8u;");
                self.line("uint v = ok ? cb[address >> 2] : 0u; uint high = ok ? cb[(address >> 2) + 1u] : 0u;");
            }
        }
        self.set(rd, "v");
        if size == 5 && rd != RZ {
            self.set(rd + 1, "high");
        }
        self.line("}");
        Ok(())
    }

    /// The texture/sampler slot of the handle at `handle` (byte offset in
    /// the bound-texture buffer) sampled as `kind`.
    fn texture_slot(&mut self, handle: u32, implicit: bool, kind: TextureKind) -> usize {
        let found = (0..self.textures.len())
            .find(|&slot| self.textures[slot] == handle && self.kinds[slot] == kind);
        let slot = match found {
            Some(slot) => slot,
            None => {
                self.textures.push(handle);
                self.kinds.push(kind);
                self.implicit_lod.push(false);
                self.compare.push(false);
                self.textures.len() - 1
            }
        };
        self.implicit_lod[slot] |= implicit;
        slot
    }

    /// The depth comparison of texture `slot` (2D or 2D array) at
    /// `coordinates` with the reference in register `reference`, at the
    /// level of detail `lod` (before the sampler's clamp): every tap
    /// compared, then filtered (`compare_sample`), by the sampler's
    /// function, filters and range of levels the draw gives (`CMP_CFG`,
    /// `CMP_MIN`, `CMP_MAX`).
    fn compare_call(
        &mut self,
        slot: usize,
        coordinates: &str,
        layer: Option<&str>,
        reference: u32,
        lod: &str,
    ) -> String {
        self.compare[slot] = true;
        let reference = self.read(reference);
        let layer = match layer {
            Some(layer) => format!("min({layer}, uint(t{slot}.get_array_size()) - 1u), "),
            None => String::new(),
        };
        format!(
            "compare_sample(t{slot}, s{slot}, {coordinates}, {layer}as_type<float>({reference}), CMP_CFG({slot}u), {lod}, CMP_MIN({slot}u), CMP_MAX({slot}u))"
        )
    }

    /// The sample of texture `slot` at `coordinates` (two components, or
    /// three for a cube or a volume), of layer `layer` of an array
    /// (clamped to its last, as the interpreter's texture unit does),
    /// with the level-of-detail option `option`.
    fn sample_call(slot: usize, coordinates: &str, layer: Option<&str>, option: &str) -> String {
        match layer {
            Some(layer) => format!(
                "t{slot}.sample(s{slot}, {coordinates}, min({layer}, uint(t{slot}.get_array_size()) - 1u), {option})"
            ),
            None => format!("t{slot}.sample(s{slot}, {coordinates}, {option})"),
        }
    }

    /// Records, before a sample of texture `slot`, that an unbound slot
    /// was sampled (see the module documentation).
    fn note_sample(&mut self, slot: usize) {
        self.line(&format!(
            "if (atomic_load_explicit(&texture_faults[{}], memory_order_relaxed) != 0u) {{ atomic_store_explicit(&texture_faults[{}], 1u, memory_order_relaxed); }}",
            2 * slot,
            2 * slot + 1
        ));
    }

    /// `TEXS`: 2D with implicit LOD (encoding 1), level zero (2) or an
    /// explicit level (3), the array forms of the first two (7, 8), 3D
    /// with implicit LOD or level zero (10, 11) and cube with implicit
    /// LOD or an explicit level (12, 13); destinations as `exec/tex.rs`.
    pub(super) fn texs(&mut self) -> Result<(), String> {
        let w = self.word;
        let encoding = bits(w, 53, 4);
        let kind = match encoding {
            1..=6 => TextureKind::Flat,
            7..=9 => TextureKind::Array,
            10 | 11 => TextureKind::Volume,
            12 | 13 => TextureKind::Cube,
            0 => return self.refuse("1D target"),
            _ => return self.refuse("target encoding"),
        };
        let a = bits(w, 8, 8);
        let b = bits(w, 20, 8);
        let d = bits(w, 0, 8);
        let e = bits(w, 28, 8);
        let mask_code = bits(w, 50, 3) as usize;
        let mask = if e == RZ {
            [1u8, 2, 4, 8, 3, 9, 10, 12][mask_code]
        } else {
            match [7u8, 11, 13, 14, 15].get(mask_code) {
                Some(&mask) => mask,
                None => return self.refuse("component mask"),
            }
        };
        if encoding >= 3 {
            self.require(a == RZ || a.is_multiple_of(2), "unaligned coordinate pair")?;
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
        let implicit = matches!(encoding, 1 | 4 | 7 | 10 | 12);
        let slot = self.texture_slot(bits(w, 36, 13) * 4, implicit, kind);
        let float = |writer: &mut Self, register: u32| {
            format!("as_type<float>({})", writer.read(register))
        };
        let coordinates = match encoding {
            1 | 2 => format!("float2({}, {})", float(self, a), float(self, b)),
            3..=6 => format!("float2({}, {})", float(self, a), float(self, a + 1)),
            // Array: layer in `a`, coordinates in `a + 1` and `b`.
            7..=9 => format!("float2({}, {})", float(self, a + 1), float(self, b)),
            _ => format!(
                "float3({}, {}, {})",
                float(self, a),
                float(self, a + 1),
                float(self, b)
            ),
        };
        let layer = (kind == TextureKind::Array).then(|| format!("({} & 0xffffu)", self.read(a)));
        let option = match encoding {
            1 | 7 | 10 | 12 => format!("bias(lod_bias[{slot}])"),
            3 => format!("level({})", float(self, b)),
            13 => format!("level({})", float(self, b + 1)),
            _ => "level(0.0f)".into(),
        };
        // The level of detail a comparison takes: the quad's (with the
        // sampler's bias), the register's, or zero.
        let implicit_lod =
            format!("(t{slot}.calculate_unclamped_lod(s{slot}, {coordinates}) + lod_bias[{slot}])");
        let sample = match encoding {
            4 => self.compare_call(slot, &coordinates, None, b, &implicit_lod),
            6 => self.compare_call(slot, &coordinates, None, b, "0.0f"),
            5 => {
                let lod = float(self, b);
                self.compare_call(slot, &coordinates, None, b + 1, &lod)
            }
            9 => self.compare_call(slot, &coordinates, layer.as_deref(), b + 1, "0.0f"),
            _ => Self::sample_call(slot, &coordinates, layer.as_deref(), &option),
        };
        self.note_sample(slot);
        self.line(&format!("{{ float4 v = {sample};"));
        let mut index = 0;
        for c in 0..4 {
            if mask & (1 << c) == 0 {
                continue;
            }
            let dest = if index < 2 {
                if d == RZ { RZ } else { d + index }
            } else if e == RZ {
                RZ
            } else {
                e + index - 2
            };
            let component = ["x", "y", "z", "w"][c];
            self.set(dest, &format!("as_type<uint>(v.{component})"));
            index += 1;
        }
        self.line("}");
        Ok(())
    }

    /// `TEX` (bound handle) on a 2D, 2D array, 3D or cube texture:
    /// implicit LOD (LOD mode 0), level zero (1), implicit with a bias
    /// register (2, 6) or an explicit level register (3, 7).
    pub(super) fn tex(&mut self) -> Result<(), String> {
        let w = self.word;
        let dim = bits(w, 28, 3);
        let kind = match dim {
            2 => TextureKind::Flat,
            3 => TextureKind::Array,
            4 => TextureKind::Volume,
            6 => TextureKind::Cube,
            0 | 1 => return self.refuse("1D target"),
            5 => return self.refuse("3D array"),
            _ => return self.refuse("cube array target"),
        };
        self.require(bits(w, 51, 3) == PT, "residency predicate")?;
        self.require(!bit(w, 58), "LOD clamp")?;
        let lod_mode = bits(w, 55, 3);
        self.require(matches!(lod_mode, 0 | 1 | 2 | 3 | 6 | 7), "LOD mode")?;
        self.require(!bit(w, 54), "AOFFI (texel offsets)")?;
        let compare = bit(w, 50);
        self.require(
            !compare || matches!(kind, TextureKind::Flat | TextureKind::Array),
            "depth compare of a cube or 3D texture",
        )?;
        let (a, b, d) = (bits(w, 8, 8), bits(w, 20, 8), bits(w, 0, 8));
        let mask = bits(w, 31, 4);
        let slot = self.texture_slot(bits(w, 36, 13) * 4, matches!(lod_mode, 0 | 2 | 6), kind);
        // Array: the layer register comes first.
        let first = a + u32::from(kind == TextureKind::Array);
        let s = self.read(first);
        let t = self.read(first + 1);
        let coordinates = if matches!(kind, TextureKind::Volume | TextureKind::Cube) {
            let r = self.read(first + 2);
            format!("float3(as_type<float>({s}), as_type<float>({t}), as_type<float>({r}))")
        } else {
            format!("float2(as_type<float>({s}), as_type<float>({t}))")
        };
        let layer = (kind == TextureKind::Array).then(|| self.read(a));
        let option = match lod_mode {
            0 => format!("bias(lod_bias[{slot}])"),
            1 => "level(0.0f)".into(),
            2 | 6 => format!("bias(lod_bias[{slot}] + as_type<float>({}))", self.read(b)),
            _ => format!("level(as_type<float>({}))", self.read(b)),
        };
        let sample = if compare {
            let lod = match lod_mode {
                0 => format!(
                    "(t{slot}.calculate_unclamped_lod(s{slot}, {coordinates}) + lod_bias[{slot}])"
                ),
                1 => "0.0f".to_string(),
                2 | 6 => format!(
                    "(t{slot}.calculate_unclamped_lod(s{slot}, {coordinates}) + lod_bias[{slot}] + as_type<float>({}))",
                    self.read(b)
                ),
                _ => format!("as_type<float>({})", self.read(b)),
            };
            // The reference follows the level or bias register.
            self.compare_call(
                slot,
                &coordinates,
                layer.as_deref(),
                b + u32::from(lod_mode > 1),
                &lod,
            )
        } else {
            Self::sample_call(slot, &coordinates, layer.as_deref(), &option)
        };
        self.note_sample(slot);
        self.line(&format!("{{ float4 v = {sample};"));
        let mut index = 0;
        for c in 0..4 {
            if mask & (1 << c) != 0 {
                let component = ["x", "y", "z", "w"][c];
                self.set(d + index, &format!("as_type<uint>(v.{component})"));
                index += 1;
            }
        }
        self.line("}");
        Ok(())
    }

    /// `TMML` (bound handle): the level of detail a sample at the
    /// coordinates would take, clamped to the texture's levels and
    /// unclamped, each times 256 as an integer
    /// (`exec/texture_vector.rs`). The level is Metal's, of its own
    /// footprint estimate, plus the sampler's bias [known difference:
    /// the interpreter's is the larger of the two derivatives'
    /// lengths, clamped by the sampler's range too].
    pub(super) fn tmml(&mut self) -> Result<(), String> {
        let w = self.word;
        let dim = bits(w, 28, 3);
        let kind = match dim {
            2 => TextureKind::Flat,
            3 => TextureKind::Array,
            4 => TextureKind::Volume,
            6 => TextureKind::Cube,
            _ => return self.refuse("target"),
        };
        let (a, d) = (bits(w, 8, 8), bits(w, 0, 8));
        let mask = bits(w, 31, 4);
        self.require(mask & 12 == 0, "LOD components Z/W")?;
        let slot = self.texture_slot(bits(w, 36, 13) * 4, true, kind);
        let first = a + u32::from(kind == TextureKind::Array);
        let s = self.read(first);
        let t = self.read(first + 1);
        let coordinates = if matches!(kind, TextureKind::Volume | TextureKind::Cube) {
            let r = self.read(first + 2);
            format!("float3(as_type<float>({s}), as_type<float>({t}), as_type<float>({r}))")
        } else {
            format!("float2(as_type<float>({s}), as_type<float>({t}))")
        };
        self.note_sample(slot);
        self.line(&format!(
            "{{ const float raw = t{slot}.calculate_unclamped_lod(s{slot}, {coordinates}) + lod_bias[{slot}]; const float top = float(t{slot}.get_num_mip_levels()) - 1.0f; float4 v = float4(as_type<float>(as_type<uint>(int(clamp(raw, 0.0f, top) * 256.0f))), as_type<float>(as_type<uint>(int(raw * 256.0f))), 0.0f, 0.0f);"
        ));
        self.store_sample(mask, d, None);
        self.line("}");
        Ok(())
    }

    /// Stores the components of `float4 v` that `mask` selects: to the
    /// registers from `d` on, or, with the second destination of the
    /// scalar forms, two to `d` and the rest to `e`.
    fn store_sample(&mut self, mask: u32, d: u32, e: Option<u32>) {
        let mut index = 0;
        for c in 0..4 {
            if mask & (1 << c) == 0 {
                continue;
            }
            let dest = match e {
                None => d + index,
                Some(_) if index < 2 => {
                    if d == RZ {
                        RZ
                    } else {
                        d + index
                    }
                }
                Some(e) if e == RZ => RZ,
                Some(e) => e + index - 2,
            };
            let component = ["x", "y", "z", "w"][c];
            self.set(dest, &format!("as_type<uint>(v.{component})"));
            index += 1;
        }
    }

    /// The four texels of a gather: `float4 v` from texture `slot` at
    /// the coordinates in registers `s` and `t`, moved by the offsets
    /// packed in `packed` (6 bits each, at 0 and 8; none: no offset).
    /// Metal returns them in the interpreter's order
    /// (`texture/filtering.rs` `gather`).
    fn gather(&mut self, slot: usize, s: u32, t: u32, packed: Option<u32>, component: u32) {
        let (s, t) = (self.read(s), self.read(t));
        let offset = match packed {
            Some(register) => {
                let p = self.read(register);
                format!("int2(as_type<int>({p} << 26) >> 26, as_type<int>({p} << 18) >> 26)")
            }
            None => "int2(0)".into(),
        };
        self.note_sample(slot);
        self.line(&format!(
            "{{ float4 v = t{slot}.gather(s{slot}, float2(as_type<float>({s}), as_type<float>({t})), {offset}, component::{});",
            ["x", "y", "z", "w"][component as usize]
        ));
    }

    /// `TLD4` (bound handle) on a 2D texture, with or without offsets
    /// in a register; no depth compare (`exec/texture_vector.rs`).
    pub(super) fn tld4(&mut self) -> Result<(), String> {
        let w = self.word;
        let dim = bits(w, 28, 3);
        self.require(dim == 2, "target other than 2D")?;
        self.require(bits(w, 51, 3) == PT, "residency predicate")?;
        self.require(!bit(w, 58), "LOD clamp")?;
        let offset_mode = bits(w, 54, 2);
        self.require(offset_mode <= 1, "per-texel gather offsets")?;
        self.require(!bit(w, 50), "depth compare")?;
        let (a, b, d) = (bits(w, 8, 8), bits(w, 20, 8), bits(w, 0, 8));
        let slot = self.texture_slot(bits(w, 36, 13) * 4, false, TextureKind::Flat);
        let first = a;
        self.gather(
            slot,
            first,
            first + 1,
            (offset_mode == 1).then_some(b),
            bits(w, 56, 2),
        );
        self.store_sample(bits(w, 31, 4), d, None);
        self.line("}");
        Ok(())
    }

    /// `TLD4S`: the scalar form, all four texels to two register pairs.
    pub(super) fn tld4s(&mut self) -> Result<(), String> {
        let w = self.word;
        self.require(!bit(w, 50), "depth compare")?;
        let (a, b, d, e) = (bits(w, 8, 8), bits(w, 20, 8), bits(w, 0, 8), bits(w, 28, 8));
        let slot = self.texture_slot(bits(w, 36, 13) * 4, false, TextureKind::Flat);
        let offset = bit(w, 51);
        let t = if offset { a + 1 } else { b };
        self.gather(slot, a, t, offset.then_some(b), bits(w, 52, 2));
        self.store_sample(15, d, Some(e));
        self.line("}");
        Ok(())
    }

    /// A texel fetch: `float4 v` from texture `slot` at the integer
    /// coordinates in registers `x` and `y` plus the offsets packed in
    /// `packed` (4 bits each, at 0 and 4), of the level in `level`
    /// (none: the base level). Zeros outside the texture or its levels,
    /// as the interpreter's `fetch` gives.
    fn fetch(&mut self, slot: usize, x: u32, y: u32, level: Option<u32>, packed: Option<u32>) {
        let (x, y) = (self.read(x), self.read(y));
        let level = level.map_or("0u".to_string(), |register| self.read(register));
        let offset = match packed {
            Some(register) => {
                let p = self.read(register);
                format!("int2(as_type<int>({p} << 28) >> 28, as_type<int>({p} << 24) >> 28)")
            }
            None => "int2(0)".into(),
        };
        self.note_sample(slot);
        self.line(&format!(
            "{{ const uint level = {level}; const int2 p = int2(as_type<int>({x}), as_type<int>({y})) + {offset}; float4 v = float4(0.0f);"
        ));
        self.line(&format!(
            "if (level < t{slot}.get_num_mip_levels() && p.x >= 0 && p.y >= 0 && uint(p.x) < t{slot}.get_width(level) && uint(p.y) < t{slot}.get_height(level)) {{ v = t{slot}.read(uint2(p), level); }}"
        ));
    }

    /// `TLDS` on a 2D texture: coordinates in two registers (encoding
    /// 2), or in a pair with texel offsets (4), a level (5) or both (12).
    pub(super) fn tlds(&mut self) -> Result<(), String> {
        let w = self.word;
        let encoding = bits(w, 53, 4);
        let (a, b, d, e) = (bits(w, 8, 8), bits(w, 20, 8), bits(w, 0, 8), bits(w, 28, 8));
        let (y, level, packed) = match encoding {
            2 => (b, None, None),
            4 => (a + 1, None, Some(b)),
            5 => (a + 1, Some(b), None),
            12 => (a + 1, Some(b), Some(b + 1)),
            _ => return self.refuse("target other than 2D"),
        };
        let code = bits(w, 50, 3) as usize;
        let mask = if e == RZ {
            [1, 2, 4, 8, 3, 9, 10, 12][code]
        } else {
            match [7, 11, 13, 14, 15].get(code) {
                Some(&mask) => mask,
                None => return self.refuse("component mask"),
            }
        };
        let slot = self.texture_slot(bits(w, 36, 13) * 4, false, TextureKind::Flat);
        self.fetch(slot, a, y, level, packed);
        self.store_sample(mask, d, Some(e));
        self.line("}");
        Ok(())
    }

    /// `TLD` (bound handle) on a 2D texture: the base level or the level
    /// in a register, with or without texel offsets.
    pub(super) fn tld(&mut self) -> Result<(), String> {
        let w = self.word;
        let dim = bits(w, 28, 3);
        self.require(dim == 2, "target other than 2D")?;
        self.require(bits(w, 51, 3) == PT, "residency predicate")?;
        self.require(!bit(w, 50), "multisample")?;
        self.require(!bit(w, 54), "CL")?;
        let (a, b, d) = (bits(w, 8, 8), bits(w, 20, 8), bits(w, 0, 8));
        let first = a;
        let level = bit(w, 55).then_some(b);
        let packed = bit(w, 35).then_some(b + u32::from(level.is_some()));
        let slot = self.texture_slot(bits(w, 36, 13) * 4, false, TextureKind::Flat);
        self.fetch(slot, first, first + 1, level, packed);
        self.store_sample(bits(w, 31, 4), d, None);
        self.line("}");
        Ok(())
    }
}
