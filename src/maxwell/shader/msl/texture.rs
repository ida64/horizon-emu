//! Inputs of a pixel program besides registers: interpolated attributes
//! (`IPA`), indexed constant buffer reads (`LDC`) and texture samples
//! (`TEXS`, `TEX`), as the interpreter executes them (`exec.rs`,
//! `exec/tex.rs`, `exec/texture_vector.rs`).
//!
//! Textures are bound per handle offset: the instruction's 13-bit index
//! times 4 is the byte offset of the 32-bit handle in the bound-texture
//! constant buffer (`TEXTURE_BUFFER_INDEX`); the draw reads the handle
//! there, as the interpreter does, and binds the texture and sampler it
//! names at the slot [`Translated::textures`](super::Translated) lists the
//! offset in. Only 2D, non-compare samples are translated: the renderer
//! binds `texture2d<float>` slots, single-layer 2D textures only
//! (`metal::format::check_texture`). Array-target instructions are
//! translated for them: the interpreter's texture unit clamps the layer
//! to the texture's last one (`texture/filtering.rs` `coordinates`, after
//! [public: eden `texture_pass.cpp`, Ryujinx `GetTargetTexture`], which view
//! a 2D texture as a one-layer array), so the layer operand cannot matter.
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

use super::super::exec::{bit, bits, sext};
use super::{PT, RZ, SLOTS, Writer};

impl Writer<'_> {
    /// `IPA`: the attribute's plane at the pixel center (`position.xy` of a
    /// single-sample fragment is the center), optionally multiplied by a
    /// register, optionally saturated.
    pub(super) fn ipa(&mut self) -> Result<(), String> {
        let w = self.word;
        let mode = bits(w, 54, 2);
        self.require(
            mode <= 1,
            &format!("mode {}", ["", "", "CONSTANT", "SC"][mode as usize]),
        )?;
        self.require(!bit(w, 38), "IDX")?;
        self.require(bits(w, 52, 2) == 0, "sample mode")?;
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
        let mut value = format!("plane(planes, plane_base + {plane}u, px, py)");
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
    /// the bound-texture buffer).
    fn texture_slot(&mut self, handle: u32, implicit: bool) -> usize {
        let slot = match self.textures.iter().position(|&h| h == handle) {
            Some(slot) => slot,
            None => {
                self.textures.push(handle);
                self.implicit_lod.push(false);
                self.textures.len() - 1
            }
        };
        self.implicit_lod[slot] |= implicit;
        slot
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
    /// explicit level (3), and the array forms of the first two (7, 8;
    /// layer operand without effect, see the module documentation);
    /// destinations as [public: eden
    /// `translate/impl/texture_fetch_swizzled.cpp`].
    pub(super) fn texs(&mut self) -> Result<(), String> {
        let w = self.word;
        let encoding = bits(w, 53, 4);
        let refusal = match encoding {
            1..=3 | 7 | 8 => None,
            0 => Some("1D target"),
            4..=6 | 9 => Some("depth compare"),
            10 | 11 => Some("3D target"),
            12 | 13 => Some("cube target"),
            _ => Some("target encoding"),
        };
        if let Some(what) = refusal {
            return self.refuse(what);
        }
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
        let implicit = matches!(encoding, 1 | 7);
        let slot = self.texture_slot(bits(w, 36, 13) * 4, implicit);
        let (s, t) = match encoding {
            1 | 2 => (self.read(a), self.read(b)),
            // Array: layer in `a`, coordinates in `a + 1` and `b`.
            7 | 8 => (self.read(a + 1), self.read(b)),
            _ => (self.read(a), self.read(a + 1)),
        };
        let coordinates = format!("float2(as_type<float>({s}), as_type<float>({t}))");
        let sample = match encoding {
            1 | 7 => format!("t{slot}.sample(s{slot}, {coordinates}, bias(lod_bias[{slot}]))"),
            2 | 8 => format!("t{slot}.sample(s{slot}, {coordinates}, level(0.0f))"),
            _ => {
                let lod = self.read(b);
                format!("t{slot}.sample(s{slot}, {coordinates}, level(as_type<float>({lod})))")
            }
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

    /// `TEX` (bound handle) on a 2D texture (array target included, see
    /// the module documentation): implicit LOD (LOD mode 0), level zero
    /// (1), implicit with a bias register (2, 6) or an explicit level
    /// register (3, 7).
    pub(super) fn tex(&mut self) -> Result<(), String> {
        let w = self.word;
        let dim = bits(w, 28, 3);
        let target = match dim {
            2 | 3 => None,
            0 | 1 => Some("1D target"),
            4 | 5 => Some("3D target"),
            _ => Some("cube target"),
        };
        if let Some(what) = target {
            return self.refuse(what);
        }
        self.require(bits(w, 51, 3) == PT, "residency predicate")?;
        self.require(!bit(w, 58), "LOD clamp")?;
        let lod_mode = bits(w, 55, 3);
        self.require(matches!(lod_mode, 0 | 1 | 2 | 3 | 6 | 7), "LOD mode")?;
        self.require(!bit(w, 54), "AOFFI (texel offsets)")?;
        self.require(!bit(w, 50), "depth compare")?;
        let (a, b, d) = (bits(w, 8, 8), bits(w, 20, 8), bits(w, 0, 8));
        let mask = bits(w, 31, 4);
        let slot = self.texture_slot(bits(w, 36, 13) * 4, matches!(lod_mode, 0 | 2 | 6));
        // Array: the layer register comes first.
        let first = a + u32::from(dim == 3);
        let s = self.read(first);
        let t = self.read(first + 1);
        let coordinates = format!("float2(as_type<float>({s}), as_type<float>({t}))");
        let sample = match lod_mode {
            0 => format!("t{slot}.sample(s{slot}, {coordinates}, bias(lod_bias[{slot}]))"),
            1 => format!("t{slot}.sample(s{slot}, {coordinates}, level(0.0f))"),
            2 | 6 => {
                let bias = self.read(b);
                format!(
                    "t{slot}.sample(s{slot}, {coordinates}, bias(lod_bias[{slot}] + as_type<float>({bias})))"
                )
            }
            _ => {
                let lod = self.read(b);
                format!("t{slot}.sample(s{slot}, {coordinates}, level(as_type<float>({lod})))")
            }
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
}
