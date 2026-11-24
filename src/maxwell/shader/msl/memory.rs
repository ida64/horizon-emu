//! Global memory loads of a pixel program (`LDG`), as the interpreter
//! executes them (`exec.rs` `load_global`).
//!
//! The interpreter reads any mapped address. A Metal function reads only
//! what the draw binds, so the draw must know beforehand which memory a
//! program can address. NVN programs take the addresses of the buffers
//! they read from descriptors the driver keeps in a constant buffer: a
//! 64-bit address followed by a 32-bit size, the address loaded into a
//! register pair with `MOV` or `IADD` \[observed: pixel programs of an
//! application load c\[0\]\[0x510 + 16 i\] and the word after it into
//! an even register and the next, and the 12 bytes there are an address,
//! a size and zero\]. [`candidates`] lists every place of a constant
//! buffer the program loads into a register pair that way; the draw
//! binds the memory each names, as far as it is a descriptor of mapped
//! memory, in one buffer ([`GLOBAL_BUFFER`](super::GLOBAL_BUFFER)):
//!
//! * four words per candidate: the address (low, high), the size in
//!   bytes (0: nothing bound) and the word index of its bytes in the
//!   buffer;
//! * then the bytes of each, padded to words.
//!
//! A load finds the candidate its address falls in and reads there. A
//! load of a covered fragment that falls in none sets the renderer's
//! fault word (the buffer after the memory's), which the renderer turns
//! into a refusal before any result reaches guest memory: the memory the
//! program addressed was not what its constants named, and the
//! interpreter would have read it.

use std::collections::{BTreeMap, BTreeSet};

use super::super::Program;
use super::super::decode::Op;
use super::super::exec::{bit, bits, sext};
use super::{RZ, Writer};

/// The (slot, byte offset) pairs of constant buffer words a program
/// loads into an even register, with the word after it loaded into the
/// next register: where 64-bit addresses may come from. None for a
/// program without `LDG`.
pub(super) fn candidates(program: &Program, pcs: &[u32]) -> Vec<(u32, u32)> {
    let instructions = || {
        pcs.iter()
            .map(|&pc| program.code[(pc / 8) as usize].expect("instruction"))
    };
    if !instructions().any(|inst| inst.op == Op::Ldg) {
        return Vec::new();
    }
    let mut loads: BTreeMap<(u32, u32), BTreeSet<u32>> = BTreeMap::new();
    for inst in instructions() {
        if matches!(inst.op, Op::MovC | Op::IaddC) {
            let w = inst.word;
            loads
                .entry((bits(w, 34, 5), bits(w, 20, 14) * 4))
                .or_default()
                .insert(bits(w, 0, 8));
        }
    }
    loads
        .iter()
        .filter(|&(&(slot, offset), lows)| {
            offset.is_multiple_of(8)
                && loads.get(&(slot, offset + 4)).is_some_and(|highs| {
                    lows.iter()
                        .any(|&low| low.is_multiple_of(2) && highs.contains(&(low + 1)))
                })
        })
        .map(|(&place, _)| place)
        .collect()
}

impl Writer<'_> {
    /// `LDG` with a 64-bit address (`E`): address register pair 8-15,
    /// offset 20-43 signed, size 48-50 (bytes and halves zero- or
    /// sign-extended, one, two or four words).
    pub(super) fn ldg(&mut self) -> Result<(), String> {
        let w = self.word;
        self.require(bit(w, 45), "32-bit address")?;
        let (rd, ra) = (bits(w, 0, 8), bits(w, 8, 8));
        self.require(
            ra.is_multiple_of(2) || ra == RZ,
            "64-bit address in an odd register",
        )?;
        let size = match bits(w, 48, 3) {
            7 => 6,
            size => size,
        };
        let width: u32 = [1, 1, 2, 2, 4, 8, 16][size as usize];
        let count = width.div_ceil(4);
        self.require(
            rd == RZ || rd.is_multiple_of(count),
            "vector destination not aligned",
        )?;
        self.require(
            !self.globals.is_empty(),
            "global memory no constant names",
        )?;
        let offset = i64::from(sext(bits(w, 20, 24), 24));
        let low = self.read(ra);
        let high = if ra == RZ {
            "0u".to_string()
        } else {
            self.read(ra + 1)
        };
        self.line(&format!(
            "{{ const ulong ad = ((ulong({high}) << 32) | ulong({low})) + ulong({offset}l); uint g0 = 0u, g1 = 0u, g2 = 0u, g3 = 0u; bool hit = false;"
        ));
        let words: String = (0..count)
            .map(|i| format!("g{i} = gm[at + {i}u]; "))
            .collect();
        self.line(&format!(
            "for (uint k = 0u; k < {}u && (ad & {}ul) == 0ul; k++) {{ const ulong rel = ad - ((ulong(gm[4u * k + 1u]) << 32) | ulong(gm[4u * k])); const ulong size = ulong(gm[4u * k + 2u]); if (rel < size && size - rel >= {width}ul) {{ const uint at = gm[4u * k + 3u] + uint(rel >> 2); {words}g0 >>= uint(rel & 3ul) * 8u; hit = true; break; }} }}",
            self.globals.len(),
            width - 1,
        ));
        self.line(
            "if (!hit && GM_LIVE) { atomic_store_explicit(gm_fault, 1u, memory_order_relaxed); }",
        );
        match size {
            0 => self.set(rd, "(g0 & 0xffu)"),
            1 => self.set(rd, "as_type<uint>(as_type<int>(g0 << 24) >> 24)"),
            2 => self.set(rd, "(g0 & 0xffffu)"),
            3 => self.set(rd, "as_type<uint>(as_type<int>(g0 << 16) >> 16)"),
            _ => {
                for i in 0..count {
                    if rd != RZ {
                        self.set(rd + i, &format!("g{i}"));
                    }
                }
            }
        }
        self.line("}");
        Ok(())
    }
}
