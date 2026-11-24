//! Global memory accesses of a pixel or vertex program (`LDG`, and in
//! pixel programs `STG` and `RED`), as the interpreter executes them
//! (`exec.rs` `load_global`, `store_global`, `atomic`).
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
//! buffer the program loads into a register pair that way. The renderer
//! keeps the memory such places name in one buffer every program is
//! given ([`GLOBAL_BUFFER`](super::GLOBAL_BUFFER) of the fragment
//! function, [`VERTEX_GLOBAL_BUFFER`](super::VERTEX_GLOBAL_BUFFER) of
//! the vertex function), so what one program stores the next loads
//! without the bytes leaving the GPU; its first word is the fault word.
//! A draw gives its program a table of four words per candidate
//! (`GT(i)`: a buffer of its own for a fragment function, words of the
//! draw's for a vertex function): the address (low, high), the size in
//! bytes (0: nothing bound) and the word index of its bytes in the
//! buffer.
//!
//! An access finds the candidate its address falls in and reads or
//! writes there. An access of a vertex or a covered fragment that falls
//! in none sets the fault word, which the renderer turns into a refusal
//! before any result reaches guest memory: the memory the program
//! addressed was not what its constants named, and the interpreter
//! would have read it.

use std::collections::{BTreeMap, BTreeSet};

use super::super::Program;
use super::super::decode::Op;
use super::super::exec::{bit, bits, sext};
use super::{RZ, Writer};

/// The (slot, byte offset) pairs of constant buffer words a program
/// loads into an even register (`MOV`, `IADD`, or `ISCADD`, which adds
/// an index to the low word), with the word after it loaded into the
/// next register: where 64-bit addresses may come from. None for a
/// program without `LDG`, `STG`, `RED` or `ATOM`.
pub(super) fn candidates(program: &Program, pcs: &[u32]) -> Vec<(u32, u32)> {
    let instructions = || {
        pcs.iter()
            .map(|&pc| program.code[(pc / 8) as usize].expect("instruction"))
    };
    if !instructions().any(|inst| matches!(inst.op, Op::Ldg | Op::Stg | Op::Red | Op::Atom)) {
        return Vec::new();
    }
    let mut loads: BTreeMap<(u32, u32), BTreeSet<u32>> = BTreeMap::new();
    for inst in instructions() {
        if matches!(inst.op, Op::MovC | Op::IaddC | Op::IscaddC) {
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

/// Whether a program stores to global memory (`STG`, `RED`): its global
/// memory buffer is written, and written back to guest memory.
pub(super) fn stores(program: &Program, pcs: &[u32]) -> bool {
    pcs.iter().any(|&pc| {
        matches!(
            program.code[(pc / 8) as usize].expect("instruction").op,
            Op::Stg | Op::Red
        )
    })
}

impl Writer<'_> {
    /// Opens a block with `ad`, the address in the register pair at `ra`
    /// (in `ra` alone, zero-extended, without `wide`: a 32-bit address)
    /// plus `offset`.
    fn global_address(&mut self, ra: u32, offset: i64, wide: bool) -> Result<(), String> {
        self.require(
            !wide || ra.is_multiple_of(2) || ra == RZ,
            "64-bit address in an odd register",
        )?;
        self.require(!self.globals.is_empty(), "global memory no constant names")?;
        let low = self.read(ra);
        let high = if ra == RZ || !wide {
            "0u".to_string()
        } else {
            self.read(ra + 1)
        };
        self.line(&format!(
            "{{ const ulong ad = ((ulong({high}) << 32) | ulong({low})) + ulong({offset}l); bool hit = false;"
        ));
        Ok(())
    }

    /// Runs `body` with `at`, the word index in the bound memory of the
    /// `width` bytes at `ad` (word-aligned, in one bound buffer), and
    /// sets `hit`; a live fragment whose address is in none reports the
    /// fault.
    fn global_words(&mut self, width: u32, body: &str) {
        self.line(&format!(
            "for (uint k = 0u; k < {}u && (ad & {}ul) == 0ul; k++) {{ const ulong rel = ad - ((ulong(GT(4u * k + 1u)) << 32) | ulong(GT(4u * k))); const ulong size = ulong(GT(4u * k + 2u)); if (rel < size && size - rel >= {width}ul) {{ const uint at = GT(4u * k + 3u) + uint(rel >> 2); {body} hit = true; break; }} }}",
            self.globals.len(),
            width.max(4) - 1,
        ));
        self.line(
            "if (!hit && GM_LIVE) { atomic_store_explicit(&gm[0], 1u, memory_order_relaxed); }",
        );
    }

    /// `STG` of one, two or four words: source register 0-7, address
    /// register 8-15 (a pair with `E`, bit 45), offset 20-43
    /// signed, size 48-50. Fragments outside the primitive store
    /// nothing (`exec.rs` `store_global`).
    pub(super) fn stg(&mut self) -> Result<(), String> {
        let w = self.word;
        let wide = bit(w, 45);
        let (rd, ra) = (bits(w, 0, 8), bits(w, 8, 8));
        let size = bits(w, 48, 3);
        self.require(matches!(size, 4..=6), "store narrower than a word")?;
        let count = [1u32, 2, 4][size as usize - 4];
        self.require(
            rd == RZ || rd.is_multiple_of(count),
            "vector source not aligned",
        )?;
        let values: Vec<String> = (0..count)
            .map(|i| self.read(if rd == RZ { RZ } else { rd + i }))
            .collect();
        self.global_address(ra, i64::from(sext(bits(w, 20, 24), 24)), wide)?;
        let body: String = values
            .iter()
            .enumerate()
            .map(|(i, value)| {
                format!("if (GM_LIVE) {{ atomic_store_explicit(&gm[at + {i}u], {value}, memory_order_relaxed); }} ")
            })
            .collect();
        self.global_words(4 * count, &body);
        self.line("}");
        Ok(())
    }

    /// `RED` (`exec.rs` `atomic`): source register 0-7, address register
    /// 8-15 (a pair with `E`, bit 48), type 20-22 (U32, S32), operation
    /// 23-25 (ADD, MIN, MAX, AND, OR, XOR), offset 28-47 signed.
    pub(super) fn red(&mut self) -> Result<(), String> {
        let w = self.word;
        let wide = bit(w, 48);
        let (source, ra) = (bits(w, 0, 8), bits(w, 8, 8));
        let (kind, operation) = (bits(w, 20, 3), bits(w, 23, 3));
        self.require(kind < 2, &format!("type {kind}"))?;
        let signed = kind == 1;
        let function = match operation {
            0 => "atomic_fetch_add_explicit",
            1 => "atomic_fetch_min_explicit",
            2 => "atomic_fetch_max_explicit",
            5 => "atomic_fetch_and_explicit",
            6 => "atomic_fetch_or_explicit",
            7 => "atomic_fetch_xor_explicit",
            other => return self.refuse(&format!("operation {other}")),
        };
        let value = self.read(source);
        self.global_address(ra, i64::from(sext(bits(w, 28, 20), 20)), wide)?;
        let body = if signed && matches!(operation, 1 | 2) {
            format!(
                "if (GM_LIVE) {{ {function}((device atomic_int*)&gm[at], as_type<int>({value}), memory_order_relaxed); }}"
            )
        } else {
            format!("if (GM_LIVE) {{ {function}(&gm[at], {value}, memory_order_relaxed); }}")
        };
        self.global_words(4, &body);
        self.line("}");
        Ok(())
    }

    /// `ATOM` (`exec.rs` `atomic`): destination 0-7 (the value memory
    /// held), address register 8-15 (a pair with `E`, bit 48), source
    /// 20-27, type 49-51 (U32, S32), operation 52-55 (ADD, MIN, MAX, AND,
    /// OR, XOR, EXCH), offset 28-47 signed.
    pub(super) fn atom(&mut self) -> Result<(), String> {
        let w = self.word;
        let wide = bit(w, 48);
        let (rd, ra, source) = (bits(w, 0, 8), bits(w, 8, 8), bits(w, 20, 8));
        let (kind, operation) = (bits(w, 49, 3), bits(w, 52, 4));
        self.require(kind < 2, &format!("type {kind}"))?;
        let signed = kind == 1;
        let function = match operation {
            0 => "atomic_fetch_add_explicit",
            1 => "atomic_fetch_min_explicit",
            2 => "atomic_fetch_max_explicit",
            5 => "atomic_fetch_and_explicit",
            6 => "atomic_fetch_or_explicit",
            7 => "atomic_fetch_xor_explicit",
            8 => "atomic_exchange_explicit",
            other => return self.refuse(&format!("operation {other}")),
        };
        let value = self.read(source);
        self.global_address(ra, i64::from(sext(bits(w, 28, 20), 20)), wide)?;
        self.line("uint old = 0u;");
        let body = if signed && matches!(operation, 1 | 2) {
            format!(
                "old = as_type<uint>({function}((device atomic_int*)&gm[at], as_type<int>({value}), memory_order_relaxed));"
            )
        } else {
            format!("old = {function}(&gm[at], {value}, memory_order_relaxed);")
        };
        self.global_words(4, &body);
        self.set(rd, "old");
        self.line("}");
        Ok(())
    }

    /// `LDG`: address register 8-15 (a pair with `E`, bit 45),
    /// offset 20-43 signed, size 48-50 (bytes and halves zero- or
    /// sign-extended, one, two or four words).
    pub(super) fn ldg(&mut self) -> Result<(), String> {
        let w = self.word;
        let wide = bit(w, 45);
        let (rd, ra) = (bits(w, 0, 8), bits(w, 8, 8));
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
        self.global_address(ra, i64::from(sext(bits(w, 20, 24), 24)), wide)?;
        self.line("uint g0 = 0u, g1 = 0u, g2 = 0u, g3 = 0u;");
        let words: String = (0..count)
            .map(|i| format!("g{i} = GM(at + {i}u); "))
            .collect();
        self.line(&format!(
            "for (uint k = 0u; k < {}u && (ad & {}ul) == 0ul; k++) {{ const ulong rel = ad - ((ulong(GT(4u * k + 1u)) << 32) | ulong(GT(4u * k))); const ulong size = ulong(GT(4u * k + 2u)); if (rel < size && size - rel >= {width}ul) {{ const uint at = GT(4u * k + 3u) + uint(rel >> 2); {words}g0 >>= uint(rel & 3ul) * 8u; hit = true; break; }} }}",
            self.globals.len(),
            width - 1,
        ));
        self.line(
            "if (!hit && GM_LIVE) { atomic_store_explicit(&gm[0], 1u, memory_order_relaxed); }",
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

    /// `LDL`/`STL` of whole words (`exec.rs` `local_access`): the
    /// thread's local memory is the array `lm` ([`local_words`]),
    /// zeros until stored to. An access outside it loads zeros and
    /// stores nothing \[assumption: programs stay inside the local
    /// memory their constant offsets show\].
    pub(super) fn local(&mut self, store: bool) -> Result<(), String> {
        let w = self.word;
        let size = bits(w, 48, 3);
        self.require(matches!(size, 4..=6), "access narrower than a word")?;
        let count = [1u32, 2, 4][size as usize - 4];
        let (rd, ra) = (bits(w, 0, 8), bits(w, 8, 8));
        self.require(
            rd == RZ || rd.is_multiple_of(count),
            "vector register not aligned",
        )?;
        let immediate = bits(w, 20, 24);
        let offset = if ra == RZ {
            immediate
        } else {
            sext(immediate, 24) as u32
        };
        let a = self.read(ra);
        let values: Vec<String> = (0..count)
            .map(|i| self.read(if rd == RZ { RZ } else { rd + i }))
            .collect();
        self.line(&format!(
            "{{ const uint la = {a} + {offset:#x}u; const bool lok = (la & {}u) == 0u && (la >> 2) + {count}u <= LM_WORDS;",
            4 * count - 1
        ));
        for i in 0..count {
            if store {
                self.line(&format!("if (lok) {{ lm[(la >> 2) + {i}u] = {}; }}", values[i as usize]));
            } else if rd != RZ {
                self.set(rd + i, &format!("(lok ? lm[(la >> 2) + {i}u] : 0u)"));
            }
        }
        self.line("}");
        Ok(())
    }
}

/// The least words of local memory a translated program gets.
const LOCAL_WORDS: u32 = 64;

/// The words of the local memory array of a program that uses `LDL` or
/// `STL` (0: it uses neither): what its accesses at constant addresses
/// reach, at least [`LOCAL_WORDS`], at most 1024.
pub(super) fn local_words(program: &Program, pcs: &[u32]) -> u32 {
    let mut words = 0;
    for &pc in pcs {
        let inst = program.code[(pc / 8) as usize].expect("instruction");
        if !matches!(inst.op, Op::Ldl | Op::Stl) {
            continue;
        }
        words = words.max(LOCAL_WORDS);
        if bits(inst.word, 8, 8) == RZ {
            let end = bits(inst.word, 20, 24).saturating_add(16);
            words = words.max(end.div_ceil(4).min(1024));
        }
    }
    words
}
