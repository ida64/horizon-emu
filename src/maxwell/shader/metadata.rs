//! Constant-buffer access bounds for draw-time snapshots. LDC's slot is
//! at bit 36, unlike the ordinary constant operand's slot at bit 34.
use super::{
    Program,
    decode::Op,
    exec::{Inst, bit, bits},
};
impl Program {
    /// Leading bytes of `slot` that may be read, or `None` for dynamic LDC
    /// addressing. Bound texture handles are accounted separately by
    /// [`Self::texture_handle_extent`], since their cbuf slot is an Env value.
    /// No read gives `Some(0)`. This is conservative across all control paths.
    pub fn constant_buffer_extent(&self, slot: usize) -> Option<usize> {
        match self.extents().0.get(slot) {
            Some(extent) => *extent,
            None => constant_buffer_extent(&self.code, slot),
        }
    }
    /// Leading bytes required from Env's texture-handle buffer. Bindless
    /// instructions read handles from registers and do not contribute.
    pub fn texture_handle_extent(&self) -> usize {
        self.extents().1
    }
    /// Whether the program addresses global memory (`LDG`, `LD`, `STG`,
    /// `ST`, `RED`, `ATOM`).
    pub fn addresses_global_memory(&self) -> bool {
        self.extents().2
    }
    /// The extents and whether global memory is addressed, found once
    /// per program (a draw asks for every bound slot's): per slot `LDC`
    /// can name, and of the texture handles.
    fn extents(&self) -> &super::Extents {
        self.extents.get_or_init(|| {
            (
                std::array::from_fn(|slot| constant_buffer_extent(&self.code, slot)),
                texture_handle_extent(&self.code),
                addresses_global_memory(&self.code),
            )
        })
    }
}
/// [`Program::addresses_global_memory`] of decoded code.
fn addresses_global_memory(code: &[Option<Inst>]) -> bool {
    code.iter().flatten().any(|inst| {
        matches!(
            inst.op,
            Op::Ldg | Op::Ld | Op::Stg | Op::St | Op::Red | Op::Atom
        )
    })
}
/// [`Program::constant_buffer_extent`] of decoded code.
pub(super) fn constant_buffer_extent(code: &[Option<Inst>], slot: usize) -> Option<usize> {
    let mut extent = 0;
    // A program that reads global memory takes addresses from the
    // buffer descriptors the driver keeps in constant buffers (a 64-bit
    // address, then a 32-bit size); the Metal renderer reads the size
    // of each address the program loads (`msl::Translated::globals`).
    let global = addresses_global_memory(code);
    for inst in code.iter().flatten() {
        let w = inst.word;
        if inst.op == Op::Ldc {
            if bits(w, 36, 5) as usize != slot {
                continue;
            }
            if bits(w, 8, 8) != 255 || bits(w, 44, 2) != 0 {
                return None;
            }
            let width = match bits(w, 48, 3) {
                0 | 1 => 1,
                2 | 3 => 2,
                4 => 4,
                5 => 8,
                _ => return None,
            };
            // The signed immediate is added modulo 2^32 to the address register.
            let offset = bits(w, 20, 16) as i16 as u32 as usize;
            extent = extent.max(offset + width);
            continue;
        }
        if matches!(inst.op, Op::Bra | Op::Brx | Op::Ssy | Op::Pbk | Op::Pcnt) && bit(w, 5) {
            return None;
        }
        let width = match inst.op {
            Op::DaddC
            | Op::DfmaC
            | Op::DfmaRc
            | Op::DmnmxC
            | Op::DmulC
            | Op::DsetC
            | Op::DsetpC => 8,
            Op::F2fC | Op::F2iC | Op::I2fC | Op::I2iC => {
                if bits(w, 10, 2) == 3 {
                    8
                } else {
                    4
                }
            }
            Op::BfeC
            | Op::BfiC
            | Op::BfiRc
            | Op::FaddC
            | Op::FchkC
            | Op::FcmpC
            | Op::FcmpRc
            | Op::FfmaC
            | Op::FfmaRc
            | Op::FloC
            | Op::FmnmxC
            | Op::FmulC
            | Op::FsetC
            | Op::FsetpC
            | Op::Hadd2C
            | Op::Hfma2C
            | Op::Hfma2Rc
            | Op::Hmul2C
            | Op::Hset2C
            | Op::Hsetp2C
            | Op::IaddC
            | Op::Iadd3C
            | Op::IcmpC
            | Op::IcmpRc
            | Op::IdpC
            | Op::ImadC
            | Op::ImadRc
            | Op::ImadspC
            | Op::ImadspRc
            | Op::ImnmxC
            | Op::ImulC
            | Op::IscaddC
            | Op::IsetC
            | Op::IsetpC
            | Op::LeaC
            | Op::LeaHiC
            | Op::LopC
            | Op::Lop3C
            | Op::MovC
            | Op::OutC
            | Op::P2rC
            | Op::PopcC
            | Op::PrmtC
            | Op::PrmtRc
            | Op::R2pC
            | Op::RroC
            | Op::SelC
            | Op::ShlC
            | Op::ShrC
            | Op::XmadC
            | Op::XmadRc => 4,
            _ => continue,
        };
        if bits(w, 34, 5) as usize == slot {
            let offset = bits(w, 20, 14) as usize * 4;
            extent = extent.max(offset + width);
            if global
                && matches!(inst.op, Op::MovC | Op::IaddC | Op::IscaddC)
                && offset.is_multiple_of(8)
            {
                extent = extent.max(offset + 12);
            }
        }
    }
    Some(extent)
}
/// [`Program::texture_handle_extent`] of decoded code.
pub(super) fn texture_handle_extent(code: &[Option<Inst>]) -> usize {
    code.iter()
        .flatten()
        .filter(|i| {
            matches!(
                i.op,
                Op::Tex
                    | Op::Texs
                    | Op::TexsF16
                    | Op::Tld
                    | Op::Tlds
                    | Op::TldsF16
                    | Op::Tld4
                    | Op::Tld4s
                    | Op::Tld4sF16
                    | Op::Tmml
                    | Op::Txd
                    | Op::Txq
            )
        })
        .map(|i| bits(i.word, 36, 13) as usize * 4 + 4)
        .max()
        .unwrap_or(0)
}
