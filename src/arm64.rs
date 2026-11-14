//! AArch64 architectural encodings shared by any CPU backend: exception
//! syndromes, system register identifiers and the stage 1 table walk.

use std::fmt;

/// Exception class, `ESR_ELx[31:26]`.
pub mod exception_class {
    pub const WFX: u32 = 0x01;
    pub const SVC64: u32 = 0x15;
    pub const SMC64: u32 = 0x17;
    pub const SYSTEM_REGISTER: u32 = 0x18;
    pub const INSTRUCTION_ABORT_LOWER: u32 = 0x20;
    pub const DATA_ABORT_LOWER: u32 = 0x24;
}

pub const fn exception_class(syndrome: u64) -> u32 {
    (syndrome >> 26) as u32 & 0x3F
}

/// `op0:op1:CRn:CRm:op2` packed as 2:3:4:4:3 bits, the layout used by MRS/MSR
/// encodings (shifted) and by Hypervisor.framework register identifiers.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct SysReg(pub u16);

impl SysReg {
    pub const MIDR_EL1: Self = Self::new(3, 0, 0, 0, 0);
    pub const MPIDR_EL1: Self = Self::new(3, 0, 0, 0, 5);
    pub const SCTLR_EL1: Self = Self::new(3, 0, 1, 0, 0);
    pub const TTBR0_EL1: Self = Self::new(3, 0, 2, 0, 0);
    pub const TTBR1_EL1: Self = Self::new(3, 0, 2, 0, 1);
    pub const TCR_EL1: Self = Self::new(3, 0, 2, 0, 2);
    pub const SPSR_EL1: Self = Self::new(3, 0, 4, 0, 0);
    pub const ELR_EL1: Self = Self::new(3, 0, 4, 0, 1);
    pub const ESR_EL1: Self = Self::new(3, 0, 5, 2, 0);
    pub const FAR_EL1: Self = Self::new(3, 0, 6, 0, 0);
    pub const VBAR_EL1: Self = Self::new(3, 0, 12, 0, 0);
    pub const CPUACTLR_EL1: Self = Self::new(3, 1, 15, 2, 0);
    pub const CPUECTLR_EL1: Self = Self::new(3, 1, 15, 2, 1);
    pub const OSLAR_EL1: Self = Self::new(2, 0, 1, 0, 4);
    pub const CNTPCT_EL0: Self = Self::new(3, 3, 14, 0, 1);
    pub const CNTP_CTL_EL0: Self = Self::new(3, 3, 14, 2, 1);
    pub const CNTP_CVAL_EL0: Self = Self::new(3, 3, 14, 2, 2);
    pub const CONTEXTIDR_EL1: Self = Self::new(3, 0, 13, 0, 1);

    pub const fn new(op0: u16, op1: u16, crn: u16, crm: u16, op2: u16) -> Self {
        Self(op0 << 14 | op1 << 11 | crn << 7 | crm << 3 | op2)
    }

    pub const fn op0(self) -> u16 {
        self.0 >> 14
    }

    pub const fn op1(self) -> u16 {
        self.0 >> 11 & 7
    }

    pub const fn crn(self) -> u16 {
        self.0 >> 7 & 15
    }

    pub const fn crm(self) -> u16 {
        self.0 >> 3 & 15
    }

    pub const fn op2(self) -> u16 {
        self.0 & 7
    }
}

impl fmt::Debug for SysReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "S{}_{}_C{}_C{}_{}",
            self.op0(),
            self.op1(),
            self.crn(),
            self.crm(),
            self.op2()
        )
    }
}

/// A trapped MRS, MSR or SYS instruction.
#[derive(Debug, Clone, Copy)]
pub struct SysRegAccess {
    pub register: SysReg,
    /// General-purpose register index; 31 is XZR.
    pub rt: u8,
    pub is_read: bool,
}

impl SysRegAccess {
    /// Decodes the ISS of an [`exception_class::SYSTEM_REGISTER`] syndrome.
    pub fn from_syndrome(syndrome: u64) -> Self {
        let field = |shift: u32, mask: u64| (syndrome >> shift & mask) as u16;
        Self {
            register: SysReg::new(
                field(20, 3),
                field(14, 7),
                field(10, 15),
                field(1, 15),
                field(17, 7),
            ),
            rt: (syndrome >> 5 & 31) as u8,
            is_read: syndrome & 1 != 0,
        }
    }
}

/// A data abort taken to the hypervisor, i.e. a load or store that reached a
/// guest-physical address with no memory behind it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DataAbort {
    pub is_write: bool,
    /// How to complete the access. `None` when the syndrome does not describe
    /// the instruction (ISV clear: pair, writeback and SIMD forms); those
    /// need [`LoadStore::decode`] on the instruction itself.
    pub transfer: Option<Transfer>,
}

/// The register side of a single-register load or store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Transfer {
    /// Access width in bytes: 1, 2, 4 or 8.
    pub size: u8,
    /// General-purpose register index; 31 is XZR.
    pub rt: u8,
    sign_extend: bool,
    /// Whether the register operand is 64 bits wide.
    wide_register: bool,
}

impl DataAbort {
    /// Decodes the ISS of an [`exception_class::DATA_ABORT_LOWER`] syndrome.
    pub fn from_syndrome(syndrome: u64) -> Self {
        let valid = syndrome >> 24 & 1 != 0;
        Self {
            is_write: syndrome >> 6 & 1 != 0,
            transfer: valid.then(|| Transfer {
                size: 1 << (syndrome >> 22 & 3),
                rt: (syndrome >> 16 & 31) as u8,
                sign_extend: syndrome >> 21 & 1 != 0,
                wide_register: syndrome >> 15 & 1 != 0,
            }),
        }
    }
}

impl Transfer {
    /// Value the destination register takes when a load reads `value`.
    pub fn loaded(self, value: u64) -> u64 {
        let bits = u32::from(self.size) * 8;
        let shift = 64 - bits;
        let extended = if self.sign_extend {
            ((value << shift) as i64 >> shift) as u64
        } else {
            value << shift >> shift
        };
        if self.wide_register {
            extended
        } else {
            extended & 0xFFFF_FFFF
        }
    }

    /// Value a store of register contents `register` puts on the bus.
    pub fn stored(self, register: u64) -> u64 {
        let shift = 64 - u32::from(self.size) * 8;
        register << shift >> shift
    }
}

/// A load or store instruction, as far as completing it on a device needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoadStore {
    pub is_load: bool,
    /// Transfers at consecutive addresses starting at the accessed one.
    pub first: Transfer,
    pub second: Option<Transfer>,
    pub writeback: Option<Writeback>,
}

/// Base register update of a pre- or post-indexed access.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Writeback {
    /// Base register index; never 31 (SP-based forms are not decoded).
    pub rn: u8,
    /// Added to the base register, whichever of pre or post it is.
    pub offset: i64,
}

impl LoadStore {
    /// The single-register access a valid abort syndrome describes.
    pub fn single(is_load: bool, transfer: Transfer) -> Self {
        Self {
            is_load,
            first: transfer,
            second: None,
            writeback: None,
        }
    }

    /// Decodes the general-purpose-register load/store forms whose data
    /// aborts carry no instruction syndrome: pre- and post-indexed single
    /// registers, and pairs in every addressing mode. Anything else (SIMD,
    /// exclusives, SP as base) is `None`.
    pub fn decode(instruction: u32) -> Option<Self> {
        let field = |shift: u32, bits: u32| instruction >> shift & ((1 << bits) - 1);
        let signed = |shift: u32, bits: u32| {
            i64::from((instruction << (32 - shift - bits)) as i32 >> (32 - bits))
        };
        let rt = field(0, 5) as u8;
        let rn = field(5, 5) as u8;
        let is_simd = field(26, 1) != 0;
        if is_simd || rn == 31 {
            return None;
        }

        // Load/store register, immediate pre/post-indexed:
        // size:111:V:00:opc:0:imm9:P1:Rn:Rt, P = 0 post, 1 pre.
        if field(27, 3) == 0b111 && field(24, 2) == 0 && field(21, 1) == 0 && field(10, 1) == 1 {
            let size_log2 = field(30, 2);
            let (is_load, sign_extend, wide_register) = match (field(22, 2), size_log2) {
                (0b00, _) => (false, false, size_log2 == 3),
                (0b01, _) => (true, false, size_log2 == 3),
                (0b10, 0..=2) => (true, true, true),
                (0b11, 0..=1) => (true, true, false),
                _ => return None,
            };
            return Some(Self {
                is_load,
                first: Transfer {
                    size: 1 << size_log2,
                    rt,
                    sign_extend,
                    wide_register,
                },
                second: None,
                writeback: Some(Writeback {
                    rn,
                    offset: signed(12, 9),
                }),
            });
        }

        // Load/store pair: opc:101:V:0:mode:L:imm7:Rt2:Rn:Rt, mode = 01 post,
        // 10 signed offset, 11 pre.
        if field(27, 3) == 0b101 && field(25, 1) == 0 && field(23, 2) != 0 {
            let is_load = field(22, 1) != 0;
            let (size, sign_extend) = match (field(30, 2), is_load) {
                (0b00, _) => (4, false),
                (0b10, _) => (8, false),
                // LDPSW; the store encoding of this opc is not a plain pair.
                (0b01, true) => (4, true),
                _ => return None,
            };
            let transfer = |rt| Transfer {
                size,
                rt,
                sign_extend,
                wide_register: size == 8 || sign_extend,
            };
            let indexed = field(23, 2) != 0b10;
            return Some(Self {
                is_load,
                first: transfer(rt),
                second: Some(transfer(field(10, 5) as u8)),
                writeback: indexed.then(|| Writeback {
                    rn,
                    offset: signed(15, 7) * i64::from(size),
                }),
            });
        }
        None
    }
}

/// Stage 1 translation regime of one core.
#[derive(Debug, Clone, Copy)]
pub struct TranslationRegime {
    pub sctlr: u64,
    pub tcr: u64,
    pub ttbr0: u64,
    pub ttbr1: u64,
}

impl TranslationRegime {
    /// Walks the tables for `va`, reading descriptors through `read_u64`
    /// (physical address to little-endian value). Only the 4 KiB granule is
    /// implemented; other granules and unmapped addresses yield `None`.
    pub fn translate(&self, va: u64, read_u64: impl Fn(u64) -> Option<u64>) -> Option<u64> {
        const ADDRESS: u64 = 0x0000_FFFF_FFFF_F000;

        if self.sctlr & 1 == 0 {
            return Some(va);
        }
        let upper = va >> 55 & 1 != 0;
        let (ttbr, size_offset, granule_4k) = if upper {
            (self.ttbr1, self.tcr >> 16 & 0x3F, self.tcr >> 30 & 3 == 2)
        } else {
            (self.ttbr0, self.tcr & 0x3F, self.tcr >> 14 & 3 == 0)
        };
        if !granule_4k || !(16..=39).contains(&size_offset) {
            return None;
        }
        let va_bits = 64 - size_offset as u32;
        let expected_top = if upper { u64::MAX >> va_bits } else { 0 };
        if va >> va_bits != expected_top {
            return None;
        }

        let first_level = 4 - (va_bits - 12).div_ceil(9);
        let mut table = ttbr & ADDRESS;
        for level in first_level..=3 {
            let shift = 12 + 9 * (3 - level);
            let index_bits = (va_bits - shift).min(9);
            let index = va >> shift & ((1 << index_bits) - 1);
            let descriptor = read_u64(table + index * 8)?;
            let is_table_or_page = descriptor & 2 != 0;
            if descriptor & 1 == 0 || (level == 3 && !is_table_or_page) {
                return None;
            }
            if level == 3 || !is_table_or_page {
                let offset_mask = (1u64 << shift) - 1;
                return Some(descriptor & ADDRESS & !offset_mask | va & offset_mask);
            }
            table = descriptor & ADDRESS;
        }
        unreachable!("level 3 always returns")
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    const TABLE: u64 = 0b11;
    const BLOCK: u64 = 0b01;
    const PAGE: u64 = 0b11;

    /// The kernel's regime: 39-bit halves, 4 KiB granule, 16-bit ASID.
    const TCR: u64 = 0x11_B519_3519;

    #[test]
    fn walks_pages_and_blocks_in_both_halves() {
        // TTBR1: L1 @0x1000 -> L2 @0x2000 -> L3 @0x3000 -> page 0x8012_3000,
        // plus a 2 MiB block at the next L2 slot. TTBR0: 1 GiB block at L1.
        let high = 0xFFFF_FFB4_3960_0000u64;
        let index = |shift: u32| high >> shift & 0x1FF;
        let memory = HashMap::from([
            (0x1000 + index(30) * 8, 0x2000 | TABLE),
            (0x2000 + index(21) * 8, 0x3000 | TABLE),
            (0x3000 + 0xC6 * 8, 0x8012_3000 | PAGE | 1 << 54),
            (0x2000 + (index(21) + 1) * 8, 0x9000_0000 | BLOCK),
            (0x5000 + 2 * 8, 0x8000_0000 | BLOCK),
        ]);
        // ASID in TTBR[63:48] must not leak into the table address.
        let regime = TranslationRegime {
            sctlr: 1,
            tcr: TCR,
            ttbr0: 0x5000,
            ttbr1: 0x1000 | 0x2A << 48,
        };
        let translate = |va| regime.translate(va, |address| memory.get(&address).copied());

        assert_eq!(translate(high + 0xC_6ABC), Some(0x8012_3ABC));
        assert_eq!(translate(high + 0x20_0000 + 0x12_3456), Some(0x9012_3456));
        assert_eq!(translate(0x8765_4321), Some(0x8765_4321));
        // Valid table, invalid leaf.
        assert_eq!(translate(high + 0xC_7000), None);
        // Outside the 39-bit half the tables cover.
        assert_eq!(translate(0xFFFF_FF00_0000_0000 - 0x1000), None);
        assert_eq!(translate(0x80_0000_0000), None);
    }

    #[test]
    fn translation_is_identity_while_the_mmu_is_off() {
        let regime = TranslationRegime {
            sctlr: 0x30D0_0800,
            tcr: 0,
            ttbr0: 0,
            ttbr1: 0,
        };
        assert_eq!(regime.translate(0x8006_0000, |_| None), Some(0x8006_0000));
    }

    #[test]
    fn system_register_trap_syndromes_decode_to_the_accessed_register() {
        // MRS X2, S3_1_C15_C2_1 (CPUECTLR_EL1) and MSR OSLAR_EL1, X10.
        let read = SysRegAccess::from_syndrome(
            0x18 << 26 | 3 << 20 | 1 << 17 | 1 << 14 | 15 << 10 | 2 << 5 | 2 << 1 | 1,
        );
        assert_eq!(
            (read.register, read.rt, read.is_read),
            (SysReg::CPUECTLR_EL1, 2, true)
        );
        let write = SysRegAccess::from_syndrome(0x18 << 26 | 2 << 20 | 4 << 17 | 1 << 10 | 10 << 5);
        assert_eq!(
            (write.register, write.rt, write.is_read),
            (SysReg::OSLAR_EL1, 10, false)
        );
    }

    #[test]
    fn data_abort_syndromes_describe_how_to_complete_the_access() {
        const VALID: u64 = 0x24 << 26 | 1 << 24;

        // LDR W1, [Xn]: zero-extends into the whole register.
        let load = DataAbort::from_syndrome(VALID | 2 << 22 | 1 << 16);
        let transfer = load.transfer.unwrap();
        assert_eq!((load.is_write, transfer.size, transfer.rt), (false, 4, 1));
        assert_eq!(transfer.loaded(0xFFFF_FFFF_8000_0001), 0x8000_0001);

        // LDRSB X3, [Xn] sign-extends to 64 bits; LDRSH W3 only to 32.
        let signed_byte = DataAbort::from_syndrome(VALID | 1 << 21 | 3 << 16 | 1 << 15)
            .transfer
            .unwrap();
        assert_eq!(
            (signed_byte.size, signed_byte.loaded(0x80)),
            (1, 0xFFFF_FFFF_FFFF_FF80)
        );
        let signed_half = DataAbort::from_syndrome(VALID | 1 << 22 | 1 << 21 | 3 << 16)
            .transfer
            .unwrap();
        assert_eq!(
            (signed_half.size, signed_half.loaded(0x8000)),
            (2, 0xFFFF_8000)
        );

        // STRB W9, [Xn] puts only the low byte on the bus.
        let store = DataAbort::from_syndrome(VALID | 9 << 16 | 1 << 6);
        assert!(store.is_write);
        assert_eq!(store.transfer.unwrap().stored(0x1234_5678), 0x78);
        let wide = DataAbort::from_syndrome(VALID | 3 << 22 | 1 << 15)
            .transfer
            .unwrap();
        assert_eq!(
            (wide.size, wide.stored(u64::MAX), wide.loaded(u64::MAX)),
            (8, u64::MAX, u64::MAX)
        );

        // Without ISV (e.g. STP) the direction is known but nothing else.
        let pair = DataAbort::from_syndrome(0x24 << 26 | 1 << 6);
        assert_eq!(
            pair,
            DataAbort {
                is_write: true,
                transfer: None
            }
        );
    }

    #[test]
    fn decodes_the_load_store_forms_that_abort_without_a_syndrome() {
        // STR WZR, [X13], #0x10: the kernel's GICD_ICFGR clearing loop (0x800d5088).
        let store = LoadStore::decode(0xB801_05BF).unwrap();
        assert_eq!(
            (
                store.is_load,
                store.first.size,
                store.first.rt,
                store.second
            ),
            (false, 4, 31, None)
        );
        assert_eq!(
            store.writeback,
            Some(Writeback {
                rn: 13,
                offset: 0x10
            })
        );

        // LDR X1, [X2, #-8]!
        let load = LoadStore::decode(0xF85F_8C41).unwrap();
        assert_eq!((load.is_load, load.first.size, load.first.rt), (true, 8, 1));
        assert_eq!(load.writeback, Some(Writeback { rn: 2, offset: -8 }));
        assert_eq!(load.first.loaded(u64::MAX), u64::MAX);

        // LDRSW X3, [X4], #4 sign-extends a word into the whole register.
        let signed = LoadStore::decode(0xB880_4483).unwrap();
        assert_eq!(
            (signed.first.size, signed.first.loaded(0x8000_0000)),
            (4, 0xFFFF_FFFF_8000_0000)
        );

        // STP W8, W9, [X10]: two words, no writeback.
        let pair = LoadStore::decode(0x2900_2548).unwrap();
        assert_eq!(
            (pair.is_load, pair.first.rt, pair.second.unwrap().rt),
            (false, 8, 9)
        );
        assert_eq!((pair.first.size, pair.writeback), (4, None));

        // LDP X0, X1, [X2], #16: the immediate is scaled by the register size.
        let pair = LoadStore::decode(0xA8C1_0440).unwrap();
        assert_eq!(
            (pair.is_load, pair.first.size, pair.second.unwrap().rt),
            (true, 8, 1)
        );
        assert_eq!(pair.writeback, Some(Writeback { rn: 2, offset: 16 }));

        // LDUR X1, [X2] (no writeback form, has a syndrome), STR W0, [SP], #4
        // (SP base), STR Q0, [X0], #16 (SIMD), ADD X0, X0, #1 (not a load).
        for other in [0xF840_0041, 0xB800_47E0, 0x3C81_0400, 0x9100_0400] {
            assert_eq!(LoadStore::decode(other), None, "{other:#010x}");
        }
    }
}
