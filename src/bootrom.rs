//! The T210 boot ROM's cold boot from the eMMC, high-level emulated: the
//! state it leaves when it starts the bootloader (package1ldr) on the BPMP.
//!
//! The ROM is NVIDIA's code and is not part of this program, so this is
//! built from public descriptions only: switchbrew ("BCT", "Package1",
//! "Fuses", "Cryptosystem", "Flash Filesystem"), the Fusée Gelée disclosure
//! (ReSwitched, 2018), NVIDIA's T210 BCT header as published with
//! cbootimage (layout facts), U-Boot's and coreboot's Tegra210 headers, and
//! what package1ldr of firmware 23.5.0 reads (its code at `0x400100FA`,
//! called from `0x40010CEE`). A machine given a dump of the ROM runs it on
//! the BPMP instead ([`crate::soc::irom`], [`crate::machine::PowerOn::Rom`])
//! and takes from here only the console's fuses ([`Fuses::at_reset`]).
//!
//! [`cold_boot`], in the ROM's order:
//!
//! 1. `PMC_SCRATCH0` bit 0 asks for a warm boot, bit 1 for recovery mode
//!    (Fusée Gelée's pseudo-code of the T210 ROM tests "bit 1" and "bit
//!    2", counting from one: the monitor of firmware 23.5.0 writes 1 there
//!    before it suspends, and Linux's Tegra PMC driver has the recovery
//!    mode bit at bit 1). A warm boot is [`warm_boot`]'s (SC7 resume);
//!    recovery mode is not modelled and refused.
//! 2. The boot device comes from `FUSE_RESERVED_SW` bits 0-2 when its bit 3
//!    ("skip device selection straps") is set (switchbrew "Fuses"); the
//!    straps are not modelled, so a clear bit 3 is refused, and so is any
//!    device but the SDMMC (the eMMC).
//! 3. The secure boot scheme: on an ODM production-fused unit whose
//!    `FUSE_PUBLIC_KEY` is set, the BCT and the bootloader are checked by
//!    RSASSA-PSS (the BCT's `CryptoHash`, an AES-CMAC, is for the symmetric
//!    scheme instead; cbootimage's T210 header). Retail Erista consoles are
//!    such units: the BCT's CMAC field is empty and its PSS signature
//!    verifies. Any other scheme is refused.
//! 4. The BCT: up to [`BCT_SEARCH_BLOCKS`] blocks of 16 KiB from the start
//!    of the boot partitions are tried, each holding a BCT at its start
//!    (the BCT slots of switchbrew's boot partition layout are 0x4000
//!    apart; 64 blocks is cbootimage's `NVBOOT_MAX_BCT_SEARCH_BLOCKS`). A
//!    BCT is accepted when its boot data version is `BOOTDATA_VERSION_T210`,
//!    the SHA-256 of its modulus (0x210, 0x100 bytes) is the fused key hash
//!    (switchbrew "Fuses": `FUSE_PUBLIC_KEY` is "the SHA256 hash of the
//!    2048-bit RSA key expected at BCT+0x210"), its PSS signature (0x320)
//!    verifies over 0x510 to its end (the signed section; switchbrew "BCT",
//!    cbootimage), and it describes what is modelled: 16 KiB blocks of
//!    512-byte pages, one SDMMC parameter set, no bad blocks, one to four
//!    bootloaders. \[assumption\] The first accepted copy in block order
//!    wins; what NVIDIA's "journal blocks" search does when several are
//!    valid is not public. Normally block 0 (the normal BCT) is valid and
//!    nothing else is looked at.
//! 5. The bootloader: entries are tried in order (\[assumption\], the
//!    meaning of `EnableFailBack`, 0 in Nintendo's BCTs, is not public).
//!    An entry's image starts at its start block and page, the two boot
//!    partitions counted as one (the normal BCT points at blocks 0x40 and
//!    0x50 in boot partition 0, the safe mode one at 0x100 and 0x110, which
//!    is boot partition 1; switchbrew "BCT"). It must load into IRAM, above
//!    what the ROM leaves there, with its entry point inside, and its PSS
//!    signature (entry + 0x2C) must verify over its `Length` bytes with the
//!    BCT's key. \[inference\] It is not decrypted: the SBK is unique per
//!    console (switchbrew "Cryptosystem"), yet every console takes the
//!    firmware's package1 as the updater writes it, and that is plaintext.
//! 6. IRAM ([`IRAM_BASE`]): the ROM keeps the boot information table (BIT)
//!    at its start and a copy of the BCT at [`BCT_ADDRESS`]. Package1ldr
//!    passes 0x40000000 and 0x40000100 to its version check, which reads
//!    the BIT's bootloader states at 0x50 (0x18 bytes each) and takes the
//!    version (BCT + 0x2330 + 0x12C each) of the one whose first word is 1;
//!    switchbrew's description of package1ldr reads the BCT pointer at
//!    offset 0x4C of the BIT, and U-Boot's Tegra210 header puts the BCT
//!    size at 0x48 and the BCT pointer at 0x4C. The boot type at 0x0C
//!    (U-Boot: there, 2 is recovery mode) is 1 on a cold boot: the NX
//!    bootloader of 23.5.0 refuses to go on otherwise. Only those words are
//!    written; the rest of the BIT, the ROM's work area and stack
//!    (coreboot: the ROM uses 0x40000000..0x4000E000; Fusée Gelée: its
//!    stack grows down from 0x40010000) and IRAM the bootloader does not
//!    fill are listed in [`BootRomHandoff::iram_undefined`].
//! 7. The security engine: the SBK (`FUSE_PRIVATE_KEY0-3`) is left in
//!    keyslot 14 and the SSK, derived from it as switchbrew "Cryptosystem"
//!    describes, in keyslot 15, neither readable (the NX bootloader's
//!    hardware-state check expects their access registers to read 0x7E, all
//!    permissions but key read). The private key fuses then read as all
//!    ones (switchbrew "Fuses"); [`fuses_after_boot`] is what the fuse
//!    block shows from then on.
//! 8. The lock-out (Fusée Gelée: applied before the bootloader starts):
//!    the IROM is hidden, `SB_CSR` bit 4 (`PIROM_DISABLE`) set
//!    ([`ROM_LOCKOUT`]) from `SB_PIROM_START` on, which is 0x1000 from
//!    reset ([`crate::soc::sb`]). The ROM's last store before the
//!    bootloader (ROM `0x100FDC`).
//! 9. The BPMP starts at the entry point in ARM state (package1ldr's first
//!    instruction, at 0x40010040, is the ARM `ldr sp, =0x40008000`).
//!
//! Not established by any public source, and therefore not in the handoff
//! (the machine keeps its own reset values, or refuses): what the ROM
//! leaves in the clock and reset controller and the PLLs, in PMC
//! registers, in the exception vectors (package1ldr overwrites all eight
//! as its first stores), in the rest of the SB registers, and in the
//! security engine beyond the two keyslots (RSA keyslots, configuration).
//! The NX bootloader's hardware-state check gives the values many of these
//! have after package1ldr and the TSEC firmware ran, which bounds them; it
//! belongs to the devices' own power-on state, not to this module. The
//! BPMP's mode and interrupt masks at entry are an \[assumption\], see
//! [`BootRomHandoff::cpsr`].

use std::{fmt, ops::Range};

use aes::{
    Aes128,
    cipher::{BlockEncrypt, KeyInit},
};
use rsa::{BigUint, RsaPublicKey, pss::Pss};
use sha2::{Digest, Sha256};

use crate::{
    keys::Key,
    nand::boot_images::{
        BCT_BLOCK_SIZE_LOG2, BCT_BOOT_DATA_VERSION, BCT_BOOTLOADER_SIZE, BCT_BOOTLOADERS,
        BCT_BOOTLOADERS_USED, BCT_PAGE_SIZE_LOG2, BCT_SIZE, BOOT_DATA_VERSION_T210,
        BOOT_PARTITION_SIZE, Bootloader,
    },
    soc::se::{self, SeContext},
};

/// The BPMP's view of IRAM: 256 KiB.
pub const IRAM_BASE: u32 = 0x4000_0000;
pub const IRAM_SIZE: usize = 0x4_0000;
const IRAM_END: u32 = IRAM_BASE + IRAM_SIZE as u32;

/// Words of the BIT this module fills, offsets from [`IRAM_BASE`].
const BIT_BOOT_TYPE: u32 = 0x0C;
/// `BootType` of a cold boot.
const BOOT_TYPE_COLD: u32 = 1;
const BIT_BCT_SIZE: u32 = 0x48;
const BIT_BCT_POINTER: u32 = 0x4C;
/// `BlState[4]`, of which only the first word, the status, is known.
const BIT_BOOTLOADER_STATES: u32 = 0x50;
const BIT_BOOTLOADER_STATE_SIZE: u32 = 0x18;
/// The status package1ldr looks for: the entry the ROM started.
const BOOTLOADER_STARTED: u32 = 1;
/// Where the ROM keeps its copy of the BCT.
pub const BCT_ADDRESS: u32 = 0x4000_0100;
/// Below this the ROM keeps its work area and stack.
const ROM_STACK_TOP: u32 = 0x4001_0000;

/// The BCT search: blocks of 16 KiB, pages of 512 bytes.
const BLOCK_SIZE_LOG2: u32 = 14;
const PAGE_SIZE_LOG2: u32 = 9;
pub const BCT_SEARCH_BLOCKS: u32 = 64;

/// Fields of an Erista BCT (switchbrew "BCT"; cbootimage).
const BCT_BAD_BLOCKS: Range<usize> = 0x6..0x206;
const BCT_MODULUS: Range<usize> = 0x210..0x310;
const BCT_SIGNATURE: Range<usize> = 0x320..0x420;
const BCT_SIGNED_START: usize = 0x510;
const BCT_PARAM_SETS: usize = 0x540;
const BCT_DEVICE_TYPE: usize = 0x544;
/// `nvboot_dev_type_sdmmc`.
const DEVICE_TYPE_SDMMC: u32 = 4;
const MAX_BOOTLOADERS: u32 = 4;
/// In a bootloader entry: seven words, the 16-byte CMAC, then the PSS
/// signature.
const BOOTLOADER_SIGNATURE: usize = 0x2C;

/// RSASSA-PSS as the ROM checks it. The modulus and the signature are
/// stored least significant byte first; the exponent is not stored, and is
/// 65537; SHA-256 with MGF1-SHA-256 and a 32-byte salt. All of this is
/// what makes the firmware's signatures verify (see the tests).
const RSA_EXPONENT: u32 = 65537;
const PSS_SALT_LENGTH: usize = 32;
const RSA_SIZE: usize = 0x100;

/// The SHA-256 of the modulus in the Erista BCTs of firmware 23.5.0 (normal
/// and safe mode alike): what `FUSE_PUBLIC_KEY0-7` hold on a retail Erista,
/// derived, not read from one. `FUSE_PUBLIC_KEY0` holds bytes 0-3 as a
/// little-endian word, and so on: the ROM copies the fuse words out that
/// way and compares the bytes with the SHA-256 the security engine writes
/// to memory (`0x102248`, `0x102896`).
pub const ERISTA_PUBLIC_KEY_HASH: [u8; 32] = [
    0x4F, 0xBD, 0xB8, 0x60, 0x7C, 0x0B, 0xCC, 0x95, 0x20, 0xD7, 0x86, 0x34, 0xCB, 0x1E, 0xD2, 0xEE,
    0x96, 0x64, 0xE6, 0x66, 0x5D, 0xFE, 0xED, 0xCE, 0x18, 0x59, 0x13, 0xD1, 0x00, 0xE1, 0x39, 0x7E,
];

/// Fuse cache registers, offsets in the fuse block at 0x7000F800
/// (switchbrew "Fuses").
const FUSE_PRIVATEKEYDISABLE: u64 = 0x028;
const FUSE_SKU_INFO: u64 = 0x110;
/// `FUSE_SECURITY_MODE` (0x7000F9A0): 1 when the ODM production fuse is
/// burnt, as on every retail unit ([`Fuses::retail`]'s `odm_production`;
/// Fusée Gelée). The warmboot firmware reads it (`0x400104BC`) and takes
/// its production path on 1.
const FUSE_SECURITY_MODE: u64 = 0x1A0;
const FUSE_PUBLIC_KEY0: u64 = 0x164;
const FUSE_PRIVATE_KEY0: u64 = 0x1A4;
/// `FUSE_PRODUCTION_MODE` (0x7000F900): burnt on every chip NVIDIA ships
/// \[inference: it marks a chip as past NVIDIA's own production; the ROM
/// treats one without it as a pre-production part, `0x100294`\].
/// `FUSE_FA` (0x7000F948): failure analysis mode, not burnt on a unit in
/// the field.
const FUSE_PRODUCTION_MODE: u64 = 0x100;
const FUSE_FA: u64 = 0x148;
/// `FUSE_RESERVED_SW` (0x7000F9C0) and `FUSE_OPT_VENDOR_CODE` (0x7000FA00),
/// the first of the die's identity fuses ([`ChipFuses`], in its order,
/// with the unnamed one at 0x7000FA20).
const FUSE_RESERVED_SW: u64 = 0x1C0;
const FUSE_OPT_VENDOR_CODE: u64 = 0x200;
/// "Must be 0x83"; package1ldr compares it (`0x400100D8`).
const SKU_ID: u32 = 0x83;

/// `FUSE_RESERVED_SW`: the boot device, and whether it overrides the
/// straps.
const RESERVED_SW_BOOT_DEVICE: u32 = 0b111;
const RESERVED_SW_SKIP_STRAPS: u32 = 1 << 3;
/// Recovery mode asks for two buttons (VOLUME_UP and HOME): set on
/// production units (switchbrew "Fuses"). Bit 7, recovery over XUSB, is
/// clear on the original Erista units.
const RESERVED_SW_TWO_BUTTON_RCM: u32 = 1 << 6;
/// The SDMMC is boot device 0: the real ROM, given 0, reads its BCT from
/// the eMMC (`0x101EFE`).
const FUSE_BOOT_DEVICE_SDMMC: u32 = 0;
/// Fuses only the real ROM reads (switchbrew "Fuses" for the names).
/// `FUSE_BOOT_DEVICE_INFO`: the boot device's configuration index, of
/// which the ROM takes bits 13:0 (`0x102156`). \[assumption\] 0, the eMMC's
/// default configuration: a retail unit's word is not public.
const FUSE_BOOT_DEVICE_INFO: u64 = 0x1BC;
/// `FUSE_PKC_DISABLE`: 0, since a retail unit's ROM checks the BCT's RSA
/// signature (Fusée Gelée; the ROM takes its public-key scheme only on 0,
/// `0x102020`).
const FUSE_PKC_DISABLE: u64 = 0x268;
/// `FUSE_ODM_INFO`: `0x1F00` on production units, every debug signal
/// disabled (switchbrew "Fuses").
const FUSE_ODM_INFO: u64 = 0x29C;
const ODM_INFO_PRODUCTION: u32 = 0x1F00;
/// `FUSE_SECURE_PROVISION_INDEX` and `_INFO`: NVIDIA's factory
/// provisioning, which picks a provisioning key by the index and, by bit
/// 1 of the info word, keeps the ROM out of its ODM production mode
/// (`0x100286`, `0x10033A`). 0 on a unit in the field \[inference: its ROM
/// is in that mode, which is why it insists on the BCT's signature\].
const FUSE_SECURE_PROVISION_INDEX: u64 = 0x2E8;
const FUSE_SECURE_PROVISION_INFO: u64 = 0x2EC;

/// `PMC_SCRATCH0` bits the ROM tests first.
const SCRATCH0_WARM_BOOT: u32 = 1;
const SCRATCH0_FORCE_RECOVERY: u32 = 1 << 1;

/// The security engine keyslots the ROM fills.
const KEYSLOT_SBK: u8 = 14;
const KEYSLOT_SSK: u8 = 15;

/// The fuses the cold boot reads.
#[derive(Clone)]
pub struct Fuses {
    /// `FUSE_SECURITY_MODE`: the ODM production fuse, burnt on consumer
    /// units (Fusée Gelée).
    pub odm_production: bool,
    /// `FUSE_PUBLIC_KEY0-7`.
    pub public_key_hash: [u8; 32],
    /// `FUSE_PRIVATE_KEY0-3`: the secure boot key, the key file's
    /// `secure_boot_key`.
    pub secure_boot_key: Key,
    /// `FUSE_PRIVATE_KEY4`: the 32-bit device key.
    pub device_key: u32,
    /// `FUSE_RESERVED_SW`.
    pub reserved_sw: u32,
    pub chip: ChipFuses,
}

impl Fuses {
    /// A retail Erista with this console's private key fuses and chip
    /// identity. `reserved_sw` is the production units' two-button recovery
    /// mode and \[assumption\] "skip the straps, boot the SDMMC": a retail
    /// unit boots its eMMC, but the rest of its fuse word is not public.
    pub fn retail(secure_boot_key: Key, device_key: u32, chip: ChipFuses) -> Self {
        Self {
            odm_production: true,
            public_key_hash: ERISTA_PUBLIC_KEY_HASH,
            secure_boot_key,
            device_key,
            reserved_sw: RESERVED_SW_TWO_BUTTON_RCM
                | RESERVED_SW_SKIP_STRAPS
                | FUSE_BOOT_DEVICE_SDMMC,
            chip,
        }
    }

    /// The fuse cache words these fuses give at reset, before the ROM has
    /// run, as `(offset, value)` in the block at 0x7000F800: what
    /// [`fuses_after_boot`] gives with the private key fuses readable
    /// (`FUSE_PRIVATE_KEY0-3` the secure boot key as the key file has it,
    /// each word little-endian, the way open-source tools copy the fuse
    /// words out; `FUSE_PRIVATE_KEY4` the device key) and
    /// `FUSE_PRIVATEKEYDISABLE` clear, and the fuses only the ROM reads:
    /// `FUSE_RESERVED_SW`, the die's identity, and those of its boot
    /// device and mode (see their constants).
    pub fn at_reset(&self) -> Vec<(u64, u32)> {
        let hash = self.public_key_hash.as_chunks::<4>().0.iter();
        let key = self.secure_boot_key.as_chunks::<4>().0.iter();
        let chip = &self.chip;
        let mut words = vec![
            (FUSE_PRODUCTION_MODE, 1),
            (FUSE_FA, 0),
            (FUSE_SKU_INFO, SKU_ID),
            (FUSE_SECURITY_MODE, u32::from(self.odm_production)),
            (FUSE_PRIVATEKEYDISABLE, 0),
            (FUSE_PRIVATE_KEY0 + 16, self.device_key),
            (FUSE_RESERVED_SW, self.reserved_sw),
            (FUSE_BOOT_DEVICE_INFO, 0),
            (FUSE_PKC_DISABLE, 0),
            (FUSE_ODM_INFO, ODM_INFO_PRODUCTION),
            (FUSE_SECURE_PROVISION_INDEX, 0),
            (FUSE_SECURE_PROVISION_INFO, 0),
            (FUSE_OPT_VENDOR_CODE, chip.vendor_code),
            (FUSE_OPT_VENDOR_CODE + 0x04, chip.fab_code),
            (FUSE_OPT_VENDOR_CODE + 0x08, chip.lot_code_0),
            (FUSE_OPT_VENDOR_CODE + 0x0C, chip.lot_code_1),
            (FUSE_OPT_VENDOR_CODE + 0x10, chip.wafer_id),
            (FUSE_OPT_VENDOR_CODE + 0x14, chip.x_coordinate),
            (FUSE_OPT_VENDOR_CODE + 0x18, chip.y_coordinate),
            (FUSE_OPT_VENDOR_CODE + 0x20, chip.unknown_0x220),
        ];
        for (index, word) in hash.enumerate() {
            let value = u32::from_le_bytes(*word);
            words.push((FUSE_PUBLIC_KEY0 + 4 * index as u64, value));
        }
        for (index, word) in key.enumerate() {
            let value = u32::from_le_bytes(*word);
            words.push((FUSE_PRIVATE_KEY0 + 4 * index as u64, value));
        }
        words
    }
}

/// Key values stay out of logs.
impl fmt::Debug for Fuses {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Fuses")
            .field("odm_production", &self.odm_production)
            .field("public_key_hash", &self.public_key_hash)
            .field("reserved_sw", &self.reserved_sw)
            .field("chip", &self.chip)
            .finish_non_exhaustive()
    }
}

/// The fuses identifying the die, from which the ROM derives the SSK
/// (switchbrew "Cryptosystem", by address 0x7000FA00 onward). They belong
/// to one chip; which values a console has is a choice, like its device
/// ID. Only the SSK depends on them, and on a retail unit package1ldr
/// clears the SSK's keyslot without using it (switchbrew "Package1").
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ChipFuses {
    pub vendor_code: u32,
    pub fab_code: u32,
    pub lot_code_0: u32,
    pub lot_code_1: u32,
    pub wafer_id: u32,
    pub x_coordinate: u32,
    pub y_coordinate: u32,
    /// The unnamed fuse at 0x7000FA20.
    pub unknown_0x220: u32,
}

/// The fuse cache words the fuse block shows a retail Erista after the
/// ROM, as `(offset, value)` in the block at 0x7000F800: the SKU, the
/// security mode (production), the key hash, and the private key fuses
/// hidden (all ones) with `FUSE_PRIVATEKEYDISABLE` set (\[inference\] it
/// reads back as written, bit 0).
pub fn fuses_after_boot() -> impl Iterator<Item = (u64, u32)> {
    let key_hash = ERISTA_PUBLIC_KEY_HASH
        .as_chunks::<4>()
        .0
        .iter()
        .enumerate()
        .map(|(index, word)| {
            (
                FUSE_PUBLIC_KEY0 + 4 * index as u64,
                u32::from_le_bytes(*word),
            )
        });
    let private_key = (0..5).map(|index| (FUSE_PRIVATE_KEY0 + 4 * index, u32::MAX));
    [
        (FUSE_SKU_INFO, SKU_ID),
        (FUSE_SECURITY_MODE, 1),
        (FUSE_PRIVATEKEYDISABLE, 1),
    ]
    .into_iter()
    .chain(key_hash)
    .chain(private_key)
}

/// What the ROM leaves when it starts the bootloader. It has written
/// IRAM itself (the `iram` [`cold_boot`] was given).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BootRomHandoff {
    /// Where the BPMP starts, in ARM state.
    pub entry: u32,
    /// The BPMP's CPSR at `entry`: \[assumption\] Supervisor mode, IRQ and
    /// FIQ masked, as out of reset. Package1ldr sets its own stack pointer
    /// and does not read the mode; a dump would settle it.
    pub cpsr: u32,
    /// The block whose BCT was taken.
    pub bct_block: u32,
    /// The bootloader entry of that BCT that was started.
    pub bootloader: usize,
    /// What the ROM wrote into IRAM, by address.
    pub iram_written: Vec<Range<u32>>,
    /// The rest of IRAM: what the real ROM leaves there is not established.
    /// Reads of it should be refused with the note until the guest has
    /// written the bytes itself.
    pub iram_undefined: Vec<(Range<u32>, &'static str)>,
    /// SBK, then SSK.
    pub keyslots: [SeKeyslot; 2],
}

/// A security engine AES keyslot the ROM fills.
#[derive(Clone, PartialEq, Eq)]
pub struct SeKeyslot {
    pub slot: u8,
    pub key: Key,
    pub access: KeyslotAccess,
}

/// Key values stay out of logs.
impl fmt::Debug for SeKeyslot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SeKeyslot")
            .field("slot", &self.slot)
            .field("access", &self.access)
            .finish_non_exhaustive()
    }
}

/// What the keyslot's access register allows. The ROM disables reads of
/// both of its keyslots (switchbrew "Cryptosystem"); use and update stay
/// allowed, as do the IV permissions: the NX bootloader expects both
/// access registers to read 0x7E (permission bits 6:1 set, key read clear).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyslotAccess {
    pub readable: bool,
    pub updatable: bool,
    pub usable: bool,
}

const ROM_KEYSLOT_ACCESS: KeyslotAccess = KeyslotAccess {
    readable: false,
    updatable: true,
    usable: true,
};

/// Supervisor mode (0x13), IRQ and FIQ masked, ARM state.
const ENTRY_CPSR: u32 = 0xD3;

/// Bits of a register the ROM leaves set as given, the rest of the register
/// as the device's power-on value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegisterBits {
    pub address: u32,
    pub mask: u32,
    pub value: u32,
}

/// The ROM's lock-out: `SB_CSR` (0x6000C200, the NX bootloader checks bits
/// 15:4 and 1:0) with `PIROM_DISABLE`.
pub const ROM_LOCKOUT: RegisterBits = RegisterBits {
    address: 0x6000_C200,
    mask: 0xFFF3,
    value: 1 << 4,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BootRomError {
    /// `iram` is not [`IRAM_SIZE`] bytes.
    IramSize(usize),
    /// `PMC_SCRATCH0` asks for a warm boot: [`warm_boot`] runs that.
    WarmBoot,
    /// `PMC_SCRATCH0` forces recovery mode.
    ForcedRecovery,
    /// The fuses select a boot path that is not modelled: which.
    Unmodelled(&'static str),
    /// The fuses select a boot device other than the SDMMC.
    BootDevice(u32),
    /// No searched block holds an acceptable BCT: each block and why. The
    /// ROM would enter recovery mode, which is not modelled.
    NoBct(Vec<(u32, BctRejection)>),
    /// The BCT taken has no acceptable bootloader: each entry and why.
    NoBootloader {
        bct_block: u32,
        rejected: Vec<(usize, BootloaderRejection)>,
    },
    /// The bootloader started has an odd entry point: whether the ROM
    /// would enter it in Thumb state is not established.
    ThumbEntry(u32),
}

impl fmt::Display for BootRomError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IramSize(size) => write!(f, "IRAM is {size} bytes, not {IRAM_SIZE}"),
            Self::WarmBoot => write!(f, "PMC_SCRATCH0 asks for a warm boot, not modelled"),
            Self::ForcedRecovery => write!(f, "PMC_SCRATCH0 forces recovery mode, not modelled"),
            Self::Unmodelled(what) => write!(f, "boot ROM: {what}, not modelled"),
            Self::BootDevice(device) => {
                write!(f, "the fuses select boot device {device}, not the SDMMC")
            }
            Self::NoBct(rejected) => {
                write!(f, "no valid BCT (the ROM would enter recovery mode):")?;
                for (block, why) in rejected {
                    write!(f, " block {block:#x}: {why};")?;
                }
                Ok(())
            }
            Self::NoBootloader {
                bct_block,
                rejected,
            } => {
                write!(
                    f,
                    "the BCT of block {bct_block:#x} has no valid bootloader:"
                )?;
                for (index, why) in rejected {
                    write!(f, " entry {index}: {why};")?;
                }
                Ok(())
            }
            Self::ThumbEntry(entry) => write!(
                f,
                "bootloader entry point {entry:#010x} is odd: a Thumb entry is not established"
            ),
        }
    }
}

impl std::error::Error for BootRomError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BctRejection {
    /// The block lies past the boot partitions.
    OutsideMedia,
    BootDataVersion(u32),
    /// The modulus is not the one the fuses hold the hash of.
    PublicKeyHash,
    Signature,
    /// Block and page sizes other than those searched with.
    Geometry {
        block_size_log2: u32,
        page_size_log2: u32,
    },
    /// Not exactly one parameter set, for the SDMMC.
    Device {
        param_sets: u32,
        device_type: u32,
    },
    /// The bad block table marks blocks: not modelled.
    BadBlocks,
    BootloaderCount(u32),
}

impl fmt::Display for BctRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutsideMedia => write!(f, "past the boot partitions"),
            Self::BootDataVersion(version) => write!(f, "boot data version {version:#x}"),
            Self::PublicKeyHash => write!(f, "its key is not the fused one"),
            Self::Signature => write!(f, "its signature does not verify"),
            Self::Geometry {
                block_size_log2,
                page_size_log2,
            } => write!(
                f,
                "blocks of 2^{block_size_log2} and pages of 2^{page_size_log2} bytes"
            ),
            Self::Device {
                param_sets,
                device_type,
            } => write!(f, "{param_sets} parameter sets, device type {device_type}"),
            Self::BadBlocks => write!(f, "bad blocks are marked"),
            Self::BootloaderCount(count) => write!(f, "{count} bootloaders"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BootloaderRejection {
    /// The image lies past the boot partitions, or across their boundary.
    OutsideMedia,
    /// The image does not load into free IRAM.
    LoadRange {
        address: u32,
        length: u32,
    },
    /// The entry point lies outside the image.
    EntryPoint(u32),
    Signature,
}

impl fmt::Display for BootloaderRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutsideMedia => write!(f, "past the boot partitions"),
            Self::LoadRange { address, length } => {
                write!(f, "{length:#x} bytes at {address:#010x} are not free IRAM")
            }
            Self::EntryPoint(entry) => write!(f, "entry point {entry:#010x} outside the image"),
            Self::Signature => write!(f, "its signature does not verify"),
        }
    }
}

/// Runs the ROM's cold boot from the eMMC whose boot partitions are
/// `boot0` and `boot1` (each at most [`BOOT_PARTITION_SIZE`]), with these
/// fuses and this `PMC_SCRATCH0`, into `iram` ([`IRAM_SIZE`] bytes at
/// [`IRAM_BASE`]).
pub fn cold_boot(
    boot0: &[u8],
    boot1: &[u8],
    fuses: &Fuses,
    pmc_scratch0: u32,
    iram: &mut [u8],
) -> Result<BootRomHandoff, BootRomError> {
    if iram.len() != IRAM_SIZE {
        return Err(BootRomError::IramSize(iram.len()));
    }
    if pmc_scratch0 & SCRATCH0_WARM_BOOT != 0 {
        return Err(BootRomError::WarmBoot);
    }
    if pmc_scratch0 & SCRATCH0_FORCE_RECOVERY != 0 {
        return Err(BootRomError::ForcedRecovery);
    }
    if fuses.reserved_sw & RESERVED_SW_SKIP_STRAPS == 0 {
        return Err(BootRomError::Unmodelled(
            "the boot device straps (FUSE_RESERVED_SW does not skip them)",
        ));
    }
    let device = fuses.reserved_sw & RESERVED_SW_BOOT_DEVICE;
    if device != FUSE_BOOT_DEVICE_SDMMC {
        return Err(BootRomError::BootDevice(device));
    }
    secure_scheme(fuses).map_err(BootRomError::Unmodelled)?;
    let media = Media { boot0, boot1 };

    let mut rejected = Vec::new();
    let (bct_block, bct) = 'search: {
        for block in 0..BCT_SEARCH_BLOCKS {
            let Some(bct) = media.read(u64::from(block) << BLOCK_SIZE_LOG2, BCT_SIZE) else {
                rejected.push((block, BctRejection::OutsideMedia));
                break;
            };
            match check_bct(bct, fuses) {
                Ok(()) => break 'search (block, bct),
                Err(why) => rejected.push((block, why)),
            }
        }
        return Err(BootRomError::NoBct(rejected));
    };

    let mut rejected = Vec::new();
    let used = u32_at(bct, BCT_BOOTLOADERS_USED) as usize;
    let (index, bootloader, image) = 'search: {
        for index in 0..used {
            match check_bootloader(bct, index, &media) {
                Ok((bootloader, image)) => break 'search (index, bootloader, image),
                Err(why) => rejected.push((index, why)),
            }
        }
        return Err(BootRomError::NoBootloader {
            bct_block,
            rejected,
        });
    };
    if bootloader.entry_point & 1 != 0 {
        return Err(BootRomError::ThumbEntry(bootloader.entry_point));
    }

    iram.fill(0);
    let mut written = Vec::new();
    let mut put = |address: u32, bytes: &[u8]| {
        let start = (address - IRAM_BASE) as usize;
        iram[start..start + bytes.len()].copy_from_slice(bytes);
        written.push(address..address + bytes.len() as u32);
    };
    put(IRAM_BASE + BIT_BOOT_TYPE, &BOOT_TYPE_COLD.to_le_bytes());
    put(IRAM_BASE + BIT_BCT_SIZE, &(BCT_SIZE as u32).to_le_bytes());
    put(IRAM_BASE + BIT_BCT_POINTER, &BCT_ADDRESS.to_le_bytes());
    put(
        IRAM_BASE + BIT_BOOTLOADER_STATES + index as u32 * BIT_BOOTLOADER_STATE_SIZE,
        &BOOTLOADER_STARTED.to_le_bytes(),
    );
    put(BCT_ADDRESS, bct);
    put(bootloader.load_address, image);
    written.sort_by_key(|range| range.start);
    written.dedup_by(|next, previous| {
        let adjacent = next.start == previous.end;
        if adjacent {
            previous.end = next.end;
        }
        adjacent
    });
    let iram_undefined = undefined(&written, &COLD_REGIONS);

    let keyslot = |slot, key| SeKeyslot {
        slot,
        key,
        access: ROM_KEYSLOT_ACCESS,
    };
    Ok(BootRomHandoff {
        entry: bootloader.entry_point,
        cpsr: ENTRY_CPSR,
        bct_block,
        bootloader: index,
        iram_written: written,
        iram_undefined,
        keyslots: [
            keyslot(KEYSLOT_SBK, fuses.secure_boot_key),
            keyslot(KEYSLOT_SSK, secure_storage_key(fuses)),
        ],
    })
}

/// The two boot partitions, read as one device.
struct Media<'a> {
    boot0: &'a [u8],
    boot1: &'a [u8],
}

impl<'a> Media<'a> {
    /// `length` bytes at `offset`, within one partition.
    fn read(&self, offset: u64, length: usize) -> Option<&'a [u8]> {
        let (partition, offset) = if offset < BOOT_PARTITION_SIZE {
            (self.boot0, offset)
        } else {
            (self.boot1, offset - BOOT_PARTITION_SIZE)
        };
        let end = offset.checked_add(length as u64)?;
        if end > BOOT_PARTITION_SIZE {
            return None;
        }
        partition.get(usize::try_from(offset).ok()?..usize::try_from(end).ok()?)
    }
}

fn check_bct(bct: &[u8], fuses: &Fuses) -> Result<(), BctRejection> {
    let version = u32_at(bct, BCT_BOOT_DATA_VERSION);
    if version != BOOT_DATA_VERSION_T210 {
        return Err(BctRejection::BootDataVersion(version));
    }
    let modulus = &bct[BCT_MODULUS];
    if Sha256::digest(modulus)[..] != fuses.public_key_hash {
        return Err(BctRejection::PublicKeyHash);
    }
    if !pss_verifies(modulus, &bct[BCT_SIGNATURE], &bct[BCT_SIGNED_START..]) {
        return Err(BctRejection::Signature);
    }
    let (block_size_log2, page_size_log2) = (
        u32_at(bct, BCT_BLOCK_SIZE_LOG2),
        u32_at(bct, BCT_PAGE_SIZE_LOG2),
    );
    if (block_size_log2, page_size_log2) != (BLOCK_SIZE_LOG2, PAGE_SIZE_LOG2) {
        return Err(BctRejection::Geometry {
            block_size_log2,
            page_size_log2,
        });
    }
    let (param_sets, device_type) = (u32_at(bct, BCT_PARAM_SETS), u32_at(bct, BCT_DEVICE_TYPE));
    if (param_sets, device_type) != (1, DEVICE_TYPE_SDMMC) {
        return Err(BctRejection::Device {
            param_sets,
            device_type,
        });
    }
    if bct[BCT_BAD_BLOCKS].iter().any(|&bits| bits != 0) {
        return Err(BctRejection::BadBlocks);
    }
    let count = u32_at(bct, BCT_BOOTLOADERS_USED);
    if !(1..=MAX_BOOTLOADERS).contains(&count) {
        return Err(BctRejection::BootloaderCount(count));
    }
    Ok(())
}

fn check_bootloader<'a>(
    bct: &[u8],
    index: usize,
    media: &Media<'a>,
) -> Result<(Bootloader, &'a [u8]), BootloaderRejection> {
    let bootloader = Bootloader::of(bct, index);
    let (address, length) = (bootloader.load_address, bootloader.length);
    let image = media
        .read(bootloader.offset(bct), length as usize)
        .ok_or(BootloaderRejection::OutsideMedia)?;
    let end = address.checked_add(length);
    if address < BCT_ADDRESS + BCT_SIZE as u32 || end.is_none_or(|end| end > IRAM_END) {
        return Err(BootloaderRejection::LoadRange { address, length });
    }
    if !(address..address + length).contains(&bootloader.entry_point) {
        return Err(BootloaderRejection::EntryPoint(bootloader.entry_point));
    }
    let signature = BCT_BOOTLOADERS + index * BCT_BOOTLOADER_SIZE + BOOTLOADER_SIGNATURE;
    if !pss_verifies(
        &bct[BCT_MODULUS],
        &bct[signature..signature + RSA_SIZE],
        image,
    ) {
        return Err(BootloaderRejection::Signature);
    }
    Ok((bootloader, image))
}

/// Whether `signature` signs `message` under `modulus`, both as the ROM
/// stores them.
fn pss_verifies(modulus: &[u8], signature: &[u8], message: &[u8]) -> bool {
    let Ok(key) = RsaPublicKey::new(BigUint::from_bytes_le(modulus), BigUint::from(RSA_EXPONENT))
    else {
        return false;
    };
    let signature: Vec<u8> = signature.iter().rev().copied().collect();
    key.verify(
        Pss::new_with_salt::<Sha256>(PSS_SALT_LENGTH),
        &Sha256::digest(message),
        &signature,
    )
    .is_ok()
}

/// The SSK as switchbrew "Cryptosystem" describes the ROM deriving it:
/// the device key four times, encrypted with the SBK, XORed with the
/// chip's identity, encrypted with the SBK again.
fn secure_storage_key(fuses: &Fuses) -> Key {
    let chip = &fuses.chip;
    let vendor_code = chip.vendor_code & 0xF;
    let fab_code = chip.fab_code & 0x3F;
    let lot_code_0 = chip.lot_code_0;
    let lot_code_1 = chip.lot_code_1 & 0x0FFF_FFFF;
    let wafer_id = chip.wafer_id & 0x3F;
    let x_coordinate = chip.x_coordinate & 0x1FF;
    let y_coordinate = chip.y_coordinate & 0x1FF;
    let unknown = chip.unknown_0x220 & 0x3F;
    let identity = [
        lot_code_1 << 30 | wafer_id << 24 | x_coordinate << 15 | y_coordinate << 6 | unknown,
        lot_code_0 << 26 | lot_code_1 >> 2,
        fab_code << 26 | lot_code_0 >> 6,
        vendor_code,
    ];
    let cipher = Aes128::new((&fuses.secure_boot_key).into());
    let mut block = [0; 16];
    for word in block.as_chunks_mut::<4>().0 {
        *word = fuses.device_key.to_le_bytes();
    }
    let mut block: aes::Block = block.into();
    cipher.encrypt_block(&mut block);
    for (bytes, word) in block.as_chunks_mut::<4>().0.iter_mut().zip(identity) {
        for (byte, mask) in bytes.iter_mut().zip(word.to_le_bytes()) {
            *byte ^= mask;
        }
    }
    cipher.encrypt_block(&mut block);
    block.into()
}

/// What IRAM holds after a cold boot where the ROM did not write it.
const COLD_REGIONS: [(Range<u32>, &str); 3] = [
    (
        IRAM_BASE..BCT_ADDRESS,
        "boot information table field the ROM fills; its value is not established",
    ),
    (
        BCT_ADDRESS..ROM_STACK_TOP,
        "the boot ROM's work area and stack; what it leaves there is not established",
    ),
    (
        ROM_STACK_TOP..IRAM_END,
        "IRAM the boot ROM did not load; its contents at power-on are not established",
    ),
];

/// The secure boot scheme this module models: PKC on an ODM
/// production-fused unit (see the module).
fn secure_scheme(fuses: &Fuses) -> Result<(), &'static str> {
    if !fuses.odm_production {
        return Err("a unit not ODM production-fused");
    }
    if fuses.public_key_hash == [0; 32] {
        return Err("secure boot by AES-CMAC (no public key hash fused)");
    }
    Ok(())
}

/// The parts of IRAM outside `written` (sorted), each with the note of its
/// region.
fn undefined(
    written: &[Range<u32>],
    regions: &[(Range<u32>, &'static str)],
) -> Vec<(Range<u32>, &'static str)> {
    let mut undefined = Vec::new();
    for (region, note) in regions {
        let (region, note) = (region.clone(), *note);
        let mut start = region.start;
        for range in written {
            if range.end <= start || range.start >= region.end {
                continue;
            }
            if range.start > start {
                undefined.push((start..range.start, note));
            }
            start = start.max(range.end);
        }
        if start < region.end {
            undefined.push((start..region.end, note));
        }
    }
    undefined
}

fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

/// PMC registers the warm path reads, offsets from the PMC at 0x7000E400,
/// with their writers in firmware 23.5.0.
const PMC_SCRATCH0: u32 = 0x050;
/// The warmboot image's address in DRAM, written by the monitor at cold
/// boot (0x8000D000).
const PMC_SCRATCH1: u32 = 0x054;
/// The SE context's address in DRAM, written by the monitor before SC7
/// (TF-A's Tegra210 port: "SE context address").
const PMC_SCRATCH43: u32 = 0x22C;
/// `SECURE_SCRATCH4..7`: the SRK the context was saved with.
const PMC_SRK: [u32; 4] = [0x0C0, 0x0C4, 0x224, 0x228];
/// The PMC registers the NX bootloader packs the SDRAM parameters into
/// (`0x4000D904`), by inclusive word range: 251 words, 171 `SCRATCHn` and
/// the rest `SECURE_SCRATCHn`. coreboot's `sdram_lp0.c` packs the same
/// parameters "into PMC scratch registers (where the BootROM expects them
/// on LP0 resume)".
const SDRAM_SCRATCH: [(u32, u32); 14] = [
    (0x058, 0x09C),
    (0x0A8, 0x0AC),
    (0x0FC, 0x120),
    (0x128, 0x12C),
    (0x13C, 0x13C),
    (0x144, 0x144),
    (0x230, 0x240),
    (0x248, 0x24C),
    (0x300, 0x330),
    (0x37C, 0x41C),
    (0x600, 0x6EC),
    (0x70C, 0x81C),
    (0xA98, 0xAA0),
    (0xAA8, 0xB04),
];

/// The warmboot (LP0) image header (coreboot's `struct lp0_header`,
/// matched against firmware 23.5.0's warmboot).
const WB0_LENGTH_INSECURE: usize = 0x000;
const WB0_MODULUS: Range<usize> = 0x010..0x110;
const WB0_SIGNATURE: Range<usize> = 0x120..0x220;
/// The signature covers the image from the random AES block on, to
/// `length_secure`: of all start offsets, only this one makes 23.5.0's
/// verify.
const WB0_SIGNED_START: usize = 0x220;
const WB0_LENGTH_SECURE: usize = 0x230;
const WB0_DESTINATION: usize = 0x234;
const WB0_ENTRY: usize = 0x238;
const WB0_CODE_LENGTH: usize = 0x23C;
/// The code follows the header.
const WB0_HEADER_SIZE: usize = 0x240;

/// DRAM leaves self refresh: `EMC_SELF_REF` (EMC 0x7001B000 + 0xE0) bit 0,
/// which the monitor's SC7-entry stub set. \[inference\] The ROM clears it
/// as part of its SDRAM restore: the warmboot reads DRAM through the SE
/// right away.
pub const SELF_REFRESH_EXIT: RegisterBits = RegisterBits {
    address: 0x7001_B0E0,
    mask: 1,
    value: 0,
};

/// What IRAM holds after a warm boot where the ROM did not write it: all
/// of IRAM lost power in SC7.
const WARM_REGIONS: [(Range<u32>, &str); 2] = [
    (
        IRAM_BASE..ROM_STACK_TOP,
        "the boot ROM's work area and stack after SC7; what it leaves there is not established",
    ),
    (
        ROM_STACK_TOP..IRAM_END,
        "IRAM lost power in SC7 and the boot ROM did not load this part",
    ),
];

/// What the ROM leaves when it starts the warmboot firmware.
#[derive(Debug)]
pub struct WarmBootHandoff {
    /// Where the BPMP starts, in ARM state: the header's entry point.
    pub entry: u32,
    /// As for the cold boot ([`BootRomHandoff::cpsr`]).
    pub cpsr: u32,
    /// What the ROM wrote into IRAM: the warmboot's code.
    pub iram_written: Range<u32>,
    /// The rest of IRAM; reads of it should be refused with the note until
    /// the guest has written the bytes itself.
    pub iram_undefined: Vec<(Range<u32>, &'static str)>,
    /// Register bits to apply, in order: [`SELF_REFRESH_EXIT`], then
    /// [`ROM_LOCKOUT`] (Fusée Gelée's pseudo-code applies the lock-out
    /// after either path).
    pub registers: Vec<RegisterBits>,
    /// The security engine's state as it was saved before SC7, for
    /// [`se::Se::restore_context`]. \[assumption\] The ROM's own SBK and SSK
    /// loads, if it makes them on this path, precede the restore and are
    /// overwritten by it, so the saved keyslots and access settings are what
    /// the warmboot finds (it checks slots 0-5 read zero and 6-15 read
    /// protected).
    pub se_context: SeContext,
}

#[derive(Debug)]
pub enum WarmBootError {
    /// `iram` is not [`IRAM_SIZE`] bytes.
    IramSize(usize),
    /// `PMC_SCRATCH0` (its value, if it was ever written) does not ask for
    /// a warm boot.
    NotWarm(Option<u32>),
    /// The fuses select a boot path that is not modelled: which.
    Unmodelled(&'static str),
    /// A PMC register the ROM reads on this path was never written: its
    /// offset from 0x7000E400 and what it holds.
    Unwritten {
        offset: u32,
        holds: &'static str,
    },
    /// DRAM the ROM reads is not there.
    Unreadable {
        address: u32,
        length: usize,
        what: &'static str,
    },
    /// The security engine context does not open.
    SeContext(se::ContextError),
    /// The warmboot's modulus is not the one the fuses hold the hash of.
    PublicKeyHash,
    Signature,
    /// A header field this module does not accept: which, and its value.
    Header {
        field: &'static str,
        value: u32,
    },
}

impl fmt::Display for WarmBootError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IramSize(size) => write!(f, "IRAM is {size} bytes, not {IRAM_SIZE}"),
            Self::NotWarm(Some(value)) => {
                write!(f, "PMC_SCRATCH0 is {value:#x}: not a warm boot")
            }
            Self::NotWarm(None) => write!(f, "PMC_SCRATCH0 was never written: not a warm boot"),
            Self::Unmodelled(what) => write!(f, "boot ROM: {what}, not modelled"),
            Self::Unwritten { offset, holds } => write!(
                f,
                "warm boot: PMC+{offset:#x} ({holds}) was never written, so the boot ROM \
                 would restore nothing from it"
            ),
            Self::Unreadable {
                address,
                length,
                what,
            } => write!(
                f,
                "warm boot: {what}: {length:#x} bytes at {address:#010x} unreadable"
            ),
            Self::SeContext(error) => write!(f, "warm boot: SE context: {error}"),
            Self::PublicKeyHash => write!(f, "warmboot: its key is not the fused one"),
            Self::Signature => write!(f, "warmboot: its signature does not verify"),
            Self::Header { field, value } => {
                write!(f, "warmboot header: {field} {value:#x} not accepted")
            }
        }
    }
}

impl std::error::Error for WarmBootError {}

/// Runs the ROM's warm boot (SC7 resume) into `iram` ([`IRAM_SIZE`] bytes
/// at [`IRAM_BASE`]), reading the PMC through `pmc` (an offset from
/// 0x7000E400 to the register's value, `None` if it was never written) and
/// DRAM through `dram` (`length` bytes at a physical address, `None` where
/// there is no DRAM). In the ROM's order:
///
/// 1. `PMC_SCRATCH0` bit 0 says warm boot. The recovery mode bit is not
///    looked at: Fusée Gelée's pseudo-code tests it only on the cold path.
///    Whether the ROM clears bit 0 is not established; it is left as is.
/// 2. SDRAM: the parameters come from the PMC scratch registers the NX
///    bootloader packed them into (251 words); each must have been
///    written. Decoding them and programming the EMC and MC from them is
///    not modelled: \[assumption\] the EMC and MC keep their pre-sleep
///    values across SC7 here, and those are the values the scratch was
///    packed from, so the ROM's programming would write back what is
///    already there. It would also be a large table inferred from the
///    packing, with nothing to check it against. Only the self-refresh exit
///    is given ([`SELF_REFRESH_EXIT`]). The same assumption keeps the clock
///    and reset controller pre-sleep; coreboot notes that the ROM leaves
///    the system clock at PLLP_OUT2 (204 MHz) on this path, but not with
///    which register values, so nothing of it is written.
/// 3. The SE context: [`se::CONTEXT_SIZE`] bytes at the address in
///    `PMC_SCRATCH43`, opened with the SRK from `SECURE_SCRATCH4..7` (TF-A's
///    Tegra210 port: the "BootROM ... verifies and restores the security
///    engine context on warm boot").
/// 4. The warmboot image at `PMC_SCRATCH1`: the SHA-256 of its header
///    modulus must be the fused key hash, as for the BCT, and its PSS
///    signature must verify over 0x220 to `length_secure`. Then its code
///    (from 0x240, `code_length` bytes, all inside the signed part) goes to
///    `destination`, which must be free IRAM above the ROM's stack, with an
///    ARM entry point inside it. \[assumption\] `length_insecure` is how
///    much the ROM reads and is at least `length_secure`; it is at most
///    [`IRAM_SIZE`]. What the real ROM does with a bad image or context is
///    not established: here it is refused.
pub fn warm_boot(
    fuses: &Fuses,
    pmc: &dyn Fn(u32) -> Option<u32>,
    dram: &dyn Fn(u32, usize) -> Option<Vec<u8>>,
    iram: &mut [u8],
) -> Result<WarmBootHandoff, WarmBootError> {
    if iram.len() != IRAM_SIZE {
        return Err(WarmBootError::IramSize(iram.len()));
    }
    let scratch0 = pmc(PMC_SCRATCH0);
    if scratch0.is_none_or(|value| value & SCRATCH0_WARM_BOOT == 0) {
        return Err(WarmBootError::NotWarm(scratch0));
    }
    secure_scheme(fuses).map_err(WarmBootError::Unmodelled)?;
    let read_pmc = |offset, holds| pmc(offset).ok_or(WarmBootError::Unwritten { offset, holds });
    for (first, last) in SDRAM_SCRATCH {
        for offset in (first..=last).step_by(4) {
            read_pmc(offset, "SDRAM parameters")?;
        }
    }
    let read_dram = |address, length, what| {
        dram(address, length)
            .filter(|bytes| bytes.len() == length)
            .ok_or(WarmBootError::Unreadable {
                address,
                length,
                what,
            })
    };

    let context_address = read_pmc(PMC_SCRATCH43, "the SE context's address")?;
    let mut srk_words = [0; 4];
    for (word, offset) in srk_words.iter_mut().zip(PMC_SRK) {
        *word = read_pmc(offset, "the SRK")?;
    }
    let blob = read_dram(context_address, se::CONTEXT_SIZE, "the SE context")?;
    let se_context = SeContext::open(&blob, &se::srk_from_scratch(srk_words))
        .map_err(WarmBootError::SeContext)?;

    let image_address = read_pmc(PMC_SCRATCH1, "the warmboot's address")?;
    let header = read_dram(image_address, WB0_HEADER_SIZE, "the warmboot header")?;
    let field = |offset| u32_at(&header, offset);
    let (length_insecure, length_secure) = (field(WB0_LENGTH_INSECURE), field(WB0_LENGTH_SECURE));
    let header_error = |field, value| WarmBootError::Header { field, value };
    if length_insecure as usize > IRAM_SIZE {
        return Err(header_error("length_insecure", length_insecure));
    }
    if !(WB0_HEADER_SIZE as u32..=length_insecure).contains(&length_secure) {
        return Err(header_error("length_secure", length_secure));
    }
    let image = read_dram(image_address, length_insecure as usize, "the warmboot")?;
    let modulus = &image[WB0_MODULUS];
    if Sha256::digest(modulus)[..] != fuses.public_key_hash {
        return Err(WarmBootError::PublicKeyHash);
    }
    if !pss_verifies(
        modulus,
        &image[WB0_SIGNATURE],
        &image[WB0_SIGNED_START..length_secure as usize],
    ) {
        return Err(WarmBootError::Signature);
    }
    // From here on, fields of the verified image.
    let field = |offset| u32_at(&image, offset);
    let code_length = field(WB0_CODE_LENGTH);
    if code_length == 0 || code_length > length_secure - WB0_HEADER_SIZE as u32 {
        return Err(header_error("code_length", code_length));
    }
    let destination = field(WB0_DESTINATION);
    let end = destination.checked_add(code_length);
    if destination < ROM_STACK_TOP || end.is_none_or(|end| end > IRAM_END) {
        return Err(header_error("destination", destination));
    }
    let entry = field(WB0_ENTRY);
    if entry & 1 != 0 || !(destination..destination + code_length).contains(&entry) {
        return Err(header_error("entry_point", entry));
    }

    iram.fill(0);
    let code = &image[WB0_HEADER_SIZE..WB0_HEADER_SIZE + code_length as usize];
    let start = (destination - IRAM_BASE) as usize;
    iram[start..start + code.len()].copy_from_slice(code);
    let written = destination..destination + code_length;
    let iram_undefined = undefined(std::slice::from_ref(&written), &WARM_REGIONS);
    Ok(WarmBootHandoff {
        entry,
        cpsr: ENTRY_CPSR,
        iram_written: written,
        iram_undefined,
        registers: vec![SELF_REFRESH_EXIT, ROM_LOCKOUT],
        se_context,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nand::boot_images::{BootImagePackage, BootImages, Kind};
    use aes::cipher::BlockDecrypt;

    const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../");
    const NORMAL: &str =
        "fw_dump/0100000000000819/da6ae627428f5fe36e26aafa833515a8/section0_romfs/nx/";
    const SAFE: &str =
        "fw_dump/010000000000081a/61685e9d7698430619001f8e1c3b06f4/section0_romfs/nx/";

    fn package(kind: Kind, directory: &str) -> BootImagePackage {
        let file = |name: &str| {
            std::fs::read(format!("{ROOT}{directory}{name}"))
                .expect("required boot image package extraction in fw_dump")
        };
        BootImagePackage {
            kind,
            bct: file("bct"),
            package1: file("package1"),
            package2: Vec::new(),
        }
    }

    fn images() -> BootImages {
        BootImages {
            normal: package(Kind::Normal, NORMAL),
            safe: package(Kind::Safe, SAFE),
        }
    }

    /// Both boot partitions as the updater lays them out.
    fn boot_partitions(images: &BootImages) -> [Vec<u8>; 2] {
        let mut boot = [0, 1].map(|_| vec![0; BOOT_PARTITION_SIZE as usize]);
        for piece in images.boot_partition_pieces() {
            let start = piece.offset as usize;
            boot[piece.partition][start..start + piece.data.len()].copy_from_slice(piece.data);
        }
        boot
    }

    /// A key from the key file by name, for those `KeySet` keeps no
    /// accessor for.
    fn key(wanted: &str) -> Key {
        let text =
            std::fs::read_to_string(format!("{ROOT}prod_newer.keys")).expect("required key file");
        let value = text
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once('=')?;
                (name.trim() == wanted).then(|| value.trim().to_owned())
            })
            .expect("the key file has the key");
        let mut key = [0; 16];
        for (byte, pair) in key.iter_mut().zip(value.as_bytes().as_chunks::<2>().0) {
            *byte = u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap();
        }
        key
    }

    /// This console's SBK.
    fn secure_boot_key() -> Key {
        key("secure_boot_key")
    }

    fn fuses() -> Fuses {
        Fuses::retail(secure_boot_key(), 0x1234_5678, ChipFuses::default())
    }

    fn power_on(
        boot: &[Vec<u8>; 2],
        fuses: &Fuses,
    ) -> (Result<BootRomHandoff, BootRomError>, Vec<u8>) {
        let mut iram = vec![0; IRAM_SIZE];
        let handoff = cold_boot(&boot[0], &boot[1], fuses, 0, &mut iram);
        (handoff, iram)
    }

    fn iram_word(iram: &[u8], address: u32) -> u32 {
        u32_at(iram, (address - IRAM_BASE) as usize)
    }

    #[test]
    fn the_fused_key_hash_is_that_of_the_firmware_bcts() {
        let images = images();
        for bct in [&images.normal.bct, &images.safe.bct] {
            assert_eq!(
                Sha256::digest(&bct[BCT_MODULUS])[..],
                ERISTA_PUBLIC_KEY_HASH
            );
        }
        let fuses: Vec<_> = fuses_after_boot().collect();
        assert!(fuses.contains(&(FUSE_PUBLIC_KEY0, 0x60B8_BD4F)));
        assert!(fuses.contains(&(FUSE_SKU_INFO, 0x83)));
        assert!(fuses.contains(&(FUSE_PRIVATE_KEY0 + 16, u32::MAX)));
    }

    #[test]
    fn package1ldr_is_loaded_where_the_bct_says_and_entered_at_its_entry() {
        let images = images();
        let normal = &images.normal;
        let fuses = fuses();
        let (handoff, iram) = power_on(&boot_partitions(&images), &fuses);
        let handoff = handoff.unwrap();

        let entry = Bootloader::of(&normal.bct, 0);
        assert_eq!((handoff.bct_block, handoff.bootloader), (0, 0));
        assert_eq!(handoff.entry, entry.entry_point);
        assert_eq!(handoff.entry, 0x4001_0040);
        assert_eq!(handoff.cpsr, 0xD3);
        assert_eq!(entry.load_address, 0x4001_0000);
        assert_eq!(entry.length as usize, normal.package1.len());
        let load = (entry.load_address - IRAM_BASE) as usize;
        assert_eq!(
            iram[load..load + normal.package1.len()],
            normal.package1[..]
        );
        // ldr sp, =0x40008000
        assert_eq!(iram_word(&iram, handoff.entry), 0xE59F_D00C);

        let bct = (BCT_ADDRESS - IRAM_BASE) as usize;
        assert_eq!(iram[bct..bct + BCT_SIZE], normal.bct[..]);
        assert_eq!(iram_word(&iram, IRAM_BASE + 0x4C), BCT_ADDRESS);
        assert_eq!(iram_word(&iram, IRAM_BASE + 0x48), BCT_SIZE as u32);
        assert_eq!(iram_word(&iram, IRAM_BASE + 0x50), 1);
        assert_eq!(iram_word(&iram, IRAM_BASE + 0x0C), 1);
        // What package1ldr's version check reads: the version of the
        // started entry, which it requires to be 0x17.
        assert_eq!(iram_word(&iram, BCT_ADDRESS + 0x2330), 0x17);

        assert_eq!(
            handoff.iram_written,
            [
                0x4000_000C..0x4000_0010,
                0x4000_0048..0x4000_0054,
                0x4000_0100..0x4000_2900,
                0x4001_0000..0x4001_0000 + normal.package1.len() as u32,
            ]
        );
        let ranges: Vec<_> = handoff
            .iram_undefined
            .iter()
            .map(|(range, _)| range.clone())
            .collect();
        assert_eq!(
            ranges,
            [
                0x4000_0000..0x4000_000C,
                0x4000_0010..0x4000_0048,
                0x4000_0054..0x4000_0100,
                0x4000_2900..0x4001_0000,
                0x4001_0000 + normal.package1.len() as u32..IRAM_END,
            ]
        );

        let [sbk, ssk] = &handoff.keyslots;
        assert_eq!((sbk.slot, ssk.slot), (14, 15));
        assert!(sbk.key == fuses.secure_boot_key);
        assert!(ssk.key != sbk.key);
        assert!(!sbk.access.readable && sbk.access.usable && sbk.access.updatable);
        assert_eq!(ssk.access, sbk.access);
    }

    #[test]
    fn the_ssk_depends_on_the_device_key_and_the_chip() {
        let fuses = fuses();
        let ssk = secure_storage_key(&fuses);
        let other_device = Fuses {
            device_key: 0,
            ..fuses.clone()
        };
        let other_chip = Fuses {
            chip: ChipFuses {
                wafer_id: 1,
                ..ChipFuses::default()
            },
            ..fuses.clone()
        };
        assert!(secure_storage_key(&other_device) != ssk);
        assert!(secure_storage_key(&other_chip) != ssk);
    }

    #[test]
    fn keys_stay_out_of_debug_output() {
        let images = images();
        let fuses = fuses();
        let handoff = power_on(&boot_partitions(&images), &fuses).0.unwrap();
        let hex: String = fuses
            .secure_boot_key
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        for text in [format!("{handoff:?}"), format!("{fuses:?}")] {
            assert!(!text.to_lowercase().contains(&hex));
            assert!(!text.contains(&format!("{:?}", fuses.secure_boot_key)));
        }
    }

    #[test]
    fn a_corrupted_bct_signature_is_refused() {
        let images = images();
        let mut boot = boot_partitions(&images);
        for offset in [0, 0x4000, 0x8000, 0xC000] {
            boot[0][offset + BCT_SIGNATURE.start + 7] ^= 1;
        }
        let (handoff, iram) = power_on_retail(&boot);
        let Err(BootRomError::NoBct(rejected)) = handoff else {
            panic!("booted with corrupted BCTs: {handoff:?}");
        };
        assert_eq!(
            rejected[..4],
            [0, 1, 2, 3].map(|block| (block, BctRejection::Signature))
        );
        assert_eq!(rejected[4], (4, BctRejection::BootDataVersion(0)));
        assert_eq!(rejected.len(), BCT_SEARCH_BLOCKS as usize);
        assert!(iram.iter().all(|&byte| byte == 0));
    }

    #[test]
    fn a_corrupted_signed_field_is_refused_and_the_next_copy_taken() {
        let images = images();
        let mut boot = boot_partitions(&images);
        boot[0][0x600] ^= 0x80;
        let handoff = power_on_retail(&boot).0.unwrap();
        // Block 1: the safe mode BCT, whose bootloader is in boot partition 1.
        assert_eq!((handoff.bct_block, handoff.bootloader), (1, 0));
        assert_eq!(
            handoff.entry,
            Bootloader::of(&images.safe.bct, 0).entry_point
        );
    }

    #[test]
    fn a_bct_whose_key_is_not_the_fused_one_is_refused() {
        let images = images();
        let fuses = Fuses {
            public_key_hash: [1; 32],
            ..fuses()
        };
        let Err(BootRomError::NoBct(rejected)) = power_on(&boot_partitions(&images), &fuses).0
        else {
            panic!("booted with another key");
        };
        assert_eq!(rejected[0], (0, BctRejection::PublicKeyHash));
    }

    #[test]
    fn a_corrupted_bootloader_falls_back_to_the_next_entry_then_is_refused() {
        let images = images();
        let mut boot = boot_partitions(&images);
        boot[0][0x10_0000 + 0x1000] ^= 1;
        let (handoff, iram) = power_on_retail(&boot);
        let handoff = handoff.unwrap();
        assert_eq!((handoff.bct_block, handoff.bootloader), (0, 1));
        assert_eq!(iram_word(&iram, IRAM_BASE + 0x50 + 0x18), 1);
        assert_eq!(handoff.iram_undefined[2].0, 0x4000_0050..0x4000_0068);

        boot[0][0x14_0000 + 0x1000] ^= 1;
        let error = power_on_retail(&boot).0.unwrap_err();
        assert_eq!(
            error,
            BootRomError::NoBootloader {
                bct_block: 0,
                rejected: vec![
                    (0, BootloaderRejection::Signature),
                    (1, BootloaderRejection::Signature)
                ],
            }
        );
    }

    #[test]
    fn warm_boot_recovery_and_other_boot_paths_are_refused() {
        let boot = boot_partitions(&images());
        let mut iram = vec![0; IRAM_SIZE];
        let fuses = fuses();
        let run = |fuses: &Fuses, scratch0, iram: &mut [u8]| {
            cold_boot(&boot[0], &boot[1], fuses, scratch0, iram).unwrap_err()
        };
        assert_eq!(run(&fuses, 1, &mut iram), BootRomError::WarmBoot);
        assert_eq!(run(&fuses, 1 << 1, &mut iram), BootRomError::ForcedRecovery);
        let straps = Fuses {
            reserved_sw: 0,
            ..fuses.clone()
        };
        assert!(matches!(
            run(&straps, 0, &mut iram),
            BootRomError::Unmodelled(_)
        ));
        let spi = Fuses {
            reserved_sw: RESERVED_SW_SKIP_STRAPS | 2,
            ..fuses.clone()
        };
        assert_eq!(run(&spi, 0, &mut iram), BootRomError::BootDevice(2));
        let symmetric = Fuses {
            public_key_hash: [0; 32],
            ..fuses.clone()
        };
        assert!(matches!(
            run(&symmetric, 0, &mut iram),
            BootRomError::Unmodelled(_)
        ));
        assert_eq!(run(&fuses, 0, &mut [0; 16]), BootRomError::IramSize(16));
    }

    fn power_on_retail(boot: &[Vec<u8>; 2]) -> (Result<BootRomHandoff, BootRomError>, Vec<u8>) {
        power_on(boot, &fuses())
    }

    /// The warmboot of firmware 23.5.0: the last section of the PK11 blob in
    /// package1 (at 0x6FE0: its size, the IV at +0x10, AES-128-CBC
    /// ciphertext under `package1_key_08` from +0x20), after the 0x20-byte
    /// PK11 header, the NX bootloader and the secure monitor.
    fn warmboot() -> Vec<u8> {
        let package1 = package(Kind::Normal, NORMAL).package1;
        let blob = &package1[0x6FE0..];
        let size = u32_at(blob, 0) as usize;
        let cipher = Aes128::new((&key("package1_key_08")).into());
        let mut chain: [u8; 16] = blob[0x10..0x20].try_into().unwrap();
        let mut plain = Vec::with_capacity(size);
        for block in blob[0x20..0x20 + size].as_chunks::<16>().0 {
            let mut decrypted: aes::Block = (*block).into();
            cipher.decrypt_block(&mut decrypted);
            for (byte, chained) in decrypted.iter_mut().zip(chain) {
                *byte ^= chained;
            }
            plain.extend_from_slice(&decrypted);
            chain = *block;
        }
        assert_eq!(&plain[..4], b"PK11");
        let (warmboot, bootloader, monitor) = (
            u32_at(&plain, 4) as usize,
            u32_at(&plain, 0x10) as usize,
            u32_at(&plain, 0x18) as usize,
        );
        let start = 0x20 + bootloader + monitor;
        plain[start..start + warmboot].to_vec()
    }

    const DRAM_BASE: u32 = 0x8000_0000;
    const WARMBOOT_DRAM: u32 = 0x8000_D000;
    const CONTEXT_DRAM: u32 = 0x8000_F000;
    const SRK_WORDS: [u32; 4] = [0x0123_4567, 0x89AB_CDEF, 0x0F1E_2D3C, 0x4B5A_6978];

    /// A console asleep in SC7 as the monitor and the NX bootloader leave
    /// it: the warmboot and an SE context in DRAM, their addresses, the SRK
    /// and the SDRAM parameters in the PMC.
    struct Asleep {
        pmc: std::collections::BTreeMap<u32, u32>,
        dram: Vec<u8>,
    }

    impl Asleep {
        fn new(warmboot: &[u8]) -> Self {
            let mut pmc = std::collections::BTreeMap::new();
            for (first, last) in SDRAM_SCRATCH {
                for offset in (first..=last).step_by(4) {
                    pmc.insert(offset, 0);
                }
            }
            pmc.insert(PMC_SCRATCH0, 1);
            pmc.insert(PMC_SCRATCH1, WARMBOOT_DRAM);
            pmc.insert(PMC_SCRATCH43, CONTEXT_DRAM);
            for (offset, word) in PMC_SRK.into_iter().zip(SRK_WORDS) {
                pmc.insert(offset, word);
            }
            let mut dram = vec![0; 0x2_0000];
            let at = |address: u32| (address - DRAM_BASE) as usize;
            dram[at(WARMBOOT_DRAM)..at(WARMBOOT_DRAM) + warmboot.len()].copy_from_slice(warmboot);
            let context = sealed_context(&se::srk_from_scratch(SRK_WORDS));
            dram[at(CONTEXT_DRAM)..at(CONTEXT_DRAM) + context.len()].copy_from_slice(&context);
            Self { pmc, dram }
        }

        fn wake(&self, fuses: &Fuses) -> (Result<WarmBootHandoff, WarmBootError>, Vec<u8>) {
            let mut iram = vec![0; IRAM_SIZE];
            let pmc = |offset| self.pmc.get(&offset).copied();
            let dram = |address: u32, length: usize| {
                let start = address.checked_sub(DRAM_BASE)? as usize;
                self.dram.get(start..start + length).map(<[u8]>::to_vec)
            };
            let handoff = warm_boot(fuses, &pmc, &dram, &mut iram);
            (handoff, iram)
        }
    }

    /// An SE context the way the engine saves it: AES-128-CBC under the SRK
    /// with a zero IV, its last block the known pattern 00 01 .. 0F. The
    /// rest (keys, IVs, settings) is zero here.
    fn sealed_context(srk: &[u8; 16]) -> Vec<u8> {
        let mut plain = vec![0; se::CONTEXT_SIZE];
        let pattern = se::CONTEXT_SIZE - 16;
        for (index, byte) in plain[pattern..].iter_mut().enumerate() {
            *byte = index as u8;
        }
        let cipher = Aes128::new(srk.into());
        let mut chain = [0; 16];
        for block in plain.as_chunks_mut::<16>().0 {
            for (byte, chained) in block.iter_mut().zip(chain) {
                *byte ^= chained;
            }
            let mut encrypted: aes::Block = (*block).into();
            cipher.encrypt_block(&mut encrypted);
            block.copy_from_slice(&encrypted);
            chain = encrypted.into();
        }
        plain
    }

    #[test]
    fn the_warmboot_header_is_as_expected_and_its_key_is_the_bcts() {
        let warmboot = warmboot();
        assert_eq!(warmboot.len(), 0x1430);
        let field = |offset| u32_at(&warmboot, offset);
        assert_eq!(field(WB0_LENGTH_INSECURE), 0x1430);
        assert_eq!(field(WB0_LENGTH_SECURE), 0x1430);
        assert_eq!(field(WB0_DESTINATION), 0x4001_0240);
        assert_eq!(field(WB0_ENTRY), 0x4001_0240);
        assert_eq!(field(WB0_CODE_LENGTH), 0x11F0);
        let bct = package(Kind::Normal, NORMAL).bct;
        assert_eq!(warmboot[WB0_MODULUS], bct[BCT_MODULUS]);
    }

    #[test]
    fn a_warm_boot_loads_the_warmboot_where_its_header_says() {
        let warmboot = warmboot();
        let (handoff, iram) = Asleep::new(&warmboot).wake(&fuses());
        let handoff = handoff.unwrap();
        assert_eq!(handoff.entry, 0x4001_0240);
        assert_eq!(handoff.cpsr, 0xD3);
        assert_eq!(iram[0x1_0240..0x1_1430], warmboot[0x240..0x1430]);
        assert_eq!(handoff.iram_written, 0x4001_0240..0x4001_1430);
        let ranges: Vec<_> = handoff
            .iram_undefined
            .iter()
            .map(|(range, _)| range.clone())
            .collect();
        assert_eq!(
            ranges,
            [
                IRAM_BASE..ROM_STACK_TOP,
                ROM_STACK_TOP..0x4001_0240,
                0x4001_1430..IRAM_END
            ]
        );
        assert_eq!(handoff.registers[0], SELF_REFRESH_EXIT);
        assert_eq!(handoff.registers[1], ROM_LOCKOUT);
        // The entry holds the first word of the code: the warmboot's branch
        // to 0x40010444.
        assert_eq!(iram_word(&iram, handoff.entry), u32_at(&warmboot, 0x240));
    }

    #[test]
    fn a_corrupted_warmboot_is_refused_but_not_for_its_unused_cmac() {
        let warmboot = warmboot();
        let fuses = fuses();
        let corrupt = |offset: usize| {
            let mut image = warmboot.clone();
            image[offset] ^= 1;
            Asleep::new(&image).wake(&fuses).0
        };
        assert!(matches!(corrupt(0x1000), Err(WarmBootError::Signature)));
        assert!(matches!(corrupt(0x220), Err(WarmBootError::Signature)));
        assert!(matches!(corrupt(0x130), Err(WarmBootError::Signature)));
        assert!(matches!(corrupt(0x20), Err(WarmBootError::PublicKeyHash)));
        // The CMAC field (0x110..0x120) lies outside the signed part.
        assert_eq!(corrupt(0x115).unwrap().entry, 0x4001_0240);
    }

    #[test]
    fn a_warm_boot_needs_what_the_monitor_and_the_bootloader_left_in_the_pmc() {
        let warmboot = warmboot();
        let fuses = fuses();
        let without = |offset| {
            let mut asleep = Asleep::new(&warmboot);
            asleep.pmc.remove(&offset);
            asleep.wake(&fuses).0.unwrap_err()
        };
        assert!(matches!(
            without(PMC_SCRATCH0),
            WarmBootError::NotWarm(None)
        ));
        assert!(matches!(
            without(0x41C),
            WarmBootError::Unwritten { offset: 0x41C, .. }
        ));
        assert!(matches!(
            without(PMC_SCRATCH43),
            WarmBootError::Unwritten { offset: 0x22C, .. }
        ));
        assert!(matches!(
            without(PMC_SRK[3]),
            WarmBootError::Unwritten { offset: 0x228, .. }
        ));
        assert!(matches!(
            without(PMC_SCRATCH1),
            WarmBootError::Unwritten { offset: 0x054, .. }
        ));

        let mut asleep = Asleep::new(&warmboot);
        asleep.pmc.insert(PMC_SCRATCH0, 2);
        assert!(matches!(
            asleep.wake(&fuses).0,
            Err(WarmBootError::NotWarm(Some(2)))
        ));
        let mut asleep = Asleep::new(&warmboot);
        let last = (CONTEXT_DRAM - DRAM_BASE) as usize + se::CONTEXT_SIZE - 1;
        asleep.dram[last] ^= 1;
        assert!(matches!(
            asleep.wake(&fuses).0,
            Err(WarmBootError::SeContext(_))
        ));
    }
}
