//! The secure monitor's SoC security set-up: what the 23.5.0 Erista monitor
//! does at its cold boot and again on the resume from SC7, high-level
//! emulated through the bus as core 0 (and, for the interrupt controller,
//! its Secure accesses, [`Gic::secure_distributor_write`]). Addresses are
//! the monitor's virtual addresses (TZRAM code at `0x1F0140000` = physical
//! `0x7C012000`), as in [`super::sc7`] and [`super::cold_boot`].
//!
//! # Cold boot (`0x1F01E15D4`)
//!
//! 1. The security engine idle (`0x1F01413A8`), the distributor's set-up
//!    (`distributor`, `0x1F0144084`).
//! 2. `0x1F0146C80`: `SE_TZRAM_SECURITY` = 0 (`0x1F01423DC`), the engine's
//!    per-key setting made secure (`SE_SECURITY` bit 2 cleared,
//!    `0x1F014259C`), what the warm path reads (`PMC_SCRATCH1` =
//!    `0x8000D000`, `SECURE_SCRATCH21` bit 4, `SECURE_SCRATCH32` = `0x339`),
//!    a new secure random key, the key slots' permissions
//!    (`SLOT_LOCKS`, `0x1F01E1368`), and the digest of the key slots the
//!    resume checks (`key_digest`, `0x1F0143DD8`) kept in TZRAM at
//!    [`DIGEST`].
//! 3. The shared reset path (`reset_path`, `0x1F0147010`), then core 0's
//!    part of the interrupt controller (`core_interrupts`, `0x1F0146F40`).
//! 4. At the bootloader's state 2, two more peripherals secure
//!    (`apb_secure`, `0x1F0143FA4`).
//! 5. At its end, `final_setup` (`0x1F0147430`).
//!
//! Not done of these (no device here models what they set, or nothing
//! reads it back): the system counter's check and start, `PMC+0x0A0`
//! bits 19:18 and `SCRATCH190` bit 0, the PMC's general locks
//! (`0x1F0143568(1)`), the key slots' contents (the monitor's keys are
//! [`super::crypto`]'s here).
//!
//! # Resume (core 0 released at [`super::sc7::RESUME_ENTRY`])
//!
//! When `MC_SECURITY_CFG3` reads 0, the SoC lost power:
//!
//! 1. The reset vector's part (`0x7C012C28`): the TSEC's host1x syncpoint
//!    (160, `0x50003300`) cleared, AHB arbitration (`AHB_MASTER_SWID` all
//!    ones, `_SWID_1` 0, `AHB_ARBITRATION_DISABLE` `0x40060`,
//!    `_PRIORITY_CTRL` `0xE0000001`, `AHB_GIZMO_TZRAM` `0x80`) and the
//!    `MSELECT_CONFIG` read-modify-write (bits 25:24 cleared, 29:27 set)
//!    (`reset_vector`). Not done: bit 0 of `0x500038F8` cleared
//!    beforehand (Atmosphère: the syncpoint made secure), a host1x
//!    register past what [`crate::soc::host1x`] models, whose value after
//!    reset is not public.
//! 2. `0x1F014414C`: the engine idle, the distributor's set-up, a new
//!    secure random key, `reset_path`, `apb_secure`, `final_setup`;
//!    then (`0x1F0146F40`) core 0's part of the interrupt controller.
//!
//! The development units' greeting on the UART and the charger's
//! high-impedance mode (refused before the suspend, [`super`]) are not
//! done.
//!
//! # The shared reset path (`0x1F0147010`)
//!
//! Done: the engine's soft security setting cleared (`0x1F0142EF8(1)`);
//! the engine's settings checked against the monitor's table
//! (`0x1F01423EC` with `0x1F01480A8`) and the key slots' digest against the
//! cold boot's (`0x1F0143DD8`), each a panic if they differ; key slots 0 to
//! 9 and both RSA slots cleared; the random number generator reseeded
//! (`0x1F01427A8`); random keys into slots 8 and 2 and a new secure random
//! key; the PMC's secure scratch scrubbed ([`SCRUBBED`], `0x1F0143D78`)
//! and locked; the memory controller's carveouts (`CARVEOUTS`, and the
//! kernel's two from what the kernel configured, `0x1F0143C90`) and
//! `MC_SECURITY_CFG3` = 3; APB slave security (`0x70000C00..0x70000C08`);
//! `SECURE_SCRATCH34` = [`super::sc7::RESUME_ENTRY`] and its lock;
//! interrupts 90 and 77 into Group 0 for core 3.
//!
//! Not done (no device here models them, or nothing reads them back): the
//! fuse block's visibility and its key-disable writes (`MISC_CLK_ENB` bit
//! 28, `FUSE_PRIVATEKEYDISABLE`, `FUSE_DISABLEREGPROGRAM`), the timers'
//! secure configuration, the flow controller's BPMP cluster control and
//! debug qualifier, `DPD_ENABLE` = 0, the SMMU's set-up (translation
//! enables, TLB and PTC configuration, address space 0), the CPU reset
//! vector in the exception vectors, the secure boot registers (the
//! warmboot firmware wrote them on the resume), and the monitor's own
//! interrupt handlers.

use std::sync::atomic::AtomicBool;

use super::sc7::{self, Port, Sc7Fail};
use crate::{
    gic::Gic,
    soc::{mc, pmc, se, tzram},
};

type Result<T, E = Sc7Fail> = std::result::Result<T, E>;

const PMC: u64 = pmc::BASE;
const SCRATCH1: u64 = PMC + 0x054;
const SEC_DISABLE: u64 = PMC + 0x004;
const SEC_DISABLE3: u64 = PMC + 0x2D8;
const SECURE_SCRATCH21: u64 = PMC + 0x334;
const SECURE_SCRATCH32: u64 = PMC + 0x360;
const SECURE_SCRATCH34: u64 = PMC + 0x368;

/// `SECURE_SCRATCH21` bit 4, which the warmboot firmware ORs into
/// `FUSE_PRIVATEKEYDISABLE`.
const PRIVATE_KEY_DISABLE: u32 = 1 << 4;
/// `PMC_SCRATCH1`: where the boot ROM finds the warmboot firmware.
const WARMBOOT_VECTOR: u32 = 0x8000_D000;

/// The PMC secure scratch registers the reset path fills with random
/// words, in its order, and the `SEC_DISABLE` bits it sets after each
/// group (`0x1F0143568(0x100)`): `SECURE_SCRATCH4..7` (where the engine's
/// context save left the secure random key for the boot ROM), then
/// `112..115` and `24..27` (the suspend's TZRAM MAC and key seed).
pub const SCRUBBED: [&[u64]; 2] = [
    &[0x0C0, 0x0C4, 0x224, 0x228],
    &[0xB18, 0xB1C, 0xB20, 0xB24, 0x340, 0x344, 0x348, 0x34C],
];
const SCRUB_LOCK: u32 = 0xFF000;
/// `SEC_DISABLE3` bits 23:20, the lock of `SECURE_SCRATCH34`
/// (`0x1F0143568(4)`).
const RESUME_VECTOR_LOCK: u32 = 0xF0_0000;

const SE: u64 = se::BASE;
const SE_SECURITY: u64 = SE;
const SE_TZRAM_SECURITY: u64 = SE + 0x004;
const SE_CONFIG: u64 = SE + 0x014;
const SE_CRYPTO_CONFIG: u64 = SE + 0x304;
const SE_CRYPTO_LAST_BLOCK: u64 = SE + 0x318;
const SE_CRYPTO_SECURITY_PERKEY: u64 = SE + 0x280;
const SE_CRYPTO_KEYTABLE_ACCESS: u64 = SE + 0x284;
const SE_RNG_CONFIG: u64 = SE + 0x340;
const SE_RNG_SRC_CONFIG: u64 = SE + 0x344;
const SE_RNG_RESEED_INTERVAL: u64 = SE + 0x348;
const SE_RSA_SECURITY_PERKEY: u64 = SE + 0x40C;
const SE_RSA_KEYTABLE_ACCESS: u64 = SE + 0x410;
const SE_RSA_KEYTABLE_ADDR: u64 = SE + 0x420;
const SE_RSA_KEYTABLE_DATA: u64 = SE + 0x424;
const SE_STATUS: u64 = SE + 0x800;
const OP_START: u32 = 1;
/// `SE_SECURITY` bit 2 (the per-key setting) and bit 16 (the soft one).
const SE_PER_KEY_SETTING: u32 = 1 << 2;
const SE_SOFT_SETTING: u32 = 1 << 16;

/// Where TZRAM holds, at the monitor's virtual addresses `0x1F01480A8`
/// and `0x1F01499B8`, the engine's settings table the reset path checks
/// and the digest of the key slots the cold boot keeps. Both lie in the
/// TZRAM the suspend saves and the warmboot firmware restores.
const SETTINGS_TABLE: u64 = 0x7C01_A0A8;
pub const DIGEST: u64 = 0x7C01_B9B8;
const _: () = assert!(sc7::SAVED <= SETTINGS_TABLE && DIGEST + 16 <= tzram::BASE + tzram::SIZE);

/// The key slots the digest goes through, in its order, twice over; the
/// slot each one's key is decrypted into first.
const DIGEST_SLOTS: [u32; 5] = [10, 11, 13, 12, 15];
const DIGEST_SCRATCH_SLOT: u32 = 8;
/// `SE_CONFIG` bits 31:16 of an AES-128 encryption (`0x1F0142324`).
const MODES_AES128: u32 = 0;

/// Buffers on the monitor's stack, below both the cold boot's (`SP_EL0`
/// `0x1F01E0800`, physical `0x7C010800`, under its resident image) and
/// the resume's (`SP_EL3` `0x1F01F6740`, physical `0x7C010740`): the
/// engine's linked lists at [`BOOT_STACK`] (the port's `stack`), a 16-byte
/// work block and a zero block.
pub const BOOT_STACK: u64 = 0x7C01_0600;
const WORK_BLOCK: u64 = BOOT_STACK + 0x100;
const ZERO_BLOCK: u64 = BOOT_STACK + 0x110;

/// The key slots' permissions the cold boot leaves (`0x1F01E1368`, called
/// from `0x1F0146C80`), in its order, as `(slots, flags)` in the encoding
/// of its helper `0x1F0141544`: bits 6:0 the permissions taken away (key
/// read, key update, OIV read, OIV update, UIV read, UIV update, key use;
/// Atmosphère's `AesKeySlotLockFlags`), bit 8 the slot's per-key bit
/// cleared. Every slot loses its read permissions, slots 8 to 15 key use
/// and their per-key bit, slots 6, 7 and 10 to 15 everything. The key
/// operations between them, by which the monitor loads its own keys into
/// slots 10 to 14, are not done: the monitor's keys are
/// [`super::crypto`]'s here, and the slots' contents end read-protected.
const SLOT_LOCKS: [(std::ops::RangeInclusive<u32>, u32); 9] = [
    (0..=15, 0x015),
    (8..=15, 0x140),
    (6..=7, 0x1FF),
    (15..=15, 0x1FF),
    (13..=13, 0x17F),
    (12..=12, 0x1FF),
    (11..=11, 0x17F),
    (14..=14, 0x1FF),
    (10..=10, 0x1FF),
];
/// Both RSA slots' (`0x1F01415A8` with `0x141`): only key update left,
/// per-key bits cleared.
const RSA_SLOT_LOCK: u32 = 0x141;
const LOCK_PER_KEY: u32 = 1 << 8;

/// `MC_SECURITY_CFG3`: 3 once the monitor set the SoC up, 0 after reset.
const SECURITY_CFG3_SET_UP: u32 = 3;

/// The memory controller's carveout registers the reset path writes
/// (`0x1F0147010`), in its order, as `(offset, value)`: `0x984`/`0x988`,
/// the video protect region (`0x648`..`0x650`: empty, locked), the
/// security carveout (`0x670`..`0x678`: empty, locked), `0x9A0`..`0x9AC`
/// (empty, locked), the TrustZone range (`MC_SECURITY_CFG0/1`, `0x070`,
/// `0x074`: empty), `MC_SECURITY_CFG3` = 3, then general carveouts 1
/// (`0xC08`: no range, its configuration last) and 3 (`0xCA8`). Names
/// after Linux `tegra210-mc` and Atmosphère; the values are the monitor's.
const CARVEOUTS: [(u64, u32); 43] = [
    (0x984, 1),
    (0x988, 0),
    (0x648, 0),
    (0x64C, 0),
    (0x650, 1),
    (0x670, 0),
    (0x674, 0),
    (0x678, 1),
    (0x9A0, 0),
    (0x9A4, 0),
    (0x9A8, 0),
    (0x9AC, 1),
    (0x070, 0),
    (0x074, 0),
    (0x9BC, SECURITY_CFG3_SET_UP),
    (0xC0C, 0),
    (0xC10, 0),
    (0xC14, 0),
    (0xC18, 0),
    (0xC1C, 0),
    (0xC20, 0),
    (0xC24, 0),
    (0xC28, 0),
    (0xC2C, 0),
    (0xC30, 0),
    (0xC34, 0),
    (0xC38, 0),
    (0xC3C, 0),
    (0xC08, 0x0400_0006),
    (0xCAC, 0),
    (0xCB0, 0),
    (0xCB4, 0),
    (0xCB8, 0),
    (0xCBC, 0),
    (0xCC0, 0x0300_0000),
    (0xCC4, 0),
    (0xCC8, 0x300),
    (0xCCC, 0),
    (0xCD0, 0),
    (0xCD4, 0),
    (0xCD8, 0),
    (0xCDC, 0),
    (0xCA8, 0x0440_1E7E),
];

/// The kernel's two carveouts, general carveouts 4 and 5 (`0x1F0143C90`),
/// by offset of their configuration register: the base (`+0x04`), the
/// size in 128 KiB units (`+0x0C`), the client access masks (`+0x10`..,
/// the clients [`mc`]'s kernel carveouts let through), the forced
/// internal accesses (carveout 4 all five, `+0x24`..; carveout 5 the last
/// four), the configuration last.
const KERNEL_CARVEOUTS: [(u64, &[(u64, u32)]); 2] = [
    (
        0xCF8,
        &[
            (0x24, 0x8000),
            (0x28, 0x4_0000),
            (0x2C, 0),
            (0x30, 0),
            (0x34, 0),
        ],
    ),
    (0xD48, &[(0x28, 0), (0x2C, 0), (0x30, 0), (0x34, 0)]),
];
const KERNEL_CARVEOUT_ACCESS: [u32; 5] = [0x70E3_407F, 0x1A62_0880, 0x3C00, 0xCC08_07BB, 3];
const KERNEL_CARVEOUT_CONFIG: u32 = 0x4CB;

/// General carveout 2 (`0xC58`), which [`final_setup`] writes
/// (`0x1F0147430`): `0x8002_0000`, 256 KiB.
const FINAL_CARVEOUT: [(u64, u32); 14] = [
    (0xC5C, 0x8002_0000),
    (0xC60, 0),
    (0xC64, 2),
    (0xC68, 0),
    (0xC6C, 0),
    (0xC70, 0x0310_0000),
    (0xC74, 0),
    (0xC78, 0x300),
    (0xC7C, 0),
    (0xC80, 0),
    (0xC84, 0),
    (0xC88, 0),
    (0xC8C, 0),
    (0xC58, 0x0440_167E),
];

/// `APB_MISC_SECURE_REGS_APB_SLAVE_SECURITY_ENABLE_REG0..2`
/// (`0x70000C00`): the reset path's values, register 1's by hardware type
/// (`0x1F01439D0`: type 1 gets `0xA3700000`, every other `0x83700000`),
/// and the bits [`apb_secure`] adds.
const APB_SLAVE_SECURITY: [u64; 3] = [0x7000_0C00, 0x7000_0C04, 0x7000_0C08];
const APB_SLAVE_SECURITY_0: u32 = 0x0050_4244;
const APB_SLAVE_SECURITY_1: [u32; 2] = [0x8370_0000, 0xA370_0000];
const APB_SLAVE_SECURITY_2: u32 = 0x304;
const APB_SECURE_LATER: [u32; 2] = [0x2000, 0x510];
/// `FUSE_RESERVED_ODM4`, from which the monitor reads the hardware type.
const FUSE_RESERVED_ODM4: u64 = 0x7000_F9D8;

/// The reset vector's registers (`0x7C012C28`).
const HOST1X_TSEC_SYNCPT: u64 = 0x5000_3300;
const AHB_ARBITRATION_DISABLE: u64 = 0x6000_C004;
const AHB_ARBITRATION_PRIORITY_CTRL: u64 = 0x6000_C008;
const AHB_MASTER_SWID: u64 = 0x6000_C018;
const AHB_MASTER_SWID_1: u64 = 0x6000_C038;
const AHB_GIZMO_TZRAM: u64 = 0x6000_C054;
const MSELECT_CONFIG: u64 = crate::soc::mselect::BASE;
/// What [`final_setup`] adds to `AHB_ARBITRATION_DISABLE`.
const AHB_ARBITRATION_BPMP: u32 = 1 << 1;

/// The interrupt controller, at offsets from its distributor and CPU
/// interface.
const GICD_CTLR: u64 = 0x000;
const GICD_IGROUPR: u64 = 0x080;
const GICD_ISENABLER: u64 = 0x100;
const GICD_IPRIORITYR: u64 = 0x400;
const GICD_ITARGETSR: u64 = 0x800;
const GICD_ICFGR: u64 = 0xC00;
const GICC_CTLR: u64 = 0x00;
const GICC_PMR: u64 = 0x04;
const GICC_BPR: u64 = 0x08;
/// The shared interrupts the distributor's set-up covers: `GICD_IGROUPR1`
/// to `6`, priorities of 32 to 223.
const SHARED_GROUPS: std::ops::Range<u64> = 1..7;
const SHARED_PRIORITIES: std::ops::Range<u64> = 32..224;
/// The priority (Secure view) the set-up gives every interrupt.
const PRIORITY_DEFAULT: u32 = 0x8080_8080;
/// The monitor's interrupts: the security engine's (90) and the activity
/// monitor's (77), at priority 0 for core 3, and core-local 28.
const MONITOR_INTERRUPTS: [u32; 2] = [90, 77];
const MONITOR_TARGET: u32 = 1 << 3;
const CORE_INTERRUPT: u32 = 28;
/// The CPU interface's Secure set-up (`0x1F01431C8`): `EnableGrp0`,
/// `FIQEn`, `CBPR` and the bypass disables; mask `0x80`; binary point 7.
const GICC_CTLR_SECURE: u32 = 0x1D9;
const GICC_PMR_SECURE: u32 = 0x80;
const GICC_BPR_SECURE: u32 = 7;

/// The interrupt controller's physical addresses, for what a refusal
/// names.
const GICD_BASE: u64 = 0x5004_1000;
const GICC_BASE: u64 = 0x5004_2000;

/// A Secure access the controller does not model stops the set-up.
fn gicd_write(gic: &Gic, core: usize, offset: u64, value: u32) -> Result<()> {
    gic.secure_distributor_write(core, offset, value)
        .ok_or(Sc7Fail::Register {
            address: GICD_BASE + offset,
            is_write: true,
        })
}

fn gicd_modify(gic: &Gic, core: usize, offset: u64, mask: u32, value: u32) -> Result<()> {
    let old = gic
        .secure_distributor_read(core, offset)
        .ok_or(Sc7Fail::Register {
            address: GICD_BASE + offset,
            is_write: false,
        })?;
    gicd_write(gic, core, offset, old & !mask | value & mask)
}

fn gicc_write(gic: &Gic, core: usize, offset: u64, value: u32) -> Result<()> {
    gic.secure_cpu_interface_write(core, offset, value)
        .ok_or(Sc7Fail::Register {
            address: GICC_BASE + offset,
            is_write: true,
        })
}

/// The distributor's set-up (`0x1F01430D8`, and `0x1F0144084` at the cold
/// boot): shared interrupts 32 to 223 into Group 1 at priority `0x80`,
/// Group 0 forwarding on (Group 1's is the kernel's to turn on). The
/// monitor writes the priorities a byte at a time; here a word holds four.
pub(super) fn distributor(gic: &Gic) -> Result<()> {
    for register in SHARED_GROUPS {
        gicd_write(gic, 0, GICD_IGROUPR + register * 4, u32::MAX)?;
    }
    for intid in SHARED_PRIORITIES.step_by(4) {
        gicd_write(gic, 0, GICD_IPRIORITYR + intid, PRIORITY_DEFAULT)?;
    }
    gicd_write(gic, 0, GICD_CTLR, 1)
}

/// One of the monitor's interrupts, as its helpers set it
/// (`0x1F0143040` priority, `0x1F014311C` group, `0x1F0143238` enable,
/// `0x1F014308C` target, `0x1F0143164` level-sensitive): Group 0,
/// priority 0, enabled; a shared one to `target`.
fn secure_interrupt(gic: &Gic, core: usize, intid: u32, target: Option<u32>) -> Result<()> {
    let intid = u64::from(intid);
    let byte = |base: u64| (base + (intid & !3), 0xFF << (8 * (intid & 3)));
    let (priority, mask) = byte(GICD_IPRIORITYR);
    gicd_modify(gic, core, priority, mask, 0)?;
    gicd_modify(
        gic,
        core,
        GICD_IGROUPR + intid / 32 * 4,
        1 << (intid % 32),
        0,
    )?;
    gicd_write(
        gic,
        core,
        GICD_ISENABLER + intid / 32 * 4,
        1 << (intid % 32),
    )?;
    if let Some(target) = target {
        let (targets, mask) = byte(GICD_ITARGETSR);
        gicd_modify(gic, core, targets, mask, target << (8 * (intid & 3)))?;
        let config = GICD_ICFGR + intid / 16 * 4;
        gicd_modify(gic, core, config, 3 << (2 * (intid % 16)), 0)?;
    }
    Ok(())
}

/// `core`'s part of the interrupt controller, which the monitor sets up on
/// that core whenever it starts it (`0x1F0146F40`: `0x1F01431C8`, then
/// interrupt 28): its banked interrupts 0 to 31 into Group 1 at priority
/// `0x80`, its CPU interface's Secure settings, and interrupt 28 into
/// Group 0, enabled.
pub(super) fn core_interrupts(gic: &Gic, core: usize) -> Result<()> {
    gicd_write(gic, core, GICD_IGROUPR, u32::MAX)?;
    for intid in (0..32).step_by(4) {
        gicd_write(gic, core, GICD_IPRIORITYR + intid, PRIORITY_DEFAULT)?;
    }
    gicc_write(gic, core, GICC_CTLR, GICC_CTLR_SECURE)?;
    gicc_write(gic, core, GICC_PMR, GICC_PMR_SECURE)?;
    gicc_write(gic, core, GICC_BPR, GICC_BPR_SECURE)?;
    secure_interrupt(gic, core, CORE_INTERRUPT, None)
}

/// Panics as the monitor does unless the engine is idle (`0x1F01413A8`).
fn engine_idle(port: &Port<'_>) -> Result<()> {
    if port.read(SE_STATUS)? & 3 != 0 {
        return Err(Sc7Fail::Panic("the security engine is busy"));
    }
    Ok(())
}

/// The digest of the key slots (`0x1F0143DD8`): from a zero block, for
/// each of [`DIGEST_SLOTS`] twice over, the zero block decrypted with that
/// slot's key into slot 8, then the digest encrypted with slot 8; slot 8
/// cleared at the end.
fn key_digest(port: &Port<'_>) -> Result<[u8; 16]> {
    port.write_bytes(WORK_BLOCK, &[0; 16])?;
    port.write_bytes(ZERO_BLOCK, &[0; 16])?;
    for slot in DIGEST_SLOTS.into_iter().chain(DIGEST_SLOTS) {
        port.decrypt_into_slot(DIGEST_SCRATCH_SLOT, slot, ZERO_BLOCK)?;
        port.encrypt_block(DIGEST_SCRATCH_SLOT, MODES_AES128, WORK_BLOCK)?;
    }
    let mut digest = [0; 16];
    port.read_bytes(WORK_BLOCK, &mut digest)?;
    port.clear_slot(DIGEST_SCRATCH_SLOT)?;
    Ok(digest)
}

/// The engine's settings as the monitor's table (`0x1F01480A8`, read from
/// TZRAM) gives them, compared as its `0x1F01423EC` does: the low half of
/// `SE_SECURITY`, `SE_TZRAM_SECURITY` and `SE_CRYPTO_SECURITY_PERKEY`, of
/// each slot's access register, of `SE_RSA_SECURITY_PERKEY` and of both
/// RSA access registers.
fn settings_match(port: &Port<'_>) -> Result<bool> {
    let mut table = [0; 24];
    port.read_bytes(SETTINGS_TABLE, &mut table)?;
    let mut expected = vec![
        (SE_SECURITY, u32::from(table[0])),
        (SE_TZRAM_SECURITY, u32::from(table[1])),
        (
            SE_CRYPTO_SECURITY_PERKEY,
            u32::from(u16::from_le_bytes([table[2], table[3]])),
        ),
    ];
    for slot in 0..16 {
        let access = SE_CRYPTO_KEYTABLE_ACCESS + slot * 4;
        expected.push((access, u32::from(table[4 + slot as usize])));
    }
    expected.push((SE_RSA_SECURITY_PERKEY, u32::from(table[20])));
    expected.push((SE_RSA_KEYTABLE_ACCESS, u32::from(table[21])));
    expected.push((SE_RSA_KEYTABLE_ACCESS + 4, u32::from(table[22])));
    for (register, value) in expected {
        if port.read(register)? & 0xFFFF != value {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Both RSA slots cleared (`0x1F01419A0`): each modulus word, then each
/// exponent word, through `SE_RSA_KEYTABLE_ADDR`/`_DATA` (the monitor
/// stores the address and the zero data with one 64-bit write).
fn clear_rsa_slots(port: &Port<'_>) -> Result<()> {
    for slot in 0..2 {
        for modulus in [0x40, 0] {
            for word in 0..64 {
                port.write(SE_RSA_KEYTABLE_ADDR, slot << 7 | modulus | word)?;
                port.write(SE_RSA_KEYTABLE_DATA, 0)?;
            }
        }
    }
    Ok(())
}

/// The random number generator reseeded (`0x1F01427A8`): source and
/// reseed interval set, one block drawn in force-reseed mode into the
/// work block.
fn reseed(port: &Port<'_>) -> Result<()> {
    port.write(SE_RNG_SRC_CONFIG, 3)?;
    port.write(SE_RNG_RESEED_INTERVAL, 70_001)?;
    port.write(SE_CONFIG, 0x2000)?;
    port.write(SE_CRYPTO_CONFIG, 0x108)?;
    port.write(SE_RNG_CONFIG, 5)?;
    port.write(SE_CRYPTO_LAST_BLOCK, 0)?;
    port.engine_op(OP_START, (WORK_BLOCK, 16), (0, 0))
}

/// A random word into the PMC register at `offset`, read back and
/// compared, a panic otherwise (`0x1F0143D78`; the word from one block
/// the generator draws through the work block, `0x1F0142A4C`).
fn scrub(port: &Port<'_>, offset: u64) -> Result<()> {
    port.write(SE_CONFIG, 0x2000)?;
    port.write(SE_CRYPTO_CONFIG, 0x108)?;
    port.write(SE_RNG_CONFIG, 4)?;
    port.write(SE_CRYPTO_LAST_BLOCK, 0)?;
    port.engine_op(OP_START, (WORK_BLOCK, 16), (WORK_BLOCK, 16))?;
    let word = port.read(WORK_BLOCK)?;
    port.write(PMC + offset, word)?;
    if port.read(PMC + offset)? != word {
        return Err(Sc7Fail::Panic(
            "a scrubbed secure scratch register does not hold its word",
        ));
    }
    Ok(())
}

/// The shared reset path (`0x1F0147010`): see the module documentation.
/// `kernel_carveouts` are the kernel's two carveouts as `(base, size)`.
pub(super) fn reset_path(
    port: &Port<'_>,
    gic: &Gic,
    kernel_carveouts: [(u64, u64); 2],
) -> Result<()> {
    port.modify(SE_SECURITY, SE_SOFT_SETTING, 0)?;
    if !settings_match(port)? {
        return Err(Sc7Fail::Panic(
            "the security engine's settings differ from the monitor's table",
        ));
    }
    let mut stored = [0; 16];
    port.read_bytes(DIGEST, &mut stored)?;
    if key_digest(port)? != stored {
        return Err(Sc7Fail::Panic(
            "the key slots' digest differs from the cold boot's",
        ));
    }
    for slot in 0..10 {
        port.clear_slot(slot)?;
    }
    clear_rsa_slots(port)?;
    reseed(port)?;
    port.random_key(8)?;
    port.new_srk()?;
    port.random_key(2)?;
    for group in SCRUBBED {
        for &offset in group {
            scrub(port, offset)?;
        }
        port.modify(SEC_DISABLE, 0, SCRUB_LOCK)?;
    }

    for (offset, value) in CARVEOUTS {
        port.write(mc::BASE + offset, value)?;
    }
    for ((first, forced), (base, size)) in KERNEL_CARVEOUTS.into_iter().zip(kernel_carveouts) {
        let register = |offset: u64| mc::BASE + first + offset;
        port.write(register(0x04), base as u32)?;
        port.write(register(0x0C), (size >> 17) as u32)?;
        for (index, access) in KERNEL_CARVEOUT_ACCESS.into_iter().enumerate() {
            port.write(register(0x10 + index as u64 * 4), access)?;
        }
        for &(offset, value) in forced {
            port.write(register(offset), value)?;
        }
        port.write(register(0), KERNEL_CARVEOUT_CONFIG)?;
    }

    let hardware_type = {
        let odm4 = port.read(FUSE_RESERVED_ODM4)?;
        odm4 >> 7 & 2 | u32::from(odm4 & 4 != 0) | odm4 >> 14 & 0x3C
    };
    port.write(APB_SLAVE_SECURITY[0], APB_SLAVE_SECURITY_0)?;
    port.write(
        APB_SLAVE_SECURITY[1],
        APB_SLAVE_SECURITY_1[usize::from(hardware_type == 1)],
    )?;
    port.write(APB_SLAVE_SECURITY[2], APB_SLAVE_SECURITY_2)?;

    port.write(SECURE_SCRATCH34, sc7::RESUME_ENTRY as u32)?;
    port.modify(SEC_DISABLE3, 0, RESUME_VECTOR_LOCK)?;
    for intid in MONITOR_INTERRUPTS {
        secure_interrupt(gic, 0, intid, Some(MONITOR_TARGET))?;
    }
    Ok(())
}

/// Two more peripherals secure (`0x1F0143FA4`; on the resume inline in
/// `0x1F014414C`): `APB_SLAVE_SECURITY_ENABLE_REG0` bit 13,
/// `_REG1` bits 4, 8 and 10.
pub(super) fn apb_secure(port: &Port<'_>) -> Result<()> {
    for (register, bits) in APB_SLAVE_SECURITY.into_iter().zip(APB_SECURE_LATER) {
        port.modify(register, 0, bits)?;
    }
    Ok(())
}

/// What the cold boot does that the warm path relies on, and the reset
/// path (`0x1F0144084`, `0x1F0146C80`, `0x1F0147010`, then core 0's part
/// of `0x1F0146F40`): see the module documentation.
pub(super) fn cold_boot(
    port: &Port<'_>,
    gic: &Gic,
    kernel_carveouts: [(u64, u64); 2],
) -> Result<()> {
    engine_idle(port)?;
    distributor(gic)?;
    port.write(SE_TZRAM_SECURITY, 0)?;
    port.modify(SE_SECURITY, SE_PER_KEY_SETTING, 0)?;
    port.write(SCRATCH1, WARMBOOT_VECTOR)?;
    port.modify(SECURE_SCRATCH21, 0, PRIVATE_KEY_DISABLE)?;
    port.write(SECURE_SCRATCH32, sc7::WARMBOOT_MAGIC)?;
    port.new_srk()?;
    slot_permissions(port)?;
    let digest = key_digest(port)?;
    port.write_bytes(DIGEST, &digest)?;
    reset_path(port, gic, kernel_carveouts)?;
    core_interrupts(gic, 0)
}

fn slot_permissions(port: &Port<'_>) -> Result<()> {
    let mut rsa_done = false;
    for (slots, flags) in SLOT_LOCKS {
        for slot in slots {
            if flags & !LOCK_PER_KEY != 0 {
                let access = SE_CRYPTO_KEYTABLE_ACCESS + u64::from(slot) * 4;
                let old = port.read(access)?;
                port.write(access, (old & !0x7F | flags & 0x7F) ^ 0x7F)?;
            }
            if flags & LOCK_PER_KEY != 0 {
                port.modify(SE_CRYPTO_SECURITY_PERKEY, 1 << slot, 0)?;
            }
        }
        // The RSA slots come after the third step.
        if !rsa_done && flags == 0x1FF {
            for slot in 0..2 {
                let access = (RSA_SLOT_LOCK >> 4 & 4 | RSA_SLOT_LOCK & 3) ^ 7;
                port.write(SE_RSA_KEYTABLE_ACCESS + slot * 4, access)?;
                port.modify(SE_RSA_SECURITY_PERKEY, 1 << slot, 0)?;
            }
            rsa_done = true;
        }
    }
    Ok(())
}

/// The reset vector's part of the resume (`0x7C012C28`, `0x7C012C90`):
/// see the module documentation.
fn reset_vector(port: &Port<'_>) -> Result<()> {
    port.write(HOST1X_TSEC_SYNCPT, 0)?;
    port.write(AHB_MASTER_SWID, u32::MAX)?;
    port.write(AHB_MASTER_SWID_1, 0)?;
    port.modify(MSELECT_CONFIG, 0x0300_0000, 0x3800_0000)?;
    port.write(AHB_ARBITRATION_DISABLE, 0x4_0060)?;
    port.write(AHB_ARBITRATION_PRIORITY_CTRL, 0xE000_0001)?;
    port.write(AHB_GIZMO_TZRAM, 0x80)
}

/// The resume's set-up before [`final_setup`], if the SoC lost power
/// (`MC_SECURITY_CFG3` reads 0); `false` if it did not, when the monitor
/// skips it and [`final_setup`].
pub(super) fn resume(
    port: &Port<'_>,
    gic: &Gic,
    kernel_carveouts: [(u64, u64); 2],
) -> Result<bool> {
    if port.read(sc7::MC_SECURITY_CFG3)? != 0 {
        return Ok(false);
    }
    reset_vector(port)?;
    engine_idle(port)?;
    distributor(gic)?;
    port.new_srk()?;
    reset_path(port, gic, kernel_carveouts)?;
    apb_secure(port)?;
    Ok(true)
}

/// How both boots end (`0x1F0147430`): general carveout 2, the BPMP kept
/// off the AHB arbiter, then once the BPMP halted itself the activity
/// monitor clocked (`0x1F0142FD4`; the warmboot firmware turned its clock
/// off with every other one it does not keep, `warmboot-research` §7 step
/// 6; the kernel's next sleep needs it) and the BPMP parked. Not done (no
/// device here models them, or nothing reads them back): the AHB redirect
/// aperture closed (`MC_IRAM_BOM`/`_TOM`), `MC+0x964` bit 0, `CAR+0x3A4`
/// bit 19, the PMC secure scratch copies of the memory controller's
/// settings and their locks, the BPMP's exception vectors and SMMU address
/// space walled off, and the activity monitor's watch on the BPMP.
pub(super) fn final_setup(port: &Port<'_>, stop: &AtomicBool) -> Result<()> {
    for (offset, value) in FINAL_CARVEOUT {
        port.write(mc::BASE + offset, value)?;
    }
    port.modify(AHB_ARBITRATION_DISABLE, 0, AHB_ARBITRATION_BPMP)?;
    port.wait_for_bpmp_halt(stop)?;
    super::cold_boot::actmon_clock(port)?;
    port.park_halted_bpmp()
}
