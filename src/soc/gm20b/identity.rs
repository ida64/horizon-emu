//! What this particular GPU is: identification, configuration and
//! floor-sweeping registers, the engine table, and the few sizes only the
//! chip or its firmware could tell. The driver reads them to size loops and
//! allocations and reports some to its clients; it checks only the chip ID.
//!
//! Sources, with the confidence each deserves:
//!
//! * The chip ID is the value a Jetson TX1 (same GM20B) prints at boot
//!   (`nouveau 57000000.gpu: NVIDIA GM20B (12b000a1)`, kernel log on
//!   lore.freedesktop.org, nouveau mailing list, 2024-10-10).
//! * Counts follow the GPU characteristics a Switch reports to its
//!   applications (switchbrew "NV services", `GET_CHARACTERISTICS`: 1 GPC,
//!   2 TPCs per GPC, 1 FBP, 2 LTCs per FBP, 1 slice per LTC, SM version
//!   0x503, 128 warps, 256 KiB L2), put back into the registers nvgpu
//!   (L4T r32.7.6, `gr_gk20a.c`, `gr_gm20b.c`, `ltc_gm20b.c`) derives them
//!   from, with field layouts from nvgpu's `hw_*_gm20b.h`. Derived, not
//!   read from a chip.
//! * Everything marked "assumed" has no public source; it is chosen so the
//!   driver accepts it, and says why.

/// `mc_boot_0`: architecture 0x12 (bits 28:24), implementation 0xB (bits
/// 23:20), revision A1. nvservices accepts 0x12B and 0x12E in bits 28:20
/// (text `0x7A3F0`).
pub const BOOT_0: u32 = 0x12B0_00A1;

/// `top_num_gpcs`, `top_tpc_per_gpc`, `top_num_fbps`, `top_ltc_per_fbp`,
/// `top_slices_per_ltc`: 1, 2, 1, 2, 1 (characteristics `max_gpc_count`,
/// `num_tpc_per_gpc`, `max_fbps_count`, `max_ltc_per_fbp`,
/// `max_lts_per_ltc`).
pub const TOP_NUM_GPCS: u32 = 1;
pub const TOP_TPC_PER_GPC: u32 = 2;
pub const TOP_NUM_FBPS: u32 = 1;
pub const TOP_LTC_PER_FBP: u32 = 2;
pub const TOP_SLICES_PER_LTC: u32 = 1;
/// `top_num_ltcs`: one FBP with two LTCs. Not reported anywhere; the
/// product of the two counts above, and nvservices' own L2 flush polls the
/// two LTCs at `0x1402A4` and `0x1422A4` (`NV_FlushLtcCleanAndInvalidate_84CF0`).
pub const TOP_NUM_LTCS: u32 = 2;

/// `pri_ringmaster_enum_fbp`, `_enum_gpc`, `_enum_ltc`: the stations the
/// priv ring found, the same counts as above (characteristics `num_gpc` is
/// read from `enum_gpc` by nvgpu).
pub const RING_ENUM_FBP: u32 = 1;
pub const RING_ENUM_GPC: u32 = 1;
pub const RING_ENUM_LTC: u32 = 2;

/// Floor-sweeping fuses: nothing disabled. `fuse_status_opt_gpc` 0 gives
/// the reported GPC mask 1; `fuse_status_opt_tpc_gpc0` 0 enables both TPCs
/// (`~fuse & 3`); the FBIO, FBP and ROP/L2 fuses 0 leave the one FBP and
/// its two LTCs working, as the reported L2 size requires.
pub const FUSE_STATUS_OPT_FBIO: u32 = 0;
pub const FUSE_STATUS_OPT_GPC: u32 = 0;
pub const FUSE_STATUS_OPT_TPC_GPC0: u32 = 0;
pub const FUSE_STATUS_OPT_FBP: u32 = 0;
pub const FUSE_STATUS_OPT_ROP_L2_FBP0: u32 = 0;

/// `gr_gpc0_fs_gpc`: 2 TPCs available (bits 4:0), 4 ZCULL banks (bits
/// 20:16, nvgpu `proj_scal_litter_num_zcull_banks` = 4; consistent with
/// the zcull geometry a Switch reports through yuzu's recorded answers).
pub const GPC0_FS_GPC: u32 = 4 << 16 | 2;
/// `gr_gpc0_gpm_pd_pes_tpc_id_mask(0)`: the GPC's single PES (nvgpu
/// `proj_scal_litter_num_pes_per_gpc` = 1) serves both TPCs.
pub const GPC0_PES0_TPC_MASK: u32 = 0x3;
/// `gr_gpc0_zcull_total_ram_size`: 0x800 aliquots (the `aliquot_total` a
/// Switch reports, as recorded by yuzu).
pub const GPC0_ZCULL_TOTAL_RAM_SIZE: u32 = 0x800;
/// `gr_gpc0_tpc0_sm_arch`: SM version 0x503 (bits 31:20), SPA version
/// 0x503 (bits 19:8), 0x80 warps.
pub const GPC0_TPC0_SM_ARCH: u32 = 0x503 << 20 | 0x503 << 8 | 0x80;
/// `ltc_ltc0_lts0_tstg_cfg1`: all 16 ways active (bits 15:0), all sets
/// (bits 17:16 = 0). With 2 LTCs this is nvgpu's 256 KiB; other splits of
/// ways and sets would give the same size, so this one is a choice.
pub const LTC0_LTS0_TSTG_CFG1: u32 = 0xFFFF;
/// `ltc_ltcs_ltss_cbc_param`: one slice per LTC (bits 31:28, as
/// `top_slices_per_ltc`), and no compression tags per cache line (bits
/// 15:0). Assumed: the chip's compression-bit cache geometry is not
/// published. With 0 tags nvservices allocates no compression backing
/// store (`0x84800`, nvgpu's `gm20b_ltc_init_comptags` path for a chip
/// without comptags), so buffers are never compressed — which is also
/// all a model without compression can render into.
pub const LTCS_CBC_PARAM: u32 = 1 << 28;

/// `fifo_pbdma_map(0)`: the single PBDMA serves runlist 0, the only one
/// (nvgpu `fifo_eng_runlist__size_1` = 1, `proj_host_num_pbdma` = 1).
pub const PBDMA_MAP_0: u32 = 1;

/// The engine table, `top_device_info(i)` (layout: nvgpu `hw_top_gm20b.h`,
/// nouveau `top/gk104.c`). One device: graphics, an `ENGINE_TYPE` entry
/// (type 0, chained) followed by its `ENUM` entry with engine 0, runlist 0,
/// interrupt 12 and reset 12 — PGRAPH's bit in the PMC interrupt and
/// enable registers (nvgpu `mc_enable_pgraph_enabled_f` = 0x1000; the
/// interrupt bit is 12 on gk20a and gp10b). The rest of the table reads as
/// "not valid".
///
/// Assumed: the copy engine (CE2) is left out. Its interrupt bit is not
/// published, and nvservices only uses the entry to enable that interrupt
/// (`NV_BuildConfiguredEngineBitmask_70BD0`), which this model never
/// raises; copies are executed as part of the channel's engines.
pub const DEVICE_INFO: [u32; 2] = [1 << 31 | 3, 12 << 15 | 12 << 9 | 0x3E];

/// `pwr_falcon_hwcfg`: instruction memory in 256-byte blocks in bits 8:0,
/// data memory in bits 17:9. Assumed: 48 KiB and 64 KiB. Nothing public
/// gives the PMU's sizes. nvservices reads this to place the bootloader at
/// the top of instruction memory (taking the block count as 8 bits, so a
/// real count is below 256) and the PMU's arguments 32 bytes below the top
/// of data memory; the model executes neither, so only the places the
/// driver writes to follow from it.
pub const PMU_HWCFG: u32 = 0x100 << 9 | 0xC0;

/// What FECS answers for its image sizes (methods 0x10, 0x16, 0x25, 0x30).
/// Assumed: the firmware computes these, and they are not published. The
/// driver only requires them to be non-zero (`NV_QueryContextImageSizes_639E0`
/// and `0x65880`) and allocates buffers of these sizes; they are 4 KiB
/// multiples, large enough for the context header words the driver
/// patches (up to +0xF4, annex boot §7.4).
pub const FECS_GOLDEN_IMAGE_SIZE: u32 = 0x2_0000;
pub const FECS_ZCULL_IMAGE_SIZE: u32 = 0x1000;
pub const FECS_PM_IMAGE_SIZE: u32 = 0x1000;
pub const FECS_REGLIST_IMAGE_SIZE: u32 = 0x1000;

/// `trim_sys_gpcpll_cfg3.dfs_testout`: the GPC PLL's voltage ADC reading
/// after its power-on calibration. pcv (like nvgpu's
/// `clk_gm20b.c:gm20b_clk_calibrate`) uses it only to place the ADC's
/// offset: offset = power-on voltage − reading × 10 mV, so the reading
/// stands for the power-on voltage, and every later ADC code is that
/// reading plus (target − power-on voltage) / 10 mV, a 7-bit field.
/// Assumed 0x30, which keeps codes for targets from 0.52 V to 1.79 V in
/// range and the offset non-zero (pcv treats a zero offset as "not
/// calibrated"). The value belongs to a particular chip's ADC.
pub const GPCPLL_DFS_TESTOUT: u32 = 0x30;

/// `fb_mmu_ctrl.pri_fifo_space` (bits 23:16): the free entries of the MMU's
/// request FIFO, all of them since the model never queues one. Assumed
/// depth; the driver only waits for a non-zero value before invalidating.
pub const MMU_PRI_FIFO_DEPTH: u32 = 0x10;

/// `pwr_pmu_scpctl_stat`: bit 20 set on debug-fused chips, where the driver
/// patches the ACR with its debug signature (spec §4.4). A console is
/// production-fused; the modeled ACR checks no signature either way.
pub const PMU_SCPCTL_STAT: u32 = 0;

/// Registers that report the above, by BAR0 offset.
pub const READ_ONLY: [(u32, u32); 23] = [
    (0x00_0000, BOOT_0),
    (0x00_2390, PBDMA_MAP_0),
    (0x02_1C14, FUSE_STATUS_OPT_FBIO),
    (0x02_1C1C, FUSE_STATUS_OPT_GPC),
    (0x02_1C38, FUSE_STATUS_OPT_TPC_GPC0),
    (0x02_1D38, FUSE_STATUS_OPT_FBP),
    (0x02_1D70, FUSE_STATUS_OPT_ROP_L2_FBP0),
    (0x02_2430, TOP_NUM_GPCS),
    (0x02_2434, TOP_TPC_PER_GPC),
    (0x02_2438, TOP_NUM_FBPS),
    (0x02_2450, TOP_LTC_PER_FBP),
    (0x02_2454, TOP_NUM_LTCS),
    (0x02_245C, TOP_SLICES_PER_LTC),
    (0x10_A108, PMU_HWCFG),
    (0x10_AC08, PMU_SCPCTL_STAT),
    (0x12_006C, RING_ENUM_LTC),
    (0x12_0074, RING_ENUM_FBP),
    (0x12_0078, RING_ENUM_GPC),
    (0x14_0494, LTC0_LTS0_TSTG_CFG1),
    (0x17_E280, LTCS_CBC_PARAM),
    (0x50_0920, GPC0_ZCULL_TOTAL_RAM_SIZE),
    (0x50_0C30, GPC0_PES0_TPC_MASK),
    (0x50_2608, GPC0_FS_GPC),
];

/// `0x50469C` reads the SM architecture; listed apart because the table
/// above is sorted by unit and this one is per TPC.
pub const SM_ARCH_OFFSET: u32 = 0x50_469C;

/// The device table's first word; entries `DEVICE_INFO.len()..64` read 0.
pub const DEVICE_INFO_BASE: u32 = 0x02_2700;
pub const DEVICE_INFO_ENTRIES: u32 = 64;
