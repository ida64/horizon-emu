//! Boot ABI: the machine state the kernel is entitled to when it gains control.
//!
//! Everything here is what the secure monitor (TrustZone) and package2 loader
//! establish on hardware. Addresses in parentheses are link-time addresses in
//! the retail kernel image this was derived from
//! (sha256 `7a93c43a…f4ec29`, load address `0x80060000`).
//!
//! # Primary core
//!
//! * The whole package2 kernel blob sits contiguously in DRAM at its load
//!   address; execution starts at its first byte (`0x80060000`, a branch to
//!   early entry at `0x800d7000`).
//! * Exception level is EL1. Early entry reads `CurrentEL` and parks the core
//!   for anything else (`0x800d700c`).
//! * `X0`/`X1` are stashed in `X19`/`X20` and never read again, so no register
//!   carries an argument. Interrupt masks, MMU and cache state are overwritten
//!   by the first instructions and are likewise not part of the contract.
//! * No stack: the loader points `SP` into its own image.
//! * DRAM is identity-usable from [`DRAM_BASE`]; the loader places its page
//!   tables and the relocated INI1 purely from the SMC answers documented in
//!   [`crate::secmon`], it never probes memory.
//!
//! # Secondary cores
//!
//! Started by the kernel through PSCI `CPU_ON`. The monitor enters the given
//! physical entry point (`0x800d70e8`) at EL1 with the caller's context
//! pointer in `X0`. That code installs the translation registers the loader
//! saved, then, on Cortex-A57/A53 only, spins unless `CPUACTLR_EL1` and
//! `CPUECTLR_EL1` equal the values core 0 recorded (`0x800d7184`), so those
//! two registers must read identically on every core.
//!
//! # Core identity
//!
//! The kernel derives the core index from `MPIDR_EL1[23:0]` (`0x800c5ff4`),
//! and keys CPU-specific setup on `MIDR_EL1`.

/// Physical address of the first byte of DRAM.
pub const DRAM_BASE: u64 = 0x8000_0000;

/// The SoC has four application cores and the kernel is built for exactly that.
pub const CORE_COUNT: usize = 4;

/// EL1h with D, A, I and F masked.
pub const ENTRY_PSTATE: u64 = 0x3C5;

/// Cortex-A57 r1p3, the application core of the SoC.
pub const MIDR_EL1: u64 = 0x411F_D073;

/// `SCTLR_EL1` with only the ARMv8.0 RES1 bits set: MMU and caches off.
pub const ENTRY_SCTLR_EL1: u64 = 0x30D0_0800;

/// `CPUACTLR_EL1` (`S3_1_C15_C2_0`). The kernel only requires all cores to
/// agree; zero is the Cortex-A57 reset value.
pub const CPUACTLR_EL1: u64 = 0;

/// `CPUECTLR_EL1` (`S3_1_C15_C2_1`) with SMPEN set, which firmware must do
/// before a Cortex-A57 takes part in SMP.
pub const CPUECTLR_EL1: u64 = 1 << 6;

/// `MPIDR_EL1` for `core`: uniprocessor-format bit (RES1) plus Aff0.
pub const fn mpidr_el1(core: usize) -> u64 {
    0x8000_0000 | core as u64
}

/// Installed DRAM. One choice drives every place the size surfaces: the
/// memory controller register, two secure monitor config items, and how much
/// guest memory exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DramSize {
    #[default]
    Gb4,
    Gb6,
    Gb8,
}

impl DramSize {
    pub const fn bytes(self) -> u64 {
        self.megabytes() as u64 * 0x10_0000
    }

    /// Value of the memory controller's `MC_EMEM_CFG` size field.
    pub const fn megabytes(self) -> u32 {
        match self {
            Self::Gb4 => 4096,
            Self::Gb6 => 6144,
            Self::Gb8 => 8192,
        }
    }

    /// Bits 17:16 of the kernel configuration word.
    const fn kernel_configuration_field(self) -> u64 {
        match self {
            Self::Gb4 => 0,
            Self::Gb6 => 1,
            Self::Gb8 => 2,
        }
    }

    /// Memory mode config item for the standard (retail) arrangement of this
    /// size. The kernel switches on the low six bits to size its pools
    /// (`0x800d1d28`).
    pub const fn memory_mode(self) -> u64 {
        match self {
            Self::Gb4 => 0x01,
            Self::Gb6 => 0x11,
            Self::Gb8 => 0x21,
        }
    }

    /// DRAM ID config item, a fuse value naming the memory chips: Samsung
    /// ones of this size, numbered as publicly documented. The firmware's
    /// clock driver asks for it (pcv, `spl:` request 0 with item 2). No
    /// original console was made with 8 GiB.
    pub const fn dram_id(self) -> Option<u64> {
        match self {
            Self::Gb4 => Some(0),
            Self::Gb6 => Some(4),
            Self::Gb8 => None,
        }
    }
}

/// Secure monitor config item 12. Bit positions are what the kernel tests
/// (`0x800d3e20` caches them); names for bits 0-2 and 4 follow public
/// Atmosphère documentation of the same word, not anything recoverable from
/// the kernel alone.
///
/// The default is all clear. The kernel only checks that the word does not
/// change between its two reads, so what a retail monitor reports is not
/// established here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct KernelConfiguration {
    /// Bit 0.
    pub debug_fill_memory: bool,
    /// Bit 1.
    pub user_exception_handlers: bool,
    /// Bit 2.
    pub user_pmu_access: bool,
    /// Bit 3. Grows the slab reservation after the kernel image from
    /// `0x1EB8000` to `0x233C000` bytes (`0x8031c7f0`, `0x800d24d0`).
    pub increase_thread_resource_limit: bool,
    /// Bit 4.
    pub disable_dynamic_resource_limits: bool,
    /// Bit 8. While clear, a fatal kernel error issues the Panic SMC before
    /// parking the core (`0x800d4750`); set, the core parks silently.
    pub skip_panic_smc: bool,
}

impl KernelConfiguration {
    pub const fn bits(self, dram: DramSize) -> u64 {
        (self.debug_fill_memory as u64)
            | (self.user_exception_handlers as u64) << 1
            | (self.user_pmu_access as u64) << 2
            | (self.increase_thread_resource_limit as u64) << 3
            | (self.disable_dynamic_resource_limits as u64) << 4
            | (self.skip_panic_smc as u64) << 8
            | dram.kernel_configuration_field() << 16
    }
}

/// Everything about a boot that is a choice rather than a property of the
/// kernel image.
#[derive(Debug, Clone, Copy, Default)]
pub struct BootParams {
    pub dram: DramSize,
    pub kernel_configuration: KernelConfiguration,
    /// Secure monitor config item 11. The kernel's debug log, its only use of
    /// the UART, is live exactly when this is set (`0x800d3e90`, `0x8006a1c0`).
    pub development_function_enabled: bool,
    /// Seeds the monitor's random number SMC. The kernel's KASLR, stack
    /// placement and physical page shuffle are functions of this value, so a
    /// fixed seed gives a reproducible address space.
    pub rng_seed: u64,
    /// The console's device ID, which the secure monitor derives from the
    /// SoC's fuses (fabrication lot, wafer and die position). Zero is a die
    /// nobody recorded any of that for. It has to match the device ID in
    /// the calibration data (`nand::calibration::Identity::device_id`):
    /// `settings` compares the two (text `0x8920`), and `am` and `nim` abort
    /// when they differ.
    pub device_id: u64,
}

/// Register state a core starts with; everything not listed is don't-care.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoreEntry {
    pub pc: u64,
    pub x0: u64,
}

impl CoreEntry {
    pub const fn primary(kernel_load_address: u64) -> Self {
        Self {
            pc: kernel_load_address,
            x0: 0,
        }
    }

    /// Entry established by PSCI `CPU_ON`.
    pub const fn secondary(entry: u64, context: u64) -> Self {
        Self {
            pc: entry,
            x0: context,
        }
    }
}
