//! Secure monitor (TrustZone) as the kernel sees it.
//!
//! The kernel never touches EL3 state; it issues `SMC #1` with a function
//! identifier in `W0` and arguments in `X1`-`X7`, and gets `X0`-`X7` back with
//! a status in `X0` (`0x80065460`).
//!
//! `SMC #0` carries calls the kernel forwards for a privileged process
//! (`0x800654a0`), with its own function numbering. Bits 15:9 of those
//! identifiers flag which of `X1`-`X7` are user pointers; the kernel replaces
//! each with the physical address of the page before issuing the SMC
//! (`0x800d4440`).
//!
//! Calls the kernel makes for itself while booting, in order of first use
//! (addresses are link-time addresses in the retail image described in
//! [`crate::boot`]):
//!
//! | Function | In | Out | First use |
//! |---|---|---|---|
//! | [`READ_WRITE_REGISTER`](function::READ_WRITE_REGISTER) | `X1` address, `W2` mask, `W3` value | `W1` previous value | loader reads `MC_EMEM_CFG` (`0x8031c858`) |
//! | [`GET_CONFIG`](function::GET_CONFIG) | `W1` item | `X1`-`X4` value | loader, item 12 (`0x8031c874`) |
//! | [`GENERATE_RANDOM_BYTES`](function::GENERATE_RANDOM_BYTES) | `X1` byte count, 1-56 | bytes packed into `X1`-`X7` | loader KASLR (`0x8031c7bc`) |
//! | [`CPU_ON`](function::CPU_ON) | `X1` core, `X2` entry PA, `X3` context | PSCI status | `0x800cecc0` |
//! | [`CONFIGURE_CARVEOUT`](function::CONFIGURE_CARVEOUT) | `X1` index, `X2` address, `X3` size | | `0x800d25f0` |
//! | [`PANIC`](function::PANIC) | `W1` code | does not return | `0x800d2650` |
//!
//! And when the console goes to sleep, from the kernel's sleep handler on
//! each core (mesosphere `kern_k_sleep_manager_asm.s`, `CpuSleepHandler`):
//! cores 1 to 3 call [`CPU_OFF`](function::CPU_OFF), core 0 waits for their
//! power partitions to read off and calls
//! [`CPU_SUSPEND`](function::CPU_SUSPEND). Neither returns.
//!
//! # Sleep
//!
//! On the console the monitor then powers the whole SoC down but for the
//! PMC and the memory, which keeps itself; a wake event starts the SoC
//! from its boot ROM, whose warm boot path leads back into the monitor,
//! and the monitor starts core 0 where the kernel asked. In a machine that
//! ran the boot chain (the monitor did its [`cold_boot`]) that is what
//! happens: the monitor prepares SC7 as the real one does and restarts the
//! BPMP on its SC7 stub ([`sc7`], [`SmcEffect::SystemSuspend`]), and the
//! warmboot firmware releases core 0 into its resume
//! ([`SecureMonitor::core0_released`]). In one started from a kernel ELF
//! nothing loses power: core 0 stands still until a wake event and then
//! starts at that address, and every device is as the firmware left it
//! instead of out of reset; the firmware's drivers put back what a real
//! sleep loses, which there writes what is already there. Before sleeping
//! the monitor checks that the other cores are off and the SoC is in the
//! state its sleep sequence expects (exosphere
//! `secmon_smc_power_management.cpp`, `ValidateSocStateForSuspend`;
//! `SLEEP_REQUIREMENTS` here), and aborts otherwise.
//!
//! Config items read while booting: 1, 10 (memory mode), 11 and 12 (kernel
//! configuration). Every wrapper in the kernel treats a non-zero status as
//! fatal, so the only error paths here are argument bounds the kernel itself
//! stays within; anything not modeled stops the machine instead of handing the
//! guest a made-up answer.

use std::{fmt, sync::Mutex};

pub mod cold_boot;
pub mod crypto;
pub mod sc7;

pub use cold_boot::{ColdBoot, ColdBootContext, ColdBootError, Released};

use crate::{
    boot::{BootParams, CORE_COUNT, DRAM_BASE, DramSize},
    bus::{Access, Bus},
    keys::{Key, KeySet},
    memory::GuestRam,
    soc::mc::DeviceMemory,
};

use self::cold_boot::BootSettings;
use self::crypto::{CipherMode, Crypto, CryptoError, RSA_SIZE, SavedCrypto};
use crate::snapshot::{Snapshot, SnapshotError, decode, encode};

/// Immediate of the SMC instruction for kernel-originated calls.
pub const KERNEL_SMC: u16 = 1;

/// Immediate of the SMC instruction for calls forwarded from userland.
pub const USER_SMC: u16 = 0;

pub mod function {
    /// `W1` power state, `X2` entry PA, `X3` context. The one state there
    /// is: [`super::POWER_STATE_SLEEP`].
    pub const CPU_SUSPEND: u32 = 0xC400_0001;
    /// No arguments: switches the calling core off.
    pub const CPU_OFF: u32 = 0x8400_0002;
    pub const CPU_ON: u32 = 0xC400_0003;
    pub const GET_CONFIG: u32 = 0xC300_0004;
    pub const GENERATE_RANDOM_BYTES: u32 = 0xC300_0005;
    pub const PANIC: u32 = 0xC300_0006;
    pub const CONFIGURE_CARVEOUT: u32 = 0xC300_0007;
    pub const READ_WRITE_REGISTER: u32 = 0xC300_0008;
    /// The kernel's SetConfig: `W1` item, `X3` value.
    pub const SET_CONFIG: u32 = 0xC300_0409;
}

/// The kernel's call table: the function ID of each entry, by the entry's
/// number (the ID's low byte). Entry 0 is empty. The tables, the rule that
/// an ID must be exactly the one at its entry, and the abort on anything
/// else are those of Atmosphère's reimplementation of the monitor
/// (exosphere `secmon_smc_handler.cpp`), which mirrors Nintendo's
/// interface \[inference: the firmware's own calls are all in them\].
const KERNEL_TABLE: [Option<u32>; 10] = [
    None,
    Some(function::CPU_SUSPEND),
    Some(function::CPU_OFF),
    Some(function::CPU_ON),
    Some(function::GET_CONFIG),
    Some(function::GENERATE_RANDOM_BYTES),
    Some(function::PANIC),
    Some(function::CONFIGURE_CARVEOUT),
    Some(function::READ_WRITE_REGISTER),
    Some(function::SET_CONFIG),
];

/// Userland's call table, likewise; entries 0 and 14 are empty.
const USER_TABLE: [Option<u32>; 19] = [
    None,
    Some(user_function::SET_CONFIG),
    Some(user_function::GET_CONFIG),
    Some(user_function::GET_RESULT),
    Some(user_function::GET_RESULT_DATA),
    Some(user_function::MODULAR_EXPONENTIATE),
    Some(user_function::GENERATE_RANDOM_BYTES),
    Some(user_function::GENERATE_AES_KEK),
    Some(user_function::LOAD_AES_KEY),
    Some(user_function::COMPUTE_AES),
    Some(user_function::GENERATE_SPECIFIC_AES_KEY),
    Some(user_function::COMPUTE_CMAC),
    Some(user_function::REENCRYPT_DEVICE_UNIQUE_DATA),
    Some(user_function::DECRYPT_DEVICE_UNIQUE_DATA),
    None,
    Some(user_function::MODULAR_EXPONENTIATE_BY_STORAGE_KEY),
    Some(user_function::PREPARE_ES_DEVICE_UNIQUE_KEY),
    Some(user_function::LOAD_PREPARED_AES_KEY),
    Some(user_function::PREPARE_ES_COMMON_TITLE_KEY),
];

/// The core userland's calls come from: `spl`, the one process the kernel
/// lets forward them, runs there.
const USER_CALL_CORE: usize = 3;

/// Why the monitor gives up. On the console that ends everything: the
/// monitor stops the console with an error of its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Abort {
    /// A call with an ID the gate's table does not have at that entry, an
    /// empty entry, or an SMC immediate with no table.
    UnknownFunction { immediate: u16, id: u32 },
    /// A userland call made on another core than core 3.
    UserCallOnCore { core: usize, id: u32 },
    /// Sleep asked for while another core was on.
    SleepWithCoresOn,
    /// Sleep asked for with the SoC in a state the monitor does not put to
    /// sleep: a register the monitor checks off its required value.
    SocNotReadyForSleep { what: &'static str, value: u32 },
    /// The monitor panics while preparing SC7 ([`sc7`]).
    SuspendPanic(&'static str),
    /// A step of the SC7 preparation that is not modelled here.
    SuspendNotModelled(&'static str),
}

impl fmt::Display for Abort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::UnknownFunction { immediate, id } => {
                write!(
                    f,
                    "SMC #{immediate} function {id:#010x} is in no call table"
                )
            }
            Self::UserCallOnCore { core, id } => {
                write!(f, "userland call {id:#010x} on core {core}, not core 3")
            }
            Self::SleepWithCoresOn => write!(f, "sleep asked for while another core was on"),
            Self::SocNotReadyForSleep { what, value } => {
                write!(
                    f,
                    "sleep asked for without {what} (register reads {value:#010x})"
                )
            }
            Self::SuspendPanic(what) => write!(f, "while preparing SC7: {what}"),
            Self::SuspendNotModelled(what) => {
                write!(f, "SC7 preparation not modelled here: {what}")
            }
        }
    }
}

/// Whether `id` is the function at its entry of the table of the gate
/// `immediate`.
fn in_table(immediate: u16, id: u32) -> bool {
    let table: &[Option<u32>] = match immediate {
        USER_SMC => &USER_TABLE,
        KERNEL_SMC => &KERNEL_TABLE,
        _ => &[],
    };
    table.get((id & 0xFF) as usize) == Some(&Some(id))
}

/// Functions reachable through [`USER_SMC`].
pub mod user_function {
    /// `W1` item in, `X1`-`X4` value out; the items are those of the kernel's
    /// [`GET_CONFIG`](super::function::GET_CONFIG).
    pub const GET_CONFIG: u32 = 0xC300_0002;
    /// `W1` item, `X3` value in. Only
    /// [`IS_CHARGER_HI_Z_MODE_ENABLED`](super::config_item::IS_CHARGER_HI_Z_MODE_ENABLED)
    /// can be set; `ptm` sets it to 0 on a console that has been set up
    /// before (`spl` SetConfig, command 5). That the value is in `X3`, with
    /// `X2` unused, follows Atmosphère's public monitor
    /// (`secmon_smc_info.cpp`), from memory \[assumption\]: the one call seen
    /// has zero in both.
    pub const SET_CONFIG: u32 = 0xC300_0401;
    /// `X1` operation key in; the operation's own status out in `X1` (`spl`
    /// fails with "unknown secure monitor error" on anything it does not
    /// know there).
    pub const GET_RESULT: u32 = 0xC300_0003;
    /// As the kernel's [`GENERATE_RANDOM_BYTES`](super::function::GENERATE_RANDOM_BYTES).
    pub const GENERATE_RANDOM_BYTES: u32 = 0xC300_0006;
    /// `X1`-`X2` kek source, `X3` key generation, `W4` options in; sealed
    /// access key out in `X1`-`X2`.
    pub const GENERATE_AES_KEK: u32 = 0xC300_0007;
    /// `X1` key slot, `X2`-`X3` access key, `X4`-`X5` key source in.
    pub const LOAD_AES_KEY: u32 = 0xC300_0008;
    /// `X1`-`X2` key source, `X3` key generation, `W4` which key in; the key
    /// out in `X1`-`X2`.
    pub const GENERATE_SPECIFIC_AES_KEY: u32 = 0xC300_000A;
    /// `W1` mode (key slot in bits 2..0, cipher mode in bits 5..4),
    /// `X2`-`X3` IV or counter, `W4` and `W5` addresses of the security
    /// engine's source and destination buffer lists, `W6` byte count in;
    /// operation key out in `X1`. All addresses are the engine's device
    /// addresses: `spl` maps its work buffer for the engine at `0x8003_5000`
    /// behind the SMMU. The monitor raises
    /// [`SECURITY_ENGINE_INTERRUPT`](super::SECURITY_ENGINE_INTERRUPT) when
    /// the engine is done, and the caller collects the outcome with
    /// [`GET_RESULT`].
    pub const COMPUTE_AES: u32 = 0xC300_0009;
    /// `X1` key slot, `X2` address of the data, `X3` its length in; the
    /// AES-CMAC out in `X1`-`X2`. Bit 10 of the identifier marks `X2` as a
    /// user pointer, so what arrives is the physical address the kernel
    /// found for it, good for the rest of that page.
    pub const COMPUTE_CMAC: u32 = 0xC300_040B;
    /// `X1`-`X2` access key, `W3` mode, `X4` address of the data, `X5` its
    /// length, `X6`-`X7` key source in; the data decrypted in place (the
    /// format is [`super::crypto::seal_device_unique_data`]'s). Bit 12 of
    /// the identifier marks `X4` as a user pointer, which the kernel turns
    /// into a physical address; `spl` passes its work buffer (`0x35000`,
    /// text `0x11340`). Mode 0 hands the plain data back; modes 1 to 4
    /// keep the RSA key inside instead (Lotus, ES device, TLS, ES client
    /// certificate) for the exponentiation calls. `ssl`
    /// reaches it through `spl` to unwrap its TLS client key of key
    /// generation 0 (`0x35f80`), and `spl` copies the payload, all but the
    /// last `0x30` bytes, back to it.
    pub const DECRYPT_DEVICE_UNIQUE_DATA: u32 = 0xC300_100D;
    /// `X1` operation key, `X2` address of the caller's buffer (a user
    /// pointer), `X3` its size in; the operation's status in `X1` and its
    /// output copied to the buffer, which must ask for exactly its size.
    pub const GET_RESULT_DATA: u32 = 0xC300_0404;
    /// `X1` message, `X2` exponent, `X3` modulus (user pointers, one page),
    /// `X4` exponent length in; asynchronous, the 256-byte result through
    /// [`GET_RESULT_DATA`].
    pub const MODULAR_EXPONENTIATE: u32 = 0xC300_0E05;
    /// Addresses of the opening access key `X1`, the sealing access key
    /// `X2`, the opening and sealing key sources `X6` and `X7`, `W3` mode
    /// of the result, `X4` data and `X5` its length; the data rewrapped in
    /// place.
    pub const REENCRYPT_DEVICE_UNIQUE_DATA: u32 = 0xC300_D60C;
    /// `X1` message, `X2` modulus (user pointers), `W3` which stored key
    /// (Lotus, TLS, ES client certificate) in; asynchronous like
    /// [`MODULAR_EXPONENTIATE`].
    pub const MODULAR_EXPONENTIATE_BY_STORAGE_KEY: u32 = 0xC300_060F;
    /// `X1` personalized title key block, `X2` modulus (user pointers),
    /// `X3`-`X6` OAEP label digest, `W7` key generation and common key
    /// type in; asynchronous, a 16-byte sealed title key through
    /// [`GET_RESULT_DATA`].
    pub const PREPARE_ES_DEVICE_UNIQUE_KEY: u32 = 0xC300_0610;
    /// `X1` key slot, `X2`-`X3` sealed title key in.
    pub const LOAD_PREPARED_AES_KEY: u32 = 0xC300_0011;
    /// `X1`-`X2` encrypted title key, `X3` key generation in; the sealed
    /// title key in `X1`-`X2`.
    pub const PREPARE_ES_COMMON_TITLE_KEY: u32 = 0xC300_0012;
}

/// Config items. Numbers 1 and 10-12 are established by what the kernel
/// does with them; the meaning of the others, and the values served for
/// them, follow the public documentation of the monitor's interface and
/// Atmosphère's reimplementation (exosphere `secmon_smc_info.cpp`). The
/// monitor knows items 1 to 18; any other is an invalid argument, and
/// only [`IS_CHARGER_HI_Z_MODE_ENABLED`](config_item::IS_CHARGER_HI_Z_MODE_ENABLED) can be set.
pub mod config_item {
    pub const DISABLE_PROGRAM_VERIFICATION: u32 = 1;
    pub const DRAM_ID: u32 = 2;
    /// Interrupt the monitor raises when an asynchronous operation completes.
    pub const SECURITY_ENGINE_INTERRUPT_NUMBER: u32 = 3;
    /// How many times the anti-downgrade fuses have been burnt, as the
    /// monitor expects them for the firmware it belongs to: one more for
    /// each firmware that burnt another (exosphere `fuse_api.cpp`,
    /// `FuseVersionIncrementFirmwares`, 25 entries with 23.5.0 the
    /// newest), so 25 for 23.5.0.
    pub const FUSE_VERSION: u32 = 4;
    pub const HARDWARE_TYPE: u32 = 5;
    pub const HARDWARE_STATE: u32 = 6;
    /// Removed in firmware 4.0.0; the monitor answers invalid argument.
    pub const BOOT_REASON: u32 = 9;
    /// The SHA-256 of the whole package2, in `X1`-`X4`: given only in a
    /// recovery boot, invalid argument otherwise (the 23.5.0 monitor's
    /// GetConfig; the hash is taken on cold boot, [`super::cold_boot`]).
    pub const PACKAGE2_HASH: u32 = 17;
    /// The quadword at `+0x70` of the boot config the bootloader hands the
    /// monitor. A production console's monitor ignores the boot config
    /// (the 23.5.0 monitor zeroes its copy), so the field is zero.
    pub const UNKNOWN_18: u32 = 18;
    /// Whether the console was started into its recovery (maintenance)
    /// mode: bit 0 of IRAM `0x4000001C` on cold boot, else not.
    pub const IS_RECOVERY_BOOT: u32 = 7;
    pub const DEVICE_ID: u32 = 8;
    /// Whether the console is a shop's interactive display unit, a fuse
    /// setting (meaning as publicly documented). It is not.
    pub const RETAIL_INTERACTIVE_DISPLAY_STATE: u32 = 14;
    /// Which regulators feed the processor and the GPU. Zero: a MAX77621
    /// each, the only arrangement of the original console (value as publicly
    /// documented).
    pub const REGULATOR_TYPE: u32 = 15;
    /// Generation of the console-unique key the console was provisioned
    /// with. Zero: the original one, the only one the key ladder here has.
    pub const DEVICE_UNIQUE_KEY_GENERATION: u32 = 16;
    pub const MEMORY_MODE: u32 = 10;
    pub const IS_DEVELOPMENT_FUNCTION_ENABLED: u32 = 11;
    pub const KERNEL_CONFIGURATION: u32 = 12;
    /// Whether the monitor puts the charger into high-impedance mode when
    /// the console powers off. Settable; starts out disabled \[assumption:
    /// nothing set it before the first call seen, which writes 0\].
    pub const IS_CHARGER_HI_Z_MODE_ENABLED: u32 = 13;
    /// The last item Nintendo's monitor knows. Atmosphère's monitor
    /// answers more from 65000 on (its version and settings, which
    /// homebrew asks for to learn whether it runs under Atmosphère);
    /// Nintendo's, which this is, has no such items.
    pub const LAST: u32 = 18;
}

/// [`config_item::FUSE_VERSION`] for firmware 23.5.0.
const FUSE_VERSION: u64 = 25;

/// Interrupt ID the monitor signals asynchronous completions with.
pub const SECURITY_ENGINE_INTERRUPT: u64 = 44;

/// `GICD_ISPENDR1`, where the monitor sets that interrupt pending.
const SET_PENDING_REGISTER: u64 = 0x5004_1204;

/// Most the engine is asked to process in one operation here; its buffers
/// are a few pages of the caller's.
const MAX_ENGINE_BYTES: u64 = 0x10_0000;

/// Where `ComputeAes` takes its engine buffer lists from: device addresses
/// in the first 4 MiB of DRAM less 2 KiB, room for a whole 12-byte list
/// (exosphere `IsValidLinkedListAddress`).
const ENGINE_LISTS: std::ops::RangeInclusive<u64> =
    DRAM_BASE..=DRAM_BASE + (4 << 20) - (2 << 10) - 12;

/// Most data `ComputeCmac` takes (`CmacSizeMax`).
const MAX_CMAC_BYTES: u64 = 0x400;
/// Most data `GetResultData` hands back (its work buffer).
const MAX_RESULT_BYTES: u64 = 0x400;

const PAGE_SIZE: u64 = 0x1000;

/// Hardware type of the original console model (the Tegra X1 board this
/// machine is).
const HARDWARE_TYPE_ICOSA: u64 = 0;

/// Hardware state of a retail unit, as opposed to a development one.
const HARDWARE_STATE_PRODUCTION: u64 = 1;

const STATUS_SUCCESS: u64 = 0;
const STATUS_NOT_SUPPORTED: u64 = 1;
const STATUS_INVALID_ARGUMENT: u64 = 2;
/// The security engine is taken: an asynchronous operation is still to be
/// collected.
const STATUS_BUSY: u64 = 3;
const STATUS_NO_ASYNC_OPERATION: u64 = 4;
const STATUS_INVALID_ASYNC_OPERATION: u64 = 5;
/// A key the call needs has not been imported.
const STATUS_NOT_INITIALIZED: u64 = 7;

const PSCI_SUCCESS: u64 = 0;
const PSCI_INVALID_PARAMETERS: u64 = -2i64 as u64;
const PSCI_DENIED: u64 = -3i64 as u64;
const PSCI_ALREADY_ON: u64 = -4i64 as u64;

/// The power state the kernel suspends core 0 into: a power-down state
/// (bit 16) with ID 27, the whole SoC off (exosphere's `PowerStateId_Sc7`;
/// the kernel passes `0x0201001b`). The monitor denies any other.
pub const POWER_STATE_SLEEP: u32 = 1 << 16 | 27;
const POWER_STATE_TYPE_AND_ID: u32 = 0x1_FFFF;

/// The PMC's power partition switch, its status, and the partition of
/// each core (exosphere `secmon_smc_power_management.cpp`,
/// `CpuPowerGateTogglePartitionIds`: CE0 is 14, CE1 to CE3 are 9 to 11).
const PMC_PWRGATE_TOGGLE: u64 = 0x7000_E430;
const PMC_PWRGATE_STATUS: u64 = 0x7000_E438;
const PWRGATE_TOGGLE_START: u64 = 1 << 8;
const CORE_PARTITIONS: [u32; CORE_COUNT] = [14, 9, 10, 11];

/// What the monitor checks of the SoC before it puts it to sleep, as
/// `(register, mask, required value, what it means)`; anything else
/// aborts (exosphere `ValidateSocStateForSuspend`, bit positions from the
/// TRM's clock and reset device numbering). Two of its checks are not
/// here: the BPMP parked by the flow controller, which only a machine
/// that runs the boot chain has and checks ([`sc7`]), and USB, USB2, AHB
/// DMA and the BPMP kept off the AHB arbiter (the boot chain's setting, in
/// a block not modelled here).
pub const SLEEP_REQUIREMENTS: [(u64, u32, u32, &str); 8] = [
    (PMC_PWRGATE_STATUS, 7 << 9, 0, "cores 1 to 3 power-gated"),
    (
        CAR_BASE + 0x008,
        1 << 26 | 1 << 2 | 1 << 1,
        1 << 26 | 1 << 2 | 1 << 1,
        "USB2, APB DMA and AHB DMA held in reset",
    ),
    (
        CAR_BASE + 0x004,
        1 << 22,
        1 << 22,
        "the USB device controller held in reset",
    ),
    (
        CAR_BASE + 0x010,
        1 << 8,
        1 << 8,
        "the GPIO controller clocked",
    ),
    (CAR_BASE + 0x014, 3 << 7, 3 << 7, "FUSE and KFUSE clocked"),
    (
        CAR_BASE + 0x018,
        0xF << 20,
        0xF << 20,
        "all of IRAM clocked",
    ),
    (
        CAR_BASE + 0x360,
        1 << 23,
        1 << 23,
        "the activity monitor clocked",
    ),
    (
        CAR_BASE + 0x364,
        1 << 21,
        1 << 21,
        "the entropy source clocked",
    ),
];
const CAR_BASE: u64 = crate::soc::car::BASE;

const MAX_RANDOM_BYTES: u64 = 56;
const CARVEOUT_COUNT: usize = crate::soc::mc::KERNEL_CARVEOUT_COUNT;
/// Kernel carveouts start and end on 128 KiB boundaries, the memory
/// controller's carveout unit, and are at most 512 MiB less one unit
/// (exosphere `SmcSetKernelCarveoutRegion`, `CarveoutSizeMax`).
const CARVEOUT_GRANULE: u64 = 128 << 10;
const CARVEOUT_SIZE_MAX: u64 = (512 << 20) - CARVEOUT_GRANULE;

/// The registers the monitor reads and writes on the kernel's behalf
/// (ReadWriteRegister), each block as runs of allowed register offsets,
/// first and last inclusive, 4 apart within a run. Any other address is
/// an invalid argument. The sets are the monitor's, as Atmosphère's
/// reimplementation lists them (exosphere `secmon_smc_register_access.cpp`
/// and its PMC, MC and MC channel tables); here as the offsets alone.
const PROXIED_REGISTERS: [(u64, &[(u64, u64)]); 4] = [
    (0x7000_E400, &PMC_ALLOWED),
    (0x7001_9000, &MC_ALLOWED),
    (0x7001_C000, &MC_CHANNEL_ALLOWED),
    (0x7001_D000, &MC_CHANNEL_ALLOWED),
];

/// PMC: control, wake mask/level/status, deep power down, clamping, the
/// power partitions, timers, I/O power, and a few scratch and pad
/// registers.
const PMC_ALLOWED: [(u64, u64); 13] = [
    (0x000, 0x000),
    (0x00C, 0x014),
    (0x01C, 0x020),
    (0x02C, 0x048),
    (0x0DC, 0x0E4),
    (0x160, 0x168),
    (0x170, 0x170),
    (0x1A8, 0x1A8),
    (0x1B8, 0x1C8),
    (0x2B4, 0x2B4),
    (0x2D4, 0x2D4),
    (0x440, 0x440),
    (0x4D8, 0x4D8),
];

/// The memory controller: 179 registers, its configuration and the
/// SMMU's, without the security, carveout and secure SMMU controls.
const MC_ALLOWED: [(u64, u64); 49] = [
    (0x000, 0x010),
    (0x01C, 0x020),
    (0x030, 0x034),
    (0x050, 0x054),
    (0x090, 0x0C8),
    (0x0D0, 0x0E0),
    (0x200, 0x204),
    (0x238, 0x238),
    (0x240, 0x244),
    (0x250, 0x258),
    (0x264, 0x274),
    (0x280, 0x28C),
    (0x294, 0x294),
    (0x2E4, 0x2EC),
    (0x2F4, 0x2F8),
    (0x310, 0x314),
    (0x320, 0x320),
    (0x328, 0x328),
    (0x344, 0x348),
    (0x370, 0x374),
    (0x37C, 0x380),
    (0x390, 0x398),
    (0x3AC, 0x3AC),
    (0x3B8, 0x3C4),
    (0x3D8, 0x3D8),
    (0x3E8, 0x3E8),
    (0x41C, 0x43C),
    (0x44C, 0x44C),
    (0x47C, 0x484),
    (0x4AC, 0x4B4),
    (0x4C4, 0x4CC),
    (0x4DC, 0x4F0),
    (0x50C, 0x50C),
    (0x518, 0x55C),
    (0x584, 0x58C),
    (0x610, 0x630),
    (0x670, 0x674),
    (0x690, 0x6A4),
    (0x6C0, 0x6C4),
    (0x6F0, 0x6F4),
    (0x960, 0x960),
    (0x970, 0x974),
    (0x9B8, 0x9B8),
    (0xA88, 0xA88),
    (0xA94, 0xAE0),
    (0xBCC, 0xBEC),
    (0xC00, 0xC00),
    (0xC5C, 0xC5C),
    (0xCAC, 0xCAC),
];

/// Each of the two memory controller channels (the same set for both).
const MC_CHANNEL_ALLOWED: [(u64, u64); 9] = [
    (0x100, 0x100),
    (0x108, 0x10C),
    (0x118, 0x11C),
    (0x124, 0x13C),
    (0x158, 0x15C),
    (0x164, 0x17C),
    (0xA20, 0xA24),
    (0xB88, 0xB8C),
    (0xBC4, 0xBC8),
];

/// Two memory controller registers the monitor answers success for
/// without touching: the SMMU address-space bindings of the BPMP
/// (`MC_SMMU_AVPC_ASID`, +0x23c) and of APB DMA (`MC_SMMU_PPCS1_ASID`,
/// +0x298), which would let a non-secure caller point those processors'
/// DMA where it likes (exosphere notes the behaviour; offsets as in
/// Linux `tegra210.c`). Nothing is read; `X1` comes back zero.
const DECOY_REGISTERS: [u64; 2] = [0x7001_9000 + 0x23C, 0x7001_9000 + 0x298];

/// Whether the monitor reads and writes `address` for the kernel.
fn register_allowed(address: u64) -> bool {
    PROXIED_REGISTERS.iter().any(|&(base, runs)| {
        address.checked_sub(base).is_some_and(|offset| {
            runs.iter()
                .any(|&(first, last)| (first..=last).contains(&offset))
        })
    })
}

/// What the machine must do once [`SecureMonitor::call`] returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmcEffect {
    /// Result registers are final; resume the caller.
    Resume,
    /// Resume the caller and start `core` at physical `entry` with `context` in `X0`.
    StartCore {
        core: usize,
        entry: u64,
        context: u64,
    },
    /// The calling core is switched off. The call does not return; the
    /// core runs again when another turns it on.
    PowerOff,
    /// The console sleeps. The call does not return; when the console
    /// wakes, the calling core starts at physical `entry` with `context`
    /// in `X0`.
    Sleep { entry: u64, context: u64 },
    /// The console goes into SC7 through the boot chain's own firmware
    /// ([`sc7`]): the monitor has prepared it and restarted the BPMP on
    /// its SC7 stub. The call does not return and the calling core (core
    /// 0) is off; it starts again only when the warmboot firmware releases
    /// it, which [`SecureMonitor::core0_released`] turns into the resume.
    SystemSuspend,
    /// The kernel reported a fatal error. The call does not return.
    Panic { code: u32 },
    /// The monitor gives up, as the console's does; nothing runs on.
    Abort(Abort),
    /// The call is outside what is modeled; registers are untouched.
    Unimplemented(Unimplemented),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unimplemented {
    Function {
        immediate: u16,
        id: u32,
    },
    ConfigItem(u32),
    /// A register the proxied device does not model.
    Register {
        address: u64,
        is_write: bool,
    },
    /// A key operation, with what it would need.
    Crypto(&'static str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct Carveout {
    pub address: u64,
    pub size: u64,
}

pub struct SecureMonitor {
    dram: DramSize,
    /// What the monitor was built with; on the boot chain, the options
    /// that override what the boot config says ([`cold_boot`]).
    params: BootParams,
    device_id: u64,
    state: Mutex<State>,
}

struct State {
    rng: SplitMix64,
    /// A stream of its own, so that key handling does not shift the random
    /// numbers the kernel's layout is derived from.
    key_rng: SplitMix64,
    cores_on: [bool; CORE_COUNT],
    carveouts: [Carveout; CARVEOUT_COUNT],
    /// The config items the boot config and the mailbox decide: the boot
    /// parameters' until a cold boot sets them.
    settings: BootSettings,
    crypto: Crypto,
    /// The asynchronous operation nobody has collected yet. The security
    /// engine stays taken until then: every call that uses it answers
    /// [`STATUS_BUSY`] (exosphere `LockSecurityEngineAndInvokeAsync`).
    pending: Option<Pending>,
    /// [`config_item::IS_CHARGER_HI_Z_MODE_ENABLED`]. Nothing here powers
    /// the console off, so the flag is only remembered.
    charger_hi_z_mode: bool,
    /// Whether the monitor did its cold boot ([`cold_boot`]): the machine
    /// runs the boot chain, and sleep goes through SC7 ([`sc7`]).
    boot_chain: bool,
    /// The kernel's `(entry, context)` while the console is in SC7.
    suspended: Option<(u64, u64)>,
}

/// An asynchronous operation's outcome, waiting for its key.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct Pending {
    key: u64,
    result: AsyncResult,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
enum AsyncResult {
    /// `ComputeAes`: the engine wrote its output already.
    Aes,
    /// An RSA operation: the bytes `GetResultData` must ask for exactly,
    /// or the status the operation failed with.
    Data(Result<Vec<u8>, u64>),
}

/// [`State`] as a snapshot holds it.
#[derive(serde::Serialize, serde::Deserialize)]
struct SavedState {
    rng: SplitMix64,
    key_rng: SplitMix64,
    cores_on: [bool; CORE_COUNT],
    carveouts: [Carveout; CARVEOUT_COUNT],
    settings: BootSettings,
    crypto: SavedCrypto,
    pending: Option<Pending>,
    charger_hi_z_mode: bool,
    boot_chain: bool,
    suspended: Option<(u64, u64)>,
}

/// Everything the monitor learned since boot. What it was built with
/// (the boot parameters, the key file) a restored machine is built with
/// again.
impl Snapshot for SecureMonitor {
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        let state = self.state.lock().unwrap();
        encode(&SavedState {
            rng: SplitMix64(state.rng.0),
            key_rng: SplitMix64(state.key_rng.0),
            cores_on: state.cores_on,
            carveouts: state.carveouts,
            settings: state.settings.clone(),
            crypto: state.crypto.save_state(),
            pending: state.pending.clone(),
            charger_hi_z_mode: state.charger_hi_z_mode,
            boot_chain: state.boot_chain,
            suspended: state.suspended,
        })
    }

    fn restore(&self, bytes: &[u8]) -> Result<(), SnapshotError> {
        let saved: SavedState = decode("secure monitor", bytes)?;
        let mut state = self.state.lock().unwrap();
        state.rng = saved.rng;
        state.key_rng = saved.key_rng;
        state.cores_on = saved.cores_on;
        state.carveouts = saved.carveouts;
        state.settings = saved.settings;
        state.crypto.restore_state(saved.crypto);
        state.pending = saved.pending;
        state.charger_hi_z_mode = saved.charger_hi_z_mode;
        state.boot_chain = saved.boot_chain;
        state.suspended = saved.suspended;
        Ok(())
    }
}

impl SecureMonitor {
    /// Monitor state at kernel entry: only core 0 is running. `keys` is the
    /// console's key material; key operations that need a missing key stop
    /// the machine.
    pub fn new(params: &BootParams, keys: KeySet) -> Self {
        let mut key_rng = SplitMix64(!params.rng_seed);
        let mut seal_key = Key::default();
        key_rng.fill(&mut seal_key);
        let mut cores_on = [false; CORE_COUNT];
        cores_on[0] = true;
        Self {
            dram: params.dram,
            params: *params,
            device_id: params.device_id,
            state: Mutex::new(State {
                rng: SplitMix64(params.rng_seed),
                key_rng,
                cores_on,
                carveouts: [Carveout::default(); CARVEOUT_COUNT],
                settings: BootSettings::from_params(params),
                crypto: Crypto::new(keys, seal_key),
                pending: None,
                charger_hi_z_mode: false,
                boot_chain: false,
                suspended: None,
            }),
        }
    }

    /// Serves one SMC of `core`. `x` is `X0`-`X7` at the instruction and is
    /// rewritten with the values the caller resumes with. `bus` is where
    /// the registers the monitor proxies live, `engine` the memory the
    /// security engine reaches.
    pub fn call(
        &self,
        core: usize,
        immediate: u16,
        x: &mut [u64; 8],
        bus: &Bus,
        engine: &DeviceMemory,
    ) -> SmcEffect {
        let id = x[0] as u32;
        let unimplemented = SmcEffect::Unimplemented(Unimplemented::Function { immediate, id });
        if !in_table(immediate, id) {
            return SmcEffect::Abort(Abort::UnknownFunction { immediate, id });
        }
        if immediate == USER_SMC {
            if core != USER_CALL_CORE {
                return SmcEffect::Abort(Abort::UserCallOnCore { core, id });
            }
            let takes_engine = !matches!(
                id,
                user_function::GET_CONFIG
                    | user_function::SET_CONFIG
                    | user_function::GET_RESULT
                    | user_function::GET_RESULT_DATA
            );
            if takes_engine && self.state.lock().unwrap().pending.is_some() {
                x[0] = STATUS_BUSY;
                return SmcEffect::Resume;
            }
            let ram = engine.ram();
            return match id {
                user_function::GET_CONFIG => self.get_config(x),
                user_function::SET_CONFIG => self.set_config(x),
                user_function::GENERATE_RANDOM_BYTES => self.generate_random_bytes(x),
                user_function::GENERATE_AES_KEK => self.key_operation(x, |crypto, x| {
                    let access_key = crypto.generate_aes_kek(&key_at(x, 1), x[3], x[4] as u32)?;
                    set_key_at(x, 1, &access_key);
                    Ok(())
                }),
                user_function::LOAD_AES_KEY => self.key_operation(x, |crypto, x| {
                    crypto.load_aes_key(x[1], &key_at(x, 2), &key_at(x, 4))
                }),
                user_function::COMPUTE_AES => self.compute_aes(x, bus, engine),
                user_function::COMPUTE_CMAC => {
                    let (address, len) = (x[2], x[3]);
                    let mut data = vec![0; len.min(MAX_CMAC_BYTES) as usize];
                    if len > MAX_CMAC_BYTES
                        || self
                            .user_page(ram, address)
                            .read(address, &mut data)
                            .is_none()
                    {
                        x[0] = STATUS_INVALID_ARGUMENT;
                        return SmcEffect::Resume;
                    }
                    self.key_operation(x, |crypto, x| {
                        let mac = crypto.compute_cmac(x[1], &data)?;
                        set_key_at(x, 1, &mac);
                        Ok(())
                    })
                }
                user_function::DECRYPT_DEVICE_UNIQUE_DATA => {
                    self.decrypt_device_unique_data(x, ram)
                }
                user_function::REENCRYPT_DEVICE_UNIQUE_DATA => {
                    self.reencrypt_device_unique_data(x, ram)
                }
                user_function::GENERATE_SPECIFIC_AES_KEY => self.key_operation(x, |crypto, x| {
                    let key = crypto.generate_specific_aes_key(&key_at(x, 1), x[3], x[4])?;
                    set_key_at(x, 1, &key);
                    Ok(())
                }),
                user_function::PREPARE_ES_COMMON_TITLE_KEY => self.key_operation(x, |crypto, x| {
                    let access_key = crypto.prepare_es_common_title_key(&key_at(x, 1), x[3])?;
                    set_key_at(x, 1, &access_key);
                    Ok(())
                }),
                user_function::LOAD_PREPARED_AES_KEY => self.key_operation(x, |crypto, x| {
                    crypto.load_prepared_aes_key(x[1], &key_at(x, 2))
                }),
                user_function::MODULAR_EXPONENTIATE
                | user_function::MODULAR_EXPONENTIATE_BY_STORAGE_KEY
                | user_function::PREPARE_ES_DEVICE_UNIQUE_KEY => {
                    self.rsa_operation(id, x, bus, ram)
                }
                user_function::GET_RESULT => self.get_result(x, None),
                user_function::GET_RESULT_DATA => self.get_result(x, Some(ram)),
                _ => unimplemented,
            };
        }
        // These take the state lock themselves.
        match id {
            function::GET_CONFIG => return self.get_config(x),
            function::SET_CONFIG => return self.set_config(x),
            _ => {}
        }
        let mut state = self.state.lock().unwrap();
        match id {
            function::GENERATE_RANDOM_BYTES => {
                drop(state);
                return self.generate_random_bytes(x);
            }
            function::READ_WRITE_REGISTER => {
                let (address, mask, value) = (x[1], x[2] as u32, x[3] as u32);
                if address % 4 != 0 || !register_allowed(address) {
                    x[0] = if address % 4 == 0 && DECOY_REGISTERS.contains(&address) {
                        x[1] = 0;
                        STATUS_SUCCESS
                    } else {
                        STATUS_INVALID_ARGUMENT
                    };
                    return SmcEffect::Resume;
                }
                let failed = |is_write| {
                    SmcEffect::Unimplemented(Unimplemented::Register { address, is_write })
                };
                // A full mask replaces the register, so nothing is read.
                let previous = if mask == u32::MAX {
                    0
                } else {
                    match bus.read(0, address, 4) {
                        Access::Done(previous) => previous as u32,
                        _ => return failed(false),
                    }
                };
                if mask != 0 {
                    let merged = (previous & !mask) | (value & mask);
                    if bus.write(0, address, 4, merged.into()) != Access::Done(()) {
                        return failed(true);
                    }
                }
                x[0] = STATUS_SUCCESS;
                x[1] = previous.into();
            }
            function::CONFIGURE_CARVEOUT => {
                let (address, size) = (x[2], x[3]);
                let aligned = address % CARVEOUT_GRANULE == 0 && size % CARVEOUT_GRANULE == 0;
                let index = usize::try_from(x[1]).ok().filter(|&index| {
                    index < CARVEOUT_COUNT && aligned && size <= CARVEOUT_SIZE_MAX
                });
                let Some(index) = index else {
                    x[0] = STATUS_INVALID_ARGUMENT;
                    return SmcEffect::Resume;
                };
                state.carveouts[index] = Carveout { address, size };
                engine.mc().set_kernel_carveout(index, address, size);
                x[0] = STATUS_SUCCESS;
            }
            function::CPU_ON => {
                let (entry, context) = (x[2], x[3]);
                let Some((core, on)) = usize::try_from(x[1])
                    .ok()
                    .and_then(|core| Some((core, state.cores_on.get_mut(core)?)))
                else {
                    x[0] = PSCI_INVALID_PARAMETERS;
                    return SmcEffect::Resume;
                };
                if *on {
                    x[0] = PSCI_ALREADY_ON;
                    return SmcEffect::Resume;
                }
                if !switch_core(bus, core, true) {
                    return unimplemented;
                }
                *on = true;
                x[0] = PSCI_SUCCESS;
                return SmcEffect::StartCore {
                    core,
                    entry,
                    context,
                };
            }
            function::CPU_OFF => {
                if !switch_core(bus, core, false) {
                    return unimplemented;
                }
                state.cores_on[core] = false;
                return SmcEffect::PowerOff;
            }
            function::CPU_SUSPEND => {
                // Suspending takes the security engine (it saves its state).
                if state.pending.is_some() {
                    x[0] = STATUS_BUSY;
                    return SmcEffect::Resume;
                }
                if x[1] as u32 & POWER_STATE_TYPE_AND_ID != POWER_STATE_SLEEP {
                    x[0] = PSCI_DENIED;
                    return SmcEffect::Resume;
                }
                let others = state.cores_on.iter().enumerate();
                if others.filter(|&(other, &on)| on && other != core).count() != 0 {
                    return SmcEffect::Abort(Abort::SleepWithCoresOn);
                }
                let port = sc7::Port {
                    bus,
                    ram: engine.ram(),
                };
                if state.boot_chain {
                    if state.charger_hi_z_mode {
                        return SmcEffect::Abort(Abort::SuspendNotModelled(
                            "the charger's high-impedance mode over I2C1",
                        ));
                    }
                    if let Err(fail) = sc7::prepare(&port) {
                        return sc7_effect(fail);
                    }
                }
                for (register, mask, required, what) in SLEEP_REQUIREMENTS {
                    let Access::Done(value) = bus.read(0, register, 4) else {
                        let register = Unimplemented::Register {
                            address: register,
                            is_write: false,
                        };
                        return SmcEffect::Unimplemented(register);
                    };
                    let value = value as u32;
                    if value & mask != required {
                        return SmcEffect::Abort(Abort::SocNotReadyForSleep { what, value });
                    }
                }
                if state.boot_chain {
                    match sc7::bpmp_parked(&port) {
                        Ok(true) => {}
                        Ok(false) => {
                            return SmcEffect::Abort(Abort::SocNotReadyForSleep {
                                what: "the BPMP parked by the flow controller",
                                value: port.read(cold_boot::HALT_COP_EVENTS).unwrap_or(0),
                            });
                        }
                        Err(fail) => return sc7_effect(fail),
                    }
                    if let Err(fail) = sc7::suspend(&port) {
                        return sc7_effect(fail);
                    }
                    state.suspended = Some((x[2], x[3]));
                    state.cores_on[core] = false;
                    return SmcEffect::SystemSuspend;
                }
                return SmcEffect::Sleep {
                    entry: x[2],
                    context: x[3],
                };
            }
            function::PANIC => return SmcEffect::Panic { code: x[1] as u32 },
            _ => return unimplemented,
        }
        SmcEffect::Resume
    }

    /// Carveouts as last configured by the kernel.
    pub fn carveouts(&self) -> [Carveout; CARVEOUT_COUNT] {
        self.state.lock().unwrap().carveouts
    }

    /// SetConfig, from userland: item in `W1`, value in `X3`.
    fn set_config(&self, x: &mut [u64; 8]) -> SmcEffect {
        x[0] = match x[1] as u32 {
            config_item::IS_CHARGER_HI_Z_MODE_ENABLED => {
                self.state.lock().unwrap().charger_hi_z_mode = x[3] != 0;
                STATUS_SUCCESS
            }
            _ => STATUS_INVALID_ARGUMENT,
        };
        SmcEffect::Resume
    }

    /// GetConfig, through either gate: item in `W1`, value out in `X1`-`X4`.
    fn get_config(&self, x: &mut [u64; 8]) -> SmcEffect {
        let item = x[1] as u32;
        let settings = self.state.lock().unwrap().settings.clone();
        let value = match item {
            config_item::DISABLE_PROGRAM_VERIFICATION => {
                settings.program_verification_disabled.into()
            }
            config_item::DRAM_ID => match self.dram.dram_id() {
                Some(id) => id,
                None => return SmcEffect::Unimplemented(Unimplemented::ConfigItem(item)),
            },
            config_item::SECURITY_ENGINE_INTERRUPT_NUMBER => SECURITY_ENGINE_INTERRUPT,
            config_item::FUSE_VERSION => FUSE_VERSION,
            config_item::HARDWARE_TYPE => HARDWARE_TYPE_ICOSA,
            config_item::HARDWARE_STATE => HARDWARE_STATE_PRODUCTION,
            config_item::IS_RECOVERY_BOOT => settings.recovery_package2_hash.is_some().into(),
            config_item::PACKAGE2_HASH => match settings.recovery_package2_hash {
                Some(hash) => {
                    x[0] = STATUS_SUCCESS;
                    for (register, word) in x[1..5].iter_mut().zip(hash.as_chunks::<8>().0) {
                        *register = u64::from_le_bytes(*word);
                    }
                    return SmcEffect::Resume;
                }
                None => {
                    x[0] = STATUS_INVALID_ARGUMENT;
                    return SmcEffect::Resume;
                }
            },
            config_item::DEVICE_ID => self.device_id,
            config_item::RETAIL_INTERACTIVE_DISPLAY_STATE => 0,
            config_item::REGULATOR_TYPE => 0,
            config_item::DEVICE_UNIQUE_KEY_GENERATION => 0,
            config_item::UNKNOWN_18 => settings.unknown_18,
            config_item::IS_DEVELOPMENT_FUNCTION_ENABLED => {
                settings.development_function_enabled.into()
            }
            config_item::MEMORY_MODE => settings.memory_mode,
            config_item::KERNEL_CONFIGURATION => settings.kernel_configuration,
            config_item::IS_CHARGER_HI_Z_MODE_ENABLED => {
                self.state.lock().unwrap().charger_hi_z_mode.into()
            }
            // Unknown items.
            _ => {
                x[0] = STATUS_INVALID_ARGUMENT;
                return SmcEffect::Resume;
            }
        };
        x[0] = STATUS_SUCCESS;
        x[1..5].copy_from_slice(&[value, 0, 0, 0]);
        SmcEffect::Resume
    }

    /// Runs a key ladder operation and turns its outcome into a status.
    fn key_operation(
        &self,
        x: &mut [u64; 8],
        operation: impl FnOnce(&mut Crypto, &mut [u64; 8]) -> Result<(), CryptoError>,
    ) -> SmcEffect {
        x[0] = match operation(&mut self.state.lock().unwrap().crypto, x) {
            Ok(()) => STATUS_SUCCESS,
            Err(CryptoError::InvalidArgument) => STATUS_INVALID_ARGUMENT,
            Err(CryptoError::Unimplemented(what)) => {
                return SmcEffect::Unimplemented(Unimplemented::Crypto(what));
            }
        };
        SmcEffect::Resume
    }

    /// ComputeAes. The engine finishes before the call returns; what is
    /// asynchronous for the caller is only how it learns of that.
    fn compute_aes(&self, x: &mut [u64; 8], bus: &Bus, memory: &DeviceMemory) -> SmcEffect {
        let unimplemented = |what| SmcEffect::Unimplemented(Unimplemented::Crypto(what));
        let option = x[1] as u32;
        let (slot, source, destination, size) = (option & 7, x[4] as u32, x[5] as u32, x[6] as u32);
        let lists = [source, destination].map(u64::from);
        if slot as usize >= crypto::USER_SLOT_COUNT
            || !size.is_multiple_of(16)
            || !lists.iter().all(|list| ENGINE_LISTS.contains(list))
        {
            x[0] = STATUS_INVALID_ARGUMENT;
            return SmcEffect::Resume;
        }
        let cipher = match option >> 4 & 3 {
            0 => CipherMode::CbcEncrypt,
            1 => CipherMode::CbcDecrypt,
            2 => CipherMode::Ctr,
            // CMAC: only through ComputeCmac.
            _ => {
                x[0] = STATUS_NOT_SUPPORTED;
                return SmcEffect::Resume;
            }
        };
        let size = u64::from(size);
        if size > MAX_ENGINE_BYTES {
            return unimplemented("an AES operation on more than 1 MiB");
        }
        let buffers =
            engine_buffer(memory, lists[0], size).zip(engine_buffer(memory, lists[1], size));
        let mut data = vec![0; size as usize];
        let Some((_, destination)) =
            buffers.filter(|&(source, _)| memory.read(source, &mut data).is_some())
        else {
            return unimplemented(
                "an AES operation whose buffer lists are not one mapped buffer each",
            );
        };
        let mut state = self.state.lock().unwrap();
        if let Err(error) = state
            .crypto
            .compute_aes(slot.into(), cipher, &key_at(x, 2), &mut data)
        {
            let CryptoError::Unimplemented(what) = error else {
                unreachable!("the slot and the size were validated above");
            };
            return unimplemented(what);
        }
        if memory.write(destination, &data).is_none() {
            return unimplemented("an AES operation whose destination is not mapped");
        }
        begin_async(&mut state, bus, x, AsyncResult::Aes)
    }

    /// The RSA calls, asynchronous like ComputeAes: ModularExponentiate
    /// with the caller's key, ModularExponentiateByStorageKey with an
    /// imported one, and PrepareEsDeviceUniqueKey, which also decodes and
    /// seals a title key. The message, the modulus and the exponent must
    /// share the message's page.
    fn rsa_operation(&self, id: u32, x: &mut [u64; 8], bus: &Bus, ram: &GuestRam) -> SmcEffect {
        let invalid = |x: &mut [u64; 8], status| {
            x[0] = status;
            SmcEffect::Resume
        };
        let message_at = x[1];
        let page = self.user_page(ram, message_at);
        let (mut message, mut modulus) = ([0; RSA_SIZE], [0; RSA_SIZE]);
        let mut read_operands = |modulus_at: u64| {
            page.read(message_at, &mut message)
                .and_then(|()| page.read(modulus_at, &mut modulus))
        };
        let mut state = self.state.lock().unwrap();
        let outcome = match id {
            user_function::MODULAR_EXPONENTIATE => {
                let exponent_len = x[4];
                if !exponent_len.is_multiple_of(4) || exponent_len > RSA_SIZE as u64 {
                    return invalid(x, STATUS_INVALID_ARGUMENT);
                }
                let mut exponent = vec![0; exponent_len as usize];
                if read_operands(x[3])
                    .and_then(|()| page.read(x[2], &mut exponent))
                    .is_none()
                {
                    return invalid(x, STATUS_INVALID_ARGUMENT);
                }
                Crypto::modular_exponentiate(&message, &exponent, &modulus)
                    .map(|block| block.to_vec())
            }
            user_function::MODULAR_EXPONENTIATE_BY_STORAGE_KEY => {
                // Exosphere checks bits 8 and up as reserved here, the
                // layout of PrepareEsDeviceUniqueKey's option word.
                let option = x[3] as u32;
                let mode = option & 3;
                if option >> 8 != 0
                    || mode >= crypto::STORAGE_KEY_MODES
                    || read_operands(x[2]).is_none()
                {
                    return invalid(x, STATUS_INVALID_ARGUMENT);
                }
                state
                    .crypto
                    .exponentiate_by_storage_key(mode, &message, &modulus)
                    .map(|block| block.to_vec())
            }
            _ => {
                let option = x[7] as u32;
                let (generation, kind) = (u64::from(option & 0x3F), option >> 6 & 3);
                if option >> 8 != 0
                    || !state.crypto.valid_generation(generation)
                    || kind >= crypto::ES_COMMON_KEY_TYPES
                    || read_operands(x[2]).is_none()
                {
                    return invalid(x, STATUS_INVALID_ARGUMENT);
                }
                if !state.crypto.has_es_device_key() {
                    return invalid(x, STATUS_NOT_INITIALIZED);
                }
                let mut label_digest = [0; 32];
                for (bytes, word) in label_digest.chunks_mut(8).zip(&x[3..7]) {
                    bytes.copy_from_slice(&word.to_le_bytes());
                }
                state
                    .crypto
                    .prepare_es_device_unique_key(&message, &label_digest, generation, kind)
                    .map(|key| key.to_vec())
            }
        };
        let result = match outcome {
            Ok(data) => Ok(data),
            // What the operation itself refuses shows in its result.
            Err(CryptoError::InvalidArgument) => Err(STATUS_INVALID_ARGUMENT),
            Err(CryptoError::Unimplemented(what)) => {
                return SmcEffect::Unimplemented(Unimplemented::Crypto(what));
            }
        };
        begin_async(&mut state, bus, x, AsyncResult::Data(result))
    }

    /// GetResult (`ram` is `None`) and GetResultData: collects the pending
    /// operation by its key, which ends it and frees the engine. The
    /// result's status comes back in `X1`; GetResultData also copies the
    /// result to the caller, who must ask for exactly its size, so the
    /// RSA calls are collected only through it. \[assumption\] When the
    /// operation failed, exosphere copies its work buffer regardless;
    /// here the caller's buffer keeps its contents. \[inference\] Exosphere
    /// leaves the engine taken when an RSA result is refused (wrong size,
    /// failed operation), so every later engine call would answer busy;
    /// here the engine is freed whenever a result is collected.
    fn get_result(&self, x: &mut [u64; 8], ram: Option<&GuestRam>) -> SmcEffect {
        let (address, size) = (x[2], x[3]);
        let mut state = self.state.lock().unwrap();
        x[0] = match &state.pending {
            None => STATUS_NO_ASYNC_OPERATION,
            Some(pending) if pending.key != x[1] => STATUS_INVALID_ASYNC_OPERATION,
            Some(_) if ram.is_some() && size > MAX_RESULT_BYTES => STATUS_INVALID_ARGUMENT,
            Some(_) => {
                let wanted = if ram.is_some() { size as usize } else { 0 };
                let pending = state.pending.take().expect("matched above");
                let (status, data) = match pending.result {
                    AsyncResult::Aes => (STATUS_SUCCESS, None),
                    AsyncResult::Data(Err(status)) => (status, None),
                    AsyncResult::Data(Ok(data)) if data.len() == wanted => {
                        (STATUS_SUCCESS, Some(data))
                    }
                    AsyncResult::Data(Ok(_)) => (STATUS_INVALID_ARGUMENT, None),
                };
                x[1] = status;
                match ram.map(|ram| self.user_page(ram, address)) {
                    None => STATUS_SUCCESS,
                    Some(page) => {
                        let copied = match data {
                            Some(data) => page.write(address, &data),
                            None => page.holds(address, wanted).then_some(()),
                        };
                        if copied.is_some() {
                            STATUS_SUCCESS
                        } else {
                            STATUS_INVALID_ARGUMENT
                        }
                    }
                }
            }
        };
        SmcEffect::Resume
    }

    /// DecryptDeviceUniqueData, synchronous like the monitor's: mode in
    /// the option word's low three bits, the rest reserved. Mode 0 hands
    /// the plain data back in place; the others keep the key inside.
    fn decrypt_device_unique_data(&self, x: &mut [u64; 8], ram: &GuestRam) -> SmcEffect {
        let (option, address, len) = (x[3] as u32, x[4], x[5]);
        let page = self.user_page(ram, address);
        let mut data = vec![0; len.min(crypto::DEVICE_UNIQUE_DATA_MAX as u64) as usize];
        if option >> 3 != 0
            || len > crypto::DEVICE_UNIQUE_DATA_MAX as u64
            || page.read(address, &mut data).is_none()
        {
            x[0] = STATUS_INVALID_ARGUMENT;
            return SmcEffect::Resume;
        }
        let mode = option & 7;
        let device_id = self.device_id;
        let (access_key, key_source) = (key_at(x, 1), key_at(x, 6));
        let effect = self.key_operation(x, |crypto, _| {
            crypto.decrypt_device_unique_data(mode, &access_key, &key_source, &mut data, device_id)
        });
        if x[0] == STATUS_SUCCESS && mode == 0 {
            page.write(address, &data)
                .expect("the page the data came from");
        }
        effect
    }

    /// ReencryptDeviceUniqueData: the data at `X4` (`X5` bytes) opened with
    /// the access key and key source at the addresses in `X1` and `X6`,
    /// wrapped again for the mode in `X3` with those at `X2` and `X7`, all
    /// within the data's page.
    fn reencrypt_device_unique_data(&self, x: &mut [u64; 8], ram: &GuestRam) -> SmcEffect {
        let (option, address, len) = (x[3] as u32, x[4], x[5]);
        let page = self.user_page(ram, address);
        let mut data = vec![0; len.min(crypto::DEVICE_UNIQUE_DATA_MAX as u64) as usize];
        let mut keys = [Key::default(); 4];
        let read = page.read(address, &mut data).and_then(|()| {
            [x[1], x[2], x[6], x[7]]
                .iter()
                .zip(&mut keys)
                .try_for_each(|(&at, key)| page.read(at, key))
        });
        if option >> 3 != 0 || len > crypto::DEVICE_UNIQUE_DATA_MAX as u64 || read.is_none() {
            x[0] = STATUS_INVALID_ARGUMENT;
            return SmcEffect::Resume;
        }
        let mut iv = Key::default();
        self.state.lock().unwrap().key_rng.fill(&mut iv);
        let [open_access, seal_access, open_source, seal_source] = keys;
        let device_id = self.device_id;
        let effect = self.key_operation(x, |crypto, _| {
            crypto.reencrypt_device_unique_data(
                option & 7,
                &open_access,
                &open_source,
                &seal_access,
                &seal_source,
                &mut data,
                device_id,
                &iv,
            )
        });
        if x[0] == STATUS_SUCCESS {
            page.write(address, &data)
                .expect("the page the data came from");
        }
        effect
    }

    /// The page of the caller's memory the monitor maps for a call
    /// (exosphere `UserPageMapper`): the one holding `address`, which must
    /// be DRAM. Every buffer of the call must lie within it.
    fn user_page<'a>(&self, ram: &'a GuestRam, address: u64) -> UserPage<'a> {
        let in_dram = address.wrapping_sub(DRAM_BASE) < self.dram.bytes();
        UserPage {
            ram,
            page: in_dram.then_some(address & !(PAGE_SIZE - 1)),
        }
    }

    /// GenerateRandomBytes, through either gate: byte count in `X1`, bytes
    /// out packed into `X1`-`X7`, the rest of which keep their values.
    fn generate_random_bytes(&self, x: &mut [u64; 8]) -> SmcEffect {
        let count = x[1];
        if count > MAX_RANDOM_BYTES {
            x[0] = STATUS_INVALID_ARGUMENT;
            return SmcEffect::Resume;
        }
        let mut bytes = [0u8; MAX_RANDOM_BYTES as usize];
        for (word, chunk) in x[1..].iter().zip(bytes.as_chunks_mut::<8>().0) {
            *chunk = word.to_le_bytes();
        }
        self.state
            .lock()
            .unwrap()
            .rng
            .fill(&mut bytes[..count as usize]);
        for (word, chunk) in x[1..].iter_mut().zip(bytes.as_chunks::<8>().0) {
            *word = u64::from_le_bytes(*chunk);
        }
        x[0] = STATUS_SUCCESS;
        SmcEffect::Resume
    }
}

/// One page of the caller's memory, as the monitor maps it for a call;
/// `page` is `None` when the address it was asked for is not DRAM.
struct UserPage<'a> {
    ram: &'a GuestRam,
    page: Option<u64>,
}

impl UserPage<'_> {
    /// Whether `len` bytes at `address` lie within the page.
    fn holds(&self, address: u64, len: usize) -> bool {
        let page = |at: u64| at & !(PAGE_SIZE - 1);
        self.page.is_some_and(|base| {
            page(address) == base
                && (len == 0
                    || address
                        .checked_add(len as u64 - 1)
                        .is_some_and(|last| page(last) == base))
        })
    }

    fn read(&self, address: u64, buffer: &mut [u8]) -> Option<()> {
        self.holds(address, buffer.len()).then_some(())?;
        self.ram.read(address, buffer)
    }

    fn write(&self, address: u64, data: &[u8]) -> Option<()> {
        self.holds(address, data.len()).then_some(())?;
        self.ram.write(address, data)
    }
}

/// Starts an asynchronous operation whose outcome is `result`: signals its
/// completion at once and hands the caller the key to collect it with.
fn begin_async(state: &mut State, bus: &Bus, x: &mut [u64; 8], result: AsyncResult) -> SmcEffect {
    let pending = 1 << (SECURITY_ENGINE_INTERRUPT % 32);
    if bus.write(0, SET_PENDING_REGISTER, 4, pending) != Access::Done(()) {
        let register = Unimplemented::Register {
            address: SET_PENDING_REGISTER,
            is_write: true,
        };
        return SmcEffect::Unimplemented(register);
    }
    let key = state.key_rng.next();
    state.pending = Some(Pending { key, result });
    x[0] = STATUS_SUCCESS;
    x[1] = key;
    SmcEffect::Resume
}

/// Name of an SMC function for traces.
pub fn function_name(immediate: u16, id: u32) -> &'static str {
    match (immediate, id) {
        (KERNEL_SMC, function::CPU_ON) => "CpuOn",
        (KERNEL_SMC, function::CPU_OFF) => "CpuOff",
        (KERNEL_SMC, function::CPU_SUSPEND) => "CpuSuspend",
        (KERNEL_SMC, function::GET_CONFIG) => "GetConfig",
        (KERNEL_SMC, function::GENERATE_RANDOM_BYTES) => "GenerateRandomBytes",
        (KERNEL_SMC, function::PANIC) => "Panic",
        (KERNEL_SMC, function::CONFIGURE_CARVEOUT) => "ConfigureCarveout",
        (KERNEL_SMC, function::READ_WRITE_REGISTER) => "ReadWriteRegister",
        (USER_SMC, user_function::GET_CONFIG) => "GetConfig (user)",
        (USER_SMC, user_function::SET_CONFIG) => "SetConfig",
        (USER_SMC, user_function::GENERATE_RANDOM_BYTES) => "GenerateRandomBytes (user)",
        (USER_SMC, user_function::GENERATE_AES_KEK) => "GenerateAesKek",
        (USER_SMC, user_function::LOAD_AES_KEY) => "LoadAesKey",
        (USER_SMC, user_function::COMPUTE_AES) => "ComputeAes",
        (USER_SMC, user_function::COMPUTE_CMAC) => "ComputeCmac",
        (USER_SMC, user_function::GENERATE_SPECIFIC_AES_KEY) => "GenerateSpecificAesKey",
        (USER_SMC, user_function::DECRYPT_DEVICE_UNIQUE_DATA) => "DecryptDeviceUniqueData",
        (USER_SMC, user_function::GET_RESULT) => "GetResult",
        (USER_SMC, user_function::GET_RESULT_DATA) => "GetResultData",
        (USER_SMC, user_function::MODULAR_EXPONENTIATE) => "ModularExponentiate",
        (USER_SMC, user_function::REENCRYPT_DEVICE_UNIQUE_DATA) => "ReencryptDeviceUniqueData",
        (USER_SMC, user_function::MODULAR_EXPONENTIATE_BY_STORAGE_KEY) => {
            "ModularExponentiateByStorageKey"
        }
        (USER_SMC, user_function::PREPARE_ES_DEVICE_UNIQUE_KEY) => "PrepareEsDeviceUniqueKey",
        (USER_SMC, user_function::LOAD_PREPARED_AES_KEY) => "LoadPreparedAesKey",
        (USER_SMC, user_function::PREPARE_ES_COMMON_TITLE_KEY) => "PrepareEsCommonTitleKey",
        _ => "?",
    }
}

/// What the machine is told when the SC7 preparation fails.
fn sc7_effect(fail: sc7::Sc7Fail) -> SmcEffect {
    match fail {
        sc7::Sc7Fail::Register { address, is_write } => {
            SmcEffect::Unimplemented(Unimplemented::Register { address, is_write })
        }
        sc7::Sc7Fail::Panic(what) => SmcEffect::Abort(Abort::SuspendPanic(what)),
        sc7::Sc7Fail::Stopped => SmcEffect::Abort(Abort::SuspendPanic("the machine stopped")),
    }
}

/// Switches the power partition of `core` on or off, as the monitor does
/// around starting and stopping it. `false` if the PMC refused.
fn switch_core(bus: &Bus, core: usize, on: bool) -> bool {
    let partition = CORE_PARTITIONS[core];
    let Access::Done(status) = bus.read(0, PMC_PWRGATE_STATUS, 4) else {
        return false;
    };
    (status >> partition & 1 != 0) == on
        || bus.write(
            0,
            PMC_PWRGATE_TOGGLE,
            4,
            PWRGATE_TOGGLE_START | u64::from(partition),
        ) == Access::Done(())
}

/// The 128-bit value `x[index]`, `x[index + 1]` carry, in memory order.
fn key_at(x: &[u64; 8], index: usize) -> Key {
    let mut key = Key::default();
    key[..8].copy_from_slice(&x[index].to_le_bytes());
    key[8..].copy_from_slice(&x[index + 1].to_le_bytes());
    key
}

fn set_key_at(x: &mut [u64; 8], index: usize, key: &Key) {
    let (low, high) = key.split_at(8);
    x[index] = u64::from_le_bytes(low.try_into().unwrap());
    x[index + 1] = u64::from_le_bytes(high.try_into().unwrap());
}

/// The one buffer a security engine buffer list at `list` names, if it
/// holds `size` bytes: a count of further entries, which must be zero, then
/// address and length.
fn engine_buffer(memory: &DeviceMemory, list: u64, size: u64) -> Option<u64> {
    let mut raw = [0; 12];
    memory.read(list, &mut raw)?;
    let word = |index: usize| u32::from_le_bytes(raw[index * 4..][..4].try_into().unwrap());
    (word(0) == 0 && u64::from(word(2)) >= size).then(|| word(1).into())
}

/// SplitMix64. Hardware uses a true random source; a seeded generator keeps
/// the kernel's randomized layout reproducible from run to run.
#[derive(serde::Serialize, serde::Deserialize)]
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn fill(&mut self, bytes: &mut [u8]) {
        for chunk in bytes.chunks_mut(8) {
            chunk.copy_from_slice(&self.next().to_le_bytes()[..chunk.len()]);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{
        boot::KernelConfiguration,
        memory::GuestRam,
        soc::{mc, regs::RegisterFile},
    };

    const MC_EMEM_CFG: u64 = mc::BASE + 0x50;
    const MEMORY_BASE: u64 = 0x8000_0000;

    /// A monitor with the register blocks it reaches and a little DRAM.
    struct Console {
        monitor: SecureMonitor,
        bus: Bus,
        memory: Arc<GuestRam>,
        engine: DeviceMemory,
    }

    impl Console {
        /// A call through `immediate`'s gate on the core such calls come
        /// from: userland's on core 3, the kernel's on core 0.
        fn call(&self, immediate: u16, x: &mut [u64; 8]) -> SmcEffect {
            let core = if immediate == USER_SMC {
                USER_CALL_CORE
            } else {
                0
            };
            self.call_on(core, immediate, x)
        }

        fn call_on(&self, core: usize, immediate: u16, x: &mut [u64; 8]) -> SmcEffect {
            self.monitor
                .call(core, immediate, x, &self.bus, &self.engine)
        }

        fn carveouts(&self) -> [Carveout; CARVEOUT_COUNT] {
            self.monitor.carveouts()
        }
    }

    fn console_with_keys(params: &BootParams, keys: KeySet) -> Console {
        let memory_controller = Arc::new(mc::Mc::new(params.dram));
        let mut bus = Bus::default();
        bus.map(
            "PMC",
            0x7000_E400,
            0xC00,
            Box::new(crate::soc::pmc::Pmc::new()),
        );
        bus.map(
            "MC",
            mc::BASE,
            mc::SIZE,
            Box::new(Arc::clone(&memory_controller)),
        );
        bus.map(
            "GICD",
            0x5004_1000,
            0x1000,
            Box::new(RegisterFile::zeroed([])),
        );
        bus.map(
            "CAR",
            CAR_BASE,
            crate::soc::car::SIZE,
            Box::new(crate::soc::car::Car::new()),
        );
        let memory = Arc::new(GuestRam::new(MEMORY_BASE, 0x1_0000).unwrap());
        let engine = DeviceMemory::new(&memory_controller, &memory, mc::SMMU_SE_ASID);
        Console {
            monitor: SecureMonitor::new(params, keys),
            bus,
            memory,
            engine,
        }
    }

    fn console(params: &BootParams) -> Console {
        console_with_keys(params, KeySet::default())
    }

    fn monitor(dram: DramSize) -> Console {
        console(&BootParams {
            dram,
            rng_seed: 7,
            ..BootParams::default()
        })
    }

    #[test]
    fn compute_aes_works_on_the_engine_buffers_and_is_collected_once_by_its_key() {
        let keys = KeySet::parse(
            "aes_kek_generation_source = 000102030405060708090a0b0c0d0e0f\n\
             master_key_00 = 101112131415161718191a1b1c1d1e1f\n",
        )
        .unwrap();
        let console = console_with_keys(&BootParams::default(), keys);
        let user = |arguments: [u64; 7]| {
            let mut x = [0; 8];
            x[..7].copy_from_slice(&arguments);
            assert_eq!(console.call(USER_SMC, &mut x), SmcEffect::Resume);
            x
        };
        let kek = user([
            user_function::GENERATE_AES_KEK.into(),
            0x33,
            0x44,
            1,
            0,
            0,
            0,
        ]);
        assert_eq!(kek[0], STATUS_SUCCESS);
        assert_eq!(
            user([
                user_function::LOAD_AES_KEY.into(),
                3,
                kek[1],
                kek[2],
                0x55,
                0x66,
                0
            ])[0],
            STATUS_SUCCESS
        );

        // The kernel's setup for `spl`: address space 5 for the engine,
        // with device page 0x8003_5000 on a page of DRAM.
        const DEVICE: u64 = 0x8003_5000;
        const PHYSICAL: u64 = MEMORY_BASE + 0x5000;
        let (directory, table) = (MEMORY_BASE + 0x1000, MEMORY_BASE + 0x2000);
        let word = |address: u64, value: u64| {
            console
                .memory
                .write(address, &(value as u32).to_le_bytes())
                .unwrap()
        };
        word(directory + (DEVICE >> 22) * 4, 0xF000_0000 | table >> 12);
        word(
            table + (DEVICE >> 12 & 0x3FF) * 4,
            0xE000_0000 | PHYSICAL >> 12,
        );
        for (register, value) in [
            (0x1C, 5),
            (0x20, 0xE000_0000 | directory >> 12),
            (mc::SMMU_SE_ASID, 0x8000_0005),
        ] {
            let write = [
                function::READ_WRITE_REGISTER.into(),
                mc::BASE + register,
                0xFFFF_FFFF,
                value,
            ];
            assert_eq!(smc(&console, write).1[0], STATUS_SUCCESS);
        }

        // Source list at +0, destination list at +0xC, as `spl` lays them
        // out; two blocks of data at +0x100, results to +0x200.
        let list = |at: u64, buffer: u64| {
            word(PHYSICAL + at, 0);
            word(PHYSICAL + at + 4, DEVICE + buffer);
            word(PHYSICAL + at + 8, 0x20);
        };
        list(0, 0x100);
        list(0xC, 0x200);
        let plaintext: Vec<u8> = (0..0x20).collect();
        console.memory.write(PHYSICAL + 0x100, &plaintext).unwrap();
        let compute = |mode: u64, from: u64, to: u64| {
            user([
                user_function::COMPUTE_AES.into(),
                mode,
                0xAAAA,
                0xBBBB,
                DEVICE + from,
                DEVICE + to,
                0x20,
            ])
        };

        // CBC encryption with slot 3.
        let encrypt = compute(0x03, 0, 0xC);
        assert_eq!(encrypt[0], STATUS_SUCCESS);
        assert_eq!(
            console.bus.read(0, SET_PENDING_REGISTER, 4),
            Access::Done(1 << 12)
        );
        let mut ciphertext = [0; 0x20];
        console
            .memory
            .read(PHYSICAL + 0x200, &mut ciphertext)
            .unwrap();
        assert_ne!(ciphertext[..], plaintext[..]);
        assert_ne!(ciphertext[..0x10], ciphertext[0x10..]);

        // The engine is taken until the result is collected, by its key.
        assert_eq!(compute(0x13, 0xC, 0)[0], STATUS_BUSY);
        let get_result = |key: u64| user([user_function::GET_RESULT.into(), key, 0, 0, 0, 0, 0]);
        assert_eq!(
            get_result(encrypt[1] ^ 1)[0],
            STATUS_INVALID_ASYNC_OPERATION
        );
        assert_eq!(
            get_result(encrypt[1])[..2],
            [STATUS_SUCCESS, STATUS_SUCCESS]
        );
        assert_eq!(get_result(encrypt[1])[0], STATUS_NO_ASYNC_OPERATION);

        // CBC decryption the other way round restores the data.
        console.memory.write(PHYSICAL + 0x100, &[0; 0x20]).unwrap();
        assert_eq!(compute(0x13, 0xC, 0)[0], STATUS_SUCCESS);
        let mut restored = [0; 0x20];
        console
            .memory
            .read(PHYSICAL + 0x100, &mut restored)
            .unwrap();
        assert_eq!(restored[..], plaintext[..]);
    }

    #[test]
    fn device_unique_data_is_decrypted_in_place_for_this_console_only() {
        let text = "aes_kek_generation_source = 000102030405060708090a0b0c0d0e0f\n\
                    master_key_00 = 101112131415161718191a1b1c1d1e1f\n";
        let keys = KeySet::parse(text).unwrap();
        let device_id = 0x0011_2233_4455_6677;
        let params = BootParams {
            device_id,
            ..BootParams::default()
        };
        let console = console_with_keys(&params, keys.clone());
        let user = |arguments: [u64; 8]| {
            let mut x = arguments;
            let effect = console.call(USER_SMC, &mut x);
            (effect, x)
        };
        let (kek_source, key_source) = ([0x33; 16], [0x44; 16]);
        let option = crypto::kek_option(crypto::SEAL_DECRYPT_DEVICE_UNIQUE_DATA);
        let (_, access) = user([
            user_function::GENERATE_AES_KEK.into(),
            0x3333_3333_3333_3333,
            0x3333_3333_3333_3333,
            0,
            option.into(),
            0,
            0,
            0,
        ]);
        assert_eq!(access[0], STATUS_SUCCESS);

        let key =
            crypto::device_unique_data_key(&keys, &kek_source, &key_source, 0, option).unwrap();
        let payload: Vec<u8> = (0..0x100).map(|index| index as u8).collect();
        let sealed = crypto::seal_device_unique_data(&key, &[0x77; 16], &payload, device_id);
        const PHYSICAL: u64 = MEMORY_BASE + 0x3010;
        let decrypt = |data: &[u8], mode: u64| {
            console.memory.write(PHYSICAL, data).unwrap();
            user([
                user_function::DECRYPT_DEVICE_UNIQUE_DATA.into(),
                access[1],
                access[2],
                mode,
                PHYSICAL,
                data.len() as u64,
                0x4444_4444_4444_4444,
                0x4444_4444_4444_4444,
            ])
        };
        let (effect, x) = decrypt(&sealed, 0);
        assert_eq!((effect, x[0]), (SmcEffect::Resume, STATUS_SUCCESS));
        let mut plain = vec![0; 0x100];
        console.memory.read(PHYSICAL, &mut plain).unwrap();
        assert_eq!(plain, payload);

        let mut tampered = sealed.clone();
        tampered[0x20] ^= 0x80;
        assert_eq!(decrypt(&tampered, 0).1[0], STATUS_INVALID_ARGUMENT);
        assert_eq!(decrypt(&sealed, 1 << 3).1[0], STATUS_INVALID_ARGUMENT);
        // Mode 3 imports the TLS key, whose access keys are sealed apart.
        assert_eq!(decrypt(&sealed, 3).1[0], STATUS_INVALID_ARGUMENT);
        // Data must lie within one page of DRAM.
        for address in [MEMORY_BASE + 0x3F80, 0x1000] {
            let mut x = [0; 8];
            x[..6].copy_from_slice(&[
                user_function::DECRYPT_DEVICE_UNIQUE_DATA.into(),
                access[1],
                access[2],
                0,
                address,
                sealed.len() as u64,
            ]);
            assert_eq!(console.call(USER_SMC, &mut x), SmcEffect::Resume);
            assert_eq!(x[0], STATUS_INVALID_ARGUMENT, "{address:#x}");
        }
    }

    #[test]
    fn an_rsa_result_holds_the_engine_until_collected_whole_by_get_result_data() {
        let console = monitor(DramSize::Gb4);
        let user = |arguments: &[u64]| {
            let mut x = [0; 8];
            x[..arguments.len()].copy_from_slice(arguments);
            assert_eq!(console.call(USER_SMC, &mut x), SmcEffect::Resume);
            x
        };
        // 2^3 mod 1000, operands in one page.
        const PAGE: u64 = MEMORY_BASE + 0x4000;
        let block = |value: u64| {
            let mut block = [0; RSA_SIZE];
            block[RSA_SIZE - 8..].copy_from_slice(&value.to_be_bytes());
            block
        };
        console.memory.write(PAGE, &block(2)).unwrap();
        console.memory.write(PAGE + 0x100, &[0, 0, 0, 3]).unwrap();
        console.memory.write(PAGE + 0x200, &block(1000)).unwrap();
        let exponentiate = |modulus_at: u64, exponent_len: u64| {
            user(&[
                user_function::MODULAR_EXPONENTIATE.into(),
                PAGE,
                PAGE + 0x100,
                modulus_at,
                exponent_len,
            ])
        };
        let result_data = |key: u64, address: u64, size: u64| {
            user(&[user_function::GET_RESULT_DATA.into(), key, address, size])
        };
        assert_eq!(exponentiate(PAGE + 0xF80, 4)[0], STATUS_INVALID_ARGUMENT);
        assert_eq!(exponentiate(PAGE + 0x200, 6)[0], STATUS_INVALID_ARGUMENT);

        let started = exponentiate(PAGE + 0x200, 4);
        assert_eq!(started[0], STATUS_SUCCESS);
        // Everything that uses the engine waits, the kernel's sleep too;
        // configuration does not.
        assert_eq!(
            user(&[user_function::GENERATE_AES_KEK.into(), 0, 0, 1])[0],
            STATUS_BUSY
        );
        assert_eq!(exponentiate(PAGE + 0x200, 4)[0], STATUS_BUSY);
        let suspend = [function::CPU_SUSPEND.into(), POWER_STATE_SLEEP.into(), 0, 0];
        assert_eq!(smc(&console, suspend).1[0], STATUS_BUSY);
        assert_eq!(
            user(&[user_function::GET_CONFIG.into(), 1])[0],
            STATUS_SUCCESS
        );
        // Too large a request leaves the operation pending.
        assert_eq!(
            result_data(started[1], PAGE + 0x300, 0x401)[0],
            STATUS_INVALID_ARGUMENT
        );
        let collected = result_data(started[1], PAGE + 0x300, 0x100);
        assert_eq!(collected[..2], [STATUS_SUCCESS, STATUS_SUCCESS]);
        let mut result = [0; RSA_SIZE];
        console.memory.read(PAGE + 0x300, &mut result).unwrap();
        assert_eq!(result, block(8));
        assert_eq!(
            result_data(started[1], PAGE + 0x300, 0x100)[0],
            STATUS_NO_ASYNC_OPERATION
        );

        // Asking for another size ends the operation with nothing copied.
        let started = exponentiate(PAGE + 0x200, 4);
        let wrong = user(&[user_function::GET_RESULT.into(), started[1]]);
        assert_eq!(wrong[..2], [STATUS_SUCCESS, STATUS_INVALID_ARGUMENT]);
        assert_eq!(
            user(&[user_function::GET_RESULT.into(), started[1]])[0],
            STATUS_NO_ASYNC_OPERATION
        );

        // The ES device key was never imported.
        let mut prepare = [0; 8];
        prepare[..3].copy_from_slice(&[
            user_function::PREPARE_ES_DEVICE_UNIQUE_KEY.into(),
            PAGE,
            PAGE + 0x200,
        ]);
        prepare[7] = 1;
        assert_eq!(user(&prepare)[0], STATUS_NOT_INITIALIZED);
    }

    /// Issues a kernel SMC and returns the effect with the result registers.
    fn smc(monitor: &Console, arguments: [u64; 4]) -> (SmcEffect, [u64; 8]) {
        let mut x = [0; 8];
        x[..4].copy_from_slice(&arguments);
        (monitor.call(KERNEL_SMC, &mut x), x)
    }

    #[test]
    fn dram_size_is_consistent_across_every_source_the_kernel_reads() {
        for (dram, megabytes, size_field, memory_mode) in [
            (DramSize::Gb4, 4096, 0, 0x01),
            (DramSize::Gb6, 6144, 1, 0x11),
            (DramSize::Gb8, 8192, 2, 0x21),
        ] {
            let monitor = monitor(dram);
            let (_, emem) = smc(
                &monitor,
                [function::READ_WRITE_REGISTER.into(), MC_EMEM_CFG, 0, 0],
            );
            let (_, config) = smc(
                &monitor,
                [
                    function::GET_CONFIG.into(),
                    config_item::KERNEL_CONFIGURATION.into(),
                    0,
                    0,
                ],
            );
            let (_, mode) = smc(
                &monitor,
                [
                    function::GET_CONFIG.into(),
                    config_item::MEMORY_MODE.into(),
                    0,
                    0,
                ],
            );
            assert_eq!((emem[0], emem[1]), (STATUS_SUCCESS, megabytes));
            assert_eq!(
                (config[0], config[1] >> 16 & 3),
                (STATUS_SUCCESS, size_field)
            );
            assert_eq!((mode[0], mode[1]), (STATUS_SUCCESS, memory_mode));
        }
    }

    #[test]
    fn kernel_configuration_flags_land_on_the_bits_the_kernel_tests() {
        let params = BootParams {
            kernel_configuration: KernelConfiguration {
                increase_thread_resource_limit: true,
                skip_panic_smc: true,
                ..KernelConfiguration::default()
            },
            ..BootParams::default()
        };
        let (_, x) = smc(
            &console(&params),
            [
                function::GET_CONFIG.into(),
                config_item::KERNEL_CONFIGURATION.into(),
                0,
                0,
            ],
        );
        assert_eq!(x[1], 1 << 3 | 1 << 8);
    }

    #[test]
    fn development_functions_are_reported_only_when_enabled() {
        let item = [
            function::GET_CONFIG.into(),
            config_item::IS_DEVELOPMENT_FUNCTION_ENABLED.into(),
            0,
            0,
        ];
        let enabled = BootParams {
            development_function_enabled: true,
            ..BootParams::default()
        };
        assert_eq!(smc(&console(&enabled), item).1[..2], [STATUS_SUCCESS, 1]);
        assert_eq!(
            smc(&monitor(DramSize::Gb4), item).1[..2],
            [STATUS_SUCCESS, 0]
        );
    }

    #[test]
    fn program_verification_is_disabled_only_when_chosen() {
        let item = [
            function::GET_CONFIG.into(),
            config_item::DISABLE_PROGRAM_VERIFICATION.into(),
            0,
            0,
        ];
        let disabled = BootParams {
            program_verification_disabled: true,
            ..BootParams::default()
        };
        assert_eq!(smc(&console(&disabled), item).1[..2], [STATUS_SUCCESS, 1]);
        assert_eq!(
            smc(&monitor(DramSize::Gb4), item).1[..2],
            [STATUS_SUCCESS, 0]
        );
    }

    #[test]
    fn config_items_beyond_the_known_ones_are_invalid_arguments() {
        let monitor = monitor(DramSize::Gb4);
        let get = |item: u64| smc(&monitor, [function::GET_CONFIG.into(), item, 0, 0]).1;
        assert_eq!(get(4)[..2], [STATUS_SUCCESS, 25], "fuse version of 23.5.0");
        assert_eq!(get(18)[..2], [STATUS_SUCCESS, 0]);
        // Boot reason (gone since 4.0.0), package2 hash (recovery boot
        // only), nothing (0, 19), and Atmosphère's items.
        for item in [0, 9, 17, 19, 65000] {
            assert_eq!(get(item)[0], STATUS_INVALID_ARGUMENT, "item {item}");
        }
        // The kernel's SetConfig is the same call as userland's.
        let (effect, x) = smc(&monitor, [function::SET_CONFIG.into(), 13, 0, 1]);
        assert_eq!((effect, x[0]), (SmcEffect::Resume, STATUS_SUCCESS));
        assert_eq!(get(13)[1], 1);
    }

    #[test]
    fn calls_outside_the_tables_or_off_core_three_abort_the_monitor() {
        let monitor = monitor(DramSize::Gb4);
        // Calls in no table make the monitor give up, as do userland's on
        // another core than spl's.
        let (effect, _) = smc(&monitor, [0x8400_0009, 0, 0, 0]);
        let unknown = |immediate, id| SmcEffect::Abort(Abort::UnknownFunction { immediate, id });
        assert_eq!(effect, unknown(1, 0x8400_0009));
        // The kernel's GetConfig ID through the userland gate, where entry
        // 4 is another function.
        let mut x = [function::GET_CONFIG.into(), 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(
            monitor.call(USER_SMC, &mut x),
            unknown(0, function::GET_CONFIG)
        );
        let mut x = [user_function::GET_CONFIG.into(), 5, 0, 0, 0, 0, 0, 0];
        assert_eq!(
            monitor.call(2, &mut x),
            unknown(2, user_function::GET_CONFIG)
        );
        assert_eq!(
            monitor.call_on(1, USER_SMC, &mut x),
            SmcEffect::Abort(Abort::UserCallOnCore {
                core: 1,
                id: user_function::GET_CONFIG
            })
        );
    }

    #[test]
    fn user_gate_serves_get_config_under_its_own_numbering() {
        let monitor = monitor(DramSize::Gb6);
        let mut x = [
            user_function::GET_CONFIG.into(),
            config_item::MEMORY_MODE.into(),
            0,
            0,
            0,
            0,
            0,
            0,
        ];
        assert_eq!(monitor.call(USER_SMC, &mut x), SmcEffect::Resume);
        assert_eq!(x[..2], [STATUS_SUCCESS, 0x11]);

        // The user number means nothing through the kernel gate.
        let (effect, _) = smc(
            &monitor,
            [
                user_function::GET_CONFIG.into(),
                config_item::MEMORY_MODE.into(),
                0,
                0,
            ],
        );
        assert_eq!(
            effect,
            SmcEffect::Abort(Abort::UnknownFunction {
                immediate: 1,
                id: user_function::GET_CONFIG
            })
        );
    }

    #[test]
    fn charger_hi_z_mode_is_the_one_settable_item_and_reads_back() {
        let monitor = monitor(DramSize::Gb4);
        let item = config_item::IS_CHARGER_HI_Z_MODE_ENABLED.into();
        let get = || {
            let mut x = [user_function::GET_CONFIG.into(), item, 0, 0, 0, 0, 0, 0];
            assert_eq!(monitor.call(USER_SMC, &mut x), SmcEffect::Resume);
            x[1]
        };
        assert_eq!(get(), 0);
        let mut x = [user_function::SET_CONFIG.into(), item, 0, 1, 0, 0, 0, 0];
        assert_eq!(monitor.call(USER_SMC, &mut x), SmcEffect::Resume);
        assert_eq!(x[0], STATUS_SUCCESS);
        assert_eq!(get(), 1);
        let mut x = [
            user_function::SET_CONFIG.into(),
            config_item::MEMORY_MODE.into(),
            0,
            1,
            0,
            0,
            0,
            0,
        ];
        assert_eq!(monitor.call(USER_SMC, &mut x), SmcEffect::Resume);
        assert_eq!(x[0], STATUS_INVALID_ARGUMENT);
    }

    #[test]
    fn random_bytes_fill_exactly_the_requested_prefix() {
        let monitor = monitor(DramSize::Gb4);
        let mut x = [function::GENERATE_RANDOM_BYTES.into(), 11, 0, 0, 0, 0, 0, 0];
        assert_eq!(monitor.call(KERNEL_SMC, &mut x), SmcEffect::Resume);
        assert_eq!(x[0], STATUS_SUCCESS);
        assert_ne!(x[1], 11, "first word still holds the request");
        assert_ne!(x[2] & 0xFF_FFFF, 0);
        // Bytes 11.. keep what the caller passed (zero).
        assert_eq!((x[2] >> 24, &x[3..]), (0, &[0; 5][..]));

        let (_, x) = smc(&monitor, [function::GENERATE_RANDOM_BYTES.into(), 57, 0, 0]);
        assert_eq!(x[0], STATUS_INVALID_ARGUMENT);
    }

    #[test]
    fn random_stream_is_a_function_of_the_seed() {
        let draw = |seed| {
            let monitor = console(&BootParams {
                rng_seed: seed,
                ..BootParams::default()
            });
            let first = smc(&monitor, [function::GENERATE_RANDOM_BYTES.into(), 56, 0, 0]).1;
            let second = smc(&monitor, [function::GENERATE_RANDOM_BYTES.into(), 56, 0, 0]).1;
            (first, second)
        };
        let (first, second) = draw(1);
        assert_eq!(draw(1), (first, second));
        assert_ne!(first, second);
        assert_ne!(draw(2).0, first);
    }

    #[test]
    fn register_writes_merge_under_the_mask_and_return_the_previous_value() {
        let monitor = monitor(DramSize::Gb4);
        // A PMC register the monitor proxies (+0x1a8).
        let register = 0x7000_E5A8;
        let access = |mask, value| {
            smc(
                &monitor,
                [function::READ_WRITE_REGISTER.into(), register, mask, value],
            )
            .1[..2]
                .to_vec()
        };
        assert_eq!(access(0xFFFF_FFFF, 0x1234_5678), [STATUS_SUCCESS, 0]);
        assert_eq!(
            access(0x0000_FF00, 0xFFFF_FFFF),
            [STATUS_SUCCESS, 0x1234_5678]
        );
        assert_eq!(access(0, 0xDEAD_BEEF), [STATUS_SUCCESS, 0x1234_FF78]);
        assert_eq!(access(0, 0), [STATUS_SUCCESS, 0x1234_FF78]);
    }

    #[test]
    fn register_access_is_confined_to_aligned_proxied_registers() {
        let monitor = monitor(DramSize::Gb4);
        // Unaligned, just below PMC, just past the memory controller,
        // DRAM, a PMC secure scratch register and the memory controller's
        // security carveout registers: none is proxied.
        for address in [
            MC_EMEM_CFG + 2,
            0x7000_E3FC,
            0x7001_A000,
            0x8000_0000,
            0x7000_E400 + 0x2C0,
            mc::BASE + 0x670 - 4,
            mc::BASE + 0x9C0,
        ] {
            let (effect, x) = smc(
                &monitor,
                [function::READ_WRITE_REGISTER.into(), address, 0, 0],
            );
            assert_eq!(
                (effect, x[0]),
                (SmcEffect::Resume, STATUS_INVALID_ARGUMENT),
                "{address:#x}"
            );
        }
        // The BPMP's and APB DMA's SMMU bindings: success, nothing done.
        for address in [mc::BASE + 0x23C, mc::BASE + 0x298] {
            let (effect, x) = smc(
                &monitor,
                [function::READ_WRITE_REGISTER.into(), address, u64::MAX, 7],
            );
            assert_eq!((effect, x[0], x[1]), (SmcEffect::Resume, STATUS_SUCCESS, 0));
        }
    }

    #[test]
    fn cpu_on_starts_each_secondary_core_once() {
        let monitor = monitor(DramSize::Gb4);
        let on = |core| {
            smc(
                &monitor,
                [function::CPU_ON.into(), core, 0x800D_60E8, 0x800E_8380],
            )
        };

        let (effect, x) = on(2);
        assert_eq!(
            effect,
            SmcEffect::StartCore {
                core: 2,
                entry: 0x800D_60E8,
                context: 0x800E_8380
            }
        );
        assert_eq!(x[0], PSCI_SUCCESS);
        assert_eq!((on(2).0, on(2).1[0]), (SmcEffect::Resume, PSCI_ALREADY_ON));
        // The boot core is running from the start.
        assert_eq!((on(0).0, on(0).1[0]), (SmcEffect::Resume, PSCI_ALREADY_ON));
        assert_eq!(
            (on(4).0, on(4).1[0]),
            (SmcEffect::Resume, PSCI_INVALID_PARAMETERS)
        );
    }

    #[test]
    fn sleep_needs_the_other_cores_off_and_their_partitions_follow() {
        let monitor = monitor(DramSize::Gb4);
        let partitions = || {
            let status = smc(
                &monitor,
                [
                    function::READ_WRITE_REGISTER.into(),
                    PMC_PWRGATE_STATUS,
                    0,
                    0,
                ],
            );
            status.1[1] >> 9 & 7
        };
        assert_eq!(partitions(), 0);
        for core in 1..4 {
            smc(&monitor, [function::CPU_ON.into(), core, 0x8000_0000, 0]);
        }
        assert_eq!(partitions(), 7);

        let sleep = |state: u64| {
            smc(
                &monitor,
                [function::CPU_SUSPEND.into(), state, 0x8000_1000, 7],
            )
        };
        // Any state but the one there is is denied.
        assert_eq!(
            (sleep(27).0, sleep(27).1[0]),
            (SmcEffect::Resume, PSCI_DENIED)
        );
        assert!(matches!(
            sleep(0x0201_001B).0,
            SmcEffect::Abort(Abort::SleepWithCoresOn)
        ));
        for core in 1..4 {
            let mut x = [function::CPU_OFF.into(), 0, 0, 0, 0, 0, 0, 0];
            assert_eq!(
                monitor.call_on(core, KERNEL_SMC, &mut x),
                SmcEffect::PowerOff
            );
        }
        assert_eq!(partitions(), 0);
        // The SoC as the firmware leaves it for sleep, then with a clock it
        // must keep stopped.
        let set_bits = |register: u64, bits: u32| {
            let Access::Done(value) = monitor.bus.read(0, register, 4) else {
                panic!("CAR register {register:#x}");
            };
            let value = value as u32 | bits;
            monitor.bus.write(0, register, 4, value.into());
        };
        for (register, mask, required, _) in SLEEP_REQUIREMENTS.into_iter().skip(1) {
            set_bits(register, mask & required);
        }
        let Access::Done(enables_h) = monitor.bus.read(0, CAR_BASE + 0x014, 4) else {
            panic!("CLK_OUT_ENB_H");
        };
        monitor
            .bus
            .write(0, CAR_BASE + 0x014, 4, enables_h & !(1 << 8));
        assert!(matches!(
            sleep(0x0201_001B).0,
            SmcEffect::Abort(Abort::SocNotReadyForSleep { what, .. }) if what.contains("KFUSE")
        ));
        set_bits(CAR_BASE + 0x014, 1 << 8);
        assert_eq!(
            sleep(0x0201_001B).0,
            SmcEffect::Sleep {
                entry: 0x8000_1000,
                context: 7
            }
        );
        // A core that was switched off can be turned on again.
        let again = smc(&monitor, [function::CPU_ON.into(), 2, 0x8000_0000, 0]);
        assert!(matches!(again.0, SmcEffect::StartCore { core: 2, .. }));
        assert_eq!(partitions(), 2);
    }

    #[test]
    fn carveouts_are_recorded_per_index() {
        let monitor = monitor(DramSize::Gb4);
        // What the kernel asks for (kernel carveout 0 of firmware 23.5.0).
        let (_, x) = smc(
            &monitor,
            [
                function::CONFIGURE_CARVEOUT.into(),
                1,
                0x8006_0000,
                0x1038_0000,
            ],
        );
        assert_eq!(x[0], STATUS_SUCCESS);
        assert_eq!(
            monitor.carveouts(),
            [
                Carveout::default(),
                Carveout {
                    address: 0x8006_0000,
                    size: 0x1038_0000
                }
            ]
        );
        // No third carveout, nothing off the 128 KiB grid, nothing of
        // 512 MiB or more.
        for (index, address, size) in [
            (2, 0, 0),
            (0, 0x8001_0000, 0x2_0000),
            (0, 0x8000_0000, 0x1_0000),
            (0, 0x8000_0000, 512 << 20),
        ] {
            let (_, x) = smc(
                &monitor,
                [function::CONFIGURE_CARVEOUT.into(), index, address, size],
            );
            assert_eq!(
                x[0], STATUS_INVALID_ARGUMENT,
                "{index} {address:#x} {size:#x}"
            );
        }
    }

    #[test]
    fn panic_reports_the_kernel_code_and_does_not_resume() {
        let (effect, _) = smc(
            &monitor(DramSize::Gb4),
            [function::PANIC.into(), 0xF00, 0, 0],
        );
        assert_eq!(effect, SmcEffect::Panic { code: 0xF00 });
    }
}
