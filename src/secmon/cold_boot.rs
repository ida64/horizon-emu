//! The secure monitor's cold boot: what it does between the NX bootloader
//! releasing CCPLEX core 0 at its first instruction and its drop to the
//! kernel at EL1.
//!
//! The monitor runs at EL3, which this machine cannot give, so its duties
//! are done here instead of its code running. What follows is read from
//! the 23.5.0 Erista monitor itself (the third PK11 section of
//! package1, `0xE000` bytes): it runs from IRAM at [`ENTRY`],
//! LZ4-decompresses its resident part to TZRAM (`0x7C010800`), turns its MMU
//! on and runs a one-time boot routine (virtual `0x1F01E15D4`, TZRAM
//! `0x7C0115D4`) that it erases before entering the kernel. Addresses below
//! in parentheses are virtual addresses of that routine and its callees;
//! the monitor maps IRAM's first page at `0x1F009F000`.
//!
//! # The mailbox
//!
//! The bootloader and the monitor talk through IRAM's first page while
//! both run:
//!
//! | IRAM | Written by | |
//! |---|---|---|
//! | `+0x08` | monitor | `TIMERUS` when it starts (boot stage `0x40032020`) |
//! | `+0x1C` bit 0 | bootloader side | recovery boot: config item 7, and whether item 17 has a hash |
//! | `+0xF8` | bootloader | its progress, [`BOOTLOADER_STATES`] |
//! | `+0xFC` | monitor | 1 once it is set up (`0x1F01E16D4`) |
//!
//! The monitor copies `+0x10..+0x20` at start-up (`0x1F0143C38`) and uses
//! only that bit of it. It waits for the bootloader's state to reach a
//! value by polling every microsecond (`0x1F01E0A5C`); the bootloader
//! (23.5.0 NX bootloader, its main routine) waits for the monitor's state
//! to be non-zero, then loads the boot config into IRAM and sets 1,
//! initialises DRAM and sets 2, loads package2 to DRAM and sets 4: it never
//! sets 3, which the monitor's `>= 3` wait lets through.
//!
//! # The monitor's steps, in order
//!
//! 1. Its resident part LZ4-decompressed to TZRAM [`RESIDENT`] (the boot
//!    stage, `0x400307FC`); its SoC security set-up ([`super::setup`]:
//!    the interrupt distributor, the security engine's settings and key
//!    slot permissions and the digest the resume checks, what the warm
//!    path reads, the shared reset path, core 0's interrupt set-up), less
//!    what nothing here models ([`NOT_MODELLED`]); then it zeroes its own
//!    IRAM image (`0x40030000`, `0xE000` bytes; the bootloader reads
//!    package2's partition there next) and sets its state to 1.
//! 2. Bootloader state 1: it copies the boot config (`0x400` bytes at IRAM
//!    `0x4003F800`) unless the console is not a development unit, in which
//!    case the copy is zeroes (`0x1F01E0A04`; the hardware state from fuse
//!    `RESERVED_ODM4`, `0x1F0143A04`). This console is a production one
//!    ([`super::config_item::HARDWARE_STATE`]), so the boot config is all
//!    zeroes and what it says is what an all-zero boot config gives.
//! 3. Bootloader state 2: two more peripherals secure
//!    (`setup::apb_secure`); it copies the warmboot firmware from
//!    IRAM `0x4003E000` to DRAM `0x8000D000` (`0x17F0` bytes,
//!    `0x1F01E1734`) and writes [`DRAM_MARKERS`] (`0x1F01E10E8`).
//! 4. Bootloader state 3: it takes package2 from DRAM [`PACKAGE2`], sized by
//!    its header's counter, verifies, decrypts and places it
//!    ([`crate::package2::unpack`] does it the same way), hashes it whole
//!    in a recovery boot (for config item 17), and sets core 0 to start at
//!    DRAM base + the header's entry point with `X0` = 0 (`0x1F0143C20`).
//! 5. Bootloader state 4: its last set-up (`setup::final_setup`:
//!    a carveout, the BPMP off the AHB arbiter, a wait for the BPMP to halt
//!    itself, [`HALT_COP_EVENTS`] mode 2, the activity monitor's clock,
//!    `0x1F0142FD4`, the BPMP parked for good, `0x40000000`), then the drop
//!    to EL1 (`SPSR_EL3` `0x3C5`, `0x1F0140CC0`).
//!
//! # What TZRAM holds here
//!
//! The monitor's resident image exactly as its boot stage decompresses it
//! (code, rodata and initialised data, `0xAC18` bytes at [`RESIDENT`]),
//! nothing else: its code never runs, and its data does not track this
//! monitor's state. The SC7 preparation reads two things out of it (the
//! BPMP's SC7 stub and the block its engine context ends with) and saves
//! `0x7C012000..0x7C020000` of it for the warmboot firmware to verify.
//! The SC7 routine's stack (linked lists, bounce buffers) is at
//! [`sc7::STACK`].
//!
//! # Boot parameters
//!
//! The config items the boot config answers (1, 10, 11, 12, 18) come from
//! it on this path, the recovery flag (items 7, 17) from the mailbox. The
//! command line's `--dram`, `--debug-mode` and `--disable-program-verification`
//! still override them: an option set to something other than its default
//! replaces what the boot config gives, standing for the development boot
//! config a retail console cannot have (a production monitor ignores the
//! boot config, so `--dram 6` or `8` is reachable no other way). The DRAM
//! ID and device ID stay the boot parameters': the monitor takes them from
//! fuses this machine does not model.

use std::{
    fmt,
    ops::RangeInclusive,
    sync::atomic::{AtomicBool, Ordering},
    thread,
    time::Duration,
};

use sha2::{Digest, Sha256};

use super::SecureMonitor;
use crate::{
    boot::{BootParams, CoreEntry},
    bus::{Access, Bus},
    image::{ImageError, KernelImage},
    memory::GuestRam,
    package2::{self, Package2Error},
    soc::{
        car,
        iram::{self, Iram},
        tmr, tzram,
    },
};

use super::sc7::{self, Port, Sc7Fail};

/// Where the bootloader starts core 0: the monitor's first instruction.
/// Its code is linked there (it jumps to `0x40032000` and copies to
/// `0x4003E000` less a size, absolute addresses).
pub const ENTRY: u64 = 0x4003_0000;

/// The monitor's image in IRAM from [`ENTRY`], which it zeroes once it no
/// longer runs from there (`0x1F01E1334`).
const IMAGE_SIZE: usize = 0xE000;

/// SHA-256 of the 23.5.0 Erista monitor's image, the only monitor whose
/// cold boot this is. (Its first four bytes are also bytes 4-7 of
/// package1's header.)
const IMAGE_SHA256: [u8; 32] = [
    0xB0, 0xB5, 0x85, 0xF0, 0x26, 0x59, 0x0E, 0x9D, 0xBF, 0xF7, 0x88, 0x29, 0xD0, 0x8C, 0x1A, 0x70,
    0x28, 0x93, 0x3D, 0x84, 0xD6, 0x6B, 0x42, 0x7E, 0x1C, 0x68, 0x37, 0xD2, 0xCE, 0xE9, 0x34, 0x3D,
];

/// The monitor's resident part, LZ4-compressed in its IRAM image
/// (`0x40030DE4..0x40038328`), which its boot stage decompresses to TZRAM
/// [`RESIDENT`] first of all (`0x400307FC`, at most `0xC800` bytes).
const RESIDENT_IMAGE: std::ops::Range<usize> = 0xDE4..0x8328;
pub const RESIDENT: u64 = 0x7C01_0800;
const RESIDENT_LIMIT: usize = 0xC800;
const RESIDENT_SIZE: usize = 0xAC18;
const _: () = assert!(RESIDENT + RESIDENT_LIMIT as u64 <= tzram::BASE + tzram::SIZE);

/// Mailbox words, as offsets into IRAM.
const START_TIME: u32 = 0x08;
const RECOVERY_BOOT: u32 = 0x1C;
const BOOTLOADER_STATE: u32 = 0xF8;
const MONITOR_STATE: u32 = 0xFC;

/// The bootloader's states the monitor waits for, by name.
pub const BOOTLOADER_STATES: [(u32, &str); 4] = [
    (1, "boot config loaded"),
    (2, "DRAM initialised"),
    (3, "package2 loaded"),
    (4, "done"),
];
const BOOT_CONFIG_LOADED: u32 = 1;
const DRAM_INITIALISED: u32 = 2;
const PACKAGE2_LOADED: u32 = 3;
const BOOTLOADER_DONE: u32 = 4;

/// The monitor's state once it no longer needs its IRAM image.
const MONITOR_INITIALISED: u32 = 1;

/// How often the wait for the bootloader looks again. The monitor polls
/// every microsecond of guest time; the value only bounds how late the
/// wait notices.
const POLL_INTERVAL: Duration = Duration::from_micros(50);

const TIMERUS_CNTR_1US: u64 = tmr::BASE + 0x10;

/// `FLOW_CTLR_HALT_COP_EVENTS`: after the bootloader's last state the
/// monitor polls it, every microsecond, until its mode (bits 31:29) is 2,
/// the BPMP halted waiting for an event, which is how package1ldr parks
/// the BPMP once the bootloader returns (it writes `0x50000000` in a
/// loop); then it writes `0x40000000` (in `0x1F0147430`).
pub const HALT_COP_EVENTS: u64 = 0x6000_7004;

/// The boot config block, as the monitor keeps it.
const BOOT_CONFIG_SIZE: usize = 0x400;

const WARMBOOT_IRAM: u64 = 0x4003_E000;
const WARMBOOT_DRAM: u64 = 0x8000_D000;
const WARMBOOT_SIZE: usize = 0x17F0;

/// What the monitor writes below the kernel once DRAM works, as
/// `(address, little-endian bytes)` (`0x1F01E10E8`). \[unknown\] what reads
/// it: nothing in the kernel's boot does.
pub const DRAM_MARKERS: [(u64, &[u8]); 3] = [
    (0x8005_FF00, &0x2_0000_0083u64.to_le_bytes()),
    (0x8005_FF08, &0x210u32.to_le_bytes()),
    (0x8005_FFFC, &0xC0ED_BBCCu32.to_le_bytes()),
];

/// Where the bootloader puts package2 and the monitor reads its header
/// (both in their main routines).
pub const PACKAGE2: u64 = 0xA980_0000;

/// Sizes the header may give: the bootloader refuses others; the monitor's
/// own check (`0x1F01E0B70`) also needs it to hold the payloads.
const PACKAGE2_SIZE: RangeInclusive<u64> = 0x200..=0x7F_C000;

/// Set-up the monitor does that nothing here models, so none of it is done:
/// its device drivers' bases, the system counter (SYSCTR0: checked to be
/// unprogrammed, else panic, then started at the boot config's value or 0),
/// the PMC, fuse, timer, flow controller, SMMU and memory controller
/// settings [`super::setup`] lists as not done, the keys it loads into
/// the security engine (the package2 key goes through keyslot 8 from the
/// master key in keyslot 13; the kernel's later key calls are served by
/// [`super::crypto`]), walling the halted BPMP off (its exception
/// vectors, SMMU address space and the activity monitor's watch on it),
/// `SCR_EL3` and the EL2/EL1 system registers (the kernel overwrites the
/// EL1 ones it needs; `SCTLR_EL1` is entered as `0x00C50838`).
pub const NOT_MODELLED: &str = "SoC security, counter and power set-up";

/// Machine state the cold boot reads and writes.
pub struct ColdBootContext<'a> {
    pub ram: &'a GuestRam,
    pub iram: &'a Iram,
    /// For `TIMERUS` and the flow controller, read as core 0.
    pub bus: &'a Bus,
}

/// Where core 0 enters the kernel, and the kernel the monitor placed.
#[derive(Debug)]
pub struct ColdBoot {
    pub entry: CoreEntry,
    pub kernel: KernelImage,
}

/// What core 0's release at a given entry turned out to be.
#[derive(Debug)]
pub enum Released {
    /// The bootloader released it at [`ENTRY`].
    ColdBoot(ColdBoot),
    /// The warmboot firmware released it at [`sc7::RESUME_ENTRY`] after
    /// SC7: the kernel resumes where `CPU_SUSPEND` said.
    Resume(CoreEntry),
}

#[derive(Debug)]
pub enum ColdBootError {
    /// The machine stopped while the monitor waited for the bootloader.
    Stopped {
        waiting_for: u32,
    },
    /// Core 0 was released somewhere other than [`ENTRY`].
    Entry(u64),
    /// IRAM at [`ENTRY`] does not hold the monitor this cold boot is.
    NotTheMonitor,
    /// A register the monitor reads, by name, and why it could not.
    Register {
        name: &'static str,
        why: &'static str,
    },
    /// The machine stopped while the monitor waited for the BPMP to halt.
    StoppedBeforeBpmpHalt,
    /// The monitor's SoC set-up could not be done.
    Setup(String),
    /// Core 0 released at the resume vector without a suspend to resume.
    NotSuspended,
    /// The resume's SoC set-up could not be done.
    Resume(String),
    /// A DRAM range the monitor reads or writes is not memory here.
    Dram {
        address: u64,
        len: usize,
    },
    /// The header's package2 size is out of bounds.
    Package2Size(u64),
    Package2(Package2Error),
    Kernel(ImageError),
    /// The entry point is not the kernel image's first byte.
    KernelEntry {
        entry: u64,
        load_address: u64,
    },
    /// Something the monitor would do that is not modelled.
    Unmodelled(&'static str),
}

impl fmt::Display for ColdBootError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotSuspended => {
                return write!(
                    f,
                    "secure monitor resume: core 0 released at {:#x} with no suspend pending",
                    sc7::RESUME_ENTRY
                );
            }
            Self::Resume(why) => return write!(f, "secure monitor resume: {why}"),
            _ => write!(f, "secure monitor cold boot: ")?,
        }
        match self {
            Self::NotSuspended | Self::Resume(_) => Ok(()),
            Self::Stopped { waiting_for } => {
                let name = BOOTLOADER_STATES
                    .iter()
                    .find(|(state, _)| state == waiting_for)
                    .map_or("?", |(_, name)| name);
                write!(
                    f,
                    "stopped waiting for the bootloader's state {waiting_for} ({name})"
                )
            }
            Self::Entry(entry) => write!(
                f,
                "core 0 released at {entry:#x}, not the monitor's entry {ENTRY:#x}"
            ),
            Self::NotTheMonitor => write!(
                f,
                "IRAM at {ENTRY:#x} does not hold the 23.5.0 Erista secure monitor"
            ),
            Self::Register { name, why } => write!(f, "{name} unreadable: {why}"),
            Self::StoppedBeforeBpmpHalt => {
                write!(f, "stopped waiting for the BPMP to halt")
            }
            Self::Setup(why) => write!(f, "SoC set-up: {why}"),
            Self::Dram { address, len } => {
                write!(f, "{len:#x} bytes at {address:#x} are not DRAM here")
            }
            Self::Package2Size(size) => write!(
                f,
                "package2 header gives size {size:#x}, outside {:#x}..={:#x}",
                PACKAGE2_SIZE.start(),
                PACKAGE2_SIZE.end()
            ),
            Self::Package2(error) => write!(f, "{error}"),
            Self::Kernel(error) => write!(f, "kernel image: {error}"),
            Self::KernelEntry {
                entry,
                load_address,
            } => write!(
                f,
                "package2 entry {entry:#x} is not the kernel's first byte {load_address:#x}"
            ),
            Self::Unmodelled(what) => write!(f, "not modelled: {what}"),
        }
    }
}

impl std::error::Error for ColdBootError {}

/// What the monitor answers for the config items the boot config and the
/// mailbox decide.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(super) struct BootSettings {
    /// Item 1.
    pub(super) program_verification_disabled: bool,
    /// Item 10.
    pub(super) memory_mode: u64,
    /// Item 11.
    pub(super) development_function_enabled: bool,
    /// Item 12.
    pub(super) kernel_configuration: u64,
    /// Item 18.
    pub(super) unknown_18: u64,
    /// In a recovery boot (item 7), package2's SHA-256 (item 17).
    pub(super) recovery_package2_hash: Option<[u8; 32]>,
}

impl BootSettings {
    /// The answers the boot parameters give, as when the kernel is entered
    /// from an ELF without the boot chain.
    pub(super) fn from_params(params: &BootParams) -> Self {
        Self {
            program_verification_disabled: params.program_verification_disabled,
            memory_mode: params.dram.memory_mode(),
            development_function_enabled: params.development_function_enabled,
            kernel_configuration: params.kernel_configuration.bits(params.dram),
            unknown_18: 0,
            recovery_package2_hash: None,
        }
    }

    /// What the monitor answers from the boot config `block` (GetConfig,
    /// `0x1F0144624`). Item 1 is `+0x320` bit 0; items 10-12 depend on the
    /// development functions flag, `+0x10` bit 1: clear, the memory mode is
    /// 1 and the kernel configuration and item 11 are 0; item 18 is the
    /// quadword at `+0x70`. With the flag set the monitor decodes a memory
    /// mode from `+0x23` and kernel flags from `+0x11` and `+0x21`, which is
    /// not modelled (a production console never gets there).
    fn from_boot_config(block: &[u8; BOOT_CONFIG_SIZE]) -> Result<Self, ColdBootError> {
        if block[0x10] & 2 != 0 {
            return Err(ColdBootError::Unmodelled(
                "a boot config enabling development functions",
            ));
        }
        Ok(Self {
            program_verification_disabled: block[0x320] & 1 != 0,
            memory_mode: 1,
            development_function_enabled: false,
            kernel_configuration: 0,
            unknown_18: u64::from_le_bytes(block[0x70..0x78].try_into().unwrap()),
            recovery_package2_hash: None,
        })
    }

    /// `self` with every answer `params` sets to something other than the
    /// default parameters' answer replaced by that.
    fn overridden_by(self, params: &BootParams) -> Self {
        let cli = Self::from_params(params);
        let default = Self::from_params(&BootParams::default());
        fn pick<T: PartialEq>(cli: T, default: T, boot: T) -> T {
            if cli != default { cli } else { boot }
        }
        Self {
            program_verification_disabled: pick(
                cli.program_verification_disabled,
                default.program_verification_disabled,
                self.program_verification_disabled,
            ),
            memory_mode: pick(cli.memory_mode, default.memory_mode, self.memory_mode),
            development_function_enabled: pick(
                cli.development_function_enabled,
                default.development_function_enabled,
                self.development_function_enabled,
            ),
            kernel_configuration: pick(
                cli.kernel_configuration,
                default.kernel_configuration,
                self.kernel_configuration,
            ),
            unknown_18: self.unknown_18,
            recovery_package2_hash: self.recovery_package2_hash,
        }
    }
}

/// IRAM offset of physical `address`, which the callers keep inside it.
fn iram_offset(address: u64) -> u32 {
    (address - iram::BASE) as u32
}

/// Reads the 32-bit register at `address` through `bus` as core 0.
fn read_register(bus: &Bus, name: &'static str, address: u64) -> Result<u32, ColdBootError> {
    match bus.read(0, address, 4) {
        Access::Done(value) => Ok(value as u32),
        Access::Unmapped => Err(ColdBootError::Register {
            name,
            why: "no device mapped",
        }),
        Access::Unimplemented(device) => Err(ColdBootError::Register { name, why: device }),
    }
}

fn read_word(iram: &Iram, offset: u32) -> u32 {
    iram.read(offset, 4).expect("mailbox word inside IRAM")
}

fn read_dram(ram: &GuestRam, address: u64, len: usize) -> Result<Vec<u8>, ColdBootError> {
    let mut bytes = vec![0; len];
    ram.read(address, &mut bytes)
        .ok_or(ColdBootError::Dram { address, len })?;
    Ok(bytes)
}

fn write_dram(ram: &GuestRam, address: u64, bytes: &[u8]) -> Result<(), ColdBootError> {
    ram.write(address, bytes).ok_or(ColdBootError::Dram {
        address,
        len: bytes.len(),
    })
}

/// LZ4 block decompression, as the monitor's boot stage does it
/// (`0x400306C4`). `None` if `source` is malformed or decompresses to more
/// than `limit` bytes.
fn lz4_block(source: &[u8], limit: usize) -> Option<Vec<u8>> {
    fn length(source: &[u8], at: &mut usize, mut length: usize) -> Option<usize> {
        if length == 15 {
            loop {
                let byte = *source.get(*at)?;
                *at += 1;
                length += usize::from(byte);
                if byte != 255 {
                    break;
                }
            }
        }
        Some(length)
    }
    let mut out = Vec::new();
    let mut at = 0;
    while at < source.len() {
        let token = source[at];
        at += 1;
        let literals = length(source, &mut at, usize::from(token >> 4))?;
        out.extend_from_slice(source.get(at..at.checked_add(literals)?)?);
        at += literals;
        if at >= source.len() {
            break;
        }
        let offset = usize::from(u16::from_le_bytes([*source.get(at)?, *source.get(at + 1)?]));
        at += 2;
        let count = length(source, &mut at, usize::from(token & 15))? + 4;
        if offset == 0 || offset > out.len() || out.len() + count > limit {
            return None;
        }
        for _ in 0..count {
            out.push(out[out.len() - offset]);
        }
    }
    (out.len() <= limit).then_some(out)
}

/// Waits until the bootloader's state is at least `state`.
fn wait_for_bootloader(iram: &Iram, state: u32, stop: &AtomicBool) -> Result<(), ColdBootError> {
    while read_word(iram, BOOTLOADER_STATE) < state {
        if stop.load(Ordering::Relaxed) {
            return Err(ColdBootError::Stopped { waiting_for: state });
        }
        thread::sleep(POLL_INTERVAL);
    }
    Ok(())
}

/// `CLK_RST_CONTROLLER` registers of the activity monitor (hekate
/// `clock.h`): its reset and clock enable are bit 23 of the V group.
const RST_DEVICES_V: u64 = car::BASE + 0x358;
const CLK_OUT_ENB_V: u64 = car::BASE + 0x360;
const CLK_SOURCE_ACTMON: u64 = car::BASE + 0x3E8;
const ACTMON_BIT: u32 = 1 << 23;
/// CLK_M, divider 0.
const ACTMON_SOURCE: u32 = 0xC000_0000;

/// The activity monitor's clock, once the BPMP has halted (`0x1F0142FD4`,
/// called from `0x1F0147430`): held in reset and gated, its source set,
/// clocked, let out of reset. The kernel's sleep needs it running
/// ([`super::SLEEP_REQUIREMENTS`]).
pub(super) fn actmon_clock(port: &Port<'_>) -> Result<(), Sc7Fail> {
    port.modify(RST_DEVICES_V, 0, ACTMON_BIT)?;
    port.modify(CLK_OUT_ENB_V, ACTMON_BIT, 0)?;
    port.write(CLK_SOURCE_ACTMON, ACTMON_SOURCE)?;
    port.modify(CLK_OUT_ENB_V, 0, ACTMON_BIT)?;
    port.modify(RST_DEVICES_V, ACTMON_BIT, 0)
}

impl SecureMonitor {
    /// The monitor's cold boot, from core 0's release at `entry` to the
    /// kernel's entry at EL1 (see the [module](self)). Runs on core 0's
    /// thread while the bootloader goes on on the BPMP; returns
    /// [`ColdBootError::Stopped`] soon after `stop` is set. On success the
    /// kernel is in DRAM and the monitor answers config items as the boot
    /// config and mailbox decided.
    pub fn cold_boot(
        &self,
        entry: u64,
        context: ColdBootContext<'_>,
        stop: &AtomicBool,
    ) -> Result<ColdBoot, ColdBootError> {
        let ColdBootContext { ram, iram, bus } = context;
        if entry != ENTRY {
            return Err(ColdBootError::Entry(entry));
        }
        let image_offset = iram_offset(ENTRY);
        let mut image = vec![0; IMAGE_SIZE];
        iram.copy_out(image_offset, &mut image)
            .expect("monitor image inside IRAM");
        if Sha256::digest(&image)[..] != IMAGE_SHA256 {
            return Err(ColdBootError::NotTheMonitor);
        }
        // The boot stage's first step: the resident part to TZRAM.
        let resident = lz4_block(&image[RESIDENT_IMAGE], RESIDENT_LIMIT)
            .filter(|resident| resident.len() == RESIDENT_SIZE)
            .ok_or(ColdBootError::NotTheMonitor)?;
        let port = Port {
            bus,
            ram,
            stack: super::setup::BOOT_STACK,
        };
        let setup = |result: Result<(), Sc7Fail>| match result {
            Ok(()) => Ok(()),
            Err(Sc7Fail::Stopped) => Err(ColdBootError::StoppedBeforeBpmpHalt),
            Err(fail) => Err(ColdBootError::Setup(fail.to_string())),
        };
        setup(port.write_bytes(RESIDENT, &resident))?;

        let now = read_register(bus, "TIMERUS", TIMERUS_CNTR_1US)?;
        iram.write(START_TIME, 4, now).expect("mailbox inside IRAM");
        let recovery = read_word(iram, RECOVERY_BOOT) & 1 != 0;

        iram.load(image_offset, &[0; IMAGE_SIZE])
            .expect("monitor image inside IRAM");
        let gic = self
            .gic()
            .map_err(|fail| ColdBootError::Setup(fail.to_string()))?;
        let carveouts = self.kernel_carveouts();
        setup(super::setup::cold_boot(&port, gic, carveouts))?;
        iram.write(MONITOR_STATE, 4, MONITOR_INITIALISED)
            .expect("mailbox inside IRAM");

        wait_for_bootloader(iram, BOOT_CONFIG_LOADED, stop)?;
        // A production unit's monitor takes zeroes instead of the boot
        // config the bootloader left at 0x4003F800.
        let boot_config = [0; BOOT_CONFIG_SIZE];
        let mut settings = BootSettings::from_boot_config(&boot_config)?;

        wait_for_bootloader(iram, DRAM_INITIALISED, stop)?;
        setup(super::setup::apb_secure(&port))?;
        let mut warmboot = vec![0; WARMBOOT_SIZE];
        iram.copy_out(iram_offset(WARMBOOT_IRAM), &mut warmboot)
            .expect("warmboot inside IRAM");
        write_dram(ram, WARMBOOT_DRAM, &warmboot)?;
        for (address, bytes) in DRAM_MARKERS {
            write_dram(ram, address, bytes)?;
        }

        wait_for_bootloader(iram, PACKAGE2_LOADED, stop)?;
        let header = read_dram(ram, PACKAGE2, package2::HEADER_SIZE)?;
        // The counter, stored in the clear, encodes the size
        // (`0x1F01E0B70`; the bootloader reads it the same way).
        let word =
            |index: usize| u32::from_le_bytes(header[0x100 + index * 4..][..4].try_into().unwrap());
        let size = u64::from(word(0) ^ word(2) ^ word(3));
        if !PACKAGE2_SIZE.contains(&size) {
            return Err(ColdBootError::Package2Size(size));
        }
        let package2 = read_dram(ram, PACKAGE2, size as usize)?;
        if recovery {
            settings.recovery_package2_hash = Some(Sha256::digest(&package2).into());
        }
        let unpacked = package2::unpack(&package2, self.state.lock().unwrap().crypto.keys())
            .map_err(ColdBootError::Package2)?;
        for payload in &unpacked.payloads {
            write_dram(ram, payload.address, &payload.bytes)?;
        }
        let entry = unpacked.entry;
        let kernel = unpacked
            .into_kernel_image()
            .map_err(ColdBootError::Kernel)?;
        if kernel.load_address() != entry {
            return Err(ColdBootError::KernelEntry {
                entry,
                load_address: kernel.load_address(),
            });
        }

        wait_for_bootloader(iram, BOOTLOADER_DONE, stop)?;
        setup(super::setup::final_setup(&port, stop))?;
        let mut state = self.state.lock().unwrap();
        state.settings = settings.overridden_by(&self.params);
        state.boot_chain = true;
        drop(state);
        Ok(ColdBoot {
            entry: CoreEntry::primary(entry),
            kernel,
        })
    }
}

impl SecureMonitor {
    /// Core 0 let out of reset at `entry`: the cold boot at [`ENTRY`]
    /// ([`Self::cold_boot`]), or the resume from SC7 at
    /// [`sc7::RESUME_ENTRY`] once a suspend is pending ([`super::sc7`]).
    /// Anything else is refused.
    pub fn core0_released(
        &self,
        entry: u64,
        context: ColdBootContext<'_>,
        stop: &AtomicBool,
    ) -> Result<Released, ColdBootError> {
        if entry != sc7::RESUME_ENTRY {
            return self.cold_boot(entry, context, stop).map(Released::ColdBoot);
        }
        let Some((kernel_entry, kernel_context)) = self.state.lock().unwrap().suspended else {
            return Err(ColdBootError::NotSuspended);
        };
        let port = Port {
            bus: context.bus,
            ram: context.ram,
            stack: super::setup::BOOT_STACK,
        };
        let resumed = self.gic().and_then(|gic| {
            if super::setup::resume(&port, gic, self.kernel_carveouts())? {
                super::setup::final_setup(&port, stop)?;
            }
            super::setup::core_interrupts(gic, 0)
        });
        resumed.map_err(|fail| ColdBootError::Resume(fail.to_string()))?;
        let mut state = self.state.lock().unwrap();
        state.suspended = None;
        state.cores_on[0] = true;
        Ok(Released::Resume(CoreEntry::secondary(
            kernel_entry,
            kernel_context,
        )))
    }
}

#[cfg(test)]
mod tests {
    use std::{path::Path, sync::Arc};

    use aes::{
        Aes128,
        cipher::{BlockDecrypt, KeyInit},
    };

    use super::*;
    use crate::{
        gic::Gic,
        keys::KeySet,
        secmon::{KERNEL_SMC, SmcEffect, config_item, function},
        soc::{
            ahb,
            mc::{self, DeviceMemory, Mc},
            regs::RegisterFile,
            se,
        },
    };

    const FIRMWARE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../fw_dump/0100000000000819/da6ae627428f5fe36e26aafa833515a8/section0_romfs/nx"
    );
    const KEYS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../prod_newer.keys");
    const KERNEL_ELF: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../fw_kernel/kernel_normal.elf"
    );

    /// DRAM up to the end of package2's place, and a little more.
    const DRAM_SIZE: usize = 0x2B00_0000;

    /// The pieces of the firmware the bootloader leaves behind.
    struct Firmware {
        keys: KeySet,
        monitor: Vec<u8>,
        warmboot: Vec<u8>,
        package2: Vec<u8>,
    }

    /// The monitor and warmboot firmware out of package1's PK11, which
    /// package1ldr decrypts with AES-128-CBC under `package1_key_08` (the
    /// only one that gives `PK11` for this package1), the IV being the 16
    /// bytes before it at 0x6FF0. PK11: a 0x20-byte header with the
    /// section sizes at +0x04 (warmboot), +0x10 (bootloader) and +0x18
    /// (monitor), then the bootloader, the monitor and warmboot.
    fn firmware() -> Option<Firmware> {
        let paths = [
            KEYS.to_string(),
            format!("{FIRMWARE}/package1"),
            format!("{FIRMWARE}/package2"),
            KERNEL_ELF.to_string(),
        ];
        if let Some(missing) = paths.iter().find(|path| !Path::new(path).exists()) {
            eprintln!("skipped: {missing} absent");
            return None;
        }
        let text = std::fs::read_to_string(KEYS).unwrap();
        let key: [u8; 16] = text
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once('=')?;
                (name.trim() == "package1_key_08").then(|| value.trim().to_string())
            })
            .map(|hex| {
                std::array::from_fn(|index| {
                    u8::from_str_radix(&hex[index * 2..index * 2 + 2], 16).unwrap()
                })
            })
            .expect("package1_key_08 in the key file");
        let package1 = std::fs::read(format!("{FIRMWARE}/package1")).unwrap();
        let cipher = Aes128::new(&key.into());
        let mut previous: [u8; 16] = package1[0x6FF0..0x7000].try_into().unwrap();
        let mut pk11 = Vec::new();
        for block in package1[0x7000..].as_chunks::<16>().0 {
            let mut plain = aes::Block::from(*block);
            cipher.decrypt_block(&mut plain);
            pk11.extend(plain.iter().zip(previous).map(|(byte, iv)| byte ^ iv));
            previous = *block;
        }
        assert_eq!(&pk11[..4], b"PK11");
        let size = |offset: usize| {
            u32::from_le_bytes(pk11[offset..offset + 4].try_into().unwrap()) as usize
        };
        let (warmboot, bootloader, monitor) = (size(0x04), size(0x10), size(0x18));
        let monitor_start = 0x20 + bootloader;
        let warmboot_start = monitor_start + monitor;
        Some(Firmware {
            keys: KeySet::load(Path::new(KEYS)).unwrap(),
            monitor: pk11[monitor_start..warmboot_start].to_vec(),
            warmboot: pk11[warmboot_start..warmboot_start + warmboot].to_vec(),
            package2: std::fs::read(format!("{FIRMWARE}/package2")).unwrap(),
        })
    }

    struct Console {
        monitor: SecureMonitor,
        ram: Arc<GuestRam>,
        iram: Arc<Iram>,
        tzram: Arc<tzram::Tzram>,
        gic: Arc<Gic>,
        bus: Bus,
        engine: DeviceMemory,
    }

    impl Console {
        fn context(&self) -> ColdBootContext<'_> {
            ColdBootContext {
                ram: &self.ram,
                iram: &self.iram,
                bus: &self.bus,
            }
        }

        fn config(&self, item: u32) -> [u64; 8] {
            let mut x = [0; 8];
            x[0] = u64::from(function::GET_CONFIG);
            x[1] = item.into();
            assert_eq!(
                self.monitor
                    .call(0, KERNEL_SMC, &mut x, &self.bus, &self.engine),
                SmcEffect::Resume
            );
            x
        }
    }

    /// IRAM as the bootloader leaves it when it releases core 0: the
    /// monitor at its entry, warmboot at 0x4003E000, the mailbox zeroed;
    /// the blocks the monitor's set-up reaches, the interrupt controller
    /// out of reset.
    fn console(firmware: &Firmware, params: &BootParams) -> Console {
        let ram = Arc::new(GuestRam::new(crate::boot::DRAM_BASE, DRAM_SIZE).unwrap());
        let iram = Arc::new(Iram::new());
        iram.load(iram_offset(ENTRY), &firmware.monitor).unwrap();
        iram.load(iram_offset(WARMBOOT_IRAM), &firmware.warmboot)
            .unwrap();
        let mut bus = Bus::default();
        bus.map(
            "TMR",
            tmr::BASE,
            tmr::SIZE,
            Box::new(tmr::Timers::new(Box::new(|| Duration::from_micros(1234)))),
        );
        bus.map(
            "FLOW",
            HALT_COP_EVENTS & !0xFFF,
            0x1000,
            Box::new(RegisterFile::zeroed([(HALT_COP_EVENTS & 0xFFF, 0)])),
        );
        bus.map(
            "CAR",
            car::BASE,
            car::SIZE,
            Box::new(crate::soc::car::Car::power_on()),
        );
        bus.map("AHB", ahb::BASE, ahb::SIZE, Box::new(ahb::power_on()));
        bus.map(
            "APB slave security",
            0x7000_0C00,
            0x10,
            Box::new(RegisterFile::zeroed([(0, 0), (4, 0), (8, 0)])),
        );
        bus.map(
            "FUSE_RESERVED_ODM4",
            0x7000_F9D8,
            4,
            Box::new(RegisterFile::zeroed([(0, 0)])),
        );
        bus.map(
            "host1x syncpoint 160",
            0x5000_3300,
            4,
            Box::new(RegisterFile::zeroed([(0, 0)])),
        );
        bus.map(
            "MSELECT",
            crate::soc::mselect::BASE,
            4,
            Box::new(RegisterFile::zeroed([(0, 0x0300_0000)])),
        );
        let controller = Arc::new(Mc::new(params.dram));
        let tzram = Arc::new(tzram::Tzram::new());
        bus.map(
            "PMC",
            crate::soc::pmc::BASE,
            crate::soc::pmc::SIZE,
            Box::new(crate::soc::pmc::Pmc::new()),
        );
        bus.map("MC", mc::BASE, mc::SIZE, Box::new(Arc::clone(&controller)));
        bus.map(
            "TZRAM",
            tzram::BASE,
            tzram::SIZE,
            Box::new(Arc::clone(&tzram)),
        );
        // The engine as a machine powered on has it: over IRAM, TZRAM and
        // physical DRAM.
        let memory = se::BootMemory::new(Arc::clone(&iram), Arc::clone(&tzram), Arc::clone(&ram));
        bus.map(
            "SE",
            se::BASE,
            se::SIZE,
            Box::new(se::Se::new(Box::new(memory), Arc::default(), 1)),
        );
        let gic = Arc::new(Gic::new(crate::boot::CORE_COUNT, Box::new(|_| {})));
        gic.reset();
        let monitor = SecureMonitor::new(params, firmware.keys.clone());
        monitor.attach_gic(Arc::clone(&gic));
        Console {
            monitor,
            ram: Arc::clone(&ram),
            iram,
            tzram,
            gic,
            bus,
            engine: DeviceMemory::new(&controller, &ram, mc::SMMU_SE_ASID),
        }
    }

    fn register(bus: &Bus, address: u64) -> u32 {
        match bus.read(0, address, 4) {
            Access::Done(value) => value as u32,
            _ => panic!("register {address:#x}"),
        }
    }

    /// Stops the monitor's waits however the test bootloader ends, so a
    /// failing test fails instead of hanging.
    struct StopOnExit<'a>(&'a AtomicBool);

    impl Drop for StopOnExit<'_> {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Relaxed);
        }
    }

    /// The bootloader's part once core 0 runs, as the 23.5.0 NX bootloader
    /// does it: wait for the monitor, then states 1, 2 and, with package2
    /// in DRAM, 4. Gives up after ten seconds; sets `stop` when it ends.
    fn bootloader(console: &Console, package2: &[u8], stop: &AtomicBool) {
        let (iram, ram) = (&*console.iram, &*console.ram);
        let _stop = StopOnExit(stop);
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while read_word(iram, MONITOR_STATE) == 0 {
            assert!(
                std::time::Instant::now() < deadline,
                "the monitor never set its state"
            );
            thread::sleep(POLL_INTERVAL);
        }
        let mut image = vec![0xFF; IMAGE_SIZE];
        iram.copy_out(iram_offset(ENTRY), &mut image).unwrap();
        assert!(
            image.iter().all(|&byte| byte == 0),
            "monitor image erased before its state says so"
        );
        iram.write(BOOTLOADER_STATE, 4, 1).unwrap();
        iram.write(BOOTLOADER_STATE, 4, 2).unwrap();
        ram.write(PACKAGE2, package2).unwrap();
        iram.write(BOOTLOADER_STATE, 4, 4).unwrap();
        // package1ldr then parks the BPMP.
        let parked = console.bus.write(0, HALT_COP_EVENTS, 4, 0x5000_0000);
        assert!(matches!(parked, Access::Done(())));
    }

    #[test]
    fn places_the_kernel_the_elf_path_enters() {
        let Some(firmware) = firmware() else {
            return;
        };
        let console = console(&firmware, &BootParams::default());
        let stop = AtomicBool::new(false);
        let booted = thread::scope(|scope| {
            scope.spawn(|| bootloader(&console, &firmware.package2, &stop));
            console.monitor.cold_boot(ENTRY, console.context(), &stop)
        })
        .unwrap();

        let elf = KernelImage::load(Path::new(KERNEL_ELF)).unwrap();
        assert_eq!(booted.kernel.load_address(), elf.load_address());
        assert_eq!(booted.entry, CoreEntry::primary(elf.load_address()));
        assert_eq!(
            Sha256::digest(booted.kernel.bytes()),
            Sha256::digest(elf.bytes())
        );
        assert_eq!(booted.kernel.ini1, elf.ini1);
        assert!(elf.ini1.is_some());
        let placed = read_dram(&console.ram, elf.load_address(), elf.bytes().len()).unwrap();
        assert_eq!(Sha256::digest(&placed), Sha256::digest(elf.bytes()));

        assert_eq!(read_word(&console.iram, START_TIME), 1234);
        assert_eq!(read_word(&console.iram, MONITOR_STATE), 1);
        let warmboot = read_dram(&console.ram, WARMBOOT_DRAM, firmware.warmboot.len()).unwrap();
        assert_eq!(warmboot, firmware.warmboot);
        for (address, bytes) in DRAM_MARKERS {
            assert_eq!(
                read_dram(&console.ram, address, bytes.len()).unwrap(),
                bytes
            );
        }

        // What the warm path relies on.
        let pmc = crate::soc::pmc::BASE;
        assert_eq!(register(&console.bus, pmc + 0x054), 0x8000_D000);
        assert_eq!(register(&console.bus, pmc + 0x334) & 0x10, 0x10);
        assert_eq!(register(&console.bus, pmc + 0x360), 0x339);
        assert_eq!(register(&console.bus, pmc + 0x368), 0x7C01_2800);
        assert_eq!(register(&console.bus, sc7::MC_SECURITY_CFG3), 3);
        assert_eq!(register(&console.bus, HALT_COP_EVENTS), sc7::COP_PARKED);
        // The activity monitor clocked and out of reset, for sleep.
        assert_eq!(
            register(&console.bus, CLK_OUT_ENB_V) & ACTMON_BIT,
            ACTMON_BIT
        );
        assert_eq!(register(&console.bus, RST_DEVICES_V) & ACTMON_BIT, 0);
        assert_eq!(register(&console.bus, CLK_SOURCE_ACTMON), ACTMON_SOURCE);
        // The key slots as the warmboot firmware checks them: slots 0 to 5
        // read-protected, reading 0; 6 to 15 per-key secure, all ones;
        // the RSA slots all ones.
        let se_register = |offset: u64| se::BASE + offset;
        assert_eq!(register(&console.bus, se_register(0x280)), 0x003F);
        assert_eq!(register(&console.bus, se_register(0x284)), 0x6A);
        assert_eq!(register(&console.bus, se_register(0x284 + 8 * 4)), 0x2A);
        assert_eq!(register(&console.bus, se_register(0x284 + 15 * 4)), 0);
        for (slot, expected) in [(0, 0), (5, 0), (6, u32::MAX), (15, u32::MAX)] {
            let selected = console.bus.write(0, se_register(0x31C), 4, slot << 4);
            assert!(matches!(selected, Access::Done(())));
            assert_eq!(register(&console.bus, se_register(0x320)), expected);
        }
        assert_eq!(register(&console.bus, se_register(0x40C)), 0);
        assert_eq!(register(&console.bus, se_register(0x410)), 2);
        // The resident image in TZRAM, and the SC7 stub in it.
        let mut resident = vec![0; RESIDENT_SIZE];
        console
            .tzram
            .copy_out((RESIDENT - tzram::BASE) as u32, &mut resident)
            .unwrap();
        assert_eq!(
            hex(&Sha256::digest(&resident)),
            "0904f151081c92300fe03d4abd6531727d0d9a247db9c75bb943b650d21eb70f"
        );
        let mut stub = vec![0; sc7::STUB_SIZE];
        console
            .tzram
            .copy_out((sc7::STUB - tzram::BASE) as u32, &mut stub)
            .unwrap();
        assert_eq!(
            hex(&Sha256::digest(&stub)),
            "a851aeca06065f8209cad0e788b970125673764ebcd82409e0aa599dbaf61cf9"
        );

        // A production console's answers, which the defaults equal.
        assert_eq!(console.config(config_item::MEMORY_MODE)[..2], [0, 1]);
        assert_eq!(
            console.config(config_item::KERNEL_CONFIGURATION)[..2],
            [0, 0]
        );
        assert_eq!(
            console.config(config_item::IS_DEVELOPMENT_FUNCTION_ENABLED)[..2],
            [0, 0]
        );
        assert_eq!(console.config(config_item::IS_RECOVERY_BOOT)[..2], [0, 0]);
        assert_eq!(
            console.config(config_item::PACKAGE2_HASH)[0],
            super::super::STATUS_INVALID_ARGUMENT
        );
    }

    #[test]
    fn recovery_boot_hashes_package2_and_options_override() {
        let Some(firmware) = firmware() else {
            return;
        };
        let params = BootParams {
            dram: crate::boot::DramSize::Gb6,
            development_function_enabled: true,
            ..BootParams::default()
        };
        let console = console(&firmware, &params);
        console.iram.write(RECOVERY_BOOT, 4, 1).unwrap();
        console.iram.write(BOOTLOADER_STATE, 4, 4).unwrap();
        let parked = console.bus.write(0, HALT_COP_EVENTS, 4, 0x5000_0000);
        assert!(matches!(parked, Access::Done(())));
        console.ram.write(PACKAGE2, &firmware.package2).unwrap();
        console
            .monitor
            .cold_boot(ENTRY, console.context(), &AtomicBool::new(false))
            .unwrap();

        assert_eq!(console.config(config_item::IS_RECOVERY_BOOT)[..2], [0, 1]);
        let x = console.config(config_item::PACKAGE2_HASH);
        let hash: Vec<u8> = x[1..5].iter().flat_map(|word| word.to_le_bytes()).collect();
        let size = 0x2C_0200; // what the header's counter gives
        assert_eq!(hash[..], Sha256::digest(&firmware.package2[..size])[..]);
        assert_eq!(console.config(config_item::MEMORY_MODE)[..2], [0, 0x11]);
        assert_eq!(
            console.config(config_item::IS_DEVELOPMENT_FUNCTION_ENABLED)[..2],
            [0, 1]
        );
    }

    #[test]
    fn refuses_other_entries_and_stops_while_waiting() {
        let Some(firmware) = firmware() else {
            return;
        };
        let console = console(&firmware, &BootParams::default());
        let stop = AtomicBool::new(false);
        assert!(matches!(
            console
                .monitor
                .cold_boot(0x4001_0000, console.context(), &stop),
            Err(ColdBootError::Entry(0x4001_0000))
        ));
        stop.store(true, Ordering::Relaxed);
        let error = console
            .monitor
            .cold_boot(ENTRY, console.context(), &stop)
            .unwrap_err();
        assert!(matches!(error, ColdBootError::Stopped { waiting_for: 1 }));
        assert_eq!(
            error.to_string(),
            "secure monitor cold boot: stopped waiting for the bootloader's state 1 \
             (boot config loaded)"
        );
        // The image is gone now, as on the console.
        assert!(matches!(
            console.monitor.cold_boot(ENTRY, console.context(), &stop),
            Err(ColdBootError::NotTheMonitor)
        ));
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    /// A console cold-booted, then as the warmboot firmware leaves it when
    /// it releases core 0 after SC7: the kernel suspended,
    /// `MC_SECURITY_CFG3` at `security_cfg3` (0: the SoC lost power, and
    /// its interrupt controller with it), the BPMP halting itself.
    fn asleep(security_cfg3: u32) -> Option<Console> {
        let firmware = firmware()?;
        let console = console(&firmware, &BootParams::default());
        let stop = AtomicBool::new(false);
        thread::scope(|scope| {
            scope.spawn(|| bootloader(&console, &firmware.package2, &stop));
            console.monitor.cold_boot(ENTRY, console.context(), &stop)
        })
        .unwrap();
        {
            let mut state = console.monitor.state.lock().unwrap();
            state.cores_on[0] = false;
            state.suspended = Some((0x8000_1234, 7));
        }
        if security_cfg3 == 0 {
            console.gic.reset();
        }
        for (address, value) in [
            (sc7::MC_SECURITY_CFG3, security_cfg3),
            (HALT_COP_EVENTS, 0x5000_0000),
        ] {
            let written = console.bus.write(0, address, 4, value.into());
            assert!(matches!(written, Access::Done(())));
        }
        Some(console)
    }

    #[test]
    fn resume_redoes_the_security_set_up_after_a_reset_and_enters_the_saved_entry() {
        let Some(console) = asleep(0) else {
            return;
        };
        let bus = &console.bus;
        let stop = AtomicBool::new(false);
        let released = console
            .monitor
            .core0_released(sc7::RESUME_ENTRY, console.context(), &stop)
            .unwrap();
        assert!(matches!(
            released,
            Released::Resume(entry) if entry == CoreEntry::secondary(0x8000_1234, 7)
        ));
        let pmc = crate::soc::pmc::BASE;
        assert_eq!(register(bus, pmc + 0x368), 0x7C01_2800);
        assert_eq!(register(bus, sc7::MC_SECURITY_CFG3), 3);
        assert_eq!(register(bus, HALT_COP_EVENTS), sc7::COP_PARKED);
        // The activity monitor clocked again from CLK_M.
        assert_eq!(register(bus, car::BASE + 0x3E8), 0xC000_0000);
        // AHB arbitration and MSELECT as the reset vector leaves them, the
        // BPMP off the arbiter as the end of the set-up does.
        assert_eq!(register(bus, ahb::BASE + 0x004), 0x4_0062);
        assert_eq!(register(bus, ahb::BASE + 0x008), 0xE000_0001);
        assert_eq!(register(bus, crate::soc::mselect::BASE), 0x3800_0000);
        // Carveouts, APB security, the scrubbed scratch locked.
        assert_eq!(register(bus, mc::BASE + 0xC08), 0x0400_0006);
        assert_eq!(register(bus, mc::BASE + 0xC58), 0x0440_167E);
        assert_eq!(register(bus, 0x7000_0C00), 0x0050_6244);
        assert_eq!(register(bus, 0x7000_0C04), 0x8370_0510);
        assert_eq!(register(bus, pmc + 0x004) & 0xFF000, 0xFF000);
        // The interrupt controller: the kernel's interrupts in Group 1,
        // the monitor's in Group 0.
        let groups = |core: usize, register: u64| {
            console
                .gic
                .secure_distributor_read(core, 0x080 + register * 4)
                .unwrap()
        };
        // SGIs 0 to 15 and PPIs 25 to 31 exist.
        assert_eq!(groups(0, 0), 0xFE00_FFFF & !(1 << 28));
        assert_eq!(groups(0, 1), u32::MAX);
        assert_eq!(groups(0, 2), !(1 << (90 - 64) | 1 << (77 - 64)));
        assert_eq!(groups(1, 0), 0, "core 1 is set up when it starts");
        assert!(console.monitor.state.lock().unwrap().cores_on[0]);
        // Nothing is pending any more.
        let again = console
            .monitor
            .core0_released(sc7::RESUME_ENTRY, console.context(), &stop);
        assert!(matches!(again, Err(ColdBootError::NotSuspended)));
        assert_eq!(
            again.unwrap_err().to_string(),
            "secure monitor resume: core 0 released at 0x7c012800 with no suspend pending"
        );
    }

    #[test]
    fn resume_panics_as_the_monitor_does_on_a_different_key_digest() {
        let Some(console) = asleep(0) else {
            return;
        };
        let digest = (super::super::setup::DIGEST - tzram::BASE) as u32;
        console.tzram.load(digest, &[0x5A; 16]).unwrap();
        let error = console
            .monitor
            .core0_released(
                sc7::RESUME_ENTRY,
                console.context(),
                &AtomicBool::new(false),
            )
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "secure monitor resume: the monitor panics: the key slots' digest differs \
             from the cold boot's"
        );
    }

    #[test]
    fn resume_without_a_reset_leaves_the_soc_alone() {
        let Some(console) = asleep(3) else {
            return;
        };
        let released = console
            .monitor
            .core0_released(
                sc7::RESUME_ENTRY,
                console.context(),
                &AtomicBool::new(false),
            )
            .unwrap();
        assert!(matches!(released, Released::Resume(_)));
        assert_eq!(register(&console.bus, HALT_COP_EVENTS), 0x5000_0000);
        assert_eq!(register(&console.bus, ahb::BASE + 0x004), 2);
        assert_eq!(
            register(&console.bus, crate::soc::mselect::BASE),
            0x0300_0000
        );
    }

    #[test]
    fn lz4_blocks_decode_and_malformed_ones_are_refused() {
        // "abc" then a match of 6 at distance 3, then "d".
        let block = [0x32, b'a', b'b', b'c', 3, 0, 0x10, b'd'];
        assert_eq!(lz4_block(&block, 64).unwrap(), b"abcabcabcd");
        assert_eq!(lz4_block(&block, 8), None);
        assert_eq!(lz4_block(&[0x30, b'a', b'b', b'c', 9, 0], 64), None);
    }
}
