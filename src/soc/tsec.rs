//! TSEC, the Tegra security co-processor: a Falcon microcontroller with
//! crypto extensions, `0x40000` bytes at `0x5450_0000` (Switchbrew "TSEC":
//! THI at `+0`, the Falcon at `+0x1000`, its SCP at `+0x1400`, TFBIF at
//! `+0x1600`).
//!
//! # The host interface
//!
//! What the BPMP drives is the Falcon's public host interface, with the
//! register layout and bit names of Switchbrew's "TSEC" register table
//! (the same layout as the nouveau falcon registers):
//!
//! * interrupts: `IRQSSET`/`IRQSCLR` set and clear `IRQSTAT` bits,
//!   `IRQMSET`/`IRQMCLR` set and clear `IRQMASK` bits, `IRQMODE` and
//!   `IRQDEST` hold what is written. The engine's interrupt output is not
//!   wired to an interrupt controller: no public source gives its number
//!   on this SoC (Linux's `tegra210.dtsi` gives the node none), and
//!   neither package1ldr nor anything after it enables it \[inference\].
//! * `MAILBOX0`/`MAILBOX1`, `ITFEN`, `BOOTVEC` hold what is written.
//! * `IDLESTATE` reads 0: the processor is never seen busy (below).
//! * `DMACTL`: the scrubbing bits read 0, the memories are scrubbed (zero)
//!   at once. `REQUIRE_CTX` set makes transfers refuse: there is no
//!   context model.
//! * The DMA engine: `DMATRFCMD` reads `IDLE` (bit 1) at all times, as
//!   every transfer completes within its command write. The external
//!   address is `DMATRFBASE << 8` plus `DMATRFFBOFFS`, the Falcon-side
//!   offset `DMATRFMOFFS`. A code (`IMEM`) transfer moves one 256-byte
//!   block whatever the size field says \[inference: envytools' Falcon
//!   DMA notes; package1ldr issues 0x10, size field 0, with a 0x100
//!   stride\]; a data transfer moves `4 << SIZE` bytes (size field up to
//!   6), either way (`WRITE`). The `SEC` and `CTXDMA` fields must be 0:
//!   secure transfers and context DMA are not modeled.
//! * `IMEMC`/`IMEMD`/`IMEMT` and `DMEMC`/`DMEMD`, port 0: programmed I/O
//!   at the byte offset in bits 15:2 of the control register, with
//!   auto-increment on write (bit 24) and read (bit 25). Secure uploads
//!   (bits 28-31) are refused.
//! * `CPUCTL`: `STARTCPU` alone starts the processor (see below); the
//!   register reads `HALTED` once the firmware has finished, 0 before.
//!
//! Everything else in the block (THI, SCP, TFBIF, `HWCFG` whose memory
//! sizes are not published for this unit, the other Falcon registers) is
//! refused with a note. Both memories are 64 KiB here, the reach of the
//! 16-bit offsets, because their real sizes are not established.
//!
//! The DMA engine reaches memory through the memory controller, without
//! the SMMU (a TSEC address space enabled in `MC_SMMU_TSEC_ASID` is
//! refused): DRAM directly, and IRAM and MMIO through the controller's AHB
//! redirect aperture `MC_IRAM_BOM..MC_IRAM_TOM`, which package1ldr opens
//! from 0x4000_0000 to 0x8000_0000 so that the TSEC reaches IRAM and all
//! MMIO (Switchbrew "Package1", 6.2.0+). That side of the bus is the
//! machine's, through [`TsecSoc`].
//!
//! # The firmware: modeled by what it leaves behind
//!
//! package1ldr (BPMP) loads NVIDIA's and Nintendo's signed TSEC firmware
//! from its own data (IRAM 0x4001_0E00, 0x3300 bytes, 51 IMEM transfers),
//! sets MAILBOX1 to 0 and MAILBOX0 to 1, writes the magic 0x34C2E1DA to
//! HOST1X+0x3300, sets BOOTVEC to 0, starts the processor and spins
//! forever: it reads nothing back (package1ldr `tsec_boot`, text 0xBB4).
//! The firmware's authenticated parts run in Heavy Secure mode, keyed by
//! secrets inside the chip, and are encrypted, so they cannot run here.
//! No Falcon instruction is executed. Instead, starting the processor
//! with exactly the firmware of 23.5.0 (SHA-256 of IMEM `0..0x3300`,
//! [`FIRMWARE_SHA256`]) and BOOTVEC 0 does at once what Switchbrew's
//! account of that firmware ("TSEC Firmware", SecureBootLdr and
//! SecureBoot, 8.1.0+) says it leaves behind:
//!
//! 1. The checks Switchbrew lists whose expected values are public are
//!    made, each a refusal if it fails: `FUSE_SKU_INFO` is 0x83,
//!    `CLK_RST_CONTROLLER_CLK_SOURCE_TSEC` is 2, the AHB aperture is
//!    0x4000_0000..0x8000_0000, and HOST1X+0x3300 holds 0x34C2E1DA
//!    \[assumption: SecureBoot's failure path, random keyslots and a BPMP
//!    left spinning, is not modeled\]. The other anti-tamper tests (random
//!    CAR toggles, memory-transfer and TZRAM tests, the SMMU/fuse
//!    virtualization probe) are not made.
//! 2. PK11 at IRAM 0x4001_6FE0 (size word, IV at +0x10, ciphertext at
//!    +0x20, CMAC trailer after it) is checked with AES-CMAC under
//!    `package1_mac_key_08` and decrypted in place with AES-128-CBC under
//!    `package1_key_08`; its header is checked and its NX-bootloader entry
//!    found exactly as package1ldr's own (unreachable) parser does
//!    (text 0xA10/0xA80; the key and mode were established by decrypting
//!    this package1 with the key set). A size above 0x287F0, a CMAC mismatch or a
//!    bad header is refused \[inference: the firmware checks; what it does
//!    then is not public\].
//! 3. Security engine keyslot 0x0D gets `tsec_root_key_02` and keyslot
//!    0x0C `tsec_key`, in that order.
//! 4. `EVP_COP_RESET_VECTOR` (0x6000_F200) gets the entry (`set_excp_vec`)
//!    \[assumption: only the reset vector; Switchbrew names the call
//!    without saying which vectors\].
//! 5. MAILBOX1 reads 0xB0B0B0B0 (SecureBootLdr's success write), DMEM is
//!    cleared, the processor is halted (`CPUCTL` `HALTED`, `IRQSTAT`
//!    `HALT` pending).
//! 6. The machine restarts the BPMP in ARM state at the entry
//!    ([`TsecSoc::restart_bpmp`]). Switchbrew has SecureBoot halt the BPMP
//!    through the flow controller, set its vectors and resume it; a BPMP
//!    spinning with interrupts masked leaves its loop only through a reset,
//!    so the restart is \[inference\] and what triggers it on a console
//!    (a CAR COP reset or the flow controller) is not public. The NX
//!    bootloader's entry sets its mode and registers itself, so any
//!    register file is equivalent (its first instructions, at 0x4002_E104).
//!
//! Any other firmware (another digest, another extent of IMEM written,
//! another boot vector), a second start, and DMA or memory-port access
//! after the run are refused: what the firmware rewrote and locked in its
//! memories is not known.
//!
//! Timing: all of this happens within the BPMP's `CPUCTL` write, on the
//! writing core's thread, before the write returns. On a console it takes
//! an unpublished time, which package1ldr never measures (it spins without
//! reading a timer); the restart request is the machine's to apply after
//! the current instruction.
//!
//! # Honesty limit: what comes from the key set
//!
//! On a console every value the firmware leaves is computed inside the
//! TSEC from hardware secrets. Here the following come from the key set
//! ([`crate::keys::KeySet`]), by name, as published derivations of those
//! secrets; none is computed by the model from TSEC hardware:
//!
//! * `tsec_key` (keyslot 0x0C): Keygen's console-unique output (from the
//!   Falcon's csecrets and the KFUSE private key). It is this console's
//!   only if the key set is.
//! * `tsec_root_key_02` (keyslot 0x0D), `package1_key_08` (PK11
//!   decryption) and `package1_mac_key_08` (PK11 CMAC): on hardware the
//!   SecureBoot stage derives them (csigenc of `tsec_root_kek_02`,
//!   `package1_kek_02`, `package1_mac_kek_02` with the image's auth
//!   signature at IMEM 0x2F00); the keks never leave the TSEC.
//!
//! Not modeled, refused where it would be read: the SOR1 HDCP registers
//! Keygen writes the TSEC key into, the SCP's state, `car_init` and
//! `ccplex_init`'s side effects, the ACL and lock state of keyslots
//! 0x0C/0x0D, the PK11 header, IV and CMAC bytes around the plaintext
//! (left as they were).

use std::{
    fmt,
    sync::{Arc, Mutex, OnceLock},
};

use aes::{
    Aes128,
    cipher::{BlockDecrypt, KeyInit},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{Notes, mc::DeviceMemory};
use crate::{
    bus::MmioDevice,
    keys::{Key, KeySet},
    secmon::crypto::cmac,
    snapshot::{Snapshot, SnapshotError, decode, encode},
};

pub const BASE: u64 = 0x5450_0000;
pub const SIZE: u64 = 0x4_0000;

/// The TSEC's address space register in the memory controller (the public
/// Linux `tegra210.c` swgroup table: `tsec`, register 0x294).
pub const SMMU_TSEC_ASID: u64 = 0x294;

/// Falcon registers, relative to [`BASE`] (Switchbrew "TSEC").
const IRQSSET: u64 = 0x1000;
const IRQSCLR: u64 = 0x1004;
const IRQSTAT: u64 = 0x1008;
const IRQMODE: u64 = 0x100C;
const IRQMSET: u64 = 0x1010;
const IRQMCLR: u64 = 0x1014;
const IRQMASK: u64 = 0x1018;
const IRQDEST: u64 = 0x101C;
const MAILBOX0: u64 = 0x1040;
const MAILBOX1: u64 = 0x1044;
const ITFEN: u64 = 0x1048;
const IDLESTATE: u64 = 0x104C;
const CPUCTL: u64 = 0x1100;
const BOOTVEC: u64 = 0x1104;
const DMACTL: u64 = 0x110C;
const DMATRFBASE: u64 = 0x1110;
const DMATRFMOFFS: u64 = 0x1114;
const DMATRFCMD: u64 = 0x1118;
const DMATRFFBOFFS: u64 = 0x111C;
const IMEMC: u64 = 0x1180;
const IMEMD: u64 = 0x1184;
const IMEMT: u64 = 0x1188;
const DMEMC: u64 = 0x11C0;
const DMEMD: u64 = 0x11C4;

const CPUCTL_STARTCPU: u32 = 1 << 1;
const CPUCTL_HALTED: u32 = 1 << 4;
const IRQ_HALT: u32 = 1 << 4;
const DMACTL_REQUIRE_CTX: u32 = 1 << 0;
const DMACTL_SCRUBBING: u32 = 3 << 1;
const DMATRFCMD_FULL: u32 = 1 << 0;
const DMATRFCMD_IDLE: u32 = 1 << 1;
const DMATRFCMD_SEC: u32 = 3 << 2;
const DMATRFCMD_IMEM: u32 = 1 << 4;
const DMATRFCMD_WRITE: u32 = 1 << 5;
const DMATRFCMD_SIZE_SHIFT: u32 = 8;
const DMATRFCMD_SIZE: u32 = 7 << DMATRFCMD_SIZE_SHIFT;
const DMATRFCMD_CTXDMA: u32 = 7 << 12;
/// `DMATRFMOFFS` bits 15:0.
const DMATRFMOFFS_OFFS: u32 = 0xFFFF;
/// Memory port control: byte offset (`OFFS` bits 7:2, `BLK` bits 15:8),
/// auto-increment on write and on read.
const PORT_OFFSET: u32 = 0xFFFC;
const PORT_AINCW: u32 = 1 << 24;
const PORT_AINCR: u32 = 1 << 25;
/// The bits of a port control register this model knows the meaning of.
const PORT_MODELED: u32 = PORT_OFFSET | PORT_AINCW | PORT_AINCR;
/// `IMEMT` bits 15:0.
const IMEMT_TAG: u32 = 0xFFFF;
/// Size of each memory here: what the 16-bit offsets reach.
const MEMORY_SIZE: usize = 0x1_0000;
/// A code transfer's block.
const IMEM_BLOCK: usize = 0x100;

/// Memory controller registers (public Linux/hekate `mc` offsets).
const MC_IRAM_BOM: u64 = 0x65C;
const MC_IRAM_TOM: u64 = 0x660;
const MC_SMMU_ASID_ENABLE: u32 = 1 << 31;
const DRAM_BASE: u64 = 0x8000_0000;

/// SHA-256 of the TSEC firmware 23.5.0's Erista package1ldr loads: IMEM
/// `0..0x3300`, from package1 offset 0xE00 (`tsec_boot`'s 51 transfers
/// from IRAM 0x4001_0E00).
pub const FIRMWARE_SHA256: &str =
    "3b71f74c88cff331876ca8585b4e4dbe37d07450c24d043ce02a49ef889d5b38";
const FIRMWARE_SIZE: u32 = 0x3300;

/// Inputs the firmware checks (Switchbrew "TSEC Firmware", "TSEC").
const FUSE_SKU_INFO: u32 = 0x7000_F910;
const SKU_INFO: u32 = 0x83;
const CLK_SOURCE_TSEC: u32 = 0x6000_61F4;
const TSEC_CLOCK_SOURCE: u32 = 2;
/// HOST1X+0x3300, where package1ldr leaves the magic \[inference: by
/// [`super::host1x`]'s layout, syncpoint 160's value register\].
const HOST1X_MAGIC_AT: u32 = 0x5000_3300;
const HOST1X_MAGIC: u32 = 0x34C2_E1DA;
const IRAM_APERTURE: (u32, u32) = (0x4000_0000, 0x8000_0000);

/// The PK11 blob in IRAM: size word, IV at +0x10, ciphertext at +0x20,
/// then its CMAC.
const PK11_BLOB: u32 = 0x4001_6FE0;
const PK11_IV: usize = 0x10;
const PK11_DATA: usize = 0x20;
const PK11_HEADER: u64 = 0x20;
/// package1ldr's bound on the size word (its `find_pk11`, text 0xA10).
const PK11_MAX_SIZE: usize = 0x287F0;

const TSEC_KEY_SLOT: usize = 0x0C;
const TSEC_ROOT_KEY_SLOT: usize = 0x0D;
const EVP_COP_RESET_VECTOR: u32 = 0x6000_F200;
/// SecureBootLdr's success value in MAILBOX1 (Switchbrew "SCRATCH1").
const BOOT_SUCCESS: u32 = 0xB0B0_B0B0;

/// The rest of the SoC as the TSEC reaches it and acts on it.
///
/// Called from within the BPMP's write of `CPUCTL` and of `DMATRFCMD`, on
/// the thread that made it.
pub trait TsecSoc: Send + Sync {
    /// Reads `buffer.len()` bytes at `address` on the AHB side of the
    /// memory controller's IRAM aperture: IRAM and MMIO. MMIO registers
    /// are read as single aligned 4-byte accesses.
    fn ahb_read(&self, address: u32, buffer: &mut [u8]) -> Option<()>;
    /// Writes `data` at `address` there, likewise.
    fn ahb_write(&self, address: u32, data: &[u8]) -> Option<()>;
    /// Loads `key` into AES keyslot `slot` of the security engine.
    fn load_se_keyslot(&self, slot: usize, key: &Key) -> Option<()>;
    /// Restarts the BPMP in ARM state at `entry`, once the current
    /// instruction has finished.
    fn restart_bpmp(&self, entry: u32) -> Option<()>;
}

/// The key set's values the modeled firmware leaves behind or uses; see
/// the module's honesty limit.
#[derive(Clone)]
pub struct SecureBootKeys {
    tsec_key: Key,
    tsec_root_key: Key,
    package1_key: Key,
    package1_mac_key: Key,
}

impl SecureBootKeys {
    /// The names the 23.5.0 firmware's values go by: `tsec_key`,
    /// `tsec_root_key_02`, `package1_key_08`, `package1_mac_key_08`.
    /// `Err` lists the missing names.
    pub fn from_key_set(keys: &KeySet) -> Result<Self, String> {
        let found = [
            ("tsec_key", keys.tsec_key()),
            ("tsec_root_key_02", keys.tsec_root_key(2)),
            ("package1_key_08", keys.package1_key(8)),
            ("package1_mac_key_08", keys.package1_mac_key(8)),
        ];
        match found {
            [
                (_, Some(tsec_key)),
                (_, Some(tsec_root_key)),
                (_, Some(package1_key)),
                (_, Some(package1_mac_key)),
            ] => Ok(Self {
                tsec_key: *tsec_key,
                tsec_root_key: *tsec_root_key,
                package1_key: *package1_key,
                package1_mac_key: *package1_mac_key,
            }),
            _ => Err(found
                .iter()
                .filter(|(_, key)| key.is_none())
                .map(|(name, _)| *name)
                .collect::<Vec<_>>()
                .join(", ")),
        }
    }
}

/// Key values stay out of logs.
impl fmt::Debug for SecureBootKeys {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SecureBootKeys { .. }")
    }
}

struct Wiring {
    soc: Arc<dyn TsecSoc>,
    keys: Result<SecureBootKeys, String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum Cpu {
    /// Out of reset, never started.
    Stopped,
    /// The 23.5.0 firmware has done its work and halted.
    Halted,
}

#[derive(Serialize, Deserialize)]
struct State {
    irq_status: u32,
    irq_mode: u32,
    irq_mask: u32,
    irq_dest: u32,
    mailbox: [u32; 2],
    itfen: u32,
    bootvec: u32,
    dma_control: u32,
    dma_base: u32,
    dma_memory_offset: u32,
    dma_external_offset: u32,
    /// The last command's fields, without the status bits.
    dma_command: u32,
    imem_port: u32,
    imem_tag: u32,
    dmem_port: u32,
    imem: Vec<u8>,
    dmem: Vec<u8>,
    /// IMEM bytes from 0 to the highest the host has written.
    imem_loaded: u32,
    cpu: Cpu,
}

impl Default for State {
    fn default() -> Self {
        Self {
            irq_status: 0,
            irq_mode: 0,
            irq_mask: 0,
            irq_dest: 0,
            mailbox: [0; 2],
            itfen: 0,
            bootvec: 0,
            dma_control: 0,
            dma_base: 0,
            dma_memory_offset: 0,
            dma_external_offset: 0,
            dma_command: 0,
            imem_port: 0,
            imem_tag: 0,
            dmem_port: 0,
            imem: vec![0; MEMORY_SIZE],
            dmem: vec![0; MEMORY_SIZE],
            imem_loaded: 0,
            cpu: Cpu::Stopped,
        }
    }
}

pub struct Tsec {
    state: Mutex<State>,
    /// The memory controller and DRAM as the TSEC's DMA engine reaches them.
    memory: DeviceMemory,
    wiring: OnceLock<Wiring>,
    notes: Arc<Notes>,
}

impl Tsec {
    /// `memory` is the TSEC's memory client ([`SMMU_TSEC_ASID`]).
    pub fn new(memory: DeviceMemory, notes: Arc<Notes>) -> Self {
        Self {
            state: Mutex::default(),
            memory,
            wiring: OnceLock::new(),
            notes,
        }
    }

    /// Connects the TSEC to the rest of the SoC and gives it the key set's
    /// values ([`SecureBootKeys`]); a key set without them makes the start
    /// of the firmware refuse, naming them. Once per machine.
    pub fn connect(&self, soc: Arc<dyn TsecSoc>, keys: &KeySet) {
        let wiring = Wiring {
            soc,
            keys: SecureBootKeys::from_key_set(keys),
        };
        if self.wiring.set(wiring).is_err() {
            panic!("the TSEC is connected once");
        }
    }

    fn served<T: Default>(&self, result: Result<T, String>) -> Option<T> {
        match result {
            Ok(value) => Some(value),
            Err(note) => {
                self.notes.add(format!("TSEC: {note}"));
                self.notes.survey().then(T::default)
            }
        }
    }

    fn soc(&self, what: &str) -> Result<&Wiring, String> {
        self.wiring
            .get()
            .ok_or_else(|| format!("{what} before the machine connected the TSEC to the SoC"))
    }

    fn mc_register(&self, offset: u64) -> Result<u32, String> {
        MmioDevice::read(self.memory.mc(), 0, offset, 4)
            .map(|value| value as u32)
            .ok_or_else(|| format!("memory controller register {offset:#x} unreadable"))
    }

    /// Where a transfer of `len` bytes at external `address` goes: `Some`
    /// AHB address inside the IRAM aperture, `None` for DRAM.
    fn route(&self, address: u64, len: usize) -> Result<Option<u32>, String> {
        if self.mc_register(SMMU_TSEC_ASID)? & MC_SMMU_ASID_ENABLE != 0 {
            return Err(format!(
                "DMA at {address:#x} through the SMMU (MC_SMMU_TSEC_ASID enabled) is not modeled"
            ));
        }
        let end = address + len as u64;
        let (bottom, top) = (
            u64::from(self.mc_register(MC_IRAM_BOM)?),
            u64::from(self.mc_register(MC_IRAM_TOM)?),
        );
        if bottom <= address && end <= top {
            if address < BASE + SIZE && BASE < end {
                return Err(format!("DMA at {address:#x} into the TSEC's own registers"));
            }
            return Ok(Some(address as u32));
        }
        if address >= DRAM_BASE {
            return Ok(None);
        }
        Err(format!(
            "DMA at {address:#x}..{end:#x}: outside DRAM and the memory controller's \
             IRAM aperture {bottom:#x}..{top:#x}"
        ))
    }

    fn dma_read(&self, address: u64, buffer: &mut [u8]) -> Result<(), String> {
        let done = match self.route(address, buffer.len())? {
            Some(ahb) => self.soc("DMA")?.soc.ahb_read(ahb, buffer),
            None => self.memory.ram().read(address, buffer),
        };
        done.ok_or_else(|| {
            format!(
                "DMA read of {:#x} bytes at {address:#x} failed",
                buffer.len()
            )
        })
    }

    fn dma_write(&self, address: u64, data: &[u8]) -> Result<(), String> {
        let done = match self.route(address, data.len())? {
            Some(ahb) => self.soc("DMA")?.soc.ahb_write(ahb, data),
            None => self.memory.ram().write(address, data),
        };
        done.ok_or_else(|| {
            format!(
                "DMA write of {:#x} bytes at {address:#x} failed",
                data.len()
            )
        })
    }

    fn read_register(&self, offset: u64) -> Result<u32, String> {
        let mut s = self.state.lock().unwrap();
        Ok(match offset {
            IRQSTAT => s.irq_status,
            IRQMODE => s.irq_mode,
            IRQMASK => s.irq_mask,
            IRQDEST => s.irq_dest,
            MAILBOX0 => s.mailbox[0],
            MAILBOX1 => s.mailbox[1],
            ITFEN => s.itfen,
            IDLESTATE => 0,
            CPUCTL => match s.cpu {
                Cpu::Stopped => 0,
                Cpu::Halted => CPUCTL_HALTED,
            },
            BOOTVEC => s.bootvec,
            DMACTL => s.dma_control & !DMACTL_SCRUBBING,
            DMATRFBASE => s.dma_base,
            DMATRFMOFFS => s.dma_memory_offset,
            DMATRFFBOFFS => s.dma_external_offset,
            DMATRFCMD => s.dma_command | DMATRFCMD_IDLE,
            IMEMC => s.imem_port,
            IMEMT => s.imem_tag,
            DMEMC => s.dmem_port,
            IMEMD | DMEMD => {
                s.running_out("memory port read")?;
                let s = &mut *s;
                let (port, memory) = if offset == IMEMD {
                    (&mut s.imem_port, &s.imem)
                } else {
                    (&mut s.dmem_port, &s.dmem)
                };
                let at = (*port & PORT_OFFSET) as usize;
                let value = u32::from_le_bytes(memory[at..at + 4].try_into().unwrap());
                if *port & PORT_AINCR != 0 {
                    *port = step(*port);
                }
                value
            }
            _ => return Err(format!("unmodeled read of +{offset:#x}")),
        })
    }

    fn write_register(&self, offset: u64, value: u32) -> Result<(), String> {
        let mut s = self.state.lock().unwrap();
        match offset {
            IRQSSET => s.irq_status |= value,
            IRQSCLR => s.irq_status &= !value,
            IRQMODE => s.irq_mode = value,
            IRQMSET => s.irq_mask |= value,
            IRQMCLR => s.irq_mask &= !value,
            IRQDEST => s.irq_dest = value,
            MAILBOX0 => s.mailbox[0] = value,
            MAILBOX1 => s.mailbox[1] = value,
            ITFEN => s.itfen = value,
            BOOTVEC => s.bootvec = value,
            DMACTL => s.dma_control = value,
            DMATRFBASE => s.dma_base = value,
            DMATRFMOFFS => s.dma_memory_offset = value & DMATRFMOFFS_OFFS,
            DMATRFFBOFFS => s.dma_external_offset = value,
            DMATRFCMD => self.transfer(&mut s, value)?,
            IMEMC | DMEMC => {
                if value & !PORT_MODELED != 0 {
                    return Err(format!(
                        "memory port control {value:#x}: secure uploads are not modeled"
                    ));
                }
                if offset == IMEMC {
                    s.imem_port = value;
                } else {
                    s.dmem_port = value;
                }
            }
            IMEMT => s.imem_tag = value & IMEMT_TAG,
            IMEMD | DMEMD => {
                s.running_out("memory port write")?;
                let s = &mut *s;
                let (port, memory) = if offset == IMEMD {
                    (&mut s.imem_port, &mut s.imem)
                } else {
                    (&mut s.dmem_port, &mut s.dmem)
                };
                let at = (*port & PORT_OFFSET) as usize;
                memory[at..at + 4].copy_from_slice(&value.to_le_bytes());
                if *port & PORT_AINCW != 0 {
                    *port = step(*port);
                }
                if offset == IMEMD {
                    s.imem_loaded = s.imem_loaded.max(at as u32 + 4);
                }
            }
            CPUCTL => {
                if value != CPUCTL_STARTCPU {
                    return Err(format!("CPUCTL {value:#x}: only STARTCPU alone is modeled"));
                }
                if s.cpu != Cpu::Stopped {
                    return Err("second start of the processor".into());
                }
                self.run_firmware(&mut s)?;
            }
            _ => return Err(format!("unmodeled write of {value:#x} to +{offset:#x}")),
        }
        Ok(())
    }

    /// A `DMATRFCMD` write: the transfer it describes, complete on return.
    fn transfer(&self, s: &mut State, command: u32) -> Result<(), String> {
        s.running_out("DMA")?;
        if command & (DMATRFCMD_SEC | DMATRFCMD_CTXDMA) != 0 {
            return Err(format!(
                "DMA command {command:#x}: secure and context DMA are not modeled"
            ));
        }
        if s.dma_control & DMACTL_REQUIRE_CTX != 0 {
            return Err("DMA with DMACTL REQUIRE_CTX set: no context is modeled".into());
        }
        let external = (u64::from(s.dma_base) << 8) + u64::from(s.dma_external_offset);
        let local = s.dma_memory_offset as usize;
        let write = command & DMATRFCMD_WRITE != 0;
        if command & DMATRFCMD_IMEM != 0 {
            if write || !local.is_multiple_of(IMEM_BLOCK) {
                return Err(format!(
                    "code DMA command {command:#x} at IMEM {local:#x}: only block reads \
                     are modeled"
                ));
            }
            self.dma_read(external, &mut s.imem[local..local + IMEM_BLOCK])?;
            s.imem_loaded = s.imem_loaded.max((local + IMEM_BLOCK) as u32);
        } else {
            let size = (command & DMATRFCMD_SIZE) >> DMATRFCMD_SIZE_SHIFT;
            if size > 6 {
                return Err(format!("data DMA command {command:#x}: size field {size}"));
            }
            let len = 4usize << size;
            if !local.is_multiple_of(len) || local + len > MEMORY_SIZE {
                return Err(format!("data DMA of {len:#x} bytes at DMEM {local:#x}"));
            }
            let span = local..local + len;
            if write {
                self.dma_write(external, &s.dmem[span])?;
            } else {
                self.dma_read(external, &mut s.dmem[span])?;
            }
        }
        s.dma_command = command & !(DMATRFCMD_FULL | DMATRFCMD_IDLE);
        Ok(())
    }

    /// `STARTCPU`: the firmware's whole run, as the module describes it.
    fn run_firmware(&self, s: &mut State) -> Result<(), String> {
        if s.bootvec != 0 || s.imem_loaded != FIRMWARE_SIZE {
            return Err(format!(
                "start at boot vector {:#x} with IMEM loaded up to {:#x}: only the 23.5.0 \
                 firmware (vector 0, {FIRMWARE_SIZE:#x} bytes) is modeled",
                s.bootvec, s.imem_loaded
            ));
        }
        let digest = hex(&Sha256::digest(&s.imem[..FIRMWARE_SIZE as usize]));
        if digest != FIRMWARE_SHA256 {
            return Err(format!(
                "unknown TSEC firmware (IMEM SHA-256 {digest}); only 23.5.0's is modeled"
            ));
        }
        let wiring = self.soc("firmware start")?;
        let keys = wiring
            .keys
            .as_ref()
            .map_err(|missing| format!("firmware start: the key set lacks {missing}"))?;
        let soc = &*wiring.soc;
        let word = |address: u32| -> Result<u32, String> {
            let mut bytes = [0; 4];
            soc.ahb_read(address, &mut bytes)
                .ok_or_else(|| format!("firmware read of {address:#x} failed"))?;
            Ok(u32::from_le_bytes(bytes))
        };
        let aperture = (
            self.mc_register(MC_IRAM_BOM)?,
            self.mc_register(MC_IRAM_TOM)?,
        );
        for (what, value, expected) in [
            ("FUSE_SKU_INFO", word(FUSE_SKU_INFO)?, SKU_INFO),
            ("CLK_SOURCE_TSEC", word(CLK_SOURCE_TSEC)?, TSEC_CLOCK_SOURCE),
            ("HOST1X+0x3300 magic", word(HOST1X_MAGIC_AT)?, HOST1X_MAGIC),
            ("MC_IRAM_BOM", aperture.0, IRAM_APERTURE.0),
            ("MC_IRAM_TOM", aperture.1, IRAM_APERTURE.1),
        ] {
            if value != expected {
                return Err(format!(
                    "firmware check {what} = {value:#x}, {expected:#x} expected; its failure \
                     path is not modeled"
                ));
            }
        }
        let size = word(PK11_BLOB)? as usize;
        if size > PK11_MAX_SIZE || !size.is_multiple_of(16) {
            return Err(format!("PK11 size {size:#x}"));
        }
        let mut blob = vec![0; PK11_DATA + size + 16];
        soc.ahb_read(PK11_BLOB, &mut blob)
            .ok_or_else(|| format!("firmware read of the PK11 blob at {PK11_BLOB:#x} failed"))?;
        let (signed, mac) = blob.split_at(PK11_DATA + size);
        if cmac(&keys.package1_mac_key, signed)[..] != *mac {
            return Err("PK11 CMAC does not match under package1_mac_key_08".into());
        }
        let iv: Key = blob[PK11_IV..PK11_DATA].try_into().unwrap();
        let mut plain = blob[PK11_DATA..PK11_DATA + size].to_vec();
        cbc_decrypt(&keys.package1_key, &iv, &mut plain);
        let header = PK11_BLOB + PK11_DATA as u32;
        let entry = pk11_entry(&plain, u64::from(header))
            .ok_or("decrypted PK11 fails package1ldr's header check")?;
        let effect = |done: Option<()>, what: &str| done.ok_or_else(|| format!("{what} failed"));
        effect(soc.ahb_write(header, &plain), "writing the PK11 plaintext")?;
        effect(
            soc.load_se_keyslot(TSEC_ROOT_KEY_SLOT, &keys.tsec_root_key),
            "loading SE keyslot 0x0D",
        )?;
        effect(
            soc.load_se_keyslot(TSEC_KEY_SLOT, &keys.tsec_key),
            "loading SE keyslot 0x0C",
        )?;
        effect(
            soc.ahb_write(EVP_COP_RESET_VECTOR, &entry.to_le_bytes()),
            "writing EVP_COP_RESET_VECTOR",
        )?;
        s.mailbox[1] = BOOT_SUCCESS;
        s.dmem.fill(0);
        s.cpu = Cpu::Halted;
        s.irq_status |= IRQ_HALT;
        effect(soc.restart_bpmp(entry), "restarting the BPMP")
    }
}

impl State {
    fn running_out(&self, what: &str) -> Result<(), String> {
        match self.cpu {
            Cpu::Stopped => Ok(()),
            Cpu::Halted => Err(format!(
                "{what} after the firmware ran: what it left in its memories is not modeled"
            )),
        }
    }
}

/// A memory port's control after one auto-increment.
fn step(port: u32) -> u32 {
    port & !PORT_OFFSET | (port & PORT_OFFSET).wrapping_add(4) & PORT_OFFSET
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn cbc_decrypt(key: &Key, iv: &Key, data: &mut [u8]) {
    let cipher = Aes128::new(key.into());
    let mut chain = *iv;
    for block in data.as_chunks_mut::<16>().0 {
        let ciphertext = *block;
        cipher.decrypt_block(block.into());
        for (byte, previous) in block.iter_mut().zip(chain) {
            *byte ^= previous;
        }
        chain = ciphertext;
    }
}

/// The NX bootloader's entry in the PK11 plaintext `plain` placed at
/// `header`, checked the way package1ldr's parser checks it (its
/// `check_pk11`, text 0xA80): magic, section sizes within the blob, entry
/// offsets within their sections, sizes adding up to the blob.
fn pk11_entry(plain: &[u8], header: u64) -> Option<u32> {
    let len = plain.len() as u64;
    let field = |index: usize| {
        let at = index * 4;
        Some(u64::from(u32::from_le_bytes(
            plain.get(at..at + 4)?.try_into().ok()?,
        )))
    };
    let [warmboot, bootloader, entry, secmon, secmon_entry] =
        [field(1)?, field(4)?, field(5)?, field(6)?, field(7)?];
    let fits = plain.get(..4)? == b"PK11"
        && warmboot < len
        && bootloader < len
        && entry < bootloader
        && secmon < len
        && secmon_entry < secmon
        && (warmboot + bootloader + secmon + PK11_HEADER).next_multiple_of(16) == len;
    fits.then(|| u32::try_from(header + PK11_HEADER + entry).ok())
        .flatten()
}

impl MmioDevice for Tsec {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        let result = if size == 4 && offset.is_multiple_of(4) {
            self.read_register(offset)
        } else {
            Err(format!("{size}-byte read at +{offset:#x}"))
        };
        self.served(result).map(u64::from)
    }

    fn write(&self, _core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        let result = if size == 4 && offset.is_multiple_of(4) {
            self.write_register(offset, value as u32)
        } else {
            Err(format!("{size}-byte write of {value:#x} at +{offset:#x}"))
        };
        self.served(result)
    }

    fn snapshot(&self) -> Option<&dyn Snapshot> {
        Some(self)
    }
}

/// Saves the registers, both memories and whether the firmware has run.
/// The wiring and the key set's values are the restoring machine's.
impl Snapshot for Tsec {
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        encode(&*self.state.lock().unwrap())
    }

    fn restore(&self, state: &[u8]) -> Result<(), SnapshotError> {
        let state: State = decode("TSEC", state)?;
        if state.imem.len() != MEMORY_SIZE || state.dmem.len() != MEMORY_SIZE {
            return Err(SnapshotError::Corrupt(format!(
                "TSEC: memories of {:#x}/{:#x} bytes saved, {MEMORY_SIZE:#x} expected",
                state.imem.len(),
                state.dmem.len()
            )));
        }
        *self.state.lock().unwrap() = state;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::{boot::DramSize, memory::GuestRam, soc::mc::Mc};

    const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../");
    const PACKAGE1: &str =
        "fw_dump/0100000000000819/da6ae627428f5fe36e26aafa833515a8/section0_romfs/nx/package1";
    const IRAM: u32 = 0x4000_0000;
    const IRAM_SIZE: usize = 0x4_0000;
    const DRAM: u64 = 0x8000_0000;

    /// IRAM, a few MMIO registers, the security engine's keyslots and the
    /// BPMP's restart, as a machine provides them.
    #[derive(Default)]
    struct Soc {
        iram: Mutex<Vec<u8>>,
        mmio: Mutex<HashMap<u32, u32>>,
        keyslots: Mutex<HashMap<usize, Key>>,
        restarted: Mutex<Option<u32>>,
    }

    impl Soc {
        fn new() -> Arc<Self> {
            let soc = Self::default();
            *soc.iram.lock().unwrap() = vec![0; IRAM_SIZE];
            Arc::new(soc)
        }

        fn iram(&self, address: u32, len: usize) -> Vec<u8> {
            let at = (address - IRAM) as usize;
            self.iram.lock().unwrap()[at..at + len].to_vec()
        }
    }

    impl TsecSoc for Soc {
        fn ahb_read(&self, address: u32, buffer: &mut [u8]) -> Option<()> {
            if let Some(at) = address.checked_sub(IRAM).map(|at| at as usize)
                && at + buffer.len() <= IRAM_SIZE
            {
                buffer.copy_from_slice(&self.iram.lock().unwrap()[at..at + buffer.len()]);
                return Some(());
            }
            let value = *self.mmio.lock().unwrap().get(&address)?;
            (buffer.len() == 4).then(|| buffer.copy_from_slice(&value.to_le_bytes()))
        }

        fn ahb_write(&self, address: u32, data: &[u8]) -> Option<()> {
            if let Some(at) = address.checked_sub(IRAM).map(|at| at as usize)
                && at + data.len() <= IRAM_SIZE
            {
                self.iram.lock().unwrap()[at..at + data.len()].copy_from_slice(data);
                return Some(());
            }
            let value = u32::from_le_bytes(data.try_into().ok()?);
            self.mmio.lock().unwrap().insert(address, value);
            Some(())
        }

        fn load_se_keyslot(&self, slot: usize, key: &Key) -> Option<()> {
            self.keyslots.lock().unwrap().insert(slot, *key);
            Some(())
        }

        fn restart_bpmp(&self, entry: u32) -> Option<()> {
            *self.restarted.lock().unwrap() = Some(entry);
            Some(())
        }
    }

    struct Bench {
        tsec: Tsec,
        mc: Arc<Mc>,
        ram: Arc<GuestRam>,
        soc: Arc<Soc>,
        notes: Arc<Notes>,
    }

    impl Bench {
        /// The memory controller with package1ldr's IRAM aperture open.
        fn new(keys: &KeySet) -> Self {
            let mc = Arc::new(Mc::new(DramSize::Gb4));
            mc.write(0, MC_IRAM_BOM, 4, 0x4000_0000).unwrap();
            mc.write(0, MC_IRAM_TOM, 4, 0x8000_0000).unwrap();
            let ram = Arc::new(GuestRam::new(DRAM, 0x4000).unwrap());
            let notes: Arc<Notes> = Arc::default();
            let tsec = Tsec::new(
                DeviceMemory::new(&mc, &ram, SMMU_TSEC_ASID),
                Arc::clone(&notes),
            );
            let soc = Soc::new();
            tsec.connect(Arc::clone(&soc) as Arc<dyn TsecSoc>, keys);
            Self {
                tsec,
                mc,
                ram,
                soc,
                notes,
            }
        }

        fn read(&self, offset: u64) -> Option<u32> {
            self.tsec.read(0, offset, 4).map(|value| value as u32)
        }

        fn write(&self, offset: u64, value: u32) -> Option<()> {
            self.tsec.write(0, offset, 4, value.into())
        }

        /// What package1ldr does from its DMA setup to the start
        /// (`tsec_boot`, text 0xBB4), but the host1x magic, which is
        /// host1x's register; returns the `CPUCTL` write's outcome.
        fn boot_like_package1ldr(&self) -> Option<()> {
            self.write(DMACTL, 0).unwrap();
            self.write(IRQMSET, 0xFFF2).unwrap();
            self.write(IRQDEST, 0xFFF0).unwrap();
            self.write(ITFEN, 3).unwrap();
            assert_ne!(self.read(DMATRFCMD).unwrap() & DMATRFCMD_IDLE, 0);
            self.write(DMATRFBASE, 0x4001_0E00 >> 8).unwrap();
            for offset in (0..FIRMWARE_SIZE).step_by(0x100) {
                self.write(DMATRFMOFFS, offset).unwrap();
                self.write(DMATRFFBOFFS, offset).unwrap();
                self.write(DMATRFCMD, 0x10).unwrap();
                assert_ne!(self.read(DMATRFCMD).unwrap() & DMATRFCMD_IDLE, 0);
            }
            self.write(MAILBOX1, 0).unwrap();
            self.write(MAILBOX0, 1).unwrap();
            self.write(BOOTVEC, 0).unwrap();
            self.write(CPUCTL, CPUCTL_STARTCPU)
        }
    }

    fn sha256(bytes: &[u8]) -> String {
        hex(&Sha256::digest(bytes))
    }

    #[test]
    fn host_interface_registers_behave_as_documented() {
        let bench = Bench::new(&KeySet::default());
        // Interrupts: set/clear pairs over status and mask.
        bench.write(IRQSSET, 0x11).unwrap();
        bench.write(IRQSCLR, 0x01).unwrap();
        assert_eq!(bench.read(IRQSTAT), Some(0x10));
        bench.write(IRQMSET, 0xFFF2).unwrap();
        bench.write(IRQMCLR, 0x2).unwrap();
        assert_eq!(bench.read(IRQMASK), Some(0xFFF0));
        bench.write(IRQDEST, 0xFFF0).unwrap();
        assert_eq!(bench.read(IRQDEST), Some(0xFFF0));
        // Mailboxes and plain registers hold what is written.
        bench.write(MAILBOX0, 1).unwrap();
        bench.write(MAILBOX1, 0xdead_beef).unwrap();
        assert_eq!(bench.read(MAILBOX0), Some(1));
        assert_eq!(bench.read(MAILBOX1), Some(0xdead_beef));
        bench.write(ITFEN, 3).unwrap();
        assert_eq!(bench.read(ITFEN), Some(3));
        // Scrubbing is over, the processor idle and stopped, the DMA idle.
        bench.write(DMACTL, 0x6).unwrap();
        assert_eq!(bench.read(DMACTL), Some(0));
        assert_eq!(bench.read(IDLESTATE), Some(0));
        assert_eq!(bench.read(CPUCTL), Some(0));
        assert_eq!(bench.read(DMATRFCMD), Some(DMATRFCMD_IDLE));
        // Programmed I/O with auto-increment, both memories.
        bench.write(IMEMC, PORT_AINCW | 0x100).unwrap();
        bench.write(IMEMT, 1).unwrap();
        bench.write(IMEMD, 0x1122_3344).unwrap();
        bench.write(IMEMD, 0x5566_7788).unwrap();
        assert_eq!(bench.read(IMEMC), Some(PORT_AINCW | 0x108));
        bench.write(IMEMC, PORT_AINCR | 0x104).unwrap();
        assert_eq!(bench.read(IMEMD), Some(0x5566_7788));
        assert_eq!(bench.read(IMEMC), Some(PORT_AINCR | 0x108));
        assert_eq!(bench.read(IMEMT), Some(1));
        bench.write(DMEMC, PORT_AINCW).unwrap();
        bench.write(DMEMD, 7).unwrap();
        bench.write(DMEMC, 0).unwrap();
        assert_eq!(bench.read(DMEMD), Some(7));
        // Secure uploads, unknown registers and widths are refused.
        assert_eq!(bench.write(IMEMC, 1 << 28), None);
        assert_eq!(bench.read(0x1108), None); // HWCFG
        assert_eq!(bench.read(0x0000), None); // THI
        assert_eq!(bench.tsec.read(0, MAILBOX0, 1), None);
        assert!(
            bench
                .notes
                .take()
                .iter()
                .all(|note| note.starts_with("TSEC: "))
        );
        // A start with anything but the known firmware is refused, and so
        // is any CPUCTL value but STARTCPU.
        assert_eq!(bench.write(CPUCTL, 3), None);
        assert_eq!(bench.write(CPUCTL, CPUCTL_STARTCPU), None);
        assert_eq!(bench.read(CPUCTL), Some(0));
    }

    #[test]
    fn dma_moves_blocks_through_the_memory_controller() {
        let bench = Bench::new(&KeySet::default());
        // Code from IRAM through the aperture: one 256-byte block, whatever
        // the size field.
        let pattern: Vec<u8> = (0..=255).collect();
        bench.soc.ahb_write(0x4001_0E00, &pattern).unwrap();
        bench.write(DMATRFBASE, 0x4001_0E00 >> 8).unwrap();
        bench.write(DMATRFMOFFS, 0x200).unwrap();
        bench.write(DMATRFFBOFFS, 0).unwrap();
        bench.write(DMATRFCMD, 0x10).unwrap();
        assert_eq!(bench.read(DMATRFCMD), Some(0x10 | DMATRFCMD_IDLE));
        bench.write(IMEMC, PORT_AINCR | 0x2FC).unwrap();
        assert_eq!(bench.read(IMEMD), Some(0xFFFE_FDFC));
        assert_eq!(bench.read(IMEMD), Some(0)); // past the block
        // Data from DRAM, 16 bytes (size field 2), and back out.
        bench.ram.write(DRAM + 0x40, &[0xA5; 16]).unwrap();
        bench.write(DMATRFBASE, (DRAM >> 8) as u32).unwrap();
        bench.write(DMATRFMOFFS, 0x10).unwrap();
        bench.write(DMATRFFBOFFS, 0x40).unwrap();
        bench.write(DMATRFCMD, 2 << 8).unwrap();
        bench.write(DMEMC, 0x1C).unwrap();
        assert_eq!(bench.read(DMEMD), Some(0xA5A5_A5A5));
        bench.write(DMATRFFBOFFS, 0x80).unwrap();
        bench.write(DMATRFCMD, DMATRFCMD_WRITE | 2 << 8).unwrap();
        let mut out = [0; 16];
        bench.ram.read(DRAM + 0x80, &mut out).unwrap();
        assert_eq!(out, [0xA5; 16]);
        // Outside DRAM and the aperture, through the SMMU, unaligned,
        // secure or context DMA: refused.
        bench.write(DMATRFBASE, 0x10).unwrap();
        assert_eq!(bench.write(DMATRFCMD, 0x10), None);
        bench.write(DMATRFBASE, 0x4001_0E00 >> 8).unwrap();
        bench.write(DMATRFMOFFS, 0x80).unwrap();
        assert_eq!(bench.write(DMATRFCMD, 0x10), None);
        bench.write(DMATRFMOFFS, 0).unwrap();
        assert_eq!(bench.write(DMATRFCMD, 0x10 | 1 << 2), None);
        assert_eq!(bench.write(DMATRFCMD, 0x10 | 1 << 12), None);
        bench.mc.write(0, SMMU_TSEC_ASID, 4, 0x8000_0001).unwrap();
        assert_eq!(bench.write(DMATRFCMD, 0x10), None);
        bench.mc.write(0, SMMU_TSEC_ASID, 4, 0).unwrap();
        // With the aperture closed IRAM is out of reach.
        bench.mc.write(0, MC_IRAM_TOM, 4, 0).unwrap();
        assert_eq!(bench.write(DMATRFCMD, 0x10), None);
        assert!(!bench.notes.take().is_empty());
    }

    /// package1 of 23.5.0 in IRAM where the boot ROM puts it, with the
    /// registers the firmware checks as package1ldr leaves them.
    fn bench_with_package1() -> (Bench, Vec<u8>, KeySet) {
        let package1 = std::fs::read(format!("{ROOT}{PACKAGE1}"))
            .expect("required package1 extraction in fw_dump");
        let keys = KeySet::load(std::path::Path::new(&format!("{ROOT}prod_newer.keys")))
            .expect("required key file");
        let bench = Bench::new(&keys);
        bench.soc.ahb_write(0x4001_0000, &package1).unwrap();
        for (address, value) in [
            (FUSE_SKU_INFO, SKU_INFO),
            (CLK_SOURCE_TSEC, TSEC_CLOCK_SOURCE),
            (HOST1X_MAGIC_AT, HOST1X_MAGIC),
        ] {
            bench.soc.mmio.lock().unwrap().insert(address, value);
        }
        (bench, package1, keys)
    }

    #[test]
    fn the_23_5_0_firmware_leaves_what_secure_boot_leaves() {
        let (bench, package1, keys) = bench_with_package1();
        let ciphertext = bench.soc.iram(0x4001_7000, 0x20);
        bench.boot_like_package1ldr().unwrap();
        assert!(bench.notes.take().is_empty());
        // The firmware halted with its success value.
        assert_eq!(bench.read(CPUCTL), Some(CPUCTL_HALTED));
        assert_eq!(bench.read(MAILBOX1), Some(BOOT_SUCCESS));
        assert_ne!(bench.read(IRQSTAT).unwrap() & IRQ_HALT, 0);
        // PK11 decrypted in place: its sections hash to the prefixes
        // package1's plaintext header carries (+4 secure monitor, +8 NX
        // bootloader).
        let header = bench.soc.iram(0x4001_7000, 0x20);
        assert_ne!(header, ciphertext);
        assert_eq!(&header[..4], b"PK11");
        let bootloader = sha256(&bench.soc.iram(0x4001_7020, 0x18FE0));
        let secmon = sha256(&bench.soc.iram(0x4003_0000, 0xE000));
        assert_eq!(bootloader[..8], hex(&package1[8..12]));
        assert_eq!(secmon[..8], hex(&package1[4..8]));
        // The BPMP restarts at the NX bootloader's entry.
        let entry = 0x4002_E104;
        assert_eq!(*bench.soc.restarted.lock().unwrap(), Some(entry));
        assert_eq!(
            bench.soc.mmio.lock().unwrap().get(&EVP_COP_RESET_VECTOR),
            Some(&entry)
        );
        // Keyslots 0x0C and 0x0D hold the key set's values (digests only).
        let keyslots = bench.soc.keyslots.lock().unwrap();
        assert_eq!(keyslots.len(), 2);
        assert_eq!(
            sha256(&keyslots[&TSEC_KEY_SLOT]),
            sha256(keys.tsec_key().unwrap())
        );
        assert_eq!(
            sha256(&keyslots[&TSEC_ROOT_KEY_SLOT]),
            sha256(keys.tsec_root_key(2).unwrap())
        );
        drop(keyslots);
        // Nothing more is modeled: no second start, no memory access.
        assert_eq!(bench.write(CPUCTL, CPUCTL_STARTCPU), None);
        assert_eq!(bench.write(DMATRFCMD, 0x10), None);
        assert_eq!(bench.read(IMEMD), None);
        // The state survives a snapshot.
        let saved = bench.tsec.save().unwrap();
        let other = Bench::new(&keys);
        other.tsec.restore(&saved).unwrap();
        assert_eq!(other.read(CPUCTL), Some(CPUCTL_HALTED));
        assert_eq!(other.read(MAILBOX1), Some(BOOT_SUCCESS));
        assert_eq!(other.read(IRQMASK), Some(0xFFF2));
    }

    #[test]
    fn an_unknown_firmware_is_refused() {
        let (bench, _, _) = bench_with_package1();
        let mut byte = [0];
        bench.soc.ahb_read(0x4001_1000, &mut byte).unwrap();
        bench.soc.ahb_write(0x4001_1000, &[byte[0] ^ 1]).unwrap();
        let ciphertext = bench.soc.iram(0x4001_7000, 0x20);
        assert_eq!(bench.boot_like_package1ldr(), None);
        let notes = bench.notes.take();
        assert!(
            notes
                .iter()
                .any(|note| note.contains("unknown TSEC firmware")),
            "{notes:?}"
        );
        assert_eq!(bench.read(CPUCTL), Some(0));
        assert_eq!(bench.read(MAILBOX1), Some(0));
        assert_eq!(bench.soc.iram(0x4001_7000, 0x20), ciphertext);
        assert!(bench.soc.keyslots.lock().unwrap().is_empty());
        assert_eq!(*bench.soc.restarted.lock().unwrap(), None);
    }

    #[test]
    fn a_failed_firmware_check_or_missing_keys_refuse_the_start() {
        let (bench, _, _) = bench_with_package1();
        bench.soc.mmio.lock().unwrap().insert(HOST1X_MAGIC_AT, 0);
        assert_eq!(bench.boot_like_package1ldr(), None);
        assert!(bench.notes.take().iter().any(|note| note.contains("magic")));
        assert!(bench.soc.keyslots.lock().unwrap().is_empty());

        let package1 = std::fs::read(format!("{ROOT}{PACKAGE1}"))
            .expect("required package1 extraction in fw_dump");
        let bench = Bench::new(&KeySet::default());
        bench.soc.ahb_write(0x4001_0000, &package1).unwrap();
        assert_eq!(bench.boot_like_package1ldr(), None);
        let notes = bench.notes.take();
        assert!(
            notes.iter().any(|note| note
                .contains("tsec_key, tsec_root_key_02, package1_key_08, package1_mac_key_08")),
            "{notes:?}"
        );
    }
}
