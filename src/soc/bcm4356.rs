//! The WiFi function of the Broadcom BCM4356 radio combo chip, behind the
//! Tegra210 PCIe root port.
//!
//! `wlan` (program 0100000000000016) drives it with an embedded copy of
//! Broadcom's dongle host driver (DHD) in its "full dongle" PCIe design: the
//! chip runs a firmware of its own on an ARM Cortex-R4, which the host
//! downloads into the chip's RAM and which then talks to the host through
//! message rings in host memory. The emulator does not execute that
//! firmware. It models what the host observes of the chip:
//!
//! * the PCI function (configuration space, a 32 KiB register BAR and a
//!   4 MiB BAR onto the chip's tightly coupled memory);
//! * the silicon backplane reached through the register BAR: the chip ID,
//!   the enumeration ROM listing the cores, the ARM core's memory banks,
//!   and the core wrappers through which the host holds and releases the
//!   ARM;
//! * once the host has written the firmware image and NVRAM and released
//!   the ARM, what the started firmware does that the host can see: it
//!   publishes its shared structure (`pciedev_shared_t`) and answers the
//!   message rings — buffer posts, flow rings and control requests
//!   (ioctls) — as a radio with no network in range.
//!
//! Firmware evidence, as `wlan` text offsets: the probe accepts vendor
//! 0x14E4 with device 0x43EC or 0x4354 (`0xfafc0`); `0xfd670` attaches
//! (BAR0 window at configuration 0x80, `si_attach` at `0x106930`, the
//! enumeration ROM parsed at `0x108040`, RAM size from the ARM CR4 banks at
//! `0x107a70`, RAM base 0x180000 for chip 0x4356); `0xfe680` downloads the
//! 583641-byte image to the RAM base and `0x100970` the NVRAM below the
//! last word, which holds the NVRAM length token; `0x1003a0` halts the ARM
//! before and releases it after; `0xffcc0` polls the last RAM word for the
//! shared structure (500 x 10 ms) and reads it; `0x102870` sets up the
//! rings (`dhd_bus_cmn_writeshared`, `0xff030`); `0x105ed0` sends control
//! requests; the D2H handlers are the table at data `0x16e698`
//! (`0x105620` ack, `0x105670` completion, `0x1057e0` event); the interrupt
//! status is the PCIe core's mailbox (`0xfd960`), the doorbell its H2D
//! mailbox 0 (`0xffb90`). Structure layouts follow the public `brcmfmac`
//! driver (Linux, `drivers/net/wireless/broadcom/brcm80211/brcmfmac/pcie.c`,
//! `msgbuf.c`), used for facts only, and agree with the offsets `wlan` uses.

use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex},
};

use super::{
    Notes,
    pcie_endpoint::{PcieFunction, Upstream},
};

// ---------------------------------------------------------------------------
// Identity of the function. These are values only real hardware supplies.
// ---------------------------------------------------------------------------

/// Broadcom. `wlan` text `0xfafc0` matches it before choosing its driver.
const VENDOR_ID: u16 = 0x14e4;
/// BCM4356 802.11ac 2x2, PCIe function. `wlan` text `0xfafc0` accepts 0x43EC
/// (and 0x4354); the embedded NVRAM says `devid=0x43ec`.
const DEVICE_ID: u16 = 0x43ec;
/// Revision 2, as public `lspci` listings of BCM4356 adapters show
/// (assumed: nothing in `wlan` tests it).
const REVISION_ID: u8 = 2;
/// Network controller, other (0x0280), as public listings show.
const CLASS_CODE: u32 = 0x02_8000;
/// Sizes of BAR0 (registers) and BAR2 (RAM window); public listings of
/// BCM4356 adapters show 32K and 4M, which is also what DHD's BAR0 layout
/// (window, wrapper window, PCIe core, ChipCommon at 0x0000/0x1000/0x2000/
/// 0x3000; `wlan` `0x107480`) and the RAM at 0x180000-0x23ffff need.
const BAR0_SIZE: u64 = 0x8000;
const BAR2_SIZE: u64 = 0x40_0000;

/// Chip number, revision and package in the ChipCommon chip ID register.
/// `wlan` `0x106930` reads them; the embedded firmware is named
/// `4356a3-roml`, so revision 3 (A3) is assumed.
const CHIP_ID: u32 = 0x4356;
const CHIP_REV: u32 = 3;
const CHIP_PACKAGE: u32 = 0;
/// AXI interconnect (`SOCI_AI`), which makes `wlan` parse the enumeration
/// ROM (`0x106930` → `0x108040`).
const SOCI_AI: u32 = 1;

/// Core revisions. Assumed values: `wlan` only distinguishes PCIe core
/// revisions 2, 4 and 6 (old interrupt registers) from the others, and needs
/// ChipCommon above 10 to read the chip status; nothing else it does
/// depends on them.
const CC_REV: u8 = 50;
const D11_REV: u8 = 42;
const ARMCR4_REV: u8 = 7;
const PCIE2_REV: u8 = 11;

/// Backplane addresses of the cores and their wrappers: the usual AI layout
/// (cores at 0x18000000 + n * 0x1000, wrappers 1 MiB above). Assumed; the
/// host learns them from the enumeration ROM.
const CC_BASE: u32 = 0x1800_0000;
const D11_BASE: u32 = 0x1800_1000;
const ARMCR4_BASE: u32 = 0x1800_2000;
const PCIE2_BASE: u32 = 0x1800_3000;
const WRAPPER_OFFSET: u32 = 0x10_0000;
/// Where ChipCommon's `eromptr` points (assumed placement, within the
/// enumeration space behind the wrappers).
const EROM_BASE: u32 = 0x1810_f000;

const CORE_CC: u16 = 0x800;
const CORE_D11: u16 = 0x812;
const CORE_PCIE2: u16 = 0x83c;
const CORE_ARMCR4: u16 = 0x83e;
const MFG_BRCM: u32 = 0x4bf;

/// ChipCommon capabilities: a PMU is present (`CC_CAP_PMU`, bit 28), which
/// `wlan` `0x106930` tests before reading the PMU capabilities. The other
/// bits (UARTs, flash, OTP) are assumed clear; nothing on the PCIe path
/// reads them.
const CC_CAPABILITIES: u32 = 0x1000_0000;
/// Extended capabilities: no always-on bus (`CC_CAP_EXT_AOB_PRESENT` clear),
/// so the PMU registers are ChipCommon's own (assumed).
const CC_CAPABILITIES_EXT: u32 = 0;
/// Chip status straps. Assumed zero: `wlan` stores the value but tests it
/// only for other chips (0x4335 family, `0x106930`).
const CC_CHIPSTATUS: u32 = 0;
/// PMU capabilities: PMU revision 24 in the low byte (assumed; `wlan` only
/// stores it).
const CC_PMUCAPABILITIES: u32 = 24;

/// RAM: base 0x180000 for chip 0x4356 (`wlan` `0xfd670`), 768 KiB in six
/// 128 KiB banks. The split into four A and two B banks is assumed; the
/// host only adds the bank sizes (`0x107a70`) and needs room for the
/// 583641-byte image (`0xfe680` checks 0x8e7d9).
const RAM_BASE: u32 = 0x18_0000;
const RAM_BANKS_A: u32 = 4;
const RAM_BANKS_B: u32 = 2;
/// Bank size field of `bankinfo`: (n + 1) * 8 KiB.
const RAM_BANK_SIZE_FIELD: u32 = 15;
const RAM_SIZE: u32 = (RAM_BANKS_A + RAM_BANKS_B) * (RAM_BANK_SIZE_FIELD + 1) * 0x2000;

// ---------------------------------------------------------------------------
// Configuration space layout.
// ---------------------------------------------------------------------------

/// Capabilities, as public `lspci` listings of BCM4356 adapters show them:
/// power management at 0x48, MSI at 0x58, the vendor-specific block (which
/// holds Broadcom's backplane window registers) at 0x68, PCI Express at 0xAC.
const CAP_PM: u16 = 0x48;
const CAP_MSI: u16 = 0x58;
const CAP_VENDOR: u16 = 0x68;
const CAP_PCIE: u16 = 0xac;
/// Power management capabilities: version 3, PME from D0, D3hot and D3cold
/// (public listing).
const PM_CAPABILITIES: u32 = 0xc803;
/// MSI: 64-bit addresses, 32 vectors requested (public listing).
const MSI_CONTROL_RO: u32 = 0x0080 | (5 << 1);
/// PCI Express capability register: version 2, endpoint.
const PCIE_CAPABILITIES: u32 = 0x0002;
/// Device capabilities, as public `lspci -vv` listings of BCM4350/4356
/// adapters decode them: max payload 512 bytes, extended tags, L0s
/// acceptable latency < 1 us, L1 unlimited, role-based error reporting.
const PCIE_DEVICE_CAPABILITIES: u32 = 0x8f22;
/// Link capabilities (same listings): port 0, 5 GT/s, x1, ASPM L0s and L1,
/// exit latencies L0s < 2 us and L1 < 32 us, clock power management,
/// ASPM optionality compliance.
const PCIE_LINK_CAPABILITIES: u32 = 0x47_5c12;
/// Link status: trained at 5 GT/s, x1, common slot clock (assumed: the
/// root port's own link status says what the link actually trained at).
const PCIE_LINK_STATUS: u32 = 0x1012;
/// Device capabilities 2 (same listings): completion timeout ranges A-D
/// and its disabling, latency tolerance reporting.
const PCIE_DEVICE_CAPABILITIES_2: u32 = 0x81f;
/// Link status 2: current de-emphasis -6 dB (same listings).
const PCIE_LINK_STATUS_2: u32 = 0x1;
/// Length byte of the vendor-specific capability (public listing: 44).
const VENDOR_CAP_LENGTH: u32 = 0x2c;
/// Extended capability headers (ID, version 1, next): advanced error
/// reporting, virtual channel, device serial number, power budgeting,
/// latency tolerance reporting, resizable BAR, L1 PM substates, at the
/// offsets public `lspci -vv` listings of BCM435x adapters show (assumed
/// for this chip). Their bodies are refused until something reads them.
const EXTENDED_CAPABILITIES: [(u16, u32); 7] = [
    (0x100, 0x0001),
    (0x13c, 0x0002),
    (0x160, 0x0003),
    (0x16c, 0x0004),
    (0x1b0, 0x0018),
    (0x220, 0x0015),
    (0x240, 0x001e),
];

/// Broadcom configuration registers (inside the vendor-specific block, and
/// one in the extended space): BAR0 wrapper window (`wlan` `0x108810` for a
/// PCIe gen2 core), BAR0 window, BAR1 (RAM) window, SPROM control (`wlan`
/// `0x106930` only checks it is not all ones; assumed zero: no SPROM write in
/// progress), and the BAR2 configuration `wlan` `0xfd670` reads and writes
/// back through the PCIe core's indirect configuration access.
const CFG_BAR0_WRAPPER: u16 = 0x70;
const CFG_BAR0_WINDOW: u16 = 0x80;
const CFG_BAR1_WINDOW: u16 = 0x84;
const CFG_SPROM_CONTROL: u16 = 0x88;
const CFG_BAR2_CONFIG: u16 = 0x4e0;
/// Reset value of the BAR2 configuration register (assumed: the host only
/// writes back what it read).
const BAR2_CONFIG_RESET: u32 = 0;

// ---------------------------------------------------------------------------
// Backplane registers.
// ---------------------------------------------------------------------------

const AI_IOCTRL: u32 = 0x408;
const AI_IOSTATUS: u32 = 0x500;
const AI_RESETCTRL: u32 = 0x800;
const AI_RESETSTATUS: u32 = 0x804;
const SICF_CLOCK_EN: u32 = 0x1;
const SICF_FGC: u32 = 0x2;
const SICF_CPUHALT: u32 = 0x20;
const AIRC_RESET: u32 = 0x1;

const ARMCR4_CAP: u32 = 0x04;
const ARMCR4_BANKIDX: u32 = 0x40;
const ARMCR4_BANKINFO: u32 = 0x44;

const PCIE2_INTMASK: u32 = 0x24;
const PCIE2_MAILBOXINT: u32 = 0x48;
const PCIE2_MAILBOXMASK: u32 = 0x4c;
const PCIE2_CONFIGADDR: u32 = 0x120;
const PCIE2_CONFIGDATA: u32 = 0x124;
const PCIE2_H2D_MAILBOX_0: u32 = 0x140;
/// Mailbox interrupt the dongle raises for "new D2H messages" (one of the
/// doorbell bits 16-23 `wlan` enables in its mask 0xff0300, `0xfd670`;
/// brcmfmac calls it `D2H0_DB0`).
const MAILBOX_D2H_DB: u32 = 0x1_0000;

// ---------------------------------------------------------------------------
// The shared structure and message rings.
// ---------------------------------------------------------------------------

/// `pciedev_shared_t` version. The firmware 7.35 generation reports 5
/// (assumed; `wlan` `0xffcc0` accepts up to 6). No DMA'd indices, 4-byte
/// index slots, no D2H sync marker: the host then reads indices from RAM
/// and takes messages as they are (`0x102870` picks the plain handler).
const SHARED_VERSION: u32 = 5;
const SHARED_SIZE: u32 = 80;
const RING_INFO_SIZE: u32 = 56;
const RING_MEM_SIZE: u32 = 16;
/// Index slot size without `PCIE_SHARED_2BYTE_INDICES` (`0xffcc0`).
const INDEX_SIZE: u32 = 4;
/// Common rings: H2D control submit and RX post; D2H control complete, TX
/// complete and RX complete.
const COMMON_RINGS: usize = 5;
const H2D_COMMON_RINGS: u16 = 2;
/// Submission queues the firmware offers (common plus flow rings); the
/// host allocates `max_sub_queues - 2` flow rings (`0x102870`). Assumed.
const MAX_SUB_QUEUES: u16 = 40;

const MSG_FLOW_RING_CREATE: u8 = 0x03;
const MSG_FLOW_RING_CREATE_CMPLT: u8 = 0x04;
const MSG_FLOW_RING_DELETE: u8 = 0x05;
const MSG_FLOW_RING_DELETE_CMPLT: u8 = 0x06;
const MSG_FLOW_RING_FLUSH: u8 = 0x07;
const MSG_FLOW_RING_FLUSH_CMPLT: u8 = 0x08;
const MSG_IOCTLPTR_REQ: u8 = 0x09;
const MSG_IOCTLPTR_REQ_ACK: u8 = 0x0a;
const MSG_IOCTLRESP_BUF_POST: u8 = 0x0b;
const MSG_IOCTL_CMPLT: u8 = 0x0c;
const MSG_EVENT_BUF_POST: u8 = 0x0d;
const MSG_RXBUF_POST: u8 = 0x11;

/// Ring indices of the D2H common rings.
const RING_D2H_CONTROL: usize = 2;

/// `WLC_GET_VAR` / `WLC_SET_VAR`: named variables ("iovars").
const WLC_GET_VAR: u32 = 262;
const WLC_SET_VAR: u32 = 263;
/// `WLC_GET_REVINFO` (`wlioctl_defs.h`, public; `wlan` `0x103490`).
const WLC_GET_REVINFO: u32 = 98;
/// Revision information only the silicon knows, assumed: radio revision
/// and analog core revision zero-based placeholders of an ACPHY chip; PHY
/// type AC (11, public `wlioctl_defs.h`), PHY revision 2 (the 4354/4356
/// generation's ACPHY, assumed); bus type PCI (1, public `bcmdevs.h`).
/// `wlan` reads none of them (`0x103490` only needs success).
const RADIO_REV: u32 = 0;
const ANA_REV: u32 = 0;
const PHY_TYPE_AC: u32 = 11;
const PHY_REV: u32 = 2;
const BUS_PCIE: u32 = 1;
/// Control request numbers (public `wlioctl_defs.h`).
const WLC_UP: u32 = 2;
const WLC_DOWN: u32 = 3;
const WLC_SET_INFRA: u32 = 20;
const WLC_SET_CHANNEL: u32 = 30;
const WLC_SET_PM: u32 = 86;
const WLC_SET_AP: u32 = 118;
const WLC_GET_BANDLIST: u32 = 140;
const WLC_SET_BAND: u32 = 142;
const WLC_GET_VALID_CHANNELS: u32 = 217;
/// Two bands: 5 GHz (1) and 2.4 GHz (2) (`WLC_BAND_5G`/`WLC_BAND_2G`); the
/// chip is dual-band (the NVRAM calibrates both).
const BANDLIST: [u32; 3] = [2, 1, 2];
/// The channels the firmware's regulatory data allows for the country
/// `wlan` sets ("XV", revision 984, Nintendo's). The firmware's CLM data
/// is not interpreted here: assumed to be the channels the console uses,
/// 2.4 GHz 1-13 and 5 GHz 36-48 (public console specifications: 5 GHz
/// limited to the W52 band).
const CHANNELS: [u8; 17] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 36, 40, 44, 48];
/// 802.11n enabled (`nmode` 1), 20 MHz in 2.4 GHz and 40 MHz in 5 GHz
/// (`mimo_bw_cap` 1): the defaults of a dual-band 11ac firmware (assumed;
/// `wlan` then sets `bw_cap` itself).
const NMODE: u32 = 1;
const MIMO_BW_CAP: u32 = 1;
/// Broadcom error codes (`bcmutils.h`, public).
const BCME_BADARG: i16 = -2;

/// The function.
pub struct Bcm4356Wifi {
    state: Mutex<State>,
    upstream: Box<dyn Upstream>,
    notes: Arc<Notes>,
}

#[derive(Clone, Copy, Default)]
struct Wrapper {
    ioctrl: u32,
    resetctrl: u32,
}

struct Config {
    command: u16,
    bars: [u32; 4],
    cache_line: u8,
    interrupt_line: u8,
    pmcsr: u32,
    msi_control: u16,
    msi_address: u64,
    msi_data: u16,
    device_control: u16,
    link_control: u16,
    device_control_2: u16,
    link_control_2: u16,
    bar0_window: u32,
    bar0_wrapper: u32,
    bar1_window: u32,
    bar2_config: u32,
}

struct Ring {
    base: u64,
    max_items: u16,
    item_len: u16,
}

#[derive(Clone, Copy)]
struct HostBuffer {
    request_id: u32,
    address: u64,
    len: u16,
}

/// The started firmware.
struct Dongle {
    /// Addresses of the index arrays in RAM.
    h2d_write: u32,
    h2d_read: u32,
    d2h_write: u32,
    d2h_read: u32,
    ring_mem: u32,
    /// The common rings, read from `ring_mem` at the first doorbell (the
    /// host fills them in after reading the shared structure).
    rings: Option<Vec<Ring>>,
    flow_rings: Vec<(u16, Ring)>,
    ioctl_buffers: VecDeque<HostBuffer>,
    event_buffers: VecDeque<HostBuffer>,
    rx_buffers: VecDeque<HostBuffer>,
    mac: [u8; 6],
    /// The NVRAM text the host downloaded.
    nvram: Vec<u8>,
    /// Version and microcode version from the image's trailer
    /// ("Version: 7.35.252.108 ... Ucode Ver: 1020.228").
    version: [u32; 4],
    ucode: [u32; 2],
    /// "7.35.252.108 (c216b7c)", the build date as "Aug 29 2021 19:24:39",
    /// and the firmware ID, from the trailer.
    version_text: String,
    build_date: String,
    fwid: String,
    up: bool,
    /// What the host set and the firmware keeps.
    settings: HashMap<String, Vec<u8>>,
}

struct State {
    config: Config,
    wrappers: [Wrapper; 4],
    armcr4_bank: u32,
    cc_gpio_pull: [u32; 2],
    pcie2_intmask: u32,
    mailbox_int: u32,
    mailbox_mask: u32,
    config_address: u32,
    /// What a host write to address 0 of the RAM window stores: the ARM's
    /// reset vector (`wlan` `0x1003a0` writes the image's first word there
    /// and reads it back).
    reset_vector: u32,
    ram: Vec<u8>,
    /// Highest RAM offset written while the ARM was halted, below the NVRAM:
    /// the end of the downloaded image.
    image_end: u32,
    halted: bool,
    dongle: Option<Dongle>,
    interrupt: bool,
}

/// Why an access was not served.
type Refusal = String;

impl Bcm4356Wifi {
    pub fn new(upstream: Box<dyn Upstream>, notes: Arc<Notes>) -> Self {
        // At power-on the ROM boot loader runs: ChipCommon, the PCIe core
        // and the ARM are out of reset and clocked, the 802.11 core is held
        // in reset (assumed; the host resets the ARM before downloading and
        // never looks at the 802.11 core).
        let up = Wrapper {
            ioctrl: SICF_CLOCK_EN,
            resetctrl: 0,
        };
        let held = Wrapper {
            ioctrl: 0,
            resetctrl: AIRC_RESET,
        };
        Self {
            state: Mutex::new(State {
                config: Config {
                    command: 0,
                    bars: [0x4, 0, 0x4, 0],
                    cache_line: 0,
                    interrupt_line: 0,
                    pmcsr: 0,
                    msi_control: 0,
                    msi_address: 0,
                    msi_data: 0,
                    device_control: 0,
                    link_control: 0,
                    device_control_2: 0,
                    link_control_2: 0,
                    bar0_window: CC_BASE,
                    bar0_wrapper: CC_BASE + WRAPPER_OFFSET,
                    bar1_window: 0,
                    bar2_config: BAR2_CONFIG_RESET,
                },
                wrappers: [up, held, up, up],
                armcr4_bank: 0,
                cc_gpio_pull: [0; 2],
                pcie2_intmask: 0,
                mailbox_int: 0,
                mailbox_mask: 0,
                config_address: 0,
                reset_vector: 0,
                ram: vec![0; RAM_SIZE as usize],
                image_end: 0,
                halted: false,
                dongle: None,
                interrupt: false,
            }),
            upstream,
            notes,
        }
    }

    fn served<T: Default>(&self, result: Result<T, Refusal>) -> Option<T> {
        match result {
            Ok(value) => Some(value),
            Err(note) => {
                self.notes.add(format!("BCM4356 WiFi: {note}"));
                self.notes.survey().then(T::default)
            }
        }
    }
}

/// The cores in enumeration order: (core ID, revision, base, master wrapper).
const CORES: [(u16, u8, u32, bool); 4] = [
    (CORE_CC, CC_REV, CC_BASE, false),
    (CORE_D11, D11_REV, D11_BASE, true),
    (CORE_ARMCR4, ARMCR4_REV, ARMCR4_BASE, true),
    (CORE_PCIE2, PCIE2_REV, PCIE2_BASE, true),
];
const WRAPPER_ARMCR4: usize = 2;

/// The enumeration ROM: per core a component descriptor (two words), a
/// master port descriptor for cores that master the bus, the core's 4 KiB
/// register space as slave port 0, and its wrapper (a master wrapper for
/// masters, else a slave wrapper); then the end marker. Encoding as
/// `wlan` `0x108040` parses it (and public `aiutils.c`).
fn erom() -> Vec<u32> {
    let mut words = Vec::new();
    for (id, rev, base, master) in CORES {
        let (nmp, nmw, nsw) = if master { (1, 1, 0) } else { (0, 0, 1) };
        let nsp = 1;
        words.push((MFG_BRCM << 20) | (u32::from(id) << 8) | 1);
        words
            .push((u32::from(rev) << 24) | (nsw << 19) | (nmw << 14) | (nsp << 9) | (nmp << 4) | 1);
        if master {
            words.push(0x2 | 1);
        }
        // Address descriptor: tag 4 (address), valid; 4 KiB size class 0.
        words.push(base | 0x4 | 1);
        let wrapper_type = if master { 0xc0 } else { 0x80 };
        words.push((base + WRAPPER_OFFSET) | wrapper_type | 0x4 | 1);
    }
    words.push(0xf);
    words
}

impl State {
    fn bar_address(&self, index: usize, size: u64) -> u64 {
        let low = u64::from(self.config.bars[index * 2] & !0xf);
        let high = u64::from(self.config.bars[index * 2 + 1]);
        ((high << 32) | low) & !(size - 1)
    }

    // ----- configuration space -----

    fn config_dword(&self, offset: u16) -> Result<u32, Refusal> {
        let c = &self.config;
        Ok(match offset {
            0x00 => u32::from(VENDOR_ID) | u32::from(DEVICE_ID) << 16,
            // Status: capabilities list present; INTx status mirrors the
            // line.
            0x04 => u32::from(c.command) | (0x10 | u32::from(self.interrupt) << 3) << 16,
            0x08 => u32::from(REVISION_ID) | CLASS_CODE << 8,
            0x0c => u32::from(c.cache_line),
            0x10..=0x1c => c.bars[usize::from((offset - 0x10) / 4)],
            // BAR4/5 not implemented, no CardBus CIS, no expansion ROM.
            0x20 | 0x24 | 0x28 | 0x30 | 0x38 => 0,
            0x34 => u32::from(CAP_PM),
            0x3c => u32::from(c.interrupt_line) | 1 << 8,
            o if o == CAP_PM => 0x01 | u32::from(CAP_MSI) << 8 | PM_CAPABILITIES << 16,
            o if o == CAP_PM + 4 => c.pmcsr,
            o if o == CAP_MSI => {
                0x05 | u32::from(CAP_VENDOR) << 8
                    | (MSI_CONTROL_RO | u32::from(c.msi_control)) << 16
            }
            o if o == CAP_MSI + 4 => c.msi_address as u32,
            o if o == CAP_MSI + 8 => (c.msi_address >> 32) as u32,
            o if o == CAP_MSI + 12 => u32::from(c.msi_data),
            o if o == CAP_VENDOR => 0x09 | u32::from(CAP_PCIE) << 8 | VENDOR_CAP_LENGTH << 16,
            CFG_BAR0_WRAPPER => c.bar0_wrapper,
            CFG_BAR0_WINDOW => c.bar0_window,
            CFG_BAR1_WINDOW => c.bar1_window,
            CFG_SPROM_CONTROL => 0,
            o if o == CAP_PCIE => 0x10 | PCIE_CAPABILITIES << 16,
            o if o == CAP_PCIE + 4 => PCIE_DEVICE_CAPABILITIES,
            o if o == CAP_PCIE + 8 => u32::from(c.device_control),
            o if o == CAP_PCIE + 0xc => PCIE_LINK_CAPABILITIES,
            o if o == CAP_PCIE + 0x10 => u32::from(c.link_control) | PCIE_LINK_STATUS << 16,
            o if o == CAP_PCIE + 0x20 => PCIE_DEVICE_CAPABILITIES_2,
            o if o == CAP_PCIE + 0x24 => u32::from(c.device_control_2),
            o if o == CAP_PCIE + 0x2c => u32::from(c.link_control_2) | PCIE_LINK_STATUS_2 << 16,
            CFG_BAR2_CONFIG => c.bar2_config,
            o if let Some(index) = EXTENDED_CAPABILITIES.iter().position(|cap| cap.0 == o) => {
                let next = EXTENDED_CAPABILITIES
                    .get(index + 1)
                    .map_or(0, |cap| u32::from(cap.0));
                EXTENDED_CAPABILITIES[index].1 | 1 << 16 | next << 20
            }
            _ => return Err(format!("configuration register {offset:#x} not modeled")),
        })
    }

    fn config_write_dword(&mut self, offset: u16, value: u32, mask: u32) -> Result<(), Refusal> {
        let old = self.config_dword(offset)?;
        let new = old & !mask | value & mask;
        let c = &mut self.config;
        match offset {
            // Memory space, bus master, INTx disable are what a host sets;
            // status bits are read-only or write-one-to-clear (none are set).
            0x04 => c.command = (new as u16) & 0x0546,
            0x0c => c.cache_line = new as u8,
            0x10 => c.bars[0] = new & !(BAR0_SIZE as u32 - 1) | 0x4,
            0x14 => c.bars[1] = new,
            0x18 => c.bars[2] = new & !(BAR2_SIZE as u32 - 1) | 0x4,
            0x1c => c.bars[3] = new,
            0x20 | 0x24 | 0x28 | 0x30 | 0x38 | 0x00 | 0x08 | 0x34 => {}
            0x3c => c.interrupt_line = new as u8,
            o if o == CAP_PM + 4 => c.pmcsr = new & 0x8103,
            o if o == CAP_MSI => c.msi_control = ((new >> 16) as u16) & 0x71,
            o if o == CAP_MSI + 4 => {
                c.msi_address = c.msi_address & !0xffff_ffff | u64::from(new & !3)
            }
            o if o == CAP_MSI + 8 => {
                c.msi_address = c.msi_address & 0xffff_ffff | u64::from(new) << 32
            }
            o if o == CAP_MSI + 12 => c.msi_data = new as u16,
            CFG_BAR0_WRAPPER => c.bar0_wrapper = new & !0xfff,
            CFG_BAR0_WINDOW => c.bar0_window = new & !0xfff,
            CFG_BAR1_WINDOW => c.bar1_window = new,
            o if o == CAP_PCIE + 8 => c.device_control = new as u16,
            o if o == CAP_PCIE + 0x10 => c.link_control = new as u16,
            o if o == CAP_PCIE + 0x24 => c.device_control_2 = new as u16,
            o if o == CAP_PCIE + 0x2c => c.link_control_2 = new as u16,
            // Read-only. `wlan` `0x108810` writes the old (gen1) wrapper
            // window register at 0xAC while it still scans for the bus
            // core; on this core 0xAC is the read-only PCIe capability.
            o if o == CAP_PCIE
                || o == CAP_PCIE + 4
                || o == CAP_PCIE + 0xc
                || o == CAP_PCIE + 0x20 => {}
            o if EXTENDED_CAPABILITIES.iter().any(|cap| cap.0 == o) => {}
            CFG_BAR2_CONFIG => c.bar2_config = new,
            _ => {
                return Err(format!(
                    "write of configuration register {offset:#x} not modeled"
                ));
            }
        }
        Ok(())
    }

    // ----- backplane -----

    fn backplane_read(&mut self, address: u32) -> Result<u32, Refusal> {
        if (EROM_BASE..EROM_BASE + 0x1000).contains(&address) {
            let index = ((address - EROM_BASE) / 4) as usize;
            return erom()
                .get(index)
                .copied()
                .ok_or_else(|| format!("enumeration ROM read past its end at {address:#x}"));
        }
        if let Some(index) = CORES.iter().position(|core| {
            (core.2 + WRAPPER_OFFSET..core.2 + WRAPPER_OFFSET + 0x1000).contains(&address)
        }) {
            let wrapper = self.wrappers[index];
            return match address & 0xfff {
                AI_IOCTRL => Ok(wrapper.ioctrl),
                AI_RESETCTRL => Ok(wrapper.resetctrl),
                // Resets complete at once; no I/O status to report.
                AI_RESETSTATUS | AI_IOSTATUS => Ok(0),
                other => Err(format!(
                    "wrapper register {other:#x} of core {:#x} not modeled",
                    CORES[index].0
                )),
            };
        }
        let offset = address & 0xfff;
        match address & !0xfff {
            CC_BASE => match offset {
                0x00 => Ok(SOCI_AI << 28
                    | (CORES.len() as u32) << 24
                    | CHIP_PACKAGE << 20
                    | CHIP_REV << 16
                    | CHIP_ID),
                0x04 => Ok(CC_CAPABILITIES),
                0x2c => Ok(CC_CHIPSTATUS),
                0x58 => Ok(self.cc_gpio_pull[0]),
                0x5c => Ok(self.cc_gpio_pull[1]),
                0xac => Ok(CC_CAPABILITIES_EXT),
                0xfc => Ok(EROM_BASE),
                0x604 => Ok(CC_PMUCAPABILITIES),
                _ => Err(format!("ChipCommon register {offset:#x} not modeled")),
            },
            ARMCR4_BASE => match offset {
                ARMCR4_CAP => Ok(RAM_BANKS_A | RAM_BANKS_B << 4),
                ARMCR4_BANKIDX => Ok(self.armcr4_bank),
                ARMCR4_BANKINFO if self.armcr4_bank < RAM_BANKS_A + RAM_BANKS_B => {
                    Ok(RAM_BANK_SIZE_FIELD)
                }
                _ => Err(format!("ARM CR4 register {offset:#x} not modeled")),
            },
            PCIE2_BASE => match offset {
                PCIE2_INTMASK => Ok(self.pcie2_intmask),
                PCIE2_MAILBOXINT => Ok(self.mailbox_int),
                PCIE2_MAILBOXMASK => Ok(self.mailbox_mask),
                PCIE2_CONFIGADDR => Ok(self.config_address),
                PCIE2_CONFIGDATA => self.config_dword(self.config_address as u16 & 0xffc),
                _ => Err(format!("PCIe core register {offset:#x} not modeled")),
            },
            _ => Err(format!("backplane read at {address:#x} not modeled")),
        }
    }

    fn backplane_write(
        &mut self,
        address: u32,
        value: u32,
        upstream: &dyn Upstream,
    ) -> Result<(), Refusal> {
        if let Some(index) = CORES.iter().position(|core| {
            (core.2 + WRAPPER_OFFSET..core.2 + WRAPPER_OFFSET + 0x1000).contains(&address)
        }) {
            match address & 0xfff {
                AI_IOCTRL => self.wrappers[index].ioctrl = value,
                AI_RESETCTRL => self.wrappers[index].resetctrl = value & AIRC_RESET,
                other => {
                    return Err(format!(
                        "write of wrapper register {other:#x} of core {:#x}",
                        CORES[index].0
                    ));
                }
            }
            if index == WRAPPER_ARMCR4 {
                self.arm_changed();
            }
            return Ok(());
        }
        let offset = address & 0xfff;
        match address & !0xfff {
            CC_BASE => match offset {
                0x58 => self.cc_gpio_pull[0] = value,
                0x5c => self.cc_gpio_pull[1] = value,
                _ => {
                    return Err(format!(
                        "write of ChipCommon register {offset:#x} = {value:#x}"
                    ));
                }
            },
            ARMCR4_BASE => match offset {
                ARMCR4_BANKIDX => self.armcr4_bank = value,
                _ => {
                    return Err(format!(
                        "write of ARM CR4 register {offset:#x} = {value:#x}"
                    ));
                }
            },
            PCIE2_BASE => match offset {
                PCIE2_INTMASK => self.pcie2_intmask = value,
                PCIE2_MAILBOXINT => {
                    self.mailbox_int &= !value;
                    self.update_interrupt(upstream)?;
                }
                PCIE2_MAILBOXMASK => {
                    self.mailbox_mask = value;
                    self.update_interrupt(upstream)?;
                }
                PCIE2_CONFIGADDR => self.config_address = value,
                PCIE2_CONFIGDATA => {
                    self.config_write_dword(self.config_address as u16 & 0xffc, value, !0)?
                }
                PCIE2_H2D_MAILBOX_0 => self.doorbell(upstream)?,
                _ => {
                    return Err(format!(
                        "write of PCIe core register {offset:#x} = {value:#x}"
                    ));
                }
            },
            _ => {
                return Err(format!(
                    "backplane write at {address:#x} = {value:#x} not modeled"
                ));
            }
        }
        Ok(())
    }

    /// The host changed the ARM core's wrapper: holding it halted marks a
    /// download; releasing it afterwards starts the downloaded firmware.
    fn arm_changed(&mut self) {
        let wrapper = self.wrappers[WRAPPER_ARMCR4];
        let in_reset = wrapper.resetctrl & AIRC_RESET != 0;
        let halted = in_reset || wrapper.ioctrl & SICF_CPUHALT != 0;
        if halted {
            if !self.halted {
                self.halted = true;
                self.dongle = None;
                self.image_end = 0;
            }
            return;
        }
        let runs = wrapper.ioctrl & (SICF_CLOCK_EN | SICF_FGC) == SICF_CLOCK_EN;
        if self.halted && runs {
            self.halted = false;
            self.boot();
        }
    }

    fn ram_word(&self, offset: u32) -> u32 {
        let o = offset as usize;
        u32::from_le_bytes(self.ram[o..o + 4].try_into().unwrap())
    }

    fn set_ram_word(&mut self, offset: u32, value: u32) {
        let o = offset as usize;
        self.ram[o..o + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn set_ram_u16(&mut self, offset: u32, value: u16) {
        let o = offset as usize;
        self.ram[o..o + 2].copy_from_slice(&value.to_le_bytes());
    }

    fn ram_u16(&self, offset: u32) -> u16 {
        let o = offset as usize;
        u16::from_le_bytes(self.ram[o..o + 2].try_into().unwrap())
    }

    /// The firmware starts. Its reset vector must be what the image begins
    /// with, and the NVRAM length token must be intact; otherwise the ARM
    /// runs into whatever is there and the host never sees the shared
    /// structure (it times out, `0xffcc0`).
    fn boot(&mut self) {
        let top = RAM_SIZE - 4;
        let token = self.ram_word(top);
        let words = token & 0xffff;
        if self.reset_vector != self.ram_word(0) || words == 0 || token >> 16 != !words & 0xffff {
            return;
        }
        let nvram_start = top - words * 4;
        let nvram = &self.ram[nvram_start as usize..top as usize];
        let mac = nvram_mac(nvram).unwrap_or_default();
        let nvram = nvram.to_vec();
        let image = &self.ram[..self.image_end as usize];
        let version = trailer_numbers::<4>(image, b"Version: ").unwrap_or_default();
        let ucode = trailer_numbers::<2>(image, b"Ucode Ver: ").unwrap_or_default();
        let version_text = trailer_text(image, b"Version: ", b" CRC").unwrap_or_default();
        let fwid = trailer_text(image, b"FWID 01-", b"\0").unwrap_or_default();
        let build_date = trailer_text(image, b"Date: ", b" Ucode")
            .and_then(|date| build_date(&date))
            .unwrap_or_default();
        // The firmware's own data follows its image: place the shared
        // structure, ring information, ring descriptors and index arrays
        // there.
        let area = (self.image_end + 0xff) & !0xff;
        let shared = area;
        let ring_info = shared + SHARED_SIZE;
        let ring_mem = ring_info + RING_INFO_SIZE;
        let h2d_write = ring_mem + RING_MEM_SIZE * COMMON_RINGS as u32;
        let h2d_read = h2d_write + INDEX_SIZE * u32::from(MAX_SUB_QUEUES);
        let d2h_write = h2d_read + INDEX_SIZE * u32::from(MAX_SUB_QUEUES);
        let d2h_read = d2h_write + INDEX_SIZE * 3;
        let mailboxes = d2h_read + INDEX_SIZE * 3;
        let end = mailboxes + 8;
        if end > nvram_start {
            return;
        }
        self.ram[area as usize..end as usize].fill(0);
        let address = |offset: u32| RAM_BASE + offset;
        self.set_ram_word(shared, SHARED_VERSION);
        // h2d and d2h mailbox data words.
        self.set_ram_word(shared + 40, address(mailboxes));
        self.set_ram_word(shared + 44, address(mailboxes + 4));
        self.set_ram_word(shared + 48, address(ring_info));
        self.set_ram_word(ring_info, address(ring_mem));
        self.set_ram_word(ring_info + 4, address(h2d_write));
        self.set_ram_word(ring_info + 8, address(h2d_read));
        self.set_ram_word(ring_info + 12, address(d2h_write));
        self.set_ram_word(ring_info + 16, address(d2h_read));
        self.set_ram_u16(ring_info + 52, MAX_SUB_QUEUES);
        for ring in 0..COMMON_RINGS as u32 {
            self.set_ram_u16(ring_mem + ring * RING_MEM_SIZE, ring as u16);
        }
        self.set_ram_word(top, address(shared));
        self.dongle = Some(Dongle {
            h2d_write,
            h2d_read,
            d2h_write,
            d2h_read,
            ring_mem,
            rings: None,
            flow_rings: Vec::new(),
            ioctl_buffers: VecDeque::new(),
            event_buffers: VecDeque::new(),
            rx_buffers: VecDeque::new(),
            mac,
            nvram,
            version,
            ucode,
            version_text,
            build_date,
            fwid,
            up: false,
            settings: HashMap::new(),
        });
    }

    fn update_interrupt(&mut self, upstream: &dyn Upstream) -> Result<(), Refusal> {
        let pending = self.mailbox_int & self.mailbox_mask != 0;
        let msi = self.config.msi_control & 1 != 0;
        if msi {
            if pending && !self.interrupt {
                let data = u32::from(self.config.msi_data).to_le_bytes();
                upstream
                    .write(self.config.msi_address, &data)
                    .ok_or_else(|| {
                        format!(
                            "MSI write to {:#x} not accepted upstream",
                            self.config.msi_address
                        )
                    })?;
            }
        } else if self.config.command & 0x400 == 0 {
            upstream.intx(pending);
        }
        self.interrupt = pending;
        Ok(())
    }

    // ----- the firmware's message rings -----

    /// The host rang the doorbell: consume what it submitted.
    fn doorbell(&mut self, upstream: &dyn Upstream) -> Result<(), Refusal> {
        let Some(mut dongle) = self.dongle.take() else {
            return Err("doorbell before the firmware runs".into());
        };
        let result = self.service(&mut dongle, upstream);
        self.dongle = Some(dongle);
        result?;
        self.update_interrupt(upstream)
    }

    fn service(&mut self, dongle: &mut Dongle, upstream: &dyn Upstream) -> Result<(), Refusal> {
        if dongle.rings.is_none() {
            let rings = (0..COMMON_RINGS as u32)
                .map(|ring| {
                    let at = dongle.ring_mem + ring * RING_MEM_SIZE;
                    let base =
                        u64::from(self.ram_word(at + 8)) | u64::from(self.ram_word(at + 12)) << 32;
                    Ring {
                        base,
                        max_items: self.ram_u16(at + 4),
                        item_len: self.ram_u16(at + 6),
                    }
                })
                .collect::<Vec<_>>();
            if rings
                .iter()
                .any(|ring| ring.base == 0 || ring.max_items == 0 || ring.item_len == 0)
            {
                return Err("doorbell before the host described the common rings".into());
            }
            dongle.rings = Some(rings);
        }
        // Submissions: control first, then RX buffer posts.
        for ring in 0..H2D_COMMON_RINGS {
            while let Some(item) = self.next_submission(dongle, ring, upstream)? {
                self.submission(dongle, ring, &item, upstream)?;
            }
        }
        if let Some((id, _)) = dongle.flow_rings.iter().find(|(id, ring)| {
            let slot = u32::from(*id);
            ring.max_items != 0
                && self.ram_u16(dongle.h2d_write + slot * INDEX_SIZE)
                    != self.ram_u16(dongle.h2d_read + slot * INDEX_SIZE)
        }) {
            return Err(format!("transmit on flow ring {id} not modeled"));
        }
        Ok(())
    }

    /// The next item the host submitted on H2D ring `ring`, if any.
    fn next_submission(
        &mut self,
        dongle: &Dongle,
        ring: u16,
        upstream: &dyn Upstream,
    ) -> Result<Option<Vec<u8>>, Refusal> {
        let rings = dongle.rings.as_ref().unwrap();
        let desc = &rings[usize::from(ring)];
        let slot = u32::from(ring) * INDEX_SIZE;
        let write = self.ram_u16(dongle.h2d_write + slot);
        let read = self.ram_u16(dongle.h2d_read + slot);
        if write == read {
            return Ok(None);
        }
        if write >= desc.max_items {
            return Err(format!(
                "H2D ring {ring} write index {write} beyond {} items",
                desc.max_items
            ));
        }
        let mut item = vec![0; usize::from(desc.item_len)];
        let at = desc.base + u64::from(read) * u64::from(desc.item_len);
        upstream
            .read(at, &mut item)
            .ok_or_else(|| format!("H2D ring {ring} item at {at:#x} not readable"))?;
        let next = (read + 1) % desc.max_items;
        self.set_ram_u16(dongle.h2d_read + slot, next);
        Ok(Some(item))
    }

    fn submission(
        &mut self,
        dongle: &mut Dongle,
        ring: u16,
        item: &[u8],
        upstream: &dyn Upstream,
    ) -> Result<(), Refusal> {
        let kind = item[0];
        let if_id = item[1];
        let request_id = le32(item, 4);
        let buffer = |item: &[u8], len_at: usize, address_at: usize| HostBuffer {
            request_id,
            address: le64(item, address_at),
            len: le16(item, len_at),
        };
        match (ring, kind) {
            (1, MSG_RXBUF_POST) => dongle.rx_buffers.push_back(buffer(item, 10, 24)),
            (0, MSG_IOCTLRESP_BUF_POST) => dongle.ioctl_buffers.push_back(buffer(item, 8, 16)),
            (0, MSG_EVENT_BUF_POST) => dongle.event_buffers.push_back(buffer(item, 8, 16)),
            (0, MSG_IOCTLPTR_REQ) => self.ioctl(dongle, item, upstream)?,
            (0, MSG_FLOW_RING_CREATE) => {
                let id = le16(item, 22);
                let ring = Ring {
                    base: le64(item, 32),
                    max_items: le16(item, 28),
                    item_len: le16(item, 30),
                };
                if !(H2D_COMMON_RINGS..MAX_SUB_QUEUES).contains(&id) {
                    return Err(format!("flow ring {id} out of range"));
                }
                dongle.flow_rings.retain(|(other, _)| *other != id);
                dongle.flow_rings.push((id, ring));
                let mut reply = header(MSG_FLOW_RING_CREATE_CMPLT, if_id, request_id);
                reply[10..12].copy_from_slice(&id.to_le_bytes());
                self.complete(dongle, RING_D2H_CONTROL, &reply, upstream)?;
            }
            (0, MSG_FLOW_RING_DELETE | MSG_FLOW_RING_FLUSH) => {
                let id = le16(item, 8);
                let reply_kind = if kind == MSG_FLOW_RING_DELETE {
                    dongle.flow_rings.retain(|(other, _)| *other != id);
                    MSG_FLOW_RING_DELETE_CMPLT
                } else {
                    MSG_FLOW_RING_FLUSH_CMPLT
                };
                let mut reply = header(reply_kind, if_id, request_id);
                reply[10..12].copy_from_slice(&id.to_le_bytes());
                self.complete(dongle, RING_D2H_CONTROL, &reply, upstream)?;
            }
            _ => {
                return Err(format!(
                    "H2D ring {ring} message type {kind:#x} not modeled: {item:02x?}"
                ));
            }
        }
        Ok(())
    }

    /// A control request: acknowledge it, carry it out, and complete it into
    /// a posted response buffer (`wlan` takes the acknowledgement first,
    /// `0x105620`, then the completion, `0x105670`).
    fn ioctl(
        &mut self,
        dongle: &mut Dongle,
        item: &[u8],
        upstream: &dyn Upstream,
    ) -> Result<(), Refusal> {
        let if_id = item[1];
        let request_id = le32(item, 4);
        let command = le32(item, 8);
        let transaction = le16(item, 12);
        let input_len = le16(item, 14);
        let output_len = le16(item, 16);
        let address = le64(item, 24);
        let mut input = vec![0; usize::from(input_len)];
        upstream
            .read(address, &mut input)
            .ok_or_else(|| format!("ioctl buffer at {address:#x} not readable"))?;
        let outcome = dongle_ioctl(dongle, if_id, command, &input)?;
        let mut ack = header(MSG_IOCTLPTR_REQ_ACK, if_id, request_id);
        ack[12..16].copy_from_slice(&command.to_le_bytes());
        self.complete(dongle, RING_D2H_CONTROL, &ack, upstream)?;
        let buffer = dongle
            .ioctl_buffers
            .pop_front()
            .ok_or("ioctl completion with no response buffer posted")?;
        let (status, data) = match outcome {
            Ok(data) => (0, data),
            Err(code) => (code, Vec::new()),
        };
        let len = data
            .len()
            .min(usize::from(output_len))
            .min(usize::from(buffer.len));
        if len != 0 {
            upstream
                .write(buffer.address, &data[..len])
                .ok_or_else(|| {
                    format!(
                        "ioctl response buffer at {:#x} not writable",
                        buffer.address
                    )
                })?;
        }
        let mut done = header(MSG_IOCTL_CMPLT, if_id, buffer.request_id);
        done[8..10].copy_from_slice(&status.to_le_bytes());
        done[12..14].copy_from_slice(&(len as u16).to_le_bytes());
        done[14..16].copy_from_slice(&transaction.to_le_bytes());
        done[16..20].copy_from_slice(&command.to_le_bytes());
        self.complete(dongle, RING_D2H_CONTROL, &done, upstream)
    }

    /// Writes `message` to D2H ring `ring` and raises the doorbell interrupt.
    fn complete(
        &mut self,
        dongle: &Dongle,
        ring: usize,
        message: &[u8],
        upstream: &dyn Upstream,
    ) -> Result<(), Refusal> {
        let desc = &dongle.rings.as_ref().unwrap()[ring];
        let slot = (ring as u32 - u32::from(H2D_COMMON_RINGS)) * INDEX_SIZE;
        let write = self.ram_u16(dongle.d2h_write + slot);
        let read = self.ram_u16(dongle.d2h_read + slot);
        let next = (write + 1) % desc.max_items;
        if next == read {
            return Err(format!("D2H ring {ring} full"));
        }
        let mut item = vec![0; usize::from(desc.item_len)];
        let len = message.len().min(item.len());
        item[..len].copy_from_slice(&message[..len]);
        let at = desc.base + u64::from(write) * u64::from(desc.item_len);
        upstream
            .write(at, &item)
            .ok_or_else(|| format!("D2H ring {ring} item at {at:#x} not writable"))?;
        self.set_ram_u16(dongle.d2h_write + slot, next);
        self.mailbox_int |= MAILBOX_D2H_DB;
        Ok(())
    }

    // ----- the BARs -----

    fn bar0_read(&mut self, offset: u64, size: u8) -> Result<u64, Refusal> {
        if size != 4 || !offset.is_multiple_of(4) {
            return Err(format!("{size}-byte register access at BAR0 {offset:#x}"));
        }
        let offset = offset as u32;
        let address = self.bar0_address(offset)?;
        self.backplane_read(address).map(u64::from)
    }

    fn bar0_write(
        &mut self,
        offset: u64,
        size: u8,
        value: u64,
        upstream: &dyn Upstream,
    ) -> Result<(), Refusal> {
        if size != 4 || !offset.is_multiple_of(4) {
            return Err(format!("{size}-byte register write at BAR0 {offset:#x}"));
        }
        let address = self.bar0_address(offset as u32)?;
        self.backplane_write(address, value as u32, upstream)
    }

    /// BAR0 layout of a PCIe gen2 core: the configurable window, the window
    /// onto that core's wrapper, the PCIe core and ChipCommon (`wlan`
    /// `0x107480`, `0x108810`).
    fn bar0_address(&self, offset: u32) -> Result<u32, Refusal> {
        Ok(match offset {
            0x0000..0x1000 => self.config.bar0_window + offset,
            0x1000..0x2000 => self.config.bar0_wrapper + offset - 0x1000,
            0x2000..0x3000 => PCIE2_BASE + offset - 0x2000,
            0x3000..0x4000 => CC_BASE + offset - 0x3000,
            _ => return Err(format!("BAR0 offset {offset:#x} not modeled")),
        })
    }

    /// RAM window (BAR2, window base from configuration 0x84, which `wlan`
    /// sets to 0): address 0 holds the ARM's reset vector, the RAM is at
    /// its base.
    fn ram_range(&self, offset: u64, size: u8) -> Result<Option<usize>, Refusal> {
        if self.config.bar1_window != 0 {
            return Err(format!(
                "RAM window base {:#x} not modeled",
                self.config.bar1_window
            ));
        }
        let end = offset + u64::from(size);
        if end <= 4 {
            return Ok(None);
        }
        if offset >= u64::from(RAM_BASE) && end <= u64::from(RAM_BASE + RAM_SIZE) {
            return Ok(Some((offset - u64::from(RAM_BASE)) as usize));
        }
        Err(format!("RAM window access at {offset:#x} outside the RAM"))
    }

    fn bar2_read(&self, offset: u64, size: u8) -> Result<u64, Refusal> {
        let mut bytes = [0; 8];
        let n = usize::from(size);
        match self.ram_range(offset, size)? {
            None => bytes[..n].copy_from_slice(
                &self.reset_vector.to_le_bytes()[offset as usize..offset as usize + n],
            ),
            Some(o) => bytes[..n].copy_from_slice(&self.ram[o..o + n]),
        }
        Ok(u64::from_le_bytes(bytes))
    }

    fn bar2_write(&mut self, offset: u64, size: u8, value: u64) -> Result<(), Refusal> {
        let n = usize::from(size);
        let bytes = value.to_le_bytes();
        match self.ram_range(offset, size)? {
            None => {
                let mut vector = self.reset_vector.to_le_bytes();
                vector[offset as usize..offset as usize + n].copy_from_slice(&bytes[..n]);
                self.reset_vector = u32::from_le_bytes(vector);
            }
            Some(o) => {
                self.ram[o..o + n].copy_from_slice(&bytes[..n]);
                // The NVRAM sits at the top; what is written below it while
                // the ARM is held is the image.
                if self.halted && (o + n) as u32 <= RAM_SIZE - 0x4000 {
                    self.image_end = self.image_end.max((o + n) as u32);
                }
            }
        }
        Ok(())
    }
}

/// The firmware's answer to control request `command` on interface `if_id`:
/// the response data, or a Broadcom error code. Requests not modeled are
/// refused (the machine stops and says which).
///
/// What `wlan` asks, in order (traced): revision info; `ampdu_hostreorder`,
/// the event mask, down; the country (Nintendo's "XV"/984), the valid
/// channels, the version string; and once nifm opens `wl0`, the station
/// configuration (keys, power save, rates, bandwidth, chains, coexistence,
/// the Nintendo-specific `nsc_*` filters) and up. With no network in range
/// nothing follows: the radio idles. Settings the firmware only stores
/// are accepted and remembered; numeric commands with arguments are
/// commands from public `wlioctl_defs.h`.
fn dongle_ioctl(
    dongle: &mut Dongle,
    if_id: u8,
    command: u32,
    input: &[u8],
) -> Result<Result<Vec<u8>, i16>, Refusal> {
    let (name, value) = match command {
        WLC_GET_VAR | WLC_SET_VAR => {
            let end = input.iter().position(|&b| b == 0).unwrap_or(input.len());
            (
                String::from_utf8_lossy(&input[..end]).into_owned(),
                input.get(end + 1..).unwrap_or(&[]),
            )
        }
        WLC_GET_REVINFO => return Ok(Ok(revinfo(dongle))),
        WLC_UP => {
            dongle.up = true;
            return Ok(Ok(Vec::new()));
        }
        WLC_DOWN => {
            dongle.up = false;
            return Ok(Ok(Vec::new()));
        }
        // Settings taking one word, which the firmware stores: infrastructure
        // mode, channel, power save mode, access point mode (off), band.
        WLC_SET_INFRA | WLC_SET_CHANNEL | WLC_SET_PM | WLC_SET_AP | WLC_SET_BAND => {
            let Some(word) = input.get(..4) else {
                return Ok(Err(BCME_BADARG));
            };
            dongle
                .settings
                .insert(format!("ioctl {command}"), word.to_vec());
            return Ok(Ok(Vec::new()));
        }
        WLC_GET_BANDLIST => return Ok(Ok(words(&BANDLIST))),
        WLC_GET_VALID_CHANNELS => {
            let mut list = vec![CHANNELS.len() as u32];
            list.extend(CHANNELS.iter().map(|&c| u32::from(c)));
            return Ok(Ok(words(&list)));
        }
        _ => {
            return Err(format!(
                "ioctl {command} on interface {if_id} not modeled: {:02x?}",
                &input[..input.len().min(32)]
            ));
        }
    };
    let set = command == WLC_SET_VAR;
    Ok(match (set, name.as_str()) {
        // Channel flags for a chanspec: valid in hardware and software
        // (1 | 2), 5 GHz (4) for the 5 GHz ones; channels outside the list
        // are not valid (0). No radar or passive flags: assumed for the
        // W52 channels the list holds.
        (false, "per_chan_info") => {
            let channel = value.first().copied().unwrap_or(0);
            let flags = match CHANNELS.contains(&channel) {
                true if channel > 14 => 1 | 2 | 4,
                true => 1 | 2,
                false => 0,
            };
            Ok(words(&[flags]))
        }
        // The station address: from the NVRAM until the host sets its own.
        (false, "cur_etheraddr") => Ok(dongle.mac.to_vec()),
        (true, "cur_etheraddr") => match value.get(..6) {
            Some(mac) => {
                dongle.mac.copy_from_slice(mac);
                Ok(Vec::new())
            }
            None => Err(BCME_BADARG),
        },
        (false, "ver") => Ok(version_string(dongle)),
        (false, "nmode") => Ok(NMODE.to_le_bytes().to_vec()),
        (false, "mimo_bw_cap") => Ok(MIMO_BW_CAP.to_le_bytes().to_vec()),
        (false, "chanspecs") => {
            let mut list = vec![CHANNELS.len() as u32];
            list.extend(CHANNELS.iter().map(|&c| u32::from(chanspec_20(c))));
            Ok(words(&list))
        }
        (true, name) if SETTINGS.contains(&name) || READ_BACK.contains(&name) => {
            dongle.settings.insert(name.to_owned(), value.to_vec());
            Ok(Vec::new())
        }
        // Settings the host reads before changing them: what it last set,
        // else the firmware's power-on state, which is nothing enabled (no
        // events, no Nintendo-specific subtypes, no coexistence override).
        (false, name) if READ_BACK.contains(&name) => {
            let mut data = dongle.settings.get(name).cloned().unwrap_or_default();
            data.resize(value.len(), 0);
            Ok(data)
        }
        _ => {
            let verb = if set { "set" } else { "get" };
            return Err(format!(
                "{verb} of iovar \"{name}\" on interface {if_id} not modeled: {:02x?}",
                &value[..value.len().min(32)]
            ));
        }
    })
}

/// Iovars `wlan` sets and the firmware just stores.
const SETTINGS: [&str; 37] = [
    "ampdu_hostreorder",
    "country",
    "buf_key_b4_m4",
    "wsec_key",
    "pmkid_info",
    "mpc",
    "nsc_subtypeoui",
    "bcn_timeout",
    "wpaie",
    "maxassoc",
    "wsec",
    "wpa_auth",
    "roam_off",
    "af_flying_max",
    "ds3_mode",
    "sgi_tx",
    "sgi_rx",
    "vhtmode",
    "bw_cap",
    "txchain",
    "rxchain",
    "2g_mrate",
    "5g_mrate",
    "2g_rate",
    "5g_rate",
    "force_maf_rspec",
    "force_af_rspec",
    "ldpc_cap",
    "ampdu_tid",
    "pm2_sleep_ret_ext",
    "aci_max_desense_bphy",
    "aci_max_desense_ofdm",
    "apsta",
    "arp_ol",
    "arpoe",
    "bcn_li_dtim",
    "obss_coex",
];
/// Iovars `wlan` reads, modifies and writes back.
const READ_BACK: [&str; 3] = ["event_msgs", "nsc_subtypes", "btc_dynctl"];

fn words(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// A 20 MHz chanspec (802.11ac encoding, public `bcmwifi_channels.h`):
/// channel, bandwidth 20 (0x1000), band 2.4 GHz (0) or 5 GHz (0xc000).
fn chanspec_20(channel: u8) -> u16 {
    let band = if channel > 14 { 0xc000 } else { 0 };
    0x1000 | band | u16::from(channel)
}

/// "wl0: <build date> version <version> (<hash>) FWID 01-<id>", the format
/// Broadcom firmware answers "ver" with, filled from the image's trailer
/// ("Version: 7.35.252.108 (c216b7c) CRC: … Date: Sun 2021-08-29 19:24:39
/// PDT … FWID 01-65333037").
fn version_string(dongle: &Dongle) -> Vec<u8> {
    let mut text = format!(
        "wl0: {} version {} FWID 01-{}\n",
        dongle.build_date, dongle.version_text, dongle.fwid
    );
    text.push('\0');
    text.into_bytes()
}

/// `wlc_rev_info_t` up to `nvramrev`, the 17 words `wlan` asks for (68
/// bytes, `0x103490`; it only needs the request to succeed). Sources:
/// identity constants and the downloaded NVRAM (`vendid`, `devid`,
/// `boardtype`, `boardrev`, `NVRAMRev`) and image trailer (version,
/// microcode). Radio, PHY and analog revisions only the silicon reports;
/// they are assumed (named constants).
fn revinfo(dongle: &Dongle) -> Vec<u8> {
    let nv = |key: &str| nvram_number(&dongle.nvram, key);
    let [major, minor, rc, increment] = dongle.version;
    let vendor = nv("vendid").unwrap_or(u32::from(VENDOR_ID));
    let words = [
        vendor,
        nv("devid").unwrap_or(u32::from(DEVICE_ID)),
        RADIO_REV,
        CHIP_REV,
        u32::from(D11_REV),
        nv("boardtype").unwrap_or(0),
        nv("boardvendor").unwrap_or(vendor),
        nv("boardrev").unwrap_or(0),
        major << 24 | minor << 16 | rc << 8 | increment,
        dongle.ucode[0] << 16 | dongle.ucode[1],
        BUS_PCIE,
        CHIP_ID,
        PHY_TYPE_AC,
        PHY_REV,
        ANA_REV,
        CHIP_PACKAGE,
        nv("NVRAMRev").unwrap_or(0),
    ];
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// The value of NVRAM variable `key` as a number (decimal or 0x hex).
fn nvram_number(nvram: &[u8], key: &str) -> Option<u32> {
    let entry = nvram
        .split(|&b| b == 0)
        .find_map(|entry| entry.strip_prefix(key.as_bytes())?.strip_prefix(b"="))?;
    let text = std::str::from_utf8(entry).ok()?;
    match text.strip_prefix("0x") {
        Some(hex) => u32::from_str_radix(hex, 16).ok(),
        None => text.parse().ok(),
    }
}

/// The `N` dot-separated numbers after `label` in the image.
fn trailer_numbers<const N: usize>(image: &[u8], label: &[u8]) -> Option<[u32; N]> {
    let at = image.windows(label.len()).rposition(|w| w == label)? + label.len();
    let text: Vec<u8> = image[at..]
        .iter()
        .copied()
        .take_while(|b| b.is_ascii_digit() || *b == b'.')
        .collect();
    let text = std::str::from_utf8(&text).ok()?;
    let mut numbers = [0; N];
    let mut parts = text.split('.');
    for number in &mut numbers {
        *number = parts.next()?.parse().ok()?;
    }
    Some(numbers)
}

/// The text between `label` and `end` in the image's trailer.
fn trailer_text(image: &[u8], label: &[u8], end: &[u8]) -> Option<String> {
    let at = image.windows(label.len()).rposition(|w| w == label)? + label.len();
    let len = image[at..].windows(end.len()).position(|w| w == end)?;
    String::from_utf8(image[at..at + len].to_vec()).ok()
}

/// "Sun 2021-08-29 19:24:39 PDT" as a C `__DATE__ __TIME__` pair,
/// "Aug 29 2021 19:24:39".
fn build_date(date: &str) -> Option<String> {
    const MONTHS: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let mut fields = date.split(' ');
    fields.next()?;
    let mut day = fields.next()?.split('-');
    let (year, month, day) = (day.next()?, day.next()?.parse::<usize>().ok()?, day.next()?);
    let time = fields.next()?;
    Some(format!(
        "{} {day} {year} {time}",
        MONTHS.get(month.checked_sub(1)?)?
    ))
}

/// `macaddr=` from NVRAM text (NUL-separated `key=value` entries).
fn nvram_mac(nvram: &[u8]) -> Option<[u8; 6]> {
    let entry = nvram
        .split(|&b| b == 0)
        .find_map(|entry| entry.strip_prefix(b"macaddr="))?;
    let text = std::str::from_utf8(entry).ok()?;
    let mut mac = [0; 6];
    let mut parts = text.split(':');
    for byte in &mut mac {
        *byte = u8::from_str_radix(parts.next()?, 16).ok()?;
    }
    Some(mac)
}

/// Common message header: type, interface, flags, epoch, request ID.
fn header(kind: u8, if_id: u8, request_id: u32) -> Vec<u8> {
    let mut message = vec![0; 24];
    message[0] = kind;
    message[1] = if_id;
    message[4..8].copy_from_slice(&request_id.to_le_bytes());
    message
}

fn le16(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes(bytes[at..at + 2].try_into().unwrap())
}

fn le32(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}

fn le64(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
}

impl PcieFunction for Bcm4356Wifi {
    fn config_read(&self, offset: u16, size: u8) -> Option<u32> {
        let state = self.state.lock().unwrap();
        let shift = u32::from(offset & 3) * 8;
        let result = state.config_dword(offset & !3).map(|dword| {
            let value = dword >> shift;
            match size {
                1 => value & 0xff,
                2 => value & 0xffff,
                _ => value,
            }
        });
        drop(state);
        self.served(result)
    }

    fn config_write(&self, offset: u16, size: u8, value: u32) -> Option<()> {
        let shift = u32::from(offset & 3) * 8;
        let mask = match size {
            1 => 0xff,
            2 => 0xffff,
            _ => !0,
        } << shift;
        let result =
            self.state
                .lock()
                .unwrap()
                .config_write_dword(offset & !3, value << shift, mask);
        self.served(result)
    }

    fn memory_read(&self, address: u64, size: u8) -> Option<u64> {
        let mut state = self.state.lock().unwrap();
        let result = if state.config.command & 2 == 0 {
            Err(format!(
                "memory read at {address:#x} with memory space disabled"
            ))
        } else if let Some(offset) = address
            .checked_sub(state.bar_address(0, BAR0_SIZE))
            .filter(|&o| o < BAR0_SIZE)
        {
            state.bar0_read(offset, size)
        } else if let Some(offset) = address
            .checked_sub(state.bar_address(1, BAR2_SIZE))
            .filter(|&o| o < BAR2_SIZE)
        {
            state.bar2_read(offset, size)
        } else {
            Err(format!("memory read at {address:#x} outside the BARs"))
        };
        drop(state);
        self.served(result)
    }

    fn memory_write(&self, address: u64, size: u8, value: u64) -> Option<()> {
        let mut state = self.state.lock().unwrap();
        let result = if state.config.command & 2 == 0 {
            Err(format!(
                "memory write at {address:#x} with memory space disabled"
            ))
        } else if let Some(offset) = address
            .checked_sub(state.bar_address(0, BAR0_SIZE))
            .filter(|&o| o < BAR0_SIZE)
        {
            state.bar0_write(offset, size, value, self.upstream.as_ref())
        } else if let Some(offset) = address
            .checked_sub(state.bar_address(1, BAR2_SIZE))
            .filter(|&o| o < BAR2_SIZE)
        {
            state.bar2_write(offset, size, value)
        } else {
            Err(format!("memory write at {address:#x} outside the BARs"))
        };
        drop(state);
        self.served(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Host memory at bus addresses `HOST..HOST + 64 KiB`; MSI writes are
    /// recorded instead.
    const HOST: u64 = 0x8000_0000;
    const MSI: u64 = 0xfd_0000;
    #[derive(Default)]
    struct Host {
        memory: Mutex<Vec<u8>>,
        msis: Mutex<Vec<u32>>,
    }
    struct HostPort(Arc<Host>);
    impl Upstream for HostPort {
        fn read(&self, address: u64, buffer: &mut [u8]) -> Option<()> {
            let o = usize::try_from(address.checked_sub(HOST)?).ok()?;
            buffer.copy_from_slice(self.0.memory.lock().unwrap().get(o..o + buffer.len())?);
            Some(())
        }
        fn write(&self, address: u64, data: &[u8]) -> Option<()> {
            if address == MSI {
                self.0
                    .msis
                    .lock()
                    .unwrap()
                    .push(u32::from_le_bytes(data.try_into().ok()?));
                return Some(());
            }
            let o = usize::try_from(address.checked_sub(HOST)?).ok()?;
            self.0
                .memory
                .lock()
                .unwrap()
                .get_mut(o..o + data.len())?
                .copy_from_slice(data);
            Some(())
        }
        fn intx(&self, _level: bool) {}
    }

    const BAR0: u64 = 0x1000_0000;
    const BAR2: u64 = 0x2000_0000;

    fn setup() -> (Bcm4356Wifi, Arc<Host>) {
        let host = Arc::new(Host::default());
        host.memory.lock().unwrap().resize(0x1_0000, 0);
        let wifi = Bcm4356Wifi::new(Box::new(HostPort(Arc::clone(&host))), Arc::default());
        wifi.config_write(0x10, 4, BAR0 as u32).unwrap();
        wifi.config_write(0x18, 4, BAR2 as u32).unwrap();
        wifi.config_write(0x04, 2, 0x6).unwrap();
        (wifi, host)
    }

    fn put(host: &Host, address: u64, data: &[u8]) {
        let o = (address - HOST) as usize;
        host.memory.lock().unwrap()[o..o + data.len()].copy_from_slice(data);
    }

    fn get(host: &Host, address: u64, len: usize) -> Vec<u8> {
        let o = (address - HOST) as usize;
        host.memory.lock().unwrap()[o..o + len].to_vec()
    }

    /// Downloads an image and NVRAM the way `wlan` does and releases the
    /// ARM; returns the RAM offset of the shared structure.
    fn boot(wifi: &Bcm4356Wifi) -> u32 {
        // Halt the ARM through its wrapper.
        wifi.config_write(CFG_BAR0_WRAPPER, 4, ARMCR4_BASE + WRAPPER_OFFSET)
            .unwrap();
        wifi.memory_write(
            BAR0 + 0x1000 + u64::from(AI_IOCTRL),
            4,
            u64::from(SICF_CPUHALT | SICF_FGC | SICF_CLOCK_EN),
        )
        .unwrap();
        wifi.memory_write(BAR0 + 0x1000 + u64::from(AI_RESETCTRL), 4, 0)
            .unwrap();
        let image: Vec<u8> = (0..0x2000u32).map(|i| (i * 7) as u8).collect();
        for (i, chunk) in image.chunks(8).enumerate() {
            let value = u64::from_le_bytes(chunk.try_into().unwrap());
            wifi.memory_write(BAR2 + u64::from(RAM_BASE) + 8 * i as u64, 8, value)
                .unwrap();
        }
        let vector = u32::from_le_bytes(image[..4].try_into().unwrap());
        wifi.memory_write(BAR2, 4, u64::from(vector)).unwrap();
        let mut nvram = b"boardrev=0x1250\0macaddr=00:90:4c:1a:a0:01\0\0".to_vec();
        nvram.resize(nvram.len().next_multiple_of(4), 0);
        let top = u64::from(RAM_BASE + RAM_SIZE - 4);
        for (i, b) in nvram.iter().enumerate() {
            wifi.memory_write(BAR2 + top - nvram.len() as u64 + i as u64, 1, u64::from(*b))
                .unwrap();
        }
        let words = (nvram.len() / 4) as u32;
        wifi.memory_write(BAR2 + top, 4, u64::from(words | (!words & 0xffff) << 16))
            .unwrap();
        // Release: clocked, not halted.
        wifi.memory_write(
            BAR0 + 0x1000 + u64::from(AI_IOCTRL),
            4,
            u64::from(SICF_CLOCK_EN),
        )
        .unwrap();
        let shared = wifi.memory_read(BAR2 + top, 4).unwrap() as u32;
        assert!(
            (RAM_BASE..RAM_BASE + RAM_SIZE - 4).contains(&shared),
            "shared at {shared:#x}"
        );
        shared
    }

    #[test]
    fn enumeration_and_banks() {
        let (wifi, _) = setup();
        assert_eq!(wifi.config_read(0, 4), Some(0x43ec_14e4));
        wifi.config_write(CFG_BAR0_WINDOW, 4, CC_BASE).unwrap();
        let chip = wifi.memory_read(BAR0, 4).unwrap();
        assert_eq!(chip & 0xffff, 0x4356);
        assert_eq!(chip >> 28, 1);
        let erom = wifi.memory_read(BAR0 + 0xfc, 4).unwrap() as u32;
        wifi.config_write(CFG_BAR0_WINDOW, 4, erom).unwrap();
        // First component: ChipCommon by Broadcom.
        let cia = wifi.memory_read(BAR0, 4).unwrap() as u32;
        assert_eq!((cia >> 8) & 0xfff, u32::from(CORE_CC));
        assert_eq!(cia >> 20, MFG_BRCM);
        wifi.config_write(CFG_BAR0_WINDOW, 4, ARMCR4_BASE).unwrap();
        let cap = wifi.memory_read(BAR0 + u64::from(ARMCR4_CAP), 4).unwrap() as u32;
        let banks = (cap & 0xf) + (cap >> 4 & 0xf);
        let mut size = 0;
        for bank in 0..banks {
            wifi.memory_write(BAR0 + u64::from(ARMCR4_BANKIDX), 4, u64::from(bank))
                .unwrap();
            size += ((wifi
                .memory_read(BAR0 + u64::from(ARMCR4_BANKINFO), 4)
                .unwrap() as u32
                & 0x3f)
                + 1)
                * 0x2000;
        }
        assert_eq!(size, RAM_SIZE);
        assert!(size >= 0x8e7d9);
    }

    #[test]
    fn firmware_publishes_shared_structure_after_release() {
        let (wifi, _) = setup();
        let shared = boot(&wifi);
        assert_eq!(
            wifi.memory_read(BAR2 + u64::from(shared), 4),
            Some(u64::from(SHARED_VERSION))
        );
        let ring_info = wifi.memory_read(BAR2 + u64::from(shared) + 48, 4).unwrap();
        assert_eq!(
            wifi.memory_read(BAR2 + ring_info + 52, 2),
            Some(u64::from(MAX_SUB_QUEUES))
        );
    }

    #[test]
    fn control_request_is_acknowledged_and_completed() {
        let (wifi, host) = setup();
        let shared = u64::from(boot(&wifi));
        let ring_info = wifi.memory_read(BAR2 + shared + 48, 4).unwrap();
        let ring_mem = wifi.memory_read(BAR2 + ring_info, 4).unwrap();
        let h2d_w = wifi.memory_read(BAR2 + ring_info + 4, 4).unwrap();
        let d2h_w = wifi.memory_read(BAR2 + ring_info + 12, 4).unwrap();
        // Common rings: (base, items, item length).
        let rings = [
            (HOST, 64, 40),
            (HOST + 0x1000, 64, 32),
            (HOST + 0x2000, 64, 24),
            (HOST + 0x3000, 64, 16),
            (HOST + 0x4000, 64, 32),
        ];
        for (i, (base, items, len)) in rings.iter().enumerate() {
            let at = BAR2 + ring_mem + 16 * i as u64;
            wifi.memory_write(at + 4, 2, *items).unwrap();
            wifi.memory_write(at + 6, 2, *len).unwrap();
            wifi.memory_write(at + 8, 8, *base).unwrap();
        }
        // Interrupts by MSI, D2H doorbells enabled as `wlan` does.
        wifi.config_write(CAP_MSI + 4, 4, MSI as u32).unwrap();
        wifi.config_write(CAP_MSI + 12, 2, 3).unwrap();
        wifi.config_write(CAP_MSI + 2, 2, 1).unwrap();
        wifi.memory_write(BAR0 + 0x2000 + u64::from(PCIE2_MAILBOXMASK), 4, 0xff0300)
            .unwrap();
        // A response buffer, then "get cur_etheraddr".
        let response = HOST + 0x8000;
        let request = HOST + 0x9000;
        let mut post = vec![0; 40];
        post[0] = MSG_IOCTLRESP_BUF_POST;
        post[4..8].copy_from_slice(&7u32.to_le_bytes());
        post[8..10].copy_from_slice(&8160u16.to_le_bytes());
        post[16..24].copy_from_slice(&response.to_le_bytes());
        put(&host, HOST, &post);
        put(&host, request, b"cur_etheraddr\0\0\0\0\0\0\0");
        let mut ioctl = vec![0; 40];
        ioctl[0] = MSG_IOCTLPTR_REQ;
        ioctl[4..8].copy_from_slice(&0xfffeu32.to_le_bytes());
        ioctl[8..12].copy_from_slice(&WLC_GET_VAR.to_le_bytes());
        ioctl[12..14].copy_from_slice(&1u16.to_le_bytes());
        ioctl[14..16].copy_from_slice(&32u16.to_le_bytes());
        ioctl[16..18].copy_from_slice(&32u16.to_le_bytes());
        ioctl[24..32].copy_from_slice(&request.to_le_bytes());
        put(&host, HOST + 40, &ioctl);
        wifi.memory_write(BAR2 + h2d_w, 2, 2).unwrap();
        wifi.memory_write(
            BAR0 + 0x2000 + u64::from(PCIE2_H2D_MAILBOX_0),
            4,
            0x1234_5678,
        )
        .unwrap();
        // Both submissions consumed; two control completions written.
        let h2d_r = wifi.memory_read(BAR2 + ring_info + 8, 4).unwrap();
        assert_eq!(wifi.memory_read(BAR2 + h2d_r, 2), Some(2));
        assert_eq!(wifi.memory_read(BAR2 + d2h_w, 2), Some(2));
        let ack = get(&host, HOST + 0x2000, 24);
        assert_eq!(ack[0], MSG_IOCTLPTR_REQ_ACK);
        assert_eq!(le32(&ack, 4), 0xfffe);
        let done = get(&host, HOST + 0x2000 + 24, 24);
        assert_eq!(done[0], MSG_IOCTL_CMPLT);
        assert_eq!(le32(&done, 4), 7, "completes into the posted buffer");
        assert_eq!(le16(&done, 8), 0, "status");
        assert_eq!(le16(&done, 12), 6, "response length");
        assert_eq!(le16(&done, 14), 1, "transaction");
        assert_eq!(
            get(&host, response, 6),
            [0x00, 0x90, 0x4c, 0x1a, 0xa0, 0x01]
        );
        // The doorbell interrupt: mailbox bit set, one MSI sent.
        assert_eq!(
            wifi.memory_read(BAR0 + 0x2000 + u64::from(PCIE2_MAILBOXINT), 4),
            Some(u64::from(MAILBOX_D2H_DB))
        );
        assert_eq!(*host.msis.lock().unwrap(), [3]);
        wifi.memory_write(
            BAR0 + 0x2000 + u64::from(PCIE2_MAILBOXINT),
            4,
            u64::from(MAILBOX_D2H_DB),
        )
        .unwrap();
        assert_eq!(
            wifi.memory_read(BAR0 + 0x2000 + u64::from(PCIE2_MAILBOXINT), 4),
            Some(0)
        );
    }

    #[test]
    fn unknown_request_is_refused() {
        let (wifi, _) = setup();
        assert_eq!(wifi.config_read(0x2c, 4), None);
        wifi.config_write(CFG_BAR0_WINDOW, 4, D11_BASE).unwrap();
        assert_eq!(wifi.memory_read(BAR0, 4), None);
    }
}
