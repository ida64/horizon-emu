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
    sync::{Arc, Condvar, Mutex},
    time::Duration,
};

use serde::{Deserialize, Serialize};

use super::{
    Notes,
    pcie_endpoint::{PcieFunction, Upstream},
    wifi_ap::{self, AccessPoint},
};
use crate::{
    clock::GuestClock,
    snapshot::{Snapshot, SnapshotError, decode, encode},
};

mod on_air;
use on_air::{Effect, Radio};

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
/// Default scan type, 0 active / 1 passive (`WLC_SET_PASSIVE_SCAN`).
const WLC_SET_PASSIVE_SCAN: u32 = 49;
const WLC_GET_VALID_CHANNELS: u32 = 217;
/// Beacon and DTIM periods of the BSS the firmware runs as access point,
/// whether the interface is up, and sending a station of that BSS away
/// (public `wlioctl_defs.h`).
const WLC_SET_BCNPRD: u32 = 76;
const WLC_SET_DTIMPRD: u32 = 78;
const WLC_GET_UP: u32 = 162;
const WLC_SCB_DEAUTHENTICATE_FOR_REASON: u32 = 201;
const WLC_GET_INSTANCE: u32 = 14;
const WLC_GET_BCNPRD: u32 = 75;
const WLC_GET_AP: u32 = 117;
/// An action frame the host asked to be sent went out (public
/// `bcmevent.h`), and one received, with `wl_event_rx_frame_data_t`
/// before the frame.
const WLC_E_ACTION_FRAME_COMPLETE: u32 = 60;
const WLC_E_ACTION_FRAME_RX: u32 = 75;
const WLC_SET_WPA_AUTH: u32 = 165;
const WLC_GET_ASSOCLIST: u32 = 159;
/// Interference mitigation mode (`WLC_SET_INTERFERENCE_OVERRIDE_MODE`,
/// public `wlioctl_defs.h`): `wlan` `0x65130` sets it for local
/// communication next to Bluetooth coexistence; the radio model has no
/// interference to mitigate, so the firmware only keeps it.
const WLC_SET_INTERFERENCE_OVERRIDE_MODE: u32 = 311;
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
/// The EDCA parameters the firmware starts with, per access category in
/// Broadcom's order (BE, BK, VI, VO; `edcf_acparam_t`: ACI and AIFSN,
/// ECWmax << 4 | ECWmin, TXOP in 32 us units): the defaults IEEE
/// 802.11-2016 Table 9-138 gives an access point (for the BSS it runs,
/// `wme_ac_ap`) and a station (`wme_ac_sta`). Assumed to be the
/// firmware's: `wlan` `0x65130` reads them, changes BE's ECW and writes
/// that entry back.
const WME_AC_AP: [[u8; 4]; 4] = [
    [0x03, 0x64, 0, 0],
    [0x27, 0xa4, 0, 0],
    [0x41, 0x43, 94, 0],
    [0x61, 0x32, 47, 0],
];
const WME_AC_STA: [[u8; 4]; 4] = [
    [0x03, 0xa4, 0, 0],
    [0x27, 0xa4, 0, 0],
    [0x42, 0x43, 94, 0],
    [0x62, 0x32, 47, 0],
];
/// Broadcom error codes (`bcmutils.h`, public).
const BCME_BADARG: i16 = -2;
/// A message carrying a firmware event to the host, in a posted event
/// buffer (`wlan` `0x1057e0`; brcmfmac `MSGBUF_TYPE_WL_EVENT`).
const MSG_WL_EVENT: u8 = 0x0e;
/// Event type and status of an escan's results (public `wlioctl_defs.h`:
/// `WLC_E_ESCAN_RESULT` 69, `WLC_E_STATUS_SUCCESS` 0).
const WLC_E_ESCAN_RESULT: u32 = 69;
const WLC_E_STATUS_SUCCESS: u32 = 0;
/// The scan was aborted (public `wlioctl_defs.h`, 4). Both public host
/// drivers end a scan on an escan result with this status (brcmfmac
/// `brcmf_cfg80211_escan_handler`, bcmdhd `wl_escan_handler`), which is
/// how they learn that the abort they asked for took effect.
const WLC_E_STATUS_ABORT: u32 = 4;
/// Interface event (`WLC_E_IF`, 54) with `wl_event_data_if_t` {ifidx,
/// opcode `WLC_E_IF_ADD` (1), reserved, bssidx, role STA (0)}, public
/// `wlioctl_defs.h`/`dhd.h`. Broadcom firmware announces the primary
/// interface this way (public brcmfmac `fweh.c` handles, and ignores, the
/// firmware's ADD for interface 0). `wlan` waits for it after "up": its
/// event handler (`0x5f5a0`, case 54, opcode 1) raises the supplicant's
/// "CTRL-EVENT-INTERFACE-UP" (`0x44eb0`), which its state machine turns
/// into event 5 (`0xe62e0`) completing the pending interface-up request
/// (`0xdb6e0` for request 10, `0xde260`); scans wait behind that request
/// (`0xcb7f0`). Sent when the interface comes up, if the host's event
/// mask enables it (assumed timing: right after the "up" completes).
const WLC_E_IF: u32 = 54;
const WLC_E_IF_ADD: u8 = 1;
/// `wl_escan_params_t` action starting a scan, and the one aborting it
/// (public `wlioctl.h`).
const WL_SCAN_ACTION_START: u16 = 1;
const WL_SCAN_ACTION_ABORT: u16 = 3;
/// `wl_event_msg_t` version (`BCM_EVENT_MSG_VERSION`) and the BSS info
/// version escan results carry (`WL_BSS_INFO_VERSION`), public headers.
const EVENT_MSG_VERSION: u16 = 2;
const BSS_INFO_VERSION: u32 = 109;
/// Events of a station's association (public `wlioctl_defs.h`). After a
/// join the firmware reports, in this order, authentication, association,
/// "join" (`WLC_E_JOIN`, on which `wlan`'s event handler `0x5f5a0`, cases
/// 1/8/10, raises the supplicant's association event), link up
/// (`WLC_E_LINK` with `WLC_EVENT_MSG_LINK`) and the join's completion
/// (`WLC_E_SET_SSID` with its status), as Broadcom firmware does (public
/// brcmfmac `cfg80211.c` and DHD event logs show the same sequence).
const WLC_E_SET_SSID: u32 = 0;
const WLC_E_JOIN: u32 = 1;
const WLC_E_AUTH: u32 = 3;
const WLC_E_ASSOC: u32 = 7;
const WLC_E_DISASSOC: u32 = 11;
const WLC_E_LINK: u32 = 16;
const WLC_E_STATUS_FAIL: u32 = 1;
const WLC_E_STATUS_PARTIAL: u32 = 8;
const WLC_EVENT_MSG_LINK: u16 = 1;
/// How long authentication and association take on the air (four
/// management frames and their acknowledgements, a few milliseconds on a
/// quiet channel; assumed).
const ASSOCIATION_TIME: Duration = Duration::from_millis(20);
/// When the access point sends message 1 of the 4-way handshake after
/// association, how long it waits for message 2 before sending it again
/// and how often: hostapd's defaults (`wpa_auth.c`: 100 ms first timeout
/// is used for later messages; 1 s and 4 attempts for message 1; assumed
/// for an ordinary home router).
const EAPOL_START_DELAY: Duration = Duration::from_millis(10);
const EAPOL_RETRY: Duration = Duration::from_secs(1);
const EAPOL_ATTEMPTS: u32 = 4;
/// Control requests of association (public `wlioctl_defs.h`).
const WLC_GET_BSSID: u32 = 23;
const WLC_GET_SSID: u32 = 25;
const WLC_SET_SSID: u32 = 26;
const WLC_DISASSOC: u32 = 52;
const WLC_GET_BSS_INFO: u32 = 136;
/// Authorizing (or deauthorizing) the 802.1X port of a peer, named by its
/// address: `wlan` authorizes the access point after joining an open
/// network (traced: input is the BSSID). The firmware's station port is
/// open on an open network anyway, so nothing changes here.
const WLC_SCB_AUTHORIZE: u32 = 121;
const WLC_SCB_DEAUTHORIZE: u32 = 122;
/// The signal strength of the access point, in dBm (`scb_val_t.val`);
/// nifm polls it for the signal bars while connected (traced, once a
/// second, through `wlan#0` request 106).
const WLC_GET_RSSI: u32 = 127;
const BCME_NOTUP: i16 = -4;
const BCME_NOTASSOCIATED: i16 = -17;
/// Data messages (public brcmfmac `msgbuf.c`, DHD `bcmmsgbuf.h`): a frame
/// to transmit on a flow ring, its completion on the TX completion ring,
/// and a received frame in a posted receive buffer on the RX completion
/// ring.
const MSG_TX_POST: u8 = 0x0f;
const MSG_TX_STATUS: u8 = 0x10;
const MSG_RX_CMPLT: u8 = 0x12;
const RING_D2H_TX_COMPLETE: usize = 3;
const RING_D2H_RX_COMPLETE: usize = 4;
/// Per-channel dwell when the host leaves it to the firmware (-1): the
/// firmware's defaults for an unassociated active scan (40 ms) and a
/// passive one (110 ms), Broadcom's `scan_unassoc_time` and
/// `scan_passive_time` defaults (public `wl` documentation; assumed for
/// this firmware).
const DEFAULT_ACTIVE_DWELL_MS: u64 = 40;
const DEFAULT_PASSIVE_DWELL_MS: u64 = 110;

/// Work the firmware finishes later, on the air: tagged with what it
/// belongs to so that work overtaken by events (an aborted scan, a new
/// join) does nothing. Kept in the function's state with the guest time
/// it is due at, so a snapshot carries it.
#[derive(Serialize, Deserialize)]
enum Deferred {
    /// The scan reached the access point's channel and heard it.
    ScanFound {
        sync_id: u16,
        if_id: u8,
    },
    ScanDone {
        sync_id: u16,
        if_id: u8,
    },
    /// The host aborted the scan in flight.
    ScanAborted {
        sync_id: u16,
        if_id: u8,
    },
    Associated {
        epoch: u32,
    },
    AssociationFailed {
        epoch: u32,
        if_id: u8,
        status: u32,
    },
    Disassociated {
        if_id: u8,
    },
    /// The access point sends (or resends) handshake message 1.
    EapolStart {
        epoch: u32,
        attempt: u32,
    },
    /// Work of the radio on the air shared with other emulators.
    Air(on_air::Work),
}

/// The station's association with the access point.
#[derive(Serialize, Deserialize)]
struct Association {
    if_id: u8,
    epoch: u32,
    /// Whether the link is up (the association completed).
    linked: bool,
    /// The access point's authenticator (WPA2-PSK only).
    authenticator: Option<wifi_ap::Authenticator>,
}

/// Takes the Ethernet frames the access point forwards from the station.
pub type Uplink = Box<dyn Fn(&[u8]) + Send + Sync>;

/// Takes the 802.11 frames the radio transmits on the air shared with
/// other emulators, with the channel it transmits on (`crate::air`).
pub type AirLink = Box<dyn Fn(u8, &[u8]) + Send + Sync>;

/// The network around the radio: the access point and where the frames
/// it forwards from the station go (the router behind it).
struct Network {
    ap: AccessPoint,
    uplink: Uplink,
}

/// A firmware event to the host.
struct Event<'a> {
    kind: u32,
    status: u32,
    reason: u32,
    flags: u16,
    addr: [u8; 6],
    data: &'a [u8],
}

impl<'a> Event<'a> {
    fn new(kind: u32, status: u32, data: &'a [u8]) -> Self {
        Self {
            kind,
            status,
            reason: 0,
            flags: 0,
            addr: [0; 6],
            data,
        }
    }

    fn from(mut self, addr: [u8; 6]) -> Self {
        self.addr = addr;
        self
    }
}

/// The function.
///
/// Its state, saved whole by its [`Snapshot`]: configuration space, core
/// wrappers, the PCIe core's mailbox and interrupt registers, the chip's
/// RAM, the started firmware's rings, posted buffers, settings, scan and
/// association (with the access point's authenticator), the work it has
/// scheduled on the air, and the access point's own state. The access
/// point's uplink to the router is wiring and stays as built; frames from
/// the network are dropped while guest time stands still.
pub struct Bcm4356Wifi {
    /// Shared with the thread that runs scheduled work.
    state: Arc<Mutex<State>>,
    /// Signalled when work is scheduled or the function goes away.
    due: Arc<Condvar>,
    upstream: Arc<dyn Upstream>,
    notes: Arc<Notes>,
    clock: Arc<GuestClock>,
}

#[derive(Clone, Copy, Default, Serialize, Deserialize)]
struct Wrapper {
    ioctrl: u32,
    resetctrl: u32,
}

#[derive(Serialize, Deserialize)]
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

#[derive(Clone, Copy, Serialize, Deserialize)]
struct Ring {
    base: u64,
    max_items: u16,
    item_len: u16,
}

#[derive(Clone, Copy, Serialize, Deserialize)]
struct HostBuffer {
    request_id: u32,
    address: u64,
    len: u16,
}

/// The started firmware.
#[derive(Serialize, Deserialize)]
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
    /// Work to finish later, after these delays; moved to
    /// [`State::scheduled`] once the request that made it is done.
    timers: Vec<(Duration, Deferred)>,
    /// The sync ID of the scan in flight (cleared by an abort).
    scan_in_flight: Option<u16>,
    event_sequence: u16,
    association: Option<Association>,
    /// Counts joins, to tell stale association work from current.
    epoch: u32,
    /// What the firmware does on the air shared with other emulators.
    radio: Radio,
    /// What the radio did in the request being served, carried out once
    /// the request is complete.
    #[serde(skip)]
    effects: Vec<Effect>,
}

#[derive(Serialize, Deserialize)]
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
    /// The level of the interrupt the function last signalled.
    interrupt: bool,
    /// Work due on the air, by the guest time it is due at; in that order,
    /// and in the order it was scheduled among work due at the same time.
    scheduled: Vec<(Duration, Deferred)>,
    /// The access point and the router behind it: wiring, set up by the
    /// machine's builder; the access point's state is saved on its own.
    #[serde(skip)]
    network: Option<Network>,
    /// The air shared with other emulators: wiring, set by the machine's
    /// builder (and again after a restore).
    #[serde(skip)]
    air: Option<AirLink>,
    /// The function is gone: the thread running scheduled work ends.
    #[serde(skip)]
    stopped: bool,
}

/// What the function's snapshot holds: its state and the access point's.
#[derive(Serialize)]
struct Saving<'a> {
    state: &'a State,
    access_point: Option<&'a AccessPoint>,
}

#[derive(Deserialize)]
struct Saved {
    state: State,
    access_point: Option<AccessPoint>,
}

/// Why an access was not served.
type Refusal = String;

impl Bcm4356Wifi {
    /// The function, as at power-on, with the thread that runs the work it
    /// schedules when guest time (`clock`) reaches it.
    pub fn new(upstream: Box<dyn Upstream>, notes: Arc<Notes>, clock: Arc<GuestClock>) -> Self {
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
        let state = Arc::new(Mutex::new(State {
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
            scheduled: Vec::new(),
            network: None,
            air: None,
            stopped: false,
        }));
        let due = Arc::new(Condvar::new());
        let upstream: Arc<dyn Upstream> = Arc::from(upstream);
        {
            let (state, due, upstream, notes, clock) = (
                Arc::clone(&state),
                Arc::clone(&due),
                Arc::clone(&upstream),
                Arc::clone(&notes),
                Arc::clone(&clock),
            );
            std::thread::Builder::new()
                .name("bcm4356 air".into())
                .spawn(move || run_scheduled(&state, &due, upstream.as_ref(), &notes, &clock))
                .expect("BCM4356 WiFi: spawn the thread of scheduled work");
        }
        Self {
            state,
            due,
            upstream,
            notes,
            clock,
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

    /// Places the access point configured on the command line around the
    /// radio; `uplink` takes the Ethernet frames the station sends through
    /// it.
    pub fn set_access_point(&self, ap: AccessPoint, uplink: Uplink) {
        self.state.lock().unwrap().network = Some(Network { ap, uplink });
    }

    /// An Ethernet frame from the network behind the access point for the
    /// station: delivered if the station is associated (and, on a WPA2
    /// network, authorized), else lost on the air. While guest time stands
    /// still (the machine is paused, as for a snapshot) it is lost too: the
    /// function's state and guest memory must not change then, and the
    /// sender sends again, as after any loss on the air.
    pub fn from_network(&self, frame: &[u8]) {
        let mut state = self.state.lock().unwrap();
        if self.clock.is_paused() {
            return;
        }
        let result = state.network_sent(frame, self.upstream.as_ref());
        if state.arm(self.clock.now()) {
            self.due.notify_one();
        }
        drop(state);
        if let Err(note) = result {
            self.notes.add(format!("BCM4356 WiFi: {note}"));
        }
    }

    /// Puts the radio on the air shared with other emulators: `link` takes
    /// the frames it transmits.
    pub fn set_air(&self, link: AirLink) {
        self.state.lock().unwrap().air = Some(link);
    }

    /// A frame another emulator's radio transmitted on `channel`: heard if
    /// the radio listens there. While guest time stands still it is lost,
    /// as frames from the network are ([`Self::from_network`]).
    pub fn from_air(&self, channel: u8, frame: &[u8]) {
        let mut state = self.state.lock().unwrap();
        if self.clock.is_paused() {
            return;
        }
        let now = self.clock.now();
        let result = state.air_received(channel, frame, now, self.upstream.as_ref());
        if state.arm(now) {
            self.due.notify_one();
        }
        drop(state);
        if let Err(note) = result {
            self.notes.add(format!("BCM4356 WiFi: {note}"));
        }
    }
}

impl Drop for Bcm4356Wifi {
    fn drop(&mut self) {
        self.state.lock().unwrap().stopped = true;
        self.due.notify_all();
    }
}

/// How long the thread of scheduled work waits at most before it looks
/// again while guest time stands still.
const PAUSED_POLL: Duration = Duration::from_millis(20);

/// The thread of scheduled work: runs each piece when guest time reaches
/// the moment it is due at, in order, and schedules what it leaves from
/// that moment. Nothing runs while guest time stands still: the check and
/// the work happen under the state's lock, which
/// [`Snapshot::quiesce`] takes after the clock stopped.
fn run_scheduled(
    state: &Mutex<State>,
    due: &Condvar,
    upstream: &dyn Upstream,
    notes: &Notes,
    clock: &GuestClock,
) {
    let mut locked = state.lock().unwrap();
    while !locked.stopped {
        let wait = match locked.scheduled.first() {
            None => None,
            Some(_) if clock.is_paused() => Some(PAUSED_POLL),
            Some(&(deadline, _)) => Some(clock.until(deadline)),
        };
        locked = match wait {
            None => due.wait(locked).unwrap(),
            Some(wait) if !wait.is_zero() => due.wait_timeout(locked, wait).unwrap().0,
            Some(_) => {
                let (deadline, work) = locked.scheduled.remove(0);
                let result = locked.deferred(work, deadline, upstream);
                locked.arm(deadline);
                if let Err(note) = result {
                    notes.add(format!("BCM4356 WiFi: {note}"));
                }
                locked
            }
        };
    }
}

impl Snapshot for Bcm4356Wifi {
    /// Waits for a frame from the network or scheduled work under way when
    /// guest time stopped; neither starts again until it runs on.
    fn quiesce(&self) -> Result<(), SnapshotError> {
        drop(self.state.lock().unwrap());
        Ok(())
    }

    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        let state = self.state.lock().unwrap();
        encode(&Saving {
            state: &state,
            access_point: state.network.as_ref().map(|network| &network.ap),
        })
    }

    /// Takes the saved state back, with the work it had scheduled, which
    /// runs when guest time reaches it. The access point takes back its
    /// group key; its uplink and the air link stay
    /// as built (the link to the other emulators is made anew, as after a
    /// dropped link; what was on the air meanwhile is lost).
    fn restore(&self, bytes: &[u8]) -> Result<(), SnapshotError> {
        let Saved {
            state: mut restored,
            access_point,
        } = decode("BCM4356 WiFi", bytes)?;
        if restored.ram.len() != RAM_SIZE as usize {
            return Err(SnapshotError::Corrupt(format!(
                "BCM4356 WiFi: {} bytes of RAM where the chip has {RAM_SIZE}",
                restored.ram.len()
            )));
        }
        let mut state = self.state.lock().unwrap();
        match (state.network.as_mut(), access_point) {
            (Some(network), Some(saved)) => network
                .ap
                .restore(saved)
                .map_err(|error| SnapshotError::Corrupt(format!("BCM4356 WiFi: {error}")))?,
            (None, None) => {}
            _ => {
                return Err(SnapshotError::Corrupt(
                    "BCM4356 WiFi: an access point in one machine and not in the other".into(),
                ));
            }
        }
        restored.network = state.network.take();
        restored.air = state.air.take();
        restored.stopped = state.stopped;
        *state = restored;
        self.due.notify_all();
        Ok(())
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
    /// A halted firmware is gone, and so is the work it had scheduled on
    /// the air: the next firmware starts with none. `wlan` halts it to
    /// download the firmware again each time the console wakes (traced:
    /// before sleeping it disassociates, takes the radio down and puts the
    /// function in D3hot; `pcie` holds the GPIO pad 0x38000002 low through
    /// the sleep and raises it again on waking, then `wlan` enumerates the
    /// function and downloads anew).
    fn arm_changed(&mut self) {
        let wrapper = self.wrappers[WRAPPER_ARMCR4];
        let in_reset = wrapper.resetctrl & AIRC_RESET != 0;
        let halted = in_reset || wrapper.ioctrl & SICF_CPUHALT != 0;
        if halted {
            if !self.halted {
                self.halted = true;
                self.dongle = None;
                self.image_end = 0;
                self.scheduled.clear();
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
            timers: Vec::new(),
            association: None,
            epoch: 0,
            scan_in_flight: None,
            event_sequence: 0,
            radio: Radio::default(),
            effects: Vec::new(),
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
        // Frames the host transmits, on the flow rings it created.
        let flow_rings: Vec<(u16, Ring)> = dongle.flow_rings.clone();
        for (id, ring) in flow_rings {
            while let Some(item) = self.next_item(dongle, id, ring, upstream)? {
                self.transmit(dongle, id, &item, upstream)?;
            }
        }
        Ok(())
    }

    /// The next item the host submitted on H2D common ring `ring`, if any.
    fn next_submission(
        &mut self,
        dongle: &Dongle,
        ring: u16,
        upstream: &dyn Upstream,
    ) -> Result<Option<Vec<u8>>, Refusal> {
        let desc = dongle.rings.as_ref().unwrap()[usize::from(ring)];
        self.next_item(dongle, ring, desc, upstream)
    }
    /// The next item the host submitted on H2D ring `ring` (common or
    /// flow ring, by its index slot), if any.
    fn next_item(
        &mut self,
        dongle: &Dongle,
        ring: u16,
        desc: Ring,
        upstream: &dyn Upstream,
    ) -> Result<Option<Vec<u8>>, Refusal> {
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
                // A new ring is read from its first item. The host starts
                // writing there too; a ring ID can be deleted and created
                // again, and what was read of the old ring must not carry
                // over.
                self.set_ram_u16(dongle.h2d_read + u32::from(id) * INDEX_SIZE, 0);
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
        let ap = self.network.as_ref().map(|network| &network.ap);
        let outcome = dongle_ioctl(dongle, ap, if_id, command, &input)?;
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
        self.complete(dongle, RING_D2H_CONTROL, &done, upstream)?;
        if command == WLC_UP && status == 0 && event_enabled(dongle, WLC_E_IF) {
            self.event(
                dongle,
                0,
                Event::new(WLC_E_IF, WLC_E_STATUS_SUCCESS, &[0, WLC_E_IF_ADD, 0, 0, 0]),
                upstream,
            )?;
        }
        self.apply_effects(dongle, upstream)
    }

    /// Schedules the work the firmware left, after its delays from guest
    /// time `base`: now for a host request, the moment it was due at for
    /// scheduled work. Whether there was any.
    fn arm(&mut self, base: Duration) -> bool {
        let Some(dongle) = self.dongle.as_mut() else {
            return false;
        };
        let timers = std::mem::take(&mut dongle.timers);
        let any = !timers.is_empty();
        for (delay, work) in timers {
            let at = base + delay;
            let index = self.scheduled.partition_point(|&(due, _)| due <= at);
            self.scheduled.insert(index, (at, work));
        }
        any
    }

    /// Finishes work the firmware began earlier, due at guest time `at`.
    fn deferred(
        &mut self,
        work: Deferred,
        at: Duration,
        upstream: &dyn Upstream,
    ) -> Result<(), Refusal> {
        let Some(mut dongle) = self.dongle.take() else {
            return Ok(());
        };
        let result = self.run(&mut dongle, work, at, upstream);
        self.dongle = Some(dongle);
        result.and_then(|()| self.update_interrupt(upstream))
    }

    fn run(
        &mut self,
        dongle: &mut Dongle,
        work: Deferred,
        at: Duration,
        upstream: &dyn Upstream,
    ) -> Result<(), Refusal> {
        match work {
            // A probe response from the access point: a partial escan
            // result with its BSS.
            Deferred::ScanFound { sync_id, if_id } => {
                let Some(bss) = self.network.as_ref().map(|n| n.ap.bss_info()) else {
                    return Ok(());
                };
                if dongle.scan_in_flight == Some(sync_id) {
                    let data = escan_result(sync_id, &[bss]);
                    self.event(
                        dongle,
                        if_id,
                        Event::new(WLC_E_ESCAN_RESULT, WLC_E_STATUS_PARTIAL, &data),
                        upstream,
                    )?;
                }
            }
            // The scan's time is up: the escan finishes with no further
            // BSS (status success, `bss_count` 0).
            Deferred::ScanDone { sync_id, if_id } => {
                if dongle.scan_in_flight == Some(sync_id) {
                    dongle.scan_in_flight = None;
                    dongle.radio.end_scan(sync_id);
                    let data = escan_result(sync_id, &[]);
                    self.event(
                        dongle,
                        if_id,
                        Event::new(WLC_E_ESCAN_RESULT, WLC_E_STATUS_SUCCESS, &data),
                        upstream,
                    )?;
                }
            }
            // `nifm` stops its scan when the console goes to sleep (`wlan`
            // request 61) and waits for the scan to end before it lets the
            // sleep go on.
            Deferred::ScanAborted { sync_id, if_id } => {
                let data = escan_result(sync_id, &[]);
                self.event(
                    dongle,
                    if_id,
                    Event::new(WLC_E_ESCAN_RESULT, WLC_E_STATUS_ABORT, &data),
                    upstream,
                )?;
            }
            Deferred::Associated { epoch } => {
                let Some(association) = dongle.association.as_mut().filter(|a| a.epoch == epoch)
                else {
                    return Ok(());
                };
                association.linked = true;
                let if_id = association.if_id;
                let secured = association.authenticator.is_some();
                let ssid = self
                    .network
                    .as_ref()
                    .map(|n| n.ap.ssid().to_vec())
                    .unwrap_or_default();
                for kind in [WLC_E_AUTH, WLC_E_ASSOC, WLC_E_JOIN] {
                    self.event(
                        dongle,
                        if_id,
                        Event::new(kind, WLC_E_STATUS_SUCCESS, &[]).from(wifi_ap::BSSID),
                        upstream,
                    )?;
                }
                let mut link =
                    Event::new(WLC_E_LINK, WLC_E_STATUS_SUCCESS, &[]).from(wifi_ap::BSSID);
                link.flags = WLC_EVENT_MSG_LINK;
                self.event(dongle, if_id, link, upstream)?;
                self.event(
                    dongle,
                    if_id,
                    Event::new(WLC_E_SET_SSID, WLC_E_STATUS_SUCCESS, &ssid).from(wifi_ap::BSSID),
                    upstream,
                )?;
                if secured {
                    dongle.timers.push((
                        EAPOL_START_DELAY,
                        Deferred::EapolStart { epoch, attempt: 0 },
                    ));
                }
            }
            Deferred::AssociationFailed {
                epoch,
                if_id,
                status,
            } => {
                if dongle.epoch == epoch {
                    dongle.association = None;
                    self.event(
                        dongle,
                        if_id,
                        Event::new(WLC_E_SET_SSID, status, &[]),
                        upstream,
                    )?;
                }
            }
            Deferred::Disassociated { if_id } => {
                self.event(
                    dongle,
                    if_id,
                    Event::new(WLC_E_DISASSOC, WLC_E_STATUS_SUCCESS, &[]).from(wifi_ap::BSSID),
                    upstream,
                )?;
                self.event(
                    dongle,
                    if_id,
                    Event::new(WLC_E_LINK, WLC_E_STATUS_SUCCESS, &[]).from(wifi_ap::BSSID),
                    upstream,
                )?;
            }
            Deferred::EapolStart { epoch, attempt } => {
                let Some(association) = dongle.association.as_mut().filter(|a| a.epoch == epoch)
                else {
                    return Ok(());
                };
                let if_id = association.if_id;
                if let Some(m1) = association
                    .authenticator
                    .as_mut()
                    .and_then(|a| a.message_1())
                {
                    if attempt + 1 < EAPOL_ATTEMPTS {
                        dongle.timers.push((
                            EAPOL_RETRY,
                            Deferred::EapolStart {
                                epoch,
                                attempt: attempt + 1,
                            },
                        ));
                    }
                    self.deliver(dongle, if_id, &m1, upstream)?;
                }
            }
            Deferred::Air(work) => {
                let config = air_config(dongle);
                let effects = match work {
                    on_air::Work::Beacon { epoch } => dongle.radio.beacon(&config, epoch, at),
                    on_air::Work::ScanChannel { sync_id, channel } => {
                        dongle.radio.scan_channel(&config, sync_id, channel)
                    }
                    on_air::Work::Probe { epoch, index } => {
                        dongle.radio.probe(&config, epoch, index)
                    }
                    on_air::Work::Timeout { epoch, stage } => dongle.radio.timeout(epoch, stage),
                    on_air::Work::PeriodicActionFrame { setting } => {
                        periodic_action_frame(dongle, setting)
                    }
                };
                dongle.effects.extend(effects);
                self.apply_effects(dongle, upstream)?;
            }
        }
        Ok(())
    }

    /// Carries out what the radio did: frames onto the air (lost if the
    /// radio shares no air), events and frames for the host, and work for
    /// later.
    fn apply_effects(
        &mut self,
        dongle: &mut Dongle,
        upstream: &dyn Upstream,
    ) -> Result<(), Refusal> {
        for effect in std::mem::take(&mut dongle.effects) {
            match effect {
                Effect::Transmit { channel, frame } => {
                    if let Some(air) = self.air.as_ref() {
                        air(channel, &frame);
                    }
                }
                Effect::Event {
                    if_id,
                    kind,
                    status,
                    reason,
                    flags,
                    addr,
                    data,
                } => {
                    let mut e = Event::new(kind, status, &data).from(addr);
                    e.reason = reason;
                    e.flags = flags;
                    self.event(dongle, if_id, e, upstream)?;
                }
                Effect::ScanResult {
                    if_id,
                    sync_id,
                    bss_info,
                } => {
                    if dongle.scan_in_flight == Some(sync_id) {
                        let data = escan_result(sync_id, &[bss_info]);
                        self.event(
                            dongle,
                            if_id,
                            Event::new(WLC_E_ESCAN_RESULT, WLC_E_STATUS_PARTIAL, &data),
                            upstream,
                        )?;
                    }
                }
                Effect::Deliver { if_id, frame } => {
                    self.deliver(dongle, if_id, &frame, upstream)?
                }
                Effect::ActionFrame { channel, frame } => {
                    self.action_frame_heard(dongle, channel, &frame, upstream)?
                }
                Effect::Later { delay, work } => dongle.timers.push((delay, Deferred::Air(work))),
            }
        }
        Ok(())
    }

    /// An action frame heard on `channel`: a Nintendo vendor-specific one
    /// (category 127, the OUI the host set with `nsc_subtypeoui`, its type
    /// byte one of those listed in `nsc_subtypes`) goes to the host as
    /// `WLC_E_ACTION_FRAME_RX`: `wl_event_rx_frame_data_t` (version 1,
    /// chanspec, RSSI, MAC time, rate; big-endian like the event header,
    /// public `bcmevent.h`), then the frame's body from its category on;
    /// the sender is the event's address. `wlan`'s handler of event 75
    /// (`0x5f5a0`) rebuilds the MAC header from that address, its own and
    /// the BSSID (`0x687b0` with the data from offset 16), as brcmfmac
    /// does (`brcmf_p2p_notify_action_frame_rx`).
    /// \[inference\] The filter's meaning is taken from what `wlan` sets:
    /// the OUI 00:22:AA (data `0x130d2d`, `0x69b20`), and `nsc_subtypes` as
    /// eight type bytes with 0xFF for none (all 0xFF when a mode is set up,
    /// `0x69d80`; then 04 and 05 added one by one, traced), and from the
    /// host's event mask, which enables event 75 (and not 59). Other
    /// action frames are not passed on.
    fn action_frame_heard(
        &mut self,
        dongle: &mut Dongle,
        channel: u8,
        frame: &[u8],
        upstream: &dyn Upstream,
    ) -> Result<(), Refusal> {
        let body = frame.get(super::dot11::HEADER_LEN..).unwrap_or(&[]);
        let oui = dongle.settings.get("nsc_subtypeoui");
        let types = dongle
            .settings
            .get("nsc_subtypes")
            .cloned()
            .unwrap_or_default();
        let wanted = body.first() == Some(&127)
            && oui.is_some_and(|oui| body.get(1..4) == oui.get(..3))
            && body
                .get(4)
                .is_some_and(|t| *t != 0xff && types.iter().take(8).any(|x| x == t));
        if !wanted {
            return Ok(());
        }
        let mut data = 1u16.to_be_bytes().to_vec();
        data.extend(chanspec_20(channel).to_be_bytes());
        data.extend(i32::from(on_air::RSSI_DBM).to_be_bytes());
        data.extend(0u32.to_be_bytes());
        data.extend(0u32.to_be_bytes());
        data.extend(body);
        let addr: [u8; 6] = frame[10..16].try_into().unwrap();
        self.event(
            dongle,
            0,
            Event::new(WLC_E_ACTION_FRAME_RX, WLC_E_STATUS_SUCCESS, &data).from(addr),
            upstream,
        )
    }

    /// A frame from the air on `channel`, at guest time `now`.
    fn air_received(
        &mut self,
        channel: u8,
        frame: &[u8],
        now: Duration,
        upstream: &dyn Upstream,
    ) -> Result<(), Refusal> {
        let Some(mut dongle) = self.dongle.take() else {
            return Ok(());
        };
        let result = if dongle.up {
            let config = air_config(&dongle);
            let effects = dongle.radio.receive(&config, channel, frame, now);
            dongle.effects.extend(effects);
            self.apply_effects(&mut dongle, upstream)
        } else {
            Ok(())
        };
        self.dongle = Some(dongle);
        result.and_then(|()| self.update_interrupt(upstream))
    }

    /// A frame the host transmits: `msgbuf_tx_msghdr` with the Ethernet
    /// header inline (offset 8) and the rest in a host buffer (address at
    /// 32, length at 42). It is completed at once on the TX completion
    /// ring and goes to the access point.
    fn transmit(
        &mut self,
        dongle: &mut Dongle,
        ring: u16,
        item: &[u8],
        upstream: &dyn Upstream,
    ) -> Result<(), Refusal> {
        if item[0] != MSG_TX_POST {
            return Err(format!(
                "flow ring {ring} message type {:#x} not modeled: {item:02x?}",
                item[0]
            ));
        }
        let if_id = item[1];
        let address = le64(item, 32);
        let mut frame = item[8..22].to_vec();
        let start = frame.len();
        frame.resize(start + usize::from(le16(item, 42)), 0);
        upstream
            .read(address, &mut frame[start..])
            .ok_or_else(|| format!("transmit buffer at {address:#x} not readable"))?;
        let mut status = header(MSG_TX_STATUS, if_id, le32(item, 4));
        status[10..12].copy_from_slice(&ring.to_le_bytes());
        self.complete(dongle, RING_D2H_TX_COMPLETE, &status, upstream)?;
        let config = air_config(dongle);
        if let Some(effects) = dongle.radio.transmit(&config, &frame) {
            dongle.effects.extend(effects);
            return self.apply_effects(dongle, upstream);
        }
        self.station_sent(dongle, &frame, upstream)
    }

    /// A frame the station sends: lost without an association; EAPOL goes
    /// to the authenticator; data goes through the access point once the
    /// port is open.
    fn station_sent(
        &mut self,
        dongle: &mut Dongle,
        frame: &[u8],
        upstream: &dyn Upstream,
    ) -> Result<(), Refusal> {
        let Some(association) = dongle.association.as_mut().filter(|a| a.linked) else {
            return Ok(());
        };
        let if_id = association.if_id;
        if frame.get(12..14) == Some(&wifi_ap::ETHERTYPE_EAPOL.to_be_bytes()) {
            let reply = association
                .authenticator
                .as_mut()
                .and_then(|a| a.receive(frame));
            if let Some(reply) = reply {
                self.deliver(dongle, if_id, &reply, upstream)?;
            }
            return Ok(());
        }
        let open = association
            .authenticator
            .as_ref()
            .is_none_or(|a| a.authorized());
        if let Some(network) = self.network.as_ref().filter(|_| open) {
            (network.uplink)(frame);
        }
        Ok(())
    }

    /// A frame from the network for the station.
    fn network_sent(&mut self, frame: &[u8], upstream: &dyn Upstream) -> Result<(), Refusal> {
        let Some(mut dongle) = self.dongle.take() else {
            return Ok(());
        };
        let station = dongle.mac;
        let result = match dongle.association.as_ref() {
            Some(a)
                if a.linked
                    && a.authenticator.as_ref().is_none_or(|a| a.authorized())
                    && (frame.get(..6) == Some(&station)
                        || frame.first().is_some_and(|b| b & 1 != 0)) =>
            {
                let if_id = a.if_id;
                self.deliver(&mut dongle, if_id, frame, upstream)
            }
            _ => Ok(()),
        };
        self.dongle = Some(dongle);
        result.and_then(|()| self.update_interrupt(upstream))
    }

    /// Hands a received frame to the host in the next posted receive
    /// buffer (`msgbuf_rx_complete`: data length at 14, data offset 0,
    /// 802.3 frame). With no buffer posted the firmware has nowhere to put
    /// it and the frame is lost, as on the air.
    fn deliver(
        &mut self,
        dongle: &mut Dongle,
        if_id: u8,
        frame: &[u8],
        upstream: &dyn Upstream,
    ) -> Result<(), Refusal> {
        let Some(buffer) = dongle.rx_buffers.pop_front() else {
            return Ok(());
        };
        if frame.len() > usize::from(buffer.len) {
            return Err(format!(
                "received frame of {} bytes for a {}-byte buffer",
                frame.len(),
                buffer.len
            ));
        }
        upstream
            .write(buffer.address, frame)
            .ok_or_else(|| format!("receive buffer at {:#x} not writable", buffer.address))?;
        let mut message = header(MSG_RX_CMPLT, if_id, buffer.request_id);
        message[14..16].copy_from_slice(&(frame.len() as u16).to_le_bytes());
        self.complete(dongle, RING_D2H_RX_COMPLETE, &message, upstream)
    }

    /// Sends firmware event `e` to the host, if its event mask enables it:
    /// an Ethernet frame to the station address with Broadcom's event
    /// ether type 0x886C, `bcmeth_hdr_t` (subtype 0x8001, OUI 00:10:18,
    /// user subtype 1) and a big-endian `wl_event_msg_t`, then the data
    /// (the layout `wlan` `0x100d40` checks), in the next posted event
    /// buffer.
    fn event(
        &mut self,
        dongle: &mut Dongle,
        if_id: u8,
        e: Event,
        upstream: &dyn Upstream,
    ) -> Result<(), Refusal> {
        if !event_enabled(dongle, e.kind) {
            return Ok(());
        }
        let data = e.data;
        let mut frame = Vec::with_capacity(72 + data.len());
        frame.extend(dongle.mac);
        frame.extend(dongle.mac);
        frame.extend(0x886cu16.to_be_bytes());
        frame.extend(0x8001u16.to_be_bytes());
        frame.extend(((8 + 48 + data.len()) as u16).to_be_bytes());
        frame.extend([0, 0x00, 0x10, 0x18]);
        frame.extend(1u16.to_be_bytes());
        frame.extend(EVENT_MSG_VERSION.to_be_bytes());
        frame.extend(e.flags.to_be_bytes());
        frame.extend(e.kind.to_be_bytes());
        frame.extend(e.status.to_be_bytes());
        frame.extend(e.reason.to_be_bytes());
        frame.extend(0u32.to_be_bytes()); // auth type: open system
        frame.extend((data.len() as u32).to_be_bytes());
        frame.extend(e.addr);
        let mut ifname = [0u8; 16];
        ifname[..3].copy_from_slice(b"wl0");
        frame.extend(ifname);
        frame.extend([if_id, 0]);
        frame.extend(data);
        let buffer = dongle
            .event_buffers
            .pop_front()
            .ok_or("event with no event buffer posted")?;
        if frame.len() > usize::from(buffer.len) {
            return Err(format!(
                "event of {} bytes for a {}-byte buffer",
                frame.len(),
                buffer.len
            ));
        }
        upstream
            .write(buffer.address, &frame)
            .ok_or_else(|| format!("event buffer at {:#x} not writable", buffer.address))?;
        let mut message = header(MSG_WL_EVENT, if_id, buffer.request_id);
        message[12..14].copy_from_slice(&(frame.len() as u16).to_le_bytes());
        message[14..16].copy_from_slice(&dongle.event_sequence.to_le_bytes());
        dongle.event_sequence = dongle.event_sequence.wrapping_add(1);
        self.complete(dongle, RING_D2H_CONTROL, &message, upstream)
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
    ap: Option<&AccessPoint>,
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
        // Down, the radio is off the air: its BSS and its link end.
        WLC_DOWN => {
            dongle.up = false;
            dongle.radio.down();
            return Ok(Ok(Vec::new()));
        }
        // Whether the interface is up (`wlan` `0x65130` asks before it
        // changes coexistence settings).
        WLC_GET_UP => return Ok(Ok(u32::from(dongle.up).to_le_bytes().to_vec())),
        // The driver instance: the chip's only radio, unit 0 (Broadcom's
        // `wlc->pub->unit` of the first interface).
        WLC_GET_INSTANCE => return Ok(Ok(0u32.to_le_bytes().to_vec())),
        // Access point mode, as the host set it (off at power-on).
        WLC_GET_AP => {
            let mode = setting_word(dongle, &format!("ioctl {WLC_SET_AP}")).unwrap_or(0);
            return Ok(Ok(mode.to_le_bytes().to_vec()));
        }
        WLC_GET_BCNPRD => {
            let period = setting_word(dongle, &format!("ioctl {WLC_SET_BCNPRD}")).unwrap_or(100);
            return Ok(Ok(period.to_le_bytes().to_vec()));
        }
        // The authenticated key management, as the "wpa_auth" iovar sets it.
        WLC_SET_WPA_AUTH => {
            let Some(word) = input.get(..4) else {
                return Ok(Err(BCME_BADARG));
            };
            dongle.settings.insert("wpa_auth".into(), word.to_vec());
            return Ok(Ok(Vec::new()));
        }
        // `maclist`: count, then the addresses of the stations associated
        // with the BSS the radio runs, or of the access point it joined.
        WLC_GET_ASSOCLIST => {
            let peers: Vec<[u8; 6]> = match (&dongle.radio.bss, air_link(dongle)) {
                (Some(bss), _) => bss
                    .stations
                    .iter()
                    .filter(|s| s.associated)
                    .map(|s| s.mac)
                    .collect(),
                (None, Some(link)) => link.bssid.into_iter().collect(),
                _ if dongle.association.as_ref().is_some_and(|a| a.linked) => {
                    vec![wifi_ap::BSSID]
                }
                _ => Vec::new(),
            };
            let mut list = (peers.len() as u32).to_le_bytes().to_vec();
            list.extend(peers.concat());
            return Ok(Ok(list));
        }
        // Settings taking one word, which the firmware stores: infrastructure
        // mode, channel, power save mode, access point mode, band, default
        // scan type, beacon and DTIM periods, interference mitigation.
        WLC_SET_INFRA
        | WLC_SET_CHANNEL
        | WLC_SET_PM
        | WLC_SET_AP
        | WLC_SET_BAND
        | WLC_SET_PASSIVE_SCAN
        | WLC_SET_BCNPRD
        | WLC_SET_DTIMPRD
        | WLC_SET_INTERFERENCE_OVERRIDE_MODE => {
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
        // `wl_join_params_t`: SSID, then `wl_assoc_params_t` (BSSID at 36,
        // chanspec count at 44, chanspecs from 48). In access point mode it
        // starts the BSS (an empty SSID stops it); otherwise it is a join
        // without the "join" iovar, `wlan`'s fallback at `0x646e0`.
        WLC_SET_SSID => {
            let (Some(len), Some(bssid)) = (input.get(..4), input.get(36..42)) else {
                return Ok(Err(BCME_BADARG));
            };
            let len = (u32::from_le_bytes(len.try_into().unwrap()) as usize).min(32);
            let ssid = input[4..4 + len].to_vec();
            if access_point_mode(dongle) {
                return Ok(start_bss(dongle, if_id, &ssid));
            }
            let channels = chanspec_list(input, 44);
            return Ok(join(
                dongle,
                ap,
                if_id,
                &ssid,
                bssid.try_into().unwrap(),
                channels,
            ));
        }
        WLC_GET_BSSID => {
            if dongle.radio.bss.is_some() {
                return Ok(Ok(dongle.mac.to_vec()));
            }
            if let Some(bssid) = air_link(dongle).and_then(|l| l.bssid) {
                return Ok(Ok(bssid.to_vec()));
            }
            return Ok(match dongle.association.as_ref().filter(|a| a.linked) {
                Some(_) => Ok(wifi_ap::BSSID.to_vec()),
                None => Err(BCME_NOTASSOCIATED),
            });
        }
        // `wlc_ssid_t` of the network joined or run: length, 32 bytes.
        WLC_GET_SSID => {
            let mut ssid = vec![0u8; 36];
            let own = match (&dongle.radio.bss, air_link(dongle)) {
                (Some(bss), _) => Some(bss.ssid.as_slice()),
                (None, Some(link)) => Some(link.ssid.as_slice()),
                _ => None,
            };
            if let Some(name) = own {
                ssid[..4].copy_from_slice(&(name.len() as u32).to_le_bytes());
                ssid[4..4 + name.len()].copy_from_slice(name);
            } else if let (Some(ap), Some(_)) =
                (ap, dongle.association.as_ref().filter(|a| a.linked))
            {
                ssid[..4].copy_from_slice(&(ap.ssid().len() as u32).to_le_bytes());
                ssid[4..4 + ap.ssid().len()].copy_from_slice(ap.ssid());
            }
            return Ok(Ok(ssid));
        }
        WLC_GET_RSSI => {
            if air_link(dongle).is_some() {
                return Ok(Ok(i32::from(on_air::RSSI_DBM).to_le_bytes().to_vec()));
            }
            return Ok(match dongle.association.as_ref().filter(|a| a.linked) {
                Some(_) => Ok(i32::from(wifi_ap::RSSI_DBM).to_le_bytes().to_vec()),
                None => Err(BCME_NOTASSOCIATED),
            });
        }
        // The 802.1X port of a peer: a station of the BSS the radio runs,
        // or the access point it joined. The firmware does not filter data
        // by it here (the host's supplicant or authenticator holds frames
        // back until the handshake is done).
        WLC_SCB_AUTHORIZE | WLC_SCB_DEAUTHORIZE => {
            let peer = input.get(..6);
            let station = dongle.radio.bss.as_ref().is_some_and(|bss| {
                bss.stations
                    .iter()
                    .any(|s| s.associated && Some(&s.mac[..]) == peer)
            });
            let access_point =
                air_link(dongle).is_some_and(|l| l.bssid.as_ref().map(|b| &b[..]) == peer);
            if station || access_point {
                return Ok(Ok(Vec::new()));
            }
            return Ok(match dongle.association.as_ref().filter(|a| a.linked) {
                Some(_) if peer == Some(&wifi_ap::BSSID) => Ok(Vec::new()),
                _ => Err(BCME_NOTASSOCIATED),
            });
        }
        // Sending a station of the BSS away: `scb_val_t` (reason, address).
        WLC_SCB_DEAUTHENTICATE_FOR_REASON => {
            let (Some(reason), Some(mac)) = (input.get(..4), input.get(4..10)) else {
                return Ok(Err(BCME_BADARG));
            };
            let reason = u32::from_le_bytes(reason.try_into().unwrap()) as u16;
            let config = air_config(dongle);
            let effects = dongle
                .radio
                .deauthenticate(&config, mac.try_into().unwrap(), reason);
            dongle.effects.extend(effects);
            return Ok(Ok(Vec::new()));
        }
        // Leaving the network: the firmware reports the disassociation and
        // the link going down.
        WLC_DISASSOC => {
            if let Some(link) = air_link(dongle) {
                let (if_id, bssid) = (link.if_id, link.bssid.unwrap_or_default());
                let config = air_config(dongle);
                let effects = dongle.radio.leave(&config);
                dongle.effects.extend(effects);
                for kind in [WLC_E_DISASSOC, WLC_E_LINK] {
                    dongle.effects.push(Effect::Event {
                        if_id,
                        kind,
                        status: WLC_E_STATUS_SUCCESS,
                        reason: 0,
                        flags: 0,
                        addr: bssid,
                        data: Vec::new(),
                    });
                }
                return Ok(Ok(Vec::new()));
            }
            dongle.radio.link = None;
            if let Some(association) = dongle.association.take().filter(|a| a.linked) {
                dongle.timers.push((
                    Duration::ZERO,
                    Deferred::Disassociated {
                        if_id: association.if_id,
                    },
                ));
            }
            return Ok(Ok(Vec::new()));
        }
        // The current BSS: the buffer's length word, then `wl_bss_info_t`.
        WLC_GET_BSS_INFO => {
            if let Some(link) = air_link(dongle) {
                let mut info = input.get(..4).unwrap_or(&[0; 4]).to_vec();
                info.extend(&link.bss_info);
                return Ok(Ok(info));
            }
            return Ok(
                match (ap, dongle.association.as_ref().filter(|a| a.linked)) {
                    (Some(ap), Some(_)) => {
                        let mut info = input.get(..4).unwrap_or(&[0; 4]).to_vec();
                        info.extend(ap.bss_info());
                        Ok(info)
                    }
                    _ => Err(BCME_NOTASSOCIATED),
                },
            );
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
        // The radio's channel: its BSS's, the access point's while
        // associated, else what the host last set.
        (false, "chanspec") if dongle.radio.bss.is_some() || air_link(dongle).is_some() => {
            let channel = dongle.radio.channel().unwrap_or(0);
            Ok(u32::from(chanspec_20(channel)).to_le_bytes().to_vec())
        }
        (false, "chanspec") if dongle.association.as_ref().is_some_and(|a| a.linked) => {
            Ok(u32::from(ap.map_or(0, AccessPoint::chanspec))
                .to_le_bytes()
                .to_vec())
        }
        (true, "escan") => escan(dongle, ap, if_id, value),
        // `wl_extjoin_params_t`: SSID (length, 32 bytes), join scan
        // parameters, then `wl_join_assoc_params_t`: the BSSID at 56, the
        // chanspec count at 64 and chanspecs from 68 (`wlan` `0x646e0`).
        (true, "join") => {
            let (Some(len), Some(bssid)) = (value.get(..4), value.get(56..62)) else {
                return Ok(Err(BCME_BADARG));
            };
            let len = (u32::from_le_bytes(len.try_into().unwrap()) as usize).min(32);
            let ssid = value[4..4 + len].to_vec();
            let channels = chanspec_list(value, 64);
            join(
                dongle,
                ap,
                if_id,
                &ssid,
                bssid.try_into().unwrap(),
                channels,
            )
        }
        // Keys the radio encrypts and decrypts with (CCMP only).
        (true, "wsec_key") => {
            let change = on_air::key_change(value)?;
            dongle.radio.change_key(change);
            Ok(Vec::new())
        }
        // Vendor elements for the frames the radio sends.
        (true, "vndr_ie") => {
            dongle.radio.vendor_elements(value)?;
            Ok(Vec::new())
        }
        // Ends the periodic action frames (\[inference\] from the name and
        // where `wlan` sends it: before it asks for a new series, with no
        // value of its own, `0x10c8b0`'s buffer still holding the name).
        (true, "nsc_aloe_cancel") => {
            dongle.settings.remove("nsc_aloe_af_periodic");
            Ok(Vec::new())
        }
        // Nintendo-specific periodic action frames (`wlan` `0x10c8b0`,
        // called by `0xfa9a0` with the beacon period it read with
        // `WLC_GET_BCNPRD`): the period in TU at 0, then from 20 a
        // `wl_af_params_t` (BSSID at 28, destination at 36, the frame's
        // length at 42, packet ID at 44, the frame from its category on
        // from 48; traced: 1364-byte frames). Sent every period from now
        // on, while `nsc_afmode` is on, until the setting changes.
        // \[inference\] from the layout `wlan` builds; that the firmware
        // sends the first frame at once is assumed.
        (true, "nsc_aloe_af_periodic") => {
            if value.len() < 48 {
                return Ok(Err(BCME_BADARG));
            }
            dongle
                .settings
                .insert("nsc_aloe_af_periodic".into(), value.to_vec());
            dongle.timers.push((
                Duration::ZERO,
                Deferred::Air(on_air::Work::PeriodicActionFrame {
                    setting: value.to_vec(),
                }),
            ));
            Ok(Vec::new())
        }
        // One action frame: `wl_af_params_t` (channel at 0, dwell time at
        // 4, BSSID at 8, then `wl_action_frame_t`: destination at 16,
        // length at 22, packet ID at 24, the frame from 28; public
        // `wlioctl.h`, the layout `wlan` `0x10cb70` fills). It goes out on
        // the channel given (the radio's own if 0) and the host hears it
        // sent (`WLC_E_ACTION_FRAME_COMPLETE` with the packet ID).
        (true, "actframe") => {
            let (Some(channel), Some(bssid), Some(da), Some(length), Some(id)) = (
                value.get(..4),
                value.get(8..14),
                value.get(16..22),
                value.get(22..24),
                value.get(24..28),
            ) else {
                return Ok(Err(BCME_BADARG));
            };
            let length = usize::from(u16::from_le_bytes([length[0], length[1]]));
            let Some(body) = value.get(28..28 + length) else {
                return Ok(Err(BCME_BADARG));
            };
            let channel = match u32::from_le_bytes(channel.try_into().unwrap()) as u8 {
                0 => dongle
                    .radio
                    .channel()
                    .unwrap_or_else(|| home_channel(dongle)),
                c => c,
            };
            let config = air_config(dongle);
            let effect = dongle.radio.action_frame(
                &config,
                channel,
                da.try_into().unwrap(),
                bssid.try_into().unwrap(),
                body,
            );
            dongle.effects.push(effect);
            dongle.effects.push(Effect::Event {
                if_id,
                kind: WLC_E_ACTION_FRAME_COMPLETE,
                status: WLC_E_STATUS_SUCCESS,
                reason: 0,
                flags: 0,
                addr: da.try_into().unwrap(),
                data: id.to_vec(),
            });
            Ok(Vec::new())
        }
        // The EDCA parameters of the BSS the radio runs and of its station
        // side: the four entries; a set replaces the entry of its access
        // category (the ACI in bits 5-6 of the first byte).
        (_, name @ ("wme_ac_ap" | "wme_ac_sta")) => {
            let defaults = if name == "wme_ac_ap" {
                WME_AC_AP
            } else {
                WME_AC_STA
            };
            let mut table = dongle
                .settings
                .get(name)
                .cloned()
                .unwrap_or_else(|| defaults.concat());
            if !set {
                return Ok(Ok(table));
            }
            let Some(entry) = value.get(..4) else {
                return Ok(Err(BCME_BADARG));
            };
            // Entries are in ACI order (BE 0, BK 1, VI 2, VO 3).
            let aci = usize::from(entry[0] >> 5 & 3);
            table[aci * 4..aci * 4 + 4].copy_from_slice(entry);
            dongle.settings.insert(name.to_owned(), table);
            Ok(Vec::new())
        }
        // The bandwidth allowed per band: set {band, bandwidth}, get with
        // the band (`WLC_BAND_5G` 1, `WLC_BAND_2G` 2) as parameter. `wlan`
        // sets both bands when it configures a mode (`0x69d80`) before it
        // reads one (`0x71aa0`).
        (_, "bw_cap") => {
            let Some(band) = value
                .get(..4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            else {
                return Ok(Err(BCME_BADARG));
            };
            let key = format!("bw_cap {band}");
            if set {
                let Some(bandwidth) = value.get(4..8) else {
                    return Ok(Err(BCME_BADARG));
                };
                dongle.settings.insert(key, bandwidth.to_vec());
                Ok(Vec::new())
            } else {
                match dongle.settings.get(&key) {
                    Some(bandwidth) => Ok(bandwidth.clone()),
                    None => return Err(format!("get of bw_cap for band {band} never set")),
                }
            }
        }
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
        // Channel measurements (`wl_chanim_stats_t` version 2: buffer
        // length, version, count, then `chanim_stats_t`, 36 bytes each;
        // public `wlioctl.h`), which `wlan`'s channel selection reads
        // (`0x72790`, asking for all, 255). The firmware measures the
        // channel it is on; the air around the emulated radio carries only
        // what other emulators send, so it reports one sample of an idle
        // channel: no glitches or bad PLCPs, no busy time (`ccastats`
        // zero), the noise floor of the `--wifi-ssid` model, idle 255 of
        // 255. \[assumption\] The values of an idle channel.
        (false, "chanim_stats") => {
            let channel = dongle
                .radio
                .channel()
                .unwrap_or_else(|| home_channel(dongle));
            let mut sample = vec![0u8; 36];
            sample[17] = wifi_ap::NOISE_DBM as u8;
            sample[18..20].copy_from_slice(&chanspec_20(channel).to_le_bytes());
            sample[32] = 255;
            let mut stats = words(&[12 + 36, 2, 1]);
            stats.extend(sample);
            Ok(stats)
        }
        // Whether a BSS configuration is up: get with its index, set with
        // {index, up} (public `wlioctl.h`; brcmfmac's `bss` iovar). Only the
        // primary configuration (index 0) exists here; `wlan` starts its BSS
        // with `WLC_SET_SSID`, so taking one up this way is refused.
        (_, "bss") => {
            let Some(index) = value
                .get(..4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            else {
                return Ok(Err(BCME_BADARG));
            };
            if index != 0 {
                return Err(format!("bss of configuration {index}: {value:02x?}"));
            }
            if !set {
                return Ok(Ok(u32::from(dongle.radio.bss.is_some())
                    .to_le_bytes()
                    .to_vec()));
            }
            let Some(up) = value.get(4..8) else {
                return Ok(Err(BCME_BADARG));
            };
            return Err(format!("set of bss up {up:02x?}: not modeled"));
        }
        // The radio's channel before the host set one: the home channel's
        // 20 MHz chanspec (channel 1, see `home_channel`).
        (false, "chanspec") if !dongle.settings.contains_key("chanspec") => {
            Ok(u32::from(chanspec_20(home_channel(dongle)))
                .to_le_bytes()
                .to_vec())
        }
        // 20/40 MHz coexistence before the host set it: automatic (-1,
        // `WL_OBSS_COEX_AUTO`), Broadcom's default \[assumption\]; `wlan`
        // `0x71aa0` reads it before it starts a BSS.
        (false, "obss_coex") if !dongle.settings.contains_key("obss_coex") => {
            Ok((-1i32).to_le_bytes().to_vec())
        }
        // Settings the host reads before changing them: what it last set,
        // else the firmware's power-on state, which is nothing enabled (no
        // events, no Nintendo-specific subtypes, no coexistence override).
        (false, name) if READ_BACK.contains(&name) => {
            let mut data = dongle.settings.get(name).cloned().unwrap_or_default();
            data.resize(value.len(), 0);
            Ok(data)
        }
        // A stored setting reads back as the host set it.
        (false, name) if SETTINGS.contains(&name) && dongle.settings.contains_key(name) => {
            let mut data = dongle.settings[name].clone();
            data.resize(value.len().max(4), 0);
            Ok(data)
        }
        // The radio chains in use before the host chose them: what the
        // NVRAM lists (`txchain=3`, `rxchain=3` in `wlan`'s embedded NVRAM).
        (false, "txchain" | "rxchain") => match nvram_number(&dongle.nvram, &name) {
            Some(chains) => Ok(chains.to_le_bytes().to_vec()),
            None => Err(BCME_BADARG),
        },
        // The coexistence mode before the host set one: the NVRAM's
        // `btcdyn_default_btc_mode` (5 in `wlan`'s embedded NVRAM; assumed
        // to be what the firmware starts in when dynamic coexistence is
        // configured, `btcdyn_flags=0x3`).
        (false, "btc_mode") => match dongle.settings.get("btc_mode") {
            Some(mode) => Ok(mode.clone()),
            None => match nvram_number(&dongle.nvram, "btcdyn_default_btc_mode") {
                Some(mode) => Ok(mode.to_le_bytes().to_vec()),
                None => Err(BCME_BADARG),
            },
        },
        // Coexistence parameters, indexed: set {index, value}, get by index.
        // Unset ones read as the NVRAM's `btc_params<index>`, else 0
        // (assumed: the firmware's built-in value; `wlan` `0x64e70`/
        // `0x64f90` only compare it with what they want and set it when
        // it differs).
        (_, "btc_params") => {
            let Some(index) = value
                .get(..4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            else {
                return Ok(Err(BCME_BADARG));
            };
            let key = format!("btc_params{index}");
            if set {
                let Some(word) = value.get(4..8) else {
                    return Ok(Err(BCME_BADARG));
                };
                dongle.settings.insert(key, word.to_vec());
                Ok(Vec::new())
            } else {
                let word = match dongle.settings.get(&key) {
                    Some(word) => u32::from_le_bytes(word[..4].try_into().unwrap()),
                    None => nvram_number(&dongle.nvram, &key).unwrap_or(0),
                };
                Ok(word.to_le_bytes().to_vec())
            }
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
const SETTINGS: [&str; 53] = [
    "ampdu_hostreorder",
    "auth",
    "country",
    "buf_key_b4_m4",
    "closednet",
    "pmkid_info",
    "mpc",
    "nsc_subtypeoui",
    // Nintendo-specific: configuration of the firmware's offloads while the
    // console sleeps (`wlan` `0x10c8b0` clears it, 9 zero bytes, before it
    // asks for periodic action frames); nothing here depends on it.
    "nsc_aloe_cfg",
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
    // The channel the firmware prefers when it picks one itself (`wlan`
    // `0x71aa0` clears it before starting a BSS on a channel it names).
    "pref_chanspec",
    // Channel measurement for the automatic channel selection `wlan` runs
    // before it starts a BSS (`0x725e0`, `0x71aa0`): sampling period, noise
    // metric, interference parameters.
    "chanim_sample_period",
    "noise_metric",
    "intfer_params",
    "acs_update",
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
    "mkeep_alive",
    "ap_isolate",
    "chanspec",
    "mcast_list",
    // Bluetooth coexistence: the receive gain stage while Bluetooth is
    // busy, set for local communication (`wlan` `0x6b0c0`).
    "phy_btc_restage_rxgain",
    // What `wlan` sets when it starts a BSS (`0x67650` and the helpers it
    // calls): the beacons' rate, the coexistence mode and strobe, whether
    // the BSS uses WME, and how stations are probed for liveness.
    "force_bcn_rspec",
    "btc_mode",
    "btc_start_strobe",
    "wme_bss_disable",
    "scb_probe",
];
/// Iovars `wlan` reads, modifies and writes back.
const READ_BACK: [&str; 4] = ["event_msgs", "nsc_subtypes", "btc_dynctl", "nsc_afmode"];

/// A join request: the firmware finds the network, authenticates and
/// associates, which ends in events after [`ASSOCIATION_TIME`]. The access
/// point of `--wifi-ssid` is found when its SSID matches (and the BSSID,
/// unless the request leaves it open); it refuses a station whose security
/// settings (`wpa_auth`, the RSN element given with `wpaie`) do not match
/// its own. Any other network is looked for on the air shared with other
/// emulators, on `channels` (every valid channel if the request names
/// none); one nobody answers for is reported after a join scan of each of
/// them at the default active dwell.
fn join(
    dongle: &mut Dongle,
    ap: Option<&AccessPoint>,
    if_id: u8,
    ssid: &[u8],
    bssid: [u8; 6],
    channels: Vec<u8>,
) -> Result<Vec<u8>, i16> {
    if !dongle.up {
        return Err(BCME_NOTUP);
    }
    dongle.epoch += 1;
    let epoch = dongle.epoch;
    dongle.association = None;
    dongle.radio.link = None;
    let any_bssid = bssid == [0; 6] || bssid == [0xff; 6];
    let found = ap.filter(|ap| ap.ssid() == ssid && (any_bssid || bssid == wifi_ap::BSSID));
    let setting = |name: &str| dongle.settings.get(name).cloned().unwrap_or_default();
    let wpa_auth = setting("wpa_auth")
        .get(..4)
        .map_or(0, |w| u32::from_le_bytes(w.try_into().unwrap()));
    match found {
        Some(ap) if ap.accepts(wpa_auth, &setting("wpaie")) => {
            dongle.association = Some(Association {
                if_id,
                epoch,
                linked: false,
                authenticator: ap.authenticator(dongle.mac),
            });
            dongle
                .timers
                .push((ASSOCIATION_TIME, Deferred::Associated { epoch }));
        }
        Some(_) => dongle.timers.push((
            ASSOCIATION_TIME,
            Deferred::AssociationFailed {
                epoch,
                if_id,
                status: WLC_E_STATUS_FAIL,
            },
        )),
        None => {
            let channels = match channels.is_empty() {
                true => CHANNELS.to_vec(),
                false => channels,
            };
            let config = air_config(dongle);
            let effects = dongle
                .radio
                .join(&config, if_id, epoch, ssid, bssid, channels);
            dongle.effects.extend(effects);
        }
    }
    Ok(Vec::new())
}

/// One of the periodic action frames, if the host still asks for them as
/// it did when this one was scheduled: sent on the radio's channel, and
/// the next one scheduled a period later.
fn periodic_action_frame(dongle: &mut Dongle, setting: Vec<u8>) -> Vec<Effect> {
    let on = setting_word(dongle, "nsc_afmode").unwrap_or(0) != 0;
    if !on || dongle.settings.get("nsc_aloe_af_periodic") != Some(&setting) || !dongle.up {
        return Vec::new();
    }
    let period = u32::from_le_bytes(setting[..4].try_into().unwrap());
    let bssid: [u8; 6] = setting[28..34].try_into().unwrap();
    let da: [u8; 6] = setting[36..42].try_into().unwrap();
    let length = usize::from(u16::from_le_bytes([setting[42], setting[43]]));
    let Some(body) = setting.get(48..48 + length) else {
        return Vec::new();
    };
    let channel = dongle
        .radio
        .channel()
        .unwrap_or_else(|| home_channel(dongle));
    let config = air_config(dongle);
    let frame = dongle.radio.action_frame(&config, channel, da, bssid, body);
    let mut effects = vec![frame];
    if period != 0 {
        effects.push(Effect::Later {
            delay: on_air::TU * period,
            work: on_air::Work::PeriodicActionFrame { setting },
        });
    }
    effects
}

/// Whether the host put the firmware in access point mode (`WLC_SET_AP`).
fn access_point_mode(dongle: &Dongle) -> bool {
    setting_word(dongle, &format!("ioctl {WLC_SET_AP}")) == Some(1)
}

/// A setting the host gave as one word.
fn setting_word(dongle: &Dongle, name: &str) -> Option<u32> {
    dongle
        .settings
        .get(name)
        .and_then(|w| w.get(..4))
        .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
}

/// The station's association on the air, once complete.
fn air_link(dongle: &Dongle) -> Option<&on_air::Link> {
    dongle
        .radio
        .link
        .as_ref()
        .filter(|l| l.stage == on_air::Stage::Associated)
}

/// The channels of a chanspec list: a count at `at`, the chanspecs
/// (16 bits, channel in the low byte) after it.
fn chanspec_list(params: &[u8], at: usize) -> Vec<u8> {
    let count = params
        .get(at..at + 4)
        .map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap()) as usize);
    (0..count.min(CHANNELS.len()))
        .filter_map(|i| params.get(at + 4 + 2 * i).copied())
        .collect()
}

/// The channel the radio's own BSS goes on: the chanspec the host set,
/// else the channel it set (`WLC_SET_CHANNEL`), else channel 1
/// \[assumption: the firmware's default\].
fn home_channel(dongle: &Dongle) -> u8 {
    setting_word(dongle, "chanspec")
        .map(|c| c as u8)
        .or_else(|| setting_word(dongle, &format!("ioctl {WLC_SET_CHANNEL}")).map(|c| c as u8))
        .filter(|c| CHANNELS.contains(c))
        .unwrap_or(1)
}

/// `WLC_SET_SSID` in access point mode: an SSID starts the BSS on the home
/// channel, an empty one takes it down.
fn start_bss(dongle: &mut Dongle, if_id: u8, ssid: &[u8]) -> Result<Vec<u8>, i16> {
    if !dongle.up {
        return Err(BCME_NOTUP);
    }
    let config = air_config(dongle);
    let effects = match ssid.is_empty() {
        true => dongle.radio.stop_bss(&config),
        false => {
            let channel = home_channel(dongle);
            dongle.radio.start_bss(&config, if_id, ssid, channel)
        }
    };
    dongle.effects.extend(effects);
    Ok(Vec::new())
}

/// What the frames the radio sends depend on, from the host's settings.
/// Unset, the firmware's defaults: no security, open network, beacon
/// period 100 TU and DTIM period 1 (the usual defaults, assumed), at
/// most 8 stations (assumed; `wlan` sets `maxassoc` itself).
fn air_config(dongle: &Dongle) -> on_air::Config {
    let word = |name: &str| setting_word(dongle, name);
    on_air::Config {
        mac: dongle.mac,
        wsec: word("wsec").unwrap_or(0),
        wpa_ie: dongle.settings.get("wpaie").cloned().unwrap_or_default(),
        closednet: word("closednet").unwrap_or(0) != 0,
        max_assoc: word("maxassoc").unwrap_or(8),
        beacon_period: word(&format!("ioctl {WLC_SET_BCNPRD}")).map_or(100, |p| p as u16),
        dtim_period: word(&format!("ioctl {WLC_SET_DTIMPRD}")).map_or(1, |p| p as u8),
        ap_isolate: word("ap_isolate").unwrap_or(0) != 0,
    }
}

/// `wl_escan_result_t`: buffer length, version, sync ID, BSS count, BSSs.
fn escan_result(sync_id: u16, bss: &[Vec<u8>]) -> Vec<u8> {
    let length = 12 + bss.iter().map(Vec::len).sum::<usize>();
    let mut data = Vec::with_capacity(length);
    data.extend((length as u32).to_le_bytes());
    data.extend(BSS_INFO_VERSION.to_le_bytes());
    data.extend(sync_id.to_le_bytes());
    data.extend((bss.len() as u16).to_le_bytes());
    for info in bss {
        data.extend(info);
    }
    data
}

/// "escan": `wl_escan_params_t` (version 1, action, sync ID, then
/// `wl_scan_params_t`: SSID, BSSID, BSS type, scan type at 43, probes,
/// active/passive/home dwell at 48/52/56, channel and SSID counts at 60,
/// chanspecs from 64; `wlan`'s builder at `0x637d0`). Starting a scan
/// arms its end after the dwell on every channel (all valid channels when
/// the list is empty), and, if the access point is on one of them and
/// matches the SSID asked for (if any), a partial result with it when the
/// scan has dwelt on its channel; aborting a scan in flight ends it with a
/// result of status abort and drops the rest (with none in flight there
/// is nothing to report).
fn escan(
    dongle: &mut Dongle,
    ap: Option<&AccessPoint>,
    if_id: u8,
    value: &[u8],
) -> Result<Vec<u8>, i16> {
    let field = |at: usize| {
        value
            .get(at..at + 4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
    };
    let (Some(1), Some(action)) = (field(0), value.get(4..6)) else {
        return Err(BCME_BADARG);
    };
    let action = u16::from_le_bytes(action.try_into().unwrap());
    let sync_id = le16(value, 6);
    match action {
        WL_SCAN_ACTION_START => {
            let params = &value[8..];
            let (Some(&scan_type), Some(active), Some(passive), Some(counts)) =
                (params.get(43), field(8 + 48), field(8 + 52), field(8 + 60))
            else {
                return Err(BCME_BADARG);
            };
            let channels: Vec<u8> = match (counts & 0xffff) as usize {
                0 => CHANNELS.to_vec(),
                n => (0..n)
                    .map(|i| value.get(72 + 2 * i).copied().unwrap_or(0))
                    .collect(),
            };
            let passive_scan = scan_type & 1 != 0;
            let dwell_ms = match (passive_scan, active, passive) {
                (false, ms, _) if ms > 0 => ms as u64,
                (false, _, _) => DEFAULT_ACTIVE_DWELL_MS,
                (true, _, ms) if ms > 0 => ms as u64,
                (true, _, _) => DEFAULT_PASSIVE_DWELL_MS,
            };
            let dwell = Duration::from_millis(dwell_ms);
            let ssid_len = field(8).unwrap_or(0).clamp(0, 32) as usize;
            let ssid = value.get(12..12 + ssid_len).unwrap_or(&[]);
            let heard = ap.filter(|ap| ssid.is_empty() || ap.ssid() == ssid);
            if let Some(index) =
                heard.and_then(|_| channels.iter().position(|&c| c == wifi_ap::CHANNEL))
            {
                dongle.timers.push((
                    dwell * (index as u32 + 1),
                    Deferred::ScanFound { sync_id, if_id },
                ));
            }
            dongle.scan_in_flight = Some(sync_id);
            // On the air: the scan dwells on each channel in turn, hearing
            // (and, active, asking with a probe request for) what is there.
            dongle.radio.start_scan(sync_id, if_id, ssid, passive_scan);
            for (index, &channel) in channels.iter().enumerate() {
                dongle.timers.push((
                    dwell * index as u32,
                    Deferred::Air(on_air::Work::ScanChannel { sync_id, channel }),
                ));
            }
            dongle.timers.push((
                dwell * channels.len() as u32,
                Deferred::ScanDone { sync_id, if_id },
            ));
            Ok(Vec::new())
        }
        WL_SCAN_ACTION_ABORT => {
            if let Some(sync_id) = dongle.scan_in_flight.take() {
                dongle.radio.end_scan(sync_id);
                dongle
                    .timers
                    .push((Duration::ZERO, Deferred::ScanAborted { sync_id, if_id }));
            }
            Ok(Vec::new())
        }
        _ => Err(BCME_BADARG),
    }
}

/// Whether the host's event mask (`event_msgs`, one bit per event type)
/// enables `event_type`.
fn event_enabled(dongle: &Dongle, event_type: u32) -> bool {
    dongle
        .settings
        .get("event_msgs")
        .and_then(|mask| mask.get((event_type / 8) as usize))
        .is_some_and(|byte| byte >> (event_type % 8) & 1 != 0)
}

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
        if state.arm(self.clock.now()) {
            self.due.notify_one();
        }
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
        setup_on(Arc::new(GuestClock::new()))
    }

    /// The function with its BARs placed and memory decoding on, keeping
    /// guest time with `clock`.
    fn setup_on(clock: Arc<GuestClock>) -> (Bcm4356Wifi, Arc<Host>) {
        let host = Arc::new(Host::default());
        host.memory.lock().unwrap().resize(0x1_0000, 0);
        let wifi = Bcm4356Wifi::new(Box::new(HostPort(Arc::clone(&host))), Arc::default(), clock);
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

    /// Enables every firmware event, as `wlan` does for those it handles.
    fn enable_all_events(wifi: &Bcm4356Wifi) {
        let mut state = wifi.state.lock().unwrap();
        let dongle = state.dongle.as_mut().unwrap();
        dongle.settings.insert("event_msgs".into(), vec![0xff; 16]);
        dongle.up = true;
    }

    /// Joining an open network: the association events in order, then a
    /// frame transmitted on a flow ring reaches the network behind the
    /// access point and one from it reaches a posted receive buffer.
    #[test]
    fn open_network_association_and_data() {
        let (wifi, host) = setup();
        let shared = u64::from(boot(&wifi));
        let ring_info = wifi.memory_read(BAR2 + shared + 48, 4).unwrap();
        let ring_mem = wifi.memory_read(BAR2 + ring_info, 4).unwrap();
        let h2d_w = wifi.memory_read(BAR2 + ring_info + 4, 4).unwrap();
        let d2h_w = wifi.memory_read(BAR2 + ring_info + 12, 4).unwrap();
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
        enable_all_events(&wifi);
        let uplinked: Arc<Mutex<Vec<Vec<u8>>>> = Arc::default();
        let sink = Arc::clone(&uplinked);
        wifi.set_access_point(
            AccessPoint::new("emulated", None).unwrap(),
            Box::new(move |frame| sink.lock().unwrap().push(frame.to_vec())),
        );
        let post = |kind: u8, id: u32, address: u64| {
            let mut item = vec![0; 40];
            item[0] = kind;
            item[4..8].copy_from_slice(&id.to_le_bytes());
            item[8..10].copy_from_slice(&0x100u16.to_le_bytes());
            item[16..24].copy_from_slice(&address.to_le_bytes());
            item
        };
        put(&host, HOST, &post(MSG_IOCTLRESP_BUF_POST, 7, HOST + 0x8000));
        for i in 0..5u64 {
            put(
                &host,
                HOST + 40 * (i + 1),
                &post(MSG_EVENT_BUF_POST, 20 + i as u32, HOST + 0xa000 + 0x100 * i),
            );
        }
        let mut join = b"join\0".to_vec();
        let mut params = vec![0u8; 0x44];
        params[0..4].copy_from_slice(&8u32.to_le_bytes());
        params[4..12].copy_from_slice(b"emulated");
        params[56..62].fill(0xff);
        join.extend(params);
        put(&host, HOST + 0x9000, &join);
        let mut ioctl = vec![0; 40];
        ioctl[0] = MSG_IOCTLPTR_REQ;
        ioctl[8..12].copy_from_slice(&WLC_SET_VAR.to_le_bytes());
        ioctl[14..16].copy_from_slice(&(join.len() as u16).to_le_bytes());
        ioctl[24..32].copy_from_slice(&(HOST + 0x9000).to_le_bytes());
        put(&host, HOST + 240, &ioctl);
        // A receive buffer on the RX post ring.
        let mut rx = vec![0; 32];
        rx[0] = MSG_RXBUF_POST;
        rx[4..8].copy_from_slice(&30u32.to_le_bytes());
        rx[10..12].copy_from_slice(&2048u16.to_le_bytes());
        rx[24..32].copy_from_slice(&(HOST + 0xc000).to_le_bytes());
        put(&host, HOST + 0x1000, &rx);
        wifi.memory_write(BAR2 + h2d_w, 2, 7).unwrap();
        wifi.memory_write(BAR2 + h2d_w + 4, 2, 1).unwrap();
        wifi.memory_write(
            BAR0 + 0x2000 + u64::from(PCIE2_H2D_MAILBOX_0),
            4,
            0x1234_5678,
        )
        .unwrap();
        std::thread::sleep(Duration::from_millis(200));
        // Acknowledgement, completion, then the five events in order.
        assert_eq!(wifi.memory_read(BAR2 + d2h_w, 2), Some(7));
        let kinds: Vec<u32> = (0..5u64)
            .map(|i| {
                let frame = get(&host, HOST + 0xa000 + 0x100 * i, 32);
                u32::from_be_bytes(frame[28..32].try_into().unwrap())
            })
            .collect();
        assert_eq!(
            kinds,
            [
                WLC_E_AUTH,
                WLC_E_ASSOC,
                WLC_E_JOIN,
                WLC_E_LINK,
                WLC_E_SET_SSID
            ]
        );
        // Transmit on flow ring 2.
        wifi.state
            .lock()
            .unwrap()
            .dongle
            .as_mut()
            .unwrap()
            .flow_rings
            .push((
                2,
                Ring {
                    base: HOST + 0x5000,
                    max_items: 16,
                    item_len: 48,
                },
            ));
        let mut tx = vec![0; 48];
        tx[0] = MSG_TX_POST;
        tx[4..8].copy_from_slice(&40u32.to_le_bytes());
        tx[8..14].copy_from_slice(&wifi_ap::BSSID);
        tx[14..20].copy_from_slice(&[0x00, 0x90, 0x4c, 0x1a, 0xa0, 0x01]);
        tx[20..22].copy_from_slice(&[0x08, 0x00]);
        tx[32..40].copy_from_slice(&(HOST + 0x6000).to_le_bytes());
        tx[42..44].copy_from_slice(&4u16.to_le_bytes());
        put(&host, HOST + 0x5000, &tx);
        put(&host, HOST + 0x6000, b"ping");
        wifi.memory_write(BAR2 + h2d_w + 8, 2, 1).unwrap();
        wifi.memory_write(
            BAR0 + 0x2000 + u64::from(PCIE2_H2D_MAILBOX_0),
            4,
            0x1234_5678,
        )
        .unwrap();
        let sent = uplinked.lock().unwrap().clone();
        assert_eq!(sent.len(), 1);
        assert_eq!(&sent[0][14..], b"ping");
        assert_eq!(get(&host, HOST + 0x3000, 1), [MSG_TX_STATUS]);
        // A frame from the network for the station.
        let mut frame = vec![0x00, 0x90, 0x4c, 0x1a, 0xa0, 0x01];
        frame.extend(wifi_ap::BSSID);
        frame.extend([0x08, 0x00]);
        frame.extend(b"pong");
        wifi.from_network(&frame);
        let complete = get(&host, HOST + 0x4000, 32);
        assert_eq!(complete[0], MSG_RX_CMPLT);
        assert_eq!(le32(&complete, 4), 30);
        assert_eq!(usize::from(le16(&complete, 14)), frame.len());
        assert_eq!(get(&host, HOST + 0xc000, frame.len()), frame);
    }

    /// A message posting a host buffer of 2016 bytes.
    fn post(kind: u8, id: u32, address: u64) -> Vec<u8> {
        let mut item = vec![0; 40];
        item[0] = kind;
        item[4..8].copy_from_slice(&id.to_le_bytes());
        item[8..10].copy_from_slice(&2016u16.to_le_bytes());
        item[16..24].copy_from_slice(&address.to_le_bytes());
        item
    }

    /// Boots the firmware with every event enabled, sets up the common
    /// rings, posts an ioctl response buffer and an event buffer, then
    /// starts "escan" (sync ID 0x1234) over channels 1 and 6 with
    /// `dwell_ms` of active dwell. Returns the addresses of the H2D and D2H
    /// write indices, the request and its control message.
    fn start_escan(wifi: &Bcm4356Wifi, host: &Host, dwell_ms: i32) -> (u64, u64, Vec<u8>, Vec<u8>) {
        let shared = u64::from(boot(wifi));
        enable_all_events(wifi);
        let ring_info = wifi.memory_read(BAR2 + shared + 48, 4).unwrap();
        let ring_mem = wifi.memory_read(BAR2 + ring_info, 4).unwrap();
        let h2d_w = wifi.memory_read(BAR2 + ring_info + 4, 4).unwrap();
        let d2h_w = wifi.memory_read(BAR2 + ring_info + 12, 4).unwrap();
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
        put(host, HOST, &post(MSG_IOCTLRESP_BUF_POST, 7, HOST + 0x8000));
        put(host, HOST + 40, &post(MSG_EVENT_BUF_POST, 9, HOST + 0xa000));
        let mut request = b"escan\0".to_vec();
        let mut params = vec![0u8; 76];
        params[0..4].copy_from_slice(&1u32.to_le_bytes());
        params[4..6].copy_from_slice(&WL_SCAN_ACTION_START.to_le_bytes());
        params[6..8].copy_from_slice(&0x1234u16.to_le_bytes());
        params[56..60].copy_from_slice(&dwell_ms.to_le_bytes());
        params[60..64].copy_from_slice(&(-1i32).to_le_bytes());
        params[68..72].copy_from_slice(&2u32.to_le_bytes());
        params[72..76].copy_from_slice(&[1, 0x10, 6, 0x10]);
        request.extend(params);
        put(host, HOST + 0x9000, &request);
        let mut ioctl = vec![0; 40];
        ioctl[0] = MSG_IOCTLPTR_REQ;
        ioctl[8..12].copy_from_slice(&WLC_SET_VAR.to_le_bytes());
        ioctl[12..14].copy_from_slice(&1u16.to_le_bytes());
        ioctl[14..16].copy_from_slice(&(request.len() as u16).to_le_bytes());
        ioctl[24..32].copy_from_slice(&(HOST + 0x9000).to_le_bytes());
        put(host, HOST + 80, &ioctl);
        wifi.memory_write(BAR2 + h2d_w, 2, 3).unwrap();
        wifi.memory_write(
            BAR0 + 0x2000 + u64::from(PCIE2_H2D_MAILBOX_0),
            4,
            0x1234_5678,
        )
        .unwrap();
        (h2d_w, d2h_w, request, ioctl)
    }

    /// The escan result in the first posted event buffer: its status and
    /// sync ID.
    fn escan_result_event(host: &Host) -> (u32, u16) {
        let message = get(host, HOST + 0x2000 + 48, 24);
        assert_eq!(message[0], MSG_WL_EVENT);
        assert_eq!(le32(&message, 4), 9, "in the posted event buffer");
        let frame = get(host, HOST + 0xa000, usize::from(le16(&message, 12)));
        assert_eq!(
            u32::from_be_bytes(frame[28..32].try_into().unwrap()),
            WLC_E_ESCAN_RESULT
        );
        (
            u32::from_be_bytes(frame[32..36].try_into().unwrap()),
            le16(&frame, 72 + 8),
        )
    }

    #[test]
    fn escan_ends_after_its_dwell_time_or_when_aborted() {
        let (wifi, host) = setup();
        // "escan" over two channels with 5 ms active dwell.
        let (h2d_w, d2h_w, request, ioctl) = start_escan(&wifi, &host, 5);
        // Acknowledged and completed at once; the result follows the dwell.
        assert_eq!(wifi.memory_read(BAR2 + d2h_w, 2), Some(2));
        assert_eq!(
            le16(&get(&host, HOST + 0x2000 + 24, 24), 8),
            0,
            "escan accepted"
        );
        std::thread::sleep(std::time::Duration::from_millis(200));
        assert_eq!(wifi.memory_read(BAR2 + d2h_w, 2), Some(3));
        let message = get(&host, HOST + 0x2000 + 48, 24);
        assert_eq!(message[0], MSG_WL_EVENT);
        assert_eq!(le32(&message, 4), 9, "in the posted event buffer");
        let len = usize::from(le16(&message, 12));
        let frame = get(&host, HOST + 0xa000, len);
        assert_eq!(&frame[12..14], &[0x88, 0x6c]);
        assert_eq!(&frame[19..22], &[0x00, 0x10, 0x18]);
        assert_eq!(
            u32::from_be_bytes(frame[28..32].try_into().unwrap()),
            WLC_E_ESCAN_RESULT
        );
        assert_eq!(
            u32::from_be_bytes(frame[32..36].try_into().unwrap()),
            WLC_E_STATUS_SUCCESS
        );
        assert_eq!(u32::from_be_bytes(frame[44..48].try_into().unwrap()), 12);
        assert_eq!(le16(&frame, 72 + 8), 0x1234, "sync ID");
        assert_eq!(le16(&frame, 72 + 10), 0, "no BSS");

        // A scan that dwells a minute on each channel, and its abort: it
        // ends at once, with a result that says so.
        let mut long_scan = request.clone();
        long_scan[6 + 6..6 + 8].copy_from_slice(&0x4321u16.to_le_bytes());
        long_scan[6 + 56..6 + 60].copy_from_slice(&60_000i32.to_le_bytes());
        let mut abort = long_scan.clone();
        abort[6 + 4..6 + 6].copy_from_slice(&WL_SCAN_ACTION_ABORT.to_le_bytes());
        put(&host, HOST + 0x9400, &long_scan);
        put(&host, HOST + 0x9800, &abort);
        put(
            &host,
            HOST + 120,
            &post(MSG_IOCTLRESP_BUF_POST, 8, HOST + 0x8000),
        );
        put(
            &host,
            HOST + 160,
            &post(MSG_IOCTLRESP_BUF_POST, 10, HOST + 0x8800),
        );
        put(
            &host,
            HOST + 200,
            &post(MSG_EVENT_BUF_POST, 11, HOST + 0xb000),
        );
        for (slot, transaction, address) in [(240, 2u16, HOST + 0x9400), (280, 3, HOST + 0x9800)] {
            let mut next = ioctl.clone();
            next[12..14].copy_from_slice(&transaction.to_le_bytes());
            next[24..32].copy_from_slice(&address.to_le_bytes());
            put(&host, HOST + slot, &next);
        }
        wifi.memory_write(BAR2 + h2d_w, 2, 8).unwrap();
        wifi.memory_write(
            BAR0 + 0x2000 + u64::from(PCIE2_H2D_MAILBOX_0),
            4,
            0x1234_5678,
        )
        .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(200));
        assert_eq!(wifi.memory_read(BAR2 + d2h_w, 2), Some(8));
        let message = get(&host, HOST + 0x2000 + 7 * 24, 24);
        assert_eq!((message[0], le32(&message, 4)), (MSG_WL_EVENT, 11));
        let frame = get(&host, HOST + 0xb000, usize::from(le16(&message, 12)));
        assert_eq!(
            u32::from_be_bytes(frame[32..36].try_into().unwrap()),
            WLC_E_STATUS_ABORT
        );
        assert_eq!(le16(&frame, 72 + 8), 0x4321, "sync ID");
    }

    /// A scan's end, due when the machine is saved, is reported by a
    /// function built afresh and restored from the snapshot when guest time
    /// reaches the moment it was due at: not while guest time stands still,
    /// not before, and once.
    #[test]
    fn scheduled_work_survives_a_snapshot() {
        let saved_at = Duration::from_secs(10);
        let clock = Arc::new(GuestClock::new());
        clock.pause();
        clock.set(saved_at);
        let (wifi, host) = setup_on(Arc::clone(&clock));
        // Two channels of 300 ms each: the scan ends 600 ms on.
        let (_, d2h_w, _, _) = start_escan(&wifi, &host, 300);
        let due = saved_at + Duration::from_millis(600);
        assert_eq!(wifi.memory_read(BAR2 + d2h_w, 2), Some(2));
        wifi.quiesce().unwrap();
        let saved = wifi.save().unwrap();
        drop(wifi);

        // The host memory is the machine's, restored with it.
        let clock = Arc::new(GuestClock::new());
        clock.pause();
        clock.set(saved_at);
        let restored = Bcm4356Wifi::new(
            Box::new(HostPort(Arc::clone(&host))),
            Arc::default(),
            Arc::clone(&clock),
        );
        restored.restore(&saved).unwrap();
        clock.set(due + Duration::from_millis(100));
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(
            restored.memory_read(BAR2 + d2h_w, 2),
            Some(2),
            "nothing runs while guest time stands still"
        );
        clock.set(due - Duration::from_millis(200));
        clock.resume();
        let host_deadline = std::time::Instant::now() + Duration::from_secs(5);
        while restored.memory_read(BAR2 + d2h_w, 2) == Some(2) {
            assert!(std::time::Instant::now() < host_deadline, "never reported");
            std::thread::sleep(Duration::from_millis(2));
        }
        let seen_at = clock.now();
        assert!(seen_at >= due, "reported at {seen_at:?}, due at {due:?}");
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(restored.memory_read(BAR2 + d2h_w, 2), Some(3), "once");
        assert_eq!(escan_result_event(&host), (WLC_E_STATUS_SUCCESS, 0x1234));
    }

    /// A new download (what `wlan` does on every wake) halts the running
    /// firmware: the work it had scheduled goes with it, and nothing of it
    /// reaches the firmware started next.
    #[test]
    fn a_new_download_drops_the_work_the_old_firmware_scheduled() {
        let (wifi, host) = setup();
        start_escan(&wifi, &host, 300);
        assert!(!wifi.state.lock().unwrap().scheduled.is_empty());
        boot(&wifi);
        let state = wifi.state.lock().unwrap();
        assert!(state.scheduled.is_empty());
        assert!(
            state
                .dongle
                .as_ref()
                .is_some_and(|d| d.scan_in_flight.is_none())
        );
    }

    /// After "up" the firmware announces wl0 (WLC_E_IF, ADD) when the
    /// host's event mask enables it, after the request's completion.
    #[test]
    fn up_announces_the_primary_interface() {
        let (wifi, host) = setup();
        let shared = u64::from(boot(&wifi));
        let ring_info = wifi.memory_read(BAR2 + shared + 48, 4).unwrap();
        let ring_mem = wifi.memory_read(BAR2 + ring_info, 4).unwrap();
        let h2d_w = wifi.memory_read(BAR2 + ring_info + 4, 4).unwrap();
        let d2h_w = wifi.memory_read(BAR2 + ring_info + 12, 4).unwrap();
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
        let post = |kind: u8, id: u32, address: u64| {
            let mut item = vec![0; 40];
            item[0] = kind;
            item[4..8].copy_from_slice(&id.to_le_bytes());
            item[8..10].copy_from_slice(&2016u16.to_le_bytes());
            item[16..24].copy_from_slice(&address.to_le_bytes());
            item
        };
        let ioctl = |command: u32, len: usize, request: u64| {
            let mut item = vec![0; 40];
            item[0] = MSG_IOCTLPTR_REQ;
            item[8..12].copy_from_slice(&command.to_le_bytes());
            item[14..16].copy_from_slice(&(len as u16).to_le_bytes());
            item[24..32].copy_from_slice(&request.to_le_bytes());
            item
        };
        put(&host, HOST, &post(MSG_IOCTLRESP_BUF_POST, 7, HOST + 0x8000));
        put(
            &host,
            HOST + 40,
            &post(MSG_IOCTLRESP_BUF_POST, 8, HOST + 0x8800),
        );
        put(
            &host,
            HOST + 80,
            &post(MSG_EVENT_BUF_POST, 9, HOST + 0xa000),
        );
        // Event mask with WLC_E_IF (54: byte 6, bit 6).
        let mut mask = b"event_msgs\0".to_vec();
        mask.extend([0, 0, 0, 0, 0, 0, 0x40, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        put(&host, HOST + 0x9000, &mask);
        put(
            &host,
            HOST + 120,
            &ioctl(WLC_SET_VAR, mask.len(), HOST + 0x9000),
        );
        put(&host, HOST + 160, &ioctl(WLC_UP, 0, HOST + 0x9800));
        wifi.memory_write(BAR2 + h2d_w, 2, 5).unwrap();
        wifi.memory_write(
            BAR0 + 0x2000 + u64::from(PCIE2_H2D_MAILBOX_0),
            4,
            0x1234_5678,
        )
        .unwrap();
        // Two acknowledgements and completions, then the event.
        assert_eq!(wifi.memory_read(BAR2 + d2h_w, 2), Some(5));
        let message = get(&host, HOST + 0x2000 + 4 * 24, 24);
        assert_eq!(message[0], MSG_WL_EVENT);
        let frame = get(&host, HOST + 0xa000, usize::from(le16(&message, 12)));
        assert_eq!(
            u32::from_be_bytes(frame[28..32].try_into().unwrap()),
            WLC_E_IF
        );
        assert_eq!(&frame[72..74], &[0, WLC_E_IF_ADD], "ifidx 0, ADD");
    }

    #[test]
    fn unknown_request_is_refused() {
        let (wifi, _) = setup();
        assert_eq!(wifi.config_read(0x2c, 4), None);
        wifi.config_write(CFG_BAR0_WINDOW, 4, D11_BASE).unwrap();
        assert_eq!(wifi.memory_read(BAR0, 4), None);
    }
}
