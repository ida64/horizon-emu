//! Power management controller: pad voltages, deep power down requests,
//! wake sources, scratch registers and the power partitions. Reached only
//! through the secure monitor's register call.
//!
//! Most of it is configuration storage. `boot` sets up the wake pins by
//! reading a register whole and writing it back whole, so what the block
//! starts with only survives in bits `boot` leaves alone; registers start
//! at zero, which stands for whatever the earlier boot stages programmed.
//!
//! The power partitions behave: the firmware's clock driver turns a
//! partition on or off by writing its ID with the start bit to
//! `PWRGATE_TOGGLE`, waits for the start bit to clear ("accept"), then for
//! the partition's bit in `PWRGATE_STATUS` (`VccPgSetPowerEnabled`, pcv
//! text `0x3d730`: up to 100 polls each, and it fails the request
//! otherwise). Here a partition switches at once. Register and partition
//! names are as in Linux (`drivers/soc/tegra/pmc.c`, Tegra210 table),
//! recalled.
//!
//! So do the wake events, as far as something here raises one: an event
//! that is enabled and at its wake level is recorded in the status
//! registers, which is where the firmware looks, one event at a time,
//! for why the console woke. See [`Pmc::wake`].
//!
//! # Deep power down (SC7)
//!
//! What the secure monitor's SC7-entry stub does on the BPMP
//! (`warmboot-research` §5; names and offsets hekate `pmc_regs.h`):
//!
//! * `SET_SW_CLAMP` (`0x47c`) bit 0 clamps the CPU rail: `CLAMP_STATUS`
//!   (`0x02c`) bit 0 reads set until `REMOVE_CLAMPING` (`0x034`) is written
//!   with that bit (the warmboot firmware does, after powering the rail);
//!   a write of `REMOVE_CLAMPING` clears the clamp bits it has set, at
//!   once. \[assumption\] Nothing else sets a clamp here: toggling a
//!   partition leaves `CLAMP_STATUS` alone (the sequences in the boot
//!   chain remove a clamp before waiting on it, which then passes either
//!   way).
//! * `IO_DPD3_REQ` / `IO_DPD4_REQ` (`0x45c`, `0x464`): bits 31:30 the code
//!   (2 puts the pads selected by the low bits in deep power down, 1 takes
//!   them out, Linux `tegra_io_pad_prepare`), and `IO_DPD3_STATUS` /
//!   `IO_DPD4_STATUS` (`0x460`, `0x468`) follow at once. Pads that do not
//!   exist never read set: \[inference\] masks `0x0FFFFFFF` and
//!   `0x0FFF1FFF`, the status the monitor's SC7 stub waits for after
//!   requesting all ones (its literals at text `0x400033A0`/`0x400033A4`).
//! * `DPD_ENABLE` (`0x024`) bit 0 puts the SoC into SC7: storage here; the
//!   machine's BPMP loop looks at it ([`dpd_entered`]) and powers the SoC
//!   down.
//! * `PMC_CNTRL` (`0x000`) bit 4 `MAIN_RST` resets the SoC (the stub's and
//!   the warmboot firmware's failure path): refused in a machine powered
//!   on, which does not model a SoC reset.
//!
//! In SC7 the PMC keeps its state (it is always on), except that the
//! wake takes the SoC out of deep power down (`DPD_ENABLE` bit 0 clear)
//! and the secure-scratch lock registers (`SEC_DISABLE` .. `SEC_DISABLE8`)
//! reset to their power-on values ([`SECURE_LOCKS`], [`power_on_value`]);
//! the machine's SC7 cycle writes both when it wakes the SoC.

use std::sync::Mutex;

use super::regs::{RegisterFile, word};
use crate::{
    bus::MmioDevice,
    snapshot::{Snapshot, SnapshotError, decode, encode},
};

pub const BASE: u64 = 0x7000_E400;
pub const SIZE: u64 = 0xC00;

/// `PMC_CNTRL` and its `MAIN_RST` bit (hekate `PMC_CNTRL_MAIN_RST`).
const CNTRL: u64 = 0x000;
const CNTRL_MAIN_RST: u32 = 1 << 4;
/// `DPD_SAMPLE` and `DPD_ENABLE`; bit 0 of the latter: SC7.
pub const DPD_SAMPLE: u64 = 0x020;
pub const DPD_ENABLE: u64 = 0x024;
pub const DPD_ENABLE_ON: u32 = 1 << 0;
const CLAMP_STATUS: u64 = 0x02C;
const REMOVE_CLAMPING: u64 = 0x034;
const SET_SW_CLAMP: u64 = 0x47C;
/// `SET_SW_CLAMP` bit 0 and the `CLAMP_STATUS` bit it sets: the CPU rail
/// (CRAIL, partition 0).
const CLAMP_CRAIL: u32 = 1 << 0;
/// `IO_DPD3`/`IO_DPD4`: request, status, and the pads that exist.
const IO_DPD: [(u64, u64, u32); 2] = [(0x45C, 0x460, 0x0FFF_FFFF), (0x464, 0x468, 0x0FFF_1FFF)];
const IO_DPD_CODE_SHIFT: u32 = 30;
const IO_DPD_ON: u32 = 2;
const IO_DPD_OFF: u32 = 1;

/// `SEC_DISABLE` .. `SEC_DISABLE8` (hekate `pmc_regs.h`): the write and
/// read locks of the scratch registers. \[inference\] They reset on SC7
/// exit: the secure monitor locks `SECURE_SCRATCH24..27` and `112..115`
/// (`SEC_DISABLE3` bits 7:0, `SEC_DISABLE8` bits 23:16) at every suspend
/// and panics if it finds them already locked (`warmboot-research` §4 step
/// 9), and the console sleeps more than once; the warmboot firmware applies
/// its own locks again after the wake (§7 step 11.8).
pub const SECURE_LOCKS: [u64; 8] = [0x004, 0x2C4, 0x2D8, 0x5B0, 0x5B4, 0x5B8, 0x5BC, 0x5C0];

const PWRGATE_TOGGLE: u64 = 0x30;
const TOGGLE_START: u32 = 1 << 8;
const TOGGLE_PARTITION: u32 = 0x1F;
/// Which power partitions are on, a bit per partition ID.
pub const PWRGATE_STATUS: u64 = 0x38;
/// `RST_STATUS`: why the SoC last came out of reset, bits 2:0. 0 is a
/// power-on. \[inference\] 4 is a wake from deep power down (LP0): the
/// boot ROM takes 1, 2, 3 and 5 for the watchdog, thermal and software
/// resets it handles apart (`0x102FA6`) and does its LP0 step on 4, and
/// the Switch's fourth ROM patch restores the straps from scratch
/// whenever the register is not 0 (switchbrew "Fuses", "IROM patch 4":
/// "backing up and restoring strapping options for warmboot"). The
/// machine's SC7 cycle writes it when it wakes a SoC that runs the ROM.
pub const RST_STATUS: u64 = 0x1B4;
pub const RST_STATUS_LP0: u32 = 4;

/// Whether `DPD_ENABLE` (as read) puts the SoC into SC7.
pub fn dpd_entered(dpd_enable: u32) -> bool {
    dpd_enable & DPD_ENABLE_ON != 0
}

/// What register `offset` holds as the SoC comes out of reset
/// ([`Pmc::power_on`]), if that is established.
pub fn power_on_value(offset: u64) -> Option<u32> {
    HWCHECK_STATE
        .iter()
        .chain(&POWER_ON_UNCHECKED)
        .find_map(|&(register, value)| (register == offset).then_some(value))
}

/// The 64 wake events, 32 to a register: which may wake the console
/// (`WAKE_MASK`, `WAKE2_MASK`), at which level of their input, high if
/// set (`WAKE_LVL`, `WAKE2_LVL`: the firmware sets the one event its
/// table has as active high, the radio's, and leaves the active low ones
/// clear), and which have (`WAKE_STATUS`, `WAKE2_STATUS`). The firmware
/// clears the status by writing all ones before the console sleeps and
/// reads it 64 times after (Bus through the monitor), so a write clears
/// the bits it has set.
const WAKE_MASK: [u64; 2] = [0x0C, 0x160];
const WAKE_LVL: [u64; 2] = [0x10, 0x164];
const WAKE_STATUS: [u64; 2] = [0x14, 0x168];
/// The events whose wake level the PMC takes from their pads as the SoC
/// enters deep power down (`AUTO_WAKE_LVL_MASK`, `AUTO_WAKE2_LVL_MASK`):
/// see [`Pmc::sample_wake_levels`]. The firmware sets the bit of every
/// event its table gives the level "auto" and clears that event's
/// `WAKE_LVL` bit (Atmosphère `wec_api.cpp` `SetWakeEventLevel`; a 23.5.0
/// boot traced through the monitor's register call left `0x17FEBCFF` and
/// `0xEFE7E01A` here, the rail CTS events among them, and never wrote
/// `AUTO_WAKE_LVL` at `0xD8` other than with 0).
const AUTO_WAKE_LVL_MASK: [u64; 2] = [0xDC, 0x170];

/// The wake event of the power management IC's interrupt output, active
/// low (`PwrIntN` in Atmosphère's list of this board's wake events,
/// `wec_wake_event.board.nintendo_nx.hpp`; the firmware enables it, bit
/// 19 of `WAKE2_MASK`).
pub const WAKE_EVENT_PMIC: u8 = 51;
/// The wake events of the CTS pads of UART-B and UART-C, the right and
/// the left Joy-Con rail's (`Uart2Cts`, `Uart3Cts` in the same list; the
/// pad map `gpio_internal_pad_map_combination.inc` gives them to pads G3
/// and D4). The firmware enables both, level "auto" (bits 6 and 7 of
/// `WAKE_MASK` and `AUTO_WAKE_LVL_MASK` in the trace above).
pub const WAKE_EVENT_UART2_CTS: u8 = 6;
pub const WAKE_EVENT_UART3_CTS: u8 = 7;

/// Partitions powered at kernel entry: the CPU rail (CRAIL, 0), the
/// non-CPU part of the fast cluster (C0NC, 15) and its first core (CE0, 14)
/// — fusee `fusee_cpu.cpp:141-143` powers these before starting the CPU.
/// The other three cores (CE1-3, 9-11) are the secure monitor's to switch,
/// when the kernel starts and stops them. Every other partition is assumed
/// off: no public stage of the boot chain powers one, and the firmware's
/// clock driver turns on what it needs.
const POWERED_AT_ENTRY: u32 = 1 << 0 | 1 << 14 | 1 << 15;

/// The GPU rail's gate: bit 0 set clamps the rail off. The firmware's
/// clock driver reads it when it starts, to count the rail as in use if
/// it finds it open, and writes 0 to open and 1 to close it (pcv text
/// `0x3db30`, `0x3d9c0`); with the rail counted twice it refuses the power
/// request by which nvservices ends its start-up GPU cycle. The rail is
/// closed at entry: the GPU's regulator is off (see `max77621`), and no
/// boot stage uses the GPU.
const GPU_RG_CNTRL: u64 = 0x2D4;
const GPU_RAIL_CLAMPED: u32 = 1 << 0;

/// `nxbl_hwcheck.csv`'s PMC rows: `(offset, expected & mask)`; see
/// [`Pmc::power_on`].
#[rustfmt::skip]
const HWCHECK_STATE: [(u64, u32); 143] = [
    (0x000, 0x00410A00), (0x004, 0x00000000), (0x008, 0x00000000),
    (0x00C, 0x00000000), (0x010, 0xFF9FFFFF), (0x014, 0x00000000),
    (0x018, 0x00000000), (0x01C, 0x00200000), (0x020, 0x00000000),
    (0x024, 0x00000000), (0x028, 0xEDCBA987), (0x030, 0x00000000),
    (0x034, 0x00000000), (0x03C, 0x003F007F), (0x040, 0xFFFFFFFF),
    (0x044, 0x00010080), (0x048, 0x00FEB425), (0x04C, 0x00000001),
    (0x0C8, 0x0000FFFF), (0x0CC, 0x0000FFFF), (0x0D0, 0xFFFFFFFF),
    (0x0D4, 0xFFFFFFFF), (0x0D8, 0x00000000), (0x0DC, 0x00000000),
    (0x0E0, 0x00000000), (0x0E4, 0x007C3C2C), (0x0E8, 0x0000000F),
    (0x0EC, 0x00000000), (0x0F4, 0x00000000), (0x0F8, 0x00018416),
    (0x154, 0x00000000), (0x158, 0x00000000), (0x15C, 0x00000000),
    (0x160, 0x00000000), (0x164, 0xF9FFFFE7), (0x168, 0x00000000),
    (0x16C, 0x00000000), (0x170, 0x00000000), (0x174, 0xFFFFFFFF),
    (0x178, 0x000000FF), (0x17C, 0x000000FF), (0x180, 0x000000FF),
    (0x184, 0xEDCBA987), (0x188, 0x00000000), (0x18C, 0x00000000),
    (0x190, 0x00000000), (0x194, 0x00000000), (0x198, 0x00000000),
    (0x19C, 0x00000000), (0x1A0, 0x00000000), (0x1A4, 0x0000007E),
    (0x1A8, 0x00000000), (0x1AC, 0x00000000), (0x1B0, 0x00000000),
    (0x1B8, 0x00000000), (0x1BC, 0x00000000), (0x1C0, 0x00000000),
    (0x1C4, 0x02000000), (0x1C8, 0x0000007F), (0x1CC, 0x00000003),
    (0x1D0, 0x00000000), (0x1D4, 0x00000000), (0x1DC, 0x00002A02),
    (0x1E0, 0x00000000), (0x1E4, 0x0000001B), (0x1E8, 0x00000000),
    (0x1F0, 0x0F0F0F0F), (0x1F8, 0x00041041), (0x1FC, 0xC0C0C0C0),
    (0x200, 0x00000000), (0x204, 0x23232363), (0x208, 0x23232363),
    (0x20C, 0x23232363), (0x210, 0x16161616), (0x218, 0x01111111),
    (0x264, 0x80000003), (0x26C, 0x0000001F), (0x270, 0x0000000D),
    (0x274, 0x00000000), (0x278, 0x0000000F), (0x280, 0x00000F07),
    (0x284, 0x000000C0), (0x288, 0x00000000), (0x28C, 0x06060606),
    (0x294, 0x00001101), (0x298, 0x00000001), (0x29C, 0x00000000),
    (0x2A0, 0x00000000), (0x2A4, 0x000000FF), (0x2A8, 0xFFFFFFFF),
    (0x2AC, 0xFFFFFFFF), (0x2B0, 0x00000000), (0x2B4, 0x000016E0),
    (0x2B8, 0x0000001F), (0x2BC, 0x00020000), (0x2C0, 0x00000000),
    (0x2C4, 0x00000000), (0x2C8, 0x00000000), (0x2CC, 0x00000000),
    (0x2D0, 0x00000000), (0x2D4, 0x00000001), (0x2D8, 0x00000000),
    (0x2DC, 0xFFFFFFFF), (0x2E0, 0x000000FF), (0x440, 0x00000001),
    (0x444, 0x01F00000), (0x448, 0x1DF30000), (0x44C, 0x00000000),
    (0x450, 0x00020200), (0x45C, 0x0FFF0000), (0x460, 0x0FFF0000),
    (0x464, 0x00000000), (0x468, 0x00000000), (0x474, 0x00000010),
    (0x478, 0x0000007F), (0x47C, 0x00000000), (0x480, 0x00000000),
    (0x484, 0x00000001), (0x488, 0x007F80FA), (0x48C, 0x00000005),
    (0x490, 0x000000FA), (0x498, 0x00000000), (0x49C, 0x00000001),
    (0x4A0, 0x00000000), (0x4A4, 0x00000000), (0x4B4, 0x00000000),
    (0x4B8, 0x00000000), (0x4BC, 0x00000000), (0x4C0, 0x28400000),
    (0x4C4, 0x28400000), (0x4C8, 0x28400000), (0x4CC, 0x28400000),
    (0x4D0, 0x000000C0), (0x4D4, 0x00000000), (0x4D8, 0x00000000),
    (0x4DC, 0x000000FC), (0x4E0, 0x23232363), (0x4E4, 0x0007FF9F),
    (0x5B0, 0x00000000), (0x5B4, 0x00000000), (0x5B8, 0x00000000),
    (0x5BC, 0x00000000), (0x5C0, 0x00004000),
];

/// Registers the boot chain reads that the check does not: see
/// [`Pmc::power_on`].
const POWER_ON_UNCHECKED: [(u64, u32); 11] = [
    (SCRATCH0, 0),
    (SCRATCH20, 0),
    (SCRATCH37, 0),
    (USB_AO, 0),
    (RST_STATUS, 0),
    (SCRATCH49, 0),
    (0x810, 0),
    (0x840, 0),
    (CLAMP_STATUS, 0),
    (SECURE_SCRATCH21, 0),
    (SECURE_SCRATCH35, 0),
];

/// `USB_AO`: the boot ROM's USB set-up clears bits 2 and 3 (the wake
/// detectors' power-downs of port 0, as hekate's same step names them) by
/// read-modify-write (`0x10706A`); nothing else in the boot chain reads
/// it. \[assumption\] It reads 0 at power-on: its reset value is not in
/// the public sources, and the only use made of it leaves the bits the
/// ROM clears clear either way.
const USB_AO: u64 = 0x0F0;
/// `SCRATCH49`: the boot ROM clears bits 1:0 by read-modify-write as it
/// leaves a cold boot (`0x101328`). 0 at power-on like the other scratch
/// registers ([`Pmc::power_on`]).
const SCRATCH49: u64 = 0x244;
/// `SCRATCH20`: "ODM data/config scratch" in hekate's header; hekate
/// clears its debug console bits 19:18 by read-modify-write at start.
/// 0 at power-on like the other scratch registers.
const SCRATCH20: u64 = 0x0A0;
/// `SCRATCH37`: hekate looks there for the mark of a kernel panic it left
/// before a reboot. 0 at power-on like the other scratch registers.
const SCRATCH37: u64 = 0x130;

/// `SCRATCH0`: the boot ROM's flags, which it reads first of all
/// (`0x1023BE`): bit 0 a warm boot (the monitor sets it before SC7), bit 1
/// recovery mode.
const SCRATCH0: u64 = 0x050;

/// `SECURE_SCRATCH21`: the secure monitor's cold boot sets bit 4 by
/// read-modify-write (`warmboot-research` §3, monitor `0x1F0146D14`).
const SECURE_SCRATCH21: u64 = 0x334;
/// `SECURE_SCRATCH35`: the high half of the resume vector, which the
/// warmboot firmware puts in `SB_AA64_RESET_HIGH` (`0x4001077C`). The
/// monitor writes only the low half, `SECURE_SCRATCH34` (`0x1F0147010`
/// stores `0x1F0089768` and nothing at `0x76C`), so the high half is what
/// the register holds from power-on.
const SECURE_SCRATCH35: u64 = 0x36C;

/// The scratch and secure scratch registers the NX bootloader packs the
/// SDRAM parameters into for the boot ROM's warm path (text `0x4000D904`,
/// `nxbootloader-research` §2.2 and §2.5): 251 words, each changed by
/// read-modify-write (`SCRATCHn = field | SCRATCHn & mask`; the offsets
/// listed by the power-on driver from that routine), first and last of
/// each run.
const LP0_SCRATCH: [(u64, u64); 14] = [
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

/// The scratch registers of [`LP0_SCRATCH`], each with its power-on value
/// 0 (\[assumption\], as for `SCRATCH188` in [`Pmc::power_on`]).
fn lp0_scratch_at_power_on() -> impl Iterator<Item = (u64, u32)> {
    LP0_SCRATCH
        .into_iter()
        .flat_map(|(first, last)| (first..=last).step_by(4))
        .map(|offset| (offset, 0))
}

pub struct Pmc {
    registers: RegisterFile,
    powered: Mutex<u32>,
    /// A machine powered on: a SoC reset request is refused.
    powered_on: bool,
}

impl Pmc {
    pub fn new() -> Self {
        Self {
            registers: RegisterFile::zeroed([(GPU_RG_CNTRL, GPU_RAIL_CLAMPED)]),
            powered: Mutex::new(POWERED_AT_ENTRY),
            powered_on: false,
        }
    }

    /// The block as the SoC comes out of reset, for a machine that runs
    /// the boot chain itself ([`Pmc::new`] is the state at kernel entry).
    ///
    /// Every partition is off, the CPU's included: the NX bootloader powers
    /// CRAIL, C0NC and CE0 before it starts core 0, each only if
    /// `PWRGATE_STATUS` lacks it (`nxbootloader-research` §2.2; hekate
    /// `ccplex_boot_cpu0` toggles them unconditionally).
    ///
    /// The registers hold what the NX bootloader's hardware-state check
    /// requires before it writes any (`nxbl_hwcheck.csv`, text
    /// `0x40002A7C`; 143 registers), the unchecked bits clear
    /// (\[assumption\]); see `car::Car::power_on` for why that check is the
    /// evidence. Other registers the boot chain reads are not checked:
    /// `RST_STATUS` (`0x1b4`) reads 0, the reset reason of a power-on
    /// (Switchbrew "Fuses" via `pkg1-research` §4; the bootloader takes any
    /// other value for a warm reset); the scratch registers `SCRATCH49`
    /// (`0x244`), `SCRATCH188` (`0x810`, changed by read-modify-write),
    /// `SCRATCH200` (`0x840`, error codes of an earlier boot), those of
    /// `LP0_SCRATCH` and `SECURE_SCRATCH21` and `35` (`0x334`, `0x36c`)
    /// read 0 (\[assumption\]: scratch
    /// registers lose their contents on a power-on reset, which is why they
    /// carry state only across warm resets; where the check has a row for
    /// one, its value wins); and `CLAMP_STATUS` reads 0 (no software clamp
    /// set, see the module documentation). Every other register refuses to
    /// be read until written.
    pub fn power_on() -> Self {
        Self {
            registers: RegisterFile::strict(
                lp0_scratch_at_power_on()
                    .chain(POWER_ON_UNCHECKED)
                    .chain(HWCHECK_STATE),
            ),
            powered: Mutex::new(0),
            powered_on: true,
        }
    }

    /// The input of wake event `event` is at this level. If the event is
    /// enabled and that is its wake level, it is recorded and the answer
    /// is `true`: the console wakes.
    pub fn wake(&self, event: u8, high: bool) -> bool {
        let (index, bit) = (usize::from(event / 32), 1 << (event % 32));
        let is_set = |register| self.registers.get(register).unwrap_or(0) & bit != 0;
        let wakes = is_set(WAKE_MASK[index]) && is_set(WAKE_LVL[index]) == high;
        if wakes {
            let status = self.registers.get(WAKE_STATUS[index]).unwrap_or(0);
            self.registers.set(WAKE_STATUS[index], status | bit);
        }
        wakes
    }

    /// The SoC enters deep power down: each event marked in
    /// `AUTO_WAKE_LVL_MASK` takes as its wake level the opposite of what
    /// `pad` says its input is now (`None`: a pad not modeled, whose level
    /// is left as programmed). So such an event wakes the console when its
    /// input changes, whichever way.
    ///
    /// The public Tegra2 TRM (§9.1.15.55-56, `APBDEV_PMC_AUTO_WAKE_LVL_0`
    /// and `_MASK_0`, the same offsets as here) has the PMC sample the
    /// masked wake pads "just before entering DPD" and transfer the value
    /// to `WAKE_LVL`. \[inference\] Transferred inverted: a wake level
    /// equal to the input at entry would match it at once, as the level
    /// match in [`Pmc::wake`] is the wake.
    pub fn sample_wake_levels(&self, pad: impl Fn(u8) -> Option<bool>) {
        for index in 0..2 {
            let auto = self.registers.get(AUTO_WAKE_LVL_MASK[index]).unwrap_or(0);
            let mut level = self.registers.get(WAKE_LVL[index]).unwrap_or(0);
            for bit in (0..32).filter(|bit| auto & 1 << bit != 0) {
                if let Some(high) = pad(index as u8 * 32 + bit) {
                    level = level & !(1 << bit) | u32::from(!high) << bit;
                }
            }
            self.registers.set(WAKE_LVL[index], level);
        }
    }
}

impl Default for Pmc {
    fn default() -> Self {
        Self::new()
    }
}

impl MmioDevice for Pmc {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        let value = match word(offset, size)? {
            PWRGATE_STATUS => *self.powered.lock().unwrap(),
            // A toggle is accepted at once, so the start bit never reads set.
            PWRGATE_TOGGLE => self.registers.get(PWRGATE_TOGGLE)? & !TOGGLE_START,
            register => self.registers.get(register)?,
        };
        Some(value.into())
    }

    fn write(&self, _core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        let register = word(offset, size)?;
        let value = value as u32;
        match register {
            // Read-only.
            PWRGATE_STATUS | CLAMP_STATUS => return None,
            CNTRL if self.powered_on && value & CNTRL_MAIN_RST != 0 => return None,
            PWRGATE_TOGGLE if value & TOGGLE_START != 0 => {
                *self.powered.lock().unwrap() ^= 1 << (value & TOGGLE_PARTITION);
            }
            SET_SW_CLAMP if value & CLAMP_CRAIL != 0 => {
                let clamped = self.registers.get(CLAMP_STATUS).unwrap_or(0);
                self.registers.set(CLAMP_STATUS, clamped | CLAMP_CRAIL);
            }
            REMOVE_CLAMPING => {
                let clamped = self.registers.get(CLAMP_STATUS).unwrap_or(0);
                self.registers.set(CLAMP_STATUS, clamped & !value);
            }
            _ => {
                if let Some(&(_, status, pads)) =
                    IO_DPD.iter().find(|&&(request, ..)| request == register)
                {
                    let current = self.registers.get(status).unwrap_or(0);
                    let selected = value & pads;
                    let updated = match value >> IO_DPD_CODE_SHIFT {
                        IO_DPD_ON => current | selected,
                        IO_DPD_OFF => current & !selected,
                        _ => current,
                    };
                    self.registers.set(status, updated);
                }
            }
        }
        let value = if WAKE_STATUS.contains(&register) {
            self.registers.get(register)? & !value
        } else {
            value
        };
        self.registers.set(register, value);
        Some(())
    }

    fn snapshot(&self) -> Option<&dyn Snapshot> {
        Some(self)
    }
}

/// Saves the registers, wake status included, and which partitions are
/// powered.
impl Snapshot for Pmc {
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        encode(&(self.registers.save()?, *self.powered.lock().unwrap()))
    }

    fn restore(&self, state: &[u8]) -> Result<(), SnapshotError> {
        let (registers, powered): (Vec<u8>, u32) = decode("PMC", state)?;
        self.registers.restore(&registers)?;
        *self.powered.lock().unwrap() = powered;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_toggle_switches_one_partition_and_is_accepted_at_once() {
        let pmc = Pmc::new();
        let status = |pmc: &Pmc| pmc.read(0, PWRGATE_STATUS, 4).unwrap() as u32;
        assert_eq!(status(&pmc) & 1 << 3, 0);
        pmc.write(0, PWRGATE_TOGGLE, 4, u64::from(TOGGLE_START | 3))
            .unwrap();
        assert_eq!(pmc.read(0, PWRGATE_TOGGLE, 4), Some(3));
        assert_eq!(status(&pmc), POWERED_AT_ENTRY | 1 << 3);
        pmc.write(0, PWRGATE_TOGGLE, 4, u64::from(TOGGLE_START | 3))
            .unwrap();
        assert_eq!(status(&pmc), POWERED_AT_ENTRY);
        // Without the start bit nothing switches.
        pmc.write(0, PWRGATE_TOGGLE, 4, 3).unwrap();
        assert_eq!(status(&pmc), POWERED_AT_ENTRY);
    }

    #[test]
    fn power_on_has_every_partition_off_and_the_rest_unestablished() {
        let pmc = Pmc::power_on();
        assert_eq!(pmc.read(4, PWRGATE_STATUS, 4), Some(0));
        // The bootloader powers CRAIL, C0NC and CE0 in turn.
        for partition in [0, 15, 14] {
            pmc.write(4, PWRGATE_TOGGLE, 4, u64::from(TOGGLE_START | partition))
                .unwrap();
        }
        assert_eq!(
            pmc.read(4, PWRGATE_STATUS, 4),
            Some(u64::from(POWERED_AT_ENTRY))
        );
        // As the NX bootloader's check requires; the reset reason power-on.
        assert_eq!(pmc.read(4, WAKE_MASK[0], 4), Some(0));
        assert_eq!(pmc.read(4, GPU_RG_CNTRL, 4), Some(1));
        assert_eq!(pmc.read(4, 0x1B4, 4), Some(0));
        // Scratch the monitor and the NX bootloader read-modify-write.
        assert_eq!(pmc.read(4, SECURE_SCRATCH21, 4), Some(0));
        assert_eq!(pmc.read(4, 0x600, 4), Some(0));
        assert_eq!(pmc.read(4, 0xB04, 4), Some(0));
        assert_eq!(lp0_scratch_at_power_on().count(), 251);
        assert_eq!(pmc.read(4, 0x6F0, 4), None, "unestablished");
    }

    #[test]
    fn the_sc7_stub_clamps_the_cpu_rail_and_puts_the_pads_in_deep_power_down() {
        let pmc = Pmc::power_on();
        let read = |offset| pmc.read(4, offset, 4).unwrap() as u32;
        assert_eq!(read(CLAMP_STATUS), 0);
        pmc.write(4, SET_SW_CLAMP, 4, 1).unwrap();
        assert_eq!(read(CLAMP_STATUS), CLAMP_CRAIL);
        assert_eq!(pmc.write(4, CLAMP_STATUS, 4, 0), None, "read-only");
        // The warmboot firmware: SW clamp off, then the clamp removed.
        pmc.write(4, SET_SW_CLAMP, 4, 0).unwrap();
        assert_eq!(read(CLAMP_STATUS), CLAMP_CRAIL);
        pmc.write(4, REMOVE_CLAMPING, 4, 1).unwrap();
        assert_eq!(read(CLAMP_STATUS), 0);
        // What the stub requests and waits for.
        pmc.write(4, 0x45C, 4, 0x8FFF_FFFF).unwrap();
        assert_eq!(read(0x460), 0x0FFF_FFFF);
        pmc.write(4, 0x464, 4, 0x8FFF_FFFF).unwrap();
        assert_eq!(read(0x468), 0x0FFF_1FFF);
        pmc.write(4, 0x464, 4, 0x4000_00FF).unwrap();
        assert_eq!(read(0x468), 0x0FFF_1F00);
        // SC7 entry is storage the machine looks at; a SoC reset refused.
        pmc.write(4, DPD_ENABLE, 4, 3).unwrap();
        assert!(dpd_entered(read(DPD_ENABLE)));
        assert_eq!(pmc.write(4, CNTRL, 4, u64::from(CNTRL_MAIN_RST)), None);
        assert_eq!(
            Pmc::new().write(0, CNTRL, 4, 0x10),
            Some(()),
            "kernel entry"
        );
        assert_eq!(power_on_value(SECURE_LOCKS[7]), Some(0x4000));
    }

    #[test]
    fn an_enabled_wake_event_at_its_level_is_recorded_until_cleared() {
        let pmc = Pmc::new();
        let status = |pmc: &Pmc| pmc.read(0, WAKE_STATUS[1], 4).unwrap();
        assert!(!pmc.wake(WAKE_EVENT_PMIC, false), "not enabled");
        pmc.write(0, WAKE_MASK[1], 4, 1 << 19).unwrap();
        assert!(!pmc.wake(WAKE_EVENT_PMIC, true), "its wake level is low");
        assert_eq!(status(&pmc), 0);
        assert!(pmc.wake(WAKE_EVENT_PMIC, false));
        assert_eq!(status(&pmc), 1 << 19);
        pmc.write(0, WAKE_STATUS[1], 4, 0xFFFF_FFFF).unwrap();
        assert_eq!(status(&pmc), 0);
    }

    #[test]
    fn an_auto_event_wakes_on_a_change_of_its_pad_from_deep_power_down_entry() {
        let pmc = Pmc::new();
        let read = |offset| pmc.read(0, offset, 4).unwrap() as u32;
        // As the firmware leaves it: both rail CTS events enabled, level
        // "auto"; the radio's active high and not auto; PwrIntN low.
        pmc.write(0, WAKE_MASK[0], 4, 0x0800_45D0).unwrap();
        pmc.write(0, WAKE_LVL[0], 4, 0x100).unwrap();
        pmc.write(0, AUTO_WAKE_LVL_MASK[0], 4, 0x17FE_BCFF).unwrap();
        pmc.write(0, WAKE_MASK[1], 4, 0x0019_000C).unwrap();
        pmc.write(0, AUTO_WAKE_LVL_MASK[1], 4, 0xEFE7_E01A).unwrap();
        // The right rail's line is low (asserted), the left one's high; the
        // PMIC's interrupt is not modeled as an "auto" pad.
        let pads = |event| match event {
            WAKE_EVENT_UART2_CTS => Some(false),
            WAKE_EVENT_UART3_CTS => Some(true),
            WAKE_EVENT_PMIC => Some(true),
            _ => None,
        };
        pmc.sample_wake_levels(pads);
        assert_eq!(read(WAKE_LVL[0]), 0x100 | 1 << 6, "inverted, the rest kept");
        assert_eq!(read(WAKE_LVL[1]), 0, "PwrIntN keeps its level");
        // Unchanged pads do not wake; a change does, and is recorded.
        assert!(!pmc.wake(WAKE_EVENT_UART2_CTS, false));
        assert!(!pmc.wake(WAKE_EVENT_UART3_CTS, true));
        assert!(pmc.wake(WAKE_EVENT_UART2_CTS, true));
        assert_eq!(read(WAKE_STATUS[0]), 1 << 6);
        assert!(pmc.wake(WAKE_EVENT_UART3_CTS, false));
        assert_eq!(read(WAKE_STATUS[0]), 1 << 6 | 1 << 7);
    }
}
