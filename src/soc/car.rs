//! Clock and reset controller.
//!
//! Nothing here is clocked, so the block is mostly storage: the guest gates
//! clocks, picks sources and dividers, holds modules in reset and reads back
//! what it wrote. What the block holds before the guest writes it is the
//! state the boot chain hands the kernel, and that is what the firmware's
//! clock driver (`pcv`) reads to learn the clock tree when it starts
//! (`NvClockSocReloadSourceClocks`, pcv text `0x48290`), so registers start
//! from `ENTRY_STATE` and a register outside it refuses to be read until
//! the guest has written it.
//!
//! Behaviours beyond storage:
//!
//! * the set and clear registers of the enable and reset banks change bits
//!   of the register they shadow, and read as it;
//! * a PLL's lock bit is read-only and follows the PLL: it reads set while
//!   the PLL is enabled, powered (IDDQ clear) and out of reset (`PLLS`);
//! * the CPU cluster's burst policy and divider are one register each,
//!   reachable at two offsets (`ALIASES`);
//! * the oscillator frequency detector measures the 38.4 MHz crystal
//!   (`OSC_FREQ_DET`);
//! * writing `CLK_SOURCE_EMC` switches the memory controller's clock, which
//!   completes its clock change ([`Car::on_emc_clock_change`]);
//! * modules with state of their own learn when their reset is asserted
//!   or released ([`Car::on_reset_asserted`], [`Car::on_reset_released`]).
//!
//! # What `pcv` requires at start-up
//!
//! `pcv` initialises its clock driver in `sub_4A000` (pcv text): it sets the
//! USB2 tracking clock divider (`CLK_SOURCE_USB2_HSIC_TRK = 6`), then
//! `NvClockSocReloadSourceClocks` reads every PLL and the system and CPU
//! clock policies, then it reads each module's enable bit and, for enabled
//! modules and those without an enable bit, its clock source register
//! (`sub_49DB0`), and finally sets up the system bus and a few modules. The
//! reload aborts the process (`sub_3CB0`, reached through text `0x49ba0`)
//! unless:
//!
//! | register | requirement | text |
//! |---|---|---|
//! | `SPARE_REG0` | CLK_M divisor field (bits 3:2) is 1, CLK_M = OSC / 2 | `0x48310` |
//! | `OSC_CTRL` | PLL reference divider (bits 27:26) is 0 | `0x48354` |
//! | `PLLU_OUTA` | output 1 and 2 overrides (bits 2, 18) clear | `0x490a8`, `0x4910c` |
//! | `PLLP_BASE` | bit 0 clear | `0x493c0` |
//! | `PLLP_OUTB` | output 3 and 4 overrides (bits 2, 18) clear | `0x493fc`, `0x49434` |
//! | `PLLP_OUTC` | output 5 override (bit 18) clear | `0x49478` |
//!
//! Everything else it reads it only records: PLL dividers and enables
//! become the rates of its clock tree, the burst policies the CPU and
//! system bus rates. The oscillator rate itself is not read: `pcv` assumes
//! 38.4 MHz.

use super::regs::{RegisterFile, word};
use crate::bus::MmioDevice;

pub const BASE: u64 = 0x6000_6000;
pub const SIZE: u64 = 0x1000;

const PLLU_BASE: u64 = 0xC0;
const UTMIP_PLL_CFG0: u64 = 0x480;

/// The state the boot chain leaves the controller in when it starts the
/// kernel, as far as anything reads it.
///
/// The chain that runs before the kernel (boot ROM, Nintendo's first-stage
/// bootloader, the secure monitor) is not available here; its effect on
/// this block is known from public reimplementations of it: Atmosphère's
/// `fusee` "secure initialize" (`fusee_secure_initialize.cpp`, which
/// mirrors the first-stage bootloader), `fusee_cpu.cpp` and `exosphere`'s
/// `secmon_setup.cpp`, and hekate's `hw_init.c`, `sdram.c`, `clock.c` and
/// `sdram_config.inl`. Values from them are recalled from those sources,
/// not verified on hardware; where the firmware's own checks constrain a
/// value, the value meets them. Groups:
///
/// * [`OSCILLATOR_AND_BUSES`], [`RUNNING_PLLS`], [`CLOCK_ENABLES`],
///   [`RESETS`], [`CLOCK_GATE_OVERRIDES`], [`MODULE_SOURCES`],
///   [`PLL_CONTROLS`]: values with evidence;
/// * [`ASSUMED_ZERO`] and [`GATED_MODULE_SOURCES`]: registers whose entry
///   value nothing public pins down, which read zero.
const ENTRY_STATE: [&[(u64, u32)]; 8] = [
    &OSCILLATOR_AND_BUSES,
    &RUNNING_PLLS,
    &CLOCK_ENABLES,
    &RESETS,
    &CLOCK_GATE_OVERRIDES,
    &MODULE_SOURCES,
    &ASSUMED_ZERO,
    &PLL_CONTROLS,
];

/// Oscillator, CLK_M and the system and CPU clock policies.
///
/// | offset | register | value | source |
/// |---|---|---|---|
/// | `0x050` | `OSC_CTRL` | `0x5000_0071` | 38.4 MHz crystal, oscillator enabled, drive strength 7, reference divider 1: fusee `InitializeClock`, hekate `hw_init.c:75` |
/// | `0x55c` | `SPARE_REG0` | `0x4` | CLK_M divisor 2 (19.2 MHz): fusee `InitializeClock`, hekate `hw_init.c:72`; the other bits are assumed zero |
/// | `0x028` | `SCLK_BURST_POLICY` | `0x2000_3333` | run state, every wake-up source PLLP_OUT0 (408 MHz): fusee `SecureInitialize` end, hekate `hw_init.c:435` |
/// | `0x02c` | `SUPER_SCLK_DIVIDER` | `0x8000_0000` | divider enabled at 1/1: fusee `InitializeClock`, hekate `hw_init.c:89` |
/// | `0x030` | `CLK_SYSTEM_RATE` | `0x2` | AHB at SCLK, APB at SCLK / 3: fusee `InitializeClock`, hekate `hw_init.c:90` |
/// | `0x400` | `CLK_SOURCE_SYS` | `0x0` | SCLK divider 1: hekate `hw_init.c:87` |
/// | `0x368` | `CCLKG_BURST_POLICY` | `0x2000_8888` | run state, every source PLLX_OUT0_LJ: written through `CCLK_BURST_POLICY` by fusee `fusee_cpu.cpp:113`, hekate `ccplex.c:83` (see [`ALIASES`]) |
/// | `0x36c` | `SUPER_CCLKG_DIVIDER` | `0x8000_0000` | divider enabled at 1/1: fusee `fusee_cpu.cpp:120`, hekate `ccplex.c:84` |
///
/// `pcv` decodes the policies the same way: state 2 (run) selects the run
/// source field, SCLK source 3 is its `PllP0` and CPU source 8 its PLLX
/// (tables built in `NvClockSocReloadSourceClocks`).
const OSCILLATOR_AND_BUSES: [(u64, u32); 8] = [
    (0x050, 0x5000_0071),
    (0x55C, 0x0000_0004),
    (0x028, 0x2000_3333),
    (0x02C, 0x8000_0000),
    (0x030, 0x0000_0002),
    (0x400, 0x0000_0000),
    (0x368, 0x2000_8888),
    (0x36C, 0x8000_0000),
];

/// PLLs that run at entry, and the clocks taken from them.
///
/// Physically necessary: the system bus and peripherals run from PLLP, the
/// CPU from PLLX, the memory controller from PLLP through the EMC divider.
/// PLL bases are `bypass(31) enable(30) lock(27) DIVP DIVN DIVM`; the lock
/// bits here are what [`PLLS`] computes anyway.
///
/// | offset | register | value | source |
/// |---|---|---|---|
/// | `0x0a0` | `PLLP_BASE` | `0x4800_5508` | 408 MHz from 38.4 MHz: M 8, N 85, P 1, the reference table of Linux `clk-tegra210.c` (`pll_p_freq_table`); set up by the boot ROM. M even meets `pcv`'s check of bit 0 |
/// | `0x0a8` | `PLLP_OUTB` | `0x0003_0003` | outputs 3 and 4 enabled and out of reset, no overrides: hekate `hw_init.c:393` |
/// | `0x0e0` | `PLLX_BASE` | `0x4840_4e02` | M 2, N 78, P index 4 (/5): fusee `fusee_cpu.cpp:90-101`, which starts the CPU |
/// | `0x090` | `PLLM_BASE` | `0x4800_2201` | M 1, N 34, enabled by SDRAM bring-up: hekate `sdram.c:227-231` with the dividers of `sdram_config.inl:25-29` |
/// | `0x09c` | `PLLM_MISC2` | `0x0` | hekate `sdram.c:224` |
/// | `0x19c` | `CLK_SOURCE_EMC` | `0x4018_8002` | EMC from PLLP_OUT0 / 2 (204 MHz) at the end of SDRAM bring-up: hekate `sdram.c:240` with `emc_clock_source` and `mc_emem_arb_misc0` of `sdram_config.inl` |
/// | `0x664` | `CLK_SOURCE_EMC_DLL` | `0x4000_0000` | hekate `sdram.c:242`, `emc_clock_source_dll` |
/// | `0x0c0` | `PLLU_BASE` | `0x4801_1902` | see below |
/// | `0x480` | `UTMIP_PLL_CFG0` | `0x0019_0100` | see below |
/// | `0x484` | `UTMIP_PLL_CFG1` | `0x0000_8177` | power-up forced (bit 15), power-down not forced (bit 14), crystal frequency count 375: hekate `clock.c:656` (bit names from Linux `clk-tegra210.c`); the bits hekate preserves are assumed zero |
///
/// The USB PLLs are what the `boot` process insists on before anything
/// else: it restarts the console unless `PLLU_BASE[20:0]` is `0x11902` and
/// `UTMIP_PLL_CFG0[23:8]` is `0x1901` (`boot` text `0x1cb0`). Both enabled
/// and locked; `pcv` takes the UTMI PLL as running because
/// `UTMIP_PLL_CFG1` bit 14 (force power-down) is clear.
const RUNNING_PLLS: [(u64, u32); 10] = [
    (0x0A0, 0x4800_5508),
    (0x0A8, 0x0003_0003),
    (0x0E0, 0x4840_4E02),
    (0x090, 0x4800_2201),
    (0x09C, 0x0000_0000),
    (0x19C, 0x4018_8002),
    (0x664, 0x4000_0000),
    (PLLU_BASE, 0x4801_1902),
    (UTMIP_PLL_CFG0, 0x0019_0100),
    (0x484, 0x0000_8177),
];

/// `CLK_OUT_ENB_L/H/U/V/W/X/Y`: which module clocks run at entry.
///
/// The first-stage bootloader writes all seven registers whole at the end
/// of its memory self-test workaround (fusee `DoMbistWorkaround`, hekate
/// `_mbist_workaround_bl`), which fixes every bit; later stages only turn
/// single clocks on (or on and off again), so the result is that write plus
/// these enables (bit names as in hekate `clock.h`):
///
/// * L `0x8000_1130`: RTC, TMR, GPIO, CACHE2 (workaround); I2C1 (fusee
///   `SecureInitialize`).
/// * H `0x0200_80c1`: PMC, FUSE (workaround); MEM, EMC (fusee `EnableArc`);
///   I2C5 (fusee `SecureInitialize`).
/// * U `0x01f0_0200`: CSITE, IRAMA-D, BPMP cache RAM (workaround, and fusee
///   `EnableCsiteClock`/`EnableCram2Clock`).
/// * V `0xc0c0_0809`: MSELECT, APB2APE, SPDIF doubler, SE (workaround);
///   TZRAM (fusee `SecureInitialize`); CPUG (fusee `fusee_cpu.cpp:130`);
///   ACTMON (exosphere `secmon_setup.cpp:1208`).
/// * W `0x4820_00fc`: PCIERX0-5, ENTROPY, MC1 (workaround); DVFS (fusee
///   `EnableCldvfsClock`).
/// * X `0x2300_4780`: MC_CAPA/CBPA/CPU/BBC, GPU, DBGAPB, PLLG_REF
///   (workaround); EMC_DLL (fusee `EnableArc`, hekate `sdram.c:248`).
/// * Y `0x0000_0300`: MC_CDPA, MC_CCPA (workaround).
///
/// Clocks a stage enables only for its own use and disables again (the
/// eMMC controller the bootloader reads the firmware with, the TSEC and
/// display blocks of the first stage) end gated.
const CLOCK_ENABLES: [(u64, u32); 7] = [
    (0x010, 0x8000_1130),
    (0x014, 0x0200_80C1),
    (0x018, 0x01F0_0200),
    (0x360, 0xC0C0_0809),
    (0x364, 0x4820_00FC),
    (0x280, 0x2300_4780),
    (0x298, 0x0000_0300),
];

/// `RST_DEVICES_L/H/U/V/W/X/Y`: which modules are held in reset at entry.
///
/// A module is held in reset exactly when its clock is gated in
/// [`CLOCK_ENABLES`]. That is how the boot chain leaves every module it
/// touches: its clock helpers hold reset whenever they gate a clock and
/// release it whenever they enable one (exosphere `clkrst_api.cpp`
/// `EnableClock`/`DisableClock`), the memory self-test workaround puts the
/// modules it briefly ran back into reset (APE, VIC, HOST1X, DISP1), and
/// fusee `SecureInitialize` ends by holding USBD in reset. For modules no
/// stage touches it is an assumption about the power-on state. Bits that
/// have no reset function (hekate `clock.h`: "only enable", reserved,
/// deprecated and spare bits) read zero; of the modules that have a reset
/// but no clock enable, the BPMP and system bus are out of reset (they run)
/// and the rest (PCIe clock, XUSB pad control, the USB PHYs, ADSP
/// sub-blocks) are assumed held in reset.
const RESETS: [(u64, u32); 7] = [
    (0x004, 0x1CD2_C2C8),
    (0x008, 0x85D1_5326),
    (0x00C, 0x820E_557A),
    (0x358, 0x3F01_8080),
    (0x35C, 0x1000_4103),
    (0x28C, 0x00E4_2048),
    (0x2A4, 0x0FE9_F0DC),
];

/// `LVL2_CLK_GATE_OVRA/B/C/D/E`: second-level clock gating left automatic.
/// The first-stage bootloader zeroes all five (fusee `DoMbistWorkaround`,
/// hekate `hw_init.c:257-261`); the one override set later, ARC_CLK_OVR_ON
/// in OVRD (fusee `EnableArc`), the secure monitor clears again (exosphere
/// `secmon_setup.cpp:812`).
const CLOCK_GATE_OVERRIDES: [(u64, u32); 5] =
    [(0x0F8, 0), (0x0FC, 0), (0x3A0, 0), (0x3A4, 0), (0x554, 0)];

/// Clock sources of modules whose clock runs at entry, as the stage that
/// enabled them programmed them (`source << 29 | divider`).
///
/// | offset | module | value | source |
/// |---|---|---|---|
/// | `0x124` | I2C1 | `0xc000_0000` | CLK_M, divider 0: exosphere `clkrst_api.cpp` `I2c1Clock` |
/// | `0x128` | I2C5 | `0xc000_0000` | CLK_M, divider 0: exosphere `I2c5Clock` |
/// | `0x1d4` | CSITE | `0x4` | PLLP_OUT0, divider 4: exosphere `CsiteClock` |
/// | `0x3b4` | MSELECT | `0x6` | PLLP_OUT0, divider 6: fusee `fusee_cpu.cpp:106`, hekate `ccplex.c:79` |
/// | `0x3e8` | ACTMON | `0xc000_0000` | CLK_M, divider 0: exosphere `ActmonClock` |
/// | `0x42c` | SE | `0x0` | PLLP_OUT0, divider 0: exosphere `SeClock` |
/// | `0x180` | HOST1X | `0x8000_0000` | PLLP_OUT0 (source 4), written by the memory self-test workaround: hekate `hw_init.c:267`, fusee `DoMbistWorkaround`; the bits it preserves are assumed zero |
/// | `0x6a0` | NVENC | `0x8000_0000` | likewise, hekate `hw_init.c:268` |
const MODULE_SOURCES: [(u64, u32); 8] = [
    (0x124, 0xC000_0000),
    (0x128, 0xC000_0000),
    (0x1D4, 0x0000_0004),
    (0x3B4, 0x0000_0006),
    (0x3E8, 0xC000_0000),
    (0x42C, 0x0000_0000),
    (0x180, 0x8000_0000),
    (0x6A0, 0x8000_0000),
];

/// Registers read before anything writes them whose entry value nothing
/// public pins down: they read zero, which is an assumption.
///
/// * PLLs the boot chain does not use, disabled with zero dividers (`pcv`
///   records a disabled PLL as stopped whatever its dividers):
///   `PLLC_BASE 0x080`, `PLLC_OUT 0x084`, `PLLC2_BASE 0x4e8`,
///   `PLLC3_BASE 0x4fc`, `PLLMB_BASE 0x5e8` (the bootloader only clears its
///   enable, hekate `hw_init.c:83`), `PLLD2_BASE 0x4b8`,
///   `PLLRE_BASE 0x4c4`, `PLLRE_OUT1 0x4cc`, `PLLE_BASE 0x0e8`,
///   `PLLDP_BASE 0x590`, `PLLC4_BASE 0x5a4` (the file system process
///   checks it and sets it up itself), `PLLC4_OUT 0x5e4`, `PLLA_BASE 0x0b0`,
///   `PLLA_OUT 0x0b4`, `PLLA1_BASE 0x6a4`, and the audio DSP cluster's
///   `ACLK_BURST_POLICY 0x6e0` and `SUPER_ACLK_DIVIDER 0x6e4`.
/// * Outputs and controls of running PLLs nobody documents:
///   `PLLU_OUTA 0x0c4` and `PLLP_OUTC 0x67c` (outputs off; `pcv` requires
///   their override bits clear), `PLLU_MISC1 0x0c8`, `PLLP_MISC0 0x0ac`
///   (IDDQ clear: PLLP runs), `PLLP_MISC1 0x680`.
/// * Clock sources `pcv` reads unconditionally, of modules without an
///   enable bit (XUSB core host/falcon/FS/core device/SS
///   `0x600-0x610`, DVFS reference and SoC `0x62c`, `0x630`, SPDIF input
///   `0x10c`), and of ENTROPY `0x628`, whose clock runs; and the PWM source
///   `0x110`, which `boot` changes by read-modify-write. `pcv` reprograms a
///   module's source before relying on its rate.
/// * PLLE's controls, which `pcv` changes by read-modify-write when the
///   PCIe driver first asks for its clock (`PLLEConfigSwCtrl`, text
///   `0x4cc80`): `PLLE_MISC 0x0ec`, `PLLE_SS_CNTL 0x068` and `PLLE_AUX
///   0x48c`, whose bit 24 (hardware sequencing, Linux `PLLE_AUX_SEQ_ENABLE`)
///   `pcv` requires clear.
/// * `XUSBIO_PLL_CFG0 0x51c`, the USB 3 / PCIe pad PLL's control, which no
///   stage of the boot chain touches: zero is software control with the PLL
///   neither enabled nor in IDDQ. `pcv` reads it whenever a module asks for
///   that PLL's state and enables the PLL by read-modify-write the way Linux
///   does (`NvQryClockXusbIoPllEnable`, text `0x4eaa0`: clear bits 0 and 2,
///   set 6 and 13; bit names `XUSBIO_PLL_CFG0_*` in Linux `clk-tegra210.c`).
const ASSUMED_ZERO: [(u64, u32); 36] = [
    (0x51C, 0),
    (0x0EC, 0),
    (0x068, 0),
    (0x48C, 0),
    (0x080, 0),
    (0x084, 0),
    (0x4E8, 0),
    (0x4FC, 0),
    (0x5E8, 0),
    (0x4B8, 0),
    (0x4C4, 0),
    (0x4CC, 0),
    (0x0E8, 0),
    (0x590, 0),
    (0x5A4, 0),
    (0x5E4, 0),
    (0x0B0, 0),
    (0x0B4, 0),
    (0x6A4, 0),
    (0x6E0, 0),
    (0x6E4, 0),
    (0x0C4, 0),
    (0x67C, 0),
    (0x0C8, 0),
    (0x0AC, 0),
    (0x680, 0),
    (0x600, 0),
    (0x604, 0),
    (0x608, 0),
    (0x60C, 0),
    (0x610, 0),
    (0x62C, 0),
    (0x630, 0),
    (0x10C, 0),
    (0x628, 0),
    (0x110, 0),
];

/// PLL controls the clock driver changes by read-modify-write before
/// anything else writes them (it sets up the unused PLLs the way Linux
/// does: reset and power-down bits, then lock settings).
///
/// | offset | register | value | source |
/// |---|---|---|---|
/// | `0x4ec`, `0x500`, `0x6a8` | `PLLC2/PLLC3/PLLA1_MISC0` | `0x4008_0000` | Linux `clk-tegra210.c` `PLLCX_MISC0_DEFAULT_VALUE` (PLL held in reset) |
/// | `0x4f0`, `0x504`, `0x6ac` | `PLLC2/PLLC3/PLLA1_MISC1` | `0x0800_0000` | `PLLCX_MISC1_DEFAULT_VALUE` (IDDQ: powered down) |
/// | `0x4c8` | `PLLRE_MISC0` | `0x4100_0000` | `PLLRE_MISC0_DEFAULT_VALUE` (IDDQ, lock enable) |
/// | `0x570` | `PLLD2_MISC1` | `0x1000_0000` | `PLLD2_MISC1_CFG_DEFAULT_VALUE` |
/// | `0x578` | `PLLD2_MISC3` | `0x0` | `PLLD2_MISC3_CTRL2_DEFAULT_VALUE` |
/// | `0x598` | `PLLDP_SS_CFG` | `0xc000_0000` | `PLLDP_MISC1_CFG_DEFAULT_VALUE` |
/// | `0x594`, `0x59c`, `0x5a0` | `PLLDP_MISC`, `PLLDP_SS_CTRL1/2` | `0x4000_0020`, `0xf400_f0da`, `0x2004_f400` | `PLLDP_MISC0`, `PLLDP_MISC2_CTRL1`, `PLLDP_MISC3_CTRL2_DEFAULT_VALUE` |
/// | `0x4bc`, `0x574` | `PLLD2_MISC0`, `PLLD2_MISC2` | `0x4000_0020`, `0x0` | `PLLD2_MISC0`, `PLLD2_MISC2_CTRL1_DEFAULT_VALUE` |
/// | `0x5d8` | `PLLA_MISC2` | `0x0` | `PLLA_MISC2_DEFAULT_VALUE` |
/// | `0x518` | `PLLX_MISC3` | `0x0` | PLLX runs, so its IDDQ bit (3) is clear: fusee `fusee_cpu.cpp:84` clears it before enabling PLLX; the other bits as `PLLX_MISC3_DEFAULT_VALUE` |
/// | `0x548` | `PLLX_HW_CTRL_CFG` | `0x1` | software control of PLLX (`SWCTRL`, Linux name): `pcv` aborts unless bit 0 is set (`sub_4ff80`, `sub_50010`); no stage of the boot chain writes the register, so this is its power-on value; other bits assumed zero |
/// | `0x488` | `UTMIP_PLL_CFG2` | `0x0060_0000` | delay count 24 for a 38.4 MHz crystal: hekate `clock.c:655`; the bits hekate preserves are assumed zero |
/// | `0x52c` | `UTMIPLL_HW_PWRDN_CFG0` | `0x8000_0000` | lock (bit 31, hekate `clock.h` `UTMIPLL_LOCK`) of the running UTMI PLL; hardware power-down control off (bit 24 clear), which is what makes `pcv` hand the PLL to hardware control (`sub_4c980`); other bits assumed zero |
///
/// Linux's `*_DEFAULT_VALUE` constants are the values NVIDIA's driver
/// programs and checks unused PLLs against; that a PLL nobody used still
/// holds them at kernel entry is an assumption (they are what `pcv` itself
/// writes back, text `sub_4ad90` family).
const PLL_CONTROLS: [(u64, u32); 20] = [
    (0x594, 0x4000_0020),
    (0x59C, 0xF400_F0DA),
    (0x5A0, 0x2004_F400),
    (0x4BC, 0x4000_0020),
    (0x574, 0x0000_0000),
    (0x4EC, 0x4008_0000),
    (0x500, 0x4008_0000),
    (0x6A8, 0x4008_0000),
    (0x4F0, 0x0800_0000),
    (0x504, 0x0800_0000),
    (0x6AC, 0x0800_0000),
    (0x4C8, 0x4100_0000),
    (0x570, 0x1000_0000),
    (0x578, 0x0000_0000),
    (0x598, 0xC000_0000),
    (0x5D8, 0x0000_0000),
    (0x518, 0x0000_0000),
    (0x548, 0x0000_0001),
    (0x488, 0x0060_0000),
    (0x52C, 0x8000_0000),
];

/// Clock sources of modules gated at entry that no stage of the boot chain
/// programs, which `pcv` reads when it first sets their rate (it only
/// changes a source while the clock runs, and then by read-modify-write:
/// `sub_3f210`, `sub_3f560`). Their power-on value is not public here;
/// they read zero (source 0, divider 0), an assumption that only reaches
/// bits outside the mux and divider fields, which `pcv` overwrites.
/// Also `PLLA_MISC1 0x0b8` and `XUSB_PLL_CFG0 0x534`, which it changes by
/// read-modify-write in the same way.
const GATED_MODULE_SOURCES: [u64; 41] = [
    0x100, 0x104, 0x108, 0x118, 0x11C, 0x134, 0x13C, 0x150, 0x154, 0x178, 0x17C, 0x198, 0x1A0,
    0x1B4, 0x1B8, 0x1BC, 0x1C0, 0x1CC, 0x1D8, 0x3BC, 0x3C0, 0x3C4, 0x3D0, 0x3E4, 0x3EC, 0x3FC,
    0x410, 0x428, 0x644, 0x65C, 0x678, 0x698, 0x69C, 0x6C0, 0x6C4, 0x6D4, 0x6D8, 0x710, 0x0B8,
    0x534, 0x1F4,
];

/// `(set register, register it sets bits in)`; the matching clear register
/// follows each set register.
const SET_CLEAR_PAIRS: [(u64, u64); 14] = [
    (0x300, 0x004), // RST_DEV_L
    (0x308, 0x008), // RST_DEV_H
    (0x310, 0x00C), // RST_DEV_U
    (0x320, 0x010), // CLK_ENB_L
    (0x328, 0x014), // CLK_ENB_H
    (0x330, 0x018), // CLK_ENB_U
    (0x284, 0x280), // CLK_ENB_X
    (0x290, 0x28C), // RST_DEV_X
    (0x29C, 0x298), // CLK_ENB_Y
    (0x2A8, 0x2A4), // RST_DEV_Y
    (0x430, 0x358), // RST_DEV_V
    (0x438, 0x35C), // RST_DEV_W
    (0x440, 0x360), // CLK_ENB_V
    (0x448, 0x364), // CLK_ENB_W
];

/// `(alias, register)`: offsets that reach the same register.
///
/// The boot chain programs the CPU's clock through `CCLK_BURST_POLICY` and
/// `SUPER_CCLK_DIVIDER` (`0x20`, `0x24`), `pcv` reads and changes it through
/// the G cluster's `CCLKG_BURST_POLICY` and `SUPER_CCLKG_DIVIDER` (`0x368`,
/// `0x36c`, as in Linux `clk-tegra210.c`). The CPU runs on the G cluster
/// from what the boot chain wrote, so both pairs describe the same clock
/// mux; that they are one register is an inference.
const ALIASES: [(u64, u64); 2] = [(0x020, 0x368), (0x024, 0x36C)];

/// A PLL whose lock bit the block computes.
struct Pll {
    base: u64,
    enable: u32,
    /// Register and bit that report lock.
    lock: (u64, u32),
    /// Bits that power the PLL down (IDDQ) or hold it in reset.
    stop: &'static [(u64, u32)],
}

/// The PLLs and where their lock, power-down and reset bits are.
///
/// The lock bits are those `pcv`'s PLL descriptors poll (table at text
/// `0xb5810`, 184 bytes per PLL: PLLC, PLLC2, PLLC3 and PLLA1 lock in bit
/// 26, the others in bit 27); PLLE's (MISC bit 11) and PLLRE's (MISC bit
/// 27), and the IDDQ and reset bits, are those of Linux `clk-tegra210.c`.
/// A real PLL takes up to a millisecond to lock; here it is locked as soon
/// as it runs, and the lock-detector enable bits some PLLs have are not
/// modelled (the detector is taken to always report). A stop bit in a
/// register without a value does not count as set.
const PLLS: [Pll; 16] = [
    pll(0x0E0, (0x0E0, 27), &[(0x518, 3)]), // PLLX: MISC3 IDDQ
    pll(0x080, (0x080, 26), &[(0x08C, 27), (0x088, 30)]), // PLLC: MISC1 IDDQ, MISC0 reset
    pll(0x4E8, (0x4E8, 26), &[(0x4F0, 27), (0x4EC, 30)]), // PLLC2
    pll(0x4FC, (0x4FC, 26), &[(0x504, 27), (0x500, 30)]), // PLLC3
    pll(0x6A4, (0x6A4, 26), &[(0x6AC, 27), (0x6A8, 30)]), // PLLA1
    pll(0x0B0, (0x0B0, 27), &[(0x0B0, 25)]), // PLLA: base IDDQ
    pll(0x0D0, (0x0D0, 27), &[(0x0DC, 20)]), // PLLD: MISC0 IDDQ
    pll(0x4B8, (0x4B8, 27), &[(0x4B8, 18)]), // PLLD2: base IDDQ
    pll(0x590, (0x590, 27), &[(0x590, 18)]), // PLLDP: base IDDQ
    pll(0x5A4, (0x5A4, 27), &[(0x5A4, 18)]), // PLLC4: base IDDQ
    pll(0x090, (0x090, 27), &[(0x09C, 5)]), // PLLM: MISC2 IDDQ
    pll(0x5E8, (0x5E8, 27), &[(0x5EC, 17)]), // PLLMB: MISC1 IDDQ
    pll(0x0A0, (0x0A0, 27), &[(0x0AC, 3)]), // PLLP: MISC0 IDDQ
    pll(0x4C4, (0x4C8, 27), &[(0x4C8, 24)]), // PLLRE: MISC0 lock, IDDQ
    pll(0x0C0, (0x0C0, 27), &[(0x0CC, 31)]), // PLLU: MISC0 IDDQ
    Pll {
        base: 0x0E8,
        enable: 1 << 31,
        lock: (0x0EC, 1 << 11),
        stop: &[],
    }, // PLLE: enable in bit 31, lock in MISC0
];

const fn pll(base: u64, lock: (u64, u32), stop: &'static [(u64, u32)]) -> Pll {
    Pll {
        base,
        enable: 1 << 30,
        lock: (lock.0, 1 << lock.1),
        stop,
    }
}

pub struct Car {
    registers: RegisterFile,
    /// Told when the guest switches the memory controller's clock.
    emc_clock_change: Option<Box<dyn Fn() + Send + Sync>>,
    /// `(reset register, bit, hook)`, see [`Car::on_reset_asserted`].
    reset_hooks: Vec<(u64, u32, ResetHook)>,
    /// `(reset register, bit, hook)`, see [`Car::on_reset_released`].
    release_hooks: Vec<(u64, u32, ResetHook)>,
}

type ResetHook = Box<dyn Fn() + Send + Sync>;

const CLK_SOURCE_EMC: u64 = 0x19C;

/// The oscillator frequency detector: a write with the trigger bit counts
/// oscillator cycles over `REF_CLK_WIN_CFG + 1` periods of the 32.768 kHz
/// clock (CLK_S) and leaves the count in the status register's low 16 bits
/// with the busy bit (31) clear (layout as hekate `clock.h`
/// `OSC_FREQ_DET_*`, use as hekate `clock_get_osc_freq`). The count takes
/// tens of microseconds on hardware; here it is ready at once. `pcv`
/// triggers two windows and requires 2338-2348, i.e. a 38.4 MHz crystal
/// (`NvClockDetectOSCClock`, text `0x47ee0`); the status reads nothing
/// before the first measurement.
const OSC_FREQ_DET: u64 = 0x58;
const OSC_FREQ_DET_STATUS: u64 = 0x5C;
const OSC_FREQ_DET_TRIGGER: u32 = 1 << 31;
/// The crystal (see `OSC_CTRL` in [`OSCILLATOR_AND_BUSES`]) and CLK_S, the
/// 32.768 kHz clock the PMIC supplies (hekate `hw_init.c` header comment).
const OSCILLATOR_HZ: u64 = 38_400_000;
const CLK_S_HZ: u64 = 32_768;

impl Car {
    pub fn new() -> Self {
        Self {
            registers: RegisterFile::strict(
                ENTRY_STATE
                    .into_iter()
                    .flatten()
                    .copied()
                    .chain(GATED_MODULE_SOURCES.map(|register| (register, 0))),
            ),
            emc_clock_change: None,
            reset_hooks: Vec::new(),
            release_hooks: Vec::new(),
        }
    }

    /// Has `clock_change` called whenever the guest writes
    /// `CLK_SOURCE_EMC`, which is what makes the memory controller switch
    /// to the new clock.
    pub fn on_emc_clock_change(mut self, clock_change: Box<dyn Fn() + Send + Sync>) -> Self {
        self.emc_clock_change = Some(clock_change);
        self
    }

    /// Has `hook` called whenever bit `bit` of the reset register
    /// `register` (one of `RST_DEVICES_*`) goes from released to held,
    /// whether written directly or through its set register: a module's
    /// model that has state the reset clears registers here.
    pub fn on_reset_asserted(
        mut self,
        register: u64,
        bit: u32,
        hook: Box<dyn Fn() + Send + Sync>,
    ) -> Self {
        self.reset_hooks.push((register, bit, hook));
        self
    }

    /// Has `hook` called whenever bit `bit` of the reset register
    /// `register` goes from held to released, whether written directly or
    /// through its clear register: a module's model that starts running
    /// when let out of reset registers here.
    pub fn on_reset_released(
        mut self,
        register: u64,
        bit: u32,
        hook: Box<dyn Fn() + Send + Sync>,
    ) -> Self {
        self.release_hooks.push((register, bit, hook));
        self
    }

    /// Whether `pll` runs and so reports lock.
    fn locked(&self, pll: &Pll) -> bool {
        let set = |(register, mask): (u64, u32)| {
            self.registers
                .get(register)
                .is_some_and(|value| value & mask != 0)
        };
        set((pll.base, pll.enable))
            && !pll
                .stop
                .iter()
                .any(|&(register, bit)| set((register, 1 << bit)))
    }
}

impl Default for Car {
    fn default() -> Self {
        Self::new()
    }
}

/// The register a set or clear register acts on, and whether it sets.
fn set_clear_target(register: u64) -> Option<(u64, bool)> {
    SET_CLEAR_PAIRS
        .iter()
        .find_map(|&(set, target)| match register.checked_sub(set)? {
            0 => Some((target, true)),
            4 => Some((target, false)),
            _ => None,
        })
}

/// The register an offset reaches.
fn canonical(register: u64) -> u64 {
    ALIASES
        .iter()
        .find_map(|&(alias, target)| (alias == register).then_some(target))
        .unwrap_or(register)
}

impl MmioDevice for Car {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        let register = canonical(word(offset, size)?);
        // Set and clear registers read as the register they act on.
        let register = set_clear_target(register).map_or(register, |(target, _)| target);
        let mut value = self.registers.get(register)?;
        for pll in PLLS.iter().filter(|pll| pll.lock.0 == register) {
            let mask = pll.lock.1;
            value = if self.locked(pll) {
                value | mask
            } else {
                value & !mask
            };
        }
        Some(u64::from(value))
    }

    fn write(&self, _core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        let register = canonical(word(offset, size)?);
        let value = value as u32;
        let target = set_clear_target(register).map_or(register, |(target, _)| target);
        let before = self.registers.get(target);
        match set_clear_target(register) {
            Some((target, true)) => self.registers.update(target, |bits| bits | value)?,
            Some((target, false)) => self.registers.update(target, |bits| bits & !value)?,
            None => self.registers.set(register, value),
        }
        let after = self.registers.get(target).unwrap_or(0);
        let rose = after & !before.unwrap_or(0);
        for (hook_register, bit, hook) in &self.reset_hooks {
            if *hook_register == target && rose & 1 << bit != 0 {
                hook();
            }
        }
        let fell = before.unwrap_or(0) & !after;
        for (hook_register, bit, hook) in &self.release_hooks {
            if *hook_register == target && fell & 1 << bit != 0 {
                hook();
            }
        }
        if register == OSC_FREQ_DET && value & OSC_FREQ_DET_TRIGGER != 0 {
            let windows = u64::from(value & 0xF) + 1;
            let count = OSCILLATOR_HZ * windows / CLK_S_HZ;
            self.registers.set(OSC_FREQ_DET_STATUS, count as u32);
        }
        if register == CLK_SOURCE_EMC
            && let Some(clock_change) = &self.emc_clock_change
        {
            clock_change();
        }
        Some(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_and_clear_registers_act_on_the_register_they_shadow() {
        let car = Car::new();
        car.write(0, 0x014, 4, 0x0000_8000).unwrap();
        car.write(0, 0x328, 4, 0x0000_0041).unwrap();
        assert_eq!(car.read(0, 0x014, 4), Some(0x0000_8041));
        car.write(0, 0x32C, 4, 0x0000_8001).unwrap();
        assert_eq!(car.read(0, 0x014, 4), Some(0x0000_0040));
        assert_eq!(car.read(0, 0x328, 4), Some(0x0000_0040));
        // A neighbouring plain register is not mistaken for half of a pair.
        car.write(0, 0x318, 4, 7).unwrap();
        assert_eq!(car.read(0, 0x318, 4), Some(7));
        assert_eq!(car.read(0, 0x00C, 4), Some(0x820E_557A));
    }

    #[test]
    fn usb_pll_passes_the_boot_process_check() {
        let car = Car::new();
        let pllu = car.read(0, PLLU_BASE, 4).unwrap();
        let utmip = car.read(0, UTMIP_PLL_CFG0, 4).unwrap();
        assert_eq!(pllu & 0x1F_FFFF, 0x11902);
        assert_eq!(utmip & 0xFF_FF00, 0x19_0100);
        assert_ne!(pllu & 1 << 27, 0);
    }

    #[test]
    fn a_pll_locks_once_enabled_powered_and_out_of_reset() {
        let car = Car::new();
        let lock = |car: &Car| car.read(0, 0x080, 4).unwrap() & 1 << 26 != 0;
        car.write(0, 0x088, 4, 1 << 30).unwrap(); // PLLC held in reset
        car.write(0, 0x08C, 4, 1 << 27).unwrap(); // and powered down
        car.write(0, 0x080, 4, 0x4000_1F04 | 1 << 26).unwrap();
        assert!(!lock(&car), "the lock bit is not writable");
        car.write(0, 0x08C, 4, 0).unwrap();
        assert!(!lock(&car));
        car.write(0, 0x088, 4, 0).unwrap();
        assert!(lock(&car));
        car.write(0, 0x080, 4, 0x0000_1F04).unwrap();
        assert!(!lock(&car));
    }

    #[test]
    fn unknown_registers_refuse_reads_until_written() {
        let car = Car::new();
        assert_eq!(car.read(0, 0x5F0, 4), None);
        car.write(0, 0x5F0, 4, 8).unwrap();
        assert_eq!(car.read(0, 0x5F0, 4), Some(8));
    }

    #[test]
    fn the_frequency_detector_measures_the_crystal() {
        let car = Car::new();
        assert_eq!(car.read(0, OSC_FREQ_DET_STATUS, 4), None);
        car.write(0, OSC_FREQ_DET, 4, 0x8000_0001).unwrap();
        assert_eq!(car.read(0, OSC_FREQ_DET_STATUS, 4), Some(2343));
    }

    #[test]
    fn hooks_hear_reset_assertion_and_clock_changes() {
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };
        let resets = Arc::new(AtomicUsize::new(0));
        let changes = Arc::new(AtomicUsize::new(0));
        let (r, c) = (Arc::clone(&resets), Arc::clone(&changes));
        let car = Car::new()
            .on_reset_asserted(
                0x004,
                27,
                Box::new(move || _ = r.fetch_add(1, Ordering::Relaxed)),
            )
            .on_emc_clock_change(Box::new(move || _ = c.fetch_add(1, Ordering::Relaxed)));
        // DISP1 is held in reset at entry: holding it again is no edge.
        car.write(0, 0x300, 4, 1 << 27).unwrap();
        car.write(0, 0x304, 4, 1 << 27).unwrap();
        car.write(0, 0x300, 4, 1 << 27).unwrap();
        car.write(0, 0x004, 4, 0).unwrap();
        car.write(0, 0x004, 4, 1 << 27).unwrap();
        assert_eq!(resets.load(Ordering::Relaxed), 2);
        car.write(0, CLK_SOURCE_EMC, 4, 0x4018_8002).unwrap();
        assert_eq!(changes.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn hooks_hear_reset_release() {
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };
        let releases = Arc::new(AtomicUsize::new(0));
        let r = Arc::clone(&releases);
        let car = Car::new().on_reset_released(
            0x004,
            27,
            Box::new(move || _ = r.fetch_add(1, Ordering::Relaxed)),
        );
        // DISP1 is held in reset at entry: its clear register releases it,
        // releasing it again is no edge, and holding it is none either.
        car.write(0, 0x304, 4, 1 << 27).unwrap();
        car.write(0, 0x304, 4, 1 << 27).unwrap();
        car.write(0, 0x300, 4, 1 << 27).unwrap();
        assert_eq!(releases.load(Ordering::Relaxed), 1);
        car.write(0, 0x004, 4, 0).unwrap();
        assert_eq!(releases.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn the_cpu_burst_policy_is_reachable_at_both_offsets() {
        let car = Car::new();
        assert_eq!(car.read(0, 0x020, 4), Some(0x2000_8888));
        car.write(0, 0x368, 4, 0x2000_2222).unwrap();
        assert_eq!(car.read(0, 0x020, 4), Some(0x2000_2222));
    }
}
