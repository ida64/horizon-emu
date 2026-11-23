//! Tegra X1 blocks outside the CPU cluster, and the board devices wired to
//! them, as far as the kernel and its initial processes use them.

pub mod adsp;
pub mod agic;
pub mod ahb;
pub mod ahub;
pub mod alc5639;
pub mod apb_dma;
pub mod apb_misc;
pub mod apb_security;
pub mod bcm4356;
pub mod bh1730;
pub mod bluetooth_uart;
pub mod bpmp_cache;
pub mod bq24193;
pub mod car;
pub mod cldvfs;
pub mod dc;
pub mod dot11;
pub mod dpaux;
pub mod dsi;
pub mod emc;
pub mod emmc;
pub mod evp;
pub mod falcon_host;
pub mod flow;
pub mod fuse;
pub mod gm20b;
pub mod gmix;
pub mod gpio;
pub mod hda;
pub mod host1x;
pub mod host1x_channel;
#[cfg(test)]
mod hwcheck;
pub mod i2c;
pub mod ictlr;
pub mod iram;
pub mod irom;
pub mod joycon;
pub mod lsm6ds3;
pub mod m92t36;
pub mod max17050;
pub mod max77620;
pub mod max77621;
pub mod mc;
pub mod mc_channel;
pub mod medium;
pub mod mipi_cal;
pub mod mselect;
pub mod ntag;
pub mod nvdec;
pub mod nvenc;
pub mod pcie;
pub mod pcie_endpoint;
pub mod pg_up;
pub mod pinmux;
pub mod pmc;
pub mod regs;
pub mod rtc;
pub mod sb;
pub mod sdcard;
pub mod sdmmc;
pub mod se;
pub mod serial;
pub mod soctherm;
pub mod spi;
pub mod sysctr0;
pub mod tmp451;
pub mod tmr;
pub mod touch;
pub mod tsec;
pub mod tzram;
pub mod usb2;
pub mod vic;
pub mod wifi_ap;
pub mod xusb_device;
pub mod xusb_padctl;

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

use crate::{boot::BootParams, bus::Bus, clock::GuestClock, gic::Gic, memory::GuestRam};

use self::{
    bq24193::Bq24193,
    car::Car,
    cldvfs::ClDvfs,
    dc::{Dc, Frame, NoFrame, Pixel},
    dpaux::Dpaux,
    dsi::Dsi,
    emc::{EmcPort, View},
    emmc::{Emmc, EmmcImages, ImageError},
    gm20b::Gm20b,
    gpio::Gpio,
    host1x::{Host1xSync, Syncpoints},
    i2c::{ByteRegisters, I2cController, I2cDevice, RegisterMap},
    max17050::{Battery, Max17050},
    max77621::Max77621,
    mc::{DeviceMemory, Mc},
    mipi_cal::MipiCal,
    pmc::Pmc,
    regs::RegisterFile,
    sdcard::{SdCard, SdCardImage},
    sdmmc::{Card, Sdmmc},
    tmr::Timers,
};

/// I2C1: battery charger and fuel gauge.
const I2C1_BASE: u64 = 0x7000_C000;
const I2C1_INTID: u32 = 32 + 38;
/// I2C5: power management IC and real-time clock.
const I2C5_BASE: u64 = 0x7000_D000;
const I2C5_INTID: u32 = 32 + 53;
/// The power management IC's interrupt output (pcv binds this interrupt,
/// level-triggered, when it starts; Linux's `tegra210.dtsi` has it as the
/// PMC's, shared interrupt 86).
const PMU_INTID: u32 = 32 + 86;

/// SDMMC1: the microSD slot's port (base and interrupt as in the public
/// Linux `tegra210.dtsi`). The filesystem process starts it once the
/// slot's card detect reports a card ([`GPIO_INPUTS`], port Z).
const SDMMC1_BASE: u64 = 0x700B_0000;
const SDMMC1_INTID: u32 = 32 + 14;
/// The microSD card's supply enable, GPIO E4. The clock and power driver
/// drives it high when the filesystem process enables the card's
/// regulator (FS `rgltr` request 0 is followed by a masked write of
/// `0x1010` to `0x6000d1a0`) and low when it disables it.
const SD_CARD_SUPPLY: (usize, u8) = (4, 4);
/// The microSD slot's card detect switch, GPIO Z1; see [`GPIO_INPUTS`].
const SD_CARD_DETECT: (usize, u8) = (25, 1);
/// SDMMC4: the eMMC port.
const SDMMC4_BASE: u64 = 0x700B_0600;
const SDMMC4_INTID: u32 = 32 + 31;
/// SDMMC2: the port of the game card slot's controller chip. The
/// filesystem process sets the port up when it starts its game card
/// driver (accesses from `0x700b0200` about ten seconds into the boot).
/// Nothing answers on it here: the slot is empty, and the controller chip
/// behind the port is not modeled, so commands to it time out as on an
/// empty port.
///
/// Without the chip the driver tries again about every 2 s. In the IPC
/// trace, FS sets `gpio` pad output low then high, turns the regulator on
/// and sets the `clkrst` rate to 200 MHz every 2.0 s \[inference: that this
/// is a retry of the chip's start-up\]. Requests that need the driver wait
/// for it. At boot, `ns`'s `IDeviceOperator` command 225 (after
/// `IsGameCardInserted` says no) took from 0.3 ms to 5.9 s across runs
/// before failing with 2002-2629, and `am` waits for `ns` before it starts
/// `qlaunch`. So the lock screen comes up that much later, and with it the
/// lock screen's auto-sleep (see `script.rs`).
const SDMMC2_BASE: u64 = 0x700B_0200;
const SDMMC2_INTID: u32 = 32 + 15;

/// The AHB arbiter's priority registers (hekate `t210.h` `ARB_PRI_BASE`:
/// one each for the CPU, the BPMP, the VCP and DMA, and one more), which
/// hekate writes once at start (`hw_init.c`). Buses are not arbitrated
/// here, so they keep what is written and do nothing; unwritten, their
/// values are not public and they refuse to be read.
const ARB_PRI_BASE: u64 = 0x6000_3000;
const ARB_PRI_SIZE: u64 = 0x14;

/// Explanations devices leave when they refuse an access, for whoever reads
/// the machine's stop report.
#[derive(Default)]
pub struct Notes {
    notes: Mutex<Vec<String>>,
    survey: AtomicBool,
}

impl Notes {
    /// Adds `note` unless the same one is already waiting to be read.
    pub fn add(&self, note: String) {
        let mut notes = self.notes.lock().unwrap();
        if !notes.contains(&note) {
            notes.push(note);
        }
    }

    pub fn take(&self) -> Vec<String> {
        std::mem::take(&mut self.notes.lock().unwrap())
    }

    /// Whether a survey is being run: devices then answer what they do not
    /// model with zero, and swallow such writes, instead of refusing. They
    /// leave their note all the same. What the guest does afterwards is not
    /// what a console does; the point is the list of notes.
    pub fn survey(&self) -> bool {
        self.survey.load(Ordering::Relaxed)
    }

    pub fn set_survey(&self, survey: bool) {
        self.survey.store(survey, Ordering::Relaxed);
    }
}

/// Host1x syncpoint threshold interrupt (nvservices text `0x5a454`).
const HOST1X_SYNCPT_INTID: u32 = 32 + 65;
/// Display controllers A and B (nvservices text `0x23634`).
const DC_INTID: u32 = 32 + 73;
const DCB_INTID: u32 = 32 + 74;

/// Pulse width modulators: one register per channel, each `boot` changes
/// by read-modify-write (text `0x46f70`); they start at zero, their reset
/// value. Channel 0 dims the backlight.
const PWM_BASE: u64 = 0x7000_A000;
const PWM_SIZE: u64 = 0x40;
const PWM_ENABLE: u32 = 1 << 31;
/// High time in 256ths of the period.
const PWM_WIDTH: u32 = 0x7FFF << 16;

/// GPIO port V, whose pin 1 powers the backlight driver.
const BACKLIGHT_PORT: usize = 21;
const BACKLIGHT_ENABLE_PIN: u8 = 1;

/// The fuse block, as far as it is read: the chip's speed and leakage
/// measurements from the factory, by which the firmware's clock driver
/// picks voltages (pcv reads them once, in this order). Register names as
/// publicly documented. A measurement belongs to one particular chip, and
/// this console has none: they read zero, the slowest and least leaky chip
/// there could be. Like the device ID, that is a choice, not a fact. With
/// them: the SKU, key hash and hidden private key fuses the boot ROM leaves
/// a retail Erista ([`crate::bootrom::fuses_after_boot`]).
const FUSE_BASE: u64 = 0x7000_F800;
const FUSE_SIZE: u64 = 0x400;
const FUSE_CPU_SPEEDO_0_CALIB: u64 = 0x114;
const FUSE_CPU_IDDQ_CALIB: u64 = 0x118;
const FUSE_CPU_SPEEDO_2_CALIB: u64 = 0x130;
const FUSE_SOC_SPEEDO_0_CALIB: u64 = 0x134;
const FUSE_SOC_IDDQ_CALIB: u64 = 0x140;
const FUSE_GPU_IDDQ_CALIB: u64 = 0x228;
/// `FUSE_OPT_PRIV_SEC_DIS` (name as switchbrew's fuse register table lists
/// it at 0x7000FA64): bit 0 set disables privileged security. nvservices
/// takes the GPU's secure-boot path when bit 0 is clear (text `0x8BCB4`),
/// and only that path works with the GPU firmware it carries (spec §4.2).
/// Assumed clear, as on a retail, production-fused console.
const FUSE_OPT_PRIV_SEC_DIS: u64 = 0x264;
/// `FUSE_RESERVED_CALIB0` (name as in open-source bootloaders): the
/// factory's calibration of the GPU PLL's voltage sensor. The clock driver
/// reads it when it first enables the GPU clock (pcv, on nvservices'
/// `clkrst` request for device `0x40000002`). Like the speed measurements
/// above it belongs to one chip; zero is a chip that was never calibrated,
/// a case the public driver for this PLL provides for.
const FUSE_RESERVED_CALIB0: u64 = 0x304;
/// USB pad calibration read by usb text `0x5840`, consumed by `0x5dc0`.
/// No physical die trim is available: assume zero trim codes. The driver
/// extracts fields without rejecting zero (Erista uses fixed termination).
const FUSE_USB_CALIB: u64 = 0x1f0;
/// Additional USB pad trim fields, with the same assumption as above.
const FUSE_USB_CALIB_EXT: u64 = 0x350;
/// `FUSE_SOC_SPEEDO_1_CALIB`: the boot ROM's patch version (switchbrew
/// "Fuses"), which package1ldr requires to be at least 0x7F (its `0x7A0`).
/// This console's ROM patch level is not known: \[assumption\] 0x7F, the
/// least package1ldr accepts. Not among the measurements pcv is seen to
/// read above; should it read this one too, 0x7F is what it gets.
const FUSE_SOC_SPEEDO_1_CALIB: u64 = 0x138;
const BOOTROM_PATCH_VERSION: u32 = 0x7F;
/// `FUSE_RESERVED_ODM4`, the console's configuration (switchbrew "Fuses"):
/// hardware type Icosa (bit 2 set, bit 8 and bits 19:16 clear) and state
/// production (bits 1:0 clear, bit 9 set) as package1ldr decodes them
/// (`0x24C`, `0x280`) and as the monitor reports them, and the DRAM ID in
/// bits 7:3 and 14:12 (\[inference\] its low and high bits). A memory size
/// with no DRAM ID has no such fuse word: reads of it are refused.
const FUSE_RESERVED_ODM4: u64 = 0x1D8;
const ODM4_ICOSA_PRODUCTION: u32 = 1 << 2 | 1 << 9;
/// `FUSE_RESERVED_ODM6` and `ODM7`, the anti-downgrade fuses: 25 burnt in
/// ODM7, what firmware 23.5.0 expects of a production unit (package1ldr
/// `0x638`; the monitor's fuse version, 25), none in ODM6.
const FUSE_RESERVED_ODM6: u64 = 0x1E0;
const FUSE_RESERVED_ODM7: u64 = 0x1E4;
const ANTI_DOWNGRADE_FUSES: u32 = 0x01FF_FFFF;
/// `FUSE_SPARE_BIT_5`: set on production units (switchbrew "Fuses");
/// package1ldr requires bit 0 (`0x828`).
const FUSE_SPARE_BIT_5: u64 = 0x394;
/// What the NX bootloader of firmware 23.5.0 requires of the fuse block in
/// its hardware-state check (`0x40002A7C`, as NxBootloaderResearch tabled
/// it), by offset, besides what is above: the fuse driver as the boot ROM
/// leaves it (timings, software writes to the cache allowed, programming
/// not disabled) and fuses that must be clear on this console. Bits its
/// masks leave out read zero \[assumption\], except `FUSE_FUSECTRL`'s
/// state field (bits 20:16), which reads idle (4), what package1ldr's
/// fuse-burn path waits for.
const FUSES_THE_BOOTLOADER_CHECKS: [(u64, u32); 36] = [
    (0x000, 4 << 16), // FUSE_FUSECTRL
    (0x00C, 0),       // FUSE_FUSEWDATA
    (0x010, 0x0003_0202),
    (0x014, 0x21),
    (0x018, 0x0403_0202),
    (0x01C, 0x0304_00F0), // FUSE_FUSETIME_RD1, _RD2, _PGM1, _PGM2
    (0x020, 0),           // FUSE_PRIV2INTFC_START
    (0x024, 0),           // FUSE_FUSEBYPASS
    (0x02C, 0),           // FUSE_DISABLEREGPROGRAM
    (0x030, 1),           // FUSE_WRITE_ACCESS_SW
    (0x04C, 0x24),        // FUSE_FUSETIME_RD3
    (0x104, 1),
    (0x160, 0),
    (0x194, 0),
    (0x230, 0),
    (0x234, 0),
    (0x2A8, 0),
    (0x2C0, 0),
    (0x2F4, 0),
    (0x30C, 0),
    (0x310, 0),
    (0x314, 0),
    (0x318, 0),
    (0x324, 0),
    (0x328, 0),
    (0x32C, 0),
    (0x330, 0),
    (0x334, 0),
    (0x338, 0),
    (0x33C, 0),
    (0x340, 0),
    (0x344, 0),
    (0x348, 0),
    (0x34C, 0),
    (0x354, 0),
    (0x37C, 0),
];

/// Levels the board puts on GPIO input pins, by port (`controller * 4 +
/// port`). A port not listed refuses to have its inputs read.
///
/// Port X: the volume buttons on pins 6 and 7 pull to ground when pressed
/// (pin assignment recalled from open-source bootloaders); nobody is
/// pressing them. The port's other pins read low, which is an assumption.
///
/// Port Z: the card detect switch of the microSD slot on pin 1 pulls to
/// ground when a card is in (pin assignment recalled from the same
/// sources; the filesystem process's driver starts the port, and asks for
/// the card's supply, after reading the pin low). The slot is empty unless
/// a card image is given ([`SD_CARD_DETECT`]). The firmware's GPIO driver
/// samples the port once its clock driver is up (Bus, read of
/// `0x6000d634`). The port's other pins read low, which is an assumption.
/// Touch IRQ X1 is active-low; motion IRQ X2 is idle low. Rail detect E6
/// is pulled high with no left Joy-Con (hekate `joycon.c`, active-low
/// detection). Assignments: Atmosphere `gpio_internal_pad_map_combination.inc`.
/// Port K4: PD interrupt, active low; other K inputs assumed low.
/// H6 likewise reads high with no right Joy-Con attached; other H inputs
/// are assumed low until driven by their board devices. Both rails start
/// out empty; [`Soc::seat_joycons`] is what changes that.
///
/// Port S: the game card slot's detect switch on pin 3 (pin assignment
/// recalled from open-source system software), taken to pull to ground
/// when a card is in, like the microSD slot's; the slot is empty. The
/// filesystem process's game card driver has the port sampled (read of
/// `0x6000d438`). The port's other pins read low, which is an assumption.
///
/// Port V: `audio`'s headphone jack detect, GPIO pad `0x15`, is pin 6 (the
/// GPIO driver clears that pin's output enable and programs it as an
/// any-edge interrupt with debounce when `audio` opens the pad, writes to
/// `0x6000d594`/`0x6000d564`/`0x6000d5b4`). `audio` takes a level of zero
/// as a headphone inserted (text `0x6be90`); nothing is plugged in, so
/// the pin reads high and never changes. The port's other input pins read
/// low, which is an assumption.
///
/// Ports D and G: the Joy-Con rails' CTS lines on D4 and G3
/// ([`joycon::LEFT_CTS`], [`joycon::RIGHT_CTS`]), high on an empty rail.
/// The ports' other pins read low, which is an assumption.
const GPIO_INPUTS: [(usize, u8); 9] = [
    (3, 0x10),
    (4, 0x40),
    (6, 0x08),
    (7, 0x40),
    (10, 0x10),
    (18, 0x08),
    (21, 0x40),
    (23, 0xC2),
    (25, 0x02),
];

/// The PMC wake events whose pads are modeled ([`Soc::wake_pad`]).
const WAKE_EVENTS: [u8; 3] = [
    pmc::WAKE_EVENT_PMIC,
    pmc::WAKE_EVENT_UART2_CTS,
    pmc::WAKE_EVENT_UART3_CTS,
];

/// What the console's screen shows.
pub struct Screen {
    /// The picture as someone holding the console sees it. The panel is a
    /// portrait one mounted on its side, its first scan line along the
    /// console's right edge, so this is the display controller's picture
    /// turned a quarter turn clockwise. Shared with every other look at the
    /// same frame.
    pub frame: Arc<Frame>,
    /// Whether the panel is awake and unblanked.
    pub panel_on: bool,
    /// Whether the backlight driver is powered and given a nonzero duty
    /// cycle.
    pub backlight_on: bool,
}

/// The blocks only a machine that runs the boot chain itself has (see
/// [`Soc::new`]): the BPMP's memory and the registers it starts and stops
/// processors with. A machine started at kernel entry has none of them on
/// its bus, exactly as before the boot chain was modelled.
pub struct BootBlocks {
    pub iram: Arc<iram::Iram>,
    pub tzram: Arc<tzram::Tzram>,
    pub flow: Arc<flow::FlowController>,
    pub evp: Arc<evp::Evp>,
    pub sb: Arc<sb::Sb>,
    pub ictlr: Arc<ictlr::Ictlr>,
    /// What the clock and reset controller tells the BPMP's machine: see
    /// [`BootSignals`].
    pub signals: Arc<BootSignals>,
    /// The timer block, whose watchdogs the BPMP's machine watches
    /// ([`tmr::Timers::watchdog_expired`]).
    pub timers: Arc<tmr::Timers>,
    /// USB1, the port of the ROM's recovery mode ([`usb2`]).
    pub usb2: Arc<usb2::Usb2>,
    /// What only a machine that runs the boot ROM itself has
    /// ([`Start::Reset`]); without, the ROM is high-level emulated.
    pub rom: Option<RomBlocks>,
}

/// The blocks of a machine given a dump of the boot ROM: its memory.
pub struct RomBlocks {
    pub irom: Arc<irom::Irom>,
}

impl BootBlocks {
    /// The boot ROM's memory, in a machine that runs the ROM.
    pub fn irom(&self) -> Option<&Arc<irom::Irom>> {
        self.rom.as_ref().map(|rom| &rom.irom)
    }
}

/// The state a SoC is built in.
pub enum Start<'a> {
    /// As the boot chain leaves it for the kernel.
    Kernel,
    /// Out of reset, as far as the high-level emulated boot ROM
    /// ([`crate::bootrom`]) leaves it for package1ldr: the boot chain runs
    /// from there ([`BootBlocks`], [`car::Car::power_on`],
    /// [`pmc::Pmc::power_on`]). The fuse block holds the console's fuses
    /// and patch records as at reset, with the private key hidden as the
    /// ROM leaves it.
    AfterRom { fuses: &'a crate::bootrom::Fuses },
    /// At reset, with a dump of the boot ROM in IROM and the fuses the ROM
    /// reads: the BPMP runs the ROM itself.
    Reset {
        rom: irom::RomImage,
        fuses: &'a crate::bootrom::Fuses,
    },
}

impl Start<'_> {
    /// Whether the boot chain runs in this machine.
    fn powers_on(&self) -> bool {
        !matches!(self, Start::Kernel)
    }
}

/// Reset events of the clock and reset controller that start or stop a
/// processor, for the BPMP's run loop to act on between instructions.
#[derive(Default)]
pub struct BootSignals {
    /// CPU-cluster core 0 has just been let out of reset
    /// ([`car::Car::on_cpu0_released`]).
    pub cpu0_released: AtomicBool,
    /// The BPMP is held in reset: `RST_DEVICES_L` bit 1 (`COP`, hekate
    /// `clock.h`) set.
    pub cop_held: AtomicBool,
    /// The BPMP's reset was released since the run loop last looked: it
    /// starts again at `EVP_COP_RESET_VECTOR`, or at [`Self::restart_at`]
    /// if that is set.
    pub cop_restarted: AtomicBool,
    /// Where a restart the TSEC asks for begins
    /// ([`tsec::TsecSoc::restart_bpmp`]).
    pub restart_at: Mutex<Option<u32>>,
}

/// `RST_DEVICES_L` bit 1: the BPMP ("COP").
const COP_RESET: (u64, u32) = (0x004, 1);

/// How the panel and its backlight stand, whether or not the display
/// controller's picture can be composed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Panel {
    /// Whether the panel is awake and unblanked.
    pub panel_on: bool,
    /// Whether the backlight driver is powered and given a nonzero duty
    /// cycle.
    pub backlight_on: bool,
}
pub struct Soc {
    pub mc: Arc<Mc>,
    pub gpio: Arc<Gpio>,
    pub notes: Arc<Notes>,
    /// Window-facing touchscreen source, in native 1280x720 coordinates.
    pub touch_input: Arc<touch::TouchInput>,
    /// The pair of Joy-Con: where they are and what is held on them.
    pub joycons: Arc<joycon::Controls>,
    /// The power management IC: the power key, the clock's alarms, and
    /// the interrupt output that wakes a sleeping console.
    pub pmic: max77620::Max77620,
    /// The power management controller, where wake events arrive.
    pmc: Arc<Pmc>,
    /// The fuse block; set once it is built.
    pub fuse: std::sync::OnceLock<Arc<fuse::Fuse>>,
    /// The boot chain's blocks, in a machine that runs it.
    pub boot: Option<BootBlocks>,
    /// The security engine, for the boot ROM's and the TSEC's key slots.
    pub se: Arc<se::Se>,
    /// The TSEC, for the machine to connect to the boot chain's blocks
    /// ([`tsec::Tsec::connect`]).
    pub tsec: Arc<tsec::Tsec>,
    /// The WiFi function of the radio, for placing an access point around
    /// it (`--wifi-ssid`); set once the PCIe root complex is built.
    pub wifi: std::sync::OnceLock<Arc<bcm4356::Bcm4356Wifi>>,
    /// What is in range of the Bluetooth radio (`--pro-controller`).
    pub bluetooth: Arc<bluetooth_uart::Air>,
    /// Guest time, which the SoC's clocks count.
    pub clock: Arc<GuestClock>,
    gic: Arc<Gic>,
    dc: Arc<Dc>,
    /// The external output's display controller; nothing is attached to it.
    dcb: Arc<Dc>,
    dsi: Arc<Dsi>,
    /// Memory as the display controller reaches it.
    display_memory: DeviceMemory,
    pwm: Arc<RegisterFile>,
    /// Kept for its host thread, which it stops when dropped, and its
    /// channel report.
    gpu: Gm20b,
}

impl Soc {
    /// Builds every block and places it at its address on `bus`. `emmc`
    /// holds the contents of the eMMC module and `sd_card` those of the
    /// microSD card; without them their slots are empty. `start` is the
    /// state the SoC is in: as the boot chain leaves it for the kernel, or
    /// out of reset for the boot chain to run on ([`Start`]).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        params: &BootParams,
        gic: &Arc<Gic>,
        ram: &Arc<GuestRam>,
        clock: &Arc<GuestClock>,
        emmc: Option<EmmcImages>,
        sd_card: Option<SdCardImage>,
        start: Start<'_>,
        bus: &mut Bus,
    ) -> Result<Self, ImageError> {
        let power_on = start.powers_on();
        let mc = Arc::new(if power_on {
            Mc::power_on()
        } else {
            Mc::new(params.dram)
        });
        let notes: Arc<Notes> = Arc::default();
        let violations = Arc::clone(&notes);
        mc.on_violation(Box::new(move |violation| violations.add(violation)));
        let syncpoints = Arc::new(Syncpoints::new(line(gic, HOST1X_SYNCPT_INTID)));
        let elapsed = |clock: &Arc<GuestClock>| {
            let clock = Arc::clone(clock);
            Box::new(move || clock.now())
        };
        let controller = |intid| {
            Arc::new(Dc::new(
                Arc::clone(&syncpoints),
                elapsed(clock),
                line(gic, intid),
            ))
        };
        let gpu = Gm20b::new(
            DeviceMemory::new(&mc, ram, mc::SMMU_GPU_ASID),
            Arc::clone(&syncpoints),
            [
                line(gic, gm20b::STALL_INTID),
                line(gic, gm20b::NONSTALL_INTID),
            ],
            Arc::clone(clock),
            gm20b::maxwell_engines(Arc::clone(&syncpoints)),
            Arc::clone(&notes),
        );
        bus.map("GPU BAR0", gm20b::BAR0_BASE, gm20b::BAR0_SIZE, gpu.bar0());
        bus.map("GPU BAR1", gm20b::BAR1_BASE, gm20b::BAR1_SIZE, gpu.bar1());
        let vic = Arc::new(falcon_host::HostFalcon::new(
            DeviceMemory::new(&mc, ram, mc::SMMU_VIC_ASID),
            vic::Vic::new(
                DeviceMemory::new(&mc, ram, mc::SMMU_VIC_ASID),
                Arc::clone(&syncpoints),
                Arc::clone(&notes),
            ),
            Arc::clone(&notes),
        ));
        bus.map(
            "VIC",
            falcon_host::VIC_BASE,
            falcon_host::SIZE,
            Box::new(Arc::clone(&vic)),
        );
        let nvdec = Arc::new(falcon_host::HostFalcon::new(
            DeviceMemory::new(&mc, ram, mc::SMMU_NVDEC_ASID),
            nvdec::Nvdec::new(
                DeviceMemory::new(&mc, ram, mc::SMMU_NVDEC_ASID),
                Arc::clone(&syncpoints),
                Arc::clone(&notes),
            ),
            Arc::clone(&notes),
        ));
        bus.map(
            "NVDEC",
            falcon_host::NVDEC_BASE,
            falcon_host::SIZE,
            Box::new(Arc::clone(&nvdec)),
        );
        let nvenc = Arc::new(falcon_host::HostFalcon::new(
            DeviceMemory::new(&mc, ram, mc::SMMU_NVENC_ASID),
            nvenc::Nvenc::new(
                DeviceMemory::new(&mc, ram, mc::SMMU_NVENC_ASID),
                Arc::clone(&syncpoints),
                Arc::clone(&notes),
            ),
            Arc::clone(&notes),
        ));
        bus.map(
            "NVENC",
            falcon_host::NVENC_BASE,
            falcon_host::SIZE,
            Box::new(Arc::clone(&nvenc)),
        );
        let channels: Vec<Arc<host1x_channel::Channel>> = (0..7)
            .map(|index| {
                let (vic, nvdec, nvenc) =
                    (Arc::clone(&vic), Arc::clone(&nvdec), Arc::clone(&nvenc));
                let channel = Arc::new(host1x_channel::Channel::new(
                    DeviceMemory::new(&mc, ram, mc::SMMU_HOST1X_ASID),
                    Arc::clone(&syncpoints),
                    Arc::clone(clock),
                    // The VIC refuses any class but its own.
                    Arc::new(move |class, offset, value| match class {
                        nvdec::CLASS => nvdec.method(class, offset, value),
                        nvenc::CLASS => nvenc.method(class, offset, value),
                        _ => vic.method(class, offset, value),
                    }),
                    Arc::clone(&notes),
                ));
                bus.map(
                    "host1x channel",
                    0x5000_0000 + index * 0x4000,
                    0x100,
                    Box::new(Arc::clone(&channel)),
                );
                channel
            })
            .collect();
        let iram: Option<Arc<iram::Iram>> = power_on.then(Arc::default);
        let tzram: Option<Arc<tzram::Tzram>> = power_on.then(Arc::default);
        if let Some(iram) = &iram {
            // Memory clients reach IRAM through the redirect aperture.
            mc.attach_iram(Arc::clone(iram));
        }
        let se_memory: Box<dyn se::EngineMemory> = match (&iram, &tzram) {
            (Some(iram), Some(tzram)) => Box::new(se::BootMemory::new(
                Arc::clone(iram),
                Arc::clone(tzram),
                Arc::clone(ram),
            )),
            _ => Box::new(DeviceMemory::new(&mc, ram, mc::SMMU_SE_ASID)),
        };
        let se = Arc::new(se::Se::new(se_memory, Arc::clone(&notes), params.rng_seed));
        bus.map(
            "security engine",
            se::BASE,
            se::SIZE,
            Box::new(Arc::clone(&se)),
        );
        let tsec = Arc::new(tsec::Tsec::new(
            DeviceMemory::new(&mc, ram, tsec::SMMU_TSEC_ASID),
            Arc::clone(&notes),
        ));
        bus.map("TSEC", tsec::BASE, tsec::SIZE, Box::new(Arc::clone(&tsec)));
        let gpio = Gpio::with_interrupts(GPIO_INPUTS, gpio::INTIDS.map(|intid| line(gic, intid)));
        let gpio = Arc::new(if power_on { gpio.power_on() } else { gpio });
        if sd_card.is_some() {
            let (port, pin) = SD_CARD_DETECT;
            gpio.set_input(port, pin, false)
                .expect("the card detect port has its inputs listed");
        }
        let touch_gpio = Arc::downgrade(&gpio);
        let touch_input = Arc::new(touch::TouchInput::new(Box::new(move |asserted| {
            if let Some(gpio) = touch_gpio.upgrade() {
                gpio.set_input(23, 1, !asserted);
            }
        })));
        let touch_reset = Arc::clone(&touch_input);
        gpio.observe_output(Box::new(move |port, pin, high| {
            if port == 9 && pin == 7 {
                touch_reset.set_reset(high);
            }
        }));
        let joycons = Arc::new(joycon::Controls::default());
        let cts_gpio = Arc::downgrade(&gpio);
        joycons.connect_cts(Box::new(move |side, high| {
            let (port, pin) = match side {
                joycon::Side::Left => joycon::LEFT_CTS,
                joycon::Side::Right => joycon::RIGHT_CTS,
                joycon::Side::Pro => return,
            };
            if let Some(gpio) = cts_gpio.upgrade() {
                gpio.set_input(port, pin, high)
                    .expect("the rails' CTS ports have their inputs listed");
            }
        }));
        let sb: Arc<sb::Sb> = Arc::default();
        // USB1, which the ROM sets up on every cold boot: at reset where
        // the ROM runs, as the ROM leaves it where it is high-level
        // emulated.
        let usb2 = power_on.then(|| {
            let usb2 = Arc::new(match &start {
                Start::Reset { .. } => usb2::Usb2::new(),
                _ => usb2::Usb2::after_rom(),
            });
            bus.map("USB1", usb2::BASE, usb2::SIZE, Box::new(Arc::clone(&usb2)));
            usb2
        });
        let rom = match &start {
            Start::Reset { rom, .. } => {
                let irom = Arc::new(irom::Irom::new(rom.clone(), Arc::clone(&sb)));
                bus.map(
                    "IPATCH",
                    irom::IPATCH_BASE,
                    irom::IPATCH_SIZE,
                    Box::new(irom::Ipatch(Arc::clone(&irom))),
                );
                Some(RomBlocks { irom })
            }
            _ => None,
        };
        let soc = Self {
            gpio,
            touch_input,
            joycons,
            pmic: max77620::Max77620::new(Arc::clone(clock), line(gic, PMU_INTID)),
            pmc: Arc::new(if power_on {
                Pmc::power_on()
            } else {
                Pmc::new()
            }),
            fuse: std::sync::OnceLock::new(),
            boot: iram.map(|iram| BootBlocks {
                iram,
                tzram: tzram.clone().expect("TZRAM is made with IRAM"),
                flow: Arc::new(flow::FlowController::new(
                    Arc::clone(clock),
                    Arc::clone(&notes),
                )),
                evp: Arc::default(),
                sb,
                ictlr: Arc::default(),
                signals: Arc::default(),
                timers: Arc::new(Timers::power_on(elapsed(clock))),
                usb2: usb2.expect("USB1 is made with IRAM"),
                rom,
            }),
            se,
            tsec,
            wifi: std::sync::OnceLock::new(),
            bluetooth: Arc::new(bluetooth_uart::Air::new(Arc::clone(clock))),
            clock: Arc::clone(clock),
            gic: Arc::clone(gic),
            dc: controller(DC_INTID),
            dcb: controller(DCB_INTID),
            dsi: Arc::new(Dsi::new(Arc::clone(&notes))),
            display_memory: DeviceMemory::new(&mc, ram, mc::SMMU_DC_ASID),
            pwm: Arc::new(RegisterFile::zeroed([])),
            gpu,
            mc,
            notes,
        };
        let srk_pmc = Arc::clone(&soc.pmc);
        soc.se.on_srk_saved(Box::new(move |words| {
            for (offset, word) in se::SRK_SCRATCH.into_iter().zip(words) {
                crate::bus::MmioDevice::write(srk_pmc.as_ref(), 0, offset, 4, word.into())
                    .expect("the PMC takes its secure scratch registers");
            }
        }));
        bus.map(
            "XUSB device",
            xusb_device::BASE,
            xusb_device::SIZE,
            Box::new(xusb_device::XusbDevice::default()),
        );
        bus.map(
            "SOCTHERM",
            soctherm::BASE,
            soctherm::SIZE,
            Box::new(soctherm::Soctherm::default()),
        );
        bus.map(
            "display controller",
            dc::BASE,
            dc::SIZE,
            Box::new(Arc::clone(&soc.dc)),
        );
        bus.map(
            "display controller B",
            dc::DCB_BASE,
            dc::SIZE,
            Box::new(Arc::clone(&soc.dcb)),
        );
        bus.map("DSI", dsi::BASE, dsi::SIZE, Box::new(Arc::clone(&soc.dsi)));
        let host1x = Host1xSync::new(
            syncpoints,
            vec![Arc::clone(&soc.dc), Arc::clone(&soc.dcb)],
            channels,
        );
        bus.map(
            "host1x",
            host1x::SYNC_BASE,
            host1x::SYNC_SIZE,
            Box::new(host1x),
        );
        bus.map(
            "MIPI calibration",
            mipi_cal::BASE,
            mipi_cal::SIZE,
            Box::new(MipiCal::default()),
        );
        bus.map(
            "DPAUX1",
            dpaux::BASE,
            dpaux::SIZE,
            Box::new(Dpaux::default()),
        );
        let serial = |peer: Box<dyn serial::Peer>, intid| {
            Arc::new(serial::Serial::new(
                peer,
                line(gic, intid),
                Arc::clone(&soc.notes),
            ))
        };
        let rail = |side| {
            Box::new(joycon::Rail::new(
                side,
                Arc::clone(&soc.joycons),
                Arc::clone(&soc.clock),
            ))
        };
        let uarts = vec![
            (
                joycon::RIGHT_UART_BASE,
                "UART-B",
                serial(rail(joycon::Side::Right), joycon::RIGHT_UART_INTID),
            ),
            (
                joycon::LEFT_UART_BASE,
                "UART-C",
                serial(rail(joycon::Side::Left), joycon::LEFT_UART_INTID),
            ),
            (
                bluetooth_uart::BASE,
                "Bluetooth UART",
                serial(
                    Box::new(bluetooth_uart::Line::new(
                        Arc::clone(&soc.clock),
                        Arc::clone(&soc.bluetooth),
                    )),
                    bluetooth_uart::INTID,
                ),
            ),
        ];
        bluetooth_uart::keep_radio(&uarts[2].2, Arc::clone(&soc.clock));
        for (base, name, uart) in &uarts {
            bus.map(name, *base, serial::SIZE, Box::new(Arc::clone(uart)));
        }
        bus.map(
            "APB DMA",
            apb_dma::BASE,
            apb_dma::SIZE,
            Box::new(apb_dma::ApbDma::new(
                DeviceMemory::new(&soc.mc, ram, apb_dma::SMMU_ASID),
                uarts
                    .into_iter()
                    .map(|(base, _, uart)| (base, uart))
                    .collect(),
                (0..32).map(|c| line(gic, apb_dma::intid(c))).collect(),
                Arc::clone(&soc.notes),
            )),
        );
        bus.map(
            "XUSB pads",
            xusb_padctl::BASE,
            xusb_padctl::SIZE,
            Box::new(xusb_padctl::XusbPadctl::default()),
        );
        let pcie = pcie::Pcie::new(
            DeviceMemory::new(&soc.mc, ram, pcie::SMMU_ASID),
            line(gic, pcie::INTR_INTID),
            line(gic, pcie::MSI_INTID),
        );
        bus.map("PCIe", pcie::BASE, pcie::SIZE, Box::new(pcie.clone()));
        bus.map(
            "PCIe configuration",
            pcie::CONFIG_BASE,
            pcie::CONFIG_SIZE,
            Box::new(pcie.configuration()),
        );
        bus.map(
            "PCIe memory",
            pcie::MEMORY_BASE,
            pcie::MEMORY_SIZE,
            Box::new(pcie.memory_window()),
        );
        let wifi = Arc::new(bcm4356::Bcm4356Wifi::new(
            pcie.upstream(),
            Arc::clone(&soc.notes),
            Arc::clone(clock),
        ));
        pcie.attach(Arc::clone(&wifi) as Arc<dyn pcie_endpoint::PcieFunction>);
        let _ = soc.wifi.set(wifi);
        bus.map("PWM", PWM_BASE, PWM_SIZE, Box::new(Arc::clone(&soc.pwm)));
        bus.map("PMC", pmc::BASE, pmc::SIZE, Box::new(Arc::clone(&soc.pmc)));
        bus.map(
            "pin multiplexer",
            pinmux::BASE,
            pinmux::SIZE,
            Box::new(if power_on {
                pinmux::power_on()
            } else {
                pinmux::at_kernel_entry()
            }),
        );
        if power_on {
            bus.map(
                "APB_MISC",
                apb_misc::BASE,
                apb_misc::SIZE,
                Box::new(apb_misc::ApbMisc::power_on()),
            );
        } else {
            // The pad control groups, configuration that starts at zero
            // like the pin multiplexer, and the chip ID: what the kernel's
            // processes reach of the block.
            let pads = apb_misc::PAD_GROUPS;
            bus.map(
                "pad groups",
                apb_misc::BASE + pads.start,
                pads.end - pads.start,
                Box::new(RegisterFile::zeroed([])),
            );
            bus.map(
                "chip ID",
                apb_misc::BASE + apb_misc::HIDREV,
                4,
                Box::new(RegisterFile::strict([(0, apb_misc::HIDREV_VALUE)])),
            );
        }
        bus.map(
            "memory controller",
            mc::BASE,
            mc::SIZE,
            Box::new(Arc::clone(&soc.mc)),
        );
        bus.map(
            "GPIO",
            gpio::BASE,
            gpio::SIZE,
            Box::new(Arc::clone(&soc.gpio)),
        );
        let emc: Arc<emc::Emc> = Arc::new(if power_on {
            emc::Emc::power_on()
        } else {
            emc::Emc::default()
        });
        let (clock_change, disp1, disp2) =
            (Arc::clone(&emc), Arc::clone(&soc.dc), Arc::clone(&soc.dcb));
        let agic = Arc::new(agic::Agic::new(soc.line(agic::INTID)));
        bus.map(
            "AGIC distributor",
            agic::DISTRIBUTOR_BASE,
            agic::DISTRIBUTOR_SIZE,
            Box::new(agic.distributor()),
        );
        bus.map(
            "AGIC CPU interface",
            agic::CPU_INTERFACE_BASE,
            agic::CPU_INTERFACE_SIZE,
            Box::new(agic.cpu_interface()),
        );
        let adsp = Arc::new(adsp::Adsp::new(
            DeviceMemory::new(&soc.mc, ram, adsp::SMMU_ASID),
            agic,
            Arc::clone(&soc.notes),
            clock,
        ));
        bus.map(
            "AMISC",
            adsp::AMISC_BASE,
            adsp::AMISC_SIZE,
            Box::new(Arc::clone(&adsp)),
        );
        bus.map(
            "ADSP vectors",
            adsp::EVP_BASE,
            adsp::EVP_SIZE,
            Box::new(RegisterFile::strict([])),
        );
        bus.map("I2S", ahub::I2S_BASE, ahub::I2S_SIZE, Box::new(ahub::i2s()));
        bus.map(
            "ADMAIF",
            ahub::ADMAIF_BASE,
            ahub::ADMAIF_SIZE,
            Box::new(ahub::admaif()),
        );
        bus.map(
            "ADMA",
            ahub::ADMA_BASE,
            ahub::ADMA_SIZE,
            Box::new(ahub::adma()),
        );
        bus.map(
            "HD Audio",
            hda::BASE,
            hda::SIZE,
            Box::new(hda::Hda::new(
                DeviceMemory::new(&soc.mc, ram, hda::SMMU_ASID),
                soc.line(hda::INTID),
                Arc::clone(&soc.notes),
            )),
        );
        let adsp_reset = Arc::clone(&adsp);
        let car = if power_on {
            Car::power_on()
        } else {
            Car::new()
        };
        let car = car
            .on_emc_clock_change(Box::new(move || clock_change.clock_changed()))
            // RST_DEVICES_L bits 27 and 26: DISP1 and DISP2 (hekate `clock.h`).
            .on_reset_asserted(0x004, 27, Box::new(move || disp1.reset()))
            .on_reset_asserted(0x004, 26, Box::new(move || disp2.reset()))
            .on_reset_asserted(
                adsp::RESET_REGISTER,
                adsp::RESET_BIT,
                Box::new(move || adsp_reset.reset()),
            )
            .on_reset_released(
                adsp::RESET_REGISTER,
                adsp::RESET_BIT,
                Box::new(move || adsp.release()),
            );
        let car = match &soc.boot {
            Some(boot) => {
                let signal = |set: fn(&BootSignals)| {
                    let signals = Arc::clone(&boot.signals);
                    Box::new(move || set(&signals))
                };
                bus.map(
                    "IRAM",
                    iram::BASE,
                    iram::SIZE,
                    Box::new(Arc::clone(&boot.iram)),
                );
                bus.map(
                    "address space after IRAM",
                    iram::RESERVED_BASE,
                    iram::RESERVED_SIZE,
                    Box::new(iram::AfterIram::new(Arc::clone(&soc.notes))),
                );
                bus.map(
                    "BPMP cache controller",
                    bpmp_cache::BASE,
                    bpmp_cache::SIZE,
                    Box::new(bpmp_cache::BpmpCache::new()),
                );
                bus.map(
                    "bus arbitration priorities",
                    ARB_PRI_BASE,
                    ARB_PRI_SIZE,
                    Box::new(RegisterFile::strict([])),
                );
                bus.map(
                    "TZRAM",
                    tzram::BASE,
                    tzram::SIZE,
                    Box::new(Arc::clone(&boot.tzram)),
                );
                bus.map(
                    "flow controller",
                    flow::BASE,
                    flow::SIZE,
                    Box::new(Arc::clone(&boot.flow)),
                );
                bus.map(
                    "legacy interrupt controller",
                    ictlr::BASE,
                    ictlr::SIZE,
                    Box::new(Arc::clone(&boot.ictlr)),
                );
                bus.map(
                    "exception vectors",
                    evp::BASE,
                    evp::SIZE,
                    Box::new(Arc::clone(&boot.evp)),
                );
                bus.map(
                    "secure boot registers",
                    sb::BASE,
                    sb::SIZE,
                    Box::new(Arc::clone(&boot.sb)),
                );
                bus.map(
                    "processor tag",
                    pg_up::BASE,
                    pg_up::SIZE,
                    Box::new(pg_up::PgUp),
                );
                bus.map(
                    "AHB arbitration",
                    ahb::BASE,
                    ahb::SIZE,
                    Box::new(ahb::power_on()),
                );
                for (channel, base) in mc_channel::CHANNEL_BASES.into_iter().enumerate() {
                    let aperture = mc_channel::McChannel::new(&soc.mc, channel)
                        .expect("a powered-on memory controller has channel apertures");
                    bus.map(
                        ["MC0", "MC1"][channel],
                        base,
                        mc_channel::SIZE,
                        Box::new(aperture),
                    );
                }
                bus.map(
                    "SYSCTR0",
                    sysctr0::BASE,
                    sysctr0::SIZE,
                    Box::new(sysctr0::Sysctr0::power_on()),
                );
                let mselect = Arc::new(mselect::Mselect::power_on());
                bus.map(
                    "MSELECT",
                    mselect::BASE,
                    mselect::SIZE,
                    Box::new(Arc::clone(&mselect)),
                );
                let (held, released) = (Arc::clone(&mselect), mselect);
                car.on_cpu0_released(signal(|s| s.cpu0_released.store(true, Ordering::Release)))
                    .on_reset_asserted(
                        mselect::RESET.0,
                        mselect::RESET.1,
                        Box::new(move || held.hold()),
                    )
                    .on_reset_released(
                        mselect::RESET.0,
                        mselect::RESET.1,
                        Box::new(move || released.release()),
                    )
                    .on_reset_asserted(
                        COP_RESET.0,
                        COP_RESET.1,
                        signal(|s| s.cop_held.store(true, Ordering::Release)),
                    )
                    .on_reset_released(
                        COP_RESET.0,
                        COP_RESET.1,
                        signal(|s| {
                            s.cop_held.store(false, Ordering::Release);
                            s.cop_restarted.store(true, Ordering::Release);
                        }),
                    )
            }
            None => car,
        };
        bus.map("clock and reset", car::BASE, car::SIZE, Box::new(car));
        bus.map(
            "CL-DVFS",
            cldvfs::BASE,
            cldvfs::SIZE,
            Box::new(ClDvfs::new()),
        );
        bus.map(
            "EMC",
            emc::BASE,
            emc::SIZE,
            Box::new(EmcPort::new(&emc, View::Broadcast)),
        );
        for (channel, base) in emc::CHANNEL_BASES.into_iter().enumerate() {
            let port = EmcPort::new(&emc, View::Channel(channel));
            bus.map(["EMC0", "EMC1"][channel], base, emc::SIZE, Box::new(port));
        }
        if let Some(boot) = &soc.boot {
            bus.map(
                "timers",
                tmr::BASE,
                tmr::SIZE,
                Box::new(Arc::clone(&boot.timers)),
            );
        } else {
            bus.map(
                "timers",
                tmr::BASE,
                tmr::SIZE,
                Box::new(Timers::new(elapsed(clock))),
            );
        }
        bus.map(
            "RTC",
            rtc::BASE,
            rtc::SIZE,
            Box::new(rtc::Rtc::new(elapsed(clock))),
        );
        let measurements = [
            FUSE_CPU_SPEEDO_0_CALIB,
            FUSE_CPU_IDDQ_CALIB,
            FUSE_CPU_SPEEDO_2_CALIB,
            FUSE_SOC_SPEEDO_0_CALIB,
            FUSE_SOC_IDDQ_CALIB,
            FUSE_GPU_IDDQ_CALIB,
            FUSE_OPT_PRIV_SEC_DIS,
            FUSE_RESERVED_CALIB0,
            FUSE_USB_CALIB,
            FUSE_USB_CALIB_EXT,
        ];
        let configuration = params.dram.dram_id().map(|id| {
            let id = id as u32;
            let dram_id = (id & 0x1F) << 3 | (id >> 5 & 7) << 12;
            (FUSE_RESERVED_ODM4, ODM4_ICOSA_PRODUCTION | dram_id)
        });
        let console = [
            (FUSE_SOC_SPEEDO_1_CALIB, BOOTROM_PATCH_VERSION),
            (FUSE_RESERVED_ODM6, 0),
            (FUSE_RESERVED_ODM7, ANTI_DOWNGRADE_FUSES),
            (FUSE_SPARE_BIT_5, 1),
        ];
        let registers = measurements
            .map(|fuse| (fuse, 0))
            .into_iter()
            .chain(console)
            .chain(FUSES_THE_BOOTLOADER_CHECKS)
            .chain(configuration);
        let fuse = Arc::new(match &start {
            Start::Reset { fuses, .. } | Start::AfterRom { fuses } => {
                let (array, size) = fuse::patch_records();
                let fuse = fuse::Fuse::new(
                    registers
                        .chain(fuses.at_reset())
                        .chain([(fuse::FIRST_BOOTROM_PATCH_SIZE, size)]),
                    array,
                );
                if matches!(start, Start::AfterRom { .. }) {
                    // The ROM's last word on the fuses: the private key
                    // hidden.
                    let hide = (fuse::PRIVATEKEYDISABLE, 1);
                    crate::bus::MmioDevice::write(
                        &fuse,
                        crate::boot::CORE_COUNT,
                        hide.0,
                        4,
                        hide.1,
                    )
                    .expect("the register exists");
                }
                fuse
            }
            Start::Kernel => {
                fuse::Fuse::new(registers.chain(crate::bootrom::fuses_after_boot()), [])
            }
        });
        bus.map("fuses", FUSE_BASE, FUSE_SIZE, Box::new(Arc::clone(&fuse)));
        soc.fuse
            .set(fuse)
            .ok()
            .expect("the fuse block is built once");
        let gauge: Box<dyn I2cDevice> =
            Box::new(Max17050::new(Battery::FULL, Arc::clone(&soc.notes)));
        let i2c1 = [
            soc.byte_registers("charger", bq24193::ADDRESS, Bq24193::new()),
            soc.byte_registers(
                "temperature sensor",
                tmp451::ADDRESS,
                tmp451::Tmp451::default(),
            ),
            (max17050::ADDRESS, gauge),
            (
                m92t36::ADDRESS,
                Box::new(m92t36::M92t36::new(&soc.gpio, Arc::clone(&soc.notes))),
            ),
            (
                alc5639::ADDRESS,
                Box::new(alc5639::Alc5639::new(Arc::clone(&soc.notes))),
            ),
        ];
        bus.map(
            "I2C1",
            I2C1_BASE,
            i2c::SIZE,
            Box::new(soc.i2c("I2C1", I2C1_INTID, i2c1)),
        );
        // Tegra210 DTS: I2C2 SPI 84, I2C3 SPI 92, SPI4 SPI 93.
        let light: Box<dyn I2cDevice> = Box::new(bh1730::Bh1730::new(
            Arc::clone(&soc.clock),
            Arc::clone(&soc.notes),
        ));
        bus.map(
            "I2C2",
            0x7000_C400,
            i2c::SIZE,
            Box::new(soc.i2c("I2C2", 32 + 84, [(bh1730::ADDRESS, light)])),
        );
        let touch: Box<dyn I2cDevice> = Box::new(touch::Touch::new(
            Arc::clone(&soc.touch_input),
            Arc::clone(&soc.notes),
        ));
        bus.map(
            "I2C3",
            0x7000_C500,
            i2c::SIZE,
            Box::new(soc.i2c("I2C3", 32 + 92, [(touch::ADDRESS, touch)])),
        );
        bus.map(
            "SPI4",
            spi::BASE,
            spi::SIZE,
            Box::new(spi::Spi::new(soc.line(spi::INTID), Arc::clone(&soc.notes))),
        );
        let i2c5 = [
            soc.byte_registers("PMIC", max77620::PMIC_ADDRESS, soc.pmic.power_block()),
            soc.byte_registers("RTC", max77620::RTC_ADDRESS, soc.pmic.clock_block()),
            soc.byte_registers("CPU regulator", max77621::CPU_ADDRESS, Max77621::cpu()),
            soc.byte_registers("GPU regulator", max77621::GPU_ADDRESS, Max77621::gpu()),
        ];
        bus.map(
            "I2C5",
            I2C5_BASE,
            i2c::SIZE,
            Box::new(soc.i2c("I2C5", I2C5_INTID, i2c5)),
        );
        let emmc = emmc
            .map(|images| Emmc::new(images, Arc::clone(&soc.notes)))
            .transpose()?
            .map(|emmc| Box::new(emmc) as Box<dyn Card>);
        let sdmmc4 = Sdmmc::new(
            soc.line(SDMMC4_INTID),
            DeviceMemory::new(&soc.mc, ram, mc::SMMU_SDMMC4A_ASID),
            emmc,
            sdmmc::Timing::EMMC,
            Arc::clone(&soc.notes),
            Arc::clone(&soc.clock),
        );
        bus.map("SDMMC4", SDMMC4_BASE, sdmmc::SIZE, Box::new(sdmmc4));
        let sdmmc2 = Sdmmc::new(
            soc.line(SDMMC2_INTID),
            DeviceMemory::new(&soc.mc, ram, mc::SMMU_SDMMC2A_ASID),
            None,
            sdmmc::Timing::INSTANT,
            Arc::clone(&soc.notes),
            Arc::clone(&soc.clock),
        );
        bus.map("SDMMC2", SDMMC2_BASE, sdmmc::SIZE, Box::new(sdmmc2));
        let sd_card = sd_card
            .map(|image| Box::new(SdCard::new(image, Arc::clone(&soc.notes))) as Box<dyn Card>);
        let sdmmc1 = Sdmmc::new(
            soc.line(SDMMC1_INTID),
            DeviceMemory::new(&soc.mc, ram, mc::SMMU_SDMMC1A_ASID),
            sd_card,
            sdmmc::Timing::SD_CARD,
            Arc::clone(&soc.notes),
            Arc::clone(&soc.clock),
        );
        let supply = sdmmc1.card_supply();
        soc.gpio.observe_output(Box::new(move |port, pin, high| {
            if (port, pin) == SD_CARD_SUPPLY {
                supply(high);
            }
        }));
        bus.map("SDMMC1", SDMMC1_BASE, sdmmc::SIZE, Box::new(sdmmc1));
        Ok(soc)
    }

    /// Slides both Joy-Con onto their rails, or off them. Each pulls its
    /// rail's detect pin low while it is on, and its CTS line
    /// ([`joycon::Controls::cts_high`]).
    pub fn seat_joycons(&self, seated: bool) {
        self.joycons.set_seated(seated);
        for (port, pin) in [joycon::LEFT_DETECT, joycon::RIGHT_DETECT] {
            // Both ports have their inputs listed.
            let _ = self.gpio.set_input(port, pin, !seated);
        }
    }

    /// The SoC enters deep power down (SC7): the PMC takes the wake level
    /// of its "auto" events from their pads ([`Pmc::sample_wake_levels`]).
    ///
    /// USB1 loses its settings with the SoC's power. In a machine that
    /// runs the boot ROM itself, so do the other blocks the ROM sets up,
    /// for the ROM to find them as at power-on when it runs again at the
    /// wake: the fuse block (the private key fuses readable again), the
    /// patch unit and the security engine, whose context the ROM restores
    /// from the monitor's save.
    pub fn enter_sc7(&self) {
        self.pmc.sample_wake_levels(|event| self.wake_pad(event));
        if let Some(boot) = &self.boot {
            boot.usb2.power_off();
        }
        if let Some(rom) = self.boot.as_ref().and_then(|boot| boot.rom.as_ref()) {
            rom.irom.power_off();
            self.se.power_off();
            if let Some(fuse) = self.fuse.get() {
                fuse.power_off();
            }
        }
    }

    /// Whether something wakes the sleeping console, which the power
    /// management controller then has on record. Three wake sources are
    /// modeled: the power management IC's interrupt output, which carries
    /// the power key and the clock's alarms, and the rails' CTS lines,
    /// which a sleeping Joy-Con's HOME button changes. The radios, the
    /// charger, the card slot and the dock are wake sources of the console
    /// that are not.
    pub fn wake_pending(&self) -> bool {
        WAKE_EVENTS.into_iter().fold(false, |woke, event| {
            let wakes = self
                .wake_pad(event)
                .is_some_and(|high| self.pmc.wake(event, high));
            woke | wakes
        })
    }

    /// The level of the pad of wake event `event`, if it is one of
    /// [`WAKE_EVENTS`].
    fn wake_pad(&self, event: u8) -> Option<bool> {
        let (port, pin) = match event {
            pmc::WAKE_EVENT_PMIC => return Some(!self.pmic.interrupt()),
            pmc::WAKE_EVENT_UART2_CTS => joycon::RIGHT_CTS,
            pmc::WAKE_EVENT_UART3_CTS => joycon::LEFT_CTS,
            _ => return None,
        };
        self.gpio.input(port, pin)
    }

    /// Parts with state that no bus device saves, keyed by a name unique
    /// within the SoC: devices reached only through [`Soc`]'s fields, and
    /// state a bus device's [`MmioDevice::snapshot`](crate::bus::MmioDevice::snapshot)
    /// leaves to another part.
    pub fn snapshot_parts(&self) -> Vec<(&'static str, &dyn crate::snapshot::Snapshot)> {
        let mut parts: Vec<(&'static str, &dyn crate::snapshot::Snapshot)> = Vec::new();
        // The WiFi function: the PCIe root complex reaches it only as a
        // `PcieFunction`.
        if let Some(wifi) = self.wifi.get() {
            parts.push(("BCM4356 WiFi", wifi.as_ref()));
        }
        // The GPU: its BAR windows on the bus answer `NoState`.
        parts.push(("GM20B", &self.gpu));
        // The PMIC and its clock: both its I2C addresses answer `NoState`.
        parts.push(("MAX77620", &self.pmic));
        // The touch controller's events and modes, and the finger held on
        // the glass: shared by the I2C3 device and the host's input.
        parts.push(("touch input", self.touch_input.as_ref()));
        // Whether the Joy-Con are on their rails and what is held on them.
        parts.push(("Joy-Con controls", self.joycons.as_ref()));
        parts
    }

    /// Where each GPU channel stands, a line each.
    pub fn gpu_channels(&self) -> Vec<String> {
        self.gpu.channels()
    }

    /// The screen as it is right now.
    pub fn screen(&self) -> Result<Screen, NoFrame> {
        let frame = self.dc.frame_turned_clockwise(&self.display_memory)?;
        let Panel {
            panel_on,
            backlight_on,
        } = self.panel();
        Ok(Screen {
            frame,
            panel_on,
            backlight_on,
        })
    }

    /// Point `(x, y)` of the screen's picture right now, worked out alone;
    /// see [`Dc::pixel_turned_clockwise`].
    pub fn screen_pixel(&self, x: u32, y: u32) -> Result<Pixel, NoFrame> {
        self.dc.pixel_turned_clockwise(&self.display_memory, x, y)
    }

    /// How the panel and its backlight stand right now.
    pub fn panel(&self) -> Panel {
        let powered = self.gpio.output(BACKLIGHT_PORT, BACKLIGHT_ENABLE_PIN) == Some(true);
        let dimmer = self.pwm.get(0).unwrap_or(0);
        Panel {
            panel_on: self.dsi.panel_is_showing(),
            backlight_on: powered && dimmer & PWM_ENABLE != 0 && dimmer & PWM_WIDTH != 0,
        }
    }

    /// Pictures the panel's display controller has been given to show
    /// ([`dc::Dc::flips`]).
    pub fn display_flips(&self) -> u64 {
        self.dc.flips()
    }

    /// Keeps the display controllers' frames going until `stop` is set; see
    /// [`dc::keep_time`].
    pub fn keep_time(&self, stop: &AtomicBool) {
        dc::keep_time(&[&self.dc, &self.dcb], &self.clock, stop);
    }

    /// The input line of shared interrupt `intid` at the interrupt controller.
    fn line(&self, intid: u32) -> Box<dyn Fn(bool) + Send + Sync> {
        line(&self.gic, intid)
    }

    fn i2c(
        &self,
        name: &'static str,
        intid: u32,
        devices: impl IntoIterator<Item = (u8, Box<dyn I2cDevice>)>,
    ) -> I2cController {
        I2cController::new(name, devices, self.line(intid), Arc::clone(&self.notes))
    }

    /// A bus entry for a device made of byte registers.
    fn byte_registers(
        &self,
        name: &'static str,
        address: u8,
        registers: impl RegisterMap + 'static,
    ) -> (u8, Box<dyn I2cDevice>) {
        (
            address,
            Box::new(ByteRegisters::new(name, registers, Arc::clone(&self.notes))),
        )
    }
}

/// The input line of shared interrupt `intid` at the interrupt controller.
fn line(gic: &Arc<Gic>, intid: u32) -> Box<dyn Fn(bool) + Send + Sync> {
    let gic = Arc::clone(gic);
    Box::new(move |level| gic.set_line(0, intid, level))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        boot::{CORE_COUNT, DRAM_BASE},
        bus::MmioDevice,
    };

    #[test]
    fn the_panel_reads_lit_while_the_display_controller_composes_no_picture() {
        let params = BootParams::default();
        let gic = Arc::new(Gic::new(CORE_COUNT, Box::new(|_| {})));
        let ram = Arc::new(GuestRam::new(DRAM_BASE, params.dram.bytes() as usize).unwrap());
        let clock = Arc::new(GuestClock::new());
        let soc = Soc::new(
            &params,
            &gic,
            &ram,
            &clock,
            None,
            None,
            Start::Kernel,
            &mut Bus::default(),
        )
        .unwrap();
        let dark = Panel {
            panel_on: false,
            backlight_on: false,
        };
        assert_eq!(soc.panel(), dark);

        // DSI: controller on, then exit sleep and display on, as short
        // packets through the write FIFO (`WR_DATA`, `TRIGGER`).
        let dsi = |offset: u64, value: u32| soc.dsi.write(0, offset, 4, value.into()).unwrap();
        dsi(0x2C, 1);
        for packet in [0x1105, 0x2905] {
            dsi(0x28, packet);
            dsi(0x4C, 1 << 1);
        }
        // The backlight driver's enable, GPIO port V pin 1 (controller 5,
        // port 1), driven high, and the dimmer at full duty.
        let pin = 1 << BACKLIGHT_ENABLE_PIN;
        for row in [0x00, 0x10, 0x20] {
            soc.gpio.write(0, 0x504 + row, 4, pin).unwrap();
        }
        soc.pwm.set(0, PWM_ENABLE | PWM_WIDTH);

        assert!(
            soc.screen().is_err(),
            "the display controller was never set up"
        );
        assert_eq!(
            soc.panel(),
            Panel {
                panel_on: true,
                backlight_on: true,
            }
        );
    }

    #[test]
    fn a_change_of_a_rail_cts_line_after_sc7_entry_wakes_the_console() {
        let params = BootParams::default();
        let gic = Arc::new(Gic::new(CORE_COUNT, Box::new(|_| {})));
        let ram = Arc::new(GuestRam::new(DRAM_BASE, params.dram.bytes() as usize).unwrap());
        let clock = Arc::new(GuestClock::new());
        let soc = Soc::new(
            &params,
            &gic,
            &ram,
            &clock,
            None,
            None,
            Start::Kernel,
            &mut Bus::default(),
        )
        .unwrap();
        soc.seat_joycons(true);
        let cts = |(port, pin)| soc.gpio.input(port, pin).unwrap();
        assert!(!cts(joycon::RIGHT_CTS) && !cts(joycon::LEFT_CTS));
        // The firmware's wake set-up: the rail CTS events enabled, level
        // "auto" (WAKE_MASK, AUTO_WAKE_LVL_MASK), PwrIntN enabled.
        let pmc = |offset: u64, value: u32| soc.pmc.write(0, offset, 4, value.into()).unwrap();
        pmc(0x0C, 1 << 6 | 1 << 7);
        pmc(0xDC, 1 << 6 | 1 << 7);
        pmc(0x160, 1 << 19);
        soc.enter_sc7();
        assert!(!soc.wake_pending(), "nothing changed");
        // Buttons of an awake Joy-Con do not touch the line.
        soc.joycons.press(joycon::Button::HOME, true);
        assert!(!soc.wake_pending());
        soc.joycons.press(joycon::Button::HOME, false);
        // Taking the Joy-Con off releases both lines: both events record.
        soc.seat_joycons(false);
        assert!(soc.wake_pending());
        assert_eq!(soc.pmc.read(0, 0x14, 4), Some(1 << 6 | 1 << 7));
    }
}
