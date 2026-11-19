//! Tegra X1 blocks outside the CPU cluster, and the board devices wired to
//! them, as far as the kernel and its initial processes use them.

pub mod adsp;
pub mod agic;
pub mod ahub;
pub mod alc5639;
pub mod apb_dma;
pub mod apb_security;
pub mod bcm4356;
pub mod bh1730;
pub mod bluetooth_uart;
pub mod bq24193;
pub mod car;
pub mod cldvfs;
pub mod dc;
pub mod dot11;
pub mod dpaux;
pub mod dsi;
pub mod emc;
pub mod emmc;
pub mod falcon_host;
pub mod gm20b;
pub mod gmix;
pub mod gpio;
pub mod hda;
pub mod host1x;
pub mod host1x_channel;
pub mod i2c;
pub mod joycon;
pub mod lsm6ds3;
pub mod m92t36;
pub mod max17050;
pub mod max77620;
pub mod max77621;
pub mod mc;
mod medium;
pub mod mipi_cal;
pub mod ntag;
pub mod pcie;
pub mod pcie_endpoint;
pub mod pmc;
pub mod regs;
pub mod sdcard;
pub mod sdmmc;
pub mod serial;
pub mod soctherm;
pub mod spi;
pub mod tmp451;
pub mod tmr;
pub mod touch;
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
    dc::{Dc, Frame, NoFrame},
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

/// Pin multiplexer: one configuration register per pin. `boot`'s pin
/// driver reads all 175 into its tables, clearing any park bit it finds
/// (text `0x2e9e0`), and later programs each pin's function, pull,
/// tristate and input enable from its own configuration. Like the other
/// pure configuration blocks the registers start at zero.
const PINMUX_BASE: u64 = 0x7000_3000;
const PINMUX_SIZE: u64 = 0x300;

/// Pad control groups: drive strength and slew rate per group of pins,
/// the part of the miscellaneous block the same driver reads into its
/// tables (text `0x2eaac`) before it programs 47 of the groups.
/// Configuration that starts at zero, like the pin multiplexer.
const PAD_GROUPS_BASE: u64 = 0x7000_08E4;
const PAD_GROUPS_SIZE: u64 = 0x28C;

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
const SDMMC2_BASE: u64 = 0x700B_0200;
const SDMMC2_INTID: u32 = 32 + 15;

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

/// `APB_MISC_GP_HIDREV`: which chip this is. The firmware's clock driver
/// tells the original console's SoC from its successor by the major
/// revision in bits 7:4, where 2 is the successor (pcv text `0x50cc0`).
/// Here: chip ID 0x21 in bits 15:8 and major revision 1, as publicly
/// documented for this SoC. The minor revision (bits 19:16) and the family
/// (bits 3:0) of a console's chip are not known here and read zero, which
/// is an assumption.
const HIDREV_BASE: u64 = 0x7000_0804;
const HIDREV: u32 = 0x21 << 8 | 1 << 4;

/// The fuse block, as far as it is read: the chip's speed and leakage
/// measurements from the factory, by which the firmware's clock driver
/// picks voltages (pcv reads them once, in this order). Register names as
/// publicly documented. A measurement belongs to one particular chip, and
/// this console has none: they read zero, the slowest and least leaky chip
/// there could be. Like the device ID, that is a choice, not a fact.
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

/// What the console's screen shows.
pub struct Screen {
    /// The picture as someone holding the console sees it. The panel is a
    /// portrait one mounted on its side, its first scan line along the
    /// console's right edge, so this is the display controller's picture
    /// turned a quarter turn clockwise.
    pub frame: Frame,
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
    /// microSD card; without them their slots are empty.
    pub fn new(
        params: &BootParams,
        gic: &Arc<Gic>,
        ram: &Arc<GuestRam>,
        clock: &Arc<GuestClock>,
        emmc: Option<EmmcImages>,
        sd_card: Option<SdCardImage>,
        bus: &mut Bus,
    ) -> Result<Self, ImageError> {
        let mc = Arc::new(Mc::new(params.dram));
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
        for channel in 0..7 {
            let vic = Arc::clone(&vic);
            bus.map(
                "host1x channel",
                0x5000_0000 + channel * 0x4000,
                0x100,
                Box::new(host1x_channel::Channel::new(
                    DeviceMemory::new(&mc, ram, mc::SMMU_HOST1X_ASID),
                    Arc::clone(&syncpoints),
                    Arc::clone(clock),
                    Arc::new(move |class, offset, value| vic.method(class, offset, value)),
                    Arc::clone(&notes),
                )),
            );
        }
        let gpio = Arc::new(Gpio::with_interrupts(
            GPIO_INPUTS,
            gpio::INTIDS.map(|intid| line(gic, intid)),
        ));
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
        let soc = Self {
            gpio,
            touch_input,
            joycons: Arc::default(),
            pmic: max77620::Max77620::new(Arc::clone(clock), line(gic, PMU_INTID)),
            pmc: Arc::new(Pmc::new()),
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
        let host1x = Host1xSync::new(syncpoints, vec![Arc::clone(&soc.dc), Arc::clone(&soc.dcb)]);
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
            PINMUX_BASE,
            PINMUX_SIZE,
            Box::new(RegisterFile::zeroed([])),
        );
        bus.map(
            "pad groups",
            PAD_GROUPS_BASE,
            PAD_GROUPS_SIZE,
            Box::new(RegisterFile::zeroed([])),
        );
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
        let emc: Arc<emc::Emc> = Arc::default();
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
        let car = Car::new()
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
        bus.map(
            "timers",
            tmr::BASE,
            tmr::SIZE,
            Box::new(Timers::new(elapsed(clock))),
        );
        bus.map(
            "chip ID",
            HIDREV_BASE,
            4,
            Box::new(RegisterFile::strict([(0, HIDREV)])),
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
        bus.map(
            "fuses",
            FUSE_BASE,
            FUSE_SIZE,
            Box::new(RegisterFile::strict(measurements.map(|fuse| (fuse, 0)))),
        );
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
    /// rail's detect pin low while it is on.
    pub fn seat_joycons(&self, seated: bool) {
        self.joycons.set_seated(seated);
        for (port, pin) in [
            joycon::LEFT_DETECT,
            joycon::RIGHT_DETECT,
            joycon::LEFT_CTS,
            joycon::RIGHT_CTS,
        ] {
            // Both ports have their inputs listed.
            let _ = self.gpio.set_input(port, pin, !seated);
        }
    }

    /// Whether something wakes the sleeping console, which the power
    /// management controller then has on record. One wake source is
    /// modeled: the power management IC's interrupt output, which carries
    /// the power key and the clock's alarms. A Joy-Con on a rail, the
    /// radios, the charger, the card slot and the dock are wake sources of
    /// the console that are not.
    pub fn wake_pending(&self) -> bool {
        self.pmic.interrupt() && self.pmc.wake(pmc::WAKE_EVENT_PMIC, false)
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
        let powered = self.gpio.output(BACKLIGHT_PORT, BACKLIGHT_ENABLE_PIN) == Some(true);
        let dimmer = self.pwm.get(0).unwrap_or(0);
        Ok(Screen {
            frame: self.dc.frame_turned_clockwise(&self.display_memory)?,
            panel_on: self.dsi.panel_is_showing(),
            backlight_on: powered && dimmer & PWM_ENABLE != 0 && dimmer & PWM_WIDTH != 0,
        })
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
