//! Tegra210 PCIe root ports, PADS and AFI, and the routes between the CPU,
//! the endpoint function behind root port 1 and DRAM.
//!
//! Firmware evidence: `pcie` text 0x1adc0 maps the root ports at +0/+0x1000,
//! PADS at +0x3000 and AFI at +0x3800, the downstream configuration aperture
//! at 0x0200_0000 (256 MiB) and the memory windows from 0x1200_0000. Text
//! 0x1a340 programs AFI; 0x1a500 and 0x1aab0 program the root-port PHY and
//! bridge windows; 0x1af10 registers GIC interrupts 130 ("intr", SPI 98),
//! 131 (MSI, SPI 99) and 132. Register names and bits are corroborated by
//! the public Linux driver
//! <https://github.com/torvalds/linux/blob/master/drivers/pci/controller/pci-tegra.c>.
//! Configuration fields start zero (an assumption about unused reset fields),
//! but unknown offsets refuse access.
//!
//! The link: a port trains once its AFI_PEXn_CTRL reset (bit 0) is released
//! and a function is attached behind it — the WiFi function of the on-board
//! combo chip on port 1, the x1 port ([`WIFI_PORT`]): `pcie` releases only
//! AFI_PEX1_CTRL's reset and polls port 1, leaving port 0 in reset (traced
//! accesses to +0x3910/+0x3918). `pcie` 0x19ec0 then sees DL_UP
//! (+0xf00 bit 30) and Data Link Layer Link Active (+0x90 bit 29). A port
//! without a function never trains: 0x19ec0 times out with result 632 and
//! its caller 0x1ccc0 carries on.
//!
//! Outbound: CPU accesses in the memory windows go through the AFI AXI BARs
//! (AFI_AXI_BARn_SZ/START, AFI_FPCI_BARn, as Linux `tegra_pcie_setup_translations`
//! programs them) to FPCI = PCI bus addresses. Inbound: the function's
//! memory requests land in DRAM through the AFI's SMMU client, except writes
//! into the MSI window (AFI_MSI_FPCI_BAR_ST), which set the vector bit in
//! AFI_MSI_VECn.

use super::{
    mc::DeviceMemory,
    pcie_endpoint::{PcieFunction, Upstream},
};
use crate::bus::MmioDevice;
use std::sync::{Arc, Mutex, OnceLock};

pub const BASE: u64 = 0x0100_0000;
pub const SIZE: u64 = 0x4000;
/// Downstream configuration aperture (`pcie` text 0x1adc0).
pub const CONFIG_BASE: u64 = 0x0200_0000;
pub const CONFIG_SIZE: u64 = 0x1000_0000;
/// Memory windows, up to the end of the 1 GiB PCIe aperture.
pub const MEMORY_BASE: u64 = 0x1200_0000;
pub const MEMORY_SIZE: u64 = 0x4000_0000 - MEMORY_BASE;
/// AFI "intr" interrupt, which also carries legacy INTA-INTD (public Linux
/// `tegra210.dtsi` interrupt-map), and the MSI interrupt (`pcie` 0x1af10).
pub const INTR_INTID: u32 = 32 + 98;
pub const MSI_INTID: u32 = 32 + 99;
/// The AFI's SMMU client ASID register (public Linux `tegra210.c` swgroup
/// table, `afi`).
pub const SMMU_ASID: u64 = 0x238;
/// The root port the WiFi function sits behind.
pub const WIFI_PORT: usize = 1;

/// AFI register offsets within this block (AFI at +0x3800).
const AFI: usize = 0x3800;
const AFI_AXI_BAR_SZ: usize = AFI;
const AFI_AXI_BAR_START: usize = AFI + 0x18;
const AFI_FPCI_BAR: usize = AFI + 0x30;
const AFI_MSI_BAR_SZ: usize = AFI + 0x60;
const AFI_MSI_FPCI_BAR_ST: usize = AFI + 0x64;
const AFI_MSI_VEC: usize = AFI + 0x6c;
const AFI_MSI_EN_VEC: usize = AFI + 0x8c;
const AFI_INTR_MASK: usize = AFI + 0xb4;
const AFI_INTR_MASK_MSI: u32 = 1 << 8;
const AFI_PEX_CTRL: [usize; 2] = [AFI + 0x110, AFI + 0x118];
/// AFI_PCIE_PME (AFI + 0xf0): per port, a bit that sends PME_Turn_Off down
/// the link and one that reports the PME_TO_Ack coming back (Linux
/// `pci-tegra.c`, `tegra_pcie_pme_turnoff` and the port table this SoC
/// shares with Tegra20: turn-off bits 0 and 8, acknowledge bits 5 and 10).
/// `pcie` does this before the console sleeps (read at +0x38f0). A
/// function on a trained link acknowledges, as the specification requires
/// of every function (PCIe Base 5.3.3.2.1); a port with no link never
/// does.
const PME_TURN_OFF: [u32; 2] = [1 << 0, 1 << 8];
const PME_ACK: [u32; 2] = [1 << 5, 1 << 10];
/// FPCI addresses from here up are the AFI's own spaces (configuration,
/// I/O, MSI), not memory a function decodes (Linux's FPCI map).
const FPCI_SPECIAL: u64 = 0xfc_0000_0000;

struct Inner {
    registers: Mutex<[u32; 0x1000]>,
    function: OnceLock<Arc<dyn PcieFunction>>,
    memory: DeviceMemory,
    intr: Box<dyn Fn(bool) + Send + Sync>,
    msi: Box<dyn Fn(bool) + Send + Sync>,
}

impl Inner {
    fn reg(&self, offset: usize) -> u32 {
        self.registers.lock().unwrap()[offset / 4]
    }

    /// Whether root port `port` has a trained link.
    fn link_up(regs: &[u32; 0x1000], function: bool, port: usize) -> bool {
        port == WIFI_PORT && function && regs[AFI_PEX_CTRL[port] / 4] & 1 != 0
    }

    fn update_msi(&self, regs: &[u32; 0x1000]) {
        let pending = (0..8).any(|i| regs[AFI_MSI_VEC / 4 + i] & regs[AFI_MSI_EN_VEC / 4 + i] != 0);
        (self.msi)(pending && regs[AFI_INTR_MASK / 4] & AFI_INTR_MASK_MSI != 0);
    }

    /// The bus address of a CPU address in the memory windows, through the
    /// AFI AXI BAR that contains it.
    fn outbound(&self, cpu: u64) -> Option<u64> {
        let regs = self.registers.lock().unwrap();
        (0..6)
            .find_map(|n| {
                let size = u64::from(regs[AFI_AXI_BAR_SZ / 4 + n]) << 12;
                let start = u64::from(regs[AFI_AXI_BAR_START / 4 + n]);
                let fpci = u64::from(regs[AFI_FPCI_BAR / 4 + n]);
                (size != 0 && fpci & 1 != 0 && (start..start + size).contains(&cpu))
                    .then(|| ((fpci >> 4) << 12) + (cpu - start))
            })
            .filter(|&bus| bus < FPCI_SPECIAL)
    }
}

/// The root complex. Clones share one state; [`Pcie::configuration`] and
/// [`Pcie::memory_window`] are its other two apertures, [`Pcie::upstream`]
/// the side an attached function masters.
#[derive(Clone)]
pub struct Pcie(Arc<Inner>);

impl Pcie {
    pub fn new(
        memory: DeviceMemory,
        intr: Box<dyn Fn(bool) + Send + Sync>,
        msi: Box<dyn Fn(bool) + Send + Sync>,
    ) -> Self {
        Self(Arc::new(Inner {
            registers: Mutex::new([0; 0x1000]),
            function: OnceLock::new(),
            memory,
            intr,
            msi,
        }))
    }

    /// Puts `function` behind root port [`WIFI_PORT`].
    pub fn attach(&self, function: Arc<dyn PcieFunction>) {
        if self.0.function.set(function).is_err() {
            panic!("a function is already attached to the PCIe root port");
        }
    }

    pub fn upstream(&self) -> Box<dyn Upstream> {
        Box::new(RootUpstream(Arc::clone(&self.0)))
    }

    pub fn configuration(&self) -> DownstreamConfig {
        DownstreamConfig(Arc::clone(&self.0))
    }

    pub fn memory_window(&self) -> MemoryWindow {
        MemoryWindow(Arc::clone(&self.0))
    }

    fn known(offset: u64) -> bool {
        if offset < 0x2000 {
            matches!(
                offset & 0xfff,
                0x000..=0x03c
                    | 0x090
                    | 0xc00
                    | 0xc04
                    | 0xc08
                    | 0xc10..=0xc1c
                    | 0xe24
                    | 0xe28
                    | 0xe84
                    | 0xe8c
                    | 0xe90
                    | 0xe94
                    | 0xea4
                    | 0xeac
                    | 0xeb0
                    | 0xeb4
                    | 0xf00
                    | 0xf04
                    | 0xf30
                    | 0xf44
                    | 0xf48
                    | 0xf4c
                    | 0xf50
                    | 0xf8c
                    | 0xf90
                    | 0xf94
                    | 0xfa8
                    | 0xfe0
            )
        } else {
            matches!(offset, 0x30c8 | 0x30d0 | 0x3800..=0x3844 |
                0x3860..=0x38c8 | 0x38f0 | 0x38f8 | 0x3904 | 0x3910 | 0x3918 |
                0x3934..=0x3954 | 0x3960 | 0x3968)
        }
    }

    fn access(offset: u64, size: u8) -> Option<(usize, u32, u32)> {
        if !matches!(size, 1 | 2 | 4)
            || !offset.is_multiple_of(u64::from(size))
            || !Self::known(offset & !3)
        {
            return None;
        }
        let shift = (offset & 3) as u32 * 8;
        let mask = u32::MAX >> (32 - u32::from(size) * 8);
        Some(((offset / 4) as usize, shift, mask))
    }
}

/// Link status bits for a trained link (PCIe Base Specification Link Status
/// in the upper half of +0x90): Data Link Layer Link Active, negotiated
/// width x1, current speed 2.5 GT/s. The BCM4356 is a one-lane device; the
/// first training is at 2.5 GT/s, and `pcie` 0x19ec0 only maps the speed
/// field (1 or 2) to its own state.
const LINK_STATUS_UP: u32 = 1 << 29 | 1 << 20 | 1 << 16;

impl MmioDevice for Pcie {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        let (index, shift, mask) = Self::access(offset, size)?;
        let regs = self.0.registers.lock().unwrap();
        let up = |port| Inner::link_up(&regs, self.0.function.get().is_some(), port);
        // Public pci.ids: 10de:0fae x4 and 10de:0faf x1 Tegra X1 roots.
        let value = match offset & !3 {
            0 => 0x0fae_10de,
            0x1000 => 0x0faf_10de,
            // PCI bridge class and type-1 header. Revision zero is assumed.
            0x008 | 0x1008 => 0x0604_0000,
            0x00c | 0x100c => (regs[index] & 0xffff) | 0x0001_0000,
            // Root ports expose no local BAR aperture.
            0x010 | 0x1010 | 0x014 | 0x1014 => 0,
            0xf00 if up(0) => regs[index] | 1 << 30,
            0x1f00 if up(1) => regs[index] | 1 << 30,
            0x090 if up(0) => regs[index] | LINK_STATUS_UP,
            0x1090 if up(1) => regs[index] | LINK_STATUS_UP,
            0x38f0 => (0..2).fold(regs[index], |value, port| {
                let acknowledged = regs[index] & PME_TURN_OFF[port] != 0 && up(port);
                value & !PME_ACK[port] | if acknowledged { PME_ACK[port] } else { 0 }
            }),
            _ => regs[index],
        };
        Some(u64::from((value >> shift) & mask))
    }

    fn write(&self, _core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        let (index, shift, mask) = Self::access(offset, size)?;
        let mut regs = self.0.registers.lock().unwrap();
        let value = (regs[index] & !(mask << shift)) | (((value as u32) & mask) << shift);
        regs[index] = match offset & !3 {
            // DL_UP and the link status half are read-only.
            0xf00 | 0x1f00 => value & !(1 << 30),
            0x090 | 0x1090 => value & 0xffff,
            // PCI status is write-one-to-clear; command remains writable.
            0x004 | 0x1004 => value & 0xffff,
            // AFI_MSI_VECn are write-one-to-clear.
            0x386c..=0x3888 => regs[index] & !value,
            _ => value,
        };
        self.0.update_msi(&regs);
        Some(())
    }
}

/// Downstream configuration aperture: `pcie` text 0x1b000 encodes bus
/// (bits 16-23), device (11-15), function (8-10) and register (bits 2-7,
/// extended register number in bits 24-27). Requests for device 0 function
/// 0 on the secondary bus of a trained port reach the attached function;
/// all others receive no completion and read all ones, the configuration-
/// space convention for no responder.
pub struct DownstreamConfig(Arc<Inner>);

impl DownstreamConfig {
    fn target(&self, offset: u64) -> Option<(&dyn PcieFunction, u16)> {
        let function = self.0.function.get()?;
        let bus = ((offset >> 16) & 0xff) as u32;
        let regs = self.0.registers.lock().unwrap();
        let port = WIFI_PORT;
        let secondary = (regs[(port * 0x1000 + 0x18) / 4] >> 8) & 0xff;
        let reaches = Inner::link_up(&regs, true, port)
            && bus != 0
            && bus == secondary
            && (offset >> 8) & 0xff == 0;
        reaches.then(|| {
            let register = (offset & 0xff) | ((offset >> 16) & 0xf00);
            (function.as_ref(), register as u16)
        })
    }
}

impl MmioDevice for DownstreamConfig {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        if !(matches!(size, 1 | 2 | 4) && offset.is_multiple_of(u64::from(size))) {
            return None;
        }
        match self.target(offset) {
            Some((function, register)) => function.config_read(register, size).map(u64::from),
            None => Some(u64::from(u32::MAX >> (32 - u32::from(size) * 8))),
        }
    }

    fn write(&self, _core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        if !(matches!(size, 1 | 2 | 4) && offset.is_multiple_of(u64::from(size))) {
            return None;
        }
        match self.target(offset) {
            Some((function, register)) => function.config_write(register, size, value as u32),
            None => Some(()),
        }
    }
}

/// The memory windows: CPU address -> AFI AXI BAR -> bus address -> the
/// attached function's BARs. Addresses no window translates, and requests
/// no BAR claims, are refused.
pub struct MemoryWindow(Arc<Inner>);

impl MmioDevice for MemoryWindow {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        let bus = self.0.outbound(MEMORY_BASE + offset)?;
        self.0.function.get()?.memory_read(bus, size)
    }

    fn write(&self, _core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        let bus = self.0.outbound(MEMORY_BASE + offset)?;
        self.0.function.get()?.memory_write(bus, size, value)
    }
}

/// The attached function's view of the root complex.
struct RootUpstream(Arc<Inner>);

impl RootUpstream {
    /// The MSI vector a write at `address` signals, if it hits the MSI
    /// window: AFI_MSI_FPCI_BAR_ST holds the FPCI address >> 8 and
    /// AFI_MSI_BAR_SZ the size in 4 KiB pages (Linux programs both).
    fn msi_window(&self, address: u64, len: usize) -> bool {
        let start = u64::from(self.0.reg(AFI_MSI_FPCI_BAR_ST)) << 8;
        let size = u64::from(self.0.reg(AFI_MSI_BAR_SZ)) << 12;
        size != 0 && address >= start && address + len as u64 <= start + size
    }
}

impl Upstream for RootUpstream {
    fn read(&self, address: u64, buffer: &mut [u8]) -> Option<()> {
        if self.msi_window(address, buffer.len()) {
            return None;
        }
        self.0.memory.read(address, buffer)
    }

    fn write(&self, address: u64, data: &[u8]) -> Option<()> {
        if !self.msi_window(address, data.len()) {
            return self.0.memory.write(address, data);
        }
        // An MSI is a 32-bit write of the vector number (0..=255).
        let vector = u32::from_le_bytes(data.try_into().ok()?);
        if vector > 0xff {
            return None;
        }
        let mut regs = self.0.registers.lock().unwrap();
        regs[AFI_MSI_VEC / 4 + (vector / 32) as usize] |= 1 << (vector % 32);
        self.0.update_msi(&regs);
        Some(())
    }

    fn intx(&self, level: bool) {
        (self.0.intr)(level);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{boot::DramSize, memory::GuestRam, soc::mc::Mc};
    use std::sync::atomic::{AtomicBool, Ordering};

    fn root() -> (Pcie, Arc<AtomicBool>) {
        let ram = Arc::new(GuestRam::new(0x8000_0000, 0x4000).unwrap());
        let mc = Arc::new(Mc::new(DramSize::Gb4));
        let msi = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&msi);
        let pcie = Pcie::new(
            DeviceMemory::new(&mc, &ram, SMMU_ASID),
            Box::new(|_| {}),
            Box::new(move |level| flag.store(level, Ordering::Relaxed)),
        );
        (pcie, msi)
    }

    /// Answers configuration reads with the register number and BAR reads
    /// with the bus address.
    struct Echo;
    impl PcieFunction for Echo {
        fn config_read(&self, offset: u16, _size: u8) -> Option<u32> {
            Some(offset.into())
        }
        fn config_write(&self, _offset: u16, _size: u8, _value: u32) -> Option<()> {
            Some(())
        }
        fn memory_read(&self, address: u64, _size: u8) -> Option<u64> {
            Some(address)
        }
        fn memory_write(&self, _address: u64, _size: u8, _value: u64) -> Option<()> {
            None
        }
    }

    #[test]
    fn partial_writes_preserve_neighbors_and_link_is_read_only() {
        let (p, _) = root();
        p.write(0, 0x18, 4, 0x1234_5678).unwrap();
        p.write(0, 0x19, 1, 0xab).unwrap();
        assert_eq!(p.read(0, 0x18, 4), Some(0x1234_ab78));
        p.write(0, 0x1f00, 4, u32::MAX.into()).unwrap();
        assert_eq!(p.read(0, 0x1f00, 4).unwrap() & (1 << 30), 0);
        assert_eq!(p.read(0, 0x2000, 4), None);
    }

    #[test]
    fn link_trains_only_with_a_function_out_of_reset() {
        let (p, _) = root();
        let config = p.configuration();
        p.write(0, 0x3918, 4, 1).unwrap();
        assert_eq!(p.read(0, 0x1f00, 4).unwrap() & (1 << 30), 0);
        p.attach(Arc::new(Echo));
        assert_ne!(p.read(0, 0x1f00, 4).unwrap() & (1 << 30), 0);
        assert_eq!(p.read(0, 0x1092, 2), Some(0x2011));
        assert_eq!(p.read(0, 0xf00, 4).unwrap() & (1 << 30), 0);
        // Bus 1 is reached once the bridge's secondary bus says so.
        assert_eq!(config.read(0, 1 << 16 | 0x10, 4), Some(u64::from(u32::MAX)));
        p.write(0, 0x1019, 1, 1).unwrap();
        assert_eq!(config.read(0, 1 << 16 | 0x10, 4), Some(0x10));
        assert_eq!(config.read(0, 1 << 16 | 1 << 24 | 0x04, 4), Some(0x104));
        assert_eq!(
            config.read(0, 1 << 16 | 1 << 11, 4),
            Some(u64::from(u32::MAX))
        );
    }

    #[test]
    fn memory_window_translates_through_axi_and_fpci_bars() {
        let (p, _) = root();
        p.attach(Arc::new(Echo));
        let window = p.memory_window();
        // 1 MiB at CPU 0x1300_0000 to bus 0x1300_0000, Linux style.
        p.write(0, 0x3808, 4, 0x100).unwrap();
        p.write(0, 0x3820, 4, 0x1300_0000).unwrap();
        p.write(0, 0x3838, 4, (0x1300_0000 >> 12 << 4) | 1).unwrap();
        assert_eq!(window.read(0, 0x0100_0010, 4), Some(0x1300_0010));
        assert_eq!(window.read(0, 0, 4), None);
    }

    #[test]
    fn msi_write_sets_vector_and_raises_when_enabled() {
        let (p, msi) = root();
        let up = p.upstream();
        p.write(0, 0x3860, 4, 1).unwrap();
        p.write(0, 0x3864, 4, 0xf000_0000 >> 8).unwrap();
        p.write(0, 0x388c, 4, 1 << 3).unwrap();
        p.write(0, 0x38b4, 4, AFI_INTR_MASK_MSI.into()).unwrap();
        up.write(0xf000_0000, &3u32.to_le_bytes()).unwrap();
        assert!(msi.load(Ordering::Relaxed));
        assert_eq!(p.read(0, 0x386c, 4), Some(1 << 3));
        p.write(0, 0x386c, 4, 1 << 3).unwrap();
        assert!(!msi.load(Ordering::Relaxed));
    }
}
