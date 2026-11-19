//! USB1: the USB 2.0 device controller and its UTMI PHY at `0x7D00_0000`,
//! as far as the boot ROM sets them up.
//!
//! The boot ROM brings this port up on every cold boot, before it looks at
//! the boot device: it is the port of its recovery mode, and of its
//! charger detection (ROM `0x106CD0`, called through the table `0x102B74`
//! fills when `FUSE_RESERVED_SW` bit 7 selects the USB 2.0 controller, as
//! on an unpatched Erista; switchbrew "Fuses"). Nothing after the ROM
//! touches the block in a normal boot: the NX bootloader's check of the
//! hardware state has no row for it, and Horizon's `usb` uses the XUSB
//! device controller ([`super::xusb_device`]). The UTMI PLL the ROM
//! programs with it is the clock and reset controller's.
//!
//! Register names as the public drivers have them (U-Boot
//! `arch/arm/include/asm/arch-tegra/usb.h` and `ehci-tegra.c`, Linux
//! `drivers/usb/phy/phy-tegra-usb.c` and the ChipIdea core's
//! `include/linux/usb/chipidea.h`):
//!
//! | offset | register | here |
//! |---|---|---|
//! | `0x130` | `USBCMD` | storage; writing bit 1 (`RST`) resets the controller's registers (`0x130`..`0x21C`) to zero at once, and the bit reads clear, what the ROM waits for (`0x106B7C`) |
//! | `0x134` | `USBSTS` | write one to clear; no event is ever raised, reads 0 |
//! | `0x1F4` | `OTGSC` | the enables hold what is written; the status and status-change bits read 0: no cable, no session \[assumption: the state of a port nothing is plugged into\] |
//! | `0x208`, `0x218` | `ENDPTSETUPSTAT`, `ENDPTCOMPLETE` | write one to clear; read 0 |
//! | others in `0x130`..`0x21C` | | storage, zero after a controller reset |
//! | `0x400` | `USB_SUSP_CTRL` | storage; bit 7 (`USB_PHY_CLK_VALID`) is read-only and set while the PHY is enabled (bit 12, `UTMIP_PHY_ENB`) and out of reset (bit 11, `UTMIP_RESET`, clear) |
//! | `0x404`..`0x40C`, `0x800`..`0x85C` | VBUS sensors and wake-up, the `UTMIP_*` configuration registers | storage |
//!
//! Every register reads zero until written \[assumption\]: the reset
//! values are in NVIDIA's reference manual, not in a public driver, and
//! the ROM changes the configuration registers by read-modify-write and
//! never reads them for a decision. The two waits the ROM makes are the
//! ones above. The PHY clock is valid as soon as the conditions hold; the
//! ROM polls for it once a microsecond.
//!
//! Not modelled, and refused: any other offset, and running the
//! controller (`USBCMD` bit 0, `RS`): recovery mode needs a host on the
//! other end of the cable.

use std::{collections::HashMap, sync::Mutex};

use super::regs::word;
use crate::{
    bus::MmioDevice,
    snapshot::{Snapshot, SnapshotError, decode, encode},
};

pub const BASE: u64 = 0x7D00_0000;
pub const SIZE: u64 = 0x1000;

const USBCMD: u64 = 0x130;
const USBSTS: u64 = 0x134;
const OTGSC: u64 = 0x1F4;
const ENDPTSETUPSTAT: u64 = 0x208;
const ENDPTCOMPLETE: u64 = 0x218;
/// The controller's registers: what a controller reset clears.
const CONTROLLER: std::ops::RangeInclusive<u64> = 0x130..=0x21C;
const USB_SUSP_CTRL: u64 = 0x400;
/// `USB_PHY_VBUS_SENSORS`, `USB_PHY_VBUS_WAKEUP_ID`, `USB_PHY_ALT_VBUS_STS`.
const VBUS: std::ops::RangeInclusive<u64> = 0x404..=0x40C;
/// `UTMIP_PLL_CFG0` to `UTMIP_XCVR_CFG3`.
const UTMIP: std::ops::RangeInclusive<u64> = 0x800..=0x85C;

const CMD_RUN: u32 = 1 << 0;
const CMD_RESET: u32 = 1 << 1;
const SUSP_PHY_CLK_VALID: u32 = 1 << 7;
const SUSP_UTMIP_RESET: u32 = 1 << 11;
const SUSP_UTMIP_PHY_ENB: u32 = 1 << 12;
/// `OTGSC`'s interrupt enables (bits 30:24) and controls (bits 7:0); the
/// rest are status (15:8) and status changes (22:16), write one to clear.
const OTGSC_STORED: u32 = 0x7F00_00FF;

#[derive(Default)]
pub struct Usb2 {
    registers: Mutex<HashMap<u64, u32>>,
}

impl Usb2 {
    pub fn new() -> Self {
        Self::default()
    }

    /// The block loses its settings with the SoC's power (SC7).
    pub fn power_off(&self) {
        self.registers.lock().unwrap().clear();
    }

    fn modelled(offset: u64) -> bool {
        CONTROLLER.contains(&offset)
            || offset == USB_SUSP_CTRL
            || VBUS.contains(&offset)
            || UTMIP.contains(&offset)
    }
}

impl MmioDevice for Usb2 {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        let offset = word(offset, size)?;
        if !Self::modelled(offset) {
            return None;
        }
        let registers = self.registers.lock().unwrap();
        let stored = registers.get(&offset).copied().unwrap_or(0);
        let value = match offset {
            USBSTS | ENDPTSETUPSTAT | ENDPTCOMPLETE => 0,
            USB_SUSP_CTRL => {
                let running = stored & SUSP_UTMIP_PHY_ENB != 0 && stored & SUSP_UTMIP_RESET == 0;
                stored & !SUSP_PHY_CLK_VALID | if running { SUSP_PHY_CLK_VALID } else { 0 }
            }
            _ => stored,
        };
        Some(u64::from(value))
    }

    fn write(&self, _core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        let offset = word(offset, size)?;
        let value = value as u32;
        if !Self::modelled(offset) {
            return None;
        }
        let mut registers = self.registers.lock().unwrap();
        match offset {
            USBCMD if value & CMD_RUN != 0 => return None,
            USBCMD if value & CMD_RESET != 0 => {
                registers.retain(|offset, _| !CONTROLLER.contains(offset));
            }
            USBSTS | ENDPTSETUPSTAT | ENDPTCOMPLETE => {}
            OTGSC => {
                registers.insert(offset, value & OTGSC_STORED);
            }
            _ => {
                registers.insert(offset, value);
            }
        }
        Some(())
    }

    fn snapshot(&self) -> Option<&dyn Snapshot> {
        Some(self)
    }
}

impl Snapshot for Usb2 {
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        encode(&*self.registers.lock().unwrap())
    }

    fn restore(&self, state: &[u8]) -> Result<(), SnapshotError> {
        *self.registers.lock().unwrap() = decode("USB1", state)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BPMP: usize = crate::boot::CORE_COUNT;

    #[test]
    fn the_phy_clock_is_valid_once_the_phy_is_enabled_and_out_of_reset() {
        let usb = Usb2::new();
        assert_eq!(usb.read(BPMP, USB_SUSP_CTRL, 4), Some(0));
        // As the ROM: reset asserted, PHY enabled, reset released.
        usb.write(BPMP, USB_SUSP_CTRL, 4, 0x800).unwrap();
        usb.write(BPMP, USB_SUSP_CTRL, 4, 0x1800).unwrap();
        assert_eq!(usb.read(BPMP, USB_SUSP_CTRL, 4), Some(0x1800));
        usb.write(BPMP, USB_SUSP_CTRL, 4, 0x1000).unwrap();
        assert_eq!(usb.read(BPMP, USB_SUSP_CTRL, 4), Some(0x1080));
        // The bit is not the guest's to set.
        usb.write(BPMP, USB_SUSP_CTRL, 4, 0x80).unwrap();
        assert_eq!(usb.read(BPMP, USB_SUSP_CTRL, 4), Some(0));
    }

    #[test]
    fn a_controller_reset_clears_the_controller_and_completes_at_once() {
        let usb = Usb2::new();
        usb.write(BPMP, 0x1F8, 4, 2).unwrap();
        usb.write(BPMP, 0x800, 4, 0x1234).unwrap();
        usb.write(BPMP, USBCMD, 4, u64::from(CMD_RESET)).unwrap();
        assert_eq!(usb.read(BPMP, USBCMD, 4), Some(0));
        assert_eq!(usb.read(BPMP, 0x1F8, 4), Some(0));
        assert_eq!(usb.read(BPMP, 0x800, 4), Some(0x1234));
        usb.write(BPMP, 0x1F8, 4, 2).unwrap();
        assert_eq!(usb.read(BPMP, 0x1F8, 4), Some(2));
    }

    #[test]
    fn status_registers_clear_and_otgsc_keeps_only_its_enables() {
        let usb = Usb2::new();
        usb.write(BPMP, USBSTS, 4, 0x1FF).unwrap();
        assert_eq!(usb.read(BPMP, USBSTS, 4), Some(0));
        usb.write(BPMP, ENDPTSETUPSTAT, 4, 7).unwrap();
        assert_eq!(usb.read(BPMP, ENDPTSETUPSTAT, 4), Some(0));
        usb.write(BPMP, OTGSC, 4, 0x7F_0020).unwrap();
        assert_eq!(usb.read(BPMP, OTGSC, 4), Some(0x20));
    }

    #[test]
    fn refuses_running_the_controller_and_other_registers() {
        let usb = Usb2::new();
        assert_eq!(usb.write(BPMP, USBCMD, 4, 1), None);
        assert_eq!(usb.read(BPMP, 0x000, 4), None);
        assert_eq!(usb.write(BPMP, 0x860, 4, 0), None);
        assert_eq!(usb.read(BPMP, 0x400, 2), None);
    }

    #[test]
    fn snapshot_and_power_off() {
        let usb = Usb2::new();
        usb.write(BPMP, 0x808, 4, 5).unwrap();
        let other = Usb2::new();
        other.restore(&usb.save().unwrap()).unwrap();
        assert_eq!(other.read(BPMP, 0x808, 4), Some(5));
        other.power_off();
        assert_eq!(other.read(BPMP, 0x808, 4), Some(0));
    }
}
