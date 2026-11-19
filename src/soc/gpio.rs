//! GPIO: eight controllers of four 8-pin ports each.
//!
//! Every per-port register is eight bits wide in a 32-bit word. The upper
//! half of each controller holds "masked" aliases of the configuration
//! registers, which write only the bits selected by the high byte.

use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use super::regs::word;
use crate::{
    bus::MmioDevice,
    snapshot::{self, Snapshot, SnapshotError, encode},
};

pub const BASE: u64 = 0x6000_D000;
pub const SIZE: u64 = 0x800;

const PORT_COUNT: usize = 32;
/// GPIO bank SPI assignments from public Linux `tegra210.dtsi`; GIC
/// interrupt IDs include the 32 private interrupts.
pub const INTIDS: [u32; 8] = [64, 65, 66, 67, 87, 119, 121, 157];
type Line = Box<dyn Fn(bool) + Send + Sync>;
type OutputObserver = Box<dyn Fn(usize, u8, bool) + Send + Sync>;

/// Register rows of one controller; each row has one word per port.
const CNF: u64 = 0x00;
const OE: u64 = 0x10;
const OUT: u64 = 0x20;
const IN: u64 = 0x30;
const INT_STA: u64 = 0x40;
const INT_ENB: u64 = 0x50;
const INT_LVL: u64 = 0x60;
const INT_CLR: u64 = 0x70;
/// Added to a row's offset to reach its masked-write alias.
const MASKED: u64 = 0x80;
/// Two rows of the upper half are registers of their own (as in the
/// public Linux driver for this SoC): where the alias of `IN` would be,
/// the debounce enables, written masked; where the alias of `INT_CLR`
/// would be, the port's debounce time. The firmware's GPIO driver reads the
/// time before setting it (Bus, at controller 6 port Z).
const DEBOUNCE_ENABLE: u64 = IN;
const DEBOUNCE_COUNT: u64 = INT_CLR;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
struct Port {
    /// 1: the pin is a GPIO rather than its special function.
    configuration: u8,
    output_enable: u8,
    output: u8,
    interrupt_enable: u8,
    interrupt_level: u32,
    interrupt_status: u8,
    input: Option<u8>,
    debounce_enable: u8,
    /// Starts at zero, no debounce time: an assumption about what the boot
    /// chain left, in a register the driver sets itself.
    debounce_count: u8,
}

pub struct Gpio {
    ports: Mutex<[Port; PORT_COUNT]>,
    interrupts: [Line; 8],
    output_observers: Mutex<Vec<OutputObserver>>,
    /// Whether `IN` reads zero for pins not in GPIO mode (`CNF` clear):
    /// see [`Gpio::power_on`].
    special_function_inputs_read_low: bool,
}

impl Gpio {
    /// `inputs` gives, per port index (`controller * 4 + port`), the levels
    /// of its pins when they are inputs.
    pub fn new(inputs: impl IntoIterator<Item = (usize, u8)>) -> Self {
        Self::with_interrupts(inputs, std::array::from_fn(|_| Box::new(|_| {}) as Line))
    }

    /// Connects bank interrupts. Trigger encoding follows the public Linux
    /// `gpio-tegra.c`: polarity bits 0..7, edge select 8..15, both edges 16..23.
    pub fn with_interrupts(
        inputs: impl IntoIterator<Item = (usize, u8)>,
        interrupts: [Line; 8],
    ) -> Self {
        let mut ports = [Port::default(); PORT_COUNT];
        for (index, value) in inputs {
            ports[index].input = Some(value);
        }
        Self {
            ports: Mutex::new(ports),
            interrupts,
            output_observers: Mutex::new(Vec::new()),
            special_function_inputs_read_low: false,
        }
    }

    /// The controller as a machine that runs the boot chain has it. Every
    /// register the NX bootloader's hardware-state check reads is zero
    /// (`nxbl_hwcheck.csv`, the check `0x40002A7C`, bootloader text
    /// `0x40005288`..`0x40006AB8`: all eight controllers' rows and their
    /// masked aliases), as the ports start here. That includes `IN` of
    /// every port, while the board holds pins high that the boot chain
    /// later reads high once it has made them GPIOs (rail detects E6 and
    /// H6, the volume buttons X6 and X7: `0x400174BC`, `0x40017480`).
    /// \[inference\] A pin's `IN` bit reads zero while the pin is not in
    /// GPIO mode (`CNF` clear), and its level once it is. A machine
    /// started at kernel entry keeps reading levels regardless of `CNF`,
    /// as it always has.
    pub fn power_on(mut self) -> Self {
        self.special_function_inputs_read_low = true;
        self
    }

    /// Observes transitions on enabled output pins, e.g. a board device reset.
    pub fn observe_output(&self, observer: OutputObserver) {
        self.output_observers.lock().unwrap().push(observer);
    }

    /// Changes a board input on a port declared in the initial input map.
    /// Unknown ports/pins refuse rather than inventing other input levels.
    pub fn set_input(&self, index: usize, pin: u8, high: bool) -> Option<()> {
        if pin >= 8 {
            return None;
        }
        let mut ports = self.ports.lock().unwrap();
        let port = ports.get_mut(index)?;
        let old = port.input?;
        let bit = 1 << pin;
        let new = (old & !bit) | if high { bit } else { 0 };
        port.input = Some(new);
        let edge = (port.interrupt_level >> 8) as u8;
        let both = (port.interrupt_level >> 16) as u8;
        let polarity = port.interrupt_level as u8;
        port.interrupt_status |= (old ^ new)
            & edge
            & (both | !(new ^ polarity))
            & port.configuration
            & !port.output_enable;
        self.update_bank(&mut ports, index / 4);
        Some(())
    }

    fn update_bank(&self, ports: &mut [Port; PORT_COUNT], bank: usize) {
        let mut pending = false;
        for port in &mut ports[bank * 4..bank * 4 + 4] {
            if let Some(input) = port.input {
                let edge = (port.interrupt_level >> 8) as u8;
                port.interrupt_status |= !(input ^ port.interrupt_level as u8)
                    & !edge
                    & port.configuration
                    & !port.output_enable
                    & port.interrupt_enable;
            }
            pending |= port.interrupt_status & port.interrupt_enable != 0;
        }
        (self.interrupts[bank])(pending);
    }

    /// Level of a pin the guest drives, or `None` while it is not an
    /// enabled GPIO output.
    pub fn output(&self, port: usize, pin: u8) -> Option<bool> {
        let port = self.ports.lock().unwrap()[port];
        let bit = 1 << pin;
        (port.configuration & port.output_enable & bit != 0).then_some(port.output & bit != 0)
    }

    /// The level the board puts on a pin, as [`Gpio::set_input`] last set
    /// it, or `None` on a port without listed inputs (or no such pin).
    pub fn input(&self, port: usize, pin: u8) -> Option<bool> {
        let input = self.ports.lock().unwrap().get(port)?.input?;
        Some(input & 1u8.checked_shl(pin.into())? != 0)
    }
}

/// Splits a register offset into port index, row, and whether it is the
/// masked alias.
fn decode(offset: u64) -> (usize, u64, bool) {
    let controller = (offset >> 8) as usize;
    let port = (offset >> 2 & 3) as usize;
    let row = offset & 0x70;
    (controller * 4 + port, row, offset & MASKED != 0)
}

impl MmioDevice for Gpio {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        // A one-byte load at a register's address gets its low byte, where
        // every row but `INT_LVL` has all it holds: the driver saves the
        // debounce enables that way before the console sleeps (Bus, read
        // at 0x6000d0b0).
        let offset = match size {
            1 => word(offset, 4)?,
            _ => word(offset, size)?,
        };
        let low_end = if size == 1 { 0xFF } else { u32::MAX };
        let (index, row, masked) = decode(offset);
        let port = self.ports.lock().unwrap()[index];
        let value = match row {
            DEBOUNCE_ENABLE if masked => port.debounce_enable.into(),
            DEBOUNCE_COUNT if masked => port.debounce_count.into(),
            CNF => port.configuration.into(),
            OE => port.output_enable.into(),
            OUT => port.output.into(),
            IN => {
                // Output pins read back what they drive; with
                // `special_function_inputs_read_low`, pins not in GPIO mode
                // read low.
                let driven = port.configuration & port.output_enable;
                let sensed = if self.special_function_inputs_read_low {
                    port.configuration & !driven
                } else {
                    !driven
                };
                let external = match sensed {
                    0 => 0,
                    _ => port.input? & sensed,
                };
                u32::from(port.output & driven | external)
            }
            INT_STA => port.interrupt_status.into(),
            INT_ENB => port.interrupt_enable.into(),
            INT_LVL => port.interrupt_level,
            // The public Tegra K1 TRM (same controller) describes INT_CLR as
            // a write-only set-to-clear register with reset value 0: it
            // stores nothing, so reads return 0. Bus clears a handled pin
            // with a read-modify-write of this row and reads it back as a
            // write barrier (010000000000000a, .text 0x16af8..0x16b04); a
            // zero read keeps the write from clearing other pins' flags.
            INT_CLR if !masked => 0,
            _ => return None,
        };
        Some((value & low_end).into())
    }

    fn write(&self, _core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        let (index, row, masked) = decode(word(offset, size)?);
        let value = value as u32;
        let merge = |current: u8| {
            let mask = if masked { (value >> 8) as u8 } else { 0xFF };
            current & !mask | value as u8 & mask
        };
        let mut ports = self.ports.lock().unwrap();
        let port = &mut ports[index];
        let old_output = port.output & port.output_enable & port.configuration;
        match row {
            DEBOUNCE_ENABLE if masked => port.debounce_enable = merge(port.debounce_enable),
            DEBOUNCE_COUNT if masked => port.debounce_count = value as u8,
            CNF => port.configuration = merge(port.configuration),
            OE => port.output_enable = merge(port.output_enable),
            OUT => port.output = merge(port.output),
            INT_ENB => port.interrupt_enable = merge(port.interrupt_enable),
            INT_LVL if !masked => port.interrupt_level = value & 0x00FF_FFFF,
            INT_CLR if !masked => port.interrupt_status &= !(value as u8),
            _ => return None,
        }
        let output = port.output & port.output_enable & port.configuration;
        let changed = old_output ^ output;
        self.update_bank(&mut ports, index / 4);
        drop(ports);
        for observer in self.output_observers.lock().unwrap().iter() {
            for pin in 0..8 {
                if changed & (1 << pin) != 0 {
                    observer(index, pin, output & (1 << pin) != 0);
                }
            }
        }
        Some(())
    }

    fn snapshot(&self) -> Option<&dyn Snapshot> {
        Some(self)
    }
}

/// Saves every port: configuration, outputs, interrupt setup and status,
/// debounce, and the levels the board puts on its inputs. The bank
/// interrupt lines are levels the interrupt controller saves, and the
/// devices that observe outputs save what they made of them, so restoring
/// drives nothing.
impl Snapshot for Gpio {
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        encode(&*self.ports.lock().unwrap())
    }

    fn restore(&self, state: &[u8]) -> Result<(), SnapshotError> {
        *self.ports.lock().unwrap() = snapshot::decode("GPIO", state)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Controller 6, port 3: port index 27.
    const PORT: u64 = 0x60C;

    #[test]
    fn masked_aliases_change_only_the_selected_bits() {
        let gpio = Gpio::new([]);
        gpio.write(0, PORT + OE, 4, 0x0F).unwrap();
        gpio.write(0, PORT + OE + MASKED, 4, 0x8180).unwrap();
        assert_eq!(gpio.read(0, PORT + OE, 4), Some(0x8E));
        // The neighbouring port is untouched.
        assert_eq!(gpio.read(0, PORT - 4 + OE, 4), Some(0));
    }

    #[test]
    fn input_register_mixes_driven_and_external_levels() {
        let gpio = Gpio::new([(27, 0b1100_0000)]);
        gpio.write(0, PORT + CNF, 4, 0x03).unwrap();
        gpio.write(0, PORT + OE, 4, 0x01).unwrap();
        gpio.write(0, PORT + OUT, 4, 0x03).unwrap();
        // Pin 0 is driven high; pin 1 is an input the board holds low.
        assert_eq!(gpio.read(0, PORT + IN, 4), Some(0b1100_0001));
        assert_eq!((gpio.output(27, 0), gpio.output(27, 1)), (Some(true), None));
        // A port whose external levels are not known cannot be sampled.
        assert_eq!(gpio.read(0, 0x30, 4), None);
    }

    #[test]
    fn edge_latches_and_level_reasserts_after_acknowledgement() {
        use std::sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        };
        let asserted = Arc::new(AtomicBool::new(false));
        let observed = Arc::clone(&asserted);
        let mut lines: [Line; 8] = std::array::from_fn(|_| Box::new(|_| {}) as Line);
        lines[0] = Box::new(move |high| observed.store(high, Ordering::SeqCst));
        let gpio = Gpio::with_interrupts([(0, 1)], lines);
        gpio.write(0, CNF, 4, 1).unwrap();
        gpio.write(0, INT_LVL, 4, 0x100).unwrap();
        gpio.write(0, INT_ENB, 4, 1).unwrap();
        gpio.set_input(0, 0, false).unwrap();
        assert!(asserted.load(Ordering::SeqCst));
        gpio.set_input(0, 0, true).unwrap();
        assert_eq!(gpio.read(0, INT_STA, 4), Some(1));
        // The clear row stores nothing: Bus's read-modify-write sees 0.
        assert_eq!(gpio.read(0, INT_CLR, 4), Some(0));
        gpio.write(0, INT_CLR, 4, 1).unwrap();
        assert!(!asserted.load(Ordering::SeqCst));
        gpio.write(0, INT_LVL, 4, 0).unwrap();
        gpio.set_input(0, 0, false).unwrap();
        gpio.write(0, INT_CLR, 4, 1).unwrap();
        assert!(asserted.load(Ordering::SeqCst));
        gpio.write(0, INT_ENB + MASKED, 4, 0x100).unwrap();
        assert!(!asserted.load(Ordering::SeqCst));
        assert_eq!(gpio.set_input(1, 0, true), None);
    }

    #[test]
    fn at_power_on_inputs_read_low_until_the_pin_is_a_gpio() {
        // Port X (controller 5, port 3: index 23; the bootloader reads its
        // `IN` at 0x6000_D53C) with the volume buttons (pins 6 and 7) held
        // high by the board, as the bootloader's check finds it, then makes
        // them inputs (`0x40017480`).
        const X: u64 = 0x50C;
        let gpio = Gpio::new([(23, 0b1100_0000)]).power_on();
        assert_eq!(gpio.read(0, X + IN, 4), Some(0));
        assert_eq!(
            gpio.read(0, X + IN + MASKED, 4),
            Some(0),
            "debounce enables"
        );
        // A port whose levels are not known reads zero too while no pin is
        // a GPIO.
        assert_eq!(gpio.read(0, IN, 4), Some(0));
        gpio.write(0, X + CNF + MASKED, 4, 0xC0C0).unwrap();
        gpio.write(0, X + OE + MASKED, 4, 0xC000).unwrap();
        assert_eq!(gpio.read(0, X + IN, 4), Some(0b1100_0000));
        // Started at kernel entry, the levels show regardless of CNF.
        let gpio = Gpio::new([(23, 0b1100_0000)]);
        assert_eq!(gpio.read(0, X + IN, 4), Some(0b1100_0000));
    }
}
