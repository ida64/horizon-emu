//! I2C controller in packet mode and in normal master mode, and the bus
//! behind it.
//!
//! The `boot` process drives the controller the same way for every
//! transfer (its text `0x3d770`, `0x3dac0`, `0x3e110`): it flushes the
//! FIFOs, unmasks interrupts, pushes a three-word packet header and, for a
//! write, the payload into the TX FIFO, then sleeps on the controller's
//! interrupt until `PACKET_XFER_COMPLETE`, checking for a missing
//! acknowledge or lost arbitration each time it wakes. Reads are drained
//! from the RX FIFO according to the word count in `FIFO_STATUS`.
//!
//! The NX bootloader uses normal (non-packet) master mode instead, on I2C5
//! for the PMIC and the CPU regulator (its `0x4000075C` write and
//! `0x40000704` read): the device address goes into `CMD_ADDR0` (bit 0 set
//! for a read), up to four data bytes into `CMD_DATA1`, then `CNFG` gets
//! the byte count less one in bits 3:1, `CMD1` (bit 6) for a read and
//! `NEW_MASTER_FSM` (bit 11); after a configuration load it sets `SEND`
//! (bit 9) and polls `STATUS` until `BUSY` (bit 8) clears, the transfer
//! having succeeded if `CMD1_STAT` (bits 3:0) reads zero. A read leaves
//! the bytes in `CMD_DATA1`, the first in its low byte. Register and bit
//! names are the public Tegra ones (Linux `i2c-tegra.c`, U-Boot
//! `tegra_i2c.h`); the sequence is the bootloader's. `CMD_DATA2` carries
//! bytes 5 to 8 of a longer transfer \[inference from the 3-bit length\].
//! The second-slave and 10-bit address forms are not modeled. A missing
//! acknowledge of the address byte reads as `CMD1_STAT` 1, "no
//! acknowledge for byte 1" \[assumption: the bootloader only tests for
//! zero\]. `SEND` reads back clear once the transfer is over
//! \[assumption\]. Normal mode raises no interrupt status bit
//! \[assumption: the bootloader polls, with interrupts off\].
//!
//! Transfers complete the moment their last word is written (in normal
//! mode: the moment `SEND` is set): there is no bus timing, so the TX FIFO
//! is always empty and the controller is never busy.

use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex},
};

use serde::{Deserialize, Serialize};

use super::{Notes, regs::word};
use crate::{
    bus::MmioDevice,
    snapshot::{Snapshot, SnapshotError, decode, encode},
};

pub const SIZE: u64 = 0x100;

const CNFG: u64 = 0x00;
const CMD_ADDR0: u64 = 0x04;
const CMD_ADDR1: u64 = 0x08;
const CMD_DATA1: u64 = 0x0C;
const CMD_DATA2: u64 = 0x10;
const STATUS: u64 = 0x1C;
const TX_PACKET_FIFO: u64 = 0x50;
const RX_FIFO: u64 = 0x54;
const PACKET_TRANSFER_STATUS: u64 = 0x58;
const FIFO_CONTROL: u64 = 0x5C;
const FIFO_STATUS: u64 = 0x60;
const INTERRUPT_MASK: u64 = 0x64;
const INTERRUPT_STATUS: u64 = 0x68;
const CLK_DIVISOR: u64 = 0x6C;
const INTERRUPT_SOURCE: u64 = 0x70;
const BUS_CLEAR_CONFIG: u64 = 0x84;
const BUS_CLEAR_STATUS: u64 = 0x88;
const CONFIG_LOAD: u64 = 0x8C;
const CLKEN_OVERRIDE: u64 = 0x90;
const INTERFACE_TIMING_0: u64 = 0x94;
const INTERFACE_TIMING_1: u64 = 0x98;
const HS_INTERFACE_TIMING_0: u64 = 0x9C;
const HS_INTERFACE_TIMING_1: u64 = 0xA0;

const FIFO_DEPTH: usize = 8;

const FIFO_CONTROL_RX_FLUSH: u32 = 1 << 0;
const FIFO_CONTROL_TX_FLUSH: u32 = 1 << 1;

const INT_RX_FIFO_DATA_REQ: u32 = 1 << 0;
const INT_TX_FIFO_DATA_REQ: u32 = 1 << 1;
const INT_NO_ACK: u32 = 1 << 3;
const INT_ALL_PACKETS_XFER_COMPLETE: u32 = 1 << 6;
const INT_PACKET_XFER_COMPLETE: u32 = 1 << 7;
const INT_BUS_CLEAR_DONE: u32 = 1 << 11;
/// Bits that reflect a condition rather than latch an event.
const INT_LEVELS: u32 = INT_RX_FIFO_DATA_REQ | INT_TX_FIFO_DATA_REQ;

const TRANSFER_STATUS_NO_ACK_FOR_ADDRESS: u32 = 1 << 3;
/// Bits 15:4: the number of the last byte transferred, counted from 0
/// (Linux `i2c-tegra.c` names the field `TRANSFER_BYTENUM`; hekate waits
/// for it to reach the packet's size less one before it goes on, `i2c.c`).
const TRANSFER_STATUS_BYTE_SHIFT: u32 = 4;
const TRANSFER_STATUS_COMPLETE: u32 = 1 << 24;

const BUS_CLEAR_ENABLE: u32 = 1 << 0;

/// `CNFG` in normal mode: 10-bit addressing (`A_MOD`), the byte count
/// less one, a second-slave transaction, the direction of the first, the
/// trigger, and packet mode, which turns all of them off.
const CNFG_TEN_BIT_ADDRESS: u32 = 1 << 0;
const CNFG_LENGTH_SHIFT: u32 = 1;
const CNFG_LENGTH: u32 = 7;
const CNFG_SLAVE_2: u32 = 1 << 4;
const CNFG_CMD1_READ: u32 = 1 << 6;
const CNFG_SEND: u32 = 1 << 9;
const CNFG_PACKET_MODE: u32 = 1 << 10;
/// `CMD_ADDR0` bit 0: the address byte asks for a read.
const ADDRESS_READ: u32 = 1 << 0;
/// `STATUS` `CMD1_STAT`: the address byte went unacknowledged.
const STATUS_NO_ACK_FOR_BYTE_1: u32 = 1;

/// Third header word: the transfer is a read.
const HEADER_READ: u32 = 1 << 19;
const HEADER_WORDS: usize = 3;

/// A device on an I2C bus, addressed by its 7-bit address.
///
/// `None` from either method means the device does not model what was asked
/// of it, which stops the machine; `Some(false)` is a missing acknowledge.
///
/// The controller a device is on saves it with itself in a machine
/// snapshot, through [`I2cDevice::save`] and [`I2cDevice::restore`].
pub trait I2cDevice: Send {
    /// A write transfer: `data` is everything after the address byte.
    fn write(&mut self, data: &[u8]) -> Option<bool>;
    /// A read transfer filling `data`.
    fn read(&mut self, data: &mut [u8]) -> Option<bool>;
    /// The device's state, encoded: everything that decides what it
    /// answers from now on ([`Snapshot::save`]).
    fn save(&self) -> Result<Vec<u8>, SnapshotError>;
    /// Takes back what [`I2cDevice::save`] of the same device, in a machine
    /// built the same way, returned ([`Snapshot::restore`]).
    fn restore(&mut self, state: &[u8]) -> Result<(), SnapshotError>;
}

/// Called with the level of the controller's interrupt output.
pub type InterruptLine = Box<dyn Fn(bool) + Send + Sync>;

pub struct I2cController {
    name: &'static str,
    state: Mutex<State>,
    interrupt: InterruptLine,
    notes: Arc<Notes>,
}

/// Everything but the devices is the controller's own state, which a
/// snapshot saves as is; the devices save theirs through [`I2cDevice`].
#[derive(Serialize, Deserialize)]
struct State {
    /// Wiring: the restoring machine has the same devices at the same
    /// addresses.
    #[serde(skip)]
    devices: HashMap<u8, Box<dyn I2cDevice>>,
    /// Plain registers by offset: the configuration, normal mode's
    /// address and data registers, which a read transfer fills, and its
    /// `STATUS`, which only a transfer sets (in this map, so that the
    /// saved form stays what it was before normal mode).
    configuration: HashMap<u64, u32>,
    /// Trigger levels; the flush bits never read as set.
    fifo_control: u32,
    interrupt_mask: u32,
    /// Latched interrupt bits.
    interrupt_status: u32,
    transfer_status: u32,
    bus_cleared: bool,
    /// Words of the packet being assembled from the TX FIFO.
    packet: Vec<u32>,
    received: VecDeque<u32>,
}

impl I2cController {
    /// `name` identifies the bus in diagnostics.
    pub fn new(
        name: &'static str,
        devices: impl IntoIterator<Item = (u8, Box<dyn I2cDevice>)>,
        interrupt: InterruptLine,
        notes: Arc<Notes>,
    ) -> Self {
        Self {
            name,
            state: Mutex::new(State {
                devices: devices.into_iter().collect(),
                configuration: HashMap::new(),
                fifo_control: 0,
                interrupt_mask: 0,
                interrupt_status: 0,
                transfer_status: 0,
                bus_cleared: false,
                packet: Vec::new(),
                received: VecDeque::new(),
            }),
            interrupt,
            notes,
        }
    }

    /// Drives the interrupt output from the state the access left behind.
    /// Called with the state lock held so that levels reach the interrupt
    /// controller in order.
    fn update_interrupt(&self, state: &State) {
        (self.interrupt)(state.pending_interrupts() & state.interrupt_mask != 0);
    }
}

impl State {
    fn pending_interrupts(&self) -> u32 {
        let rx_trigger = (self.fifo_control >> 2 & 7) as usize + 1;
        let rx_request = if self.received.len() >= rx_trigger {
            INT_RX_FIFO_DATA_REQ
        } else {
            0
        };
        // The TX FIFO is always empty, hence always at its trigger level.
        self.interrupt_status | rx_request | INT_TX_FIFO_DATA_REQ
    }

    /// Takes one word from the TX FIFO, running the transfer once the
    /// packet it belongs to is complete. `Err` carries what a device did not
    /// model.
    fn push(&mut self, value: u32) -> Result<(), String> {
        self.packet.push(value);
        if self.packet.len() < HEADER_WORDS {
            return Ok(());
        }
        let size = (self.packet[1] & 0xFFF) as usize + 1;
        let control = self.packet[2];
        let is_read = control & HEADER_READ != 0;
        let payload_words = if is_read { 0 } else { size.div_ceil(4) };
        if self.packet.len() < HEADER_WORDS + payload_words {
            return Ok(());
        }
        let address = (control >> 1 & 0x7F) as u8;
        let payload: Vec<u8> = self.packet[HEADER_WORDS..]
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .take(size)
            .collect();
        self.packet.clear();

        let mut data = vec![0; size];
        let acknowledged = match self.devices.get_mut(&address) {
            // Nothing answers at this address.
            None => false,
            Some(device) if is_read => device.read(&mut data).ok_or_else(|| {
                format!("device {address:#04x}: read of {size} bytes is not implemented")
            })?,
            Some(device) => device.write(&payload).ok_or_else(|| {
                format!("device {address:#04x}: write of {payload:02x?} is not implemented")
            })?,
        };
        if !acknowledged {
            self.transfer_status = TRANSFER_STATUS_NO_ACK_FOR_ADDRESS;
            self.interrupt_status |= INT_NO_ACK;
            return Ok(());
        }
        if is_read {
            self.received.extend(data.chunks(4).map(|chunk| {
                let mut word = [0; 4];
                word[..chunk.len()].copy_from_slice(chunk);
                u32::from_le_bytes(word)
            }));
        }
        self.transfer_status =
            TRANSFER_STATUS_COMPLETE | (size as u32 - 1) << TRANSFER_STATUS_BYTE_SHIFT;
        self.interrupt_status |= INT_PACKET_XFER_COMPLETE | INT_ALL_PACKETS_XFER_COMPLETE;
        Ok(())
    }

    /// Runs the normal-mode transfer `cnfg` triggers: one transaction with
    /// the device `CMD_ADDR0` addresses, of up to eight bytes from or into
    /// `CMD_DATA1` and `CMD_DATA2`. `Err` carries what is not modeled.
    fn send(&mut self, cnfg: u32) -> Result<(), String> {
        if cnfg & (CNFG_TEN_BIT_ADDRESS | CNFG_SLAVE_2) != 0 {
            return Err(format!(
                "normal mode transfer with CNFG {cnfg:#x} (10-bit address or second slave) \
                 is not modeled"
            ));
        }
        let register = |state: &Self, offset| state.configuration.get(&offset).copied();
        let address_byte = register(self, CMD_ADDR0).unwrap_or(0);
        let is_read = cnfg & CNFG_CMD1_READ != 0;
        if (address_byte & ADDRESS_READ != 0) != is_read {
            return Err(format!(
                "normal mode transfer whose CMD_ADDR0 {address_byte:#x} and CNFG {cnfg:#x} \
                 disagree about its direction is not modeled"
            ));
        }
        let address = (address_byte >> 1 & 0x7F) as u8;
        let size = (cnfg >> CNFG_LENGTH_SHIFT & CNFG_LENGTH) as usize + 1;
        let mut data = [0; 8];
        for (offset, bytes) in [CMD_DATA1, CMD_DATA2].into_iter().zip(data.chunks_mut(4)) {
            bytes.copy_from_slice(&register(self, offset).unwrap_or(0).to_le_bytes());
        }
        let acknowledged = match self.devices.get_mut(&address) {
            // Nothing answers at this address.
            None => false,
            Some(device) if is_read => device.read(&mut data[..size]).ok_or_else(|| {
                format!("device {address:#04x}: read of {size} bytes is not implemented")
            })?,
            Some(device) => device.write(&data[..size]).ok_or_else(|| {
                format!(
                    "device {address:#04x}: write of {:02x?} is not implemented",
                    &data[..size]
                )
            })?,
        };
        if acknowledged && is_read {
            for (offset, bytes) in [CMD_DATA1, CMD_DATA2].into_iter().zip(data.chunks(4)) {
                let value = u32::from_le_bytes(bytes.try_into().unwrap());
                self.configuration.insert(offset, value);
            }
        }
        let status = if acknowledged {
            0
        } else {
            STATUS_NO_ACK_FOR_BYTE_1
        };
        self.configuration.insert(STATUS, status);
        self.configuration.insert(CNFG, cnfg & !CNFG_SEND);
        Ok(())
    }
}

impl MmioDevice for I2cController {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        let register = word(offset, size)?;
        let mut state = self.state.lock().unwrap();
        let value = match register {
            RX_FIFO => {
                let value = state.received.pop_front()?;
                self.update_interrupt(&state);
                value
            }
            PACKET_TRANSFER_STATUS => state.transfer_status,
            FIFO_CONTROL => state.fifo_control,
            FIFO_STATUS => (FIFO_DEPTH as u32) << 4 | state.received.len().min(FIFO_DEPTH) as u32,
            INTERRUPT_MASK => state.interrupt_mask,
            INTERRUPT_STATUS => state.pending_interrupts(),
            INTERRUPT_SOURCE => state.pending_interrupts() & state.interrupt_mask,
            BUS_CLEAR_STATUS => state.bus_cleared.into(),
            // Loading the configuration takes no time.
            CONFIG_LOAD => 0,
            CNFG
            | CMD_ADDR0
            | CMD_ADDR1
            | CMD_DATA1
            | CMD_DATA2
            | STATUS
            | CLK_DIVISOR
            | BUS_CLEAR_CONFIG
            | CLKEN_OVERRIDE
            | INTERFACE_TIMING_0
            | INTERFACE_TIMING_1
            | HS_INTERFACE_TIMING_0
            | HS_INTERFACE_TIMING_1 => {
                // Unwritten configuration reads as its reset value, zero.
                state.configuration.get(&register).copied().unwrap_or(0)
            }
            _ => return None,
        };
        Some(value.into())
    }

    fn write(&self, _core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        let register = word(offset, size)?;
        let value = value as u32;
        let mut state = self.state.lock().unwrap();
        match register {
            TX_PACKET_FIFO => {
                if let Err(what) = state.push(value) {
                    self.notes.add(format!("{}: {what}", self.name));
                    return None;
                }
            }
            FIFO_CONTROL => {
                if value & FIFO_CONTROL_RX_FLUSH != 0 {
                    state.received.clear();
                }
                if value & FIFO_CONTROL_TX_FLUSH != 0 {
                    state.packet.clear();
                }
                state.fifo_control = value & !(FIFO_CONTROL_RX_FLUSH | FIFO_CONTROL_TX_FLUSH);
            }
            INTERRUPT_MASK => state.interrupt_mask = value,
            // Write one to clear; the level bits follow their conditions.
            INTERRUPT_STATUS => state.interrupt_status &= !value | INT_LEVELS,
            BUS_CLEAR_CONFIG => {
                if value & BUS_CLEAR_ENABLE != 0 {
                    state.bus_cleared = true;
                    state.interrupt_status |= INT_BUS_CLEAR_DONE;
                }
                state
                    .configuration
                    .insert(register, value & !BUS_CLEAR_ENABLE);
            }
            CONFIG_LOAD => {}
            CNFG => {
                state.configuration.insert(register, value);
                if value & CNFG_SEND != 0
                    && value & CNFG_PACKET_MODE == 0
                    && let Err(what) = state.send(value)
                {
                    self.notes.add(format!("{}: {what}", self.name));
                    return None;
                }
            }
            CMD_ADDR0
            | CMD_ADDR1
            | CMD_DATA1
            | CMD_DATA2
            | CLK_DIVISOR
            | CLKEN_OVERRIDE
            | INTERFACE_TIMING_0
            | INTERFACE_TIMING_1
            | HS_INTERFACE_TIMING_0
            | HS_INTERFACE_TIMING_1 => {
                state.configuration.insert(register, value);
            }
            _ => return None,
        }
        self.update_interrupt(&state);
        Some(())
    }
    fn snapshot(&self) -> Option<&dyn Snapshot> {
        Some(self)
    }
}

/// The controller's registers and FIFOs, the packet it is assembling, and
/// each device's state by address. The interrupt output follows from the
/// registers at the next access and is not driven on restore.
impl Snapshot for I2cController {
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        let state = self.state.lock().unwrap();
        let mut devices = state
            .devices
            .iter()
            .map(|(&address, device)| Ok((address, device.save()?)))
            .collect::<Result<Vec<(u8, Vec<u8>)>, SnapshotError>>()?;
        devices.sort_unstable_by_key(|&(address, _)| address);
        encode(&(&*state, devices))
    }

    fn restore(&self, bytes: &[u8]) -> Result<(), SnapshotError> {
        let (saved, devices): (State, Vec<(u8, Vec<u8>)>) = decode(self.name, bytes)?;
        let mut state = self.state.lock().unwrap();
        let mut addresses: Vec<u8> = state.devices.keys().copied().collect();
        addresses.sort_unstable();
        if !devices
            .iter()
            .map(|&(address, _)| address)
            .eq(addresses.iter().copied())
        {
            return Err(SnapshotError::Corrupt(format!(
                "{}: devices at {addresses:02x?}, saved with others",
                self.name
            )));
        }
        for (address, device) in &devices {
            if let Some(target) = state.devices.get_mut(address) {
                target.restore(device)?;
            }
        }
        let wiring = std::mem::take(&mut state.devices);
        *state = saved;
        state.devices = wiring;
        Ok(())
    }
}

/// A device made of byte registers behind a register pointer: a write sets
/// the pointer from its first byte and stores the rest in consecutive
/// registers, a read returns consecutive registers from the pointer on.
pub struct ByteRegisters<R> {
    name: &'static str,
    pointer: u8,
    registers: R,
    notes: Arc<Notes>,
}

/// The registers of a [`ByteRegisters`] device. `None` means the register is
/// not modeled.
pub trait RegisterMap: Send {
    fn read(&mut self, register: u8) -> Option<u8>;
    fn write(&mut self, register: u8, value: u8) -> Option<()>;
    /// The registers' state, encoded, as [`I2cDevice::save`].
    fn save(&self) -> Result<Vec<u8>, SnapshotError>;
    /// Takes back what [`RegisterMap::save`] returned.
    fn restore(&mut self, state: &[u8]) -> Result<(), SnapshotError>;
}

impl<R: RegisterMap> ByteRegisters<R> {
    /// `name` identifies the device in the notes it leaves when one of its
    /// registers refuses an access.
    pub fn new(name: &'static str, registers: R, notes: Arc<Notes>) -> Self {
        Self {
            name,
            pointer: 0,
            registers,
            notes,
        }
    }

    /// Leaves a note about the access the registers refused. `None` unless
    /// a survey is being run.
    fn refuse(&self, what: &str) -> Option<()> {
        self.notes.add(format!(
            "{}: {what} register {:#04x} is not modeled",
            self.name, self.pointer
        ));
        self.notes.survey().then_some(())
    }
}

impl<R: RegisterMap> I2cDevice for ByteRegisters<R> {
    fn write(&mut self, data: &[u8]) -> Option<bool> {
        let (&pointer, values) = data.split_first()?;
        self.pointer = pointer;
        for &value in values {
            if self.registers.write(self.pointer, value).is_none() {
                self.refuse("write to")?;
            }
            self.pointer = self.pointer.wrapping_add(1);
        }
        Some(true)
    }

    fn read(&mut self, data: &mut [u8]) -> Option<bool> {
        for byte in data {
            *byte = match self.registers.read(self.pointer) {
                Some(value) => value,
                None => {
                    self.refuse("read of")?;
                    0
                }
            };
            self.pointer = self.pointer.wrapping_add(1);
        }
        Some(true)
    }

    /// The register pointer and the registers.
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        encode(&(self.pointer, self.registers.save()?))
    }

    fn restore(&mut self, state: &[u8]) -> Result<(), SnapshotError> {
        let (pointer, registers): (u8, Vec<u8>) = decode(self.name, state)?;
        self.registers.restore(&registers)?;
        self.pointer = pointer;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::*;

    /// Registers that hold what is written; register 0x7F is not modeled.
    struct Scratch([u8; 256]);

    impl RegisterMap for Scratch {
        fn read(&mut self, register: u8) -> Option<u8> {
            (register != 0x7F).then(|| self.0[usize::from(register)])
        }

        fn write(&mut self, register: u8, value: u8) -> Option<()> {
            self.0[usize::from(register)] = value;
            Some(())
        }

        fn save(&self) -> Result<Vec<u8>, SnapshotError> {
            encode(&self.0.as_slice())
        }

        fn restore(&mut self, state: &[u8]) -> Result<(), SnapshotError> {
            let registers: Vec<u8> = decode("scratch", state)?;
            self.0 = registers
                .try_into()
                .map_err(|_| SnapshotError::Corrupt("scratch".into()))?;
            Ok(())
        }
    }

    struct Bench {
        controller: I2cController,
        line: Arc<AtomicBool>,
    }

    impl Bench {
        fn new() -> Self {
            let line = Arc::new(AtomicBool::new(false));
            let output = Arc::clone(&line);
            let device: Box<dyn I2cDevice> = Box::new(ByteRegisters::new(
                "scratch",
                Scratch([0; 256]),
                Arc::default(),
            ));
            let controller = I2cController::new(
                "test bus",
                [(0x3C, device)],
                Box::new(move |level| output.store(level, Ordering::SeqCst)),
                Arc::new(Notes::default()),
            );
            Self { controller, line }
        }

        fn write(&self, register: u64, value: u32) -> Option<()> {
            self.controller.write(0, register, 4, value.into())
        }

        fn read(&self, register: u64) -> u32 {
            self.controller.read(0, register, 4).unwrap() as u32
        }

        /// Pushes a packet header the way the guest driver builds it.
        fn header(&self, address: u8, size: usize, is_read: bool) -> Option<()> {
            self.write(TX_PACKET_FIFO, 0x10)?;
            self.write(TX_PACKET_FIFO, size as u32 - 1)?;
            let read = if is_read { HEADER_READ | 1 } else { 0 };
            self.write(TX_PACKET_FIFO, 1 << 17 | read | u32::from(address) << 1)
        }
    }

    #[test]
    fn write_then_read_moves_bytes_through_the_fifos_and_signals_completion() {
        let bench = Bench::new();
        bench.write(INTERRUPT_MASK, 0x8C).unwrap();
        assert!(!bench.line.load(Ordering::SeqCst));

        // Write registers 0x10..0x15, a payload spanning two words.
        bench.header(0x3C, 6, false).unwrap();
        bench.write(TX_PACKET_FIFO, 0x0302_0110).unwrap();
        assert_eq!(
            bench.read(INTERRUPT_STATUS) & INT_PACKET_XFER_COMPLETE,
            0,
            "payload incomplete"
        );
        bench.write(TX_PACKET_FIFO, 0x0000_0504).unwrap();
        assert!(bench.line.load(Ordering::SeqCst));
        assert_ne!(bench.read(INTERRUPT_STATUS) & INT_PACKET_XFER_COMPLETE, 0);
        // hekate's driver waits for the number of the last byte sent.
        assert_eq!(bench.read(PACKET_TRANSFER_STATUS) >> 4 & 0xFFF, 5);
        bench.write(INTERRUPT_STATUS, 0xFC).unwrap();
        assert!(!bench.line.load(Ordering::SeqCst));

        // Set the pointer, then read five bytes back.
        bench.header(0x3C, 1, false).unwrap();
        bench.write(TX_PACKET_FIFO, 0x10).unwrap();
        bench.header(0x3C, 5, true).unwrap();
        assert_eq!(bench.read(FIFO_STATUS) & 0xF, 2);
        assert_eq!(bench.read(RX_FIFO), 0x0403_0201);
        assert_eq!(bench.read(RX_FIFO), 0x0000_0005);
        assert_eq!(bench.controller.read(0, RX_FIFO, 4), None, "FIFO drained");
    }

    #[test]
    fn absent_device_is_a_missing_acknowledge_not_a_completed_transfer() {
        let bench = Bench::new();
        bench.write(INTERRUPT_MASK, 0x8C).unwrap();
        bench.header(0x50, 1, true).unwrap();
        let status = bench.read(INTERRUPT_STATUS);
        assert_eq!(status & (INT_NO_ACK | INT_PACKET_XFER_COMPLETE), INT_NO_ACK);
        assert_ne!(
            bench.read(PACKET_TRANSFER_STATUS) & TRANSFER_STATUS_NO_ACK_FOR_ADDRESS,
            0
        );
        assert!(bench.line.load(Ordering::SeqCst));
    }

    #[test]
    fn transmit_request_is_a_level_that_only_the_mask_silences() {
        let bench = Bench::new();
        bench.write(INTERRUPT_MASK, 0x8E).unwrap();
        assert!(bench.line.load(Ordering::SeqCst));
        bench.write(INTERRUPT_STATUS, 0xFF).unwrap();
        assert!(bench.line.load(Ordering::SeqCst), "the FIFO is still empty");
        bench.write(INTERRUPT_MASK, 0x8C).unwrap();
        assert!(!bench.line.load(Ordering::SeqCst));
    }

    #[test]
    fn flush_bits_act_and_read_back_clear() {
        let bench = Bench::new();
        bench.header(0x3C, 4, true).unwrap();
        bench.write(FIFO_CONTROL, 0xFF).unwrap();
        assert_eq!(bench.read(FIFO_CONTROL), 0xFC);
        assert_eq!(bench.read(FIFO_STATUS) & 0xF, 0);
    }

    #[test]
    fn an_unmodeled_device_register_refuses_the_access() {
        let bench = Bench::new();
        bench.header(0x3C, 1, false).unwrap();
        bench.write(TX_PACKET_FIFO, 0x7F).unwrap();
        assert_eq!(bench.header(0x3C, 1, true), None);
    }

    #[test]
    fn a_restored_controller_completes_the_packet_and_drains_the_reads_of_the_saved_one() {
        let saved = Bench::new();
        saved.write(INTERRUPT_MASK, 0x8C).unwrap();
        // Registers 0x10..0x13 written; then a read of two bytes, its
        // answer left in the RX FIFO; then half of another write's header.
        saved.header(0x3C, 5, false).unwrap();
        saved.write(TX_PACKET_FIFO, 0x0302_0110).unwrap();
        saved.write(TX_PACKET_FIFO, 0x04).unwrap();
        saved.header(0x3C, 1, false).unwrap();
        saved.write(TX_PACKET_FIFO, 0x11).unwrap();
        saved.header(0x3C, 2, true).unwrap();
        saved.write(TX_PACKET_FIFO, 0x10).unwrap();
        saved.write(TX_PACKET_FIFO, 0x0).unwrap();

        let restored = Bench::new();
        restored
            .controller
            .restore(&saved.controller.save().unwrap())
            .unwrap();
        assert!(
            !restored.line.load(Ordering::SeqCst),
            "not driven on restore"
        );
        assert_eq!(restored.read(INTERRUPT_MASK), 0x8C);
        assert_eq!(restored.read(FIFO_STATUS) & 0xF, 1);
        assert_eq!(restored.read(RX_FIFO), 0x0302);
        // The rest of the header: a one-byte write of the pointer, 0x13.
        restored
            .write(TX_PACKET_FIFO, 1 << 17 | u32::from(0x3Cu8) << 1)
            .unwrap();
        restored.write(TX_PACKET_FIFO, 0x13).unwrap();
        restored.header(0x3C, 1, true).unwrap();
        assert_eq!(restored.read(RX_FIFO), 0x04);

        // A controller with other devices does not take it.
        let wrong = I2cController::new("other", [], Box::new(|_| {}), Arc::default());
        assert!(matches!(
            wrong.restore(&saved.controller.save().unwrap()),
            Err(SnapshotError::Corrupt(_))
        ));
    }

    /// The NX bootloader's I2C driver (normal master mode), register for
    /// register: `0x4000075C` (write), `0x40000704` (read), `0x400006E0`
    /// (configuration load), `0x400025F0` (bus clear), `0x4000267C` and
    /// `0x40002638` (device register write and read).
    struct Bootloader<'a>(&'a I2cController);

    impl Bootloader<'_> {
        fn write(&self, register: u64, value: u32) {
            self.0.write(0, register, 4, value.into()).unwrap();
        }

        fn read(&self, register: u64) -> u32 {
            self.0.read(0, register, 4).unwrap() as u32
        }

        fn load_config(&self) {
            self.write(CONFIG_LOAD, 0x25);
            let mut tries = 0;
            while self.read(CONFIG_LOAD) & 1 != 0 && tries < 20 {
                tries += 1;
            }
        }

        /// Sets `SEND` and waits for the transfer: whether it succeeded.
        fn send(&self) -> bool {
            let cnfg = self.read(CNFG);
            self.write(CNFG, cnfg & !0x200 | 0x200);
            while self.read(STATUS) & 0x100 != 0 {}
            self.read(STATUS) & 0xF == 0
        }

        fn transmit(&self, address: u8, data: &[u8]) -> bool {
            let mut word = [0; 4];
            word[..data.len()].copy_from_slice(data);
            self.write(CMD_ADDR0, u32::from(address) << 1);
            self.write(CMD_DATA1, u32::from_le_bytes(word));
            self.write(CNFG, (2 * data.len() as u32 - 2) | 0x2800);
            self.load_config();
            self.send()
        }

        fn receive(&self, address: u8, data: &mut [u8]) -> bool {
            self.write(CMD_ADDR0, u32::from(address) << 1 | 1);
            self.write(CNFG, (2 * data.len() as u32 - 2) | 0x2840);
            self.load_config();
            if !self.send() {
                return false;
            }
            let word = self.read(CMD_DATA1).to_le_bytes();
            data.copy_from_slice(&word[..data.len()]);
            true
        }

        fn set(&self, address: u8, register: u8, value: u8) {
            assert!(
                self.transmit(address, &[register, value]),
                "write of {register:#x}"
            );
        }

        fn get(&self, address: u8, register: u8) -> u8 {
            let mut value = [0];
            assert!(self.transmit(address, &[register]) && self.receive(address, &mut value));
            value[0]
        }

        fn clear_bus(&self) {
            self.write(CLK_DIVISOR, 0x50001);
            self.write(BUS_CLEAR_CONFIG, 0x90003);
            self.load_config();
            let mut tries = 0;
            while self.read(INTERRUPT_STATUS) & 1 << 11 == 0 {
                tries += 1;
                assert!(tries < 10, "bus clear never done");
            }
            self.read(BUS_CLEAR_STATUS);
            let status = self.read(INTERRUPT_STATUS);
            self.write(INTERRUPT_STATUS, status);
        }
    }

    /// I2C5 as the bootloader finds it: the PMIC's power block and the two
    /// MAX77621 regulators.
    fn power_bus() -> I2cController {
        use super::super::{
            max77620::{self, Max77620},
            max77621::{self, Max77621},
        };
        let pmic = Max77620::new(Arc::default(), |_| {});
        let notes: Arc<Notes> = Arc::default();
        let devices: [(u8, Box<dyn I2cDevice>); 3] = [
            (
                max77620::PMIC_ADDRESS,
                Box::new(ByteRegisters::new(
                    "MAX77620",
                    pmic.power_block(),
                    Arc::clone(&notes),
                )),
            ),
            (
                max77621::CPU_ADDRESS,
                Box::new(ByteRegisters::new(
                    "CPU",
                    Max77621::cpu(),
                    Arc::clone(&notes),
                )),
            ),
            (
                max77621::GPU_ADDRESS,
                Box::new(ByteRegisters::new(
                    "GPU",
                    Max77621::gpu(),
                    Arc::clone(&notes),
                )),
            ),
        ];
        I2cController::new("I2C5", devices, Box::new(|_| {}), notes)
    }

    #[test]
    fn the_bootloaders_power_sequence_lands_in_the_pmic_and_the_cpu_regulator() {
        const PMIC: u8 = 0x3C;
        const CPU: u8 = 0x1B;
        let controller = power_bus();
        let bootloader = Bootloader(&controller);
        bootloader.clear_bus();
        // Research §2.3, rows 1 to 12: configuration, power sequencers,
        // and SD0 at 1125 mV ((1_125_000 - 600_000) / 12_500, `0x40002860`).
        let early = [
            (0x04, 0x40),
            (0x41, 0x78),
            (0x43, 0x38),
            (0x44, 0x3A),
            (0x45, 0x38),
            (0x4A, 0x0F),
            (0x4E, 0xC7),
            (0x4F, 0x4F),
            (0x50, 0x29),
            (0x52, 0x1B),
            (0x56, 0x22),
            (0x16, 42),
        ];
        for (register, value) in early {
            bootloader.set(PMIC, register, value);
        }
        // CCPLEX power-up (`0x40000806`): AME_GPIO read back with bit 5
        // cleared, GPIO5 on, then the CPU regulator.
        let ame_gpio = bootloader.get(PMIC, 0x40);
        bootloader.set(PMIC, 0x40, ame_gpio & 0xDF);
        bootloader.set(PMIC, 0x3B, 0x09);
        let regulator = [(2, 0xB0), (3, 0xC1), (0, 0xB7), (1, 0xB7)];
        for (register, value) in regulator {
            bootloader.set(CPU, register, value);
        }
        // SDRAM init (`0x40002840`): SD_CFG2, SD1 at 1100 mV.
        let late = [(0x22, 0x05), (0x17, 40)];
        for (register, value) in late {
            bootloader.set(PMIC, register, value);
        }

        let pmic = early
            .into_iter()
            .chain([(0x40, ame_gpio & 0xDF), (0x3B, 0x09)])
            .chain(late);
        for (register, value) in pmic {
            assert_eq!(bootloader.get(PMIC, register), value, "PMIC {register:#x}");
        }
        for (register, value) in regulator {
            assert_eq!(bootloader.get(CPU, register), value, "CPU {register}");
        }
        // Two bytes in one read, out of CMD_DATA1 in bus order.
        let mut both = [0; 2];
        assert!(bootloader.transmit(CPU, &[0]) && bootloader.receive(CPU, &mut both));
        assert_eq!(both, [0xB7, 0xB7]);
        assert_eq!(
            bootloader.read(CNFG) & CNFG_SEND,
            0,
            "SEND reads back clear"
        );
        assert_eq!(
            bootloader.read(INTERRUPT_STATUS) & !INT_LEVELS,
            0,
            "normal mode raises nothing"
        );
    }

    #[test]
    fn normal_mode_reports_a_missing_acknowledge_and_refuses_what_it_does_not_model() {
        let bench = Bench::new();
        let bootloader = Bootloader(&bench.controller);
        assert!(!bootloader.transmit(0x50, &[0x00]), "nothing at 0x50");
        assert_eq!(bootloader.read(STATUS), STATUS_NO_ACK_FOR_BYTE_1);
        assert!(bootloader.transmit(0x3C, &[0x10, 0xAB]));
        assert_eq!(bootloader.read(STATUS), 0);

        // Eight bytes through both data registers.
        bootloader.write(CMD_ADDR0, 0x3C << 1);
        bootloader.write(CMD_DATA1, 0x0302_0120);
        bootloader.write(CMD_DATA2, 0x0706_0504);
        bootloader.write(CNFG, 7 << 1 | 0x2800);
        assert!(bootloader.send());
        assert!(bootloader.transmit(0x3C, &[0x20]));
        bootloader.write(CMD_ADDR0, 0x3C << 1 | 1);
        bootloader.write(CNFG, 6 << 1 | 0x2840);
        assert!(bootloader.send());
        assert_eq!(bootloader.read(CMD_DATA1), 0x0403_0201);
        assert_eq!(bootloader.read(CMD_DATA2) & 0xFF_FFFF, 0x07_0605);

        // Saved and restored with the rest of the registers.
        let restored = Bench::new();
        restored
            .controller
            .restore(&bench.controller.save().unwrap())
            .unwrap();
        assert_eq!(
            Bootloader(&restored.controller).read(CMD_DATA1),
            0x0403_0201
        );

        // A direction the address byte contradicts, a register the device
        // lacks, and the second-slave form stop the machine.
        bootloader.write(CMD_ADDR0, 0x3C << 1);
        assert_eq!(bench.controller.write(0, CNFG, 4, 0x2A40), None);
        assert!(bootloader.transmit(0x3C, &[0x7F]));
        bootloader.write(CMD_ADDR0, 0x3C << 1 | 1);
        assert_eq!(bench.controller.write(0, CNFG, 4, 0x2A40), None);
        assert_eq!(bench.controller.write(0, CNFG, 4, 0x2A10), None);
        // Packet mode is left alone by SEND.
        bootloader.write(CNFG, 0x2E00);
        assert_eq!(bootloader.read(CNFG), 0x2E00);
    }
}
