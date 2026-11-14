//! SD/MMC host controller: SD Host Controller specification registers plus
//! the Tegra vendor block, and the card in its slot.
//!
//! The filesystem process brings up the eMMC port as soon as it starts.
//! With a module fitted, commands reach the card and data moves by SDMA:
//! the controller masters the bus and copies blocks between the card and
//! the buffer the driver names, through the SMMU, pausing at every buffer
//! boundary for the driver to confirm the next address. Without a module
//! the controller is still fully there, but no card ever answers, and every
//! command that expects a response ends in a command timeout.
//!
//! Clocks are stable, and resets and calibrations over, by the time the
//! register write that starts them returns, and so is a command that moves
//! no data. Data takes the card time ([`Timing`]): the registers have the
//! outcome at once, but the status bits and the interrupt line tell the
//! driver of it only when a card would be done, in the order a controller
//! reports them, and the controller reads as busy until then.
//!
//! That wait is part of how the console's software runs. The filesystem
//! process sleeps through it, which leaves the core to threads of lower
//! priority; with a card that is done at once, a burst of filesystem work
//! shuts those out for as long as it lasts. That was seen to decide a
//! race on a first boot: `account` creates its saves and then asks `nifm`
//! for the network interfaces, which a worker thread of `nifm` has to
//! have opened by then (account text `0xBA770`, nifm text `0x8AEC0`); it
//! gets to only while `account`'s saves keep the card busy, and `account`
//! aborts if it has not.
//!
//! Programmed I/O through the buffer data port and ADMA descriptor tables
//! are not modeled; a transfer that asks for either stops the machine with
//! a note. The capabilities report only what the driver insists on.

use std::{
    mem,
    ops::Range,
    sync::{Arc, Mutex, mpsc},
    thread,
    time::{Duration, Instant},
};

use super::{
    Notes,
    emmc::{Emmc, Response},
    mc::DeviceMemory,
};
use crate::bus::MmioDevice;

/// Register block of one controller.
pub const SIZE: u64 = 0x200;

const SDMA_ADDRESS: usize = 0x00;
const BLOCK_SIZE: usize = 0x04;
const BLOCK_COUNT: usize = 0x06;
const ARGUMENT: usize = 0x08;
const TRANSFER_MODE: usize = 0x0C;
const COMMAND: usize = 0x0E;
const RESPONSE: usize = 0x10;
const PRESENT_STATE: usize = 0x24;
const HOST_CONTROL: usize = 0x28;
const CLOCK_CONTROL: usize = 0x2C;
const SOFTWARE_RESET: usize = 0x2F;
const NORMAL_INTERRUPT_STATUS: usize = 0x30;
const ERROR_INTERRUPT_STATUS: usize = 0x32;
const NORMAL_STATUS_ENABLE: usize = 0x34;
const ERROR_STATUS_ENABLE: usize = 0x36;
const NORMAL_SIGNAL_ENABLE: usize = 0x38;
const ERROR_SIGNAL_ENABLE: usize = 0x3A;
const HOST_CONTROL_2: usize = 0x3E;
const CAPABILITIES: usize = 0x40;
/// Doubles as the 64-bit SDMA address once version 4 mode is enabled.
const ADMA_ADDRESS: usize = 0x58;
const HOST_VERSION: usize = 0xFE;
const VENDOR_DLLCAL_CFG: usize = 0x1B0;
const VENDOR_TUNING_CNTRL0: usize = 0x1C0;
const AUTO_CAL_CONFIG: usize = 0x1E4;

/// The CMD and DAT lines idle high: nothing is driving them.
const PRESENT_STATE_IDLE: u32 = 0x01F0_0000;
/// Command inhibit, for the command line and for the data lines.
const PRESENT_STATE_INHIBIT: u8 = 0x03;

/// Block size register: the transfer block size, and above it the size of
/// the SDMA buffer as a power of two times 4 KiB.
const BLOCK_SIZE_BYTES: u16 = 0x0FFF;
const SDMA_BOUNDARY_SHIFT: u16 = 12;
const SDMA_BOUNDARY: u16 = 7;

const TRANSFER_DMA: u16 = 1 << 0;
const TRANSFER_BLOCK_COUNT: u16 = 1 << 1;
const TRANSFER_AUTO_CMD12: u16 = 1 << 2;
const TRANSFER_AUTO_CMD23: u16 = 1 << 3;
const TRANSFER_TO_HOST: u16 = 1 << 4;
const TRANSFER_MULTI_BLOCK: u16 = 1 << 5;

/// Response type field of the command register; zero expects no response.
const COMMAND_RESPONSE_TYPE: u16 = 3;
const COMMAND_DATA_PRESENT: u16 = 1 << 5;

/// DMA select field of host control; zero is SDMA.
const HOST_CONTROL_DMA_SELECT: u8 = 3 << 3;

const CLOCK_INTERNAL_ENABLE: u8 = 1 << 0;
const CLOCK_INTERNAL_STABLE: u8 = 1 << 1;

const RESET_ALL: u8 = 1 << 0;
const RESET_DATA: u8 = 1 << 2;

const NORMAL_COMMAND_COMPLETE: u16 = 1 << 0;
const NORMAL_TRANSFER_COMPLETE: u16 = 1 << 1;
const NORMAL_DMA: u16 = 1 << 3;
const NORMAL_BUFFER_READ_READY: u16 = 1 << 5;
/// What a data line reset clears: transfer complete, block gap, DMA, and
/// the two buffer ready bits.
const NORMAL_DATA: u16 = 0x3E;
/// Read-only summary: some enabled error status bit is set.
const NORMAL_ERROR: u16 = 1 << 15;
const ERROR_COMMAND_TIMEOUT: u16 = 1 << 0;
const ERROR_DATA_TIMEOUT: u16 = 1 << 4;

const CONTROL_2_EXECUTE_TUNING: u16 = 1 << 6;
const CONTROL_2_SAMPLING_CLOCK_SELECT: u16 = 1 << 7;
const CONTROL_2_VERSION_4: u16 = 1 << 12;
const CONTROL_2_64_BIT_ADDRESSING: u16 = 1 << 13;

/// Starts a pad calibration cycle; reads back clear because the cycle is
/// over at once.
const AUTO_CAL_START: u32 = 1 << 31;
/// Starts a delay line calibration; likewise.
const DLLCAL_CALIBRATE: u32 = 1 << 31;

/// Tuning commands the controller wants before it settles on a sampling
/// point, by the `TRIES` field (bits 15..13) of the vendor tuning control.
const TUNING_TRIES: [u32; 8] = [40, 64, 128, 192, 256, 256, 256, 256];
const TUNING_TRIES_SHIFT: u32 = 13;

/// Specification version 3.0, vendor version 3.
const HOST_VERSION_VALUE: u16 = 0x0303;

/// 64-bit system addressing, the one capability the filesystem process
/// insists on before it goes further (it aborts without it). Nothing
/// establishes the other capability fields, so they read as absent.
const CAPABILITIES_VALUE: u32 = 1 << 28;

/// Called with the level of the controller's interrupt output.
pub type InterruptLine = Box<dyn Fn(bool) + Send + Sync>;

/// How long the card in the slot takes over data.
#[derive(Clone, Copy)]
pub struct Timing {
    /// From a command that moves data to the first block.
    pub access: Duration,
    /// For each block it sends.
    pub block_read: Duration,
    /// For each block it takes.
    pub block_write: Duration,
}

impl Timing {
    /// A card that is done before the driver can look.
    pub const INSTANT: Self = Self {
        access: Duration::ZERO,
        block_read: Duration::ZERO,
        block_write: Duration::ZERO,
    };

    /// The console's eMMC module. Nothing here was measured on a console:
    /// these are round figures for a module of its kind (HS400, a few
    /// thousand random accesses a second, reads well above a hundred
    /// megabytes a second, writes below that), which is an assumption.
    pub const EMMC: Self = Self {
        access: Duration::from_micros(150),
        block_read: Duration::from_micros(3),
        block_write: Duration::from_micros(10),
    };
}

pub struct Sdmmc {
    host: Arc<Mutex<Host>>,
    interrupt: Arc<dyn Fn(bool) + Send + Sync>,
    memory: DeviceMemory,
    notes: Arc<Notes>,
    /// Where the end of each wait for the card is sent, for the interrupt
    /// line to follow at that time. `None` with a card that never makes
    /// the controller wait.
    waits: Option<mpsc::Sender<Instant>>,
}

struct Host {
    /// The register block, byte for byte.
    registers: [u8; SIZE as usize],
    card: Option<Emmc>,
    /// The data transfer in progress, waiting at a buffer boundary.
    transfer: Option<Transfer>,
    /// Tuning commands seen since tuning was started.
    tuning_commands: u32,
    timing: Timing,
    /// How long the card takes over what the driver has just started,
    /// added up while the registers are brought to its outcome.
    busy: Duration,
    /// What the driver is told once the card is done.
    held: Option<Held>,
}

/// Interrupt status bits of an operation the card is still busy with.
struct Held {
    normal: u16,
    error: u16,
    due: Instant,
}

struct Transfer {
    to_host: bool,
    blocks_left: u32,
    /// The block crossing the bus, and how much of it has.
    block: Vec<u8>,
    moved: usize,
    /// Device address of the next byte.
    address: u64,
    /// Buffer size: the transfer pauses at each multiple.
    boundary: u64,
    /// The controller sends `STOP_TRANSMISSION` after the last block.
    auto_stop: bool,
    /// Bytes of the address register in use. Writing the last of them
    /// resumes the transfer.
    address_register: Range<usize>,
}

fn u16_at(registers: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([registers[offset], registers[offset + 1]])
}

fn u32_at(registers: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(registers[offset..offset + 4].try_into().unwrap())
}

fn set_u16(registers: &mut [u8], offset: usize, value: u16) {
    registers[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn set_u32(registers: &mut [u8], offset: usize, value: u32) {
    registers[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

/// Registers the guest cannot change by writing them.
fn is_read_only(offset: usize) -> bool {
    // Responses, present state, capabilities and maximum currents, and the
    // slot interrupt status and host version.
    matches!(offset, 0x10..0x20 | 0x24..0x28 | 0x40..0x50 | 0xFC..0x100)
}

fn reset_state() -> [u8; SIZE as usize] {
    let mut registers = [0; SIZE as usize];
    set_u32(&mut registers, PRESENT_STATE, PRESENT_STATE_IDLE);
    set_u16(&mut registers, HOST_VERSION, HOST_VERSION_VALUE);
    set_u32(&mut registers, CAPABILITIES, CAPABILITIES_VALUE);
    registers
}

impl Host {
    /// Latches normal interrupt status bits whose status enable is set.
    fn raise(&mut self, bits: u16) {
        let enabled = u16_at(&self.registers, NORMAL_STATUS_ENABLE);
        let status = u16_at(&self.registers, NORMAL_INTERRUPT_STATUS);
        set_u16(
            &mut self.registers,
            NORMAL_INTERRUPT_STATUS,
            status | bits & enabled,
        );
    }

    fn raise_error(&mut self, bits: u16) {
        let enabled = u16_at(&self.registers, ERROR_STATUS_ENABLE);
        let status = u16_at(&self.registers, ERROR_INTERRUPT_STATUS);
        set_u16(
            &mut self.registers,
            ERROR_INTERRUPT_STATUS,
            status | bits & enabled,
        );
    }

    fn status(&self) -> (u16, u16) {
        (
            u16_at(&self.registers, NORMAL_INTERRUPT_STATUS),
            u16_at(&self.registers, ERROR_INTERRUPT_STATUS),
        )
    }

    /// Shows the driver what was held back if the card is done by now,
    /// or `regardless`.
    fn settle(&mut self, regardless: bool) {
        let now = Instant::now();
        if let Some(held) = self.held.take_if(|held| regardless || held.due <= now) {
            let (normal, error) = self.status();
            set_u16(
                &mut self.registers,
                NORMAL_INTERRUPT_STATUS,
                normal | held.normal,
            );
            set_u16(
                &mut self.registers,
                ERROR_INTERRUPT_STATUS,
                error | held.error,
            );
        }
    }

    /// Holds back the status bits set since `before` for as long as the
    /// card was found to take. Returns when it is done.
    fn hold(&mut self, before: (u16, u16)) -> Option<Instant> {
        let busy = mem::take(&mut self.busy);
        if busy.is_zero() {
            return None;
        }
        let (normal, error) = self.status();
        let held = Held {
            normal: normal & !before.0,
            error: error & !before.1,
            due: Instant::now() + busy,
        };
        set_u16(
            &mut self.registers,
            NORMAL_INTERRUPT_STATUS,
            normal & before.0,
        );
        set_u16(
            &mut self.registers,
            ERROR_INTERRUPT_STATUS,
            error & before.1,
        );
        let due = held.due;
        self.held = Some(held);
        Some(due)
    }

    /// Bytes of the register holding the SDMA address.
    fn sdma_address_register(&self) -> Range<usize> {
        let control = u16_at(&self.registers, HOST_CONTROL_2);
        match (
            control & CONTROL_2_VERSION_4 != 0,
            control & CONTROL_2_64_BIT_ADDRESSING != 0,
        ) {
            (false, _) => SDMA_ADDRESS..SDMA_ADDRESS + 4,
            (true, false) => ADMA_ADDRESS..ADMA_ADDRESS + 4,
            (true, true) => ADMA_ADDRESS..ADMA_ADDRESS + 8,
        }
    }

    fn read_address(&self, register: &Range<usize>) -> u64 {
        let mut address = [0; 8];
        address[..register.len()].copy_from_slice(&self.registers[register.clone()]);
        u64::from_le_bytes(address)
    }
}

impl Sdmmc {
    /// `memory` is what the controller reaches when it masters the bus;
    /// `card` is the module in the slot, if one is fitted.
    pub fn new(
        interrupt: InterruptLine,
        memory: DeviceMemory,
        card: Option<Emmc>,
        timing: Timing,
        notes: Arc<Notes>,
    ) -> Self {
        let host = Arc::new(Mutex::new(Host {
            registers: reset_state(),
            card,
            transfer: None,
            tuning_commands: 0,
            timing,
            busy: Duration::ZERO,
            held: None,
        }));
        let interrupt: Arc<dyn Fn(bool) + Send + Sync> = Arc::from(interrupt);
        let instant = [timing.access, timing.block_read, timing.block_write]
            .iter()
            .all(Duration::is_zero);
        let waits = (!instant).then(|| {
            let (waits, ends) = mpsc::channel::<Instant>();
            let (host, interrupt) = (Arc::clone(&host), Arc::clone(&interrupt));
            // Ends with the controller: the channel closes when it goes.
            thread::spawn(move || {
                for end in ends {
                    thread::sleep(end.saturating_duration_since(Instant::now()));
                    let mut host = host.lock().unwrap();
                    host.settle(false);
                    Self::update_interrupt(&*interrupt, &host.registers);
                }
            });
            waits
        });
        Self {
            host,
            interrupt,
            memory,
            notes,
            waits,
        }
    }

    /// A command was issued. `None` if it needs something that is not
    /// modeled.
    fn issue_command(&self, host: &mut Host) -> Option<()> {
        let command = u16_at(&host.registers, COMMAND);
        let index = (command >> 8) as u8 & 0x3F;
        let argument = u32_at(&host.registers, ARGUMENT);
        match host
            .card
            .as_mut()
            .and_then(|card| card.command(index, argument))
        {
            // Nobody sent the response the controller waited for.
            None if command & COMMAND_RESPONSE_TYPE != 0 => {
                host.raise_error(ERROR_COMMAND_TIMEOUT);
                return Some(());
            }
            None => {}
            Some(Response::Short(value)) => set_u32(&mut host.registers, RESPONSE, value),
            // Bits 127..8 of the response land in bits 119..0 of the register.
            Some(Response::Long(bytes)) => {
                for (register, byte) in host.registers[RESPONSE..]
                    .iter_mut()
                    .zip(bytes.iter().rev())
                {
                    *register = *byte;
                }
            }
        }
        host.raise(NORMAL_COMMAND_COMPLETE);
        if command & COMMAND_DATA_PRESENT != 0 {
            self.start_data(host)?;
        }
        Some(())
    }

    fn refuse(&self, what: &str) -> Option<()> {
        self.notes
            .add(format!("SD/MMC host controller: {what} is not modeled"));
        None
    }

    /// Begins the data phase of the command just issued.
    fn start_data(&self, host: &mut Host) -> Option<()> {
        let control = u16_at(&host.registers, HOST_CONTROL_2);
        if control & CONTROL_2_EXECUTE_TUNING != 0 {
            // The controller samples the tuning block itself and only tells
            // the driver that one arrived.
            if !host.card.as_mut().is_some_and(Emmc::take_tuning_block) {
                host.raise_error(ERROR_DATA_TIMEOUT);
                return Some(());
            }
            host.raise(NORMAL_BUFFER_READ_READY);
            host.tuning_commands += 1;
            let tries = u32_at(&host.registers, VENDOR_TUNING_CNTRL0) >> TUNING_TRIES_SHIFT & 7;
            if host.tuning_commands >= TUNING_TRIES[tries as usize] {
                host.tuning_commands = 0;
                let done = control & !CONTROL_2_EXECUTE_TUNING | CONTROL_2_SAMPLING_CLOCK_SELECT;
                set_u16(&mut host.registers, HOST_CONTROL_2, done);
            }
            return Some(());
        }

        host.busy += host.timing.access;
        let mode = u16_at(&host.registers, TRANSFER_MODE);
        if mode & TRANSFER_DMA == 0 {
            return self.refuse("a data transfer by programmed I/O");
        }
        if host.registers[HOST_CONTROL] & HOST_CONTROL_DMA_SELECT != 0 {
            return self.refuse("a data transfer by ADMA");
        }
        if mode & TRANSFER_AUTO_CMD23 != 0 {
            return self.refuse("automatic SET_BLOCK_COUNT");
        }
        let blocks_left = match (
            mode & TRANSFER_MULTI_BLOCK != 0,
            mode & TRANSFER_BLOCK_COUNT != 0,
        ) {
            (false, _) => 1,
            (true, true) => u16_at(&host.registers, BLOCK_COUNT).into(),
            (true, false) => return self.refuse("a transfer of unlimited length"),
        };
        let size = u16_at(&host.registers, BLOCK_SIZE);
        let block_size = usize::from(size & BLOCK_SIZE_BYTES);
        let address_register = host.sdma_address_register();
        host.transfer = Some(Transfer {
            to_host: mode & TRANSFER_TO_HOST != 0,
            blocks_left,
            block: vec![0; block_size],
            // Nothing of a block is pending yet.
            moved: block_size,
            address: host.read_address(&address_register),
            boundary: 0x1000 << (size >> SDMA_BOUNDARY_SHIFT & SDMA_BOUNDARY),
            auto_stop: mode & TRANSFER_AUTO_CMD12 != 0,
            address_register,
        });
        self.run_transfer(host)
    }

    /// Moves data until the transfer completes, fails, or reaches the end
    /// of the driver's buffer. `None` if the buffer is not memory the
    /// controller can reach.
    fn run_transfer(&self, host: &mut Host) -> Option<()> {
        let Some(mut transfer) = host.transfer.take() else {
            return Some(());
        };
        let card = host.card.as_mut()?;
        loop {
            if transfer.moved == transfer.block.len() {
                if transfer.blocks_left == 0 {
                    break;
                }
                if transfer.to_host && card.send_block(&mut transfer.block).is_none() {
                    host.raise_error(ERROR_DATA_TIMEOUT);
                    return Some(());
                }
                transfer.moved = 0;
            }
            let to_boundary = transfer.boundary - transfer.address % transfer.boundary;
            let count = (transfer.block.len() - transfer.moved).min(to_boundary as usize);
            let piece = &mut transfer.block[transfer.moved..transfer.moved + count];
            let moved = if transfer.to_host {
                self.memory.write(transfer.address, piece)
            } else {
                self.memory.read(transfer.address, piece)
            };
            if moved.is_none() {
                self.notes.add(format!(
                    "SD/MMC host controller: DMA buffer at device address {:#x} is not mapped to memory",
                    transfer.address
                ));
                return None;
            }
            transfer.moved += count;
            transfer.address += count as u64;
            if transfer.moved == transfer.block.len() {
                if !transfer.to_host && card.receive_block(&transfer.block).is_none() {
                    host.raise_error(ERROR_DATA_TIMEOUT);
                    return Some(());
                }
                transfer.blocks_left -= 1;
                host.busy += match transfer.to_host {
                    true => host.timing.block_read,
                    false => host.timing.block_write,
                };
                set_u16(
                    &mut host.registers,
                    BLOCK_COUNT,
                    transfer.blocks_left as u16,
                );
            }
            let finished = transfer.moved == transfer.block.len() && transfer.blocks_left == 0;
            if transfer.address.is_multiple_of(transfer.boundary) && !finished {
                // The buffer is used up: report where the transfer stands
                // and wait for the driver to name the next address.
                Self::store_address(&mut host.registers, &transfer);
                host.raise(NORMAL_DMA);
                host.transfer = Some(transfer);
                return Some(());
            }
        }
        Self::store_address(&mut host.registers, &transfer);
        if transfer.auto_stop
            && let Some(Response::Short(status)) = card.command(12, 0)
        {
            // The response to the automatic command has the upper word.
            set_u32(&mut host.registers, RESPONSE + 12, status);
        }
        host.raise(NORMAL_TRANSFER_COMPLETE);
        Some(())
    }

    fn store_address(registers: &mut [u8], transfer: &Transfer) {
        let register = transfer.address_register.clone();
        let width = register.len();
        registers[register].copy_from_slice(&transfer.address.to_le_bytes()[..width]);
    }

    /// The driver rewrote the SDMA address: the paused transfer goes on
    /// from there.
    fn resume_transfer(&self, host: &mut Host) -> Option<()> {
        if let Some(mut transfer) = host.transfer.take() {
            transfer.address = host.read_address(&transfer.address_register);
            host.transfer = Some(transfer);
        }
        self.run_transfer(host)
    }

    fn update_interrupt(interrupt: &(dyn Fn(bool) + Send + Sync), registers: &[u8]) {
        let error = u16_at(registers, ERROR_INTERRUPT_STATUS);
        let summary = if error != 0 { NORMAL_ERROR } else { 0 };
        let normal = u16_at(registers, NORMAL_INTERRUPT_STATUS) | summary;
        let signalled = normal & u16_at(registers, NORMAL_SIGNAL_ENABLE)
            | error & u16_at(registers, ERROR_SIGNAL_ENABLE);
        interrupt(signalled != 0);
    }
}

impl MmioDevice for Sdmmc {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        let (offset, size) = (offset as usize, usize::from(size));
        if !offset.is_multiple_of(size) || size > 4 {
            return None;
        }
        let mut host = self.host.lock().unwrap();
        host.settle(false);
        let registers = &host.registers;
        let mut bytes = [0; 8];
        bytes[..size].copy_from_slice(&registers[offset..offset + size]);
        // Neither line is free while the card is at work.
        if offset == PRESENT_STATE && host.held.is_some() {
            bytes[0] |= PRESENT_STATE_INHIBIT;
        }
        // The error summary bit lives in the high byte of the normal status.
        if (offset..offset + size).contains(&(NORMAL_INTERRUPT_STATUS + 1))
            && u16_at(registers, ERROR_INTERRUPT_STATUS) != 0
        {
            bytes[NORMAL_INTERRUPT_STATUS + 1 - offset] |= (NORMAL_ERROR >> 8) as u8;
        }
        Some(u64::from_le_bytes(bytes))
    }

    fn write(&self, _core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        let (offset, size) = (offset as usize, usize::from(size));
        if !offset.is_multiple_of(size) || size > 4 {
            return None;
        }
        let mut guard = self.host.lock().unwrap();
        let host = &mut *guard;
        host.settle(false);
        for (index, byte) in value.to_le_bytes()[..size].iter().copied().enumerate() {
            let at = offset + index;
            let registers = &mut host.registers;
            match at {
                _ if is_read_only(at) => {}
                // Interrupt status bits are cleared by writing one to them.
                0x30..0x34 => registers[at] &= !byte,
                CLOCK_CONTROL => {
                    // The internal clock is stable as soon as it is enabled.
                    let stable = if byte & CLOCK_INTERNAL_ENABLE != 0 {
                        CLOCK_INTERNAL_STABLE
                    } else {
                        0
                    };
                    registers[at] = byte & !CLOCK_INTERNAL_STABLE | stable;
                }
                // Resets finish at once, so the bits never read as set.
                SOFTWARE_RESET => {
                    if byte & RESET_ALL != 0 {
                        let vendor: Vec<u8> = registers[0x100..].to_vec();
                        *registers = reset_state();
                        registers[0x100..].copy_from_slice(&vendor);
                        host.tuning_commands = 0;
                    }
                    if byte & (RESET_ALL | RESET_DATA) != 0 {
                        host.transfer = None;
                        host.held = None;
                        let status = u16_at(&host.registers, NORMAL_INTERRUPT_STATUS);
                        set_u16(
                            &mut host.registers,
                            NORMAL_INTERRUPT_STATUS,
                            status & !NORMAL_DATA,
                        );
                    }
                }
                _ => registers[at] = byte,
            }
        }
        let written = offset..offset + size;
        let self_clearing = [
            (AUTO_CAL_CONFIG, AUTO_CAL_START),
            (VENDOR_DLLCAL_CFG, DLLCAL_CALIBRATE),
        ];
        for (register, bit) in self_clearing {
            if written.contains(&(register + 3)) {
                host.registers[register + 3] &= !((bit >> 24) as u8);
            }
        }
        let resumed = host
            .transfer
            .as_ref()
            .is_some_and(|transfer| written.contains(&(transfer.address_register.end - 1)));
        let issued = written.contains(&(COMMAND + 1));
        // A driver that goes on before the card is done has stopped
        // waiting for it; what it missed is there for it to find.
        host.settle(issued || resumed);
        let before = host.status();
        let outcome = if issued {
            // The command goes out when the upper byte of its register is
            // written.
            self.issue_command(host)
        } else if resumed {
            self.resume_transfer(host)
        } else {
            Some(())
        };
        if let Some(due) = host.hold(before)
            && let Some(waits) = &self.waits
        {
            // The receiving thread lives as long as this controller.
            let _ = waits.send(due);
        }
        Self::update_interrupt(&*self.interrupt, &host.registers);
        outcome
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::{
        super::{emmc, mc},
        *,
    };
    use crate::{boot::DramSize, memory::GuestRam};

    const RAM_BASE: u64 = 0x8000_0000;
    /// Device address of the driver's buffer, and where it is in DRAM.
    const BUFFER: u64 = 0x1_0000;
    const BUFFER_PHYSICAL: u64 = RAM_BASE + 0x8000;

    struct Bench {
        sdmmc: Sdmmc,
        line: Arc<AtomicBool>,
        ram: Arc<GuestRam>,
    }

    impl Bench {
        /// A controller whose device addresses `BUFFER..BUFFER + 0x4000`
        /// map to DRAM, with a card already in the transfer state if
        /// `card` is set. The card is done at once.
        fn new(card: bool) -> Self {
            Self::timed(card, Timing::INSTANT)
        }

        fn timed(card: bool, timing: Timing) -> Self {
            let ram = Arc::new(GuestRam::new(RAM_BASE, 0x2_0000).unwrap());
            let memory_controller = Arc::new(mc::Mc::new(DramSize::Gb4));
            let write = |register: u64, value: u32| {
                memory_controller
                    .write(0, register, 4, value.into())
                    .unwrap()
            };
            // Address space 6: directory at RAM_BASE, one table after it.
            write(0x1C, 6);
            write(0x20, 0xE000_0000 | (RAM_BASE >> 12) as u32);
            write(mc::SMMU_SDMMC4A_ASID, 0x8404_0406);
            let table = RAM_BASE + 0x1000;
            ram.write(
                RAM_BASE,
                &(0xF000_0000 | (table >> 12) as u32).to_le_bytes(),
            )
            .unwrap();
            for page in 0..4 {
                let entry = 0xE000_0000 | ((BUFFER_PHYSICAL >> 12) + page) as u32;
                ram.write(table + ((BUFFER >> 12) + page) * 4, &entry.to_le_bytes())
                    .unwrap();
            }
            let line = Arc::new(AtomicBool::new(false));
            let output = Arc::clone(&line);
            let sdmmc = Sdmmc::new(
                Box::new(move |level| output.store(level, Ordering::SeqCst)),
                DeviceMemory::new(&memory_controller, &ram, mc::SMMU_SDMMC4A_ASID),
                card.then(|| emmc::tests::selected(0x80_0000)),
                timing,
                Arc::default(),
            );
            let bench = Self { sdmmc, line, ram };
            bench.write(NORMAL_STATUS_ENABLE, 2, 0xFFFF);
            bench.write(ERROR_STATUS_ENABLE, 2, 0xFFFF);
            bench.write(NORMAL_SIGNAL_ENABLE, 2, 0x000B);
            bench.write(ERROR_SIGNAL_ENABLE, 2, 0xFFFF);
            bench
        }

        fn write(&self, offset: usize, size: u8, value: u64) {
            self.sdmmc.write(0, offset as u64, size, value).unwrap();
        }

        fn read(&self, offset: usize, size: u8) -> u64 {
            self.sdmmc.read(0, offset as u64, size).unwrap()
        }

        /// Reads and clears both interrupt status registers.
        fn take_status(&self) -> u32 {
            let status = self.read(NORMAL_INTERRUPT_STATUS, 4);
            self.write(NORMAL_INTERRUPT_STATUS, 4, status);
            assert!(!self.line.load(Ordering::SeqCst));
            status as u32
        }

        /// Issues a multi-block command with the buffer at `address`, in
        /// the 64-bit SDMA mode the filesystem process uses.
        fn transfer(&self, index: u16, sector: u32, blocks: u16, address: u64, to_host: bool) {
            self.write(
                HOST_CONTROL_2,
                2,
                (CONTROL_2_VERSION_4 | CONTROL_2_64_BIT_ADDRESSING).into(),
            );
            self.write(ADMA_ADDRESS, 4, address & 0xFFFF_FFFF);
            self.write(ADMA_ADDRESS + 4, 4, address >> 32);
            // 512-byte blocks, 4 KiB buffer boundary.
            self.write(BLOCK_SIZE, 2, 0x200);
            self.write(BLOCK_COUNT, 2, blocks.into());
            self.write(ARGUMENT, 4, sector.into());
            let direction = if to_host { TRANSFER_TO_HOST } else { 0 };
            let mode = TRANSFER_DMA
                | TRANSFER_BLOCK_COUNT
                | TRANSFER_AUTO_CMD12
                | TRANSFER_MULTI_BLOCK
                | direction;
            self.write(TRANSFER_MODE, 2, mode.into());
            self.write(
                COMMAND,
                2,
                (index << 8 | COMMAND_DATA_PRESENT | 0x1A).into(),
            );
        }
    }

    #[test]
    fn the_driver_hears_of_data_only_once_the_card_is_done() {
        let access = Duration::from_millis(50);
        let bench = Bench::timed(
            true,
            Timing {
                access,
                ..Timing::INSTANT
            },
        );
        let issued = Instant::now();
        bench.transfer(18, 8, 1, BUFFER, true);
        if issued.elapsed() < access {
            assert_eq!(bench.read(NORMAL_INTERRUPT_STATUS, 4), 0);
            assert_eq!(
                bench.read(PRESENT_STATE, 4),
                u64::from(PRESENT_STATE_IDLE) | u64::from(PRESENT_STATE_INHIBIT)
            );
            assert!(!bench.line.load(Ordering::SeqCst));
        }
        // The line rises by itself, without the driver looking.
        while !bench.line.load(Ordering::SeqCst) {
            assert!(issued.elapsed() < Duration::from_secs(10), "never done");
            thread::sleep(Duration::from_millis(1));
        }
        assert!(issued.elapsed() >= access);
        assert_eq!(bench.read(PRESENT_STATE, 4), u64::from(PRESENT_STATE_IDLE));
        assert_eq!(
            bench.take_status(),
            u32::from(NORMAL_COMMAND_COMPLETE | NORMAL_TRANSFER_COMPLETE)
        );
    }

    #[test]
    fn commands_without_a_card_complete_or_time_out_by_response_type() {
        let bench = Bench::new(false);
        // CMD0, no response.
        bench.write(COMMAND, 2, 0x0000);
        assert!(bench.line.load(Ordering::SeqCst));
        assert_eq!(bench.take_status(), 0x0001);

        // CMD1 with a 48-bit response nobody sends.
        bench.write(COMMAND, 2, 0x0102);
        assert_eq!(bench.read(NORMAL_INTERRUPT_STATUS, 2), 0x8000);
        assert!(bench.line.load(Ordering::SeqCst));
        assert_eq!(bench.take_status(), 0x0001_8000);
        assert_eq!(bench.read(NORMAL_INTERRUPT_STATUS, 4), 0);
    }

    #[test]
    fn status_bits_latch_only_when_their_enable_is_set() {
        let bench = Bench::new(false);
        bench.write(ERROR_STATUS_ENABLE, 2, 0);
        bench.write(COMMAND, 2, 0x0102);
        assert_eq!(bench.read(NORMAL_INTERRUPT_STATUS, 4), 0);
    }

    #[test]
    fn clock_and_reset_complete_immediately() {
        let bench = Bench::new(false);
        bench.write(CLOCK_CONTROL, 2, 0x8001);
        assert_eq!(bench.read(CLOCK_CONTROL, 2), 0x8003);
        bench.write(HOST_CONTROL, 1, 0x20);
        bench.write(SOFTWARE_RESET, 1, 0x01);
        assert_eq!(bench.read(SOFTWARE_RESET, 1), 0);
        assert_eq!(bench.read(HOST_CONTROL, 1), 0, "host control was reset");
        assert_eq!(bench.read(PRESENT_STATE, 4), u64::from(PRESENT_STATE_IDLE));
        // Read-only registers ignore writes.
        bench.write(HOST_VERSION, 2, 0);
        assert_eq!(bench.read(HOST_VERSION, 2), u64::from(HOST_VERSION_VALUE));
    }

    #[test]
    fn sdma_transfers_pause_at_buffer_boundaries_until_the_address_is_rewritten() {
        let bench = Bench::new(true);
        // Three sectors written from a buffer that ends after the second.
        let data: Vec<u8> = (0..3 * 512).map(|index| (index / 7) as u8).collect();
        bench.ram.write(BUFFER_PHYSICAL + 0xC00, &data).unwrap();
        bench.transfer(25, 8, 3, BUFFER + 0xC00, false);
        assert_eq!(
            bench.take_status(),
            u32::from(NORMAL_COMMAND_COMPLETE | NORMAL_DMA)
        );
        assert_eq!(bench.read(BLOCK_COUNT, 2), 1);
        assert_eq!(bench.read(ADMA_ADDRESS, 4), BUFFER + 0x1000);
        bench.write(ADMA_ADDRESS, 4, BUFFER + 0x1000);
        assert_eq!(
            bench.read(NORMAL_INTERRUPT_STATUS, 2),
            0,
            "resumes with the top byte"
        );
        bench.write(ADMA_ADDRESS + 4, 4, 0);
        assert_eq!(bench.take_status(), u32::from(NORMAL_TRANSFER_COMPLETE));
        assert_eq!(bench.read(BLOCK_COUNT, 2), 0);

        // Read back to another part of the buffer, without a boundary.
        bench.transfer(18, 8, 3, BUFFER + 0x2000, true);
        assert_eq!(
            bench.take_status(),
            u32::from(NORMAL_COMMAND_COMPLETE | NORMAL_TRANSFER_COMPLETE)
        );
        let mut read = vec![0; data.len()];
        bench.ram.read(BUFFER_PHYSICAL + 0x2000, &mut read).unwrap();
        assert_eq!(read, data);
        // The automatic STOP_TRANSMISSION left the card ready for the next
        // command: its status reports the transfer state.
        bench.write(ARGUMENT, 4, 2 << 16);
        bench.write(COMMAND, 2, 13 << 8 | 0x1A);
        assert_eq!(bench.read(RESPONSE, 4) >> 9 & 0xF, 4);
    }

    #[test]
    fn a_buffer_outside_mapped_memory_is_refused_and_a_read_past_the_end_times_out() {
        let bench = Bench::new(true);
        bench.write(
            HOST_CONTROL_2,
            2,
            (CONTROL_2_VERSION_4 | CONTROL_2_64_BIT_ADDRESSING).into(),
        );
        bench.write(ADMA_ADDRESS, 4, 0x4000_0000);
        bench.write(BLOCK_SIZE, 2, 0x200);
        bench.write(ARGUMENT, 4, 0);
        bench.write(TRANSFER_MODE, 2, (TRANSFER_DMA | TRANSFER_TO_HOST).into());
        assert_eq!(
            bench.sdmmc.write(0, COMMAND as u64, 2, 17 << 8 | 0x3A),
            None
        );

        let bench = Bench::new(true);
        bench.transfer(18, 0x80_0000, 1, BUFFER, true);
        assert_eq!(
            bench.read(ERROR_INTERRUPT_STATUS, 2),
            u64::from(ERROR_DATA_TIMEOUT)
        );
    }

    #[test]
    fn tuning_takes_the_configured_number_of_commands() {
        let bench = Bench::new(true);
        // 40 tries.
        bench.write(VENDOR_TUNING_CNTRL0, 4, 0);
        bench.write(HOST_CONTROL_2, 2, CONTROL_2_EXECUTE_TUNING.into());
        for _ in 0..39 {
            bench.write(COMMAND, 2, 21 << 8 | 0x3A);
            assert_eq!(
                bench.take_status(),
                u32::from(NORMAL_COMMAND_COMPLETE | NORMAL_BUFFER_READ_READY)
            );
        }
        assert_eq!(
            bench.read(HOST_CONTROL_2, 2),
            u64::from(CONTROL_2_EXECUTE_TUNING)
        );
        bench.write(COMMAND, 2, 21 << 8 | 0x3A);
        assert_eq!(
            bench.read(HOST_CONTROL_2, 2),
            u64::from(CONTROL_2_SAMPLING_CLOCK_SELECT)
        );
    }
}
