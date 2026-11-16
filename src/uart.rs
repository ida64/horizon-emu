//! Tegra X1 UART: a 16550 with one register per 32-bit word, as the kernel's
//! debug log drives it.
//!
//! # Which port
//!
//! The kernel has exactly one candidate: the memory layout registers
//! `0x70006000..0x70006040` (UART-A) as the only device region whose type
//! contains all of the bits `0x1D` (`0x80061014`, type `0x1000001D`), and the
//! driver takes its base from the first region in address order with those
//! bits (`0x800d5e00`-`0x800d5e24`). The argument of the driver's initializer
//! is not a port number but a register layout: 0, 1 and 2 pick accessors that
//! treat registers as bytes, halfwords or words spaced as far apart as they
//! are wide (tables at `0x800da220`/`0x800da238`), and the kernel always
//! passes 2 (`0x8006a238`). Every access it makes is therefore 32 bits wide
//! at `4 * index`.
//!
//! # What the kernel does with it
//!
//! | Access | Where |
//! |---|---|
//! | IER = 0 | initialization, its only step (`0x800d5e58`) |
//! | read LSR until `TX_FIFO_FULL` (bit 8) is clear, then THR = byte | every character (`0x800d618c`-`0x800d61a8`) |
//! | read LSR until `TMTY` (bit 6) is set | end of every string (`0x800d5fd0`) |
//! | read LCR, IER, IIR; LCR = 0x80; read LCR, DLL, DLM; LCR = saved; read LCR | before a core suspends (`0x800d6000`) |
//! | LCR = 0x80; read LCR; DLL, DLM = saved; read DLM; LCR = saved; read LCR; IER = saved; FCR = saved IIR \| 6; read IIR | after it resumes (`0x800d60c0`) |
//!
//! The two LSR loops have no timeout: the log hangs with interrupts masked
//! if the port never reports room or never drains. Nothing else is touched
//! to bring the port up: no divisor, line format or FIFO setup, and no
//! clock, reset, pin or GPIO register, so on a console the baud rate and
//! framing are whatever the boot chain left behind.
//!
//! The log is live only when secure monitor config item 11 is non-zero: a
//! static constructor (`0x800d3e20`, third call) stores `item 11 == 0` at
//! `0x800d7480` (`0x800d3eb8`), and every log entry point returns at once
//! unless that word and the one at `0x800db004` are both zero
//! (`0x8006a1c0`-`0x8006a1dc`); the latter is cleared right before the log is
//! initialized (`0x80062a10`). All the retail kernel ever prints is
//! `"Break() called. "` (`0x800b0854`) and `"Exception occurred. "`
//! (`0x800c8c28`, `0x800c9164`), each followed by sixteen hex digits read
//! from the current process and a line feed, with a carriage return sent
//! before every line feed (`0x80069d40`).
//!
//! # Model
//!
//! Transmission is instantaneous: a byte written to THR is handed to the
//! host sink before the write returns, so the transmitter and its FIFO are
//! always empty. Nothing is ever received and no modem-control input ever
//! changes, which leaves the THR-empty interrupt as the only one that can
//! be raised. The sink gets the low eight bits of every THR write as they
//! are; word length, parity, stop bits, break and the divisor are stored
//! but describe a wire that does not exist here, and the kernel relies on
//! its boot chain rather than on itself to program them.
//!
//! Registers are addressed by the offset of their word, with any of the
//! three access widths reaching the low-order bytes. Whatever has no
//! defined answer in this model is refused rather than invented: the
//! receive buffer (there is never a byte in it), the modem status inputs
//! (their wiring is unknown), the vendor registers from `0x20` up (IrDA,
//! RX FIFO configuration, modem interrupt enable, vendor status, auto
//! baud), bytes of a word above its first, and writes that set reserved
//! bits or select loopback, CTS flow control or DMA mode 1.

use std::{
    io::{self, Write},
    sync::Mutex,
};

use serde::{Deserialize, Serialize};

use crate::snapshot::{Snapshot, SnapshotError, decode, encode};

/// Physical base of UART-A, the port the kernel logs to (`0x80061014`).
pub const UART_A_BASE: u64 = 0x7000_6000;

/// Size of the register block, as the kernel declares it (`0x80061014`).
pub const REGISTER_BLOCK_SIZE: u64 = 0x40;

const THR_RBR_DLL: u64 = 0x00;
const IER_DLM: u64 = 0x04;
const IIR_FCR: u64 = 0x08;
const LCR: u64 = 0x0C;
const MCR: u64 = 0x10;
const LSR: u64 = 0x14;
const SPR: u64 = 0x1C;

const IER_IE_THR: u8 = 1 << 1;
/// RHR, THR, RXS, MSI, RX_TIMEOUT and EORD enables.
const IER_DEFINED: u8 = 0x3F;

/// `IS_STA`: set while no interrupt is pending.
const IIR_NO_INTERRUPT: u8 = 1 << 0;
const IIR_THR_EMPTY: u8 = 1 << 1;
const IIR_FIFO_MODE: u8 = 3 << 6;

const FCR_EN_FIFO: u8 = 1 << 0;
/// Everything but DMA mode 1, which changes when the port raises DMA
/// requests that nothing here listens to.
const FCR_MODELED: u8 = 0xF7;

const LCR_DLAB: u8 = 1 << 7;

/// DTR, RTS, OUT1, OUT2 and RTS_EN: output pins and receive-side flow
/// control. LOOPBK would feed the transmitter into the receiver and CTS_EN
/// would gate it on an input pin.
const MCR_MODELED: u8 = 0x4F;

const LSR_THRE: u32 = 1 << 5;
const LSR_TMTY: u32 = 1 << 6;
const LSR_RX_FIFO_EMPTY: u32 = 1 << 9;
/// The only line status there is: holding register, shift register and TX
/// FIFO drained (`TX_FIFO_FULL`, bit 8, clear), nothing received, no errors.
const LSR_IDLE: u32 = LSR_THRE | LSR_TMTY | LSR_RX_FIFO_EMPTY;

pub struct Uart {
    state: Mutex<State>,
}

struct State {
    sink: Box<dyn Write + Send>,
    /// First failure of the sink since it was last collected.
    sink_error: Option<io::Error>,
    registers: Registers,
}

/// What the guest sees of the port: everything but the host-side sink.
#[derive(Default, Serialize, Deserialize)]
struct Registers {
    divisor_low: u8,
    divisor_high: u8,
    interrupt_enable: u8,
    fifo_enabled: bool,
    line_control: u8,
    modem_control: u8,
    scratch: u8,
    /// The holding register became empty, or its interrupt was enabled while
    /// it was, and no IIR read has reported that yet.
    thr_empty_event: bool,
}

impl Uart {
    /// A port in its reset state. Bytes the guest transmits are written to
    /// `sink` in order, each flushed before the write that sent it returns.
    pub fn new(sink: Box<dyn Write + Send>) -> Self {
        Self {
            state: Mutex::new(State {
                sink,
                sink_error: None,
                registers: Registers::default(),
            }),
        }
    }

    /// MMIO read at `offset` from the port's base, `size` bytes wide (1, 2
    /// or 4). `None` means the register or width is not modeled and nothing
    /// happened.
    pub fn read(&self, offset: u64, size: u8) -> Option<u32> {
        let width = width_mask(size)?;
        let registers = &mut self.state.lock().unwrap().registers;
        let value = match offset {
            THR_RBR_DLL if registers.divisor_latch() => registers.divisor_low.into(),
            IER_DLM if registers.divisor_latch() => registers.divisor_high.into(),
            IER_DLM => registers.interrupt_enable.into(),
            IIR_FCR => registers.read_interrupt_identification().into(),
            LCR => registers.line_control.into(),
            MCR => registers.modem_control.into(),
            LSR => LSR_IDLE,
            SPR => registers.scratch.into(),
            _ => return None,
        };
        Some(value & width)
    }

    /// MMIO write at `offset` from the port's base, `size` bytes wide (1, 2
    /// or 4). `None` means the register, width or value is not modeled and
    /// nothing happened.
    pub fn write(&self, offset: u64, size: u8, value: u32) -> Option<()> {
        let value = value & width_mask(size)?;
        let mut state = self.state.lock().unwrap();
        let registers = &mut state.registers;
        match offset {
            THR_RBR_DLL if registers.divisor_latch() => {
                registers.divisor_low = defined(value, 0xFF)?
            }
            THR_RBR_DLL => state.transmit(defined(value, 0xFF)?),
            IER_DLM if registers.divisor_latch() => registers.divisor_high = defined(value, 0xFF)?,
            IER_DLM => registers.set_interrupt_enable(defined(value, IER_DEFINED)?),
            // Trigger levels and the self-clearing FIFO resets act on FIFO
            // contents, and both FIFOs are always empty.
            IIR_FCR => registers.fifo_enabled = defined(value, FCR_MODELED)? & FCR_EN_FIFO != 0,
            LCR => registers.line_control = defined(value, 0xFF)?,
            MCR => registers.modem_control = defined(value, MCR_MODELED)?,
            SPR => registers.scratch = defined(value, 0xFF)?,
            _ => return None,
        }
        Some(())
    }

    /// Level of the port's interrupt output.
    pub fn interrupt(&self) -> bool {
        self.state.lock().unwrap().registers.interrupt()
    }

    /// Takes the first error the sink has returned since the last call. The
    /// guest cannot observe a failing sink: the bytes are lost on the host
    /// side of the wire.
    pub fn take_sink_error(&self) -> Option<io::Error> {
        self.state.lock().unwrap().sink_error.take()
    }
}

/// Saves the port's registers and its pending THR-empty event; the host sink
/// and its errors stay with the restoring machine. Its interrupt output is
/// derived from the registers, so nothing else is driven on restore.
impl Snapshot for Uart {
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        encode(&self.state.lock().unwrap().registers)
    }

    fn restore(&self, state: &[u8]) -> Result<(), SnapshotError> {
        self.state.lock().unwrap().registers = decode("UART-A", state)?;
        Ok(())
    }
}

impl Registers {
    /// DLAB: offsets 0 and 4 address the divisor instead of data and IER.
    fn divisor_latch(&self) -> bool {
        self.line_control & LCR_DLAB != 0
    }

    fn interrupt(&self) -> bool {
        self.interrupt_enable & IER_IE_THR != 0 && self.thr_empty_event
    }

    /// Reading IIR acknowledges the THR-empty interrupt, but only when that
    /// is what the read reports.
    fn read_interrupt_identification(&mut self) -> u8 {
        let mode = if self.fifo_enabled { IIR_FIFO_MODE } else { 0 };
        if self.interrupt() {
            self.thr_empty_event = false;
            mode | IIR_THR_EMPTY
        } else {
            mode | IIR_NO_INTERRUPT
        }
    }

    /// The holding register is always empty, so enabling its interrupt
    /// raises it; rewriting an already set enable does not.
    fn set_interrupt_enable(&mut self, value: u8) {
        if value & !self.interrupt_enable & IER_IE_THR != 0 {
            self.thr_empty_event = true;
        }
        self.interrupt_enable = value;
    }
}

impl State {
    /// The flush is what makes the idle line status true: once the guest is
    /// told the transmitter has drained, the byte has left the emulator.
    fn transmit(&mut self, byte: u8) {
        let sent = self
            .sink
            .write_all(&[byte])
            .and_then(|()| self.sink.flush());
        if let Err(error) = sent {
            self.sink_error.get_or_insert(error);
        }
        self.registers.thr_empty_event = true;
    }
}

fn width_mask(size: u8) -> Option<u32> {
    match size {
        1 => Some(0xFF),
        2 => Some(0xFFFF),
        4 => Some(0xFFFF_FFFF),
        _ => None,
    }
}

/// The register value a write carries, unless it sets bits outside `mask`.
fn defined(value: u32, mask: u8) -> Option<u8> {
    (value & !u32::from(mask) == 0).then_some(value as u8)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[derive(Clone, Default)]
    struct Wire(Arc<Mutex<Vec<u8>>>);

    impl Wire {
        fn bytes(&self) -> Vec<u8> {
            self.0.lock().unwrap().clone()
        }
    }

    impl Write for Wire {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    /// Fails every write until the shared flag is cleared.
    struct Flaky(Arc<Mutex<bool>>, Wire);

    impl Write for Flaky {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if *self.0.lock().unwrap() {
                return Err(io::ErrorKind::BrokenPipe.into());
            }
            self.1.write(bytes)
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn uart() -> (Uart, Wire) {
        let wire = Wire::default();
        (Uart::new(Box::new(wire.clone())), wire)
    }

    /// The kernel's character output (`0x800d6180`) and end-of-string wait
    /// (`0x800d5fd0`), failing instead of spinning where the kernel would.
    fn kernel_print(uart: &Uart, text: &[u8]) {
        for &byte in text {
            assert_eq!(uart.read(LSR, 4).unwrap() & 1 << 8, 0, "TX FIFO full");
            uart.write(THR_RBR_DLL, 4, byte.into()).unwrap();
        }
        assert_ne!(uart.read(LSR, 4).unwrap() & 1 << 6, 0, "transmitter busy");
    }

    #[test]
    fn kernel_log_reaches_the_sink_byte_for_byte() {
        let (uart, wire) = uart();
        uart.write(IER_DLM, 4, 0).unwrap();
        let text = b"Exception occurred. 0100000000001000\r\n\x00\xFF";
        kernel_print(&uart, text);
        assert_eq!(wire.bytes(), text);
    }

    #[test]
    fn line_status_is_idle_at_every_width() {
        let (uart, _) = uart();
        for _ in 0..2 {
            assert_eq!(uart.read(LSR, 4), Some(0x260));
            assert_eq!(uart.read(LSR, 2), Some(0x260));
            assert_eq!(uart.read(LSR, 1), Some(0x60));
            uart.write(THR_RBR_DLL, 4, 0x41).unwrap();
        }
    }

    #[test]
    fn divisor_latch_banks_data_and_interrupt_enable() {
        let (uart, wire) = uart();
        uart.write(IER_DLM, 4, 0x05).unwrap();

        uart.write(LCR, 4, 0x83).unwrap();
        assert_eq!(uart.read(THR_RBR_DLL, 4), Some(0));
        assert_eq!(uart.read(IER_DLM, 4), Some(0));
        uart.write(THR_RBR_DLL, 4, 0xDD).unwrap();
        uart.write(IER_DLM, 4, 0xC0).unwrap();
        assert_eq!(uart.read(THR_RBR_DLL, 4), Some(0xDD));
        assert_eq!(uart.read(IER_DLM, 4), Some(0xC0));
        assert_eq!(wire.bytes(), b"", "a divisor write is not a transmission");

        uart.write(LCR, 4, 0x03).unwrap();
        assert_eq!(
            uart.read(IER_DLM, 4),
            Some(0x05),
            "DLM write leaked into IER"
        );
        assert_eq!(uart.read(THR_RBR_DLL, 4), None, "receive buffer is empty");
        uart.write(THR_RBR_DLL, 4, 0x55).unwrap();
        uart.write(IER_DLM, 4, 0x01).unwrap();
        assert_eq!(wire.bytes(), b"\x55");

        uart.write(LCR, 4, 0x83).unwrap();
        assert_eq!(
            uart.read(THR_RBR_DLL, 4),
            Some(0xDD),
            "THR write leaked into DLL"
        );
        assert_eq!(
            uart.read(IER_DLM, 4),
            Some(0xC0),
            "IER write leaked into DLM"
        );
    }

    /// The exact access sequences of `0x800d6000` and `0x800d60c0` around a
    /// suspend that loses the port's state.
    #[test]
    fn kernel_suspend_sequence_round_trips() {
        let (before, _) = uart();
        before.write(LCR, 4, 0x80).unwrap();
        before.write(THR_RBR_DLL, 4, 0xDD).unwrap();
        before.write(IER_DLM, 4, 0x01).unwrap();
        before.write(LCR, 4, 0x03).unwrap();
        before.write(IIR_FCR, 4, 0x01).unwrap();

        let lcr = before.read(LCR, 4).unwrap();
        let ier = before.read(IER_DLM, 4).unwrap();
        let iir = before.read(IIR_FCR, 4).unwrap();
        before.write(LCR, 4, 0x80).unwrap();
        before.read(LCR, 4).unwrap();
        let dll = before.read(THR_RBR_DLL, 4).unwrap();
        let dlm = before.read(IER_DLM, 4).unwrap();
        before.write(LCR, 4, lcr).unwrap();
        assert_eq!(before.read(LCR, 4), Some(0x03));
        assert_eq!((lcr, ier, iir, dll, dlm), (0x03, 0x00, 0xC1, 0xDD, 0x01));

        let (after, wire) = uart();
        after.write(LCR, 4, 0x80).unwrap();
        after.read(LCR, 4).unwrap();
        after.write(THR_RBR_DLL, 4, dll).unwrap();
        after.write(IER_DLM, 4, dlm).unwrap();
        after.read(IER_DLM, 4).unwrap();
        after.write(LCR, 4, lcr).unwrap();
        after.read(LCR, 4).unwrap();
        after.write(IER_DLM, 4, ier).unwrap();
        after.write(IIR_FCR, 4, iir | 6).unwrap();
        assert_eq!(after.read(IIR_FCR, 4), Some(0xC1));
        assert_eq!(after.read(LCR, 4), Some(0x03));
        assert_eq!(after.read(IER_DLM, 4), Some(0x00));
        assert_eq!(wire.bytes(), b"", "restoring the divisor transmitted");
        kernel_print(&after, b"ok");
        assert_eq!(wire.bytes(), b"ok");
    }

    #[test]
    fn narrow_accesses_reach_the_low_bytes_only() {
        let (uart, wire) = uart();
        uart.write(THR_RBR_DLL, 1, 0x1234_5641).unwrap();
        uart.write(THR_RBR_DLL, 2, 0x1234_0042).unwrap();
        assert_eq!(wire.bytes(), b"AB");

        assert_eq!(uart.write(THR_RBR_DLL, 2, 0x0143), None);
        assert_eq!(uart.write(THR_RBR_DLL, 4, 0x0001_0043), None);
        assert_eq!(wire.bytes(), b"AB", "a refused write transmitted");

        uart.write(SPR, 1, 0xA5).unwrap();
        assert_eq!(uart.read(SPR, 4), Some(0xA5));
        for offset in [1, 2, 3, 0x1D, 0x1E] {
            assert_eq!(uart.read(offset, 1), None, "offset {offset:#x}");
            assert_eq!(uart.write(offset, 1, 0), None, "offset {offset:#x}");
        }
        for size in [0, 3, 8] {
            assert_eq!(uart.read(SPR, size), None, "size {size}");
            assert_eq!(uart.write(THR_RBR_DLL, size, 0x44), None, "size {size}");
        }
        assert_eq!(wire.bytes(), b"AB");
    }

    #[test]
    fn thr_empty_interrupt_follows_enable_and_acknowledge() {
        let (uart, _) = uart();
        uart.write(THR_RBR_DLL, 4, 0x41).unwrap();
        assert!(!uart.interrupt(), "raised while disabled");
        assert_eq!(uart.read(IIR_FCR, 4), Some(0x01));

        uart.write(IER_DLM, 4, 0x02).unwrap();
        assert!(uart.interrupt(), "enabling with an empty THR must raise");
        assert_eq!(uart.read(IIR_FCR, 4), Some(0x02));
        assert!(!uart.interrupt(), "IIR read must acknowledge");
        assert_eq!(uart.read(IIR_FCR, 4), Some(0x01));

        uart.write(IER_DLM, 4, 0x03).unwrap();
        assert!(!uart.interrupt(), "rewriting a set enable raised again");

        uart.write(THR_RBR_DLL, 4, 0x42).unwrap();
        assert!(uart.interrupt(), "THR drained again");

        uart.write(IER_DLM, 4, 0x01).unwrap();
        assert!(!uart.interrupt());
        assert_eq!(uart.read(IIR_FCR, 4), Some(0x01), "masked source reported");
        uart.write(IER_DLM, 4, 0x02).unwrap();
        assert_eq!(uart.read(IIR_FCR, 4), Some(0x02));
    }

    #[test]
    fn fifo_mode_shows_in_interrupt_identification() {
        let (uart, _) = uart();
        assert_eq!(uart.read(IIR_FCR, 4), Some(0x01));
        uart.write(IIR_FCR, 4, 0xC7).unwrap();
        assert_eq!(uart.read(IIR_FCR, 4), Some(0xC1));
        uart.write(IIR_FCR, 4, 0x06).unwrap();
        assert_eq!(uart.read(IIR_FCR, 4), Some(0x01));
    }

    #[test]
    fn unmodeled_accesses_are_refused_without_effect() {
        let (uart, wire) = uart();
        uart.write(MCR, 4, 0x43).unwrap();
        uart.write(IER_DLM, 4, 0x3D).unwrap();
        uart.write(IIR_FCR, 4, 0x01).unwrap();

        assert_eq!(uart.write(MCR, 4, 0x13), None, "loopback");
        assert_eq!(uart.write(MCR, 4, 0x23), None, "CTS flow control");
        assert_eq!(uart.write(IER_DLM, 4, 0x40), None, "reserved IER bit");
        assert_eq!(uart.write(IIR_FCR, 4, 0x08), None, "DMA mode 1");
        assert_eq!(uart.write(LSR, 4, 0), None, "LSR is read-only");
        assert_eq!(uart.read(THR_RBR_DLL, 4), None, "receive buffer");
        for offset in [0x18, 0x20, 0x24, 0x28, 0x2C, 0x30, 0x3C, 0x40, 1 << 32] {
            assert_eq!(uart.read(offset, 4), None, "offset {offset:#x}");
            assert_eq!(uart.write(offset, 4, 0), None, "offset {offset:#x}");
        }

        assert_eq!(uart.read(MCR, 4), Some(0x43));
        assert_eq!(uart.read(IER_DLM, 4), Some(0x3D));
        assert_eq!(uart.read(IIR_FCR, 4), Some(0xC1));
        assert_eq!(wire.bytes(), b"");
    }

    #[test]
    fn sink_failure_is_recorded_and_invisible_to_the_guest() {
        let failing = Arc::new(Mutex::new(true));
        let wire = Wire::default();
        let uart = Uart::new(Box::new(Flaky(failing.clone(), wire.clone())));

        assert_eq!(uart.write(THR_RBR_DLL, 4, 0x41), Some(()));
        assert_eq!(uart.write(THR_RBR_DLL, 4, 0x42), Some(()));
        assert_eq!(uart.read(LSR, 4), Some(0x260));
        let error = uart.take_sink_error().expect("the failure was dropped");
        assert_eq!(error.kind(), io::ErrorKind::BrokenPipe);
        assert!(uart.take_sink_error().is_none(), "reported twice");

        *failing.lock().unwrap() = false;
        uart.write(THR_RBR_DLL, 4, 0x43).unwrap();
        assert!(uart.take_sink_error().is_none());
        assert_eq!(wire.bytes(), b"C");
    }
}
