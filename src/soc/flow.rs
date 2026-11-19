//! Flow controller: halts and wakes the processors.
//!
//! What is modelled is what the boot processor (BPMP, "COP" in the
//! register names) does with it: the boot chain's code halts the BPMP
//! through `HALT_COP_EVENTS` to sleep for a number of microseconds or
//! milliseconds, to wait for an interrupt, or for good once it has started
//! the CPU cluster (the public reimplementation hekate's `bpmp_usleep`,
//! `bpmp_msleep` and `bpmp_halt`, `bdk/soc/bpmp.c`). The machine's BPMP
//! thread asks [`FlowController::cop_status`] after every instruction that
//! may have halted it and waits accordingly.
//!
//! Register offsets and fields are as in hekate `bdk/soc/t210.h` (`Flow
//! controller registers`) and Linux `include/soc/tegra/flowctrl.h`:
//!
//! | offset | register | here |
//! |---|---|---|
//! | `0x000`, `0x014`, `0x01c`, `0x024` | `HALT_CPU0..3_EVENTS` | storage; only mode 0 (no halt) accepted |
//! | `0x004` | `HALT_COP_EVENTS` | halts the BPMP, see below |
//! | `0x008`, `0x018`, `0x020`, `0x028` | `CPU0..3_CSR` | storage; power-gating sequencing (`ENABLE`, bit 0) refused |
//! | `0x00c` | `COP_CSR` | storage, same rule |
//! | `0x040` | `RAM_REPAIR` | a request (bit 0) completes at once: status (bit 1) set |
//! | `0x098` | `BPMP_CLUSTER_CONTROL` | storage (bit 0 `ACTIVE_SLOW`: which CPU cluster is active) |
//!
//! Anything else is refused.
//!
//! `HALT_COP_EVENTS`: bits 31:29 the mode, 28 `JTAG`, 25 `USEC`, 24
//! `MSEC`, 23 `SEC`, 11 `LIC_IRQ`, 10 `LIC_FIQ`, 9 `GIC_IRQ`, 8 `GIC_FIQ`,
//! 7:0 the count of the timer event. Modes: 0 none (runs), 2 `WAITEVENT`
//! (halted until one of the selected events), 4 `STOP_UNTIL_IRQ` (halted
//! until an interrupt). The modes that also raise an interrupt (1, 3, 5)
//! and mode 6 are refused: nothing here raises flow controller
//! interrupts. In `WAITEVENT` the timer event ends the halt after `count`
//! units of guest time (hekate counts on exactly that: one microsecond per
//! count with `USEC`); `LIC_IRQ`/`LIC_FIQ` end it when the legacy
//! interrupt controller's IRQ or FIQ to the BPMP is asserted; `JTAG` is an
//! event no debugger here ever produces, so with it alone (hekate's
//! `bpmp_halt`) the BPMP stays halted until something writes the register
//! again. The BPMP's interrupts come from the legacy controller, not the
//! CPU cluster's GIC, so `GIC_IRQ`/`GIC_FIQ` in the BPMP's register are
//! refused. \[assumption\] After a timed or interrupt wake the register
//! keeps the value written; nothing public says what it reads back.
//! \[assumption\] A count of zero ends a timed halt at once.
//!
//! Reset values: every register modelled here reads zero after reset
//! (\[assumption\], from the public drivers treating zero as "no halt, no
//! power-gating"; the boot chain overwrites them before relying on them,
//! except `BPMP_CLUSTER_CONTROL`, which it changes by read-modify-write of
//! bit 0 only).

use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use serde::{Deserialize, Serialize};

use super::{Notes, regs::word};
use crate::{
    bus::MmioDevice,
    clock::GuestClock,
    snapshot::{Snapshot, SnapshotError, decode, encode},
};

pub const BASE: u64 = 0x6000_7000;
pub const SIZE: u64 = 0x1000;

const HALT_CPU_EVENTS: [u64; 4] = [0x000, 0x014, 0x01C, 0x024];
const HALT_COP_EVENTS: u64 = 0x004;
const CPU_CSR: [u64; 4] = [0x008, 0x018, 0x020, 0x028];
const COP_CSR: u64 = 0x00C;
const RAM_REPAIR: u64 = 0x040;
const BPMP_CLUSTER_CONTROL: u64 = 0x098;
/// Bits of `RAM_REPAIR` that read set before any request: the NX
/// bootloader's hardware-state check requires `RAM_REPAIR & 0xff0d == 4`
/// (`nxbl_hwcheck.csv`). What bit 2 means is not public; it is kept as is.
const RAM_REPAIR_AT_RESET: u32 = 1 << 2;

/// Registers whose function is not modelled, as the NX bootloader's
/// hardware-state check requires them to read when it starts
/// (`nxbl_hwcheck.csv`, text `0x40002A7C`; the unchecked bits clear,
/// \[assumption\]): `CLUSTER_CONTROL` (`0x2c`), `CPU_PWR_CSR` (`0x38`),
/// and unnamed ones. Read only: writes to them are refused.
const READ_ONLY: [(u64, u32); 22] = [
    (0x010, 0),
    (0x02C, 0x0400_4000),
    (0x030, 0),
    (0x034, 0),
    (0x038, 0x100),
    (0x03C, 0),
    (0x044, 0),
    (0x048, 0),
    (0x04C, 0),
    (0x050, 0),
    (0x054, 0xFFFF_0000),
    (0x05C, 0),
    (0x060, 0),
    (0x064, 0),
    (0x068, 0),
    (0x06C, 0),
    (0x070, 0),
    (0x074, 0),
    (0x078, 0),
    (0x08C, 0),
    (0x090, 0),
    (0x094, 1),
];

const MODE_SHIFT: u32 = 29;
const MODE_NONE: u32 = 0;
const MODE_WAITEVENT: u32 = 2;
const MODE_STOP_UNTIL_IRQ: u32 = 4;
const EVENT_JTAG: u32 = 1 << 28;
const EVENT_USEC: u32 = 1 << 25;
const EVENT_MSEC: u32 = 1 << 24;
const EVENT_SEC: u32 = 1 << 23;
const EVENT_LIC_IRQ: u32 = 1 << 11;
const EVENT_LIC_FIQ: u32 = 1 << 10;
const EVENT_GIC_IRQ: u32 = 1 << 9;
const EVENT_GIC_FIQ: u32 = 1 << 8;
const EVENT_COUNT: u32 = 0xFF;
/// The event bits this model knows.
const EVENTS: u32 = EVENT_JTAG
    | EVENT_USEC
    | EVENT_MSEC
    | EVENT_SEC
    | EVENT_LIC_IRQ
    | EVENT_LIC_FIQ
    | EVENT_GIC_IRQ
    | EVENT_GIC_FIQ
    | EVENT_COUNT;

const CSR_ENABLE: u32 = 1 << 0;
/// `CSR_EVENT_FLAG` and `CSR_INTR_FLAG`: write one to clear.
const CSR_FLAGS: u32 = 1 << 14 | 1 << 15;

const RAM_REPAIR_REQ: u32 = 1 << 0;
const RAM_REPAIR_STS: u32 = 1 << 1;

/// What holds the BPMP.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CopHalt {
    /// Not halted.
    Running,
    /// Until guest time reaches this many nanoseconds, or an interrupt
    /// selected by `irq`/`fiq` arrives.
    Until { nanos: u64, irq: bool, fiq: bool },
    /// Until the BPMP's IRQ (`irq`) or FIQ (`fiq`) input is asserted.
    Interrupt { irq: bool, fiq: bool },
    /// Until something writes `HALT_COP_EVENTS` again: the halt waits for
    /// an event nothing here produces.
    Forever,
}

/// Whether the BPMP may run, as [`FlowController::cop_status`] tells it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CopStatus {
    Running,
    /// Halted; nothing can end the halt before this moment of guest time
    /// unless an interrupt (if one is selected) or a register write does.
    HaltedUntil(Duration),
    /// Halted until an interrupt or a register write.
    Halted,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct State {
    halt_cop: u32,
    cop: CopHalt,
    halt_cpu: [u32; 4],
    cpu_csr: [u32; 4],
    cop_csr: u32,
    ram_repair: u32,
    bpmp_cluster_control: u32,
}

pub struct FlowController {
    state: Mutex<State>,
    /// Set whenever the BPMP's halt changes, so the BPMP thread need not
    /// take the lock after every instruction.
    halt_changed: AtomicBool,
    clock: Arc<GuestClock>,
    notes: Arc<Notes>,
}

impl FlowController {
    pub fn new(clock: Arc<GuestClock>, notes: Arc<Notes>) -> Self {
        Self {
            state: Mutex::new(State {
                halt_cop: 0,
                cop: CopHalt::Running,
                halt_cpu: [0; 4],
                cpu_csr: [0; 4],
                cop_csr: 0,
                ram_repair: RAM_REPAIR_AT_RESET,
                bpmp_cluster_control: 0,
            }),
            halt_changed: AtomicBool::new(false),
            clock,
            notes,
        }
    }

    /// Whether a register write may have halted or released the BPMP since
    /// the last call; clears the flag. Cheap: for the BPMP thread to call
    /// after every instruction.
    pub fn take_halt_change(&self) -> bool {
        self.halt_changed.swap(false, Ordering::AcqRel)
    }

    /// Where the BPMP stands, given the levels of its IRQ and FIQ inputs.
    /// A timed halt whose time has come, or an interrupt halt whose
    /// interrupt is asserted, ends here.
    pub fn cop_status(&self, irq: bool, fiq: bool) -> CopStatus {
        let mut state = self.state.lock().unwrap();
        let now = self.clock.now();
        let woken = |want_irq: bool, want_fiq: bool| want_irq && irq || want_fiq && fiq;
        let status = match state.cop {
            CopHalt::Running => CopStatus::Running,
            CopHalt::Until {
                nanos,
                irq: i,
                fiq: f,
            } => {
                let deadline = Duration::from_nanos(nanos);
                if now >= deadline || woken(i, f) {
                    CopStatus::Running
                } else {
                    CopStatus::HaltedUntil(deadline)
                }
            }
            CopHalt::Interrupt { irq: i, fiq: f } if woken(i, f) => CopStatus::Running,
            CopHalt::Interrupt { .. } | CopHalt::Forever => CopStatus::Halted,
        };
        if status == CopStatus::Running {
            state.cop = CopHalt::Running;
        }
        status
    }

    /// The halt a write of `value` to `HALT_COP_EVENTS` puts the BPMP in,
    /// or why it is refused.
    fn cop_halt(&self, value: u32) -> Result<CopHalt, String> {
        let mode = value >> MODE_SHIFT;
        let events = value & !(7 << MODE_SHIFT);
        if events & !EVENTS != 0 {
            return Err(format!(
                "HALT_COP_EVENTS {value:#010x}: event bits {:#x} not modelled",
                events & !EVENTS
            ));
        }
        if events & (EVENT_GIC_IRQ | EVENT_GIC_FIQ) != 0 {
            return Err(format!(
                "HALT_COP_EVENTS {value:#010x}: GIC events for the BPMP, whose \
                 interrupts come from the legacy interrupt controller"
            ));
        }
        let irq = events & EVENT_LIC_IRQ != 0;
        let fiq = events & EVENT_LIC_FIQ != 0;
        match mode {
            MODE_NONE => Ok(CopHalt::Running),
            MODE_WAITEVENT => {
                let timers = [
                    (EVENT_USEC, Duration::from_micros(1)),
                    (EVENT_MSEC, Duration::from_millis(1)),
                    (EVENT_SEC, Duration::from_secs(1)),
                ];
                let selected: Vec<_> = timers.iter().filter(|(bit, _)| events & bit != 0).collect();
                match selected[..] {
                    [] if irq || fiq => Ok(CopHalt::Interrupt { irq, fiq }),
                    [] => Ok(CopHalt::Forever),
                    [(_, unit)] => {
                        let until = self.clock.now() + *unit * (events & EVENT_COUNT);
                        Ok(CopHalt::Until {
                            nanos: until.as_nanos() as u64,
                            irq,
                            fiq,
                        })
                    }
                    _ => Err(format!(
                        "HALT_COP_EVENTS {value:#010x}: more than one timer event"
                    )),
                }
            }
            MODE_STOP_UNTIL_IRQ if events & (EVENT_USEC | EVENT_MSEC | EVENT_SEC) != 0 => Err(
                format!("HALT_COP_EVENTS {value:#010x}: timer events in STOP_UNTIL_IRQ mode"),
            ),
            // Stopped until an interrupt: which one the event bits select;
            // with none selected, either. \[assumption\]
            MODE_STOP_UNTIL_IRQ if irq || fiq => Ok(CopHalt::Interrupt { irq, fiq }),
            MODE_STOP_UNTIL_IRQ => Ok(CopHalt::Interrupt {
                irq: true,
                fiq: true,
            }),
            _ => Err(format!(
                "HALT_COP_EVENTS {value:#010x}: mode {mode} (raises flow controller \
                 interrupts or is undefined) not modelled"
            )),
        }
    }

    fn refuse(&self, note: String) -> Option<()> {
        self.notes.add(format!("flow controller: {note}"));
        None
    }
}

impl MmioDevice for FlowController {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        let offset = word(offset, size)?;
        let state = self.state.lock().unwrap();
        let value = match offset {
            HALT_COP_EVENTS => state.halt_cop,
            COP_CSR => state.cop_csr,
            RAM_REPAIR => state.ram_repair,
            BPMP_CLUSTER_CONTROL => state.bpmp_cluster_control,
            _ => {
                if let Some(cpu) = HALT_CPU_EVENTS.iter().position(|&r| r == offset) {
                    state.halt_cpu[cpu]
                } else if let Some(cpu) = CPU_CSR.iter().position(|&r| r == offset) {
                    state.cpu_csr[cpu]
                } else {
                    READ_ONLY
                        .iter()
                        .find_map(|&(register, value)| (register == offset).then_some(value))?
                }
            }
        };
        Some(u64::from(value))
    }

    fn write(&self, _core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        let offset = word(offset, size)?;
        let value = value as u32;
        let csr = |old: u32, register: &str| {
            if value & CSR_ENABLE != 0 {
                return Err(format!(
                    "{register} {value:#010x}: power-gating sequencing not modelled"
                ));
            }
            Ok(value & !CSR_FLAGS | old & CSR_FLAGS & !value)
        };
        match offset {
            HALT_COP_EVENTS => match self.cop_halt(value) {
                Ok(halt) => {
                    let mut state = self.state.lock().unwrap();
                    state.halt_cop = value;
                    state.cop = halt;
                    self.halt_changed.store(true, Ordering::Release);
                }
                Err(note) => return self.refuse(note),
            },
            COP_CSR => {
                let mut state = self.state.lock().unwrap();
                match csr(state.cop_csr, "COP_CSR") {
                    Ok(new) => state.cop_csr = new,
                    Err(note) => {
                        drop(state);
                        return self.refuse(note);
                    }
                }
            }
            RAM_REPAIR => {
                if value & !RAM_REPAIR_REQ != 0 {
                    return self.refuse(format!("RAM_REPAIR {value:#010x}"));
                }
                // The repair is done as soon as it is asked for.
                self.state.lock().unwrap().ram_repair = if value & RAM_REPAIR_REQ != 0 {
                    RAM_REPAIR_REQ | RAM_REPAIR_STS | RAM_REPAIR_AT_RESET
                } else {
                    RAM_REPAIR_AT_RESET
                };
            }
            BPMP_CLUSTER_CONTROL => self.state.lock().unwrap().bpmp_cluster_control = value,
            _ => {
                if let Some(cpu) = HALT_CPU_EVENTS.iter().position(|&r| r == offset) {
                    if value >> MODE_SHIFT != MODE_NONE {
                        return self.refuse(format!(
                            "HALT_CPU{cpu}_EVENTS {value:#010x}: halting a CPU core not modelled"
                        ));
                    }
                    self.state.lock().unwrap().halt_cpu[cpu] = value;
                } else {
                    let cpu = CPU_CSR.iter().position(|&r| r == offset)?;
                    let mut state = self.state.lock().unwrap();
                    match csr(state.cpu_csr[cpu], "CPU_CSR") {
                        Ok(new) => state.cpu_csr[cpu] = new,
                        Err(note) => {
                            drop(state);
                            return self.refuse(format!("CPU{cpu}: {note}"));
                        }
                    }
                }
            }
        }
        Some(())
    }

    fn snapshot(&self) -> Option<&dyn Snapshot> {
        Some(self)
    }
}

/// Saves every register and the BPMP's halt (a timed halt as the guest
/// time it ends at).
impl Snapshot for FlowController {
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        encode(&*self.state.lock().unwrap())
    }

    fn restore(&self, state: &[u8]) -> Result<(), SnapshotError> {
        *self.state.lock().unwrap() = decode("flow controller", state)?;
        self.halt_changed.store(true, Ordering::Release);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flow() -> (FlowController, Arc<GuestClock>) {
        let clock = Arc::new(GuestClock::new());
        clock.pause();
        clock.set(Duration::from_secs(1));
        (
            FlowController::new(Arc::clone(&clock), Arc::default()),
            clock,
        )
    }

    #[test]
    fn microsecond_sleep_ends_after_its_count() {
        let (flow, clock) = flow();
        assert_eq!(flow.cop_status(false, false), CopStatus::Running);
        // hekate `bpmp_usleep(200)`: WAITEVENT | USEC | 200.
        flow.write(0, HALT_COP_EVENTS, 4, u64::from(2 << 29 | EVENT_USEC | 200))
            .unwrap();
        assert!(flow.take_halt_change());
        assert!(!flow.take_halt_change());
        let end = Duration::from_secs(1) + Duration::from_micros(200);
        assert_eq!(flow.cop_status(false, false), CopStatus::HaltedUntil(end));
        clock.set(end - Duration::from_nanos(1));
        assert_eq!(flow.cop_status(true, true), CopStatus::HaltedUntil(end));
        clock.set(end);
        assert_eq!(flow.cop_status(false, false), CopStatus::Running);
        // The register keeps what was written.
        assert_eq!(
            flow.read(0, HALT_COP_EVENTS, 4),
            Some(u64::from(2 << 29 | EVENT_USEC | 200))
        );
    }

    #[test]
    fn millisecond_sleep_and_jtag_halt() {
        let (flow, clock) = flow();
        flow.write(0, HALT_COP_EVENTS, 4, u64::from(2 << 29 | EVENT_MSEC | 3))
            .unwrap();
        clock.set(Duration::from_secs(1) + Duration::from_millis(3));
        assert_eq!(flow.cop_status(false, false), CopStatus::Running);
        // hekate `bpmp_halt`: WAITEVENT | JTAG, for good.
        flow.write(0, HALT_COP_EVENTS, 4, u64::from(2 << 29 | EVENT_JTAG))
            .unwrap();
        clock.set(Duration::from_secs(100));
        assert_eq!(flow.cop_status(true, true), CopStatus::Halted);
        // Another master writing mode 0 lets it go.
        flow.write(1, HALT_COP_EVENTS, 4, 0).unwrap();
        assert_eq!(flow.cop_status(false, false), CopStatus::Running);
    }

    #[test]
    fn stop_until_interrupt() {
        let (flow, _clock) = flow();
        flow.write(0, HALT_COP_EVENTS, 4, u64::from(4 << 29 | EVENT_LIC_IRQ))
            .unwrap();
        assert_eq!(flow.cop_status(false, true), CopStatus::Halted);
        assert_eq!(flow.cop_status(true, false), CopStatus::Running);
        // And stays running once woken.
        assert_eq!(flow.cop_status(false, false), CopStatus::Running);
    }

    #[test]
    fn refuses_what_it_does_not_model() {
        let (flow, _clock) = flow();
        for value in [
            1 << 29,                  // RUN_AND_INT
            3 << 29 | EVENT_USEC | 1, // WAITEVENT_AND_INT
            6 << 29,                  // STOP_UNTIL_EVENT_AND_IRQ
            2 << 29 | EVENT_GIC_IRQ,  // GIC event for the BPMP
            2 << 29 | EVENT_USEC | EVENT_MSEC | 1,
            2 << 29 | 1 << 20, // unknown event bit
        ] {
            assert_eq!(flow.write(0, HALT_COP_EVENTS, 4, u64::from(value)), None);
        }
        assert_eq!(flow.read(0, HALT_COP_EVENTS, 4), Some(0));
        assert_eq!(flow.cop_status(false, false), CopStatus::Running);
        assert_eq!(flow.write(0, HALT_CPU_EVENTS[0], 4, 2 << 29), None);
        assert_eq!(flow.write(0, CPU_CSR[1], 4, 1), None);
        assert_eq!(flow.read(0, 0x30, 4), Some(0));
        assert_eq!(flow.read(0, 0x2C, 4), Some(0x0400_4000));
        assert_eq!(flow.write(0, 0x2C, 4, 0), None, "read only");
        assert_eq!(flow.read(0, 0x58, 4), None);
        assert_eq!(flow.read(0, HALT_COP_EVENTS, 2), None);
    }

    #[test]
    fn ram_repair_completes_and_cluster_control_stores() {
        let (flow, _clock) = flow();
        assert_eq!(flow.read(0, RAM_REPAIR, 4), Some(4));
        flow.write(0, RAM_REPAIR, 4, 1).unwrap();
        assert_eq!(flow.read(0, RAM_REPAIR, 4), Some(7));
        // hekate `ccplex_boot_cpu0`: clear ACTIVE_SLOW by read-modify-write.
        let control = flow.read(0, BPMP_CLUSTER_CONTROL, 4).unwrap();
        flow.write(0, BPMP_CLUSTER_CONTROL, 4, control & !1)
            .unwrap();
        assert_eq!(flow.read(0, BPMP_CLUSTER_CONTROL, 4), Some(0));
        // CSR flags are write-one-to-clear; other bits store.
        flow.write(0, CPU_CSR[0], 4, 1 << 12).unwrap();
        assert_eq!(flow.read(0, CPU_CSR[0], 4), Some(1 << 12));
    }

    #[test]
    fn snapshot_keeps_a_timed_halt() {
        let (flow, clock) = flow();
        flow.write(0, HALT_COP_EVENTS, 4, u64::from(2 << 29 | EVENT_USEC | 10))
            .unwrap();
        let saved = flow.save().unwrap();
        let other = FlowController::new(Arc::clone(&clock), Arc::default());
        other.restore(&saved).unwrap();
        assert!(other.take_halt_change());
        let end = Duration::from_secs(1) + Duration::from_micros(10);
        assert_eq!(other.cop_status(false, false), CopStatus::HaltedUntil(end));
    }
}
