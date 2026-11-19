//! The BPMP in a machine powered on from the boot ROM: its view of memory,
//! its run loop, and the release of the CPU cluster.
//!
//! # The BPMP's address space
//!
//! The BPMP ([`Arm7`]) has a 32-bit address space and no MMU. Its accesses
//! go, by address ([`BpmpView`]):
//!
//! | range | what | here |
//! |---|---|---|
//! | `0x0000_0000`..`0x0000_0020` | the architected vectors | never taken: exceptions go where `EVP_COP_*_VECTOR` say ([`crate::soc::evp`]); an exception whose register is unwritten stops the BPMP with that said |
//! | `0x0010_0000`..`0x0012_0000` | IROM, the boot ROM | refused: no dump is available and the ROM is high-level emulated ([`crate::bootrom`]); it has also locked itself out before the bootloader runs (Fusée Gelée write-up) |
//! | `0x4000_0000`..`0x4004_0000` | IRAM ([`crate::soc::iram`]) | read and written directly; bytes the ROM's hand-off does not establish refuse reads until written |
//! | `0x5004_0000`..`0x5004_1000` | the BPMP's cache controller (hekate `BPMP_CACHE_BASE`) | refused: not modelled, and package1ldr does not use it |
//! | `0x5004_1000`..`0x5004_3000` | the CPU cluster's GIC | refused: the BPMP takes its interrupts from the legacy interrupt controller ([`crate::soc::ictlr`]) \[inference: the GIC is the CPU complex's private peripheral, and Linux routes the BPMP's interrupts through the LIC\] |
//! | DRAM (`0x8000_0000` up to 4 GiB) | [`GuestRam`] | read and written directly; the memory controller's carveouts are not applied to the BPMP (its client, AVPC, is not modelled) |
//! | anything else | [`Bus`] with core index [`BPMP_CORE`] | as the device says; unmapped or unmodelled accesses stop the BPMP |
//!
//! Devices that look at the core index see [`BPMP_CORE`], distinct from
//! the four CPU-cluster cores: the secure boot registers treat it as a
//! secure master ([`crate::soc::sb`]), the GIC is never reached with it.
//!
//! # Running
//!
//! [`run`] steps the core in slices of [`SLICE`] instructions. Between
//! slices it looks at what the core cannot see from inside an instruction:
//!
//! * the flow controller's halt of the BPMP ([`FlowController::cop_status`]),
//!   which the BPMP's own write to `HALT_COP_EVENTS` ends a slice for
//!   ([`BpmpBus::should_pause`]); a halted BPMP waits in guest time;
//! * its IRQ and FIQ lines from the legacy interrupt controller;
//! * its own reset (`RST_DEVICES_L` bit 1): held, it does not run; released,
//!   it starts again at `EVP_COP_RESET_VECTOR` in ARM state
//!   ([`crate::soc::BootSignals`]), the way the TSEC's firmware hands the
//!   BPMP to the NX bootloader \[inference, `pkg1-research` §5.4\];
//! * the release of CPU-cluster core 0 by the clock and reset controller
//!   ([`crate::soc::car::Car::on_cpu0_released`]). Core 0 runs only if its
//!   power partitions are on (PMC `PWRGATE_STATUS` bits 0 CRAIL, 14 CE0,
//!   15 C0NC, Linux `tegra210` power-gate table) and the secure boot
//!   registers give an AArch64 reset vector; the vector is handed to the
//!   release callback, which in the machine is the high-level emulated
//!   secure monitor's cold boot on a thread of core 0's own. Anything else
//!   (an unpowered core, a start in AArch32) is refused.
//!
//! The BPMP goes on after the release: the NX bootloader talks to the
//! monitor through IRAM and halts itself through the flow controller.
//!
//! One instruction takes no guest time of its own: the BPMP runs as fast as
//! the host interprets it, and what it reads of time (`TIMERUS`) is guest
//! time. Snapshots: the core stands still while guest time is paused
//! (checked under the core's lock before each slice), so the [`Snapshot`]
//! of [`Bpmp`] and of every device is consistent once the clock stands.

use std::{
    collections::VecDeque,
    ops::Range,
    panic::{self, AssertUnwindSafe},
    sync::{
        Arc, Mutex, OnceLock, Weak,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Duration,
};

use serde::{Deserialize, Serialize};

use super::{CoreReport, CoreStop, Machine, Session, Start};
use crate::{
    boot::{CORE_COUNT, CoreEntry},
    bootrom::BootRomHandoff,
    bpmp::{Arm7, BpmpBus, BusError, Exception},
    bus::{Access, Bus},
    clock::GuestClock,
    image::KernelImage,
    keys::{Key, KeySet},
    memory::GuestRam,
    secmon::{ColdBootContext, Released},
    snapshot::{Snapshot, SnapshotError, decode, encode},
    soc::{
        BootBlocks, BootSignals, Notes, Soc,
        flow::CopStatus,
        iram::{self, Iram},
        pmc,
        se::Se,
        tsec::TsecSoc,
    },
};

/// The core index the BPMP's bus accesses carry: after the CPU cluster's.
pub const BPMP_CORE: usize = CORE_COUNT;

/// Instructions between two looks at the world outside the core.
pub const SLICE: u64 = 4096;

/// How long a halted or held BPMP waits before it looks again (for an
/// interrupt, a register write by another master, a stop).
const IDLE_POLL: Duration = Duration::from_millis(1);

const IROM: Range<u32> = 0x0010_0000..0x0012_0000;
const VECTORS: Range<u32> = 0..0x20;
const BPMP_CACHE: Range<u32> = 0x5004_0000..0x5004_1000;
const GIC: Range<u32> = 0x5004_1000..0x5004_3000;
const IRAM: Range<u32> = iram::BASE as u32..(iram::BASE + iram::SIZE) as u32;

/// `PWRGATE_STATUS` bits of what core 0 needs powered: CRAIL (0), CE0
/// (14), C0NC (15).
const CPU0_PARTITIONS: u32 = 1 << 0 | 1 << 14 | 1 << 15;

/// What a machine powered on starts from: the boot ROM's hand-off and the
/// IRAM it wrote ([`crate::bootrom::cold_boot`]).
pub struct PowerOn {
    pub handoff: BootRomHandoff,
    /// [`crate::bootrom::IRAM_SIZE`] bytes.
    pub iram: Vec<u8>,
}

/// The BPMP's state: the core, and which IRAM bytes hold nothing
/// established yet.
pub struct Bpmp {
    core: Mutex<Arm7>,
    unestablished: Unestablished,
    /// What [`Bpmp::set_trace`] asked to keep. A debugging aid, not state:
    /// snapshots leave it out.
    trace: Mutex<Trace>,
}

/// The last instructions and device accesses of the BPMP.
#[derive(Default)]
struct Trace {
    capacity: usize,
    instructions: VecDeque<(u32, bool)>,
    accesses: VecDeque<String>,
}

impl Trace {
    fn push<T>(capacity: usize, ring: &mut VecDeque<T>, item: T) {
        if ring.len() == capacity {
            ring.pop_front();
        }
        ring.push_back(item);
    }
}

/// IRAM bytes the boot ROM's hand-off leaves unestablished, until the BPMP
/// writes them.
struct Unestablished {
    ranges: Vec<(Range<u32>, &'static str)>,
    /// A bit per IRAM byte, set while it is unestablished.
    bits: Mutex<Vec<u64>>,
}

impl Unestablished {
    fn new(ranges: Vec<(Range<u32>, &'static str)>) -> Self {
        let mut bits = vec![0u64; (iram::SIZE / 64) as usize];
        for (range, _) in &ranges {
            for address in range.clone().filter(|a| IRAM.contains(a)) {
                let byte = (address - IRAM.start) as usize;
                bits[byte / 64] |= 1 << (byte % 64);
            }
        }
        Self {
            ranges,
            bits: Mutex::new(bits),
        }
    }

    /// The note of the first unestablished byte of `size` at IRAM offset
    /// `offset`, if there is one.
    fn check(&self, offset: u32, size: u8) -> Option<&'static str> {
        let bits = self.bits.lock().unwrap();
        let byte = (offset..offset + u32::from(size))
            .find(|&b| bits[b as usize / 64] & 1 << (b % 64) != 0)?;
        let address = IRAM.start + byte;
        Some(
            self.ranges
                .iter()
                .find(|(range, _)| range.contains(&address))
                .map_or("unestablished", |(_, note)| note),
        )
    }

    fn written(&self, offset: u32, size: u8) {
        let mut bits = self.bits.lock().unwrap();
        for byte in offset..offset + u32::from(size) {
            bits[byte as usize / 64] &= !(1 << (byte % 64));
        }
    }
}

#[derive(Serialize, Deserialize)]
struct SavedBpmp {
    core: Arm7,
    unestablished: Vec<u64>,
}

impl Bpmp {
    /// The BPMP as the boot ROM hands it over: `iram_image` (what the ROM
    /// wrote, [`crate::bootrom::IRAM_SIZE`] bytes) loaded into `iram`, the
    /// core at `handoff.entry` in ARM state with `handoff.cpsr`.
    pub fn from_handoff(
        handoff: &BootRomHandoff,
        iram_image: &[u8],
        iram: &iram::Iram,
    ) -> Result<Self, String> {
        iram.load(0, iram_image)
            .ok_or_else(|| format!("{} bytes of IRAM image", iram_image.len()))?;
        Self::new(handoff.entry, handoff.cpsr, handoff.iram_undefined.clone())
    }

    /// A core starting at `entry` in ARM state with `cpsr` (its T bit
    /// clear), with the IRAM bytes in `unestablished` refusing reads until
    /// written.
    pub fn new(
        entry: u32,
        cpsr: u32,
        unestablished: Vec<(Range<u32>, &'static str)>,
    ) -> Result<Self, String> {
        if crate::bpmp::Mode::from_bits(cpsr).is_none() || cpsr & crate::bpmp::PSR_T != 0 {
            return Err(format!("BPMP CPSR {cpsr:#x} at hand-off"));
        }
        let mut core = Arm7::new();
        core.reset(entry);
        core.set_cpsr(cpsr);
        Ok(Self {
            core: Mutex::new(core),
            unestablished: Unestablished::new(unestablished),
            trace: Mutex::default(),
        })
    }

    /// Keeps the last `depth` instructions the core starts and the last
    /// `depth` accesses it makes outside IRAM and DRAM (with their value or
    /// refusal), for [`Bpmp::trace`]; 0 keeps none. Tracing steps the core
    /// one instruction at a time from the run loop, which is slower.
    pub fn set_trace(&self, depth: usize) {
        *self.trace.lock().unwrap() = Trace {
            capacity: depth,
            ..Trace::default()
        };
    }

    /// What [`Bpmp::set_trace`] kept, oldest first: instructions, then
    /// device accesses.
    pub fn trace(&self) -> Vec<String> {
        let trace = self.trace.lock().unwrap();
        let instructions = trace
            .instructions
            .iter()
            .map(|&(pc, thumb)| format!("at {pc:#010x} ({})", if thumb { "Thumb" } else { "ARM" }));
        instructions.chain(trace.accesses.iter().cloned()).collect()
    }

    /// The core's registers, for reports and tests.
    pub fn core(&self) -> Arm7 {
        self.core.lock().unwrap().clone()
    }

    /// Runs at most `budget` instructions, unless something outside the
    /// core must be looked at first; see the module documentation.
    pub fn slice(&self, env: &Env<'_>, budget: u64) -> Slice {
        let mut core = self.core.lock().unwrap();
        if env.clock.is_paused() {
            return Slice::Paused;
        }
        let signals = &env.blocks.signals;
        if signals.cop_restarted.swap(false, Ordering::AcqRel) {
            let requested = signals.restart_at.lock().unwrap().take();
            match requested.or_else(|| env.blocks.evp.cop_vector(0)) {
                Some(entry) => core.reset(entry),
                None => {
                    return Slice::Fault(
                        "BPMP reset released with EVP_COP_RESET_VECTOR unwritten".into(),
                    );
                }
            }
        }
        if signals.cop_held.load(Ordering::Acquire) {
            return Slice::Held;
        }
        if signals.cpu0_released.swap(false, Ordering::AcqRel) {
            return release(env);
        }
        let (irq, fiq) = env.blocks.ictlr.cop_lines();
        match env.blocks.flow.cop_status(irq, fiq) {
            CopStatus::Running => {}
            halted => return Slice::Halted(halted),
        }
        core.set_irq(irq);
        core.set_fiq(fiq);
        let mut trace = self.trace.lock().unwrap();
        let capacity = trace.capacity;
        let mut view = BpmpView {
            env,
            unestablished: &self.unestablished,
            missing_vector: None,
            trace: (capacity > 0).then_some(&mut trace.accesses),
            capacity,
        };
        if capacity == 0 {
            return match core.run(&mut view, budget) {
                Ok(_) => Slice::Ran,
                Err(fault) => Slice::Fault(fault.to_string()),
            };
        }
        let mut instructions = Vec::new();
        let mut outcome = Slice::Ran;
        for _ in 0..budget {
            instructions.push((core.pc(), core.thumb()));
            if let Err(fault) = core.step(&mut view) {
                outcome = Slice::Fault(fault.to_string());
                break;
            }
            if view.should_pause() {
                break;
            }
        }
        for instruction in instructions {
            Trace::push(capacity, &mut trace.instructions, instruction);
        }
        outcome
    }
}

/// The world the BPMP runs in.
pub struct Env<'a> {
    pub blocks: &'a BootBlocks,
    pub ram: &'a GuestRam,
    pub bus: &'a Bus,
    pub clock: &'a GuestClock,
    pub notes: &'a Notes,
}

/// What one [`Bpmp::slice`] came to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Slice {
    /// Instructions ran.
    Ran,
    /// Guest time stands still: nothing ran.
    Paused,
    /// The flow controller halts the BPMP.
    Halted(CopStatus),
    /// The BPMP is held in reset.
    Held,
    /// CPU-cluster core 0 was let out of reset at this AArch64 address.
    Released(u64),
    /// The BPMP cannot go on: what it ran into.
    Fault(String),
}

/// Whether core 0, just let out of reset, can run, and where.
fn release(env: &Env<'_>) -> Slice {
    let status = match env.bus.read(BPMP_CORE, pmc::BASE + pmc::PWRGATE_STATUS, 4) {
        Access::Done(status) => status as u32,
        _ => return Slice::Fault("core 0 released; PWRGATE_STATUS unreadable".into()),
    };
    if status & CPU0_PARTITIONS != CPU0_PARTITIONS {
        return Slice::Fault(format!(
            "core 0 released with its power partitions off (PWRGATE_STATUS \
             {status:#x}): an unpowered core is not modelled"
        ));
    }
    let vector = env.blocks.sb.aa64_reset_vector();
    if vector.aarch64 {
        Slice::Released(vector.address)
    } else {
        Slice::Fault(format!(
            "core 0 released to start in AArch32 (SB_AA64_RESET_LOW {:#x} without \
             AARCH64_MODE_EN): not modelled",
            vector.address
        ))
    }
}

/// Why [`run`] returned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BpmpExit {
    /// `stop` was set.
    Stopped,
    /// The BPMP cannot go on, or the release callback refused.
    Fault(String),
}

/// Runs the BPMP until `stop` is set or it cannot go on; `released` is
/// called with core 0's entry each time core 0 is let out of reset, and an
/// error from it ends the run.
pub fn run(
    bpmp: &Bpmp,
    env: &Env<'_>,
    stop: &AtomicBool,
    released: &mut dyn FnMut(u64) -> Result<(), String>,
) -> BpmpExit {
    while !stop.load(Ordering::SeqCst) {
        match bpmp.slice(env, SLICE) {
            Slice::Ran => {}
            Slice::Paused | Slice::Held | Slice::Halted(CopStatus::Halted) => {
                thread::sleep(IDLE_POLL)
            }
            Slice::Halted(CopStatus::HaltedUntil(deadline)) => {
                thread::sleep(env.clock.until(deadline).min(IDLE_POLL))
            }
            Slice::Halted(CopStatus::Running) => {}
            Slice::Released(entry) => {
                if let Err(error) = released(entry) {
                    return BpmpExit::Fault(error);
                }
            }
            Slice::Fault(fault) => return BpmpExit::Fault(fault),
        }
    }
    BpmpExit::Stopped
}

/// The BPMP's bus: see the module documentation.
struct BpmpView<'a> {
    env: &'a Env<'a>,
    unestablished: &'a Unestablished,
    /// The exception last taken with its vector register unwritten.
    missing_vector: Option<Exception>,
    /// Where device accesses are traced, if they are ([`Bpmp::set_trace`]).
    trace: Option<&'a mut VecDeque<String>>,
    capacity: usize,
}

fn refuse<T>(reason: impl Into<String>) -> Result<T, BusError> {
    Err(BusError::Unmodelled(reason.into()))
}

impl BpmpView<'_> {
    /// Refuses what the BPMP's map has but this machine does not model.
    fn unmodelled(&self, address: u32) -> Result<(), BusError> {
        if VECTORS.contains(&address) {
            return match self.missing_vector {
                Some(exception) => refuse(format!(
                    "{exception:?} exception taken with its EVP_COP vector unwritten \
                     (what the boot ROM leaves there is not established)"
                )),
                None => refuse("low vectors: the BPMP's vectors are the EVP registers"),
            };
        }
        if IROM.contains(&address) {
            return refuse("the boot ROM (IROM) is not available: it is high-level emulated");
        }
        if BPMP_CACHE.contains(&address) {
            return refuse("the BPMP cache controller is not modelled");
        }
        if GIC.contains(&address) {
            return refuse(
                "the CPU cluster's GIC: the BPMP's interrupts come from the legacy \
                 interrupt controller",
            );
        }
        Ok(())
    }

    fn dram(&self, address: u32, size: u8) -> Option<u64> {
        let ram = self.env.ram;
        let address = u64::from(address);
        (address >= ram.base() && address + u64::from(size) <= ram.base() + ram.size() as u64)
            .then_some(address)
    }

    /// Keeps a device access (anything but IRAM and DRAM) in the trace.
    fn traced<T: std::fmt::LowerHex>(
        &mut self,
        address: u32,
        size: u8,
        write: Option<u32>,
        result: Result<T, BusError>,
    ) -> Result<T, BusError> {
        let memory = IRAM.contains(&address) || self.dram(address, size).is_some();
        if let Some(ring) = self.trace.as_deref_mut()
            && !memory
        {
            let outcome = match (&result, write) {
                (Ok(_), Some(value)) => format!("write {value:#x}"),
                (Ok(value), None) => format!("read {value:#x}"),
                (Err(BusError::Abort), _) => "aborted".into(),
                (Err(BusError::Unmodelled(reason)), _) => format!("refused: {reason}"),
            };
            let access = if write.is_some() { "W" } else { "R" };
            Trace::push(
                self.capacity,
                ring,
                format!("{access} {address:#010x}/{size}: {outcome}"),
            );
        }
        result
    }

    fn read_inner(&mut self, address: u32, size: u8) -> Result<u32, BusError> {
        self.unmodelled(address)?;
        if IRAM.contains(&address) {
            let offset = address - IRAM.start;
            if let Some(note) = self.unestablished.check(offset, size) {
                return refuse(format!("IRAM not established by the boot ROM: {note}"));
            }
            return self
                .env
                .blocks
                .iram
                .read(offset, size)
                .map_or_else(|| refuse("access across the end of IRAM"), Ok);
        }
        if let Some(address) = self.dram(address, size) {
            let mut bytes = [0; 4];
            self.env
                .ram
                .read(address, &mut bytes[..usize::from(size)])
                .expect("checked to be DRAM");
            return Ok(u32::from_le_bytes(bytes));
        }
        match self.env.bus.read(BPMP_CORE, u64::from(address), size) {
            Access::Done(value) => Ok(value as u32),
            Access::Unmapped => refuse("nothing is mapped there"),
            Access::Unimplemented(device) => refuse(format!(
                "the {device} does not model this register or width"
            )),
        }
    }

    fn write_inner(&mut self, address: u32, size: u8, value: u32) -> Result<(), BusError> {
        self.unmodelled(address)?;
        if IRAM.contains(&address) {
            let offset = address - IRAM.start;
            self.env
                .blocks
                .iram
                .write(offset, size, value)
                .map_or_else(|| refuse("access across the end of IRAM"), Ok)?;
            self.unestablished.written(offset, size);
            return Ok(());
        }
        if let Some(address) = self.dram(address, size) {
            self.env
                .ram
                .write(address, &value.to_le_bytes()[..usize::from(size)])
                .expect("checked to be DRAM");
            return Ok(());
        }
        match self
            .env
            .bus
            .write(BPMP_CORE, u64::from(address), size, u64::from(value))
        {
            Access::Done(()) => Ok(()),
            Access::Unmapped => refuse("nothing is mapped there"),
            Access::Unimplemented(device) => refuse(format!(
                "the {device} does not model this register or width"
            )),
        }
    }
}

impl BpmpBus for BpmpView<'_> {
    fn read(&mut self, address: u32, size: u8) -> Result<u32, BusError> {
        let result = self.read_inner(address, size);
        self.traced(address, size, None, result)
    }

    fn write(&mut self, address: u32, size: u8, value: u32) -> Result<(), BusError> {
        let result = self.write_inner(address, size, value).map(|()| 0u32);
        self.traced(address, size, Some(value), result).map(|_| ())
    }

    fn exception_vector(&mut self, exception: Exception) -> u32 {
        match self.env.blocks.evp.cop_vector(exception.offset()) {
            Some(vector) => vector,
            None => {
                self.missing_vector = Some(exception);
                exception.offset()
            }
        }
    }

    fn should_pause(&mut self) -> bool {
        let signals = &self.env.blocks.signals;
        self.env.blocks.flow.take_halt_change()
            || signals.cpu0_released.load(Ordering::Acquire)
            || signals.cop_held.load(Ordering::Acquire)
            || signals.cop_restarted.load(Ordering::Acquire)
    }
}

/// The core's registers and which IRAM bytes are still unestablished.
impl Snapshot for Bpmp {
    /// A slice in progress ends; the next one sees guest time paused.
    fn quiesce(&self) -> Result<(), SnapshotError> {
        drop(self.core.lock().unwrap());
        Ok(())
    }

    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        encode(&SavedBpmp {
            core: self.core.lock().unwrap().clone(),
            unestablished: self.unestablished.bits.lock().unwrap().clone(),
        })
    }

    fn restore(&self, state: &[u8]) -> Result<(), SnapshotError> {
        let saved: SavedBpmp = decode("BPMP", state)?;
        if crate::bpmp::Mode::from_bits(saved.core.cpsr()).is_none() {
            return Err(SnapshotError::Corrupt("BPMP: reserved mode bits".into()));
        }
        let mut bits = self.unestablished.bits.lock().unwrap();
        if saved.unestablished.len() != bits.len() {
            return Err(SnapshotError::Corrupt("BPMP: IRAM map size".into()));
        }
        *bits = saved.unestablished;
        *self.core.lock().unwrap() = saved.core;
        Ok(())
    }
}

/// What the TSEC model reaches of the machine ([`TsecSoc`]): the AHB side
/// of the memory controller's IRAM aperture (IRAM and devices), the
/// security engine's keyslots, and the BPMP's restart.
///
/// Devices are reached as [`BPMP_CORE`] reaches them: the TSEC is a secure
/// master, like the BPMP, and the device models tell masters apart only by
/// secure or not (\[assumption\]). The bus is held weakly: the TSEC is one
/// of its devices.
pub struct TsecPort {
    iram: Arc<Iram>,
    bus: Weak<Bus>,
    se: Arc<Se>,
    signals: Arc<BootSignals>,
}

impl TsecPort {
    /// Connects the TSEC of `soc`, a SoC powered on, to the machine around
    /// it on `bus`; `keys` supplies what the TSEC firmware's hardware
    /// secrets would derive. `None` if the SoC was not powered on.
    pub fn connect(soc: &Soc, bus: &Arc<Bus>, keys: &KeySet) -> Option<()> {
        let boot = soc.boot.as_ref()?;
        let port = Arc::new(Self {
            iram: Arc::clone(&boot.iram),
            bus: Arc::downgrade(bus),
            se: Arc::clone(&soc.se),
            signals: Arc::clone(&boot.signals),
        });
        soc.tsec.connect(port, keys);
        Some(())
    }

    /// Word-sized device accesses for `len` bytes at `address`.
    fn words(address: u32, len: usize) -> Option<impl Iterator<Item = u32>> {
        (address.is_multiple_of(4) && len.is_multiple_of(4))
            .then(|| (0..len as u32).step_by(4).map(move |at| address + at))
    }
}

impl TsecSoc for TsecPort {
    fn ahb_read(&self, address: u32, buffer: &mut [u8]) -> Option<()> {
        let end = address.checked_add(u32::try_from(buffer.len()).ok()?)?;
        if IRAM.contains(&address) && end <= IRAM.end {
            return self.iram.copy_out(address - IRAM.start, buffer);
        }
        let bus = self.bus.upgrade()?;
        for (word, chunk) in Self::words(address, buffer.len())?.zip(buffer.as_chunks_mut::<4>().0)
        {
            match bus.read(BPMP_CORE, u64::from(word), 4) {
                Access::Done(value) => chunk.copy_from_slice(&(value as u32).to_le_bytes()),
                _ => return None,
            }
        }
        Some(())
    }

    fn ahb_write(&self, address: u32, data: &[u8]) -> Option<()> {
        let end = address.checked_add(u32::try_from(data.len()).ok()?)?;
        if IRAM.contains(&address) && end <= IRAM.end {
            return self.iram.load(address - IRAM.start, data);
        }
        let bus = self.bus.upgrade()?;
        for (word, chunk) in Self::words(address, data.len())?.zip(data.as_chunks::<4>().0) {
            let value = u32::from_le_bytes(*chunk);
            match bus.write(BPMP_CORE, u64::from(word), 4, u64::from(value)) {
                Access::Done(()) => {}
                _ => return None,
            }
        }
        Some(())
    }

    /// \[assumption\] The slot keeps every permission: what the TSEC
    /// firmware leaves in its access register is not established, and is
    /// refused when read ([`crate::soc::se`] stores what was loaded).
    fn load_se_keyslot(&self, slot: usize, key: &Key) -> Option<()> {
        self.se.load_key(slot, key, 0).ok()
    }

    fn restart_bpmp(&self, entry: u32) -> Option<()> {
        *self.signals.restart_at.lock().unwrap() = Some(entry);
        self.signals.cop_restarted.store(true, Ordering::Release);
        Some(())
    }
}

impl Bpmp {
    /// Applies the boot ROM's hand-off to `soc`, a SoC powered on whose
    /// devices are on `bus`, and gives the BPMP it starts: IRAM as the ROM
    /// wrote it, the ROM's keyslots in the security engine with the
    /// permissions it leaves them, its lock-outs in the secure boot
    /// registers ([`crate::bootrom::ROM_LOCKOUTS`]), and the TSEC connected
    /// to the machine ([`TsecPort`]) with what `keys` hold for it.
    pub fn power_on(
        power_on: &PowerOn,
        soc: &Soc,
        bus: &Arc<Bus>,
        keys: &KeySet,
    ) -> Result<Self, String> {
        let boot = soc.boot.as_ref().ok_or("the SoC was not powered on")?;
        for slot in &power_on.handoff.keyslots {
            let access = &slot.access;
            let flags = [
                (access.readable, crate::soc::se::LOCK_KEY_READ),
                (access.updatable, crate::soc::se::LOCK_KEY_UPDATE),
                (access.usable, crate::soc::se::LOCK_KEY_USE),
            ]
            .into_iter()
            .filter(|&(allowed, _)| !allowed)
            .fold(0, |flags, (_, lock)| flags | lock);
            soc.se
                .load_key(usize::from(slot.slot), &slot.key, flags)
                .map_err(|error| error.to_string())?;
        }
        for lockout in crate::bootrom::ROM_LOCKOUTS {
            let offset = u64::from(lockout.address)
                .checked_sub(crate::soc::sb::BASE)
                .filter(|&offset| offset < crate::soc::sb::SIZE);
            offset
                .and_then(|offset| boot.sb.apply_rom(offset, lockout.mask, lockout.value))
                .ok_or_else(|| format!("boot ROM lock-out at {:#x}", lockout.address))?;
        }
        TsecPort::connect(soc, bus, keys).ok_or("the SoC has no boot blocks")?;
        Self::from_handoff(&power_on.handoff, &power_on.iram, &boot.iram)
    }
}

/// The machine's kernel image: set once, at construction for a machine
/// started at kernel entry, by the secure monitor's cold boot in one
/// powered on (`Session::cold_boot`).
#[derive(Default)]
pub struct Kernel(OnceLock<KernelImage>);

impl Kernel {
    pub fn get(&self) -> Option<&KernelImage> {
        self.0.get()
    }

    /// Sets the kernel; `false` if a different one is set already.
    pub fn set(&self, kernel: KernelImage) -> bool {
        let (address, bytes) = (kernel.load_address(), kernel.bytes().to_vec());
        match self.0.set(kernel) {
            Ok(()) => true,
            Err(_) => self
                .get()
                .is_some_and(|set| set.load_address() == address && set.bytes() == bytes),
        }
    }
}

/// Saves the kernel a powered-on machine's cold boot placed (its load
/// address and bytes), so that a restored machine resolves breakpoints and
/// symbols against it again.
impl Snapshot for Kernel {
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        encode(
            &self
                .get()
                .map(|kernel| (kernel.load_address(), kernel.bytes().to_vec())),
        )
    }

    fn restore(&self, state: &[u8]) -> Result<(), SnapshotError> {
        let saved: Option<(u64, Vec<u8>)> = decode("kernel", state)?;
        if let Some((address, bytes)) = saved {
            let kernel = KernelImage::from_flat(address, bytes)
                .map_err(|error| SnapshotError::Corrupt(format!("kernel: {error}")))?;
            if !self.set(kernel) {
                return Err(SnapshotError::Corrupt("kernel: another one is set".into()));
            }
        }
        Ok(())
    }
}

impl Machine {
    /// Keeps the BPMP's last `depth` instructions and device accesses, and
    /// adds them to the note when it stops ([`Bpmp::set_trace`]). Nothing
    /// in a machine that is not powered on.
    pub fn set_bpmp_trace(&self, depth: usize) {
        if let Some(bpmp) = &self.bpmp {
            bpmp.set_trace(depth);
        }
    }

    /// The BPMP's world, in a machine powered on from the boot ROM.
    fn bpmp_env(&self) -> Option<(&Bpmp, Env<'_>)> {
        let bpmp = self.bpmp.as_ref()?;
        let blocks = self.soc.boot.as_ref()?;
        Some((
            bpmp,
            Env {
                blocks,
                ram: &self.ram,
                bus: &self.bus,
                clock: &self.clock,
                notes: &self.soc.notes,
            },
        ))
    }
}

impl<'env> Session<'env> {
    /// Whether this machine starts from the boot ROM's hand-off: then its
    /// BPMP runs first and CPU-cluster core 0 starts when it is released.
    pub(super) fn powers_on(&self) -> bool {
        self.machine.bpmp.is_some()
    }

    /// Starts the BPMP's thread; it reports as core [`BPMP_CORE`].
    pub(super) fn start_bpmp<'scope>(&'env self, scope: &'scope thread::Scope<'scope, 'env>) {
        self.spawn_reporting(scope, BPMP_CORE, move || self.run_bpmp(scope));
    }

    fn run_bpmp<'scope>(&'env self, scope: &'scope thread::Scope<'scope, 'env>) -> CoreReport {
        let (bpmp, env) = self
            .machine
            .bpmp_env()
            .expect("only a machine powered on runs its BPMP");
        let exit = run(bpmp, &env, &self.stop, &mut |entry| {
            self.start_cold_boot(scope, entry);
            Ok(())
        });
        let stop = match exit {
            BpmpExit::Stopped => CoreStop::Requested,
            BpmpExit::Fault(fault) => {
                let trace = bpmp.trace();
                let trace = if trace.is_empty() {
                    String::new()
                } else {
                    format!(
                        "; last of what it did, oldest first:\n  {}",
                        trace.join("\n  ")
                    )
                };
                env.notes.add(format!("BPMP: {fault}{trace}"));
                CoreStop::Bpmp {
                    pc: bpmp.core().pc(),
                }
            }
        };
        CoreReport {
            core: BPMP_CORE,
            stop,
            state: None,
        }
    }

    /// Starts core 0 at `entry` the way the secure monitor would: its
    /// high-level emulated cold boot or SC7 resume runs on core 0's thread
    /// (concurrently with the BPMP), and core 0 then enters the kernel
    /// where it says.
    fn start_cold_boot<'scope>(&'env self, scope: &'scope thread::Scope<'scope, 'env>, entry: u64) {
        self.spawn_reporting(scope, 0, move || match self.core0_released(entry) {
            Ok(entry) => {
                self.switches[0].lock().unwrap().0 = true;
                self.machine.pause.state.lock().unwrap().threads[0] = true;
                self.run_core(scope, 0, Start::Entry(entry))
            }
            Err(stop) => CoreReport {
                core: 0,
                stop,
                state: None,
            },
        });
    }

    /// Waits until core 0's thread, if it has one, has ended: one powered
    /// off for SC7 ends just after the monitor returns
    /// ([`CoreStop::Suspended`]), and a release could come that soon. A
    /// release while core 0 runs is refused.
    fn core0_off(&self) -> Result<(), CoreStop> {
        const GRACE: Duration = Duration::from_secs(1);
        let deadline = std::time::Instant::now() + GRACE;
        while self.switches[0].lock().unwrap().0 {
            if self.stop.load(Ordering::SeqCst) {
                return Err(CoreStop::Requested);
            }
            if std::time::Instant::now() >= deadline {
                self.machine
                    .soc
                    .notes
                    .add("core 0 released while it runs: not modelled".into());
                return Err(CoreStop::ColdBoot);
            }
            thread::sleep(Duration::from_millis(1));
        }
        Ok(())
    }

    /// The monitor's part of core 0's release ([`SecureMonitor::core0_released`](crate::secmon::SecureMonitor::core0_released)):
    /// its cold boot, or its resume from SC7. The kernel a cold boot places
    /// becomes the machine's ([`Kernel`]): breakpoints, symbols and
    /// supervisor call tracing resolve against it from then on. A second
    /// cold boot must place the same kernel.
    fn core0_released(&self, entry: u64) -> Result<CoreEntry, CoreStop> {
        self.core0_off()?;
        let machine = self.machine;
        let blocks = machine.soc.boot.as_ref().expect("a machine powered on");
        let context = ColdBootContext {
            ram: &machine.ram,
            iram: &blocks.iram,
            bus: &machine.bus,
        };
        let fail = |note: String| {
            machine.soc.notes.add(note);
            CoreStop::ColdBoot
        };
        let released = machine
            .secmon
            .core0_released(entry, context, &self.stop)
            .map_err(|error| fail(error.to_string()))?;
        match released {
            Released::ColdBoot(cold_boot) => {
                if !machine.image.set(cold_boot.kernel) {
                    return Err(fail(
                        "secure monitor cold boot: a kernel other than the one placed before"
                            .into(),
                    ));
                }
                Ok(cold_boot.entry)
            }
            Released::Resume(entry) => Ok(entry),
        }
    }

    /// Runs `body` on a thread of `scope` as a core of this run: counted as
    /// started, its report sent when it ends, a panic reported before it
    /// propagates (as [`Session::start_core`] does).
    fn spawn_reporting<'scope>(
        &'env self,
        scope: &'scope thread::Scope<'scope, 'env>,
        core: usize,
        body: impl FnOnce() -> CoreReport + Send + 'scope,
    ) {
        self.started.fetch_add(1, Ordering::SeqCst);
        scope.spawn(move || {
            let outcome = panic::catch_unwind(AssertUnwindSafe(body));
            let report = outcome.as_ref().copied().unwrap_or(CoreReport {
                core,
                stop: CoreStop::HostPanic,
                state: None,
            });
            // The receiver outlives the scope, so this cannot fail. A core
            // powered off for SC7 is not reported (see `Session::start_core`).
            if matches!(report.stop, CoreStop::Suspended) {
                self.started.fetch_sub(1, Ordering::SeqCst);
            } else {
                let _ = self.reports.send(report);
            }
            if let Err(payload) = outcome {
                panic::resume_unwind(payload);
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex as StdMutex};

    use super::*;
    use crate::soc::{
        BootSignals, car::Car, evp::Evp, flow::FlowController, ictlr::Ictlr, iram::Iram, pmc::Pmc,
        sb::Sb,
    };

    /// The SoC as far as the BPMP needs it, wired as `Soc::new` wires a
    /// machine powered on.
    struct Bench {
        blocks: BootBlocks,
        ram: GuestRam,
        bus: Bus,
        clock: Arc<GuestClock>,
        notes: Arc<Notes>,
    }

    impl Bench {
        fn new() -> Self {
            let clock = Arc::new(GuestClock::new());
            let notes: Arc<Notes> = Arc::default();
            let blocks = BootBlocks {
                iram: Arc::new(Iram::new()),
                flow: Arc::new(FlowController::new(Arc::clone(&clock), Arc::clone(&notes))),
                evp: Arc::new(Evp::new()),
                sb: Arc::new(Sb::new()),
                ictlr: Arc::new(Ictlr::new()),
                tzram: Arc::default(),
                signals: Arc::new(BootSignals::default()),
            };
            let mut bus = Bus::default();
            let signals = Arc::clone(&blocks.signals);
            let car = Car::power_on().on_cpu0_released(Box::new(move || {
                signals.cpu0_released.store(true, Ordering::Release)
            }));
            bus.map(
                "clock and reset",
                crate::soc::car::BASE,
                crate::soc::car::SIZE,
                Box::new(car),
            );
            bus.map("PMC", pmc::BASE, pmc::SIZE, Box::new(Pmc::power_on()));
            bus.map(
                "flow controller",
                crate::soc::flow::BASE,
                crate::soc::flow::SIZE,
                Box::new(Arc::clone(&blocks.flow)),
            );
            bus.map(
                "exception vectors",
                crate::soc::evp::BASE,
                crate::soc::evp::SIZE,
                Box::new(Arc::clone(&blocks.evp)),
            );
            bus.map(
                "secure boot registers",
                crate::soc::sb::BASE,
                crate::soc::sb::SIZE,
                Box::new(Arc::clone(&blocks.sb)),
            );
            Self {
                blocks,
                ram: GuestRam::new(0x8000_0000, 0x10_0000).unwrap(),
                bus,
                clock,
                notes,
            }
        }

        fn env(&self) -> Env<'_> {
            Env {
                blocks: &self.blocks,
                ram: &self.ram,
                bus: &self.bus,
                clock: &self.clock,
                notes: &self.notes,
            }
        }
    }

    /// A little ARM assembler: each instruction, with `ldr rd, =value`
    /// taking its value from a literal pool after the code.
    enum Op {
        LdrLiteral(u32, u32),
        Str(u32, u32, u32),
        Word(u32),
    }

    fn assemble(ops: &[Op]) -> Vec<u8> {
        let pool_start = ops.len() as u32 * 4;
        let mut pool: Vec<u32> = Vec::new();
        let mut words = Vec::new();
        for (index, op) in ops.iter().enumerate() {
            let pc = index as u32 * 4;
            words.push(match *op {
                Op::LdrLiteral(rd, value) => {
                    let slot = pool.iter().position(|&v| v == value).unwrap_or_else(|| {
                        pool.push(value);
                        pool.len() - 1
                    });
                    let literal = pool_start + slot as u32 * 4;
                    // LDR rd, [pc, #literal - (pc + 8)]
                    0xE59F_0000 | rd << 12 | (literal - (pc + 8))
                }
                // STR rd, [rn, #offset]
                Op::Str(rd, rn, offset) => 0xE580_0000 | rn << 16 | rd << 12 | offset,
                Op::Word(word) => word,
            });
        }
        words
            .into_iter()
            .chain(pool)
            .flat_map(u32::to_le_bytes)
            .collect()
    }

    const BASE: u32 = 0x4001_0000;
    const SECMON_ENTRY: u32 = 0x4003_0000;
    const B_SELF: u32 = 0xEAFF_FFFE;

    /// What the NX bootloader does to start the CPU cluster, in the order
    /// of hekate `ccplex_boot_cpu0`, then halt itself (`bpmp_halt`).
    fn release_program() -> Vec<u8> {
        use Op::*;
        assemble(&[
            LdrLiteral(0, 0x6000_C230),      // SB_AA64_RESET_LOW
            LdrLiteral(1, SECMON_ENTRY | 1), // entry, AARCH64_MODE_EN
            Str(1, 0, 0),
            LdrLiteral(1, 0),
            Str(1, 0, 4),               // SB_AA64_RESET_HIGH
            LdrLiteral(0, 0x7000_E430), // PMC_PWRGATE_TOGGLE
            LdrLiteral(1, 0x100),       // START | CRAIL
            Str(1, 0, 0),
            LdrLiteral(1, 0x10F), // START | C0NC
            Str(1, 0, 0),
            LdrLiteral(1, 0x10E), // START | CE0
            Str(1, 0, 0),
            LdrLiteral(0, 0x6000_6000), // CAR
            LdrLiteral(1, 1),
            Str(1, 0, 0x360), // CLK_OUT_ENB_V: CPUG
            LdrLiteral(1, 1 << 29),
            Str(1, 0, 0x454), // RST_CPUG_CMPLX_CLR: NONCPU
            LdrLiteral(1, 1 << 30 | 1 << 24 | 1 << 16 | 1),
            Str(1, 0, 0x454),           // core 0
            LdrLiteral(0, 0x6000_7000), // flow controller
            LdrLiteral(1, 0x5000_0000), // WAITEVENT | JTAG
            Str(1, 0, 4),               // HALT_COP_EVENTS
            Word(B_SELF),
        ])
    }

    fn load(bench: &Bench, program: &[u8]) -> Bpmp {
        bench.blocks.iram.load(BASE - IRAM.start, program).unwrap();
        Bpmp::new(BASE, 0xD3, Vec::new()).unwrap()
    }

    #[test]
    fn a_bootloader_releases_core_0_at_the_secure_monitor_entry() {
        let bench = Bench::new();
        let bpmp = load(&bench, &release_program());
        let entries = StdMutex::new(Vec::new());
        let stop = AtomicBool::new(false);
        thread::scope(|scope| {
            let runner = scope.spawn(|| {
                run(&bpmp, &bench.env(), &stop, &mut |entry| {
                    entries.lock().unwrap().push(entry);
                    Ok(())
                })
            });
            // The BPMP goes on after the release until it halts itself.
            let halted = || {
                crate::bus::MmioDevice::read(&*bench.blocks.flow, BPMP_CORE, 4, 4)
                    == Some(0x5000_0000)
                    && bpmp.core().pc() == BASE + 22 * 4
            };
            let deadline = std::time::Instant::now() + Duration::from_secs(10);
            while !halted() && std::time::Instant::now() < deadline {
                thread::sleep(Duration::from_millis(1));
            }
            stop.store(true, Ordering::SeqCst);
            assert_eq!(runner.join().unwrap(), BpmpExit::Stopped);
        });
        assert_eq!(*entries.lock().unwrap(), [u64::from(SECMON_ENTRY)]);
        assert_eq!(bpmp.core().pc(), BASE + 22 * 4, "halted before `b .`");
    }

    #[test]
    fn release_without_power_is_refused() {
        let bench = Bench::new();
        // The program without its PMC toggles.
        let mut program = release_program();
        let skip = 5 * 4..12 * 4;
        let nop = 0xE1A0_0000u32.to_le_bytes(); // MOV r0, r0
        for at in skip.step_by(4) {
            program[at..at + 4].copy_from_slice(&nop);
        }
        let bpmp = load(&bench, &program);
        let env = bench.env();
        let exit = run(&bpmp, &env, &AtomicBool::new(false), &mut |_| {
            panic!("core 0 must not start")
        });
        let BpmpExit::Fault(fault) = exit else {
            panic!("{exit:?}")
        };
        assert!(fault.contains("power partitions off"), "{fault}");
    }

    #[test]
    fn unestablished_iram_refuses_reads_until_written() {
        let bench = Bench::new();
        use Op::*;
        let program = assemble(&[
            LdrLiteral(0, 0x4000_0040),
            LdrLiteral(1, 7),
            Str(1, 0, 0),      // write the word first: fine
            Word(0xE590_2000), // LDR r2, [r0]
            Word(0xE590_3004), // LDR r3, [r0, #4]: unestablished
            Word(B_SELF),
        ]);
        bench.blocks.iram.load(BASE - IRAM.start, &program).unwrap();
        let bpmp = Bpmp::new(
            BASE,
            0xD3,
            vec![(0x4000_0000..0x4001_0000, "the ROM's work area")],
        )
        .unwrap();
        let Slice::Fault(fault) = bpmp.slice(&bench.env(), 100) else {
            panic!("the read must be refused")
        };
        assert!(fault.contains("the ROM's work area"), "{fault}");
        assert!(fault.contains("0x40000044"), "{fault}");
        assert_eq!(bpmp.core().reg(2), 7);
        assert_eq!(bpmp.core().pc(), BASE + 4 * 4);
    }

    #[test]
    fn exceptions_go_to_the_evp_vectors_or_stop_the_bpmp() {
        let bench = Bench::new();
        let swi = 0xEF00_0000u32; // SWI #0
        let program = assemble(&[Op::Word(swi), Op::Word(B_SELF)]);
        bench.blocks.iram.load(BASE - IRAM.start, &program).unwrap();
        let bpmp = Bpmp::new(BASE, 0xD3, Vec::new()).unwrap();
        let Slice::Fault(fault) = bpmp.slice(&bench.env(), 10) else {
            panic!("an unwritten vector must stop the BPMP")
        };
        assert!(fault.contains("SoftwareInterrupt"), "{fault}");
        // With the vector written, the SWI lands there.
        let bench = Bench::new();
        bench.blocks.iram.load(BASE - IRAM.start, &program).unwrap();
        use crate::bus::MmioDevice;
        bench
            .blocks
            .evp
            .write(BPMP_CORE, 0x208, 4, u64::from(BASE + 4))
            .unwrap();
        let bpmp = Bpmp::new(BASE, 0xD3, Vec::new()).unwrap();
        assert_eq!(bpmp.slice(&bench.env(), 3), Slice::Ran);
        assert_eq!(bpmp.core().pc(), BASE + 4);
        assert_eq!(bpmp.core().mode(), crate::bpmp::Mode::Supervisor);
    }

    #[test]
    fn the_cpu_clusters_gic_and_the_boot_rom_are_refused_dram_is_reached() {
        let bench = Bench::new();
        bench
            .ram
            .write(0x8000_0010, &0x1234_5678u32.to_le_bytes())
            .unwrap();
        let bpmp = Bpmp::new(BASE, 0xD3, Vec::new()).unwrap();
        let unestablished = &bpmp.unestablished;
        let env = bench.env();
        let mut view = BpmpView {
            env: &env,
            unestablished,
            missing_vector: None,
            trace: None,
            capacity: 0,
        };
        assert_eq!(view.read(0x8000_0010, 4), Ok(0x1234_5678));
        view.write(0x8000_0014, 2, 0xBEEF).unwrap();
        assert_eq!(view.read(0x8000_0014, 2), Ok(0xBEEF));
        for address in [0x5004_1000, 0x0010_0000, 0x5004_0000, 0x0000_0008] {
            assert!(
                matches!(view.read(address, 4), Err(BusError::Unmodelled(_))),
                "{address:#x}"
            );
        }
        // MMIO through the bus, as the BPMP.
        use crate::bus::MmioDevice;
        view.write(0x6000_C230, 4, 0x4003_0001).unwrap();
        assert_eq!(bench.blocks.sb.read(0, 0x30, 4), Some(0x4003_0001));
        assert!(matches!(
            view.read(0x6000_C204, 4),
            Err(BusError::Unmodelled(_))
        ));
    }

    #[test]
    fn a_timed_halt_waits_in_guest_time() {
        let bench = Bench::new();
        use Op::*;
        let program = assemble(&[
            LdrLiteral(0, 0x6000_7000),
            LdrLiteral(1, 2 << 29 | 1 << 25 | 50), // WAITEVENT | USEC | 50
            Str(1, 0, 4),
            Word(B_SELF),
        ]);
        bench.blocks.iram.load(BASE - IRAM.start, &program).unwrap();
        let bpmp = Bpmp::new(BASE, 0xD3, Vec::new()).unwrap();
        bench.clock.pause();
        let env = bench.env();
        assert_eq!(bpmp.slice(&env, 100), Slice::Paused);
        bench.clock.resume();
        assert_eq!(bpmp.slice(&env, 100), Slice::Ran);
        assert!(matches!(
            bpmp.slice(&env, 100),
            Slice::Halted(CopStatus::HaltedUntil(_)) | Slice::Ran
        ));
        bench.clock.sleep(Duration::from_micros(60));
        assert_eq!(bpmp.slice(&env, 100), Slice::Ran);
        assert_eq!(bpmp.core().pc(), BASE + 3 * 4);
    }

    #[test]
    fn snapshot_round_trip() {
        let bench = Bench::new();
        let program = assemble(&[Op::Word(0xE3A0_0005), Op::Word(B_SELF)]); // MOV r0, #5
        bench.blocks.iram.load(BASE - IRAM.start, &program).unwrap();
        let bpmp = Bpmp::new(BASE, 0xD3, vec![(0x4000_0000..0x4000_0010, "x")]).unwrap();
        assert_eq!(bpmp.slice(&bench.env(), 1), Slice::Ran);
        bpmp.quiesce().unwrap();
        let saved = bpmp.save().unwrap();
        let other = Bpmp::new(0, 0xD3, Vec::new()).unwrap();
        other.restore(&saved).unwrap();
        assert_eq!(other.core(), bpmp.core());
        assert!(other.unestablished.check(0, 4).is_some());
    }
}
