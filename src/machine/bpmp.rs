//! The BPMP in a machine powered on from the boot ROM: its view of memory,
//! its run loop, the release of the CPU cluster, and SC7.
//!
//! # The BPMP's address space
//!
//! The BPMP ([`Arm7`]) has a 32-bit address space and no MMU. Its accesses
//! go, by address ([`BpmpView`]):
//!
//! | range | what | here |
//! |---|---|---|
//! | `0x0000_0000`..`0x0000_0020` | the architected vectors | never taken: exceptions go where `EVP_COP_*_VECTOR` say ([`crate::soc::evp`]); an exception whose register is unwritten stops the BPMP with that said |
//! | `0x0010_0000`..`0x0012_0000` | IROM, the boot ROM | with a dump of the ROM ([`PowerOn::Rom`]): its first 96 KiB read through the patch unit, until the ROM's lock-out hides them ([`crate::soc::irom`]); writes and the rest refused. Without a dump refused: the ROM is high-level emulated ([`crate::bootrom`]) |
//! | `0x4000_0000`..`0x4004_0000` | IRAM ([`crate::soc::iram`]) | read and written directly; bytes no master has written since the ROM's hand-off (or since SC7) refuse reads ([`Iram::bpmp_read`]) |
//! | `0x5004_0000`..`0x5004_1000` | the BPMP's cache controller (hekate `BPMP_CACHE_BASE`) | on the [`Bus`]: its configuration register alone, as storage; no cache is emulated |
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
//! [`run`] steps the core in slices of [`SLICE`] instructions, with its
//! code cache on IRAM ([`Assist`]). Between slices it looks at what the
//! core cannot see from inside an instruction:
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
//!   secure monitor's cold boot or SC7 resume on a thread of core 0's own.
//!   Anything else (an unpowered core, a start in AArch32) is refused;
//! * the SoC entering SC7 (PMC `DPD_ENABLE` bit 0), and while it is in SC7,
//!   the wake (see below).
//!
//! The BPMP goes on after the release: the NX bootloader talks to the
//! monitor through IRAM and halts itself through the flow controller.
//!
//! One instruction takes no guest time of its own: the BPMP runs as fast as
//! the host interprets it, and what it reads of time (`TIMERUS`) is guest
//! time. Snapshots: the core stands still while guest time is paused
//! (checked under the core's lock before each slice), so the [`Snapshot`]
//! of [`Bpmp`] and of every device is consistent once the clock stands.
//! SC7 entry and exit happen inside one slice, so a snapshot sees the SoC
//! before, in, or after SC7, never between. Refused: a snapshot while core
//! 0 is being started, when the monitor's high-level emulated cold boot or
//! resume runs on core 0's thread and its state is in that thread.
//!
//! # SC7
//!
//! On the console (`warmboot-research` §2, §5, §6): the secure monitor's
//! SC7-entry stub on the BPMP ends with `DPD_ENABLE` bit 0, which powers
//! the SoC down except the always-on domain (PMC, RTC), the PMIC and DRAM
//! in self-refresh; a wake event at the PMC powers it up again, and the
//! boot ROM's warm path loads the warmboot firmware and starts the BPMP
//! there. Here, when a slice sees `DPD_ENABLE` bit 0 (`power_down`):
//!
//! * the BPMP is off: it does not run until the wake;
//! * reset to their power-on state, because the warm path depends on them
//!   (\[decision\]: only these; the list follows `warmboot-research` §13):
//!   IRAM (zeroed and unestablished: it loses power), TZRAM (zeroed: it
//!   loses power on Erista, and the warmboot firmware restores it from
//!   DRAM), the flow controller, the exception vectors, the legacy
//!   interrupt controller and the secure boot registers (the warmboot
//!   writes the AArch64 reset vector again), the CPU cluster's resets in
//!   the clock and reset controller (`RST_CPUG_CMPLX` asserted again, so
//!   the warmboot's release is a release), MSELECT (back in reset: it
//!   loses its settings), `MC_SECURITY_CFG3` (0 after reset: how the
//!   monitor learns the SoC lost power), and the CPU cluster's interrupt
//!   controller ([`PowerDomain::power_lost`]; the monitor's resume sets
//!   its Secure side up again and the kernel restores its own,
//!   [`crate::secmon::setup`]);
//! * in a machine that runs the ROM itself, also what only the ROM sets
//!   up again: the fuse block's driver and key hiding, the patch unit,
//!   USB1 and the security engine ([`crate::soc::Soc::enter_sc7`]);
//! * kept, though lost on the console \[assumption\]: every other block,
//!   among them the rest of the clock and reset controller, the memory and
//!   EMC controllers (the boot ROM's restore from PMC scratch writes back
//!   what they hold: `bootrom::warm_boot` leaves it out, the ROM itself
//!   does it) and, without a dump of the ROM, the security engine (its
//!   context comes back from the monitor's save on the warm path). So a
//!   ROM run after SC7 starts from the clocks the kernel left, not from
//!   the SoC's reset values, and writes its own over them;
//! * kept, as on the console: DRAM, the PMC (bar what the wake changes),
//!   the PMIC and its clock.
//!
//! Then the SoC waits in SC7, a state of [`Bpmp`] that snapshots save,
//! until the machine has a wake event ([`PowerDomain::wake_pending`]: the
//! PMIC's interrupt at the PMC, the power key or an alarm). At the wake
//! (`wake`) the PMC leaves deep power down and its secure-scratch locks
//! reset ([`pmc::SECURE_LOCKS`]). Without a dump of the ROM, the boot
//! ROM's warm path runs high-level emulated ([`PowerDomain::warm_boot`]:
//! `bootrom::warm_boot`, its register writes and the security engine's
//! context), its IRAM image is loaded with what it leaves undefined
//! unestablished, and the BPMP starts at the warmboot's entry. With one,
//! the BPMP comes out of reset at the ROM's vector as at power-on (IRAM
//! zero, the fuse block, the patch unit and USB1 as at power-on,
//! [`crate::soc::Soc::enter_sc7`]) with `PMC_RST_STATUS` saying LP0, and
//! the ROM takes its warm path by `PMC_SCRATCH0` bit 0, which the monitor
//! set before the sleep.

use std::{
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
    boot::{CORE_COUNT, CoreEntry, DRAM_BASE},
    bootrom::{self, BootRomHandoff, Fuses, RegisterBits},
    bpmp::{Arm7, Assist, BpmpBus, BusError, Exception},
    bus::{Access, Bus},
    clock::GuestClock,
    image::KernelImage,
    keys::{Key, KeySet},
    memory::GuestRam,
    secmon::{ColdBootContext, Released},
    snapshot::{Snapshot, SnapshotError, decode, encode},
    soc::{
        BootBlocks, BootSignals, Notes, Soc, car,
        flow::CopStatus,
        iram::{self, Iram, Refusal},
        irom::{self, RomImage},
        mc, pmc, sb,
        se::Se,
        tsec::TsecSoc,
        tzram,
    },
};

/// The core index the BPMP's bus accesses carry: after the CPU cluster's.
pub const BPMP_CORE: usize = CORE_COUNT;

/// Instructions between two looks at the world outside the core.
pub const SLICE: u64 = 4096;

/// How long a halted or held BPMP waits before it looks again (for an
/// interrupt, a register write by another master, a stop).
const IDLE_POLL: Duration = Duration::from_millis(1);

/// How often a SoC in SC7 looks for a wake event.
const WAKE_POLL: Duration = Duration::from_millis(5);

const IROM: Range<u32> = 0x0010_0000..0x0012_0000;
/// The part of [`IROM`] a dump of the ROM fills.
const IROM_IMAGE: Range<u32> = irom::BASE..irom::BASE + irom::SIZE as u32;
const VECTORS: Range<u32> = 0..0x20;
const GIC: Range<u32> = 0x5004_1000..0x5004_3000;
const IRAM: Range<u32> = iram::BASE as u32..(iram::BASE + iram::SIZE) as u32;

/// `PWRGATE_STATUS` bits of what core 0 needs powered: CRAIL (0), CE0
/// (14), C0NC (15).
const CPU0_PARTITIONS: u32 = 1 << 0 | 1 << 14 | 1 << 15;

/// `SCTLR_EL1` as the 23.5.0 secure monitor leaves it when it enters the
/// kernel (SecmonColdBoot's reading of the monitor's EL1 set-up): MMU and
/// caches off, the RES1 bits and the EL0 controls it sets. A machine
/// powered on enters the kernel with it instead of
/// [`crate::boot::ENTRY_SCTLR_EL1`].
pub(super) const MONITOR_SCTLR_EL1: u64 = 0x00C5_0838;

/// Why IRAM holds nothing established after SC7.
const IRAM_LOST: &str = "IRAM lost power in SC7";

/// The CPSR of a core at reset: Supervisor mode, ARM state, IRQ and FIQ
/// masked (ARM ARM A2, "Reset").
const RESET_CPSR: u32 = 0xD3;

/// A [`PowerDomain`] with no wake source, for a harness that runs the BPMP
/// without a machine: a SoC that enters SC7 stays there.
pub struct NeverWakes;

impl PowerDomain for NeverWakes {
    fn wake_pending(&self) -> bool {
        false
    }

    /// No block outside the BPMP's is here.
    fn power_lost(&self) {}

    fn warm_boot(&self) -> Result<WarmStart, String> {
        Err("no wake source".into())
    }
}

/// What the SoC's power-up after SC7 resets outside the BPMP's own blocks,
/// as register writes of the reset state: `RST_CPUG_CMPLX_SET` asserting
/// the CPU cluster's resets as after SC7 (`car::CPUG_RESETS_AFTER_SC7`;
/// the bits the warmboot firmware does not release are already clear,
/// released by the NX bootloader at the cold boot), `RST_DEV_V_SET`
/// putting MSELECT back into reset (it loses its settings in SC7,
/// [`crate::soc::mselect`]; the warmboot firmware releases it again), and
/// `MC_SECURITY_CFG3` at 0 (`warmboot-research` §3: "0 after reset").
const SC7_RESETS: [(u64, u32); 3] = [
    (car::BASE + 0x450, car::CPUG_RESETS_AFTER_SC7),
    (car::BASE + 0x430, 1 << crate::soc::mselect::RESET.1),
    (mc::BASE + 0x9BC, 0),
];

/// What a machine powered on starts from.
pub enum PowerOn {
    /// The high-level emulated boot ROM's hand-off and the IRAM it wrote
    /// ([`crate::bootrom::cold_boot`], [`crate::bootrom::IRAM_SIZE`]
    /// bytes), and the fuses the ROM read, which its warm path after SC7
    /// reads again.
    Emulated {
        handoff: BootRomHandoff,
        iram: Vec<u8>,
        fuses: Fuses,
    },
    /// A dump of the boot ROM, which the BPMP runs from its reset vector,
    /// and the console's fuses, which the ROM reads from the fuse block.
    Rom { rom: RomImage, fuses: Fuses },
}

impl PowerOn {
    pub fn fuses(&self) -> &Fuses {
        match self {
            Self::Emulated { fuses, .. } | Self::Rom { fuses, .. } => fuses,
        }
    }

    /// The state the SoC is built in for this start.
    pub fn soc_start(&self) -> crate::soc::Start<'_> {
        match self {
            Self::Emulated { .. } => crate::soc::Start::AfterRom,
            Self::Rom { rom, fuses } => crate::soc::Start::Reset {
                rom: rom.clone(),
                fuses,
            },
        }
    }
}

/// The BPMP: the core, what runs with it, whether the SoC is in SC7.
pub struct Bpmp {
    running: Mutex<Running>,
    /// The fuses the boot ROM's warm path reads; `None` for a core not
    /// started from the ROM's hand-off.
    fuses: Option<Fuses>,
    /// Core 0 was released and the monitor's cold boot or resume runs on
    /// its thread: from the slice that saw the release until core 0 runs
    /// or its start failed.
    starting_core0: AtomicBool,
}

/// What [`Bpmp::slice`] works on, under one lock.
struct Running {
    core: Arm7,
    /// The code cache and the trace: not state, snapshots leave it out.
    assist: Assist,
    power: Power,
}

/// Whether the SoC is powered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Power {
    On,
    /// In SC7 until a wake event: the BPMP does not run.
    Sc7,
}

#[derive(Serialize, Deserialize)]
struct SavedBpmp {
    core: Arm7,
    power: Power,
}

/// Which data accesses the trace keeps: device accesses, not memory.
fn traced_access(address: u32) -> bool {
    !IRAM.contains(&address) && u64::from(address) < DRAM_BASE
}

/// IRAM as the boot ROM finds it when the SoC powers up: all zero.
///
/// \[assumption\] What static RAM holds after power is applied is not
/// defined, and the ROM reads it before writing it: the first thing its
/// main routine does (`0x102296`) is look for a debugger's table of
/// messages by a marker at `0x4003_8000`. Any contents but that marker and
/// a matching checksum take the ROM the same way; zeros are that.
fn blank_iram(iram: &Iram) {
    iram.load(0, &vec![0; iram::SIZE as usize])
        .expect("all of IRAM");
}

/// A core at `entry` in ARM state with `cpsr` (its T bit clear).
fn reset_core(core: &mut Arm7, entry: u32, cpsr: u32) -> Result<(), String> {
    if crate::bpmp::Mode::from_bits(cpsr).is_none() || cpsr & crate::bpmp::PSR_T != 0 {
        return Err(format!("BPMP CPSR {cpsr:#x} at hand-off"));
    }
    core.reset(entry);
    core.set_cpsr(cpsr);
    Ok(())
}

impl Bpmp {
    /// The BPMP as the boot ROM hands it over: `iram_image` (what the ROM
    /// wrote, [`crate::bootrom::IRAM_SIZE`] bytes) loaded into `iram` with
    /// what the ROM leaves undefined unestablished, the core at
    /// `handoff.entry` in ARM state with `handoff.cpsr`; `fuses` for the
    /// ROM's warm path.
    pub fn from_handoff(
        handoff: &BootRomHandoff,
        iram_image: &[u8],
        iram: &Iram,
        fuses: Fuses,
    ) -> Result<Self, String> {
        iram.load(0, iram_image)
            .ok_or_else(|| format!("{} bytes of IRAM image", iram_image.len()))?;
        iram.set_unestablished(&handoff.iram_undefined);
        let mut bpmp = Self::new(handoff.entry, handoff.cpsr, iram)?;
        bpmp.fuses = Some(fuses);
        Ok(bpmp)
    }

    /// A core starting at `entry` in ARM state with `cpsr` (its T bit
    /// clear), caching what it fetches from `iram`.
    pub fn new(entry: u32, cpsr: u32, iram: &Iram) -> Result<Self, String> {
        let mut core = Arm7::new();
        reset_core(&mut core, entry, cpsr)?;
        let mut assist = Assist::new();
        assist.set_code_cache(IRAM, iram.code_generation());
        Ok(Self {
            running: Mutex::new(Running {
                core,
                assist,
                power: Power::On,
            }),
            fuses: None,
            starting_core0: AtomicBool::new(false),
        })
    }

    /// The BPMP at its reset in a SoC whose IROM holds a dump of the boot
    /// ROM: at the ROM's reset vector in the state the architecture gives a
    /// core at reset ([`Arm7::reset`]), the ROM's instructions cached like
    /// IRAM's; `fuses` are the console's, kept for reports.
    ///
    /// IRAM reads zero (`blank_iram`).
    pub fn at_reset(blocks: &BootBlocks, fuses: Fuses) -> Result<Self, String> {
        let irom = blocks.irom().ok_or("the SoC has no boot ROM")?;
        blank_iram(&blocks.iram);
        let mut bpmp = Self::new(irom::BASE, RESET_CPSR, &blocks.iram)?;
        let running = bpmp.running.get_mut().unwrap();
        running
            .assist
            .add_code_cache(IROM_IMAGE, irom.code_generation());
        bpmp.fuses = Some(fuses);
        Ok(bpmp)
    }

    /// Keeps the last `depth` instructions the core starts and the last
    /// `depth` accesses it makes outside IRAM and DRAM (with their value or
    /// refusal), for [`Bpmp::trace`]; 0 keeps none. Tracing steps the core
    /// one instruction at a time, which is slower.
    pub fn set_trace(&self, depth: usize) {
        let mut running = self.running.lock().unwrap();
        running.assist.set_trace(depth, depth, traced_access);
    }

    /// What [`Bpmp::set_trace`] kept, oldest first: instructions, then
    /// device accesses.
    pub fn trace(&self) -> Vec<String> {
        let running = self.running.lock().unwrap();
        running
            .assist
            .trace()
            .map(|trace| trace.lines())
            .unwrap_or_default()
    }

    /// The core's registers, for reports and tests.
    pub fn core(&self) -> Arm7 {
        self.running.lock().unwrap().core.clone()
    }

    /// Whether the SoC is powered or in SC7.
    pub fn power(&self) -> Power {
        self.running.lock().unwrap().power
    }

    /// The fuses the boot ROM's warm path reads, if this BPMP started from
    /// its hand-off.
    pub fn fuses(&self) -> Option<&Fuses> {
        self.fuses.as_ref()
    }

    /// Core 0 runs, or its start failed: a snapshot can be taken again.
    pub fn core0_started(&self) {
        self.starting_core0.store(false, Ordering::Release);
    }

    /// Runs at most `budget` instructions, unless something outside the
    /// core must be looked at first; see the module documentation.
    pub fn slice(&self, env: &Env<'_>, budget: u64) -> Slice {
        let mut running = self.running.lock().unwrap();
        if env.clock.is_paused() {
            return Slice::Paused;
        }
        if running.power == Power::Sc7 {
            if !env.power.wake_pending() {
                return Slice::Asleep;
            }
            return match wake(&mut running, env) {
                Ok(()) => Slice::Woke,
                Err(fault) => Slice::Fault(format!("waking from SC7: {fault}")),
            };
        }
        let dpd = env.bus.read(BPMP_CORE, pmc::BASE + pmc::DPD_ENABLE, 4);
        if matches!(dpd, Access::Done(value) if pmc::dpd_entered(value as u32)) {
            return match power_down(&mut running, env) {
                Ok(()) => Slice::PoweredDown,
                Err(fault) => Slice::Fault(format!("entering SC7: {fault}")),
            };
        }
        let Running { core, assist, .. } = &mut *running;
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
            let release = release(env);
            if matches!(release, Slice::Released(_)) {
                self.starting_core0.store(true, Ordering::Release);
            }
            return release;
        }
        let (irq, fiq) = env.blocks.ictlr.cop_lines();
        match env.blocks.flow.cop_status(irq, fiq) {
            CopStatus::Running => {}
            halted => return Slice::Halted(halted),
        }
        core.set_irq(irq);
        core.set_fiq(fiq);
        let mut view = BpmpView {
            env,
            missing_vector: None,
            software_interrupt: false,
        };
        let run = core.run_assisted(&mut view, assist, budget);
        if view.software_interrupt
            && let Some(fault) = foreign_patch(env, core)
        {
            return Slice::Fault(fault);
        }
        match run {
            Ok(_) => Slice::Ran,
            Err(fault) => Slice::Fault(fault.to_string()),
        }
    }
}

/// What the BPMP's loop needs of the machine around it for SC7.
pub trait PowerDomain: Sync {
    /// Whether a wake event is pending at the PMC.
    fn wake_pending(&self) -> bool;
    /// The SoC has powered down: the PMC takes the wake levels of its
    /// "auto" events from their pads ([`crate::soc::Soc::enter_sc7`]), and
    /// what the power-down resets outside the BPMP's own blocks and the
    /// registers of `SC7_RESETS` is reset: the CPU cluster's interrupt
    /// controller ([`crate::gic::Gic::reset`]).
    fn power_lost(&self);
    /// Runs the boot ROM's warm path on the SoC as SC7 left it and applies
    /// what it does outside the BPMP and IRAM; gives what it leaves those.
    fn warm_boot(&self) -> Result<WarmStart, String>;
}

/// What the boot ROM's warm path leaves the BPMP and IRAM.
pub struct WarmStart {
    /// Where the BPMP starts, in ARM state, with `cpsr`.
    pub entry: u32,
    pub cpsr: u32,
    /// All of IRAM ([`crate::bootrom::IRAM_SIZE`] bytes).
    pub iram: Vec<u8>,
    /// IRAM the ROM leaves undefined, by bus address, with why.
    pub undefined: Vec<(Range<u32>, &'static str)>,
}

/// Writes `value` to the register at `address` as the BPMP would.
fn write_register(bus: &Bus, address: u64, value: u32) -> Result<(), String> {
    match bus.write(BPMP_CORE, address, 4, u64::from(value)) {
        Access::Done(()) => Ok(()),
        _ => Err(format!("write of {value:#x} to {address:#010x} not served")),
    }
}

fn read_register(bus: &Bus, address: u64) -> Result<u32, String> {
    match bus.read(BPMP_CORE, address, 4) {
        Access::Done(value) => Ok(value as u32),
        _ => Err(format!("read of {address:#010x} not served")),
    }
}

/// Applies what the boot ROM leaves in a register: the bits of `bits.mask`
/// are `bits.value`. The secure boot registers take them as the ROM's own
/// (lock-outs that bus writes cannot set, [`sb::Sb::apply_rom`]); every
/// other register by read-modify-write as the BPMP.
pub fn apply_rom_bits(blocks: &BootBlocks, bus: &Bus, bits: &RegisterBits) -> Result<(), String> {
    let address = u64::from(bits.address);
    if let Some(offset) = address
        .checked_sub(sb::BASE)
        .filter(|&offset| offset < sb::SIZE)
    {
        return blocks
            .sb
            .apply_rom(offset, bits.mask, bits.value)
            .ok_or_else(|| format!("boot ROM lock-out at {address:#x}"));
    }
    let old = read_register(bus, address)?;
    write_register(bus, address, old & !bits.mask | bits.value & bits.mask)
}

/// The SoC enters SC7: see the module documentation.
fn power_down(running: &mut Running, env: &Env<'_>) -> Result<(), String> {
    let blocks = env.blocks;
    running.power = Power::Sc7;
    running.core.reset(0);
    let signals = &blocks.signals;
    signals.cop_restarted.store(false, Ordering::Release);
    signals.cpu0_released.store(false, Ordering::Release);
    *signals.restart_at.lock().unwrap() = None;
    blocks.flow.reset();
    blocks.evp.reset();
    blocks.ictlr.reset();
    blocks.sb.reset();
    blocks.iram.power_off(IRAM_LOST);
    blocks
        .tzram
        .load(0, &vec![0; tzram::SIZE as usize])
        .expect("all of TZRAM");
    for (address, value) in SC7_RESETS {
        write_register(env.bus, address, value)?;
    }
    env.power.power_lost();
    Ok(())
}

/// The SoC wakes from SC7: see the module documentation.
fn wake(running: &mut Running, env: &Env<'_>) -> Result<(), String> {
    let bus = env.bus;
    let dpd = pmc::BASE + pmc::DPD_ENABLE;
    write_register(bus, dpd, read_register(bus, dpd)? & !pmc::DPD_ENABLE_ON)?;
    for register in pmc::SECURE_LOCKS {
        let value = pmc::power_on_value(register).unwrap_or(0);
        write_register(bus, pmc::BASE + register, value)?;
    }
    if env.blocks.irom().is_some() {
        // The ROM itself: the BPMP comes out of reset at its vector, with
        // the reset reason of a wake from deep power down for it to read.
        write_register(bus, pmc::BASE + pmc::RST_STATUS, pmc::RST_STATUS_LP0)?;
        blank_iram(&env.blocks.iram);
        reset_core(&mut running.core, irom::BASE, RESET_CPSR)?;
        running.power = Power::On;
        return Ok(());
    }
    let start = env.power.warm_boot()?;
    let iram = &env.blocks.iram;
    iram.load(0, &start.iram)
        .ok_or_else(|| format!("{} bytes of IRAM image", start.iram.len()))?;
    iram.set_unestablished(&start.undefined);
    reset_core(&mut running.core, start.entry, start.cpsr)?;
    running.power = Power::On;
    Ok(())
}

/// The world the BPMP runs in.
pub struct Env<'a> {
    pub blocks: &'a BootBlocks,
    pub ram: &'a GuestRam,
    pub bus: &'a Bus,
    pub clock: &'a GuestClock,
    pub notes: &'a Notes,
    pub power: &'a dyn PowerDomain,
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
    /// The SoC entered SC7.
    PoweredDown,
    /// The SoC is in SC7 and nothing wakes it yet.
    Asleep,
    /// The SoC woke from SC7: the BPMP is at the warmboot's entry.
    Woke,
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
            Slice::Ran | Slice::PoweredDown | Slice::Woke => {}
            Slice::Paused | Slice::Held | Slice::Halted(CopStatus::Halted) => {
                thread::sleep(IDLE_POLL)
            }
            Slice::Asleep => thread::sleep(WAKE_POLL),
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
    /// The exception last taken with its vector register unwritten.
    missing_vector: Option<Exception>,
    /// A software interrupt was taken in this run: the run ends there, for
    /// [`foreign_patch`] to look at where it came from.
    software_interrupt: bool,
}

/// Why the software interrupt the core has just taken cannot be served,
/// if it cannot: an `svc` in the boot ROM's dump that the patch unit did
/// not put there (see [`crate::soc::irom`]).
fn foreign_patch(env: &Env<'_>, core: &Arm7) -> Option<String> {
    let irom = env.blocks.irom()?;
    // Thumb: the link register holds the address after the `svc`.
    let from_thumb = core.spsr()? & crate::bpmp::PSR_T != 0;
    let address = core.reg(14).wrapping_sub(2);
    (from_thumb && IROM_IMAGE.contains(&address) && irom.foreign_svc(address - irom::BASE)).then(
        || {
            format!(
                "svc at {address:#010x} in the boot ROM dump is not one of this console's \
                 patches: the dump was read through another device's patch unit, and what \
                 the ROM itself has there is not in it"
            )
        },
    )
}

fn refuse<T>(reason: impl Into<String>) -> Result<T, BusError> {
    Err(BusError::Unmodelled(reason.into()))
}

/// An IRAM read the BPMP cannot make.
fn iram_refused<T>(refusal: Refusal) -> Result<T, BusError> {
    match refusal {
        Refusal::Outside => refuse("access across the end of IRAM"),
        Refusal::Unestablished { address, note } => refuse(format!(
            "IRAM at {address:#010x} holds nothing established: {note}"
        )),
    }
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
        if IROM.contains(&address) && self.env.blocks.irom().is_none() {
            return refuse("the boot ROM (IROM) is not available: it is high-level emulated");
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

    /// Whether `len` bytes at `address` lie in IRAM, and where.
    fn iram(address: u32, len: usize) -> Option<u32> {
        let end = u64::from(address) + len as u64;
        (IRAM.contains(&address) && end <= u64::from(IRAM.end)).then(|| address - IRAM.start)
    }

    /// A device access nothing models: refused, unless a survey is being
    /// run ([`Notes::survey`]), which notes it and goes on (a read then
    /// gives 0).
    fn surveyed(&self, access: String, why: String) -> Result<(), BusError> {
        if !self.env.notes.survey() {
            return refuse(why);
        }
        self.env
            .notes
            .add(format!("BPMP {access}: {why} (answered by the survey)"));
        Ok(())
    }
}

impl BpmpBus for BpmpView<'_> {
    fn read(&mut self, address: u32, size: u8) -> Result<u32, BusError> {
        if IROM.contains(&address)
            && let Some(irom) = self.env.blocks.irom()
        {
            return match irom.read(address - IROM.start, size) {
                Ok(value) => Ok(value),
                Err(irom::Refusal::Outside) => refuse("IROM past the boot ROM's 96 KiB"),
                Err(irom::Refusal::Hidden) => refuse(
                    "the boot ROM hid itself (SB_CSR PIROM_DISABLE): what a read there gives \
                     is not established",
                ),
            };
        }
        if IRAM.contains(&address) {
            let offset = address - IRAM.start;
            return self
                .env
                .blocks
                .iram
                .bpmp_read(offset, size)
                .or_else(iram_refused);
        }
        self.unmodelled(address)?;
        if let Some(address) = self.dram(address, size) {
            let mut bytes = [0; 4];
            self.env
                .ram
                .read(address, &mut bytes[..usize::from(size)])
                .expect("checked to be DRAM");
            return Ok(u32::from_le_bytes(bytes));
        }
        let why = match self.env.bus.read(BPMP_CORE, u64::from(address), size) {
            Access::Done(value) => return Ok(value as u32),
            Access::Unmapped => "nothing is mapped there".to_string(),
            Access::Unimplemented(device) => {
                format!("the {device} does not model this register or width")
            }
        };
        self.surveyed(format!("{size}-byte read at {address:#010x}"), why)
            .map(|()| 0)
    }

    fn write(&mut self, address: u32, size: u8, value: u32) -> Result<(), BusError> {
        if IRAM.contains(&address) {
            let offset = address - IRAM.start;
            return self
                .env
                .blocks
                .iram
                .bpmp_write(offset, size, value)
                .map_or_else(|| refuse("access across the end of IRAM"), Ok);
        }
        if IROM.contains(&address) && self.env.blocks.irom().is_some() {
            return refuse("a write to the boot ROM");
        }
        self.unmodelled(address)?;
        if let Some(address) = self.dram(address, size) {
            self.env
                .ram
                .write(address, &value.to_le_bytes()[..usize::from(size)])
                .expect("checked to be DRAM");
            return Ok(());
        }
        let why = match self
            .env
            .bus
            .write(BPMP_CORE, u64::from(address), size, u64::from(value))
        {
            Access::Done(()) => return Ok(()),
            Access::Unmapped => "nothing is mapped there".to_string(),
            Access::Unimplemented(device) => {
                format!("the {device} does not model this register or width")
            }
        };
        self.surveyed(
            format!("{size}-byte write of {value:#x} at {address:#010x}"),
            why,
        )
    }

    /// IRAM in one go when all of it is established; word by word
    /// otherwise, which finds the word that is refused.
    fn read_words(&mut self, address: u32, words: &mut [u32]) -> Result<(), (usize, BusError)> {
        if let Some(offset) = Self::iram(address, 4 * words.len())
            && self.env.blocks.iram.bpmp_read_words(offset, words).is_ok()
        {
            return Ok(());
        }
        for (i, word) in words.iter_mut().enumerate() {
            *word = self
                .read(address.wrapping_add(4 * i as u32), 4)
                .map_err(|error| (i, error))?;
        }
        Ok(())
    }

    fn write_words(&mut self, address: u32, words: &[u32]) -> Result<(), (usize, BusError)> {
        if let Some(offset) = Self::iram(address, 4 * words.len())
            && self
                .env
                .blocks
                .iram
                .bpmp_write_words(offset, words)
                .is_some()
        {
            return Ok(());
        }
        for (i, &word) in words.iter().enumerate() {
            self.write(address.wrapping_add(4 * i as u32), 4, word)
                .map_err(|error| (i, error))?;
        }
        Ok(())
    }

    fn exception_vector(&mut self, exception: Exception) -> u32 {
        match self.env.blocks.evp.cop_vector(exception.offset()) {
            Some(vector) => {
                self.software_interrupt |= exception == Exception::SoftwareInterrupt;
                vector
            }
            None => {
                self.missing_vector = Some(exception);
                exception.offset()
            }
        }
    }

    fn should_pause(&mut self) -> bool {
        let signals = &self.env.blocks.signals;
        self.software_interrupt
            || self.env.blocks.flow.take_halt_change()
            || signals.cpu0_released.load(Ordering::Acquire)
            || signals.cop_held.load(Ordering::Acquire)
            || signals.cop_restarted.load(Ordering::Acquire)
    }
}

/// The core's registers and whether the SoC is in SC7 (which IRAM bytes
/// are unestablished is IRAM's).
impl Snapshot for Bpmp {
    /// A slice in progress ends; the next one sees guest time paused.
    /// Refused while core 0 is being started ([`Bpmp::core0_started`]).
    fn quiesce(&self) -> Result<(), SnapshotError> {
        drop(self.running.lock().unwrap());
        if self.starting_core0.load(Ordering::Acquire) {
            return Err(SnapshotError::Unsupported(
                "core 0 being started: the secure monitor's cold boot or SC7 resume runs \
                 on its thread"
                    .into(),
            ));
        }
        Ok(())
    }

    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        let running = self.running.lock().unwrap();
        encode(&SavedBpmp {
            core: running.core.clone(),
            power: running.power,
        })
    }

    fn restore(&self, state: &[u8]) -> Result<(), SnapshotError> {
        let saved: SavedBpmp = decode("BPMP", state)?;
        if crate::bpmp::Mode::from_bits(saved.core.cpsr()).is_none() {
            return Err(SnapshotError::Corrupt("BPMP: reserved mode bits".into()));
        }
        let mut running = self.running.lock().unwrap();
        running.core = saved.core;
        running.power = saved.power;
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

    /// A secure-only slot is loaded with [`crate::soc::se::LOCK_PER_KEY`],
    /// which clears its bit of `SE_CRYPTO_SECURITY_PERKEY`; the slot keeps
    /// every access permission either way (what the TSEC model leaves, see
    /// [`crate::soc::tsec`]).
    fn load_se_keyslot(&self, slot: usize, key: &Key, secure_only: bool) -> Option<()> {
        let flags = if secure_only {
            crate::soc::se::LOCK_PER_KEY
        } else {
            0
        };
        self.se.load_key(slot, key, flags).ok()
    }

    fn restart_bpmp(&self, entry: u32) -> Option<()> {
        *self.signals.restart_at.lock().unwrap() = Some(entry);
        self.signals.cop_restarted.store(true, Ordering::Release);
        Some(())
    }
}

impl Bpmp {
    /// The BPMP of `soc`, a SoC powered on whose devices are on `bus`, as
    /// `power_on` starts it, with the TSEC connected to the machine
    /// ([`TsecPort`]) and what `keys` hold for it.
    ///
    /// With the high-level emulated ROM its hand-off is applied: IRAM as
    /// the ROM wrote it, the ROM's keyslots in the security engine with
    /// the permissions it leaves them, its lock-out in the secure boot
    /// registers ([`crate::bootrom::ROM_LOCKOUT`]). With a dump of the
    /// ROM nothing is applied: the BPMP starts at its reset
    /// ([`Bpmp::at_reset`]) and the ROM does all of that itself.
    pub fn power_on(
        power_on: &PowerOn,
        soc: &Soc,
        bus: &Arc<Bus>,
        keys: &KeySet,
    ) -> Result<Self, String> {
        let boot = soc.boot.as_ref().ok_or("the SoC was not powered on")?;
        TsecPort::connect(soc, bus, keys).ok_or("the SoC has no boot blocks")?;
        let (handoff, iram, fuses) = match power_on {
            PowerOn::Rom { fuses, .. } => return Self::at_reset(boot, fuses.clone()),
            PowerOn::Emulated {
                handoff,
                iram,
                fuses,
            } => (handoff, iram, fuses),
        };
        for slot in &handoff.keyslots {
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
        apply_rom_bits(boot, bus, &bootrom::ROM_LOCKOUT)?;
        Self::from_handoff(handoff, iram, &boot.iram, fuses.clone())
    }
}

/// The machine around the BPMP in SC7: the wake comes from the SoC's PMC
/// and the pads it watches ([`Soc::wake_pending`]), the warm path is the
/// boot ROM's ([`bootrom::warm_boot`]) with the fuses it read at power-on,
/// the PMC and DRAM as the BPMP reads them.
impl PowerDomain for Machine {
    fn wake_pending(&self) -> bool {
        self.soc.wake_pending()
    }

    fn power_lost(&self) {
        self.soc.enter_sc7();
        self.gic.reset();
    }

    fn warm_boot(&self) -> Result<WarmStart, String> {
        let fuses = self
            .bpmp
            .as_ref()
            .and_then(Bpmp::fuses)
            .ok_or("the BPMP did not start from the boot ROM's hand-off")?;
        warm_start(&self.soc, &self.bus, &self.ram, fuses)
    }
}

/// The boot ROM's warm path on `soc` as SC7 left it ([`bootrom::warm_boot`]
/// with the `fuses` it read at power-on, the PMC and DRAM as the BPMP reads
/// them): its register writes and the security engine's context applied,
/// what it leaves the BPMP and IRAM returned.
pub fn warm_start(
    soc: &Soc,
    bus: &Bus,
    ram: &GuestRam,
    fuses: &Fuses,
) -> Result<WarmStart, String> {
    let blocks = soc.boot.as_ref().ok_or("the SoC was not powered on")?;
    let pmc = |offset: u32| read_register(bus, pmc::BASE + u64::from(offset)).ok();
    let dram = |address: u32, length: usize| {
        let mut bytes = vec![0; length];
        ram.read(u64::from(address), &mut bytes).map(|()| bytes)
    };
    let mut iram = vec![0; bootrom::IRAM_SIZE];
    let handoff =
        bootrom::warm_boot(fuses, &pmc, &dram, &mut iram).map_err(|error| error.to_string())?;
    for bits in &handoff.registers {
        apply_rom_bits(blocks, bus, bits)?;
    }
    soc.se.restore_context(&handoff.se_context);
    Ok(WarmStart {
        entry: handoff.entry,
        cpsr: handoff.cpsr,
        iram,
        undefined: handoff.iram_undefined,
    })
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
                power: self,
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
        let trace = || {
            let trace = bpmp.trace();
            if trace.is_empty() {
                String::new()
            } else {
                format!(
                    "; last of what it did, oldest first:\n  {}",
                    trace.join("\n  ")
                )
            }
        };
        let stop = match exit {
            BpmpExit::Stopped => {
                let power = match bpmp.power() {
                    Power::On => "powered",
                    Power::Sc7 => "in SC7",
                };
                env.notes.add(format!(
                    "BPMP when the machine stopped: {power}, pc {:#010x}{}",
                    bpmp.core().pc(),
                    trace()
                ));
                CoreStop::Requested
            }
            BpmpExit::Fault(fault) => {
                env.notes.add(format!("BPMP: {fault}{}", trace()));
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
        self.spawn_reporting(scope, 0, move || {
            let released = self.core0_released(entry);
            if released.is_ok() {
                self.switches[0].lock().unwrap().0 = true;
                self.machine.pause.state.lock().unwrap().threads[0] = true;
            }
            if let Some(bpmp) = &self.machine.bpmp {
                bpmp.core0_started();
            }
            match released {
                Ok(entry) => self.run_core(scope, 0, Start::Entry(entry)),
                Err(stop) => CoreReport {
                    core: 0,
                    stop,
                    state: None,
                },
            }
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
    use crate::{
        bus::MmioDevice,
        soc::{
            BootSignals, car::Car, evp, evp::Evp, flow::FlowController, ictlr::Ictlr, iram::Iram,
            pmc::Pmc, regs::RegisterFile, sb::Sb, tzram::Tzram,
        },
    };

    /// The machine's side of SC7, as a test drives it: a wake when `wake`
    /// is set, and the warm start it holds.
    #[derive(Default)]
    struct MockPower {
        wake: AtomicBool,
        start: StdMutex<Option<WarmStart>>,
    }

    impl PowerDomain for MockPower {
        fn wake_pending(&self) -> bool {
            self.wake.load(Ordering::SeqCst)
        }

        /// No interrupt controller in these tests.
        fn power_lost(&self) {}

        fn warm_boot(&self) -> Result<WarmStart, String> {
            self.start
                .lock()
                .unwrap()
                .take()
                .ok_or_else(|| "no warm start".into())
        }
    }

    /// The SoC as far as the BPMP needs it, wired as `Soc::new` wires a
    /// machine powered on (the memory controller as plain storage).
    struct Bench {
        blocks: BootBlocks,
        ram: GuestRam,
        bus: Bus,
        clock: Arc<GuestClock>,
        notes: Arc<Notes>,
        power: MockPower,
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
                tzram: Arc::new(Tzram::new()),
                signals: Arc::new(BootSignals::default()),
                rom: None,
            };
            let mut bus = Bus::default();
            let signal = |set: fn(&BootSignals)| {
                let signals = Arc::clone(&blocks.signals);
                Box::new(move || set(&signals))
            };
            // `RST_DEVICES_L` bit 1: the BPMP, as `Soc::new` wires it.
            let car = Car::power_on()
                .on_cpu0_released(signal(|s| s.cpu0_released.store(true, Ordering::Release)))
                .on_reset_asserted(
                    0x004,
                    1,
                    signal(|s| s.cop_held.store(true, Ordering::Release)),
                )
                .on_reset_released(
                    0x004,
                    1,
                    signal(|s| {
                        s.cop_held.store(false, Ordering::Release);
                        s.cop_restarted.store(true, Ordering::Release);
                    }),
                );
            bus.map("clock and reset", car::BASE, car::SIZE, Box::new(car));
            bus.map("PMC", pmc::BASE, pmc::SIZE, Box::new(Pmc::power_on()));
            bus.map(
                "memory controller",
                mc::BASE,
                0x1000,
                Box::new(RegisterFile::zeroed([])),
            );
            bus.map(
                "IRAM",
                iram::BASE,
                iram::SIZE,
                Box::new(Arc::clone(&blocks.iram)),
            );
            bus.map(
                "TZRAM",
                tzram::BASE,
                tzram::SIZE,
                Box::new(Arc::clone(&blocks.tzram)),
            );
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
                sb::BASE,
                sb::SIZE,
                Box::new(Arc::clone(&blocks.sb)),
            );
            Self {
                blocks,
                ram: GuestRam::new(0x8000_0000, 0x10_0000).unwrap(),
                bus,
                clock,
                notes,
                power: MockPower::default(),
            }
        }

        fn env(&self) -> Env<'_> {
            Env {
                blocks: &self.blocks,
                ram: &self.ram,
                bus: &self.bus,
                clock: &self.clock,
                notes: &self.notes,
                power: &self.power,
            }
        }

        /// A register as the BPMP reads it.
        fn register(&self, address: u64) -> u32 {
            read_register(&self.bus, address).unwrap()
        }

        /// A register written by CPU-cluster core 0.
        fn write(&self, address: u64, value: u32) {
            assert_eq!(
                self.bus.write(0, address, 4, u64::from(value)),
                Access::Done(()),
                "{address:#x}"
            );
        }

        fn bpmp(&self, entry: u32) -> Bpmp {
            Bpmp::new(entry, 0xD3, &self.blocks.iram).unwrap()
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
        bench.bpmp(BASE)
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
    fn unestablished_iram_refuses_reads_until_any_master_writes_it() {
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
        bench
            .blocks
            .iram
            .set_unestablished(&[(0x4000_0000..0x4001_0000, "the ROM's work area")]);
        let bpmp = bench.bpmp(BASE);
        let Slice::Fault(fault) = bpmp.slice(&bench.env(), 100) else {
            panic!("the read must be refused")
        };
        assert!(fault.contains("the ROM's work area"), "{fault}");
        assert!(fault.contains("0x40000044"), "{fault}");
        assert_eq!(bpmp.core().reg(2), 7);
        assert_eq!(bpmp.core().pc(), BASE + 4 * 4);
        // The CPU cluster writes the word over the bus: the BPMP, still at
        // the load that was refused, reads it.
        bench.write(0x4000_0044, 0x1234);
        assert_eq!(bpmp.slice(&bench.env(), 1), Slice::Ran);
        assert_eq!(bpmp.core().reg(3), 0x1234);
    }

    #[test]
    fn exceptions_go_to_the_evp_vectors_or_stop_the_bpmp() {
        let bench = Bench::new();
        let swi = 0xEF00_0000u32; // SWI #0
        let program = assemble(&[Op::Word(swi), Op::Word(B_SELF)]);
        bench.blocks.iram.load(BASE - IRAM.start, &program).unwrap();
        let bpmp = bench.bpmp(BASE);
        let Slice::Fault(fault) = bpmp.slice(&bench.env(), 10) else {
            panic!("an unwritten vector must stop the BPMP")
        };
        assert!(fault.contains("SoftwareInterrupt"), "{fault}");
        // With the vector written, the SWI lands there.
        let bench = Bench::new();
        bench.blocks.iram.load(BASE - IRAM.start, &program).unwrap();
        bench
            .blocks
            .evp
            .write(BPMP_CORE, 0x208, 4, u64::from(BASE + 4))
            .unwrap();
        let bpmp = bench.bpmp(BASE);
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
        let env = bench.env();
        let mut view = BpmpView {
            env: &env,
            missing_vector: None,
            software_interrupt: false,
        };
        assert_eq!(view.read(0x8000_0010, 4), Ok(0x1234_5678));
        view.write(0x8000_0014, 2, 0xBEEF).unwrap();
        assert_eq!(view.read(0x8000_0014, 2), Ok(0xBEEF));
        for address in [0x5004_1000, 0x0010_0000, 0x0000_0008] {
            assert!(
                matches!(view.read(address, 4), Err(BusError::Unmodelled(_))),
                "{address:#x}"
            );
        }
        // MMIO through the bus, as the BPMP.
        view.write(0x6000_C230, 4, 0x4003_0001).unwrap();
        assert_eq!(bench.blocks.sb.read(0, 0x30, 4), Some(0x4003_0001));
        assert!(matches!(
            view.read(0x6000_C238, 4),
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
        let bpmp = bench.bpmp(BASE);
        bench.clock.pause();
        let env = bench.env();
        assert_eq!(bpmp.slice(&env, 100), Slice::Paused);
        bench.clock.resume();
        assert_eq!(bpmp.slice(&env, 100), Slice::Ran);
        assert!(matches!(
            bpmp.slice(&env, 100),
            Slice::Halted(CopStatus::HaltedUntil(_)) | Slice::Ran
        ));
        let due = bench.clock.now() + Duration::from_micros(60);
        while !bench.clock.until(due).is_zero() {
            std::thread::sleep(bench.clock.until(due));
        }
        assert_eq!(bpmp.slice(&env, 100), Slice::Ran);
        assert_eq!(bpmp.core().pc(), BASE + 3 * 4);
    }

    #[test]
    fn snapshot_round_trip() {
        let bench = Bench::new();
        let program = assemble(&[Op::Word(0xE3A0_0005), Op::Word(B_SELF)]); // MOV r0, #5
        bench.blocks.iram.load(BASE - IRAM.start, &program).unwrap();
        let bpmp = bench.bpmp(BASE);
        assert_eq!(bpmp.slice(&bench.env(), 1), Slice::Ran);
        bpmp.quiesce().unwrap();
        let saved = bpmp.save().unwrap();
        let other = bench.bpmp(0);
        other.restore(&saved).unwrap();
        assert_eq!(other.core(), bpmp.core());
        assert_eq!(other.power(), Power::On);
        // While core 0 is being started, a snapshot is refused.
        bpmp.starting_core0.store(true, Ordering::Release);
        assert!(bpmp.quiesce().is_err());
        bpmp.core0_started();
        assert!(bpmp.quiesce().is_ok());
    }

    /// What the NX bootloader's hardware check reads of keyslots 0x0C and
    /// 0x0D after the TSEC has loaded them (its CSV rows for SE+0x280 and
    /// SE+0x2B4/0x2B8).
    #[test]
    fn the_tsec_s_secure_only_keyslot_loses_only_its_per_key_bit() {
        use crate::soc::mc::{DeviceMemory, Mc, SMMU_SE_ASID};
        let mc = Arc::new(Mc::new(crate::boot::DramSize::Gb4));
        let ram = Arc::new(GuestRam::new(0x8000_0000, 0x1000).unwrap());
        let se = Arc::new(Se::new(
            Box::new(DeviceMemory::new(&mc, &ram, SMMU_SE_ASID)),
            Arc::default(),
            0,
        ));
        let port = TsecPort {
            iram: Arc::new(Iram::new()),
            bus: Weak::new(),
            se: Arc::clone(&se),
            signals: Arc::default(),
        };
        port.load_se_keyslot(0x0D, &[1; 16], true).unwrap();
        port.load_se_keyslot(0x0C, &[2; 16], false).unwrap();
        let read = |offset| se.read(BPMP_CORE, offset, 4).unwrap() as u32;
        assert_eq!(read(0x280) & 0xFFFF, 0xDFFF);
        assert_eq!(read(0x284 + 0x0C * 4) & 0x7F, 0x7F);
        assert_eq!(read(0x284 + 0x0D * 4) & 0x7F, 0x7F);
    }

    /// What the secure monitor's SC7-entry stub ends with: `DPD_ENABLE`
    /// bit 0, then a loop.
    fn sc7_stub() -> Vec<u8> {
        use Op::*;
        assemble(&[
            LdrLiteral(0, 0x7000_E424), // DPD_ENABLE
            LdrLiteral(1, 1),
            Str(1, 0, 0),
            Word(B_SELF),
        ])
    }

    const WARMBOOT: u32 = 0x4001_0240;
    const RESUME: u32 = 0x7C01_2800;
    const SEC_DISABLE3: u64 = pmc::BASE + 0x2D8;

    /// What the warmboot firmware does to start core 0 again: the reset
    /// vector from `SECURE_SCRATCH34`, the CPU partitions on, the cluster's
    /// clock, then `RST_CPUG_CMPLX_CLR` 0x20000000 and 0x10001
    /// (`warmboot-research` §7), then a halt.
    fn warmboot_program() -> Vec<u8> {
        use Op::*;
        assemble(&[
            LdrLiteral(0, 0x6000_C230),
            LdrLiteral(1, RESUME | 1),
            Str(1, 0, 0),
            LdrLiteral(1, 0),
            Str(1, 0, 4),
            LdrLiteral(0, 0x7000_E430),
            LdrLiteral(1, 0x100),
            Str(1, 0, 0),
            LdrLiteral(1, 0x10F),
            Str(1, 0, 0),
            LdrLiteral(1, 0x10E),
            Str(1, 0, 0),
            LdrLiteral(0, 0x6000_6000),
            LdrLiteral(1, 1),
            Str(1, 0, 0x360),
            LdrLiteral(1, 1 << 29),
            Str(1, 0, 0x454),
            LdrLiteral(1, 0x10001),
            Str(1, 0, 0x454),
            LdrLiteral(0, 0x6000_7000),
            LdrLiteral(1, 0x5000_0000),
            Str(1, 0, 4),
            Word(B_SELF),
        ])
    }

    /// The boot ROM's warm path as the mock gives it: the warmboot's code
    /// at its destination, the rest of IRAM undefined.
    fn warm_start(program: &[u8]) -> WarmStart {
        let mut iram = vec![0; bootrom::IRAM_SIZE];
        let at = (WARMBOOT - IRAM.start) as usize;
        iram[at..at + program.len()].copy_from_slice(program);
        WarmStart {
            entry: WARMBOOT,
            cpsr: 0xD3,
            iram,
            undefined: vec![(IRAM.start..WARMBOOT, "the boot ROM's work area")],
        }
    }

    #[test]
    fn sc7_entry_resets_the_documented_blocks_and_zeroes_tzram() {
        let bench = Bench::new();
        let bpmp = load(&bench, &sc7_stub());
        let env = bench.env();
        // What the SoC holds before: written by the cold boot and the
        // monitor's suspend.
        bench.blocks.tzram.load(0x2000, &[0xAB; 16]).unwrap();
        bench.write(evp::BASE + 0x200, 0x4000_3000);
        bench.write(sb::BASE + 0x30, RESUME | 1);
        bench.write(mc::BASE + 0x9BC, 3);
        bench.write(car::BASE + 0x454, 0x4101_0001);
        bench.write(crate::soc::flow::BASE + 0x098, 1);
        bench.write(SEC_DISABLE3, 0xFF);
        assert_eq!(bpmp.slice(&env, 100), Slice::Ran);
        assert_eq!(bpmp.slice(&env, 100), Slice::PoweredDown);
        assert_eq!(bpmp.power(), Power::Sc7);
        // Reset or lost.
        let mut tzram = vec![0xFF; tzram::SIZE as usize];
        bench.blocks.tzram.copy_out(0, &mut tzram).unwrap();
        assert!(tzram.iter().all(|&byte| byte == 0), "TZRAM zeroed");
        assert!(matches!(
            bench.blocks.iram.bpmp_read(BASE - IRAM.start, 4),
            Err(Refusal::Unestablished { note, .. }) if note == IRAM_LOST
        ));
        assert_eq!(
            Iram::read(&bench.blocks.iram, BASE - IRAM.start, 4),
            Some(0)
        );
        assert_eq!(bench.blocks.evp.cop_vector(0), None);
        assert!(!bench.blocks.sb.aa64_reset_vector().aarch64);
        assert_eq!(bench.register(mc::BASE + 0x9BC), 0);
        assert_eq!(
            bench.register(car::BASE + 0x450),
            car::CPUG_RESETS_AFTER_SC7
        );
        assert_eq!(bench.register(crate::soc::flow::BASE + 0x098), 0);
        // Kept until the wake: the PMC.
        assert_eq!(bench.register(SEC_DISABLE3), 0xFF);
        assert_eq!(bench.register(pmc::BASE + pmc::DPD_ENABLE), 1);
        // Asleep until a wake event, a state snapshots keep.
        assert_eq!(bpmp.slice(&env, 100), Slice::Asleep);
        let other = bench.bpmp(0);
        other.restore(&bpmp.save().unwrap()).unwrap();
        assert_eq!(other.power(), Power::Sc7);

        // The wake: the ROM's warm path starts the warmboot.
        *bench.power.start.lock().unwrap() = Some(warm_start(&warmboot_program()));
        bench.power.wake.store(true, Ordering::SeqCst);
        assert_eq!(bpmp.slice(&env, 100), Slice::Woke);
        assert_eq!(bpmp.power(), Power::On);
        assert_eq!(bpmp.core().pc(), WARMBOOT);
        assert_eq!(bench.register(pmc::BASE + pmc::DPD_ENABLE), 0);
        assert_eq!(
            bench.register(SEC_DISABLE3),
            pmc::power_on_value(0x2D8).unwrap()
        );
        assert!(matches!(
            bench.blocks.iram.bpmp_read(0x100, 4),
            Err(Refusal::Unestablished { note, .. }) if note == "the boot ROM's work area"
        ));
        assert!(bench.blocks.iram.bpmp_read(0x2_0000, 4).is_ok());
    }

    /// The whole path through the hooks, with the test as the secure
    /// monitor: the cold boot's release, `CPU_SUSPEND` (the stub copied to
    /// IRAM over the bus, the BPMP restarted on it, core 0's partitions
    /// off), SC7, the wake, the warmboot, and core 0's release at the
    /// monitor's resume vector.
    #[test]
    fn core_0_suspends_and_is_released_again_at_the_resume_vector() {
        let bench = Bench::new();
        let bpmp = load(&bench, &release_program());
        let stub_at = 0x4000_3000u32;
        bench
            .blocks
            .iram
            .set_unestablished(&[(stub_at..stub_at + 0x100, "not written yet")]);
        let entries = StdMutex::new(Vec::new());
        let stop = AtomicBool::new(false);
        let env = bench.env();
        let monitor = |entry: u64| {
            entries.lock().unwrap().push(entry);
            bpmp.core0_started();
            if entry != u64::from(SECMON_ENTRY) {
                return Ok(());
            }
            // CPU_SUSPEND, as the monitor's HLE does it over the bus.
            for (offset, chunk) in sc7_stub().chunks(4).enumerate() {
                let word = u32::from_le_bytes(chunk.try_into().unwrap());
                bench.write(u64::from(stub_at) + 4 * offset as u64, word);
            }
            bench.write(evp::BASE + 0x200, stub_at);
            bench.write(car::BASE + 0x300, 2);
            bench.write(car::BASE + 0x304, 2);
            bench.write(crate::soc::flow::BASE + 0x004, 0);
            for partition in [14, 15, 0] {
                bench.write(pmc::BASE + 0x30, 0x100 | partition);
            }
            *bench.power.start.lock().unwrap() = Some(warm_start(&warmboot_program()));
            Ok(())
        };
        let mut monitor = monitor;
        thread::scope(|scope| {
            let runner = scope.spawn(|| run(&bpmp, &env, &stop, &mut monitor));
            let wait = |done: &dyn Fn() -> bool| {
                let deadline = std::time::Instant::now() + Duration::from_secs(10);
                while !done() && std::time::Instant::now() < deadline {
                    thread::sleep(Duration::from_millis(1));
                }
                assert!(done(), "timed out");
            };
            wait(&|| bpmp.power() == Power::Sc7);
            // Asleep: nothing runs until the PMIC's wake event.
            thread::sleep(Duration::from_millis(20));
            assert_eq!(entries.lock().unwrap().len(), 1);
            bench.power.wake.store(true, Ordering::SeqCst);
            wait(&|| entries.lock().unwrap().len() == 2);
            stop.store(true, Ordering::SeqCst);
            assert_eq!(runner.join().unwrap(), BpmpExit::Stopped);
        });
        assert_eq!(
            *entries.lock().unwrap(),
            [u64::from(SECMON_ENTRY), u64::from(RESUME)]
        );
        assert_eq!(bpmp.power(), Power::On);
    }
}
