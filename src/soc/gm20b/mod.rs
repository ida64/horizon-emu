//! The GM20B GPU as nvservices' driver sees it: its register window BAR0,
//! its memory window BAR1, two interrupt lines, and a host that executes
//! channels' command streams.
//!
//! The driver powers the GPU on during its own start-up and off again at
//! once, and on again whenever a client opens it (spec §4.1-4.3). Power-on
//! is a long register sequence; what it needs from the hardware, and what
//! is modeled for it here:
//!
//! * Identification and configuration ([`identity`]).
//! * Registers that hold what the driver writes, so its read-modify-write
//!   sequences and read-backs work. Writes anywhere in the units the
//!   driver programs are kept; reading a register nobody wrote refuses,
//!   unless it is an identity register or one of the few configuration
//!   registers listed in [`RESET_ZERO`].
//! * Operations that are over at once and say so: framebuffer and L2
//!   flushes, MMU invalidates, priv ring commands, VPR fetches, falcon
//!   memory scrubbing, LTC maintenance.
//! * The falcons: the secure-boot hand-off of the PMU and the FECS
//!   handshake and methods ([`falcon`]).
//! * BAR1: accesses go through the page tables of the BAR1 address space
//!   whose instance block `bus_bar1_block` (`0x1704`) names, then through
//!   the GPU's SMMU address space onto memory ([`mmu`]). The driver checks
//!   this at every power-on by writing a word through its own mapping and
//!   reading it back through BAR1 (`NV_InitializeSchedulerMemory_70FD0`).
//! * The host ([`fifo`]): channels, runlist, preemption and the execution
//!   of GPFIFO entries, on a thread of its own so that a channel waiting
//!   for a syncpoint holds up no processor. Engine methods go to each
//!   channel's [`Engines`].
//!
//! Interrupts: the stall line (`mc_intr_0`, INTID 189) carries the FIFO's
//! stall-routed interrupts and the PMU's; the non-stall line (`mc_intr_1`,
//! INTID 190) the FIFO's non-stall-routed ones, which is how the runlist
//! interrupt the driver waits for in the nvsched runlist swap arrives. The
//! model raises nothing else, so the stall line stays quiet.
//!
//! When a channel meets something the model does not do, the GPU stops:
//! the reason is left as a note, and every later access to either window
//! is refused with it, which stops the machine.

pub mod falcon;
pub mod fifo;
pub mod identity;
pub mod mmu;
pub mod pushbuffer;

use std::{
    collections::HashMap,
    ops::Range,
    sync::{
        Arc, Condvar, Mutex, MutexGuard,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::Duration,
};

use self::{
    falcon::{Fecs, Gpccs, Pmu},
    fifo::{Context, Env, Fifo, Outcome},
    mmu::{AddressSpace, BusMemory, SmmuBus},
};
use super::{Notes, host1x::Syncpoints, mc::DeviceMemory, regs::word};
use crate::{
    bus::MmioDevice,
    maxwell::{self, Engines, Unmodeled},
};

pub const BAR0_BASE: u64 = 0x5700_0000;
pub const BAR0_SIZE: u64 = 0x100_0000;
pub const BAR1_BASE: u64 = 0x5800_0000;
pub const BAR1_SIZE: u64 = 0x100_0000;
/// The stall and non-stall interrupt lines (SPI 157 and 158; the IDs
/// nvservices binds, spec §4.1).
pub const STALL_INTID: u32 = 32 + 157;
pub const NONSTALL_INTID: u32 = 32 + 158;

/// How often a channel waiting for a syncpoint or semaphore looks again.
/// Nothing tells the host when a syncpoint moves, so waits are polled.
const WAIT_POLL: Duration = Duration::from_micros(500);

/// Called with the level of an interrupt output.
pub type InterruptLine = Box<dyn Fn(bool) + Send + Sync>;
/// Time since the GPU's timer started.
pub type Clock = Box<dyn Fn() -> Duration + Send + Sync>;
/// Makes the engines of a new channel, given the method-init list the
/// driver loaded into the graphics engine (`(index, value)`, index =
/// method << 16 | class) and the GPU timer in nanoseconds.
pub type EngineFactory = Box<
    dyn Fn(&[(u32, u32)], Box<dyn Fn() -> u64 + Send>) -> Result<Box<dyn Engines>, Unmodeled>
        + Send
        + Sync,
>;

/// Engines from [`maxwell::Channel`]: started from the method-init list,
/// given the GPU timer, and counting the 3D class's own syncpoint
/// increments on `syncpoints`.
pub fn maxwell_engines(syncpoints: Arc<Syncpoints>) -> EngineFactory {
    Box::new(move |method_init, timer| {
        let mut channel = maxwell::Channel::new();
        for &(index, value) in method_init {
            channel.set_initial_state(index as u16, index >> 16, value)?;
        }
        channel.set_timer(timer);
        let syncpoints = Arc::clone(&syncpoints);
        channel.set_syncpoint_increment(Box::new(move |index| syncpoints.add(index as usize, 1)));
        Ok(maxwell::capture::wrap(method_init, Box::new(channel)))
    })
}

/// BAR0 registers with behavior of their own here.
mod reg {
    pub const MC_INTR_0: u32 = 0x100;
    pub const MC_INTR_1: u32 = 0x104;
    pub const MC_INTR_EN_0: u32 = 0x140;
    pub const MC_INTR_EN_1: u32 = 0x144;
    pub const MC_ENABLE: u32 = 0x200;
    pub const MC_INTR_MASK_0: u32 = 0x640;
    pub const MC_INTR_MASK_1: u32 = 0x644;
    pub const BAR1_BLOCK: u32 = 0x1704;
    pub const TIMER_TIME_0: u32 = 0x9400;
    pub const TIMER_TIME_1: u32 = 0x9410;
    pub const FB_FLUSH: u32 = 0x7_0000;
    pub const L2_SYSTEM_INVALIDATE: u32 = 0x7_0004;
    pub const L2_CLEAN_COMPTAGS: u32 = 0x7_000C;
    pub const L2_FLUSH_DIRTY: u32 = 0x7_0010;
    pub const MMU_CTRL: u32 = 0x10_0C80;
    pub const MMU_VPR_INFO: u32 = 0x10_0CD0;
    pub const RING_COMMAND: u32 = 0x12_004C;
    pub const RING_INTR_STATUS_0: u32 = 0x12_0058;
    pub const GR_STATUS: u32 = 0x40_060C;
    pub const GR_ACTIVITY: u32 = 0x40_0700;
    pub const MME_SHADOW_RAW_INDEX: u32 = 0x40_4488;
    pub const MME_SHADOW_RAW_DATA: u32 = 0x40_448C;
    pub const GPCPLL_CFG: u32 = 0x13_7000;
    pub const GPCPLL_DVFS1: u32 = 0x13_7014;
    pub const GPCPLL_CFG3: u32 = 0x13_7018;
    pub const GPCPLL_NDIV_SLOWDOWN: u32 = 0x13_701C;
    pub const GPCPLL_NDIV_SLOWDOWN_DEBUG: u32 = 0x13_28A0;
    pub const ENGINE_HOLD: u32 = 0x02_0290;
    pub const ENGINE_HOLD_STATUS: u32 = 0x02_0294;
}

const PMU: Range<u32> = 0x10_A000..0x10_B000;
const FECS: Range<u32> = 0x40_9000..0x40_A000;
const GPCCS: Range<u32> = 0x41_A000..0x41_B000;

/// `mc_enable` bits whose clearing resets a unit the model keeps state
/// for (nvgpu `mc_enable_*`).
const ENABLE_PFIFO: u32 = 1 << 8;
const ENABLE_PGRAPH: u32 = 1 << 12;
const ENABLE_PWR: u32 = 1 << 13;
/// `mc_intr` bits (nvgpu `mc_intr_*_pending_f`).
const INTR_PFIFO: u32 = 1 << 8;
const INTR_PMU: u32 = 1 << 24;

/// Pending bits of the flush and L2 maintenance registers (bit 0 pending,
/// bit 1 outstanding).
const FLUSH_BUSY: u32 = 3;
/// `fb_mmu_ctrl`: request FIFO empty (bit 15) and its free space.
const MMU_PRI_FIFO_EMPTY: u32 = 1 << 15;
const MMU_PRI_FIFO_FIELDS: u32 = 0xFF << 16 | MMU_PRI_FIFO_EMPTY;
/// `fb_mmu_vpr_info` bit 2: a fetch of the protected-region settings is
/// under way (`NV_FlushFramebufferWithTimeout_85400` waits for it clear).
const VPR_FETCH_PENDING: u32 = 1 << 2;
/// `pri_ringmaster_command` command field, 0 when the command is done.
const RING_COMMAND_FIELD: u32 = 0x3F;
/// `bus_bar1_block`: instance pointer bits 27:0, target 29:28, virtual
/// mode bit 31.
const BAR1_POINTER: u32 = 0x0FFF_FFFF;
const BAR1_TARGET: u32 = 3 << 28;
const BAR1_VIRTUAL: u32 = 1 << 31;
/// Write trigger of `gr_pri_mme_shadow_raw_index` (`[RECALLED]` nvgpu
/// writes the method-init value to the data register, then index | bit 31).
const MME_SHADOW_WRITE: u32 = 1 << 31;

/// The GPC clock PLL (nvgpu `hw_trim_gm20b.h`), which pcv, not nvservices,
/// programs through this window when nvservices asks for the GPU clock (a
/// port of nvgpu's `clk_gm20b.c` in pcv, text `0x44C10`-`0x47E20`, with its
/// `gm20b_readl` traces). What it waits for, all over at once in a model
/// without an analog PLL:
///
/// * `cfg.pll_lock` (bit 17) once the PLL is enabled (bit 0), powered
///   (`iddq`, bit 1, clear) and its lock detector on (bit 4 clear);
/// * `dvfs1.dfs_cal_done` (bit 31) once calibration is enabled (bit 30);
/// * `ndiv_slowdown_debug.pll_dynramp_done_synced` (bit 24) while dynamic
///   ramping is enabled (`ndiv_slowdown` bit 31).
///
/// `cfg3.dfs_testout` (bits 30:24) is the ADC reading the calibration
/// leaves ([`identity::GPCPLL_DFS_TESTOUT`]).
const GPCPLL_LOCK: u32 = 1 << 17;
const GPCPLL_ENABLE: u32 = 1 << 0;
const GPCPLL_IDDQ: u32 = 1 << 1;
const GPCPLL_LOCK_DETECT_OFF: u32 = 1 << 4;
const GPCPLL_CAL_ENABLE: u32 = 1 << 30;
const GPCPLL_CAL_DONE: u32 = 1 << 31;
const GPCPLL_TESTOUT: u32 = 0x7F << 24;
const GPCPLL_DYNRAMP: u32 = 1 << 31;
const GPCPLL_DYNRAMP_DONE: u32 = 1 << 24;

/// The units whose registers the driver programs, by BAR0 range (nvgpu
/// `hw_*_gm20b.h` register placement). Writes there are kept.
const UNITS: [(Range<u32>, &str); 17] = [
    (0x00_0000..0x00_1000, "PMC"),
    (0x00_1000..0x00_2000, "PBUS"),
    (0x00_2000..0x00_4000, "PFIFO"),
    (0x00_9000..0x00_A000, "PTIMER"),
    (0x02_0000..0x02_1000, "PTHERM"),
    (0x02_1000..0x02_2000, "FUSE"),
    (0x02_2000..0x02_3000, "PTOP"),
    (0x04_0000..0x04_2000, "PBDMA"),
    (0x07_0000..0x07_1000, "flush"),
    (0x10_0000..0x10_1000, "PFB"),
    (0x10_4000..0x10_8000, "copy engine"),
    (0x10_A000..0x10_B000, "PMU"),
    (0x12_0000..0x14_0000, "priv ring and clocks"),
    (0x14_0000..0x18_0000, "LTC"),
    (0x18_0000..0x1C_0000, "PERF"),
    (0x40_0000..0x60_0000, "PGRAPH"),
    (0x80_0000..0x80_1000, "CCSR"),
];

/// Configuration registers the driver reads before it writes them, to
/// change some bits and keep the rest: interrupt routing masks,
/// unit enables, clock-gating and idle-filter controls (`0x20200`,
/// `0x20204`, `0x20288`, `0x2028C`), thermal configuration (`0x20130`,
/// `0x202C0`, `0x203C0`, `NV_InitializeGm20bFifoRegisters_8BD80`), the
/// PMU interface enable, PBDMA configuration, the FIFO timeout, the LTC
/// interrupt enables and data-stage configuration, a GPC configuration
/// bit, the CROP debug register of ROP 0 (read to set bit 31 in the
/// broadcast copy, `0x87750`), the MMU debug control (read and written
/// back unchanged), the FE go-idle timeout (saved and restored around the
/// golden-context build). Assumed: they come out of reset as 0. Their
/// reset values are not published; the driver sets every bit it relies
/// on, so only bits it never looks at depend on this.
pub const RESET_ZERO: [u32; 20] = [
    reg::MC_ENABLE,
    reg::MC_INTR_MASK_0,
    reg::MC_INTR_MASK_1,
    0x00_2A04,
    0x02_0130,
    0x02_0200,
    0x02_0204,
    0x02_0288,
    0x02_028C,
    0x02_02C0,
    0x02_03C0,
    0x04_013C,
    0x10_0CC4,
    0x10_A048,
    0x14_0518,
    0x17_E20C,
    0x40_4154,
    0x41_0108,
    0x50_3018,
    // `pbdma_timeout`: read, then written whole (`0x70F80`, nvgpu's
    // read-modify-write with every bit set).
    0x04_012C,
];

/// The GPC clock PLL's and its neighbors' configuration, which pcv reads
/// before it changes it (see [`GPCPLL_LOCK`] and the following): PLL
/// control, coefficients, DVFS and slide controls, VCO selection, output
/// dividers, bypass control, the broadcast DVFS register, the clock
/// slowdown and `use_a` thermal controls, and the RAM SVOP PDP option and
/// its override. Assumed: 0 out of reset, which leaves the PLL disabled
/// and in bypass, and the SVOP PDP option unfused (pcv then programs the
/// override itself, as nvgpu's `gm20b_init_clk_setup_hw` does).
const CLOCK_RESET_ZERO: [u32; 15] = [
    reg::GPCPLL_CFG,
    0x13_7004,
    0x13_700C,
    0x13_7010,
    reg::GPCPLL_DVFS1,
    reg::GPCPLL_CFG3,
    reg::GPCPLL_NDIV_SLOWDOWN,
    0x13_7100,
    0x13_7250,
    0x13_7340,
    0x13_2820,
    0x02_0160,
    0x02_0798,
    0x02_1944,
    0x02_1948,
];

/// An LTC status register read per LTC (`0x2000` apart) that mirrors the
/// broadcast register the operation was started through, with the
/// operation's busy bits clear: (unicast offset in LTC 0, broadcast
/// offset, busy bits). L2 flush (`NV_FlushLtcSlices_84A80`), clean and
/// invalidate (`0x84CF0`), comptag maintenance (`0x84930`, per slice).
const LTC_OPERATIONS: [(u32, u32, u32); 4] = [
    (0x14_0214, 0x17_E214, 1),
    (0x14_02A0, 0x17_E2A0, 1),
    (0x14_02A4, 0x17_E2A4, 1),
    (0x14_046C, 0x17_E26C, 7),
];
const LTC_STRIDE: u32 = 0x2000;
const LTS_STRIDE: u32 = 0x200;

pub struct Gm20b {
    shared: Arc<Shared>,
    worker: Option<JoinHandle<()>>,
}

struct Shared {
    state: Mutex<State>,
    /// The worker waits here for work.
    wake: Condvar,
    /// Asks the worker to stop the channel it executes between methods.
    attention: AtomicBool,
    bus: Box<dyn BusMemory>,
    syncpoints: Arc<Syncpoints>,
    notes: Arc<Notes>,
    stall: InterruptLine,
    nonstall: InterruptLine,
    nanoseconds: Arc<dyn Fn() -> u64 + Send + Sync>,
    engines: EngineFactory,
}

struct State {
    /// What the driver wrote, by BAR0 offset.
    registers: HashMap<u32, u32>,
    fifo: Fifo,
    pmu: Pmu,
    fecs: Fecs,
    gpccs: Gpccs,
    /// The method-init list, in the order loaded, and where each index is
    /// in it.
    method_init: Vec<(u32, u32)>,
    method_init_at: HashMap<u32, usize>,
    /// The channel whose wait keeps the PBDMA (a wait without the switch
    /// bit).
    holding: Option<usize>,
    /// Why the GPU stopped.
    failure: Option<String>,
    kicked: bool,
    stop: bool,
    lines: (bool, bool),
}

impl Gm20b {
    /// `memory` is memory as the GPU's SMMU client reaches it.
    pub fn new(
        memory: DeviceMemory,
        syncpoints: Arc<Syncpoints>,
        [stall, nonstall]: [InterruptLine; 2],
        clock: Clock,
        engines: EngineFactory,
        notes: Arc<Notes>,
    ) -> Self {
        Self::with_bus(
            Box::new(SmmuBus(memory)),
            syncpoints,
            [stall, nonstall],
            clock,
            engines,
            notes,
        )
    }

    fn with_bus(
        bus: Box<dyn BusMemory>,
        syncpoints: Arc<Syncpoints>,
        [stall, nonstall]: [InterruptLine; 2],
        clock: Clock,
        engines: EngineFactory,
        notes: Arc<Notes>,
    ) -> Self {
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                registers: HashMap::new(),
                fifo: Fifo::default(),
                pmu: Pmu::new(),
                fecs: Fecs::new(),
                gpccs: Gpccs::new(),
                method_init: Vec::new(),
                method_init_at: HashMap::new(),
                holding: None,
                failure: None,
                kicked: false,
                stop: false,
                lines: (false, false),
            }),
            wake: Condvar::new(),
            attention: AtomicBool::new(false),
            bus,
            syncpoints,
            notes,
            stall,
            nonstall,
            nanoseconds: Arc::new(move || clock().as_nanos() as u64),
            engines,
        });
        let worker = {
            let shared = Arc::clone(&shared);
            thread::Builder::new()
                .name("GM20B host".into())
                .spawn(move || shared.work())
                .expect("the GPU host thread starts")
        };
        Self {
            shared,
            worker: Some(worker),
        }
    }

    pub fn bar0(&self) -> Box<dyn MmioDevice> {
        Box::new(Bar0(Arc::clone(&self.shared)))
    }

    pub fn bar1(&self) -> Box<dyn MmioDevice> {
        Box::new(Bar1(Arc::clone(&self.shared)))
    }

    /// One line per bound channel: ID, enable and runlist state, and where
    /// its command stream stands (for `--verbose` run reports).
    pub fn channels(&self) -> Vec<String> {
        let state = self.shared.state.lock().unwrap();
        let fifo = &state.fifo;
        let bus = self.shared.bus.as_ref();
        let mut lines: Vec<String> = fifo
            .channels
            .iter()
            .enumerate()
            .filter_map(|(chid, ccsr)| {
                let instance = ccsr.bound()?;
                let context = if fifo.executing == Some(chid) {
                    "executing now".into()
                } else {
                    fifo.contexts
                        .get(&instance)
                        .map_or_else(|| "never run".into(), |context| context.describe(bus))
                };
                Some(format!(
                    "channel {chid}: {}, {}runlist{}: {context}",
                    if ccsr.enabled { "enabled" } else { "disabled" },
                    if fifo.runlist.contains(&chid) {
                        "on the "
                    } else {
                        "off the "
                    },
                    if state.holding == Some(chid) {
                        ", holding the PBDMA"
                    } else {
                        ""
                    },
                ))
            })
            .collect();
        let (stall, nonstall) = state.interrupts();
        lines.push(format!(
            "{}; mc interrupts stall {stall:#x} non-stall {nonstall:#x}, lines (stall, non-stall) {:?}",
            fifo.describe(),
            state.lines
        ));
        if let Some(failure) = &state.failure {
            lines.push(format!("stopped: {failure}"));
        }
        lines
    }
}

impl Drop for Gm20b {
    fn drop(&mut self) {
        self.shared.state.lock().unwrap().stop = true;
        self.shared.wake.notify_all();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

struct Bar0(Arc<Shared>);
struct Bar1(Arc<Shared>);

impl MmioDevice for Bar0 {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        let offset = word(offset, size)? as u32;
        let mut state = self.0.usable()?;
        let value = state.read(&self.0, offset);
        self.0.refuse(value).map(u64::from)
    }

    fn write(&self, _core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        let offset = word(offset, size)? as u32;
        let mut state = self.0.usable()?;
        let done = state.write(&self.0, offset, value as u32);
        self.0.update_lines(&mut state);
        drop(state);
        self.0.refuse(done)
    }
}

impl MmioDevice for Bar1 {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        let mut buffer = [0; 8];
        let bytes = &mut buffer[..usize::from(size)];
        let done = self
            .0
            .bar1()
            .and_then(|space| space.read(self.0.bus.as_ref(), offset, bytes));
        self.0.refuse(done)?;
        Some(u64::from_le_bytes(buffer))
    }

    fn write(&self, _core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        let bytes = &value.to_le_bytes()[..usize::from(size)];
        let done = self
            .0
            .bar1()
            .and_then(|space| space.write(self.0.bus.as_ref(), offset, bytes));
        self.0.refuse(done)?;
        // This may have been a channel's GP_PUT: the host looks.
        self.0.kick();
        Some(())
    }
}

impl Shared {
    /// The state, unless the GPU has stopped.
    fn usable(&self) -> Option<MutexGuard<'_, State>> {
        let state = self.state.lock().unwrap();
        if let Some(failure) = &state.failure {
            self.notes.add(format!("GM20B stopped earlier: {failure}"));
            return None;
        }
        Some(state)
    }

    fn refuse<T>(&self, outcome: Result<T, Unmodeled>) -> Option<T> {
        outcome
            .map_err(|Unmodeled(what)| self.notes.add(format!("GM20B: {what}")))
            .ok()
    }

    /// The BAR1 address space, from the instance block `bus_bar1_block`
    /// names.
    fn bar1(&self) -> Result<AddressSpace, Unmodeled> {
        let block = {
            let state = self
                .usable()
                .ok_or_else(|| Unmodeled("BAR1 access after the GPU stopped".into()))?;
            state.registers.get(&reg::BAR1_BLOCK).copied()
        };
        let block = block
            .filter(|block| block & BAR1_VIRTUAL != 0 && block & BAR1_TARGET == 0)
            .ok_or_else(|| {
                Unmodeled(format!(
                    "BAR1 access while bus_bar1_block is {block:x?} (only a virtual BAR1 \
                     with its instance block in video memory is modeled)"
                ))
            })?;
        AddressSpace::from_instance(self.bus.as_ref(), u64::from(block & BAR1_POINTER) << 12)
    }

    fn kick(&self) {
        self.state.lock().unwrap().kicked = true;
        self.wake.notify_one();
    }

    fn update_lines(&self, state: &mut State) {
        let (stall, nonstall) = state.interrupts();
        let lines = (
            stall != 0 && state.stored(reg::MC_INTR_EN_0) & 1 != 0,
            nonstall != 0 && state.stored(reg::MC_INTR_EN_1) & 1 != 0,
        );
        if lines.0 != state.lines.0 {
            (self.stall)(lines.0);
        }
        if lines.1 != state.lines.1 {
            (self.nonstall)(lines.1);
        }
        state.lines = lines;
    }

    fn fail(&self, state: &mut State, Unmodeled(what): Unmodeled) {
        self.notes.add(format!("GM20B: {what}"));
        state.failure = Some(what);
    }

    /// The host's thread: runs channels whenever there may be work.
    fn work(&self) {
        let mut state = self.state.lock().unwrap();
        loop {
            if state.stop {
                return;
            }
            state.kicked = false;
            drop(state);
            let waiting = self.service();
            state = self.state.lock().unwrap();
            if state.stop || state.kicked {
                continue;
            }
            state = if waiting {
                self.wake.wait_timeout(state, WAIT_POLL).unwrap().0
            } else {
                self.wake.wait(state).unwrap()
            };
        }
    }

    /// Gives every runnable channel on the runlist one turn, in runlist
    /// order. Whether one of them is left waiting.
    fn service(&self) -> bool {
        let mut waiting = false;
        let mut done: Vec<usize> = Vec::new();
        loop {
            let mut state = self.state.lock().unwrap();
            if state.failure.is_some() || state.stop {
                return false;
            }
            let fifo = &state.fifo;
            let next = match state.holding {
                Some(chid) if fifo.runnable(chid) => (!done.contains(&chid)).then_some(chid),
                _ => fifo
                    .runlist
                    .iter()
                    .copied()
                    .find(|&chid| !done.contains(&chid) && fifo.runnable(chid)),
            };
            let Some(chid) = next else {
                return waiting;
            };
            done.push(chid);
            let instance = fifo.channels[chid]
                .bound()
                .expect("a runnable channel is bound");
            let context = match state.fifo.contexts.remove(&instance) {
                Some(context) => Ok(context),
                None => self.new_context(&state, instance),
            };
            let mut context = match context {
                Ok(context) => context,
                Err(Unmodeled(what)) => {
                    let what = format!("channel {chid}: {what}");
                    self.fail(&mut state, Unmodeled(what));
                    return false;
                }
            };
            state.fifo.executing = Some(chid);
            self.attention.store(false, Ordering::SeqCst);
            drop(state);
            let outcome = context.run(&Env {
                bus: self.bus.as_ref(),
                syncpoints: &self.syncpoints,
                attention: &self.attention,
                now: self.nanoseconds.as_ref(),
                survey: self.notes.survey(),
                note: &|what| self.notes.add(format!("GM20B: channel {chid}: {what}")),
            });
            let mut state = self.state.lock().unwrap();
            if let Ok(outcome) = &outcome {
                context.last = Some(*outcome);
            }
            state.fifo.contexts.insert(instance, context);
            state.fifo.left(chid);
            if state.holding == Some(chid) {
                state.holding = None;
            }
            match outcome {
                Ok(Outcome::Idle) => {}
                Ok(Outcome::Blocked(wait)) => {
                    waiting = true;
                    if !wait.switch() {
                        state.holding = Some(chid);
                    }
                }
                Ok(Outcome::Interrupted) => state.kicked = true,
                Err(Unmodeled(what)) => {
                    let what = format!("channel {chid}: {what}");
                    self.fail(&mut state, Unmodeled(what));
                }
            }
            self.update_lines(&mut state);
        }
    }

    fn new_context(&self, state: &State, instance: u64) -> Result<Context, Unmodeled> {
        let nanoseconds = Arc::clone(&self.nanoseconds);
        let engines = (self.engines)(&state.method_init, Box::new(move || nanoseconds()))?;
        Context::load(self.bus.as_ref(), instance, engines)
    }
}

/// The unit a BAR0 offset belongs to.
fn unit(offset: u32) -> Option<&'static str> {
    UNITS
        .iter()
        .find(|(range, _)| range.contains(&offset))
        .map(|&(_, name)| name)
}

fn identity(offset: u32) -> Option<u32> {
    if offset == identity::SM_ARCH_OFFSET {
        return Some(identity::GPC0_TPC0_SM_ARCH);
    }
    let entry = offset.checked_sub(identity::DEVICE_INFO_BASE)? / 4;
    if offset.is_multiple_of(4) && entry < identity::DEVICE_INFO_ENTRIES {
        return Some(
            identity::DEVICE_INFO
                .get(entry as usize)
                .copied()
                .unwrap_or(0),
        );
    }
    None
}

fn read_only(offset: u32) -> Option<u32> {
    identity::READ_ONLY
        .iter()
        .find(|&&(at, _)| at == offset)
        .map(|&(_, value)| value)
        .or_else(|| identity(offset))
}

impl State {
    /// What a register holds: written, or its known reset value.
    fn get(&self, offset: u32) -> Option<u32> {
        self.registers.get(&offset).copied().or_else(|| {
            (RESET_ZERO.contains(&offset) || CLOCK_RESET_ZERO.contains(&offset)).then_some(0)
        })
    }

    /// For registers whose reset value has been established.
    fn stored(&self, offset: u32) -> u32 {
        self.get(offset).unwrap_or(0)
    }

    /// The stall and non-stall interrupt words.
    fn interrupts(&self) -> (u32, u32) {
        let fifo = self.fifo.intr();
        let mut stall = 0;
        let mut nonstall = 0;
        if fifo & self.fifo.intr_en_0 != 0 {
            stall |= INTR_PFIFO;
        }
        if fifo & self.fifo.intr_en_1 != 0 {
            nonstall |= INTR_PFIFO;
        }
        if self.pmu.interrupt() {
            stall |= INTR_PMU;
        }
        (
            stall & self.stored(reg::MC_INTR_MASK_0),
            nonstall & self.stored(reg::MC_INTR_MASK_1),
        )
    }

    fn read(&mut self, shared: &Shared, offset: u32) -> Result<u32, Unmodeled> {
        let at = |range: &Range<u32>| range.contains(&offset).then(|| offset - range.start);
        let handled = if let Some(at) = at(&PMU) {
            self.pmu.read(at)
        } else if let Some(at) = at(&FECS) {
            self.fecs.read(at)
        } else if let Some(at) = at(&GPCCS) {
            self.gpccs.read(at)
        } else {
            self.fifo.read(offset)
        };
        if let Some(value) = handled {
            return value;
        }
        if let Some(value) = read_only(offset) {
            return Ok(value);
        }
        let operation =
            |state: &Self, busy: u32| state.registers.get(&offset).map_or(0, |v| v & !busy);
        Ok(match offset {
            reg::MC_INTR_0 => self.interrupts().0,
            reg::MC_INTR_1 => self.interrupts().1,
            // The GPU timer counts nanoseconds (`[RECALLED]` PTIMER).
            reg::TIMER_TIME_0 => (shared.nanoseconds)() as u32,
            reg::TIMER_TIME_1 => ((shared.nanoseconds)() >> 32) as u32,
            reg::FB_FLUSH
            | reg::L2_SYSTEM_INVALIDATE
            | reg::L2_CLEAN_COMPTAGS
            | reg::L2_FLUSH_DIRTY => operation(self, FLUSH_BUSY),
            reg::MMU_CTRL => {
                self.stored(reg::MMU_CTRL) & !MMU_PRI_FIFO_FIELDS
                    | identity::MMU_PRI_FIFO_DEPTH << 16
                    | MMU_PRI_FIFO_EMPTY
            }
            reg::MMU_VPR_INFO => operation(self, VPR_FETCH_PENDING),
            reg::RING_COMMAND => operation(self, RING_COMMAND_FIELD),
            // The model never raises a priv ring error.
            reg::RING_INTR_STATUS_0 => 0,
            // Graphics busy while the host executes a channel; the
            // front-end activity (bit 2, polled during the golden-context
            // bundle replay) is over at once.
            reg::GR_STATUS => u32::from(self.fifo.executing.is_some()),
            reg::GR_ACTIVITY => 0,
            // Read after every nvsched runlist swap
            // (`NV_ReplaceRunlistEligibilityAndResubmit_729F0`): when the
            // graphics engine's bit is set in the first, the driver waits
            // up to its timeout for the bit to clear in the second. Neither
            // register is in nvgpu's gm20b headers; they sit among the
            // thermal block's engine clock and power gating controls.
            // Assumed: nothing for the driver to wait out, as the model
            // gates no engine (no PMU power gating ever starts, falcon.rs).
            reg::ENGINE_HOLD | reg::ENGINE_HOLD_STATUS => 0,
            reg::GPCPLL_CFG => {
                let cfg = self.stored(reg::GPCPLL_CFG);
                let locked =
                    cfg & GPCPLL_ENABLE != 0 && cfg & (GPCPLL_IDDQ | GPCPLL_LOCK_DETECT_OFF) == 0;
                cfg & !GPCPLL_LOCK | if locked { GPCPLL_LOCK } else { 0 }
            }
            reg::GPCPLL_DVFS1 => {
                let dvfs1 = self.stored(reg::GPCPLL_DVFS1) & !GPCPLL_CAL_DONE;
                dvfs1
                    | if dvfs1 & GPCPLL_CAL_ENABLE != 0 {
                        GPCPLL_CAL_DONE
                    } else {
                        0
                    }
            }
            reg::GPCPLL_CFG3 => {
                self.stored(reg::GPCPLL_CFG3) & !GPCPLL_TESTOUT | identity::GPCPLL_DFS_TESTOUT << 24
            }
            reg::GPCPLL_NDIV_SLOWDOWN_DEBUG => {
                if self.stored(reg::GPCPLL_NDIV_SLOWDOWN) & GPCPLL_DYNRAMP != 0 {
                    GPCPLL_DYNRAMP_DONE
                } else {
                    0
                }
            }
            _ => {
                if let Some(value) = self.ltc_operation(offset) {
                    return Ok(value);
                }
                return self.get(offset).ok_or_else(|| {
                    Unmodeled(format!(
                        "read of BAR0 {offset:#08x} ({}), which nothing has written and whose \
                         value is not known",
                        unit(offset).unwrap_or("no known unit")
                    ))
                });
            }
        })
    }

    fn ltc_operation(&self, offset: u32) -> Option<u32> {
        let ltcs = identity::TOP_NUM_LTCS;
        let slices = identity::TOP_SLICES_PER_LTC;
        LTC_OPERATIONS
            .iter()
            .find(|&&(unicast, _, _)| {
                (0..ltcs).any(|ltc| {
                    (0..slices)
                        .any(|slice| offset == unicast + ltc * LTC_STRIDE + slice * LTS_STRIDE)
                })
            })
            .map(|&(_, broadcast, busy)| self.stored(broadcast) & !busy)
    }

    fn write(&mut self, shared: &Shared, offset: u32, value: u32) -> Result<(), Unmodeled> {
        if read_only(offset).is_some() {
            return Err(Unmodeled(format!(
                "write of {value:#x} to the read-only BAR0 register {offset:#08x}"
            )));
        }
        unit(offset).ok_or_else(|| {
            Unmodeled(format!(
                "write of {value:#x} to BAR0 {offset:#08x}, outside the units the model knows"
            ))
        })?;
        let at = |range: &Range<u32>| range.contains(&offset).then(|| offset - range.start);
        let handled = if let Some(at) = at(&PMU) {
            self.pmu.write(at, value, &mut self.fecs)?
        } else if let Some(at) = at(&FECS) {
            self.fecs.write(at, value, &self.gpccs)?
        } else if let Some(at) = at(&GPCCS) {
            self.gpccs.write(at, value, &mut self.fecs)?
        } else if let Some(done) = self.fifo.write(offset, value, shared.bus.as_ref()) {
            done?;
            self.wake_host(shared);
            true
        } else {
            false
        };
        if handled {
            return Ok(());
        }
        match offset {
            reg::MC_ENABLE => {
                let cleared = self.stored(reg::MC_ENABLE) & !value;
                if cleared & ENABLE_PFIFO != 0 {
                    self.fifo.reset();
                    self.holding = None;
                    self.wake_host(shared);
                }
                if cleared & ENABLE_PGRAPH != 0 {
                    self.fecs.reset();
                    self.gpccs.reset();
                }
                if cleared & ENABLE_PWR != 0 {
                    self.pmu.reset();
                }
            }
            reg::MME_SHADOW_RAW_INDEX if value & MME_SHADOW_WRITE != 0 => {
                let index = value & !MME_SHADOW_WRITE;
                let entry = (index, self.stored(reg::MME_SHADOW_RAW_DATA));
                match self.method_init_at.get(&index) {
                    Some(&at) => self.method_init[at] = entry,
                    None => {
                        self.method_init_at.insert(index, self.method_init.len());
                        self.method_init.push(entry);
                    }
                }
            }
            reg::MC_INTR_0 | reg::MC_INTR_1 | reg::GPCPLL_NDIV_SLOWDOWN_DEBUG => {
                return Err(Unmodeled(format!(
                    "write of {value:#x} to the status register {offset:#x}"
                )));
            }
            _ => {}
        }
        self.registers.insert(offset, value);
        Ok(())
    }

    /// Something the host acts on changed: the worker leaves the channel
    /// it executes and looks again.
    fn wake_host(&mut self, shared: &Shared) {
        if self.fifo.executing.is_some() {
            shared.attention.store(true, Ordering::SeqCst);
        }
        self.kicked = true;
        shared.wake.notify_one();
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::atomic::AtomicBool, time::Instant};

    use super::{mmu::tests::TestBus, *};
    use crate::maxwell::GpuMemory;

    const IOVA: u64 = mmu::SMMU_BIT;

    /// Shares a [`TestBus`] between the device and the test.
    struct SharedBus(Arc<TestBus>);

    impl BusMemory for SharedBus {
        fn read(&self, address: u64, buffer: &mut [u8]) -> Option<()> {
            self.0.read(address, buffer)
        }

        fn write(&self, address: u64, data: &[u8]) -> Option<()> {
            self.0.write(address, data)
        }
    }

    struct Bench {
        gpu: Gm20b,
        bar0: Box<dyn MmioDevice>,
        bar1: Box<dyn MmioDevice>,
        bus: Arc<TestBus>,
        syncpoints: Arc<Syncpoints>,
        notes: Arc<Notes>,
        nonstall: Arc<AtomicBool>,
    }

    struct NoEngines;

    impl Engines for NoEngines {
        fn call(&mut self, _: u8, method: u32, _: u32, _: &dyn GpuMemory) -> Result<(), Unmodeled> {
            Err(Unmodeled(format!("engine method {method:#x}")))
        }
    }

    impl Bench {
        fn new() -> Self {
            let bus = Arc::new(TestBus::default());
            let syncpoints: Arc<Syncpoints> = Arc::default();
            let notes: Arc<Notes> = Arc::default();
            let nonstall = Arc::new(AtomicBool::new(false));
            let line = Arc::clone(&nonstall);
            let gpu = Gm20b::with_bus(
                Box::new(SharedBus(Arc::clone(&bus))),
                Arc::clone(&syncpoints),
                [
                    Box::new(|_| {}),
                    Box::new(move |level| line.store(level, Ordering::SeqCst)),
                ],
                Box::new(Duration::default),
                Box::new(|_, _| Ok(Box::new(NoEngines))),
                Arc::clone(&notes),
            );
            Self {
                bar0: gpu.bar0(),
                bar1: gpu.bar1(),
                gpu,
                bus,
                syncpoints,
                notes,
                nonstall,
            }
        }

        fn read(&self, offset: u32) -> Option<u32> {
            self.bar0
                .read(0, offset.into(), 4)
                .map(|value| value as u32)
        }

        fn write(&self, offset: u32, value: u32) {
            self.bar0
                .write(0, offset.into(), 4, value.into())
                .unwrap_or_else(|| panic!("{offset:#x}: {:?}", self.notes.take()));
        }

        /// The BAR1 address space as `NV_InitializeGmmuAddressSpaces_73E80`
        /// builds it: 64 KiB big pages, 128 MiB; VA 0x10000 maps the
        /// page at IOVA 0x70000.
        fn bar1_space(&self) {
            let instance = IOVA | 0x1000;
            let directory = IOVA | 0x2000;
            let table = IOVA | 0x4000;
            self.bus
                .instance(instance, directory, mmu::BigPage::K64, 0x800_0000);
            self.bus.big_table(directory, table);
            self.bus.page(table + 8, IOVA | 0x7_0000);
            self.write(reg::BAR1_BLOCK, ((instance >> 12) as u32) | BAR1_VIRTUAL);
        }
    }

    #[test]
    fn the_power_on_read_back_probe_passes_through_bar1() {
        let bench = Bench::new();
        // Before the block is bound BAR1 refuses.
        assert_eq!(bench.bar1.read(0, 0x1_0000, 4), None);
        bench.bar1_space();
        // NV_InitializeSchedulerMemory_70FD0: the CPU writes 0x33 into
        // USERD through its own mapping and reads it through BAR1...
        bench.bus.put(IOVA | (0x7_0000 + 0x88), 0x33);
        assert_eq!(bench.bar1.read(0, 0x1_0088, 4), Some(0x33));
        // ...then writes 0x55 through BAR1 and reads it back.
        bench.bar1.write(0, 0x1_0088, 4, 0x55).unwrap();
        assert_eq!(bench.bar1.read(0, 0x1_0088, 4), Some(0x55));
        assert_eq!(bench.bus.get(IOVA | (0x7_0000 + 0x88)), 0x55);
        assert_eq!(bench.bar1.read(0, 0x1_008A, 2), Some(0));
        // Outside the mapping.
        assert_eq!(bench.bar1.read(0, 0x2_0000, 4), None);
    }

    #[test]
    fn power_on_handshakes() {
        let bench = Bench::new();
        assert_eq!(bench.read(0), Some(identity::BOOT_0));
        // Read-modify-write of PMC_ENABLE from reset.
        let enable = bench.read(reg::MC_ENABLE).unwrap();
        bench.write(reg::MC_ENABLE, enable | 0x2000_0120);
        assert_eq!(bench.read(reg::MC_ENABLE), Some(0x2000_0120));
        // Flushes are done at once.
        bench.write(reg::FB_FLUSH, 1);
        assert_eq!(bench.read(reg::FB_FLUSH).unwrap() & 3, 0);
        // The TLB invalidate handshake.
        let ctrl = bench.read(reg::MMU_CTRL).unwrap();
        assert_ne!(ctrl >> 16 & 0xFF, 0);
        assert_ne!(ctrl & MMU_PRI_FIFO_EMPTY, 0);
        bench.write(reg::MMU_CTRL, ctrl | 0x800);
        assert_eq!(bench.read(reg::MMU_CTRL).unwrap() & 0x800, 0x800);
        // Priv ring command.
        bench.write(reg::RING_COMMAND, 4);
        assert_eq!(bench.read(reg::RING_COMMAND).unwrap() & 0x3F, 0);
        // The LTC flush polled per LTC.
        bench.write(0x17_E214, 1);
        assert_eq!(bench.read(0x14_0214), Some(0));
        assert_eq!(bench.read(0x14_2214), Some(0));
        // Engine table: graphics on runlist 0, interrupt 12.
        let info = bench.read(identity::DEVICE_INFO_BASE + 4).unwrap();
        assert_eq!((info & 3, info >> 21 & 0xF, info >> 15 & 0x1F), (2, 0, 12));
        assert_eq!(bench.read(identity::DEVICE_INFO_BASE + 8), Some(0));
        // Unwritten registers without a known value refuse.
        assert_eq!(bench.read(0x40_1234), None);
        bench.write(0x40_1234, 7);
        assert_eq!(bench.read(0x40_1234), Some(7));
        assert_eq!(bench.read(0x0F_0000), None);
    }

    /// pcv's GPC PLL bring-up (text `0x44C10`): calibration, lock, slide.
    #[test]
    fn the_gpc_pll_calibrates_locks_and_slides() {
        let bench = Bench::new();
        let dvfs1 = bench.read(reg::GPCPLL_DVFS1).unwrap();
        assert_eq!(dvfs1 & GPCPLL_CAL_DONE, 0);
        bench.write(reg::GPCPLL_DVFS1, dvfs1 | GPCPLL_CAL_ENABLE);
        assert_ne!(bench.read(reg::GPCPLL_DVFS1).unwrap() & GPCPLL_CAL_DONE, 0);
        let testout = bench.read(reg::GPCPLL_CFG3).unwrap() >> 24 & 0x7F;
        assert_eq!(testout, identity::GPCPLL_DFS_TESTOUT);
        // Enabled with the lock detector off: no lock; on: locked.
        bench.write(reg::GPCPLL_CFG, GPCPLL_ENABLE | GPCPLL_LOCK_DETECT_OFF);
        assert_eq!(bench.read(reg::GPCPLL_CFG).unwrap() & GPCPLL_LOCK, 0);
        bench.write(reg::GPCPLL_CFG, GPCPLL_ENABLE);
        assert_ne!(bench.read(reg::GPCPLL_CFG).unwrap() & GPCPLL_LOCK, 0);
        bench.write(reg::GPCPLL_CFG, GPCPLL_ENABLE | GPCPLL_IDDQ);
        assert_eq!(bench.read(reg::GPCPLL_CFG).unwrap() & GPCPLL_LOCK, 0);
        // The dynamic ramp is done while enabled.
        assert_eq!(bench.read(reg::GPCPLL_NDIV_SLOWDOWN_DEBUG), Some(0));
        bench.write(reg::GPCPLL_NDIV_SLOWDOWN, GPCPLL_DYNRAMP | 1 << 22);
        assert_eq!(
            bench.read(reg::GPCPLL_NDIV_SLOWDOWN_DEBUG),
            Some(GPCPLL_DYNRAMP_DONE)
        );
    }

    #[test]
    fn the_runlist_swap_raises_the_nonstall_line() {
        let bench = Bench::new();
        bench.write(0x2528, 0x4000_0000);
        bench.write(reg::MC_INTR_MASK_1, 0x100);
        bench.write(reg::MC_INTR_EN_1, 1);
        assert!(!bench.nonstall.load(Ordering::SeqCst));
        // NV_ReplaceRunlistEligibilityAndResubmit_729F0: disable, preempt.
        bench.write(0x2630, 1);
        bench.write(0x2638, 1);
        assert!(bench.nonstall.load(Ordering::SeqCst));
        assert_eq!(bench.read(reg::MC_INTR_1), Some(0x100));
        bench.write(reg::MC_INTR_EN_1, 0);
        assert!(!bench.nonstall.load(Ordering::SeqCst));
        // Acknowledged and re-enabled: still there for the second wait.
        bench.write(0x2A00, 1);
        bench.write(reg::MC_INTR_EN_1, 1);
        assert!(bench.nonstall.load(Ordering::SeqCst));
        // Runlist resubmitted and enabled: gone.
        bench.write(0x2630, 0);
        assert!(!bench.nonstall.load(Ordering::SeqCst));
    }

    /// A channel as the driver brings one up, kicked by a GP_PUT write
    /// through BAR1; the host thread increments the fence syncpoint.
    #[test]
    fn a_submission_through_bar1_signals_its_fence() {
        let bench = Bench::new();
        bench.bar1_space();
        let userd = IOVA | 0x7_0000;
        let instance = IOVA | 0x8_0000;
        let directory = IOVA | 0x9_0000;
        let table = IOVA | 0xA_0000;
        bench
            .bus
            .instance(instance, directory, mmu::BigPage::K128, 1 << 37);
        bench.bus.small_table(directory, table);
        // Ring at VA 0x1000 (16 entries), pushbuffer at VA 0x2000.
        bench.bus.page(table + 8, IOVA | 0xB_1000);
        bench.bus.page(table + 16, IOVA | 0xB_2000);
        bench.bus.put(instance + 0x08, userd as u32);
        bench.bus.put(instance + 0x0C, (userd >> 32) as u32);
        bench.bus.put(instance + 0x48, 0x1000);
        bench.bus.put(instance + 0x4C, 4 << 16);
        let fence = [0x2001_001C, 0, 0x2001_001D, 0x18 << 8 | 1];
        for (index, word) in fence.iter().enumerate() {
            bench.bus.put(IOVA | (0xB_2000 + index as u64 * 4), *word);
        }
        bench.bus.put(IOVA | 0xB_1000, 0x2000);
        bench.bus.put(IOVA | 0xB_1004, 4 << 10);
        // Runlist with channel 0, then CCSR bind and enable.
        bench.bus.put(IOVA | 0xC_0000, 0);
        bench.write(0x2270, ((IOVA | 0xC_0000) >> 12) as u32);
        bench.write(0x2274, 1);
        bench.write(0x80_0000, ((instance >> 12) as u32) | 1 << 31);
        bench.write(0x80_0004, 0x400);
        // GP_PUT through BAR1.
        bench.bar1.write(0, 0x1_008C, 4, 1).unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        while bench.syncpoints.get(0x18) != Some(1) {
            assert!(Instant::now() < deadline, "{:?}", bench.notes.take());
            thread::sleep(Duration::from_millis(1));
        }
        while bench.bar1.read(0, 0x1_0088, 4) != Some(1) {
            assert!(Instant::now() < deadline);
            thread::sleep(Duration::from_millis(1));
        }
        // An engine method nothing models stops the GPU, loudly.
        bench.bus.put(IOVA | (0xB_2000 + 0x40), 0x2001_0040);
        bench.bus.put(IOVA | (0xB_2000 + 0x44), 0);
        bench.bus.put(IOVA | 0xB_1008, 0x2040);
        bench.bus.put(IOVA | 0xB_100C, 2 << 10);
        bench.bar1.write(0, 0x1_008C, 4, 2).unwrap();
        while bench.gpu.shared.state.lock().unwrap().failure.is_none() {
            assert!(Instant::now() < deadline);
            thread::sleep(Duration::from_millis(1));
        }
        assert!(
            bench
                .notes
                .take()
                .iter()
                .any(|note| note.contains("engine method 0x40"))
        );
        assert_eq!(bench.read(0), None);
    }
}
