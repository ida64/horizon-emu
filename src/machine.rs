//! The emulated console: DRAM holding the kernel image, the secure monitor,
//! the devices the kernel drives, and one host thread per started core.

use std::{
    collections::{BTreeMap, HashMap},
    fmt, io,
    panic::{self, AssertUnwindSafe},
    sync::{
        Arc, Condvar, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

use crate::{
    arm64::{DataAbort, LoadStore, SysReg, SysRegAccess, TranslationRegime, exception_class},
    boot::{self, BootParams, CORE_COUNT, CoreEntry},
    bus::{Access, Bus, MmioDevice},
    gic::Gic,
    hvf::{
        Exit, HvError, Vcpu, VcpuHandle, Vm,
        debug::{BreakpointError, DebugExit, GuestDebug},
    },
    image::{KernelImage, PAGE_SIZE},
    ipc,
    keys::KeySet,
    memory::GuestRam,
    pmu::Pmu,
    secmon::{self, SecureMonitor, SmcEffect, Unimplemented},
    soc::{
        Screen, Soc,
        dc::NoFrame,
        emmc::{EmmcImages, ImageError},
        mc::{self, DeviceMemory},
        sdcard::SdCardImage,
    },
    svc,
    timer::{PHYSICAL_TIMER_INTID, TimerRegister, Timers},
    uart::{self, Uart},
};

/// Interrupt controller distributor (`GICD_*`).
const GIC_DISTRIBUTOR_BASE: u64 = 0x5004_1000;
/// Interrupt controller CPU interface (`GICC_*`); `GICC_DIR` is in its second page.
const GIC_CPU_INTERFACE_BASE: u64 = 0x5004_2000;
const GIC_DISTRIBUTOR_SIZE: u64 = 0x1000;
const GIC_CPU_INTERFACE_SIZE: u64 = 0x2000;

#[derive(Debug)]
pub enum MachineError {
    Ram(io::Error),
    /// The image's load range is not inside DRAM.
    ImageOutsideRam,
    Emmc(ImageError),
    Hypervisor(HvError),
}

impl fmt::Display for MachineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ram(error) => write!(f, "cannot allocate guest DRAM: {error}"),
            Self::ImageOutsideRam => {
                write!(f, "kernel image does not fit in DRAM at its load address")
            }
            Self::Emmc(error) => write!(f, "eMMC: {error}"),
            Self::Hypervisor(error) => write!(f, "hypervisor: {error}"),
        }
    }
}

impl std::error::Error for MachineError {}

impl From<HvError> for MachineError {
    fn from(error: HvError) -> Self {
        Self::Hypervisor(error)
    }
}

/// Offset of the "synchronous exception from a lower EL using AArch64"
/// vector, where every supervisor call enters the kernel.
const LOWER_EL_SYNC_VECTOR: u64 = 0x400;

/// Image address of the `ERET` through which a 64-bit supervisor call
/// returns to its caller: the last instruction of the kernel's dispatcher
/// (0x800c5114, which the lower-EL synchronous handler branches to for
/// exception class `SVC64`). Just before it the dispatcher has restored the
/// caller's `ELR_EL1` and stack pointer; `X0`-`X7` are as the call's handler
/// (called at 0x800c51cc) left them, and `SP` is back at the top of the
/// thread's kernel stack. Calls that do not return (`ExitThread`,
/// `ExitProcess`) and calls the dispatcher refuses, which leave through the
/// user exception path and its own `ERET` at 0x800c5320, never get here.
const SVC_RETURN: u64 = 0x800C_5488;
const ERET: u32 = 0xD69F_03E0;

/// At most this much of the memory a supervisor call names is captured.
const CALL_DATA_LIMIT: u64 = 0x1000;

/// Address of the thread-local region of the thread a core runs, which the
/// kernel sets when it switches threads (image 0x8009e590).
const TPIDRRO_EL0: SysReg = SysReg::new(3, 3, 13, 0, 3);
const SP_EL1: SysReg = SysReg::new(3, 4, 4, 1, 0);

/// Where kernel 23.5.0 keeps what identifies a process. The top of every
/// thread's kernel stack holds a block with a pointer to the thread object
/// (the supervisor call dispatcher loads `X18` from it at image 0x800c5134);
/// the thread points to its process (the context switch at 0x8009e590
/// compares these), and the process holds the name and program ID it was
/// created with (stored at 0x800948a0, for initial processes as for those of
/// `CreateProcess`) and its ID (which `GetProcessId` returns at 0x800bfd80).
/// The low half of that ID is what the context switch writes to
/// `CONTEXTIDR_EL1`.
mod kernel_layout {
    pub const STACK_TOP_THREAD: u64 = 0x20;
    pub const THREAD_PROCESS: u64 = 0x288;
    pub const PROCESS_NAME: u64 = 0x234;
    pub const PROCESS_NAME_SIZE: usize = 12;
    pub const PROCESS_PROGRAM_ID: u64 = 0x318;
    pub const PROCESS_ID: u64 = 0x320;
}

/// A process as the kernel describes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Process {
    pub id: u64,
    pub program_id: u64,
    name: [u8; kernel_layout::PROCESS_NAME_SIZE],
}

impl Process {
    /// The name the process was created with: that of its KIP for an initial
    /// process, else the one its creator passed to `CreateProcess`.
    pub fn name(&self) -> &str {
        let end = self.name.iter().position(|&byte| byte == 0);
        let name = &self.name[..end.unwrap_or(self.name.len())];
        str::from_utf8(name).unwrap_or("?")
    }
}

/// The thread that makes a supervisor call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Caller {
    /// `CONTEXTIDR_EL1`: the low half of the ID of the process whose address
    /// space is current.
    pub context: u64,
    /// Address of the thread's thread-local region (`TPIDRRO_EL0`), which
    /// tells the threads of one process apart.
    pub thread: u64,
}

impl Caller {
    /// The thread a core runs. Both registers are the kernel's to set and
    /// change only when it switches threads, so a core entering or leaving a
    /// supervisor call has the values of the thread that makes it.
    fn current(vcpu: &Vcpu) -> Result<Self, HvError> {
        Ok(Self {
            context: vcpu.sys_reg(SysReg::CONTEXTIDR_EL1)?,
            thread: vcpu.sys_reg(TPIDRRO_EL0)?,
        })
    }
}

/// Where a breakpoint or tracepoint sits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Place {
    /// A kernel image address. It matches while the kernel runs at its load
    /// address and, once it has relocated itself, at the relocated address.
    Image(u64),
    /// A virtual address taken as it is, in whichever address space is
    /// current: the way to reach code outside the kernel image, such as a
    /// process.
    Virtual(u64),
}

impl fmt::Display for Place {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Image(address) => write!(f, "image {address:#x}"),
            Self::Virtual(address) => write!(f, "virtual {address:#x}"),
        }
    }
}

impl Place {
    /// Whether this place is at `pc` for a kernel slid by `slide`.
    fn is_at(self, pc: u64, slide: u64) -> bool {
        match self {
            Self::Image(image) => image.wrapping_add(slide) == pc,
            Self::Virtual(address) => address == pc,
        }
    }
}

/// A debugger's memory write: each time a core is about to execute `place`,
/// the low `size` bytes of `value` are stored, in the core's current
/// address space, at the address in general register `base` plus `offset`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Poke {
    pub place: Place,
    pub base: u8,
    pub offset: i64,
    pub size: u8,
    pub value: u64,
}

#[derive(Clone, Copy)]
pub struct RunOptions<'a> {
    /// Wall-clock bound; the machine is stopped when it expires. Without
    /// one it runs until a core stops or `stop` is set.
    pub timeout: Option<Duration>,
    /// Stops the machine once it is set, from any thread.
    pub stop: &'a AtomicBool,
    /// Report every secure monitor call.
    pub trace_smc: bool,
    /// Report every supervisor call and its return. Costs two breakpoints.
    pub trace_svc: bool,
    /// Report every device register access.
    pub trace_devices: bool,
    /// Answer device accesses nothing models (reads with zero, writes by
    /// dropping them) and report each, instead of stopping the machine.
    /// What the guest does afterwards is not what a console does: this is
    /// for listing what a guest wants, not for running it.
    pub survey: bool,
    /// How many times slower than the host the guest's processors run; 1 is
    /// as fast as the host goes. Everything else keeps real time: timers,
    /// the clock, and the waits drivers make for their hardware. The
    /// console's software has only ever met its own processor, about a
    /// tenth of the host's speed, and its boot depends on it. The program
    /// launcher starts some forty programs one after the other; on the
    /// console loading each takes long enough that few are up while the
    /// clock driver (pcv) starts, which waits on its regulators in real
    /// time. At host speed nearly all of them are, and they keep settings
    /// busy in the filesystem process at the moment that process hands
    /// its card clock over to pcv, which is itself waiting for settings:
    /// the three then wait for each other for good.
    pub cpu_slowdown: u32,
    /// For how long from the start of the run the processors are slowed;
    /// `None` is throughout. The launcher is done within seconds.
    pub cpu_slowdown_for: Option<Duration>,
    /// Places that stop the machine when a core is about to execute them.
    pub breakpoints: &'a [Place],
    /// Places whose execution is reported without stopping.
    pub tracepoints: &'a [Place],
    /// Memory writes made on reaching a place. Each is reported like a
    /// tracepoint.
    pub pokes: &'a [Poke],
    /// Instructions to single-step, reporting each, between a breakpoint
    /// being reached and the machine stopping for it.
    pub steps_after_breakpoint: u32,
    /// Receives the reports, on the thread of the core they concern.
    pub trace: &'a (dyn Fn(TraceEvent<'_>) + Sync),
}

/// Something a run reports while it is in progress.
#[derive(Debug, Clone, Copy)]
pub enum TraceEvent<'a> {
    SecureMonitorCall {
        core: usize,
        immediate: u16,
        /// `X0`-`X7` at the call.
        arguments: [u64; 8],
        /// `X0`-`X7` the caller resumes with, when it does.
        results: [u64; 8],
        effect: SmcEffect,
    },
    /// A thread made a supervisor call.
    SupervisorCall {
        core: usize,
        /// Immediate of the `SVC` instruction.
        id: u16,
        /// `X0`-`X7` at the call.
        arguments: [u64; 8],
        /// Address of the `SVC` instruction.
        pc: u64,
        caller: Caller,
        /// `None` when the kernel's description of the process is not where
        /// expected.
        process: Option<Process>,
        /// The memory [`svc::input`] names for the call, as far as it is
        /// mapped and up to a limit.
        data: &'a [u8],
        /// For a call that sends a request: the start of each buffer the
        /// request hands over.
        buffers: &'a [ipc::Contents],
    },
    /// A supervisor call returned to the thread that made it. Calls return
    /// in any order relative to those of other threads, and some never do.
    SupervisorReturn {
        core: usize,
        id: u16,
        /// `X0`-`X7` at the call.
        arguments: [u64; 8],
        /// `X0`-`X7` at the return.
        results: [u64; 8],
        caller: Caller,
        process: Option<Process>,
        /// The memory [`svc::output`] names for the call, likewise.
        data: &'a [u8],
        /// For a call that sent a request: the start of each buffer the
        /// answer brings or has filled.
        buffers: &'a [ipc::Contents],
    },
    /// A tracepoint, or a breakpoint about to be stepped from, was reached.
    Reached {
        core: usize,
        at: CodeAddress,
        registers: [u64; 8],
    },
    DeviceAccess {
        core: usize,
        device: &'static str,
        address: u64,
        size: u8,
        is_write: bool,
        value: u64,
    },
    /// The console woke from sleep; `core` runs again.
    Woke { core: usize },
    /// A survey answered a device access that nothing models.
    Unmodeled {
        core: usize,
        /// `None` where there is no device at all.
        device: Option<&'static str>,
        address: u64,
        size: u8,
        is_write: bool,
    },
    /// One instruction was single-stepped; `at` is the next one. With
    /// `exception_entry` the stepped instruction entered a kernel exception
    /// handler instead of completing.
    Stepped {
        core: usize,
        at: CodeAddress,
        exception_entry: bool,
    },
}

/// Why a core left the run loop.
#[derive(Debug, Clone, Copy)]
pub enum CoreStop {
    /// Another core stopped the machine, or the run timed out.
    Requested,
    /// The kernel issued the Panic SMC.
    KernelPanic {
        code: u32,
    },
    UnimplementedSmc(Unimplemented),
    UnimplementedSysReg(SysRegAccess),
    /// Data access to a guest-physical address with nothing behind it.
    UnmappedAccess {
        address: u64,
        is_write: bool,
    },
    /// Access to a register or with a width the named device does not model.
    UnimplementedDevice {
        device: &'static str,
        address: u64,
        size: u8,
        is_write: bool,
    },
    /// Device access by an instruction form the abort syndrome does not describe.
    UndecodedAccess {
        address: u64,
        is_write: bool,
    },
    /// Instruction fetch from a guest-physical address with nothing behind it.
    UnmappedFetch {
        address: u64,
    },
    /// Exception to the hypervisor of a class that is not handled.
    UnhandledException {
        syndrome: u64,
    },
    /// The virtual timer fired; the kernel is not expected to arm it.
    VirtualTimer,
    UnknownExit(u32),
    Backend(HvError),
    /// The core's host thread panicked: an emulator bug, re-raised once the
    /// machine has stopped.
    HostPanic,
    /// A breakpoint was reached.
    Breakpoint {
        place: Place,
    },
    /// The requested breakpoints and tracepoints cannot be armed.
    CannotArm(BreakpointError),
    /// A [`Poke`]'s target is not mapped memory.
    CannotPoke {
        address: u64,
    },
    /// Supervisor call tracing was asked for, but the kernel image does not
    /// return from supervisor calls where the kernel this was written for
    /// does.
    UnknownSupervisorReturn,
}

impl fmt::Display for CoreStop {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let direction = |is_write| if is_write { "write to" } else { "read of" };
        match *self {
            Self::Requested => write!(f, "stopped with the machine"),
            Self::KernelPanic { code } => write!(f, "kernel panic (Panic SMC, code {code:#x})"),
            Self::UnimplementedSmc(Unimplemented::Function { immediate, id }) => {
                write!(f, "SMC #{immediate} function {id:#010x} is not implemented")
            }
            Self::UnimplementedSmc(Unimplemented::ConfigItem(item)) => {
                write!(f, "GetConfig item {item} is not implemented")
            }
            Self::UnimplementedSmc(Unimplemented::Crypto(what)) => {
                write!(f, "secure monitor key operation is not available: {what}")
            }
            Self::UnimplementedSmc(Unimplemented::Abort(why)) => {
                write!(f, "the secure monitor aborts: {why}")
            }
            Self::UnimplementedSmc(Unimplemented::Register { address, is_write }) => {
                write!(
                    f,
                    "{} register {address:#x} through the secure monitor is not implemented",
                    direction(is_write)
                )
            }
            Self::UnimplementedSysReg(access) => {
                write!(
                    f,
                    "trapped {} {:?} is not implemented",
                    direction(!access.is_read),
                    access.register
                )
            }
            Self::UnmappedAccess { address, is_write } => {
                write!(
                    f,
                    "{} physical {address:#x}, where no memory or device exists",
                    direction(is_write)
                )
            }
            Self::UnimplementedDevice {
                device,
                address,
                size,
                is_write,
            } => {
                write!(
                    f,
                    "{size}-byte {} {device} register at {address:#x} is not implemented",
                    direction(is_write)
                )
            }
            Self::UndecodedAccess { address, is_write } => write!(
                f,
                "{} device address {address:#x} by an instruction whose form the abort does not describe",
                direction(is_write)
            ),
            Self::UnmappedFetch { address } => {
                write!(
                    f,
                    "instruction fetch from physical {address:#x}, where no memory exists"
                )
            }
            Self::UnhandledException { syndrome } => {
                write!(
                    f,
                    "unhandled exception class {:#x} (syndrome {syndrome:#x})",
                    exception_class(syndrome)
                )
            }
            Self::VirtualTimer => write!(f, "virtual timer fired, which the kernel never arms"),
            Self::UnknownExit(reason) => write!(f, "unknown hypervisor exit reason {reason}"),
            Self::Backend(error) => write!(f, "hypervisor: {error}"),
            Self::HostPanic => write!(f, "emulator bug: the core's host thread panicked"),
            Self::Breakpoint { place } => write!(f, "breakpoint at {place}"),
            Self::CannotArm(error) => write!(f, "cannot arm breakpoints: {error}"),
            Self::CannotPoke { address } => {
                write!(
                    f,
                    "cannot poke virtual {address:#x}: it is not mapped memory"
                )
            }
            Self::UnknownSupervisorReturn => write!(
                f,
                "cannot trace supervisor calls: this kernel image has no ERET at \
                 {SVC_RETURN:#x}, where the kernel that tracing knows returns from them"
            ),
        }
    }
}

/// A code address as the core sees it, and where that is in the kernel image
/// when the page it lies on is an unmodified image page.
#[derive(Debug, Clone, Copy)]
pub struct CodeAddress {
    pub address: u64,
    pub image: Option<u64>,
}

/// Final state of one core.
#[derive(Debug, Clone, Copy)]
pub struct CoreReport {
    pub core: usize,
    pub stop: CoreStop,
    /// `None` when the core's registers could not be read.
    pub state: Option<CoreState>,
}

#[derive(Debug, Clone, Copy)]
pub struct CoreState {
    pub pc: CodeAddress,
    pub lr: CodeAddress,
    pub pstate: u64,
    /// Last exception the guest took to its own EL1 vectors.
    pub esr_el1: u64,
    pub elr_el1: CodeAddress,
    pub far_el1: u64,
    /// `CONTEXTIDR_EL1`: which address space the core was in.
    pub context: u64,
    /// The process with that address space, when the kernel's description
    /// of it was found.
    pub process: Option<Process>,
}

/// What ended a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunEnd {
    /// This core stopped, and the machine with it.
    Core(usize),
    Timeout,
    /// [`RunOptions::stop`] was set.
    Requested,
}

#[derive(Debug)]
pub struct RunReport {
    pub ended: RunEnd,
    /// One entry per started core, ordered by core index.
    pub cores: Vec<CoreReport>,
    /// Secure monitor calls served, by function name.
    pub smc_counts: BTreeMap<&'static str, u64>,
    /// Supervisor calls made and not returned from when the machine
    /// stopped, while they were being watched ([`RunOptions::trace_svc`]):
    /// what each waiting thread waits in. A thread that ended in
    /// `ExitThread` or whose process was ended is listed with its last call,
    /// and its return addresses are read from memory that may have been
    /// reused since.
    pub calls_in_progress: Vec<CallInProgress>,
}

/// A supervisor call that had not returned when the machine stopped.
#[derive(Debug, Clone)]
pub struct CallInProgress {
    pub caller: Caller,
    pub process: Option<Process>,
    /// Immediate of the `SVC` instruction.
    pub id: u16,
    /// `X0`-`X7` at the call.
    pub arguments: [u64; 8],
    /// Address of the `SVC` instruction.
    pub pc: u64,
    /// `X30` at the call: where the function that made it returns to.
    pub lr: u64,
    /// Return addresses further up, from the frame records on the
    /// thread's stack as they are when the machine stops; code built
    /// without frame pointers breaks the chain early.
    pub callers: Vec<u64>,
    /// How long before the stop the call was made.
    pub waited: Duration,
}

/// At most this many return addresses are taken from a waiting thread's
/// frame records.
const FRAME_CHAIN_LIMIT: usize = 16;

pub struct Machine {
    // Dropped first: the VM must be gone before the memory it maps, which
    // lives until the last device sharing it is dropped.
    vm: Vm,
    ram: Arc<GuestRam>,
    image: KernelImage,
    secmon: SecureMonitor,
    /// Memory as the security engine, which the monitor drives, reaches it.
    engine_memory: DeviceMemory,
    bus: Bus,
    gic: Arc<Gic>,
    timers: Timers,
    /// Each core's performance monitor registers.
    pmus: [Mutex<Pmu>; CORE_COUNT],
    soc: Soc,
    signals: Arc<CoreSignals>,
}

impl Machine {
    /// Builds the machine in its state at kernel entry: image in DRAM, no
    /// core running. `emmc` holds the contents of the eMMC module, if one is
    /// fitted, and `sd_card` those of the microSD card, if one is in the
    /// slot. `uart_output` receives what the guest sends to UART-A.
    pub fn new(
        image: KernelImage,
        params: &BootParams,
        keys: KeySet,
        emmc: Option<EmmcImages>,
        sd_card: Option<SdCardImage>,
        uart_output: Box<dyn io::Write + Send>,
    ) -> Result<Self, MachineError> {
        let size =
            usize::try_from(params.dram.bytes()).expect("DRAM size fits the host address space");
        let ram = Arc::new(GuestRam::new(boot::DRAM_BASE, size).map_err(MachineError::Ram)?);
        ram.write(image.load_address(), image.bytes())
            .ok_or(MachineError::ImageOutsideRam)?;
        let vm = Vm::new()?;
        // SAFETY: `ram` is a page-aligned mmap that outlives `vm`.
        unsafe { vm.map(ram.host_ptr(), ram.base(), ram.size())? };

        let signals = Arc::new(CoreSignals::default());
        let gic = Arc::new(Gic::new(CORE_COUNT, {
            let signals = Arc::clone(&signals);
            Box::new(move |core| signals.notify(core))
        }));
        let timers = Timers::new({
            let gic = Arc::clone(&gic);
            Box::new(move |core, level| gic.set_line(core, PHYSICAL_TIMER_INTID, level))
        });
        let mut bus = Bus::default();
        bus.map(
            "UART-A",
            uart::UART_A_BASE,
            uart::REGISTER_BLOCK_SIZE,
            Box::new(UartPort(Uart::new(uart_output))),
        );
        bus.map(
            "interrupt distributor",
            GIC_DISTRIBUTOR_BASE,
            GIC_DISTRIBUTOR_SIZE,
            Box::new(GicDistributor(Arc::clone(&gic))),
        );
        bus.map(
            "interrupt CPU interface",
            GIC_CPU_INTERFACE_BASE,
            GIC_CPU_INTERFACE_SIZE,
            Box::new(GicCpuInterface(Arc::clone(&gic))),
        );
        let soc =
            Soc::new(params, &gic, &ram, emmc, sd_card, &mut bus).map_err(MachineError::Emmc)?;

        Ok(Self {
            vm,
            engine_memory: DeviceMemory::new(&soc.mc, &ram, mc::SMMU_SE_ASID),
            ram,
            image,
            secmon: SecureMonitor::new(params, keys),
            bus,
            gic,
            timers,
            pmus: Default::default(),
            soc,
            signals,
        })
    }

    /// Explanations left by devices that refused an access, oldest first.
    /// Each call returns the notes added since the previous one.
    pub fn device_notes(&self) -> Vec<String> {
        self.soc.notes.take()
    }

    /// Where each GPU channel stands, a line each.
    pub fn gpu_channels(&self) -> Vec<String> {
        self.soc.gpu_channels()
    }

    /// What the console's screen shows at this moment.
    pub fn screen(&self) -> Result<Screen, NoFrame> {
        self.soc.screen()
    }

    /// Puts a finger on the touchscreen at `(x, y)` or moves it there
    /// (screen coordinates, 1280x720, origin top left), or lifts it with
    /// `None`. `false` if the touch controller is not scanning yet, so the
    /// change went unreported; see [`crate::soc::touch::TouchInput`].
    pub fn set_touch(&self, contact: Option<(u16, u16)>) -> bool {
        self.soc.touch_input.set_contact(contact)
    }

    /// Slides both Joy-Con onto their rails, or off them.
    pub fn seat_joycons(&self, seated: bool) {
        self.soc.seat_joycons(seated);
    }

    /// Whether the Joy-Con are on their rails.
    pub fn joycons_seated(&self) -> bool {
        self.soc.joycons.seated()
    }

    /// Holds a button of the Joy-Con, or lets go of it.
    pub fn press(&self, button: crate::soc::joycon::Button, held: bool) {
        self.soc.joycons.press(button, held);
    }

    /// Holds the console's power button, or lets go of it.
    pub fn press_power(&self, held: bool) {
        self.soc.pmic.power_key(held);
    }

    /// The WiFi function of the console's radio.
    pub fn wifi(&self) -> &std::sync::Arc<crate::soc::bcm4356::Bcm4356Wifi> {
        self.soc
            .wifi
            .get()
            .expect("the SoC attaches its WiFi function when built")
    }

    pub fn secure_monitor(&self) -> &SecureMonitor {
        &self.secmon
    }

    /// Enters the kernel on core 0 and runs until a core stops, the timeout
    /// expires or the caller asks for a stop.
    pub fn run(&self, options: RunOptions<'_>) -> RunReport {
        // How often a stop request is looked for, and stragglers are
        // re-kicked while the machine winds down.
        const KICK_INTERVAL: Duration = Duration::from_millis(10);

        self.soc.notes.set_survey(options.survey);
        let (reports, finished) = mpsc::channel();
        let paced_until = options.cpu_slowdown_for.map(|span| Instant::now() + span);
        let session = Session {
            machine: self,
            options,
            paced_until,
            slide: OnceLock::new(),
            calls: Mutex::new(HashMap::new()),
            processes: Mutex::new(HashMap::new()),
            stop: AtomicBool::new(false),
            started: AtomicUsize::new(0),
            switches: Default::default(),
            reports,
            smc_counts: Mutex::new(BTreeMap::new()),
        };

        let mut cores: Vec<CoreReport> = Vec::new();
        let mut ended = RunEnd::Timeout;
        thread::scope(|scope| {
            scope.spawn(|| self.timers.run(&session.stop));
            scope.spawn(|| self.soc.keep_time(&session.stop));
            if options.cpu_slowdown > 1 {
                scope.spawn(|| {
                    while !session.stop.load(Ordering::SeqCst) && session.paced() {
                        thread::sleep(PACING_SLICE);
                        (0..CORE_COUNT).for_each(|core| self.signals.interrupt_run(core));
                    }
                });
            }
            session.start_core(scope, 0, CoreEntry::primary(self.image.load_address()));
            let deadline = options.timeout.map(|timeout| Instant::now() + timeout);
            ended = loop {
                let left =
                    deadline.map(|deadline| deadline.saturating_duration_since(Instant::now()));
                let wait = left.map_or(KICK_INTERVAL, |left| left.min(KICK_INTERVAL));
                if let Ok(first) = finished.recv_timeout(wait) {
                    let core = first.core;
                    cores.push(first);
                    break RunEnd::Core(core);
                }
                if options.stop.load(Ordering::SeqCst) {
                    break RunEnd::Requested;
                }
                if left.is_some_and(|left| left.is_zero()) {
                    break RunEnd::Timeout;
                }
            };
            session.stop.store(true, Ordering::SeqCst);
            self.timers.wake();
            // A core reports only after any core it started is counted, so
            // `started` is final once every counted core has reported.
            while cores.len() < session.started.load(Ordering::SeqCst) {
                (0..CORE_COUNT).for_each(|core| self.signals.notify(core));
                cores.extend(finished.recv_timeout(KICK_INTERVAL).ok());
            }
        });
        cores.sort_by_key(|report| report.core);
        let stopped = Instant::now();
        let processes = session.processes.into_inner().unwrap();
        let mut calls_in_progress: Vec<CallInProgress> = session
            .calls
            .into_inner()
            .unwrap()
            .into_iter()
            .map(|(caller, call)| CallInProgress {
                caller,
                process: processes.get(&caller.context).copied(),
                id: call.id,
                arguments: call.arguments,
                pc: call.pc,
                lr: call.lr,
                callers: self.frame_chain(&call.regime, call.fp),
                waited: stopped.saturating_duration_since(call.made),
            })
            .collect();
        calls_in_progress.sort_by_key(|call| (call.caller.context, call.caller.thread));
        RunReport {
            ended,
            cores,
            smc_counts: session.smc_counts.into_inner().unwrap(),
            calls_in_progress,
        }
    }

    /// Return addresses from the chain of AArch64 frame records (`X29`
    /// pointing at the caller's `X29` and `X30`) that starts at `fp`, up to
    /// [`FRAME_CHAIN_LIMIT`], until a record is unreadable or does not lie
    /// above the one before it, as frames on a descending stack do.
    fn frame_chain(&self, regime: &TranslationRegime, mut fp: u64) -> Vec<u64> {
        let mut callers = Vec::new();
        while fp != 0 && callers.len() < FRAME_CHAIN_LIMIT {
            let (Some(next), Some(returns)) = (
                self.read_virtual_u64(regime, fp),
                self.read_virtual_u64(regime, fp.wrapping_add(8)),
            ) else {
                break;
            };
            if returns == 0 {
                break;
            }
            callers.push(returns);
            if next <= fp {
                break;
            }
            fp = next;
        }
        callers
    }

    fn create_vcpu(&self, core: usize, entry: CoreEntry) -> Result<Vcpu<'_>, HvError> {
        let vcpu = Vcpu::new(&self.vm)?;
        vcpu.set_sys_reg(SysReg::MIDR_EL1, boot::MIDR_EL1)?;
        vcpu.set_sys_reg(SysReg::MPIDR_EL1, boot::mpidr_el1(core))?;
        self.start_vcpu(&vcpu, entry)?;
        Ok(vcpu)
    }

    /// Puts a core where the monitor hands it to the kernel: at `entry`
    /// with the MMU off and interrupts masked. That is all the kernel
    /// relies on, when a core first starts and when one starts again
    /// after having been switched off; the rest of its state it sets up
    /// itself, so what a core that starts again still holds from before
    /// does not matter.
    fn start_vcpu(&self, vcpu: &Vcpu, entry: CoreEntry) -> Result<(), HvError> {
        vcpu.set_sys_reg(SysReg::SCTLR_EL1, boot::ENTRY_SCTLR_EL1)?;
        vcpu.set_pstate(boot::ENTRY_PSTATE)?;
        vcpu.set_pc(entry.pc)?;
        vcpu.set_gpr(0, entry.x0)
    }

    fn translation_regime(&self, vcpu: &Vcpu) -> Result<TranslationRegime, HvError> {
        Ok(TranslationRegime {
            sctlr: vcpu.sys_reg(SysReg::SCTLR_EL1)?,
            tcr: vcpu.sys_reg(SysReg::TCR_EL1)?,
            ttbr0: vcpu.sys_reg(SysReg::TTBR0_EL1)?,
            ttbr1: vcpu.sys_reg(SysReg::TTBR1_EL1)?,
        })
    }

    /// Carries out `poke`; the address it could not write to otherwise.
    fn poke(&self, vcpu: &Vcpu, poke: &Poke) -> Result<Result<(), u64>, HvError> {
        let address = vcpu.gpr(poke.base)?.wrapping_add_signed(poke.offset);
        let regime = self.translation_regime(vcpu)?;
        let bytes = poke.value.to_le_bytes();
        let written = bytes[..usize::from(poke.size)]
            .iter()
            .enumerate()
            .all(|(index, byte)| {
                regime
                    .translate(address.wrapping_add(index as u64), |table| {
                        self.ram.read_u64(table)
                    })
                    .and_then(|physical| self.ram.write(physical, &[*byte]))
                    .is_some()
            });
        Ok(if written { Ok(()) } else { Err(address) })
    }

    /// The instruction the core is about to execute, if its page is mapped RAM.
    fn current_instruction(&self, vcpu: &Vcpu) -> Result<Option<u32>, HvError> {
        let regime = self.translation_regime(vcpu)?;
        let mut instruction = [0; 4];
        Ok(regime
            .translate(vcpu.pc()?, |table| self.ram.read_u64(table))
            .and_then(|physical| self.ram.read(physical, &mut instruction))
            .map(|()| u32::from_le_bytes(instruction)))
    }

    /// Fills `buffer` from `address` in the address space of `regime`.
    /// Returns how much of it was filled: less than all where the range
    /// leaves mapped DRAM.
    fn read_virtual(&self, regime: &TranslationRegime, address: u64, buffer: &mut [u8]) -> usize {
        let mut done = 0;
        while done < buffer.len() {
            let at = address.wrapping_add(done as u64);
            let in_page = (PAGE_SIZE - at % PAGE_SIZE) as usize;
            let len = in_page.min(buffer.len() - done);
            let read = regime
                .translate(at, |table| self.ram.read_u64(table))
                .and_then(|physical| self.ram.read(physical, &mut buffer[done..done + len]));
            if read.is_none() {
                break;
            }
            done += len;
        }
        done
    }

    fn read_virtual_u64(&self, regime: &TranslationRegime, address: u64) -> Option<u64> {
        let mut value = [0; 8];
        (self.read_virtual(regime, address, &mut value) == value.len())
            .then(|| u64::from_le_bytes(value))
    }

    /// Up to `len` bytes, and no more than [`CALL_DATA_LIMIT`], at `address`
    /// in the core's current address space.
    fn call_data(&self, vcpu: &Vcpu, address: u64, len: u64) -> Result<Vec<u8>, HvError> {
        let regime = self.translation_regime(vcpu)?;
        let mut data = vec![0; len.min(CALL_DATA_LIMIT) as usize];
        let read = self.read_virtual(&regime, address, &mut data);
        data.truncate(read);
        Ok(data)
    }

    /// The process that the thread object at `thread` belongs to, if that is
    /// the process `context` (a `CONTEXTIDR_EL1` value) stands for. Anything
    /// else, such as a `thread` that is no thread object or belongs to no
    /// process, is `None`.
    fn process_of_thread(
        &self,
        regime: &TranslationRegime,
        thread: u64,
        context: u64,
    ) -> Option<Process> {
        use kernel_layout::*;

        // Kernel objects live in the upper half; a null or stray pointer
        // must not be followed into the process's own memory.
        let kernel_u64 = |address: u64| {
            (address >> 63 != 0)
                .then(|| self.read_virtual_u64(regime, address))
                .flatten()
        };
        let process = kernel_u64(thread.checked_add(THREAD_PROCESS)?)?;
        let id = kernel_u64(process.checked_add(PROCESS_ID)?)?;
        if id as u32 != context as u32 {
            return None;
        }
        let mut name = [0; PROCESS_NAME_SIZE];
        (self.read_virtual(regime, process + PROCESS_NAME, &mut name) == name.len())
            .then_some(())?;
        Some(Process {
            id,
            program_id: kernel_u64(process + PROCESS_PROGRAM_ID)?,
            name,
        })
    }

    /// The thread object of the thread the core runs, read where the kernel
    /// keeps it for the core's exception level. Meaningless (but harmless to
    /// [`Machine::process_of_thread`]) before the kernel has threads.
    fn current_thread(
        &self,
        vcpu: &Vcpu,
        regime: &TranslationRegime,
    ) -> Result<Option<u64>, HvError> {
        const PSTATE_EL: u64 = 0b1100;
        if vcpu.pstate()? & PSTATE_EL != 0 {
            return Ok(Some(vcpu.gpr(18)?));
        }
        self.stack_top_thread(vcpu, regime)
    }

    /// The thread object named at the top of the core's kernel stack. `SP_EL1`
    /// is that top whenever the core is not inside the kernel: in a process,
    /// on the first instruction of an exception from one, and on the `ERET`
    /// back to one.
    fn stack_top_thread(
        &self,
        vcpu: &Vcpu,
        regime: &TranslationRegime,
    ) -> Result<Option<u64>, HvError> {
        let top = vcpu.sys_reg(SP_EL1)?;
        Ok(top
            .checked_add(kernel_layout::STACK_TOP_THREAD)
            .and_then(|address| self.read_virtual_u64(regime, address)))
    }

    /// `address` as the core's current address space maps it.
    fn code_address(&self, vcpu: &Vcpu, address: u64) -> Result<CodeAddress, HvError> {
        let regime = self.translation_regime(vcpu)?;
        Ok(CodeAddress {
            address,
            image: self.image_address(&regime, address),
        })
    }

    fn core_state(&self, vcpu: &Vcpu) -> Result<CoreState, HvError> {
        let regime = self.translation_regime(vcpu)?;
        let code = |address| CodeAddress {
            address,
            image: self.image_address(&regime, address),
        };
        let context = vcpu.sys_reg(SysReg::CONTEXTIDR_EL1)?;
        let thread = self.current_thread(vcpu, &regime)?;
        Ok(CoreState {
            pc: code(vcpu.pc()?),
            lr: code(vcpu.gpr(30)?),
            pstate: vcpu.pstate()?,
            esr_el1: vcpu.sys_reg(SysReg::ESR_EL1)?,
            elr_el1: code(vcpu.sys_reg(SysReg::ELR_EL1)?),
            far_el1: vcpu.sys_reg(SysReg::FAR_EL1)?,
            context,
            process: thread.and_then(|thread| self.process_of_thread(&regime, thread, context)),
        })
    }

    /// Maps a guest virtual code address back to its link-time image address.
    fn image_address(&self, regime: &TranslationRegime, address: u64) -> Option<u64> {
        let offset = address % PAGE_SIZE;
        let physical = regime.translate(address - offset, |table| self.ram.read_u64(table))?;
        let mut page = [0; PAGE_SIZE as usize];
        self.ram.read(physical, &mut page)?;
        Some(self.image.locate_page(&page)? + offset)
    }

    /// Value of a system register whose reads trap to the host.
    fn read_system_register(&self, core: usize, register: SysReg) -> Option<u64> {
        match register {
            SysReg::CPUACTLR_EL1 => Some(boot::CPUACTLR_EL1),
            SysReg::CPUECTLR_EL1 => Some(boot::CPUECTLR_EL1),
            _ => match timer_register(register) {
                Some(timer) => Some(self.timers.read(core, timer)),
                None => self.pmus[core].lock().unwrap().read(register),
            },
        }
    }

    /// Applies a trapped system register write.
    fn write_system_register(&self, core: usize, register: SysReg, value: u64) -> Option<()> {
        match timer_register(register) {
            Some(timer) => self.timers.write(core, timer, value),
            None => self.pmus[core].lock().unwrap().write(register, value),
        }
    }
}

fn timer_register(register: SysReg) -> Option<TimerRegister> {
    match register {
        SysReg::CNTPCT_EL0 => Some(TimerRegister::Count),
        SysReg::CNTP_CTL_EL0 => Some(TimerRegister::Control),
        SysReg::CNTP_CVAL_EL0 => Some(TimerRegister::CompareValue),
        _ => None,
    }
}

struct GicDistributor(Arc<Gic>);

impl MmioDevice for GicDistributor {
    fn read(&self, core: usize, offset: u64, size: u8) -> Option<u64> {
        self.0.distributor_read(core, offset, size).map(u64::from)
    }

    fn write(&self, core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        self.0
            .distributor_write(core, offset, size, u32::try_from(value).ok()?)
    }
}

struct GicCpuInterface(Arc<Gic>);

impl MmioDevice for GicCpuInterface {
    fn read(&self, core: usize, offset: u64, size: u8) -> Option<u64> {
        self.0.cpu_interface_read(core, offset, size).map(u64::from)
    }

    fn write(&self, core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        self.0
            .cpu_interface_write(core, offset, size, u32::try_from(value).ok()?)
    }
}

struct UartPort(Uart);

impl MmioDevice for UartPort {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        self.0.read(offset, size).map(u64::from)
    }

    fn write(&self, _core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        self.0.write(offset, size, u32::try_from(value).ok()?)
    }
}

/// Lets any thread make a core re-evaluate its interrupt input, whether the
/// core is executing guest code or waiting in WFI.
#[derive(Default)]
struct CoreSignals([CoreSignal; CORE_COUNT]);

#[derive(Default)]
struct CoreSignal {
    /// The core's vCPU while it is inside the run loop; cleared, under this
    /// lock, before the vCPU is destroyed.
    vcpu: Mutex<Option<VcpuHandle>>,
    notified: Mutex<bool>,
    wake: Condvar,
}

impl CoreSignals {
    fn attach(&self, core: usize, vcpu: Option<VcpuHandle>) {
        *self.0[core].vcpu.lock().unwrap() = vcpu;
    }

    fn notify(&self, core: usize) {
        let signal = &self.0[core];
        *signal.notified.lock().unwrap() = true;
        signal.wake.notify_one();
        if let Some(vcpu) = *signal.vcpu.lock().unwrap() {
            vcpu.kick().expect("an attached vCPU is live");
        }
    }

    /// Makes `core` leave the guest if it is executing it, for its thread
    /// to look at the time.
    fn interrupt_run(&self, core: usize) {
        if let Some(vcpu) = *self.0[core].vcpu.lock().unwrap() {
            vcpu.kick().expect("an attached vCPU is live");
        }
    }

    /// Blocks until [`CoreSignals::notify`] has been called for `core` since
    /// the last wait returned.
    fn wait(&self, core: usize) {
        let signal = &self.0[core];
        let mut notified = signal
            .wake
            .wait_while(signal.notified.lock().unwrap(), |notified| !*notified)
            .unwrap();
        *notified = false;
    }
}

/// State shared by the core threads of one [`Machine::run`].
struct Session<'a> {
    machine: &'a Machine,
    options: RunOptions<'a>,
    /// When the processors stop being slowed, if they are and it ends.
    paced_until: Option<Instant>,
    stop: AtomicBool,
    started: AtomicUsize,
    /// Per core: whether it has a thread, and what a thread whose core was
    /// switched off is to start it with.
    switches: [Mutex<(bool, Option<CoreEntry>)>; CORE_COUNT],
    reports: mpsc::Sender<CoreReport>,
    smc_counts: Mutex<BTreeMap<&'static str, u64>>,
    /// Relocated minus image address of the kernel, once a core has been
    /// seen executing relocated kernel code.
    slide: OnceLock<u64>,
    /// Supervisor calls that have not returned, by who made them. A thread
    /// is in one call at a time, so its next call replaces one that never
    /// returned.
    calls: Mutex<HashMap<Caller, Call>>,
    /// Processes seen making supervisor calls, by `CONTEXTIDR_EL1` value.
    /// The kernel takes process IDs from counters that only go up (image
    /// 0x80094130 and 0x80094bd0), so entries stay true.
    processes: Mutex<HashMap<u64, Process>>,
}

/// A supervisor call in progress.
struct Call {
    id: u16,
    /// `X0`-`X7` at the call.
    arguments: [u64; 8],
    /// Buffers of a request that its answer fills.
    filled: Vec<ipc::Buffer>,
    /// Address of the `SVC` instruction and `X30` at it.
    pc: u64,
    lr: u64,
    /// `X29` at the call and the caller's address space, from which the
    /// chain of frame records is read if the call is still in progress
    /// when the machine stops.
    fp: u64,
    regime: TranslationRegime,
    made: Instant,
}

/// Debug state of one core: the hardware facility and what this run wants
/// from it.
struct Probes {
    debug: GuestDebug,
    /// What is added to the image addresses of the armed breakpoints: zero
    /// while the kernel runs at its load address, the kernel's slide once it
    /// has relocated. `None` until they are first armed.
    offset: Option<u64>,
    /// The armed supervisor call breakpoints.
    svc: Option<SvcProbes>,
    /// Breakpoint being stepped from and steps still to take.
    stepping: Option<(Place, u32)>,
    /// Time the core still has to stand still for the guest code it ran,
    /// with [`RunOptions::cpu_slowdown`].
    pacing_debt: Duration,
    /// [`Vcpu::exec_time`] when the debt was last added to.
    paced_to: Duration,
}

/// How long a core runs guest code at most before it pays its pacing debt.
const PACING_SLICE: Duration = Duration::from_micros(100);

/// How often a sleeping console looks for what would wake it.
const WAKE_POLL: Duration = Duration::from_millis(5);

/// Addresses of the two breakpoints supervisor call tracing takes.
#[derive(Clone, Copy)]
struct SvcProbes {
    /// The vector every call enters the kernel through.
    vector: u64,
    /// [`SVC_RETURN`] where the kernel has relocated it.
    returns: u64,
}

impl<'env> Session<'env> {
    /// Whether the processors are being slowed at this moment.
    fn paced(&self) -> bool {
        self.options.cpu_slowdown > 1 && self.paced_until.is_none_or(|end| Instant::now() < end)
    }

    fn start_core<'scope>(
        &'env self,
        scope: &'scope thread::Scope<'scope, 'env>,
        core: usize,
        entry: CoreEntry,
    ) {
        self.started.fetch_add(1, Ordering::SeqCst);
        self.switches[core].lock().unwrap().0 = true;
        scope.spawn(move || {
            // A core that dies without reporting would leave `run` waiting
            // for it forever, so a panic is reported before it propagates.
            let outcome =
                panic::catch_unwind(AssertUnwindSafe(|| self.run_core(scope, core, entry)));
            let report = outcome.as_ref().copied().unwrap_or(CoreReport {
                core,
                stop: CoreStop::HostPanic,
                state: None,
            });
            // The receiver outlives the scope, so this cannot fail.
            let _ = self.reports.send(report);
            if let Err(payload) = outcome {
                panic::resume_unwind(payload);
            }
        });
    }

    fn run_core<'scope>(
        &'env self,
        scope: &'scope thread::Scope<'scope, 'env>,
        core: usize,
        entry: CoreEntry,
    ) -> CoreReport {
        let machine = self.machine;
        let failed = |error| CoreReport {
            core,
            stop: CoreStop::Backend(error),
            state: None,
        };
        let vcpu = match machine.create_vcpu(core, entry) {
            Ok(vcpu) => vcpu,
            Err(error) => return failed(error),
        };
        let mut probes = match GuestDebug::attach(&vcpu) {
            Ok(debug) => Probes {
                debug,
                offset: None,
                svc: None,
                stepping: None,
                pacing_debt: Duration::ZERO,
                paced_to: Duration::ZERO,
            },
            Err(error) => return failed(error),
        };
        machine.signals.attach(core, Some(vcpu.handle()));
        let stop = loop {
            if self.stop.load(Ordering::SeqCst) {
                break CoreStop::Requested;
            }
            match self.enter_guest(scope, core, &vcpu, &mut probes) {
                Ok(None) => {}
                Ok(Some(stop)) => break stop,
                Err(error) => break CoreStop::Backend(error),
            }
        };
        machine.signals.attach(core, None);
        let mut state = machine.core_state(&vcpu).ok();
        // What the core runs at this instant may not be a thread of the
        // process whose address space it is in; tracing may still know it.
        if let Some(state) = &mut state
            && state.process.is_none()
        {
            state.process = self.known_process(state.context);
        }
        CoreReport { core, stop, state }
    }

    /// Runs the guest until its next exit and services that. `Ok(None)`
    /// means the core can be entered again.
    fn enter_guest<'scope>(
        &'env self,
        scope: &'scope thread::Scope<'scope, 'env>,
        core: usize,
        vcpu: &Vcpu,
        probes: &mut Probes,
    ) -> Result<Option<CoreStop>, HvError> {
        if let Some(stop) = self.arm(vcpu, probes)? {
            return Ok(Some(stop));
        }
        // Stepping off a reported breakpoint must execute its instruction,
        // not an interrupt handler that then returns to the breakpoint.
        let interrupt =
            self.machine.gic.irq_asserted(core) && !probes.debug.wants_interrupts_deferred(vcpu)?;
        vcpu.set_pending_irq(interrupt)?;
        let exit = probes.debug.run(vcpu)?;
        if self.paced() {
            // Standing still after running is running slowly, as far as
            // the guest can tell: its clocks go on. Only the time its own
            // code ran is stretched. Getting into and out of the guest
            // costs the host microseconds where the console's processor
            // spends nanoseconds on the device access that caused it, and
            // stretching that too makes whatever the console does to its
            // devices many times a second (its touch panel, its sensors)
            // more than one slowed core can do.
            let ran = vcpu.exec_time()?;
            probes.pacing_debt += (ran - probes.paced_to) * (self.options.cpu_slowdown - 1);
            probes.paced_to = ran;
            if probes.pacing_debt >= PACING_SLICE {
                thread::sleep(probes.pacing_debt);
                probes.pacing_debt = Duration::ZERO;
            }
        }
        match exit {
            DebugExit::Serviced => Ok(None),
            DebugExit::Guest(exit) => self.handle_exit(scope, core, vcpu, &mut probes.debug, exit),
            DebugExit::Breakpoint { pc } => self.breakpoint(core, vcpu, probes, pc),
            DebugExit::Step {
                pc,
                exception_entry,
            } => {
                let Some((breakpoint, left)) = probes.stepping.take() else {
                    return Ok(None);
                };
                let at = self.machine.code_address(vcpu, pc)?;
                (self.options.trace)(TraceEvent::Stepped {
                    core,
                    at,
                    exception_entry,
                });
                if left == 1 {
                    return Ok(Some(CoreStop::Breakpoint { place: breakpoint }));
                }
                probes.stepping = Some((breakpoint, left - 1));
                probes.debug.request_step();
                Ok(None)
            }
        }
    }

    /// Keeps the core's breakpoints where the kernel currently executes:
    /// at its load address until it relocates, at the relocated address
    /// after, and on the supervisor call path once the kernel has installed
    /// its vectors. `Ok(Some(_))` is why that cannot be done.
    fn arm(&self, vcpu: &Vcpu, probes: &mut Probes) -> Result<Option<CoreStop>, HvError> {
        let options = &self.options;
        let places = || {
            options
                .breakpoints
                .iter()
                .chain(options.tracepoints)
                .copied()
                .chain(options.pokes.iter().map(|poke| poke.place))
        };
        if places().next().is_some() {
            let offset = self.kernel_slide(vcpu)?.unwrap_or(0);
            if probes.offset != Some(offset) {
                for place in places() {
                    let address = match place {
                        Place::Image(image) => {
                            if let Some(previous) = probes.offset {
                                probes.debug.remove_breakpoint(image.wrapping_add(previous));
                            }
                            image.wrapping_add(offset)
                        }
                        // Stays where it was first armed.
                        Place::Virtual(_) if probes.offset.is_some() => continue,
                        Place::Virtual(address) => address,
                    };
                    if let Err(error) = probes.debug.add_breakpoint(address) {
                        return Ok(Some(CoreStop::CannotArm(error)));
                    }
                }
                probes.offset = Some(offset);
            }
        }
        if options.trace_svc && probes.svc.is_none() {
            let vector = vcpu
                .sys_reg(SysReg::VBAR_EL1)?
                .wrapping_add(LOWER_EL_SYNC_VECTOR);
            // The kernel's own vectors are the first ones in the upper half.
            // Both breakpoints are armed together, so that a core reports a
            // return only for calls it could also have reported.
            if vector >> 63 != 0
                && let Some(slide) = self.kernel_slide(vcpu)?
            {
                let image = &self.machine.image;
                let instruction = SVC_RETURN
                    .checked_sub(image.load_address())
                    .and_then(|offset| image.bytes().get(offset as usize..)?.first_chunk());
                if instruction != Some(&ERET.to_le_bytes()) {
                    return Ok(Some(CoreStop::UnknownSupervisorReturn));
                }
                let svc = SvcProbes {
                    vector,
                    returns: SVC_RETURN.wrapping_add(slide),
                };
                for address in [svc.vector, svc.returns] {
                    if let Err(error) = probes.debug.add_breakpoint(address) {
                        return Ok(Some(CoreStop::CannotArm(error)));
                    }
                }
                probes.svc = Some(svc);
            }
        }
        Ok(None)
    }

    /// Relocated minus image address of the kernel, learned from the first
    /// core found executing relocated kernel code.
    fn kernel_slide(&self, vcpu: &Vcpu) -> Result<Option<u64>, HvError> {
        if let Some(&slide) = self.slide.get() {
            return Ok(Some(slide));
        }
        let pc = vcpu.pc()?;
        if pc >> 63 == 0 {
            return Ok(None);
        }
        let image = self.machine.code_address(vcpu, pc)?.image;
        Ok(image.map(|image| *self.slide.get_or_init(|| pc.wrapping_sub(image))))
    }

    fn breakpoint(
        &self,
        core: usize,
        vcpu: &Vcpu,
        probes: &mut Probes,
        pc: u64,
    ) -> Result<Option<CoreStop>, HvError> {
        let options = &self.options;
        match probes.svc {
            Some(svc) if svc.vector == pc => return self.supervisor_call(core, vcpu),
            Some(svc) if svc.returns == pc => return self.supervisor_return(core, vcpu),
            _ => {}
        }
        let offset = probes.offset.unwrap_or(0);
        for poke in options
            .pokes
            .iter()
            .filter(|poke| poke.place.is_at(pc, offset))
        {
            if let Err(address) = self.machine.poke(vcpu, poke)? {
                return Ok(Some(CoreStop::CannotPoke { address }));
            }
        }
        let stop = options
            .breakpoints
            .iter()
            .copied()
            .find(|place| place.is_at(pc, offset));
        if let Some(place) = stop
            && options.steps_after_breakpoint == 0
        {
            return Ok(Some(CoreStop::Breakpoint { place }));
        }
        let at = self.machine.code_address(vcpu, pc)?;
        (options.trace)(TraceEvent::Reached {
            core,
            at,
            registers: argument_registers(vcpu)?,
        });
        if let Some(place) = stop {
            probes.stepping = Some((place, options.steps_after_breakpoint));
            probes.debug.request_step();
        }
        Ok(None)
    }

    /// The core is on the first instruction of the kernel's handler for
    /// synchronous exceptions from a process.
    fn supervisor_call(&self, core: usize, vcpu: &Vcpu) -> Result<Option<CoreStop>, HvError> {
        // Aborts and trapped instructions from EL0 arrive here too.
        let syndrome = vcpu.sys_reg(SysReg::ESR_EL1)?;
        if exception_class(syndrome) != exception_class::SVC64 {
            return Ok(None);
        }
        let id = syndrome as u16;
        let arguments = argument_registers(vcpu)?;
        let caller = Caller::current(vcpu)?;
        let data = match svc::input(id, &arguments, caller.thread) {
            Some((address, len)) => self.machine.call_data(vcpu, address, len)?,
            None => Vec::new(),
        };
        let process = self.process(vcpu, caller.context)?;
        let (given, filled) = match svc::session_handle(id, &arguments) {
            Some(_) => ipc::carried(&data),
            None => Default::default(),
        };
        let buffers = self.buffer_contents(vcpu, &given)?;
        let pc = vcpu.sys_reg(SysReg::ELR_EL1)?.wrapping_sub(4);
        let call = Call {
            id,
            arguments,
            filled,
            pc,
            lr: vcpu.gpr(30)?,
            fp: vcpu.gpr(29)?,
            regime: self.machine.translation_regime(vcpu)?,
            made: Instant::now(),
        };
        self.calls.lock().unwrap().insert(caller, call);
        (self.options.trace)(TraceEvent::SupervisorCall {
            core,
            id,
            arguments,
            pc,
            caller,
            process,
            data: &data,
            buffers: &buffers,
        });
        Ok(None)
    }

    /// The core is on the `ERET` that ends a supervisor call.
    fn supervisor_return(&self, core: usize, vcpu: &Vcpu) -> Result<Option<CoreStop>, HvError> {
        let caller = Caller::current(vcpu)?;
        let call = self.calls.lock().unwrap().remove(&caller);
        // A call that entered on a core whose breakpoints were not armed
        // yet was not reported; neither is its return.
        let Some(Call {
            id,
            arguments,
            mut filled,
            ..
        }) = call
        else {
            return Ok(None);
        };
        let data = match svc::output(id, &arguments, caller.thread) {
            Some((address, len)) => self.machine.call_data(vcpu, address, len)?,
            None => Vec::new(),
        };
        if svc::session_handle(id, &arguments).is_some() {
            filled.extend(ipc::carried(&data).0);
        }
        let buffers = self.buffer_contents(vcpu, &filled)?;
        (self.options.trace)(TraceEvent::SupervisorReturn {
            core,
            id,
            arguments,
            results: argument_registers(vcpu)?,
            caller,
            process: self.process(vcpu, caller.context)?,
            data: &data,
            buffers: &buffers,
        });
        Ok(None)
    }

    /// The start of each of `buffers` in the core's current address space.
    fn buffer_contents(
        &self,
        vcpu: &Vcpu,
        buffers: &[ipc::Buffer],
    ) -> Result<Vec<ipc::Contents>, HvError> {
        buffers
            .iter()
            .map(|&buffer| {
                let shown = buffer.size.min(ipc::SHOWN_BUFFER_BYTES);
                let bytes = self.machine.call_data(vcpu, buffer.address, shown)?;
                Ok(ipc::Contents { buffer, bytes })
            })
            .collect()
    }

    fn known_process(&self, context: u64) -> Option<Process> {
        self.processes.lock().unwrap().get(&context).copied()
    }

    /// The process a supervisor call comes from or returns to, for a core at
    /// either end of the call, where its kernel stack is at its top.
    fn process(&self, vcpu: &Vcpu, context: u64) -> Result<Option<Process>, HvError> {
        if let Some(process) = self.known_process(context) {
            return Ok(Some(process));
        }
        let machine = self.machine;
        let regime = machine.translation_regime(vcpu)?;
        let process = machine
            .stack_top_thread(vcpu, &regime)?
            .and_then(|thread| machine.process_of_thread(&regime, thread, context));
        if let Some(process) = process {
            self.processes.lock().unwrap().insert(context, process);
        }
        Ok(process)
    }

    /// Services one exit. `Ok(None)` resumes the guest.
    fn handle_exit<'scope>(
        &'env self,
        scope: &'scope thread::Scope<'scope, 'env>,
        core: usize,
        vcpu: &Vcpu,
        debug: &mut GuestDebug,
        exit: Exit,
    ) -> Result<Option<CoreStop>, HvError> {
        let (syndrome, physical_address) = match exit {
            Exit::Canceled => return Ok(None),
            Exit::VtimerActivated => return Ok(Some(CoreStop::VirtualTimer)),
            Exit::Unknown(reason) => return Ok(Some(CoreStop::UnknownExit(reason))),
            Exit::Exception {
                syndrome,
                physical_address,
                ..
            } => (syndrome, physical_address),
        };
        match exception_class(syndrome) {
            exception_class::SMC64 => self.secure_monitor_call(scope, core, vcpu, syndrome as u16),
            exception_class::SYSTEM_REGISTER => {
                self.system_register(core, vcpu, debug, SysRegAccess::from_syndrome(syndrome))
            }
            exception_class::DATA_ABORT_LOWER => {
                self.device_access(core, vcpu, syndrome, physical_address)
            }
            exception_class::WFX => {
                // ISS bit 0 distinguishes WFE, which may always complete at once.
                if syndrome & 1 == 0 {
                    self.wait_for_interrupt(core);
                }
                skip_instruction(vcpu)?;
                Ok(None)
            }
            exception_class::INSTRUCTION_ABORT_LOWER => Ok(Some(CoreStop::UnmappedFetch {
                address: physical_address,
            })),
            _ => Ok(Some(CoreStop::UnhandledException { syndrome })),
        }
    }

    /// WFI: parks the core until its interrupt input is asserted, whatever
    /// its PSTATE masks, or the machine stops.
    fn wait_for_interrupt(&self, core: usize) {
        let machine = self.machine;
        while !machine.gic.irq_asserted(core) && !self.stop.load(Ordering::SeqCst) {
            machine.signals.wait(core);
        }
    }

    fn secure_monitor_call<'scope>(
        &'env self,
        scope: &'scope thread::Scope<'scope, 'env>,
        core: usize,
        vcpu: &Vcpu,
        immediate: u16,
    ) -> Result<Option<CoreStop>, HvError> {
        let mut registers = argument_registers(vcpu)?;
        let arguments = registers;
        let machine = self.machine;
        let effect = machine.secmon.call(
            core,
            immediate,
            &mut registers,
            &machine.bus,
            &machine.engine_memory,
        );

        let name = secmon::function_name(immediate, arguments[0] as u32);
        *self.smc_counts.lock().unwrap().entry(name).or_default() += 1;
        if self.options.trace_smc {
            (self.options.trace)(TraceEvent::SecureMonitorCall {
                core,
                immediate,
                arguments,
                results: registers,
                effect,
            });
        }

        let restart = match effect {
            SmcEffect::Panic { code } => return Ok(Some(CoreStop::KernelPanic { code })),
            SmcEffect::Unimplemented(what) => return Ok(Some(CoreStop::UnimplementedSmc(what))),
            SmcEffect::PowerOff => loop {
                if self.stop.load(Ordering::SeqCst) {
                    return Ok(Some(CoreStop::Requested));
                }
                if let Some(entry) = self.switches[core].lock().unwrap().1.take() {
                    break entry;
                }
                machine.signals.wait(core);
            },
            SmcEffect::Sleep { entry, context } => {
                while !machine.soc.wake_pending() {
                    if self.stop.load(Ordering::SeqCst) {
                        return Ok(Some(CoreStop::Requested));
                    }
                    thread::sleep(WAKE_POLL);
                }
                (self.options.trace)(TraceEvent::Woke { core });
                CoreEntry::secondary(entry, context)
            }
            SmcEffect::Resume | SmcEffect::StartCore { .. } => {
                for (index, &register) in registers.iter().enumerate() {
                    vcpu.set_gpr(index as u8, register)?;
                }
                // An SMC trapped to the hypervisor reports the SMC itself
                // as the return address, unlike HVC.
                skip_instruction(vcpu)?;
                if let SmcEffect::StartCore {
                    core,
                    entry,
                    context,
                } = effect
                {
                    let entry = CoreEntry::secondary(entry, context);
                    let mut switch = self.switches[core].lock().unwrap();
                    if switch.0 {
                        // Its thread is waiting for this.
                        switch.1 = Some(entry);
                        drop(switch);
                        machine.signals.notify(core);
                    } else {
                        drop(switch);
                        self.start_core(scope, core, entry);
                    }
                }
                return Ok(None);
            }
        };
        machine.start_vcpu(vcpu, restart).map(|()| None)
    }

    /// Emulates a trapped system register access: CPU identification and
    /// timer registers from the machine, debug registers from the core's
    /// shadow of them.
    fn system_register(
        &self,
        core: usize,
        vcpu: &Vcpu,
        debug: &mut GuestDebug,
        access: SysRegAccess,
    ) -> Result<Option<CoreStop>, HvError> {
        let machine = self.machine;
        let register = access.register;
        let done = if access.is_read {
            let value = machine
                .read_system_register(core, register)
                .or_else(|| debug.read_register(register));
            match value {
                Some(value) => Some(vcpu.set_gpr(access.rt, value)?),
                None => None,
            }
        } else {
            let value = vcpu.gpr(access.rt)?;
            machine
                .write_system_register(core, register, value)
                .or_else(|| debug.write_register(register, value))
        };
        match done {
            Some(()) => skip_instruction(vcpu).map(|()| None),
            None => Ok(Some(CoreStop::UnimplementedSysReg(access))),
        }
    }

    /// Completes a load or store that reached a device.
    fn device_access(
        &self,
        core: usize,
        vcpu: &Vcpu,
        syndrome: u64,
        address: u64,
    ) -> Result<Option<CoreStop>, HvError> {
        let abort = DataAbort::from_syndrome(syndrome);
        let is_write = abort.is_write;
        let described = abort
            .transfer
            .map(|transfer| LoadStore::single(!is_write, transfer));
        let decoded = match described {
            Some(access) => Some(access),
            None => self
                .machine
                .current_instruction(vcpu)?
                .and_then(LoadStore::decode),
        };
        let Some(access) = decoded.filter(|access| access.is_load != is_write) else {
            return Ok(Some(CoreStop::UndecodedAccess { address, is_write }));
        };

        let bus = &self.machine.bus;
        let mut address = address;
        for transfer in [Some(access.first), access.second].into_iter().flatten() {
            let size = transfer.size;
            let stored = match is_write {
                true => Some(transfer.stored(vcpu.gpr(transfer.rt)?)),
                false => None,
            };
            let outcome = match stored {
                Some(value) => match bus.write(core, address, size, value) {
                    Access::Done(()) => Access::Done(value),
                    Access::Unmapped => Access::Unmapped,
                    Access::Unimplemented(device) => Access::Unimplemented(device),
                },
                None => bus.read(core, address, size),
            };
            let unmodeled = match outcome {
                Access::Done(_) => None,
                Access::Unmapped => Some(None),
                Access::Unimplemented(device) => Some(Some(device)),
            };
            let value = match (outcome, unmodeled) {
                (Access::Done(value), _) => value,
                (_, Some(device)) if self.options.survey => {
                    (self.options.trace)(TraceEvent::Unmodeled {
                        core,
                        device,
                        address,
                        size,
                        is_write,
                    });
                    // A read is answered with zero; a write is dropped,
                    // and still traced with what it stored.
                    stored.unwrap_or(0)
                }
                (Access::Unimplemented(device), _) => {
                    return Ok(Some(CoreStop::UnimplementedDevice {
                        device,
                        address,
                        size,
                        is_write,
                    }));
                }
                (Access::Unmapped, _) => {
                    return Ok(Some(CoreStop::UnmappedAccess { address, is_write }));
                }
            };
            if !is_write {
                vcpu.set_gpr(transfer.rt, transfer.loaded(value))?;
            }
            if self.options.trace_devices {
                let device = bus.device_at(address).unwrap_or("?");
                (self.options.trace)(TraceEvent::DeviceAccess {
                    core,
                    device,
                    address,
                    size,
                    is_write,
                    value,
                });
            }
            address += u64::from(size);
        }
        if let Some(writeback) = access.writeback {
            vcpu.set_gpr(
                writeback.rn,
                vcpu.gpr(writeback.rn)?
                    .wrapping_add_signed(writeback.offset),
            )?;
        }
        skip_instruction(vcpu).map(|()| None)
    }
}

fn skip_instruction(vcpu: &Vcpu) -> Result<(), HvError> {
    vcpu.set_pc(vcpu.pc()?.wrapping_add(4))
}

/// `X0`-`X7`, the registers that carry call arguments and results.
fn argument_registers(vcpu: &Vcpu) -> Result<[u64; 8], HvError> {
    let mut registers = [0; 8];
    for (index, register) in registers.iter_mut().enumerate() {
        *register = vcpu.gpr(index as u8)?;
    }
    Ok(registers)
}
