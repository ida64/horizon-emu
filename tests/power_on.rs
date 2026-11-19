//! The console's power-on, from the boot ROM to the kernel's entry, run on
//! the BPMP alone: no Hypervisor.framework VM, no CPU-cluster core.
//!
//! The boot ROM (high-level emulated) takes the BCT and package1 from the
//! eMMC's BOOT0, package1ldr runs on the interpreted BPMP and starts the
//! TSEC, whose firmware (modelled by what it leaves) decrypts PK11 and
//! restarts the BPMP at the NX bootloader; the bootloader checks the
//! hardware, releases CPU-cluster core 0 at the secure monitor, reads the
//! GPT and BCPKG2, initialises DRAM, loads package2 and halts the BPMP.
//! Core 0's part is the secure monitor's high-level emulated cold boot,
//! which runs here on a thread of its own exactly as the machine runs it,
//! and ends with the kernel placed in DRAM and core 0's entry: the point
//! where a machine would start its first vCPU.
//!
//! The test records each stage of that chain as it is reached. It fails
//! if any stage up to [`REACHED`], the furthest the emulated chain is known
//! to get, is missed, and once the kernel's entry is reached, if any stage
//! at all is; either way with the device notes and the BPMP's trace (its
//! last instructions and MMIO accesses) of the stop. `--nocapture` shows
//! the stages of a run that passes.
//!
//! Inputs (required, as the unit tests that use them): `../prod_newer.keys`,
//! the 23.5.0 firmware's content archives in `../fw` (for the boot images,
//! installed as `install-boot` installs them) and the kernel package2 holds
//! as an ELF in `../fw_kernel/kernel_normal.elf` (`pkg2elf.py`).

use std::{
    fs::{File, OpenOptions},
    os::unix::fs::FileExt,
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use horizon_emu::{
    boot::{BootParams, CORE_COUNT, DRAM_BASE},
    bootrom,
    bus::Bus,
    clock::GuestClock,
    gic::Gic,
    image::KernelImage,
    keys::KeySet,
    machine::{Bpmp, Env, PowerOn, Slice},
    memory::GuestRam,
    nand::{self, boot_images::BootImages},
    secmon::{ColdBootContext, SecureMonitor},
    soc::{BootBlocks, Soc, emmc::EmmcImages, flow::CopStatus},
};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../");

/// Instructions per look at the world outside the BPMP, as the machine's
/// run loop.
const SLICE: u64 = 4096;

/// Host time the whole chain may take. Guest time is host time; the
/// bootloader's own waits (PLL locks, eMMC, DRAM training delays) add up
/// to well under a second, the rest is interpretation.
const DEADLINE: Duration = Duration::from_secs(180);

/// How many instructions and MMIO accesses the BPMP's trace keeps.
const TRACE: usize = 128;

/// The NX bootloader's mailbox in IRAM (research §2.8): its own state, and
/// the monitor's.
const BOOTLOADER_STATE: u32 = 0xF8;
const MONITOR_STATE: u32 = 0xFC;

/// Where the PK11 plaintext's header lands once the TSEC has decrypted the
/// blob in place (package1ldr's PK11 parser, text 0xA10/0xA80).
const PK11_HEADER: u32 = 0x1_7000;
/// The NX bootloader's stub, where the BPMP restarts after the TSEC.
const NX_BOOTLOADER_STUB: std::ops::Range<u32> = 0x4002_E104..0x4003_0000;
/// The NX bootloader's code once its stub has decompressed it.
const NX_BOOTLOADER_MAIN: std::ops::Range<u32> = 0x4000_0200..0x4001_7A0E;
/// Where the bootloader releases core 0: the secure monitor's first
/// instruction.
const MONITOR_ENTRY: u64 = 0x4003_0000;

/// A stage of the chain, in the order a console reaches them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Stage {
    /// The boot ROM accepted the BCT and package1 and handed the BPMP over.
    BootRomHandoff,
    /// The TSEC's firmware ran and the BPMP restarted at the NX
    /// bootloader's stub.
    NxBootloaderStub,
    /// The stub decompressed the bootloader and entered its main code.
    NxBootloaderMain,
    /// CPU-cluster core 0 released at the monitor's entry (after the
    /// hardware check, PMIC set-up and CPU power-up).
    Cpu0Released,
    /// The monitor's cold boot set its state (it no longer needs its IRAM
    /// image).
    MonitorReady,
    /// Bootloader state 1: GPT and BCPKG2 read, boot config handed over.
    BootConfigLoaded,
    /// Bootloader state 2: DRAM initialised.
    DramInitialised,
    /// Bootloader state 4: package2 in DRAM.
    Package2Loaded,
    /// The bootloader halted the BPMP through the flow controller.
    BpmpHalted,
    /// The monitor placed the kernel and gave core 0's entry at EL1.
    KernelEntry,
}

/// The stages reached, with the guest time of each.
#[derive(Default)]
struct Progress {
    reached: Mutex<Vec<(Stage, Duration)>>,
}

impl Progress {
    fn reach(&self, stage: Stage, clock: &GuestClock) {
        let mut reached = self.reached.lock().unwrap();
        if !reached.iter().any(|&(done, _)| done == stage) {
            reached.push((stage, clock.now()));
        }
    }

    fn has(&self, stage: Stage) -> bool {
        self.reached
            .lock()
            .unwrap()
            .iter()
            .any(|&(done, _)| done == stage)
    }

    fn report(&self) -> String {
        self.reached
            .lock()
            .unwrap()
            .iter()
            .map(|(stage, at)| format!("  {stage:?} at {:.6} s guest time\n", at.as_secs_f64()))
            .collect()
    }
}

/// Sparse image files of a console's eMMC as `install-boot` leaves them:
/// a partitioned user area with package2 in its BCPKG2 partitions, and
/// both boot partitions with their BCTs and package1.
fn media(keys: &KeySet) -> (EmmcImages, [Vec<u8>; 2]) {
    let directory =
        std::env::temp_dir().join(format!("horizon-emu-power-on-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let create = |name: &str| {
        OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(directory.join(name))
            .unwrap()
    };
    let [user, boot0, boot1] = ["user.img", "boot0.img", "boot1.img"].map(create);
    // The files stay open; their names are not needed.
    std::fs::remove_dir_all(&directory).unwrap();
    nand::format_user_area(&user).unwrap();
    let boot_images = BootImages::load(Path::new(&format!("{ROOT}fw")), keys)
        .expect("required 23.5.0 firmware content archives in ../fw");
    boot_images.write_package2(&user).unwrap();
    boot_images.write_boot_partitions([&boot0, &boot1]).unwrap();
    let read = |file: &File| {
        let mut bytes = vec![0; file.metadata().unwrap().len() as usize];
        file.read_exact_at(&mut bytes, 0).unwrap();
        bytes
    };
    let boot_bytes = [read(&boot0), read(&boot1)];
    let images = EmmcImages {
        user,
        boot: [Some(boot0), Some(boot1)],
        writable: false,
        overlay: Default::default(),
    };
    (images, boot_bytes)
}

fn iram_word(blocks: &BootBlocks, offset: u32) -> u32 {
    blocks.iram.read(offset, 4).expect("mailbox inside IRAM")
}

/// What the run came to: the stages reached and, if it stopped short, why.
struct Outcome {
    progress: Arc<Progress>,
    stop: Option<String>,
    notes: Vec<String>,
    trace: Vec<String>,
    pc: u32,
}

impl Outcome {
    fn explain(&self) -> String {
        let mut text = format!("stages reached:\n{}", self.progress.report());
        if let Some(stop) = &self.stop {
            text += &format!("stopped: {stop}\nBPMP pc {:#010x}\n", self.pc);
        }
        for note in &self.notes {
            text += &format!("note: {note}\n");
        }
        // Delay loops read TIMERUS over and over: runs of accesses to one
        // register are shown once, and instructions already shown among
        // the last few (a loop going round) are counted instead.
        const LOOP: usize = 8;
        text += "BPMP trace (oldest first):\n";
        let mut shown: Vec<&str> = Vec::new();
        let mut looped = 0;
        let mut lines = self.trace.iter().peekable();
        while let Some(line) = lines.next() {
            if line.starts_with("at ") {
                if shown.iter().rev().take(LOOP).any(|seen| seen == line) {
                    looped += 1;
                    continue;
                }
                shown.push(line);
            }
            if looped > 0 {
                text += &format!("  … {looped} instructions round a loop\n");
                looped = 0;
            }
            let key = |line: &str| line.split(": ").next().unwrap_or(line).to_string();
            let mut repeats = 0;
            let mut last = line;
            while let Some(next) = lines.next_if(|next| key(next) == key(line)) {
                repeats += 1;
                last = next;
            }
            match repeats {
                0 => text += &format!("  {line}\n"),
                _ => text += &format!("  {line}\n  … {repeats} more, the last: {last}\n"),
            }
        }
        text
    }
}

fn power_on() -> Outcome {
    let keys =
        KeySet::load(Path::new(&format!("{ROOT}prod_newer.keys"))).expect("required key file");
    let kernel = KernelImage::load(Path::new(&format!("{ROOT}fw_kernel/kernel_normal.elf")))
        .expect("required kernel ELF of 23.5.0's package2 in ../fw_kernel");
    let (emmc, boot_bytes) = media(&keys);

    // The boot ROM, as `boot --power-on` runs it (src/main.rs `boot_rom`).
    let fuses = bootrom::Fuses::retail(
        *keys
            .secure_boot_key()
            .expect("secure_boot_key in the key set"),
        0,
        bootrom::ChipFuses::default(),
    );
    let mut iram = vec![0; bootrom::IRAM_SIZE];
    let handoff = bootrom::cold_boot(&boot_bytes[0], &boot_bytes[1], &fuses, 0, &mut iram)
        .expect("the boot ROM boots package1 from BOOT0");
    let entry = handoff.entry;
    let power_on = PowerOn { handoff, iram };

    // The machine of `Machine::new` with `power_on`, without the VM and the
    // CPU cluster's devices only its cores reach.
    let params = BootParams::default();
    let ram = Arc::new(GuestRam::new(DRAM_BASE, params.dram.bytes() as usize).unwrap());
    let gic = Arc::new(Gic::new(CORE_COUNT, Box::new(|_| {})));
    let clock = Arc::new(GuestClock::new());
    let mut bus = Bus::default();
    let soc = Soc::new(
        &params,
        &gic,
        &ram,
        &clock,
        Some(emmc),
        None,
        true,
        &mut bus,
    )
    .expect("the SoC builds on these images");
    let bus = Arc::new(bus);
    let bpmp = Bpmp::power_on(&power_on, &soc, &bus, &keys).expect("the hand-off applies");
    let blocks = soc
        .boot
        .as_ref()
        .expect("a SoC powered on has the boot blocks");
    bpmp.set_trace(TRACE);
    let monitor = SecureMonitor::new(&params, keys);
    let env = Env {
        blocks,
        ram: &ram,
        bus: &bus,
        clock: &clock,
        notes: &soc.notes,
    };

    let progress = Arc::new(Progress::default());
    if entry == 0x4001_0040 && bpmp.core().pc() == entry {
        progress.reach(Stage::BootRomHandoff, &clock);
    }
    let stop = AtomicBool::new(false);
    let cold_boot = Mutex::new(None::<Result<(), String>>);
    let started = Instant::now();
    let mut stopped = None;
    thread::scope(|scope| {
        scope.spawn(|| soc.keep_time(&stop));
        let mut released = false;
        while stopped.is_none() {
            if started.elapsed() > DEADLINE {
                stopped = Some(format!("no further stage within {DEADLINE:?} of host time"));
                break;
            }
            if let Some(result) = cold_boot.lock().unwrap().as_ref() {
                match result {
                    Ok(()) => {
                        // The monitor went on only once the BPMP halted; the
                        // halt may have come after the BPMP's last slice.
                        if blocks.flow.cop_status(false, false) != CopStatus::Running {
                            progress.reach(Stage::BpmpHalted, &clock);
                        }
                        break;
                    }
                    Err(error) => {
                        stopped = Some(error.clone());
                        break;
                    }
                }
            }
            let slice = bpmp.slice(&env, SLICE);
            let pc = bpmp.core().pc();
            if NX_BOOTLOADER_STUB.contains(&pc)
                && iram_word(blocks, PK11_HEADER).to_le_bytes() == *b"PK11"
            {
                progress.reach(Stage::NxBootloaderStub, &clock);
            }
            if progress.has(Stage::NxBootloaderStub) && NX_BOOTLOADER_MAIN.contains(&pc) {
                progress.reach(Stage::NxBootloaderMain, &clock);
            }
            if released && iram_word(blocks, MONITOR_STATE) != 0 {
                progress.reach(Stage::MonitorReady, &clock);
            }
            if progress.has(Stage::Cpu0Released) {
                for (state, stage) in [
                    (1, Stage::BootConfigLoaded),
                    (2, Stage::DramInitialised),
                    (4, Stage::Package2Loaded),
                ] {
                    if iram_word(blocks, BOOTLOADER_STATE) >= state {
                        progress.reach(stage, &clock);
                    }
                }
            }
            match slice {
                Slice::Ran | Slice::Halted(CopStatus::Running) => {}
                Slice::Paused | Slice::Held => thread::sleep(Duration::from_millis(1)),
                Slice::Halted(halt) => {
                    if progress.has(Stage::Package2Loaded) {
                        progress.reach(Stage::BpmpHalted, &clock);
                    }
                    match halt {
                        CopStatus::HaltedUntil(deadline) => {
                            thread::sleep(clock.until(deadline).min(Duration::from_millis(1)))
                        }
                        _ => thread::sleep(Duration::from_millis(1)),
                    }
                }
                Slice::Released(entry) => {
                    if released {
                        stopped = Some(format!("core 0 released a second time, at {entry:#x}"));
                        continue;
                    }
                    released = true;
                    if entry != MONITOR_ENTRY {
                        stopped = Some(format!("core 0 released at {entry:#x}"));
                        continue;
                    }
                    progress.reach(Stage::Cpu0Released, &clock);
                    let (progress, monitor, cold_boot) = (&progress, &monitor, &cold_boot);
                    let (ram, iram, bus, clock, stop, kernel) =
                        (&ram, &blocks.iram, &bus, &clock, &stop, &kernel);
                    scope.spawn(move || {
                        let context = ColdBootContext { ram, iram, bus };
                        let result = monitor
                            .cold_boot(entry, context, stop)
                            .map_err(|error| error.to_string())
                            .and_then(|placed| {
                                if placed.kernel.load_address() != kernel.load_address()
                                    || placed.kernel.bytes() != kernel.bytes()
                                {
                                    return Err("the kernel package2 holds is not \
                                                fw_kernel/kernel_normal.elf"
                                        .into());
                                }
                                if placed.entry.pc != kernel.load_address() {
                                    return Err(format!(
                                        "core 0 enters the kernel at {:#x}, not its first byte",
                                        placed.entry.pc
                                    ));
                                }
                                progress.reach(Stage::KernelEntry, clock);
                                Ok(())
                            });
                        *cold_boot.lock().unwrap() = Some(result);
                    });
                }
                Slice::Fault(fault) => stopped = Some(format!("BPMP: {fault}")),
            }
        }
        stop.store(true, Ordering::SeqCst);
    });
    Outcome {
        progress,
        stop: stopped,
        notes: soc.notes.take(),
        trace: bpmp.trace(),
        pc: bpmp.core().pc(),
    }
}

/// The furthest stage the chain reaches today: raised as the models get it
/// further, up to [`Stage::KernelEntry`]. The next divergence is the stop
/// the test prints.
const REACHED: Stage = Stage::Cpu0Released;

/// The chain on the BPMP, towards the kernel's entry.
#[test]
fn power_on_runs_the_boot_chain_on_the_bpmp_alone() {
    let outcome = power_on();
    let explained = outcome.explain();
    let reached: Vec<Stage> = outcome
        .progress
        .reached
        .lock()
        .unwrap()
        .iter()
        .map(|&(stage, _)| stage)
        .collect();
    let complete = outcome.stop.is_none() && reached.contains(&Stage::KernelEntry);
    let floor = if complete {
        Stage::KernelEntry
    } else {
        REACHED
    };
    // Every stage up to the floor (the monitor's own state may come before
    // or after the bootloader's first).
    for stage in [
        Stage::BootRomHandoff,
        Stage::NxBootloaderStub,
        Stage::NxBootloaderMain,
        Stage::Cpu0Released,
        Stage::MonitorReady,
        Stage::BootConfigLoaded,
        Stage::DramInitialised,
        Stage::Package2Loaded,
        Stage::BpmpHalted,
        Stage::KernelEntry,
    ] {
        assert!(
            stage > floor || reached.contains(&stage),
            "{stage:?} never reached\n{explained}"
        );
    }
    println!("{explained}");
}
