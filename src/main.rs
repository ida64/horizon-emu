mod window;

use std::{
    fs::{File, OpenOptions},
    io::{self, Write},
    panic,
    path::Path,
    process::ExitCode,
    sync::{
        Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use horizon_emu::{
    boot::{BootParams, DramSize},
    image::KernelImage,
    ipc::{Client, ResultCode, Tracker},
    keys::KeySet,
    machine::{Caller, CodeAddress, Machine, Place, Poke, Process, RunEnd, RunOptions, TraceEvent},
    maxwell, nand,
    secmon::{self, SmcEffect},
    soc::{dc::Frame, emmc::EmmcImages, joycon::Button},
    svc,
    symbols::SymbolTable,
};

use crate::window::{LogOverlay, ScreenWindow};

const USAGE: &str = "\
usage: horizon-emu info <kernel.elf>
       horizon-emu boot <kernel.elf> [options]
       horizon-emu format-emmc <new image file> [--keys FILE [--firmware DIR]]

<kernel.elf> is the package2 kernel wrapped as an ELF (see pkg2elf.py).
boot runs until a core needs something that is not emulated, the kernel
panics, the timeout expires, or its window is closed. format-emmc creates
an eMMC user area image for --emmc (a sparse file): the 32 GB module's
partition table; with --keys also the empty, encrypted file systems of a
console's partitions and the calibration data of this emulator's own
console (its serial number, addresses and sensor values are made up for
it; it has no device keys or certificates); with --firmware also the
contents (*.nca) of that directory, installed as the system's programs
and data.

options:
  --dram 4|6|8        installed DRAM in GiB (default 4)
  --seed N            seed of the secure monitor's random numbers (default 0)
  --timeout SECONDS   wall-clock limit (default 5, or none with --window)
  --cpu-slowdown N[:SECONDS]
                      run the guest's processors N times slower than the
                      host can, for the first SECONDS of the run or, without
                      SECONDS, throughout. Timers and device waits keep real
                      time either way. The console's own processor is
                      roughly a tenth of the host's speed, and its boot
                      counts on that: while the system's programs are being
                      launched, loading them (processor work) has to take
                      its time next to what the first drivers wait for
                      (real time), or three processes can end up waiting for
                      each other and the boot stops with nothing on screen
                      (seen in 1 of 6 boots at full speed, none of 8 with
                      the default). Default 10:12; 1 for full speed
                      throughout
  --symbols FILE      function names for kernel addresses (JSON export)
  --keys FILE         console keys (name = hex lines); without them the
                      secure monitor cannot derive keys for the guest
  --emmc FILE         raw image of the eMMC user area; without it the eMMC
                      module is absent. Writes stay in memory.
  --emmc-boot0 FILE   raw images of the eMMC boot partitions (default:
  --emmc-boot1 FILE   erased)
  --emmc-writable     write to the eMMC images instead: what the guest
                      stores is there on the next run
  --window            show the console's screen in a window. Closing it
                      stops the machine and ends the program; when the
                      machine stops by itself, the window stays, showing
                      the screen as it was left, until it is closed
  --screenshot FILE   when the machine stops, save what its screen shows as
                      a PNG
  --screenshot-at SECONDS:FILE
                      save what the screen shows as a PNG SECONDS after the
                      machine starts; may be given more than once
  --renderer NAME     draw the GPU's 3D work with `metal` (the host GPU, the
                      default when there is one; what it cannot do falls
                      back to software per draw) or `software` (the
                      reference, the default without a Metal device)
  --gpu-capture FILE  record every GPU channel's method calls and the guest
                      memory its engines read, for replay
                      (horizon_emu::maxwell::capture)
  --touch SECONDS:X,Y[:HOLD_MS]
                      put a finger on the screen at X,Y (1280x720, origin
                      top left, as the picture is seen) SECONDS after the
                      machine starts, lifting it HOLD_MS milliseconds later
                      (default 100); may be given more than once. With
                      --window, holding the left mouse button on the
                      picture touches it too
  --joycons SECONDS[:off]
                      slide both Joy-Con onto the console's rails SECONDS
                      after the machine starts (0: they are on from power
                      on), or with :off, slide them off; may be given more
                      than once. Without this the rails are empty
  --press SECONDS:BUTTON[:HOLD_MS]
                      hold a button of the Joy-Con SECONDS after the machine
                      starts and let go HOLD_MS milliseconds later (default
                      100); may be given more than once. BUTTON is one of a
                      b x y l r zl zr plus minus home capture up down left
                      right lstick rstick
  --debug-mode        report development functions as enabled to the kernel,
                      which turns on its debug log (UART-A, shown on stderr)

tracing and debugging (output on stderr; WHERE is a kernel image address
as 0x..., a function name from --symbols, or @0x... for a virtual address
taken as it is, such as one inside a process):
  --trace-smc         print every secure monitor call
  --trace-svc         print every supervisor call and its return, with the
                      process and thread making it (threads are told apart
                      by the address of their thread-local region)
  --trace-ipc         print every request a process sends to a service and
                      the answer to it: service or session, kind and command,
                      data words (which for the spl services include key
                      material), handles, buffers with the first 32 bytes of
                      each (not for spl and csrng) and the result code
  --trace-devices     print every device register access
  --trace-times       start every trace line with the seconds since the
                      machine was started
  --verbose           narrate the boot, with times: cores started, each
                      program loaded (name, program ID, load address),
                      services registered, processes exiting or stopping
                      themselves, fatal errors reported, kernel debug text.
                      Like --trace-svc it watches every supervisor call,
                      which slows the guest
  --survey            do not stop at device accesses nothing models: answer
                      reads with zero, drop writes, and print each. Not a
                      faithful run; it lists what the guest wants
  --trace WHERE       print the registers each time a core reaches WHERE
  --break WHERE       stop the machine when a core reaches WHERE
  --step N            after a --break is reached, single-step N instructions
                      on that core, printing each, before stopping
  --poke WHERE:xN+OFFSET:SIZE:VALUE
                      each time a core reaches WHERE, store VALUE (SIZE
                      bytes, 1-8) at the address in register xN plus OFFSET,
                      in that core's address space; numbers as 0x... or
                      decimal
Each --trace, --break and --poke place takes one of the CPU's hardware
breakpoints on every core; --trace-svc and --trace-ipc take two between them.";

fn main() -> ExitCode {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    match run(&arguments) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("horizon-emu: {message}");
            ExitCode::FAILURE
        }
    }
}

fn run(arguments: &[String]) -> Result<(), String> {
    let [command, path, flags @ ..] = arguments else {
        return Err(USAGE.into());
    };
    if command == "format-emmc" {
        return format_emmc(path, flags).map_err(|error| format!("{path}: {error}"));
    }
    let image = KernelImage::load(Path::new(path)).map_err(|error| error.to_string())?;
    match command.as_str() {
        "info" if flags.is_empty() => {
            info(&image);
            Ok(())
        }
        "boot" => boot(image, flags),
        _ => Err(USAGE.into()),
    }
}

fn format_emmc(path: &str, flags: &[String]) -> Result<(), String> {
    let (mut keys, mut firmware) = (None, None);
    let mut flags = flags.iter();
    while let Some(flag) = flags.next() {
        let value = flags.next().ok_or(USAGE)?;
        match flag.as_str() {
            "--keys" => {
                keys = Some(KeySet::load(Path::new(value)).map_err(|error| error.to_string())?)
            }
            "--firmware" => firmware = Some(Path::new(value)),
            _ => return Err(USAGE.into()),
        }
    }
    if firmware.is_some() && keys.is_none() {
        return Err("--firmware needs --keys".into());
    }
    let image = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| error.to_string())?;
    nand::format_user_area(&image).map_err(|error| error.to_string())?;
    let Some(keys) = keys else {
        return Ok(());
    };
    nand::format_filesystems(&image, &keys).map_err(|error| error.to_string())?;
    nand::calibration::install(&image, &keys, &nand::calibration::Identity::VIRTUAL_CONSOLE)
        .map_err(|error| error.to_string())?;
    if let Some(firmware) = firmware {
        let installed =
            nand::install_firmware(&image, &keys, firmware).map_err(|error| error.to_string())?;
        println!(
            "installed {} contents, {} bytes",
            installed.contents, installed.bytes
        );
    }
    Ok(())
}

fn info(image: &KernelImage) {
    let layout = &image.layout;
    println!(
        "load address   {:#x}..{:#x}",
        image.load_address(),
        image.end()
    );
    println!("early entry    {:#x}", image.early_entry);
    println!("layout block   {:#x}", image.layout_address);
    println!(
        "  text         {:#x}..{:#x}",
        layout.text.start, layout.text.end
    );
    println!(
        "  rodata       {:#x}..{:#x}",
        layout.rodata.start, layout.rodata.end
    );
    println!(
        "  data         {:#x}..{:#x}",
        layout.data.start, layout.data.end
    );
    println!(
        "  bss          {:#x}..{:#x}",
        layout.bss.start, layout.bss.end
    );
    println!("  image end    {:#x}", layout.image_end);
    println!("  dynamic      {:#x}", layout.dynamic);
    println!(
        "  init_array   {:#x}..{:#x}",
        layout.init_array.start, layout.init_array.end
    );
    println!("  ttbr/tcr     {:#x}", layout.translation_registers);
    match image.ini1 {
        Some(ini1) => println!(
            "INI1           {:#x}, {:#x} bytes, {} processes",
            ini1.address, ini1.size, ini1.process_count
        ),
        None => println!("INI1           none"),
    }
    println!("loader entry   {:#x}", image.loader_entry);
}

fn boot(image: KernelImage, flags: &[String]) -> Result<(), String> {
    let mut params = BootParams::default();
    let mut timeout = None;
    let mut show_window = false;
    let mut symbols = SymbolTable::default();
    let mut keys = KeySet::default();
    let (mut trace_smc, mut trace_svc, mut trace_devices) = (false, false, false);
    let (mut trace_ipc, mut trace_times, mut verbose) = (false, false, false);
    let mut survey = false;
    let mut cpu_slowdown = None;
    let (mut breakpoints, mut tracepoints) = (Vec::new(), Vec::new());
    let mut pokes = Vec::new();
    let mut steps_after_breakpoint = 0;
    let (mut emmc_user, mut emmc_boot) = (None, [None, None]);
    let mut emmc_writable = false;
    let mut screenshot = None;
    let mut inputs = Vec::new();
    let mut timed_screenshots = Vec::new();
    if maxwell::metal::available() {
        maxwell::select_renderer(maxwell::RendererKind::Metal);
    }

    let mut flags = flags.iter();
    while let Some(flag) = flags.next() {
        let mut value = || flags.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--dram" => {
                params.dram = match value()?.as_str() {
                    "4" => DramSize::Gb4,
                    "6" => DramSize::Gb6,
                    "8" => DramSize::Gb8,
                    other => return Err(format!("--dram: {other} is not 4, 6 or 8")),
                }
            }
            "--seed" => {
                params.rng_seed = value()?
                    .parse()
                    .map_err(|error| format!("--seed: {error}"))?
            }
            "--cpu-slowdown" => {
                let value = value()?;
                let (factor, seconds) = match value.split_once(':') {
                    Some((factor, seconds)) => (factor, Some(seconds)),
                    None => (value.as_str(), None),
                };
                let factor = factor.parse().ok().filter(|&factor: &u32| factor >= 1);
                let seconds = seconds.map(|seconds| {
                    seconds
                        .parse()
                        .ok()
                        .and_then(|seconds| Duration::try_from_secs_f64(seconds).ok())
                });
                cpu_slowdown = match (factor, seconds) {
                    (Some(factor), None) => Some((factor, None)),
                    (Some(factor), Some(Some(span))) => Some((factor, Some(span))),
                    _ => return Err("--cpu-slowdown: expected N or N:SECONDS, N from 1 up".into()),
                }
            }
            "--timeout" => {
                let seconds = value()?
                    .parse()
                    .map_err(|error| format!("--timeout: {error}"))?;
                let limit = Duration::try_from_secs_f64(seconds)
                    .map_err(|error| format!("--timeout: {error}"))?;
                timeout = Some(limit);
            }
            "--keys" => {
                keys = KeySet::load(Path::new(value()?)).map_err(|error| error.to_string())?
            }
            "--renderer" => match value()?.as_str() {
                "software" => maxwell::select_renderer(maxwell::RendererKind::Software),
                "metal" => {
                    if !maxwell::metal::available() {
                        return Err("--renderer metal: no Metal device".into());
                    }
                    maxwell::select_renderer(maxwell::RendererKind::Metal)
                }
                other => return Err(format!("--renderer: unknown renderer {other}")),
            },
            "--gpu-capture" => {
                let path = value()?;
                maxwell::capture::start(Path::new(path))
                    .map_err(|error| format!("--gpu-capture {path}: {error}"))?;
            }
            "--screenshot" => screenshot = Some(value()?),
            "--screenshot-at" => {
                let value = value()?;
                let (seconds, path) = value
                    .split_once(':')
                    .ok_or("--screenshot-at: expected SECONDS:FILE")?;
                let at = seconds
                    .parse()
                    .ok()
                    .and_then(|seconds| Duration::try_from_secs_f64(seconds).ok())
                    .ok_or("--screenshot-at: expected SECONDS:FILE")?;
                timed_screenshots.push((at, path.to_string()));
            }
            "--window" => show_window = true,
            "--touch" => inputs.extend(parse_touch(value()?)?),
            "--joycons" => {
                let value = value()?;
                let (seconds, seated) = match value.strip_suffix(":off") {
                    Some(seconds) => (seconds, false),
                    None => (value.as_str(), true),
                };
                let at = seconds
                    .parse()
                    .ok()
                    .and_then(|seconds| Duration::try_from_secs_f64(seconds).ok())
                    .ok_or("--joycons: expected SECONDS or SECONDS:off")?;
                inputs.push((at, Input::Seat(seated)));
            }
            "--press" => inputs.extend(parse_press(value()?)?),
            "--emmc" => emmc_user = Some(value()?),
            "--emmc-boot0" => emmc_boot[0] = Some(value()?),
            "--emmc-boot1" => emmc_boot[1] = Some(value()?),
            "--emmc-writable" => emmc_writable = true,
            "--symbols" => {
                symbols =
                    SymbolTable::load(Path::new(value()?)).map_err(|error| error.to_string())?
            }
            "--trace-smc" => trace_smc = true,
            "--trace-svc" => trace_svc = true,
            "--trace-ipc" => trace_ipc = true,
            "--trace-times" => trace_times = true,
            "--verbose" => verbose = true,
            "--trace-devices" => trace_devices = true,
            "--survey" => survey = true,
            "--break" => breakpoints.push(value()?),
            "--trace" => tracepoints.push(value()?),
            "--poke" => pokes.push(value()?),
            "--step" => {
                steps_after_breakpoint = value()?
                    .parse()
                    .map_err(|error| format!("--step: {error}"))?
            }
            "--debug-mode" => params.development_function_enabled = true,
            _ => return Err(USAGE.into()),
        }
    }
    // Names can only be looked up once every --symbols has been seen.
    let locate = |places: Vec<&String>| {
        places
            .into_iter()
            .map(|place| locate_place(place, &symbols))
            .collect::<Result<Vec<Place>, String>>()
    };
    let (breakpoints, tracepoints) = (locate(breakpoints)?, locate(tracepoints)?);
    let pokes = pokes
        .into_iter()
        .map(|poke| parse_poke(poke, &symbols))
        .collect::<Result<Vec<Poke>, String>>()?;
    // Whether an image is opened for writing depends on a flag that may
    // come after it.
    let open_image = |path: &String| {
        OpenOptions::new()
            .read(true)
            .write(emmc_writable)
            .open(path)
            .map_err(|error| format!("{path}: {error}"))
    };
    let [boot0, boot1] = emmc_boot.map(|path| path.map(open_image).transpose());
    let emmc = match emmc_user {
        Some(user) => Some(EmmcImages {
            user: open_image(user)?,
            boot: [boot0?, boot1?],
            writable: emmc_writable,
        }),
        None if emmc_boot.iter().all(Option::is_none) => None,
        None => return Err("eMMC boot partition images need --emmc for the module itself".into()),
    };
    // A machine somebody is watching runs for as long as they watch.
    let timeout = timeout.or((!show_window).then_some(Duration::from_secs(5)));

    let machine = Machine::new(image, &params, keys, emmc, Box::new(io::stderr()))
        .map_err(|error| error.to_string())?;
    let mut window = show_window.then(ScreenWindow::open).transpose()?;
    let stop = AtomicBool::new(false);
    let log = LogOverlay::default();
    let tracing = Tracing {
        symbols: &symbols,
        svc: trace_svc,
        ipc: (trace_ipc || verbose).then(Mutex::default),
        ipc_lines: trace_ipc,
        smc: trace_smc,
        verbose,
        times: (trace_times || verbose).then(Instant::now),
        overlay: window.is_some().then_some(&log),
    };
    let cpu_slowdown = cpu_slowdown.unwrap_or((10, Some(Duration::from_secs(12))));
    let options = RunOptions {
        timeout,
        stop: &stop,
        trace_smc: trace_smc || verbose,
        // Requests and answers are read off the supervisor calls that carry
        // them.
        trace_svc: trace_svc || trace_ipc || verbose,
        trace_devices,
        survey,
        cpu_slowdown: cpu_slowdown.0,
        cpu_slowdown_for: cpu_slowdown.1,
        breakpoints: &breakpoints,
        tracepoints: &tracepoints,
        pokes: &pokes,
        steps_after_breakpoint,
        trace: &|event| print_event(event, &tracing),
    };
    // What the screen was last said to show: panel awake, backlight lit.
    let mut shown = None;
    let mut watch_screen = || {
        if !verbose {
            return;
        }
        let now = machine
            .screen()
            .ok()
            .map(|screen| (screen.panel_on, screen.backlight_on));
        if now != shown {
            let state = |on| if on { "on" } else { "off" };
            let line = match now {
                Some((panel, backlight)) => format!(
                    "screen: panel {}, backlight {}",
                    state(panel),
                    state(backlight)
                ),
                None => "screen: no picture".into(),
            };
            narrate(&line, &tracing);
            shown = now;
        }
    };
    // What is done to the console, in time order.
    inputs.sort_by_key(|&(at, _)| at);
    let mut inputs = inputs.into_iter().peekable();
    timed_screenshots.sort_by_key(|&(at, _)| at);
    let mut timed_screenshots = timed_screenshots.into_iter().peekable();
    let started = Instant::now();
    let report = thread::scope(|scope| {
        // A window needs this thread, so the machine gets another.
        let run = scope.spawn(|| machine.run(options));
        while !run.is_finished() {
            watch_screen();
            match &mut window {
                Some(window) => {
                    if !window.show(&machine, &log) {
                        stop.store(true, Ordering::SeqCst);
                        break;
                    }
                }
                None => thread::sleep(Duration::from_millis(20)),
            }
            let now = started.elapsed();
            while let Some((at, input)) = inputs.next_if(|&(at, _)| at <= now) {
                match input {
                    Input::Seat(seated) => machine.seat_joycons(seated),
                    Input::Press(button, held) => machine.press(button, held),
                    Input::Touch(finger) => {
                        if !machine.set_touch(finger)
                            && let Some((x, y)) = finger
                        {
                            let seconds = at.as_secs_f64();
                            narrate(
                                &format!(
                                    "touch at {x},{y} after {seconds:.3} s: the touchscreen is not scanning yet"
                                ),
                                &tracing,
                            );
                        }
                    }
                }
            }
            while let Some((at, path)) = timed_screenshots.next_if(|(at, _)| *at <= now) {
                let line = match machine.screen() {
                    Ok(screen) => match File::create(&path)
                        .map_err(|error| error.to_string())
                        .and_then(|file| {
                            write_png(file, &screen.frame).map_err(|error| error.to_string())
                        }) {
                        Ok(()) => format!("screen at {:.3} s saved to {path}", at.as_secs_f64()),
                        Err(error) => format!("{path}: {error}"),
                    },
                    Err(reason) => {
                        format!("screen at {:.3} s: no picture, {reason}", at.as_secs_f64())
                    }
                };
                narrate(&line, &tracing);
            }
        }
        run.join()
            .unwrap_or_else(|payload| panic::resume_unwind(payload))
    });

    let ended = match report.ended {
        RunEnd::Core(core) => format!("machine stopped by core {core}"),
        RunEnd::Timeout => format!(
            "machine stopped by timeout after {:?}",
            timeout.unwrap_or_default()
        ),
        RunEnd::Requested => "machine stopped: the window was closed".into(),
    };
    println!("{ended}");
    log.push(&ended);
    for core in report
        .cores
        .iter()
        .filter(|core| report.ended == RunEnd::Core(core.core))
    {
        log.push(&format!("core {}: {}", core.core, core.stop));
    }
    let code = |address| describe(address, &symbols);
    for core in &report.cores {
        println!("core {}: {}", core.core, core.stop);
        if let Some(state) = core.state {
            println!("  pc  {}", code(state.pc));
            println!("  lr  {}", code(state.lr));
            let process = state.process.map(|process| {
                format!(" ({}, program {:016x})", process.name(), process.program_id)
            });
            println!(
                "  pstate {:#x}  esr_el1 {:#x}  elr_el1 {}  far_el1 {:#x}  context {:#x}{}",
                state.pstate,
                state.esr_el1,
                code(state.elr_el1),
                state.far_el1,
                state.context,
                process.unwrap_or_default()
            );
        }
    }
    println!("secure monitor calls:");
    for (name, count) in &report.smc_counts {
        println!("  {name:<20} {count}");
    }
    for note in machine.device_notes() {
        println!("note: {note}");
    }
    if verbose {
        for channel in machine.gpu_channels() {
            println!("gpu: {channel}");
        }
    }
    let carveouts = machine.secure_monitor().carveouts();
    for (index, carveout) in carveouts
        .iter()
        .enumerate()
        .filter(|(_, carveout)| carveout.size != 0)
    {
        println!(
            "carveout {index}: {:#x} + {:#x}",
            carveout.address, carveout.size
        );
    }
    if let Some(path) = screenshot {
        match machine.screen() {
            Ok(screen) => {
                let file = File::create(path).map_err(|error| format!("{path}: {error}"))?;
                write_png(file, &screen.frame).map_err(|error| format!("{path}: {error}"))?;
                let state = |on| if on { "on" } else { "off" };
                println!(
                    "screen: {}x{}, panel {}, backlight {}",
                    screen.frame.width,
                    screen.frame.height,
                    state(screen.panel_on),
                    state(screen.backlight_on)
                );
            }
            Err(reason) => println!("screen: no picture, {reason}"),
        }
    }
    if let Some(window) = &mut window {
        window.set_stopped();
        while window.show(&machine, &log) {}
    }
    Ok(())
}

fn write_png(file: File, frame: &Frame) -> Result<(), png::EncodingError> {
    let mut encoder = png::Encoder::new(io::BufWriter::new(file), frame.width, frame.height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header()?.write_image_data(&frame.rgb())
}

/// Something done to the console at a moment of the run.
#[derive(Clone, Copy)]
enum Input {
    /// Where the finger on the screen then is (`None`: lifted).
    Touch(Option<(u16, u16)>),
    /// Both Joy-Con slide onto the rails, or off them.
    Seat(bool),
    /// A button of the Joy-Con is held, or let go of.
    Press(Button, bool),
}

/// When something starts and when, `HOLD_MS` later, it ends: the parts of
/// `SECONDS:WHAT[:HOLD_MS]`, with `WHAT` left as it is.
fn parse_timed(timed: &str) -> Option<(Duration, &str, Duration)> {
    let mut parts = timed.split(':');
    let at = Duration::try_from_secs_f64(parts.next()?.parse().ok()?).ok()?;
    let what = parts.next()?;
    let hold = match parts.next() {
        Some(milliseconds) => milliseconds.parse().ok()?,
        None => 100,
    };
    parts
        .next()
        .is_none()
        .then_some((at, what, at + Duration::from_millis(hold)))
}

/// `--touch SECONDS:X,Y[:HOLD_MS]`: when the finger goes down where, and
/// when it lifts.
fn parse_touch(touch: &str) -> Result<[(Duration, Input); 2], String> {
    parse_timed(touch)
        .and_then(|(at, place, until)| {
            let (x, y) = place.split_once(',')?;
            let place: (u16, u16) = (x.parse().ok()?, y.parse().ok()?);
            (place.0 < 1280 && place.1 < 720)
                .then_some([(at, Input::Touch(Some(place))), (until, Input::Touch(None))])
        })
        .ok_or_else(|| format!("--touch {touch}: not SECONDS:X,Y[:HOLD_MS] on the 1280x720 screen"))
}

/// `--press SECONDS:BUTTON[:HOLD_MS]`: when the button goes down, and when
/// it comes up.
fn parse_press(press: &str) -> Result<[(Duration, Input); 2], String> {
    parse_timed(press)
        .and_then(|(at, button, until)| {
            let button = Button::named(button)?;
            Some([
                (at, Input::Press(button, true)),
                (until, Input::Press(button, false)),
            ])
        })
        .ok_or_else(|| format!("--press {press}: not SECONDS:BUTTON[:HOLD_MS]"))
}

/// A kernel image address given as `0x…` or as a function name, or a
/// virtual address given as `@0x…`.
fn locate_place(place: &str, symbols: &SymbolTable) -> Result<Place, String> {
    let hex = |hex| u64::from_str_radix(hex, 16).map_err(|error| format!("{place}: {error}"));
    if let Some(address) = place.strip_prefix("@0x") {
        return hex(address).map(Place::Virtual);
    }
    match place.strip_prefix("0x") {
        Some(address) => hex(address).map(Place::Image),
        None => symbols
            .address_of(place)
            .map(Place::Image)
            .ok_or_else(|| format!("{place}: no such function in the symbols (see --symbols)")),
    }
}

/// A `--poke` argument: `WHERE:xN+OFFSET:SIZE:VALUE`.
fn parse_poke(text: &str, symbols: &SymbolTable) -> Result<Poke, String> {
    let malformed = || format!("{text}: expected WHERE:xN+OFFSET:SIZE:VALUE");
    let number = |digits: &str| {
        match digits.strip_prefix("0x") {
            Some(hex) => u64::from_str_radix(hex, 16),
            None => digits.parse(),
        }
        .map_err(|_| malformed())
    };
    let [place, target, size, value] = text.split(':').collect::<Vec<_>>()[..] else {
        return Err(malformed());
    };
    let (register, offset) = target.split_once('+').unwrap_or((target, "0"));
    let base = register
        .strip_prefix('x')
        .and_then(|index| index.parse().ok())
        .filter(|&index: &u8| index <= 30)
        .ok_or_else(malformed)?;
    let size = u8::try_from(number(size)?)
        .ok()
        .filter(|size| (1..=8).contains(size));
    Ok(Poke {
        place: locate_place(place, symbols)?,
        base,
        offset: number(offset)? as i64,
        size: size.ok_or_else(malformed)?,
        value: number(value)?,
    })
}

/// What the tracing flags ask of [`print_event`].
struct Tracing<'a> {
    symbols: &'a SymbolTable,
    /// Print supervisor calls and their returns.
    svc: bool,
    /// Print requests and answers; this follows the handles they use.
    ipc: Option<Mutex<Tracker>>,
    /// Print what the tracker says about each request and answer.
    ipc_lines: bool,
    /// Print secure monitor calls.
    smc: bool,
    /// Print the boot's milestones.
    verbose: bool,
    /// Prefix every line with the seconds since this moment.
    times: Option<Instant>,
    /// Where every printed line also goes, to be shown in the window.
    overlay: Option<&'a LogOverlay>,
}

/// Prints the trace lines for one event to stderr. A closed stderr must not
/// take the machine down, so write errors are dropped.
fn print_event(event: TraceEvent, tracing: &Tracing) {
    if tracing.times.is_none() && tracing.overlay.is_none() {
        let _ = write_event(&mut io::stderr().lock(), event, tracing);
        return;
    }
    let mut lines = Vec::new();
    let _ = write_event(&mut lines, event, tracing);
    for line in String::from_utf8_lossy(&lines).lines() {
        narrate(line, tracing);
    }
}

/// Prints one line of the emulator's own to stderr, timed and copied to
/// the window as the tracing flags ask.
fn narrate(line: &str, tracing: &Tracing) {
    let line = match tracing.times {
        Some(start) => format!("{:9.4} {line}", start.elapsed().as_secs_f64()),
        None => line.to_string(),
    };
    let _ = writeln!(io::stderr().lock(), "{line}");
    if let Some(overlay) = tracing.overlay {
        overlay.push(line.trim_start());
    }
}

/// At most this much of a `Break` call's reason data is printed.
const SHOWN_BREAK_DATA: usize = 96;

fn write_event(out: &mut impl Write, event: TraceEvent, tracing: &Tracing) -> io::Result<()> {
    let symbols = tracing.symbols;
    match event {
        TraceEvent::SecureMonitorCall {
            core,
            immediate,
            arguments,
            results,
            effect,
        } => {
            if tracing.verbose
                && let SmcEffect::StartCore { core: started, .. } = effect
            {
                writeln!(out, "core {started} started by core {core}")?;
            }
            if !tracing.smc {
                return Ok(());
            }
            writeln!(
                out,
                "core {core}: smc #{immediate} {} {:x?} -> {effect:x?} {:x?}",
                secmon::function_name(immediate, arguments[0] as u32),
                arguments,
                results
            )
        }
        TraceEvent::SupervisorCall {
            core,
            id,
            arguments,
            pc,
            caller,
            process,
            data,
            buffers,
        } => {
            if tracing.svc {
                let name = svc::name(id).unwrap_or("?");
                let who = describe_caller(caller, process);
                write!(
                    out,
                    "core {core}: svc {id:#04x} {name} {arguments:x?} from {pc:#x} ({who})"
                )?;
                match id {
                    svc::OUTPUT_DEBUG_STRING => {
                        write!(out, " {:?}", String::from_utf8_lossy(data))?
                    }
                    svc::CONNECT_TO_NAMED_PORT => write!(out, " {:?}", svc::text(data))?,
                    svc::CREATE_PROCESS => {
                        if let Some((name, program_id)) = svc::created_process(data) {
                            write!(out, " {name:?} program {program_id:016x}")?;
                        }
                    }
                    svc::BREAK if !data.is_empty() => {
                        write!(out, " {:02x?}", &data[..data.len().min(SHOWN_BREAK_DATA)])?
                    }
                    _ => {}
                }
                writeln!(out)?;
            }
            if tracing.verbose {
                write_milestone(out, id, process, data)?;
            }
            let line = tracing.ipc.as_ref().and_then(|tracker| {
                tracker
                    .lock()
                    .unwrap()
                    .call(client(caller), id, &arguments, data, buffers)
            });
            if tracing.verbose
                && let Some(line) = &line
            {
                write_request_milestone(out, process, line)?;
            }
            match tracing.ipc_lines {
                true => write_ipc(out, core, caller, process, line),
                false => Ok(()),
            }
        }
        TraceEvent::SupervisorReturn {
            core,
            id,
            arguments: _,
            results,
            caller,
            process,
            data,
            buffers,
        } => {
            if tracing.svc {
                let name = svc::name(id).unwrap_or("?");
                let who = describe_caller(caller, process);
                writeln!(
                    out,
                    "core {core}: svc {id:#04x} {name} -> {results:x?} ({who})"
                )?;
            }
            let line = tracing.ipc.as_ref().and_then(|tracker| {
                tracker
                    .lock()
                    .unwrap()
                    .returned(client(caller), id, &results, data, buffers)
            });
            match tracing.ipc_lines {
                true => write_ipc(out, core, caller, process, line),
                false => Ok(()),
            }
        }
        TraceEvent::DeviceAccess {
            core,
            device,
            address,
            size,
            is_write,
            value,
        } => writeln!(
            out,
            "core {core}: {device} {address:#x} {} {value:#0width$x}",
            if is_write { "<-" } else { "->" },
            width = usize::from(size) * 2 + 2
        ),
        TraceEvent::Unmodeled {
            core,
            device,
            address,
            size,
            is_write,
        } => writeln!(
            out,
            "core {core}: survey: {size}-byte {} {} at {address:#x} is not modeled",
            if is_write { "write to" } else { "read of" },
            device.map_or("nothing".into(), |device| format!("{device} register"))
        ),
        TraceEvent::Reached {
            core,
            at,
            registers,
        } => {
            writeln!(
                out,
                "core {core}: reached {} {registers:x?}",
                describe(at, symbols)
            )
        }
        TraceEvent::Stepped {
            core,
            at,
            exception_entry,
        } => writeln!(
            out,
            "core {core}: stepped to {}{}",
            describe(at, symbols),
            if exception_entry {
                " (exception entry)"
            } else {
                ""
            }
        ),
    }
}

/// The name a process goes by in the boot narration.
fn process_name(process: Option<Process>) -> String {
    process.map_or("a process".into(), |process| process.name().to_string())
}

/// Narrates the supervisor calls that are milestones of a boot.
fn write_milestone(
    out: &mut impl Write,
    id: u16,
    process: Option<Process>,
    data: &[u8],
) -> io::Result<()> {
    let who = process_name(process);
    match id {
        svc::CREATE_PROCESS => {
            let Some((name, program_id)) = svc::created_process(data) else {
                return Ok(());
            };
            let address = data.get(0x18..).and_then(|rest| rest.first_chunk());
            let address = address.map_or(0, |bytes| u64::from_le_bytes(*bytes));
            writeln!(
                out,
                "program {name} ({program_id:016x}) loaded at {address:#x}"
            )
        }
        svc::EXIT_PROCESS => writeln!(out, "{who} exited"),
        svc::BREAK => writeln!(
            out,
            "{who} stopped itself (Break) {:02x?}",
            &data[..data.len().min(SHOWN_BREAK_DATA)]
        ),
        svc::OUTPUT_DEBUG_STRING => {
            writeln!(out, "{who} says {:?}", String::from_utf8_lossy(data))
        }
        _ => Ok(()),
    }
}

/// Narrates the requests that are milestones of a boot, from the tracker's
/// wording of them: a service being registered with the service manager,
/// and a process reporting a fatal error (the result code is the first data
/// word of that request).
fn write_request_milestone(
    out: &mut impl Write,
    process: Option<Process>,
    line: &str,
) -> io::Result<()> {
    let who = process_name(process);
    if let Some(rest) = line.strip_prefix("-> sm: tipc request 2 name \"")
        && let Some((service, _)) = rest.split_once('"')
    {
        return writeln!(out, "{who} registers service {service}");
    }
    if let Some(rest) = line.strip_prefix("-> fatal:u cmif request ") {
        let code = rest
            .split_once("data [")
            .and_then(|(_, words)| words.split([',', ']']).next())
            .and_then(|word| u32::from_str_radix(word, 16).ok());
        return match code {
            Some(code) => writeln!(out, "{who} reports a fatal error {}", ResultCode(code)),
            None => writeln!(out, "{who} reports a fatal error"),
        };
    }
    Ok(())
}

/// Prints what the tracker has to say about a call or return, if anything.
fn write_ipc(
    out: &mut impl Write,
    core: usize,
    caller: Caller,
    process: Option<Process>,
    line: Option<String>,
) -> io::Result<()> {
    let Some(line) = line else {
        return Ok(());
    };
    let thread = caller.thread;
    match process {
        Some(process) => write!(out, "core {core}: ipc {}", process.name())?,
        None => write!(out, "core {core}: ipc context {:#x}", caller.context)?,
    }
    writeln!(out, " {line} (thread {thread:#x})")
}

fn client(caller: Caller) -> Client {
    Client {
        process: caller.context,
        thread: caller.thread,
    }
}

/// `boot, context 0x7, thread 0x…`, without the name when it is not known.
fn describe_caller(caller: Caller, process: Option<Process>) -> String {
    let Caller { context, thread } = caller;
    let name = process.map(|process| format!("{}, ", process.name()));
    format!(
        "{}context {context:#x}, thread {thread:#x}",
        name.unwrap_or_default()
    )
}

/// `0xffff… (image 0x8006… Function+0x10)`, with as much of that as is known.
fn describe(code: CodeAddress, symbols: &SymbolTable) -> String {
    let Some(image) = code.image else {
        return format!("{:#x}", code.address);
    };
    let symbol = symbols
        .resolve(image)
        .map(|symbol| format!(" {symbol}"))
        .unwrap_or_default();
    if image == code.address {
        format!("{image:#x}{symbol}")
    } else {
        format!("{:#x} (image {image:#x}{symbol})", code.address)
    }
}
