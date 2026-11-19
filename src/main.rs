mod control;
mod homebrew_options;
mod script;
mod window;

use std::{
    fs::{File, OpenOptions},
    io::{self, Write},
    panic,
    path::{Path, PathBuf},
    process::ExitCode,
    sync::{
        Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use horizon_emu::{
    arm64::{
        exception_class,
        exception_class::{DATA_ABORT_LOWER, INSTRUCTION_ABORT_LOWER},
    },
    boot::{BootParams, DramSize},
    bootrom,
    image::KernelImage,
    ipc::{Client, ResultCode, Tracker},
    keys::KeySet,
    machine::{
        Caller, CodeAddress, ExceptionState, Firmware, Machine, Module, Place, Poke, PowerOn,
        Process, RunEnd, RunOptions, SaveReport, TraceEvent,
    },
    maxwell, nand, net, sdcard,
    secmon::{self, SmcEffect},
    soc::{dc::Frame, emmc::EmmcImages, joycon::Button, sdcard::SdCardImage, wifi_ap::AccessPoint},
    svc,
    symbols::SymbolTable,
};

use crate::{
    homebrew_options::HomebrewOptions,
    script::{Event, Script},
    window::{LogOverlay, ScreenWindow},
};

const USAGE: &str = "\
usage: horizon-emu info <kernel.elf>
       horizon-emu boot <kernel.elf> [options]
       horizon-emu power-on [options]
       horizon-emu restore <snapshot dir> [options]
       horizon-emu format-emmc <new image file> [--keys FILE [--firmware DIR
                   [--boot0 <new file> --boot1 <new file>]]]
       horizon-emu install-boot <image file> --keys FILE --firmware DIR
                   --boot0 FILE --boot1 FILE
       horizon-emu patch-prodinfo <image file> --keys FILE
       horizon-emu format-sd <new image file> --size SIZE [--add HOST=CARD ...]

<kernel.elf> is the package2 kernel wrapped as an ELF (see pkg2elf.py).
boot runs until a core needs something that is not emulated, the kernel
panics, the timeout expires, or its window is closed. power-on starts the
console the way it powers on instead, from its boot ROM (high-level
emulated): the ROM boots package1 from the BCT in --emmc-boot0 and
--emmc-boot1 (see format-emmc --boot0), the boot processor (BPMP) runs
package1's bootloader instruction by instruction, and once it starts the
CPU the secure monitor's cold boot takes the kernel from package2 in the
--emmc user area; breakpoints and symbols apply once that kernel is in
place. It needs --keys with the console's secure_boot_key, and takes the
options of boot except those that patch the kernel (--homebrew-keys,
--homebrew: the real package2 runs as it is). format-emmc creates
an eMMC user area image for --emmc (a sparse file): the 32 GB module's
partition table; with --keys also the empty, encrypted file systems of a
console's partitions and the calibration data of this emulator's own
console (its serial number, addresses and sensor values are made up for
it; it has no device keys or Nintendo-issued certificates, only a made-up
TLS client key and self-signed certificate, which Nintendo's servers
reject); with --firmware also the contents (*.nca) of that directory,
installed as the system's programs and data, and package2 of its boot
image packages in the BCPKG2 partitions; with --boot0 and --boot1 also the
images of the two boot partitions for --emmc-boot0/--emmc-boot1 (sparse
files), holding its BCTs and package1. install-boot puts those boot
images of the firmware DIR into an existing user area image (its BCPKG2
partitions, nothing else, after rewriting a partition table of 128
entries in the eleven-entry form the NX bootloader accepts; partition
contents untouched) and into the boot partition images FILE (made
if missing). patch-prodinfo gives the calibration data of an image
formatted before it had one that made-up TLS client identity, and
changes nothing else. format-sd creates a
microSD card image for --sd (a sparse file) of SIZE bytes (suffix K, M or
G for binary multiples; a multiple of 512K above 2088M and up to 32G),
formatted as the console formats a card, with each --add's host file HOST
at the absolute path CARD (/hbmenu.nro, /switch/foo.nro), directories
created as needed.

restore builds the machine a snapshot was taken of (see the control
socket's save_state) and goes on from where it was: the same kernel, keys,
DRAM, seed, storage images as they were then, WiFi network and homebrew
options, which the snapshot recorded and which cannot be given again; the
other options (window, control, scripts, tracing, --dns, --forward...)
apply as for boot. The snapshot itself is never changed, so it can be
restored any number of times: images the saved machine wrote to are
cloned for the restored one into
  --workdir DIR       (default: a new directory under the system's
                      temporary directory, printed at start)
Host connections of the saved machine are gone, as after a dropped link;
the console makes new ones. The processors run at full speed unless
--cpu-slowdown says otherwise.

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
  --sd FILE           raw image of a microSD card, in the slot from power
                      on (see format-sd); without it the slot is empty.
                      Writes stay in memory
  --sd-writable       write to the microSD card image instead
  --wifi-ssid NAME    an access point in range of the console's WiFi radio,
                      with network NAME on 2.4 GHz channel 6, and behind it
                      a router with DHCP, DNS and NAT to the host's
                      internet connection (the console can go online); an
                      open network unless
  --wifi-passphrase P makes it WPA2-Personal (CCMP) with passphrase P
                      (8 to 63 printable ASCII characters)
  --forward HOST_PORT[:CONSOLE_PORT]
                      with --wifi-ssid: TCP connections to HOST_PORT of this
                      machine's 127.0.0.1 reach CONSOLE_PORT (the same number
                      unless given) of the console, which sees them come
                      from the router. The only way in: the router lets
                      nothing else through to the console. May be given
                      more than once
  --dns ADDRESS[:PORT]
                      with --wifi-ssid: the router's DNS server asks this
                      resolver (port 53 unless given) instead of the
                      host's. A resolver that answers the console's
                      connection test host (ctest.cdn.nintendo.net) with
                      the address of a web server puts that server in front
                      of the console as a captive portal
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
                      right lstick rstick, or power, the console's own
                      button: it puts the console to sleep and wakes it
  --script FILE       do to the console what FILE says, a step per line, each
                      when the one before is over (# starts a comment):
                        wait X,Y RRGGBB      until that pixel of the screen
                                             (1280x720) has that colour
                        sleep SECONDS
                        press BUTTON [HOLD_MS]    (default 300)
                        touch X,Y [HOLD_MS]
                        type TEXT            the rest of the line, key by key
                                             on the open on-screen keyboard
                        joycons on|off
                        screenshot FILE
                      A press or touch line may end in
                        until X,Y RRGGBB [SECONDS]
                      and is then done again every SECONDS (default 3) until
                      that pixel has that colour. Unlike the times of --press
                      and --touch, none of this depends on how fast the host
                      gets the guest anywhere
  --debug-mode        report development functions as enabled to the kernel,
                      which turns on its debug log (UART-A, shown on stderr)

tracing and debugging (output on stderr; WHERE is a kernel image address
as 0x..., a function name from --symbols, or @0x... for a virtual address
taken as it is, such as one inside a process):
  --trace-smc         print every secure monitor call
  --trace-svc         print every supervisor call and its return, with the
                      process and thread making it (threads are told apart
                      by the address of their thread-local region). With
                      this, --trace-ipc or --verbose, the report at the end
                      lists the calls not returned from when the machine
                      stopped: what each waiting thread waits in, with its
                      return addresses from the frame records on its stack
  --trace-ipc         print every request a process sends to a service and
                      the answer to it: service or session, kind and command,
                      data words (which for the spl services include key
                      material), handles, buffers with the first 32 bytes of
                      each (not for spl and csrng) and the result code
  --trace-devices     print every device register access
  --trace-bpmp N      with power-on: when the boot processor stops, add its
                      last N instructions and device accesses to the note
  --trace-times       start every trace line with the seconds since the
                      machine was started
  --verbose           narrate the boot, with times: cores started, each
                      program loaded (name, program ID, load address),
                      services registered, processes exiting or stopping
                      themselves, fatal errors reported, kernel debug text.
                      Like --trace-svc it watches every supervisor call,
                      which slows the guest
  --crash-dump DIR    each time a process takes an exception the firmware
                      treats as a crash (abort, alignment fault, undefined
                      instruction, BRK), write DIR/crash-NNN-NAME.txt: the
                      fault address and syndrome, PC, LR, SP, PSTATE,
                      X0-X30, the frame-record call chain and the process's
                      modules (base, size, path, build ID), with each code
                      address as module+offset. Watches every supervisor
                      call, like --verbose
  --control SOCKET    serve requests on a Unix socket at SOCKET while the
                      machine runs: script steps, screenshots, pixels,
                      status, stop, save_state (a snapshot of the running
                      machine into a directory, for restore). One JSON
                      object per line
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

/// The whole usage message: the general options and those of homebrew.
fn usage() -> String {
    format!("{USAGE}\n\n{}", homebrew_options::USAGE)
}

fn run(arguments: &[String]) -> Result<(), String> {
    if let [command, flags @ ..] = arguments
        && command == "power-on"
    {
        return boot(None, flags, None);
    }
    let [command, path, flags @ ..] = arguments else {
        return Err(usage());
    };
    if command == "format-emmc" {
        return format_emmc(path, flags).map_err(|error| format!("{path}: {error}"));
    }
    if command == "install-boot" {
        return install_boot(path, flags).map_err(|error| format!("{path}: {error}"));
    }
    if command == "format-sd" {
        return format_sd(path, flags).map_err(|error| format!("{path}: {error}"));
    }
    if command == "patch-prodinfo" {
        return patch_prodinfo(path, flags).map_err(|error| format!("{path}: {error}"));
    }
    if command == "restore" {
        return restore(Path::new(path), flags);
    }
    let image = KernelImage::load(Path::new(path)).map_err(|error| error.to_string())?;
    match command.as_str() {
        "info" if flags.is_empty() => {
            info(&image);
            Ok(())
        }
        "boot" => boot(Some((image, Path::new(path))), flags, None),
        _ => Err(usage()),
    }
}

/// Flags that make the machine what it is, and whether each takes a
/// value. A snapshot records them; a restored machine is built with them
/// again, and they cannot be given to `restore`.
const MACHINE_FLAGS: &[(&str, bool)] = &[
    ("--dram", true),
    ("--seed", true),
    ("--keys", true),
    ("--emmc", true),
    ("--emmc-boot0", true),
    ("--emmc-boot1", true),
    ("--emmc-writable", false),
    ("--sd", true),
    ("--sd-writable", false),
    ("--wifi-ssid", true),
    ("--wifi-passphrase", true),
    ("--debug-mode", false),
    ("--disable-program-verification", false),
    ("--homebrew-keys", false),
    ("--replace-program", true),
    ("--homebrew", true),
    ("--homebrew-in", true),
    ("--hbl", true),
    ("--homebrew-app", true),
];

/// Machine flags naming host files that are read when the machine is
/// built, recorded as absolute paths.
const FILE_FLAGS: &[&str] = &["--keys", "--homebrew", "--hbl", "--homebrew-app"];

/// Storage image flags: a snapshot holds a copy of each image, under the
/// name given, and the flag that makes the image written to.
const IMAGE_FLAGS: &[(&str, &str, &str)] = &[
    ("--emmc", "emmc.img", "--emmc-writable"),
    ("--emmc-boot0", "emmc-boot0.img", "--emmc-writable"),
    ("--emmc-boot1", "emmc-boot1.img", "--emmc-writable"),
    ("--sd", "sd.img", "--sd-writable"),
];

/// `boot.json` of a snapshot: how to build the machine again.
#[derive(serde::Serialize, serde::Deserialize)]
struct SavedBoot {
    /// Machine flags with their values; images by their names in the
    /// snapshot.
    flags: Vec<String>,
    /// A machine powered on (`power-on`): no `kernel.elf` beside it, the
    /// kernel is part of its state.
    #[serde(default)]
    power_on: bool,
}

/// Takes snapshots of a running machine: its state, a copy of the kernel
/// (of a machine started at kernel entry) and of each storage image, and
/// how it was built.
struct Snapshotter {
    kernel: Option<PathBuf>,
    /// Machine flags as given, with host files as absolute paths.
    flags: Vec<String>,
}

impl Snapshotter {
    fn save(&self, machine: &Machine, dir: &Path) -> Result<SaveReport, String> {
        let mut beside = |dir: &Path| -> io::Result<()> {
            let mut saved = SavedBoot {
                flags: Vec::new(),
                power_on: self.kernel.is_none(),
            };
            let mut flags = self.flags.iter();
            while let Some(flag) = flags.next() {
                saved.flags.push(flag.clone());
                let Some(&(_, name, _)) = IMAGE_FLAGS.iter().find(|(image, ..)| image == flag)
                else {
                    continue;
                };
                let source = flags
                    .next()
                    .expect("an image flag was recorded with its value");
                clone_file(Path::new(source), &dir.join(name))?;
                saved.flags.push(name.into());
            }
            if let Some(kernel) = &self.kernel {
                clone_file(kernel, &dir.join("kernel.elf"))?;
            }
            let json = serde_json::to_vec_pretty(&saved).map_err(io::Error::other)?;
            std::fs::write(dir.join("boot.json"), json)
        };
        machine
            .save_state(dir, &mut beside)
            .map_err(|error| format!("{}: {error}", dir.display()))
    }
}

/// Copies `source` to `target`, replacing it: a clone on APFS, made at once
/// whatever the size, sharing blocks until either is written.
fn clone_file(source: &Path, target: &Path) -> io::Result<()> {
    match std::fs::remove_file(target) {
        Err(error) if error.kind() != io::ErrorKind::NotFound => return Err(error),
        _ => {}
    }
    std::fs::copy(source, target)
        .map(drop)
        .map_err(|error| io::Error::new(error.kind(), format!("{}: {error}", source.display())))
}

/// `restore DIR [options]`: the snapshot's machine flags, its images as
/// the restored machine is to use them, then the options given.
fn restore(dir: &Path, flags: &[String]) -> Result<(), String> {
    let file = dir.join("boot.json");
    let saved: SavedBoot = std::fs::read(&file)
        .map_err(|error| error.to_string())
        .and_then(|bytes| serde_json::from_slice(&bytes).map_err(|error| error.to_string()))
        .map_err(|error| format!("{}: {error}", file.display()))?;
    let mut workdir = None;
    let mut given = Vec::new();
    let mut rest = flags.iter();
    while let Some(flag) = rest.next() {
        if flag == "--workdir" {
            workdir = Some(PathBuf::from(rest.next().ok_or("--workdir needs a value")?));
        } else if MACHINE_FLAGS.iter().any(|(machine, _)| machine == flag) {
            return Err(format!("{flag}: the snapshot says what the machine is"));
        } else {
            given.push(flag.clone());
        }
    }
    let workdir = workdir.unwrap_or_else(|| {
        std::env::temp_dir().join(format!("horizon-emu-restore-{}", std::process::id()))
    });
    let mut all = Vec::new();
    let mut saved_flags = saved.flags.iter();
    while let Some(flag) = saved_flags.next() {
        all.push(flag.clone());
        let Some(&(_, _, writable)) = IMAGE_FLAGS.iter().find(|(image, ..)| image == flag) else {
            continue;
        };
        let name = saved_flags
            .next()
            .ok_or_else(|| format!("{}: {flag} without an image", file.display()))?;
        let image = dir.join(name);
        if saved.flags.iter().any(|flag| flag == writable) {
            // Written to: a clone, so that the snapshot stays as it is.
            std::fs::create_dir_all(&workdir)
                .map_err(|error| format!("--workdir {}: {error}", workdir.display()))?;
            let copy = workdir.join(name);
            clone_file(&image, &copy).map_err(|error| error.to_string())?;
            println!("restore: {flag} {}", copy.display());
            all.push(copy.display().to_string());
        } else {
            all.push(image.display().to_string());
        }
    }
    all.extend(given);
    if saved.power_on {
        return boot(None, &all, Some(dir));
    }
    let kernel = dir.join("kernel.elf");
    let image = KernelImage::load(&kernel).map_err(|error| error.to_string())?;
    boot(Some((image, &kernel)), &all, Some(dir))
}

fn format_sd(path: &str, flags: &[String]) -> Result<(), String> {
    let (mut size, mut files) = (None, Vec::new());
    let mut flags = flags.iter();
    while let Some(flag) = flags.next() {
        let value = flags.next().ok_or_else(usage)?;
        match flag.as_str() {
            "--size" => size = Some(parse_size(value).ok_or("--size: expected bytes, K, M or G")?),
            "--add" => {
                let (host, card) = value.split_once('=').ok_or("--add: expected HOST=CARD")?;
                files.push((host.to_string(), card.to_string()));
            }
            _ => return Err(usage()),
        }
    }
    let size = size.ok_or("format-sd needs --size")?;
    let mut image = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| error.to_string())?;
    sdcard::format_file(&mut image, size, &files)
}

/// A byte count, with an optional binary multiple: `4G`, `512K`.
fn parse_size(text: &str) -> Option<u64> {
    let (digits, shift) = match text.as_bytes().last()? {
        b'K' => (&text[..text.len() - 1], 10),
        b'M' => (&text[..text.len() - 1], 20),
        b'G' => (&text[..text.len() - 1], 30),
        _ => (text, 0),
    };
    digits.parse::<u64>().ok()?.checked_mul(1 << shift)
}

fn format_emmc(path: &str, flags: &[String]) -> Result<(), String> {
    let (mut keys, mut firmware, mut boot) = (None, None, [None, None]);
    let mut flags = flags.iter();
    while let Some(flag) = flags.next() {
        let value = flags.next().ok_or_else(usage)?;
        match flag.as_str() {
            "--keys" => {
                keys = Some(KeySet::load(Path::new(value)).map_err(|error| error.to_string())?)
            }
            "--firmware" => firmware = Some(Path::new(value)),
            "--boot0" => boot[0] = Some(value),
            "--boot1" => boot[1] = Some(value),
            _ => return Err(usage()),
        }
    }
    if firmware.is_some() && keys.is_none() {
        return Err("--firmware needs --keys".into());
    }
    let boot = match boot {
        [None, None] => None,
        [Some(_), Some(_)] if firmware.is_none() => {
            return Err("--boot0 and --boot1 need --firmware".into());
        }
        [Some(boot0), Some(boot1)] => Some([boot0, boot1]),
        _ => return Err("--boot0 and --boot1 go together".into()),
    };
    // Before any file is made: the firmware has to have them.
    let boot_images = match (&keys, firmware) {
        (Some(keys), Some(firmware)) => Some(
            nand::boot_images::BootImages::load(firmware, keys)
                .map_err(|error| error.to_string())?,
        ),
        _ => None,
    };
    let create = |path: &str| {
        OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|error| format!("{path}: {error}"))
    };
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
    if let Some(boot_images) = boot_images {
        let boot = match boot {
            Some([boot0, boot1]) => Some([create(boot0)?, create(boot1)?]),
            None => None,
        };
        install_boot_images(&boot_images, &image, boot.as_ref())?;
    }
    Ok(())
}

fn install_boot(path: &str, flags: &[String]) -> Result<(), String> {
    let (mut keys, mut firmware, mut boot) = (None, None, [None, None]);
    let mut flags = flags.iter();
    while let Some(flag) = flags.next() {
        let value = flags.next().ok_or_else(usage)?;
        match flag.as_str() {
            "--keys" => {
                keys = Some(KeySet::load(Path::new(value)).map_err(|error| error.to_string())?)
            }
            "--firmware" => firmware = Some(Path::new(value)),
            "--boot0" => boot[0] = Some(value),
            "--boot1" => boot[1] = Some(value),
            _ => return Err(usage()),
        }
    }
    let (Some(keys), Some(firmware), [Some(boot0), Some(boot1)]) = (keys, firmware, boot) else {
        return Err("install-boot needs --keys, --firmware, --boot0 and --boot1".into());
    };
    let boot_images =
        nand::boot_images::BootImages::load(firmware, &keys).map_err(|error| error.to_string())?;
    let image = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .map_err(|error| error.to_string())?;
    let open = |path: &str| {
        OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .map_err(|error| format!("{path}: {error}"))
    };
    let boot = [open(boot0)?, open(boot1)?];
    let rewritten = nand::rewrite_partition_table(&image).map_err(|error| error.to_string())?;
    if rewritten {
        println!("partition table rewritten: eleven entries, the partitions untouched");
    }
    install_boot_images(&boot_images, &image, Some(&boot))
}

/// Writes `boot_images` into the BCPKG2 partitions of the user area
/// `image` and, given, the boot partition images `boot`.
fn install_boot_images(
    boot_images: &nand::boot_images::BootImages,
    image: &File,
    boot: Option<&[File; 2]>,
) -> Result<(), String> {
    boot_images
        .write_package2(image)
        .map_err(|error| error.to_string())?;
    if let Some([boot0, boot1]) = boot {
        boot_images
            .write_boot_partitions([boot0, boot1])
            .map_err(|error| format!("boot partitions: {error}"))?;
    }
    let normal = &boot_images.normal;
    println!(
        "boot images: package1 of {}, {} bytes; package2 {} bytes, safe mode {} bytes{}",
        nand::boot_images::package1_build_date(&normal.package1).unwrap_or("?"),
        normal.package1.len(),
        normal.package2.len(),
        boot_images.safe.package2.len(),
        if boot.is_some() {
            "; BCTs and package1 in the boot partitions"
        } else {
            "; boot partitions not written (no --boot0/--boot1)"
        }
    );
    Ok(())
}

fn patch_prodinfo(path: &str, flags: &[String]) -> Result<(), String> {
    let [flag, keys] = flags else {
        return Err(usage());
    };
    if flag != "--keys" {
        return Err(usage());
    }
    let keys = KeySet::load(Path::new(keys)).map_err(|error| error.to_string())?;
    let image = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .map_err(|error| error.to_string())?;
    let device_id = nand::calibration::install_ssl_identity(&image, &keys)
        .map_err(|error| error.to_string())?;
    println!("console {device_id:016X}: made-up TLS client identity installed");
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

/// How a snapshot records the value of machine flag `flag`: files the
/// machine is built from as absolute paths, so that a restore from
/// elsewhere finds them.
fn recorded_value(flag: &str, value: &str) -> String {
    let absolute = |path: &str| {
        std::path::absolute(path)
            .map_or_else(|_| path.to_string(), |path| path.display().to_string())
    };
    if FILE_FLAGS.contains(&flag) || IMAGE_FLAGS.iter().any(|(image, ..)| *image == flag) {
        absolute(value)
    } else if flag == "--replace-program"
        && let Some((program, nsp)) = value.split_once('=')
    {
        format!("{program}={}", absolute(nsp))
    } else {
        value.to_string()
    }
}

/// The console's boot ROM run on its eMMC boot partitions, for
/// `power-on`: what it leaves the boot processor.
fn boot_rom(keys: &KeySet, boot: [Option<&String>; 2]) -> Result<PowerOn, String> {
    let secure_boot_key = keys
        .secure_boot_key()
        .ok_or("power-on needs --keys with the console's secure_boot_key")?;
    let read = |path: Option<&String>| {
        path.map(|path| std::fs::read(path).map_err(|error| format!("{path}: {error}")))
            .transpose()
            .map(Option::unwrap_or_default)
    };
    let (boot0, boot1) = (read(boot[0])?, read(boot[1])?);
    if boot0.is_empty() && boot1.is_empty() {
        return Err("power-on needs --emmc-boot0 or --emmc-boot1 with a BCT and package1".into());
    }
    // The device key fuse and the die's identity fuses belong to one chip,
    // and no key set records them: they are zero here, a choice like the
    // console's serial number. Only the SSK depends on them, which
    // package1ldr never uses.
    let fuses = bootrom::Fuses::retail(*secure_boot_key, 0, bootrom::ChipFuses::default());
    let mut iram = vec![0; bootrom::IRAM_SIZE];
    // PMC_SCRATCH0 is clear on a cold boot: no warm boot, no forced recovery.
    let handoff = bootrom::cold_boot(&boot0, &boot1, &fuses, 0, &mut iram)
        .map_err(|error| format!("boot ROM: {error}"))?;
    Ok(PowerOn {
        handoff,
        iram,
        fuses,
    })
}

/// Boots `kernel` (an image and the file it was read from) as `flags` say,
/// or without one powers the console on from its boot ROM; with `restore`,
/// goes on from the snapshot in that directory instead.
fn boot(
    kernel: Option<(KernelImage, &Path)>,
    flags: &[String],
    restore: Option<&Path>,
) -> Result<(), String> {
    let mut machine_flags = Vec::new();
    let mut params = BootParams::default();
    let mut timeout = None;
    let mut show_window = false;
    let mut symbols = SymbolTable::default();
    let mut keys = KeySet::default();
    let (mut trace_smc, mut trace_svc, mut trace_devices) = (false, false, false);
    let (mut trace_ipc, mut trace_times, mut verbose) = (false, false, false);
    let mut survey = false;
    let mut bpmp_trace = 0;
    let mut crash_dump = None::<PathBuf>;
    let mut control = None::<String>;
    let mut cpu_slowdown = None;
    let (mut breakpoints, mut tracepoints) = (Vec::new(), Vec::new());
    let mut pokes = Vec::new();
    let mut steps_after_breakpoint = 0;
    let (mut emmc_user, mut emmc_boot) = (None, [None, None]);
    let mut emmc_writable = false;
    let (mut sd_card, mut sd_writable) = (None, false);
    let mut screenshot = None;
    let mut inputs = Vec::new();
    let mut timed_screenshots = Vec::new();
    let mut script = None;
    let mut homebrew = HomebrewOptions::default();
    let (mut wifi_ssid, mut wifi_passphrase) = (None::<String>, None::<String>);
    let mut forwards = Vec::new();
    let mut dns = None;
    if maxwell::metal::available() {
        maxwell::select_renderer(maxwell::RendererKind::Metal);
    }

    let mut flags = flags.iter();
    while let Some(flag) = flags.next() {
        if let Some(&(_, takes_value)) = MACHINE_FLAGS.iter().find(|(name, _)| name == flag) {
            machine_flags.push(flag.clone());
            if takes_value && let Some(value) = flags.clone().next() {
                machine_flags.push(recorded_value(flag, value));
            }
        }
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
            "--script" => {
                let path = value()?;
                let text =
                    std::fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
                script = Some(Script::parse(&text).map_err(|error| format!("{path}: {error}"))?);
            }
            "--emmc" => emmc_user = Some(value()?),
            "--emmc-boot0" => emmc_boot[0] = Some(value()?),
            "--emmc-boot1" => emmc_boot[1] = Some(value()?),
            "--emmc-writable" => emmc_writable = true,
            "--sd" => sd_card = Some(value()?),
            "--sd-writable" => sd_writable = true,
            "--wifi-ssid" => wifi_ssid = Some(value()?.clone()),
            "--wifi-passphrase" => wifi_passphrase = Some(value()?.clone()),
            "--dns" => {
                let value = value()?.as_str();
                let address = value
                    .parse::<std::net::SocketAddr>()
                    .or_else(|_| value.parse().map(|ip| std::net::SocketAddr::new(ip, 53)));
                dns = Some(address.map_err(|_| format!("--dns {value}: not ADDRESS[:PORT]"))?);
            }
            "--forward" => {
                let value = value()?.as_str();
                let (host, console) = value.split_once(':').unwrap_or((value, value));
                let ports = host.parse::<u16>().ok().zip(console.parse::<u16>().ok());
                forwards.push(
                    ports.ok_or_else(|| {
                        format!("--forward {value}: not HOST_PORT[:CONSOLE_PORT]")
                    })?,
                );
            }
            "--symbols" => {
                symbols =
                    SymbolTable::load(Path::new(value()?)).map_err(|error| error.to_string())?
            }
            "--trace-smc" => trace_smc = true,
            "--trace-svc" => trace_svc = true,
            "--trace-ipc" => trace_ipc = true,
            "--trace-times" => trace_times = true,
            "--verbose" => verbose = true,
            "--crash-dump" => crash_dump = Some(PathBuf::from(value()?)),
            "--control" => control = Some(value()?.clone()),
            "--trace-devices" => trace_devices = true,
            "--trace-bpmp" => {
                bpmp_trace = value()?
                    .parse()
                    .map_err(|error| format!("--trace-bpmp: {error}"))?
            }
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
            "--disable-program-verification" => params.program_verification_disabled = true,
            other if homebrew.parse(other, &mut value)? => {}
            _ => return Err(usage()),
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
    let user = emmc_user.map(open_image).transpose()?;
    let (overlay, replaced) = homebrew.overlay(user.as_ref(), &keys)?;
    let emmc = match user {
        Some(user) => Some(EmmcImages {
            user,
            boot: [boot0?, boot1?],
            writable: emmc_writable,
            overlay,
        }),
        None if emmc_boot.iter().all(Option::is_none) => None,
        None => return Err("eMMC boot partition images need --emmc for the module itself".into()),
    };
    let sd_card = sd_card
        .map(|path| {
            let file = OpenOptions::new()
                .read(true)
                .write(sd_writable)
                .open(path)
                .map_err(|error| format!("{path}: {error}"))?;
            SdCardImage::file(file, sd_writable).map_err(|error| format!("{path}: {error}"))
        })
        .transpose()?;
    let sd_card = match sd_card {
        Some(_) if homebrew.makes_sd_card() => {
            return Err("--homebrew makes the microSD card; it cannot go with --sd".into());
        }
        Some(card) => Some(card),
        None => homebrew.sd_card()?,
    };
    params.program_verification_disabled |= homebrew.disables_verification();
    let (firmware, kernel) = match kernel {
        Some((mut image, path)) => {
            let installed = homebrew.install_keys(&mut image)?;
            for line in installed.iter().chain(&replaced) {
                println!("{line}");
            }
            let path = std::path::absolute(path).map_err(|error| error.to_string())?;
            (Firmware::Kernel(image), Some(path))
        }
        None => {
            if homebrew.installs_keys() {
                return Err(
                    "--homebrew-keys and --homebrew patch the kernel's INI1, and power-on runs \
                     the real package2 as it is"
                        .into(),
                );
            }
            for line in &replaced {
                println!("{line}");
            }
            (Firmware::PowerOn(boot_rom(&keys, emmc_boot)?), None)
        }
    };
    // A machine somebody is watching, or something drives, runs for as
    // long as they do.
    let timeout = timeout.or((!show_window && control.is_none()).then_some(Duration::from_secs(5)));

    // Bound before the machine is built: a client may connect at once,
    // and is answered once the machine runs.
    let control = control
        .map(|path| {
            let _ = std::fs::remove_file(&path);
            std::os::unix::net::UnixListener::bind(&path)
                .map_err(|error| format!("--control {path}: {error}"))
        })
        .transpose()?;
    let machine = Machine::new(
        firmware,
        &params,
        keys,
        emmc,
        sd_card,
        Box::new(io::stderr()),
    )
    .map_err(|error| error.to_string())?;
    machine.set_bpmp_trace(bpmp_trace);
    let snapshotter = Snapshotter {
        kernel,
        flags: machine_flags,
    };
    if let Some(ssid) = &wifi_ssid {
        let ap = AccessPoint::new(ssid, wifi_passphrase.as_deref())
            .map_err(|error| format!("--wifi-ssid: {error}"))?;
        let wifi = std::sync::Arc::clone(machine.wifi());
        let router = net::Router::new(Box::new(move |frame| wifi.from_network(&frame)), dns);
        // One line per DHCP lease, DNS query, new flow and plain-HTTP
        // exchange: what the console does on the network, never payload.
        router.set_trace(Box::new(|line| eprintln!("net: {line}")));
        for &(host_port, console_port) in &forwards {
            router
                .forward(host_port, console_port)
                .map_err(|error| format!("--forward {host_port}: {error}"))?;
        }
        machine
            .wifi()
            .set_access_point(ap, Box::new(move |frame| router.send(frame)));
    } else if wifi_passphrase.is_some() {
        return Err("--wifi-passphrase needs --wifi-ssid".into());
    } else if !forwards.is_empty() {
        return Err("--forward needs --wifi-ssid".into());
    } else if dns.is_some() {
        return Err("--dns needs --wifi-ssid".into());
    }
    // The WiFi radio's access point is part of what is restored, so it is
    // in place first.
    if let Some(dir) = restore {
        let started = Instant::now();
        machine
            .restore_state(dir)
            .map_err(|error| format!("{}: {error}", dir.display()))?;
        println!(
            "restored {} in {:.2} s",
            dir.display(),
            started.elapsed().as_secs_f64()
        );
    }
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
        crash_dumps: crash_dump
            .map(|directory| {
                std::fs::create_dir_all(&directory)
                    .map_err(|error| format!("--crash-dump {}: {error}", directory.display()))?;
                // Numbered after the dumps of earlier runs, never over them.
                let earlier = std::fs::read_dir(&directory)
                    .map_err(|error| format!("--crash-dump {}: {error}", directory.display()))?
                    .filter_map(Result::ok)
                    .filter(|entry| entry.file_name().to_string_lossy().starts_with("crash-"))
                    .count();
                Ok::<_, String>(CrashDumps {
                    directory,
                    written: AtomicUsize::new(earlier),
                })
            })
            .transpose()?,
    };
    // A restored machine is past the boot that the default slows.
    let cpu_slowdown = cpu_slowdown.unwrap_or(match restore {
        Some(_) => (1, None),
        None => (10, Some(Duration::from_secs(12))),
    });
    let options = RunOptions {
        timeout,
        stop: &stop,
        trace_smc: trace_smc || verbose,
        // Requests and answers are read off the supervisor calls that carry
        // them.
        trace_svc: trace_svc || trace_ipc || verbose || tracing.crash_dumps.is_some(),
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
    let ended = AtomicBool::new(false);
    let report = thread::scope(|scope| {
        // A window needs this thread, so the machine gets another.
        let run = scope.spawn(|| machine.run(options));
        if let Some(listener) = control {
            let controlled = control::Controlled {
                machine: &machine,
                stop: &stop,
                started,
                ended: &ended,
                snapshot: &snapshotter,
            };
            scope.spawn(move || control::serve(listener, &controlled));
        }
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
                    Input::Power(held) => machine.press_power(held),
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
                narrate(&save_screen(&machine, &path, at), &tracing);
            }
            if let Some(script) = &mut script {
                script.advance(&machine, |event| {
                    let line = match event {
                        Event::Done { line, text } => format!("script line {line}: {text}"),
                        Event::Screenshot(path) => save_screen(&machine, path, now),
                    };
                    narrate(&line, &tracing);
                });
            }
        }
        let report = run
            .join()
            .unwrap_or_else(|payload| panic::resume_unwind(payload));
        ended.store(true, Ordering::SeqCst);
        report
    });

    let ended = match report.ended {
        RunEnd::Core(core) => format!("machine stopped by core {core}"),
        RunEnd::Timeout => format!(
            "machine stopped by timeout after {:?}",
            timeout.unwrap_or_default()
        ),
        RunEnd::Requested => "machine stopped: asked to (window closed or stop requested)".into(),
    };
    println!("{ended}");
    log.push(&ended);
    if let Some((line, text)) = script.as_ref().and_then(Script::waiting_at) {
        println!("script: did not get past line {line}: {text}");
    }
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
    if !report.calls_in_progress.is_empty() {
        println!("supervisor calls not returned from:");
    }
    for call in &report.calls_in_progress {
        let name = svc::name(call.id).unwrap_or("?");
        let process = call.process.as_ref().map_or("?", Process::name);
        println!(
            "  {process} (context {:#x}) thread {:#x}: {:#04x} {name} {:x?} from {:#x}, lr {:#x}, callers {:x?}, for {:.3} s",
            call.caller.context,
            call.caller.thread,
            call.id,
            call.arguments,
            call.pc,
            call.lr,
            call.callers,
            call.waited.as_secs_f64()
        );
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

fn write_png(writer: impl Write, frame: &Frame) -> Result<(), png::EncodingError> {
    let mut encoder = png::Encoder::new(io::BufWriter::new(writer), frame.width, frame.height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header()?.write_image_data(&frame.rgb())
}

/// Saves what the screen shows to `path`; the line to report it with.
fn save_screen(machine: &Machine, path: &str, at: Duration) -> String {
    let at = at.as_secs_f64();
    match machine.screen() {
        Ok(screen) => match File::create(path)
            .map_err(|error| error.to_string())
            .and_then(|file| write_png(file, &screen.frame).map_err(|error| error.to_string()))
        {
            Ok(()) => format!("screen at {at:.3} s saved to {path}"),
            Err(error) => format!("{path}: {error}"),
        },
        Err(reason) => format!("screen at {at:.3} s: no picture, {reason}"),
    }
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
    /// The console's power button is held, or let go of.
    Power(bool),
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
            if button == "power" {
                return Some([(at, Input::Power(true)), (until, Input::Power(false))]);
            }
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
    /// Where crash dumps go, with `--crash-dump`.
    crash_dumps: Option<CrashDumps>,
}

/// The directory `--crash-dump` names, and how many dumps went there.
struct CrashDumps {
    directory: PathBuf,
    written: AtomicUsize,
}

/// What a user exception was, from its syndrome.
fn exception_kind(syndrome: u64) -> &'static str {
    match exception_class(syndrome) {
        INSTRUCTION_ABORT_LOWER => "an instruction abort",
        DATA_ABORT_LOWER if syndrome & 1 << 6 != 0 => "a data abort writing",
        DATA_ABORT_LOWER => "a data abort reading",
        exception_class::PC_ALIGNMENT => "a PC alignment fault",
        exception_class::SP_ALIGNMENT => "an SP alignment fault",
        exception_class::BRK64 => "a BRK",
        _ => "an undefined instruction",
    }
}

/// Process code addresses are below 2^39; the bits above may carry a
/// return-address tag (the browser engine XORs one into bits 40 and up).
const USER_ADDRESS_MASK: u64 = (1 << 39) - 1;

/// `module+0x…` for a code address, its tag bits stripped, if a module
/// holds it.
fn module_offset(address: u64, modules: &[Module]) -> Option<String> {
    let address = address & USER_ADDRESS_MASK;
    let (index, module) = (modules.iter().enumerate())
        .find(|(_, module)| (module.base..module.base + module.size).contains(&address))?;
    let name = module
        .name
        .as_deref()
        .and_then(|path| path.rsplit(['/', '\\']).next())
        .map_or_else(|| format!("module{index}"), str::to_string);
    Some(format!("{name}+{:#x}", address - module.base))
}

/// The text of a crash dump.
fn crash_report(
    core: usize,
    caller: Caller,
    process: Option<Process>,
    state: &ExceptionState,
) -> String {
    use std::fmt::Write as _;
    let at = |address: u64| {
        let stripped = address & USER_ADDRESS_MASK;
        let tag = (stripped != address).then(|| format!(" (untagged {stripped:#x})"));
        let module = module_offset(address, &state.modules).map(|place| format!(" {place}"));
        format!(
            "{address:#018x}{}{}",
            tag.unwrap_or_default(),
            module.unwrap_or_default()
        )
    };
    let mut text = String::new();
    let _ = writeln!(text, "horizon-emu crash dump");
    match process {
        Some(process) => {
            let _ = writeln!(
                text,
                "process:   {} (program {:016x}, process ID {})",
                process.name(),
                process.program_id,
                process.id
            );
        }
        None => {
            let _ = writeln!(text, "process:   ?");
        }
    }
    let _ = writeln!(
        text,
        "thread:    TLS {:#x}, context {:#x}, core {core}",
        caller.thread, caller.context
    );
    let _ = writeln!(
        text,
        "exception: {} (ESR {:#010x}, class {:#04x})",
        exception_kind(state.syndrome),
        state.syndrome,
        exception_class(state.syndrome)
    );
    let _ = writeln!(text, "far:       {:#018x}", state.address);
    let _ = writeln!(text, "pc:        {}", at(state.pc));
    let _ = writeln!(text, "lr:        {}", at(state.registers[30]));
    let _ = writeln!(text, "sp:        {:#018x}", state.sp);
    let _ = writeln!(text, "pstate:    {:#010x}", state.pstate);
    let _ = writeln!(text, "\nregisters:");
    for (index, value) in state.registers.iter().enumerate() {
        let _ = writeln!(text, "  x{index:<2} {}", at(*value));
    }
    let _ = writeln!(text, "\ncall chain (frame records from x29):");
    for (depth, address) in state.callers.iter().enumerate() {
        let _ = writeln!(text, "  #{depth:<2} {}", at(*address));
    }
    let _ = writeln!(text, "\nmodules:");
    for (index, module) in state.modules.iter().enumerate() {
        let build_id = module.build_id.as_ref().map(|id| {
            id.iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        });
        let _ = writeln!(
            text,
            "  {index:<2} {:#012x}-{:#012x} text {:#x}{} build {} {}",
            module.base,
            module.base + module.size,
            module.text_size,
            if module.mod0 { "" } else { " (no MOD0)" },
            build_id.as_deref().unwrap_or("?"),
            module.name.as_deref().unwrap_or("?")
        );
    }
    text
}

impl CrashDumps {
    /// Writes one dump; its path.
    fn write(&self, process: Option<Process>, report: &str) -> io::Result<PathBuf> {
        let number = self.written.fetch_add(1, Ordering::Relaxed);
        let name = process.as_ref().map_or("unknown", Process::name);
        let name: String = (name.chars())
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect();
        let path = self.directory.join(format!("crash-{number:03}-{name}.txt"));
        std::fs::write(&path, report)?;
        Ok(path)
    }
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
            if tracing.verbose {
                match effect {
                    SmcEffect::StartCore { core: started, .. } => {
                        writeln!(out, "core {started} started by core {core}")?;
                    }
                    SmcEffect::PowerOff => writeln!(out, "core {core} switched off")?,
                    SmcEffect::Sleep { .. } => writeln!(out, "console asleep")?,
                    _ => {}
                }
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
        TraceEvent::UserException {
            core,
            caller,
            process,
            state,
        } => {
            if let Some(dumps) = &tracing.crash_dumps {
                let report = crash_report(core, caller, process, state);
                match dumps.write(process, &report) {
                    Ok(path) => writeln!(out, "core {core}: crash dump {}", path.display())?,
                    Err(error) => writeln!(out, "core {core}: crash dump not written: {error}")?,
                }
            }
            if !(tracing.verbose || tracing.svc) {
                return Ok(());
            }
            let place = module_offset(state.pc, &state.modules)
                .map(|place| format!(" ({place})"))
                .unwrap_or_default();
            writeln!(
                out,
                "core {core}: {} took {} at {:#x}, pc {:#x}{place}, lr {:#x}, \
                 callers {:x?} (syndrome {:#x})",
                describe_caller(caller, process),
                exception_kind(state.syndrome),
                state.address,
                state.pc,
                state.registers[30],
                state.callers,
                state.syndrome
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
        TraceEvent::Woke { core } => writeln!(out, "console woke, core {core} runs"),
        TraceEvent::SecureAccess {
            core,
            block,
            address,
            is_write,
        } => {
            if !(tracing.verbose || tracing.svc) {
                return Ok(());
            }
            let what = if is_write { "write to" } else { "read of" };
            writeln!(
                out,
                "core {core}: {what} secure-only {block} at {address:#x} from the non-secure \
                 world: external abort"
            )
        }
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
