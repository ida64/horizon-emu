//! Machines as the UI keeps them: a saved set of `horizon-emu` flags per
//! machine, in a directory of its own under [`data_dir`] with its
//! snapshots beside it, and how they become a command line.

use std::{
    fs, io,
    path::{Path, PathBuf},
    time::SystemTime,
};

use serde::{Deserialize, Serialize};

/// Where the UI keeps machines, downloaded firmware and its settings.
pub fn data_dir() -> PathBuf {
    let home = std::env::var_os("HOME").map_or_else(|| PathBuf::from("."), PathBuf::from);
    home.join("Library/Application Support/horizon-emu")
}

pub fn machines_dir() -> PathBuf {
    data_dir().join("machines")
}

/// Where downloaded system updates go, one directory per version.
pub fn firmware_dir() -> PathBuf {
    data_dir().join("firmware")
}

/// How the machine starts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Start {
    /// `power-on`: from the boot ROM (built in or a dump), or a payload.
    PowerOn,
    /// `boot KERNEL`: at kernel entry.
    Kernel { path: Option<PathBuf> },
    /// `restore DIR`: a snapshot, which says what the machine is.
    Snapshot { dir: Option<PathBuf> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Renderer {
    Metal,
    Software,
}

impl Renderer {
    pub fn label(self) -> &'static str {
        match self {
            Self::Metal => "Metal",
            Self::Software => "Software",
        }
    }

    fn flag(self) -> &'static str {
        match self {
            Self::Metal => "metal",
            Self::Software => "software",
        }
    }
}

/// `--cpu-slowdown`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CpuSpeed {
    /// The emulator's default: a boot at the console's speed (10 times
    /// slower) until boot2 ends, a restored machine at full speed.
    Default,
    Full,
    /// `factor` times slower throughout.
    Slower {
        factor: u32,
    },
}

impl CpuSpeed {
    pub fn label(self) -> String {
        match self {
            Self::Default => "10× slower until boot2 ends".into(),
            Self::Full => "Full speed".into(),
            Self::Slower { factor } => format!("{factor}× slower throughout"),
        }
    }
}

/// Options for programs Nintendo did not sign (`horizon_emu::homebrew`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Homebrew {
    /// `--homebrew NRO`.
    pub nro: Option<PathBuf>,
    /// `--homebrew-in PROGRAM_ID`.
    pub run_in: String,
    /// `--hbl NSP`.
    pub hbl: Option<PathBuf>,
    /// `--homebrew-app NRO`, each.
    pub apps: Vec<PathBuf>,
    pub keys: bool,
    pub disable_verification: bool,
    /// `--replace-program PROGRAM_ID=NSP`, each.
    pub replacements: Vec<(String, PathBuf)>,
}

/// Options of a run that are not part of what the machine is.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Debug {
    pub verbose: bool,
    pub log: bool,
    pub trace_smc: bool,
    pub trace_svc: bool,
    pub trace_ipc: bool,
    pub trace_devices: bool,
    pub trace_times: bool,
    pub survey: bool,
    pub list_threads: bool,
    pub crash_dumps: bool,
    /// Seconds; none for no limit.
    pub timeout: Option<f64>,
    pub breaks: Vec<String>,
    pub traces: Vec<String>,
    pub pokes: Vec<String>,
    /// `--step N` after a break.
    pub steps: Option<u32>,
    /// `--trace-bpmp N`.
    pub trace_bpmp: Option<u32>,
    pub gpu_capture: Option<PathBuf>,
}

impl Default for Debug {
    fn default() -> Self {
        Self {
            verbose: true,
            log: false,
            trace_smc: false,
            trace_svc: false,
            trace_ipc: false,
            trace_devices: false,
            trace_times: false,
            survey: false,
            list_threads: true,
            crash_dumps: false,
            timeout: None,
            breaks: Vec::new(),
            traces: Vec::new(),
            pokes: Vec::new(),
            steps: None,
            trace_bpmp: None,
            gpu_capture: None,
        }
    }
}

/// A machine: what `horizon-emu` is told to build and how to run it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Machine {
    pub name: String,
    pub start: Start,
    pub dram: u8,
    pub seed: u64,
    pub renderer: Renderer,
    pub cpu: CpuSpeed,
    pub keys: Option<PathBuf>,
    pub emmc: Option<PathBuf>,
    pub boot0: Option<PathBuf>,
    pub boot1: Option<PathBuf>,
    /// `--emmc-writable`: the user area and both boot partitions alike.
    pub emmc_writable: bool,
    pub sd: Option<PathBuf>,
    pub sd_writable: bool,
    pub symbols: Option<PathBuf>,
    pub bootrom: Option<PathBuf>,
    pub payload: Option<PathBuf>,
    pub debug_mode: bool,
    pub wifi_ssid: String,
    pub wifi_passphrase: String,
    pub air: String,
    pub dns: String,
    pub joycons_at_start: bool,
    pub pro_controller: Option<PathBuf>,
    pub amiibo: Option<PathBuf>,
    pub amiibo_writable: bool,
    pub homebrew: Homebrew,
    pub debug: Debug,
}

impl Default for Machine {
    fn default() -> Self {
        Self {
            name: "Cold boot".into(),
            start: Start::PowerOn,
            dram: 4,
            seed: 0,
            renderer: Renderer::Metal,
            cpu: CpuSpeed::Default,
            keys: None,
            emmc: None,
            boot0: None,
            boot1: None,
            emmc_writable: false,
            sd: None,
            sd_writable: false,
            symbols: None,
            bootrom: None,
            payload: None,
            debug_mode: false,
            wifi_ssid: String::new(),
            wifi_passphrase: String::new(),
            air: String::new(),
            dns: String::new(),
            joycons_at_start: true,
            pro_controller: None,
            amiibo: None,
            amiibo_writable: false,
            homebrew: Homebrew::default(),
            debug: Debug::default(),
        }
    }
}

fn text(path: &Path) -> String {
    path.display().to_string()
}

impl Machine {
    /// What the sidebar says the machine is.
    pub fn kind(&self) -> &'static str {
        match self.start {
            Start::Snapshot { .. } => "Snapshot",
            _ if self.homebrew.nro.is_some() => "Homebrew",
            Start::Kernel { .. } => "Kernel ELF",
            Start::PowerOn => "Power-on",
        }
    }

    /// How it starts, for the header.
    pub fn boot_label(&self) -> &'static str {
        match self.start {
            Start::Snapshot { .. } => "Restored from a snapshot",
            Start::Kernel { .. } => "Kernel entry",
            Start::PowerOn if self.payload.is_some() => "Power-on · payload",
            Start::PowerOn if self.bootrom.is_some() => "Power-on · boot ROM dump",
            Start::PowerOn => "Power-on · HLE boot ROM",
        }
    }

    /// Flags that make the machine what it is; a snapshot records them and
    /// `restore` refuses them.
    fn machine_flags(&self) -> Vec<String> {
        let mut flags = Vec::new();
        let mut put = |flag: &str, value: Option<String>| {
            flags.push(flag.to_string());
            flags.extend(value);
        };
        if self.dram != 4 {
            put("--dram", Some(self.dram.to_string()));
        }
        if self.seed != 0 {
            put("--seed", Some(self.seed.to_string()));
        }
        for (flag, path) in [
            ("--keys", &self.keys),
            ("--emmc", &self.emmc),
            ("--emmc-boot0", &self.boot0),
            ("--emmc-boot1", &self.boot1),
            ("--sd", &self.sd),
            ("--pro-controller", &self.pro_controller),
        ] {
            if let Some(path) = path {
                put(flag, Some(text(path)));
            }
        }
        if self.emmc_writable
            && (self.emmc.is_some() || self.boot0.is_some() || self.boot1.is_some())
        {
            put("--emmc-writable", None);
        }
        if self.sd_writable && self.sd.is_some() {
            put("--sd-writable", None);
        }
        if self.start == Start::PowerOn {
            if let Some(path) = &self.bootrom {
                put("--bootrom", Some(text(path)));
            }
            if let Some(path) = &self.payload {
                put("--payload", Some(text(path)));
            }
        }
        if !self.wifi_ssid.trim().is_empty() {
            put("--wifi-ssid", Some(self.wifi_ssid.trim().into()));
            if !self.wifi_passphrase.is_empty() {
                put("--wifi-passphrase", Some(self.wifi_passphrase.clone()));
            }
        }
        if self.debug_mode {
            put("--debug-mode", None);
        }
        let homebrew = &self.homebrew;
        if homebrew.disable_verification {
            put("--disable-program-verification", None);
        }
        if homebrew.keys {
            put("--homebrew-keys", None);
        }
        for (program, nsp) in &homebrew.replacements {
            put(
                "--replace-program",
                Some(format!("{program}={}", nsp.display())),
            );
        }
        if let Some(nro) = &homebrew.nro {
            put("--homebrew", Some(text(nro)));
            if !homebrew.run_in.trim().is_empty() {
                put("--homebrew-in", Some(homebrew.run_in.trim().into()));
            }
            if let Some(hbl) = &homebrew.hbl {
                put("--hbl", Some(text(hbl)));
            }
            for app in &homebrew.apps {
                put("--homebrew-app", Some(text(app)));
            }
        }
        flags
    }

    /// Flags of a run, which a restore takes as well.
    fn run_flags(&self, control: &Path, crash_dir: &Path, restoring: bool) -> Vec<String> {
        let mut flags = vec!["--control".to_string(), text(control)];
        let mut put = |flag: &str, value: Option<String>| {
            flags.push(flag.to_string());
            flags.extend(value);
        };
        put("--renderer", Some(self.renderer.flag().into()));
        match self.cpu {
            CpuSpeed::Default => {}
            CpuSpeed::Full => put("--cpu-slowdown", Some("1".into())),
            CpuSpeed::Slower { factor } => put("--cpu-slowdown", Some(factor.max(1).to_string())),
        }
        if let Some(symbols) = &self.symbols {
            put("--symbols", Some(text(symbols)));
        }
        if !self.dns.trim().is_empty() {
            put("--dns", Some(self.dns.trim().into()));
        }
        if !self.air.trim().is_empty() {
            put("--air", Some(self.air.trim().into()));
        }
        // A restored machine has the Joy-Con where the snapshot had them.
        if self.joycons_at_start && !restoring {
            put("--joycons", Some("0".into()));
        }
        if let Some(amiibo) = &self.amiibo {
            put("--amiibo", Some(text(amiibo)));
            if self.amiibo_writable {
                put("--amiibo-writable", None);
            }
        }
        let debug = &self.debug;
        for (on, flag) in [
            (debug.verbose, "--verbose"),
            (debug.log, "--log"),
            (debug.trace_smc, "--trace-smc"),
            (debug.trace_svc, "--trace-svc"),
            (debug.trace_ipc, "--trace-ipc"),
            (debug.trace_devices, "--trace-devices"),
            (debug.trace_times, "--trace-times"),
            (debug.survey, "--survey"),
            (debug.list_threads, "--list-threads"),
        ] {
            if on {
                put(flag, None);
            }
        }
        if debug.crash_dumps {
            put("--crash-dump", Some(text(crash_dir)));
        }
        if let Some(seconds) = debug.timeout {
            put("--timeout", Some(seconds.to_string()));
        }
        for (flag, places) in [
            ("--break", &debug.breaks),
            ("--trace", &debug.traces),
            ("--poke", &debug.pokes),
        ] {
            for place in places.iter().filter(|place| !place.trim().is_empty()) {
                put(flag, Some(place.trim().into()));
            }
        }
        if let Some(steps) = debug.steps {
            put("--step", Some(steps.to_string()));
        }
        if let Some(count) = debug.trace_bpmp {
            put("--trace-bpmp", Some(count.to_string()));
        }
        if let Some(path) = &debug.gpu_capture {
            put("--gpu-capture", Some(text(path)));
        }
        flags
    }

    /// The command line of a run of this machine kept in `dir`, its control
    /// socket at `control`: a cold start, or the restore of `snapshot`.
    pub fn arguments(
        &self,
        dir: &Path,
        control: &Path,
        snapshot: Option<&Path>,
    ) -> Result<Vec<String>, String> {
        let crash_dir = dir.join("crashes");
        let snapshot = snapshot
            .map(Path::to_path_buf)
            .or_else(|| match &self.start {
                Start::Snapshot { dir } => dir.clone(),
                _ => None,
            });
        if let Some(snapshot) = snapshot {
            let name = snapshot.file_name().map_or_else(
                || "snapshot".into(),
                |name| name.to_string_lossy().into_owned(),
            );
            let mut arguments = vec![
                "restore".to_string(),
                text(&snapshot),
                "--workdir".into(),
                text(&dir.join("work").join(name)),
            ];
            arguments.extend(self.run_flags(control, &crash_dir, true));
            return Ok(arguments);
        }
        if let Start::Snapshot { dir: None } = self.start {
            return Err("choose the snapshot this machine starts from".into());
        }
        let mut arguments = match &self.start {
            Start::PowerOn => vec!["power-on".to_string()],
            Start::Kernel { path: Some(kernel) } => vec!["boot".to_string(), text(kernel)],
            Start::Kernel { path: None } => return Err("choose the kernel ELF to boot".into()),
            Start::Snapshot { .. } => unreachable!("restores are handled above"),
        };
        arguments.extend(self.machine_flags());
        arguments.extend(self.run_flags(control, &crash_dir, false));
        Ok(arguments)
    }
}

/// A snapshot in a machine's `snapshots` directory.
#[derive(Debug, Clone)]
pub struct Snapshot {
    pub name: String,
    pub dir: PathBuf,
    pub created: Option<SystemTime>,
    /// What the UI recorded when it took it.
    pub info: SnapshotInfo,
}

/// What the control socket answered when the snapshot was taken, kept in
/// `<name>.json` beside the snapshot's directory.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SnapshotInfo {
    pub guest_seconds: Option<f64>,
    pub paused_seconds: Option<f64>,
    pub ram_bytes: Option<u64>,
}

/// What `boot.json` of a snapshot says the machine is.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct SavedBoot {
    pub flags: Vec<String>,
    pub power_on: bool,
}

impl SavedBoot {
    pub fn load(snapshot: &Path) -> Result<Self, String> {
        let file = snapshot.join("boot.json");
        let bytes = fs::read(&file).map_err(|error| format!("{}: {error}", file.display()))?;
        serde_json::from_slice(&bytes).map_err(|error| format!("{}: {error}", file.display()))
    }

    /// The value given to `flag`, if it was.
    pub fn value(&self, flag: &str) -> Option<&str> {
        let index = self.flags.iter().position(|given| given == flag)?;
        self.flags.get(index + 1).map(String::as_str)
    }
}

/// A machine as stored: its directory and definition.
pub struct Stored {
    pub dir: PathBuf,
    pub machine: Machine,
}

impl Stored {
    fn file(dir: &Path) -> PathBuf {
        dir.join("machine.json")
    }

    /// Every machine in [`machines_dir`], by name.
    pub fn load_all() -> Vec<Self> {
        let Ok(entries) = fs::read_dir(machines_dir()) else {
            return Vec::new();
        };
        let mut machines: Vec<Self> = entries
            .filter_map(Result::ok)
            .filter_map(|entry| {
                let dir = entry.path();
                let bytes = fs::read(Self::file(&dir)).ok()?;
                let machine = serde_json::from_slice(&bytes).ok()?;
                Some(Self { dir, machine })
            })
            .collect();
        machines.sort_by(|a, b| a.machine.name.cmp(&b.machine.name));
        machines
    }

    /// A new machine's directory, named after it.
    pub fn create(machine: Machine) -> io::Result<Self> {
        let slug: String = machine
            .name
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() {
                    c.to_ascii_lowercase()
                } else {
                    '-'
                }
            })
            .collect();
        let slug = slug.trim_matches('-');
        let slug = if slug.is_empty() { "machine" } else { slug };
        let base = machines_dir();
        let mut dir = base.join(slug);
        let mut index = 2;
        while dir.exists() {
            dir = base.join(format!("{slug}-{index}"));
            index += 1;
        }
        fs::create_dir_all(&dir)?;
        let stored = Self { dir, machine };
        stored.save()?;
        Ok(stored)
    }

    pub fn save(&self) -> io::Result<()> {
        let json = serde_json::to_vec_pretty(&self.machine).map_err(io::Error::other)?;
        let file = Self::file(&self.dir);
        let part = file.with_extension("json.part");
        fs::write(&part, json)?;
        fs::rename(part, file)
    }

    pub fn snapshots_dir(&self) -> PathBuf {
        self.dir.join("snapshots")
    }

    /// The machine's snapshots, newest first.
    pub fn snapshots(&self) -> Vec<Snapshot> {
        let Ok(entries) = fs::read_dir(self.snapshots_dir()) else {
            return Vec::new();
        };
        let mut snapshots: Vec<Snapshot> = entries
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.join("boot.json").is_file())
            .map(|dir| {
                let name = dir
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned();
                let info = fs::read(dir.with_extension("json"))
                    .ok()
                    .and_then(|bytes| serde_json::from_slice(&bytes).ok())
                    .unwrap_or_default();
                let created = fs::metadata(dir.join("boot.json"))
                    .and_then(|meta| meta.modified())
                    .ok();
                Snapshot {
                    name,
                    dir,
                    created,
                    info,
                }
            })
            .collect();
        snapshots.sort_by(|a, b| b.created.cmp(&a.created));
        snapshots
    }
}

/// Settings of the UI itself, remembered between sessions.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub keys: Option<PathBuf>,
    /// The decrypted calibration data of the console whose identity
    /// downloads firmware from Nintendo, and that console's keys.
    pub prodinfo: Option<PathBuf>,
    pub prodinfo_keys: Option<PathBuf>,
}

/// The entries of the key file at `path` that the emulated console needs
/// and the file lacks: those that make its eMMC image (storage, content
/// archive header, key area, save MAC and TLS key wrapping keys, and their
/// sources) and, with `power_on`, the secure boot key the boot ROM's
/// emulation needs. Empty when nothing is missing.
pub fn missing_keys(path: &Path, power_on: bool) -> Result<Vec<&'static str>, String> {
    let keys = horizon_emu::keys::KeySet::load(path)
        .map_err(|error| format!("{}: {error}", path.display()))?;
    let mut missing = Vec::new();
    let mut need = |present: bool, name: &'static str| {
        if !present {
            missing.push(name);
        }
    };
    need(keys.device_key().is_some(), "device_key");
    need(keys.master_key(0).is_some(), "master_key_00");
    need(
        keys.aes_kek_generation_source().is_some(),
        "aes_kek_generation_source",
    );
    need(
        keys.aes_key_generation_source().is_some(),
        "aes_key_generation_source",
    );
    need(
        keys.retail_specific_aes_key_source().is_some(),
        "retail_specific_aes_key_source",
    );
    need(keys.bis_kek_source().is_some(), "bis_kek_source");
    for (index, name) in [
        "bis_key_source_00",
        "bis_key_source_01",
        "bis_key_source_02",
    ]
    .into_iter()
    .enumerate()
    {
        need(keys.bis_key_source(index).is_some(), name);
    }
    need(keys.header_kek_source().is_some(), "header_kek_source");
    need(keys.header_key_source().is_some(), "header_key_source");
    need(
        keys.key_area_key_source(2).is_some(),
        "key_area_key_system_source",
    );
    need(keys.save_mac_kek_source().is_some(), "save_mac_kek_source");
    need(
        keys.save_mac_key_source().is_some(),
        "save_mac_key_source_00",
    );
    need(
        keys.ssl_rsa_kekek_source().is_some(),
        "ssl_rsa_kekek_source",
    );
    need(keys.ssl_rsa_kek_source().is_some(), "ssl_rsa_kek_source");
    if power_on {
        need(keys.secure_boot_key().is_some(), "secure_boot_key");
    }
    Ok(missing)
}

/// Why the key file at `path` cannot be the emulated console's, if it
/// cannot.
pub fn keys_problem(path: &Path, power_on: bool) -> Option<String> {
    match missing_keys(path, power_on) {
        Err(error) => Some(error),
        Ok(missing) if missing.is_empty() => None,
        Ok(missing) => Some(format!(
            "{} lacks {}. The emulated console needs a complete key dump of an original \
             console (Lockpick_RCM's prod.keys with device_key and secure_boot_key); a \
             partial one cannot make or start its eMMC.",
            path.file_name().unwrap_or_default().to_string_lossy(),
            missing.join(", ")
        )),
    }
}

impl Settings {
    fn file() -> PathBuf {
        data_dir().join("settings.json")
    }

    pub fn load() -> Self {
        fs::read(Self::file())
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .unwrap_or_default()
    }

    pub fn save(&self) {
        if let Ok(json) = serde_json::to_vec_pretty(self) {
            let _ = fs::create_dir_all(data_dir());
            let _ = fs::write(Self::file(), json);
        }
    }
}

/// System updates downloaded before, as `(version, directory)`, newest
/// first.
pub fn downloaded_firmware() -> Vec<(String, PathBuf)> {
    let Ok(entries) = fs::read_dir(firmware_dir()) else {
        return Vec::new();
    };
    let mut found: Vec<(Vec<u32>, String, PathBuf)> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter_map(|dir| {
            let name = dir.file_name()?.to_str()?.to_string();
            let parts: Vec<u32> = name
                .split('.')
                .map(str::parse)
                .collect::<Result<_, _>>()
                .ok()?;
            Some((parts, name, dir))
        })
        .collect();
    found.sort_by(|a, b| b.0.cmp(&a.0));
    found
        .into_iter()
        .map(|(_, name, dir)| (name, dir))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_args(machine: &Machine) -> Vec<String> {
        machine
            .arguments(Path::new("/m"), Path::new("/tmp/c.sock"), None)
            .unwrap()
    }

    #[test]
    fn a_restore_carries_no_machine_flags_and_a_cold_boot_carries_them() {
        let machine = Machine {
            keys: Some("/k.keys".into()),
            emmc: Some("/e.img".into()),
            emmc_writable: true,
            wifi_ssid: "lab".into(),
            dram: 6,
            ..Machine::default()
        };
        let cold = run_args(&machine);
        assert_eq!(cold[0], "power-on");
        for flag in [
            "--keys",
            "--emmc",
            "--emmc-writable",
            "--wifi-ssid",
            "--dram",
            "--joycons",
        ] {
            assert!(cold.iter().any(|given| given == flag), "{flag}");
        }
        let restored = machine
            .arguments(
                Path::new("/m"),
                Path::new("/tmp/c.sock"),
                Some(Path::new("/m/snapshots/a")),
            )
            .unwrap();
        assert_eq!(
            restored[..4],
            ["restore", "/m/snapshots/a", "--workdir", "/m/work/a"]
        );
        for flag in [
            "--keys",
            "--emmc",
            "--emmc-writable",
            "--wifi-ssid",
            "--dram",
            "--joycons",
        ] {
            assert!(!restored.iter().any(|given| given == flag), "{flag}");
        }
        assert!(restored.iter().any(|given| given == "--control"));
    }

    #[test]
    fn boot_rom_and_payload_go_only_to_a_power_on() {
        let machine = Machine {
            start: Start::Kernel {
                path: Some("/kernel.elf".into()),
            },
            bootrom: Some("/rom.bin".into()),
            payload: Some("/hekate.bin".into()),
            ..Machine::default()
        };
        let arguments = run_args(&machine);
        assert_eq!(arguments[..2], ["boot", "/kernel.elf"]);
        assert!(
            !arguments
                .iter()
                .any(|given| given == "--bootrom" || given == "--payload")
        );
    }
}
