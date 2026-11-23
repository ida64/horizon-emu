//! The tabs of a machine: its settings, run options for debugging, the
//! guest's threads at the last stop, its snapshots and its output.

use std::path::{Path, PathBuf};

use eframe::egui::{self, Align, Color32, Layout, RichText, TextEdit, Ui};

use crate::{
    app::Entry,
    child::{self, Job, Stream},
    model::{CpuSpeed, Machine, Renderer, SavedBoot, Start},
    theme::{self, AMBER, MUTED, RED, Rows},
};

/// A file dialog starting where `current` is.
fn dialog_at(current: Option<&Path>) -> rfd::FileDialog {
    let dialog = rfd::FileDialog::new();
    match current.and_then(Path::parent) {
        Some(dir) => dialog.set_directory(dir),
        None => dialog,
    }
}

pub fn pick_file(current: Option<&Path>) -> Option<PathBuf> {
    dialog_at(current).pick_file()
}

pub fn pick_folder(current: Option<&Path>) -> Option<PathBuf> {
    dialog_at(current).pick_folder()
}

pub fn save_file(current: Option<&Path>, name: &str) -> Option<PathBuf> {
    dialog_at(current).set_file_name(name).save_file()
}

/// A row's controls for an optional file: its name, "Change" (or
/// `choose` while there is none) and a way to clear it.
fn file_controls(
    ui: &mut Ui,
    value: &mut Option<PathBuf>,
    choose: &str,
    none: &str,
    clearable: bool,
) {
    if theme::small_button(ui, if value.is_some() { "Change" } else { choose }).clicked()
        && let Some(path) = pick_file(value.as_deref())
    {
        *value = Some(path);
    }
    if clearable && value.is_some() && theme::clear_button(ui).clicked() {
        *value = None;
    }
    theme::path_label(ui, value.as_deref(), none);
}

fn text_field(ui: &mut Ui, value: &mut String, hint: &str, width: f32) {
    ui.add(
        TextEdit::singleline(value)
            .hint_text(hint)
            .desired_width(width),
    );
}

pub fn settings(ui: &mut Ui, entry: &mut Entry) {
    let locked = entry.locked();
    let running = entry.running();
    let snapshot_boot = match &entry.machine().start {
        Start::Snapshot { dir: Some(dir) } => Some(SavedBoot::load(dir)),
        _ => None,
    };
    let (install, installed) = (&mut entry.install, &mut entry.installed);
    let machine = &mut entry.stored.machine;
    ui.columns(2, |columns| {
        let (left, right) = columns.split_at_mut(1);
        let (left, right) = (&mut left[0], &mut right[0]);
        theme::section(left, "Hardware", |ui| hardware(ui, machine, locked));
        left.add_space(12.0);
        theme::section(left, "Boot", |ui| boot(ui, machine, locked));
        left.add_space(12.0);
        theme::section(left, "Controllers", |ui| controllers(ui, machine, locked));
        if let Some(saved) = &snapshot_boot {
            theme::section(right, "From the snapshot", |ui| recorded(ui, saved));
        } else {
            theme::section(right, "Storage", |ui| storage(ui, machine, locked));
            right.add_space(12.0);
            theme::section(right, "Games", |ui| {
                games(ui, machine, running, install, installed)
            });
            right.add_space(12.0);
        }
        theme::section(right, "Network", |ui| network(ui, machine, locked));
        right.add_space(12.0);
        theme::section(right, "Homebrew", |ui| homebrew(ui, machine, locked));
    });
    ui.add_space(10.0);
    ui.label(
        RichText::new(
            "DRAM, seed, keys, media, boot ROM, Wi-Fi network, Pro Controller and homebrew \
             options are what a snapshot records: they are locked once a snapshot of this \
             machine exists.",
        )
        .size(13.0)
        .color(MUTED),
    );
}

/// The machine flags a snapshot recorded, which a restore uses.
fn recorded(ui: &mut Ui, saved: &Result<SavedBoot, String>) {
    let saved = match saved {
        Ok(saved) => saved,
        Err(error) => {
            ui.label(RichText::new(error).color(RED));
            return;
        }
    };
    let mut rows = Rows::new();
    rows.row(ui, "Started by", |ui| {
        ui.label(if saved.power_on {
            "Power-on"
        } else {
            "Kernel entry"
        });
    });
    let mut flags = saved.flags.iter().peekable();
    while let Some(flag) = flags.next() {
        let value = flags.next_if(|value| !value.starts_with("--")).cloned();
        rows.row(ui, flag.trim_start_matches("--"), |ui| match value {
            Some(value) => {
                theme::path_label(ui, Some(Path::new(&value)), "");
            }
            None => {
                ui.label(RichText::new("on").color(MUTED));
            }
        });
    }
}

fn hardware(ui: &mut Ui, machine: &mut Machine, locked: bool) {
    let mut rows = Rows::new();
    rows.row(ui, "DRAM", |ui| {
        ui.add_enabled_ui(!locked, |ui| {
            egui::ComboBox::from_id_salt("dram")
                .selected_text(format!("{} GiB", machine.dram))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut machine.dram, 4, "4 GiB");
                    ui.selectable_value(&mut machine.dram, 6, "6 GiB");
                });
        });
    });
    rows.row(ui, "Secure monitor seed", |ui| {
        ui.add_enabled(!locked, egui::DragValue::new(&mut machine.seed).speed(1.0));
    });
    rows.row(ui, "Renderer", |ui| {
        egui::ComboBox::from_id_salt("renderer")
            .selected_text(machine.renderer.label())
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut machine.renderer, Renderer::Metal, "Metal");
                ui.selectable_value(&mut machine.renderer, Renderer::Software, "Software");
            });
    });
    rows.row(ui, "CPU speed", |ui| {
        egui::ComboBox::from_id_salt("cpu")
            .selected_text(machine.cpu.label())
            .width(230.0)
            .show_ui(ui, |ui| {
                for option in [
                    CpuSpeed::Default,
                    CpuSpeed::Full,
                    CpuSpeed::Slower { factor: 10 },
                ] {
                    ui.selectable_value(&mut machine.cpu, option, option.label());
                }
            })
            .response
            .on_hover_text(
                "A boot runs at the console's speed, 10 times slower, until its launcher \
                 (boot2) has ended; a restored machine at full speed (--cpu-slowdown).",
            );
    });
}

fn boot(ui: &mut Ui, machine: &mut Machine, locked: bool) {
    let mut rows = Rows::new();
    rows.row(ui, "Start from", |ui| {
        ui.add_enabled_ui(!locked, |ui| {
            let label = match machine.start {
                Start::PowerOn => "Power-on (boot ROM)",
                Start::Kernel { .. } => "Kernel ELF",
                Start::Snapshot { .. } => "Snapshot",
            };
            egui::ComboBox::from_id_salt("start")
                .selected_text(label)
                .width(190.0)
                .show_ui(ui, |ui| {
                    if ui
                        .selectable_label(machine.start == Start::PowerOn, "Power-on (boot ROM)")
                        .clicked()
                    {
                        machine.start = Start::PowerOn;
                    }
                    if ui
                        .selectable_label(
                            matches!(machine.start, Start::Kernel { .. }),
                            "Kernel ELF",
                        )
                        .clicked()
                        && !matches!(machine.start, Start::Kernel { .. })
                    {
                        machine.start = Start::Kernel { path: None };
                    }
                });
        });
    });
    match &mut machine.start {
        Start::Kernel { path } => rows.row(ui, "Kernel", |ui| {
            ui.add_enabled_ui(!locked, |ui| {
                file_controls(ui, path, "Choose", "None", false)
            });
        }),
        Start::Snapshot { dir } => rows.row(ui, "Snapshot", |ui| {
            theme::path_label(ui, dir.as_deref(), "None");
        }),
        Start::PowerOn => {
            rows.row(ui, "Boot ROM", |ui| {
                ui.add_enabled_ui(!locked, |ui| {
                    file_controls(
                        ui,
                        &mut machine.bootrom,
                        "Use dump",
                        "Built-in emulation",
                        true,
                    )
                })
                .response
                .on_hover_text(
                    "A dump of the Tegra X1's boot ROM (96 KiB from 0x100000), run on the boot \
                     processor from its reset vector instead of the high-level emulation.",
                );
            });
            rows.row(ui, "Payload", |ui| {
                ui.add_enabled_ui(!locked, |ui| {
                    file_controls(ui, &mut machine.payload, "Choose", "None", true)
                })
                .response
                .on_hover_text(
                    "A program for the boot processor (such as hekate), started where the boot \
                     ROM would have started package1. It finds its files on the microSD card.",
                );
            });
        }
    }
    rows.row(ui, "Report debug mode to kernel", |ui| {
        ui.add_enabled_ui(!locked, |ui| theme::switch(ui, &mut machine.debug_mode));
    });
}

fn controllers(ui: &mut Ui, machine: &mut Machine, locked: bool) {
    let mut rows = Rows::new();
    rows.row(ui, "Joy-Con on rails at start", |ui| {
        theme::switch(ui, &mut machine.joycons_at_start);
    });
    rows.row(ui, "Pro Controller", |ui| {
        ui.add_enabled_ui(!locked, |ui| {
            if machine.pro_controller.is_some() && theme::clear_button(ui).clicked() {
                machine.pro_controller = None;
            }
            let label = if machine.pro_controller.is_some() {
                "Change"
            } else {
                "Add"
            };
            if theme::small_button(ui, label)
                .on_hover_text(
                    "A Pro Controller in range of the console's radio. Its file holds what it \
                     remembers of the console it pairs with; pair it from the screen's \
                     controller button.",
                )
                .clicked()
                && let Some(path) =
                    save_file(machine.pro_controller.as_deref(), "pro-controller.bin")
            {
                machine.pro_controller = Some(path);
            }
            let paired = machine
                .pro_controller
                .as_deref()
                .is_some_and(|path| std::fs::metadata(path).is_ok_and(|meta| meta.len() > 0));
            let state = match (&machine.pro_controller, paired) {
                (None, _) => "None",
                (Some(_), false) => "Not paired",
                (Some(_), true) => "Paired",
            };
            ui.label(RichText::new(state).color(MUTED));
        });
    });
    rows.row(ui, "Amiibo", |ui| {
        if machine.amiibo.is_some() {
            theme::labelled_switch(ui, &mut machine.amiibo_writable, "Writable");
        }
        file_controls(ui, &mut machine.amiibo, "Choose", "None", true);
    });
}

/// An image row: the file, its writable switch, a change button.
fn image_row(
    rows: &mut Rows,
    ui: &mut Ui,
    label: &str,
    image: &mut Option<PathBuf>,
    writable: Option<&mut bool>,
    locked: bool,
) {
    rows.row(ui, label, |ui| {
        ui.add_enabled_ui(!locked, |ui| {
            if theme::small_button(ui, if image.is_some() { "Change" } else { "Choose" }).clicked()
                && let Some(path) = pick_file(image.as_deref())
            {
                *image = Some(path);
            }
            if image.is_some() && theme::clear_button(ui).clicked() {
                *image = None;
            }
            if let Some(writable) = writable {
                theme::labelled_switch(ui, writable, "Writable");
            }
            theme::path_label(ui, image.as_deref(), "None");
        });
    });
}

fn storage(ui: &mut Ui, machine: &mut Machine, locked: bool) {
    let mut rows = Rows::new();
    rows.row(ui, "Keys", |ui| {
        ui.add_enabled_ui(!locked, |ui| {
            file_controls(ui, &mut machine.keys, "Choose", "None", false)
        });
    });
    let mut writable = machine.emmc_writable;
    image_row(
        &mut rows,
        ui,
        "eMMC user area",
        &mut machine.emmc,
        Some(&mut writable),
        locked,
    );
    rows.row(ui, "eMMC boot0 / boot1", |ui| {
        ui.add_enabled_ui(!locked, |ui| {
            if theme::small_button(
                ui,
                if machine.boot0.is_some() {
                    "Change"
                } else {
                    "Choose"
                },
            )
            .clicked()
                && let Some(boot0) = pick_file(machine.boot0.as_deref())
            {
                let boot1 = pick_file(Some(&boot0));
                machine.boot0 = Some(boot0);
                machine.boot1 = boot1;
            }
            theme::labelled_switch(ui, &mut writable, "Writable").on_hover_text(
                "One switch for every eMMC image, the user area and both boot partitions \
                 (--emmc-writable). Off, writes stay in memory.",
            );
            let name = |path: &Option<PathBuf>| {
                path.as_deref().and_then(Path::file_name).map_or_else(
                    || "erased".into(),
                    |name| name.to_string_lossy().into_owned(),
                )
            };
            ui.label(
                RichText::new(theme::shorten(
                    &format!("{} · {}", name(&machine.boot0), name(&machine.boot1)),
                    28,
                ))
                .monospace()
                .color(MUTED),
            );
        });
    });
    machine.emmc_writable = writable;
    let mut sd_writable = machine.sd_writable;
    image_row(
        &mut rows,
        ui,
        "microSD",
        &mut machine.sd,
        Some(&mut sd_writable),
        locked,
    );
    machine.sd_writable = sd_writable;
    rows.row(ui, "Symbols", |ui| {
        file_controls(ui, &mut machine.symbols, "Choose", "None", true);
    });
}

/// Installing an NSP onto the machine's eMMC image (`install-nsp`), which
/// the console must have started from once and which must not be in use.
fn games(
    ui: &mut Ui,
    machine: &mut Machine,
    running: bool,
    install: &mut Option<Job>,
    installed: &mut Option<Result<String, String>>,
) {
    if let Some(outcome) = install.as_ref().and_then(Job::finished) {
        let printed = install
            .take()
            .map(|job| {
                let log = job.log.lock().unwrap();
                log.lines
                    .iter()
                    .filter(|line| line.text.starts_with("installed "))
                    .map(|line| line.text.clone())
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .unwrap_or_default();
        if outcome.is_ok() {
            machine.accept_forged_tickets = true;
        }
        *installed = Some(outcome.map(|()| printed));
    }
    let mut rows = Rows::new();
    rows.row(ui, "Install NSP", |ui| {
        let ready = !running && install.is_none() && machine.emmc.is_some() && machine.keys.is_some();
        let button = ui.add_enabled_ui(ready, |ui| theme::small_button(ui, "Install…")).inner;
        let button = button.on_disabled_hover_text(if running {
            "Stop the machine first: the image must not be in use."
        } else {
            "Needs the machine's eMMC image and keys."
        });
        if button.clicked()
            && let (Some(emmc), Some(keys)) = (&machine.emmc, &machine.keys)
            && let Some(nsp) = rfd::FileDialog::new()
                .add_filter("NSP", &["nsp"])
                .pick_file()
        {
            let ctx = ui.ctx().clone();
            *installed = None;
            *install = Some(Job::run(
                vec![(
                    child::emulator(),
                    vec![
                        "install-nsp".into(),
                        emmc.display().to_string(),
                        "--keys".into(),
                        keys.display().to_string(),
                        nsp.display().to_string(),
                    ],
                )],
                |_, _, _| Ok(()),
                move || ctx.request_repaint(),
            ));
        }
        if let Some(job) = install {
            ui.spinner();
            let log = job.log.lock().unwrap();
            if let Some(line) = log.lines.back() {
                ui.label(RichText::new(&line.text).color(MUTED));
            }
        } else if let Some(outcome) = installed {
            match outcome {
                Ok(titles) => ui.label(RichText::new(titles.as_str()).color(MUTED)),
                Err(error) => ui.label(RichText::new(error.as_str()).color(RED)),
            };
        }
    });
    rows.row(ui, "Converted tickets", |ui| {
        theme::labelled_switch(ui, &mut machine.accept_forged_tickets, "Accept").on_hover_text(
            "--accept-forged-tickets: ES gives out the title keys of tickets whose signature \
             an NSP's packer made, as consoles with signature patches do. Turned on by an \
             install.",
        );
    });
}

fn network(ui: &mut Ui, machine: &mut Machine, locked: bool) {
    let mut rows = Rows::new();
    rows.row(ui, "Wi-Fi network", |ui| {
        ui.add_enabled_ui(!locked, |ui| {
            text_field(ui, &mut machine.wifi_ssid, "No access point", 190.0)
        });
    });
    rows.row(ui, "Passphrase", |ui| {
        ui.add_enabled_ui(!locked && !machine.wifi_ssid.trim().is_empty(), |ui| {
            text_field(ui, &mut machine.wifi_passphrase, "Open network", 190.0)
        });
    });
    rows.row(ui, "Shared air", |ui| {
        text_field(ui, &mut machine.air, "127.0.0.1:9000", 190.0);
    });
    rows.row(ui, "DNS server", |ui| {
        ui.add_enabled_ui(!machine.wifi_ssid.trim().is_empty(), |ui| {
            text_field(ui, &mut machine.dns, "The host's resolver", 190.0)
        });
    });
    if machine.wifi_passphrase.len() > 63
        || (!machine.wifi_passphrase.is_empty() && machine.wifi_passphrase.len() < 8)
    {
        ui.label(
            RichText::new("A WPA2 passphrase has 8 to 63 characters.")
                .color(AMBER)
                .size(13.0),
        );
    }
}

fn homebrew(ui: &mut Ui, machine: &mut Machine, locked: bool) {
    let homebrew = &mut machine.homebrew;
    ui.add_enabled_ui(!locked, |ui| {
        let mut rows = Rows::new();
        rows.row(ui, "hbmenu (NRO)", |ui| {
            file_controls(ui, &mut homebrew.nro, "Choose", "None", true);
        });
        if homebrew.nro.is_some() {
            rows.row(ui, "hbloader (NSP)", |ui| {
                file_controls(
                    ui,
                    &mut homebrew.hbl,
                    "Choose",
                    "hbl.nsp beside the NRO",
                    true,
                );
            });
            rows.row(ui, "Run in place of", |ui| {
                text_field(ui, &mut homebrew.run_in, "Album applet", 190.0);
            });
            rows.row(ui, "Programs on the card", |ui| {
                if theme::small_button(ui, "Add").clicked()
                    && let Some(path) = pick_file(homebrew.apps.last().map(PathBuf::as_path))
                {
                    homebrew.apps.push(path);
                }
                ui.label(RichText::new(format!("{}", homebrew.apps.len())).color(MUTED));
            });
            let mut removed = None;
            for (index, app) in homebrew.apps.iter().enumerate() {
                ui.horizontal(|ui| {
                    ui.add_space(16.0);
                    theme::path_label(ui, Some(app), "");
                    if theme::clear_button(ui).clicked() {
                        removed = Some(index);
                    }
                });
            }
            if let Some(index) = removed {
                homebrew.apps.remove(index);
            }
        }
        rows.row(ui, "Homebrew keys", |ui| {
            theme::switch(ui, &mut homebrew.keys).on_hover_text(
                "Replace the filesystem process's and the loader's public keys of signature \
                 key generation 1 with the emulator's test key (a modification of the \
                 firmware's code; --homebrew implies it).",
            );
        });
        rows.row(ui, "Disable program verification", |ui| {
            theme::switch(ui, &mut homebrew.disable_verification);
        });
        rows.row(ui, "Replaced programs", |ui| {
            if theme::small_button(ui, "Add").clicked()
                && let Some(path) = pick_file(None)
            {
                homebrew.replacements.push((String::new(), path));
            }
        });
        let mut removed = None;
        for (index, (program, nsp)) in homebrew.replacements.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                ui.add_space(16.0);
                text_field(ui, program, "program ID", 150.0);
                ui.label("=");
                theme::path_label(ui, Some(nsp), "");
                if theme::clear_button(ui).clicked() {
                    removed = Some(index);
                }
            });
        }
        if let Some(index) = removed {
            homebrew.replacements.remove(index);
        }
    });
}

/// An editable list of places for `--break`, `--trace` and `--poke`.
fn places(ui: &mut Ui, id: &str, places: &mut Vec<String>, hint: &str) {
    let mut removed = None;
    for (index, place) in places.iter_mut().enumerate() {
        ui.horizontal(|ui| {
            ui.add(
                TextEdit::singleline(place)
                    .id_salt((id, index))
                    .hint_text(hint)
                    .desired_width(320.0)
                    .code_editor(),
            );
            if theme::clear_button(ui).clicked() {
                removed = Some(index);
            }
        });
    }
    if let Some(index) = removed {
        places.remove(index);
    }
    ui.horizontal(|ui| {
        if ui.small_button("+ Add").clicked() {
            places.push(String::new());
        }
    });
}

fn optional_number(ui: &mut Ui, value: &mut Option<u32>, default: u32) {
    let mut on = value.is_some();
    theme::switch(ui, &mut on);
    match (on, value.as_mut()) {
        (true, Some(number)) => {
            ui.add(egui::DragValue::new(number).range(1..=1_000_000));
        }
        (true, None) => *value = Some(default),
        (false, _) => *value = None,
    }
}

pub fn debug(ui: &mut Ui, entry: &mut Entry) {
    let crash_dir = entry.stored.dir.join("crashes");
    let debug = &mut entry.stored.machine.debug;
    ui.label(
        RichText::new(
            "These apply from the next start or restore; they are not part of a snapshot.",
        )
        .size(13.0)
        .color(MUTED),
    );
    ui.add_space(6.0);
    ui.columns(2, |columns| {
        let (left, right) = columns.split_at_mut(1);
        theme::section(&mut left[0], "Output", |ui| {
            let mut rows = Rows::new();
            for (value, label, help) in [
                (&mut debug.verbose, "Narrate the boot", "--verbose: cores started, programs loaded, services registered, fatal errors, kernel debug text. Watches every supervisor call."),
                (&mut debug.log, "Process logs", "--log: what processes log, as the log manager receives it."),
                (&mut debug.trace_smc, "Secure monitor calls", "--trace-smc"),
                (&mut debug.trace_svc, "Supervisor calls", "--trace-svc"),
                (&mut debug.trace_ipc, "IPC requests", "--trace-ipc: every request to a service and its answer (spl data includes key material)."),
                (&mut debug.trace_devices, "Device register accesses", "--trace-devices"),
                (&mut debug.trace_times, "Times on trace lines", "--trace-times"),
                (&mut debug.list_threads, "List threads at stop", "--list-threads: the guest's threads from the kernel's objects when the machine stops (kernel 23.5.0), shown in Threads."),
            ] {
                rows.row(ui, label, |ui| {
                    theme::switch(ui, value).on_hover_text(help);
                });
            }
        });
        let right = &mut right[0];
        theme::section(right, "Run", |ui| {
            let mut rows = Rows::new();
            rows.row(ui, "Survey unmodeled devices", |ui| {
                theme::switch(ui, &mut debug.survey).on_hover_text(
                    "--survey: answer reads of unmodeled devices with zero and print each. Not a faithful run.",
                );
            });
            rows.row(ui, "Crash dumps", |ui| {
                theme::switch(ui, &mut debug.crash_dumps)
                    .on_hover_text(format!("--crash-dump {}", crash_dir.display()));
            });
            rows.row(ui, "Wall-clock limit (s)", |ui| {
                let mut on = debug.timeout.is_some();
                theme::switch(ui, &mut on);
                match (on, debug.timeout.as_mut()) {
                    (true, Some(seconds)) => {
                        ui.add(egui::DragValue::new(seconds).range(1.0..=86_400.0));
                    }
                    (true, None) => debug.timeout = Some(60.0),
                    (false, _) => debug.timeout = None,
                }
            });
            rows.row(ui, "Single-step after a break", |ui| optional_number(ui, &mut debug.steps, 16));
            rows.row(ui, "Boot processor history at stop", |ui| {
                optional_number(ui, &mut debug.trace_bpmp, 64)
            });
            rows.row(ui, "GPU capture", |ui| {
                if theme::small_button(ui, if debug.gpu_capture.is_some() { "Change" } else { "Choose" }).clicked()
                    && let Some(path) = save_file(debug.gpu_capture.as_deref(), "gpu-capture.bin")
                {
                    debug.gpu_capture = Some(path);
                }
                if debug.gpu_capture.is_some() && theme::clear_button(ui).clicked() {
                    debug.gpu_capture = None;
                }
                theme::path_label(ui, debug.gpu_capture.as_deref(), "None");
            });
        });
        right.add_space(12.0);
        theme::section(right, "Breakpoints", |ui| {
            ui.label(RichText::new("Kernel image address (0x…), @0x… for a user address, or a symbol. Each takes one of the CPU's hardware breakpoints.").size(12.5).color(MUTED));
            ui.label("Stop at");
            places(ui, "break", &mut debug.breaks, "0x80060000");
            ui.label("Print registers at");
            places(ui, "trace", &mut debug.traces, "@0x737e53dd08");
            ui.label("Poke");
            places(ui, "poke", &mut debug.pokes, "WHERE:xN+OFFSET:SIZE:VALUE");
        });
    });
}

/// The guest's threads as the last stop listed them (`--list-threads`).
pub fn threads(ui: &mut Ui, entry: &mut Entry) {
    let Some(child) = &entry.child else {
        theme::section(ui, "Threads", |ui| {
            ui.label(RichText::new("The machine has not run in this session.").color(MUTED));
        });
        return;
    };
    let log = child.log.lock().unwrap();
    let start = log
        .lines
        .iter()
        .rposition(|line| line.text.starts_with("threads when the machine stopped"));
    theme::section(ui, "Threads when the machine stopped", |ui| {
        let Some(start) = start else {
            let text = if entry.stored.machine.debug.list_threads {
                "The machine has not stopped yet; its threads are listed when it does."
            } else {
                "Turn on \"List threads at stop\" in Debug to have them listed when the machine stops."
            };
            ui.label(RichText::new(text).color(MUTED));
            return;
        };
        let lines: Vec<&str> = log
            .lines
            .iter()
            .skip(start + 1)
            .map(|line| line.text.as_str())
            .take_while(|text| text.starts_with("  "))
            .collect();
        ui.label(RichText::new(format!("{} threads", lines.len())).color(MUTED));
        egui::ScrollArea::both().max_height(520.0).show(ui, |ui| {
            for line in lines {
                ui.label(RichText::new(line.trim_start()).monospace().size(12.5));
            }
        });
    });
}

pub enum SnapshotAction {
    Restore(PathBuf),
    Delete(PathBuf),
    NewMachine(PathBuf),
}

pub fn snapshots(ui: &mut Ui, entry: &mut Entry) -> Option<SnapshotAction> {
    let mut action = None;
    let running = entry.running();
    theme::section(ui, "Snapshots", |ui| {
        if entry.snapshots.is_empty() {
            ui.label(
                RichText::new("No snapshots yet. Save state while the machine runs to take one.")
                    .color(MUTED),
            );
            return;
        }
        let mut rows = Rows::new();
        for snapshot in &entry.snapshots {
            let detail = [
                snapshot
                    .info
                    .guest_seconds
                    .map(|seconds| format!("guest {seconds:.1} s")),
                snapshot
                    .info
                    .ram_bytes
                    .map(|bytes| format!("{:.0} MiB of DRAM", bytes as f64 / (1 << 20) as f64)),
            ]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .join(" · ");
            let label = RichText::new(format!("{}   ", snapshot.name));
            rows.row_rich(ui, label, |ui| {
                if theme::small_button(ui, "Delete").clicked() {
                    action = Some(SnapshotAction::Delete(snapshot.dir.clone()));
                }
                if theme::small_button(ui, "New machine")
                    .on_hover_text("A machine that starts from this snapshot")
                    .clicked()
                {
                    action = Some(SnapshotAction::NewMachine(snapshot.dir.clone()));
                }
                if ui
                    .add_enabled(!running, egui::Button::new("Restore"))
                    .on_disabled_hover_text("Stop the machine first")
                    .clicked()
                {
                    action = Some(SnapshotAction::Restore(snapshot.dir.clone()));
                }
                ui.label(RichText::new(detail).color(MUTED).size(13.0));
            });
        }
    });
    action
}

pub fn log(ui: &mut Ui, entry: &mut Entry) {
    let Some(child) = &entry.child else {
        theme::section(ui, "Log", |ui| {
            ui.label(RichText::new("The machine has not run in this session.").color(MUTED));
        });
        return;
    };
    theme::section(ui, "Log", |ui| {
        ui.horizontal(|ui| {
            ui.add(
                TextEdit::singleline(&mut entry.log_filter)
                    .hint_text("Filter")
                    .desired_width(260.0),
            );
            theme::labelled_switch(ui, &mut entry.log_follow, "Follow");
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if theme::small_button(ui, "Copy").clicked() {
                    let text: Vec<String> = child
                        .log
                        .lock()
                        .unwrap()
                        .lines
                        .iter()
                        .map(|line| line.text.clone())
                        .collect();
                    ui.ctx().copy_text(text.join("\n"));
                }
                if theme::small_button(ui, "Clear").clicked() {
                    child.log.lock().unwrap().lines.clear();
                }
            });
        });
        let log = child.log.lock().unwrap();
        let filter = entry.log_filter.to_lowercase();
        let lines: Vec<_> = log
            .lines
            .iter()
            .filter(|line| filter.is_empty() || line.text.to_lowercase().contains(&filter))
            .collect();
        let height = ui.text_style_height(&egui::TextStyle::Monospace);
        egui::ScrollArea::both()
            .max_height(560.0)
            .auto_shrink([false, false])
            .stick_to_bottom(entry.log_follow)
            .show_rows(ui, height, lines.len(), |ui, range| {
                for line in &lines[range] {
                    ui.label(
                        RichText::new(&line.text)
                            .monospace()
                            .color(line_color(line.stream, &line.text)),
                    );
                }
            });
    });
}

/// How a line of output is shown: warnings and errors stand out.
pub fn line_color(stream: Stream, text: &str) -> Color32 {
    if stream == Stream::Ui {
        return theme::BLUE;
    }
    let lower = text.to_ascii_lowercase();
    if lower.starts_with("survey:") || lower.starts_with("note:") {
        AMBER
    } else if lower.contains("error") || lower.contains("panic") || lower.contains("fatal") {
        RED
    } else if stream == Stream::Err {
        Color32::from_rgb(0xD8, 0xDC, 0xE6)
    } else {
        theme::TEXT
    }
}
