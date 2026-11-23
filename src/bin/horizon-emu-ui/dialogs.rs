//! Dialogs: a new machine, the eMMC and microSD images it can be given
//! (with the firmware downloaded from Nintendo or taken from a folder),
//! and deleting a machine.

use std::path::{Path, PathBuf};

use eframe::egui::{
    self, Align, Color32, CornerRadius, Frame, Layout, Margin, RichText, ScrollArea, Stroke,
    TextEdit, Ui, Vec2,
};

use crate::{
    app::{App, View},
    child::{Job, Stream},
    model::{self, Machine, Renderer, Settings, Start},
    tabs::{self, line_color},
    theme::{self, MUTED, RED, Rows, white},
};

pub enum Dialog {
    NewMachine(Box<NewMachine>),
    Delete(usize),
    Duplicate(Machine),
}

impl Dialog {
    pub fn new_machine(settings: &Settings, count: usize) -> Self {
        let name = if count == 0 {
            "Cold boot".to_string()
        } else {
            format!("Machine {}", count + 1)
        };
        Self::NewMachine(Box::new(NewMachine {
            machine: Machine {
                name,
                keys: settings.keys.clone(),
                wifi_ssid: "lab".into(),
                ..Machine::default()
            },
            sub: None,
            error: None,
        }))
    }
}

pub struct NewMachine {
    machine: Machine,
    sub: Option<Sub>,
    error: Option<String>,
}

enum Sub {
    Emmc(FormatEmmc),
    Sd(FormatSd),
}

/// Where the firmware installed on a new eMMC image comes from.
#[derive(Clone, PartialEq)]
enum Firmware {
    /// Downloaded from Nintendo's servers as the console whose
    /// calibration data is given (`download-firmware`).
    Download,
    /// One downloaded before.
    Downloaded(PathBuf),
    /// A folder of `.nca` files the user has.
    Folder(Option<PathBuf>),
    None,
}

struct FormatEmmc {
    image: PathBuf,
    firmware: Firmware,
    prodinfo: Option<PathBuf>,
    boot_images: bool,
    job: Option<Job>,
    error: Option<String>,
}

struct FormatSd {
    image: PathBuf,
    size: String,
    adds: Vec<(PathBuf, String)>,
    job: Option<Job>,
    error: Option<String>,
}

fn images_dir() -> PathBuf {
    model::data_dir().join("images")
}

/// A file name for a new image of `machine`, `what` being its kind, that
/// does not exist yet.
fn new_image(machine: &str, what: &str) -> PathBuf {
    let base: String = machine
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    let base = base.trim_matches('-');
    let dir = images_dir();
    let mut path = dir.join(format!("{base}-{what}.img"));
    let mut index = 2;
    while path.exists() {
        path = dir.join(format!("{base}-{what}-{index}.img"));
        index += 1;
    }
    path
}

fn boot_image(image: &Path, which: &str) -> PathBuf {
    let stem = image.file_stem().unwrap_or_default().to_string_lossy();
    image.with_file_name(format!("{stem}-{which}.img"))
}

/// The panel a dialog's rows sit in.
fn group<R>(ui: &mut Ui, add: impl FnOnce(&mut Ui) -> R) -> R {
    theme::group(ui, add)
}

/// A dialog: a thicker, darker pane of glass over a dimmed window.
fn modal_frame() -> Frame {
    Frame::new()
        .fill(Color32::from_rgba_unmultiplied(0x22, 0x28, 0x3E, 0xE4))
        .stroke(Stroke::new(1.0, white(70)))
        .corner_radius(CornerRadius::same(30))
        .inner_margin(Margin::same(30))
        .shadow(egui::Shadow {
            offset: [0, 20],
            blur: 50,
            spread: 0,
            color: theme::black(110),
        })
}

/// A row with a path and its "Choose…" button.
fn choose_row(
    rows: &mut Rows,
    ui: &mut Ui,
    label: &str,
    value: &mut Option<PathBuf>,
    none: &str,
    writable: Option<&mut bool>,
) {
    rows.row(ui, label, |ui| {
        if theme::small_button(ui, "Choose…").clicked()
            && let Some(path) = tabs::pick_file(value.as_deref())
        {
            *value = Some(path);
        }
        if let Some(writable) = writable {
            theme::labelled_switch(ui, writable, "Writable");
        }
        if value.is_some() && theme::clear_button(ui).clicked() {
            *value = None;
        }
        ui.with_layout(Layout::left_to_right(Align::Center), |ui| {
            ui.add_space(24.0);
            theme::path_label(ui, value.as_deref(), none);
        });
    });
}

pub fn show(app: &mut App, ctx: &egui::Context) {
    let Some(dialog) = app.dialog.take() else {
        return;
    };
    app.dialog = match dialog {
        Dialog::NewMachine(mut form) => match new_machine(app, ctx, &mut form) {
            Outcome::Open => Some(Dialog::NewMachine(form)),
            Outcome::Done => {
                if form.machine.keys.is_some() {
                    app.settings.keys = form.machine.keys.clone();
                }
                app.add_machine(form.machine);
                None
            }
            Outcome::Cancel => None,
        },
        Dialog::Duplicate(machine) => {
            app.add_machine(machine);
            None
        }
        Dialog::Delete(index) => delete(app, ctx, index),
    };
}

enum Outcome {
    Open,
    Done,
    Cancel,
}

fn delete(app: &mut App, ctx: &egui::Context, index: usize) -> Option<Dialog> {
    let Some(entry) = app.entries.get(index) else {
        return None;
    };
    let (name, dir) = (entry.machine().name.clone(), entry.stored.dir.clone());
    let mut outcome = Outcome::Open;
    let response = egui::Modal::new(egui::Id::unique("delete")).frame(modal_frame()).show(ctx, |ui| {
        ui.set_width(440.0);
        ui.label(RichText::new(format!("Delete {name}?")).size(22.0).strong());
        ui.label(
            RichText::new(format!(
                "Its folder goes, with its snapshots, screenshots and crash dumps:\n{}\nThe images it uses stay where they are.",
                dir.display()
            ))
            .color(MUTED),
        );
        ui.add_space(12.0);
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            if theme::big_button(ui, "Delete", false).clicked() {
                outcome = Outcome::Done;
            }
            if theme::big_button(ui, "Cancel", false).clicked() {
                outcome = Outcome::Cancel;
            }
        });
    });
    if response.should_close() {
        outcome = Outcome::Cancel;
    }
    match outcome {
        Outcome::Open => Some(Dialog::Delete(index)),
        Outcome::Cancel => None,
        Outcome::Done => {
            if let Err(error) = std::fs::remove_dir_all(&dir) {
                app.toast(format!("{}: {error}", dir.display()), true);
                return None;
            }
            app.entries.remove(index);
            app.selected = (!app.entries.is_empty()).then(|| index.min(app.entries.len() - 1));
            app.view = View::Machine;
            None
        }
    }
}

fn new_machine(app: &mut App, ctx: &egui::Context, form: &mut NewMachine) -> Outcome {
    let mut outcome = Outcome::Open;
    let blocked = form.sub.is_some();
    let response = egui::Modal::new(egui::Id::unique("new-machine"))
        .frame(modal_frame())
        .show(ctx, |ui| {
            ui.set_width(860.0);
            ui.add_enabled_ui(!blocked, |ui| {
                let height = ctx.content_rect().height() - 120.0;
                ScrollArea::vertical()
                    .max_height(height)
                    .min_scrolled_height(height)
                    .show(ui, |ui| {
                        outcome = new_machine_form(app, ui, form);
                    });
            });
        });
    if response.should_close() && !blocked {
        outcome = Outcome::Cancel;
    }
    match form.sub.take() {
        Some(Sub::Emmc(mut emmc)) => {
            if format_emmc(app, ctx, &mut emmc, &mut form.machine) {
                form.sub = Some(Sub::Emmc(emmc));
            }
        }
        Some(Sub::Sd(mut sd)) => {
            if format_sd(ctx, &mut sd, &mut form.machine) {
                form.sub = Some(Sub::Sd(sd));
            }
        }
        None => {}
    }
    outcome
}

fn new_machine_form(app: &mut App, ui: &mut Ui, form: &mut NewMachine) -> Outcome {
    let mut outcome = Outcome::Open;
    let machine = &mut form.machine;
    ui.label(RichText::new("New machine").size(26.0).strong());
    ui.label(
        RichText::new("A saved set of machine flags. DRAM, seed, keys and media are fixed once a snapshot is taken.")
            .color(MUTED),
    );
    ui.add_space(14.0);
    ui.columns(2, |columns| {
        columns[0].label(theme::caption("Name"));
        columns[0].add(
            TextEdit::singleline(&mut machine.name)
                .desired_width(f32::INFINITY)
                .margin(Margin::symmetric(10, 8)),
        );
        columns[1].label(theme::caption("Start from"));
        let mut kind = match machine.start {
            Start::PowerOn => 0,
            Start::Kernel { .. } => 1,
            Start::Snapshot { .. } => 2,
        };
        if theme::segmented(
            &mut columns[1],
            &mut kind,
            &[
                (0, "  Power-on  "),
                (1, "  Kernel ELF  "),
                (2, "  Snapshot  "),
            ],
        ) {
            machine.start = match kind {
                0 => Start::PowerOn,
                1 => Start::Kernel { path: None },
                _ => Start::Snapshot { dir: None },
            };
        }
    });
    ui.add_space(12.0);
    match &mut machine.start {
        Start::Kernel { path } => {
            ui.label(theme::caption("Kernel"));
            group(ui, |ui| {
                let mut rows = Rows::new();
                choose_row(
                    &mut rows,
                    ui,
                    "Kernel ELF",
                    path,
                    "package2's kernel wrapped as an ELF",
                    None,
                );
            });
            ui.add_space(12.0);
        }
        Start::Snapshot { dir } => {
            ui.label(theme::caption("Snapshot"));
            group(ui, |ui| {
                let mut rows = Rows::new();
                rows.row(ui, "Snapshot folder", |ui| {
                    if theme::small_button(ui, "Choose…").clicked()
                        && let Some(folder) = tabs::pick_folder(dir.as_deref())
                    {
                        *dir = Some(folder);
                    }
                    ui.with_layout(Layout::left_to_right(Align::Center), |ui| {
                        ui.add_space(24.0);
                        theme::path_label(ui, dir.as_deref(), "A folder with boot.json");
                    });
                });
            });
            ui.label(
                RichText::new("The snapshot says what the machine is: its DRAM, seed, keys, media and Wi-Fi network.")
                    .size(13.0)
                    .color(MUTED),
            );
            ui.add_space(12.0);
        }
        Start::PowerOn => {}
    }
    let snapshot = matches!(machine.start, Start::Snapshot { .. });
    if !snapshot {
        ui.horizontal(|ui| {
            ui.label(theme::caption("Media"));
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if ui.link("Format SD…").clicked() {
                    form.sub = Some(Sub::Sd(FormatSd {
                        image: new_image(&machine.name, "sd"),
                        size: "4G".into(),
                        adds: Vec::new(),
                        job: None,
                        error: None,
                    }));
                }
                if ui.link("Format eMMC…").clicked() {
                    let downloaded = model::downloaded_firmware();
                    let firmware = if app.settings.prodinfo.is_some() {
                        Firmware::Download
                    } else if let Some((_, dir)) = downloaded.first() {
                        Firmware::Downloaded(dir.clone())
                    } else {
                        Firmware::Download
                    };
                    form.sub = Some(Sub::Emmc(FormatEmmc {
                        image: new_image(&machine.name, "emmc"),
                        firmware,
                        prodinfo: app.settings.prodinfo.clone(),
                        boot_images: machine.start == Start::PowerOn,
                        job: None,
                        error: None,
                    }));
                }
            });
        });
        group(ui, |ui| {
            let mut rows = Rows::new();
            choose_row(&mut rows, ui, "Keys", &mut machine.keys, "None", None);
            choose_row(
                &mut rows,
                ui,
                "eMMC user area",
                &mut machine.emmc,
                "None",
                Some(&mut machine.emmc_writable),
            );
            rows.row(ui, "eMMC boot0 / boot1", |ui| {
                if theme::small_button(ui, "Choose…").clicked()
                    && let Some(boot0) = tabs::pick_file(machine.boot0.as_deref())
                {
                    machine.boot1 = tabs::pick_file(Some(&boot0));
                    machine.boot0 = Some(boot0);
                }
                theme::labelled_switch(ui, &mut machine.emmc_writable, "Writable")
                    .on_hover_text("One switch for every eMMC image (--emmc-writable).");
                ui.with_layout(Layout::left_to_right(Align::Center), |ui| {
                    ui.add_space(24.0);
                    let name = |path: &Option<PathBuf>| {
                        path.as_deref().and_then(Path::file_name).map_or_else(
                            || "erased".into(),
                            |name| name.to_string_lossy().into_owned(),
                        )
                    };
                    ui.label(
                        RichText::new(format!(
                            "{} · {}",
                            name(&machine.boot0),
                            name(&machine.boot1)
                        ))
                        .monospace()
                        .color(MUTED),
                    );
                });
            });
            choose_row(
                &mut rows,
                ui,
                "microSD",
                &mut machine.sd,
                "None",
                Some(&mut machine.sd_writable),
            );
            choose_row(&mut rows, ui, "Symbols", &mut machine.symbols, "None", None);
        });
        ui.add_space(12.0);
        if machine.start == Start::PowerOn {
            ui.label(theme::caption("Boot processor"));
            group(ui, |ui| {
                let mut rows = Rows::new();
                choose_row(
                    &mut rows,
                    ui,
                    "Boot ROM",
                    &mut machine.bootrom,
                    "High-level emulated (no dump)",
                    None,
                );
                choose_row(
                    &mut rows,
                    ui,
                    "Payload",
                    &mut machine.payload,
                    "none — package1 from boot0/boot1",
                    None,
                );
            });
            ui.add_space(12.0);
        }
        ui.columns(3, |columns| {
            columns[0].label(theme::caption("DRAM"));
            theme::segmented(
                &mut columns[0],
                &mut machine.dram,
                &[(4, "    4 GiB    "), (6, "    6 GiB    ")],
            );
            columns[1].label(theme::caption("Monitor seed"));
            let mut seed = machine.seed.to_string();
            if columns[1]
                .add(
                    TextEdit::singleline(&mut seed)
                        .desired_width(f32::INFINITY)
                        .margin(Margin::symmetric(10, 8))
                        .code_editor(),
                )
                .changed()
            {
                machine.seed = seed.trim().parse().unwrap_or(machine.seed);
            }
            columns[2].label(theme::caption("Renderer"));
            theme::segmented(
                &mut columns[2],
                &mut machine.renderer,
                &[
                    (Renderer::Metal, "    Metal    "),
                    (Renderer::Software, "  Software  "),
                ],
            );
        });
        ui.add_space(12.0);
        ui.label(theme::caption("Wi-Fi access point"));
        ui.columns(3, |columns| {
            columns[0].label(RichText::new("SSID").size(13.0));
            columns[0].add(
                TextEdit::singleline(&mut machine.wifi_ssid)
                    .hint_text("no access point")
                    .desired_width(f32::INFINITY)
                    .margin(Margin::symmetric(10, 8)),
            );
            columns[1].label(RichText::new("WPA2 passphrase").size(13.0));
            columns[1].add(
                TextEdit::singleline(&mut machine.wifi_passphrase)
                    .hint_text("open network")
                    .desired_width(f32::INFINITY)
                    .margin(Margin::symmetric(10, 8)),
            );
            columns[2].label(RichText::new("Shared air (UDP)").size(13.0));
            columns[2].add(
                TextEdit::singleline(&mut machine.air)
                    .hint_text("127.0.0.1:9000")
                    .desired_width(f32::INFINITY)
                    .margin(Margin::symmetric(10, 8))
                    .code_editor(),
            );
        });
    }
    if let Some(error) = &form.error {
        ui.add_space(8.0);
        ui.label(RichText::new(error).color(RED));
    }
    ui.add_space(16.0);
    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
        if theme::big_button(ui, "Create machine", true).clicked() {
            form.error = validate(&form.machine);
            if form.error.is_none() {
                outcome = Outcome::Done;
            }
        }
        if theme::big_button(ui, "Cancel", false).clicked() {
            outcome = Outcome::Cancel;
        }
    });
    outcome
}

/// What keeps `machine` from being created as it is.
fn validate(machine: &Machine) -> Option<String> {
    if machine.name.trim().is_empty() {
        return Some("Give the machine a name.".into());
    }
    match &machine.start {
        Start::Kernel { path: None } => return Some("Choose the kernel ELF.".into()),
        Start::Snapshot { dir: None } => return Some("Choose the snapshot.".into()),
        Start::Snapshot { dir: Some(dir) } if !dir.join("boot.json").is_file() => {
            return Some(format!(
                "{} is not a snapshot (no boot.json).",
                dir.display()
            ));
        }
        _ => {}
    }
    let passphrase = machine.wifi_passphrase.len();
    if !machine.wifi_ssid.trim().is_empty() && passphrase != 0 && !(8..=63).contains(&passphrase) {
        return Some("A WPA2 passphrase has 8 to 63 characters.".into());
    }
    None
}

/// The output of a job, as a dialog shows it.
fn job_output(ui: &mut Ui, job: &Job) {
    let log = job.log.lock().unwrap();
    Frame::new()
        .fill(Color32::from_rgba_unmultiplied(0x10, 0x14, 0x20, 0x90))
        .corner_radius(CornerRadius::same(10))
        .inner_margin(Margin::same(10))
        .show(ui, |ui| {
            ScrollArea::vertical()
                .max_height(200.0)
                .stick_to_bottom(true)
                .auto_shrink([false, true])
                .show(ui, |ui| {
                    for line in &log.lines {
                        let text = if line.stream == Stream::Ui {
                            theme::shorten(&line.text, 140)
                        } else {
                            line.text.clone()
                        };
                        ui.label(
                            RichText::new(text)
                                .monospace()
                                .size(12.0)
                                .color(line_color(line.stream, &line.text)),
                        );
                    }
                });
        });
}

/// The eMMC dialog; `false` once it is closed.
fn format_emmc(
    app: &mut App,
    ctx: &egui::Context,
    form: &mut FormatEmmc,
    machine: &mut Machine,
) -> bool {
    let mut open = true;
    let running = form
        .job
        .as_ref()
        .is_some_and(|job| job.finished().is_none());
    if let Some(job) = &form.job
        && let Some(result) = job.finished()
    {
        match result {
            Ok(()) => {
                machine.emmc = Some(form.image.clone());
                machine.emmc_writable = true;
                if form.boot_images {
                    machine.boot0 = Some(boot_image(&form.image, "boot0"));
                    machine.boot1 = Some(boot_image(&form.image, "boot1"));
                }
                app.toast(format!("eMMC image made: {}", form.image.display()), false);
                return false;
            }
            Err(error) => {
                form.error = Some(error);
                form.job = None;
            }
        }
    }
    let downloaded = model::downloaded_firmware();
    let response = egui::Modal::new(egui::Id::unique("format-emmc")).frame(modal_frame()).show(ctx, |ui| {
        ui.set_width(640.0);
        ui.label(RichText::new("Format eMMC").size(22.0).strong());
        ui.label(
            RichText::new(
                "A new eMMC user-area image (a sparse file), partitioned and formatted as the console \
                 does, with calibration data for the virtual console and the firmware installed.",
            )
            .color(MUTED),
        );
        ui.add_space(10.0);
        ui.add_enabled_ui(!running, |ui| {
            group(ui, |ui| {
                let mut rows = Rows::new();
                rows.row(ui, "Image", |ui| {
                    if theme::small_button(ui, "Change…").clicked()
                        && let Some(path) = tabs::save_file(Some(&form.image), &form.image.file_name().unwrap_or_default().to_string_lossy())
                    {
                        form.image = path;
                    }
                    ui.with_layout(Layout::left_to_right(Align::Center), |ui| {
                        ui.add_space(24.0);
                        theme::path_label(ui, Some(&form.image), "");
                    });
                });
                choose_row(&mut rows, ui, "Keys", &mut machine.keys, "Needed", None);
                rows.row(ui, "Boot partitions", |ui| {
                    theme::labelled_switch(ui, &mut form.boot_images, "Also make boot0 and boot1 images")
                        .on_hover_text("For power-on: the BCTs and package1 of the firmware, beside the image.");
                });
            });
            ui.add_space(10.0);
            ui.label(theme::caption("Firmware"));
            group(ui, |ui| {
                ui.radio_value(&mut form.firmware, Firmware::Download, "Download the newest from Nintendo");
                if form.firmware == Firmware::Download {
                    ui.horizontal(|ui| {
                        ui.add_space(26.0);
                        ui.label(RichText::new("As the console whose PRODINFO and keys are given:").size(13.0).color(MUTED));
                    });
                    ui.horizontal(|ui| {
                        ui.add_space(26.0);
                        theme::path_label(ui, form.prodinfo.as_deref(), "decrypted PRODINFO (CAL0)");
                        if theme::small_button(ui, "Choose…").clicked()
                            && let Some(path) = tabs::pick_file(form.prodinfo.as_deref())
                        {
                            form.prodinfo = Some(path);
                        }
                    });
                }
                for (version, dir) in &downloaded {
                    ui.radio_value(&mut form.firmware, Firmware::Downloaded(dir.clone()), format!("{version}, downloaded before"));
                }
                let folder = match &form.firmware {
                    Firmware::Folder(folder) => folder.clone(),
                    _ => None,
                };
                if ui.radio(matches!(form.firmware, Firmware::Folder(_)), "A folder of .nca files").clicked() {
                    form.firmware = Firmware::Folder(folder.clone());
                }
                if let Firmware::Folder(folder) = &mut form.firmware {
                    ui.horizontal(|ui| {
                        ui.add_space(26.0);
                        theme::path_label(ui, folder.as_deref(), "None");
                        if theme::small_button(ui, "Choose…").clicked()
                            && let Some(path) = tabs::pick_folder(folder.as_deref())
                        {
                            *folder = Some(path);
                        }
                    });
                }
                ui.radio_value(&mut form.firmware, Firmware::None, "None (formatted only)");
            });
        });
        if let Some(job) = &form.job {
            ui.add_space(10.0);
            job_output(ui, job);
        }
        if let Some(error) = &form.error {
            ui.add_space(8.0);
            ui.label(RichText::new(error).color(RED));
        }
        ui.add_space(14.0);
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            let label = if running { "Working…" } else { "Format" };
            if ui.add_enabled_ui(!running, |ui| theme::big_button(ui, label, true)).inner.clicked() {
                match emmc_job(app, ctx, form, machine) {
                    Ok(job) => {
                        form.error = None;
                        form.job = Some(job);
                    }
                    Err(error) => form.error = Some(error),
                }
            }
            if ui.add_enabled_ui(!running, |ui| theme::big_button(ui, "Cancel", false)).inner.clicked() {
                open = false;
            }
        });
    });
    if response.should_close() && !running {
        open = false;
    }
    if running {
        ctx.request_repaint_after(std::time::Duration::from_millis(200));
    }
    open
}

/// The commands that make the image: the download first, if it is one.
fn emmc_job(
    app: &mut App,
    ctx: &egui::Context,
    form: &FormatEmmc,
    machine: &Machine,
) -> Result<Job, String> {
    let keys = machine.keys.clone().ok_or("Choose the keys first.")?;
    if form.image.exists() {
        return Err(format!(
            "{} exists already; choose a new file.",
            form.image.display()
        ));
    }
    let boot = [
        boot_image(&form.image, "boot0"),
        boot_image(&form.image, "boot1"),
    ];
    if form.boot_images
        && let Some(existing) = boot.iter().find(|path| path.exists())
    {
        return Err(format!("{} exists already.", existing.display()));
    }
    if let Some(dir) = form.image.parent() {
        std::fs::create_dir_all(dir).map_err(|error| format!("{}: {error}", dir.display()))?;
    }
    let emulator = app.emulator.clone();
    let text = |path: &Path| path.display().to_string();
    let mut steps = Vec::new();
    let downloading = form.firmware == Firmware::Download;
    let firmware = match &form.firmware {
        Firmware::Download => {
            let prodinfo = form
                .prodinfo
                .clone()
                .ok_or("Choose the console's PRODINFO to download with.")?;
            app.settings.prodinfo = Some(prodinfo.clone());
            let download = model::firmware_dir().join("download");
            steps.push((
                emulator.clone(),
                vec![
                    "download-firmware".into(),
                    text(&download),
                    "--keys".into(),
                    text(&keys),
                    "--prodinfo".into(),
                    text(&prodinfo),
                ],
            ));
            Some(download)
        }
        Firmware::Downloaded(dir) => Some(dir.clone()),
        Firmware::Folder(Some(dir)) => Some(dir.clone()),
        Firmware::Folder(None) => return Err("Choose the firmware folder.".into()),
        Firmware::None => None,
    };
    let mut format = vec![
        "format-emmc".to_string(),
        text(&form.image),
        "--keys".into(),
        text(&keys),
    ];
    if let Some(firmware) = &firmware {
        format.extend(["--firmware".into(), text(firmware)]);
        if form.boot_images {
            format.extend([
                "--boot0".into(),
                text(&boot[0]),
                "--boot1".into(),
                text(&boot[1]),
            ]);
        }
    } else if form.boot_images {
        return Err("Boot partition images need firmware.".into());
    }
    steps.push((emulator, format));
    let wake_ctx = ctx.clone();
    Ok(Job::run(
        steps,
        move |index, printed, rest| {
            if !(downloading && index == 0) {
                return Ok(());
            }
            // "console …: system update 23.5.0.640 (0x5c500280)"
            let version = printed
                .iter()
                .find_map(|line| line.split("system update ").nth(1))
                .and_then(|rest| rest.split_whitespace().next())
                .ok_or("the download did not say which version it fetched")?
                .to_string();
            let download = model::firmware_dir().join("download");
            let kept = model::firmware_dir().join(&version);
            if kept.exists() {
                let _ = std::fs::remove_dir_all(&download);
            } else {
                std::fs::rename(&download, &kept)
                    .map_err(|error| format!("{}: {error}", kept.display()))?;
            }
            for argument in &mut rest[0].1 {
                if *argument == download.display().to_string() {
                    *argument = kept.display().to_string();
                }
            }
            Ok(())
        },
        move || wake_ctx.request_repaint(),
    ))
}

/// The microSD dialog; `false` once it is closed.
fn format_sd(ctx: &egui::Context, form: &mut FormatSd, machine: &mut Machine) -> bool {
    let mut open = true;
    let running = form
        .job
        .as_ref()
        .is_some_and(|job| job.finished().is_none());
    if let Some(job) = &form.job
        && let Some(result) = job.finished()
    {
        match result {
            Ok(()) => {
                machine.sd = Some(form.image.clone());
                machine.sd_writable = true;
                return false;
            }
            Err(error) => {
                form.error = Some(error);
                form.job = None;
            }
        }
    }
    let response = egui::Modal::new(egui::Id::unique("format-sd")).frame(modal_frame()).show(ctx, |ui| {
        ui.set_width(600.0);
        ui.label(RichText::new("Format SD").size(22.0).strong());
        ui.label(RichText::new("A new microSD card image (a sparse file), formatted as the console formats a card.").color(MUTED));
        ui.add_space(10.0);
        ui.add_enabled_ui(!running, |ui| {
            group(ui, |ui| {
                let mut rows = Rows::new();
                rows.row(ui, "Image", |ui| {
                    if theme::small_button(ui, "Change…").clicked()
                        && let Some(path) = tabs::save_file(Some(&form.image), &form.image.file_name().unwrap_or_default().to_string_lossy())
                    {
                        form.image = path;
                    }
                    ui.with_layout(Layout::left_to_right(Align::Center), |ui| {
                        ui.add_space(24.0);
                        theme::path_label(ui, Some(&form.image), "");
                    });
                });
                rows.row(ui, "Size", |ui| {
                    ui.add(TextEdit::singleline(&mut form.size).desired_width(90.0))
                        .on_hover_text("Bytes, or K, M or G: a multiple of 512K above 2088M, up to 32G.");
                });
                rows.row(ui, "Files", |ui| {
                    if theme::small_button(ui, "Add…").clicked()
                        && let Some(host) = tabs::pick_file(form.adds.last().map(|(host, _)| host.as_path()))
                    {
                        let name = host.file_name().unwrap_or_default().to_string_lossy().into_owned();
                        let card = if name.ends_with(".nro") && name != "hbmenu.nro" {
                            format!("/switch/{name}")
                        } else {
                            format!("/{name}")
                        };
                        form.adds.push((host, card));
                    }
                });
                let mut removed = None;
                for (index, (host, card)) in form.adds.iter_mut().enumerate() {
                    ui.horizontal(|ui| {
                        theme::path_label(ui, Some(host), "");
                        ui.label("as");
                        ui.add(TextEdit::singleline(card).desired_width(220.0).code_editor());
                        if theme::clear_button(ui).clicked() {
                            removed = Some(index);
                        }
                    });
                }
                if let Some(index) = removed {
                    form.adds.remove(index);
                }
            });
        });
        if let Some(job) = &form.job {
            ui.add_space(10.0);
            job_output(ui, job);
        }
        if let Some(error) = &form.error {
            ui.add_space(8.0);
            ui.label(RichText::new(error).color(RED));
        }
        ui.add_space(14.0);
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            if ui.add_enabled_ui(!running, |ui| theme::big_button(ui, if running { "Working…" } else { "Format" }, true)).inner.clicked() {
                if form.image.exists() {
                    form.error = Some(format!("{} exists already; choose a new file.", form.image.display()));
                } else {
                    if let Some(dir) = form.image.parent() {
                        let _ = std::fs::create_dir_all(dir);
                    }
                    let mut arguments = vec![
                        "format-sd".to_string(),
                        form.image.display().to_string(),
                        "--size".into(),
                        form.size.trim().to_string(),
                    ];
                    for (host, card) in &form.adds {
                        arguments.extend(["--add".into(), format!("{}={card}", host.display())]);
                    }
                    let wake_ctx = ctx.clone();
                    form.error = None;
                    form.job = Some(Job::run(
                        vec![(crate::child::emulator(), arguments)],
                        |_, _, _| Ok(()),
                        move || wake_ctx.request_repaint(),
                    ));
                }
            }
            if ui.add_enabled_ui(!running, |ui| theme::big_button(ui, "Cancel", false)).inner.clicked() {
                open = false;
            }
        });
    });
    if response.should_close() && !running {
        open = false;
    }
    if running {
        ctx.request_repaint_after(std::time::Duration::from_millis(200));
    }
    let _ = Vec2::ZERO;
    open
}
