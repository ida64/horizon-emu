//! The window: the machines at the left, the selected one at the right,
//! and its screen in place of both while it is shown.

use std::{
    collections::HashSet,
    fs::File,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use eframe::egui::{
    self, Align, Align2, Color32, CornerRadius, Frame, Layout, Margin, RichText, Sense, Stroke,
    TextureHandle, Ui, Vec2,
};
use horizon_emu::{keys::KeySet, nand::system_version};
use serde_json::json;

use crate::{
    child::{self, Child},
    dialogs::{self, Dialog},
    model::{Machine, Settings, Snapshot, SnapshotInfo, Start, Stored},
    screen, tabs,
    theme::{self, AMBER, FAINT, GREEN, GREY, MUTED, RED, white},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Settings,
    Debug,
    Threads,
    Snapshots,
    Log,
}

/// What a background request reports back to the UI thread.
pub enum Message {
    Toast { text: String, error: bool },
    SnapshotsChanged(PathBuf),
}

/// The installed firmware of a machine's eMMC image, read in the
/// background for the eMMC image and keys it was read with.
type FirmwareRead = Arc<Mutex<Option<Result<String, String>>>>;

/// A machine with what the UI knows of it.
pub struct Entry {
    pub stored: Stored,
    /// The definition as last written to disk.
    saved: Machine,
    pub child: Option<Child>,
    pub tab: Tab,
    pub snapshots: Vec<Snapshot>,
    firmware: Option<((Option<PathBuf>, Option<PathBuf>), FirmwareRead)>,
    pub log_filter: String,
    pub log_follow: bool,
    /// Whether an amiibo is held over the reader, as the UI last said;
    /// `None` until it says anything to this run.
    pub amiibo_on: Option<bool>,
}

impl Entry {
    fn new(stored: Stored) -> Self {
        let snapshots = stored.snapshots();
        Self {
            saved: stored.machine.clone(),
            stored,
            child: None,
            tab: Tab::Settings,
            snapshots,
            firmware: None,
            log_filter: String::new(),
            log_follow: true,
            amiibo_on: None,
        }
    }

    pub fn machine(&self) -> &Machine {
        &self.stored.machine
    }

    pub fn running(&self) -> bool {
        self.child.as_ref().is_some_and(|child| !child.has_ended())
    }

    /// Whether an amiibo is held over the reader of the running machine.
    pub fn amiibo_held(&self) -> bool {
        self.amiibo_on
            .unwrap_or(self.stored.machine.amiibo.is_some())
    }

    /// Whether machine flags can no longer change: a snapshot records them.
    pub fn locked(&self) -> bool {
        !self.snapshots.is_empty() || matches!(self.machine().start, Start::Snapshot { .. })
    }

    pub fn refresh_snapshots(&mut self) {
        self.snapshots = self.stored.snapshots();
    }

    /// "Firmware 23.5.0", or why it is not known, read in the background.
    fn firmware(&mut self) -> Option<Result<String, String>> {
        let machine = &self.stored.machine;
        let key = (machine.emmc.clone(), machine.keys.clone());
        if self
            .firmware
            .as_ref()
            .is_none_or(|(read_for, _)| *read_for != key)
        {
            let read: FirmwareRead = Arc::default();
            let slot = Arc::clone(&read);
            let (emmc, keys) = key.clone();
            std::thread::spawn(move || {
                let result = (|| {
                    let emmc = emmc.ok_or("no eMMC image")?;
                    let keys = keys.ok_or("no keys")?;
                    let keys = KeySet::load(&keys).map_err(|error| error.to_string())?;
                    let image = File::open(&emmc)
                        .map_err(|error| format!("{}: {error}", emmc.display()))?;
                    let version = system_version::installed_system_version(&image, &keys)
                        .map_err(|error| error.to_string())?;
                    Ok::<_, String>(format!(
                        "{}.{}.{}",
                        version.major, version.minor, version.micro
                    ))
                })();
                *slot.lock().unwrap() = Some(result);
            });
            self.firmware = Some((key, read));
        }
        self.firmware.as_ref()?.1.lock().unwrap().clone()
    }
}

pub enum View {
    Machine,
    Screen,
}

/// Width of the sidebar, in points.
const SIDEBAR_WIDTH: f32 = 236.0;

pub struct App {
    pub entries: Vec<Entry>,
    pub selected: Option<usize>,
    pub view: View,
    pub dialog: Option<Dialog>,
    pub settings: Settings,
    saved_settings: Settings,
    inbox: Arc<Mutex<Vec<Message>>>,
    toasts: Vec<(String, bool, Instant)>,
    /// Whether the emulator carries the hypervisor entitlement, once
    /// looked at.
    signed: Option<Result<(), String>>,
    pub emulator: PathBuf,
    // The screen view's state.
    pub texture: Option<TextureHandle>,
    pub texture_serial: u64,
    pub held: HashSet<&'static str>,
    pub touching: Option<(u16, u16)>,
    pub log_overlay: bool,
    /// Whether the window's glass backdrop is in place.
    glass_tried: bool,
    /// The window's native materials, once in place.
    glass: Option<crate::glass::Glass>,
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        theme::apply(&cc.egui_ctx);
        let entries: Vec<Entry> = Stored::load_all().into_iter().map(Entry::new).collect();
        let settings = Settings::load();
        Self {
            selected: (!entries.is_empty()).then_some(0),
            entries,
            view: View::Machine,
            dialog: None,
            saved_settings: settings.clone(),
            settings,
            inbox: Arc::default(),
            toasts: Vec::new(),
            signed: None,
            emulator: child::emulator(),
            texture: None,
            texture_serial: 0,
            held: HashSet::new(),
            touching: None,
            log_overlay: true,
            glass_tried: false,
            glass: None,
        }
    }

    pub fn toast(&mut self, text: impl Into<String>, error: bool) {
        self.toasts.push((text.into(), error, Instant::now()));
    }

    pub fn inbox(&self) -> Arc<Mutex<Vec<Message>>> {
        Arc::clone(&self.inbox)
    }

    /// Adds a machine and selects it.
    pub fn add_machine(&mut self, machine: Machine) {
        match Stored::create(machine) {
            Ok(stored) => {
                self.entries.push(Entry::new(stored));
                self.selected = Some(self.entries.len() - 1);
                self.view = View::Machine;
            }
            Err(error) => self.toast(format!("cannot create the machine: {error}"), true),
        }
    }

    /// Starts machine `index`: cold, or from `snapshot`.
    pub fn start(&mut self, index: usize, snapshot: Option<PathBuf>, ctx: &egui::Context) {
        if self.signed.is_none() {
            self.signed = Some(child::ensure_entitlement(&self.emulator));
        }
        if let Some(Err(error)) = &self.signed {
            let error = format!("the emulator cannot use the hypervisor: {error}");
            self.toast(error, true);
            return;
        }
        let machine = self.entries[index].machine();
        let cold = snapshot.is_none() && !matches!(machine.start, Start::Snapshot { .. });
        if cold
            && let Some(keys) = &machine.keys
            && let Some(problem) = crate::model::keys_problem(keys, machine.start == Start::PowerOn)
        {
            self.toast(problem, true);
            return;
        }
        let entry = &mut self.entries[index];
        let socket = std::env::temp_dir().join(format!(
            "hemu-{}-{}.sock",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
        ));
        let started = entry
            .machine()
            .arguments(&entry.stored.dir, &socket, snapshot.as_deref())
            .and_then(|arguments| {
                let wake_ctx = ctx.clone();
                Child::spawn(&self.emulator, &arguments, socket, move || {
                    wake_ctx.request_repaint();
                })
            });
        match started {
            Ok(child) => {
                entry.child = Some(child);
                entry.amiibo_on = None;
                self.texture_serial = 0;
            }
            Err(error) => self.toast(error, true),
        }
    }

    /// Takes a snapshot of running machine `index` into its snapshots.
    pub fn save_state(&mut self, index: usize, ctx: &egui::Context) {
        let entry = &self.entries[index];
        let Some(child) = entry.child.as_ref().filter(|_| entry.running()) else {
            return;
        };
        let dir = entry.stored.snapshots_dir().join(timestamp());
        let (inbox, ctx, machine_dir) = (self.inbox(), ctx.clone(), entry.stored.dir.clone());
        self.inbox.lock().unwrap().push(Message::Toast {
            text: "Saving state…".into(),
            error: false,
        });
        child.request(json!({"op": "save_state", "path": dir}), move |answer| {
            let message = match answer {
                Ok(answer) => {
                    let info = SnapshotInfo {
                        guest_seconds: answer["guest_seconds"].as_f64(),
                        paused_seconds: answer["paused_seconds"].as_f64(),
                        ram_bytes: answer["ram_bytes"].as_u64(),
                    };
                    if let Ok(json) = serde_json::to_vec_pretty(&info) {
                        let _ = std::fs::write(dir.with_extension("json"), json);
                    }
                    inbox
                        .lock()
                        .unwrap()
                        .push(Message::SnapshotsChanged(machine_dir));
                    Message::Toast {
                        text: format!(
                            "State saved at {:.1} s of guest time",
                            info.guest_seconds.unwrap_or_default()
                        ),
                        error: false,
                    }
                }
                Err(error) => Message::Toast {
                    text: format!("save state: {error}"),
                    error: true,
                },
            };
            inbox.lock().unwrap().push(message);
            ctx.request_repaint();
        });
    }

    /// Saves what the screen of machine `index` shows as a PNG in its
    /// directory.
    pub fn screenshot(&mut self, index: usize, ctx: &egui::Context) {
        let entry = &self.entries[index];
        let Some(child) = entry.child.as_ref() else {
            return;
        };
        let dir = entry.stored.dir.join("screenshots");
        if let Err(error) = std::fs::create_dir_all(&dir) {
            self.toast(format!("{}: {error}", dir.display()), true);
            return;
        }
        let path = dir.join(format!("{}.png", timestamp()));
        let (inbox, ctx) = (self.inbox(), ctx.clone());
        child.request(json!({"op": "screenshot", "path": path}), move |answer| {
            let message = match answer {
                Ok(_) => Message::Toast {
                    text: format!("Screenshot saved to {}", path.display()),
                    error: false,
                },
                Err(error) => Message::Toast {
                    text: format!("screenshot: {error}"),
                    error: true,
                },
            };
            inbox.lock().unwrap().push(message);
            ctx.request_repaint();
        });
    }

    fn take_messages(&mut self) {
        let messages = std::mem::take(&mut *self.inbox.lock().unwrap());
        for message in messages {
            match message {
                Message::Toast { text, error } => self.toast(text, error),
                Message::SnapshotsChanged(dir) => {
                    for entry in self
                        .entries
                        .iter_mut()
                        .filter(|entry| entry.stored.dir == dir)
                    {
                        entry.refresh_snapshots();
                    }
                }
            }
        }
    }

    /// Writes definitions and settings that changed.
    fn persist(&mut self) {
        let mut failures = Vec::new();
        for entry in &mut self.entries {
            if entry.stored.machine != entry.saved {
                match entry.stored.save() {
                    Ok(()) => entry.saved = entry.stored.machine.clone(),
                    Err(error) => failures.push(format!("{}: {error}", entry.stored.dir.display())),
                }
            }
        }
        for failure in failures {
            self.toast(failure, true);
        }
        if self.settings != self.saved_settings {
            self.settings.save();
            self.saved_settings = self.settings.clone();
        }
    }

    /// The sidebar's contents; the glass under it is native (see
    /// [`crate::glass`]), or painted where the system has none.
    fn sidebar(&mut self, ui: &mut Ui) {
        let rect = ui.max_rect();
        let native = self.glass.as_mut().is_some_and(|glass| {
            glass.place_sidebar(Some(rect));
            glass.has_sidebar()
        });
        if !native {
            ui.painter()
                .rect_filled(rect, CornerRadius::same(24), white(18));
        }
        let inner = rect.shrink2(Vec2::new(10.0, 12.0));
        ui.scope_builder(egui::UiBuilder::new().max_rect(inner), |ui| {
            // Clear of the window's buttons, which sit over the glass.
            ui.add_space(30.0);
            ui.horizontal(|ui| {
                ui.add_space(6.0);
                ui.label(
                    RichText::new("Machines")
                        .font(theme::semibold(11.5))
                        .color(MUTED),
                );
            });
            ui.add_space(2.0);
            let footer = 46.0;
            egui::ScrollArea::vertical()
                .max_height(ui.available_height() - footer)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing.y = 2.0;
                    for index in 0..self.entries.len() {
                        self.machine_item(ui, index);
                    }
                    if self.entries.is_empty() {
                        ui.horizontal(|ui| {
                            ui.add_space(6.0);
                            ui.label(RichText::new("No machines yet").color(FAINT));
                        });
                    }
                });
            ui.with_layout(Layout::bottom_up(Align::Min), |ui| {
                let button =
                    egui::Button::new(RichText::new("+   New Machine").size(13.0).color(MUTED))
                        .fill(Color32::TRANSPARENT)
                        .min_size(Vec2::new(ui.available_width(), 30.0));
                if ui.add(button).clicked() {
                    self.dialog = Some(Dialog::new_machine(&self.settings, self.entries.len()));
                }
            });
        });
    }

    fn machine_item(&mut self, ui: &mut Ui, index: usize) {
        let entry = &self.entries[index];
        let selected = self.selected == Some(index);
        let (state, color) = state_of(entry);
        let response = Frame::new()
            .fill(if selected {
                white(30)
            } else {
                Color32::TRANSPARENT
            })
            .corner_radius(CornerRadius::same(10))
            .inner_margin(Margin::symmetric(8, 6))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 8.0;
                    theme::status_dot(ui, color);
                    ui.vertical(|ui| {
                        ui.spacing_mut().item_spacing.y = 0.0;
                        ui.label(RichText::new(&entry.machine().name).font(theme::semibold(13.0)));
                        ui.label(
                            RichText::new(format!("{} · {state}", entry.machine().kind()))
                                .size(11.5)
                                .color(MUTED),
                        );
                    });
                });
            })
            .response
            .interact(Sense::click());
        if response.clicked() {
            self.selected = Some(index);
            self.view = View::Machine;
        }
    }

    fn machine_panel(&mut self, ui: &mut Ui, index: usize) {
        let ctx = ui.ctx().clone();
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.add_space(8.0);
                self.header(ui, index, &ctx);
                ui.add_space(18.0);
                let entry = &mut self.entries[index];
                let mut tab = entry.tab;
                ui.vertical_centered(|ui| {
                    theme::segmented(
                        ui,
                        &mut tab,
                        &[
                            (Tab::Settings, "Settings"),
                            (Tab::Debug, "Debug"),
                            (Tab::Threads, "Threads"),
                            (Tab::Snapshots, "Snapshots"),
                            (Tab::Log, "Log"),
                        ],
                    );
                });
                if tab != entry.tab && tab == Tab::Snapshots {
                    entry.refresh_snapshots();
                }
                entry.tab = tab;
                ui.add_space(16.0);
                match tab {
                    Tab::Settings => tabs::settings(ui, entry),
                    Tab::Debug => tabs::debug(ui, entry),
                    Tab::Threads => tabs::threads(ui, entry),
                    Tab::Snapshots => {
                        if let Some(action) = tabs::snapshots(ui, entry) {
                            self.snapshot_action(index, action, &ctx);
                        }
                    }
                    Tab::Log => tabs::log(ui, entry),
                }
                ui.add_space(20.0);
            });
    }

    fn snapshot_action(&mut self, index: usize, action: tabs::SnapshotAction, ctx: &egui::Context) {
        match action {
            tabs::SnapshotAction::Restore(dir) => {
                if self.entries[index].running() {
                    self.toast("Stop the machine before restoring a snapshot", true);
                } else {
                    self.start(index, Some(dir), ctx);
                }
            }
            tabs::SnapshotAction::Delete(dir) => {
                let removed = std::fs::remove_dir_all(&dir);
                let _ = std::fs::remove_file(dir.with_extension("json"));
                if let Err(error) = removed {
                    self.toast(format!("{}: {error}", dir.display()), true);
                }
                self.entries[index].refresh_snapshots();
            }
            tabs::SnapshotAction::NewMachine(dir) => {
                let source = self.entries[index].machine().clone();
                let name = format!(
                    "{} ({})",
                    source.name,
                    dir.file_name().unwrap_or_default().to_string_lossy()
                );
                self.add_machine(Machine {
                    name,
                    start: Start::Snapshot { dir: Some(dir) },
                    ..source
                });
            }
        }
    }

    fn header(&mut self, ui: &mut Ui, index: usize, ctx: &egui::Context) {
        let firmware = self.entries[index].firmware();
        let entry = &self.entries[index];
        let running = entry.running();
        let stopping = entry.child.as_ref().is_some_and(Child::stopping);
        let mut show = false;
        let (mut start, mut stop, mut save) = (false, false, false);
        ui.horizontal(|ui| {
            ui.label(theme::title(&entry.machine().name));
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.spacing_mut().item_spacing.x = 8.0;
                ui.add_enabled_ui(!running, |ui| {
                    ui.menu_button(RichText::new("•••").size(13.0), |ui| {
                        if ui.button("Reveal in Finder").clicked() {
                            let _ = std::process::Command::new("open")
                                .arg(&entry.stored.dir)
                                .spawn();
                            ui.close();
                        }
                        if ui.button("Duplicate").clicked() {
                            let mut copy = entry.machine().clone();
                            copy.name = format!("{} copy", copy.name);
                            ui.close();
                            self.dialog = Some(Dialog::Duplicate(copy));
                        }
                        if ui.button(RichText::new("Delete…").color(RED)).clicked() {
                            ui.close();
                            self.dialog = Some(Dialog::Delete(index));
                        }
                    });
                });
                if running {
                    let label = if stopping { "Stopping…" } else { "Stop" };
                    stop = ui
                        .add_enabled_ui(!stopping, |ui| theme::big_button(ui, label, false))
                        .inner
                        .clicked();
                    save = ui
                        .add_enabled_ui(!stopping, |ui| theme::big_button(ui, "Save State", false))
                        .inner
                        .clicked();
                } else {
                    start = theme::big_button(ui, "▶  Start", false).clicked();
                }
                show = theme::big_button(ui, "Show Screen", true).clicked();
            });
        });
        let entry = &self.entries[index];
        let machine = entry.machine();
        ui.add_space(4.0);
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing.x = 6.0;
            let (state, color) = state_of(entry);
            theme::chip(ui, state, Some(color));
            theme::chip(ui, machine.boot_label(), None);
            if let Some(Ok(version)) = firmware {
                theme::chip(ui, format!("Firmware {version}"), None);
            }
            if let Start::Snapshot { dir: Some(dir) } = &machine.start {
                let saved = crate::model::SavedBoot::load(dir).unwrap_or_default();
                let dram = saved.value("--dram").unwrap_or("4");
                let seed = saved.value("--seed").unwrap_or("0");
                theme::chip(ui, format!("{dram} GiB · seed {seed}"), None);
            } else {
                theme::chip(
                    ui,
                    format!("{} GiB · seed {}", machine.dram, machine.seed),
                    None,
                );
            }
            theme::chip(ui, machine.renderer.label(), None);
        });
        if show {
            if !running && entry.child.is_none() {
                self.start(index, None, ctx);
            }
            self.view = View::Screen;
        }
        if start {
            self.start(index, None, ctx);
        }
        if stop && let Some(child) = &self.entries[index].child {
            child.stop();
        }
        if save {
            self.save_state(index, ctx);
        }
    }

    fn toasts(&mut self, ctx: &egui::Context) {
        const SHOWN: Duration = Duration::from_secs(5);
        self.toasts.retain(|(_, _, at)| at.elapsed() < SHOWN);
        if self.toasts.is_empty() {
            return;
        }
        egui::Area::new(egui::Id::unique("toasts"))
            .anchor(Align2::RIGHT_BOTTOM, Vec2::new(-24.0, -24.0))
            .order(egui::Order::Tooltip)
            .show(ctx, |ui| {
                for (text, error, _) in &self.toasts {
                    let frame = Frame::new()
                        .fill(Color32::from_rgba_unmultiplied(0x1C, 0x22, 0x34, 0xD0))
                        .stroke(Stroke::new(1.0, if *error { RED } else { white(70) }))
                        .corner_radius(CornerRadius::same(18))
                        .inner_margin(Margin::symmetric(16, 10));
                    frame.show(ui, |ui| {
                        ui.set_max_width(460.0);
                        ui.label(RichText::new(text).color(if *error { RED } else { theme::TEXT }));
                    });
                }
            });
        ctx.request_repaint_after(Duration::from_millis(500));
    }
}

/// "Running", "Stopped" and the like, with the dot's colour.
pub fn state_of(entry: &Entry) -> (&'static str, Color32) {
    let Some(child) = &entry.child else {
        return ("Stopped", GREY);
    };
    if child.has_ended() {
        let failed = child
            .exited
            .lock()
            .unwrap()
            .is_some_and(|status| !status.success());
        return if failed {
            ("Failed", RED)
        } else {
            ("Stopped", GREY)
        };
    }
    if child.stopping() {
        return ("Stopping", AMBER);
    }
    match child.status.lock().unwrap().state.as_str() {
        "running" => ("Running", GREEN),
        "restarting" => ("Restarting", AMBER),
        "stopped" => ("Stopped", GREY),
        _ => ("Starting", AMBER),
    }
}

/// Now, as a name for a snapshot or a screenshot: `2026-11-23 19-41-12`,
/// in UTC.
pub fn timestamp() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let (days, rest) = (seconds / 86_400, seconds % 86_400);
    // Civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02} {:02}-{:02}-{:02}",
        rest / 3600,
        rest / 60 % 60,
        rest % 60
    )
}

impl eframe::App for App {
    /// Clear to nothing: the window's glass shows through.
    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        [0.0; 4]
    }

    fn ui(&mut self, ui: &mut Ui, frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        if !self.glass_tried {
            self.glass_tried = true;
            match crate::glass::Glass::install(frame) {
                Ok(glass) => self.glass = Some(glass),
                Err(error) => self.toast(format!("window glass: {error}"), true),
            }
        }
        self.take_messages();
        let shown = match self.view {
            View::Screen => self.selected,
            View::Machine => None,
        };
        for (index, entry) in self.entries.iter().enumerate() {
            if let Some(child) = &entry.child {
                child
                    .streaming
                    .store(shown == Some(index), std::sync::atomic::Ordering::SeqCst);
            }
        }
        match (shown, &self.view) {
            (Some(index), View::Screen) => {
                if let Some(glass) = &mut self.glass {
                    glass.place_sidebar(None);
                }
                screen::show(self, ui, index);
            }
            _ => {
                let window = ctx.content_rect();
                // The sidebar floats inside the window's edge, as macOS 26
                // draws one; the content takes the rest.
                let sidebar = egui::Rect::from_min_max(
                    window.min + Vec2::splat(8.0),
                    egui::pos2(window.left() + 8.0 + SIDEBAR_WIDTH, window.bottom() - 8.0),
                );
                let content = egui::Rect::from_min_max(
                    egui::pos2(sidebar.right() + 28.0, window.top() + 14.0),
                    egui::pos2(window.right() - 28.0, window.bottom()),
                );
                ui.scope_builder(egui::UiBuilder::new().max_rect(sidebar), |ui| {
                    self.sidebar(ui)
                });
                ui.scope_builder(egui::UiBuilder::new().max_rect(content), |ui| {
                    match self.selected {
                        Some(index) if index < self.entries.len() => self.machine_panel(ui, index),
                        _ => {
                            ui.vertical_centered(|ui| {
                                ui.add_space(ui.available_height() / 3.0);
                                ui.label(
                                    RichText::new("No Machine Selected")
                                        .font(theme::semibold(20.0))
                                        .color(MUTED),
                                );
                                ui.label(
                                    RichText::new("Create one with New Machine in the sidebar.")
                                        .color(FAINT),
                                );
                            });
                        }
                    }
                });
            }
        }
        if self.dialog.is_some() {
            dialogs::show(self, &ctx);
        }
        self.toasts(&ctx);
        self.persist();
        // Running machines change on their own: their status comes in a
        // few times a second.
        if self.entries.iter().any(Entry::running) {
            ctx.request_repaint_after(Duration::from_millis(250));
        }
    }
}

impl Drop for App {
    /// Machines still running are asked to stop and given a moment, then
    /// ended: their writable images should not be cut off mid-write.
    fn drop(&mut self) {
        let running: Vec<&Child> = self
            .entries
            .iter()
            .filter_map(|entry| entry.child.as_ref())
            .filter(|child| !child.has_ended())
            .collect();
        for child in &running {
            child.stop();
        }
        let deadline = Instant::now() + Duration::from_secs(5);
        while running.iter().any(|child| !child.has_ended()) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}
