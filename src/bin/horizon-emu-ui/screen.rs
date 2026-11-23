//! A machine's screen: the picture the display controller sends, with the
//! machine's state over it and a toolbar for what is done to the console.
//! The mouse on the picture is a finger on the touchscreen; the keyboard
//! is the pair of Joy-Con (J slides them on or off the rails, P is the
//! console's power button), as in the emulator's own window.

use std::time::Duration;

use eframe::egui::{
    self, Align, Align2, Color32, ColorImage, CornerRadius, Key, Layout, Margin, Pos2, Rect,
    RichText, Sense, Stroke, TextureOptions, Ui, Vec2,
};

use crate::{
    app::{App, View, state_of},
    child::Input,
    model::CpuSpeed,
    tabs::{self, line_color},
    theme::{self, AMBER, MUTED, RED, white},
};

/// Which key holds which button (the emulator's `window.rs` mapping).
const KEYS: [(Key, &str); 19] = [
    (Key::A, "a"),
    (Key::B, "b"),
    (Key::X, "x"),
    (Key::Y, "y"),
    (Key::L, "l"),
    (Key::R, "r"),
    (Key::Q, "zl"),
    (Key::E, "zr"),
    (Key::ArrowUp, "up"),
    (Key::ArrowDown, "down"),
    (Key::ArrowLeft, "left"),
    (Key::ArrowRight, "right"),
    (Key::Equals, "plus"),
    (Key::Minus, "minus"),
    (Key::H, "home"),
    (Key::C, "capture"),
    (Key::Comma, "lstick"),
    (Key::Period, "rstick"),
    (Key::P, "power"),
];

/// Lines of output the overlay shows.
const OVERLAY_LINES: usize = 6;

/// Lets go of everything held, as the view is left.
pub fn release(app: &mut App, index: usize) {
    let held: Vec<&'static str> = app.held.drain().collect();
    let child = app
        .entries
        .get(index)
        .and_then(|entry| entry.child.as_ref());
    if let Some(child) = child {
        for button in held {
            child.input(Input::Press {
                button,
                held: false,
            });
        }
        if app.touching.take().is_some() {
            child.input(Input::Touch(None));
        }
    }
    app.touching = None;
}

/// Holds `button` while `down`, telling the machine of changes only.
fn hold(app: &mut App, index: usize, button: &'static str, down: bool) {
    let changed = if down {
        app.held.insert(button)
    } else {
        app.held.remove(button)
    };
    if changed && let Some(child) = &app.entries[index].child {
        child.input(Input::Press { button, held: down });
    }
}

pub fn show(app: &mut App, ui: &mut Ui, index: usize) {
    let ctx = ui.ctx().clone();
    let rect = ctx.content_rect();
    // The screen sits on darkened glass, so that the picture reads.
    ui.painter().rect_filled(rect, 0.0, theme::black(190));
    upload(app, &ctx, index);

    // The picture, as large as fits below the window's buttons.
    let area = Rect::from_min_max(
        rect.min + Vec2::new(12.0, 60.0),
        rect.max - Vec2::new(12.0, 12.0),
    );
    let picture = theme::fit(area, 16.0 / 9.0);
    ui.painter()
        .rect_filled(picture.expand(1.0), 10.0, theme::white(40));
    let lit = app.entries[index].child.as_ref().map(|child| {
        let frame = child.frame.lock().unwrap();
        (
            frame.panel_on && frame.backlight_on && frame.reason.is_none() && frame.width > 0,
            frame.reason.clone(),
        )
    });
    match (&app.texture, lit) {
        (Some(texture), Some((true, _))) => {
            ui.painter().image(
                texture.id(),
                picture,
                Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                Color32::WHITE,
            );
        }
        (_, lit) => {
            ui.painter().rect_filled(picture, 0.0, Color32::BLACK);
            let why = match lit {
                None => "The machine has not run in this session.".to_string(),
                Some((_, Some(reason))) => format!("No picture: {reason}"),
                Some(_) => String::new(),
            };
            if !why.is_empty() {
                ui.painter().text(
                    picture.center(),
                    Align2::CENTER_CENTER,
                    why,
                    egui::FontId::proportional(15.0),
                    MUTED,
                );
            }
        }
    }

    // The mouse on the picture is a finger.
    let response = ui.interact(
        picture,
        ui.scope_id().with("picture"),
        Sense::click_and_drag(),
    );
    let finger = response
        .is_pointer_button_down_on()
        .then(|| response.interact_pointer_pos())
        .flatten()
        .filter(|pos| picture.contains(*pos))
        .map(|pos| {
            let x = (pos.x - picture.left()) / picture.width() * 1280.0;
            let y = (pos.y - picture.top()) / picture.height() * 720.0;
            (x.clamp(0.0, 1279.0) as u16, y.clamp(0.0, 719.0) as u16)
        });
    if finger != app.touching {
        app.touching = finger;
        if let Some(child) = &app.entries[index].child {
            child.input(Input::Touch(finger));
        }
    }

    keyboard(app, &ctx, index);
    top_left(app, &ctx, index);
    top_right(app, &ctx, index);
    toolbar(app, &ctx, index);

    if app.entries[index].running() {
        ctx.request_repaint_after(Duration::from_millis(16));
    }
}

/// Puts the newest frame into the texture, once.
fn upload(app: &mut App, ctx: &egui::Context, index: usize) {
    let Some(child) = &app.entries[index].child else {
        return;
    };
    let frame = child.frame.lock().unwrap();
    if frame.serial == app.texture_serial
        || frame.width == 0
        || frame.rgba.len() != frame.width * frame.height * 4
    {
        return;
    }
    let image = ColorImage::from_rgba_unmultiplied([frame.width, frame.height], &frame.rgba);
    app.texture_serial = frame.serial;
    match &mut app.texture {
        Some(texture) => texture.set(image, TextureOptions::LINEAR),
        None => app.texture = Some(ctx.load_texture("screen", image, TextureOptions::LINEAR)),
    }
}

fn keyboard(app: &mut App, ctx: &egui::Context, index: usize) {
    if ctx.memory(|memory| memory.focused().is_some()) || app.dialog.is_some() {
        return;
    }
    let (toggle_rails, back, down): (bool, bool, Vec<bool>) = ctx.input(|input| {
        (
            input.key_pressed(Key::J),
            input.key_pressed(Key::Escape),
            KEYS.iter().map(|(key, _)| input.key_down(*key)).collect(),
        )
    });
    for ((_, button), down) in KEYS.iter().zip(down) {
        hold(app, index, button, down);
    }
    if toggle_rails {
        toggle_joycons(app, index);
    }
    if back {
        release(app, index);
        app.view = View::Machine;
    }
}

fn toggle_joycons(app: &App, index: usize) {
    if let Some(child) = &app.entries[index].child {
        let seated = child.status.lock().unwrap().joycons_seated;
        child.input(Input::Joycons(!seated));
    }
}

fn top_left(app: &mut App, ctx: &egui::Context, index: usize) {
    egui::Area::new(egui::Id::unique("screen-top-left"))
        .anchor(Align2::LEFT_TOP, Vec2::new(84.0, 12.0))
        .order(egui::Order::Foreground)
        .show(ctx, |ui| {
            theme::overlay(ui, |ui| {
                ui.horizontal(|ui| {
                    if ui
                        .add(egui::Button::new(RichText::new("‹").size(20.0)).frame(false))
                        .on_hover_text("Back (Esc)")
                        .clicked()
                    {
                        release(app, index);
                        app.view = View::Machine;
                    }
                    let entry = &app.entries[index];
                    let (state, color) = state_of(entry);
                    theme::status_dot(ui, color);
                    ui.label(RichText::new(&entry.machine().name).size(16.0).strong());
                    let mut detail = state.to_string();
                    if let Some(child) = &entry.child
                        && entry.running()
                    {
                        let status = child.status.lock().unwrap();
                        let on = |on: bool| if on { "on" } else { "off" };
                        detail += &format!(
                            " · panel {} · backlight {}",
                            on(status.panel_on),
                            on(status.backlight_on)
                        );
                    }
                    ui.label(RichText::new(detail).color(MUTED));
                });
            });
        });
}

/// What a slowed machine's processors are doing, for the overlay.
fn paced_label(cpu: CpuSpeed) -> String {
    match cpu {
        CpuSpeed::Default => "CPU ×10 until boot2 ends".into(),
        CpuSpeed::Slower { factor } => format!("CPU ×{factor} slower"),
        CpuSpeed::Full => "CPU at full speed".into(),
    }
}

fn top_right(app: &mut App, ctx: &egui::Context, index: usize) {
    let entry = &app.entries[index];
    let Some(child) = &entry.child else { return };
    let status = child.status.lock().unwrap().clone();
    let renderer = entry.machine().renderer.label();
    let overlay_lines: Vec<(crate::child::Stream, String)> = if app.log_overlay {
        let log = child.log.lock().unwrap();
        let mut lines: Vec<_> = log
            .lines
            .iter()
            .rev()
            .filter(|line| line.stream != crate::child::Stream::Ui)
            .take(OVERLAY_LINES)
            .map(|line| (line.stream, theme::shorten(&line.text, 72)))
            .collect();
        lines.reverse();
        lines
    } else {
        Vec::new()
    };
    egui::Area::new(egui::Id::unique("screen-top-right"))
        .anchor(Align2::RIGHT_TOP, Vec2::new(-18.0, 18.0))
        .order(egui::Order::Foreground)
        .show(ctx, |ui| {
            ui.with_layout(Layout::top_down(Align::Max), |ui| {
                theme::overlay(ui, |ui| {
                    // Laid out from the right edge: the last item first.
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 14.0;
                        ui.label(RichText::new(renderer).monospace());
                        if entry.running() {
                            let (text, color) = if status.cpu_paced {
                                (paced_label(entry.machine().cpu), AMBER)
                            } else {
                                ("CPU at full speed".to_string(), MUTED)
                            };
                            ui.label(RichText::new(text).monospace().color(color));
                        }
                        let guest = status
                            .guest_seconds
                            .map_or_else(|| "–".into(), |seconds| format!("{seconds:.1}"));
                        ui.label(RichText::new(format!("guest {guest} s")).monospace());
                    });
                });
                if !overlay_lines.is_empty() {
                    ui.add_space(8.0);
                    theme::overlay(ui, |ui| {
                        ui.set_max_width(520.0);
                        ui.vertical(|ui| {
                            ui.spacing_mut().item_spacing.y = 3.0;
                            for (stream, line) in &overlay_lines {
                                ui.label(
                                    RichText::new(line)
                                        .monospace()
                                        .size(12.5)
                                        .color(line_color(*stream, line)),
                                );
                            }
                        });
                    });
                }
            });
        });
}

/// A round toolbar button with a tooltip.
fn tool(ui: &mut Ui, icon: &str, tip: &str, lit: bool, enabled: bool) -> egui::Response {
    let button = egui::Button::new(RichText::new(icon).size(18.0))
        .fill(if lit { white(46) } else { Color32::TRANSPARENT })
        .stroke(Stroke::NONE)
        .corner_radius(CornerRadius::same(14))
        .min_size(Vec2::new(44.0, 40.0));
    ui.add_enabled(enabled, button).on_hover_text(tip)
}

fn toolbar(app: &mut App, ctx: &egui::Context, index: usize) {
    let mut actions = Actions::default();
    let entry = &app.entries[index];
    let running = entry.running();
    let status = entry
        .child
        .as_ref()
        .map(|child| child.status.lock().unwrap().clone())
        .unwrap_or_default();
    let has_pro = entry.machine().pro_controller.is_some();
    let amiibo = entry.amiibo_held();
    let log_overlay = app.log_overlay;
    egui::Area::new(egui::Id::unique("screen-toolbar"))
        .anchor(Align2::CENTER_BOTTOM, Vec2::new(0.0, -20.0))
        .order(egui::Order::Foreground)
        .show(ctx, |ui| {
            ui.with_layout(Layout::top_down(Align::Center), |ui| {
                let bar = egui::Frame::new()
                    .fill(Color32::from_rgba_unmultiplied(0x1C, 0x1C, 0x1E, 0xE8))
                    .stroke(Stroke::new(1.0, white(64)))
                    .corner_radius(CornerRadius::same(28))
                    .inner_margin(Margin::same(6))
                    .shadow(egui::Shadow {
                        offset: [0, 10],
                        blur: 30,
                        spread: 0,
                        color: theme::black(80),
                    });
                bar.show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 6.0;
                            let rails = if status.joycons_seated { "Joy-Con  on rails" } else { "Joy-Con  detached" };
                            let button = egui::Button::new(RichText::new(format!("🎮  {rails}")).size(15.0))
                                .fill(if status.joycons_seated { white(46) } else { white(16) })
                                .stroke(Stroke::NONE)
                                .corner_radius(CornerRadius::same(14))
                                .min_size(Vec2::new(0.0, 40.0));
                            actions.rails = ui.add_enabled(running, button).on_hover_text("Slide the Joy-Con on or off the rails (J)").clicked();
                            let power = tool(ui, "🔌", "Hold the console's power button (P)", app.held.contains("power"), running);
                            actions.power = Some(power.is_pointer_button_down_on());
                            let pro_tip = if has_pro {
                                "Press SYNC on the Pro Controller (pair it in System Settings > Controllers)"
                            } else {
                                "Add a Pro Controller in Settings > Controllers"
                            };
                            actions.pro_sync = tool(ui, "🕹", pro_tip, false, running && has_pro).clicked();
                            let nfc_tip = if amiibo { "Take the amiibo away" } else { "Hold an amiibo (NTAG215 dump) over the right Joy-Con" };
                            actions.amiibo = tool(ui, "📶", nfc_tip, amiibo, running).clicked();
                            ui.add(egui::Separator::default().vertical().spacing(10.0));
                            actions.screenshot = tool(ui, "📷", "Save a screenshot to the machine's folder", false, entry.child.is_some()).clicked();
                            actions.save = tool(ui, "💾", "Save state (a snapshot)", false, running).clicked();
                            actions.log = tool(ui, "☰", "Show the emulator's output over the picture", log_overlay, true).clicked();
                            actions.stop = ui
                                .add_enabled(
                                    running,
                                    egui::Button::new(RichText::new("■").size(16.0).color(RED))
                                        .fill(Color32::TRANSPARENT)
                                        .stroke(Stroke::NONE)
                                        .min_size(Vec2::new(44.0, 40.0)),
                                )
                                .on_hover_text("Stop the machine")
                                .clicked();
                            if !running {
                                actions.start = ui
                                    .add(
                                        egui::Button::new(RichText::new("▶  Start").size(15.0))
                                            .fill(white(30))
                                            .corner_radius(CornerRadius::same(14))
                                            .min_size(Vec2::new(0.0, 40.0)),
                                    )
                                    .clicked();
                            }
                        });
                    });
                ui.add_space(8.0);
                let hint = egui::Frame::new()
                    .fill(Color32::from_rgba_unmultiplied(0x1C, 0x1C, 0x1E, 0xD0))
                    .stroke(Stroke::new(1.0, white(48)))
                    .corner_radius(CornerRadius::same(16))
                    .inner_margin(Margin::symmetric(12, 4));
                hint.show(ui, |ui| {
                        ui.horizontal(|ui| {
                            for (key, what) in [("J", "rails"), ("P", "power"), ("H", "home"), ("Esc", "back")] {
                                ui.label(RichText::new(key).monospace().background_color(white(30)));
                                ui.label(RichText::new(what).size(13.0));
                            }
                            ui.label(RichText::new("mouse on picture = touch").size(13.0).color(MUTED))
                                .on_hover_text(
                                    "A B X Y, L R, Q E (ZL ZR), arrows (D-pad), = and - (+ and -), \
                                     H (HOME), C (Capture), comma and period (stick clicks).",
                                );
                        });
                    });
            });
        });
    apply(app, ctx, index, actions);
}

#[derive(Default)]
struct Actions {
    rails: bool,
    power: Option<bool>,
    pro_sync: bool,
    amiibo: bool,
    screenshot: bool,
    save: bool,
    log: bool,
    stop: bool,
    start: bool,
}

fn apply(app: &mut App, ctx: &egui::Context, index: usize, actions: Actions) {
    if actions.rails {
        toggle_joycons(app, index);
    }
    // The power button is held for as long as the mouse is, unless the
    // keyboard holds it.
    if let Some(down) = actions.power
        && !ctx.input(|input| input.key_down(Key::P))
    {
        hold(app, index, "power", down);
    }
    if actions.pro_sync
        && let Some(child) = &app.entries[index].child
    {
        child.input(Input::Press {
            button: "pro:sync",
            held: true,
        });
        child.input(Input::Press {
            button: "pro:sync",
            held: false,
        });
    }
    if actions.amiibo {
        let entry = &mut app.entries[index];
        if entry.amiibo_held() {
            if let Some(child) = &entry.child {
                child.input(Input::Amiibo(None));
            }
            entry.amiibo_on = Some(false);
        } else if let Some(path) = tabs::pick_file(entry.machine().amiibo.as_deref()) {
            let writable = entry.machine().amiibo_writable;
            if let Some(child) = &entry.child {
                child.input(Input::Amiibo(Some((path, writable))));
            }
            entry.amiibo_on = Some(true);
        }
    }
    if actions.screenshot {
        app.screenshot(index, ctx);
    }
    if actions.save {
        app.save_state(index, ctx);
    }
    if actions.log {
        app.log_overlay = !app.log_overlay;
    }
    if actions.stop {
        release(app, index);
        if let Some(child) = &app.entries[index].child {
            child.stop();
        }
    }
    if actions.start {
        app.start(index, None, ctx);
    }
}
