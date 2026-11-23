//! The look of macOS 26 in dark mode: San Francisco type, content on the
//! window's material (see [`crate::glass`]), grouped rows on quiet fills
//! as System Settings lays them out, capsule controls, switches for what
//! is on or off, and the system's colours.

use std::{path::Path, sync::Arc};

use eframe::egui::{
    self, Align, Color32, CornerRadius, FontData, FontDefinitions, FontFamily, FontId, FontTweak,
    Frame, Layout, Margin, Rect, Response, RichText, Sense, Shadow, Stroke, TextStyle, Ui, Vec2,
    epaint::text::VariationCoords,
};

/// `labelColor`, `secondaryLabelColor`, `tertiaryLabelColor` (dark).
pub const TEXT: Color32 = Color32::from_rgba_unmultiplied_const(255, 255, 255, 235);
pub const MUTED: Color32 = Color32::from_rgba_unmultiplied_const(235, 235, 245, 150);
pub const FAINT: Color32 = Color32::from_rgba_unmultiplied_const(235, 235, 245, 80);
/// The system's colours (dark).
pub const BLUE: Color32 = Color32::from_rgb(0x0A, 0x84, 0xFF);
pub const GREEN: Color32 = Color32::from_rgb(0x30, 0xD1, 0x58);
pub const AMBER: Color32 = Color32::from_rgb(0xFF, 0x9F, 0x0A);
pub const RED: Color32 = Color32::from_rgb(0xFF, 0x45, 0x3A);
pub const GREY: Color32 = Color32::from_rgb(0x8E, 0x8E, 0x93);

/// The weight of titles and headings.
pub const SEMIBOLD: &str = "semibold";

/// White at `alpha` (0-255), unmultiplied.
pub const fn white(alpha: u8) -> Color32 {
    Color32::from_rgba_unmultiplied_const(255, 255, 255, alpha)
}

pub const fn black(alpha: u8) -> Color32 {
    Color32::from_rgba_unmultiplied_const(0, 0, 0, alpha)
}

/// San Francisco from the system, regular and semibold (one variable font,
/// two weights), and SF Mono; egui's own fonts behind them for what they
/// lack.
fn fonts() -> FontDefinitions {
    let mut fonts = FontDefinitions::default();
    let load = |path: &str| std::fs::read(path).ok();
    let mut proportional = Vec::new();
    let mut semibold = Vec::new();
    let mut monospace = Vec::new();
    if let Some(sf) = load("/System/Library/Fonts/SFNS.ttf") {
        let weight = |wght: f32| FontTweak {
            coords: VariationCoords::new([("wght", wght)]),
            ..FontTweak::default()
        };
        fonts.font_data.insert(
            "sf".into(),
            Arc::new(FontData::from_owned(sf.clone()).tweak(weight(400.0))),
        );
        fonts.font_data.insert(
            "sf-semibold".into(),
            Arc::new(FontData::from_owned(sf).tweak(weight(600.0))),
        );
        proportional.push("sf".to_string());
        semibold.push("sf-semibold".to_string());
    }
    if let Some(mono) = load("/System/Library/Fonts/SFNSMono.ttf") {
        fonts
            .font_data
            .insert("sf-mono".into(), Arc::new(FontData::from_owned(mono)));
        monospace.push("sf-mono".to_string());
    }
    let defaults = fonts
        .families
        .get(&FontFamily::Proportional)
        .cloned()
        .unwrap_or_default();
    let mono_defaults = fonts
        .families
        .get(&FontFamily::Monospace)
        .cloned()
        .unwrap_or_default();
    proportional.extend(defaults.iter().cloned());
    semibold.extend(proportional.iter().cloned());
    monospace.extend(mono_defaults);
    fonts
        .families
        .insert(FontFamily::Proportional, proportional);
    fonts.families.insert(FontFamily::Monospace, monospace);
    fonts
        .families
        .insert(FontFamily::Name(SEMIBOLD.into()), semibold);
    fonts
}

pub fn semibold(size: f32) -> FontId {
    FontId::new(size, FontFamily::Name(SEMIBOLD.into()))
}

pub fn apply(ctx: &egui::Context) {
    ctx.set_fonts(fonts());
    ctx.set_visuals(egui::Visuals::dark());
    ctx.global_style_mut(|style| {
        style.text_styles = [
            (TextStyle::Heading, semibold(26.0)),
            (TextStyle::Body, FontId::new(13.5, FontFamily::Proportional)),
            (
                TextStyle::Button,
                FontId::new(13.0, FontFamily::Proportional),
            ),
            (
                TextStyle::Small,
                FontId::new(11.0, FontFamily::Proportional),
            ),
            (
                TextStyle::Monospace,
                FontId::new(12.0, FontFamily::Monospace),
            ),
        ]
        .into();
        style.spacing.item_spacing = Vec2::new(8.0, 8.0);
        style.spacing.button_padding = Vec2::new(12.0, 4.0);
        style.spacing.interact_size = Vec2::new(24.0, 26.0);
        style.spacing.combo_width = 140.0;
        let visuals = &mut style.visuals;
        visuals.override_text_color = Some(TEXT);
        visuals.panel_fill = Color32::TRANSPARENT;
        visuals.window_fill = Color32::from_rgba_unmultiplied(0x2A, 0x2A, 0x2E, 0xF5);
        visuals.window_stroke = Stroke::new(1.0, white(28));
        visuals.window_corner_radius = CornerRadius::same(20);
        visuals.window_shadow = Shadow {
            offset: [0, 18],
            blur: 48,
            spread: 0,
            color: black(120),
        };
        visuals.popup_shadow = Shadow {
            offset: [0, 8],
            blur: 24,
            spread: 0,
            color: black(110),
        };
        visuals.menu_corner_radius = CornerRadius::same(12);
        visuals.extreme_bg_color = white(12);
        visuals.text_edit_bg_color = Some(white(14));
        visuals.selection.bg_fill = BLUE.gamma_multiply(0.6);
        visuals.selection.stroke = Stroke::new(1.5, BLUE);
        visuals.hyperlink_color = Color32::from_rgb(0x40, 0x9C, 0xFF);
        visuals.faint_bg_color = white(8);
        for (widgets, fill) in [
            (&mut visuals.widgets.inactive, white(26)),
            (&mut visuals.widgets.hovered, white(38)),
            (&mut visuals.widgets.active, white(52)),
            (&mut visuals.widgets.open, white(38)),
        ] {
            widgets.bg_fill = fill;
            widgets.weak_bg_fill = fill;
            widgets.bg_stroke = Stroke::NONE;
            widgets.corner_radius = CornerRadius::same(13);
            widgets.fg_stroke = Stroke::new(1.5, TEXT);
            widgets.expansion = 0.0;
        }
        visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0, white(20));
        visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0, TEXT);
        visuals.widgets.noninteractive.corner_radius = CornerRadius::same(10);
    });
}

/// A title, as large as a window's.
pub fn title(text: &str) -> RichText {
    RichText::new(text).font(semibold(26.0))
}

/// A heading of a section, above its group.
pub fn caption(text: &str) -> RichText {
    RichText::new(text).font(semibold(13.0)).color(MUTED)
}

/// A grouped box of rows, as System Settings draws them.
pub fn group<R>(ui: &mut Ui, add: impl FnOnce(&mut Ui) -> R) -> R {
    Frame::new()
        .fill(white(14))
        .corner_radius(CornerRadius::same(12))
        .inner_margin(Margin::symmetric(14, 2))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            add(ui)
        })
        .inner
}

/// A section: its heading, then its rows in a [`group`].
pub fn section<R>(ui: &mut Ui, title: &str, add: impl FnOnce(&mut Ui) -> R) -> R {
    ui.add_space(2.0);
    ui.horizontal(|ui| {
        ui.add_space(6.0);
        ui.label(caption(title));
    });
    group(ui, add)
}

/// Rows of a group: a label at the left, controls at the right, a hairline
/// between two rows.
pub struct Rows {
    count: usize,
}

impl Rows {
    pub fn new() -> Self {
        Self { count: 0 }
    }

    pub fn row(&mut self, ui: &mut Ui, label: &str, controls: impl FnOnce(&mut Ui)) {
        self.row_rich(ui, RichText::new(label), controls);
    }

    pub fn row_rich(&mut self, ui: &mut Ui, label: RichText, controls: impl FnOnce(&mut Ui)) {
        if self.count > 0 {
            let width = ui.available_width();
            let (rect, _) = ui.allocate_exact_size(Vec2::new(width, 1.0), Sense::hover());
            ui.painter()
                .hline(rect.x_range(), rect.center().y, Stroke::new(0.5, white(26)));
        }
        self.count += 1;
        ui.allocate_ui_with_layout(
            Vec2::new(ui.available_width(), 38.0),
            Layout::left_to_right(Align::Center),
            |ui| {
                ui.set_min_height(38.0);
                ui.label(label);
                ui.with_layout(Layout::right_to_left(Align::Center), controls);
            },
        );
    }
}

/// A switch, on or off.
pub fn switch(ui: &mut Ui, on: &mut bool) -> Response {
    let size = Vec2::new(34.0, 20.0);
    let (rect, mut response) = ui.allocate_exact_size(size, Sense::click());
    if response.clicked() {
        *on = !*on;
        response.mark_changed();
    }
    if ui.is_rect_visible(rect) {
        let how_on = ui.ctx().animate_bool_responsive(response.id, *on);
        let enabled = ui.is_enabled();
        let track = if enabled { white(46) } else { white(20) };
        let lit = if enabled {
            BLUE
        } else {
            BLUE.gamma_multiply(0.4)
        };
        let fill = egui::lerp(egui::Rgba::from(track)..=egui::Rgba::from(lit), how_on);
        let painter = ui.painter();
        painter.rect_filled(rect, CornerRadius::same(10), Color32::from(fill));
        let radius = rect.height() / 2.0 - 2.0;
        let x = egui::lerp(
            rect.left() + radius + 2.0..=rect.right() - radius - 2.0,
            how_on,
        );
        painter.circle_filled(egui::pos2(x, rect.center().y + 0.5), radius, black(50));
        painter.circle_filled(
            egui::pos2(x, rect.center().y),
            radius,
            if enabled { Color32::WHITE } else { white(140) },
        );
    }
    response
}

/// A switch with its label at its left, for a row's controls.
pub fn labelled_switch(ui: &mut Ui, on: &mut bool, label: &str) -> Response {
    let response = switch(ui, on);
    ui.label(RichText::new(label).color(MUTED));
    response
}

/// A capsule button: `prominent` is the blue one.
pub fn button(ui: &mut Ui, text: &str, prominent: bool) -> Response {
    let text = RichText::new(text).size(13.0);
    let button = if prominent {
        egui::Button::new(text.color(Color32::WHITE)).fill(BLUE)
    } else {
        egui::Button::new(text)
    };
    ui.add(
        button
            .corner_radius(CornerRadius::same(14))
            .min_size(Vec2::new(0.0, 28.0)),
    )
}

/// A button of the window's toolbar: a little larger, with an icon.
pub fn big_button(ui: &mut Ui, text: &str, prominent: bool) -> Response {
    let text = RichText::new(text).size(14.0);
    let button = if prominent {
        egui::Button::new(text.color(Color32::WHITE)).fill(BLUE)
    } else {
        egui::Button::new(text)
    };
    ui.add(
        button
            .corner_radius(CornerRadius::same(17))
            .min_size(Vec2::new(0.0, 34.0)),
    )
}

/// A row's small action button.
pub fn small_button(ui: &mut Ui, text: &str) -> Response {
    button(ui, text, false)
}

/// A borderless round button with a glyph, to clear a value.
pub fn clear_button(ui: &mut Ui) -> Response {
    ui.add(
        egui::Button::new(RichText::new("✖").size(10.0).color(MUTED))
            .fill(white(18))
            .corner_radius(CornerRadius::same(9))
            .min_size(Vec2::splat(18.0)),
    )
    .on_hover_text("Remove")
}

/// A capsule with an optional coloured dot.
pub fn chip(ui: &mut Ui, text: impl Into<String>, dot: Option<Color32>) {
    Frame::new()
        .fill(white(16))
        .corner_radius(CornerRadius::same(11))
        .inner_margin(Margin::symmetric(9, 3))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 5.0;
                if let Some(color) = dot {
                    status_dot(ui, color);
                }
                ui.label(RichText::new(text.into()).size(12.0).color(MUTED));
            });
        });
}

pub fn status_dot(ui: &mut Ui, color: Color32) {
    let (rect, _) = ui.allocate_exact_size(Vec2::splat(8.0), Sense::hover());
    ui.painter().circle_filled(rect.center(), 3.5, color);
}

/// A segmented control: `options` side by side in a capsule, the chosen
/// one raised.
pub fn segmented<T: PartialEq + Clone>(ui: &mut Ui, value: &mut T, options: &[(T, &str)]) -> bool {
    let mut changed = false;
    // Segments of one width, the widest label's, as AppKit draws them.
    let widest = options
        .iter()
        .map(|(_, label)| {
            egui::WidgetText::from(RichText::new(*label).size(13.0))
                .into_galley(
                    ui,
                    Some(egui::TextWrapMode::Extend),
                    f32::INFINITY,
                    TextStyle::Button,
                )
                .size()
                .x
        })
        .fold(0.0, f32::max);
    let segment = (widest + 28.0).max(64.0);
    let size = Vec2::new(segment * options.len() as f32 + 4.0, 30.0);
    ui.allocate_ui_with_layout(size, Layout::left_to_right(Align::Center), |ui| {
        Frame::new()
            .fill(white(18))
            .corner_radius(CornerRadius::same(15))
            .inner_margin(Margin::same(2))
            .show(ui, |ui| {
                ui.spacing_mut().item_spacing.x = 0.0;
                for (option, label) in options {
                    let selected = *value == *option;
                    let text =
                        RichText::new(*label)
                            .size(13.0)
                            .color(if selected { TEXT } else { MUTED });
                    let button = egui::Button::new(text)
                        .fill(if selected {
                            white(52)
                        } else {
                            Color32::TRANSPARENT
                        })
                        .corner_radius(CornerRadius::same(13))
                        .min_size(Vec2::new(segment, 26.0));
                    if ui.add(button).clicked() && !selected {
                        *value = option.clone();
                        changed = true;
                    }
                }
            });
    });
    changed
}

/// A path as a row shows it: its file name, the whole path on hover.
pub fn path_label(ui: &mut Ui, path: Option<&Path>, none: &str) -> Response {
    match path {
        Some(path) => {
            let name = path.file_name().map_or_else(
                || path.display().to_string(),
                |name| name.to_string_lossy().into_owned(),
            );
            ui.add(egui::Label::new(RichText::new(shorten(&name, 30)).color(MUTED)).truncate())
                .on_hover_text(path.display().to_string())
        }
        None => ui.label(RichText::new(none).color(FAINT)),
    }
}

/// `text` cut to `limit` characters with an ellipsis.
pub fn shorten(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    let kept: String = text.chars().take(limit.saturating_sub(1)).collect();
    format!("{kept}…")
}

/// A dark capsule behind what `add` draws, for what floats over the
/// console's screen.
pub fn overlay<R>(ui: &mut Ui, add: impl FnOnce(&mut Ui) -> R) -> R {
    Frame::new()
        .fill(Color32::from_rgba_unmultiplied(0x1C, 0x1C, 0x1E, 0xD8))
        .stroke(Stroke::new(0.5, white(36)))
        .corner_radius(CornerRadius::same(18))
        .inner_margin(Margin::symmetric(12, 6))
        .shadow(Shadow {
            offset: [0, 6],
            blur: 20,
            spread: 0,
            color: black(90),
        })
        .show(ui, add)
        .inner
}

/// A rectangle `aspect` wide per unit of height, as large as fits in
/// `within`, centred.
pub fn fit(within: Rect, aspect: f32) -> Rect {
    let size = if within.width() / within.height() > aspect {
        Vec2::new(within.height() * aspect, within.height())
    } else {
        Vec2::new(within.width(), within.width() / aspect)
    };
    Rect::from_center_size(within.center(), size)
}
