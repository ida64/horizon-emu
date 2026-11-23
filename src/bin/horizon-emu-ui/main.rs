//! `horizon-emu-ui`: a native window for making, running and watching
//! machines of the emulator. Each machine runs as a `horizon-emu` process
//! of its own (Hypervisor.framework allows one virtual machine per
//! process), driven through its control socket.

mod app;
mod child;
mod dialogs;
mod glass;
mod model;
mod screen;
mod tabs;
mod theme;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("Horizon Emu")
            .with_inner_size([1380.0, 940.0])
            .with_min_inner_size([980.0, 640.0])
            // Glass behind everything, edge to edge: the window's own
            // buttons float over it.
            .with_transparent(true)
            .with_fullsize_content_view(true)
            .with_title_shown(false)
            .with_titlebar_shown(false),
        ..Default::default()
    };
    eframe::run_native(
        "Horizon Emu",
        options,
        Box::new(|cc| Ok(Box::new(app::App::new(cc)))),
    )
}
