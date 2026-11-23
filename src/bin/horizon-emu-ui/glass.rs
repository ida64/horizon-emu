//! The window's native materials, under everything egui draws (which
//! clears to transparent):
//!
//! - the backdrop, the whole window: behind-window vibrancy
//!   (`NSVisualEffectView`), the desktop showing through, blurred;
//! - the sidebar, a floating pane of Liquid Glass (`NSGlassEffectView`,
//!   macOS 26) that egui places where it lays the sidebar out, each frame.
//!
//! Both are siblings of the window's content view, below it in the
//! window's frame view. The content view stays winit's own: winit takes
//! the window's content view to be its view (`WindowDelegate::view`), and
//! crashes on any other. The window is dark whatever the system's
//! appearance, since the UI's text is light.

use objc2::{MainThreadMarker, msg_send, rc::Retained, runtime::AnyClass};
use objc2_app_kit::{
    NSAppearance, NSAppearanceNameDarkAqua, NSAutoresizingMaskOptions, NSColor, NSGlassEffectView,
    NSGlassEffectViewStyle, NSView, NSVisualEffectBlendingMode, NSVisualEffectMaterial,
    NSVisualEffectState, NSVisualEffectView, NSWindowOrderingMode,
};
use objc2_foundation::{NSPoint, NSRect, NSSize};
use raw_window_handle::{HasWindowHandle, RawWindowHandle};

/// Corner radius of the sidebar's glass, in points.
pub const SIDEBAR_RADIUS: f64 = 24.0;

pub struct Glass {
    /// winit's view, which egui draws into.
    content: Retained<NSView>,
    /// The window's frame view, which holds the materials and `content`.
    frame_view: Retained<NSView>,
    /// The sidebar's pane, `None` before macOS 26.
    sidebar: Option<Retained<NSView>>,
    /// Where the sidebar was put last, in egui's points.
    placed: Option<eframe::egui::Rect>,
}

impl Glass {
    /// Puts the materials behind the content of `frame`'s window.
    pub fn install(frame: &eframe::Frame) -> Result<Self, String> {
        let mtm = MainThreadMarker::new().ok_or("the window belongs to the main thread")?;
        let handle = frame.window_handle().map_err(|error| error.to_string())?;
        let RawWindowHandle::AppKit(appkit) = handle.as_raw() else {
            return Err("not an AppKit window".into());
        };
        // SAFETY: the handle names the window's live content view.
        let content: Retained<NSView> =
            unsafe { Retained::retain(appkit.ns_view.as_ptr().cast::<NSView>()) }
                .ok_or("no content view")?;
        let window = content.window().ok_or("the view is in no window")?;
        // SAFETY: the window's frame view, alive as long as the window.
        let frame_view =
            unsafe { content.superview() }.ok_or("the content view has no frame view")?;
        if let Some(dark) = NSAppearance::appearanceNamed(unsafe { NSAppearanceNameDarkAqua }) {
            // SAFETY: NSWindow conforms to NSAppearanceCustomization.
            let _: () = unsafe { msg_send![&*window, setAppearance: &*dark] };
        }
        window.setOpaque(false);
        window.setBackgroundColor(Some(&NSColor::clearColor()));
        window.setTitlebarAppearsTransparent(true);

        let sizable =
            NSAutoresizingMaskOptions::ViewWidthSizable | NSAutoresizingMaskOptions::ViewHeightSizable;
        let backdrop = NSVisualEffectView::initWithFrame(mtm.alloc(), frame_view.bounds());
        backdrop.setMaterial(NSVisualEffectMaterial::UnderWindowBackground);
        backdrop.setBlendingMode(NSVisualEffectBlendingMode::BehindWindow);
        backdrop.setState(NSVisualEffectState::Active);
        backdrop.setAutoresizingMask(sizable);
        frame_view.addSubview_positioned_relativeTo(&backdrop, NSWindowOrderingMode::Below, Some(&content));

        let sidebar = AnyClass::get(c"NSGlassEffectView").is_some().then(|| {
            let glass = NSGlassEffectView::initWithFrame(mtm.alloc(), NSRect::ZERO);
            glass.setStyle(NSGlassEffectViewStyle::Regular);
            glass.setCornerRadius(SIDEBAR_RADIUS);
            glass.setHidden(true);
            frame_view.addSubview_positioned_relativeTo(&glass, NSWindowOrderingMode::Below, Some(&content));
            Retained::into_super(glass)
        });
        Ok(Self {
            content,
            frame_view,
            sidebar,
            placed: None,
        })
    }

    /// Puts the sidebar's glass at `rect` (egui points, from the top left
    /// of the content view), or hides it.
    pub fn place_sidebar(&mut self, rect: Option<eframe::egui::Rect>) {
        if rect == self.placed {
            return;
        }
        self.placed = rect;
        let Some(sidebar) = &self.sidebar else { return };
        let Some(rect) = rect else {
            sidebar.setHidden(true);
            return;
        };
        let top = if self.content.isFlipped() {
            f64::from(rect.top())
        } else {
            self.content.bounds().size.height - f64::from(rect.bottom())
        };
        let local = NSRect::new(
            NSPoint::new(f64::from(rect.left()), top),
            NSSize::new(f64::from(rect.width()), f64::from(rect.height())),
        );
        sidebar.setFrame(self.content.convertRect_toView(local, Some(&self.frame_view)));
        sidebar.setHidden(false);
    }

    /// Whether the sidebar is real glass (macOS 26), so that egui need not
    /// paint one.
    pub fn has_sidebar(&self) -> bool {
        self.sidebar.is_some()
    }
}
