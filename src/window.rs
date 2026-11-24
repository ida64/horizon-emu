//! A host window that shows the console's screen while the machine runs.
//!
//! The window shows what someone looking at the console would see: the
//! picture the display controller sends when the panel is awake and the
//! backlight is lit, and darkness otherwise. It is redrawn from the
//! machine's state at the host display's pace, whatever the guest is doing.
//! Holding the left mouse button on the picture touches the screen there.
//! The keyboard is the pair of Joy-Con: [`RAILS_KEY`] slides them onto the
//! rails or off, [`KEYS`] are their buttons and [`STICK_KEYS`] push their
//! sticks (WASD the left one, IJKL the right one, all the way).
//! [`POWER_KEY`] is the console's power button.
//!
//! AppKit only lets a process's first thread own windows, so the window
//! belongs to the thread that called `main` and the machine runs on others.

use std::{collections::VecDeque, sync::Mutex};

use font8x8::legacy::BASIC_LEGACY;
use horizon_emu::{
    machine::Machine,
    soc::{
        dc::NoFrame,
        joycon::{Button, Stick},
    },
};
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, ScaleMode, Window, WindowOptions};

const TITLE: &str = "horizon-emu";
/// Size of the console's screen, used until the machine has a picture.
const INITIAL_SIZE: (usize, usize) = (1280, 720);
const REFRESH_RATE: usize = 60;

/// Size of a character of the log, in pixels of the picture, and the
/// distance between the tops of two lines.
const GLYPH_SIZE: usize = 8;
const LINE_PITCH: usize = 10;
/// How much of the log is shown: its newest lines, each cut to a width.
const LOG_ROWS: usize = 34;
const LOG_COLUMNS: usize = 96;
/// How many lines are kept for when the window is taller than it is now.
const LOG_KEPT: usize = 256;
const LOG_COLOR: u32 = 0x00E8_E8E8;

/// Which key holds which button of the Joy-Con.
const KEYS: [(Key, Button); 18] = [
    (Key::Space, Button::A),
    (Key::B, Button::B),
    (Key::X, Button::X),
    (Key::Y, Button::Y),
    (Key::L, Button::L),
    (Key::R, Button::R),
    (Key::Q, Button::ZL),
    (Key::E, Button::ZR),
    (Key::Up, Button::UP),
    (Key::Down, Button::DOWN),
    (Key::Left, Button::LEFT),
    (Key::Right, Button::RIGHT),
    (Key::Equal, Button::PLUS),
    (Key::Minus, Button::MINUS),
    (Key::H, Button::HOME),
    (Key::C, Button::CAPTURE),
    (Key::Comma, Button::LEFT_STICK),
    (Key::Period, Button::RIGHT_STICK),
];

/// The key that holds the console's power button.
const POWER_KEY: Key = Key::P;

/// The key that slides the Joy-Con onto the rails or off.
const RAILS_KEY: Key = Key::Backquote;

/// Which keys push which stick: up, down, left, right.
const STICK_KEYS: [(Stick, [Key; 4]); 2] = [
    (Stick::Left, [Key::W, Key::S, Key::A, Key::D]),
    (Stick::Right, [Key::I, Key::K, Key::J, Key::L]),
];

/// The emulator's own log lines, drawn over the top right corner of the
/// window's picture. They are the host talking, not something the
/// console's screen shows; screenshots do not have them.
#[derive(Default)]
pub struct LogOverlay(Mutex<VecDeque<String>>);

impl LogOverlay {
    pub fn push(&self, line: &str) {
        let mut lines = self.0.lock().unwrap();
        if lines.len() == LOG_KEPT {
            lines.pop_front();
        }
        lines.push_back(line.chars().take(LOG_COLUMNS).collect());
    }

    /// Draws the newest lines into `pixels`, a picture `width` wide, on a
    /// darkened patch so that they read on any picture.
    fn draw(&self, pixels: &mut [u32], width: usize) {
        let lines = self.0.lock().unwrap();
        let shown = lines.len().min(LOG_ROWS);
        let lines = lines.iter().skip(lines.len() - shown);
        let columns = lines.clone().map(|line| line.len()).max().unwrap_or(0);
        let patch = (columns * GLYPH_SIZE + GLYPH_SIZE).min(width);
        let (left, height) = (width - patch, pixels.len() / width.max(1));
        for row in pixels
            .chunks_exact_mut(width.max(1))
            .take((shown * LINE_PITCH + 4).min(height) * usize::from(shown > 0))
        {
            for pixel in &mut row[left..] {
                *pixel = *pixel >> 2 & 0x003F_3F3F;
            }
        }
        for (index, line) in lines.enumerate() {
            let top = 2 + index * LINE_PITCH;
            if top + GLYPH_SIZE > height {
                break;
            }
            for (column, character) in line.bytes().enumerate() {
                let glyph =
                    BASIC_LEGACY[usize::from(if character < 0x80 { character } else { b'?' })];
                let x = left + GLYPH_SIZE / 2 + column * GLYPH_SIZE;
                if x + GLYPH_SIZE > width {
                    break;
                }
                for (y, bits) in glyph.into_iter().enumerate() {
                    for bit in (0..GLYPH_SIZE).filter(|bit| bits >> bit & 1 != 0) {
                        pixels[(top + y) * width + x + bit] = LOG_COLOR;
                    }
                }
            }
        }
    }
}

pub struct ScreenWindow {
    window: Window,
    /// The picture last shown, a `0x00RRGGBB` word per pixel.
    pixels: Vec<u32>,
    size: (usize, usize),
    /// Why the screen last had no picture, so that each reason is reported
    /// once.
    problem: Option<NoFrame>,
    /// Where the mouse last touched the screen, as the machine was told.
    /// Changes only are passed on, so `--touch` fingers are left alone.
    touching: Option<(u16, u16)>,
    /// Which of [`KEYS`] were down, as the machine was told. Changes only
    /// are passed on, so `--press` is left alone.
    held: [bool; KEYS.len()],
    /// Whether [`POWER_KEY`] was down, as the machine was told.
    power_held: bool,
    /// Where the keys last pushed each stick, as the machine was told.
    sticks: [(i8, i8); 2],
}

impl ScreenWindow {
    pub fn open() -> Result<Self, String> {
        let options = WindowOptions {
            resize: true,
            scale_mode: ScaleMode::AspectRatioStretch,
            ..WindowOptions::default()
        };
        let (width, height) = INITIAL_SIZE;
        let mut window = Window::new(TITLE, width, height, options)
            .map_err(|error| format!("cannot open a window: {error}"))?;
        window.set_target_fps(REFRESH_RATE);
        Ok(Self {
            window,
            pixels: vec![0; width * height],
            size: INITIAL_SIZE,
            problem: None,
            touching: None,
            held: [false; KEYS.len()],
            power_held: false,
            sticks: [(0, 0); 2],
        })
    }

    /// Marks the window as showing a machine that no longer runs.
    pub fn set_stopped(&mut self) {
        self.window.set_title(&format!("{TITLE} (stopped)"));
    }

    /// Shows the machine's screen as it is now, with `log` over its top
    /// right corner, and waits for the next refresh. `false` once the user
    /// has closed the window.
    pub fn show(&mut self, machine: &Machine, log: &LogOverlay) -> bool {
        if !self.window.is_open() {
            return false;
        }
        let problem = match machine.screen() {
            Ok(screen) if screen.panel_on && screen.backlight_on => {
                let frame = &screen.frame;
                self.size = (frame.width as usize, frame.height as usize);
                // Already in the window's 0x00RRGGBB words; the log is
                // drawn over a copy, the frame being shared.
                self.pixels.clear();
                self.pixels.extend_from_slice(&frame.pixels);
                None
            }
            // A picture nobody can see.
            Ok(_) | Err(NoFrame::NotRunning) => {
                self.pixels.fill(0);
                None
            }
            Err(problem) => {
                self.pixels.fill(0);
                Some(problem)
            }
        };
        if let Some(problem) = &problem
            && self.problem.as_ref() != Some(problem)
        {
            eprintln!("screen: no picture, {problem}");
        }
        self.problem = problem;
        let (width, height) = self.size;
        self.pixels.resize(width * height, 0);
        log.draw(&mut self.pixels, width);
        if let Err(error) = self.window.update_with_buffer(&self.pixels, width, height) {
            eprintln!("horizon-emu: cannot draw to the window: {error}");
            return false;
        }
        // A held left button over the picture is a finger on the glass.
        let finger = self.finger();
        if finger != self.touching && machine.set_touch(finger) {
            self.touching = finger;
        }
        if self.window.is_key_pressed(RAILS_KEY, KeyRepeat::No) {
            machine.seat_joycons(!machine.joycons_seated());
        }
        for ((key, button), held) in KEYS.into_iter().zip(&mut self.held) {
            let down = self.window.is_key_down(key);
            if down != *held {
                machine.press(button, down);
                *held = down;
            }
        }
        let down = self.window.is_key_down(POWER_KEY);
        if down != self.power_held {
            machine.press_power(down);
            self.power_held = down;
        }
        for ((stick, [up, down, left, right]), last) in STICK_KEYS.into_iter().zip(&mut self.sticks) {
            let axis = |positive: Key, negative: Key| {
                i8::from(self.window.is_key_down(positive)) - i8::from(self.window.is_key_down(negative))
            };
            let now = (axis(right, left), axis(up, down));
            if now != *last {
                machine.set_stick(stick, f32::from(now.0), f32::from(now.1));
                *last = now;
            }
        }
        true
    }

    /// Where the mouse holds a finger to the screen: the left button down
    /// over the picture, in screen coordinates (1280x720, origin top left).
    /// `ScaleMode::AspectRatioStretch` centres the picture in the window,
    /// scaled to fit; the bars beside it are not the screen.
    fn finger(&self) -> Option<(u16, u16)> {
        if !self.window.get_mouse_down(MouseButton::Left) {
            return None;
        }
        let (mouse_x, mouse_y) = self.window.get_unscaled_mouse_pos(MouseMode::Pass)?;
        let (window_width, window_height) = self.window.get_size();
        let (window_width, window_height) = (window_width as f32, window_height as f32);
        let (width, height) = (self.size.0 as f32, self.size.1 as f32);
        let scale = (window_width / width).min(window_height / height);
        let x = (mouse_x - (window_width - width * scale) / 2.0) / scale;
        let y = (mouse_y - (window_height - height * scale) / 2.0) / scale;
        ((0.0..width).contains(&x) && (0.0..height).contains(&y))
            .then(|| ((x * 1280.0 / width) as u16, (y * 720.0 / height) as u16))
    }
}
