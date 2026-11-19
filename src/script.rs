//! Scripts for `--script`: things done to the console one after the other,
//! each when the one before is over, with waits for what the screen shows.
//!
//! The times of `--press` and `--touch` are seconds since the machine
//! started, and how far the guest has got by then depends on how busy the
//! host is. A script waits for the picture instead: every set-up screen
//! has something that tells it apart (the step dots at the top right, a
//! button's frame), so a step can wait for a pixel to take a colour.
//!
//! A screen that is drawn is not always a screen that listens yet, and
//! on a busy host a short press can fall between two of the guest's
//! frames. A press or a touch can therefore be given what it is for (a
//! pixel again) and is then repeated until that has happened. HOME (23.5.0)
//! shows both: it ignores buttons and touches for about a second after it
//! comes up from the lock screen or opens a dialog, and it takes up a
//! finger only if the touch lasts until its next frame. The measurements
//! are in `horizon_emu::soc::touch::TouchInput`.
//!
//! The lock screen goes to sleep by itself 20 s after the last input in
//! handheld mode. That is the firmware's doing (23.5.0), not the
//! emulator's:
//! - When the lock screen comes up, `am` hands `idle:sys` (in `omm`) the
//!   handling context `[0x14, 0x3c, 0, ...]` (`SetHandlingContext`,
//!   command 3). `omm` `0x48e10` takes the auto-sleep time from its first
//!   word in handheld mode and from its second otherwise \[inference:
//!   which flag tells the modes apart\], so 20 s in handheld mode.
//! - Once a second (`0x6420`, `0x48930`) `idle` reads the time of the last
//!   input from `ins:r` command 0 (a 19.2 MHz tick; in traces it follows
//!   every button press \[inference\]). When that time is older than the
//!   limit (`0x48cb0`), it signals its auto-power-down event.
//! - `spsm` (in `omm`) turns that event into notification message 3 for
//!   `am` (`0x78810` posts power state event 9, and `0x80970` maps it to
//!   message 3). About 1.4 s later `qlaunch` calls `StartSleepSequence`.
//!
//! After a boot, the emulated lock screen comes up 14 to 23 s after the
//! last input. The boot is slower than the hardware's, see `SDMMC2_BASE`
//! in `soc` \[inference: hardware boot times not measured\]. So a script
//! that waits for the lock screen and then holds POWER can lose the race
//! with this auto-sleep. If the time has already run out, the console
//! sleeps while POWER is held instead of opening the power menu, and with
//! no press at all it sleeps the same way. To avoid this, press a button
//! that does nothing there while waiting, so that the lock screen comes up
//! with nearly all of its 20 s left:
//!
//! ```text
//! press zl 300 until 820,330 2d2d2d 4   # the lock screen's home icon
//! press power 3500                      # the power menu
//! ```

use std::time::{Duration, Instant};

use horizon_emu::{
    machine::Machine,
    soc::{dc::Pixel, joycon::Button, ntag::Ntag215},
};

/// How long a button or a finger is held when a line does not say. That is
/// longer than the longest gap between two of HOME's frames measured on a
/// busy host (132 ms), so HOME sees the touch. 50 ms touches were lost 7
/// times in 74, and none of 108 touches held 100–500 ms was lost.
const HOLD: Duration = Duration::from_millis(300);
/// How long a repeated press or touch is given to have its effect.
const AGAIN: Duration = Duration::from_secs(3);
/// How long a drag takes when a line does not say.
const DRAG: Duration = Duration::from_secs(1);
/// How far each of red, green and blue may be from the colour waited for.
const TOLERANCE: u8 = 16;
/// Size of the console's screen.
const SCREEN: (u32, u32) = (1280, 720);

/// Something that is pressed: a button of the Joy-Con, the console's
/// power button (`power`), or a button of the Pro Controller in range of
/// its radio (`pro:BUTTON`), SYNC (`pro:sync`) among them.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Key {
    Joycon(Button),
    Power,
    Pro(Button),
    ProSync,
}

impl Key {
    /// `power`, `pro:sync`, or buttons joined by `+` (held together),
    /// prefixed `pro:` for the Pro Controller's.
    fn named(name: &str) -> Option<Self> {
        let chord = |names: &str| {
            names
                .split('+')
                .map(Button::named)
                .reduce(|chord, button| Some(chord?.and(button?)))?
        };
        match name {
            "power" => Some(Self::Power),
            "pro:sync" => Some(Self::ProSync),
            _ => match name.strip_prefix("pro:") {
                Some(buttons) => chord(buttons).map(Self::Pro),
                None => chord(name).map(Self::Joycon),
            },
        }
    }

    fn hold(self, machine: &Machine, held: bool) {
        match self {
            Self::Joycon(button) => machine.press(button, held),
            Self::Power => machine.press_power(held),
            Self::Pro(button) => machine.bluetooth().press(button, held),
            // The controller acts on SYNC as it goes down.
            Self::ProSync if held => machine.bluetooth().sync(),
            Self::ProSync => {}
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Step {
    /// Until the pixel at `(x, y)` of the screen has this colour.
    Wait {
        x: u32,
        y: u32,
        rgb: [u8; 3],
    },
    Sleep(Duration),
    Press(Key, Duration),
    Touch((u16, u16), Duration),
    /// A finger put down at `from`, moved along the straight line to `to`
    /// over `span`, then lifted.
    Drag {
        from: (u16, u16),
        to: (u16, u16),
        span: Duration,
    },
    /// A press or a touch done again, `every` so often, until the pixel
    /// at `(x, y)` has this colour; not at all if it has it already.
    Until {
        action: Box<Step>,
        x: u32,
        y: u32,
        rgb: [u8; 3],
        every: Duration,
    },
    Joycons(bool),
    /// An NFC tag held over the right Joy-Con's reader, or taken away.
    Amiibo(Option<Box<Ntag215>>),
    Screenshot(String),
}

/// What is let go of when a step's hold is over.
#[derive(Debug, Clone, Copy)]
enum Held {
    Nothing,
    Button(Key),
    Finger,
    /// A dragging finger, at `from` at `start` and at `to` `span` later.
    Drag {
        from: (u16, u16),
        to: (u16, u16),
        start: Instant,
        span: Duration,
    },
}

/// Something a script did that the caller reports or carries out.
pub enum Event<'a> {
    /// The step on this line of the file is over.
    Done { line: usize, text: &'a str },
    /// The screen is to be saved to this file.
    Screenshot(&'a str),
}

pub struct Script {
    /// Line number, text and meaning of each step.
    steps: Vec<(usize, String, Step)>,
    next: usize,
    /// The step in progress ends then, letting go of this.
    until: Option<(Instant, Held)>,
    /// When a step that repeats does what it does again.
    again_at: Option<Instant>,
}

impl Script {
    /// Reads a script: a step per line, nothing after `#` (but in `type`).
    ///
    /// ```text
    /// wait X,Y RRGGBB       until that pixel of the 1280x720 screen is that colour
    /// sleep SECONDS
    /// press BUTTON [HOLD_MS] [until X,Y RRGGBB [SECONDS]]
    ///                       BUTTON: one of the Joy-Con's (several joined
    ///                       by `+` are held together, `l+r`), `power`, or
    ///                       `pro:` and the Pro Controller's or `sync`
    ///                       (with --pro-controller)
    /// touch X,Y [HOLD_MS] [until X,Y RRGGBB [SECONDS]]
    ///                       with `until`: again every SECONDS (3) until
    ///                       that pixel is that colour; not at all if it is
    /// drag X,Y X,Y [MS]     a finger down at the first point, moved along
    ///                       the straight line to the second over MS (1000),
    ///                       then lifted
    /// type TEXT             the rest of the line, on the open on-screen
    ///                       keyboard, key by key
    /// joycons on|off
    /// amiibo FILE [writable]  hold the NTAG215 dumped in FILE (540 bytes)
    ///                       over the right Joy-Con's NFC reader, with
    ///                       `writable` saving what is written to it there
    /// amiibo off            take it away
    /// screenshot FILE
    /// ```
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut steps = Vec::new();
        for (index, line) in text.lines().enumerate() {
            // What `type` types is the rest of its line, `#` and spaces too.
            if let Some(typed) = line.trim_start().strip_prefix("type ") {
                let touches = keyboard_touches(typed)
                    .map_err(|error| format!("line {}: {error}", index + 1))?;
                // Only the last touch reports the line as done.
                let count = touches.len();
                for (at, touch) in touches.into_iter().enumerate() {
                    let text = if at + 1 == count { line.trim() } else { "" };
                    steps.push((index + 1, text.to_string(), touch));
                }
                continue;
            }
            let line = line.split('#').next().unwrap_or_default().trim();
            if line.is_empty() {
                continue;
            }
            let step = parse_step(line)
                .ok_or_else(|| format!("line {}: not a script step: {line}", index + 1))?;
            steps.push((index + 1, line.to_string(), step));
        }
        Ok(Self {
            steps,
            next: 0,
            until: None,
            again_at: None,
        })
    }

    /// Does what is due now. Called every few milliseconds while the
    /// machine runs.
    pub fn advance(&mut self, machine: &Machine, mut report: impl FnMut(Event)) {
        loop {
            if let Some((end, held)) = self.until {
                if Instant::now() < end {
                    if let Held::Drag {
                        from,
                        to,
                        start,
                        span,
                    } = held
                    {
                        machine.set_touch(Some(drag_point(from, to, start.elapsed(), span)));
                    }
                    return;
                }
                match held {
                    Held::Nothing => {}
                    Held::Button(key) => key.hold(machine, false),
                    Held::Finger => {
                        machine.set_touch(None);
                    }
                    Held::Drag { to, .. } => {
                        machine.set_touch(Some(to));
                        machine.set_touch(None);
                    }
                }
                self.until = None;
                match self.steps.get(self.next) {
                    Some((_, _, Step::Until { every, .. })) => {
                        self.again_at = Some(Instant::now() + *every);
                    }
                    _ => self.finish(&mut report),
                }
                continue;
            }
            let Some((_, _, step)) = self.steps.get(self.next) else {
                return;
            };
            let now = Instant::now();
            let hold = |span: Duration, held| Some((now + span, held));
            match step {
                Step::Wait { x, y, rgb } => {
                    if !shows(machine, *x, *y, *rgb) {
                        return;
                    }
                    self.finish(&mut report);
                }
                Step::Sleep(span) => self.until = hold(*span, Held::Nothing),
                Step::Press(key, span) => {
                    key.hold(machine, true);
                    self.until = hold(*span, Held::Button(*key));
                }
                Step::Touch(place, span) => {
                    // The touch controller is told once it scans.
                    if !machine.set_touch(Some(*place)) {
                        return;
                    }
                    self.until = hold(*span, Held::Finger);
                }
                Step::Drag { from, to, span } => {
                    if !machine.set_touch(Some(*from)) {
                        return;
                    }
                    self.until = hold(
                        *span,
                        Held::Drag {
                            from: *from,
                            to: *to,
                            start: now,
                            span: *span,
                        },
                    );
                }
                Step::Until {
                    action, x, y, rgb, ..
                } => {
                    if shows(machine, *x, *y, *rgb) {
                        self.again_at = None;
                        self.finish(&mut report);
                        continue;
                    }
                    if self.again_at.is_some_and(|again| now < again) {
                        return;
                    }
                    match **action {
                        Step::Press(key, span) => {
                            key.hold(machine, true);
                            self.until = hold(span, Held::Button(key));
                        }
                        Step::Touch(place, span) if machine.set_touch(Some(place)) => {
                            self.until = hold(span, Held::Finger);
                        }
                        _ => return,
                    }
                }
                Step::Joycons(seated) => {
                    machine.seat_joycons(*seated);
                    self.finish(&mut report);
                }
                Step::Amiibo(tag) => {
                    machine.place_tag(tag.as_deref().cloned());
                    self.finish(&mut report);
                }
                Step::Screenshot(path) => {
                    report(Event::Screenshot(path));
                    self.finish(&mut report);
                }
            }
        }
    }

    fn finish(&mut self, report: &mut impl FnMut(Event)) {
        let (line, text, _) = &self.steps[self.next];
        if !text.is_empty() {
            report(Event::Done { line: *line, text });
        }
        self.next += 1;
    }

    /// The step the script has not got past, if it has not ended.
    pub fn waiting_at(&self) -> Option<(usize, &str)> {
        self.steps
            .get(self.next)
            .map(|(line, text, _)| (*line, text.as_str()))
    }
}

/// Whether someone looking at the console sees that colour at `(x, y)`.
/// Only that pixel is worked out, not the whole picture.
fn shows(machine: &Machine, x: u32, y: u32, rgb: [u8; 3]) -> bool {
    let panel = machine.panel();
    if !(panel.panel_on && panel.backlight_on) {
        return false;
    }
    let Ok(Pixel {
        value: Some(value), ..
    }) = machine.screen_pixel(x, y)
    else {
        return false;
    };
    let [_, red, green, blue] = value.to_be_bytes();
    [red, green, blue]
        .into_iter()
        .zip(rgb)
        .all(|(shown, wanted)| shown.abs_diff(wanted) <= TOLERANCE)
}

/// The key pages of the system's on-screen keyboard (US English), as the
/// four key rows above the space bar: the lower case page, the shifted
/// page and the `#+=` page. Keys are 96 pixels apart from x = 99; the rows
/// are at y = 363, 428, 492 and 556 (measured on the keyboard System
/// Settings opens for a proxy server).
const KEYBOARD_PAGES: [[&str; 4]; 3] = [
    ["1234567890@", "qwertyuiop+", "asdfghjkl_:", "zxcvbnm,.-/"],
    ["1234567890@", "QWERTYUIOP=", "ASDFGHJKL&;", "ZXCVBNM*#!?"],
    ["1234567890-", "!@#$%^&*()_", "~`=\\+{}|[]", "<>;:\"',.?/"],
];
const KEY_ROWS: [u16; 4] = [363, 428, 492, 556];
const KEY_X: (u16, u16) = (99, 96);
const SPACE_KEY: (u16, u16) = (723, 619);
/// Shift (one character), and the key that switches between the letters
/// and the `#+=` page (it reads `ABC` there).
const SHIFT_KEY: (u16, u16) = (195, 619);
const SYMBOLS_KEY: (u16, u16) = (291, 619);
/// How long a key is touched, and the pause after it.
const KEY_HOLD: Duration = Duration::from_millis(120);
const KEY_GAP: Duration = Duration::from_millis(300);

/// The touches that type `text` on the on-screen keyboard, from its lower
/// case page and leaving it there. A character on the lower case page is
/// one touch; one on the shifted page is shift and then it (shift lasts one
/// character); one only on the `#+=` page goes there and back.
fn keyboard_touches(text: &str) -> Result<Vec<Step>, String> {
    let find = |page: usize, character: char| {
        KEYBOARD_PAGES[page]
            .iter()
            .enumerate()
            .find_map(|(row, keys)| {
                let column = keys.chars().position(|key| key == character)?;
                Some((KEY_X.0 + KEY_X.1 * column as u16, KEY_ROWS[row]))
            })
    };
    let mut keys = Vec::new();
    for character in text.chars() {
        if character == ' ' {
            keys.push(SPACE_KEY);
        } else if let Some(key) = find(0, character) {
            keys.push(key);
        } else if let Some(key) = find(1, character) {
            keys.extend([SHIFT_KEY, key]);
        } else if let Some(key) = find(2, character) {
            keys.extend([SYMBOLS_KEY, key, SYMBOLS_KEY]);
        } else {
            return Err(format!("{character:?} is on no page of the keyboard"));
        }
    }
    Ok(keys
        .into_iter()
        .flat_map(|key| [Step::Touch(key, KEY_HOLD), Step::Sleep(KEY_GAP)])
        .collect())
}

fn parse_step(line: &str) -> Option<Step> {
    let mut words = line.split_whitespace().peekable();
    let command = words.next()?;
    type Words<'a> = std::iter::Peekable<std::str::SplitWhitespace<'a>>;
    let place = |words: &mut Words| {
        let (x, y) = words.next()?.split_once(',')?;
        let (x, y): (u32, u32) = (x.parse().ok()?, y.parse().ok()?);
        (x < SCREEN.0 && y < SCREEN.1).then_some((x, y))
    };
    let hold = |words: &mut Words| match words.next_if(|word| *word != "until") {
        Some(milliseconds) => milliseconds.parse().ok().map(Duration::from_millis),
        None => Some(HOLD),
    };
    let colour = |words: &mut Words| {
        let colour = words.next().filter(|colour| colour.len() == 6)?;
        let [_, red, green, blue] = u32::from_str_radix(colour, 16).ok()?.to_be_bytes();
        Some([red, green, blue])
    };
    // `... until X,Y RRGGBB [SECONDS]` after a press or a touch.
    let until = |words: &mut Words, action: Step| {
        if words.next_if_eq(&"until").is_none() {
            return Some(action);
        }
        let (x, y) = place(words)?;
        let rgb = colour(words)?;
        let every = match words.next() {
            Some(seconds) => Duration::try_from_secs_f64(seconds.parse().ok()?).ok()?,
            None => AGAIN,
        };
        Some(Step::Until {
            action: Box::new(action),
            x,
            y,
            rgb,
            every,
        })
    };
    let step = match command {
        "wait" => {
            let (x, y) = place(&mut words)?;
            let rgb = colour(&mut words)?;
            Step::Wait { x, y, rgb }
        }
        "sleep" => Step::Sleep(Duration::try_from_secs_f64(words.next()?.parse().ok()?).ok()?),
        "press" => {
            let press = Step::Press(Key::named(words.next()?)?, hold(&mut words)?);
            until(&mut words, press)?
        }
        "touch" => {
            let (x, y) = place(&mut words)?;
            let touch = Step::Touch((x as u16, y as u16), hold(&mut words)?);
            until(&mut words, touch)?
        }
        "drag" => {
            let (x0, y0) = place(&mut words)?;
            let (x1, y1) = place(&mut words)?;
            let span = match words.next() {
                Some(milliseconds) => Duration::from_millis(milliseconds.parse().ok()?),
                None => DRAG,
            };
            Step::Drag {
                from: (x0 as u16, y0 as u16),
                to: (x1 as u16, y1 as u16),
                span,
            }
        }
        "joycons" => Step::Joycons(match words.next()? {
            "on" => true,
            "off" => false,
            _ => return None,
        }),
        "amiibo" => match (words.next()?, words.next_if_eq(&"writable")) {
            ("off", None) => Step::Amiibo(None),
            (file, writable) => Step::Amiibo(Some(Box::new(
                Ntag215::load(std::path::Path::new(file), writable.is_some()).ok()?,
            ))),
        },
        "screenshot" => Step::Screenshot(words.next()?.to_string()),
        _ => return None,
    };
    words.next().is_none().then_some(step)
}

/// Where a dragging finger is `elapsed` into a drag of `span` from `from`
/// to `to`.
fn drag_point(from: (u16, u16), to: (u16, u16), elapsed: Duration, span: Duration) -> (u16, u16) {
    let t = (elapsed.as_secs_f64() / span.as_secs_f64().max(f64::EPSILON)).min(1.0);
    let along = |a: u16, b: u16| (f64::from(a) + (f64::from(b) - f64::from(a)) * t).round() as u16;
    (along(from.0, to.0), along(from.1, to.1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steps_are_read_and_what_is_not_a_step_is_named() {
        let script = Script::parse(
            "# to the region screen\n\
             wait 999,54 2f4fe8\n\
             press a 500   # English\n\
             press power\n\
             \n\
             touch 640,219 until 1059,54 3250f0\n\
             joycons off\n\
             sleep 1.5\n\
             screenshot /tmp/a.png\n",
        )
        .unwrap();
        let steps: Vec<_> = script
            .steps
            .iter()
            .map(|(line, _, step)| (*line, step.clone()))
            .collect();
        assert_eq!(
            steps,
            [
                (
                    2,
                    Step::Wait {
                        x: 999,
                        y: 54,
                        rgb: [0x2F, 0x4F, 0xE8]
                    }
                ),
                (
                    3,
                    Step::Press(Key::Joycon(Button::A), Duration::from_millis(500))
                ),
                (4, Step::Press(Key::Power, HOLD)),
                (
                    6,
                    Step::Until {
                        action: Box::new(Step::Touch((640, 219), HOLD)),
                        x: 1059,
                        y: 54,
                        rgb: [0x32, 0x50, 0xF0],
                        every: AGAIN,
                    }
                ),
                (7, Step::Joycons(false)),
                (8, Step::Sleep(Duration::from_millis(1500))),
                (9, Step::Screenshot("/tmp/a.png".into())),
            ]
        );
        assert_eq!(script.waiting_at(), Some((2, "wait 999,54 2f4fe8")));
        for bad in [
            "touch 1280,10",
            "wait 10,10 fff",
            "press select",
            "joycons maybe",
            "sleep 1 2",
            "press a until 10,10",
            "wait 10,10 3250f0 until 10,10 3250f0",
        ] {
            let error = Script::parse(&format!("sleep 1\n{bad}\n")).err().unwrap();
            assert_eq!(error, format!("line 2: not a script step: {bad}"));
        }
    }

    #[test]
    fn typed_text_keeps_hashes_and_takes_each_page_its_way() {
        let script = Script::parse("type a#A~\n").unwrap();
        let touched: Vec<_> = (script.steps.iter())
            .filter_map(|(_, _, step)| match step {
                Step::Touch(place, _) => Some(*place),
                _ => None,
            })
            .collect();
        // `a`; shift, `#`; shift, `A`; `~` there and back from `#+=`.
        assert_eq!(
            touched,
            [
                (99, 492),
                SHIFT_KEY,
                (867, 556),
                SHIFT_KEY,
                (99, 492),
                SYMBOLS_KEY,
                (99, 492),
                SYMBOLS_KEY
            ]
        );
        // The line is reported done once, by its last touch's pause.
        let reported: Vec<_> = (script.steps.iter())
            .filter(|(_, text, _)| !text.is_empty())
            .collect();
        assert_eq!(reported.len(), 1);
        assert_eq!(
            Script::parse("sleep 1\ntype é\n").err().unwrap(),
            "line 2: 'é' is on no page of the keyboard"
        );
    }

    /// `drag` reads two points and an optional duration, and moves the
    /// finger along the line between them in proportion to the time.
    #[test]
    fn drag_is_read_and_moves_along_its_line() {
        let script = Script::parse("drag 100,200 300,240 500\ndrag 0,0 10,10\n").unwrap();
        let steps: Vec<_> = script.steps.iter().map(|(_, _, s)| s.clone()).collect();
        assert_eq!(
            steps,
            [
                Step::Drag {
                    from: (100, 200),
                    to: (300, 240),
                    span: Duration::from_millis(500),
                },
                Step::Drag {
                    from: (0, 0),
                    to: (10, 10),
                    span: DRAG,
                },
            ]
        );
        let span = Duration::from_millis(500);
        let at = |ms| drag_point((100, 200), (300, 240), Duration::from_millis(ms), span);
        assert_eq!(at(0), (100, 200));
        assert_eq!(at(250), (200, 220));
        assert_eq!(at(900), (300, 240));
        for bad in [
            "drag 1,1",
            "drag 1,1 1280,1",
            "drag 1,1 2,2 x",
            "drag 1,1 2,2 3 4",
        ] {
            assert!(Script::parse(bad).is_err(), "{bad}");
        }
    }
}
