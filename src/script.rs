//! Scripts for `--script`: things done to the console one after the other,
//! each when the one before is over, with waits for what the screen shows.
//!
//! The times of `--press` and `--touch` are seconds since the machine
//! started, and how far the guest has got by then depends on how busy the
//! host is. A script waits for the picture instead: every set-up screen
//! has something that tells it apart (the step dots at the top right, a
//! button's frame), so a step can wait for a pixel to take a colour.

use std::time::{Duration, Instant};

use horizon_emu::{machine::Machine, soc::joycon::Button};

/// How long a button or a finger is held when a line does not say.
const HOLD: Duration = Duration::from_millis(300);
/// How far each of red, green and blue may be from the colour waited for.
const TOLERANCE: u8 = 16;
/// Size of the console's screen.
const SCREEN: (u32, u32) = (1280, 720);

#[derive(Debug, Clone, PartialEq)]
enum Step {
    /// Until the pixel at `(x, y)` of the screen has this colour.
    Wait {
        x: u32,
        y: u32,
        rgb: [u8; 3],
    },
    Sleep(Duration),
    Press(Button, Duration),
    Touch((u16, u16), Duration),
    Joycons(bool),
    Screenshot(String),
}

/// What is let go of when a step's hold is over.
#[derive(Debug, Clone, Copy)]
enum Held {
    Nothing,
    Button(Button),
    Finger,
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
}

impl Script {
    /// Reads a script: a step per line, nothing after `#`.
    ///
    /// ```text
    /// wait X,Y RRGGBB       until that pixel of the 1280x720 screen is that colour
    /// sleep SECONDS
    /// press BUTTON [HOLD_MS]
    /// touch X,Y [HOLD_MS]
    /// joycons on|off
    /// screenshot FILE
    /// ```
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut steps = Vec::new();
        for (index, line) in text.lines().enumerate() {
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
        })
    }

    /// Does what is due now. Called every few milliseconds while the
    /// machine runs.
    pub fn advance(&mut self, machine: &Machine, mut report: impl FnMut(Event)) {
        loop {
            if let Some((end, held)) = self.until {
                if Instant::now() < end {
                    return;
                }
                match held {
                    Held::Nothing => {}
                    Held::Button(button) => machine.press(button, false),
                    Held::Finger => {
                        machine.set_touch(None);
                    }
                }
                self.until = None;
                self.finish(&mut report);
                continue;
            }
            let Some((_, _, step)) = self.steps.get(self.next) else {
                return;
            };
            let hold = |span: Duration, held| Some((Instant::now() + span, held));
            match step {
                Step::Wait { x, y, rgb } => {
                    if !shows(machine, *x, *y, *rgb) {
                        return;
                    }
                    self.finish(&mut report);
                }
                Step::Sleep(span) => self.until = hold(*span, Held::Nothing),
                Step::Press(button, span) => {
                    machine.press(*button, true);
                    self.until = hold(*span, Held::Button(*button));
                }
                Step::Touch(place, span) => {
                    // The touch controller is told once it scans.
                    if !machine.set_touch(Some(*place)) {
                        return;
                    }
                    self.until = hold(*span, Held::Finger);
                }
                Step::Joycons(seated) => {
                    machine.seat_joycons(*seated);
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
        report(Event::Done { line: *line, text });
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
fn shows(machine: &Machine, x: u32, y: u32, rgb: [u8; 3]) -> bool {
    let Ok(screen) = machine.screen() else {
        return false;
    };
    let frame = &screen.frame;
    if !(screen.panel_on && screen.backlight_on) || x >= frame.width || y >= frame.height {
        return false;
    }
    let [_, red, green, blue] = frame.pixels[(y * frame.width + x) as usize].to_be_bytes();
    [red, green, blue]
        .into_iter()
        .zip(rgb)
        .all(|(shown, wanted)| shown.abs_diff(wanted) <= TOLERANCE)
}

fn parse_step(line: &str) -> Option<Step> {
    let mut words = line.split_whitespace();
    let command = words.next()?;
    let place = |words: &mut std::str::SplitWhitespace| {
        let (x, y) = words.next()?.split_once(',')?;
        let (x, y): (u32, u32) = (x.parse().ok()?, y.parse().ok()?);
        (x < SCREEN.0 && y < SCREEN.1).then_some((x, y))
    };
    let hold = |words: &mut std::str::SplitWhitespace| match words.next() {
        Some(milliseconds) => milliseconds.parse().ok().map(Duration::from_millis),
        None => Some(HOLD),
    };
    let step = match command {
        "wait" => {
            let (x, y) = place(&mut words)?;
            let colour = words.next().filter(|colour| colour.len() == 6)?;
            let [_, red, green, blue] = u32::from_str_radix(colour, 16).ok()?.to_be_bytes();
            Step::Wait {
                x,
                y,
                rgb: [red, green, blue],
            }
        }
        "sleep" => Step::Sleep(Duration::try_from_secs_f64(words.next()?.parse().ok()?).ok()?),
        "press" => Step::Press(Button::named(words.next()?)?, hold(&mut words)?),
        "touch" => {
            let (x, y) = place(&mut words)?;
            Step::Touch((x as u16, y as u16), hold(&mut words)?)
        }
        "joycons" => Step::Joycons(match words.next()? {
            "on" => true,
            "off" => false,
            _ => return None,
        }),
        "screenshot" => Step::Screenshot(words.next()?.to_string()),
        _ => return None,
    };
    words.next().is_none().then_some(step)
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
             \n\
             touch 640,219\n\
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
                (3, Step::Press(Button::A, Duration::from_millis(500))),
                (5, Step::Touch((640, 219), HOLD)),
                (6, Step::Joycons(false)),
                (7, Step::Sleep(Duration::from_millis(1500))),
                (8, Step::Screenshot("/tmp/a.png".into())),
            ]
        );
        assert_eq!(script.waiting_at(), Some((2, "wait 999,54 2f4fe8")));
        for bad in [
            "touch 1280,10",
            "wait 10,10 fff",
            "press select",
            "joycons maybe",
            "sleep 1 2",
        ] {
            let error = Script::parse(&format!("sleep 1\n{bad}\n")).err().unwrap();
            assert_eq!(error, format!("line 2: not a script step: {bad}"));
        }
    }
}
