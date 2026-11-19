//! `--control SOCKET`: a Unix socket through which another program drives
//! a running machine while it runs.
//!
//! One request per line, one JSON object each, answered with one JSON line:
//!
//! ```text
//! {"op":"script","text":"press a\nsleep 1","timeout":60}
//!     the steps of a --script file, done in order; answered when the last
//!     is over, or with an error naming the step it did not get past
//! {"op":"screenshot","path":"/tmp/screen.png"}
//!     what the screen shows, saved as a PNG
//! {"op":"pixel","x":640,"y":360}
//!     the colour at a point of the 1280x720 screen, "rrggbb"
//! {"op":"status"}
//!     seconds since the machine started and whether the panel and its
//!     backlight are on, whether or not a picture can be composed
//! {"op":"stop"}
//!     stops the machine, as closing its window does
//! {"op":"save_state","path":"/tmp/snap"}
//!     a snapshot of the machine as it is into that directory (created if
//!     missing; a snapshot already there is replaced), for
//!     `horizon-emu restore`; answered with the guest time it was taken
//!     at, how long the machine stood still and how much DRAM was written
//! ```
//!
//! Every answer has `"ok"`: true, with the request's results, or false
//! with an `"error"`. Connections are served one at a time; a request runs
//! until it is answered.

use std::{
    io::{BufRead, BufReader, ErrorKind, Write},
    os::{
        fd::AsRawFd,
        unix::net::{UnixListener, UnixStream},
    },
    sync::atomic::{AtomicBool, Ordering},
    thread,
    time::{Duration, Instant},
};

use horizon_emu::machine::Machine;
use std::path::Path;

use crate::Snapshotter;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::script::{Event, Script};

/// How often a waiting server looks at whether the machine has stopped.
const POLL: Duration = Duration::from_millis(20);
/// How long a script request runs when it does not say.
const SCRIPT_TIMEOUT: Duration = Duration::from_secs(120);

#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum Request {
    Script {
        text: String,
        /// Seconds.
        timeout: Option<f64>,
    },
    Screenshot {
        path: String,
    },
    Pixel {
        x: u32,
        y: u32,
    },
    Status,
    Stop,
    SaveState {
        path: String,
    },
}

/// What a request reaches: the machine, its stop switch, when it started,
/// whether it has ended and what takes its snapshots.
pub struct Controlled<'a> {
    pub machine: &'a Machine,
    pub stop: &'a AtomicBool,
    pub started: Instant,
    pub ended: &'a AtomicBool,
    pub snapshot: &'a Snapshotter,
}

/// Serves `listener` until the machine has ended.
pub fn serve(listener: UnixListener, controlled: &Controlled) {
    if listener.set_nonblocking(true).is_err() {
        return;
    }
    while !controlled.ended.load(Ordering::SeqCst) {
        match listener.accept() {
            Ok((stream, _)) => connection(stream, controlled),
            Err(error) if error.kind() == ErrorKind::WouldBlock => {
                // Until someone connects, or for a look at whether the
                // machine has ended.
                let mut waiting = libc::pollfd {
                    fd: listener.as_raw_fd(),
                    events: libc::POLLIN,
                    revents: 0,
                };
                // SAFETY: one valid pollfd, for as long as the call.
                unsafe { libc::poll(&mut waiting, 1, POLL.as_millis() as libc::c_int) };
            }
            Err(_) => return,
        }
    }
}

fn connection(stream: UnixStream, controlled: &Controlled) {
    if stream.set_nonblocking(false).is_err() || stream.set_read_timeout(Some(POLL)).is_err() {
        return;
    }
    let Ok(mut writer) = stream.try_clone() else {
        return;
    };
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    loop {
        if controlled.ended.load(Ordering::SeqCst) {
            return;
        }
        match reader.read_line(&mut line) {
            Ok(0) => return,
            Ok(_) => {}
            Err(error) if matches!(error.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {
                continue;
            }
            Err(_) => return,
        }
        let answer = match serde_json::from_str::<Request>(&line) {
            Ok(request) => answer(request, controlled),
            Err(error) => Err(format!("not a request: {error}")),
        };
        line.clear();
        let answer = match answer {
            Ok(Value::Object(mut fields)) => {
                fields.insert("ok".into(), true.into());
                Value::Object(fields)
            }
            Ok(other) => json!({"ok": true, "result": other}),
            Err(error) => json!({"ok": false, "error": error}),
        };
        if writeln!(writer, "{answer}").is_err() {
            return;
        }
    }
}

fn answer(request: Request, controlled: &Controlled) -> Result<Value, String> {
    let machine = controlled.machine;
    match request {
        Request::Script { text, timeout } => {
            let mut script = Script::parse(&text)?;
            let timeout = match timeout {
                Some(seconds) => Duration::try_from_secs_f64(seconds)
                    .map_err(|_| format!("timeout {seconds}: not a duration"))?,
                None => SCRIPT_TIMEOUT,
            };
            let deadline = Instant::now() + timeout;
            let mut done = Vec::new();
            loop {
                script.advance(machine, |event| match event {
                    Event::Done { text, .. } => done.push(text.to_string()),
                    Event::Screenshot(path) => done.push(crate::save_screen(
                        machine,
                        path,
                        controlled.started.elapsed(),
                    )),
                });
                let Some((line, text)) = script.waiting_at() else {
                    return Ok(json!({"done": done}));
                };
                let stuck = if controlled.ended.load(Ordering::SeqCst) {
                    "the machine stopped"
                } else if Instant::now() >= deadline {
                    "timed out"
                } else {
                    thread::sleep(Duration::from_millis(5));
                    continue;
                };
                return Err(format!("{stuck} at line {line}: {text}"));
            }
        }
        Request::Screenshot { path } => {
            let screen = machine
                .screen()
                .map_err(|reason| format!("no picture: {reason}"))?;
            let file = std::fs::File::create(&path).map_err(|error| format!("{path}: {error}"))?;
            crate::write_png(file, &screen.frame).map_err(|error| format!("{path}: {error}"))?;
            Ok(json!({
                "path": path,
                "panel_on": screen.panel_on,
                "backlight_on": screen.backlight_on,
            }))
        }
        Request::Pixel { x, y } => {
            let pixel = machine
                .screen_pixel(x, y)
                .map_err(|reason| format!("no picture: {reason}"))?;
            let Some(value) = pixel.value else {
                return Err(format!(
                    "{x},{y} is outside the {}x{} screen",
                    pixel.width, pixel.height
                ));
            };
            let [_, red, green, blue] = value.to_be_bytes();
            Ok(json!({"rgb": format!("{red:02x}{green:02x}{blue:02x}")}))
        }
        Request::Status => {
            let panel = machine.panel();
            Ok(json!({
                "seconds": controlled.started.elapsed().as_secs_f64(),
                "panel_on": panel.panel_on,
                "backlight_on": panel.backlight_on,
            }))
        }
        Request::Stop => {
            controlled.stop.store(true, Ordering::SeqCst);
            Ok(json!({}))
        }
        Request::SaveState { path } => {
            let report = controlled.snapshot.save(machine, Path::new(&path))?;
            Ok(json!({
                "path": path,
                "guest_seconds": report.guest_time.as_secs_f64(),
                "paused_seconds": report.paused_for.as_secs_f64(),
                "ram_bytes": report.ram_bytes,
            }))
        }
    }
}
