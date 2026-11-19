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
//!     seconds since the program started, whether the panel and its
//!     backlight are on, whether or not a picture can be composed, and
//!     "state": "running", "restarting" (between the machine a restart
//!     took down and the one it powers up) or "stopped", with "stopped"
//!     saying why (a power-off among the reasons)
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
//! until it is answered. Requests are answered after the machine stopped,
//! for as long as the program runs (its window open): a status, a
//! screenshot of the screen as left.

use std::{
    io::{BufRead, BufReader, ErrorKind, Write},
    os::{
        fd::AsRawFd,
        unix::net::{UnixListener, UnixStream},
    },
    path::Path,
    sync::{
        Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    Console, Snapshotter,
    script::{Event, Script},
};

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

/// What a request reaches: the console, its stop switch, when the program
/// started, why the machine stopped once it has, whether the program is
/// closing and what takes snapshots.
pub struct Controlled<'a> {
    pub console: &'a Console,
    pub stop: &'a AtomicBool,
    pub started: Instant,
    pub stopped: &'a Mutex<Option<String>>,
    pub closing: &'a AtomicBool,
    pub snapshot: &'a Snapshotter,
}

impl Controlled<'_> {
    fn has_stopped(&self) -> bool {
        self.stopped.lock().unwrap().is_some()
    }
}

/// Serves `listener` until the program closes.
pub fn serve(listener: UnixListener, controlled: &Controlled) {
    if listener.set_nonblocking(true).is_err() {
        return;
    }
    while !controlled.closing.load(Ordering::SeqCst) {
        match listener.accept() {
            Ok((stream, _)) => connection(stream, controlled),
            Err(error) if error.kind() == ErrorKind::WouldBlock => {
                // Until someone connects, or for a look at whether the
                // program is closing.
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
        if controlled.closing.load(Ordering::SeqCst) {
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

/// What a request for the machine gets while a restart has none in place.
const RESTARTING: &str = "the console is restarting";

fn answer(request: Request, controlled: &Controlled) -> Result<Value, String> {
    let console = controlled.console;
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
                // Across a restart the steps wait for the next machine.
                console.with(|machine| {
                    script.advance(machine, |event| match event {
                        Event::Done { text, .. } => done.push(text.to_string()),
                        Event::Screenshot(path) => done.push(crate::save_screen(
                            machine,
                            path,
                            controlled.started.elapsed(),
                        )),
                    })
                });
                let Some((line, text)) = script.waiting_at() else {
                    return Ok(json!({"done": done}));
                };
                let stuck = if controlled.has_stopped() {
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
            let screen = console
                .with(|machine| machine.screen())
                .ok_or(RESTARTING)?
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
            let screen = console
                .with(|machine| machine.screen())
                .ok_or(RESTARTING)?
                .map_err(|reason| format!("no picture: {reason}"))?;
            let frame = &screen.frame;
            if x >= frame.width || y >= frame.height {
                return Err(format!(
                    "{x},{y} is outside the {}x{} screen",
                    frame.width, frame.height
                ));
            }
            let [_, red, green, blue] = frame.pixels[(y * frame.width + x) as usize].to_be_bytes();
            Ok(json!({"rgb": format!("{red:02x}{green:02x}{blue:02x}")}))
        }
        Request::Status => {
            let stopped = controlled.stopped.lock().unwrap().clone();
            let panel = console.with(|machine| machine.panel());
            let state = match (&stopped, panel) {
                (Some(_), _) => "stopped",
                (None, Some(_)) => "running",
                (None, None) => "restarting",
            };
            Ok(json!({
                "seconds": controlled.started.elapsed().as_secs_f64(),
                "panel_on": panel.is_some_and(|panel| panel.panel_on),
                "backlight_on": panel.is_some_and(|panel| panel.backlight_on),
                "state": state,
                "stopped": stopped,
            }))
        }
        Request::Stop => {
            controlled.stop.store(true, Ordering::SeqCst);
            Ok(json!({}))
        }
        Request::SaveState { path } => {
            if controlled.has_stopped() {
                return Err("the machine has stopped".into());
            }
            let report = console
                .with(|machine| controlled.snapshot.save(machine, Path::new(&path)))
                .ok_or(RESTARTING)??;
            Ok(json!({
                "path": path,
                "guest_seconds": report.guest_time.as_secs_f64(),
                "paused_seconds": report.paused_for.as_secs_f64(),
                "ram_bytes": report.ram_bytes,
            }))
        }
    }
}
