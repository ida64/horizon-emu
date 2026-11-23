//! A running machine: a `horizon-emu` process the UI started, what it
//! prints, and its control socket (`--control`, see the emulator's
//! `control` module), through which the UI reads its status and frames
//! and sends input, each on a connection of its own.

use std::{
    collections::VecDeque,
    io::{self, BufRead, BufReader, Read, Write},
    os::unix::net::UnixStream,
    path::{Path, PathBuf},
    process::{Command, ExitStatus, Stdio},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

use serde_json::{Value, json};

/// Lines of output kept per machine.
const LOG_KEPT: usize = 50_000;
/// How often the status is asked for.
const STATUS_EVERY: Duration = Duration::from_millis(250);
/// Frames asked for at most this often.
const FRAME_EVERY: Duration = Duration::from_millis(16);
/// How long a stopped machine is given to end before it is killed.
const STOP_GRACE: Duration = Duration::from_secs(15);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stream {
    Out,
    Err,
    /// The UI's own remarks.
    Ui,
}

#[derive(Debug, Clone)]
pub struct Line {
    pub stream: Stream,
    pub text: String,
}

/// What the process printed, oldest first, with a count of lines ever
/// printed so that views can tell what is new.
#[derive(Default)]
pub struct Log {
    pub lines: VecDeque<Line>,
    pub total: u64,
}

impl Log {
    pub fn push(&mut self, stream: Stream, text: String) {
        if self.lines.len() == LOG_KEPT {
            self.lines.pop_front();
        }
        self.lines.push_back(Line { stream, text });
        self.total += 1;
    }
}

/// The control socket's `status` answer.
#[derive(Debug, Clone, Default)]
pub struct Status {
    pub state: String,
    pub guest_seconds: Option<f64>,
    pub panel_on: bool,
    pub backlight_on: bool,
    pub joycons_seated: bool,
    pub cpu_paced: bool,
}

impl Status {
    fn from(value: &Value) -> Self {
        Self {
            state: value["state"].as_str().unwrap_or_default().to_string(),
            guest_seconds: value["guest_seconds"].as_f64(),
            panel_on: value["panel_on"].as_bool().unwrap_or(false),
            backlight_on: value["backlight_on"].as_bool().unwrap_or(false),
            joycons_seated: value["joycons_seated"].as_bool().unwrap_or(false),
            cpu_paced: value["cpu_paced"].as_bool().unwrap_or(false),
        }
    }
}

/// The last frame the display controller sent.
#[derive(Default)]
pub struct Frame {
    pub width: usize,
    pub height: usize,
    /// Red, green, blue, 0xFF per pixel.
    pub rgba: Vec<u8>,
    pub panel_on: bool,
    pub backlight_on: bool,
    /// Why there is none, when there is none.
    pub reason: Option<String>,
    /// Counts frames received, so that the view uploads each once.
    pub serial: u64,
}

/// One connection to the control socket.
struct Control {
    reader: BufReader<UnixStream>,
    writer: UnixStream,
}

impl Control {
    fn connect(path: &Path) -> io::Result<Self> {
        let stream = UnixStream::connect(path)?;
        Ok(Self {
            writer: stream.try_clone()?,
            reader: BufReader::new(stream),
        })
    }

    /// Sends `request`, returns the answer's fields; an answer with `ok`
    /// false is an error with its message.
    fn request(&mut self, request: &Value) -> io::Result<Result<Value, String>> {
        writeln!(self.writer, "{request}")?;
        let mut line = String::new();
        if self.reader.read_line(&mut line)? == 0 {
            return Err(io::ErrorKind::UnexpectedEof.into());
        }
        let answer: Value = serde_json::from_str(&line).map_err(io::Error::other)?;
        Ok(if answer["ok"].as_bool() == Some(true) {
            Ok(answer)
        } else {
            Err(answer["error"].as_str().unwrap_or("failed").to_string())
        })
    }

    fn frame(&mut self, into: &mut Frame) -> io::Result<()> {
        let answer = match self.request(&json!({"op": "frame"}))? {
            Ok(answer) => answer,
            Err(error) => {
                into.reason = Some(error);
                into.serial += 1;
                return Ok(());
            }
        };
        let bytes = answer["bytes"].as_u64().unwrap_or(0) as usize;
        let mut rgba = std::mem::take(&mut into.rgba);
        rgba.resize(bytes, 0);
        self.reader.read_exact(&mut rgba)?;
        into.rgba = rgba;
        into.width = answer["width"].as_u64().unwrap_or(0) as usize;
        into.height = answer["height"].as_u64().unwrap_or(0) as usize;
        into.panel_on = answer["panel_on"].as_bool().unwrap_or(false);
        into.backlight_on = answer["backlight_on"].as_bool().unwrap_or(false);
        into.reason = answer["reason"].as_str().map(str::to_string);
        into.serial += 1;
        Ok(())
    }
}

/// Whether the executable carries the hypervisor entitlement, which
/// Hypervisor.framework requires (the emulator's `tools/hvf-run.sh` adds it
/// to what cargo builds); ad-hoc signs it with the entitlement otherwise.
pub fn ensure_entitlement(executable: &Path) -> Result<(), String> {
    const ENTITLEMENTS: &str = include_str!("../../../tools/hypervisor.entitlements");
    let shown = Command::new("codesign")
        .args(["-d", "--entitlements", "-"])
        .arg(executable)
        .output()
        .map_err(|error| format!("codesign: {error}"))?;
    let text = String::from_utf8_lossy(&shown.stdout) + String::from_utf8_lossy(&shown.stderr);
    if text.contains("com.apple.security.hypervisor") {
        return Ok(());
    }
    let file = std::env::temp_dir().join(format!(
        "horizon-emu-ui-{}.entitlements",
        std::process::id()
    ));
    std::fs::write(&file, ENTITLEMENTS).map_err(|error| format!("{}: {error}", file.display()))?;
    let signed = Command::new("codesign")
        .args(["--sign", "-", "--force", "--entitlements"])
        .arg(&file)
        .arg(executable)
        .output()
        .map_err(|error| format!("codesign: {error}"))?;
    let _ = std::fs::remove_file(&file);
    if !signed.status.success() {
        return Err(format!(
            "codesign {}: {}",
            executable.display(),
            String::from_utf8_lossy(&signed.stderr).trim()
        ));
    }
    Ok(())
}

/// The `horizon-emu` executable beside this one.
pub fn emulator() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|exe| Some(exe.parent()?.join("horizon-emu")))
        .unwrap_or_else(|| PathBuf::from("horizon-emu"))
}

/// Reads `source` line by line into `log`.
fn capture(
    source: impl Read + Send + 'static,
    stream: Stream,
    log: Arc<Mutex<Log>>,
    wake: impl Fn() + Send + 'static,
) {
    thread::spawn(move || {
        for line in BufReader::new(source).lines() {
            let Ok(line) = line else { break };
            log.lock().unwrap().push(stream, line);
            wake();
        }
    });
}

/// What the UI does to a running machine.
pub enum Input {
    Press { button: &'static str, held: bool },
    Touch(Option<(u16, u16)>),
    Joycons(bool),
    Amiibo(Option<(PathBuf, bool)>),
}

impl Input {
    fn request(&self) -> Value {
        match self {
            Self::Press { button, held } => json!({"op": "press", "button": button, "held": held}),
            Self::Touch(Some((x, y))) => json!({"op": "touch", "x": x, "y": y}),
            Self::Touch(None) => json!({"op": "touch"}),
            Self::Joycons(seated) => json!({"op": "joycons", "seated": seated}),
            Self::Amiibo(Some((path, writable))) => {
                json!({"op": "amiibo", "path": path, "writable": writable})
            }
            Self::Amiibo(None) => json!({"op": "amiibo"}),
        }
    }
}

/// A process the UI started and its connections.
pub struct Child {
    pub socket: PathBuf,
    pub log: Arc<Mutex<Log>>,
    pub status: Arc<Mutex<Status>>,
    pub frame: Arc<Mutex<Frame>>,
    pub exited: Arc<Mutex<Option<ExitStatus>>>,
    /// Whether frames are wanted now (the screen is shown).
    pub streaming: Arc<AtomicBool>,
    /// Since the process was asked to stop.
    stop_asked: Mutex<Option<Instant>>,
    process: Arc<Mutex<std::process::Child>>,
    inputs: mpsc::Sender<Input>,
    /// The process has ended; the threads wind down.
    ended: Arc<AtomicBool>,
}

impl Child {
    /// Starts `executable` with `arguments`, its control socket at
    /// `socket`; `wake` asks the UI to repaint.
    pub fn spawn(
        executable: &Path,
        arguments: &[String],
        socket: PathBuf,
        wake: impl Fn() + Clone + Send + Sync + 'static,
    ) -> Result<Self, String> {
        let _ = std::fs::remove_file(&socket);
        let log = Arc::new(Mutex::new(Log::default()));
        log.lock().unwrap().push(
            Stream::Ui,
            format!("$ {} {}", executable.display(), arguments.join(" ")),
        );
        let mut process = Command::new(executable)
            .args(arguments)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|error| format!("{}: {error}", executable.display()))?;
        capture(
            process.stdout.take().expect("piped"),
            Stream::Out,
            Arc::clone(&log),
            wake.clone(),
        );
        capture(
            process.stderr.take().expect("piped"),
            Stream::Err,
            Arc::clone(&log),
            wake.clone(),
        );
        let (inputs, received) = mpsc::channel::<Input>();
        let child = Self {
            socket,
            log,
            status: Arc::new(Mutex::new(Status {
                state: "starting".into(),
                ..Status::default()
            })),
            frame: Arc::new(Mutex::new(Frame::default())),
            exited: Arc::new(Mutex::new(None)),
            streaming: Arc::new(AtomicBool::new(false)),
            stop_asked: Mutex::new(None),
            process: Arc::new(Mutex::new(process)),
            inputs,
            ended: Arc::new(AtomicBool::new(false)),
        };
        child.watch(wake.clone());
        child.stream_frames(wake);
        child.send_inputs(received);
        Ok(child)
    }

    /// A connection once the socket accepts one, or none once the process
    /// has ended.
    fn connect(socket: &Path, ended: &AtomicBool) -> Option<Control> {
        loop {
            if ended.load(Ordering::SeqCst) {
                return None;
            }
            if let Ok(control) = Control::connect(socket) {
                return Some(control);
            }
            thread::sleep(Duration::from_millis(100));
        }
    }

    /// Polls the status and the process.
    fn watch(&self, wake: impl Fn() + Send + 'static) {
        let (socket, status, exited, process, ended, log) = (
            self.socket.clone(),
            Arc::clone(&self.status),
            Arc::clone(&self.exited),
            Arc::clone(&self.process),
            Arc::clone(&self.ended),
            Arc::clone(&self.log),
        );
        thread::spawn(move || {
            let mut control: Option<Control> = None;
            loop {
                if let Ok(Some(code)) = process.lock().unwrap().try_wait() {
                    *exited.lock().unwrap() = Some(code);
                    ended.store(true, Ordering::SeqCst);
                    let mut status = status.lock().unwrap();
                    status.state = "stopped".into();
                    log.lock()
                        .unwrap()
                        .push(Stream::Ui, format!("process ended: {code}"));
                    wake();
                    return;
                }
                if control.is_none() {
                    control = Control::connect(&socket).ok();
                }
                if let Some(connection) = &mut control {
                    match connection.request(&json!({"op": "status"})) {
                        Ok(Ok(answer)) => {
                            *status.lock().unwrap() = Status::from(&answer);
                            wake();
                        }
                        Ok(Err(_)) => {}
                        Err(_) => control = None,
                    }
                }
                thread::sleep(STATUS_EVERY);
            }
        });
    }

    fn stream_frames(&self, wake: impl Fn() + Send + 'static) {
        let (socket, frame, streaming, ended) = (
            self.socket.clone(),
            Arc::clone(&self.frame),
            Arc::clone(&self.streaming),
            Arc::clone(&self.ended),
        );
        thread::spawn(move || {
            let mut control: Option<Control> = None;
            // Filled outside the lock, then swapped in.
            let mut next = Frame::default();
            while !ended.load(Ordering::SeqCst) {
                if !streaming.load(Ordering::SeqCst) {
                    thread::sleep(Duration::from_millis(50));
                    continue;
                }
                let asked = Instant::now();
                if control.is_none() {
                    control = Self::connect(&socket, &ended);
                }
                let Some(connection) = &mut control else {
                    return;
                };
                next.serial = frame.lock().unwrap().serial;
                if connection.frame(&mut next).is_err() {
                    control = None;
                    thread::sleep(Duration::from_millis(100));
                    continue;
                }
                std::mem::swap(&mut *frame.lock().unwrap(), &mut next);
                wake();
                thread::sleep(FRAME_EVERY.saturating_sub(asked.elapsed()));
            }
        });
    }

    fn send_inputs(&self, received: mpsc::Receiver<Input>) {
        let (socket, ended, log) = (
            self.socket.clone(),
            Arc::clone(&self.ended),
            Arc::clone(&self.log),
        );
        thread::spawn(move || {
            let mut control: Option<Control> = None;
            for input in received {
                if control.is_none() {
                    control = Self::connect(&socket, &ended);
                }
                let Some(connection) = &mut control else {
                    return;
                };
                match connection.request(&input.request()) {
                    Ok(Ok(_)) => {}
                    Ok(Err(error)) => log
                        .lock()
                        .unwrap()
                        .push(Stream::Ui, format!("input: {error}")),
                    Err(_) => control = None,
                }
            }
        });
    }

    pub fn input(&self, input: Input) {
        let _ = self.inputs.send(input);
    }

    /// Sends `request` on a connection of its own and hands the answer to
    /// `done` on another thread.
    pub fn request(
        &self,
        request: Value,
        done: impl FnOnce(Result<Value, String>) + Send + 'static,
    ) {
        let socket = self.socket.clone();
        thread::spawn(move || {
            let answer = Control::connect(&socket)
                .map_err(|error| format!("control socket: {error}"))
                .and_then(|mut control| {
                    control
                        .request(&request)
                        .map_err(|error| format!("control socket: {error}"))?
                });
            done(answer);
        });
    }

    pub fn has_ended(&self) -> bool {
        self.ended.load(Ordering::SeqCst)
    }

    /// Asks the machine to stop; the process ends once it has. One that
    /// was asked [`STOP_GRACE`] ago is killed.
    pub fn stop(&self) {
        let mut asked = self.stop_asked.lock().unwrap();
        match *asked {
            Some(at) if at.elapsed() >= STOP_GRACE => self.kill(),
            Some(_) => {}
            None => {
                *asked = Some(Instant::now());
                self.request(json!({"op": "stop"}), |_| {});
            }
        }
    }

    pub fn stopping(&self) -> bool {
        self.stop_asked.lock().unwrap().is_some() && !self.has_ended()
    }

    pub fn kill(&self) {
        let _ = self.process.lock().unwrap().kill();
    }
}

impl Drop for Child {
    fn drop(&mut self) {
        if !self.has_ended() {
            self.kill();
        }
        let _ = std::fs::remove_file(&self.socket);
    }
}

/// A command the UI runs to completion in the background (formatting an
/// image, downloading firmware), with its output.
pub struct Job {
    pub log: Arc<Mutex<Log>>,
    pub result: Arc<Mutex<Option<Result<(), String>>>>,
}

impl Job {
    /// Runs the commands one after the other, stopping at the first that
    /// fails. `between` runs after each with its output, and may stop the
    /// rest with an error or change the next command's arguments.
    pub fn run(
        steps: Vec<(PathBuf, Vec<String>)>,
        between: impl Fn(usize, &[String], &mut [(PathBuf, Vec<String>)]) -> Result<(), String>
        + Send
        + 'static,
        wake: impl Fn() + Clone + Send + Sync + 'static,
    ) -> Self {
        let log = Arc::new(Mutex::new(Log::default()));
        let result = Arc::new(Mutex::new(None));
        let (job_log, job_result) = (Arc::clone(&log), Arc::clone(&result));
        thread::spawn(move || {
            let mut steps = steps;
            let outcome = (|| {
                for index in 0..steps.len() {
                    let (executable, arguments) = steps[index].clone();
                    job_log.lock().unwrap().push(
                        Stream::Ui,
                        format!("$ {} {}", executable.display(), arguments.join(" ")),
                    );
                    let mut process = Command::new(&executable)
                        .args(&arguments)
                        .stdin(Stdio::null())
                        .stdout(Stdio::piped())
                        .stderr(Stdio::piped())
                        .spawn()
                        .map_err(|error| format!("{}: {error}", executable.display()))?;
                    let first = job_log.lock().unwrap().total;
                    let (out, err) = (
                        process.stdout.take().unwrap(),
                        process.stderr.take().unwrap(),
                    );
                    let err_log = Arc::clone(&job_log);
                    let err_wake = wake.clone();
                    let errors = thread::spawn(move || {
                        for line in BufReader::new(err).lines().map_while(Result::ok) {
                            err_log.lock().unwrap().push(Stream::Err, line);
                            err_wake();
                        }
                    });
                    for line in BufReader::new(out).lines().map_while(Result::ok) {
                        job_log.lock().unwrap().push(Stream::Out, line);
                        wake();
                    }
                    let _ = errors.join();
                    let status = process.wait().map_err(|error| error.to_string())?;
                    let printed: Vec<String> = {
                        let log = job_log.lock().unwrap();
                        let skip = log.lines.len().saturating_sub((log.total - first) as usize);
                        log.lines
                            .iter()
                            .skip(skip)
                            .map(|line| line.text.clone())
                            .collect()
                    };
                    if !status.success() {
                        return Err(printed
                            .iter()
                            .rev()
                            .find(|line| !line.trim().is_empty())
                            .cloned()
                            .unwrap_or_else(|| {
                                format!("{} failed: {status}", executable.display())
                            }));
                    }
                    between(index, &printed, &mut steps[index + 1..])?;
                }
                Ok(())
            })();
            *job_result.lock().unwrap() = Some(outcome);
            wake();
        });
        Self { log, result }
    }

    pub fn finished(&self) -> Option<Result<(), String>> {
        self.result.lock().unwrap().clone()
    }
}
