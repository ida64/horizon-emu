//! `horizon-emu mcp`: a Model Context Protocol server on standard input and
//! output, through which an agent starts machines with the options it
//! chooses, drives them and reads what they did.
//!
//! Each run is a `horizon-emu boot` child process (Hypervisor.framework
//! gives a process one virtual machine), started with `--control` on a
//! socket in the run's own directory, `--crash-dump` into it and its
//! output going to a log file there. Its eMMC image is a copy (a clone on
//! APFS) of the one asked for, written to by the guest, so base images stay
//! as they are. Tools then reach the run through the socket (script steps,
//! screenshots, pixels, status, stop) or its files (log, crash dumps).
//!
//! Requests are served on threads of their own, so a long `drive` of one
//! run does not hold up another. Standard output carries protocol messages
//! only; when standard input closes, every run still going is stopped.
//!
//! The binary must carry the hypervisor entitlement for its runs to start:
//! run it through cargo (`cargo run --release -- mcp ...`) or through
//! `tools/hvf-run.sh target/release/horizon-emu mcp ...`.

use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{self, BufRead, BufReader, Write},
    os::unix::net::UnixStream,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use serde_json::{Value, json};

pub const USAGE: &str = "\
horizon-emu mcp [--kernel FILE] [--keys FILE] [--emmc FILE] [--runs-dir DIR]
  serve the Model Context Protocol on standard input and output: tools that
  start machines (each a `boot` child with the options asked for), drive
  them (buttons, touches, script steps, pixels, screenshots) and read
  their logs and crash dumps. The flags are defaults for start_run; runs
  live in DIR (/tmp/horizon-emu-runs unless given), one directory each";

/// Protocol version answered when the client names none.
const PROTOCOL_VERSION: &str = "2025-06-18";
/// How long a run's control socket is waited for to appear.
const SOCKET_WAIT: Duration = Duration::from_secs(60);
/// How long a run is given to stop by itself before it is killed.
const STOP_WAIT: Duration = Duration::from_secs(10);
/// Lines `read_log` returns when not told.
const LOG_LINES: usize = 200;
/// The on-screen keyboard's OK key.
const OK_KEY: (u16, u16) = (1167, 589);

/// What `start_run` uses when a call does not say.
#[derive(Default)]
struct Defaults {
    kernel: Option<PathBuf>,
    keys: Option<PathBuf>,
    emmc: Option<PathBuf>,
    runs_dir: PathBuf,
}

struct Run {
    dir: PathBuf,
    socket: PathBuf,
    log: PathBuf,
    crashes: PathBuf,
    started: Instant,
    child: Mutex<Child>,
    /// Screenshots taken, to name the next.
    screenshots: AtomicUsize,
    /// What keeping the console awake has done, for `run_status`.
    kept_awake: Mutex<Vec<String>>,
}

/// From the lock screen to the HOME menu: wait for the lock screen's home
/// icon, then the presses that wake the screen and unlock it.
const UNLOCK: &str = "wait 820,330 2d2d2d\nsleep 2\npress a 300\nsleep 2\npress a 150\n\
                      sleep 0.6\npress a 150\nsleep 0.6\npress a 150\nsleep 3";
/// How often the log is looked at for the console going to sleep.
const AWAKE_POLL: Duration = Duration::from_secs(2);

impl Run {
    fn note_awake(&self, what: String) {
        let line = format!("{:.0} s: {what}", self.started.elapsed().as_secs_f64());
        self.kept_awake.lock().unwrap().push(line);
    }

    /// Keeps the console on the HOME menu's side of sleep: unlocks it once
    /// the lock screen shows, if `unlock_first` (then calls `unlocked`),
    /// and afterwards, whenever the log says it went to sleep (the lock
    /// screen sleeps after half a minute untouched; a maintenance wake goes
    /// back to sleep with the screen off), presses power and unlocks it
    /// again. Runs until the run ends.
    fn keep_awake(&self, unlock_first: bool, unlocked: impl FnOnce()) {
        if unlock_first {
            match self.script(UNLOCK, 300.0) {
                Ok(_) => self.note_awake("unlocked".into()),
                Err(error) => self.note_awake(format!("could not unlock: {error}")),
            }
        }
        unlocked();
        let asleep = |run: &Run| {
            run.log_lines()
                .iter()
                .filter(|line| line.contains("console asleep"))
                .count()
        };
        let mut seen = asleep(self);
        while self.exited().is_none() {
            thread::sleep(AWAKE_POLL);
            if asleep(self) == seen {
                continue;
            }
            match self.script(&format!("press power 200\n{UNLOCK}"), 120.0) {
                Ok(_) => self.note_awake("woke and unlocked after sleep".into()),
                Err(error) => self.note_awake(format!("could not wake: {error}")),
            }
            seen = asleep(self);
        }
    }

    /// `None` while running, else how the process ended.
    fn exited(&self) -> Option<String> {
        let status = self.child.lock().unwrap().try_wait().ok()??;
        Some(match status.code() {
            Some(code) => format!("exited with status {code}"),
            None => "killed".into(),
        })
    }

    /// One request on the run's control socket, answered.
    fn control(&self, request: &Value, wait: Duration) -> Result<Value, String> {
        let deadline = Instant::now() + SOCKET_WAIT;
        let stream = loop {
            match UnixStream::connect(&self.socket) {
                Ok(stream) => break stream,
                Err(error) => {
                    if let Some(exited) = self.exited() {
                        return Err(format!("the run has {exited}; see its log"));
                    }
                    if Instant::now() >= deadline {
                        return Err(format!("control socket not reachable: {error}"));
                    }
                    thread::sleep(Duration::from_millis(200));
                }
            }
        };
        stream
            .set_read_timeout(Some(wait))
            .map_err(|error| error.to_string())?;
        let mut writer = stream.try_clone().map_err(|error| error.to_string())?;
        writeln!(writer, "{request}").map_err(|error| error.to_string())?;
        let mut line = String::new();
        BufReader::new(stream)
            .read_line(&mut line)
            .map_err(|error| format!("no answer: {error}"))?;
        if line.is_empty() {
            return Err(match self.exited() {
                Some(exited) => format!("the run has {exited}; see its log"),
                None => "the run closed the control connection".into(),
            });
        }
        let answer: Value =
            serde_json::from_str(&line).map_err(|error| format!("bad answer: {error}"))?;
        if answer["ok"] == true {
            Ok(answer)
        } else {
            Err(answer["error"].as_str().unwrap_or("failed").to_string())
        }
    }

    /// Runs script steps, given up on after `timeout` seconds.
    fn script(&self, text: &str, timeout: f64) -> Result<Value, String> {
        let request = json!({"op": "script", "text": text, "timeout": timeout});
        self.control(&request, Duration::from_secs_f64(timeout + 30.0))
    }

    /// The log's lines, without those of `spl`, whose requests and answers
    /// carry key material.
    fn log_lines(&self) -> Vec<String> {
        let text = fs::read(&self.log).unwrap_or_default();
        String::from_utf8_lossy(&text)
            .lines()
            .filter(|line| {
                !line
                    .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                    .any(|word| word == "spl")
            })
            .map(str::to_string)
            .collect()
    }

    fn crash_dumps(&self) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(&self.crashes)
            .map(|entries| {
                entries
                    .filter_map(Result::ok)
                    .map(|entry| entry.file_name().to_string_lossy().into_owned())
                    .collect()
            })
            .unwrap_or_default();
        names.sort();
        names
    }
}

struct Server {
    defaults: Defaults,
    runs: Mutex<BTreeMap<String, Arc<Run>>>,
    next_run: AtomicUsize,
    out: Mutex<io::Stdout>,
}

pub fn serve(flags: &[String]) -> Result<(), String> {
    let mut defaults = Defaults {
        runs_dir: PathBuf::from("/tmp/horizon-emu-runs"),
        ..Defaults::default()
    };
    let mut flags = flags.iter();
    while let Some(flag) = flags.next() {
        let mut value = || {
            flags
                .next()
                .map(PathBuf::from)
                .ok_or_else(|| format!("{flag} needs a value\n{USAGE}"))
        };
        match flag.as_str() {
            "--kernel" => defaults.kernel = Some(value()?),
            "--keys" => defaults.keys = Some(value()?),
            "--emmc" => defaults.emmc = Some(value()?),
            "--runs-dir" => defaults.runs_dir = value()?,
            _ => return Err(format!("unknown flag {flag}\n{USAGE}")),
        }
    }
    fs::create_dir_all(&defaults.runs_dir)
        .map_err(|error| format!("{}: {error}", defaults.runs_dir.display()))?;
    let server = Arc::new(Server {
        defaults,
        runs: Mutex::default(),
        next_run: AtomicUsize::new(0),
        out: Mutex::new(io::stdout()),
    });
    let mut workers = Vec::new();
    for line in io::stdin().lock().lines() {
        let Ok(line) = line else { break };
        let Ok(message) = serde_json::from_str::<Value>(&line) else {
            server.send(&json!({
                "jsonrpc": "2.0",
                "id": null,
                "error": {"code": -32700, "message": "parse error"},
            }));
            continue;
        };
        // Notifications and answers to requests of ours need no answer.
        let (Some(id), Some(method)) = (message.get("id"), message["method"].as_str()) else {
            continue;
        };
        let (id, method, params) = (id.clone(), method.to_string(), message["params"].clone());
        let server = Arc::clone(&server);
        workers.push(thread::spawn(move || {
            let answer = match server.request(&method, &params) {
                Ok(result) => json!({"jsonrpc": "2.0", "id": id, "result": result}),
                Err((code, message)) => json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": {"code": code, "message": message},
                }),
            };
            server.send(&answer);
        }));
        workers.retain(|worker| !worker.is_finished());
    }
    // Requests already received are answered before the runs are stopped.
    for worker in workers {
        let _ = worker.join();
    }
    server.stop_all();
    Ok(())
}

impl Server {
    fn send(&self, message: &Value) {
        let mut out = self.out.lock().unwrap();
        let _ = writeln!(out, "{message}");
        let _ = out.flush();
    }

    fn request(&self, method: &str, params: &Value) -> Result<Value, (i64, String)> {
        match method {
            "initialize" => Ok(json!({
                "protocolVersion": params["protocolVersion"].as_str().unwrap_or(PROTOCOL_VERSION),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "horizon-emu", "version": env!("CARGO_PKG_VERSION")},
                "instructions": INSTRUCTIONS,
            })),
            "ping" => Ok(json!({})),
            "tools/list" => Ok(json!({"tools": tools()})),
            "tools/call" => {
                let name = params["name"].as_str().unwrap_or_default();
                let arguments = &params["arguments"];
                Ok(match self.call(name, arguments) {
                    Ok(content) => json!({"content": content, "isError": false}),
                    Err(error) => {
                        json!({"content": [{"type": "text", "text": error}], "isError": true})
                    }
                })
            }
            _ => Err((-32601, format!("no method {method}"))),
        }
    }

    fn run(&self, arguments: &Value) -> Result<Arc<Run>, String> {
        let id = arguments["run"].as_str().ok_or("`run` is required")?;
        self.runs
            .lock()
            .unwrap()
            .get(id)
            .cloned()
            .ok_or_else(|| format!("no run {id}"))
    }

    fn call(&self, name: &str, arguments: &Value) -> Result<Vec<Value>, String> {
        let text = |value: Value| vec![json!({"type": "text", "text": value.to_string()})];
        let number = |key: &str| arguments[key].as_f64();
        match name {
            "start_run" => self.start_run(arguments).map(text),
            "list_runs" => {
                let runs = self.runs.lock().unwrap();
                let list: Vec<Value> = runs
                    .iter()
                    .map(|(id, run)| {
                        json!({
                            "run": id,
                            "state": run.exited().unwrap_or_else(|| "running".into()),
                            "seconds": run.started.elapsed().as_secs(),
                            "dir": run.dir,
                        })
                    })
                    .collect();
                Ok(text(json!(list)))
            }
            "run_status" => {
                let run = self.run(arguments)?;
                let lines = run.log_lines();
                let stopped: Vec<&String> = lines
                    .iter()
                    .skip_while(|line| !line.starts_with("machine stopped"))
                    .take(12)
                    .collect();
                let machine = match run.exited() {
                    Some(_) => Value::Null,
                    None => run
                        .control(&json!({"op": "status"}), Duration::from_secs(10))
                        .unwrap_or_else(|error| json!({"error": error})),
                };
                Ok(text(json!({
                    "state": run.exited().unwrap_or_else(|| "running".into()),
                    "seconds": run.started.elapsed().as_secs(),
                    "machine": machine,
                    "log_lines": lines.len(),
                    "stopped": stopped,
                    "crash_dumps": run.crash_dumps(),
                    "kept_awake": *run.kept_awake.lock().unwrap(),
                    "dir": run.dir,
                })))
            }
            "stop_run" => {
                let run = self.run(arguments)?;
                Ok(text(json!({"state": stop(&run)})))
            }
            "drive" => {
                let run = self.run(arguments)?;
                let script = arguments["script"].as_str().ok_or("`script` is required")?;
                let answer = run.script(script, number("timeout_seconds").unwrap_or(120.0))?;
                Ok(text(answer["done"].clone()))
            }
            "press" => {
                let run = self.run(arguments)?;
                let button = arguments["button"].as_str().ok_or("`button` is required")?;
                let hold = number("hold_ms").unwrap_or(150.0);
                run.script(&format!("press {button} {hold:.0}"), 30.0)?;
                Ok(text(json!("pressed")))
            }
            "touch" => {
                let run = self.run(arguments)?;
                let (x, y) = (
                    number("x").ok_or("`x` is required")?,
                    number("y").ok_or("`y` is required")?,
                );
                let hold = number("hold_ms").unwrap_or(150.0);
                run.script(&format!("touch {x:.0},{y:.0} {hold:.0}"), 30.0)?;
                Ok(text(json!("touched")))
            }
            "type_text" => {
                let run = self.run(arguments)?;
                let typed = arguments["text"].as_str().ok_or("`text` is required")?;
                if typed.contains('\n') {
                    return Err("`text` is one line: submit with `submit`".into());
                }
                let mut script = format!("type {typed}");
                if arguments["submit"] == true {
                    script.push_str(&format!("\ntouch {},{} 150\nsleep 1", OK_KEY.0, OK_KEY.1));
                }
                run.script(&script, 120.0 + typed.len() as f64)?;
                Ok(text(json!("typed")))
            }
            "wait_pixel" => {
                let run = self.run(arguments)?;
                let (x, y) = (
                    number("x").ok_or("`x` is required")?,
                    number("y").ok_or("`y` is required")?,
                );
                let rgb = arguments["rgb"].as_str().ok_or("`rgb` is required")?;
                run.script(
                    &format!("wait {x:.0},{y:.0} {rgb}"),
                    number("timeout_seconds").unwrap_or(120.0),
                )?;
                Ok(text(json!("shown")))
            }
            "pixel" => {
                let run = self.run(arguments)?;
                let coordinate = |key| number(key).map(|value| value as u32);
                let request = json!({"op": "pixel", "x": coordinate("x"), "y": coordinate("y")});
                let answer = run.control(&request, Duration::from_secs(10))?;
                Ok(text(answer["rgb"].clone()))
            }
            "screenshot" => {
                let run = self.run(arguments)?;
                let number = run.screenshots.fetch_add(1, Ordering::Relaxed);
                let path = run.dir.join(format!("screen-{number:03}.png"));
                let request = json!({"op": "screenshot", "path": path});
                let answer = run.control(&request, Duration::from_secs(30))?;
                let png =
                    fs::read(&path).map_err(|error| format!("{}: {error}", path.display()))?;
                Ok(vec![
                    json!({"type": "image", "data": base64(&png), "mimeType": "image/png"}),
                    json!({"type": "text", "text": json!({
                        "path": path,
                        "panel_on": answer["panel_on"],
                        "backlight_on": answer["backlight_on"],
                    }).to_string()}),
                ])
            }
            "save_state" => {
                let run = self.run(arguments)?;
                let name = match arguments["name"].as_str() {
                    Some(name) if !name.is_empty() && !name.contains('/') => name.to_string(),
                    Some(name) => {
                        return Err(format!("{name:?}: a snapshot name is one path component"));
                    }
                    None => format!(
                        "{}-{}",
                        arguments["run"].as_str().unwrap_or_default(),
                        std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map_or(0, |since| since.as_secs())
                    ),
                };
                let dir = self.snapshot_dir(&name);
                let request = json!({"op": "save_state", "path": dir});
                let answer = run.control(&request, Duration::from_secs(300))?;
                Ok(text(json!({
                    "snapshot": name,
                    "path": dir,
                    "guest_seconds": answer["guest_seconds"],
                    "paused_seconds": answer["paused_seconds"],
                })))
            }
            "list_snapshots" => {
                let mut names: Vec<String> = fs::read_dir(self.snapshot_dir(""))
                    .into_iter()
                    .flatten()
                    .filter_map(Result::ok)
                    .filter(|entry| entry.path().join("boot.json").exists())
                    .map(|entry| entry.file_name().to_string_lossy().into_owned())
                    .collect();
                names.sort();
                Ok(text(json!(names)))
            }
            "read_log" => {
                let run = self.run(arguments)?;
                let lines = run.log_lines();
                let since = number("since").unwrap_or(0.0) as usize;
                let most = number("max_lines").map_or(LOG_LINES, |most| most as usize);
                let wanted: Vec<&str> = arguments["contains"]
                    .as_array()
                    .map(|words| words.iter().filter_map(Value::as_str).collect())
                    .unwrap_or_default();
                let matching = lines.iter().enumerate().skip(since).filter(|(_, line)| {
                    wanted.is_empty() || wanted.iter().any(|word| line.contains(word))
                });
                let picked: Vec<(usize, &String)> = if arguments["tail"] == true {
                    let all: Vec<_> = matching.collect();
                    all[all.len().saturating_sub(most)..].to_vec()
                } else {
                    matching.take(most).collect()
                };
                let next = picked.last().map_or(since, |(index, _)| index + 1);
                let body: String = picked
                    .iter()
                    .map(|(index, line)| format!("{index}: {line}\n"))
                    .collect();
                Ok(vec![json!({
                    "type": "text",
                    "text": format!("{body}(next: {next}, lines in log: {})", lines.len()),
                })])
            }
            "crash_dumps" => {
                let run = self.run(arguments)?;
                match arguments["name"].as_str() {
                    None => Ok(text(json!(run.crash_dumps()))),
                    Some(name) if run.crash_dumps().iter().any(|dump| dump == name) => {
                        let path = run.crashes.join(name);
                        let dump = fs::read_to_string(&path)
                            .map_err(|error| format!("{}: {error}", path.display()))?;
                        Ok(vec![json!({"type": "text", "text": dump})])
                    }
                    Some(name) => Err(format!("no crash dump {name}")),
                }
            }
            _ => Err(format!("no tool {name}")),
        }
    }

    /// Where snapshot `name` lives: `snapshots/` beside the runs.
    fn snapshot_dir(&self, name: &str) -> PathBuf {
        self.defaults.runs_dir.join("snapshots").join(name)
    }

    fn start_run(&self, arguments: &Value) -> Result<Value, String> {
        let path = |key: &str, default: &Option<PathBuf>| {
            arguments[key]
                .as_str()
                .map(PathBuf::from)
                .or_else(|| default.clone())
        };
        // A snapshot by name, or a directory.
        let restore = arguments["restore"].as_str().map(|snapshot| {
            if snapshot.contains('/') {
                PathBuf::from(snapshot)
            } else {
                self.snapshot_dir(snapshot)
            }
        });
        if let Some(snapshot) = &restore
            && !snapshot.join("boot.json").exists()
        {
            return Err(format!("{}: not a snapshot", snapshot.display()));
        }
        let kernel = path("kernel", &self.defaults.kernel);
        let keys = path("keys", &self.defaults.keys);
        if restore.is_none() {
            kernel
                .as_ref()
                .ok_or("no kernel: pass `kernel` or start the server with --kernel")?;
            keys.as_ref()
                .ok_or("no keys: pass `keys` or start the server with --keys")?;
        }
        let number = self.next_run.fetch_add(1, Ordering::Relaxed);
        let id = format!("run-{number}");
        let dir = self.defaults.runs_dir.join(format!(
            "{id}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |since| since.as_secs())
        ));
        fs::create_dir_all(&dir).map_err(|error| format!("{}: {error}", dir.display()))?;
        let socket = dir.join("control.sock");
        let log = dir.join("log.txt");
        let crashes = dir.join("crashes");
        let mut args: Vec<String> = match &restore {
            Some(snapshot) => vec!["restore".into(), display(snapshot)],
            None => vec!["boot".into(), display(&kernel.clone().unwrap_or_default())],
        };
        let mut flag = |name: &str, value: String| {
            args.push(name.into());
            args.push(value);
        };
        flag("--control", display(&socket));
        flag("--crash-dump", display(&crashes));
        // What the snapshot fixes is not given to a restore.
        if restore.is_some() {
            flag("--workdir", display(&dir));
        } else {
            flag("--keys", display(&keys.clone().unwrap_or_default()));
            if let Some(emmc) = path("emmc", &self.defaults.emmc) {
                // A copy the guest writes to; on APFS a clone, made at once.
                let copy = dir.join("emmc.img");
                fs::copy(&emmc, &copy).map_err(|error| format!("{}: {error}", emmc.display()))?;
                flag("--emmc", display(&copy));
            }
            for key in [
                "sd",
                "wifi_ssid",
                "wifi_passphrase",
                "homebrew",
                "hbl",
                "homebrew_app",
                "homebrew_in",
            ] {
                if let Some(value) = arguments[key].as_str() {
                    flag(&format!("--{}", key.replace('_', "-")), value.to_string());
                }
            }
            // The Joy-Con slide onto the rails as the machine starts, so
            // that buttons work, unless asked not to.
            if arguments["joycons"] != false {
                flag("--joycons", "0".into());
            }
        }
        for key in ["dns", "symbols"] {
            if let Some(value) = arguments[key].as_str() {
                flag(&format!("--{key}"), value.to_string());
            }
        }
        if let Some(seconds) = arguments["timeout_seconds"].as_f64() {
            flag("--timeout", format!("{seconds}"));
        }
        for port in arguments["forward"].as_array().into_iter().flatten() {
            if let Some(port) = port.as_str() {
                flag("--forward", port.to_string());
            }
        }
        if restore.is_none() && (arguments["emmc"].is_string() || self.defaults.emmc.is_some()) {
            args.push("--emmc-writable".into());
        }
        if arguments["verbose"] != false {
            args.push("--verbose".into());
        }
        if arguments["window"] == true {
            args.push("--window".into());
        }
        for extra in arguments["args"].as_array().into_iter().flatten() {
            if let Some(extra) = extra.as_str() {
                args.push(extra.to_string());
            }
        }
        let output = File::create(&log).map_err(|error| format!("{}: {error}", log.display()))?;
        let errors = output.try_clone().map_err(|error| error.to_string())?;
        let program = std::env::current_exe().map_err(|error| error.to_string())?;
        let child = Command::new(program)
            .args(&args)
            .stdin(Stdio::null())
            .stdout(output)
            .stderr(errors)
            .spawn()
            .map_err(|error| format!("cannot start the run: {error}"))?;
        let run = Arc::new(Run {
            dir: dir.clone(),
            socket,
            log: log.clone(),
            crashes,
            started: Instant::now(),
            child: Mutex::new(child),
            screenshots: AtomicUsize::new(0),
            kept_awake: Mutex::default(),
        });
        self.runs
            .lock()
            .unwrap()
            .insert(id.clone(), Arc::clone(&run));
        // A restored machine answers once it is running and has had a
        // second to settle. \[observed\] A 150 ms press given as a
        // restored machine starts went unnoticed (5 tries of 5, one
        // snapshot), while the same press 0.2 s or more after the control
        // socket first answered was taken (7 of 7); why is not known.
        if restore.is_some() {
            run.script("sleep 1", SOCKET_WAIT.as_secs_f64())
                .map_err(|error| format!("the restored run did not start: {error}"))?;
        }
        if arguments["stay_awake"] == false {
            return Ok(json!({"run": id, "dir": dir, "log": log, "args": args}));
        }
        // Answered once the console is on the HOME menu; kept there.
        let keeper = Arc::clone(&run);
        let (unlocked, ready) = std::sync::mpsc::channel();
        // A restored console is where it was saved: on the HOME menu, or
        // past it, if it was kept awake.
        let unlock_first = restore.is_none();
        thread::spawn(move || {
            keeper.keep_awake(unlock_first, move || {
                let _ = unlocked.send(());
            })
        });
        let _ = ready.recv();
        let state = run.kept_awake.lock().unwrap().first().cloned();
        Ok(json!({"run": id, "dir": dir, "log": log, "args": args, "awake": state}))
    }

    fn stop_all(&self) {
        let runs: Vec<Arc<Run>> = self.runs.lock().unwrap().values().cloned().collect();
        for run in runs {
            if run.exited().is_none() {
                stop(&run);
            }
        }
    }
}

/// Stops a run: asked through its socket, killed if it does not end.
fn stop(run: &Run) -> String {
    if let Some(exited) = run.exited() {
        return exited;
    }
    let _ = run.control(&json!({"op": "stop"}), Duration::from_secs(5));
    let deadline = Instant::now() + STOP_WAIT;
    while Instant::now() < deadline {
        if let Some(exited) = run.exited() {
            return exited;
        }
        thread::sleep(Duration::from_millis(100));
    }
    let _ = run.child.lock().unwrap().kill();
    let _ = run.child.lock().unwrap().wait();
    "killed".into()
}

fn display(path: &Path) -> String {
    path.display().to_string()
}

/// Standard base64 (RFC 4648, with padding).
fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut text = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let word = chunk.iter().enumerate().fold(0u32, |word, (index, &byte)| {
            word | u32::from(byte) << (16 - 8 * index)
        });
        for index in 0..4 {
            text.push(if index <= chunk.len() {
                ALPHABET[(word >> (18 - 6 * index) & 0x3F) as usize] as char
            } else {
                '='
            });
        }
    }
    text
}

const INSTRUCTIONS: &str = "\
Runs an emulated Nintendo Switch (Horizon OS 23.5.0) per run. start_run boots \
one (about a minute to the lock screen) and returns its id; every other tool \
takes that id as `run`. The screen is 1280x720. Drive it with press (buttons: \
a b x y l r zl zr plus minus home capture up down left right, or power), touch, \
wait_pixel and drive (several script steps: wait X,Y RRGGBB / sleep S / press \
BUTTON [MS] / touch X,Y [MS] / type TEXT / joycons on|off / screenshot FILE, \
press and touch optionally `until X,Y RRGGBB` to repeat). type_text types on \
the on-screen keyboard once it is open. start_run answers once the console is \
on the HOME menu, unlocked, and the server keeps it awake: if it ever goes to \
sleep it is woken and unlocked again (run_status lists this as kept_awake; \
pass stay_awake false to drive the lock screen yourself). screenshot \
shows the screen. read_log pages the emulator's narration (since/next \
cursors, contains filters, tail); crash_dumps lists and reads the full \
register and module dump of each process crash. stop_run ends a run. \
save_state snapshots a running machine (it stands still for a moment) \
under a name; start_run with restore set to that name starts a new run \
from it in a few seconds instead of booting, as many times as wanted. \
Network connections of the saved machine are gone after a restore; \
the console makes new ones.";

fn tools() -> Value {
    let run = json!({"type": "string", "description": "run id from start_run"});
    let number = |description: &str| json!({"type": "number", "description": description});
    let string = |description: &str| json!({"type": "string", "description": description});
    json!([
        {
            "name": "start_run",
            "description": "Boot a new machine, or with `restore` go on from a snapshot. Returns its run id, directory and log path. The eMMC image is copied first; the original is not written.",
            "inputSchema": {"type": "object", "properties": {
                "restore": string("snapshot name from save_state (or a snapshot directory): start from it instead of booting. The snapshot fixes kernel, keys, images, WiFi network, homebrew and Joy-Con; dns, forward, symbols, timeout_seconds, window, verbose, stay_awake and args still apply"),
                "kernel": string("kernel ELF (default: the server's --kernel)"),
                "keys": string("key file (default: the server's --keys)"),
                "emmc": string("raw eMMC user image to copy and boot (default: the server's --emmc)"),
                "sd": string("microSD image"),
                "wifi_ssid": string("an open access point of this name in range, with DHCP/DNS/NAT to the host"),
                "wifi_passphrase": string("makes it WPA2 with this passphrase"),
                "dns": string("resolver ADDRESS[:PORT] the router's DNS asks instead of the host's (captive portals)"),
                "forward": {"type": "array", "items": {"type": "string"}, "description": "HOST_PORT[:CONSOLE_PORT] inbound TCP forwards"},
                "homebrew": string("hbmenu NRO to run as the Album applet"),
                "hbl": string("hbloader NSP"),
                "homebrew_app": string("NRO to put on the SD card for hbmenu"),
                "homebrew_in": string("program ID whose application slot runs the homebrew"),
                "symbols": string("symbol file"),
                "joycons": {"type": "boolean", "description": "Joy-Con on the rails from the start, so buttons work (default true)"},
                "timeout_seconds": number("stop the machine after this long (default: never)"),
                "window": {"type": "boolean", "description": "also show the screen in a window on this Mac"},
                "verbose": {"type": "boolean", "description": "narrate the boot in the log (default true)"},
                "stay_awake": {"type": "boolean", "description": "unlock the console and keep it awake, waking and unlocking it if it sleeps; start_run then answers once the HOME menu shows (default true)"},
                "args": {"type": "array", "items": {"type": "string"}, "description": "further `horizon-emu boot` flags"}
            }}
        },
        {"name": "list_runs", "description": "Every run with its state.", "inputSchema": {"type": "object", "properties": {}}},
        {
            "name": "run_status",
            "description": "State, uptime, screen, why it stopped, crash dumps.",
            "inputSchema": {"type": "object", "properties": {"run": run}, "required": ["run"]}
        },
        {
            "name": "stop_run",
            "description": "Stop a run (killed if it does not stop within 10 s).",
            "inputSchema": {"type": "object", "properties": {"run": run}, "required": ["run"]}
        },
        {
            "name": "drive",
            "description": "Do script steps in order, one per line: wait X,Y RRGGBB | sleep SECONDS | press BUTTON [HOLD_MS] [until X,Y RRGGBB [SECONDS]] | touch X,Y [HOLD_MS] [until ...] | type TEXT (on the open on-screen keyboard; the rest of the line) | joycons on|off | screenshot FILE. Returns the steps done; fails naming the step it did not get past.",
            "inputSchema": {"type": "object", "properties": {
                "run": run,
                "script": string("the steps, one per line"),
                "timeout_seconds": number("give up after this long (default 120)")
            }, "required": ["run", "script"]}
        },
        {
            "name": "press",
            "description": "Press a button: a b x y l r zl zr plus minus home capture up down left right, or power.",
            "inputSchema": {"type": "object", "properties": {
                "run": run, "button": string("button name"), "hold_ms": number("how long (default 150)")
            }, "required": ["run", "button"]}
        },
        {
            "name": "touch",
            "description": "Touch the 1280x720 screen at a point.",
            "inputSchema": {"type": "object", "properties": {
                "run": run, "x": number("0-1279"), "y": number("0-719"), "hold_ms": number("how long (default 150)")
            }, "required": ["run", "x", "y"]}
        },
        {
            "name": "type_text",
            "description": "Type on the on-screen keyboard (it must be open, on its letters page): letters, digits, space and ASCII punctuation, by touching its keys.",
            "inputSchema": {"type": "object", "properties": {
                "run": run,
                "text": string("one line of text"),
                "submit": {"type": "boolean", "description": "then touch OK"}
            }, "required": ["run", "text"]}
        },
        {
            "name": "wait_pixel",
            "description": "Wait until a point of the screen has a colour (each channel within 16).",
            "inputSchema": {"type": "object", "properties": {
                "run": run, "x": number("0-1279"), "y": number("0-719"),
                "rgb": string("RRGGBB"), "timeout_seconds": number("default 120")
            }, "required": ["run", "x", "y", "rgb"]}
        },
        {
            "name": "pixel",
            "description": "The colour at a point of the screen, RRGGBB.",
            "inputSchema": {"type": "object", "properties": {
                "run": run, "x": number("0-1279"), "y": number("0-719")
            }, "required": ["run", "x", "y"]}
        },
        {
            "name": "screenshot",
            "description": "What the screen shows, as a PNG image (also saved in the run's directory).",
            "inputSchema": {"type": "object", "properties": {"run": run}, "required": ["run"]}
        },
        {
            "name": "save_state",
            "description": "Snapshot the running machine (memory, processors, every device) for start_run's `restore`. Returns the snapshot name and how long the machine stood still. Refused while the console sleeps.",
            "inputSchema": {"type": "object", "properties": {
                "run": run,
                "name": string("snapshot name, one path component; an existing snapshot of that name is replaced (default: run id and time)")
            }, "required": ["run"]}
        },
        {"name": "list_snapshots", "description": "Names of the snapshots start_run can restore.", "inputSchema": {"type": "object", "properties": {}}},
        {
            "name": "read_log",
            "description": "Lines of the run's log (boot narration, programs loaded, network flows, crashes, fatal errors), numbered. Lines of spl are left out: they carry key material.",
            "inputSchema": {"type": "object", "properties": {
                "run": run,
                "since": number("first line number (default 0); pass the previous call's `next` to follow"),
                "max_lines": number("at most this many (default 200)"),
                "contains": {"type": "array", "items": {"type": "string"}, "description": "only lines containing any of these"},
                "tail": {"type": "boolean", "description": "the last max_lines matching lines instead of the first"}
            }, "required": ["run"]}
        },
        {
            "name": "crash_dumps",
            "description": "Without `name`: the run's crash dump files. With it: that dump (fault, PC/LR/SP, X0-X30, call chain and modules with build IDs, every code address as module+offset).",
            "inputSchema": {"type": "object", "properties": {
                "run": run, "name": string("file name from the list")
            }, "required": ["run"]}
        }
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_matches_rfc_4648_vectors() {
        for (plain, encoded) in [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("fooba", "Zm9vYmE="),
            ("foobar", "Zm9vYmFy"),
        ] {
            assert_eq!(base64(plain.as_bytes()), encoded);
        }
    }
}
