//! Saving a running machine and building one back from what was saved.
//!
//! A snapshot is a directory:
//!
//! * `state.bin`: [`SavedMachine`], bincode: guest time, each core's
//!   registers, and every part's own state by name.
//! * `ram.bin`: DRAM, one byte of file per byte of DRAM, written sparse:
//!   64 KiB blocks of zeros are left as holes.
//!
//! A machine is restored by building it again from the same boot
//! parameters, kernel, keys and storage images, then loading the saved
//! state over it before it first runs; so what wires the parts together is
//! never saved, only what they hold. A part that cannot save its state
//! makes the snapshot fail rather than leave a half-restored console.

use std::{
    collections::BTreeSet,
    fs::{self, File},
    io,
    os::{fd::AsRawFd, unix::fs::FileExt},
    path::Path,
    sync::atomic::Ordering,
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};

use super::{CORE_COUNT, Machine, ParkedCore};
use crate::snapshot::{Snapshot, SnapshotError, decode, encode};

/// Bumped whenever [`SavedMachine`] or any part's saved form changes in a
/// way an older snapshot would not decode into.
const FORMAT_VERSION: u32 = 9;

/// Granule of the sparse RAM file.
const RAM_BLOCK: usize = 64 * 1024;

/// How long the cores get to reach a point where they can be parked.
const PARK_TIMEOUT: Duration = Duration::from_secs(5);

/// How often cores that have not parked yet are kicked again.
const KICK_INTERVAL: Duration = Duration::from_millis(10);

#[derive(Serialize, Deserialize)]
struct SavedMachine {
    version: u32,
    dram_bytes: u64,
    /// Guest time when the machine was paused.
    clock: Duration,
    cores: [Option<ParkedCore>; CORE_COUNT],
    /// Each part's state, by name, in the order they were saved.
    parts: Vec<(String, Vec<u8>)>,
}

/// What [`Machine::save_state`] did.
#[derive(Debug, Clone, Copy)]
pub struct SaveReport {
    /// Guest time at the snapshot.
    pub guest_time: Duration,
    /// Host time the machine stood still.
    pub paused_for: Duration,
    /// Bytes of DRAM written (not zero).
    pub ram_bytes: u64,
}

fn io_error(what: &str, path: &Path, error: io::Error) -> SnapshotError {
    SnapshotError::Io(format!("{what} {}: {error}", path.display()))
}

impl Machine {
    /// Saves the running machine into `dir`, which is created if missing:
    /// stops guest time, parks every core, lets each part finish what it
    /// is doing, writes it all, calls `beside` (for the caller's own files,
    /// such as copies of writable storage images, which have to be taken
    /// while nothing writes them), then lets the machine go on.
    ///
    /// Refused while a core is switched off or the console sleeps: the
    /// state of a core waiting there lives in its host thread, not in
    /// anything a restored machine can be given.
    pub fn save_state(
        &self,
        dir: &Path,
        beside: &mut dyn FnMut(&Path) -> io::Result<()>,
    ) -> Result<SaveReport, SnapshotError> {
        fs::create_dir_all(dir).map_err(|error| io_error("creating", dir, error))?;
        let paused = Instant::now();
        self.clock.pause();
        let result = self.park_cores().and_then(|cores| {
            let ram_bytes = self.write_state(dir, cores)?;
            beside(dir).map_err(|error| io_error("saving beside", dir, error))?;
            Ok(ram_bytes)
        });
        self.release_cores();
        let guest_time = self.clock.now();
        self.clock.resume();
        result.map(|ram_bytes| SaveReport {
            guest_time,
            paused_for: paused.elapsed(),
            ram_bytes,
        })
    }

    /// Loads the state saved in `dir` into this machine, freshly built from
    /// what the saved one was built from and not run yet. Its next
    /// [`Machine::run`] resumes the cores where they were.
    pub fn restore_state(&self, dir: &Path) -> Result<(), SnapshotError> {
        let path = dir.join("state.bin");
        let bytes = fs::read(&path).map_err(|error| io_error("reading", &path, error))?;
        let saved: SavedMachine = decode("machine", &bytes)?;
        if saved.version != FORMAT_VERSION {
            return Err(SnapshotError::Corrupt(format!(
                "format version {} where this build reads {FORMAT_VERSION}",
                saved.version
            )));
        }
        if saved.dram_bytes != self.ram.size() as u64 {
            return Err(SnapshotError::Corrupt(format!(
                "{:#x} bytes of DRAM saved, {:#x} here",
                saved.dram_bytes,
                self.ram.size()
            )));
        }
        if self.restored.lock().unwrap().is_some()
            || self.pause.state.lock().unwrap().threads != [false; CORE_COUNT]
        {
            return Err(SnapshotError::Unsupported(
                "restoring into a machine that has run".into(),
            ));
        }
        let parts = self.parts()?;
        let here: BTreeSet<&str> = parts.iter().map(|(name, _)| name.as_str()).collect();
        let there: BTreeSet<&str> = saved.parts.iter().map(|(name, _)| name.as_str()).collect();
        if here != there || here.len() != parts.len() || there.len() != saved.parts.len() {
            let missing: Vec<_> = there.difference(&here).collect();
            let new: Vec<_> = here.difference(&there).collect();
            return Err(SnapshotError::Corrupt(format!(
                "parts differ: saved but not here {missing:?}, here but not saved {new:?}"
            )));
        }
        self.clock.pause();
        self.clock.set(saved.clock);
        self.read_ram(dir)?;
        for (name, bytes) in &saved.parts {
            let (_, part) = parts
                .iter()
                .find(|(here, _)| here == name)
                .expect("the names were matched above");
            part.restore(bytes)?;
        }
        *self.restored.lock().unwrap() = Some(saved.cores);
        Ok(())
    }

    /// Every part with state of its own, by a name unique in the machine.
    fn parts(&self) -> Result<Vec<(String, &dyn Snapshot)>, SnapshotError> {
        let mut parts: Vec<(String, &dyn Snapshot)> = vec![
            ("gic".into(), &*self.gic),
            ("timers".into(), &self.timers),
            ("secure monitor".into(), &self.secmon),
        ];
        for (core, pmu) in self.pmus.iter().enumerate() {
            parts.push((format!("pmu{core}"), pmu));
        }
        if let Some(bpmp) = &self.bpmp {
            parts.push(("bpmp".into(), bpmp));
            // The kernel the cold boot took from package2, once it has.
            parts.push(("kernel".into(), &self.image));
        }
        for (name, device) in self.bus.devices() {
            let Some(part) = device.snapshot() else {
                return Err(SnapshotError::Unsupported(format!("device {name}")));
            };
            parts.push((name, part));
        }
        parts.extend(
            self.soc
                .snapshot_parts()
                .into_iter()
                .map(|(name, part)| (name.to_string(), part)),
        );
        Ok(parts)
    }

    /// Asks every core with a thread to park and waits until they have.
    fn park_cores(&self) -> Result<[Option<ParkedCore>; CORE_COUNT], SnapshotError> {
        let mut state = self.pause.state.lock().unwrap();
        if self.pause.requested() {
            return Err(SnapshotError::Unsupported(
                "a snapshot is being taken already".into(),
            ));
        }
        // A machine powered on may have only its BPMP running (before core
        // 0 is released, or in SC7), whose state its part saves.
        if state.threads == [false; CORE_COUNT] && self.bpmp.is_none() {
            return Err(SnapshotError::Unsupported(
                "a machine that is not running".into(),
            ));
        }
        self.pause.requested.store(true, Ordering::SeqCst);
        let deadline = Instant::now() + PARK_TIMEOUT;
        loop {
            let threads = state.threads;
            let waiting: Vec<usize> = (0..CORE_COUNT)
                .filter(|&core| threads[core] && state.parked[core].is_none())
                .collect();
            if waiting.is_empty() {
                break;
            }
            if Instant::now() >= deadline {
                return Err(SnapshotError::Unsupported(format!(
                    "cores {waiting:?} did not reach a point to park at within {PARK_TIMEOUT:?}"
                )));
            }
            drop(state);
            // Wakes a core out of WFI and out of the guest; one that was
            // about to enter it is kicked again on the next round.
            waiting.iter().for_each(|&core| self.signals.notify(core));
            state = self.pause.state.lock().unwrap();
            state = self
                .pause
                .changed
                .wait_timeout(state, KICK_INTERVAL)
                .unwrap()
                .0;
        }
        for (core, parked) in state.parked.iter().enumerate() {
            match parked {
                Some(ParkedCore::Off) => {
                    return Err(SnapshotError::Unsupported(format!(
                        "core {core} switched off"
                    )));
                }
                Some(ParkedCore::Asleep) => {
                    return Err(SnapshotError::Unsupported("the console asleep".into()));
                }
                _ => {}
            }
        }
        Ok(state.parked.clone())
    }

    fn release_cores(&self) {
        let _state = self.pause.state.lock().unwrap();
        self.pause.requested.store(false, Ordering::SeqCst);
        self.pause.changed.notify_all();
    }

    /// Lets the parts settle, then writes `state.bin` and `ram.bin`; the
    /// bytes of DRAM written.
    fn write_state(
        &self,
        dir: &Path,
        cores: [Option<ParkedCore>; CORE_COUNT],
    ) -> Result<u64, SnapshotError> {
        let parts = self.parts()?;
        // Settling may write DRAM (the GPU writes back what it rendered),
        // so DRAM is saved after every part has.
        for (_, part) in &parts {
            part.quiesce()?;
        }
        let saved = SavedMachine {
            version: FORMAT_VERSION,
            dram_bytes: self.ram.size() as u64,
            clock: self.clock.now(),
            cores,
            parts: parts
                .iter()
                .map(|(name, part)| Ok((name.clone(), part.save()?)))
                .collect::<Result<_, SnapshotError>>()?,
        };
        let path = dir.join("state.bin");
        fs::write(&path, encode(&saved)?).map_err(|error| io_error("writing", &path, error))?;
        self.write_ram(dir)
    }

    fn write_ram(&self, dir: &Path) -> Result<u64, SnapshotError> {
        let path = dir.join("ram.bin");
        let failed = |error| io_error("writing", &path, error);
        // A new file, so that blocks of zeros are holes.
        let _ = fs::remove_file(&path);
        let file = File::create(&path).map_err(failed)?;
        file.set_len(self.ram.size() as u64).map_err(failed)?;
        // SAFETY: the mapping lives as long as `self.ram`; every core is
        // parked and every part has settled, so nothing writes it now.
        let ram = unsafe { std::slice::from_raw_parts(self.ram.host_ptr(), self.ram.size()) };
        let mut written = 0;
        let mut run: Option<usize> = None;
        for (index, block) in ram.chunks(RAM_BLOCK).enumerate() {
            let offset = index * RAM_BLOCK;
            let zero = block.iter().all(|&byte| byte == 0);
            match (run, zero) {
                (None, false) => run = Some(offset),
                (Some(start), true) => {
                    file.write_all_at(&ram[start..offset], start as u64)
                        .map_err(failed)?;
                    written += (offset - start) as u64;
                    run = None;
                }
                _ => {}
            }
        }
        if let Some(start) = run {
            file.write_all_at(&ram[start..], start as u64)
                .map_err(failed)?;
            written += (ram.len() - start) as u64;
        }
        Ok(written)
    }

    /// Fills DRAM from `ram.bin`, reading only the parts of the file that
    /// hold data; the rest of DRAM becomes zero.
    fn read_ram(&self, dir: &Path) -> Result<(), SnapshotError> {
        let path = dir.join("ram.bin");
        let failed = |error| io_error("reading", &path, error);
        let file = File::open(&path).map_err(failed)?;
        let size = self.ram.size();
        if file.metadata().map_err(failed)?.len() != size as u64 {
            return Err(SnapshotError::Corrupt(format!(
                "{} is not {size:#x} bytes",
                path.display()
            )));
        }
        // SAFETY: as in `write_ram`; no core has run yet.
        let ram = unsafe { std::slice::from_raw_parts_mut(self.ram.host_ptr(), size) };
        // Only the kernel image is in DRAM of a machine just built (zeroing
        // all of DRAM would make every page of it resident); the saved
        // machine may hold zeros there. A machine powered on has nothing
        // there yet.
        if self.bpmp.is_none()
            && let Some(kernel) = self.image.get()
        {
            let image = (kernel.load_address() - self.ram.base()) as usize;
            ram[image..image + kernel.bytes().len()].fill(0);
        }
        let fd = file.as_raw_fd();
        let mut offset = 0i64;
        while (offset as usize) < size {
            // SAFETY: plain system calls on an open descriptor.
            let data = unsafe { libc::lseek(fd, offset, libc::SEEK_DATA) };
            if data < 0 {
                let error = io::Error::last_os_error();
                // ENXIO: no data at or after `offset`.
                if error.raw_os_error() == Some(libc::ENXIO) {
                    break;
                }
                return Err(failed(error));
            }
            // SAFETY: as above.
            let hole = unsafe { libc::lseek(fd, data, libc::SEEK_HOLE) };
            if hole < 0 {
                return Err(failed(io::Error::last_os_error()));
            }
            let (start, end) = (data as usize, (hole as usize).min(size));
            file.read_exact_at(&mut ram[start..end], start as u64)
                .map_err(failed)?;
            offset = hole;
        }
        Ok(())
    }
}
