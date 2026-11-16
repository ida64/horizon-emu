//! The host side of the GPU (PFIFO): channels, the runlist, and the
//! execution of each channel's command stream.
//!
//! A channel is an instance block (bound to a channel ID through the CCSR
//! registers) whose RAMFC names the GPFIFO ring and the USERD page. When
//! the CPU moves GP_PUT in USERD (through BAR1; there is no doorbell,
//! spec §4.8), the host fetches GPFIFO entries from GP_GET up to GP_PUT
//! through the channel's page tables, decodes the pushbuffer segments
//! they name, executes host methods itself and hands every other method
//! to the channel's engines, and moves GP_GET on in USERD and RAMFC.
//!
//! Layouts are nvgpu's (`hw_ram_gm20b.h`): RAMFC USERD +0x08/+0x0C,
//! GP_GET +0x14, GP_BASE +0x48/+0x4C (ring size as a power of two in bits
//! 20:16 of the high word); USERD GP_GET +0x88, GP_PUT +0x8C, REF +0x48.
//!
//! Execution is in order and complete: by the time a channel's GP_GET
//! has passed an entry, everything that entry asked for has been done, so
//! wait-for-idle, flushes and TLB invalidates have nothing to wait for. A
//! channel stops between methods for a syncpoint wait or a semaphore
//! acquire that is not yet satisfied ([`Wait`]); executing it again
//! retries that method. Where execution stood inside a segment is kept
//! with the channel ([`Context`]), as the PBDMA keeps it in RAMFC; GP_GET
//! is also written to RAMFC so that a channel re-bound after a power cycle
//! resumes where it stopped.

use std::{
    collections::HashMap,
    sync::atomic::{AtomicBool, Ordering},
};

use serde::{Deserialize, Serialize};

use super::{
    super::host1x::Syncpoints,
    mmu::{self, AddressSpace, BusMemory, ChannelMemory},
    pushbuffer::{GpEntry, Header, MethodRun},
};
use crate::{
    maxwell::{Engines, Unmodeled},
    snapshot::SnapshotError,
};

/// Channels the CCSR table has (`ccsr_channel__size_1` = 512).
pub const CHANNELS: usize = 512;

const RAMFC_USERD: u64 = 0x08;
const RAMFC_USERD_HI: u64 = 0x0C;
const RAMFC_GP_GET: u64 = 0x14;
const RAMFC_GP_BASE: u64 = 0x48;
const RAMFC_GP_BASE_HI: u64 = 0x4C;
const USERD_REF: u64 = 0x48;
const USERD_GP_GET: u64 = 0x88;
const USERD_GP_PUT: u64 = 0x8C;
/// USERD address bits 31:9; bits 1:0 the target, 0 video memory.
const USERD_ADDRESS: u32 = 0xFFFF_FE00;
const TARGET: u32 = 3;

/// Host methods, by word index (`clb06f.h`; SYNCPOINTA/B are Tegra's,
/// encoded by nvservices as `0x2001001C`/`0x2001001D`, spec §4.8).
mod method {
    pub const SET_OBJECT: u32 = 0x00;
    pub const NOP: u32 = 0x02;
    pub const SEMAPHORE_A: u32 = 0x04;
    pub const SEMAPHORE_B: u32 = 0x05;
    pub const SEMAPHORE_C: u32 = 0x06;
    pub const SEMAPHORE_D: u32 = 0x07;
    pub const FB_FLUSH: u32 = 0x09;
    /// `MEM_OP_A`/`MEM_OP_B`: Kepler's memory operations (`cla06f.h`, same
    /// operation field and codes as `MEM_OP_C`/`D`). `clb06f.h` notes them
    /// "removed for gm20x", but the console's NVN emits `MEM_OP_B` =
    /// `0x80000000` (L2 flush dirty) on its GM20B channels, so this host
    /// still takes them.
    pub const MEM_OP_A: u32 = 0x0A;
    pub const MEM_OP_B: u32 = 0x0B;
    pub const MEM_OP_C: u32 = 0x0C;
    pub const MEM_OP_D: u32 = 0x0D;
    pub const SET_REFERENCE: u32 = 0x14;
    pub const SYNCPOINT_A: u32 = 0x1C;
    pub const SYNCPOINT_B: u32 = 0x1D;
    pub const WFI: u32 = 0x1E;
    pub const YIELD: u32 = 0x20;
    /// Methods below this on any subchannel belong to the host.
    pub const HOST_END: u32 = 0x40;
}

/// `SEMAPHORED` operations (one-hot, bits 4:0) and fields.
const SEMAPHORE_ACQUIRE: u32 = 0x01;
const SEMAPHORE_RELEASE: u32 = 0x02;
const SEMAPHORE_ACQ_GEQ: u32 = 0x04;
const SEMAPHORE_ACQ_AND: u32 = 0x08;
const SEMAPHORE_OPERATION: u32 = 0x1F;
const SEMAPHORE_ACQUIRE_SWITCH: u32 = 1 << 12;
const SEMAPHORE_RELEASE_WFI_DISABLE: u32 = 1 << 20;
const SEMAPHORE_RELEASE_4_BYTES: u32 = 1 << 24;

/// `MEM_OP_B`/`MEM_OP_D` operations (bits 31:27) with nothing to do in a
/// model without caches or TLBs: membar, TLB invalidate, L2 invalidates,
/// clean comptags, L2 flush; and `MEM_OP_B`'s soft flush.
const MEM_OPS_WITHOUT_EFFECT: [u32; 6] = [0x05, 0x09, 0x0D, 0x0E, 0x0F, 0x10];
const MEM_OP_B_SOFT_FLUSH: u32 = 0x06;

/// `SYNCPOINTB` (nvgpu `pbdma_syncpointb_*`): operation bits 1:0 (0 wait,
/// 1 increment), wait-switch bit 4, syncpoint index bits 15:8.
const SYNCPOINT_OPERATION: u32 = 3;
const SYNCPOINT_WAIT: u32 = 0;
const SYNCPOINT_INCREMENT: u32 = 1;
const SYNCPOINT_SWITCH: u32 = 1 << 4;
const SYNCPOINT_FIELDS: u32 = 0xFF13;

/// Words fetched from a segment at a time.
const FETCH_WORDS: u32 = 1024;

/// What a stopped channel waits for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Wait {
    /// Host1x syncpoint `index` reaching `threshold`, compared as host1x
    /// does, modulo 2^32 (nvservices' own test at text `0x5E2D4`).
    Syncpoint {
        index: usize,
        threshold: u32,
        /// Whether the channel lets others run meanwhile. Without it the
        /// host's single PBDMA stays with this channel.
        switch: bool,
    },
    /// The semaphore word at virtual address `address`.
    Semaphore {
        address: u64,
        payload: u32,
        operation: u32,
        switch: bool,
    },
}

impl Wait {
    pub fn switch(self) -> bool {
        match self {
            Self::Syncpoint { switch, .. } | Self::Semaphore { switch, .. } => switch,
        }
    }
}

/// How far a channel got.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Outcome {
    /// GP_GET reached GP_PUT.
    Idle,
    /// Stopped at a method that waits.
    Blocked(Wait),
    /// Stopped between methods because the host was asked to look again
    /// (preemption, disabling, a runlist change).
    Interrupted,
}

/// What a channel's execution reaches outside it.
pub struct Env<'a> {
    pub bus: &'a dyn BusMemory,
    pub syncpoints: &'a Syncpoints,
    /// Set when the host must stop the channel between methods.
    pub attention: &'a AtomicBool,
    /// The GPU timer, in nanoseconds.
    pub now: &'a (dyn Fn() -> u64 + Sync),
    /// Whether a survey is being run: a method nothing models is then
    /// reported through `note` and skipped instead of stopping the
    /// channel, so the run shows what follows it (`Notes::survey`).
    pub survey: bool,
    pub note: &'a (dyn Fn(String) + Sync),
}

/// A pushbuffer segment being executed. Machine snapshots save where
/// execution stands in it, not the words fetched ahead: a restored channel
/// fetches them from memory again.
#[derive(Serialize, Deserialize)]
struct Segment {
    address: u64,
    length: u32,
    /// Next word.
    position: u32,
    /// Words from `cached_from` on, as fetched.
    #[serde(skip)]
    cached_from: u32,
    #[serde(skip)]
    cache: Vec<u32>,
}

/// Host method state of a channel (the PBDMA's registers, saved in RAMFC
/// on hardware).
#[derive(Default, Serialize, Deserialize)]
struct Host {
    semaphore_address: u64,
    semaphore_payload: u32,
    syncpoint_payload: u32,
}

/// One channel's execution state.
pub struct Context {
    instance: u64,
    space: AddressSpace,
    userd: u64,
    gp_base: u64,
    gp_entries: u32,
    gp_get: u32,
    segment: Option<Segment>,
    run: Option<MethodRun>,
    host: Host,
    engines: Box<dyn Engines>,
    /// Classes bound to the subchannels by `SET_OBJECT`, for reports.
    classes: [Option<u16>; 8],
    /// How the last turn on the host ended, for reports.
    pub last: Option<Outcome>,
}

/// A [`Context`] as a machine snapshot holds it: everything but the words
/// fetched ahead, with its engines' state ([`Engines::save`]).
#[derive(Serialize, Deserialize)]
pub struct SavedContext {
    instance: u64,
    space: AddressSpace,
    userd: u64,
    gp_base: u64,
    gp_entries: u32,
    gp_get: u32,
    segment: Option<Segment>,
    run: Option<MethodRun>,
    host: Host,
    engines: Vec<u8>,
    classes: [Option<u16>; 8],
    last: Option<Outcome>,
}

/// What a channel's RAMFC says about its rings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Rings {
    userd: u64,
    gp_base: u64,
    gp_entries: u32,
    gp_get: u32,
}

fn rings(bus: &dyn BusMemory, instance: u64) -> Result<Rings, Unmodeled> {
    let word = |offset| mmu::read_u32(bus, instance + offset);
    let userd = word(RAMFC_USERD)?;
    if userd & TARGET != 0 {
        return Err(Unmodeled(format!(
            "channel instance {instance:#x}: USERD in aperture {} (only video memory is modeled)",
            userd & TARGET
        )));
    }
    let gp_base_hi = word(RAMFC_GP_BASE_HI)?;
    Ok(Rings {
        userd: u64::from(word(RAMFC_USERD_HI)? & 0xFF) << 32 | u64::from(userd & USERD_ADDRESS),
        gp_base: u64::from(gp_base_hi & 0xFF) << 32 | u64::from(word(RAMFC_GP_BASE)? & !7),
        gp_entries: 1 << (gp_base_hi >> 16 & 0x1F),
        gp_get: word(RAMFC_GP_GET)?,
    })
}

impl Context {
    /// The channel whose instance block is at bus address `instance`, as
    /// its RAMFC describes it.
    pub fn load(
        bus: &dyn BusMemory,
        instance: u64,
        engines: Box<dyn Engines>,
    ) -> Result<Self, Unmodeled> {
        let rings = rings(bus, instance)?;
        if rings.gp_get >= rings.gp_entries {
            return Err(Unmodeled(format!(
                "channel instance {instance:#x}: GP_GET {} outside its {}-entry ring",
                rings.gp_get, rings.gp_entries
            )));
        }
        Ok(Self {
            instance,
            space: AddressSpace::from_instance(bus, instance)?,
            userd: rings.userd,
            gp_base: rings.gp_base,
            gp_entries: rings.gp_entries,
            gp_get: rings.gp_get,
            segment: None,
            run: None,
            host: Host::default(),
            engines,
            classes: [None; 8],
            last: None,
        })
    }

    /// Whether this is still the channel the instance block describes, or
    /// the driver has set the block up afresh for a new one.
    pub fn still_described(&self, bus: &dyn BusMemory) -> bool {
        rings(bus, self.instance).is_ok_and(|rings| {
            rings
                == Rings {
                    userd: self.userd,
                    gp_base: self.gp_base,
                    gp_entries: self.gp_entries,
                    gp_get: self.gp_get,
                }
        })
    }

    /// One line on where the channel stands: rings, position, classes,
    /// and how its last turn ended.
    pub fn describe(&self, bus: &dyn BusMemory) -> String {
        let put = mmu::read_u32(bus, self.userd + USERD_GP_PUT)
            .map_or_else(|_| "unreadable".into(), |put| put.to_string());
        let segment = self.segment.as_ref().map_or_else(String::new, |segment| {
            format!(
                ", in segment {:#x} at word {}/{}",
                segment.address, segment.position, segment.length
            )
        });
        let run = self.run.map_or_else(String::new, |run| {
            format!(
                ", {} argument(s) of method {:#x} on subchannel {} pending",
                run.left,
                run.method * 4,
                run.subchannel
            )
        });
        let classes: Vec<String> = self
            .classes
            .iter()
            .enumerate()
            .filter_map(|(subchannel, class)| {
                class.map(|class| format!("{subchannel}:{class:#06x}"))
            })
            .collect();
        let last = self
            .last
            .as_ref()
            .map_or_else(|| "none yet".into(), |last| format!("{last:?}"));
        format!(
            "instance {:#x}, GP_GET {} GP_PUT {put} of {}{segment}{run}, classes [{}], last turn {last}",
            self.instance,
            self.gp_get,
            self.gp_entries,
            classes.join(" "),
        )
    }

    /// Makes the engines' work visible in memory before the channel
    /// signals (`Engines::flush`).
    fn flush(&mut self, memory: &ChannelMemory) -> Result<(), Unmodeled> {
        self.engines
            .flush(memory)
            .map_err(|unmodeled| with_fault(memory, unmodeled))
    }

    /// Writes everything the engines hold back to memory
    /// ([`Engines::flush`]), between methods, before a machine snapshot:
    /// what memory holds is then all of the channel's work.
    pub fn write_back(&mut self, bus: &dyn BusMemory) -> Result<(), Unmodeled> {
        let memory = ChannelMemory::new(self.space, bus);
        self.flush(&memory)
    }

    /// The channel's state for a machine snapshot, taken between methods
    /// after [`write_back`](Self::write_back).
    pub fn save(&self) -> Result<SavedContext, SnapshotError> {
        Ok(SavedContext {
            instance: self.instance,
            space: self.space,
            userd: self.userd,
            gp_base: self.gp_base,
            gp_entries: self.gp_entries,
            gp_get: self.gp_get,
            segment: self.segment.as_ref().map(|segment| Segment {
                address: segment.address,
                length: segment.length,
                position: segment.position,
                cached_from: 0,
                cache: Vec::new(),
            }),
            run: self.run,
            host: Host {
                semaphore_address: self.host.semaphore_address,
                semaphore_payload: self.host.semaphore_payload,
                syncpoint_payload: self.host.syncpoint_payload,
            },
            engines: self.engines.save()?,
            classes: self.classes,
            last: self.last,
        })
    }

    /// The channel `saved` describes, with `engines` made for it as for a
    /// new channel and then given their saved state. It goes on from the
    /// saved GP_GET and segment position; words it had fetched ahead are
    /// fetched again.
    pub fn restore(
        saved: SavedContext,
        mut engines: Box<dyn Engines>,
    ) -> Result<Self, SnapshotError> {
        let instance = saved.instance;
        if !saved.gp_entries.is_power_of_two()
            || saved.gp_get >= saved.gp_entries
            || saved
                .segment
                .as_ref()
                .is_some_and(|segment| segment.position > segment.length)
        {
            return Err(SnapshotError::Corrupt(format!(
                "GPU channel instance {instance:#x}: position outside its ring or segment"
            )));
        }
        engines.restore(&saved.engines)?;
        Ok(Self {
            instance,
            space: saved.space,
            userd: saved.userd,
            gp_base: saved.gp_base,
            gp_entries: saved.gp_entries,
            gp_get: saved.gp_get,
            segment: saved.segment,
            run: saved.run,
            host: saved.host,
            engines,
            classes: saved.classes,
            last: saved.last,
        })
    }

    fn gp_put(&self, bus: &dyn BusMemory) -> Result<u32, Unmodeled> {
        let put = mmu::read_u32(bus, self.userd + USERD_GP_PUT)?;
        if put >= self.gp_entries {
            return Err(Unmodeled(format!(
                "GP_PUT {put} outside the {}-entry ring",
                self.gp_entries
            )));
        }
        Ok(put)
    }

    /// Executes until the channel is idle, waits, or is asked to stop.
    /// The engines' work need not be in memory when the turn ends: no one
    /// may look at it before a signal says it is done (`signals`).
    pub fn run(&mut self, env: &Env) -> Result<Outcome, Unmodeled> {
        let memory = ChannelMemory::new(self.space, env.bus);
        self.turn(env, &memory)
    }

    fn turn(&mut self, env: &Env, memory: &ChannelMemory) -> Result<Outcome, Unmodeled> {
        loop {
            if env.attention.load(Ordering::SeqCst) {
                return Ok(Outcome::Interrupted);
            }
            let Some(segment) = &mut self.segment else {
                if self.gp_put(env.bus)? == self.gp_get {
                    return Ok(Outcome::Idle);
                }
                let mut entry = [0; 8];
                let at = self.gp_base + u64::from(self.gp_get) * 8;
                self.space.read(env.bus, at, &mut entry)?;
                let entry = [
                    u32::from_le_bytes(entry[..4].try_into().unwrap()),
                    u32::from_le_bytes(entry[4..].try_into().unwrap()),
                ];
                match GpEntry::decode(entry)? {
                    GpEntry::Segment { address, length } => {
                        self.segment = Some(Segment {
                            address,
                            length,
                            position: 0,
                            cached_from: 0,
                            cache: Vec::new(),
                        })
                    }
                    GpEntry::Nop => self.finish_entry(env.bus)?,
                }
                continue;
            };
            if segment.position == segment.length {
                self.finish_entry(env.bus)?;
                continue;
            }
            let word = fetch(segment, &self.space, env.bus)?;
            let executed = match self.run {
                Some(run) => {
                    let done = self.method(env, memory, run.subchannel, run.method, word)?;
                    if done.is_none() {
                        self.run = run.advance();
                    }
                    done
                }
                None => match Header::decode(word)? {
                    header @ Header::Methods { .. } => {
                        self.run = MethodRun::start(header);
                        None
                    }
                    Header::Immediate {
                        subchannel,
                        method,
                        argument,
                    } => self.method(env, memory, subchannel, method, argument)?,
                    Header::EndSegment => {
                        self.finish_entry(env.bus)?;
                        continue;
                    }
                },
            };
            if let Some(wait) = executed {
                return Ok(Outcome::Blocked(wait));
            }
            if let Some(segment) = &mut self.segment {
                segment.position += 1;
            }
        }
    }

    /// GP_GET moves past the current entry, in RAMFC and in USERD where the
    /// driver reads it.
    fn finish_entry(&mut self, bus: &dyn BusMemory) -> Result<(), Unmodeled> {
        self.segment = None;
        self.gp_get = (self.gp_get + 1) % self.gp_entries;
        mmu::write_u32(bus, self.instance + RAMFC_GP_GET, self.gp_get)?;
        mmu::write_u32(bus, self.userd + USERD_GP_GET, self.gp_get)
    }

    /// One method; `Some` if it has to wait, in which case it has had no
    /// effect yet. In a survey, one nothing models is noted and skipped.
    fn method(
        &mut self,
        env: &Env,
        memory: &ChannelMemory,
        subchannel: u8,
        method: u32,
        argument: u32,
    ) -> Result<Option<Wait>, Unmodeled> {
        match self.execute(env, memory, subchannel, method, argument) {
            Err(Unmodeled(what)) if env.survey => {
                (env.note)(format!("{what} (skipped by the survey)"));
                Ok(None)
            }
            outcome => outcome,
        }
    }

    fn execute(
        &mut self,
        env: &Env,
        memory: &ChannelMemory,
        subchannel: u8,
        method: u32,
        argument: u32,
    ) -> Result<Option<Wait>, Unmodeled> {
        if method == method::SET_OBJECT || method >= method::HOST_END {
            self.engines
                .call(subchannel, method, argument, memory)
                .map_err(|unmodeled| with_fault(memory, unmodeled))?;
            if method == method::SET_OBJECT {
                self.classes[usize::from(subchannel & 7)] = Some(argument as u16);
            }
            return Ok(None);
        }
        if signals(method, argument) {
            self.flush(memory)?;
        }
        let host = &mut self.host;
        match method {
            method::NOP | method::FB_FLUSH | method::WFI | method::MEM_OP_A | method::MEM_OP_C => {}
            // Yielding only changes which channel runs next.
            method::YIELD if argument & !3 == 0 => {}
            method::SEMAPHORE_A if argument & !0xFF == 0 => {
                host.semaphore_address =
                    host.semaphore_address & 0xFFFF_FFFF | u64::from(argument) << 32
            }
            method::SEMAPHORE_B => {
                host.semaphore_address =
                    host.semaphore_address & !0xFFFF_FFFF | u64::from(argument & !3)
            }
            method::SEMAPHORE_C => host.semaphore_payload = argument,
            method::SEMAPHORE_D => return self.semaphore(env, argument),
            method::MEM_OP_B | method::MEM_OP_D
                if MEM_OPS_WITHOUT_EFFECT.contains(&(argument >> 27)) => {}
            method::MEM_OP_B if argument >> 27 == MEM_OP_B_SOFT_FLUSH => {}
            method::SET_REFERENCE => mmu::write_u32(env.bus, self.userd + USERD_REF, argument)?,
            method::SYNCPOINT_A => host.syncpoint_payload = argument,
            method::SYNCPOINT_B if argument & !SYNCPOINT_FIELDS == 0 => {
                let index = (argument >> 8 & 0xFF) as usize;
                let value = env.syncpoints.get(index).ok_or_else(|| {
                    Unmodeled(format!("SYNCPOINTB {argument:#x}: no syncpoint {index}"))
                })?;
                match argument & SYNCPOINT_OPERATION {
                    SYNCPOINT_INCREMENT => env.syncpoints.add(index, 1),
                    SYNCPOINT_WAIT => {
                        let threshold = host.syncpoint_payload;
                        if (value.wrapping_sub(threshold) as i32) < 0 {
                            return Ok(Some(Wait::Syncpoint {
                                index,
                                threshold,
                                switch: argument & SYNCPOINT_SWITCH != 0,
                            }));
                        }
                    }
                    _ => {
                        return Err(Unmodeled(format!(
                            "SYNCPOINTB operation {argument:#x} is not modeled"
                        )));
                    }
                }
            }
            _ => {
                return Err(Unmodeled(format!(
                    "host method {method:#x} (offset {:#x}) = {argument:#x} on subchannel \
                     {subchannel} is not modeled",
                    method * 4
                )));
            }
        }
        Ok(None)
    }

    fn semaphore(&mut self, env: &Env, argument: u32) -> Result<Option<Wait>, Unmodeled> {
        let address = self.host.semaphore_address;
        let payload = self.host.semaphore_payload;
        let unmodeled = || Unmodeled(format!("SEMAPHORED {argument:#x} is not modeled"));
        let operation = argument & SEMAPHORE_OPERATION;
        match operation {
            SEMAPHORE_ACQUIRE | SEMAPHORE_ACQ_GEQ | SEMAPHORE_ACQ_AND => {
                if argument & !(SEMAPHORE_OPERATION | SEMAPHORE_ACQUIRE_SWITCH) != 0 {
                    return Err(unmodeled());
                }
                let wait = Wait::Semaphore {
                    address,
                    payload,
                    operation,
                    switch: argument & SEMAPHORE_ACQUIRE_SWITCH != 0,
                };
                Ok((!self.acquired(env, wait)?).then_some(wait))
            }
            SEMAPHORE_RELEASE => {
                let fields = SEMAPHORE_OPERATION | SEMAPHORE_RELEASE_WFI_DISABLE;
                if argument & !(fields | SEMAPHORE_RELEASE_4_BYTES) != 0 {
                    return Err(unmodeled());
                }
                if argument & SEMAPHORE_RELEASE_4_BYTES != 0 {
                    self.space.write(env.bus, address, &payload.to_le_bytes())?;
                } else {
                    // Payload, a zero word, then the 64-bit GPU time.
                    let mut release = [0; 16];
                    release[..4].copy_from_slice(&payload.to_le_bytes());
                    release[8..].copy_from_slice(&(env.now)().to_le_bytes());
                    self.space.write(env.bus, address, &release)?;
                }
                Ok(None)
            }
            _ => Err(unmodeled()),
        }
    }

    /// Whether `wait`, the wait this channel stopped at, is over.
    pub fn acquired(&self, env: &Env, wait: Wait) -> Result<bool, Unmodeled> {
        Ok(match wait {
            Wait::Syncpoint {
                index, threshold, ..
            } => env
                .syncpoints
                .get(index)
                .is_some_and(|value| value.wrapping_sub(threshold) as i32 >= 0),
            Wait::Semaphore {
                address,
                payload,
                operation,
                ..
            } => {
                let mut value = [0; 4];
                self.space.read(env.bus, address, &mut value)?;
                let value = u32::from_le_bytes(value);
                match operation {
                    SEMAPHORE_ACQUIRE => value == payload,
                    SEMAPHORE_ACQ_GEQ => value.wrapping_sub(payload) as i32 >= 0,
                    _ => value & payload != 0,
                }
            }
        })
    }
}

/// An engine refusal, with the memory fault behind it if there was one.
fn with_fault(memory: &ChannelMemory, Unmodeled(what): Unmodeled) -> Unmodeled {
    let fault = memory
        .take_fault()
        .map(|Unmodeled(fault)| format!(" ({fault})"))
        .unwrap_or_default();
    Unmodeled(format!("{what}{fault}"))
}

/// Host methods after which others may observe what the channel's engines
/// wrote: semaphore releases and syncpoint increments (fences). Wait for
/// idle and the explicit flushes and barriers (`FB_FLUSH`, `MEM_OP_B`/`D`)
/// let no one observe anything until a signal follows.
fn signals(method: u32, argument: u32) -> bool {
    match method {
        method::SEMAPHORE_D => argument & SEMAPHORE_OPERATION == SEMAPHORE_RELEASE,
        method::SYNCPOINT_B => argument & SYNCPOINT_OPERATION == SYNCPOINT_INCREMENT,
        _ => false,
    }
}

/// The word at the segment's position, fetching ahead.
fn fetch(
    segment: &mut Segment,
    space: &AddressSpace,
    bus: &dyn BusMemory,
) -> Result<u32, Unmodeled> {
    let cached = segment.cached_from..segment.cached_from + segment.cache.len() as u32;
    if !cached.contains(&segment.position) {
        let count = (segment.length - segment.position).min(FETCH_WORDS);
        let mut bytes = vec![0; count as usize * 4];
        space.read(
            bus,
            segment.address + u64::from(segment.position) * 4,
            &mut bytes,
        )?;
        let (words, _) = bytes.as_chunks::<4>();
        segment.cache = words.iter().map(|word| u32::from_le_bytes(*word)).collect();
        segment.cached_from = segment.position;
    }
    Ok(segment.cache[(segment.position - segment.cached_from) as usize])
}

/// One CCSR entry.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Ccsr {
    /// `ccsr_channel_inst`: instance pointer bits 27:0 (address >> 12),
    /// target bits 29:28, bound bit 31.
    pub instance: u32,
    pub enabled: bool,
}

const CCSR_BIND: u32 = 1 << 31;
const CCSR_TARGET: u32 = 3 << 28;
const CCSR_POINTER: u32 = 0x0FFF_FFFF;
const CCSR_ENABLE: u32 = 1 << 0;
const CCSR_ENABLE_SET: u32 = 1 << 10;
const CCSR_ENABLE_CLEAR: u32 = 1 << 11;

impl Ccsr {
    /// Bus address of the bound instance block.
    pub fn bound(self) -> Option<u64> {
        (self.instance & CCSR_BIND != 0).then(|| u64::from(self.instance & CCSR_POINTER) << 12)
    }
}

/// PFIFO registers, relative to BAR0.
mod reg {
    pub const INTR_0: u32 = 0x2100;
    pub const INTR_EN_0: u32 = 0x2140;
    pub const RUNLIST_BASE: u32 = 0x2270;
    pub const RUNLIST: u32 = 0x2274;
    pub const ENG_RUNLIST_BASE: u32 = 0x2280;
    pub const ENG_RUNLIST: u32 = 0x2284;
    pub const INTR_EN_1: u32 = 0x2528;
    pub const SCHED_DISABLE: u32 = 0x2630;
    pub const PREEMPT: u32 = 0x2634;
    pub const RUNLIST_PREEMPT: u32 = 0x2638;
    pub const ENGINE_STATUS_GR: u32 = 0x2640;
    pub const ENGINE_STATUS_CE: u32 = 0x2648;
    pub const INTR_RUNLIST: u32 = 0x2A00;
    pub const PBDMA_STATUS: u32 = 0x3080;
    pub const PBDMA_INTR_0: u32 = 0x4_0108;
    pub const PBDMA_INTR_1: u32 = 0x4_0148;
    pub const CCSR: u32 = 0x80_0000;
}

/// `fifo_intr_0` bit for a pending runlist interrupt.
const INTR_RUNLIST_EVENT: u32 = 1 << 30;
/// Pending bit of `fifo_eng_runlist` and `fifo_preempt`.
const PENDING: u32 = 1 << 20;
const PREEMPT_TYPE_TSG: u32 = 1 << 24;
/// Status words of an engine or the PBDMA holding a channel: busy (engine
/// only), context status "valid" in bits 15:13, channel ID.
const ENGINE_BUSY: u32 = 1 << 31;
const CONTEXT_VALID: u32 = 1 << 13;
/// Runlist entry: channel ID in bits 11:0; bit 13 marks a TSG entry.
const RUNLIST_CHID: u32 = 0xFFF;
const RUNLIST_TSG: u32 = 1 << 13;

/// The host's registers and channels. Machine snapshots save the
/// registers, CCSR table and runlist here; the channel states separately
/// ([`Context::save`]), and only while no channel is executing.
#[derive(Serialize, Deserialize)]
pub struct Fifo {
    pub channels: Vec<Ccsr>,
    /// Channel IDs of the last runlist submitted, in its order.
    pub runlist: Vec<usize>,
    runlist_base: u32,
    runlist_length: u32,
    /// Pending `fifo_intr_0` bits other than the runlist summary. The model
    /// raises none of them.
    intr: u32,
    pub intr_en_0: u32,
    pub intr_en_1: u32,
    runlist_intr: u32,
    pub sched_disable: u32,
    preempt: u32,
    preempt_pending: bool,
    runlist_preempt_pending: u32,
    /// The channel the worker is executing.
    #[serde(skip)]
    pub executing: Option<usize>,
    /// Channel states by instance block address.
    #[serde(skip)]
    pub contexts: HashMap<u64, Context>,
}

impl Default for Fifo {
    fn default() -> Self {
        Self {
            channels: vec![Ccsr::default(); CHANNELS],
            runlist: Vec::new(),
            runlist_base: 0,
            runlist_length: 0,
            intr: 0,
            intr_en_0: 0,
            intr_en_1: 0,
            runlist_intr: 0,
            sched_disable: 0,
            preempt: 0,
            preempt_pending: false,
            runlist_preempt_pending: 0,
            executing: None,
            contexts: HashMap::new(),
        }
    }
}

impl Fifo {
    /// The PFIFO reset (`PMC_ENABLE` bit 8 cleared): registers back to
    /// rest. Channel states stay: on hardware they live in each channel's
    /// RAMFC in memory.
    pub fn reset(&mut self) {
        let contexts = std::mem::take(&mut self.contexts);
        let executing = self.executing;
        *self = Self {
            contexts,
            executing,
            ..Self::default()
        };
    }

    /// Whether restored registers describe this host: a full CCSR table
    /// and a runlist of its channels.
    pub fn fits(&self) -> bool {
        self.channels.len() == CHANNELS && self.runlist.iter().all(|&chid| chid < CHANNELS)
    }

    /// `fifo_intr_0` as read: the runlist interrupt shows while any
    /// runlist bit is pending; the driver acknowledges it only through
    /// `fifo_intr_runlist` (spec §4.9).
    pub fn intr(&self) -> u32 {
        self.intr
            | if self.runlist_intr() != 0 {
                INTR_RUNLIST_EVENT
            } else {
                0
            }
    }

    /// `fifo_intr_runlist`: a runlist's bit is set while the runlist is
    /// preempted (its preempt through `0x2638` has completed) and its
    /// scheduling stays disabled through `0x2630`. It is a level, not a
    /// latched event: in `NV_ReplaceRunlistEligibilityAndResubmit_729F0`,
    /// when the 100 µs spin for it times out, the driver waits for the
    /// bit on the non-stall interrupt, acknowledges it, rebuilds the
    /// runlist and then waits for it a second time before it resubmits
    /// and re-enables the runlist, which only a status that persists while
    /// the runlist stays preempted satisfies [INFERENCE from that code].
    fn runlist_intr(&self) -> u32 {
        self.runlist_intr & self.sched_disable
    }

    /// One line on the host's scheduling state, for reports.
    pub fn describe(&self) -> String {
        format!(
            "host: runlist {:?}, scheduling disable {:#x}, preempt {:#x}{}, runlist \
             preempt pending {:#x}, runlist interrupt {:#x}, interrupt {:#x} (stall enable \
             {:#x}, non-stall enable {:#x})",
            self.runlist,
            self.sched_disable,
            self.preempt,
            if self.preempt_pending { " pending" } else { "" },
            self.runlist_preempt_pending,
            self.runlist_intr(),
            self.intr(),
            self.intr_en_0,
            self.intr_en_1
        )
    }

    /// Whether channel `chid` may run.
    pub fn runnable(&self, chid: usize) -> bool {
        let ccsr = self.channels[chid];
        ccsr.enabled
            && ccsr.bound().is_some()
            && self.sched_disable & 1 == 0
            && self.runlist.contains(&chid)
            && !(self.preempt_pending && self.preempted() == Some(chid))
    }

    fn preempted(&self) -> Option<usize> {
        (self.preempt & PREEMPT_TYPE_TSG == 0).then_some((self.preempt & 0xFFF) as usize)
    }

    /// The worker left channel `chid`: whatever waited for that is done.
    pub fn left(&mut self, chid: usize) {
        if self.preempted() == Some(chid) {
            self.preempt_pending = false;
        }
        self.executing = None;
        self.complete_runlist_preempt();
    }

    fn complete_runlist_preempt(&mut self) {
        if self.executing.is_none() {
            self.runlist_intr |= self.runlist_preempt_pending;
            self.runlist_preempt_pending = 0;
        }
    }

    /// `None`: not a register with behavior here.
    pub fn read(&self, offset: u32) -> Option<Result<u32, Unmodeled>> {
        let on_engine = |status| {
            self.executing
                .map_or(0, |chid| status | CONTEXT_VALID | chid as u32)
        };
        Some(Ok(match offset {
            reg::INTR_0 => self.intr(),
            reg::INTR_EN_0 => self.intr_en_0,
            reg::INTR_EN_1 => self.intr_en_1,
            reg::RUNLIST_BASE => self.runlist_base,
            reg::ENG_RUNLIST_BASE => self.runlist_base,
            // Runlist updates are over by the time the write returns.
            reg::ENG_RUNLIST => self.runlist_length,
            reg::SCHED_DISABLE => self.sched_disable,
            reg::PREEMPT => self.preempt | if self.preempt_pending { PENDING } else { 0 },
            reg::ENGINE_STATUS_GR => on_engine(ENGINE_BUSY),
            // The copy engine's work is done inside the graphics channel's
            // engines; it never holds a context of its own.
            reg::ENGINE_STATUS_CE => 0,
            reg::INTR_RUNLIST => self.runlist_intr(),
            reg::PBDMA_STATUS => on_engine(0),
            // No PBDMA error is ever raised.
            reg::PBDMA_INTR_0 | reg::PBDMA_INTR_1 => 0,
            reg::CCSR..0x80_1000 => {
                let ccsr = self.channels[((offset - reg::CCSR) / 8) as usize];
                if offset.is_multiple_of(8) {
                    ccsr.instance
                } else if ccsr.enabled {
                    CCSR_ENABLE
                } else {
                    0
                }
            }
            _ => return None,
        }))
    }

    /// `None`: not a register with behavior here. `bus` is where runlists
    /// and instance blocks are read from.
    pub fn write(
        &mut self,
        offset: u32,
        value: u32,
        bus: &dyn BusMemory,
    ) -> Option<Result<(), Unmodeled>> {
        match offset {
            reg::INTR_0 => self.intr &= !value,
            reg::INTR_EN_0 => self.intr_en_0 = value,
            reg::INTR_EN_1 => self.intr_en_1 = value,
            reg::RUNLIST_BASE => self.runlist_base = value,
            reg::RUNLIST => return Some(self.submit_runlist(value, bus)),
            reg::SCHED_DISABLE => {
                self.sched_disable = value;
                // Re-enabled runlists run again: no longer preempted.
                self.runlist_intr &= value;
            }
            reg::PREEMPT => {
                if value & !(0xFFF | PREEMPT_TYPE_TSG) != 0 || value & PREEMPT_TYPE_TSG != 0 {
                    return Some(Err(Unmodeled(format!(
                        "preempt request {value:#x}: only channel preemption is modeled"
                    ))));
                }
                self.preempt = value;
                self.preempt_pending = self.executing == Some((value & 0xFFF) as usize);
            }
            // Not in nvgpu's gm20b register list; nvservices writes the
            // runlist mask here and waits for that runlist's bit in
            // `fifo_intr_runlist` on the non-stall line, with no timeout
            // (`NV_ReplaceRunlistEligibilityAndResubmit_729F0`, spec §4.9).
            // The bit comes once no channel of the runlist is executing.
            reg::RUNLIST_PREEMPT => {
                if value & !1 != 0 {
                    return Some(Err(Unmodeled(format!(
                        "runlist preempt {value:#x}: there is one runlist"
                    ))));
                }
                self.runlist_preempt_pending |= value;
                self.complete_runlist_preempt();
            }
            // Acknowledging changes nothing while the runlist stays
            // preempted (see `runlist_intr`).
            reg::INTR_RUNLIST => {}
            reg::PBDMA_INTR_0 | reg::PBDMA_INTR_1 => {}
            reg::CCSR..0x80_1000 => {
                let chid = ((offset - reg::CCSR) / 8) as usize;
                return Some(if offset.is_multiple_of(8) {
                    self.bind(chid, value, bus)
                } else {
                    self.control(chid, value)
                });
            }
            _ => return None,
        }
        Some(Ok(()))
    }

    fn submit_runlist(&mut self, value: u32, bus: &dyn BusMemory) -> Result<(), Unmodeled> {
        let length = value & 0xFFFF;
        if value & !0xFFFF != 0 {
            return Err(Unmodeled(format!(
                "runlist submit {value:#x}: there is one runlist, 0"
            )));
        }
        let mut runlist = Vec::with_capacity(length as usize);
        if length > 0 {
            if self.runlist_base & CCSR_TARGET != 0 {
                return Err(Unmodeled(format!(
                    "runlist base {:#x}: only video memory is modeled",
                    self.runlist_base
                )));
            }
            let base = u64::from(self.runlist_base & CCSR_POINTER) << 12;
            for index in 0..u64::from(length) {
                let entry = mmu::read_u32(bus, base + index * 8)?;
                if entry & RUNLIST_TSG != 0 || entry & !RUNLIST_CHID != 0 {
                    return Err(Unmodeled(format!(
                        "runlist entry {entry:#x}: only channel entries are modeled"
                    )));
                }
                let chid = (entry & RUNLIST_CHID) as usize;
                if chid >= CHANNELS {
                    return Err(Unmodeled(format!("runlist entry for channel {chid}")));
                }
                runlist.push(chid);
            }
        }
        self.runlist = runlist;
        self.runlist_length = length;
        Ok(())
    }

    fn bind(&mut self, chid: usize, value: u32, bus: &dyn BusMemory) -> Result<(), Unmodeled> {
        if value & CCSR_BIND != 0 && value & CCSR_TARGET != 0 {
            return Err(Unmodeled(format!(
                "channel {chid} bound to an instance block in aperture {} (only video \
                 memory is modeled)",
                value >> 28 & 3
            )));
        }
        self.channels[chid].instance = value;
        if let Some(instance) = self.channels[chid].bound() {
            // A channel set up afresh on a used instance block starts over.
            if self
                .contexts
                .get(&instance)
                .is_some_and(|context| !context.still_described(bus))
            {
                self.contexts.remove(&instance);
            }
        }
        Ok(())
    }

    fn control(&mut self, chid: usize, value: u32) -> Result<(), Unmodeled> {
        if value & !(CCSR_ENABLE | CCSR_ENABLE_SET | CCSR_ENABLE_CLEAR) != 0 {
            return Err(Unmodeled(format!(
                "channel {chid} control write {value:#x}: only enable and disable are modeled"
            )));
        }
        if value & CCSR_ENABLE_SET != 0 {
            self.channels[chid].enabled = true;
        }
        if value & CCSR_ENABLE_CLEAR != 0 {
            self.channels[chid].enabled = false;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::{super::mmu::tests::TestBus, *};
    use crate::maxwell::GpuMemory;

    const IOVA: u64 = mmu::SMMU_BIT;
    const INSTANCE: u64 = IOVA | 0x1_0000;
    const USERD: u64 = IOVA | 0x2_0000;
    const DIRECTORY: u64 = IOVA | 0x3_0000;
    const TABLE: u64 = IOVA | 0x4_0000;
    /// Virtual addresses: the GPFIFO ring and a pushbuffer.
    const RING: u64 = 0x1000;
    const PUSHBUFFER: u64 = 0x2000;
    const SEMAPHORE: u64 = 0x3000;
    const ENTRIES: u32 = 16;

    /// What test engines are flushed with: the call log gets this entry.
    const FLUSH: (u8, u32, u32) = (0xFF, 0, 0);

    /// Engines that record what they are given.
    #[derive(Clone, Default)]
    struct Recorder(Arc<Mutex<Vec<(u8, u32, u32)>>>);

    impl Engines for Recorder {
        fn call(
            &mut self,
            subchannel: u8,
            method: u32,
            argument: u32,
            _memory: &dyn GpuMemory,
        ) -> Result<(), Unmodeled> {
            if method == 0x7FF {
                return Err(Unmodeled("method 0x7ff".into()));
            }
            self.0.lock().unwrap().push((subchannel, method, argument));
            Ok(())
        }

        fn flush(&mut self, _memory: &dyn GpuMemory) -> Result<(), Unmodeled> {
            self.0.lock().unwrap().push(FLUSH);
            Ok(())
        }

        fn save(&self) -> Result<Vec<u8>, SnapshotError> {
            crate::snapshot::encode(&*self.0.lock().unwrap())
        }

        fn restore(&mut self, state: &[u8]) -> Result<(), SnapshotError> {
            *self.0.lock().unwrap() = crate::snapshot::decode("recorder", state)?;
            Ok(())
        }
    }

    /// A channel as nvservices sets one up (annex fifo §4.1): instance
    /// block with a 4 KiB-page address space mapping the ring, a
    /// pushbuffer page and a semaphore page one to one onto low IOVAs.
    fn channel(bus: &TestBus) -> (Context, Recorder) {
        bus.instance(INSTANCE, DIRECTORY, mmu::BigPage::K128, 1 << 37);
        bus.small_table(DIRECTORY, TABLE);
        for page in [RING, PUSHBUFFER, SEMAPHORE] {
            bus.page(TABLE + (page >> 12) * 8, IOVA | (0x10_0000 + page));
        }
        bus.put(INSTANCE + RAMFC_USERD, USERD as u32);
        bus.put(INSTANCE + RAMFC_USERD_HI, (USERD >> 32) as u32);
        bus.put(INSTANCE + RAMFC_GP_BASE, RING as u32);
        bus.put(INSTANCE + RAMFC_GP_BASE_HI, ENTRIES.trailing_zeros() << 16);
        let engines = Recorder::default();
        let context = Context::load(bus, INSTANCE, Box::new(engines.clone())).unwrap();
        (context, engines)
    }

    /// Puts `words` at `PUSHBUFFER + offset` and a GPFIFO entry for them at
    /// ring index `index`, as `NV_SubmitChannelGpFifo_7C200` does.
    fn submit(bus: &TestBus, index: u32, offset: u64, words: &[u32]) {
        for (at, word) in words.iter().enumerate() {
            bus.put(
                IOVA | (0x10_0000 + PUSHBUFFER + offset + at as u64 * 4),
                *word,
            );
        }
        let entry = IOVA | (0x10_0000 + RING + u64::from(index) * 8);
        bus.put(entry, (PUSHBUFFER + offset) as u32);
        bus.put(entry + 4, (words.len() as u32) << 10);
        bus.put(USERD + USERD_GP_PUT, (index + 1) % ENTRIES);
    }

    fn run(context: &mut Context, bus: &TestBus, syncpoints: &Syncpoints) -> Outcome {
        let attention = AtomicBool::new(false);
        context
            .run(&Env {
                bus,
                syncpoints,
                attention: &attention,
                now: &|| 0x1234_5678_9ABC,
                survey: false,
                note: &|_| {},
            })
            .unwrap()
    }

    #[test]
    fn fence_increments_and_gp_get_advance() {
        let bus = TestBus::default();
        let syncpoints = Syncpoints::default();
        let (mut context, engines) = channel(&bus);
        // A client segment (bind 3D on subchannel 0, one 3D method), then
        // the driver's fence: WFI and two increments of syncpoint 0x19.
        submit(&bus, 0, 0, &[0x2001_0000, 0xB197, 0x2001_0040, 7]);
        submit(
            &bus,
            1,
            0x100,
            &[
                0x2001_001E,
                0,
                0x2001_001C,
                0,
                0x2001_001D,
                0x19 << 8 | 1,
                0x2001_001D,
                0x19 << 8 | 1,
            ],
        );
        assert_eq!(run(&mut context, &bus, &syncpoints), Outcome::Idle);
        // Flushed before each increment of the fence.
        assert_eq!(
            *engines.0.lock().unwrap(),
            [(0, 0, 0xB197), (0, 0x40, 7), FLUSH, FLUSH]
        );
        assert_eq!(syncpoints.get(0x19), Some(2));
        assert_eq!(bus.get(USERD + USERD_GP_GET), 2);
        assert_eq!(bus.get(INSTANCE + RAMFC_GP_GET), 2);
        assert!(context.still_described(&bus));
    }

    #[test]
    fn a_syncpoint_wait_suspends_and_resumes() {
        let bus = TestBus::default();
        let syncpoints = Syncpoints::default();
        let (mut context, _) = channel(&bus);
        // The driver's pre-fence: wait for syncpoint 5 to reach 3, switch
        // enabled; then an increment of syncpoint 6.
        submit(
            &bus,
            0,
            0,
            &[
                0x2001_001C,
                3,
                0x2001_001D,
                5 << 8 | 0x10,
                0x2001_001D,
                6 << 8 | 1,
            ],
        );
        syncpoints.add(5, 2);
        let wait = Wait::Syncpoint {
            index: 5,
            threshold: 3,
            switch: true,
        };
        assert_eq!(run(&mut context, &bus, &syncpoints), Outcome::Blocked(wait));
        assert_eq!(bus.get(USERD + USERD_GP_GET), 0);
        assert_eq!(syncpoints.get(6), Some(0));
        let env_attention = AtomicBool::new(false);
        let env = Env {
            bus: &bus,
            syncpoints: &syncpoints,
            attention: &env_attention,
            now: &|| 0,
            survey: false,
            note: &|_| {},
        };
        assert!(!context.acquired(&env, wait).unwrap());
        // Still waiting when run again.
        assert_eq!(run(&mut context, &bus, &syncpoints), Outcome::Blocked(wait));
        syncpoints.add(5, 1);
        assert!(context.acquired(&env, wait).unwrap());
        assert_eq!(run(&mut context, &bus, &syncpoints), Outcome::Idle);
        assert_eq!(syncpoints.get(6), Some(1));
        assert_eq!(bus.get(USERD + USERD_GP_GET), 1);
    }

    #[test]
    fn semaphores_acquire_and_release() {
        let bus = TestBus::default();
        let syncpoints = Syncpoints::default();
        let (mut context, _) = channel(&bus);
        let backing = IOVA | (0x10_0000 + SEMAPHORE);
        // Acquire value 9 at SEMAPHORE (switch on), then release 0x55 with
        // the 4-byte form at +0x10 and the timestamped form at +0x20.
        submit(
            &bus,
            0,
            0,
            &[
                0x2004_0004,
                0,
                SEMAPHORE as u32,
                9,
                0x1001,
                0x2003_0005,
                SEMAPHORE as u32 + 0x10,
                0x55,
                0x0100_0002,
                0x2003_0005,
                SEMAPHORE as u32 + 0x20,
                0x66,
                0x0000_0002,
            ],
        );
        let outcome = run(&mut context, &bus, &syncpoints);
        assert!(matches!(
            outcome,
            Outcome::Blocked(Wait::Semaphore { payload: 9, .. })
        ));
        bus.put(backing, 9);
        assert_eq!(run(&mut context, &bus, &syncpoints), Outcome::Idle);
        assert_eq!(bus.get(backing + 0x10), 0x55);
        assert_eq!(bus.get(backing + 0x20), 0x66);
        assert_eq!(bus.get(backing + 0x24), 0);
        assert_eq!(bus.get(backing + 0x28), 0x5678_9ABC);
        assert_eq!(bus.get(backing + 0x2C), 0x1234);
    }

    #[test]
    fn unmodeled_methods_fail_loudly() {
        let bus = TestBus::default();
        let syncpoints = Syncpoints::default();
        let attention = AtomicBool::new(false);
        let env = Env {
            bus: &bus,
            syncpoints: &syncpoints,
            attention: &attention,
            now: &|| 0,
            survey: false,
            note: &|_| {},
        };
        let (mut context, _) = channel(&bus);
        submit(&bus, 0, 0, &[0x2001_07FF, 1]);
        assert!(context.run(&env).is_err());
        let (mut context, _) = channel(&bus);
        // NON_STALL_INTERRUPT.
        submit(&bus, 0, 0, &[0x2001_0008, 0]);
        assert!(context.run(&env).is_err());
        // A survey notes them and goes on to the next.
        let notes = Mutex::new(Vec::new());
        let note = |what: String| notes.lock().unwrap().push(what);
        let surveyed = Env {
            survey: true,
            note: &note,
            ..env
        };
        let (mut context, _) = channel(&bus);
        submit(&bus, 0, 0, &[0x2001_07FF, 1, 0x2001_001D, 3 << 8 | 1]);
        assert_eq!(context.run(&surveyed).unwrap(), Outcome::Idle);
        assert_eq!(syncpoints.get(3), Some(1));
        assert_eq!(notes.lock().unwrap().len(), 1);
    }

    #[test]
    fn interruption_stops_between_methods() {
        let bus = TestBus::default();
        let syncpoints = Syncpoints::default();
        let (mut context, engines) = channel(&bus);
        submit(&bus, 0, 0, &[0x2002_0040, 1, 2]);
        let attention = AtomicBool::new(true);
        let env = Env {
            bus: &bus,
            syncpoints: &syncpoints,
            attention: &attention,
            now: &|| 0,
            survey: false,
            note: &|_| {},
        };
        assert_eq!(context.run(&env).unwrap(), Outcome::Interrupted);
        attention.store(false, Ordering::SeqCst);
        assert_eq!(context.run(&env).unwrap(), Outcome::Idle);
        // Both methods; nothing signals, so nothing is flushed.
        assert_eq!(*engines.0.lock().unwrap(), [(0, 0x40, 1), (0, 0x41, 2)]);
    }

    #[test]
    fn runlist_ccsr_and_preemption() {
        let bus = TestBus::default();
        let mut fifo = Fifo::default();
        // Runlist with channels 3 and 5 at IOVA 0x50000.
        bus.put(IOVA | 0x5_0000, 3);
        bus.put(IOVA | 0x5_0008, 5);
        let write = |fifo: &mut Fifo, offset, value| fifo.write(offset, value, &bus).unwrap();
        write(
            &mut fifo,
            reg::RUNLIST_BASE,
            ((IOVA | 0x5_0000) >> 12) as u32,
        )
        .unwrap();
        write(&mut fifo, reg::RUNLIST, 2).unwrap();
        assert_eq!(fifo.read(reg::ENG_RUNLIST).unwrap().unwrap() & PENDING, 0);
        assert!(!fifo.runnable(3));
        write(
            &mut fifo,
            reg::CCSR + 3 * 8,
            ((INSTANCE >> 12) as u32) | CCSR_BIND,
        )
        .unwrap();
        let control = fifo.read(reg::CCSR + 3 * 8 + 4).unwrap().unwrap();
        write(&mut fifo, reg::CCSR + 3 * 8 + 4, control | CCSR_ENABLE_SET).unwrap();
        assert!(fifo.runnable(3));
        assert_eq!(fifo.read(reg::CCSR + 3 * 8 + 4).unwrap().unwrap(), 1);
        // Preempting the channel the worker holds stays pending until it
        // lets go of it.
        fifo.executing = Some(3);
        write(&mut fifo, reg::PREEMPT, 3).unwrap();
        assert_ne!(fifo.read(reg::PREEMPT).unwrap().unwrap() & PENDING, 0);
        assert!(!fifo.runnable(3));
        assert_ne!(
            fifo.read(reg::ENGINE_STATUS_GR).unwrap().unwrap() & ENGINE_BUSY,
            0
        );
        fifo.left(3);
        assert_eq!(fifo.read(reg::PREEMPT).unwrap().unwrap() & PENDING, 0);
        assert_eq!(fifo.read(reg::ENGINE_STATUS_GR).unwrap().unwrap(), 0);
        // The runlist preempt of the nvsched swap raises the runlist
        // interrupt, which shows in fifo_intr_0, survives acknowledgment,
        // and goes once the runlist is enabled again.
        write(&mut fifo, reg::SCHED_DISABLE, 1).unwrap();
        write(&mut fifo, reg::RUNLIST_PREEMPT, 1).unwrap();
        assert_eq!(fifo.read(reg::INTR_RUNLIST).unwrap().unwrap(), 1);
        assert_ne!(fifo.intr() & INTR_RUNLIST_EVENT, 0);
        write(&mut fifo, reg::INTR_RUNLIST, 1).unwrap();
        assert_eq!(fifo.read(reg::INTR_RUNLIST).unwrap().unwrap(), 1);
        write(&mut fifo, reg::SCHED_DISABLE, 0).unwrap();
        assert_eq!(fifo.intr(), 0);
        // A preempt that completes later, when the worker lets go.
        write(&mut fifo, reg::SCHED_DISABLE, 1).unwrap();
        fifo.executing = Some(3);
        write(&mut fifo, reg::RUNLIST_PREEMPT, 1).unwrap();
        assert_eq!(fifo.read(reg::INTR_RUNLIST).unwrap().unwrap(), 0);
        fifo.left(3);
        assert_eq!(fifo.read(reg::INTR_RUNLIST).unwrap().unwrap(), 1);
        write(&mut fifo, reg::SCHED_DISABLE, 0).unwrap();
        // Disable.
        write(&mut fifo, reg::CCSR + 3 * 8 + 4, 1 | CCSR_ENABLE_CLEAR).unwrap();
        assert!(!fifo.runnable(3));
    }
}
