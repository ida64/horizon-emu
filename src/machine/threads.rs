//! The guest kernel's own objects, read from memory: the threads of the
//! guest's processes once the machine has stopped, each with the
//! supervisor call it is in and, where that call waits for something
//! another thread provides, who provides it (the server of a session, the
//! owner of a lock, the holders of an event's writable side); and, while
//! it runs, whether the boot's program launcher is still a process, and
//! the address spaces of a program's processes. Nothing is watched in the
//! guest, so a stalled boot looked at this way stalls with the timing it has
//! without the looking.
//!
//! The layout is that of kernel 23.5.0 (`kernel_normal.elf`); every offset
//! below names the code it is read off, by image address.

use std::collections::HashMap;

use super::{
    Machine, Process,
    kernel_layout::{STACK_TOP_THREAD, THREAD_PROCESS},
};
use crate::{arm64::TranslationRegime, svc};

/// Kernel image addresses and object offsets the walk uses.
mod layout {
    /// `KSlabHeap<KThread>`: `KSlabHeap_InitializeThread` (0x80068580)
    /// publishes its bounds there.
    pub const THREAD_SLAB: u64 = 0x800d_b6b0;
    /// `KSlabHeap<KProcess>`, whose objects the context switch (0x8009e654)
    /// numbers from the slab's allocation start to pick their `TTBR0_EL1`.
    pub const PROCESS_SLAB: u64 = 0x800d_b688;
    /// A pointer to the `TTBR0_EL1` values: the kernel's own first, then one
    /// per process slab object, in slab order (0x8009e678-0x8009e694).
    pub const PROCESS_TTBRS: u64 = 0x800d_a520;
    /// Fields of a slab heap: object size, first object, start of the
    /// objects handed out (what the context switch numbers from), end.
    pub const SLAB_OBJECT_SIZE: u64 = 0x08;
    pub const SLAB_BEGIN: u64 = 0x10;
    pub const SLAB_ALLOCATION_BEGIN: u64 = 0x18;
    pub const SLAB_END: u64 = 0x20;

    /// Every kernel object: its reference count (`u32`, 0 once destroyed)
    /// and class token (`u16`, which `SendSyncRequest` at 0x800ba990 tests
    /// against the client-session mask 0xD00).
    pub const REFERENCES: u64 = 0x08;
    pub const CLASS_TOKEN: u64 = 0x0C;

    /// `KThread`, as `KThread_Initialize` (0x800a4ce0) fills it besides its
    /// process (`kernel_layout::THREAD_PROCESS`): thread ID, top of the
    /// kernel stack, thread-local region's address in the process and in
    /// the kernel.
    pub const THREAD_ID: u64 = 0x270;
    pub const THREAD_KERNEL_STACK_TOP: u64 = 0x290;
    pub const THREAD_TLS: u64 = 0x2A0;
    pub const THREAD_TLS_KERNEL: u64 = 0x2A8;
    /// The scheduling state, in the low four bits (`KThread_SetState`,
    /// 0x800a5910).
    pub const THREAD_STATE: u64 = 0x5B0;

    /// The parameter block at the top of every thread's kernel stack, this
    /// far below its top (zeroed by `KThread_Initialize`): the supervisor
    /// call entry (0x800c5114) records the call's immediate and sets bit 0
    /// of the flags while a call is in progress.
    pub const STACK_PARAMETERS: u64 = 0x140;
    pub const PARAMETERS_SVC_ID: u64 = 0x2B;
    pub const PARAMETERS_FLAGS: u64 = 0x2D;
    /// The registers the supervisor call entry saves, in the 0x120 bytes
    /// below the parameter block: `X0`-`X7` first, `X29` and `X30` at 0xE8,
    /// `ELR_EL1` at 0x100.
    pub const SAVED_REGISTERS: u64 = 0x120;
    pub const SAVED_X29: u64 = 0xE8;
    pub const SAVED_ELR: u64 = 0x100;

    /// A process's handle table (`SendSyncRequest`, 0x800ba990): the linear
    /// ID of each entry (`u16`) a handle must repeat in its upper bits, the
    /// object of each entry, and how many entries there are (`u16`).
    pub const PROCESS_HANDLE_IDS: u64 = 0x350;
    pub const PROCESS_HANDLE_OBJECTS: u64 = 0xB50;
    pub const PROCESS_HANDLE_COUNT: u64 = 0x2B58;

    /// A client session points to its session (`SendSyncRequest`); the
    /// session holds its server and client ends and points to the client
    /// end of the port it came from, if any (`KSession_Initialize`,
    /// 0x800a39f0).
    pub const CLIENT_SESSION_PARENT: u64 = 0x18;
    pub const SESSION_SERVER: u64 = 0x38;
    pub const SESSION_CLIENT: u64 = 0xB8;
    pub const SESSION_CLIENT_PORT: u64 = 0xD8;
    /// A port holds its server end and its client end, and the name its
    /// creator gave `CreatePort` (`KPort_Initialize`, 0x80093b10).
    pub const PORT_SERVER: u64 = 0x38;
    pub const PORT_CLIENT: u64 = 0xA8;
    pub const PORT_NAME: u64 = 0x108;
    /// An event holds its readable end (`CreateEvent`, 0x800b3d70).
    pub const EVENT_READABLE: u64 = 0x38;

    /// Class tokens: client and server session are the masks
    /// `SendSyncRequest` and `ReplyAndReceive` (0x800babf0) test; the
    /// others are Atmosphère's (`kern_k_class_token.hpp`), which agree with
    /// those two \[inference\]; the thread's is checked on every thread
    /// object the walk takes.
    pub const THREAD_TOKEN: u16 = 0x1301;
    pub const CLIENT_SESSION_TOKEN: u16 = 0x0D00;
    pub const SERVER_SESSION_TOKEN: u16 = 0x4301;
    pub const SERVER_PORT_TOKEN: u16 = 0x2301;
    pub const READABLE_EVENT_TOKEN: u16 = 0x0003;
    pub const EVENT_TOKEN: u16 = 0x4900;
    pub const PROCESS_TOKEN: u16 = 0x1501;

    /// Program ID of the boot's program launcher, boot2.
    pub const LAUNCHER_PROGRAM_ID: u64 = 0x0100_0000_0000_0008;

    /// The pseudo-handle for the calling thread.
    pub const CURRENT_THREAD: u32 = 0xFFFF_8000;
    /// `WaitSynchronization` takes at most this many handles.
    pub const MOST_WAITED: u64 = 64;
}

/// Supervisor calls whose handles name what the caller waits for.
const WAIT_SYNCHRONIZATION: u16 = 0x18;
const ARBITRATE_LOCK: u16 = 0x1A;

/// Scheduling state of a thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreadState {
    Initialized,
    Waiting,
    Runnable,
    Terminated,
    Unknown(u16),
}

/// A thread of a process.
#[derive(Debug, Clone)]
pub struct GuestThread {
    /// The kernel's thread ID.
    pub id: u64,
    pub process: Process,
    /// Address of its thread-local region in the process, which tells it
    /// apart in the supervisor call traces.
    pub tls: u64,
    pub state: ThreadState,
    /// The supervisor call it is in, if any.
    pub call: Option<ThreadCall>,
}

/// A supervisor call a thread is in.
#[derive(Debug, Clone)]
pub struct ThreadCall {
    /// Immediate of the `SVC` instruction.
    pub id: u16,
    /// `X0`-`X7` at the call.
    pub arguments: [u64; 8],
    /// Address of the `SVC` instruction.
    pub pc: u64,
    /// `X30` at the call.
    pub lr: u64,
    /// `X29` at the call.
    pub fp: u64,
    /// The request a request-sending call sends, as far as it could be
    /// read: the thread's message buffer or the call's user buffer.
    pub message: Vec<u8>,
    /// What the call's handles that name what it waits for stand for: the
    /// session of a request, the owner of a lock, the objects waited for.
    pub objects: Vec<(u32, KernelObject)>,
}

/// What a handle stands for, and who is on the other side of it.
#[derive(Debug, Clone)]
pub enum KernelObject {
    /// Not an open handle of the caller's.
    NotOpen,
    /// A session: the name its port was created with (`CreatePort`'s `name`
    /// argument, whatever the creator passed; `None` for a session that
    /// came from no port) and the processes holding the session's server
    /// end, or the port's until the session is accepted.
    Session {
        port_name: Option<u64>,
        servers: Vec<Process>,
    },
    /// The server end of a session: the name of its port, as for
    /// [`KernelObject::Session`], and the processes holding the client end.
    ServerSession {
        port_name: Option<u64>,
        clients: Vec<Process>,
    },
    /// The server end of a port, and the name it was created with.
    ServerPort {
        port_name: u64,
    },
    Thread {
        process: Option<Process>,
        tls: u64,
    },
    /// The readable end of an event, and the processes holding its writable
    /// end, who signal it.
    ReadableEvent {
        signallers: Vec<Process>,
    },
    /// Anything else: its class token and the processes holding it.
    Other {
        token: u16,
        holders: Vec<Process>,
    },
}

/// Reads of the kernel's half of the address space.
struct Kernel<'a> {
    machine: &'a Machine,
    regime: &'a TranslationRegime,
}

impl Kernel<'_> {
    fn bytes<const N: usize>(&self, address: u64) -> Option<[u8; N]> {
        let mut value = [0; N];
        (address >> 63 != 0 && self.machine.read_virtual(self.regime, address, &mut value) == N)
            .then_some(value)
    }

    fn u64(&self, address: u64) -> Option<u64> {
        self.bytes(address).map(u64::from_le_bytes)
    }

    fn u32(&self, address: u64) -> Option<u32> {
        self.bytes(address).map(u32::from_le_bytes)
    }

    fn u16(&self, address: u64) -> Option<u16> {
        self.bytes(address).map(u16::from_le_bytes)
    }

    fn u8(&self, address: u64) -> Option<u8> {
        self.bytes::<1>(address).map(|[byte]| byte)
    }

    /// A live object's class token.
    fn token(&self, object: u64) -> Option<u16> {
        (self.u32(object + layout::REFERENCES)? != 0)
            .then(|| self.u16(object + layout::CLASS_TOKEN))
            .flatten()
    }

    /// The live process objects of the kernel slid by `slide`: `None` while
    /// its process slab cannot be read (before the kernel has set it up, or
    /// not kernel 23.5.0).
    fn processes(&self, slide: u64) -> Option<impl Iterator<Item = u64> + '_> {
        let slab = layout::PROCESS_SLAB.wrapping_add(slide);
        let size = self.u64(slab + layout::SLAB_OBJECT_SIZE)?;
        let begin = self.u64(slab + layout::SLAB_BEGIN)?;
        let end = self.u64(slab + layout::SLAB_END)?;
        if size == 0 || end <= begin || (end - begin) / size > 0x10000 {
            return None;
        }
        Some(
            (begin..end)
                .step_by(size as usize)
                .filter(|&process| self.token(process) == Some(layout::PROCESS_TOKEN)),
        )
    }
}

/// The open handles of each process, by the object they stand for.
type Holders = HashMap<u64, Vec<Process>>;

impl Machine {
    /// Every thread of a process, read through `regime` (any core's: the
    /// kernel's half of the address space is the same on all) with `slide`
    /// the kernel's relocated minus image address. Empty where the kernel's
    /// objects are not as kernel 23.5.0 lays them out.
    pub(super) fn guest_threads(&self, regime: &TranslationRegime, slide: u64) -> Vec<GuestThread> {
        let kernel = Kernel {
            machine: self,
            regime,
        };
        let slab = layout::THREAD_SLAB.wrapping_add(slide);
        let (Some(size), Some(begin), Some(end)) = (
            kernel.u64(slab + layout::SLAB_OBJECT_SIZE),
            kernel.u64(slab + layout::SLAB_BEGIN),
            kernel.u64(slab + layout::SLAB_END),
        ) else {
            return Vec::new();
        };
        if size == 0 || end < begin || (end - begin) / size > 0x10000 {
            return Vec::new();
        }
        let mut found = Vec::new();
        for thread in (begin..end).step_by(size as usize) {
            if kernel.token(thread) != Some(layout::THREAD_TOKEN) {
                continue;
            }
            let Some(top) = kernel.u64(thread + layout::THREAD_KERNEL_STACK_TOP) else {
                continue;
            };
            let parameters = top.wrapping_sub(layout::STACK_PARAMETERS);
            // A thread object of the slab that is a thread names itself at
            // the top of its stack.
            if kernel.u64(parameters + STACK_TOP_THREAD) != Some(thread) {
                continue;
            }
            let Some(process) = kernel.u64(thread + THREAD_PROCESS).filter(|&p| p != 0) else {
                continue;
            };
            if let Some(described) = self.process_object(regime, process) {
                found.push((thread, parameters, process, described));
            }
        }
        let holders = self.holders(
            &kernel,
            found.iter().map(|&(_, _, process, d)| (process, d)),
        );
        found
            .iter()
            .map(|&(thread, parameters, process, described)| {
                let state = match kernel.u16(thread + layout::THREAD_STATE).unwrap_or(0xFFFF) & 0xF
                {
                    0 => ThreadState::Initialized,
                    1 => ThreadState::Waiting,
                    2 => ThreadState::Runnable,
                    3 => ThreadState::Terminated,
                    other => ThreadState::Unknown(other),
                };
                let tls = kernel.u64(thread + layout::THREAD_TLS).unwrap_or(0);
                let in_call = kernel
                    .u8(parameters + layout::PARAMETERS_FLAGS)
                    .is_some_and(|flags| flags & 1 != 0);
                let call = in_call
                    .then(|| {
                        self.thread_call(&kernel, &holders, slide, thread, parameters, process, tls)
                    })
                    .flatten();
                GuestThread {
                    id: kernel.u64(thread + layout::THREAD_ID).unwrap_or(0),
                    process: described,
                    tls,
                    state,
                    call,
                }
            })
            .collect()
    }

    /// Whether the boot's program launcher (boot2) is a process of the
    /// kernel's: `None` while the kernel's process objects cannot be read
    /// (before the kernel has set them up, or not kernel 23.5.0).
    pub(super) fn launcher_running(&self, regime: &TranslationRegime, slide: u64) -> Option<bool> {
        let kernel = Kernel {
            machine: self,
            regime,
        };
        Some(kernel.processes(slide)?.any(|process| {
            kernel.u64(process + super::kernel_layout::PROCESS_PROGRAM_ID)
                == Some(layout::LAUNCHER_PROGRAM_ID)
        }))
    }

    /// The processes running `program_id`, each with its address space
    /// (the kernel's translation regime with the process's `TTBR0_EL1`):
    /// `None` while the kernel's process objects cannot be read, as for
    /// [`Self::launcher_running`].
    pub(super) fn program_processes(
        &self,
        regime: &TranslationRegime,
        slide: u64,
        program_id: u64,
    ) -> Option<Vec<(Process, TranslationRegime)>> {
        let kernel = Kernel {
            machine: self,
            regime,
        };
        let found = kernel
            .processes(slide)?
            .filter(|&process| {
                kernel.u64(process + super::kernel_layout::PROCESS_PROGRAM_ID) == Some(program_id)
            })
            .filter_map(|process| {
                Some((
                    self.process_object(regime, process)?,
                    self.process_regime(&kernel, slide, process)?,
                ))
            })
            .collect();
        Some(found)
    }

    /// The process a process object describes, if it is readable as one.
    pub(super) fn process_object(
        &self,
        regime: &TranslationRegime,
        process: u64,
    ) -> Option<Process> {
        use super::kernel_layout::*;

        let kernel = Kernel {
            machine: self,
            regime,
        };
        Some(Process {
            id: kernel.u64(process.checked_add(PROCESS_ID)?)?,
            program_id: kernel.u64(process + PROCESS_PROGRAM_ID)?,
            name: kernel.bytes(process + PROCESS_NAME)?,
        })
    }

    /// The open handles of every process in `processes`.
    fn holders(&self, kernel: &Kernel, processes: impl Iterator<Item = (u64, Process)>) -> Holders {
        let mut seen = HashMap::new();
        for (process, described) in processes {
            seen.insert(process, described);
        }
        let mut holders = Holders::new();
        for (&process, &described) in &seen {
            let count = kernel
                .u16(process + layout::PROCESS_HANDLE_COUNT)
                .unwrap_or(0);
            for index in 0..u64::from(count) {
                let entry = process + layout::PROCESS_HANDLE_OBJECTS + 8 * index;
                if let Some(object) = kernel.u64(entry).filter(|&object| object != 0) {
                    holders.entry(object).or_default().push(described);
                }
            }
        }
        holders
    }

    /// The call the thread at `thread` is in, from the registers saved below
    /// its stack's parameter block at `parameters`.
    #[allow(clippy::too_many_arguments)]
    fn thread_call(
        &self,
        kernel: &Kernel,
        holders: &Holders,
        slide: u64,
        thread: u64,
        parameters: u64,
        process: u64,
        tls: u64,
    ) -> Option<ThreadCall> {
        let id = u16::from(kernel.u8(parameters + layout::PARAMETERS_SVC_ID)?);
        let saved = parameters - layout::SAVED_REGISTERS;
        let mut arguments = [0; 8];
        for (index, argument) in (0..).zip(&mut arguments) {
            *argument = kernel.u64(saved + 8 * index)?;
        }
        let user = self.process_regime(kernel, slide, process);
        let read_user = |address: u64, len: u64| {
            let mut data = vec![0; len.min(super::CALL_DATA_LIMIT) as usize];
            let read = user.map_or(0, |user| self.read_virtual(&user, address, &mut data));
            data.truncate(read);
            data
        };
        let message = match (
            svc::session_handle(id, &arguments),
            svc::input(id, &arguments, tls),
        ) {
            (Some(_), Some((address, len))) if address == tls => {
                // The message buffer is read where the kernel keeps it.
                let kernel_tls = kernel.u64(thread + layout::THREAD_TLS_KERNEL)?;
                let mut data = vec![0; len as usize];
                let read = self.read_virtual(kernel.regime, kernel_tls, &mut data);
                data.truncate(read);
                data
            }
            (Some(_), Some((address, len))) => read_user(address, len),
            _ => Vec::new(),
        };
        let handles: Vec<u32> = match id {
            WAIT_SYNCHRONIZATION => {
                let count = arguments[2].min(layout::MOST_WAITED);
                read_user(arguments[1], 4 * count)
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|&word| u32::from_le_bytes(word))
                    .collect()
            }
            ARBITRATE_LOCK => vec![arguments[0] as u32],
            _ => svc::session_handle(id, &arguments).into_iter().collect(),
        };
        let objects = handles
            .into_iter()
            .map(|handle| {
                (
                    handle,
                    self.handle_object(kernel, holders, process, thread, handle),
                )
            })
            .collect();
        Some(ThreadCall {
            id,
            arguments,
            pc: kernel.u64(saved + layout::SAVED_ELR)?.wrapping_sub(4),
            lr: kernel.u64(saved + layout::SAVED_X29 + 8)?,
            fp: kernel.u64(saved + layout::SAVED_X29)?,
            message,
            objects,
        })
    }

    /// The translation regime of the process at `process`: the kernel's
    /// with the process's `TTBR0_EL1`.
    fn process_regime(
        &self,
        kernel: &Kernel,
        slide: u64,
        process: u64,
    ) -> Option<TranslationRegime> {
        let slab = layout::PROCESS_SLAB.wrapping_add(slide);
        let size = kernel
            .u64(slab + layout::SLAB_OBJECT_SIZE)
            .filter(|&size| size != 0)?;
        let offset = process.checked_sub(kernel.u64(slab + layout::SLAB_ALLOCATION_BEGIN)?)?;
        if offset % size != 0 {
            return None;
        }
        let ttbrs = kernel.u64(layout::PROCESS_TTBRS.wrapping_add(slide))?;
        let ttbr0 = kernel.u64(ttbrs + 8 * (offset / size + 1))?;
        Some(TranslationRegime {
            ttbr0,
            ..*kernel.regime
        })
    }

    /// What `handle` of the process at `process` stands for; `thread` is
    /// the thread whose call names it.
    fn handle_object(
        &self,
        kernel: &Kernel,
        holders: &Holders,
        process: u64,
        thread: u64,
        handle: u32,
    ) -> KernelObject {
        let object = match handle {
            layout::CURRENT_THREAD => Some(thread),
            _ => self.table_entry(kernel, process, handle),
        };
        let Some(object) = object else {
            return KernelObject::NotOpen;
        };
        let held = |object: u64| holders.get(&object).cloned().unwrap_or_default();
        // The port a session came from; none for one `CreateSession` made.
        let port_of = |session: u64| {
            kernel
                .u64(session + layout::SESSION_CLIENT_PORT)
                .filter(|&client| client != 0)
                .map(|client| client.wrapping_sub(layout::PORT_CLIENT))
        };
        let name_of =
            |port: Option<u64>| port.map(|port| kernel.u64(port + layout::PORT_NAME).unwrap_or(0));
        match kernel.token(object) {
            Some(layout::CLIENT_SESSION_TOKEN) => {
                let session = kernel
                    .u64(object + layout::CLIENT_SESSION_PARENT)
                    .unwrap_or(0);
                let port = port_of(session);
                let mut servers = held(session + layout::SESSION_SERVER);
                if servers.is_empty()
                    && let Some(port) = port
                {
                    servers = held(port + layout::PORT_SERVER);
                }
                KernelObject::Session {
                    port_name: name_of(port),
                    servers,
                }
            }
            Some(layout::SERVER_SESSION_TOKEN) => {
                let session = object.wrapping_sub(layout::SESSION_SERVER);
                KernelObject::ServerSession {
                    port_name: name_of(port_of(session)),
                    clients: held(session + layout::SESSION_CLIENT),
                }
            }
            Some(layout::SERVER_PORT_TOKEN) => KernelObject::ServerPort {
                port_name: name_of(Some(object.wrapping_sub(layout::PORT_SERVER))).unwrap_or(0),
            },
            Some(layout::THREAD_TOKEN) => KernelObject::Thread {
                process: kernel
                    .u64(object + THREAD_PROCESS)
                    .and_then(|process| self.process_object(kernel.regime, process)),
                tls: kernel.u64(object + layout::THREAD_TLS).unwrap_or(0),
            },
            Some(layout::READABLE_EVENT_TOKEN) => {
                let event = object.wrapping_sub(layout::EVENT_READABLE);
                KernelObject::ReadableEvent {
                    signallers: match kernel.token(event) {
                        Some(layout::EVENT_TOKEN) => held(event),
                        _ => Vec::new(),
                    },
                }
            }
            token => KernelObject::Other {
                token: token.unwrap_or(0),
                holders: held(object),
            },
        }
    }

    /// The object `handle` names in the handle table of the process at
    /// `process`, checked as the kernel checks it.
    fn table_entry(&self, kernel: &Kernel, process: u64, handle: u32) -> Option<u64> {
        let index = u64::from(handle & 0x7FFF);
        if handle >> 30 != 0 || handle < 0x8000 {
            return None;
        }
        let count = kernel.u16(process + layout::PROCESS_HANDLE_COUNT)?;
        let linear = kernel.u16(process + layout::PROCESS_HANDLE_IDS + 2 * index)?;
        (index < u64::from(count) && u32::from(linear) == handle >> 15)
            .then(|| kernel.u64(process + layout::PROCESS_HANDLE_OBJECTS + 8 * index))
            .flatten()
            .filter(|&object| object != 0)
    }
}
