//! Horizon supervisor calls, for traces.
//!
//! Names are the publicly documented ones for each `SVC` immediate; they
//! label trace lines and are not derived from the kernel image. Which
//! register carries what, for the calls [`input`] and [`output`] know, is
//! read off the kernel's 64-bit call wrappers and cited there.

pub const EXIT_PROCESS: u16 = 0x07;
pub const CLOSE_HANDLE: u16 = 0x16;
pub const CONNECT_TO_NAMED_PORT: u16 = 0x1F;
pub const SEND_SYNC_REQUEST: u16 = 0x21;
pub const SEND_SYNC_REQUEST_WITH_USER_BUFFER: u16 = 0x22;
pub const SEND_ASYNC_REQUEST_WITH_USER_BUFFER: u16 = 0x23;
pub const BREAK: u16 = 0x26;
pub const OUTPUT_DEBUG_STRING: u16 = 0x27;
pub const CREATE_PROCESS: u16 = 0x79;

/// Size of the message buffer every thread has at the start of its
/// thread-local region, which `SendSyncRequest` sends and receives in: the
/// length the kernel assumes for a message without a user buffer (image
/// 0x8009fdb0), and the one the initial processes pass wherever they name the
/// buffer (ProcessMana text 0xa060, sm text 0x1c0).
pub const MESSAGE_BUFFER_SIZE: u64 = 0x100;

/// Longest name `ConnectToNamedPort` reads (kernel image 0x800be320 copies at
/// most this many bytes).
pub const PORT_NAME_SIZE: u64 = 12;

/// Size of the parameter block of `CreateProcess` (kernel image 0x800bf230
/// copies this many bytes).
pub const CREATE_PROCESS_PARAMETER_SIZE: u64 = 0x30;

/// The caller's memory that call `id` hands to the kernel, as address and
/// length, for the calls whose point is that memory. `arguments` are `X0`-`X7`
/// at the call and `thread_local_region` is the calling thread's
/// `TPIDRRO_EL0`.
///
/// The registers are those the kernel's wrappers pass on: `SendSyncRequest`
/// (image 0x800b6d88) takes only a handle and uses the thread's message
/// buffer; `SendSyncRequestWithUserBuffer` (0x800b6db4) takes buffer, size,
/// handle; `SendAsyncRequestWithUserBuffer` (0x800b6de0) and `CreateProcess`
/// (0x800b7dac) reserve `X0` for an output and take their inputs from `X1`
/// on, as does `ConnectToNamedPort` (0x800b6d4c). `Break` and
/// `OutputDebugString` are as publicly documented.
pub fn input(id: u16, arguments: &[u64; 8], thread_local_region: u64) -> Option<(u64, u64)> {
    Some(match id {
        BREAK => (arguments[1], arguments[2]),
        OUTPUT_DEBUG_STRING => (arguments[0], arguments[1]),
        CONNECT_TO_NAMED_PORT => (arguments[1], PORT_NAME_SIZE),
        CREATE_PROCESS => (arguments[1], CREATE_PROCESS_PARAMETER_SIZE),
        SEND_SYNC_REQUEST => (thread_local_region, MESSAGE_BUFFER_SIZE),
        SEND_SYNC_REQUEST_WITH_USER_BUFFER => (arguments[0], arguments[1]),
        SEND_ASYNC_REQUEST_WITH_USER_BUFFER => (arguments[1], arguments[2]),
        _ => return None,
    })
}

/// The caller's memory that call `id` has filled in when it returns: the
/// reply to a synchronous request, which replaces the request in the same
/// buffer. Parameters as for [`input`], with the registers of the call, not
/// of the return.
pub fn output(id: u16, arguments: &[u64; 8], thread_local_region: u64) -> Option<(u64, u64)> {
    match id {
        SEND_SYNC_REQUEST | SEND_SYNC_REQUEST_WITH_USER_BUFFER => {
            input(id, arguments, thread_local_region)
        }
        _ => None,
    }
}

/// The session handle a request-sending call names, from the same wrappers
/// as [`input`].
pub fn session_handle(id: u16, arguments: &[u64; 8]) -> Option<u32> {
    Some(match id {
        SEND_SYNC_REQUEST => arguments[0],
        SEND_SYNC_REQUEST_WITH_USER_BUFFER => arguments[2],
        SEND_ASYNC_REQUEST_WITH_USER_BUFFER => arguments[3],
        _ => return None,
    } as u32)
}

/// The text in `bytes` up to its first NUL: a name as the calls here pass
/// names, in a field of fixed size.
pub fn text(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&byte| byte == 0);
    String::from_utf8_lossy(&bytes[..end.unwrap_or(bytes.len())]).into_owned()
}

/// Name and program ID of the process a `CreateProcess` call describes, given
/// its parameter block: the first 12 bytes and the eight at 0x10, which is
/// where the kernel takes them from (image 0x800948a0). `None` if the block
/// is cut short.
pub fn created_process(parameters: &[u8]) -> Option<(String, u64)> {
    let program_id = parameters.get(0x10..)?.first_chunk()?;
    Some((text(&parameters[..12]), u64::from_le_bytes(*program_id)))
}

/// Name of supervisor call `id`, if it is a documented one.
pub fn name(id: u16) -> Option<&'static str> {
    Some(match id {
        0x01 => "SetHeapSize",
        0x02 => "SetMemoryPermission",
        0x03 => "SetMemoryAttribute",
        0x04 => "MapMemory",
        0x05 => "UnmapMemory",
        0x06 => "QueryMemory",
        0x07 => "ExitProcess",
        0x08 => "CreateThread",
        0x09 => "StartThread",
        0x0A => "ExitThread",
        0x0B => "SleepThread",
        0x0C => "GetThreadPriority",
        0x0D => "SetThreadPriority",
        0x0E => "GetThreadCoreMask",
        0x0F => "SetThreadCoreMask",
        0x10 => "GetCurrentProcessorNumber",
        0x11 => "SignalEvent",
        0x12 => "ClearEvent",
        0x13 => "MapSharedMemory",
        0x14 => "UnmapSharedMemory",
        0x15 => "CreateTransferMemory",
        0x16 => "CloseHandle",
        0x17 => "ResetSignal",
        0x18 => "WaitSynchronization",
        0x19 => "CancelSynchronization",
        0x1A => "ArbitrateLock",
        0x1B => "ArbitrateUnlock",
        0x1C => "WaitProcessWideKeyAtomic",
        0x1D => "SignalProcessWideKey",
        0x1E => "GetSystemTick",
        0x1F => "ConnectToNamedPort",
        0x20 => "SendSyncRequestLight",
        0x21 => "SendSyncRequest",
        0x22 => "SendSyncRequestWithUserBuffer",
        0x23 => "SendAsyncRequestWithUserBuffer",
        0x24 => "GetProcessId",
        0x25 => "GetThreadId",
        0x26 => "Break",
        0x27 => "OutputDebugString",
        0x28 => "ReturnFromException",
        0x29 => "GetInfo",
        0x2A => "FlushEntireDataCache",
        0x2B => "FlushDataCache",
        0x2C => "MapPhysicalMemory",
        0x2D => "UnmapPhysicalMemory",
        0x2E => "GetDebugFutureThreadInfo",
        0x2F => "GetLastThreadInfo",
        0x30 => "GetResourceLimitLimitValue",
        0x31 => "GetResourceLimitCurrentValue",
        0x32 => "SetThreadActivity",
        0x33 => "GetThreadContext3",
        0x34 => "WaitForAddress",
        0x35 => "SignalToAddress",
        0x36 => "SynchronizePreemptionState",
        0x37 => "GetResourceLimitPeakValue",
        0x39 => "CreateIoPool",
        0x3A => "CreateIoRegion",
        0x3C => "KernelDebug",
        0x3D => "ChangeKernelTraceState",
        0x40 => "CreateSession",
        0x41 => "AcceptSession",
        0x42 => "ReplyAndReceiveLight",
        0x43 => "ReplyAndReceive",
        0x44 => "ReplyAndReceiveWithUserBuffer",
        0x45 => "CreateEvent",
        0x46 => "MapIoRegion",
        0x47 => "UnmapIoRegion",
        0x48 => "MapPhysicalMemoryUnsafe",
        0x49 => "UnmapPhysicalMemoryUnsafe",
        0x4A => "SetUnsafeLimit",
        0x4B => "CreateCodeMemory",
        0x4C => "ControlCodeMemory",
        0x4D => "SleepSystem",
        0x4E => "ReadWriteRegister",
        0x4F => "SetProcessActivity",
        0x50 => "CreateSharedMemory",
        0x51 => "MapTransferMemory",
        0x52 => "UnmapTransferMemory",
        0x53 => "CreateInterruptEvent",
        0x54 => "QueryPhysicalAddress",
        0x55 => "QueryMemoryMapping",
        0x56 => "CreateDeviceAddressSpace",
        0x57 => "AttachDeviceAddressSpace",
        0x58 => "DetachDeviceAddressSpace",
        0x59 => "MapDeviceAddressSpaceByForce",
        0x5A => "MapDeviceAddressSpaceAligned",
        0x5C => "UnmapDeviceAddressSpace",
        0x5D => "InvalidateProcessDataCache",
        0x5E => "StoreProcessDataCache",
        0x5F => "FlushProcessDataCache",
        0x60 => "DebugActiveProcess",
        0x61 => "BreakDebugProcess",
        0x62 => "TerminateDebugProcess",
        0x63 => "GetDebugEvent",
        0x64 => "ContinueDebugEvent",
        0x65 => "GetProcessList",
        0x66 => "GetThreadList",
        0x67 => "GetDebugThreadContext",
        0x68 => "SetDebugThreadContext",
        0x69 => "QueryDebugProcessMemory",
        0x6A => "ReadDebugProcessMemory",
        0x6B => "WriteDebugProcessMemory",
        0x6C => "SetHardwareBreakPoint",
        0x6D => "GetDebugThreadParam",
        0x6F => "GetSystemInfo",
        0x70 => "CreatePort",
        0x71 => "ManageNamedPort",
        0x72 => "ConnectToPort",
        0x73 => "SetProcessMemoryPermission",
        0x74 => "MapProcessMemory",
        0x75 => "UnmapProcessMemory",
        0x76 => "QueryProcessMemory",
        0x77 => "MapProcessCodeMemory",
        0x78 => "UnmapProcessCodeMemory",
        0x79 => "CreateProcess",
        0x7A => "StartProcess",
        0x7B => "TerminateProcess",
        0x7C => "GetProcessInfo",
        0x7D => "CreateResourceLimit",
        0x7E => "SetResourceLimitLimitValue",
        0x7F => "CallSecureMonitor",
        0x90 => "MapInsecurePhysicalMemory",
        0x91 => "UnmapInsecurePhysicalMemory",
        _ => return None,
    })
}
