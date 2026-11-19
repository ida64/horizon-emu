//! Video decoder (NVDEC 2.0, host1x class 0xF0): its THI registers, its
//! methods and the firmware nvservices boots it with.
//!
//! # Firmware
//!
//! nvservices 23.5.0 boots the engine when a client first opens it
//! (`0x835e0`, the same host falcon interface as the VIC's,
//! [`HostFalcon`](super::falcon_host::HostFalcon)): it waits for DMACTL
//! bits 1-2 (memory scrubbing) to clear, points DMATRFBASE at its pinned
//! copy of the embedded `nvhost_nvdec020_ns.fw` (looked up at `0x83820`;
//! table entry `0x199180`: 0x20000 bytes at `0x142000`), transfers the
//! image's data segment (0x400 bytes) to DMEM 0 and the first 256 bytes of
//! its code segment to IMEM 0, unmasks interrupts, starts CPUCTL and waits
//! for IDLESTATE to read 0. The image has the falcon "bin header" Linux's
//! `drivers/gpu/host1x`-era `falcon.c` parses (magic 0x10DE, version 1,
//! OS header at 0x100, OS data at 0x200: code 0..0x300, data 0x1F600..
//! 0x1FA00). Only that firmware is known: the start is refused unless IMEM
//! and DMEM hold exactly what it puts there ([`IMEM_SHA256`],
//! [`DMEM_SHA256`]). No Falcon instruction is executed; the decoder it
//! would run is the model below.
//!
//! # Methods
//!
//! Method offsets and the codec numbers are NVIDIA's public
//! `open-gpu-doc` `classes/video/clc5b0.h` (`NVC5B0_VIDEO_DECODER`, a
//! later NVDEC whose method layout is taken to be NVDEC 2.0's
//! \[assumption\]). They arrive as on the VIC: THI word 0x10 (method offset
//! in words) and 0x11 (data), and THI word 0 increments a syncpoint (Linux
//! `hw_host1x04_uclass.h`). Buffer addresses are IOVA >> 8 in the NVDEC's
//! SMMU address space.
//!
//! EXECUTE is refused: decoding is not modeled. The refusal names the
//! methods set, from which the submission can be replayed.

use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{Notes, falcon_host::Engine, host1x::Syncpoints, mc::DeviceMemory};
use crate::snapshot::{Snapshot, SnapshotError, decode, encode};

/// Host1x class of the NVDEC (Linux `include/linux/host1x.h`
/// `HOST1X_CLASS_NVDEC`).
pub const CLASS: u32 = 0xF0;

/// SHA-256 of the 64 KiB IMEM once nvservices 23.5.0 has loaded
/// `nvhost_nvdec020_ns.fw` (its first 256 code bytes at 0, zero after).
pub const IMEM_SHA256: &str = "7684223bc3a0a63c02c9e98c87ee1ad0f782a503e71095660576404c8b56aef9";
/// SHA-256 of the 64 KiB DMEM then (the 0x400-byte data segment at 0,
/// zero after).
pub const DMEM_SHA256: &str = "de2f256064a0af797747c2b97505dc0b9f3df0de4f489eac731c23ae9ca9cc31";

/// THI registers, word offsets (as the VIC's, libdrm `vic.h`).
const THI_INCR_SYNCPT: u32 = 0x00;
const THI_METHOD_OFFSET: u32 = 0x10;
const THI_METHOD_DATA: u32 = 0x11;

/// Conditions of a THI syncpoint increment (Linux `hw_host1x04_uclass.h`):
/// immediate, operation done, read done, register write safe. EXECUTE
/// finishes within the method that starts it, so each holds at once.
const INCREMENT_CONDITIONS: u32 = 4;

/// Method word indices (byte offset / 4), `clc5b0.h` names.
mod method {
    pub const SET_APPLICATION_ID: usize = 0x200 / 4;
    pub const SET_WATCHDOG_TIMER: usize = 0x204 / 4;
    pub const EXECUTE: usize = 0x300 / 4;
    /// `SET_CONTROL_PARAMS` 0x400 to `SET_EXTERNAL_MVBUFFER_OFFSET` 0x4BC:
    /// the codec, the picture setup, bitstream and output buffers.
    pub const SETUP: std::ops::RangeInclusive<usize> = 0x400 / 4..=0x4BC / 4;
    pub const H264_SET_MBHIST_BUF_OFFSET: usize = 0x500 / 4;
    /// `VP9_SET_PROB_TAB_BUF_OFFSET` 0x5C0 to
    /// `VP9_SET_FILTER_BUFFER_OFFSET` 0x5DC.
    pub const VP9: std::ops::RangeInclusive<usize> = 0x5C0 / 4..=0x5DC / 4;
    pub const COUNT: usize = 0x600 / 4;
}

/// The NVDEC engine behind the THI.
pub struct Nvdec {
    syncpoints: Arc<Syncpoints>,
    notes: Arc<Notes>,
    state: Mutex<State>,
}

#[derive(Serialize, Deserialize)]
struct State {
    offset: u32,
    #[serde(with = "crate::snapshot::array")]
    methods: [u32; method::COUNT],
    /// Whether the firmware was started.
    started: bool,
}

impl Nvdec {
    /// `memory` is memory as the NVDEC's SMMU client reaches it.
    pub fn new(_memory: DeviceMemory, syncpoints: Arc<Syncpoints>, notes: Arc<Notes>) -> Self {
        Self {
            syncpoints,
            notes,
            state: Mutex::new(State {
                offset: 0,
                methods: [0; method::COUNT],
                started: false,
            }),
        }
    }

    fn method(&self, value: u32) -> Option<()> {
        let mut state = self.state.lock().unwrap();
        if !state.started {
            return self.refuse(format!(
                "method {:#x} = {value:#x} before the firmware was started",
                state.offset * 4
            ));
        }
        let index = state.offset as usize;
        let accepted = matches!(
            index,
            method::SET_APPLICATION_ID | method::SET_WATCHDOG_TIMER
        ) || method::SETUP.contains(&index)
            || index == method::H264_SET_MBHIST_BUF_OFFSET
            || method::VP9.contains(&index);
        if index == method::EXECUTE {
            let methods = state.methods;
            drop(state);
            return self.refuse(format!(
                "EXECUTE {value:#x}: decoding is not modeled; {}",
                dump(&methods)
            ));
        }
        if !accepted {
            return self.refuse(format!("method {:#x} = {value:#x}", index * 4));
        }
        state.methods[index] = value;
        Some(())
    }

    fn refuse(&self, reason: String) -> Option<()> {
        self.notes.add(format!("NVDEC: {reason}"));
        None
    }
}

impl Engine for Nvdec {
    const NAME: &'static str = "NVDEC";
    const CLASS: u32 = CLASS;

    fn thi_write(&self, offset: u32, value: u32) -> Option<()> {
        match offset {
            THI_INCR_SYNCPT => {
                let (index, condition) = ((value & 0xFF) as usize, value >> 8);
                if condition >= INCREMENT_CONDITIONS {
                    return self.refuse(format!("syncpoint increment {value:#x}"));
                }
                if self.syncpoints.get(index).is_none() {
                    return self.refuse(format!("increment of syncpoint {index}"));
                }
                self.syncpoints.add(index, 1);
                Some(())
            }
            THI_METHOD_OFFSET => {
                self.state.lock().unwrap().offset = value;
                Some(())
            }
            THI_METHOD_DATA => self.method(value),
            _ => self.refuse(format!("THI word {offset:#x} = {value:#x}")),
        }
    }

    fn method_offset(&self) -> u32 {
        self.state.lock().unwrap().offset
    }

    fn start(&self, imem: &[u8], dmem: &[u8]) -> Result<(), String> {
        let digest = |bytes: &[u8]| {
            Sha256::digest(bytes)
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        };
        let (code, data) = (digest(imem), digest(dmem));
        if code != IMEM_SHA256 || data != DMEM_SHA256 {
            return Err(format!(
                "unknown firmware (IMEM SHA-256 {code}, DMEM SHA-256 {data}); only \
                 nvservices 23.5.0's nvhost_nvdec020_ns.fw is modeled"
            ));
        }
        self.state.lock().unwrap().started = true;
        Ok(())
    }
}

/// Saves the THI method offset, the methods set and whether the firmware
/// runs. Saved by its host, the
/// [`HostFalcon`](super::falcon_host::HostFalcon).
impl Snapshot for Nvdec {
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        encode(&*self.state.lock().unwrap())
    }

    fn restore(&self, state: &[u8]) -> Result<(), SnapshotError> {
        *self.state.lock().unwrap() = decode("NVDEC", state)?;
        Ok(())
    }
}

/// The non-zero methods, for a refusal note.
fn dump(methods: &[u32; method::COUNT]) -> String {
    let set: Vec<String> = (methods.iter().enumerate())
        .filter(|(_, value)| **value != 0)
        .map(|(i, value)| format!("{:03x}={value:x}", i * 4))
        .collect();
    format!("methods {}", set.join(" "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{boot::DramSize, memory::GuestRam, soc::mc::Mc};

    fn nvdec() -> (Nvdec, Arc<Notes>) {
        let ram = Arc::new(GuestRam::new(0x8000_0000, 0x1000).unwrap());
        let mc = Arc::new(Mc::new(DramSize::Gb4));
        let notes: Arc<Notes> = Arc::default();
        let memory = DeviceMemory::new(&mc, &ram, crate::soc::mc::SMMU_NVDEC_ASID);
        let syncpoints = Arc::new(Syncpoints::default());
        (Nvdec::new(memory, syncpoints, Arc::clone(&notes)), notes)
    }

    #[test]
    fn only_the_known_firmware_starts_it_and_methods_wait_for_it() {
        let (nvdec, _) = nvdec();
        nvdec.thi_write(THI_METHOD_OFFSET, 0x200 / 4).unwrap();
        assert_eq!(nvdec.thi_write(THI_METHOD_DATA, 3), None);
        let zero = [0u8; 0x10000];
        assert!(
            nvdec
                .start(&zero, &zero)
                .unwrap_err()
                .contains("unknown firmware")
        );
        nvdec.state.lock().unwrap().started = true;
        nvdec.thi_write(THI_METHOD_DATA, 3).unwrap();
        assert_eq!(nvdec.method_offset(), 0x200 / 4);
        nvdec.thi_write(THI_METHOD_OFFSET, 0x240 / 4).unwrap();
        assert_eq!(nvdec.thi_write(THI_METHOD_DATA, 1), None);
    }
}
