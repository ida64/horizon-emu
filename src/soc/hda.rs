//! The HD Audio controller and the SoC's internal HDMI/DP audio codec it
//! talks to: the console's audio path to a TV, with nothing docked.
//!
//! # What `audio` does (text `0x1ec10` onwards)
//!
//! It maps the controller's PCI-style wrapper at `0x70030000` and the
//! controller's own registers at `0x70038000` (text `0x1e9b0`), and:
//!
//! * enables the wrapper (`text 0x1ef20`): sets bit 0 of `0x180`, writes 64
//!   to `0x80`, 6 to the configuration command register (`0x1004`) and
//!   `0x4000` to the configuration BAR (`0x1010`); later it writes `0x10000`
//!   to the interrupt mask (`0x188`). Register names as in the public Linux
//!   `hda_tegra.c` (`HDA_IPFS_CONFIG`, `HDA_IPFS_FPCI_BAR0`, `HDA_CFG_CMD`,
//!   `HDA_CFG_BAR0`, `HDA_IPFS_INTR_MASK`), which programs the same values;
//! * resets the controller through `GCTL` and polls for the reset bit, reads
//!   `GCAP` (stream counts), the codec presence bits (`STATESTS`) and the
//!   command ring sizes, and sets up the command outbound ring (CORB, 256
//!   entries) and response inbound ring (RIRB) in memory it maps for the
//!   controller (text `0x1f050`); then it enables interrupts (`INTCTL`) and
//!   resets the stream it will use, the first output stream;
//! * talks to the codec at the first present address: each command is a
//!   CORB entry and a `CORBWP` write, and it waits (without a time limit)
//!   for the response its interrupt handler takes from the RIRB (text
//!   `0x20b80`; interrupt 113).
//!
//! The register layout and ring behaviour are those of the public Intel
//! High Definition Audio specification. A response interrupt comes after
//! every batch of responses: the specification raises it after `RINTCNT`
//! responses or at the first frame with no response, whichever is first,
//! and `audio` sets `RINTCNT` to 192 and waits for every single response.
//!
//! # The codec (text `0x1f240`, `0x20410`)
//!
//! `audio` asks the root node for its vendor ID, revision and subordinate
//! nodes, takes the first node whose function group type is 1 (audio),
//! resets and powers it, asks it for its subordinate nodes and takes the
//! first audio output converter (widget type 0) and the first pin (type 4)
//! by their capabilities. It then configures both with set commands (which
//! answer 0, as set commands do), and finally reads the pin's sense: with
//! presence detect and ELD valid both set it would read the monitor's ELD
//! (`0xF2F` commands); nothing is docked, so the pin senses nothing and
//! the codec never sends an unsolicited response.
//!
//! What the codec reports, and the basis:
//!
//! * vendor and device ID `0x10DE0029`, the public Linux `patch_hdmi.c`
//!   ID of the "Tegra210 HDMI/DP" codec; revision `0x100000` (HDA 1.0,
//!   revision 0), an assumption `audio` does not check;
//! * address 3 on the link: public Jetson TX1 Linux systems list the
//!   internal codec as codec #3 (recalled, not verified);
//! * one audio function group (node 1) holding one converter (node 2) and
//!   one pin (node 3): an assumption. `audio` takes the first function
//!   group, converter and pin wherever they are, so their numbers do not
//!   matter to it. The capability bits besides the widget type (digital,
//!   stereo, unsolicited-capable pin) are what an HDMI codec reports and
//!   are not checked.
//!
//! Commands other than the parameters above, the pin sense and set
//! commands refuse. No audio path is modeled: stream descriptors only
//! support their reset bit, which is all start-up uses.
use std::sync::{Arc, Mutex};

use super::{Notes, mc::DeviceMemory, regs::RegisterFile};
use crate::bus::MmioDevice;

/// The wrapper and configuration space, then the controller at `+0x8000`.
pub const BASE: u64 = 0x7003_0000;
pub const SIZE: u64 = 0x1_0000;
const CONTROLLER: u64 = 0x8000;
/// `audio` creates its interrupt event for interrupt 113 (text `0x1ec10`).
pub const INTID: u32 = 113;
/// The `hda` SMMU client, from the public Linux `tegra210.c` swgroups.
pub const SMMU_ASID: u64 = 0x254;

/// `GCAP`, `VMIN`, `VMAJ`: 64-bit addressing, no bidirectional streams,
/// four input and four output streams, specification version 1.0. The
/// stream counts are an assumption; `audio` uses the first output stream,
/// numbered after the input streams.
const CAPABILITIES: u32 = 0x0100_0000 | 4 << 12 | 4 << 8 | 1;
const INPUT_STREAMS: u64 = 4;
const STREAMS: u64 = 8;
/// The codec's link address.
const CODEC: u32 = 3;
/// `CORBSIZE`/`RIRBSIZE` capability: only 256 entries.
const RING_CAPABILITY: u32 = 0x4 << 4;
const RING_ENTRIES: u32 = 256;

const GCTL_RESET: u32 = 1;
const INTCTL_GLOBAL: u32 = 1 << 31;
const INTCTL_CONTROLLER: u32 = 1 << 30;
const CORB_RUN: u32 = 1 << 1;
const RIRB_INTERRUPT: u32 = 1 << 0;
const RIRB_RUN: u32 = 1 << 1;
const RIRB_RESPONSE: u32 = 1 << 0;

#[derive(Default)]
struct State {
    control: u32,
    wake_enable: u32,
    interrupt_control: u32,
    corb_base: [u32; 2],
    corb_write: u32,
    corb_read: u32,
    corb_control: u32,
    corb_size: u32,
    rirb_base: [u32; 2],
    rirb_write: u32,
    response_count: u32,
    rirb_control: u32,
    rirb_status: u32,
    rirb_size: u32,
    streams: [u32; STREAMS as usize],
}

pub struct Hda {
    state: Mutex<State>,
    wrapper: RegisterFile,
    memory: DeviceMemory,
    interrupt: Box<dyn Fn(bool) + Send + Sync>,
    notes: Arc<Notes>,
}

impl Hda {
    /// The wrapper registers start at zero, an assumption: `audio`, like
    /// Linux, sets the bits it needs. The controller is in reset.
    pub fn new(
        memory: DeviceMemory,
        interrupt: Box<dyn Fn(bool) + Send + Sync>,
        notes: Arc<Notes>,
    ) -> Self {
        Self {
            state: Mutex::default(),
            wrapper: RegisterFile::strict(
                [0x80, 0x180, 0x188, 0x1004, 0x1010].map(|register| (register, 0)),
            ),
            memory,
            interrupt,
            notes,
        }
    }

    fn update(&self, state: &State) {
        (self.interrupt)(interrupt_status(state) & INTCTL_GLOBAL != 0);
    }

    /// Runs the commands between the CORB's read and write pointers.
    fn run_commands(&self, state: &mut State) -> Result<(), String> {
        if state.corb_control & CORB_RUN == 0 || state.rirb_control & RIRB_RUN == 0 {
            return Ok(());
        }
        let corb = u64::from(state.corb_base[1]) << 32 | u64::from(state.corb_base[0]);
        let rirb = u64::from(state.rirb_base[1]) << 32 | u64::from(state.rirb_base[0]);
        while state.corb_read != state.corb_write {
            state.corb_read = (state.corb_read + 1) % RING_ENTRIES;
            let mut bytes = [0; 4];
            self.memory
                .read(corb + u64::from(state.corb_read) * 4, &mut bytes)
                .ok_or("the command ring is not mapped")?;
            let command = u32::from_le_bytes(bytes);
            let response = respond(command)?;
            state.rirb_write = (state.rirb_write + 1) % RING_ENTRIES;
            let mut entry = [0; 8];
            entry[..4].copy_from_slice(&response.to_le_bytes());
            entry[4..].copy_from_slice(&CODEC.to_le_bytes());
            self.memory
                .write(rirb + u64::from(state.rirb_write) * 8, &entry)
                .ok_or("the response ring is not mapped")?;
            state.rirb_status |= RIRB_RESPONSE;
        }
        Ok(())
    }
}

fn interrupt_status(state: &State) -> u32 {
    let controller =
        state.rirb_control & RIRB_INTERRUPT != 0 && state.rirb_status & RIRB_RESPONSE != 0;
    let mut status = 0;
    if controller {
        status |= INTCTL_CONTROLLER;
    }
    if status & state.interrupt_control != 0 && state.interrupt_control & INTCTL_GLOBAL != 0 {
        status |= INTCTL_GLOBAL;
    }
    status
}

/// The codec's answer to `command` (see the module documentation).
fn respond(command: u32) -> Result<u32, String> {
    let (codec, node, verb) = (command >> 28, command >> 20 & 0x7F, command & 0xF_FFFF);
    if codec != CODEC {
        return Err(format!(
            "command {command:#010x} for a codec that is not there"
        ));
    }
    let response = match (node, verb >> 8, verb & 0xFF) {
        (0, 0xF00, 0x00) => 0x10DE_0029,
        (0, 0xF00, 0x02) => 0x0010_0000,
        // Subordinate nodes: starting number in bits 23:16, count in 7:0.
        (0, 0xF00, 0x04) => 1 << 16 | 1,
        (1, 0xF00, 0x04) => 2 << 16 | 2,
        // Function group type 1: audio, unsolicited responses capable.
        (1, 0xF00, 0x05) => 1 << 8 | 1,
        // Widget capabilities: type in bits 23:20; digital, stereo.
        (2, 0xF00, 0x09) => 1 << 9 | 1,
        // A pin, also unsolicited-capable.
        (3, 0xF00, 0x09) => 4 << 20 | 1 << 9 | 1 << 7 | 1,
        // Pin sense: no presence detect, no valid ELD.
        (3, 0xF09, _) => 0,
        // Set commands: 12-bit ones `0x7xx`, 4-bit ones 2 (converter
        // format) and 3 (amplifier gain).
        (1..=3, 0x700..=0x7FF | 0x200..=0x3FF, _) => 0,
        _ => return Err(format!("codec command {command:#010x} is not modeled")),
    };
    Ok(response)
}

impl MmioDevice for Hda {
    fn read(&self, core: usize, offset: u64, size: u8) -> Option<u64> {
        if offset < CONTROLLER {
            return self.wrapper.read(core, offset, size);
        }
        let state = self.state.lock().unwrap();
        let value = match (offset - CONTROLLER, size) {
            (0x00, 4) => CAPABILITIES,
            (0x08, 4) => state.control,
            // WAKEEN, then STATESTS: the codec is present once the link is
            // out of reset.
            (0x0C, 4) => {
                let present = if state.control & GCTL_RESET != 0 {
                    1 << CODEC
                } else {
                    0
                };
                state.wake_enable | present << 16
            }
            (0x20, 4) => state.interrupt_control,
            (0x24, 4) => interrupt_status(&state),
            (0x40, 4) => state.corb_base[0],
            (0x44, 4) => state.corb_base[1],
            (0x48, 4) => state.corb_write | state.corb_read << 16,
            (0x4C, 4) => state.corb_control | (RING_CAPABILITY | state.corb_size) << 16,
            (0x50, 4) => state.rirb_base[0],
            (0x54, 4) => state.rirb_base[1],
            (0x58, 4) => state.rirb_write | state.response_count << 16,
            (0x5C, 4) => {
                state.rirb_control
                    | state.rirb_status << 8
                    | (RING_CAPABILITY | state.rirb_size) << 16
            }
            (offset @ 0x80..0x180, 4) if offset % 0x20 == 0 => {
                state.streams[((offset - 0x80) / 0x20) as usize]
            }
            _ => return None,
        };
        Some(u64::from(value))
    }

    fn write(&self, core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        if offset < CONTROLLER {
            return self.wrapper.write(core, offset, size, value);
        }
        let mut state = self.state.lock().unwrap();
        let low = value as u32;
        match (offset - CONTROLLER, size) {
            // Leaving reset clears the controller's state, ready for set-up.
            (0x08, 4) => {
                if low & GCTL_RESET == 0 {
                    let control = low;
                    *state = State::default();
                    state.control = control;
                } else {
                    state.control = low & 0x103;
                }
            }
            (0x0C, 4) => state.wake_enable = low & 0xFFFF,
            (0x20, 4) => state.interrupt_control = low,
            // INTSTS is read-only; `audio` writes back what it read.
            (0x24, 4) => {}
            (0x40, 4) => state.corb_base[0] = low,
            (0x44, 4) => state.corb_base[1] = low,
            (0x48, 4) => {
                // CORBRP reset (bit 15 of the read pointer) or the write
                // pointer, which starts the commands up to it.
                if low & 1 << 31 != 0 {
                    state.corb_read = 0;
                } else {
                    state.corb_write = low & 0xFF;
                    if let Err(what) = self.run_commands(&mut state) {
                        self.notes.add(format!("HD Audio: {what}"));
                        return None;
                    }
                }
            }
            // CORBCTL, CORBSTS (write one to clear), CORBSIZE.
            (0x4C, 4) => {
                state.corb_control = low & 3;
                state.corb_size = low >> 16 & 3;
            }
            (0x50, 4) => state.rirb_base[0] = low,
            (0x54, 4) => state.rirb_base[1] = low,
            // RIRBWP (bit 15 resets it), RINTCNT, and in an 8-byte write
            // also RIRBCTL, RIRBSTS and RIRBSIZE.
            (0x58, 4 | 8) => {
                if low & 1 << 15 != 0 {
                    state.rirb_write = 0;
                }
                state.response_count = low >> 16 & 0xFF;
                if size == 8 {
                    let high = (value >> 32) as u32;
                    state.rirb_control = high & 7;
                    state.rirb_status &= !(high >> 8);
                    state.rirb_size = high >> 16 & 3;
                }
            }
            (0x5C, 4) => {
                state.rirb_control = low & 7;
                state.rirb_status &= !(low >> 8);
                state.rirb_size = low >> 16 & 3;
            }
            // A stream descriptor's control: only its reset bit is used.
            (offset @ 0x80..0x180, 4) if offset % 0x20 == 0 && low & !1 == 0 => {
                let stream = ((offset - 0x80) / 0x20) as usize;
                if stream as u64 != INPUT_STREAMS {
                    return None;
                }
                state.streams[stream] = low;
            }
            _ => return None,
        }
        self.update(&state);
        Some(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{boot::DramSize, memory::GuestRam, soc::mc::Mc};
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn audio_start_up_and_codec_enumeration() {
        let ram = Arc::new(GuestRam::new(0x8000_0000, 0x40_0000).unwrap());
        let mc = Arc::new(Mc::new(DramSize::Gb4));
        // Address space 1: device 0x10000000 is DRAM 0x80000000 (4 MiB
        // section), directory at 0x80200000.
        ram.write(
            0x8020_0000 + 0x40 * 4,
            &(0xC000_0000u32 | 0x80000).to_le_bytes(),
        )
        .unwrap();
        for (register, value) in [(0x1C, 1), (0x20, 0xC008_0200), (SMMU_ASID, 0x8000_0001)] {
            mc.write(0, register, 4, value).unwrap();
        }
        let line = Arc::new(AtomicBool::new(false));
        let seen = Arc::clone(&line);
        let hda = Hda::new(
            DeviceMemory::new(&mc, &ram, SMMU_ASID),
            Box::new(move |high| seen.store(high, Ordering::Relaxed)),
            Arc::new(Notes::default()),
        );
        let bar = |offset| CONTROLLER + offset;
        hda.write(0, 0x180, 4, 1).unwrap();
        hda.write(0, bar(0x08), 4, 0).unwrap();
        hda.write(0, bar(0x08), 4, 1).unwrap();
        assert_eq!(hda.read(0, bar(0x08), 4), Some(1));
        assert_eq!(hda.read(0, bar(0x0C), 4), Some(0x8_0000), "codec 3 present");
        assert_eq!(hda.read(0, bar(0x4C), 4).unwrap() >> 20 & 0xF, 4);
        for (offset, size, value) in [
            (0x4C, 4, 0),
            (0x48, 4, 0x8000_0000),
            (0x48, 4, 0),
            (0x40, 4, 0x1000_1000),
            (0x44, 4, 0),
            (0x4C, 4, 0x2_0002),
            (0x58, 8, 0xC0_8000),
            (0x58, 4, 0xC0_0000),
            (0x50, 4, 0x1000_2000),
            (0x54, 4, 0),
            (0x5C, 4, 0x2_0003),
            (0x08, 4, 0x101),
            (0x20, 4, 0xC000_0010),
        ] {
            hda.write(0, bar(offset), size, value).unwrap();
        }
        let ask = |index: u32, command: u32| {
            ram.write(0x8000_1000 + u64::from(index) * 4, &command.to_le_bytes())
                .unwrap();
            hda.write(0, bar(0x48), 4, u64::from(index)).unwrap();
            assert!(line.load(Ordering::Relaxed));
            let mut entry = [0; 8];
            ram.read(0x8000_2000 + u64::from(index) * 8, &mut entry)
                .unwrap();
            hda.write(0, bar(0x5C), 4, 0x100 | 3).unwrap();
            assert!(!line.load(Ordering::Relaxed));
            u32::from_le_bytes(entry[..4].try_into().unwrap())
        };
        assert_eq!(ask(1, 0x300F_0000), 0x10DE_0029);
        assert_eq!(ask(2, 0x300F_0004), 0x0001_0001);
        assert_eq!(ask(3, 0x301F_0005) & 0xFF, 1);
        assert_eq!(ask(4, 0x301F_0004), 0x0002_0002);
        assert_eq!(ask(5, 0x302F_0009) >> 20 & 0xF, 0);
        assert_eq!(ask(6, 0x303F_0009) >> 20 & 0xF, 4);
        assert_eq!(ask(7, 0x3037_0740), 0);
        assert_eq!(ask(8, 0x303F_0900) >> 30, 0, "nothing docked");
        ram.write(0x8000_1000 + 9 * 4, &0x303F_2F04u32.to_le_bytes())
            .unwrap();
        assert_eq!(hda.write(0, bar(0x48), 4, 9), None);
    }
}
