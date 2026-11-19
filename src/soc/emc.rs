//! External memory controller (EMC): the DRAM interface's timing, clock
//! change and training machinery.
//!
//! Nothing in the emulator has DRAM timing, so the block is storage plus
//! the handshakes the firmware's clock driver (`pcv`) waits on when it
//! changes the memory clock. At start-up `pcv` (`TegraInitEmcDvfs`, text
//! `0x5a9f0`) installs the timing table for the board's DRAM (DRAM ID 0:
//! `SdevEmcDvfsTableS4gb01`, ten rates from 40.8 to 1600 MHz, text
//! `0x65120`), switches to the boot rate (204 MHz), then switches to every
//! rate in turn, training those that need it (write, read and VREF training
//! from 408 MHz up), switches back and checks the rate is unchanged
//! ("Unexpected rate change ... after training"); `InitEmc` (text
//! `0x39640`) then moves to 800 MHz through 408 MHz. Its code is a port of
//! NVIDIA's Tegra210 clock-change and training sequence; the public port
//! closest to it is hekate's Minerva (`sys_sdrammtc.c`), whose order of
//! accesses matches what `pcv` does here, and which names the registers
//! below together with Linux `drivers/memory/tegra/tegra210-emc.h`
//! (recalled from those sources, not verified on hardware).
//!
//! The block appears three times: a broadcast view at [`BASE`], whose
//! writes reach both channels and whose reads return channel 0, and one
//! view per channel at [`CHANNEL_BASES`] (Minerva `EMC`, `EMC_CH0`,
//! `EMC_CH1`).
//!
//! Behaviours (everything completes at once, since no command takes time
//! here):
//!
//! * timing update (`TIMING_CONTROL`): `EMC_STATUS` never shows it stalled;
//! * clock change: when the guest switches `CLK_SOURCE_EMC` in the clock
//!   controller, the controller runs the writes queued in its clock-change
//!   FIFO and sets `CLKCHANGE_COMPLETE` in `INTSTATUS` (write one to clear);
//! * self refresh: `EMC_STATUS` shows both devices in self refresh while
//!   `SELF_REF` requests it;
//! * DLL: once enabled the digital DLL reports lock and an updated private
//!   setting (`DIG_DLL_STATUS` bits 15 and 17) and keeps reporting them: the
//!   firmware waits for lock again after disabling the DLL during training,
//!   as the Minerva sequence does (`_minerva_train`);
//! * mode register reads (`MRR`): the selected mode register of the DRAM,
//!   with the data-valid bit in `EMC_STATUS` until read (`MODE_REGISTERS`);
//! * training: runs during the clock change and finds the settings already
//!   programmed optimal, so trimmers keep their values and the write VREF
//!   result reports the VREF the guest set (`TRAINING_OPT_DQ_OB_VREF`).
//!
//! A register the guest reads before writing it and that is not in
//! `ENTRY_STATE` refuses the read.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use serde::{Deserialize, Serialize};

use super::regs::word;
use crate::{
    bus::{MmioDevice, NoState},
    snapshot::{Snapshot, SnapshotError, decode, encode},
};

pub const BASE: u64 = 0x7001_B000;
pub const CHANNEL_BASES: [u64; 2] = [0x7001_E000, 0x7001_F000];
pub const SIZE: u64 = 0x1000;

const INTSTATUS: u64 = 0x000;
const CLKCHANGE_COMPLETE: u32 = 1 << 4;
const SELF_REF: u64 = 0x0E0;
const MRR: u64 = 0x0EC;
const EMC_STATUS: u64 = 0x2B4;
const MRR_DIVLD: u32 = 1 << 20;
/// Both devices (ranks) in self refresh.
const IN_SELF_REFRESH: u32 = 3 << 8;
const CFG_DIG_DLL: u64 = 0x2BC;
const DIG_DLL_STATUS: u64 = 0x2C4;
const DLL_LOCK: u32 = 1 << 15;
const DLL_PRIV_UPDATED: u32 = 1 << 17;
const CCFIFO_ADDR: u64 = 0x3E8;
const CCFIFO_DATA: u64 = 0x3EC;
/// MR14 (DQ VREF) write commands for the two sub-partitions.
const MRW12: u64 = 0x4BC;
const MRW13: u64 = 0x4C0;
const TRAINING_OPT_DQ_OB_VREF: u64 = 0xEC4;

/// What the boot chain's SDRAM bring-up leaves in the registers the
/// clock driver reads before writing them. The first-stage bootloader
/// programs the EMC from the boot configuration table's SDRAM parameters
/// for the board's DRAM; hekate replicates that (`sdram.c`
/// `_sdram_config_t210`) with Nintendo's parameters for DRAM ID 0, Samsung
/// K4F6E304HB-MGCH 4 GB (`sdram_config.inl` `_dram_cfg_0_samsung_4gb`, no
/// per-ID patches). The value is the last one `_sdram_config_t210` writes:
///
/// | offset | register | value | sdram.c |
/// |---|---|---|---|
/// | `0x104` | `FBIO_CFG5` | `0x9160_a00d` | 572 (`emc_fbio_cfg5`: LPDDR4) |
/// | `0x584` | `FBIO_CFG7` | `0x0000_3bbf` | 269 (`emc_fbio_cfg7`: both channels enabled) |
/// | `0x00c` | `CFG` | `0xf320_0000` | 769 (`emc_cfg`) |
/// | `0x2a4` | `AUTO_CAL_CONFIG` | `0xa01a_51d8` | 556 (`emc_auto_cal_config`; the interval is non-zero, so 677 does not apply) |
/// | `0x008` | `DBG` | `0x0100_0c00` | 610 (`emc_dbg`) |
/// | `0x024` | `PIN` | `0x0003_3101` | 707 (`emc_pin_gpio_enable` 3, `emc_pin_gpio` 3, clock enable and reset released) |
/// | `0x558` | `CFG_PIPE_CLK` | `0x0` | 781 (`emc_cfg_pipe_clk`) |
/// | `0x2bc` | `CFG_DIG_DLL` | `0x002c_00a0` | 647 (`emc_cfg_dig_dll`: DLL off) |
/// | `0x5f4` | `CFG_UPDATE` | `0x7000_0301` | 768 (`emc_cfg_update`) |
///
/// Both channels hold the same values (broadcast writes).
const ENTRY_STATE: [(u64, u32); 9] = [
    (0x104, 0x9160_A00D),
    (0x584, 0x0000_3BBF),
    (0x00C, 0xF320_0000),
    (0x2A4, 0xA01A_51D8),
    (0x008, 0x0100_0C00),
    (0x024, 0x0003_3101),
    (0x558, 0x0000_0000),
    (CFG_DIG_DLL, 0x002C_00A0),
    (0x5F4, 0x7000_0301),
];

/// Mode registers of the LPDDR4 devices that the firmware reads, as
/// `(register, data)`; data holds one byte per byte lane. A read of any
/// other mode register refuses.
///
/// MR18 and MR19 are the low and high bytes of the DQS interval
/// oscillator count, which `pcv` reads around every training and for
/// periodic compensation and turns into the DRAM's clock-tree delay
/// (Minerva `_minerva_update_clock_tree_delay`: delay = 10^6 · run clocks /
/// 2 / (rate in MHz · count)). The count is a measurement of the chips'
/// silicon; it is kept constant, so the derived delay never drifts and
/// compensation never changes a trimmer. 1280 (`0x0500`) gives 500 ps at
/// 1600 MHz with the table's 2048-clock run time, within the range JEDEC
/// allows for LPDDR4's DQS-to-DQ delay (200-800 ps, recalled); the exact
/// value is an assumption.
const MODE_REGISTERS: [(u32, u32); 2] = [(18, 0x0000), (19, 0x0505)];

#[derive(Serialize, Deserialize)]
struct State {
    channels: [HashMap<u64, u32>; 2],
    /// Whether each channel's DLL has locked since power-on.
    dll_locked: [bool; 2],
    interrupts: u32,
    ccfifo_data: u32,
    /// Writes queued for the next clock change.
    ccfifo: Vec<(u64, u32)>,
    /// A mode register read waiting to be collected: request bits.
    mrr: [Option<u32>; 2],
}

pub struct Emc {
    state: Mutex<State>,
}

impl Default for Emc {
    /// The block as the boot chain leaves it for the kernel.
    fn default() -> Self {
        Self::with_registers(ENTRY_STATE.into_iter().collect())
    }
}

impl Emc {
    /// The block as the SoC comes out of reset: no register holds a value
    /// until written. The NX bootloader programs it during its SDRAM
    /// bring-up (about 273 registers, `0x40015348`, research §2.5) and
    /// reads none of them back; whatever of `ENTRY_STATE` it does not
    /// write refuses when the kernel's clock driver reads it.
    pub fn power_on() -> Self {
        Self::with_registers(HashMap::new())
    }

    fn with_registers(channel: HashMap<u64, u32>) -> Self {
        Self {
            state: Mutex::new(State {
                channels: [channel.clone(), channel],
                dll_locked: [false; 2],
                interrupts: 0,
                ccfifo_data: 0,
                ccfifo: Vec::new(),
                mrr: [None; 2],
            }),
        }
    }
}

impl Emc {
    /// The clock controller switched the memory clock: run the clock-change
    /// FIFO and report completion.
    pub fn clock_changed(&self) {
        let mut state = self.state.lock().unwrap();
        for (register, value) in std::mem::take(&mut state.ccfifo) {
            for channel in 0..2 {
                write_channel(&mut state, channel, register, value);
            }
        }
        state.interrupts |= CLKCHANGE_COMPLETE;
    }
}

/// Saves both channels' registers, DLL lock, the interrupt status, the
/// clock-change FIFO not yet run and the mode register reads not yet
/// collected.
impl Snapshot for Emc {
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        encode(&*self.state.lock().unwrap())
    }

    fn restore(&self, state: &[u8]) -> Result<(), SnapshotError> {
        *self.state.lock().unwrap() = decode("EMC", state)?;
        Ok(())
    }
}

/// Which registers a view of the block reaches.
#[derive(Clone, Copy)]
pub enum View {
    /// Writes reach both channels, reads return channel 0.
    Broadcast,
    Channel(usize),
}

/// One of the three address ranges of the block.
pub struct EmcPort {
    emc: Arc<Emc>,
    view: View,
}

impl EmcPort {
    pub fn new(emc: &Arc<Emc>, view: View) -> Self {
        Self {
            emc: Arc::clone(emc),
            view,
        }
    }
}

fn write_channel(state: &mut State, channel: usize, register: u64, value: u32) {
    match register {
        MRR => state.mrr[channel] = Some(value & 0xFFFF_0000),
        CFG_DIG_DLL if value & 1 != 0 => state.dll_locked[channel] = true,
        _ => {}
    }
    state.channels[channel].insert(register, value);
}

fn read_channel(state: &mut State, channel: usize, register: u64) -> Option<u32> {
    let stored = |register| state.channels[channel].get(&register).copied();
    Some(match register {
        INTSTATUS => state.interrupts,
        EMC_STATUS => {
            let mut value = 0;
            if state.mrr[channel].is_some() {
                value |= MRR_DIVLD;
            }
            if stored(SELF_REF).unwrap_or(0) & 1 != 0 {
                value |= IN_SELF_REFRESH;
            }
            value
        }
        DIG_DLL_STATUS => {
            if state.dll_locked[channel] {
                DLL_LOCK | DLL_PRIV_UPDATED
            } else {
                0
            }
        }
        MRR => {
            let request = state.mrr[channel].take()?;
            let mode_register = request >> 16 & 0xFF;
            let (_, data) = MODE_REGISTERS
                .into_iter()
                .find(|&(register, _)| register == mode_register)?;
            request | data
        }
        TRAINING_OPT_DQ_OB_VREF => stored(MRW12)? & 0xFFFF | (stored(MRW13)? & 0xFFFF) << 16,
        register => stored(register)?,
    })
}

impl MmioDevice for EmcPort {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        let register = word(offset, size)?;
        let channel = match self.view {
            View::Broadcast => 0,
            View::Channel(channel) => channel,
        };
        let mut state = self.emc.state.lock().unwrap();
        read_channel(&mut state, channel, register).map(u64::from)
    }

    fn write(&self, _core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        let register = word(offset, size)?;
        let value = value as u32;
        let mut state = self.emc.state.lock().unwrap();
        match (self.view, register) {
            (_, INTSTATUS) => state.interrupts &= !value,
            (View::Broadcast, CCFIFO_DATA) => state.ccfifo_data = value,
            // The low 16 bits are the register to write, the rest a delay.
            (View::Broadcast, CCFIFO_ADDR) => {
                let data = state.ccfifo_data;
                state.ccfifo.push((u64::from(value & 0xFFFF), data));
            }
            (View::Broadcast, _) => {
                for channel in 0..2 {
                    write_channel(&mut state, channel, register, value);
                }
            }
            (View::Channel(channel), _) => write_channel(&mut state, channel, register, value),
        }
        Some(())
    }

    /// The block is saved once, through its broadcast range; the channel
    /// ranges are views of the same state.
    fn snapshot(&self) -> Option<&dyn Snapshot> {
        match self.view {
            View::Broadcast => Some(&*self.emc),
            View::Channel(_) => Some(&NoState),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ports() -> (Arc<Emc>, EmcPort, EmcPort) {
        let emc: Arc<Emc> = Arc::default();
        let broadcast = EmcPort::new(&emc, View::Broadcast);
        let channel1 = EmcPort::new(&emc, View::Channel(1));
        (emc, broadcast, channel1)
    }

    #[test]
    fn a_clock_change_runs_the_fifo_and_completes() {
        let (emc, broadcast, channel1) = ports();
        broadcast.write(0, CCFIFO_DATA, 4, 1).unwrap();
        broadcast
            .write(0, CCFIFO_ADDR, 4, 0x8000_0000 | SELF_REF)
            .unwrap();
        assert_eq!(broadcast.read(0, SELF_REF, 4), None);
        assert_eq!(broadcast.read(0, INTSTATUS, 4), Some(0));
        emc.clock_changed();
        assert_eq!(channel1.read(0, SELF_REF, 4), Some(1));
        assert_eq!(
            channel1.read(0, EMC_STATUS, 4),
            Some(u64::from(IN_SELF_REFRESH))
        );
        assert_eq!(
            broadcast.read(0, INTSTATUS, 4),
            Some(u64::from(CLKCHANGE_COMPLETE))
        );
        broadcast
            .write(0, INTSTATUS, 4, u64::from(CLKCHANGE_COMPLETE))
            .unwrap();
        assert_eq!(broadcast.read(0, INTSTATUS, 4), Some(0));
    }

    #[test]
    fn mode_register_reads_go_through_the_valid_bit() {
        let (_, broadcast, channel1) = ports();
        broadcast.write(0, MRR, 4, 0x8013_0000).unwrap();
        assert_eq!(channel1.read(0, EMC_STATUS, 4), Some(u64::from(MRR_DIVLD)));
        assert_eq!(channel1.read(0, MRR, 4), Some(0x8013_0505));
        assert_eq!(channel1.read(0, EMC_STATUS, 4), Some(0));
        // Channel 0's result is still waiting.
        assert_eq!(broadcast.read(0, MRR, 4), Some(0x8013_0505));
        broadcast.write(0, MRR, 4, 0x8005_0000).unwrap();
        assert_eq!(broadcast.read(0, MRR, 4), None);
    }

    #[test]
    fn the_dll_stays_locked_once_it_has_locked() {
        let (_, broadcast, _) = ports();
        assert_eq!(broadcast.read(0, DIG_DLL_STATUS, 4), Some(0));
        broadcast.write(0, CFG_DIG_DLL, 4, 0x3C9).unwrap();
        broadcast.write(0, CFG_DIG_DLL, 4, 0x3C8).unwrap();
        assert_eq!(
            broadcast.read(0, DIG_DLL_STATUS, 4),
            Some(u64::from(DLL_LOCK | DLL_PRIV_UPDATED))
        );
    }
}
