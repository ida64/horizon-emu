//! The memory controller's power-on registers and its per-channel
//! apertures, MC0 (`0x7001_C000`) and MC1 (`0x7001_D000`), as a machine
//! that runs the boot chain has them. A machine started at kernel entry
//! has neither ([`super::mc::Mc::new`]).
//!
//! Power-on values: what the NX bootloader's hardware-state check reads
//! (`nxbl_hwcheck.csv`, the check `0x40002A7C`): the same 464 registers,
//! with the same values, in each of the three apertures (broadcast at
//! bootloader text `0x40006AC4`..`0x400085A2`, MC0 `0x400085AE`..`0x4000A0A4`,
//! MC1 `0x4000A0B2`..`0x4000BBCA`), each row's expected value under its
//! mask. \[inference\] What a console's block holds when the bootloader
//! starts, the check halting it on any difference; whether the hardware's
//! reset or the boot ROM put them there is not public. Among them
//! `MC_EMEM_CFG` (`0x050`) reads `0x2000` until the bootloader programs
//! the DRAM size. Bits outside the masks, and registers the check does not
//! read, are \[assumption\] zero, as the broadcast aperture has all its
//! registers at kernel entry.
//!
//! Two of the rows are not here: `MC_IRAM_BOM`/`MC_IRAM_TOM` (`0x65C`,
//! `0x660`, expected `0x4000_0000` and `0x8000_0000`) are written by
//! package1ldr, which runs on the BPMP before the bootloader (its
//! `enable_mem_ctl`, text `0xB04`, pkg1-research §3.1), through the
//! broadcast aperture only. That both channel apertures read them back
//! is the evidence for the rule here: \[inference\] a write to the
//! broadcast aperture reaches both channels' copies (as the EMC's does,
//! [`super::emc`]), a write to a channel aperture only that channel's.
//! Reads of a channel aperture return that channel's copy.

use std::sync::Arc;

use super::{
    mc::Mc,
    regs::{RegisterFile, word},
};
use crate::{bus::MmioDevice, snapshot::Snapshot};

pub const CHANNEL_BASES: [u64; 2] = [0x7001_C000, 0x7001_D000];
pub const SIZE: u64 = 0x1000;

/// The registers the check finds bits set in, by offset; the other 231 it
/// reads (233 in the broadcast aperture, with the two above) are zero in
/// the bits it looks at.
pub const POWER_ON_STATE: [(u64, u32); 231] = [
    (0x014, 0x3000_0030),
    (0x018, 0x2800_0F3F),
    (0x034, 0x0000_0008),
    (0x044, 0x2022_0000),
    (0x048, 0x0000_2800),
    (0x050, 0x0000_2000),
    (0x058, 0x0004_0202),
    (0x05C, 0x0004_0202),
    (0x064, 0xEB77_2400),
    (0x068, 0x249E_4800),
    (0x06C, 0x9DBB_B000),
    (0x090, 0x0000_0008),
    (0x094, 0x8000_0040),
    (0x098, 0x0000_003F),
    (0x09C, 0x0000_007F),
    (0x0A0, 0x0000_00FF),
    (0x0A4, 0x0000_007F),
    (0x0A8, 0x0000_007F),
    (0x0AC, 0x0000_001F),
    (0x0B0, 0x0000_007F),
    (0x0B4, 0x0000_007F),
    (0x0B8, 0x0000_001F),
    (0x0BC, 0x0000_001F),
    (0x0C0, 0x0000_007F),
    (0x0C4, 0x0000_007F),
    (0x0D0, 0x0F0F_0205),
    (0x0D4, 0x000F_0F0F),
    (0x0D8, 0x720A_4010),
    (0x0DC, 0x80A0_000A),
    (0x0E0, 0x001F_0000),
    (0x0E8, 0x1000_0000),
    (0x0EC, 0xFF00_FF00),
    (0x0F4, 0x0000_8000),
    (0x108, 0xFFFF_FFFF),
    (0x10C, 0x0000_FFFF),
    (0x11C, 0xFFFF_FFF0),
    (0x120, 0x7FFF_8000),
    (0x128, 0xF0E3_C07F),
    (0x12C, 0x3A66_0880),
    (0x130, 0x07F3_7CD0),
    (0x134, 0xCF0C_30FF),
    (0x15C, 0xFFFF_FFF0),
    (0x160, 0x7FFF_8000),
    (0x168, 0xF0E3_C07F),
    (0x16C, 0x3A66_0880),
    (0x170, 0x07F3_7CD0),
    (0x174, 0xCF0C_30FF),
    (0x19C, 0xFFFF_FFF0),
    (0x1AC, 0xFFFF_FFF0),
    (0x1C0, 0x0000_0030),
    (0x1D0, 0x0000_0030),
    (0x1EC, 0x8000_0000),
    (0x208, 0x0023_007E),
    (0x210, 0x0400_0000),
    (0x214, 0x0008_0000),
    (0x218, 0x0003_007E),
    (0x220, 0x0400_0000),
    (0x224, 0x0008_0000),
    (0x228, 0xF0E3_C07E),
    (0x22C, 0x3866_0800),
    (0x230, 0x04F3_7CD0),
    (0x234, 0xCF0C_30FF),
    (0x2A0, 0x0000_0004),
    (0x2A4, 0x0000_0004),
    (0x2A8, 0x0000_0004),
    (0x2B8, 0x0000_0004),
    (0x2BC, 0x0000_0004),
    (0x2C0, 0x0000_0004),
    (0x2D8, 0x0000_0004),
    (0x2E0, 0x0080_002E),
    (0x2E4, 0x0080_0004),
    (0x2E8, 0x001E_001E),
    (0x2EC, 0x0000_001E),
    (0x2F0, 0x001E_001E),
    (0x2F4, 0x001E_001E),
    (0x2F8, 0x0000_001E),
    (0x2FC, 0x0000_001E),
    (0x310, 0x0050_001E),
    (0x314, 0x0000_0080),
    (0x318, 0x0080_0024),
    (0x320, 0x0080_0004),
    (0x328, 0x0080_0023),
    (0x344, 0x001A_0049),
    (0x348, 0x0080_0080),
    (0x350, 0x0080_0065),
    (0x370, 0x0000_0018),
    (0x374, 0x0080_0080),
    (0x37C, 0x0080_007A),
    (0x380, 0x0080_0039),
    (0x384, 0x0000_0018),
    (0x388, 0x0080_0080),
    (0x390, 0x0080_009B),
    (0x394, 0x0080_001A),
    (0x398, 0x0000_0080),
    (0x3A0, 0x0080_00FF),
    (0x3A4, 0x0080_0004),
    (0x3AC, 0x0080_001A),
    (0x3B8, 0x0080_0049),
    (0x3BC, 0x0080_005A),
    (0x3C0, 0x0080_0049),
    (0x3C4, 0x0080_005A),
    (0x3C8, 0x0000_001E),
    (0x3D8, 0x0080_0023),
    (0x3DC, 0x0080_00FF),
    (0x3E0, 0x0080_002E),
    (0x3E4, 0x0080_0023),
    (0x3E8, 0x0080_001A),
    (0x3EC, 0x0080_00FF),
    (0x3F0, 0x0080_009B),
    (0x3FC, 0xF000_0000),
    (0x404, 0x0000_0004),
    (0x40C, 0x0000_0004),
    (0x410, 0x0000_0004),
    (0x414, 0x0000_0004),
    (0x418, 0xE4BA_C343),
    (0x420, 0x0000_003F),
    (0x424, 0x0000_003F),
    (0x42C, 0x0000_003F),
    (0x430, 0x0000_003F),
    (0x438, 0x0000_003F),
    (0x43C, 0x0000_003F),
    (0x444, 0x0000_003F),
    (0x448, 0x0000_003F),
    (0x450, 0x0000_003F),
    (0x454, 0x0000_003F),
    (0x45C, 0x0000_003F),
    (0x460, 0x0000_003F),
    (0x480, 0x0000_003F),
    (0x484, 0x0000_003F),
    (0x48C, 0x0000_003F),
    (0x490, 0x0000_003F),
    (0x498, 0x0000_003F),
    (0x49C, 0x0000_003F),
    (0x4A4, 0x0000_003F),
    (0x4A8, 0x0000_003F),
    (0x4B0, 0x0000_003F),
    (0x4B4, 0x0000_003F),
    (0x4BC, 0x0000_003F),
    (0x4C0, 0x0000_003F),
    (0x4C8, 0x0000_003F),
    (0x4CC, 0x0000_003F),
    (0x4D4, 0x0000_003F),
    (0x4D8, 0x0000_003F),
    (0x4E0, 0x0000_003F),
    (0x4E4, 0x0000_003F),
    (0x4EC, 0x0000_003F),
    (0x4F0, 0x0000_003F),
    (0x4F8, 0x0000_003F),
    (0x4FC, 0x0000_003F),
    (0x510, 0x0000_003F),
    (0x514, 0x0000_003F),
    (0x51C, 0x0000_003F),
    (0x520, 0x0000_003F),
    (0x528, 0x0000_003F),
    (0x52C, 0x0000_003F),
    (0x534, 0x0000_003F),
    (0x538, 0x0000_003F),
    (0x540, 0x0000_003F),
    (0x544, 0x0000_003F),
    (0x54C, 0x0000_003F),
    (0x550, 0x0000_003F),
    (0x558, 0x0000_003F),
    (0x55C, 0x0000_003F),
    (0x588, 0x0000_003F),
    (0x58C, 0x0000_003F),
    (0x590, 0x0000_1ED3),
    (0x600, 0x0000_8000),
    (0x614, 0x0000_003F),
    (0x618, 0x0000_003F),
    (0x620, 0x0000_003F),
    (0x624, 0x0000_003F),
    (0x62C, 0x0000_003F),
    (0x630, 0x0000_003F),
    (0x638, 0x0000_003F),
    (0x63C, 0x0000_003F),
    (0x648, 0xFFF0_0000),
    (0x66C, 0x8000_0080),
    (0x670, 0xFFF0_0000),
    (0x684, 0x0000_001F),
    (0x690, 0x001E_001E),
    (0x694, 0x001E_001E),
    (0x698, 0x001E_001E),
    (0x69C, 0x001E_001E),
    (0x6A0, 0x001E_001E),
    (0x6A4, 0x001E_001E),
    (0x6B4, 0x8000_0080),
    (0x6BC, 0x8000_8041),
    (0x6C0, 0x0000_007F),
    (0x6C4, 0x0000_0001),
    (0x6F0, 0x000A_1020),
    (0x6F4, 0x0000_2636),
    (0x958, 0xFF00_FF00),
    (0x960, 0x0000_1000),
    (0x990, 0x003F_0004),
    (0x994, 0x003F_0004),
    (0x9A0, 0xFFF0_0000),
    (0x9DC, 0x0000_0001),
    (0xA00, 0x0000_0004),
    (0xA04, 0x0000_0004),
    (0xA08, 0x0000_0004),
    (0xA10, 0x0000_0004),
    (0xA14, 0x0000_0004),
    (0xA18, 0x0000_0004),
    (0xA1C, 0x0000_0004),
    (0xA20, 0x0000_0300),
    (0xA24, 0x0000_0300),
    (0xA28, 0x0000_0300),
    (0xA2C, 0x0000_0300),
    (0xA3C, 0x0000_0004),
    (0xA40, 0x0000_0004),
    (0xA44, 0x0000_0004),
    (0xA48, 0x0000_0004),
    (0xA4C, 0x0000_0004),
    (0xB88, 0x0000_03FF),
    (0xB8C, 0x0000_03FF),
    (0xB98, 0x0000_00FF),
    (0xBB0, 0x1000_0000),
    (0xBB4, 0x0000_0800),
    (0xBB8, 0x0303_40D0),
    (0xBBC, 0xC304_3000),
    (0xBC0, 0x0000_0300),
    (0xBCC, 0x0000_0002),
    (0xBD0, 0x0000_0100),
    (0xBD4, 0x0000_00A0),
    (0xBD8, 0x0000_0060),
    (0xBDC, 0x0000_0034),
    (0xBE0, 0x0000_0018),
    (0xBE4, 0x0000_000C),
    (0xBE8, 0x0000_0004),
    (0xDD4, 0x2022_0000),
    (0xDD8, 0x0000_2800),
];

/// Both channels' copies of the registers.
pub struct Channels {
    copies: [RegisterFile; 2],
}

impl Channels {
    pub fn power_on() -> Self {
        Self {
            copies: std::array::from_fn(|_| RegisterFile::zeroed(POWER_ON_STATE)),
        }
    }

    /// A write through the broadcast aperture.
    pub fn broadcast(&self, register: u64, value: u32) {
        for copy in &self.copies {
            copy.set(register, value);
        }
    }
}

/// One channel's aperture.
pub struct McChannel {
    mc: Arc<Mc>,
    channel: usize,
}

impl McChannel {
    /// `None` for a memory controller without channel apertures.
    pub fn new(mc: &Arc<Mc>, channel: usize) -> Option<Self> {
        mc.channels()?;
        Some(Self {
            mc: Arc::clone(mc),
            channel,
        })
    }

    fn copy(&self) -> &RegisterFile {
        &self
            .mc
            .channels()
            .expect("checked when the aperture was made")
            .copies[self.channel]
    }
}

impl MmioDevice for McChannel {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        self.copy().get(word(offset, size)?).map(u64::from)
    }

    fn write(&self, _core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        self.copy().set(word(offset, size)?, value as u32);
        Some(())
    }

    /// Each aperture saves its channel's copy.
    fn snapshot(&self) -> Option<&dyn Snapshot> {
        Some(self.copy())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcast_writes_reach_both_channels() {
        let mc = Arc::new(Mc::power_on());
        let channels = [0, 1].map(|channel| McChannel::new(&mc, channel).unwrap());
        assert_eq!(channels[1].read(0, 0x050, 4), Some(0x2000));
        mc.write(0, 0x65C, 4, 0x4000_0000).unwrap();
        channels[0].write(0, 0x050, 4, 0x1000).unwrap();
        assert_eq!(channels[0].read(0, 0x65C, 4), Some(0x4000_0000));
        assert_eq!(channels[1].read(0, 0x65C, 4), Some(0x4000_0000));
        assert_eq!(channels[0].read(0, 0x050, 4), Some(0x1000));
        assert_eq!(channels[1].read(0, 0x050, 4), Some(0x2000));
        assert_eq!(mc.read(0, 0x050, 4), Some(0x2000));
        assert!(McChannel::new(&Arc::new(Mc::new(crate::boot::DramSize::Gb4)), 0).is_none());
    }
}
