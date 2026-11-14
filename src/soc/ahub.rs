//! The audio hub's I2S controllers, DMA interface and crossbar, and the
//! engine's DMA controller, as far as they are reached from the CPU.
//!
//! The audio processing engine's hub (base `0x702D0000`) routes audio
//! between the DSP's DMA, the I2S and DMIC interfaces and its mixers; the
//! DSP's mixer (`gmix`, see [`super::adsp`]) programs most of it. `pcv`,
//! whenever it powers the engine up (on `audio`'s `clkrst` request 4 for
//! module `0x40000036`), runs the memory built-in
//! self-test workaround the public Linux `clk-tegra210.c`
//! (`tegra210_ape_mbist_war`) also performs: for each of the five I2S
//! controllers it sets bit 10 of `I2S_CTRL` (`0xA0`), clears and sets
//! `I2S_CG` (`0x88`, second-level clock gating) and restores `I2S_CTRL`,
//! reading each back.
//!
//! Both registers are configuration and hold what was written. `I2S_CG`
//! starts at 1, its reset value in the public Linux `tegra210_i2s.c`
//! register defaults; `I2S_CTRL` starts at 0, which is an assumption (the
//! workaround restores whatever it read). `I2S_SOFT_RESET` (`0x84`, read
//! during start-up) reads 0: no reset in progress.
//!
//! # `audio`'s own start-up of the hub and the engine's DMA
//!
//! Later in its start-up `audio` (which maps the whole engine from
//! `0x702C0000`, its NPDM) programs, by write and read-back, a few
//! registers of the hub's DMA interface and crossbar (`0x702D0000`:
//! `0x000`, `0x020`, `0x300`, `0x320`, `0x700`, `0x704`, `0x800`, `0x840`)
//! and of the engine's DMA controller (`0x702E2000`: channel 0 at `0x000`,
//! channel 1 at `0x080`, global registers at `0xC00`); register names and
//! layout as in the public Linux `tegra210_admaif.c`, `tegra210_ahub.c`
//! and `tegra210-adma.c`. The code doing it has not been located in
//! `audio`'s text (the addresses are computed from the engine's base);
//! the list is what a survey run observed.
//!
//! They are configuration that holds what was written, starting at zero
//! (an assumption). The soft-reset registers among them (the DMA
//! interface's global one at `0x704`, the DMA controller's per channel at
//! `+0x04` and global one at `0xC04`, Linux `ADMA_GLOBAL_SOFT_RESET`) read
//! 0 at once: nothing is in flight, so a reset completes immediately. No
//! transfer is ever started here; registers not listed here refuse.
use crate::bus::MmioDevice;

use super::regs::{RegisterFile, word};

/// I2S1 to I2S5, `0x100` bytes apart.
pub const I2S_BASE: u64 = 0x702D_1000;
pub const I2S_SIZE: u64 = 0x500;
const CONTROLLERS: u64 = 5;
const I2S_SOFT_RESET: u64 = 0x84;
const I2S_CG: u64 = 0x88;
const I2S_CTRL: u64 = 0xA0;
/// Configuration `audio` writes and reads back once it has routed its
/// output (I2S1, seen in a survey run): RX and TX enable, CIF control,
/// control and slot control, the controller enable, timing and slot
/// control (`tegra210_i2s.h` names). Reset values from the public
/// `tegra210_i2s.c` defaults (CIF control `0x7700`, timing `0x1F`), zero
/// otherwise (an assumption; each is written before it is read).
const I2S_CONFIGURATION: [(u64, u32); 11] = [
    (0x00, 0),
    (0x20, 0x7700),
    (0x24, 0),
    (0x28, 0),
    (0x40, 0),
    (0x60, 0x7700),
    (0x64, 0),
    (0x68, 0),
    (0x80, 0),
    (0xA4, 0x1F),
    (0xA8, 0),
];

/// The hub's DMA interface and crossbar.
pub const ADMAIF_BASE: u64 = 0x702D_0000;
pub const ADMAIF_SIZE: u64 = 0x1000;
const ADMAIF_REGISTERS: [u64; 8] = [0x000, 0x020, 0x300, 0x320, 0x700, 0x704, 0x800, 0x840];
const ADMAIF_RESETS: [u64; 1] = [0x704];
/// The engine's DMA controller.
pub const ADMA_BASE: u64 = 0x702E_2000;
pub const ADMA_SIZE: u64 = 0x1000;
const ADMA_REGISTERS: [u64; 14] = [
    0x000, 0x004, 0x024, 0x028, 0x034, 0x044, 0x080, 0x084, 0x0A4, 0x0A8, 0x0BC, 0x0C4, 0xC00,
    0xC04,
];
const ADMA_RESETS: [u64; 3] = [0x004, 0x084, 0xC04];

/// Configuration registers, some of which are soft resets that complete
/// at once.
pub struct Block {
    registers: RegisterFile,
    resets: &'static [u64],
}

impl MmioDevice for Block {
    fn read(&self, core: usize, offset: u64, size: u8) -> Option<u64> {
        self.registers.read(core, offset, size)
    }

    fn write(&self, core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        let offset = word(offset, size)?;
        self.registers.get(offset)?;
        let value = if self.resets.contains(&offset) {
            0
        } else {
            value
        };
        self.registers.write(core, offset, size, value)
    }
}

/// The five controllers' registers.
pub fn i2s() -> Block {
    const RESETS: [u64; 5] = [0x084, 0x184, 0x284, 0x384, 0x484];
    Block {
        registers: RegisterFile::strict((0..CONTROLLERS).flat_map(|controller| {
            let base = controller * 0x100;
            [
                (base + I2S_SOFT_RESET, 0),
                (base + I2S_CG, 1),
                (base + I2S_CTRL, 0),
            ]
            .into_iter()
            .chain(I2S_CONFIGURATION.map(|(register, value)| (base + register, value)))
        })),
        resets: &RESETS,
    }
}

pub fn admaif() -> Block {
    Block {
        registers: RegisterFile::strict(ADMAIF_REGISTERS.map(|register| (register, 0))),
        resets: &ADMAIF_RESETS,
    }
}

pub fn adma() -> Block {
    Block {
        registers: RegisterFile::strict(ADMA_REGISTERS.map(|register| (register, 0))),
        resets: &ADMA_RESETS,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soft_resets_complete_and_configuration_holds() {
        let adma = adma();
        adma.write(0, 0xC04, 4, 1).unwrap();
        assert_eq!(adma.read(0, 0xC04, 4), Some(0));
        adma.write(0, 0x028, 4, 0x1234).unwrap();
        assert_eq!(adma.read(0, 0x028, 4), Some(0x1234));
        assert_eq!(adma.write(0, 0x030, 4, 1), None);
        assert_eq!(i2s().read(0, 0x188, 4), Some(1));
    }
}
