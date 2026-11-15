//! Memory controller: DRAM size and the SMMU that maps device-visible
//! addresses to DRAM.
//!
//! The kernel never maps this block; it reaches it through the secure
//! monitor's register call. It reads `MC_EMEM_CFG` for the DRAM size
//! (`0x800d2560`) and programs the SMMU (`0x800cf120`): one page directory
//! per address space ID through the `PTB_ASID`/`PTB_DATA` pair, and one
//! ASID register per client device.

use std::sync::{Arc, Mutex};

use super::regs::{RegisterFile, word};
use crate::{boot::DramSize, bus::MmioDevice, memory::GuestRam};

pub const BASE: u64 = 0x7001_9000;
pub const SIZE: u64 = 0x1000;

const PAGE_SIZE: u64 = 0x1000;

/// Selects the address space `PTB_DATA` accesses.
const SMMU_PTB_ASID: u64 = 0x1C;
/// Page directory of the selected address space.
const SMMU_PTB_DATA: u64 = 0x20;
/// DRAM size in MiB.
const EMEM_CFG: u64 = 0x50;

/// Address space register of the display controller.
pub const SMMU_DC_ASID: u64 = 0x240;
/// VIC client ASID, from the public Linux `tegra210.c` SMMU swgroup table.
pub const SMMU_VIC_ASID: u64 = 0x284;
/// Host1x CDMA client ASID, from the same public swgroup table.
pub const SMMU_HOST1X_ASID: u64 = 0x250;
/// Address space register of the eMMC port's host controller.
pub const SMMU_SDMMC4A_ASID: u64 = 0xAA0;
/// Address space register of the game card port's host controller (two
/// registers below the eMMC port's, as the SDMMC ports are numbered).
pub const SMMU_SDMMC2A_ASID: u64 = 0xA98;
/// Address space register of the microSD port's host controller (the one
/// below the game card port's).
pub const SMMU_SDMMC1A_ASID: u64 = 0xA94;
/// Address space register of the GPU: the one the kernel's device table
/// (kernel `0x800D83B8`, indexed by device name) gives device name 25, the
/// only one nvservices attaches the GPU's address space to
/// (`MC_SMMU_GPUB_ASID` in Atmosphère; Linux's single `gpu` swgroup).
pub const SMMU_GPU_ASID: u64 = 0xAAC;
/// Address space register of the security engine (`spl` attaches device
/// 0x1D, and the kernel changes this register from `0x80000004` to
/// `0x80000005`).
pub const SMMU_SE_ASID: u64 = 0xABC;

const ASID_COUNT: usize = 128;
/// A client's ASID register holds one address space per 4 GiB of device
/// address, a byte each, lowest first (the kernel attaches the eMMC port by
/// changing `0x84040404` to `0x84040406`).
const ASID_ENABLE: u32 = 1 << 31;
const ASID_FIELD: u32 = 0x7F;
/// Device addresses are 34 bits wide.
const ADDRESS_LIMIT: u64 = 1 << 34;

/// Page directory and page table entries share a layout: a frame number,
/// and for directory entries whether it names a page table.
const ENTRY_FRAME: u32 = 0x003F_FFFF;
const ENTRY_NEXT: u32 = 1 << 28;
const ENTRY_ACCESS: u32 = 3 << 30;

/// Client hot-reset (flush) control and status, for clients 0-31 and
/// 32-63. Software sets a client's control bit and waits for the status
/// bit before resetting the client (pcv `NvClockSocMCFlush`, text
/// `0x4f050`: polls up to 100 000 times 10 µs). Register names and the
/// meaning of the status bit (flush done: no transaction of the client
/// outstanding) are recalled from Linux `tegra210-mc` (`MC_CLIENT_HOTRESET_*`).
const CLIENT_HOTRESET_CTRL: u64 = 0x200;
const CLIENT_HOTRESET_STATUS: u64 = 0x204;
const CLIENT_HOTRESET_CTRL_1: u64 = 0x970;
const CLIENT_HOTRESET_STATUS_1: u64 = 0x974;

pub struct Mc {
    /// Everything without behavior of its own. The kernel programs these
    /// registers by read-modify-write through the monitor and acts on none of
    /// the initial contents.
    registers: RegisterFile,
    page_directories: Mutex<[u32; ASID_COUNT]>,
}

impl Mc {
    pub fn new(dram: DramSize) -> Self {
        Self {
            registers: RegisterFile::zeroed([(EMEM_CFG, dram.megabytes())]),
            page_directories: Mutex::new([0; ASID_COUNT]),
        }
    }

    fn selected_asid(&self) -> usize {
        self.registers.get(SMMU_PTB_ASID).unwrap_or(0) as usize % ASID_COUNT
    }

    /// Translates `address` as the client whose ASID register is at
    /// `client` sees it, reading page tables through `read_u32` (physical
    /// address to little-endian word). `None` if the client is not behind the
    /// SMMU or the address is unmapped.
    pub fn translate(
        &self,
        client: u64,
        address: u64,
        read_u32: impl Fn(u64) -> Option<u32>,
    ) -> Option<u64> {
        let asids = self
            .registers
            .get(client)
            .filter(|asids| asids & ASID_ENABLE != 0)?;
        (address < ADDRESS_LIMIT).then_some(())?;
        let asid = asids >> (8 * (address >> 32)) & ASID_FIELD;
        let address = address as u32;
        let directory = self.page_directories.lock().unwrap()[asid as usize];
        let entry_at = |table: u32, index: u32| {
            let entry = read_u32(frame_address(table) + u64::from(index) * 4)?;
            (entry & ENTRY_ACCESS != 0).then_some(entry)
        };
        (directory & ENTRY_ACCESS != 0).then_some(())?;
        let directory_entry = entry_at(directory, address >> 22)?;
        if directory_entry & ENTRY_NEXT == 0 {
            // A 4 MiB section.
            return Some(frame_address(directory_entry) + u64::from(address & 0x3F_FFFF));
        }
        let page_entry = entry_at(directory_entry, address >> 12 & 0x3FF)?;
        Some(frame_address(page_entry) + u64::from(address & 0xFFF))
    }
}

fn frame_address(entry: u32) -> u64 {
    u64::from(entry & ENTRY_FRAME) << 12
}

/// Guest memory as one SMMU client sees it: what a device that masters the
/// bus reads and writes.
pub struct DeviceMemory {
    mc: Arc<Mc>,
    ram: Arc<GuestRam>,
    client: u64,
}

impl DeviceMemory {
    /// `client` is the offset of the device's ASID register.
    pub fn new(mc: &Arc<Mc>, ram: &Arc<GuestRam>, client: u64) -> Self {
        Self {
            mc: Arc::clone(mc),
            ram: Arc::clone(ram),
            client,
        }
    }

    /// The memory behind the SMMU, as the processor addresses it.
    pub fn ram(&self) -> &GuestRam {
        &self.ram
    }

    /// Splits `len` bytes at device address `address` into physically
    /// contiguous pieces: `piece(physical, offset, len)`.
    fn pieces(
        &self,
        address: u64,
        len: usize,
        mut piece: impl FnMut(u64, usize, usize) -> Option<()>,
    ) -> Option<()> {
        let mut done = 0;
        while done < len {
            let at = address.checked_add(done as u64)?;
            let physical = self.mc.translate(self.client, at, |table| {
                let mut word = [0; 4];
                self.ram.read(table, &mut word)?;
                Some(u32::from_le_bytes(word))
            })?;
            let in_page = (PAGE_SIZE - at % PAGE_SIZE) as usize;
            let count = in_page.min(len - done);
            piece(physical, done, count)?;
            done += count;
        }
        Some(())
    }

    /// `None` if any of the range is unmapped or outside DRAM.
    pub fn read(&self, address: u64, buffer: &mut [u8]) -> Option<()> {
        self.pieces(address, buffer.len(), |physical, offset, len| {
            self.ram.read(physical, &mut buffer[offset..offset + len])
        })
    }

    /// `None` if any of the range is unmapped or outside DRAM; pieces before
    /// the failing one have been written.
    pub fn write(&self, address: u64, data: &[u8]) -> Option<()> {
        self.pieces(address, data.len(), |physical, offset, len| {
            self.ram.write(physical, &data[offset..offset + len])
        })
    }
}

impl MmioDevice for Mc {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        let value = match word(offset, size)? {
            SMMU_PTB_DATA => self.page_directories.lock().unwrap()[self.selected_asid()],
            // A client's flush completes at once: nothing is in flight.
            CLIENT_HOTRESET_STATUS => self.registers.get(CLIENT_HOTRESET_CTRL)?,
            CLIENT_HOTRESET_STATUS_1 => self.registers.get(CLIENT_HOTRESET_CTRL_1)?,
            register => self.registers.get(register)?,
        };
        Some(value.into())
    }

    fn write(&self, _core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        match word(offset, size)? {
            SMMU_PTB_DATA => {
                self.page_directories.lock().unwrap()[self.selected_asid()] = value as u32
            }
            register => self.registers.set(register, value as u32),
        }
        Some(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    fn write(mc: &Mc, register: u64, value: u32) {
        mc.write(0, register, 4, value.into()).unwrap();
    }

    #[test]
    fn page_directory_register_is_banked_by_the_selected_address_space() {
        let mc = Mc::new(DramSize::Gb4);
        write(&mc, SMMU_PTB_ASID, 3);
        write(&mc, SMMU_PTB_DATA, 0xE000_1234);
        write(&mc, SMMU_PTB_ASID, 4);
        assert_eq!(mc.read(0, SMMU_PTB_DATA, 4), Some(0));
        write(&mc, SMMU_PTB_ASID, 3);
        assert_eq!(mc.read(0, SMMU_PTB_DATA, 4), Some(0xE000_1234));
        assert_eq!(mc.read(0, EMEM_CFG, 4), Some(4096));
    }

    #[test]
    fn translates_pages_and_sections_for_enabled_clients_only() {
        let mc = Mc::new(DramSize::Gb4);
        // ASID 5: directory at frame 0x80100; entry 0x300 -> table at frame
        // 0x80101 whose entry 2 maps frame 0x9ABCD; entry 0x301 is a section.
        let memory = HashMap::from([
            (0x8010_0000 + 0x300 * 4, 0xC000_0000 | ENTRY_NEXT | 0x80101),
            (0x8010_1000 + 2 * 4, 0xC000_0000 | 0x9ABCD),
            (0x8010_0000 + 0x301 * 4, 0xC000_0000 | 0x8_8400),
        ]);
        let read = |address| memory.get(&address).copied();
        write(&mc, SMMU_PTB_ASID, 5);
        write(&mc, SMMU_PTB_DATA, 0xE000_0000 | 0x80100);

        assert_eq!(
            mc.translate(SMMU_DC_ASID, 0xC000_2ABC, read),
            None,
            "client not enabled"
        );
        write(&mc, SMMU_DC_ASID, ASID_ENABLE | 5);
        assert_eq!(
            mc.translate(SMMU_DC_ASID, 0xC000_2ABC, read),
            Some(0x9ABC_DABC)
        );
        assert_eq!(
            mc.translate(SMMU_DC_ASID, 0xC040_1234 + 0x12_0000, read),
            Some(0x8840_0000 + 0x12_1234)
        );
        assert_eq!(
            mc.translate(SMMU_DC_ASID, 0xC000_3000, read),
            None,
            "no page entry"
        );
        assert_eq!(
            mc.translate(SMMU_DC_ASID, 0x1000, read),
            None,
            "no directory entry"
        );

        // Each 4 GiB of device address has its own address space.
        write(&mc, SMMU_DC_ASID, ASID_ENABLE | 5 << 8 | 4);
        assert_eq!(mc.translate(SMMU_DC_ASID, 0xC000_2ABC, read), None);
        assert_eq!(
            mc.translate(SMMU_DC_ASID, 0x1_C000_2ABC, read),
            Some(0x9ABC_DABC)
        );
    }
}
