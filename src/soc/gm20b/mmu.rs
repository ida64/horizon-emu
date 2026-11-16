//! The GPU's memory management unit: GPU virtual addresses through the
//! two-level page tables an instance block names, onto bus addresses, and
//! bus addresses through the SMMU onto memory.
//!
//! Every table, page and instance block the driver hands the GPU is named
//! by its SMMU address with bit 34 set (`NV_GetGpuBackingAddressWithAperture_75310`
//! ORs `1 << 0x22` into the pinned device address; spec §4.6), in the
//! "video memory" aperture. On Tegra that bit selects translation through
//! the SMMU address space of the GPU's memory client (`[RECALLED]` the
//! nvgpu "IOMMU bit"); the driver never hands out anything else, so an
//! address without it, or in another aperture, is refused rather than
//! guessed at.
//!
//! Layouts, all from the driver's own encoders (spec §4.6, annex fifo
//! §2.1-2.4):
//!
//! * Instance block +0x200: page directory address bits 31:12, target in
//!   bits 1:0 (0, video memory), bit 11 set when the address space uses
//!   64 KiB big pages instead of 128 KiB; +0x204 address bits 39:32;
//!   +0x208/+0x20C the highest valid address (`(limit - 1) | 0xFFF`).
//! * Page directory entry, 8 bytes: word 0 holds the big-page table
//!   (aperture bits 1:0, size bits 3:2, address bits 39:12 in bits 31:4),
//!   word 1 the small-page table (aperture bits 1:0, address in bits 31:4).
//!   Aperture 1 is video memory, 0 no table.
//! * Page table entry, 8 bytes: word 0 bit 0 valid, bit 2 read-only,
//!   address bits 39:12 in bits 31:4; word 1 aperture in bits 2:1 (0,
//!   video memory), bit 30 read-disable, bit 31 write-disable.
//!
//! A directory entry covers 128 MiB (64 MiB with 64 KiB big pages); small
//! pages are 4 KiB. The driver points each directory entry at either a
//! small-page or a big-page table, never both (annex fifo §2.1); which one
//! the hardware would use when both are there is not known here, so that
//! is refused. There are no TLBs: every access walks the tables, so the
//! invalidate handshakes have nothing to do.

use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use super::super::mc::DeviceMemory;
use crate::maxwell::{GpuMemory, Unmodeled};

/// Bus address bit that sends an access through the SMMU.
pub const SMMU_BIT: u64 = 1 << 34;

const SMALL_PAGE_SHIFT: u32 = 12;

const INSTANCE_DIRECTORY_LO: u64 = 0x200;
const INSTANCE_DIRECTORY_HI: u64 = 0x204;
const INSTANCE_LIMIT_LO: u64 = 0x208;
const INSTANCE_LIMIT_HI: u64 = 0x20C;
const INSTANCE_TARGET: u32 = 3;
const INSTANCE_BIG_PAGE_64K: u32 = 1 << 11;
const ADDRESS_LO: u32 = 0xFFFF_F000;
const ADDRESS_HI: u32 = 0xFF;

/// Aperture field of directory entries; 1 is video memory.
const PDE_APERTURE: u32 = 3;
const PDE_APERTURE_VIDEO: u32 = 1;
/// Size field of the big-page half: 0 is a full table.
const PDE_SIZE: u32 = 3 << 2;
/// Address bits 39:12 of a table or page, in bits 31:4 of an entry word.
const ENTRY_ADDRESS: u32 = 0xFFFF_FFF0;

const PTE_VALID: u32 = 1 << 0;
const PTE_READ_ONLY: u32 = 1 << 2;
const PTE_APERTURE: u32 = 3 << 1;
const PTE_READ_DISABLE: u32 = 1 << 30;
const PTE_WRITE_DISABLE: u32 = 1 << 31;

/// Memory as the GPU's memory interface reaches it, by bus address.
pub trait BusMemory: Send + Sync {
    fn read(&self, address: u64, buffer: &mut [u8]) -> Option<()>;
    fn write(&self, address: u64, data: &[u8]) -> Option<()>;
}

/// Reads the little-endian word at bus address `address`.
pub fn read_u32(bus: &dyn BusMemory, address: u64) -> Result<u32, Unmodeled> {
    let mut word = [0; 4];
    bus.read(address, &mut word)
        .ok_or_else(|| unreachable_memory(address, 4, false))?;
    Ok(u32::from_le_bytes(word))
}

pub fn write_u32(bus: &dyn BusMemory, address: u64, value: u32) -> Result<(), Unmodeled> {
    bus.write(address, &value.to_le_bytes())
        .ok_or_else(|| unreachable_memory(address, 4, true))
}

fn unreachable_memory(address: u64, len: usize, write: bool) -> Unmodeled {
    let access = if write { "write of" } else { "read of" };
    Unmodeled(format!(
        "GPU {access} {len:#x} bytes at bus address {address:#x}, not all of which is memory \
         behind the SMMU (bit 34 clear, or unmapped in the GPU's SMMU address space)"
    ))
}

/// The GPU's side of the memory controller: SMMU addresses with bit 34
/// set, through the address space the kernel attached to the GPU's memory
/// client.
pub struct SmmuBus(pub DeviceMemory);

/// The SMMU address of the `len` bytes at bus address `address`, if they
/// all carry the SMMU bit and nothing above it.
fn smmu_address(address: u64, len: usize) -> Option<u64> {
    let last = address.checked_add(len.saturating_sub(1) as u64)?;
    let through_smmu = |at: u64| at & !(SMMU_BIT - 1) == SMMU_BIT;
    (through_smmu(address) && through_smmu(last)).then_some(address & (SMMU_BIT - 1))
}

impl BusMemory for SmmuBus {
    fn read(&self, address: u64, buffer: &mut [u8]) -> Option<()> {
        self.0.read(smmu_address(address, buffer.len())?, buffer)
    }

    fn write(&self, address: u64, data: &[u8]) -> Option<()> {
        self.0.write(smmu_address(address, data.len())?, data)
    }
}

/// Size of the big pages of an address space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BigPage {
    K64,
    K128,
}

impl BigPage {
    fn shift(self) -> u32 {
        match self {
            Self::K64 => 16,
            Self::K128 => 17,
        }
    }

    /// Address bits a directory entry covers.
    fn directory_shift(self) -> u32 {
        self.shift() + 10
    }
}

/// One GPU virtual address space, as an instance block describes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AddressSpace {
    /// Bus address of the page directory.
    pub directory: u64,
    pub big_page: BigPage,
    /// Highest valid virtual address.
    pub limit: u64,
}

/// Where a virtual address lands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Translation {
    /// Bus address.
    pub address: u64,
    /// Bytes from there to the end of the page.
    pub left: u64,
}

impl AddressSpace {
    /// The address space of the instance block at bus address `instance`.
    pub fn from_instance(bus: &dyn BusMemory, instance: u64) -> Result<Self, Unmodeled> {
        let word = |offset| read_u32(bus, instance + offset);
        let directory_lo = word(INSTANCE_DIRECTORY_LO)?;
        if directory_lo & INSTANCE_TARGET != 0 {
            return Err(Unmodeled(format!(
                "GPU instance block at {instance:#x} puts its page directory in aperture {} \
                 (only video memory is modeled)",
                directory_lo & INSTANCE_TARGET
            )));
        }
        let wide = |lo: u32, hi: u32| u64::from(hi & ADDRESS_HI) << 32 | u64::from(lo & ADDRESS_LO);
        Ok(Self {
            directory: wide(directory_lo, word(INSTANCE_DIRECTORY_HI)?),
            big_page: if directory_lo & INSTANCE_BIG_PAGE_64K != 0 {
                BigPage::K64
            } else {
                BigPage::K128
            },
            limit: wide(word(INSTANCE_LIMIT_LO)?, word(INSTANCE_LIMIT_HI)?) | 0xFFF,
        })
    }

    /// Walks the page tables for `address`; `write` says whether the access
    /// writes.
    pub fn translate(
        &self,
        bus: &dyn BusMemory,
        address: u64,
        write: bool,
    ) -> Result<Translation, Unmodeled> {
        let fault = |what: &str| {
            Unmodeled(format!(
                "GPU MMU fault (not delivered by the model): virtual address {address:#x} in \
                 the address space with page directory {:#x}: {what}",
                self.directory
            ))
        };
        if address > self.limit {
            return Err(fault(&format!("beyond its limit {:#x}", self.limit)));
        }
        let directory_shift = self.big_page.directory_shift();
        let entry = self.directory + (address >> directory_shift) * 8;
        let (big, small) = (read_u32(bus, entry)?, read_u32(bus, entry + 4)?);
        let table = |word: u32| u64::from(word & ENTRY_ADDRESS) << 8;
        let (table, page_shift) = match (big & PDE_APERTURE, small & PDE_APERTURE) {
            (0, 0) => return Err(fault("no page table")),
            (PDE_APERTURE_VIDEO, 0) if big & PDE_SIZE == 0 => (table(big), self.big_page.shift()),
            (0, PDE_APERTURE_VIDEO) => (table(small), SMALL_PAGE_SHIFT),
            (PDE_APERTURE_VIDEO, 0 | PDE_APERTURE_VIDEO) => {
                return Err(fault(&format!(
                    "directory entry {big:#010x} {small:#010x} names both page tables or a \
                     partial one, which the model does not resolve"
                )));
            }
            _ => {
                return Err(fault(&format!(
                    "directory entry {big:#010x} {small:#010x} is not in video memory"
                )));
            }
        };
        let within = address & ((1 << directory_shift) - 1);
        let entry = table + (within >> page_shift) * 8;
        let (low, high) = (read_u32(bus, entry)?, read_u32(bus, entry + 4)?);
        if low & PTE_VALID == 0 {
            return Err(fault(&format!(
                "page table entry {low:#010x} {high:#010x} is invalid"
            )));
        }
        if high & PTE_APERTURE != 0 {
            return Err(fault(&format!(
                "page table entry {low:#010x} {high:#010x} is not in video memory"
            )));
        }
        let denied = if write {
            low & PTE_READ_ONLY != 0 || high & PTE_WRITE_DISABLE != 0
        } else {
            high & PTE_READ_DISABLE != 0
        };
        if denied {
            return Err(fault(&format!(
                "page table entry {low:#010x} {high:#010x} forbids the access"
            )));
        }
        let page_mask = (1 << page_shift) - 1;
        Ok(Translation {
            address: (u64::from(low & ENTRY_ADDRESS) << 8) + (address & page_mask),
            left: (1 << page_shift) - (address & page_mask),
        })
    }

    /// Splits `len` bytes at `address` into pieces that each lie in one
    /// page: `piece(bus address, offset, len)`.
    fn pieces(
        &self,
        bus: &dyn BusMemory,
        address: u64,
        len: usize,
        write: bool,
        mut piece: impl FnMut(u64, usize, usize) -> Result<(), Unmodeled>,
    ) -> Result<(), Unmodeled> {
        let mut done = 0;
        while done < len {
            let at = address.wrapping_add(done as u64);
            let page = self.translate(bus, at, write)?;
            let count = usize::try_from(page.left).map_or(len - done, |left| left.min(len - done));
            piece(page.address, done, count)?;
            done += count;
        }
        Ok(())
    }

    pub fn read(
        &self,
        bus: &dyn BusMemory,
        address: u64,
        buffer: &mut [u8],
    ) -> Result<(), Unmodeled> {
        self.pieces(bus, address, buffer.len(), false, |at, offset, len| {
            bus.read(at, &mut buffer[offset..offset + len])
                .ok_or_else(|| unreachable_memory(at, len, false))
        })
    }

    pub fn write(&self, bus: &dyn BusMemory, address: u64, data: &[u8]) -> Result<(), Unmodeled> {
        self.pieces(bus, address, data.len(), true, |at, offset, len| {
            bus.write(at, &data[offset..offset + len])
                .ok_or_else(|| unreachable_memory(at, len, true))
        })
    }
}

/// One address space as the engines of a channel use it. A refused access
/// leaves its reason behind for the report that follows it.
pub struct ChannelMemory<'a> {
    pub space: AddressSpace,
    pub bus: &'a dyn BusMemory,
    pub last_fault: Mutex<Option<Unmodeled>>,
}

impl<'a> ChannelMemory<'a> {
    pub fn new(space: AddressSpace, bus: &'a dyn BusMemory) -> Self {
        Self {
            space,
            bus,
            last_fault: Mutex::new(None),
        }
    }

    fn keep(&self, outcome: Result<(), Unmodeled>) -> Option<()> {
        outcome
            .map_err(|fault| *self.last_fault.lock().unwrap() = Some(fault))
            .ok()
    }

    /// The reason the last refused access was refused, if any.
    pub fn take_fault(&self) -> Option<Unmodeled> {
        self.last_fault.lock().unwrap().take()
    }
}

impl GpuMemory for ChannelMemory<'_> {
    fn read(&self, address: u64, buffer: &mut [u8]) -> Option<()> {
        self.keep(self.space.read(self.bus, address, buffer))
    }

    fn write(&self, address: u64, data: &[u8]) -> Option<()> {
        self.keep(self.space.write(self.bus, address, data))
    }
}

#[cfg(test)]
pub(super) mod tests {
    use std::collections::HashMap;

    use super::*;

    /// Bus memory for tests: bytes at SMMU addresses, the SMMU mapping
    /// everything one to one.
    #[derive(Default)]
    pub struct TestBus(pub Mutex<HashMap<u64, u8>>);

    impl BusMemory for TestBus {
        fn read(&self, address: u64, buffer: &mut [u8]) -> Option<()> {
            let start = smmu_address(address, buffer.len())?;
            let memory = self.0.lock().unwrap();
            for (index, byte) in buffer.iter_mut().enumerate() {
                *byte = *memory.get(&(start + index as u64)).unwrap_or(&0);
            }
            Some(())
        }

        fn write(&self, address: u64, data: &[u8]) -> Option<()> {
            let start = smmu_address(address, data.len())?;
            let mut memory = self.0.lock().unwrap();
            for (index, byte) in data.iter().enumerate() {
                memory.insert(start + index as u64, *byte);
            }
            Some(())
        }
    }

    impl TestBus {
        pub fn put(&self, address: u64, value: u32) {
            write_u32(self, address, value).unwrap();
        }

        pub fn get(&self, address: u64) -> u32 {
            read_u32(self, address).unwrap()
        }

        /// Encodes the directory entry pointing at the small-page table at
        /// `table`, as `NV_EncodeGpuPageDirectoryEntry_78080` does.
        pub fn small_table(&self, entry: u64, table: u64) {
            self.put(entry, 0);
            self.put(entry + 4, ((table >> 8) as u32 & ENTRY_ADDRESS) | 5);
        }

        pub fn big_table(&self, entry: u64, table: u64) {
            self.put(entry, ((table >> 8) as u32 & ENTRY_ADDRESS) | 1);
            self.put(entry + 4, 8);
        }

        /// Encodes a valid read-write page as `NV_EncodeGpuPageTableEntry_78130`
        /// does (kind 0, cached).
        pub fn page(&self, entry: u64, page: u64) {
            self.put(entry, ((page >> 8) as u32 & ENTRY_ADDRESS) | 1);
            self.put(entry + 4, 0);
        }

        /// An instance block whose address space is `directory`, as
        /// `NV_EncodeInstancePageDirectory_77730` and
        /// `NV_CommitGpuInstancePageDirectory_766A0` write it.
        pub fn instance(&self, instance: u64, directory: u64, big_page: BigPage, limit: u64) {
            let big = if big_page == BigPage::K64 { 1 << 11 } else { 0 };
            self.put(instance + 0x200, directory as u32 & ADDRESS_LO | 4 | big);
            self.put(instance + 0x204, (directory >> 32) as u32);
            self.put(instance + 0x208, ((limit - 1) | 0xFFF) as u32);
            self.put(instance + 0x20C, ((limit - 1) >> 32) as u32);
        }
    }

    const IOVA: u64 = SMMU_BIT;

    #[test]
    fn small_and_big_pages_of_the_128k_geometry() {
        let bus = TestBus::default();
        let instance = IOVA | 0x1000;
        let directory = IOVA | 0x2000;
        bus.instance(instance, directory, BigPage::K128, 1 << 37);
        let space = AddressSpace::from_instance(&bus, instance).unwrap();
        assert_eq!(space.big_page, BigPage::K128);
        assert_eq!(space.directory, directory);
        assert_eq!(space.limit, (1 << 37) - 1);
        // Directory entry 1 covers [128 MiB, 256 MiB): small pages.
        bus.small_table(directory + 8, IOVA | 0x10_0000);
        bus.page(IOVA | (0x10_0000 + 3 * 8), IOVA | 0x7_6543_0000);
        let at = space.translate(&bus, 0x800_3123, false).unwrap();
        assert_eq!(at.address, IOVA | 0x7_6543_0123);
        assert_eq!(at.left, 0x1000 - 0x123);
        // Directory entry 128 covers [16 GiB, ...): 128 KiB big pages.
        bus.big_table(directory + 128 * 8, IOVA | 0x20_0000);
        bus.page(IOVA | (0x20_0000 + 2 * 8), IOVA | 0x1_0006_0000);
        let at = space.translate(&bus, (1 << 34) + 0x5_4321, true).unwrap();
        assert_eq!(at.address, IOVA | 0x1_0007_4321);
        assert_eq!(at.left, 0x2_0000 - 0x1_4321);
    }

    #[test]
    fn big_pages_of_the_64k_geometry() {
        let bus = TestBus::default();
        let instance = IOVA | 0x1000;
        let directory = IOVA | 0x3000;
        bus.instance(instance, directory, BigPage::K64, 0x800_0000);
        let space = AddressSpace::from_instance(&bus, instance).unwrap();
        assert_eq!(space.big_page, BigPage::K64);
        // Directory entry 1 covers [64 MiB, 128 MiB); big page 1 of it.
        bus.big_table(directory + 8, IOVA | 0x40_0000);
        bus.page(IOVA | (0x40_0000 + 8), IOVA | 0x9_0000);
        let at = space.translate(&bus, 0x401_2345, false).unwrap();
        assert_eq!(at.address, IOVA | 0x9_2345);
        assert_eq!(at.left, 0x1_0000 - 0x2345);
        // Past the limit.
        assert!(space.translate(&bus, 0x800_0000, false).is_err());
    }

    #[test]
    fn unmapped_entries_and_physical_addresses_fault() {
        let bus = TestBus::default();
        let instance = IOVA | 0x1000;
        let directory = IOVA | 0x2000;
        bus.instance(instance, directory, BigPage::K128, 1 << 37);
        let space = AddressSpace::from_instance(&bus, instance).unwrap();
        // No directory entry.
        assert!(space.translate(&bus, 0x1000, false).is_err());
        // A table with an invalid entry, and a sparse one ({0, 1}).
        bus.small_table(directory, IOVA | 0x10_0000);
        bus.put(IOVA | (0x10_0000 + 8 + 4), 1);
        assert!(space.translate(&bus, 0x0000, false).is_err());
        assert!(space.translate(&bus, 0x1000, false).is_err());
        // A page without the SMMU bit: the table walk succeeds, the access
        // does not.
        bus.page(IOVA | (0x10_0000 + 2 * 8), 0x8000_0000);
        assert_eq!(
            space.translate(&bus, 0x2000, false).unwrap().address,
            0x8000_0000
        );
        assert!(space.read(&bus, 0x2000, &mut [0; 4]).is_err());
        // A directory outside the SMMU.
        bus.instance(instance, 0x2000, BigPage::K128, 1 << 37);
        let space = AddressSpace::from_instance(&bus, instance).unwrap();
        assert!(space.translate(&bus, 0x2000, false).is_err());
        // Both tables in one entry.
        bus.instance(instance, directory, BigPage::K128, 1 << 37);
        bus.put(directory, ((0x30_0000u64 >> 8) as u32) | 1);
        assert!(space.translate(&bus, 0x2000, false).is_err());
    }

    #[test]
    fn read_only_pages_refuse_writes() {
        let bus = TestBus::default();
        let instance = IOVA | 0x1000;
        let directory = IOVA | 0x2000;
        bus.instance(instance, directory, BigPage::K128, 1 << 37);
        let space = AddressSpace::from_instance(&bus, instance).unwrap();
        bus.small_table(directory, IOVA | 0x10_0000);
        bus.put(IOVA | 0x10_0000, ((0x5000u64 >> 8) as u32) | 1 | 4);
        assert!(space.translate(&bus, 0x10, false).is_ok());
        assert!(space.translate(&bus, 0x10, true).is_err());
    }

    #[test]
    fn accesses_split_at_page_boundaries() {
        let bus = TestBus::default();
        let instance = IOVA | 0x1000;
        let directory = IOVA | 0x2000;
        bus.instance(instance, directory, BigPage::K128, 1 << 37);
        let space = AddressSpace::from_instance(&bus, instance).unwrap();
        bus.small_table(directory, IOVA | 0x10_0000);
        // Virtual pages 0 and 1 land far apart.
        bus.page(IOVA | 0x10_0000, IOVA | 0x50_0000);
        bus.page(IOVA | (0x10_0000 + 8), IOVA | 0x30_0000);
        let memory = ChannelMemory::new(space, &bus);
        memory.write(0xFFE, &[1, 2, 3, 4]).unwrap();
        assert_eq!(bus.get(IOVA | 0x50_0FFC) >> 16, 0x0201);
        assert_eq!(bus.get(IOVA | 0x30_0000) & 0xFFFF, 0x0403);
        let mut back = [0; 4];
        memory.read(0xFFE, &mut back).unwrap();
        assert_eq!(back, [1, 2, 3, 4]);
        assert!(memory.read(0x2000, &mut back).is_none());
        assert!(memory.take_fault().is_some());
    }
}
