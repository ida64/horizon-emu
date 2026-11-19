//! Memory controller: DRAM size, the SMMU that maps device-visible
//! addresses to DRAM, and the kernel carveouts that keep devices out of the
//! kernel's memory.
//!
//! The kernel never maps this block; it reaches it through the secure
//! monitor's register call. It reads `MC_EMEM_CFG` for the DRAM size
//! (`0x800d2560`) and programs the SMMU (`0x800cf120`): one page directory
//! per address space ID through the `PTB_ASID`/`PTB_DATA` pair, and one
//! ASID register per client device.
//!
//! The kernel carveouts are general security carveouts the secure monitor
//! programs and locks when the kernel asks for them (ConfigureCarveout):
//! DRAM ranges only the clients of a fixed list may reach (exosphere
//! `secmon_setup.cpp`, `SetupKernelCarveouts`, the client access masks).
//! Every client here is on it but the GPU. A
//! blocked client's reads get zeros and its writes are dropped, and the
//! memory controller records a violation \[assumption: what a blocked
//! request returns is not public\]; here the violation becomes a note.
//! The processor itself is on the list.
//!
//! A client whose ASID register does not enable translation bypasses the
//! SMMU: its addresses are physical (Linux `tegra-smmu.c` clears the
//! enable bit to take a client out of translation). That is every client
//! after reset, the ASID registers reading zero (the NX bootloader's
//! hardware-state check expects `MC_SMMU_SDMMC4A_ASID` and the others at
//! zero), which is how the bootloader's SDMMC and the boot ROM's engines
//! reach memory. `MC_SMMU_CONFIG` is not consulted \[assumption: nothing
//! here programs a client's ASID register without the SMMU enabled\].
//!
//! Physical addresses inside the AHB redirect aperture,
//! `MC_IRAM_BOM..MC_IRAM_TOM` (0x65C/0x660, the T210 TRM's "ARC"; hekate's
//! `mc_enable_ahb_redirect` names the pair), go to the AHB side instead of
//! DRAM, which is how a memory client reaches IRAM. Both read zero after
//! reset; package1ldr opens the aperture at 0x4000_0000..0x8000_0000 (pkg1
//! text `0xB04`), which the bootloader's hardware-state check expects, and
//! the bootloader's SDMMC4 transfers the GPT and the package2 header into
//! IRAM through it. Of the AHB side only
//! IRAM is modeled, once the machine attaches it ([`Mc::attach_iram`]);
//! another AHB address inside the aperture is out of reach. The top is
//! exclusive \[assumption\]. Zero, the registers' value here when the
//! machine starts at the kernel, leaves the aperture empty, as the secure
//! monitor does before the kernel runs.

use std::sync::{Arc, Mutex, OnceLock};

use super::{
    iram::{self, Iram},
    mc_channel,
    regs::{RegisterFile, word},
};
use crate::{
    boot::DramSize,
    bus::MmioDevice,
    memory::GuestRam,
    snapshot::{Snapshot, SnapshotError, decode, encode},
};

pub const BASE: u64 = 0x7001_9000;
pub const SIZE: u64 = 0x1000;

const PAGE_SIZE: u64 = 0x1000;

/// Selects the address space `PTB_DATA` accesses.
const SMMU_PTB_ASID: u64 = 0x1C;
/// Page directory of the selected address space.
const SMMU_PTB_DATA: u64 = 0x20;
/// DRAM size in MiB.
const EMEM_CFG: u64 = 0x50;
/// Bottom and top of the AHB redirect aperture.
const IRAM_BOM: u64 = 0x65C;
const IRAM_TOM: u64 = 0x660;

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
    /// The two kernel carveouts, `(base, size)`; a size of zero is none.
    kernel_carveouts: Mutex<[(u64, u64); KERNEL_CARVEOUT_COUNT]>,
    /// Told about each request a carveout blocks.
    on_violation: Mutex<Option<ViolationReport>>,
    /// IRAM, where the AHB redirect aperture leads: wiring, attached by
    /// a machine that has IRAM to offer.
    iram: OnceLock<Arc<Iram>>,
    /// The per-channel apertures' copies, in a machine that runs the boot
    /// chain ([`mc_channel`](super::mc_channel)).
    channels: Option<mc_channel::Channels>,
}

/// What the memory controller tells about a request a carveout blocked.
pub type ViolationReport = Box<dyn Fn(String) + Send + Sync>;

/// How many kernel carveouts there are.
pub const KERNEL_CARVEOUT_COUNT: usize = 2;

/// The clients the kernel carveouts let through, by their ASID register:
/// the display controllers, host1x, VIC, the SDMMC ports, APB DMA (its
/// `PPCS` AHB master), PCIe (`AFI`), the audio processor (`APE`), HDA and
/// the security engine (`spl` hands it buffers in system memory, which
/// the carveout covers). The GPU is not on the list.
const KERNEL_CARVEOUT_CLIENTS: [u64; 12] = [
    SMMU_DC_ASID,
    SMMU_HOST1X_ASID,
    SMMU_VIC_ASID,
    SMMU_SDMMC1A_ASID,
    SMMU_SDMMC2A_ASID,
    SMMU_SDMMC4A_ASID,
    SMMU_SE_ASID,
    0x270, // PPCS: APB DMA
    0x238, // AFI: PCIe
    0xAB8, // APE
    0x254, // HDA
    0x244, // DCB: the second display controller
];

impl Mc {
    pub fn new(dram: DramSize) -> Self {
        Self {
            registers: RegisterFile::zeroed([(EMEM_CFG, dram.megabytes())]),
            page_directories: Mutex::new([0; ASID_COUNT]),
            kernel_carveouts: Mutex::new([(0, 0); KERNEL_CARVEOUT_COUNT]),
            on_violation: Mutex::new(None),
            iram: OnceLock::new(),
            channels: None,
        }
    }

    /// The block as the SoC comes out of reset and package1ldr finds it,
    /// with its channel apertures ([`mc_channel`](super::mc_channel)).
    pub fn power_on() -> Self {
        Self {
            registers: RegisterFile::zeroed(mc_channel::POWER_ON_STATE),
            channels: Some(mc_channel::Channels::power_on()),
            ..Self::new(DramSize::default())
        }
    }

    /// The per-channel apertures' copies, if the block has them.
    pub fn channels(&self) -> Option<&mc_channel::Channels> {
        self.channels.as_ref()
    }

    /// Programs kernel carveout `index` (the secure monitor does this for
    /// the kernel).
    pub fn set_kernel_carveout(&self, index: usize, base: u64, size: u64) {
        self.kernel_carveouts.lock().unwrap()[index] = (base, size);
    }

    /// Has `report` told about each request a carveout blocks.
    pub fn on_violation(&self, report: ViolationReport) {
        *self.on_violation.lock().unwrap() = Some(report);
    }

    /// Has the AHB redirect aperture lead to `iram`. Done once, by a
    /// machine that has IRAM; later calls change nothing.
    pub fn attach_iram(&self, iram: Arc<Iram>) {
        let _ = self.iram.set(iram);
    }

    /// Whether the ASID register at `client` sends the client's addresses
    /// through the SMMU; if not, they are physical.
    fn translates(&self, client: u64) -> bool {
        self.registers
            .get(client)
            .is_some_and(|asids| asids & ASID_ENABLE != 0)
    }

    /// Where the `len` bytes at `physical` go. `None` if they start inside
    /// the AHB redirect aperture but are not all IRAM there.
    fn target(&self, physical: u64, len: usize) -> Option<Target<'_>> {
        let bottom = u64::from(self.registers.get(IRAM_BOM).unwrap_or(0));
        let top = u64::from(self.registers.get(IRAM_TOM).unwrap_or(0));
        if !(bottom <= physical && physical < top) {
            return Some(Target::Dram);
        }
        let end = physical.checked_add(len as u64)?;
        let iram = self.iram.get()?;
        let offset = physical.checked_sub(iram::BASE)?;
        (end <= top && end <= iram::BASE + iram::SIZE).then_some(Target::Iram(iram, offset as u32))
    }

    /// Whether a carveout keeps `client` from the physical range starting
    /// at `physical`, `len` bytes long, reporting it if so.
    fn blocked(&self, client: u64, physical: u64, len: usize, is_write: bool) -> bool {
        if KERNEL_CARVEOUT_CLIENTS.contains(&client) {
            return false;
        }
        let end = physical + len as u64;
        let hit = self
            .kernel_carveouts
            .lock()
            .unwrap()
            .iter()
            .any(|&(base, size)| size != 0 && physical < base + size && base < end);
        if hit && let Some(report) = &*self.on_violation.lock().unwrap() {
            let what = if is_write { "write to" } else { "read of" };
            report(format!(
                "memory controller: client {client:#x} blocked: {what} {physical:#x}..{end:#x} \
                 in a kernel carveout"
            ));
        }
        hit
    }

    fn selected_asid(&self) -> usize {
        self.registers.get(SMMU_PTB_ASID).unwrap_or(0) as usize % ASID_COUNT
    }

    /// Translates `address` as the client whose ASID register is at
    /// `client` sees it, reading page tables through `read_u32` (physical
    /// address to little-endian word). `None` if the client's translation
    /// is not enabled or the address is unmapped.
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

/// Where a physical address leads.
enum Target<'a> {
    Dram,
    /// Through the AHB redirect aperture, at this offset into IRAM.
    Iram(&'a Iram, u32),
}

fn frame_address(entry: u32) -> u64 {
    u64::from(entry & ENTRY_FRAME) << 12
}

/// Guest memory as one memory client sees it: what a device that masters
/// the bus reads and writes, through the SMMU if its ASID register enables
/// translation, and through the AHB redirect aperture to IRAM.
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

    /// The memory controller this client's requests go through.
    pub fn mc(&self) -> &Mc {
        &self.mc
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
            let physical = if self.mc.translates(self.client) {
                self.mc.translate(self.client, at, |table| {
                    let mut word = [0; 4];
                    self.ram.read(table, &mut word)?;
                    Some(u32::from_le_bytes(word))
                })?
            } else {
                at
            };
            let in_page = (PAGE_SIZE - at % PAGE_SIZE) as usize;
            let count = in_page.min(len - done);
            piece(physical, done, count)?;
            done += count;
        }
        Some(())
    }

    /// `None` if any of the range is unmapped or outside DRAM and IRAM.
    /// What a kernel carveout keeps from this client reads as zeros.
    pub fn read(&self, address: u64, buffer: &mut [u8]) -> Option<()> {
        self.pieces(address, buffer.len(), |physical, offset, len| {
            let piece = &mut buffer[offset..offset + len];
            match self.mc.target(physical, len)? {
                Target::Iram(iram, at) => iram.copy_out(at, piece),
                Target::Dram if self.mc.blocked(self.client, physical, len, false) => {
                    piece.fill(0);
                    Some(())
                }
                Target::Dram => self.ram.read(physical, piece),
            }
        })
    }

    /// `None` if any of the range is unmapped or outside DRAM and IRAM;
    /// pieces before the failing one have been written. What a kernel
    /// carveout keeps from this client is dropped.
    pub fn write(&self, address: u64, data: &[u8]) -> Option<()> {
        self.pieces(address, data.len(), |physical, offset, len| {
            let piece = &data[offset..offset + len];
            match self.mc.target(physical, len)? {
                Target::Iram(iram, at) => iram.load(at, piece),
                Target::Dram if self.mc.blocked(self.client, physical, len, true) => Some(()),
                Target::Dram => self.ram.write(physical, piece),
            }
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
            register => {
                self.registers.set(register, value as u32);
                if let Some(channels) = &self.channels {
                    channels.broadcast(register, value as u32);
                }
            }
        }
        Some(())
    }

    fn snapshot(&self) -> Option<&dyn Snapshot> {
        Some(self)
    }
}

/// Saves the registers, every address space's page directory and the
/// kernel carveouts. Translations are read from guest memory on each
/// access, so nothing is cached to drop.
impl Snapshot for Mc {
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        let page_directories = self.page_directories.lock().unwrap();
        encode(&(
            self.registers.save()?,
            &page_directories[..],
            *self.kernel_carveouts.lock().unwrap(),
        ))
    }

    fn restore(&self, state: &[u8]) -> Result<(), SnapshotError> {
        type Saved = (Vec<u8>, Vec<u32>, [(u64, u64); KERNEL_CARVEOUT_COUNT]);
        let (registers, page_directories, kernel_carveouts): Saved =
            decode("memory controller", state)?;
        let count = page_directories.len();
        let page_directories: [u32; ASID_COUNT] = page_directories.try_into().map_err(|_| {
            SnapshotError::Corrupt(format!(
                "memory controller: {count} address spaces saved, {ASID_COUNT} here"
            ))
        })?;
        self.registers.restore(&registers)?;
        *self.page_directories.lock().unwrap() = page_directories;
        *self.kernel_carveouts.lock().unwrap() = kernel_carveouts;
        Ok(())
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

    #[test]
    fn kernel_carveouts_keep_out_the_gpu_only() {
        let mc = Mc::new(DramSize::Gb4);
        let reported = Arc::new(Mutex::new(Vec::new()));
        let report = Arc::clone(&reported);
        mc.on_violation(Box::new(move |line| report.lock().unwrap().push(line)));
        // Nothing set yet.
        assert!(!mc.blocked(SMMU_GPU_ASID, 0x8006_0000, 4, false));
        mc.set_kernel_carveout(0, 0x8006_0000, 0x1038_0000);
        assert!(mc.blocked(SMMU_GPU_ASID, 0x8006_0000, 4, false));
        assert!(
            mc.blocked(SMMU_GPU_ASID, 0x903D_FFFC, 8, true),
            "overlapping its end"
        );
        assert!(!mc.blocked(SMMU_SE_ASID, 0x8006_0000, 4, true));
        assert!(
            !mc.blocked(SMMU_GPU_ASID, 0x903E_0000, 4, false),
            "just past it"
        );
        assert!(!mc.blocked(SMMU_DC_ASID, 0x8006_0000, 4, false));
        assert!(!mc.blocked(SMMU_SDMMC4A_ASID, 0x8006_0000, 4, true));
        assert_eq!(reported.lock().unwrap().len(), 2);
    }

    #[test]
    fn untranslated_clients_reach_dram_and_iram_through_the_redirect_aperture() {
        let mc = Arc::new(Mc::new(DramSize::Gb4));
        let ram = Arc::new(GuestRam::new(0x8000_0000, 0x1_0000).unwrap());
        let iram = Arc::new(Iram::new());
        mc.attach_iram(Arc::clone(&iram));
        // The ASID register reads zero, as after reset: no translation.
        let memory = DeviceMemory::new(&mc, &ram, SMMU_SDMMC4A_ASID);
        memory.write(0x8000_1FFE, &[1, 2, 3, 4]).unwrap();
        let mut back = [0; 4];
        ram.read(0x8000_1FFE, &mut back).unwrap();
        assert_eq!(back, [1, 2, 3, 4], "physical, across a page");

        // No aperture: IRAM's addresses lead nowhere.
        assert_eq!(memory.write(0x4003_0000, &[0xAA; 8]), None);
        // The aperture package1ldr opens.
        write(&mc, IRAM_BOM, 0x4000_0000);
        write(&mc, IRAM_TOM, 0x8000_0000);
        memory.write(0x4003_0FFC, &[5, 6, 7, 8, 9, 10]).unwrap();
        let mut in_iram = [0; 6];
        iram.copy_out(0x3_0FFC, &mut in_iram).unwrap();
        assert_eq!(in_iram, [5, 6, 7, 8, 9, 10]);
        memory.read(0x4003_0FFE, &mut back).unwrap();
        assert_eq!(back, [7, 8, 9, 10]);
        // Other AHB addresses inside it are not modeled; DRAM is unaffected.
        assert_eq!(memory.read(0x6000_6000, &mut back), None);
        memory.read(0x8000_1FFE, &mut back).unwrap();
        assert_eq!(back, [1, 2, 3, 4]);

        // A client whose translation is enabled sees only its address space.
        write(&mc, SMMU_SDMMC4A_ASID, ASID_ENABLE | 5);
        assert_eq!(memory.read(0x8000_1FFE, &mut back), None);
    }
}
