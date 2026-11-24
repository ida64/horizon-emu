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

use std::sync::{
    Arc, Mutex, OnceLock,
    atomic::{AtomicU64, Ordering, fence},
};

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
/// NVDEC client ASID (`nvdec` swgroup), from the same public swgroup table.
pub const SMMU_NVDEC_ASID: u64 = 0xAB4;
/// NVENC client ASID (`nvenc` swgroup), from the same public swgroup table.
pub const SMMU_NVENC_ASID: u64 = 0x264;
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
    /// chain ([`mc_channel`]).
    channels: Option<mc_channel::Channels>,
    /// Moves on every register write, carveout change and restore: what
    /// each [`DeviceMemory`] has cached from the block is out of date then.
    generation: AtomicU64,
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
            generation: AtomicU64::new(0),
        }
    }

    /// The block as the SoC comes out of reset and package1ldr finds it,
    /// with its channel apertures ([`mc_channel`]).
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
        self.changed();
    }

    /// Something a client's view depends on changed: every cached view and
    /// translation is dropped before the next access. Called after the
    /// change, so that a client that sees the new generation also sees it.
    fn changed(&self) {
        self.generation.fetch_add(1, Ordering::Release);
    }

    /// How many times something a translation depends on changed (a
    /// register write, a carveout, a restore); what was cached at another
    /// count may be out of date.
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
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

    /// What [`DeviceMemory`] keeps for the client whose ASID register is at
    /// `client`, as of `generation`.
    fn view(&self, client: u64, generation: u64) -> View {
        let exempt = KERNEL_CARVEOUT_CLIENTS.contains(&client);
        View {
            generation,
            asids: self
                .registers
                .get(client)
                .filter(|asids| asids & ASID_ENABLE != 0),
            aperture: (
                u64::from(self.registers.get(IRAM_BOM).unwrap_or(0)),
                u64::from(self.registers.get(IRAM_TOM).unwrap_or(0)),
            ),
            carveouts: if exempt {
                [(0, 0); KERNEL_CARVEOUT_COUNT]
            } else {
                *self.kernel_carveouts.lock().unwrap()
            },
        }
    }

    /// Translates `address` as the client whose ASID register is at
    /// `client` sees it, reading page tables through `read_u32` (physical
    /// address to little-endian word). `None` if the client's translation
    /// is not enabled or the address is unmapped. [`DeviceMemory`] caches
    /// what this works out; the tests compare the two.
    #[cfg(test)]
    fn translate(
        &self,
        client: u64,
        address: u64,
        read_u32: impl Fn(u64) -> Option<u32>,
    ) -> Option<u64> {
        let asids = self
            .registers
            .get(client)
            .filter(|asids| asids & ASID_ENABLE != 0)?;
        self.walk(asids, address, read_u32)
    }

    /// Translates `address` through the address spaces of ASID register
    /// value `asids`, reading page tables through `read_u32`.
    fn walk(&self, asids: u32, address: u64, read_u32: impl Fn(u64) -> Option<u32>) -> Option<u64> {
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

/// Translations a client keeps, by device page (4 KiB, direct-mapped).
const TLB_ENTRIES: usize = 1024;

/// What a client's accesses depend on in the memory controller, read at
/// [`Mc::generation`] `generation`.
///
/// The SMMU's own TLB and page table cache are flushed by software after
/// it changes page tables (`MC_SMMU_TLB_FLUSH`, `MC_SMMU_PTC_FLUSH`, Linux
/// `tegra-smmu.c`), and changing a page directory or a client's ASID
/// register is a register write too. Every register write moves the
/// generation, and a [`DeviceMemory`] uses a view or a translation only at
/// the generation it was read at, so a translation is never kept past a
/// flush. Failed translations are not kept: a page mapped later is found
/// by the next access.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct View {
    generation: u64,
    /// The client's ASID register, if it enables translation.
    asids: Option<u32>,
    /// The AHB redirect aperture, bottom and top.
    aperture: (u64, u64),
    /// The kernel carveouts that can block this client (none set for a
    /// client on the list).
    carveouts: [(u64, u64); KERNEL_CARVEOUT_COUNT],
}

/// [`View`] as the words of a [`SeqLock`].
const VIEW_WORDS: usize = 4 + 2 * KERNEL_CARVEOUT_COUNT;
/// Bit of the packed ASID register word saying translation is enabled.
const VIEW_ASIDS_SET: u64 = 1 << 32;

impl View {
    fn pack(&self) -> [u64; VIEW_WORDS] {
        let mut words = [0; VIEW_WORDS];
        words[0] = self.generation;
        words[1] = self
            .asids
            .map_or(0, |asids| VIEW_ASIDS_SET | u64::from(asids));
        (words[2], words[3]) = self.aperture;
        for (index, (base, size)) in self.carveouts.into_iter().enumerate() {
            (words[4 + 2 * index], words[5 + 2 * index]) = (base, size);
        }
        words
    }

    fn unpack(words: [u64; VIEW_WORDS]) -> View {
        View {
            generation: words[0],
            asids: (words[1] & VIEW_ASIDS_SET != 0).then_some(words[1] as u32),
            aperture: (words[2], words[3]),
            carveouts: std::array::from_fn(|index| (words[4 + 2 * index], words[5 + 2 * index])),
        }
    }
}

/// `N` words read and written whole without a lock: a sequence count, odd
/// while a write is under way. A read that sees it odd, or changed across
/// the words, gets nothing (the caller works the value out afresh); a
/// write that finds another under way leaves the words to that one. So a
/// read gets the words some write stored, never a mix.
struct SeqLock<const N: usize> {
    sequence: AtomicU64,
    words: [AtomicU64; N],
}

impl<const N: usize> SeqLock<N> {
    fn new(words: [u64; N]) -> Self {
        Self {
            sequence: AtomicU64::new(0),
            words: words.map(AtomicU64::new),
        }
    }

    fn load(&self) -> Option<[u64; N]> {
        let before = self.sequence.load(Ordering::Acquire);
        if before & 1 != 0 {
            return None;
        }
        let words = std::array::from_fn(|index| self.words[index].load(Ordering::Relaxed));
        fence(Ordering::Acquire);
        (self.sequence.load(Ordering::Relaxed) == before).then_some(words)
    }

    fn store(&self, words: [u64; N]) {
        let before = self.sequence.load(Ordering::Relaxed);
        if before & 1 != 0
            || self
                .sequence
                .compare_exchange(before, before + 1, Ordering::Acquire, Ordering::Relaxed)
                .is_err()
        {
            return;
        }
        fence(Ordering::Release);
        for (slot, word) in self.words.iter().zip(words) {
            slot.store(word, Ordering::Relaxed);
        }
        self.sequence.store(before + 2, Ordering::Release);
    }
}

impl View {
    /// Where the `len` bytes at physical `physical` go, as [`Mc`]'s
    /// aperture registers were read. `None` if they start inside the AHB
    /// redirect aperture but are not all IRAM there.
    fn target<'a>(&self, mc: &'a Mc, physical: u64, len: usize) -> Option<Target<'a>> {
        let (bottom, top) = self.aperture;
        if !(bottom <= physical && physical < top) {
            return Some(Target::Dram);
        }
        let end = physical.checked_add(len as u64)?;
        let iram = mc.iram.get()?;
        let offset = physical.checked_sub(iram::BASE)?;
        (end <= top && end <= iram::BASE + iram::SIZE).then_some(Target::Iram(iram, offset as u32))
    }

    /// Whether a carveout may keep the client from `len` bytes at
    /// `physical` ([`Mc::blocked`] decides and reports).
    fn may_block(&self, physical: u64, len: usize) -> bool {
        let end = physical + len as u64;
        self.carveouts
            .iter()
            .any(|&(base, size)| size != 0 && physical < base + size && base < end)
    }
}

/// Where a physical address leads.
enum Target<'a> {
    Dram,
    /// Through the AHB redirect aperture, at this offset into IRAM.
    Iram(&'a Iram, u32),
}

/// Where one piece of a client's access goes.
enum Place<'a> {
    Dram(u64),
    Iram(&'a Iram, u32),
    /// DRAM a kernel carveout keeps from the client.
    Blocked,
}

fn frame_address(entry: u32) -> u64 {
    u64::from(entry & ENTRY_FRAME) << 12
}

/// Guest memory as one memory client sees it: what a device that masters
/// the bus reads and writes, through the SMMU if its ASID register enables
/// translation, and through the AHB redirect aperture to IRAM.
///
/// Accesses take no lock: the client's [`View`] and its translations are
/// kept in sequence locks ([`SeqLock`]), each with the generation it was
/// read at, and used only at that generation.
pub struct DeviceMemory {
    mc: Arc<Mc>,
    ram: Arc<GuestRam>,
    client: u64,
    view: SeqLock<VIEW_WORDS>,
    /// By device page: the generation, device page + 1 (0: empty) and
    /// physical page address of a translation.
    tlb: Box<[SeqLock<3>]>,
}

impl DeviceMemory {
    /// `client` is the offset of the device's ASID register.
    pub fn new(mc: &Arc<Mc>, ram: &Arc<GuestRam>, client: u64) -> Self {
        let view = View {
            // Read before the first access.
            generation: u64::MAX,
            ..mc.view(client, 0)
        };
        Self {
            mc: Arc::clone(mc),
            ram: Arc::clone(ram),
            client,
            view: SeqLock::new(view.pack()),
            tlb: (0..TLB_ENTRIES).map(|_| SeqLock::new([0; 3])).collect(),
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

    /// The client's [`View`] at `generation`: the one kept, or read from
    /// the block and kept.
    fn view(&self, generation: u64) -> View {
        if let Some(words) = self.view.load()
            && words[0] == generation
        {
            return View::unpack(words);
        }
        let view = self.mc.view(self.client, generation);
        self.view.store(view.pack());
        view
    }

    /// The physical address of device address `address` through `view`;
    /// `None` if the client translates and `address` is unmapped.
    fn physical(&self, view: &View, address: u64) -> Option<u64> {
        let Some(asids) = view.asids else {
            return Some(address);
        };
        let page = address / PAGE_SIZE;
        let slot = &self.tlb[page as usize % TLB_ENTRIES];
        if let Some([generation, kept, frame]) = slot.load()
            && generation == view.generation
            && kept == page + 1
        {
            return Some(frame + address % PAGE_SIZE);
        }
        let physical = self.mc.walk(asids, address, |table| {
            let mut word = [0; 4];
            self.ram.read(table, &mut word)?;
            Some(u32::from_le_bytes(word))
        })?;
        slot.store([view.generation, page + 1, physical - address % PAGE_SIZE]);
        Some(physical)
    }

    /// Splits `len` bytes at device address `address` into pieces:
    /// `piece(place, offset, len)`, a page each, but consecutive DRAM pages
    /// that lie together physically in one piece. `None` at the first page
    /// that is unmapped or outside DRAM and IRAM, the pieces before it
    /// given.
    fn pieces(
        &self,
        address: u64,
        len: usize,
        is_write: bool,
        mut piece: impl FnMut(Place<'_>, usize, usize) -> Option<()>,
    ) -> Option<()> {
        let view = self.view(self.mc.generation.load(Ordering::Acquire));
        // DRAM pages not yet given: physical address, offset, length.
        let mut run: Option<(u64, usize, usize)> = None;
        let mut done = 0;
        while done < len {
            let place = address.checked_add(done as u64).and_then(|at| {
                let count = ((PAGE_SIZE - at % PAGE_SIZE) as usize).min(len - done);
                let physical = self.physical(&view, at)?;
                Some((
                    match view.target(&self.mc, physical, count)? {
                        Target::Iram(iram, offset) => Place::Iram(iram, offset),
                        Target::Dram
                            if view.may_block(physical, count)
                                && self.mc.blocked(self.client, physical, count, is_write) =>
                        {
                            Place::Blocked
                        }
                        Target::Dram => Place::Dram(physical),
                    },
                    count,
                ))
            });
            let flush =
                |run: &mut Option<(u64, usize, usize)>,
                 piece: &mut dyn FnMut(Place<'_>, usize, usize) -> Option<()>| {
                    match run.take() {
                        Some((physical, offset, length)) => {
                            piece(Place::Dram(physical), offset, length)
                        }
                        None => Some(()),
                    }
                };
            let Some((place, count)) = place else {
                flush(&mut run, &mut piece)?;
                return None;
            };
            match place {
                Place::Dram(physical) => match &mut run {
                    Some((start, _, length)) if *start + *length as u64 == physical => {
                        *length += count
                    }
                    _ => {
                        flush(&mut run, &mut piece)?;
                        run = Some((physical, done, count));
                    }
                },
                place => {
                    flush(&mut run, &mut piece)?;
                    piece(place, done, count)?;
                }
            }
            done += count;
        }
        if let Some((physical, offset, length)) = run {
            piece(Place::Dram(physical), offset, length)?;
        }
        Some(())
    }

    /// `None` if any of the range is unmapped or outside DRAM and IRAM.
    /// What a kernel carveout keeps from this client reads as zeros.
    pub fn read(&self, address: u64, buffer: &mut [u8]) -> Option<()> {
        self.pieces(address, buffer.len(), false, |place, offset, len| {
            let piece = &mut buffer[offset..offset + len];
            match place {
                Place::Iram(iram, at) => iram.copy_out(at, piece),
                Place::Blocked => {
                    piece.fill(0);
                    Some(())
                }
                Place::Dram(physical) => self.ram.read(physical, piece),
            }
        })
    }

    /// Whether the range holds `expected` ([`DeviceMemory::read`] of it
    /// would give it), compared in place where it is DRAM.
    pub fn equals(&self, address: u64, expected: &[u8]) -> Option<bool> {
        let mut same = true;
        let mut scratch = Vec::new();
        self.pieces(address, expected.len(), false, |place, offset, len| {
            let piece = &expected[offset..offset + len];
            if !same {
                return Some(());
            }
            same = match place {
                Place::Dram(physical) => self.ram.equals(physical, piece)?,
                Place::Iram(iram, at) => {
                    scratch.resize(len, 0);
                    iram.copy_out(at, &mut scratch)?;
                    scratch == piece
                }
                Place::Blocked => piece.iter().all(|&byte| byte == 0),
            };
            Some(())
        })?;
        Some(same)
    }

    /// `None` if any of the range is unmapped or outside DRAM and IRAM;
    /// pages before the failing one have been written. What a kernel
    /// carveout keeps from this client is dropped.
    pub fn write(&self, address: u64, data: &[u8]) -> Option<()> {
        self.pieces(address, data.len(), true, |place, offset, len| {
            let piece = &data[offset..offset + len];
            match place {
                Place::Iram(iram, at) => iram.load(at, piece),
                Place::Blocked => Some(()),
                // A run that is not all RAM: its pages before the first
                // outside it are written, as page by page.
                Place::Dram(physical) => self.ram.write(physical, piece).or_else(|| {
                    let mut at = 0;
                    while at < piece.len() {
                        let page = physical + at as u64;
                        let count = ((PAGE_SIZE - page % PAGE_SIZE) as usize).min(piece.len() - at);
                        self.ram.write(page, &piece[at..at + count])?;
                        at += count;
                    }
                    Some(())
                }),
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
        // Among them the TLB and PTC flushes after a page table change.
        self.changed();
        Some(())
    }

    fn snapshot(&self) -> Option<&dyn Snapshot> {
        Some(self)
    }
}

/// Saves the registers, every address space's page directory and the
/// kernel carveouts. What clients cached of them is dropped on restore.
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
        self.changed();
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

    /// A tiny deterministic generator for the randomized tests.
    struct Xorshift(u64);

    impl Xorshift {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    /// The cached translations of [`DeviceMemory`] agree with walking the
    /// tables afresh ([`Mc::translate`]) after every register write, over
    /// random page tables, page table changes, page directory and ASID
    /// register changes, with reads checked byte for byte.
    #[test]
    fn cached_translations_equal_fresh_walks_after_every_register_write() {
        let mc = Arc::new(Mc::new(DramSize::Gb4));
        let ram = Arc::new(GuestRam::new(0x8000_0000, 0x40_0000).unwrap());
        // Distinct contents per word, to tell pages apart.
        for word in 0..0x10_0000u64 {
            ram.write(
                0x8000_0000 + word * 4,
                &(word as u32 ^ 0x5A5A_0000).to_le_bytes(),
            )
            .unwrap();
        }
        let memory = DeviceMemory::new(&mc, &ram, SMMU_GPU_ASID);
        let mut rng = Xorshift(0x9E37_79B9_7F4A_7C15);
        // Directories of ASIDs 5 and 6 at frames 0x80000/0x80001; page
        // tables at frames 0x80002..0x80012.
        let directory = |asid: u64| 0x8000_0000 + (asid - 5) * 0x1000;
        let entry = |rng: &mut Xorshift| -> u32 {
            match rng.below(8) {
                0 => 0,
                1 => 0x8000_0000 | 0x80000, // a section over all of RAM
                _ => 0xC000_0000 | ENTRY_NEXT | (0x80002 + rng.below(16) as u32),
            }
        };
        let page = |rng: &mut Xorshift| -> u32 {
            match rng.below(6) {
                0 => 0x80100 + rng.below(0x300) as u32, // no access bits
                _ => 0xC000_0000 | (0x80100 + rng.below(0x300) as u32),
            }
        };
        for asid in [5, 6] {
            for index in 0..16 {
                ram.write(directory(asid) + index * 4, &entry(&mut rng).to_le_bytes())
                    .unwrap();
            }
        }
        for word in 0..16 * 1024 {
            ram.write(0x8000_2000 + word * 4, &page(&mut rng).to_le_bytes())
                .unwrap();
        }
        let read_u32 = |table: u64| {
            let mut word = [0; 4];
            ram.read(table, &mut word)?;
            Some(u32::from_le_bytes(word))
        };
        let device_address = |rng: &mut Xorshift| rng.below(1 << 26) | rng.below(2) << 32;
        let mut checked = 0;
        for _ in 0..4000 {
            match rng.below(5) {
                // Page tables change in memory, as the kernel changes them
                // before it flushes.
                0 | 1 => {
                    let (at, value) = if rng.below(4) == 0 {
                        (
                            directory(5 + rng.below(2)) + rng.below(16) * 4,
                            entry(&mut rng),
                        )
                    } else {
                        (0x8000_2000 + rng.below(16 * 1024) * 4, page(&mut rng))
                    };
                    ram.write(at, &value.to_le_bytes()).unwrap();
                    // Reads in between may see the old tables or the new.
                    let mut scratch = [0; 8];
                    let _ = memory.read(device_address(&mut rng), &mut scratch);
                    continue;
                }
                2 => write(&mc, 0x34, rng.next() as u32), // PTC flush
                3 => {
                    let enable = if rng.below(8) == 0 { 0 } else { ASID_ENABLE };
                    let asids = (5 + rng.below(2)) as u32 | (5 + rng.below(2) as u32) << 8;
                    write(&mc, SMMU_GPU_ASID, enable | asids);
                }
                _ => {
                    write(&mc, SMMU_PTB_ASID, 5 + rng.below(2) as u32);
                    let frame = 0x80000 + rng.below(2) as u32;
                    let access = if rng.below(10) == 0 { 0 } else { 0xE000_0000 };
                    write(&mc, SMMU_PTB_DATA, access | frame);
                }
            }
            for _ in 0..32 {
                let at = device_address(&mut rng);
                let fresh = if mc.registers.get(SMMU_GPU_ASID).unwrap() & ASID_ENABLE != 0 {
                    mc.translate(SMMU_GPU_ASID, at, read_u32)
                } else {
                    Some(at)
                };
                let cached = memory.physical(&memory.view(mc.generation()), at);
                assert_eq!(cached, fresh, "device address {at:#x}");
                let mut through = [0; 8];
                let outcome = memory.read(at, &mut through);
                let in_page = (PAGE_SIZE - at % PAGE_SIZE).min(8) as usize;
                if let Some(physical) = fresh
                    && in_page == 8
                {
                    let mut direct = [0; 8];
                    assert_eq!(outcome, ram.read(physical, &mut direct));
                    assert_eq!(through, direct);
                    checked += 1;
                } else if fresh.is_none() {
                    assert_eq!(outcome, None);
                }
            }
        }
        assert!(checked > 10_000, "{checked} reads compared");
    }

    /// A failed translation is not kept: a page mapped after it is found
    /// at once, flushed or not.
    #[test]
    fn unmapped_pages_are_walked_again() {
        let mc = Arc::new(Mc::new(DramSize::Gb4));
        let ram = Arc::new(GuestRam::new(0x8000_0000, 0x1_0000).unwrap());
        let memory = DeviceMemory::new(&mc, &ram, SMMU_DC_ASID);
        ram.write(
            0x8000_0000,
            &(0xC000_0000 | ENTRY_NEXT | 0x80001).to_le_bytes(),
        )
        .unwrap();
        write(&mc, SMMU_PTB_ASID, 5);
        write(&mc, SMMU_PTB_DATA, 0xE000_0000 | 0x80000);
        write(&mc, SMMU_DC_ASID, ASID_ENABLE | 5);
        let mut back = [0; 4];
        assert_eq!(memory.read(0x3000, &mut back), None);
        ram.write(
            0x8000_1000 + 3 * 4,
            &(0xC000_0000u32 | 0x80002).to_le_bytes(),
        )
        .unwrap();
        ram.write(0x8000_2000, &[1, 2, 3, 4]).unwrap();
        memory.read(0x3000, &mut back).unwrap();
        assert_eq!(back, [1, 2, 3, 4]);
    }

    /// RAM of 4 MiB at 0x8000_0000 with ASID `asid`'s page directory at
    /// `directory` (physical) pointing its first 4 MiB of device address at
    /// the page table at `table`, and `client` translating through it.
    fn address_space(mc: &Mc, ram: &GuestRam, client: u64, asid: u32, directory: u64, table: u64) {
        ram.write(
            directory,
            &(0xC000_0000 | ENTRY_NEXT | (table >> 12) as u32).to_le_bytes(),
        )
        .unwrap();
        write(mc, SMMU_PTB_ASID, asid);
        write(mc, SMMU_PTB_DATA, 0xE000_0000 | (directory >> 12) as u32);
        write(mc, client, ASID_ENABLE | asid);
    }

    /// Accesses over many pages, in one piece where pages lie together
    /// physically, read and write what the pages do one by one: over
    /// random runs of contiguous, scattered, unmapped and carveout-blocked
    /// pages, pages outside RAM among them, and every access ending where
    /// the first unmapped page stops it.
    #[test]
    fn bulk_accesses_are_page_by_page_accesses() {
        let mc = Arc::new(Mc::new(DramSize::Gb4));
        let ram = Arc::new(GuestRam::new(0x8000_0000, 0x40_0000).unwrap());
        let memory = DeviceMemory::new(&mc, &ram, SMMU_GPU_ASID);
        address_space(&mc, &ram, SMMU_GPU_ASID, 5, 0x8000_0000, 0x8000_1000);
        // A carveout over physical pages 0x80380..0x80390.
        mc.set_kernel_carveout(0, 0x8038_0000, 0x1_0000);
        let mut rng = Xorshift(0x1234_5678_9ABC_DEF1);
        let mut next = 0x80100u32;
        let mut frames = Vec::new();
        for page in 0..256u64 {
            let frame = match rng.below(10) {
                0..=4 => next,
                5..=7 => 0x80100 + rng.below(0x2F0) as u32,
                // Past the end of RAM.
                8 if rng.below(3) == 0 => 0x80400 + rng.below(4) as u32,
                _ => 0,
            };
            if frame != 0 {
                ram.write(0x8000_1000 + page * 4, &(0xC000_0000 | frame).to_le_bytes())
                    .unwrap();
                next = frame + 1;
            }
            frames.push(frame);
        }
        // Distinct bytes everywhere.
        let fill: Vec<u8> = (0..0x3E_0000).map(|i: u32| (i ^ i >> 9) as u8).collect();
        ram.write(0x8002_0000, &fill).unwrap();
        let carved = |physical: u64| (0x8038_0000..0x8039_0000).contains(&physical);
        for case in 0..2000 {
            let address = rng.below(255 * 0x1000);
            let len = 1 + rng.below(0x6000).min(256 * 0x1000 - address - 1) as usize;
            // Page by page: where each page goes, until the first unmapped.
            let mut pages = Vec::new();
            let mut done = 0;
            let mut mapped = true;
            while done < len {
                let at = address + done as u64;
                let count = ((PAGE_SIZE - at % PAGE_SIZE) as usize).min(len - done);
                let frame = frames[(at / PAGE_SIZE) as usize];
                if frame == 0 {
                    mapped = false;
                    break;
                }
                pages.push((u64::from(frame) << 12 | at % PAGE_SIZE, done, count));
                done += count;
            }
            if rng.below(2) == 0 {
                let mut through = vec![0; len];
                let outcome = memory.read(address, &mut through);
                let mut expected = vec![0; len];
                let mut all = mapped;
                for &(physical, offset, count) in &pages {
                    let piece = &mut expected[offset..offset + count];
                    if carved(physical) {
                        piece.fill(0);
                    } else {
                        all &= ram.read(physical, piece).is_some();
                    }
                }
                assert_eq!(
                    outcome.is_some(),
                    all,
                    "case {case}: read {address:#x}+{len:#x}"
                );
                if all {
                    assert!(
                        through == expected,
                        "case {case}: read {address:#x}+{len:#x}"
                    );
                }
            } else {
                let data: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
                let before = {
                    let mut bytes = vec![0; 0x40_0000];
                    ram.read(0x8000_0000, &mut bytes).unwrap();
                    bytes
                };
                let outcome = memory.write(address, &data);
                let mut after = vec![0; 0x40_0000];
                ram.read(0x8000_0000, &mut after).unwrap();
                // The same writes page by page, on the copy from before.
                let mut expected = before;
                let mut all = mapped;
                for &(physical, offset, count) in &pages {
                    if carved(physical) {
                        continue;
                    }
                    let Some(at) = physical.checked_sub(0x8000_0000).map(|at| at as usize) else {
                        all = false;
                        break;
                    };
                    if at + count > expected.len() {
                        all = false;
                        break;
                    }
                    expected[at..at + count].copy_from_slice(&data[offset..offset + count]);
                }
                assert_eq!(
                    outcome.is_some(),
                    all,
                    "case {case}: write {address:#x}+{len:#x}"
                );
                assert!(
                    after == expected,
                    "case {case}: write {address:#x}+{len:#x}"
                );
            }
        }
    }

    /// Threads reading through one client while another switches its
    /// address space back and forth get each page whole from one space or
    /// the other, and once the switching stops, from the last one.
    #[test]
    fn concurrent_accesses_see_whole_translations() {
        let mc = Arc::new(Mc::new(DramSize::Gb4));
        let ram = Arc::new(GuestRam::new(0x8000_0000, 0x40_0000).unwrap());
        let memory = DeviceMemory::new(&mc, &ram, SMMU_GPU_ASID);
        // ASID 5 maps 64 device pages to 0x8010_0000 on, ASID 6 to
        // 0x8020_0000 on; their bytes are 0x55 and 0x66.
        for (asid, directory, table, frames) in [
            (5u32, 0x8000_0000u64, 0x8000_1000u64, 0x80100u32),
            (6, 0x8000_2000, 0x8000_3000, 0x80200),
        ] {
            address_space(&mc, &ram, SMMU_GPU_ASID, asid, directory, table);
            for page in 0..64u32 {
                ram.write(
                    table + u64::from(page) * 4,
                    &(0xC000_0000 | (frames + page)).to_le_bytes(),
                )
                .unwrap();
            }
            ram.write(u64::from(frames) << 12, &[asid as u8 * 0x11; 64 * 0x1000])
                .unwrap();
        }
        let stop = std::sync::atomic::AtomicBool::new(false);
        std::thread::scope(|scope| {
            for thread in 0..4u64 {
                let (memory, stop) = (&memory, &stop);
                scope.spawn(move || {
                    let mut page = vec![0; 0x1000];
                    let mut round = 0u64;
                    while !stop.load(Ordering::Relaxed) || round < 1000 {
                        let at = (round * 7 + thread) % 64 * 0x1000;
                        memory.read(at, &mut page).unwrap();
                        assert!(
                            page.iter().all(|&b| b == page[0]) && [0x55, 0x66].contains(&page[0]),
                            "page {at:#x} mixed"
                        );
                        round += 1;
                    }
                });
            }
            for flip in 0..2000u32 {
                write(&mc, SMMU_GPU_ASID, ASID_ENABLE | 5 + flip % 2);
            }
            stop.store(true, Ordering::Relaxed);
        });
        // The last write selected ASID 6.
        let mut page = vec![0; 64 * 0x1000];
        memory.read(0, &mut page).unwrap();
        assert!(page.iter().all(|&b| b == 0x66));
    }
}
