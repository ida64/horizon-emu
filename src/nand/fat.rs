//! FAT volumes laid out the way the console's filesystem process formats
//! its built-in partitions, and a writer for the FAT32 ones.
//!
//! The filesystem process carries a formatter for these partitions; all
//! offsets below are into its image. It chooses cluster size and FAT type
//! from the partition size (`0x118d40`, table at `0x1f1848`), sizes the
//! FATs and the reserved area (`0x1190d0`), and writes the boot sector
//! (`0x119b30`, serialized by `0xf7090`) with, on FAT32, its copy at sector
//! 6, the information sector at 1 and 7, a sector holding only the boot
//! signature at 2 and 8 and a zero sector at 3 and 9 (`0x1197e0`). Its FAT
//! driver then clears the FATs, marks entries 0 and 1 (`0xfd840`, markers at
//! `0x1ed418`), clears the root directory and ends its chain (`0x1137d0`).
//! [`format()`] produces the same volume. What it cannot reproduce is the
//! volume serial number, which the formatter takes from the clock.
//!
//! Two properties matter beyond a valid volume. Partitions whose driver
//! instance runs in "FAT safe" mode (`0xe2020` enables it for the safe
//! mode, user and system partitions) get their FATs rounded up to 32
//! sectors. And a mount of such a volume (`0xe4e00`) journals it. The
//! driver keeps a record in reserved sector 3: the record sector of its
//! context as `0x11c570` sets it, which request 11 of `0x11ecb0`, the only
//! way to change it, is never asked to (so the searches of requests 1 and
//! 4 for another sector never run). The record holds `PRF2SAFE`, the first
//! cluster of the recovery file `PRF2SAFE.RCV` in the root directory and
//! its length, the FAT sectors over 4096 rounded up plus 21, and a CRC-32
//! (requests 1 and 4). Sector 3 without that mark is no record: the mount
//! (`safe_backup_on`, `0x11af10`) writes one there and makes a new recovery
//! file. The formatter writes the whole reserved area, sector 3 zero;
//! booted on volumes from [`format()`], the driver put its record there
//! and the file in the root.
//!
//! The recovery file starts with a header and its copy, whose word at byte
//! 32 is the operation in progress (`0x11cf30`). Every operation writes it
//! before changing the volume (`0x11ce30`), saves the directory sectors it
//! overwrites into the file first (`0x11e200`), copies the FAT sectors it
//! changed from the second FAT to the first (`0x11d1d0`) and writes the
//! word zero again (`0x11d0c0`). A mount that finds it nonzero completes
//! the operation or puts the saved sectors back (`safe_recover`,
//! `0x11aae0`, request 12), over whatever was written since. One that
//! finds it zero only clears the file where it is (request 7, `0x11fb80`).
//!
//! [`Fat32`] adds directories and files to a volume no console has mounted
//! yet, and to one a console has mounted and stopped between operations,
//! whose header says zero: it never writes reserved sectors, takes only
//! free clusters, and writes both FATs alike. It refuses a volume whose
//! console stopped during an operation ([`FatError::RecoveryPending`]) and
//! one whose record or recovery file is damaged or does not match the
//! volume ([`FatError::RecoveryDamaged`]). Names are stored the VFAT way,
//! a long name in front of a generated `BASIS~N.EXT` alias that no entry of
//! the directory has yet, unless a name already is an upper-case 8.3 name.

use std::{
    collections::BTreeMap,
    fmt,
    fs::File,
    io::{self, Read},
    ops::Range,
    os::unix::fs::FileExt,
};

pub const SECTOR_SIZE: usize = 512;

/// Storage addressed in sectors of [`SECTOR_SIZE`] bytes; buffers are whole
/// sectors.
pub trait Sectors {
    fn read(&mut self, sector: u64, data: &mut [u8]) -> io::Result<()>;
    fn write(&mut self, sector: u64, data: &[u8]) -> io::Result<()>;
}

impl<S: Sectors + ?Sized> Sectors for &mut S {
    fn read(&mut self, sector: u64, data: &mut [u8]) -> io::Result<()> {
        (**self).read(sector, data)
    }

    fn write(&mut self, sector: u64, data: &[u8]) -> io::Result<()> {
        (**self).write(sector, data)
    }
}

/// A raw image file.
impl Sectors for File {
    fn read(&mut self, sector: u64, data: &mut [u8]) -> io::Result<()> {
        self.read_exact_at(data, sector * SECTOR_SIZE as u64)
    }

    fn write(&mut self, sector: u64, data: &[u8]) -> io::Result<()> {
        self.write_all_at(data, sector * SECTOR_SIZE as u64)
    }
}

/// An image in memory, whose length is that of the storage.
impl Sectors for Vec<u8> {
    fn read(&mut self, sector: u64, data: &mut [u8]) -> io::Result<()> {
        let range = byte_range(self.len(), sector, data.len())?;
        data.copy_from_slice(&self[range]);
        Ok(())
    }

    fn write(&mut self, sector: u64, data: &[u8]) -> io::Result<()> {
        let range = byte_range(self.len(), sector, data.len())?;
        self[range].copy_from_slice(data);
        Ok(())
    }
}

/// Where `length` bytes from `sector` lie in storage of `size` bytes.
fn byte_range(size: usize, sector: u64, length: usize) -> io::Result<Range<usize>> {
    usize::try_from(sector)
        .ok()
        .and_then(|sector| sector.checked_mul(SECTOR_SIZE))
        .and_then(|start| Some(start..start.checked_add(length)?))
        .filter(|range| range.end <= size)
        .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "beyond the end"))
}

#[derive(Debug)]
pub enum FatError {
    Io(io::Error),
    /// The console's formatter has no parameters for a partition of this
    /// many bytes.
    UnsupportedSize(u64),
    /// The partition does not hold a FAT32 volume of 512-byte sectors.
    NotFat32(&'static str),
    /// The volume contradicts itself.
    Corrupt(&'static str),
    /// A console stopped while its FAT driver was in the middle of operation
    /// `operation` (transaction `transaction`) on the volume. Its next mount
    /// completes or undoes that operation from `PRF2SAFE.RCV`, over
    /// whatever is written in between.
    RecoveryPending {
        transaction: u32,
        operation: u32,
    },
    /// The FAT driver's record or recovery file is damaged or does not
    /// match the volume: why. Depending on which, the console's mount fails
    /// or makes a new recovery file, where this writer may have put files.
    RecoveryDamaged(&'static str),
    /// Not a name this writer stores: empty, longer than 255 characters,
    /// outside printable ASCII, or with a character FAT forbids.
    InvalidName(String),
    NotFound(String),
    NotADirectory(String),
    Exists(String),
    /// A file of 4 GiB or more.
    TooLarge(String),
    /// A directory at the size limit of 65536 entries.
    DirectoryFull(String),
    NoSpace,
    /// New contents for a file in place that are longer than the file.
    LongerThanFile(String),
}

impl fmt::Display for FatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "{error}"),
            Self::UnsupportedSize(size) => {
                write!(
                    f,
                    "the console has no FAT format for a partition of {size:#x} bytes"
                )
            }
            Self::NotFat32(why) => write!(f, "not a FAT32 volume: {why}"),
            Self::Corrupt(what) => write!(f, "corrupt volume: {what}"),
            Self::RecoveryPending {
                transaction,
                operation,
            } => write!(
                f,
                "the console stopped in the middle of FAT operation {operation} \
                 (transaction {transaction}) on this volume, which its next mount \
                 completes or undoes from PRF2SAFE.RCV over anything written \
                 now; boot the image once and stop it while idle, then retry"
            ),
            Self::RecoveryDamaged(why) => {
                write!(f, "the FAT driver's recovery data is unusable: {why}")
            }
            Self::InvalidName(name) => write!(f, "{name:?} is not a storable name"),
            Self::NotFound(path) => write!(f, "{path} does not exist"),
            Self::NotADirectory(path) => write!(f, "{path} is not a directory"),
            Self::Exists(path) => write!(f, "{path} already exists"),
            Self::TooLarge(path) => write!(f, "{path} is too large for FAT32"),
            Self::DirectoryFull(path) => write!(f, "the directory of {path} is full"),
            Self::NoSpace => write!(f, "no space left"),
            Self::LongerThanFile(path) => {
                write!(f, "the new contents are longer than {path}")
            }
        }
    }
}

impl std::error::Error for FatError {}

impl From<io::Error> for FatError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FatType {
    Fat12,
    Fat32,
}

const FAT_COUNT: u32 = 2;
/// Root directory entries of a FAT12 volume.
const ROOT_ENTRIES: u32 = 512;
const ENTRY_SIZE: usize = 32;
/// Reserved sectors before the formatter aligns the data area: the boot
/// sectors it writes, 0 to 8.
const MINIMUM_RESERVED: u32 = 9;
/// Granularity of the FATs of a FAT-safe volume, in sectors.
const SAFE_FAT_ALIGNMENT: u32 = 32;

/// Rows of the formatter's table for built-in partitions: largest partition
/// in bytes, sectors per cluster, alignment of the data area in sectors,
/// FAT type. A row without type is a range the formatter rejects.
const SIZE_CLASSES: [(u64, u32, u32, Option<FatType>); 8] = [
    (0x210_0000, 32, 32, Some(FatType::Fat12)),
    (0x4010_0000, 1, 32, Some(FatType::Fat32)),
    (0x8000_0000, 32, 32, Some(FatType::Fat32)),
    (0x8287_FFFF, 0, 0, None),
    (0x8_0000_0000, 32, 32, Some(FatType::Fat32)),
    (0x8_07FF_FFFF, 0, 0, None),
    (0x10_0000_0000, 32, 32, Some(FatType::Fat32)),
    (0x40_0000_0000, 128, 32, Some(FatType::Fat32)),
];

/// The rows of the filesystem process's formatter for microSD cards
/// (`FS08_SelectPartitionedFatGeometry`, text `0x119c90`, table at
/// `0x1f19e8`) for the one FAT type this writer makes: cards above
/// 0x8287_FFFF bytes and up to 32 GiB, SDHC, get FAT32 with 64 sectors per
/// cluster and a boundary unit of 8192 sectors (4 MiB), which is the area
/// in front of the partition and the alignment of its data area. The table
/// gives smaller cards FAT12 or FAT16, refuses the 40 MiB above 2 GiB and
/// the 128 MiB above 32 GiB, and gives larger cards exFAT, which this
/// build of the filesystem process does not format (result `0x249a02`).
/// These are the rules of the SD Association's file system specification.
const SD_CARD_FAT32: (std::ops::RangeInclusive<u64>, u32, u32) =
    (0x8288_0000..=0x8_0000_0000, 64, 8192);

/// The disk geometry the formatter declares (table at `0x1f1908`): largest
/// partition in bytes, heads, sectors per track. The ranges
/// [`SIZE_CLASSES`] rejects have no row here.
const GEOMETRIES: [(u64, u8, u8); 12] = [
    (0x20_0000, 2, 16),
    (0x100_0000, 2, 32),
    (0x200_0000, 4, 32),
    (0x800_0000, 8, 32),
    (0x1000_0000, 16, 32),
    (0x1F80_0000, 16, 63),
    (0x3F00_0000, 32, 63),
    (0x7E00_0000, 64, 63),
    (0x8000_0000, 128, 63),
    (0xFC00_0000, 128, 63),
    (0x8_0000_0000, 255, 63),
    (0x8000_0000_0000, 255, 63),
];

/// Where the parts of a volume lie, in sectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout {
    pub fat_type: FatType,
    pub total_sectors: u32,
    pub sectors_per_cluster: u32,
    pub reserved_sectors: u32,
    /// Sectors of one of the two FATs.
    pub fat_sectors: u32,
    /// Sectors in front of the volume on its medium: the partition table's
    /// area on a microSD card, none on a built-in partition.
    pub hidden_sectors: u32,
    heads: u8,
    sectors_per_track: u8,
}

impl Layout {
    /// What the formatter computes for a partition of `sectors` sectors.
    pub fn for_partition(sectors: u64, fat_safe: bool) -> Result<Self, FatError> {
        let size = sectors * SECTOR_SIZE as u64;
        let unsupported = || FatError::UnsupportedSize(size);
        let &(_, sectors_per_cluster, alignment, fat_type) = SIZE_CLASSES
            .iter()
            .find(|class| size <= class.0)
            .ok_or_else(unsupported)?;
        let fat_type = fat_type.ok_or_else(unsupported)?;
        let &(_, heads, sectors_per_track) = geometry(size).ok_or_else(unsupported)?;
        let total = u32::try_from(sectors).map_err(|_| unsupported())?;

        let (entry_bits, root_sectors) = match fat_type {
            FatType::Fat12 => (12, ROOT_ENTRIES * ENTRY_SIZE as u32 / SECTOR_SIZE as u32),
            FatType::Fat32 => (32, 0),
        };
        let fat_alignment = if fat_safe { SAFE_FAT_ALIGNMENT } else { 1 };
        // Sectors of a FAT with entries for the clusters of `data` sectors
        // and the two entries before them.
        let fat_for = |data: u32| {
            ((data / sectors_per_cluster + 2) * entry_bits)
                .div_ceil(8 * SECTOR_SIZE as u32)
                .next_multiple_of(fat_alignment)
        };
        // The formatter starts from a FAT for nearly the whole partition
        // and shrinks it for as long as a smaller one still covers what the
        // FATs themselves leave over.
        let (reserved_sectors, fat_sectors) = match fat_type {
            FatType::Fat12 => {
                let reserved = 1;
                let available = total
                    .checked_sub(reserved + root_sectors)
                    .ok_or_else(unsupported)?;
                let mut fat = fat_for(available);
                while fat != 0 && fat_for(available.saturating_sub(fat * FAT_COUNT)) < fat {
                    fat -= fat_alignment;
                }
                (reserved, fat)
            }
            FatType::Fat32 => fat32_areas(total, sectors_per_cluster, alignment, fat_alignment)
                .ok_or_else(unsupported)?,
        };
        if fat_sectors == 0 {
            return Err(unsupported());
        }
        Ok(Self {
            fat_type,
            total_sectors: total,
            sectors_per_cluster,
            reserved_sectors,
            fat_sectors,
            hidden_sectors: 0,
            heads,
            sectors_per_track,
        })
    }

    /// What the formatter computes for the partition of a microSD card of
    /// `card_sectors` sectors (`SD_CARD_FAT32`): the volume starts one
    /// boundary unit into the card and its data area on a multiple of one.
    pub fn for_sd_card(card_sectors: u64) -> Result<Self, FatError> {
        let size = card_sectors * SECTOR_SIZE as u64;
        let unsupported = || FatError::UnsupportedSize(size);
        let (sizes, sectors_per_cluster, boundary_unit) = SD_CARD_FAT32;
        if !sizes.contains(&size) {
            return Err(unsupported());
        }
        let &(_, heads, sectors_per_track) = geometry(size).ok_or_else(unsupported)?;
        let total =
            u32::try_from(card_sectors - u64::from(boundary_unit)).map_err(|_| unsupported())?;
        let (reserved_sectors, fat_sectors) =
            fat32_areas(total, sectors_per_cluster, boundary_unit, 1).ok_or_else(unsupported)?;
        Ok(Self {
            fat_type: FatType::Fat32,
            total_sectors: total,
            sectors_per_cluster,
            reserved_sectors,
            fat_sectors,
            hidden_sectors: boundary_unit,
            heads,
            sectors_per_track,
        })
    }

    /// Heads and sectors per track the volume declares.
    pub fn geometry(&self) -> (u8, u8) {
        (self.heads, self.sectors_per_track)
    }

    fn root_sectors(&self) -> u32 {
        match self.fat_type {
            FatType::Fat12 => ROOT_ENTRIES * ENTRY_SIZE as u32 / SECTOR_SIZE as u32,
            FatType::Fat32 => 0,
        }
    }

    /// First sector of cluster 2.
    pub fn data_start(&self) -> u32 {
        self.reserved_sectors + self.fat_sectors * FAT_COUNT + self.root_sectors()
    }

    pub fn cluster_count(&self) -> u32 {
        (self.total_sectors - self.data_start()) / self.sectors_per_cluster
    }
}

/// The geometry row for a volume of `size` bytes.
fn geometry(size: u64) -> Option<&'static (u64, u8, u8)> {
    GEOMETRIES.iter().find(|geometry| size <= geometry.0)
}

/// Reserved sectors and sectors per FAT of a FAT32 volume of `total`
/// sectors whose data area starts on a multiple of `alignment` sectors.
/// The formatter starts from a FAT for nearly the whole volume and shrinks
/// it for as long as a smaller one still covers what the FATs themselves
/// leave over; the reserved area takes up the slack the alignment leaves.
/// `None` if no FAT fits.
fn fat32_areas(
    total: u32,
    sectors_per_cluster: u32,
    alignment: u32,
    fat_alignment: u32,
) -> Option<(u32, u32)> {
    // Sectors of a FAT with entries for the clusters of `data` sectors
    // and the two entries before them.
    let fat_for = |data: u32| {
        ((data / sectors_per_cluster + 2) * 32)
            .div_ceil(8 * SECTOR_SIZE as u32)
            .next_multiple_of(fat_alignment)
    };
    let mut reserved = MINIMUM_RESERVED;
    let mut fat = fat_for(total.checked_sub(reserved)?);
    while fat != 0 {
        reserved = (reserved + fat * FAT_COUNT).next_multiple_of(alignment) - fat * FAT_COUNT;
        let needed = fat_for(total.checked_sub(reserved + fat * FAT_COUNT)?);
        if needed == fat {
            break;
        }
        if needed < fat {
            fat -= fat_alignment;
        } else {
            reserved += alignment;
        }
    }
    (fat != 0).then_some((reserved, fat))
}

const BOOT_SIGNATURE: [u8; 2] = [0x55, 0xAA];
const MEDIA_FIXED: u8 = 0xF8;
/// Root directory cluster of a freshly formatted FAT32 volume.
const ROOT_CLUSTER: u32 = 2;
const INFORMATION_SECTOR: u16 = 1;
const BACKUP_BOOT_SECTOR: u16 = 6;
const END_OF_CHAIN: u32 = 0x0FFF_FFFF;
const ENTRY_MASK: u32 = 0x0FFF_FFFF;

fn put16(sector: &mut [u8], offset: usize, value: u16) {
    sector[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn put32(sector: &mut [u8], offset: usize, value: u32) {
    sector[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn get16(data: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([data[offset], data[offset + 1]])
}

fn get32(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ])
}

/// The boot sector as the formatter fills it in: no OEM name, the label
/// `NO NAME`, and a jump that leads nowhere.
fn boot_sector(layout: &Layout, serial: u32) -> [u8; SECTOR_SIZE] {
    let mut sector = [0; SECTOR_SIZE];
    sector[..3].copy_from_slice(&[0xEB, 0xE9, 0x90]);
    put16(&mut sector, 11, SECTOR_SIZE as u16);
    sector[13] = layout.sectors_per_cluster as u8;
    put16(&mut sector, 14, layout.reserved_sectors as u16);
    sector[16] = FAT_COUNT as u8;
    sector[21] = MEDIA_FIXED;
    put16(&mut sector, 24, layout.sectors_per_track.into());
    put16(&mut sector, 26, layout.heads.into());
    put32(&mut sector, 28, layout.hidden_sectors);
    // What follows the parameter block: drive number, extended signature,
    // serial, label and type name.
    let (extension, type_name) = match layout.fat_type {
        FatType::Fat12 => {
            put16(&mut sector, 17, ROOT_ENTRIES as u16);
            match u16::try_from(layout.total_sectors) {
                Ok(total) => put16(&mut sector, 19, total),
                Err(_) => put32(&mut sector, 32, layout.total_sectors),
            }
            put16(&mut sector, 22, layout.fat_sectors as u16);
            (36, b"FAT12   ")
        }
        FatType::Fat32 => {
            put32(&mut sector, 32, layout.total_sectors);
            put32(&mut sector, 36, layout.fat_sectors);
            put32(&mut sector, 44, ROOT_CLUSTER);
            put16(&mut sector, 48, INFORMATION_SECTOR);
            put16(&mut sector, 50, BACKUP_BOOT_SECTOR);
            (64, b"FAT32   ")
        }
    };
    sector[extension] = 0x80;
    sector[extension + 2] = 0x29;
    put32(&mut sector, extension + 3, serial);
    sector[extension + 7..extension + 18].copy_from_slice(b"NO NAME    ");
    sector[extension + 18..extension + 26].copy_from_slice(type_name);
    sector[510..].copy_from_slice(&BOOT_SIGNATURE);
    sector
}

/// The FAT32 information sector, with free space and allocation hint both
/// "unknown", which is how the formatter leaves them.
fn information_sector() -> [u8; SECTOR_SIZE] {
    let mut sector = [0; SECTOR_SIZE];
    sector[..4].copy_from_slice(b"RRaA");
    sector[484..488].copy_from_slice(b"rrAa");
    put32(&mut sector, 488, u32::MAX);
    put32(&mut sector, 492, u32::MAX);
    sector[510..].copy_from_slice(&BOOT_SIGNATURE);
    sector
}

/// Makes the `sectors` sectors of `storage` an empty volume. `serial` is
/// the volume serial number, for which the console would use its clock.
pub fn format(
    storage: &mut impl Sectors,
    sectors: u64,
    fat_safe: bool,
    serial: u32,
) -> Result<Layout, FatError> {
    let layout = Layout::for_partition(sectors, fat_safe)?;
    format_volume(storage, &layout, serial)?;
    Ok(layout)
}

/// Makes `storage`, from its first sector, the empty volume `layout`
/// describes.
pub fn format_volume(
    storage: &mut impl Sectors,
    layout: &Layout,
    serial: u32,
) -> Result<(), FatError> {
    let sector_bytes = |sectors: u32| sectors as usize * SECTOR_SIZE;

    let mut reserved = vec![0; sector_bytes(layout.reserved_sectors)];
    let mut put = |sector: u16, contents: &[u8; SECTOR_SIZE]| {
        reserved[sector_bytes(sector.into())..][..SECTOR_SIZE].copy_from_slice(contents);
    };
    let boot = boot_sector(layout, serial);
    put(0, &boot);
    let mut fat = vec![0; sector_bytes(layout.fat_sectors)];
    let root_sectors = match layout.fat_type {
        FatType::Fat12 => {
            fat[..3].copy_from_slice(&[MEDIA_FIXED, 0xFF, 0xFF]);
            layout.root_sectors()
        }
        FatType::Fat32 => {
            let information = information_sector();
            let mut signature = [0; SECTOR_SIZE];
            signature[510..].copy_from_slice(&BOOT_SIGNATURE);
            put(BACKUP_BOOT_SECTOR, &boot);
            for copy in [0, BACKUP_BOOT_SECTOR] {
                put(copy + INFORMATION_SECTOR, &information);
                put(copy + INFORMATION_SECTOR + 1, &signature);
            }
            put32(&mut fat, 0, 0x0FFF_FF00 | u32::from(MEDIA_FIXED));
            put32(&mut fat, 4, END_OF_CHAIN);
            put32(&mut fat, 4 * ROOT_CLUSTER as usize, END_OF_CHAIN);
            layout.sectors_per_cluster
        }
    };
    storage.write(0, &reserved)?;
    for copy in 0..FAT_COUNT {
        let sector = layout.reserved_sectors + copy * layout.fat_sectors;
        storage.write(sector.into(), &fat)?;
    }
    // On FAT12 the root directory precedes the clusters; on FAT32 it is the
    // first of them.
    let root = layout.reserved_sectors + layout.fat_sectors * FAT_COUNT;
    storage.write(root.into(), &vec![0; sector_bytes(root_sectors)])?;
    Ok(())
}

const ATTRIBUTE_VOLUME: u8 = 0x08;
const ATTRIBUTE_DIRECTORY: u8 = 0x10;
const ATTRIBUTE_ARCHIVE: u8 = 0x20;
const ATTRIBUTE_LONG_NAME: u8 = 0x0F;
const DELETED: u8 = 0xE5;
const LAST_LONG_ENTRY: u8 = 0x40;
/// UTF-16 units of a long name one directory entry holds.
const LONG_UNITS: usize = 13;
const LONG_UNIT_OFFSETS: [usize; LONG_UNITS] = [1, 3, 5, 7, 9, 14, 16, 18, 20, 22, 24, 28, 30];
const MAXIMUM_NAME: usize = 255;
const MAXIMUM_DIRECTORY: usize = 65536 * ENTRY_SIZE;
/// Date of every entry this writer makes, 1 January 1980: an image is the
/// same whenever it is built.
const DATE: u16 = 1 << 5 | 1;
/// The FAT-safe driver's recovery file, in the root directory, as its
/// short entry names it.
const RECOVERY_FILE: ShortName = *b"PRF2SAFERCV";
/// Mark of the driver's record, of the headers of the recovery file and
/// of the entries in it (`0x120190`).
const SAFE_MAGIC: &[u8; 8] = b"PRF2SAFE";
/// Reserved sector of the driver's record: the driver context's record
/// sector as `0x11c570` sets it, which nothing changes.
const RECORD_SECTOR: u32 = 3;
/// Sectors of the recovery file besides its bitmap of FAT sectors, one bit
/// per FAT sector: the header and its copy, 18 of saved directory sectors
/// and one after the bitmap.
const RECOVERY_FIXED_SECTORS: u32 = 21;
/// Bytes of file data handed to the storage at once.
const WRITE_CHUNK: usize = 1 << 20;

type ShortName = [u8; 11];

fn is_short_character(character: u8) -> bool {
    character.is_ascii_uppercase()
        || character.is_ascii_digit()
        || b"!#$%&'()-@^_`{}~".contains(&character)
}

/// `name` as the 11 bytes of a directory entry, if it is an upper-case 8.3
/// name.
fn pack_short(name: &str) -> Option<ShortName> {
    let (base, extension) = name.rsplit_once('.').unwrap_or((name, ""));
    let fits = (1..=8).contains(&base.len())
        && extension.len() <= 3
        && (extension.is_empty() != name.contains('.'))
        && base
            .bytes()
            .chain(extension.bytes())
            .all(is_short_character);
    fits.then(|| {
        let mut short = [b' '; 11];
        short[..base.len()].copy_from_slice(base.as_bytes());
        short[8..8 + extension.len()].copy_from_slice(extension.as_bytes());
        short
    })
}

/// The 8.3 alias of a long name: what is left of its first six characters
/// and of its extension, and the first numeric tail that is `free`.
fn alias(name: &str, free: impl Fn(&ShortName) -> bool) -> ShortName {
    let (base, extension) = match name.rfind('.') {
        Some(dot) if dot > 0 => (&name[..dot], &name[dot + 1..]),
        _ => (name, ""),
    };
    let squeeze = |part: &str, length: usize| {
        let mut kept = [0; 8];
        let characters = part
            .bytes()
            .filter(|character| !b" .".contains(character))
            .map(|character| character.to_ascii_uppercase())
            .map(|character| {
                if is_short_character(character) {
                    character
                } else {
                    b'_'
                }
            });
        let mut count = 0;
        for (slot, character) in kept[..length].iter_mut().zip(characters) {
            *slot = character;
            count += 1;
        }
        (kept, count)
    };
    let (base, base_length) = squeeze(base, 6);
    let (extension, extension_length) = squeeze(extension, 3);
    (1..=MAXIMUM_DIRECTORY / ENTRY_SIZE)
        .find_map(|number| {
            let tail = format!("~{number}");
            let kept = base_length.min(8 - tail.len());
            let mut short = [b' '; 11];
            short[..kept].copy_from_slice(&base[..kept]);
            short[kept..kept + tail.len()].copy_from_slice(tail.as_bytes());
            short[8..8 + extension_length].copy_from_slice(&extension[..extension_length]);
            free(&short).then_some(short)
        })
        .expect("a directory has fewer entries than there are numeric tails")
}

/// Checksum that ties the entries of a long name to their short entry.
fn short_checksum(short: &ShortName) -> u8 {
    short.iter().fold(0, |sum: u8, &character| {
        sum.rotate_right(1).wrapping_add(character)
    })
}

/// Whether `sector` is a record or recovery-file header of the FAT-safe
/// driver: its mark, a version the driver reads (byte 17 below 0x66 or
/// byte 16 100; it writes 100 and 101), and at byte 508 the CRC-32 of the
/// sector with those four bytes zero (`0x120220`).
fn safe_sector_valid(sector: &[u8]) -> bool {
    let mut unsummed = [0; SECTOR_SIZE];
    unsummed.copy_from_slice(sector);
    unsummed[508..].fill(0);
    sector[..SAFE_MAGIC.len()] == *SAFE_MAGIC
        && (sector[17] < 0x66 || sector[16] == 100)
        && super::crc32(&unsummed) == get32(sector, 508)
}

fn validate_name(name: &str) -> Result<(), FatError> {
    let valid = !name.is_empty()
        && name.len() <= MAXIMUM_NAME
        && name.bytes().all(|character| {
            (0x20..0x7F).contains(&character) && !b"\"*/:<>?\\|".contains(&character)
        })
        && !name.starts_with(' ')
        && !name.ends_with([' ', '.']);
    if valid {
        Ok(())
    } else {
        Err(FatError::InvalidName(name.into()))
    }
}

fn short_record(short: &ShortName, attributes: u8, cluster: u32, size: u32) -> [u8; ENTRY_SIZE] {
    let mut record = [0; ENTRY_SIZE];
    record[..11].copy_from_slice(short);
    record[11] = attributes;
    for date in [16, 18, 24] {
        put16(&mut record, date, DATE);
    }
    put16(&mut record, 20, (cluster >> 16) as u16);
    put16(&mut record, 26, cluster as u16);
    put32(&mut record, 28, size);
    record
}

/// The directory entries of `name`: its long name, last part first, in
/// front of the short entry, or the short entry alone when `long` is not
/// needed.
fn records(name: &str, short: &ShortName, long: bool, tail: [u8; ENTRY_SIZE]) -> Vec<u8> {
    let mut records = Vec::new();
    if long {
        let mut units: Vec<u16> = name.encode_utf16().collect();
        if !units.len().is_multiple_of(LONG_UNITS) {
            units.push(0);
            units.resize(units.len().next_multiple_of(LONG_UNITS), 0xFFFF);
        }
        let checksum = short_checksum(short);
        let parts = units.as_chunks::<LONG_UNITS>().0;
        for (index, part) in parts.iter().enumerate().rev() {
            let mut record = [0; ENTRY_SIZE];
            record[0] = index as u8 + 1;
            if index + 1 == parts.len() {
                record[0] |= LAST_LONG_ENTRY;
            }
            record[11] = ATTRIBUTE_LONG_NAME;
            record[13] = checksum;
            for (&unit, offset) in part.iter().zip(LONG_UNIT_OFFSETS) {
                put16(&mut record, offset, unit);
            }
            records.extend_from_slice(&record);
        }
    }
    records.extend_from_slice(&tail);
    records
}

struct Entry {
    long_name: Option<String>,
    short_name: ShortName,
    directory: bool,
    cluster: u32,
    size: u32,
}

impl Entry {
    fn name(&self) -> String {
        self.long_name.clone().unwrap_or_else(|| {
            let part = |bytes: &[u8]| String::from_utf8_lossy(bytes).trim_end().to_string();
            let (base, extension) = (part(&self.short_name[..8]), part(&self.short_name[8..]));
            if extension.is_empty() {
                base
            } else {
                format!("{base}.{extension}")
            }
        })
    }
}

struct Directory {
    clusters: Vec<u32>,
    /// Contents of `clusters`.
    data: Vec<u8>,
    /// Offset of the first entry that was never used, where the directory
    /// ends.
    end: usize,
    entries: Vec<Entry>,
    dirty: bool,
}

/// The entries of a directory's `data` and where they end.
fn parse_directory(data: &[u8]) -> (Vec<Entry>, usize) {
    let mut entries = Vec::new();
    // Parts of a long name seen since the last short entry, in disk order,
    // with the checksum each claims.
    let mut long: Vec<(u8, [u16; LONG_UNITS], u8)> = Vec::new();
    let (records, _) = data.as_chunks::<ENTRY_SIZE>();
    for (index, record) in records.iter().enumerate() {
        let attributes = record[11];
        if record[0] == 0 {
            return (entries, index * ENTRY_SIZE);
        } else if record[0] == DELETED {
            long.clear();
        } else if attributes & 0x3F == ATTRIBUTE_LONG_NAME {
            long.push((
                record[0],
                LONG_UNIT_OFFSETS.map(|offset| get16(record, offset)),
                record[13],
            ));
        } else {
            let short_name: ShortName = record[..11].try_into().expect("11 of 32 bytes");
            let checksum = short_checksum(&short_name);
            // A long name counts when its parts are numbered down to one
            // from a marked last part and all carry this entry's checksum.
            let complete = long
                .first()
                .is_some_and(|part| part.0 & LAST_LONG_ENTRY != 0)
                && long.iter().rev().enumerate().all(|(index, part)| {
                    usize::from(part.0 & !LAST_LONG_ENTRY) == index + 1 && part.2 == checksum
                });
            let long_name = complete
                .then(|| {
                    let units: Vec<u16> = long
                        .iter()
                        .rev()
                        .flat_map(|part| part.1)
                        .take_while(|&unit| unit != 0)
                        .collect();
                    String::from_utf16(&units).ok()
                })
                .flatten();
            long.clear();
            if attributes & ATTRIBUTE_VOLUME == 0 && short_name[0] != b'.' {
                entries.push(Entry {
                    long_name,
                    short_name,
                    directory: attributes & ATTRIBUTE_DIRECTORY != 0,
                    cluster: u32::from(get16(record, 20)) << 16 | u32::from(get16(record, 26)),
                    size: get32(record, 28),
                });
            }
        }
    }
    (entries, records.len() * ENTRY_SIZE)
}

/// A FAT32 volume being added to. Nothing but file contents reaches the
/// storage before [`Fat32::finish`], which writes directories and FATs:
/// never a reserved sector, where the FAT-safe driver keeps its record and
/// the information sector it leaves at "unknown", and every FAT copy
/// alike, the first being the one the driver goes by.
pub struct Fat32<S> {
    storage: S,
    sectors_per_cluster: u32,
    fat_start: u32,
    fat_sectors: u32,
    fat_count: u32,
    data_start: u32,
    /// Clusters are numbered from 2 to this, exclusive.
    cluster_end: u32,
    root: u32,
    /// The first FAT, of which the others are copies.
    fat: Vec<u8>,
    fat_changed: bool,
    free_clusters: u32,
    /// Where the search for a free cluster resumes.
    next: u32,
    /// Directories read or made so far, by first cluster.
    directories: BTreeMap<u32, Directory>,
}

impl<S: Sectors> Fat32<S> {
    /// The volume on `storage`, which no console has mounted or one has
    /// mounted and left between two operations (see the module
    /// documentation).
    pub fn open(storage: S) -> Result<Self, FatError> {
        let mut volume = Self::read_volume(storage)?;
        volume.check_recovery()?;
        Ok(volume)
    }

    /// That the FAT-safe driver's next mount neither undoes nor completes
    /// anything over what is written now, and keeps its recovery file in
    /// clusters the FAT gives to it: the checks of `safe_check`
    /// (`0x11a610`) and `safe_backup_on` (`0x11af10`). Stricter than the
    /// driver where it would go on regardless: two damaged headers, or a
    /// record without its file, which it would replace.
    fn check_recovery(&mut self) -> Result<(), FatError> {
        let damaged = FatError::RecoveryDamaged;
        if self.fat_start <= RECORD_SECTOR {
            return Ok(());
        }
        let mut record = [0; SECTOR_SIZE];
        self.storage.read(RECORD_SECTOR.into(), &mut record)?;
        // Without its mark the sector is no record: the next mount makes a
        // new one and a new recovery file, whatever the root holds.
        if record[..SAFE_MAGIC.len()] != *SAFE_MAGIC {
            return Ok(());
        }
        // Request 1 of `0x11ecb0`.
        let (first, sectors) = (get32(&record, 32), get32(&record, 36));
        let bitmap = self.fat_sectors.div_ceil(8 * SECTOR_SIZE as u32);
        if !safe_sector_valid(&record)
            || !(2..self.cluster_end - 2).contains(&first)
            || sectors != bitmap + RECOVERY_FIXED_SECTORS
        {
            return Err(damaged("its record in reserved sector 3 is invalid"));
        }
        // Request 3: the file the record names, in clusters of its own.
        self.load(self.root)?;
        let file = self.directories[&self.root]
            .entries
            .iter()
            .find(|entry| entry.short_name == RECOVERY_FILE && !entry.directory);
        if !file
            .is_some_and(|file| file.cluster == first && file.size == sectors * SECTOR_SIZE as u32)
        {
            return Err(damaged("PRF2SAFE.RCV is not where its record says"));
        }
        let clusters = self.chain(first)?;
        let contiguous = clusters.windows(2).all(|pair| pair[1] == pair[0] + 1);
        if !contiguous || (clusters.len() as u32) * self.sectors_per_cluster < sectors {
            return Err(damaged("PRF2SAFE.RCV is not in one run of clusters"));
        }
        // Request 2: the header, or its copy when the header is damaged.
        let mut headers = [0; 2 * SECTOR_SIZE];
        self.storage
            .read(self.cluster_sector(first), &mut headers)?;
        let header = headers
            .chunks(SECTOR_SIZE)
            .find(|header| safe_sector_valid(header))
            .ok_or(damaged("both headers of PRF2SAFE.RCV are damaged"))?;
        match get32(header, 32) {
            0 => Ok(()),
            operation => Err(FatError::RecoveryPending {
                transaction: get32(header, 22),
                operation,
            }),
        }
    }

    /// The volume's parameters and FAT, whoever mounted it before.
    fn read_volume(mut storage: S) -> Result<Self, FatError> {
        let mut boot = [0; SECTOR_SIZE];
        storage.read(0, &mut boot)?;
        let not = FatError::NotFat32;
        if boot[510..] != BOOT_SIGNATURE {
            return Err(not("no boot signature"));
        }
        if usize::from(get16(&boot, 11)) != SECTOR_SIZE {
            return Err(not("sectors are not 512 bytes"));
        }
        let sectors_per_cluster = u32::from(boot[13]);
        let (reserved, fat_count) = (u32::from(get16(&boot, 14)), u32::from(boot[16]));
        let (total, fat_sectors, root) = (get32(&boot, 32), get32(&boot, 36), get32(&boot, 44));
        if get16(&boot, 17) != 0 || get16(&boot, 22) != 0 {
            return Err(not("it has a FAT12 or FAT16 root directory"));
        }
        if !sectors_per_cluster.is_power_of_two() || reserved == 0 || fat_count == 0 {
            return Err(not("impossible parameters"));
        }
        let data_start = fat_sectors
            .checked_mul(fat_count)
            .and_then(|fats| fats.checked_add(reserved))
            .filter(|&start| start < total)
            .ok_or(not("the FATs do not fit"))?;
        let clusters = (total - data_start) / sectors_per_cluster;
        if clusters < 65525 {
            return Err(not("too few clusters"));
        }
        let cluster_end = clusters + 2;
        let fat_bytes = fat_sectors as usize * SECTOR_SIZE;
        if fat_bytes / 4 < cluster_end as usize {
            return Err(not("the FAT is shorter than the clusters"));
        }
        let mut fat = vec![0; fat_bytes];
        storage.read(reserved.into(), &mut fat)?;
        let mut volume = Self {
            storage,
            sectors_per_cluster,
            fat_start: reserved,
            fat_sectors,
            fat_count,
            data_start,
            cluster_end,
            root,
            fat,
            fat_changed: false,
            free_clusters: 0,
            next: 2,
            directories: BTreeMap::new(),
        };
        volume.free_clusters = (2..cluster_end)
            .filter(|&cluster| volume.entry(cluster) == 0)
            .count() as u32;
        Ok(volume)
    }

    fn entry(&self, cluster: u32) -> u32 {
        get32(&self.fat, cluster as usize * 4) & ENTRY_MASK
    }

    /// The four bits above an entry are not part of it and stay.
    fn set_entry(&mut self, cluster: u32, value: u32) {
        let offset = cluster as usize * 4;
        let kept = get32(&self.fat, offset) & !ENTRY_MASK;
        put32(&mut self.fat, offset, kept | value);
        self.fat_changed = true;
    }

    fn cluster_bytes(&self) -> usize {
        self.sectors_per_cluster as usize * SECTOR_SIZE
    }

    fn cluster_sector(&self, cluster: u32) -> u64 {
        u64::from(self.data_start) + u64::from(cluster - 2) * u64::from(self.sectors_per_cluster)
    }

    fn chain(&self, first: u32) -> Result<Vec<u32>, FatError> {
        let mut clusters = Vec::new();
        let mut cluster = first;
        while cluster < 0x0FFF_FFF8 {
            if !(2..self.cluster_end).contains(&cluster) {
                return Err(FatError::Corrupt("a chain leaves the volume"));
            }
            if clusters.len() as u32 >= self.cluster_end {
                return Err(FatError::Corrupt("a chain loops"));
            }
            clusters.push(cluster);
            cluster = self.entry(cluster);
        }
        Ok(clusters)
    }

    /// Takes at least one and up to `most` free clusters that follow one
    /// another, chains them, and appends them to the chain ending in
    /// `previous`.
    fn allocate(&mut self, previous: Option<u32>, most: u32) -> Result<Range<u32>, FatError> {
        let first = (self.next..self.cluster_end)
            .chain(2..self.next)
            .find(|&cluster| self.entry(cluster) == 0)
            .ok_or(FatError::NoSpace)?;
        let limit = self.cluster_end.min(first.saturating_add(most));
        let end = (first + 1..limit)
            .find(|&cluster| self.entry(cluster) != 0)
            .unwrap_or(limit);
        for cluster in first..end - 1 {
            self.set_entry(cluster, cluster + 1);
        }
        self.set_entry(end - 1, END_OF_CHAIN);
        if let Some(previous) = previous {
            self.set_entry(previous, first);
        }
        self.free_clusters -= end - first;
        self.next = end;
        Ok(first..end)
    }

    /// Frees the chain from `first`, which nothing refers to.
    fn release(&mut self, first: u32) -> Result<(), FatError> {
        for cluster in self.chain(first)? {
            self.set_entry(cluster, 0);
            self.free_clusters += 1;
        }
        self.next = self.next.min(first);
        Ok(())
    }

    /// Reads the directory at `first` unless it is at hand already.
    fn load(&mut self, first: u32) -> Result<(), FatError> {
        if self.directories.contains_key(&first) {
            return Ok(());
        }
        let clusters = self.chain(first)?;
        if clusters.is_empty() {
            return Err(FatError::Corrupt("a directory without clusters"));
        }
        let cluster_bytes = self.cluster_bytes();
        let mut data = vec![0; clusters.len() * cluster_bytes];
        for (&cluster, contents) in clusters.iter().zip(data.chunks_mut(cluster_bytes)) {
            self.storage.read(self.cluster_sector(cluster), contents)?;
        }
        let (entries, end) = parse_directory(&data);
        let directory = Directory {
            clusters,
            data,
            end,
            entries,
            dirty: false,
        };
        self.directories.insert(first, directory);
        Ok(())
    }

    /// Whether `name` in the directory at `directory` is a directory, and
    /// its first cluster.
    fn find(&mut self, directory: u32, name: &str) -> Result<Option<(bool, u32)>, FatError> {
        self.load(directory)?;
        let short = pack_short(&name.to_ascii_uppercase());
        let entries = &self.directories[&directory].entries;
        let entry = entries.iter().find(|entry| {
            let long = entry.long_name.as_deref();
            long.is_some_and(|long| long.eq_ignore_ascii_case(name))
                || short == Some(entry.short_name)
        });
        Ok(entry.map(|entry| (entry.directory, entry.cluster)))
    }

    /// Adds `name` to the directory at `parent`. `path` names the entry in
    /// errors.
    fn add_entry(
        &mut self,
        parent: u32,
        name: &str,
        path: &str,
        attributes: u8,
        cluster: u32,
        size: u32,
    ) -> Result<(), FatError> {
        validate_name(name)?;
        if self.find(parent, name)?.is_some() {
            return Err(FatError::Exists(path.into()));
        }
        let mut directory = self.directories.remove(&parent).expect("find loaded it");
        let exact = pack_short(name);
        let short_name = exact.unwrap_or_else(|| {
            alias(name, |short| {
                directory
                    .entries
                    .iter()
                    .all(|entry| entry.short_name != *short)
            })
        });
        let tail = short_record(&short_name, attributes, cluster, size);
        let records = records(name, &short_name, exact.is_none(), tail);
        let grown = self.grow(&mut directory, records.len(), path);
        if grown.is_ok() {
            directory.data[directory.end..][..records.len()].copy_from_slice(&records);
            directory.end += records.len();
            directory.dirty = true;
            directory.entries.push(Entry {
                long_name: exact.is_none().then(|| name.into()),
                short_name,
                directory: attributes & ATTRIBUTE_DIRECTORY != 0,
                cluster,
                size,
            });
        }
        self.directories.insert(parent, directory);
        grown
    }

    /// Makes room for `length` more bytes of entries.
    fn grow(
        &mut self,
        directory: &mut Directory,
        length: usize,
        path: &str,
    ) -> Result<(), FatError> {
        let needed = directory.end + length;
        if needed > MAXIMUM_DIRECTORY {
            return Err(FatError::DirectoryFull(path.into()));
        }
        while needed > directory.data.len() {
            let last = *directory
                .clusters
                .last()
                .expect("a directory has a cluster");
            let cluster = self.allocate(Some(last), 1)?.start;
            directory.clusters.push(cluster);
            directory
                .data
                .resize(directory.data.len() + self.cluster_bytes(), 0);
            directory.dirty = true;
        }
        Ok(())
    }

    /// Follows the directories of `path`, making those that are missing if
    /// it is to `create`, and returns the first cluster of the last.
    fn walk<'a>(
        &mut self,
        path: &str,
        components: impl Iterator<Item = &'a str>,
        create: bool,
    ) -> Result<u32, FatError> {
        let mut directory = self.root;
        for name in components {
            directory = match self.find(directory, name)? {
                Some((true, cluster)) => cluster,
                Some((false, _)) => return Err(FatError::NotADirectory(path.into())),
                None if create => self.make_directory(directory, name, path)?,
                None => return Err(FatError::NotFound(path.into())),
            };
        }
        Ok(directory)
    }

    fn make_directory(&mut self, parent: u32, name: &str, path: &str) -> Result<u32, FatError> {
        let cluster = self.allocate(None, 1)?.start;
        let added = self.add_entry(parent, name, path, ATTRIBUTE_DIRECTORY, cluster, 0);
        if let Err(error) = added {
            self.release(cluster)?;
            return Err(error);
        }
        let mut data = vec![0; self.cluster_bytes()];
        // The root directory is cluster 0 to its children, wherever it is.
        let above = if parent == self.root { 0 } else { parent };
        let links = [(b".          ", cluster), (b"..         ", above)];
        for ((short, target), record) in links.into_iter().zip(data.chunks_mut(ENTRY_SIZE)) {
            record.copy_from_slice(&short_record(short, ATTRIBUTE_DIRECTORY, target, 0));
        }
        let directory = Directory {
            clusters: vec![cluster],
            data,
            end: links.len() * ENTRY_SIZE,
            entries: Vec::new(),
            dirty: true,
        };
        self.directories.insert(cluster, directory);
        Ok(cluster)
    }

    /// Makes the directory `path`, components separated by `/`, and those
    /// above it that do not exist.
    pub fn create_dir_all(&mut self, path: &str) -> Result<(), FatError> {
        let components = path.split('/').filter(|component| !component.is_empty());
        self.walk(path, components, true).map(drop)
    }

    /// Makes the file `path` of the `size` bytes that `contents` yields. The
    /// directory it goes into has to exist.
    pub fn write_file(
        &mut self,
        path: &str,
        contents: &mut dyn Read,
        size: u64,
    ) -> Result<(), FatError> {
        let mut components = path.split('/').filter(|component| !component.is_empty());
        let name = components
            .next_back()
            .ok_or_else(|| FatError::InvalidName(path.into()))?;
        let parent = self.walk(path, components, false)?;
        validate_name(name)?;
        if self.find(parent, name)?.is_some() {
            return Err(FatError::Exists(path.into()));
        }
        let size32 = u32::try_from(size).map_err(|_| FatError::TooLarge(path.into()))?;
        if size.div_ceil(self.cluster_bytes() as u64) > self.free_clusters.into() {
            return Err(FatError::NoSpace);
        }
        let mut first = None;
        let written = self
            .write_contents(contents, size, &mut first)
            .and_then(|()| {
                let cluster = first.unwrap_or(0);
                self.add_entry(parent, name, path, ATTRIBUTE_ARCHIVE, cluster, size32)
            });
        if let (Err(_), Some(first)) = (&written, first) {
            self.release(first)?;
        }
        written
    }

    /// Chains clusters for `size` bytes of `contents` and fills them,
    /// storing the first in `first` as soon as there is one.
    fn write_contents(
        &mut self,
        contents: &mut dyn Read,
        size: u64,
        first: &mut Option<u32>,
    ) -> Result<(), FatError> {
        let cluster_bytes = self.cluster_bytes();
        let chunk_clusters = (WRITE_CHUNK / cluster_bytes).max(1) as u64;
        let mut buffer = Vec::new();
        let (mut remaining, mut previous) = (size, None);
        while remaining != 0 {
            let wanted = remaining.div_ceil(cluster_bytes as u64).min(chunk_clusters);
            let run = self.allocate(previous, wanted as u32)?;
            first.get_or_insert(run.start);
            previous = Some(run.end - 1);
            let run_bytes = run.len() * cluster_bytes;
            let length = usize::try_from(remaining).map_or(run_bytes, |rest| rest.min(run_bytes));
            // The last cluster is whole on the storage, zero after the
            // end of the file.
            buffer.clear();
            buffer.resize(run_bytes, 0);
            contents.read_exact(&mut buffer[..length])?;
            self.storage
                .write(self.cluster_sector(run.start), &buffer)?;
            remaining -= length as u64;
        }
        Ok(())
    }

    /// Writes the directories and FATs that changed and hands the storage
    /// back.
    pub fn finish(mut self) -> Result<S, FatError> {
        let cluster_bytes = self.cluster_bytes();
        for directory in self
            .directories
            .values()
            .filter(|directory| directory.dirty)
        {
            let contents = directory.data.chunks(cluster_bytes);
            for (&cluster, contents) in directory.clusters.iter().zip(contents) {
                let sector = self.cluster_sector(cluster);
                self.storage.write(sector, contents)?;
            }
        }
        if self.fat_changed {
            for copy in 0..self.fat_count {
                let sector = self.fat_start + copy * self.fat_sectors;
                self.storage.write(sector.into(), &self.fat)?;
            }
        }
        Ok(self.storage)
    }
}

/// A file as a directory lists it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileInfo {
    pub name: String,
    pub directory: bool,
    pub size: u32,
}

/// A FAT32 volume a console may have mounted, in whatever state, for
/// reading files and for rewriting what an existing file holds. Neither
/// touches a directory or a FAT, the only sectors the driver's recovery
/// file (see the module documentation) saves and, when the console
/// stopped during an operation, puts back, so it stays valid.
pub struct MountedFat32<S>(Fat32<S>);

impl<S: Sectors> MountedFat32<S> {
    pub fn open(storage: S) -> Result<Self, FatError> {
        Fat32::read_volume(storage).map(Self)
    }

    /// The entries of the directory `path`, without `.` and `..`.
    pub fn list(&mut self, path: &str) -> Result<Vec<FileInfo>, FatError> {
        let components = path.split('/').filter(|component| !component.is_empty());
        let directory = self.0.walk(path, components, false)?;
        self.0.load(directory)?;
        Ok(self.0.directories[&directory]
            .entries
            .iter()
            .map(|entry| FileInfo {
                name: entry.name(),
                directory: entry.directory,
                size: entry.size,
            })
            .collect())
    }

    /// First cluster and size of the file `path`.
    fn file(&mut self, path: &str) -> Result<(u32, u32), FatError> {
        let mut components = path.split('/').filter(|component| !component.is_empty());
        let name = components
            .next_back()
            .ok_or_else(|| FatError::NotFound(path.into()))?;
        let parent = self.0.walk(path, components, false)?;
        self.0.load(parent)?;
        let entry = self.0.directories[&parent]
            .entries
            .iter()
            .find(|entry| entry.name().eq_ignore_ascii_case(name))
            .ok_or_else(|| FatError::NotFound(path.into()))?;
        if entry.directory {
            return Err(FatError::NotFound(path.into()));
        }
        Ok((entry.cluster, entry.size))
    }

    /// The sectors of the file `path` that hold its first `length` bytes,
    /// or all of them when it is shorter, each with the bytes of the file
    /// it holds.
    fn sectors(&mut self, path: &str, length: usize) -> Result<Vec<(u64, usize)>, FatError> {
        let (first, size) = self.file(path)?;
        let length = length.min(size as usize);
        let chain = if size == 0 {
            Vec::new()
        } else {
            self.0.chain(first)?
        };
        let per_cluster = self.0.sectors_per_cluster as usize;
        let sectors = length.div_ceil(SECTOR_SIZE);
        if sectors > chain.len() * per_cluster {
            return Err(FatError::Corrupt("a file is longer than its clusters"));
        }
        Ok((0..sectors)
            .map(|index| {
                let cluster = chain[index / per_cluster];
                let sector = self.0.cluster_sector(cluster) + (index % per_cluster) as u64;
                (sector, (length - index * SECTOR_SIZE).min(SECTOR_SIZE))
            })
            .collect())
    }

    /// The first `length` bytes of the file `path`, or all of it when it is
    /// shorter.
    pub fn read(&mut self, path: &str, length: usize) -> Result<Vec<u8>, FatError> {
        let mut data = Vec::with_capacity(length);
        let mut sector_data = [0; SECTOR_SIZE];
        for (sector, bytes) in self.sectors(path, length)? {
            self.0.storage.read(sector, &mut sector_data)?;
            data.extend_from_slice(&sector_data[..bytes]);
        }
        Ok(data)
    }

    /// Stores `contents` at the start of the file `path`, which keeps its
    /// size and its clusters: whatever it held after `contents` (from the
    /// end of their last sector on) stays. `contents` may not be longer
    /// than the file.
    pub fn overwrite(&mut self, path: &str, contents: &[u8]) -> Result<(), FatError> {
        let (_, size) = self.file(path)?;
        if contents.len() > size as usize {
            return Err(FatError::LongerThanFile(path.into()));
        }
        let mut sector_data = [0; SECTOR_SIZE];
        for ((sector, bytes), chunk) in self
            .sectors(path, contents.len())?
            .into_iter()
            .zip(contents.chunks(SECTOR_SIZE))
        {
            sector_data.fill(0);
            sector_data[..bytes].copy_from_slice(chunk);
            self.0.storage.write(sector, &sector_data)?;
        }
        Ok(())
    }

    pub fn into_storage(self) -> S {
        self.0.storage
    }
}

/// A second, read-only implementation of the on-disk format, for tests to
/// check volumes against.
#[cfg(test)]
pub(crate) mod reader {
    use super::{SECTOR_SIZE, Sectors};

    pub(crate) struct Reader<S> {
        storage: S,
        sectors_per_cluster: u64,
        reserved: u64,
        fat_sectors: u64,
        data_start: u64,
        pub(crate) clusters: u32,
        root: u32,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(crate) struct Item {
        pub(crate) name: String,
        pub(crate) short: String,
        pub(crate) directory: bool,
        pub(crate) cluster: u32,
        pub(crate) size: u32,
    }

    fn le(bytes: &[u8]) -> u64 {
        bytes
            .iter()
            .rev()
            .fold(0, |value, &byte| value << 8 | u64::from(byte))
    }

    impl<S: Sectors> Reader<S> {
        /// Checks the boot sector, its copy, the information sectors and
        /// that both FATs agree.
        pub(crate) fn open(mut storage: S) -> Self {
            let mut boot = [0; SECTOR_SIZE * 9];
            storage.read(0, &mut boot).unwrap();
            assert_eq!(boot[510..512], [0x55, 0xAA]);
            assert_eq!(le(&boot[11..13]), 512);
            assert_eq!(boot[16], 2, "two FATs");
            assert_eq!(le(&boot[17..19]), 0, "FAT32 has no fixed root");
            assert_eq!(&boot[82..90], b"FAT32   ");
            assert_eq!(boot[..512], boot[6 * 512..7 * 512], "backup boot sector");
            assert_eq!((le(&boot[48..50]), le(&boot[50..52])), (1, 6));
            for information in [1, 7] {
                let sector = &boot[information * 512..][..512];
                assert_eq!(&sector[..4], b"RRaA");
                assert_eq!(&sector[484..488], b"rrAa");
                assert_eq!(sector[508..], [0, 0, 0x55, 0xAA]);
            }
            let (reserved, fat_sectors) = (le(&boot[14..16]), le(&boot[36..40]));
            let sectors_per_cluster = u64::from(boot[13]);
            let data_start = reserved + 2 * fat_sectors;
            let clusters = ((le(&boot[32..36]) - data_start) / sectors_per_cluster) as u32;
            assert!(clusters >= 65525, "a FAT32 cluster count");
            assert!(fat_sectors * 128 >= u64::from(clusters) + 2);
            let mut reader = Self {
                storage,
                sectors_per_cluster,
                reserved,
                fat_sectors,
                data_start,
                clusters,
                root: le(&boot[44..48]) as u32,
            };
            assert_eq!(reader.fat(0), reader.fat(1), "the FATs are copies");
            assert_eq!(reader.fat(0)[..2], [0x0FFF_FFF8, 0x0FFF_FFFF]);
            reader
        }

        pub(crate) fn sector(&mut self, sector: u64) -> [u8; SECTOR_SIZE] {
            let mut data = [0; SECTOR_SIZE];
            self.storage.read(sector, &mut data).unwrap();
            data
        }

        pub(crate) fn fat(&mut self, copy: u64) -> Vec<u32> {
            let mut raw = vec![0; self.fat_sectors as usize * SECTOR_SIZE];
            self.storage
                .read(self.reserved + copy * self.fat_sectors, &mut raw)
                .unwrap();
            raw.chunks(4).map(|entry| le(entry) as u32).collect()
        }

        pub(crate) fn chain(&mut self, first: u32) -> Vec<u32> {
            let fat = self.fat(0);
            let mut chain = Vec::new();
            let mut cluster = first;
            while cluster < 0x0FFF_FFF8 {
                assert!(
                    (2..self.clusters + 2).contains(&cluster),
                    "chain within the volume"
                );
                chain.push(cluster);
                cluster = fat[cluster as usize];
            }
            chain
        }

        fn contents(&mut self, first: u32) -> Vec<u8> {
            let cluster_bytes = self.sectors_per_cluster as usize * SECTOR_SIZE;
            let mut data = Vec::new();
            for cluster in self.chain(first) {
                let start = data.len();
                data.resize(start + cluster_bytes, 0);
                let sector = self.data_start + u64::from(cluster - 2) * self.sectors_per_cluster;
                self.storage.read(sector, &mut data[start..]).unwrap();
            }
            data
        }

        /// The entries of the directory at `cluster`, long names put
        /// together and checked against their short entries.
        pub(crate) fn directory(&mut self, cluster: u32) -> Vec<Item> {
            let data = self.contents(cluster);
            let mut items = Vec::new();
            let mut long: Vec<u16> = Vec::new();
            let mut expected = None;
            for record in data.chunks(32) {
                if record[0] == 0 {
                    break;
                }
                if record[11] == 0x0F {
                    let number = record[0] & 0x3F;
                    if record[0] & 0x40 != 0 {
                        long.clear();
                    } else {
                        assert_eq!(expected, Some((number + 1, record[13])), "parts count down");
                    }
                    expected = Some((number, record[13]));
                    let units = [1, 3, 5, 7, 9, 14, 16, 18, 20, 22, 24, 28, 30]
                        .map(|offset| le(&record[offset..offset + 2]) as u16);
                    long.splice(0..0, units);
                    continue;
                }
                let mut sum = 0u8;
                for &character in &record[..11] {
                    sum = sum.rotate_right(1).wrapping_add(character);
                }
                let base = String::from_utf8_lossy(&record[..8]).trim_end().to_string();
                let extension = String::from_utf8_lossy(&record[8..11])
                    .trim_end()
                    .to_string();
                let short = if extension.is_empty() {
                    base
                } else {
                    format!("{base}.{extension}")
                };
                let name = match expected.take() {
                    Some((number, checksum)) => {
                        assert_eq!((number, checksum), (1, sum), "long name belongs to entry");
                        let end = long
                            .iter()
                            .position(|&unit| unit == 0)
                            .unwrap_or(long.len());
                        assert!(long[end..].iter().skip(1).all(|&unit| unit == 0xFFFF));
                        String::from_utf16(&long[..end]).unwrap()
                    }
                    None => short.clone(),
                };
                items.push(Item {
                    name,
                    short,
                    directory: record[11] & 0x10 != 0,
                    cluster: (le(&record[20..22]) << 16 | le(&record[26..28])) as u32,
                    size: le(&record[28..32]) as u32,
                });
            }
            items
        }

        fn resolve(&mut self, path: &str) -> Item {
            let mut item = Item {
                name: String::new(),
                short: String::new(),
                directory: true,
                cluster: self.root,
                size: 0,
            };
            for name in path.split('/').filter(|name| !name.is_empty()) {
                assert!(item.directory, "{path}: a file on the way");
                item = self
                    .directory(item.cluster)
                    .into_iter()
                    .find(|item| item.name == name)
                    .unwrap_or_else(|| panic!("{path}: no {name}"));
            }
            item
        }

        /// Names in the directory `path`, without the two links every
        /// directory but the root starts with.
        pub(crate) fn names(&mut self, path: &str) -> Vec<String> {
            let cluster = self.resolve(path).cluster;
            let items = self.directory(cluster);
            items
                .into_iter()
                .map(|item| item.name)
                .filter(|name| !name.starts_with('.'))
                .collect()
        }

        pub(crate) fn item(&mut self, path: &str) -> Item {
            self.resolve(path)
        }

        pub(crate) fn file(&mut self, path: &str) -> Vec<u8> {
            let item = self.resolve(path);
            assert!(!item.directory, "{path} is a directory");
            let mut data = if item.size == 0 {
                Vec::new()
            } else {
                self.contents(item.cluster)
            };
            assert!(
                data[item.size as usize..].iter().all(|&byte| byte == 0),
                "{path}: zero after the end"
            );
            data.truncate(item.size as usize);
            data
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::{reader::Reader, *};

    /// Storage in memory that remembers which sectors were written.
    pub(crate) struct Memory {
        data: Vec<u8>,
        written: Vec<bool>,
    }

    impl Memory {
        pub(crate) fn new(sectors: usize) -> Self {
            Self {
                data: vec![0; sectors * SECTOR_SIZE],
                written: vec![false; sectors],
            }
        }
    }

    impl Sectors for Memory {
        fn read(&mut self, sector: u64, data: &mut [u8]) -> io::Result<()> {
            let start = sector as usize * SECTOR_SIZE;
            data.copy_from_slice(&self.data[start..start + data.len()]);
            Ok(())
        }

        fn write(&mut self, sector: u64, data: &[u8]) -> io::Result<()> {
            let start = sector as usize * SECTOR_SIZE;
            self.data[start..start + data.len()].copy_from_slice(data);
            self.written[sector as usize..][..data.len() / SECTOR_SIZE].fill(true);
            Ok(())
        }
    }

    /// The smallest partition the console formats as FAT32 is just above 33
    /// MiB; its clusters are single sectors.
    const SECTORS: usize = 0x220_0000 / SECTOR_SIZE;

    fn pattern(length: usize, seed: u8) -> Vec<u8> {
        (0..length)
            .map(|index| (index as u8).wrapping_mul(31).wrapping_add(seed))
            .collect()
    }

    #[test]
    fn the_console_partitions_get_the_fat_type_their_cluster_count_implies() {
        // Sectors and FAT safe; then type, sectors per cluster, reserved
        // sectors and FAT sectors.
        for (sectors, fat_safe, fat_type, per_cluster, reserved, fat) in [
            (0x2000, false, FatType::Fat12, 32, 1, 1),
            (0x2_0000, true, FatType::Fat32, 1, 32, 1024),
            (0x50_0000, true, FatType::Fat32, 32, 32, 1280),
            (0x340_0000, true, FatType::Fat32, 32, 32, 13312),
        ] {
            let layout = Layout::for_partition(sectors, fat_safe).unwrap();
            assert_eq!(
                (
                    layout.fat_type,
                    layout.sectors_per_cluster,
                    layout.reserved_sectors,
                    layout.fat_sectors
                ),
                (fat_type, per_cluster, reserved, fat)
            );
            // A driver tells the types apart by the cluster count alone, and
            // every cluster needs its entry.
            let clusters = layout.cluster_count();
            let entry_bits = match fat_type {
                FatType::Fat12 => {
                    assert!(clusters < 4085);
                    12
                }
                FatType::Fat32 => {
                    assert!(clusters >= 65525);
                    assert_eq!(layout.data_start() % 32, 0, "clusters on cipher sectors");
                    32
                }
            };
            assert!(u64::from(fat) * 4096 >= (u64::from(clusters) + 2) * entry_bits);
        }
        // Two gigabytes and a bit: a range the formatter rejects.
        assert!(matches!(
            Layout::for_partition(0x8100_0000 / 512, true),
            Err(FatError::UnsupportedSize(0x8100_0000))
        ));
    }

    #[test]
    fn format_writes_every_sector_a_mount_reads_and_leaves_the_clusters_alone() {
        let mut memory = Memory::new(SECTORS);
        let layout = format(&mut memory, SECTORS as u64, true, 0x1234_5678).unwrap();
        let data_start = layout.data_start() as usize;
        // Reserved area and FATs, then the root directory's cluster.
        assert!(memory.written[..=data_start].iter().all(|&written| written));
        assert!(
            memory.written[data_start + 1..]
                .iter()
                .all(|&written| !written)
        );

        let mut reader = Reader::open(&mut memory);
        assert_eq!(reader.names("/"), [""; 0]);
        assert_eq!(
            reader.sector(3),
            [0; SECTOR_SIZE],
            "free for the recovery file's location"
        );
        let fat = reader.fat(0);
        assert_eq!(fat[2], 0x0FFF_FFFF, "the root directory is one cluster");
        assert!(fat[3..].iter().all(|&entry| entry == 0));
    }

    #[test]
    fn an_empty_fat12_volume_is_its_boot_sector_fats_and_root_directory() {
        let mut memory = Memory::new(0x2000);
        let layout = format(&mut memory, 0x2000, false, 1).unwrap();
        assert_eq!(layout.data_start(), 35);
        assert!(memory.written[..35].iter().all(|&written| written));
        let boot = &memory.data[..SECTOR_SIZE];
        assert_eq!(&boot[54..62], b"FAT12   ");
        assert_eq!(
            (get16(boot, 17), get16(boot, 19), get16(boot, 22)),
            (512, 0x2000, 1)
        );
        for fat in [1, 2] {
            assert_eq!(memory.data[fat * SECTOR_SIZE..][..4], [0xF8, 0xFF, 0xFF, 0]);
        }
    }

    #[test]
    fn nested_directories_and_long_named_files_read_back_through_another_parser() {
        let mut memory = Memory::new(SECTORS);
        format(&mut memory, SECTORS as u64, true, 7).unwrap();
        let mut volume = Fat32::open(&mut memory).unwrap();
        volume.create_dir_all("/Contents/registered").unwrap();
        volume.create_dir_all("Contents/placehld/").unwrap();
        volume.create_dir_all("/SAVE").unwrap();
        // Enough long names that the directory outgrows its first cluster,
        // and names that share their first six characters.
        let files: Vec<(String, Vec<u8>)> = (0..40u8)
            .map(|index| {
                let name = format!("/Contents/registered/00e7f53a{index:024x}.nca");
                (name, pattern(usize::from(index) * 300, index))
            })
            .collect();
        for (path, contents) in &files {
            volume
                .write_file(path, &mut contents.as_slice(), contents.len() as u64)
                .unwrap();
        }
        volume
            .write_file("/SAVE/A.BIN", &mut &[1u8, 2, 3][..], 3)
            .unwrap();
        volume.finish().unwrap();

        let mut reader = Reader::open(&mut memory);
        assert_eq!(reader.names("/"), ["Contents", "SAVE"]);
        assert_eq!(reader.names("/Contents"), ["registered", "placehld"]);
        assert_eq!(reader.names("/Contents/placehld"), [""; 0]);
        for (path, contents) in &files {
            assert_eq!(reader.file(path), *contents, "{path}");
        }
        assert_eq!(reader.file("/SAVE/A.BIN"), [1, 2, 3]);

        // A name that is 8.3 as it stands has no long name; the others have
        // distinct aliases.
        assert_eq!(reader.item("/SAVE").short, "SAVE");
        assert_eq!(reader.item("/Contents").short, "CONTEN~1");
        let registered = reader.item("/Contents/registered");
        let mut aliases: Vec<String> = reader
            .directory(registered.cluster)
            .into_iter()
            .map(|item| item.short)
            .collect();
        assert_eq!(aliases[..3], [".", "..", "00E7F5~1.NCA"]);
        assert!(aliases.contains(&"00E7F~10.NCA".to_string()));
        aliases.sort();
        aliases.dedup();
        assert_eq!(aliases.len(), 42);
        assert!(
            reader.chain(registered.cluster).len() > 1,
            "the directory grew"
        );

        // Every directory but the root links to itself and to its parent,
        // the root being cluster 0 there.
        let links = reader.directory(registered.cluster);
        let contents = reader.item("/Contents").cluster;
        assert_eq!(
            (links[0].cluster, links[1].cluster),
            (registered.cluster, contents)
        );
        assert_eq!(reader.directory(contents)[1].cluster, 0);

        // No cluster is marked used that nothing reaches: the root, the
        // three directories of one cluster, the one that grew, the files
        // in it and the one cluster of the file beside them.
        let used = reader.fat(0)[2..]
            .iter()
            .filter(|&&entry| entry != 0)
            .count();
        let file_clusters: usize = files.iter().map(|file| file.1.len().div_ceil(512)).sum();
        let directories = 4 + reader.chain(registered.cluster).len();
        assert_eq!(used, file_clusters + 1 + directories);
    }

    #[test]
    fn a_volume_can_be_added_to_again_and_keeps_what_it_had() {
        let mut memory = Memory::new(SECTORS);
        format(&mut memory, SECTORS as u64, true, 7).unwrap();
        let mut volume = Fat32::open(&mut memory).unwrap();
        volume.create_dir_all("/a/Long Directory Name").unwrap();
        let first = pattern(5000, 1);
        volume
            .write_file(
                "/a/Long Directory Name/one.bin",
                &mut first.as_slice(),
                5000,
            )
            .unwrap();
        volume.finish().unwrap();

        let mut volume = Fat32::open(&mut memory).unwrap();
        // Found again by long name, in any case, and by alias.
        volume.create_dir_all("/A/long directory name/sub").unwrap();
        volume.create_dir_all("/a/LONGDI~1/sub").unwrap();
        let second = pattern(700, 2);
        volume
            .write_file(
                "/a/Long Directory Name/two.bin",
                &mut second.as_slice(),
                700,
            )
            .unwrap();
        let error = volume.write_file("/a/long directory name/ONE.BIN", &mut &[][..], 0);
        assert!(matches!(error, Err(FatError::Exists(_))));
        let error = volume.write_file("/a/missing/x", &mut &[][..], 0);
        assert!(matches!(error, Err(FatError::NotFound(_))));
        let error = volume.create_dir_all("/a/Long Directory Name/one.bin/x");
        assert!(matches!(error, Err(FatError::NotADirectory(_))));
        let error = volume.create_dir_all("/a/b:c");
        assert!(matches!(error, Err(FatError::InvalidName(_))));
        volume.finish().unwrap();

        let mut reader = Reader::open(&mut memory);
        assert_eq!(
            reader.names("/a/Long Directory Name"),
            ["one.bin", "sub", "two.bin"]
        );
        assert_eq!(reader.file("/a/Long Directory Name/one.bin"), first);
        assert_eq!(reader.file("/a/Long Directory Name/two.bin"), second);
    }

    #[test]
    fn a_file_that_cannot_be_completed_leaves_no_trace() {
        let mut memory = Memory::new(SECTORS);
        format(&mut memory, SECTORS as u64, true, 7).unwrap();
        let mut volume = Fat32::open(&mut memory).unwrap();
        // The contents end before the size they were announced with.
        let error = volume.write_file("/short.bin", &mut &[0u8; 100][..], 4000);
        assert!(matches!(error, Err(FatError::Io(_))));
        let error = volume.write_file("/huge.bin", &mut io::empty(), 1 << 30);
        assert!(matches!(error, Err(FatError::NoSpace)));
        volume
            .write_file("/kept.bin", &mut &[9u8; 600][..], 600)
            .unwrap();
        volume.finish().unwrap();

        let mut reader = Reader::open(&mut memory);
        assert_eq!(reader.names("/"), ["kept.bin"]);
        let used = reader.fat(0)[2..]
            .iter()
            .filter(|&&entry| entry != 0)
            .count();
        assert_eq!(
            used,
            1 + 2,
            "the root directory and the two clusters of the file"
        );
    }

    /// A record or recovery-file header of the FAT-safe driver with the
    /// words `words` (offset, value), as `0x11ecb0` request 4 and
    /// `0x11cf30` write them.
    fn safe_sector(words: &[(usize, u32)]) -> [u8; SECTOR_SIZE] {
        let mut sector = [0; SECTOR_SIZE];
        sector[..SAFE_MAGIC.len()].copy_from_slice(SAFE_MAGIC);
        sector[16] = 100;
        sector[17] = 101;
        for &(offset, value) in words {
            put32(&mut sector, offset, value);
        }
        let sum = super::super::crc32(&sector);
        put32(&mut sector, 508, sum);
        sector
    }

    const CONSOLE_FILE: &str = "/Contents/registered/00e7f53ae3ee4d6fd915b1ae26fbd1be.nca";

    /// Makes `memory` a FAT-safe volume as a console leaves it after
    /// mounting it: things of its own, then the recovery file, full of
    /// stale entries behind a header that has `operation` in progress (0
    /// when the console stopped between operations), and the record in
    /// sector 3. Where the recovery file lies, in sectors.
    fn mounted(memory: &mut Memory, operation: u32) -> Range<usize> {
        let layout = format(memory, SECTORS as u64, true, 7).unwrap();
        let mut volume = Fat32::open(&mut *memory).unwrap();
        volume.create_dir_all("/save").unwrap();
        volume.create_dir_all("/Contents/registered").unwrap();
        let console = pattern(3000, 3);
        volume
            .write_file(CONSOLE_FILE, &mut console.as_slice(), 3000)
            .unwrap();
        let sectors = layout.fat_sectors.div_ceil(4096) + RECOVERY_FIXED_SECTORS;
        let mut recovery = pattern(sectors as usize * SECTOR_SIZE, 5);
        let transaction = if operation == 0 { 0 } else { 6 };
        let header = safe_sector(&[(22, transaction), (32, operation)]);
        for copy in recovery.chunks_mut(SECTOR_SIZE).take(2) {
            copy.copy_from_slice(&header);
        }
        volume
            .write_file(
                "/PRF2SAFE.RCV",
                &mut recovery.as_slice(),
                recovery.len() as u64,
            )
            .unwrap();
        volume.finish().unwrap();
        let first = Reader::open(&mut *memory).item("/PRF2SAFE.RCV").cluster;
        let record = safe_sector(&[(32, first), (36, sectors)]);
        memory.write(RECORD_SECTOR.into(), &record).unwrap();
        // Clusters are single sectors here.
        let start = layout.data_start() as usize + first as usize - 2;
        start..start + sectors as usize
    }

    #[test]
    fn a_volume_a_console_left_between_operations_takes_files_around_its_recovery_file() {
        let mut memory = Memory::new(SECTORS);
        let recovery = mounted(&mut memory, 0);
        let reserved = get16(&memory.data, 14) as usize;
        let reserved_before = memory.data[..reserved * SECTOR_SIZE].to_vec();
        let bytes = |sectors: &Range<usize>| sectors.start * SECTOR_SIZE..sectors.end * SECTOR_SIZE;
        let recovery_before = memory.data[bytes(&recovery)].to_vec();
        memory.written.fill(false);

        let mut volume = Fat32::open(&mut memory).unwrap();
        volume
            .create_dir_all("/Contents/registered/000000EE")
            .unwrap();
        let nca = pattern(7000, 8);
        let path = "/Contents/registered/000000EE/test.nca";
        volume.write_file(path, &mut nca.as_slice(), 7000).unwrap();
        // Shares its first six characters with the console's file.
        let neighbour = "/Contents/registered/00e7f53a00000000000000000000000f.nca";
        volume
            .write_file(neighbour, &mut &[4u8; 10][..], 10)
            .unwrap();
        volume
            .write_file("/save/8000000000000120", &mut &[5u8; 600][..], 600)
            .unwrap();
        let error = volume.write_file("/prf2safe.rcv", &mut &[][..], 0);
        assert!(matches!(error, Err(FatError::Exists(_))));
        let error = volume.create_dir_all("/PRF2SAFE.RCV/x");
        assert!(matches!(error, Err(FatError::NotADirectory(_))));
        volume.finish().unwrap();

        assert!(
            memory.written[..reserved].iter().all(|&written| !written),
            "no reserved sector written"
        );
        assert!(
            memory.written[recovery.clone()]
                .iter()
                .all(|&written| !written)
        );
        assert_eq!(memory.data[..reserved * SECTOR_SIZE], reserved_before);
        assert_eq!(memory.data[bytes(&recovery)], recovery_before);

        let mut reader = Reader::open(&mut memory);
        assert_eq!(reader.file(path), nca);
        assert_eq!(reader.file(neighbour), [4; 10]);
        assert_eq!(reader.file(CONSOLE_FILE), pattern(3000, 3));
        assert_eq!(reader.file("/save/8000000000000120"), [5; 600]);
        assert_eq!(
            reader.item("/Contents/registered/000000EE").short,
            "000000EE"
        );
        let registered = reader.item("/Contents/registered").cluster;
        let aliases: Vec<String> = reader
            .directory(registered)
            .into_iter()
            .map(|item| item.short)
            .collect();
        assert_eq!(
            aliases,
            [".", "..", "00E7F5~1.NCA", "000000EE", "00E7F5~2.NCA"]
        );
        let first = reader.item("/PRF2SAFE.RCV").cluster;
        assert_eq!(reader.chain(first).len(), recovery.len());
        assert_eq!(reader.fat(0), reader.fat(1));
        drop(reader);
        assert!(Fat32::open(&mut memory).is_ok(), "still between operations");
    }

    #[test]
    fn a_volume_a_console_left_during_an_operation_is_refused() {
        let mut memory = Memory::new(SECTORS);
        mounted(&mut memory, 9);
        assert!(matches!(
            Fat32::open(&mut memory),
            Err(FatError::RecoveryPending {
                transaction: 6,
                operation: 9
            })
        ));
    }

    #[test]
    fn recovery_data_that_does_not_match_the_volume_is_refused() {
        // A damaged record.
        let mut memory = Memory::new(SECTORS);
        mounted(&mut memory, 0);
        memory.data[RECORD_SECTOR as usize * SECTOR_SIZE + 40] ^= 1;
        assert!(matches!(
            Fat32::open(&mut memory),
            Err(FatError::RecoveryDamaged(_))
        ));
        // A header and its copy damaged.
        let mut memory = Memory::new(SECTORS);
        let recovery = mounted(&mut memory, 0);
        for copy in 0..2 {
            memory.data[(recovery.start + copy) * SECTOR_SIZE + 100] ^= 1;
        }
        assert!(matches!(
            Fat32::open(&mut memory),
            Err(FatError::RecoveryDamaged(_))
        ));
        // Only the header damaged: its copy counts.
        let mut memory = Memory::new(SECTORS);
        let recovery = mounted(&mut memory, 0);
        memory.data[recovery.start * SECTOR_SIZE + 100] ^= 1;
        assert!(Fat32::open(&mut memory).is_ok());
        // A record that names another place than the file's.
        let mut memory = Memory::new(SECTORS);
        let recovery = mounted(&mut memory, 0);
        let elsewhere = safe_sector(&[(32, 3), (36, recovery.len() as u32)]);
        memory.write(RECORD_SECTOR.into(), &elsewhere).unwrap();
        assert!(matches!(
            Fat32::open(&mut memory),
            Err(FatError::RecoveryDamaged(_))
        ));
    }

    #[test]
    fn a_recovery_file_without_a_record_is_the_writers_own_business() {
        // Without the record the driver's next mount ignores the file and
        // makes another, so the volume counts as never mounted.
        let mut memory = Memory::new(SECTORS);
        format(&mut memory, SECTORS as u64, true, 7).unwrap();
        let mut volume = Fat32::open(&mut memory).unwrap();
        volume
            .write_file("/PRF2SAFE.RCV", &mut &[0u8; 16][..], 16)
            .unwrap();
        volume.finish().unwrap();
        assert!(Fat32::open(&mut memory).is_ok());
    }

    #[test]
    fn a_mounted_volume_has_file_contents_replaced_and_nothing_else() {
        let mut memory = Memory::new(SECTORS);
        format(&mut memory, SECTORS as u64, true, 7).unwrap();
        let mut volume = Fat32::open(&mut memory).unwrap();
        volume.create_dir_all("/Contents/registered").unwrap();
        let old = pattern(5000, 1);
        let path = "/Contents/registered/0123456789abcdef0123456789abcdef.nca";
        volume.write_file(path, &mut old.as_slice(), 5000).unwrap();
        volume
            .write_file("/PRF2SAFE.RCV", &mut &[0u8; 16][..], 16)
            .unwrap();
        volume.finish().unwrap();
        memory.written.fill(false);

        let mut mounted = MountedFat32::open(&mut memory).unwrap();
        let listed = mounted.list("/Contents/registered").unwrap();
        assert_eq!(
            listed,
            [FileInfo {
                name: "0123456789abcdef0123456789abcdef.nca".into(),
                directory: false,
                size: 5000,
            }]
        );
        assert_eq!(mounted.read(path, 700).unwrap(), old[..700]);
        let new = pattern(1200, 9);
        mounted.overwrite(path, &new).unwrap();
        assert!(matches!(
            mounted.overwrite(path, &pattern(5001, 9)),
            Err(FatError::LongerThanFile(_))
        ));
        drop(mounted);

        // Three sectors of file data, the last zero after the new bytes,
        // and the rest of the file as it was.
        assert_eq!(memory.written.iter().filter(|&&written| written).count(), 3);
        let mut expected = new.clone();
        expected.resize(1536, 0);
        expected.extend_from_slice(&old[1536..]);
        assert_eq!(Reader::open(&mut memory).file(path), expected);
    }
}
