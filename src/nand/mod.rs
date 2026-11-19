//! A console's eMMC user area, made on the host: the partition table, the
//! empty filesystems of a console that left the factory, and the system
//! contents of a firmware.
//!
//! The filesystem process finds its partitions by type GUID in the GUID
//! partition table at sector 1; it holds the eleven GUIDs in a table at
//! offset `0x1eccac` of its image and picks one by partition ID through the
//! jump table at `0x1ecdd0`. The names, offsets and sizes below are the
//! fixed layout of the 32 GB module. [`format_user_area`] writes the table
//! and nothing else: a module that was partitioned and never written.
//!
//! Four of the partitions hold FAT volumes, each encrypted as a whole with
//! one of the console's storage keys ([`bis`]): the calibration files
//! (`PRODINFOF`, FAT12), and `SAFE`, `SYSTEM` and `USER` (FAT32).
//! [`format_filesystems`] makes them the empty volumes the filesystem
//! process's own formatter would ([`fat`]). The partitions without a
//! filesystem stay untouched, so the calibration data is still missing;
//! [`boot_images`] fills the boot partitions and the raw `BCPKG2-*`
//! partitions from a firmware's boot image packages.
//!
//! [`install_firmware`] puts the content archives of a firmware where the
//! content manager looks for built-in system contents. Its storage for
//! them is the directory `Contents` of the system partition (the
//! filesystem process opens partition 31 and that path for content
//! storage 0, at `0x81b10`). In there the content manager wants the
//! directories `registered` and `placehld` (`0x35600` of its image), and
//! names the archive of a content ID `registered/<32 hex digits>.nca`:
//! of its four ways to build a path it gives the storage of ID 3, the
//! built-in system one, the flat one (`0x305e0` choosing `0x39a30`), and
//! keeps those that spread archives over directories named after a hash
//! for the user partition and the SD card. Which archive belongs to which
//! program it reads from its database, a system save, which this module
//! does not make.
//!
//! The image stays sparse. The storage cipher turns what was never written
//! into noise, so everything the guest will read is written, zero or not,
//! and nothing else is.

use std::{
    collections::HashMap,
    fmt,
    fs::File,
    io,
    ops::Range,
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
};

use crate::keys::KeySet;
use bis::{BisKeys, MissingKey, Xts};
use fat::{Fat32, FatError, MountedFat32, Sectors};

pub mod bis;
pub mod boot_images;
pub mod calibration;
pub mod cnmt;
pub mod content_db;
pub mod crypto;
pub mod fat;
pub mod nca;
pub mod romfs;
pub mod save;
pub mod save_indexer;

const SECTOR_SIZE: u64 = 512;

/// User area of the 32 GB module.
pub const USER_AREA_SIZE: u64 = 0x7_47C0_0000;
const SECTORS: u64 = USER_AREA_SIZE / SECTOR_SIZE;

const ENTRY_COUNT: u32 = 128;
const ENTRY_SIZE: u32 = 128;
/// Sectors of the partition entry array.
const ENTRY_SECTORS: u64 = (ENTRY_COUNT * ENTRY_SIZE) as u64 / SECTOR_SIZE;
const HEADER_SIZE: u32 = 92;
/// First sector after the protective MBR, the header and the entries.
const FIRST_USABLE: u64 = 2 + ENTRY_SECTORS;
const LAST_USABLE: u64 = SECTORS - ENTRY_SECTORS - 2;

struct Partition {
    name: &'static str,
    /// Type GUID as it is stored on disk.
    type_guid: [u8; 16],
    sectors: u64,
}

const fn guid(time_low: u32, time_mid: u16, time_high: u16, rest: [u8; 8]) -> [u8; 16] {
    let (a, b, c) = (
        time_low.to_le_bytes(),
        time_mid.to_le_bytes(),
        time_high.to_le_bytes(),
    );
    [
        a[0], a[1], a[2], a[3], b[0], b[1], c[0], c[1], rest[0], rest[1], rest[2], rest[3],
        rest[4], rest[5], rest[6], rest[7],
    ]
}

/// In disk order, each starting where the previous one ends.
const PARTITIONS: [Partition; 11] = [
    Partition {
        name: "PRODINFO",
        type_guid: guid(
            0x98109E25,
            0x64E2,
            0x4C95,
            [0x8A, 0x77, 0x41, 0x49, 0x16, 0xF5, 0xBC, 0xEB],
        ),
        sectors: 0x1FDE,
    },
    Partition {
        name: "PRODINFOF",
        type_guid: guid(
            0xF3056AEC,
            0x5449,
            0x494C,
            [0x9F, 0x2C, 0x5F, 0xDC, 0xB7, 0x5B, 0x6E, 0x6E],
        ),
        sectors: 0x2000,
    },
    Partition {
        name: "BCPKG2-1-Normal-Main",
        type_guid: guid(
            0x5365DE36,
            0x911B,
            0x4BB4,
            [0x8F, 0xF9, 0xAA, 0x1E, 0xBC, 0xD7, 0x39, 0x90],
        ),
        sectors: 0x4000,
    },
    Partition {
        name: "BCPKG2-2-Normal-Sub",
        type_guid: guid(
            0x8455717B,
            0xBD2B,
            0x4162,
            [0x84, 0x54, 0x91, 0x69, 0x52, 0x18, 0xFC, 0x38],
        ),
        sectors: 0x4000,
    },
    Partition {
        name: "BCPKG2-3-SafeMode-Main",
        type_guid: guid(
            0x8ED6C9A6,
            0x9C48,
            0x490B,
            [0xBB, 0xEB, 0x00, 0x1D, 0x17, 0xA4, 0xC0, 0xF7],
        ),
        sectors: 0x4000,
    },
    Partition {
        name: "BCPKG2-4-SafeMode-Sub",
        type_guid: guid(
            0x5E99751C,
            0x56C9,
            0x47CC,
            [0xAA, 0x30, 0xB6, 0x50, 0x39, 0x88, 0x89, 0x17],
        ),
        sectors: 0x4000,
    },
    Partition {
        name: "BCPKG2-5-Repair-Main",
        type_guid: guid(
            0xC447D9A2,
            0x24B7,
            0x468A,
            [0x98, 0xC8, 0x59, 0x5C, 0xD0, 0x77, 0x16, 0x5A],
        ),
        sectors: 0x4000,
    },
    Partition {
        name: "BCPKG2-6-Repair-Sub",
        type_guid: guid(
            0x9586E1A1,
            0x3AA2,
            0x4C90,
            [0x91, 0xB3, 0x2F, 0x4A, 0x51, 0x95, 0xB4, 0xD2],
        ),
        sectors: 0x4000,
    },
    Partition {
        name: "SAFE",
        type_guid: guid(
            0xA44F9F6B,
            0x4ED3,
            0x441F,
            [0xA3, 0x4A, 0x56, 0xAA, 0xA1, 0x36, 0xBC, 0x6A],
        ),
        sectors: 0x2_0000,
    },
    Partition {
        name: "SYSTEM",
        type_guid: guid(
            0xACB0CDF0,
            0x4F72,
            0x432D,
            [0xAA, 0x0D, 0x53, 0x88, 0xC7, 0x33, 0xB2, 0x24],
        ),
        sectors: 0x50_0000,
    },
    Partition {
        name: "USER",
        type_guid: guid(
            0x2B777F63,
            0xE842,
            0x47AF,
            [0x94, 0xC4, 0x25, 0xA7, 0xF1, 0x8B, 0x22, 0x80],
        ),
        sectors: 0x340_0000,
    },
];

/// A console's are random. These only have to be distinct: the table is
/// the emulator's, and says so.
fn unique_guid(index: u8) -> [u8; 16] {
    let mut guid = *b"horizon-emu\0\0\0\0\0";
    guid[15] = index;
    guid
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = crc >> 1 ^ (0xEDB8_8320 & (crc & 1).wrapping_neg());
        }
    }
    !crc
}

fn entries() -> Vec<u8> {
    let mut entries = vec![0; (ENTRY_COUNT * ENTRY_SIZE) as usize];
    let mut first = FIRST_USABLE;
    let slots = entries.as_chunks_mut::<{ ENTRY_SIZE as usize }>().0;
    for (index, (partition, entry)) in PARTITIONS.iter().zip(slots).enumerate() {
        entry[..16].copy_from_slice(&partition.type_guid);
        entry[16..32].copy_from_slice(&unique_guid(index as u8 + 1));
        entry[32..40].copy_from_slice(&first.to_le_bytes());
        entry[40..48].copy_from_slice(&(first + partition.sectors - 1).to_le_bytes());
        let name = entry[56..].as_chunks_mut::<2>().0;
        for (unit, bytes) in partition.name.encode_utf16().zip(name) {
            *bytes = unit.to_le_bytes();
        }
        first += partition.sectors;
    }
    assert!(first - 1 <= LAST_USABLE);
    entries
}

/// One sector holding the table header at `own`, with its copy at `other`
/// and its entries at `entries_at`.
fn header(own: u64, other: u64, entries_at: u64, entries_crc: u32) -> Vec<u8> {
    let mut header = vec![0; SECTOR_SIZE as usize];
    header[..8].copy_from_slice(b"EFI PART");
    header[8..12].copy_from_slice(&0x0001_0000u32.to_le_bytes());
    header[12..16].copy_from_slice(&HEADER_SIZE.to_le_bytes());
    header[24..32].copy_from_slice(&own.to_le_bytes());
    header[32..40].copy_from_slice(&other.to_le_bytes());
    header[40..48].copy_from_slice(&FIRST_USABLE.to_le_bytes());
    header[48..56].copy_from_slice(&LAST_USABLE.to_le_bytes());
    header[56..72].copy_from_slice(&unique_guid(0));
    header[72..80].copy_from_slice(&entries_at.to_le_bytes());
    header[80..84].copy_from_slice(&ENTRY_COUNT.to_le_bytes());
    header[84..88].copy_from_slice(&ENTRY_SIZE.to_le_bytes());
    header[88..92].copy_from_slice(&entries_crc.to_le_bytes());
    let crc = crc32(&header[..HEADER_SIZE as usize]);
    header[16..20].copy_from_slice(&crc.to_le_bytes());
    header
}

/// The one partition of the MBR at sector 0, which keeps tools that do not
/// know GUID partition tables away from the disk.
fn protective_mbr() -> Vec<u8> {
    let mut mbr = vec![0; SECTOR_SIZE as usize];
    let entry = &mut mbr[446..462];
    entry[1..4].copy_from_slice(&[0x00, 0x02, 0x00]);
    entry[4] = 0xEE;
    entry[5..8].copy_from_slice(&[0xFF, 0xFF, 0xFF]);
    entry[8..12].copy_from_slice(&1u32.to_le_bytes());
    entry[12..16].copy_from_slice(&((SECTORS - 1).min(u32::MAX.into()) as u32).to_le_bytes());
    mbr[510..].copy_from_slice(&[0x55, 0xAA]);
    mbr
}

/// Everything the table consists of: first sector and contents of each
/// piece.
fn layout() -> [(u64, Vec<u8>); 5] {
    let entries = entries();
    let crc = crc32(&entries);
    let (last, backup_entries) = (SECTORS - 1, SECTORS - 1 - ENTRY_SECTORS);
    [
        (0, protective_mbr()),
        (1, header(1, last, 2, crc)),
        (2, entries.clone()),
        (backup_entries, entries),
        (last, header(last, 1, backup_entries, crc)),
    ]
}

/// Makes `image` a blank, partitioned user area. The file is sparse: only
/// the partition tables are written.
pub fn format_user_area(image: &File) -> io::Result<()> {
    image.set_len(USER_AREA_SIZE)?;
    for (sector, data) in layout() {
        image.write_all_at(&data, sector * SECTOR_SIZE)?;
    }
    Ok(())
}

/// Bytes of the partition `name` within the user area.
pub fn partition_range(name: &str) -> Option<Range<u64>> {
    let mut first = FIRST_USABLE;
    PARTITIONS.iter().find_map(|partition| {
        let range = first * SECTOR_SIZE..(first + partition.sectors) * SECTOR_SIZE;
        first += partition.sectors;
        (partition.name == name).then_some(range)
    })
}

/// A partition that holds a FAT volume.
struct Volume {
    partition: &'static str,
    /// Storage key ID, see [`bis`].
    key: usize,
    /// Whether the driver journals the volume, see [`fat`].
    fat_safe: bool,
}

const CALIBRATION_FILES: Volume = Volume {
    partition: "PRODINFOF",
    key: bis::KEY_CALIBRATION,
    fat_safe: false,
};
const SAFE: Volume = Volume {
    partition: "SAFE",
    key: bis::KEY_SAFE,
    fat_safe: true,
};
const SYSTEM: Volume = Volume {
    partition: "SYSTEM",
    key: bis::KEY_SYSTEM,
    fat_safe: true,
};
const USER: Volume = Volume {
    partition: "USER",
    key: bis::KEY_USER,
    fat_safe: true,
};
const VOLUMES: [Volume; 4] = [CALIBRATION_FILES, SAFE, SYSTEM, USER];

/// Serial number of the first volume, the others counting up from it. A
/// console's are its clock at the time of formatting; these say whose the
/// volumes are.
const VOLUME_SERIAL: u32 = u32::from_le_bytes(*b"hEmu");

/// Where the content manager keeps built-in system contents, and where it
/// keeps those it is still receiving.
const REGISTERED: &str = "/Contents/registered";
const PLACEHOLDERS: &str = "/Contents/placehld";
/// Where the filesystem process keeps system saves (its string at
/// `0x1df8c0`). Booted on a user partition without it, it made the
/// directory itself.
const SAVES: &str = "/save";
const CONTENT_ID_DIGITS: usize = 32;

#[derive(Debug)]
pub enum NandError {
    /// The image itself cannot be read.
    Image(io::Error),
    /// The image does not start with the table of [`format_user_area`].
    NotPartitioned,
    Key(MissingKey),
    /// The TLS client key cannot be wrapped: what is missing.
    SslKey(&'static str),
    /// The calibration partition does not hold what was expected.
    Calibration(&'static str),
    /// No partition of that name holds a filesystem.
    NoSuchVolume(String),
    Filesystem {
        partition: &'static str,
        error: FatError,
    },
    /// The firmware directory cannot be listed.
    Firmware {
        path: PathBuf,
        error: io::Error,
    },
    /// The firmware directory holds no content archive.
    NoContents(PathBuf),
    /// A content archive whose file name is not its content ID.
    ContentName(PathBuf),
    /// A content archive that could not be read or stored.
    Content {
        path: PathBuf,
        error: FatError,
    },
    /// The record of the installed contents could not be built.
    ContentDatabase(content_db::ContentDbError),
    /// The program content to replace is not there once, or its
    /// replacement could not be made.
    Replacement(String),
}

impl fmt::Display for NandError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Image(error) => write!(f, "cannot read the image: {error}"),
            Self::NotPartitioned => write!(f, "the image is not a partitioned user area"),
            Self::Key(missing) => write!(f, "cannot derive the storage keys: {missing}"),
            Self::SslKey(what) => write!(f, "cannot wrap the TLS client key: {what}"),
            Self::Calibration(what) => write!(f, "{what}"),
            Self::NoSuchVolume(name) => write!(f, "no partition {name} with a filesystem"),
            Self::Filesystem { partition, error } => write!(f, "{partition}: {error}"),
            Self::Firmware { path, error } => write!(f, "{}: {error}", path.display()),
            Self::NoContents(path) => write!(f, "{}: no .nca files", path.display()),
            Self::ContentName(path) => {
                write!(f, "{}: the file name is not a content ID", path.display())
            }
            Self::Content { path, error } => write!(f, "{}: {error}", path.display()),
            Self::ContentDatabase(error) => {
                write!(f, "cannot build the content meta database: {error}")
            }
            Self::Replacement(why) => write!(f, "cannot replace the program: {why}"),
        }
    }
}

impl std::error::Error for NandError {}

/// A partition as the filesystem process sees it: sectors of plain text.
pub struct EncryptedPartition<'a> {
    image: &'a File,
    range: Range<u64>,
    cipher: Xts,
    /// Cipher text on its way to the image.
    scratch: Vec<u8>,
    /// Where writes go instead of the image, when they do: cipher text
    /// by sector of the user area.
    overlay: Option<Overlay>,
}

/// Sectors of the user area, encrypted as stored, laid over its image.
pub type Overlay = HashMap<u64, Box<[u8; SECTOR_SIZE as usize]>>;

impl<'a> EncryptedPartition<'a> {
    fn new(image: &'a File, keys: &BisKeys, volume: &Volume) -> Self {
        Self {
            image,
            range: partition_range(volume.partition).expect("volumes are partitions"),
            cipher: keys.cipher(volume.key),
            scratch: Vec::new(),
            overlay: None,
        }
    }

    fn sectors(&self) -> u64 {
        (self.range.end - self.range.start) / SECTOR_SIZE
    }

    /// Offset into the partition of `length` bytes from `sector`, which
    /// have to lie within it: the neighbours are under other keys.
    fn offset(&self, sector: u64, length: usize) -> io::Result<u64> {
        sector
            .checked_mul(SECTOR_SIZE)
            .filter(|offset| {
                offset
                    .checked_add(length as u64)
                    .is_some_and(|end| end <= self.range.end - self.range.start)
            })
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "beyond the partition"))
    }
}

impl Sectors for EncryptedPartition<'_> {
    fn read(&mut self, sector: u64, data: &mut [u8]) -> io::Result<()> {
        let offset = self.offset(sector, data.len())?;
        self.image.read_exact_at(data, self.range.start + offset)?;
        if let Some(overlay) = &self.overlay {
            let first = (self.range.start + offset) / SECTOR_SIZE;
            for (index, block) in data.chunks_mut(SECTOR_SIZE as usize).enumerate() {
                if let Some(laid) = overlay.get(&(first + index as u64)) {
                    block.copy_from_slice(&laid[..]);
                }
            }
        }
        self.cipher.decrypt(offset, data);
        Ok(())
    }

    fn write(&mut self, sector: u64, data: &[u8]) -> io::Result<()> {
        let offset = self.offset(sector, data.len())?;
        self.scratch.clear();
        self.scratch.extend_from_slice(data);
        self.cipher.encrypt(offset, &mut self.scratch);
        match &mut self.overlay {
            Some(overlay) => {
                let first = (self.range.start + offset) / SECTOR_SIZE;
                for (index, block) in self
                    .scratch
                    .as_chunks::<{ SECTOR_SIZE as usize }>()
                    .0
                    .iter()
                    .enumerate()
                {
                    overlay.insert(first + index as u64, Box::new(*block));
                }
                Ok(())
            }
            None => self
                .image
                .write_all_at(&self.scratch, self.range.start + offset),
        }
    }
}

/// That `image` has the partitions this module assumes.
fn check_partitioned(image: &File) -> Result<(), NandError> {
    let expected = entries();
    let mut found = vec![0; expected.len()];
    match image.read_exact_at(&mut found, 2 * SECTOR_SIZE) {
        Ok(()) if found == expected => Ok(()),
        Err(error) if error.kind() != io::ErrorKind::UnexpectedEof => Err(NandError::Image(error)),
        _ => Err(NandError::NotPartitioned),
    }
}

/// The storage keys of the console `keys` describes, once `image` is known
/// to have the partitions this module assumes.
fn storage_keys(image: &File, keys: &KeySet) -> Result<BisKeys, NandError> {
    check_partitioned(image)?;
    BisKeys::derive(keys).map_err(NandError::Key)
}

/// Makes the four filesystem partitions of the partitioned `image` empty
/// volumes, under the storage keys that follow from `keys`. What they held
/// is lost.
pub fn format_filesystems(image: &File, keys: &KeySet) -> Result<(), NandError> {
    let keys = storage_keys(image, keys)?;
    for (serial, volume) in (VOLUME_SERIAL..).zip(&VOLUMES) {
        let mut partition = EncryptedPartition::new(image, &keys, volume);
        let sectors = partition.sectors();
        fat::format(&mut partition, sectors, volume.fat_safe, serial).map_err(|error| {
            NandError::Filesystem {
                partition: volume.partition,
                error,
            }
        })?;
    }
    Ok(())
}

/// The FAT32 volume of the partition named `partition`, for adding
/// directories and files to before a console first boots from `image`;
/// they are stored once [`Fat32::finish`] returns.
pub fn open_volume<'a>(
    image: &'a File,
    keys: &KeySet,
    partition: &str,
) -> Result<Fat32<EncryptedPartition<'a>>, NandError> {
    let volume = VOLUMES
        .iter()
        .find(|volume| volume.partition == partition)
        .ok_or_else(|| NandError::NoSuchVolume(partition.into()))?;
    let keys = storage_keys(image, keys)?;
    Fat32::open(EncryptedPartition::new(image, &keys, volume)).map_err(|error| {
        NandError::Filesystem {
            partition: volume.partition,
            error,
        }
    })
}

/// What [`install_firmware`] stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Installed {
    pub contents: usize,
    pub bytes: u64,
}

/// The content archives in the directory `firmware` by content ID, which
/// is what such a file is named after: `<id>.nca`, or `<id>.cnmt.nca` for
/// the archives that describe a title.
fn firmware_contents(firmware: &Path) -> Result<Vec<(String, PathBuf)>, NandError> {
    let unlistable = |error| NandError::Firmware {
        path: firmware.into(),
        error,
    };
    let mut contents = Vec::new();
    for entry in std::fs::read_dir(firmware).map_err(unlistable)? {
        let path = entry.map_err(unlistable)?.path();
        let name = path.file_name().and_then(|name| name.to_str());
        let Some(stem) = name.and_then(|name| name.strip_suffix(".nca")) else {
            continue;
        };
        let id = stem.strip_suffix(".cnmt").unwrap_or(stem);
        if id.len() != CONTENT_ID_DIGITS || !id.bytes().all(|digit| digit.is_ascii_hexdigit()) {
            return Err(NandError::ContentName(path));
        }
        contents.push((id.to_ascii_lowercase(), path));
    }
    if contents.is_empty() {
        return Err(NandError::NoContents(firmware.into()));
    }
    contents.sort();
    Ok(contents)
}

/// Stores every content archive of the directory `firmware` as a built-in
/// system content on the formatted `image`, which no console may have
/// booted from yet, and beside them the two system saves through which the
/// console's content manager knows them (see
/// [`content_db::system_saves`]): it keeps that record itself as contents
/// are installed, and has no way to make it from the archives.
pub fn install_firmware(
    image: &File,
    keys: &KeySet,
    firmware: &Path,
) -> Result<Installed, NandError> {
    let contents = firmware_contents(firmware)?;
    let saves = content_db::system_saves(firmware, keys).map_err(NandError::ContentDatabase)?;
    install(image, keys, &contents, &saves)
}

/// Stores `contents` (content ID and file) and system `saves` (save data
/// ID and image) on the system partition.
fn install(
    image: &File,
    keys: &KeySet,
    contents: &[(String, PathBuf)],
    saves: &[(u64, Vec<u8>)],
) -> Result<Installed, NandError> {
    let filesystem = |error| NandError::Filesystem {
        partition: SYSTEM.partition,
        error,
    };
    let mut volume = open_volume(image, keys, SYSTEM.partition)?;
    for directory in [REGISTERED, PLACEHOLDERS, SAVES] {
        volume.create_dir_all(directory).map_err(filesystem)?;
    }
    let mut bytes = 0;
    for (id, path) in contents {
        let failed = |error| NandError::Content {
            path: path.clone(),
            error,
        };
        let mut archive = File::open(path).map_err(|error| failed(error.into()))?;
        let size = archive
            .metadata()
            .map_err(|error| failed(error.into()))?
            .len();
        volume
            .write_file(&format!("{REGISTERED}/{id}.nca"), &mut archive, size)
            .map_err(failed)?;
        bytes += size;
    }
    for (id, save) in saves {
        volume
            .write_file(
                &format!("{SAVES}/{id:016x}"),
                &mut save.as_slice(),
                save.len() as u64,
            )
            .map_err(filesystem)?;
    }
    volume.finish().map_err(filesystem)?;
    Ok(Installed {
        contents: contents.len(),
        bytes,
    })
}

/// What [`replace_program`] lays over the user area.
pub struct ProgramReplacement {
    /// The file of the system partition that holds the program content.
    pub path: String,
    pub original_size: u64,
    pub size: u64,
    /// The sectors that hold the new archive, encrypted as stored.
    pub overlay: Overlay,
}

/// Lays the content archive `build` makes over the program content of
/// `program_id` among the built-in system contents of `image`, in memory:
/// the image does not change. `build` gets the archive it replaces.
///
/// The content manager's database keeps naming the same content ID, so
/// the new archive takes the place of the old one's contents: the file
/// keeps its size and clusters, and only its first sectors change. That
/// leaves the directories and FATs alone, which keeps the recovery file of
/// a system partition a console has mounted valid (see [`fat`]), and
/// nothing reads a content archive past the size its header gives. The
/// archive found is the one file of `/Contents/registered` whose header
/// names a program content (type 0, offset `0x205`) of `program_id`
/// (offset `0x210`).
pub fn replace_program(
    image: &File,
    keys: &KeySet,
    program_id: u64,
    build: impl FnOnce(&nca::Nca) -> Result<Vec<u8>, String>,
) -> Result<ProgramReplacement, NandError> {
    let filesystem = |error| NandError::Filesystem {
        partition: SYSTEM.partition,
        error,
    };
    let mut partition = EncryptedPartition::new(image, &storage_keys(image, keys)?, &SYSTEM);
    partition.overlay = Some(Overlay::new());
    let mut volume = MountedFat32::open(partition).map_err(filesystem)?;
    let mut found = Vec::new();
    for file in volume.list(REGISTERED).map_err(filesystem)? {
        if file.directory || !file.name.ends_with(".nca") {
            continue;
        }
        let path = format!("{REGISTERED}/{}", file.name);
        let header = volume.read(&path, nca::HEADER_SIZE).map_err(filesystem)?;
        let archive = match nca::Nca::open(&header, keys) {
            Ok(archive) => archive,
            Err(nca::NcaError::MissingKey(missing)) => return Err(NandError::Key(missing)),
            Err(_) => continue,
        };
        if archive.content_type() == nca::CONTENT_TYPE_PROGRAM && archive.program_id() == program_id
        {
            found.push((path, u64::from(file.size)));
        }
    }
    let (path, original_size) = match found.len() {
        1 => found.pop().expect("one"),
        0 => {
            return Err(NandError::Replacement(format!(
                "no program content of {program_id:016x} in {REGISTERED}"
            )));
        }
        _ => {
            return Err(NandError::Replacement(format!(
                "{} program contents of {program_id:016x} in {REGISTERED}",
                found.len()
            )));
        }
    };
    let whole = volume
        .read(&path, original_size as usize)
        .map_err(filesystem)?;
    let original = nca::Nca::open(&whole, keys).expect("opened before");
    let archive = build(&original).map_err(NandError::Replacement)?;
    volume.overwrite(&path, &archive).map_err(filesystem)?;
    let overlay = volume
        .into_storage()
        .overlay
        .expect("the partition was given an overlay");
    Ok(ProgramReplacement {
        path,
        original_size,
        size: archive.len() as u64,
        overlay,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u64_at(data: &[u8], offset: usize) -> u64 {
        u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap())
    }

    #[test]
    fn checksums_and_cross_references_of_both_tables_hold() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        let [
            _,
            (_, primary),
            (_, entries),
            (backup_entries_at, backup_entries),
            (last, backup),
        ] = layout();
        assert_eq!(entries, backup_entries);
        for (header, own, other, entries_at) in [
            (&primary, 1, last, 2),
            (&backup, last, 1, backup_entries_at),
        ] {
            let mut unsigned = header[..HEADER_SIZE as usize].to_vec();
            unsigned[16..20].fill(0);
            assert_eq!(header[16..20], crc32(&unsigned).to_le_bytes());
            assert_eq!(header[88..92], crc32(&entries).to_le_bytes());
            assert_eq!(
                (u64_at(header, 24), u64_at(header, 32), u64_at(header, 72)),
                (own, other, entries_at)
            );
        }
        assert!(backup_entries_at > LAST_USABLE);
    }

    #[test]
    fn partitions_sit_at_the_offsets_of_the_console_layout() {
        let [_, _, (_, entries), ..] = layout();
        let range = |index: usize| {
            let entry = &entries[index * ENTRY_SIZE as usize..];
            (
                u64_at(entry, 32) * SECTOR_SIZE,
                (u64_at(entry, 40) + 1) * SECTOR_SIZE,
            )
        };
        assert_eq!(range(0), (0x4400, 0x40_0000), "PRODINFO");
        assert_eq!(range(8), (0x380_0000, 0x780_0000), "SAFE");
        assert_eq!(range(9), (0x780_0000, 0xA780_0000), "SYSTEM");
        assert_eq!(range(10), (0xA780_0000, 0x7_2780_0000), "USER");
        assert_eq!(
            entries[10 * 128 + 56..10 * 128 + 66],
            [b'U', 0, b'S', 0, b'E', 0, b'R', 0, 0, 0]
        );
        assert_eq!(entries[11 * 128..12 * 128], [0; 128], "eleven entries");
        assert_eq!(partition_range("SYSTEM"), Some(0x780_0000..0xA780_0000));
        assert_eq!(partition_range("system"), None);
    }

    /// A directory in the host's temporary directory that goes away with
    /// the test.
    struct Scratch(PathBuf);

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn firmware_lands_where_the_content_manager_looks_and_the_image_stays_sparse() {
        use std::os::unix::fs::MetadataExt;

        // The key file of a console that does not exist.
        let mut text = String::new();
        for (index, name) in [
            "device_key",
            "retail_specific_aes_key_source",
            "aes_kek_generation_source",
            "aes_key_generation_source",
            "bis_kek_source",
        ]
        .into_iter()
        .enumerate()
        {
            text += &format!("{name} = {:032x}\n", 0x0123_4567 * (index as u128 + 1));
        }
        for index in 0..3 {
            text += &format!("bis_key_source_{index:02x} = {:064x}\n", index + 0x89AB);
        }
        let keys = KeySet::parse(&text).unwrap();

        let scratch = Scratch(
            std::env::temp_dir().join(format!("horizon-emu-nand-test-{}", std::process::id())),
        );
        let firmware = scratch.0.join("firmware");
        std::fs::create_dir_all(&firmware).unwrap();
        let program: Vec<u8> = (0..40_000u32).map(|index| (index * 13) as u8).collect();
        let (program_id, meta_id) = ("0f".repeat(16), "A1".repeat(16));
        std::fs::write(firmware.join(format!("{program_id}.nca")), &program).unwrap();
        std::fs::write(firmware.join(format!("{meta_id}.cnmt.nca")), b"meta").unwrap();
        std::fs::write(firmware.join("notes.txt"), b"not a content").unwrap();

        let image = File::options()
            .read(true)
            .write(true)
            .create_new(true)
            .open(scratch.0.join("emmc.img"))
            .unwrap();
        let unpartitioned = format_filesystems(&image, &keys);
        assert!(matches!(unpartitioned, Err(NandError::NotPartitioned)));
        format_user_area(&image).unwrap();
        format_filesystems(&image, &keys).unwrap();
        let contents = firmware_contents(&firmware).unwrap();
        let installed = install(
            &image,
            &keys,
            &contents,
            &[(0x8000_0000_0000_0120, vec![7; 5])],
        );
        let installed = installed.unwrap();
        assert_eq!(
            installed,
            Installed {
                contents: 2,
                bytes: 40_004
            }
        );

        let storage_keys = BisKeys::derive(&keys).unwrap();
        let system = EncryptedPartition::new(&image, &storage_keys, &SYSTEM);
        let mut reader = fat::reader::Reader::open(system);
        assert_eq!(reader.names("/"), ["Contents", "save"]);
        assert_eq!(reader.names("/save"), ["8000000000000120"]);
        assert_eq!(reader.names("/Contents"), ["registered", "placehld"]);
        // Meta archives go by their content ID like the others, in the
        // lower case the content manager prints IDs in.
        let stored = [
            format!("{program_id}.nca"),
            format!("{}.nca", "a1".repeat(16)),
        ];
        assert_eq!(reader.names("/Contents/registered"), stored);
        assert_eq!(
            reader.file(&format!("/Contents/registered/{}", stored[0])),
            program
        );
        assert_eq!(
            reader.file(&format!("/Contents/registered/{}", stored[1])),
            b"meta"
        );
        let user = EncryptedPartition::new(&image, &storage_keys, &USER);
        assert_eq!(fat::reader::Reader::open(user).names("/"), [""; 0]);

        // 29 GiB of image, of which the boot sectors, FATs, directories
        // and files exist: the largest part 13 MiB of FATs for the user
        // partition.
        let metadata = image.metadata().unwrap();
        assert_eq!(metadata.len(), USER_AREA_SIZE);
        assert!(metadata.blocks() * 512 < 32 << 20);

        std::fs::write(firmware.join("update.nca"), b"").unwrap();
        let misnamed = firmware_contents(&firmware);
        assert!(
            matches!(misnamed, Err(NandError::ContentName(path)) if path.ends_with("update.nca"))
        );
    }
}
