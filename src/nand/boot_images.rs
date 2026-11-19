//! The boot images of a console: the BCTs and package1 in the eMMC's boot
//! partitions and package2 in the raw `BCPKG2-*` partitions of the user
//! area, installed from a firmware's boot image packages the way the system
//! updater leaves them.
//!
//! A firmware carries them as two data archives: `BootImagePackage`
//! (0100000000000819, normal) and `BootImagePackageSafe` (010000000000081A,
//! safe mode), each a RomFS with `nx/bct`, `nx/package1` and `nx/package2`
//! for the T210 (Erista) and `a/bct`, `a/package1` for the T214 (Mariko);
//! this console is an Erista and takes `nx`. (010000000000081B/081C are the
//! same for consoles with the exFAT driver installed, which this one has
//! not.)
//!
//! Boot partition 0 (switchbrew "Flash Filesystem"):
//!
//! | offset   | size    | contents |
//! |----------|---------|----------|
//! | 0x000000 | 0x4000  | normal BCT |
//! | 0x004000 | 0x4000  | safe mode BCT |
//! | 0x008000 | 0x4000  | normal BCT, backup |
//! | 0x00C000 | 0x4000  | safe mode BCT, backup |
//! | 0x010000 | 0xEC000 | 59 more BCT slots, empty |
//! | 0x0FC000 | 0x4000  | system update control, zero: no update pending |
//! | 0x100000 | 0x40000 | normal package1 |
//! | 0x140000 | 0x40000 | normal package1, backup |
//! | 0x180000 | 0x4000  | keyblobs (`EKS`), console-unique |
//! | 0x184000 | 0x200   | NAND patrol record of the filesystem process |
//!
//! Boot partition 1: safe mode package1 at 0x0, its backup at 0x40000, then
//! 0x80000 bytes reserved. Each `BCPKG2-*` partition: 0x4000 bytes of boot
//! configuration (empty on retail consoles), then package2; `-1-Normal-Main`
//! and `-2-Normal-Sub` take the normal one, `-3-SafeMode-Main` and
//! `-4-SafeMode-Sub` the safe mode one. `-5`/`-6-Repair` are written at the
//! factory only and stay erased here.
//!
//! The BCT of a package is complete as the firmware ships it: the boot
//! data version (0x530, `0x210001`), block and page size (0x534/0x538,
//! 16 KiB and 512 bytes), two bootloaders (0x232C) whose entries at 0x2330
//! and 0x245C (0x12C bytes each) give version, start block, start page,
//! length, load address, entry point and attribute, each with Nintendo's
//! RSA-PSS signature, and the RSA modulus (0x210) the signature over
//! 0x510 onward checks with. The entries point at the package1 slots above,
//! the boot ROM counting the two boot partitions as one: the normal BCT's
//! blocks 0x40 and 0x50 (0x100000 and 0x140000 in boot partition 0), the
//! safe mode BCT's blocks 0x100 and 0x110 (0x0 and 0x40000 in boot
//! partition 1). [`BootImagePackage::check`] holds the package to that, so
//! what is written is where its own BCT says.
//!
//! What the system updater adds is console-unique: it copies the keyblob
//! of index `bootloader version - 1` from this console's `EKS` area into
//! the BCT's customer data at 0x450, outside the signed part (switchbrew
//! "BCT", "Flash Filesystem"). \[assumption\] Nothing here can make that:
//! the factory-encrypted keyblobs (with their per-console counters) are not
//! in the key file, which has only their decrypted contents. The customer
//! data therefore stays as the template has it, zero, and so does the
//! `EKS` area; whether the bootloader of firmware 23.5.0 (version 0x17)
//! reads them at all is for when it runs. The bad block table (0x0) is the
//! template's too, none on an eMMC.

use std::{
    fmt,
    fs::File,
    io::{self, Read},
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
};

use super::{
    NandError, check_partitioned, firmware_contents,
    nca::{self, Nca, NcaError},
    partition_range,
    romfs::{self, RomFsError},
};
use crate::keys::KeySet;

/// Each boot partition: 4 MiB.
pub const BOOT_PARTITION_SIZE: u64 = 0x40_0000;

/// `BootImagePackage`.
pub const NORMAL_TITLE: u64 = 0x0100_0000_0000_0819;
/// `BootImagePackageSafe`.
pub const SAFE_TITLE: u64 = 0x0100_0000_0000_081A;

/// The SoC's directory in a boot image package.
const ERISTA: &str = "nx";

/// An Erista BCT.
pub const BCT_SIZE: usize = 0x2800;
/// Each BCT slot of boot partition 0.
const BCT_SLOT: u64 = 0x4000;
/// Each package1 slot.
pub const PACKAGE1_SLOT: u64 = 0x4_0000;
/// Where package2 starts in a `BCPKG2-*` partition, after the boot
/// configuration.
pub const PACKAGE2_OFFSET: u64 = 0x4000;

pub const BCT_BOOT_DATA_VERSION: usize = 0x530;
pub const BCT_BLOCK_SIZE_LOG2: usize = 0x534;
pub const BCT_PAGE_SIZE_LOG2: usize = 0x538;
pub const BCT_BOOTLOADERS_USED: usize = 0x232C;
pub const BCT_BOOTLOADERS: usize = 0x2330;
pub const BCT_BOOTLOADER_SIZE: usize = 0x12C;
/// The customer data's keyblob, left empty (see the module).
pub const BCT_KEYBLOB: usize = 0x450;
/// `BOOTDATA_VERSION_T210`.
pub const BOOT_DATA_VERSION_T210: u32 = 0x21_0001;

/// The two packages and where the updater puts each.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Normal,
    Safe,
}

impl Kind {
    pub fn title(self) -> u64 {
        match self {
            Self::Normal => NORMAL_TITLE,
            Self::Safe => SAFE_TITLE,
        }
    }

    /// The BCT's slots in boot partition 0, main then backup.
    pub fn bct_offsets(self) -> [u64; 2] {
        match self {
            Self::Normal => [0, 2 * BCT_SLOT],
            Self::Safe => [BCT_SLOT, 3 * BCT_SLOT],
        }
    }

    /// The package1 slots, main then backup, in the boot partitions taken
    /// as one (boot partition 1 starting at [`BOOT_PARTITION_SIZE`]): what
    /// the BCT's bootloader entries 0 and 1 point at.
    pub fn package1_offsets(self) -> [u64; 2] {
        match self {
            Self::Normal => [0x10_0000, 0x14_0000],
            Self::Safe => [BOOT_PARTITION_SIZE, BOOT_PARTITION_SIZE + PACKAGE1_SLOT],
        }
    }

    /// The user area partitions holding its package2, main then backup.
    pub fn bcpkg2_partitions(self) -> [&'static str; 2] {
        match self {
            Self::Normal => ["BCPKG2-1-Normal-Main", "BCPKG2-2-Normal-Sub"],
            Self::Safe => ["BCPKG2-3-SafeMode-Main", "BCPKG2-4-SafeMode-Sub"],
        }
    }
}

#[derive(Debug)]
pub enum BootImageError {
    /// The firmware directory cannot be listed, or the user area image is
    /// not partitioned.
    Nand(NandError),
    /// A firmware file cannot be read.
    Firmware { path: PathBuf, error: io::Error },
    /// An archive that does not open or whose RomFS does not verify.
    Nca { path: PathBuf, error: NcaError },
    /// The firmware has no data archive of this title.
    NoPackage(u64),
    /// The package lacks a file.
    RomFs { title: u64, error: RomFsError },
    /// The package is not laid out as this module knows: what.
    Layout { title: u64, what: String },
    /// An image file cannot be written.
    Image(io::Error),
    /// A boot partition image is larger than the partition.
    BootPartitionSize(u64),
}

impl fmt::Display for BootImageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Nand(error) => error.fmt(f),
            Self::Firmware { path, error } => write!(f, "{}: {error}", path.display()),
            Self::Nca { path, error } => write!(f, "{}: {error}", path.display()),
            Self::NoPackage(title) => write!(f, "the firmware has no data archive of {title:016X}"),
            Self::RomFs { title, error } => write!(f, "{title:016X}: {error}"),
            Self::Layout { title, what } => write!(f, "{title:016X}: {what}"),
            Self::Image(error) => write!(f, "cannot write the image: {error}"),
            Self::BootPartitionSize(size) => write!(
                f,
                "boot partition image is {size} bytes; the partition holds {BOOT_PARTITION_SIZE}"
            ),
        }
    }
}

impl std::error::Error for BootImageError {}

impl From<NandError> for BootImageError {
    fn from(error: NandError) -> Self {
        Self::Nand(error)
    }
}

fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

/// A bootloader entry of a BCT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bootloader {
    pub version: u32,
    pub start_block: u32,
    pub start_page: u32,
    pub length: u32,
    pub load_address: u32,
    pub entry_point: u32,
    pub attribute: u32,
}

impl Bootloader {
    /// Entry `index` of the Erista BCT `bct`, which is [`BCT_SIZE`] long.
    pub fn of(bct: &[u8], index: usize) -> Self {
        assert!(index < 4 && bct.len() >= BCT_SIZE);
        let entry = &bct[BCT_BOOTLOADERS + index * BCT_BOOTLOADER_SIZE..];
        let word = |slot: usize| u32_at(entry, slot * 4);
        Self {
            version: word(0),
            start_block: word(1),
            start_page: word(2),
            length: word(3),
            load_address: word(4),
            entry_point: word(5),
            attribute: word(6),
        }
    }

    /// Where it starts in the boot partitions taken as one, with the block
    /// and page sizes of `bct`.
    pub fn offset(&self, bct: &[u8]) -> u64 {
        (u64::from(self.start_block) << u32_at(bct, BCT_BLOCK_SIZE_LOG2))
            + (u64::from(self.start_page) << u32_at(bct, BCT_PAGE_SIZE_LOG2))
    }
}

/// One boot image package's files for this SoC.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BootImagePackage {
    pub kind: Kind,
    pub bct: Vec<u8>,
    pub package1: Vec<u8>,
    pub package2: Vec<u8>,
}

impl BootImagePackage {
    /// The files of the package's RomFS, once [`check`](Self::check)ed.
    pub fn from_romfs(kind: Kind, image: &[u8]) -> Result<Self, BootImageError> {
        let file = |name: &str| {
            romfs::file(image, &format!("{ERISTA}/{name}"))
                .map(<[u8]>::to_vec)
                .map_err(|error| BootImageError::RomFs {
                    title: kind.title(),
                    error,
                })
        };
        let package = Self {
            kind,
            bct: file("bct")?,
            package1: file("package1")?,
            package2: file("package2")?,
        };
        package.check()?;
        Ok(package)
    }

    /// That the files are what the layout of the module expects: an Erista
    /// BCT whose two bootloaders are this package1 at the slots it goes to,
    /// a package1 with its plain header, a package2 that fits.
    pub fn check(&self) -> Result<(), BootImageError> {
        let wrong = |what: String| {
            Err(BootImageError::Layout {
                title: self.kind.title(),
                what,
            })
        };
        let bct = &self.bct;
        if bct.len() != BCT_SIZE {
            return wrong(format!(
                "the BCT is {:#x} bytes, not {BCT_SIZE:#x}",
                bct.len()
            ));
        }
        let version = u32_at(bct, BCT_BOOT_DATA_VERSION);
        if version != BOOT_DATA_VERSION_T210 {
            return wrong(format!("BCT boot data version {version:#x} is not T210's"));
        }
        let used = u32_at(bct, BCT_BOOTLOADERS_USED);
        if used != 2 {
            return wrong(format!("the BCT has {used} bootloaders, not 2"));
        }
        if self.package1.len() as u64 > PACKAGE1_SLOT {
            return wrong(format!("package1 is {:#x} bytes", self.package1.len()));
        }
        for (index, expected) in self.kind.package1_offsets().into_iter().enumerate() {
            let bootloader = Bootloader::of(bct, index);
            let offset = bootloader.offset(bct);
            if offset != expected {
                return wrong(format!(
                    "BCT bootloader {index} is at {offset:#x}, the package1 slot at {expected:#x}"
                ));
            }
            if bootloader.length as usize != self.package1.len() {
                return wrong(format!(
                    "BCT bootloader {index} is {:#x} bytes, package1 {:#x}",
                    bootloader.length,
                    self.package1.len()
                ));
            }
        }
        if package1_build_date(&self.package1).is_none() {
            return wrong("package1 has no plain header with its build date".into());
        }
        let room = partition_size(self.kind.bcpkg2_partitions()[0]) - PACKAGE2_OFFSET;
        if self.package2.len() as u64 > room {
            return wrong(format!("package2 is {:#x} bytes", self.package2.len()));
        }
        Ok(())
    }
}

fn partition_size(name: &str) -> u64 {
    let range = partition_range(name).expect("a partition of the table");
    range.end - range.start
}

/// The build date (`yyyyMMddHHmmss`) in the plain header of an Erista
/// package1 at 0x10, after the four hashes and the build ID (switchbrew
/// "Package1").
pub fn package1_build_date(package1: &[u8]) -> Option<&str> {
    let date = package1.get(0x10..0x1E)?;
    date.iter()
        .all(u8::is_ascii_digit)
        .then(|| std::str::from_utf8(date).unwrap())
}

/// Both packages of a firmware.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BootImages {
    pub normal: BootImagePackage,
    pub safe: BootImagePackage,
}

/// A piece of a boot partition: which (0 or 1), at what offset, what it
/// holds, and the slot it fills (the rest of the slot is zero).
#[derive(Debug, PartialEq, Eq)]
pub struct Piece<'a> {
    pub partition: usize,
    pub offset: u64,
    pub data: &'a [u8],
    pub slot: u64,
}

impl BootImages {
    /// The packages among the content archives of the directory `firmware`
    /// (as [`super::install_firmware`] takes it): the data archives of
    /// [`NORMAL_TITLE`] and [`SAFE_TITLE`], their RomFS verified.
    pub fn load(firmware: &Path, keys: &KeySet) -> Result<Self, BootImageError> {
        let mut found = [None, None];
        for (_, path) in firmware_contents(firmware)? {
            let unreadable = |error| BootImageError::Firmware {
                path: path.clone(),
                error,
            };
            let invalid = |error| BootImageError::Nca {
                path: path.clone(),
                error,
            };
            let mut file = File::open(&path).map_err(unreadable)?;
            let mut data = vec![0; nca::HEADER_SIZE];
            file.read_exact(&mut data).map_err(unreadable)?;
            let archive = Nca::open(&data, keys).map_err(invalid)?;
            if archive.content_type() != nca::CONTENT_TYPE_DATA {
                continue;
            }
            let Some(index) = [NORMAL_TITLE, SAFE_TITLE]
                .iter()
                .position(|&title| title == archive.program_id())
            else {
                continue;
            };
            file.read_to_end(&mut data).map_err(unreadable)?;
            let romfs = Nca::open(&data, keys)
                .and_then(|archive| archive.romfs_section(0))
                .map_err(invalid)?;
            let kind = [Kind::Normal, Kind::Safe][index];
            found[index] = Some(BootImagePackage::from_romfs(kind, &romfs)?);
        }
        let [Some(normal), Some(safe)] = found else {
            let missing = if found[0].is_none() {
                NORMAL_TITLE
            } else {
                SAFE_TITLE
            };
            return Err(BootImageError::NoPackage(missing));
        };
        Ok(Self { normal, safe })
    }

    /// What goes into the boot partitions, in offset order per partition.
    pub fn boot_partition_pieces(&self) -> Vec<Piece<'_>> {
        let mut pieces = Vec::new();
        for package in [&self.normal, &self.safe] {
            for offset in package.kind.bct_offsets() {
                pieces.push(Piece {
                    partition: 0,
                    offset,
                    data: &package.bct,
                    slot: BCT_SLOT,
                });
            }
            for offset in package.kind.package1_offsets() {
                pieces.push(Piece {
                    partition: (offset / BOOT_PARTITION_SIZE) as usize,
                    offset: offset % BOOT_PARTITION_SIZE,
                    data: &package.package1,
                    slot: PACKAGE1_SLOT,
                });
            }
        }
        pieces.sort_by_key(|piece| (piece.partition, piece.offset));
        pieces
    }

    /// Makes `boot` the two boot partitions, [`BOOT_PARTITION_SIZE`] each
    /// (sparse files): the BCTs and package1 copies fill their slots, and
    /// everything else keeps what it held, zero in a new file.
    pub fn write_boot_partitions(&self, boot: [&File; 2]) -> Result<(), BootImageError> {
        for image in boot {
            let size = image.metadata().map_err(BootImageError::Image)?.len();
            if size > BOOT_PARTITION_SIZE {
                return Err(BootImageError::BootPartitionSize(size));
            }
            image
                .set_len(BOOT_PARTITION_SIZE)
                .map_err(BootImageError::Image)?;
        }
        for piece in self.boot_partition_pieces() {
            let mut slot = piece.data.to_vec();
            slot.resize(piece.slot as usize, 0);
            boot[piece.partition]
                .write_all_at(&slot, piece.offset)
                .map_err(BootImageError::Image)?;
        }
        Ok(())
    }

    /// Writes each package2 after the boot configuration of its two
    /// `BCPKG2-*` partitions of the partitioned user area `image`. The boot
    /// configuration and what follows package2 keep what they held; the
    /// partitions are not encrypted.
    pub fn write_package2(&self, image: &File) -> Result<(), BootImageError> {
        check_partitioned(image)?;
        for package in [&self.normal, &self.safe] {
            for name in package.kind.bcpkg2_partitions() {
                let range = partition_range(name).expect("a partition of the table");
                image
                    .write_all_at(&package.package2, range.start + PACKAGE2_OFFSET)
                    .map_err(BootImageError::Image)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nand::{crypto, format_user_area};

    const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../");
    const NORMAL_ROMFS: &str =
        "fw_dump/0100000000000819/da6ae627428f5fe36e26aafa833515a8/section0_romfs/nx/";
    const SAFE_ROMFS: &str =
        "fw_dump/010000000000081a/61685e9d7698430619001f8e1c3b06f4/section0_romfs/nx/";

    /// Firmware 23.5.0's packages as the reference extraction has them.
    fn extracted(kind: Kind, directory: &str) -> BootImagePackage {
        let file = |name: &str| {
            std::fs::read(format!("{ROOT}{directory}{name}"))
                .expect("required boot image package extraction in fw_dump")
        };
        BootImagePackage {
            kind,
            bct: file("bct"),
            package1: file("package1"),
            package2: file("package2"),
        }
    }

    fn images() -> BootImages {
        BootImages {
            normal: extracted(Kind::Normal, NORMAL_ROMFS),
            safe: extracted(Kind::Safe, SAFE_ROMFS),
        }
    }

    /// A file in the host's temporary directory that goes away with the
    /// test.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(name: &str) -> (Self, File) {
            let path = std::env::temp_dir().join(format!(
                "horizon-emu-boot-images-{name}-{}",
                std::process::id()
            ));
            let _ = std::fs::remove_file(&path);
            let file = File::options()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&path)
                .unwrap();
            (Self(path), file)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn read(file: &File, offset: u64, size: usize) -> Vec<u8> {
        let mut bytes = vec![0; size];
        file.read_exact_at(&mut bytes, offset).unwrap();
        bytes
    }

    #[test]
    fn firmware_packages_match_the_layout() {
        let images = images();
        images.normal.check().unwrap();
        images.safe.check().unwrap();
        let normal = Bootloader::of(&images.normal.bct, 0);
        assert_eq!(
            (
                normal.version,
                normal.start_block,
                normal.load_address,
                normal.entry_point
            ),
            (0x17, 0x40, 0x4001_0000, 0x4001_0040)
        );
        assert_eq!(Bootloader::of(&images.safe.bct, 1).start_block, 0x110);
        assert_eq!(
            package1_build_date(&images.normal.package1),
            Some("20260930202100")
        );

        let mut moved = images.normal.clone();
        moved.bct[BCT_BOOTLOADERS + 4] = 0x41;
        assert!(matches!(
            moved.check(),
            Err(BootImageError::Layout {
                title: NORMAL_TITLE,
                ..
            })
        ));
        let mut longer = images.safe.clone();
        longer.package1.push(0);
        assert!(longer.check().is_err());
    }

    #[test]
    fn boot_partitions_read_back_as_the_boot_rom_finds_them() {
        let images = images();
        let (_boot0_file, boot0) = Scratch::new("boot0");
        let (_boot1_file, boot1) = Scratch::new("boot1");
        images.write_boot_partitions([&boot0, &boot1]).unwrap();
        for image in [&boot0, &boot1] {
            assert_eq!(image.metadata().unwrap().len(), BOOT_PARTITION_SIZE);
        }
        let boot = [&boot0, &boot1];

        for package in [&images.normal, &images.safe] {
            for offset in package.kind.bct_offsets() {
                let bct = read(&boot0, offset, BCT_SLOT as usize);
                assert_eq!(bct[..BCT_SIZE], package.bct[..]);
                assert!(bct[BCT_SIZE..].iter().all(|&byte| byte == 0));
                assert_eq!(u32_at(&bct, BCT_BOOT_DATA_VERSION), BOOT_DATA_VERSION_T210);
                assert!(
                    bct[BCT_KEYBLOB..BCT_KEYBLOB + 0xB0]
                        .iter()
                        .all(|&byte| byte == 0)
                );
            }
            // The boot ROM's way: each bootloader entry of the BCT, over
            // the two partitions as one.
            let bct = read(&boot0, package.kind.bct_offsets()[0], BCT_SIZE);
            for index in 0..2 {
                let bootloader = Bootloader::of(&bct, index);
                let offset = bootloader.offset(&bct);
                let partition = boot[(offset / BOOT_PARTITION_SIZE) as usize];
                let loaded = read(
                    partition,
                    offset % BOOT_PARTITION_SIZE,
                    bootloader.length as usize,
                );
                assert_eq!(crypto::sha256(&loaded), crypto::sha256(&package.package1));
                assert_eq!(package1_build_date(&loaded), Some("20260930202100"));
            }
        }
        // The safe mode bootloaders sit in boot partition 1.
        assert_eq!(read(&boot1, 0, 0x20), images.safe.package1[..0x20]);
        assert_eq!(
            read(&boot1, PACKAGE1_SLOT, 0x20),
            images.safe.package1[..0x20]
        );
        // Unused BCT slots, the update control area and the keyblobs stay
        // erased.
        assert!(
            read(&boot0, 0x1_0000, 0xF0000)
                .iter()
                .all(|&byte| byte == 0)
        );
        assert!(
            read(&boot0, 0x18_0000, 0x4200)
                .iter()
                .all(|&byte| byte == 0)
        );
        assert!(
            read(&boot1, 0x8_0000, 0x8_0000)
                .iter()
                .all(|&byte| byte == 0)
        );

        let (_large_file, large) = Scratch::new("large");
        large.set_len(BOOT_PARTITION_SIZE + 1).unwrap();
        assert!(matches!(
            images.write_boot_partitions([&boot0, &large]),
            Err(BootImageError::BootPartitionSize(size)) if size == BOOT_PARTITION_SIZE + 1
        ));
    }

    #[test]
    fn bcpkg2_partitions_hold_package2_after_the_boot_configuration() {
        let images = images();
        let (_user_file, user) = Scratch::new("user");
        assert!(matches!(
            images.write_package2(&user),
            Err(BootImageError::Nand(NandError::NotPartitioned))
        ));
        format_user_area(&user).unwrap();
        images.write_package2(&user).unwrap();
        for (package, name) in [
            (&images.normal, "BCPKG2-1-Normal-Main"),
            (&images.normal, "BCPKG2-2-Normal-Sub"),
            (&images.safe, "BCPKG2-3-SafeMode-Main"),
            (&images.safe, "BCPKG2-4-SafeMode-Sub"),
        ] {
            let range = partition_range(name).unwrap();
            assert_eq!(range.end - range.start, 0x80_0000);
            assert!(
                read(&user, range.start, PACKAGE2_OFFSET as usize)
                    .iter()
                    .all(|&byte| byte == 0)
            );
            let stored = read(&user, range.start + PACKAGE2_OFFSET, package.package2.len());
            assert_eq!(crypto::sha256(&stored), crypto::sha256(&package.package2));
        }
        assert_eq!(
            partition_range("BCPKG2-1-Normal-Main").unwrap().start,
            0x80_0000
        );
        let repair = partition_range("BCPKG2-5-Repair-Main").unwrap();
        assert!(
            read(&user, repair.start, 0x10_0000)
                .iter()
                .all(|&byte| byte == 0)
        );
    }

    /// The packages as the firmware's archives hold them, through the NCA
    /// and RomFS readers, are the reference extraction's.
    #[test]
    fn firmware_archives_yield_the_extracted_packages() {
        let keys =
            KeySet::load(Path::new(&format!("{ROOT}prod_newer.keys"))).expect("required key file");
        let loaded = BootImages::load(Path::new(&format!("{ROOT}fw")), &keys).unwrap();
        let expected = images();
        for (loaded, expected) in [
            (&loaded.normal, &expected.normal),
            (&loaded.safe, &expected.safe),
        ] {
            assert_eq!(loaded.kind, expected.kind);
            for (found, wanted) in [
                (&loaded.bct, &expected.bct),
                (&loaded.package1, &expected.package1),
                (&loaded.package2, &expected.package2),
            ] {
                assert_eq!(crypto::sha256(found), crypto::sha256(wanted));
            }
        }
    }
}
