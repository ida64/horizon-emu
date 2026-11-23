//! The system version installed on a console's eMMC, read the way the
//! system finds it: from system data 0100000000000809 (SystemVersion).
//!
//! The content manager's database of the built-in system contents (see
//! [`content_db`]) names the data content of that title; the content is
//! `SYSTEM:/Contents/registered/<content ID>.nca`, whose RomFS holds the
//! one file `file`. That file is 0x100 bytes (switchbrew "System Version
//! Title"): major, minor and micro version at 0x0 to 0x2, revision major
//! and minor at 0x4 and 0x5, the platform string at 0x8 (0x20 bytes), a
//! version hash at 0x28 (0x40), the display version at 0x68 (0x18) and
//! the display title at 0x80 (0x80), strings NUL-padded. Firmware
//! 23.5.0's (`fw_dump/0100000000000809`) reads 23.5.0, revision 1.0,
//! platform `NX`, display version `23.5.0`.
//!
//! Everything here only reads, through [`MountedFat32`], so it works on an
//! image a console has booted from as well as on one fresh from
//! [`install_firmware`](super::install_firmware). On the user areas of
//! the emulator's machines running firmware 23.5.0 (`wk/emu`, its
//! `snap_driver` snapshot, `.scratch/hekate`) it finds 23.5.0, revision
//! 1.0.

use std::fmt;
use std::fs::File;

use super::{
    EncryptedPartition, NandError, REGISTERED, SAVES, SYSTEM,
    cnmt::{self, CONTENT_INFO_SIZE, ContentInfo},
    content_db,
    fat::{FatError, MountedFat32},
    nca::{Nca, NcaError},
    romfs::{self, RomFsError},
    save::{self, SaveError},
    storage_keys,
};
use crate::keys::KeySet;

/// The system data title that holds the system version.
pub const PROGRAM_ID: u64 = 0x0100_0000_0000_0809;
/// Its file in the RomFS of its data content.
const FILE: &str = "file";
const FILE_SIZE: usize = 0x100;
/// Database value header: extended header size, content count, content
/// meta count, attributes, platform.
const VALUE_HEADER_SIZE: usize = 8;
const KEY_SIZE: usize = 0x10;

/// What [`installed_system_version`] found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SystemVersion {
    pub major: u8,
    pub minor: u8,
    pub micro: u8,
    pub revision_major: u8,
    pub revision_minor: u8,
    pub platform: String,
    pub display_version: String,
}

#[derive(Debug)]
pub enum SystemVersionError {
    /// NCM's database save cannot be read.
    Save(SaveError),
    /// The save holds no database file.
    NoDatabase,
    /// The database file is not a key-value archive: what is wrong.
    Database(&'static str),
    /// The database has no record of the SystemVersion title.
    NotInstalled,
    /// The record names no data content.
    NoDataContent,
    /// The data content cannot be read.
    Nca(NcaError),
    RomFs(RomFsError),
    /// The version file is not laid out as expected: what is wrong.
    Malformed(&'static str),
}

impl fmt::Display for SystemVersionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Save(error) => write!(f, "NCM's database save: {error}"),
            Self::NoDatabase => write!(
                f,
                "NCM's database save holds no {}/{}",
                content_db::DIRECTORY,
                content_db::FILE_NAME
            ),
            Self::Database(what) => write!(f, "NCM's database: {what}"),
            Self::NotInstalled => write!(f, "no record of {PROGRAM_ID:016x} in NCM's database"),
            Self::NoDataContent => {
                write!(f, "the record of {PROGRAM_ID:016x} names no data content")
            }
            Self::Nca(error) => write!(f, "the data content of {PROGRAM_ID:016x}: {error}"),
            Self::RomFs(error) => write!(f, "the data content of {PROGRAM_ID:016x}: {error}"),
            Self::Malformed(what) => write!(f, "the system version file: {what}"),
        }
    }
}

impl std::error::Error for SystemVersionError {}

impl From<SystemVersionError> for NandError {
    fn from(error: SystemVersionError) -> Self {
        match error {
            SystemVersionError::Save(SaveError::MissingKey(missing))
            | SystemVersionError::Nca(NcaError::MissingKey(missing)) => Self::Key(missing),
            error => Self::SystemVersion(error),
        }
    }
}

impl SystemVersion {
    /// The version the 0x100-byte version `file` holds.
    pub fn parse(file: &[u8]) -> Result<Self, SystemVersionError> {
        let file: &[u8; FILE_SIZE] = file
            .get(..FILE_SIZE)
            .ok_or(SystemVersionError::Malformed("shorter than 0x100 bytes"))?
            .try_into()
            .unwrap();
        let string = |at: usize, size: usize| {
            let field = &file[at..at + size];
            let length = field.iter().position(|&byte| byte == 0).unwrap_or(size);
            String::from_utf8(field[..length].to_vec())
                .map_err(|_| SystemVersionError::Malformed("a string that is not UTF-8"))
        };
        Ok(Self {
            major: file[0],
            minor: file[1],
            micro: file[2],
            revision_major: file[4],
            revision_minor: file[5],
            platform: string(0x8, 0x20)?,
            display_version: string(0x68, 0x18)?,
        })
    }
}

/// The system version installed on the eMMC user-area `image` of the
/// console `keys` describes: as system data 0100000000000809
/// (SystemVersion) holds it.
pub fn installed_system_version(image: &File, keys: &KeySet) -> Result<SystemVersion, NandError> {
    let filesystem = |error| NandError::Filesystem {
        partition: SYSTEM.partition,
        error,
    };
    let partition = EncryptedPartition::new(image, &storage_keys(image, keys)?, &SYSTEM);
    let mut volume = MountedFat32::open(partition).map_err(filesystem)?;
    let save_image = read_whole(
        &mut volume,
        &format!("{SAVES}/{:016x}", content_db::SAVE.id),
    )
    .map_err(filesystem)?;
    let save = save::open(&save_image, keys).map_err(SystemVersionError::Save)?;
    let database = save
        .file(&format!(
            "{}/{}",
            content_db::DIRECTORY,
            content_db::FILE_NAME
        ))
        .ok_or(SystemVersionError::NoDatabase)?;
    let content_id = data_content(database)?;
    let path = format!("{REGISTERED}/{}.nca", hex(&content_id));
    let archive = read_whole(&mut volume, &path).map_err(filesystem)?;
    let romfs = Nca::open(&archive, keys)
        .and_then(|nca| nca.romfs_section(0))
        .map_err(SystemVersionError::Nca)?;
    let file = romfs::file(&romfs, FILE).map_err(SystemVersionError::RomFs)?;
    Ok(SystemVersion::parse(file)?)
}

/// The whole file `path` of `volume`.
fn read_whole(
    volume: &mut MountedFat32<EncryptedPartition<'_>>,
    path: &str,
) -> Result<Vec<u8>, FatError> {
    let (parent, name) = path.rsplit_once('/').expect("an absolute path");
    let size = volume
        .list(parent)?
        .into_iter()
        .find(|file| !file.directory && file.name.eq_ignore_ascii_case(name))
        .ok_or_else(|| FatError::NotFound(path.into()))?
        .size;
    volume.read(path, size as usize)
}

/// The content ID of the data content of the newest SystemVersion record
/// in NCM's database file `database`.
fn data_content(database: &[u8]) -> Result<[u8; 16], SystemVersionError> {
    let (_, value) = content_db::key_value_entries(database)
        .map_err(SystemVersionError::Database)?
        .into_iter()
        .filter(|(key, _)| key.len() == KEY_SIZE)
        .filter(|(key, _)| u64::from_le_bytes(key[..8].try_into().unwrap()) == PROGRAM_ID)
        .max_by_key(|(key, _)| u32::from_le_bytes(key[8..12].try_into().unwrap()))
        .ok_or(SystemVersionError::NotInstalled)?;
    let short = || SystemVersionError::Database("a record shorter than its header says");
    let header = value.get(..VALUE_HEADER_SIZE).ok_or_else(short)?;
    let extended_header_size = usize::from(u16::from_le_bytes([header[0], header[1]]));
    let content_count = usize::from(u16::from_le_bytes([header[2], header[3]]));
    let start = VALUE_HEADER_SIZE + extended_header_size;
    let infos = value
        .get(start..start + content_count * CONTENT_INFO_SIZE)
        .ok_or_else(short)?;
    infos
        .as_chunks::<CONTENT_INFO_SIZE>()
        .0
        .iter()
        .map(ContentInfo::from_bytes)
        .find(|info| info.content_type == cnmt::CONTENT_TYPE_DATA)
        .map(|info| info.id)
        .ok_or(SystemVersionError::NoDataContent)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The first 0x100 bytes of firmware 23.5.0's version file, the
    /// version hash and display title left out (zeros).
    #[test]
    fn firmware_23_5_0_reads_as_it_displays() {
        let mut file = [0; FILE_SIZE];
        file[..8].copy_from_slice(&[0x17, 0x05, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00]);
        file[0x8..0xA].copy_from_slice(b"NX");
        file[0x68..0x6E].copy_from_slice(b"23.5.0");
        assert_eq!(
            SystemVersion::parse(&file).unwrap(),
            SystemVersion {
                major: 23,
                minor: 5,
                micro: 0,
                revision_major: 1,
                revision_minor: 0,
                platform: "NX".into(),
                display_version: "23.5.0".into(),
            }
        );
    }
}
