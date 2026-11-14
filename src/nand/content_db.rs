//! The content meta database of the built-in system storage, which the
//! content manager (NCM) reads from its system save and which retail NCM
//! never builds: it has to be on the NAND before the console first starts.
//!
//! NCM keeps the database of `BuiltInSystem` in system save
//! `0x8000000000000120` (`SYSTEM:/save/8000000000000120`): its image formats
//! `"%s:/meta"` (string at image offset 0x616e7) as the database's
//! directory and names the file in it `imkvdb.arc` (0x6132a). NCM's save as
//! the guest wrote it after an import holds exactly that directory and
//! file, and nothing else.
//!
//! The file is a key-value archive: a 12-byte header, `IMKV`, a reserved
//! word and the entry count (checked and read at 0x4ad40 of the image),
//! then per entry a 12-byte header, `IMEN`, key size and value size
//! (0x4ade0), followed by the 16-byte key and the value; words are
//! little-endian. The loop at 0x348c0 appends the entries to the in-memory
//! index in file order without sorting it, and lookups search the index
//! by key, so the entries are written in key order (the order of
//! [`ContentMetaKey`]'s fields, as in Atmosphère's `ContentMetaKey`). One
//! entry per content meta, keyed and valued as [`cnmt`] describes, the
//! value's meta content info pointing at the meta NCA the record came
//! from.
//!
//! For firmware 23.5.0 the archive is 121 records, 12004 bytes. Put into
//! the save [`system_saves`] makes, with the save data indexer's save
//! beside it, the guest's NCM mounts the save, reads the extra data, opens
//! the file and reads its 0x2ee4 bytes without creating anything, and
//! Loader starts boot2 (0100000000000008) and the programs after it.

use std::{
    fmt, fs,
    io::{self, Read},
    path::{Path, PathBuf},
};

use super::{
    cnmt::{self, CnmtError, ContentInfo, ContentMetaKey, PackagedContentMeta},
    crypto,
    nca::{self, Nca, NcaError},
    save::{self, Entry, SaveError, SystemSave},
    save_indexer::{self, IndexedSave},
};
use crate::keys::KeySet;

/// The system save NCM keeps the `BuiltInSystem` database in, as NCM
/// creates it: 0x6C000 bytes of data and journal, owner 0, flags 3 (kept
/// when system saves are reset and on refurbishment), the extra data of the
/// save the guest made.
pub const SAVE: SystemSave = SystemSave {
    id: 0x8000_0000_0000_0120,
    owner_id: 0,
    flags: 3,
    size: 0x6C000,
};
/// Where in that save.
pub const DIRECTORY: &str = "meta";
pub const FILE_NAME: &str = "imkvdb.arc";

const ARCHIVE_MAGIC: &[u8; 4] = b"IMKV";
const ENTRY_MAGIC: &[u8; 4] = b"IMEN";
const NCA_HEADER_SIZE: usize = 0xC00;

#[derive(Debug)]
pub enum ContentDbError {
    Io {
        path: PathBuf,
        error: io::Error,
    },
    Nca {
        path: PathBuf,
        error: NcaError,
    },
    Cnmt {
        path: PathBuf,
        error: CnmtError,
    },
    /// A meta NCA's file name is not the content ID its contents hash to.
    ContentId {
        path: PathBuf,
    },
    /// A meta NCA's section holds no `.cnmt` file, or several.
    NoContentMeta {
        path: PathBuf,
    },
    /// Two meta NCAs carry the same content meta key.
    Duplicate {
        key: ContentMetaKey,
    },
    /// The directory holds no meta NCA at all.
    Empty,
    Save(SaveError),
}

impl fmt::Display for ContentDbError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, error } => write!(f, "{}: {error}", path.display()),
            Self::Nca { path, error } => write!(f, "{}: {error}", path.display()),
            Self::Cnmt { path, error } => write!(f, "{}: {error}", path.display()),
            Self::ContentId { path } => {
                write!(
                    f,
                    "{}: the name is not the content ID of the data",
                    path.display()
                )
            }
            Self::NoContentMeta { path } => {
                write!(
                    f,
                    "{}: not exactly one .cnmt file in the meta NCA",
                    path.display()
                )
            }
            Self::Duplicate { key } => write!(
                f,
                "two meta NCAs hold content meta {:016x} version {}",
                key.id, key.version
            ),
            Self::Empty => write!(f, "no meta NCA in the firmware directory"),
            Self::Save(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for ContentDbError {}

/// A database record: one content meta.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Record {
    pub key: ContentMetaKey,
    pub value: Vec<u8>,
}

/// The records of every meta NCA (`*.nca` whose header says meta) in
/// `firmware_dir`, in key order.
pub fn records(firmware_dir: &Path, keys: &KeySet) -> Result<Vec<Record>, ContentDbError> {
    let io_error = |path: &Path| {
        let path = path.to_path_buf();
        move |error| ContentDbError::Io { path, error }
    };
    let mut records = Vec::new();
    for entry in fs::read_dir(firmware_dir).map_err(io_error(firmware_dir))? {
        let path = entry.map_err(io_error(firmware_dir))?.path();
        let Some(content_id) = content_id_from_name(&path) else {
            continue;
        };
        if let Some(record) = record(&path, content_id, keys)? {
            records.push(record);
        }
    }
    if records.is_empty() {
        return Err(ContentDbError::Empty);
    }
    records.sort_by_key(|record| record.key);
    if let Some(pair) = records.windows(2).find(|pair| pair[0].key == pair[1].key) {
        return Err(ContentDbError::Duplicate { key: pair[0].key });
    }
    Ok(records)
}

/// `<32 hex digits>.nca` or `<32 hex digits>.cnmt.nca`, as firmware
/// packages name their NCAs.
fn content_id_from_name(path: &Path) -> Option<[u8; 16]> {
    let name = path.file_name()?.to_str()?.strip_suffix(".nca")?;
    let digits = name.strip_suffix(".cnmt").unwrap_or(name);
    if digits.len() != 32 {
        return None;
    }
    let mut id = [0; 16];
    for (byte, at) in id.iter_mut().zip((0..32).step_by(2)) {
        *byte = u8::from_str_radix(digits.get(at..at + 2)?, 16).ok()?;
    }
    Some(id)
}

/// The record of the NCA at `path`, `None` if it is not a meta NCA.
fn record(
    path: &Path,
    content_id: [u8; 16],
    keys: &KeySet,
) -> Result<Option<Record>, ContentDbError> {
    let io_error = |error| ContentDbError::Io {
        path: path.to_path_buf(),
        error,
    };
    let nca_error = |error| ContentDbError::Nca {
        path: path.to_path_buf(),
        error,
    };
    let mut file = fs::File::open(path).map_err(io_error)?;
    let mut header = vec![0; NCA_HEADER_SIZE];
    file.read_exact(&mut header).map_err(io_error)?;
    if Nca::open(&header, keys).map_err(nca_error)?.content_type() != nca::CONTENT_TYPE_META {
        return Ok(None);
    }
    file.read_to_end(&mut header).map_err(io_error)?;
    let data = header;
    if crypto::sha256(&data)[..16] != content_id {
        return Err(ContentDbError::ContentId {
            path: path.to_path_buf(),
        });
    }
    let section = Nca::open(&data, keys)
        .and_then(|nca| nca.partition_section(0))
        .map_err(nca_error)?;
    let files = nca::pfs0_files(&section).map_err(nca_error)?;
    let [(_, packaged)] = files
        .iter()
        .filter(|(name, _)| name.ends_with(".cnmt"))
        .copied()
        .collect::<Vec<_>>()[..]
    else {
        return Err(ContentDbError::NoContentMeta {
            path: path.to_path_buf(),
        });
    };
    let meta = PackagedContentMeta::parse(packaged).map_err(|error| ContentDbError::Cnmt {
        path: path.to_path_buf(),
        error,
    })?;
    let meta_info = ContentInfo {
        id: content_id,
        size: data.len() as u64,
        attributes: 0,
        content_type: cnmt::CONTENT_TYPE_META,
        id_offset: 0,
    };
    Ok(Some(Record {
        key: meta.key(),
        value: meta.database_value(&meta_info),
    }))
}

/// The archive file of `records`, which must be in key order.
pub fn archive(records: &[Record]) -> Vec<u8> {
    let keys: Vec<[u8; 0x10]> = records.iter().map(|record| record.key.to_bytes()).collect();
    key_value_archive(
        keys.iter()
            .zip(records)
            .map(|(key, record)| (&key[..], &record.value[..])),
    )
}

/// A key-value archive (`IMKV`) of `entries`, in their order. FS's save
/// data indexer stores its index in the same format.
pub fn key_value_archive<'a>(
    entries: impl ExactSizeIterator<Item = (&'a [u8], &'a [u8])>,
) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(ARCHIVE_MAGIC);
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&u32::try_from(entries.len()).unwrap().to_le_bytes());
    for (key, value) in entries {
        out.extend_from_slice(ENTRY_MAGIC);
        out.extend_from_slice(&u32::try_from(key.len()).unwrap().to_le_bytes());
        out.extend_from_slice(&u32::try_from(value.len()).unwrap().to_le_bytes());
        out.extend_from_slice(key);
        out.extend_from_slice(value);
    }
    out
}

/// The `imkvdb.arc` of the firmware in `firmware_dir`.
pub fn build(firmware_dir: &Path, keys: &KeySet) -> Result<Vec<u8>, ContentDbError> {
    Ok(archive(&records(firmware_dir, keys)?))
}

/// The files of `SYSTEM:/save` that make the firmware in `firmware_dir`
/// known to NCM, by save data ID: NCM's database save, and the save data
/// indexer's save listing it. FS opens a system save only through the
/// indexer (see [`save_indexer`]), so both are needed.
pub fn system_saves(
    firmware_dir: &Path,
    keys: &KeySet,
) -> Result<[(u64, Vec<u8>); 2], ContentDbError> {
    let entries = [
        Entry::Directory(DIRECTORY.into()),
        Entry::File(
            format!("{DIRECTORY}/{FILE_NAME}"),
            build(firmware_dir, keys)?,
        ),
    ];
    let database = save::system_save(&SAVE, &entries, keys).map_err(ContentDbError::Save)?;
    let indexed = IndexedSave {
        id: SAVE.id,
        image_size: database.len() as u64,
    };
    let indexer = save_indexer::save(&[indexed], keys).map_err(ContentDbError::Save)?;
    Ok([(save_indexer::SAVE_DATA_ID, indexer), (SAVE.id, database)])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn archive_frames_each_record_after_the_count() {
        let record = Record {
            key: ContentMetaKey {
                id: 0x0100_0000_0000_0008,
                version: 1,
                meta_type: 1,
                install_type: 0,
            },
            value: vec![0xAA; 3],
        };
        let archive = archive(&[record]);
        assert_eq!(&archive[..12], b"IMKV\0\0\0\0\x01\0\0\0");
        assert_eq!(&archive[12..24], b"IMEN\x10\0\0\0\x03\0\0\0");
        assert_eq!(archive.len(), 12 + 12 + 0x10 + 3);
        assert_eq!(&archive[archive.len() - 3..], [0xAA; 3]);
    }

    #[test]
    fn only_content_id_names_are_ncas_to_look_at() {
        let id = content_id_from_name(Path::new("fw/38ac7257e3e822af9d4eb6f22d778826.cnmt.nca"));
        assert_eq!(id.unwrap()[..2], [0x38, 0xAC]);
        assert!(
            content_id_from_name(Path::new("fw/9d7556bb740ee7a90e972a8416e25f10.nca")).is_some()
        );
        assert!(content_id_from_name(Path::new("fw/notes.nca")).is_none());
        assert!(content_id_from_name(Path::new("fw/9d7556bb740ee7a90e972a8416e25f10")).is_none());
    }
}
