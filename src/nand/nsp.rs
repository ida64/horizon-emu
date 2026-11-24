//! Installing an NSP (a partition FS holding the content archives of a
//! title, its content meta, and the tickets of its title keys) onto the
//! user area of a console that has started at least once, the way the
//! console's own services would have stored it had they installed it:
//!
//! * the content archives in the content storage `BuiltInUser`
//!   (`USER:/Contents`), each at `registered/<%08X>/<content ID>.nca`, the
//!   directory being the first byte of the SHA-256 of the 16-byte content
//!   ID: the path NCM 23.5.0 builds for that storage (`0x39C80`, chosen at
//!   `0x305E0`; NCM picks the flat form of the system partition only for
//!   `BuiltInSystem`);
//! * the content meta record in NCM's database of that storage, system save
//!   `0x8000000000000121`, file `meta/imkvdb.arc` ([`content_db`]'s format,
//!   key order);
//! * each ticket in ES's ticket database: common tickets (device ID 0) in
//!   save `0x80000000000000E1` (`escommon`), personalized ones in
//!   `0x80000000000000E2` (`espersonalized`), `ticket_list.bin` holding a
//!   0x20-byte entry per slot (rights ID, ticket ID, account ID, property
//!   mask, two zero bytes) up to an end marker (28 bytes of `FF`, four of
//!   zero), `ticket.bin` the raw ticket of slot `i` at `i * 0x400` (ES
//!   `sub_4E960`, `sub_4EC40`, `sub_4E5C0`, `sub_51D10`);
//! * the application record in NS's database, save `0x8000000000000040`
//!   (`apprecdb`): `index`, newest record first, and
//!   `kvs/<application ID, little-endian hex>.val`, the content meta keys
//!   of the application with the storage that holds them (NS's install
//!   commit `0x119480`, then `0x80EBC`).
//!
//! The console creates those saves itself on its first start (NCM when ns
//! activates the storage, ES and NS at boot), so the install edits them
//! and adds nothing to the save data indexer.

use std::{
    fmt,
    fs::File,
    io::{self, Read, Seek, SeekFrom},
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
};

use super::{
    EncryptedPartition, NandError, SAVES, SYSTEM, USER,
    cnmt::{self, CnmtError, ContentInfo, ContentMetaKey, PackagedContentMeta},
    content_db, crypto,
    fat::{Fat32, FatError, MountedFat32},
    nca::{self, Nca, NcaError},
    save::{SaveEditor, SaveError},
    storage_keys,
    system_version::{SystemVersion, installed_system_version, read_whole},
};
use crate::keys::KeySet;
use sha2::{Digest, Sha256};

/// NCM's content meta database of `BuiltInUser`.
pub const NCM_USER_SAVE: u64 = 0x8000_0000_0000_0121;
/// ES's ticket databases.
pub const ES_COMMON_SAVE: u64 = 0x8000_0000_0000_00E1;
pub const ES_PERSONALIZED_SAVE: u64 = 0x8000_0000_0000_00E2;
/// NS's application record database.
pub const NS_RECORD_SAVE: u64 = 0x8000_0000_0000_0040;

/// Where in the user partition NCM keeps the contents of `BuiltInUser`.
pub const USER_REGISTERED: &str = "/Contents/registered";
pub const USER_PLACEHOLDERS: &str = "/Contents/placehld";
/// Size of each part of a concatenation file: LibHac's
/// `ConcatenationFileSystem` default, 0xFFFF0000 \[public\].
const CONCATENATION_PART: u64 = 0xFFFF_0000;

const META_TYPE_APPLICATION: u8 = 0x80;
const META_TYPE_ADD_ON_CONTENT: u8 = 0x82;
/// NCM's and NS's storage ID of the user partition.
const STORAGE_BUILT_IN_USER: u8 = 4;

/// Tickets as ES takes them: RSA-2048 with SHA-256 (`sub_8BF20` takes no
/// other), format version 2, at most 0x400 bytes.
const TICKET_SIGNATURE_TYPE: u32 = 0x0001_0004;
const TICKET_BODY: usize = 4 + 0x100 + 0x3C;
const TICKET_FORMAT_VERSION: u8 = 2;
pub const TICKET_RECORD_SIZE: usize = 0x400;
/// Bytes of a ticket ES reads its fields from: the body up to the account
/// ID and the section size.
const TICKET_MIN_SIZE: usize = TICKET_BODY + 0x180;
/// Ticket property: deleted by ES at boot (`sub_4E750`).
const TICKET_VOLATILE: u16 = 0x10;

pub const TICKET_LIST_ENTRY_SIZE: usize = 0x20;
const TICKET_LIST_END: [u8; TICKET_LIST_ENTRY_SIZE] = {
    let mut end = [0xFF; TICKET_LIST_ENTRY_SIZE];
    let mut at = 28;
    while at < TICKET_LIST_ENTRY_SIZE {
        end[at] = 0;
        at += 1;
    }
    end
};

/// NS's `index`: sequence, count, then 0x2000 records of 0x18 bytes.
pub const NS_INDEX_SIZE: usize = 0x30010;
const NS_INDEX_RECORDS: usize = 0x2000;
const NS_RECORD_SIZE: usize = 0x18;
/// `last_event` NS's install leaves (`0x80EBC`).
const NS_EVENT_INSTALLED: u8 = 0x0C;
/// A `.val`: a 0x208-byte header, then 0x18-byte content storage records.
const NS_VALUE_HEADER: usize = 0x208;
const NS_VALUE_RECORDS_MAX: usize = 0x800;
const NS_VALUE_TIME: usize = 0x68;
const NS_VALUE_CHANGES: usize = 0x72;
const NS_VALUE_FLAGS: usize = 0x74;

#[derive(Debug)]
pub enum NspError {
    Io { path: PathBuf, error: io::Error },
    /// Not a partition FS, or a malformed one: what is wrong.
    NotNsp(&'static str),
    Nca { name: String, error: NcaError },
    Cnmt { name: String, error: CnmtError },
    /// The package holds no content meta.
    NoContentMeta,
    /// The content meta names a content the package lacks, or one of
    /// another size.
    MissingContent { title: u64, id: String },
    /// A content does not hash to what its content meta says.
    ContentHash { id: String },
    /// A content needs a title key, and no ticket of its rights ID came.
    NoTicket { rights_id: String },
    /// A ticket ES would not take: why.
    Ticket { name: String, why: String },
}

impl fmt::Display for NspError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, error } => write!(f, "{}: {error}", path.display()),
            Self::NotNsp(what) => write!(f, "not an NSP: {what}"),
            Self::Nca { name, error } => write!(f, "{name}: {error}"),
            Self::Cnmt { name, error } => write!(f, "{name}: {error}"),
            Self::NoContentMeta => write!(f, "the NSP holds no content meta (.cnmt.nca)"),
            Self::MissingContent { title, id } => write!(
                f,
                "the content meta of {title:016x} names content {id}, which the NSP lacks \
                 or holds at another size"
            ),
            Self::ContentHash { id } => {
                write!(f, "content {id} does not match the hash its content meta gives")
            }
            Self::NoTicket { rights_id } => {
                write!(f, "no ticket in the NSP for rights ID {rights_id}")
            }
            Self::Ticket { name, why } => write!(f, "{name}: {why}"),
        }
    }
}

impl std::error::Error for NspError {}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// A file of the package: name and where its bytes are in the NSP.
#[derive(Debug, Clone)]
pub struct PackageFile {
    pub name: String,
    pub offset: u64,
    pub size: u64,
}

/// An NSP on the host, its file table read.
pub struct Nsp {
    path: PathBuf,
    file: File,
    files: Vec<PackageFile>,
}

impl Nsp {
    pub fn open(path: &Path) -> Result<Self, NspError> {
        let io_error = |error| NspError::Io {
            path: path.into(),
            error,
        };
        let file = File::open(path).map_err(io_error)?;
        let length = file.metadata().map_err(io_error)?.len();
        let mut header = [0; 0x10];
        file.read_exact_at(&mut header, 0).map_err(io_error)?;
        if &header[..4] != b"PFS0" {
            return Err(NspError::NotNsp("no PFS0 magic"));
        }
        let count = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
        let names_size = u32::from_le_bytes(header[8..12].try_into().unwrap()) as usize;
        let table_size = count * 0x18 + names_size;
        if 0x10 + table_size as u64 > length {
            return Err(NspError::NotNsp("the file table is past the end"));
        }
        let mut table = vec![0; table_size];
        file.read_exact_at(&mut table, 0x10).map_err(io_error)?;
        let (entries, names) = table.split_at(count * 0x18);
        let data_start = 0x10 + table_size as u64;
        let files = entries
            .as_chunks::<0x18>()
            .0
            .iter()
            .map(|entry| {
                let offset = u64::from_le_bytes(entry[..8].try_into().unwrap());
                let size = u64::from_le_bytes(entry[8..16].try_into().unwrap());
                let name_at = u32::from_le_bytes(entry[16..20].try_into().unwrap()) as usize;
                let name = names
                    .get(name_at..)
                    .and_then(|rest| rest.split(|&byte| byte == 0).next())
                    .and_then(|name| std::str::from_utf8(name).ok())
                    .ok_or(NspError::NotNsp("a file name is not a string in the table"))?;
                let offset = data_start
                    .checked_add(offset)
                    .filter(|&start| start.checked_add(size).is_some_and(|end| end <= length))
                    .ok_or(NspError::NotNsp("a file is past the end"))?;
                Ok(PackageFile {
                    name: name.into(),
                    offset,
                    size,
                })
            })
            .collect::<Result<_, NspError>>()?;
        Ok(Self {
            path: path.into(),
            file,
            files,
        })
    }

    pub fn files(&self) -> &[PackageFile] {
        &self.files
    }

    fn io_error(&self, error: io::Error) -> NspError {
        NspError::Io {
            path: self.path.clone(),
            error,
        }
    }

    /// The whole of `file`, for the small ones.
    pub fn read(&self, file: &PackageFile) -> Result<Vec<u8>, NspError> {
        let mut data = vec![0; file.size as usize];
        self.file
            .read_exact_at(&mut data, file.offset)
            .map_err(|error| self.io_error(error))?;
        Ok(data)
    }

    /// A reader of the bytes of `file`.
    pub fn reader(&self, file: &PackageFile) -> Result<impl Read + use<>, NspError> {
        let mut handle = self.file.try_clone().map_err(|error| self.io_error(error))?;
        handle
            .seek(SeekFrom::Start(file.offset))
            .map_err(|error| self.io_error(error))?;
        Ok(handle.take(file.size))
    }

    fn by_name(&self, name: &str) -> Option<&PackageFile> {
        self.files.iter().find(|file| file.name == name)
    }
}

/// A content archive of a title, as its content meta names it.
#[derive(Debug, Clone)]
pub struct Content {
    pub info: ContentInfo,
    pub file: PackageFile,
    /// Zero unless the archive is encrypted with a title key.
    pub rights_id: [u8; 16],
}

impl Content {
    /// Its path in the user partition.
    pub fn path(&self) -> String {
        content_path(&self.info.id)
    }
}

/// Where NCM keeps content `id` of `BuiltInUser` (`0x39C80`).
pub fn content_path(id: &[u8; 16]) -> String {
    format!(
        "{USER_REGISTERED}/{:08X}/{}.nca",
        crypto::sha256(id)[0],
        hex(id)
    )
}

/// One title of the package: its content meta and contents, the meta
/// archive first.
#[derive(Debug, Clone)]
pub struct Title {
    pub key: ContentMetaKey,
    /// The application the title belongs to: itself for an application, the
    /// one its extended header names for a patch or an add-on.
    pub application_id: u64,
    pub required_system_version: u32,
    /// NCM's database value.
    pub record: Vec<u8>,
    pub contents: Vec<Content>,
}

/// A ticket of the package, checked the way ES checks what it imports,
/// short of the signature.
#[derive(Debug, Clone)]
pub struct Ticket {
    pub name: String,
    pub data: Vec<u8>,
    pub rights_id: [u8; 16],
    pub ticket_id: u64,
    pub device_id: u64,
    pub account_id: u32,
    pub property_mask: u16,
    pub version: u16,
}

impl Ticket {
    pub fn parse(name: &str, data: Vec<u8>) -> Result<Self, NspError> {
        let refuse = |why: &str| NspError::Ticket {
            name: name.into(),
            why: why.into(),
        };
        if data.len() < TICKET_MIN_SIZE || data.len() > TICKET_RECORD_SIZE {
            return Err(refuse("not 0x2C0 to 0x400 bytes long"));
        }
        if u32::from_le_bytes(data[..4].try_into().unwrap()) != TICKET_SIGNATURE_TYPE {
            return Err(refuse(
                "signed other than with RSA-2048 and SHA-256, which ES takes alone",
            ));
        }
        let body = &data[TICKET_BODY..];
        if body[0x140] != TICKET_FORMAT_VERSION {
            return Err(refuse("format version other than 2"));
        }
        if body[0x141] > 1 {
            return Err(refuse("a title key type other than common or personalized"));
        }
        Ok(Self {
            name: name.into(),
            rights_id: body[0x160..0x170].try_into().unwrap(),
            ticket_id: u64::from_le_bytes(body[0x150..0x158].try_into().unwrap()),
            device_id: u64::from_le_bytes(body[0x158..0x160].try_into().unwrap()),
            account_id: u32::from_le_bytes(body[0x170..0x174].try_into().unwrap()),
            property_mask: u16::from_le_bytes(body[0x146..0x148].try_into().unwrap()),
            version: u16::from_le_bytes(body[0x142..0x144].try_into().unwrap()),
            data,
        })
    }

    /// The database ES keeps it in.
    pub fn save(&self) -> u64 {
        if self.device_id == 0 {
            ES_COMMON_SAVE
        } else {
            ES_PERSONALIZED_SAVE
        }
    }

    fn list_entry(&self) -> [u8; TICKET_LIST_ENTRY_SIZE] {
        let mut entry = [0; TICKET_LIST_ENTRY_SIZE];
        entry[..16].copy_from_slice(&self.rights_id);
        entry[0x10..0x18].copy_from_slice(&self.ticket_id.to_le_bytes());
        entry[0x18..0x1C].copy_from_slice(&self.account_id.to_le_bytes());
        entry[0x1C..0x1E].copy_from_slice(&self.property_mask.to_le_bytes());
        entry
    }
}

/// What an NSP holds, checked: every content its content metas name is
/// there at its size and hash, and every title key it needs has a ticket.
#[derive(Debug, Clone)]
pub struct Package {
    pub titles: Vec<Title>,
    pub tickets: Vec<Ticket>,
}

/// Reads and checks the package `nsp`. `progress` is told the bytes hashed
/// so far and in all.
pub fn read_package(
    nsp: &Nsp,
    keys: &KeySet,
    mut progress: impl FnMut(u64, u64),
) -> Result<Package, NspError> {
    let mut titles = Vec::new();
    for file in nsp.files() {
        let Some(digits) = file.name.strip_suffix(".cnmt.nca") else {
            continue;
        };
        let data = nsp.read(file)?;
        let id = content_id(digits).filter(|id| crypto::sha256(&data)[..16] == *id);
        let Some(id) = id else {
            return Err(NspError::ContentHash {
                id: digits.into(),
            });
        };
        titles.push(title(nsp, keys, file, id, &data)?);
    }
    if titles.is_empty() {
        return Err(NspError::NoContentMeta);
    }
    let mut tickets = Vec::new();
    for file in nsp.files().iter().filter(|file| file.name.ends_with(".tik")) {
        tickets.push(Ticket::parse(&file.name, nsp.read(file)?)?);
    }
    for content in titles.iter().flat_map(|title| &title.contents) {
        if content.rights_id != [0; 16]
            && !tickets.iter().any(|ticket| ticket.rights_id == content.rights_id)
        {
            return Err(NspError::NoTicket {
                rights_id: hex(&content.rights_id),
            });
        }
    }
    let total: u64 = titles
        .iter()
        .flat_map(|title| &title.contents[1..])
        .map(|content| content.file.size)
        .sum();
    let mut done = 0;
    for title in &titles {
        for content in &title.contents[1..] {
            let expected = title_hash(nsp, keys, title, content)?;
            let mut reader = nsp.reader(&content.file)?;
            let mut hasher = Sha256::new();
            let mut buffer = vec![0; 1 << 20];
            loop {
                let read = reader
                    .read(&mut buffer)
                    .map_err(|error| nsp.io_error(error))?;
                if read == 0 {
                    break;
                }
                hasher.update(&buffer[..read]);
                done += read as u64;
                progress(done, total);
            }
            if <[u8; 32]>::from(hasher.finalize()) != expected {
                return Err(NspError::ContentHash {
                    id: hex(&content.info.id),
                });
            }
        }
    }
    Ok(Package { titles, tickets })
}

fn content_id(digits: &str) -> Option<[u8; 16]> {
    if digits.len() != 32 {
        return None;
    }
    let mut id = [0; 16];
    for (byte, at) in id.iter_mut().zip((0..32).step_by(2)) {
        *byte = u8::from_str_radix(digits.get(at..at + 2)?, 16).ok()?;
    }
    Some(id)
}

/// The packaged SHA-256 of `content`, from the title's meta archive.
fn title_hash(
    nsp: &Nsp,
    keys: &KeySet,
    title: &Title,
    content: &Content,
) -> Result<[u8; 32], NspError> {
    let meta = &title.contents[0].file;
    let data = nsp.read(meta)?;
    let packaged = packaged_meta(keys, &meta.name, &data)?;
    let meta = PackagedContentMeta::parse(&packaged).map_err(|error| NspError::Cnmt {
        name: title.contents[0].file.name.clone(),
        error,
    })?;
    meta.packaged_contents()
        .find(|(_, info)| info.id == content.info.id)
        .map(|(hash, _)| hash)
        .ok_or(NspError::MissingContent {
            title: title.key.id,
            id: hex(&content.info.id),
        })
}

/// The `.cnmt` file in meta archive `data`.
fn packaged_meta(keys: &KeySet, name: &str, data: &[u8]) -> Result<Vec<u8>, NspError> {
    let nca_error = |error| NspError::Nca {
        name: name.into(),
        error,
    };
    let section = Nca::open(data, keys)
        .and_then(|nca| nca.partition_section(0))
        .map_err(nca_error)?;
    let files = nca::pfs0_files(&section).map_err(nca_error)?;
    match files
        .iter()
        .filter(|(name, _)| name.ends_with(".cnmt"))
        .collect::<Vec<_>>()[..]
    {
        [(_, packaged)] => Ok(packaged.to_vec()),
        _ => Err(NspError::NoContentMeta),
    }
}

fn title(
    nsp: &Nsp,
    keys: &KeySet,
    meta_file: &PackageFile,
    meta_id: [u8; 16],
    data: &[u8],
) -> Result<Title, NspError> {
    let packaged = packaged_meta(keys, &meta_file.name, data)?;
    let meta = PackagedContentMeta::parse(&packaged).map_err(|error| NspError::Cnmt {
        name: meta_file.name.clone(),
        error,
    })?;
    let key = meta.key();
    let meta_info = ContentInfo {
        id: meta_id,
        size: meta_file.size,
        attributes: 0,
        content_type: cnmt::CONTENT_TYPE_META,
        id_offset: 0,
    };
    let mut contents = vec![Content {
        info: meta_info,
        file: meta_file.clone(),
        rights_id: [0; 16],
    }];
    for (_, info) in meta.packaged_contents() {
        if info.content_type == cnmt::CONTENT_TYPE_DELTA_FRAGMENT {
            continue;
        }
        let name = format!("{}.nca", hex(&info.id));
        let file = nsp
            .by_name(&name)
            .filter(|file| file.size == info.size)
            .ok_or_else(|| NspError::MissingContent {
                title: key.id,
                id: hex(&info.id),
            })?;
        let mut header = vec![0; nca::HEADER_SIZE];
        nsp.file
            .read_exact_at(&mut header, file.offset)
            .map_err(|error| nsp.io_error(error))?;
        let archive = Nca::open(&header, keys).map_err(|error| NspError::Nca {
            name: name.clone(),
            error,
        })?;
        contents.push(Content {
            info,
            file: file.clone(),
            rights_id: archive.header()[0x230..0x240].try_into().unwrap(),
        });
    }
    let extended = meta.extended_header();
    let first_word = |at: usize| {
        extended
            .get(at..at + 8)
            .map(|bytes| u64::from_le_bytes(bytes.try_into().unwrap()))
            .unwrap_or(0)
    };
    let (application_id, required_system_version) = match key.meta_type {
        META_TYPE_APPLICATION => (key.id, first_word(8) as u32),
        cnmt::META_TYPE_PATCH => (first_word(0), first_word(8) as u32),
        META_TYPE_ADD_ON_CONTENT => (first_word(0), 0),
        _ => (key.id, 0),
    };
    Ok(Title {
        key,
        application_id,
        required_system_version,
        record: meta.database_value(&meta_info),
        contents,
    })
}

/// NCM's database `archive` (absent: empty) with `title`'s record in its
/// place in key order, replacing one of the same key.
pub fn ncm_database(archive: Option<&[u8]>, title: &Title) -> Result<Vec<u8>, &'static str> {
    let mut records: Vec<content_db::Record> = match archive {
        Some(archive) => content_db::key_value_entries(archive)?
            .into_iter()
            .map(|(key, value)| {
                let key: &[u8; 0x10] = key.try_into().map_err(|_| "a key of other than 16 bytes")?;
                Ok(content_db::Record {
                    key: ContentMetaKey::from_bytes(key),
                    value: value.to_vec(),
                })
            })
            .collect::<Result<_, &'static str>>()?,
        None => Vec::new(),
    };
    records.retain(|record| record.key != title.key);
    let at = records.partition_point(|record| record.key < title.key);
    records.insert(
        at,
        content_db::Record {
            key: title.key,
            value: title.record.clone(),
        },
    );
    Ok(content_db::archive(&records))
}

/// Where `ticket` goes in ES's `ticket_list.bin` `list`: the slot, and the
/// bytes to write from that slot on (its entry, and the end marker after
/// it when the slot was the end marker's), as `sub_4EC40` inserts.
pub fn ticket_slot(list: &[u8], ticket: &Ticket) -> Result<(usize, Vec<u8>), String> {
    let entries = list.as_chunks::<TICKET_LIST_ENTRY_SIZE>().0;
    let capacity = entries.len();
    let end = entries
        .iter()
        .position(|entry| entry[..28] == TICKET_LIST_END[..28])
        .ok_or("the ticket list has no end marker")?;
    let entry = ticket.list_entry();
    if let Some(slot) = entries[..end]
        .iter()
        .position(|stored| stored[..0x18] == entry[..0x18])
    {
        return Ok((slot, entry.to_vec()));
    }
    if let Some(slot) = entries[..end].iter().position(|stored| *stored == [0; 0x20]) {
        return Ok((slot, entry.to_vec()));
    }
    let mut bytes = entry.to_vec();
    if end + 1 < capacity {
        bytes.extend_from_slice(&TICKET_LIST_END);
    } else if end + 1 > capacity {
        return Err("the ticket database is full".into());
    }
    Ok((end, bytes))
}

/// The version of the ticket stored in `record` (a `ticket.bin` slot), if
/// it is newer than `ticket`, which ES refuses (0xC91).
pub fn newer_stored_ticket(record: &[u8], ticket: &Ticket) -> Option<u16> {
    let stored = record.get(TICKET_BODY + 0x142..TICKET_BODY + 0x144)?;
    let stored = u16::from_le_bytes(stored.try_into().unwrap());
    (stored > ticket.version).then_some(stored)
}

/// Whether ES deletes `ticket` at boot.
pub fn is_volatile(ticket: &Ticket) -> bool {
    ticket.property_mask & TICKET_VOLATILE != 0
}

/// NS's `index` with the record of `application_id` moved or put first,
/// as NS's install leaves it.
pub fn ns_index(index: &[u8], application_id: u64) -> Result<Vec<u8>, &'static str> {
    if index.len() != NS_INDEX_SIZE {
        return Err("the record index is not 0x30010 bytes");
    }
    let mut index = index.to_vec();
    let sequence = u64::from_le_bytes(index[..8].try_into().unwrap()) + 2;
    let count = (i32::from_le_bytes(index[8..12].try_into().unwrap()).max(0) as usize)
        .min(NS_INDEX_RECORDS);
    let records = &mut index[0x10..];
    let existing = (0..count).find(|&slot| {
        records[slot * NS_RECORD_SIZE..slot * NS_RECORD_SIZE + 8] == application_id.to_le_bytes()
    });
    let (shifted, count) = match existing {
        Some(slot) => (slot, count),
        None if count == NS_INDEX_RECORDS => return Err("the record index is full"),
        None => (count, count + 1),
    };
    records.copy_within(0..shifted * NS_RECORD_SIZE, NS_RECORD_SIZE);
    let mut record = [0; NS_RECORD_SIZE];
    record[..8].copy_from_slice(&application_id.to_le_bytes());
    record[8] = NS_EVENT_INSTALLED;
    record[0x10..].copy_from_slice(&sequence.to_le_bytes());
    records[..NS_RECORD_SIZE].copy_from_slice(&record);
    index[..8].copy_from_slice(&sequence.to_le_bytes());
    index[8..12].copy_from_slice(&(count as i32).to_le_bytes());
    Ok(index)
}

/// The `.val` path of `application_id` in NS's database.
pub fn ns_value_path(application_id: u64) -> String {
    format!("kvs/{}.val", hex(&application_id.to_le_bytes()))
}

/// NS's `.val` of an application (absent: a new one) with `key` stored in
/// `BuiltInUser`, replacing an entry of the same key; `now` is the POSIX
/// time NS takes from the network clock for a new value.
pub fn ns_value(value: Option<&[u8]>, key: &ContentMetaKey, now: u64) -> Result<Vec<u8>, &'static str> {
    let mut value = match value {
        Some(value) => value.to_vec(),
        None => {
            let mut value = vec![0; NS_VALUE_HEADER];
            value[NS_VALUE_TIME..NS_VALUE_TIME + 8].copy_from_slice(&now.to_le_bytes());
            value
        }
    };
    if value.len() < NS_VALUE_HEADER {
        return Err("an application record shorter than its header");
    }
    let count = i32::from_le_bytes(value[..4].try_into().unwrap()).max(0) as usize;
    if value.len() < NS_VALUE_HEADER + count * NS_RECORD_SIZE {
        return Err("an application record shorter than its entries");
    }
    value.truncate(NS_VALUE_HEADER + count * NS_RECORD_SIZE);
    let mut entry = [0; NS_RECORD_SIZE];
    entry[..0x10].copy_from_slice(&key.to_bytes());
    entry[0x10] = STORAGE_BUILT_IN_USER;
    let same = (0..count).find(|&slot| {
        let at = NS_VALUE_HEADER + slot * NS_RECORD_SIZE;
        value[at..at + 0x0E] == entry[..0x0E]
    });
    match same {
        Some(slot) => {
            let at = NS_VALUE_HEADER + slot * NS_RECORD_SIZE;
            value[at..at + NS_RECORD_SIZE].copy_from_slice(&entry);
        }
        None if count == NS_VALUE_RECORDS_MAX => return Err("the application has 0x800 contents"),
        None => {
            value.extend_from_slice(&entry);
            value[..4].copy_from_slice(&((count + 1) as i32).to_le_bytes());
        }
    }
    value[NS_VALUE_CHANGES] = value[NS_VALUE_CHANGES].wrapping_add(1);
    value[NS_VALUE_FLAGS] &= !1;
    Ok(value)
}

/// NS's `signature` of its record database (`sub_85158`): version 0 and
/// "ARDB". With any other, NS wipes the database as it mounts it.
const NS_SIGNATURE: [u8; 0x18] = {
    let mut signature = [0; 0x18];
    signature[8] = b'A';
    signature[9] = b'R';
    signature[10] = b'D';
    signature[11] = b'B';
    signature
};

/// What [`install`] stored.
#[derive(Debug, Clone)]
pub struct Installed {
    pub titles: Vec<ContentMetaKey>,
    /// Content archives written, and those already there at their size.
    pub written: usize,
    pub kept: usize,
    pub bytes: u64,
}

/// What [`install`] is doing, for a progress display.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// Copying content archives: bytes so far and in all.
    Copying,
    /// Recording them in the databases.
    Registering,
}

/// Why [`install`] stopped. The image is unchanged unless the error says
/// otherwise: the archives are written before any database, so an
/// interrupted install leaves only files the databases do not name.
#[derive(Debug)]
pub enum InstallError {
    Nand(NandError),
    /// A save the console makes on its first start is not there yet.
    NoSave(u64),
    Save { id: u64, error: SaveError },
    /// A database file is not what its owner writes: what is wrong.
    Database { id: u64, what: String },
    /// The title needs a newer system than the image has.
    SystemTooOld { title: u64, required: u32, installed: u32 },
    /// A ticket ES would refuse to replace: the stored one is newer.
    NewerTicket { rights_id: String, stored: u16 },
    /// A content archive is there at another size.
    ContentSize { path: String },
}

impl fmt::Display for InstallError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let version = |v: u32| format!("{}.{}.{}", v >> 26, v >> 20 & 0x3F, v >> 16 & 0xF);
        match self {
            Self::Nand(error) => error.fmt(f),
            Self::NoSave(id) => write!(
                f,
                "system save {id:016x} does not exist yet: start the console once and let it \
                 reach HOME, then stop it and install"
            ),
            Self::Save { id, error } => write!(f, "system save {id:016x}: {error}"),
            Self::Database { id, what } => write!(f, "system save {id:016x}: {what}"),
            Self::SystemTooOld {
                title,
                required,
                installed,
            } => write!(
                f,
                "{title:016x} needs system version {} or newer; the image has {}",
                version(*required),
                version(*installed)
            ),
            Self::NewerTicket { rights_id, stored } => write!(
                f,
                "the console holds a newer ticket (version {stored}) for rights ID {rights_id}"
            ),
            Self::ContentSize { path } => {
                write!(f, "USER:{path} exists at another size than the content's")
            }
        }
    }
}

impl std::error::Error for InstallError {}

impl From<NandError> for InstallError {
    fn from(error: NandError) -> Self {
        Self::Nand(error)
    }
}

/// A file of a save, `None` if it is not there.
fn save_file(save: &SaveEditor, id: u64, path: &str) -> Result<Option<Vec<u8>>, InstallError> {
    match save.read(path) {
        Ok(data) => Ok(Some(data)),
        Err(SaveError::NotFound(_)) => Ok(None),
        Err(error) => Err(InstallError::Save { id, error }),
    }
}

/// Puts `data` in the save as file `path`, replacing what is there.
fn put_save_file(
    save: &mut SaveEditor,
    id: u64,
    path: &str,
    old_size: Option<usize>,
    data: &[u8],
) -> Result<(), InstallError> {
    let failed = |error| InstallError::Save { id, error };
    match old_size {
        Some(size) if size == data.len() => save.write(path, 0, data).map_err(failed),
        Some(_) => {
            save.delete_file(path).map_err(failed)?;
            save.create_file(path, data).map_err(failed)
        }
        None => save.create_file(path, data).map_err(failed),
    }
}

fn database_error(id: u64) -> impl Fn(&str) -> InstallError {
    move |what| InstallError::Database {
        id,
        what: what.into(),
    }
}

/// `major.minor.micro` of `version` as content metas encode system
/// versions, the release step zero.
fn encoded(version: &SystemVersion) -> u32 {
    u32::from(version.major) << 26 | u32::from(version.minor) << 20 | u32::from(version.micro) << 16
}

/// Installs `package` (read from `nsp` by [`read_package`]) onto the user
/// area `image` of the console `keys` describes, which has to have started
/// once (its ES, NS and NCM saves exist) and be stopped: no emulator may
/// have it open. `now` is the POSIX time NS records for a new application.
pub fn install(
    image: &File,
    keys: &KeySet,
    nsp: &Nsp,
    package: &Package,
    now: u64,
    mut progress: impl FnMut(Stage, u64, u64),
) -> Result<Installed, InstallError> {
    let storage = storage_keys(image, keys)?;
    let filesystem = |partition: &'static str| {
        move |error| InstallError::Nand(NandError::Filesystem { partition, error })
    };
    let installed = encoded(&installed_system_version(image, keys)?);
    for title in &package.titles {
        if title.required_system_version & !0xFFFF > installed {
            return Err(InstallError::SystemTooOld {
                title: title.key.id,
                required: title.required_system_version,
                installed,
            });
        }
    }
    // The system partition has to be one a console left between two of
    // its operations, as the writer's checks tell; its saves change in
    // place only.
    drop(
        Fat32::open(EncryptedPartition::new(image, &storage, &SYSTEM))
            .map_err(filesystem(SYSTEM.partition))?,
    );
    let mut system = MountedFat32::open(EncryptedPartition::new(image, &storage, &SYSTEM))
        .map_err(filesystem(SYSTEM.partition))?;
    let mut ids = vec![NCM_USER_SAVE, NS_RECORD_SAVE];
    for ticket in &package.tickets {
        if !ids.contains(&ticket.save()) {
            ids.push(ticket.save());
        }
    }
    let mut saves = Vec::new();
    for &id in &ids {
        let path = format!("{SAVES}/{id:016x}");
        let data = match read_whole(&mut system, &path) {
            Ok(data) => data,
            Err(FatError::NotFound(_)) => return Err(InstallError::NoSave(id)),
            Err(error) => return Err(filesystem(SYSTEM.partition)(error)),
        };
        let editor =
            SaveEditor::open(data, keys).map_err(|error| InstallError::Save { id, error })?;
        saves.push((id, path, editor));
    }
    let save = |saves: &[(u64, String, SaveEditor)], id: u64| -> usize {
        saves
            .iter()
            .position(|(save, ..)| *save == id)
            .expect("every save needed is loaded")
    };

    // NCM: the content meta records.
    let ncm = save(&saves, NCM_USER_SAVE);
    {
        let editor = &mut saves[ncm].2;
        let database = format!("{}/{}", content_db::DIRECTORY, content_db::FILE_NAME);
        if let Err(error) = editor.create_directory(content_db::DIRECTORY)
            && !matches!(error, SaveError::AlreadyExists(_))
        {
            return Err(InstallError::Save {
                id: NCM_USER_SAVE,
                error,
            });
        }
        let old = save_file(editor, NCM_USER_SAVE, &database)?;
        let mut archive = old.clone();
        for title in &package.titles {
            archive = Some(
                ncm_database(archive.as_deref(), title).map_err(database_error(NCM_USER_SAVE))?,
            );
        }
        let archive = archive.expect("a package has a title");
        put_save_file(
            editor,
            NCM_USER_SAVE,
            &database,
            old.as_ref().map(Vec::len),
            &archive,
        )?;
    }

    // ES: the tickets.
    for ticket in &package.tickets {
        let id = ticket.save();
        let at = save(&saves, id);
        let editor = &mut saves[at].2;
        let failed = |error| InstallError::Save { id, error };
        let list = editor.read("ticket_list.bin").map_err(failed)?;
        let (slot, entry) = ticket_slot(&list, ticket).map_err(|what| InstallError::Database { id, what })?;
        let records = editor.read("ticket.bin").map_err(failed)?;
        let record = records
            .get(slot * TICKET_RECORD_SIZE..(slot + 1) * TICKET_RECORD_SIZE)
            .ok_or_else(|| database_error(id)("ticket.bin is shorter than the list"))?;
        if list[slot * TICKET_LIST_ENTRY_SIZE..][..0x18] == entry[..0x18]
            && let Some(stored) = newer_stored_ticket(record, ticket)
        {
            return Err(InstallError::NewerTicket {
                rights_id: hex(&ticket.rights_id),
                stored,
            });
        }
        let mut padded = ticket.data.clone();
        padded.resize(TICKET_RECORD_SIZE, 0);
        editor
            .write("ticket.bin", (slot * TICKET_RECORD_SIZE) as u64, &padded)
            .map_err(failed)?;
        editor
            .write("ticket_list.bin", (slot * TICKET_LIST_ENTRY_SIZE) as u64, &entry)
            .map_err(failed)?;
    }

    // NS: the application records, the application's own title last so
    // that it ends up first in the index.
    {
        let at = save(&saves, NS_RECORD_SAVE);
        let editor = &mut saves[at].2;
        let id = NS_RECORD_SAVE;
        if save_file(editor, id, "signature")?.as_deref() != Some(&NS_SIGNATURE[..]) {
            return Err(database_error(id)(
                "NS has not set up its application record database (signature)",
            ));
        }
        let mut applications: Vec<u64> = Vec::new();
        for title in &package.titles {
            if !applications.contains(&title.application_id) {
                applications.push(title.application_id);
            }
        }
        for &application in &applications {
            let path = ns_value_path(application);
            let old = save_file(editor, id, &path)?;
            let mut value = old.clone();
            for title in package
                .titles
                .iter()
                .filter(|title| title.application_id == application)
            {
                value = Some(ns_value(value.as_deref(), &title.key, now).map_err(database_error(id))?);
            }
            let value = value.expect("the application has a title");
            put_save_file(editor, id, &path, old.as_ref().map(Vec::len), &value)?;
            let index = editor
                .read("index")
                .map_err(|error| InstallError::Save { id, error })?;
            let index = ns_index(&index, application).map_err(database_error(id))?;
            editor
                .write("index", 0, &index)
                .map_err(|error| InstallError::Save { id, error })?;
        }
    }

    // The archives, then the databases that name them.
    let mut user = MountedFat32::open(EncryptedPartition::new(image, &storage, &USER))
        .map_err(filesystem(USER.partition))?;
    let mut kept_paths = Vec::new();
    let contents: Vec<&Content> = package
        .titles
        .iter()
        .flat_map(|title| &title.contents)
        .collect();
    for content in &contents {
        let path = content.path();
        let (parent, name) = path.rsplit_once('/').expect("an absolute path");
        let existing = match user.list(parent) {
            Ok(files) => files
                .into_iter()
                .find(|file| file.name.eq_ignore_ascii_case(name)),
            Err(FatError::NotFound(_)) => None,
            Err(error) => return Err(filesystem(USER.partition)(error)),
        };
        if let Some(file) = existing {
            // A concatenation file: a directory of parts.
            let size = if file.directory {
                user.list(&path)
                    .map_err(filesystem(USER.partition))?
                    .iter()
                    .filter(|part| !part.directory)
                    .map(|part| u64::from(part.size))
                    .sum()
            } else {
                u64::from(file.size)
            };
            if size != content.info.size {
                return Err(InstallError::ContentSize { path });
            }
            kept_paths.push(path);
        }
    }
    drop(user);
    let mut user = Fat32::open(EncryptedPartition::new(image, &storage, &USER))
        .map_err(filesystem(USER.partition))?;
    let total: u64 = contents
        .iter()
        .filter(|content| !kept_paths.contains(&content.path()))
        .map(|content| content.info.size)
        .sum();
    let mut done = 0;
    for directory in [USER_REGISTERED, USER_PLACEHOLDERS] {
        user.create_dir_all(directory)
            .map_err(filesystem(USER.partition))?;
    }
    for content in &contents {
        let path = content.path();
        if kept_paths.contains(&path) {
            continue;
        }
        let (parent, _) = path.rsplit_once('/').expect("an absolute path");
        user.create_dir_all(parent)
            .map_err(filesystem(USER.partition))?;
        let mut reader = Counted {
            inner: nsp.reader(&content.file).map_err(|error| match error {
                NspError::Io { error, .. } => InstallError::Nand(NandError::Image(error)),
                other => database_error(0)(&other.to_string()),
            })?,
            done: &mut done,
            total,
            progress: &mut progress,
        };
        // FAT32 holds files below 4 GiB; a larger content is stored the way
        // FS's concatenation file system splits it, which is how NCM itself
        // creates every content (see `Fat32::write_concatenated`).
        if content.info.size > u64::from(u32::MAX) {
            user.write_concatenated(&path, &mut reader, content.info.size, CONCATENATION_PART)
        } else {
            user.write_file(&path, &mut reader, content.info.size)
        }
        .map_err(filesystem(USER.partition))?;
    }
    user.finish().map_err(filesystem(USER.partition))?;

    progress(Stage::Registering, 0, saves.len() as u64);
    let count = saves.len() as u64;
    for (index, (id, path, editor)) in saves.into_iter().enumerate() {
        let image = editor
            .finish()
            .map_err(|error| InstallError::Save { id, error })?;
        system
            .overwrite(&path, &image)
            .map_err(filesystem(SYSTEM.partition))?;
        progress(Stage::Registering, index as u64 + 1, count);
    }
    Ok(Installed {
        titles: package.titles.iter().map(|title| title.key).collect(),
        written: contents.len() - kept_paths.len(),
        kept: kept_paths.len(),
        bytes: total,
    })
}

/// A reader that reports the bytes read through it.
struct Counted<'a, R, P> {
    inner: R,
    done: &'a mut u64,
    total: u64,
    progress: &'a mut P,
}

impl<R: Read, P: FnMut(Stage, u64, u64)> Read for Counted<'_, R, P> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let read = self.inner.read(buffer)?;
        *self.done += read as u64;
        (self.progress)(Stage::Copying, *self.done, self.total);
        Ok(read)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ticket(rights: u8, ticket_id: u64) -> Ticket {
        let mut data = vec![0; 0x2C0];
        data[..4].copy_from_slice(&TICKET_SIGNATURE_TYPE.to_le_bytes());
        data[TICKET_BODY + 0x140] = TICKET_FORMAT_VERSION;
        data[TICKET_BODY + 0x150..TICKET_BODY + 0x158].copy_from_slice(&ticket_id.to_le_bytes());
        data[TICKET_BODY + 0x160..TICKET_BODY + 0x170].fill(rights);
        Ticket::parse("t.tik", data).unwrap()
    }

    fn empty_list(capacity: usize) -> Vec<u8> {
        let mut list = vec![0; capacity * TICKET_LIST_ENTRY_SIZE];
        list[..TICKET_LIST_ENTRY_SIZE].copy_from_slice(&TICKET_LIST_END);
        list
    }

    #[test]
    fn a_ticket_takes_the_end_markers_slot_and_moves_it_on() {
        let list = empty_list(4);
        let (slot, bytes) = ticket_slot(&list, &ticket(7, 0)).unwrap();
        assert_eq!(slot, 0);
        assert_eq!(&bytes[..16], &[7; 16]);
        assert_eq!(&bytes[0x20..], &TICKET_LIST_END);
    }

    #[test]
    fn a_ticket_replaces_its_own_entry_and_fills_holes_first() {
        let mut list = empty_list(4);
        let first = ticket(1, 0);
        list[..0x20].copy_from_slice(&first.list_entry());
        list[0x20..0x40].fill(0);
        list[0x40..0x60].copy_from_slice(&ticket(3, 0).list_entry());
        list[0x60..0x80].copy_from_slice(&TICKET_LIST_END);
        assert_eq!(ticket_slot(&list, &first).unwrap().0, 0);
        let (slot, bytes) = ticket_slot(&list, &ticket(2, 0)).unwrap();
        assert_eq!((slot, bytes.len()), (1, 0x20));
        assert_eq!(ticket_slot(&list, &ticket(1, 5)).unwrap().0, 1);
    }

    #[test]
    fn a_full_ticket_list_is_refused() {
        let mut list = empty_list(1);
        list[..0x20].copy_from_slice(&ticket(1, 0).list_entry());
        assert!(ticket_slot(&list, &ticket(2, 0)).is_err());
    }

    #[test]
    fn ns_index_puts_the_application_first() {
        let mut index = vec![0; NS_INDEX_SIZE];
        index[8] = 1;
        index[0x10..0x18].copy_from_slice(&5u64.to_le_bytes());
        let index = ns_index(&index, 9).unwrap();
        assert_eq!(index[..8], 2u64.to_le_bytes());
        assert_eq!(index[8..12], 2i32.to_le_bytes());
        assert_eq!(index[0x10..0x18], 9u64.to_le_bytes());
        assert_eq!(index[0x18], NS_EVENT_INSTALLED);
        assert_eq!(index[0x20..0x28], 2u64.to_le_bytes());
        assert_eq!(index[0x28..0x30], 5u64.to_le_bytes());
        let again = ns_index(&index, 5).unwrap();
        assert_eq!(again[8..12], 2i32.to_le_bytes());
        assert_eq!(again[0x10..0x18], 5u64.to_le_bytes());
        assert_eq!(again[0x28..0x30], 9u64.to_le_bytes());
    }

    #[test]
    fn ns_value_matches_what_ns_installs() {
        let key = ContentMetaKey {
            id: 0x0100_D710_0469_4000,
            version: 0,
            meta_type: META_TYPE_APPLICATION,
            install_type: 0,
        };
        let value = ns_value(None, &key, 0).unwrap();
        assert_eq!(value.len(), 0x220);
        assert_eq!(value[..4], 1i32.to_le_bytes());
        assert_eq!(value[NS_VALUE_CHANGES], 1);
        assert_eq!(
            value[0x208..],
            [
                0x00, 0x40, 0x69, 0x04, 0x10, 0xd7, 0x00, 0x01, 0, 0, 0, 0, 0x80, 0, 0, 0, 4, 0,
                0, 0, 0, 0, 0, 0
            ]
        );
        let again = ns_value(Some(&value), &key, 0).unwrap();
        assert_eq!(again.len(), 0x220);
        assert_eq!(again[NS_VALUE_CHANGES], 2);
    }

    #[test]
    fn contents_go_under_the_first_byte_of_their_ids_hash() {
        let id = content_id("f8123cc7a1c5ed87f65f85e037c0ba75").unwrap();
        assert_eq!(
            content_path(&id),
            "/Contents/registered/000000EE/f8123cc7a1c5ed87f65f85e037c0ba75.nca"
        );
    }
}
