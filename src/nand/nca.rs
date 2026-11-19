//! Nintendo Content Archives, as far as reading the content meta out of the
//! meta NCAs of a firmware and the files out of its data NCAs takes.
//!
//! An NCA starts with a 0xC00-byte header encrypted with AES-XTS under the
//! [header key](crypto::nca_header_key) in 0x200-byte sectors numbered from
//! zero, the tweak big-endian ([`Xts`]). In it (offsets into the plain
//! header):
//!
//! | offset | field |
//! |--------|-------|
//! | 0x200  | magic `NCA3` |
//! | 0x205  | content type, 1 for meta |
//! | 0x206  | key generation, old field |
//! | 0x207  | key area kek index: 0 application, 1 ocean, 2 system |
//! | 0x208  | size of the whole NCA |
//! | 0x220  | key generation |
//! | 0x230  | rights ID, zero unless the content has a title key |
//! | 0x240  | four sections: start and end in 0x200-byte units, 8 reserved |
//! | 0x280  | SHA-256 of each section's 0x200-byte header |
//! | 0x300  | key area: four AES keys sealed with the key area key |
//! | 0x400  | the four section headers |
//!
//! The key area key is that of revision `max(generation, old) - 1` (0 and 1
//! both meaning revision 0), under the master key of that number. In a
//! section header: 0x02 filesystem type (1 partition FS), 0x03 hash type
//! (2 hierarchical SHA-256), 0x04 encryption type (1 none, 3 AES-CTR), at
//! 0x08 the hash data (master hash, block size at 0x28, at 0x30 the hash
//! table and the data regions as offset/size pairs) and at 0x140 the upper
//! half of the CTR counter, stored little-endian; the lower half is the
//! absolute offset in the NCA in 16-byte blocks. A CTR section decrypts
//! with key 2 of the key area.
//!
//! This layout is the public one (switchbrew "NCA"); every meta NCA of
//! firmware 23.5.0 decrypts, passes all three hash checks below and yields
//! the same CNMT as the reference extraction in `fw_dump`.

use std::fmt;

use super::{
    bis::{MissingKey, Xts},
    crypto,
};
use crate::keys::KeySet;

pub const HEADER_SIZE: usize = 0xC00;
pub(crate) const XTS_SECTOR: u64 = 0x200;
pub(crate) const MEDIA_UNIT: u64 = 0x200;
const SECTION_COUNT: usize = 4;
pub(crate) const SECTION_HEADER_SIZE: usize = 0x200;

pub const CONTENT_TYPE_PROGRAM: u8 = 0;
pub const CONTENT_TYPE_META: u8 = 1;
pub const CONTENT_TYPE_DATA: u8 = 4;

const FS_TYPE_ROMFS: u8 = 0;
pub(crate) const FS_TYPE_PARTITION: u8 = 1;
pub(crate) const HASH_TYPE_HIERARCHICAL_SHA256: u8 = 2;
const HASH_TYPE_HIERARCHICAL_INTEGRITY: u8 = 3;
/// Levels of an `IVFC` tree in a RomFS section, the data being the last.
const INTEGRITY_LEVELS: usize = 6;
const ENCRYPTION_NONE: u8 = 1;
pub(crate) const ENCRYPTION_CTR: u8 = 3;
pub(crate) const CTR_KEY_SLOT: usize = 2;

#[derive(Debug)]
pub enum NcaError {
    MissingKey(MissingKey),
    /// The key file has no master key of the NCA's key revision.
    MissingMasterKey {
        revision: u8,
    },
    /// Shorter than its header or than a section it declares.
    Truncated,
    /// Not `NCA3` after decryption: not an NCA, or the header key is wrong.
    BadMagic,
    /// A section header does not match the hash the NCA header holds for it.
    SectionHeaderHash {
        section: usize,
    },
    /// The section's hash table or data does not match its hashes: the key
    /// area key or the counter is wrong, or the file is damaged.
    DataHash {
        section: usize,
    },
    /// Something this reader does not handle, named.
    Unsupported(&'static str),
    /// The partition FS in a section is malformed.
    Pfs0(&'static str),
}

impl fmt::Display for NcaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingKey(missing) => missing.fmt(f),
            Self::MissingMasterKey { revision } => {
                write!(f, "master_key_{revision:02x} is not in the key file")
            }
            Self::Truncated => write!(f, "the NCA is shorter than its header says"),
            Self::BadMagic => write!(f, "no NCA3 magic in the decrypted header"),
            Self::SectionHeaderHash { section } => {
                write!(f, "section {section}: header does not match its hash")
            }
            Self::DataHash { section } => {
                write!(f, "section {section}: data does not match its hashes")
            }
            Self::Unsupported(what) => write!(f, "unsupported NCA: {what}"),
            Self::Pfs0(what) => write!(f, "malformed partition FS: {what}"),
        }
    }
}

impl std::error::Error for NcaError {}

impl From<MissingKey> for NcaError {
    fn from(missing: MissingKey) -> Self {
        Self::MissingKey(missing)
    }
}

/// An NCA held in memory with its header decrypted.
pub struct Nca<'a> {
    header: Box<[u8; HEADER_SIZE]>,
    data: &'a [u8],
    keys: &'a KeySet,
}

fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn u64_at(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

impl<'a> Nca<'a> {
    pub fn open(data: &'a [u8], keys: &'a KeySet) -> Result<Self, NcaError> {
        let mut header: Box<[u8; HEADER_SIZE]> = Box::new(
            data.get(..HEADER_SIZE)
                .ok_or(NcaError::Truncated)?
                .try_into()
                .unwrap(),
        );
        Xts::new(&crypto::nca_header_key(keys)?, XTS_SECTOR).decrypt(0, &mut header[..]);
        if &header[0x200..0x204] != b"NCA3" {
            return Err(NcaError::BadMagic);
        }
        Ok(Self { header, data, keys })
    }

    pub fn content_type(&self) -> u8 {
        self.header[0x205]
    }

    pub fn size(&self) -> u64 {
        u64_at(&self.header[..], 0x208)
    }

    pub fn program_id(&self) -> u64 {
        u64_at(&self.header[..], 0x210)
    }

    /// The whole header, decrypted.
    pub fn header(&self) -> &[u8; HEADER_SIZE] {
        &self.header
    }

    fn section_header(&self, section: usize) -> &[u8] {
        let start = 0x400 + section * SECTION_HEADER_SIZE;
        &self.header[start..start + SECTION_HEADER_SIZE]
    }

    /// The header of section `section`, once it matches the hash the NCA
    /// header holds for it.
    fn checked_section_header(&self, section: usize) -> Result<&[u8], NcaError> {
        assert!(section < SECTION_COUNT);
        let header = self.section_header(section);
        let expected = &self.header[0x280 + section * 0x20..0x2A0 + section * 0x20];
        if crypto::sha256(header) != expected {
            return Err(NcaError::SectionHeaderHash { section });
        }
        Ok(header)
    }

    /// The whole of section `section`, decrypted as its `header` says.
    fn decrypted_section(&self, section: usize, header: &[u8]) -> Result<Vec<u8>, NcaError> {
        let entry = 0x240 + section * 0x10;
        let start = u64::from(u32_at(&self.header[..], entry)) * MEDIA_UNIT;
        let end = u64::from(u32_at(&self.header[..], entry + 4)) * MEDIA_UNIT;
        let mut plain = self
            .data
            .get(start as usize..end as usize)
            .ok_or(NcaError::Truncated)?
            .to_vec();
        match header[0x04] {
            ENCRYPTION_NONE => {}
            ENCRYPTION_CTR => {
                let counter = (u128::from(u64_at(header, 0x140)) << 64) | u128::from(start / 16);
                crypto::ctr_apply(&self.ctr_key()?, counter, &mut plain);
            }
            _ => {
                return Err(NcaError::Unsupported(
                    "an encryption other than none or AES-CTR",
                ));
            }
        }
        Ok(plain)
    }

    /// The plain bytes of the data region of partition FS section `section`,
    /// after checking its header, hash table and data against their hashes.
    pub fn partition_section(&self, section: usize) -> Result<Vec<u8>, NcaError> {
        let header = self.checked_section_header(section)?;
        if header[0x02] != FS_TYPE_PARTITION || header[0x03] != HASH_TYPE_HIERARCHICAL_SHA256 {
            return Err(NcaError::Unsupported(
                "a section other than a partition FS hashed with hierarchical SHA-256",
            ));
        }
        let plain = self.decrypted_section(section, header)?;
        let region = |index: usize| -> Result<&[u8], NcaError> {
            let offset = u64_at(header, 0x30 + index * 0x10) as usize;
            let size = u64_at(header, 0x38 + index * 0x10) as usize;
            plain.get(offset..offset + size).ok_or(NcaError::Truncated)
        };
        let master_hash = &header[0x08..0x28];
        let block_size = u32_at(header, 0x28) as usize;
        let (table, data) = (region(0)?, region(1)?);
        if block_size == 0
            || crypto::sha256(table) != master_hash
            || table.len() < data.len().div_ceil(block_size) * 32
            || data
                .chunks(block_size)
                .zip(table.as_chunks::<32>().0)
                .any(|(block, hash)| crypto::sha256(block) != *hash)
        {
            return Err(NcaError::DataHash { section });
        }
        Ok(data.to_vec())
    }

    /// The RomFS image in section `section`, after checking its header and
    /// every level of its integrity tree against their hashes.
    ///
    /// The tree (`IVFC`, at 0x08 of the section header): 0x08 master hash
    /// size, 0x0C level count including the master hash, at 0x10 six levels
    /// of offset (in the section), size and log2 of the block size (8, 8
    /// and 4 bytes, 4 reserved), the master hash at 0xC0. Each level holds
    /// the SHA-256 of the blocks of the next, the master hash those of the
    /// first, and the last level is the RomFS. A short last block is
    /// hashed as if filled up with zeros to the block size (switchbrew
    /// "NCA"; every level of firmware 23.5.0's boot image packages matches
    /// that way).
    pub fn romfs_section(&self, section: usize) -> Result<Vec<u8>, NcaError> {
        let header = self.checked_section_header(section)?;
        if header[0x02] != FS_TYPE_ROMFS || header[0x03] != HASH_TYPE_HIERARCHICAL_INTEGRITY {
            return Err(NcaError::Unsupported(
                "a section other than a RomFS with an integrity tree",
            ));
        }
        let ivfc = &header[0x08..];
        if &ivfc[..4] != b"IVFC" || u32_at(ivfc, 0x0C) as usize != INTEGRITY_LEVELS + 1 {
            return Err(NcaError::Unsupported(
                "an integrity tree other than IVFC with six levels",
            ));
        }
        let master_size = u32_at(ivfc, 0x08) as usize;
        let master = ivfc
            .get(0xC0..0xC0 + master_size)
            .ok_or(NcaError::Unsupported(
                "a master hash past the section header",
            ))?;
        let plain = self.decrypted_section(section, header)?;
        let mut hashes = master;
        let mut level = &plain[..0];
        for index in 0..INTEGRITY_LEVELS {
            let info = &ivfc[0x10 + index * 0x18..];
            let (offset, size) = (u64_at(info, 0) as usize, u64_at(info, 8) as usize);
            let block_order = u32_at(info, 0x10);
            level = plain
                .get(offset..offset + size)
                .ok_or(NcaError::Truncated)?;
            if !(1..32).contains(&block_order) {
                return Err(NcaError::Unsupported(
                    "an integrity block size out of range",
                ));
            }
            let block_size = 1usize << block_order;
            if hashes.len() < level.len().div_ceil(block_size) * 32 {
                return Err(NcaError::DataHash { section });
            }
            let mut padded = vec![0; block_size];
            for (block, hash) in level.chunks(block_size).zip(hashes.as_chunks::<32>().0) {
                padded[..block.len()].copy_from_slice(block);
                padded[block.len()..].fill(0);
                if crypto::sha256(&padded) != *hash {
                    return Err(NcaError::DataHash { section });
                }
            }
            hashes = level;
        }
        Ok(level.to_vec())
    }

    fn ctr_key(&self) -> Result<[u8; 16], NcaError> {
        if self.header[0x230..0x240].iter().any(|&byte| byte != 0) {
            return Err(NcaError::Unsupported("content encrypted with a title key"));
        }
        let index = usize::from(self.header[0x207]);
        if index > 2 {
            return Err(NcaError::Unsupported("a key area kek index above 2"));
        }
        let revision = self.header[0x206].max(self.header[0x220]).saturating_sub(1);
        let master_key = self
            .keys
            .master_key(revision)
            .ok_or(NcaError::MissingMasterKey { revision })?;
        let key_area_key = crypto::key_area_key(self.keys, index, master_key)?;
        let sealed = &self.header[0x300 + CTR_KEY_SLOT * 16..0x310 + CTR_KEY_SLOT * 16];
        Ok(crypto::aes_decrypt(
            &key_area_key,
            sealed.try_into().unwrap(),
        ))
    }
}

/// The files of a partition FS (`PFS0`): a header with the file count and
/// the size of the name table, 0x18-byte entries (offset and size relative
/// to the end of the name table, name offset), the names, then the data.
pub fn pfs0_files(image: &[u8]) -> Result<Vec<(&str, &[u8])>, NcaError> {
    const ENTRY_SIZE: usize = 0x18;
    if image.get(..4) != Some(b"PFS0") || image.len() < 0x10 {
        return Err(NcaError::Pfs0("no PFS0 magic"));
    }
    let count = u32_at(image, 4) as usize;
    let names_size = u32_at(image, 8) as usize;
    let names_start = 0x10 + count * ENTRY_SIZE;
    let data_start = names_start + names_size;
    let names = image
        .get(names_start..data_start)
        .ok_or(NcaError::Pfs0("the name table is past the end"))?;
    (0..count)
        .map(|index| {
            let entry = 0x10 + index * ENTRY_SIZE;
            let offset = data_start + u64_at(image, entry) as usize;
            let size = u64_at(image, entry + 8) as usize;
            let name = names
                .get(u32_at(image, entry + 16) as usize..)
                .and_then(|rest| rest.split(|&byte| byte == 0).next())
                .and_then(|name| std::str::from_utf8(name).ok())
                .ok_or(NcaError::Pfs0("a file name is not a string in the table"))?;
            let data = image
                .get(offset..offset + size)
                .ok_or(NcaError::Pfs0("a file is past the end"))?;
            Ok((name, data))
        })
        .collect()
}
