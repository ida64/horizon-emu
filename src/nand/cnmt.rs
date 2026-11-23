//! Content meta: the packaged form a meta NCA carries (a `.cnmt` file) and
//! the record the content manager (NCM) keeps for it in its database.
//!
//! Packaged, after a 0x20-byte header (id, version, type, platform,
//! extended header size, content count, content meta count, attributes,
//! storage ID, install type, committed flag, required download system
//! version), come the extended header, 0x38-byte content infos (a SHA-256
//! of the content, then the 0x18-byte [`ContentInfo`]), 0x10-byte content
//! meta infos, extended data and a 0x20-byte digest.
//!
//! The database keeps it as `key = ContentMetaKey` (id, version, type,
//! install type, two bytes of padding) and as value an 8-byte header
//! (extended header size, content count, content meta count, attributes,
//! platform), the extended header, the content info of the meta NCA itself,
//! the packaged content infos without their hashes (and, for patches,
//! without delta fragments), and the content meta infos; extended data and
//! digest are not kept. That is Atmosphère's
//! `PackagedContentMetaReader::ConvertToContentMeta`; the guest's own NCM
//! accepts records in this form, finds the programs of firmware 23.5.0 by
//! them and lets Loader start boot2 from them.

use std::fmt;

/// Size of the packaged header and of the database's key.
const PACKAGED_HEADER_SIZE: usize = 0x20;
const PACKAGED_CONTENT_INFO_SIZE: usize = 0x38;
pub const CONTENT_INFO_SIZE: usize = 0x18;
const CONTENT_META_INFO_SIZE: usize = 0x10;

pub const CONTENT_TYPE_META: u8 = 0;
pub const CONTENT_TYPE_DATA: u8 = 2;
pub const CONTENT_TYPE_DELTA_FRAGMENT: u8 = 6;
pub const META_TYPE_SYSTEM_UPDATE: u8 = 3;
pub const META_TYPE_PATCH: u8 = 0x81;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CnmtError {
    /// The file ends before the parts its header declares.
    Truncated,
}

impl fmt::Display for CnmtError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => write!(f, "the content meta is shorter than its header says"),
        }
    }
}

impl std::error::Error for CnmtError {}

/// The database key. Fields in the order the database sorts by.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ContentMetaKey {
    pub id: u64,
    pub version: u32,
    pub meta_type: u8,
    pub install_type: u8,
}

impl ContentMetaKey {
    pub fn to_bytes(&self) -> [u8; 0x10] {
        let mut bytes = [0; 0x10];
        bytes[..8].copy_from_slice(&self.id.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.version.to_le_bytes());
        bytes[12] = self.meta_type;
        bytes[13] = self.install_type;
        bytes
    }

    /// The key as the database stores it, [`Self::to_bytes`]'s inverse.
    pub fn from_bytes(bytes: &[u8; 0x10]) -> Self {
        Self {
            id: u64::from_le_bytes(bytes[..8].try_into().unwrap()),
            version: u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            meta_type: bytes[12],
            install_type: bytes[13],
        }
    }
}

/// Where one content is and what it is: its ID (the first half of its
/// SHA-256), its size in 40 bits, attributes, type and ID offset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContentInfo {
    pub id: [u8; 16],
    pub size: u64,
    pub attributes: u8,
    pub content_type: u8,
    pub id_offset: u8,
}

impl ContentInfo {
    pub fn to_bytes(&self) -> [u8; CONTENT_INFO_SIZE] {
        let mut bytes = [0; CONTENT_INFO_SIZE];
        bytes[..16].copy_from_slice(&self.id);
        bytes[16..21].copy_from_slice(&self.size.to_le_bytes()[..5]);
        bytes[21] = self.attributes;
        bytes[22] = self.content_type;
        bytes[23] = self.id_offset;
        bytes
    }

    pub fn from_bytes(bytes: &[u8; CONTENT_INFO_SIZE]) -> Self {
        let mut size = [0; 8];
        size[..5].copy_from_slice(&bytes[16..21]);
        Self {
            id: bytes[..16].try_into().unwrap(),
            size: u64::from_le_bytes(size),
            attributes: bytes[21],
            content_type: bytes[22],
            id_offset: bytes[23],
        }
    }
}

/// A title a content meta names, as a system update names the titles it
/// is made of: ID, version, meta type, attributes, two bytes of padding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContentMetaInfo {
    pub id: u64,
    pub version: u32,
    pub meta_type: u8,
    pub attributes: u8,
}

/// A packaged content meta, bounds-checked.
pub struct PackagedContentMeta<'a> {
    bytes: &'a [u8],
}

impl<'a> PackagedContentMeta<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self, CnmtError> {
        let meta = Self { bytes };
        if bytes.len() < PACKAGED_HEADER_SIZE || bytes.len() < meta.content_meta_infos_end() {
            return Err(CnmtError::Truncated);
        }
        Ok(meta)
    }

    fn u16_at(&self, offset: usize) -> usize {
        usize::from(u16::from_le_bytes([
            self.bytes[offset],
            self.bytes[offset + 1],
        ]))
    }

    pub fn key(&self) -> ContentMetaKey {
        let bytes = self.bytes;
        ContentMetaKey {
            id: u64::from_le_bytes(bytes[..8].try_into().unwrap()),
            version: u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            meta_type: bytes[0x0C],
            install_type: bytes[0x16],
        }
    }

    fn extended_header_size(&self) -> usize {
        self.u16_at(0x0E)
    }

    fn content_count(&self) -> usize {
        self.u16_at(0x10)
    }

    fn content_meta_count(&self) -> usize {
        self.u16_at(0x12)
    }

    fn content_infos_start(&self) -> usize {
        PACKAGED_HEADER_SIZE + self.extended_header_size()
    }

    /// The extended header of the meta type: for an application its patch
    /// ID and required system version, for a patch or an add-on the
    /// application's ID first.
    pub fn extended_header(&self) -> &'a [u8] {
        &self.bytes[PACKAGED_HEADER_SIZE..self.content_infos_start()]
    }

    fn content_meta_infos_start(&self) -> usize {
        self.content_infos_start() + self.content_count() * PACKAGED_CONTENT_INFO_SIZE
    }

    fn content_meta_infos_end(&self) -> usize {
        self.content_meta_infos_start() + self.content_meta_count() * CONTENT_META_INFO_SIZE
    }

    /// The content infos without their hashes, in their order.
    fn content_infos(&self) -> impl Iterator<Item = &'a [u8]> {
        let start = self.content_infos_start();
        self.bytes[start..self.content_meta_infos_start()]
            .as_chunks::<PACKAGED_CONTENT_INFO_SIZE>()
            .0
            .iter()
            .map(|info| &info[PACKAGED_CONTENT_INFO_SIZE - CONTENT_INFO_SIZE..])
    }

    /// The content infos with the SHA-256 of each content, in their order.
    pub fn packaged_contents(&self) -> impl Iterator<Item = ([u8; 32], ContentInfo)> + 'a {
        let start = self.content_infos_start();
        self.bytes[start..self.content_meta_infos_start()]
            .as_chunks::<PACKAGED_CONTENT_INFO_SIZE>()
            .0
            .iter()
            .map(|info| {
                let (hash, info) = info.split_first_chunk::<32>().unwrap();
                (*hash, ContentInfo::from_bytes(info.try_into().unwrap()))
            })
    }

    /// The titles this content meta names, in their order.
    pub fn content_meta_infos(&self) -> impl Iterator<Item = ContentMetaInfo> + 'a {
        self.bytes[self.content_meta_infos_start()..self.content_meta_infos_end()]
            .as_chunks::<CONTENT_META_INFO_SIZE>()
            .0
            .iter()
            .map(|info| ContentMetaInfo {
                id: u64::from_le_bytes(info[..8].try_into().unwrap()),
                version: u32::from_le_bytes(info[8..12].try_into().unwrap()),
                meta_type: info[12],
                attributes: info[13],
            })
    }

    /// The database value of this content meta, `meta` being the content
    /// info of the NCA it came in.
    pub fn database_value(&self, meta: &ContentInfo) -> Vec<u8> {
        let skip_deltas = self.key().meta_type == META_TYPE_PATCH;
        let infos: Vec<&[u8]> = self
            .content_infos()
            .filter(|info| !(skip_deltas && info[22] == CONTENT_TYPE_DELTA_FRAGMENT))
            .collect();
        let content_count = u16::try_from(infos.len() + 1).expect("at most 0xFFFF in the package");
        let mut value = Vec::new();
        value.extend_from_slice(&self.bytes[0x0E..0x10]);
        value.extend_from_slice(&content_count.to_le_bytes());
        value.extend_from_slice(&self.bytes[0x12..0x14]);
        value.push(self.bytes[0x14]);
        value.push(self.bytes[0x0D]);
        value.extend_from_slice(&self.bytes[PACKAGED_HEADER_SIZE..self.content_infos_start()]);
        value.extend_from_slice(&meta.to_bytes());
        for info in infos {
            value.extend_from_slice(info);
        }
        value.extend_from_slice(
            &self.bytes[self.content_meta_infos_start()..self.content_meta_infos_end()],
        );
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(text: &str) -> Vec<u8> {
        (0..text.len())
            .step_by(2)
            .map(|at| u8::from_str_radix(&text[at..at + 2], 16).unwrap())
            .collect()
    }

    /// boot2 (0100000000000008) of firmware 23.5.0: meta NCA
    /// 38ac7257e3e822af9d4eb6f22d778826 (0xE00 bytes), one program NCA.
    #[test]
    fn boot2_record_is_the_one_ncm_accepted() {
        let packaged = hex(concat!(
            "0800000000000001b004005c01000000010000000000000000000000000000",
            "009d7556bb740ee7a90e972a8416e25f107744680e1f7cdb4cf103fe780f3a",
            "17e99d7556bb740ee7a90e972a8416e25f1000a6010000000100",
            "23efee309cd34f454a42e629132a2b1df6b18650fc66346ebf74f03efe21ba3e",
        ));
        let meta = PackagedContentMeta::parse(&packaged).unwrap();
        let key = meta.key();
        assert_eq!(
            key.to_bytes().to_vec(),
            hex("0800000000000001b004005c01000000")
        );
        let program_id: [u8; 16] = hex("9d7556bb740ee7a90e972a8416e25f10").try_into().unwrap();
        let contents: Vec<_> = meta.packaged_contents().collect();
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0].0[..16], program_id);
        assert_eq!(
            contents[0].1,
            ContentInfo {
                id: program_id,
                size: 0x1A600,
                attributes: 0,
                content_type: 1,
                id_offset: 0,
            }
        );
        assert_eq!(meta.content_meta_infos().count(), 0);
        let meta_info = ContentInfo {
            id: hex("38ac7257e3e822af9d4eb6f22d778826").try_into().unwrap(),
            size: 0xE00,
            attributes: 0,
            content_type: CONTENT_TYPE_META,
            id_offset: 0,
        };
        assert_eq!(
            meta.database_value(&meta_info),
            hex(concat!(
                "0000020000000000",
                "38ac7257e3e822af9d4eb6f22d778826000e000000000000",
                "9d7556bb740ee7a90e972a8416e25f1000a6010000000100",
            ))
        );
    }

    #[test]
    fn a_header_promising_more_than_the_file_holds_is_refused() {
        let mut packaged = vec![0; 0x20];
        packaged[0x10] = 1;
        assert_eq!(
            PackagedContentMeta::parse(&packaged).err(),
            Some(CnmtError::Truncated)
        );
    }
}
