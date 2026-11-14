//! The save data indexer: FS's own system save `8000000000000000`, which
//! lists every other save it created, keyed by `SaveDataAttribute`.
//!
//! FS opens a system save by ID only if the indexer lists it. With NCM's
//! database save in `SYSTEM:/save` and no indexer save, FS answered NCM's
//! `OpenSaveDataFileSystemBySystemSaveDataId` (fsp-srv 52) for
//! `0x8000000000000120` with 2002-1002 (target not found), and NCM created
//! the save anew (fsp-srv 23), losing the database. With the indexer below
//! beside it, the open succeeds, and so does NCM's read of the save's extra
//! data (fsp-srv 58).
//!
//! The indexer the guest's FS made on a first start, after NCM had created
//! its database save, holds two files, created in this order:
//! `lastPublishedId` (eight zero bytes; by its name the last dynamic save
//! ID handed out, none yet) and `imkvdb.arc`, a key-value archive (see
//! [`content_db::key_value_archive`]) with one entry:
//!
//! * key, 0x40 bytes: program ID 0, user ID 0 (16 bytes), the system save
//!   data ID at 0x18, type 0 (system) at 0x20, rank and index 0;
//! * value, 0x40 bytes: the save data ID, then the size of the save's image
//!   file at 8, the rest 0 (by LibHac's `SaveDataIndexerValue`, recalled
//!   and not checked here: space ID system and state normal).
//!
//! Its own save is 0xC0000 bytes of data and of journal, owner 0, flags 0.
//! Entries are kept in order of the system save ID, which with every other
//! key field equal is the order under any comparison of the fields.

use super::{
    content_db,
    save::{self, Entry, SaveError, SystemSave},
};
use crate::keys::KeySet;

pub const SAVE_DATA_ID: u64 = 0x8000_0000_0000_0000;
const SAVE_SIZE: u64 = 0xC0000;
const KEY_SIZE: usize = 0x40;
const VALUE_SIZE: usize = 0x40;

/// A system save the indexer is to list: its ID and the size of its image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexedSave {
    pub id: u64,
    pub image_size: u64,
}

/// The index file listing `saves`.
pub fn archive(saves: &[IndexedSave]) -> Vec<u8> {
    let mut saves = saves.to_vec();
    saves.sort_by_key(|save| save.id);
    let records: Vec<([u8; KEY_SIZE], [u8; VALUE_SIZE])> = saves
        .iter()
        .map(|save| {
            let mut key = [0; KEY_SIZE];
            key[0x18..0x20].copy_from_slice(&save.id.to_le_bytes());
            let mut value = [0; VALUE_SIZE];
            value[..8].copy_from_slice(&save.id.to_le_bytes());
            value[8..16].copy_from_slice(&save.image_size.to_le_bytes());
            (key, value)
        })
        .collect();
    content_db::key_value_archive(records.iter().map(|(key, value)| (&key[..], &value[..])))
}

/// The indexer's save image, listing `saves`.
pub fn save(saves: &[IndexedSave], keys: &KeySet) -> Result<Vec<u8>, SaveError> {
    save::system_save(
        &SystemSave {
            id: SAVE_DATA_ID,
            owner_id: 0,
            flags: 0,
            size: SAVE_SIZE,
        },
        &[
            Entry::File("lastPublishedId".into(), vec![0; 8]),
            Entry::File("imkvdb.arc".into(), archive(saves)),
        ],
        keys,
    )
}
