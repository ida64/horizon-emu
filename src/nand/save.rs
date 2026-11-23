//! Save data images as the filesystem process (FS) keeps them in
//! `SYSTEM:/save/<id>`: version 5 of the journaled, integrity-verified
//! format with the `DISF` header.
//!
//! # Layers
//!
//! From the file inwards:
//!
//! * Two copies of a 0x4000-byte header at 0 and 0x4000. Bytes 0x300 to
//!   0x4000 are hashed with SHA-256 into 0x108, and bytes 0x100 to 0x300
//!   (the `DISF` layout, which holds that hash) are authenticated by the
//!   AES-CMAC at 0 under the [save MAC key](crypto::save_mac_key).
//! * The main remap storage (`RMAP` at 0x650, its entries at the offset the
//!   layout names): five segments of the file area after 0x10000, the
//!   virtual address holding the segment in its top four bits.
//! * A hierarchical duplex storage: a master bitmap in the header, then a
//!   level-1 bitmap and the data, each in two copies (segments 0 to 3); a
//!   set bit selects copy B of the block below it.
//! * The meta remap storage (`RMAP` at 0x690, top five bits) over the duplex
//!   data: the journal's map table and three bitmaps, the allocation table,
//!   and the hash levels of both integrity layers.
//! * The journal (`JNGL` at 0x408, segment 4 of the main remap storage):
//!   virtual blocks mapped to physical ones through the map table, entries
//!   of two words, the physical index with bit 31 set and a zero word.
//! * Integrity verification (`IVFC` at 0x344 over the journal, at 0xAD8 over
//!   the allocation table): levels of salted SHA-256 hashes, the master hash
//!   in the header. A block hashes to `SHA-256(salt || block)` with bit 7 of
//!   the last byte set; the salt of the hashes stored in level `n` (0 being
//!   the master hash) is `HMAC-SHA-256(key = "HierarchicalIntegrity
//!   VerificationStorage::" + ["Master", "L1", "L2", …][n], message = the
//!   32-byte seed after the level table)`; an all-zero hash stands for a
//!   block never written, which reads as zeros whatever its bytes are (FS's
//!   "cleared" block, LibHac's and Atmosphère's `IsCleared` rule), at every
//!   level: a hash block never written makes all the blocks below it read
//!   as never written.
//! * The file system (`SAVE` at 0x608 with the allocation table header at
//!   0x620): a FAT-like allocation table of 0x4000-byte blocks and two
//!   tables of 0x60-byte entries (parent index, 0x40-byte name, next
//!   sibling, a 0x14-byte value, next in the table's list) for directories
//!   and files, each a chain of blocks. Entry 0 of a table holds its length
//!   and capacity and heads the free list, entry 1 heads the list of
//!   entries in use, entry 2 of the directory table is the root.
//!
//! This is the structure LibHac reads (`Tools/FsSystem/Save`). Every hash
//! and the CMAC of six saves the guest's FS wrote itself check out with it
//! ([`open`]): NCM's empty database, NCM's database after an import (one
//! directory, one file), the save data indexer (two files), and on a
//! console that went through its first-time setup ES's common tickets
//! (`80000000000000e1`, data 20000000 and journal 10000000 bytes), NCM's
//! empty user database (`8000000000000121`) and NS's application records
//! (`8000000000000040`, data 0x80C8000, journal 0x60000). The last three
//! hold garbage in most blocks: blocks of the data never written (1216 of
//! 1221 in the tickets' save, whose 16 MB `ticket.bin` is allocated but
//! never written), allocation table blocks, and hash blocks whose own hash
//! is zero, so that the stale, non-zero hashes in them (for data blocks
//! 512 on in the tickets' and the records' save) are never checked. Read
//! with the all-zero rule above, not a block fails.
//!
//! Which copy is current: in all six both header copies are equal and the
//! duplex index (layout word 38) is 0. It selects duplex master bitmap A,
//! and with it the master hashes of both integrity layers are the A
//! copies: in the three later saves, bitmap B and the B master hashes
//! form another consistent state, an earlier commit's, and the B master
//! hashes fail under bitmap A (A ones under bitmap B). So the index is
//! taken to select bitmap and master hashes alike \[inference: index 1
//! never seen\]. FS hashes every block it allocated, a block of zeros too
//! (the indexer's `lastPublishedId`), and leaves the others' hashes zero.
//!
//! # Editing a save the guest made
//!
//! [`SaveEditor`] changes an existing save of any geometry the reader
//! takes, through the file system as FS would (LibHac's
//! `SaveDataFileSystemCore`): new files take blocks off the front of the
//! allocation table's free list (`AllocationTable::Allocate`, splitting a
//! segment where needed) and deleted files give theirs back to its front
//! (`Free`); new entries come off a table's free entry list, else after its
//! last entry, and go first in the list of entries in use and first among
//! their parent's children. Tables are not grown: a full one refuses the
//! entry.
//!
//! The commit keeps everything else as it was. A changed data block goes
//! into the physical block the journal already maps it to (the map does
//! not change), its hash into the journal's integrity layer; changed
//! allocation table and hash blocks go into the duplex copies they were
//! read from (the bitmaps do not change). A block that was never written
//! is written whole, zeros around the new bytes. Both header copies get
//! the new master hashes, header hash and CMAC, and the non-current duplex
//! bitmap and master hashes become copies of the current ones, so that
//! either state reads the new contents; the extra data (commit ID
//! included) stays. Every other byte of the image stays as it was.
//!
//! # Producing a save
//!
//! [`build`] lays out a save from scratch rather than patching an embedded
//! save the guest made: what varies between saves (sizes, extra data,
//! files) is spread over every layer and both header copies, and a template
//! would still need all of the writer below to re-hash and re-sign it,
//! while adding a megabyte of mostly zero, console-specific bytes. The
//! layout is the one FS formats, as far as the samples pin it down: one
//! block per metadata region, which they show for saves of 27 and 48 blocks
//! and which holds up to [`MAX_BLOCKS`]; beyond that FS sizes regions by a
//! rule the samples do not show, so larger saves are refused. Data and
//! journal sizes are equal in all three samples, and the two header words
//! that would tell them apart cannot be told apart there, so they must be
//! equal here too.
//!
//! The image is fully committed: both header copies, both duplex copies
//! and both master copies equal, the duplex bitmaps clear, the journal
//! mapping each virtual block to the physical block of the same index, the
//! blocks after them free. Made for the files of the guest's NCM database
//! save and of its indexer save, the journaled data, the allocation table
//! and its integrity levels are byte for byte the guest's; the headers
//! differ in the CMAC and header hash, the salt seed of the journal's
//! integrity layer and the hashes that depend on it, the commit IDs, and
//! the duplex index and B copies that the guest's later commits moved.
//! Salt seed and commit ID are random in FS (the samples' are unrelated);
//! here they are derived from the contents, so that the same input gives
//! the same image. Regions FS never wrote hold random bytes in the samples
//! (remap entry tables past their entries, bitmaps past their blocks) and
//! zeros here. Zero-length files are stored with start block
//! `0x8000_0000`, LibHac's convention; none of the samples has one.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, btree_map};
use std::fmt;

use sha2::{Digest, Sha256};

use super::{bis::MissingKey, crypto};
use crate::keys::{Key, KeySet};

/// Block of the journal, the file system and every hashed level.
pub const BLOCK_SIZE: usize = 0x4000;
/// Most data blocks a save made here may have: the journal's map table
/// (eight bytes per block) still fits the 0x200 bytes it has in the
/// samples.
pub const MAX_BLOCKS: usize = 0x200 / 8;

const HEADER_SIZE: usize = 0x4000;
const DISF_MAGIC: &[u8; 4] = b"DISF";
const DISF_VERSION: u32 = 0x5_0000;
const CMAC_RANGE: std::ops::Range<usize> = 0x100..0x300;
const HASHED_RANGE: std::ops::Range<usize> = 0x300..HEADER_SIZE;
const HEADER_HASH: usize = 0x108;
/// The layout words after the header hash, indexed by the `L_*` constants.
const LAYOUT: usize = 0x128;
const DUPLEX_HEADER: usize = 0x300;
const IVFC_HEADER: usize = 0x344;
const JOURNAL_HEADER: usize = 0x408;
const SAVE_HEADER: usize = 0x608;
const FAT_HEADER: usize = 0x620;
const MAIN_REMAP_HEADER: usize = 0x650;
const META_REMAP_HEADER: usize = 0x690;
const EXTRA_DATA: [usize; 2] = [0x6D8, 0x8D8];
const EXTRA_DATA_SIZE: usize = 0x200;
const FAT_IVFC_HEADER: usize = 0xAD8;

const L_MAIN_MAP_ENTRIES: usize = 0;
const L_META_MAP_ENTRIES: usize = 2;
const L_MAIN_DATA: usize = 4;
const L_DUPLEX_L1_A: usize = 6;
const L_DUPLEX_L1_B: usize = 7;
const L_DUPLEX_L1_SIZE: usize = 8;
const L_DUPLEX_DATA_A: usize = 9;
const L_DUPLEX_DATA_B: usize = 10;
const L_DUPLEX_DATA_SIZE: usize = 11;
const L_JOURNAL_DATA: usize = 12;
const L_JOURNAL_DATA_SIZE_A: usize = 13;
const L_JOURNAL_DATA_SIZE_B: usize = 14;
const L_JOURNAL_SIZE: usize = 15;
const L_DUPLEX_MASTER_A: usize = 16;
const L_DUPLEX_MASTER_B: usize = 17;
const L_DUPLEX_MASTER_SIZE: usize = 18;
const L_IVFC_MASTER_A: usize = 19;
const L_IVFC_MASTER_B: usize = 20;
const L_IVFC_MASTER_SIZE: usize = 21;
const L_JOURNAL_TABLE: usize = 22;
const L_JOURNAL_PHYSICAL_BITMAP: usize = 24;
const L_JOURNAL_VIRTUAL_BITMAP: usize = 26;
const L_JOURNAL_FREE_BITMAP: usize = 28;
const L_IVFC_L1: usize = 30;
const L_IVFC_L2: usize = 32;
const L_IVFC_L3: usize = 34;
const L_FAT: usize = 36;
const L_DUPLEX_INDEX: usize = 38;
const L_FAT_IVFC_MASTER_A: usize = 39;
const L_FAT_IVFC_MASTER_B: usize = 40;
const L_FAT_IVFC_L1: usize = 41;
const L_FAT_IVFC_L2: usize = 43;
const LAYOUT_WORDS: usize = 45;

/// Where the header places the masters (duplex bitmaps, then the master
/// hashes of the two integrity layers), each A then B.
const DUPLEX_MASTER: [usize; 2] = [0xC00, 0xC40];
const DUPLEX_MASTER_SIZE: usize = 0x40;
const IVFC_MASTER: [usize; 2] = [0xC80, 0xCA0];
const FAT_IVFC_MASTER: [usize; 2] = [0xCC0, 0xCE0];
const HASH_SIZE: usize = 0x20;

/// The file after the two headers: the entry tables of the two remap
/// storages, then the area the main one maps.
const MAIN_MAP_ENTRIES: usize = 0x8000;
const META_MAP_ENTRIES: usize = 0xC000;
const MAP_ENTRIES_SIZE: usize = 0x4000;
const MAIN_DATA: usize = 0x10000;
const MAP_ENTRY_SIZE: usize = 0x20;
const MAIN_SEGMENT_BITS: u32 = 4;
const META_SEGMENT_BITS: u32 = 5;

const DUPLEX_L1_SIZE: usize = 0x4000;
/// The meta area (0x18800 bytes below) in whole blocks.
const DUPLEX_DATA_SIZE: usize = 0x1C000;

/// The regions of the meta area, in segment order, as (offset, size) in
/// the duplex data.
const META_REGIONS: [(usize, usize); 10] = [
    (0x0, 0x200),      // journal map table
    (0x200, 0x200),    // journal: physical blocks modified
    (0x400, 0x200),    // journal: virtual blocks modified
    (0x600, 0x200),    // journal: free physical blocks
    (0x800, 0x4000),   // allocation table
    (0x4800, 0x4000),  // journal integrity level 1
    (0x8800, 0x4000),  // level 2
    (0xC800, 0x4000),  // level 3
    (0x10800, 0x4000), // allocation table integrity level 1
    (0x14800, 0x4000), // level 2
];
const META_JOURNAL_TABLE: usize = 0;
const META_FREE_BITMAP: usize = 3;
const META_FAT: usize = 4;
const META_IVFC: [usize; 3] = [5, 6, 7];
const META_FAT_IVFC: [usize; 2] = [8, 9];

const TABLE_ENTRY_SIZE: usize = 0x60;
const NAME_SIZE: usize = 0x40;
const ROOT_INDEX: u32 = 2;
const EMPTY_FILE_START: u32 = 0x8000_0000;
const CHAIN_FLAG: u32 = 0x8000_0000;

const SALT_SOURCES: [&str; 4] = [
    "HierarchicalIntegrityVerificationStorage::Master",
    "HierarchicalIntegrityVerificationStorage::L1",
    "HierarchicalIntegrityVerificationStorage::L2",
    "HierarchicalIntegrityVerificationStorage::L3",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SaveError {
    MissingKey(MissingKey),
    /// Sizes this writer has no established layout for (see the module
    /// documentation).
    UnsupportedSize {
        size: u64,
        journal_size: u64,
    },
    /// The entries need more blocks than the save has.
    Full {
        needed: usize,
        blocks: usize,
    },
    /// An entry path that is empty, has an empty, `.` or `..` component or
    /// one longer than 0x40 bytes, or names both a file and a directory.
    BadPath(String),
    /// The image ends before a region its header places.
    Truncated(&'static str),
    BadMagic(&'static str),
    /// Neither header copy matches its hash.
    HeaderHash,
    /// The header's CMAC does not match: another console's save, or a
    /// damaged one.
    Cmac,
    /// A block of an integrity layer does not match its hash.
    Hash {
        layer: &'static str,
        level: usize,
        block: usize,
    },
    /// A structure contradicts itself, named.
    Corrupt(&'static str),
    /// No entry by that path, or not of the kind needed.
    NotFound(String),
    AlreadyExists(String),
    /// A write would run past the end of the file.
    PastEnd(String),
    /// The allocation table has fewer free blocks than a new file needs.
    NoSpace {
        needed: usize,
        free: usize,
    },
    /// The table a new entry goes into is full (FS would grow it; this
    /// editor does not).
    TableFull(String),
}

impl fmt::Display for SaveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingKey(missing) => missing.fmt(f),
            Self::UnsupportedSize { size, journal_size } => write!(
                f,
                "no known layout for a save of {size:#x} bytes with a {journal_size:#x}-byte \
                 journal (equal multiples of {BLOCK_SIZE:#x}, at most {MAX_BLOCKS} blocks)"
            ),
            Self::Full { needed, blocks } => {
                write!(f, "the entries need {needed} blocks, the save has {blocks}")
            }
            Self::BadPath(path) => write!(f, "unusable save entry path {path:?}"),
            Self::Truncated(what) => write!(f, "save image ends before its {what}"),
            Self::BadMagic(what) => write!(f, "save image: bad {what} magic"),
            Self::HeaderHash => write!(f, "save image: no header copy matches its hash"),
            Self::Cmac => write!(f, "save image: header CMAC mismatch"),
            Self::Hash {
                layer,
                level,
                block,
            } => write!(
                f,
                "save image: {layer} level {level} block {block} fails its hash"
            ),
            Self::Corrupt(what) => write!(f, "save image: {what}"),
            Self::NotFound(path) => write!(f, "save: no entry {path:?} of that kind"),
            Self::AlreadyExists(path) => write!(f, "save: {path:?} exists already"),
            Self::PastEnd(path) => write!(f, "save: write past the end of {path:?}"),
            Self::NoSpace { needed, free } => {
                write!(f, "save: {needed} blocks needed, {free} free")
            }
            Self::TableFull(path) => write!(f, "save: no room in the table for {path:?}"),
        }
    }
}

impl std::error::Error for SaveError {}

impl From<MissingKey> for SaveError {
    fn from(missing: MissingKey) -> Self {
        Self::MissingKey(missing)
    }
}

/// What the header's extra data records about a save (the fields FS's
/// `SaveDataExtraData` has values for in system saves).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtraData {
    pub save_data_id: u64,
    pub owner_id: u64,
    pub flags: u32,
    pub size: u64,
    pub journal_size: u64,
}

/// An entry to create, or one read back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Entry {
    /// A directory; its parents need not be listed.
    Directory(String),
    File(String, Vec<u8>),
}

impl Entry {
    fn path(&self) -> &str {
        match self {
            Self::Directory(path) | Self::File(path, _) => path,
        }
    }
}

/// A system save: extra data of type System for user 0 and program 0.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SystemSave {
    pub id: u64,
    pub owner_id: u64,
    pub flags: u32,
    /// Size of the data and of the journal alike.
    pub size: u64,
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

/// Bit `index` of a bitmap stored as little-endian words, most significant
/// bit first.
fn bit(bitmap: &[u8], index: usize) -> bool {
    read_u32(bitmap, index / 32 * 4) >> (31 - index % 32) & 1 != 0
}

fn set_bit(bitmap: &mut [u8], index: usize) {
    let word = read_u32(bitmap, index / 32 * 4) | 1 << (31 - index % 32);
    put_u32(bitmap, index / 32 * 4, word);
}

fn level_salts(seed: &[u8]) -> [[u8; 32]; 4] {
    SALT_SOURCES.map(|source| crypto::hmac_sha256(source.as_bytes(), seed))
}

/// The hash of `block`, padded with zeros to `size` bytes, as a level
/// salted with `salt` stores it.
fn salted_hash(salt: &[u8; 32], block: &[u8], size: usize) -> [u8; HASH_SIZE] {
    let mut hasher = Sha256::new();
    hasher.update(salt);
    hasher.update(block);
    let mut padding = size.saturating_sub(block.len());
    while padding > 0 {
        let length = padding.min(BLOCK_SIZE);
        hasher.update(&ZERO_BLOCK[..length]);
        padding -= length;
    }
    let mut hash: [u8; HASH_SIZE] = hasher.finalize().into();
    hash[HASH_SIZE - 1] |= 0x80;
    hash
}

// Writing.

/// The image of system save `save` holding `entries`, created in their
/// order, under the save MAC key of `keys`.
pub fn system_save(
    save: &SystemSave,
    entries: &[Entry],
    keys: &KeySet,
) -> Result<Vec<u8>, SaveError> {
    build(
        &ExtraData {
            save_data_id: save.id,
            owner_id: save.owner_id,
            flags: save.flags,
            size: save.size,
            journal_size: save.size,
        },
        entries,
        keys,
    )
}

/// The image of a save with extra data `extra` holding `entries`.
pub fn build(extra: &ExtraData, entries: &[Entry], keys: &KeySet) -> Result<Vec<u8>, SaveError> {
    let mac_key = crypto::save_mac_key(keys)?;
    let unsupported = SaveError::UnsupportedSize {
        size: extra.size,
        journal_size: extra.journal_size,
    };
    let block = BLOCK_SIZE as u64;
    if extra.size != extra.journal_size || !extra.size.is_multiple_of(block) {
        return Err(unsupported);
    }
    let blocks = (extra.size / block) as usize;
    // Two blocks for the directory and file tables at least.
    if !(2..=MAX_BLOCKS).contains(&blocks) {
        return Err(unsupported);
    }
    let journal_blocks = blocks;

    let (core, fat, tables, used_blocks) = file_system(blocks, entries)?;

    // The meta area, the duplex storage's data.
    let mut meta = vec![0; DUPLEX_DATA_SIZE];
    let region = |index: usize| {
        let (offset, size) = META_REGIONS[index];
        offset..offset + size
    };
    let table = &mut meta[region(META_JOURNAL_TABLE)];
    for virtual_block in 0..blocks {
        put_u32(table, virtual_block * 8, CHAIN_FLAG | virtual_block as u32);
    }
    let free = &mut meta[region(META_FREE_BITMAP)];
    for physical_block in blocks..blocks + journal_blocks {
        set_bit(free, physical_block);
    }
    meta[region(META_FAT)].copy_from_slice(&fat);

    // Integrity levels: hashes of the data in level 3, of level 3 in level
    // 2, and so on up to the master hash.
    let seed = crypto::sha256(&[&extra.save_data_id.to_le_bytes()[..], &core].concat());
    let salts = level_salts(&seed);
    let ivfc_master = hash_levels(&mut meta, &META_IVFC, &salts, &core, used_blocks);
    // The samples' seed for the allocation table's layer is zero.
    let fat_master = hash_levels(&mut meta, &META_FAT_IVFC, &level_salts(&[0; 32]), &fat, 1);

    // The file.
    let journal_area = (blocks + journal_blocks) * BLOCK_SIZE;
    let main_size = 2 * DUPLEX_L1_SIZE + 2 * DUPLEX_DATA_SIZE + journal_area;
    let mut image = vec![0; MAIN_DATA + main_size];
    let main_segments = [
        DUPLEX_L1_SIZE,
        DUPLEX_L1_SIZE,
        DUPLEX_DATA_SIZE,
        DUPLEX_DATA_SIZE,
        journal_area,
    ];
    let mut physical = 0;
    for (segment, &size) in main_segments.iter().enumerate() {
        let entry = MAIN_MAP_ENTRIES + segment * MAP_ENTRY_SIZE;
        // The journal's entry is aligned to its blocks, the others to 0x200.
        let alignment = if segment == 4 { BLOCK_SIZE } else { 0x200 };
        write_map_entry(
            &mut image[entry..],
            (segment as u64) << (64 - MAIN_SEGMENT_BITS),
            physical as u64,
            size as u64,
            alignment as u32,
        );
        physical += size;
    }
    for (segment, &(offset, size)) in META_REGIONS.iter().enumerate() {
        let entry = META_MAP_ENTRIES + segment * MAP_ENTRY_SIZE;
        write_map_entry(
            &mut image[entry..],
            (segment as u64) << (64 - META_SEGMENT_BITS),
            offset as u64,
            size as u64,
            0x200,
        );
    }
    let duplex_data = MAIN_DATA + 2 * DUPLEX_L1_SIZE;
    image[duplex_data..duplex_data + DUPLEX_DATA_SIZE].copy_from_slice(&meta);
    image[duplex_data + DUPLEX_DATA_SIZE..duplex_data + 2 * DUPLEX_DATA_SIZE]
        .copy_from_slice(&meta);
    let journal = duplex_data + 2 * DUPLEX_DATA_SIZE;
    image[journal..journal + core.len()].copy_from_slice(&core);

    let commit_id = u64::from_le_bytes(ivfc_master[..8].try_into().unwrap());
    let header = header(
        extra,
        blocks,
        journal_blocks,
        &tables,
        &seed,
        &ivfc_master,
        &fat_master,
        commit_id,
        &mac_key,
    );
    image[..HEADER_SIZE].copy_from_slice(&header[..]);
    image[HEADER_SIZE..2 * HEADER_SIZE].copy_from_slice(&header[..]);
    Ok(image)
}

fn write_map_entry(
    entry: &mut [u8],
    virtual_offset: u64,
    physical: u64,
    size: u64,
    alignment: u32,
) {
    put_u64(entry, 0, virtual_offset);
    put_u64(entry, 8, physical);
    put_u64(entry, 0x10, size);
    put_u32(entry, 0x18, alignment);
}

/// Fills the hash levels `levels` of the meta area (level 1 first) over
/// `data` and returns the master hash. Like FS, which hashes what it wrote
/// even where that is zeros (the indexer's `lastPublishedId` block), only
/// the blocks of `data` that were never written, from block `written` on,
/// get the zero hash.
fn hash_levels(
    meta: &mut [u8],
    levels: &[usize],
    salts: &[[u8; 32]; 4],
    data: &[u8],
    mut written: usize,
) -> [u8; HASH_SIZE] {
    let mut below = data.to_vec();
    for (depth, &level) in levels.iter().enumerate().rev() {
        let (offset, size) = META_REGIONS[level];
        let salt = &salts[depth + 1];
        let hashes = &mut meta[offset..offset + size];
        hashes.fill(0);
        for (index, block) in below.chunks(BLOCK_SIZE).take(written).enumerate() {
            hashes[index * HASH_SIZE..(index + 1) * HASH_SIZE]
                .copy_from_slice(&salted_hash(salt, block, BLOCK_SIZE));
        }
        below = hashes.to_vec();
        written = usize::MAX;
    }
    salted_hash(&salts[0], &below, BLOCK_SIZE)
}

/// Where the file system's tables went: first blocks of the directory and
/// the file table.
struct TableBlocks {
    directories: u32,
    files: u32,
}

#[allow(clippy::too_many_arguments)]
fn header(
    extra: &ExtraData,
    blocks: usize,
    journal_blocks: usize,
    tables: &TableBlocks,
    seed: &[u8; 32],
    ivfc_master: &[u8; HASH_SIZE],
    fat_master: &[u8; HASH_SIZE],
    commit_id: u64,
    mac_key: &[u8; 16],
) -> Box<[u8; HEADER_SIZE]> {
    let mut header = Box::new([0; HEADER_SIZE]);
    let data_size = (blocks * BLOCK_SIZE) as u64;
    let journal_size = (journal_blocks * BLOCK_SIZE) as u64;
    let meta = |index: usize| (index as u64) << (64 - META_SEGMENT_BITS);
    let main = |index: usize| (index as u64) << (64 - MAIN_SEGMENT_BITS);

    header[0x100..0x104].copy_from_slice(DISF_MAGIC);
    put_u32(&mut header[..], 0x104, DISF_VERSION);
    let mut layout = [0u64; LAYOUT_WORDS];
    layout[L_MAIN_MAP_ENTRIES] = MAIN_MAP_ENTRIES as u64;
    layout[L_MAIN_MAP_ENTRIES + 1] = MAP_ENTRIES_SIZE as u64;
    layout[L_META_MAP_ENTRIES] = META_MAP_ENTRIES as u64;
    layout[L_META_MAP_ENTRIES + 1] = MAP_ENTRIES_SIZE as u64;
    layout[L_MAIN_DATA] = MAIN_DATA as u64;
    layout[L_MAIN_DATA + 1] =
        (2 * DUPLEX_L1_SIZE + 2 * DUPLEX_DATA_SIZE) as u64 + data_size + journal_size;
    layout[L_DUPLEX_L1_A] = main(0);
    layout[L_DUPLEX_L1_B] = main(1);
    layout[L_DUPLEX_L1_SIZE] = DUPLEX_L1_SIZE as u64;
    layout[L_DUPLEX_DATA_A] = main(2);
    layout[L_DUPLEX_DATA_B] = main(3);
    layout[L_DUPLEX_DATA_SIZE] = DUPLEX_DATA_SIZE as u64;
    layout[L_JOURNAL_DATA] = main(4);
    layout[L_JOURNAL_DATA_SIZE_A] = data_size;
    layout[L_JOURNAL_DATA_SIZE_B] = data_size;
    layout[L_JOURNAL_SIZE] = journal_size;
    layout[L_DUPLEX_MASTER_A] = DUPLEX_MASTER[0] as u64;
    layout[L_DUPLEX_MASTER_B] = DUPLEX_MASTER[1] as u64;
    layout[L_DUPLEX_MASTER_SIZE] = DUPLEX_MASTER_SIZE as u64;
    layout[L_IVFC_MASTER_A] = IVFC_MASTER[0] as u64;
    layout[L_IVFC_MASTER_B] = IVFC_MASTER[1] as u64;
    layout[L_IVFC_MASTER_SIZE] = HASH_SIZE as u64;
    for (word, region) in [
        (L_JOURNAL_TABLE, 0),
        (L_JOURNAL_PHYSICAL_BITMAP, 1),
        (L_JOURNAL_VIRTUAL_BITMAP, 2),
        (L_JOURNAL_FREE_BITMAP, 3),
        (L_IVFC_L1, 5),
        (L_IVFC_L2, 6),
        (L_IVFC_L3, 7),
        (L_FAT, 4),
        (L_FAT_IVFC_L1, 8),
        (L_FAT_IVFC_L2, 9),
    ] {
        layout[word] = meta(region);
        layout[word + 1] = META_REGIONS[region].1 as u64;
    }
    layout[L_DUPLEX_INDEX] = 0;
    layout[L_FAT_IVFC_MASTER_A] = FAT_IVFC_MASTER[0] as u64;
    layout[L_FAT_IVFC_MASTER_B] = FAT_IVFC_MASTER[1] as u64;
    for (index, word) in layout.iter().enumerate() {
        put_u64(&mut header[..], LAYOUT + index * 8, *word);
    }

    // Duplex: master, level 1 and data; the offsets are the samples'.
    header[DUPLEX_HEADER..DUPLEX_HEADER + 4].copy_from_slice(b"DPFS");
    put_u32(&mut header[..], DUPLEX_HEADER + 4, 0x1_0000);
    for (index, (offset, size, order)) in [
        (0, DUPLEX_MASTER_SIZE as u64, 0),
        (0x80, DUPLEX_L1_SIZE as u64, 14),
        (0xC000, DUPLEX_DATA_SIZE as u64, 14),
    ]
    .into_iter()
    .enumerate()
    {
        let at = DUPLEX_HEADER + 8 + index * 0x14;
        put_u64(&mut header[..], at, offset);
        put_u64(&mut header[..], at + 8, size);
        put_u32(&mut header[..], at + 0x10, order);
    }

    let ivfc_levels = [
        (meta(5), META_REGIONS[5].1 as u64),
        (meta(6), META_REGIONS[6].1 as u64),
        (meta(7), META_REGIONS[7].1 as u64),
        (0, data_size),
    ];
    ivfc_header(&mut header[IVFC_HEADER..], &ivfc_levels, seed);
    let fat_ivfc_levels = [
        (meta(8), META_REGIONS[8].1 as u64),
        (meta(9), META_REGIONS[9].1 as u64),
        (meta(4), META_REGIONS[4].1 as u64),
    ];
    ivfc_header(&mut header[FAT_IVFC_HEADER..], &fat_ivfc_levels, &[0; 32]);

    header[JOURNAL_HEADER..JOURNAL_HEADER + 4].copy_from_slice(b"JNGL");
    put_u32(&mut header[..], JOURNAL_HEADER + 4, 0x1_0000);
    put_u64(
        &mut header[..],
        JOURNAL_HEADER + 8,
        data_size + journal_size,
    );
    put_u64(&mut header[..], JOURNAL_HEADER + 0x10, journal_size);
    put_u64(&mut header[..], JOURNAL_HEADER + 0x18, BLOCK_SIZE as u64);
    put_u32(&mut header[..], JOURNAL_HEADER + 0x20, 1);
    put_u32(&mut header[..], JOURNAL_HEADER + 0x24, blocks as u32);
    put_u32(
        &mut header[..],
        JOURNAL_HEADER + 0x28,
        journal_blocks as u32,
    );

    header[SAVE_HEADER..SAVE_HEADER + 4].copy_from_slice(b"SAVE");
    put_u32(&mut header[..], SAVE_HEADER + 4, 0x6_0000);
    put_u64(&mut header[..], SAVE_HEADER + 8, blocks as u64);
    put_u64(&mut header[..], SAVE_HEADER + 0x10, BLOCK_SIZE as u64);
    put_u64(&mut header[..], FAT_HEADER, BLOCK_SIZE as u64);
    put_u32(&mut header[..], FAT_HEADER + 0x10, blocks as u32);
    put_u32(&mut header[..], FAT_HEADER + 0x20, blocks as u32);
    put_u32(&mut header[..], FAT_HEADER + 0x28, tables.directories);
    put_u32(&mut header[..], FAT_HEADER + 0x2C, tables.files);

    for (at, entries, bits) in [
        (MAIN_REMAP_HEADER, 5, MAIN_SEGMENT_BITS),
        (
            META_REMAP_HEADER,
            META_REGIONS.len() as u32,
            META_SEGMENT_BITS,
        ),
    ] {
        header[at..at + 4].copy_from_slice(b"RMAP");
        put_u32(&mut header[..], at + 4, 0x1_0000);
        put_u32(&mut header[..], at + 8, entries);
        put_u32(&mut header[..], at + 0xC, entries);
        put_u32(&mut header[..], at + 0x10, bits);
    }

    for at in EXTRA_DATA {
        put_u64(&mut header[..], at + 0x18, extra.save_data_id);
        put_u64(&mut header[..], at + 0x40, extra.owner_id);
        put_u32(&mut header[..], at + 0x50, extra.flags);
        put_u64(&mut header[..], at + 0x58, extra.size);
        put_u64(&mut header[..], at + 0x60, extra.journal_size);
        put_u64(&mut header[..], at + 0x68, commit_id);
    }

    for at in IVFC_MASTER {
        header[at..at + HASH_SIZE].copy_from_slice(ivfc_master);
    }
    for at in FAT_IVFC_MASTER {
        header[at..at + HASH_SIZE].copy_from_slice(fat_master);
    }

    let hash = crypto::sha256(&header[HASHED_RANGE]);
    header[HEADER_HASH..HEADER_HASH + 32].copy_from_slice(&hash);
    let mac = crypto::cmac(mac_key, &header[CMAC_RANGE]);
    header[..16].copy_from_slice(&mac);
    header
}

/// An `IVFC` header: magic, version, master hash size, a zero word, six
/// (offset, size, block order, reserved) levels and the salt seed.
fn ivfc_header(out: &mut [u8], levels: &[(u64, u64)], seed: &[u8; 32]) {
    out[..4].copy_from_slice(b"IVFC");
    put_u32(out, 4, 0x2_0000);
    put_u32(out, 8, HASH_SIZE as u32);
    for (index, &(offset, size)) in levels.iter().enumerate() {
        let at = 0x10 + index * 0x18;
        put_u64(out, at, offset);
        put_u64(out, at + 8, size);
        put_u32(out, at + 0x10, BLOCK_SIZE.trailing_zeros());
    }
    out[0xA0..0xC0].copy_from_slice(seed);
}

/// A table of the file system being built.
struct Table {
    entries: Vec<[u8; TABLE_ENTRY_SIZE]>,
}

impl Table {
    /// The two list heads, and for directories the root.
    fn new(with_root: bool) -> Self {
        let mut table = Self {
            entries: vec![[0; TABLE_ENTRY_SIZE]; 2],
        };
        if with_root {
            table.push(0, b"");
        }
        table
    }

    /// Appends an entry and puts it first in the list of entries in use.
    fn push(&mut self, parent: u32, name: &[u8]) -> u32 {
        let index = self.entries.len() as u32;
        let mut entry = [0; TABLE_ENTRY_SIZE];
        put_u32(&mut entry, 0, parent);
        entry[4..4 + name.len()].copy_from_slice(name);
        put_u32(&mut entry, 0x5C, read_u32(&self.entries[1], 0x5C));
        put_u32(&mut self.entries[1], 0x5C, index);
        self.entries.push(entry);
        index
    }

    fn blocks(&self) -> usize {
        (self.entries.len() * TABLE_ENTRY_SIZE).div_ceil(BLOCK_SIZE)
    }

    fn bytes(mut self) -> Vec<u8> {
        let capacity = self.blocks() * BLOCK_SIZE / TABLE_ENTRY_SIZE;
        let length = self.entries.len() as u32;
        put_u32(&mut self.entries[0], 0, length);
        put_u32(&mut self.entries[0], 4, capacity as u32);
        let mut bytes = self.entries.concat();
        bytes.resize(self.blocks() * BLOCK_SIZE, 0);
        bytes
    }
}

fn components(path: &str) -> Result<Vec<&[u8]>, SaveError> {
    let parts: Vec<&[u8]> = path
        .trim_start_matches('/')
        .split('/')
        .map(str::as_bytes)
        .collect();
    if parts
        .iter()
        .any(|part| part.is_empty() || *part == b"." || *part == b".." || part.len() > NAME_SIZE)
    {
        return Err(SaveError::BadPath(path.into()));
    }
    Ok(parts)
}

/// What a path names while the tree is built.
#[derive(Clone, Copy)]
enum Node {
    Directory(u32),
    File,
}

/// Puts `child` first among the children `parent` lists in its value word
/// at `field` (0x48 for directories, 0x4C for files).
fn link(
    directories: &mut Table,
    parent: u32,
    field: usize,
    child: &mut [u8; TABLE_ENTRY_SIZE],
    index: u32,
) {
    let parent = &mut directories.entries[parent as usize];
    put_u32(child, 0x44, read_u32(parent, field));
    put_u32(parent, field, index);
}

/// Builds the file system: the data blocks (what the journal presents),
/// the allocation table, where the tables start and how many blocks, from
/// the first, are allocated.
fn file_system(
    blocks: usize,
    entries: &[Entry],
) -> Result<(Vec<u8>, Vec<u8>, TableBlocks, usize), SaveError> {
    let mut directories = Table::new(true);
    let mut files = Table::new(false);
    let mut paths = std::collections::BTreeMap::<Vec<&[u8]>, Node>::new();
    paths.insert(Vec::new(), Node::Directory(ROOT_INDEX));
    let mut file_data: Vec<(u32, &[u8])> = Vec::new();

    for entry in entries {
        let bad_path = || SaveError::BadPath(entry.path().into());
        let parts = components(entry.path())?;
        let directory_count = match entry {
            Entry::Directory(_) => parts.len(),
            Entry::File(..) => parts.len() - 1,
        };
        let mut parent = ROOT_INDEX;
        for depth in 1..=directory_count {
            parent = match paths.get(&parts[..depth]) {
                Some(Node::Directory(index)) => *index,
                Some(Node::File) => return Err(bad_path()),
                None => {
                    let index = directories.push(parent, parts[depth - 1]);
                    let mut child = directories.entries[index as usize];
                    link(&mut directories, parent, 0x48, &mut child, index);
                    directories.entries[index as usize] = child;
                    paths.insert(parts[..depth].to_vec(), Node::Directory(index));
                    index
                }
            };
        }
        if let Entry::File(_, data) = entry {
            if paths.insert(parts.clone(), Node::File).is_some() {
                return Err(bad_path());
            }
            let index = files.push(parent, parts[parts.len() - 1]);
            link(
                &mut directories,
                parent,
                0x4C,
                &mut files.entries[index as usize],
                index,
            );
            file_data.push((index, data));
        }
    }

    // Blocks: the directory table, the file table, the files in order,
    // then one free run.
    let mut fat = vec![0; META_REGIONS[META_FAT].1];
    let mut core = vec![0; blocks * BLOCK_SIZE];
    let mut next = 0;
    let mut allocate = |count: usize, fat: &mut [u8]| -> Result<usize, SaveError> {
        let start = next;
        next += count;
        if next > blocks {
            return Err(SaveError::Full {
                needed: next,
                blocks,
            });
        }
        write_run(fat, start, count);
        Ok(start)
    };
    let tables = TableBlocks {
        directories: allocate(directories.blocks(), &mut fat)? as u32,
        files: allocate(files.blocks(), &mut fat)? as u32,
    };
    for (index, data) in file_data {
        let start = if data.is_empty() {
            EMPTY_FILE_START
        } else {
            let start = allocate(data.len().div_ceil(BLOCK_SIZE), &mut fat)?;
            core[start * BLOCK_SIZE..start * BLOCK_SIZE + data.len()].copy_from_slice(data);
            start as u32
        };
        let entry = &mut files.entries[index as usize];
        put_u32(entry, 0x48, start);
        put_u64(entry, 0x4C, data.len() as u64);
    }
    let used = next;
    if used < blocks {
        write_run(&mut fat, used, blocks - used);
        put_u32(&mut fat, 4, used as u32 + 1);
    }
    for (start, table) in [(tables.directories, directories), (tables.files, files)] {
        let bytes = table.bytes();
        let at = start as usize * BLOCK_SIZE;
        core[at..at + bytes.len()].copy_from_slice(&bytes);
    }
    Ok((core, fat, tables, used))
}

fn name_of(entry: &[u8]) -> &[u8] {
    let name = &entry[4..4 + NAME_SIZE];
    &name[..name.iter().position(|&byte| byte == 0).unwrap_or(NAME_SIZE)]
}

/// Makes blocks `start..start + count` one list in the allocation table.
/// Entry `n + 1` describes block `n`: the previous segment (bit 31 alone
/// at a list's start) and the next (0 at its end, bit 31 when the segment
/// has more than one block); a longer segment repeats its first and last
/// entry, bit 31 set in the first word, in its second and last entry.
fn write_run(fat: &mut [u8], start: usize, count: usize) {
    let first = start + 1;
    let last = first + count - 1;
    put_u32(fat, first * 8, CHAIN_FLAG);
    put_u32(fat, first * 8 + 4, if count > 1 { CHAIN_FLAG } else { 0 });
    if count > 1 {
        for entry in [first + 1, last] {
            put_u32(fat, entry * 8, CHAIN_FLAG | first as u32);
            put_u32(fat, entry * 8 + 4, last as u32);
        }
    }
}

// Reading and editing.

/// What a block never written reads as.
static ZERO_BLOCK: [u8; BLOCK_SIZE] = [0; BLOCK_SIZE];

/// What [`open`] found in a save after checking all of it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SaveData {
    pub extra: ExtraData,
    /// Directories and files, parents before children, siblings in the
    /// order of the save's lists.
    pub entries: Vec<Entry>,
}

impl SaveData {
    pub fn file(&self, path: &str) -> Option<&[u8]> {
        self.entries.iter().find_map(|entry| match entry {
            Entry::File(name, data) if name == path => Some(&data[..]),
            _ => None,
        })
    }
}

/// An entry of a save as [`SaveEditor::list`] names it, in the order of
/// [`SaveData::entries`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Listing {
    Directory(String),
    File { path: String, size: u64 },
}

/// Checks a save image completely — header hash and CMAC, every hash of
/// both integrity layers, the journal map, the allocation chains and the
/// tables — and returns what it holds, as FS reads it: blocks never
/// written read as zeros.
pub fn open(image: &[u8], keys: &KeySet) -> Result<SaveData, SaveError> {
    let layers = Layers::read(image, keys)?;
    let changed = BTreeMap::new();
    let fs = FileSystem {
        layers: &layers,
        image,
        changed: &changed,
    };
    let entries = fs
        .nodes()?
        .into_iter()
        .map(|node| match node.file {
            None => Ok(Entry::Directory(node.path)),
            Some(file) => Ok(Entry::File(node.path, fs.contents(&file)?)),
        })
        .collect::<Result<_, SaveError>>()?;
    Ok(SaveData {
        extra: layers.extra(),
        entries,
    })
}

/// A save image being edited: opened with every check [`open`] makes,
/// changed through the file system as FS changes it, and committed by
/// [`finish`](Self::finish) into an image of the same length.
pub struct SaveEditor {
    image: Vec<u8>,
    layers: Layers,
    /// Data blocks changed, whole, by index.
    changed: BTreeMap<usize, Box<[u8]>>,
}

impl SaveEditor {
    pub fn open(image: Vec<u8>, keys: &KeySet) -> Result<Self, SaveError> {
        let layers = Layers::read(&image, keys)?;
        let editor = Self {
            image,
            layers,
            changed: BTreeMap::new(),
        };
        editor.fs().nodes()?;
        Ok(editor)
    }

    pub fn extra(&self) -> ExtraData {
        self.layers.extra()
    }

    pub fn list(&self) -> Result<Vec<Listing>, SaveError> {
        Ok(self
            .fs()
            .nodes()?
            .into_iter()
            .map(|node| match node.file {
                None => Listing::Directory(node.path),
                Some(file) => Listing::File {
                    path: node.path,
                    size: file.size,
                },
            })
            .collect())
    }

    /// What the file `path` holds.
    pub fn read(&self, path: &str) -> Result<Vec<u8>, SaveError> {
        let fs = self.fs();
        match fs.locate(&fs.tables()?, path)?.1 {
            Some(Found::File(file)) => fs.contents(&file),
            _ => Err(SaveError::NotFound(path.into())),
        }
    }

    /// Stores `bytes` at `offset` of the file `path`, which keeps its size:
    /// they have to lie within it.
    pub fn write(&mut self, path: &str, offset: u64, bytes: &[u8]) -> Result<(), SaveError> {
        let fs = self.fs();
        let Some(Found::File(file)) = fs.locate(&fs.tables()?, path)?.1 else {
            return Err(SaveError::NotFound(path.into()));
        };
        if offset
            .checked_add(bytes.len() as u64)
            .is_none_or(|end| end > file.size)
        {
            return Err(SaveError::PastEnd(path.into()));
        }
        if bytes.is_empty() {
            return Ok(());
        }
        let blocks = fs.blocks(file.start)?;
        self.write_blocks(&blocks, offset as usize, bytes)
    }

    /// Creates the directory `path`, whose parent has to exist, as FS's
    /// `CreateDirectory` does: a new entry first among its parent's
    /// directories.
    pub fn create_directory(&mut self, path: &str) -> Result<(), SaveError> {
        let (parent, name) = self.vacancy(path, DIRECTORIES)?;
        self.add_entry(DIRECTORIES, parent, name, [0; 0x14])
    }

    /// Creates the file `path` holding `contents`, whose parent directory
    /// has to exist, as FS's `CreateFile` and a write of `contents` do:
    /// blocks taken off the front of the allocation table's free list and
    /// a new entry first among its parent's files.
    pub fn create_file(&mut self, path: &str, contents: &[u8]) -> Result<(), SaveError> {
        let (parent, name) = self.vacancy(path, FILES)?;
        let start = if contents.is_empty() {
            EMPTY_FILE_START
        } else {
            let start = self.allocate(contents.len().div_ceil(BLOCK_SIZE))?;
            let blocks = self.fs().blocks(start)?;
            self.write_blocks(&blocks, 0, contents)?;
            start
        };
        let mut value = [0; 0x14];
        put_u32(&mut value, 0, start);
        put_u64(&mut value, 4, contents.len() as u64);
        self.add_entry(FILES, parent, name, value)
    }

    /// Deletes the file `path` as FS's `DeleteFile` does: its blocks go
    /// back to the front of the allocation table's free list (they keep
    /// what they hold), its entry leaves its parent's files and the list
    /// of entries in use for the front of the table's free list.
    pub fn delete_file(&mut self, path: &str) -> Result<(), SaveError> {
        let fs = self.fs();
        let tables = fs.tables()?;
        let (parent, Some(Found::File(file))) = fs.locate(&tables, path)? else {
            return Err(SaveError::NotFound(path.into()));
        };
        let mut free = Vec::new();
        if file.start != EMPTY_FILE_START {
            free = fs.segments(file.start)?;
            let head = read_u32(fs.fat(), 4);
            if head != 0 {
                free.extend(fs.segments(head - 1)?);
            }
        }
        let entries = &tables[FILES].bytes;
        let removed = table_entry(entries, file.index)?;
        let (sibling, next) = (read_u32(removed, 0x44), read_u32(removed, 0x5C));
        // Where it is linked from: its parent or its previous sibling, then
        // the entry before it in the list in use.
        let sibling_link = if read_u32(table_entry(&tables[DIRECTORIES].bytes, parent)?, 0x4C)
            == file.index
        {
            (DIRECTORIES, parent as usize * TABLE_ENTRY_SIZE + 0x4C)
        } else {
            let (previous, _) = fs
                .children(&tables, parent, FILES)?
                .into_iter()
                .find(|(_, entry)| read_u32(entry, 0x44) == file.index)
                .ok_or(SaveError::Corrupt("file missing from its parent's list"))?;
            (FILES, previous as usize * TABLE_ENTRY_SIZE + 0x44)
        };
        let mut in_use = 1;
        for _ in 0..=entries.len() / TABLE_ENTRY_SIZE {
            match read_u32(table_entry(entries, in_use)?, 0x5C) {
                following if following == file.index => break,
                0 => return Err(SaveError::Corrupt("entry missing from the list in use")),
                following => in_use = following,
            }
        }
        if read_u32(table_entry(entries, in_use)?, 0x5C) != file.index {
            return Err(SaveError::Corrupt("cycle in the list in use"));
        }
        let free_entries = read_u32(table_entry(entries, 0)?, 0x5C);

        if let Some(&(first, _)) = free.first() {
            self.write_list(&free);
            self.layers.fat.put(4, &(first as u32 + 1).to_le_bytes());
        }
        let (table, at) = sibling_link;
        self.write_blocks(&tables[table].blocks, at, &sibling.to_le_bytes())?;
        let blocks = &tables[FILES].blocks;
        let link = |index: u32| index as usize * TABLE_ENTRY_SIZE + 0x5C;
        self.write_blocks(blocks, link(in_use), &next.to_le_bytes())?;
        self.write_blocks(blocks, link(file.index), &free_entries.to_le_bytes())?;
        self.write_blocks(blocks, link(0), &file.index.to_le_bytes())
    }

    /// The image with every change committed: changed data blocks in the
    /// physical blocks the journal maps them to, the levels of both
    /// integrity layers re-hashed up to the master hashes, the header
    /// re-hashed and re-signed into both copies. Bytes nothing changed stay
    /// as they were; an editor that changed nothing gives back the image.
    pub fn finish(self) -> Result<Vec<u8>, SaveError> {
        let Self {
            mut image,
            mut layers,
            changed,
        } = self;
        if !changed.is_empty() || !layers.fat.dirty.is_empty() {
            layers.commit(&mut image, &changed)?;
        }
        Ok(image)
    }

    fn fs(&self) -> FileSystem<'_> {
        FileSystem {
            layers: &self.layers,
            image: &self.image,
            changed: &self.changed,
        }
    }

    /// The directory to create `path` in and the name to give it there,
    /// once the parent exists, the name is free and table `table` has room
    /// for an entry.
    fn vacancy<'p>(&self, path: &'p str, table: usize) -> Result<(u32, &'p [u8]), SaveError> {
        let parts = components(path)?;
        let fs = self.fs();
        let tables = fs.tables()?;
        let (parent, found) = fs.locate(&tables, path)?;
        if found.is_some() {
            return Err(SaveError::AlreadyExists(path.into()));
        }
        let table = &tables[table];
        let head = table_entry(&table.bytes, 0)?;
        let length = read_u32(head, 0) as usize;
        let room = read_u32(head, 0x5C) != 0
            || (length < read_u32(head, 4) as usize
                && (length + 1) * TABLE_ENTRY_SIZE <= table.bytes.len());
        if !room {
            return Err(SaveError::TableFull(path.into()));
        }
        Ok((parent, parts[parts.len() - 1]))
    }

    /// Adds an entry named `name` under directory `parent` to table `table`
    /// as FS's lists do (an entry off the table's free list if it has one,
    /// else the one after the last, put first in the list of entries in
    /// use) and puts it first among its parent's children of its kind.
    /// `value` is what the entry records after the next sibling.
    fn add_entry(
        &mut self,
        table: usize,
        parent: u32,
        name: &[u8],
        value: [u8; 0x14],
    ) -> Result<(), SaveError> {
        let tables = self.fs().tables()?;
        let entries = &tables[table];
        let mut head: [u8; TABLE_ENTRY_SIZE] = table_entry(&entries.bytes, 0)?
            .try_into()
            .expect("table entries are TABLE_ENTRY_SIZE long");
        let mut used: [u8; TABLE_ENTRY_SIZE] = table_entry(&entries.bytes, 1)?
            .try_into()
            .expect("table entries are TABLE_ENTRY_SIZE long");
        let free = read_u32(&head, 0x5C);
        let index = if free != 0 {
            put_u32(
                &mut head,
                0x5C,
                read_u32(table_entry(&entries.bytes, free)?, 0x5C),
            );
            free
        } else {
            let length = read_u32(&head, 0);
            put_u32(&mut head, 0, length + 1);
            length
        };
        if index <= 1 || (table == DIRECTORIES && index == ROOT_INDEX) {
            return Err(SaveError::Corrupt("table entry in use handed out"));
        }
        // The parent lists its directories at 0x48 and its files at 0x4C.
        let field = 0x48 + 4 * table;
        let mut parent_entry: [u8; TABLE_ENTRY_SIZE] =
            table_entry(&tables[DIRECTORIES].bytes, parent)?
                .try_into()
                .expect("table entries are TABLE_ENTRY_SIZE long");
        let mut entry = [0; TABLE_ENTRY_SIZE];
        put_u32(&mut entry, 0, parent);
        entry[4..4 + name.len()].copy_from_slice(name);
        put_u32(&mut entry, 0x44, read_u32(&parent_entry, field));
        entry[0x48..0x5C].copy_from_slice(&value);
        put_u32(&mut entry, 0x5C, read_u32(&used, 0x5C));
        put_u32(&mut used, 0x5C, index);
        put_u32(&mut parent_entry, field, index);

        let blocks = &entries.blocks;
        self.write_blocks(blocks, 0, &head)?;
        self.write_blocks(blocks, TABLE_ENTRY_SIZE, &used)?;
        self.write_blocks(blocks, index as usize * TABLE_ENTRY_SIZE, &entry)?;
        self.write_blocks(
            &tables[DIRECTORIES].blocks,
            parent as usize * TABLE_ENTRY_SIZE,
            &parent_entry,
        )
    }

    /// Takes `count` blocks off the front of the allocation table's free
    /// list, as FS's allocation table does, and returns the first: they
    /// end up a list of their own, the rest of the free list its new head.
    fn allocate(&mut self, count: usize) -> Result<u32, SaveError> {
        let fs = self.fs();
        let head = read_u32(fs.fat(), 4);
        let free = if head == 0 {
            Vec::new()
        } else {
            fs.segments(head - 1)?
        };
        let available: usize = free.iter().map(|&(_, length)| length).sum();
        if available < count {
            return Err(SaveError::NoSpace {
                needed: count,
                free: available,
            });
        }
        let mut taken = Vec::new();
        let mut rest = Vec::new();
        let mut needed = count;
        for (first, length) in free {
            if needed == 0 {
                rest.push((first, length));
            } else if length <= needed {
                taken.push((first, length));
                needed -= length;
            } else {
                taken.push((first, needed));
                rest.push((first + needed, length - needed));
                needed = 0;
            }
        }
        self.write_list(&taken);
        self.write_list(&rest);
        let rest_head = rest.first().map_or(0, |&(first, _)| first as u32 + 1);
        self.layers.fat.put(4, &rest_head.to_le_bytes());
        Ok(taken[0].0 as u32)
    }

    /// Makes `segments` (first block, length) one list in the allocation
    /// table, in their order (see [`write_run`] for the entries).
    fn write_list(&mut self, segments: &[(usize, usize)]) {
        let mut put = |index: usize, previous: u32, next: u32| {
            let mut entry = [0; 8];
            put_u32(&mut entry, 0, previous);
            put_u32(&mut entry, 4, next);
            self.layers.fat.put(index * 8, &entry);
        };
        for (position, &(first, length)) in segments.iter().enumerate() {
            let head = first + 1;
            let previous = match position {
                0 => CHAIN_FLAG,
                _ => segments[position - 1].0 as u32 + 1,
            };
            let next = segments
                .get(position + 1)
                .map_or(0, |&(next, _)| next as u32 + 1);
            let multiple = if length > 1 { CHAIN_FLAG } else { 0 };
            put(head, previous, next | multiple);
            if length > 1 {
                let last = head + length - 1;
                for index in [head + 1, last] {
                    put(index, CHAIN_FLAG | head as u32, last as u32);
                }
            }
        }
    }

    /// Stores `bytes` at `offset` of the data that the blocks `chain` hold
    /// in their order.
    fn write_blocks(
        &mut self,
        chain: &[usize],
        offset: usize,
        bytes: &[u8],
    ) -> Result<(), SaveError> {
        let mut done = 0;
        while done < bytes.len() {
            let at = offset + done;
            let index = *chain
                .get(at / BLOCK_SIZE)
                .ok_or(SaveError::Corrupt("write past the blocks of a chain"))?;
            let into = at % BLOCK_SIZE;
            let length = (BLOCK_SIZE - into).min(bytes.len() - done);
            let block = match self.changed.entry(index) {
                btree_map::Entry::Occupied(block) => block.into_mut(),
                btree_map::Entry::Vacant(vacant) => {
                    let current = self.layers.data_block(&self.image, index)?;
                    vacant.insert(current.into_owned().into_boxed_slice())
                }
            };
            block[into..into + length].copy_from_slice(&bytes[done..done + length]);
            done += length;
        }
        Ok(())
    }
}

struct MapEntry {
    virtual_offset: u64,
    physical: u64,
    size: u64,
}

/// The entries of the remap storage whose header is at `at`.
fn map_entries(header: &[u8], at: usize, entries: &[u8]) -> Result<Vec<MapEntry>, SaveError> {
    if &header[at..at + 4] != b"RMAP" {
        return Err(SaveError::BadMagic("remap storage"));
    }
    let count = read_u32(header, at + 8) as usize;
    (0..count)
        .map(|index| {
            let entry = entries
                .get(index * MAP_ENTRY_SIZE..(index + 1) * MAP_ENTRY_SIZE)
                .ok_or(SaveError::Truncated("remap entries"))?;
            Ok(MapEntry {
                virtual_offset: read_u64(entry, 0),
                physical: read_u64(entry, 8),
                size: read_u64(entry, 0x10),
            })
        })
        .collect()
}

/// Cuts `size` bytes from `offset` of a remap storage at the ends of its
/// `entries` and hands `piece` each part: its offset in the storage below,
/// its offset in the range and its length.
fn remap_pieces(
    entries: &[MapEntry],
    offset: u64,
    size: usize,
    what: &'static str,
    mut piece: impl FnMut(u64, usize, usize) -> Result<(), SaveError>,
) -> Result<(), SaveError> {
    let mut done = 0;
    while done < size {
        let at = offset
            .checked_add(done as u64)
            .ok_or(SaveError::Corrupt(what))?;
        let (entry, into) = entries
            .iter()
            .find_map(|entry| {
                let into = at.checked_sub(entry.virtual_offset)?;
                (into < entry.size).then_some((entry, into))
            })
            .ok_or(SaveError::Corrupt(what))?;
        let length = usize::try_from(entry.size - into)
            .unwrap_or(usize::MAX)
            .min(size - done);
        let physical = entry
            .physical
            .checked_add(into)
            .ok_or(SaveError::Corrupt(what))?;
        piece(physical, done, length)?;
        done += length;
    }
    Ok(())
}

/// The range of `size` bytes at `offset` of something `length` long.
fn span(
    length: usize,
    offset: u64,
    size: usize,
    what: &'static str,
) -> Result<std::ops::Range<usize>, SaveError> {
    usize::try_from(offset)
        .ok()
        .and_then(|start| Some(start..start.checked_add(size)?))
        .filter(|range| range.end <= length)
        .ok_or(SaveError::Truncated(what))
}

fn slice<'a>(
    image: &'a [u8],
    offset: u64,
    size: u64,
    what: &'static str,
) -> Result<&'a [u8], SaveError> {
    let size = usize::try_from(size).map_err(|_| SaveError::Truncated(what))?;
    Ok(&image[span(image.len(), offset, size, what)?])
}

fn block_size(order: u32) -> Result<usize, SaveError> {
    1usize
        .checked_shl(order)
        .filter(|&size| size >= HASH_SIZE)
        .ok_or(SaveError::Corrupt("block order"))
}

/// The main remap storage: its entries over a range of the image.
struct MainStorage {
    entries: Vec<MapEntry>,
    area: std::ops::Range<usize>,
}

impl MainStorage {
    /// `size` bytes from `offset`, borrowed when the image holds them in
    /// one piece.
    fn read<'a>(
        &self,
        image: &'a [u8],
        offset: u64,
        size: usize,
        what: &'static str,
    ) -> Result<Cow<'a, [u8]>, SaveError> {
        let area = &image[self.area.clone()];
        let whole = self.entries.iter().find_map(|entry| {
            let into = offset.checked_sub(entry.virtual_offset)?;
            (into.checked_add(size as u64)? <= entry.size)
                .then(|| entry.physical.checked_add(into))
                .flatten()
        });
        if let Some(physical) = whole {
            return Ok(Cow::Borrowed(&area[span(area.len(), physical, size, what)?]));
        }
        let mut out = vec![0; size];
        remap_pieces(&self.entries, offset, size, what, |physical, at, length| {
            out[at..at + length].copy_from_slice(&area[span(area.len(), physical, length, what)?]);
            Ok(())
        })?;
        Ok(Cow::Owned(out))
    }

    fn write(
        &self,
        image: &mut [u8],
        offset: u64,
        bytes: &[u8],
        what: &'static str,
    ) -> Result<(), SaveError> {
        let area = &mut image[self.area.clone()];
        remap_pieces(&self.entries, offset, bytes.len(), what, |physical, at, length| {
            let range = span(area.len(), physical, length, what)?;
            area[range].copy_from_slice(&bytes[at..at + length]);
            Ok(())
        })
    }
}

/// A level of the duplex storage as FS reads it, each block from the
/// copy its bitmap selects.
struct Duplex {
    bytes: Vec<u8>,
    block: usize,
    /// Where in the main storage each block was read from.
    copies: Vec<u64>,
    /// Blocks changed since.
    dirty: BTreeSet<usize>,
}

impl Duplex {
    /// Reads `size` bytes, block `n` from copy `b` where bit `n` of
    /// `selector` is set and from copy `a` where it is clear.
    fn read(
        main: &MainStorage,
        image: &[u8],
        selector: &[u8],
        (a, b): (u64, u64),
        size: u64,
        order: u32,
    ) -> Result<Self, SaveError> {
        let block = block_size(order)?;
        let size = usize::try_from(size).map_err(|_| SaveError::Truncated("duplex storage"))?;
        if size.div_ceil(block) > selector.len() / 4 * 32 {
            return Err(SaveError::Corrupt("duplex bitmap smaller than its level"));
        }
        let mut bytes = vec![0; size];
        let mut copies = Vec::with_capacity(size.div_ceil(block));
        for (index, chunk) in bytes.chunks_mut(block).enumerate() {
            let copy = if bit(selector, index) { b } else { a };
            let offset = copy
                .checked_add((index * block) as u64)
                .ok_or(SaveError::Corrupt("duplex storage"))?;
            chunk.copy_from_slice(&main.read(image, offset, chunk.len(), "duplex storage")?);
            copies.push(offset);
        }
        Ok(Self {
            bytes,
            block,
            copies,
            dirty: BTreeSet::new(),
        })
    }

    /// `size` bytes from `offset` of the meta remap storage with `entries`
    /// over this level.
    fn meta_read(&self, entries: &[MapEntry], offset: u64, size: u64) -> Result<Vec<u8>, SaveError> {
        let what = "meta remap storage";
        let size = usize::try_from(size).map_err(|_| SaveError::Truncated(what))?;
        let mut out = vec![0; size];
        remap_pieces(entries, offset, size, what, |physical, at, length| {
            let range = span(self.bytes.len(), physical, length, what)?;
            out[at..at + length].copy_from_slice(&self.bytes[range]);
            Ok(())
        })?;
        Ok(out)
    }

    fn meta_write(&mut self, entries: &[MapEntry], offset: u64, bytes: &[u8]) -> Result<(), SaveError> {
        let what = "meta remap storage";
        remap_pieces(entries, offset, bytes.len(), what, |physical, at, length| {
            let range = span(self.bytes.len(), physical, length, what)?;
            self.dirty
                .extend(range.start / self.block..range.end.div_ceil(self.block));
            self.bytes[range].copy_from_slice(&bytes[at..at + length]);
            Ok(())
        })
    }
}

/// A level of an integrity layer, or the allocation table, in the meta
/// storage, as FS reads it.
struct Level {
    bytes: Vec<u8>,
    /// Where it lies in the meta storage.
    offset: u64,
    block: usize,
    /// Blocks changed since the image was read.
    dirty: BTreeSet<usize>,
}

impl Level {
    fn chunk(&self, index: usize) -> &[u8] {
        let start = index * self.block;
        &self.bytes[start..(start + self.block).min(self.bytes.len())]
    }

    /// Stores `bytes` at `at`, noting the blocks that change.
    fn put(&mut self, at: usize, bytes: &[u8]) {
        let range = at..at + bytes.len();
        if self.bytes[range.clone()] != *bytes {
            self.bytes[range.clone()].copy_from_slice(bytes);
            self.dirty
                .extend(range.start / self.block..range.end.div_ceil(self.block));
        }
    }

    /// Writes the changed blocks into the meta storage.
    fn store(&mut self, entries: &[MapEntry], duplex: &mut Duplex) -> Result<(), SaveError> {
        for index in std::mem::take(&mut self.dirty) {
            let offset = self.offset + (index * self.block) as u64;
            duplex.meta_write(entries, offset, self.chunk(index))?;
        }
        Ok(())
    }
}

/// Whether block `block` of level `level` of `layer`, `chunk`, was ever
/// written (its hash `expected` is not all zeros), checking it against
/// its hash when it was.
fn written(
    layer: &'static str,
    level: usize,
    block: usize,
    salt: &[u8; 32],
    expected: &[u8],
    chunk: &[u8],
    block_size: usize,
) -> Result<bool, SaveError> {
    if expected.iter().all(|&byte| byte == 0) {
        return Ok(false);
    }
    if salted_hash(salt, chunk, block_size) != expected {
        return Err(SaveError::Hash {
            layer,
            level,
            block,
        });
    }
    Ok(true)
}

/// Level `level` of `layer` as FS reads it from `bytes`, one hash of
/// `hashes` per block: blocks never written as zeros.
fn check_level(
    layer: &'static str,
    level: usize,
    mut bytes: Vec<u8>,
    hashes: &[u8],
    salt: &[u8; 32],
    block_size: usize,
) -> Result<Vec<u8>, SaveError> {
    for (block, chunk) in bytes.chunks_mut(block_size).enumerate() {
        let expected = hashes
            .get(block * HASH_SIZE..(block + 1) * HASH_SIZE)
            .ok_or(SaveError::Corrupt(
                "hash level smaller than the level below",
            ))?;
        if !written(layer, level, block, salt, expected, chunk, block_size)? {
            chunk.fill(0);
        }
    }
    Ok(bytes)
}

/// The hash levels of an integrity layer, first (the one the master hash
/// covers) first.
struct HashTree {
    /// Where the master hash is in the header.
    master: usize,
    salts: [[u8; 32]; 4],
    levels: Vec<Level>,
}

impl HashTree {
    /// Reads `levels` (offset, size and block order in the meta storage),
    /// each checked against the one above, the first against the master
    /// hash at `master` in `header`.
    fn read(
        layer: &'static str,
        header: &[u8],
        master: usize,
        seed: &[u8],
        levels: &[(u64, u64, u32)],
        entries: &[MapEntry],
        duplex: &Duplex,
    ) -> Result<Self, SaveError> {
        let salts = level_salts(seed);
        // The data below the last level needs a salt too.
        if levels.len() >= salts.len() {
            return Err(SaveError::Corrupt("too many hash levels"));
        }
        let master_hash = header
            .get(master..master + HASH_SIZE)
            .ok_or(SaveError::Truncated("master hash"))?;
        let mut read: Vec<Level> = Vec::with_capacity(levels.len());
        for (index, &(offset, size, order)) in levels.iter().enumerate() {
            let block = block_size(order)?;
            let hashes = read.last().map_or(master_hash, |level| &level.bytes[..]);
            let bytes = check_level(
                layer,
                index + 1,
                duplex.meta_read(entries, offset, size)?,
                hashes,
                &salts[index],
                block,
            )?;
            read.push(Level {
                bytes,
                offset,
                block,
                dirty: BTreeSet::new(),
            });
        }
        Ok(Self {
            master,
            salts,
            levels: read,
        })
    }

    /// The hashes of the data below the last level, and its salt.
    fn bottom(&self) -> (&[u8], &[u8; 32]) {
        let depth = self.levels.len();
        (&self.levels[depth - 1].bytes, &self.salts[depth])
    }

    /// Stores the hash of block `index` of the data below the last level.
    fn hash_below(&mut self, index: usize, block: &[u8], block_size: usize) {
        let depth = self.levels.len();
        let hash = salted_hash(&self.salts[depth], block, block_size);
        self.levels[depth - 1].put(index * HASH_SIZE, &hash);
    }

    /// Hashes the changed blocks of every level into the one above, the
    /// first into the master hash in `header`, and writes them into the
    /// meta storage.
    fn commit(
        &mut self,
        header: &mut [u8],
        entries: &[MapEntry],
        duplex: &mut Duplex,
    ) -> Result<(), SaveError> {
        for depth in (0..self.levels.len()).rev() {
            let (above, below) = self.levels.split_at_mut(depth);
            let level = &mut below[0];
            for &index in &level.dirty {
                let hash = salted_hash(&self.salts[depth], level.chunk(index), level.block);
                match above.last_mut() {
                    Some(above) => above.put(index * HASH_SIZE, &hash),
                    None => header[self.master..self.master + HASH_SIZE].copy_from_slice(&hash),
                }
            }
            level.store(entries, duplex)?;
        }
        Ok(())
    }
}

/// Everything of a save below its file system, read and checked.
struct Layers {
    /// The header the image was read with.
    header: Box<[u8; HEADER_SIZE]>,
    mac_key: Key,
    main: MainStorage,
    meta: Vec<MapEntry>,
    duplex: Duplex,
    /// The duplex bitmap and the master hashes of the two integrity
    /// layers: where the current and the other copy are in the header, and
    /// their size.
    masters: [(usize, usize, usize); 3],
    /// The physical block of each virtual block of the journal.
    journal: Vec<u64>,
    /// Where the journal's physical blocks start in the main storage.
    journal_area: u64,
    /// The journal's integrity layer; the journaled data is below it.
    data: HashTree,
    /// The allocation table's integrity layer, and the table below it.
    allocation: HashTree,
    fat: Level,
    fs_blocks: usize,
    /// First blocks of the directory and the file table.
    tables: [u32; 2],
}

impl Layers {
    fn read(image: &[u8], keys: &KeySet) -> Result<Self, SaveError> {
        let mac_key = crypto::save_mac_key(keys)?;
        if image.len() < 2 * HEADER_SIZE {
            return Err(SaveError::Truncated("header"));
        }
        let header = [0, HEADER_SIZE]
            .into_iter()
            .map(|at| &image[at..at + HEADER_SIZE])
            .find(|header| {
                crypto::sha256(&header[HASHED_RANGE]) == header[HEADER_HASH..HEADER_HASH + 32]
            })
            .ok_or(SaveError::HeaderHash)?;
        if &header[0x100..0x104] != DISF_MAGIC {
            return Err(SaveError::BadMagic("DISF"));
        }
        if crypto::cmac(&mac_key, &header[CMAC_RANGE]) != header[..16] {
            return Err(SaveError::Cmac);
        }
        if read_u32(header, 0x104) != DISF_VERSION {
            return Err(SaveError::Corrupt("not version 5 of the format"));
        }
        let layout: [u64; LAYOUT_WORDS] =
            std::array::from_fn(|index| read_u64(header, LAYOUT + index * 8));
        let word = |index: usize| layout[index];
        let pair = |index: usize| (layout[index], layout[index + 1]);
        let offset = |index: usize| usize::try_from(layout[index]).unwrap_or(usize::MAX);

        let main = MainStorage {
            entries: map_entries(
                header,
                MAIN_REMAP_HEADER,
                slice(
                    image,
                    word(L_MAIN_MAP_ENTRIES),
                    word(L_MAIN_MAP_ENTRIES + 1),
                    "remap entries",
                )?,
            )?,
            area: span(
                image.len(),
                word(L_MAIN_DATA),
                usize::try_from(word(L_MAIN_DATA + 1)).unwrap_or(usize::MAX),
                "remapped area",
            )?,
        };

        // Duplex: the master bitmap picks level-1 blocks, level 1 data
        // blocks. The duplex index picks the master bitmap and, as the
        // saves pair them, the master hashes.
        if &header[DUPLEX_HEADER..DUPLEX_HEADER + 4] != b"DPFS" {
            return Err(SaveError::BadMagic("duplex"));
        }
        let order = |layer: usize| read_u32(header, DUPLEX_HEADER + 8 + layer * 0x14 + 0x10);
        let selected = usize::from(word(L_DUPLEX_INDEX) & 0xFF != 0);
        let duplex_master_size = offset(L_DUPLEX_MASTER_SIZE);
        let masters = [
            (L_DUPLEX_MASTER_A, duplex_master_size),
            (L_IVFC_MASTER_A, HASH_SIZE),
            (L_FAT_IVFC_MASTER_A, HASH_SIZE),
        ]
        .map(|(a, size)| (offset(a + selected), offset(a + 1 - selected), size));
        for (current, other, size) in masters {
            if [current, other]
                .iter()
                .any(|at| at.checked_add(size).is_none_or(|end| end > HEADER_SIZE))
            {
                return Err(SaveError::Truncated("master"));
            }
        }
        let [
            (duplex_master, ..),
            (ivfc_master, ..),
            (fat_master, ..),
        ] = masters;
        let level1 = Duplex::read(
            &main,
            image,
            &header[duplex_master..duplex_master + duplex_master_size],
            (word(L_DUPLEX_L1_A), word(L_DUPLEX_L1_B)),
            word(L_DUPLEX_L1_SIZE),
            order(1),
        )?;
        let duplex = Duplex::read(
            &main,
            image,
            &level1.bytes,
            (word(L_DUPLEX_DATA_A), word(L_DUPLEX_DATA_B)),
            word(L_DUPLEX_DATA_SIZE),
            order(2),
        )?;
        let meta = map_entries(
            header,
            META_REMAP_HEADER,
            slice(
                image,
                word(L_META_MAP_ENTRIES),
                word(L_META_MAP_ENTRIES + 1),
                "remap entries",
            )?,
        )?;
        let meta_read = |(offset, size): (u64, u64)| duplex.meta_read(&meta, offset, size);

        // Journal.
        if &header[JOURNAL_HEADER..JOURNAL_HEADER + 4] != b"JNGL" {
            return Err(SaveError::BadMagic("journal"));
        }
        let blocks = u64::from(read_u32(header, JOURNAL_HEADER + 0x24));
        let journal_blocks = u64::from(read_u32(header, JOURNAL_HEADER + 0x28));
        if read_u64(header, JOURNAL_HEADER + 0x18) != BLOCK_SIZE as u64 {
            return Err(SaveError::Corrupt("journal blocks other than 0x4000 bytes"));
        }
        if read_u64(header, JOURNAL_HEADER + 8) != (blocks + journal_blocks) * BLOCK_SIZE as u64 {
            return Err(SaveError::Corrupt("journal sizes disagree"));
        }
        let table = meta_read(pair(L_JOURNAL_TABLE))?;
        let mut mapped = vec![false; (blocks + journal_blocks) as usize];
        let journal = (0..blocks as usize)
            .map(|virtual_block| {
                let entry = table
                    .get(virtual_block * 8..virtual_block * 8 + 8)
                    .ok_or(SaveError::Truncated("journal map table"))?;
                let physical = u64::from(read_u32(entry, 0) & !CHAIN_FLAG);
                if physical >= blocks + journal_blocks
                    || std::mem::replace(&mut mapped[physical as usize], true)
                {
                    return Err(SaveError::Corrupt(
                        "journal map table entry out of range or repeated",
                    ));
                }
                Ok(physical)
            })
            .collect::<Result<Vec<_>, SaveError>>()?;
        let journal_area = word(L_JOURNAL_DATA);

        // Integrity of the journaled data and of the allocation table.
        for at in [IVFC_HEADER, FAT_IVFC_HEADER] {
            if &header[at..at + 4] != b"IVFC" {
                return Err(SaveError::BadMagic("integrity"));
            }
            if read_u32(header, at + 8) != HASH_SIZE as u32 {
                return Err(SaveError::Corrupt("master hash size"));
            }
        }
        let ivfc_level = |at: usize, index: usize| {
            (
                read_u64(header, at + 0x10 + index * 0x18),
                read_u64(header, at + 0x18 + index * 0x18),
                read_u32(header, at + 0x20 + index * 0x18),
            )
        };
        let seed = |at: usize| &header[at + 0xA0..at + 0xC0];
        let data = HashTree::read(
            "journal integrity",
            header,
            ivfc_master,
            seed(IVFC_HEADER),
            &[0, 1, 2].map(|index| ivfc_level(IVFC_HEADER, index)),
            &meta,
            &duplex,
        )?;
        let (data_offset, data_size, data_order) = ivfc_level(IVFC_HEADER, 3);
        if data_offset != 0 || block_size(data_order)? != BLOCK_SIZE {
            return Err(SaveError::Corrupt(
                "journaled data not in 0x4000-byte blocks from 0",
            ));
        }
        if data_size > blocks * BLOCK_SIZE as u64 {
            return Err(SaveError::Truncated("journaled data"));
        }
        let data_size = data_size as usize;

        let fat_levels = [0, 1, 2].map(|index| ivfc_level(FAT_IVFC_HEADER, index));
        let allocation = HashTree::read(
            "allocation table integrity",
            header,
            fat_master,
            seed(FAT_IVFC_HEADER),
            &fat_levels[..2],
            &meta,
            &duplex,
        )?;
        let (fat_offset, fat_size, fat_order) = fat_levels[2];
        let fat_block = block_size(fat_order)?;
        let (hashes, salt) = allocation.bottom();
        let fat = Level {
            bytes: check_level(
                "allocation table integrity",
                3,
                meta_read((fat_offset, fat_size))?,
                hashes,
                salt,
                fat_block,
            )?,
            offset: fat_offset,
            block: fat_block,
            dirty: BTreeSet::new(),
        };

        // The file system.
        if &header[SAVE_HEADER..SAVE_HEADER + 4] != b"SAVE" {
            return Err(SaveError::BadMagic("save file system"));
        }
        let fs_blocks = read_u32(header, FAT_HEADER + 0x10) as usize;
        if read_u64(header, FAT_HEADER) != BLOCK_SIZE as u64 {
            return Err(SaveError::Corrupt("file system blocks other than 0x4000 bytes"));
        }
        if fs_blocks * BLOCK_SIZE > data_size || (fs_blocks + 1) * 8 > fat.bytes.len() {
            return Err(SaveError::Corrupt("file system larger than its storage"));
        }

        let layers = Self {
            header: Box::new(header.try_into().expect("headers are HEADER_SIZE long")),
            mac_key,
            main,
            meta,
            duplex,
            masters,
            journal,
            journal_area,
            data,
            allocation,
            fat,
            fs_blocks,
            tables: [
                read_u32(header, FAT_HEADER + 0x28),
                read_u32(header, FAT_HEADER + 0x2C),
            ],
        };
        // Every data block written has to match its hash.
        let (hashes, salt) = layers.data.bottom();
        for index in 0..data_size.div_ceil(BLOCK_SIZE) {
            let expected = hashes
                .get(index * HASH_SIZE..(index + 1) * HASH_SIZE)
                .ok_or(SaveError::Corrupt("hash level smaller than the level below"))?;
            if expected.iter().any(|&byte| byte != 0) {
                let block = layers.journal_block(image, index)?;
                let length = BLOCK_SIZE.min(data_size - index * BLOCK_SIZE);
                written("journal integrity", 4, index, salt, expected, &block[..length], BLOCK_SIZE)?;
            }
        }
        Ok(layers)
    }

    fn extra(&self) -> ExtraData {
        let extra = &self.header[EXTRA_DATA[0]..EXTRA_DATA[0] + EXTRA_DATA_SIZE];
        ExtraData {
            save_data_id: read_u64(extra, 0x18),
            owner_id: read_u64(extra, 0x40),
            flags: read_u32(extra, 0x50),
            size: read_u64(extra, 0x58),
            journal_size: read_u64(extra, 0x60),
        }
    }

    /// Where the journal keeps data block `index` in the main storage.
    fn journal_offset(&self, index: usize) -> u64 {
        self.journal_area + self.journal[index] * BLOCK_SIZE as u64
    }

    /// What the journal's physical block for data block `index` holds.
    fn journal_block<'a>(&self, image: &'a [u8], index: usize) -> Result<Cow<'a, [u8]>, SaveError> {
        self.main
            .read(image, self.journal_offset(index), BLOCK_SIZE, "journal")
    }

    /// Data block `index` as FS reads it.
    fn data_block<'a>(&self, image: &'a [u8], index: usize) -> Result<Cow<'a, [u8]>, SaveError> {
        if index >= self.fs_blocks {
            return Err(SaveError::Corrupt("block outside the file system"));
        }
        let hash = &self.data.bottom().0[index * HASH_SIZE..(index + 1) * HASH_SIZE];
        if hash.iter().all(|&byte| byte == 0) {
            Ok(Cow::Borrowed(&ZERO_BLOCK))
        } else {
            self.journal_block(image, index)
        }
    }

    /// Commits `changed` data blocks and the changed allocation table into
    /// `image`.
    fn commit(
        &mut self,
        image: &mut [u8],
        changed: &BTreeMap<usize, Box<[u8]>>,
    ) -> Result<(), SaveError> {
        for (&index, block) in changed {
            self.main
                .write(image, self.journal_offset(index), block, "journal")?;
            self.data.hash_below(index, block, BLOCK_SIZE);
        }
        for &index in &self.fat.dirty {
            self.allocation
                .hash_below(index, self.fat.chunk(index), self.fat.block);
        }
        self.fat.store(&self.meta, &mut self.duplex)?;
        let header = &mut self.header[..];
        self.data.commit(header, &self.meta, &mut self.duplex)?;
        self.allocation
            .commit(header, &self.meta, &mut self.duplex)?;
        for index in std::mem::take(&mut self.duplex.dirty) {
            let start = index * self.duplex.block;
            let end = (start + self.duplex.block).min(self.duplex.bytes.len());
            self.main.write(
                image,
                self.duplex.copies[index],
                &self.duplex.bytes[start..end],
                "duplex storage",
            )?;
        }

        // The other duplex bitmap and master hashes describe the state
        // before the last commit; they become this one, so that both read
        // the same.
        for (current, other, size) in self.masters {
            header.copy_within(current..current + size, other);
        }

        let hash = crypto::sha256(&header[HASHED_RANGE]);
        header[HEADER_HASH..HEADER_HASH + 32].copy_from_slice(&hash);
        let mac = crypto::cmac(&self.mac_key, &header[CMAC_RANGE]);
        header[..16].copy_from_slice(&mac);
        image[..HEADER_SIZE].copy_from_slice(header);
        image[HEADER_SIZE..2 * HEADER_SIZE].copy_from_slice(header);
        Ok(())
    }
}

const DIRECTORIES: usize = 0;
const FILES: usize = 1;

/// A table of the file system and the blocks that hold it.
struct StoredTable {
    blocks: Vec<usize>,
    bytes: Vec<u8>,
}

fn table_entry(table: &[u8], index: u32) -> Result<&[u8], SaveError> {
    let start = index as usize * TABLE_ENTRY_SIZE;
    table
        .get(start..start + TABLE_ENTRY_SIZE)
        .ok_or(SaveError::Corrupt("table index out of range"))
}

/// A file as its entry describes it.
struct FileEntry {
    /// Its index in the file table.
    index: u32,
    start: u32,
    size: u64,
}

impl FileEntry {
    fn new(index: u32, entry: &[u8]) -> Self {
        Self {
            index,
            start: read_u32(entry, 0x48),
            size: read_u64(entry, 0x4C),
        }
    }
}

/// What a path names.
enum Found {
    Directory(u32),
    File(FileEntry),
}

/// An entry the walk of the tree found.
struct Walked {
    path: String,
    /// What a file holds; directories have none.
    file: Option<FileEntry>,
}

/// The file system of a save: the allocation table and the tables over
/// the journaled data, `changed` blocks laid over the image's.
struct FileSystem<'a> {
    layers: &'a Layers,
    image: &'a [u8],
    changed: &'a BTreeMap<usize, Box<[u8]>>,
}

impl<'a> FileSystem<'a> {
    fn block(&self, index: usize) -> Result<Cow<'a, [u8]>, SaveError> {
        match self.changed.get(&index) {
            Some(block) => Ok(Cow::Borrowed(&block[..])),
            None => self.layers.data_block(self.image, index),
        }
    }

    fn fat(&self) -> &'a [u8] {
        &self.layers.fat.bytes[..(self.layers.fs_blocks + 1) * 8]
    }

    /// The segments (first block, length) of the list starting at block
    /// `start`, checking the links.
    fn segments(&self, start: u32) -> Result<Vec<(usize, usize)>, SaveError> {
        let fat = self.fat();
        let entries = fat.len() / 8;
        let corrupt = SaveError::Corrupt("allocation table chain");
        let mut out = Vec::new();
        let mut total = 0;
        let mut index = start as usize + 1;
        let mut previous = None;
        loop {
            if index == 0 || index >= entries || total >= entries {
                return Err(corrupt);
            }
            let (prev, next) = (read_u32(fat, index * 8), read_u32(fat, index * 8 + 4));
            let expected_prev = previous.map_or(CHAIN_FLAG, |previous: usize| previous as u32);
            if prev != expected_prev {
                return Err(corrupt);
            }
            let last = if next & CHAIN_FLAG != 0 {
                if index + 1 >= entries {
                    return Err(corrupt);
                }
                let range = read_u32(fat, (index + 1) * 8 + 4) as usize;
                if read_u32(fat, (index + 1) * 8) != CHAIN_FLAG | index as u32
                    || range < index + 1
                    || range >= entries
                {
                    return Err(corrupt);
                }
                range
            } else {
                index
            };
            out.push((index - 1, last - index + 1));
            total += last - index + 1;
            let next = (next & !CHAIN_FLAG) as usize;
            if next == 0 {
                return Ok(out);
            }
            previous = Some(index);
            index = next;
        }
    }

    /// The blocks of the list starting at block `start`, in order.
    fn blocks(&self, start: u32) -> Result<Vec<usize>, SaveError> {
        Ok(self
            .segments(start)?
            .into_iter()
            .flat_map(|(first, length)| first..first + length)
            .collect())
    }

    fn chain(&self, start: u32) -> Result<Vec<u8>, SaveError> {
        let blocks = self.blocks(start)?;
        let mut out = Vec::with_capacity(blocks.len() * BLOCK_SIZE);
        for block in blocks {
            out.extend_from_slice(&self.block(block)?);
        }
        Ok(out)
    }

    fn contents(&self, file: &FileEntry) -> Result<Vec<u8>, SaveError> {
        if file.start == EMPTY_FILE_START {
            return Ok(Vec::new());
        }
        let mut data = self.chain(file.start)?;
        let size = usize::try_from(file.size).unwrap_or(usize::MAX);
        if data.len() < size {
            return Err(SaveError::Corrupt("file longer than its blocks"));
        }
        data.truncate(size);
        Ok(data)
    }

    /// The directory and the file table.
    fn tables(&self) -> Result<[StoredTable; 2], SaveError> {
        let table = |start: u32| -> Result<StoredTable, SaveError> {
            Ok(StoredTable {
                blocks: self.blocks(start)?,
                bytes: self.chain(start)?,
            })
        };
        Ok([table(self.layers.tables[0])?, table(self.layers.tables[1])?])
    }

    /// The children of directory `directory` in table `table`, as the
    /// directory lists them at 0x48 (directories) or 0x4C (files).
    fn children<'t>(
        &self,
        tables: &'t [StoredTable; 2],
        directory: u32,
        table: usize,
    ) -> Result<Vec<(u32, &'t [u8])>, SaveError> {
        let this = table_entry(&tables[DIRECTORIES].bytes, directory)?;
        let limit = tables[table].bytes.len() / TABLE_ENTRY_SIZE;
        let mut out = Vec::new();
        let mut child = read_u32(this, 0x48 + 4 * table);
        while child != 0 {
            if out.len() > limit {
                return Err(SaveError::Corrupt("sibling cycle"));
            }
            let entry = table_entry(&tables[table].bytes, child)?;
            if read_u32(entry, 0) != directory {
                return Err(SaveError::Corrupt("entry listed under another directory"));
            }
            out.push((child, entry));
            child = read_u32(entry, 0x44);
        }
        Ok(out)
    }

    /// Where `path` would be: the directory holding its last component and
    /// what has that name there, if anything.
    fn locate(
        &self,
        tables: &[StoredTable; 2],
        path: &str,
    ) -> Result<(u32, Option<Found>), SaveError> {
        let parts = components(path)?;
        let mut parent = ROOT_INDEX;
        for (depth, part) in parts.iter().enumerate() {
            let directory = self
                .children(tables, parent, DIRECTORIES)?
                .into_iter()
                .find(|(_, entry)| name_of(entry) == *part)
                .map(|(index, _)| Found::Directory(index));
            let found = match directory {
                Some(found) => Some(found),
                None => self
                    .children(tables, parent, FILES)?
                    .into_iter()
                    .find(|(_, entry)| name_of(entry) == *part)
                    .map(|(index, entry)| Found::File(FileEntry::new(index, entry))),
            };
            if depth + 1 == parts.len() {
                return Ok((parent, found));
            }
            match found {
                Some(Found::Directory(index)) => parent = index,
                _ => return Err(SaveError::NotFound(path.into())),
            }
        }
        unreachable!("components gives at least one")
    }

    /// Every directory and file, parents before children, each
    /// directory's files before its directories.
    fn nodes(&self) -> Result<Vec<Walked>, SaveError> {
        let tables = self.tables()?;
        let mut out = Vec::new();
        self.walk(&tables, ROOT_INDEX, "", &mut out, 0)?;
        Ok(out)
    }

    fn walk(
        &self,
        tables: &[StoredTable; 2],
        directory: u32,
        path: &str,
        out: &mut Vec<Walked>,
        depth: usize,
    ) -> Result<(), SaveError> {
        if depth > tables[DIRECTORIES].bytes.len() / TABLE_ENTRY_SIZE {
            return Err(SaveError::Corrupt("directory cycle"));
        }
        let child_path = |entry: &[u8]| -> Result<String, SaveError> {
            let name = std::str::from_utf8(name_of(entry))
                .map_err(|_| SaveError::Corrupt("entry name"))?;
            Ok(if path.is_empty() {
                name.into()
            } else {
                format!("{path}/{name}")
            })
        };
        for (index, entry) in self.children(tables, directory, FILES)? {
            let file = FileEntry::new(index, entry);
            if file.start != EMPTY_FILE_START {
                let blocks = self.segments(file.start)?.iter().map(|&(_, length)| length).sum::<usize>();
                if ((blocks * BLOCK_SIZE) as u64) < file.size {
                    return Err(SaveError::Corrupt("file longer than its blocks"));
                }
            }
            out.push(Walked {
                path: child_path(entry)?,
                file: Some(file),
            });
        }
        for (index, entry) in self.children(tables, directory, DIRECTORIES)? {
            let child_path = child_path(entry)?;
            out.push(Walked {
                path: child_path.clone(),
                file: None,
            });
            self.walk(tables, index, &child_path, out, depth + 1)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys() -> KeySet {
        KeySet::parse(
            "aes_kek_generation_source = 000102030405060708090a0b0c0d0e0f\n\
             device_key = 101112131415161718191a1b1c1d1e1f\n\
             save_mac_kek_source = 202122232425262728292a2b2c2d2e2f\n\
             save_mac_key_source_00 = 303132333435363738393a3b3c3d3e3f\n",
        )
        .unwrap()
    }

    fn save() -> SystemSave {
        SystemSave {
            id: 0x8000_0000_0000_0120,
            owner_id: 0,
            flags: 3,
            size: 0x6C000,
        }
    }

    fn sample_entries() -> Vec<Entry> {
        vec![
            Entry::Directory("meta".into()),
            Entry::File(
                "meta/imkvdb.arc".into(),
                (0..0x5123).map(|i| i as u8).collect(),
            ),
            Entry::File("lastPublishedId".into(), vec![0; 8]),
            Entry::Directory("a/b/c".into()),
            Entry::File("a/b/empty".into(), Vec::new()),
            Entry::File("a/one".into(), vec![1; BLOCK_SIZE]),
        ]
    }

    #[test]
    fn a_built_save_reads_back_with_every_check_passing() {
        let keys = keys();
        let image = system_save(&save(), &sample_entries(), &keys).unwrap();
        assert_eq!(image.len(), 0x128000);
        let read = open(&image, &keys).unwrap();
        assert_eq!(
            read.extra,
            ExtraData {
                save_data_id: 0x8000_0000_0000_0120,
                owner_id: 0,
                flags: 3,
                size: 0x6C000,
                journal_size: 0x6C000,
            }
        );
        for entry in sample_entries() {
            if let Entry::File(path, data) = entry {
                assert_eq!(read.file(&path), Some(&data[..]), "{path}");
            }
        }
        let directories: Vec<&str> = read
            .entries
            .iter()
            .filter_map(|entry| match entry {
                Entry::Directory(path) => Some(path.as_str()),
                Entry::File(..) => None,
            })
            .collect();
        assert_eq!(directories, ["a", "a/b", "a/b/c", "meta"]);
    }

    #[test]
    fn a_flipped_bit_anywhere_it_matters_is_caught() {
        let keys = keys();
        let image = system_save(&save(), &sample_entries(), &keys).unwrap();
        let journal = MAIN_DATA + 2 * DUPLEX_L1_SIZE + 2 * DUPLEX_DATA_SIZE;
        let duplex = MAIN_DATA + 2 * DUPLEX_L1_SIZE;
        let journal_block_2 = SaveError::Hash {
            layer: "journal integrity",
            level: 4,
            block: 2,
        };
        let fat = SaveError::Hash {
            layer: "allocation table integrity",
            level: 3,
            block: 0,
        };
        for (at, expected) in [
            (0x130, SaveError::Cmac),
            (0x2000, SaveError::HeaderHash),
            (journal + 2 * BLOCK_SIZE + 5, journal_block_2),
            (duplex + META_REGIONS[META_FAT].0 + 0x10, fat),
        ] {
            let mut damaged = image.clone();
            damaged[at] ^= 1;
            // A damaged first header falls back to the second, which is
            // intact; damage both to see the header checks.
            if at < HEADER_SIZE {
                damaged[at + HEADER_SIZE] ^= 1;
            }
            assert_eq!(open(&damaged, &keys).unwrap_err(), expected, "{at:#x}");
        }
    }

    #[test]
    fn another_consoles_key_fails_the_cmac() {
        let image = system_save(&save(), &[], &keys()).unwrap();
        let other = KeySet::parse(
            "aes_kek_generation_source = 000102030405060708090a0b0c0d0e0f\n\
             device_key = 111112131415161718191a1b1c1d1e1f\n\
             save_mac_kek_source = 202122232425262728292a2b2c2d2e2f\n\
             save_mac_key_source_00 = 303132333435363738393a3b3c3d3e3f\n",
        )
        .unwrap();
        assert_eq!(open(&image, &other).unwrap_err(), SaveError::Cmac);
    }

    #[test]
    fn sizes_without_a_known_layout_and_overfull_saves_are_refused() {
        let keys = keys();
        for size in [
            0x6C001,
            BLOCK_SIZE as u64,
            (MAX_BLOCKS as u64 + 1) * BLOCK_SIZE as u64,
        ] {
            let save = SystemSave { size, ..save() };
            assert!(matches!(
                system_save(&save, &[], &keys),
                Err(SaveError::UnsupportedSize { .. })
            ));
        }
        let save = SystemSave {
            size: 4 * BLOCK_SIZE as u64,
            ..save()
        };
        let big = Entry::File("big".into(), vec![1; 3 * BLOCK_SIZE]);
        assert_eq!(
            system_save(&save, &[big], &keys),
            Err(SaveError::Full {
                needed: 5,
                blocks: 4
            })
        );
    }

    /// Where the built image keeps physical journal block `block`.
    fn journal_block(block: usize) -> std::ops::Range<usize> {
        let start = MAIN_DATA + 2 * DUPLEX_L1_SIZE + 2 * DUPLEX_DATA_SIZE + block * BLOCK_SIZE;
        start..start + BLOCK_SIZE
    }

    /// The blocks of the sample files: tables, imkvdb.arc (two),
    /// lastPublishedId and a/one; the rest of the 27 are free.
    const SAMPLE_USED: usize = 6;

    fn directories(entries: &[Entry]) -> Vec<&str> {
        entries
            .iter()
            .filter_map(|entry| match entry {
                Entry::Directory(path) => Some(path.as_str()),
                Entry::File(..) => None,
            })
            .collect()
    }

    #[test]
    fn a_block_never_written_reads_as_zeros_whatever_it_holds() {
        let keys = keys();
        let mut image = system_save(&save(), &sample_entries(), &keys).unwrap();
        let clean = open(&image, &keys).unwrap();
        for block in SAMPLE_USED..27 {
            image[journal_block(block)].fill(0xA5);
        }
        assert_eq!(open(&image, &keys).unwrap(), clean);

        // A new file's last block reads as zeros past what it was given,
        // and the garbage of the blocks it did not touch stays.
        let mut editor = SaveEditor::open(image.clone(), &keys).unwrap();
        let contents: Vec<u8> = (0..BLOCK_SIZE + 0x10).map(|i| (i * 7) as u8).collect();
        editor.create_file("new", &contents).unwrap();
        let edited = editor.finish().unwrap();
        let block = &edited[journal_block(SAMPLE_USED + 1)];
        assert_eq!(&block[..0x10], &contents[BLOCK_SIZE..]);
        assert!(block[0x10..].iter().all(|&byte| byte == 0));
        for block in SAMPLE_USED + 2..27 {
            assert_eq!(edited[journal_block(block)], image[journal_block(block)]);
        }
        assert_eq!(open(&edited, &keys).unwrap().file("new"), Some(&contents[..]));
    }

    #[test]
    fn an_edited_save_reads_back_with_every_check_passing() {
        let keys = keys();
        let image = system_save(&save(), &sample_entries(), &keys).unwrap();
        let mut editor = SaveEditor::open(image.clone(), &keys).unwrap();
        assert_eq!(
            editor.read("meta/imkvdb.arc").unwrap(),
            open(&image, &keys).unwrap().file("meta/imkvdb.arc").unwrap()
        );
        // Across the end of the file's first block.
        editor
            .write("meta/imkvdb.arc", BLOCK_SIZE as u64 - 0x10, &[0xEE; 0x20])
            .unwrap();
        editor.create_directory("meta/sub").unwrap();
        let big: Vec<u8> = (0..0x5000).map(|i| (i * 13) as u8).collect();
        editor.create_file("meta/sub/big", &big).unwrap();
        editor.create_file("empty", &[]).unwrap();
        assert_eq!(
            editor.list().unwrap()[..2],
            [
                Listing::File {
                    path: "empty".into(),
                    size: 0
                },
                Listing::File {
                    path: "lastPublishedId".into(),
                    size: 8
                },
            ]
        );
        let edited = editor.finish().unwrap();
        assert_eq!(edited.len(), image.len());
        assert_eq!(edited[..HEADER_SIZE], edited[HEADER_SIZE..2 * HEADER_SIZE]);
        for [a, b] in [DUPLEX_MASTER, IVFC_MASTER, FAT_IVFC_MASTER] {
            assert_eq!(edited[a..a + HASH_SIZE], edited[b..b + HASH_SIZE]);
        }
        // The new file took the first free blocks.
        assert_eq!(edited[journal_block(SAMPLE_USED)], big[..BLOCK_SIZE]);

        let read = open(&edited, &keys).unwrap();
        let mut arc: Vec<u8> = (0..0x5123).map(|i| i as u8).collect();
        arc[BLOCK_SIZE - 0x10..BLOCK_SIZE + 0x10].fill(0xEE);
        assert_eq!(read.file("meta/imkvdb.arc"), Some(&arc[..]));
        assert_eq!(read.file("meta/sub/big"), Some(&big[..]));
        assert_eq!(read.file("empty"), Some(&[][..]));
        assert_eq!(read.file("a/one"), Some(&[1; BLOCK_SIZE][..]));
        assert_eq!(
            directories(&read.entries),
            ["a", "a/b", "a/b/c", "meta", "meta/sub"]
        );

        // Edited again from the edited image.
        let mut again = SaveEditor::open(edited, &keys).unwrap();
        again.create_file("meta/sub/more", &[3; 10]).unwrap();
        let read = open(&again.finish().unwrap(), &keys).unwrap();
        assert_eq!(read.file("meta/sub/more"), Some(&[3; 10][..]));
        assert_eq!(read.file("meta/sub/big"), Some(&big[..]));
    }

    #[test]
    fn an_edit_changes_only_what_it_has_to() {
        let keys = keys();
        let image = system_save(&save(), &sample_entries(), &keys).unwrap();
        let unchanged = SaveEditor::open(image.clone(), &keys).unwrap();
        assert!(unchanged.finish().unwrap() == image);

        let mut editor = SaveEditor::open(image.clone(), &keys).unwrap();
        editor.write("a/one", 5, &[9; 3]).unwrap();
        let edited = editor.finish().unwrap();
        let duplex = MAIN_DATA + 2 * DUPLEX_L1_SIZE..MAIN_DATA + 2 * DUPLEX_L1_SIZE + 2 * DUPLEX_DATA_SIZE;
        // a/one is the sixth block, after the tables, imkvdb.arc and
        // lastPublishedId.
        let block = journal_block(5);
        assert_eq!(edited[block.start + 5..block.start + 8], [9; 3]);
        for (at, (old, new)) in image.iter().zip(&edited).enumerate() {
            if old != new {
                assert!(
                    at < 2 * HEADER_SIZE || duplex.contains(&at) || block.contains(&at),
                    "{at:#x}"
                );
            }
        }
        // Only the selected (A) duplex copy of the meta area changed.
        let copy_b = duplex.start + DUPLEX_DATA_SIZE..duplex.end;
        assert_eq!(edited[copy_b.clone()], image[copy_b]);
    }

    #[test]
    fn a_deleted_file_gives_back_its_blocks_and_entry() {
        let keys = keys();
        let image = system_save(&save(), &sample_entries(), &keys).unwrap();
        let mut editor = SaveEditor::open(image.clone(), &keys).unwrap();
        editor.delete_file("meta/imkvdb.arc").unwrap();
        editor.delete_file("a/b/empty").unwrap();
        assert_eq!(
            editor.read("meta/imkvdb.arc"),
            Err(SaveError::NotFound("meta/imkvdb.arc".into()))
        );
        // A bigger replacement: the two freed blocks, then one of the free
        // run after the used blocks; its entry is the one freed last.
        let bigger = vec![0x77; 2 * BLOCK_SIZE + 1];
        editor.create_file("meta/imkvdb.arc", &bigger).unwrap();
        let edited = editor.finish().unwrap();
        assert_eq!(edited[journal_block(2)], bigger[..BLOCK_SIZE]);
        assert_eq!(edited[journal_block(SAMPLE_USED)][0], 0x77);
        let read = open(&edited, &keys).unwrap();
        assert_eq!(read.file("meta/imkvdb.arc"), Some(&bigger[..]));
        assert_eq!(read.file("a/b/empty"), None);
        assert_eq!(read.file("lastPublishedId"), Some(&[0; 8][..]));
        assert_eq!(read.file("a/one"), Some(&[1; BLOCK_SIZE][..]));
        assert_eq!(read.entries.len(), open(&image, &keys).unwrap().entries.len() - 1);
    }

    #[test]
    fn edits_that_cannot_be_made_are_refused() {
        let keys = keys();
        let image = system_save(&save(), &sample_entries(), &keys).unwrap();
        let mut editor = SaveEditor::open(image, &keys).unwrap();
        assert_eq!(
            editor.create_file("nowhere/file", &[1]),
            Err(SaveError::NotFound("nowhere/file".into()))
        );
        assert_eq!(
            editor.create_file("meta/imkvdb.arc", &[1]),
            Err(SaveError::AlreadyExists("meta/imkvdb.arc".into()))
        );
        assert_eq!(
            editor.create_directory("meta"),
            Err(SaveError::AlreadyExists("meta".into()))
        );
        assert_eq!(
            editor.write("lastPublishedId", 4, &[0; 5]),
            Err(SaveError::PastEnd("lastPublishedId".into()))
        );
        assert_eq!(
            editor.write("meta", 0, &[0]),
            Err(SaveError::NotFound("meta".into()))
        );
        assert_eq!(
            editor.create_file("huge", &vec![0; 22 * BLOCK_SIZE]),
            Err(SaveError::NoSpace {
                needed: 22,
                free: 27 - SAMPLE_USED
            })
        );
        assert_eq!(editor.finish(), Ok(system_save(&save(), &sample_entries(), &keys).unwrap()));
    }
}
