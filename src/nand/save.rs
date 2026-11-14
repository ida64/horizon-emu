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
//!   block never written, which reads as zeros.
//! * The file system (`SAVE` at 0x608 with the allocation table header at
//!   0x620): a FAT-like allocation table of 0x4000-byte blocks and two
//!   tables of 0x60-byte entries (parent index, 0x40-byte name, next
//!   sibling, a 0x14-byte value, next in the table's list) for directories
//!   and files, each a chain of blocks. Entry 0 of a table holds its length
//!   and capacity and heads the free list, entry 1 heads the list of
//!   entries in use, entry 2 of the directory table is the root.
//!
//! This is the structure LibHac reads (`Tools/FsSystem/Save`). Every hash
//! and the CMAC of the three saves the guest's FS wrote itself check out
//! with it ([`open`]): NCM's empty database, NCM's database after an import
//! (one directory, one file) and the save data indexer (two files). In all
//! three the A copies of the master hashes are current and the B copies,
//! where they differ, fail: they hold an earlier commit's state, so the
//! master hashes are read from copy A. The duplex index selects the master
//! bitmap. FS hashes every block it allocated, a block of zeros too (the
//! indexer's `lastPublishedId`), and leaves the others' hashes zero.
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

use std::fmt;

use super::{bis::MissingKey, crypto};
use crate::keys::KeySet;

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

fn salted_hash(salt: &[u8; 32], block: &[u8]) -> [u8; HASH_SIZE] {
    let mut salted = Vec::with_capacity(salt.len() + block.len());
    salted.extend_from_slice(salt);
    salted.extend_from_slice(block);
    let mut hash = crypto::sha256(&salted);
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
                .copy_from_slice(&salted_hash(salt, block));
        }
        below = hashes.to_vec();
        written = usize::MAX;
    }
    salted_hash(&salts[0], &below)
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

// Reading.

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

struct MapEntry {
    virtual_offset: u64,
    physical: u64,
    size: u64,
}

/// A remap storage over `base`.
struct Remap<'a> {
    entries: Vec<MapEntry>,
    base: &'a [u8],
}

impl Remap<'_> {
    fn read(
        &self,
        virtual_offset: u64,
        size: u64,
        what: &'static str,
    ) -> Result<Vec<u8>, SaveError> {
        let mut out = Vec::with_capacity(size as usize);
        let mut at = virtual_offset;
        while at < virtual_offset + size {
            let entry = self
                .entries
                .iter()
                .find(|entry| {
                    (entry.virtual_offset..entry.virtual_offset + entry.size).contains(&at)
                })
                .ok_or(SaveError::Corrupt(what))?;
            let length = (entry.virtual_offset + entry.size).min(virtual_offset + size) - at;
            let physical = (entry.physical + at - entry.virtual_offset) as usize;
            out.extend_from_slice(
                self.base
                    .get(physical..physical + length as usize)
                    .ok_or(SaveError::Truncated(what))?,
            );
            at += length;
        }
        Ok(out)
    }
}

fn remap<'a>(
    header: &[u8],
    at: usize,
    entries: &[u8],
    base: &'a [u8],
) -> Result<Remap<'a>, SaveError> {
    if &header[at..at + 4] != b"RMAP" {
        return Err(SaveError::BadMagic("remap storage"));
    }
    let count = read_u32(header, at + 8) as usize;
    let entries = (0..count)
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
        .collect::<Result<_, SaveError>>()?;
    Ok(Remap { entries, base })
}

fn slice<'a>(
    image: &'a [u8],
    offset: u64,
    size: u64,
    what: &'static str,
) -> Result<&'a [u8], SaveError> {
    usize::try_from(offset)
        .ok()
        .zip(usize::try_from(size).ok())
        .and_then(|(offset, size)| image.get(offset..offset.checked_add(size)?))
        .ok_or(SaveError::Truncated(what))
}

/// Checks a save image completely — header hash and CMAC, every hash of
/// both integrity layers, the journal map, the allocation chains and the
/// tables — and returns what it holds.
pub fn open(image: &[u8], keys: &KeySet) -> Result<SaveData, SaveError> {
    let mac_key = crypto::save_mac_key(keys)?;
    let header = [0, HEADER_SIZE]
        .into_iter()
        .filter_map(|at| image.get(at..at + HEADER_SIZE))
        .find(|header| {
            crypto::sha256(&header[HASHED_RANGE]) == header[HEADER_HASH..HEADER_HASH + 32]
        })
        .ok_or(if image.len() < HEADER_SIZE {
            SaveError::Truncated("header")
        } else {
            SaveError::HeaderHash
        })?;
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

    let main = remap(
        header,
        MAIN_REMAP_HEADER,
        slice(
            image,
            word(L_MAIN_MAP_ENTRIES),
            word(L_MAIN_MAP_ENTRIES + 1),
            "remap entries",
        )?,
        slice(
            image,
            word(L_MAIN_DATA),
            word(L_MAIN_DATA + 1),
            "remapped area",
        )?,
    )?;

    // Duplex: the master bitmap picks level-1 blocks, level 1 data blocks.
    if &header[DUPLEX_HEADER..DUPLEX_HEADER + 4] != b"DPFS" {
        return Err(SaveError::BadMagic("duplex"));
    }
    let order = |layer: usize| read_u32(header, DUPLEX_HEADER + 8 + layer * 0x14 + 0x10);
    let master_at = if word(L_DUPLEX_INDEX) & 0xFF == 0 {
        L_DUPLEX_MASTER_A
    } else {
        L_DUPLEX_MASTER_B
    };
    let master = slice(
        header,
        word(master_at),
        word(L_DUPLEX_MASTER_SIZE),
        "duplex master",
    )?;
    let duplex_level =
        |selector: &[u8], a: u64, b: u64, size: u64, order: u32| -> Result<Vec<u8>, SaveError> {
            let block = 1u64 << order;
            if size.div_ceil(block) > selector.len() as u64 * 8 {
                return Err(SaveError::Corrupt("duplex bitmap smaller than its level"));
            }
            let mut out = Vec::with_capacity(size as usize);
            for index in 0..size.div_ceil(block) {
                let copy = if bit(selector, index as usize) { b } else { a };
                let length = block.min(size - index * block);
                out.extend(main.read(copy + index * block, length, "duplex storage")?);
            }
            Ok(out)
        };
    let level1 = duplex_level(
        master,
        word(L_DUPLEX_L1_A),
        word(L_DUPLEX_L1_B),
        word(L_DUPLEX_L1_SIZE),
        order(1),
    )?;
    let duplex = duplex_level(
        &level1,
        word(L_DUPLEX_DATA_A),
        word(L_DUPLEX_DATA_B),
        word(L_DUPLEX_DATA_SIZE),
        order(2),
    )?;
    let meta = remap(
        header,
        META_REMAP_HEADER,
        slice(
            image,
            word(L_META_MAP_ENTRIES),
            word(L_META_MAP_ENTRIES + 1),
            "remap entries",
        )?,
        &duplex,
    )?;
    let meta_read = |(offset, size): (u64, u64)| meta.read(offset, size, "meta remap storage");

    // Journal.
    if &header[JOURNAL_HEADER..JOURNAL_HEADER + 4] != b"JNGL" {
        return Err(SaveError::BadMagic("journal"));
    }
    let block_size = read_u64(header, JOURNAL_HEADER + 0x18);
    let blocks = u64::from(read_u32(header, JOURNAL_HEADER + 0x24));
    let journal_blocks = u64::from(read_u32(header, JOURNAL_HEADER + 0x28));
    if block_size == 0
        || read_u64(header, JOURNAL_HEADER + 8) != (blocks + journal_blocks) * block_size
    {
        return Err(SaveError::Corrupt("journal sizes disagree"));
    }
    let table = meta_read(pair(L_JOURNAL_TABLE))?;
    let journal_area = word(L_JOURNAL_DATA);
    let mut core = Vec::with_capacity((blocks * block_size) as usize);
    let mut mapped = vec![false; (blocks + journal_blocks) as usize];
    for virtual_block in 0..blocks as usize {
        let physical = u64::from(
            read_u32(
                table
                    .get(virtual_block * 8..virtual_block * 8 + 8)
                    .ok_or(SaveError::Truncated("journal map table"))?,
                0,
            ) & !CHAIN_FLAG,
        );
        if physical >= blocks + journal_blocks
            || std::mem::replace(&mut mapped[physical as usize], true)
        {
            return Err(SaveError::Corrupt(
                "journal map table entry out of range or repeated",
            ));
        }
        core.extend(main.read(journal_area + physical * block_size, block_size, "journal")?);
    }

    // Integrity of the journaled data and of the allocation table.
    let ivfc_level = |at: usize, index: usize| {
        (
            read_u64(header, at + 0x10 + index * 0x18),
            read_u64(header, at + 0x18 + index * 0x18),
            read_u32(header, at + 0x20 + index * 0x18),
        )
    };
    for at in [IVFC_HEADER, FAT_IVFC_HEADER] {
        if &header[at..at + 4] != b"IVFC" {
            return Err(SaveError::BadMagic("integrity"));
        }
    }
    let ivfc_hashes = [0, 1, 2]
        .map(|index| ivfc_level(IVFC_HEADER, index))
        .map(|(offset, size, order)| meta_read((offset, size)).map(|level| (level, order)));
    let (data_offset, data_size, data_order) = ivfc_level(IVFC_HEADER, 3);
    let data = core
        .get(data_offset as usize..(data_offset + data_size) as usize)
        .ok_or(SaveError::Truncated("journaled data"))?
        .to_vec();
    let [l1, l2, l3] = ivfc_hashes;
    verify_levels(
        "journal integrity",
        slice(
            header,
            word(L_IVFC_MASTER_A),
            HASH_SIZE as u64,
            "master hash",
        )?,
        &[l1?, l2?, l3?, (data.clone(), data_order)],
        &header[IVFC_HEADER + 0xA0..IVFC_HEADER + 0xC0],
    )?;
    let fat_levels = [0, 1, 2]
        .map(|index| ivfc_level(FAT_IVFC_HEADER, index))
        .map(|(offset, size, order)| meta_read((offset, size)).map(|level| (level, order)));
    let [f1, f2, fat] = fat_levels;
    let fat = fat?;
    verify_levels(
        "allocation table integrity",
        slice(
            header,
            word(L_FAT_IVFC_MASTER_A),
            HASH_SIZE as u64,
            "master hash",
        )?,
        &[f1?, f2?, fat.clone()],
        &header[FAT_IVFC_HEADER + 0xA0..FAT_IVFC_HEADER + 0xC0],
    )?;
    let fat = fat.0;

    // The file system.
    if &header[SAVE_HEADER..SAVE_HEADER + 4] != b"SAVE" {
        return Err(SaveError::BadMagic("save file system"));
    }
    let fs_block = read_u64(header, FAT_HEADER) as usize;
    let fs_blocks = read_u32(header, FAT_HEADER + 0x10) as usize;
    if fs_block == 0 || fs_blocks * fs_block > data.len() || (fs_blocks + 1) * 8 > fat.len() {
        return Err(SaveError::Corrupt("file system larger than its storage"));
    }
    let fs = FileSystem {
        fat: &fat[..(fs_blocks + 1) * 8],
        data: &data,
        block: fs_block,
    };
    let directories = fs.chain(read_u32(header, FAT_HEADER + 0x28))?;
    let files = fs.chain(read_u32(header, FAT_HEADER + 0x2C))?;
    let mut entries = Vec::new();
    fs.walk(&directories, &files, ROOT_INDEX, "", &mut entries, 0)?;

    let extra = &header[EXTRA_DATA[0]..EXTRA_DATA[0] + EXTRA_DATA_SIZE];
    Ok(SaveData {
        extra: ExtraData {
            save_data_id: read_u64(extra, 0x18),
            owner_id: read_u64(extra, 0x40),
            flags: read_u32(extra, 0x50),
            size: read_u64(extra, 0x58),
            journal_size: read_u64(extra, 0x60),
        },
        entries,
    })
}

/// Checks `levels` (hash levels from the top, data last, each with its
/// block order) against `master`.
fn verify_levels(
    layer: &'static str,
    master: &[u8],
    levels: &[(Vec<u8>, u32)],
    seed: &[u8],
) -> Result<(), SaveError> {
    let salts = level_salts(seed);
    let mut hashes = master;
    for (level, (data, order)) in levels.iter().enumerate() {
        let block_size = 1usize
            .checked_shl(*order)
            .ok_or(SaveError::Corrupt("block order"))?;
        let salt = salts
            .get(level)
            .ok_or(SaveError::Corrupt("too many hash levels"))?;
        for (block, chunk) in data.chunks(block_size).enumerate() {
            let expected = hashes
                .get(block * HASH_SIZE..(block + 1) * HASH_SIZE)
                .ok_or(SaveError::Corrupt(
                    "hash level smaller than the level below",
                ))?;
            let mut padded = chunk.to_vec();
            padded.resize(block_size, 0);
            let valid = if expected.iter().all(|&byte| byte == 0) {
                padded.iter().all(|&byte| byte == 0)
            } else {
                salted_hash(salt, &padded) == expected
            };
            if !valid {
                return Err(SaveError::Hash {
                    layer,
                    level: level + 1,
                    block,
                });
            }
        }
        hashes = data;
    }
    Ok(())
}

struct FileSystem<'a> {
    fat: &'a [u8],
    data: &'a [u8],
    block: usize,
}

impl FileSystem<'_> {
    /// The blocks of the list starting at `start`, checking the links.
    fn blocks(&self, start: u32) -> Result<Vec<usize>, SaveError> {
        let entries = self.fat.len() / 8;
        let corrupt = SaveError::Corrupt("allocation table chain");
        let mut out = Vec::new();
        let mut index = start as usize + 1;
        let mut previous = None;
        loop {
            if index == 0 || index >= entries || out.len() >= entries {
                return Err(corrupt);
            }
            let (prev, next) = (
                read_u32(self.fat, index * 8),
                read_u32(self.fat, index * 8 + 4),
            );
            let expected_prev = previous.map_or(CHAIN_FLAG, |previous: usize| previous as u32);
            if prev != expected_prev {
                return Err(corrupt);
            }
            let last = if next & CHAIN_FLAG != 0 {
                let range = read_u32(self.fat, (index + 1) * 8 + 4) as usize;
                if read_u32(self.fat, (index + 1) * 8) != CHAIN_FLAG | index as u32
                    || range < index + 1
                    || range >= entries
                {
                    return Err(corrupt);
                }
                range
            } else {
                index
            };
            out.extend(index - 1..last);
            let next = (next & !CHAIN_FLAG) as usize;
            if next == 0 {
                return Ok(out);
            }
            previous = Some(index);
            index = next;
        }
    }

    fn chain(&self, start: u32) -> Result<Vec<u8>, SaveError> {
        let mut out = Vec::new();
        for block in self.blocks(start)? {
            out.extend_from_slice(&self.data[block * self.block..(block + 1) * self.block]);
        }
        Ok(out)
    }

    fn walk(
        &self,
        directories: &[u8],
        files: &[u8],
        directory: u32,
        path: &str,
        out: &mut Vec<Entry>,
        depth: usize,
    ) -> Result<(), SaveError> {
        let entry = |table: &[u8], index: u32| -> Result<Vec<u8>, SaveError> {
            table
                .get(index as usize * TABLE_ENTRY_SIZE..(index as usize + 1) * TABLE_ENTRY_SIZE)
                .map(<[u8]>::to_vec)
                .ok_or(SaveError::Corrupt("table index out of range"))
        };
        let limit = directories.len().max(files.len()) / TABLE_ENTRY_SIZE;
        if depth > limit {
            return Err(SaveError::Corrupt("directory cycle"));
        }
        let this = entry(directories, directory)?;
        let child_path = |entry: &[u8]| -> Result<String, SaveError> {
            let name = std::str::from_utf8(name_of(entry))
                .map_err(|_| SaveError::Corrupt("entry name"))?;
            Ok(if path.is_empty() {
                name.into()
            } else {
                format!("{path}/{name}")
            })
        };
        let mut file = read_u32(&this, 0x4C);
        for _ in 0..=limit {
            if file == 0 {
                break;
            }
            let entry = entry(files, file)?;
            if read_u32(&entry, 0) != directory {
                return Err(SaveError::Corrupt("file listed under another directory"));
            }
            let start = read_u32(&entry, 0x48);
            let length = read_u64(&entry, 0x4C) as usize;
            let data = if start == EMPTY_FILE_START {
                Vec::new()
            } else {
                let mut data = self.chain(start)?;
                if data.len() < length {
                    return Err(SaveError::Corrupt("file longer than its blocks"));
                }
                data.truncate(length);
                data
            };
            out.push(Entry::File(child_path(&entry)?, data));
            file = read_u32(&entry, 0x44);
        }
        let mut child = read_u32(&this, 0x48);
        for _ in 0..=limit {
            if child == 0 {
                return Ok(());
            }
            let entry = entry(directories, child)?;
            if read_u32(&entry, 0) != directory {
                return Err(SaveError::Corrupt("directory listed under another one"));
            }
            let child_path = child_path(&entry)?;
            out.push(Entry::Directory(child_path.clone()));
            self.walk(directories, files, child, &child_path, out, depth + 1)?;
            child = read_u32(&entry, 0x44);
        }
        Err(SaveError::Corrupt("sibling cycle"))
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
}
