//! The read-only filesystem (RomFS) of the data NCAs, as far as finding a
//! file by its path takes.
//!
//! A RomFS starts with a header of ten little-endian 64-bit words: its own
//! size (0x50), then offset and size of the directory hash table, of the
//! directory entries, of the file hash table and of the file entries, and
//! last the offset of the file data. A directory entry is six 32-bit words
//! (parent, next sibling, first child directory, first file, next in its
//! hash bucket, name length) and the name; a file entry is parent, next
//! sibling, then the 64-bit offset (from the file data) and size, the next
//! in its hash bucket, the name length and the name. Entries name each
//! other by their offset in their table, `0xFFFF_FFFF` for none, and the
//! root directory is the one at offset 0 (switchbrew "RomFS"). The lookup
//! here walks the sibling chains and leaves the hash tables alone.

use std::fmt;

const HEADER_SIZE: usize = 0x50;
const NONE: u32 = u32::MAX;
const DIRECTORY_ENTRY: usize = 0x18;
const FILE_ENTRY: usize = 0x20;

#[derive(Debug, PartialEq, Eq)]
pub enum RomFsError {
    /// The header or an entry points past the image: what.
    Malformed(&'static str),
    /// No file at that path.
    NotFound(String),
}

impl fmt::Display for RomFsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Malformed(what) => write!(f, "malformed RomFS: {what}"),
            Self::NotFound(path) => write!(f, "no file {path} in the RomFS"),
        }
    }
}

impl std::error::Error for RomFsError {}

fn u32_at(bytes: &[u8], offset: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        bytes.get(offset..offset + 4)?.try_into().unwrap(),
    ))
}

fn u64_at(bytes: &[u8], offset: usize) -> Option<u64> {
    Some(u64::from_le_bytes(
        bytes.get(offset..offset + 8)?.try_into().unwrap(),
    ))
}

/// The table at header word `word` (offset, then size).
fn table(image: &[u8], word: usize) -> Result<&[u8], RomFsError> {
    let offset = u64_at(image, word * 8).ok_or(RomFsError::Malformed("short header"))? as usize;
    let size = u64_at(image, word * 8 + 8).ok_or(RomFsError::Malformed("short header"))? as usize;
    image
        .get(offset..offset.saturating_add(size))
        .ok_or(RomFsError::Malformed("a table past the end"))
}

/// The name of the entry at `at` of `entries`, whose fixed part is `fixed`
/// bytes with the name length in its last word.
fn name(entries: &[u8], at: usize, fixed: usize) -> Result<&[u8], RomFsError> {
    let length =
        u32_at(entries, at + fixed - 4).ok_or(RomFsError::Malformed("an entry past its table"))?;
    entries
        .get(at + fixed..at + fixed + length as usize)
        .ok_or(RomFsError::Malformed("a name past its table"))
}

/// The bytes of the file at `path` (components separated by `/`, no
/// leading one) in the RomFS `image`.
pub fn file<'a>(image: &'a [u8], path: &str) -> Result<&'a [u8], RomFsError> {
    if u64_at(image, 0) != Some(HEADER_SIZE as u64) {
        return Err(RomFsError::Malformed("no 0x50-byte header"));
    }
    let directories = table(image, 3)?;
    let files = table(image, 7)?;
    let data = u64_at(image, 9 * 8).ok_or(RomFsError::Malformed("short header"))? as usize;
    let not_found = || RomFsError::NotFound(path.into());
    let word = |entries: &[u8], at: usize| {
        u32_at(entries, at).ok_or(RomFsError::Malformed("an entry past its table"))
    };

    let (parents, leaf) = path.rsplit_once('/').unwrap_or(("", path));
    let mut directory = 0;
    for component in parents.split('/').filter(|component| !component.is_empty()) {
        let mut child = word(directories, directory + 8)?;
        loop {
            if child == NONE {
                return Err(not_found());
            }
            if name(directories, child as usize, DIRECTORY_ENTRY)? == component.as_bytes() {
                break;
            }
            child = word(directories, child as usize + 4)?;
        }
        directory = child as usize;
    }
    let mut entry = word(directories, directory + 12)?;
    while entry != NONE {
        let at = entry as usize;
        if name(files, at, FILE_ENTRY)? == leaf.as_bytes() {
            let past = || RomFsError::Malformed("a file entry past its table");
            let offset = u64_at(files, at + 8).ok_or_else(past)? as usize;
            let size = u64_at(files, at + 16).ok_or_else(past)? as usize;
            let start = data.saturating_add(offset);
            return image
                .get(start..start.saturating_add(size))
                .ok_or(RomFsError::Malformed("file data past the end"));
        }
        entry = word(files, at + 4)?;
    }
    Err(not_found())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A RomFS with `/top.bin` and `/nx/bct`, laid out as the tables say.
    fn image() -> Vec<u8> {
        let entry = |words: &[u32], name: &str| {
            let mut bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
            bytes.extend((name.len() as u32).to_le_bytes());
            bytes.extend(name.as_bytes());
            bytes.resize(bytes.len().next_multiple_of(4), 0);
            bytes
        };
        // Root at 0, `nx` at 0x18.
        let mut directories = entry(&[0, NONE, 0x18, 0, NONE], "");
        directories.extend(entry(&[0, NONE, NONE, 0x28, NONE], "nx"));
        // `top.bin` at 0, `bct` at 0x28.
        let file = |parent: u32, offset: u64, size: u64, name: &str| {
            let mut bytes = parent.to_le_bytes().to_vec();
            bytes.extend(NONE.to_le_bytes());
            bytes.extend(offset.to_le_bytes());
            bytes.extend(size.to_le_bytes());
            bytes.extend(NONE.to_le_bytes());
            bytes.extend((name.len() as u32).to_le_bytes());
            bytes.extend(name.as_bytes());
            bytes.resize(bytes.len().next_multiple_of(4), 0);
            bytes
        };
        let mut files = file(0, 0, 3, "top.bin");
        assert_eq!(files.len(), 0x28);
        files.extend(file(0x18, 3, 5, "bct"));
        let directories_at = HEADER_SIZE;
        let files_at = directories_at + directories.len();
        let data_at = files_at + files.len();
        let mut image = Vec::new();
        for word in [
            HEADER_SIZE,
            directories_at,
            0,
            directories_at,
            directories.len(),
            files_at,
            0,
            files_at,
            files.len(),
            data_at,
        ] {
            image.extend((word as u64).to_le_bytes());
        }
        image.extend(directories);
        image.extend(files);
        image.extend(b"topBCT!!");
        image
    }

    #[test]
    fn files_are_found_by_path() {
        let image = image();
        assert_eq!(file(&image, "top.bin"), Ok(&b"top"[..]));
        assert_eq!(file(&image, "nx/bct"), Ok(&b"BCT!!"[..]));
        assert_eq!(
            file(&image, "a/bct"),
            Err(RomFsError::NotFound("a/bct".into()))
        );
        assert_eq!(
            file(&image, "nx/package1"),
            Err(RomFsError::NotFound("nx/package1".into()))
        );
        assert!(matches!(
            file(&image[..0x40], "nx/bct"),
            Err(RomFsError::Malformed(_))
        ));
    }
}
