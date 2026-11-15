//! The key replacement of [`super`]: one public key each in the filesystem
//! process and the loader, changed where the kernel image carries them, in
//! its INI1, before the console starts.
//!
//! An INI1 is a header (`INI1`, size, process count) and the processes'
//! KIP1 images one after the other. A KIP1 has a 0x100-byte header (name
//! at `0x04`, flags at `0x1f`, bit N set when segment N is compressed, and
//! from `0x20` per segment its address, size and stored size) followed by
//! its text, read-only data and data segments as stored. A compressed
//! segment is in the backwards LZ format the kernel unpacks in place: a
//! footer of three words (stored size of the compressed part, footer
//! size, growth) ends it, and the unpacker walks from the end to the
//! front, each control byte announcing eight items, either a literal byte
//! or a back-reference (length and distance) into what it has already
//! written. A key that comes out of literals only, with nothing referring
//! back into it, is changed by changing those literals: the segment keeps
//! its size, and nothing else of the image moves. [`replace_key`] checks
//! both and refuses otherwise.
//!
//! The two modules are those of firmware 23.5.0, recognized by the GNU
//! build ID in their read-only data and by the SHA-256 of the key at its
//! place; anything else is refused.

use std::fmt;

use sha2::{Digest, Sha256};

use super::SIGNATURE_SIZE;

/// A public key in a module's read-only data that [`install_homebrew_keys`]
/// replaces.
struct PinnedKey {
    module: &'static str,
    build_id: [u8; 20],
    /// Offset of the key in the read-only segment.
    offset: usize,
    /// Its address in the module.
    address: u64,
    /// SHA-256 of the key that is to be there.
    expected: [u8; 32],
    what: &'static str,
}

const fn hex_digit(character: u8) -> u8 {
    match character {
        b'0'..=b'9' => character - b'0',
        b'a'..=b'f' => character - b'a' + 10,
        _ => panic!("not a hex digit"),
    }
}

const fn hex<const N: usize>(text: &str) -> [u8; N] {
    let text = text.as_bytes();
    let mut bytes = [0; N];
    let mut index = 0;
    while index < N {
        bytes[index] = hex_digit(text[2 * index]) << 4 | hex_digit(text[2 * index + 1]);
        index += 1;
    }
    bytes
}

const PINNED_KEYS: [PinnedKey; 2] = [
    PinnedKey {
        module: "FS",
        build_id: hex("d74281247f248db5b9a700e7c72f19d4d71e022b"),
        offset: 0x4a07,
        address: 0x1e3a07,
        expected: hex("5f6be31c316e7cb21ca7b9a1706a9d5804eb905372efcb56d193f2af9e8ad1fa"),
        what: "retail NCA header key, signature key generation 1",
    },
    PinnedKey {
        module: "Loader",
        build_id: hex("5f98fb58e2f419b903b94b7c68481e71c469b7f6"),
        offset: 0xa04,
        address: 0x34a04,
        expected: hex("a83598cf449c7f8720e3b8f9cd12da396b5ddf40f8647a94e2bb4fd92c1b699e"),
        what: "retail ACID key, signature key generation 1",
    },
];

#[derive(Debug, PartialEq, Eq)]
pub enum KeyError {
    /// The INI1 or one of its KIPs is not laid out as described.
    Malformed(&'static str),
    NoModule(&'static str),
    /// The module is there, but not the build this knows.
    UnknownBuild {
        module: &'static str,
    },
    /// The build is known, but the key at the place is not the expected one.
    UnexpectedKey {
        module: &'static str,
    },
    /// The key does not come out of literals alone, or something refers to
    /// it.
    NotLiteral {
        module: &'static str,
    },
}

impl fmt::Display for KeyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Malformed(what) => write!(f, "malformed INI1: {what}"),
            Self::NoModule(module) => write!(f, "the INI1 has no {module}"),
            Self::UnknownBuild { module } => {
                write!(f, "{module} is not the firmware 23.5.0 build")
            }
            Self::UnexpectedKey { module } => {
                write!(f, "{module} does not hold the expected key at its place")
            }
            Self::NotLiteral { module } => write!(
                f,
                "{module}: the key cannot be changed in its compressed segment"
            ),
        }
    }
}

impl std::error::Error for KeyError {}

fn u32_at(bytes: &[u8], offset: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        bytes.get(offset..offset + 4)?.try_into().ok()?,
    ))
}

/// One KIP of an INI1: its name and where its read-only segment is
/// stored.
struct Kip {
    name: String,
    compressed: bool,
    /// Bytes of the INI1 that hold the segment.
    stored: std::ops::Range<usize>,
    size: usize,
}

fn kips(ini1: &[u8]) -> Result<Vec<Kip>, KeyError> {
    const HEADER: usize = 0x100;
    let malformed = KeyError::Malformed;
    if ini1.get(..4) != Some(b"INI1") {
        return Err(malformed("no INI1 magic"));
    }
    let count = u32_at(ini1, 8).ok_or(malformed("short header"))?;
    let mut at = 0x10;
    let mut kips = Vec::new();
    for _ in 0..count {
        let header = ini1
            .get(at..at + HEADER)
            .ok_or(malformed("a KIP header is past the end"))?;
        if &header[..4] != b"KIP1" {
            return Err(malformed("no KIP1 magic"));
        }
        let name = header[4..0x10].split(|&byte| byte == 0).next().unwrap();
        let segment = |index: usize| {
            let field = |offset| u32_at(header, 0x20 + 0x10 * index + offset).unwrap() as usize;
            (field(4), field(8))
        };
        let (text_stored, (ro_size, ro_stored), data_stored) =
            (segment(0).1, segment(1), segment(2).1);
        let ro_start = at + HEADER + text_stored;
        kips.push(Kip {
            name: String::from_utf8_lossy(name).into_owned(),
            compressed: header[0x1f] & 2 != 0,
            stored: ro_start..ro_start + ro_stored,
            size: ro_size,
        });
        at = ro_start + ro_stored + data_stored;
        if at > ini1.len() {
            return Err(malformed("a KIP is past the end"));
        }
    }
    Ok(kips)
}

/// What unpacking a segment made, and where each byte came from.
struct Unpacked {
    bytes: Vec<u8>,
    /// For each byte, the byte of the stored segment it is a copy of, if
    /// it is a literal (or in the part stored as it is).
    origin: Vec<Option<usize>>,
    /// Bytes a back-reference copied from.
    referenced: Vec<bool>,
}

/// Unpacks a compressed segment the way the kernel does.
fn unpack(stored: &[u8], size: usize) -> Result<Unpacked, KeyError> {
    let malformed = KeyError::Malformed;
    let end = stored.len();
    if end < 12 || size < end {
        return Err(malformed("a compressed segment is too short"));
    }
    let compressed = u32_at(stored, end - 12).unwrap() as usize;
    let footer = u32_at(stored, end - 8).unwrap() as usize;
    let growth = u32_at(stored, end - 4).unwrap() as usize;
    if compressed > end || footer > compressed || end + growth > size {
        return Err(malformed("an impossible compressed segment footer"));
    }
    let mut bytes = stored.to_vec();
    bytes.resize(size, 0);
    let mut origin: Vec<Option<usize>> = (0..end)
        .map(Some)
        .chain((end..size).map(|_| None))
        .collect();
    let mut referenced = vec![false; size];
    let start = end - compressed;
    let (mut input, mut output) = (compressed - footer, compressed + growth);
    while output > 0 {
        input = input
            .checked_sub(1)
            .ok_or(malformed("compressed data ran out"))?;
        let mut control = bytes[start + input];
        for _ in 0..8 {
            if control & 0x80 != 0 {
                input = input
                    .checked_sub(2)
                    .ok_or(malformed("compressed data ran out"))?;
                let value = u16::from_le_bytes([bytes[start + input], bytes[start + input + 1]]);
                let length = (usize::from(value >> 12) + 3).min(output);
                let distance = usize::from(value & 0xFFF) + 3;
                output -= length;
                for index in start + output..start + output + length {
                    let from = index + distance;
                    if from >= size {
                        return Err(malformed("a back-reference past the end"));
                    }
                    bytes[index] = bytes[from];
                    origin[index] = None;
                    referenced[from] = true;
                }
            } else {
                input = input
                    .checked_sub(1)
                    .ok_or(malformed("compressed data ran out"))?;
                output -= 1;
                bytes[start + output] = bytes[start + input];
                origin[start + output] = Some(start + input);
            }
            control <<= 1;
            if output == 0 {
                break;
            }
        }
    }
    Ok(Unpacked {
        bytes,
        origin,
        referenced,
    })
}

/// The GNU build ID note of a module's read-only data.
fn build_id(ro: &[u8]) -> Option<&[u8]> {
    const NOTE: [u8; 16] = *b"\x04\0\0\0\x14\0\0\0\x03\0\0\0GNU\0";
    let at = ro.windows(NOTE.len()).position(|window| window == NOTE)?;
    ro.get(at + NOTE.len()..at + NOTE.len() + 20)
}

/// A stored segment, unpacked if it is compressed.
fn unpack_segment(stored: &[u8], compressed: bool, size: usize) -> Result<Unpacked, KeyError> {
    if compressed {
        unpack(stored, size)
    } else {
        Ok(Unpacked {
            bytes: stored.to_vec(),
            origin: (0..stored.len()).map(Some).collect(),
            referenced: vec![false; stored.len()],
        })
    }
}

/// Changes the bytes `place` of what `stored` unpacks to into `new`, by
/// changing the literals they come from, if they all do and nothing
/// refers back to them. Whether the result unpacks to exactly that is
/// checked.
fn replace_literals(
    stored: &mut [u8],
    compressed: bool,
    size: usize,
    place: std::ops::Range<usize>,
    new: &[u8],
) -> Result<bool, KeyError> {
    let unpacked = unpack_segment(stored, compressed, size)?;
    if place.end > unpacked.bytes.len() {
        return Ok(false);
    }
    let mut changed = stored.to_vec();
    for (index, &byte) in place.clone().zip(new) {
        match unpacked.origin[index] {
            Some(source) if !unpacked.referenced[index] => changed[source] = byte,
            _ => return Ok(false),
        }
    }
    let mut expected = unpacked.bytes;
    expected[place].copy_from_slice(new);
    if unpack_segment(&changed, compressed, size)?.bytes != expected {
        return Ok(false);
    }
    stored.copy_from_slice(&changed);
    Ok(true)
}

/// Puts `key` in place of the key `pinned` names, in the stored read-only
/// segment `stored` of its module.
fn replace_key(
    stored: &mut [u8],
    kip: &Kip,
    pinned: &PinnedKey,
    key: &[u8; SIGNATURE_SIZE],
) -> Result<(), KeyError> {
    let module = pinned.module;
    let ro = unpack_segment(stored, kip.compressed, kip.size)?.bytes;
    if build_id(&ro) != Some(&pinned.build_id[..]) {
        return Err(KeyError::UnknownBuild { module });
    }
    let place = pinned.offset..pinned.offset + SIGNATURE_SIZE;
    let found = ro
        .get(place.clone())
        .ok_or(KeyError::UnexpectedKey { module })?;
    if <[u8; 32]>::from(Sha256::digest(found)) != pinned.expected {
        return Err(KeyError::UnexpectedKey { module });
    }
    if !replace_literals(stored, kip.compressed, kip.size, place, key)? {
        return Err(KeyError::NotLiteral { module });
    }
    Ok(())
}

/// Replaces, in `ini1`, the filesystem process's NCA header key and the
/// loader's ACID key of signature key generation 1 with `key`. Returns a
/// line per key saying what changed where.
pub fn install_homebrew_keys(
    ini1: &mut [u8],
    key: &[u8; SIGNATURE_SIZE],
) -> Result<Vec<String>, KeyError> {
    let kips = kips(ini1)?;
    let mut report = Vec::new();
    for pinned in &PINNED_KEYS {
        let kip = kips
            .iter()
            .find(|kip| kip.name == pinned.module)
            .ok_or(KeyError::NoModule(pinned.module))?;
        replace_key(&mut ini1[kip.stored.clone()], kip, pinned, key)?;
        report.push(format!(
            "{} (build {}): {} at {:#x} (.rodata+{:#x}) replaced by the homebrew test key",
            pinned.module,
            pinned
                .build_id
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>(),
            pinned.what,
            pinned.address,
            pinned.offset,
        ));
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `PFX!` stored as it is, then a compressed part that unpacks to the
    /// 18 letters `a` to `r` twice: from the back, eight literals (`r` to
    /// `k`), eight more (`j` to `c`), then `b`, `a` and a back-reference of
    /// 18 bytes, 18 bytes on, each group of eight after its control byte.
    fn segment() -> (Vec<u8>, usize) {
        let mut stored = b"PFX!".to_vec();
        stored.extend_from_slice(&[0x0F, 0xF0, b'a', b'b', 0x20]);
        stored.extend_from_slice(b"cdefghij");
        stored.push(0);
        stored.extend_from_slice(b"klmnopqr");
        stored.push(0);
        // 23 bytes and the footer are compressed, unpacking to 36.
        for word in [35u32, 12, 1] {
            stored.extend_from_slice(&word.to_le_bytes());
        }
        (stored, 40)
    }

    #[test]
    fn unpacking_traces_literals_and_references() {
        let (stored, size) = segment();
        let unpacked = unpack(&stored, size).unwrap();
        assert_eq!(unpacked.bytes, b"PFX!abcdefghijklmnopqrabcdefghijklmnopqr");
        assert_eq!(unpacked.origin[..4], [Some(0), Some(1), Some(2), Some(3)]);
        assert!(unpacked.origin[4..22].iter().all(Option::is_none));
        assert_eq!(unpacked.origin[22], Some(6), "a");
        assert_eq!(unpacked.origin[39], Some(25), "r");
        assert!(
            unpacked.referenced[22..]
                .iter()
                .all(|&referenced| referenced)
        );
        assert!(
            !unpacked.referenced[..22]
                .iter()
                .any(|&referenced| referenced)
        );
    }

    #[test]
    fn only_unreferenced_literals_are_replaced() {
        let (mut stored, size) = segment();
        let original = stored.clone();
        // Copied by the back-reference: changing them would change more.
        assert!(!replace_literals(&mut stored, true, size, 22..24, b"XY").unwrap());
        // Made by the back-reference: no literal to change.
        assert!(!replace_literals(&mut stored, true, size, 4..6, b"XY").unwrap());
        assert_eq!(stored, original);
        assert!(replace_literals(&mut stored, true, size, 1..3, b"XY").unwrap());
        assert_eq!(
            unpack(&stored, size).unwrap().bytes,
            b"PXY!abcdefghijklmnopqrabcdefghijklmnopqr"
        );
    }
}
