//! A program content archive (NCA) made from an ExeFS, signed with the
//! homebrew test key, for [`crate::nand::replace_program`].
//!
//! The ExeFS comes as a partition FS (`PFS0`, the `.nsp` homebrew is
//! shipped as) holding `main` and `main.npdm`, and possibly `rtld`, `sdk`
//! and `subsdk0` to `subsdk9`: the code files the loader looks for
//! (`code:/rtld`, `code:/main`, `code:/subsdk%d`, `code:/sdk`, loader
//! `0x5350`) and the program's metadata (`code:/main.npdm`).
//!
//! The NPDM is re-signed ([`sign_npdm`]): its signature key generation
//! (byte 4) becomes [`KEY_GENERATION`], its ACID carries the test key's
//! modulus and a signature by it, and before that its pool partition
//! becomes that of the program it replaces ([`build_program_nca`]).
//! Layout (public, switchbrew "NPDM"): the
//! `META` header holds the ACI and ACID offsets and sizes at `0x70`; the
//! ACID has its signature at `0x000`, the NCA signature 2 key at `0x100`,
//! `ACID` at `0x200`, its size less 0x100 at `0x204`; the ACI has `ACI0`
//! and the program ID at `0x10`.
//!
//! The archive ([`build_program_nca`]) is NCA3, the format of the
//! firmware's own: a 0xC00-byte header encrypted with the header key, and
//! one section, the ExeFS, at `0xC00`. The section is a partition FS under
//! hierarchical SHA-256 (block size [`HASH_BLOCK_SIZE`]: the hash table,
//! then the PFS0 right after it, as in the firmware's program archives),
//! encrypted with AES-CTR under a key of its own sealed in the header's key
//! area. The header takes over from the archive it replaces what tells the
//! key ladder and the filesystem process what to expect: distribution and
//! content type, key generation (both fields) and key area kek index, and
//! the SDK addon version (the filesystem process refuses one of 10.0 or
//! below, `0x171670`). Its signature key generation is [`KEY_GENERATION`],
//! and both header signatures are by the test key.

use std::fmt;

use super::{KEY_GENERATION, SIGNATURE_SIZE, modulus, sign};
use crate::{
    keys::KeySet,
    nand::{
        bis::Xts,
        crypto,
        nca::{self, Nca},
    },
};

/// Block size of the ExeFS section's hash table.
pub const HASH_BLOCK_SIZE: usize = 0x4000;

const SECTION_START: usize = nca::HEADER_SIZE;

/// Names of the files an ExeFS may hold.
const CODE_FILES: [&str; 13] = [
    "rtld", "main", "subsdk0", "subsdk1", "subsdk2", "subsdk3", "subsdk4", "subsdk5", "subsdk6",
    "subsdk7", "subsdk8", "subsdk9", "sdk",
];
const NPDM: &str = "main.npdm";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProgramError {
    /// Not a partition FS, or one that holds something else.
    ExeFs(String),
    /// The NPDM is not laid out as described.
    Npdm(&'static str),
    /// The program ID of the NPDM is not the one asked for.
    ProgramId { npdm: u64, expected: u64 },
    /// A key the header or key area needs.
    Key(String),
}

impl fmt::Display for ProgramError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExeFs(why) => write!(f, "not an ExeFS: {why}"),
            Self::Npdm(why) => write!(f, "malformed main.npdm: {why}"),
            Self::ProgramId { npdm, expected } => write!(
                f,
                "main.npdm is for program {npdm:016x}, not {expected:016x}"
            ),
            Self::Key(why) => write!(f, "{why}"),
        }
    }
}

impl std::error::Error for ProgramError {}

fn u32_at(bytes: &[u8], offset: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        bytes.get(offset..offset + 4)?.try_into().ok()?,
    ))
}

fn u64_at(bytes: &[u8], offset: usize) -> Option<u64> {
    Some(u64::from_le_bytes(
        bytes.get(offset..offset + 8)?.try_into().ok()?,
    ))
}

/// The files of the ExeFS `nsp`, in the order the loader reads them, the
/// NPDM last.
pub fn exefs_files(nsp: &[u8]) -> Result<Vec<(String, Vec<u8>)>, ProgramError> {
    let files = nca::pfs0_files(nsp).map_err(|error| ProgramError::ExeFs(error.to_string()))?;
    if let Some((name, _)) = files
        .iter()
        .find(|(name, _)| *name != NPDM && !CODE_FILES.contains(name))
    {
        return Err(ProgramError::ExeFs(format!("it holds {name}")));
    }
    let mut ordered = Vec::new();
    for wanted in CODE_FILES.iter().chain([&NPDM]) {
        match files.iter().filter(|(name, _)| name == wanted).count() {
            0 => {}
            1 => {
                let (name, data) = files.iter().find(|(name, _)| name == wanted).unwrap();
                ordered.push((name.to_string(), data.to_vec()));
            }
            _ => return Err(ProgramError::ExeFs(format!("{wanted} twice"))),
        }
    }
    for needed in ["main", NPDM] {
        if !ordered.iter().any(|(name, _)| name == needed) {
            return Err(ProgramError::ExeFs(format!("no {needed}")));
        }
    }
    Ok(ordered)
}

/// The ACID of `npdm`: where it is and how long.
fn acid(npdm: &[u8]) -> Result<std::ops::Range<usize>, ProgramError> {
    if npdm.get(..4) != Some(b"META") {
        return Err(ProgramError::Npdm("no META magic"));
    }
    let offset = u32_at(npdm, 0x78).ok_or(ProgramError::Npdm("short META"))? as usize;
    let size = u32_at(npdm, 0x7C).ok_or(ProgramError::Npdm("short META"))? as usize;
    let acid = npdm
        .get(offset..offset + size)
        .ok_or(ProgramError::Npdm("the ACID is past the end"))?;
    if acid.get(0x200..0x204) != Some(b"ACID") {
        return Err(ProgramError::Npdm("no ACID magic"));
    }
    if u32_at(acid, 0x204).map(|signed| signed as usize + SIGNATURE_SIZE) != Some(size) {
        return Err(ProgramError::Npdm("the ACID size field disagrees"));
    }
    Ok(offset..offset + size)
}

/// The program ID the ACI of `npdm` gives.
pub fn npdm_program_id(npdm: &[u8]) -> Result<u64, ProgramError> {
    if npdm.get(..4) != Some(b"META") {
        return Err(ProgramError::Npdm("no META magic"));
    }
    let offset = u32_at(npdm, 0x70).ok_or(ProgramError::Npdm("short META"))? as usize;
    let aci = npdm
        .get(offset..)
        .ok_or(ProgramError::Npdm("the ACI is past the end"))?;
    if aci.get(..4) != Some(b"ACI0") {
        return Err(ProgramError::Npdm("no ACI0 magic"));
    }
    u64_at(aci, 0x10).ok_or(ProgramError::Npdm("short ACI"))
}

/// Signs `npdm` with the test key: see the module documentation.
pub fn sign_npdm(npdm: &mut [u8]) -> Result<(), ProgramError> {
    let range = acid(npdm)?;
    npdm[4] = KEY_GENERATION;
    let acid = &mut npdm[range];
    acid[SIGNATURE_SIZE..2 * SIGNATURE_SIZE].copy_from_slice(&modulus());
    let signature = sign(&acid[SIGNATURE_SIZE..]);
    acid[..SIGNATURE_SIZE].copy_from_slice(&signature);
    Ok(())
}

/// A partition FS of `files`, each name in the string table padded the way
/// Nintendo's tools do so that the data starts 0x20-aligned.
pub fn pfs0(files: &[(String, Vec<u8>)]) -> Vec<u8> {
    let mut names = Vec::new();
    let mut name_offsets = Vec::new();
    for (name, _) in files {
        name_offsets.push(names.len() as u32);
        names.extend_from_slice(name.as_bytes());
        names.push(0);
    }
    let header_size = 0x10 + 0x18 * files.len();
    names.resize(
        (header_size + names.len()).next_multiple_of(0x20) - header_size,
        0,
    );
    let mut image = Vec::new();
    image.extend_from_slice(b"PFS0");
    image.extend_from_slice(&(files.len() as u32).to_le_bytes());
    image.extend_from_slice(&(names.len() as u32).to_le_bytes());
    image.extend_from_slice(&[0; 4]);
    let mut offset = 0u64;
    for ((_, data), name_offset) in files.iter().zip(name_offsets) {
        image.extend_from_slice(&offset.to_le_bytes());
        image.extend_from_slice(&(data.len() as u64).to_le_bytes());
        image.extend_from_slice(&name_offset.to_le_bytes());
        image.extend_from_slice(&[0; 4]);
        offset += data.len() as u64;
    }
    image.extend_from_slice(&names);
    for (_, data) in files {
        image.extend_from_slice(data);
    }
    image
}

/// The pool partition bits of an ACID's flags (`0x20c`).
const POOL_PARTITION: u32 = 0x3C;

/// The pool partition the ACID of the program in `original` gives, as its
/// flag bits: from the NPDM of its ExeFS (section 0).
fn pool_partition(original: &Nca) -> Result<u32, ProgramError> {
    let exefs = original
        .partition_section(0)
        .map_err(|error| ProgramError::ExeFs(format!("the replaced program's: {error}")))?;
    let files = nca::pfs0_files(&exefs)
        .map_err(|error| ProgramError::ExeFs(format!("the replaced program's: {error}")))?;
    let (_, npdm) = files
        .iter()
        .find(|(name, _)| *name == NPDM)
        .ok_or_else(|| ProgramError::ExeFs("the replaced program's has no main.npdm".into()))?;
    let acid = acid(npdm)?;
    Ok(u32_at(npdm, acid.start + 0x20C).expect("the ACID is long enough") & POOL_PARTITION)
}

/// Region type of a map-region kernel capability (descriptor ending in
/// ten ones, three seven-bit fields from bit 11: a six-bit type and a
/// read-only bit) for the kernel trace buffer.
const REGION_KERNEL_TRACE_BUFFER: u32 = 1;
/// A kernel capability word that stands for nothing (the loader skips it,
/// `0x5540`; so does the kernel).
const PADDING: u32 = 0xFFFF_FFFF;

/// The kernel capabilities of `npdm`'s ACI and of its ACID, as byte
/// ranges.
fn capability_ranges(npdm: &[u8]) -> Result<[std::ops::Range<usize>; 2], ProgramError> {
    let short = ProgramError::Npdm("a capability table is past the end");
    let aci = u32_at(npdm, 0x70).ok_or(ProgramError::Npdm("short META"))? as usize;
    let acid = acid(npdm)?.start;
    let mut ranges = Vec::new();
    for (start, field) in [(aci, 0x30), (acid, 0x230)] {
        let offset = u32_at(npdm, start + field).ok_or(short.clone())? as usize;
        let size = u32_at(npdm, start + field + 4).ok_or(short.clone())? as usize;
        let range = start + offset..start + offset + size;
        if range.end > npdm.len() || !size.is_multiple_of(4) {
            return Err(short);
        }
        ranges.push(range);
    }
    Ok(ranges.try_into().expect("two"))
}

/// Takes out of `npdm` (ACI and ACID alike) every request to map the
/// kernel trace buffer, which this console's kernel does not have: it
/// refuses to create a process that asks for a region it lacks (result
/// `0xee01`, `KCapabilities_ValidateMapRegion` at `0x8006e070`, called
/// for `svcCreateProcess` from `0x8006e38c`). Retail kernels have no trace
/// buffer; hbloader's NPDM asks for it read-only (`0x20bff`). This is a
/// rule about the homebrew's own metadata, the one Atmosphère's loader
/// applied to hbloader's capabilities before its own kernel learned to
/// accept the request. A descriptor left with no region becomes padding.
/// Returns a line per descriptor changed.
fn drop_absent_regions(npdm: &mut [u8]) -> Result<Vec<String>, ProgramError> {
    let mut dropped = Vec::new();
    for (table, range) in ["ACI", "ACID"].into_iter().zip(capability_ranges(npdm)?) {
        for at in range.step_by(4) {
            let word = u32_at(npdm, at).expect("within the table");
            if word & 0x7FF != 0x3FF {
                continue;
            }
            let mut kept = word & 0x7FF;
            for shift in [11, 18, 25] {
                let field = (word >> shift) & 0x7F;
                if field & 0x3F != REGION_KERNEL_TRACE_BUFFER {
                    kept |= field << shift;
                }
            }
            if kept == word {
                continue;
            }
            let new = if kept == 0x3FF { PADDING } else { kept };
            npdm[at..at + 4].copy_from_slice(&new.to_le_bytes());
            dropped.push(format!(
                "{table} capability {word:#x} (map the kernel trace buffer) became {new:#x}: \
                 this kernel has no trace buffer"
            ));
        }
    }
    Ok(dropped)
}

/// The program archive for the ExeFS `nsp`, which must be for
/// `program_id`, in place of `original`, and a line for each change
/// [`drop_absent_regions`] made; see the module documentation.
///
/// The ACID's pool partition becomes that of `original`: the system
/// starts the program as the one it replaces, and the loader refuses an
/// applet (application type 2 in its kernel capabilities) whose ACID puts
/// it in the application pool (result `0x809`, `0x5b90`), as hbloader's
/// does when it stands in for an applet.
pub fn build_program_nca(
    keys: &KeySet,
    original: &Nca,
    program_id: u64,
    nsp: &[u8],
) -> Result<(Vec<u8>, Vec<String>), ProgramError> {
    let pool = pool_partition(original)?;
    let mut files = exefs_files(nsp)?;
    let npdm = &mut files.last_mut().expect("main.npdm is last").1;
    let npdm_id = npdm_program_id(npdm)?;
    if npdm_id != program_id {
        return Err(ProgramError::ProgramId {
            npdm: npdm_id,
            expected: program_id,
        });
    }
    let flags_at = acid(npdm)?.start + 0x20C;
    let flags = u32_at(npdm, flags_at).expect("the ACID is long enough") & !POOL_PARTITION | pool;
    npdm[flags_at..flags_at + 4].copy_from_slice(&flags.to_le_bytes());
    let dropped = drop_absent_regions(npdm)?;
    sign_npdm(npdm)?;
    let archive = seal(keys, original.header(), program_id, &pfs0(&files))?;
    Ok((archive, dropped))
}

/// The archive of the ExeFS `exefs`, with what its header takes over from
/// another archive's from `template` (decrypted).
fn seal(
    keys: &KeySet,
    template: &[u8; nca::HEADER_SIZE],
    program_id: u64,
    exefs: &[u8],
) -> Result<Vec<u8>, ProgramError> {
    let mut header = vec![0u8; nca::HEADER_SIZE];
    header[0x200..0x204].copy_from_slice(b"NCA3");
    // Distribution type, content type, old key generation, kek index.
    header[0x204..0x208].copy_from_slice(&template[0x204..0x208]);
    header[0x205] = nca::CONTENT_TYPE_PROGRAM;
    header[0x210..0x218].copy_from_slice(&program_id.to_le_bytes());
    // SDK addon version, key generation.
    header[0x21C..0x221].copy_from_slice(&template[0x21C..0x221]);
    header[0x221] = KEY_GENERATION;

    // The section: hash table, then the PFS0, encrypted to its end.
    let hashes: Vec<u8> = exefs
        .chunks(HASH_BLOCK_SIZE)
        .flat_map(crypto::sha256)
        .collect();
    let mut section = hashes.clone();
    section.extend_from_slice(exefs);
    section.resize(section.len().next_multiple_of(nca::MEDIA_UNIT as usize), 0);
    let section_end = SECTION_START + section.len();

    let mut fs_header = [0u8; nca::SECTION_HEADER_SIZE];
    fs_header[0..2].copy_from_slice(&2u16.to_le_bytes());
    fs_header[0x02] = nca::FS_TYPE_PARTITION;
    fs_header[0x03] = nca::HASH_TYPE_HIERARCHICAL_SHA256;
    fs_header[0x04] = nca::ENCRYPTION_CTR;
    fs_header[0x08..0x28].copy_from_slice(&crypto::sha256(&hashes));
    fs_header[0x28..0x2C].copy_from_slice(&(HASH_BLOCK_SIZE as u32).to_le_bytes());
    fs_header[0x2C..0x30].copy_from_slice(&2u32.to_le_bytes());
    let regions = [(0, hashes.len()), (hashes.len(), exefs.len())];
    for (index, (offset, size)) in regions.into_iter().enumerate() {
        fs_header[0x30 + 0x10 * index..0x38 + 0x10 * index]
            .copy_from_slice(&(offset as u64).to_le_bytes());
        fs_header[0x38 + 0x10 * index..0x40 + 0x10 * index]
            .copy_from_slice(&(size as u64).to_le_bytes());
    }
    // The upper half of the counter, as the replaced ExeFS section had it.
    let ctr_upper: [u8; 8] = template[0x400 + 0x140..0x400 + 0x148]
        .try_into()
        .expect("8 bytes");
    fs_header[0x140..0x148].copy_from_slice(&ctr_upper);

    let unit = nca::MEDIA_UNIT as usize;
    header[0x240..0x244].copy_from_slice(&((SECTION_START / unit) as u32).to_le_bytes());
    header[0x244..0x248].copy_from_slice(&((section_end / unit) as u32).to_le_bytes());
    // What the firmware's archives have after each section's bounds.
    header[0x248] = 1;
    header[0x280..0x2A0].copy_from_slice(&crypto::sha256(&fs_header));
    header[0x400..0x600].copy_from_slice(&fs_header);
    header[0x208..0x210].copy_from_slice(&(section_end as u64).to_le_bytes());

    // The section key: one that follows from what it encrypts.
    let section_key: [u8; 16] = crypto::sha256(&section)[..16].try_into().unwrap();
    let revision = template[0x206].max(template[0x220]).saturating_sub(1);
    let master_key = keys
        .master_key(revision)
        .ok_or_else(|| ProgramError::Key(format!("master_key_{revision:02x} is missing")))?;
    let key_area_key = crypto::key_area_key(keys, usize::from(template[0x207]), master_key)
        .map_err(|missing| ProgramError::Key(missing.to_string()))?;
    let slot = 0x300 + nca::CTR_KEY_SLOT * 16;
    header[slot..slot + 16].copy_from_slice(&crypto::aes_encrypt(&key_area_key, &section_key));
    let counter = (u128::from(u64::from_le_bytes(ctr_upper)) << 64) | (SECTION_START as u128 / 16);
    crypto::ctr_apply(&section_key, counter, &mut section);

    let signature = sign(&header[0x200..0x400]);
    header[..SIGNATURE_SIZE].copy_from_slice(&signature);
    header[SIGNATURE_SIZE..2 * SIGNATURE_SIZE].copy_from_slice(&signature);
    let header_key =
        crypto::nca_header_key(keys).map_err(|missing| ProgramError::Key(missing.to_string()))?;
    Xts::new(&header_key, nca::XTS_SECTOR).encrypt(0, &mut header);

    let mut archive = header;
    archive.extend_from_slice(&section);
    Ok(archive)
}

#[cfg(test)]
mod tests {
    use rsa::{BigUint, RsaPublicKey, pss::Pss};
    use sha2::{Digest, Sha256};

    use super::*;

    const PROGRAM_ID: u64 = 0x0100_0000_0000_100D;

    /// Made-up keys: the ladder only has to be the same both ways.
    fn keys() -> KeySet {
        let key = |byte: u8| format!("{byte:02x}").repeat(16);
        KeySet::parse(&format!(
            "aes_kek_generation_source = {}\naes_key_generation_source = {}\n\
             header_kek_source = {}\nheader_key_source = {}{}\n\
             key_area_key_application_source = {}\nmaster_key_00 = {}\n",
            key(1),
            key(2),
            key(3),
            key(4),
            key(5),
            key(6),
            key(7),
        ))
        .unwrap()
    }

    /// The archive to replace: an applet's, in the applet pool.
    fn original(keys: &KeySet) -> Vec<u8> {
        let mut template = [0u8; nca::HEADER_SIZE];
        template[0x206] = 1;
        template[0x21C..0x220].copy_from_slice(&0x1702_0300u32.to_le_bytes());
        template[0x220] = 1;
        template[0x540..0x548].copy_from_slice(&[0, 0, 0, 0, 1, 0, 0, 0]);
        seal(keys, &template, PROGRAM_ID, &nsp_with(PROGRAM_ID, 1 << 2)).unwrap()
    }

    fn npdm(program_id: u64, acid_flags: u32) -> Vec<u8> {
        let (acid_offset, acid_size, aci_offset) = (0x80, 0x240, 0x2C0);
        let mut npdm = vec![0u8; aci_offset + 0x40];
        npdm[..4].copy_from_slice(b"META");
        for (offset, value) in [
            (0x70, aci_offset),
            (0x74, 0x40),
            (0x78, acid_offset),
            (0x7C, acid_size),
        ] {
            npdm[offset..offset + 4].copy_from_slice(&(value as u32).to_le_bytes());
        }
        let acid = &mut npdm[acid_offset..acid_offset + acid_size];
        acid[0x200..0x204].copy_from_slice(b"ACID");
        acid[0x204..0x208].copy_from_slice(&((acid_size - 0x100) as u32).to_le_bytes());
        acid[0x20C..0x210].copy_from_slice(&acid_flags.to_le_bytes());
        npdm[aci_offset..aci_offset + 4].copy_from_slice(b"ACI0");
        npdm[aci_offset + 0x10..aci_offset + 0x18].copy_from_slice(&program_id.to_le_bytes());
        npdm
    }

    fn nsp_with(program_id: u64, acid_flags: u32) -> Vec<u8> {
        let code: Vec<u8> = (0..0x9000u32).map(|index| index as u8).collect();
        pfs0(&[
            ("main".into(), code),
            (NPDM.into(), npdm(program_id, acid_flags)),
        ])
    }

    /// A production ACID in the application pool, as hbloader's.
    fn nsp(program_id: u64) -> Vec<u8> {
        nsp_with(program_id, 1)
    }

    fn verifies(message: &[u8], signature: &[u8]) -> bool {
        let public =
            RsaPublicKey::new(BigUint::from_bytes_be(&modulus()), BigUint::from(65537u32)).unwrap();
        public
            .verify(Pss::new::<Sha256>(), &Sha256::digest(message), signature)
            .is_ok()
    }

    #[test]
    fn the_archive_opens_and_is_signed_with_the_test_key() {
        let keys = keys();
        let original = original(&keys);
        let original = Nca::open(&original, &keys).unwrap();
        let (archive, dropped) =
            build_program_nca(&keys, &original, PROGRAM_ID, &nsp(PROGRAM_ID)).unwrap();
        assert!(dropped.is_empty());

        let built = Nca::open(&archive, &keys).unwrap();
        assert_eq!(built.program_id(), PROGRAM_ID);
        assert_eq!(built.content_type(), nca::CONTENT_TYPE_PROGRAM);
        assert_eq!(built.size(), archive.len() as u64);
        let header = built.header();
        assert_eq!(header[0x221], KEY_GENERATION);
        assert_eq!(header[0x21C..0x221], original.header()[0x21C..0x221]);
        assert!(
            verifies(&header[0x200..0x400], &header[..0x100]),
            "signature 1"
        );
        assert!(
            verifies(&header[0x200..0x400], &header[0x100..0x200]),
            "signature 2"
        );

        // Decrypted and checked against its hashes by the reader.
        let exefs = built.partition_section(0).unwrap();
        let files = nca::pfs0_files(&exefs).unwrap();
        let names: Vec<&str> = files.iter().map(|(name, _)| *name).collect();
        assert_eq!(names, ["main", NPDM]);
        assert_eq!(files[0].1.len(), 0x9000);
        let npdm = files[1].1;
        assert_eq!(npdm[4], KEY_GENERATION);
        let acid = &npdm[0x80..0x80 + 0x240];
        assert_eq!(acid[0x100..0x200], modulus());
        assert_eq!(
            acid[0x20C..0x210],
            (1u32 | 1 << 2).to_le_bytes(),
            "applet pool"
        );
        assert!(verifies(&acid[0x100..], &acid[..0x100]), "ACID signature");
    }

    #[test]
    fn trace_buffer_requests_are_dropped_from_both_tables() {
        let mut npdm = npdm(PROGRAM_ID, 1);
        let (aci, acid) = (0x2C0, 0x80);
        // Eight bytes of kernel capabilities at the end of each.
        for (start, field, offset) in [(aci, 0x30, 0x38usize), (acid, 0x230, 0x238)] {
            npdm[start + field..start + field + 4].copy_from_slice(&(offset as u32).to_le_bytes());
            npdm[start + field + 4..start + field + 8].copy_from_slice(&8u32.to_le_bytes());
            // Trace buffer read-only and a second region kept; program type.
            let map = 0x3FFu32 | 0x41 << 11 | 0x02 << 18;
            npdm[start + offset..start + offset + 4].copy_from_slice(&map.to_le_bytes());
            npdm[start + offset + 4..start + offset + 8].copy_from_slice(&0x9FFFu32.to_le_bytes());
        }
        let mut alone = npdm.clone();
        let at = aci + 0x38;
        alone[at..at + 4].copy_from_slice(&0x20BFFu32.to_le_bytes());

        assert_eq!(drop_absent_regions(&mut npdm).unwrap().len(), 2);
        for start in [aci + 0x38, acid + 0x238] {
            assert_eq!(u32_at(&npdm, start), Some(0x3FF | 0x02 << 18));
            assert_eq!(u32_at(&npdm, start + 4), Some(0x9FFF));
        }
        drop_absent_regions(&mut alone).unwrap();
        assert_eq!(u32_at(&alone, aci + 0x38), Some(PADDING));
    }

    #[test]
    fn the_npdm_has_to_be_for_the_program() {
        let keys = keys();
        let original = original(&keys);
        let original = Nca::open(&original, &keys).unwrap();
        assert_eq!(
            build_program_nca(&keys, &original, PROGRAM_ID, &nsp(PROGRAM_ID + 1)),
            Err(ProgramError::ProgramId {
                npdm: PROGRAM_ID + 1,
                expected: PROGRAM_ID
            })
        );
    }
}
