//! Package2 as the secure monitor takes it on cold boot: the kernel and
//! whatever else Nintendo signs into it, checked, decrypted and placed in
//! DRAM before the monitor drops to the kernel at EL1.
//!
//! The format is public (Switchbrew "Package2"). The file is a 0x100-byte
//! RSA-2048 signature, a 0x100-byte encrypted header ("meta") and the
//! encrypted payloads, one after the other. The monitor, in this order:
//!
//! 1. verifies the signature (RSASSA-PSS, SHA-256, salt as long as the
//!    digest) over the meta *as stored*, still encrypted, with the retail
//!    public modulus it carries ([`RETAIL_MODULUS`]);
//! 2. decrypts the meta with AES-128-CTR, its first 16 bytes being the
//!    counter, under `package2_key = D(master_key_XX, package2_key_source)`,
//!    `XX` being the key generation the counter encodes; the stored counter
//!    is then copied back over the first 16 decrypted bytes;
//! 3. checks the meta: magic `PK21`, total size, key generation, payload
//!    bounds and entry point, then the package2 and bootloader versions;
//! 4. checks each payload's SHA-256, taken over its *encrypted* bytes, and
//!    decrypts it with its own counter to DRAM base + its offset;
//! 5. enters the kernel at DRAM base + the entry point the meta names.
//!
//! \[inference\] Step order, the key derivation and the meta checks other
//! than those Switchbrew lists follow Atmosphère's exosphère (reference
//! only, nothing copied), and were confirmed against the 23.5.0 package2:
//! the derived key equals the published `package2_key_16`, the signature
//! verifies over the encrypted meta and not the decrypted one, and the
//! payload hashes match the encrypted bytes.
//!
//! For this generation (8.0.0 and later, Switchbrew) the initial process
//! archive INI1 sits inside the kernel image and payload 1 is empty; the
//! kernel's own loader finds it, which [`KernelImage`] reproduces.
//!
//! Development units can tell the monitor, through the boot config, that
//! package2 is plaintext or unsigned; a retail console cannot, and only the
//! retail path is modelled.

use std::{fmt, ops::Range};

use rsa::{BigUint, RsaPublicKey, pss::Pss};
use sha2::Sha256;

use crate::{
    image::{ImageError, KernelImage},
    keys::{Key, KeySet},
    nand::crypto::{aes_decrypt, ctr_apply, sha256},
};

/// Signature plus meta.
pub const HEADER_SIZE: usize = 0x200;
const SIGNATURE: Range<usize> = 0..0x100;
const META: Range<usize> = 0x100..HEADER_SIZE;
const META_SIZE: usize = META.end - META.start;

/// \[inference\] The meta has room for four payloads (Switchbrew), but the
/// fourth slot's counter, size, offset and hash are all zero in every
/// firmware here; Atmosphère's monitor treats the format as three payloads
/// and that slot as padding.
pub const PAYLOAD_COUNT: usize = 3;

/// Payload offsets and the entry point are relative to the start of DRAM
/// (Switchbrew: "typically 0x80000000"); the kernel of 23.5.0 is linked at
/// `0x80060000`, its offset `0x60000`.
pub const DRAM_BASE: u64 = 0x8000_0000;

/// The retail package2 public modulus, big-endian, exponent 65537
/// (Switchbrew "Package2", "Public Keys", "Retail Modulus"). It is public
/// and the monitor holds it as a constant, so it lives here rather than in
/// the key file.
pub const RETAIL_MODULUS: [u8; 256] = [
    0x8D, 0x13, 0xA7, 0x77, 0x6A, 0xE5, 0xDC, 0xC0, 0x3B, 0x25, 0xD0, 0x58, 0xE4, 0x20, 0x69, 0x59,
    0x55, 0x4B, 0xAB, 0x70, 0x40, 0x08, 0x28, 0x07, 0xA8, 0xA7, 0xFD, 0x0F, 0x31, 0x2E, 0x11, 0xFE,
    0x47, 0xA0, 0xF9, 0x9D, 0xDF, 0x80, 0xDB, 0x86, 0x5A, 0x27, 0x89, 0xCD, 0x97, 0x6C, 0x85, 0xC5,
    0x6C, 0x39, 0x7F, 0x41, 0xF2, 0xFF, 0x24, 0x20, 0xC3, 0x95, 0xA6, 0xF7, 0x9D, 0x4A, 0x45, 0x74,
    0x8B, 0x5D, 0x28, 0x8A, 0xC6, 0x99, 0x35, 0x68, 0x85, 0xA5, 0x64, 0x32, 0x80, 0x9F, 0xD3, 0x48,
    0x39, 0xA2, 0x1D, 0x24, 0x67, 0x69, 0xDF, 0x75, 0xAC, 0x12, 0xB5, 0xBD, 0xC3, 0x29, 0x90, 0xBE,
    0x37, 0xE4, 0xA0, 0x80, 0x9A, 0xBE, 0x36, 0xBF, 0x1F, 0x2C, 0xAB, 0x2B, 0xAD, 0xF5, 0x97, 0x32,
    0x9A, 0x42, 0x9D, 0x09, 0x8B, 0x08, 0xF0, 0x63, 0x47, 0xA3, 0xE9, 0x1B, 0x36, 0xD8, 0x2D, 0x8A,
    0xD7, 0xE1, 0x54, 0x11, 0x95, 0xE4, 0x45, 0x88, 0x69, 0x8A, 0x2B, 0x35, 0xCE, 0xD0, 0xA5, 0x0B,
    0xD5, 0x5D, 0xAC, 0xDB, 0xAF, 0x11, 0x4D, 0xCA, 0xB8, 0x1E, 0xE7, 0x01, 0x9E, 0xF4, 0x46, 0xA3,
    0x8A, 0x94, 0x6D, 0x76, 0xBD, 0x8A, 0xC8, 0x3B, 0xD2, 0x31, 0x58, 0x0C, 0x79, 0xA8, 0x26, 0xE9,
    0xD1, 0x79, 0x9C, 0xCB, 0xD4, 0x2B, 0x6A, 0x4F, 0xC6, 0xCC, 0xCF, 0x90, 0xA7, 0xB9, 0x98, 0x47,
    0xFD, 0xFA, 0x4C, 0x6C, 0x6F, 0x81, 0x87, 0x3B, 0xCA, 0xB8, 0x50, 0xF6, 0x3E, 0x39, 0x5D, 0x4D,
    0x97, 0x3F, 0x0F, 0x35, 0x39, 0x53, 0xFB, 0xFA, 0xCD, 0xAB, 0xA8, 0x7A, 0x62, 0x9A, 0x3F, 0xF2,
    0x09, 0x27, 0x96, 0x3F, 0x07, 0x9A, 0x91, 0xF7, 0x16, 0xBF, 0xC6, 0x3A, 0x82, 0x5A, 0x4B, 0xCF,
    0x49, 0x50, 0x95, 0x8C, 0x55, 0x80, 0x7E, 0x39, 0xB1, 0x48, 0x05, 0x1E, 0x21, 0xC7, 0x24, 0x4F,
];
const PUBLIC_EXPONENT: u32 = 65537;
/// RSASSA-PSS salt length: the SHA-256 digest length.
const PSS_SALT_SIZE: usize = 32;

/// Offsets in the decrypted meta (Switchbrew "Header").
const META_COUNTER: Range<usize> = 0x00..0x10;
const META_PAYLOAD_COUNTERS: usize = 0x10;
const META_MAGIC: Range<usize> = 0x50..0x54;
const META_ENTRY: usize = 0x54;
const META_PACKAGE2_VERSION: usize = 0x5C;
const META_BOOTLOADER_VERSION: usize = 0x5D;
const META_PAYLOAD_SIZES: usize = 0x60;
const META_PAYLOAD_OFFSETS: usize = 0x70;
const META_PAYLOAD_HASHES: usize = 0x80;
const MAGIC: &[u8; 4] = b"PK21";

/// \[inference\] The newest key generation the monitor of this firmware
/// generation can load a master key for: `0x16`, the generation the
/// 23.0.0 and 23.5.0 package2 counters encode and the newest
/// `master_key_XX` published for them. The real monitor refuses a higher
/// one (Atmosphère: `key_generation >= KeyGeneration_Count`).
pub const NEWEST_KEY_GENERATION: u8 = 0x16;

/// \[inference\] The monitor accepts a package2 only if its bootloader
/// version is at most the monitor's own and its package2 version at least
/// the monitor's minimum (Switchbrew, whose table of the two constants
/// ends at 13.0.0). Both have gone up by one with each firmware here
/// (21.0.0: 0x16 and 0x1D, 22.5.0: 0x17 and 0x1E, 23.0.0: 0x18 and 0x1F,
/// 23.5.0: 0x19 and 0x20), so the HLE monitor, which stands for the
/// 23.5.0 one, takes the values its own package2 carries. The real
/// minimum may be lower; a package2 between it and this value would be
/// refused here and accepted by the console.
pub const CURRENT_BOOTLOADER_VERSION: u8 = 0x19;
/// See [`CURRENT_BOOTLOADER_VERSION`].
pub const MINIMUM_PACKAGE2_VERSION: u8 = 0x20;

/// Payload 0 is the kernel (Switchbrew "Section 0").
pub const KERNEL_PAYLOAD: usize = 0;
/// Payload 1 held INI1 before 8.0.0 (Switchbrew "Section 1").
const INI1_PAYLOAD: usize = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Package2Error {
    Truncated {
        needed: u64,
        available: usize,
    },
    MissingKey(&'static str),
    MissingMasterKey(u8),
    /// The RSA-PSS signature over the encrypted meta does not verify.
    Signature,
    /// The decrypted meta lacks `PK21`: the wrong key or not a package2.
    Magic,
    Header(&'static str),
    KeyGeneration(u8),
    Version {
        package2: u8,
        bootloader: u8,
    },
    PayloadHash(usize),
    /// Valid, but not something this emulator models.
    Unsupported(&'static str),
}

impl fmt::Display for Package2Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated { needed, available } => write!(
                f,
                "package2 needs {needed:#x} bytes, {available:#x} are there"
            ),
            Self::MissingKey(name) => write!(f, "package2 needs `{name}` from the key file"),
            Self::MissingMasterKey(generation) => write!(
                f,
                "package2 needs `master_key_{generation:02x}` from the key file"
            ),
            Self::Signature => write!(f, "package2 signature does not verify"),
            Self::Magic => write!(f, "package2 header does not decrypt to PK21"),
            Self::Header(what) => write!(f, "package2 header rejected: {what}"),
            Self::KeyGeneration(generation) => write!(
                f,
                "package2 key generation {generation:#x} is newer than the monitor's \
                 {NEWEST_KEY_GENERATION:#x}"
            ),
            Self::Version {
                package2,
                bootloader,
            } => write!(
                f,
                "package2 version {package2:#x} / bootloader version {bootloader:#x} rejected \
                 (monitor: package2 >= {MINIMUM_PACKAGE2_VERSION:#x}, bootloader <= \
                 {CURRENT_BOOTLOADER_VERSION:#x})"
            ),
            Self::PayloadHash(index) => write!(f, "package2 payload {index} hash mismatch"),
            Self::Unsupported(what) => write!(f, "package2 not modelled: {what}"),
        }
    }
}

impl std::error::Error for Package2Error {}

/// A decrypted payload and where the monitor puts it.
#[derive(Clone, PartialEq, Eq)]
pub struct Payload {
    /// Physical address: [`DRAM_BASE`] + the payload's offset.
    pub address: u64,
    pub bytes: Vec<u8>,
}

/// The bytes stay out of logs: megabytes of kernel.
impl fmt::Debug for Payload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Payload")
            .field("address", &format_args!("{:#x}", self.address))
            .field("size", &format_args!("{:#x}", self.bytes.len()))
            .finish()
    }
}

/// What the monitor has verified and placed once it is done with package2.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Package2 {
    /// Index of the master key the package2 key was derived from.
    pub key_generation: u8,
    pub package2_version: u8,
    pub bootloader_version: u8,
    /// Physical address the monitor enters at EL1: [`DRAM_BASE`] + the
    /// meta's entry point.
    pub entry: u64,
    /// By payload index; an empty payload has no bytes.
    pub payloads: [Payload; PAYLOAD_COUNT],
}

impl Package2 {
    pub fn kernel(&self) -> &Payload {
        &self.payloads[KERNEL_PAYLOAD]
    }

    /// The kernel as [`KernelImage`] decodes it, embedded INI1 included.
    pub fn into_kernel_image(self) -> Result<KernelImage, ImageError> {
        let [kernel, ..] = self.payloads;
        KernelImage::from_flat(kernel.address, kernel.bytes)
    }
}

/// Verifies, decrypts and places `package2` as the monitor does on cold
/// boot. `package2` may run past the size the header encodes (the rest of
/// a BCPKG2 partition); those bytes are not read.
pub fn unpack(package2: &[u8], keys: &KeySet) -> Result<Package2, Package2Error> {
    let header = package2
        .get(..HEADER_SIZE)
        .ok_or(Package2Error::Truncated {
            needed: HEADER_SIZE as u64,
            available: package2.len(),
        })?;
    verify_signature(&header[SIGNATURE], &header[META])?;

    let stored: &[u8; META_SIZE] = header[META].try_into().expect("meta range");
    let counter: &Key = stored[META_COUNTER].try_into().expect("counter range");
    let word = |index: usize| u32::from_le_bytes(counter[index * 4..][..4].try_into().unwrap());
    // Switchbrew: size = ctr_word0 ^ ctr_word2 ^ ctr_word3; key generation
    // = ctr byte 4 ^ byte 6 ^ byte 7. [inference] The master key index is
    // one less, floored at 0 (Atmosphère `GetKeyGeneration`): for 23.5.0
    // the bytes give 0x17 and master_key_16 is the key.
    let size = u64::from(word(0) ^ word(2) ^ word(3));
    let key_generation = (counter[4] ^ counter[6] ^ counter[7]).saturating_sub(1);

    let source = keys
        .package2_key_source()
        .ok_or(Package2Error::MissingKey("package2_key_source"))?;
    let master_key = keys
        .master_key(key_generation)
        .ok_or(Package2Error::MissingMasterKey(key_generation))?;
    let key = aes_decrypt(master_key, source);

    let mut meta = *stored;
    ctr_apply(&key, u128::from_be_bytes(*counter), &mut meta);
    meta[META_COUNTER].copy_from_slice(counter);

    let layout = verify_meta(&meta, size, key_generation)?;
    let package2_version = meta[META_PACKAGE2_VERSION];
    let bootloader_version = meta[META_BOOTLOADER_VERSION];
    if bootloader_version > CURRENT_BOOTLOADER_VERSION
        || package2_version < MINIMUM_PACKAGE2_VERSION
    {
        return Err(Package2Error::Version {
            package2: package2_version,
            bootloader: bootloader_version,
        });
    }
    if package2.len() < size as usize {
        return Err(Package2Error::Truncated {
            needed: size,
            available: package2.len(),
        });
    }

    let mut cursor = HEADER_SIZE;
    let mut payloads = [const { Vec::new() }; PAYLOAD_COUNT];
    for (index, bytes) in payloads.iter_mut().enumerate() {
        let encrypted = &package2[cursor..][..layout.sizes[index] as usize];
        cursor += encrypted.len();
        if sha256(encrypted) != meta[META_PAYLOAD_HASHES + index * 32..][..32] {
            return Err(Package2Error::PayloadHash(index));
        }
        let counter: &Key = meta[META_PAYLOAD_COUNTERS + index * 16..][..16]
            .try_into()
            .unwrap();
        bytes.extend_from_slice(encrypted);
        ctr_apply(&key, u128::from_be_bytes(*counter), bytes);
    }

    if !payloads[INI1_PAYLOAD].is_empty() {
        return Err(Package2Error::Unsupported(
            "INI1 as payload 1 (firmware before 8.0.0); this generation's kernel carries it",
        ));
    }
    // The monitor would enter wherever the meta points; the kernel image
    // starts with its entry stub, and [`KernelImage`] needs it there.
    if layout.entry != layout.offsets[KERNEL_PAYLOAD] {
        return Err(Package2Error::Unsupported(
            "entry point other than the start of the kernel payload",
        ));
    }

    let [kernel, ini1, reserved] = payloads;
    let place = |index: usize, bytes| Payload {
        address: DRAM_BASE + u64::from(layout.offsets[index]),
        bytes,
    };
    Ok(Package2 {
        key_generation,
        package2_version,
        bootloader_version,
        entry: DRAM_BASE + u64::from(layout.entry),
        payloads: [place(0, kernel), place(1, ini1), place(2, reserved)],
    })
}

fn verify_signature(signature: &[u8], meta: &[u8]) -> Result<(), Package2Error> {
    let public = RsaPublicKey::new(
        BigUint::from_bytes_be(&RETAIL_MODULUS),
        BigUint::from(PUBLIC_EXPONENT),
    )
    .expect("the retail modulus is a valid 2048-bit key");
    public
        .verify(
            Pss::new_with_salt::<Sha256>(PSS_SALT_SIZE),
            &sha256(meta),
            signature,
        )
        .map_err(|_| Package2Error::Signature)
}

struct Layout {
    entry: u32,
    sizes: [u32; PAYLOAD_COUNT],
    offsets: [u32; PAYLOAD_COUNT],
}

/// The monitor's checks of the decrypted meta. \[inference\] Those beyond
/// the magic and the size (Switchbrew) are Atmosphère's: the key
/// generation, payloads that neither wrap nor overlap, an entry point in
/// one of them, and sizes that add up to the encoded total. The real
/// monitor also bounds the total size and requires an alignment of entry
/// and sizes; their values are not established here, so the check is only
/// that the entry is a 4-byte aligned AArch64 instruction address.
fn verify_meta(
    meta: &[u8; META_SIZE],
    size: u64,
    key_generation: u8,
) -> Result<Layout, Package2Error> {
    if &meta[META_MAGIC] != MAGIC {
        return Err(Package2Error::Magic);
    }
    if key_generation > NEWEST_KEY_GENERATION {
        return Err(Package2Error::KeyGeneration(key_generation));
    }
    let read = |offset: usize| u32::from_le_bytes(meta[offset..][..4].try_into().unwrap());
    let layout = Layout {
        entry: read(META_ENTRY),
        sizes: std::array::from_fn(|index| read(META_PAYLOAD_SIZES + index * 4)),
        offsets: std::array::from_fn(|index| read(META_PAYLOAD_OFFSETS + index * 4)),
    };
    if size <= HEADER_SIZE as u64 {
        return Err(Package2Error::Header("total size within the header"));
    }
    if !layout.entry.is_multiple_of(4) {
        return Err(Package2Error::Header("entry point not 4-byte aligned"));
    }
    let span = |index: usize| {
        let start = u64::from(layout.offsets[index]);
        start..start + u64::from(layout.sizes[index])
    };
    for index in 0..PAYLOAD_COUNT {
        if span(index).end > u64::from(u32::MAX) {
            return Err(Package2Error::Header("payload wraps the address space"));
        }
        for other in index + 1..PAYLOAD_COUNT {
            let (a, b) = (span(index), span(other));
            if !a.is_empty() && !b.is_empty() && a.start < b.end && b.start < a.end {
                return Err(Package2Error::Header("payloads overlap"));
            }
        }
    }
    if !(0..PAYLOAD_COUNT).any(|index| span(index).contains(&u64::from(layout.entry))) {
        return Err(Package2Error::Header("entry point outside every payload"));
    }
    let total = HEADER_SIZE as u64
        + layout
            .sizes
            .iter()
            .map(|&size| u64::from(size))
            .sum::<u64>();
    if total != size {
        return Err(Package2Error::Header(
            "payload sizes do not add up to the encoded total",
        ));
    }
    Ok(layout)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    const KEYS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../prod_newer.keys");
    const PACKAGE2: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../fw_dump/0100000000000819/da6ae627428f5fe36e26aafa833515a8/section0_romfs/nx/package2"
    );
    const KERNEL_ELF: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../fw_kernel/kernel_normal.elf"
    );

    fn present(paths: &[&str]) -> bool {
        let missing: Vec<_> = paths
            .iter()
            .filter(|path| !Path::new(path).exists())
            .collect();
        if !missing.is_empty() {
            eprintln!("skipped: {missing:?} absent");
        }
        missing.is_empty()
    }

    fn firmware() -> Option<(Vec<u8>, KeySet)> {
        present(&[KEYS, PACKAGE2]).then(|| {
            (
                std::fs::read(PACKAGE2).unwrap(),
                KeySet::load(Path::new(KEYS)).unwrap(),
            )
        })
    }

    /// The bytes and physical address of the ELF's one PT_LOAD.
    fn pt_load(elf: &[u8]) -> (u64, &[u8]) {
        let u16_at = |offset: usize| u16::from_le_bytes(elf[offset..][..2].try_into().unwrap());
        let u32_at = |offset: usize| u32::from_le_bytes(elf[offset..][..4].try_into().unwrap());
        let u64_at = |offset: usize| u64::from_le_bytes(elf[offset..][..8].try_into().unwrap());
        let (table, count) = (u64_at(0x20) as usize, u16_at(0x38) as usize);
        let mut loads = (0..count)
            .map(|index| table + index * 56)
            .filter(|&entry| u32_at(entry) == 1);
        let entry = loads.next().expect("a PT_LOAD");
        assert!(loads.next().is_none(), "one PT_LOAD");
        let (offset, size) = (u64_at(entry + 8) as usize, u64_at(entry + 32) as usize);
        (u64_at(entry + 24), &elf[offset..offset + size])
    }

    #[test]
    fn unpacks_the_firmware_kernel_as_the_monitor_places_it() {
        let Some((package2, keys)) = firmware() else {
            return;
        };
        if !present(&[KERNEL_ELF]) {
            return;
        }
        let unpacked = unpack(&package2, &keys).unwrap();
        assert_eq!(unpacked.key_generation, 0x16);
        assert_eq!(
            (unpacked.package2_version, unpacked.bootloader_version),
            (0x20, 0x19)
        );
        assert_eq!(unpacked.entry, 0x8006_0000);
        assert!(unpacked.payloads[1].bytes.is_empty() && unpacked.payloads[2].bytes.is_empty());

        let elf = std::fs::read(KERNEL_ELF).unwrap();
        let (address, segment) = pt_load(&elf);
        assert_eq!(unpacked.kernel().address, address);
        assert_eq!(sha256(&unpacked.kernel().bytes), sha256(segment));

        let image = unpacked.into_kernel_image().unwrap();
        assert_eq!(image.load_address(), 0x8006_0000);
        let ini1 = image.ini1.expect("INI1 embedded in the kernel");
        let start = (ini1.address - image.load_address()) as usize;
        assert_eq!(&image.bytes()[start..start + 4], b"INI1");
        assert!(ini1.process_count > 0);
        eprintln!(
            "INI1 at {:#x}, {:#x} bytes, {} KIPs",
            ini1.address, ini1.size, ini1.process_count
        );
    }

    #[test]
    fn a_flipped_byte_in_a_payload_is_refused() {
        let Some((mut package2, keys)) = firmware() else {
            return;
        };
        package2[HEADER_SIZE + 0x1234] ^= 0x01;
        assert_eq!(unpack(&package2, &keys), Err(Package2Error::PayloadHash(0)));
    }

    #[test]
    fn a_flipped_byte_in_the_header_is_refused_by_the_signature() {
        let Some((mut package2, keys)) = firmware() else {
            return;
        };
        package2[META.start + 0x60] ^= 0x01;
        assert_eq!(unpack(&package2, &keys), Err(Package2Error::Signature));
        assert_eq!(
            unpack(&package2[..0x1FF], &keys),
            Err(Package2Error::Truncated {
                needed: 0x200,
                available: 0x1FF
            })
        );
    }

    #[test]
    fn a_truncated_body_is_refused() {
        let Some((package2, keys)) = firmware() else {
            return;
        };
        let cut = &package2[..package2.len() - 16];
        assert!(matches!(
            unpack(cut, &keys),
            Err(Package2Error::Truncated { .. })
        ));
    }

    #[test]
    fn without_the_key_source_the_header_cannot_be_opened() {
        let Some((package2, _)) = firmware() else {
            return;
        };
        assert_eq!(
            unpack(&package2, &KeySet::default()),
            Err(Package2Error::MissingKey("package2_key_source"))
        );
    }
}
