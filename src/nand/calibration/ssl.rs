//! The console's TLS client identity: an RSA-2048 key pair and an X.509
//! certificate, which `ssl` (program `0100000000000024`) loads at boot and
//! presents when a client asks for the console's built-in certificate
//! (`RegisterInternalPki`, as `nim` does before it talks to Nintendo's
//! servers).
//!
//! # What the firmware reads
//!
//! `settings` serves both halves through `set:cal`:
//!
//! - Command 16 (`0x4ca0`), the key: the extended SSL key at `0x3AE0`
//!   ([`super::EXTENDED_SSL_KEY`]: CRC over `0x13E` bytes at `0x3C1E`),
//!   answered as a size of `0x134` and the bytes; from a version of 8 or
//!   less, with the key generation (its last four bytes) zeroed. When that
//!   CRC fails it answers the old key at `0x9B0` (`0x110` bytes) instead,
//!   and when that fails too, `0x48C69` (2105-0582).
//! - Command 17 (`0x4dc0`), the certificate: its size, eight bytes at
//!   `0xAD0` with their CRC at `0xADE`, at most `0x800` bytes from `0xAE0`,
//!   and their SHA-256 at `0x12E0` (`0x48E69` on a mismatch).
//!
//! `ssl` builds a PKCS #11 token of them at boot (`0x14cde4`):
//!
//! - It asks for the key (`0x14e230`; a size of `0x135` or more aborts)
//!   and reads the key generation, the data's last four bytes (`0x14d9a4`).
//! - Generation 0 (`0x36080`, `0x35f80`): it has `spl` derive an access key
//!   (`GenerateAesKek` with the kek source at `0x2633E0`, the generation,
//!   and option 32, sealed for `DecryptDeviceUniqueData`, not device
//!   unique) and decrypt the data before the generation with it and the
//!   key source of a retail console, the second of the two constants of
//!   `0x36080` (a development console uses the first). It gets back
//!   `0x100` bytes: the private exponent, which it reads as an unsigned
//!   big-endian number (`0x14dc58`, `0x11a854`) and keeps.
//! - A later generation (`0x360d0`) has the monitor import the key instead
//!   (`DecryptAndStoreSslClientCertKey`), and uses it through the monitor.
//! - Then it asks for the certificate (`0x14e2a8`), parses it as DER
//!   (`0xc1b00`) and takes modulus and public exponent from it
//!   (`0x14d8e4`): the key data carries no modulus.
//!
//! `spl` (`0x11340`) takes `0x31` to `0x130` bytes for
//! `DecryptDeviceUniqueData`, has the monitor decrypt them
//! ([`crate::secmon::crypto::seal_device_unique_data`] is the format: IV,
//! encrypted payload with padding and device ID, GMAC), and copies all but
//! the last `0x30` back. The keys are the console's: master key 0 and the
//! monitor's static source for that seal key, then the two sources `ssl`
//! passes (key file names in [`crate::keys`]).
//!
//! # What the virtual console has
//!
//! A console's identity comes from Nintendo's factory, signed by
//! Nintendo, and only that is accepted by Nintendo's servers. This one is
//! MADE UP, a CHOICE so that the console's TLS stack has a client identity
//! to load, as every console's does: it lets programs that require one get
//! as far as the server, which rejects it. It is
//!
//! - an RSA-2048 key with the usual public exponent 65537, generated from a
//!   seed that is a function of the device ID, so the same console gets the
//!   same identity (as long as the key generation of the `rsa` crate does
//!   not change: what an image holds stays as it was written);
//! - in the format of key generation 0, the one `ssl` decrypts itself
//!   (CHOICE: a later generation needs the monitor to keep the key, which
//!   it does not model);
//! - with a self-signed X.509 v3 certificate whose subject names the
//!   console by its device ID and whose issuer is a "horizon-emu test CA"
//!   that does not exist: the certificate is signed with its own key.

use rand_chacha::{ChaCha20Rng, rand_core::SeedableRng};
use rsa::{
    BigUint, Pkcs1v15Sign, RsaPrivateKey,
    traits::{PrivateKeyParts, PublicKeyParts},
};
use sha2::{Digest, Sha256};

use super::super::NandError;
use crate::{
    keys::KeySet,
    secmon::crypto::{self, CryptoError, SEAL_DECRYPT_DEVICE_UNIQUE_DATA},
};

/// Bytes of the key data `settings` answers for the extended SSL key: the
/// wrapped private exponent, then the key generation.
pub const KEY_SIZE: usize = WRAPPED_SIZE + 4;
const WRAPPED_SIZE: usize = MODULUS_SIZE + crypto::DEVICE_UNIQUE_DATA_OVERHEAD;
const MODULUS_SIZE: usize = 0x100;
/// Most certificate `settings` serves.
pub const CERTIFICATE_CAPACITY: usize = 0x800;

/// The key generation of the format `ssl` decrypts itself.
const KEY_GENERATION: u32 = 0;
const PUBLIC_EXPONENT: u32 = 65537;
/// What the seed is derived from besides the device ID.
const SEED_LABEL: &[u8] = b"horizon-emu stand-in TLS client identity\0";

/// The `DigestInfo` of a SHA-256 digest without the digest (RFC 8017,
/// section 9.2, note 1), which PKCS #1 v1.5 signatures carry.
const SHA256_DIGEST_INFO: [u8; 19] = [
    0x30, 0x31, 0x30, 0x0D, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x01, 0x05,
    0x00, 0x04, 0x20,
];

// Object identifiers, DER contents.
/// 1.2.840.113549.1.1.11, `sha256WithRSAEncryption`.
const SHA256_WITH_RSA: &[u8] = &[0x2A, 0x86, 0x48, 0x86, 0xF7, 0x0D, 0x01, 0x01, 0x0B];
/// 1.2.840.113549.1.1.1, `rsaEncryption`.
const RSA_ENCRYPTION: &[u8] = &[0x2A, 0x86, 0x48, 0x86, 0xF7, 0x0D, 0x01, 0x01, 0x01];
/// 2.5.4.3, `commonName`.
const COMMON_NAME: &[u8] = &[0x55, 0x04, 0x03];
/// 2.5.4.10, `organizationName`.
const ORGANIZATION: &[u8] = &[0x55, 0x04, 0x0A];
/// 2.5.4.11, `organizationalUnitName`.
const ORGANIZATIONAL_UNIT: &[u8] = &[0x55, 0x04, 0x0B];
/// 2.5.29.19, `basicConstraints`.
const BASIC_CONSTRAINTS: &[u8] = &[0x55, 0x1D, 0x13];
/// 2.5.29.15, `keyUsage`.
const KEY_USAGE: &[u8] = &[0x55, 0x1D, 0x0F];
/// 2.5.29.37, `extKeyUsage`.
const EXTENDED_KEY_USAGE: &[u8] = &[0x55, 0x1D, 0x25];
/// 1.3.6.1.5.5.7.3.2, `id-kp-clientAuth`.
const CLIENT_AUTH: &[u8] = &[0x2B, 0x06, 0x01, 0x05, 0x05, 0x07, 0x03, 0x02];

const ISSUER_ORGANIZATION: &str = "horizon-emu";
const ISSUER_COMMON_NAME: &str = "horizon-emu test CA";
const SUBJECT_ORGANIZATION: &str = "horizon-emu virtual console";
const SUBJECT_UNIT: &str = "made-up TLS client identity, not issued by Nintendo";
/// CHOICE: valid from the day the console was released, and for good
/// (RFC 5280, section 4.1.2.5: `99991231235959Z` for no expiry).
const NOT_BEFORE: &str = "170303000000Z";
const NOT_AFTER: &str = "99991231235959Z";

/// The made-up identity of one console, and what its calibration data
/// holds of it.
pub struct SslIdentity {
    key: RsaPrivateKey,
    certificate: Vec<u8>,
    seed: [u8; 32],
}

impl SslIdentity {
    /// The identity of the console `device_id` names; the same every time.
    pub fn generate(device_id: u64) -> Self {
        let seed: [u8; 32] = Sha256::new()
            .chain_update(SEED_LABEL)
            .chain_update(device_id.to_be_bytes())
            .finalize()
            .into();
        let key = RsaPrivateKey::new_with_exp(
            &mut ChaCha20Rng::from_seed(seed),
            MODULUS_SIZE * 8,
            &BigUint::from(PUBLIC_EXPONENT),
        )
        .expect("a 2048-bit key can be generated");
        let certificate = certificate(&key, device_id, &derive(&seed, b"serial"));
        assert!(certificate.len() <= CERTIFICATE_CAPACITY);
        Self {
            key,
            certificate,
            seed,
        }
    }

    /// The DER certificate, what command 17 answers.
    pub fn certificate(&self) -> &[u8] {
        &self.certificate
    }

    pub fn key(&self) -> &RsaPrivateKey {
        &self.key
    }

    /// The key data of command 16 for the console `device_id` names, with
    /// `keys` its key set: the private exponent wrapped as the monitor
    /// unwraps it for `ssl`, then the key generation.
    pub fn wrapped_key(&self, keys: &KeySet, device_id: u64) -> Result<[u8; KEY_SIZE], NandError> {
        let source = |source: Option<_>, name| source.ok_or(NandError::SslKey(name));
        let kek_source = source(
            keys.ssl_rsa_kekek_source(),
            "ssl_rsa_kekek_source is not in the key file",
        )?;
        let key_source = source(
            keys.ssl_rsa_kek_source(),
            "ssl_rsa_kek_source is not in the key file",
        )?;
        let key = crypto::device_unique_data_key(
            keys,
            kek_source,
            key_source,
            KEY_GENERATION.into(),
            crypto::kek_option(SEAL_DECRYPT_DEVICE_UNIQUE_DATA),
        )
        .map_err(|error| match error {
            CryptoError::Unimplemented(what) => NandError::SslKey(what),
            CryptoError::InvalidArgument => unreachable!("the option is a valid one"),
        })?;
        // A factory draws the IV at random; this one follows from the seed.
        let iv = derive(&self.seed, b"iv");
        let wrapped =
            crypto::seal_device_unique_data(&key, &iv, &self.private_exponent(), device_id);
        let mut data = [0; KEY_SIZE];
        data[..WRAPPED_SIZE].copy_from_slice(&wrapped);
        data[WRAPPED_SIZE..].copy_from_slice(&KEY_GENERATION.to_le_bytes());
        Ok(data)
    }

    /// Big-endian, as `ssl` reads it.
    fn private_exponent(&self) -> [u8; MODULUS_SIZE] {
        unsigned(self.key.d())
    }
}

/// Sixteen bytes for `purpose` that follow from `seed`.
fn derive(seed: &[u8; 32], purpose: &[u8]) -> [u8; 16] {
    let hash = Sha256::new()
        .chain_update(seed)
        .chain_update(purpose)
        .finalize();
    hash[..16].try_into().expect("sixteen bytes")
}

/// `value` as [`MODULUS_SIZE`] big-endian bytes.
fn unsigned(value: &BigUint) -> [u8; MODULUS_SIZE] {
    let bytes = value.to_bytes_be();
    let mut padded = [0; MODULUS_SIZE];
    padded[MODULUS_SIZE - bytes.len()..].copy_from_slice(&bytes);
    padded
}

/// A DER element: `tag`, the definite length of `content`, `content`.
fn element(tag: u8, content: &[u8]) -> Vec<u8> {
    let mut encoded = vec![tag];
    let len = content.len();
    if len < 0x80 {
        encoded.push(len as u8);
    } else {
        let digits = len.to_be_bytes();
        let significant = &digits[digits.iter().position(|&byte| byte != 0).unwrap()..];
        encoded.push(0x80 | significant.len() as u8);
        encoded.extend_from_slice(significant);
    }
    encoded.extend_from_slice(content);
    encoded
}

fn sequence(parts: &[Vec<u8>]) -> Vec<u8> {
    element(0x30, &parts.concat())
}

/// A non-negative INTEGER from its big-endian bytes.
fn integer(bytes: &[u8]) -> Vec<u8> {
    let first = bytes.iter().position(|&byte| byte != 0);
    let mut content = first.map_or(vec![0], |first| bytes[first..].to_vec());
    if content[0] & 0x80 != 0 {
        content.insert(0, 0);
    }
    element(0x02, &content)
}

fn oid(encoded: &[u8]) -> Vec<u8> {
    element(0x06, encoded)
}

/// A BIT STRING of whole bytes.
fn bits(bytes: &[u8]) -> Vec<u8> {
    element(0x03, &[&[0], bytes].concat())
}

/// An `AlgorithmIdentifier` with NULL parameters.
fn algorithm(identifier: &[u8]) -> Vec<u8> {
    sequence(&[oid(identifier), element(0x05, &[])])
}

/// A `Name` of one attribute per relative name, each a UTF8String.
fn name(attributes: &[(&[u8], &str)]) -> Vec<u8> {
    let relative_names: Vec<Vec<u8>> = attributes
        .iter()
        .map(|&(kind, value)| {
            element(
                0x31,
                &sequence(&[oid(kind), element(0x0C, value.as_bytes())]),
            )
        })
        .collect();
    sequence(&relative_names)
}

fn extension(identifier: &[u8], critical: bool, value: &[u8]) -> Vec<u8> {
    let mut parts = vec![oid(identifier)];
    if critical {
        parts.push(element(0x01, &[0xFF]));
    }
    parts.push(element(0x04, value));
    sequence(&parts)
}

/// The certificate of `key` for the console `device_id` names (see the
/// module documentation).
fn certificate(key: &RsaPrivateKey, device_id: u64, serial_seed: &[u8; 16]) -> Vec<u8> {
    // Positive, eight bytes long.
    let mut serial = serial_seed[..8].to_vec();
    serial[0] = serial[0] & 0x7F | 0x40;
    let public_key = sequence(&[
        integer(&key.n().to_bytes_be()),
        integer(&key.e().to_bytes_be()),
    ]);
    let subject_public_key_info = sequence(&[algorithm(RSA_ENCRYPTION), bits(&public_key)]);
    let device = format!("{device_id:016X}");
    let extensions = [
        // Not a CA.
        extension(BASIC_CONSTRAINTS, true, &sequence(&[])),
        // digitalSignature and keyEncipherment: bits 0 and 2, five unused.
        extension(KEY_USAGE, true, &element(0x03, &[5, 0xA0])),
        extension(EXTENDED_KEY_USAGE, false, &sequence(&[oid(CLIENT_AUTH)])),
    ];
    let tbs = sequence(&[
        // Version 3.
        element(0xA0, &integer(&[2])),
        integer(&serial),
        algorithm(SHA256_WITH_RSA),
        name(&[
            (ORGANIZATION, ISSUER_ORGANIZATION),
            (COMMON_NAME, ISSUER_COMMON_NAME),
        ]),
        sequence(&[
            element(0x17, NOT_BEFORE.as_bytes()),
            element(0x18, NOT_AFTER.as_bytes()),
        ]),
        name(&[
            (ORGANIZATION, SUBJECT_ORGANIZATION),
            (ORGANIZATIONAL_UNIT, SUBJECT_UNIT),
            (COMMON_NAME, &device),
        ]),
        subject_public_key_info,
        element(0xA3, &sequence(&extensions)),
    ]);
    let signature = key
        .sign(sha256_pkcs1v15(), &Sha256::digest(&tbs))
        .expect("a 2048-bit key signs a SHA-256 digest");
    sequence(&[tbs, algorithm(SHA256_WITH_RSA), bits(&signature)])
}

/// RSASSA-PKCS1-v1_5 with SHA-256.
fn sha256_pkcs1v15() -> Pkcs1v15Sign {
    Pkcs1v15Sign {
        hash_len: Some(32),
        prefix: SHA256_DIGEST_INFO.into(),
    }
}

#[cfg(test)]
pub(super) mod tests {
    use std::sync::LazyLock;

    use rsa::{RsaPublicKey, pkcs8::DecodePublicKey};

    use super::*;
    use crate::{
        keys::Key,
        secmon::crypto::{Crypto, kek_option},
    };

    pub const DEVICE_ID: u64 = 0x0012_3456_789A_BCDE;

    /// Generating a key takes a while; the tests share one.
    static IDENTITY: LazyLock<SslIdentity> = LazyLock::new(|| SslIdentity::generate(DEVICE_ID));

    pub fn identity() -> &'static SslIdentity {
        &IDENTITY
    }

    /// Made-up keys, enough for the calibration and the SSL key.
    pub fn key_file() -> String {
        let mut text = String::new();
        for (index, name) in [
            "device_key",
            "retail_specific_aes_key_source",
            "aes_kek_generation_source",
            "aes_key_generation_source",
            "bis_kek_source",
            "master_key_00",
            "rsa_private_kek_generation_source",
            "ssl_rsa_kekek_source",
            "ssl_rsa_kek_source",
        ]
        .into_iter()
        .enumerate()
        {
            text += &format!("{name} = {:032x}\n", 0x0765_4321 * (index as u128 + 3));
        }
        for index in 0..3 {
            text += &format!("bis_key_source_{index:02x} = {:064x}\n", index + 0x5A5A);
        }
        text
    }

    /// Reads one DER element off the front of `data`: tag, content, rest.
    fn read(data: &[u8]) -> (u8, &[u8], &[u8]) {
        let (tag, first) = (data[0], data[1] as usize);
        let (len, start) = if first < 0x80 {
            (first, 2)
        } else {
            let count = first & 0x7F;
            let len = data[2..2 + count]
                .iter()
                .fold(0, |len, &byte| len << 8 | byte as usize);
            (len, 2 + count)
        };
        (tag, &data[start..start + len], &data[start + len..])
    }

    /// The elements of a constructed element's content.
    fn elements(mut content: &[u8]) -> Vec<(u8, &[u8])> {
        let mut found = Vec::new();
        while !content.is_empty() {
            let (tag, inner, rest) = read(content);
            found.push((tag, inner));
            content = rest;
        }
        found
    }

    #[test]
    fn the_certificate_is_a_self_signed_x509_certificate_of_the_key() {
        let identity = identity();
        let (tag, content, rest) = read(identity.certificate());
        assert_eq!((tag, rest.len()), (0x30, 0));
        let outer = elements(content);
        assert_eq!(outer.len(), 3);
        // The signed part, as it is encoded: the first element's bytes.
        let tbs = &content[..content.len() - read(content).2.len()];
        assert_eq!(outer[1].1, &algorithm(SHA256_WITH_RSA)[2..]);

        let fields = elements(outer[0].1);
        assert_eq!(fields.len(), 8);
        assert_eq!(fields[0], (0xA0, &[2, 1, 2][..]));
        let text = |field: &[u8]| String::from_utf8_lossy(field).into_owned();
        assert!(text(fields[3].1).contains("horizon-emu test CA"));
        assert!(text(fields[5].1).contains("00123456789ABCDE"));
        assert!(text(fields[5].1).contains("not issued by Nintendo"));

        // The public key is the key's, and it signed the certificate.
        let spki = element(0x30, fields[6].1);
        let public = RsaPublicKey::from_public_key_der(&spki).unwrap();
        assert_eq!(public, identity.key().to_public_key());
        assert_eq!(public.e(), &BigUint::from(65537u32));
        assert_eq!(public.n().bits(), 2048);
        let signature = &outer[2].1[1..];
        public
            .verify(sha256_pkcs1v15(), &Sha256::digest(tbs), signature)
            .unwrap();
        assert!(identity.certificate().len() <= CERTIFICATE_CAPACITY);
    }

    #[test]
    fn the_identity_follows_from_the_device_id() {
        let again = SslIdentity::generate(DEVICE_ID);
        assert_eq!(again.certificate(), identity().certificate());
        assert_eq!(again.private_exponent(), identity().private_exponent());
    }

    #[test]
    fn the_monitor_unwraps_the_key_for_ssl_and_it_matches_the_certificate() {
        let keys = KeySet::parse(&key_file()).unwrap();
        let identity = identity();
        let data = identity.wrapped_key(&keys, DEVICE_ID).unwrap();
        assert_eq!(data[0x130..], KEY_GENERATION.to_le_bytes());

        // What `ssl` and `spl` do with it: an access key for the kek
        // source, then the data before the generation.
        let crypto = Crypto::new(keys.clone(), [0x5A; 16]);
        let source = |name: Option<&Key>| *name.unwrap();
        let access_key = crypto
            .generate_aes_kek(
                &source(keys.ssl_rsa_kekek_source()),
                KEY_GENERATION.into(),
                kek_option(SEAL_DECRYPT_DEVICE_UNIQUE_DATA),
            )
            .unwrap();
        let mut wrapped = data[..0x130].to_vec();
        crypto
            .decrypt_device_unique_data(
                &access_key,
                &source(keys.ssl_rsa_kek_source()),
                &mut wrapped,
                DEVICE_ID,
            )
            .unwrap();
        let exponent = BigUint::from_bytes_be(&wrapped[..0x100]);
        assert_eq!(&exponent, identity.key().d());

        // The exponent with the certificate's modulus is a working key pair.
        let n = identity.key().n();
        let message = BigUint::from_bytes_be(b"a message to sign");
        let signed = message.modpow(&exponent, n);
        assert_eq!(signed.modpow(identity.key().e(), n), message);

        // Another console cannot unwrap it.
        let mut wrapped = data[..0x130].to_vec();
        assert!(
            crypto
                .decrypt_device_unique_data(
                    &access_key,
                    &source(keys.ssl_rsa_kek_source()),
                    &mut wrapped,
                    DEVICE_ID + 1,
                )
                .is_err()
        );
    }

    #[test]
    fn wrapping_names_the_missing_key() {
        let keys = KeySet::parse(&key_file().replace("ssl_rsa_kek_source", "x")).unwrap();
        assert!(matches!(
            identity().wrapped_key(&keys, DEVICE_ID),
            Err(NandError::SslKey(what)) if what.starts_with("ssl_rsa_kek_source")
        ));
    }

    #[test]
    fn der_lengths_take_the_long_form_from_128_bytes() {
        assert_eq!(element(4, &[7; 0x7F])[..2], [4, 0x7F]);
        assert_eq!(element(4, &[7; 0x80])[..3], [4, 0x81, 0x80]);
        assert_eq!(element(4, &[7; 0x123])[..4], [4, 0x82, 0x01, 0x23]);
        assert_eq!(integer(&[0, 0, 0x80]), [2, 2, 0, 0x80]);
        assert_eq!(integer(&[0, 0]), [2, 1, 0]);
    }
}
