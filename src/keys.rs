//! Console key material, read from a key file of `name = hex` lines.
//!
//! Kept are the keys the secure monitor's own derivations start from, and
//! the source values with which the filesystem process has the monitor
//! derive the keys of the built-in storage: making a storage image on the
//! host ([`crate::nand`]) means walking the same ladder with the same
//! inputs. Everything else in such a file is derived by the guest.

use std::{collections::BTreeMap, fmt, io, path::Path};

pub type Key = [u8; 16];
/// Source of the two halves, data key then tweak key, of an AES-XTS key.
pub type KeyPair = [u8; 32];

/// Lines whose value seeds a monitor derivation.
const AES_KEK_GENERATION_SOURCE: &str = "aes_kek_generation_source";
const MASTER_KEY_PREFIX: &str = "master_key_";
const DEVICE_KEY: &str = "device_key";
/// The secure boot key (SBK), the console-unique key in the private key
/// fuses the boot ROM loads into the security engine.
const SECURE_BOOT_KEY: &str = "secure_boot_key";
const RETAIL_SPECIFIC_AES_KEY_SOURCE: &str = "retail_specific_aes_key_source";
/// Lines whose value `ssl` hands the monitor to unwrap the console's TLS
/// client key: the kek source (`ssl` data `0x2633E0`) and the key source of
/// a retail console (`ssl` text `0x36080`).
const SSL_RSA_KEKEK_SOURCE: &str = "ssl_rsa_kekek_source";
const SSL_RSA_KEK_SOURCE: &str = "ssl_rsa_kek_source";
/// Lines whose value seeds the derivation of a built-in storage key.
const AES_KEY_GENERATION_SOURCE: &str = "aes_key_generation_source";
const BIS_KEK_SOURCE: &str = "bis_kek_source";
const BIS_KEY_SOURCE_PREFIX: &str = "bis_key_source_";
/// `bis_key_source_00` to `_02`; the fourth storage key shares the source
/// of the third.
pub const BIS_KEY_SOURCE_COUNT: usize = 3;
/// Lines whose value seeds the derivation of the keys of content archive
/// headers and key areas, and of save data headers.
const HEADER_KEK_SOURCE: &str = "header_kek_source";
const HEADER_KEY_SOURCE: &str = "header_key_source";
/// By the key area kek index a content archive names.
pub const KEY_AREA_KEY_SOURCES: [&str; 3] = [
    "key_area_key_application_source",
    "key_area_key_ocean_source",
    "key_area_key_system_source",
];
const SAVE_MAC_KEK_SOURCE: &str = "save_mac_kek_source";
/// A key file holds two; this one keys the CMAC of the system saves the
/// filesystem process wrote itself, the other does not.
const SAVE_MAC_KEY_SOURCE: &str = "save_mac_key_source_00";
/// The same value under the name key files gave it before there were two
/// (seen to be equal to `save_mac_key_source_00` in two key files).
const SAVE_MAC_KEY_SOURCE_UNNUMBERED: &str = "save_mac_key_source";
/// The monitor's source of the title key's ES common key: decrypted with a
/// master key it gives that generation's `titlekek`.
const TITLEKEK_SOURCE: &str = "titlekek_source";
/// The monitor's source of the package2 key: decrypted with the master key
/// of the generation the package2 header names, it gives `package2_key_XX`.
const PACKAGE2_KEY_SOURCE: &str = "package2_key_source";
/// What the TSEC firmware of package1 leaves in the security engine and
/// decrypts package1 with, as published key sets name the values it
/// derives from the TSEC's hardware secrets (see [`crate::soc::tsec`]):
/// the console-unique `tsec_key`, and `tsec_root_key_XX`,
/// `package1_key_XX` and `package1_mac_key_XX` by TSEC firmware revision
/// and key generation (the two hex digits).
const TSEC_KEY: &str = "tsec_key";
const TSEC_ROOT_KEY_PREFIX: &str = "tsec_root_key_";
const PACKAGE1_KEY_PREFIX: &str = "package1_key_";
const PACKAGE1_MAC_KEY_PREFIX: &str = "package1_mac_key_";

#[derive(Debug)]
pub enum KeyError {
    Io(io::Error),
    /// A key that is kept has a value that is not the hex digits of its
    /// size.
    Malformed {
        name: String,
    },
}

impl fmt::Display for KeyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "cannot read key file: {error}"),
            Self::Malformed { name } => {
                write!(
                    f,
                    "key file: {name} is not a hex value of the size of that key"
                )
            }
        }
    }
}

impl std::error::Error for KeyError {}

#[derive(Clone, Default)]
pub struct KeySet {
    /// By key generation (the two hex digits of `master_key_XX`).
    master_keys: BTreeMap<u8, Key>,
    aes_kek_generation_source: Option<Key>,
    device_key: Option<Key>,
    retail_specific_aes_key_source: Option<Key>,
    ssl_rsa_kekek_source: Option<Key>,
    ssl_rsa_kek_source: Option<Key>,
    aes_key_generation_source: Option<Key>,
    bis_kek_source: Option<Key>,
    bis_key_sources: [Option<KeyPair>; BIS_KEY_SOURCE_COUNT],
    header_kek_source: Option<Key>,
    header_key_source: Option<KeyPair>,
    key_area_key_sources: [Option<Key>; KEY_AREA_KEY_SOURCES.len()],
    save_mac_kek_source: Option<Key>,
    save_mac_key_source: Option<Key>,
    titlekek_source: Option<Key>,
    package2_key_source: Option<Key>,
    tsec_key: Option<Key>,
    secure_boot_key: Option<Key>,
    /// By the two hex digits of each name.
    tsec_root_keys: BTreeMap<u8, Key>,
    package1_keys: BTreeMap<u8, Key>,
    package1_mac_keys: BTreeMap<u8, Key>,
}

/// Key values stay out of logs.
impl fmt::Debug for KeySet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KeySet")
            .field("master_key_generations", &self.master_keys.keys())
            .field(
                "aes_kek_generation_source",
                &self.aes_kek_generation_source.is_some(),
            )
            .field("device_key", &self.device_key.is_some())
            .field(
                "retail_specific_aes_key_source",
                &self.retail_specific_aes_key_source.is_some(),
            )
            .field("ssl_rsa_kekek_source", &self.ssl_rsa_kekek_source.is_some())
            .field("ssl_rsa_kek_source", &self.ssl_rsa_kek_source.is_some())
            .field(
                "aes_key_generation_source",
                &self.aes_key_generation_source.is_some(),
            )
            .field("bis_kek_source", &self.bis_kek_source.is_some())
            .field(
                "bis_key_sources",
                &self.bis_key_sources.each_ref().map(Option::is_some),
            )
            .field("header_kek_source", &self.header_kek_source.is_some())
            .field("header_key_source", &self.header_key_source.is_some())
            .field(
                "key_area_key_sources",
                &self.key_area_key_sources.each_ref().map(Option::is_some),
            )
            .field("save_mac_kek_source", &self.save_mac_kek_source.is_some())
            .field("save_mac_key_source", &self.save_mac_key_source.is_some())
            .field("titlekek_source", &self.titlekek_source.is_some())
            .field("package2_key_source", &self.package2_key_source.is_some())
            .field("tsec_key", &self.tsec_key.is_some())
            .field("secure_boot_key", &self.secure_boot_key.is_some())
            .field("tsec_root_key_generations", &self.tsec_root_keys.keys())
            .field("package1_key_generations", &self.package1_keys.keys())
            .field(
                "package1_mac_key_generations",
                &self.package1_mac_keys.keys(),
            )
            .finish()
    }
}

impl KeySet {
    pub fn load(path: &Path) -> Result<Self, KeyError> {
        Self::parse(&std::fs::read_to_string(path).map_err(KeyError::Io)?)
    }

    pub fn parse(text: &str) -> Result<Self, KeyError> {
        let mut keys = Self::default();
        for line in text.lines() {
            let Some((name, value)) = line.split_once('=') else {
                continue;
            };
            let (name, value) = (name.trim(), value.trim());
            let malformed = || KeyError::Malformed { name: name.into() };
            let key = || parse_hex::<16>(value).ok_or_else(malformed);
            if name == AES_KEK_GENERATION_SOURCE {
                keys.aes_kek_generation_source = Some(key()?);
            } else if name == DEVICE_KEY {
                keys.device_key = Some(key()?);
            } else if name == RETAIL_SPECIFIC_AES_KEY_SOURCE {
                keys.retail_specific_aes_key_source = Some(key()?);
            } else if name == SSL_RSA_KEKEK_SOURCE {
                keys.ssl_rsa_kekek_source = Some(key()?);
            } else if name == SSL_RSA_KEK_SOURCE {
                keys.ssl_rsa_kek_source = Some(key()?);
            } else if name == AES_KEY_GENERATION_SOURCE {
                keys.aes_key_generation_source = Some(key()?);
            } else if name == BIS_KEK_SOURCE {
                keys.bis_kek_source = Some(key()?);
            } else if let Some(source) = name
                .strip_prefix(BIS_KEY_SOURCE_PREFIX)
                .and_then(|digits| usize::from_str_radix(digits, 16).ok())
                .and_then(|index| keys.bis_key_sources.get_mut(index))
            {
                *source = Some(parse_hex::<32>(value).ok_or_else(malformed)?);
            } else if name == HEADER_KEK_SOURCE {
                keys.header_kek_source = Some(key()?);
            } else if name == HEADER_KEY_SOURCE {
                keys.header_key_source = Some(parse_hex::<32>(value).ok_or_else(malformed)?);
            } else if let Some(index) = KEY_AREA_KEY_SOURCES.iter().position(|&known| known == name)
            {
                keys.key_area_key_sources[index] = Some(key()?);
            } else if name == SAVE_MAC_KEK_SOURCE {
                keys.save_mac_kek_source = Some(key()?);
            } else if name == SAVE_MAC_KEY_SOURCE || name == SAVE_MAC_KEY_SOURCE_UNNUMBERED {
                keys.save_mac_key_source = Some(key()?);
            } else if name == TITLEKEK_SOURCE {
                keys.titlekek_source = Some(key()?);
            } else if name == PACKAGE2_KEY_SOURCE {
                keys.package2_key_source = Some(key()?);
            } else if name == TSEC_KEY {
                keys.tsec_key = Some(key()?);
            } else if name == SECURE_BOOT_KEY {
                keys.secure_boot_key = Some(key()?);
            } else if let Some((map, generation)) = [
                (TSEC_ROOT_KEY_PREFIX, &mut keys.tsec_root_keys),
                (PACKAGE1_KEY_PREFIX, &mut keys.package1_keys),
                (PACKAGE1_MAC_KEY_PREFIX, &mut keys.package1_mac_keys),
                (MASTER_KEY_PREFIX, &mut keys.master_keys),
            ]
            .into_iter()
            .find_map(|(prefix, map)| {
                let digits = name.strip_prefix(prefix)?;
                Some((map, u8::from_str_radix(digits, 16).ok()?))
            }) {
                map.insert(generation, key()?);
            }
        }
        Ok(keys)
    }

    pub fn master_key(&self, generation: u8) -> Option<&Key> {
        self.master_keys.get(&generation)
    }

    /// The newest master key generation present: the key generation of the
    /// console these keys describe.
    pub fn newest_generation(&self) -> Option<u8> {
        self.master_keys.keys().next_back().copied()
    }

    pub fn aes_kek_generation_source(&self) -> Option<&Key> {
        self.aes_kek_generation_source.as_ref()
    }

    /// The console-unique key of the first key generation.
    pub fn device_key(&self) -> Option<&Key> {
        self.device_key.as_ref()
    }

    /// Seed of the keys the monitor hands out in the clear on retail
    /// consoles (`GenerateSpecificAesKey`).
    pub fn retail_specific_aes_key_source(&self) -> Option<&Key> {
        self.retail_specific_aes_key_source.as_ref()
    }

    /// The kek source `ssl` asks the monitor for when it unwraps the TLS
    /// client key of key generation 0.
    pub fn ssl_rsa_kekek_source(&self) -> Option<&Key> {
        self.ssl_rsa_kekek_source.as_ref()
    }

    /// The key source `ssl` unwraps that key with on a retail console.
    pub fn ssl_rsa_kek_source(&self) -> Option<&Key> {
        self.ssl_rsa_kek_source.as_ref()
    }

    /// Constant of `spl` with which it turns a key encryption key from the
    /// monitor into the key that decrypts a source (`GenerateAesKey`).
    pub fn aes_key_generation_source(&self) -> Option<&Key> {
        self.aes_key_generation_source.as_ref()
    }

    /// Source of the key encryption key of the built-in storage keys
    /// after the first.
    pub fn bis_kek_source(&self) -> Option<&Key> {
        self.bis_kek_source.as_ref()
    }

    /// `bis_key_source_XX`.
    pub fn bis_key_source(&self, index: usize) -> Option<&KeyPair> {
        self.bis_key_sources.get(index)?.as_ref()
    }

    /// Source of the key encryption key of content archive headers.
    pub fn header_kek_source(&self) -> Option<&Key> {
        self.header_kek_source.as_ref()
    }

    /// The content archive header key pair, sealed.
    pub fn header_key_source(&self) -> Option<&KeyPair> {
        self.header_key_source.as_ref()
    }

    /// Source of the key encryption key of content archive key areas of
    /// kek index `index` (0 application, 1 ocean, 2 system), the line
    /// [`KEY_AREA_KEY_SOURCES`]`[index]`.
    pub fn key_area_key_source(&self, index: usize) -> Option<&Key> {
        self.key_area_key_sources.get(index)?.as_ref()
    }

    /// Source of the key encryption key of the save data MAC key.
    pub fn save_mac_kek_source(&self) -> Option<&Key> {
        self.save_mac_kek_source.as_ref()
    }

    /// `save_mac_key_source_00`.
    pub fn save_mac_key_source(&self) -> Option<&Key> {
        self.save_mac_key_source.as_ref()
    }

    /// `titlekek_source`.
    pub fn titlekek_source(&self) -> Option<&Key> {
        self.titlekek_source.as_ref()
    }

    /// `package2_key_source`.
    pub fn package2_key_source(&self) -> Option<&Key> {
        self.package2_key_source.as_ref()
    }

    /// `tsec_key`: the console-unique key the TSEC firmware generates.
    pub fn tsec_key(&self) -> Option<&Key> {
        self.tsec_key.as_ref()
    }

    /// `secure_boot_key`: the SBK of this console's fuses.
    pub fn secure_boot_key(&self) -> Option<&Key> {
        self.secure_boot_key.as_ref()
    }

    /// `tsec_root_key_XX`.
    pub fn tsec_root_key(&self, revision: u8) -> Option<&Key> {
        self.tsec_root_keys.get(&revision)
    }

    /// `package1_key_XX`.
    pub fn package1_key(&self, generation: u8) -> Option<&Key> {
        self.package1_keys.get(&generation)
    }

    /// `package1_mac_key_XX`.
    pub fn package1_mac_key(&self, generation: u8) -> Option<&Key> {
        self.package1_mac_keys.get(&generation)
    }
}

fn parse_hex<const N: usize>(hex: &str) -> Option<[u8; N]> {
    let (pairs, rest) = hex.as_bytes().as_chunks::<2>();
    if pairs.len() != N || !rest.is_empty() {
        return None;
    }
    let mut value = [0; N];
    for (byte, pair) in value.iter_mut().zip(pairs) {
        *byte = u8::from_str_radix(std::str::from_utf8(pair).ok()?, 16).ok()?;
    }
    Some(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_the_monitor_keys_and_ignores_the_rest() {
        let keys = KeySet::parse(
            "aes_kek_generation_source = 000102030405060708090a0b0c0d0e0f\n\
             master_key_00 = ffeeddccbbaa99887766554433221100\n\
             master_key_1a=0123456789abcdef0123456789ABCDEF\n\
             package2_key_source = 5566778899aabbccddeeff0011223344\n\
             header_key = 00112233\n\
             ; comment\n",
        )
        .unwrap();
        assert_eq!(keys.aes_kek_generation_source().unwrap()[15], 0x0F);
        assert_eq!(keys.master_key(0).unwrap()[0], 0xFF);
        assert_eq!(keys.master_key(0x1A).unwrap()[7], 0xEF);
        assert!(keys.master_key(1).is_none());
        assert_eq!(keys.package2_key_source().unwrap()[0], 0x55);
        assert!(!format!("{keys:?}").contains("ffee"));
        assert!(!format!("{keys:?}").contains("5566"));
    }

    #[test]
    fn keeps_the_tsec_and_package1_keys_by_generation() {
        let keys = KeySet::parse(
            "tsec_key = 00112233445566778899aabbccddeeff\n\
             tsec_root_key_02 = 102132435465768798a9bacbdcedfe0f\n\
             tsec_root_kek_02 = ffffffffffffffffffffffffffffffff\n\
             package1_key_08 = 2233445566778899aabbccddeeff0011\n\
             package1_mac_key_08 = 33445566778899aabbccddeeff001122\n\
             package1_kek_02 = eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee\n",
        )
        .unwrap();
        assert_eq!(keys.tsec_key().unwrap()[1], 0x11);
        assert!(keys.secure_boot_key().is_none());
        let sbk = KeySet::parse("secure_boot_key = 0f0e0d0c0b0a09080706050403020100\n").unwrap();
        assert_eq!(sbk.secure_boot_key().unwrap()[0], 0x0F);
        assert!(!format!("{sbk:?}").contains("0f0e"));
        assert_eq!(keys.tsec_root_key(2).unwrap()[0], 0x10);
        assert!(keys.tsec_root_key(0).is_none());
        assert_eq!(keys.package1_key(8).unwrap()[0], 0x22);
        assert_eq!(keys.package1_mac_key(8).unwrap()[0], 0x33);
        assert!(keys.package1_key(2).is_none());
        let debug = format!("{keys:?}");
        for value in ["0011", "1021", "2233", "3344"] {
            assert!(!debug.contains(value));
        }
    }

    #[test]
    fn a_needed_key_of_the_wrong_length_is_an_error_naming_it() {
        let error = KeySet::parse("master_key_03 = abcd\n").unwrap_err();
        assert!(matches!(error, KeyError::Malformed { name } if name == "master_key_03"));
    }

    #[test]
    fn storage_key_sources_are_pairs_and_only_three_exist() {
        let pair = "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f";
        let keys = KeySet::parse(&format!(
            "bis_key_source_02 = {pair}\nbis_key_source_03 = {pair}\n"
        ))
        .unwrap();
        assert_eq!(keys.bis_key_source(2).unwrap()[31], 0x1F);
        assert!(keys.bis_key_source(0).is_none());
        assert!(keys.bis_key_source(3).is_none());
        let error = KeySet::parse(&format!("bis_key_source_00 = {}\n", &pair[..32])).unwrap_err();
        assert!(matches!(error, KeyError::Malformed { name } if name == "bis_key_source_00"));
    }

    #[test]
    fn the_header_key_source_is_a_pair_and_only_the_first_save_mac_source_is_kept() {
        let pair = "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f";
        let keys = KeySet::parse(&format!(
            "header_key_source = {pair}\n\
             key_area_key_system_source = {}\n\
             save_mac_key_source_01 = {}\n",
            &pair[..32],
            &pair[32..]
        ))
        .unwrap();
        assert_eq!(keys.header_key_source().unwrap()[31], 0x1F);
        assert_eq!(keys.key_area_key_source(2).unwrap()[0], 0x00);
        assert!(keys.key_area_key_source(0).is_none());
        assert!(keys.save_mac_key_source().is_none());
        let error = KeySet::parse(&format!("header_key_source = {}\n", &pair[..32])).unwrap_err();
        assert!(matches!(error, KeyError::Malformed { name } if name == "header_key_source"));
    }
}
