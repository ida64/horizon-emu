//! Keys and cipher of the built-in storage (BIS).
//!
//! The filesystem process does not derive its storage keys itself: it has
//! the monitor do it, through `spl`. Its routine at text offset `0xe15e0`
//! takes one of four key IDs and produces the two halves of an AES-XTS key
//! from the sources its start-up code (`0x28c00`) gathers, which are the
//! `bis_kek_source` and `bis_key_source_XX` of a key file:
//!
//! ```text
//! key 0:    half = GenerateSpecificAesKey(source_00 half, generation, half index)
//!                = D(D(device_key, retail_specific_aes_key_source), half)
//! keys 1-3: kek  = GenerateAesKek(bis_kek_source, generation, device unique)
//!                = D(D(device_key, aes_kek_generation_source), bis_kek_source)
//!           half = GenerateAesKey(kek, source half)
//!                = D(D(kek, aes_key_generation_source), half)
//! ```
//!
//! with `D` one AES-128 block decryption, the generation the monitor's
//! device-unique key generation (this machine's monitor reports the first)
//! and `aes_key_generation_source` the constant with which `spl` turns a
//! kek into the key that decrypts a source (it is in `spl`'s image at offset
//! `0x1a313`). The start-up code gives keys 2 and 3 the same source: key 3
//! differs from key 2 only on a recovery boot, where its kek is requested
//! with another option. The ladder reproduces the four `bis_key_XX` values
//! published alongside a device key.
//!
//! Which partition is under which key is the table at offset `0x1e3b80` of
//! the image, indexed by partition ID from 27: calibration data and
//! calibration files key 0, safe mode key 1, user key 2, system key 3.
//!
//! The cipher (`0xdfb30` creating the storage, `0x13f950` reading it) is
//! AES-XTS over units of [`CRYPTO_SECTOR`] bytes counted from the start of
//! the partition. The tweak of a unit is its index added to an all-zero
//! initial counter as a big-endian number (`0x17d1e0`), where the XTS
//! standard encodes it little-endian; content archive headers use the same
//! variant with 512-byte units.

use std::fmt;

use aes::{
    Aes128, Block,
    cipher::{BlockDecrypt, BlockEncrypt, KeyInit},
};

use crate::keys::{BIS_KEY_SOURCE_COUNT, Key, KeyPair, KeySet};

/// Bytes sharing one tweak on the built-in storage.
pub const CRYPTO_SECTOR: u64 = 0x4000;

const BLOCK: usize = 16;
/// Blocks handed to the cipher at once, so that it can run them in parallel.
const BATCH: usize = 64;

/// The key file lacks an entry the derivation needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MissingKey(pub &'static str);

impl fmt::Display for MissingKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} is not in the key file", self.0)
    }
}

impl std::error::Error for MissingKey {}

/// The four storage keys of one console, by the filesystem process's key ID.
#[derive(Clone)]
pub struct BisKeys([KeyPair; 4]);

/// Key ID of the calibration partitions.
pub const KEY_CALIBRATION: usize = 0;
pub const KEY_SAFE: usize = 1;
pub const KEY_USER: usize = 2;
pub const KEY_SYSTEM: usize = 3;

fn decrypt(key: &Key, block: &Key) -> Key {
    let mut block = (*block).into();
    Aes128::new(key.into()).decrypt_block(&mut block);
    block.into()
}

fn decrypt_halves(key: &Key, source: &KeyPair) -> KeyPair {
    let (halves, _) = source.as_chunks::<BLOCK>();
    let mut pair = [0; 32];
    for (out, half) in pair.as_chunks_mut::<BLOCK>().0.iter_mut().zip(halves) {
        *out = decrypt(key, half);
    }
    pair
}

impl BisKeys {
    /// Walks the ladder of the module documentation for the first
    /// device-unique key generation.
    pub fn derive(keys: &KeySet) -> Result<Self, MissingKey> {
        let device_key = keys.device_key().ok_or(MissingKey("device_key"))?;
        let specific_source = keys
            .retail_specific_aes_key_source()
            .ok_or(MissingKey("retail_specific_aes_key_source"))?;
        let kek_generation_source = keys
            .aes_kek_generation_source()
            .ok_or(MissingKey("aes_kek_generation_source"))?;
        let key_generation_source = keys
            .aes_key_generation_source()
            .ok_or(MissingKey("aes_key_generation_source"))?;
        let kek_source = keys.bis_kek_source().ok_or(MissingKey("bis_kek_source"))?;
        const SOURCE_NAMES: [&str; BIS_KEY_SOURCE_COUNT] = [
            "bis_key_source_00",
            "bis_key_source_01",
            "bis_key_source_02",
        ];
        let source = |index: usize| {
            keys.bis_key_source(index)
                .ok_or(MissingKey(SOURCE_NAMES[index]))
        };

        let calibration = decrypt_halves(&decrypt(device_key, specific_source), source(0)?);
        let kek = decrypt(&decrypt(device_key, kek_generation_source), kek_source);
        let key = decrypt(&kek, key_generation_source);
        let safe = decrypt_halves(&key, source(1)?);
        let user = decrypt_halves(&key, source(2)?);
        Ok(Self([calibration, safe, user, user]))
    }

    /// Cipher of a partition under key `id`.
    pub fn cipher(&self, id: usize) -> Xts {
        Xts::new(&self.0[id], CRYPTO_SECTOR)
    }
}

/// AES-128-XTS with the sector number big-endian in the tweak.
pub struct Xts {
    data: Aes128,
    tweak: Aes128,
    sector_size: u64,
}

#[derive(Clone, Copy)]
enum Direction {
    Encrypt,
    Decrypt,
}

/// Next tweak of a sector: times x in GF(2^128), the block read as a
/// little-endian number.
fn double(tweak: u128) -> u128 {
    tweak << 1 ^ (0x87 & (tweak >> 127).wrapping_neg())
}

impl Xts {
    /// `key` is the data key followed by the tweak key; `sector_size` a
    /// multiple of the block size.
    pub fn new(key: &KeyPair, sector_size: u64) -> Self {
        assert!(sector_size != 0 && sector_size.is_multiple_of(BLOCK as u64));
        let ([data, tweak], _) = key.as_chunks::<BLOCK>() else {
            unreachable!("a key pair is two keys")
        };
        Self {
            data: Aes128::new(data.into()),
            tweak: Aes128::new(tweak.into()),
            sector_size,
        }
    }

    /// Encrypts `data`, which lies `offset` bytes into the storage, in
    /// place. Blocks do not depend on their neighbours, so any range of
    /// whole blocks will do.
    pub fn encrypt(&self, offset: u64, data: &mut [u8]) {
        self.crypt(offset, data, Direction::Encrypt);
    }

    /// Inverse of [`Self::encrypt`].
    pub fn decrypt(&self, offset: u64, data: &mut [u8]) {
        self.crypt(offset, data, Direction::Decrypt);
    }

    fn crypt(&self, mut offset: u64, mut data: &mut [u8], direction: Direction) {
        assert!(
            offset.is_multiple_of(BLOCK as u64) && data.len().is_multiple_of(BLOCK),
            "XTS works on whole blocks"
        );
        while !data.is_empty() {
            let within = offset % self.sector_size;
            let length = usize::try_from(self.sector_size - within)
                .map_or(data.len(), |rest| rest.min(data.len()));
            let (now, later) = data.split_at_mut(length);
            let mut first = Block::from(u128::from(offset / self.sector_size).to_be_bytes());
            self.tweak.encrypt_block(&mut first);
            let mut tweak = u128::from_le_bytes(first.into());
            for _ in 0..within / BLOCK as u64 {
                tweak = double(tweak);
            }
            for batch in now.chunks_mut(BATCH * BLOCK) {
                let mut tweaks = [0; BATCH];
                let mut blocks = [Block::default(); BATCH];
                let (plain, _) = batch.as_chunks_mut::<BLOCK>();
                for ((block, used), bytes) in blocks.iter_mut().zip(&mut tweaks).zip(&*plain) {
                    *used = tweak;
                    *block = (u128::from_le_bytes(*bytes) ^ tweak).to_le_bytes().into();
                    tweak = double(tweak);
                }
                let blocks = &mut blocks[..plain.len()];
                match direction {
                    Direction::Encrypt => self.data.encrypt_blocks(blocks),
                    Direction::Decrypt => self.data.decrypt_blocks(blocks),
                }
                for ((block, tweak), bytes) in blocks.iter().zip(tweaks).zip(plain) {
                    *bytes = (u128::from_le_bytes((*block).into()) ^ tweak).to_le_bytes();
                }
            }
            offset += length as u64;
            data = later;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex<const N: usize>(digits: &str) -> [u8; N] {
        core::array::from_fn(|index| {
            u8::from_str_radix(&digits[index * 2..index * 2 + 2], 16).unwrap()
        })
    }

    fn test_key() -> KeyPair {
        core::array::from_fn(|index| index as u8)
    }

    fn plain() -> Vec<u8> {
        (0..0x8000u32).map(|index| (index * 7 + 3) as u8).collect()
    }

    /// The expected values come from the AES-XTS of the Python
    /// `cryptography` package, one call per sector with the sector number
    /// as a 16-byte big-endian tweak.
    #[test]
    fn agrees_with_an_independent_xts_given_big_endian_sector_tweaks() {
        let xts = Xts::new(&test_key(), CRYPTO_SECTOR);
        let mut data = plain();
        xts.encrypt(0, &mut data);
        assert_eq!(data[..16], hex::<16>("e9e4c0294045c8e699038ed38c1492e8"));
        assert_eq!(
            data[0x4000..0x4010],
            hex::<16>("6a06992f802b9bc9c620c0c8a9111dba")
        );
        assert_eq!(
            data[0x7FF0..],
            hex::<16>("b35952a7c8b85b14839b16e3c67dc6ef")
        );

        // A sector number that fills more than the last byte of the tweak
        // tells big-endian from little-endian beyond doubt.
        let mut data = plain()[..32].to_vec();
        xts.encrypt(0x01_0203_0405 * CRYPTO_SECTOR, &mut data);
        assert_eq!(
            data,
            hex::<32>("877dfc1b6b850605ee39c2eb929f15ef66e1fba01aff47a9bf4c954cdc1e743a")
        );
    }

    #[test]
    fn any_range_of_blocks_round_trips_and_matches_the_whole() {
        let xts = Xts::new(&test_key(), CRYPTO_SECTOR);
        let mut whole = plain();
        xts.encrypt(0, &mut whole);
        // A range that starts inside one sector and ends inside the next,
        // as a filesystem sector write does.
        let range = 0x3E00..0x4400;
        let mut part = plain()[range.clone()].to_vec();
        xts.encrypt(range.start as u64, &mut part);
        assert_eq!(part, whole[range]);

        xts.decrypt(0, &mut whole);
        assert_eq!(whole, plain());
    }

    #[test]
    fn derivation_names_the_missing_entry_and_gives_system_the_user_key() {
        let mut text = String::new();
        let mut missing = Vec::new();
        for (index, name) in [
            "device_key",
            "retail_specific_aes_key_source",
            "aes_kek_generation_source",
            "aes_key_generation_source",
            "bis_kek_source",
        ]
        .into_iter()
        .enumerate()
        {
            missing.push((name, text.clone()));
            text += &format!("{name} = {:032x}\n", 0x1111 * (index as u128 + 1));
        }
        for index in 0..3 {
            let name = [
                "bis_key_source_00",
                "bis_key_source_01",
                "bis_key_source_02",
            ][index];
            missing.push((name, text.clone()));
            text += &format!("bis_key_source_{index:02x} = {:064x}\n", index + 7);
        }
        for (name, text) in missing {
            let keys = KeySet::parse(&text).unwrap();
            assert_eq!(BisKeys::derive(&keys).err(), Some(MissingKey(name)));
        }

        let keys = BisKeys::derive(&KeySet::parse(&text).unwrap()).unwrap().0;
        assert_eq!(keys[KEY_USER], keys[KEY_SYSTEM]);
        assert_ne!(keys[KEY_CALIBRATION], keys[KEY_SAFE]);
        assert_ne!(keys[KEY_SAFE], keys[KEY_USER]);
    }
}
