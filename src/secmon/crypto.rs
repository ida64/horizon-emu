//! The monitor's AES key ladder, as userland drives it through `spl`.
//!
//! Userland never holds a key. It asks the monitor to derive a key
//! encryption key from a source value (`GenerateAesKek`), gets it back
//! sealed under a key only the monitor has, and hands that "access key" back
//! together with a second source to load the final key into a key slot
//! (`LoadAesKey`). In terms of the keys a key file holds:
//!
//! ```text
//! kek  = D(D(master_key[generation], aes_kek_generation_source), kek_source)
//! key  = D(kek, key_source)
//! ```
//!
//! with `D` a single AES-128 block decryption. This is the public
//! description of the ladder; it reproduces the derived keys published
//! alongside master keys (for example the key area keys).
//!
//! The sealing key is this machine's own. On hardware it is random per boot
//! and never leaves the monitor, so access keys are opaque to the guest
//! either way.

use aes::{
    Aes128,
    cipher::{BlockDecrypt, BlockEncrypt, KeyInit},
};

use crate::keys::{Key, KeySet};

/// Key slots userland may load keys into.
const USER_SLOT_COUNT: usize = 6;

/// `GenerateAesKek` option word: the kek is specific to this console.
const OPTION_DEVICE_UNIQUE: u32 = 1 << 0;
/// Which of the monitor's constants seeds the kek; 0 is the default one.
const OPTION_KEY_TYPE: u32 = 0xF << 1;
/// Which operation the access key will be valid for; 0 is `LoadAesKey`.
const OPTION_SEAL_KEY: u32 = 7 << 5;
const OPTION_RESERVED: u32 = !0xFF;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CryptoError {
    /// The monitor itself would refuse the arguments.
    InvalidArgument,
    /// The request needs something that is not modeled or not available.
    Unimplemented(&'static str),
}

/// What `ComputeAes` can do with the key in a slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CipherMode {
    CbcEncrypt,
    CbcDecrypt,
    /// Counter mode with a big-endian 128-bit counter.
    Ctr,
}

pub struct Crypto {
    keys: KeySet,
    seal_key: Key,
    slots: [Option<Key>; USER_SLOT_COUNT],
}

fn decrypt(key: &Key, block: &Key) -> Key {
    let mut block = (*block).into();
    Aes128::new(key.into()).decrypt_block(&mut block);
    block.into()
}

fn encrypt(key: &Key, block: &Key) -> Key {
    let mut block = (*block).into();
    Aes128::new(key.into()).encrypt_block(&mut block);
    block.into()
}

impl Crypto {
    /// `seal_key` wraps the access keys handed to userland.
    pub fn new(keys: KeySet, seal_key: Key) -> Self {
        Self {
            keys,
            seal_key,
            slots: [None; USER_SLOT_COUNT],
        }
    }

    /// Derives the kek for `source` and returns it sealed. `generation` is
    /// the caller's key generation, which counts from one (zero meaning the
    /// first as well).
    pub fn generate_aes_kek(
        &self,
        source: &Key,
        generation: u64,
        option: u32,
    ) -> Result<Key, CryptoError> {
        if option & OPTION_RESERVED != 0 {
            return Err(CryptoError::InvalidArgument);
        }
        if option & OPTION_KEY_TYPE != 0 {
            return Err(CryptoError::Unimplemented(
                "key encryption keys of a non-default type",
            ));
        }
        if option & OPTION_SEAL_KEY != 0 {
            return Err(CryptoError::Unimplemented(
                "access keys for operations other than LoadAesKey",
            ));
        }
        let generation =
            u8::try_from(generation.saturating_sub(1)).map_err(|_| CryptoError::InvalidArgument)?;
        // The monitor has no key of a generation newer than its own, and
        // says so; callers probe for the newest generation this way.
        if self
            .keys
            .newest_generation()
            .is_some_and(|newest| generation > newest)
        {
            return Err(CryptoError::InvalidArgument);
        }
        let root_key = if option & OPTION_DEVICE_UNIQUE == 0 {
            self.keys
                .master_key(generation)
                .ok_or(CryptoError::Unimplemented(
                    "the master key of the requested generation is not in the key file",
                ))?
        } else if generation == 0 {
            self.keys.device_key().ok_or(CryptoError::Unimplemented(
                "device_key is not in the key file",
            ))?
        } else {
            return Err(CryptoError::Unimplemented(
                "device-unique keys of generations after the first",
            ));
        };
        let static_source =
            self.keys
                .aes_kek_generation_source()
                .ok_or(CryptoError::Unimplemented(
                    "aes_kek_generation_source is not in the key file",
                ))?;
        let kek = decrypt(&decrypt(root_key, static_source), source);
        Ok(encrypt(&self.seal_key, &kek))
    }

    /// Loads `D(kek, source)` into `slot`, `kek` being what `access_key` seals.
    pub fn load_aes_key(
        &mut self,
        slot: u64,
        access_key: &Key,
        source: &Key,
    ) -> Result<(), CryptoError> {
        let slot = usize::try_from(slot)
            .ok()
            .and_then(|slot| self.slots.get_mut(slot))
            .ok_or(CryptoError::InvalidArgument)?;
        let kek = decrypt(&self.seal_key, access_key);
        *slot = Some(decrypt(&kek, source));
        Ok(())
    }

    /// Key loaded into `slot`, if any.
    pub fn slot(&self, slot: u64) -> Option<&Key> {
        self.slots.get(usize::try_from(slot).ok()?)?.as_ref()
    }

    /// `GenerateSpecificAesKey`: a console-unique key the caller gets in the
    /// clear, `D(D(device_key, retail_specific_aes_key_source), source)`.
    /// The filesystem process asks for the two halves of the calibration
    /// partition's key this way, as `which` 0 and 1; one ladder reproduces
    /// both halves of the partition key published alongside a device key.
    pub fn generate_specific_aes_key(
        &self,
        source: &Key,
        generation: u64,
        which: u64,
    ) -> Result<Key, CryptoError> {
        if which > 1 {
            return Err(CryptoError::Unimplemented(
                "specific keys beyond the two of the calibration partition",
            ));
        }
        if generation > 1 {
            return Err(CryptoError::Unimplemented(
                "device-unique keys of generations after the first",
            ));
        }
        let device_key = self.keys.device_key().ok_or(CryptoError::Unimplemented(
            "device_key is not in the key file",
        ))?;
        let static_source =
            self.keys
                .retail_specific_aes_key_source()
                .ok_or(CryptoError::Unimplemented(
                    "retail_specific_aes_key_source is not in the key file",
                ))?;
        Ok(decrypt(&decrypt(device_key, static_source), source))
    }

    /// Runs `data`, whole blocks, through the key in `slot`, in place.
    pub fn compute_aes(
        &self,
        slot: u64,
        mode: CipherMode,
        iv: &Key,
        data: &mut [u8],
    ) -> Result<(), CryptoError> {
        if usize::try_from(slot).map_or(true, |slot| slot >= USER_SLOT_COUNT)
            || !data.len().is_multiple_of(16)
        {
            return Err(CryptoError::InvalidArgument);
        }
        let key = self.slot(slot).ok_or(CryptoError::Unimplemented(
            "AES with a key slot nothing was loaded into",
        ))?;
        let cipher = Aes128::new(key.into());
        let mut chain = *iv;
        for block in data.as_chunks_mut::<16>().0 {
            match mode {
                CipherMode::CbcEncrypt => {
                    for (byte, previous) in block.iter_mut().zip(chain) {
                        *byte ^= previous;
                    }
                    cipher.encrypt_block(block.into());
                    chain = *block;
                }
                CipherMode::CbcDecrypt => {
                    let ciphertext = *block;
                    cipher.decrypt_block(block.into());
                    for (byte, previous) in block.iter_mut().zip(chain) {
                        *byte ^= previous;
                    }
                    chain = ciphertext;
                }
                CipherMode::Ctr => {
                    let mut pad = chain;
                    cipher.encrypt_block((&mut pad).into());
                    for (byte, pad) in block.iter_mut().zip(pad) {
                        *byte ^= pad;
                    }
                    chain = (u128::from_be_bytes(chain).wrapping_add(1)).to_be_bytes();
                }
            }
        }
        Ok(())
    }

    /// AES-CMAC of `data` under the key in `slot`.
    pub fn compute_cmac(&self, slot: u64, data: &[u8]) -> Result<Key, CryptoError> {
        if usize::try_from(slot).map_or(true, |slot| slot >= USER_SLOT_COUNT) {
            return Err(CryptoError::InvalidArgument);
        }
        let key = self.slot(slot).ok_or(CryptoError::Unimplemented(
            "a CMAC with a key slot nothing was loaded into",
        ))?;
        Ok(cmac(key, data))
    }
}

/// AES-CMAC as RFC 4493 defines it.
pub(crate) fn cmac(key: &Key, data: &[u8]) -> Key {
    /// Doubling in GF(2^128), which derives the two subkeys.
    fn double(block: Key) -> Key {
        let value = u128::from_be_bytes(block);
        ((value << 1) ^ (0x87 * (value >> 127))).to_be_bytes()
    }
    let first = double(encrypt(key, &Key::default()));
    let (whole, rest) = data.as_chunks::<16>();
    // The last block is masked with one subkey when it is complete and
    // with the other, after padding, when it is not.
    let (body, last) = match (whole.split_last(), rest) {
        (Some((last, body)), []) => (
            body,
            u128::from_be_bytes(*last) ^ u128::from_be_bytes(first),
        ),
        _ => {
            let mut padded = Key::default();
            padded[..rest.len()].copy_from_slice(rest);
            padded[rest.len()] = 0x80;
            (
                whole,
                u128::from_be_bytes(padded) ^ u128::from_be_bytes(double(first)),
            )
        }
    };
    let chained = body.iter().fold(0, |chain, block| {
        u128::from_be_bytes(encrypt(
            key,
            &(chain ^ u128::from_be_bytes(*block)).to_be_bytes(),
        ))
    });
    encrypt(key, &(chained ^ last).to_be_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn crypto() -> Crypto {
        let keys = KeySet::parse(
            "aes_kek_generation_source = 000102030405060708090a0b0c0d0e0f\n\
             master_key_00 = 101112131415161718191a1b1c1d1e1f\n\
             master_key_02 = 202122232425262728292a2b2c2d2e2f\n",
        )
        .unwrap();
        Crypto::new(keys, [0x5A; 16])
    }

    #[test]
    fn ladder_is_two_decryptions_to_the_kek_and_one_more_to_the_key() {
        let mut crypto = crypto();
        let (kek_source, key_source) = ([0x33; 16], [0x44; 16]);
        let access_key = crypto.generate_aes_kek(&kek_source, 3, 0).unwrap();
        crypto.load_aes_key(2, &access_key, &key_source).unwrap();

        let master_key: Key = core::array::from_fn(|index| 0x20 + index as u8);
        let static_source: Key = core::array::from_fn(|index| index as u8);
        let kek = decrypt(&decrypt(&master_key, &static_source), &kek_source);
        assert_eq!(crypto.slot(2), Some(&decrypt(&kek, &key_source)));
        assert_ne!(access_key, kek, "the kek itself is never handed out");
        assert_eq!(crypto.slot(1), None);
    }

    #[test]
    fn generation_counts_from_one_with_zero_as_an_alias_of_the_first() {
        let crypto = crypto();
        let source = [0x33; 16];
        assert_eq!(
            crypto.generate_aes_kek(&source, 0, 0),
            crypto.generate_aes_kek(&source, 1, 0)
        );
        assert_ne!(
            crypto.generate_aes_kek(&source, 1, 0),
            crypto.generate_aes_kek(&source, 3, 0)
        );
        // Generation 2 means master_key_01, which this key set lacks.
        assert!(matches!(
            crypto.generate_aes_kek(&source, 2, 0),
            Err(CryptoError::Unimplemented(_))
        ));
    }

    #[test]
    fn cmac_matches_the_rfc_4493_examples() {
        let key = 0x2b7e151628aed2a6abf7158809cf4f3c_u128.to_be_bytes();
        let message: Vec<u8> = [
            0x6bc1bee22e409f96e93d7e117393172a_u128,
            0xae2d8a571e03ac9c9eb76fac45af8e51,
            0x30c81c46a35ce411e5fbc1191a0a52ef,
        ]
        .iter()
        .flat_map(|block| block.to_be_bytes())
        .collect();
        let mac = |len: usize| u128::from_be_bytes(cmac(&key, &message[..len]));
        assert_eq!(mac(0), 0xbb1d6929e95937287fa37d129b756746);
        assert_eq!(mac(16), 0x070a16b46b4d4144f79bdd9dd04a287c);
        assert_eq!(mac(40), 0xdfa66747de9ae63030ca32611497c827);
    }

    #[test]
    fn refuses_what_the_monitor_refuses_and_reports_what_is_not_modeled() {
        let mut crypto = crypto();
        assert_eq!(
            crypto.generate_aes_kek(&[0; 16], 1, 1 << 8),
            Err(CryptoError::InvalidArgument)
        );
        assert!(matches!(
            crypto.generate_aes_kek(&[0; 16], 1, 1),
            Err(CryptoError::Unimplemented(_))
        ));
        assert!(matches!(
            crypto.generate_aes_kek(&[0; 16], 1, 1 << 5),
            Err(CryptoError::Unimplemented(_))
        ));
        assert_eq!(
            crypto.load_aes_key(6, &[0; 16], &[0; 16]),
            Err(CryptoError::InvalidArgument)
        );
    }
}
