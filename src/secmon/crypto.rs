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
//! An access key is sealed for one operation, named in the option word of
//! `GenerateAesKek`, and the monitor mixes that choice into the ladder too:
//! its static source is the key type's source XORed with a mask per seal
//! key (Atmosphère's public monitor, `secmon_smc_aes.cpp`,
//! `GenerateAesKekImpl`). For `LoadAesKey` the mask is zero, which leaves
//! `aes_kek_generation_source`; for `DecryptDeviceUniqueData` the result
//! is what key files call `rsa_private_kek_generation_source`.
//!
//! `DecryptDeviceUniqueData` opens data a console's factory wrapped for
//! that console ([`seal_device_unique_data`] is the format), such as the
//! TLS client key `ssl` keeps in the calibration data.
//!
//! The sealing keys are this machine's own. On hardware they are random per
//! boot and never leave the monitor, so access keys are opaque to the guest
//! either way.

use aes::{
    Aes128,
    cipher::{BlockDecrypt, BlockEncrypt, KeyInit, consts::U16},
};
use aes_gcm::{AesGcm, aead::AeadInPlace};

use crate::keys::{Key, KeySet};

/// Key slots userland may load keys into.
const USER_SLOT_COUNT: usize = 6;

/// `GenerateAesKek` option word: the kek is specific to this console.
const OPTION_DEVICE_UNIQUE: u32 = 1 << 0;
/// Which of the monitor's constants seeds the kek; 0 is the default one.
const OPTION_KEY_TYPE: u32 = 0xF << 1;
/// Which operation the access key will be valid for ([`SEAL_LOAD_AES_KEY`],
/// [`SEAL_DECRYPT_DEVICE_UNIQUE_DATA`], ...).
const OPTION_SEAL_KEY_SHIFT: u32 = 5;
const OPTION_SEAL_KEY: u32 = 7 << OPTION_SEAL_KEY_SHIFT;
const OPTION_RESERVED: u32 = !0xFF;

/// Seal key 0: the access key is for `LoadAesKey`.
pub const SEAL_LOAD_AES_KEY: u32 = 0;
/// Seal key 1: the access key is for `DecryptDeviceUniqueData` in its mode
/// 0, which hands the plain data back to the caller.
pub const SEAL_DECRYPT_DEVICE_UNIQUE_DATA: u32 = 1;

/// The `GenerateAesKek` option word that seals for `seal` with the default
/// key type, not device-unique: what `ssl` passes for a key of key
/// generation 0 (`0x35f80`: 32).
pub const fn kek_option(seal: u32) -> u32 {
    seal << OPTION_SEAL_KEY_SHIFT
}

/// Bytes of device-unique data around its payload: the IV before it, then
/// eight bytes of padding, the device ID and the GMAC after it (Atmosphère's
/// `secmon_smc_device_unique_data.cpp`).
pub const DEVICE_UNIQUE_DATA_OVERHEAD: usize = IV_SIZE + PADDING_SIZE + DEVICE_ID_SIZE + MAC_SIZE;
/// Most device-unique data the monitor takes (`DeviceUniqueDataSizeMax`).
pub const DEVICE_UNIQUE_DATA_MAX: usize = 0x240;
const IV_SIZE: usize = 0x10;
const PADDING_SIZE: usize = 8;
const DEVICE_ID_SIZE: usize = 8;
const MAC_SIZE: usize = 0x10;
/// The monitor compares the low 56 bits of the stored device ID with the
/// fuses' (`GetDeviceIdLow`); the top byte is the caller's to read.
const DEVICE_ID_LOW: u64 = (1 << 56) - 1;

/// GCM with the sixteen-byte IVs of device-unique data.
type Gcm = AesGcm<Aes128, U16>;

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

/// AES-128 in counter mode with a big-endian 128-bit counter, as the
/// security engine runs it, over `data` in place; a partial last block
/// uses part of its pad.
fn ctr(key: &Key, counter: &Key, data: &mut [u8]) {
    let cipher = Aes128::new(key.into());
    let mut counter = u128::from_be_bytes(*counter);
    for block in data.chunks_mut(16) {
        let mut pad = counter.to_be_bytes().into();
        cipher.encrypt_block(&mut pad);
        for (byte, pad) in block.iter_mut().zip(pad) {
            *byte ^= pad;
        }
        counter = counter.wrapping_add(1);
    }
}

/// The GMAC of device-unique data: the GCM tag of no plaintext with `data`
/// as the additional data (`ComputeGmac`).
fn gmac(key: &Key, iv: &Key, data: &[u8]) -> [u8; MAC_SIZE] {
    Gcm::new(key.into())
        .encrypt_in_place_detached(iv.into(), data, &mut [])
        .expect("GCM takes any amount of additional data the monitor does")
        .into()
}

/// The kek `GenerateAesKek` derives for `source`, `generation` (counting
/// from one, zero meaning the first as well) and `option`, before the
/// monitor seals it, and the seal key it would be sealed for.
fn key_encryption_key(
    keys: &KeySet,
    source: &Key,
    generation: u64,
    option: u32,
) -> Result<(Key, u32), CryptoError> {
    if option & OPTION_RESERVED != 0 {
        return Err(CryptoError::InvalidArgument);
    }
    if option & OPTION_KEY_TYPE != 0 {
        return Err(CryptoError::Unimplemented(
            "key encryption keys of a non-default type",
        ));
    }
    let seal = (option & OPTION_SEAL_KEY) >> OPTION_SEAL_KEY_SHIFT;
    let generation =
        u8::try_from(generation.saturating_sub(1)).map_err(|_| CryptoError::InvalidArgument)?;
    // The monitor has no key of a generation newer than its own, and
    // says so; callers probe for the newest generation this way.
    if keys
        .newest_generation()
        .is_some_and(|newest| generation > newest)
    {
        return Err(CryptoError::InvalidArgument);
    }
    let root_key = if option & OPTION_DEVICE_UNIQUE == 0 {
        keys.master_key(generation)
            .ok_or(CryptoError::Unimplemented(
                "the master key of the requested generation is not in the key file",
            ))?
    } else if generation == 0 {
        keys.device_key().ok_or(CryptoError::Unimplemented(
            "device_key is not in the key file",
        ))?
    } else {
        return Err(CryptoError::Unimplemented(
            "device-unique keys of generations after the first",
        ));
    };
    let static_source = match seal {
        SEAL_LOAD_AES_KEY => keys
            .aes_kek_generation_source()
            .ok_or(CryptoError::Unimplemented(
                "aes_kek_generation_source is not in the key file",
            ))?,
        SEAL_DECRYPT_DEVICE_UNIQUE_DATA => {
            keys.rsa_private_kek_generation_source()
                .ok_or(CryptoError::Unimplemented(
                    "rsa_private_kek_generation_source is not in the key file",
                ))?
        }
        _ => {
            return Err(CryptoError::Unimplemented(
                "access keys for operations other than LoadAesKey and DecryptDeviceUniqueData",
            ));
        }
    };
    Ok((decrypt(&decrypt(root_key, static_source), source), seal))
}

/// The key `DecryptDeviceUniqueData` opens data with when given the access
/// key `GenerateAesKek` makes from `kek_source`, `generation` and
/// `option`, and `key_source`: `D(kek, key_source)`.
pub fn device_unique_data_key(
    keys: &KeySet,
    kek_source: &Key,
    key_source: &Key,
    generation: u64,
    option: u32,
) -> Result<Key, CryptoError> {
    let (kek, seal) = key_encryption_key(keys, kek_source, generation, option)?;
    if seal != SEAL_DECRYPT_DEVICE_UNIQUE_DATA {
        return Err(CryptoError::InvalidArgument);
    }
    Ok(decrypt(&kek, key_source))
}

/// Device-unique data as a console's factory makes it for the monitor
/// (Atmosphère's `EncryptDeviceUniqueData`): `iv`, then `payload`, eight
/// zero bytes and the big-endian `device_id` encrypted with `key` in
/// counter mode from `iv`, then the GMAC of those plain bytes under `key`
/// and `iv`. `key` is `D(kek, key_source)` of the ladder.
pub fn seal_device_unique_data(key: &Key, iv: &Key, payload: &[u8], device_id: u64) -> Vec<u8> {
    let mut plain = payload.to_vec();
    plain.extend_from_slice(&[0; PADDING_SIZE]);
    plain.extend_from_slice(&device_id.to_be_bytes());
    let mac = gmac(key, iv, &plain);
    ctr(key, iv, &mut plain);
    let mut sealed = iv.to_vec();
    sealed.extend_from_slice(&plain);
    sealed.extend_from_slice(&mac);
    sealed
}

/// The monitor's side of [`seal_device_unique_data`]: decrypts `data` in
/// place, the plain bytes (payload, padding, device ID) from its start and
/// the rest as it was, and checks the GMAC and that the device ID is
/// `device_id`. False, with `data` partly decrypted, when either is wrong.
fn open_device_unique_data(key: &Key, data: &mut [u8], device_id: u64) -> bool {
    let iv: Key = data[..IV_SIZE].try_into().expect("an IV");
    let encrypted = data.len() - IV_SIZE - MAC_SIZE;
    let mac: [u8; MAC_SIZE] = data[IV_SIZE + encrypted..].try_into().expect("a MAC");
    data.copy_within(IV_SIZE..IV_SIZE + encrypted, 0);
    let plain = &mut data[..encrypted];
    ctr(key, &iv, plain);
    let stored_id = u64::from_be_bytes(plain[encrypted - DEVICE_ID_SIZE..].try_into().unwrap());
    gmac(key, &iv, plain) == mac && stored_id & DEVICE_ID_LOW == device_id
}

impl Crypto {
    /// `seal_key` is what the access keys handed to userland are sealed
    /// with: one key per operation follows from it.
    pub fn new(keys: KeySet, seal_key: Key) -> Self {
        Self {
            keys,
            seal_key,
            slots: [None; USER_SLOT_COUNT],
        }
    }

    /// The key access keys for operation `seal` are sealed with, so that one
    /// sealed for an operation opens nothing in another, as on hardware.
    fn seal_key(&self, seal: u32) -> Key {
        encrypt(&self.seal_key, &[seal as u8; 16])
    }

    /// Derives the kek `GenerateAesKek` makes of `source`, `generation` and
    /// `option` and returns it sealed for the operation `option` names.
    pub fn generate_aes_kek(
        &self,
        source: &Key,
        generation: u64,
        option: u32,
    ) -> Result<Key, CryptoError> {
        let (kek, seal) = key_encryption_key(&self.keys, source, generation, option)?;
        Ok(encrypt(&self.seal_key(seal), &kek))
    }

    /// Loads `D(kek, source)` into `slot`, `kek` being what `access_key` seals.
    pub fn load_aes_key(
        &mut self,
        slot: u64,
        access_key: &Key,
        source: &Key,
    ) -> Result<(), CryptoError> {
        let kek = decrypt(&self.seal_key(SEAL_LOAD_AES_KEY), access_key);
        let slot = usize::try_from(slot)
            .ok()
            .and_then(|slot| self.slots.get_mut(slot))
            .ok_or(CryptoError::InvalidArgument)?;
        *slot = Some(decrypt(&kek, source));
        Ok(())
    }

    /// `DecryptDeviceUniqueData` in mode 0: opens `data`
    /// ([`seal_device_unique_data`]) in place with `D(kek, key_source)`,
    /// `kek` being what `access_key` seals, and checks it belongs to the
    /// console `device_id` names. The plain bytes are at the start of
    /// `data` afterwards. Sizes are the monitor's: more than the overhead,
    /// at most [`DEVICE_UNIQUE_DATA_MAX`].
    pub fn decrypt_device_unique_data(
        &self,
        access_key: &Key,
        key_source: &Key,
        data: &mut [u8],
        device_id: u64,
    ) -> Result<(), CryptoError> {
        if data.len() <= DEVICE_UNIQUE_DATA_OVERHEAD || data.len() > DEVICE_UNIQUE_DATA_MAX {
            return Err(CryptoError::InvalidArgument);
        }
        let kek = decrypt(&self.seal_key(SEAL_DECRYPT_DEVICE_UNIQUE_DATA), access_key);
        if open_device_unique_data(&decrypt(&kek, key_source), data, device_id) {
            Ok(())
        } else {
            Err(CryptoError::InvalidArgument)
        }
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
        if mode == CipherMode::Ctr {
            ctr(key, iv, data);
            return Ok(());
        }
        let cipher = Aes128::new(key.into());
        let mut chain = *iv;
        for block in data.as_chunks_mut::<16>().0 {
            if mode == CipherMode::CbcEncrypt {
                for (byte, previous) in block.iter_mut().zip(chain) {
                    *byte ^= previous;
                }
                cipher.encrypt_block(block.into());
                chain = *block;
            } else {
                let ciphertext = *block;
                cipher.decrypt_block(block.into());
                for (byte, previous) in block.iter_mut().zip(chain) {
                    *byte ^= previous;
                }
                chain = ciphertext;
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
             rsa_private_kek_generation_source = 303132333435363738393a3b3c3d3e3f\n\
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
            crypto.generate_aes_kek(&[0; 16], 1, 2 << 5),
            Err(CryptoError::Unimplemented(_))
        ));
        assert_eq!(
            crypto.load_aes_key(6, &[0; 16], &[0; 16]),
            Err(CryptoError::InvalidArgument)
        );
    }

    /// GMAC per NIST SP 800-38D, one bit at a time, for an IV that is not
    /// 96 bits long: `J0 = GHASH(IV || 0^64 || [len(IV)]_64)`.
    fn reference_gmac(key: &Key, iv: &Key, data: &[u8]) -> Key {
        fn multiply(x: u128, y: u128) -> u128 {
            let (mut z, mut v) = (0, y);
            for bit in (0..128).rev() {
                if x >> bit & 1 == 1 {
                    z ^= v;
                }
                v = if v & 1 == 1 {
                    v >> 1 ^ 0xE1 << 120
                } else {
                    v >> 1
                };
            }
            z
        }
        let h = u128::from_be_bytes(encrypt(key, &Key::default()));
        let ghash = |blocks: &[u8], lengths: u128| {
            let mut y = 0;
            for chunk in blocks.chunks(16) {
                let mut block = Key::default();
                block[..chunk.len()].copy_from_slice(chunk);
                y = multiply(y ^ u128::from_be_bytes(block), h);
            }
            multiply(y ^ lengths, h)
        };
        let j0 = ghash(iv, 128);
        let s = ghash(data, (data.len() as u128 * 8) << 64);
        (u128::from_be_bytes(encrypt(key, &j0.to_be_bytes())) ^ s).to_be_bytes()
    }

    #[test]
    fn gmac_is_gcm_with_a_sixteen_byte_iv() {
        let (key, iv) = ([0x11; 16], [0x22; 16]);
        for len in [0, 1, 16, 0x110, 0x123] {
            let data: Vec<u8> = (0..len).map(|index| (index * 7) as u8).collect();
            assert_eq!(gmac(&key, &iv, &data), reference_gmac(&key, &iv, &data));
        }
    }

    #[test]
    fn device_unique_data_opens_only_with_its_access_key_and_console() {
        let crypto = crypto();
        let (kek_source, key_source, iv) = ([0x33; 16], [0x44; 16], [0x55; 16]);
        let device_id = 0x0012_3456_789A_BCDE;
        let option = kek_option(SEAL_DECRYPT_DEVICE_UNIQUE_DATA);
        let (kek, seal) = key_encryption_key(&crypto.keys, &kek_source, 0, option).unwrap();
        assert_eq!(seal, SEAL_DECRYPT_DEVICE_UNIQUE_DATA);
        // The static source is the one sealed for this operation.
        let master_key: Key = core::array::from_fn(|index| 0x10 + index as u8);
        let static_source: Key = core::array::from_fn(|index| 0x30 + index as u8);
        assert_eq!(
            kek,
            decrypt(&decrypt(&master_key, &static_source), &kek_source)
        );
        let payload: Vec<u8> = (0..0x100).map(|index| index as u8 ^ 0xA5).collect();
        let sealed = seal_device_unique_data(&decrypt(&kek, &key_source), &iv, &payload, device_id);
        assert_eq!(sealed.len(), 0x130);
        assert_eq!(sealed[..16], iv);
        assert_ne!(sealed[16..0x116], payload);

        let access_key = crypto.generate_aes_kek(&kek_source, 0, option).unwrap();
        let open = |access_key: &Key, data: &[u8], device_id| {
            let mut data = data.to_vec();
            crypto
                .decrypt_device_unique_data(access_key, &key_source, &mut data, device_id)
                .map(|()| data)
        };
        let opened = open(&access_key, &sealed, device_id).unwrap();
        assert_eq!(opened[..0x100], payload);
        assert_eq!(opened[0x100..0x108], [0; 8]);
        assert_eq!(opened[0x108..0x110], device_id.to_be_bytes());
        assert_eq!(
            opened[0x110..],
            sealed[0x110..],
            "the rest is left as it was"
        );
        // The top byte of the stored ID is not compared.
        let high = seal_device_unique_data(
            &decrypt(&kek, &key_source),
            &iv,
            &payload,
            device_id | 0x7F << 56,
        );
        assert!(open(&access_key, &high, device_id).is_ok());

        let refused = Err(CryptoError::InvalidArgument);
        assert_eq!(open(&access_key, &sealed, device_id + 1), refused);
        let mut tampered = sealed.clone();
        tampered[0x40] ^= 1;
        assert_eq!(open(&access_key, &tampered, device_id), refused);
        let for_load_aes_key = crypto.generate_aes_kek(&kek_source, 0, 0).unwrap();
        assert_eq!(open(&for_load_aes_key, &sealed, device_id), refused);
        assert_eq!(open(&access_key, &sealed[..0x30], device_id), refused);
        assert_eq!(open(&access_key, &[0; 0x250], device_id), refused);
    }
}
