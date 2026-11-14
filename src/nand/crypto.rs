//! The keys and hashes the filesystem process applies to the content
//! archives and save data it reads, reproduced on the host.
//!
//! As for the storage keys ([`super::bis`]), the filesystem process asks
//! the monitor for a kek, `kek = D(D(root, aes_kek_generation_source),
//! kek_source)`, and then either loads a key under it (`LoadAesKey`,
//! `key = D(kek, source)`) or generates one (`GenerateAesKey`,
//! `key = D(D(kek, aes_key_generation_source), source)`). Which of the two
//! each key takes was settled against a key file, comparing derived and
//! published keys as booleans: the content archive header key and the 69
//! key area keys of generations 0 to 0x16 are generated keys, under master
//! key 0 and under the master key of the archive's key revision; the save
//! MAC key is a loaded one under the device key, and its CMAC authenticates
//! the headers of the system saves the guest wrote itself.

use aes::{
    Aes128,
    cipher::{BlockDecrypt, BlockEncrypt, KeyInit},
};
use sha2::{Digest, Sha256};

use super::bis::MissingKey;
use crate::keys::{KEY_AREA_KEY_SOURCES, Key, KeyPair, KeySet};
pub(crate) use crate::secmon::crypto::cmac;

pub fn aes_decrypt(key: &Key, block: &Key) -> Key {
    let mut block = (*block).into();
    Aes128::new(key.into()).decrypt_block(&mut block);
    block.into()
}

fn kek(keys: &KeySet, root: &Key, kek_source: &Key) -> Result<Key, MissingKey> {
    let static_source = keys
        .aes_kek_generation_source()
        .ok_or(MissingKey("aes_kek_generation_source"))?;
    Ok(aes_decrypt(&aes_decrypt(root, static_source), kek_source))
}

/// `GenerateAesKey`'s key under `kek`, before it decrypts anything.
fn generated_key(keys: &KeySet, kek: &Key) -> Result<Key, MissingKey> {
    let source = keys
        .aes_key_generation_source()
        .ok_or(MissingKey("aes_key_generation_source"))?;
    Ok(aes_decrypt(kek, source))
}

/// The AES-XTS key pair of content archive headers.
pub fn nca_header_key(keys: &KeySet) -> Result<KeyPair, MissingKey> {
    let master_key = keys.master_key(0).ok_or(MissingKey("master_key_00"))?;
    let kek_source = keys
        .header_kek_source()
        .ok_or(MissingKey("header_kek_source"))?;
    let key = generated_key(keys, &kek(keys, master_key, kek_source)?)?;
    let sealed = keys
        .header_key_source()
        .ok_or(MissingKey("header_key_source"))?;
    let mut pair = [0; 32];
    for (half, sealed) in pair
        .as_chunks_mut::<16>()
        .0
        .iter_mut()
        .zip(sealed.as_chunks::<16>().0)
    {
        *half = aes_decrypt(&key, sealed);
    }
    Ok(pair)
}

/// The key that opens content archive key areas of kek index `index` (0
/// application, 1 ocean, 2 system) under `master_key`, the master key of
/// the archive's key revision.
pub fn key_area_key(keys: &KeySet, index: usize, master_key: &Key) -> Result<Key, MissingKey> {
    let kek_source = keys
        .key_area_key_source(index)
        .ok_or(MissingKey(KEY_AREA_KEY_SOURCES[index]))?;
    generated_key(keys, &kek(keys, master_key, kek_source)?)
}

/// The console-unique key the save data headers are authenticated with.
pub fn save_mac_key(keys: &KeySet) -> Result<Key, MissingKey> {
    let device_key = keys.device_key().ok_or(MissingKey("device_key"))?;
    let kek_source = keys
        .save_mac_kek_source()
        .ok_or(MissingKey("save_mac_kek_source"))?;
    let source = keys
        .save_mac_key_source()
        .ok_or(MissingKey("save_mac_key_source_00"))?;
    Ok(aes_decrypt(&kek(keys, device_key, kek_source)?, source))
}

/// AES-CTR over `data` in place, `counter` being the big-endian counter
/// block of its first byte, which is block-aligned.
pub fn ctr_apply(key: &Key, counter: u128, data: &mut [u8]) {
    let cipher = Aes128::new(key.into());
    for (index, chunk) in data.chunks_mut(16).enumerate() {
        let mut stream = counter.wrapping_add(index as u128).to_be_bytes().into();
        cipher.encrypt_block(&mut stream);
        for (byte, mask) in chunk.iter_mut().zip(stream) {
            *byte ^= mask;
        }
    }
}

pub fn sha256(data: &[u8]) -> [u8; 32] {
    Sha256::digest(data).into()
}

/// HMAC-SHA-256 as RFC 2104 defines it, for keys up to a block long.
pub fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32] {
    const BLOCK: usize = 64;
    assert!(key.len() <= BLOCK);
    let pad = |byte: u8| {
        let mut pad = [byte; BLOCK];
        for (pad, key) in pad.iter_mut().zip(key) {
            *pad ^= key;
        }
        pad
    };
    let inner = Sha256::new()
        .chain_update(pad(0x36))
        .chain_update(message)
        .finalize();
    Sha256::new()
        .chain_update(pad(0x5C))
        .chain_update(inner)
        .finalize()
        .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hmac_matches_rfc_4231_case_2() {
        let mac = hmac_sha256(b"Jefe", b"what do ya want for nothing?");
        assert_eq!(mac[..4], [0x5b, 0xdc, 0xc1, 0x46]);
        assert_eq!(mac[28..], [0x64, 0xec, 0x38, 0x43]);
    }
}
