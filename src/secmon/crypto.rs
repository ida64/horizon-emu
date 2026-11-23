//! The monitor's AES key ladder and RSA key storage, as userland drives
//! them through `spl`.
//!
//! Userland never holds a key. It asks the monitor to derive a key
//! encryption key from a source value (`GenerateAesKek`), gets it back
//! sealed under a key only the monitor has, and hands that "access key" back
//! together with a second source to load the final key into a key slot
//! (`LoadAesKey`). In terms of the keys a key file holds:
//!
//! ```text
//! kek  = D(D(master_key[generation], aes_kek_generation_source ^ mask), kek_source)
//! key  = D(kek, key_source)
//! ```
//!
//! with `D` a single AES-128 block decryption. This is the public
//! description of the ladder; it reproduces the derived keys published
//! alongside master keys (for example the key area keys).
//!
//! An access key is sealed for one operation, named in the option word of
//! `GenerateAesKek`, and the monitor mixes that choice into the ladder as
//! `mask`, one per operation (`SEAL_KEY_MASKS`). The masks make the
//! sources key files list under other names: for `DecryptDeviceUniqueData`
//! the result is `rsa_private_kek_generation_source`, for importing the ES
//! device key `rsa_oaep_kek_generation_source`.
//!
//! `DecryptDeviceUniqueData` opens data a console's factory wrapped for
//! that console ([`seal_device_unique_data`] is the format), such as the
//! TLS client key `ssl` keeps in the calibration data. In its other modes
//! it keeps the RSA private key inside for later exponentiations
//! (`ModularExponentiateByStorageKey`, `PrepareEsDeviceUniqueKey`) instead
//! of handing it back.
//!
//! The sealing keys are this machine's own. On hardware they are random per
//! boot and never leave the monitor, so access keys are opaque to the guest
//! either way.

use aes::{
    Aes128,
    cipher::{BlockDecrypt, BlockEncrypt, KeyInit, consts::U16},
};
use aes_gcm::{AesGcm, aead::AeadInPlace};
use rsa::BigUint;
use sha2::{Digest, Sha256};

use crate::keys::{Key, KeySet};

/// Key slots userland may load keys into.
pub const USER_SLOT_COUNT: usize = 6;

/// `GenerateAesKek` option word: the kek is specific to this console.
const OPTION_DEVICE_UNIQUE: u32 = 1 << 0;
/// Which of the monitor's constants seeds the kek: [`KEY_TYPE_DEFAULT`] or
/// one tied to normal or recovery boots.
const OPTION_KEY_TYPE_SHIFT: u32 = 1;
const OPTION_KEY_TYPE: u32 = 0xF << OPTION_KEY_TYPE_SHIFT;
/// Which operation the access key will be valid for ([`SEAL_LOAD_AES_KEY`],
/// [`SEAL_DECRYPT_DEVICE_UNIQUE_DATA`], ...).
const OPTION_SEAL_KEY_SHIFT: u32 = 5;
const OPTION_SEAL_KEY: u32 = 7 << OPTION_SEAL_KEY_SHIFT;
const OPTION_RESERVED: u32 = !0xFF;

/// The key type whose source key files call `aes_kek_generation_source`.
const KEY_TYPE_DEFAULT: u32 = 0;
/// The key type valid only in recovery boots; this console boots normally.
const KEY_TYPE_RECOVERY_ONLY: u32 = 2;
/// Key types 0 to 3 exist.
const KEY_TYPE_COUNT: u32 = 4;

/// Seal key 0: the access key is for `LoadAesKey`.
pub const SEAL_LOAD_AES_KEY: u32 = 0;
/// Seal key 1: the access key is for `DecryptDeviceUniqueData` in its mode
/// 0, which hands the plain data back to the caller.
pub const SEAL_DECRYPT_DEVICE_UNIQUE_DATA: u32 = 1;
/// Seal key 2: for importing the Lotus (game card) RSA key.
pub const SEAL_IMPORT_LOTUS_KEY: u32 = 2;
/// Seal key 3: for importing the ES device (ticket) RSA key.
pub const SEAL_IMPORT_ES_DEVICE_KEY: u32 = 3;
/// Seal key 4: for opening data `ReencryptDeviceUniqueData` rewraps.
pub const SEAL_REENCRYPT_DEVICE_UNIQUE_DATA: u32 = 4;
/// Seal key 5: for importing the TLS client RSA key.
pub const SEAL_IMPORT_SSL_KEY: u32 = 5;
/// Seal key 6: for importing the ES client certificate RSA key.
pub const SEAL_IMPORT_ES_CLIENT_CERT_KEY: u32 = 6;
/// Seal keys 0 to 6 exist.
const SEAL_KEY_COUNT: u32 = 7;
/// What the ES calls seal title keys with for `LoadPreparedAesKey`: a seal
/// key of its own on hardware (from its own source), outside the range the
/// option word can name.
const SEAL_PREPARED_ES_KEY: u32 = SEAL_KEY_COUNT;

/// What the monitor XORs into the key type's source for each seal key
/// (Atmosphère's public monitor, `secmon_smc_aes.cpp`, `SealKeyMasks`).
const SEAL_KEY_MASKS: [Key; SEAL_KEY_COUNT as usize] = [
    [0; 16],
    0xA2AB_BF9C_922F_BBE3_7879_9BC0_CCEA_A574_u128.to_be_bytes(),
    0x57E2_D945_E492_F4FD_C3F9_8638_8978_9F3C_u128.to_be_bytes(),
    0xE54D_9A02_F04F_5FA8_AD76_0AF6_3295_59BB_u128.to_be_bytes(),
    0x59D9_31F4_A797_B814_40D6_A260_2BED_1531_u128.to_be_bytes(),
    0xFD6A_25E5_D838_7F91_49DA_F859_A828_E675_u128.to_be_bytes(),
    0x8996_439A_7CD5_5955_24D5_2418_AB6C_0461_u128.to_be_bytes(),
];

/// The `GenerateAesKek` option word that seals for `seal` with the default
/// key type, not device-unique: what `ssl` passes for a key of key
/// generation 0 (`0x35f80`: 32).
pub const fn kek_option(seal: u32) -> u32 {
    seal << OPTION_SEAL_KEY_SHIFT
}

/// What `DecryptDeviceUniqueData` does with the data, by its mode: hand it
/// back (0), or keep the RSA key inside as the Lotus key (1), the ES
/// device key (2), the TLS client key (3) or the ES client certificate key
/// (4).
pub const DEVICE_UNIQUE_DATA_MODES: u32 = 5;
/// The seal key of each mode's access key.
const MODE_SEALS: [u32; DEVICE_UNIQUE_DATA_MODES as usize] = [
    SEAL_DECRYPT_DEVICE_UNIQUE_DATA,
    SEAL_IMPORT_LOTUS_KEY,
    SEAL_IMPORT_ES_DEVICE_KEY,
    SEAL_IMPORT_SSL_KEY,
    SEAL_IMPORT_ES_CLIENT_CERT_KEY,
];

/// Bytes of device-unique data around its payload: the IV before it, then
/// eight bytes of padding, the device ID and the GMAC after it (Atmosphère's
/// `secmon_smc_device_unique_data.cpp`).
pub const DEVICE_UNIQUE_DATA_OVERHEAD: usize = IV_SIZE + PADDING_SIZE + DEVICE_ID_SIZE + MAC_SIZE;
/// Most device-unique data the monitor takes (`DeviceUniqueDataSizeMax`).
pub const DEVICE_UNIQUE_DATA_MAX: usize = 0x240;
/// Least device-unique data that holds a key: an RSA private exponent and
/// the overhead (`DeviceUniqueDataSizeMin`).
const DEVICE_UNIQUE_KEY_MIN: usize = RSA_SIZE + DEVICE_UNIQUE_DATA_OVERHEAD;
const IV_SIZE: usize = 0x10;
const PADDING_SIZE: usize = 8;
const DEVICE_ID_SIZE: usize = 8;
const MAC_SIZE: usize = 0x10;
/// The monitor compares the low 56 bits of the stored device ID with the
/// fuses' (`GetDeviceIdLow`); the top byte is the caller's to read.
const DEVICE_ID_LOW: u64 = (1 << 56) - 1;

/// Bytes of the security engine's RSA operands: 2048-bit keys only.
pub const RSA_SIZE: usize = 0x100;
pub type RsaBlock = [u8; RSA_SIZE];
/// The RSA keys the monitor stores, by import mode less one.
const RSA_KEY_LOTUS: usize = 0;
const RSA_KEY_ES_DEVICE: usize = 1;
const RSA_KEY_SSL: usize = 2;
const RSA_KEY_ES_CLIENT_CERT: usize = 3;
/// The stored key each `ModularExponentiateByStorageKey` mode uses.
const STORAGE_KEY_BY_MODE: [usize; 3] = [RSA_KEY_LOTUS, RSA_KEY_SSL, RSA_KEY_ES_CLIENT_CERT];
/// Modes of `ModularExponentiateByStorageKey`.
pub const STORAGE_KEY_MODES: u32 = STORAGE_KEY_BY_MODE.len() as u32;
/// ES common key types: title keys (0), archive keys (1) and a third.
pub const ES_COMMON_KEY_TYPES: u32 = 3;

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

/// An RSA private key the monitor keeps. A modulus that arrived without
/// its key (Lotus, TLS) is provisional until a test exponentiation with
/// it succeeds; the ES keys bring theirs.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct StoredRsaKey {
    #[serde(with = "crate::snapshot::array")]
    exponent: RsaBlock,
    #[serde(with = "crate::snapshot::array")]
    modulus: RsaBlock,
    committed: bool,
}

/// What userland has put into the monitor's keeping since boot: key slots
/// and RSA keys. The key file and the seal key follow from the boot
/// parameters, so a snapshot does not hold them.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct SavedCrypto {
    slots: [Option<Key>; USER_SLOT_COUNT],
    rsa_keys: [StoredRsaKey; 4],
}

const NO_RSA_KEY: StoredRsaKey = StoredRsaKey {
    exponent: [0; RSA_SIZE],
    modulus: [0; RSA_SIZE],
    committed: false,
};

pub struct Crypto {
    keys: KeySet,
    seal_key: Key,
    slots: [Option<Key>; USER_SLOT_COUNT],
    rsa_keys: [StoredRsaKey; 4],
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

/// The master key of `generation` as the calls number them: counting from
/// one, zero meaning the first as well. The monitor has no key of a
/// generation newer than its own, and says so; callers probe for the
/// newest generation this way.
fn master_key(keys: &KeySet, generation: u64) -> Result<&Key, CryptoError> {
    let generation =
        u8::try_from(generation.saturating_sub(1)).map_err(|_| CryptoError::InvalidArgument)?;
    if keys
        .newest_generation()
        .is_some_and(|newest| generation > newest)
    {
        return Err(CryptoError::InvalidArgument);
    }
    keys.master_key(generation)
        .ok_or(CryptoError::Unimplemented(
            "the master key of the requested generation is not in the key file",
        ))
}

/// The kek `GenerateAesKek` derives for `source`, `generation` and
/// `option`, before the monitor seals it, and the seal key it would be
/// sealed for.
fn key_encryption_key(
    keys: &KeySet,
    source: &Key,
    generation: u64,
    option: u32,
) -> Result<(Key, u32), CryptoError> {
    let key_type = (option & OPTION_KEY_TYPE) >> OPTION_KEY_TYPE_SHIFT;
    let seal = (option & OPTION_SEAL_KEY) >> OPTION_SEAL_KEY_SHIFT;
    if option & OPTION_RESERVED != 0
        || key_type >= KEY_TYPE_COUNT
        || key_type == KEY_TYPE_RECOVERY_ONLY
        || seal >= SEAL_KEY_COUNT
    {
        return Err(CryptoError::InvalidArgument);
    }
    let root_key = if option & OPTION_DEVICE_UNIQUE == 0 {
        master_key(keys, generation)?
    } else {
        // Device-unique keys exist for the first generation and from
        // 4.0.0's (the fourth) on (`IsValidDeviceUniqueKeyGeneration`).
        match generation.saturating_sub(1) {
            0 => keys.device_key().ok_or(CryptoError::Unimplemented(
                "device_key is not in the key file",
            ))?,
            1 | 2 => return Err(CryptoError::InvalidArgument),
            _ => {
                return Err(CryptoError::Unimplemented(
                    "device-unique keys of generations after the first",
                ));
            }
        }
    };
    if key_type != KEY_TYPE_DEFAULT {
        return Err(CryptoError::Unimplemented(
            "the sources of key types tied to normal boots are not in key files",
        ));
    }
    let mut static_source = *keys
        .aes_kek_generation_source()
        .ok_or(CryptoError::Unimplemented(
            "aes_kek_generation_source is not in the key file",
        ))?;
    for (byte, mask) in static_source.iter_mut().zip(SEAL_KEY_MASKS[seal as usize]) {
        *byte ^= mask;
    }
    Ok((decrypt(&decrypt(root_key, &static_source), source), seal))
}

/// The key device-unique data is wrapped with when the access key is what
/// `GenerateAesKek` makes from `kek_source`, `generation` and `option`:
/// `D(kek, key_source)`.
pub fn device_unique_data_key(
    keys: &KeySet,
    kek_source: &Key,
    key_source: &Key,
    generation: u64,
    option: u32,
) -> Result<Key, CryptoError> {
    let (kek, _) = key_encryption_key(keys, kek_source, generation, option)?;
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
/// the rest as it was, and checks the GMAC and that the low 56 bits of the
/// device ID are those of `device_id`, which may carry a top byte (the ID
/// a console's certificates name does). The stored ID's top byte, or
/// `None`, with `data` partly decrypted, when either check fails.
pub fn open_device_unique_data(key: &Key, data: &mut [u8], device_id: u64) -> Option<u8> {
    let iv: Key = data[..IV_SIZE].try_into().expect("an IV");
    let encrypted = data.len() - IV_SIZE - MAC_SIZE;
    let mac: [u8; MAC_SIZE] = data[IV_SIZE + encrypted..].try_into().expect("a MAC");
    data.copy_within(IV_SIZE..IV_SIZE + encrypted, 0);
    let plain = &mut data[..encrypted];
    ctr(key, &iv, plain);
    let stored_id = u64::from_be_bytes(plain[encrypted - DEVICE_ID_SIZE..].try_into().unwrap());
    (gmac(key, &iv, plain) == mac && stored_id & DEVICE_ID_LOW == device_id & DEVICE_ID_LOW)
        .then_some((stored_id >> 56) as u8)
}

/// Refuses device-unique data of a size the monitor refuses for `mode`.
fn check_device_unique_data(mode: u32, len: usize) -> Result<(), CryptoError> {
    let fits = match mode {
        0 => DEVICE_UNIQUE_DATA_OVERHEAD < len && len <= DEVICE_UNIQUE_DATA_MAX,
        1..DEVICE_UNIQUE_DATA_MODES => {
            (DEVICE_UNIQUE_KEY_MIN..=DEVICE_UNIQUE_DATA_MAX).contains(&len)
        }
        _ => false,
    };
    fits.then_some(()).ok_or(CryptoError::InvalidArgument)
}

/// `message^exponent mod modulus`, all big-endian, as the engine's RSA
/// unit computes it; `None` for a zero modulus.
fn exponentiate(message: &[u8], exponent: &[u8], modulus: &[u8]) -> Option<RsaBlock> {
    let modulus = BigUint::from_bytes_be(modulus);
    if modulus == BigUint::from(0u8) {
        return None;
    }
    let result = BigUint::from_bytes_be(message)
        .modpow(&BigUint::from_bytes_be(exponent), &modulus)
        .to_bytes_be();
    let mut block = [0; RSA_SIZE];
    block[RSA_SIZE - result.len()..].copy_from_slice(&result);
    Some(block)
}

/// MGF1 with SHA-256 (RFC 8017, B.2.1), `len` bytes of mask from `seed`.
fn mgf1(seed: &[u8], len: usize) -> Vec<u8> {
    let mut mask = Vec::with_capacity(len + 32);
    for counter in 0u32.. {
        if mask.len() >= len {
            break;
        }
        mask.extend_from_slice(
            &Sha256::new()
                .chain_update(seed)
                .chain_update(counter.to_be_bytes())
                .finalize(),
        );
    }
    mask.truncate(len);
    mask
}

/// EME-OAEP decoding with SHA-256 (RFC 8017, 7.1.2 step 3) of an
/// exponentiation's output, given the label's digest rather than the
/// label (`DecodeRsaOaepSha256`). The message, or `None` if the block is
/// not an encoding under that label.
fn oaep_sha256_decode(block: &RsaBlock, label_digest: &[u8; 32]) -> Option<Vec<u8>> {
    const HASH: usize = 32;
    let (masked_seed, masked_db) = block[1..].split_at(HASH);
    let mut seed = masked_seed.to_vec();
    seed.iter_mut()
        .zip(mgf1(masked_db, HASH))
        .for_each(|(byte, mask)| *byte ^= mask);
    let mut db = masked_db.to_vec();
    db.iter_mut()
        .zip(mgf1(&seed, masked_db.len()))
        .for_each(|(byte, mask)| *byte ^= mask);
    if block[0] != 0 || db[..HASH] != label_digest[..] {
        return None;
    }
    let rest = &db[HASH..];
    let one = rest.iter().position(|&byte| byte != 0)?;
    (rest[one] == 1).then(|| rest[one + 1..].to_vec())
}

impl Crypto {
    /// `seal_key` is what the access keys handed to userland are sealed
    /// with: one key per operation follows from it.
    pub fn new(keys: KeySet, seal_key: Key) -> Self {
        Self {
            keys,
            seal_key,
            slots: [None; USER_SLOT_COUNT],
            rsa_keys: [NO_RSA_KEY, NO_RSA_KEY, NO_RSA_KEY, NO_RSA_KEY],
        }
    }

    /// The console's key material, for the monitor's cold boot (package2).
    pub fn keys(&self) -> &KeySet {
        &self.keys
    }

    pub fn save_state(&self) -> SavedCrypto {
        SavedCrypto {
            slots: self.slots,
            rsa_keys: self.rsa_keys.clone(),
        }
    }

    pub fn restore_state(&mut self, saved: SavedCrypto) {
        self.slots = saved.slots;
        self.rsa_keys = saved.rsa_keys;
    }

    /// The key access keys for operation `seal` are sealed with, so that one
    /// sealed for an operation opens nothing in another, as on hardware.
    fn seal_key(&self, seal: u32) -> Key {
        encrypt(&self.seal_key, &[seal as u8; 16])
    }

    /// What `access_key`, sealed for `seal`, holds.
    fn unseal(&self, seal: u32, access_key: &Key) -> Key {
        decrypt(&self.seal_key(seal), access_key)
    }

    fn user_slot(&mut self, slot: u64) -> Result<&mut Option<Key>, CryptoError> {
        usize::try_from(slot)
            .ok()
            .and_then(|slot| self.slots.get_mut(slot))
            .ok_or(CryptoError::InvalidArgument)
    }

    /// Whether the monitor has a master key of `generation` (numbered as
    /// the calls number it).
    pub fn valid_generation(&self, generation: u64) -> bool {
        master_key(&self.keys, generation) != Err(CryptoError::InvalidArgument)
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
        let key = decrypt(&self.unseal(SEAL_LOAD_AES_KEY, access_key), source);
        *self.user_slot(slot)? = Some(key);
        Ok(())
    }

    /// `DecryptDeviceUniqueData` in `mode`: opens `data`
    /// ([`seal_device_unique_data`]) in place with `D(kek, key_source)`,
    /// `kek` being what `access_key` seals for that mode, and checks it
    /// belongs to the console `device_id` names. In mode 0 the plain bytes
    /// are at the start of `data` afterwards, for the caller; the other
    /// modes keep the RSA key they hold: the private exponent first, then,
    /// for the ES keys, the modulus. Sizes are the monitor's.
    pub fn decrypt_device_unique_data(
        &mut self,
        mode: u32,
        access_key: &Key,
        key_source: &Key,
        data: &mut [u8],
        device_id: u64,
    ) -> Result<(), CryptoError> {
        check_device_unique_data(mode, data.len())?;
        let key = decrypt(
            &self.unseal(MODE_SEALS[mode as usize], access_key),
            key_source,
        );
        open_device_unique_data(&key, data, device_id).ok_or(CryptoError::InvalidArgument)?;
        let Some(index) = (mode as usize).checked_sub(1) else {
            return Ok(());
        };
        let with_modulus = matches!(index, RSA_KEY_ES_DEVICE | RSA_KEY_ES_CLIENT_CERT);
        let plain = data.len() - DEVICE_UNIQUE_DATA_OVERHEAD;
        if with_modulus && plain < 2 * RSA_SIZE {
            return Err(CryptoError::Unimplemented(
                "an ES key import too short for its exponent and modulus",
            ));
        }
        let stored = &mut self.rsa_keys[index];
        stored.exponent.copy_from_slice(&data[..RSA_SIZE]);
        stored.modulus = [0; RSA_SIZE];
        stored.committed = with_modulus;
        if with_modulus {
            stored
                .modulus
                .copy_from_slice(&data[RSA_SIZE..2 * RSA_SIZE]);
        }
        Ok(())
    }

    /// `ReencryptDeviceUniqueData`: opens `data` with the access key sealed
    /// for re-encryption and `decrypt_source`, then wraps the payload anew
    /// for `mode` under `encrypt_access_key` and `encrypt_source`, from
    /// `iv`, keeping the device ID's top byte.
    #[allow(clippy::too_many_arguments)]
    pub fn reencrypt_device_unique_data(
        &self,
        mode: u32,
        decrypt_access_key: &Key,
        decrypt_source: &Key,
        encrypt_access_key: &Key,
        encrypt_source: &Key,
        data: &mut [u8],
        device_id: u64,
        iv: &Key,
    ) -> Result<(), CryptoError> {
        check_device_unique_data(mode, data.len())?;
        let opening = self.unseal(SEAL_REENCRYPT_DEVICE_UNIQUE_DATA, decrypt_access_key);
        let high = open_device_unique_data(&decrypt(&opening, decrypt_source), data, device_id)
            .ok_or(CryptoError::InvalidArgument)?;
        let sealing = self.unseal(MODE_SEALS[mode as usize], encrypt_access_key);
        let payload = &data[..data.len() - DEVICE_UNIQUE_DATA_OVERHEAD];
        let id = device_id | u64::from(high) << 56;
        let sealed = seal_device_unique_data(&decrypt(&sealing, encrypt_source), iv, payload, id);
        data.copy_from_slice(&sealed);
        Ok(())
    }

    /// `ModularExponentiate`: the caller's own key, `exponent` of any
    /// length up to the block's.
    pub fn modular_exponentiate(
        message: &RsaBlock,
        exponent: &[u8],
        modulus: &RsaBlock,
    ) -> Result<RsaBlock, CryptoError> {
        exponentiate(message, exponent, modulus).ok_or(CryptoError::Unimplemented(
            "an RSA exponentiation with a zero modulus",
        ))
    }

    /// `ModularExponentiateByStorageKey` in `mode` (below
    /// [`STORAGE_KEY_MODES`]): `message` to the stored private exponent.
    /// A key whose modulus is still provisional takes `modulus` and must
    /// first turn a test block of `'D'`s through its public exponent 65537
    /// and back (`TestRsaPublicKey`, `TestRsaPrivateKey`); the monitor
    /// refuses the operation if it does not come back.
    pub fn exponentiate_by_storage_key(
        &mut self,
        mode: u32,
        message: &RsaBlock,
        modulus: &RsaBlock,
    ) -> Result<RsaBlock, CryptoError> {
        let stored = &mut self.rsa_keys[STORAGE_KEY_BY_MODE[mode as usize]];
        if !stored.committed {
            stored.modulus = *modulus;
            let test = [b'D'; RSA_SIZE];
            stored.committed = exponentiate(&test, &[1, 0, 1], modulus)
                .and_then(|public| exponentiate(&public, &stored.exponent, modulus))
                == Some(test);
        }
        if !stored.committed {
            return Err(CryptoError::InvalidArgument);
        }
        Ok(exponentiate(message, &stored.exponent, &stored.modulus).expect("a tested modulus"))
    }

    /// Whether an ES device key has been imported, which
    /// `PrepareEsDeviceUniqueKey` needs.
    pub fn has_es_device_key(&self) -> bool {
        self.rsa_keys[RSA_KEY_ES_DEVICE].committed
    }

    /// The title key ES common key `kind` of `generation` opens from
    /// `source`: `D(D(master_key, common_key_source), source)`.
    fn decrypt_with_es_common_key(
        &self,
        kind: u32,
        generation: u64,
        source: &Key,
    ) -> Result<Key, CryptoError> {
        let master_key = master_key(&self.keys, generation)?;
        let common_source = match kind {
            0 => self
                .keys
                .titlekek_source()
                .ok_or(CryptoError::Unimplemented(
                    "titlekek_source is not in the key file",
                ))?,
            _ => {
                return Err(CryptoError::Unimplemented(
                    "the sources of ES common keys other than the title key's are not in key \
                     files",
                ));
            }
        };
        Ok(decrypt(&decrypt(master_key, common_source), source))
    }

    /// `PrepareEsDeviceUniqueKey`'s result: `message` (a personalized
    /// ticket's title key block) through the ES device key, OAEP-decoded
    /// under `label_digest` to a 16-byte source, opened with ES common
    /// key `kind` of `generation`, and sealed for `LoadPreparedAesKey`.
    pub fn prepare_es_device_unique_key(
        &self,
        message: &RsaBlock,
        label_digest: &[u8; 32],
        generation: u64,
        kind: u32,
    ) -> Result<Key, CryptoError> {
        let stored = &self.rsa_keys[RSA_KEY_ES_DEVICE];
        let block = exponentiate(message, &stored.exponent, &stored.modulus).ok_or(
            CryptoError::Unimplemented("an ES device key with a zero modulus"),
        )?;
        let source: Key = oaep_sha256_decode(&block, label_digest)
            .and_then(|source| source.try_into().ok())
            .ok_or(CryptoError::InvalidArgument)?;
        let key = self.decrypt_with_es_common_key(kind, generation, &source)?;
        Ok(encrypt(&self.seal_key(SEAL_PREPARED_ES_KEY), &key))
    }

    /// `PrepareEsCommonTitleKey`: a common ticket's title key `source`
    /// opened with the title key's common key of `generation`, sealed for
    /// `LoadPreparedAesKey`.
    pub fn prepare_es_common_title_key(
        &self,
        source: &Key,
        generation: u64,
    ) -> Result<Key, CryptoError> {
        let key = self.decrypt_with_es_common_key(0, generation, source)?;
        Ok(encrypt(&self.seal_key(SEAL_PREPARED_ES_KEY), &key))
    }

    /// `LoadPreparedAesKey`: loads what a prepared ES key seals into `slot`.
    pub fn load_prepared_aes_key(
        &mut self,
        slot: u64,
        access_key: &Key,
    ) -> Result<(), CryptoError> {
        let key = self.unseal(SEAL_PREPARED_ES_KEY, access_key);
        *self.user_slot(slot)? = Some(key);
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
        if which > 1 || !self.valid_generation(generation) {
            return Err(CryptoError::InvalidArgument);
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
             titlekek_source = 404142434445464748494a4b4c4d4e4f\n\
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
        let kek = |generation, option| crypto.generate_aes_kek(&[0; 16], generation, option);
        let refused = Err(CryptoError::InvalidArgument);
        // Reserved bits; a key type past the last; the recovery-only key
        // type in a normal boot; a seal key past the last; device-unique
        // keys of the second and third generations, which do not exist.
        for (generation, option) in [(1, 1 << 8), (1, 4 << 1), (1, 2 << 1), (1, 7 << 5), (3, 1)] {
            assert_eq!(kek(generation, option), refused, "{generation} {option:#x}");
        }
        // No device key in the key set; no source for the normal-only key
        // type in key files.
        for option in [1, 1 << 1] {
            assert!(matches!(kek(1, option), Err(CryptoError::Unimplemented(_))));
        }
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
        let mut crypto = crypto();
        let (kek_source, key_source, iv) = ([0x33; 16], [0x44; 16], [0x55; 16]);
        let device_id = 0x0012_3456_789A_BCDE;
        let option = kek_option(SEAL_DECRYPT_DEVICE_UNIQUE_DATA);
        let (kek, seal) = key_encryption_key(&crypto.keys, &kek_source, 0, option).unwrap();
        assert_eq!(seal, SEAL_DECRYPT_DEVICE_UNIQUE_DATA);
        // The static source is the one sealed for this operation: the
        // default source under that operation's mask.
        let master_key: Key = core::array::from_fn(|index| 0x10 + index as u8);
        let static_source: Key = core::array::from_fn(|index| {
            index as u8 ^ SEAL_KEY_MASKS[SEAL_DECRYPT_DEVICE_UNIQUE_DATA as usize][index]
        });
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
        let for_load_aes_key = crypto.generate_aes_kek(&kek_source, 0, 0).unwrap();
        let mut open = |access_key: &Key, data: &[u8], device_id| {
            let mut data = data.to_vec();
            crypto
                .decrypt_device_unique_data(0, access_key, &key_source, &mut data, device_id)
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
        assert_eq!(open(&for_load_aes_key, &sealed, device_id), refused);
        assert_eq!(open(&access_key, &sealed[..0x30], device_id), refused);
        assert_eq!(open(&access_key, &[0; 0x250], device_id), refused);
    }

    /// Wraps `payload` for this crypto's console the way the factory does
    /// for an operation `seal`s access keys open, with kek and key sources
    /// of `fill`; returns the access key with the data.
    fn wrap(
        crypto: &Crypto,
        seal: u32,
        fill: u8,
        payload: &[u8],
        device_id: u64,
    ) -> (Key, Vec<u8>) {
        let (kek_source, key_source) = ([fill; 16], [fill ^ 0xFF; 16]);
        let option = kek_option(seal);
        let key =
            device_unique_data_key(&crypto.keys, &kek_source, &key_source, 0, option).unwrap();
        let access_key = crypto.generate_aes_kek(&kek_source, 0, option).unwrap();
        (
            access_key,
            seal_device_unique_data(&key, &[fill; 16], payload, device_id),
        )
    }

    fn rsa_key() -> (RsaBlock, RsaBlock, BigUint) {
        use rand_chacha::{ChaCha8Rng, rand_core::SeedableRng};
        use rsa::traits::{PrivateKeyParts, PublicKeyParts};
        let key = rsa::RsaPrivateKey::new(&mut ChaCha8Rng::seed_from_u64(1), 2048).unwrap();
        let block = |value: &BigUint| {
            let bytes = value.to_bytes_be();
            let mut block = [0; RSA_SIZE];
            block[RSA_SIZE - bytes.len()..].copy_from_slice(&bytes);
            block
        };
        (block(key.d()), block(key.n()), key.n().clone())
    }

    #[test]
    fn stored_rsa_keys_are_tested_once_then_used_and_es_keys_reach_a_slot_sealed() {
        let mut crypto = crypto();
        let device_id = 0x0055_6677_8899_AABB;
        let (exponent, modulus, n) = rsa_key();
        let message: RsaBlock = core::array::from_fn(|index| index as u8 | 1);
        let message = exponentiate(&message, &[1], &modulus).unwrap();

        // The TLS key: its exponent only, the modulus comes with the first
        // use and must pass the test.
        let (access, data) = wrap(&crypto, SEAL_IMPORT_SSL_KEY, 0x31, &exponent, device_id);
        let mut opened = data.clone();
        let import = crypto.decrypt_device_unique_data(
            3,
            &access,
            &[0x31 ^ 0xFF; 16],
            &mut opened,
            device_id,
        );
        assert_eq!(import, Ok(()));
        let mut wrong = modulus;
        wrong[RSA_SIZE - 1] ^= 2;
        assert_eq!(
            crypto.exponentiate_by_storage_key(1, &message, &wrong),
            Err(CryptoError::InvalidArgument)
        );
        let signed = crypto
            .exponentiate_by_storage_key(1, &message, &modulus)
            .unwrap();
        assert_eq!(exponentiate(&signed, &[1, 0, 1], &modulus), Some(message));
        // Once tested, the stored modulus is the one used.
        assert_eq!(
            crypto.exponentiate_by_storage_key(1, &message, &[0; RSA_SIZE]),
            Ok(signed)
        );
        // The Lotus key was never imported.
        assert_eq!(
            crypto.exponentiate_by_storage_key(0, &message, &modulus),
            Err(CryptoError::InvalidArgument)
        );

        // The ES device key brings its modulus; a personalized title key
        // comes out sealed and loads into a slot as the common key opens it.
        assert!(!crypto.has_es_device_key());
        let mut payload = [exponent, modulus].concat();
        payload.extend_from_slice(&[0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let (access, mut data) = wrap(
            &crypto,
            SEAL_IMPORT_ES_DEVICE_KEY,
            0x32,
            &payload,
            device_id,
        );
        assert_eq!(data.len(), DEVICE_UNIQUE_DATA_MAX);
        let import =
            crypto.decrypt_device_unique_data(2, &access, &[0x32 ^ 0xFF; 16], &mut data, device_id);
        assert_eq!(import, Ok(()));
        assert!(crypto.has_es_device_key());
        let title_key_source = [0x77; 16];
        let public = rsa::RsaPublicKey::new(n, BigUint::from(65537u32)).unwrap();
        let mut rng =
            <rand_chacha::ChaCha8Rng as rand_chacha::rand_core::SeedableRng>::seed_from_u64(2);
        let block: RsaBlock = public
            .encrypt(&mut rng, rsa::Oaep::new::<Sha256>(), &title_key_source)
            .unwrap()
            .try_into()
            .unwrap();
        let empty_label: [u8; 32] = Sha256::digest([]).into();
        let access = crypto
            .prepare_es_device_unique_key(&block, &empty_label, 1, 0)
            .unwrap();
        crypto.load_prepared_aes_key(4, &access).unwrap();
        let master_key: Key = core::array::from_fn(|index| 0x10 + index as u8);
        let titlekek_source: Key = core::array::from_fn(|index| 0x40 + index as u8);
        let title_key = decrypt(&decrypt(&master_key, &titlekek_source), &title_key_source);
        assert_eq!(crypto.slot(4), Some(&title_key));
        assert_eq!(
            crypto.prepare_es_device_unique_key(&block, &[0; 32], 1, 0),
            Err(CryptoError::InvalidArgument),
            "another label"
        );
        // A common ticket's key the same way, without RSA.
        let access = crypto
            .prepare_es_common_title_key(&title_key_source, 1)
            .unwrap();
        crypto.load_prepared_aes_key(5, &access).unwrap();
        assert_eq!(crypto.slot(5), Some(&title_key));
        assert_eq!(
            crypto.prepare_es_common_title_key(&title_key_source, 4),
            Err(CryptoError::InvalidArgument),
            "a generation newer than the console's"
        );
    }

    #[test]
    fn reencryption_rewraps_for_another_operation_and_keeps_the_id_top_byte() {
        let mut crypto = crypto();
        let device_id = 0x0001_0203_0405_0607;
        let payload: Vec<u8> = (0..0x120).map(|index| index as u8).collect();
        let (from, data) = wrap(
            &crypto,
            SEAL_REENCRYPT_DEVICE_UNIQUE_DATA,
            0x41,
            &payload,
            device_id | 0x5C << 56,
        );
        let (to, _) = wrap(
            &crypto,
            SEAL_DECRYPT_DEVICE_UNIQUE_DATA,
            0x42,
            &[],
            device_id,
        );
        let mut rewrapped = data.clone();
        crypto
            .reencrypt_device_unique_data(
                0,
                &from,
                &[0x41 ^ 0xFF; 16],
                &to,
                &[0x42 ^ 0xFF; 16],
                &mut rewrapped,
                device_id,
                &[0x99; 16],
            )
            .unwrap();
        assert_eq!(rewrapped.len(), data.len());
        assert_eq!(rewrapped[..16], [0x99; 16]);
        crypto
            .decrypt_device_unique_data(0, &to, &[0x42 ^ 0xFF; 16], &mut rewrapped, device_id)
            .unwrap();
        assert_eq!(rewrapped[..0x120], payload);
        assert_eq!(rewrapped[0x128], 0x5C);
        // The data opens only with an access key sealed for re-encryption.
        let mut again = data.clone();
        assert_eq!(
            crypto.decrypt_device_unique_data(0, &from, &[0x41 ^ 0xFF; 16], &mut again, device_id),
            Err(CryptoError::InvalidArgument)
        );
    }
}
