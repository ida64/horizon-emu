//! Running programs Nintendo did not sign: the checks firmware 23.5.0 makes
//! before it runs a program, which of them a console setting turns off,
//! and the key replacement that passes the rest.
//!
//! # What stops a program
//!
//! The loader asks the filesystem process for a program's code
//! (`fsp-ldr` `OpenCodeFileSystem`), which opens the program's content
//! archive (NCA) as the content manager's database names it.
//!
//! 1. **NCA header signature 1** (filesystem process, `0x275e0`, called
//!    for every content archive it opens through `0x7a930`): RSA-2048-PSS
//!    with SHA-256 over the header's bytes `0x200..0x400`, the signature in
//!    `0x000..0x100`, under one of two fixed public keys chosen by the
//!    header's byte `0x221` (signature key generation; 2 and above are
//!    refused with result `0x237e02`). The keys are a table in its
//!    `.rodata`, retail ones at `0x1e3907` (generation 0) and `0x1e3a07`
//!    (generation 1), development ones at `0x1e3707`; which table depends
//!    on spl `IsDevelopment` (`0xab8`). A failure is result `0x234c02`. No
//!    setting skips it.
//! 2. **NCA hashes**: the filesystem process checks a section's
//!    hierarchical SHA-256 hashes as it reads it, and the section header
//!    against its hash in the archive header. Nothing skips this either;
//!    the archives built here have correct hashes.
//! 3. **ACID signature** (loader, `0x3fb0`): RSA-2048-PSS over the ACID
//!    from its byte `0x100` for its size field (`0x204`), under one of two
//!    fixed keys picked by bit 0 of the NPDM's byte 4: retail ones at
//!    `0x34904` and `0x34a04`, development ones at `0x34704` (`0x54e0`).
//!    Outside development hardware, the ACID must also carry its
//!    production flag (bit 0 of `0x20c`). A failure is result `0x1609`
//!    unless program verification is disabled (below); then the program
//!    runs, and its NCA signature 2 is not checked.
//! 4. **NCA header signature 2** (loader, `0x5730`, only when the ACID
//!    signature held): the filesystem process hands the loader the
//!    header's signature 2 (`0x100..0x200`) and the SHA-256 of its bytes
//!    `0x200..0x400` (`0x7cc30` and the header decoder `0x171670`), and the
//!    loader checks them against the public key the ACID itself carries
//!    (its bytes `0x100..0x200`). A failure is result `0x1809`.
//! 5. **ACI and ACID consistency** (loader, `0x5540`): the ACI's program
//!    ID within the ACID's range (else result `0x1209`), and its kernel
//!    capabilities within what the ACID grants. Its file system and
//!    service access go with the ACID's to the filesystem process and the
//!    service manager. These are properties of a well-made NPDM.
//! 6. **NSO hashes**: the loader checks each segment against the hash in
//!    the NSO header when the header's flags ask for it, as built NSOs do.
//!
//! # What turns checks off
//!
//! Secure monitor config item 1, DisableProgramVerification
//! ([`crate::boot::BootParams::program_verification_disabled`]), is read by
//! the loader alone (`0x518`), and kept as "verification disabled" only
//! together with spl `IsDevelopment` (`0x500`; spl text `0x11b40` answers
//! that from config item 6, hardware state, being 0, development): check 3
//! then lets a bad ACID signature pass (`0x437c`). The filesystem process
//! no longer has a switch at all: its program registry `fsp-pr` serves
//! commands 0 to 2 only (`0x25850`), without the old
//! `SetEnabledProgramVerification` (256), and the process manager sends it
//! nothing of the kind. This console reports retail hardware, as the
//! firmware's own contents need (on development hardware the filesystem
//! process would check their signature 1 against the development keys), so
//! the setting changes nothing here.
//!
//! # Key replacement
//!
//! Checks 1 and 3 survive, and pass only for archives and NPDMs signed with
//! the private halves of Nintendo's fixed keys. [`firmware`] replaces one
//! of those public keys in each of the two modules, as they sit in the
//! kernel image's INI1 before the console starts, with the key of
//! [`HOMEBREW_TEST_KEY`]: retail generation 1 of both tables, which no
//! archive or NPDM of the firmware uses (all of 23.5.0's are generation
//! 0), so everything Nintendo signed keeps passing. That is a modification
//! of the firmware's code, an explicit console option
//! (`--homebrew-keys`). [`program`] then signs what it builds with the same
//! key: the NCA's signature 1 (generation 1), the NPDM's ACID (generation
//! 1), and puts the key's public half into the ACID, which makes it the key
//! of signature 2.

pub mod firmware;
pub mod program;

use rsa::{
    Pss, RsaPrivateKey,
    pkcs1::DecodeRsaPrivateKey,
    rand_core::{CryptoRng, RngCore},
    traits::PublicKeyParts,
};
use sha2::{Digest, Sha256};

/// The RSA-2048 key everything here signs with, private half included: a
/// test key of this emulator's, made for it and published with it. It
/// proves nothing about what it signs.
pub const HOMEBREW_TEST_KEY: &str = include_str!("homebrew_test_key.pem");

/// The signature key generation (both the NCA header's and the NPDM's)
/// whose retail public key [`firmware`] replaces.
pub const KEY_GENERATION: u8 = 1;

pub const SIGNATURE_SIZE: usize = 0x100;

fn key() -> RsaPrivateKey {
    RsaPrivateKey::from_pkcs1_pem(HOMEBREW_TEST_KEY).expect("the test key is a PKCS #1 key")
}

/// The modulus of [`HOMEBREW_TEST_KEY`], big-endian, as the firmware keeps
/// its public keys (the exponent is 65537, as theirs).
pub fn modulus() -> [u8; SIGNATURE_SIZE] {
    let bytes = key().n().to_bytes_be();
    let mut modulus = [0; SIGNATURE_SIZE];
    modulus[SIGNATURE_SIZE - bytes.len()..].copy_from_slice(&bytes);
    modulus
}

/// Random numbers for signing that follow from the message, so that the
/// same input signs the same way every time: SHA-256 in counter mode.
struct MessageRng {
    seed: [u8; 32],
    counter: u64,
}

impl RngCore for MessageRng {
    fn next_u32(&mut self) -> u32 {
        let mut bytes = [0; 4];
        self.fill_bytes(&mut bytes);
        u32::from_le_bytes(bytes)
    }

    fn next_u64(&mut self) -> u64 {
        let mut bytes = [0; 8];
        self.fill_bytes(&mut bytes);
        u64::from_le_bytes(bytes)
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        for chunk in dest.chunks_mut(32) {
            let block: [u8; 32] = Sha256::new()
                .chain_update(self.seed)
                .chain_update(self.counter.to_le_bytes())
                .finalize()
                .into();
            self.counter += 1;
            chunk.copy_from_slice(&block[..chunk.len()]);
        }
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rsa::rand_core::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

impl CryptoRng for MessageRng {}

/// RSA-2048-PSS with SHA-256 and a 32-byte salt, the scheme of all the
/// checks above, over `message` with [`HOMEBREW_TEST_KEY`].
pub fn sign(message: &[u8]) -> [u8; SIGNATURE_SIZE] {
    let digest: [u8; 32] = Sha256::digest(message).into();
    let mut rng = MessageRng {
        seed: digest,
        counter: 0,
    };
    let signature = key()
        .sign_with_rng(&mut rng, Pss::new_with_salt::<Sha256>(32), &digest)
        .expect("a 2048-bit key signs a SHA-256 digest");
    signature.try_into().expect("a 2048-bit signature")
}

#[cfg(test)]
mod tests {
    use rsa::{RsaPublicKey, pss::Pss};

    use super::*;

    #[test]
    fn signatures_verify_under_the_published_modulus() {
        let public = RsaPublicKey::new(
            rsa::BigUint::from_bytes_be(&modulus()),
            rsa::BigUint::from(65537u32),
        )
        .unwrap();
        let message = b"homebrew";
        let digest = Sha256::digest(message);
        let signature = sign(message);
        assert_eq!(signature, sign(message), "deterministic");
        public
            .verify(Pss::new::<Sha256>(), &digest, &signature)
            .unwrap();
    }
}
