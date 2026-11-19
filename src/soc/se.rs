//! Security engine (SE): the SoC's AES, hash, RSA and random number engine
//! at `0x7001_2000`, as the boot code on the BPMP drives it.
//!
//! Sources: register names and addresses as Switchbrew's "Security Engine"
//! page lists them (the Erista column); bit fields as the public drivers
//! define them (Trusted Firmware-A `plat/nvidia/tegra/soc/t210/drivers/se`,
//! BSD; NVIDIA's Linux `drivers/crypto/tegra-se.h`); how the engine's
//! data paths combine, and the order of operations, from how those drivers
//! and the open-source Switch bootloaders (hekate `bdk/sec/se.c`, fusée)
//! program it. Facts are encoded here; no driver code is copied.
//!
//! # What the engine does here
//!
//! * Sixteen AES key slots of 256 bits of key plus an original IV (OIV) and
//!   an updated IV (UIV) each, written word by word through
//!   `SE_CRYPTO_KEYTABLE_ADDR`/`_DATA`. Each slot has an access register
//!   (`SE_CRYPTO_KEYTABLE_ACCESS_n`) of permissions that can only be taken
//!   away: key read, key update, OIV read, OIV update, UIV read, UIV
//!   update, key use (bits 0 to 6). `SE_CRYPTO_SECURITY_PERKEY` has one
//!   bit per slot that can likewise only be cleared. A key or IV word reads
//!   back through `_DATA` while its read permission lasts; once it is gone
//!   the word reads `0xFFFF_FFFF` if the slot's per-key bit is clear and 0
//!   otherwise, and a key or IV word whose update permission is gone
//!   ignores writes through `_DATA`. \[inference\] from Nintendo's warmboot
//!   firmware (package1's `warmboot.bin`, BPMP code): the secure monitor's
//!   cold boot takes the read permissions of all sixteen slots away and
//!   clears the per-key bits of slots 6 to 15 (TZRAM `0x1F01E1368`, the
//!   flags Atmosphère names `AllReadLock`, `KeyUse | PerKey`), and
//!   after SC7 the warmboot clears through `_DATA` the key of every slot
//!   but 2, 10-13 and 15 and the IVs of all, slots 6, 7 and 14 among them,
//!   whose update permissions the monitor took away, then reads every word
//!   of every slot (`0x4001_0D58`) and resets the SoC unless slots 0 to 5
//!   read 0 and slots 6 to 15 all ones. Other refusals of what an access
//!   register forbids: see "Refusals".
//! * AES-128/192/256 over `SE_CRYPTO_LAST_BLOCK + 1` blocks, built from the
//!   fields of `SE_CRYPTO_CONFIG` the way the hardware composes them: the
//!   AES core's input (memory, its previous output, or the linear
//!   counter), a vector (the memory block, the previous AES output, or the
//!   previous memory block; the slot's OIV or UIV before the first block)
//!   XORed before (`XOR_TOP`) or after (`XOR_BOTTOM`) the core. That gives
//!   ECB, CBC encryption and decryption, OFB and CTR as the drivers set
//!   them. With the hash bit set and the hash registers as destination, the
//!   last block goes to `SE_HASH_RESULT_0..3`: CBC-MAC, from which the
//!   drivers build CMAC (they compute the subkeys themselves).
//!   Chaining operations leave their next vector in the slot's UIV, and a
//!   counter operation leaves the advanced counter in
//!   `SE_CRYPTO_LINEAR_CTR` \[inference: hekate's CTR and CBC code finish a
//!   partial last block in a second operation that relies on both\].
//! * Results go to memory, to the hash registers, or into a key slot
//!   (`SE_CONFIG.DST = KEYTAB`, slot and word quad from
//!   `SE_CRYPTO_KEYTABLE_DST`): "decrypt into key slot", the key ladder's
//!   step, one block at a time.
//! * SHA-1, SHA-224, SHA-256, SHA-384 and SHA-512 over the input list, from
//!   the standard initial value (`SE_SHA_CONFIG` bit 0) or from the state
//!   in the hash registers. An operation whose input is exactly the
//!   remaining message length (`SE_SHA_MSG_LEFT`, bits) pads and finishes
//!   with the total length (`SE_SHA_MSG_LENGTH`); one with more left
//!   processes whole blocks and leaves the intermediate state. The state
//!   reads as 32-bit words, `H0` first (hekate byte-swaps the SHA-256
//!   words to get the digest). For SHA-384/512 each 64-bit word is two
//!   registers, high half first \[assumption: no public driver reads
//!   them\].
//! * Two RSA key slots of 2048-bit exponent and modulus, written and read
//!   word by word through `SE_RSA_KEYTABLE_ADDR`/`_DATA` (least significant
//!   word first, as the public drivers write them), with access registers
//!   of key read, key update and key use (bits 0 to 2); a read-protected
//!   word reads `0xFFFF_FFFF` if the slot's per-key bit
//!   (`SE_RSA_SECURITY_PERKEY`) is clear, as the warmboot firmware expects
//!   of both slots after the monitor's `0x1F01415A8`, and 0 otherwise.
//!   Modular exponentiation of 512 to 2048 bits (`SE_RSA_KEY_SIZE` 0 to 3,
//!   exponent `SE_RSA_EXP_SIZE` words) of the input list read as a
//!   little-endian number; the result in `SE_RSA_OUTPUT_0..63`, least
//!   significant word first.
//! * The random number generator (`SE_CONFIG.ENC_ALG = RNG`) to memory, into
//!   a key slot, or into the secure random key (SRK). Its numbers are
//!   ChaCha20 from the machine's seed (`--rng-seed`'s
//!   [`crate::boot::BootParams::rng_seed`]), not the hardware's
//!   entropy-seeded SP 800-90 DRBG: the same machine draws the same
//!   numbers every run, which is the point. `SE_RNG_CONFIG`, `_SRC_CONFIG`
//!   and `_RESEED_INTERVAL` are kept but change nothing.
//! * Linked lists in memory (`SE_IN_LL_ADDR`, `SE_OUT_LL_ADDR`): a word
//!   holding the index of the last entry, then entries of a 32-bit address
//!   and a 24-bit size; the data is their concatenation. Memory is whatever
//!   [`EngineMemory`] the engine is built with: [`BootMemory`] (IRAM and
//!   physical DRAM) in a machine that runs the boot chain, the engine's
//!   SMMU client ([`DeviceMemory`]) in one started at the kernel.
//! * An operation finishes as it starts: `SE_STATUS` reads idle and
//!   `SE_INT_STATUS` has `OP_DONE` (bit 4) set at once; `SE_INT_STATUS` and
//!   `SE_ERR_STATUS` clear by writing ones. No error is ever raised:
//!   whatever would end in one is refused instead.
//! * `SE_SECURITY`: the hard and per-key settings (bits 0, 2) can only be
//!   cleared (made secure) and the engine disable bit (1) only set, as the
//!   sticky bits the drivers treat them as \[assumption\]. The soft setting
//!   (bit 16) is an ordinary bit: the secure monitor sets it around its
//!   context save and clears it afterwards and at every boot (TZRAM
//!   `0x1F0142EF8`). Every requester that reaches this device is taken as
//!   secure: the CCPLEX's non-secure world is stopped by the bus first
//!   ([`super::apb_security`]), and the BPMP is a secure master
//!   \[inference\]. So the security registers, the per-key bits and
//!   `SE_TZRAM_SECURITY` hold what is written and gate nothing.
//! * Context save (`SE_OPERATION = CTX_SAVE`), what the secure monitor does
//!   before SC7 (see "Context save" below), and [`Se::restore_context`],
//!   what the boot ROM does on the warm boot after it.
//! * No reset input: a reset of the engine through the clock and reset
//!   controller (`RST_DEVICES_V` bit 31) keeps its key slots, IVs and
//!   access registers. Nintendo's warmboot firmware pulses that reset and
//!   then uses slots 2, 10-13 and 15 without loading them again (BPMP
//!   `0x40011038`, `warmboot-research` §7); what else the reset clears is
//!   not public, so nothing is cleared.
//!
//! Reset values: `SE_SECURITY` reads `0x0001_0005` (everything open), the
//! access registers every permission (`0x7F` for AES slots, `0x7` for RSA
//! slots, all bits of the per-key registers), key slots zero,
//! `SE_TZRAM_SECURITY` 1, every other register zero \[assumption: none of
//! these is public; the drivers only ever clear permission bits. The NX
//! bootloader's check of the state it starts in (text
//! `0x4000C4D8..0x4000C6CC`) agrees with every one of them it reads but
//! what runs before it changes: slots 14 and 15 read-protected (boot ROM),
//! slot 13's per-key bit clear (\[inference\]: the TSEC step, which installs
//! slot 13)\]. The AES access registers hold 8 bits (Trusted Firmware-A's
//! context layout saves bits 7:0); bit 7's meaning is not public, it gates
//! nothing and reads 0 from reset \[inference: the secure monitor's check
//! of the engine's settings, `0x1F01423EC` with its table at
//! `0x1F01480A8`, compares each access register's low half with a value
//! whose bit 7 is clear, after writes of its own that keep bit 7 as they
//! find it (`0x1F0141544`); a set bit 7 would be its panic at every boot\].
//!
//! # Refusals
//!
//! What the hardware does in these cases is not public, so the access is
//! refused with a note (the machine stops and says what): writing a key or
//! IV word whose update permission is gone (an RSA key word by register,
//! any word by an operation into the key table, or through
//! [`Se::load_key`] which returns an error instead; AES key and IV words
//! written through `_DATA` are ignored, see above), using an RSA slot
//! whose key use permission is gone, input or
//! output lists too short for the operation, restarts, and any field
//! combination not listed above, the context save's included. An AES
//! slot's key use permission gates nothing \[inference: the secure monitor
//! takes it from slots 8 to 15 (`0x1F01E1368`, flags `0x140`) and then
//! decrypts with slot 13's key into slot 12 there and runs every key of
//! slots 10 to 15 through the engine for its digest (`0x1F0143DD8`),
//! which a refusal would end in its panic; what the permission keeps from
//! a non-secure requester is not public, and every requester that reaches
//! this device is taken as secure\].
//!
//! # Context save
//!
//! Each context save operation encrypts one part of the engine's state, or
//! of memory, with the secure random key (SRK, drawn by an RNG operation
//! into `DST = SRK`), and writes it to the output list; `SE_CTX_SAVE_CONFIG`
//! picks the part (Trusted Firmware-A `se_private.h` field layout): source
//! (31:29) sticky bits (0, word quad in bit 24), RSA key table (1, key
//! index 17:16 = slot·2 + modulus, word quad 15:12), AES key table (2, slot
//! 11:8, word quad 1:0: key words 0-3, 4-7, OIV, UIV), memory (4,
//! `SE_CRYPTO_LAST_BLOCK + 1` blocks of the input list) or the SRK itself
//! (6), which goes to the PMC's `SECURE_SCRATCH4..7` (first word in 4,
//! little-endian) instead of memory, the way the boot ROM gets it ("Saves
//! SRK to PMC secure scratch registers for BootROM, which verifies and
//! restores the security engine context on warm boot", TF-A; hekate reads
//! those four registers back as the key). Key read protection does not
//! apply: saving protected slots is the point.
//!
//! The encryption is AES-128-CBC under the SRK, one chain across every
//! block saved since the SRK was drawn, starting from a zero vector
//! \[inference: hekate saves one block per slot and decrypts the
//! concatenation as a single CBC stream under the SRK\]. `SE_CONFIG` and
//! `SE_CRYPTO_CONFIG` do not select the cipher (the monitor saves with the
//! RNG's `SE_CRYPTO_CONFIG` still in place), but memory must be the
//! destination of every source but the SRK.
//!
//! The blob the secure monitor writes to `0x8000_F000` (TZRAM
//! `0x1F0141610`, firmware 23.5.0) and the boot ROM reads back
//! ([`SeContext::open`]), [`CONTEXT_SIZE`] bytes, decrypted:
//!
//! | Offset | Bytes | Contents |
//! |---|---|---|
//! | `0x000` | 16 | random block (memory source) |
//! | `0x010` | 32 | sticky bits, quads 0 and 1 |
//! | `0x030` | 16 × 32 | key words 0-7 of slots 0-15 |
//! | `0x230` | 16 × 16 | OIV of slots 0-15 |
//! | `0x330` | 16 × 16 | UIV of slots 0-15 |
//! | `0x430` | 2 × 512 | per RSA slot: modulus, then exponent, least significant word first |
//! | `0x830` | 16 | known pattern `00 01 .. 0f` (memory source) |
//!
//! Trusted Firmware-A's T210 code saves the same parts in a different
//! order (key, OIV and UIV per slot); the monitor's order is the one the
//! console's boot ROM accepts, so it is the one [`SeContext::open`] reads.
//! The sticky bits are packed as TF-A's `tegra_se_lp_sticky_bits_context_save`
//! lists them, bit `n` in word `n / 32` of the 256 (little-endian):
//! `SE_SECURITY` bits 2:0 at 2:0 and bit 16 at 3, `SE_TZRAM_SECURITY` 1:0
//! at 5:4, `SE_CRYPTO_SECURITY_PERKEY` 15:0 at 21:6, the AES access
//! registers 7:0 at `22 + 8·slot`, `SE_RSA_SECURITY_PERKEY` 1:0 at
//! 151:150, the RSA access registers 2:0 at 154:152 and 157:155, and
//! `SE_SECURITY` bit 4 at 158.
//!
//! What the boot ROM checks of the blob beyond its known pattern, and what
//! it does with a blob that fails, is not public: [`SeContext::open`]
//! checks the size and the pattern and refuses anything else.
//!
//! # Not shared with the HLE secure monitor
//!
//! [`crate::secmon`] keeps its own key slots and software crypto for the
//! kernel's `spl` requests. Sharing this engine would need the monitor to
//! drive it through [`Se`]'s registers (or a typed API over the same
//! state) with [`crate::machine`]'s engine memory, the monitor's key slot
//! numbering to match the hardware's (it uses slots as the firmware does),
//! and its asynchronous-operation bookkeeping to map onto `OP_DONE`.
//!
//! Key material is machine state: snapshots save it. Nothing here prints a
//! key or formats one into a note.

use std::sync::{Arc, Mutex};

use aes::{
    Aes128, Aes192, Aes256,
    cipher::{BlockDecrypt, BlockEncrypt, KeyInit, generic_array::GenericArray},
};
use rand_chacha::{
    ChaCha20Rng,
    rand_core::{RngCore, SeedableRng},
};
use rsa::BigUint;
use serde::{Deserialize, Serialize};

use super::{Notes, iram::Iram, mc::DeviceMemory, regs::word, tzram::Tzram};
use crate::{
    bus::MmioDevice,
    memory::GuestRam,
    snapshot::{Snapshot, SnapshotError, decode, encode},
};

pub const BASE: u64 = 0x7001_2000;
pub const SIZE: u64 = 0x2000;

const SECURITY: u64 = 0x000;
const TZRAM_SECURITY: u64 = 0x004;
const OPERATION: u64 = 0x008;
const INT_ENABLE: u64 = 0x00C;
const INT_STATUS: u64 = 0x010;
const CONFIG: u64 = 0x014;
const IN_LL_ADDR: u64 = 0x018;
const OUT_LL_ADDR: u64 = 0x024;
const HASH_RESULT: u64 = 0x030;
const HASH_RESULT_LAST: u64 = HASH_RESULT + 15 * 4;
const CTX_SAVE_CONFIG: u64 = 0x070;
const SHA_CONFIG: u64 = 0x200;
const SHA_MSG_LENGTH: u64 = 0x204;
const SHA_MSG_LENGTH_LAST: u64 = SHA_MSG_LENGTH + 3 * 4;
const SHA_MSG_LEFT: u64 = 0x214;
const SHA_MSG_LEFT_LAST: u64 = SHA_MSG_LEFT + 3 * 4;
const CRYPTO_SECURITY_PERKEY: u64 = 0x280;
const CRYPTO_KEYTABLE_ACCESS: u64 = 0x284;
const CRYPTO_KEYTABLE_ACCESS_LAST: u64 = CRYPTO_KEYTABLE_ACCESS + 15 * 4;
const CRYPTO_CONFIG: u64 = 0x304;
const CRYPTO_LINEAR_CTR: u64 = 0x308;
const CRYPTO_LINEAR_CTR_LAST: u64 = CRYPTO_LINEAR_CTR + 3 * 4;
const CRYPTO_LAST_BLOCK: u64 = 0x318;
const CRYPTO_KEYTABLE_ADDR: u64 = 0x31C;
const CRYPTO_KEYTABLE_DATA: u64 = 0x320;
const CRYPTO_KEYTABLE_DST: u64 = 0x330;
const RNG_CONFIG: u64 = 0x340;
const RNG_SRC_CONFIG: u64 = 0x344;
const RNG_RESEED_INTERVAL: u64 = 0x348;
const RSA_CONFIG: u64 = 0x400;
const RSA_KEY_SIZE: u64 = 0x404;
const RSA_EXP_SIZE: u64 = 0x408;
const RSA_SECURITY_PERKEY: u64 = 0x40C;
const RSA_KEYTABLE_ACCESS: u64 = 0x410;
const RSA_KEYTABLE_ACCESS_LAST: u64 = RSA_KEYTABLE_ACCESS + 4;
const RSA_KEYTABLE_ADDR: u64 = 0x420;
const RSA_KEYTABLE_DATA: u64 = 0x424;
const RSA_OUTPUT: u64 = 0x428;
const RSA_OUTPUT_LAST: u64 = RSA_OUTPUT + 63 * 4;
const STATUS: u64 = 0x800;
const ERR_STATUS: u64 = 0x804;
const MISC: u64 = 0x808;
const SPARE: u64 = 0x80C;
/// Three registers after `SE_ENTROPY_DEBUG_COUNTER` that Switchbrew does
/// not name. The NX bootloader checks them when it starts (its
/// hardware-state check, text `0x4000C6AA`, `0x4000C6BC`, `0x4000C6CC`):
/// `0x814` masked by `0x8F9F_FFFF` reads `0x8904_0800`, `0x818` masked by
/// `0x1F_0000` and `0x81C` masked by `0x101` read zero. They read just
/// that, the bits outside the masks zero \[assumption\]; what they hold
/// and what writing them does is not known, so writes are refused.
const UNNAMED_814: u64 = 0x814;
const UNNAMED_818: u64 = 0x818;
const UNNAMED_81C: u64 = 0x81C;
/// `SE_TZRAM_SECURITY` as the NX bootloader finds it when it starts
/// (bits 1:0 read 1, text `0x4000C4E4`): taken as its reset value, the
/// boot ROM not being known to write it \[inference\].
const TZRAM_SECURITY_RESET: u32 = 1;

/// `SE_SECURITY` bits that can only be cleared (hard and per-key
/// settings), and the engine disable bit, which can only be set.
const SECURITY_CLEAR_ONLY: u32 = 1 << 0 | 1 << 2;
const SECURITY_ENGINE_DISABLE: u32 = 1 << 1;
const SECURITY_SOFT: u32 = 1 << 16;
const SECURITY_RESET: u32 = SECURITY_CLEAR_ONLY | SECURITY_SOFT;

/// `SE_CTX_SAVE_CONFIG` sources (bits 31:29) and the fields each uses.
const CTX_SOURCE_STICKY: u32 = 0;
const CTX_SOURCE_RSA: u32 = 1;
const CTX_SOURCE_AES: u32 = 2;
const CTX_SOURCE_MEMORY: u32 = 4;
const CTX_SOURCE_SRK: u32 = 6;
const CTX_STICKY_FIELDS: u32 = 1 << 24;
const CTX_RSA_FIELDS: u32 = 0x3 << 16 | 0xF << 12;
const CTX_AES_FIELDS: u32 = 0xF << 8 | 0x3;

/// Bytes of a saved context ([`SeContext`]).
pub const CONTEXT_SIZE: usize = 0x840;
const CONTEXT_STICKY: usize = 0x010;
const CONTEXT_KEYS: usize = 0x030;
const CONTEXT_OIVS: usize = 0x230;
const CONTEXT_UIVS: usize = 0x330;
const CONTEXT_RSA: usize = 0x430;
const CONTEXT_PATTERN: usize = 0x830;
/// The last block of a saved context, as the secure monitor (TZRAM
/// `0x1F0148098`) and Trusted Firmware-A (`se_ctx_known_pattern_data`)
/// both write it.
const KNOWN_PATTERN: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

const OP_ABORT: u32 = 0;
const OP_START: u32 = 1;
const OP_RESTART_OUT: u32 = 2;
const OP_CTX_SAVE: u32 = 3;
const OP_RESTART_IN: u32 = 4;

const INT_OP_DONE: u32 = 1 << 4;

/// `SE_CONFIG` fields: destination (bits 4:2), decryption algorithm
/// (11:8), encryption algorithm (15:12), decryption mode (23:16),
/// encryption mode (31:24).
const CONFIG_FIELDS: u32 = 0x7 << 2 | 0xFF_FF_FF << 8;
const DST_MEMORY: u32 = 0;
const DST_HASHREG: u32 = 1;
const DST_KEYTABLE: u32 = 2;
const DST_SRK: u32 = 3;
const DST_RSAREG: u32 = 4;
const ALG_NOP: u32 = 0;
/// AES as encryption algorithm, and as decryption algorithm.
const ALG_AES: u32 = 1;
const ALG_RNG: u32 = 2;
const ALG_SHA: u32 = 3;
const ALG_RSA: u32 = 4;

/// `SE_CRYPTO_CONFIG` fields: hash enable (bit 0), XOR position (2:1),
/// input select (4:3), vector select (6:5), IV select (7), core select
/// (8), key schedule bypass (10), counter increment (18:11), key slot
/// (27:24), memory interface (31). Bits 9, 23:19 and 30:28 are not
/// public.
const CRYPTO_FIELDS: u32 = 0x1FF | 1 << 10 | 0xFF << 11 | 0xF << 24 | 1 << 31;
const CRYPTO_HASH: u32 = 1 << 0;
const CRYPTO_KEY_SCHEDULE_BYPASS: u32 = 1 << 10;
const XOR_BYPASS: u32 = 0;
const XOR_TOP: u32 = 2;
const XOR_BOTTOM: u32 = 3;
const INPUT_MEMORY: u32 = 0;
const INPUT_RANDOM: u32 = 1;
const INPUT_AES_OUT: u32 = 2;
const INPUT_COUNTER: u32 = 3;
const VECTOR_MEMORY: u32 = 0;
const VECTOR_AES_OUT: u32 = 2;
const VECTOR_PREVIOUS_MEMORY: u32 = 3;

/// `SE_CRYPTO_LAST_BLOCK`: 20 bits (Trusted Firmware-A's
/// `SE_MAX_LAST_BLOCK_SIZE`).
const LAST_BLOCK_MAX: u32 = 0xF_FFFF;
/// `SE_CRYPTO_KEYTABLE_ADDR`: word (3:0; 0-7 key, 8-11 OIV, 12-15 UIV)
/// and slot (7:4).
const KEYTABLE_ADDR_FIELDS: u32 = 0xFF;
/// `SE_CRYPTO_KEYTABLE_DST`: word quad (1:0) and slot (11:8).
const KEYTABLE_DST_FIELDS: u32 = 0x3 | 0xF << 8;
/// `SE_RSA_KEYTABLE_ADDR`: word (5:0), modulus rather than exponent (6),
/// slot (7), and the read/write bit (10) hekate's header names, which the
/// data register's direction makes redundant here. Bit 8 selects input by
/// DMA, which is not modelled.
const RSA_KEYTABLE_ADDR_FIELDS: u32 = 0xFF | 1 << 10;
/// `SE_RSA_CONFIG`: the key slot (bit 24).
const RSA_CONFIG_FIELDS: u32 = 1 << 24;

/// Words of an AES slot: key, then OIV, then UIV.
const SLOT_WORDS: usize = 16;
const OIV_WORD: usize = 8;
const UIV_WORD: usize = 12;
const AES_SLOTS: usize = 16;
const RSA_SLOTS: usize = 2;
/// 2048 bits.
const RSA_WORDS: usize = 64;

/// AES slot access register bits (permissions; clearing one revokes it).
const ACCESS_KEY_READ: u32 = 1 << 0;
const ACCESS_KEY_UPDATE: u32 = 1 << 1;
const ACCESS_OIV_READ: u32 = 1 << 2;
const ACCESS_OIV_UPDATE: u32 = 1 << 3;
const ACCESS_UIV_READ: u32 = 1 << 4;
const ACCESS_UIV_UPDATE: u32 = 1 << 5;
const ACCESS_KEY_USE: u32 = 1 << 6;
const ACCESS_RESET: u32 = 0x7F;
/// RSA slot access register bits.
const RSA_ACCESS_KEY_READ: u32 = 1 << 0;
const RSA_ACCESS_KEY_UPDATE: u32 = 1 << 1;
const RSA_ACCESS_KEY_USE: u32 = 1 << 2;
const RSA_ACCESS_RESET: u32 = 0x7;

/// Linked list entries the engine follows at most \[assumption: no limit
/// is public; the drivers use one entry\].
const LIST_ENTRIES_MAX: usize = 4096;
/// Bytes one list entry can name.
const LIST_ENTRY_SIZE_MAX: u32 = 0xFF_FFFF;

/// [`Se::load_key`] flags, in the encoding the public drivers pass to
/// their key slot lock functions: bits 0 to 6 revoke the access register
/// bits of the same number (if any of them is given, the register is
/// written with the complement of the flags), and [`LOCK_PER_KEY`] clears
/// the slot's bit of `SE_CRYPTO_SECURITY_PERKEY`.
pub const LOCK_KEY_READ: u32 = ACCESS_KEY_READ;
pub const LOCK_KEY_UPDATE: u32 = ACCESS_KEY_UPDATE;
pub const LOCK_OIV_READ: u32 = ACCESS_OIV_READ;
pub const LOCK_OIV_UPDATE: u32 = ACCESS_OIV_UPDATE;
pub const LOCK_UIV_READ: u32 = ACCESS_UIV_READ;
pub const LOCK_UIV_UPDATE: u32 = ACCESS_UIV_UPDATE;
pub const LOCK_KEY_USE: u32 = ACCESS_KEY_USE;
pub const LOCK_PER_KEY: u32 = 1 << 7;
const LOCK_ACCESS: u32 = 0x7F;

/// Memory as the engine, a bus master, reaches it: what its linked lists
/// and the buffers they name address (32-bit physical addresses: IRAM and
/// DRAM on the console).
pub trait EngineMemory: Send + Sync {
    /// `None` if any of the range cannot be read.
    fn read(&self, address: u32, buffer: &mut [u8]) -> Option<()>;
    /// `None` if any of the range cannot be written.
    fn write(&self, address: u32, data: &[u8]) -> Option<()>;
}

/// DRAM through the memory controller, as the engine's SMMU client sees it
/// (what the kernel's address spaces give it once it attaches the engine).
impl EngineMemory for DeviceMemory {
    fn read(&self, address: u32, buffer: &mut [u8]) -> Option<()> {
        DeviceMemory::read(self, address.into(), buffer)
    }

    fn write(&self, address: u32, data: &[u8]) -> Option<()> {
        DeviceMemory::write(self, address.into(), data)
    }
}

/// Memory as the engine reaches it while the boot chain runs: IRAM, TZRAM
/// and DRAM at its physical addresses. TZRAM is what the secure monitor
/// encrypts to DRAM before SC7 and the warmboot firmware decrypts back
/// (both through this engine, `warmboot-research` §4 and §7). The engine's
/// ASID register stays disabled until the kernel attaches it
/// ([`super::mc`]), and a client whose register is disabled is taken to
/// bypass the SMMU \[assumption\]. A range that is not all in one of them
/// cannot be reached.
pub struct BootMemory {
    iram: Arc<Iram>,
    tzram: Arc<Tzram>,
    dram: Arc<GuestRam>,
}

/// Where `len` bytes at `address` are relative to the region of `size`
/// bytes at `base`: `Some(Some(offset))` all in it, `Some(None)` not in
/// it, `None` partly.
fn place(address: u32, len: usize, base: u64, size: u64) -> Option<Option<u32>> {
    let start = u64::from(address);
    let end = start + len as u64;
    match ((base..base + size).contains(&start), end <= base + size) {
        (true, true) => Some(Some((start - base) as u32)),
        (true, false) => None,
        (false, _) => (end <= base || start >= base + size).then_some(None),
    }
}

impl BootMemory {
    pub fn new(iram: Arc<Iram>, tzram: Arc<Tzram>, dram: Arc<GuestRam>) -> Self {
        Self { iram, tzram, dram }
    }
}

impl EngineMemory for BootMemory {
    fn read(&self, address: u32, buffer: &mut [u8]) -> Option<()> {
        let len = buffer.len();
        if let Some(offset) = place(address, len, super::iram::BASE, super::iram::SIZE)? {
            return self.iram.copy_out(offset, buffer);
        }
        if let Some(offset) = place(address, len, super::tzram::BASE, super::tzram::SIZE)? {
            return self.tzram.copy_out(offset, buffer);
        }
        self.dram.read(address.into(), buffer)
    }

    fn write(&self, address: u32, data: &[u8]) -> Option<()> {
        let len = data.len();
        if let Some(offset) = place(address, len, super::iram::BASE, super::iram::SIZE)? {
            return self.iram.load(offset, data);
        }
        if let Some(offset) = place(address, len, super::tzram::BASE, super::tzram::SIZE)? {
            return self.tzram.load(offset, data);
        }
        self.dram.write(address.into(), data)
    }
}

/// Why [`Se::load_key`] did not load a key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SeError {
    /// There are 16 slots.
    NoSuchSlot(usize),
    /// Keys are 16, 24 or 32 bytes.
    KeySize(usize),
    /// The slot's key update permission is gone.
    Locked(usize),
}

impl std::fmt::Display for SeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoSuchSlot(slot) => write!(f, "security engine: no key slot {slot}"),
            Self::KeySize(size) => write!(f, "security engine: {size}-byte key"),
            Self::Locked(slot) => write!(f, "security engine: key slot {slot} is locked"),
        }
    }
}

impl std::error::Error for SeError {}

/// Offsets in the PMC (from `0x7000_E400`) of `SECURE_SCRATCH4..7`, where a
/// context save of the SRK puts its words (Trusted Firmware-A
/// `se_private.h`, `PMC_SECURE_SCRATCH4_OFFSET` to `7`).
pub const SRK_SCRATCH: [u64; 4] = [0x0C0, 0x0C4, 0x224, 0x228];

/// The SRK as the boot ROM finds it in the PMC: `SECURE_SCRATCH4..7`, in
/// that order, each little-endian (hekate reads the four registers and
/// loads them as key words in this order).
pub fn srk_from_scratch(words: [u32; 4]) -> [u8; 16] {
    words_to_block(&words)
}

/// A saved context, decrypted and checked: what the boot ROM puts back
/// into the engine on a warm boot ([`Se::restore_context`]). Its `Debug`
/// shows no key material.
pub struct SeContext {
    sticky: [u8; 32],
    /// Key, OIV and UIV words of each AES slot.
    slots: [[u32; SLOT_WORDS]; AES_SLOTS],
    /// Modulus and exponent of each RSA slot.
    rsa: [([u32; RSA_WORDS], [u32; RSA_WORDS]); RSA_SLOTS],
}

impl std::fmt::Debug for SeContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SeContext { .. }")
    }
}

/// Why [`SeContext::open`] refused a blob.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContextError {
    /// A context is [`CONTEXT_SIZE`] bytes.
    Size(usize),
    /// The last block does not decrypt to the known pattern: another SRK,
    /// or not a context.
    KnownPattern,
}

impl std::fmt::Display for ContextError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Size(size) => write!(
                f,
                "security engine context of {size:#x} bytes, not {CONTEXT_SIZE:#x}"
            ),
            Self::KnownPattern => {
                f.write_str("security engine context does not end in the known pattern")
            }
        }
    }
}

impl std::error::Error for ContextError {}

impl SeContext {
    /// Decrypts `blob` (AES-128-CBC under `srk`, zero IV, as context save
    /// encrypted it) and checks it ends in the known pattern; the layout is
    /// the module's "Context save" table.
    pub fn open(blob: &[u8], srk: &[u8; 16]) -> Result<Self, ContextError> {
        if blob.len() != CONTEXT_SIZE {
            return Err(ContextError::Size(blob.len()));
        }
        let cipher = Cipher::new(srk);
        let mut plain = blob.to_vec();
        let mut chain = [0; 16];
        for block in plain.as_chunks_mut::<16>().0 {
            let cipher_block = *block;
            cipher.run(false, block);
            xor_block(block, &chain);
            chain = cipher_block;
        }
        if plain[CONTEXT_PATTERN..] != KNOWN_PATTERN {
            return Err(ContextError::KnownPattern);
        }
        let words = |at: usize, out: &mut [u32]| {
            for (word, bytes) in out.iter_mut().zip(plain[at..].as_chunks::<4>().0) {
                *word = u32::from_le_bytes(*bytes);
            }
        };
        let mut slots = [[0; SLOT_WORDS]; AES_SLOTS];
        for (index, slot) in slots.iter_mut().enumerate() {
            words(CONTEXT_KEYS + 32 * index, &mut slot[..OIV_WORD]);
            words(CONTEXT_OIVS + 16 * index, &mut slot[OIV_WORD..UIV_WORD]);
            words(CONTEXT_UIVS + 16 * index, &mut slot[UIV_WORD..]);
        }
        let mut rsa = [([0; RSA_WORDS], [0; RSA_WORDS]); RSA_SLOTS];
        for (index, (modulus, exponent)) in rsa.iter_mut().enumerate() {
            let at = CONTEXT_RSA + 0x200 * index;
            words(at, modulus);
            words(at + 0x100, exponent);
        }
        Ok(Self {
            sticky: plain[CONTEXT_STICKY..CONTEXT_KEYS].try_into().unwrap(),
            slots,
            rsa,
        })
    }
}

type Refusal = String;

/// Takes the SRK's four words when a context save writes them to the
/// PMC's `SECURE_SCRATCH4..7`, in that order.
pub type SrkSink = Box<dyn Fn([u32; 4]) + Send + Sync>;

pub struct Se {
    memory: Box<dyn EngineMemory>,
    notes: Arc<Notes>,
    state: Mutex<State>,
    srk_sink: Mutex<Option<SrkSink>>,
}

impl Se {
    /// `memory` is what the engine's linked lists address; `seed` seeds its
    /// random number generator.
    pub fn new(memory: Box<dyn EngineMemory>, notes: Arc<Notes>, seed: u64) -> Self {
        Self {
            memory,
            notes,
            state: Mutex::new(State::new(seed)),
            srk_sink: Mutex::new(None),
        }
    }

    /// Has `sink` take the SRK when a context save sends it to the PMC.
    /// Until then such a save is refused.
    pub fn on_srk_saved(&self, sink: SrkSink) {
        *self.srk_sink.lock().unwrap() = Some(sink);
    }

    /// Puts the engine's state back from a saved context, as the boot ROM
    /// does on a warm boot: every key slot's key, OIV and UIV, both RSA
    /// slots, and the sticky bits (access and per-key registers, the
    /// `SE_SECURITY` and `SE_TZRAM_SECURITY` bits they hold). Nothing else
    /// changes.
    pub fn restore_context(&self, context: &SeContext) {
        let mut state = self.state.lock().unwrap();
        for (slot, words) in state.slots.iter_mut().zip(&context.slots) {
            slot.words = *words;
        }
        for (slot, (modulus, exponent)) in state.rsa_slots.iter_mut().zip(&context.rsa) {
            slot.modulus = *modulus;
            slot.exponent = *exponent;
        }
        state.set_sticky_bits(&context.sticky);
    }

    /// Writes `key` (16, 24 or 32 bytes) into the key words of `slot`, as
    /// the boot ROM loads the secure boot key and secure storage key, then
    /// locks it as `flags` say ([`LOCK_KEY_READ`] and the other `LOCK_`
    /// constants). The key's words are written as through the key table
    /// registers: little-endian, the rest of the slot unchanged.
    pub fn load_key(&self, slot: usize, key: &[u8], flags: u32) -> Result<(), SeError> {
        if slot >= AES_SLOTS {
            return Err(SeError::NoSuchSlot(slot));
        }
        if !matches!(key.len(), 16 | 24 | 32) {
            return Err(SeError::KeySize(key.len()));
        }
        let mut state = self.state.lock().unwrap();
        let target = &mut state.slots[slot];
        if target.access & ACCESS_KEY_UPDATE == 0 {
            return Err(SeError::Locked(slot));
        }
        for (word, bytes) in target.words.iter_mut().zip(key.as_chunks::<4>().0) {
            *word = u32::from_le_bytes(*bytes);
        }
        if flags & LOCK_ACCESS != 0 {
            target.access &= !flags;
        }
        if flags & LOCK_PER_KEY != 0 {
            state.crypto_perkey &= !(1 << slot);
        }
        Ok(())
    }

    /// Whether the engine's interrupt output is asserted: an enabled
    /// `SE_INT_STATUS` bit is set. For whoever wires the engine to an
    /// interrupt controller; the boot code polls instead.
    pub fn interrupt_pending(&self) -> bool {
        let state = self.state.lock().unwrap();
        state.int_status & state.int_enable != 0
    }

    fn served<T: Default>(&self, result: Result<T, Refusal>) -> Option<T> {
        match result {
            Ok(value) => Some(value),
            Err(note) => {
                self.notes.add(format!("security engine: {note}"));
                self.notes.survey().then(T::default)
            }
        }
    }
}

impl MmioDevice for Se {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        let result = word(offset, size)
            .ok_or_else(|| format!("{size}-byte read at {offset:#x}"))
            .and_then(|offset| self.state.lock().unwrap().read(offset));
        self.served(result).map(u64::from)
    }

    fn write(&self, _core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        let result = word(offset, size)
            .ok_or_else(|| format!("{size}-byte write at {offset:#x}"))
            .and_then(|offset| {
                self.state
                    .lock()
                    .unwrap()
                    .write(self.memory.as_ref(), offset, value as u32)
            })
            .and_then(|srk| {
                let Some(words) = srk else {
                    return Ok(());
                };
                let sink = self.srk_sink.lock().unwrap();
                let sink = sink
                    .as_ref()
                    .ok_or("context save of the SRK with no PMC to take it")?;
                sink(words);
                Ok(())
            });
        self.served(result)
    }

    fn snapshot(&self) -> Option<&dyn Snapshot> {
        Some(self)
    }
}

/// Saves the key slots with the rest: they are machine state.
impl Snapshot for Se {
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        encode(&*self.state.lock().unwrap())
    }

    fn restore(&self, state: &[u8]) -> Result<(), SnapshotError> {
        *self.state.lock().unwrap() = decode("security engine", state)?;
        Ok(())
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct AesSlot {
    /// Key (0-7), OIV (8-11), UIV (12-15).
    words: [u32; SLOT_WORDS],
    access: u32,
}

impl AesSlot {
    fn bytes(&self, word: usize, len: usize) -> Vec<u8> {
        self.words[word..word + len / 4]
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect()
    }

    fn block(&self, word: usize) -> [u8; 16] {
        self.bytes(word, 16).try_into().unwrap()
    }

    fn set_block(&mut self, word: usize, block: &[u8; 16]) {
        for (word, bytes) in self.words[word..word + 4]
            .iter_mut()
            .zip(block.as_chunks::<4>().0)
        {
            *word = u32::from_le_bytes(*bytes);
        }
    }
}

/// The access bits that let `word` of a slot be read, and updated.
fn permissions(word: usize) -> (u32, u32) {
    match word {
        0..OIV_WORD => (ACCESS_KEY_READ, ACCESS_KEY_UPDATE),
        OIV_WORD..UIV_WORD => (ACCESS_OIV_READ, ACCESS_OIV_UPDATE),
        _ => (ACCESS_UIV_READ, ACCESS_UIV_UPDATE),
    }
}

/// What a word whose read permission is gone reads, by the per-key bits
/// (`perkey`) of its slot (see the module documentation).
fn protected_word(perkey: u32, slot: usize) -> u32 {
    if perkey & 1 << slot == 0 {
        0xFFFF_FFFF
    } else {
        0
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct RsaSlot {
    /// Least significant word first, as are the modulus's.
    #[serde(with = "crate::snapshot::array")]
    exponent: [u32; RSA_WORDS],
    #[serde(with = "crate::snapshot::array")]
    modulus: [u32; RSA_WORDS],
    access: u32,
}

/// The random number generator's position in its stream.
#[derive(Serialize, Deserialize)]
struct Drbg {
    seed: [u8; 32],
    /// Words of the stream drawn so far.
    drawn: u128,
}

impl Drbg {
    fn fill(&mut self, out: &mut [u8]) {
        let mut stream = ChaCha20Rng::from_seed(self.seed);
        stream.set_word_pos(self.drawn);
        stream.fill_bytes(out);
        self.drawn = stream.get_word_pos();
    }
}

#[derive(Serialize, Deserialize)]
struct State {
    security: u32,
    tzram_security: u32,
    int_enable: u32,
    int_status: u32,
    config: u32,
    in_list: u32,
    out_list: u32,
    hash: [u32; 16],
    ctx_save_config: u32,
    sha_config: u32,
    msg_length: [u32; 4],
    msg_left: [u32; 4],
    crypto_perkey: u32,
    slots: [AesSlot; AES_SLOTS],
    crypto_config: u32,
    linear_counter: [u32; 4],
    last_block: u32,
    keytable_addr: u32,
    keytable_dst: u32,
    rng_config: u32,
    rng_src_config: u32,
    rng_reseed_interval: u32,
    rsa_config: u32,
    rsa_key_size: u32,
    rsa_exp_size: u32,
    rsa_perkey: u32,
    rsa_slots: [RsaSlot; RSA_SLOTS],
    rsa_keytable_addr: u32,
    #[serde(with = "crate::snapshot::array")]
    rsa_output: [u32; RSA_WORDS],
    err_status: u32,
    misc: u32,
    spare: u32,
    /// The secure random key, once generated.
    srk: Option<[u8; 16]>,
    /// The context save's CBC chaining vector: the last block it wrote
    /// since the SRK was drawn, zero before the first.
    context_chain: [u8; 16],
    drbg: Drbg,
}

/// Index of `offset` in the register array starting at `first`.
fn index(offset: u64, first: u64) -> usize {
    ((offset - first) / 4) as usize
}

impl State {
    fn new(seed: u64) -> Self {
        Self {
            security: SECURITY_RESET,
            tzram_security: TZRAM_SECURITY_RESET,
            int_enable: 0,
            int_status: 0,
            config: 0,
            in_list: 0,
            out_list: 0,
            hash: [0; 16],
            ctx_save_config: 0,
            sha_config: 0,
            msg_length: [0; 4],
            msg_left: [0; 4],
            crypto_perkey: (1 << AES_SLOTS) - 1,
            slots: std::array::from_fn(|_| AesSlot {
                words: [0; SLOT_WORDS],
                access: ACCESS_RESET,
            }),
            crypto_config: 0,
            linear_counter: [0; 4],
            last_block: 0,
            keytable_addr: 0,
            keytable_dst: 0,
            rng_config: 0,
            rng_src_config: 0,
            rng_reseed_interval: 0,
            rsa_config: 0,
            rsa_key_size: 0,
            rsa_exp_size: 0,
            rsa_perkey: (1 << RSA_SLOTS) - 1,
            rsa_slots: std::array::from_fn(|_| RsaSlot {
                exponent: [0; RSA_WORDS],
                modulus: [0; RSA_WORDS],
                access: RSA_ACCESS_RESET,
            }),
            rsa_keytable_addr: 0,
            rsa_output: [0; RSA_WORDS],
            err_status: 0,
            misc: 0,
            spare: 0,
            srk: None,
            context_chain: [0; 16],
            drbg: Drbg {
                seed: ChaCha20Rng::seed_from_u64(seed).get_seed(),
                drawn: 0,
            },
        }
    }

    /// The slot and word `SE_CRYPTO_KEYTABLE_ADDR` selects.
    fn key_word(&self) -> (usize, usize) {
        (
            (self.keytable_addr >> 4 & 0xF) as usize,
            (self.keytable_addr & 0xF) as usize,
        )
    }

    /// The slot, whether the modulus (rather than the exponent), and the
    /// word `SE_RSA_KEYTABLE_ADDR` selects.
    fn rsa_key_word(&self) -> (usize, bool, usize) {
        let address = self.rsa_keytable_addr;
        (
            (address >> 7 & 1) as usize,
            address & 1 << 6 != 0,
            (address & 0x3F) as usize,
        )
    }

    fn read(&self, offset: u64) -> Result<u32, Refusal> {
        Ok(match offset {
            SECURITY => self.security,
            TZRAM_SECURITY => self.tzram_security,
            INT_ENABLE => self.int_enable,
            INT_STATUS => self.int_status,
            CONFIG => self.config,
            IN_LL_ADDR => self.in_list,
            OUT_LL_ADDR => self.out_list,
            HASH_RESULT..=HASH_RESULT_LAST => self.hash[index(offset, HASH_RESULT)],
            CTX_SAVE_CONFIG => self.ctx_save_config,
            SHA_CONFIG => self.sha_config,
            SHA_MSG_LENGTH..=SHA_MSG_LENGTH_LAST => self.msg_length[index(offset, SHA_MSG_LENGTH)],
            SHA_MSG_LEFT..=SHA_MSG_LEFT_LAST => self.msg_left[index(offset, SHA_MSG_LEFT)],
            CRYPTO_SECURITY_PERKEY => self.crypto_perkey,
            CRYPTO_KEYTABLE_ACCESS..=CRYPTO_KEYTABLE_ACCESS_LAST => {
                self.slots[index(offset, CRYPTO_KEYTABLE_ACCESS)].access
            }
            CRYPTO_CONFIG => self.crypto_config,
            CRYPTO_LINEAR_CTR..=CRYPTO_LINEAR_CTR_LAST => {
                self.linear_counter[index(offset, CRYPTO_LINEAR_CTR)]
            }
            CRYPTO_LAST_BLOCK => self.last_block,
            CRYPTO_KEYTABLE_ADDR => self.keytable_addr,
            CRYPTO_KEYTABLE_DATA => {
                let (slot, word) = self.key_word();
                let (read, _) = permissions(word);
                if self.slots[slot].access & read == 0 {
                    protected_word(self.crypto_perkey, slot)
                } else {
                    self.slots[slot].words[word]
                }
            }
            CRYPTO_KEYTABLE_DST => self.keytable_dst,
            RNG_CONFIG => self.rng_config,
            RNG_SRC_CONFIG => self.rng_src_config,
            RNG_RESEED_INTERVAL => self.rng_reseed_interval,
            RSA_CONFIG => self.rsa_config,
            RSA_KEY_SIZE => self.rsa_key_size,
            RSA_EXP_SIZE => self.rsa_exp_size,
            RSA_SECURITY_PERKEY => self.rsa_perkey,
            RSA_KEYTABLE_ACCESS..=RSA_KEYTABLE_ACCESS_LAST => {
                self.rsa_slots[index(offset, RSA_KEYTABLE_ACCESS)].access
            }
            RSA_KEYTABLE_ADDR => self.rsa_keytable_addr,
            RSA_KEYTABLE_DATA => {
                let (slot, modulus, word) = self.rsa_key_word();
                let key = &self.rsa_slots[slot];
                if key.access & RSA_ACCESS_KEY_READ == 0 {
                    protected_word(self.rsa_perkey, slot)
                } else if modulus {
                    key.modulus[word]
                } else {
                    key.exponent[word]
                }
            }
            RSA_OUTPUT..=RSA_OUTPUT_LAST => self.rsa_output[index(offset, RSA_OUTPUT)],
            // Every operation is over as soon as it starts.
            STATUS => 0,
            ERR_STATUS => self.err_status,
            MISC => self.misc,
            SPARE => self.spare,
            UNNAMED_814 => 0x8904_0800,
            UNNAMED_818 | UNNAMED_81C => 0,
            _ => return Err(format!("read of unmodelled register {offset:#x}")),
        })
    }

    /// Writes a register. A context save of the SRK answers the four words
    /// it sends to the PMC.
    fn write(
        &mut self,
        memory: &dyn EngineMemory,
        offset: u64,
        value: u32,
    ) -> Result<Option<[u32; 4]>, Refusal> {
        let mut srk = None;
        match offset {
            SECURITY => {
                self.security = value & !(SECURITY_CLEAR_ONLY | SECURITY_ENGINE_DISABLE)
                    | value & self.security & SECURITY_CLEAR_ONLY
                    | (value | self.security) & SECURITY_ENGINE_DISABLE;
            }
            TZRAM_SECURITY => self.tzram_security = value,
            OPERATION => srk = self.operation(memory, value)?,
            INT_ENABLE => self.int_enable = value,
            INT_STATUS => self.int_status &= !value,
            CONFIG => self.config = value,
            IN_LL_ADDR => self.in_list = value,
            OUT_LL_ADDR => self.out_list = value,
            HASH_RESULT..=HASH_RESULT_LAST => self.hash[index(offset, HASH_RESULT)] = value,
            CTX_SAVE_CONFIG => self.ctx_save_config = value,
            SHA_CONFIG => self.sha_config = value,
            SHA_MSG_LENGTH..=SHA_MSG_LENGTH_LAST => {
                self.msg_length[index(offset, SHA_MSG_LENGTH)] = value;
            }
            SHA_MSG_LEFT..=SHA_MSG_LEFT_LAST => {
                self.msg_left[index(offset, SHA_MSG_LEFT)] = value;
            }
            CRYPTO_SECURITY_PERKEY => self.crypto_perkey &= value,
            CRYPTO_KEYTABLE_ACCESS..=CRYPTO_KEYTABLE_ACCESS_LAST => {
                self.slots[index(offset, CRYPTO_KEYTABLE_ACCESS)].access &= value;
            }
            CRYPTO_CONFIG => self.crypto_config = value,
            CRYPTO_LINEAR_CTR..=CRYPTO_LINEAR_CTR_LAST => {
                self.linear_counter[index(offset, CRYPTO_LINEAR_CTR)] = value;
            }
            CRYPTO_LAST_BLOCK => {
                if value > LAST_BLOCK_MAX {
                    return Err(format!(
                        "SE_CRYPTO_LAST_BLOCK {value:#x} is wider than 20 bits"
                    ));
                }
                self.last_block = value;
            }
            CRYPTO_KEYTABLE_ADDR => {
                if value & !KEYTABLE_ADDR_FIELDS != 0 {
                    return Err(format!(
                        "SE_CRYPTO_KEYTABLE_ADDR {value:#x}: unmodelled bits"
                    ));
                }
                self.keytable_addr = value;
            }
            CRYPTO_KEYTABLE_DATA => {
                let (slot, word) = self.key_word();
                let (_, update) = permissions(word);
                // A word whose update permission is gone keeps its value
                // (see the module documentation).
                if self.slots[slot].access & update != 0 {
                    self.slots[slot].words[word] = value;
                }
            }
            CRYPTO_KEYTABLE_DST => {
                if value & !KEYTABLE_DST_FIELDS != 0 {
                    return Err(format!(
                        "SE_CRYPTO_KEYTABLE_DST {value:#x}: unmodelled bits"
                    ));
                }
                self.keytable_dst = value;
            }
            RNG_CONFIG => self.rng_config = value,
            RNG_SRC_CONFIG => self.rng_src_config = value,
            RNG_RESEED_INTERVAL => self.rng_reseed_interval = value,
            RSA_CONFIG => self.rsa_config = value,
            RSA_KEY_SIZE => self.rsa_key_size = value,
            RSA_EXP_SIZE => self.rsa_exp_size = value,
            RSA_SECURITY_PERKEY => self.rsa_perkey &= value,
            RSA_KEYTABLE_ACCESS..=RSA_KEYTABLE_ACCESS_LAST => {
                self.rsa_slots[index(offset, RSA_KEYTABLE_ACCESS)].access &= value;
            }
            RSA_KEYTABLE_ADDR => {
                if value & !RSA_KEYTABLE_ADDR_FIELDS != 0 {
                    return Err(format!("SE_RSA_KEYTABLE_ADDR {value:#x}: unmodelled bits"));
                }
                self.rsa_keytable_addr = value;
            }
            RSA_KEYTABLE_DATA => {
                let (slot, modulus, word) = self.rsa_key_word();
                let target = &mut self.rsa_slots[slot];
                if target.access & RSA_ACCESS_KEY_UPDATE == 0 {
                    return Err(format!(
                        "write to RSA key slot {slot}, which its access register forbids: \
                         what the engine does with it is not modelled"
                    ));
                }
                if modulus {
                    target.modulus[word] = value;
                } else {
                    target.exponent[word] = value;
                }
            }
            ERR_STATUS => self.err_status &= !value,
            MISC => self.misc = value,
            SPARE => self.spare = value,
            _ => {
                return Err(format!(
                    "write of {value:#x} to unmodelled register {offset:#x}"
                ));
            }
        }
        Ok(srk)
    }

    fn operation(
        &mut self,
        memory: &dyn EngineMemory,
        value: u32,
    ) -> Result<Option<[u32; 4]>, Refusal> {
        let srk = match value {
            // Nothing is ever in flight to abort.
            OP_ABORT => return Ok(None),
            OP_START => {
                self.start(memory)?;
                None
            }
            OP_CTX_SAVE => self.context_save(memory)?,
            OP_RESTART_OUT | OP_RESTART_IN => {
                return Err(format!("SE_OPERATION {value}: restarts are not modelled"));
            }
            _ => return Err(format!("SE_OPERATION {value:#x} is not an operation")),
        };
        self.int_status |= INT_OP_DONE;
        Ok(srk)
    }

    /// One context save operation (see the module's "Context save"): the
    /// SRK's words for the PMC if that is the source, else `None` and the
    /// encrypted part in the output list.
    fn context_save(&mut self, memory: &dyn EngineMemory) -> Result<Option<[u32; 4]>, Refusal> {
        let config = self.ctx_save_config;
        let source = config >> 29;
        let fields = match source {
            CTX_SOURCE_STICKY => CTX_STICKY_FIELDS,
            CTX_SOURCE_RSA => CTX_RSA_FIELDS,
            CTX_SOURCE_AES => CTX_AES_FIELDS,
            CTX_SOURCE_MEMORY | CTX_SOURCE_SRK => 0,
            _ => {
                return Err(format!(
                    "context save from source {source} (SE_CTX_SAVE_CONFIG {config:#x}): \
                     not one of this engine's"
                ));
            }
        };
        if config & !(0x7 << 29 | fields) != 0 {
            return Err(format!("SE_CTX_SAVE_CONFIG {config:#x}: unmodelled bits"));
        }
        let Some(srk) = self.srk else {
            return Err("context save before an SRK was drawn".into());
        };
        if source == CTX_SOURCE_SRK {
            return Ok(Some(block_to_words(&srk)));
        }
        let destination = self.config >> 2 & 0x7;
        if destination != DST_MEMORY {
            return Err(format!(
                "context save to destination {destination}: only memory is modelled"
            ));
        }
        let mut data = match source {
            CTX_SOURCE_STICKY => {
                let quad = (config >> 24 & 1) as usize;
                self.sticky_bits()[quad * 16..quad * 16 + 16].to_vec()
            }
            CTX_SOURCE_RSA => {
                let key = (config >> 16 & 0x3) as usize;
                let quad = (config >> 12 & 0xF) as usize;
                let slot = &self.rsa_slots[key / 2];
                let words = if key % 2 == 1 {
                    &slot.modulus
                } else {
                    &slot.exponent
                };
                words[quad * 4..quad * 4 + 4]
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect()
            }
            CTX_SOURCE_AES => {
                let slot = (config >> 8 & 0xF) as usize;
                let quad = (config & 0x3) as usize;
                self.slots[slot].block(quad * 4).to_vec()
            }
            _ => gather(
                memory,
                self.in_list,
                Some((self.last_block as usize + 1) * 16),
            )?,
        };
        let cipher = Cipher::new(&srk);
        for block in data.as_chunks_mut::<16>().0 {
            xor_block(block, &self.context_chain);
            cipher.run(true, block);
            self.context_chain = *block;
        }
        scatter(memory, self.out_list, &data)?;
        Ok(None)
    }

    /// The 256 sticky bits a context save saves (see the module's "Context
    /// save"), little-endian.
    fn sticky_bits(&self) -> [u8; 32] {
        let mut bits = [0u32; 8];
        let mut put = |at: usize, len: usize, value: u32| {
            for bit in 0..len {
                if value >> bit & 1 != 0 {
                    bits[(at + bit) / 32] |= 1 << ((at + bit) % 32);
                }
            }
        };
        put(0, 3, self.security);
        put(3, 1, self.security >> 16);
        put(4, 2, self.tzram_security);
        put(6, 16, self.crypto_perkey);
        for (index, slot) in self.slots.iter().enumerate() {
            put(22 + 8 * index, 8, slot.access);
        }
        put(150, 2, self.rsa_perkey);
        put(152, 3, self.rsa_slots[0].access);
        put(155, 3, self.rsa_slots[1].access);
        put(158, 1, self.security >> 4);
        let mut bytes = [0; 32];
        for (chunk, word) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(bits) {
            *chunk = word.to_le_bytes();
        }
        bytes
    }

    /// Takes back what [`State::sticky_bits`] packed. The `SE_SECURITY`
    /// bits it does not hold keep their value.
    fn set_sticky_bits(&mut self, bytes: &[u8; 32]) {
        let get = |at: usize, len: usize| -> u32 {
            (0..len).fold(0, |value, bit| {
                let at = at + bit;
                value | u32::from(bytes[at / 8] >> (at % 8) & 1) << bit
            })
        };
        let held = 0x7 | 1 << 4 | SECURITY_SOFT;
        self.security = self.security & !held | get(0, 3) | get(3, 1) << 16 | get(158, 1) << 4;
        self.tzram_security = get(4, 2);
        self.crypto_perkey = get(6, 16);
        for index in 0..AES_SLOTS {
            self.slots[index].access = get(22 + 8 * index, 8);
        }
        self.rsa_perkey = get(150, 2);
        self.rsa_slots[0].access = get(152, 3);
        self.rsa_slots[1].access = get(155, 3);
    }

    fn start(&mut self, memory: &dyn EngineMemory) -> Result<(), Refusal> {
        if self.security & SECURITY_ENGINE_DISABLE != 0 {
            return Err("operation started with the engine disabled (SE_SECURITY bit 1)".into());
        }
        let config = self.config;
        if config & !CONFIG_FIELDS != 0 {
            return Err(format!("SE_CONFIG {config:#x}: unmodelled bits"));
        }
        let destination = config >> 2 & 0x7;
        let decryption = config >> 8 & 0xF;
        let encryption = config >> 12 & 0xF;
        let decryption_mode = config >> 16 & 0xFF;
        let encryption_mode = config >> 24 & 0xFF;
        match (encryption, decryption) {
            (ALG_AES, ALG_NOP) => self.aes(memory, true, encryption_mode, destination),
            (ALG_NOP, ALG_AES) => self.aes(memory, false, decryption_mode, destination),
            (ALG_RNG, ALG_NOP) => self.random(memory, destination),
            (ALG_SHA, ALG_NOP) => self.sha(memory, encryption_mode, destination),
            (ALG_RSA, ALG_NOP) => self.rsa(memory, destination),
            _ => Err(format!("SE_CONFIG {config:#x}: no modelled algorithm")),
        }
    }

    fn aes(
        &mut self,
        memory: &dyn EngineMemory,
        encrypt: bool,
        mode: u32,
        destination: u32,
    ) -> Result<(), Refusal> {
        let crypto = self.crypto_config;
        let refuse = |what: &str| Err(format!("AES with SE_CRYPTO_CONFIG {crypto:#x}: {what}"));
        if crypto & !CRYPTO_FIELDS != 0 {
            return refuse("unmodelled bits");
        }
        if crypto & CRYPTO_KEY_SCHEDULE_BYPASS != 0 {
            return refuse("key schedule bypass is not modelled");
        }
        let hash = crypto & CRYPTO_HASH != 0;
        let xor = crypto >> 1 & 0x3;
        let input = crypto >> 3 & 0x3;
        let vector_select = crypto >> 5 & 0x3;
        let updated_iv = crypto >> 7 & 1 != 0;
        let core_encrypts = crypto >> 8 & 1 != 0;
        let step = u128::from(crypto >> 11 & 0xFF);
        let slot = (crypto >> 24 & 0xF) as usize;
        if core_encrypts != encrypt {
            return refuse("the core select disagrees with SE_CONFIG's algorithm");
        }
        if xor == 1 || vector_select == 1 || input == INPUT_RANDOM {
            return refuse("unmodelled XOR position, vector or input");
        }
        let key_len = match mode {
            0 => 16,
            1 => 24,
            2 => 32,
            _ => return Err(format!("AES key mode {mode} is not one")),
        };
        let xors = xor != XOR_BYPASS;
        let reads_memory = input == INPUT_MEMORY
            || xors && matches!(vector_select, VECTOR_MEMORY | VECTOR_PREVIOUS_MEMORY);
        let blocks = self.last_block as usize + 1;
        let data = if reads_memory {
            gather(memory, self.in_list, Some(blocks * 16))?
        } else {
            Vec::new()
        };
        let key_slot = &self.slots[slot];
        let cipher = Cipher::new(&key_slot.bytes(0, key_len));
        let iv = key_slot.block(if updated_iv { UIV_WORD } else { OIV_WORD });
        let mut counter = u128::from_be_bytes(words_to_block(&self.linear_counter));
        let mut previous_out = iv;
        let mut previous_in = iv;
        let mut output = Vec::with_capacity(blocks * 16);
        for index in 0..blocks {
            let from_memory: [u8; 16] = if reads_memory {
                data[index * 16..index * 16 + 16].try_into().unwrap()
            } else {
                [0; 16]
            };
            let mut block = match input {
                INPUT_MEMORY => from_memory,
                INPUT_AES_OUT => previous_out,
                _ => counter.to_be_bytes(),
            };
            let vector = match vector_select {
                VECTOR_MEMORY => from_memory,
                VECTOR_AES_OUT => previous_out,
                _ => previous_in,
            };
            if xor == XOR_TOP {
                xor_block(&mut block, &vector);
            }
            cipher.run(encrypt, &mut block);
            previous_out = block;
            if xor == XOR_BOTTOM {
                xor_block(&mut block, &vector);
            }
            previous_in = from_memory;
            counter = counter.wrapping_add(step);
            output.extend_from_slice(&block);
        }
        match (destination, hash) {
            (DST_MEMORY, false) => scatter(memory, self.out_list, &output)?,
            (DST_HASHREG, true) => {
                let last = &output[output.len() - 16..];
                for (word, bytes) in self.hash.iter_mut().zip(last.as_chunks::<4>().0) {
                    *word = u32::from_le_bytes(*bytes);
                }
            }
            (DST_KEYTABLE, false) => self.write_key_table(&output)?,
            _ => {
                return Err(format!(
                    "AES result to destination {destination} with the hash bit {}: \
                     not modelled",
                    u32::from(hash)
                ));
            }
        }
        // The vector the next block would have used.
        let chained = if xors && vector_select == VECTOR_PREVIOUS_MEMORY {
            Some(previous_in)
        } else if xors && vector_select == VECTOR_AES_OUT || input == INPUT_AES_OUT {
            Some(previous_out)
        } else {
            None
        };
        if let Some(vector) = chained {
            self.slots[slot].set_block(UIV_WORD, &vector);
        }
        if input == INPUT_COUNTER {
            self.linear_counter = block_to_words(&counter.to_be_bytes());
        }
        Ok(())
    }

    /// Puts an operation's one-block result where `SE_CRYPTO_KEYTABLE_DST`
    /// says.
    fn write_key_table(&mut self, output: &[u8]) -> Result<(), Refusal> {
        let slot = (self.keytable_dst >> 8 & 0xF) as usize;
        let quad = (self.keytable_dst & 0x3) as usize;
        let Ok(block) = <[u8; 16]>::try_from(output) else {
            return Err(format!(
                "{} blocks into key slot {slot}: only one is modelled",
                output.len() / 16
            ));
        };
        let word = quad * 4;
        let (_, update) = permissions(word);
        if self.slots[slot].access & update == 0 {
            return Err(format!(
                "result into key slot {slot} word {word}, which its access register forbids"
            ));
        }
        self.slots[slot].set_block(word, &block);
        Ok(())
    }

    fn random(&mut self, memory: &dyn EngineMemory, destination: u32) -> Result<(), Refusal> {
        if destination == DST_SRK {
            let mut srk = [0; 16];
            self.drbg.fill(&mut srk);
            self.srk = Some(srk);
            // A new key starts a new chain.
            self.context_chain = [0; 16];
            return Ok(());
        }
        let crypto = self.crypto_config;
        if crypto >> 3 & 0x3 != INPUT_RANDOM
            || crypto >> 1 & 0x3 != XOR_BYPASS
            || crypto & CRYPTO_HASH != 0
        {
            return Err(format!(
                "random numbers with SE_CRYPTO_CONFIG {crypto:#x}: only the random input, \
                 unhashed and without XOR, is modelled"
            ));
        }
        let mut output = vec![0; (self.last_block as usize + 1) * 16];
        match destination {
            DST_MEMORY => {
                self.drbg.fill(&mut output);
                scatter(memory, self.out_list, &output)
            }
            DST_KEYTABLE => {
                self.drbg.fill(&mut output);
                self.write_key_table(&output)
            }
            _ => Err(format!(
                "random numbers to destination {destination}: not modelled"
            )),
        }
    }

    fn sha(
        &mut self,
        memory: &dyn EngineMemory,
        mode: u32,
        destination: u32,
    ) -> Result<(), Refusal> {
        let Some(sha) = Sha::from_mode(mode) else {
            return Err(format!("SHA mode {mode} is not one"));
        };
        if destination != DST_HASHREG {
            return Err(format!("SHA to destination {destination}: not modelled"));
        }
        if self.sha_config & !1 != 0 {
            return Err(format!(
                "SE_SHA_CONFIG {:#x}: unmodelled bits",
                self.sha_config
            ));
        }
        let data = gather(memory, self.in_list, None)?;
        let bits = data.len() as u128 * 8;
        let left = words_to_u128(&self.msg_left);
        let total = words_to_u128(&self.msg_length);
        let last = match left.cmp(&bits) {
            std::cmp::Ordering::Equal => true,
            std::cmp::Ordering::Greater => false,
            std::cmp::Ordering::Less => {
                return Err(format!(
                    "SHA over {bits} bits of input with {left} bits of message left: \
                     not modelled"
                ));
            }
        };
        if last && total < bits {
            return Err(format!(
                "SHA message of {total} bits ending in {bits} bits of input"
            ));
        }
        if !last && !data.len().is_multiple_of(sha.block_size()) {
            return Err(format!(
                "SHA continued after {} bytes, not whole blocks",
                data.len()
            ));
        }
        sha.run(
            &mut self.hash,
            self.sha_config & 1 != 0,
            &data,
            last.then_some(total),
        );
        Ok(())
    }

    fn rsa(&mut self, memory: &dyn EngineMemory, destination: u32) -> Result<(), Refusal> {
        if destination != DST_RSAREG {
            return Err(format!("RSA to destination {destination}: not modelled"));
        }
        if self.rsa_config & !RSA_CONFIG_FIELDS != 0 {
            return Err(format!(
                "SE_RSA_CONFIG {:#x}: unmodelled bits",
                self.rsa_config
            ));
        }
        let slot = (self.rsa_config >> 24 & 1) as usize;
        let width = match self.rsa_key_size {
            size @ 0..=3 => (size as usize + 1) * 16,
            size => return Err(format!("SE_RSA_KEY_SIZE {size} is not a key width")),
        };
        let exponent_words = self.rsa_exp_size as usize;
        if !(1..=RSA_WORDS).contains(&exponent_words) {
            return Err(format!("SE_RSA_EXP_SIZE {exponent_words}: not modelled"));
        }
        let key = &self.rsa_slots[slot];
        if key.access & RSA_ACCESS_KEY_USE == 0 {
            return Err(format!(
                "RSA with key slot {slot}, whose access register forbids its use"
            ));
        }
        let modulus = BigUint::from_slice(&key.modulus[..width]);
        if modulus == BigUint::default() {
            return Err(format!("RSA with key slot {slot}, whose modulus is zero"));
        }
        let exponent = BigUint::from_slice(&key.exponent[..exponent_words]);
        let input = gather(memory, self.in_list, Some(width * 4))?;
        let result = BigUint::from_bytes_le(&input).modpow(&exponent, &modulus);
        self.rsa_output = [0; RSA_WORDS];
        for (word, bytes) in self
            .rsa_output
            .iter_mut()
            .zip(result.to_bytes_le().chunks(4))
        {
            let mut le = [0; 4];
            le[..bytes.len()].copy_from_slice(bytes);
            *word = u32::from_le_bytes(le);
        }
        Ok(())
    }
}

fn xor_block(block: &mut [u8; 16], vector: &[u8; 16]) {
    for (byte, other) in block.iter_mut().zip(vector) {
        *byte ^= other;
    }
}

/// Register words to the bytes they hold, as a driver's `memcpy` into
/// them would have put them.
fn words_to_block(words: &[u32; 4]) -> [u8; 16] {
    let mut block = [0; 16];
    for (bytes, word) in block.as_chunks_mut::<4>().0.iter_mut().zip(words) {
        *bytes = word.to_le_bytes();
    }
    block
}

fn block_to_words(block: &[u8; 16]) -> [u32; 4] {
    std::array::from_fn(|index| u32::from_le_bytes(block[index * 4..][..4].try_into().unwrap()))
}

/// A 128-bit number spread over four registers, least significant first.
fn words_to_u128(words: &[u32; 4]) -> u128 {
    words
        .iter()
        .rev()
        .fold(0, |value, &word| value << 32 | u128::from(word))
}

fn read_word(memory: &dyn EngineMemory, address: u32) -> Result<u32, Refusal> {
    let mut bytes = [0; 4];
    memory
        .read(address, &mut bytes)
        .ok_or_else(|| format!("linked list word at {address:#x} cannot be read"))?;
    Ok(u32::from_le_bytes(bytes))
}

/// The entries of the linked list at `list`: `(address, size)`.
fn entries(
    memory: &dyn EngineMemory,
    list: u32,
) -> Result<impl Iterator<Item = Result<(u32, u32), Refusal>> + '_, Refusal> {
    if list == 0 {
        return Err("operation without its linked list (address 0)".into());
    }
    let count = read_word(memory, list)? as usize + 1;
    if count > LIST_ENTRIES_MAX {
        return Err(format!("linked list at {list:#x} of {count} entries"));
    }
    Ok((0..count).map(move |index| {
        let entry = (index as u32)
            .checked_mul(8)
            .and_then(|offset| list.checked_add(4 + offset))
            .ok_or_else(|| format!("linked list at {list:#x} runs past 4 GiB"))?;
        let address = read_word(memory, entry)?;
        let size = read_word(memory, entry + 4)?;
        if size > LIST_ENTRY_SIZE_MAX {
            return Err(format!("linked list entry of {size:#x} bytes"));
        }
        Ok((address, size))
    }))
}

/// The data the input list at `list` names, or its first `need` bytes.
fn gather(memory: &dyn EngineMemory, list: u32, need: Option<usize>) -> Result<Vec<u8>, Refusal> {
    let mut data = Vec::with_capacity(need.unwrap_or(0));
    for entry in entries(memory, list)? {
        if need.is_some_and(|need| data.len() >= need) {
            break;
        }
        let (address, size) = entry?;
        let size = need.map_or(size as usize, |need| (size as usize).min(need - data.len()));
        let start = data.len();
        data.resize(start + size, 0);
        memory
            .read(address, &mut data[start..])
            .ok_or_else(|| format!("input of {size:#x} bytes at {address:#x} cannot be read"))?;
    }
    if let Some(need) = need
        && data.len() < need
    {
        return Err(format!(
            "input list at {list:#x} holds {:#x} bytes where the operation needs {need:#x}",
            data.len()
        ));
    }
    Ok(data)
}

/// Writes `data` over the buffers the output list at `list` names.
fn scatter(memory: &dyn EngineMemory, list: u32, mut data: &[u8]) -> Result<(), Refusal> {
    let total = data.len();
    for entry in entries(memory, list)? {
        if data.is_empty() {
            break;
        }
        let (address, size) = entry?;
        let (piece, rest) = data.split_at((size as usize).min(data.len()));
        memory.write(address, piece).ok_or_else(|| {
            format!(
                "output of {:#x} bytes at {address:#x} cannot be written",
                piece.len()
            )
        })?;
        data = rest;
    }
    if !data.is_empty() {
        return Err(format!(
            "output list at {list:#x} holds {:#x} of the {total:#x} bytes of the result",
            total - data.len()
        ));
    }
    Ok(())
}

enum Cipher {
    Aes128(Aes128),
    Aes192(Aes192),
    Aes256(Aes256),
}

impl Cipher {
    /// `key` is 16, 24 or 32 bytes.
    fn new(key: &[u8]) -> Self {
        match key.len() {
            16 => Self::Aes128(Aes128::new(GenericArray::from_slice(key))),
            24 => Self::Aes192(Aes192::new(GenericArray::from_slice(key))),
            _ => Self::Aes256(Aes256::new(GenericArray::from_slice(key))),
        }
    }

    fn run(&self, encrypt: bool, block: &mut [u8; 16]) {
        let block = GenericArray::from_mut_slice(block);
        match (self, encrypt) {
            (Self::Aes128(cipher), true) => cipher.encrypt_block(block),
            (Self::Aes128(cipher), false) => cipher.decrypt_block(block),
            (Self::Aes192(cipher), true) => cipher.encrypt_block(block),
            (Self::Aes192(cipher), false) => cipher.decrypt_block(block),
            (Self::Aes256(cipher), true) => cipher.encrypt_block(block),
            (Self::Aes256(cipher), false) => cipher.decrypt_block(block),
        }
    }
}

/// Initial hash values (FIPS 180-4, 5.3).
const SHA1_IV: [u32; 5] = [
    0x6745_2301,
    0xEFCD_AB89,
    0x98BA_DCFE,
    0x1032_5476,
    0xC3D2_E1F0,
];
const SHA224_IV: [u32; 8] = [
    0xC105_9ED8,
    0x367C_D507,
    0x3070_DD17,
    0xF70E_5939,
    0xFFC0_0B31,
    0x6858_1511,
    0x64F9_8FA7,
    0xBEFA_4FA4,
];
const SHA256_IV: [u32; 8] = [
    0x6A09_E667,
    0xBB67_AE85,
    0x3C6E_F372,
    0xA54F_F53A,
    0x510E_527F,
    0x9B05_688C,
    0x1F83_D9AB,
    0x5BE0_CD19,
];
const SHA384_IV: [u64; 8] = [
    0xCBBB_9D5D_C105_9ED8,
    0x629A_292A_367C_D507,
    0x9159_015A_3070_DD17,
    0x152F_ECD8_F70E_5939,
    0x6733_2667_FFC0_0B31,
    0x8EB4_4A87_6858_1511,
    0xDB0C_2E0D_64F9_8FA7,
    0x47B5_481D_BEFA_4FA4,
];
const SHA512_IV: [u64; 8] = [
    0x6A09_E667_F3BC_C908,
    0xBB67_AE85_84CA_A73B,
    0x3C6E_F372_FE94_F82B,
    0xA54F_F53A_5F1D_36F1,
    0x510E_527F_ADE6_82D1,
    0x9B05_688C_2B3E_6C1F,
    0x1F83_D9AB_FB41_BD6B,
    0x5BE0_CD19_137E_2179,
];

/// The hash `SE_CONFIG.ENC_MODE` selects.
#[derive(Clone, Copy)]
enum Sha {
    Sha1,
    Sha224,
    Sha256,
    Sha384,
    Sha512,
}

impl Sha {
    fn from_mode(mode: u32) -> Option<Self> {
        Some(match mode {
            0 => Self::Sha1,
            4 => Self::Sha224,
            5 => Self::Sha256,
            6 => Self::Sha384,
            7 => Self::Sha512,
            _ => return None,
        })
    }

    fn block_size(self) -> usize {
        match self {
            Self::Sha384 | Self::Sha512 => 128,
            _ => 64,
        }
    }

    /// Runs the compression function over `data`, from the initial value
    /// if `init` and from the state in `hash` otherwise, and leaves the
    /// state in `hash`. With `total_bits`, `data` is the end of the
    /// message, padded with the message's length.
    fn run(self, hash: &mut [u32; 16], init: bool, data: &[u8], total_bits: Option<u128>) {
        let size = self.block_size();
        let whole = data.len() / size * size;
        let mut tail = Vec::new();
        if let Some(total) = total_bits {
            tail.extend_from_slice(&data[whole..]);
            tail.push(0x80);
            // The length field is an eighth of a block.
            let length = size / 8;
            tail.resize((tail.len() + length).div_ceil(size) * size - length, 0);
            if size == 64 {
                tail.extend_from_slice(&(total as u64).to_be_bytes());
            } else {
                tail.extend_from_slice(&total.to_be_bytes());
            }
        }
        let blocks = data[..whole]
            .chunks_exact(size)
            .chain(tail.chunks_exact(size));
        match self {
            Self::Sha1 => {
                let mut state = if init {
                    SHA1_IV
                } else {
                    hash[..5].try_into().unwrap()
                };
                for block in blocks {
                    sha1::compress(
                        &mut state,
                        std::slice::from_ref(GenericArray::from_slice(block)),
                    );
                }
                hash[..5].copy_from_slice(&state);
            }
            Self::Sha224 | Self::Sha256 => {
                let mut state = match (init, self) {
                    (true, Self::Sha224) => SHA224_IV,
                    (true, _) => SHA256_IV,
                    (false, _) => hash[..8].try_into().unwrap(),
                };
                for block in blocks {
                    sha2::compress256(
                        &mut state,
                        std::slice::from_ref(GenericArray::from_slice(block)),
                    );
                }
                hash[..8].copy_from_slice(&state);
            }
            Self::Sha384 | Self::Sha512 => {
                let mut state = match (init, self) {
                    (true, Self::Sha384) => SHA384_IV,
                    (true, _) => SHA512_IV,
                    (false, _) => std::array::from_fn(|index| {
                        u64::from(hash[2 * index]) << 32 | u64::from(hash[2 * index + 1])
                    }),
                };
                for block in blocks {
                    sha2::compress512(
                        &mut state,
                        std::slice::from_ref(GenericArray::from_slice(block)),
                    );
                }
                for (words, value) in hash.as_chunks_mut::<2>().0.iter_mut().zip(state) {
                    *words = [(value >> 32) as u32, value as u32];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const IRAM: u32 = 0x4000_0000;
    /// Where the tests put their lists and buffers.
    const IN_LIST: u32 = IRAM + 0x100;
    const OUT_LIST: u32 = IRAM + 0x200;
    const INPUT: u32 = IRAM + 0x1000;
    const OUTPUT: u32 = IRAM + 0x8000;

    /// 64 KiB of memory at `IRAM`.
    struct Ram(Mutex<Vec<u8>>);

    impl EngineMemory for Arc<Ram> {
        fn read(&self, address: u32, buffer: &mut [u8]) -> Option<()> {
            let start = address.checked_sub(IRAM)? as usize;
            buffer.copy_from_slice(self.0.lock().unwrap().get(start..start + buffer.len())?);
            Some(())
        }

        fn write(&self, address: u32, data: &[u8]) -> Option<()> {
            let start = address.checked_sub(IRAM)? as usize;
            self.0
                .lock()
                .unwrap()
                .get_mut(start..start + data.len())?
                .copy_from_slice(data);
            Some(())
        }
    }

    fn hex(text: &str) -> Vec<u8> {
        (0..text.len())
            .step_by(2)
            .map(|at| u8::from_str_radix(&text[at..at + 2], 16).unwrap())
            .collect()
    }

    fn config(encryption: u32, decryption: u32, mode: u32, destination: u32) -> u32 {
        let mode_shift = if decryption != ALG_NOP { 16 } else { 24 };
        destination << 2 | decryption << 8 | encryption << 12 | mode << mode_shift
    }

    const ENCRYPT: u32 = 1 << 8;
    const UPDATED_IV: u32 = 1 << 7;
    const ECB_ENCRYPT: u32 = ENCRYPT;
    const ECB_DECRYPT: u32 = 0;
    const CBC_ENCRYPT: u32 = ENCRYPT | VECTOR_AES_OUT << 5 | XOR_TOP << 1;
    const CBC_DECRYPT: u32 = VECTOR_PREVIOUS_MEMORY << 5 | XOR_BOTTOM << 1;
    const CTR: u32 = ENCRYPT | INPUT_COUNTER << 3 | XOR_BOTTOM << 1 | 1 << 11;

    struct Bench {
        se: Se,
        ram: Arc<Ram>,
        notes: Arc<Notes>,
        /// What the engine sent to `SECURE_SCRATCH4..7`.
        scratch: Arc<Mutex<Option<[u32; 4]>>>,
    }

    impl Bench {
        fn new() -> Self {
            let ram = Arc::new(Ram(Mutex::new(vec![0; 0x1_0000])));
            let notes: Arc<Notes> = Arc::default();
            let scratch: Arc<Mutex<Option<[u32; 4]>>> = Arc::default();
            let se = Se::new(Box::new(Arc::clone(&ram)), Arc::clone(&notes), 7);
            let sink = Arc::clone(&scratch);
            se.on_srk_saved(Box::new(move |words| *sink.lock().unwrap() = Some(words)));
            Self {
                se,
                ram,
                notes,
                scratch,
            }
        }

        fn write(&self, offset: u64, value: u32) {
            self.se
                .write(0, offset, 4, value.into())
                .unwrap_or_else(|| panic!("write to {offset:#x}: {:?}", self.notes.take()));
        }

        fn read(&self, offset: u64) -> u32 {
            self.se
                .read(0, offset, 4)
                .unwrap_or_else(|| panic!("read of {offset:#x}: {:?}", self.notes.take()))
                as u32
        }

        fn poke(&self, address: u32, data: &[u8]) {
            Arc::clone(&self.ram).write(address, data).unwrap();
        }

        fn peek(&self, address: u32, len: usize) -> Vec<u8> {
            let mut data = vec![0; len];
            Arc::clone(&self.ram).read(address, &mut data).unwrap();
            data
        }

        /// Writes a linked list of `entries` at `list`.
        fn list(&self, list: u32, entries: &[(u32, u32)]) {
            let mut bytes = (entries.len() as u32 - 1).to_le_bytes().to_vec();
            for (address, size) in entries {
                bytes.extend_from_slice(&address.to_le_bytes());
                bytes.extend_from_slice(&size.to_le_bytes());
            }
            self.poke(list, &bytes);
        }

        /// Writes `data` into `slot` from `word` on through the key table.
        fn key_words(&self, slot: u32, word: u32, data: &[u8]) {
            for (index, bytes) in data.as_chunks::<4>().0.iter().enumerate() {
                self.write(CRYPTO_KEYTABLE_ADDR, slot << 4 | (word + index as u32));
                self.write(CRYPTO_KEYTABLE_DATA, u32::from_le_bytes(*bytes));
            }
        }

        fn start(&self) {
            self.write(OPERATION, OP_START);
            assert_eq!(self.read(INT_STATUS) & INT_OP_DONE, INT_OP_DONE);
            assert_eq!(self.read(STATUS), 0);
            assert_eq!(self.read(ERR_STATUS), 0);
            self.write(INT_STATUS, self.read(INT_STATUS));
        }

        /// Runs the configured operation over `input` (whole blocks, unless
        /// not AES), its result to memory.
        fn run(&self, input: &[u8], output_len: usize) -> Vec<u8> {
            self.poke(INPUT, input);
            self.list(IN_LIST, &[(INPUT, input.len() as u32)]);
            self.list(OUT_LIST, &[(OUTPUT, output_len as u32)]);
            self.write(IN_LL_ADDR, IN_LIST);
            self.write(OUT_LL_ADDR, OUT_LIST);
            self.start();
            self.peek(OUTPUT, output_len)
        }

        fn aes(&self, slot: u32, mode: u32, crypto: u32, input: &[u8]) -> Vec<u8> {
            let config = if crypto & ENCRYPT != 0 {
                config(ALG_AES, ALG_NOP, mode, DST_MEMORY)
            } else {
                config(ALG_NOP, ALG_AES, mode, DST_MEMORY)
            };
            self.write(CONFIG, config);
            self.write(CRYPTO_CONFIG, crypto | slot << 24);
            self.write(CRYPTO_LAST_BLOCK, input.len() as u32 / 16 - 1);
            self.run(input, input.len())
        }

        fn hash(&self) -> Vec<u8> {
            (0..16)
                .flat_map(|index| self.read(HASH_RESULT + 4 * index).to_be_bytes())
                .collect()
        }

        /// CMAC as a driver builds it from the engine's CBC-MAC.
        fn cmac(&self, slot: u32, message: &[u8]) -> Vec<u8> {
            let subkey = |block: &[u8]| {
                let value = u128::from_be_bytes(block.try_into().unwrap());
                let doubled = value << 1 ^ if value >> 127 != 0 { 0x87 } else { 0 };
                doubled.to_be_bytes()
            };
            let k1 = subkey(&self.aes(slot, 0, ECB_ENCRYPT, &[0; 16]));
            let k2 = subkey(&k1);
            let blocks = message.len().div_ceil(16).max(1);
            let complete = !message.is_empty() && message.len().is_multiple_of(16);
            let mut last = [0; 16];
            let rest = &message[(blocks - 1) * 16..];
            last[..rest.len()].copy_from_slice(rest);
            if !complete {
                last[rest.len()] = 0x80;
            }
            xor_block(&mut last, if complete { &k1 } else { &k2 });
            self.key_words(slot, OIV_WORD as u32, &[0; 16]);
            self.write(CONFIG, config(ALG_AES, ALG_NOP, 0, DST_HASHREG));
            let mut crypto = CBC_ENCRYPT | CRYPTO_HASH | slot << 24;
            if blocks > 1 {
                self.write(CRYPTO_CONFIG, crypto);
                self.write(CRYPTO_LAST_BLOCK, blocks as u32 - 2);
                self.run(&message[..(blocks - 1) * 16], 0);
                crypto |= UPDATED_IV;
            }
            self.write(CRYPTO_CONFIG, crypto);
            self.write(CRYPTO_LAST_BLOCK, 0);
            self.run(&last, 0);
            (0..4)
                .flat_map(|index| self.read(HASH_RESULT + 4 * index).to_le_bytes())
                .collect()
        }

        /// One SHA operation over `input`; `total` is the message's length
        /// in bytes, `None` for a part that is not its end.
        fn sha(&self, mode: u32, init: bool, input: &[u8], total: Option<usize>) {
            self.write(CONFIG, config(ALG_SHA, ALG_NOP, mode, DST_HASHREG));
            let length = total.unwrap_or(0) as u64 * 8;
            let left = (input.len() + usize::from(total.is_none())) as u64 * 8;
            self.write(SHA_MSG_LENGTH, length as u32);
            self.write(SHA_MSG_LENGTH + 4, (length >> 32) as u32);
            self.write(SHA_MSG_LEFT, left as u32);
            self.write(SHA_MSG_LEFT + 4, (left >> 32) as u32);
            self.write(SHA_CONFIG, u32::from(init));
            self.run(input, 0);
        }
    }

    const NIST_KEY: &str = "2b7e151628aed2a6abf7158809cf4f3c";
    const NIST_PLAINTEXT: &str = "6bc1bee22e409f96e93d7e117393172aae2d8a571e03ac9c9eb76fac45af8e51\
                                  30c81c46a35ce411e5fbc1191a0a52eff69f2445df4f9b17ad2b417be66c3710";

    #[test]
    fn ecb_matches_fips_197() {
        let bench = Bench::new();
        let plaintext = hex("00112233445566778899aabbccddeeff");
        bench.key_words(3, 0, &hex("000102030405060708090a0b0c0d0e0f"));
        let ciphertext = bench.aes(3, 0, ECB_ENCRYPT, &plaintext);
        assert_eq!(ciphertext, hex("69c4e0d86a7b0430d8cdb78070b4c55a"));
        assert_eq!(bench.aes(3, 0, ECB_DECRYPT, &ciphertext), plaintext);
        bench.key_words(
            4,
            0,
            &hex("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"),
        );
        let ciphertext = bench.aes(4, 2, ECB_ENCRYPT, &plaintext);
        assert_eq!(ciphertext, hex("8ea2b7ca516745bfeafc49904b496089"));
        assert_eq!(bench.aes(4, 2, ECB_DECRYPT, &ciphertext), plaintext);
    }

    #[test]
    fn cbc_matches_sp800_38a_and_continues_from_the_updated_iv() {
        let bench = Bench::new();
        bench.key_words(0, 0, &hex(NIST_KEY));
        bench.key_words(0, OIV_WORD as u32, &hex("000102030405060708090a0b0c0d0e0f"));
        let plaintext = hex(NIST_PLAINTEXT);
        let expected = hex(
            "7649abac8119b246cee98e9b12e9197d5086cb9b507219ee95db113a917678b2\
             73bed6b8e3c1743b7116e69e222295163ff1caa1681fac09120eca307586e1a7",
        );
        assert_eq!(bench.aes(0, 0, CBC_ENCRYPT, &plaintext), expected);
        assert_eq!(bench.aes(0, 0, CBC_DECRYPT, &expected), plaintext);
        // In two parts, the second from where the first left the IV.
        let first = bench.aes(0, 0, CBC_ENCRYPT, &plaintext[..32]);
        let second = bench.aes(0, 0, CBC_ENCRYPT | UPDATED_IV, &plaintext[32..]);
        assert_eq!([first, second].concat(), expected);
    }

    #[test]
    fn ctr_matches_sp800_38a_and_leaves_the_counter_advanced() {
        let bench = Bench::new();
        bench.key_words(5, 0, &hex(NIST_KEY));
        let counter = hex("f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff");
        for (index, bytes) in counter.as_chunks::<4>().0.iter().enumerate() {
            bench.write(
                CRYPTO_LINEAR_CTR + 4 * index as u64,
                u32::from_le_bytes(*bytes),
            );
        }
        let plaintext = hex(NIST_PLAINTEXT);
        let first = bench.aes(5, 0, CTR, &plaintext[..16]);
        let rest = bench.aes(5, 0, CTR, &plaintext[16..]);
        assert_eq!(
            [first, rest].concat(),
            hex(
                "874d6191b620e3261bef6864990db6ce9806f66b7970fdff8617187bb9fffdff\
                 5ae4df3edbd5d35e5b4f09020db03eab1e031dda2fbe03d1792170a0f3009cee"
            )
        );
    }

    #[test]
    fn cmac_matches_sp800_38b() {
        let bench = Bench::new();
        bench.key_words(2, 0, &hex(NIST_KEY));
        let message = hex(NIST_PLAINTEXT);
        for (len, mac) in [
            (0, "bb1d6929e95937287fa37d129b756746"),
            (16, "070a16b46b4d4144f79bdd9dd04a287c"),
            (40, "dfa66747de9ae63030ca32611497c827"),
            (64, "51f0bebf7e3b9d92fc49741779363cfe"),
        ] {
            assert_eq!(bench.cmac(2, &message[..len]), hex(mac), "{len} bytes");
        }
    }

    #[test]
    fn sha_matches_fips_180_for_abc() {
        let bench = Bench::new();
        for (mode, digest) in [
            (0, "a9993e364706816aba3e25717850c26c9cd0d89d"),
            (
                4,
                "23097d223405d8228642a477bda255b32aadbce4bda0b3f7e36c9da7",
            ),
            (
                5,
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            ),
            (
                6,
                "cb00753f45a35e8bb5a03d699ac65007272c32ab0eded1631a8b605a43ff5bed\
                 8086072ba1e7cc2358baeca134c825a7",
            ),
            (
                7,
                "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a\
                 2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f",
            ),
        ] {
            bench.sha(mode, true, b"abc", Some(3));
            let digest = hex(digest);
            assert_eq!(bench.hash()[..digest.len()], digest, "mode {mode}");
        }
        let two_blocks = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        bench.sha(5, true, two_blocks, Some(two_blocks.len()));
        assert_eq!(
            bench.hash()[..32],
            hex("248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1")
        );
    }

    #[test]
    fn sha_continues_from_the_state_in_the_hash_registers() {
        use sha2::Digest;
        let bench = Bench::new();
        let message: Vec<u8> = (0..300u32).map(|index| (index * 7) as u8).collect();
        bench.sha(5, true, &message[..256], None);
        bench.sha(5, false, &message[256..], Some(message.len()));
        assert_eq!(bench.hash()[..32], sha2::Sha256::digest(&message)[..]);
        bench.sha(7, true, &message[..256], None);
        bench.sha(7, false, &message[256..], Some(message.len()));
        assert_eq!(bench.hash(), sha2::Sha512::digest(&message)[..]);
    }

    #[test]
    fn read_protected_key_words_read_zero_or_all_ones_if_per_key_secure() {
        let bench = Bench::new();
        let key = hex(NIST_KEY);
        bench.key_words(14, 0, &key);
        bench.write(CRYPTO_KEYTABLE_ADDR, 14 << 4 | 1);
        assert_eq!(
            bench.read(CRYPTO_KEYTABLE_DATA),
            u32::from_le_bytes(key[4..8].try_into().unwrap())
        );
        bench.write(CRYPTO_KEYTABLE_ACCESS + 14 * 4, !LOCK_KEY_READ);
        assert_eq!(bench.read(CRYPTO_KEYTABLE_ACCESS + 14 * 4), 0x7E);
        assert_eq!(bench.read(CRYPTO_KEYTABLE_DATA), 0);
        bench.write(CRYPTO_SECURITY_PERKEY, !(1 << 14));
        assert_eq!(bench.read(CRYPTO_KEYTABLE_DATA), 0xFFFF_FFFF);
        // Through load_key too.
        bench
            .se
            .load_key(6, &[0; 16], LOCK_KEY_READ | LOCK_PER_KEY)
            .unwrap();
        bench.write(CRYPTO_KEYTABLE_ADDR, 6 << 4);
        assert_eq!(bench.read(CRYPTO_KEYTABLE_DATA), 0xFFFF_FFFF);
        // The OIV keeps its own permission; the key can still be used.
        bench.write(CRYPTO_KEYTABLE_ADDR, 14 << 4 | OIV_WORD as u32);
        assert_eq!(bench.read(CRYPTO_KEYTABLE_DATA), 0);
        bench.key_words(0, 0, &key);
        assert_eq!(
            bench.aes(14, 0, ECB_ENCRYPT, &[0x5A; 16]),
            bench.aes(0, 0, ECB_ENCRYPT, &[0x5A; 16])
        );
    }

    #[test]
    fn locked_slots_ignore_key_writes_and_stay_locked() {
        let bench = Bench::new();
        let key = hex(NIST_KEY);
        bench
            .se
            .load_key(15, &key, LOCK_KEY_READ | LOCK_KEY_UPDATE | LOCK_PER_KEY)
            .unwrap();
        // Bit 7 reads 0 from reset.
        assert_eq!(bench.read(CRYPTO_KEYTABLE_ACCESS + 15 * 4), 0x7C);
        assert_eq!(bench.read(CRYPTO_SECURITY_PERKEY), 0x7FFF);
        // Permissions only go.
        bench.write(CRYPTO_KEYTABLE_ACCESS + 15 * 4, 0xFF);
        bench.write(CRYPTO_SECURITY_PERKEY, 0xFFFF);
        assert_eq!(bench.read(CRYPTO_KEYTABLE_ACCESS + 15 * 4), 0x7C);
        assert_eq!(bench.read(CRYPTO_SECURITY_PERKEY), 0x7FFF);
        bench.write(CRYPTO_KEYTABLE_ADDR, 15 << 4);
        bench.write(CRYPTO_KEYTABLE_DATA, 0);
        assert!(bench.notes.take().is_empty());
        assert_eq!(bench.se.load_key(15, &[0; 16], 0), Err(SeError::Locked(15)));
        // The key is the one loaded.
        bench.key_words(0, 0, &key);
        assert_eq!(
            bench.aes(15, 0, ECB_ENCRYPT, &[0xA5; 16]),
            bench.aes(0, 0, ECB_ENCRYPT, &[0xA5; 16])
        );
        // Without key use the slot still serves: every requester is secure.
        bench.se.load_key(1, &key, LOCK_KEY_USE).unwrap();
        assert_eq!(bench.read(CRYPTO_KEYTABLE_ACCESS + 4), 0x3F);
        assert_eq!(
            bench.aes(1, 0, ECB_ENCRYPT, &[0xA5; 16]),
            bench.aes(15, 0, ECB_ENCRYPT, &[0xA5; 16])
        );
        assert!(bench.notes.take().is_empty());
    }

    #[test]
    fn decrypts_a_wrapped_key_into_a_key_slot() {
        let bench = Bench::new();
        let kek = hex(NIST_KEY);
        let key = hex("000102030405060708090a0b0c0d0e0f");
        let mut wrapped = key.clone();
        Cipher::new(&kek).run(true, (&mut wrapped[..]).try_into().unwrap());
        bench.se.load_key(12, &kek, LOCK_KEY_READ).unwrap();
        bench.write(CONFIG, config(ALG_NOP, ALG_AES, 0, DST_KEYTABLE));
        bench.write(CRYPTO_CONFIG, ECB_DECRYPT | 12 << 24);
        bench.write(CRYPTO_LAST_BLOCK, 0);
        bench.write(CRYPTO_KEYTABLE_DST, 13 << 8);
        bench.run(&wrapped, 0);
        assert_eq!(
            bench.aes(13, 0, ECB_ENCRYPT, &hex("00112233445566778899aabbccddeeff")),
            hex("69c4e0d86a7b0430d8cdb78070b4c55a")
        );
        // Not into a slot that is locked.
        bench.se.load_key(13, &key, LOCK_KEY_UPDATE).unwrap();
        bench.write(CONFIG, config(ALG_NOP, ALG_AES, 0, DST_KEYTABLE));
        bench.write(CRYPTO_CONFIG, ECB_DECRYPT | 12 << 24);
        bench.write(CRYPTO_LAST_BLOCK, 0);
        assert_eq!(bench.se.write(0, OPERATION, 4, OP_START.into()), None);
        assert!(bench.notes.take()[0].contains("into key slot 13"));
    }

    #[test]
    fn follows_linked_lists_of_several_entries() {
        let bench = Bench::new();
        bench.key_words(0, 0, &hex(NIST_KEY));
        bench.key_words(0, OIV_WORD as u32, &hex("000102030405060708090a0b0c0d0e0f"));
        let plaintext = hex(NIST_PLAINTEXT);
        let expected = bench.aes(0, 0, CBC_ENCRYPT, &plaintext);
        // The input in three pieces, the output in two, none block-sized.
        bench.poke(INPUT, &plaintext[..20]);
        bench.poke(INPUT + 0x100, &plaintext[20..48]);
        bench.poke(INPUT + 0x200, &plaintext[48..]);
        bench.list(
            IN_LIST,
            &[(INPUT, 20), (INPUT + 0x100, 28), (INPUT + 0x200, 16)],
        );
        bench.list(OUT_LIST, &[(OUTPUT, 40), (OUTPUT + 0x100, 24)]);
        bench.write(CONFIG, config(ALG_AES, ALG_NOP, 0, DST_MEMORY));
        bench.write(CRYPTO_CONFIG, CBC_ENCRYPT);
        bench.write(CRYPTO_LAST_BLOCK, 3);
        bench.write(IN_LL_ADDR, IN_LIST);
        bench.write(OUT_LL_ADDR, OUT_LIST);
        bench.start();
        assert_eq!(
            [bench.peek(OUTPUT, 40), bench.peek(OUTPUT + 0x100, 24)].concat(),
            expected
        );
        // An output list too short for the result is refused.
        bench.list(OUT_LIST, &[(OUTPUT, 48)]);
        assert_eq!(bench.se.write(0, OPERATION, 4, OP_START.into()), None);
        assert!(bench.notes.take()[0].contains("output list"));
    }

    /// Writes `value` into RSA slot `slot`'s exponent or modulus.
    fn rsa_key(bench: &Bench, slot: u32, modulus: bool, value: &BigUint) {
        let mut bytes = value.to_bytes_le();
        bytes.resize(RSA_WORDS * 4, 0);
        for (index, word) in bytes.as_chunks::<4>().0.iter().enumerate() {
            bench.write(
                RSA_KEYTABLE_ADDR,
                slot << 7 | u32::from(modulus) << 6 | index as u32,
            );
            bench.write(RSA_KEYTABLE_DATA, u32::from_le_bytes(*word));
        }
    }

    /// `input` to the power of slot `slot`'s exponent of `exponent_words`.
    fn rsa_exponentiate(bench: &Bench, slot: u32, exponent_words: u32, input: &BigUint) -> BigUint {
        let mut bytes = input.to_bytes_le();
        bytes.resize(RSA_WORDS * 4, 0);
        bench.write(CONFIG, config(ALG_RSA, ALG_NOP, 0, DST_RSAREG));
        bench.write(RSA_CONFIG, slot << 24);
        bench.write(RSA_KEY_SIZE, 3);
        bench.write(RSA_EXP_SIZE, exponent_words);
        bench.run(&bytes, 0);
        let output: Vec<u8> = (0..RSA_WORDS as u64)
            .flat_map(|index| bench.read(RSA_OUTPUT + 4 * index).to_le_bytes())
            .collect();
        BigUint::from_bytes_le(&output)
    }

    #[test]
    fn rsa_2048_exponentiation_round_trips_a_test_key() {
        use rand_chacha::ChaCha8Rng;
        use rsa::traits::{PrivateKeyParts, PublicKeyParts};
        let key = rsa::RsaPrivateKey::new(&mut ChaCha8Rng::seed_from_u64(1), 2048).unwrap();
        let bench = Bench::new();
        rsa_key(&bench, 0, true, key.n());
        rsa_key(&bench, 0, false, key.e());
        rsa_key(&bench, 1, true, key.n());
        rsa_key(&bench, 1, false, key.d());
        let message = BigUint::from_bytes_be(&[0x42; 200]);
        let encrypted = rsa_exponentiate(&bench, 0, 1, &message);
        assert_eq!(encrypted, message.modpow(key.e(), key.n()));
        assert_eq!(rsa_exponentiate(&bench, 1, 64, &encrypted), message);
        // The modulus reads back until reading is revoked.
        bench.write(RSA_KEYTABLE_ADDR, 1 << 7 | 1 << 6);
        assert_eq!(
            bench.read(RSA_KEYTABLE_DATA),
            u32::from_le_bytes(key.n().to_bytes_le()[..4].try_into().unwrap())
        );
        bench.write(RSA_KEYTABLE_ACCESS + 4, !RSA_ACCESS_KEY_READ);
        assert_eq!(bench.read(RSA_KEYTABLE_DATA), 0);
        bench.write(RSA_SECURITY_PERKEY, !(1 << 1));
        assert_eq!(bench.read(RSA_KEYTABLE_DATA), 0xFFFF_FFFF);
        // A slot whose use is revoked refuses.
        bench.write(RSA_KEYTABLE_ACCESS + 4, !RSA_ACCESS_KEY_USE);
        assert_eq!(bench.read(RSA_KEYTABLE_ACCESS + 4), RSA_ACCESS_KEY_UPDATE);
        assert_eq!(bench.se.write(0, OPERATION, 4, OP_START.into()), None);
        assert!(bench.notes.take()[0].contains("RSA with key slot 1"));
    }

    fn random(bench: &Bench) -> Vec<u8> {
        bench.write(CONFIG, config(ALG_RNG, ALG_NOP, 0, DST_MEMORY));
        bench.write(CRYPTO_CONFIG, INPUT_RANDOM << 3);
        bench.write(CRYPTO_LAST_BLOCK, 1);
        bench.list(OUT_LIST, &[(OUTPUT, 32)]);
        bench.write(OUT_LL_ADDR, OUT_LIST);
        bench.start();
        bench.peek(OUTPUT, 32)
    }

    #[test]
    fn random_numbers_follow_the_seed_and_snapshots() {
        let first = Bench::new();
        let second = Bench::new();
        let drawn = random(&first);
        assert_eq!(drawn, random(&second));
        assert_ne!(drawn, random(&second), "the stream moves on");
        first.se.load_key(9, &hex(NIST_KEY), LOCK_KEY_READ).unwrap();
        let saved = first.se.save().unwrap();
        let restored = Bench::new();
        restored.se.restore(&saved).unwrap();
        assert_eq!(random(&restored), random(&first));
        assert_eq!(restored.read(CRYPTO_KEYTABLE_ACCESS + 9 * 4), 0x7E);
        assert_eq!(
            restored.aes(9, 0, ECB_ENCRYPT, &[1; 16]),
            first.aes(9, 0, ECB_ENCRYPT, &[1; 16])
        );
    }

    #[test]
    fn security_settings_close_except_the_soft_setting() {
        let bench = Bench::new();
        assert_eq!(bench.read(SECURITY), 0x1_0005);
        bench.write(SECURITY, 0);
        bench.write(SECURITY, 0x0005);
        assert_eq!(bench.read(SECURITY), 0);
        // The monitor sets bit 16 around its context save and clears it after.
        bench.write(SECURITY, SECURITY_SOFT);
        assert_eq!(bench.read(SECURITY), SECURITY_SOFT);
        bench.write(SECURITY, SECURITY_ENGINE_DISABLE);
        bench.write(SECURITY, 0);
        assert_eq!(bench.read(SECURITY), SECURITY_ENGINE_DISABLE);
        assert_eq!(bench.se.write(0, OPERATION, 4, OP_START.into()), None);
        assert!(bench.notes.take()[0].contains("disabled"));
    }

    /// Where the context-save tests put the context and their scratch data.
    const CONTEXT: u32 = IRAM + 0x4000;
    const RANDOM: u32 = IRAM + 0x3000;
    const PATTERN: u32 = IRAM + 0x3100;

    impl Bench {
        /// One context save operation as the monitor's helper runs it:
        /// `input` (address, size) for the memory source, the result to
        /// `output` (address, size).
        fn save(&self, config: u32, input: (u32, u32), output: (u32, u32)) {
            self.write(CTX_SAVE_CONFIG, config);
            self.write(CRYPTO_LAST_BLOCK, 0);
            self.list(IN_LIST, &[input]);
            self.list(OUT_LIST, &[output]);
            self.write(IN_LL_ADDR, IN_LIST);
            self.write(OUT_LL_ADDR, OUT_LIST);
            self.write(OPERATION, OP_CTX_SAVE);
            assert_eq!(self.read(INT_STATUS) & INT_OP_DONE, INT_OP_DONE);
            self.write(INT_STATUS, self.read(INT_STATUS));
        }

        /// The 23.5.0 secure monitor's context save (TZRAM `0x1F0141610`)
        /// to `CONTEXT`, register for register.
        fn monitor_context_save(&self) {
            // SRK (0x1F01414F8).
            self.write(CONFIG, 0x200C);
            self.write(CRYPTO_CONFIG, 0x108);
            self.write(RNG_CONFIG, 6);
            self.write(CRYPTO_LAST_BLOCK, 0);
            self.start();
            // 16 random bytes (0x1F0142A4C).
            self.write(CONFIG, 0x2000);
            self.write(CRYPTO_CONFIG, 0x108);
            self.write(RNG_CONFIG, 4);
            self.write(CRYPTO_LAST_BLOCK, 0);
            self.list(OUT_LIST, &[(RANDOM, 16)]);
            self.write(OUT_LL_ADDR, OUT_LIST);
            self.start();
            self.write(SECURITY, self.read(SECURITY) | SECURITY_SOFT);
            self.write(CONFIG, 0x1000);
            let block = |offset: usize| (CONTEXT + offset as u32, 16);
            self.save(0x8000_0000, (RANDOM, 16), block(0));
            for quad in 0..2 {
                self.save(quad << 24, (0, 0), block(0x10 + 0x10 * quad as usize));
            }
            for slot in 0..16 {
                for quad in 0..2 {
                    let at = CONTEXT_KEYS + 32 * slot as usize + 16 * quad as usize;
                    self.save(0x4000_0000 | slot << 8 | quad, (0, 0), block(at));
                }
            }
            for slot in 0..16 {
                let at = CONTEXT_OIVS + 16 * slot as usize;
                self.save(0x4000_0002 | slot << 8, (0, 0), block(at));
            }
            for slot in 0..16 {
                let at = CONTEXT_UIVS + 16 * slot as usize;
                self.save(0x4000_0003 | slot << 8, (0, 0), block(at));
            }
            for slot in 0..2u32 {
                for part in 0..2u32 {
                    let key = 2 * slot + 1 - part;
                    for quad in 0..16u32 {
                        let at = CONTEXT_RSA + (0x200 * slot + 0x100 * part + 16 * quad) as usize;
                        self.save(0x2000_0000 | key << 16 | quad << 12, (0, 0), block(at));
                    }
                }
            }
            self.poke(PATTERN, &KNOWN_PATTERN);
            self.save(0x8000_0000, (PATTERN, 16), block(CONTEXT_PATTERN));
            self.save(0xC000_0000, (0, 0), (CONTEXT, 0));
            self.write(CONFIG, 0);
            self.save(0xC000_0000, (0, 0), (CONTEXT, 0));
            self.write(SECURITY, self.read(SECURITY) & !SECURITY_SOFT);
        }
    }

    #[test]
    fn the_monitors_context_save_reopens_and_restores_as_the_boot_rom_does() {
        let saved = Bench::new();
        for slot in 0..16u8 {
            let flags = if slot >= 6 {
                LOCK_KEY_READ | LOCK_KEY_UPDATE
            } else {
                0
            };
            saved
                .se
                .load_key(slot.into(), &[slot.wrapping_mul(17) ^ 0x3C; 32], flags)
                .unwrap();
        }
        saved.key_words(4, OIV_WORD as u32, &[0x11; 16]);
        saved.key_words(5, UIV_WORD as u32, &[0x22; 16]);
        let modulus = BigUint::from_bytes_be(&[0xC3; 256]);
        let exponent = BigUint::from(65537u32);
        rsa_key(&saved, 0, true, &modulus);
        rsa_key(&saved, 0, false, &exponent);
        saved.write(RSA_KEYTABLE_ACCESS + 4, !RSA_ACCESS_KEY_READ);
        saved.write(SECURITY, 0x1_0001);
        saved.monitor_context_save();
        let words = saved
            .scratch
            .lock()
            .unwrap()
            .expect("the SRK went to the PMC");
        let srk = srk_from_scratch(words);
        let blob = saved.peek(CONTEXT, CONTEXT_SIZE);
        // The chain starts from zero at the random block.
        let mut first: [u8; 16] = saved.peek(RANDOM, 16).try_into().unwrap();
        Cipher::new(&srk).run(true, &mut first);
        assert_eq!(blob[..16], first);

        let context = SeContext::open(&blob, &srk).unwrap();
        assert_eq!(format!("{context:?}"), "SeContext { .. }");
        let restored = Bench::new();
        restored.se.restore_context(&context);
        for slot in 0..16u32 {
            let access = CRYPTO_KEYTABLE_ACCESS + 4 * u64::from(slot);
            assert_eq!(restored.read(access), saved.read(access), "slot {slot}");
            for word in 0..16 {
                saved.write(CRYPTO_KEYTABLE_ADDR, slot << 4 | word);
                restored.write(CRYPTO_KEYTABLE_ADDR, slot << 4 | word);
                assert_eq!(
                    restored.read(CRYPTO_KEYTABLE_DATA),
                    saved.read(CRYPTO_KEYTABLE_DATA),
                    "slot {slot} word {word}"
                );
            }
            assert_eq!(
                restored.aes(slot, 2, ECB_ENCRYPT, &[7; 16]),
                saved.aes(slot, 2, ECB_ENCRYPT, &[7; 16]),
                "slot {slot}"
            );
        }
        for register in [
            CRYPTO_SECURITY_PERKEY,
            RSA_SECURITY_PERKEY,
            RSA_KEYTABLE_ACCESS,
            RSA_KEYTABLE_ACCESS + 4,
        ] {
            assert_eq!(restored.read(register), saved.read(register));
        }
        // The soft setting was set while the sticky bits were saved.
        assert_eq!(restored.read(SECURITY), 0x1_0001);
        for word in [0, 63] {
            restored.write(RSA_KEYTABLE_ADDR, 1 << 6 | word);
            saved.write(RSA_KEYTABLE_ADDR, 1 << 6 | word);
            assert_eq!(
                restored.read(RSA_KEYTABLE_DATA),
                saved.read(RSA_KEYTABLE_DATA)
            );
        }
        assert_eq!(
            rsa_exponentiate(&restored, 0, 1, &BigUint::from(3u32)),
            BigUint::from(3u32).modpow(&exponent, &modulus)
        );

        let mut other = srk;
        other[0] ^= 1;
        assert_eq!(
            SeContext::open(&blob, &other).unwrap_err(),
            ContextError::KnownPattern
        );
        assert_eq!(
            SeContext::open(&blob[..0x830], &srk).unwrap_err(),
            ContextError::Size(0x830)
        );
    }

    #[test]
    fn context_save_needs_an_srk_a_pmc_and_a_known_source() {
        let bench = Bench::new();
        bench.write(CONFIG, 0x1000);
        bench.write(CTX_SAVE_CONFIG, 0x4000_0000);
        bench.list(OUT_LIST, &[(OUTPUT, 16)]);
        bench.write(OUT_LL_ADDR, OUT_LIST);
        assert_eq!(bench.se.write(0, OPERATION, 4, OP_CTX_SAVE.into()), None);
        assert!(bench.notes.take()[0].contains("before an SRK"));
        bench.write(CONFIG, 0x200C);
        bench.write(CRYPTO_LAST_BLOCK, 0);
        bench.start();
        bench.write(CONFIG, 0x1000);
        for config in [0x6000_0000, 0x4000_0000 | 1 << 20] {
            bench.write(CTX_SAVE_CONFIG, config);
            assert_eq!(bench.se.write(0, OPERATION, 4, OP_CTX_SAVE.into()), None);
            assert!(bench.notes.take()[0].contains("SE_CTX_SAVE_CONFIG"));
        }
        let ram = Arc::new(Ram(Mutex::new(vec![0; 0x1_0000])));
        let notes: Arc<Notes> = Arc::default();
        let unwired = Se::new(Box::new(ram), Arc::clone(&notes), 7);
        for (offset, value) in [
            (CONFIG, 0x200C),
            (OPERATION, OP_START),
            (CTX_SAVE_CONFIG, 0xC000_0000),
        ] {
            unwired.write(0, offset, 4, value.into()).unwrap();
        }
        assert_eq!(unwired.write(0, OPERATION, 4, OP_CTX_SAVE.into()), None);
        assert!(notes.take()[0].contains("no PMC"));
    }

    #[test]
    fn the_boot_machines_engine_reaches_tzram() {
        let iram = Arc::new(Iram::new());
        let tzram = Arc::new(Tzram::new());
        let dram = Arc::new(GuestRam::new(0x8000_0000, 0x2_0000).unwrap());
        let memory = BootMemory::new(Arc::clone(&iram), Arc::clone(&tzram), Arc::clone(&dram));
        let se = Se::new(Box::new(memory), Arc::default(), 7);
        let write = |offset: u64, value: u32| se.write(0, offset, 4, value.into()).unwrap();
        let list = |at: u32, address: u32, size: u32| {
            let bytes: Vec<u8> = [0, address, size]
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect();
            iram.load(at - IRAM, &bytes).unwrap();
        };
        se.load_key(3, &hex(&NIST_KEY.repeat(2)), 0).unwrap();
        let plaintext: Vec<u8> = (0..0x200u32).map(|index| (index * 13) as u8).collect();
        tzram.load(0x2000, &plaintext).unwrap();
        // The monitor's TZRAM → DRAM encryption, AES-256-CBC with slot 3.
        list(IN_LIST, 0x7C01_2000, 0x200);
        list(OUT_LIST, 0x8001_0000, 0x200);
        write(IN_LL_ADDR, IN_LIST);
        write(OUT_LL_ADDR, OUT_LIST);
        write(CONFIG, config(ALG_AES, ALG_NOP, 2, DST_MEMORY));
        write(CRYPTO_CONFIG, CBC_ENCRYPT | 3 << 24);
        write(CRYPTO_LAST_BLOCK, 0x1F);
        write(OPERATION, OP_START);
        let mut encrypted = vec![0; 0x200];
        dram.read(0x8001_0000, &mut encrypted).unwrap();
        assert_ne!(encrypted, plaintext);
        // The warmboot's decryption back, its registers as WB0 writes them.
        tzram.load(0x2000, &[0; 0x200]).unwrap();
        list(IN_LIST, 0x8001_0000, 0x200);
        list(OUT_LIST, 0x7C01_2000, 0x200);
        write(CONFIG, 0x0202_0100);
        write(CRYPTO_CONFIG, 3 << 24 | 0x66);
        write(OPERATION, OP_START);
        let mut decrypted = vec![0; 0x200];
        tzram.copy_out(0x2000, &mut decrypted).unwrap();
        assert_eq!(decrypted, plaintext);
        // A range across the end of TZRAM is reached by nobody.
        let memory = BootMemory::new(iram, tzram, dram);
        assert_eq!(memory.read(0x7C01_FFF8, &mut [0; 16]), None);
    }

    #[test]
    fn a_car_reset_of_the_engine_keeps_its_key_slots() {
        // The warmboot pulses the engine's reset (RST_DEVICES_V bit 31,
        // through RST_DEV_V_SET/CLR) and then uses slots 2, 10-13 and 15.
        let car = super::super::car::Car::power_on();
        let bench = Bench::new();
        let key = hex(NIST_KEY);
        bench
            .se
            .load_key(2, &key, LOCK_KEY_READ | LOCK_PER_KEY)
            .unwrap();
        let before = bench.aes(2, 0, ECB_ENCRYPT, &[9; 16]);
        let access = bench.read(CRYPTO_KEYTABLE_ACCESS + 8);
        for (register, value) in [(0x430, 1 << 31), (0x440, 1 << 31), (0x434, 1 << 31)] {
            crate::bus::MmioDevice::write(&car, 0, register, 4, value).unwrap();
        }
        assert_eq!(bench.read(CRYPTO_KEYTABLE_ACCESS + 8), access);
        assert_eq!(bench.read(CRYPTO_SECURITY_PERKEY), 0xFFFB);
        assert_eq!(bench.aes(2, 0, ECB_ENCRYPT, &[9; 16]), before);
    }

    /// The NX bootloader's check of the engine when it starts
    /// (`nxbl_hwcheck.csv`): offset, mask, value.
    const CHECKED: [(u64, u32, u32); 13] = [
        (0x000, 0x1_0007, 0x1_0005),
        (0x004, 0x3, 0x1),
        (0x00C, 0x1_003F, 0),
        (0x280, 0xFFFF, 0xDFFF),
        (0x2BC, 0x7F, 0x7E),
        (0x2C0, 0x7F, 0x7E),
        (0x340, 0xF, 0),
        (0x344, 0x177, 0),
        (0x40C, 0x3, 0x3),
        (0x808, 0x7, 0),
        (0x814, 0x8F9F_FFFF, 0x8904_0800),
        (0x818, 0x1F_0000, 0),
        (0x81C, 0x101, 0),
    ];

    #[test]
    fn the_engine_after_the_boot_roms_and_tsecs_locks_passes_the_nx_bootloaders_check() {
        let notes: Arc<Notes> = Arc::default();
        let dram = Arc::new(GuestRam::new(0x8000_0000, 0x1000).unwrap());
        let memory = BootMemory::new(Arc::default(), Arc::default(), dram);
        let se = Se::new(Box::new(memory), notes, 7);
        // What runs before the bootloader changes: SBK and SSK
        // read-protected (boot ROM), slot 13's per-key bit (TSEC step).
        for slot in [14, 15] {
            se.load_key(slot, &[0; 16], LOCK_KEY_READ).unwrap();
        }
        se.write(0, CRYPTO_SECURITY_PERKEY, 4, u64::from(!(1u32 << 13)))
            .unwrap();
        let read = |offset| se.read(0, offset, 4).unwrap() as u32;
        for (offset, mask, value) in CHECKED {
            assert_eq!(read(offset) & mask, value, "{offset:#x}");
        }
        for slot in 0..14 {
            assert_eq!(read(CRYPTO_KEYTABLE_ACCESS + 4 * slot) & 0x7F, 0x7F);
        }
        for slot in 0..2 {
            assert_eq!(read(RSA_KEYTABLE_ACCESS + 4 * slot) & 0x7, 0x7);
        }
    }
}
