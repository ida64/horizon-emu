//! The secure monitor's SC7 (deep sleep) in a machine powered on from the
//! boot ROM: what it does before the SoC powers down and after the warmboot
//! firmware brings core 0 back, high-level emulated from the 23.5.0 Erista
//! monitor (`warmboot-research` §3, §4, §8; addresses below are the
//! monitor's virtual addresses, TZRAM code at `0x1F0140000` = physical
//! `0x7C012000`).
//!
//! A machine started from a kernel ELF has none of the blocks this needs
//! and keeps the monitor's plain sleep ([`super::SmcEffect::Sleep`]).
//!
//! # Suspend (`CPU_SUSPEND`, `0x1F0145F40`, then `0x1F0144C84`)
//!
//! Everything goes through the bus as core 0, the security engine
//! included: its registers are driven with the values the monitor writes,
//! so its key slots and the context it saves are the device's own.
//!
//! 1. Wake detection on, wake status cleared (`0x1F014371C`).
//! 2. The preconditions ([`super::SLEEP_REQUIREMENTS`]) and the BPMP parked
//!    (`HALT_COP_EVENTS` = `0x40000000`, as the cold boot leaves it).
//! 3. `PMC_SCRATCH0` = 1 (warm boot), `DPD_ENABLE` bit 1 (`0x1F0143548`).
//! 4. 16 random bytes R from the engine into `SECURE_SCRATCH24..27`,
//!    write-locked, checked, read-locked.
//! 5. Key slot 2 from the engine's random number generator (256 bits).
//! 6. The engine's context saved to DRAM [`CONTEXT`] (`0x1F0141610`);
//!    `PMC_SCRATCH43` = its address.
//! 7. Key slot 3 cleared, then the engine decrypts R with slot 2 into it.
//! 8. TZRAM [`SAVED`] encrypted with AES-256-CBC (slot 3, IV 0) into IRAM
//!    `0x40020000`, copied to DRAM [`SAVED_DRAM`]; its AES-256-CMAC (slot 3)
//!    into `SECURE_SCRATCH112..115`, locked like R.
//! 9. The BPMP restarted on the monitor's SC7 stub: its exception vectors
//!    at `0x40003000`, held in reset, the stub copied from TZRAM
//!    [`STUB`] to IRAM `0x40003000` and compared, reset released, its halt
//!    ended.
//! 10. Core 0 waits for interrupt with power-gating armed, so the flow
//!     controller turns its partitions off (CE0, C0NC, then the CPU rail);
//!     here the monitor does that last step itself through
//!     `PWRGATE_TOGGLE` \[inference: the flow controller model does not
//!     power-gate on WFI\]. The stub waits for the rail to be off.
//!
//! Not done here (no device models them, nothing reads them back here): the
//! charger's high-impedance set-up over I2C1 (refused when the kernel asked
//! for it), `ONOFFCNFG1.SLPEN` of the MAX77620 over I2C5, the flow
//! controller's per-core settings for core 0 (its model refuses
//! power-gating), `APB_SLAVE_SECURITY` (APB_MISC `0x70000C00`), the AHB
//! block at `0x6000C8C0` and `AHB_ARBITRATION_DISABLE` (not mapped).
//!
//! The memory the monitor's routine uses for engine linked lists and
//! bounce buffers is its stack, TZRAM [`STACK`] (`0x1F01F6E80` grows down
//! from physical `0x7C010E80`), outside the saved range.
//!
//! # Resume (core 0 released at [`RESUME_ENTRY`], `0x1F014414C`)
//!
//! The monitor redoes its SoC security set-up only if `MC_SECURITY_CFG3`
//! reads 0, which is how it learns the SoC lost power; the part of it
//! modelled is the cold boot's ([`cold_boot_setup`], the BPMP parked again).
//! Then core 0 enters the kernel at the entry `CPU_SUSPEND` was given,
//! with its context in `X0`.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::{
    bus::{Access, Bus},
    memory::GuestRam,
    soc::{car, evp, flow, iram, mc, pmc, se, tzram},
};

/// Where the warmboot firmware releases core 0 after SC7: the monitor's
/// resident reset vector, `SECURE_SCRATCH34` (`dword_1F0149400`).
pub const RESUME_ENTRY: u64 = 0x7C01_2800;

/// TZRAM the monitor saves across SC7, and where its ciphertext goes.
pub const SAVED: u64 = 0x7C01_2000;
pub const SAVED_SIZE: usize = 0xE000;
const SAVED_BOUNCE: u64 = 0x4002_0000;
pub const SAVED_DRAM: u64 = 0x8001_0000;

/// Where the security engine's context goes (`0x1F01C8000`), and its size.
pub const CONTEXT: u64 = 0x8000_F000;
pub const CONTEXT_SIZE: usize = 0x840;

/// The BPMP's SC7 entry stub in the monitor's resident image (rodata,
/// `0x1F0148800..0x1F0148D64`), and where it runs.
pub const STUB: u64 = 0x7C01_A800;
pub const STUB_SIZE: usize = 0x564;
pub const STUB_IRAM: u64 = 0x4000_3000;

/// The 16-byte block the context save ends with (`unk_1F0148098`).
const CONTEXT_PATTERN: u64 = 0x7C01_A098;

/// The SC7 routine's stack: linked lists and bounce buffers.
pub const STACK: u64 = 0x7C01_0C00;
const IN_LIST: u64 = STACK;
const OUT_LIST: u64 = STACK + 0x10;
const BOUNCE: u64 = STACK + 0x40;
const SEED: u64 = STACK + 0x80;
const LAST_BLOCK_BUFFER: u64 = STACK + 0xC0;
const RANDOM_BLOCK: u64 = STACK + 0x100;

const PMC: u64 = pmc::BASE;
const DPD_ENABLE: u64 = PMC + 0x024;
const WAKE_STATUS: u64 = PMC + 0x014;
const PWRGATE_TOGGLE: u64 = PMC + 0x030;
const PWRGATE_STATUS: u64 = PMC + pmc::PWRGATE_STATUS;
const SCRATCH0: u64 = PMC + 0x050;
const SCRATCH1: u64 = PMC + 0x054;
const WAKE2_STATUS: u64 = PMC + 0x168;
const SCRATCH43: u64 = PMC + 0x22C;
const SEC_DISABLE3: u64 = PMC + 0x2D8;
const SECURE_SCRATCH21: u64 = PMC + 0x334;
const SECURE_SCRATCH24: u64 = PMC + 0x340;
const SECURE_SCRATCH32: u64 = PMC + 0x360;
const SECURE_SCRATCH34: u64 = PMC + 0x368;
const CNTRL2: u64 = PMC + 0x440;
const SEC_DISABLE8: u64 = PMC + 0x5C0;
const SECURE_SCRATCH112: u64 = PMC + 0xB18;

/// `SECURE_SCRATCH32`: the warm boot magic the warmboot firmware checks.
pub const WARMBOOT_MAGIC: u32 = 0x339;
/// `SECURE_SCRATCH21` bit 4, which the warmboot firmware ORs into
/// `FUSE_PRIVATEKEYDISABLE`.
const PRIVATE_KEY_DISABLE: u32 = 1 << 4;
/// `PMC_SCRATCH1`: where the boot ROM finds the warmboot firmware.
const WARMBOOT_VECTOR: u32 = 0x8000_D000;

/// `MC_SECURITY_CFG3`: 3 once the monitor set the SoC up, 0 after reset.
pub const MC_SECURITY_CFG3: u64 = mc::BASE + 0x9BC;
const SECURITY_CFG3_SET_UP: u32 = 3;

const HALT_COP_EVENTS: u64 = flow::BASE + 0x004;
/// `WAITEVENT` with no event: the BPMP parked for good.
pub const COP_PARKED: u32 = 0x4000_0000;
const HALT_MODE_WAIT_EVENT: u32 = 2;

const EVP_COP_VECTORS: u64 = evp::BASE + 0x200;
const RST_DEV_L_SET: u64 = car::BASE + 0x300;
const RST_DEV_L_CLR: u64 = car::BASE + 0x304;
const COP_RESET: u32 = 1 << 1;

/// Partitions core 0's power-down turns off, in order: CE0, C0NC, CRAIL.
const CORE0_PARTITIONS: [u32; 3] = [14, 15, 0];
const TOGGLE_START: u32 = 1 << 8;

const SE: u64 = se::BASE;
const SE_SECURITY: u64 = SE + 0x000;
const SE_OPERATION: u64 = SE + 0x008;
const SE_INT_STATUS: u64 = SE + 0x010;
const SE_CONFIG: u64 = SE + 0x014;
const SE_IN_LL_ADDR: u64 = SE + 0x018;
const SE_OUT_LL_ADDR: u64 = SE + 0x024;
const SE_HASH_RESULT: u64 = SE + 0x030;
const SE_CTX_SAVE_CONFIG: u64 = SE + 0x070;
const SE_CRYPTO_CONFIG: u64 = SE + 0x304;
const SE_CRYPTO_LAST_BLOCK: u64 = SE + 0x318;
const SE_CRYPTO_KEYTABLE_ADDR: u64 = SE + 0x31C;
const SE_CRYPTO_KEYTABLE_DATA: u64 = SE + 0x320;
const SE_CRYPTO_KEYTABLE_DST: u64 = SE + 0x330;
const SE_RNG_CONFIG: u64 = SE + 0x340;
const SE_STATUS: u64 = SE + 0x800;
const SE_ERR_STATUS: u64 = SE + 0x804;
const OP_START: u32 = 1;
const OP_CTX_SAVE: u32 = 3;
const INT_OP_DONE: u32 = 1 << 4;
const INT_ERROR: u32 = 1 << 16;
/// How many times an operation's completion is looked for: the engine
/// model completes at once.
const OP_POLLS: usize = 16;

/// Key slots of the suspend: the random one and the TZRAM key.
const SLOT_RANDOM: u32 = 2;
const SLOT_TZRAM: u32 = 3;
/// `SE_CONFIG` bits 31:16 for AES-256 encryption and decryption.
const MODES_AES256: u32 = 0x0202 << 16;

/// Why the suspend or resume could not be done.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Sc7Fail {
    /// A register no device here serves.
    Register { address: u64, is_write: bool },
    /// What the monitor panics at.
    Panic(&'static str),
    /// The machine stopped while the monitor waited.
    Stopped,
}

impl std::fmt::Display for Sc7Fail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Register { address, is_write } => write!(
                f,
                "{} register {address:#x} not served",
                if *is_write { "write to" } else { "read of" }
            ),
            Self::Panic(what) => write!(f, "the monitor panics: {what}"),
            Self::Stopped => write!(f, "the machine stopped"),
        }
    }
}

type Result<T, E = Sc7Fail> = std::result::Result<T, E>;

/// The SoC as core 0 reaches it: the bus, and DRAM.
pub(super) struct Port<'a> {
    pub(super) bus: &'a Bus,
    pub(super) ram: &'a GuestRam,
}

impl Port<'_> {
    pub(super) fn read(&self, address: u64) -> Result<u32> {
        match self.bus.read(0, address, 4) {
            Access::Done(value) => Ok(value as u32),
            _ => Err(Sc7Fail::Register {
                address,
                is_write: false,
            }),
        }
    }

    pub(super) fn write(&self, address: u64, value: u32) -> Result<()> {
        match self.bus.write(0, address, 4, value.into()) {
            Access::Done(()) => Ok(()),
            _ => Err(Sc7Fail::Register {
                address,
                is_write: true,
            }),
        }
    }

    fn modify(&self, address: u64, clear: u32, set: u32) -> Result<()> {
        let value = self.read(address)?;
        self.write(address, value & !clear | set)
    }

    /// Bytes on the bus (IRAM, TZRAM), a word at a time.
    fn read_bytes(&self, address: u64, out: &mut [u8]) -> Result<()> {
        for (index, word) in out.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            *word = self.read(address + index as u64 * 4)?.to_le_bytes();
        }
        Ok(())
    }

    pub(super) fn write_bytes(&self, address: u64, data: &[u8]) -> Result<()> {
        for (index, word) in data.as_chunks::<4>().0.iter().enumerate() {
            self.write(address + index as u64 * 4, u32::from_le_bytes(*word))?;
        }
        Ok(())
    }

    fn write_dram(&self, address: u64, data: &[u8]) -> Result<()> {
        self.ram
            .write(address, data)
            .ok_or(Sc7Fail::Panic("DRAM the monitor writes is not memory here"))
    }

    /// Starts engine operation `op` on one-entry lists and waits for it
    /// (`0x1F014140C`, then the error check `0x1F01421FC`).
    fn engine_op(&self, op: u32, output: (u64, u32), input: (u64, u32)) -> Result<()> {
        let list = |(address, size): (u64, u32)| {
            let (address, size) = if address == 0 {
                (0, 0)
            } else {
                (address, size)
            };
            let mut bytes = [0u8; 12];
            bytes[4..8].copy_from_slice(&(address as u32).to_le_bytes());
            bytes[8..].copy_from_slice(&size.to_le_bytes());
            bytes
        };
        self.write_bytes(IN_LIST, &list(input))?;
        self.write_bytes(OUT_LIST, &list(output))?;
        self.write(SE_IN_LL_ADDR, IN_LIST as u32)?;
        self.write(SE_OUT_LL_ADDR, OUT_LIST as u32)?;
        self.write(SE_ERR_STATUS, self.read(SE_ERR_STATUS)?)?;
        self.write(SE_INT_STATUS, self.read(SE_INT_STATUS)?)?;
        self.write(SE_OPERATION, op)?;
        if !(0..OP_POLLS).any(|_| self.read(SE_INT_STATUS).is_ok_and(|s| s & INT_OP_DONE != 0)) {
            return Err(Sc7Fail::Panic(
                "a security engine operation does not finish",
            ));
        }
        if self.read(SE_INT_STATUS)? & INT_ERROR != 0
            || self.read(SE_STATUS)? & 3 != 0
            || self.read(SE_ERR_STATUS)? != 0
        {
            return Err(Sc7Fail::Panic("the security engine reports an error"));
        }
        Ok(())
    }

    /// `0x1F01423B8`.
    fn check_engine(&self) -> Result<()> {
        if self.read(SE_ERR_STATUS)? != 0 || self.read(SE_INT_STATUS)? & INT_ERROR != 0 {
            return Err(Sc7Fail::Panic("the security engine reports an error"));
        }
        Ok(())
    }

    /// 16 random bytes to `buffer` (`0x1F0142A4C`).
    fn random_block(&self, buffer: u64) -> Result<()> {
        self.write(SE_CONFIG, 0x2000)?;
        self.write(SE_CRYPTO_CONFIG, 0x108)?;
        self.write(SE_RNG_CONFIG, 4)?;
        self.write(SE_CRYPTO_LAST_BLOCK, 0)?;
        self.engine_op(OP_START, (buffer, 16), (0, 0))
    }

    /// 256 random bits into `slot` (`0x1F0142338`).
    fn random_key(&self, slot: u32) -> Result<()> {
        self.write(SE_CONFIG, 0x2008)?;
        self.write(SE_CRYPTO_CONFIG, 0x108)?;
        self.write(SE_RNG_CONFIG, 4)?;
        self.write(SE_CRYPTO_KEYTABLE_DST, slot << 8)?;
        self.write(SE_CRYPTO_LAST_BLOCK, 0)?;
        self.engine_op(OP_START, (0, 0), (0, 0))?;
        self.write(SE_CRYPTO_KEYTABLE_DST, slot << 8 | 1)?;
        self.engine_op(OP_START, (0, 0), (0, 0))
    }

    /// A new secure random key (`0x1F01414F8`).
    fn new_srk(&self) -> Result<()> {
        self.write(SE_CONFIG, 0x200C)?;
        self.write(SE_CRYPTO_CONFIG, 0x108)?;
        self.write(SE_RNG_CONFIG, 6)?;
        self.write(SE_CRYPTO_LAST_BLOCK, 0)?;
        self.engine_op(OP_START, (0, 0), (0, 0))
    }

    /// Every word of `slot` zero (`0x1F0141960`).
    fn clear_slot(&self, slot: u32) -> Result<()> {
        for word in 0..16 {
            self.write(SE_CRYPTO_KEYTABLE_ADDR, slot << 4 | word)?;
            self.write(SE_CRYPTO_KEYTABLE_DATA, 0)?;
        }
        Ok(())
    }

    /// `SE_CONFIG` = `config`, `SE_CRYPTO_CONFIG` = `crypto`, then the
    /// AES-256 modes ORed into the config's low half kept, as the monitor
    /// writes them.
    fn aes_config(&self, config: u32, crypto: u32) -> Result<()> {
        self.write(SE_CONFIG, config)?;
        self.write(SE_CRYPTO_CONFIG, crypto)?;
        let config = self.read(SE_CONFIG)?;
        self.write(SE_CONFIG, config & 0xFFFF | MODES_AES256)
    }

    /// The 16 bytes at `source` decrypted with `key` into the first word
    /// quad of `slot` (`0x1F0141FBC`).
    fn decrypt_into_slot(&self, slot: u32, key: u32, source: u64) -> Result<()> {
        self.aes_config(0x108, key << 24)?;
        self.write(SE_CRYPTO_LAST_BLOCK, 0)?;
        self.write(SE_CRYPTO_KEYTABLE_DST, slot << 8)?;
        self.engine_op(OP_START, (0, 0), (source, 16))
    }

    /// `slot`'s original IV (`0x1F0141B8C`).
    fn set_iv(&self, slot: u32, iv: [u32; 4]) -> Result<()> {
        for (word, value) in iv.into_iter().enumerate() {
            self.write(SE_CRYPTO_KEYTABLE_ADDR, slot << 4 | word as u32 | 8)?;
            self.write(SE_CRYPTO_KEYTABLE_DATA, value)?;
        }
        Ok(())
    }

    /// AES-256-CBC encryption of `size` bytes, IV 0 (`0x1F01426CC`).
    fn cbc_encrypt(&self, slot: u32, output: u64, input: u64, size: u32) -> Result<()> {
        self.aes_config(0x1000, slot << 24 | 0x144)?;
        self.set_iv(slot, [0; 4])?;
        self.write(SE_CRYPTO_LAST_BLOCK, size / 16 - 1)?;
        self.engine_op(OP_START, (output, size), (input, size))
    }

    /// AES-256-CMAC of `size` bytes, a multiple of 16 (`0x1F0142828`).
    fn cmac(&self, slot: u32, input: u64, size: u32) -> Result<[u8; 16]> {
        // Subkey: the zero block encrypted (`0x1F0141A60`), doubled.
        self.aes_config(0x1000, slot << 24 | 0x100)?;
        self.write(SE_CRYPTO_LAST_BLOCK, 0)?;
        self.write_bytes(BOUNCE, &[0; 16])?;
        self.engine_op(OP_START, (BOUNCE, 16), (BOUNCE, 16))?;
        let mut subkey = [0; 16];
        self.read_bytes(BOUNCE, &mut subkey)?;
        let subkey = double(u128::from_be_bytes(subkey));

        self.aes_config(0x1004, slot << 24 | 0x145)?;
        self.set_iv(slot, [0; 4])?;
        let blocks = size / 16;
        if blocks > 1 {
            self.write(SE_CRYPTO_LAST_BLOCK, blocks - 2)?;
            self.engine_op(OP_START, (0, 0), (input, size))?;
            self.modify(SE_CRYPTO_CONFIG, 0, 0x80)?;
        }
        self.write(SE_CRYPTO_LAST_BLOCK, 0)?;
        let mut last = [0; 16];
        self.read_bytes(input + u64::from(size) - 16, &mut last)?;
        let last = (u128::from_be_bytes(last) ^ subkey).to_be_bytes();
        self.write_bytes(LAST_BLOCK_BUFFER, &last)?;
        self.engine_op(OP_START, (0, 0), (LAST_BLOCK_BUFFER, 16))?;
        let mut mac = [0; 16];
        self.read_bytes(SE_HASH_RESULT, &mut mac)?;
        Ok(mac)
    }

    /// One context save step: the engine's 16 bytes from the source
    /// `config` names into the context at `offset` (`0x1F0142074`).
    fn save_block(&self, config: u32, offset: u64, input: (u64, u32)) -> Result<()> {
        self.write(SE_CTX_SAVE_CONFIG, config)?;
        self.write(SE_CRYPTO_LAST_BLOCK, 0)?;
        self.engine_op(OP_CTX_SAVE, (BOUNCE, 16), input)?;
        let mut block = [0; 16];
        self.read_bytes(BOUNCE, &mut block)?;
        self.write_dram(CONTEXT + offset, &block)
    }

    /// The engine's context to [`CONTEXT`] (`0x1F0141610`).
    fn save_context(&self) -> Result<()> {
        self.new_srk()?;
        self.random_block(RANDOM_BLOCK)?;
        self.write(SE_CONFIG, 0x1000)?;
        self.save_block(0x8000_0000, 0, (RANDOM_BLOCK, 16))?;
        for quad in 0..2 {
            self.save_block(quad << 24, 0x10 + u64::from(quad) * 16, (0, 0))?;
        }
        for slot in 0..16 {
            for quad in 0..2 {
                let offset = 0x30 + u64::from(slot) * 32 + u64::from(quad) * 16;
                self.save_block(0x4000_0000 | slot << 8 | quad, offset, (0, 0))?;
            }
        }
        for (word, base) in [(2, 0x230), (3, 0x330)] {
            for slot in 0..16 {
                let offset = base + u64::from(slot) * 16;
                self.save_block(0x4000_0000 | slot << 8 | word, offset, (0, 0))?;
            }
        }
        for slot in 0..2 {
            for half in 0..2 {
                let key = 2 * slot + 1 - half;
                for block in 0..16 {
                    let offset = 0x430
                        + u64::from(slot) * 0x200
                        + u64::from(half) * 0x100
                        + u64::from(block) * 16;
                    self.save_block(0x2000_0000 | key << 16 | block << 12, offset, (0, 0))?;
                }
            }
        }
        self.save_block(0x8000_0000, 0x830, (CONTEXT_PATTERN, 16))?;
        self.write(SE_CTX_SAVE_CONFIG, 0xC000_0000)?;
        self.write(SE_CRYPTO_LAST_BLOCK, 0)?;
        self.engine_op(OP_CTX_SAVE, (CONTEXT, 0), (0, 0))?;
        self.write(SE_CONFIG, 0)?;
        self.engine_op(OP_CTX_SAVE, (CONTEXT, 0), (0, 0))
    }

    /// Writes `value` to four secure scratch registers from `first`, locks
    /// them for writing, checks them, and locks them for reading, as the
    /// monitor does for R and the MAC (`0x1F0143568`, `0x1F0143768`).
    fn locked_scratch(&self, first: u64, value: [u8; 16], locks: ScratchLocks) -> Result<()> {
        if self.lock_state(locks, true, true)? != LockState::None {
            return Err(Sc7Fail::Panic("secure scratch already locked"));
        }
        let words: [u32; 4] = std::array::from_fn(|index| {
            u32::from_le_bytes(value[index * 4..][..4].try_into().unwrap())
        });
        for (index, word) in words.iter().enumerate() {
            self.write(first + index as u64 * 4, *word)?;
        }
        self.modify(locks.register, 0, locks.write)?;
        let read_back: Result<Vec<u32>> =
            (0..4).map(|index| self.read(first + index * 4)).collect();
        if self.lock_state(locks, true, false)? != LockState::All || read_back? != words {
            return Err(Sc7Fail::Panic(
                "secure scratch does not hold what was written",
            ));
        }
        self.modify(locks.register, 0, locks.read)?;
        if self.lock_state(locks, true, true)? != LockState::All {
            return Err(Sc7Fail::Panic("secure scratch locks do not hold"));
        }
        Ok(())
    }

    fn lock_state(&self, locks: ScratchLocks, write: bool, read: bool) -> Result<LockState> {
        let value = self.read(locks.register)?;
        let mut set = 0;
        let mut asked = 0;
        for (wanted, bits) in [(write, locks.write), (read, locks.read)] {
            if !wanted {
                continue;
            }
            asked += 1;
            match value & bits {
                0 => {}
                held if held == bits => set += 1,
                _ => return Ok(LockState::Some),
            }
        }
        Ok(match set {
            0 => LockState::None,
            set if set == asked => LockState::All,
            _ => LockState::Some,
        })
    }

    /// Waits for the BPMP to halt itself, then parks it (end of
    /// `0x1F0147430`).
    pub(super) fn park_bpmp(&self, stop: &AtomicBool) -> Result<()> {
        while self.read(HALT_COP_EVENTS)? >> 29 != HALT_MODE_WAIT_EVENT {
            if stop.load(Ordering::Relaxed) {
                return Err(Sc7Fail::Stopped);
            }
            std::thread::sleep(std::time::Duration::from_micros(50));
        }
        self.write(HALT_COP_EVENTS, COP_PARKED)
    }
}

#[derive(Clone, Copy)]
struct ScratchLocks {
    register: u64,
    write: u32,
    read: u32,
}

/// `SECURE_SCRATCH24..27`: `SEC_DISABLE3` bits.
const SEED_LOCKS: ScratchLocks = ScratchLocks {
    register: SEC_DISABLE3,
    write: 0x55,
    read: 0xAA,
};
/// `SECURE_SCRATCH112..115`: `SEC_DISABLE8` bits.
const MAC_LOCKS: ScratchLocks = ScratchLocks {
    register: SEC_DISABLE8,
    write: 0x55_0000,
    read: 0xAA_0000,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LockState {
    None,
    Some,
    All,
}

/// CMAC's subkey doubling in GF(2^128).
fn double(block: u128) -> u128 {
    block << 1 ^ if block >> 127 != 0 { 0x87 } else { 0 }
}

/// What the cold boot sets up that the warm path relies on
/// (`0x1F0146C80`, `0x1F0147010`): where the boot ROM finds the warmboot
/// firmware, the warmboot's magic and the private key disable bit it
/// re-applies, the resume vector it hands core 0, and the memory
/// controller's "set up" mark.
pub(super) fn cold_boot_setup(port: &Port<'_>) -> Result<()> {
    port.write(SCRATCH1, WARMBOOT_VECTOR)?;
    port.modify(SECURE_SCRATCH21, 0, PRIVATE_KEY_DISABLE)?;
    port.write(SECURE_SCRATCH32, WARMBOOT_MAGIC)?;
    resume_setup(port)
}

/// The part of the set-up the resume redoes (`0x1F0147010`).
fn resume_setup(port: &Port<'_>) -> Result<()> {
    port.write(SECURE_SCRATCH34, RESUME_ENTRY as u32)?;
    port.write(MC_SECURITY_CFG3, SECURITY_CFG3_SET_UP)
}

/// Steps 1 to 3 of the module's suspend, the preconditions excepted.
pub(super) fn prepare(port: &Port<'_>) -> Result<()> {
    port.modify(CNTRL2, 0, 0x200)?;
    port.write(WAKE_STATUS, u32::MAX)?;
    port.write(WAKE2_STATUS, u32::MAX)
}

/// Whether the BPMP is parked as `CPU_SUSPEND` requires.
pub(super) fn bpmp_parked(port: &Port<'_>) -> Result<bool> {
    Ok(port.read(HALT_COP_EVENTS)? == COP_PARKED)
}

/// Steps 3 to 10 of the module's suspend.
pub(super) fn suspend(port: &Port<'_>) -> Result<()> {
    port.write(SCRATCH0, 1)?;
    port.modify(DPD_ENABLE, 0, 2)?;

    port.random_block(SEED)?;
    let mut seed = [0; 16];
    port.read_bytes(SEED, &mut seed)?;
    port.locked_scratch(SECURE_SCRATCH24, seed, SEED_LOCKS)?;

    port.random_key(SLOT_RANDOM)?;
    port.check_engine()?;
    port.modify(SE_SECURITY, 0, 1 << 16)?;
    port.save_context()?;
    port.write(SCRATCH43, CONTEXT as u32)?;
    port.modify(SE_SECURITY, 1 << 16, 0)?;
    port.check_engine()?;

    port.clear_slot(SLOT_TZRAM)?;
    port.decrypt_into_slot(SLOT_TZRAM, SLOT_RANDOM, SEED)?;
    port.cbc_encrypt(SLOT_TZRAM, SAVED_BOUNCE, SAVED, SAVED_SIZE as u32)?;
    let mut ciphertext = vec![0; SAVED_SIZE];
    port.read_bytes(SAVED_BOUNCE, &mut ciphertext)?;
    port.write_dram(SAVED_DRAM, &ciphertext)?;
    let mac = port.cmac(SLOT_TZRAM, SAVED, SAVED_SIZE as u32)?;
    port.locked_scratch(SECURE_SCRATCH112, mac, MAC_LOCKS)?;

    port.write(EVP_COP_VECTORS, STUB_IRAM as u32)?;
    for vector in 1..8 {
        port.write(EVP_COP_VECTORS + vector * 4, STUB_IRAM as u32 + 4)?;
    }
    port.write(RST_DEV_L_SET, COP_RESET)?;
    let mut stub = vec![0; STUB_SIZE];
    port.read_bytes(STUB, &mut stub)?;
    port.write_bytes(STUB_IRAM, &stub)?;
    let mut copied = vec![0; STUB_SIZE];
    port.read_bytes(STUB_IRAM, &mut copied)?;
    if copied != stub {
        return Err(Sc7Fail::Panic(
            "the SC7 stub does not compare equal in IRAM",
        ));
    }
    port.write(RST_DEV_L_CLR, COP_RESET)?;
    port.write(HALT_COP_EVENTS, 0)?;

    for partition in CORE0_PARTITIONS {
        if port.read(PWRGATE_STATUS)? & 1 << partition != 0 {
            port.write(PWRGATE_TOGGLE, TOGGLE_START | partition)?;
        }
    }
    Ok(())
}

/// The resume's SoC part: the set-up again if the SoC lost it.
pub(super) fn resume(port: &Port<'_>, stop: &AtomicBool) -> Result<()> {
    if port.read(MC_SECURITY_CFG3)? != 0 {
        return Ok(());
    }
    resume_setup(port)?;
    port.park_bpmp(stop)
}

const _: () = assert!(STUB_IRAM + STUB_SIZE as u64 <= iram::BASE + iram::SIZE);
const _: () = assert!(SAVED + SAVED_SIZE as u64 == tzram::BASE + tzram::SIZE);

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use aes::{
        Aes256,
        cipher::{BlockDecrypt, BlockEncrypt, KeyInit},
    };

    use super::*;
    use crate::{
        boot::{BootParams, DramSize},
        keys::KeySet,
        secmon::{KERNEL_SMC, SLEEP_REQUIREMENTS, SecureMonitor, SmcEffect, function},
        soc::{
            Notes, car::Car, evp::Evp, iram::Iram, pmc::Pmc, regs::RegisterFile, se::Se,
            tzram::Tzram,
        },
    };

    /// A console after the monitor's cold boot, as far as its SC7
    /// preparation reaches: PMC, CAR, flow, EVP, IRAM, TZRAM, the security
    /// engine over them, and DRAM.
    struct Console {
        monitor: SecureMonitor,
        bus: Bus,
        tzram: Arc<Tzram>,
        iram: Arc<Iram>,
        ram: Arc<GuestRam>,
        engine: mc::DeviceMemory,
        notes: Arc<Notes>,
    }

    fn console() -> Console {
        let ram = Arc::new(GuestRam::new(0x8000_0000, 0x2_0000).unwrap());
        let iram = Arc::new(Iram::new());
        let tzram = Arc::new(Tzram::new());
        // TZRAM's saved part: something that is not all one byte.
        let contents: Vec<u8> = (0..SAVED_SIZE)
            .map(|index| (index * 7 + index / 251) as u8)
            .collect();
        tzram.load((SAVED - tzram::BASE) as u32, &contents).unwrap();
        let mut bus = Bus::default();
        let pmc_device = Arc::new(Pmc::new());
        bus.map(
            "PMC",
            pmc::BASE,
            pmc::SIZE,
            Box::new(Arc::clone(&pmc_device)),
        );
        bus.map("CAR", car::BASE, car::SIZE, Box::new(Car::new()));
        bus.map(
            "FLOW",
            flow::BASE,
            flow::SIZE,
            Box::new(RegisterFile::zeroed([(0x004, COP_PARKED)])),
        );
        bus.map("EVP", evp::BASE, evp::SIZE, Box::new(Evp::new()));
        bus.map("IRAM", iram::BASE, iram::SIZE, Box::new(Arc::clone(&iram)));
        bus.map(
            "TZRAM",
            tzram::BASE,
            tzram::SIZE,
            Box::new(Arc::clone(&tzram)),
        );
        let memory = se::BootMemory::new(Arc::clone(&iram), Arc::clone(&tzram), Arc::clone(&ram));
        let notes = Arc::new(Notes::default());
        let se = Arc::new(Se::new(Box::new(memory), Arc::clone(&notes), 5));
        // As `Soc::new` wires it: the SRK goes to PMC secure scratch.
        se.on_srk_saved(Box::new(move |words| {
            for (offset, word) in se::SRK_SCRATCH.into_iter().zip(words) {
                crate::bus::MmioDevice::write(pmc_device.as_ref(), 0, offset, 4, word.into())
                    .expect("the PMC takes its secure scratch registers");
            }
        }));
        bus.map("SE", se::BASE, se::SIZE, Box::new(se));
        // The SoC as the firmware leaves it for sleep.
        for (register, mask, required, _) in SLEEP_REQUIREMENTS.into_iter().skip(1) {
            let value = read(&bus, register);
            assert!(matches!(
                bus.write(0, register, 4, (value | mask & required).into()),
                Access::Done(())
            ));
        }
        let controller = Arc::new(mc::Mc::new(DramSize::Gb4));
        let engine = mc::DeviceMemory::new(&controller, &ram, mc::SMMU_SE_ASID);
        let monitor = SecureMonitor::new(&BootParams::default(), KeySet::default());
        monitor.state.lock().unwrap().boot_chain = true;
        Console {
            monitor,
            bus,
            tzram,
            iram,
            ram,
            engine,
            notes,
        }
    }

    fn read(bus: &Bus, address: u64) -> u32 {
        match bus.read(0, address, 4) {
            Access::Done(value) => value as u32,
            _ => panic!("register {address:#x}"),
        }
    }

    fn scratch(bus: &Bus, first: u64) -> [u8; 16] {
        let mut bytes = [0; 16];
        for (index, word) in bytes.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            *word = read(bus, first + index as u64 * 4).to_le_bytes();
        }
        bytes
    }

    /// The 32 key bytes of `slot`, read through the key table registers.
    fn key(bus: &Bus, slot: u32) -> [u8; 32] {
        let mut bytes = [0; 32];
        for (word, chunk) in bytes.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            bus.write(
                0,
                SE_CRYPTO_KEYTABLE_ADDR,
                4,
                u64::from(slot << 4 | word as u32),
            );
            *chunk = read(bus, SE_CRYPTO_KEYTABLE_DATA).to_le_bytes();
        }
        bytes
    }

    fn cmac(key: &[u8; 32], data: &[u8]) -> [u8; 16] {
        let cipher = Aes256::new(key.into());
        let mut subkey = aes::Block::default();
        cipher.encrypt_block(&mut subkey);
        let subkey = double(u128::from_be_bytes(subkey.into()));
        let mut state = 0u128;
        let blocks = data.as_chunks::<16>().0;
        for (index, block) in blocks.iter().enumerate() {
            let mut value = u128::from_be_bytes(*block) ^ state;
            if index + 1 == blocks.len() {
                value ^= subkey;
            }
            let mut block = aes::Block::from(value.to_be_bytes());
            cipher.encrypt_block(&mut block);
            state = u128::from_be_bytes(block.into());
        }
        state.to_be_bytes()
    }

    #[test]
    fn suspend_saves_tzram_encrypted_and_maced_under_a_key_the_warmboot_can_derive() {
        let console = console();
        let mut x = [0; 8];
        x[..4].copy_from_slice(&[function::CPU_SUSPEND.into(), 0x0201_001B, 0x8000_1000, 7]);
        let effect = console
            .monitor
            .call(0, KERNEL_SMC, &mut x, &console.bus, &console.engine);
        assert_eq!(
            effect,
            SmcEffect::SystemSuspend,
            "{:?}",
            console.notes.take()
        );
        assert_eq!(
            console.monitor.state.lock().unwrap().suspended,
            Some((0x8000_1000, 7))
        );
        let bus = &console.bus;

        assert_eq!(read(bus, SCRATCH0), 1);
        assert_eq!(read(bus, DPD_ENABLE) & 2, 2);
        assert_eq!(read(bus, WAKE_STATUS), 0);
        assert_eq!(read(bus, CNTRL2) & 0x200, 0x200);
        assert_eq!(read(bus, SCRATCH43), CONTEXT as u32);
        assert_eq!(read(bus, SEC_DISABLE3) & 0xFF, 0xFF);
        assert_eq!(read(bus, SEC_DISABLE8) & 0xFF_0000, 0xFF_0000);

        // Slot 3 is AES-256-ECB-decrypt(slot 2, R) in its first quad.
        let seed = scratch(bus, SECURE_SCRATCH24);
        assert_ne!(seed, [0; 16]);
        let random = key(bus, SLOT_RANDOM);
        let mut derived = aes::Block::from(seed);
        Aes256::new(&random.into()).decrypt_block(&mut derived);
        let mut tzram_key = [0; 32];
        tzram_key[..16].copy_from_slice(&derived);
        assert_eq!(key(bus, SLOT_TZRAM), tzram_key);

        // DRAM holds TZRAM under AES-256-CBC with IV 0, and the MAC is its CMAC.
        let mut plain = vec![0; SAVED_SIZE];
        console
            .tzram
            .copy_out((SAVED - tzram::BASE) as u32, &mut plain)
            .unwrap();
        let mut saved = vec![0; SAVED_SIZE];
        console.ram.read(SAVED_DRAM, &mut saved).unwrap();
        assert_ne!(saved, plain);
        let cipher = Aes256::new(&tzram_key.into());
        let mut previous = [0u8; 16];
        let mut decrypted = Vec::with_capacity(SAVED_SIZE);
        for block in saved.as_chunks::<16>().0 {
            let mut value = aes::Block::from(*block);
            cipher.decrypt_block(&mut value);
            decrypted.extend(value.iter().zip(previous).map(|(byte, iv)| byte ^ iv));
            previous = *block;
        }
        assert_eq!(decrypted, plain);
        assert_eq!(scratch(bus, SECURE_SCRATCH112), cmac(&tzram_key, &plain));

        // The engine's context went to DRAM.
        let mut context = vec![0; CONTEXT_SIZE];
        console.ram.read(CONTEXT, &mut context).unwrap();
        assert!(context.iter().any(|&byte| byte != 0));

        // The BPMP restarted on the stub, core 0's partitions off.
        let mut stub = vec![0; STUB_SIZE];
        console
            .tzram
            .copy_out((STUB - tzram::BASE) as u32, &mut stub)
            .unwrap();
        let mut copied = vec![0; STUB_SIZE];
        console
            .iram
            .copy_out((STUB_IRAM - iram::BASE) as u32, &mut copied)
            .unwrap();
        assert_eq!(copied, stub);
        assert_eq!(read(bus, EVP_COP_VECTORS), 0x4000_3000);
        assert_eq!(read(bus, EVP_COP_VECTORS + 4), 0x4000_3004);
        assert_eq!(read(bus, HALT_COP_EVENTS), 0);
        assert_eq!(read(bus, PWRGATE_STATUS) & (1 << 0 | 1 << 14 | 1 << 15), 0);

        // Asleep, the monitor lets core 0 back in only at the resume vector.
        assert_eq!(console.monitor.state.lock().unwrap().cores_on[0], false);
    }

    #[test]
    fn a_second_suspend_on_locked_scratch_panics_as_the_monitor_does() {
        let console = console();
        let bus = &console.bus;
        assert!(matches!(
            bus.write(0, SEC_DISABLE3, 4, 0x55),
            Access::Done(())
        ));
        let mut x = [0; 8];
        x[..4].copy_from_slice(&[function::CPU_SUSPEND.into(), 0x0201_001B, 0x8000_1000, 7]);
        let effect = console
            .monitor
            .call(0, KERNEL_SMC, &mut x, bus, &console.engine);
        assert!(matches!(
            effect,
            SmcEffect::Abort(crate::secmon::Abort::SuspendPanic(_))
        ));
    }

    #[test]
    fn suspend_needs_the_bpmp_parked() {
        let console = console();
        assert!(matches!(
            console.bus.write(0, HALT_COP_EVENTS, 4, 0x5000_0000),
            Access::Done(())
        ));
        let mut x = [0; 8];
        x[..4].copy_from_slice(&[function::CPU_SUSPEND.into(), 0x0201_001B, 0x8000_1000, 7]);
        let effect = console
            .monitor
            .call(0, KERNEL_SMC, &mut x, &console.bus, &console.engine);
        assert!(matches!(
            effect,
            SmcEffect::Abort(crate::secmon::Abort::SocNotReadyForSleep {
                value: 0x5000_0000,
                ..
            })
        ));
    }
}
