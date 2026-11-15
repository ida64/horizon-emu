//! Secure monitor (TrustZone) as the kernel sees it.
//!
//! The kernel never touches EL3 state; it issues `SMC #1` with a function
//! identifier in `W0` and arguments in `X1`-`X7`, and gets `X0`-`X7` back with
//! a status in `X0` (`0x80065460`).
//!
//! `SMC #0` carries calls the kernel forwards for a privileged process
//! (`0x800654a0`), with its own function numbering. Bits 15:9 of those
//! identifiers flag which of `X1`-`X7` are user pointers; the kernel replaces
//! each with the physical address of the page before issuing the SMC
//! (`0x800d4440`).
//!
//! Calls the kernel makes for itself while booting, in order of first use
//! (addresses are link-time addresses in the retail image described in
//! [`crate::boot`]):
//!
//! | Function | In | Out | First use |
//! |---|---|---|---|
//! | [`READ_WRITE_REGISTER`](function::READ_WRITE_REGISTER) | `X1` address, `W2` mask, `W3` value | `W1` previous value | loader reads `MC_EMEM_CFG` (`0x8031c858`) |
//! | [`GET_CONFIG`](function::GET_CONFIG) | `W1` item | `X1`-`X4` value | loader, item 12 (`0x8031c874`) |
//! | [`GENERATE_RANDOM_BYTES`](function::GENERATE_RANDOM_BYTES) | `X1` byte count, 1-56 | bytes packed into `X1`-`X7` | loader KASLR (`0x8031c7bc`) |
//! | [`CPU_ON`](function::CPU_ON) | `X1` core, `X2` entry PA, `X3` context | PSCI status | `0x800cecc0` |
//! | [`CONFIGURE_CARVEOUT`](function::CONFIGURE_CARVEOUT) | `X1` index, `X2` address, `X3` size | | `0x800d25f0` |
//! | [`PANIC`](function::PANIC) | `W1` code | does not return | `0x800d2650` |
//!
//! And when the console goes to sleep, from the kernel's sleep handler on
//! each core (mesosphere `kern_k_sleep_manager_asm.s`, `CpuSleepHandler`):
//! cores 1 to 3 call [`CPU_OFF`](function::CPU_OFF), core 0 waits for their
//! power partitions to read off and calls
//! [`CPU_SUSPEND`](function::CPU_SUSPEND). Neither returns.
//!
//! # Sleep
//!
//! On the console the monitor then powers the whole SoC down but for the
//! PMC and the memory, which keeps itself; a wake event starts the SoC
//! from its boot ROM, whose warm boot path leads back into the monitor,
//! and the monitor starts core 0 where the kernel asked. Here nothing
//! loses power: core 0 stands still until a wake event and then starts at
//! that address, and every device is as the firmware left it instead of
//! out of reset. The firmware's drivers put back what a real sleep loses,
//! which here writes what is already there. Of what the monitor checks
//! before sleeping (exosphere `secmon_smc_power_management.cpp`,
//! `ValidateSocStateForSuspend`) only that the other cores are off is
//! checked.
//!
//! Config items read while booting: 1, 10 (memory mode), 11 and 12 (kernel
//! configuration). Every wrapper in the kernel treats a non-zero status as
//! fatal, so the only error paths here are argument bounds the kernel itself
//! stays within; anything not modeled stops the machine instead of handing the
//! guest a made-up answer.

use std::{ops::Range, sync::Mutex};

pub mod crypto;

use crate::{
    boot::{BootParams, CORE_COUNT, DramSize, KernelConfiguration},
    bus::{Access, Bus},
    keys::{Key, KeySet},
    soc::mc::DeviceMemory,
};

use self::crypto::{CipherMode, Crypto, CryptoError};

/// Immediate of the SMC instruction for kernel-originated calls.
pub const KERNEL_SMC: u16 = 1;

/// Immediate of the SMC instruction for calls forwarded from userland.
pub const USER_SMC: u16 = 0;

pub mod function {
    /// `W1` power state, `X2` entry PA, `X3` context. The one state there
    /// is: [`super::POWER_STATE_SLEEP`].
    pub const CPU_SUSPEND: u32 = 0xC400_0001;
    /// No arguments: switches the calling core off.
    pub const CPU_OFF: u32 = 0x8400_0002;
    pub const CPU_ON: u32 = 0xC400_0003;
    pub const GET_CONFIG: u32 = 0xC300_0004;
    pub const GENERATE_RANDOM_BYTES: u32 = 0xC300_0005;
    pub const PANIC: u32 = 0xC300_0006;
    pub const CONFIGURE_CARVEOUT: u32 = 0xC300_0007;
    pub const READ_WRITE_REGISTER: u32 = 0xC300_0008;
}

/// Functions reachable through [`USER_SMC`].
pub mod user_function {
    /// `W1` item in, `X1`-`X4` value out; the items are those of the kernel's
    /// [`GET_CONFIG`](super::function::GET_CONFIG).
    pub const GET_CONFIG: u32 = 0xC300_0002;
    /// `W1` item, `X3` value in. Only
    /// [`IS_CHARGER_HI_Z_MODE_ENABLED`](super::config_item::IS_CHARGER_HI_Z_MODE_ENABLED)
    /// can be set; `ptm` sets it to 0 on a console that has been set up
    /// before (`spl` SetConfig, command 5). That the value is in `X3`, with
    /// `X2` unused, follows Atmosphère's public monitor
    /// (`secmon_smc_info.cpp`), from memory \[assumption\]: the one call seen
    /// has zero in both.
    pub const SET_CONFIG: u32 = 0xC300_0401;
    /// `X1` operation key in; the operation's own status out in `X1` (`spl`
    /// fails with "unknown secure monitor error" on anything it does not
    /// know there).
    pub const GET_RESULT: u32 = 0xC300_0003;
    /// As the kernel's [`GENERATE_RANDOM_BYTES`](super::function::GENERATE_RANDOM_BYTES).
    pub const GENERATE_RANDOM_BYTES: u32 = 0xC300_0006;
    /// `X1`-`X2` kek source, `X3` key generation, `W4` options in; sealed
    /// access key out in `X1`-`X2`.
    pub const GENERATE_AES_KEK: u32 = 0xC300_0007;
    /// `X1` key slot, `X2`-`X3` access key, `X4`-`X5` key source in.
    pub const LOAD_AES_KEY: u32 = 0xC300_0008;
    /// `X1`-`X2` key source, `X3` key generation, `W4` which key in; the key
    /// out in `X1`-`X2`.
    pub const GENERATE_SPECIFIC_AES_KEY: u32 = 0xC300_000A;
    /// `W1` mode (key slot in bits 2..0, cipher mode in bits 5..4),
    /// `X2`-`X3` IV or counter, `W4` and `W5` addresses of the security
    /// engine's source and destination buffer lists, `W6` byte count in;
    /// operation key out in `X1`. All addresses are the engine's device
    /// addresses: `spl` maps its work buffer for the engine at `0x8003_5000`
    /// behind the SMMU. The monitor raises
    /// [`SECURITY_ENGINE_INTERRUPT`](super::SECURITY_ENGINE_INTERRUPT) when
    /// the engine is done, and the caller collects the outcome with
    /// [`GET_RESULT`].
    pub const COMPUTE_AES: u32 = 0xC300_0009;
    /// `X1` key slot, `X2` address of the data, `X3` its length in; the
    /// AES-CMAC out in `X1`-`X2`. Bit 10 of the identifier marks `X2` as a
    /// user pointer, so what arrives is the physical address the kernel
    /// found for it, good for the rest of that page.
    pub const COMPUTE_CMAC: u32 = 0xC300_040B;
    /// `X1`-`X2` access key, `W3` mode, `X4` address of the data, `X5` its
    /// length, `X6`-`X7` key source in; the data decrypted in place (the
    /// format is [`super::crypto::seal_device_unique_data`]'s). Bit 12 of
    /// the identifier marks `X4` as a user pointer, which the kernel turns
    /// into a physical address; `spl` passes its work buffer (`0x35000`,
    /// text `0x11340`). Mode 0 hands the plain data back; the modes that
    /// import a key into the monitor (1 to 4) are not modeled. `ssl`
    /// reaches it through `spl` to unwrap its TLS client key of key
    /// generation 0 (`0x35f80`), and `spl` copies the payload, all but the
    /// last `0x30` bytes, back to it.
    pub const DECRYPT_DEVICE_UNIQUE_DATA: u32 = 0xC300_100D;
}

/// Config items. Numbers 1 and 10-12 are established by what the kernel
/// does with them; the meaning of the others, and the values served for
/// them, follow the public documentation of the monitor's interface.
pub mod config_item {
    pub const DISABLE_PROGRAM_VERIFICATION: u32 = 1;
    pub const DRAM_ID: u32 = 2;
    /// Interrupt the monitor raises when an asynchronous operation completes.
    pub const SECURITY_ENGINE_INTERRUPT_NUMBER: u32 = 3;
    pub const HARDWARE_TYPE: u32 = 5;
    pub const HARDWARE_STATE: u32 = 6;
    /// Whether the console was started into its recovery (maintenance) mode.
    pub const IS_RECOVERY_BOOT: u32 = 7;
    pub const DEVICE_ID: u32 = 8;
    /// Whether the console is a shop's interactive display unit, a fuse
    /// setting (meaning as publicly documented). It is not.
    pub const RETAIL_INTERACTIVE_DISPLAY_STATE: u32 = 14;
    /// Which regulators feed the processor and the GPU. Zero: a MAX77621
    /// each, the only arrangement of the original console (value as publicly
    /// documented).
    pub const REGULATOR_TYPE: u32 = 15;
    /// Generation of the console-unique key the console was provisioned
    /// with. Zero: the original one, the only one the key ladder here has.
    pub const DEVICE_UNIQUE_KEY_GENERATION: u32 = 16;
    pub const MEMORY_MODE: u32 = 10;
    pub const IS_DEVELOPMENT_FUNCTION_ENABLED: u32 = 11;
    pub const KERNEL_CONFIGURATION: u32 = 12;
    /// Whether the monitor puts the charger into high-impedance mode when
    /// the console powers off. Settable; starts out disabled \[assumption:
    /// nothing set it before the first call seen, which writes 0\].
    pub const IS_CHARGER_HI_Z_MODE_ENABLED: u32 = 13;
    /// From here up the numbers are Atmosphère's: its monitor answers them
    /// with its own version and settings (exosphere `secmon_smc_info.hpp`,
    /// `ExosphereApiVersion` on), and homebrew asks for the first to learn
    /// whether it runs under Atmosphère (libnx; NX-Shell does so when it
    /// starts). Nintendo's monitor, which this is, has no such items: the
    /// call fails as for any item it does not know, with the status
    /// exosphere gives in that case \[inference: exosphere reproduces the
    /// original's answers\].
    pub const FIRST_OF_ATMOSPHERE: u32 = 65000;
}

/// Interrupt ID the monitor signals asynchronous completions with.
pub const SECURITY_ENGINE_INTERRUPT: u64 = 44;

/// `GICD_ISPENDR1`, where the monitor sets that interrupt pending.
const SET_PENDING_REGISTER: u64 = 0x5004_1204;

/// Most the engine is asked to process in one operation here; its buffers
/// are a few pages of the caller's.
const MAX_ENGINE_BYTES: u64 = 0x10_0000;

const PAGE_SIZE: u64 = 0x1000;

/// Hardware type of the original console model (the Tegra X1 board this
/// machine is).
const HARDWARE_TYPE_ICOSA: u64 = 0;

/// Hardware state of a retail unit, as opposed to a development one.
const HARDWARE_STATE_PRODUCTION: u64 = 1;

const STATUS_SUCCESS: u64 = 0;
const STATUS_INVALID_ARGUMENT: u64 = 2;
/// An asynchronous operation is still to be collected.
const STATUS_BUSY: u64 = 3;
const STATUS_NO_ASYNC_OPERATION: u64 = 4;
const STATUS_INVALID_ASYNC_OPERATION: u64 = 5;

const PSCI_SUCCESS: u64 = 0;
const PSCI_INVALID_PARAMETERS: u64 = -2i64 as u64;
const PSCI_DENIED: u64 = -3i64 as u64;
const PSCI_ALREADY_ON: u64 = -4i64 as u64;

/// The power state the kernel suspends core 0 into: a power-down state
/// (bit 16) with ID 27, the whole SoC off (exosphere's `PowerStateId_Sc7`;
/// the kernel passes `0x0201001b`). The monitor denies any other.
pub const POWER_STATE_SLEEP: u32 = 1 << 16 | 27;
const POWER_STATE_TYPE_AND_ID: u32 = 0x1_FFFF;

/// The PMC's power partition switch, its status, and the partition of
/// each core (exosphere `secmon_smc_power_management.cpp`,
/// `CpuPowerGateTogglePartitionIds`: CE0 is 14, CE1 to CE3 are 9 to 11).
const PMC_PWRGATE_TOGGLE: u64 = 0x7000_E430;
const PMC_PWRGATE_STATUS: u64 = 0x7000_E438;
const PWRGATE_TOGGLE_START: u64 = 1 << 8;
const CORE_PARTITIONS: [u32; CORE_COUNT] = [14, 9, 10, 11];

const MAX_RANDOM_BYTES: u64 = 56;
const CARVEOUT_COUNT: usize = 2;

/// Register blocks the monitor accesses on the kernel's behalf: PMC and the
/// memory controller.
const PROXIED_REGISTERS: [Range<u64>; 2] = [0x7000_E400..0x7000_F000, 0x7001_9000..0x7001_A000];

/// What the machine must do once [`SecureMonitor::call`] returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmcEffect {
    /// Result registers are final; resume the caller.
    Resume,
    /// Resume the caller and start `core` at physical `entry` with `context` in `X0`.
    StartCore {
        core: usize,
        entry: u64,
        context: u64,
    },
    /// The calling core is switched off. The call does not return; the
    /// core runs again when another turns it on.
    PowerOff,
    /// The console sleeps. The call does not return; when the console
    /// wakes, the calling core starts at physical `entry` with `context`
    /// in `X0`.
    Sleep { entry: u64, context: u64 },
    /// The kernel reported a fatal error. The call does not return.
    Panic { code: u32 },
    /// The call is outside what is modeled; registers are untouched.
    Unimplemented(Unimplemented),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unimplemented {
    Function {
        immediate: u16,
        id: u32,
    },
    ConfigItem(u32),
    /// A register the proxied device does not model.
    Register {
        address: u64,
        is_write: bool,
    },
    /// A key operation, with what it would need.
    Crypto(&'static str),
    /// The console's monitor would not have carried on either: it aborts.
    Abort(&'static str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Carveout {
    pub address: u64,
    pub size: u64,
}

pub struct SecureMonitor {
    dram: DramSize,
    kernel_configuration: KernelConfiguration,
    development_function_enabled: bool,
    program_verification_disabled: bool,
    device_id: u64,
    state: Mutex<State>,
}

struct State {
    rng: SplitMix64,
    /// A stream of its own, so that key handling does not shift the random
    /// numbers the kernel's layout is derived from.
    key_rng: SplitMix64,
    cores_on: [bool; CORE_COUNT],
    carveouts: [Carveout; CARVEOUT_COUNT],
    crypto: Crypto,
    /// Key of the finished asynchronous operation nobody has collected yet.
    async_operation: Option<u64>,
    /// [`config_item::IS_CHARGER_HI_Z_MODE_ENABLED`]. Nothing here powers
    /// the console off, so the flag is only remembered.
    charger_hi_z_mode: bool,
}

impl SecureMonitor {
    /// Monitor state at kernel entry: only core 0 is running. `keys` is the
    /// console's key material; key operations that need a missing key stop
    /// the machine.
    pub fn new(params: &BootParams, keys: KeySet) -> Self {
        let mut key_rng = SplitMix64(!params.rng_seed);
        let mut seal_key = Key::default();
        key_rng.fill(&mut seal_key);
        let mut cores_on = [false; CORE_COUNT];
        cores_on[0] = true;
        Self {
            dram: params.dram,
            kernel_configuration: params.kernel_configuration,
            development_function_enabled: params.development_function_enabled,
            program_verification_disabled: params.program_verification_disabled,
            device_id: params.device_id,
            state: Mutex::new(State {
                rng: SplitMix64(params.rng_seed),
                key_rng,
                cores_on,
                carveouts: [Carveout::default(); CARVEOUT_COUNT],
                crypto: Crypto::new(keys, seal_key),
                async_operation: None,
                charger_hi_z_mode: false,
            }),
        }
    }

    /// Serves one SMC of `core`. `x` is `X0`-`X7` at the instruction and is
    /// rewritten with the values the caller resumes with. `bus` is where
    /// the registers the monitor proxies live, `engine` the memory the
    /// security engine reaches.
    pub fn call(
        &self,
        core: usize,
        immediate: u16,
        x: &mut [u64; 8],
        bus: &Bus,
        engine: &DeviceMemory,
    ) -> SmcEffect {
        let id = x[0] as u32;
        let unimplemented = SmcEffect::Unimplemented(Unimplemented::Function { immediate, id });
        if immediate == USER_SMC {
            return match id {
                user_function::GET_CONFIG => self.get_config(x),
                user_function::SET_CONFIG => self.set_config(x),
                user_function::GENERATE_RANDOM_BYTES => self.generate_random_bytes(x),
                user_function::GENERATE_AES_KEK => self.key_operation(x, |crypto, x| {
                    let access_key = crypto.generate_aes_kek(&key_at(x, 1), x[3], x[4] as u32)?;
                    set_key_at(x, 1, &access_key);
                    Ok(())
                }),
                user_function::LOAD_AES_KEY => self.key_operation(x, |crypto, x| {
                    crypto.load_aes_key(x[1], &key_at(x, 2), &key_at(x, 4))
                }),
                user_function::COMPUTE_AES => self.compute_aes(x, bus, engine),
                user_function::COMPUTE_CMAC => {
                    let (address, len) = (x[2], x[3]);
                    let mut data = vec![0; len.min(PAGE_SIZE) as usize];
                    if address % PAGE_SIZE + len > PAGE_SIZE
                        || engine.ram().read(address, &mut data).is_none()
                    {
                        let what = "a CMAC over data that is not within one page of DRAM";
                        return SmcEffect::Unimplemented(Unimplemented::Crypto(what));
                    }
                    self.key_operation(x, |crypto, x| {
                        let mac = crypto.compute_cmac(x[1], &data)?;
                        set_key_at(x, 1, &mac);
                        Ok(())
                    })
                }
                user_function::DECRYPT_DEVICE_UNIQUE_DATA => {
                    self.decrypt_device_unique_data(x, engine)
                }
                user_function::GENERATE_SPECIFIC_AES_KEY => self.key_operation(x, |crypto, x| {
                    let key = crypto.generate_specific_aes_key(&key_at(x, 1), x[3], x[4])?;
                    set_key_at(x, 1, &key);
                    Ok(())
                }),
                user_function::GET_RESULT => {
                    let mut state = self.state.lock().unwrap();
                    x[0] = match state.async_operation {
                        None => STATUS_NO_ASYNC_OPERATION,
                        Some(key) if key != x[1] => STATUS_INVALID_ASYNC_OPERATION,
                        Some(_) => {
                            state.async_operation = None;
                            // The engine's operations cannot fail here.
                            x[1] = STATUS_SUCCESS;
                            STATUS_SUCCESS
                        }
                    };
                    SmcEffect::Resume
                }
                _ => unimplemented,
            };
        }
        if immediate != KERNEL_SMC {
            return unimplemented;
        }
        let mut state = self.state.lock().unwrap();
        match id {
            function::GET_CONFIG => return self.get_config(x),
            function::GENERATE_RANDOM_BYTES => {
                drop(state);
                return self.generate_random_bytes(x);
            }
            function::READ_WRITE_REGISTER => {
                let (address, mask, value) = (x[1], x[2] as u32, x[3] as u32);
                if address % 4 != 0
                    || !PROXIED_REGISTERS
                        .iter()
                        .any(|block| block.contains(&address))
                {
                    x[0] = STATUS_INVALID_ARGUMENT;
                    return SmcEffect::Resume;
                }
                let failed = |is_write| {
                    SmcEffect::Unimplemented(Unimplemented::Register { address, is_write })
                };
                // A full mask replaces the register, so nothing is read.
                let previous = if mask == u32::MAX {
                    0
                } else {
                    match bus.read(0, address, 4) {
                        Access::Done(previous) => previous as u32,
                        _ => return failed(false),
                    }
                };
                if mask != 0 {
                    let merged = (previous & !mask) | (value & mask);
                    if bus.write(0, address, 4, merged.into()) != Access::Done(()) {
                        return failed(true);
                    }
                }
                x[0] = STATUS_SUCCESS;
                x[1] = previous.into();
            }
            function::CONFIGURE_CARVEOUT => {
                let Some(carveout) = usize::try_from(x[1])
                    .ok()
                    .and_then(|index| state.carveouts.get_mut(index))
                else {
                    x[0] = STATUS_INVALID_ARGUMENT;
                    return SmcEffect::Resume;
                };
                *carveout = Carveout {
                    address: x[2],
                    size: x[3],
                };
                x[0] = STATUS_SUCCESS;
            }
            function::CPU_ON => {
                let (entry, context) = (x[2], x[3]);
                let Some((core, on)) = usize::try_from(x[1])
                    .ok()
                    .and_then(|core| Some((core, state.cores_on.get_mut(core)?)))
                else {
                    x[0] = PSCI_INVALID_PARAMETERS;
                    return SmcEffect::Resume;
                };
                if *on {
                    x[0] = PSCI_ALREADY_ON;
                    return SmcEffect::Resume;
                }
                if !switch_core(bus, core, true) {
                    return unimplemented;
                }
                *on = true;
                x[0] = PSCI_SUCCESS;
                return SmcEffect::StartCore {
                    core,
                    entry,
                    context,
                };
            }
            function::CPU_OFF => {
                if !switch_core(bus, core, false) {
                    return unimplemented;
                }
                state.cores_on[core] = false;
                return SmcEffect::PowerOff;
            }
            function::CPU_SUSPEND => {
                if x[1] as u32 & POWER_STATE_TYPE_AND_ID != POWER_STATE_SLEEP {
                    x[0] = PSCI_DENIED;
                    return SmcEffect::Resume;
                }
                let others = state.cores_on.iter().enumerate();
                if others.filter(|&(other, &on)| on && other != core).count() != 0 {
                    return SmcEffect::Unimplemented(Unimplemented::Abort(
                        "sleep was asked for while another core was on",
                    ));
                }
                return SmcEffect::Sleep {
                    entry: x[2],
                    context: x[3],
                };
            }
            function::PANIC => return SmcEffect::Panic { code: x[1] as u32 },
            _ => return unimplemented,
        }
        SmcEffect::Resume
    }

    /// Carveouts as last configured by the kernel.
    pub fn carveouts(&self) -> [Carveout; CARVEOUT_COUNT] {
        self.state.lock().unwrap().carveouts
    }

    /// SetConfig, from userland: item in `W1`, value in `X3`.
    fn set_config(&self, x: &mut [u64; 8]) -> SmcEffect {
        let item = x[1] as u32;
        match item {
            config_item::IS_CHARGER_HI_Z_MODE_ENABLED => {
                self.state.lock().unwrap().charger_hi_z_mode = x[3] != 0;
            }
            _ => return SmcEffect::Unimplemented(Unimplemented::ConfigItem(item)),
        }
        x[0] = STATUS_SUCCESS;
        SmcEffect::Resume
    }

    /// GetConfig, through either gate: item in `W1`, value out in `X1`-`X4`.
    fn get_config(&self, x: &mut [u64; 8]) -> SmcEffect {
        let item = x[1] as u32;
        let value = match item {
            config_item::DISABLE_PROGRAM_VERIFICATION => self.program_verification_disabled.into(),
            config_item::DRAM_ID => match self.dram.dram_id() {
                Some(id) => id,
                None => return SmcEffect::Unimplemented(Unimplemented::ConfigItem(item)),
            },
            config_item::SECURITY_ENGINE_INTERRUPT_NUMBER => SECURITY_ENGINE_INTERRUPT,
            config_item::HARDWARE_TYPE => HARDWARE_TYPE_ICOSA,
            config_item::HARDWARE_STATE => HARDWARE_STATE_PRODUCTION,
            config_item::IS_RECOVERY_BOOT => 0,
            config_item::DEVICE_ID => self.device_id,
            config_item::RETAIL_INTERACTIVE_DISPLAY_STATE => 0,
            config_item::REGULATOR_TYPE => 0,
            config_item::DEVICE_UNIQUE_KEY_GENERATION => 0,
            config_item::IS_DEVELOPMENT_FUNCTION_ENABLED => {
                self.development_function_enabled.into()
            }
            config_item::MEMORY_MODE => self.dram.memory_mode(),
            config_item::KERNEL_CONFIGURATION => self.kernel_configuration.bits(self.dram),
            config_item::IS_CHARGER_HI_Z_MODE_ENABLED => {
                self.state.lock().unwrap().charger_hi_z_mode.into()
            }
            config_item::FIRST_OF_ATMOSPHERE.. => {
                x[0] = STATUS_INVALID_ARGUMENT;
                return SmcEffect::Resume;
            }
            _ => return SmcEffect::Unimplemented(Unimplemented::ConfigItem(item)),
        };
        x[0] = STATUS_SUCCESS;
        x[1..5].copy_from_slice(&[value, 0, 0, 0]);
        SmcEffect::Resume
    }

    /// Runs a key ladder operation and turns its outcome into a status.
    fn key_operation(
        &self,
        x: &mut [u64; 8],
        operation: impl FnOnce(&mut Crypto, &mut [u64; 8]) -> Result<(), CryptoError>,
    ) -> SmcEffect {
        x[0] = match operation(&mut self.state.lock().unwrap().crypto, x) {
            Ok(()) => STATUS_SUCCESS,
            Err(CryptoError::InvalidArgument) => STATUS_INVALID_ARGUMENT,
            Err(CryptoError::Unimplemented(what)) => {
                return SmcEffect::Unimplemented(Unimplemented::Crypto(what));
            }
        };
        SmcEffect::Resume
    }

    /// ComputeAes. The engine finishes before the call returns; what is
    /// asynchronous for the caller is only how it learns of that.
    fn compute_aes(&self, x: &mut [u64; 8], bus: &Bus, memory: &DeviceMemory) -> SmcEffect {
        let unimplemented = |what| SmcEffect::Unimplemented(Unimplemented::Crypto(what));
        let (mode, source, destination, size) = (x[1] as u32, x[4], x[5], x[6]);
        let cipher = match mode >> 4 & 3 {
            0 => CipherMode::CbcEncrypt,
            1 => CipherMode::CbcDecrypt,
            2 => CipherMode::Ctr,
            _ => {
                x[0] = STATUS_INVALID_ARGUMENT;
                return SmcEffect::Resume;
            }
        };
        let mut state = self.state.lock().unwrap();
        if state.async_operation.is_some() {
            x[0] = STATUS_BUSY;
            return SmcEffect::Resume;
        }
        if size > MAX_ENGINE_BYTES {
            return unimplemented("an AES operation on more than 1 MiB");
        }
        let buffers =
            engine_buffer(memory, source, size).zip(engine_buffer(memory, destination, size));
        let mut data = vec![0; size as usize];
        let Some((_, destination)) =
            buffers.filter(|&(source, _)| memory.read(source, &mut data).is_some())
        else {
            return unimplemented(
                "an AES operation whose buffer lists are not one mapped buffer each",
            );
        };
        match state
            .crypto
            .compute_aes((mode & 7).into(), cipher, &key_at(x, 2), &mut data)
        {
            Ok(()) => {}
            Err(CryptoError::InvalidArgument) => {
                x[0] = STATUS_INVALID_ARGUMENT;
                return SmcEffect::Resume;
            }
            Err(CryptoError::Unimplemented(what)) => return unimplemented(what),
        }
        if memory.write(destination, &data).is_none() {
            return unimplemented("an AES operation whose destination is not mapped");
        }
        let pending = 1 << (SECURITY_ENGINE_INTERRUPT % 32);
        if bus.write(0, SET_PENDING_REGISTER, 4, pending) != Access::Done(()) {
            let register = Unimplemented::Register {
                address: SET_PENDING_REGISTER,
                is_write: true,
            };
            return SmcEffect::Unimplemented(register);
        }
        let key = state.key_rng.next();
        state.async_operation = Some(key);
        x[0] = STATUS_SUCCESS;
        x[1] = key;
        SmcEffect::Resume
    }

    /// DecryptDeviceUniqueData, synchronous like the monitor's. Mode 0 only;
    /// the mode word's other bits are reserved and refused.
    fn decrypt_device_unique_data(&self, x: &mut [u64; 8], memory: &DeviceMemory) -> SmcEffect {
        let unimplemented = |what| SmcEffect::Unimplemented(Unimplemented::Crypto(what));
        let (mode, address, len) = (x[3] as u32, x[4], x[5]);
        if mode >> 3 != 0 || len > crypto::DEVICE_UNIQUE_DATA_MAX as u64 {
            x[0] = STATUS_INVALID_ARGUMENT;
            return SmcEffect::Resume;
        }
        if mode != 0 {
            return unimplemented("device-unique data that the monitor imports as a key");
        }
        let mut data = vec![0; len as usize];
        if address % PAGE_SIZE + len > PAGE_SIZE || memory.ram().read(address, &mut data).is_none()
        {
            return unimplemented("device-unique data that is not within one page of DRAM");
        }
        let device_id = self.device_id;
        let (access_key, key_source) = (key_at(x, 1), key_at(x, 6));
        let effect = self.key_operation(x, |crypto, _| {
            crypto.decrypt_device_unique_data(&access_key, &key_source, &mut data, device_id)
        });
        if x[0] == STATUS_SUCCESS {
            // Within the page just read.
            memory
                .ram()
                .write(address, &data)
                .expect("the data was read from RAM");
        }
        effect
    }

    /// GenerateRandomBytes, through either gate: byte count in `X1`, bytes
    /// out packed into `X1`-`X7`, the rest of which keep their values.
    fn generate_random_bytes(&self, x: &mut [u64; 8]) -> SmcEffect {
        let count = x[1];
        if count > MAX_RANDOM_BYTES {
            x[0] = STATUS_INVALID_ARGUMENT;
            return SmcEffect::Resume;
        }
        let mut bytes = [0u8; MAX_RANDOM_BYTES as usize];
        for (word, chunk) in x[1..].iter().zip(bytes.as_chunks_mut::<8>().0) {
            *chunk = word.to_le_bytes();
        }
        self.state
            .lock()
            .unwrap()
            .rng
            .fill(&mut bytes[..count as usize]);
        for (word, chunk) in x[1..].iter_mut().zip(bytes.as_chunks::<8>().0) {
            *word = u64::from_le_bytes(*chunk);
        }
        x[0] = STATUS_SUCCESS;
        SmcEffect::Resume
    }
}

/// Name of an SMC function for traces.
pub fn function_name(immediate: u16, id: u32) -> &'static str {
    match (immediate, id) {
        (KERNEL_SMC, function::CPU_ON) => "CpuOn",
        (KERNEL_SMC, function::CPU_OFF) => "CpuOff",
        (KERNEL_SMC, function::CPU_SUSPEND) => "CpuSuspend",
        (KERNEL_SMC, function::GET_CONFIG) => "GetConfig",
        (KERNEL_SMC, function::GENERATE_RANDOM_BYTES) => "GenerateRandomBytes",
        (KERNEL_SMC, function::PANIC) => "Panic",
        (KERNEL_SMC, function::CONFIGURE_CARVEOUT) => "ConfigureCarveout",
        (KERNEL_SMC, function::READ_WRITE_REGISTER) => "ReadWriteRegister",
        (USER_SMC, user_function::GET_CONFIG) => "GetConfig (user)",
        (USER_SMC, user_function::SET_CONFIG) => "SetConfig",
        (USER_SMC, user_function::GENERATE_RANDOM_BYTES) => "GenerateRandomBytes (user)",
        (USER_SMC, user_function::GENERATE_AES_KEK) => "GenerateAesKek",
        (USER_SMC, user_function::LOAD_AES_KEY) => "LoadAesKey",
        (USER_SMC, user_function::COMPUTE_AES) => "ComputeAes",
        (USER_SMC, user_function::COMPUTE_CMAC) => "ComputeCmac",
        (USER_SMC, user_function::GENERATE_SPECIFIC_AES_KEY) => "GenerateSpecificAesKey",
        (USER_SMC, user_function::DECRYPT_DEVICE_UNIQUE_DATA) => "DecryptDeviceUniqueData",
        (USER_SMC, user_function::GET_RESULT) => "GetResult",
        _ => "?",
    }
}

/// Switches the power partition of `core` on or off, as the monitor does
/// around starting and stopping it. `false` if the PMC refused.
fn switch_core(bus: &Bus, core: usize, on: bool) -> bool {
    let partition = CORE_PARTITIONS[core];
    let Access::Done(status) = bus.read(0, PMC_PWRGATE_STATUS, 4) else {
        return false;
    };
    (status >> partition & 1 != 0) == on
        || bus.write(
            0,
            PMC_PWRGATE_TOGGLE,
            4,
            PWRGATE_TOGGLE_START | u64::from(partition),
        ) == Access::Done(())
}

/// The 128-bit value `x[index]`, `x[index + 1]` carry, in memory order.
fn key_at(x: &[u64; 8], index: usize) -> Key {
    let mut key = Key::default();
    key[..8].copy_from_slice(&x[index].to_le_bytes());
    key[8..].copy_from_slice(&x[index + 1].to_le_bytes());
    key
}

fn set_key_at(x: &mut [u64; 8], index: usize, key: &Key) {
    let (low, high) = key.split_at(8);
    x[index] = u64::from_le_bytes(low.try_into().unwrap());
    x[index + 1] = u64::from_le_bytes(high.try_into().unwrap());
}

/// The one buffer a security engine buffer list at `list` names, if it
/// holds `size` bytes: a count of further entries, which must be zero, then
/// address and length.
fn engine_buffer(memory: &DeviceMemory, list: u64, size: u64) -> Option<u64> {
    let mut raw = [0; 12];
    memory.read(list, &mut raw)?;
    let word = |index: usize| u32::from_le_bytes(raw[index * 4..][..4].try_into().unwrap());
    (word(0) == 0 && u64::from(word(2)) >= size).then(|| word(1).into())
}

/// SplitMix64. Hardware uses a true random source; a seeded generator keeps
/// the kernel's randomized layout reproducible from run to run.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn fill(&mut self, bytes: &mut [u8]) {
        for chunk in bytes.chunks_mut(8) {
            chunk.copy_from_slice(&self.next().to_le_bytes()[..chunk.len()]);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{
        memory::GuestRam,
        soc::{mc, regs::RegisterFile},
    };

    const MC_EMEM_CFG: u64 = mc::BASE + 0x50;
    const MEMORY_BASE: u64 = 0x8000_0000;

    /// A monitor with the register blocks it reaches and a little DRAM.
    struct Console {
        monitor: SecureMonitor,
        bus: Bus,
        memory: Arc<GuestRam>,
        engine: DeviceMemory,
    }

    impl Console {
        fn call(&self, immediate: u16, x: &mut [u64; 8]) -> SmcEffect {
            self.call_on(0, immediate, x)
        }

        fn call_on(&self, core: usize, immediate: u16, x: &mut [u64; 8]) -> SmcEffect {
            self.monitor
                .call(core, immediate, x, &self.bus, &self.engine)
        }

        fn carveouts(&self) -> [Carveout; CARVEOUT_COUNT] {
            self.monitor.carveouts()
        }
    }

    fn console_with_keys(params: &BootParams, keys: KeySet) -> Console {
        let memory_controller = Arc::new(mc::Mc::new(params.dram));
        let mut bus = Bus::default();
        bus.map(
            "PMC",
            0x7000_E400,
            0xC00,
            Box::new(crate::soc::pmc::Pmc::new()),
        );
        bus.map(
            "MC",
            mc::BASE,
            mc::SIZE,
            Box::new(Arc::clone(&memory_controller)),
        );
        bus.map(
            "GICD",
            0x5004_1000,
            0x1000,
            Box::new(RegisterFile::zeroed([])),
        );
        let memory = Arc::new(GuestRam::new(MEMORY_BASE, 0x1_0000).unwrap());
        let engine = DeviceMemory::new(&memory_controller, &memory, mc::SMMU_SE_ASID);
        Console {
            monitor: SecureMonitor::new(params, keys),
            bus,
            memory,
            engine,
        }
    }

    fn console(params: &BootParams) -> Console {
        console_with_keys(params, KeySet::default())
    }

    fn monitor(dram: DramSize) -> Console {
        console(&BootParams {
            dram,
            rng_seed: 7,
            ..BootParams::default()
        })
    }

    #[test]
    fn compute_aes_works_on_the_engine_buffers_and_is_collected_once_by_its_key() {
        let keys = KeySet::parse(
            "aes_kek_generation_source = 000102030405060708090a0b0c0d0e0f\n\
             master_key_00 = 101112131415161718191a1b1c1d1e1f\n",
        )
        .unwrap();
        let console = console_with_keys(&BootParams::default(), keys);
        let user = |arguments: [u64; 7]| {
            let mut x = [0; 8];
            x[..7].copy_from_slice(&arguments);
            assert_eq!(console.call(USER_SMC, &mut x), SmcEffect::Resume);
            x
        };
        let kek = user([
            user_function::GENERATE_AES_KEK.into(),
            0x33,
            0x44,
            1,
            0,
            0,
            0,
        ]);
        assert_eq!(kek[0], STATUS_SUCCESS);
        assert_eq!(
            user([
                user_function::LOAD_AES_KEY.into(),
                3,
                kek[1],
                kek[2],
                0x55,
                0x66,
                0
            ])[0],
            STATUS_SUCCESS
        );

        // The kernel's setup for `spl`: address space 5 for the engine,
        // with device page 0x8003_5000 on a page of DRAM.
        const DEVICE: u64 = 0x8003_5000;
        const PHYSICAL: u64 = MEMORY_BASE + 0x5000;
        let (directory, table) = (MEMORY_BASE + 0x1000, MEMORY_BASE + 0x2000);
        let word = |address: u64, value: u64| {
            console
                .memory
                .write(address, &(value as u32).to_le_bytes())
                .unwrap()
        };
        word(directory + (DEVICE >> 22) * 4, 0xF000_0000 | table >> 12);
        word(
            table + (DEVICE >> 12 & 0x3FF) * 4,
            0xE000_0000 | PHYSICAL >> 12,
        );
        for (register, value) in [
            (0x1C, 5),
            (0x20, 0xE000_0000 | directory >> 12),
            (mc::SMMU_SE_ASID, 0x8000_0005),
        ] {
            let write = [
                function::READ_WRITE_REGISTER.into(),
                mc::BASE + register,
                0xFFFF_FFFF,
                value,
            ];
            assert_eq!(smc(&console, write).1[0], STATUS_SUCCESS);
        }

        // Source list at +0, destination list at +0xC, as `spl` lays them
        // out; two blocks of data at +0x100, results to +0x200.
        let list = |at: u64, buffer: u64| {
            word(PHYSICAL + at, 0);
            word(PHYSICAL + at + 4, DEVICE + buffer);
            word(PHYSICAL + at + 8, 0x20);
        };
        list(0, 0x100);
        list(0xC, 0x200);
        let plaintext: Vec<u8> = (0..0x20).collect();
        console.memory.write(PHYSICAL + 0x100, &plaintext).unwrap();
        let compute = |mode: u64, from: u64, to: u64| {
            user([
                user_function::COMPUTE_AES.into(),
                mode,
                0xAAAA,
                0xBBBB,
                DEVICE + from,
                DEVICE + to,
                0x20,
            ])
        };

        // CBC encryption with slot 3.
        let encrypt = compute(0x03, 0, 0xC);
        assert_eq!(encrypt[0], STATUS_SUCCESS);
        assert_eq!(
            console.bus.read(0, SET_PENDING_REGISTER, 4),
            Access::Done(1 << 12)
        );
        let mut ciphertext = [0; 0x20];
        console
            .memory
            .read(PHYSICAL + 0x200, &mut ciphertext)
            .unwrap();
        assert_ne!(ciphertext[..], plaintext[..]);
        assert_ne!(ciphertext[..0x10], ciphertext[0x10..]);

        // The engine is taken until the result is collected, by its key.
        assert_eq!(compute(0x13, 0xC, 0)[0], STATUS_BUSY);
        let get_result = |key: u64| user([user_function::GET_RESULT.into(), key, 0, 0, 0, 0, 0]);
        assert_eq!(
            get_result(encrypt[1] ^ 1)[0],
            STATUS_INVALID_ASYNC_OPERATION
        );
        assert_eq!(
            get_result(encrypt[1])[..2],
            [STATUS_SUCCESS, STATUS_SUCCESS]
        );
        assert_eq!(get_result(encrypt[1])[0], STATUS_NO_ASYNC_OPERATION);

        // CBC decryption the other way round restores the data.
        console.memory.write(PHYSICAL + 0x100, &[0; 0x20]).unwrap();
        assert_eq!(compute(0x13, 0xC, 0)[0], STATUS_SUCCESS);
        let mut restored = [0; 0x20];
        console
            .memory
            .read(PHYSICAL + 0x100, &mut restored)
            .unwrap();
        assert_eq!(restored[..], plaintext[..]);
    }

    #[test]
    fn device_unique_data_is_decrypted_in_place_for_this_console_only() {
        let text = "aes_kek_generation_source = 000102030405060708090a0b0c0d0e0f\n\
                    rsa_private_kek_generation_source = 202122232425262728292a2b2c2d2e2f\n\
                    master_key_00 = 101112131415161718191a1b1c1d1e1f\n";
        let keys = KeySet::parse(text).unwrap();
        let device_id = 0x0011_2233_4455_6677;
        let params = BootParams {
            device_id,
            ..BootParams::default()
        };
        let console = console_with_keys(&params, keys.clone());
        let user = |arguments: [u64; 8]| {
            let mut x = arguments;
            let effect = console.call(USER_SMC, &mut x);
            (effect, x)
        };
        let (kek_source, key_source) = ([0x33; 16], [0x44; 16]);
        let option = crypto::kek_option(crypto::SEAL_DECRYPT_DEVICE_UNIQUE_DATA);
        let (_, access) = user([
            user_function::GENERATE_AES_KEK.into(),
            0x3333_3333_3333_3333,
            0x3333_3333_3333_3333,
            0,
            option.into(),
            0,
            0,
            0,
        ]);
        assert_eq!(access[0], STATUS_SUCCESS);

        let key =
            crypto::device_unique_data_key(&keys, &kek_source, &key_source, 0, option).unwrap();
        let payload: Vec<u8> = (0..0x100).map(|index| index as u8).collect();
        let sealed = crypto::seal_device_unique_data(&key, &[0x77; 16], &payload, device_id);
        const PHYSICAL: u64 = MEMORY_BASE + 0x3010;
        let decrypt = |data: &[u8], mode: u64| {
            console.memory.write(PHYSICAL, data).unwrap();
            user([
                user_function::DECRYPT_DEVICE_UNIQUE_DATA.into(),
                access[1],
                access[2],
                mode,
                PHYSICAL,
                data.len() as u64,
                0x4444_4444_4444_4444,
                0x4444_4444_4444_4444,
            ])
        };
        let (effect, x) = decrypt(&sealed, 0);
        assert_eq!((effect, x[0]), (SmcEffect::Resume, STATUS_SUCCESS));
        let mut plain = vec![0; 0x100];
        console.memory.read(PHYSICAL, &mut plain).unwrap();
        assert_eq!(plain, payload);

        let mut tampered = sealed.clone();
        tampered[0x20] ^= 0x80;
        assert_eq!(decrypt(&tampered, 0).1[0], STATUS_INVALID_ARGUMENT);
        assert_eq!(decrypt(&sealed, 1 << 3).1[0], STATUS_INVALID_ARGUMENT);
        assert!(matches!(
            decrypt(&sealed, 3).0,
            SmcEffect::Unimplemented(Unimplemented::Crypto(_))
        ));
    }

    /// Issues a kernel SMC and returns the effect with the result registers.
    fn smc(monitor: &Console, arguments: [u64; 4]) -> (SmcEffect, [u64; 8]) {
        let mut x = [0; 8];
        x[..4].copy_from_slice(&arguments);
        (monitor.call(KERNEL_SMC, &mut x), x)
    }

    #[test]
    fn dram_size_is_consistent_across_every_source_the_kernel_reads() {
        for (dram, megabytes, size_field, memory_mode) in [
            (DramSize::Gb4, 4096, 0, 0x01),
            (DramSize::Gb6, 6144, 1, 0x11),
            (DramSize::Gb8, 8192, 2, 0x21),
        ] {
            let monitor = monitor(dram);
            let (_, emem) = smc(
                &monitor,
                [function::READ_WRITE_REGISTER.into(), MC_EMEM_CFG, 0, 0],
            );
            let (_, config) = smc(
                &monitor,
                [
                    function::GET_CONFIG.into(),
                    config_item::KERNEL_CONFIGURATION.into(),
                    0,
                    0,
                ],
            );
            let (_, mode) = smc(
                &monitor,
                [
                    function::GET_CONFIG.into(),
                    config_item::MEMORY_MODE.into(),
                    0,
                    0,
                ],
            );
            assert_eq!((emem[0], emem[1]), (STATUS_SUCCESS, megabytes));
            assert_eq!(
                (config[0], config[1] >> 16 & 3),
                (STATUS_SUCCESS, size_field)
            );
            assert_eq!((mode[0], mode[1]), (STATUS_SUCCESS, memory_mode));
        }
    }

    #[test]
    fn kernel_configuration_flags_land_on_the_bits_the_kernel_tests() {
        let params = BootParams {
            kernel_configuration: KernelConfiguration {
                increase_thread_resource_limit: true,
                skip_panic_smc: true,
                ..KernelConfiguration::default()
            },
            ..BootParams::default()
        };
        let (_, x) = smc(
            &console(&params),
            [
                function::GET_CONFIG.into(),
                config_item::KERNEL_CONFIGURATION.into(),
                0,
                0,
            ],
        );
        assert_eq!(x[1], 1 << 3 | 1 << 8);
    }

    #[test]
    fn development_functions_are_reported_only_when_enabled() {
        let item = [
            function::GET_CONFIG.into(),
            config_item::IS_DEVELOPMENT_FUNCTION_ENABLED.into(),
            0,
            0,
        ];
        let enabled = BootParams {
            development_function_enabled: true,
            ..BootParams::default()
        };
        assert_eq!(smc(&console(&enabled), item).1[..2], [STATUS_SUCCESS, 1]);
        assert_eq!(
            smc(&monitor(DramSize::Gb4), item).1[..2],
            [STATUS_SUCCESS, 0]
        );
    }

    #[test]
    fn program_verification_is_disabled_only_when_chosen() {
        let item = [
            function::GET_CONFIG.into(),
            config_item::DISABLE_PROGRAM_VERIFICATION.into(),
            0,
            0,
        ];
        let disabled = BootParams {
            program_verification_disabled: true,
            ..BootParams::default()
        };
        assert_eq!(smc(&console(&disabled), item).1[..2], [STATUS_SUCCESS, 1]);
        assert_eq!(
            smc(&monitor(DramSize::Gb4), item).1[..2],
            [STATUS_SUCCESS, 0]
        );
    }

    #[test]
    fn unmodeled_calls_stop_instead_of_answering() {
        let monitor = monitor(DramSize::Gb4);
        let before = [function::GET_CONFIG.into(), 4, 0, 0];
        let (effect, x) = smc(&monitor, before);
        assert_eq!(
            effect,
            SmcEffect::Unimplemented(Unimplemented::ConfigItem(4))
        );
        assert_eq!(x[..4], before);

        // An item only Atmosphère's monitor has is one this monitor does
        // not know: an answer, not a stop.
        let (effect, x) = smc(&monitor, [function::GET_CONFIG.into(), 65000, 0, 0]);
        assert_eq!((effect, x[0]), (SmcEffect::Resume, STATUS_INVALID_ARGUMENT));

        let (effect, _) = smc(&monitor, [0x8400_0009, 0, 0, 0]);
        assert_eq!(
            effect,
            SmcEffect::Unimplemented(Unimplemented::Function {
                immediate: 1,
                id: 0x8400_0009
            })
        );

        // The same identifier through the userland gate is a different call.
        let mut x = [
            function::GET_CONFIG.into(),
            config_item::KERNEL_CONFIGURATION.into(),
            0,
            0,
            0,
            0,
            0,
            0,
        ];
        assert_eq!(
            monitor.call(0, &mut x),
            SmcEffect::Unimplemented(Unimplemented::Function {
                immediate: 0,
                id: function::GET_CONFIG
            })
        );
    }

    #[test]
    fn user_gate_serves_get_config_under_its_own_numbering() {
        let monitor = monitor(DramSize::Gb6);
        let mut x = [
            user_function::GET_CONFIG.into(),
            config_item::MEMORY_MODE.into(),
            0,
            0,
            0,
            0,
            0,
            0,
        ];
        assert_eq!(monitor.call(USER_SMC, &mut x), SmcEffect::Resume);
        assert_eq!(x[..2], [STATUS_SUCCESS, 0x11]);

        // The user number means nothing through the kernel gate.
        let (effect, _) = smc(
            &monitor,
            [
                user_function::GET_CONFIG.into(),
                config_item::MEMORY_MODE.into(),
                0,
                0,
            ],
        );
        assert_eq!(
            effect,
            SmcEffect::Unimplemented(Unimplemented::Function {
                immediate: 1,
                id: user_function::GET_CONFIG
            })
        );
    }

    #[test]
    fn charger_hi_z_mode_is_the_one_settable_item_and_reads_back() {
        let monitor = monitor(DramSize::Gb4);
        let item = config_item::IS_CHARGER_HI_Z_MODE_ENABLED.into();
        let get = || {
            let mut x = [user_function::GET_CONFIG.into(), item, 0, 0, 0, 0, 0, 0];
            assert_eq!(monitor.call(USER_SMC, &mut x), SmcEffect::Resume);
            x[1]
        };
        assert_eq!(get(), 0);
        let mut x = [user_function::SET_CONFIG.into(), item, 0, 1, 0, 0, 0, 0];
        assert_eq!(monitor.call(USER_SMC, &mut x), SmcEffect::Resume);
        assert_eq!(x[0], STATUS_SUCCESS);
        assert_eq!(get(), 1);
        let mut x = [
            user_function::SET_CONFIG.into(),
            config_item::MEMORY_MODE.into(),
            0,
            1,
            0,
            0,
            0,
            0,
        ];
        assert_eq!(
            monitor.call(USER_SMC, &mut x),
            SmcEffect::Unimplemented(Unimplemented::ConfigItem(config_item::MEMORY_MODE))
        );
    }

    #[test]
    fn random_bytes_fill_exactly_the_requested_prefix() {
        let monitor = monitor(DramSize::Gb4);
        let mut x = [function::GENERATE_RANDOM_BYTES.into(), 11, 0, 0, 0, 0, 0, 0];
        assert_eq!(monitor.call(KERNEL_SMC, &mut x), SmcEffect::Resume);
        assert_eq!(x[0], STATUS_SUCCESS);
        assert_ne!(x[1], 11, "first word still holds the request");
        assert_ne!(x[2] & 0xFF_FFFF, 0);
        // Bytes 11.. keep what the caller passed (zero).
        assert_eq!((x[2] >> 24, &x[3..]), (0, &[0; 5][..]));

        let (_, x) = smc(&monitor, [function::GENERATE_RANDOM_BYTES.into(), 57, 0, 0]);
        assert_eq!(x[0], STATUS_INVALID_ARGUMENT);
    }

    #[test]
    fn random_stream_is_a_function_of_the_seed() {
        let draw = |seed| {
            let monitor = console(&BootParams {
                rng_seed: seed,
                ..BootParams::default()
            });
            let first = smc(&monitor, [function::GENERATE_RANDOM_BYTES.into(), 56, 0, 0]).1;
            let second = smc(&monitor, [function::GENERATE_RANDOM_BYTES.into(), 56, 0, 0]).1;
            (first, second)
        };
        let (first, second) = draw(1);
        assert_eq!(draw(1), (first, second));
        assert_ne!(first, second);
        assert_ne!(draw(2).0, first);
    }

    #[test]
    fn register_writes_merge_under_the_mask_and_return_the_previous_value() {
        let monitor = monitor(DramSize::Gb4);
        let register = 0x7000_E450;
        let access = |mask, value| {
            smc(
                &monitor,
                [function::READ_WRITE_REGISTER.into(), register, mask, value],
            )
            .1[..2]
                .to_vec()
        };
        assert_eq!(access(0xFFFF_FFFF, 0x1234_5678), [STATUS_SUCCESS, 0]);
        assert_eq!(
            access(0x0000_FF00, 0xFFFF_FFFF),
            [STATUS_SUCCESS, 0x1234_5678]
        );
        assert_eq!(access(0, 0xDEAD_BEEF), [STATUS_SUCCESS, 0x1234_FF78]);
        assert_eq!(access(0, 0), [STATUS_SUCCESS, 0x1234_FF78]);
    }

    #[test]
    fn register_access_is_confined_to_aligned_proxied_registers() {
        let monitor = monitor(DramSize::Gb4);
        // Unaligned, just below PMC, just past the memory controller, and DRAM.
        for address in [MC_EMEM_CFG + 2, 0x7000_E3FC, 0x7001_A000, 0x8000_0000] {
            let (effect, x) = smc(
                &monitor,
                [function::READ_WRITE_REGISTER.into(), address, 0, 0],
            );
            assert_eq!(
                (effect, x[0]),
                (SmcEffect::Resume, STATUS_INVALID_ARGUMENT),
                "{address:#x}"
            );
        }
    }

    #[test]
    fn cpu_on_starts_each_secondary_core_once() {
        let monitor = monitor(DramSize::Gb4);
        let on = |core| {
            smc(
                &monitor,
                [function::CPU_ON.into(), core, 0x800D_60E8, 0x800E_8380],
            )
        };

        let (effect, x) = on(2);
        assert_eq!(
            effect,
            SmcEffect::StartCore {
                core: 2,
                entry: 0x800D_60E8,
                context: 0x800E_8380
            }
        );
        assert_eq!(x[0], PSCI_SUCCESS);
        assert_eq!((on(2).0, on(2).1[0]), (SmcEffect::Resume, PSCI_ALREADY_ON));
        // The boot core is running from the start.
        assert_eq!((on(0).0, on(0).1[0]), (SmcEffect::Resume, PSCI_ALREADY_ON));
        assert_eq!(
            (on(4).0, on(4).1[0]),
            (SmcEffect::Resume, PSCI_INVALID_PARAMETERS)
        );
    }

    #[test]
    fn sleep_needs_the_other_cores_off_and_their_partitions_follow() {
        let monitor = monitor(DramSize::Gb4);
        let partitions = || {
            let status = smc(
                &monitor,
                [
                    function::READ_WRITE_REGISTER.into(),
                    PMC_PWRGATE_STATUS,
                    0,
                    0,
                ],
            );
            status.1[1] >> 9 & 7
        };
        assert_eq!(partitions(), 0);
        for core in 1..4 {
            smc(&monitor, [function::CPU_ON.into(), core, 0x8000_0000, 0]);
        }
        assert_eq!(partitions(), 7);

        let sleep = |state: u64| {
            smc(
                &monitor,
                [function::CPU_SUSPEND.into(), state, 0x8000_1000, 7],
            )
        };
        // Any state but the one there is is denied.
        assert_eq!(
            (sleep(27).0, sleep(27).1[0]),
            (SmcEffect::Resume, PSCI_DENIED)
        );
        assert!(matches!(
            sleep(0x0201_001B).0,
            SmcEffect::Unimplemented(Unimplemented::Abort(_))
        ));
        for core in 1..4 {
            let mut x = [function::CPU_OFF.into(), 0, 0, 0, 0, 0, 0, 0];
            assert_eq!(
                monitor.call_on(core, KERNEL_SMC, &mut x),
                SmcEffect::PowerOff
            );
        }
        assert_eq!(partitions(), 0);
        assert_eq!(
            sleep(0x0201_001B).0,
            SmcEffect::Sleep {
                entry: 0x8000_1000,
                context: 7
            }
        );
        // A core that was switched off can be turned on again.
        let again = smc(&monitor, [function::CPU_ON.into(), 2, 0x8000_0000, 0]);
        assert!(matches!(again.0, SmcEffect::StartCore { core: 2, .. }));
        assert_eq!(partitions(), 2);
    }

    #[test]
    fn carveouts_are_recorded_per_index() {
        let monitor = monitor(DramSize::Gb4);
        let (_, x) = smc(
            &monitor,
            [
                function::CONFIGURE_CARVEOUT.into(),
                1,
                0x8006_0000,
                0x1EB_8000,
            ],
        );
        assert_eq!(x[0], STATUS_SUCCESS);
        assert_eq!(
            monitor.carveouts(),
            [
                Carveout::default(),
                Carveout {
                    address: 0x8006_0000,
                    size: 0x1EB_8000
                }
            ]
        );
        let (_, x) = smc(&monitor, [function::CONFIGURE_CARVEOUT.into(), 2, 0, 0]);
        assert_eq!(x[0], STATUS_INVALID_ARGUMENT);
    }

    #[test]
    fn panic_reports_the_kernel_code_and_does_not_resume() {
        let (effect, _) = smc(
            &monitor(DramSize::Gb4),
            [function::PANIC.into(), 0xF00, 0, 0],
        );
        assert_eq!(effect, SmcEffect::Panic { code: 0xF00 });
    }
}
