//! The compute class (0xB1C0): inline-to-memory, semaphore releases and
//! kernel launches.
//!
//! A launch is the queue meta data (QMD) the driver writes to memory,
//! handed over with `SEND_PCAS_A` (QMD address >> 8) followed by
//! `SEND_SIGNALING_PCAS_B` with `SCHEDULE` set [public: NVIDIA
//! open-gpu-doc `classes/compute/clb1c0.h`]. The QMD is version 01_07
//! [public: NVIDIA
//! open-gpu-doc `classes/compute/clb1c0qmd.h`, the only version that
//! header defines for the class; field names below are its
//! `NVB1C0_QMDV01_07_*`]. [`Qmd::parse`] decodes the fields a launch
//! uses and refuses every one that would change what the launch does
//! when it is not in the state this model implements (queues, dependent
//! launches, reductions, resumed grids, other floating-point modes).
//!
//! The kernel runs synchronously with the shader interpreter
//! ([`shader::Kernel`]): when `SEND_SIGNALING_PCAS_B` returns, every
//! thread has run and the launch's semaphore releases are written.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::lru::Lru;
use super::pipeline::read_mapped;
use super::resources::{DrawTextures, Pools, TextureCache};
use super::shader::{self, Kernel, Launch};
use super::{GpuMemory, Hooks, Unmodeled, inline, three_d};

/// Method registers (word index = byte offset / 4).
const SEND_PCAS_A: u32 = 0x2b4 / 4;
const SEND_PCAS_B: u32 = 0x2b8 / 4;
const SEND_SIGNALING_PCAS_B: u32 = 0x2bc / 4;
const SET_TEX_SAMPLER_POOL_A: u32 = 0x155c / 4;
const SET_TEX_HEADER_POOL_A: u32 = 0x1574 / 4;
const SET_PROGRAM_REGION_A: u32 = 0x1608 / 4;
const SET_BINDLESS_TEXTURE: u32 = 0x2608 / 4;
const SET_REPORT_SEMAPHORE_A: u32 = 0x1b00 / 4;

/// Size of the register file: every method of the class header.
const METHODS: usize = 0xE00;

/// Size of a QMD.
pub const QMD_SIZE: usize = 0x100;

/// Host memory the decoded kernels of one channel may take \[assumption:
/// as `resources::PROGRAM_BUDGET` for graphics programs, whose sizing
/// argument holds for kernels, at most 64 KiB of code each\].
const KERNEL_BUDGET: usize = 16 << 20;

/// The compute engine of a channel.
#[derive(Default, Serialize, Deserialize)]
pub struct Compute {
    registers: Vec<u32>,
    inline: inline::InlineToMemory,
    #[serde(skip)]
    caches: Caches,
}

/// Decoded kernels and textures, rebuilt from memory when dropped.
struct Caches {
    /// Kernels by code address, with the bytes they were decoded from.
    kernels: Lru<u64, (Vec<u8>, Arc<Kernel>)>,
    textures: TextureCache,
}

impl Default for Caches {
    fn default() -> Self {
        Caches {
            kernels: Lru::new(KERNEL_BUDGET),
            textures: TextureCache::default(),
        }
    }
}

impl Compute {
    /// Whether a restored register file has the size methods index.
    pub(super) fn fits(&self) -> bool {
        self.registers.is_empty() || self.registers.len() == METHODS
    }

    fn get(&self, method: u32) -> u32 {
        self.registers[method as usize]
    }

    /// The 40-bit address of an `_A` (upper 8 bits) / `_B` (lower 32)
    /// method pair.
    fn address(&self, upper: u32) -> u64 {
        u64::from(self.get(upper) & 0xFF) << 32 | u64::from(self.get(upper + 1))
    }

    pub(super) fn method(
        &mut self,
        method: u32,
        argument: u32,
        memory: &dyn GpuMemory,
        hooks: &Hooks,
    ) -> Result<(), Unmodeled> {
        if self.registers.is_empty() {
            self.registers = vec![0; METHODS];
        }
        let index = method as usize;
        if index >= self.registers.len() {
            return Err(Unmodeled(format!(
                "compute (0xB1C0): method {:#x} = {argument:#x}",
                method * 4
            )));
        }
        self.registers[index] = argument;
        match method {
            inline::FIRST_METHOD..=inline::LAST_METHOD => {
                match self.inline.method(method, argument, memory)? {
                    inline::Completion::None => Ok(()),
                    inline::Completion::Semaphore {
                        address,
                        payload,
                        four_words,
                    } => three_d::release_semaphore(memory, hooks, address, payload, four_words),
                }
            }
            // NO_OPERATION, WAIT_FOR_IDLE, cache invalidations (0x1330,
            // 0x1334, 0x1338, 0x120c, 0x1210, 0x1288, 0x1424, 0x1698): no
            // effect with synchronous execution and no caches.
            0x40 | 0x44 | 0x4CC | 0x4CD | 0x4CE | 0x483 | 0x484 | 0x4A2 | 0x509 | 0x5A6 => Ok(()),
            // SET_REPORT_SEMAPHORE_D: release of the payload only.
            0x6C3 => {
                let address = self.address(SET_REPORT_SEMAPHORE_A);
                let payload = self.registers[0x6C2];
                if argument & 3 != 0 || argument & !(3 | (1 << 28)) != 0 {
                    return Err(Unmodeled(format!(
                        "compute (0xB1C0): SET_REPORT_SEMAPHORE_D {argument:#x}"
                    )));
                }
                three_d::release_semaphore(
                    memory,
                    hooks,
                    address,
                    payload,
                    argument & (1 << 28) == 0,
                )
            }
            // SEND_PCAS_A: the QMD address, used by the next PCAS_B.
            SEND_PCAS_A => Ok(()),
            // SEND_SIGNALING_PCAS_B: INVALIDATE (bit 0) drops the QMD
            // cache, which this model does not have (each launch reads
            // the QMD from memory); SCHEDULE (bit 1) launches.
            SEND_SIGNALING_PCAS_B if argument & !3 == 0 && argument & 2 != 0 => {
                let address = u64::from(self.get(SEND_PCAS_A)) << 8;
                self.launch(address, memory, hooks)
                    .map_err(|Unmodeled(what)| {
                        Unmodeled(format!(
                            "compute (0xB1C0): launch of the QMD at {address:#x}: {what}"
                        ))
                    })
            }
            // Actions: notify, MME loads, SEND_PCAS_B (FROM/DELTA: a QMD
            // fetched without signaling, and its partial forms),
            // SEND_SIGNALING_PCAS_B without SCHEDULE.
            0x43 | 0x45..=0x48 | SEND_PCAS_B | SEND_SIGNALING_PCAS_B => Err(Unmodeled(format!(
                "compute (0xB1C0): action method {:#x} = {argument:#x} not modeled",
                method * 4
            ))),
            _ => Ok(()),
        }
    }

    /// Runs the launch the QMD at `address` describes.
    fn launch(
        &mut self,
        address: u64,
        memory: &dyn GpuMemory,
        hooks: &Hooks,
    ) -> Result<(), Unmodeled> {
        let mut bytes = [0u8; QMD_SIZE];
        memory
            .read(address, &mut bytes)
            .ok_or_else(|| Unmodeled("the QMD is not mapped".into()))?;
        let qmd = Qmd::parse(&bytes)?;
        let code = self.address(SET_PROGRAM_REGION_A) + u64::from(qmd.program_offset);
        let kernel = self.kernel(memory, code)?;
        let texture_slot = self.get(SET_BINDLESS_TEXTURE) & 7;
        let mut buffers: [(Vec<u8>, usize); 18] = Default::default();
        for (slot, binding) in qmd.constant_buffers.iter().enumerate() {
            let Some((buffer, size)) = *binding else {
                continue;
            };
            let size = size as usize;
            let mut extent = kernel.constant_buffer_extent(slot);
            if slot as u32 == texture_slot {
                extent = extent.map(|e| e.max(kernel.texture_handle_extent()));
            }
            let length = extent.map_or(size, |e| e.min(size));
            buffers[slot] = (read_mapped(memory, buffer, length), size);
        }
        let pools = Pools {
            header_pool: self.address(SET_TEX_HEADER_POOL_A),
            header_maximum: self.get(SET_TEX_HEADER_POOL_A + 2) & 0x3F_FFFF,
            sampler_pool: self.address(SET_TEX_SAMPLER_POOL_A),
            sampler_maximum: self.get(SET_TEX_SAMPLER_POOL_A + 2) & 0xF_FFFF,
            sampler_from_header: qmd.sampler_from_header,
        };
        let textures = DrawTextures::new(memory, pools, &self.caches.textures, &hooks.writes);
        let unit = textures.unit();
        let env = shader::Env {
            const_buffers: std::array::from_fn(|slot| {
                let (data, size) = &buffers[slot];
                shader::ConstantBuffer { data, size: *size }
            }),
            textures: &unit,
            texture_buffer_index: texture_slot,
            memory,
            // Not read: S2R SR_Y_DIRECTION is refused in compute programs.
            y_direction: 0.0,
        };
        kernel.run(&env, &qmd.launch)?;
        for release in qmd.releases.iter().flatten() {
            three_d::release_semaphore(
                memory,
                hooks,
                release.address,
                release.payload,
                release.four_words,
            )?;
        }
        Ok(())
    }

    /// The kernel whose code is at `address`, as memory holds it now.
    fn kernel(&mut self, memory: &dyn GpuMemory, address: u64) -> Result<Arc<Kernel>, Unmodeled> {
        if let Some((raw, kernel)) = self.caches.kernels.get(&address) {
            let mut now = vec![0; raw.len()];
            if memory.probe(address, &mut now).is_some() && now == *raw {
                return Ok(kernel.clone());
            }
        }
        let kernel = Kernel::decode(memory, address)?;
        let mut raw = vec![0; kernel.len()];
        memory
            .read(address, &mut raw)
            .ok_or_else(|| Unmodeled(format!("kernel code at {address:#x} is not mapped")))?;
        let kernel = Arc::new(kernel);
        let cost = raw.len() + kernel.bytes();
        self.caches
            .kernels
            .insert(address, (raw, kernel.clone()), cost);
        Ok(kernel)
    }
}

/// A semaphore release at the end of a launch (`RELEASE0_*`/`RELEASE1_*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Release {
    pub address: u64,
    pub payload: u32,
    /// `STRUCTURE_SIZE` FOUR_WORDS: payload, 0 and a 64-bit timestamp.
    pub four_words: bool,
}

/// The fields of a QMD (version 01_07) a launch uses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qmd {
    /// `PROGRAM_OFFSET`: the kernel's code, from `SET_PROGRAM_REGION`.
    pub program_offset: u32,
    /// `CTA_RASTER_WIDTH/HEIGHT/DEPTH`, `CTA_THREAD_DIMENSION0-2`,
    /// `SHARED_MEMORY_SIZE`, `BARRIER_COUNT`.
    pub launch: Launch,
    /// `CONSTANT_BUFFER_VALID(i)`, `_ADDR_LOWER/UPPER(i)`, `_SIZE(i)`:
    /// address and size of slots 0-7.
    pub constant_buffers: [Option<(u64, u32)>; 8],
    /// `SAMPLER_INDEX` VIA_HEADER_INDEX: a texture's sampler is the one
    /// of the texture's index.
    pub sampler_from_header: bool,
    /// `SEMAPHORE_RELEASE_ENABLE0/1`.
    pub releases: [Option<Release>; 2],
}

/// Bits `low..=high` of the QMD (its `MW(high:low)`).
fn field(words: &[u32; QMD_SIZE / 4], high: u32, low: u32) -> u32 {
    let word = (low / 32) as usize;
    let pair = u64::from(words[word]) | u64::from(*words.get(word + 1).unwrap_or(&0)) << 32;
    let width = high - low + 1;
    ((pair >> (low % 32)) & ((1u64 << width) - 1)) as u32
}

impl Qmd {
    /// Decodes a QMD, refusing fields whose state this model does not
    /// implement (each named with its value).
    pub fn parse(bytes: &[u8; QMD_SIZE]) -> Result<Qmd, Unmodeled> {
        let words: [u32; QMD_SIZE / 4] = std::array::from_fn(|i| {
            u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().expect("4 bytes"))
        });
        let f = |high, low| field(&words, high, low);
        let (major, minor) = (f(583, 580), f(579, 576));
        if (major, minor) != (1, 7) {
            return Err(Unmodeled(format!(
                "QMD version {major:02}_{minor:02} (only 01_07)"
            )));
        }
        // Fields that must be clear (or, for the floating-point modes,
        // at their first value) for the launch this model runs: (name,
        // high, low).
        const REFUSED: &[(&str, u32, u32)] = &[
            ("IS_QUEUE", 200, 200),
            ("ADD_TO_HEAD_OF_QMD_GROUP_LINKED_LIST", 201, 201),
            ("DEPENDENT_QMD_SCHEDULE_ENABLE", 205, 205),
            ("CWD_REFERENCE_COUNT_INCR_ENABLE", 367, 367),
            ("CWD_REFERENCE_COUNT_DECR_ENABLE", 371, 371),
            // FP64_COMPATIBLE NaN behaviour, PASS_INDEFINITE F2I of NaN
            // and FLUSH_DENORMS: modes the interpreter's float operations
            // have not been checked against.
            ("FP32_NAN_BEHAVIOR", 376, 376),
            ("FP32_F2I_NAN_BEHAVIOR", 377, 377),
            ("FP32_NARROW_INSTRUCTION", 383, 383),
            ("CTA_RASTER_WIDTH_RESUME", 479, 448),
            ("CTA_RASTER_HEIGHT_RESUME", 495, 480),
            ("CTA_RASTER_DEPTH_RESUME", 511, 496),
            ("RELEASE0_REDUCTION_ENABLE", 794, 794),
            ("RELEASE1_REDUCTION_ENABLE", 890, 890),
        ];
        for &(name, high, low) in REFUSED {
            let value = f(high, low);
            if value != 0 {
                return Err(Unmodeled(format!("QMD {name} = {value:#x}")));
            }
        }
        let launch = Launch {
            grid: [f(415, 384), f(431, 416), f(447, 432)],
            block: [f(607, 592), f(623, 608), f(639, 624)],
            shared: f(561, 544),
            barriers: f(1471, 1467),
        };
        let constant_buffers = std::array::from_fn(|i| {
            let i = i as u32;
            (f(640 + i, 640 + i) != 0).then(|| {
                let address = u64::from(f(967 + i * 64, 960 + i * 64)) << 32
                    | u64::from(f(959 + i * 64, 928 + i * 64));
                (address, f(991 + i * 64, 975 + i * 64))
            })
        });
        let release = |enable: u32, base: u32| {
            (f(enable, enable) != 0).then(|| Release {
                address: u64::from(f(base + 39, base + 32)) << 32 | u64::from(f(base + 31, base)),
                payload: f(base + 95, base + 64),
                four_words: f(base + 63, base + 63) == 0,
            })
        };
        Ok(Qmd {
            program_offset: f(287, 256),
            launch,
            constant_buffers,
            sampler_from_header: f(382, 382) != 0,
            releases: [release(202, 736), release(203, 832)],
        })
    }
}

#[cfg(test)]
mod tests;
