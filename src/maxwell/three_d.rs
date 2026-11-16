//! The Maxwell B 3D class (0xB197): its method register file, the methods
//! that act (draws, clears, constant buffer loads, reports, macros,
//! inline-to-memory), and the shadow RAM.
//!
//! Method offsets and fields: [public: NVIDIA open-gpu-doc
//! `classes/3d/clb197.h`]. Methods without an action are state: they are
//! stored and read back by draws, clears and macros.

use serde::{Deserialize, Serialize};

use super::inline::{self, Completion, InlineToMemory};
use super::mme::{MacroTarget, Mme};
use super::pipeline::{
    self, Batch, Bindings, Caches, ConstantBuffer, Context, Element, Registers, Topology,
};
use super::surface::{Rect, Window};
use super::{GpuMemory, Hooks, Unmodeled};

/// Method registers below the macro methods.
pub const METHODS: usize = 0xE00;

const fn m(offset: u32) -> u32 {
    offset / 4
}

const NO_OPERATION: u32 = m(0x0100);
const NOTIFY: u32 = m(0x010c);
const WAIT_FOR_IDLE: u32 = m(0x0110);
const LOAD_MME_INSTRUCTION_RAM_POINTER: u32 = m(0x0114);
const LOAD_MME_INSTRUCTION_RAM: u32 = m(0x0118);
const LOAD_MME_START_ADDRESS_RAM_POINTER: u32 = m(0x011c);
const LOAD_MME_START_ADDRESS_RAM: u32 = m(0x0120);
const SET_MME_SHADOW_RAM_CONTROL: u32 = m(0x0124);
const SET_GLOBAL_RENDER_ENABLE_C: u32 = m(0x0138);
const SEND_GO_IDLE: u32 = m(0x013c);
const PM_TRIGGER: u32 = m(0x0140);
const PM_TRIGGER_WFI: u32 = m(0x0144);
const INVALIDATE_SHADER_CACHES: u32 = m(0x021c);
const INCREMENT_SYNC_POINT: u32 = m(0x02c8);
const FLUSH_AND_INVALIDATE_ROP_MINI_CACHE: u32 = m(0x02d4);
const DECOMPRESS_SURFACE: u32 = m(0x02e0);
const PEER_SEMAPHORE_RELEASE: u32 = m(0x02f0);
const DRAW_ZERO_INDEX: u32 = m(0x0304);
const SET_VERTEX_ARRAY_START: u32 = m(0x0d74);
const DRAW_VERTEX_ARRAY: u32 = m(0x0d78);
const SET_COLOR_CLEAR_VALUE: u32 = m(0x0d80);
const SET_Z_CLEAR_VALUE: u32 = m(0x0d90);
const SET_STENCIL_CLEAR_VALUE: u32 = m(0x0da0);
const INVALIDATE_SHADER_CACHES_NO_WFI: u32 = m(0x0da4);
const PIXEL_SHADER_BARRIER: u32 = m(0x0de0);
const LOAD_ITERATED_BLEND_INSTRUCTION: u32 = m(0x0df4);
const PM_LOCAL_TRIGGER: u32 = m(0x0f18);
const UNBIND_ALL: u32 = m(0x10f4);
const SET_CLEAR_SURFACE_CONTROL: u32 = m(0x10f8);
const SET_STENCIL_FRONT_MASK: u32 = m(0x139c);
const PM_TRIGGER_END: u32 = m(0x1114);
const FLUSH_PENDING_WRITES: u32 = m(0x1144);
const FORCE_HEAVYWEIGHT_METHOD_SYNC: u32 = m(0x11f4);
const DECOMPRESS_ZETA_SURFACE: u32 = m(0x11fc);
const INVALIDATE_SAMPLER_CACHE_ALL: u32 = m(0x120c);
const INVALIDATE_TEXTURE_HEADER_CACHE_ALL: u32 = m(0x1210);
const DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_FIRST: u32 = m(0x1214);
const DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_SUBSEQUENT: u32 = m(0x1218);
const DRAW_AUTO: u32 = m(0x123c);
const INVALIDATE_TEXTURE_DATA_CACHE_NO_WFI: u32 = m(0x1288);
const CLEAR_ZCULL_REGION: u32 = m(0x12c8);
const DRAW_INLINE_INDEX4X8: u32 = m(0x1304);
const INVALIDATE_SAMPLER_CACHE: u32 = m(0x1330);
const INVALIDATE_TEXTURE_HEADER_CACHE: u32 = m(0x1334);
const INVALIDATE_TEXTURE_DATA_CACHE: u32 = m(0x1338);
const INVALIDATE_SAMPLER_CACHE_NO_WFI: u32 = m(0x1424);
const INVALIDATE_TEXTURE_HEADER_CACHE_NO_WFI: u32 = m(0x1428);
const SET_GLOBAL_BASE_VERTEX_INDEX: u32 = m(0x1434);
const SET_GLOBAL_BASE_INSTANCE_INDEX: u32 = m(0x1438);
const STORE_ZCULL: u32 = m(0x1464);
const LOAD_ZCULL: u32 = m(0x1500);
const CLEAR_REPORT_VALUE: u32 = m(0x1530);
const SET_RENDER_ENABLE_C: u32 = m(0x1558);
const DRAW_INLINE_INDEX: u32 = m(0x15e8);
const DRAW_INLINE_INDEX2X16: u32 = m(0x15f0);
const END: u32 = m(0x1614);
const BEGIN: u32 = m(0x1618);
const LOAD_PRIMITIVE_ID: u32 = m(0x1624);
const CHECK_CLASS_VERSION: u32 = m(0x16a0);
const CHECK_SPH_VERSION: u32 = m(0x16a8);
const CHECK_AAM_VERSION: u32 = m(0x1794);
const SET_INDEX_BUFFER_A: u32 = m(0x17c8);
const SET_INDEX_BUFFER_C: u32 = m(0x17d0);
const SET_INDEX_BUFFER_E: u32 = m(0x17d8);
const SET_INDEX_BUFFER_F: u32 = m(0x17dc);
const DRAW_INDEX_BUFFER: u32 = m(0x17e0);
const DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_FIRST: u32 = m(0x17e4);
const DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_SUBSEQUENT: u32 = m(0x17f8);
const SET_DA_PRIMITIVE_RESTART: u32 = m(0x1644);
const SET_DA_PRIMITIVE_RESTART_INDEX: u32 = m(0x1648);
const SET_VERTEX_ID_BASE: u32 = m(0x1118);
const SET_DA_OUTPUT: u32 = m(0x164c);
const SET_RENDER_ENABLE_OVERRIDE: u32 = m(0x1944);
const SET_PRIMITIVE_TOPOLOGY_CONTROL: u32 = m(0x1948);
const INVALIDATE_ZCULL: u32 = m(0x1958);
const ZCULL_SYNC: u32 = m(0x1978);
const CLEAR_SURFACE: u32 = m(0x19d0);
const CLEAR_CLIP_ID_SURFACE: u32 = m(0x19d4);
const PIPE_NOP: u32 = m(0x1a2c);
const SET_REPORT_SEMAPHORE_A: u32 = m(0x1b00);
const SET_REPORT_SEMAPHORE_C: u32 = m(0x1b08);
const SET_REPORT_SEMAPHORE_D: u32 = m(0x1b0c);
const SET_FALCON04: u32 = m(0x2310);
const SET_MME_SHADOW_SCRATCH: u32 = m(0x3400);
const SET_CONSTANT_BUFFER_SELECTOR_A: u32 = m(0x2380);
const SET_CONSTANT_BUFFER_SELECTOR_B: u32 = m(0x2384);
const SET_CONSTANT_BUFFER_SELECTOR_C: u32 = m(0x2388);
const LOAD_CONSTANT_BUFFER_OFFSET: u32 = m(0x238c);
const LOAD_CONSTANT_BUFFER: u32 = m(0x2390);
const BIND_GROUP_CONSTANT_BUFFER: u32 = m(0x2410);
const START_SHADER_PERFORMANCE_COUNTER: u32 = m(0x33e0);
const STOP_SHADER_PERFORMANCE_COUNTER: u32 = m(0x33e4);

/// Methods that take part in the shadow RAM even below the first 0x80
/// methods; see `ThreeD::shadow`.
const SHADOW_FIRST_METHOD: u32 = 0x80;

/// The 3D engine of a channel. Machine snapshots save everything but its
/// caches, which a restored engine rebuilds from guest memory.
#[derive(Serialize, Deserialize)]
pub struct ThreeD {
    registers: Vec<u32>,
    shadow: Vec<u32>,
    inline: InlineToMemory,
    bindings: Bindings,
    /// The render enable latched by the last `SET_RENDER_ENABLE_C`.
    render_enable: bool,
    batch: Option<Batch>,
    /// Instance of the current/last `BEGIN`.
    instance: u32,
    #[serde(skip)]
    caches: Caches,
    /// Graphics-engine priv registers written through `SET_FALCON04`:
    /// (value, mask of the bits written so far).
    priv_registers: std::collections::HashMap<u32, (u32, u32)>,
}

impl Default for ThreeD {
    fn default() -> Self {
        Self {
            registers: vec![0; METHODS],
            shadow: vec![0; METHODS],
            inline: InlineToMemory::default(),
            bindings: [[None; 18]; 5],
            // Rendering is enabled until a SET_RENDER_ENABLE_C says
            // otherwise: the netlist's initial value of that register is 0
            // (FALSE), which would leave nothing ever drawn if the
            // register itself governed rendering [public: eden
            // `Maxwell3D` keeps `execute_on = true` until the method runs].
            render_enable: true,
            batch: None,
            instance: 0,
            caches: Caches::default(),
            priv_registers: Default::default(),
        }
    }
}

fn refuse(method: u32, argument: u32, what: &str) -> Unmodeled {
    Unmodeled(format!(
        "3D (0xB197): {what}: method {:#06x} = {argument:#x}",
        method * 4
    ))
}

fn write_words(memory: &dyn GpuMemory, address: u64, words: &[u32]) -> Result<(), Unmodeled> {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    memory.write(address, &bytes).ok_or_else(|| {
        Unmodeled(format!(
            "semaphore write to unmapped GPU address {address:#x}"
        ))
    })
}

/// Writes a semaphore release: the payload alone, or the four-word
/// structure (payload, 0, 64-bit timestamp) [public: eden
/// `Maxwell3D::StampQueryResult`, NVIDIA `clb197.h` `STRUCTURE_SIZE`].
pub fn release_semaphore(
    memory: &dyn GpuMemory,
    hooks: &Hooks,
    address: u64,
    payload: u32,
    four_words: bool,
) -> Result<(), Unmodeled> {
    hooks.signal(memory)?;
    if !four_words {
        return write_words(memory, address, &[payload]);
    }
    let timer = hooks.timer.as_ref().ok_or_else(|| {
        Unmodeled("four-word semaphore release without a GPU timer connected".into())
    })?;
    let time = timer();
    write_words(
        memory,
        address,
        &[payload, 0, time as u32, (time >> 32) as u32],
    )
}

impl ThreeD {
    /// Whether a restored register file and shadow RAM have the size
    /// methods index.
    pub(super) fn fits(&self) -> bool {
        self.registers.len() == METHODS && self.shadow.len() == METHODS
    }

    pub fn registers(&self) -> Registers<'_> {
        Registers(&self.registers)
    }

    /// Sets a register's initial value (shadow RAM and register file).
    pub fn set_initial(&mut self, method: u32, value: u32) -> Result<(), Unmodeled> {
        let index = method as usize;
        if index >= METHODS {
            return Err(refuse(
                method,
                value,
                "initial state outside the method registers",
            ));
        }
        self.registers[index] = value;
        self.shadow[index] = value;
        Ok(())
    }

    /// The value a macro's register read returns.
    pub fn read(&self, method: u32) -> Result<u32, Unmodeled> {
        self.registers
            .get(method as usize)
            .copied()
            .ok_or_else(|| refuse(method, 0, "macro read beyond the method registers"))
    }

    /// One method call from the pushbuffer or from a macro.
    pub fn method(
        &mut self,
        method: u32,
        argument: u32,
        memory: &dyn GpuMemory,
        mme: &mut Mme,
        hooks: &Hooks,
    ) -> Result<(), Unmodeled> {
        if method as usize >= METHODS {
            let mut target = Target {
                three_d: self,
                memory,
                hooks,
            };
            return mme.method(method, argument, &mut target);
        }
        if mme.busy() {
            return Err(refuse(
                method,
                argument,
                "method while a macro waits for parameters",
            ));
        }
        self.execute(method, argument, memory, mme, hooks)
    }

    /// The shadow RAM [public: NVIDIA `clb197.h`
    /// `SET_MME_SHADOW_RAM_CONTROL`; semantics from Ryujinx
    /// `DeviceStateWithShadow` and eden `Maxwell3D::ProcessShadowRam`]:
    /// in the track modes a write is also recorded in the shadow RAM, in
    /// replay mode the recorded value replaces the argument. The two
    /// reimplementations disagree for the first 0x80 methods (Ryujinx
    /// passes them through, eden shadows them), so replay of those is
    /// refused.
    fn shadow(&mut self, method: u32, argument: u32) -> Result<u32, Unmodeled> {
        let index = method as usize;
        if method == SET_MME_SHADOW_RAM_CONTROL {
            return Ok(argument);
        }
        match self.registers[SET_MME_SHADOW_RAM_CONTROL as usize] & 3 {
            0 | 1 => {
                self.shadow[index] = argument;
                Ok(argument)
            }
            2 => Ok(argument),
            _ if method < SHADOW_FIRST_METHOD => Err(refuse(
                method,
                argument,
                "shadow RAM replay of a method below 0x200",
            )),
            _ => Ok(self.shadow[index]),
        }
    }

    fn execute(
        &mut self,
        method: u32,
        argument: u32,
        memory: &dyn GpuMemory,
        mme: &mut Mme,
        hooks: &Hooks,
    ) -> Result<(), Unmodeled> {
        if self.inline.busy() && method != m(0x01b4) {
            return Err(refuse(
                method,
                argument,
                "method while inline data is expected",
            ));
        }
        let argument = self.shadow(method, argument)?;
        self.registers[method as usize] = argument;
        match method {
            inline::FIRST_METHOD..=inline::LAST_METHOD => {
                match self.inline.method(method, argument, memory)? {
                    Completion::None => {}
                    Completion::Semaphore {
                        address,
                        payload,
                        four_words,
                    } => release_semaphore(memory, hooks, address, payload, four_words)?,
                }
            }
            LOAD_MME_INSTRUCTION_RAM_POINTER => mme.set_code_pointer(argument),
            LOAD_MME_INSTRUCTION_RAM => mme.load_code(argument)?,
            LOAD_MME_START_ADDRESS_RAM_POINTER => mme.set_start_pointer(argument),
            LOAD_MME_START_ADDRESS_RAM => mme.load_start_address(argument)?,
            // Execution is synchronous and in order: everything before a
            // wait, barrier, flush or idle request has completed when it
            // arrives; there are no caches of guest memory to flush or
            // invalidate (see `resources`), and no zcull, compression or
            // performance monitor whose state could be observed.
            NO_OPERATION
            | WAIT_FOR_IDLE
            | SEND_GO_IDLE
            | PM_TRIGGER
            | PM_TRIGGER_WFI
            | PM_LOCAL_TRIGGER
            | PM_TRIGGER_END
            | INVALIDATE_SHADER_CACHES
            | INVALIDATE_SHADER_CACHES_NO_WFI
            | FLUSH_AND_INVALIDATE_ROP_MINI_CACHE
            | PIXEL_SHADER_BARRIER
            | FLUSH_PENDING_WRITES
            | FORCE_HEAVYWEIGHT_METHOD_SYNC
            | INVALIDATE_SAMPLER_CACHE_ALL
            | INVALIDATE_TEXTURE_HEADER_CACHE_ALL
            | INVALIDATE_TEXTURE_DATA_CACHE_NO_WFI
            | INVALIDATE_SAMPLER_CACHE
            | INVALIDATE_TEXTURE_HEADER_CACHE
            | INVALIDATE_TEXTURE_DATA_CACHE
            | INVALIDATE_SAMPLER_CACHE_NO_WFI
            | INVALIDATE_TEXTURE_HEADER_CACHE_NO_WFI
            | CLEAR_ZCULL_REGION
            | INVALIDATE_ZCULL
            | ZCULL_SYNC
            | STORE_ZCULL
            | LOAD_ZCULL
            | PIPE_NOP
            | START_SHADER_PERFORMANCE_COUNTER
            | STOP_SHADER_PERFORMANCE_COUNTER => {}
            // Surfaces are never stored compressed by this model (the
            // memory image always holds plain texels), so there is
            // nothing to decompress. NVN's present path issues
            // DECOMPRESS_SURFACE (starter 0x262250, local nvn-methods §4).
            DECOMPRESS_SURFACE | DECOMPRESS_ZETA_SURFACE => {}
            // Version checks: the class and SPH versions this model
            // implements are those of the GM20B the firmware targets; a
            // mismatch would raise an error interrupt, never modeled.
            CHECK_CLASS_VERSION | CHECK_SPH_VERSION | CHECK_AAM_VERSION => {}
            NOTIFY => return Err(refuse(method, argument, "notifier")),
            INCREMENT_SYNC_POINT => {
                let increment = hooks.syncpoint_increment.as_ref().ok_or_else(|| {
                    refuse(
                        method,
                        argument,
                        "syncpoint increment without host1x connected",
                    )
                })?;
                hooks.signal(memory)?;
                increment(argument & 0xFFF);
            }
            SET_GLOBAL_RENDER_ENABLE_C => {
                if argument & 7 != 1 {
                    return Err(refuse(
                        method,
                        argument,
                        "global render enable other than TRUE",
                    ));
                }
            }
            SET_RENDER_ENABLE_C => {
                self.render_enable = match argument & 7 {
                    0 => false,
                    1 => true,
                    _ => {
                        return Err(refuse(method, argument, "conditional rendering"));
                    }
                };
            }
            SET_CONSTANT_BUFFER_SELECTOR_A..=SET_CONSTANT_BUFFER_SELECTOR_C
            | LOAD_CONSTANT_BUFFER_OFFSET => {}
            _ if (LOAD_CONSTANT_BUFFER..LOAD_CONSTANT_BUFFER + 16).contains(&method) => {
                self.load_constant_buffer(argument, memory)?;
            }
            _ if (BIND_GROUP_CONSTANT_BUFFER..BIND_GROUP_CONSTANT_BUFFER + 5 * 8)
                .contains(&method)
                && (method - BIND_GROUP_CONSTANT_BUFFER).is_multiple_of(8) =>
            {
                self.bind_constant_buffer((method - BIND_GROUP_CONSTANT_BUFFER) / 8, argument)?;
            }
            UNBIND_ALL => return Err(refuse(method, argument, "UNBIND_ALL")),
            SET_REPORT_SEMAPHORE_D => self.report(argument, memory, hooks)?,
            CLEAR_REPORT_VALUE => return Err(refuse(method, argument, "counter reset")),
            CLEAR_SURFACE => {
                if self.rendering() {
                    self.clear(argument, memory, hooks)?;
                }
            }
            BEGIN => self.begin(argument)?,
            END => self.end(memory, hooks)?,
            DRAW_VERTEX_ARRAY => {
                let start = self.registers[SET_VERTEX_ARRAY_START as usize];
                self.push_array(start, argument)?;
            }
            DRAW_INDEX_BUFFER => {
                let first = self.registers[SET_INDEX_BUFFER_F as usize];
                let size = self.registers[SET_INDEX_BUFFER_E as usize] & 3;
                self.push_indexed(first, argument, size, memory)?;
            }
            DRAW_INLINE_INDEX => self.push_inline(argument, 1)?,
            DRAW_INLINE_INDEX2X16 => {
                self.push_inline(argument & 0xFFFF, 1)?;
                self.push_inline(argument >> 16, 1)?;
            }
            DRAW_INLINE_INDEX4X8 => {
                for shift in [0, 8, 16, 24] {
                    self.push_inline((argument >> shift) & 0xFF, 1)?;
                }
            }
            DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_FIRST
            | DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_SUBSEQUENT => {
                let instance = if method == DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_FIRST {
                    0
                } else {
                    1
                };
                self.begin((argument >> 28) | instance << 26)?;
                self.push_array(argument & 0xFFFF, (argument >> 16) & 0xFFF)?;
                self.end(memory, hooks)?;
            }
            DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_FIRST
                ..=DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_SUBSEQUENT => {
                // 0x17e4/e8/ec: 32/16/8-bit indices, first instance;
                // 0x17f0/f4/f8: the same, subsequent instance.
                let position = method - DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_FIRST;
                let size = 2 - position % 3;
                let instance = position / 3;
                self.begin((argument >> 28) | instance << 26)?;
                self.push_indexed(argument & 0xFFFF, (argument >> 16) & 0xFFF, size, memory)?;
                self.end(memory, hooks)?;
            }
            DRAW_ZERO_INDEX
            | DRAW_AUTO
            | LOAD_PRIMITIVE_ID
            | PEER_SEMAPHORE_RELEASE
            | CLEAR_CLIP_ID_SURFACE
            | LOAD_ITERATED_BLEND_INSTRUCTION => {
                return Err(refuse(method, argument, "action not modeled"));
            }
            // SET_FALCON04: a request to the graphics context-switch
            // firmware (FECS); nvservices does not handle it as a software
            // method (its GR software-method handler
            // NV_HandleGm20bSoftwareMethod_87430 takes only 0x2DC, 0x1280
            // and 0x1528). NVN uses it as a masked write of a graphics
            // engine priv register: its macro 5 (starter upload 0x25b044)
            // stores a value in SET_MME_SHADOW_SCRATCH(1) and a mask in
            // SCRATCH(2), clears SCRATCH(0), writes the register address
            // here and polls SCRATCH(0) until it reads 1. The callers take
            // address, bit position and width from NVN's GM20B register
            // table (starter .data 0x4c6778 onwards: 0x418800, 0x419f78,
            // 0x419a08, 0x418e40, ...) and shift the value and an all-ones
            // field into place: init blast 0x25ce30,
            // nvnCommandBufferSetConservativeRasterDilate 0x256b90,
            // 0x257fd0, 0x25a7b0 (sample locations). The completion
            // handshake is what both public reimplementations do [public:
            // eden `Maxwell3D::ProcessFirmwareCall4`, Ryujinx
            // `ThreedClass.SetFalcon04`]. The write is kept in
            // `priv_registers`, so the state is there; what those
            // registers change in rasterization is not documented
            // publicly and is NOT modeled. Addresses outside the graphics
            // engine's priv range (0x400000-0x41FFFF) are refused.
            _ if method == SET_FALCON04 => {
                if !(0x40_0000..0x42_0000).contains(&argument) {
                    return Err(refuse(
                        method,
                        argument,
                        "firmware method call (SET_FALCON04) outside the graphics registers",
                    ));
                }
                let scratch = SET_MME_SHADOW_SCRATCH as usize;
                let (value, mask) = (self.registers[scratch + 1], self.registers[scratch + 2]);
                // Bits never written stay unknown (the register's reset
                // value is not known); only (value, known bits) is kept.
                let (known_value, known) = self.priv_registers.entry(argument).or_default();
                *known_value = (*known_value & !mask) | (value & mask);
                *known |= mask;
                self.registers[scratch] = 1;
            }
            _ => {}
        }
        Ok(())
    }

    fn rendering(&self) -> bool {
        match self.registers[SET_RENDER_ENABLE_OVERRIDE as usize] & 3 {
            1 => true,
            2 => false,
            _ => self.render_enable,
        }
    }

    fn load_constant_buffer(&mut self, word: u32, memory: &dyn GpuMemory) -> Result<(), Unmodeled> {
        let r = &self.registers;
        let size = r[SET_CONSTANT_BUFFER_SELECTOR_A as usize] & 0x1_FFFF;
        let address = u64::from(r[SET_CONSTANT_BUFFER_SELECTOR_B as usize] & 0xFF) << 32
            | u64::from(r[SET_CONSTANT_BUFFER_SELECTOR_C as usize]);
        let offset = r[LOAD_CONSTANT_BUFFER_OFFSET as usize];
        if offset + 4 > size {
            return Err(refuse(
                LOAD_CONSTANT_BUFFER,
                word,
                &format!("constant buffer load at {offset:#x} beyond its size {size:#x}"),
            ));
        }
        memory
            .write(address + u64::from(offset), &word.to_le_bytes())
            .ok_or_else(|| {
                Unmodeled(format!(
                    "3D: constant buffer load to unmapped GPU address {:#x}",
                    address + u64::from(offset)
                ))
            })?;
        // The offset advances by a word per load [public: eden
        // `Maxwell3D::ProcessCBData`, Ryujinx `ConstantBufferUpdater`].
        self.registers[LOAD_CONSTANT_BUFFER_OFFSET as usize] = offset + 4;
        Ok(())
    }

    fn bind_constant_buffer(&mut self, group: u32, argument: u32) -> Result<(), Unmodeled> {
        let slot = ((argument >> 4) & 0x1F) as usize;
        let slots = &mut self.bindings[group as usize];
        let Some(binding) = slots.get_mut(slot) else {
            return Err(refuse(
                BIND_GROUP_CONSTANT_BUFFER + group * 8,
                argument,
                "constant buffer slot beyond 17",
            ));
        };
        *binding = if argument & 1 != 0 {
            let r = &self.registers;
            Some(ConstantBuffer {
                address: u64::from(r[SET_CONSTANT_BUFFER_SELECTOR_B as usize] & 0xFF) << 32
                    | u64::from(r[SET_CONSTANT_BUFFER_SELECTOR_C as usize]),
                size: r[SET_CONSTANT_BUFFER_SELECTOR_A as usize] & 0x1_FFFF,
            })
        } else {
            None
        };
        Ok(())
    }

    /// `SET_REPORT_SEMAPHORE_D`: OPERATION 1:0, FLUSH_DISABLE 2,
    /// REDUCTION_ENABLE 3, SUB_REPORT 7:5, REDUCTION_OP 11:9,
    /// PIPELINE_LOCATION 15:12, COMPARISON 16, AWAKEN_ENABLE 20, REPORT
    /// 27:23, STRUCTURE_SIZE 28 (1 = one word).
    fn report(
        &mut self,
        control: u32,
        memory: &dyn GpuMemory,
        hooks: &Hooks,
    ) -> Result<(), Unmodeled> {
        let r = &self.registers;
        let address = u64::from(r[SET_REPORT_SEMAPHORE_A as usize] & 0xFF) << 32
            | u64::from(r[SET_REPORT_SEMAPHORE_A as usize + 1]);
        let payload = r[SET_REPORT_SEMAPHORE_C as usize];
        let operation = control & 3;
        let report = (control >> 23) & 0x1F;
        let one_word = control & (1 << 28) != 0;
        if control & (1 << 3) != 0 {
            return Err(refuse(
                SET_REPORT_SEMAPHORE_D,
                control,
                "semaphore reduction",
            ));
        }
        if control & (1 << 20) != 0 {
            return Err(refuse(
                SET_REPORT_SEMAPHORE_D,
                control,
                "semaphore release with interrupt",
            ));
        }
        match (operation, report) {
            // RELEASE, or REPORT_ONLY (2), of the payload with REPORT NONE:
            // both write the payload (and, four-word, a timestamp); they
            // differ only in the ordering a release implies, which a
            // synchronous model always provides [public: eden
            // `Maxwell3D::ProcessQueryGet` handles both alike]. NVN's
            // timestamp counter uses REPORT_ONLY with control 0xf002
            // (starter report table 0x3c8ba4, index 0).
            (0 | 2, 0) => release_semaphore(memory, hooks, address, payload, !one_word),
            // ACQUIRE: the engine waits until memory equals (COMPARISON
            // EQ) or is at least (GE) the payload. Execution is
            // synchronous, so a condition that does not hold now never
            // will without other agents; that wait is not modeled.
            (1, _) => {
                let mut bytes = [0; 4];
                memory.read(address, &mut bytes).ok_or_else(|| {
                    Unmodeled(format!("semaphore acquire at unmapped {address:#x}"))
                })?;
                let value = u32::from_le_bytes(bytes);
                let ge = control & (1 << 16) != 0;
                let done = if ge {
                    (value.wrapping_sub(payload) as i32) >= 0
                } else {
                    value == payload
                };
                if done {
                    Ok(())
                } else {
                    Err(refuse(
                        SET_REPORT_SEMAPHORE_D,
                        control,
                        &format!(
                            "semaphore acquire of {payload:#x} at {address:#x} holding {value:#x}"
                        ),
                    ))
                }
            }
            _ => Err(refuse(SET_REPORT_SEMAPHORE_D, control, "report or counter")),
        }
    }

    fn begin(&mut self, argument: u32) -> Result<(), Unmodeled> {
        if self.batch.is_some() {
            return Err(refuse(BEGIN, argument, "BEGIN inside BEGIN/END"));
        }
        if self.registers[SET_PRIMITIVE_TOPOLOGY_CONTROL as usize] & 1 != 0 {
            return Err(refuse(
                BEGIN,
                argument,
                "topology from SET_PRIMITIVE_TOPOLOGY",
            ));
        }
        if (argument >> 29) & 3 != 0 {
            return Err(refuse(BEGIN, argument, "split begin/end"));
        }
        // INSTANCE_ID 27:26: FIRST (0), SUBSEQUENT (1), UNCHANGED (2).
        self.instance = match (argument >> 26) & 3 {
            0 => 0,
            1 => self.instance + 1,
            2 => self.instance,
            _ => return Err(refuse(BEGIN, argument, "instance id mode 3")),
        };
        self.batch = Some(Batch {
            topology: Topology::from_raw(argument & 0xFFFF)?,
            instance: self.instance,
            base_instance: 0,
            elements: Vec::new(),
        });
        Ok(())
    }

    fn batch(&mut self, method: u32, argument: u32) -> Result<&mut Batch, Unmodeled> {
        self.batch
            .as_mut()
            .ok_or_else(|| refuse(method, argument, "draw outside BEGIN/END"))
    }

    /// Vertices `start..start + count` of the vertex arrays. The vertex ID
    /// counts from `start` when `SET_DA_OUTPUT` `VERTEX_ID_USES_ARRAY_START`
    /// (bit 12) is set, as NVN's init blast sets it, else from 0; plus
    /// `SET_VERTEX_ID_BASE`.
    fn push_array(&mut self, start: u32, count: u32) -> Result<(), Unmodeled> {
        let base = self.registers[SET_GLOBAL_BASE_VERTEX_INDEX as usize];
        if base != 0 {
            return Err(refuse(
                SET_GLOBAL_BASE_VERTEX_INDEX,
                base,
                "base vertex with a non-indexed draw",
            ));
        }
        let id_base = self.registers[SET_VERTEX_ID_BASE as usize];
        let from_start = self.registers[SET_DA_OUTPUT as usize] & (1 << 12) != 0;
        let batch = self.batch(DRAW_VERTEX_ARRAY, count)?;
        batch.elements.extend((0..count).map(|i| {
            let fetch = start.wrapping_add(i);
            let id = if from_start { fetch } else { i };
            Element::Vertex {
                fetch,
                id: id.wrapping_add(id_base),
            }
        }));
        Ok(())
    }

    /// An index: a primitive restart (`SET_DA_PRIMITIVE_RESTART`, compared
    /// before any base is added), or a vertex fetched at the index plus
    /// `SET_GLOBAL_BASE_VERTEX_INDEX` whose ID is the index plus
    /// `SET_VERTEX_ID_BASE` (NVN's base-vertex draws set both, starter
    /// macro 29).
    fn element(&self, index: u32) -> Element {
        let restart = self.registers[SET_DA_PRIMITIVE_RESTART as usize] & 1 != 0;
        if restart && index == self.registers[SET_DA_PRIMITIVE_RESTART_INDEX as usize] {
            Element::Restart
        } else {
            Element::Vertex {
                fetch: index.wrapping_add(self.registers[SET_GLOBAL_BASE_VERTEX_INDEX as usize]),
                id: index.wrapping_add(self.registers[SET_VERTEX_ID_BASE as usize]),
            }
        }
    }

    fn push_inline(&mut self, index: u32, _count: u32) -> Result<(), Unmodeled> {
        let element = self.element(index);
        self.batch(DRAW_INLINE_INDEX, index)?.elements.push(element);
        Ok(())
    }

    /// Indices from the index buffer (`SET_INDEX_BUFFER_A`-`D`: address and
    /// limit address), `size` 0/1/2 for 1/2/4-byte indices.
    fn push_indexed(
        &mut self,
        first: u32,
        count: u32,
        size: u32,
        memory: &dyn GpuMemory,
    ) -> Result<(), Unmodeled> {
        let r = &self.registers;
        let address = u64::from(r[SET_INDEX_BUFFER_A as usize] & 0xFF) << 32
            | u64::from(r[SET_INDEX_BUFFER_A as usize + 1]);
        let limit = u64::from(r[SET_INDEX_BUFFER_C as usize] & 0xFF) << 32
            | u64::from(r[SET_INDEX_BUFFER_C as usize + 1]);
        if size == 3 {
            return Err(refuse(SET_INDEX_BUFFER_E, size, "index size"));
        }
        let bytes_per_index = 1u64 << size;
        let start = address + u64::from(first) * bytes_per_index;
        let length = u64::from(count) * bytes_per_index;
        if count > 0 && start + length - 1 > limit {
            return Err(refuse(
                DRAW_INDEX_BUFFER,
                count,
                &format!("indices {start:#x}+{length:#x} beyond the limit {limit:#x}"),
            ));
        }
        let mut bytes = vec![0; length as usize];
        memory
            .read(start, &mut bytes)
            .ok_or_else(|| Unmodeled(format!("3D: index buffer at unmapped {start:#x}")))?;
        let elements: Vec<Element> = bytes
            .chunks(bytes_per_index as usize)
            .map(|chunk| {
                let mut word = [0; 4];
                word[..chunk.len()].copy_from_slice(chunk);
                self.element(u32::from_le_bytes(word))
            })
            .collect();
        self.batch(DRAW_INDEX_BUFFER, count)?
            .elements
            .extend(elements);
        Ok(())
    }

    fn end(&mut self, memory: &dyn GpuMemory, hooks: &Hooks) -> Result<(), Unmodeled> {
        let mut batch = self
            .batch
            .take()
            .ok_or_else(|| refuse(END, 0, "END without BEGIN"))?;
        if !self.rendering() || batch.elements.is_empty() {
            return Ok(());
        }
        batch.base_instance = self.registers[SET_GLOBAL_BASE_INSTANCE_INDEX as usize];
        let context = Context {
            registers: Registers(&self.registers),
            bindings: &self.bindings,
            memory,
            metal: hooks.metal.as_deref(),
            writes: &hooks.writes,
        };
        pipeline::draw(&context, &mut self.caches, &batch)
    }

    /// `CLEAR_SURFACE`: Z_ENABLE 0, STENCIL_ENABLE 1, R/G/B/A_ENABLE 5:2,
    /// MRT_SELECT 9:6, RT_ARRAY_INDEX 25:10. The rectangle is the surface
    /// clip, limited by `SET_CLEAR_SURFACE_CONTROL`'s USE_CLEAR_RECT (4),
    /// USE_SCISSOR0 (8) and USE_VIEWPORT_CLIP0 (12). A stencil clear sets
    /// `SET_STENCIL_CLEAR_VALUE` in a target whose format has stencil (one
    /// without has nothing to clear); with the control's bit 0 only the
    /// bits the front stencil write mask (`0x139c`) lets through. Offsets
    /// and that bit as in envytools' `gf100_3d.xml` (`CLEAR_STENCIL`,
    /// `CLEAR_FLAGS.STENCIL_MASK`, `STENCIL_FRONT_MASK`). deko3d sets the
    /// bit and clears depth and stencil together (`0x3`, ftpd's first
    /// frame).
    fn clear(
        &mut self,
        argument: u32,
        memory: &dyn GpuMemory,
        hooks: &Hooks,
    ) -> Result<(), Unmodeled> {
        let r = Registers(&self.registers);
        if (argument >> 10) & 0xFFFF != 0 {
            return Err(refuse(CLEAR_SURFACE, argument, "clear of an array layer"));
        }
        let control = r.get(SET_CLEAR_SURFACE_CONTROL * 4);
        let stencil_mask = match control & 1 {
            0 => 0xFF,
            _ => r.get(SET_STENCIL_FRONT_MASK * 4) as u8,
        };
        let mut rect = pipeline::surface_clip(r);
        if control & (1 << 4) != 0 {
            let horizontal = r.get(0x0d6c);
            let vertical = r.get(0x0d70);
            rect = rect.intersect(&Rect {
                x0: horizontal & 0xFFFF,
                y0: vertical & 0xFFFF,
                x1: horizontal >> 16,
                y1: vertical >> 16,
            });
        }
        if control & (1 << 8) != 0
            && let Some(scissor) = pipeline::scissor(r)
        {
            rect = rect.intersect(&scissor);
        }
        if control & (1 << 12) != 0 {
            let horizontal = r.get(0x0c00);
            let vertical = r.get(0x0c04);
            let x = horizontal & 0xFFFF;
            let y = vertical & 0xFFFF;
            rect = rect.intersect(&Rect {
                x0: x,
                y0: y,
                x1: x + (horizontal >> 16),
                y1: y + (vertical >> 16),
            });
        }
        if r.get(0x13ac) & 1 != 0 {
            return Err(refuse(
                CLEAR_SURFACE,
                argument,
                "clear with a lower-left window origin",
            ));
        }
        let components = (argument >> 2) & 0xF;
        if components != 0 {
            let slot = (argument >> 6) & 0xF;
            let select = r.get(0x121c);
            if slot >= (select & 0xF) {
                return Err(refuse(
                    CLEAR_SURFACE,
                    argument,
                    "clear of a render target slot beyond SET_CT_SELECT",
                ));
            }
            let target = (select >> (4 + 3 * slot)) & 7;
            if let Some((surface, format)) = pipeline::color_target(r, target)? {
                let color: [f32; 4] = std::array::from_fn(|i| {
                    f32::from_bits(r.get(SET_COLOR_CLEAR_VALUE * 4 + 4 * i as u32))
                });
                if let Some(metal) = &hooks.metal
                    && metal.clear(
                        memory,
                        &hooks.writes,
                        (surface, format),
                        rect,
                        color,
                        components as u8,
                    )?
                {
                    // Done on the GPU.
                } else {
                    let mut window = Window::load(memory, surface, rect)?;
                    let full = components == 0xF;
                    let mut packed = vec![0; surface.bytes_per_pixel as usize];
                    format.write(color, &mut packed);
                    for mut band in window.bands_mut() {
                        for y in band.y0.max(rect.y0)..(band.y0 + 8).min(rect.y1) {
                            for x in rect.x0..rect.x1.min(surface.width) {
                                let texel = band.pixel_mut(x, y);
                                if full {
                                    texel.copy_from_slice(&packed);
                                } else {
                                    let mut value = format.read(texel);
                                    for (i, v) in value.iter_mut().enumerate() {
                                        if components & (1 << i) != 0 {
                                            *v = color[i];
                                        }
                                    }
                                    format.write(value, texel);
                                }
                            }
                        }
                    }
                    window.store(memory)?;
                }
            }
        }
        if argument & 3 != 0 {
            let Some((surface, format)) = pipeline::depth_target(r)? else {
                return Err(refuse(
                    CLEAR_SURFACE,
                    argument,
                    "depth or stencil clear without a depth target",
                ));
            };
            let depth = f32::from_bits(r.get(SET_Z_CLEAR_VALUE * 4));
            let stencil = r.get(SET_STENCIL_CLEAR_VALUE * 4) as u8;
            let mut window = Window::load(memory, surface, rect)?;
            for mut band in window.bands_mut() {
                for y in band.y0.max(rect.y0)..(band.y0 + 8).min(rect.y1) {
                    for x in rect.x0..rect.x1.min(surface.width) {
                        let texel = band.pixel_mut(x, y);
                        if argument & 1 != 0 {
                            format.write_depth(depth, texel);
                        }
                        if argument & 2 != 0
                            && let Some(kept) = format.read_stencil(texel)
                        {
                            let merged = kept & !stencil_mask | stencil & stencil_mask;
                            format.write_stencil(merged, texel);
                        }
                    }
                }
            }
            window.store(memory)?;
        }
        Ok(())
    }
}

/// A macro's view of the 3D engine.
struct Target<'a> {
    three_d: &'a mut ThreeD,
    memory: &'a dyn GpuMemory,
    hooks: &'a Hooks,
}

impl MacroTarget for Target<'_> {
    fn read_method(&self, method: u32) -> Result<u32, Unmodeled> {
        self.three_d.read(method)
    }

    fn send_method(&mut self, method: u32, argument: u32) -> Result<(), Unmodeled> {
        if method as usize >= METHODS {
            return Err(refuse(method, argument, "macro sending a macro method"));
        }
        // A macro's methods go to the engine, not back into the MME; a
        // macro loading macro RAM is not modeled.
        if (LOAD_MME_INSTRUCTION_RAM_POINTER..=LOAD_MME_START_ADDRESS_RAM).contains(&method) {
            return Err(refuse(method, argument, "macro loading macro RAM"));
        }
        let mut unused = Mme::default();
        self.three_d
            .execute(method, argument, self.memory, &mut unused, self.hooks)
    }
}
