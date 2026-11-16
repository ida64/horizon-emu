//! The Maxwell DMA copy class (0xB0B5; 0xA0B5 shares its methods):
//! buffer copies, 2D copies between pitch and block-linear layouts, and
//! component remapping.
//!
//! Methods and fields [public: NVIDIA open-gpu-doc
//! `classes/dma-copy/clb0b5.h`]. `LAUNCH_DMA` starts the operation.
//! Semantics as the public reimplementations agree [public: eden
//! `video_core/engines/maxwell_dma.cpp`]: with `MULTI_LINE_ENABLE`,
//! `LINE_COUNT` lines of `LINE_LENGTH_IN` elements are copied; an element
//! is a byte, or with `REMAP_ENABLE` `NUM_DST_COMPONENTS x COMPONENT_SIZE`
//! bytes; block-linear surfaces give width and origin x in the same units.
//! Pitch surfaces start at the offset and step by the pitch. Without
//! `MULTI_LINE_ENABLE` it is a single line at the offsets.

use serde::{Deserialize, Serialize};

use super::surface::{Layout, Rect, Surface, Window};
use super::texture::layout::BlockLinear;
use super::three_d::release_semaphore;
use super::{GpuMemory, Hooks, Unmodeled};

const METHODS: usize = 0x800;

const fn m(offset: u32) -> usize {
    (offset / 4) as usize
}

const NOP: usize = m(0x100);
const PM_TRIGGER: usize = m(0x140);
const SET_SEMAPHORE_A: usize = m(0x240);
const SET_SEMAPHORE_PAYLOAD: usize = m(0x248);
const SET_RENDER_ENABLE_C: usize = m(0x25c);
const LAUNCH_DMA: usize = m(0x300);
const OFFSET_IN_UPPER: usize = m(0x400);
const OFFSET_OUT_UPPER: usize = m(0x408);
const PITCH_IN: usize = m(0x410);
const PITCH_OUT: usize = m(0x414);
const LINE_LENGTH_IN: usize = m(0x418);
const LINE_COUNT: usize = m(0x41c);
const SET_REMAP_CONST_A: usize = m(0x700);
const SET_REMAP_CONST_B: usize = m(0x704);
const SET_REMAP_COMPONENTS: usize = m(0x708);
const SET_DST_BLOCK_SIZE: usize = m(0x70c);
const SET_SRC_BLOCK_SIZE: usize = m(0x728);

/// The copy engine of a channel; saved whole in machine snapshots.
#[derive(Serialize, Deserialize)]
pub struct Copy {
    registers: Vec<u32>,
    render_enable: bool,
}

impl Default for Copy {
    fn default() -> Self {
        Self {
            registers: vec![0; METHODS],
            render_enable: true,
        }
    }
}

fn refuse(method: usize, argument: u32, what: &str) -> Unmodeled {
    Unmodeled(format!(
        "copy (0xB0B5): {what}: method {:#06x} = {argument:#x}",
        method * 4
    ))
}

/// Component remapping (`SET_REMAP_COMPONENTS`).
#[derive(Clone, Copy)]
struct Remap {
    /// Source of each destination component: 0-3 source component, 4
    /// CONST_A, 5 CONST_B, 6 NO_WRITE.
    select: [u32; 4],
    component_size: usize,
    source_components: usize,
    destination_components: usize,
    constants: [u32; 2],
}

impl Remap {
    fn source_size(&self) -> usize {
        self.component_size * self.source_components
    }

    fn destination_size(&self) -> usize {
        self.component_size * self.destination_components
    }

    /// One element: `source` -> `destination` (which holds the old value,
    /// kept for NO_WRITE components).
    fn apply(&self, source: &[u8], destination: &mut [u8]) {
        let size = self.component_size;
        for (component, &select) in self.select[..self.destination_components]
            .iter()
            .enumerate()
        {
            let out = &mut destination[component * size..][..size];
            match select {
                0..=3 => out.copy_from_slice(&source[select as usize * size..][..size]),
                4 | 5 => {
                    out.copy_from_slice(&self.constants[select as usize - 4].to_le_bytes()[..size])
                }
                _ => {}
            }
        }
    }
}

impl Copy {
    /// Whether a restored register file has the size methods index.
    pub(super) fn fits(&self) -> bool {
        self.registers.len() == METHODS
    }

    pub fn method(
        &mut self,
        method: u32,
        argument: u32,
        memory: &dyn GpuMemory,
        hooks: &Hooks,
    ) -> Result<(), Unmodeled> {
        let index = method as usize;
        if index >= METHODS {
            return Err(refuse(index, argument, "method beyond the class"));
        }
        self.registers[index] = argument;
        match index {
            // Execution is synchronous; performance monitors are not
            // modeled and have no effect on memory.
            NOP | PM_TRIGGER => Ok(()),
            SET_RENDER_ENABLE_C => {
                self.render_enable = match argument & 7 {
                    0 => false,
                    1 => true,
                    _ => return Err(refuse(index, argument, "conditional copy")),
                };
                Ok(())
            }
            LAUNCH_DMA => self.launch(argument, memory, hooks),
            _ => Ok(()),
        }
    }

    fn address(&self, upper: usize) -> u64 {
        u64::from(self.registers[upper] & 0xFF) << 32 | u64::from(self.registers[upper + 1])
    }

    fn remap(&self) -> Remap {
        let raw = self.registers[SET_REMAP_COMPONENTS];
        Remap {
            select: [0, 4, 8, 12].map(|shift| (raw >> shift) & 7),
            component_size: ((raw >> 16) & 3) as usize + 1,
            source_components: ((raw >> 20) & 3) as usize + 1,
            destination_components: ((raw >> 24) & 3) as usize + 1,
            constants: [
                self.registers[SET_REMAP_CONST_A],
                self.registers[SET_REMAP_CONST_B],
            ],
        }
    }

    /// The source (`SET_SRC_*`, base 0x728) or destination (`SET_DST_*`,
    /// base 0x70c) surface of a 2D copy: BLOCK_SIZE, WIDTH, HEIGHT, DEPTH,
    /// LAYER, ORIGIN at +0, +4, +8, +0xc, +0x10, +0x14.
    fn block_surface(
        &self,
        base: usize,
        address: u64,
        element: usize,
    ) -> Result<(Surface, u32, u32), Unmodeled> {
        let r = |offset: usize| self.registers[base + offset];
        let block = r(0);
        if block & 0xF != 0 {
            return Err(refuse(base, block, "block width other than one GOB"));
        }
        // GOB_HEIGHT 15:12: FERMI_8 (1) is the GOB of this GPU; the Tesla
        // 4-row GOB is not modeled.
        if (block >> 12) & 0xF != 1 {
            return Err(refuse(base, block, "GOB height other than FERMI_8"));
        }
        if r(3) > 1 || r(4) != 0 {
            return Err(refuse(base + 3, r(3), "3D or layered copy surface"));
        }
        let origin = r(5);
        Ok((
            Surface {
                address,
                width: r(1),
                height: r(2),
                bytes_per_pixel: element as u32,
                layout: Layout::BlockLinear(BlockLinear {
                    block_height_log2: (block >> 4) & 0xF,
                    block_depth_log2: (block >> 8) & 0xF,
                }),
            },
            origin & 0xFFFF,
            origin >> 16,
        ))
    }

    fn launch(
        &mut self,
        launch: u32,
        memory: &dyn GpuMemory,
        hooks: &Hooks,
    ) -> Result<(), Unmodeled> {
        // LAUNCH_DMA: DATA_TRANSFER_TYPE 1:0, FLUSH_ENABLE 2,
        // SEMAPHORE_TYPE 4:3, INTERRUPT_TYPE 6:5, SRC/DST_MEMORY_LAYOUT 7/8
        // (1 pitch), MULTI_LINE_ENABLE 9, REMAP_ENABLE 10, SRC/DST_TYPE
        // 12/13 (1 physical), SEMAPHORE_REDUCTION_ENABLE 19.
        if (launch >> 5) & 3 != 0 {
            return Err(refuse(LAUNCH_DMA, launch, "copy completion interrupt"));
        }
        if launch & (3 << 12) != 0 {
            return Err(refuse(LAUNCH_DMA, launch, "physical addressing"));
        }
        if launch & (1 << 19) != 0 {
            return Err(refuse(LAUNCH_DMA, launch, "semaphore reduction"));
        }
        if launch & 3 != 0 && self.render_enable {
            self.copy(launch, memory)?;
        }
        match (launch >> 3) & 3 {
            0 => Ok(()),
            kind @ (1 | 2) => release_semaphore(
                memory,
                hooks,
                self.address(SET_SEMAPHORE_A),
                self.registers[SET_SEMAPHORE_PAYLOAD],
                kind == 2,
            ),
            _ => Err(refuse(LAUNCH_DMA, launch, "semaphore type 3")),
        }
    }

    fn copy(&self, launch: u32, memory: &dyn GpuMemory) -> Result<(), Unmodeled> {
        let remap = (launch & (1 << 10) != 0).then(|| self.remap());
        let (source_element, destination_element) = match &remap {
            Some(remap) => (remap.source_size(), remap.destination_size()),
            None => (1, 1),
        };
        let length = self.registers[LINE_LENGTH_IN];
        let input = self.address(OFFSET_IN_UPPER);
        let output = self.address(OFFSET_OUT_UPPER);
        let fault =
            |address: u64| Unmodeled(format!("copy: access to unmapped GPU address {address:#x}"));
        if launch & (1 << 9) == 0 {
            // One line, both sides linear.
            let mut source = vec![0; length as usize * source_element];
            memory
                .read(input, &mut source)
                .ok_or_else(|| fault(input))?;
            let destination = match &remap {
                None => source,
                Some(remap) => {
                    let mut destination = vec![0; length as usize * destination_element];
                    memory
                        .read(output, &mut destination)
                        .ok_or_else(|| fault(output))?;
                    for (s, d) in source
                        .chunks(source_element)
                        .zip(destination.chunks_mut(destination_element))
                    {
                        remap.apply(s, d);
                    }
                    destination
                }
            };
            return memory
                .write(output, &destination)
                .ok_or_else(|| fault(output));
        }
        let lines = self.registers[LINE_COUNT];
        let pitch_surface = |address: u64, pitch: u32, element: usize| {
            (
                Surface {
                    address,
                    width: length,
                    height: lines,
                    bytes_per_pixel: element as u32,
                    layout: Layout::Pitch { pitch },
                },
                0,
                0,
            )
        };
        let (source, sx, sy) = if launch & (1 << 7) != 0 {
            pitch_surface(input, self.registers[PITCH_IN], source_element)
        } else {
            self.block_surface(SET_SRC_BLOCK_SIZE, input, source_element)?
        };
        let (destination, dx, dy) = if launch & (1 << 8) != 0 {
            pitch_surface(output, self.registers[PITCH_OUT], destination_element)
        } else {
            self.block_surface(SET_DST_BLOCK_SIZE, output, destination_element)?
        };
        let source_rect = Rect {
            x0: sx,
            y0: sy,
            x1: sx + length,
            y1: sy + lines,
        };
        let destination_rect = Rect {
            x0: dx,
            y0: dy,
            x1: dx + length,
            y1: dy + lines,
        };
        if source_rect.x1 > source.width || source_rect.y1 > source.height {
            return Err(refuse(LAUNCH_DMA, launch, "copy beyond the source surface"));
        }
        if destination_rect.x1 > destination.width || destination_rect.y1 > destination.height {
            return Err(refuse(
                LAUNCH_DMA,
                launch,
                "copy beyond the destination surface",
            ));
        }
        let mut source_window = Window::load(memory, source, source_rect)?;
        let mut elements = vec![0u8; (length * lines) as usize * source_element];
        for band in source_window.bands_mut() {
            for y in band.y0.max(sy)..(band.y0 + 8).min(source_rect.y1) {
                for x in sx..source_rect.x1 {
                    let at = (((y - sy) * length + (x - sx)) as usize) * source_element;
                    elements[at..at + source_element].copy_from_slice(band.pixel(x, y));
                }
            }
        }
        let mut destination_window = Window::load(memory, destination, destination_rect)?;
        for mut band in destination_window.bands_mut() {
            for y in band.y0.max(dy)..(band.y0 + 8).min(destination_rect.y1) {
                for x in dx..destination_rect.x1 {
                    let at = (((y - dy) * length + (x - dx)) as usize) * source_element;
                    let element = &elements[at..at + source_element];
                    let texel = band.pixel_mut(x, y);
                    match &remap {
                        None => texel.copy_from_slice(element),
                        Some(remap) => remap.apply(element, texel),
                    }
                }
            }
        }
        destination_window.store(memory)
    }
}
