//! Inline-to-memory: data carried in the method stream itself, written to
//! memory. The same method block (byte offsets 0x180-0x1FC) exists in the
//! 3D (0xB197), compute (0xB1C0) and inline-to-memory (0xA140) classes
//! [public: NVIDIA open-gpu-doc `classes/inline-to-memory/cla140.h`,
//! `classes/3d/clb197.h`, `classes/compute/clb1c0.h`], so one model serves
//! all three.

use serde::{Deserialize, Serialize};

use super::texture::layout::{BlockLinear, block_linear_offset};
use super::{GpuMemory, Unmodeled};

/// First method (word index) of the shared block.
pub const FIRST_METHOD: u32 = 0x180 / 4;
/// Last method (word index) of the shared block.
pub const LAST_METHOD: u32 = 0x1FC / 4;

const LINE_LENGTH_IN: u32 = 0x180 / 4;
const LINE_COUNT: u32 = 0x184 / 4;
const OFFSET_OUT_UPPER: u32 = 0x188 / 4;
const OFFSET_OUT: u32 = 0x18C / 4;
const PITCH_OUT: u32 = 0x190 / 4;
const SET_DST_BLOCK_SIZE: u32 = 0x194 / 4;
const SET_DST_WIDTH: u32 = 0x198 / 4;
const SET_DST_HEIGHT: u32 = 0x19C / 4;
const SET_DST_DEPTH: u32 = 0x1A0 / 4;
const SET_DST_LAYER: u32 = 0x1A4 / 4;
const SET_DST_ORIGIN_BYTES_X: u32 = 0x1A8 / 4;
const SET_DST_ORIGIN_SAMPLES_Y: u32 = 0x1AC / 4;
const LAUNCH_DMA: u32 = 0x1B0 / 4;
const LOAD_INLINE_DATA: u32 = 0x1B4 / 4;
const SET_I2M_SEMAPHORE_A: u32 = 0x1DC / 4;
const SET_I2M_SEMAPHORE_B: u32 = 0x1E0 / 4;
const SET_I2M_SEMAPHORE_C: u32 = 0x1E4 / 4;
const SET_I2M_SPARE_NOOP00: u32 = 0x1F0 / 4;
const SET_I2M_SPARE_NOOP03: u32 = 0x1FC / 4;

/// A transfer whose `LAUNCH_DMA` has been seen and whose data is arriving.
#[derive(Debug, Serialize, Deserialize)]
struct Transfer {
    launch: u32,
    data: Vec<u8>,
    expected: usize,
}

/// The inline-to-memory state of one class instance, saved whole in
/// machine snapshots, a transfer in flight included.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct InlineToMemory {
    registers: [u32; (LAST_METHOD - FIRST_METHOD + 1) as usize],
    transfer: Option<Transfer>,
}

/// What a completed transfer asks of the caller beyond the memory writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Completion {
    None,
    /// Write the I2M semaphore: (address, payload, four-word structure).
    Semaphore {
        address: u64,
        payload: u32,
        four_words: bool,
    },
}

impl InlineToMemory {
    fn register(&self, method: u32) -> u32 {
        self.registers[(method - FIRST_METHOD) as usize]
    }

    /// Whether a transfer still expects data words.
    pub fn busy(&self) -> bool {
        self.transfer.is_some()
    }

    /// One method of the block. Returns what the caller must do once a
    /// transfer has completed.
    pub fn method(
        &mut self,
        method: u32,
        argument: u32,
        memory: &dyn GpuMemory,
    ) -> Result<Completion, Unmodeled> {
        if self.transfer.is_some() && method != LOAD_INLINE_DATA {
            return Err(Unmodeled(format!(
                "inline-to-memory: method {:#x} while a transfer awaits data",
                method * 4
            )));
        }
        match method {
            LAUNCH_DMA => self.launch(argument, memory),
            LOAD_INLINE_DATA => self.data(argument, memory),
            LINE_LENGTH_IN
            | LINE_COUNT
            | OFFSET_OUT_UPPER
            | OFFSET_OUT
            | PITCH_OUT
            | SET_DST_BLOCK_SIZE
            | SET_DST_WIDTH
            | SET_DST_HEIGHT
            | SET_DST_DEPTH
            | SET_DST_LAYER
            | SET_DST_ORIGIN_BYTES_X
            | SET_DST_ORIGIN_SAMPLES_Y
            | SET_I2M_SEMAPHORE_A
            | SET_I2M_SEMAPHORE_B
            | SET_I2M_SEMAPHORE_C => {
                self.registers[(method - FIRST_METHOD) as usize] = argument;
                Ok(Completion::None)
            }
            // Spare no-operation methods by their class definition.
            SET_I2M_SPARE_NOOP00..=SET_I2M_SPARE_NOOP03 => Ok(Completion::None),
            _ => Err(Unmodeled(format!(
                "inline-to-memory: method {:#x} ({argument:#x})",
                method * 4
            ))),
        }
    }

    fn launch(&mut self, launch: u32, memory: &dyn GpuMemory) -> Result<Completion, Unmodeled> {
        // LAUNCH_DMA fields: 0 layout (1 pitch), 1 reduction enable, 3:2
        // reduction format, 5:4 completion type, 9:8 interrupt type, 12
        // semaphore structure size, 15:13 reduction op.
        if launch & 2 != 0 {
            return Err(Unmodeled(format!(
                "inline-to-memory: LAUNCH_DMA {launch:#x} with reduction"
            )));
        }
        if (launch >> 8) & 3 != 0 {
            return Err(Unmodeled(format!(
                "inline-to-memory: LAUNCH_DMA {launch:#x} requests an interrupt"
            )));
        }
        if (launch >> 4) & 3 == 3 {
            return Err(Unmodeled(format!(
                "inline-to-memory: LAUNCH_DMA {launch:#x} completion type 3"
            )));
        }
        let line_length = self.register(LINE_LENGTH_IN) as usize;
        let line_count = self.register(LINE_COUNT) as usize;
        // Ryujinx (`InlineToMemoryClass.LaunchDma`) starts every line on a
        // word boundary; eden (`engine_upload.cpp`) packs lines. The two
        // agree only when a line is whole words or there is one line.
        if line_count > 1 && !line_length.is_multiple_of(4) {
            return Err(Unmodeled(format!(
                "inline-to-memory: {line_count} lines of {line_length} bytes (line packing unknown)"
            )));
        }
        let expected = line_length * line_count;
        self.transfer = Some(Transfer {
            launch,
            data: Vec::with_capacity(expected.next_multiple_of(4)),
            expected,
        });
        // A transfer of no bytes takes no data words and completes at
        // once: NVN's ClearMemory macro (starter macro 20, upload
        // 0x25ba20) launches with LINE_LENGTH_IN 0 for an empty range and
        // sends no LOAD_INLINE_DATA.
        if expected == 0 {
            return self.finish(memory);
        }
        Ok(Completion::None)
    }

    fn data(&mut self, word: u32, memory: &dyn GpuMemory) -> Result<Completion, Unmodeled> {
        let Some(transfer) = self.transfer.as_mut() else {
            return Err(Unmodeled(format!(
                "inline-to-memory: LOAD_INLINE_DATA {word:#x} without a transfer"
            )));
        };
        transfer.data.extend_from_slice(&word.to_le_bytes());
        if transfer.data.len() < transfer.expected {
            return Ok(Completion::None);
        }
        self.finish(memory)
    }

    /// Writes the completed transfer and reports its semaphore.
    fn finish(&mut self, memory: &dyn GpuMemory) -> Result<Completion, Unmodeled> {
        let transfer = self.transfer.take().expect("finish follows a launch");
        self.write(&transfer, memory)?;
        Ok(if (transfer.launch >> 4) & 3 == 2 {
            Completion::Semaphore {
                address: u64::from(self.register(SET_I2M_SEMAPHORE_A) & 0xFF) << 32
                    | u64::from(self.register(SET_I2M_SEMAPHORE_B)),
                payload: self.register(SET_I2M_SEMAPHORE_C),
                four_words: transfer.launch & (1 << 12) == 0,
            }
        } else {
            Completion::None
        })
    }

    fn write(&self, transfer: &Transfer, memory: &dyn GpuMemory) -> Result<(), Unmodeled> {
        let base = u64::from(self.register(OFFSET_OUT_UPPER) & 0xFF) << 32
            | u64::from(self.register(OFFSET_OUT));
        let line_length = self.register(LINE_LENGTH_IN) as usize;
        let lines = transfer.data[..transfer.expected].chunks(line_length.max(1));
        let fault = |address: u64| {
            Unmodeled(format!(
                "inline-to-memory: write to unmapped GPU address {address:#x}"
            ))
        };
        if transfer.launch & 1 != 0 {
            let pitch = u64::from(self.register(PITCH_OUT));
            for (index, line) in lines.enumerate() {
                let address = base + index as u64 * pitch;
                memory.write(address, line).ok_or_else(|| fault(address))?;
            }
            return Ok(());
        }
        let block_size = self.register(SET_DST_BLOCK_SIZE);
        if block_size & 0xF != 0 {
            return Err(Unmodeled(format!(
                "inline-to-memory: block width {} GOBs",
                1 << (block_size & 0xF)
            )));
        }
        let block = BlockLinear {
            block_height_log2: (block_size >> 4) & 0xF,
            block_depth_log2: (block_size >> 8) & 0xF,
        };
        let width = self.register(SET_DST_WIDTH);
        let height = self.register(SET_DST_HEIGHT);
        let layer = self.register(SET_DST_LAYER);
        // Neither public reimplementation uses SET_DST_LAYER/SET_DST_DEPTH
        // (eden `engine_upload.cpp`, Ryujinx `InlineToMemoryClass`), so
        // whether a layer is a slice or an array element is unknown.
        if layer != 0 {
            return Err(Unmodeled(format!(
                "inline-to-memory: block-linear destination layer {layer}"
            )));
        }
        let x = self.register(SET_DST_ORIGIN_BYTES_X) & 0xF_FFFF;
        let y = self.register(SET_DST_ORIGIN_SAMPLES_Y) & 0xFFFF;
        let offset = |column: usize, row: usize| {
            block_linear_offset(x + column as u32, y + row as u32, 0, width, height, block)
        };
        for (row, line) in lines.enumerate() {
            // Bytes are contiguous in memory in runs inside a GOB; write
            // each run at once.
            let mut run_start = 0;
            while run_start < line.len() {
                let start = offset(run_start, row);
                let mut run_end = run_start + 1;
                while run_end < line.len()
                    && offset(run_end, row) == start + (run_end - run_start) as u64
                {
                    run_end += 1;
                }
                memory
                    .write(base + start, &line[run_start..run_end])
                    .ok_or_else(|| fault(base + start))?;
                run_start = run_end;
            }
        }
        Ok(())
    }
}
