//! Memory layouts of texture images: Tegra block-linear ("swizzled") and
//! pitch-linear.
//!
//! A block-linear surface is tiled in GOBs ("groups of bytes"): 64 bytes
//! wide by 8 rows high, 512 bytes each. GOBs are grouped into blocks one GOB
//! wide, 2^`block_height_log2` GOBs high and 2^`block_depth_log2` GOBs deep.
//! Blocks are laid out row-major across the surface (one row of blocks after
//! another, then one slice of blocks after another); within a block, GOBs
//! are stacked vertically, then in depth; within a GOB, bytes follow the
//! 16-byte-sector pattern of [`gob_offset`].
//! [public: NVIDIA open-gpu-doc `classes/3d/clb197tex.h`
//! `GOBS_PER_BLOCK_HEIGHT`/`GOBS_PER_BLOCK_DEPTH`; blocks are one GOB
//! wide]
//!
//! Sizes here are in bytes and rows of *format blocks* (a 4x4 BC block or
//! an ASTC footprint counts as one "texel" of `bytes_per_block` bytes).

use crate::maxwell::Unmodeled;

/// Width of a GOB in bytes.
pub const GOB_WIDTH: u32 = 64;
/// Height of a GOB in rows.
pub const GOB_HEIGHT: u32 = 8;
/// Size of a GOB in bytes.
pub const GOB_SIZE: u32 = GOB_WIDTH * GOB_HEIGHT;

/// GOB = 64 bytes x 8 rows (512 bytes); blocks are 1 GOB wide,
/// 2^`block_height_log2` GOBs high, 2^`block_depth_log2` GOBs deep.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockLinear {
    pub block_height_log2: u32,
    pub block_depth_log2: u32,
}

impl BlockLinear {
    /// The block a mip level with `rows` rows (of format blocks) and
    /// `depth` slices uses, given the block of the level above it: the
    /// height halves while the level fits in half a block
    /// (`rows <= GOB_HEIGHT << (block_height_log2 - 1)`), the depth halves
    /// while `depth <= 1 << (block_depth_log2 - 1)`.
    pub fn for_level(self, rows: u32, depth: u32) -> BlockLinear {
        let mut block = self;
        while block.block_height_log2 > 0 && rows <= GOB_HEIGHT << (block.block_height_log2 - 1) {
            block.block_height_log2 -= 1;
        }
        while block.block_depth_log2 > 0 && depth <= 1 << (block.block_depth_log2 - 1) {
            block.block_depth_log2 -= 1;
        }
        block
    }

    fn block_size(self) -> u64 {
        u64::from(GOB_SIZE) << (self.block_height_log2 + self.block_depth_log2)
    }
}

/// Byte offset of byte `x` (0-63) of row `y` (0-7) inside a GOB: the GOB is
/// two 32-byte-wide halves, each four 2-row bands of two 16-byte sectors
/// per row.
pub fn gob_offset(x: u32, y: u32) -> u32 {
    ((x & 0x20) << 3) | ((y & 0x6) << 5) | ((x & 0x10) << 1) | ((y & 0x1) << 4) | (x & 0xf)
}

/// Addressing of one block-linear surface: one mip level of one array
/// layer (all its depth slices).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockLinearSurface {
    block: BlockLinear,
    width_gobs: u32,
    /// Bytes of one row of blocks.
    row_size: u64,
    /// Bytes of one slice of blocks (`2^block_depth_log2` slices deep).
    slice_size: u64,
}

impl BlockLinearSurface {
    /// A surface whose rows are `width_bytes` wide (padded to whole GOBs)
    /// and `height` rows high (padded to whole blocks).
    pub fn new(width_bytes: u32, height: u32, block: BlockLinear) -> BlockLinearSurface {
        let width_gobs = width_bytes.div_ceil(GOB_WIDTH);
        let row_size = u64::from(width_gobs) * block.block_size();
        let block_rows = height.div_ceil(GOB_HEIGHT << block.block_height_log2);
        BlockLinearSurface {
            block,
            width_gobs,
            row_size,
            slice_size: u64::from(block_rows) * row_size,
        }
    }

    /// Byte offset of byte column `x_bytes`, row `y`, slice `z`.
    #[inline]
    pub fn offset(&self, x_bytes: u32, y: u32, z: u32) -> u64 {
        let (bh, bd) = (self.block.block_height_log2, self.block.block_depth_log2);
        let gob_y = y / GOB_HEIGHT;
        u64::from(z >> bd) * self.slice_size
            + u64::from(gob_y >> bh) * self.row_size
            + (u64::from(x_bytes / GOB_WIDTH) << (9 + bh + bd))
            + u64::from(((z & ((1 << bd) - 1)) << bh) + (gob_y & ((1 << bh) - 1)))
                * u64::from(GOB_SIZE)
            + u64::from(gob_offset(x_bytes % GOB_WIDTH, y % GOB_HEIGHT))
    }

    /// Bytes of `depth` slices of this surface (padded to whole blocks).
    pub fn size(&self, depth: u32) -> u64 {
        u64::from(depth.div_ceil(1 << self.block.block_depth_log2)) * self.slice_size
    }

    /// The block the surface is laid out in.
    pub fn block(&self) -> BlockLinear {
        self.block
    }

    /// Width of the surface in GOBs.
    pub fn width_gobs(&self) -> u32 {
        self.width_gobs
    }
}

/// Byte offset of byte column `x_bytes`, row `y`, slice `z` in a
/// block-linear surface whose rows are `width_bytes` wide (padded to whole
/// GOBs) and `height` rows high.
pub fn block_linear_offset(
    x_bytes: u32,
    y: u32,
    z: u32,
    width_bytes: u32,
    height: u32,
    block: BlockLinear,
) -> u64 {
    BlockLinearSurface::new(width_bytes, height, block).offset(x_bytes, y, z)
}

/// Size in bytes of one block-linear level/slice set (padded): `depth`
/// slices of a surface `width_bytes` wide and `height` rows high.
pub fn block_linear_size(width_bytes: u32, height: u32, depth: u32, block: BlockLinear) -> u64 {
    BlockLinearSurface::new(width_bytes, height, block).size(depth)
}

/// The dimensions of a texture image as its TIC describes them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageShape {
    /// Level 0 extent in texels.
    pub width: u32,
    pub height: u32,
    /// Depth of a 3D texture (1 otherwise).
    pub depth: u32,
    /// Array layers (cube faces count as layers; 1 for 3D).
    pub layers: u32,
    /// Mip levels allocated.
    pub levels: u32,
    /// Texels per format block (4x4 for BC, the footprint for ASTC).
    pub block_width: u32,
    pub block_height: u32,
    pub bytes_per_block: u32,
}

/// One mip level as laid out in memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MipLevel {
    /// Byte offset of the level from the start of its array layer.
    pub offset: u64,
    /// Bytes of the level (all depth slices).
    pub size: u64,
    /// Extent in texels.
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    /// Extent in format blocks.
    pub width_blocks: u32,
    pub height_blocks: u32,
    pub surface: BlockLinearSurface,
}

/// The layout of a whole block-linear texture: its levels within one array
/// layer and the distance between layers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockLinearImage {
    pub levels: Vec<MipLevel>,
    /// Bytes from one array layer to the next.
    pub layer_stride: u64,
    /// Bytes of all layers.
    pub size: u64,
}

/// Lays out a block-linear texture as the hardware addresses it: levels
/// follow each other within a layer, each with the block [`BlockLinear::for_level`]
/// gives it (starting from the TIC's block at level 0); layers follow each
/// other, each padded to a whole level-0 block.
///
/// The TIC's block describes level 0. If level 0 itself would shrink under
/// the mip rule, what the hardware does is not established, so that
/// case is refused.
pub fn block_linear_image(
    shape: &ImageShape,
    block: BlockLinear,
) -> Result<BlockLinearImage, Unmodeled> {
    let mut levels = Vec::with_capacity(shape.levels as usize);
    let mut level_block = block;
    let mut offset = 0u64;
    for level in 0..shape.levels {
        let width = (shape.width >> level).max(1);
        let height = (shape.height >> level).max(1);
        let depth = (shape.depth >> level).max(1);
        let width_blocks = width.div_ceil(shape.block_width);
        let height_blocks = height.div_ceil(shape.block_height);
        level_block = level_block.for_level(height_blocks, depth);
        if level == 0 && level_block != block {
            return Err(Unmodeled(format!(
                "block-linear TIC block (height 2^{} GOBs, depth 2^{} GOBs) larger than level 0 \
                 ({height_blocks} rows, {depth} slices) needs",
                block.block_height_log2, block.block_depth_log2
            )));
        }
        let surface = BlockLinearSurface::new(
            width_blocks * shape.bytes_per_block,
            height_blocks,
            level_block,
        );
        let size = surface.size(depth);
        levels.push(MipLevel {
            offset,
            size,
            width,
            height,
            depth,
            width_blocks,
            height_blocks,
            surface,
        });
        offset += size;
    }
    let layer_stride = if shape.layers > 1 {
        offset.next_multiple_of(block.block_size())
    } else {
        offset
    };
    Ok(BlockLinearImage {
        levels,
        layer_stride,
        size: layer_stride * u64::from(shape.layers.max(1) - 1) + offset,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference layout built the other way round: walk memory in order,
    /// GOB by GOB, and record which (x byte, row, slice) each byte holds.
    fn reference_map(
        width_bytes: u32,
        height: u32,
        depth: u32,
        block: BlockLinear,
    ) -> Vec<(u32, u32, u32)> {
        let gobs_wide = width_bytes.div_ceil(64);
        let block_h = 1 << block.block_height_log2;
        let block_d = 1 << block.block_depth_log2;
        let block_rows = height.div_ceil(8 * block_h);
        let block_slices = depth.div_ceil(block_d);
        let mut map = Vec::new();
        for bz in 0..block_slices {
            for by in 0..block_rows {
                for bx in 0..gobs_wide {
                    for gz in 0..block_d {
                        for gy in 0..block_h {
                            for i in 0..512u32 {
                                // Sector layout of a GOB, bit by bit.
                                let x = (i & 15) | ((i >> 5) & 1) << 4 | ((i >> 8) & 1) << 5;
                                let y = ((i >> 4) & 1) | ((i >> 6) & 3) << 1;
                                map.push((
                                    bx * 64 + x,
                                    (by * block_h + gy) * 8 + y,
                                    bz * block_d + gz,
                                ));
                            }
                        }
                    }
                }
            }
        }
        map
    }

    #[test]
    fn offsets_match_memory_order_walk() {
        let cases = [
            (64u32, 8u32, 1u32, 0u32, 0u32),
            (100, 20, 1, 1, 0),
            (4 * 37, 50, 1, 2, 0),
            (16 * 9, 33, 1, 3, 0),
            (70, 130, 1, 4, 0),
            (200, 17, 5, 1, 1),
            (128, 16, 9, 0, 2),
            (64 * 3, 40, 3, 2, 1),
        ];
        for (width_bytes, height, depth, bh, bd) in cases {
            let block = BlockLinear {
                block_height_log2: bh,
                block_depth_log2: bd,
            };
            let map = reference_map(width_bytes, height, depth, block);
            assert_eq!(
                map.len() as u64,
                block_linear_size(width_bytes, height, depth, block)
            );
            for (offset, &(x, y, z)) in map.iter().enumerate() {
                let padded_h = height.div_ceil(8 << bh) * (8 << bh);
                if x < width_bytes.div_ceil(64) * 64 && y < padded_h {
                    assert_eq!(
                        block_linear_offset(x, y, z, width_bytes, height, block),
                        offset as u64,
                        "({x},{y},{z}) in {width_bytes}x{height}x{depth} bh{bh} bd{bd}"
                    );
                }
            }
        }
    }

    #[test]
    fn mip_chain_shrinks_blocks_and_aligns_layers() {
        // 64x64 RGBA8 (256 bytes per row), 3 levels, block height 8 GOBs
        // (64 rows), 2 layers.
        let shape = ImageShape {
            width: 64,
            height: 64,
            depth: 1,
            layers: 2,
            levels: 3,
            block_width: 1,
            block_height: 1,
            bytes_per_block: 4,
        };
        let block = BlockLinear {
            block_height_log2: 3,
            block_depth_log2: 0,
        };
        let image = block_linear_image(&shape, block).unwrap();
        // Level 0: 4 GOBs wide x 8 GOBs = 16 KiB.
        assert_eq!(image.levels[0].size, 4 * 8 * 512);
        // Level 1: 32x32: 128 bytes = 2 GOBs wide; 32 rows -> 4-GOB blocks.
        assert_eq!(image.levels[1].offset, 16384);
        assert_eq!(image.levels[1].size, 2 * 4 * 512);
        // Level 2: 16x16: 1 GOB wide; 16 rows -> 2-GOB blocks.
        assert_eq!(image.levels[2].offset, 16384 + 4096);
        assert_eq!(image.levels[2].size, 2 * 512);
        // Layer: 21504 bytes, aligned to the 8-GOB (4 KiB) level-0 block.
        assert_eq!(image.layer_stride, 24576);
        assert_eq!(image.size, 24576 + 21504);
        // Level 2 offsets use its 2-GOB block, not the TIC's 8.
        let level = image.levels[2];
        assert_eq!(level.surface.offset(0, 8, 0), 512);
    }

    #[test]
    fn compressed_level_counts_block_rows() {
        // A 592x100 BC5 texture (16-byte 4x4 blocks) with 4-GOB blocks:
        // 148 x 25 blocks, 2368 bytes = 37 GOBs wide, one row of blocks.
        // The firmware's BNTX gives its image size as 75776 bytes
        // (starter `lyt/common.szs` `LogoNx^t`).
        let shape = ImageShape {
            width: 592,
            height: 100,
            depth: 1,
            layers: 1,
            levels: 1,
            block_width: 4,
            block_height: 4,
            bytes_per_block: 16,
        };
        let block = BlockLinear {
            block_height_log2: 2,
            block_depth_log2: 0,
        };
        assert_eq!(block_linear_image(&shape, block).unwrap().size, 75776);
        let oversized = BlockLinear {
            block_height_log2: 4,
            block_depth_log2: 0,
        };
        assert!(block_linear_image(&shape, oversized).is_err());
    }
}
