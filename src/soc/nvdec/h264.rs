//! H.264 on NVDEC: the picture setup structure the driver fills for each
//! picture, and the bitstream rebuilt from it for a decoder that takes a
//! whole stream.
//!
//! # Picture setup
//!
//! `nvdec_h264_pic_s` in NVIDIA's public `open-gpu-doc`
//! `classes/video/nvdec_drv.h` (MIT). Its first member, the legacy
//! `nvdec_pass2_otf_s` (three 16-byte keys and a word of flags), is 0x34
//! bytes long (the header's "0x10 bytes" comment is stale), which puts
//! `stream_len` at 0x48 and the rest at the offsets below, each listed at
//! the field it decodes. The browser's first pictures of the parental
//! controls video fit them: `stream_len` the bitstream buffer's length,
//! 80×45 macroblocks for the 1280×720 frame, pitches 1280, scaling lists
//! all 16, `frame_num` counting up from the IDR picture and the DPB
//! entries naming the pictures decoded before.
//!
//! # Bitstream
//!
//! The bitstream buffer holds the picture's slice NAL units, each after a
//! three-byte start code (ITU-T H.264 Annex B), followed by two end of
//! stream NAL units (type 11) padded with zeros — what the browser's first
//! pictures carry; later ones also carry filler data (type 12), which
//! decoders discard (§7.4.2.7). The sequence and picture parameter sets
//! are not in it:
//! NVDEC takes their content from the picture setup. They are written
//! again here (ITU-T H.264 §7.3.2.1.1 and §7.3.2.2) from it. Fields the
//! picture setup does not carry get these values:
//!
//! * `profile_idc` 100 (High), no constraint flags, `level_idc` 51: the
//!   profile whose syntax every field set here may use, the level whose
//!   DPB holds 16 frames at 1280×720 \[assumption: the original profile
//!   and level only bound the stream\].
//! * `max_num_ref_frames`: given by the caller ([`sequence_parameter_set`]).
//! * `gaps_in_frame_num_value_allowed_flag` 0, no frame cropping, no VUI.
//! * `pic_init_qs_minus26` 0: used only by SP and SI slices, which the High
//!   profile does not have.
//! * `num_slice_groups_minus1` 0: slice groups are a Baseline/Extended
//!   tool the picture setup has no field for.
//! * The scaling lists are the picture's own, in the picture parameter set
//!   (none when all are flat 16, which is what none means).
//!
//! The picture parameter set gets the ID the slices name.

/// Bytes of `nvdec_h264_pic_s`.
pub const PICTURE_INFO_SIZE: usize = 0x2FC;

/// One `nvdec_dpb_entry_s` (16 bytes from 0xC0).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DpbEntry {
    /// Bits 0-6: the picture's surface index.
    pub index: u32,
    /// Bits 12-13: top and bottom field used for reference.
    pub state: u32,
    /// Bit 14.
    pub long_term: bool,
    /// Bit 15: a frame the decoding process infers for a frame_num gap.
    pub not_existing: bool,
}

/// The fields of `nvdec_h264_pic_s` the model uses (byte offsets).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PictureInfo {
    /// 0x30 bit 0 `enable_encryption` (of `encryption_params`).
    pub encrypted: bool,
    /// 0x48: bytes of the bitstream buffer to decode.
    pub stream_len: u32,
    /// 0x4C.
    pub slice_count: u32,
    /// 0x58.
    pub log2_max_pic_order_cnt_lsb_minus4: u32,
    /// 0x5C.
    pub delta_pic_order_always_zero: bool,
    /// 0x60.
    pub frame_mbs_only: bool,
    /// 0x64 `PicWidthInMbs`.
    pub width_mbs: u32,
    /// 0x68 `FrameHeightInMbs`.
    pub height_mbs: u32,
    /// 0x6C bits 0-1 `tileFormat` (0 TBL, 1 KBL, 2 16×16 tiles).
    pub tile_format: u32,
    /// 0x6C bits 2-4 `gob_height` (0: blocks 2 GOBs high, 1: 4, …).
    pub gob_height: u32,
    /// 0x70.
    pub entropy_coding_mode: bool,
    /// 0x74 (`bottom_field_pic_order_in_frame_present_flag`).
    pub pic_order_present: bool,
    /// 0x78.
    pub num_ref_idx_l0_active_minus1: u32,
    /// 0x7C.
    pub num_ref_idx_l1_active_minus1: u32,
    /// 0x80.
    pub deblocking_filter_control_present: bool,
    /// 0x84.
    pub redundant_pic_cnt_present: bool,
    /// 0x88.
    pub transform_8x8_mode: bool,
    /// 0x8C, bytes.
    pub pitch_luma: u32,
    /// 0x90, bytes.
    pub pitch_chroma: u32,
    /// 0x9C, 256-byte units from the picture's luma surface.
    pub luma_frame_offset: u32,
    /// 0xA8, 256-byte units from the picture's chroma surface.
    pub chroma_frame_offset: u32,
    /// 0xB0 bit 0.
    pub mbaff_frame: bool,
    /// 0xB0 bit 1.
    pub direct_8x8_inference: bool,
    /// 0xB0 bit 2.
    pub weighted_pred: bool,
    /// 0xB0 bit 3.
    pub constrained_intra_pred: bool,
    /// 0xB0 bit 5.
    pub field_pic: bool,
    /// 0xB0 bits 8-11.
    pub log2_max_frame_num_minus4: u32,
    /// 0xB0 bits 12-13.
    pub chroma_format_idc: u32,
    /// 0xB0 bits 14-15.
    pub pic_order_cnt_type: u32,
    /// 0xB0 bits 16-21, signed.
    pub pic_init_qp_minus26: i32,
    /// 0xB0 bits 22-26, signed.
    pub chroma_qp_index_offset: i32,
    /// 0xB0 bits 27-31, signed.
    pub second_chroma_qp_index_offset: i32,
    /// 0xB4 bits 0-1.
    pub weighted_bipred_idc: u32,
    /// 0xB4 bits 2-8 `CurrPicIdx`: the surface the picture is decoded to.
    pub curr_pic_idx: u32,
    /// 0xB4 bits 14-29.
    pub frame_num: u32,
    /// 0xB4 bit 31 (0 NV12, 1 NV24).
    pub output_memory_layout: u32,
    /// 0xC0, 16 entries.
    pub dpb: [DpbEntry; 16],
    /// 0x1C0 `WeightScale[6][4][4]`, raster order.
    pub scaling_4x4: [[u8; 16]; 6],
    /// 0x220 `WeightScale8x8[2][8][8]`, raster order.
    pub scaling_8x8: [[u8; 64]; 2],
    /// 0x2D0 bit 1 `qpprime_y_zero_transform_bypass_flag`.
    pub transform_bypass: bool,
    /// 0x2D4 bit 0 `enableTFOutput` (of `displayPara`): a second, display
    /// copy of the picture.
    pub display_output: bool,
}

fn word(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}

fn bits(word: u32, low: u32, width: u32) -> u32 {
    (word >> low) & ((1 << width) - 1)
}

fn signed(word: u32, low: u32, width: u32) -> i32 {
    ((word << (32 - low - width)) as i32) >> (32 - width)
}

impl PictureInfo {
    pub fn parse(bytes: &[u8; PICTURE_INFO_SIZE]) -> Self {
        let flag = |at| word(bytes, at) != 0;
        let (b0, b4) = (word(bytes, 0xB0), word(bytes, 0xB4));
        let format = word(bytes, 0x6C);
        Self {
            encrypted: word(bytes, 0x30) & 1 != 0,
            stream_len: word(bytes, 0x48),
            slice_count: word(bytes, 0x4C),
            log2_max_pic_order_cnt_lsb_minus4: word(bytes, 0x58),
            delta_pic_order_always_zero: flag(0x5C),
            frame_mbs_only: flag(0x60),
            width_mbs: word(bytes, 0x64),
            height_mbs: word(bytes, 0x68),
            tile_format: bits(format, 0, 2),
            gob_height: bits(format, 2, 3),
            entropy_coding_mode: flag(0x70),
            pic_order_present: flag(0x74),
            num_ref_idx_l0_active_minus1: word(bytes, 0x78),
            num_ref_idx_l1_active_minus1: word(bytes, 0x7C),
            deblocking_filter_control_present: flag(0x80),
            redundant_pic_cnt_present: flag(0x84),
            transform_8x8_mode: flag(0x88),
            pitch_luma: word(bytes, 0x8C),
            pitch_chroma: word(bytes, 0x90),
            luma_frame_offset: word(bytes, 0x9C),
            chroma_frame_offset: word(bytes, 0xA8),
            mbaff_frame: bits(b0, 0, 1) != 0,
            direct_8x8_inference: bits(b0, 1, 1) != 0,
            weighted_pred: bits(b0, 2, 1) != 0,
            constrained_intra_pred: bits(b0, 3, 1) != 0,
            field_pic: bits(b0, 5, 1) != 0,
            log2_max_frame_num_minus4: bits(b0, 8, 4),
            chroma_format_idc: bits(b0, 12, 2),
            pic_order_cnt_type: bits(b0, 14, 2),
            pic_init_qp_minus26: signed(b0, 16, 6),
            chroma_qp_index_offset: signed(b0, 22, 5),
            second_chroma_qp_index_offset: signed(b0, 27, 5),
            weighted_bipred_idc: bits(b4, 0, 2),
            curr_pic_idx: bits(b4, 2, 7),
            frame_num: bits(b4, 14, 16),
            output_memory_layout: bits(b4, 31, 1),
            dpb: std::array::from_fn(|i| {
                let entry = word(bytes, 0xC0 + 16 * i);
                DpbEntry {
                    index: bits(entry, 0, 7),
                    state: bits(entry, 12, 2),
                    long_term: bits(entry, 14, 1) != 0,
                    not_existing: bits(entry, 15, 1) != 0,
                }
            }),
            scaling_4x4: std::array::from_fn(|i| {
                bytes[0x1C0 + 16 * i..0x1D0 + 16 * i].try_into().unwrap()
            }),
            scaling_8x8: std::array::from_fn(|i| {
                bytes[0x220 + 64 * i..0x260 + 64 * i].try_into().unwrap()
            }),
            transform_bypass: bits(word(bytes, 0x2D0), 1, 1) != 0,
            display_output: bits(word(bytes, 0x2D4), 0, 1) != 0,
        }
    }

    /// Refuses what is not modeled: encrypted streams, fields and MBAFF,
    /// other than 4:2:0 chroma, picture order count type 1 (its cycle of
    /// offsets is not in the picture setup), lossless transform bypass,
    /// redundant pictures, NV24 output, layouts other than TBL (block
    /// linear) with blocks up to 32 GOBs high, and the display copy.
    pub fn check(&self) -> Result<(), String> {
        let refusal = if self.encrypted {
            "an encrypted stream"
        } else if !self.frame_mbs_only || self.mbaff_frame || self.field_pic {
            "field or MBAFF coding"
        } else if self.chroma_format_idc != 1 {
            "chroma other than 4:2:0"
        } else if self.pic_order_cnt_type == 1 {
            "picture order count type 1"
        } else if self.transform_bypass {
            "transform bypass"
        } else if self.redundant_pic_cnt_present {
            "redundant pictures"
        } else if self.output_memory_layout != 0 {
            "NV24 output"
        } else if self.tile_format != 0 || self.gob_height > 4 {
            "an output layout other than block linear"
        } else if self.display_output {
            "the display copy (enableTFOutput)"
        } else if self.width_mbs == 0 || self.height_mbs == 0 {
            "an empty picture"
        } else {
            return Ok(());
        };
        Err(format!("H.264: {refusal} is not modeled"))
    }

    /// Frames in the DPB marked as used for reference.
    pub fn reference_frames(&self) -> u32 {
        self.dpb.iter().filter(|entry| entry.state != 0).count() as u32
    }

    fn flat(&self) -> bool {
        self.scaling_4x4.iter().flatten().all(|&v| v == 16)
            && self.scaling_8x8.iter().flatten().all(|&v| v == 16)
    }
}

/// Exp-Golomb and fixed-width bit writer for RBSPs.
#[derive(Default)]
struct BitWriter {
    bytes: Vec<u8>,
    bits: u32,
}

impl BitWriter {
    fn bit(&mut self, bit: bool) {
        if self.bits.is_multiple_of(8) {
            self.bytes.push(0);
        }
        if bit {
            *self.bytes.last_mut().unwrap() |= 0x80 >> (self.bits % 8);
        }
        self.bits += 1;
    }

    fn u(&mut self, width: u32, value: u32) {
        for i in (0..width).rev() {
            self.bit(value >> i & 1 != 0);
        }
    }

    fn flag(&mut self, value: bool) {
        self.bit(value);
    }

    fn ue(&mut self, value: u32) {
        let code = u64::from(value) + 1;
        let width = 64 - code.leading_zeros();
        self.u(width - 1, 0);
        for i in (0..width).rev() {
            self.bit(code >> i & 1 != 0);
        }
    }

    fn se(&mut self, value: i32) {
        let mapped = if value > 0 {
            2 * value as u32 - 1
        } else {
            2 * value.unsigned_abs()
        };
        self.ue(mapped);
    }

    /// `rbsp_trailing_bits`.
    fn finish(mut self) -> Vec<u8> {
        self.bit(true);
        self.bytes
    }
}

/// A NAL unit of `header` and `rbsp`, with emulation prevention bytes
/// (§7.4.1: 0x03 after two zeros before any byte 0-3).
fn nal_unit(header: u8, rbsp: &[u8]) -> Vec<u8> {
    let mut nal = vec![header];
    let mut zeros = 0;
    for &byte in rbsp {
        if zeros == 2 && byte <= 3 {
            nal.push(3);
            zeros = 0;
        }
        nal.push(byte);
        zeros = if byte == 0 { zeros + 1 } else { 0 };
    }
    nal
}

/// The sequence parameter set (ID 0) of `info`'s stream, NAL unit with
/// header, no start code.
pub fn sequence_parameter_set(info: &PictureInfo, max_num_ref_frames: u32) -> Vec<u8> {
    let mut w = BitWriter::default();
    w.u(8, 100); // profile_idc: High
    w.u(8, 0); // constraint_set0..5_flag, reserved_zero_2bits
    w.u(8, 51); // level_idc
    w.ue(0); // seq_parameter_set_id
    w.ue(info.chroma_format_idc);
    w.ue(0); // bit_depth_luma_minus8
    w.ue(0); // bit_depth_chroma_minus8
    w.flag(info.transform_bypass);
    w.flag(false); // seq_scaling_matrix_present_flag
    w.ue(info.log2_max_frame_num_minus4);
    w.ue(info.pic_order_cnt_type);
    if info.pic_order_cnt_type == 0 {
        w.ue(info.log2_max_pic_order_cnt_lsb_minus4);
    }
    w.ue(max_num_ref_frames);
    w.flag(false); // gaps_in_frame_num_value_allowed_flag
    w.ue(info.width_mbs - 1);
    let map_units = if info.frame_mbs_only {
        info.height_mbs
    } else {
        info.height_mbs / 2
    };
    w.ue(map_units - 1);
    w.flag(info.frame_mbs_only);
    if !info.frame_mbs_only {
        w.flag(info.mbaff_frame);
    }
    w.flag(info.direct_8x8_inference);
    w.flag(false); // frame_cropping_flag
    w.flag(false); // vui_parameters_present_flag
    nal_unit(0x67, &w.finish())
}

/// Raster index of each coefficient in 4×4 frame zig-zag order (§8.5.6).
const ZIGZAG_4X4: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];
/// Raster index of each coefficient in 8×8 frame zig-zag order (§8.5.7).
const ZIGZAG_8X8: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// `scaling_list()` (§7.3.2.1.1.1) giving `list` (raster order) exactly:
/// each value as the delta from the one before in zig-zag order.
fn scaling_list(w: &mut BitWriter, list: &[u8], zigzag: &[usize]) {
    let mut last = 8i32;
    for &raster in zigzag {
        let value = i32::from(list[raster]);
        let delta = (value - last + 128).rem_euclid(256) - 128;
        w.se(delta);
        last = value;
    }
}

/// The picture parameter set `id` of `info`'s picture (sequence parameter
/// set 0), NAL unit with header, no start code.
pub fn picture_parameter_set(info: &PictureInfo, id: u32) -> Vec<u8> {
    let mut w = BitWriter::default();
    w.ue(id);
    w.ue(0); // seq_parameter_set_id
    w.flag(info.entropy_coding_mode);
    w.flag(info.pic_order_present);
    w.ue(0); // num_slice_groups_minus1
    w.ue(info.num_ref_idx_l0_active_minus1);
    w.ue(info.num_ref_idx_l1_active_minus1);
    w.flag(info.weighted_pred);
    w.u(2, info.weighted_bipred_idc);
    w.se(info.pic_init_qp_minus26);
    w.se(0); // pic_init_qs_minus26
    w.se(info.chroma_qp_index_offset);
    w.flag(info.deblocking_filter_control_present);
    w.flag(info.constrained_intra_pred);
    w.flag(info.redundant_pic_cnt_present);
    w.flag(info.transform_8x8_mode);
    let matrices = !info.flat();
    w.flag(matrices);
    if matrices {
        for list in &info.scaling_4x4 {
            w.flag(true); // pic_scaling_list_present_flag
            scaling_list(&mut w, list, &ZIGZAG_4X4);
        }
        if info.transform_8x8_mode {
            for list in &info.scaling_8x8 {
                w.flag(true);
                scaling_list(&mut w, list, &ZIGZAG_8X8);
            }
        }
    }
    w.se(info.second_chroma_qp_index_offset);
    nal_unit(0x68, &w.finish())
}

/// NAL unit type of a coded slice of an IDR picture.
pub const IDR_SLICE: u8 = 5;
const SLICE: u8 = 1;
const END_OF_STREAM: u8 = 11;
const FILLER_DATA: u8 = 12;

/// The slice NAL units of a bitstream buffer (Annex B byte stream), in
/// order, without start codes. End of stream NAL units (the driver's
/// terminator) and filler data are dropped; any other kind is refused.
pub fn slices(stream: &[u8]) -> Result<Vec<&[u8]>, String> {
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 3 <= stream.len() {
        if stream[i..i + 3] == [0, 0, 1] {
            starts.push(i + 3);
            i += 3;
        } else {
            i += 1;
        }
    }
    let mut units = Vec::new();
    for (n, &start) in starts.iter().enumerate() {
        let end = starts.get(n + 1).map_or(stream.len(), |next| next - 3);
        // Zeros before the next start code are trailing_zero_8bits.
        let mut unit = &stream[start..end];
        while let [rest @ .., 0] = unit {
            unit = rest;
        }
        let Some(&header) = unit.first() else {
            return Err("an empty NAL unit".into());
        };
        match header & 0x1F {
            SLICE | IDR_SLICE => units.push(unit),
            END_OF_STREAM | FILLER_DATA => {}
            other => return Err(format!("NAL unit type {other} in the bitstream")),
        }
    }
    if units.is_empty() {
        return Err("no slice in the bitstream".into());
    }
    Ok(units)
}

/// Exp-Golomb reader over an RBSP prefix.
struct BitReader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl BitReader<'_> {
    fn bit(&mut self) -> Option<u32> {
        let byte = self.bytes.get(self.at / 8)?;
        let bit = u32::from(byte >> (7 - self.at % 8) & 1);
        self.at += 1;
        Some(bit)
    }

    fn ue(&mut self) -> Option<u32> {
        let mut zeros = 0;
        while self.bit()? == 0 {
            zeros += 1;
            if zeros > 31 {
                return None;
            }
        }
        let mut value = 1u64;
        for _ in 0..zeros {
            value = value << 1 | u64::from(self.bit()?);
        }
        u32::try_from(value - 1).ok()
    }
}

/// `pic_parameter_set_id` of a slice NAL unit (third field of its header,
/// §7.3.3).
pub fn slice_pps_id(slice: &[u8]) -> Option<u32> {
    // Remove emulation prevention from enough of the header.
    let mut rbsp = Vec::new();
    let mut zeros = 0;
    for &byte in slice.iter().skip(1).take(32) {
        if zeros == 2 && byte == 3 {
            zeros = 0;
            continue;
        }
        rbsp.push(byte);
        zeros = if byte == 0 { zeros + 1 } else { 0 };
    }
    let mut reader = BitReader {
        bytes: &rbsp,
        at: 0,
    };
    reader.ue()?; // first_mb_in_slice
    reader.ue()?; // slice_type
    reader.ue()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The picture setup of the parental controls video's second picture
    /// (a P picture after the IDR one), its non-zero words as the browser
    /// wrote them.
    fn second_picture() -> [u8; PICTURE_INFO_SIZE] {
        let mut bytes = [0; PICTURE_INFO_SIZE];
        let words = [
            (0x48, 0x6bu32),
            (0x4c, 1),
            (0x50, 0x2100),
            (0x58, 3),
            (0x60, 1),
            (0x64, 0x50),
            (0x68, 0x2d),
            (0x70, 1),
            (0x74, 1),
            (0x78, 1),
            (0x80, 1),
            (0x88, 1),
            (0x8c, 0x500),
            (0x90, 0x500),
            (0xac, 0xb2),
            (0xb0, 0x1216),
            (0xb4, 0x4204),
            (0xb8, 2),
            (0xbc, 2),
            (0xc0, 0x223000),
            (0x2d0, 1),
        ];
        for (at, value) in words {
            bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
        }
        bytes[0x1C0..0x2A0].fill(16);
        bytes
    }

    #[test]
    fn picture_setup_fields_decode_at_their_offsets() {
        let info = PictureInfo::parse(&second_picture());
        assert_eq!(info.stream_len, 0x6b);
        assert_eq!(info.slice_count, 1);
        assert_eq!((info.width_mbs, info.height_mbs), (80, 45));
        assert_eq!(info.log2_max_pic_order_cnt_lsb_minus4, 3);
        assert!(info.frame_mbs_only && info.entropy_coding_mode && info.pic_order_present);
        assert_eq!(info.num_ref_idx_l0_active_minus1, 1);
        assert!(info.deblocking_filter_control_present && info.transform_8x8_mode);
        assert_eq!((info.pitch_luma, info.pitch_chroma), (1280, 1280));
        // 0x1216: direct 8x8 inference, weighted prediction, a reference
        // picture, log2_max_frame_num_minus4 2, 4:2:0, POC type 0, QP 26.
        assert!(info.direct_8x8_inference && info.weighted_pred);
        assert!(!info.mbaff_frame && !info.constrained_intra_pred && !info.field_pic);
        assert_eq!(info.log2_max_frame_num_minus4, 2);
        assert_eq!(info.chroma_format_idc, 1);
        assert_eq!(info.pic_order_cnt_type, 0);
        assert_eq!(info.pic_init_qp_minus26, 0);
        // 0x4204: surface 1, frame_num 1.
        assert_eq!((info.curr_pic_idx, info.frame_num), (1, 1));
        assert_eq!(info.weighted_bipred_idc, 0);
        assert_eq!(
            info.dpb[0],
            DpbEntry {
                index: 0,
                state: 3,
                long_term: false,
                not_existing: false
            }
        );
        assert_eq!(info.reference_frames(), 1);
        assert!(!info.transform_bypass);
        assert_eq!(info.check(), Ok(()));
        assert!(info.flat());
    }

    #[test]
    fn signed_fields_and_refusals() {
        let mut bytes = second_picture();
        // pic_init_qp_minus26 -3, chroma_qp_index_offset -2,
        // second_chroma_qp_index_offset 5.
        let b0 = 0x1216 | (0x3D << 16) | (0x1E << 22) | (5 << 27);
        bytes[0xB0..0xB4].copy_from_slice(&(b0 as u32).to_le_bytes());
        let info = PictureInfo::parse(&bytes);
        assert_eq!(info.pic_init_qp_minus26, -3);
        assert_eq!(info.chroma_qp_index_offset, -2);
        assert_eq!(info.second_chroma_qp_index_offset, 5);
        bytes[0x60] = 0;
        assert!(PictureInfo::parse(&bytes).check().is_err());
    }

    /// Reads the whole RBSP of a NAL unit back, for the tests.
    fn rbsp(nal: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut zeros = 0;
        for &byte in &nal[1..] {
            if zeros == 2 && byte == 3 {
                zeros = 0;
                continue;
            }
            out.push(byte);
            zeros = if byte == 0 { zeros + 1 } else { 0 };
        }
        out
    }

    #[test]
    fn parameter_sets_carry_the_picture_setup() {
        let info = PictureInfo::parse(&second_picture());
        let sps = sequence_parameter_set(&info, 2);
        // High, level 5.1; then sps id 0, 4:2:0, bit depths 0 and 0, no
        // bypass, no matrices, log2_max_frame_num_minus4 2, POC type 0,
        // log2_max_pic_order_cnt_lsb_minus4 3, 2 reference frames, no gaps,
        // 80×45 macroblocks, frame macroblocks only, direct 8x8 inference,
        // no cropping, no VUI, stop bit.
        assert_eq!(&sps[..4], &[0x67, 100, 0, 51]);
        let mut reader = BitReader {
            bytes: &rbsp(&sps)[3..],
            at: 0,
        };
        let mut fields = Vec::new();
        for _ in 0..4 {
            fields.push(reader.ue().unwrap());
        }
        fields.push(reader.bit().unwrap());
        fields.push(reader.bit().unwrap());
        for _ in 0..4 {
            fields.push(reader.ue().unwrap());
        }
        fields.push(reader.bit().unwrap());
        fields.push(reader.ue().unwrap());
        fields.push(reader.ue().unwrap());
        for _ in 0..5 {
            fields.push(reader.bit().unwrap());
        }
        assert_eq!(
            fields,
            [0, 1, 0, 0, 0, 0, 2, 0, 3, 2, 0, 79, 44, 1, 1, 0, 0, 1]
        );

        let pps = picture_parameter_set(&info, 0);
        assert_eq!(pps[0], 0x68);
        let bytes = rbsp(&pps);
        let mut reader = BitReader {
            bytes: &bytes,
            at: 0,
        };
        let mut fields = vec![reader.ue().unwrap(), reader.ue().unwrap()];
        fields.push(reader.bit().unwrap()); // CABAC
        fields.push(reader.bit().unwrap()); // bottom field POC present
        fields.extend([
            reader.ue().unwrap(),
            reader.ue().unwrap(),
            reader.ue().unwrap(),
        ]);
        fields.push(reader.bit().unwrap()); // weighted_pred
        fields.push(reader.bit().unwrap() << 1 | reader.bit().unwrap());
        fields.extend([
            reader.ue().unwrap(),
            reader.ue().unwrap(),
            reader.ue().unwrap(),
        ]);
        for _ in 0..5 {
            fields.push(reader.bit().unwrap());
        }
        fields.push(reader.ue().unwrap());
        fields.push(reader.bit().unwrap());
        assert_eq!(
            fields,
            [0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1]
        );
    }

    #[test]
    fn scaling_lists_round_trip_in_zigzag_order() {
        let mut bytes = second_picture();
        for (i, value) in bytes[0x1C0..0x2A0].iter_mut().enumerate() {
            *value = 4 + (i * 7 % 200) as u8;
        }
        let info = PictureInfo::parse(&bytes);
        let pps = picture_parameter_set(&info, 3);
        let bytes = rbsp(&pps);
        let mut reader = BitReader {
            bytes: &bytes,
            at: 0,
        };
        assert_eq!(reader.ue(), Some(3));
        // Skip to the scaling matrix flag: sps id, 2 flags, 3 ue, 3 bits,
        // 3 se (all ue 0 / small), 4 flags.
        reader.ue().unwrap();
        reader.at += 2;
        for _ in 0..3 {
            reader.ue().unwrap();
        }
        reader.at += 3;
        for _ in 0..3 {
            reader.ue().unwrap();
        }
        reader.at += 4;
        assert_eq!(reader.bit(), Some(1));
        let se = |reader: &mut BitReader| {
            let v = reader.ue().unwrap() as i32;
            if v % 2 == 1 { (v + 1) / 2 } else { -(v / 2) }
        };
        for (list, zigzag) in info
            .scaling_4x4
            .iter()
            .map(|l| (&l[..], &ZIGZAG_4X4[..]))
            .chain(info.scaling_8x8.iter().map(|l| (&l[..], &ZIGZAG_8X8[..])))
        {
            assert_eq!(reader.bit(), Some(1));
            let mut last = 8;
            for &raster in zigzag {
                last = (last + se(&mut reader) + 256) % 256;
                assert_eq!(last, i32::from(list[raster]));
            }
        }
    }

    #[test]
    fn emulation_prevention_is_inserted() {
        assert_eq!(
            nal_unit(0x68, &[0, 0, 0, 0, 1, 0, 0, 4]),
            [0x68, 0, 0, 3, 0, 0, 3, 1, 0, 0, 4]
        );
    }

    #[test]
    fn slices_are_cut_at_start_codes_and_the_terminator_dropped() {
        // The second picture's buffer: one P slice (header 0x21), then the
        // driver's two end of stream units with their zero padding.
        let slice = [0x21, 0x9a, 0x08, 0x2e, 0xc4, 0xbf, 0, 0, 3, 0, 0x8b, 0x80];
        let mut stream = vec![0, 0, 1];
        stream.extend(slice);
        // Filler data (type 12), as later pictures carry, is dropped too.
        stream.extend([0, 0, 1, 0x0c, 0xff, 0xff, 0x80]);
        stream.extend([0, 0, 1, 0x0b, 0, 0, 0, 0, 0, 0, 1, 0x0b, 0, 0, 0, 0]);
        assert_eq!(slices(&stream).unwrap(), [&slice[..]]);
        // first_mb_in_slice 0 (1), slice_type 5 (00110), pps 0 (1): 0x9a.
        assert_eq!(slice_pps_id(&slice), Some(0));
        assert!(slices(&[0, 0, 1, 0x67, 1]).is_err());
        assert!(slices(&[0, 0, 1, 0x0b]).is_err());
    }
}
