//! H.264 syntax for the encoder: parameter sets and slice headers read
//! from the host encoder's stream, and slice headers written as NVENC
//! writes them, for the driver's own parameter sets (ITU-T H.264 7.3.2.1,
//! 7.3.2.2, 7.3.3).
//!
//! The driver writes the sequence and picture parameter sets itself and
//! tells NVENC what of them the slices depend on (`nvenc_h264_sps_data_s`,
//! `nvenc_h264_pps_data_s` of the picture setup); NVENC writes the slices.
//! Here the host encoder's slices are taken apart and their headers
//! written again under the driver's sets: its frame number, picture order
//! count and IDR picture ID, its parameter set ID, slice QP relative to its
//! initial QP, reference counts against its defaults. The slice data is
//! kept bit for bit, so what it depends on must agree between the two
//! (`check`): entropy coder, 8x8 transform, constrained intra prediction,
//! chroma QP offsets, no weighted prediction, frames only, 4:2:0. Slices
//! must reference one picture, the latest (see `videotoolbox`): what the
//! driver's decoder holds as the first of list 0 whatever it keeps.

/// Reads bits of an RBSP, most significant first.
pub struct Reader<'a> {
    bytes: &'a [u8],
    bit: usize,
}

impl<'a> Reader<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Reader { bytes, bit: 0 }
    }

    pub fn position(&self) -> usize {
        self.bit
    }

    pub fn bit(&mut self) -> Result<u32, String> {
        let byte = self
            .bytes
            .get(self.bit / 8)
            .ok_or("syntax runs past the end of the NAL unit")?;
        let value = byte >> (7 - self.bit % 8) & 1;
        self.bit += 1;
        Ok(value.into())
    }

    pub fn flag(&mut self) -> Result<bool, String> {
        Ok(self.bit()? == 1)
    }

    pub fn u(&mut self, bits: u32) -> Result<u32, String> {
        let mut value = 0;
        for _ in 0..bits {
            value = value << 1 | self.bit()?;
        }
        Ok(value)
    }

    pub fn ue(&mut self) -> Result<u32, String> {
        let mut zeros = 0;
        while self.bit()? == 0 {
            zeros += 1;
            if zeros > 31 {
                return Err("Exp-Golomb code longer than 32 bits".into());
            }
        }
        Ok(((1u64 << zeros) - 1 + u64::from(self.u(zeros)?)) as u32)
    }

    pub fn se(&mut self) -> Result<i32, String> {
        let code = self.ue()?;
        Ok(if code % 2 == 1 {
            code.div_ceil(2) as i32
        } else {
            -((code / 2) as i32)
        })
    }

    /// Whether syntax remains before the RBSP's stop bit.
    pub fn more_data(&self) -> bool {
        stop_bit(self.bytes).is_some_and(|stop| self.bit < stop)
    }
}

/// The position of the RBSP's stop bit: its last bit set.
pub fn stop_bit(rbsp: &[u8]) -> Option<usize> {
    let last = rbsp.iter().rposition(|&byte| byte != 0)?;
    Some(last * 8 + 7 - rbsp[last].trailing_zeros() as usize)
}

/// Writes bits, most significant first.
#[derive(Default)]
pub struct Writer {
    bytes: Vec<u8>,
    bits: usize,
}

impl Writer {
    pub fn bits(&self) -> usize {
        self.bits
    }

    pub fn bit(&mut self, value: u32) {
        if self.bits.is_multiple_of(8) {
            self.bytes.push(0);
        }
        if value & 1 != 0 {
            *self.bytes.last_mut().unwrap() |= 0x80 >> (self.bits % 8);
        }
        self.bits += 1;
    }

    pub fn u(&mut self, bits: u32, value: u32) {
        for index in (0..bits).rev() {
            self.bit(value >> index);
        }
    }

    pub fn flag(&mut self, value: bool) {
        self.bit(value.into());
    }

    pub fn ue(&mut self, value: u32) {
        let code = u64::from(value) + 1;
        let length = 64 - code.leading_zeros();
        self.u(length - 1, 0);
        for index in (0..length).rev() {
            self.bit((code >> index) as u32);
        }
    }

    pub fn se(&mut self, value: i32) {
        self.ue(if value > 0 {
            (2 * value - 1) as u32
        } else {
            (-2 * i64::from(value)) as u32
        });
    }

    /// Copies bits `from..to` of `source`.
    pub fn copy(&mut self, source: &[u8], from: usize, to: usize) {
        let mut at = from;
        if self.bits.is_multiple_of(8) && at.is_multiple_of(8) {
            let whole = (to - at) / 8;
            self.bytes
                .extend_from_slice(&source[at / 8..at / 8 + whole]);
            self.bits += whole * 8;
            at += whole * 8;
        }
        for index in at..to {
            self.bit((source[index / 8] >> (7 - index % 8)).into());
        }
    }

    /// The RBSP trailing bits: the stop bit, then zeros to a byte.
    pub fn trailing(mut self) -> Vec<u8> {
        self.bit(1);
        self.bits = self.bits.next_multiple_of(8);
        self.bytes
    }
}

/// The RBSP of a NAL unit's payload: emulation prevention bytes removed.
pub fn rbsp(payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(payload.len());
    let mut zeros = 0;
    for &byte in payload {
        if zeros >= 2 && byte == 3 {
            zeros = 0;
            continue;
        }
        zeros = if byte == 0 { zeros + 1 } else { 0 };
        out.push(byte);
    }
    out
}

/// A NAL unit of header byte `header` and payload `rbsp`, emulation
/// prevention bytes inserted.
pub fn nal_unit(header: u8, rbsp: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rbsp.len() + rbsp.len() / 64 + 1);
    out.push(header);
    let mut zeros = 0;
    for &byte in rbsp {
        if zeros >= 2 && byte <= 3 {
            out.push(3);
            zeros = 0;
        }
        zeros = if byte == 0 { zeros + 1 } else { 0 };
        out.push(byte);
    }
    out
}

pub const NAL_SLICE: u8 = 1;
pub const NAL_IDR_SLICE: u8 = 5;

/// What slices depend on of a sequence parameter set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sps {
    pub chroma_format_idc: u32,
    pub bit_depth: [u32; 2],
    pub log2_max_frame_num: u32,
    pub poc_type: u32,
    pub log2_max_poc_lsb: u32,
    pub delta_pic_order_always_zero: bool,
    pub frame_mbs_only: bool,
}

/// High profiles, whose sequence parameter sets carry the chroma format
/// (7.3.2.1.1).
fn high_profile(profile: u32) -> bool {
    matches!(
        profile,
        100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135
    )
}

pub fn parse_sps(rbsp: &[u8]) -> Result<Sps, String> {
    let mut r = Reader::new(rbsp);
    let profile = r.u(8)?;
    r.u(16)?; // constraint flags, level_idc
    r.ue()?; // seq_parameter_set_id
    let (mut chroma_format_idc, mut bit_depth) = (1, [8, 8]);
    if high_profile(profile) {
        chroma_format_idc = r.ue()?;
        if chroma_format_idc == 3 {
            r.bit()?;
        }
        bit_depth = [r.ue()? + 8, r.ue()? + 8];
        r.bit()?; // qpprime_y_zero_transform_bypass_flag
        if r.flag()? {
            return Err("host encoder: sequence scaling matrices".into());
        }
    }
    let log2_max_frame_num = r.ue()? + 4;
    let poc_type = r.ue()?;
    let (mut log2_max_poc_lsb, mut delta_pic_order_always_zero) = (0, false);
    match poc_type {
        0 => log2_max_poc_lsb = r.ue()? + 4,
        1 => {
            delta_pic_order_always_zero = r.flag()?;
            r.se()?;
            r.se()?;
            for _ in 0..r.ue()? {
                r.se()?;
            }
        }
        _ => {}
    }
    r.ue()?; // max_num_ref_frames
    r.bit()?; // gaps_in_frame_num_value_allowed_flag
    r.ue()?;
    r.ue()?;
    let frame_mbs_only = r.flag()?;
    Ok(Sps {
        chroma_format_idc,
        bit_depth,
        log2_max_frame_num,
        poc_type,
        log2_max_poc_lsb,
        delta_pic_order_always_zero,
        frame_mbs_only,
    })
}

/// What slices depend on of a picture parameter set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pps {
    pub cabac: bool,
    pub bottom_field_pic_order_present: bool,
    pub num_ref_idx_default: [u32; 2],
    pub weighted_pred: bool,
    pub weighted_bipred_idc: u32,
    pub pic_init_qp: i32,
    pub chroma_qp_index_offset: [i32; 2],
    pub deblocking_filter_control_present: bool,
    pub constrained_intra_pred: bool,
    pub redundant_pic_cnt_present: bool,
    pub transform_8x8_mode: bool,
}

pub fn parse_pps(rbsp: &[u8]) -> Result<Pps, String> {
    let mut r = Reader::new(rbsp);
    r.ue()?; // pic_parameter_set_id
    r.ue()?; // seq_parameter_set_id
    let cabac = r.flag()?;
    let bottom_field_pic_order_present = r.flag()?;
    if r.ue()? != 0 {
        return Err("host encoder: slice groups".into());
    }
    let num_ref_idx_default = [r.ue()? + 1, r.ue()? + 1];
    let weighted_pred = r.flag()?;
    let weighted_bipred_idc = r.u(2)?;
    let pic_init_qp = 26 + r.se()?;
    r.se()?; // pic_init_qs_minus26
    let chroma_qp_index_offset = r.se()?;
    let deblocking_filter_control_present = r.flag()?;
    let constrained_intra_pred = r.flag()?;
    let redundant_pic_cnt_present = r.flag()?;
    let (mut transform_8x8_mode, mut second) = (false, chroma_qp_index_offset);
    if r.more_data() {
        transform_8x8_mode = r.flag()?;
        if r.flag()? {
            return Err("host encoder: picture scaling matrices".into());
        }
        second = r.se()?;
    }
    Ok(Pps {
        cabac,
        bottom_field_pic_order_present,
        num_ref_idx_default,
        weighted_pred,
        weighted_bipred_idc,
        pic_init_qp,
        chroma_qp_index_offset: [chroma_qp_index_offset, second],
        deblocking_filter_control_present,
        constrained_intra_pred,
        redundant_pic_cnt_present,
        transform_8x8_mode,
    })
}

/// The driver's parameter sets, as the picture setup tells NVENC them,
/// and the picture's own header values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Driver {
    pub sps: Sps,
    pub pps: Pps,
    pub pps_id: u32,
    pub idr: bool,
    pub reference: bool,
    pub frame_num: u32,
    pub poc_lsb: u32,
    pub delta_poc_bottom: i32,
    pub idr_pic_id: u32,
}

/// Refuses host parameter sets whose slice data the driver's sets would
/// not decode the same.
pub fn check(host: (&Sps, &Pps), driver: &Driver) -> Result<(), String> {
    let (sps, pps) = host;
    let mismatches = [
        ("entropy coder", pps.cabac != driver.pps.cabac),
        (
            "8x8 transform",
            pps.transform_8x8_mode != driver.pps.transform_8x8_mode,
        ),
        (
            "constrained intra prediction",
            pps.constrained_intra_pred != driver.pps.constrained_intra_pred,
        ),
        (
            "chroma QP offsets",
            pps.chroma_qp_index_offset != driver.pps.chroma_qp_index_offset,
        ),
        (
            "weighted prediction",
            pps.weighted_pred || driver.pps.weighted_pred,
        ),
        (
            "frames only",
            !sps.frame_mbs_only || !driver.sps.frame_mbs_only,
        ),
        (
            "4:2:0 8-bit",
            sps.chroma_format_idc != 1
                || driver.sps.chroma_format_idc != 1
                || sps.bit_depth != [8, 8]
                || driver.sps.bit_depth != [8, 8],
        ),
        ("picture order count type 1", driver.sps.poc_type == 1),
        (
            "redundant pictures",
            pps.redundant_pic_cnt_present || driver.pps.redundant_pic_cnt_present,
        ),
    ];
    match mismatches.iter().find(|(_, differ)| *differ) {
        Some((what, _)) => Err(format!(
            "{what}: the host encoder's sets ({sps:?}, {pps:?}) and the driver's \
             ({driver:?}) disagree"
        )),
        None => Ok(()),
    }
}

/// `slice_type % 5`.
const SLICE_P: u32 = 0;
const SLICE_I: u32 = 2;

/// A slice rewritten for the driver's sets.
#[derive(Debug, PartialEq)]
pub struct Slice {
    /// The NAL unit, header byte first, no start code.
    pub nal_unit: Vec<u8>,
    /// Bits of its slice header (from the NAL header byte on).
    pub header_bits: u32,
    pub qp: i32,
    /// `slice_type % 5`.
    pub slice_type: u32,
}

/// Writes the slice NAL unit `nal` of the host encoder's stream (sets
/// `sps`, `pps`) again for `driver`.
pub fn rewrite(nal: &[u8], sps: &Sps, pps: &Pps, driver: &Driver) -> Result<Slice, String> {
    let (&header, payload) = nal.split_first().ok_or("empty NAL unit")?;
    let nal_ref_idc = u32::from(header >> 5 & 3);
    let kind = header & 0x1F;
    let host_idr = kind == NAL_IDR_SLICE;
    if !matches!(kind, NAL_SLICE | NAL_IDR_SLICE) {
        return Err(format!(
            "host encoder: NAL unit type {kind} among the slices"
        ));
    }
    let rbsp = rbsp(payload);
    let mut r = Reader::new(&rbsp);
    let first_mb = r.ue()?;
    let slice_type = r.ue()?;
    let kind5 = slice_type % 5;
    if !matches!(kind5, SLICE_P | SLICE_I) {
        return Err(format!("host encoder: slice type {slice_type}"));
    }
    r.ue()?; // pic_parameter_set_id
    r.u(sps.log2_max_frame_num)?;
    if !sps.frame_mbs_only && r.flag()? {
        return Err("host encoder: field picture".into());
    }
    if host_idr {
        r.ue()?;
    }
    if sps.poc_type == 0 {
        r.u(sps.log2_max_poc_lsb)?;
        if pps.bottom_field_pic_order_present {
            r.se()?;
        }
    } else if sps.poc_type == 1 && !sps.delta_pic_order_always_zero {
        r.se()?;
        if pps.bottom_field_pic_order_present {
            r.se()?;
        }
    }
    if pps.redundant_pic_cnt_present {
        r.ue()?;
    }
    let mut references = 0;
    let mut modifications = Vec::new();
    if kind5 == SLICE_P {
        references = if r.flag()? {
            r.ue()? + 1
        } else {
            pps.num_ref_idx_default[0]
        };
        if r.flag()? {
            loop {
                let idc = r.ue()?;
                if idc == 3 {
                    break;
                }
                modifications.push((idc, r.ue()?));
            }
        }
        if references != 1 || !modifications.is_empty() {
            return Err(format!(
                "host encoder: a P slice of {references} reference(s) and list \
                 modifications {modifications:?} (one reference, the latest, is modeled)"
            ));
        }
    }
    if pps.weighted_pred && kind5 == SLICE_P {
        return Err("host encoder: weighted prediction".into());
    }
    let mut marking = Vec::new();
    if nal_ref_idc != 0 {
        if host_idr {
            r.bit()?; // no_output_of_prior_pics_flag
            if r.flag()? {
                return Err("host encoder: a long-term IDR picture".into());
            }
        } else if r.flag()? {
            loop {
                let operation = r.ue()?;
                if operation == 0 {
                    break;
                }
                if !matches!(operation, 1 | 5) {
                    return Err(format!(
                        "host encoder: memory management operation {operation}"
                    ));
                }
                let argument = if operation == 1 { Some(r.ue()?) } else { None };
                marking.push((operation, argument));
            }
        }
    }
    let cabac_init_idc = if pps.cabac && kind5 != SLICE_I {
        Some(r.ue()?)
    } else {
        None
    };
    let qp = pps.pic_init_qp + r.se()?;
    let mut deblocking = (0, 0, 0);
    if pps.deblocking_filter_control_present {
        deblocking.0 = r.ue()?;
        if deblocking.0 != 1 {
            deblocking.1 = r.se()?;
            deblocking.2 = r.se()?;
        }
    }
    if pps.cabac {
        while !r.position().is_multiple_of(8) {
            r.bit()?;
        }
    }
    let data = r.position();
    let end = stop_bit(&rbsp).ok_or("host encoder: slice without RBSP stop bit")?;
    if end < data {
        return Err("host encoder: slice data missing".into());
    }

    // The driver's picture: an IDR picture only when it says so (the
    // host's IDR picture otherwise stands as an I picture), a reference
    // when it says so, which the host's must then be too.
    if driver.idr && !host_idr {
        return Err("host encoder: no IDR picture where one was asked for".into());
    }
    if driver.reference != (nal_ref_idc != 0) {
        return Err(format!(
            "reference picture: the driver says {}, the host encoder {}",
            driver.reference,
            nal_ref_idc != 0
        ));
    }
    if driver.idr && kind5 != SLICE_I {
        return Err("host encoder: an IDR picture with P slices".into());
    }
    let (s, p) = (&driver.sps, &driver.pps);
    let mut w = Writer::default();
    w.ue(first_mb);
    w.ue(slice_type);
    w.ue(driver.pps_id);
    w.u(s.log2_max_frame_num, driver.frame_num);
    if !s.frame_mbs_only {
        w.flag(false);
    }
    if driver.idr {
        w.ue(driver.idr_pic_id);
    }
    if s.poc_type == 0 {
        w.u(s.log2_max_poc_lsb, driver.poc_lsb);
        if p.bottom_field_pic_order_present {
            w.se(driver.delta_poc_bottom);
        }
    }
    if kind5 == SLICE_P {
        let differs = references != p.num_ref_idx_default[0];
        w.flag(differs);
        if differs {
            w.ue(references - 1);
        }
        w.flag(false); // ref_pic_list_modification_flag_l0
    }
    if driver.reference {
        if driver.idr {
            w.flag(false); // no_output_of_prior_pics_flag
            w.flag(false); // long_term_reference_flag
        } else {
            // The host's own IDR picture marks nothing: as an I picture
            // it marks by the sliding window.
            let marking = if host_idr { &[][..] } else { &marking[..] };
            w.flag(!marking.is_empty());
            if !marking.is_empty() {
                for &(operation, argument) in marking {
                    w.ue(operation);
                    if let Some(argument) = argument {
                        w.ue(argument);
                    }
                }
                w.ue(0);
            }
        }
    }
    if let Some(idc) = cabac_init_idc {
        w.ue(idc);
    }
    w.se(qp - p.pic_init_qp);
    if p.deblocking_filter_control_present {
        w.ue(deblocking.0);
        if deblocking.0 != 1 {
            w.se(deblocking.1);
            w.se(deblocking.2);
        }
    } else if deblocking != (0, 0, 0) {
        return Err(format!(
            "deblocking {deblocking:?}: the driver's sets cannot say it"
        ));
    }
    let header_bits = 8 + w.bits() as u32;
    if p.cabac {
        while !w.bits().is_multiple_of(8) {
            w.bit(1);
        }
    }
    w.copy(&rbsp, data, end);
    let idc = if driver.reference {
        nal_ref_idc.max(1)
    } else {
        0
    };
    let kind = if driver.idr { NAL_IDR_SLICE } else { NAL_SLICE };
    Ok(Slice {
        nal_unit: nal_unit((idc << 5) as u8 | kind, &w.trailing()),
        header_bits,
        qp,
        slice_type: kind5,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_golomb_codes_round_trip() {
        let mut w = Writer::default();
        for value in [0, 1, 2, 7, 255, 65535] {
            w.ue(value);
        }
        for value in [0, 1, -1, 25, -26] {
            w.se(value);
        }
        w.u(5, 0b10110);
        let bytes = w.trailing();
        let mut r = Reader::new(&bytes);
        for value in [0, 1, 2, 7, 255, 65535] {
            assert_eq!(r.ue().unwrap(), value);
        }
        for value in [0, 1, -1, 25, -26] {
            assert_eq!(r.se().unwrap(), value);
        }
        assert_eq!(r.u(5).unwrap(), 0b10110);
        assert_eq!(stop_bit(&bytes), Some(r.position()));
    }

    #[test]
    fn emulation_prevention_is_inserted_and_removed() {
        let rbsp_bytes = [0, 0, 1, 0, 0, 0, 0, 3, 0, 0, 4];
        let nal = nal_unit(0x65, &rbsp_bytes);
        assert_eq!(nal, [0x65, 0, 0, 3, 1, 0, 0, 3, 0, 0, 3, 3, 0, 0, 4]);
        assert_eq!(rbsp(&nal[1..]), rbsp_bytes);
    }

    #[test]
    fn unaligned_bits_are_copied() {
        let source = [0b1011_0011, 0b0101_1100];
        let mut w = Writer::default();
        w.bit(1);
        w.copy(&source, 3, 13);
        assert_eq!(w.bits(), 11);
        let bytes = w.trailing();
        // 1, then source bits 3..13 = 1 0011 0101 1, then the stop bit.
        assert_eq!(bytes, [0b1100_1101, 0b0111_0000]);
    }
}
