//! L2CAP framing: the basic frames ACL data carries and the signalling
//! commands on channel 1, as the Bluetooth Core Specification (Vol 3 Part
//! A §3 and §4) lays them out; public Linux `include/net/bluetooth/l2cap.h`
//! names the same fields.

/// The signalling channel of an ACL link (Vol 3 Part A §2.1).
pub const SIGNALLING: u16 = 0x0001;
/// The first channel identifier a device allocates for its own end of a
/// connection-oriented channel.
pub const FIRST_DYNAMIC: u16 = 0x0040;

/// Signalling command codes (Vol 3 Part A §4).
pub const CONNECTION_REQUEST: u8 = 0x02;
pub const CONNECTION_RESPONSE: u8 = 0x03;
pub const CONFIGURATION_REQUEST: u8 = 0x04;
pub const CONFIGURATION_RESPONSE: u8 = 0x05;
pub const DISCONNECTION_REQUEST: u8 = 0x06;
pub const DISCONNECTION_RESPONSE: u8 = 0x07;
pub const ECHO_REQUEST: u8 = 0x08;
pub const ECHO_RESPONSE: u8 = 0x09;
pub const INFORMATION_REQUEST: u8 = 0x0a;
pub const INFORMATION_RESPONSE: u8 = 0x0b;

/// Connection response results (§4.3): success, pending.
pub const CONNECTION_SUCCESSFUL: u16 = 0x0000;
pub const CONNECTION_PENDING: u16 = 0x0001;

/// A basic information frame (B-frame, §3.1): length, channel, payload.
pub fn frame(channel: u16, payload: &[u8]) -> Vec<u8> {
    let mut frame = Vec::with_capacity(4 + payload.len());
    frame.extend((payload.len() as u16).to_le_bytes());
    frame.extend(channel.to_le_bytes());
    frame.extend(payload);
    frame
}

/// The channel and payload of a complete B-frame; `None` if `pdu` is not
/// exactly one.
pub fn parse_frame(pdu: &[u8]) -> Option<(u16, &[u8])> {
    let (header, payload) = pdu.split_first_chunk::<4>()?;
    let length = usize::from(u16::from_le_bytes([header[0], header[1]]));
    (payload.len() == length).then(|| (u16::from_le_bytes([header[2], header[3]]), payload))
}

/// How long the PDU that starts with `start` is, header included, once
/// its length field is there.
pub fn pdu_length(start: &[u8]) -> Option<usize> {
    let length = start.first_chunk::<2>()?;
    Some(4 + usize::from(u16::from_le_bytes(*length)))
}

/// One signalling command (§4): code, identifier, data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Signal<'a> {
    pub code: u8,
    pub id: u8,
    pub data: &'a [u8],
}

/// The commands of a signalling frame's payload, which may hold several
/// (§4); `None` if one is cut short.
pub fn signals(mut payload: &[u8]) -> Option<Vec<Signal<'_>>> {
    let mut signals = Vec::new();
    while !payload.is_empty() {
        let (header, rest) = payload.split_first_chunk::<4>()?;
        let length = usize::from(u16::from_le_bytes([header[2], header[3]]));
        if rest.len() < length {
            return None;
        }
        let (data, rest) = rest.split_at(length);
        signals.push(Signal {
            code: header[0],
            id: header[1],
            data,
        });
        payload = rest;
    }
    Some(signals)
}

/// A signalling frame carrying one command.
pub fn signal(code: u8, id: u8, data: &[u8]) -> Vec<u8> {
    let mut command = vec![code, id];
    command.extend((data.len() as u16).to_le_bytes());
    command.extend(data);
    frame(SIGNALLING, &command)
}

/// Little-endian 16-bit words, as every field of a signalling command is.
pub fn words(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// The 16-bit word at `index` (counted in words) of `data`.
pub fn word(data: &[u8], index: usize) -> Option<u16> {
    let bytes = data.get(2 * index..2 * index + 2)?;
    Some(u16::from_le_bytes([bytes[0], bytes[1]]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_carry_their_length_and_channel() {
        let frame = frame(0x0041, &[0xa2, 0x01]);
        assert_eq!(frame, [2, 0, 0x41, 0, 0xa2, 0x01]);
        assert_eq!(parse_frame(&frame), Some((0x0041, &[0xa2, 0x01][..])));
        assert_eq!(pdu_length(&frame), Some(6));
        // One byte short, or one too many, is not a frame.
        assert_eq!(parse_frame(&frame[..5]), None);
        assert_eq!(parse_frame(&[frame.as_slice(), &[0]].concat()), None);
    }

    #[test]
    fn a_signalling_frame_may_hold_several_commands() {
        // A connection request for PSM 0x11 from channel 0x0040, then an
        // information request for the extended features.
        let request = signal(CONNECTION_REQUEST, 1, &words(&[0x0011, 0x0040]));
        assert_eq!(request, [8, 0, 1, 0, 2, 1, 4, 0, 0x11, 0, 0x40, 0]);
        let (channel, payload) = parse_frame(&request).unwrap();
        assert_eq!(channel, SIGNALLING);
        let mut both = payload.to_vec();
        both.extend([INFORMATION_REQUEST, 2, 2, 0, 2, 0]);
        let parsed = signals(&both).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!((parsed[0].code, parsed[0].id), (CONNECTION_REQUEST, 1));
        assert_eq!(word(parsed[0].data, 0), Some(0x0011));
        assert_eq!(word(parsed[0].data, 1), Some(0x0040));
        assert_eq!(parsed[1].data, [2, 0]);
        // A command whose data is cut short.
        assert_eq!(signals(&both[..both.len() - 1]), None);
    }
}
