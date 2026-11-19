//! IEEE 802.11 frames as the emulated radio puts them on the air
//! ([`crate::air`]) and takes them off it: MAC headers of management and
//! data frames, information elements, the LLC/SNAP encapsulation of
//! Ethernet payloads, and CCMP, the frame protection of WPA2 that the
//! radio applies with the keys its host installs.
//!
//! Sources: IEEE 802.11-2016 9.2 (frame formats), 9.3.3 (management frame
//! bodies), 9.4 (fields and elements), 12.5.3 (CCMP), RFC 3610 (CCM),
//! IEEE 802.2 / RFC 1042 (SNAP encapsulation of Ethernet types).

use aes::{
    Aes128,
    cipher::{BlockEncrypt, KeyInit, generic_array::GenericArray},
};

/// Frame types and subtypes (9.2.4.1.3), as the first byte of the frame
/// control field: subtype << 4 | type << 2.
pub const ASSOC_REQUEST: u8 = 0x00;
pub const ASSOC_RESPONSE: u8 = 0x10;
pub const PROBE_REQUEST: u8 = 0x40;
pub const PROBE_RESPONSE: u8 = 0x50;
pub const BEACON: u8 = 0x80;
pub const DISASSOC: u8 = 0xa0;
pub const AUTH: u8 = 0xb0;
pub const DEAUTH: u8 = 0xc0;
pub const ACTION: u8 = 0xd0;
pub const DATA: u8 = 0x08;
/// Flags of the frame control field's second byte.
pub const TO_DS: u8 = 0x01;
pub const FROM_DS: u8 = 0x02;
pub const PROTECTED: u8 = 0x40;

pub const BROADCAST: [u8; 6] = [0xff; 6];
/// Length of a three-address MAC header.
pub const HEADER_LEN: usize = 24;
/// The CCMP header and MIC around the encrypted payload (12.5.3.2).
pub const CCMP_HEADER_LEN: usize = 8;
pub const CCMP_MIC_LEN: usize = 8;

/// The parts of a three-address MAC header the radio uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    /// Type and subtype byte (one of the constants above; data frames
    /// may carry QoS subtypes, `0x88`).
    pub kind: u8,
    pub flags: u8,
    pub addr1: [u8; 6],
    pub addr2: [u8; 6],
    pub addr3: [u8; 6],
    pub sequence: u16,
}

impl Header {
    pub fn new(kind: u8, flags: u8, addr1: [u8; 6], addr2: [u8; 6], addr3: [u8; 6]) -> Self {
        Self {
            kind,
            flags,
            addr1,
            addr2,
            addr3,
            sequence: 0,
        }
    }

    /// Whether the frame is a data frame (type 2, any subtype).
    pub fn is_data(&self) -> bool {
        self.kind & 0x0c == DATA
    }

    /// Whether the frame is a management frame (type 0).
    pub fn is_management(&self) -> bool {
        self.kind & 0x0c == 0
    }

    /// The header's length in a frame: 24 bytes, 30 with a fourth address
    /// (to and from the DS), 2 more for a QoS data frame (9.2.3).
    fn length(kind: u8, flags: u8) -> usize {
        let mut length = HEADER_LEN;
        if kind & 0x0c == DATA {
            if flags & (TO_DS | FROM_DS) == TO_DS | FROM_DS {
                length += 6;
            }
            if kind & 0x80 != 0 {
                length += 2;
            }
        }
        length
    }

    /// The header of `frame` and where its body starts.
    pub fn parse(frame: &[u8]) -> Option<(Header, usize)> {
        let (&kind, &flags) = (frame.first()?, frame.get(1)?);
        let length = Self::length(kind, flags);
        if frame.len() < length {
            return None;
        }
        let address = |at: usize| -> [u8; 6] { frame[at..at + 6].try_into().unwrap() };
        Some((
            Header {
                kind,
                flags,
                addr1: address(4),
                addr2: address(10),
                addr3: address(16),
                sequence: u16::from_le_bytes([frame[22], frame[23]]),
            },
            length,
        ))
    }

    /// The 24-byte header: frame control, a zero duration (the radio does
    /// not model medium reservation), the addresses and sequence control.
    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(HEADER_LEN);
        bytes.extend([self.kind, self.flags, 0, 0]);
        bytes.extend(self.addr1);
        bytes.extend(self.addr2);
        bytes.extend(self.addr3);
        bytes.extend(self.sequence.to_le_bytes());
        bytes
    }

    /// The frame: this header and `body`.
    pub fn frame(&self, body: &[u8]) -> Vec<u8> {
        let mut frame = self.encode();
        frame.extend(body);
        frame
    }
}

/// The information elements of a management frame body: (ID, contents).
/// A truncated last element ends the list.
pub fn elements(mut bytes: &[u8]) -> impl Iterator<Item = (u8, &[u8])> {
    std::iter::from_fn(move || {
        let (&id, &length) = (bytes.first()?, bytes.get(1)?);
        let contents = bytes.get(2..2 + usize::from(length))?;
        bytes = &bytes[2 + usize::from(length)..];
        Some((id, contents))
    })
}

/// The contents of the first element `id` of `bytes`.
pub fn element(bytes: &[u8], id: u8) -> Option<&[u8]> {
    elements(bytes).find(|&(i, _)| i == id).map(|(_, c)| c)
}

/// LLC/SNAP header of an Ethernet type carried in an 802.11 data frame
/// (RFC 1042: DSAP/SSAP 0xAA, UI, OUI 00:00:00).
const SNAP: [u8; 6] = [0xaa, 0xaa, 0x03, 0x00, 0x00, 0x00];

/// The 802.11 data frame body carrying Ethernet frame `ethernet`'s
/// payload: SNAP header, its type, its payload. `None` for a frame too
/// short to be Ethernet.
pub fn snap_body(ethernet: &[u8]) -> Option<Vec<u8>> {
    let payload = ethernet.get(12..)?;
    if payload.len() < 2 {
        return None;
    }
    let mut body = SNAP.to_vec();
    body.extend(payload);
    Some(body)
}

/// The Ethernet frame from destination `da` and source `sa` of a data
/// frame body: `None` unless it is SNAP-encapsulated.
pub fn ethernet(da: [u8; 6], sa: [u8; 6], body: &[u8]) -> Option<Vec<u8>> {
    if body.get(..6) != Some(&SNAP) || body.len() < 8 {
        return None;
    }
    let mut frame = Vec::with_capacity(12 + body.len() - 6);
    frame.extend(da);
    frame.extend(sa);
    frame.extend(&body[6..]);
    Some(frame)
}

/// A CCMP key (a temporal key: pairwise or group).
pub struct Ccmp {
    cipher: Aes128,
}

impl Ccmp {
    pub fn new(key: &[u8; 16]) -> Self {
        Self {
            cipher: Aes128::new(GenericArray::from_slice(key)),
        }
    }

    fn encrypt_block(&self, block: &mut [u8; 16]) {
        self.cipher
            .encrypt_block(GenericArray::from_mut_slice(block));
    }

    /// The nonce (12.5.3.3.4): priority (0 outside QoS data; management
    /// frames are not protected here), the transmitter address and the
    /// packet number, most significant byte first.
    fn nonce(header: &Header, pn: u64) -> [u8; 13] {
        let mut nonce = [0; 13];
        nonce[1..7].copy_from_slice(&header.addr2);
        nonce[7..].copy_from_slice(&pn.to_be_bytes()[2..]);
        nonce
    }

    /// The additional authentication data (12.5.3.3.3) of a three-address
    /// non-QoS data frame: frame control with the subtype's QoS bits, retry,
    /// power management and more data masked and protected set; the three
    /// addresses; sequence control with the sequence number masked.
    fn aad(header: &Header) -> Vec<u8> {
        let mut aad = Vec::with_capacity(22);
        aad.push(header.kind & 0x8f);
        aad.push((header.flags & !0x38) | PROTECTED);
        aad.extend(header.addr1);
        aad.extend(header.addr2);
        aad.extend(header.addr3);
        aad.extend((header.sequence & 0x000f).to_le_bytes());
        aad
    }

    /// CCM's authentication (RFC 3610 2.2) with an 8-byte MIC and 2-byte
    /// length field, over `aad` and `message`.
    fn mac(&self, nonce: &[u8; 13], aad: &[u8], message: &[u8]) -> [u8; 8] {
        let mut x = [0u8; 16];
        x[0] = 0x40 | (((CCMP_MIC_LEN as u8 - 2) / 2) << 3) | 1;
        x[1..14].copy_from_slice(nonce);
        x[14..].copy_from_slice(&(message.len() as u16).to_be_bytes());
        self.encrypt_block(&mut x);
        let mut blocks = (aad.len() as u16).to_be_bytes().to_vec();
        blocks.extend(aad);
        blocks.resize(blocks.len().next_multiple_of(16), 0);
        let mut padded = message.to_vec();
        padded.resize(padded.len().next_multiple_of(16), 0);
        blocks.extend(padded);
        for block in blocks.chunks(16) {
            for (x, b) in x.iter_mut().zip(block) {
                *x ^= b;
            }
            self.encrypt_block(&mut x);
        }
        x[..8].try_into().unwrap()
    }

    /// CCM's encryption (RFC 3610 2.3): XORs `data` with the key stream
    /// from counter 1, and returns block S0 for the MIC.
    fn ctr(&self, nonce: &[u8; 13], data: &mut [u8]) -> [u8; 16] {
        let block = |counter: u16| {
            let mut a = [0u8; 16];
            a[0] = 1;
            a[1..14].copy_from_slice(nonce);
            a[14..].copy_from_slice(&counter.to_be_bytes());
            self.encrypt_block(&mut a);
            a
        };
        for (i, chunk) in data.chunks_mut(16).enumerate() {
            let stream = block(i as u16 + 1);
            for (d, s) in chunk.iter_mut().zip(stream) {
                *d ^= s;
            }
        }
        block(0)
    }

    /// Protects data frame `header` + `plain` with packet number `pn` and
    /// key ID `key_id`: the header with the protected flag, the CCMP
    /// header, the encrypted body and the encrypted MIC.
    pub fn encrypt(&self, header: &Header, pn: u64, key_id: u8, plain: &[u8]) -> Vec<u8> {
        let mut header = *header;
        header.flags |= PROTECTED;
        let nonce = Self::nonce(&header, pn);
        let mic = self.mac(&nonce, &Self::aad(&header), plain);
        let mut body = plain.to_vec();
        let s0 = self.ctr(&nonce, &mut body);
        let pn = pn.to_le_bytes();
        let mut frame = header.encode();
        frame.extend([
            pn[0],
            pn[1],
            0,
            0x20 | (key_id << 6),
            pn[2],
            pn[3],
            pn[4],
            pn[5],
        ]);
        frame.extend(body);
        frame.extend(mic.iter().zip(s0).map(|(m, s)| m ^ s));
        frame
    }

    /// The body of protected frame `frame` (header `header` ending at
    /// `at`), its packet number and key ID; `None` if it does not decrypt
    /// (another key, or tampered with).
    pub fn decrypt(&self, header: &Header, frame: &[u8], at: usize) -> Option<(Vec<u8>, u64, u8)> {
        let ccmp = frame.get(at..at + CCMP_HEADER_LEN)?;
        if ccmp[3] & 0x20 == 0 || frame.len() < at + CCMP_HEADER_LEN + CCMP_MIC_LEN {
            return None;
        }
        let pn = u64::from_le_bytes([ccmp[0], ccmp[1], ccmp[4], ccmp[5], ccmp[6], ccmp[7], 0, 0]);
        let key_id = ccmp[3] >> 6;
        let nonce = Self::nonce(header, pn);
        let end = frame.len() - CCMP_MIC_LEN;
        let mut body = frame[at + CCMP_HEADER_LEN..end].to_vec();
        let s0 = self.ctr(&nonce, &mut body);
        let mic: Vec<u8> = frame[end..].iter().zip(s0).map(|(m, s)| m ^ s).collect();
        (self.mac(&nonce, &Self::aad(header), &body)[..] == mic[..]).then_some((body, pn, key_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(s: &str) -> Vec<u8> {
        let s: String = s.split_whitespace().collect();
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    /// RFC 3610 packet vector #1 checks the CCM core (there with L = 2 and
    /// an 8-byte MIC too): key C0..CF, nonce 00000003020100A0A1A2A3A4A5,
    /// 8 bytes of header 00..07, payload 08..1E.
    #[test]
    fn ccm_core_matches_rfc_3610() {
        let key: [u8; 16] = hex("C0C1C2C3C4C5C6C7C8C9CACBCCCDCECF").try_into().unwrap();
        let ccmp = Ccmp::new(&key);
        let nonce: [u8; 13] = hex("00000003020100A0A1A2A3A4A5").try_into().unwrap();
        let aad = hex("0001020304050607");
        let mut data = hex("08090A0B0C0D0E0F101112131415161718191A1B1C1D1E");
        let mic = ccmp.mac(&nonce, &aad, &data);
        let s0 = ccmp.ctr(&nonce, &mut data);
        let tag: Vec<u8> = mic.iter().zip(s0).map(|(m, s)| m ^ s).collect();
        assert_eq!(data, hex("588C979A61C663D2F066D0C2C0F989806D5F6B61DAC384"));
        assert_eq!(tag, hex("17E8D12CFDF926E0"));
    }

    #[test]
    fn protected_frames_round_trip_and_resist_tampering() {
        let ccmp = Ccmp::new(&[7; 16]);
        let header = Header::new(DATA, TO_DS, [1; 6], [2; 6], [3; 6]);
        let frame = ccmp.encrypt(&header, 0x0102_0304_0506, 1, b"some payload");
        let (parsed, at) = Header::parse(&frame).unwrap();
        assert_eq!(parsed.flags & PROTECTED, PROTECTED);
        assert_eq!(
            ccmp.decrypt(&parsed, &frame, at),
            Some((b"some payload".to_vec(), 0x0102_0304_0506, 1))
        );
        let mut tampered = frame.clone();
        *tampered.last_mut().unwrap() ^= 1;
        assert_eq!(ccmp.decrypt(&parsed, &tampered, at), None);
        assert_eq!(Ccmp::new(&[8; 16]).decrypt(&parsed, &frame, at), None);
    }

    #[test]
    fn snap_encapsulation_round_trips() {
        let mut eth = vec![1u8; 6];
        eth.extend([2u8; 6]);
        eth.extend([0x08, 0x00, 0x45, 0x00]);
        let body = snap_body(&eth).unwrap();
        assert_eq!(body[..8], [0xaa, 0xaa, 3, 0, 0, 0, 0x08, 0x00]);
        assert_eq!(ethernet([1; 6], [2; 6], &body), Some(eth));
    }

    #[test]
    fn elements_parse_and_stop_at_truncation() {
        let body = [0, 2, b'a', b'b', 3, 1, 6, 221, 9, 1];
        let list: Vec<_> = elements(&body).collect();
        assert_eq!(list, [(0, &b"ab"[..]), (3, &[6][..])]);
        assert_eq!(element(&body, 3), Some(&[6][..]));
    }
}
