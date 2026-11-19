//! The Pro Controller's Service Discovery Protocol server: what it says
//! of itself when the console asks, on L2CAP PSM 1.
//!
//! PDUs and data elements are the Bluetooth Core Specification's (Vol 3
//! Part B §3 and §4), big-endian throughout. The one record is its HID
//! service (HID Profile §5.3), with the attribute values a real Pro
//! Controller publishes, as recorded by joycontrol's `sdp_record_hid.xml`
//! (facts only): service name "Wireless Gamepad", provider "Nintendo",
//! HID parser version 1.11, country 0, and its report descriptor.

/// PDU identifiers (§4.2).
const ERROR_RESPONSE: u8 = 0x01;
const SEARCH_REQUEST: u8 = 0x02;
const SEARCH_RESPONSE: u8 = 0x03;
const ATTRIBUTE_REQUEST: u8 = 0x04;
const ATTRIBUTE_RESPONSE: u8 = 0x05;
const SEARCH_ATTRIBUTE_REQUEST: u8 = 0x06;
const SEARCH_ATTRIBUTE_RESPONSE: u8 = 0x07;
/// Error codes (§4.4.1): invalid record handle, invalid syntax, invalid
/// continuation state.
const INVALID_HANDLE: u16 = 0x0002;
const INVALID_SYNTAX: u16 = 0x0003;
const INVALID_CONTINUATION: u16 = 0x0005;

/// The handle of the HID record (the first after the SDP server's own,
/// 0).
const HID_RECORD: u32 = 0x0001_0000;

/// The HID report descriptor the controller publishes (attribute 0x0206),
/// as joycontrol recorded it from a real one: input reports 0x30 (and the
/// simple 0x3F layout), 0x21, 0x81; output reports 0x01, 0x10, 0x80, 0x82.
const REPORT_DESCRIPTOR: &str = "\
050115000904a1018530050105091901290a150025017501950a5500650081020509190b290e15\
0025017501950481027501950281030b01000100a1000b300001000b310001000b320001000b35\
000100150027ffff0000751095048102c00b39000100150025073500463b016514750495018102\
0509190f2912150025017501950481027508953481030600ff852109017508953f81038581090275\
08953f8103850109037508953f9183851009047508953f9183858009057508953f9183858209067508\
953f9183c0";

/// A data element (§3).
#[derive(Debug, Clone, PartialEq)]
enum Element {
    U8(u8),
    U16(u16),
    U32(u32),
    Uuid(u16),
    Text(Vec<u8>),
    Bool(bool),
    Sequence(Vec<Element>),
}

impl Element {
    fn encode(&self, out: &mut Vec<u8>) {
        // Type in the high five bits, size index in the low three.
        let variable = |out: &mut Vec<u8>, kind: u8, body: &[u8]| {
            match body.len() {
                0..=0xff => out.extend([kind << 3 | 5, body.len() as u8]),
                _ => {
                    out.push(kind << 3 | 6);
                    out.extend((body.len() as u16).to_be_bytes());
                }
            }
            out.extend(body);
        };
        match self {
            Self::U8(value) => out.extend([1 << 3, *value]),
            Self::U16(value) => {
                out.push(1 << 3 | 1);
                out.extend(value.to_be_bytes());
            }
            Self::U32(value) => {
                out.push(1 << 3 | 2);
                out.extend(value.to_be_bytes());
            }
            Self::Uuid(value) => {
                out.push(3 << 3 | 1);
                out.extend(value.to_be_bytes());
            }
            Self::Text(text) => variable(out, 4, text),
            Self::Bool(value) => out.extend([5 << 3, u8::from(*value)]),
            Self::Sequence(items) => {
                let mut body = Vec::new();
                for item in items {
                    item.encode(&mut body);
                }
                variable(out, 6, &body);
            }
        }
    }

    fn bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.encode(&mut out);
        out
    }

    /// The 16-bit UUIDs anywhere in the element.
    fn uuids(&self, found: &mut Vec<u16>) {
        match self {
            Self::Uuid(uuid) => found.push(*uuid),
            Self::Sequence(items) => items.iter().for_each(|item| item.uuids(found)),
            _ => {}
        }
    }
}

/// The HID record's attributes, by ID (Assigned Numbers; HID Profile
/// §5.3.4).
fn hid_record() -> Vec<(u16, Element)> {
    use Element::*;
    let text = |text: &str| Text(text.as_bytes().to_vec());
    let descriptor = (0..REPORT_DESCRIPTOR.len())
        .step_by(2)
        .map(|at| u8::from_str_radix(&REPORT_DESCRIPTOR[at..at + 2], 16).unwrap())
        .collect();
    vec![
        (0x0000, U32(HID_RECORD)),
        // Service class: HID; protocols: L2CAP PSM 0x11, HIDP.
        (0x0001, Sequence(vec![Uuid(0x1124)])),
        (
            0x0004,
            Sequence(vec![
                Sequence(vec![Uuid(0x0100), U16(0x0011)]),
                Sequence(vec![Uuid(0x0011)]),
            ]),
        ),
        // Public browse group.
        (0x0005, Sequence(vec![Uuid(0x1002)])),
        // English, UTF-8, names from attribute 0x0100.
        (
            0x0006,
            Sequence(vec![U16(0x656e), U16(0x006a), U16(0x0100)]),
        ),
        // HID profile version 1.0.
        (
            0x0009,
            Sequence(vec![Sequence(vec![Uuid(0x1124), U16(0x0100)])]),
        ),
        // The interrupt channel: L2CAP PSM 0x13, HIDP.
        (
            0x000d,
            Sequence(vec![Sequence(vec![
                Sequence(vec![Uuid(0x0100), U16(0x0013)]),
                Sequence(vec![Uuid(0x0011)]),
            ])]),
        ),
        (0x0100, text("Wireless Gamepad")),
        (0x0101, text("Gamepad")),
        (0x0102, text("Nintendo")),
        // Device release 1.0, parser version 1.11, subclass 0x08
        // (gamepad), country 0; virtually cabled, reconnects itself.
        (0x0200, U16(0x0100)),
        (0x0201, U16(0x0111)),
        (0x0202, U8(0x08)),
        (0x0203, U8(0x00)),
        (0x0204, Bool(true)),
        (0x0205, Bool(true)),
        // The report descriptor (type 0x22).
        (
            0x0206,
            Sequence(vec![Sequence(vec![U8(0x22), Text(descriptor)])]),
        ),
        // Language: US English, base 0x0100.
        (
            0x0207,
            Sequence(vec![Sequence(vec![U16(0x0409), U16(0x0100)])]),
        ),
        // Profile version 1.0, supervision timeout 0x0c80, not normally
        // connectable, boot device, sniff intervals 0x0640 and 0x0320.
        (0x020b, U16(0x0100)),
        (0x020c, U16(0x0c80)),
        (0x020d, Bool(false)),
        (0x020e, Bool(true)),
        (0x020f, U16(0x0640)),
        (0x0210, U16(0x0320)),
    ]
}

/// A data element sequence at the start of `bytes`: its body, and what
/// follows it.
fn sequence(bytes: &[u8]) -> Option<(&[u8], &[u8])> {
    let (&header, rest) = bytes.split_first()?;
    if header >> 3 != 6 {
        return None;
    }
    let (length, rest) = match header & 7 {
        5 => (usize::from(*rest.first()?), &rest[1..]),
        6 => (
            usize::from(u16::from_be_bytes(*rest.first_chunk()?)),
            rest.get(2..)?,
        ),
        _ => return None,
    };
    (rest.len() >= length).then(|| rest.split_at(length))
}

/// The elements of a sequence body: UUIDs (16, 32 or 128 bits, as the
/// 16-bit ones of the Bluetooth base UUID) or unsigned integers.
fn items(mut body: &[u8]) -> Option<Vec<(u8, u32)>> {
    let mut items = Vec::new();
    while let Some((&header, rest)) = body.split_first() {
        let size = match header & 7 {
            1 => 2,
            2 => 4,
            4 => 16,
            _ => return None,
        };
        let value = rest.get(..size)?;
        let number = match size {
            2 => u32::from(u16::from_be_bytes([value[0], value[1]])),
            // A 128-bit UUID is the base UUID's when its last 96 bits are
            // 0000-1000-8000-00805F9B34FB.
            16 => {
                const BASE: [u8; 12] = [0, 0, 0x10, 0, 0x80, 0, 0, 0x80, 0x5f, 0x9b, 0x34, 0xfb];
                if value[4..] != BASE {
                    return None;
                }
                u32::from_be_bytes(value[..4].try_into().unwrap())
            }
            _ => u32::from_be_bytes(value.try_into().unwrap()),
        };
        // The header: type and size, which tells an ID from a range.
        items.push((header, number));
        body = &rest[size..];
    }
    Some(items)
}

/// Whether the record whose UUIDs are `has` matches every UUID of a
/// search pattern (§2.5.2).
fn matches(pattern: &[(u8, u32)], has: &[u16]) -> bool {
    !pattern.is_empty()
        && pattern
            .iter()
            .all(|&(header, uuid)| header >> 3 == 3 && has.iter().any(|&h| u32::from(h) == uuid))
}

/// The attributes of `record` an attribute ID list asks for, as an
/// attribute list (a sequence of ID and value pairs, §4.6).
fn attribute_list(record: &[(u16, Element)], wanted: &[(u8, u32)]) -> Option<Element> {
    let mut list = Vec::new();
    for (id, value) in record {
        let asked = wanted.iter().try_fold(false, |asked, &(kind, item)| {
            (kind >> 3 == 1).then_some(())?;
            Some(
                asked
                    || match kind & 7 == 2 {
                        // A range: first ID in the high half, last in the low.
                        true => (item >> 16..=item & 0xffff).contains(&u32::from(*id)),
                        false => item == u32::from(*id),
                    },
            )
        })?;
        if asked {
            list.push(Element::U16(*id));
            list.push(value.clone());
        }
    }
    Some(Element::Sequence(list))
}

/// The SDP server: one transaction's answer at a time, cut to what the
/// client takes, with the rest kept for its continuation.
#[derive(Default, Clone, serde::Serialize, serde::Deserialize)]
pub(super) struct Server {
    /// The bytes of the answer a continuation asks for more of.
    pending: Vec<u8>,
}

impl Server {
    /// Answers one request PDU; `None` if it is not one this server knows.
    pub(super) fn answer(&mut self, pdu: &[u8]) -> Option<Vec<u8>> {
        let (&id, rest) = pdu.split_first()?;
        let transaction = rest.get(..2)?;
        let length = usize::from(u16::from_be_bytes(rest.get(2..4)?.try_into().ok()?));
        let parameters = rest.get(4..)?;
        if parameters.len() != length {
            return None;
        }
        let respond = |id: u8, body: &[u8]| {
            let mut pdu = vec![id];
            pdu.extend(transaction);
            pdu.extend((body.len() as u16).to_be_bytes());
            pdu.extend(body);
            pdu
        };
        let error = |code: u16| respond(ERROR_RESPONSE, &code.to_be_bytes());
        let record = hid_record();
        let mut has = Vec::new();
        for (_, value) in &record {
            value.uuids(&mut has);
        }
        match id {
            SEARCH_REQUEST => {
                let Some((pattern, rest)) = sequence(parameters) else {
                    return Some(error(INVALID_SYNTAX));
                };
                let (Some(pattern), Some(_), Some(&[0])) =
                    (items(pattern), rest.get(..2), rest.get(2..))
                else {
                    return Some(error(INVALID_SYNTAX));
                };
                let count = u16::from(matches(&pattern, &has));
                let mut body = count.to_be_bytes().to_vec();
                body.extend(count.to_be_bytes());
                if count == 1 {
                    body.extend(HID_RECORD.to_be_bytes());
                }
                body.push(0);
                Some(respond(SEARCH_RESPONSE, &body))
            }
            ATTRIBUTE_REQUEST | SEARCH_ATTRIBUTE_REQUEST => {
                let (lists, rest) = match id {
                    ATTRIBUTE_REQUEST => {
                        let handle = u32::from_be_bytes(parameters.get(..4)?.try_into().ok()?);
                        if handle != HID_RECORD {
                            return Some(error(INVALID_HANDLE));
                        }
                        (None, &parameters[4..])
                    }
                    _ => {
                        let Some((pattern, rest)) = sequence(parameters) else {
                            return Some(error(INVALID_SYNTAX));
                        };
                        let Some(pattern) = items(pattern) else {
                            return Some(error(INVALID_SYNTAX));
                        };
                        (Some(matches(&pattern, &has)), rest)
                    }
                };
                let most = usize::from(u16::from_be_bytes(rest.get(..2)?.try_into().ok()?));
                let Some((wanted, continuation)) = sequence(&rest[2..]) else {
                    return Some(error(INVALID_SYNTAX));
                };
                let Some(wanted) = items(wanted) else {
                    return Some(error(INVALID_SYNTAX));
                };
                let Some(list) = attribute_list(&record, &wanted) else {
                    return Some(error(INVALID_SYNTAX));
                };
                // A service search attribute answer is a sequence of the
                // matching records' lists.
                let answer = match lists {
                    None => list.bytes(),
                    Some(true) => Element::Sequence(vec![list]).bytes(),
                    Some(false) => Element::Sequence(Vec::new()).bytes(),
                };
                // The continuation state is the offset reached, two bytes.
                let from = match continuation {
                    [0] => {
                        self.pending = answer;
                        0
                    }
                    [2, high, low] if self.pending == answer => {
                        usize::from(u16::from_be_bytes([*high, *low]))
                    }
                    _ => return Some(error(INVALID_CONTINUATION)),
                };
                if most < 7 || from > self.pending.len() {
                    return Some(error(INVALID_SYNTAX));
                }
                let to = (from + most).min(self.pending.len());
                let mut body = ((to - from) as u16).to_be_bytes().to_vec();
                body.extend(&self.pending[from..to]);
                match to == self.pending.len() {
                    true => {
                        body.push(0);
                        self.pending.clear();
                    }
                    false => {
                        body.push(2);
                        body.extend((to as u16).to_be_bytes());
                    }
                }
                let response = match id {
                    ATTRIBUTE_REQUEST => ATTRIBUTE_RESPONSE,
                    _ => SEARCH_ATTRIBUTE_RESPONSE,
                };
                Some(respond(response, &body))
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A service search attribute request for the HID class, all
    /// attributes, at most `most` bytes per answer, with `continuation`.
    fn request(most: u16, continuation: &[u8]) -> Vec<u8> {
        let mut parameters = vec![0x35, 3, 0x19, 0x11, 0x24];
        parameters.extend(most.to_be_bytes());
        parameters.extend([0x35, 5, 0x0a, 0, 0, 0xff, 0xff]);
        parameters.extend(continuation);
        let mut pdu = vec![SEARCH_ATTRIBUTE_REQUEST, 0, 7];
        pdu.extend((parameters.len() as u16).to_be_bytes());
        pdu.extend(parameters);
        pdu
    }

    #[test]
    fn data_elements_have_their_type_and_size_in_the_header() {
        assert_eq!(Element::U16(0x0100).bytes(), [0x09, 1, 0]);
        assert_eq!(Element::Uuid(0x1124).bytes(), [0x19, 0x11, 0x24]);
        assert_eq!(Element::Bool(true).bytes(), [0x28, 1]);
        assert_eq!(Element::Text(b"Nintendo".to_vec()).bytes()[..2], [0x25, 8]);
        let long = Element::Sequence(vec![Element::Text(vec![0; 300])]);
        assert_eq!(long.bytes()[..3], [0x36, 0x01, 0x2f]);
    }

    #[test]
    fn the_hid_record_comes_in_pieces_the_client_takes() {
        let mut server = Server::default();
        let mut whole = Vec::new();
        let mut continuation = vec![0];
        for round in 0.. {
            let answer = server.answer(&request(100, &continuation)).unwrap();
            assert_eq!(answer[..3], [SEARCH_ATTRIBUTE_RESPONSE, 0, 7]);
            let count = usize::from(u16::from_be_bytes([answer[5], answer[6]]));
            assert!(count <= 100);
            whole.extend(&answer[7..7 + count]);
            continuation = answer[7 + count..].to_vec();
            if continuation == [0] {
                // A record of some 300 bytes: three pieces at least.
                assert!(round >= 2);
                break;
            }
            assert_eq!(continuation[0], 2);
        }
        // One record's list, whose first attribute is its handle and which
        // holds the report descriptor.
        let (records, rest) = sequence(&whole).unwrap();
        assert!(rest.is_empty());
        let (list, _) = sequence(records).unwrap();
        assert_eq!(list[..8], [0x09, 0, 0, 0x0a, 0, 1, 0, 0]);
        assert!(whole.windows(3).any(|w| w == [0x05, 0x01, 0x15]));
        // A continuation that does not fit the answer under way.
        let wrong = server.answer(&request(100, &[2, 0, 5])).unwrap();
        assert_eq!(wrong[0], ERROR_RESPONSE);
    }

    #[test]
    fn a_search_finds_the_record_by_any_of_its_uuids_and_nothing_else() {
        let search = |uuid: [u8; 2]| {
            let mut server = Server::default();
            let pdu = [
                SEARCH_REQUEST,
                0,
                1,
                0,
                8,
                0x35,
                3,
                0x19,
                uuid[0],
                uuid[1],
                0,
                1,
                0,
            ];
            server.answer(&pdu).unwrap()
        };
        // L2CAP (0x0100) is in the record; Device ID (0x1200) is not.
        assert_eq!(search([0x01, 0x00])[5..], [0, 1, 0, 1, 0, 1, 0, 0, 0]);
        assert_eq!(search([0x12, 0x00])[5..], [0, 0, 0, 0, 0]);
    }
}
