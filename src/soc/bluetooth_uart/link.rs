//! The controller's links to devices on the air: inquiry, paging and
//! connections, security on a connection, and the ACL data it carries, as
//! the host commands them over HCI and is told about them by events.
//!
//! Commands, events and their parameters are the Bluetooth Core
//! Specification's (Vol 4 Part E §7: link control, link policy and
//! baseband commands, events §7.7; ACL data packets §5.4.2) as public
//! Linux `include/net/bluetooth/hci.h` transcribes them; supported-command
//! bits and event-mask bits as listed by public BlueZ `monitor/packet.c`.
//!
//! On the air there is at most one device, the [`ProController`]; a
//! connection is with it or with nothing. What happens on the air between
//! the two link managers (paging, LMP authentication and the Secure Simple
//! Pairing exchange) is not modeled bit for bit: its outcome is decided
//! here from what each side holds, and reported with the events a
//! controller reports it with. The times it takes are assumptions, named
//! where they are set.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use super::{l2cap, remote::ProController};

/// Link commands: opcode, parameter length (`usize::MAX`: variable,
/// checked where it is executed), and the bit of the supported commands
/// mask that announces it (octet * 8 + bit, BlueZ's numbering).
pub(super) const COMMANDS: &[(u16, usize, usize)] = &[
    (0x0401, 5, 0),           // Inquiry
    (0x0402, 0, 1),           // Inquiry Cancel
    (0x0405, 13, 4),          // Create Connection
    (0x0406, 3, 5),           // Disconnect
    (0x0409, 7, 8),           // Accept Connection Request
    (0x040a, 7, 9),           // Reject Connection Request
    (0x040b, 22, 10),         // Link Key Request Reply
    (0x040c, 6, 11),          // Link Key Request Negative Reply
    (0x040f, 4, 14),          // Change Connection Packet Type
    (0x0411, 2, 15),          // Authentication Requested
    (0x0413, 3, 16),          // Set Connection Encryption
    (0x0419, 10, 19),         // Remote Name Request
    (0x041b, 2, 21),          // Read Remote Supported Features
    (0x041c, 3, 22),          // Read Remote Extended Features
    (0x041d, 2, 23),          // Read Remote Version Information
    (0x041f, 2, 24),          // Read Clock Offset
    (0x042b, 9, 151),         // IO Capability Request Reply
    (0x042c, 6, 152),         // User Confirmation Request Reply
    (0x042d, 6, 153),         // User Confirmation Request Negative Reply
    (0x0803, 10, 34),         // Sniff Mode
    (0x0804, 2, 35),          // Exit Sniff Mode
    (0x0809, 2, 39),          // Role Discovery
    (0x080b, 7, 40),          // Switch Role
    (0x080d, 4, 42),          // Write Link Policy Settings
    (0x0811, 8, 140),         // Sniff Subrating
    (0x0c01, 8, 46),          // Set Event Mask
    (0x0c05, usize::MAX, 48), // Set Event Filter
    (0x0c16, 2, 58),          // Write Connection Accept Timeout
    (0x0c18, 2, 60),          // Write Page Timeout
    (0x0c1a, 1, 63),          // Write Scan Enable
    (0x0c33, 7, 86),          // Host Buffer Size
    (0x0c37, 4, 89),          // Write Link Supervision Timeout
    (0x0c45, 1, 101),         // Write Inquiry Mode
    (0x0c63, 8, 178),         // Set Event Mask Page 2
    (0x0c7b, 2, 260),         // Read Authenticated Payload Timeout
    (0x0c7c, 4, 261),         // Write Authenticated Payload Timeout
    (0x1405, 2, 125),         // Read RSSI
    (0x1408, 2, 164),         // Read Encryption Key Size
];

/// HCI error codes (Vol 2 Part D).
const SUCCESS: u8 = 0x00;
const UNKNOWN_CONNECTION: u8 = 0x02;
const PAGE_TIMEOUT: u8 = 0x04;
const AUTHENTICATION_FAILURE: u8 = 0x05;
const KEY_MISSING: u8 = 0x06;
const CONNECTION_EXISTS: u8 = 0x0b;
const COMMAND_DISALLOWED: u8 = 0x0c;
const ACCEPT_TIMEOUT: u8 = 0x10;
const REMOTE_POWER_OFF: u8 = 0x15;
const LOCAL_HOST_TERMINATED: u8 = 0x16;
const REMOTE_USER_TERMINATED: u8 = 0x13;

/// Events (§7.7).
const INQUIRY_COMPLETE: u8 = 0x01;
const INQUIRY_RESULT: u8 = 0x02;
const CONNECTION_COMPLETE: u8 = 0x03;
const CONNECTION_REQUEST: u8 = 0x04;
const DISCONNECTION_COMPLETE: u8 = 0x05;
const AUTHENTICATION_COMPLETE: u8 = 0x06;
const NAME_COMPLETE: u8 = 0x07;
const ENCRYPTION_CHANGE: u8 = 0x08;
const FEATURES_COMPLETE: u8 = 0x0b;
const VERSION_COMPLETE: u8 = 0x0c;
const COMMAND_COMPLETE: u8 = 0x0e;
const COMMAND_STATUS: u8 = 0x0f;
const ROLE_CHANGE: u8 = 0x12;
const COMPLETED_PACKETS: u8 = 0x13;
const MODE_CHANGE: u8 = 0x14;
const LINK_KEY_REQUEST: u8 = 0x17;
const LINK_KEY_NOTIFICATION: u8 = 0x18;
const CLOCK_OFFSET_COMPLETE: u8 = 0x1c;
const PACKET_TYPE_CHANGED: u8 = 0x1d;
const INQUIRY_RESULT_RSSI: u8 = 0x22;
const EXTENDED_FEATURES_COMPLETE: u8 = 0x23;
const SNIFF_SUBRATING: u8 = 0x2e;
const IO_CAPABILITY_REQUEST: u8 = 0x31;
const IO_CAPABILITY_RESPONSE: u8 = 0x32;
const USER_CONFIRMATION_REQUEST: u8 = 0x33;
const SIMPLE_PAIRING_COMPLETE: u8 = 0x36;
const HOST_FEATURES_NOTIFICATION: u8 = 0x3d;

/// The event mask after reset (§7.3.1).
const DEFAULT_EVENT_MASK: u64 = 0x0000_1fff_ffff_ffff;
/// Page timeout and connection accept timeout after reset, in slots of
/// 0.625 ms (§7.3.16, §7.3.14).
const DEFAULT_PAGE_TIMEOUT: u16 = 0x2000;
const DEFAULT_ACCEPT_TIMEOUT: u16 = 0x1f40;
/// The authenticated payload timeout of a new connection: 30 s in units of
/// 10 ms (§7.3.93).
const DEFAULT_PAYLOAD_TIMEOUT: u16 = 3000;
const SLOT: Duration = Duration::from_micros(625);
/// One unit of inquiry length (§7.1.1).
const INQUIRY_UNIT: Duration = Duration::from_millis(1280);
/// How soon an inquiry finds a discoverable device in range, and how long
/// a page of a connectable one takes \[assumption: both procedures take
/// a few train repetitions of 10 ms on a quiet channel\].
const INQUIRY_RESPONSE: Duration = Duration::from_millis(40);
const PAGE_TIME: Duration = Duration::from_millis(20);
/// The handle of the one ACL connection (any 12-bit value would do).
const HANDLE: u16 = 0x000b;
/// What the device's inquiry responses say of it: page scan repetition
/// mode R1, clock offset 0, and a strong signal of -40 dBm \[assumption:
/// a controller next to the console\].
const PAGE_SCAN_REPETITION: u8 = 0x01;
const RSSI: u8 = (-40i8) as u8;
/// Link key type from Secure Simple Pairing without MITM protection on
/// P-192 (§7.7.24): the device does not support Secure Connections.
const UNAUTHENTICATED_P192: u8 = 0x04;
/// The key size of an encrypted link: the largest, 16 octets.
const KEY_SIZE: u8 = 16;

/// An H4 event packet.
pub(super) fn event(code: u8, parameters: &[u8]) -> Vec<u8> {
    let mut packet = vec![4, code, parameters.len() as u8];
    packet.extend(parameters);
    packet
}

/// Command Complete with one command credit (§7.7.14).
pub(super) fn complete(op: u16, parameters: &[u8]) -> Vec<u8> {
    let mut all = vec![1, op as u8, (op >> 8) as u8];
    all.extend(parameters);
    event(COMMAND_COMPLETE, &all)
}

/// Command Status with one command credit (§7.7.15).
fn status(op: u16, status: u8) -> Vec<u8> {
    event(COMMAND_STATUS, &[status, 1, op as u8, (op >> 8) as u8])
}

/// What the link layer needs of the rest of the controller.
pub(super) struct Local<'a> {
    /// The controller's own address (HCI byte order).
    pub address: [u8; 6],
    /// The host enabled Secure Simple Pairing.
    pub simple_pairing: bool,
    /// The controller's random number generator.
    pub random: &'a mut dyn FnMut() -> u64,
}

/// Where Secure Simple Pairing or authentication stands on the connection.
#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum Security {
    Idle,
    /// Link Key Request sent; the host answers with a key or without.
    LinkKey,
    /// IO Capability Request sent.
    IoCapability,
    /// User Confirmation Request sent.
    Confirmation,
}

#[derive(Clone, Serialize, Deserialize)]
struct Connection {
    address: [u8; 6],
    /// The controller is the central (master) of the link.
    central: bool,
    authenticated: bool,
    encrypted: bool,
    security: Security,
    /// An L2CAP PDU arriving in fragments.
    assembling: Vec<u8>,
    /// The authenticated payload timeout, in units of 10 ms.
    payload_timeout: u16,
    /// The device started the authentication under way (it paged the
    /// console, and checks the console is the one it is paired with).
    by_device: bool,
}

/// Work the controller does at a moment of guest time.
#[derive(Clone, Serialize, Deserialize)]
enum Work {
    /// The inquiry under way reaches the devices in range.
    InquiryResponses,
    InquiryComplete,
    /// The page Create Connection started ends.
    Paged {
        address: [u8; 6],
    },
    /// A remote name request ends.
    Name {
        address: [u8; 6],
    },
    /// A device that paged the console authenticates it and encrypts the
    /// link before it opens its channels.
    DeviceAuthenticates,
}

/// A connection the device asked for, waiting for the host's answer.
#[derive(Clone, Serialize, Deserialize)]
struct Incoming {
    address: [u8; 6],
    deadline: Duration,
}

/// Everything the link layer holds: what a snapshot saves of it.
#[derive(Clone, Serialize, Deserialize)]
pub(super) struct Link {
    scan_enable: u8,
    inquiry_mode: u8,
    page_timeout: u16,
    accept_timeout: u16,
    event_mask: u64,
    /// Longest ACL packet the host takes (Host Buffer Size); 0 until
    /// told, which leaves packets whole.
    host_acl_length: u16,
    /// An inquiry is under way, with the responses it may take (0: no
    /// limit) and how many it took.
    inquiry: Option<(u8, u8)>,
    incoming: Option<Incoming>,
    connection: Option<Connection>,
    scheduled: Vec<(Duration, Work)>,
}

impl Default for Link {
    fn default() -> Self {
        Self {
            scan_enable: 0,
            inquiry_mode: 0,
            page_timeout: DEFAULT_PAGE_TIMEOUT,
            accept_timeout: DEFAULT_ACCEPT_TIMEOUT,
            event_mask: DEFAULT_EVENT_MASK,
            host_acl_length: 0,
            inquiry: None,
            incoming: None,
            connection: None,
            scheduled: Vec::new(),
        }
    }
}

type Refusal = String;

fn address(p: &[u8]) -> [u8; 6] {
    p[..6].try_into().unwrap()
}

fn handle(p: &[u8]) -> u16 {
    u16::from_le_bytes([p[0], p[1]]) & 0x0fff
}

impl Link {
    /// HCI Reset: the defaults, with no link left. A device the controller
    /// was linked to sees the link drop.
    pub(super) fn reset(&mut self, remote: Option<&mut ProController>) {
        if self.connection.is_some()
            && let Some(remote) = remote
        {
            remote.disconnected();
        }
        *self = Self::default();
    }

    /// Queues `event` for the host unless the host masked it (§7.3.1:
    /// bit `code - 1` of the mask for the events of page 1).
    fn send(&self, out: &mut Vec<u8>, code: u8, parameters: &[u8]) {
        let masked = (1..=64).contains(&code)
            && ![COMMAND_COMPLETE, COMMAND_STATUS].contains(&code)
            && self.event_mask & 1 << (code - 1) == 0;
        if !masked {
            out.extend(event(code, parameters));
        }
    }

    fn schedule(&mut self, at: Duration, work: Work) {
        self.scheduled.push((at, work));
    }

    /// The connection `handle` names, if it is the one there is.
    fn connection(&mut self, handle: u16) -> Option<&mut Connection> {
        self.connection.as_mut().filter(|_| handle == HANDLE)
    }

    /// Executes link command `op` if it is one, writing what the controller
    /// answers to `out`; `Ok(false)` for commands of other kinds.
    pub(super) fn command(
        &mut self,
        op: u16,
        p: &[u8],
        now: Duration,
        local: &mut Local,
        mut remote: Option<&mut ProController>,
        out: &mut Vec<u8>,
    ) -> Result<bool, Refusal> {
        let Some(&(_, length, _)) = COMMANDS.iter().find(|c| c.0 == op) else {
            return Ok(false);
        };
        let refusal = || {
            Err(format!(
                "Bluetooth HCI command {op:#06x} {p:02x?} is not modeled"
            ))
        };
        if length != usize::MAX && p.len() != length {
            return refusal();
        }
        let in_range = |remote: &Option<&mut ProController>, address: [u8; 6]| {
            remote.as_ref().is_some_and(|r| r.address() == address)
        };
        match op {
            // Inquiry: general or limited access code, 1 to 0x30 units of
            // 1.28 s, a response limit.
            0x0401 if (1..=0x30).contains(&p[3]) && p[2] == 0x9e && p[1] == 0x8b => {
                if self.inquiry.is_some() {
                    out.extend(status(op, COMMAND_DISALLOWED));
                    return Ok(true);
                }
                out.extend(status(op, SUCCESS));
                self.inquiry = Some((p[4], 0));
                self.schedule(now + INQUIRY_RESPONSE, Work::InquiryResponses);
                let length = INQUIRY_UNIT * u32::from(p[3]);
                self.schedule(now + length, Work::InquiryComplete);
            }
            0x0402 => {
                let was = self.inquiry.take().is_some();
                self.scheduled.retain(|(_, work)| {
                    !matches!(work, Work::InquiryResponses | Work::InquiryComplete)
                });
                out.extend(complete(
                    op,
                    &[if was { SUCCESS } else { COMMAND_DISALLOWED }],
                ));
            }
            // Create Connection: address, packet types, page scan
            // repetition mode, reserved, clock offset, role switch.
            0x0405 => {
                let target = address(p);
                if self.connection.is_some() || self.incoming.is_some() {
                    let code = match &self.connection {
                        Some(c) if c.address == target => CONNECTION_EXISTS,
                        _ => COMMAND_DISALLOWED,
                    };
                    out.extend(status(op, code));
                    return Ok(true);
                }
                out.extend(status(op, SUCCESS));
                let connectable =
                    in_range(&remote, target) && remote.as_ref().is_some_and(|r| r.connectable());
                let after = match connectable {
                    true => PAGE_TIME,
                    false => SLOT * u32::from(self.page_timeout),
                };
                self.schedule(now + after, Work::Paged { address: target });
            }
            0x0406 => {
                let handle = handle(p);
                if self.connection(handle).is_none() {
                    out.extend(status(op, UNKNOWN_CONNECTION));
                    return Ok(true);
                }
                out.extend(status(op, SUCCESS));
                self.drop_link(LOCAL_HOST_TERMINATED, remote, out);
            }
            // Accept Connection Request: address, role (0: become the
            // central, by a role switch).
            0x0409 | 0x040a => {
                let target = address(p);
                let Some(incoming) = self.incoming.take().filter(|i| i.address == target) else {
                    out.extend(status(op, UNKNOWN_CONNECTION));
                    return Ok(true);
                };
                out.extend(status(op, SUCCESS));
                if op == 0x040a {
                    // The reason the host gives is what Connection
                    // Complete reports.
                    let mut parameters = vec![p[6], 0, 0];
                    parameters.extend(incoming.address);
                    parameters.extend([1, 0]);
                    self.send(out, CONNECTION_COMPLETE, &parameters);
                    if let Some(remote) = remote {
                        remote.disconnected();
                    }
                    return Ok(true);
                }
                let central = p[6] == 0;
                if central {
                    let mut parameters = vec![SUCCESS];
                    parameters.extend(target);
                    parameters.push(0);
                    self.send(out, ROLE_CHANGE, &parameters);
                }
                self.connect(target, central, true, remote, out);
                self.schedule(now + PAGE_TIME, Work::DeviceAuthenticates);
            }
            0x040b | 0x040c => {
                let target = address(p);
                let local_address = local.address;
                let simple_pairing = local.simple_pairing;
                let Some(connection) = self
                    .connection
                    .as_mut()
                    .filter(|c| c.address == target && c.security == Security::LinkKey)
                else {
                    return refusal();
                };
                let mut answer = vec![SUCCESS];
                answer.extend(target);
                out.extend(complete(op, &answer));
                if op == 0x040b {
                    // LMP authentication with the key the host has: it
                    // succeeds when the device holds the same one.
                    let held = remote.as_ref().and_then(|r| r.link_key(local_address));
                    let result = match held {
                        Some(key) if key[..] == p[6..22] => SUCCESS,
                        Some(_) => AUTHENTICATION_FAILURE,
                        None => KEY_MISSING,
                    };
                    connection.security = Security::Idle;
                    connection.authenticated = result == SUCCESS;
                    if connection.by_device {
                        // The device asked: the host hears of the outcome
                        // as the device goes on, encrypting the link
                        // (Vol 2 Part F §4.1), or as it gives up.
                        connection.by_device = false;
                        if result != SUCCESS {
                            self.drop_link(result, remote, out);
                            return Ok(true);
                        }
                        connection.encrypted = true;
                        let mut parameters = vec![SUCCESS];
                        parameters.extend(HANDLE.to_le_bytes());
                        parameters.push(1);
                        self.send(out, ENCRYPTION_CHANGE, &parameters);
                        if let Some(remote) = remote {
                            remote.secured();
                        }
                        return Ok(true);
                    }
                    let mut parameters = vec![result];
                    parameters.extend(HANDLE.to_le_bytes());
                    self.send(out, AUTHENTICATION_COMPLETE, &parameters);
                } else if connection.by_device {
                    // The console has no key for a device that has one:
                    // the device does not pair again by itself
                    // \[assumption\]; it drops the link.
                    self.drop_link(KEY_MISSING, remote, out);
                } else if simple_pairing {
                    // No key: pairing. Both sides support Secure Simple
                    // Pairing, so it is that, and the host is asked what
                    // it can do.
                    connection.security = Security::IoCapability;
                    self.send(out, IO_CAPABILITY_REQUEST, &target);
                } else {
                    // Legacy pairing with a PIN is not modeled.
                    return refusal();
                }
            }
            0x0411 => {
                let Some(connection) = self.connection(handle(p)) else {
                    out.extend(status(op, UNKNOWN_CONNECTION));
                    return Ok(true);
                };
                if connection.security != Security::Idle {
                    return refusal();
                }
                connection.security = Security::LinkKey;
                let target = connection.address;
                out.extend(status(op, SUCCESS));
                self.send(out, LINK_KEY_REQUEST, &target);
            }
            0x0413 if p[2] <= 1 => {
                let Some(connection) = self.connection(handle(p)) else {
                    out.extend(status(op, UNKNOWN_CONNECTION));
                    return Ok(true);
                };
                // Encryption needs the link key of an authentication.
                if !connection.authenticated {
                    return refusal();
                }
                connection.encrypted = p[2] == 1;
                out.extend(status(op, SUCCESS));
                let mut parameters = vec![SUCCESS];
                parameters.extend(HANDLE.to_le_bytes());
                parameters.push(p[2]);
                self.send(out, ENCRYPTION_CHANGE, &parameters);
                if p[2] == 1
                    && let Some(remote) = remote
                {
                    remote.secured();
                }
            }
            // Change Connection Packet Type: handle, packet types. Both
            // devices take every type the host allows: nothing on the air
            // is timed by packet sizes here.
            0x040f => {
                if self.connection(handle(p)).is_none() {
                    out.extend(status(op, UNKNOWN_CONNECTION));
                    return Ok(true);
                }
                out.extend(status(op, SUCCESS));
                let mut parameters = vec![SUCCESS];
                parameters.extend(HANDLE.to_le_bytes());
                parameters.extend(&p[2..4]);
                self.send(out, PACKET_TYPE_CHANGED, &parameters);
            }
            // Remote Name Request: address, page scan repetition mode,
            // reserved, clock offset.
            0x0419 => {
                out.extend(status(op, SUCCESS));
                let target = address(p);
                let linked = self
                    .connection
                    .as_ref()
                    .is_some_and(|c| c.address == target);
                let reachable = in_range(&remote, target)
                    && (linked || remote.as_ref().is_some_and(|r| r.connectable()));
                let after = match reachable {
                    true => PAGE_TIME,
                    false => SLOT * u32::from(self.page_timeout),
                };
                self.schedule(now + after, Work::Name { address: target });
            }
            0x041b | 0x041c | 0x041d | 0x041f => {
                if self.connection(handle(p)).is_none() {
                    out.extend(status(op, UNKNOWN_CONNECTION));
                    return Ok(true);
                }
                out.extend(status(op, SUCCESS));
                let mut parameters = vec![SUCCESS];
                parameters.extend(HANDLE.to_le_bytes());
                let code = match op {
                    0x041b => {
                        parameters.extend(super::remote::FEATURES);
                        FEATURES_COMPLETE
                    }
                    0x041c => {
                        // Pages 0 and 1; there is no page beyond.
                        let page = p[2];
                        parameters.extend([page, 1]);
                        parameters.extend(match page {
                            0 => super::remote::FEATURES,
                            1 => super::remote::HOST_FEATURES,
                            _ => [0; 8],
                        });
                        EXTENDED_FEATURES_COMPLETE
                    }
                    0x041d => {
                        let (version, manufacturer, subversion) = super::remote::VERSION;
                        parameters.push(version);
                        parameters.extend(manufacturer.to_le_bytes());
                        parameters.extend(subversion.to_le_bytes());
                        VERSION_COMPLETE
                    }
                    _ => {
                        // The offset between the two clocks, which only
                        // matters to a later page: none.
                        parameters.extend([0, 0]);
                        CLOCK_OFFSET_COMPLETE
                    }
                };
                self.send(out, code, &parameters);
            }
            // IO Capability Request Reply: address, capability (0 to 3),
            // out-of-band data (0: none), authentication requirements.
            0x042b if p[6] <= 3 && p[7] == 0 => {
                let target = address(p);
                let Some(connection) = self
                    .connection
                    .as_mut()
                    .filter(|c| c.address == target && c.security == Security::IoCapability)
                else {
                    return refusal();
                };
                connection.security = Security::Confirmation;
                out.extend(complete(op, &[&[SUCCESS][..], &target].concat()));
                let mut response = target.to_vec();
                response.extend(super::remote::IO_CAPABILITY);
                self.send(out, IO_CAPABILITY_RESPONSE, &response);
                // The device can neither show nor enter a number, so the
                // association model is Just Works (Vol 3 Part C §5.2.2.6):
                // the host is asked to confirm, and may do so itself; the
                // number is what the comparison would have shown.
                let number = ((local.random)() % 1_000_000) as u32;
                let mut request = target.to_vec();
                request.extend(number.to_le_bytes());
                self.send(out, USER_CONFIRMATION_REQUEST, &request);
            }
            0x042c | 0x042d => {
                let target = address(p);
                let Some(connection) = self
                    .connection
                    .as_mut()
                    .filter(|c| c.address == target && c.security == Security::Confirmation)
                else {
                    return refusal();
                };
                connection.security = Security::Idle;
                out.extend(complete(op, &[&[SUCCESS][..], &target].concat()));
                let result = match op {
                    0x042c => SUCCESS,
                    _ => AUTHENTICATION_FAILURE,
                };
                connection.authenticated = result == SUCCESS;
                self.send(
                    out,
                    SIMPLE_PAIRING_COMPLETE,
                    &[&[result][..], &target].concat(),
                );
                if result == SUCCESS {
                    // The key both sides derive; its bits come from the
                    // controller's generator, as a real exchange's come
                    // from its random nonces and keys.
                    let mut key = [0u8; 16];
                    key[..8].copy_from_slice(&(local.random)().to_le_bytes());
                    key[8..].copy_from_slice(&(local.random)().to_le_bytes());
                    if let Some(remote) = remote.as_mut() {
                        remote.bond(local.address, key)?;
                    }
                    let mut notification = target.to_vec();
                    notification.extend(key);
                    notification.push(UNAUTHENTICATED_P192);
                    self.send(out, LINK_KEY_NOTIFICATION, &notification);
                }
                let mut parameters = vec![result];
                parameters.extend(HANDLE.to_le_bytes());
                self.send(out, AUTHENTICATION_COMPLETE, &parameters);
            }
            // Sniff Mode: handle, maximum and minimum interval, attempt,
            // timeout. The device takes the longest interval offered
            // \[assumption\].
            0x0803 | 0x0804 => {
                let handle = handle(p);
                if self.connection(handle).is_none() {
                    out.extend(status(op, UNKNOWN_CONNECTION));
                    return Ok(true);
                }
                out.extend(status(op, SUCCESS));
                let mut parameters = vec![SUCCESS];
                parameters.extend(HANDLE.to_le_bytes());
                match op {
                    0x0803 => {
                        parameters.push(2);
                        parameters.extend(&p[2..4]);
                    }
                    _ => parameters.extend([0, 0, 0]),
                }
                self.send(out, MODE_CHANGE, &parameters);
            }
            0x0809 => {
                let handle = handle(p);
                let Some(connection) = self.connection(handle) else {
                    out.extend(complete(op, &[UNKNOWN_CONNECTION, p[0], p[1], 0]));
                    return Ok(true);
                };
                let role = u8::from(!connection.central);
                out.extend(complete(op, &[SUCCESS, p[0], p[1], role]));
            }
            0x080b if p[6] <= 1 => {
                let target = address(p);
                let Some(connection) = self.connection.as_mut().filter(|c| c.address == target)
                else {
                    out.extend(status(op, UNKNOWN_CONNECTION));
                    return Ok(true);
                };
                connection.central = p[6] == 0;
                out.extend(status(op, SUCCESS));
                self.send(
                    out,
                    ROLE_CHANGE,
                    &[&[SUCCESS][..], &target, &[p[6]]].concat(),
                );
            }
            // Per-connection settings the device has no say in and that
            // change nothing modeled: link policy, link supervision
            // timeout (with no radio to lose the link to).
            0x080d | 0x0c37 => {
                let code = match self.connection(handle(p)) {
                    Some(_) => SUCCESS,
                    None => UNKNOWN_CONNECTION,
                };
                out.extend(complete(op, &[code, p[0], p[1]]));
            }
            // The authenticated payload timeout (§7.3.93, §7.3.94): how long
            // an encrypted link may go without a packet checked by the
            // encryption, 30 s unless the host says otherwise. It never
            // runs out here: the device sends often.
            0x0c7b | 0x0c7c => {
                let Some(connection) = self.connection(handle(p)) else {
                    let mut answer = vec![UNKNOWN_CONNECTION, p[0], p[1]];
                    answer.extend(if op == 0x0c7b { &[0, 0][..] } else { &[] });
                    out.extend(complete(op, &answer));
                    return Ok(true);
                };
                let mut answer = vec![SUCCESS, p[0], p[1]];
                match op {
                    0x0c7b => answer.extend(connection.payload_timeout.to_le_bytes()),
                    _ => connection.payload_timeout = u16::from_le_bytes([p[2], p[3]]),
                }
                out.extend(complete(op, &answer));
            }
            // Sniff Subrating: handle, maximum latency, minimum remote and
            // local timeouts. The device agrees to the latency asked
            // \[assumption\], which the event reports both ways.
            0x0811 => {
                let Some(_) = self.connection(handle(p)) else {
                    out.extend(complete(op, &[UNKNOWN_CONNECTION, p[0], p[1]]));
                    return Ok(true);
                };
                out.extend(complete(op, &[SUCCESS, p[0], p[1]]));
                let mut parameters = vec![SUCCESS];
                parameters.extend(HANDLE.to_le_bytes());
                parameters.extend(&p[2..4]);
                parameters.extend(&p[2..4]);
                parameters.extend(&p[4..8]);
                self.send(out, SNIFF_SUBRATING, &parameters);
            }
            0x1405 | 0x1408 => {
                let value = match op {
                    0x1405 => RSSI,
                    _ => KEY_SIZE,
                };
                let answer = match self.connection(handle(p)) {
                    Some(c) if op == 0x1405 || c.encrypted => [SUCCESS, p[0], p[1], value],
                    Some(_) => [COMMAND_DISALLOWED, p[0], p[1], 0],
                    None => [UNKNOWN_CONNECTION, p[0], p[1], 0],
                };
                out.extend(complete(op, &answer));
            }
            0x0c01 => {
                self.event_mask = u64::from_le_bytes(p.try_into().unwrap());
                out.extend(complete(op, &[SUCCESS]));
            }
            // Page 2 holds events of features not modeled (AMP, Connectionless
            // Slave Broadcast, authenticated payload timeout): kept for
            // nothing.
            0x0c63 => out.extend(complete(op, &[SUCCESS])),
            // Set Event Filter (§7.3.3): clear every filter (type 0), or
            // report inquiry results from every device (type 1, condition
            // 0), which is what the controller does without a filter.
            // Filters by class or address, and connection setup filters,
            // are not modeled.
            0x0c05 if p == [0] || p == [1, 0] => out.extend(complete(op, &[SUCCESS])),
            0x0c16 | 0x0c18 => {
                let slots = u16::from_le_bytes([p[0], p[1]]);
                if slots == 0 {
                    out.extend(complete(op, &[0x12]));
                    return Ok(true);
                }
                match op {
                    0x0c16 => self.accept_timeout = slots,
                    _ => self.page_timeout = slots,
                }
                out.extend(complete(op, &[SUCCESS]));
            }
            0x0c1a if p[0] <= 3 => {
                self.scan_enable = p[0];
                out.extend(complete(op, &[SUCCESS]));
            }
            0x0c45 if p[0] <= 2 => {
                self.inquiry_mode = p[0];
                out.extend(complete(op, &[SUCCESS]));
            }
            // Host Buffer Size: ACL length, SCO length, ACL and SCO counts.
            0x0c33 => {
                self.host_acl_length = u16::from_le_bytes([p[0], p[1]]);
                out.extend(complete(op, &[SUCCESS]));
            }
            _ => return refusal(),
        }
        Ok(true)
    }

    /// The connection is up: the host is told, and the device.
    fn connect(
        &mut self,
        address: [u8; 6],
        central: bool,
        device_paged: bool,
        remote: Option<&mut ProController>,
        out: &mut Vec<u8>,
    ) {
        self.connection = Some(Connection {
            address,
            central,
            authenticated: false,
            encrypted: false,
            security: Security::Idle,
            assembling: Vec::new(),
            payload_timeout: DEFAULT_PAYLOAD_TIMEOUT,
            by_device: false,
        });
        let mut parameters = vec![SUCCESS];
        parameters.extend(HANDLE.to_le_bytes());
        parameters.extend(address);
        parameters.extend([1, 0]);
        self.send(out, CONNECTION_COMPLETE, &parameters);
        if let Some(remote) = remote {
            remote.connected(device_paged);
        }
    }

    /// The link drops for `reason`: the host is told, and the device.
    fn drop_link(&mut self, reason: u8, remote: Option<&mut ProController>, out: &mut Vec<u8>) {
        if self.connection.take().is_none() {
            return;
        }
        let mut parameters = vec![SUCCESS];
        parameters.extend(HANDLE.to_le_bytes());
        parameters.push(reason);
        self.send(out, DISCONNECTION_COMPLETE, &parameters);
        if let Some(remote) = remote {
            remote.disconnected();
        }
    }

    /// The device dropped the link: its user pressed SYNC.
    pub(super) fn dropped_by_device(&mut self, out: &mut Vec<u8>) {
        self.drop_link(REMOTE_USER_TERMINATED, None, out);
    }

    /// An ACL data packet from the host: handle and flags, data.
    pub(super) fn acl(
        &mut self,
        header: u16,
        data: &[u8],
        remote: Option<&mut ProController>,
        out: &mut Vec<u8>,
    ) -> Result<(), Refusal> {
        let (handle, boundary, broadcast) = (header & 0x0fff, header >> 12 & 3, header >> 14);
        let refusal = |what: &str| {
            Err(format!(
                "Bluetooth ACL data {what} (handle {handle:#05x}, flags {:#x}) {data:02x?} is not modeled",
                header >> 12
            ))
        };
        if broadcast != 0 || boundary == 3 {
            return refusal("broadcast or of that kind");
        }
        // Data for a link that just went is lost on it, as on a real
        // controller.
        let Some(connection) = self.connection(handle) else {
            return Ok(());
        };
        match boundary {
            1 if !connection.assembling.is_empty() => connection.assembling.extend(data),
            0 | 2 if connection.assembling.is_empty() => connection.assembling = data.to_vec(),
            _ => return refusal("out of sequence"),
        }
        let pdu = match l2cap::pdu_length(&connection.assembling) {
            Some(length) if connection.assembling.len() == length => {
                std::mem::take(&mut connection.assembling)
            }
            Some(length) if connection.assembling.len() > length => {
                return refusal("longer than its PDU");
            }
            _ => Vec::new(),
        };
        // The packet has gone out on the air: its buffer is the host's
        // again.
        let mut completed = vec![1];
        completed.extend(HANDLE.to_le_bytes());
        completed.extend(1u16.to_le_bytes());
        self.send(out, COMPLETED_PACKETS, &completed);
        if pdu.is_empty() {
            return Ok(());
        }
        let Some(remote) = remote else {
            return Ok(());
        };
        for answer in remote.receive(&pdu)? {
            self.to_host(&answer, out);
        }
        Ok(())
    }

    /// An L2CAP PDU from the device, in ACL packets no longer than the
    /// host takes.
    fn to_host(&self, pdu: &[u8], out: &mut Vec<u8>) {
        let most = match self.host_acl_length {
            0 => pdu.len().max(1),
            length => usize::from(length),
        };
        for (index, fragment) in pdu.chunks(most).enumerate() {
            // First automatically flushable packet, then continuations.
            let flags: u16 = if index == 0 { 0x2000 } else { 0x1000 };
            out.push(2);
            out.extend((HANDLE | flags).to_le_bytes());
            out.extend((fragment.len() as u16).to_le_bytes());
            out.extend(fragment);
        }
    }

    /// What the controller does by guest time `now` on its own: scheduled
    /// work, and what the device does. Returns when it next has work.
    pub(super) fn due(
        &mut self,
        now: Duration,
        local: &mut Local,
        mut remote: Option<&mut ProController>,
        out: &mut Vec<u8>,
    ) -> Result<Option<Duration>, Refusal> {
        self.scheduled.sort_by_key(|(at, _)| *at);
        while let Some(&(at, _)) = self.scheduled.first()
            && at <= now
        {
            let (_, work) = self.scheduled.remove(0);
            self.work(work, now, remote.as_deref_mut(), out);
        }
        if let Some(incoming) = &self.incoming
            && incoming.deadline <= now
        {
            let address = incoming.address;
            self.incoming = None;
            let mut parameters = vec![ACCEPT_TIMEOUT, 0, 0];
            parameters.extend(address);
            parameters.extend([1, 0]);
            self.send(out, CONNECTION_COMPLETE, &parameters);
        }
        let mut next = self.scheduled.iter().map(|(at, _)| *at).min();
        if let Some(incoming) = &self.incoming {
            next = Some(next.map_or(incoming.deadline, |n| n.min(incoming.deadline)));
        }
        let Some(remote) = remote else {
            return Ok(next);
        };
        let due = remote.due(now, local.address)?;
        if due.disconnect {
            self.drop_link(REMOTE_POWER_OFF, Some(&mut *remote), out);
        }
        // A device pages the console: it is asked for if it scans for
        // pages and has no link already.
        if due.page
            && self.scan_enable & 2 != 0
            && self.connection.is_none()
            && self.incoming.is_none()
        {
            let address = remote.address();
            self.incoming = Some(Incoming {
                address,
                deadline: now + SLOT * u32::from(self.accept_timeout),
            });
            let mut parameters = address.to_vec();
            parameters.extend(super::remote::CLASS);
            parameters.push(1);
            self.send(out, CONNECTION_REQUEST, &parameters);
        }
        if self.connection.is_some() {
            for pdu in &due.pdus {
                self.to_host(pdu, out);
            }
        }
        if let Some(at) = due.next {
            next = Some(next.map_or(at, |n| n.min(at)));
        }
        Ok(next)
    }

    fn work(
        &mut self,
        work: Work,
        now: Duration,
        remote: Option<&mut ProController>,
        out: &mut Vec<u8>,
    ) {
        match work {
            Work::InquiryResponses => {
                let Some((limit, found)) = self.inquiry else {
                    return;
                };
                let Some(remote) = remote.filter(|r| r.discoverable()) else {
                    return;
                };
                let address = remote.address();
                let mut parameters = vec![1];
                parameters.extend(address);
                parameters.push(PAGE_SCAN_REPETITION);
                // Without an extended inquiry response from the device,
                // extended mode reports it as RSSI mode does (§7.7.38).
                let code = match self.inquiry_mode {
                    0 => {
                        parameters.extend([0, 0]);
                        parameters.extend(super::remote::CLASS);
                        parameters.extend([0, 0]);
                        INQUIRY_RESULT
                    }
                    _ => {
                        parameters.push(0);
                        parameters.extend(super::remote::CLASS);
                        parameters.extend([0, 0, RSSI]);
                        INQUIRY_RESULT_RSSI
                    }
                };
                self.send(out, code, &parameters);
                self.inquiry = Some((limit, found + 1));
                if limit != 0 && found + 1 >= limit {
                    self.scheduled
                        .retain(|(_, work)| !matches!(work, Work::InquiryComplete));
                    self.schedule(now, Work::InquiryComplete);
                }
            }
            Work::InquiryComplete => {
                if self.inquiry.take().is_some() {
                    self.send(out, INQUIRY_COMPLETE, &[SUCCESS]);
                }
            }
            Work::Paged { address } => {
                let answers = remote
                    .as_ref()
                    .is_some_and(|r| r.address() == address && r.connectable());
                if answers && self.connection.is_none() {
                    self.connect(address, true, false, remote, out);
                } else {
                    let mut parameters = vec![PAGE_TIMEOUT, 0, 0];
                    parameters.extend(address);
                    parameters.extend([1, 0]);
                    self.send(out, CONNECTION_COMPLETE, &parameters);
                }
            }
            Work::DeviceAuthenticates => {
                let Some(connection) = self
                    .connection
                    .as_mut()
                    .filter(|c| c.security == Security::Idle && !c.authenticated)
                else {
                    return;
                };
                connection.by_device = true;
                connection.security = Security::LinkKey;
                let address = connection.address;
                self.send(out, LINK_KEY_REQUEST, &address);
            }
            Work::Name { address } => {
                let linked = self
                    .connection
                    .as_ref()
                    .is_some_and(|c| c.address == address);
                let reachable = remote
                    .as_ref()
                    .is_some_and(|r| r.address() == address && (linked || r.connectable()));
                let mut parameters = vec![if reachable { SUCCESS } else { PAGE_TIMEOUT }];
                parameters.extend(address);
                let mut name = [0u8; 248];
                if reachable {
                    // Without a link, getting the name reads the device's
                    // host features too, which the host is told of first
                    // (§7.7.50).
                    if !linked {
                        let mut features = address.to_vec();
                        features.extend(super::remote::HOST_FEATURES);
                        self.send(out, HOST_FEATURES_NOTIFICATION, &features);
                    }
                    let bytes = super::remote::NAME.as_bytes();
                    name[..bytes.len()].copy_from_slice(bytes);
                }
                parameters.extend(name);
                self.send(out, NAME_COMPLETE, &parameters);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{clock::GuestClock, soc::joycon::Controls};

    const CONSOLE: [u8; 6] = [0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0];

    /// A link layer whose host wants every event (the default mask leaves
    /// out those of Secure Simple Pairing, §7.3.1).
    fn unmasked() -> Link {
        Link {
            event_mask: u64::MAX,
            ..Link::default()
        }
    }

    fn pro() -> ProController {
        pro_with_controls().0
    }

    fn pro_with_controls() -> (ProController, Arc<Controls>) {
        let memory = std::env::temp_dir().join(format!(
            "horizon-emu-link-{}-{:?}.json",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_file(&memory);
        let controls = Arc::new(Controls::default());
        let pro =
            ProController::new(Arc::clone(&controls), Arc::new(GuestClock::new()), memory).unwrap();
        (pro, controls)
    }

    /// Runs `op` and returns the events, as (code, parameters).
    fn run(
        link: &mut Link,
        op: u16,
        p: &[u8],
        now: Duration,
        pro: &mut ProController,
    ) -> Vec<(u8, Vec<u8>)> {
        let mut seed = 1u64;
        let mut random = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            seed
        };
        let mut local = Local {
            address: CONSOLE,
            simple_pairing: true,
            random: &mut random,
        };
        let mut out = Vec::new();
        assert!(
            link.command(op, p, now, &mut local, Some(pro), &mut out)
                .unwrap()
        );
        events(&out)
    }

    fn due(link: &mut Link, now: Duration, pro: &mut ProController) -> Vec<(u8, Vec<u8>)> {
        let mut random = || 7;
        let mut local = Local {
            address: CONSOLE,
            simple_pairing: true,
            random: &mut random,
        };
        let mut out = Vec::new();
        link.due(now, &mut local, Some(pro), &mut out).unwrap();
        events(&out)
    }

    fn events(mut out: &[u8]) -> Vec<(u8, Vec<u8>)> {
        let mut events = Vec::new();
        while let [4, code, length, rest @ ..] = out {
            let length = usize::from(*length);
            events.push((*code, rest[..length].to_vec()));
            out = &rest[length..];
        }
        assert!(out.is_empty(), "not only events: {out:02x?}");
        events
    }

    fn codes(events: &[(u8, Vec<u8>)]) -> Vec<u8> {
        events.iter().map(|(code, _)| *code).collect()
    }

    #[test]
    fn inquiry_finds_a_syncing_controller_and_its_name() {
        let (mut link, mut pro) = (unmasked(), pro());
        run(&mut link, 0x0c45, &[1], Duration::ZERO, &mut pro);
        // Not discoverable: the inquiry ends with nothing found.
        let inquiry = [0x33, 0x8b, 0x9e, 2, 0];
        let started = run(&mut link, 0x0401, &inquiry, Duration::ZERO, &mut pro);
        assert_eq!(started, [(COMMAND_STATUS, vec![0, 1, 1, 4])]);
        assert!(due(&mut link, Duration::from_secs(1), &mut pro).is_empty());
        let ended = due(&mut link, Duration::from_millis(2560), &mut pro);
        assert_eq!(ended, [(INQUIRY_COMPLETE, vec![0])]);

        pro.sync();
        run(&mut link, 0x0401, &inquiry, Duration::ZERO, &mut pro);
        let found = due(&mut link, Duration::from_millis(100), &mut pro);
        assert_eq!(codes(&found), [INQUIRY_RESULT_RSSI]);
        let result = &found[0].1;
        assert_eq!(result[..7], [&[1][..], &pro.address()].concat()[..]);
        assert_eq!(result[9..12], super::super::remote::CLASS);
        // Cancelled: no Inquiry Complete.
        let cancelled = run(&mut link, 0x0402, &[], Duration::ZERO, &mut pro);
        assert_eq!(cancelled, [(COMMAND_COMPLETE, vec![1, 2, 4, 0])]);
        assert!(due(&mut link, Duration::from_secs(10), &mut pro).is_empty());

        let mut request = pro.address().to_vec();
        request.extend([1, 0, 0, 0]);
        run(&mut link, 0x0419, &request, Duration::ZERO, &mut pro);
        let named = due(&mut link, Duration::from_secs(1), &mut pro);
        assert_eq!(codes(&named), [HOST_FEATURES_NOTIFICATION, NAME_COMPLETE]);
        assert_eq!(named[1].1.len(), 255);
        assert_eq!(&named[1].1[7..21], b"Pro Controller");
    }

    #[test]
    fn pairing_by_simple_pairing_gives_both_sides_the_key() {
        let (mut link, mut pro) = (unmasked(), pro());
        pro.sync();
        let mut create = pro.address().to_vec();
        create.extend([0x18, 0xcc, 1, 0, 0, 0, 1]);
        run(&mut link, 0x0405, &create, Duration::ZERO, &mut pro);
        let connected = due(&mut link, Duration::from_secs(1), &mut pro);
        assert_eq!(codes(&connected), [CONNECTION_COMPLETE]);
        assert_eq!(connected[0].1[..3], [0, HANDLE as u8, 0]);

        let handle = HANDLE.to_le_bytes();
        let address = pro.address();
        let asked = run(&mut link, 0x0411, &handle, Duration::ZERO, &mut pro);
        assert_eq!(codes(&asked), [COMMAND_STATUS, LINK_KEY_REQUEST]);
        let refused = run(&mut link, 0x040c, &address, Duration::ZERO, &mut pro);
        assert_eq!(codes(&refused), [COMMAND_COMPLETE, IO_CAPABILITY_REQUEST]);
        let mut capability = address.to_vec();
        capability.extend([1, 0, 4]);
        let answered = run(&mut link, 0x042b, &capability, Duration::ZERO, &mut pro);
        assert_eq!(
            codes(&answered),
            [
                COMMAND_COMPLETE,
                IO_CAPABILITY_RESPONSE,
                USER_CONFIRMATION_REQUEST
            ]
        );
        assert_eq!(answered[1].1[6..], [3, 0, 4]);
        let confirmed = run(&mut link, 0x042c, &address, Duration::ZERO, &mut pro);
        assert_eq!(
            codes(&confirmed),
            [
                COMMAND_COMPLETE,
                SIMPLE_PAIRING_COMPLETE,
                LINK_KEY_NOTIFICATION,
                AUTHENTICATION_COMPLETE
            ]
        );
        let key: [u8; 16] = confirmed[2].1[6..22].try_into().unwrap();
        assert_eq!(confirmed[2].1[22], UNAUTHENTICATED_P192);
        assert_eq!(pro.link_key(CONSOLE), Some(key));
        assert_eq!(confirmed[3].1, [0, HANDLE as u8, 0]);
        let encrypted = run(
            &mut link,
            0x0413,
            &[handle[0], handle[1], 1],
            Duration::ZERO,
            &mut pro,
        );
        assert_eq!(
            encrypted[1],
            (ENCRYPTION_CHANGE, vec![0, HANDLE as u8, 0, 1])
        );

        // Next time the host has the key, and the device agrees with it;
        // a wrong one fails.
        let mut reply = address.to_vec();
        reply.extend(key);
        run(&mut link, 0x0411, &handle, Duration::ZERO, &mut pro);
        let authenticated = run(&mut link, 0x040b, &reply, Duration::ZERO, &mut pro);
        assert_eq!(
            authenticated[1],
            (AUTHENTICATION_COMPLETE, vec![0, HANDLE as u8, 0])
        );
        reply[6] ^= 1;
        run(&mut link, 0x0411, &handle, Duration::ZERO, &mut pro);
        let failed = run(&mut link, 0x040b, &reply, Duration::ZERO, &mut pro);
        assert_eq!(failed[1].1[0], AUTHENTICATION_FAILURE);
    }

    #[test]
    fn a_woken_controller_pages_a_scanning_console_and_its_data_is_framed() {
        let (mut link, (mut pro, controls)) = (unmasked(), pro_with_controls());
        pro.bond(CONSOLE, [5; 16]).unwrap();
        // Asleep until a button wakes it; then it pages, but the console
        // does not scan for pages yet. Once it does, it is asked.
        assert!(due(&mut link, Duration::ZERO, &mut pro).is_empty());
        controls.press(crate::soc::joycon::Button::A, true);
        assert!(due(&mut link, Duration::ZERO, &mut pro).is_empty());
        run(&mut link, 0x0c1a, &[2], Duration::ZERO, &mut pro);
        let asked = due(&mut link, Duration::ZERO, &mut pro);
        assert_eq!(codes(&asked), [CONNECTION_REQUEST]);
        assert_eq!(asked[0].1[..6], pro.address());
        controls.press(crate::soc::joycon::Button::A, false);
        let mut accept = pro.address().to_vec();
        accept.push(0);
        let accepted = run(&mut link, 0x0409, &accept, Duration::ZERO, &mut pro);
        assert_eq!(
            codes(&accepted),
            [COMMAND_STATUS, ROLE_CHANGE, CONNECTION_COMPLETE]
        );

        // The device checks the console's key, then encrypts the link;
        // only then does it open its channels.
        let asked = due(&mut link, Duration::from_secs(1), &mut pro);
        assert_eq!(asked, [(LINK_KEY_REQUEST, pro.address().to_vec())]);
        let mut reply = pro.address().to_vec();
        reply.extend([5; 16]);
        let replied = run(&mut link, 0x040b, &reply, Duration::ZERO, &mut pro);
        assert_eq!(replied[1], (ENCRYPTION_CHANGE, vec![0, HANDLE as u8, 0, 1]));

        // Its connection request for HID control reaches the host in one
        // ACL packet, first and flushable.
        let mut out = Vec::new();
        let mut random = || 0;
        let mut local = Local {
            address: CONSOLE,
            simple_pairing: true,
            random: &mut random,
        };
        link.due(Duration::ZERO, &mut local, Some(&mut pro), &mut out)
            .unwrap();
        assert_eq!(out[..5], [2, HANDLE as u8, 0x20, 12, 0]);
        assert_eq!(
            out[5..],
            l2cap::signal(2, 1, &l2cap::words(&[0x11, 0x40]))[..]
        );

        // The host's answer in two fragments: one Number of Completed
        // Packets for each.
        let response = l2cap::signal(3, 1, &l2cap::words(&[0x0070, 0x0040, 1, 0]));
        let mut out = Vec::new();
        let header = HANDLE | 0x2000;
        link.acl(header, &response[..5], Some(&mut pro), &mut out)
            .unwrap();
        link.acl(HANDLE | 0x1000, &response[5..], Some(&mut pro), &mut out)
            .unwrap();
        assert_eq!(codes(&events(&out)), [COMPLETED_PACKETS, COMPLETED_PACKETS]);
        // A continuation with nothing to continue is out of sequence.
        assert!(
            link.acl(HANDLE | 0x1000, &[0], Some(&mut pro), &mut Vec::new())
                .is_err()
        );
        // The host's data to a handle gone is lost, without credit.
        let mut out = Vec::new();
        link.acl(0x0123 | 0x2000, &response, Some(&mut pro), &mut out)
            .unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn long_pdus_are_cut_to_the_hosts_acl_length() {
        let link = Link {
            host_acl_length: 4,
            ..Link::default()
        };
        let mut out = Vec::new();
        link.to_host(&[1, 2, 3, 4, 5, 6], &mut out);
        assert_eq!(
            out,
            [
                2,
                HANDLE as u8,
                0x20,
                4,
                0,
                1,
                2,
                3,
                4,
                2,
                HANDLE as u8,
                0x10,
                2,
                0,
                5,
                6
            ]
        );
    }

    #[test]
    fn masked_events_are_not_sent() {
        let (mut link, mut pro) = (Link::default(), pro());
        // Everything masked but Command Status (never masked).
        run(&mut link, 0x0c01, &[0; 8], Duration::ZERO, &mut pro);
        let mut create = pro.address().to_vec();
        create.extend([0x18, 0xcc, 1, 0, 0, 0, 1]);
        let started = run(&mut link, 0x0405, &create, Duration::ZERO, &mut pro);
        assert_eq!(codes(&started), [COMMAND_STATUS]);
        assert!(due(&mut link, Duration::from_secs(10), &mut pro).is_empty());
    }
}
