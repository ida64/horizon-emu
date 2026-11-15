//! Single-lease DHCP server (RFC 2131, options per RFC 2132).
//!
//! Every client is offered [`CONSOLE_IP`] (there is exactly one console on
//! the emulated link), so no lease database is kept. Replies carry the
//! options a generic client (lwIP, BSD dhclient, Nintendo's nifm) needs:
//! 53 message type, 54 server identifier ([`GATEWAY`]), 51 lease time
//! (86400 s) with 58/59 renewal/rebinding times (T1 = 50 %, T2 = 87.5 %,
//! RFC 2131 §4.4.5), 1 subnet mask, 3 router, 6 DNS server and 28 broadcast
//! address. Messages are padded to the 300-octet BOOTP minimum (RFC 1542
//! §2.1) that some clients enforce.
//!
//! Message handling (RFC 2131 §4.3):
//! * DISCOVER → OFFER.
//! * REQUEST → ACK if the requested address (option 50, else `ciaddr`) is
//!   [`CONSOLE_IP`], NAK otherwise; ignored if option 54 names another
//!   server (the client chose someone else).
//! * INFORM → ACK with configuration options but no address or lease time.
//! * RELEASE / DECLINE → no reply (nothing to free).
//!
//! Reply destination (RFC 2131 §4.1, relay agents never present): NAK is
//! broadcast; a non-zero `ciaddr` gets a unicast to it; otherwise the
//! broadcast flag selects a broadcast, and a clear flag a unicast to `yiaddr`
//! at the client's hardware address.

use std::net::Ipv4Addr;

use super::wire::{BROADCAST_MAC, be16};
use super::{CONSOLE_IP, DNS, GATEWAY, NETMASK, SUBNET_BROADCAST};

/// UDP port of DHCP servers.
pub(super) const SERVER_PORT: u16 = 67;
/// UDP port of DHCP clients.
pub(super) const CLIENT_PORT: u16 = 68;

/// Lease time handed out, in seconds.
const LEASE_SECS: u32 = 86_400;
/// RFC 2132 §9.3 "magic cookie" preceding the options.
const MAGIC_COOKIE: [u8; 4] = [99, 130, 83, 99];
/// Offset of the options field (after the fixed BOOTP header and cookie).
const OPTIONS: usize = 240;
/// Minimum BOOTP message size (RFC 1542 §2.1).
const MIN_MESSAGE: usize = 300;
/// Broadcast bit of the `flags` field.
const FLAG_BROADCAST: u16 = 0x8000;

const DISCOVER: u8 = 1;
const OFFER: u8 = 2;
const REQUEST: u8 = 3;
const DECLINE: u8 = 4;
const ACK: u8 = 5;
const NAK: u8 = 6;
const RELEASE: u8 = 7;
const INFORM: u8 = 8;

/// A DHCP message to send from `GATEWAY:67` to `dst_ip:68`.
pub(super) struct Reply {
    pub payload: Vec<u8>,
    pub dst_ip: Ipv4Addr,
    pub dst_mac: [u8; 6],
}

/// Result of processing one client message.
pub(super) struct Outcome {
    pub reply: Option<Reply>,
    /// Trace line, emitted for leases (ACKs carrying an address) only.
    pub trace: Option<String>,
}

/// The options of a client message the server cares about.
#[derive(Default)]
struct ClientOptions {
    message_type: Option<u8>,
    requested_ip: Option<Ipv4Addr>,
    server_id: Option<Ipv4Addr>,
}

fn parse_options(mut o: &[u8]) -> ClientOptions {
    let mut opts = ClientOptions::default();
    while let Some((&code, rest)) = o.split_first() {
        match code {
            0 => {
                o = rest;
                continue;
            }
            255 => break,
            _ => {}
        }
        let Some((&len, rest)) = rest.split_first() else {
            break;
        };
        let Some(value) = rest.get(..usize::from(len)) else {
            break;
        };
        match (code, value) {
            (53, [t]) => opts.message_type = Some(*t),
            (50, [a, b, c, d]) => opts.requested_ip = Some(Ipv4Addr::new(*a, *b, *c, *d)),
            (54, [a, b, c, d]) => opts.server_id = Some(Ipv4Addr::new(*a, *b, *c, *d)),
            _ => {}
        }
        o = &rest[usize::from(len)..];
    }
    opts
}

/// Processes a DHCP message received on port 67 (`msg` is the UDP payload).
/// Returns `None` for anything that is not a client message for this server.
pub(super) fn handle(msg: &[u8]) -> Option<Outcome> {
    // op BOOTREQUEST, htype Ethernet, hlen 6.
    if msg.len() < OPTIONS
        || msg[0] != 1
        || msg[1] != 1
        || msg[2] != 6
        || msg[236..240] != MAGIC_COOKIE
    {
        return None;
    }
    let flags = be16(&msg[10..12]);
    let ciaddr = Ipv4Addr::new(msg[12], msg[13], msg[14], msg[15]);
    let chaddr: [u8; 6] = msg[28..34].try_into().ok()?;
    let opts = parse_options(&msg[OPTIONS..]);

    let (kind, yiaddr) = match opts.message_type? {
        DISCOVER => (OFFER, CONSOLE_IP),
        REQUEST => {
            if opts.server_id.is_some_and(|s| s != GATEWAY) {
                return None;
            }
            let wanted = opts
                .requested_ip
                .or((!ciaddr.is_unspecified()).then_some(ciaddr));
            if wanted == Some(CONSOLE_IP) {
                (ACK, CONSOLE_IP)
            } else {
                (NAK, Ipv4Addr::UNSPECIFIED)
            }
        }
        INFORM => (ACK, Ipv4Addr::UNSPECIFIED),
        RELEASE | DECLINE => {
            return Some(Outcome {
                reply: None,
                trace: None,
            });
        }
        _ => return None,
    };

    let (dst_ip, dst_mac) = if kind == NAK || (yiaddr.is_unspecified() && ciaddr.is_unspecified()) {
        (Ipv4Addr::BROADCAST, BROADCAST_MAC)
    } else if !ciaddr.is_unspecified() {
        (ciaddr, chaddr)
    } else if flags & FLAG_BROADCAST != 0 {
        (Ipv4Addr::BROADCAST, BROADCAST_MAC)
    } else {
        (yiaddr, chaddr)
    };

    let trace = (kind == ACK && !yiaddr.is_unspecified()).then(|| {
        let m = chaddr;
        format!(
            "dhcp: leased {yiaddr} to {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x} for {LEASE_SECS} s",
            m[0], m[1], m[2], m[3], m[4], m[5]
        )
    });
    let reply = Reply {
        payload: build(msg, kind, ciaddr, yiaddr),
        dst_ip,
        dst_mac,
    };
    Some(Outcome {
        reply: Some(reply),
        trace,
    })
}

/// Builds a server message answering `req` (RFC 2131 §4.3.1 table 3).
fn build(req: &[u8], kind: u8, ciaddr: Ipv4Addr, yiaddr: Ipv4Addr) -> Vec<u8> {
    let mut m = vec![0u8; OPTIONS];
    m[0] = 2; // BOOTREPLY
    m[1] = 1;
    m[2] = 6;
    m[4..8].copy_from_slice(&req[4..8]); // xid
    m[10..12].copy_from_slice(&req[10..12]); // flags
    if kind == ACK {
        m[12..16].copy_from_slice(&ciaddr.octets());
    }
    m[16..20].copy_from_slice(&yiaddr.octets());
    m[24..28].copy_from_slice(&req[24..28]); // giaddr
    m[28..44].copy_from_slice(&req[28..44]); // chaddr
    m[236..240].copy_from_slice(&MAGIC_COOKIE);

    m.extend_from_slice(&[53, 1, kind]);
    m.extend_from_slice(&[54, 4]);
    m.extend_from_slice(&GATEWAY.octets());
    if kind != NAK {
        if !yiaddr.is_unspecified() {
            for (code, secs) in [
                (51, LEASE_SECS),
                (58, LEASE_SECS / 2),
                (59, LEASE_SECS / 8 * 7),
            ] {
                m.extend_from_slice(&[code, 4]);
                m.extend_from_slice(&secs.to_be_bytes());
            }
        }
        for (code, addr) in [(1, NETMASK), (3, GATEWAY), (6, DNS), (28, SUBNET_BROADCAST)] {
            m.extend_from_slice(&[code, 4]);
            m.extend_from_slice(&addr.octets());
        }
    }
    m.push(255);
    if m.len() < MIN_MESSAGE {
        m.resize(MIN_MESSAGE, 0);
    }
    m
}
