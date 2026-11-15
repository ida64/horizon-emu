//! DNS helpers: choosing the host's upstream resolver and reading the query
//! name for the trace. Queries themselves are relayed byte-for-byte by the
//! UDP/TCP NAT, so the DNS message format (RFC 1035 §4.1) is only parsed far
//! enough to name the first question.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};

/// Port of DNS servers.
pub(super) const PORT: u16 = 53;

/// Resolver used when `/etc/resolv.conf` names none (Google Public DNS).
const FALLBACK: SocketAddr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(8, 8, 8, 8)), PORT);

/// The host's resolver: the first `nameserver` line of `/etc/resolv.conf`
/// whose address parses (resolv.conf(5); macOS keeps that file in sync with
/// its DNS configuration), port 53, else [`FALLBACK`]. IPv6 resolvers are
/// used as-is. Entries with a zone suffix (`fe80::1%en0`) do not parse and
/// are skipped.
pub(super) fn host_resolver() -> SocketAddr {
    std::fs::read_to_string("/etc/resolv.conf")
        .ok()
        .and_then(|text| parse_resolv_conf(&text))
        .map_or(FALLBACK, |ip| SocketAddr::new(ip, PORT))
}

/// The first parsable `nameserver` address in resolv.conf-formatted text.
pub(super) fn parse_resolv_conf(text: &str) -> Option<IpAddr> {
    text.lines().find_map(|line| {
        let mut words = line.split_whitespace();
        (words.next() == Some("nameserver"))
            .then(|| words.next()?.parse().ok())
            .flatten()
    })
}

/// The name and type of the first question of a DNS query, for the trace.
/// Returns `None` for responses, question-less or malformed messages.
pub(super) fn query_name(msg: &[u8]) -> Option<(String, &'static str)> {
    if msg.len() < 12 || msg[2] & 0x80 != 0 || u16::from_be_bytes([msg[4], msg[5]]) == 0 {
        return None;
    }
    let mut name = String::new();
    let mut i = 12;
    loop {
        let len = usize::from(*msg.get(i)?);
        i += 1;
        if len == 0 {
            break;
        }
        // Compression pointers never appear in the first question name.
        if len & 0xc0 != 0 || name.len() + len > 255 {
            return None;
        }
        if !name.is_empty() {
            name.push('.');
        }
        name.extend(msg.get(i..i + len)?.iter().map(|&b| {
            if b.is_ascii_graphic() {
                char::from(b)
            } else {
                '?'
            }
        }));
        i += len;
    }
    if name.is_empty() {
        name.push('.');
    }
    let qtype = match u16::from_be_bytes(msg.get(i..i + 2)?.try_into().ok()?) {
        1 => "A",
        2 => "NS",
        5 => "CNAME",
        6 => "SOA",
        12 => "PTR",
        15 => "MX",
        16 => "TXT",
        28 => "AAAA",
        33 => "SRV",
        64 => "SVCB",
        65 => "HTTPS",
        255 => "ANY",
        _ => "other",
    };
    Some((name, qtype))
}
