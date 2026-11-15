//! Encoders/decoders for the link and network formats the router handles
//! itself (TCP segments are left to `smoltcp`).
//!
//! Sources: Ethernet II framing (IEEE 802.3 clause 3.2.6: a type field of
//! at least 0x0600), ARP (RFC 826), IPv4 (RFC 791, fragmentation per §2.3
//! and §3.2), the Internet checksum (RFC 1071), UDP (RFC 768) and ICMP echo
//! (RFC 792). Frames are not padded to the 60-byte Ethernet minimum: they
//! travel over an emulated 802.11 link, which has no minimum length.

use std::net::{Ipv4Addr, SocketAddrV4};

/// Length of an Ethernet II header (destination, source, type).
pub(super) const ETH_HEADER: usize = 14;
/// EtherType of IPv4.
pub(super) const ETHERTYPE_IPV4: u16 = 0x0800;
/// EtherType of ARP.
pub(super) const ETHERTYPE_ARP: u16 = 0x0806;
/// IP protocol number of ICMP.
pub(super) const PROTO_ICMP: u8 = 1;
/// IP protocol number of TCP.
pub(super) const PROTO_TCP: u8 = 6;
/// IP protocol number of UDP.
pub(super) const PROTO_UDP: u8 = 17;
/// The all-ones Ethernet broadcast address.
pub(super) const BROADCAST_MAC: [u8; 6] = [0xff; 6];
/// IP MTU of the console's link.
pub(super) const MTU: usize = 1500;
/// ARP operation: request.
pub(super) const ARP_REQUEST: u16 = 1;
/// ARP operation: reply.
pub(super) const ARP_REPLY: u16 = 2;
/// TCP header flag bits (RFC 9293 §3.1).
#[cfg(test)]
pub(super) const TCP_FIN: u8 = 0x01;
pub(super) const TCP_SYN: u8 = 0x02;
pub(super) const TCP_RST: u8 = 0x04;
pub(super) const TCP_ACK: u8 = 0x10;

/// Minimum IPv4 header length.
const IPV4_HEADER: usize = 20;
/// Time to live of every IPv4 packet the router originates.
const TTL: u8 = 64;

pub(super) fn be16(b: &[u8]) -> u16 {
    u16::from_be_bytes([b[0], b[1]])
}

#[cfg(test)]
pub(super) fn be32(b: &[u8]) -> u32 {
    u32::from_be_bytes([b[0], b[1], b[2], b[3]])
}

pub(super) fn ipv4_at(b: &[u8]) -> Ipv4Addr {
    Ipv4Addr::new(b[0], b[1], b[2], b[3])
}

/// A decoded Ethernet II frame (the destination MAC is not needed: the
/// router accepts frames addressed to anyone).
pub(super) struct Ethernet<'a> {
    pub src: [u8; 6],
    pub ethertype: u16,
    pub payload: &'a [u8],
}

pub(super) fn parse_ethernet(f: &[u8]) -> Option<Ethernet<'_>> {
    if f.len() < ETH_HEADER {
        return None;
    }
    Some(Ethernet {
        src: f[6..12].try_into().ok()?,
        ethertype: be16(&f[12..14]),
        payload: &f[ETH_HEADER..],
    })
}

/// Builds an Ethernet II frame.
pub(super) fn ethernet(dst: [u8; 6], src: [u8; 6], ethertype: u16, payload: &[u8]) -> Vec<u8> {
    let mut f = Vec::with_capacity(ETH_HEADER + payload.len());
    f.extend_from_slice(&dst);
    f.extend_from_slice(&src);
    f.extend_from_slice(&ethertype.to_be_bytes());
    f.extend_from_slice(payload);
    f
}

/// An Ethernet/IPv4 ARP packet (RFC 826).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Arp {
    pub op: u16,
    pub sha: [u8; 6],
    pub spa: Ipv4Addr,
    pub tha: [u8; 6],
    pub tpa: Ipv4Addr,
}

impl Arp {
    /// Decodes an Ethernet (htype 1) / IPv4 (ptype 0x0800) ARP packet.
    pub(super) fn parse(p: &[u8]) -> Option<Arp> {
        if p.len() < 28
            || be16(&p[0..2]) != 1
            || be16(&p[2..4]) != ETHERTYPE_IPV4
            || p[4] != 6
            || p[5] != 4
        {
            return None;
        }
        Some(Arp {
            op: be16(&p[6..8]),
            sha: p[8..14].try_into().ok()?,
            spa: ipv4_at(&p[14..18]),
            tha: p[18..24].try_into().ok()?,
            tpa: ipv4_at(&p[24..28]),
        })
    }

    pub(super) fn encode(&self) -> [u8; 28] {
        let mut p = [0u8; 28];
        p[0..2].copy_from_slice(&1u16.to_be_bytes());
        p[2..4].copy_from_slice(&ETHERTYPE_IPV4.to_be_bytes());
        p[4] = 6;
        p[5] = 4;
        p[6..8].copy_from_slice(&self.op.to_be_bytes());
        p[8..14].copy_from_slice(&self.sha);
        p[14..18].copy_from_slice(&self.spa.octets());
        p[18..24].copy_from_slice(&self.tha);
        p[24..28].copy_from_slice(&self.tpa.octets());
        p
    }
}

/// The Internet checksum (RFC 1071) over the concatenation of `parts`.
/// Verifying a block that contains its own checksum yields 0.
pub(super) fn checksum(parts: &[&[u8]]) -> u16 {
    let mut sum: u64 = 0;
    let mut high = true;
    for &b in parts.iter().flat_map(|p| p.iter()) {
        sum += if high {
            u64::from(b) << 8
        } else {
            u64::from(b)
        };
        high = !high;
    }
    while sum >> 16 != 0 {
        sum = (sum & 0xffff) + (sum >> 16);
    }
    !(sum as u16)
}

/// The TCP/UDP pseudo-header (RFC 768, RFC 9293 §3.1).
pub(super) fn pseudo_header(src: Ipv4Addr, dst: Ipv4Addr, protocol: u8, len: usize) -> [u8; 12] {
    let mut p = [0u8; 12];
    p[0..4].copy_from_slice(&src.octets());
    p[4..8].copy_from_slice(&dst.octets());
    p[9] = protocol;
    p[10..12].copy_from_slice(&(len as u16).to_be_bytes());
    p
}

/// A decoded IPv4 packet (options skipped, padding beyond the total length
/// stripped).
pub(super) struct Ipv4<'a> {
    pub src: Ipv4Addr,
    pub dst: Ipv4Addr,
    pub protocol: u8,
    /// Whether this is a fragment (MF set or non-zero offset).
    pub fragment: bool,
    pub payload: &'a [u8],
}

/// Decodes an IPv4 packet, verifying version, lengths and header checksum.
pub(super) fn parse_ipv4(p: &[u8]) -> Option<Ipv4<'_>> {
    if p.len() < IPV4_HEADER || p[0] >> 4 != 4 {
        return None;
    }
    let ihl = usize::from(p[0] & 0x0f) * 4;
    let total = usize::from(be16(&p[2..4]));
    if ihl < IPV4_HEADER || total < ihl || total > p.len() || checksum(&[&p[..ihl]]) != 0 {
        return None;
    }
    Some(Ipv4 {
        src: ipv4_at(&p[12..16]),
        dst: ipv4_at(&p[16..20]),
        protocol: p[9],
        fragment: be16(&p[6..8]) & 0x3fff != 0,
        payload: &p[ihl..total],
    })
}

/// Builds the Ethernet frames carrying one IPv4 datagram, fragmenting it to
/// [`MTU`] when needed (RFC 791 §3.2: every fragment but the last carries a
/// multiple of 8 payload bytes). DF is clear so fragmentation stays legal.
pub(super) fn ipv4_frames(
    dst_mac: [u8; 6],
    src_mac: [u8; 6],
    src: Ipv4Addr,
    dst: Ipv4Addr,
    protocol: u8,
    id: u16,
    payload: &[u8],
) -> Vec<Vec<u8>> {
    let max = (MTU - IPV4_HEADER) & !7;
    let mut frames = Vec::with_capacity(payload.len().div_ceil(max).max(1));
    let mut offset = 0;
    loop {
        let end = (offset + max).min(payload.len());
        let more = end < payload.len();
        let chunk = &payload[offset..end];
        let mut h = [0u8; IPV4_HEADER];
        h[0] = 0x45;
        h[2..4].copy_from_slice(&((IPV4_HEADER + chunk.len()) as u16).to_be_bytes());
        h[4..6].copy_from_slice(&id.to_be_bytes());
        let frag = (offset / 8) as u16 | if more { 0x2000 } else { 0 };
        h[6..8].copy_from_slice(&frag.to_be_bytes());
        h[8] = TTL;
        h[9] = protocol;
        h[12..16].copy_from_slice(&src.octets());
        h[16..20].copy_from_slice(&dst.octets());
        let sum = checksum(&[&h]);
        h[10..12].copy_from_slice(&sum.to_be_bytes());
        let mut f = Vec::with_capacity(ETH_HEADER + IPV4_HEADER + chunk.len());
        f.extend_from_slice(&dst_mac);
        f.extend_from_slice(&src_mac);
        f.extend_from_slice(&ETHERTYPE_IPV4.to_be_bytes());
        f.extend_from_slice(&h);
        f.extend_from_slice(chunk);
        frames.push(f);
        offset = end;
        if !more {
            return frames;
        }
    }
}

/// A decoded UDP datagram.
pub(super) struct Udp<'a> {
    pub src_port: u16,
    pub dst_port: u16,
    pub payload: &'a [u8],
}

/// Decodes a UDP header; the checksum is not verified (the link is an
/// in-memory hand-off from the WiFi model).
pub(super) fn parse_udp(p: &[u8]) -> Option<Udp<'_>> {
    if p.len() < 8 {
        return None;
    }
    let len = usize::from(be16(&p[4..6]));
    if len < 8 || len > p.len() {
        return None;
    }
    Some(Udp {
        src_port: be16(&p[0..2]),
        dst_port: be16(&p[2..4]),
        payload: &p[8..len],
    })
}

/// Largest UDP payload that fits in one IPv4 datagram.
pub(super) const MAX_UDP_PAYLOAD: usize = 65535 - IPV4_HEADER - 8;

/// Builds a UDP datagram (header + `data`) with its checksum. `data` must not
/// exceed [`MAX_UDP_PAYLOAD`].
pub(super) fn udp(src: SocketAddrV4, dst: SocketAddrV4, data: &[u8]) -> Vec<u8> {
    let len = 8 + data.len();
    let mut d = Vec::with_capacity(len);
    d.extend_from_slice(&src.port().to_be_bytes());
    d.extend_from_slice(&dst.port().to_be_bytes());
    d.extend_from_slice(&(len as u16).to_be_bytes());
    d.extend_from_slice(&[0, 0]);
    d.extend_from_slice(data);
    let pseudo = pseudo_header(*src.ip(), *dst.ip(), PROTO_UDP, len);
    let sum = match checksum(&[&pseudo, &d]) {
        0 => 0xffff,
        s => s,
    };
    d[6..8].copy_from_slice(&sum.to_be_bytes());
    d
}

/// The ports and flags of a TCP segment.
pub(super) struct TcpHeader {
    pub src_port: u16,
    pub dst_port: u16,
    pub flags: u8,
}

pub(super) fn parse_tcp_header(p: &[u8]) -> Option<TcpHeader> {
    if p.len() < 20 {
        return None;
    }
    Some(TcpHeader {
        src_port: be16(&p[0..2]),
        dst_port: be16(&p[2..4]),
        flags: p[13],
    })
}

/// Turns a valid ICMP echo request (type 8, code 0) into the matching echo
/// reply (type 0), keeping identifier, sequence number and data.
pub(super) fn icmp_echo_reply(req: &[u8]) -> Option<Vec<u8>> {
    if req.len() < 8 || req[0] != 8 || req[1] != 0 || checksum(&[req]) != 0 {
        return None;
    }
    let mut r = req.to_vec();
    r[0] = 0;
    r[2..4].fill(0);
    let sum = checksum(&[&r]);
    r[2..4].copy_from_slice(&sum.to_be_bytes());
    Some(r)
}
