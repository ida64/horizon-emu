//! Router tests using only localhost: frames are built by hand, the
//! "internet" is local sockets reached through `Settings::loopback_alias`
//! (and a fake upstream resolver for DNS).

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, TcpListener, TcpStream, UdpSocket};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::wire::{
    self, ARP_REPLY, ARP_REQUEST, Arp, BROADCAST_MAC, ETHERTYPE_ARP, PROTO_ICMP, PROTO_TCP,
    PROTO_UDP, TCP_ACK, TCP_FIN, TCP_RST, TCP_SYN,
};
use super::{CONSOLE_IP, DNS, GATEWAY, ROUTER_MAC, Router, Settings};

const CONSOLE_MAC: [u8; 6] = [0x98, 0xb6, 0xe9, 0x12, 0x34, 0x56];
/// Console-visible address mapped to the host's 127.0.0.1.
const ALIAS: Ipv4Addr = Ipv4Addr::new(192, 0, 2, 1);
const TIMEOUT: Duration = Duration::from_secs(5);
const TCP_PSH: u8 = 0x08;

struct Harness {
    router: Arc<Router>,
    frames: Receiver<Vec<u8>>,
    trace: Arc<Mutex<Vec<String>>>,
}

fn harness(dns_upstream: SocketAddr) -> Harness {
    let (tx, frames) = mpsc::channel();
    let router = Router::with_settings(
        Box::new(move |f| {
            let _ = tx.send(f);
        }),
        Settings {
            dns_upstream,
            loopback_alias: Some(ALIAS),
        },
    );
    let trace = Arc::new(Mutex::new(Vec::new()));
    let lines = trace.clone();
    router.set_trace(Box::new(move |l| lines.lock().unwrap().push(l.to_owned())));
    Harness {
        router,
        frames,
        trace,
    }
}

fn unused_dns() -> SocketAddr {
    SocketAddr::from((Ipv4Addr::LOCALHOST, 9))
}

impl Harness {
    /// Waits for the first delivered frame satisfying `pred`, skipping others.
    fn expect<T>(&self, what: &str, mut pred: impl FnMut(&[u8]) -> Option<T>) -> T {
        let deadline = Instant::now() + TIMEOUT;
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            let frame = self
                .frames
                .recv_timeout(left)
                .unwrap_or_else(|_| panic!("timed out waiting for {what}"));
            if let Some(v) = pred(&frame) {
                return v;
            }
        }
    }

    fn traced(&self, needle: &str) -> bool {
        let deadline = Instant::now() + TIMEOUT;
        while Instant::now() < deadline {
            if self
                .trace
                .lock()
                .unwrap()
                .iter()
                .any(|l| l.contains(needle))
            {
                return true;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        false
    }
}

fn console(port: u16) -> SocketAddrV4 {
    SocketAddrV4::new(CONSOLE_IP, port)
}

fn arp_frame(op: u16, spa: Ipv4Addr, tpa: Ipv4Addr) -> Vec<u8> {
    let arp = Arp {
        op,
        sha: CONSOLE_MAC,
        spa,
        tha: [0; 6],
        tpa,
    };
    wire::ethernet(BROADCAST_MAC, CONSOLE_MAC, ETHERTYPE_ARP, &arp.encode())
}

fn udp_frame(dst_mac: [u8; 6], src: SocketAddrV4, dst: SocketAddrV4, data: &[u8]) -> Vec<u8> {
    let seg = wire::udp(src, dst, data);
    wire::ipv4_frames(
        dst_mac,
        CONSOLE_MAC,
        *src.ip(),
        *dst.ip(),
        PROTO_UDP,
        7,
        &seg,
    )
    .remove(0)
}

/// A UDP datagram for the console: (source, destination, data).
fn parse_udp_frame(frame: &[u8]) -> Option<(SocketAddrV4, SocketAddrV4, Vec<u8>, [u8; 6])> {
    let eth = wire::parse_ethernet(frame)?;
    let ip = wire::parse_ipv4(eth.payload)?;
    if ip.protocol != PROTO_UDP {
        return None;
    }
    let pseudo = wire::pseudo_header(ip.src, ip.dst, PROTO_UDP, ip.payload.len());
    assert_eq!(wire::checksum(&[&pseudo, ip.payload]), 0, "UDP checksum");
    let udp = wire::parse_udp(ip.payload)?;
    Some((
        SocketAddrV4::new(ip.src, udp.src_port),
        SocketAddrV4::new(ip.dst, udp.dst_port),
        udp.payload.to_vec(),
        dst_mac(frame),
    ))
}

fn dst_mac(frame: &[u8]) -> [u8; 6] {
    frame[..6].try_into().unwrap()
}

fn tcp_frame(
    src: SocketAddrV4,
    dst: SocketAddrV4,
    seq: u32,
    ack: u32,
    flags: u8,
    data: &[u8],
) -> Vec<u8> {
    let mut t = vec![0u8; 20];
    t[0..2].copy_from_slice(&src.port().to_be_bytes());
    t[2..4].copy_from_slice(&dst.port().to_be_bytes());
    t[4..8].copy_from_slice(&seq.to_be_bytes());
    t[8..12].copy_from_slice(&ack.to_be_bytes());
    t[12] = 5 << 4;
    t[13] = flags;
    t[14..16].copy_from_slice(&65535u16.to_be_bytes());
    t.extend_from_slice(data);
    let pseudo = wire::pseudo_header(*src.ip(), *dst.ip(), PROTO_TCP, t.len());
    let sum = wire::checksum(&[&pseudo, &t]);
    t[16..18].copy_from_slice(&sum.to_be_bytes());
    wire::ipv4_frames(
        ROUTER_MAC,
        CONSOLE_MAC,
        *src.ip(),
        *dst.ip(),
        PROTO_TCP,
        9,
        &t,
    )
    .remove(0)
}

struct Segment {
    src: SocketAddrV4,
    dst: SocketAddrV4,
    seq: u32,
    ack: u32,
    flags: u8,
    data: Vec<u8>,
}

fn parse_tcp(frame: &[u8]) -> Option<Segment> {
    let eth = wire::parse_ethernet(frame)?;
    let ip = wire::parse_ipv4(eth.payload)?;
    if ip.protocol != PROTO_TCP {
        return None;
    }
    assert_eq!(
        dst_mac(frame),
        CONSOLE_MAC,
        "TCP frame to the console's MAC"
    );
    let pseudo = wire::pseudo_header(ip.src, ip.dst, PROTO_TCP, ip.payload.len());
    assert_eq!(wire::checksum(&[&pseudo, ip.payload]), 0, "TCP checksum");
    let t = ip.payload;
    let header = usize::from(t[12] >> 4) * 4;
    Some(Segment {
        src: SocketAddrV4::new(ip.src, wire::be16(&t[0..2])),
        dst: SocketAddrV4::new(ip.dst, wire::be16(&t[2..4])),
        seq: wire::be32(&t[4..8]),
        ack: wire::be32(&t[8..12]),
        flags: t[13],
        data: t[header..].to_vec(),
    })
}

#[test]
fn arp_answers_gateway_and_dns_only() {
    let h = harness(unused_dns());
    // RFC 5227 probe for the console's own address: must stay unanswered.
    h.router
        .send(&arp_frame(ARP_REQUEST, Ipv4Addr::UNSPECIFIED, CONSOLE_IP));
    for target in [GATEWAY, DNS] {
        h.router.send(&arp_frame(ARP_REQUEST, CONSOLE_IP, target));
        let reply = h.expect("ARP reply", |f| {
            let eth = wire::parse_ethernet(f)?;
            (eth.ethertype == ETHERTYPE_ARP).then(|| (dst_mac(f), Arp::parse(eth.payload).unwrap()))
        });
        assert_eq!(reply.0, CONSOLE_MAC);
        assert_eq!(
            reply.1,
            Arp {
                op: ARP_REPLY,
                sha: ROUTER_MAC,
                spa: target,
                tha: CONSOLE_MAC,
                tpa: CONSOLE_IP,
            }
        );
    }
}

#[test]
fn icmp_echo_to_gateway() {
    let h = harness(unused_dns());
    let mut echo = vec![8, 0, 0, 0, 0x12, 0x34, 0x00, 0x01, b'p', b'i', b'n', b'g'];
    let sum = wire::checksum(&[&echo]);
    echo[2..4].copy_from_slice(&sum.to_be_bytes());
    let frame = wire::ipv4_frames(
        ROUTER_MAC,
        CONSOLE_MAC,
        CONSOLE_IP,
        GATEWAY,
        PROTO_ICMP,
        3,
        &echo,
    )
    .remove(0);
    h.router.send(&frame);
    let reply = h.expect("echo reply", |f| {
        let ip = wire::parse_ipv4(wire::parse_ethernet(f)?.payload)?;
        (ip.protocol == PROTO_ICMP && ip.src == GATEWAY && ip.dst == CONSOLE_IP)
            .then(|| ip.payload.to_vec())
    });
    assert_eq!(reply[0], 0);
    assert_eq!(wire::checksum(&[&reply]), 0);
    assert_eq!(reply[4..], echo[4..]);
}

/// A BOOTREQUEST from `CONSOLE_MAC` (RFC 2131 §2).
fn dhcp_request(kind: u8, xid: u32, flags: u16, ciaddr: Ipv4Addr, opts: &[(u8, &[u8])]) -> Vec<u8> {
    let mut m = vec![0u8; 240];
    m[0] = 1;
    m[1] = 1;
    m[2] = 6;
    m[4..8].copy_from_slice(&xid.to_be_bytes());
    m[10..12].copy_from_slice(&flags.to_be_bytes());
    m[12..16].copy_from_slice(&ciaddr.octets());
    m[28..34].copy_from_slice(&CONSOLE_MAC);
    m[236..240].copy_from_slice(&[99, 130, 83, 99]);
    m.extend_from_slice(&[53, 1, kind]);
    for (code, value) in opts {
        m.push(*code);
        m.push(value.len() as u8);
        m.extend_from_slice(value);
    }
    m.extend_from_slice(&[55, 4, 1, 3, 6, 28, 255]);
    m
}

struct DhcpReply {
    ip_dst: Ipv4Addr,
    mac_dst: [u8; 6],
    xid: u32,
    ciaddr: Ipv4Addr,
    yiaddr: Ipv4Addr,
    options: Vec<(u8, Vec<u8>)>,
}

impl DhcpReply {
    fn option(&self, code: u8) -> Option<&[u8]> {
        self.options
            .iter()
            .find(|(c, _)| *c == code)
            .map(|(_, v)| v.as_slice())
    }
}

fn send_dhcp(h: &Harness, msg: &[u8], src_ip: Ipv4Addr) -> DhcpReply {
    let src = SocketAddrV4::new(src_ip, 68);
    let dst = SocketAddrV4::new(Ipv4Addr::BROADCAST, 67);
    h.router.send(&udp_frame(BROADCAST_MAC, src, dst, msg));
    h.expect("DHCP reply", |f| {
        let (src, dst, m, mac_dst) = parse_udp_frame(f)?;
        assert_eq!(src, SocketAddrV4::new(GATEWAY, 67));
        assert_eq!(dst.port(), 68);
        assert!(m.len() >= 300, "BOOTP minimum size");
        assert_eq!(&m[..3], &[2, 1, 6]);
        assert_eq!(&m[28..34], &CONSOLE_MAC);
        assert_eq!(&m[236..240], &[99, 130, 83, 99]);
        let mut options = Vec::new();
        let mut o = &m[240..];
        while let [code, rest @ ..] = o {
            match code {
                0 => o = rest,
                255 => break,
                _ => {
                    let len = usize::from(rest[0]);
                    options.push((*code, rest[1..1 + len].to_vec()));
                    o = &rest[1 + len..];
                }
            }
        }
        Some(DhcpReply {
            ip_dst: *dst.ip(),
            mac_dst,
            xid: wire::be32(&m[4..8]),
            ciaddr: wire::ipv4_at(&m[12..16]),
            yiaddr: wire::ipv4_at(&m[16..20]),
            options,
        })
    })
}

#[test]
fn dhcp_discover_request_inform() {
    let h = harness(unused_dns());
    let none = Ipv4Addr::UNSPECIFIED;

    let offer = send_dhcp(&h, &dhcp_request(1, 0x1111, 0x8000, none, &[]), none);
    assert_eq!(
        (offer.ip_dst, offer.mac_dst),
        (Ipv4Addr::BROADCAST, BROADCAST_MAC)
    );
    assert_eq!(offer.xid, 0x1111);
    assert_eq!(offer.yiaddr, CONSOLE_IP);
    assert_eq!(offer.option(53), Some(&[2][..]));
    assert_eq!(offer.option(54), Some(&GATEWAY.octets()[..]));
    assert_eq!(offer.option(51), Some(&86_400u32.to_be_bytes()[..]));
    assert_eq!(offer.option(1), Some(&[255, 255, 255, 0][..]));
    assert_eq!(offer.option(3), Some(&GATEWAY.octets()[..]));
    assert_eq!(offer.option(6), Some(&DNS.octets()[..]));
    assert_eq!(offer.option(28), Some(&[10, 0, 2, 255][..]));

    // Broadcast flag clear: unicast to yiaddr at the client's MAC.
    let ack = send_dhcp(
        &h,
        &dhcp_request(
            3,
            0x2222,
            0,
            none,
            &[(50, &CONSOLE_IP.octets()), (54, &GATEWAY.octets())],
        ),
        none,
    );
    assert_eq!((ack.ip_dst, ack.mac_dst), (CONSOLE_IP, CONSOLE_MAC));
    assert_eq!(ack.xid, 0x2222);
    assert_eq!(ack.yiaddr, CONSOLE_IP);
    assert_eq!(ack.option(53), Some(&[5][..]));
    assert_eq!(ack.option(51), Some(&86_400u32.to_be_bytes()[..]));
    assert_eq!(ack.option(6), Some(&DNS.octets()[..]));
    assert!(h.traced("dhcp: leased 10.0.2.15 to 98:b6:e9:12:34:56"));

    // Another address: NAK, broadcast.
    let other = Ipv4Addr::new(10, 0, 2, 99);
    let nak = send_dhcp(
        &h,
        &dhcp_request(3, 0x3333, 0, none, &[(50, &other.octets())]),
        none,
    );
    assert_eq!(
        (nak.ip_dst, nak.mac_dst),
        (Ipv4Addr::BROADCAST, BROADCAST_MAC)
    );
    assert_eq!(nak.option(53), Some(&[6][..]));
    assert_eq!(nak.yiaddr, none);

    // RELEASE is silent; the following INFORM gets the next reply.
    let release = dhcp_request(7, 0x4444, 0, CONSOLE_IP, &[(54, &GATEWAY.octets())]);
    h.router.send(&udp_frame(
        ROUTER_MAC,
        console(68),
        SocketAddrV4::new(GATEWAY, 67),
        &release,
    ));
    let inform = send_dhcp(&h, &dhcp_request(8, 0x5555, 0, CONSOLE_IP, &[]), CONSOLE_IP);
    assert_eq!(inform.xid, 0x5555);
    assert_eq!((inform.ip_dst, inform.mac_dst), (CONSOLE_IP, CONSOLE_MAC));
    assert_eq!(inform.ciaddr, CONSOLE_IP);
    assert_eq!(inform.yiaddr, none);
    assert_eq!(inform.option(53), Some(&[5][..]));
    assert_eq!(inform.option(51), None);
    assert_eq!(inform.option(3), Some(&GATEWAY.octets()[..]));
}

#[test]
fn udp_nat_round_trip() {
    let echo = UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
    let port = echo.local_addr().unwrap().port();
    let server = std::thread::spawn(move || {
        echo.set_read_timeout(Some(TIMEOUT)).unwrap();
        let mut buf = [0u8; 64];
        let (n, from) = echo.recv_from(&mut buf).unwrap();
        let mut reply = b"echo:".to_vec();
        reply.extend_from_slice(&buf[..n]);
        echo.send_to(&reply, from).unwrap();
    });
    let h = harness(unused_dns());
    let remote = SocketAddrV4::new(ALIAS, port);
    h.router
        .send(&udp_frame(ROUTER_MAC, console(40000), remote, b"ping"));
    let (src, dst, data, mac) = h.expect("UDP reply", parse_udp_frame);
    assert_eq!((src, dst, mac), (remote, console(40000), CONSOLE_MAC));
    assert_eq!(data, b"echo:ping");
    assert!(h.traced(&format!("udp: new flow 10.0.2.15:40000 -> {remote}")));
    server.join().unwrap();
}

#[test]
fn dns_forwarding_keeps_id_and_ports() {
    let upstream = UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
    let h = harness(upstream.local_addr().unwrap());
    // Query id 0xbeef, RD, one question: example.com A IN.
    let mut query = vec![0xbe, 0xef, 0x01, 0x00, 0, 1, 0, 0, 0, 0, 0, 0];
    query.extend_from_slice(b"\x07example\x03com\x00\x00\x01\x00\x01");
    h.router.send(&udp_frame(
        ROUTER_MAC,
        console(41000),
        SocketAddrV4::new(DNS, 53),
        &query,
    ));

    upstream.set_read_timeout(Some(TIMEOUT)).unwrap();
    let mut buf = [0u8; 512];
    let (n, from) = upstream.recv_from(&mut buf).unwrap();
    assert_eq!(&buf[..n], &query[..]);
    let mut answer = query.clone();
    answer[2] = 0x81;
    answer[3] = 0x80;
    upstream.send_to(&answer, from).unwrap();

    let (src, dst, data, _) = h.expect("DNS answer", parse_udp_frame);
    assert_eq!((src, dst), (SocketAddrV4::new(DNS, 53), console(41000)));
    assert_eq!(data, answer);
    assert!(h.traced("example.com A"));
}

/// A connection to a forwarded host port: the router opens one to the
/// console from the gateway and carries data both ways; one the console
/// refuses is reset on the host side.
#[test]
fn forwarded_port_reaches_the_console() {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
    let host_port = listener.local_addr().unwrap();
    let h = harness(unused_dns());
    h.router.forward_from(listener, 5000).unwrap();
    // The router learns the console's MAC from any frame it sends.
    h.router.send(&arp_frame(ARP_REQUEST, CONSOLE_IP, GATEWAY));
    let to_console = |h: &Harness, what: &str, flags: u8| {
        h.expect(what, |f| {
            parse_tcp(f).filter(|s| s.dst == console(5000) && s.flags & flags == flags)
        })
    };

    let mut client = TcpStream::connect(host_port).unwrap();
    client.set_read_timeout(Some(TIMEOUT)).unwrap();
    let syn = to_console(&h, "SYN", TCP_SYN);
    assert_eq!(*syn.src.ip(), GATEWAY);
    let gateway = syn.src;
    let mut rcv = syn.seq.wrapping_add(1);
    h.router.send(&tcp_frame(
        console(5000),
        gateway,
        7000,
        rcv,
        TCP_SYN | TCP_ACK,
        &[],
    ));
    assert!(h.traced("-> 10.0.2.15:5000 forwarded"));

    client.write_all(b"hello").unwrap();
    let mut data = Vec::new();
    while data.len() < 5 {
        let seg = to_console(&h, "data", TCP_ACK);
        if seg.seq == rcv {
            rcv = rcv.wrapping_add(seg.data.len() as u32);
            data.extend(seg.data);
        }
    }
    assert_eq!(data, b"hello");
    h.router.send(&tcp_frame(
        console(5000),
        gateway,
        7001,
        rcv,
        TCP_ACK | TCP_PSH,
        b"world",
    ));
    let mut got = [0u8; 5];
    client.read_exact(&mut got).unwrap();
    assert_eq!(&got, b"world");

    // A second connection, which the console refuses.
    let mut refused = TcpStream::connect(host_port).unwrap();
    refused.set_read_timeout(Some(TIMEOUT)).unwrap();
    let syn = h.expect("second SYN", |f| {
        parse_tcp(f).filter(|s| s.dst == console(5000) && s.flags == TCP_SYN && s.src != gateway)
    });
    h.router.send(&tcp_frame(
        console(5000),
        syn.src,
        0,
        syn.seq.wrapping_add(1),
        TCP_RST | TCP_ACK,
        &[],
    ));
    let mut rest = Vec::new();
    assert!(matches!(refused.read_to_end(&mut rest), Err(_) | Ok(0)));
}

#[test]
fn tcp_nat_round_trip() {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
    let remote = SocketAddrV4::new(ALIAS, listener.local_addr().unwrap().port());
    let me = console(50000);
    let h = harness(unused_dns());
    let from_router = |h: &Harness, what: &str| {
        h.expect(what, |f| {
            parse_tcp(f).filter(|s| s.src == remote && s.dst == me)
        })
    };

    h.router.send(&tcp_frame(me, remote, 1000, 0, TCP_SYN, &[]));
    let synack = from_router(&h, "SYN-ACK");
    assert_eq!(
        synack.flags & (TCP_SYN | TCP_ACK | TCP_RST),
        TCP_SYN | TCP_ACK
    );
    assert_eq!(synack.ack, 1001);
    let mut rcv = synack.seq.wrapping_add(1);
    let (mut server, _) = listener.accept().unwrap();
    server.set_read_timeout(Some(TIMEOUT)).unwrap();
    assert!(h.traced("tcp: new flow 10.0.2.15:50000"));

    // Console → server.
    h.router
        .send(&tcp_frame(me, remote, 1001, rcv, TCP_ACK, &[]));
    h.router.send(&tcp_frame(
        me,
        remote,
        1001,
        rcv,
        TCP_ACK | TCP_PSH,
        b"hello",
    ));
    let mut got = [0u8; 5];
    server.read_exact(&mut got).unwrap();
    assert_eq!(&got, b"hello");

    // Server → console, then the server's FIN.
    server.write_all(b"world").unwrap();
    server.shutdown(std::net::Shutdown::Write).unwrap();
    let mut data = Vec::new();
    let mut fin = false;
    while !fin {
        let seg = from_router(&h, "data/FIN");
        assert_eq!(seg.flags & TCP_RST, 0, "unexpected RST");
        if seg.seq == rcv && !seg.data.is_empty() {
            data.extend_from_slice(&seg.data);
            rcv = rcv.wrapping_add(seg.data.len() as u32);
        }
        if seg.flags & TCP_FIN != 0 && seg.seq.wrapping_add(seg.data.len() as u32) == rcv {
            rcv = rcv.wrapping_add(1);
            fin = true;
        }
    }
    assert_eq!(data, b"world");
    assert_eq!(rcv, synack.seq.wrapping_add(1 + 5 + 1));

    // The console acknowledges and sends its own FIN.
    h.router
        .send(&tcp_frame(me, remote, 1006, rcv, TCP_ACK, &[]));
    h.router
        .send(&tcp_frame(me, remote, 1006, rcv, TCP_FIN | TCP_ACK, &[]));
    let mut rest = Vec::new();
    server.read_to_end(&mut rest).unwrap();
    assert!(rest.is_empty());
    let last = h.expect("ACK of the console's FIN", |f| {
        parse_tcp(f).filter(|s| s.src == remote && s.ack == 1007 && s.flags & TCP_ACK != 0)
    });
    assert_eq!(last.flags & TCP_RST, 0);
}

/// Opens a connection from `me` through the router; returns the server side
/// and the next sequence number expected from the router.
fn tcp_open(
    h: &Harness,
    listener: &TcpListener,
    me: SocketAddrV4,
    remote: SocketAddrV4,
) -> (TcpStream, u32) {
    h.router.send(&tcp_frame(me, remote, 1000, 0, TCP_SYN, &[]));
    let synack = h.expect("SYN-ACK", |f| {
        parse_tcp(f).filter(|s| s.src == remote && s.dst == me && s.flags & TCP_SYN != 0)
    });
    let rcv = synack.seq.wrapping_add(1);
    h.router
        .send(&tcp_frame(me, remote, 1001, rcv, TCP_ACK, &[]));
    let (server, _) = listener.accept().unwrap();
    server.set_read_timeout(Some(TIMEOUT)).unwrap();
    (server, rcv)
}

#[test]
fn tcp_bulk_download_through_window() {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
    let remote = SocketAddrV4::new(ALIAS, listener.local_addr().unwrap().port());
    let me = console(50002);
    let h = harness(unused_dns());
    let (mut server, mut rcv) = tcp_open(&h, &listener, me, remote);

    // Several times the 64 KiB buffers and the console's 64 KiB window.
    let payload: Vec<u8> = (0..300_000u32).map(|i| (i % 251) as u8).collect();
    let sent = payload.clone();
    let writer = std::thread::spawn(move || {
        server.write_all(&sent).unwrap();
        server.shutdown(std::net::Shutdown::Write).unwrap();
        server
    });
    let mut data = Vec::new();
    loop {
        let seg = h.expect("download", |f| parse_tcp(f).filter(|s| s.src == remote));
        assert_eq!(seg.flags & TCP_RST, 0, "unexpected RST");
        if seg.seq == rcv {
            data.extend_from_slice(&seg.data);
            rcv = rcv.wrapping_add(seg.data.len() as u32);
        }
        let fin = seg.flags & TCP_FIN != 0 && seg.seq.wrapping_add(seg.data.len() as u32) == rcv;
        if fin {
            rcv = rcv.wrapping_add(1);
        }
        h.router
            .send(&tcp_frame(me, remote, 1001, rcv, TCP_ACK, &[]));
        if fin {
            break;
        }
    }
    assert!(
        data == payload,
        "downloaded {} of {} bytes intact",
        data.len(),
        payload.len()
    );
    drop(writer.join().unwrap());
}

#[test]
fn tcp_concurrent_flows_stay_separate() {
    const FLOWS: u16 = 8;
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
    let remote = SocketAddrV4::new(ALIAS, listener.local_addr().unwrap().port());
    let h = harness(unused_dns());
    // All SYNs at once: the host connects complete in any order.
    for i in 0..FLOWS {
        h.router.send(&tcp_frame(
            console(51000 + i),
            remote,
            1000,
            0,
            TCP_SYN,
            &[],
        ));
    }
    let mut rcv = HashMap::new();
    while rcv.len() < usize::from(FLOWS) {
        let synack = h.expect("SYN-ACKs", |f| {
            parse_tcp(f).filter(|s| s.src == remote && s.flags & TCP_SYN != 0)
        });
        assert_eq!(synack.ack, 1001);
        rcv.insert(synack.dst.port(), synack.seq.wrapping_add(1));
    }
    // Echo servers: whatever a connection sends comes back on it.
    let servers = std::thread::spawn(move || {
        let echoes: Vec<_> = (0..FLOWS)
            .map(|_| {
                let (mut s, _) = listener.accept().unwrap();
                std::thread::spawn(move || {
                    s.set_read_timeout(Some(TIMEOUT)).unwrap();
                    let mut buf = [0u8; 10];
                    s.read_exact(&mut buf).unwrap();
                    s.write_all(&buf).unwrap();
                    s
                })
            })
            .collect();
        echoes
            .into_iter()
            .map(|e| e.join().unwrap())
            .collect::<Vec<_>>()
    });
    for (&port, &ack) in &rcv {
        let me = console(port);
        h.router
            .send(&tcp_frame(me, remote, 1001, ack, TCP_ACK, &[]));
        h.router.send(&tcp_frame(
            me,
            remote,
            1001,
            ack,
            TCP_ACK | TCP_PSH,
            format!("flow-{port}").as_bytes(),
        ));
    }
    let mut echoed = HashMap::new();
    while echoed.len() < usize::from(FLOWS) {
        let seg = h.expect("echoes", |f| {
            parse_tcp(f).filter(|s| s.src == remote && !s.data.is_empty())
        });
        echoed.insert(seg.dst.port(), String::from_utf8(seg.data).unwrap());
    }
    for (port, data) in echoed {
        assert_eq!(data, format!("flow-{port}"));
    }
    drop(servers.join().unwrap());
}

#[test]
fn tcp_connect_failure_answers_rst() {
    let closed = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
    let remote = SocketAddrV4::new(ALIAS, closed.local_addr().unwrap().port());
    drop(closed);
    let me = console(50001);
    let h = harness(unused_dns());
    h.router.send(&tcp_frame(me, remote, 5000, 0, TCP_SYN, &[]));
    let rst = h.expect("RST", |f| {
        parse_tcp(f).filter(|s| s.src == remote && s.dst == me)
    });
    assert_ne!(rst.flags & TCP_RST, 0);
    assert_eq!(rst.flags & TCP_SYN, 0);
    assert_eq!(rst.ack, 5001);
    assert!(h.traced("failed"));
}

#[test]
fn http_trace_lines() {
    let lines = Arc::new(Mutex::new(Vec::<String>::new()));
    let sink = lines.clone();
    let trace: super::TraceFn = Box::new(move |l| sink.lock().unwrap().push(l.to_owned()));
    let mut http = super::engine::HttpTrace::new(SocketAddrV4::new(Ipv4Addr::new(1, 2, 3, 4), 80));
    http.request(b"GET /conntest HTTP/1.1\r\nUser-Agent: x\r\nhO", &trace);
    http.request(b"st: ctest.example\r\n\r\nbody", &trace);
    http.response(b"HTTP/1.1 200 ", &trace);
    http.response(b"OK\r\nX-Organization: Nintendo\r\n\r\nok", &trace);
    http.response(b"HTTP/1.1 500 later\r\n", &trace);
    assert_eq!(
        *lines.lock().unwrap(),
        [
            "http: 1.2.3.4:80 > GET /conntest HTTP/1.1 (Host: ctest.example)",
            "http: 1.2.3.4:80 < HTTP/1.1 200 OK",
        ]
    );
}

#[test]
fn resolv_conf_first_nameserver() {
    let text = "# comment\nsearch lan\nnameserver fe80::1%en0\nnameserver 2001:db8::1\nnameserver 192.168.1.1\n";
    assert_eq!(
        super::dns::parse_resolv_conf(text),
        Some("2001:db8::1".parse().unwrap())
    );
    assert_eq!(super::dns::parse_resolv_conf("search lan\n"), None);
}
