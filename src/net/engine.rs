//! The router thread: frame dispatch, host-socket multiplexing with
//! `poll(2)`, and the `smoltcp`-based TCP terminator.
//!
//! TCP design. `smoltcp` (RFC 9293 implementation with window scaling,
//! RFC 7323) runs an Ethernet interface owning [`GATEWAY`] in "any IP" mode,
//! so it accepts segments addressed to any destination. Only TCP frames (and
//! synthetic ARP replies, below) are fed to it; ARP, ICMP and UDP never are.
//! For a SYN of an unknown flow the router keeps the frame, connects to the
//! host-side target on a short-lived thread (10 s timeout) and, once the
//! result is in, either opens a smoltcp socket listening on exactly the
//! console-visible destination and feeds it the SYN (it answers SYN-ACK), or
//! feeds the SYN with no listener (smoltcp answers RST). The poll that feeds
//! a SYN to a fresh listener runs before any other connect result is
//! handled, so a listener can only ever capture its own SYN. Retransmitted
//! SYNs during the connect are dropped.
//!
//! Data moves directly between the non-blocking host socket and the 64 KiB
//! smoltcp buffers (`send`/`recv` closures), so there are no extra buffers: a
//! full transmit buffer stops reads from the host (its TCP window closes),
//! and unread receive data shrinks the window advertised to the console.
//! Host EOF becomes a FIN to the console after the buffered data; the
//! console's FIN becomes `shutdown(Write)` on the host socket after the
//! buffered data. Host errors abort the smoltcp socket (RST to the console);
//! a console RST closes the host socket with `SO_LINGER` 0 (RST to the
//! server). Timeouts: handshake with the console 30 s, no ACK progress while
//! data is outstanding 120 s (smoltcp `set_timeout`), half-closed flows
//! 120 s after the first FIN; idle open connections are kept indefinitely,
//! like a NAT with long TCP timeouts. At most 512 TCP flows and 64 pending
//! connects exist; beyond that SYNs are answered with RST.
//!
//! smoltcp resolves the console's MAC with ARP. Because "any IP" mode makes
//! it claim every address in ARP, its ARP never reaches the console: the
//! router feeds it a synthetic ARP reply (console IP → learned console MAC)
//! before the console's TCP frames, at most every 30 s, and answers any ARP
//! request smoltcp emits the same way.

use std::collections::{HashMap, VecDeque};
use std::io::{self, ErrorKind, Read, Write};
use std::net::{Ipv4Addr, Shutdown, SocketAddr, SocketAddrV4, TcpListener, TcpStream, UdpSocket};
use std::os::fd::AsRawFd;
use std::os::unix::net::UnixStream;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use smoltcp::iface::{Config, Interface, SocketHandle, SocketSet};
use smoltcp::phy::{self, Device, DeviceCapabilities, Medium};
use smoltcp::socket::tcp;
use smoltcp::wire::{EthernetAddress, HardwareAddress, IpAddress, IpCidr, IpListenEndpoint};

use super::wire::{
    self, ARP_REPLY, ARP_REQUEST, Arp, ETH_HEADER, ETHERTYPE_ARP, ETHERTYPE_IPV4, Ipv4, PROTO_ICMP,
    PROTO_TCP, PROTO_UDP, TCP_ACK, TCP_RST, TCP_SYN,
};
use super::{CONSOLE_IP, DNS, GATEWAY, Msg, ROUTER_MAC, Settings, TraceFn, dhcp, dns};

/// Size of each smoltcp TCP buffer (one receive, one transmit per flow).
const TCP_BUFFER: usize = 64 * 1024;
/// Most TCP flows (connecting or open) at once.
const MAX_TCP_FLOWS: usize = 512;
/// Most host connects in flight at once (each runs on its own thread).
const MAX_PENDING_CONNECTS: usize = 64;
/// Host connect timeout.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Time the console has to complete the handshake after our SYN-ACK.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(30);
/// smoltcp aborts when the console ACKs nothing for this long while data is
/// outstanding.
const STALL_TIMEOUT: smoltcp::time::Duration = smoltcp::time::Duration::from_secs(120);
/// Time a half-closed flow may linger after the first FIN.
const CLOSE_TIMEOUT: Duration = Duration::from_secs(120);
/// UDP flows without traffic in either direction for this long are closed.
const UDP_IDLE: Duration = Duration::from_secs(60);
/// Most UDP flows at once; the least recently used is evicted beyond that.
const MAX_UDP_FLOWS: usize = 256;
/// Datagrams read from one UDP flow per wakeup (fairness between flows).
const UDP_BURST: usize = 64;
/// Upper bound of one `poll(2)` sleep, so housekeeping and shutdown run.
const MAX_POLL_WAIT: Duration = Duration::from_secs(1);
/// How often the synthetic ARP entry for the console is refreshed (smoltcp
/// expires neighbours after 60 s).
const ARP_REFRESH: Duration = Duration::from_secs(30);
/// Bytes of an HTTP head buffered for the trace.
const HTTP_HEAD_LIMIT: usize = 4096;
/// Characters of an HTTP line put into a trace line.
const HTTP_LINE_LIMIT: usize = 200;
/// First of the ports [`GATEWAY`] makes forwarded connections from (the
/// range RFC 6335 leaves for such use).
const FIRST_FORWARD_PORT: u16 = 49152;

/// A NAT flow as seen by the console.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct FlowKey {
    console: SocketAddrV4,
    /// The destination the console addressed.
    remote: SocketAddrV4,
}

/// Result of a host connect, sent by the connect thread.
type Connected = (FlowKey, io::Result<TcpStream>);

/// The emulated link as a smoltcp device: frames for smoltcp are queued in
/// `rx`, frames it emits collect in `tx`.
#[derive(Default)]
struct Link {
    rx: VecDeque<Vec<u8>>,
    tx: Vec<Vec<u8>>,
}

struct LinkRx(Vec<u8>);

struct LinkTx<'a>(&'a mut Vec<Vec<u8>>);

impl phy::RxToken for LinkRx {
    fn consume<R, F: FnOnce(&[u8]) -> R>(self, f: F) -> R {
        f(&self.0)
    }
}

impl phy::TxToken for LinkTx<'_> {
    fn consume<R, F: FnOnce(&mut [u8]) -> R>(self, len: usize, f: F) -> R {
        let mut frame = vec![0; len];
        let r = f(&mut frame);
        self.0.push(frame);
        r
    }
}

impl Device for Link {
    type RxToken<'a> = LinkRx;
    type TxToken<'a> = LinkTx<'a>;

    fn receive(&mut self, _: smoltcp::time::Instant) -> Option<(LinkRx, LinkTx<'_>)> {
        let frame = self.rx.pop_front()?;
        Some((LinkRx(frame), LinkTx(&mut self.tx)))
    }

    fn transmit(&mut self, _: smoltcp::time::Instant) -> Option<LinkTx<'_>> {
        Some(LinkTx(&mut self.tx))
    }

    fn capabilities(&self) -> DeviceCapabilities {
        let mut caps = DeviceCapabilities::default();
        caps.medium = Medium::Ethernet;
        // For Ethernet smoltcp counts the 14-byte header in the MTU.
        caps.max_transmission_unit = ETH_HEADER + wire::MTU;
        caps
    }
}

/// A UDP NAT flow: a host socket connected to the target.
struct UdpFlow {
    socket: UdpSocket,
    last_used: Instant,
}

enum TcpFlow {
    /// Host connect in progress; holds the console's SYN frame.
    Connecting {
        syn: Vec<u8>,
    },
    Open(Box<Splice>),
}

/// An open TCP flow: a smoltcp socket spliced to a host stream.
struct Splice {
    handle: SocketHandle,
    host: TcpStream,
    /// The host socket may have data/EOF to read (cleared on `WouldBlock`).
    readable: bool,
    /// The host socket may accept writes (cleared on `WouldBlock`).
    writable: bool,
    host_eof: bool,
    host_failed: bool,
    host_write_shut: bool,
    fin_queued: bool,
    opened: Instant,
    closing_since: Option<Instant>,
    http: Option<HttpTrace>,
}

impl Splice {
    fn wants_read(&self, s: &tcp::Socket) -> bool {
        !self.readable && !self.host_eof && !self.host_failed && s.can_send()
    }

    fn wants_write(&self, s: &tcp::Socket) -> bool {
        !self.writable && !self.host_failed && s.can_recv()
    }

    /// Whether both directions ended gracefully (else the host gets a RST).
    fn finished_cleanly(&self) -> bool {
        self.host_failed || (self.host_eof && self.host_write_shut)
    }

    fn fail(&mut self, s: &mut tcp::Socket) {
        self.host_failed = true;
        s.abort();
    }

    /// Moves data between the host stream and the smoltcp socket and
    /// propagates FINs.
    fn service(&mut self, s: &mut tcp::Socket, trace: &Option<TraceFn>) {
        while self.readable && !self.host_eof && !self.host_failed && s.can_send() {
            let (host, http) = (&mut self.host, &mut self.http);
            let read = s.send(|buf| match host.read(buf) {
                Ok(n) => {
                    if let (Some(h), Some(t)) = (http.as_mut(), trace) {
                        h.response(&buf[..n], t);
                    }
                    (n, Ok(n))
                }
                Err(e) => (0, Err(e)),
            });
            match read {
                Ok(Ok(0)) => self.host_eof = true,
                Ok(Ok(_)) => {}
                Ok(Err(e)) if e.kind() == ErrorKind::WouldBlock => self.readable = false,
                Ok(Err(e)) if e.kind() == ErrorKind::Interrupted => {}
                Ok(Err(_)) => self.fail(s),
                Err(_) => break,
            }
        }
        if self.host_eof && !self.fin_queued {
            s.close();
            self.fin_queued = true;
        }

        while self.writable && !self.host_failed && s.can_recv() {
            let (host, http) = (&mut self.host, &mut self.http);
            let written = s.recv(|buf| match host.write(buf) {
                Ok(n) => {
                    if let (Some(h), Some(t)) = (http.as_mut(), trace) {
                        h.request(&buf[..n], t);
                    }
                    (n, Ok(n))
                }
                Err(e) => (0, Err(e)),
            });
            match written {
                Ok(Ok(0)) => self.fail(s),
                Ok(Ok(_)) => {}
                Ok(Err(e)) if e.kind() == ErrorKind::WouldBlock => self.writable = false,
                Ok(Err(e)) if e.kind() == ErrorKind::Interrupted => {}
                Ok(Err(_)) => self.fail(s),
                Err(_) => break,
            }
        }

        // The console's FIN, once everything before it reached the host.
        if !self.host_write_shut
            && !self.host_failed
            && !s.can_recv()
            && matches!(
                s.state(),
                tcp::State::CloseWait
                    | tcp::State::LastAck
                    | tcp::State::Closing
                    | tcp::State::TimeWait
            )
        {
            let _ = self.host.shutdown(Shutdown::Write);
            self.host_write_shut = true;
        }
    }
}

/// Closes a host stream with a RST instead of a FIN (`SO_LINGER` 0).
fn reset_host(host: &TcpStream) {
    let linger = libc::linger {
        l_onoff: 1,
        l_linger: 0,
    };
    // SAFETY: `host` owns a valid socket fd and `linger` is a correctly
    // sized SO_LINGER value that outlives the call.
    unsafe {
        libc::setsockopt(
            host.as_raw_fd(),
            libc::SOL_SOCKET,
            libc::SO_LINGER,
            (&raw const linger).cast(),
            size_of::<libc::linger>() as libc::socklen_t,
        );
    }
}

/// Captures the first HTTP request line + `Host` header and the first
/// response status line of a port-80 flow for the trace.
pub(super) struct HttpTrace {
    remote: SocketAddrV4,
    /// Request head collected so far; `None` once traced.
    request: Option<Vec<u8>>,
    /// Response head collected so far; `None` once traced.
    response: Option<Vec<u8>>,
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn printable(line: &[u8]) -> String {
    let line = line.strip_suffix(b"\r").unwrap_or(line);
    line.iter()
        .take(HTTP_LINE_LIMIT)
        .map(|&b| {
            if b == b' ' || b.is_ascii_graphic() {
                char::from(b)
            } else {
                '?'
            }
        })
        .collect()
}

impl HttpTrace {
    pub(super) fn new(remote: SocketAddrV4) -> Self {
        HttpTrace {
            remote,
            request: Some(Vec::new()),
            response: Some(Vec::new()),
        }
    }

    fn collect(buf: &mut Vec<u8>, data: &[u8]) {
        let take = data.len().min(HTTP_HEAD_LIMIT.saturating_sub(buf.len()));
        buf.extend_from_slice(&data[..take]);
    }

    pub(super) fn request(&mut self, data: &[u8], trace: &TraceFn) {
        let Some(buf) = &mut self.request else { return };
        Self::collect(buf, data);
        let end = match find(buf, b"\r\n\r\n") {
            Some(end) => end,
            None if buf.len() >= HTTP_HEAD_LIMIT => buf.len(),
            None => return,
        };
        let mut lines = buf[..end].split(|&b| b == b'\n');
        let first = printable(lines.next().unwrap_or_default());
        let host = lines
            .find_map(|l| {
                let (name, value) = l.split_at(l.iter().position(|&b| b == b':')?);
                name.eq_ignore_ascii_case(b"host")
                    .then(|| printable(value[1..].trim_ascii()))
            })
            .unwrap_or_else(|| "-".into());
        trace(&format!("http: {} > {first} (Host: {host})", self.remote));
        self.request = None;
    }

    pub(super) fn response(&mut self, data: &[u8], trace: &TraceFn) {
        let Some(buf) = &mut self.response else {
            return;
        };
        Self::collect(buf, data);
        let end = match find(buf, b"\r\n") {
            Some(end) => end,
            None if buf.len() >= HTTP_HEAD_LIMIT => buf.len(),
            None => return,
        };
        trace(&format!(
            "http: {} < {}",
            self.remote,
            printable(&buf[..end])
        ));
        self.response = None;
    }
}

/// What a `poll(2)` entry belongs to.
#[derive(Clone, Copy)]
enum PollTarget {
    Wake,
    Udp(FlowKey),
    Tcp(FlowKey),
    /// A listener of [`Engine::forwards`].
    Forward(usize),
}

/// State of the router thread.
pub(super) struct Engine {
    settings: Settings,
    deliver: Box<dyn Fn(Vec<u8>) + Send + Sync>,
    trace: Option<TraceFn>,
    inbox: Receiver<Msg>,
    queued: Arc<AtomicUsize>,
    wake_rx: UnixStream,
    /// Write end of the wakeup pipe, for connect threads.
    wake_tx: Arc<UnixStream>,
    connects_tx: Sender<Connected>,
    connects_rx: Receiver<Connected>,
    pending_connects: usize,
    link: Link,
    iface: Interface,
    sockets: SocketSet<'static>,
    /// Source MAC of the console's latest frame.
    console_mac: Option<[u8; 6]>,
    /// Last synthetic ARP entry given to smoltcp.
    arp_primed: Option<([u8; 6], Ipv4Addr, Instant)>,
    ip_id: u16,
    tcp: HashMap<FlowKey, TcpFlow>,
    /// Host listeners and the console port each leads to.
    forwards: Vec<(TcpListener, u16)>,
    /// The port the next forwarded connection is made from.
    forward_port: u16,
    udp: HashMap<FlowKey, UdpFlow>,
    last_expiry: Instant,
    pollfds: Vec<libc::pollfd>,
    poll_targets: Vec<PollTarget>,
    udp_buf: Vec<u8>,
}

impl Engine {
    pub(super) fn new(
        settings: Settings,
        deliver: Box<dyn Fn(Vec<u8>) + Send + Sync>,
        inbox: Receiver<Msg>,
        queued: Arc<AtomicUsize>,
        wake_rx: UnixStream,
        wake_tx: Arc<UnixStream>,
    ) -> Engine {
        let mut link = Link::default();
        let mut config = Config::new(HardwareAddress::Ethernet(EthernetAddress(ROUTER_MAC)));
        // Seeds smoltcp's initial sequence numbers and ephemeral choices.
        config.random_seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64);
        let mut iface = Interface::new(config, &mut link, smoltcp::time::Instant::now());
        iface.update_ip_addrs(|addrs| {
            addrs
                .push(IpCidr::new(IpAddress::Ipv4(GATEWAY), 24))
                .expect("smoltcp has room for one address");
        });
        iface.set_any_ip(true);
        let (connects_tx, connects_rx) = mpsc::channel();
        Engine {
            settings,
            deliver,
            trace: None,
            inbox,
            queued,
            wake_rx,
            wake_tx,
            connects_tx,
            connects_rx,
            pending_connects: 0,
            link,
            iface,
            sockets: SocketSet::new(Vec::new()),
            console_mac: None,
            arp_primed: None,
            ip_id: 0,
            tcp: HashMap::new(),
            forwards: Vec::new(),
            forward_port: FIRST_FORWARD_PORT,
            udp: HashMap::new(),
            last_expiry: Instant::now(),
            pollfds: Vec::new(),
            poll_targets: Vec::new(),
            udp_buf: vec![0; 65536],
        }
    }

    /// Runs until the [`super::Router`] is dropped.
    pub(super) fn run(mut self) {
        loop {
            self.wait();
            if !self.drain_inbox() {
                return;
            }
            while let Ok((key, result)) = self.connects_rx.try_recv() {
                self.on_connected(key, result);
            }
            self.pump_tcp();
            self.expire();
        }
    }

    fn emit(&self, line: impl FnOnce() -> String) {
        if let Some(t) = &self.trace {
            t(&line());
        }
    }

    /// Sleeps in `poll(2)` until a host socket, the inbox or a smoltcp timer
    /// needs attention, then reads ready UDP sockets and marks ready TCP
    /// streams.
    fn wait(&mut self) {
        self.pollfds.clear();
        self.poll_targets.clear();
        let mut add = |fd, events, target| {
            self.pollfds.push(libc::pollfd {
                fd,
                events,
                revents: 0,
            });
            self.poll_targets.push(target);
        };
        add(self.wake_rx.as_raw_fd(), libc::POLLIN, PollTarget::Wake);
        for (key, flow) in &self.udp {
            add(flow.socket.as_raw_fd(), libc::POLLIN, PollTarget::Udp(*key));
        }
        for (key, flow) in &self.tcp {
            let TcpFlow::Open(sp) = flow else { continue };
            let s = self.sockets.get::<tcp::Socket>(sp.handle);
            let mut events = 0;
            if sp.wants_read(s) {
                events |= libc::POLLIN;
            }
            if sp.wants_write(s) {
                events |= libc::POLLOUT;
            }
            if events != 0 {
                add(sp.host.as_raw_fd(), events, PollTarget::Tcp(*key));
            }
        }
        if self.tcp.len() < MAX_TCP_FLOWS {
            for (index, (listener, _)) in self.forwards.iter().enumerate() {
                add(
                    listener.as_raw_fd(),
                    libc::POLLIN,
                    PollTarget::Forward(index),
                );
            }
        }

        let mut timeout = MAX_POLL_WAIT;
        if let Some(d) = self
            .iface
            .poll_delay(smoltcp::time::Instant::now(), &self.sockets)
        {
            timeout = timeout.min(Duration::from_micros(d.total_micros()));
        }
        if !self.link.rx.is_empty() {
            timeout = Duration::ZERO;
        }
        let ms = timeout.as_micros().div_ceil(1000) as libc::c_int;
        // SAFETY: `pollfds` is a valid, exclusively borrowed array of
        // `pollfds.len()` entries for the duration of the call.
        let ready = unsafe {
            libc::poll(
                self.pollfds.as_mut_ptr(),
                self.pollfds.len() as libc::nfds_t,
                ms,
            )
        };
        if ready <= 0 {
            return;
        }
        let mut accepting = Vec::new();
        for i in 0..self.pollfds.len() {
            let revents = self.pollfds[i].revents;
            if revents == 0 {
                continue;
            }
            match self.poll_targets[i] {
                PollTarget::Wake => {
                    let mut sink = [0u8; 64];
                    while matches!((&self.wake_rx).read(&mut sink), Ok(n) if n > 0) {}
                }
                PollTarget::Udp(key) => self.on_udp_readable(key),
                PollTarget::Tcp(key) => {
                    if let Some(TcpFlow::Open(sp)) = self.tcp.get_mut(&key) {
                        let broken = revents & (libc::POLLHUP | libc::POLLERR) != 0;
                        sp.readable |= broken || revents & libc::POLLIN != 0;
                        sp.writable |= broken || revents & libc::POLLOUT != 0;
                    }
                }
                PollTarget::Forward(index) => accepting.push(index),
            }
        }
        for index in accepting {
            self.accept(index);
        }
    }

    /// Takes the connections waiting at a forwarding listener and opens,
    /// for each, one from [`GATEWAY`] to the console's port.
    fn accept(&mut self, index: usize) {
        let console = SocketAddrV4::new(CONSOLE_IP, self.forwards[index].1);
        while self.tcp.len() < MAX_TCP_FLOWS {
            let Ok((host, from)) = self.forwards[index].0.accept() else {
                return;
            };
            // With the console never heard from there is nowhere to send.
            let Some(mac) = self.console_mac else {
                self.emit(|| format!("tcp: {from} -> {console} refused: no console on the link"));
                reset_host(&host);
                continue;
            };
            if host.set_nonblocking(true).is_err() || host.set_nodelay(true).is_err() {
                continue;
            }
            let mut key = FlowKey {
                console,
                remote: SocketAddrV4::new(GATEWAY, self.forward_port),
            };
            while self.tcp.contains_key(&key) {
                key.remote.set_port(
                    key.remote
                        .port()
                        .checked_add(1)
                        .unwrap_or(FIRST_FORWARD_PORT),
                );
            }
            self.forward_port = key
                .remote
                .port()
                .checked_add(1)
                .unwrap_or(FIRST_FORWARD_PORT);
            let mut socket = tcp::Socket::new(
                tcp::SocketBuffer::new(vec![0; TCP_BUFFER]),
                tcp::SocketBuffer::new(vec![0; TCP_BUFFER]),
            );
            socket.set_nagle_enabled(false);
            socket.set_timeout(Some(STALL_TIMEOUT));
            let opened = socket.connect(
                self.iface.context(),
                (IpAddress::Ipv4(CONSOLE_IP), console.port()),
                (IpAddress::Ipv4(GATEWAY), key.remote.port()),
            );
            if opened.is_err() {
                reset_host(&host);
                continue;
            }
            let handle = self.sockets.add(socket);
            self.emit(|| format!("tcp: new flow {from} -> {console} forwarded"));
            // smoltcp asks for the console's MAC before it sends the SYN.
            self.arp_primed = Some((mac, CONSOLE_IP, Instant::now()));
            self.link.rx.push_back(synthetic_arp(mac, CONSOLE_IP));
            self.tcp.insert(
                key,
                TcpFlow::Open(Box::new(Splice {
                    handle,
                    host,
                    readable: true,
                    writable: true,
                    host_eof: false,
                    host_failed: false,
                    host_write_shut: false,
                    fin_queued: false,
                    opened: Instant::now(),
                    closing_since: None,
                    http: None,
                })),
            );
        }
    }

    /// Handles everything [`super::Router`] queued; `false` once it is gone.
    fn drain_inbox(&mut self) -> bool {
        loop {
            match self.inbox.try_recv() {
                Ok(msg) => {
                    self.queued.fetch_sub(1, Ordering::AcqRel);
                    match msg {
                        Msg::Frame(frame) => self.on_frame(&frame),
                        Msg::Trace(t) => self.trace = Some(t),
                        Msg::Forward(listener, port) => self.forwards.push((listener, port)),
                    }
                }
                Err(TryRecvError::Empty) => return true,
                Err(TryRecvError::Disconnected) => return false,
            }
        }
    }

    fn on_frame(&mut self, frame: &[u8]) {
        let Some(eth) = wire::parse_ethernet(frame) else {
            return;
        };
        if eth.src[0] & 1 == 0 && eth.src != ROUTER_MAC && eth.src != [0; 6] {
            self.console_mac = Some(eth.src);
        }
        match eth.ethertype {
            ETHERTYPE_ARP => self.on_arp(eth.payload),
            ETHERTYPE_IPV4 => {
                let Some(ip) = wire::parse_ipv4(eth.payload) else {
                    return;
                };
                // Fragments from the console are not reassembled.
                if ip.fragment {
                    return;
                }
                match ip.protocol {
                    PROTO_UDP => self.on_udp(&ip),
                    PROTO_TCP => self.on_tcp(frame, &ip),
                    PROTO_ICMP if ip.dst == GATEWAY || ip.dst == DNS => {
                        if let Some(reply) = wire::icmp_echo_reply(ip.payload) {
                            self.send_ip(eth.src, ip.dst, ip.src, PROTO_ICMP, &reply);
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    /// Answers ARP requests for the router's addresses (RFC 826).
    fn on_arp(&mut self, payload: &[u8]) {
        let Some(arp) = Arp::parse(payload) else {
            return;
        };
        if arp.op == ARP_REQUEST && (arp.tpa == GATEWAY || arp.tpa == DNS) {
            let reply = Arp {
                op: ARP_REPLY,
                sha: ROUTER_MAC,
                spa: arp.tpa,
                tha: arp.sha,
                tpa: arp.spa,
            };
            (self.deliver)(wire::ethernet(
                arp.sha,
                ROUTER_MAC,
                ETHERTYPE_ARP,
                &reply.encode(),
            ));
        }
    }

    /// Sends an IPv4 datagram to the console, fragmenting as needed.
    fn send_ip(
        &mut self,
        dst_mac: [u8; 6],
        src: Ipv4Addr,
        dst: Ipv4Addr,
        protocol: u8,
        payload: &[u8],
    ) {
        self.ip_id = self.ip_id.wrapping_add(1);
        for frame in wire::ipv4_frames(dst_mac, ROUTER_MAC, src, dst, protocol, self.ip_id, payload)
        {
            (self.deliver)(frame);
        }
    }

    /// Where traffic for the console-visible `remote` goes on the host, or
    /// `None` if it must not leave the router.
    fn host_target(&self, remote: SocketAddrV4) -> Option<SocketAddr> {
        let ip = *remote.ip();
        let port = remote.port();
        if ip == DNS {
            return (port == dns::PORT).then_some(self.settings.dns_upstream);
        }
        if Some(ip) == self.settings.loopback_alias {
            return Some(SocketAddr::from((Ipv4Addr::LOCALHOST, port)));
        }
        let o = ip.octets();
        let unroutable = ip.is_unspecified()
            || ip.is_loopback()
            || ip.is_broadcast()
            || ip.is_multicast()
            || ip.is_link_local()
            || o[0] >= 240
            || o[..3] == [10, 0, 2]
            || port == 0;
        (!unroutable).then_some(SocketAddr::V4(remote))
    }

    fn on_udp(&mut self, ip: &Ipv4) {
        let Some(udp) = wire::parse_udp(ip.payload) else {
            return;
        };
        if udp.dst_port == dhcp::SERVER_PORT && udp.src_port == dhcp::CLIENT_PORT {
            let Some(outcome) = dhcp::handle(udp.payload) else {
                return;
            };
            if let Some(line) = &outcome.trace {
                self.emit(|| line.clone());
            }
            if let Some(r) = outcome.reply {
                let seg = wire::udp(
                    SocketAddrV4::new(GATEWAY, dhcp::SERVER_PORT),
                    SocketAddrV4::new(r.dst_ip, dhcp::CLIENT_PORT),
                    &r.payload,
                );
                self.send_ip(r.dst_mac, GATEWAY, r.dst_ip, PROTO_UDP, &seg);
            }
            return;
        }
        if ip.src.is_unspecified() {
            return;
        }
        let key = FlowKey {
            console: SocketAddrV4::new(ip.src, udp.src_port),
            remote: SocketAddrV4::new(ip.dst, udp.dst_port),
        };
        let Some(target) = self.host_target(key.remote) else {
            return;
        };
        let now = Instant::now();
        if !self.udp.contains_key(&key) {
            if self.udp.len() >= MAX_UDP_FLOWS
                && let Some(oldest) = self
                    .udp
                    .iter()
                    .min_by_key(|(_, f)| f.last_used)
                    .map(|(k, _)| *k)
            {
                self.udp.remove(&oldest);
            }
            let bind = if target.is_ipv4() {
                SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0))
            } else {
                SocketAddr::from((std::net::Ipv6Addr::UNSPECIFIED, 0))
            };
            let socket = UdpSocket::bind(bind).and_then(|s| {
                s.connect(target)?;
                s.set_nonblocking(true)?;
                Ok(s)
            });
            let Ok(socket) = socket else { return };
            self.emit(|| {
                via(
                    format!("udp: new flow {} -> {}", key.console, key.remote),
                    key.remote,
                    target,
                )
            });
            self.udp.insert(
                key,
                UdpFlow {
                    socket,
                    last_used: now,
                },
            );
        }
        if udp.dst_port == dns::PORT
            && let Some((name, qtype)) = dns::query_name(udp.payload)
        {
            self.emit(|| {
                format!(
                    "dns: {} asks {} for {name} {qtype}",
                    key.console, key.remote
                )
            });
        }
        let flow = self.udp.get_mut(&key).expect("flow inserted above");
        flow.last_used = now;
        // Send errors (e.g. ICMP unreachable reported by the host) are
        // dropped datagrams, as on a real network.
        let _ = flow.socket.send(udp.payload);
    }

    /// Relays datagrams waiting on a UDP flow's host socket to the console.
    fn on_udp_readable(&mut self, key: FlowKey) {
        let mut buf = std::mem::take(&mut self.udp_buf);
        for _ in 0..UDP_BURST {
            let Some(flow) = self.udp.get_mut(&key) else {
                break;
            };
            let Ok(n) = flow.socket.recv(&mut buf) else {
                break;
            };
            flow.last_used = Instant::now();
            let Some(mac) = self.console_mac else { break };
            if n > wire::MAX_UDP_PAYLOAD {
                continue;
            }
            let seg = wire::udp(key.remote, key.console, &buf[..n]);
            self.send_ip(mac, *key.remote.ip(), *key.console.ip(), PROTO_UDP, &seg);
        }
        self.udp_buf = buf;
    }

    fn on_tcp(&mut self, frame: &[u8], ip: &Ipv4) {
        let Some(h) = wire::parse_tcp_header(ip.payload) else {
            return;
        };
        if ip.src.is_unspecified() {
            return;
        }
        let key = FlowKey {
            console: SocketAddrV4::new(ip.src, h.src_port),
            remote: SocketAddrV4::new(ip.dst, h.dst_port),
        };
        let syn = h.flags & (TCP_SYN | TCP_ACK | TCP_RST) == TCP_SYN;
        // smoltcp only accepts frames addressed to its own MAC.
        let mut fed = frame.to_vec();
        fed[..6].copy_from_slice(&ROUTER_MAC);
        match self.tcp.get(&key) {
            // Retransmitted SYN while the host connect is in progress.
            Some(TcpFlow::Connecting { .. }) => return,
            Some(TcpFlow::Open(sp)) => {
                let state = self.sockets.get::<tcp::Socket>(sp.handle).state();
                if !(syn && matches!(state, tcp::State::TimeWait | tcp::State::Closed)) {
                    self.feed(fed, ip.src);
                    return;
                }
                // The console reuses the port of a finished connection.
                self.drop_flow(key);
            }
            None => {}
        }
        if !syn {
            // Stray segment: smoltcp answers it with RST.
            self.feed(fed, ip.src);
            return;
        }
        let target = self.host_target(key.remote);
        let Some(target) = target.filter(|_| {
            self.tcp.len() < MAX_TCP_FLOWS && self.pending_connects < MAX_PENDING_CONNECTS
        }) else {
            self.emit(|| format!("tcp: {} -> {} rejected", key.console, key.remote));
            self.feed(fed, ip.src);
            return;
        };
        let done = self.connects_tx.clone();
        let wake = self.wake_tx.clone();
        let spawned = std::thread::Builder::new()
            .name("nat-connect".into())
            .spawn(move || {
                let result = TcpStream::connect_timeout(&target, CONNECT_TIMEOUT);
                if done.send((key, result)).is_ok() {
                    let _ = (&*wake).write(&[0]);
                }
            });
        if spawned.is_err() {
            self.feed(fed, ip.src);
            return;
        }
        self.pending_connects += 1;
        self.tcp.insert(key, TcpFlow::Connecting { syn: fed });
    }

    /// Queues a TCP frame for smoltcp, making sure it knows the console's MAC.
    fn feed(&mut self, frame: Vec<u8>, console_ip: Ipv4Addr) {
        if let Some(mac) = self.console_mac {
            let now = Instant::now();
            let fresh = self
                .arp_primed
                .is_some_and(|(m, ip, at)| m == mac && ip == console_ip && now - at < ARP_REFRESH);
            if !fresh {
                self.arp_primed = Some((mac, console_ip, now));
                self.link.rx.push_back(synthetic_arp(mac, console_ip));
            }
        }
        self.link.rx.push_back(frame);
    }

    fn on_connected(&mut self, key: FlowKey, result: io::Result<TcpStream>) {
        self.pending_connects -= 1;
        if !matches!(self.tcp.get(&key), Some(TcpFlow::Connecting { .. })) {
            return;
        }
        let Some(TcpFlow::Connecting { syn }) = self.tcp.remove(&key) else {
            return;
        };
        let console_ip = *key.console.ip();
        let host = result.and_then(|host| {
            host.set_nonblocking(true)?;
            host.set_nodelay(true)?;
            Ok(host)
        });
        let host = match host {
            Ok(host) => host,
            Err(e) => {
                self.emit(|| format!("tcp: {} -> {} failed: {e}", key.console, key.remote));
                // No listener: smoltcp answers the SYN with RST.
                self.feed(syn, console_ip);
                return;
            }
        };
        let mut socket = tcp::Socket::new(
            tcp::SocketBuffer::new(vec![0; TCP_BUFFER]),
            tcp::SocketBuffer::new(vec![0; TCP_BUFFER]),
        );
        socket.set_nagle_enabled(false);
        socket.set_timeout(Some(STALL_TIMEOUT));
        let endpoint = IpListenEndpoint {
            addr: Some(IpAddress::Ipv4(*key.remote.ip())),
            port: key.remote.port(),
        };
        if socket.listen(endpoint).is_err() {
            reset_host(&host);
            self.feed(syn, console_ip);
            return;
        }
        let handle = self.sockets.add(socket);
        self.emit(|| format!("tcp: new flow {} -> {} connected", key.console, key.remote));
        let http =
            (key.remote.port() == 80 && self.trace.is_some()).then(|| HttpTrace::new(key.remote));
        self.tcp.insert(
            key,
            TcpFlow::Open(Box::new(Splice {
                handle,
                host,
                readable: true,
                writable: true,
                host_eof: false,
                host_failed: false,
                host_write_shut: false,
                fin_queued: false,
                opened: Instant::now(),
                closing_since: None,
                http,
            })),
        );
        self.feed(syn, console_ip);
        // Let the listener take its SYN before any other listener exists.
        self.poll_iface();
        if self.sockets.get::<tcp::Socket>(handle).state() == tcp::State::Listen {
            self.drop_flow(key);
        }
    }

    /// Removes a TCP flow at once, resetting the host side unless it ended
    /// gracefully.
    fn drop_flow(&mut self, key: FlowKey) {
        if let Some(TcpFlow::Open(sp)) = self.tcp.remove(&key) {
            if !sp.finished_cleanly() {
                reset_host(&sp.host);
            }
            self.sockets.remove(sp.handle);
        }
    }

    /// Runs smoltcp until it has nothing more to say, delivering its frames
    /// and answering its ARP requests (see the module docs).
    fn poll_iface(&mut self) {
        for _ in 0..4 {
            self.iface.poll(
                smoltcp::time::Instant::now(),
                &mut self.link,
                &mut self.sockets,
            );
            let mut again = false;
            for frame in std::mem::take(&mut self.link.tx) {
                let Some(eth) = wire::parse_ethernet(&frame) else {
                    continue;
                };
                if eth.ethertype != ETHERTYPE_ARP {
                    (self.deliver)(frame);
                    continue;
                }
                if let (Some(arp), Some(mac)) = (Arp::parse(eth.payload), self.console_mac)
                    && arp.op == ARP_REQUEST
                {
                    self.link.rx.push_back(synthetic_arp(mac, arp.tpa));
                    again = true;
                }
            }
            if !again {
                break;
            }
        }
    }

    /// One round of TCP work: process the console's segments, move data,
    /// send the resulting segments, and remove finished flows.
    fn pump_tcp(&mut self) {
        self.poll_iface();
        for flow in self.tcp.values_mut() {
            if let TcpFlow::Open(sp) = flow {
                sp.service(self.sockets.get_mut::<tcp::Socket>(sp.handle), &self.trace);
            }
        }
        self.poll_iface();

        let now = Instant::now();
        let sockets = &mut self.sockets;
        self.tcp.retain(|_, flow| {
            let TcpFlow::Open(sp) = flow else { return true };
            let s = sockets.get_mut::<tcp::Socket>(sp.handle);
            let state = s.state();
            let connecting = matches!(state, tcp::State::SynReceived | tcp::State::SynSent);
            if connecting && now - sp.opened > HANDSHAKE_TIMEOUT {
                s.abort();
            }
            if sp.fin_queued || sp.host_write_shut {
                let since = *sp.closing_since.get_or_insert(now);
                if now - since > CLOSE_TIMEOUT
                    && !matches!(state, tcp::State::TimeWait | tcp::State::Closed)
                {
                    s.abort();
                }
            }
            // Closed without a peer left means any final RST has been sent.
            let done = state == tcp::State::Listen
                || (state == tcp::State::Closed && s.remote_endpoint().is_none());
            if !done {
                return true;
            }
            if !sp.finished_cleanly() {
                reset_host(&sp.host);
            }
            sockets.remove(sp.handle);
            false
        });
    }

    /// Closes idle UDP flows (checked at most once a second).
    fn expire(&mut self) {
        let now = Instant::now();
        if now - self.last_expiry < Duration::from_secs(1) {
            return;
        }
        self.last_expiry = now;
        self.udp.retain(|_, f| now - f.last_used < UDP_IDLE);
    }
}

/// An ARP reply "from the console" telling smoltcp `ip` is at `mac`.
fn synthetic_arp(mac: [u8; 6], ip: Ipv4Addr) -> Vec<u8> {
    let arp = Arp {
        op: ARP_REPLY,
        sha: mac,
        spa: ip,
        tha: ROUTER_MAC,
        tpa: GATEWAY,
    };
    wire::ethernet(ROUTER_MAC, mac, ETHERTYPE_ARP, &arp.encode())
}

/// Appends the host-side target to a trace line when it differs from what
/// the console addressed.
fn via(line: String, remote: SocketAddrV4, target: SocketAddr) -> String {
    if target == SocketAddr::V4(remote) {
        line
    } else {
        format!("{line} via {target}")
    }
}
