//! User-space NAT router that connects the emulated console's WiFi link to the
//! host's network using nothing but ordinary (unprivileged) host sockets.
//!
//! The WiFi chip model hands every Ethernet II frame the console transmits to
//! [`Router::send`]; the router answers or forwards it and hands every frame
//! addressed to the console back through the `deliver` callback given to
//! [`Router::new`]. The addressing plan follows QEMU's "slirp" user-mode
//! networking convention (10.0.2.0/24, gateway `.2`, DNS `.3`, guest `.15`),
//! which is well known and avoids clashing with typical home LANs.
//!
//! What the router does with each kind of frame:
//!
//! * **ARP** (RFC 826): requests for [`GATEWAY`] or [`DNS`] are answered with
//!   [`ROUTER_MAC`]; every other ARP frame is ignored (in particular RFC 5227
//!   address probes for [`CONSOLE_IP`] are *not* answered, so the console's
//!   duplicate-address detection succeeds). The console's MAC is learned from
//!   the source address of every frame it sends.
//! * **DHCP** (RFC 2131/2132): a single-lease server; see `dhcp.rs`.
//! * **DNS** (RFC 1035): UDP queries to `DNS:53` are relayed unchanged to the
//!   resolver given to [`Router::new`], else the host's (the first parsable
//!   `nameserver` line of `/etc/resolv.conf`, falling back to 8.8.8.8 only
//!   when there is none), and the answers are relayed back from `DNS:53`,
//!   so ids and ports are preserved. TCP to `DNS:53` is spliced to the same
//!   resolver.
//! * **UDP** (RFC 768): every other unicast datagram is sent from a host
//!   `UdpSocket` dedicated to the (console port, destination) pair; replies
//!   from that destination come back with the destination as their source.
//!   Flows idle for 60 s are closed; at most 256 exist at once (the least
//!   recently used is evicted).
//! * **TCP** (RFC 9293): the console's connections are terminated in a
//!   `smoltcp` stack running in "any IP" mode and each one is spliced to a
//!   host `TcpStream` to the original destination. The SYN is held until the
//!   host connect (10 s timeout) succeeds; only then is it answered with a
//!   SYN-ACK, otherwise with a RST. Data is copied directly between the host
//!   socket and the 64 KiB smoltcp buffers, so a slow side throttles the other
//!   through the TCP windows. FIN and RST propagate in both directions.
//! * **ICMP** (RFC 792): echo requests to [`GATEWAY`]/[`DNS`] are answered
//!   locally; all other ICMP is dropped because unprivileged host sockets
//!   cannot forward it.
//! * IPv6, IPv4 fragments from the console, multicast/broadcast other than
//!   DHCP, and traffic to the 10.0.2.0/24 subnet, loopback, link-local or
//!   reserved addresses are dropped.
//! * **Inbound TCP**: nothing reaches the console from outside except
//!   through [`Router::forward`], which listens on a port of the host's
//!   loopback address and opens, for each connection it accepts, one from
//!   [`GATEWAY`] (an ephemeral port) to a port of [`CONSOLE_IP`], spliced
//!   like the outbound ones. If the console refuses or has not been heard
//!   from, the host connection is reset.
//!
//! All of this runs on one dedicated thread ("nat-router") that multiplexes
//! the host sockets with `poll(2)`; [`Router::send`] only enqueues the frame
//! and never blocks. Large UDP replies are fragmented to the 1500-byte MTU.
//!
//! **Machine snapshots.** The router is not part of the machine: it holds
//! host sockets, which cannot be saved, and the machine's builder attaches
//! a new one to the WiFi model's access point (the `uplink`) whenever it
//! builds a machine, restored or not. Saving and restoring therefore leave
//! it alone, and:
//! * while the machine is paused (guest time standing still, as while a
//!   snapshot is taken) the radio model drops what the router delivers;
//!   the console sees a frame lost on the air and its peers retransmit;
//! * a restored machine meets a router with no flows: every TCP connection
//!   and UDP flow of the saved machine is gone, as after the link dropped
//!   (the console's next segment on an old connection is answered with a
//!   RST), and the console's MAC is learned again from its next frame;
//! * the console keeps its address: the DHCP server keeps no lease state
//!   (there is one lease, [`CONSOLE_IP`], for any client), so the lease the
//!   console holds in its saved memory stays valid and its renewals are
//!   acknowledged as before.

mod dhcp;
mod dns;
mod engine;
mod wire;

#[cfg(test)]
mod tests;

use std::io::Write;
use std::net::{Ipv4Addr, SocketAddr, TcpListener};
use std::os::unix::net::UnixStream;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, Sender};

/// The router's own MAC address, used for [`GATEWAY`] and [`DNS`].
///
/// `02:00:00:00:00:01`: the "locally administered" bit (0x02 in the first
/// octet, IEEE 802 clause 8.2.2) is set and the group bit is clear, so it can
/// never collide with a vendor-assigned address such as the console's.
pub const ROUTER_MAC: [u8; 6] = [0x02, 0x00, 0x00, 0x00, 0x00, 0x01];

/// The default gateway handed out by DHCP (slirp convention); also the DHCP
/// server identifier and the source of all DHCP replies.
pub const GATEWAY: Ipv4Addr = Ipv4Addr::new(10, 0, 2, 2);

/// The DNS server the console is told to use (slirp convention). Queries sent
/// to it are relayed to the host's resolver.
pub const DNS: Ipv4Addr = Ipv4Addr::new(10, 0, 2, 3);

/// The single address the DHCP server leases to the console.
pub const CONSOLE_IP: Ipv4Addr = Ipv4Addr::new(10, 0, 2, 15);

/// Subnet mask of the console's network (10.0.2.0/24).
const NETMASK: Ipv4Addr = Ipv4Addr::new(255, 255, 255, 0);

/// Directed broadcast address of the console's network.
const SUBNET_BROADCAST: Ipv4Addr = Ipv4Addr::new(10, 0, 2, 255);

/// Frames queued by [`Router::send`] but not yet processed; beyond this the
/// router drops new frames, like a NIC whose receive ring is full.
const MAX_QUEUED_FRAMES: usize = 4096;

/// Host-side settings of a [`Router`].
#[derive(Clone, Debug)]
struct Settings {
    /// Where DNS queries sent to [`DNS`]`:53` are relayed.
    dns_upstream: SocketAddr,
    /// A console-visible address whose traffic is sent to the host's
    /// 127.0.0.1 instead (same port). `None` in production; tests use it to
    /// reach local servers through an ordinary-looking destination.
    loopback_alias: Option<Ipv4Addr>,
}

/// Callback receiving trace lines; see [`Router::set_trace`].
type TraceFn = Box<dyn Fn(&str) + Send + Sync>;

/// Messages from [`Router`] to its thread.
enum Msg {
    /// An Ethernet frame sent by the console.
    Frame(Vec<u8>),
    /// Install a trace callback.
    Trace(TraceFn),
    /// Connections this host listener accepts go to that port of the
    /// console.
    Forward(TcpListener, u16),
}

/// Handle to the NAT router thread. Dropping the last `Arc` stops the thread
/// (within about a second) and closes every host socket it owns.
pub struct Router {
    /// Frames and control messages for the router thread; `None` only while
    /// dropping.
    tx: Option<Sender<Msg>>,
    /// Write end of the self-pipe that wakes the thread out of `poll(2)`.
    wake: UnixStream,
    /// Number of messages in `tx` not yet taken by the thread.
    queued: Arc<AtomicUsize>,
}

impl Router {
    /// Starts the router thread. `deliver` is called (from that thread) with
    /// every Ethernet frame destined for the console.
    ///
    /// DNS queries are relayed to `resolver`, or when it is `None` to the
    /// first `nameserver` in `/etc/resolv.conf`, read once here; 8.8.8.8 is
    /// used only if the file has none. A resolver of one's own is how a
    /// captive portal is put in front of the console: it answers the
    /// console's connection test host with the portal's address.
    pub fn new(
        deliver: Box<dyn Fn(Vec<u8>) + Send + Sync>,
        resolver: Option<SocketAddr>,
    ) -> Arc<Router> {
        Self::with_settings(
            deliver,
            Settings {
                dns_upstream: resolver.unwrap_or_else(dns::host_resolver),
                loopback_alias: None,
            },
        )
    }

    /// [`Router::new`] with explicit host-side settings.
    fn with_settings(
        deliver: Box<dyn Fn(Vec<u8>) + Send + Sync>,
        settings: Settings,
    ) -> Arc<Router> {
        let (tx, rx) = mpsc::channel();
        let (wake, wake_rx) = UnixStream::pair().expect("net: socketpair for router wakeups");
        wake.set_nonblocking(true)
            .expect("net: non-blocking wake socket");
        wake_rx
            .set_nonblocking(true)
            .expect("net: non-blocking wake socket");
        let queued = Arc::new(AtomicUsize::new(0));
        let thread_queued = queued.clone();
        let wake_tx = Arc::new(wake.try_clone().expect("net: clone wake socket"));
        std::thread::Builder::new()
            .name("nat-router".into())
            .spawn(move || {
                engine::Engine::new(settings, deliver, rx, thread_queued, wake_rx, wake_tx).run()
            })
            .expect("net: spawn router thread");
        Arc::new(Router {
            tx: Some(tx),
            wake,
            queued,
        })
    }

    /// Hands the router an Ethernet II frame sent by the console (any
    /// destination MAC). Never blocks: the frame is copied and queued for the
    /// router thread, or dropped if 4096 frames are already waiting.
    pub fn send(&self, frame: &[u8]) {
        if self.queued.fetch_add(1, Ordering::AcqRel) >= MAX_QUEUED_FRAMES {
            self.queued.fetch_sub(1, Ordering::AcqRel);
            return;
        }
        self.post(Msg::Frame(frame.to_vec()));
    }

    /// Installs a trace callback, called from the router thread with one line
    /// per: new TCP flow (with the connect result), new UDP flow, DNS query
    /// name, DHCP lease, and for TCP flows to port 80 the first request line
    /// with its `Host` header and the first response status line. Nothing
    /// else, and no other payload, is ever traced.
    pub fn set_trace(&self, f: Box<dyn Fn(&str) + Send + Sync>) {
        self.queued.fetch_add(1, Ordering::AcqRel);
        self.post(Msg::Trace(f));
    }

    /// Opens a way in: every TCP connection made to `host_port` on the
    /// host's loopback address is carried to `console_port` of the
    /// console, which sees it come from [`GATEWAY`]. Nothing on the
    /// console's side of the router is reachable otherwise. Fails if the
    /// host port cannot be listened on.
    pub fn forward(&self, host_port: u16, console_port: u16) -> std::io::Result<()> {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, host_port))?;
        self.forward_from(listener, console_port)
    }

    /// [`Router::forward`] for a listener already bound.
    fn forward_from(&self, listener: TcpListener, console_port: u16) -> std::io::Result<()> {
        listener.set_nonblocking(true)?;
        self.queued.fetch_add(1, Ordering::AcqRel);
        self.post(Msg::Forward(listener, console_port));
        Ok(())
    }

    fn post(&self, msg: Msg) {
        if let Some(tx) = &self.tx
            && tx.send(msg).is_ok()
        {
            // A full pipe already guarantees a pending wakeup.
            let _ = (&self.wake).write(&[0]);
        }
    }
}

impl Drop for Router {
    fn drop(&mut self) {
        self.tx = None;
        let _ = (&self.wake).write(&[0]);
    }
}
