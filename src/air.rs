//! The air between emulators: the 802.11 frames each emulated radio
//! transmits, carried to the radios of the other emulators that share the
//! same air (`--air ADDRESS`).
//!
//! Every emulator given the same address joins the same air. The radio
//! hands each frame it transmits to [`Air::transmit`] with the channel it
//! is tuned to; each other radio receives it through the callback given to
//! [`Air::join`] with that channel, and keeps it only if it is listening on
//! that channel then (its own business, as on a real channel). Nothing is
//! acknowledged or retransmitted here: like the air, the link may lose a
//! frame, and the 802.11 protocols above it cope.
//!
//! The carrier is UDP on the host. One of the emulators is the broker: it
//! holds a socket bound to ADDRESS, takes the datagrams of every member and
//! sends each frame to every member but its sender. Members are learnt from
//! what they send; each one says hello once a second so that a broker learns
//! it without waiting for traffic, and forgets it after
//! [`PEER_TIMEOUT`] of silence. No emulator is special: each one tries, once
//! a second, to bind ADDRESS, and the first that succeeds is the broker.
//! When it goes away, the next attempt of another member takes its place
//! and the members' next hello registers them again: the link comes back
//! as after a dropped link, which is also what a restored machine does.
//!
//! A datagram is [`MAGIC`], a kind (hello or frame), the channel and the
//! frame (802.11 MAC header, body, no frame check sequence).

use std::{
    io,
    net::{SocketAddr, UdpSocket},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

/// Start of every datagram: "HA" (horizon-emu air) and a format version.
pub const MAGIC: [u8; 3] = [b'H', b'A', 1];
const KIND_HELLO: u8 = 0;
const KIND_FRAME: u8 = 1;
const HEADER_LEN: usize = MAGIC.len() + 2;
/// The largest datagram: an 802.11 MPDU is at most 2346 bytes (IEEE
/// 802.11-2016 9.2.4.7 without aggregation), with room to spare.
const MAX_DATAGRAM: usize = 4096;
/// How often a member says hello and tries to become the broker.
pub const HELLO_PERIOD: Duration = Duration::from_secs(1);
/// How long the broker keeps a member it no longer hears from.
pub const PEER_TIMEOUT: Duration = Duration::from_secs(3);
/// How long the threads wait for a datagram before they look whether the
/// link is being closed.
const POLL: Duration = Duration::from_millis(200);

/// Takes a frame from the air: the channel it was sent on and the frame.
pub type Receiver = Box<dyn Fn(u8, &[u8]) + Send + Sync>;

/// A datagram of the link, decoded.
#[derive(Debug, PartialEq, Eq)]
pub enum Message<'a> {
    Hello,
    Frame { channel: u8, frame: &'a [u8] },
}

impl Message<'_> {
    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = MAGIC.to_vec();
        match self {
            Message::Hello => bytes.extend([KIND_HELLO, 0]),
            Message::Frame { channel, frame } => {
                bytes.extend([KIND_FRAME, *channel]);
                bytes.extend(*frame);
            }
        }
        bytes
    }

    /// `None` for what is not a datagram of this link (wrong magic or
    /// version, unknown kind, an empty frame).
    pub fn decode(bytes: &[u8]) -> Option<Message<'_>> {
        if bytes.len() < HEADER_LEN || bytes[..MAGIC.len()] != MAGIC {
            return None;
        }
        let (kind, channel, frame) = (bytes[3], bytes[4], &bytes[HEADER_LEN..]);
        match kind {
            KIND_HELLO => Some(Message::Hello),
            KIND_FRAME if !frame.is_empty() => Some(Message::Frame { channel, frame }),
            _ => None,
        }
    }
}

/// The broker's bookkeeping: the members it heard from and when.
#[derive(Default)]
pub struct Relay {
    peers: Vec<(SocketAddr, Instant)>,
}

impl Relay {
    /// Takes datagram `bytes` from `from` at `now`: whom to send it to (the
    /// other members, for a frame; nobody for a hello or what is not of
    /// this link). A member is registered by anything valid it sends;
    /// members silent for [`PEER_TIMEOUT`] are dropped.
    pub fn datagram(&mut self, from: SocketAddr, bytes: &[u8], now: Instant) -> Vec<SocketAddr> {
        self.peers
            .retain(|&(_, seen)| now.saturating_duration_since(seen) < PEER_TIMEOUT);
        let Some(message) = Message::decode(bytes) else {
            return Vec::new();
        };
        match self.peers.iter_mut().find(|(peer, _)| *peer == from) {
            Some(entry) => entry.1 = now,
            None => self.peers.push((from, now)),
        }
        match message {
            Message::Hello => Vec::new(),
            Message::Frame { .. } => self
                .peers
                .iter()
                .map(|&(peer, _)| peer)
                .filter(|&peer| peer != from)
                .collect(),
        }
    }

    pub fn members(&self) -> usize {
        self.peers.len()
    }
}

/// This emulator's link to the air.
pub struct Air {
    socket: UdpSocket,
    broker: SocketAddr,
    closed: Arc<AtomicBool>,
}

impl Air {
    /// Joins the air at `broker`: frames the other members transmit go to
    /// `receive`. Fails only if no local socket can be made; whether a
    /// broker answers does not matter (there may be nobody else yet).
    pub fn join(broker: SocketAddr, receive: Receiver) -> io::Result<Air> {
        let local: SocketAddr = match broker {
            SocketAddr::V4(_) => "0.0.0.0:0".parse().unwrap(),
            SocketAddr::V6(_) => "[::]:0".parse().unwrap(),
        };
        let socket = UdpSocket::bind(local)?;
        socket.set_read_timeout(Some(POLL))?;
        let closed = Arc::new(AtomicBool::new(false));
        let receiving = socket.try_clone()?;
        let hello = socket.try_clone()?;
        {
            let closed = Arc::clone(&closed);
            std::thread::Builder::new()
                .name("air receive".into())
                .spawn(move || receive_frames(&receiving, &closed, &receive))?;
        }
        {
            let closed = Arc::clone(&closed);
            std::thread::Builder::new()
                .name("air hello".into())
                .spawn(move || keep_up(&hello, broker, &closed))?;
        }
        let _ = socket.send_to(&Message::Hello.encode(), broker);
        Ok(Air {
            socket,
            broker,
            closed,
        })
    }

    /// Sends `frame` on `channel` to the other members. Never blocks for
    /// long and never fails: a frame the host cannot send is lost, as on
    /// the air.
    pub fn transmit(&self, channel: u8, frame: &[u8]) {
        let message = Message::Frame { channel, frame }.encode();
        let _ = self.socket.send_to(&message, self.broker);
    }
}

impl Drop for Air {
    fn drop(&mut self) {
        self.closed.store(true, Ordering::Relaxed);
    }
}

fn receive_frames(socket: &UdpSocket, closed: &AtomicBool, receive: &Receiver) {
    let mut buffer = vec![0; MAX_DATAGRAM];
    while !closed.load(Ordering::Relaxed) {
        let Ok((n, _)) = socket.recv_from(&mut buffer) else {
            continue;
        };
        if let Some(Message::Frame { channel, frame }) = Message::decode(&buffer[..n]) {
            receive(channel, frame);
        }
    }
}

/// Says hello to the broker once a second and, while this emulator is not
/// the broker, tries to become it.
fn keep_up(socket: &UdpSocket, broker: SocketAddr, closed: &Arc<AtomicBool>) {
    let mut serving = false;
    while !closed.load(Ordering::Relaxed) {
        if !serving && let Ok(listener) = UdpSocket::bind(broker) {
            let closed = Arc::clone(closed);
            serving = std::thread::Builder::new()
                .name("air broker".into())
                .spawn(move || serve(&listener, &closed))
                .is_ok();
        }
        let _ = socket.send_to(&Message::Hello.encode(), broker);
        std::thread::sleep(HELLO_PERIOD);
    }
}

/// The broker: relays each member's frames to the others.
fn serve(listener: &UdpSocket, closed: &AtomicBool) {
    if listener.set_read_timeout(Some(POLL)).is_err() {
        return;
    }
    let mut relay = Relay::default();
    let mut buffer = vec![0; MAX_DATAGRAM];
    while !closed.load(Ordering::Relaxed) {
        let Ok((n, from)) = listener.recv_from(&mut buffer) else {
            continue;
        };
        for peer in relay.datagram(from, &buffer[..n], Instant::now()) {
            let _ = listener.send_to(&buffer[..n], peer);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, mpsc};

    fn address(port: u16) -> SocketAddr {
        SocketAddr::from(([127, 0, 0, 1], port))
    }

    #[test]
    fn messages_round_trip() {
        let frame = [0x80, 0, 0, 0, 0xff, 0xff];
        let encoded = Message::Frame {
            channel: 6,
            frame: &frame,
        }
        .encode();
        assert_eq!(encoded[..3], MAGIC);
        assert_eq!(
            Message::decode(&encoded),
            Some(Message::Frame {
                channel: 6,
                frame: &frame
            })
        );
        assert_eq!(
            Message::decode(&Message::Hello.encode()),
            Some(Message::Hello)
        );
        // Not of this link: another version, a frame without bytes, an
        // unknown kind, too short.
        assert_eq!(Message::decode(&[b'H', b'A', 2, 1, 6, 0x80]), None);
        assert_eq!(Message::decode(&[b'H', b'A', 1, 1, 6]), None);
        assert_eq!(Message::decode(&[b'H', b'A', 1, 7, 6, 0]), None);
        assert_eq!(Message::decode(b"HA"), None);
    }

    #[test]
    fn relay_sends_frames_to_the_other_members() {
        let mut relay = Relay::default();
        let t = Instant::now();
        let hello = Message::Hello.encode();
        let frame = Message::Frame {
            channel: 1,
            frame: &[0xd0, 0],
        }
        .encode();
        assert!(relay.datagram(address(1), &hello, t).is_empty());
        assert!(relay.datagram(address(2), &hello, t).is_empty());
        // A frame from a member nobody heard yet registers it too.
        assert_eq!(
            relay.datagram(address(3), &frame, t),
            [address(1), address(2)]
        );
        assert_eq!(
            relay.datagram(address(1), &frame, t),
            [address(2), address(3)]
        );
        assert_eq!(relay.members(), 3);
        // Garbage changes nothing.
        assert!(relay.datagram(address(4), b"xyz", t).is_empty());
        assert_eq!(relay.members(), 3);
    }

    #[test]
    fn relay_forgets_silent_members() {
        let mut relay = Relay::default();
        let t = Instant::now();
        let hello = Message::Hello.encode();
        let frame = Message::Frame {
            channel: 1,
            frame: &[0xd0, 0],
        }
        .encode();
        relay.datagram(address(1), &hello, t);
        relay.datagram(address(2), &hello, t);
        let later = t + PEER_TIMEOUT / 2;
        relay.datagram(address(2), &hello, later);
        // Member 1 was last heard a timeout ago: gone. Member 2 stays.
        let end = t + PEER_TIMEOUT;
        assert_eq!(relay.datagram(address(3), &frame, end), [address(2)]);
        assert_eq!(relay.members(), 2);
    }

    /// Two members on the loopback interface: one becomes the broker, and
    /// each one's frames reach the other with their channel.
    #[test]
    fn two_members_exchange_frames() {
        let port = UdpSocket::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let broker = address(port);
        let (tx_a, rx_a) = mpsc::channel();
        let (tx_b, rx_b) = mpsc::channel();
        let (tx_a, tx_b) = (Mutex::new(tx_a), Mutex::new(tx_b));
        let a = Air::join(
            broker,
            Box::new(move |c, f| {
                let _ = tx_a.lock().unwrap().send((c, f.to_vec()));
            }),
        )
        .unwrap();
        let b = Air::join(
            broker,
            Box::new(move |c, f| {
                let _ = tx_b.lock().unwrap().send((c, f.to_vec()));
            }),
        )
        .unwrap();
        // Until the broker is up and knows both, frames may be lost: send
        // until one arrives, as a radio's retries would.
        let deadline = Instant::now() + Duration::from_secs(10);
        let received = loop {
            a.transmit(11, b"from a");
            if let Ok(got) = rx_b.recv_timeout(Duration::from_millis(100)) {
                break got;
            }
            assert!(Instant::now() < deadline, "no frame from a reached b");
        };
        assert_eq!(received, (11, b"from a".to_vec()));
        let received = loop {
            b.transmit(36, b"from b");
            if let Ok(got) = rx_a.recv_timeout(Duration::from_millis(100)) {
                break got;
            }
            assert!(Instant::now() < deadline, "no frame from b reached a");
        };
        assert_eq!(received, (36, b"from b".to_vec()));
        // A member never hears its own frames back.
        assert!(rx_a.try_iter().all(|(_, f)| f == b"from b"));
        assert!(rx_b.try_iter().all(|(_, f)| f == b"from a"));
    }
}
