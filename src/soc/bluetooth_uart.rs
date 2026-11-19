//! The on-board Bluetooth controller, on UART-D's lines, and what is on
//! the air around it.
//!
//! The port itself is a [`Serial`]; this is its
//! peer. H4 framing, the Command Complete layout and ACL data packets
//! follow public Linux hci.h:
//! <https://github.com/torvalds/linux/blob/master/include/net/bluetooth/hci.h>.
//! The commands themselves are executed by the `hci` controller model and
//! its `link` layer; an HCI command or ACL packet they do not know refuses
//! the packet's final byte rather than claim success.
//!
//! The [`Air`] holds what is in range of the radio: a Pro Controller, once
//! placed there (`--pro-controller`), with the buttons and the SYNC button
//! someone at the controls presses. What the controller and the device do
//! of their own accord (inquiry results, a device paging the console, the
//! device's input reports) comes over the line at the guest time it is
//! due, from a thread of the line's own ([`keep_radio`]).

mod hci;
mod l2cap;
mod link;
mod remote;
mod sdp;

use std::{
    path::Path,
    sync::{
        Arc, Mutex, Weak,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use serde::{Deserialize, Serialize};

use super::{
    joycon::{Button, Controls},
    serial::{Peer, Serial},
};
use crate::{
    clock::GuestClock,
    snapshot::{SnapshotError, decode, encode},
};

pub const BASE: u64 = 0x7000_6300;
/// UARTD interrupt, public Tegra210 device tree SPI 90.
pub const INTID: u32 = 32 + 90;

/// H4 packet types (public Linux `hci.h`).
const COMMAND: u8 = 1;
const ACL: u8 = 2;
/// The longest ACL packet the host may send: the controller's buffer
/// length ([`hci`]'s Read Buffer Size answer, 1021) and the header.
const ACL_MOST: usize = 5 + 1021;

/// How often the line looks at the air when nothing is due sooner: a
/// button pressed, the SYNC button. Guest time.
const LOOK: Duration = Duration::from_millis(20);

/// What is in range of the console's Bluetooth radio, shared between the
/// line and whoever is at the controls.
pub struct Air {
    clock: Arc<GuestClock>,
    pro: Mutex<Option<remote::ProController>>,
    /// The Pro Controller's buttons.
    buttons: Arc<Controls>,
    /// SYNC was pressed and the controller has not seen it yet.
    sync: AtomicBool,
}

impl Air {
    pub fn new(clock: Arc<GuestClock>) -> Self {
        Self {
            clock,
            pro: Mutex::new(None),
            buttons: Arc::default(),
            sync: AtomicBool::new(false),
        }
    }

    /// Places a Pro Controller in range, which remembers what it is paired
    /// with in the file `memory` (created when it first pairs).
    pub fn place_pro_controller(&self, memory: &Path) -> Result<(), String> {
        let pro = remote::ProController::new(
            Arc::clone(&self.buttons),
            Arc::clone(&self.clock),
            memory.to_path_buf(),
        )?;
        *self.pro.lock().unwrap() = Some(pro);
        Ok(())
    }

    /// Whether there is a Pro Controller in range.
    pub fn has_pro_controller(&self) -> bool {
        self.pro.lock().unwrap().is_some()
    }

    /// Holds or lets go of a button of the Pro Controller. A button wakes
    /// a sleeping controller that is paired with a console.
    pub fn press(&self, button: Button, held: bool) {
        self.buttons.press(button, held);
    }

    /// Presses the Pro Controller's SYNC button: it drops any link and
    /// becomes discoverable, for a console to pair with it.
    pub fn sync(&self) {
        self.sync.store(true, Ordering::SeqCst);
    }
}

/// Saved with the line: the Pro Controller's state, if one is in range,
/// and the buttons held on it.
#[derive(Serialize, Deserialize)]
struct SavedAir {
    pro: Option<Vec<u8>>,
    buttons: Vec<u8>,
    sync: bool,
}

/// The controller's end of the line: collects a packet, has it executed
/// and answers with the events and data that follow. A snapshot saves all
/// of it: the packet half received, the controller's state and what is on
/// the air.
pub struct Line {
    incoming: Vec<u8>,
    controller: hci::Controller,
    clock: Arc<GuestClock>,
    air: Arc<Air>,
    /// Something the controller or the device was to do on its own that
    /// is not modeled; the next byte from the host is refused with it.
    refused: Option<String>,
}

impl Line {
    pub fn new(clock: Arc<GuestClock>, air: Arc<Air>) -> Self {
        Self {
            incoming: Vec::with_capacity(ACL_MOST),
            controller: hci::Controller::default(),
            clock,
            air,
            refused: None,
        }
    }

    /// Takes the next byte of a packet; the answer once it is complete.
    fn packet_byte(&mut self, value: u8) -> Result<Vec<u8>, String> {
        let kind = *self.incoming.first().unwrap_or(&value);
        self.incoming.push(value);
        if !matches!(kind, COMMAND | ACL) || self.incoming.len() > ACL_MOST {
            let what = format!(
                "Bluetooth H4 packet not modeled: {:02x?}",
                &self.incoming[..self.incoming.len().min(64)]
            );
            self.refused = Some(what.clone());
            return Err(what);
        }
        let packet = &self.incoming[1..];
        let length = match (kind, packet) {
            (COMMAND, [_, _, length, ..]) => 3 + usize::from(*length),
            (ACL, [_, _, low, high, ..]) => 4 + usize::from(u16::from_le_bytes([*low, *high])),
            _ => return Ok(Vec::new()),
        };
        if packet.len() < length {
            return Ok(Vec::new());
        }
        let packet = std::mem::replace(&mut self.incoming, Vec::with_capacity(ACL_MOST));
        let now = self.clock.now();
        let mut out = Vec::new();
        let mut pro = self.air.pro.lock().unwrap();
        let header = u16::from_le_bytes([packet[1], packet[2]]);
        let result = match kind {
            COMMAND => self
                .controller
                .command(header, &packet[4..], now, pro.as_mut(), &mut out),
            _ => self
                .controller
                .acl(header, &packet[5..], pro.as_mut(), &mut out),
        };
        // What is not modeled stops the line: every byte after it is
        // refused the same way.
        if let Err(what) = &result {
            self.refused = Some(what.clone());
        }
        result.map(|()| out)
    }
}

impl Peer for Line {
    fn byte(&mut self, byte: u8) -> Result<Vec<u8>, String> {
        if let Some(what) = &self.refused {
            return Err(what.clone());
        }
        self.packet_byte(byte)
    }

    /// CTS is asserted by the powered, present controller.
    fn clear_to_send(&self) -> bool {
        true
    }

    fn due(&mut self, now: Duration) -> Result<(Vec<u8>, Option<Duration>), String> {
        if self.refused.is_some() {
            return Ok((Vec::new(), None));
        }
        let mut out = Vec::new();
        let mut pro = self.air.pro.lock().unwrap();
        if self.air.sync.swap(false, Ordering::SeqCst)
            && let Some(pro) = pro.as_mut()
            && pro.sync()
        {
            self.controller.dropped_by_device(&mut out);
        }
        let next = match self.controller.due(now, pro.as_mut(), &mut out) {
            Ok(next) => next,
            Err(what) => {
                self.refused = Some(what.clone());
                return Err(what);
            }
        };
        // A device in range may wake up or be pressed at any moment.
        let look = now + LOOK;
        let next = match pro.is_some() {
            true => Some(next.map_or(look, |next| next.min(look))),
            false => next,
        };
        Ok((out, next))
    }

    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        let pro = self.air.pro.lock().unwrap();
        let air = SavedAir {
            pro: pro.as_ref().map(|pro| pro.save()).transpose()?,
            buttons: crate::snapshot::Snapshot::save(self.air.buttons.as_ref())?,
            sync: self.air.sync.load(Ordering::SeqCst),
        };
        encode(&(&self.incoming, &self.controller, &self.refused, air))
    }

    /// Takes back the line, the controller and what is on the air. A
    /// snapshot with a Pro Controller in range needs one placed in this
    /// machine too, and the other way round.
    fn restore(&mut self, state: &[u8]) -> Result<(), SnapshotError> {
        type Saved = (Vec<u8>, hci::Controller, Option<String>, SavedAir);
        let (incoming, controller, refused, air): Saved = decode("Bluetooth controller", state)?;
        if incoming.len() > ACL_MOST {
            return Err(SnapshotError::Corrupt(
                "Bluetooth controller: packet longer than its buffer".into(),
            ));
        }
        let mut pro = self.air.pro.lock().unwrap();
        match (pro.as_mut(), &air.pro) {
            (Some(placed), Some(saved)) => placed.restore(saved)?,
            (None, None) => {}
            _ => {
                return Err(SnapshotError::Corrupt(
                    "Bluetooth: a Pro Controller in range of one machine and not of the other (--pro-controller)"
                        .into(),
                ));
            }
        }
        crate::snapshot::Snapshot::restore(self.air.buttons.as_ref(), &air.buttons)?;
        self.air.sync.store(air.sync, Ordering::SeqCst);
        self.incoming = incoming;
        self.controller = controller;
        self.refused = refused;
        Ok(())
    }
}

/// Delivers what the controller on `uart` does of its own accord at the
/// guest time it is due, from a thread of its own, until the port is gone.
/// Nothing is delivered while guest time stands still: the machine is
/// paused, perhaps being saved, and what was due waits in the line's
/// state.
pub fn keep_radio(uart: &Arc<Serial>, clock: Arc<GuestClock>) {
    let uart: Weak<Serial> = Arc::downgrade(uart);
    std::thread::Builder::new()
        .name("Bluetooth radio".into())
        .spawn(move || {
            let mut next = clock.now() + LOOK;
            loop {
                clock.sleep_until(next);
                let Some(uart) = uart.upgrade() else {
                    return;
                };
                let now = clock.now();
                if clock.is_paused() {
                    next = now + LOOK;
                    continue;
                }
                next = uart
                    .deliver(now)
                    .unwrap_or(now + LOOK)
                    .max(now + Duration::from_millis(1));
            }
        })
        .expect("Bluetooth radio thread");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{bus::MmioDevice, soc::serial::Serial};

    fn line() -> Line {
        let clock = Arc::new(GuestClock::new());
        Line::new(Arc::clone(&clock), Arc::new(Air::new(clock)))
    }

    #[test]
    fn fragmented_reset_returns_exact_command_complete() {
        let mut line = line();
        for byte in [1, 3, 12] {
            assert_eq!(line.byte(byte), Ok(Vec::new()));
        }
        assert_eq!(line.byte(0), Ok(vec![4, 14, 4, 1, 3, 12, 0]));
    }

    #[test]
    fn download_requires_minidriver_and_launch_exits_it() {
        let mut line = line();
        for command in [
            &[1, 0x2e, 0xfc, 0][..],
            &[1, 0x4c, 0xfc, 5, 0, 0, 0, 0, 0xaa],
            &[1, 0x4e, 0xfc, 4, 0xff, 0xff, 0xff, 0xff],
        ] {
            let answer: Vec<u8> = command
                .iter()
                .flat_map(|&byte| line.byte(byte).unwrap())
                .collect();
            assert_eq!(answer.len(), 7);
        }
        for byte in [1, 0x4c, 0xfc, 5, 0, 0, 0, 0] {
            assert_eq!(line.byte(byte), Ok(Vec::new()));
        }
        assert!(line.byte(0).is_err());
    }

    #[test]
    fn acl_data_for_no_link_is_taken_and_other_packet_types_are_refused() {
        let mut line = line();
        // Handle 0x000b, first flushable, 4 bytes of an L2CAP header.
        for byte in [2, 0x0b, 0x20, 4, 0, 0, 0, 1, 0] {
            assert_eq!(line.byte(byte), Ok(Vec::new()));
        }
        // SCO (3) is not modeled.
        assert!(line.byte(3).is_err());
    }

    #[test]
    fn saved_line_restores_into_a_machine_built_the_same_way() {
        let mut line = line();
        for byte in [1, 0x1a, 0x0c] {
            line.byte(byte).unwrap();
        }
        let saved = line.save().unwrap();
        let mut again = self::line();
        again.restore(&saved).unwrap();
        assert_eq!(again.byte(1), Ok(Vec::new()));
        assert_eq!(again.byte(2), Ok(vec![4, 14, 4, 1, 0x1a, 0x0c, 0]));
        // A Pro Controller saved in range needs one placed.
        let memory =
            std::env::temp_dir().join(format!("horizon-emu-air-{}.json", std::process::id()));
        line.air.place_pro_controller(&memory).unwrap();
        let saved = line.save().unwrap();
        assert!(self::line().restore(&saved).is_err());
    }

    #[test]
    fn fifo_dma_tail_then_eord_notifies_once() {
        let uart = Serial::new(Box::new(line()), Box::new(|_| {}), Arc::default());
        uart.write(0, 8, 4, 0x41).unwrap();
        uart.write(0, 4, 4, 0x31).unwrap();
        for byte in [1, 3, 0x0c, 0] {
            uart.write(0, 0, 4, byte).unwrap();
        }
        assert_eq!(uart.read(0, 8, 4), Some(0xc4));
        uart.write(0, 4, 4, 0x30).unwrap();
        for _ in 0..4 {
            uart.read(0, 0, 4).unwrap();
        }
        assert_eq!(uart.read(0, 8, 4), Some(0xc8));
        uart.write(0, 4, 4, 0x10).unwrap();
        uart.write(0, 4, 4, 0x30).unwrap();
        assert_eq!(uart.read(0, 8, 4), Some(0xcc));
        for _ in 0..3 {
            uart.read(0, 0, 4).unwrap();
        }
        assert_eq!(uart.read(0, 8, 4), Some(0xc1));
    }
}
