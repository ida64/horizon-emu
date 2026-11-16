//! STM FingerTip S (FTM4CD60D), on I2C3 at 0x49.
//!
//! Public protocol consulted (not copied):
//! <https://github.com/CTCaer/hekate/blob/master/bdk/input/touch.h> and `touch.c`.
//! Horizon 23.5.0 hid text 0xde800 resets with B6 00 28 80; 0xdea80
//! reads B6 00 04; 0xdee10 reads event count; 0xdee70 drains eight-byte
//! events; 0xe0cc0 queries vendor GPIOs. Unknown commands refuse.
//! GPIO assignments (reset J7, active-low IRQ X1): public Atmosphere
//! `gpio_internal_pad_map_combination.inc`, consulted rather than copied.
use super::{Notes, i2c::I2cDevice};
use crate::snapshot::{Snapshot, SnapshotError, decode, encode};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
pub const ADDRESS: u8 = 0x49;
/// Public FTM4CD60D identity; hid 0xde410 compares the bytes 36 70.
const CHIP_ID: [u8; 2] = [0x36, 0x70];
/// Assumed preinstalled image matches shipped 23.5.0 data title
/// 0100000000000822 `TouchScreenFirmwareInfo.csv`, FTM straps 1,1,1.
/// hid 0xa01b0 selects this row: no firmware update is necessary.
const FIRMWARE_ID: u32 = 0x3200_0001;
/// Version at offset 4 of shipped `FTS_32000001.fts256`.
const FIRMWARE_VERSION: u16 = 0x0681;
/// Assumed NISSHA NFT-K12D panel straps, public hekate panel table.
const PANEL_GPIO: [u8; 3] = [1, 1, 1];
/// Assumed location of the emulated firmware metadata in controller RAM.
/// hid 0xde5f0 follows this pointer, not a hardcoded physical chip address.
const INFO_ADDRESS: u16 = 0x100;

/// The finger on the glass: shared between the controller model and whoever
/// drives it (the window's mouse, a scripted `--touch`).
///
/// Coordinates are the screen's as someone holding the console sees it:
/// 1280x720, origin top left. hid takes them from contact events unchanged
/// (text `0xdf460` unpacks X from bytes 1 and 3 high nibble, Y from bytes 2
/// and 3 low nibble; `0xc6aa0` stores them as the client-visible touch
/// state's X and Y), so these are the coordinates applications see.
/// Events are queued immediately, so a press and release between two of
/// hid's polls are both reported.
pub struct TouchInput {
    state: Mutex<State>,
    interrupt: Box<dyn Fn(bool) + Send + Sync>,
}
impl Default for TouchInput {
    fn default() -> Self {
        Self::new(Box::new(|_| {}))
    }
}
/// The controller's state, and the finger the host holds on the glass,
/// which a snapshot saves with it: a restored console is touched where the
/// saved one was.
#[derive(Serialize, Deserialize)]
struct State {
    contact: Option<(u16, u16)>,
    events: [[u8; 8]; 32],
    count: usize,
    sensing: bool,
    reset_high: bool,
    interrupt_control: u8,
    sense_mode: u8,
}
impl Default for State {
    fn default() -> Self {
        Self {
            contact: None,
            events: [[0; 8]; 32],
            count: 0,
            sensing: false,
            reset_high: false,
            interrupt_control: 0,
            sense_mode: 1,
        }
    }
}
impl State {
    fn event(&mut self, event: [u8; 8]) {
        if self.count < 32 {
            self.events[self.count] = event;
            self.count += 1;
        }
    }
    fn reset(&mut self) {
        self.count = 0;
        self.sensing = false;
        self.contact = None;
        self.interrupt_control = 0;
        self.event([0x10, 0, 0, 0, 0, 0, 0, 0]);
    }
}
impl TouchInput {
    /// Callback takes the asserted level (true pulls the board pin low).
    pub fn new(interrupt: Box<dyn Fn(bool) + Send + Sync>) -> Self {
        Self {
            state: Mutex::new(State::default()),
            interrupt,
        }
    }
    fn update(&self, state: &State) {
        (self.interrupt)(state.count != 0 && state.interrupt_control & 0x40 != 0);
    }
    /// `Some((x, y))` puts the finger down or moves it there, `None` lifts
    /// it. Returns whether the controller reported the change: it reports
    /// nothing while hid has not switched sensing on (command 0x93), as a
    /// controller that is not scanning would not.
    pub fn set_contact(&self, contact: Option<(u16, u16)>) -> bool {
        let mut s = self.state.lock().unwrap();
        let next = contact.map(|(x, y)| (x.min(1279), y.min(719)));
        if !s.sensing {
            return false;
        }
        // Enter, leave and motion events (hid table at 0x1b2a3b maps them to
        // its contact start, end and move).
        let kind = match (s.contact, next) {
            (None, Some(_)) => 3,
            (Some(_), None) => 4,
            (Some(old), Some(new)) if old != new => 5,
            _ => return true,
        };
        let (x, y) = next.or(s.contact).unwrap();
        // Byte 0: event and touch ID 0 in the high nibble; bytes 4-5 contact
        // size and byte 6 its shape, which hid turns into the reported
        // diameter (0xdf460): an assumed fingertip, not a measured one.
        s.event([
            kind,
            (x >> 4) as u8,
            (y >> 4) as u8,
            ((x & 15) << 4 | (y & 15)) as u8,
            16,
            0,
            1,
            0,
        ]);
        s.contact = next;
        self.update(&s);
        true
    }
    /// Rising edge on active-low hardware reset emits CONTROLLER_READY.
    pub fn set_reset(&self, high: bool) {
        let mut s = self.state.lock().unwrap();
        if high && !s.reset_high {
            s.reset();
        }
        if !high {
            s.count = 0;
            s.sensing = false;
            s.contact = None;
        }
        s.reset_high = high;
        self.update(&s);
    }
}
/// Saved by the SoC ([`super::Soc::snapshot_parts`]): the controller's
/// event queue, modes and the contact held at the time. The interrupt pin
/// is the GPIO's input, which the GPIO saves; it is not driven on restore.
impl Snapshot for TouchInput {
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        encode(&*self.state.lock().unwrap())
    }
    fn restore(&self, state: &[u8]) -> Result<(), SnapshotError> {
        *self.state.lock().unwrap() = decode("touch input", state)?;
        Ok(())
    }
}
pub struct Touch {
    input: Arc<TouchInput>,
    command: [u8; 3],
    notes: Arc<Notes>,
}
impl Touch {
    pub fn new(input: Arc<TouchInput>, notes: Arc<Notes>) -> Self {
        Self {
            input,
            command: [0; 3],
            notes,
        }
    }
    fn refuse(&self, data: &[u8]) -> Option<bool> {
        self.notes.add(format!(
            "touch: unsupported command {:02x?}, access {:02x?}",
            self.command, data
        ));
        None
    }
}
impl I2cDevice for Touch {
    fn write(&mut self, data: &[u8]) -> Option<bool> {
        let mut s = self.input.state.lock().unwrap();
        match data {
            [0xb6, 0, 0x28, 0x80] => s.reset(),
            [0xb6, 0, 0x2c, value] => s.interrupt_control = *value,
            [0xb6, a, b] | [0xd0, a, b] => self.command = [data[0], *a, *b],
            [cmd @ 0x85..=0x87] => self.command = [*cmd, 0, 0],
            [0xa1] => s.count = 0,
            [0x93] => s.sensing = true,
            [0x92] => s.sensing = false,
            [0xcf, 1] => s.event([
                0x20,
                1,
                PANEL_GPIO[0],
                PANEL_GPIO[1],
                PANEL_GPIO[2],
                0,
                0,
                0,
            ]),
            [0xcf, 2] => {
                let mode = s.sense_mode;
                s.event([0x20, 2, mode, 0, 0, 0, 0, 0]);
            }
            [0xc3, mode @ 0..=2] => s.sense_mode = *mode,
            [0x97] => {} // Low-power oscillator trim has no completion event.
            [0xa3] => s.event([0x16, 1, 0, 0, 0, 0, 0, 0]),
            [0xa4] => s.event([0x16, 2, 0, 0, 0, 0, 0, 0]),
            [0xfc] => s.event([0x16, 4, 0, 0, 0, 0, 0, 0]),
            [0xa7] => s.event([0x0f, 5, 0, 0, 0, 0, 0, 0]),
            // hid 0xe0e10 programs detection thresholds, not a readback bank.
            [0xb0, 1, 0xd2, _, _]
            | [0xb0, 1, 0xdb, _]
            | [0xb0, 1, 0x64, _]
            | [0xb0, 1, 0xd4, _] => {}
            _ => return self.refuse(data),
        }
        self.input.update(&s);
        Some(true)
    }
    fn read(&mut self, data: &mut [u8]) -> Option<bool> {
        let mut s = self.input.state.lock().unwrap();
        match self.command {
            [0xb6, 0, 4] if data.len() == 7 || data.len() == 3 => {
                // Version and configuration bytes are consistent with the
                // shipped firmware family; no measured panel calibration.
                let info = [
                    0,
                    CHIP_ID[0],
                    CHIP_ID[1],
                    (FIRMWARE_VERSION >> 8) as u8,
                    FIRMWARE_VERSION as u8,
                    0,
                    0,
                ];
                data.copy_from_slice(&info[..data.len()]);
            }
            [0xb6, 0, 0x23] if data.len() == 2 => data.copy_from_slice(&[0, (s.count as u8) << 1]),
            [0xb6, 0, 0x2c] if data.len() == 2 => data.copy_from_slice(&[0, s.interrupt_control]),
            [0xd0, 0, 0x60] if data.len() == 3 => {
                data.copy_from_slice(&[0, INFO_ADDRESS as u8, (INFO_ADDRESS >> 8) as u8])
            }
            [0xd0, 1, 0] if data.len() == 9 => {
                data[0] = 0;
                data[1..5].copy_from_slice(&FIRMWARE_ID.to_be_bytes());
                data[5..7].copy_from_slice(&FIRMWARE_VERSION.to_le_bytes());
                // Configuration revision zero is assumed, not factory trim.
                data[7..9].fill(0);
            }
            [cmd @ 0x85..=0x87, 0, 0] if data.len().is_multiple_of(8) => {
                data.fill(0);
                for chunk in data.as_chunks_mut::<8>().0 {
                    if s.count == 0 {
                        break;
                    }
                    chunk.copy_from_slice(&s.events[0]);
                    s.count -= 1;
                    let count = s.count;
                    s.events.copy_within(1..=count, 0);
                    chunk[7] = (chunk[7] & 0xe0) | (count.min(31) as u8);
                }
                if cmd == 0x87 {
                    s.count = 0;
                }
            }
            _ => return self.refuse(data),
        }
        self.input.update(&s);
        Some(true)
    }
    /// The command being read; the rest is [`TouchInput`]'s.
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        encode(&self.command)
    }
    fn restore(&mut self, state: &[u8]) -> Result<(), SnapshotError> {
        self.command = decode("touch", state)?;
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    #[test]
    fn reset_event_and_interrupt_are_consumed() {
        let irq = Arc::new(AtomicBool::new(false));
        let line = irq.clone();
        let input = Arc::new(TouchInput::new(Box::new(move |v| {
            line.store(v, Ordering::SeqCst)
        })));
        let mut t = Touch::new(input.clone(), Arc::default());
        input.set_reset(true);
        assert_eq!(t.write(&[0xb6, 0, 0x2c, 0x48]), Some(true));
        assert!(irq.load(Ordering::SeqCst));
        t.write(&[0xb6, 0, 0x23]);
        let mut count = [0; 2];
        t.read(&mut count);
        assert_eq!(count, [0, 2]);
        t.write(&[0x85]);
        let mut event = [0; 8];
        t.read(&mut event);
        assert_eq!(event[0], 0x10);
        assert!(!irq.load(Ordering::SeqCst));
        t.read(&mut event);
        assert_eq!(event, [0; 8]);
        t.write(&[0xb6, 0, 0xff]);
        assert_eq!(t.read(&mut count), None);
    }
    /// What hid's contact parser (text 0xdf460) takes from an event: event
    /// kind, X, Y.
    fn decoded(event: [u8; 8]) -> (u8, u16, u16) {
        let x = u16::from(event[1]) << 4 | u16::from(event[3] >> 4);
        let y = u16::from(event[2]) << 4 | u16::from(event[3] & 15);
        (event[0] & 15, x, y)
    }
    #[test]
    fn contacts_enter_move_leave_without_losing_short_clicks() {
        let irq = Arc::new(AtomicBool::new(false));
        let line = irq.clone();
        let input = Arc::new(TouchInput::new(Box::new(move |v| {
            line.store(v, Ordering::SeqCst)
        })));
        let mut t = Touch::new(input.clone(), Arc::default());
        // Not scanning yet: nothing is reported.
        assert!(!input.set_contact(Some((5, 5))));
        t.write(&[0x93]);
        t.write(&[0xb6, 0, 0x2c, 0x40]);
        assert!(input.set_contact(Some((100, 200))));
        assert!(irq.load(Ordering::SeqCst));
        assert!(input.set_contact(Some((100, 200))));
        input.set_contact(Some((1279, 719)));
        input.set_contact(Some((4000, 4000)));
        input.set_contact(None);
        t.write(&[0xb6, 0, 0x23]);
        let mut count = [0; 2];
        t.read(&mut count);
        assert_eq!(count[1] >> 1, 3);
        t.write(&[0x85]);
        let mut event = [0; 8];
        for (left, expected) in [(2, (3, 100, 200)), (1, (5, 1279, 719)), (0, (4, 1279, 719))] {
            t.read(&mut event);
            assert_eq!(decoded(event), expected);
            assert_eq!(event[7] & 0x1f, left);
        }
        assert!(!irq.load(Ordering::SeqCst));
        t.read(&mut event);
        assert_eq!(event, [0; 8]);
    }
}
