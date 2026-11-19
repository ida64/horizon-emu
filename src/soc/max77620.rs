//! MAX77620 power management IC on I2C5: the regulator and power-key block
//! at address 0x3C and its real-time clock at 0x68.
//!
//! The console was switched on with its power button, runs from the
//! battery with no adapter attached, and nothing has gone wrong: that is
//! what the event and status registers say. Registers the driver programs
//! hold what was written.
//!
//! What the earlier boot stages left in the configuration registers is not
//! known here. The firmware's power driver reads each one once, to change
//! some of its bits, when it sets the regulators up from its own tables
//! (pcv text `0x1cce0`); those registers start at zero, an assumption about
//! the bits it keeps (registers `0x16` to `0x57`). Event masks start with
//! everything masked (`0x0D` to `0x12`). Any other register refuses to be
//! read.
//!
//! On a power-on boot the NX bootloader programs the chip first, in normal
//! I2C mode (its `0x400007D0`, `0x40000872`, `0x40000806`, `0x40002840`):
//! `CNFGBBC` (0x04), `ONOFFCNFG1` (0x41), the power sequencers (0x43-0x56),
//! `SD0`/`SD1` (0x16/0x17), `SD_CFG2` (0x22), `GPIO5` (0x3B, the CPU rail's
//! enable) and `AME_GPIO` (0x40, read first, bit 5 cleared). Every one is
//! held as written; `AME_GPIO` reads zero before, as the other
//! configuration registers do.
//!
//! The two blocks share the chip's interrupt output, which is the SoC's
//! PMU interrupt (pcv binds interrupt 118, level-triggered) and one of its
//! wake sources: [`Max77620`] is the chip as a whole. Two things happen on
//! it: the power key and the clock's alarms.
//!
//! The firmware ends a run of the console through the chip: Power Options
//! → Restart and → Turn Off both write `ONOFFCNFG2` (0x42) and then
//! `ONOFFCNFG1` (0x41), read-modify-write, from pcv: `0x42 = 0x81`,
//! `0x41 = 0xB8` to restart, `0x42 = 0x01`, `0x41 = 0xB8` to turn off
//! (traced through this model). `ONOFFCNFG1` bit 7 is the software reset
//! (`SFT_RST`, Linux `max77620.h`), which switches every regulator off
//! through the power sequencers; `ONOFFCNFG2` bit 7 (`SFT_RST_WK`) says
//! whether the chip then powers the console up again, a restart, or stays
//! off ([`PowerDown`]). \[inference\] That `SFT_RST_WK` decides between the
//! two is read off its name and these two writes, which differ in it
//! alone. The chip powers down at once here: [`Max77620::power_down`] says
//! so to the machine, which stops. `ONOFFCNFG1` bit 1 (`PWR_OFF`) is not
//! modeled and refuses to be set. What runs on through the power cycle is
//! the clock, on its own backup battery ([`BatteryBacked`]).

use std::{
    collections::BTreeMap,
    sync::{Arc, Condvar, Mutex, Weak},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};

use super::i2c::RegisterMap;
use crate::{
    clock::GuestClock,
    snapshot::{Snapshot, SnapshotError, decode, encode},
};

pub const PMIC_ADDRESS: u8 = 0x3C;
pub const RTC_ADDRESS: u8 = 0x68;

/// Last register of the power block (`CID5`).
const PMIC_LAST_REGISTER: u8 = 0x5D;

/// The 32 kHz oscillator: bit 7 says it runs, bit 2 drives its output
/// pin, which is the SoC's 32 kHz clock (bit meanings recalled from
/// open-source drivers). Both hold on a running console. The clock driver
/// rewrites bits 5:4 and 1:0 when it starts (pcv text `0x16850`), so their
/// factory settings, not known here, do not matter; bits 6 and 3 read
/// zero, which is an assumption.
const CNFG1_32K: u8 = 0x03;
const CNFG1_32K_AT_START: u8 = 1 << 7 | 1 << 2;
/// Which blocks have an event pending that their own mask lets through
/// (bit names as in Linux, `include/linux/mfd/max77620.h`). Only the
/// clock and the power key ever have one here.
const IRQTOP: u8 = 0x05;
const IRQTOP_RTC: u8 = 1 << 3;
const IRQTOP_ONOFF: u8 = 1 << 1;
/// Mask of the chip's interrupt sources by block, laid out as [`IRQTOP`].
/// Everything masked.
const IRQTOPM: u8 = 0x0D;
const IRQTOPM_AT_START: u8 = 0xFF;
/// Mask of the chip's own events (low battery, two temperature alarms,
/// bits 3 to 1) and, in bit 0, of its interrupt output as a whole (Linux's
/// `GLBLM`: `max77620_irq_global_mask` sets it, `..._unmask` clears it).
/// The events start masked, the output does not: the firmware's driver
/// unmasks the events one by one and writes bit 0 back as it found it
/// (`0x09`, then `0x01`, had this started as `0x0f`), yet it binds the output's
/// interrupt, lets the clock and the power key through [`IRQTOPM`], sets
/// an alarm before sleeping and has the output among the wake sources it
/// enables (Atmosphère's table of them for this board, `PwrIntN`). So the
/// output is not masked when the kernel starts, an inference.
const INTENLBT: u8 = 0x0E;
const INTENLBT_AT_START: u8 = 0x0E;
const INTENLBT_OUTPUT: u8 = 1 << 0;
/// Power key and adapter events since they were last read; reading clears
/// them. Bit 3: the power key went down; bit 2: it came up (hekate's
/// `EN0_R` and `EN0_F`). The chip also has an event for the key being
/// held a second (bit 1), which is not modeled.
const ONOFFIRQ: u8 = 0x0B;
const ONOFFIRQ_POWER_KEY_PRESSED: u8 = 1 << 3;
const ONOFFIRQ_POWER_KEY_RELEASED: u8 = 1 << 2;
/// Mask of [`ONOFFIRQ`].
const ONOFFIRQM: u8 = 0x12;
/// Why the previous power-down happened, if it was forced. Nothing was.
const NVERC: u8 = 0x0C;
/// Present levels of the power key (bit 2, hekate's `EN0`), adapter
/// detection and lid inputs. No adapter, and the lid input stays
/// inactive.
const ONOFFSTAT: u8 = 0x15;
const ONOFFSTAT_POWER_KEY: u8 = 1 << 2;
/// The version of the chip's factory programming. The firmware knows two,
/// and with this one it uses the regulator table of the original console:
/// 1.125 V memory, 1.35 V pre-regulator (pcv text `0x195d0`, tables at
/// `0x9e350` and `0x9f790`).
const CID4: u8 = 0x5C;
const CID4_ORIGINAL_CONSOLE: u8 = 0x35;
/// Regulator voltages and modes, the chip's pins, the power-key behavior
/// and the power sequencers (`SD0` to `FPS_RSO`).
const CONFIGURATION: std::ops::RangeInclusive<u8> = 0x16..=0x57;
/// Masks of the regulator and power-key events (`IRQMASKSD` to `ONOFFIRQM`).
const EVENT_MASKS: std::ops::RangeInclusive<u8> = 0x0F..=0x12;
/// One bit per step-down regulator whose output has failed. None has.
const STATSD: u8 = 0x14;
/// Each of the nine linear regulators has a pair of registers from here:
/// voltage and, in the top two bits, power mode (zero: off); then options
/// and [`LDO_CFG2_POWER_OK`].
const LDO0_CFG: u8 = 0x23;
const LDO_COUNT: u8 = 9;
const LDO_CFG_POWER_MODE: u8 = 0b11 << 6;
/// The regulator's output has reached its voltage. The driver waits for
/// this after switching one on (pcv text `0x1c4c0`); outputs settle at
/// once here.
const LDO_CFG2_POWER_OK: u8 = 1 << 3;
/// Each linear regulator's place in the power sequences, one register per
/// regulator from here. The top two bits name the sequencer that switches
/// it; with [`FPS_SOURCE_NONE`] only its power mode does.
const FPS_LDO0: u8 = 0x46;
const FPS_SOURCE: u8 = 0b11 << 6;
const FPS_SOURCE_NONE: u8 = 0b11 << 6;

/// Power-key behavior and the chip's own power-down (Linux `max77620.h`,
/// `MAX77620_REG_ONOFFCNFG1`): [`ONOFFCNFG1_SFT_RST`] and
/// [`ONOFFCNFG1_PWR_OFF`] act, the rest (the manual-reset time, the sleep
/// enable) holds what is written.
const ONOFFCNFG1: u8 = 0x41;
/// The software reset (`MAX77620_ONOFFCNFG1_SFT_RST`): every regulator
/// off, then what [`ONOFFCNFG2_SFT_RST_WK`] says.
const ONOFFCNFG1_SFT_RST: u8 = 1 << 7;
/// `MAX77620_ONOFFCNFG1_PWR_OFF`: not modeled, refused.
const ONOFFCNFG1_PWR_OFF: u8 = 1 << 1;
/// The chip's wake sources (`MAX77620_REG_ONOFFCNFG2`), held as written.
const ONOFFCNFG2: u8 = 0x42;
/// `MAX77620_ONOFFCNFG2_SFT_RST_WK`: after a software reset the chip powers
/// the console up again ([`PowerDown::Restart`]) instead of staying off.
const ONOFFCNFG2_SFT_RST_WK: u8 = 1 << 7;

/// How a software reset of the chip (`ONOFFCNFG1` `SFT_RST`) ends the run
/// of the console: every regulator goes off, and the SoC and DRAM lose
/// power as when the battery is cut.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerDown {
    /// With `ONOFFCNFG2` `SFT_RST_WK` set: the chip starts its power-up
    /// sequence again, and the console boots as from the power key.
    Restart,
    /// With it clear: the chip stays off. \[assumption\] The wake events
    /// that would power it up again (the power key, an alarm) are not
    /// modeled: nothing runs after this.
    Off,
}

/// The power block of the PMIC.
#[derive(Serialize, Deserialize)]
struct Pmic {
    registers: BTreeMap<u8, u8>,
    /// The software reset, once it is written. Not saved: a snapshot
    /// refuses to be taken once it is ([`Snapshot for Max77620`](Max77620)).
    #[serde(skip)]
    power_down: Option<PowerDown>,
}

impl Pmic {
    fn new() -> Self {
        let mut registers: BTreeMap<u8, u8> = CONFIGURATION
            .map(|register| (register, 0))
            .chain(EVENT_MASKS.map(|register| (register, 0xFF)))
            .collect();
        registers.extend([
            (CNFG1_32K, CNFG1_32K_AT_START),
            (IRQTOPM, IRQTOPM_AT_START),
            (INTENLBT, INTENLBT_AT_START),
            (ONOFFIRQ, ONOFFIRQ_POWER_KEY_PRESSED),
            (NVERC, 0),
            (ONOFFSTAT, 0),
            (CID4, CID4_ORIGINAL_CONSOLE),
            (STATSD, 0),
        ]);
        Self {
            registers,
            power_down: None,
        }
    }

    /// The power key going down or coming up.
    fn power_key(&mut self, held: bool) {
        let was_held = self.registers[&ONOFFSTAT] & ONOFFSTAT_POWER_KEY != 0;
        if held == was_held {
            return;
        }
        let (status, event) = if held {
            (ONOFFSTAT_POWER_KEY, ONOFFIRQ_POWER_KEY_PRESSED)
        } else {
            (0, ONOFFIRQ_POWER_KEY_RELEASED)
        };
        self.registers.insert(ONOFFSTAT, status);
        *self.registers.entry(ONOFFIRQ).or_default() |= event;
    }

    /// Whether a power-key event is pending that its mask lets through.
    fn power_key_event(&self) -> bool {
        self.registers[&ONOFFIRQ] & !self.registers[&ONOFFIRQM] != 0
    }
}

/// The chip: both blocks and the interrupt output they share.
#[derive(Serialize, Deserialize)]
struct Chip {
    pmic: Pmic,
    rtc: Rtc,
}

impl Chip {
    /// What [`IRQTOP`] reads.
    fn sources(&mut self) -> u8 {
        let key = if self.pmic.power_key_event() {
            IRQTOP_ONOFF
        } else {
            0
        };
        let alarm = if self.rtc.alarm_due() { IRQTOP_RTC } else { 0 };
        key | alarm
    }

    /// The level of the interrupt output.
    fn interrupt(&mut self) -> bool {
        let output_masked = self.pmic.registers[&INTENLBT] & INTENLBT_OUTPUT != 0;
        !output_masked && self.sources() & !self.pmic.registers[&IRQTOPM] != 0
    }
}

/// Longest the alarm watcher waits before it looks whether the chip still
/// exists.
const WATCH_AT_MOST: Duration = Duration::from_millis(250);

/// The chip's state, and what tells the alarm watcher it changed.
struct Shared {
    chip: Mutex<Chip>,
    changed: Condvar,
}

/// The chip as a whole: what its two bus addresses, the power key and the
/// interrupt output are parts of.
#[derive(Clone)]
pub struct Max77620 {
    shared: Arc<Shared>,
    interrupt: Arc<dyn Fn(bool) + Send + Sync>,
}

impl Max77620 {
    /// A chip whose clock shows the host's time when the machine is built
    /// and runs on in guest time (`clock`), and whose interrupt output
    /// drives `interrupt`.
    pub fn new(clock: Arc<GuestClock>, interrupt: impl Fn(bool) + Send + Sync + 'static) -> Self {
        let host = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |since| since.as_secs() as i64);
        Self::starting_at(host, clock, interrupt)
    }

    /// A chip whose clock shows `base` (seconds since the Unix epoch) at
    /// guest time zero.
    fn starting_at(
        base: i64,
        clock: Arc<GuestClock>,
        interrupt: impl Fn(bool) + Send + Sync + 'static,
    ) -> Self {
        let chip = Self {
            shared: Arc::new(Shared {
                chip: Mutex::new(Chip {
                    pmic: Pmic::new(),
                    rtc: Rtc::new(Arc::clone(&clock), base),
                }),
                changed: Condvar::new(),
            }),
            interrupt: Arc::new(interrupt),
        };
        let (shared, interrupt) = (Arc::downgrade(&chip.shared), Arc::clone(&chip.interrupt));
        std::thread::spawn(move || watch_alarms(&shared, &*interrupt, &clock));
        chip
    }

    /// The power block, as its bus address shows it.
    pub fn power_block(&self) -> impl RegisterMap + use<> {
        Block {
            chip: self.clone(),
            clock: false,
        }
    }

    /// The clock, as its bus address shows it.
    pub fn clock_block(&self) -> impl RegisterMap + use<> {
        Block {
            chip: self.clone(),
            clock: true,
        }
    }

    /// The power key going down (`held`) or coming up.
    pub fn power_key(&self, held: bool) {
        self.update(|chip| chip.pmic.power_key(held));
    }

    /// The level of the interrupt output: what wakes a sleeping console.
    pub fn interrupt(&self) -> bool {
        self.update(|_| ()).1
    }

    /// The software reset, once the firmware has written it: the console
    /// has lost power, and what follows is [`PowerDown`].
    pub fn power_down(&self) -> Option<PowerDown> {
        self.shared.chip.lock().unwrap().pmic.power_down
    }

    /// What of the chip runs on through a power cycle, as it is now.
    pub fn battery_backed(&self) -> BatteryBacked {
        let chip = self.shared.chip.lock().unwrap();
        BatteryBacked {
            shows: chip.rtc.now(),
            rtc: chip.rtc.clone(),
        }
    }

    /// Makes this chip, of a machine built and not run yet, the one a
    /// software reset with `SFT_RST_WK` powers up again: the clock goes on
    /// from where `kept` left it, with its registers and alarms, and no
    /// power-key event is pending, since no key started this power-up.
    /// \[assumption\] The software reset clears `ONOFFIRQ` as power-on does
    /// and leaves the clock alone, which runs on its backup battery.
    pub fn power_up_from_reset(&self, kept: BatteryBacked) {
        self.update(|chip| {
            let BatteryBacked { mut rtc, shows } = kept;
            rtc.clock = Arc::clone(&chip.rtc.clock);
            rtc.base = chip.rtc.base;
            rtc.offset = shows - rtc.unset_time();
            chip.rtc = rtc;
            chip.pmic.registers.insert(ONOFFIRQ, 0);
        });
    }

    /// Does `action` on the chip and brings the interrupt output up to
    /// date; returns the output's level along with the action's result.
    fn update<T>(&self, action: impl FnOnce(&mut Chip) -> T) -> (T, bool) {
        let mut chip = self.shared.chip.lock().unwrap();
        let result = action(&mut chip);
        let level = chip.interrupt();
        (self.interrupt)(level);
        // An alarm may have been set or cleared.
        self.shared.changed.notify_all();
        (result, level)
    }
}

/// The part of the chip that a power cycle leaves running: the real-time
/// clock with its registers and alarms, and the time it showed when
/// [`Max77620::battery_backed`] took it.
pub struct BatteryBacked {
    rtc: Rtc,
    /// Seconds since the Unix epoch.
    shows: i64,
}

/// Both blocks' registers, the power key's state and the clock: its time
/// as a base and the guest's offset from it, and its alarms, which go off
/// at the same guest time in the restored machine. Saved by the SoC
/// ([`super::Soc::snapshot_parts`]); the bus addresses save nothing. The
/// interrupt output is not driven on restore: the interrupt controller
/// saved its level.
impl Snapshot for Max77620 {
    /// Sets off the alarms due by the paused guest time, which the alarm
    /// watcher might otherwise do in the middle of the save. A chip that
    /// has had its software reset written is powering the console down,
    /// which is not saved: it refuses.
    fn quiesce(&self) -> Result<(), SnapshotError> {
        let mut chip = self.shared.chip.lock().unwrap();
        if let Some(down) = chip.pmic.power_down {
            return Err(SnapshotError::Unsupported(format!(
                "the MAX77620 while it powers the console down ({down:?})"
            )));
        }
        if chip.rtc.ring() {
            (self.interrupt)(chip.interrupt());
        }
        Ok(())
    }

    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        encode(&*self.shared.chip.lock().unwrap())
    }

    fn restore(&self, state: &[u8]) -> Result<(), SnapshotError> {
        let mut saved: Chip = decode("MAX77620", state)?;
        let mut chip = self.shared.chip.lock().unwrap();
        saved.rtc.clock = Arc::clone(&chip.rtc.clock);
        *chip = saved;
        self.shared.changed.notify_all();
        Ok(())
    }
}

/// Keeps the interrupt output following the clock's alarms for as long as
/// the chip exists: when guest time reaches an alarm, the alarm goes off
/// and the output takes its new level. Guest time standing still, no alarm
/// goes off.
fn watch_alarms(
    shared: &Weak<Shared>,
    interrupt: &(dyn Fn(bool) + Send + Sync),
    clock: &GuestClock,
) {
    while let Some(shared) = shared.upgrade() {
        let mut chip = shared.chip.lock().unwrap();
        if chip.rtc.ring() {
            interrupt(chip.interrupt());
        }
        let wait = chip
            .rtc
            .next_alarm()
            .map_or(WATCH_AT_MOST, |due| clock.until(due).min(WATCH_AT_MOST));
        drop(shared.changed.wait_timeout(chip, wait).unwrap());
    }
}

/// One of the chip's two bus addresses.
struct Block {
    chip: Max77620,
    clock: bool,
}

impl RegisterMap for Block {
    fn read(&mut self, register: u8) -> Option<u8> {
        let clock = self.clock;
        self.chip
            .update(|chip| match (clock, register) {
                (true, _) => chip.rtc.read(register),
                (false, IRQTOP) => Some(chip.sources()),
                (false, _) => chip.pmic.read(register),
            })
            .0
    }

    fn write(&mut self, register: u8, value: u8) -> Option<()> {
        let clock = self.clock;
        self.chip
            .update(|chip| {
                if clock {
                    chip.rtc.write(register, value)
                } else {
                    chip.pmic.write(register, value)
                }
            })
            .0
    }

    /// Nothing: the chip is one part, which the SoC saves
    /// ([`Snapshot for Max77620`](Max77620)).
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        crate::bus::NoState.save()
    }

    fn restore(&mut self, state: &[u8]) -> Result<(), SnapshotError> {
        crate::bus::NoState.restore(state)
    }
}

/// The power block's registers as the bus sees them.
impl Pmic {
    fn read(&mut self, register: u8) -> Option<u8> {
        let value = *self.registers.get(&register)?;
        if register == ONOFFIRQ {
            self.registers.insert(register, 0);
        }
        let ldo = register.wrapping_sub(LDO0_CFG) / 2;
        if register > LDO0_CFG && ldo < LDO_COUNT && (register - LDO0_CFG) % 2 == 1 {
            // The sequencers have all run: the console is up.
            let mode = self.registers[&(LDO0_CFG + 2 * ldo)] & LDO_CFG_POWER_MODE;
            let sequenced = self.registers[&(FPS_LDO0 + ldo)] & FPS_SOURCE != FPS_SOURCE_NONE;
            let ok = mode != 0 || sequenced;
            return Some(value & !LDO_CFG2_POWER_OK | if ok { LDO_CFG2_POWER_OK } else { 0 });
        }
        Some(value)
    }

    fn write(&mut self, register: u8, value: u8) -> Option<()> {
        match register {
            // Event and status registers follow the inputs, not the bus.
            IRQTOP | ONOFFIRQ | NVERC | ONOFFSTAT => {}
            ONOFFCNFG1 if value & ONOFFCNFG1_PWR_OFF != 0 => return None,
            ONOFFCNFG1 => {
                if value & ONOFFCNFG1_SFT_RST != 0 {
                    let wakes = self.registers[&ONOFFCNFG2] & ONOFFCNFG2_SFT_RST_WK != 0;
                    self.power_down = Some(if wakes {
                        PowerDown::Restart
                    } else {
                        PowerDown::Off
                    });
                }
                // [assumption] The reset bit acts and reads back clear.
                self.registers.insert(register, value & !ONOFFCNFG1_SFT_RST);
            }
            _ if register <= PMIC_LAST_REGISTER => {
                self.registers.insert(register, value);
            }
            _ => return None,
        }
        Some(())
    }
}

/// Pending clock events; reading clears them. Of the six only the two
/// alarms ever happen here (bit positions as in the Linux driver,
/// `max77686-private.h`).
const RTCINT: u8 = 0x00;
const RTCINT_ALARM: [u8; 2] = [1 << 1, 1 << 2];
/// Event mask, all six events masked as after power-on.
const RTCINTM: u8 = 0x01;
const RTCINTM_AT_POWER_ON: u8 = 0x3F;
/// Which bits of a write to [`RTCCNTL`] take effect.
const RTCCNTLM: u8 = 0x02;
/// How the time and alarm registers count: [`RTCCNTL_BCD`],
/// [`RTCCNTL_24_HOUR`].
const RTCCNTL: u8 = 0x03;
const RTCCNTL_BCD: u8 = 1 << 0;
const RTCCNTL_24_HOUR: u8 = 1 << 1;
/// Requests a transfer between the clock and the registers the bus sees:
/// [`RTCUPDATE0_WRITE`] makes what was written take effect,
/// [`RTCUPDATE0_READ`] captures the clock for reading. The bits clear
/// themselves; transfers here take no time.
const RTCUPDATE0: u8 = 0x04;
const RTCUPDATE0_WRITE: u8 = 1 << 0;
const RTCUPDATE0_READ: u8 = 1 << 4;
/// Which transfers have completed since this was last read (clearing on
/// read is recalled, like the bit positions): the driver looks here after
/// requesting one.
const RTCUPDATE1: u8 = 0x05;
const RTCUPDATE1_WRITE_DONE: u8 = 1 << 0;
const RTCUPDATE1_READ_DONE: u8 = 1 << 1;
/// Seconds, minutes, hours, day of week (one bit per day, Sunday first),
/// month, year (two digits) and day of month follow one another.
const RTCSEC: u8 = 0x07;
const RTCDOM: u8 = 0x0D;
/// In 12-hour mode the hour register counts 1 to 12 and this bit says "PM".
const RTCHOUR_PM: u8 = 1 << 6;
/// The two alarms, each with the layout of the time registers plus an
/// enable bit in every register: the alarm goes off when every enabled
/// field matches the clock.
const ALARM1_SEC: u8 = 0x0E;
const ALARM2_SEC: u8 = 0x15;
const ALARM2_DOM: u8 = 0x1B;
const ALARM_ENABLE: u8 = 1 << 7;
/// The fields the Linux driver enables for an alarm at a date and time
/// (`max77686_rtc_start_alarm`): all but the day of the week. The
/// firmware's driver sets the same; other selections, which make alarms
/// that repeat, are not modeled.
const ALARM_AT_DATE: [bool; 7] = [true, true, true, false, true, true, true];

/// The year the two-digit year register counts from.
const RTC_EPOCH_YEAR: i64 = 2000;

/// The real-time clock of the PMIC, a clock running on its own battery.
///
/// It shows the host's time as of when the machine was built, runs on in
/// guest time from there, and keeps the difference the guest's setting of
/// it made: a console that has been keeping time, which is what the
/// battery is for. Counting guest time, it stands still while the machine
/// is paused, and a restored machine reads it as the saved one would. The
/// register layout and the transfer protocol are recalled from the chip's
/// Linux driver, not read off a datasheet; the firmware's driver agreeing
/// with them is the evidence. An alarm set for a date and time goes off
/// then: the event shows in [`RTCINT`], and [`Rtc::alarm_due`] is how what
/// the alarm is wired to learns of it.
#[derive(Clone, Serialize, Deserialize)]
struct Rtc {
    /// Guest time, which the clock runs on. Wiring: the restoring machine
    /// has its own.
    #[serde(skip)]
    clock: Arc<GuestClock>,
    /// Seconds since the Unix epoch the clock showed at guest time zero,
    /// before the guest set it: the host's time when the machine was
    /// built.
    base: i64,
    /// What the guest's setting of the clock added to `base`.
    offset: i64,
    event_mask: u8,
    control: u8,
    control_mask: u8,
    /// Registers written since the last write transfer.
    written: BTreeMap<u8, u8>,
    /// Time registers captured by the last read transfer.
    captured: Option<[u8; 7]>,
    transfers_done: u8,
    alarms: BTreeMap<u8, u8>,
    /// When each alarm goes off, in the clock's seconds, while it is set
    /// and has not gone off.
    armed: [Option<i64>; 2],
    /// Alarms that went off and have not been read from [`RTCINT`].
    events: u8,
}

impl Rtc {
    /// A clock that shows `base` (seconds since the Unix epoch) at guest
    /// time zero of `clock` and counts guest time. It counts in binary
    /// with a 24-hour day: the setting the firmware's driver takes as the
    /// sign of a clock that has not lost its time (pcv text `0x16850`,
    /// `RTCCNTL & 3 == 2`).
    fn new(clock: Arc<GuestClock>, base: i64) -> Self {
        Self {
            clock,
            base,
            offset: 0,
            event_mask: RTCINTM_AT_POWER_ON,
            control: RTCCNTL_24_HOUR,
            control_mask: 0,
            written: BTreeMap::new(),
            captured: None,
            transfers_done: 0,
            // No alarm is set, and none ever was.
            alarms: (ALARM1_SEC..=ALARM2_DOM)
                .map(|register| (register, 0))
                .collect(),
            armed: [None; 2],
            events: 0,
        }
    }

    /// The clock's time before the guest's setting: seconds since the Unix
    /// epoch.
    fn unset_time(&self) -> i64 {
        self.base + self.clock.now().as_secs() as i64
    }

    /// The clock's time, in seconds since the Unix epoch.
    fn now(&self) -> i64 {
        self.unset_time() + self.offset
    }

    /// The guest time at which the earliest alarm set goes off.
    fn next_alarm(&self) -> Option<Duration> {
        let at = self.armed.iter().flatten().min()?;
        let after = at - self.base - self.offset;
        Some(Duration::from_secs(after.max(0) as u64))
    }

    fn encode(&self, value: u8) -> u8 {
        if self.control & RTCCNTL_BCD != 0 {
            ((value / 10) << 4) | (value % 10)
        } else {
            value
        }
    }

    fn decode(&self, value: u8) -> u8 {
        if self.control & RTCCNTL_BCD != 0 {
            (value >> 4) * 10 + (value & 0xF)
        } else {
            value
        }
    }

    /// The time registers for the clock's current time.
    fn time_registers(&self) -> [u8; 7] {
        let now = self.now();
        let (days, second_of_day) = (now.div_euclid(86_400), now.rem_euclid(86_400));
        let (year, month, day) = civil_from_days(days);
        let hour = (second_of_day / 3600) as u8;
        let hour = if self.control & RTCCNTL_24_HOUR != 0 {
            self.encode(hour)
        } else {
            let pm = if hour >= 12 { RTCHOUR_PM } else { 0 };
            self.encode((hour + 11) % 12 + 1) | pm
        };
        [
            self.encode((second_of_day % 60) as u8),
            self.encode((second_of_day / 60 % 60) as u8),
            hour,
            // 1970-01-01 was a Thursday.
            1 << (days + 4).rem_euclid(7),
            self.encode(month),
            self.encode((year - RTC_EPOCH_YEAR).rem_euclid(100) as u8),
            self.encode(day),
        ]
    }

    /// The clock's seconds for a full set of time registers.
    fn seconds(&self, registers: [u8; 7]) -> i64 {
        let [second, minute, hour, _day_of_week, month, year, day] = registers;
        let hour = if self.control & RTCCNTL_24_HOUR != 0 {
            self.decode(hour)
        } else {
            let pm = if hour & RTCHOUR_PM != 0 { 12 } else { 0 };
            self.decode(hour & !RTCHOUR_PM) % 12 + pm
        };
        let days = days_from_civil(
            RTC_EPOCH_YEAR + i64::from(self.decode(year)),
            self.decode(month),
            self.decode(day),
        );
        days * 86_400
            + i64::from(hour) * 3600
            + i64::from(self.decode(minute)) * 60
            + i64::from(self.decode(second))
    }

    /// Sets the clock from a full set of time registers.
    fn set_time(&mut self, registers: [u8; 7]) {
        self.offset = self.seconds(registers) - self.unset_time();
    }

    /// When the alarm whose registers start at `first` goes off: `None`
    /// for one with nothing enabled, `Err` for a selection of fields that
    /// is not modeled.
    fn alarm_time(&self, first: u8) -> Result<Option<i64>, ()> {
        let fields: [u8; 7] = std::array::from_fn(|index| self.alarms[&(first + index as u8)]);
        let enabled = fields.map(|field| field & ALARM_ENABLE != 0);
        if enabled == [false; 7] {
            Ok(None)
        } else if enabled == ALARM_AT_DATE {
            Ok(Some(
                self.seconds(fields.map(|field| field & !ALARM_ENABLE)),
            ))
        } else {
            Err(())
        }
    }

    /// Moves the alarms whose time has come to the pending events; whether
    /// any did.
    fn ring(&mut self) -> bool {
        let now = self.now();
        let mut rang = false;
        for (alarm, event) in self.armed.iter_mut().zip(RTCINT_ALARM) {
            if alarm.is_some_and(|at| at <= now) {
                *alarm = None;
                self.events |= event;
                rang = true;
            }
        }
        rang
    }

    /// Whether an alarm has gone off whose event is neither masked nor
    /// read yet: what the clock's interrupt output shows.
    fn alarm_due(&mut self) -> bool {
        self.ring();
        self.events & !self.event_mask != 0
    }

    /// Makes what was written since the last transfer take effect.
    fn commit(&mut self) -> Option<()> {
        let written = std::mem::take(&mut self.written);
        if let Some(&control) = written.get(&RTCCNTL) {
            self.control = self.control & !self.control_mask | control & self.control_mask;
        }
        let time: Vec<u8> = written
            .range(RTCSEC..=RTCDOM)
            .map(|(_, &value)| value)
            .collect();
        match <[u8; 7]>::try_from(time) {
            Ok(time) => self.set_time(time),
            // Setting part of the time: what the rest becomes is not known.
            Err(time) if !time.is_empty() => return None,
            Err(_) => {}
        }
        for (&register, &value) in written.range(ALARM1_SEC..=ALARM2_DOM) {
            self.alarms.insert(register, value);
        }
        for (index, first) in [ALARM1_SEC, ALARM2_SEC].into_iter().enumerate() {
            if written.range(first..first + 7).next().is_some() {
                self.armed[index] = self.alarm_time(first).ok()?;
            }
        }
        Some(())
    }
}

/// Year, month and day of the given day since 1970-01-01, in the Gregorian
/// calendar (the usual era-based conversion).
fn civil_from_days(days: i64) -> (i64, u8, u8) {
    let shifted = days + 719_468;
    let era = shifted.div_euclid(146_097);
    let day_of_era = shifted.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * shifted_month + 2) / 5 + 1;
    let month = if shifted_month < 10 {
        shifted_month + 3
    } else {
        shifted_month - 9
    };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    (year, month as u8, day as u8)
}

/// The inverse of [`civil_from_days`].
fn days_from_civil(year: i64, month: u8, day: u8) -> i64 {
    let year = year - i64::from(month <= 2);
    let era = year.div_euclid(400);
    let year_of_era = year.rem_euclid(400);
    let shifted_month = (i64::from(month) + 9) % 12;
    let day_of_year = (153 * shifted_month + 2) / 5 + i64::from(day) - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

/// The clock's registers as the bus sees them.
impl Rtc {
    fn read(&mut self, register: u8) -> Option<u8> {
        match register {
            RTCINT => {
                self.ring();
                Some(std::mem::take(&mut self.events))
            }
            RTCUPDATE0 => Some(0),
            RTCINTM => Some(self.event_mask),
            RTCCNTLM => Some(self.control_mask),
            RTCCNTL => Some(self.control),
            RTCUPDATE1 => Some(std::mem::take(&mut self.transfers_done)),
            RTCSEC..=RTCDOM => Some(self.captured?[usize::from(register - RTCSEC)]),
            ALARM1_SEC..=ALARM2_DOM => self.alarms.get(&register).copied(),
            _ => None,
        }
    }

    fn write(&mut self, register: u8, value: u8) -> Option<()> {
        match register {
            RTCINT => {}
            RTCINTM => self.event_mask = value,
            RTCCNTLM => self.control_mask = value,
            RTCUPDATE0 => {
                if value & RTCUPDATE0_WRITE != 0 {
                    self.commit()?;
                    self.transfers_done |= RTCUPDATE1_WRITE_DONE;
                }
                if value & RTCUPDATE0_READ != 0 {
                    self.captured = Some(self.time_registers());
                    self.transfers_done |= RTCUPDATE1_READ_DONE;
                }
            }
            RTCCNTL | RTCSEC..=ALARM2_DOM => {
                self.written.insert(register, value);
            }
            _ => return None,
        }
        Some(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::*;

    /// A guest clock standing still at `at`, which only the test moves.
    fn guest_clock(at: Duration) -> Arc<GuestClock> {
        let clock = Arc::new(GuestClock::new());
        clock.pause();
        clock.set(at);
        clock
    }

    #[test]
    fn power_key_event_is_reported_once_and_unknown_state_is_refused() {
        let mut pmic = Pmic::new();
        assert_eq!(pmic.read(ONOFFIRQ), Some(ONOFFIRQ_POWER_KEY_PRESSED));
        assert_eq!(pmic.read(ONOFFIRQ), Some(0));
        pmic.write(ONOFFSTAT, 0xFF).unwrap();
        assert_eq!(pmic.read(ONOFFSTAT), Some(0));

        // A status nothing here has a value for.
        assert_eq!(pmic.read(0x13), None);
        assert_eq!(pmic.read(0x23), Some(0));
        pmic.write(0x23, 0xD0).unwrap();
        assert_eq!(pmic.read(0x23), Some(0xD0));
        assert_eq!(pmic.write(0x5E, 0), None);
    }

    #[test]
    fn clock_reads_in_the_mode_set_and_keeps_what_it_is_set_to() {
        // 2024-02-29 15:04:05 UTC, a Thursday.
        let mut rtc = Rtc::new(guest_clock(Duration::ZERO), 1_709_219_045);
        assert_eq!(rtc.read(RTCSEC), None);
        let time = |rtc: &mut Rtc| {
            rtc.write(RTCUPDATE0, RTCUPDATE0_READ).unwrap();
            [0, 1, 2, 3, 4, 5, 6].map(|index| rtc.read(RTCSEC + index).unwrap())
        };
        assert_eq!(time(&mut rtc), [5, 4, 15, 1 << 4, 2, 24, 29]);

        // A control write only changes the bits the mask names.
        rtc.write(RTCCNTLM, RTCCNTL_BCD).unwrap();
        rtc.write(RTCCNTL, RTCCNTL_BCD).unwrap();
        assert_eq!(rtc.read(RTCCNTL), Some(RTCCNTL_24_HOUR));
        rtc.write(RTCUPDATE0, RTCUPDATE0_WRITE).unwrap();
        assert_eq!(time(&mut rtc), [0x05, 0x04, 0x15, 1 << 4, 0x02, 0x24, 0x29]);
        rtc.write(RTCCNTLM, RTCCNTL_24_HOUR).unwrap();
        rtc.write(RTCCNTL, 0).unwrap();
        rtc.write(RTCUPDATE0, RTCUPDATE0_WRITE).unwrap();
        assert_eq!(time(&mut rtc)[2], 0x03 | RTCHOUR_PM);
        rtc.write(RTCCNTL, RTCCNTL_24_HOUR).unwrap();
        rtc.write(RTCUPDATE0, RTCUPDATE0_WRITE).unwrap();

        // 2031-12-31 23:59:58: the clock runs on from what it was set to.
        for (index, value) in [0x58, 0x59, 0x23, 1 << 3, 0x12, 0x31, 0x31]
            .into_iter()
            .enumerate()
        {
            rtc.write(RTCSEC + index as u8, value).unwrap();
        }
        rtc.write(RTCUPDATE0, RTCUPDATE0_WRITE).unwrap();
        assert_eq!(time(&mut rtc), [0x58, 0x59, 0x23, 1 << 3, 0x12, 0x31, 0x31]);
    }

    #[test]
    fn an_alarm_for_a_date_goes_off_once_when_its_time_comes() {
        let clock = guest_clock(Duration::ZERO);
        let mut rtc = Rtc::new(Arc::clone(&clock), 0);
        // The clock at 2031-12-31 23:59:58, the alarm at 2032-01-01
        // 00:00:01: every field but the day of the week.
        for (index, value) in [58, 59, 23, 1 << 3, 12, 31, 31].into_iter().enumerate() {
            rtc.write(RTCSEC + index as u8, value).unwrap();
        }
        rtc.write(RTCUPDATE0, RTCUPDATE0_WRITE).unwrap();
        rtc.write(RTCINTM, !RTCINT_ALARM[0]).unwrap();
        for (index, value) in [1, 0, 0, 0, 1, 32, 1].into_iter().enumerate() {
            let enable = if index == 3 { 0 } else { ALARM_ENABLE };
            rtc.write(ALARM1_SEC + index as u8, value | enable).unwrap();
        }
        rtc.write(RTCUPDATE0, RTCUPDATE0_WRITE).unwrap();
        clock.set(Duration::from_secs(2));
        assert!(!rtc.alarm_due());
        clock.set(Duration::from_millis(2_999));
        assert!(!rtc.alarm_due());
        clock.set(Duration::from_secs(3));
        assert!(rtc.alarm_due());
        assert_eq!(rtc.read(RTCINT), Some(RTCINT_ALARM[0]));
        assert_eq!(rtc.read(RTCINT), Some(0));
        assert!(!rtc.alarm_due(), "an alarm goes off once");

        // Alarms that repeat (only some fields enabled) are refused.
        rtc.write(ALARM1_SEC, 1).unwrap();
        assert_eq!(rtc.write(RTCUPDATE0, RTCUPDATE0_WRITE), None);
    }

    #[test]
    fn power_key_drives_the_interrupt_output_through_both_masks() {
        let level = Arc::new(AtomicBool::new(false));
        let output = Arc::clone(&level);
        let chip = Max77620::new(Arc::default(), move |high| {
            output.store(high, Ordering::SeqCst)
        });
        let mut power = chip.power_block();
        let level = || level.load(Ordering::SeqCst);
        // The press that switched the console on.
        assert_eq!(power.read(ONOFFIRQ), Some(ONOFFIRQ_POWER_KEY_PRESSED));

        chip.power_key(true);
        assert_eq!(power.read(IRQTOP), Some(0), "the event is masked");
        power.write(ONOFFIRQM, 0).unwrap();
        assert_eq!(power.read(IRQTOP), Some(IRQTOP_ONOFF));
        assert!(!level(), "the block is masked");
        power.write(INTENLBT, INTENLBT_OUTPUT).unwrap();
        power.write(IRQTOPM, !IRQTOP_ONOFF).unwrap();
        assert!(!level(), "the output is masked");
        power.write(INTENLBT, 0).unwrap();
        assert!(level());
        assert_eq!(power.read(ONOFFSTAT), Some(ONOFFSTAT_POWER_KEY));
        assert_eq!(power.read(ONOFFIRQ), Some(ONOFFIRQ_POWER_KEY_PRESSED));
        assert!(!level(), "reading the event takes it back");

        chip.power_key(false);
        assert!(chip.interrupt());
        assert_eq!(power.read(ONOFFSTAT), Some(0));
        assert_eq!(power.read(ONOFFIRQ), Some(ONOFFIRQ_POWER_KEY_RELEASED));
    }

    /// The clock's time registers, captured by a read transfer.
    fn read_time(rtc: &mut impl RegisterMap) -> [u8; 7] {
        rtc.write(RTCUPDATE0, RTCUPDATE0_READ).unwrap();
        [0, 1, 2, 3, 4, 5, 6].map(|index| rtc.read(RTCSEC + index).unwrap())
    }

    /// The alarm watcher, not an access, raises the output: waits for it.
    fn wait_for(level: &AtomicBool) {
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while !level.load(Ordering::SeqCst) {
            assert!(std::time::Instant::now() < deadline, "the alarm went off");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn an_alarm_set_before_a_snapshot_goes_off_at_the_same_guest_time_after_restore() {
        let chip = |base, clock: &Arc<GuestClock>| {
            let level = Arc::new(AtomicBool::new(false));
            let output = Arc::clone(&level);
            let chip = Max77620::starting_at(base, Arc::clone(clock), move |high| {
                output.store(high, Ordering::SeqCst)
            });
            (chip, level)
        };
        let saved_clock = guest_clock(Duration::from_secs(10));
        let (saved, saved_level) = chip(1_000, &saved_clock);
        let mut rtc = saved.clock_block();
        saved.power_block().write(IRQTOPM, !IRQTOP_RTC).unwrap();
        // At guest time 10 s the clock is set to 2031-12-31 23:59:58, the
        // alarm to 2032-01-01 00:00:01: it goes off at guest time 13 s.
        for (index, value) in [58, 59, 23, 1 << 3, 12, 31, 31].into_iter().enumerate() {
            rtc.write(RTCSEC + index as u8, value).unwrap();
        }
        rtc.write(RTCUPDATE0, RTCUPDATE0_WRITE).unwrap();
        rtc.write(RTCINTM, !RTCINT_ALARM[0]).unwrap();
        for (index, value) in [1, 0, 0, 0, 1, 32, 1].into_iter().enumerate() {
            let enable = if index == 3 { 0 } else { ALARM_ENABLE };
            rtc.write(ALARM1_SEC + index as u8, value | enable).unwrap();
        }
        rtc.write(RTCUPDATE0, RTCUPDATE0_WRITE).unwrap();
        saved_clock.set(Duration::from_millis(12_500));
        let time_at_save = read_time(&mut rtc);
        let state = saved.save().unwrap();

        // A machine built later, from another host time; its guest time is
        // put where the saved one's was before its parts are restored.
        let restored_clock = guest_clock(Duration::ZERO);
        let (restored, restored_level) = chip(2_000_000_000, &restored_clock);
        restored_clock.set(Duration::from_millis(12_500));
        restored.restore(&state).unwrap();
        let mut restored_rtc = restored.clock_block();
        assert_eq!(read_time(&mut restored_rtc), time_at_save);

        // Guest time standing still short of the alarm, it does not go off.
        for clock in [&saved_clock, &restored_clock] {
            clock.set(Duration::from_millis(12_999));
        }
        std::thread::sleep(WATCH_AT_MOST + Duration::from_millis(50));
        assert!(!saved_level.load(Ordering::SeqCst));
        assert!(!restored_level.load(Ordering::SeqCst));
        for clock in [&saved_clock, &restored_clock] {
            clock.set(Duration::from_secs(13));
        }
        wait_for(&saved_level);
        wait_for(&restored_level);
        assert_eq!(restored_rtc.read(RTCINT), Some(RTCINT_ALARM[0]));
        assert_eq!(read_time(&mut restored_rtc), read_time(&mut rtc));
    }

    #[test]
    fn the_software_reset_restarts_or_powers_off_as_onoffcnfg2_says() {
        let chip = Max77620::new(Arc::default(), |_| {});
        let mut power = chip.power_block();
        // What the NX bootloader leaves, and pcv's Restart: 0x42 = 0x81,
        // then 0x41 = 0xB8.
        power.write(ONOFFCNFG1, 0x78).unwrap();
        assert_eq!(chip.power_down(), None);
        power.write(ONOFFCNFG2, 0x81).unwrap();
        power.write(ONOFFCNFG1, 0xB8).unwrap();
        assert_eq!(chip.power_down(), Some(PowerDown::Restart));
        assert_eq!(power.read(ONOFFCNFG1), Some(0x38));
        assert!(matches!(chip.quiesce(), Err(SnapshotError::Unsupported(_))));

        // Turn Off: 0x42 = 0x01, then 0x41 = 0xB8.
        let chip = Max77620::new(Arc::default(), |_| {});
        let mut power = chip.power_block();
        power.write(ONOFFCNFG2, 0x01).unwrap();
        power.write(ONOFFCNFG1, 0xB8).unwrap();
        assert_eq!(chip.power_down(), Some(PowerDown::Off));

        // PWR_OFF is not modeled.
        let chip = Max77620::new(Arc::default(), |_| {});
        assert_eq!(chip.power_block().write(ONOFFCNFG1, 0x3A), None);
        assert_eq!(chip.power_down(), None);
    }

    #[test]
    fn a_restarted_chip_keeps_the_clock_and_has_no_key_event() {
        let old_clock = guest_clock(Duration::from_secs(100));
        let old = Max77620::starting_at(1_000, Arc::clone(&old_clock), |_| {});
        let mut rtc = old.clock_block();
        // Set to 2031-12-31 23:59:58, then 100 s of guest time.
        for (index, value) in [58, 59, 23, 1 << 3, 12, 31, 31].into_iter().enumerate() {
            rtc.write(RTCSEC + index as u8, value).unwrap();
        }
        rtc.write(RTCUPDATE0, RTCUPDATE0_WRITE).unwrap();
        old_clock.set(Duration::from_secs(200));
        let shown = read_time(&mut rtc);
        let kept = old.battery_backed();

        // Built later, from another host time, at guest time zero.
        let new_clock = guest_clock(Duration::ZERO);
        let new = Max77620::starting_at(2_000_000_000, Arc::clone(&new_clock), |_| {});
        assert_eq!(
            new.power_block().read(ONOFFIRQ),
            Some(ONOFFIRQ_POWER_KEY_PRESSED)
        );
        new.power_up_from_reset(kept);
        assert_eq!(new.power_block().read(ONOFFIRQ), Some(0));
        let mut new_rtc = new.clock_block();
        assert_eq!(read_time(&mut new_rtc), shown);
        new_clock.set(Duration::from_secs(3));
        // 2032-01-01 00:01:41.
        assert_eq!(read_time(&mut new_rtc), [41, 1, 0, 1 << 4, 1, 32, 1]);
    }
}
