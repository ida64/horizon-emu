//! The microSD card: an SD memory card of the SDHC or SDXC capacity class
//! with UHS-I, as it answers on its command and data lines. The protocol is
//! that of the SD Association's Physical Layer Simplified Specification,
//! version 9.00 (sections cited as §); what the card has to answer is what
//! the filesystem process's driver asks of it (`FsSdmmc_StartupSd`, FS text
//! `0x1b67c0`, and the functions it calls):
//!
//! 1. `GO_IDLE_STATE`; `SEND_IF_COND` for 2.7-3.6 V with check pattern
//!    0xAA, which the driver wants echoed.
//! 2. `SD_SEND_OP_COND` (ACMD41) with HCS, XPC, 3.2-3.3 V and S18R, every
//!    millisecond for up to 3 s until the card reports itself ready
//!    (`FsSdmmc_NegotiateSdOcr`, `0x1b5d70`). With S18A in the answer:
//!    `VOLTAGE_SWITCH`, then the driver stops the clock, checks `DAT[3:0]`
//!    are low, enables 1.8 V signaling, waits 5 ms, starts the clock and
//!    checks `DAT[3:0]` are high (`FsSdmmc_SwitchSdVoltage`, `0x1ba1e0`).
//! 3. `ALL_SEND_CID`; `SEND_RELATIVE_ADDR` until the card publishes a
//!    nonzero address; `SEND_CSD`, from which it takes the capacity;
//!    `SELECT_CARD`; `SET_BLOCKLEN`; ACMD42 (DAT3 pull-up off); ACMD51 for
//!    the SCR, whose bus widths decide ACMD6 to 4 bits.
//! 4. With 1.8 V signaling: `SWITCH_FUNC` in check mode, then check and
//!    switch of access mode SDR104 (SDR50 if SDR104 is missing), accepted
//!    only if the card selects it and reports at most 800 mA; the host
//!    moves to SDR104 and tunes with `SEND_TUNING_BLOCK` (CMD19); a final
//!    `SEND_STATUS` (`FsSdmmc_SelectSdUhsSpeed`, `0x1b6520`). Without it:
//!    high speed (function 1) the same way, and ACMD13 for the SD status.
//! 5. Data with `READ_MULTIPLE_BLOCK` and `WRITE_MULTIPLE_BLOCK`, block
//!    addressed, ended by an automatic `STOP_TRANSMISSION`
//!    (`FsSdmmc_TransferBlocksAndValidateStatus`, `0x1b1e00`).
//!
//! The card's supply is switched by the board (GPIO E4, which the clock
//! and power driver drives when the filesystem process enables the
//! "SdCard" regulator, `FsSdmmc_PowerOnSd`, `0x1ba660`). Without supply
//! the card answers nothing; when it comes back the card is idle and
//! signals at 3.3 V again, which is the only way back from 1.8 V (§4.2.4).
//!
//! Commands execute at once, so the card is never busy programming. A
//! command its state does not accept goes unanswered and the next status
//! reports `ILLEGAL_COMMAND` (§4.6.1). Erase, write protection, lock and
//! unlock, the security system, speed class control, `GEN_CMD` and the
//! extension register commands are not modeled; they are illegal here and
//! leave a note.
//!
//! Identity, a generic card of the image's size (all choices made here,
//! not a particular product's):
//! * OCR: 2.7-3.6 V.
//! * CID ([`CID`]): no manufacturer (ID 0, unassigned), OEM "HE", product
//!   "HZNSD", revision 1.0, serial number 1, made January 2026.
//! * Relative address published by `SEND_RELATIVE_ADDR`: [`RCA`].
//! * CSD version 2.0 (§5.3.3): 1 ms read access time, the command classes
//!   of a memory card with the switch function (0x5B5), 512-byte blocks,
//!   capacity from the image. SDHC up to 32 GiB, SDXC above; nothing else
//!   in the registers differs between the two here.
//! * SCR ([`SCR`]): physical layer version 3.0X, 1- and 4-bit bus,
//!   `SET_BLOCK_COUNT` (mandatory for UHS104 cards), no security system.
//! * SD status: speed class 10, UHS speed grade 1,
//!   4 MiB allocation units, no protected area (the protected area belongs
//!   to the security system), erase timing not given.
//! * Switch functions (§4.3.10): access modes SDR12 and SDR25 at 3.3 V
//!   signaling, also SDR50, SDR104 and DDR50 at 1.8 V; only the default
//!   function in the other groups; 100 mA at the 3.3 V speeds, 200 mA at
//!   the UHS-I ones.

use std::{fmt, io, mem, sync::Arc};

use serde::{Deserialize, Serialize};

use super::{
    Notes,
    medium::{ImageFile, Medium, MediumError, MediumState, SECTOR_SIZE},
    sdmmc::{BlockError, Card, Response},
};
use crate::snapshot::{SnapshotError, decode, encode};

/// The device name in notes and messages about its image.
const DEVICE: &str = "microSD card";

/// Unit of `C_SIZE` in a version 2.0 CSD: capacity is `(C_SIZE + 1)` of
/// these (§5.3.3).
const CAPACITY_UNIT: u64 = 512 * 1024;
/// The smallest `C_SIZE` of a high capacity card, just above 2 GB, and the
/// largest of an extended capacity card, just below 2 TB (§5.3.3).
const C_SIZE_RANGE: std::ops::RangeInclusive<u64> = 0x1010..=0x3F_FEFF;

/// 2.7-3.6 V, the whole window an SD memory card supports.
const OCR_VOLTAGE_WINDOW: u32 = 0x00FF_8000;
/// Power-up is complete. Clear while the card is busy initialising.
const OCR_POWERED_UP: u32 = 1 << 31;
/// Card capacity status: block addressing, a high or extended capacity
/// card.
const OCR_CCS: u32 = 1 << 30;
/// Switching to 1.8 V signaling accepted.
const OCR_S18A: u32 = 1 << 24;
/// ACMD41 argument: the host supports high capacity cards, and asks for
/// 1.8 V signaling.
const ACMD41_HCS: u32 = 1 << 30;
const ACMD41_S18R: u32 = 1 << 24;
/// `SEND_IF_COND` supply voltage field value for 2.7-3.6 V.
const VHS_2V7_TO_3V6: u32 = 1;

/// The address the card publishes: a choice; any nonzero value is valid.
pub const RCA: u16 = 0x0001;

/// Card identification, bits 127..8 (§5.2): manufacturer, OEM, product
/// name, revision, serial number, then the manufacturing date with the
/// year counted from 2000.
pub const CID: [u8; 15] = [
    0x00, b'H', b'E', b'H', b'Z', b'N', b'S', b'D', 0x10, 0, 0, 0, 1, 0x01, 0xA1,
];

/// SD configuration register (§5.6): structure 0, `SD_SPEC` 2 with
/// `SD_SPEC3` (version 3.0X), no security, 1- and 4-bit bus widths, and in
/// `CMD_SUPPORT` `SET_BLOCK_COUNT` only.
pub const SCR: [u8; 8] = [0x02, 0x05, 0x80, 0x02, 0, 0, 0, 0];

/// Command classes 0, 2, 4, 5, 7, 8 and 10.
const COMMAND_CLASSES: u128 = 0x5B5;

/// Device status bits of an R1 response (§4.10.1).
mod status {
    pub const OUT_OF_RANGE: u32 = 1 << 31;
    pub const BLOCK_LEN_ERROR: u32 = 1 << 29;
    pub const ILLEGAL_COMMAND: u32 = 1 << 22;
    pub const READY_FOR_DATA: u32 = 1 << 8;
    pub const APP_CMD: u32 = 1 << 5;
    pub const CURRENT_STATE_SHIFT: u32 = 9;
}

/// Function group 1 (access mode): SDR12, SDR25, SDR50, SDR104, DDR50.
mod access_mode {
    pub const SDR50: u8 = 2;
    pub const SDR104: u8 = 3;
    /// Functions supported at 3.3 V signaling: SDR12 (default speed) and
    /// SDR25 (high speed).
    pub const AT_3V3: u16 = 0b11;
    /// At 1.8 V signaling, all five.
    pub const AT_1V8: u16 = 0b1_1111;
}

/// Bytes of the status `SWITCH_FUNC` returns, and of the SD status.
const STATUS_BLOCK: usize = 64;
const FUNCTION_GROUPS: usize = 6;
/// "No change" in a `SWITCH_FUNC` argument, "cannot switch" in its result.
const NO_FUNCTION: u8 = 0xF;

#[derive(Debug)]
pub enum SdCardError {
    Io(io::Error),
    /// An image of this many bytes is no card's capacity.
    Capacity(u64),
}

impl fmt::Display for SdCardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "{error}"),
            Self::Capacity(size) => write!(
                f,
                "a card image of {size} bytes: it must be a multiple of 512 KiB, from {} to {} \
                 bytes (an SDHC or SDXC card)",
                (C_SIZE_RANGE.start() + 1) * CAPACITY_UNIT,
                (C_SIZE_RANGE.end() + 1) * CAPACITY_UNIT,
            ),
        }
    }
}

/// The sectors of a card whose capacity is `bytes`.
pub fn sectors(bytes: u64) -> Result<u64, SdCardError> {
    let units = bytes / CAPACITY_UNIT;
    if !bytes.is_multiple_of(CAPACITY_UNIT) || !C_SIZE_RANGE.contains(&units.wrapping_sub(1)) {
        return Err(SdCardError::Capacity(bytes));
    }
    Ok(bytes / SECTOR_SIZE as u64)
}

/// What the card stores, of a size a card can have.
pub struct SdCardImage(Medium);

impl SdCardImage {
    /// The file's length is the capacity. Writes go to it only if
    /// `writable` (it must be open for writing then); otherwise they last
    /// for the run.
    pub fn file(image: ImageFile, writable: bool) -> Result<Self, SdCardError> {
        let sectors = sectors(image.file.metadata().map_err(SdCardError::Io)?.len())?;
        Ok(Self(Medium::new(DEVICE, sectors, Some(image), writable)))
    }

    /// Contents held in memory, written to in place.
    pub fn memory(contents: Vec<u8>) -> Result<Self, SdCardError> {
        sectors(contents.len() as u64)?;
        Ok(Self(Medium::in_memory(DEVICE, contents)))
    }
}

/// Card states (§4.3) with their `CURRENT_STATE` encoding. The
/// programming and disconnect states are never entered: writes finish at
/// once.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum State {
    Idle = 0,
    Ready = 1,
    Identification = 2,
    Standby = 3,
    Transfer = 4,
    SendingData = 5,
    ReceivingData = 6,
    /// Rejected the host's voltage window; answers nothing until power
    /// is cycled.
    Inactive = 15,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum Signaling {
    V3_3,
    /// Between `VOLTAGE_SWITCH` and a clock that comes back at 1.8 V:
    /// the card holds CMD and DAT low.
    Switching,
    V1_8,
}

/// What moves over the data lines after a command.
#[derive(Clone, Serialize, Deserialize)]
enum Data {
    /// A register or status block the card sends.
    Block(Vec<u8>),
    /// The tuning pattern; only the host's sampling logic looks at it.
    Tuning,
    /// Sectors from or to the card. `left` is `None` for an open-ended
    /// transfer, which lasts until `STOP_TRANSMISSION`.
    Sectors { next: u64, left: Option<u32> },
}

pub struct SdCard {
    medium: Medium,
    powered: bool,
    state: State,
    rca: u16,
    /// `SEND_IF_COND` was answered since the card went idle: the host
    /// knows version 2.00 and later.
    interface_checked: bool,
    signaling: Signaling,
    /// `SD_SEND_OP_COND` granted the switch to 1.8 V signaling.
    switch_granted: bool,
    /// `APP_CMD` came last: the next command is application-specific.
    application: bool,
    wide_bus: bool,
    /// Selected function per group, group 1 first.
    functions: [u8; FUNCTION_GROUPS],
    /// Error bits waiting for the next R1 response to report them.
    errors: u32,
    /// Block count announced by `SET_BLOCK_COUNT` for the next transfer.
    block_count: Option<u32>,
    data: Option<Data>,
    notes: Arc<Notes>,
}

/// The card's state in a machine snapshot: everything but its notes, with
/// what the run changed of its contents.
#[derive(Serialize, Deserialize)]
struct Saved<'a> {
    medium: MediumState<'a>,
    powered: bool,
    state: State,
    rca: u16,
    interface_checked: bool,
    signaling: Signaling,
    switch_granted: bool,
    application: bool,
    wide_bus: bool,
    functions: [u8; FUNCTION_GROUPS],
    errors: u32,
    block_count: Option<u32>,
    data: Option<Data>,
}

impl SdCard {
    /// A card with `image`'s contents, its supply off.
    pub fn new(image: SdCardImage, notes: Arc<Notes>) -> Self {
        Self::with_medium(image.0, notes)
    }

    fn with_medium(medium: Medium, notes: Arc<Notes>) -> Self {
        Self {
            medium,
            powered: false,
            state: State::Idle,
            rca: 0,
            interface_checked: false,
            signaling: Signaling::V3_3,
            switch_granted: false,
            application: false,
            wide_bus: false,
            functions: [0; FUNCTION_GROUPS],
            errors: 0,
            block_count: None,
            data: None,
            notes,
        }
    }

    /// What `GO_IDLE_STATE` and power-up have in common: everything but the
    /// signaling level starts over.
    fn go_idle(&mut self) {
        self.state = State::Idle;
        self.rca = 0;
        self.interface_checked = false;
        self.switch_granted = false;
        self.application = false;
        self.wide_bus = false;
        self.functions = [0; FUNCTION_GROUPS];
        self.errors = 0;
        self.block_count = None;
        self.data = None;
    }

    fn status(&self) -> u32 {
        let application = if self.application { status::APP_CMD } else { 0 };
        (self.state as u32) << status::CURRENT_STATE_SHIFT | status::READY_FOR_DATA | application
    }

    /// The R1 response to a command that found the card in its current
    /// state, reporting and clearing the collected errors.
    fn r1(&mut self) -> Option<Response> {
        Some(Response::Short(self.status() | mem::take(&mut self.errors)))
    }

    /// What the card's answer to a data transfer is to the controller. A
    /// sector out of range is reported in the next status as well.
    fn sectors_moved(&mut self, moved: Result<(), MediumError>) -> Result<(), BlockError> {
        match moved {
            Ok(()) => Ok(()),
            Err(MediumError::OutOfRange) => {
                self.errors |= status::OUT_OF_RANGE;
                Err(BlockError::Timeout)
            }
            Err(MediumError::Host(note)) => Err(BlockError::Host(note)),
        }
    }

    /// R1 to an application-specific command, which says it was taken as
    /// one.
    fn r1_application(&mut self) -> Option<Response> {
        let response = self.status() | status::APP_CMD | mem::take(&mut self.errors);
        Some(Response::Short(response))
    }

    /// R6: the published address and status bits 23, 22, 19 and 12..0.
    fn r6(&mut self) -> Option<Response> {
        let status = self.status() | mem::take(&mut self.errors);
        let compressed = status >> 8 & 0xC000 | status >> 6 & 0x2000 | status & 0x1FFF;
        Some(Response::Short(u32::from(self.rca) << 16 | compressed))
    }

    fn illegal(&mut self, command: &str, argument: u32) -> Option<Response> {
        self.errors |= status::ILLEGAL_COMMAND;
        self.notes.add(format!(
            "{DEVICE} ignored {command} (argument {argument:#x}) in its {:?} state",
            self.state
        ));
        None
    }

    /// Bits 127..8 of the version 2.0 CSD (§5.3.3).
    fn csd(&self) -> [u8; 15] {
        let c_size = u128::from(self.medium.sectors() * SECTOR_SIZE as u64 / CAPACITY_UNIT - 1);
        // Maximum transfer rate of the access mode in use (§5.3.3).
        let tran_speed: u128 = match self.functions[0] {
            0 => 0x32,
            access_mode::SDR50 => 0x0B,
            access_mode::SDR104 => 0x2B,
            _ => 0x5A,
        };
        let fields: u128 = 1 << 126 // CSD_STRUCTURE: version 2.0
            | 0x0E << 112 // TAAC: 1 ms
            | tran_speed << 96
            | COMMAND_CLASSES << 84
            | 9 << 80 // READ_BL_LEN: 512 bytes
            | c_size << 48
            | 1 << 46 // ERASE_BLK_EN
            | 0x7F << 39 // SECTOR_SIZE: 64 KiB
            | 0b010 << 26 // R2W_FACTOR: writes take 4 times as long
            | 9 << 22; // WRITE_BL_LEN: 512 bytes
        fields.to_be_bytes()[..15].try_into().unwrap()
    }

    /// The SD status (§4.10.2).
    fn sd_status(&self) -> Vec<u8> {
        let mut status = vec![0; STATUS_BLOCK];
        // DAT_BUS_WIDTH.
        status[0] = if self.wide_bus { 0b10 << 6 } else { 0 };
        // SPEED_CLASS: class 10.
        status[8] = 4;
        // AU_SIZE: 4 MiB.
        status[10] = 9 << 4;
        // UHS_SPEED_GRADE 1, UHS_AU_SIZE 4 MiB.
        status[14] = 1 << 4 | 9;
        status
    }

    /// `SWITCH_FUNC` (§4.3.10): the switch function status, and in switch
    /// mode the switch, unless a group cannot take what is asked.
    fn switch_function(&mut self, argument: u32) -> Vec<u8> {
        let access_modes = match self.signaling {
            Signaling::V1_8 => access_mode::AT_1V8,
            _ => access_mode::AT_3V3,
        };
        let mut supported = [1; FUNCTION_GROUPS];
        supported[0] = access_modes;
        let mut results = [0; FUNCTION_GROUPS];
        for (group, result) in results.iter_mut().enumerate() {
            let requested = (argument >> (4 * group) & 0xF) as u8;
            *result = match requested {
                NO_FUNCTION => self.functions[group],
                _ if supported[group] & 1 << requested != 0 => requested,
                _ => NO_FUNCTION,
            };
        }
        let valid = !results.contains(&NO_FUNCTION);
        if valid && argument & 1 << 31 != 0 {
            self.functions = results;
        }
        let mut block = vec![0; STATUS_BLOCK];
        let current: u16 = match (valid, results[0]) {
            (false, _) => 0,
            (true, 0 | 1) => 100,
            (true, _) => 200,
        };
        block[..2].copy_from_slice(&current.to_be_bytes());
        // Support bits, group 6 first; bit 15 stands for 0xF.
        for (group, bits) in supported.iter().enumerate() {
            let at = 2 + 2 * (FUNCTION_GROUPS - 1 - group);
            block[at..at + 2].copy_from_slice(&(bits | 1 << 15).to_be_bytes());
        }
        // Function selection, a nibble per group, group 6 first.
        for (group, result) in results.iter().enumerate() {
            let at = 16 - group / 2;
            block[at] |= result << (4 * (group % 2));
        }
        // Data structure version 1: the busy status follows, none busy.
        block[17] = 1;
        block
    }

    fn start_transfer(
        &mut self,
        sector: u32,
        blocks: Option<u32>,
        state: State,
    ) -> Option<Response> {
        let sector = u64::from(sector);
        if sector >= self.medium.sectors() {
            self.errors |= status::OUT_OF_RANGE;
            return self.r1();
        }
        let response = self.r1();
        self.data = Some(Data::Sectors {
            next: sector,
            left: blocks,
        });
        self.state = state;
        response
    }

    /// `SD_SEND_OP_COND`: R3, the OCR.
    fn send_op_cond(&mut self, argument: u32) -> Option<Response> {
        let window = argument & 0x00FF_FFFF;
        if window == 0 {
            // An inquiry: the OCR, initialisation not started.
            return Some(Response::Short(OCR_VOLTAGE_WINDOW));
        }
        if window & OCR_VOLTAGE_WINDOW == 0 {
            self.state = State::Inactive;
            return None;
        }
        // A high capacity card stays busy for a host that has not shown it
        // supports one (§4.2.3).
        if !self.interface_checked || argument & ACMD41_HCS == 0 {
            return Some(Response::Short(OCR_VOLTAGE_WINDOW));
        }
        self.state = State::Ready;
        // Already at 1.8 V, the card has nothing left to switch.
        self.switch_granted = argument & ACMD41_S18R != 0 && self.signaling == Signaling::V3_3;
        let s18a = if self.switch_granted { OCR_S18A } else { 0 };
        Some(Response::Short(
            OCR_POWERED_UP | OCR_CCS | s18a | OCR_VOLTAGE_WINDOW,
        ))
    }

    fn application_command(&mut self, index: u8, argument: u32) -> Option<Response> {
        use State::*;
        match (index, self.state) {
            // SET_BUS_WIDTH.
            (6, Transfer) if matches!(argument & 3, 0 | 2) => {
                self.wide_bus = argument & 3 == 2;
                self.r1_application()
            }
            // SD_STATUS.
            (13, Transfer) => {
                let response = self.r1_application();
                self.data = Some(Data::Block(self.sd_status()));
                self.state = SendingData;
                response
            }
            // SET_WR_BLK_ERASE_COUNT: a hint for the next write, which has
            // nothing to erase ahead here.
            (23, Transfer) => self.r1_application(),
            (41, Idle) => self.send_op_cond(argument),
            // SET_CLR_CARD_DETECT: the DAT3 pull-up is not modeled.
            (42, Transfer) => self.r1_application(),
            // SEND_SCR.
            (51, Transfer) => {
                let response = self.r1_application();
                self.data = Some(Data::Block(SCR.to_vec()));
                self.state = SendingData;
                response
            }
            _ => self.illegal(&format!("ACMD{index}"), argument),
        }
    }

    fn end_of_block(&mut self, left: Option<u32>) {
        if left == Some(0) {
            self.data = None;
            self.state = State::Transfer;
        }
    }
}

impl Card for SdCard {
    fn command(&mut self, index: u8, argument: u32) -> Option<Response> {
        use State::*;
        if !self.powered || self.state == Inactive || self.signaling == Signaling::Switching {
            return None;
        }
        let block_count = self.block_count.take();
        if mem::take(&mut self.application) {
            return self.application_command(index, argument);
        }
        let addressed = (argument >> 16) as u16 == self.rca;
        match (index, self.state) {
            // GO_IDLE_STATE.
            (0, _) => {
                self.go_idle();
                None
            }
            // ALL_SEND_CID.
            (2, Ready) => {
                self.state = Identification;
                Some(Response::Long(CID))
            }
            // SEND_RELATIVE_ADDR.
            (3, Identification | Standby) => {
                self.rca = RCA;
                let response = self.r6();
                self.state = Standby;
                response
            }
            // SWITCH_FUNC.
            (6, Transfer) => {
                let response = self.r1();
                let status = self.switch_function(argument);
                self.data = Some(Data::Block(status));
                self.state = SendingData;
                response
            }
            // SELECT/DESELECT_CARD.
            (7, Standby) if addressed => {
                let response = self.r1();
                self.state = Transfer;
                response
            }
            (7, Transfer | SendingData | ReceivingData) if !addressed => {
                self.data = None;
                self.state = Standby;
                None
            }
            // SEND_IF_COND: R7 echoes the voltage and the check pattern. A
            // card that cannot take the voltage stays silent.
            (8, Idle) if argument >> 8 & 0xF == VHS_2V7_TO_3V6 => {
                self.interface_checked = true;
                Some(Response::Short(argument & 0xFFF))
            }
            (8, Idle) => None,
            // SEND_CSD, SEND_CID.
            (9, Standby) if addressed => Some(Response::Long(self.csd())),
            (10, Standby) if addressed => Some(Response::Long(CID)),
            // VOLTAGE_SWITCH.
            (11, Ready) if self.switch_granted => {
                let response = self.r1();
                self.switch_granted = false;
                self.signaling = Signaling::Switching;
                response
            }
            // STOP_TRANSMISSION.
            (12, SendingData | ReceivingData) => {
                let response = self.r1();
                self.data = None;
                self.state = Transfer;
                response
            }
            // SEND_STATUS.
            (13, Standby | Transfer | SendingData | ReceivingData) if addressed => self.r1(),
            // SET_BLOCKLEN: memory commands use 512 bytes whatever is set;
            // the length is for lock and unlock (§4.3.2).
            (16, Transfer) => {
                if argument == 0 || argument > SECTOR_SIZE as u32 {
                    self.errors |= status::BLOCK_LEN_ERROR;
                }
                self.r1()
            }
            // READ_SINGLE_BLOCK, READ_MULTIPLE_BLOCK.
            (17, Transfer) => self.start_transfer(argument, Some(1), SendingData),
            (18, Transfer) => self.start_transfer(argument, block_count, SendingData),
            // SEND_TUNING_BLOCK, in the modes that need tuning (§4.2.4).
            (19, Transfer)
                if matches!(self.functions[0], access_mode::SDR50 | access_mode::SDR104) =>
            {
                let response = self.r1();
                self.data = Some(Data::Tuning);
                self.state = SendingData;
                response
            }
            // SET_BLOCK_COUNT.
            (23, Transfer) => {
                self.block_count = Some(argument);
                self.r1()
            }
            // WRITE_BLOCK, WRITE_MULTIPLE_BLOCK.
            (24, Transfer) => self.start_transfer(argument, Some(1), ReceivingData),
            (25, Transfer) => self.start_transfer(argument, block_count, ReceivingData),
            // APP_CMD; in the idle state the address is still 0.
            (55, Idle | Standby | Transfer | SendingData | ReceivingData) if addressed => {
                self.application = true;
                self.r1()
            }
            _ => self.illegal(&format!("CMD{index}"), argument),
        }
    }

    fn send_block(&mut self, block: &mut [u8]) -> Result<(), BlockError> {
        if !(self.powered && self.state == State::SendingData) {
            return Err(BlockError::Timeout);
        }
        match self.data.as_mut().ok_or(BlockError::Timeout)? {
            Data::Block(bytes) => {
                if block.len() != bytes.len() {
                    return Err(BlockError::Timeout);
                }
                block.copy_from_slice(bytes);
                self.end_of_block(Some(0));
            }
            Data::Tuning => return Err(BlockError::Timeout),
            Data::Sectors { next, left } => {
                let sector = *next;
                *next += 1;
                let left = left.as_mut().map(|left| {
                    *left -= 1;
                    *left
                });
                let read = self.medium.read(sector, block);
                self.sectors_moved(read)?;
                self.end_of_block(left);
            }
        }
        Ok(())
    }

    fn receive_block(&mut self, block: &[u8]) -> Result<(), BlockError> {
        if !(self.powered && self.state == State::ReceivingData) {
            return Err(BlockError::Timeout);
        }
        let Some(Data::Sectors { next, left }) = self.data.as_mut() else {
            return Err(BlockError::Timeout);
        };
        let sector = *next;
        *next += 1;
        let left = left.as_mut().map(|left| {
            *left -= 1;
            *left
        });
        let written = self.medium.write(sector, block);
        self.sectors_moved(written)?;
        self.end_of_block(left);
        Ok(())
    }

    fn take_tuning_block(&mut self) -> bool {
        let tuning = self.state == State::SendingData && matches!(self.data, Some(Data::Tuning));
        if tuning {
            self.end_of_block(Some(0));
        }
        tuning
    }

    fn holds_lines_low(&self) -> bool {
        self.powered && self.signaling == Signaling::Switching
    }

    /// The switch completes when the clock comes back at 1.8 V. At 3.3 V
    /// it has failed, and the card keeps the lines low until its supply is
    /// cycled.
    fn clock_started(&mut self, signaling_1v8: bool) {
        if self.signaling == Signaling::Switching && signaling_1v8 {
            self.signaling = Signaling::V1_8;
        }
    }

    fn power(&mut self, on: bool) {
        if on && !self.powered {
            self.go_idle();
            self.signaling = Signaling::V3_3;
        }
        self.powered = on;
    }

    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        encode(&Saved {
            medium: self.medium.save(),
            powered: self.powered,
            state: self.state,
            rca: self.rca,
            interface_checked: self.interface_checked,
            signaling: self.signaling,
            switch_granted: self.switch_granted,
            application: self.application,
            wide_bus: self.wide_bus,
            functions: self.functions,
            errors: self.errors,
            block_count: self.block_count,
            data: self.data.clone(),
        })
    }

    fn restore(&mut self, state: &[u8]) -> Result<(), SnapshotError> {
        let saved: Saved = decode(DEVICE, state)?;
        self.medium.restore(saved.medium)?;
        self.powered = saved.powered;
        self.state = saved.state;
        self.rca = saved.rca;
        self.interface_checked = saved.interface_checked;
        self.signaling = saved.signaling;
        self.switch_granted = saved.switch_granted;
        self.application = saved.application;
        self.wide_bus = saved.wide_bus;
        self.functions = saved.functions;
        self.errors = saved.errors;
        self.block_count = saved.block_count;
        self.data = saved.data;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SECTORS: u64 = (0x1010 + 1) * 1024;

    fn powered() -> SdCard {
        let medium = Medium::in_memory(DEVICE, vec![0; SECTORS as usize * SECTOR_SIZE]);
        let mut card = SdCard::with_medium(medium, Arc::default());
        card.power(true);
        card
    }

    fn short(response: Option<Response>) -> u32 {
        match response {
            Some(Response::Short(value)) => value,
            other => panic!("expected a 48-bit response, got {other:?}"),
        }
    }

    fn state(response: u32) -> u32 {
        response >> status::CURRENT_STATE_SHIFT & 0xF
    }

    fn application(card: &mut SdCard, index: u8, argument: u32) -> Option<Response> {
        let rca = u32::from(card.rca) << 16;
        assert_ne!(short(card.command(55, rca)) & status::APP_CMD, 0);
        card.command(index, argument)
    }

    /// A data block the card sends, of `length` bytes.
    fn receive(card: &mut SdCard, length: usize) -> Vec<u8> {
        let mut block = vec![0; length];
        card.send_block(&mut block).unwrap();
        block
    }

    /// The filesystem process's start-up, as `FsSdmmc_StartupSd` does it,
    /// up to the selected card on a 4-bit bus. Returns whether the card
    /// granted 1.8 V signaling.
    fn start_up(card: &mut SdCard) -> bool {
        assert_eq!(card.command(0, 0), None);
        assert_eq!(card.command(8, 0x1AA), Some(Response::Short(0x1AA)));
        let ocr = short(application(card, 41, 0x5110_0000));
        assert_eq!(ocr & 0xC0FF_8000, 0xC0FF_8000, "ready, block addressed");
        let switched = ocr & OCR_S18A != 0;
        if switched {
            assert_eq!(state(short(card.command(11, 0))), 1, "was ready");
            // The host stops the clock and finds DAT low, enables 1.8 V
            // signaling and starts the clock again.
            assert!(card.holds_lines_low());
            assert_eq!(card.command(2, 0), None, "busy switching");
            card.clock_started(true);
            assert!(!card.holds_lines_low());
        }
        assert_eq!(card.command(2, 0), Some(Response::Long(CID)));
        let r6 = short(card.command(3, 0));
        assert_eq!(r6 >> 16, u32::from(RCA));
        assert_eq!(state(r6), 2, "was identifying");
        let rca = u32::from(RCA) << 16;
        let Some(Response::Long(csd)) = card.command(9, rca) else {
            panic!("no CSD");
        };
        // FS's reading of C_SIZE (`FsSdmmc_CalculateSdCapacity`): bytes 4
        // to 7 of the response register, which holds the CSD from bit 8.
        let register: Vec<u8> = csd.iter().rev().copied().collect();
        assert!(register[14] > 0x3F, "CSD version 2.0");
        let word = |at: usize| u32::from(u16::from_le_bytes([register[at], register[at + 1]]));
        let sectors = ((4 * word(4)) & 0x3FC00 | word(6) << 18) + 1024;
        assert_eq!(u64::from(sectors), SECTORS);
        assert_eq!(state(short(card.command(7, rca))), 3, "was standing by");
        assert_eq!(state(short(card.command(16, 512))), 4);
        assert_eq!(state(short(application(card, 42, 0))), 4);
        assert_eq!(
            short(application(card, 51, 0)) & status::APP_CMD,
            status::APP_CMD
        );
        assert_eq!(receive(card, 8), SCR);
        assert_eq!(SCR[1] & 4, 4, "a 4-bit bus");
        assert_eq!(state(short(application(card, 6, 2))), 4);
        assert_eq!(state(short(card.command(13, rca))), 4);
        switched
    }

    #[test]
    fn the_drivers_start_up_reaches_sdr104_with_tuning() {
        let mut card = powered();
        assert!(start_up(&mut card));
        // Check mode, all groups unchanged: SDR104 is among the supported
        // access modes (byte 13, bit 3).
        short(card.command(6, 0x00FF_FFFF));
        let status = receive(&mut card, 64);
        assert_eq!(status[13] & 0x1F, 0x1F);
        assert_eq!(status[16] & 0xF, 0, "default speed for now");
        // Check, then switch, as `FsSdmmc_SwitchSdFunctionChecked` does.
        for argument in [0x00FF_FFF3, 0x80FF_FFF3] {
            short(card.command(6, argument));
            let status = receive(&mut card, 64);
            assert_eq!(status[16] & 0xF, 3);
            assert!(u16::from_be_bytes([status[0], status[1]]) <= 800);
        }
        assert_eq!(card.functions[0], access_mode::SDR104);
        short(card.command(19, 0));
        assert!(card.take_tuning_block());
        assert_eq!(state(short(card.command(13, u32::from(RCA) << 16))), 4);

        // The SD status reports the bus width.
        short(application(&mut card, 13, 0));
        assert_eq!(receive(&mut card, 64)[0], 0x80);
    }

    #[test]
    fn at_3v3_only_the_slower_modes_are_there_and_power_restores_3v3() {
        let mut card = powered();
        // A host that does not ask for 1.8 V.
        card.command(0, 0);
        card.command(8, 0x1AA);
        let ocr = short(application(&mut card, 41, 0x5010_0000));
        assert_eq!(ocr & OCR_S18A, 0);
        assert_eq!(card.command(11, 0), None, "no switch was granted");
        assert_ne!(card.errors & status::ILLEGAL_COMMAND, 0);

        // After a switch, idle is not enough to get back to 3.3 V...
        let mut card = powered();
        assert!(start_up(&mut card));
        assert!(!start_up(&mut card), "nothing left to switch");
        // ...a power cycle is.
        card.power(false);
        assert_eq!(card.command(0, 0), None);
        assert_eq!(card.command(8, 0x1AA), None, "no supply");
        card.power(true);
        assert!(start_up(&mut card));

        // At 3.3 V: SDR104 cannot be selected, high speed can.
        let mut card = powered();
        card.command(0, 0);
        card.command(8, 0x1AA);
        short(application(&mut card, 41, 0x4010_0000));
        assert_eq!(card.command(2, 0), Some(Response::Long(CID)));
        card.command(3, 0);
        card.command(7, u32::from(RCA) << 16);
        short(card.command(6, 0x80FF_FFF3));
        let status = receive(&mut card, 64);
        assert_eq!(status[13] & 0x1F, 0x03);
        assert_eq!(status[16] & 0xF, NO_FUNCTION);
        assert_eq!(u16::from_be_bytes([status[0], status[1]]), 0);
        assert_eq!(card.functions[0], 0, "nothing switched");
        assert_eq!(card.command(19, 0), None, "no tuning at default speed");
        short(card.command(6, 0x80FF_FFF1));
        assert_eq!(receive(&mut card, 64)[16] & 0xF, 1);
        assert_eq!(card.functions[0], 1);
    }

    #[test]
    fn a_failed_voltage_switch_keeps_the_lines_low() {
        let mut card = powered();
        card.command(0, 0);
        card.command(8, 0x1AA);
        short(application(&mut card, 41, 0x5110_0000));
        short(card.command(11, 0));
        card.clock_started(false);
        assert!(card.holds_lines_low());
        card.power(false);
        assert!(!card.holds_lines_low());
    }

    #[test]
    fn hosts_without_high_capacity_support_never_see_it_ready() {
        let mut card = powered();
        // No SEND_IF_COND: a version 1.x host.
        let ocr = short(application(&mut card, 41, 0x4010_0000));
        assert_eq!(ocr & OCR_POWERED_UP, 0);
        // No HCS.
        card.command(8, 0x1AA);
        let ocr = short(application(&mut card, 41, 0x0010_0000));
        assert_eq!(ocr & OCR_POWERED_UP, 0);
        assert_eq!(card.state, State::Idle);
        // An unsupported voltage on SEND_IF_COND goes unanswered.
        assert_eq!(card.command(8, 0x2AA), None);
        // A window outside 2.7-3.6 V makes the card inactive.
        assert_eq!(application(&mut card, 41, 0x4000_0080), None);
        assert_eq!(card.command(0, 0), None);
        assert_eq!(card.command(8, 0x1AA), None);
    }

    #[test]
    fn data_moves_by_block_address_and_stops_at_the_end() {
        let mut card = powered();
        start_up(&mut card);
        let rca = u32::from(RCA) << 16;
        let pattern: Vec<u8> = (0..SECTOR_SIZE).map(|index| index as u8 ^ 0x3C).collect();
        let last = (SECTORS - 1) as u32;
        // Two sectors announced by SET_BLOCK_COUNT, the last one first.
        short(card.command(23, 2));
        short(card.command(25, last - 1));
        card.receive_block(&pattern).unwrap();
        card.receive_block(&pattern).unwrap();
        assert_eq!(
            card.receive_block(&pattern),
            Err(BlockError::Timeout),
            "the count is used up"
        );
        assert_eq!(state(short(card.command(13, rca))), 4);

        // An open-ended read across the end of the card.
        short(card.command(18, last - 1));
        assert_eq!(receive(&mut card, SECTOR_SIZE), pattern);
        assert_eq!(receive(&mut card, SECTOR_SIZE), pattern);
        assert_eq!(
            card.send_block(&mut [0; SECTOR_SIZE]),
            Err(BlockError::Timeout)
        );
        let stop = short(card.command(12, 0));
        assert_eq!(state(stop), 5, "was sending");
        assert_ne!(stop & status::OUT_OF_RANGE, 0);

        // Out of range from the start: reported at once, nothing sent.
        let read = short(card.command(17, last + 1));
        assert_ne!(read & status::OUT_OF_RANGE, 0);
        assert_eq!(state(read), 4);
        assert_eq!(
            card.send_block(&mut [0; SECTOR_SIZE]),
            Err(BlockError::Timeout)
        );
        // Unmodeled commands: ERASE is illegal and reported next.
        assert_eq!(card.command(38, 0), None);
        assert_ne!(short(card.command(13, rca)) & status::ILLEGAL_COMMAND, 0);
    }
}
