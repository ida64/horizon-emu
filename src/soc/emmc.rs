//! The eMMC module: a JEDEC eMMC 5.0 device with a user area and two boot
//! partitions, as it answers on its command and data lines.
//!
//! What the module stores comes from image files (`soc::medium`); the
//! emulator has no contents of its own to offer.
//!
//! Erased boot partitions are a state `boot` accepts. Its boot image check
//! reads 0x4000 bytes at 0xFC000 of the first one, where the system
//! updater leaves word that the boot images await verification, and with
//! zeros there it goes on to report that booting has finished. A nonzero
//! first byte sends it to the content manager for the images to verify
//! against.
//!
//! Commands execute at once: the card is never busy, and a command sent in
//! a state that does not accept it goes unanswered, as the standard
//! prescribes for illegal commands. Erase, write protection, the replay
//! protected partition and the boot operation mode are not modeled; their
//! commands are illegal here and leave a note.

use std::{collections::HashMap, fs::File, io, sync::Arc};

use super::{
    Notes,
    medium::{Medium, SECTOR_SIZE},
    sdmmc::{Card, Response},
};

const EXT_CSD_SIZE: usize = 512;

/// Each boot partition: 4 MiB.
const BOOT_PARTITION_SECTORS: u64 = 0x2000;
/// Sector addressing, which the card announces, is for devices above 2 GiB.
const MINIMUM_USER_SECTORS: u64 = 0x40_0000;

/// Operating conditions: powered up, sector addressing, 1.70-1.95 V and
/// 2.7-3.6 V.
const OCR: u32 = 0xC0FF_8080;

/// Card identification, bits 127..8: manufacturer 0x15 (Samsung), a BGA
/// device, product name "BJNB4R" (the 32 GB part), and zero for the OEM ID,
/// revision, serial number and manufacturing date, which nothing reads.
const CID: [u8; 15] = [
    0x15, 0x01, 0x00, b'B', b'J', b'N', b'B', b'4', b'R', 0, 0, 0, 0, 0, 0,
];

/// Card-specific data, bits 127..8: structure version and capacity are "see
/// the extended CSD" (`CSD_STRUCTURE` 3, `C_SIZE` 0xFFF), specification
/// version 4 or later, 512-byte read and write blocks, 26 MHz in the
/// backwards compatible timing.
const CSD: [u8; 15] = [
    0xD0, 0x27, 0x01, 0x32, 0x0F, 0x59, 0x03, 0xFF, 0xF6, 0xDB, 0xFF, 0xEF, 0x8E, 0x40, 0x40,
];

/// Extended CSD fields, by byte index.
mod ext_csd {
    pub const POWER_OFF_NOTIFICATION: usize = 34;
    pub const HPI_MGMT: usize = 161;
    pub const RST_N_FUNCTION: usize = 162;
    pub const BKOPS_EN: usize = 163;
    pub const BKOPS_START: usize = 164;
    pub const RPMB_SIZE_MULT: usize = 168;
    pub const ERASE_GROUP_DEF: usize = 175;
    pub const BOOT_BUS_CONDITIONS: usize = 177;
    pub const PARTITION_CONFIG: usize = 179;
    pub const BUS_WIDTH: usize = 183;
    pub const HS_TIMING: usize = 185;
    pub const POWER_CLASS: usize = 187;
    pub const EXT_CSD_REV: usize = 192;
    pub const CSD_STRUCTURE: usize = 194;
    pub const DEVICE_TYPE: usize = 196;
    pub const PARTITION_SWITCH_TIME: usize = 199;
    pub const SEC_COUNT: usize = 212;
    pub const BOOT_SIZE_MULT: usize = 226;
    pub const GENERIC_CMD6_TIME: usize = 248;
    pub const PRE_EOL_INFO: usize = 267;
    pub const DEVICE_LIFE_TIME_EST_TYP_A: usize = 268;
    pub const DEVICE_LIFE_TIME_EST_TYP_B: usize = 269;
    pub const BKOPS_SUPPORT: usize = 502;
    pub const S_CMD_SET: usize = 504;

    /// Fields the host may change with `SWITCH`.
    pub const WRITABLE: [usize; 11] = [
        POWER_OFF_NOTIFICATION,
        HPI_MGMT,
        RST_N_FUNCTION,
        BKOPS_EN,
        BKOPS_START,
        ERASE_GROUP_DEF,
        BOOT_BUS_CONDITIONS,
        PARTITION_CONFIG,
        BUS_WIDTH,
        HS_TIMING,
        POWER_CLASS,
    ];

    /// `PARTITION_CONFIG`: which partition data commands address.
    pub const PARTITION_ACCESS: u8 = 0x07;
    /// `HS_TIMING`: the timing interface, below the driver strength.
    pub const TIMING_INTERFACE: u8 = 0x0F;
}

/// Device status bits of an R1 response.
mod status {
    pub const ADDRESS_OUT_OF_RANGE: u32 = 1 << 31;
    pub const BLOCK_LEN_ERROR: u32 = 1 << 29;
    pub const ILLEGAL_COMMAND: u32 = 1 << 22;
    pub const READY_FOR_DATA: u32 = 1 << 8;
    pub const SWITCH_ERROR: u32 = 1 << 7;
    pub const CURRENT_STATE_SHIFT: u32 = 9;
}

/// The device name in messages about its image files.
const DEVICE: &str = "eMMC";

/// Image files behind the module's partitions.
pub struct EmmcImages {
    /// The user area; its length is the module's capacity.
    pub user: File,
    /// The two boot partitions, erased where no image is given.
    pub boot: [Option<File>; 2],
    /// Write to the images (which must be open for writing) instead of
    /// keeping this run's writes in memory.
    pub writable: bool,
    /// Sectors of the user area that read differently from its image, as
    /// if this run had written them before it started: a change to the
    /// storage the run makes and the image never sees. Only for images
    /// that are not writable.
    pub overlay: HashMap<u64, Box<[u8; SECTOR_SIZE]>>,
}

#[derive(Debug)]
pub enum ImageError {
    Io(io::Error),
    /// The user area image is not a whole number of sectors above 2 GiB.
    UserAreaSize(u64),
    /// A boot partition image is larger than the partition.
    BootPartitionSize(u64),
}

impl std::fmt::Display for ImageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "{error}"),
            Self::UserAreaSize(size) => write!(
                f,
                "user area image is {size} bytes; it must be a multiple of {SECTOR_SIZE} above 2 GiB"
            ),
            Self::BootPartitionSize(size) => write!(
                f,
                "boot partition image is {size} bytes; the partition holds {}",
                BOOT_PARTITION_SECTORS * SECTOR_SIZE as u64
            ),
        }
    }
}

/// Device states of the standard, with their `CURRENT_STATE` encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Idle = 0,
    Ready = 1,
    Identification = 2,
    Standby = 3,
    Transfer = 4,
    SendingData = 5,
    ReceivingData = 6,
}

/// What moves over the data lines after a command.
enum Data {
    /// A fixed block the card sends: the extended CSD.
    Block(Box<[u8; EXT_CSD_SIZE]>),
    /// The tuning pattern; only the host's sampling logic looks at it.
    Tuning,
    /// Sectors from or to the selected partition. `left` is `None` for an
    /// open-ended transfer, which lasts until `STOP_TRANSMISSION`.
    Sectors { next: u64, left: Option<u32> },
}

pub struct Emmc {
    /// User area, boot partition 1, boot partition 2.
    partitions: [Medium; 3],
    state: State,
    rca: u16,
    ext_csd: [u8; EXT_CSD_SIZE],
    /// Error bits waiting for the next R1 response to report them.
    errors: u32,
    /// Block count announced by `SET_BLOCK_COUNT` for the next transfer.
    block_count: Option<u32>,
    data: Option<Data>,
    notes: Arc<Notes>,
}

impl Emmc {
    pub fn new(images: EmmcImages, notes: Arc<Notes>) -> Result<Self, ImageError> {
        let size = images.user.metadata().map_err(ImageError::Io)?.len();
        let sectors = size / SECTOR_SIZE as u64;
        if !size.is_multiple_of(SECTOR_SIZE as u64)
            || sectors <= MINIMUM_USER_SECTORS
            || sectors > u64::from(u32::MAX)
        {
            return Err(ImageError::UserAreaSize(size));
        }
        let boot = |image: Option<File>| {
            if let Some(image) = &image {
                let size = image.metadata().map_err(ImageError::Io)?.len();
                if size > BOOT_PARTITION_SECTORS * SECTOR_SIZE as u64 {
                    return Err(ImageError::BootPartitionSize(size));
                }
            }
            Ok(Medium::new(
                DEVICE,
                BOOT_PARTITION_SECTORS,
                image,
                images.writable,
            ))
        };
        let [boot1, boot2] = images.boot;
        let user = Medium::new(DEVICE, sectors, Some(images.user), images.writable)
            .with_overlay(images.overlay);
        Ok(Self::with_storage(
            [user, boot(boot1)?, boot(boot2)?],
            notes,
        ))
    }

    fn with_storage(partitions: [Medium; 3], notes: Arc<Notes>) -> Self {
        let mut ext_csd = [0; EXT_CSD_SIZE];
        let mut set = |field: usize, value: u8| ext_csd[field] = value;
        // Revision 1.7 of the register: eMMC 5.0.
        set(ext_csd::EXT_CSD_REV, 7);
        set(ext_csd::CSD_STRUCTURE, 2);
        // High speed at 26 and 52 MHz, dual data rate at 1.8 V, HS200 and
        // HS400 at 1.8 V.
        set(ext_csd::DEVICE_TYPE, 0x57);
        set(ext_csd::S_CMD_SET, 1);
        set(
            ext_csd::BOOT_SIZE_MULT,
            ((BOOT_PARTITION_SECTORS * SECTOR_SIZE as u64) >> 17) as u8,
        );
        // The replay protected partition is there, 4 MiB, but cannot be
        // selected.
        set(ext_csd::RPMB_SIZE_MULT, 0x20);
        // Background operations are supported and have nothing to do.
        set(ext_csd::BKOPS_SUPPORT, 1);
        // The shortest limits there are, 10 ms each, for operations that
        // take no time here.
        set(ext_csd::GENERIC_CMD6_TIME, 1);
        set(ext_csd::PARTITION_SWITCH_TIME, 1);
        // An unworn device.
        set(ext_csd::PRE_EOL_INFO, 1);
        set(ext_csd::DEVICE_LIFE_TIME_EST_TYP_A, 1);
        set(ext_csd::DEVICE_LIFE_TIME_EST_TYP_B, 1);
        let sectors = partitions[0].sectors() as u32;
        ext_csd[ext_csd::SEC_COUNT..ext_csd::SEC_COUNT + 4].copy_from_slice(&sectors.to_le_bytes());
        Self {
            partitions,
            state: State::Idle,
            rca: 1,
            ext_csd,
            errors: 0,
            block_count: None,
            data: None,
            notes,
        }
    }

    fn status(&self) -> u32 {
        (self.state as u32) << status::CURRENT_STATE_SHIFT | status::READY_FOR_DATA
    }

    /// The R1 response to a command that found the card in its current
    /// state, reporting and clearing the collected errors.
    fn r1(&mut self) -> Option<Response> {
        Some(Response::Short(
            self.status() | std::mem::take(&mut self.errors),
        ))
    }

    fn selected_partition(&mut self) -> Option<&mut Medium> {
        let access = self.ext_csd[ext_csd::PARTITION_CONFIG] & ext_csd::PARTITION_ACCESS;
        self.partitions.get_mut(usize::from(access))
    }

    fn start_transfer(
        &mut self,
        sector: u32,
        blocks: Option<u32>,
        state: State,
    ) -> Option<Response> {
        let sector = u64::from(sector);
        let in_range = self
            .selected_partition()
            .is_some_and(|partition| sector < partition.sectors());
        if !in_range {
            self.errors |= status::ADDRESS_OUT_OF_RANGE;
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

    /// `SWITCH`: changes one byte of the extended CSD. What goes wrong is
    /// reported by the next status response.
    fn switch(&mut self, argument: u32) {
        let [_, value, index, access] = argument.to_le_bytes();
        let index = usize::from(index);
        let current = self.ext_csd[index];
        let value = match access & 3 {
            1 => current | value,
            2 => current & !value,
            3 => value,
            // Command set selection; there is only the standard one.
            _ => return self.refuse_switch(argument),
        };
        let acceptable = match index {
            // The replay protected partition cannot be selected.
            ext_csd::PARTITION_CONFIG => value & ext_csd::PARTITION_ACCESS < 3,
            // 1, 4 or 8 lines, the wider two also at dual data rate.
            ext_csd::BUS_WIDTH => matches!(value, 0 | 1 | 2 | 5 | 6),
            // Backwards compatible, high speed, HS200, HS400.
            ext_csd::HS_TIMING => value & ext_csd::TIMING_INTERFACE <= 3,
            _ => ext_csd::WRITABLE.contains(&index),
        };
        if !acceptable {
            return self.refuse_switch(argument);
        }
        self.ext_csd[index] = value;
    }

    fn refuse_switch(&mut self, argument: u32) {
        self.errors |= status::SWITCH_ERROR;
        self.notes
            .add(format!("eMMC refused SWITCH with argument {argument:#x}"));
    }

    fn end_of_block(&mut self, left: Option<u32>) {
        if left == Some(0) {
            self.data = None;
            self.state = State::Transfer;
        }
    }
}

impl Card for Emmc {
    /// Executes one command. `None` is no response: the command expects
    /// none, or the card ignored it.
    fn command(&mut self, index: u8, argument: u32) -> Option<Response> {
        use State::*;
        let addressed = (argument >> 16) as u16 == self.rca;
        let block_count = self.block_count.take();
        match (index, self.state) {
            // GO_IDLE_STATE.
            (0, _) if argument == 0 => {
                self.state = Idle;
                self.rca = 1;
                self.errors = 0;
                self.data = None;
                for field in [
                    ext_csd::PARTITION_CONFIG,
                    ext_csd::BUS_WIDTH,
                    ext_csd::HS_TIMING,
                ] {
                    self.ext_csd[field] = 0;
                }
                None
            }
            // SEND_OP_COND.
            (1, Idle | Ready) => {
                self.state = Ready;
                Some(Response::Short(OCR))
            }
            // ALL_SEND_CID.
            (2, Ready) => {
                self.state = Identification;
                Some(Response::Long(CID))
            }
            // SET_RELATIVE_ADDR.
            (3, Identification) => {
                let response = self.r1();
                self.rca = (argument >> 16) as u16;
                self.state = Standby;
                response
            }
            // SWITCH.
            (6, Transfer) => {
                let response = self.r1();
                self.switch(argument);
                response
            }
            // SELECT/DESELECT_CARD.
            (7, Standby) if addressed => {
                let response = self.r1();
                self.state = Transfer;
                response
            }
            (7, Transfer) if !addressed => {
                self.state = Standby;
                None
            }
            // SEND_EXT_CSD.
            (8, Transfer) => {
                let response = self.r1();
                self.data = Some(Data::Block(Box::new(self.ext_csd)));
                self.state = SendingData;
                response
            }
            // SEND_CSD, SEND_CID.
            (9, Standby) if addressed => Some(Response::Long(CSD)),
            (10, Standby) if addressed => Some(Response::Long(CID)),
            // STOP_TRANSMISSION.
            (12, SendingData | ReceivingData) => {
                let response = self.r1();
                self.data = None;
                self.state = Transfer;
                response
            }
            // SEND_STATUS.
            (13, Standby | Transfer | SendingData | ReceivingData) if addressed => self.r1(),
            // SET_BLOCKLEN.
            (16, Transfer) => {
                if argument != SECTOR_SIZE as u32 {
                    self.errors |= status::BLOCK_LEN_ERROR;
                }
                self.r1()
            }
            // READ_SINGLE_BLOCK, READ_MULTIPLE_BLOCK.
            (17, Transfer) => self.start_transfer(argument, Some(1), SendingData),
            (18, Transfer) => self.start_transfer(argument, block_count, SendingData),
            // SEND_TUNING_BLOCK.
            (21, Transfer) => {
                let response = self.r1();
                self.data = Some(Data::Tuning);
                self.state = SendingData;
                response
            }
            // SET_BLOCK_COUNT.
            (23, Transfer) => {
                self.block_count = Some(argument & 0xFFFF);
                self.r1()
            }
            // WRITE_BLOCK, WRITE_MULTIPLE_BLOCK.
            (24, Transfer) => self.start_transfer(argument, Some(1), ReceivingData),
            (25, Transfer) => self.start_transfer(argument, block_count, ReceivingData),
            _ => {
                self.errors |= status::ILLEGAL_COMMAND;
                self.notes.add(format!(
                    "eMMC ignored command {index} (argument {argument:#x}) in its {:?} state",
                    self.state
                ));
                None
            }
        }
    }

    /// The next block the card sends. `None` when it is not sending, or
    /// `block` is not the size of what it sends.
    fn send_block(&mut self, block: &mut [u8]) -> Option<()> {
        (self.state == State::SendingData).then_some(())?;
        match self.data.as_mut()? {
            Data::Block(bytes) => {
                (block.len() == bytes.len()).then_some(())?;
                block.copy_from_slice(&bytes[..]);
                self.end_of_block(Some(0));
            }
            Data::Tuning => return None,
            Data::Sectors { next, left } => {
                let sector = *next;
                *next += 1;
                let left = left.as_mut().map(|left| {
                    *left -= 1;
                    *left
                });
                if self.selected_partition()?.read(sector, block).is_none() {
                    self.errors |= status::ADDRESS_OUT_OF_RANGE;
                    return None;
                }
                self.end_of_block(left);
            }
        }
        Some(())
    }

    /// Takes the next block the host sends. `None` when the card is not
    /// receiving or cannot store the block.
    fn receive_block(&mut self, block: &[u8]) -> Option<()> {
        (self.state == State::ReceivingData).then_some(())?;
        let Data::Sectors { next, left } = self.data.as_mut()? else {
            return None;
        };
        let sector = *next;
        *next += 1;
        let left = left.as_mut().map(|left| {
            *left -= 1;
            *left
        });
        if self.selected_partition()?.write(sector, block).is_none() {
            self.errors |= status::ADDRESS_OUT_OF_RANGE;
            return None;
        }
        self.end_of_block(left);
        Some(())
    }

    /// Consumes the tuning block if that is what the card is sending.
    fn take_tuning_block(&mut self) -> bool {
        let tuning = self.state == State::SendingData && matches!(self.data, Some(Data::Tuning));
        if tuning {
            self.end_of_block(Some(0));
        }
        tuning
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A card with blank storage of `sectors` user sectors.
    pub(crate) fn blank(sectors: u64) -> Emmc {
        let boot = || Medium::new(DEVICE, BOOT_PARTITION_SECTORS, None, false);
        Emmc::with_storage(
            [Medium::new(DEVICE, sectors, None, false), boot(), boot()],
            Arc::default(),
        )
    }

    fn short(response: Option<Response>) -> u32 {
        match response {
            Some(Response::Short(value)) => value,
            other => panic!("expected a 48-bit response, got {other:?}"),
        }
    }

    /// Takes a card through identification into the transfer state.
    pub(crate) fn selected(sectors: u64) -> Emmc {
        let mut card = blank(sectors);
        assert_eq!(card.command(0, 0), None);
        assert_eq!(card.command(1, 0x4000_0080), Some(Response::Short(OCR)));
        assert_eq!(card.command(2, 0), Some(Response::Long(CID)));
        assert_eq!(
            short(card.command(3, 2 << 16)) >> 9 & 0xF,
            2,
            "was in identification"
        );
        assert_eq!(card.command(9, 2 << 16), Some(Response::Long(CSD)));
        assert_eq!(
            short(card.command(7, 2 << 16)) >> 9 & 0xF,
            3,
            "was in standby"
        );
        assert_eq!(short(card.command(13, 2 << 16)) >> 9 & 0xF, 4);
        card
    }

    #[test]
    fn written_sectors_read_back_and_the_rest_is_erased() {
        let mut card = selected(0x80_0000);
        let pattern: Vec<u8> = (0..SECTOR_SIZE).map(|index| index as u8 ^ 0x5A).collect();
        // Two sectors with a pre-announced count.
        short(card.command(23, 2));
        short(card.command(25, 100));
        card.receive_block(&pattern).unwrap();
        card.receive_block(&pattern).unwrap();
        assert_eq!(card.receive_block(&pattern), None, "the count is used up");
        assert_eq!(short(card.command(13, 2 << 16)) >> 9 & 0xF, 4);

        // An open-ended read across them, stopped by the host.
        short(card.command(18, 99));
        let mut block = [0xEE; SECTOR_SIZE];
        card.send_block(&mut block).unwrap();
        assert_eq!(block, [0; SECTOR_SIZE]);
        card.send_block(&mut block).unwrap();
        assert_eq!(block[..], pattern[..]);
        card.send_block(&mut block).unwrap();
        card.send_block(&mut block).unwrap();
        assert_eq!(block, [0; SECTOR_SIZE]);
        assert_eq!(short(card.command(12, 0)) >> 9 & 0xF, 5, "was sending");
        assert_eq!(card.send_block(&mut block), None);
    }

    #[test]
    fn switch_selects_the_partition_data_commands_address() {
        let mut card = selected(0x80_0000);
        let marker = [0xB0; SECTOR_SIZE];
        // PARTITION_CONFIG := 1, boot partition 1.
        short(card.command(6, 0x03B3_0100));
        short(card.command(24, 0));
        card.receive_block(&marker).unwrap();
        assert_eq!(
            short(card.command(18, BOOT_PARTITION_SECTORS as u32)) & status::ADDRESS_OUT_OF_RANGE,
            1 << 31
        );

        short(card.command(6, 0x03B3_0000));
        let mut block = [0xEE; SECTOR_SIZE];
        short(card.command(17, 0));
        card.send_block(&mut block).unwrap();
        assert_eq!(block, [0; SECTOR_SIZE], "the user area was not written");
        short(card.command(6, 0x03B3_0100));
        short(card.command(17, 0));
        card.send_block(&mut block).unwrap();
        assert_eq!(block, marker);

        // The replay protected partition, and a read-only field.
        short(card.command(6, 0x03B3_0300));
        assert_eq!(
            short(card.command(13, 2 << 16)) & status::SWITCH_ERROR,
            status::SWITCH_ERROR
        );
        short(card.command(6, 0x03C0_0500));
        assert_eq!(
            short(card.command(13, 2 << 16)) & status::SWITCH_ERROR,
            status::SWITCH_ERROR
        );
        assert_eq!(
            short(card.command(13, 2 << 16)) & status::SWITCH_ERROR,
            0,
            "reported once"
        );
    }

    #[test]
    fn extended_csd_reports_capacity_and_commands_out_of_state_go_unanswered() {
        let mut card = selected(0x3A3_E000);
        short(card.command(8, 0));
        let mut block = [0; EXT_CSD_SIZE];
        card.send_block(&mut block).unwrap();
        assert_eq!(
            block[ext_csd::SEC_COUNT..ext_csd::SEC_COUNT + 4],
            0x3A3_E000u32.to_le_bytes()
        );
        assert_eq!(block[ext_csd::BOOT_SIZE_MULT], 0x20);

        // ALL_SEND_CID belongs to identification.
        assert_eq!(card.command(2, 0), None);
        assert_eq!(
            short(card.command(13, 2 << 16)) & status::ILLEGAL_COMMAND,
            status::ILLEGAL_COMMAND
        );
        // Another card's address.
        assert_eq!(card.command(13, 5 << 16), None);
    }
}
