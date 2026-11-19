//! The calibration data of the virtual console: what the factory writes to
//! the raw partition `PRODINFO` (storage partition 27, `CalibrationBinary`)
//! before a console leaves it, and which a freshly formatted image lacks.
//!
//! # Who reads it
//!
//! `settings` (program `0100000000000009`) reads the first [`SIZE`] bytes
//! once (`0x5890`: `OpenBisStorage(27)`, one read of `0x8000` at offset 0,
//! kept for the life of the process) and serves them through `set:cal`, and
//! some fields through `set:sys`. Of the header it reads only the version
//! ([`VERSION`]). The only other reader is `boot`, which opens the partition
//! itself for the battery lot and battery version blocks (`0x46680`,
//! `0x467b0`, `0x1580` of its image). No program of the firmware compares
//! the magic number, as a string or as an immediate, or the body hash.
//!
//! # Checksums
//!
//! Every field sits in a [`Block`] whose last two bytes hold a CRC-16 of
//! the bytes before them ([`crc16`], the routine at `0x5d20`). Each getter
//! recomputes it and fails with `0x48C69` (2105-0582) on a mismatch, so an
//! all-zero partition fails every one of them. The constant of each block
//! names the `set:cal` command and the `settings` routine that checks it.
//! Two certificates are guarded by SHA-256 instead: the SSL certificate
//! (`0x4dc0`: the first `size` bytes at `0xAE0`, hash at `0x12E0`) and the
//! game card certificate (`0x5080`: `0x400` bytes at `0x2440`, hash at
//! `0x2840`); a mismatch is `0x48E69`.
//!
//! The header is the one of the public description of the format
//! (switchbrew's "Calibration" page; the `CalibrationInfoHeader` of
//! Atmosphère's `amsmitm_prodinfo_utils.cpp`): magic, version, body size,
//! model, update count, a CRC-16 of the first `0x1E` bytes at `0x1E`, and
//! the SHA-256 of the body at `0x20`. The firmware does not check it; tools
//! that handle calibration data do, and a console's has it.
//!
//! # Where the public description differs from this firmware
//!
//! - The six-byte fields from `0x210` to `0x248` are blocks of eight bytes,
//!   not padded to sixteen.
//! - The extended keys and two amiibo keys are four bytes longer than the
//!   public sizes: a key generation, reported as zero up to version 8, and
//!   amiibo keys that grow from version 9 ([`VERSION`]).
//! - The SSL certificate size is read as eight bytes, the color variation
//!   as four.
//! - The touch controller vendor (`0x4320`) and the random number block
//!   (`0x1300`) have no reader in this firmware.
//!
//! # What the virtual console has, and what it cannot have
//!
//! Everything that describes hardware is the hardware the emulator models,
//! or an ideal part where nothing reads a measurement: an original console
//! ("Icosa") with a JDI panel, an LSM6DS3H motion sensor that needs no
//! correction, a battery and speaker without factory measurements. What
//! identifies the console is [`Identity`].
//!
//! The device keys and their Nintendo-signed certificates cannot be made:
//! a console gets them from Nintendo's factory, and only those are accepted
//! by Nintendo's servers. The key blocks are therefore left unprogrammed
//! ([`UNPROGRAMMED`]: zero, with no valid CRC), which every client handles
//! at boot: `es` loads no eticket key (`0xa3ac0` tolerates `0x48C69`), and
//! the filesystem process's game card driver fails only when it first needs
//! its key. A key block with a valid CRC would be worse: `es` imports the
//! eticket key through the monitor at boot and aborts if that fails
//! (`0x7adb0`). The certificates are sealed but carry no key and no
//! signature, except the one field the firmware checks at boot: the
//! eticket certificate's name, from which `settings` parses the device ID
//! that `am` (`0x3c2b0`) and `nim` (`0xa5c70`) abort without
//! ([`eticket_certificate`]).
//!
//! The exception is the TLS client identity: without one, `ssl` fails
//! `RegisterInternalPki` (2123-0011), and `nim` stops the console with a
//! fatal error when it does that right after the first connection test.
//! The virtual console has a MADE-UP one ([`SslIdentity`]): a generated key
//! pair, wrapped with the console's keys as the firmware unwraps it, and a
//! self-signed certificate that Nintendo's servers reject. Online services
//! (eShop, account authentication), personalized tickets and game cards
//! will not work.

mod ssl;

use std::{fs::File, ops::Range, os::unix::fs::FileExt};

pub use self::ssl::SslIdentity;
use super::{NandError, bis, crypto::sha256, partition_range, storage_keys};
use crate::keys::KeySet;

/// The partition the data goes to.
const PARTITION: &str = "PRODINFO";

/// Bytes of calibration data: what `settings` reads, and what [`install`]
/// writes.
pub const SIZE: usize = 0x8000;
const HEADER_SIZE: usize = 0x40;
/// Everything after the header, all of which the body hash covers.
const BODY_SIZE: usize = SIZE - HEADER_SIZE;

const MAGIC: [u8; 4] = *b"CAL0";

/// The format version. `settings` reads it in six getters: from version 9
/// on the amiibo keys are four bytes longer (`0x25b0`, `0x2890`), and up to
/// version 8 the key generation that closes each extended key is reported
/// as zero (`0x4ca0`, `0x4f50`, `0x5290`, `0x54f0`). A CHOICE of a version
/// with every field this firmware reads; with the keys unprogrammed, the
/// difference does not show.
pub const VERSION: u32 = 9;

/// Header field `Model`, which nothing in the firmware reads. A CHOICE: the
/// product model ([`PRODUCT_MODEL`]).
const HEADER_MODEL: u16 = 1;
/// How often calibration data was installed. A CHOICE: once, by
/// [`install`].
const UPDATE_COUNT: u16 = 1;

/// Nibble table of the CRC (`settings` data at `0x797fc`): CRC-16 with the
/// reflected polynomial `0xA001`, four bits at a time.
const CRC_TABLE: [u16; 16] = [
    0x0000, 0xCC01, 0xD801, 0x1400, 0xF001, 0x3C00, 0x2800, 0xE401, 0xA001, 0x6C00, 0x7800, 0xB401,
    0x5000, 0x9C01, 0x8801, 0x4400,
];
/// What `0x5d20` starts from.
const CRC_INITIAL: u16 = 0x55AA;

/// The checksum of every block (`settings` text `0x5d20`), the low nibble
/// of each byte first.
pub fn crc16(data: &[u8]) -> u16 {
    let step = |crc: u16, nibble: u8| {
        CRC_TABLE[usize::from(crc & 0xF)] ^ (crc >> 4) ^ CRC_TABLE[usize::from(nibble & 0xF)]
    };
    data.iter()
        .fold(CRC_INITIAL, |crc, &byte| step(step(crc, byte), byte >> 4))
}

/// A field and its padding, closed by the CRC-16 of everything before the
/// CRC.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Block {
    pub offset: usize,
    /// Including the two bytes of the CRC.
    pub len: usize,
}

impl Block {
    const fn new(offset: usize, len: usize) -> Self {
        Self { offset, len }
    }

    /// Where the CRC is stored.
    pub const fn crc_offset(self) -> usize {
        self.offset + self.len - 2
    }

    /// The bytes the CRC covers.
    pub const fn data(self) -> Range<usize> {
        self.offset..self.crc_offset()
    }
}

/// The header's own CRC.
const HEADER: Block = Block::new(0x0000, 0x20);
/// Command 1 (`0x30e0`), which answers zeros, and success, on a bad CRC.
pub const CONFIGURATION_ID_1: Block = Block::new(0x0040, 0x20);
/// Commands 7 and 8 (`0x3c70`, `0x3d30`): count at `0x80`, last index at
/// `0x84`, three bytes per code from `0x88`.
pub const WLAN_COUNTRY_CODES: Block = Block::new(0x0080, 0x190);
/// Command 6 (`0x3ba0`).
pub const WLAN_MAC_ADDRESS: Block = Block::new(0x0210, 0x08);
/// Command 0 (`0x3010`).
pub const BD_ADDRESS: Block = Block::new(0x0220, 0x08);
/// Command 2 (`0x3180`).
pub const ACCELEROMETER_OFFSET: Block = Block::new(0x0230, 0x08);
/// Command 3 (`0x3250`).
pub const ACCELEROMETER_SCALE: Block = Block::new(0x0238, 0x08);
/// Command 4 (`0x3320`).
pub const GYROSCOPE_OFFSET: Block = Block::new(0x0240, 0x08);
/// Command 5 (`0x33f0`).
pub const GYROSCOPE_SCALE: Block = Block::new(0x0248, 0x08);
/// Command 9 (`0x3730`) and `set:sys` (`0x3800`).
pub const SERIAL_NUMBER: Block = Block::new(0x0250, 0x20);
/// Public layout only.
pub const ECC_P256_DEVICE_KEY: Block = Block::new(0x0270, 0x40);
/// Public layout only.
pub const ECC_P256_DEVICE_CERTIFICATE: Block = Block::new(0x02B0, 0x190);
/// Command 20 (`0x5290`) when the extended key's CRC fails.
pub const ECC_B233_DEVICE_KEY: Block = Block::new(0x0440, 0x40);
/// Command 14 (`0x8760`).
pub const ECC_B233_DEVICE_CERTIFICATE: Block = Block::new(0x0480, 0x190);
/// Public layout only.
pub const ECC_P256_ETICKET_KEY: Block = Block::new(0x0610, 0x40);
/// Public layout only.
pub const ECC_P256_ETICKET_CERTIFICATE: Block = Block::new(0x0650, 0x190);
/// Public layout only.
pub const ECC_B233_ETICKET_KEY: Block = Block::new(0x07E0, 0x40);
/// Public layout only.
pub const ECC_B233_ETICKET_CERTIFICATE: Block = Block::new(0x0820, 0x190);
/// Command 16 (`0x4ca0`) when the extended key's CRC fails.
pub const SSL_KEY: Block = Block::new(0x09B0, 0x120);
/// Command 17 (`0x4dc0`): the size of the certificate, eight bytes.
pub const SSL_CERTIFICATE_SIZE: Block = Block::new(0x0AD0, 0x10);
/// Command 18 (`0x4f50`) when the extended key's CRC fails.
pub const GAME_CARD_KEY: Block = Block::new(0x2320, 0x120);
/// Command 21 (`0x54f0`) when the extended key's CRC fails.
pub const RSA_2048_ETICKET_KEY: Block = Block::new(0x2860, 0x230);
/// Command 15 (`0x44c0`); command 42 (`0x8920`) reads the device ID out of
/// it.
pub const RSA_2048_ETICKET_CERTIFICATE: Block = Block::new(0x2A90, 0x250);
/// Command 12 (`0x3e40`), `set:sys` (`0x3f10`), and `boot`.
pub const BATTERY_LOT: Block = Block::new(0x2CE0, 0x20);
/// Command 22 (`0x4230`), which answers the first `0x5A` bytes.
pub const SPEAKER_CALIBRATION: Block = Block::new(0x2D00, 0x810);
/// Command 119 (`0x57c0`) and `set:sys` (`0xf840`).
pub const REGION_CODE: Block = Block::new(0x3510, 0x10);
/// Command 26 (`0x25b0`).
pub const AMIIBO_KEY: Block = Block::new(0x3520, 0x60);
/// Command 27 (`0x2690`).
pub const AMIIBO_ECQV_CERTIFICATE: Block = Block::new(0x3580, 0x20);
/// Command 28 (`0x2770`).
pub const AMIIBO_ECDSA_CERTIFICATE: Block = Block::new(0x35A0, 0x80);
/// Command 29 (`0x2890`).
pub const AMIIBO_ECQV_BLS_KEY: Block = Block::new(0x3620, 0x50);
/// Command 30 (`0x2970`).
pub const AMIIBO_ECQV_BLS_CERTIFICATE: Block = Block::new(0x3670, 0x30);
/// Command 31 (`0x2a50`).
pub const AMIIBO_ECQV_BLS_ROOT_CERTIFICATE: Block = Block::new(0x36A0, 0xA0);
/// `set:sys` (`0x4590`; `0x4660` answers 0 on a bad CRC).
pub const PRODUCT_MODEL_BLOCK: Block = Block::new(0x3740, 0x10);
/// `set:sys` (`0x46f0`).
pub const COLOR_VARIATION_BLOCK: Block = Block::new(0x3750, 0x10);
/// `set:sys` (`0x51c0`).
pub const LCD_BACKLIGHT_BRIGHTNESS_MAPPING_BLOCK: Block = Block::new(0x3760, 0x10);
/// Command 20 (`0x5290`).
pub const EXTENDED_ECC_B233_DEVICE_KEY: Block = Block::new(0x3770, 0x60);
/// Public layout only.
pub const EXTENDED_ECC_P256_ETICKET_KEY: Block = Block::new(0x37D0, 0x60);
/// Public layout only.
pub const EXTENDED_ECC_B233_ETICKET_KEY: Block = Block::new(0x3830, 0x60);
/// Command 21 (`0x54f0`).
pub const EXTENDED_RSA_2048_ETICKET_KEY: Block = Block::new(0x3890, 0x250);
/// Command 16 (`0x4ca0`).
pub const EXTENDED_SSL_KEY: Block = Block::new(0x3AE0, 0x140);
/// Command 18 (`0x4f50`).
pub const EXTENDED_GAME_CARD_KEY: Block = Block::new(0x3C20, 0x140);
/// Command 23 (`0x5620`).
pub const LCD_VENDOR_ID_BLOCK: Block = Block::new(0x3D60, 0x10);
/// Command 25 (`0x5410`).
pub const EXTENDED_RSA_2048_DEVICE_KEY: Block = Block::new(0x3D70, 0x250);
/// Command 24 (`0x8800`).
pub const RSA_2048_DEVICE_CERTIFICATE: Block = Block::new(0x3FC0, 0x250);
/// Command 32 (`0x56f0`).
pub const USB_TYPE_C_POWER_SOURCE_CIRCUIT: Block = Block::new(0x4210, 0x10);
/// `set:sys` (`0x47c0`).
pub const HOUSING_SUB_COLOR: Block = Block::new(0x4220, 0x10);
/// `set:sys` (`0x4890`).
pub const HOUSING_BEZEL_COLOR: Block = Block::new(0x4230, 0x10);
/// `set:sys` (`0x4960`).
pub const HOUSING_MAIN_COLOR_1: Block = Block::new(0x4240, 0x10);
/// `set:sys` (`0x4a30`).
pub const HOUSING_MAIN_COLOR_2: Block = Block::new(0x4250, 0x10);
/// `set:sys` (`0x4b00`).
pub const HOUSING_MAIN_COLOR_3: Block = Block::new(0x4260, 0x10);
/// Command 33 (`0x2b30`).
pub const ANALOG_STICK_MODULE_TYPE_L: Block = Block::new(0x4270, 0x10);
/// Command 34 (`0x2c00`).
pub const ANALOG_STICK_MODEL_PARAMETER_L: Block = Block::new(0x4280, 0x20);
/// Command 35 (`0x2cd0`).
pub const ANALOG_STICK_FACTORY_CALIBRATION_L: Block = Block::new(0x42A0, 0x10);
/// Command 36 (`0x2da0`).
pub const ANALOG_STICK_MODULE_TYPE_R: Block = Block::new(0x42B0, 0x10);
/// Command 37 (`0x2e70`).
pub const ANALOG_STICK_MODEL_PARAMETER_R: Block = Block::new(0x42C0, 0x20);
/// Command 38 (`0x2f40`).
pub const ANALOG_STICK_FACTORY_CALIBRATION_R: Block = Block::new(0x42E0, 0x10);
/// Command 39 (`0x34c0`).
pub const CONSOLE_SIX_AXIS_SENSOR_MODULE_TYPE_BLOCK: Block = Block::new(0x42F0, 0x10);
/// Command 40 (`0x3660`).
pub const CONSOLE_SIX_AXIS_SENSOR_HORIZONTAL_OFFSET_BLOCK: Block = Block::new(0x4300, 0x10);
/// Command 41 (`0x4160`), and `boot`.
pub const BATTERY_VERSION_BLOCK: Block = Block::new(0x4310, 0x10);
/// Public layout only (touch controller vendor).
pub const TOUCH_IC_VENDOR_ID: Block = Block::new(0x4320, 0x10);
/// `set:sys` (`0x4bd0`).
pub const HOME_MENU_SCHEME_MODEL: Block = Block::new(0x4330, 0x10);
/// Command 43 (`0x3590`).
pub const CONSOLE_SIX_AXIS_SENSOR_MOUNT_TYPE_BLOCK: Block = Block::new(0x4340, 0x10);

/// The blocks with a valid CRC, in the order of the data: every getter on
/// them succeeds, and a field the virtual console has no value for reads
/// as zero.
pub const SEALED: &[Block] = &[
    CONFIGURATION_ID_1,
    WLAN_COUNTRY_CODES,
    WLAN_MAC_ADDRESS,
    BD_ADDRESS,
    ACCELEROMETER_OFFSET,
    ACCELEROMETER_SCALE,
    GYROSCOPE_OFFSET,
    GYROSCOPE_SCALE,
    SERIAL_NUMBER,
    ECC_P256_DEVICE_CERTIFICATE,
    ECC_B233_DEVICE_CERTIFICATE,
    ECC_P256_ETICKET_CERTIFICATE,
    ECC_B233_ETICKET_CERTIFICATE,
    SSL_CERTIFICATE_SIZE,
    RSA_2048_ETICKET_CERTIFICATE,
    BATTERY_LOT,
    SPEAKER_CALIBRATION,
    REGION_CODE,
    AMIIBO_ECQV_CERTIFICATE,
    AMIIBO_ECDSA_CERTIFICATE,
    AMIIBO_ECQV_BLS_CERTIFICATE,
    AMIIBO_ECQV_BLS_ROOT_CERTIFICATE,
    PRODUCT_MODEL_BLOCK,
    COLOR_VARIATION_BLOCK,
    LCD_BACKLIGHT_BRIGHTNESS_MAPPING_BLOCK,
    EXTENDED_SSL_KEY,
    LCD_VENDOR_ID_BLOCK,
    RSA_2048_DEVICE_CERTIFICATE,
    USB_TYPE_C_POWER_SOURCE_CIRCUIT,
    HOUSING_SUB_COLOR,
    HOUSING_BEZEL_COLOR,
    HOUSING_MAIN_COLOR_1,
    HOUSING_MAIN_COLOR_2,
    HOUSING_MAIN_COLOR_3,
    ANALOG_STICK_MODULE_TYPE_L,
    ANALOG_STICK_MODEL_PARAMETER_L,
    ANALOG_STICK_FACTORY_CALIBRATION_L,
    ANALOG_STICK_MODULE_TYPE_R,
    ANALOG_STICK_MODEL_PARAMETER_R,
    ANALOG_STICK_FACTORY_CALIBRATION_R,
    CONSOLE_SIX_AXIS_SENSOR_MODULE_TYPE_BLOCK,
    CONSOLE_SIX_AXIS_SENSOR_HORIZONTAL_OFFSET_BLOCK,
    BATTERY_VERSION_BLOCK,
    TOUCH_IC_VENDOR_ID,
    HOME_MENU_SCHEME_MODEL,
    CONSOLE_SIX_AXIS_SENSOR_MOUNT_TYPE_BLOCK,
];

/// The private keys, which the virtual console does not have (see the
/// module documentation): all zero, CRC included, which is no valid CRC.
/// Their getters (commands 18, 20, 21, 25, 26, 29) fail with `0x48C69`; of
/// those, only 18 and 21 have callers, and each takes the failure as "no
/// key". The old SSL key is here too: command 16 answers it only when the
/// extended one fails its CRC, and the stand-in identity has the extended
/// one.
pub const UNPROGRAMMED: &[Block] = &[
    ECC_P256_DEVICE_KEY,
    ECC_B233_DEVICE_KEY,
    ECC_P256_ETICKET_KEY,
    ECC_B233_ETICKET_KEY,
    SSL_KEY,
    GAME_CARD_KEY,
    RSA_2048_ETICKET_KEY,
    AMIIBO_KEY,
    AMIIBO_ECQV_BLS_KEY,
    EXTENDED_ECC_B233_DEVICE_KEY,
    EXTENDED_ECC_P256_ETICKET_KEY,
    EXTENDED_ECC_B233_ETICKET_KEY,
    EXTENDED_RSA_2048_ETICKET_KEY,
    EXTENDED_GAME_CARD_KEY,
    EXTENDED_RSA_2048_DEVICE_KEY,
];

/// A field guarded by the SHA-256 of its contents, stored right after it.
struct Hashed {
    offset: usize,
    capacity: usize,
}

/// Command 17 (`0x4dc0`): hashed over the size [`SSL_CERTIFICATE_SIZE`]
/// holds, at most `0x800`.
const SSL_CERTIFICATE: Hashed = Hashed {
    offset: 0x0AE0,
    capacity: 0x800,
};
/// No reader; public layout (switchbrew).
const RANDOM_NUMBER: Hashed = Hashed {
    offset: 0x1300,
    capacity: 0x1000,
};
/// Command 19 (`0x5080`), hashed whole.
const GAME_CARD_CERTIFICATE: Hashed = Hashed {
    offset: 0x2440,
    capacity: 0x400,
};

/// What sets one console apart from another of the same model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Identity {
    /// The ID the secure monitor reports, config item 8: it has to be
    /// [`crate::boot::BootParams::device_id`], because `settings` compares
    /// the two before it answers command 42 (`0x8920`).
    pub device_id: u64,
    /// At most `0x18` characters; nothing in the firmware checks the
    /// format.
    pub serial_number: &'static str,
    /// In the order the octets are written: `wlan` uses the six bytes as
    /// they are (`0xb2980`).
    pub wlan_mac_address: [u8; 6],
    /// In the order the octets are written: `bluetooth` reverses them for
    /// the controller's `Write_BD_ADDR` (`0x9f528`).
    pub bd_address: [u8; 6],
}

impl Identity {
    /// The emulator's console. CHOICES:
    /// - device ID 0, the monitor's ([`crate::boot::BootParams`] default);
    /// - serial number `XAW00000000000`: the public format of an original
    ///   console's (`XAW`, ten digits, check digit), all zero, which is
    ///   also what Atmosphère writes for a serial number it blanks, so tools
    ///   recognise it as no console's;
    /// - addresses with the locally administered bit set and the multicast
    ///   bit clear, so that they cannot be any manufacturer's, spelling
    ///   `hEmu` like the volume serial numbers of [`super`], and ending in
    ///   1 for wireless LAN and 2 for Bluetooth.
    pub const VIRTUAL_CONSOLE: Self = Self {
        device_id: 0,
        serial_number: "XAW00000000000",
        wlan_mac_address: [0x02, b'h', b'E', b'm', b'u', 0x01],
        bd_address: [0x02, b'h', b'E', b'm', b'u', 0x02],
    };
}

/// CHOICE: a configuration ID of a retail console, the one an open
/// generator writes (`prodinfo_gen`'s `default_config_id`). `wlan`, `hid`,
/// `ptm` and `omm` treat consoles whose ID starts with `SDEV`, `EDEV`,
/// `HDEV`, `ADEV` or `STEV` as development units (`wlan` `0xb2adc`, `hid`
/// `0x1753f0`, `ptm` `0x2a1b0`, `omm` `0x59670`); this one starts with none
/// of them.
const CONFIGURATION_ID: &str = "MP_00_01_00_00";

/// CHOICE: `R0`, the country code `wlan` falls back to itself when the
/// count cannot be read (`0xb2dd4`), and the one of its table (`0x139070`)
/// meant for product model 1. `wlan` uses the first entry only, two
/// characters and a terminating zero (`0xb2e48`), and needs a count of at
/// least one; nothing reads the last index.
const WLAN_COUNTRY_CODE: [u8; 3] = *b"R0\0";

/// `nn::cal::ProductModel` 1, `Nx`: the original console the emulator
/// models (switchbrew). `wlan` reads it for its 80 MHz channels (6 only),
/// `settings` for the backlight (6 only).
pub const PRODUCT_MODEL: u32 = 1;

/// CHOICE: region 1, the Americas (switchbrew's `RegionCode`: 0 Japan,
/// 1 USA, 2 Europe, 3 Australia, 4 Hong Kong/Taiwan/Korea, 5 China), as an
/// open generator writes it. Only `settings` reads it (`0xf840`).
const REGION: u32 = 1;

/// `nn::cal::LcdVendorId`: vendor, model and board byte of the panel, then
/// a reserved byte (switchbrew). The emulated JDI LPM062M326A answers the
/// display ID `[0x10, 0x81, 0x26]` (`soc::dsi`): vendor JDI, board "Icosa
/// (JDI LTPS display)". Nothing in this firmware reads the field (`boot`
/// asks the panel itself, `0x29c8`), but it has to describe that panel.
const LCD_VENDOR_ID: [u8; 4] = [0x10, 0x81, 0x26, 0x00];

/// The firmware's own brightness mapping, three coefficients of
/// `m0·b² + m1·b + m2` (`lbl` in `vi`, `0x4d940`): `settings` keeps it at
/// `0xAD02C` for consoles whose field cannot be read. CHOICE: an
/// unmeasured panel gets the default.
const LCD_BACKLIGHT_BRIGHTNESS_MAPPING: [f32; 3] = [0.0, 1.0, 0.0];

/// `ColorVariation` 1: one main color for main and back, the sub color for
/// both sub colors, and the bezel color (`settings` `0x17a10`). With the
/// colors below, `set:sys` answers exactly the defaults it falls back to
/// for consoles without colors (`0x17a10`: main `0xFF323232`, sub
/// `0xFFFFFFFF`, bezel `0xFF000000`). CHOICE: the gray console of the
/// firmware's defaults.
const COLOR_VARIATION: u32 = 1;
/// Colors as R, G, B, A (switchbrew's `HousingColor`).
const MAIN_COLOR: [u8; 4] = [0x32, 0x32, 0x32, 0xFF];
const SUB_COLOR: [u8; 4] = [0xFF, 0xFF, 0xFF, 0xFF];
const BEZEL_COLOR: [u8; 4] = [0x00, 0x00, 0x00, 0xFF];

/// `nn::cal::ConsoleSixAxisSensorModuleType` 1, `Lsm6ds3h` (switchbrew),
/// the original console's motion sensor. `hid` then uses its built-in
/// scales for the sensor (table `0x1b0cdc`), the same as for 0.
const CONSOLE_SIX_AXIS_SENSOR_MODULE_TYPE: u8 = 1;
/// Mount type 1: flat, `(0, 0, 4096)` (`hid` table `0x1b0d9c`), what its
/// drivers for this sensor assume anyway.
const CONSOLE_SIX_AXIS_SENSOR_MOUNT_TYPE: u8 = 1;
/// Read by `hid` only for mount types from 3; the flat mount's value.
const CONSOLE_SIX_AXIS_SENSOR_HORIZONTAL_OFFSET: [i16; 3] = [0, 0, 0x1000];
/// CHOICE: an ideal sensor. No offset; scales that are `hid`'s nominal
/// ones for the sensor (`0xc6170`): `0x4000` per g, and `0x343B` for its
/// 936 degrees per second. `hid` takes offsets that are all zero as "use
/// the driver's" (`0x7f570`), and reads the scales only for other module
/// types.
const ACCELEROMETER_OFFSET_VALUE: [i16; 3] = [0; 3];
const ACCELEROMETER_SCALE_VALUE: [i16; 3] = [0x4000; 3];
const GYROSCOPE_OFFSET_VALUE: [i16; 3] = [0; 3];
const GYROSCOPE_SCALE_VALUE: [i16; 3] = [0x343B; 3];

// Left zero, with valid CRCs:
// - the analog sticks: the original console has none of its own, and
//   `hid` asks only with `has_sio_mcu` set, which it is not;
// - the battery lot: no lot was recorded. `boot` and `ptm` then take the
//   gauge model of an unknown lot and do not reset the fuel gauge or keep
//   its learned parameters (`ptm` `0x1cf60`, `0x46d60`);
// - battery version 0, the original console's battery `HAC-003`
//   (switchbrew; `boot` `0x6e0` and `ptm` `0x491c0` pick its charge limits);
// - USB-C power source circuit version 0, the only one `ptm` accepts on
//   this console (`0x49694` aborts above its platform maximum, 0 here);
// - the speaker calibration: nothing in this firmware asks for it;
// - home menu scheme model 0, the only value switchbrew knows;
// - the touch controller vendor 0, `FTM4CD60DA1BE` (switchbrew);
// - the game card certificate and the random number, each with its hash;
// - the certificates other than the eticket and the SSL one, which nothing
//   reads at boot.

/// The eticket device certificate: Nintendo's certificate format with an
/// ECDSA signature block (signature type `0x00010005`, signature and
/// padding to `0x80`, issuer `0x80`, key type at `0xC0`, name at `0xC4`,
/// RSA-2048 public key at `0x108`; recalled from the public format and
/// consistent with the offsets the firmware uses: `settings` parses the
/// device ID at `0xC6`, `es` the modulus at `0x108`). Only the name is
/// filled in: `NX`, the device ID as sixteen hex digits, `-0`, which is
/// what `settings` checks against the monitor's ID (`0x8920`). There is no
/// signature, issuer or key, since there is no Nintendo-issued key pair;
/// nothing at boot checks them.
pub fn eticket_certificate(device_id: u64) -> [u8; 0x240] {
    const SIGNATURE_ECDSA_SHA256: u32 = 0x0001_0005;
    const KEY_RSA_2048: u32 = 1;
    let mut certificate = [0; 0x240];
    certificate[..4].copy_from_slice(&SIGNATURE_ECDSA_SHA256.to_be_bytes());
    certificate[0xC0..0xC4].copy_from_slice(&KEY_RSA_2048.to_be_bytes());
    let name = format!("NX{device_id:016X}-0");
    certificate[0xC4..][..name.len()].copy_from_slice(name.as_bytes());
    certificate
}

fn put(data: &mut [u8], block: Block, value: &[u8]) {
    assert!(value.len() <= block.data().len(), "{block:?} overflows");
    data[block.offset..][..value.len()].copy_from_slice(value);
}

fn triple(values: [i16; 3]) -> [u8; 6] {
    let mut bytes = [0; 6];
    for (bytes, value) in bytes.chunks_mut(2).zip(values) {
        bytes.copy_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn seal_hash(data: &mut [u8], field: &Hashed, len: usize) {
    let hash = sha256(&data[field.offset..][..len]);
    data[field.offset + field.capacity..][..hash.len()].copy_from_slice(&hash);
}

fn seal_crc(data: &mut [u8], block: Block) {
    let crc = crc16(&data[block.data()]);
    data[block.crc_offset()..][..2].copy_from_slice(&crc.to_le_bytes());
}

/// The body hash and the header's CRC, which cover everything else.
fn seal_header(data: &mut [u8]) {
    let body_hash = sha256(&data[HEADER_SIZE..]);
    data[0x20..0x40].copy_from_slice(&body_hash);
    seal_crc(data, HEADER);
}

/// Puts a TLS client identity in, sealed: `key`, the key data of command
/// 16 ([`SslIdentity::wrapped_key`]), and `certificate`, the DER
/// certificate of command 17.
fn put_ssl_identity(data: &mut [u8], key: &[u8; ssl::KEY_SIZE], certificate: &[u8]) {
    assert!(certificate.len() <= SSL_CERTIFICATE.capacity);
    data[EXTENDED_SSL_KEY.data()].fill(0);
    put(data, EXTENDED_SSL_KEY, key);
    put(
        data,
        SSL_CERTIFICATE_SIZE,
        &(certificate.len() as u64).to_le_bytes(),
    );
    let field = &mut data[SSL_CERTIFICATE.offset..][..SSL_CERTIFICATE.capacity];
    field.fill(0);
    field[..certificate.len()].copy_from_slice(certificate);
    seal_crc(data, EXTENDED_SSL_KEY);
    seal_crc(data, SSL_CERTIFICATE_SIZE);
    seal_hash(data, &SSL_CERTIFICATE, certificate.len());
}

/// The device ID `settings` reads for command 42 (`0x8920`): the sixteen
/// hex digits after `NX` in the eticket certificate's name.
fn device_id(data: &[u8]) -> Option<u64> {
    let name = &data[RSA_2048_ETICKET_CERTIFICATE.offset + 0xC4..][..18];
    let digits = name.strip_prefix(b"NX")?;
    u64::from_str_radix(std::str::from_utf8(digits).ok()?, 16).ok()
}

/// The calibration data of a console with `identity`, whose TLS client
/// identity is `ssl_key` and `ssl_certificate` ([`SslIdentity`]).
pub fn build(
    identity: &Identity,
    ssl_key: &[u8; ssl::KEY_SIZE],
    ssl_certificate: &[u8],
) -> Vec<u8> {
    assert!(identity.serial_number.len() <= SERIAL_NUMBER.data().len() - 6);
    let mut data = vec![0; SIZE];
    data[..4].copy_from_slice(&MAGIC);
    data[4..8].copy_from_slice(&VERSION.to_le_bytes());
    data[8..12].copy_from_slice(&(BODY_SIZE as u32).to_le_bytes());
    data[12..14].copy_from_slice(&HEADER_MODEL.to_le_bytes());
    data[14..16].copy_from_slice(&UPDATE_COUNT.to_le_bytes());

    put(&mut data, CONFIGURATION_ID_1, CONFIGURATION_ID.as_bytes());
    let mut country_codes = 1u32.to_le_bytes().to_vec();
    country_codes.extend_from_slice(&0u32.to_le_bytes());
    country_codes.extend_from_slice(&WLAN_COUNTRY_CODE);
    put(&mut data, WLAN_COUNTRY_CODES, &country_codes);
    put(&mut data, WLAN_MAC_ADDRESS, &identity.wlan_mac_address);
    put(&mut data, BD_ADDRESS, &identity.bd_address);
    put(
        &mut data,
        ACCELEROMETER_OFFSET,
        &triple(ACCELEROMETER_OFFSET_VALUE),
    );
    put(
        &mut data,
        ACCELEROMETER_SCALE,
        &triple(ACCELEROMETER_SCALE_VALUE),
    );
    put(&mut data, GYROSCOPE_OFFSET, &triple(GYROSCOPE_OFFSET_VALUE));
    put(&mut data, GYROSCOPE_SCALE, &triple(GYROSCOPE_SCALE_VALUE));
    put(&mut data, SERIAL_NUMBER, identity.serial_number.as_bytes());
    put(
        &mut data,
        RSA_2048_ETICKET_CERTIFICATE,
        &eticket_certificate(identity.device_id),
    );
    put(&mut data, REGION_CODE, &REGION.to_le_bytes());
    put(&mut data, PRODUCT_MODEL_BLOCK, &PRODUCT_MODEL.to_le_bytes());
    put(
        &mut data,
        COLOR_VARIATION_BLOCK,
        &COLOR_VARIATION.to_le_bytes(),
    );
    let mapping: Vec<u8> = LCD_BACKLIGHT_BRIGHTNESS_MAPPING
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    put(&mut data, LCD_BACKLIGHT_BRIGHTNESS_MAPPING_BLOCK, &mapping);
    put(&mut data, LCD_VENDOR_ID_BLOCK, &LCD_VENDOR_ID);
    put(&mut data, HOUSING_SUB_COLOR, &SUB_COLOR);
    put(&mut data, HOUSING_BEZEL_COLOR, &BEZEL_COLOR);
    put(&mut data, HOUSING_MAIN_COLOR_1, &MAIN_COLOR);
    put(
        &mut data,
        CONSOLE_SIX_AXIS_SENSOR_MODULE_TYPE_BLOCK,
        &[CONSOLE_SIX_AXIS_SENSOR_MODULE_TYPE],
    );
    put(
        &mut data,
        CONSOLE_SIX_AXIS_SENSOR_HORIZONTAL_OFFSET_BLOCK,
        &triple(CONSOLE_SIX_AXIS_SENSOR_HORIZONTAL_OFFSET),
    );
    put(
        &mut data,
        CONSOLE_SIX_AXIS_SENSOR_MOUNT_TYPE_BLOCK,
        &[CONSOLE_SIX_AXIS_SENSOR_MOUNT_TYPE],
    );
    put_ssl_identity(&mut data, ssl_key, ssl_certificate);

    for &block in SEALED {
        seal_crc(&mut data, block);
    }
    seal_hash(&mut data, &RANDOM_NUMBER, RANDOM_NUMBER.capacity);
    seal_hash(
        &mut data,
        &GAME_CARD_CERTIFICATE,
        GAME_CARD_CERTIFICATE.capacity,
    );
    seal_header(&mut data);
    data
}

/// Writes the calibration data of a console with `identity` to the
/// partitioned `image`, under the calibration storage key of the console
/// `keys` describes, with the console's stand-in TLS client identity
/// ([`SslIdentity`]) wrapped with the same keys. Only [`SIZE`] bytes are
/// written: the rest of the partition is never read.
pub fn install(image: &File, keys: &KeySet, identity: &Identity) -> Result<(), NandError> {
    let cipher = storage_keys(image, keys)?.cipher(bis::KEY_CALIBRATION);
    let ssl = SslIdentity::generate(identity.device_id);
    let ssl_key = ssl.wrapped_key(keys, identity.device_id)?;
    let mut data = build(identity, &ssl_key, ssl.certificate());
    cipher.encrypt(0, &mut data);
    image
        .write_all_at(&data, partition_start())
        .map_err(NandError::Image)
}

/// Gives the console whose calibration data the partitioned `image` holds
/// the stand-in TLS client identity [`install`] would, and leaves
/// everything else as it is but the update count: for images whose
/// calibration data was written before it had one. The console is the one
/// the eticket certificate names; its device ID is returned.
pub fn install_ssl_identity(image: &File, keys: &KeySet) -> Result<u64, NandError> {
    let cipher = storage_keys(image, keys)?.cipher(bis::KEY_CALIBRATION);
    let mut data = vec![0; SIZE];
    image
        .read_exact_at(&mut data, partition_start())
        .map_err(NandError::Image)?;
    cipher.decrypt(0, &mut data);
    if data[..4] != MAGIC {
        return Err(NandError::Calibration(
            "PRODINFO holds no calibration data under these keys",
        ));
    }
    let device_id = device_id(&data).ok_or(NandError::Calibration(
        "the eticket certificate in PRODINFO names no device ID",
    ))?;
    let ssl = SslIdentity::generate(device_id);
    put_ssl_identity(
        &mut data,
        &ssl.wrapped_key(keys, device_id)?,
        ssl.certificate(),
    );
    let updates = u16::from_le_bytes([data[14], data[15]]).saturating_add(1);
    data[14..16].copy_from_slice(&updates.to_le_bytes());
    seal_header(&mut data);
    cipher.encrypt(0, &mut data);
    image
        .write_all_at(&data, partition_start())
        .map_err(NandError::Image)?;
    Ok(device_id)
}

/// Gives the console whose calibration data the partitioned `image` holds
/// the WLAN MAC address `mac` (command 6 of the calibration data), and
/// leaves everything else as it is but the update count: a second
/// emulated console with its own address, for several emulators sharing
/// the air (`wlan` builds the radio's NVRAM from this block and sets the
/// radio's address from it).
pub fn set_wlan_mac_address(image: &File, keys: &KeySet, mac: [u8; 6]) -> Result<(), NandError> {
    let cipher = storage_keys(image, keys)?.cipher(bis::KEY_CALIBRATION);
    let mut data = vec![0; SIZE];
    image
        .read_exact_at(&mut data, partition_start())
        .map_err(NandError::Image)?;
    cipher.decrypt(0, &mut data);
    if data[..4] != MAGIC {
        return Err(NandError::Calibration(
            "PRODINFO holds no calibration data under these keys",
        ));
    }
    put(&mut data, WLAN_MAC_ADDRESS, &mac);
    seal_crc(&mut data, WLAN_MAC_ADDRESS);
    let updates = u16::from_le_bytes([data[14], data[15]]).saturating_add(1);
    data[14..16].copy_from_slice(&updates.to_le_bytes());
    seal_header(&mut data);
    cipher.encrypt(0, &mut data);
    image
        .write_all_at(&data, partition_start())
        .map_err(NandError::Image)
}

fn partition_start() -> u64 {
    partition_range(PARTITION)
        .expect("the layout has the partition")
        .start
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nand::{bis::BisKeys, format_user_area};

    /// CRC-16 one bit at a time: reflected polynomial `0xA001`.
    fn bitwise(initial: u16, data: &[u8]) -> u16 {
        data.iter().fold(initial, |mut crc, &byte| {
            crc ^= u16::from(byte);
            for _ in 0..8 {
                crc = if crc & 1 != 0 {
                    crc >> 1 ^ 0xA001
                } else {
                    crc >> 1
                };
            }
            crc
        })
    }

    #[test]
    fn crc_is_the_firmware_table_algorithm() {
        // The transcribed table is the polynomial's nibble table.
        for (nibble, &entry) in CRC_TABLE.iter().enumerate() {
            let mut crc = nibble as u16;
            for _ in 0..4 {
                crc = if crc & 1 != 0 {
                    crc >> 1 ^ 0xA001
                } else {
                    crc >> 1
                };
            }
            assert_eq!(entry, crc, "entry {nibble}");
        }
        // CRC-16/ARC check value, then the firmware's starting value.
        assert_eq!(bitwise(0, b"123456789"), 0xBB3D);
        let data: Vec<u8> = (0..600u32).map(|index| (index * 151 + 7) as u8).collect();
        for len in [0, 1, 6, 14, 30, 590] {
            assert_eq!(crc16(&data[..len]), bitwise(CRC_INITIAL, &data[..len]));
        }
        assert_eq!(crc16(&[]), 0x55AA);
    }

    /// Per getter: start of the data it checks, bytes checked, where it
    /// reads the CRC, as `settings` loads them into registers before calling
    /// `0x5d20` (call site in the comment).
    const GETTERS: [(usize, usize, usize); 56] = [
        (0x3520, 0x5E, 0x357E),  // 0x25f4, command 26
        (0x3580, 0x1E, 0x359E),  // 0x26d8, 27
        (0x35A0, 0x7E, 0x361E),  // 0x27b8, 28
        (0x3620, 0x4E, 0x366E),  // 0x28d4, 29
        (0x3670, 0x2E, 0x369E),  // 0x29b8, 30
        (0x36A0, 0x9E, 0x373E),  // 0x2a94, 31
        (0x4270, 0x0E, 0x427E),  // 0x2b74, 33
        (0x4280, 0x1E, 0x429E),  // 0x2c48, 34
        (0x42A0, 0x0E, 0x42AE),  // 0x2d18, 35
        (0x42B0, 0x0E, 0x42BE),  // 0x2de4, 36
        (0x42C0, 0x1E, 0x42DE),  // 0x2eb8, 37
        (0x42E0, 0x0E, 0x42EE),  // 0x2f88, 38
        (0x0220, 0x06, 0x0226),  // 0x3054, 0
        (0x0040, 0x1E, 0x005E),  // 0x3134, 1
        (0x0230, 0x06, 0x0236),  // 0x31c4, 2
        (0x0238, 0x06, 0x023E),  // 0x3294, 3
        (0x0240, 0x06, 0x0246),  // 0x3364, 4
        (0x0248, 0x06, 0x024E),  // 0x3434, 5
        (0x42F0, 0x0E, 0x42FE),  // 0x3504, 39
        (0x4340, 0x0E, 0x434E),  // 0x35d4, 43
        (0x4300, 0x0E, 0x430E),  // 0x36a8, 40
        (0x0250, 0x1E, 0x026E),  // 0x3774, 9 (and 0x3868, set:sys)
        (0x0210, 0x06, 0x0216),  // 0x3be4, 6
        (0x0080, 0x18E, 0x020E), // 0x3cac, 7 (and 0x3d84, 8)
        (0x2CE0, 0x1E, 0x2CFE),  // 0x3e88, 12 (and 0x3f74, set:sys)
        (0x4310, 0x0E, 0x431E),  // 0x41a4, 41
        (0x2D00, 0x80E, 0x350E), // 0x4278, 22
        (0x0480, 0x18E, 0x060E), // 0x436c, 14
        (0x3FC0, 0x24E, 0x420E), // 0x4434, 24
        (0x2A90, 0x24E, 0x2CDE), // 0x4504, 15
        (0x3740, 0x0E, 0x374E),  // 0x45d4, set:sys (and 0x46b0)
        (0x3750, 0x0E, 0x375E),  // 0x4734, set:sys
        (0x4220, 0x0E, 0x422E),  // 0x4804, set:sys
        (0x4230, 0x0E, 0x423E),  // 0x48d4, set:sys
        (0x4240, 0x0E, 0x424E),  // 0x49a4, set:sys
        (0x4250, 0x0E, 0x425E),  // 0x4a74, set:sys
        (0x4260, 0x0E, 0x426E),  // 0x4b44, set:sys
        (0x4330, 0x0E, 0x433E),  // 0x4c14, set:sys
        (0x3AE0, 0x13E, 0x3C1E), // 0x4ce4, 16
        (0x09B0, 0x11E, 0x0ACE), // 0x4d3c, 16 fallback
        (0x0AD0, 0x0E, 0x0ADE),  // 0x4e04, 17
        (0x3C20, 0x13E, 0x3D5E), // 0x4f94, 18
        (0x2320, 0x11E, 0x243E), // 0x4ff4, 18 fallback
        (0x3760, 0x0E, 0x376E),  // 0x5204, set:sys
        (0x3770, 0x5E, 0x37CE),  // 0x52d8, 20
        (0x0440, 0x3E, 0x047E),  // 0x53b8, 20 fallback
        (0x3D70, 0x24E, 0x3FBE), // 0x5454, 25
        (0x3890, 0x24E, 0x3ADE), // 0x5534, 21
        (0x2860, 0x22E, 0x2A8E), // 0x5594, 21 fallback
        (0x3D60, 0x0E, 0x3D6E),  // 0x5664, 23
        (0x4210, 0x0E, 0x421E),  // 0x5734, 32
        (0x3510, 0x0E, 0x351E),  // 0x5804, 119
        // The public layout's blocks without a getter.
        (0x0270, 0x3E, 0x02AE),
        (0x02B0, 0x18E, 0x043E),
        (0x37D0, 0x5E, 0x382E),
        (0x4320, 0x0E, 0x432E),
    ];

    fn all_blocks() -> Vec<Block> {
        let mut blocks: Vec<Block> = SEALED.iter().chain(UNPROGRAMMED).copied().collect();
        blocks.sort_by_key(|block| block.offset);
        blocks
    }

    #[test]
    fn blocks_are_where_the_getters_check_them() {
        let blocks = all_blocks();
        for (offset, len, crc_at) in GETTERS {
            assert!(
                blocks
                    .iter()
                    .any(|block| block.data() == (offset..offset + len)
                        && block.crc_offset() == crc_at),
                "{offset:#x}"
            );
        }
        for pair in blocks.windows(2) {
            assert!(pair[0].offset + pair[0].len <= pair[1].offset, "{pair:?}");
        }
        assert!(blocks[0].offset >= HEADER_SIZE);
        let hashed = [SSL_CERTIFICATE, RANDOM_NUMBER, GAME_CARD_CERTIFICATE];
        for field in &hashed {
            let range = field.offset..field.offset + field.capacity + 0x20;
            assert!(
                blocks
                    .iter()
                    .all(|block| block.offset + block.len <= range.start
                        || block.offset >= range.end)
            );
        }
    }

    /// The calibration data of `identity` with the tests' TLS identity.
    fn built(identity: &Identity) -> Vec<u8> {
        let keys = KeySet::parse(&ssl::tests::key_file()).unwrap();
        let ssl = ssl::tests::identity();
        let key = ssl.wrapped_key(&keys, ssl::tests::DEVICE_ID).unwrap();
        build(identity, &key, ssl.certificate())
    }

    #[test]
    fn every_getter_but_the_keys_succeeds() {
        let data = built(&Identity::VIRTUAL_CONSOLE);
        let crc_at =
            |block: Block| u16::from_le_bytes(data[block.crc_offset()..][..2].try_into().unwrap());
        for &block in SEALED {
            assert_eq!(crc_at(block), crc16(&data[block.data()]), "{block:?}");
        }
        for &block in UNPROGRAMMED {
            assert!(
                data[block.offset..][..block.len]
                    .iter()
                    .all(|&byte| byte == 0)
            );
            assert_ne!(crc_at(block), crc16(&data[block.data()]), "{block:?}");
        }
        // The SHA-256 fields, the body and the header.
        let certificate = ssl::tests::identity().certificate();
        assert_eq!(data[0xAD0..0xAD8], (certificate.len() as u64).to_le_bytes());
        assert_eq!(data[0x12E0..0x1300], sha256(certificate));
        assert_eq!(data[0x2300..0x2320], sha256(&data[0x1300..0x2300]));
        assert_eq!(data[0x2840..0x2860], sha256(&data[0x2440..0x2840]));
        assert_eq!(data[..4], *b"CAL0");
        assert_eq!(data[8..12], 0x7FC0u32.to_le_bytes());
        assert_eq!(data[0x20..0x40], sha256(&data[0x40..]));
        assert_eq!(data[0x1E..0x20], crc16(&data[..0x1E]).to_le_bytes());
    }

    #[test]
    fn the_tls_identity_is_where_commands_16_and_17_read_it() {
        let data = built(&Identity::VIRTUAL_CONSOLE);
        let keys = KeySet::parse(&ssl::tests::key_file()).unwrap();
        let ssl = ssl::tests::identity();
        // Command 16: 0x134 bytes from 0x3AE0, the generation (0) last,
        // zeros up to the CRC.
        let key = ssl.wrapped_key(&keys, ssl::tests::DEVICE_ID).unwrap();
        assert_eq!(data[0x3AE0..0x3C14], key);
        assert_eq!(data[0x3C10..0x3C14], [0; 4]);
        assert!(data[0x3C14..0x3C1E].iter().all(|&byte| byte == 0));
        assert_eq!(
            data[0x3C1E..0x3C20],
            crc16(&data[0x3AE0..0x3C1E]).to_le_bytes()
        );
        // The old key is not there; its getter is not reached.
        assert!(data[0x9B0..0xAD0].iter().all(|&byte| byte == 0));
        // Command 17: the size, eight bytes with their CRC, then the
        // certificate and its hash.
        let certificate = ssl.certificate();
        assert_eq!(data[0xADE..0xAE0], crc16(&data[0xAD0..0xADE]).to_le_bytes());
        assert_eq!(&data[0xAE0..][..certificate.len()], certificate);
        assert!(
            data[0xAE0 + certificate.len()..0x12E0]
                .iter()
                .all(|&byte| byte == 0)
        );
        assert_eq!(certificate[0], 0x30, "DER");
    }

    #[test]
    fn values_land_where_the_getters_read_them() {
        let identity = Identity {
            device_id: 0x0012_3456_789A_BCDE,
            ..Identity::VIRTUAL_CONSOLE
        };
        let data = built(&identity);
        // Command 42: sixteen hex digits at certificate + 0xC6.
        let digits = std::str::from_utf8(&data[0x2A90 + 0xC6..][..16]).unwrap();
        assert_eq!(u64::from_str_radix(digits, 16).unwrap(), identity.device_id);
        assert_eq!(&data[0x2A90 + 0xC4..][..2], b"NX");
        // Commands 6, 0, 7/8, 9.
        assert_eq!(data[0x210..0x216], identity.wlan_mac_address);
        assert_eq!(data[0x220..0x226], identity.bd_address);
        assert_eq!(data[0x80..0x8B], [1, 0, 0, 0, 0, 0, 0, 0, b'R', b'0', 0]);
        assert_eq!(&data[0x250..0x25E], b"XAW00000000000");
        // Commands 3, 5, 39, 40, 43, 32, 41, 23, 119; set:sys product
        // model and colors.
        assert_eq!(data[0x238..0x23E], [0x00, 0x40, 0x00, 0x40, 0x00, 0x40]);
        assert_eq!(data[0x248..0x24E], [0x3B, 0x34, 0x3B, 0x34, 0x3B, 0x34]);
        assert_eq!(data[0x42F0], 1);
        assert_eq!(data[0x4300..0x4306], [0, 0, 0, 0, 0x00, 0x10]);
        assert_eq!(data[0x4340], 1);
        assert_eq!(data[0x4210], 0);
        assert_eq!(data[0x4310], 0);
        assert_eq!(data[0x3D60..0x3D64], [0x10, 0x81, 0x26, 0x00]);
        assert_eq!(data[0x3510..0x3514], 1u32.to_le_bytes());
        assert_eq!(data[0x3740..0x3744], 1u32.to_le_bytes());
        let word = |offset: usize| u32::from_le_bytes(data[offset..][..4].try_into().unwrap());
        assert_eq!(word(0x3750), 1);
        assert_eq!(word(0x4240), 0xFF32_3232);
        assert_eq!(word(0x4220), 0xFFFF_FFFF);
        assert_eq!(word(0x4230), 0xFF00_0000);
        assert_eq!(word(0x3764), 1.0f32.to_bits());
        // Configuration ID: no development unit prefix.
        for prefix in ["SDEV", "EDEV", "HDEV", "ADEV", "STEV"] {
            assert!(!data[0x40..].starts_with(prefix.as_bytes()));
        }
    }

    #[test]
    fn installed_data_decrypts_back_to_the_built_data_and_takes_an_identity_later() {
        let keys = KeySet::parse(&ssl::tests::key_file()).unwrap();
        // The tests' TLS identity is this console's.
        let identity = Identity {
            device_id: ssl::tests::DEVICE_ID,
            ..Identity::VIRTUAL_CONSOLE
        };

        let path = std::env::temp_dir().join(format!(
            "horizon-emu-calibration-test-{}.img",
            std::process::id()
        ));
        let image = File::options()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        let unpartitioned = install(&image, &keys, &identity);
        assert!(matches!(unpartitioned, Err(NandError::NotPartitioned)));
        format_user_area(&image).unwrap();
        let empty = install_ssl_identity(&image, &keys);
        assert!(matches!(empty, Err(NandError::Calibration(_))));
        install(&image, &keys, &identity).unwrap();

        let cipher = BisKeys::derive(&keys).unwrap().cipher(bis::KEY_CALIBRATION);
        let range = partition_range("PRODINFO").unwrap();
        let stored = || {
            let mut stored = vec![0; SIZE];
            image.read_exact_at(&mut stored, range.start).unwrap();
            stored
        };
        let mut installed = stored();
        let expected = built(&identity);
        assert_ne!(installed, expected, "stored encrypted");
        cipher.decrypt(0, &mut installed);
        assert_eq!(installed, expected);

        // What an earlier formatter wrote: no key, a certificate of size 0.
        let mut earlier = installed.clone();
        earlier[EXTENDED_SSL_KEY.offset..][..EXTENDED_SSL_KEY.len].fill(0);
        earlier[0xAD0..0x12E0].fill(0);
        seal_crc(&mut earlier, SSL_CERTIFICATE_SIZE);
        seal_hash(&mut earlier, &SSL_CERTIFICATE, 0);
        seal_header(&mut earlier);
        cipher.encrypt(0, &mut earlier);
        image.write_all_at(&earlier, range.start).unwrap();

        let patched = install_ssl_identity(&image, &keys);
        let mut patched_data = stored();
        drop(image);
        let _ = std::fs::remove_file(&path);
        assert_eq!(patched.unwrap(), identity.device_id);
        cipher.decrypt(0, &mut patched_data);
        // As installed, one update later.
        assert_eq!(patched_data[14..16], 2u16.to_le_bytes());
        assert_eq!(
            patched_data[0x1E..0x20],
            crc16(&patched_data[..0x1E]).to_le_bytes()
        );
        assert_eq!(patched_data[..14], expected[..14]);
        assert_eq!(patched_data[0x20..], expected[0x20..]);
    }
}
