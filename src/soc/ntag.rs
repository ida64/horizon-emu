//! An NXP NTAG215: the NFC Forum Type 2 tag inside an amiibo.
//!
//! The tag's memory is 135 pages of 4 bytes (540 bytes): the 7-byte serial
//! number with its check bytes and the static lock bytes (pages 0 to 2),
//! the capability container (page 3), 504 bytes of user memory (pages 4 to
//! 0x81), the dynamic lock bytes (0x82) and the configuration pages (0x83
//! to 0x86: mirror and AUTH0, ACCESS, the password and its acknowledge).
//! The commands a reader sends it, what they answer and which writes the
//! lock bits and the password stop are those of NXP's data sheet
//! ("NTAG213/215/216", rev. 3.2, sections 8.5, 8.8 and 10); a dump is that
//! memory as it lies, byte for byte.
//!
//! Nothing here knows what an amiibo is: what the pages hold, their
//! encryption and their signatures are for the console's software to make
//! sense of.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Bytes of a dump: every page of the tag.
pub const SIZE: usize = 540;
/// Pages, and the last one a command can name.
const PAGES: usize = SIZE / 4;
const LAST_PAGE: u8 = (PAGES - 1) as u8;
/// The dynamic lock bytes and the configuration pages.
const DYNAMIC_LOCK: usize = 0x82;
const CFG0: usize = 0x83;
const CFG1: usize = 0x84;
const PWD: usize = 0x85;
const PACK: usize = 0x86;
/// The first page the dynamic lock bits cover, and how many pages each of
/// them locks on an NTAG215 (data sheet, 8.5.3).
const DYNAMIC_FIRST: usize = 0x10;
const DYNAMIC_GRANULARITY: usize = 16;
/// User memory ends before the dynamic lock bytes.
const USER_END: usize = DYNAMIC_LOCK;

/// What GET_VERSION answers for an NTAG215 (data sheet, 10.1, table 28):
/// fixed header, NXP, NTAG, 50 pF, major, minor, 504 bytes, ISO/IEC
/// 14443-3.
const VERSION: [u8; 8] = [0x00, 0x04, 0x04, 0x02, 0x01, 0x00, 0x11, 0x03];

/// The tag's 4-bit NAK (data sheet, 9.3): an invalid argument (a page out
/// of range or locked), or a failed password.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nak {
    InvalidArgument = 0x0,
    AuthenticationFailed = 0x5,
}

/// Command codes (data sheet, section 10).
const GET_VERSION: u8 = 0x60;
const READ: u8 = 0x30;
const FAST_READ: u8 = 0x3A;
const WRITE: u8 = 0xA2;
const PWD_AUTH: u8 = 0x1B;
const READ_SIG: u8 = 0x3C;

/// One NTAG215, with the file its memory came from if writes go back to it.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ntag215 {
    memory: Vec<u8>,
    /// The ECC originality signature NXP programs into each tag, which
    /// READ_SIG answers. A dump of the 540 bytes of memory does not hold
    /// it, so a tag made from one has none: 32 zero bytes.
    signature: [u8; 32],
    /// PWD_AUTH has succeeded since the tag came into the field.
    authenticated: bool,
    /// Where writes are saved, when the user asked for that.
    file: Option<PathBuf>,
}

impl std::fmt::Debug for Ntag215 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NTAG215 {:02x?}", self.uid())
    }
}

impl Ntag215 {
    /// A tag holding `dump`, which must be the 540 bytes of an NTAG215.
    pub fn from_dump(dump: &[u8]) -> Result<Self, String> {
        if dump.len() != SIZE {
            return Err(format!(
                "an NTAG215 dump is {SIZE} bytes, this one is {}",
                dump.len()
            ));
        }
        Ok(Self {
            memory: dump.to_vec(),
            signature: [0; 32],
            authenticated: false,
            file: None,
        })
    }

    /// The tag dumped in `path`; with `writable`, what is written to the
    /// tag is saved there too.
    pub fn load(path: &Path, writable: bool) -> Result<Self, String> {
        let dump = std::fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
        let mut tag =
            Self::from_dump(&dump).map_err(|error| format!("{}: {error}", path.display()))?;
        tag.file = writable.then(|| path.to_path_buf());
        Ok(tag)
    }

    /// The 7-byte serial number: page 0 without its check byte, then page
    /// 1 (data sheet, 8.5.1).
    pub fn uid(&self) -> [u8; 7] {
        let m = &self.memory;
        [m[0], m[1], m[2], m[4], m[5], m[6], m[7]]
    }

    /// The memory as it is now.
    pub fn dump(&self) -> &[u8] {
        &self.memory
    }

    /// Saves the memory to the tag's file, if it has one.
    pub fn save_to_file(&self) -> Result<(), String> {
        match &self.file {
            Some(path) => std::fs::write(path, &self.memory)
                .map_err(|error| format!("writing the tag back to {}: {error}", path.display())),
            None => Ok(()),
        }
    }

    fn page(&self, page: usize) -> &[u8] {
        &self.memory[page * 4..page * 4 + 4]
    }

    /// A page as a reader sees it: the password and its acknowledge read
    /// as zeros (data sheet, 8.8.1).
    fn read_page(&self, page: usize) -> [u8; 4] {
        match page {
            PWD | PACK => [0; 4],
            _ => self.page(page).try_into().expect("a page is 4 bytes"),
        }
    }

    /// AUTH0: the first page whose writing needs the password.
    fn auth0(&self) -> usize {
        self.page(CFG0)[3].into()
    }

    /// Whether the password also guards reading (ACCESS bit PROT).
    fn read_protected(&self) -> bool {
        self.page(CFG1)[0] & 0x80 != 0
    }

    fn may_read(&self, page: usize) -> bool {
        self.authenticated || !self.read_protected() || page < self.auth0()
    }

    /// Whether the lock bits leave `page` writable (data sheet, 8.5.2,
    /// 8.5.3 and 8.5.7).
    fn locked(&self, page: usize) -> bool {
        let lock = self.page(2);
        let static_bits = u16::from_le_bytes([lock[2], lock[3]]);
        match page {
            0 | 1 => true,
            // The lock bytes themselves only ever gain bits; their
            // block-locking bits are not modeled as stopping that.
            2 => false,
            3..=15 => static_bits & (1 << page) != 0,
            DYNAMIC_FIRST..USER_END => {
                let group = (page - DYNAMIC_FIRST) / DYNAMIC_GRANULARITY;
                self.page(DYNAMIC_LOCK)[0] & (1 << group) != 0
            }
            DYNAMIC_LOCK => false,
            // CFGLCK in ACCESS locks the two configuration pages; the
            // password and its acknowledge can always be written.
            CFG0 | CFG1 => self.page(CFG1)[0] & 0x40 != 0,
            _ => false,
        }
    }

    /// Answers `frame`, a command without its CRC, as the tag does: the
    /// bytes of its answer, an ACK being an empty one, or a NAK.
    pub fn command(&mut self, frame: &[u8]) -> Result<Vec<u8>, Nak> {
        match *frame {
            [GET_VERSION] => Ok(VERSION.to_vec()),
            // Four pages from `page`, rolling over to page 0 past the end
            // (data sheet, 10.2).
            [READ, page] if page <= LAST_PAGE => {
                let pages = (0..4).map(|n| (usize::from(page) + n) % PAGES);
                let mut data = Vec::with_capacity(16);
                for page in pages {
                    if !self.may_read(page) {
                        return Err(Nak::InvalidArgument);
                    }
                    data.extend(self.read_page(page));
                }
                Ok(data)
            }
            [FAST_READ, start, end] if start <= end && end <= LAST_PAGE => {
                let mut data = Vec::with_capacity(usize::from(end - start + 1) * 4);
                for page in usize::from(start)..=usize::from(end) {
                    if !self.may_read(page) {
                        return Err(Nak::InvalidArgument);
                    }
                    data.extend(self.read_page(page));
                }
                Ok(data)
            }
            [WRITE, page, a, b, c, d] if (2..=LAST_PAGE).contains(&page) => {
                let page = usize::from(page);
                if self.locked(page) || (page >= self.auth0() && !self.authenticated) {
                    return Err(Nak::InvalidArgument);
                }
                let at = page * 4;
                let data = [a, b, c, d];
                match page {
                    // Only the lock bytes of page 2 are written, and lock
                    // bits and the capability container (one-time
                    // programmable) only ever gain bits.
                    2 => {
                        self.memory[at + 2] |= c;
                        self.memory[at + 3] |= d;
                    }
                    3 | DYNAMIC_LOCK => {
                        for (byte, new) in self.memory[at..at + 4].iter_mut().zip(data) {
                            *byte |= new;
                        }
                    }
                    _ => self.memory[at..at + 4].copy_from_slice(&data),
                }
                Ok(Vec::new())
            }
            // The password against PWD; PACK answers a success (data
            // sheet, 10.7). The limit on failed attempts is not modeled.
            [PWD_AUTH, a, b, c, d] => {
                if [a, b, c, d] != self.page(PWD) {
                    return Err(Nak::AuthenticationFailed);
                }
                self.authenticated = true;
                Ok(self.page(PACK)[..2].to_vec())
            }
            [READ_SIG, 0x00] => Ok(self.signature.to_vec()),
            _ => Err(Nak::InvalidArgument),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// An NTAG215 as delivered (data sheet, 8.5, figure 8 and table 5):
    /// NXP's serial number prefix, the check bytes, an empty capability
    /// container for 504 bytes, nothing locked, no password in force.
    pub(crate) fn blank(uid: [u8; 7]) -> Ntag215 {
        let mut memory = vec![0; SIZE];
        memory[..3].copy_from_slice(&uid[..3]);
        memory[3] = 0x88 ^ uid[0] ^ uid[1] ^ uid[2];
        memory[4..8].copy_from_slice(&uid[3..]);
        memory[8] = uid[3] ^ uid[4] ^ uid[5] ^ uid[6];
        memory[12..16].copy_from_slice(&[0xE1, 0x10, 0x3E, 0x00]);
        memory[DYNAMIC_LOCK * 4 + 3] = 0xBD;
        memory[CFG0 * 4..CFG0 * 4 + 4].copy_from_slice(&[0x04, 0x00, 0x00, 0xFF]);
        memory[CFG1 * 4 + 1] = 0x05;
        memory[PWD * 4..PWD * 4 + 4].copy_from_slice(&[0xFF; 4]);
        Ntag215::from_dump(&memory).unwrap()
    }

    const UID: [u8; 7] = [0x04, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66];

    #[test]
    fn a_dump_is_the_whole_memory_and_the_serial_skips_its_check_byte() {
        assert!(Ntag215::from_dump(&[0; 532]).is_err());
        let tag = blank(UID);
        assert_eq!(tag.uid(), UID);
        assert_eq!(tag.dump().len(), SIZE);
    }

    #[test]
    fn reads_roll_over_and_hide_the_password() {
        let mut tag = blank(UID);
        assert_eq!(tag.command(&[GET_VERSION]).unwrap(), VERSION);
        let read = tag.command(&[READ, 0]).unwrap();
        assert_eq!(read[..4], [0x04, 0x11, 0x22, 0x88 ^ 0x04 ^ 0x11 ^ 0x22]);
        assert_eq!(read[12..16], [0xE1, 0x10, 0x3E, 0x00]);
        // Pages 0x85 and 0x86 read as zeros, then pages 0 and 1 again.
        let read = tag.command(&[READ, 0x85]).unwrap();
        assert_eq!(read[..8], [0; 8]);
        assert_eq!(read[8..12], tag.dump()[..4]);
        assert_eq!(tag.command(&[READ, 0x87]), Err(Nak::InvalidArgument));
        let all = tag.command(&[FAST_READ, 0, LAST_PAGE]).unwrap();
        assert_eq!(all.len(), SIZE);
        assert_eq!(all[..0x84 * 4], tag.dump()[..0x84 * 4]);
        assert_eq!(tag.command(&[FAST_READ, 3, 2]), Err(Nak::InvalidArgument));
        assert_eq!(tag.command(&[READ_SIG, 0]).unwrap(), [0; 32]);
    }

    #[test]
    fn writes_respect_lock_bits_one_time_bits_and_the_password() {
        let mut tag = blank(UID);
        // The serial number is read-only; user memory is not.
        assert_eq!(
            tag.command(&[WRITE, 0, 1, 2, 3, 4]),
            Err(Nak::InvalidArgument)
        );
        tag.command(&[WRITE, 4, 1, 2, 3, 4]).unwrap();
        assert_eq!(tag.command(&[READ, 4]).unwrap()[..4], [1, 2, 3, 4]);
        // The capability container only gains bits.
        tag.command(&[WRITE, 3, 0, 0, 0, 0x0F]).unwrap();
        assert_eq!(tag.dump()[12..16], [0xE1, 0x10, 0x3E, 0x0F]);
        // Locking pages 13 to 15 and the first dynamic group.
        tag.command(&[WRITE, 2, 0xAA, 0xAA, 0x00, 0xE0]).unwrap();
        assert_eq!(tag.dump()[8..12], [0x33 ^ 0x44 ^ 0x55 ^ 0x66, 0, 0, 0xE0]);
        assert_eq!(
            tag.command(&[WRITE, 13, 0, 0, 0, 0]),
            Err(Nak::InvalidArgument)
        );
        tag.command(&[WRITE, 12, 9, 9, 9, 9]).unwrap();
        tag.command(&[WRITE, 0x82, 0x01, 0, 0, 0]).unwrap();
        assert_eq!(
            tag.command(&[WRITE, 0x1F, 0, 0, 0, 0]),
            Err(Nak::InvalidArgument)
        );
        tag.command(&[WRITE, 0x20, 1, 1, 1, 1]).unwrap();
        // A password over pages from 4 on.
        tag.command(&[WRITE, 0x85, 0x12, 0x34, 0x56, 0x78]).unwrap();
        tag.command(&[WRITE, 0x86, 0x80, 0x80, 0, 0]).unwrap();
        tag.command(&[WRITE, 0x83, 0x04, 0, 0, 0x04]).unwrap();
        assert_eq!(
            tag.command(&[WRITE, 4, 0, 0, 0, 0]),
            Err(Nak::InvalidArgument)
        );
        assert_eq!(
            tag.command(&[PWD_AUTH, 0, 0, 0, 0]),
            Err(Nak::AuthenticationFailed)
        );
        assert_eq!(
            tag.command(&[PWD_AUTH, 0x12, 0x34, 0x56, 0x78]).unwrap(),
            [0x80, 0x80]
        );
        tag.command(&[WRITE, 4, 0, 0, 0, 0]).unwrap();
    }
}
