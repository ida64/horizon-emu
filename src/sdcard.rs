//! microSD card images: a partition table and a FAT32 volume laid out the
//! way the console formats a card, with files on it.
//!
//! The filesystem process formats a card (`FS08_FormatFatStorage`, FS
//! text `0x118740`, called with the card's user area and protected area
//! sizes from `0x946c0`) after the SD Association's file system
//! specification: for an SDHC card a master boot record whose one entry
//! starts one boundary unit (4 MiB) into the card and spans the rest, type
//! 0x0B below 8032.5 MiB and 0x0C above (`FS08_WriteMasterBootRecord`,
//! `0x119380`), and in it a FAT32 volume of 32 KiB clusters whose data
//! area starts on a boundary unit ([`Layout::for_sd_card`]). [`format()`]
//! makes the same; the volume serial number, which the console takes from
//! its clock, is [`VOLUME_SERIAL`] here. The card the emulator presents has
//! no protected area, so the user area is the whole card.
//!
//! The FAT driver mounts what it finds in the first partition entry of a
//! sector 0 that ends in 0x55AA, with a boot flag of 0x00 or 0x80, a start
//! above zero and an end within the card, and a type other than an
//! extended or GPT protective one (`FS07_Mbr_ValidatePartitionBounds`,
//! `0xf6460`; `FS07_Mbr_ReadPartitionEntry`, `0xf6940`). This build of the
//! filesystem process has no exFAT (its formatter refuses to make one), so
//! cards above 32 GiB, which the specification gives exFAT, are not made
//! here.

use std::{
    fs::File,
    io::{self, Read},
};

use crate::{
    nand::fat::{self, Fat32, FatError, Layout, SECTOR_SIZE, Sectors},
    soc::sdcard,
};

/// Volume serial number of the volumes [`format()`] makes.
pub const VOLUME_SERIAL: u32 = 0x5344_0001;

/// Partition types for FAT32 by how the partition is addressed: by
/// cylinder, head and sector, or only by block number, for partitions
/// that end beyond where the former reaches.
const TYPE_FAT32: u8 = 0x0B;
const TYPE_FAT32_LBA: u8 = 0x0C;
/// Bytes a cylinder, head and sector address reaches: 1024 cylinders of
/// 255 heads of 63 sectors, less one sector. The formatter writes larger
/// addresses as the largest ones.
const CHS_LIMIT: u64 = 0x1_F607_FFFF;
const PARTITION_TABLE: usize = 446;

/// The cylinder, head and sector address of `sector`, encoded as in a
/// partition entry, or `beyond` past [`CHS_LIMIT`].
fn chs(sector: u64, (heads, sectors_per_track): (u8, u8), beyond: [u8; 3]) -> [u8; 3] {
    if sector * SECTOR_SIZE as u64 > CHS_LIMIT {
        return beyond;
    }
    let (heads, sectors_per_track) = (u64::from(heads), u64::from(sectors_per_track));
    let cylinder = sector / (heads * sectors_per_track);
    let head = sector % (heads * sectors_per_track) / sectors_per_track;
    let sector = sector % sectors_per_track + 1;
    [
        head as u8,
        sector as u8 | (cylinder >> 2) as u8 & 0xC0,
        cylinder as u8,
    ]
}

/// The master boot record for `layout` on a card of `card_sectors`
/// sectors.
fn master_boot_record(layout: &Layout, card_sectors: u64) -> [u8; SECTOR_SIZE] {
    let start = u64::from(layout.hidden_sectors);
    let geometry = layout.geometry();
    let mut sector = [0; SECTOR_SIZE];
    let entry = &mut sector[PARTITION_TABLE..PARTITION_TABLE + 16];
    entry[1..4].copy_from_slice(&chs(start, geometry, [0xFF, 0xFF, 0xFF]));
    entry[4] = if card_sectors * (SECTOR_SIZE as u64) < CHS_LIMIT + 1 {
        TYPE_FAT32
    } else {
        TYPE_FAT32_LBA
    };
    entry[5..8].copy_from_slice(&chs(card_sectors - 1, geometry, [0xFE, 0xFF, 0xFF]));
    entry[8..12].copy_from_slice(&layout.hidden_sectors.to_le_bytes());
    entry[12..16].copy_from_slice(&layout.total_sectors.to_le_bytes());
    sector[510..].copy_from_slice(&[0x55, 0xAA]);
    sector
}

/// The sectors of a partition, numbered from its start.
pub struct Partition<S> {
    storage: S,
    start: u64,
    sectors: u64,
}

impl<S: Sectors> Partition<S> {
    fn locate(&self, sector: u64, length: usize) -> io::Result<u64> {
        let count = length.div_ceil(SECTOR_SIZE) as u64;
        sector
            .checked_add(count)
            .filter(|&end| end <= self.sectors)
            .map(|_| self.start + sector)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "beyond the partition"))
    }
}

impl<S: Sectors> Sectors for Partition<S> {
    fn read(&mut self, sector: u64, data: &mut [u8]) -> io::Result<()> {
        let sector = self.locate(sector, data.len())?;
        self.storage.read(sector, data)
    }

    fn write(&mut self, sector: u64, data: &[u8]) -> io::Result<()> {
        let sector = self.locate(sector, data.len())?;
        self.storage.write(sector, data)
    }
}

/// Makes the card of `card_sectors` sectors in `storage` an empty one, as
/// the console formats it.
pub fn format(storage: &mut impl Sectors, card_sectors: u64) -> Result<Layout, FatError> {
    let layout = Layout::for_sd_card(card_sectors)?;
    storage.write(0, &master_boot_record(&layout, card_sectors))?;
    let mut volume = Partition {
        storage,
        start: layout.hidden_sectors.into(),
        sectors: layout.total_sectors.into(),
    };
    fat::format_volume(&mut volume, &layout, VOLUME_SERIAL)?;
    Ok(layout)
}

/// The volume in the first partition of the card in `storage`, for
/// adding directories and files to; they are stored once
/// [`Fat32::finish`] returns.
pub fn open_volume<S: Sectors>(mut storage: S) -> Result<Fat32<Partition<S>>, FatError> {
    let mut sector = [0; SECTOR_SIZE];
    storage.read(0, &mut sector)?;
    if sector[510..] != [0x55, 0xAA] {
        return Err(FatError::NotFat32("no partition table"));
    }
    let entry = &sector[PARTITION_TABLE..PARTITION_TABLE + 16];
    let word = |at: usize| u32::from_le_bytes(entry[at..at + 4].try_into().unwrap());
    let (start, sectors) = (word(8), word(12));
    if start == 0 || sectors == 0 {
        return Err(FatError::NotFat32("no first partition"));
    }
    Fat32::open(Partition {
        storage,
        start: start.into(),
        sectors: sectors.into(),
    })
}

/// Adds the file `path` (absolute, `/` separated), with the directories
/// above it that are missing, of the `size` bytes `contents` yields.
pub fn add_file<S: Sectors>(
    volume: &mut Fat32<S>,
    path: &str,
    contents: &mut dyn Read,
    size: u64,
) -> Result<(), FatError> {
    if let Some((directory, _)) = path.rsplit_once('/') {
        volume.create_dir_all(directory)?;
    }
    volume.write_file(path, contents, size)
}

/// A card image of `size` bytes, formatted as the console formats a card,
/// with `files` at the absolute paths given (`/hbmenu.nro`,
/// `/switch/foo.nro`) and the directories they need. A path that ends in
/// `/` (`/switch/`) is a directory to make; its contents are ignored.
/// `size` is that of an SDHC card: a multiple of 512 KiB above
/// 0x8287_FFFF bytes and up to
/// 32 GiB. The buffer is zero-filled by the allocator on demand, so what
/// is never written takes no host memory.
pub fn image_with_files(size: u64, files: &[(&str, &[u8])]) -> Result<Vec<u8>, String> {
    let sectors = sdcard::sectors(size).map_err(|error| error.to_string())?;
    let failed = |path: &str, error: FatError| format!("{path}: {error}");
    // Refused sizes are refused before the memory is asked for.
    Layout::for_sd_card(sectors).map_err(|error| failed("format", error))?;
    let length = usize::try_from(size).map_err(|_| format!("{size} bytes do not fit in memory"))?;
    let mut image = vec![0; length];
    format(&mut image, sectors).map_err(|error| failed("format", error))?;
    let mut volume = open_volume(&mut image).map_err(|error| failed("open", error))?;
    for (path, contents) in files {
        let added = if path.ends_with('/') {
            volume.create_dir_all(path)
        } else {
            add_file(&mut volume, path, &mut &contents[..], contents.len() as u64)
        };
        added.map_err(|error| failed(path, error))?;
    }
    volume.finish().map_err(|error| failed("finish", error))?;
    Ok(image)
}

/// Makes `file` a card of `size` bytes, as [`image_with_files`] does, with
/// the files `files` names: host path, then path on the card. The file
/// stays sparse where nothing is written.
pub fn format_file(file: &mut File, size: u64, files: &[(String, String)]) -> Result<(), String> {
    let sectors = sdcard::sectors(size).map_err(|error| error.to_string())?;
    file.set_len(size).map_err(|error| error.to_string())?;
    format(file, sectors).map_err(|error| format!("format: {error}"))?;
    let mut volume = open_volume(file).map_err(|error| format!("open: {error}"))?;
    for (host, path) in files {
        let mut contents = File::open(host).map_err(|error| format!("{host}: {error}"))?;
        let size = contents
            .metadata()
            .map_err(|error| format!("{host}: {error}"))?
            .len();
        add_file(&mut volume, path, &mut contents, size)
            .map_err(|error| format!("{path}: {error}"))?;
    }
    volume
        .finish()
        .map_err(|error| format!("finish: {error}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nand::fat::reader::Reader;

    const CARD: u64 = 4 << 30;

    #[test]
    fn a_4_gib_card_is_laid_out_as_the_consoles_formatter_does_and_holds_the_files() {
        let nro = vec![0x4E; 100_000];
        let image =
            image_with_files(CARD, &[("/hbmenu.nro", &nro), ("/switch/a/b.txt", b"hi")]).unwrap();
        let sectors = CARD / SECTOR_SIZE as u64;

        // The partition table: one FAT32 entry from 4 MiB to the end.
        let mbr = &image[..SECTOR_SIZE];
        assert_eq!(mbr[510..], [0x55, 0xAA]);
        let entry = &mbr[PARTITION_TABLE..PARTITION_TABLE + 16];
        let word = |at: usize| u32::from_le_bytes(entry[at..at + 4].try_into().unwrap());
        assert_eq!(entry[0], 0, "not bootable");
        assert_eq!(entry[4], TYPE_FAT32, "below 8032.5 MiB");
        assert_eq!(u64::from(word(8)), 8192);
        assert_eq!(u64::from(word(12)), sectors - 8192);
        // Sector 8192 is head 130, sector 3 of cylinder 0 at 255/63.
        assert_eq!(entry[1..4], [130, 3, 0]);
        assert!(mbr[PARTITION_TABLE + 16..510].iter().all(|&byte| byte == 0));

        // The boot sector: 32 KiB clusters, 8192 hidden sectors, the
        // geometry, and a data area on a 4 MiB boundary of the card.
        let boot = &image[8192 * SECTOR_SIZE..][..SECTOR_SIZE];
        let half = |at: usize| u16::from_le_bytes([boot[at], boot[at + 1]]);
        let word = |at: usize| u32::from_le_bytes(boot[at..at + 4].try_into().unwrap());
        assert_eq!(boot[13], 64);
        assert_eq!(word(28), 8192);
        assert_eq!((half(24), half(26)), (63, 255));
        assert_eq!(u64::from(word(32)), sectors - 8192);
        let data_start = 8192 + u32::from(half(14)) + 2 * word(36);
        assert_eq!(data_start % 8192, 0);

        let mut reader = Reader::open(open_volume(image).unwrap().finish().unwrap());
        assert_eq!(reader.names("/"), ["hbmenu.nro", "switch"]);
        assert_eq!(reader.file("/hbmenu.nro"), nro);
        assert_eq!(reader.file("/switch/a/b.txt"), b"hi");
    }

    #[test]
    fn sizes_outside_the_fat32_range_are_refused() {
        // A valid SDHC capacity in the 40 MiB the formatter refuses.
        assert!(image_with_files(0x8280_0000, &[]).is_err());
        // Not a multiple of 512 KiB.
        assert!(image_with_files(CARD + 512, &[]).is_err());
        // SDXC: exFAT.
        assert!(image_with_files(64 << 30, &[]).is_err());
    }
}
