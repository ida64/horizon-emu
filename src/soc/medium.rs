//! What a memory card stores: 512-byte sectors over an image file or a
//! buffer in host memory, for the eMMC module and the microSD card alike.
//!
//! A file image is left as it is unless it was opened as writable: the
//! sectors written during a run are kept in host memory, laid over it, so
//! every run starts from the same contents. Whatever an image does not
//! cover reads as zeros, the erased state. Contents handed over in memory
//! belong to the run and are written in place.
//!
//! A machine snapshot saves what the run changed ([`MediumState`]): the
//! sectors kept in host memory, the overlay as it stands, and contents held
//! in memory. A file the run writes to is not saved here; the snapshot
//! keeps a copy of the file beside it, and the restored machine is built
//! over that copy.
//!
//! A host read or write of an image file that fails is no answer a card
//! gives: the medium hands back the error ([`MediumError::Host`]) for the
//! machine to stop on, naming the file.

use std::{borrow::Cow, collections::HashMap, fs::File, io, os::unix::fs::FileExt, path::PathBuf};

use serde::{Deserialize, Serialize};

use crate::snapshot::SnapshotError;

pub const SECTOR_SIZE: usize = 512;

/// An image file and the path it was opened from, which names it when a
/// host access to it fails.
pub struct ImageFile {
    pub file: File,
    pub path: PathBuf,
}

impl ImageFile {
    /// The error of a failed host access to `sector` of this file.
    fn failed(&self, device: &str, doing: &str, sector: u64, error: io::Error) -> MediumError {
        MediumError::Host(format!(
            "{device}: cannot {doing} sector {sector} of the image {}: {error}",
            self.path.display()
        ))
    }
}

/// Why a medium moved no sector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MediumError {
    /// A sector beyond the end, or a block that is not one sector: the
    /// card reports that to the guest.
    OutOfRange,
    /// The host could not read or write the image file; the message says
    /// which device, file and sector, and the host's error.
    Host(String),
}

enum Backing {
    /// Nothing but erased sectors.
    Erased,
    File {
        image: ImageFile,
        /// Whether writes go to the file.
        write_through: bool,
    },
    Memory(Vec<u8>),
}

pub struct Medium {
    sectors: u64,
    backing: Backing,
    /// Sectors written during this run over a file that is not written to.
    written: HashMap<u64, Box<[u8; SECTOR_SIZE]>>,
    /// Sectors read in place of the backing until the guest writes them;
    /// never written to the backing.
    overlay: HashMap<u64, Box<[u8; SECTOR_SIZE]>>,
    /// Names the device in the message of a failing host file.
    device: &'static str,
}

/// What a run has changed of a medium, saved in a machine snapshot.
/// Sectors are borrowed from the medium while it is saved.
#[derive(Serialize, Deserialize)]
pub struct MediumState<'a> {
    sectors: u64,
    backing: BackingState<'a>,
    /// Sectors in ascending order, each [`SECTOR_SIZE`] bytes.
    written: Vec<(u64, Cow<'a, [u8]>)>,
    overlay: Vec<(u64, Cow<'a, [u8]>)>,
}

#[derive(Serialize, Deserialize)]
enum BackingState<'a> {
    Erased,
    /// The file itself is not saved: the snapshot keeps a copy of it.
    File {
        write_through: bool,
    },
    /// Contents held in memory, as the runs of sectors that are not all
    /// zeros (first sector, bytes): a card made in memory is mostly
    /// erased, and gigabytes of zeros are not worth saving.
    Memory(Vec<(u64, Cow<'a, [u8]>)>),
}

/// The runs of whole sectors of `memory` that are not all zeros.
fn nonzero_runs(memory: &[u8]) -> Vec<(u64, Cow<'_, [u8]>)> {
    let mut runs = Vec::new();
    let mut first = None;
    let sectors = memory.len() / SECTOR_SIZE;
    for (index, sector) in memory.as_chunks::<SECTOR_SIZE>().0.iter().enumerate() {
        let erased = sector.iter().all(|&byte| byte == 0);
        match (erased, first) {
            (false, None) => first = Some(index),
            (true, Some(start)) => {
                runs.push((
                    start as u64,
                    Cow::Borrowed(&memory[start * SECTOR_SIZE..index * SECTOR_SIZE]),
                ));
                first = None;
            }
            _ => {}
        }
    }
    if let Some(start) = first {
        runs.push((
            start as u64,
            Cow::Borrowed(&memory[start * SECTOR_SIZE..sectors * SECTOR_SIZE]),
        ));
    }
    runs
}

impl Medium {
    /// `sectors` sectors over `image`, written to only if `writable`.
    pub fn new(
        device: &'static str,
        sectors: u64,
        image: Option<ImageFile>,
        writable: bool,
    ) -> Self {
        let backing = match image {
            Some(image) => Backing::File {
                image,
                write_through: writable,
            },
            None => Backing::Erased,
        };
        Self {
            sectors,
            backing,
            written: HashMap::new(),
            overlay: HashMap::new(),
            device,
        }
    }

    /// Contents held in `memory`, whose whole sectors are the medium.
    pub fn in_memory(device: &'static str, memory: Vec<u8>) -> Self {
        Self {
            sectors: (memory.len() / SECTOR_SIZE) as u64,
            backing: Backing::Memory(memory),
            written: HashMap::new(),
            overlay: HashMap::new(),
            device,
        }
    }

    /// Has `overlay` read in place of what the backing holds, as if this
    /// run had written those sectors before it started. The overlay itself
    /// never reaches the backing: a writable image keeps its own contents
    /// under it. A sector the guest writes is the guest's from then on, in
    /// the backing (or this run's memory) like any other write, so a file
    /// the guest rewrites, or clusters it reuses, are never masked.
    pub fn with_overlay(mut self, overlay: HashMap<u64, Box<[u8; SECTOR_SIZE]>>) -> Self {
        self.overlay = overlay;
        self
    }

    pub fn sectors(&self) -> u64 {
        self.sectors
    }

    /// Fills `block` with `sector`. [`MediumError::OutOfRange`] for a
    /// sector beyond the end or a block that is not one sector.
    pub fn read(&self, sector: u64, block: &mut [u8]) -> Result<(), MediumError> {
        if sector >= self.sectors || block.len() != SECTOR_SIZE {
            return Err(MediumError::OutOfRange);
        }
        if let Some(sector) = self
            .written
            .get(&sector)
            .or_else(|| self.overlay.get(&sector))
        {
            block.copy_from_slice(&sector[..]);
            return Ok(());
        }
        let offset = sector * SECTOR_SIZE as u64;
        block.fill(0);
        match &self.backing {
            Backing::Erased => {}
            Backing::Memory(memory) => {
                block.copy_from_slice(&memory[offset as usize..][..SECTOR_SIZE]);
            }
            Backing::File { image, .. } => {
                // A short image leaves the rest erased.
                let mut filled = 0;
                while filled < SECTOR_SIZE {
                    match image
                        .file
                        .read_at(&mut block[filled..], offset + filled as u64)
                    {
                        Ok(0) => break,
                        Ok(count) => filled += count,
                        Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                        Err(error) => {
                            return Err(image.failed(self.device, "read", sector, error));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Stores `block` as `sector`, under the same errors as [`Self::read`].
    pub fn write(&mut self, sector: u64, block: &[u8]) -> Result<(), MediumError> {
        let block: &[u8; SECTOR_SIZE] = block.try_into().map_err(|_| MediumError::OutOfRange)?;
        if sector >= self.sectors {
            return Err(MediumError::OutOfRange);
        }
        let offset = sector * SECTOR_SIZE as u64;
        match &mut self.backing {
            Backing::Memory(memory) => {
                memory[offset as usize..][..SECTOR_SIZE].copy_from_slice(block);
            }
            Backing::File {
                image,
                write_through: true,
            } => {
                if let Err(error) = image.file.write_all_at(block, offset) {
                    return Err(image.failed(self.device, "write", sector, error));
                }
            }
            _ => {
                self.written.insert(sector, Box::new(*block));
            }
        }
        self.overlay.remove(&sector);
        Ok(())
    }

    /// What the run has changed, borrowing the sectors.
    pub fn save(&self) -> MediumState<'_> {
        fn sorted(map: &HashMap<u64, Box<[u8; SECTOR_SIZE]>>) -> Vec<(u64, Cow<'_, [u8]>)> {
            let mut sectors: Vec<_> = map
                .iter()
                .map(|(&sector, bytes)| (sector, Cow::Borrowed(&bytes[..])))
                .collect();
            sectors.sort_unstable_by_key(|&(sector, _)| sector);
            sectors
        }
        let backing = match &self.backing {
            Backing::Erased => BackingState::Erased,
            Backing::File { write_through, .. } => BackingState::File {
                write_through: *write_through,
            },
            Backing::Memory(memory) => BackingState::Memory(nonzero_runs(memory)),
        };
        MediumState {
            sectors: self.sectors,
            backing,
            written: sorted(&self.written),
            overlay: sorted(&self.overlay),
        }
    }

    /// Takes back what [`Self::save`] of a medium built the same way
    /// returned: of the same size, over the same kind of backing.
    pub fn restore(&mut self, state: MediumState<'_>) -> Result<(), SnapshotError> {
        let device = self.device;
        let corrupt = |what: &str| SnapshotError::Corrupt(format!("{device} medium: {what}"));
        if state.sectors != self.sectors {
            return Err(corrupt(&format!(
                "{} sectors saved, {} here",
                state.sectors, self.sectors
            )));
        }
        let sectors = self.sectors;
        let sector = |(index, bytes): (u64, Cow<'_, [u8]>)| {
            let bytes: [u8; SECTOR_SIZE] = bytes[..]
                .try_into()
                .map_err(|_| corrupt("a saved sector is not a sector long"))?;
            if index >= sectors {
                return Err(corrupt("a saved sector beyond the end"));
            }
            Ok((index, Box::new(bytes)))
        };
        let written = state
            .written
            .into_iter()
            .map(sector)
            .collect::<Result<HashMap<_, _>, _>>()?;
        let overlay = state
            .overlay
            .into_iter()
            .map(sector)
            .collect::<Result<HashMap<_, _>, _>>()?;
        match (&mut self.backing, state.backing) {
            (Backing::Erased, BackingState::Erased) => {}
            (
                Backing::File { write_through, .. },
                BackingState::File {
                    write_through: saved,
                },
            ) if *write_through == saved => {}
            (Backing::Memory(memory), BackingState::Memory(runs)) => {
                let fits = |(first, bytes): &(u64, Cow<'_, [u8]>)| {
                    bytes.len().is_multiple_of(SECTOR_SIZE)
                        && usize::try_from(*first)
                            .ok()
                            .and_then(|first| first.checked_mul(SECTOR_SIZE))
                            .and_then(|start| start.checked_add(bytes.len()))
                            .is_some_and(|end| end <= memory.len())
                };
                if !runs.iter().all(fits) {
                    return Err(corrupt("saved contents beyond the end"));
                }
                memory.fill(0);
                for (first, bytes) in runs {
                    let start = first as usize * SECTOR_SIZE;
                    memory[start..start + bytes.len()].copy_from_slice(&bytes);
                }
            }
            _ => return Err(corrupt("saved over another kind of image")),
        }
        self.written = written;
        self.overlay = overlay;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_overlay_over_a_writable_image_is_read_but_never_written_and_yields_to_the_guest() {
        let path =
            std::env::temp_dir().join(format!("horizon-emu-medium-test-{}", std::process::id()));
        std::fs::write(&path, [1u8; 2 * SECTOR_SIZE]).unwrap();
        let file = File::options().read(true).write(true).open(&path).unwrap();
        let overlay = HashMap::from([(0, Box::new([7u8; SECTOR_SIZE]))]);
        let image = ImageFile {
            file,
            path: path.clone(),
        };
        let mut medium = Medium::new("test", 2, Some(image), true).with_overlay(overlay);
        let mut block = [0; SECTOR_SIZE];

        medium.read(0, &mut block).unwrap();
        assert_eq!(
            block, [7; SECTOR_SIZE],
            "the overlay is what the guest reads"
        );
        medium.write(1, &[2; SECTOR_SIZE]).unwrap();
        assert_eq!(
            std::fs::read(&path).unwrap()[..SECTOR_SIZE],
            [1; SECTOR_SIZE]
        );
        assert_eq!(
            std::fs::read(&path).unwrap()[SECTOR_SIZE..],
            [2; SECTOR_SIZE]
        );

        // Once the guest writes an overlaid sector, that write is the sector,
        // in the image as well.
        medium.write(0, &[3; SECTOR_SIZE]).unwrap();
        medium.read(0, &mut block).unwrap();
        assert_eq!(block, [3; SECTOR_SIZE]);
        assert_eq!(
            std::fs::read(&path).unwrap()[..SECTOR_SIZE],
            [3; SECTOR_SIZE]
        );
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn a_failing_host_file_is_an_error_naming_it_not_a_panic() {
        let path =
            std::env::temp_dir().join(format!("horizon-emu-medium-failing-{}", std::process::id()));
        std::fs::write(&path, [1u8; 2 * SECTOR_SIZE]).unwrap();
        let opened = |read: bool, write: bool| ImageFile {
            file: File::options().read(read).write(write).open(&path).unwrap(),
            path: path.clone(),
        };
        let mut block = [0; SECTOR_SIZE];

        // Reading a file open only for writing fails on the host.
        let medium = Medium::new("eMMC", 2, Some(opened(false, true)), false);
        let Err(MediumError::Host(note)) = medium.read(1, &mut block) else {
            panic!("the read did not fail");
        };
        let name = path.display().to_string();
        assert!(note.starts_with("eMMC: cannot read sector 1 of the image "));
        assert!(note.contains(&name), "{note}");
        assert_eq!(
            medium.read(2, &mut block),
            Err(MediumError::OutOfRange),
            "beyond the end is the card's to report"
        );

        // Writing through to a file open only for reading fails too, and
        // leaves an overlaid sector as it was.
        let mut medium = Medium::new("SD card", 2, Some(opened(true, false)), true)
            .with_overlay(HashMap::from([(0, Box::new([7u8; SECTOR_SIZE]))]));
        let Err(MediumError::Host(note)) = medium.write(0, &[2; SECTOR_SIZE]) else {
            panic!("the write did not fail");
        };
        assert!(note.starts_with("SD card: cannot write sector 0 of the image "));
        assert!(note.contains(&name), "{note}");
        medium.read(0, &mut block).unwrap();
        assert_eq!(block, [7; SECTOR_SIZE]);
        assert_eq!(std::fs::read(&path).unwrap(), [1; 2 * SECTOR_SIZE]);
        std::fs::remove_file(&path).unwrap();
    }
}
