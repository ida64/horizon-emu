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

use std::{borrow::Cow, collections::HashMap, fs::File, io, os::unix::fs::FileExt};

use serde::{Deserialize, Serialize};

use crate::snapshot::SnapshotError;

pub const SECTOR_SIZE: usize = 512;

enum Backing {
    /// Nothing but erased sectors.
    Erased,
    File {
        file: File,
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
    pub fn new(device: &'static str, sectors: u64, image: Option<File>, writable: bool) -> Self {
        let backing = match image {
            Some(file) => Backing::File {
                file,
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

    /// `None` for a sector beyond the end or a block that is not one
    /// sector.
    pub fn read(&self, sector: u64, block: &mut [u8]) -> Option<()> {
        if sector >= self.sectors || block.len() != SECTOR_SIZE {
            return None;
        }
        if let Some(sector) = self
            .written
            .get(&sector)
            .or_else(|| self.overlay.get(&sector))
        {
            block.copy_from_slice(&sector[..]);
            return Some(());
        }
        let offset = sector * SECTOR_SIZE as u64;
        block.fill(0);
        match &self.backing {
            Backing::Erased => {}
            Backing::Memory(memory) => {
                block.copy_from_slice(&memory[offset as usize..][..SECTOR_SIZE]);
            }
            Backing::File { file, .. } => {
                // A short image leaves the rest erased.
                let mut filled = 0;
                while filled < SECTOR_SIZE {
                    match file.read_at(&mut block[filled..], offset + filled as u64) {
                        Ok(0) => break,
                        Ok(count) => filled += count,
                        Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                        Err(error) => panic!("cannot read the {} image: {error}", self.device),
                    }
                }
            }
        }
        Some(())
    }

    pub fn write(&mut self, sector: u64, block: &[u8]) -> Option<()> {
        let block: &[u8; SECTOR_SIZE] = block.try_into().ok()?;
        (sector < self.sectors).then_some(())?;
        self.overlay.remove(&sector);
        let offset = sector * SECTOR_SIZE as u64;
        match &mut self.backing {
            Backing::Memory(memory) => {
                memory[offset as usize..][..SECTOR_SIZE].copy_from_slice(block);
            }
            Backing::File {
                file,
                write_through: true,
            } => file
                .write_all_at(block, offset)
                .unwrap_or_else(|error| panic!("cannot write the {} image: {error}", self.device)),
            _ => {
                self.written.insert(sector, Box::new(*block));
            }
        }
        Some(())
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
        let mut medium = Medium::new("test", 2, Some(file), true).with_overlay(overlay);
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
}
