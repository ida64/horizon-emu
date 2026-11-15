//! What a memory card stores: 512-byte sectors over an image file or a
//! buffer in host memory, for the eMMC module and the microSD card alike.
//!
//! A file image is left as it is unless it was opened as writable: the
//! sectors written during a run are kept in host memory, laid over it, so
//! every run starts from the same contents. Whatever an image does not
//! cover reads as zeros, the erased state. Contents handed over in memory
//! belong to the run and are written in place.

use std::{collections::HashMap, fs::File, io, os::unix::fs::FileExt};

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
    /// Names the device in the message of a failing host file.
    device: &'static str,
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
            device,
        }
    }

    /// Contents held in `memory`, whose whole sectors are the medium.
    pub fn in_memory(device: &'static str, memory: Vec<u8>) -> Self {
        Self {
            sectors: (memory.len() / SECTOR_SIZE) as u64,
            backing: Backing::Memory(memory),
            written: HashMap::new(),
            device,
        }
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
        if let Some(written) = self.written.get(&sector) {
            block.copy_from_slice(&written[..]);
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
}
