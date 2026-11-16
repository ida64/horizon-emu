//! Guest-physical address space outside DRAM: memory-mapped devices.

use std::{ops::Range, sync::Arc};

use crate::snapshot::{Snapshot, SnapshotError};

/// A memory-mapped device. Accesses arrive from the core threads, so devices
/// synchronize internally.
///
/// Both methods return `None` for a register or access width the device does
/// not model; the machine then stops rather than inventing a value.
pub trait MmioDevice: Send + Sync {
    /// Reads `size` bytes (1, 2, 4 or 8) at `offset` from the device's base.
    fn read(&self, core: usize, offset: u64, size: u8) -> Option<u64>;
    /// Writes the low `size` bytes of `value` at `offset`.
    fn write(&self, core: usize, offset: u64, size: u8, value: u64) -> Option<()>;
    /// Saves and restores the device's state; `None` until the device can,
    /// which makes a snapshot of the machine fail. A device without state,
    /// or whose state another part saves, answers [`NoState`].
    fn snapshot(&self) -> Option<&dyn Snapshot> {
        None
    }
}

impl<T: MmioDevice + ?Sized> MmioDevice for Arc<T> {
    fn read(&self, core: usize, offset: u64, size: u8) -> Option<u64> {
        (**self).read(core, offset, size)
    }

    fn write(&self, core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        (**self).write(core, offset, size, value)
    }

    fn snapshot(&self) -> Option<&dyn Snapshot> {
        (**self).snapshot()
    }
}

/// The [`Snapshot`] of a device with no state of its own, or one whose
/// state another part of the machine saves.
pub struct NoState;

impl Snapshot for NoState {
    fn save(&self) -> Result<Vec<u8>, SnapshotError> {
        Ok(Vec::new())
    }

    fn restore(&self, state: &[u8]) -> Result<(), SnapshotError> {
        if state.is_empty() {
            Ok(())
        } else {
            Err(SnapshotError::Corrupt(
                "state for a device that has none".into(),
            ))
        }
    }
}

/// Outcome of routing one access.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access<T> {
    Done(T),
    /// No device is mapped at the address.
    Unmapped,
    /// The named device does not model this register or width.
    Unimplemented(&'static str),
}

struct Region {
    range: Range<u64>,
    name: &'static str,
    device: Box<dyn MmioDevice>,
}

#[derive(Default)]
pub struct Bus {
    regions: Vec<Region>,
}

impl Bus {
    /// Maps `device` at `base..base + size`.
    ///
    /// # Panics
    /// If the range overlaps an existing mapping; the memory map is fixed at
    /// build time, so that is a programming error.
    pub fn map(&mut self, name: &'static str, base: u64, size: u64, device: Box<dyn MmioDevice>) {
        let range = base..base + size;
        assert!(
            self.regions
                .iter()
                .all(|region| region.range.end <= range.start || range.end <= region.range.start),
            "{name} overlaps another device"
        );
        self.regions.push(Region {
            range,
            name,
            device,
        });
    }

    pub fn read(&self, core: usize, address: u64, size: u8) -> Access<u64> {
        self.route(address, size, |device, offset| {
            device.read(core, offset, size)
        })
    }

    pub fn write(&self, core: usize, address: u64, size: u8, value: u64) -> Access<()> {
        self.route(address, size, |device, offset| {
            device.write(core, offset, size, value)
        })
    }

    /// Name of the device mapped at `address`.
    pub fn device_at(&self, address: u64) -> Option<&'static str> {
        self.regions
            .iter()
            .find(|region| region.range.contains(&address))
            .map(|region| region.name)
    }

    /// Every mapped device with the key its state is saved under: its
    /// name and base address (a device type may be mapped more than once).
    pub fn devices(&self) -> impl Iterator<Item = (String, &dyn MmioDevice)> {
        self.regions.iter().map(|region| {
            (
                format!("{}@{:#x}", region.name, region.range.start),
                region.device.as_ref(),
            )
        })
    }

    fn route<T>(
        &self,
        address: u64,
        size: u8,
        access: impl FnOnce(&dyn MmioDevice, u64) -> Option<T>,
    ) -> Access<T> {
        let Some(region) = self.regions.iter().find(|region| {
            region.range.contains(&address) && address + u64::from(size) <= region.range.end
        }) else {
            return Access::Unmapped;
        };
        match access(region.device.as_ref(), address - region.range.start) {
            Some(value) => Access::Done(value),
            None => Access::Unimplemented(region.name),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Models only 4-byte accesses below offset 8; reads return the offset.
    struct Probe;

    impl MmioDevice for Probe {
        fn read(&self, core: usize, offset: u64, size: u8) -> Option<u64> {
            (size == 4 && offset < 8).then_some(offset | (core as u64) << 32)
        }

        fn write(&self, _core: usize, offset: u64, size: u8, _value: u64) -> Option<()> {
            (size == 4 && offset < 8).then_some(())
        }
    }

    #[test]
    fn routes_by_address_and_keeps_missing_devices_apart_from_missing_registers() {
        let mut bus = Bus::default();
        bus.map("probe", 0x5004_1000, 0x10, Box::new(Probe));

        assert_eq!(bus.read(2, 0x5004_1004, 4), Access::Done(4 | 2 << 32));
        assert_eq!(bus.write(0, 0x5004_1000, 4, 1), Access::Done(()));
        assert_eq!(bus.read(0, 0x5004_1008, 4), Access::Unimplemented("probe"));
        assert_eq!(
            bus.write(0, 0x5004_1000, 1, 1),
            Access::Unimplemented("probe")
        );
        assert_eq!(bus.read(0, 0x5004_0FFC, 4), Access::Unmapped);
        assert_eq!(bus.read(0, 0x5004_1010, 4), Access::Unmapped);
        // An access running past the end of the mapping reaches no device.
        assert_eq!(bus.read(0, 0x5004_100E, 4), Access::Unmapped);
    }

    #[test]
    #[should_panic(expected = "overlaps")]
    fn overlapping_mappings_are_rejected() {
        let mut bus = Bus::default();
        bus.map("first", 0x1000, 0x1000, Box::new(Probe));
        bus.map("second", 0x1FFF, 0x10, Box::new(Probe));
    }
}
