//! The boundary between the Tegra210 PCIe root complex and an endpoint
//! function behind its root port.
//!
//! The root complex (`pcie.rs`) forwards type-1 configuration requests for
//! bus 1, device 0, function 0 and memory requests whose bus address falls
//! in the AFI's outbound windows to a [`PcieFunction`]. The function masters
//! the bus through an [`Upstream`]: memory reads and writes at bus
//! addresses, which the root complex translates (AFI inbound windows, then
//! the AFI's SMMU client); a message-signalled interrupt is one such write,
//! to the address and data the host programmed in the function's MSI
//! capability.

/// An endpoint function: its configuration space and the memory its BARs
/// decode.
pub trait PcieFunction: Send + Sync {
    /// Reads `size` (1, 2 or 4) bytes of configuration space at `offset`
    /// (below 0x1000, aligned to `size`). `None` for a register the model
    /// does not know: the access is refused, never answered with a guess.
    fn config_read(&self, offset: u16, size: u8) -> Option<u32>;
    /// Writes the low `size` bytes of `value` at `offset`.
    fn config_write(&self, offset: u16, size: u8, value: u32) -> Option<()>;
    /// A memory read request at PCI bus address `address` routed
    /// downstream. `None` if no enabled BAR claims it or the register is not
    /// modeled.
    fn memory_read(&self, address: u64, size: u8) -> Option<u64>;
    /// A memory write request at PCI bus address `address`.
    fn memory_write(&self, address: u64, size: u8, value: u64) -> Option<()>;
}

/// The root complex as the function sees it from below.
pub trait Upstream: Send + Sync {
    /// Reads host memory at PCI bus address `address`. `None` if the root
    /// complex does not translate (part of) the range.
    fn read(&self, address: u64, buffer: &mut [u8]) -> Option<()>;
    /// Writes host memory at PCI bus address `address`; a write into the
    /// root complex's MSI window signals that message instead.
    fn write(&self, address: u64, data: &[u8]) -> Option<()>;
    /// Level of the function's legacy INTA# line.
    fn intx(&self, level: bool);
}
