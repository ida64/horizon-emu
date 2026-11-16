//! Peripherals only the secure world may reach.
//!
//! The secure monitor marks some of the SoC's peripherals secure-only when
//! it starts, through the APB slave security registers of `APB_MISC`
//! (exosphere `secmon_setup.cpp`, `ConfigureSlaveSecurity`, for firmware
//! after 1.0.0 on this board): the security engine, QSPI, SATA and its
//! auxiliary block, SPI1 to SPI3, I2C6 and SDMMC3, among others. Memory
//! the secure world keeps for itself is likewise out of reach: TZRAM, where
//! the monitor lives. The kernel and every process run in the non-secure
//! world, so an access of theirs to any of these is refused by the bus and
//! comes back to the core as an external abort: see
//! [`crate::machine`], which delivers it to the guest's own exception
//! vectors, where the kernel treats it as it would on the console (a
//! process crashes, the kernel itself panics).
//!
//! Addresses are those of Linux's `tegra210.dtsi` and, for the security
//! engine and TZRAM, Atmosphère's memory layout
//! (`secmon_memory_layout.hpp`). Blocks the monitor marks secure whose
//! addresses neither gives (DTV, LA, DDS, DP2 and the SPI5 and SPI6 bits)
//! are not listed; an access to them stops the machine as any unmodeled
//! access does. Exosphere's condition for making I2C4 secure can never
//! hold (it asks for three hardware types at once), so I2C4 stays
//! non-secure, as on the console \[inference: Nintendo's monitor has the
//! same condition\].

/// `(base, size, name)` of each secure-only region.
const SECURE_ONLY: [(u64, u64, &str); 10] = [
    (0x7000_1100, 0x1000, "SATA AUX"),
    (0x7000_D100, 0x100, "I2C6"),
    (0x7000_D400, 0x200, "SPI1"),
    (0x7000_D600, 0x200, "SPI2"),
    (0x7000_D800, 0x200, "SPI3"),
    (0x7001_2000, 0x2000, "security engine"),
    (0x7002_0000, 0x9000, "SATA"),
    (0x700B_0400, 0x200, "SDMMC3"),
    (0x7041_0000, 0x1000, "QSPI"),
    (0x7C01_0000, 0x1_0000, "TZRAM"),
];

/// The secure-only region `address` lies in, by name, if any.
pub fn secure_only(address: u64) -> Option<&'static str> {
    SECURE_ONLY
        .iter()
        .find(|&&(base, size, _)| (base..base + size).contains(&address))
        .map(|&(_, _, name)| name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_security_engine_and_tzram_are_secure_but_their_neighbours_are_not() {
        assert_eq!(secure_only(0x7001_2000), Some("security engine"));
        assert_eq!(secure_only(0x7001_3FFC), Some("security engine"));
        assert_eq!(secure_only(0x7001_4000), None);
        assert_eq!(secure_only(0x7C01_FFFC), Some("TZRAM"));
        assert_eq!(secure_only(0x7000_D000), None, "I2C5, the PMIC's bus");
    }
}
