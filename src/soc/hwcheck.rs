//! The NX bootloader's hardware-state check (`0x40002A7C`) against a
//! powered-on SoC: every register it reads, read from the bus the BPMP
//! sees, must match under the check's mask (`nxbl_hwcheck.csv`, the
//! research's extraction of the check; skipped where the file is absent).
//!
//! Before the check runs, package1ldr has run on the BPMP; of what it
//! writes, only its memory controller writes reach registers the check
//! reads (pkg1-research §3.1, `enable_mem_ctl` text `0xB04`), and those
//! are applied here. The boot ROM's handoff is not: rows of blocks whose
//! power-on state is another part's are reported per owner and expected to
//! fail until those parts are in.

use std::sync::Arc;

use super::Soc;
use crate::{
    boot::{self, BootParams, CORE_COUNT},
    bus::{Access, Bus},
    clock::GuestClock,
    gic::Gic,
    machine::BPMP_CORE,
    memory::GuestRam,
};

const CSV: &str = "/tmp/boot/pk11/nxbl_hwcheck.csv";

/// Who owns the power-on state of the block a row reads.
fn owner(device: &str, address: u64) -> &'static str {
    match device {
        "CAR" | "PMC" | "FLOW_CTLR" | "ICTLR" => "BpmpMachine",
        "AHB_ARB/GIZMO" if address >= super::sb::BASE => "BpmpMachine",
        "FUSE" => "BootRom",
        "SE" => "SecurityEngine",
        _ => "HwCheck",
    }
}

fn number(text: &str) -> u32 {
    u32::from_str_radix(text.trim().trim_start_matches("0x"), 16).expect("hex in the CSV")
}

#[test]
fn bootloader_hardware_state_check() {
    let Ok(csv) = std::fs::read_to_string(CSV) else {
        eprintln!("{CSV} absent: skipped");
        return;
    };
    let params = BootParams::default();
    let gic = Arc::new(Gic::new(CORE_COUNT, Box::new(|_| {})));
    let size = usize::try_from(params.dram.bytes()).unwrap();
    let ram = Arc::new(GuestRam::new(boot::DRAM_BASE, size).unwrap());
    let clock = Arc::new(GuestClock::new());
    let mut bus = Bus::default();
    let fuses = crate::bootrom::Fuses::retail([0; 16], 0, crate::bootrom::ChipFuses::default());
    let start = super::Start::AfterRom { fuses: &fuses };
    let _soc = Soc::new(&params, &gic, &ram, &clock, None, None, start, &mut bus).unwrap();

    // package1ldr's enable_mem_ctl: MC_IRAM_REG_CTRL &= ~1, then the AHB
    // redirect aperture.
    let mc = super::mc::BASE;
    let Access::Done(control) = bus.read(BPMP_CORE, mc + 0x964, 4) else {
        panic!("MC_IRAM_REG_CTRL unreadable");
    };
    for (register, value) in [
        (0x964, control & !1),
        (0x65C, 0x4000_0000),
        (0x660, 0x8000_0000),
    ] {
        assert_eq!(
            bus.write(BPMP_CORE, mc + register, 4, value),
            Access::Done(())
        );
    }

    let mut failures: Vec<(&str, String)> = Vec::new();
    let mut checked = std::collections::BTreeMap::<&str, usize>::new();
    for line in csv.lines() {
        let fields: Vec<&str> = line.split(',').collect();
        let [device, address, mask, expected] = fields[..] else {
            continue;
        };
        if !address.starts_with("0x") {
            continue;
        }
        let address = u64::from(number(address));
        let mask = number(mask);
        let accepted: Vec<u32> = expected
            .split('|')
            .map(|value| number(value) & mask)
            .collect();
        let owner = owner(device, address);
        *checked.entry(owner).or_default() += 1;
        let failure = match bus.read(BPMP_CORE, address, 4) {
            Access::Done(value) if accepted.contains(&(value as u32 & mask)) => continue,
            Access::Done(value) => format!("reads {value:#010x}"),
            other => format!("{other:?}"),
        };
        failures.push((
            owner,
            format!("{device} {address:#010x} mask {mask:#010x} expected {expected}: {failure}"),
        ));
    }
    for (owner, count) in &checked {
        let failed: Vec<&String> = failures
            .iter()
            .filter(|(who, _)| who == owner)
            .map(|(_, failure)| failure)
            .collect();
        eprintln!("{owner}: {} of {count} rows fail", failed.len());
        if *owner != "HwCheck" {
            for failure in failed {
                eprintln!("  expected to fail: {failure}");
            }
        }
    }
    let mine: Vec<&String> = failures
        .iter()
        .filter(|(owner, _)| *owner == "HwCheck")
        .map(|(_, failure)| failure)
        .collect();
    assert!(mine.is_empty(), "{} rows fail:\n{mine:#?}", mine.len());
    assert_eq!(checked.values().sum::<usize>(), 2896);
}
