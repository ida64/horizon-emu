//! Throughput of the interpreter on the loops boot firmware spends its
//! time in. Run with
//! `cargo test --release --lib bpmp::bench -- --ignored --nocapture`;
//! `BPMP_BENCH=delay` (or `memcpy`, `bytes`) runs one workload alone.

use std::time::Instant;

use super::*;
use crate::soc::iram::{self, Iram};

/// IRAM as the machine reaches it: a device behind a mutex, one lock per
/// access (or per block transfer).
struct IramBus(Iram);

const BASE: u32 = iram::BASE as u32;
const CODE: u32 = BASE + 0x1000;

impl IramBus {
    /// Word by word, for a block the IRAM does not hold whole.
    fn words_one_by_one<T>(
        &mut self,
        address: u32,
        count: usize,
        mut each: impl FnMut(&mut Self, usize, u32) -> Result<T, BusError>,
    ) -> Result<(), (usize, BusError)> {
        for i in 0..count {
            each(self, i, address.wrapping_add(4 * i as u32)).map_err(|error| (i, error))?;
        }
        Ok(())
    }
}

impl BpmpBus for IramBus {
    fn read(&mut self, address: u32, size: u8) -> Result<u32, BusError> {
        let offset = address.wrapping_sub(BASE);
        self.0.read(offset, size).ok_or(BusError::Abort)
    }

    fn write(&mut self, address: u32, size: u8, value: u32) -> Result<(), BusError> {
        let offset = address.wrapping_sub(BASE);
        self.0
            .bpmp_write(offset, size, value)
            .ok_or(BusError::Abort)
    }

    fn read_words(&mut self, address: u32, words: &mut [u32]) -> Result<(), (usize, BusError)> {
        if self
            .0
            .read_words(address.wrapping_sub(BASE), words)
            .is_some()
        {
            return Ok(());
        }
        self.words_one_by_one(address, words.len(), |bus, i, at| {
            words[i] = bus.read(at, 4)?;
            Ok(())
        })
    }

    fn write_words(&mut self, address: u32, words: &[u32]) -> Result<(), (usize, BusError)> {
        if self
            .0
            .bpmp_write_words(address.wrapping_sub(BASE), words)
            .is_some()
        {
            return Ok(());
        }
        self.words_one_by_one(address, words.len(), |bus, i, at| {
            bus.write(at, 4, words[i])
        })
    }
}

/// A memcpy of 8 KiB with LDM/STM of eight registers, repeated.
const MEMCPY: &[u32] = &[
    0xE28B0902, // outer: add r0, r11, #0x8000
    0xE28B1901, // add r1, r11, #0x4000
    0xE3A02A02, // mov r2, #0x2000
    0xE8B107F8, // inner: ldmia r1!, {r3-r10}
    0xE8A007F8, // stmia r0!, {r3-r10}
    0xE2522020, // subs r2, r2, #32
    0x1AFFFFFB, // bne inner
    0xEAFFFFF7, // b outer
];

/// A delay loop.
const DELAY: &[u32] = &[
    0xE2500001, // 1: subs r0, r0, #1
    0x1AFFFFFD, // bne 1b
    0xE3E00000, // mvn r0, #0
    0xEAFFFFFB, // b 1b
];

/// A byte-at-a-time copy with a little work per byte, as a decompressor
/// does, in Thumb.
const BYTES: &[u16] = &[
    0x2400, // outer: movs r4, #0
    0x5D0B, // inner: ldrb r3, [r1, r4]
    0x5503, // strb r3, [r0, r4]
    0x3401, // adds r4, #1
    0x091D, // lsrs r5, r3, #4
    0x406E, // eors r6, r5
    0x4294, // cmp r4, r2
    0xD1F8, // bne inner
    0xE7F6, // b outer
];

fn workload(name: &str) -> (Arm7, IramBus) {
    let iram = Iram::new();
    let mut core = Arm7::new();
    core.reset(CODE);
    core.set_reg(11, BASE);
    match name {
        "memcpy" => {
            for (i, word) in MEMCPY.iter().enumerate() {
                iram.write(0x1000 + 4 * i as u32, 4, *word).unwrap();
            }
        }
        "delay" => {
            for (i, word) in DELAY.iter().enumerate() {
                iram.write(0x1000 + 4 * i as u32, 4, *word).unwrap();
            }
            core.set_reg(0, u32::MAX);
        }
        _ => {
            for (i, half) in BYTES.iter().enumerate() {
                iram.write(0x1000 + 2 * i as u32, 2, u32::from(*half))
                    .unwrap();
            }
            core.set_cpsr(core.cpsr() | PSR_T);
            core.set_reg(0, BASE + 0x8000);
            core.set_reg(1, BASE + 0x4000);
            core.set_reg(2, 0x1000);
        }
    }
    (core, IramBus(iram))
}

/// Millions of instructions per second over `steps`, the best of five
/// runs (the host is shared), with IRAM's code cached or not; the core's
/// state at the end, to compare.
fn measure(name: &str, steps: u64, cached: bool) -> (f64, Arm7) {
    let mut best = (0.0, Arm7::new());
    for _ in 0..5 {
        let (mut core, mut bus) = workload(name);
        let mut assist = Assist::new();
        if cached {
            let region = BASE..BASE + iram::SIZE as u32;
            assist.set_code_cache(region, bus.0.code_generation());
        }
        let start = Instant::now();
        assert_eq!(core.run_assisted(&mut bus, &mut assist, steps), Ok(steps));
        let mips = steps as f64 / start.elapsed().as_secs_f64() / 1e6;
        if mips > best.0 {
            best = (mips, core);
        }
    }
    best
}

#[test]
#[ignore = "a benchmark: run it with --ignored --nocapture"]
fn throughput() {
    const STEPS: u64 = 10_000_000;
    let only = std::env::var("BPMP_BENCH").ok();
    for name in ["delay", "memcpy", "bytes"] {
        if only.as_deref().is_some_and(|only| only != name) {
            continue;
        }
        let (plain, plain_core) = measure(name, STEPS, false);
        let (cached, cached_core) = measure(name, STEPS, true);
        assert_eq!(
            plain_core, cached_core,
            "{name}: the cache changed the outcome"
        );
        println!("{name:>8}: {plain:7.1} M instructions/s, {cached:7.1} with the code cache");
    }
}
