//! `--accept-forged-tickets`: ES's title-key path, made to take a stored
//! ticket without checking its signature.
//!
//! ES (program 0x0100000000000033) checks the RSA signature of a stored
//! ticket each time it is asked for that ticket's title key: IETicketService
//! command 503 (`sub_7B100` → `sub_36A40` → `sub_41780` → `sub_4D760` →
//! `sub_4EDA0`), which NS sends when it launches an application. In
//! `sub_4EDA0`, after reading the ticket's record, the call at 0x4EE80
//! (`BL sub_4F0A0`) loads the issuer's certificates and has `sub_4E450`
//! verify the ticket against them; a nonzero result there ends the command
//! with that result (`CBNZ W0` at 0x4EE84). A ticket with a forged
//! signature (an NSP's all-0xFF one) gets 0x291 there, and the title's key
//! never reaches FS. Addresses are those of ES 23.5.0's `main.elf`, whose
//! GNU build ID is [`ES_BUILD_ID`].
//!
//! With the option, once ES's main module is mapped, the call is replaced
//! in guest DRAM by `MOV W0, WZR`: the check reports success, the
//! certificates are not even read. Only stored tickets are spared:
//! `ImportTicket` (`sub_34D10`) calls `sub_4E450` itself (at 0x34D44) and
//! still verifies. The eMMC image is untouched; the firmware is changed in
//! memory only, so the patch lives exactly as long as that DRAM: a snapshot
//! taken after it keeps it, and a machine powered up again (a restart) or
//! restored from before it gets it anew, from the run that starts it.
//!
//! The process is found from the kernel's own objects
//! ([`Machine::program_processes`]) and its main module the way the crash
//! reporter finds modules ([`Machine::modules`]), by build ID. A poll costs
//! one walk of the process slab until ES exists, then one of ES's page
//! tables, and stops at the first answer.

use std::{
    ffi::c_void,
    fmt,
    sync::atomic::Ordering,
    thread,
    time::{Duration, Instant},
};

use super::{Machine, Process, Session, TraceEvent};
use crate::arm64::TranslationRegime;

/// Program ID of the eTicket service, ES.
const ES_PROGRAM_ID: u64 = 0x0100_0000_0000_0033;
/// GNU build ID of ES 23.5.0's main module (the note at its `main.elf`
/// vaddr 0xec37c, the end of its read-only data).
const ES_BUILD_ID: [u8; 20] = [
    0x70, 0x2c, 0xd3, 0x02, 0xfb, 0x34, 0x13, 0x00, 0xbb, 0x1d, 0x69, 0xc9, 0x6c, 0x7f, 0xc6, 0x6f,
    0xb8, 0x0b, 0xe5, 0x99,
];
/// Offset in ES's main module of the call that checks a stored ticket.
const CHECK_CALL: u64 = 0x4_EE80;
/// What is there: `BL sub_4F0A0` (`main.elf` vaddr 0x4EE80, `94000088`).
const CHECK_CALL_BL: u32 = 0x9400_0088;
/// What replaces it: `MOV W0, WZR` (`ORR W0, WZR, WZR`).
const MOV_W0_WZR: u32 = 0x2A1F_03E0;
/// How often ES is looked for until the patch is decided.
const POLL: Duration = Duration::from_millis(50);

// libSystem: invalidates the instruction cache for a range of the host's
// address space (`<libkern/OSCacheControl.h>`).
unsafe extern "C" {
    fn sys_icache_invalidate(start: *mut c_void, len: usize);
}

/// What became of the patch, reported once per run.
#[derive(Debug, Clone, Copy)]
pub enum ForgedTickets {
    /// The call was replaced; `read_back` is what DRAM then holds.
    Applied {
        process: Process,
        address: u64,
        physical: u64,
        read_back: u32,
        after: Duration,
    },
    /// The machine's DRAM already held the patch (restored from a snapshot
    /// taken after it).
    AlreadyApplied {
        process: Process,
        address: u64,
        after: Duration,
    },
    /// ES's main module is the expected one but the instruction is not:
    /// left alone.
    UnexpectedInstruction {
        process: Process,
        address: u64,
        found: u32,
    },
    /// ES's modules are all mapped and none is ES 23.5.0's main module.
    OtherEs { process: Process },
}

impl fmt::Display for ForgedTickets {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let es = |process: &Process| format!("{} (process {})", process.name(), process.id);
        match self {
            Self::Applied {
                process,
                address,
                physical,
                read_back,
                after,
            } => write!(
                f,
                "accept-forged-tickets: {} main+{CHECK_CALL:#x} (virtual {address:#x}, physical \
                 {physical:#x}): BL sub_4F0A0 ({CHECK_CALL_BL:08x}) replaced by MOV W0, WZR, \
                 read back {read_back:08x}, {:.1} s into the run",
                es(process),
                after.as_secs_f64()
            ),
            Self::AlreadyApplied {
                process,
                address,
                after,
            } => write!(
                f,
                "accept-forged-tickets: {} main+{CHECK_CALL:#x} (virtual {address:#x}) already \
                 holds MOV W0, WZR ({MOV_W0_WZR:08x}), {:.1} s into the run",
                es(process),
                after.as_secs_f64()
            ),
            Self::UnexpectedInstruction {
                process,
                address,
                found,
            } => write!(
                f,
                "accept-forged-tickets: {} main+{CHECK_CALL:#x} (virtual {address:#x}) holds \
                 {found:08x}, not BL sub_4F0A0 ({CHECK_CALL_BL:08x}): left alone, forged \
                 tickets stay refused",
                es(process)
            ),
            Self::OtherEs { process } => write!(
                f,
                "accept-forged-tickets: {} is not ES 23.5.0 (no module with build ID {}): left \
                 alone, forged tickets stay refused",
                es(process),
                ES_BUILD_ID
                    .iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect::<String>()
            ),
        }
    }
}

/// What to do with the instruction found at the check call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CheckCall {
    Patch,
    AlreadyPatched,
    Unexpected,
}

fn check_call(instruction: u32) -> CheckCall {
    match instruction {
        CHECK_CALL_BL => CheckCall::Patch,
        MOV_W0_WZR => CheckCall::AlreadyPatched,
        _ => CheckCall::Unexpected,
    }
}

impl Session<'_> {
    /// Looks for ES every [`POLL`] until the patch is decided, and reports
    /// that. Runs on a thread of its own for [`super::RunOptions::accept_forged_tickets`].
    pub(super) fn accept_forged_tickets(&self) {
        let started = Instant::now();
        while !self.stop.load(Ordering::SeqCst) {
            thread::sleep(POLL);
            let Some((regime, slide)) = self.kernel_view.get() else {
                continue;
            };
            let Some(processes) = self
                .machine
                .program_processes(regime, *slide, ES_PROGRAM_ID)
            else {
                continue;
            };
            let decided = processes
                .iter()
                .find_map(|(process, user)| self.machine.patch_es(process, user, started));
            if let Some(outcome) = decided {
                (self.options.trace)(TraceEvent::ForgedTickets(outcome));
                return;
            }
        }
    }
}

impl Machine {
    /// Patches the ES process `process`, whose address space is `user`:
    /// `None` while its main module is not all mapped yet.
    fn patch_es(
        &self,
        process: &Process,
        user: &TranslationRegime,
        started: Instant,
    ) -> Option<ForgedTickets> {
        let process = *process;
        let modules = self.modules(user);
        let Some(main) = modules
            .iter()
            .find(|module| module.build_id.as_deref() == Some(&ES_BUILD_ID[..]))
        else {
            // The loader makes the text executable and the read-only data
            // read-only only once it has written them (Atmosphère's
            // `ldr_process_creation.cpp` [inference]): until then a module
            // shows no build ID.
            let mapped = modules.iter().any(|module| module.mod0)
                && modules
                    .iter()
                    .all(|module| !module.mod0 || module.build_id.is_some());
            return mapped.then_some(ForgedTickets::OtherEs { process });
        };
        let address = main.base + CHECK_CALL;
        let physical = user.translate(address, |table| self.ram.read_u64(table))?;
        let mut word = [0; 4];
        self.ram.read(physical, &mut word)?;
        let found = u32::from_le_bytes(word);
        Some(match check_call(found) {
            CheckCall::Unexpected => ForgedTickets::UnexpectedInstruction {
                process,
                address,
                found,
            },
            CheckCall::AlreadyPatched => ForgedTickets::AlreadyApplied {
                process,
                address,
                after: started.elapsed(),
            },
            CheckCall::Patch => {
                self.ram.write(physical, &MOV_W0_WZR.to_le_bytes())?;
                // Cores fetch guest code through their instruction caches,
                // which the host's write by data does not reach. The line
                // has not been fetched (only command 503 runs it, and no
                // application has been launched this early), but an
                // unrelated fetch nearby may have brought it in; `IC IVAU`
                // for the host's mapping of the line is the invalidation by
                // physical address the host can make, which the guest's
                // view of the same memory shares [inference: Apple's cores
                // look their instruction caches up by physical address].
                // Nothing else holds the instruction: Hypervisor.framework
                // runs guest code natively, with no translated copy.
                let offset = physical - self.ram.base();
                // SAFETY: `ram.write` just succeeded for these 4 bytes, so
                // they lie inside the guest RAM mapping.
                unsafe {
                    sys_icache_invalidate(
                        self.ram.host_ptr().wrapping_add(offset as usize).cast(),
                        word.len(),
                    );
                }
                self.ram.read(physical, &mut word)?;
                ForgedTickets::Applied {
                    process,
                    address,
                    physical,
                    read_back: u32::from_le_bytes(word),
                    after: started.elapsed(),
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_expected_call_is_patched() {
        assert_eq!(check_call(CHECK_CALL_BL), CheckCall::Patch);
        assert_eq!(check_call(MOV_W0_WZR), CheckCall::AlreadyPatched);
        // `BL` to anywhere else, and the `CBNZ` after it.
        assert_eq!(check_call(0x9400_0089), CheckCall::Unexpected);
        assert_eq!(check_call(0x3500_1000), CheckCall::Unexpected);
    }

    #[test]
    fn encodings_are_what_they_say() {
        // BL: 0b100101 then imm26 = (target - pc) / 4.
        assert_eq!(
            CHECK_CALL_BL,
            0x9400_0000 | ((0x4_F0A0 - CHECK_CALL as u32) >> 2)
        );
        // ORR (shifted register), 32-bit: Rm = Rn = 31, Rd = 0.
        assert_eq!(MOV_W0_WZR, 0x2A00_0000 | (31 << 16) | (31 << 5));
    }
}
