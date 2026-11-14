//! The command stream: GPFIFO entries and the pushbuffer words they point
//! at, decoded as NVIDIA's public description of the Maxwell host class
//! defines them (open-gpu-doc `classes/host/clb06f.h`, `NVB06F_DMA_*` and
//! `NVB06F_GP_ENTRY*`).
//!
//! A header word's bits 31:29 (`SEC_OP`) select the form:
//!
//! | `SEC_OP` | form | fields |
//! |---|---|---|
//! | 0 | `GRP0_USE_TERT`: with `TERT_OP` (bits 17:16) 0 the old incrementing form | method byte address bits 12:2, subchannel 15:13, count 28:18 |
//! | 1 | incrementing | method 11:0, subchannel 15:13, count 28:16 |
//! | 2 | `GRP2_USE_TERT`: with `TERT_OP` 0 the old non-incrementing form | as the old incrementing form |
//! | 3 | non-incrementing | as incrementing |
//! | 4 | immediate: one method, its 13-bit argument in the header | data 28:16 |
//! | 5 | increment once: the first argument to the method, the rest to the next | as incrementing |
//! | 7 | end of pushbuffer segment | – |
//!
//! Methods are word indices (register offset / 4), as the engines take
//! them. The sub-device mask forms (`TERT_OP` 1-3 of group 0) and the
//! reserved forms are refused. So are old-form headers with bits 1:0 set,
//! which earlier generations used for jumps and calls that the GPFIFO
//! replaced. nvservices' own words are all new-form incrementing headers
//! (`0x2001001C`: count 1, subchannel 0, method 0x1C; spec §4.8).

use crate::maxwell::Unmodeled;

/// What a header word starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Header {
    /// `count` argument words follow.
    Methods {
        subchannel: u8,
        method: u32,
        count: u32,
        mode: Mode,
    },
    /// One method whose argument is in the header.
    Immediate {
        subchannel: u8,
        method: u32,
        argument: u32,
    },
    /// The rest of the segment is not executed.
    EndSegment,
}

/// Which method each argument word goes to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// The first to `method`, each following to the next method.
    Increasing,
    /// All to `method`.
    NonIncreasing,
    /// The first to `method`, the rest to `method + 1`.
    IncreaseOnce,
}

const SEC_OP_SHIFT: u32 = 29;
const SEC_OP_GRP0_USE_TERT: u32 = 0;
const SEC_OP_INC_METHOD: u32 = 1;
const SEC_OP_GRP2_USE_TERT: u32 = 2;
const SEC_OP_NON_INC_METHOD: u32 = 3;
const SEC_OP_IMMD_DATA_METHOD: u32 = 4;
const SEC_OP_ONE_INC: u32 = 5;
const SEC_OP_END_PB_SEGMENT: u32 = 7;
const TERT_OP_SHIFT: u32 = 16;
const TERT_OP: u32 = 3;
/// The old forms' low two bits; jumps and calls on earlier generations.
const OLD_FORM_LOW_BITS: u32 = 3;

fn subchannel(word: u32) -> u8 {
    (word >> 13 & 7) as u8
}

impl Header {
    pub fn decode(word: u32) -> Result<Self, Unmodeled> {
        let methods = |mode, method, count| Header::Methods {
            subchannel: subchannel(word),
            method,
            count,
            mode,
        };
        let new_method = word & 0xFFF;
        let new_count = word >> 16 & 0x1FFF;
        let old = |mode| {
            if word >> TERT_OP_SHIFT & TERT_OP != 0 || word & OLD_FORM_LOW_BITS != 0 {
                return Err(Unmodeled(format!(
                    "pushbuffer header {word:#010x}: sub-device mask, jump or call forms \
                     are not modeled"
                )));
            }
            Ok(methods(mode, word >> 2 & 0x7FF, word >> 18 & 0x7FF))
        };
        match word >> SEC_OP_SHIFT {
            SEC_OP_GRP0_USE_TERT => old(Mode::Increasing),
            SEC_OP_INC_METHOD => Ok(methods(Mode::Increasing, new_method, new_count)),
            SEC_OP_GRP2_USE_TERT => old(Mode::NonIncreasing),
            SEC_OP_NON_INC_METHOD => Ok(methods(Mode::NonIncreasing, new_method, new_count)),
            SEC_OP_IMMD_DATA_METHOD => Ok(Self::Immediate {
                subchannel: subchannel(word),
                method: new_method,
                argument: new_count,
            }),
            SEC_OP_ONE_INC => Ok(methods(Mode::IncreaseOnce, new_method, new_count)),
            SEC_OP_END_PB_SEGMENT => Ok(Self::EndSegment),
            _ => Err(Unmodeled(format!(
                "pushbuffer header {word:#010x} uses the reserved form 6"
            ))),
        }
    }
}

/// The arguments of a header still to come.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MethodRun {
    pub subchannel: u8,
    /// Method of the next argument.
    pub method: u32,
    /// Arguments still to come.
    pub left: u32,
    pub mode: Mode,
}

impl MethodRun {
    /// The run a header starts; `None` if no argument words follow.
    pub fn start(header: Header) -> Option<Self> {
        match header {
            Header::Methods {
                subchannel,
                method,
                count,
                mode,
            } if count > 0 => Some(Self {
                subchannel,
                method,
                left: count,
                mode,
            }),
            _ => None,
        }
    }

    /// The run after the next argument has been executed; `None` when that
    /// was the last.
    pub fn advance(self) -> Option<Self> {
        let method = match self.mode {
            Mode::Increasing => self.method + 1,
            Mode::NonIncreasing => self.method,
            Mode::IncreaseOnce => {
                return (self.left > 1).then_some(Self {
                    method: self.method + 1,
                    left: self.left - 1,
                    mode: Mode::NonIncreasing,
                    ..self
                });
            }
        };
        (self.left > 1).then_some(Self {
            method,
            left: self.left - 1,
            ..self
        })
    }
}

/// One GPFIFO entry (`NVB06F_GP_ENTRY0/1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpEntry {
    /// A pushbuffer segment: `length` words at virtual address `address`.
    Segment { address: u64, length: u32 },
    /// A control entry (length 0) with opcode `NOP`.
    Nop,
}

impl GpEntry {
    /// Decodes an entry. Conditional fetches and control entries other than
    /// `NOP` are refused. The privilege, level and sync bits change nothing
    /// for a host that executes one entry at a time to completion.
    pub fn decode(entry: [u32; 2]) -> Result<Self, Unmodeled> {
        let [low, high] = entry;
        let length = high >> 10 & 0x1F_FFFF;
        if length == 0 {
            return match high & 0xFF {
                0 => Ok(Self::Nop),
                opcode => Err(Unmodeled(format!(
                    "GPFIFO control entry {low:#010x} {high:#010x} (opcode {opcode}) is not modeled"
                ))),
            };
        }
        if low & 1 != 0 {
            return Err(Unmodeled(format!(
                "GPFIFO entry {low:#010x} {high:#010x} asks for a conditional fetch"
            )));
        }
        Ok(Self::Segment {
            address: u64::from(high & 0xFF) << 32 | u64::from(low & !3),
            length,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arguments(header: u32, words: usize) -> Vec<(u8, u32)> {
        let mut run = MethodRun::start(Header::decode(header).unwrap());
        let mut out = Vec::new();
        for _ in 0..words {
            let current = run.expect("ran out of arguments");
            out.push((current.subchannel, current.method));
            run = current.advance();
        }
        assert_eq!(run, None);
        out
    }

    #[test]
    fn the_drivers_own_words() {
        // SYNCPOINTA, SYNCPOINTB, WFI: one argument each on subchannel 0.
        assert_eq!(arguments(0x2001_001C, 1), [(0, 0x1C)]);
        assert_eq!(arguments(0x2001_001D, 1), [(0, 0x1D)]);
        assert_eq!(arguments(0x2001_001E, 1), [(0, 0x1E)]);
    }

    #[test]
    fn every_header_form() {
        // Incrementing, subchannel 3, method 0x123, three arguments.
        assert_eq!(
            arguments(1 << 29 | 3 << 16 | 3 << 13 | 0x123, 3),
            [(3, 0x123), (3, 0x124), (3, 0x125)]
        );
        // Non-incrementing.
        assert_eq!(
            arguments(3 << 29 | 2 << 16 | 1 << 13 | 0x80, 2),
            [(1, 0x80), (1, 0x80)]
        );
        // Increment once.
        assert_eq!(
            arguments(5 << 29 | 4 << 16 | 0x40, 4),
            [(0, 0x40), (0, 0x41), (0, 0x41), (0, 0x41)]
        );
        // Old incrementing: byte address 0x100 (method 0x40), count 2.
        assert_eq!(
            arguments(2 << 18 | 7 << 13 | 0x100, 2),
            [(7, 0x40), (7, 0x41)]
        );
        // Old non-incrementing.
        assert_eq!(
            arguments(2 << 29 | 3 << 18 | 0x104, 3),
            [(0, 0x41), (0, 0x41), (0, 0x41)]
        );
        // Immediate: 13 bits of data.
        assert_eq!(
            Header::decode(4 << 29 | 0x1ABC << 16 | 2 << 13 | 0x6C0).unwrap(),
            Header::Immediate {
                subchannel: 2,
                method: 0x6C0,
                argument: 0x1ABC
            }
        );
        assert_eq!(Header::decode(7 << 29).unwrap(), Header::EndSegment);
        // A zero word is an old-form header with no arguments: a no-op.
        assert_eq!(MethodRun::start(Header::decode(0).unwrap()), None);
        // Refused: reserved form, sub-device masks, old jumps.
        assert!(Header::decode(6 << 29).is_err());
        assert!(Header::decode(1 << 16 | 0x10).is_err());
        assert!(Header::decode(0x0000_1001).is_err());
    }

    #[test]
    fn gpfifo_entries() {
        // As the driver builds them: address bits 31:0, then the length in
        // words above bit 10 and address bits 39:32.
        assert_eq!(
            GpEntry::decode([0x0012_3400, 8 << 10 | 0x04]).unwrap(),
            GpEntry::Segment {
                address: 0x4_0012_3400,
                length: 8
            }
        );
        assert_eq!(GpEntry::decode([0, 0]).unwrap(), GpEntry::Nop);
        assert!(GpEntry::decode([0, 1]).is_err());
        assert!(GpEntry::decode([0x1001, 1 << 10]).is_err());
    }
}
