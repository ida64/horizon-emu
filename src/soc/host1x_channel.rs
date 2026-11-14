//! Synchronous Tegra X1 host1x CDMA execution; offsets passed to engines are words.
//!
//! Encoding evidence: Linux `drivers/gpu/host1x/hw/` [opcodes.h],
//! [hw_host1x04_channel.h], [hw_host1x04_uclass.h], and [channel_hw.c].
//! The latter emits the 32-bit wait as payload at 0x4e followed by ID at 0x50.
//! nvservices initializes GET from PUT with DMACTRL=7, then runs with zero.
//! Unimplemented methods/conditions and unmet waits fail the MMIO submission;
//! they never manufacture a fence completion. Shared wait-base registers and
//! conditional host increments are not modeled. A failed stream stays faulted
//! until DMA GET reset, so retrying a failed submission cannot replay its effects.
//!
//! [opcodes.h]: https://github.com/torvalds/linux/blob/master/drivers/gpu/host1x/hw/opcodes.h
//! [hw_host1x04_channel.h]: https://github.com/torvalds/linux/blob/master/drivers/gpu/host1x/hw/hw_host1x04_channel.h
//! [hw_host1x04_uclass.h]: https://github.com/torvalds/linux/blob/master/drivers/gpu/host1x/hw/hw_host1x04_uclass.h
//! [channel_hw.c]: https://github.com/torvalds/linux/blob/master/drivers/gpu/host1x/hw/channel_hw.c

use std::sync::{Arc, Mutex};

use super::{Notes, host1x::Syncpoints, mc::DeviceMemory};
use crate::bus::MmioDevice;

type Engine = dyn Fn(u32, u32, u32) -> Option<()> + Send + Sync;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Opcode {
    SetClass {
        class: u32,
        offset: u32,
        mask: u16,
    },
    Incr {
        offset: u32,
        count: u16,
    },
    NonIncr {
        offset: u32,
        count: u16,
    },
    Mask {
        offset: u32,
        mask: u16,
    },
    Imm {
        offset: u32,
        value: u32,
    },
    Restart(u32),
    Gather {
        offset: u32,
        count: u16,
        insert: bool,
        increment: bool,
    },
}

/// Pure decoding: unsupported opcode families are not silently treated as NOPs.
fn decode(word: u32) -> Option<Opcode> {
    let offset = (word >> 16) & 0xfff;
    let count = word as u16;
    Some(match word >> 28 {
        0 => Opcode::SetClass {
            class: (word >> 6) & 0x3ff,
            offset,
            mask: (word & 0x3f) as u16,
        },
        1 => Opcode::Incr { offset, count },
        2 => Opcode::NonIncr { offset, count },
        3 => Opcode::Mask {
            offset,
            mask: count,
        },
        4 => Opcode::Imm {
            offset,
            value: word & 0xffff,
        },
        5 => Opcode::Restart((word & 0x0fff_ffff) << 4),
        6 => Opcode::Gather {
            offset,
            count: count & 0x3fff,
            insert: word & (1 << 15) != 0,
            increment: word & (1 << 14) != 0,
        },
        _ => return None,
    })
}

#[derive(Clone, Copy)]
enum Pending {
    None,
    Sequence {
        offset: u32,
        count: u16,
        increment: bool,
    },
    Mask {
        offset: u32,
        mask: u16,
    },
    Gather {
        offset: u32,
        count: u16,
        insert: bool,
        increment: bool,
    },
}

#[derive(Clone, Copy)]
struct Frame {
    get: u64,
    end: u64,
    // Insert gathers contain raw method data, not opcodes.
    insert: Option<(u32, bool)>,
}

struct State {
    start: u32,
    put: u32,
    get: u32,
    end: u32,
    ctrl: u32,
    class: u32,
    payload: Option<u32>,
    pending: Pending,
    fault: bool,
}

pub struct Channel {
    memory: DeviceMemory,
    syncpoints: Arc<Syncpoints>,
    engine: Arc<Engine>,
    notes: Arc<Notes>,
    state: Mutex<State>,
}

impl Channel {
    pub fn new(
        memory: DeviceMemory,
        syncpoints: Arc<Syncpoints>,
        engine: Arc<Engine>,
        notes: Arc<Notes>,
    ) -> Self {
        Self {
            memory,
            syncpoints,
            engine,
            notes,
            state: Mutex::new(State {
                start: 0,
                put: 0,
                get: 0,
                end: 0,
                ctrl: 1,
                class: 0,
                payload: None,
                pending: Pending::None,
                fault: false,
            }),
        }
    }

    fn method(&self, state: &mut State, offset: u32, value: u32) -> Result<(), &'static str> {
        if offset > 0xfff {
            return Err("method offset exceeds 12-bit range");
        }
        if state.class != 1 {
            return (self.engine)(state.class, offset, value).ok_or("engine refused method");
        }
        match offset {
            0 => {
                if value & !0xff != 0 {
                    return Err("conditional/reserved host syncpoint increment is unsupported");
                }
                self.syncpoints
                    .get(value as usize)
                    .ok_or("invalid syncpoint ID")?;
                self.syncpoints.add(value as usize, 1);
            }
            8 => {
                let current = self
                    .syncpoints
                    .get((value >> 24) as usize)
                    .ok_or("invalid wait syncpoint ID")?;
                // The legacy threshold has only 24 bits; compare in that domain.
                if current.wrapping_sub(value & 0xffffff) & 0x800000 != 0 {
                    return Err("unmet 24-bit syncpoint wait");
                }
            }
            0x4e => state.payload = Some(value),
            0x50 => {
                let threshold = state.payload.ok_or("32-bit wait has no loaded payload")?;
                let current = self
                    .syncpoints
                    .get(value as usize)
                    .ok_or("invalid wait syncpoint ID")?;
                if (current.wrapping_sub(threshold) as i32) < 0 {
                    return Err("unmet 32-bit syncpoint wait");
                }
            }
            9 | 0xb => return Err("shared syncpoint wait-base methods are unsupported"),
            _ => return Err("unknown host1x class method"),
        }
        Ok(())
    }

    fn word(&self, address: u64) -> Result<u32, &'static str> {
        if address & 3 != 0 {
            return Err("unaligned CDMA address");
        }
        let mut bytes = [0; 4];
        self.memory
            .read(address, &mut bytes)
            .ok_or("unmapped CDMA command memory")?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn execute(&self, state: &mut State) -> Result<(), &'static str> {
        if state.fault {
            return Err("channel remains faulted; DMA GET reset required");
        }
        // Fixed storage avoids an allocation per submission. Limits refuse loops
        // or pathological nesting rather than claiming that work completed.
        let mut frames = [Frame {
            get: 0,
            end: 0,
            insert: None,
        }; 32];
        let mut depth = 0usize;
        let mut budget = 1_048_576usize;
        loop {
            if depth != 0 && frames[depth - 1].get == frames[depth - 1].end {
                if !matches!(state.pending, Pending::None) {
                    return Err("gather ends inside a command packet");
                }
                depth -= 1;
                continue;
            }
            if depth == 0 && state.get == state.put {
                return Ok(());
            }
            if budget == 0 {
                return Err("CDMA execution limit reached (possible restart/gather cycle)");
            }
            budget -= 1;
            let address = if depth == 0 {
                if state.get < state.start || state.get >= state.end {
                    return Err("DMA GET outside START/END");
                }
                u64::from(state.get)
            } else {
                frames[depth - 1].get
            };
            let value = self.word(address)?;
            if depth == 0 {
                state.get = state.get.checked_add(4).ok_or("DMA GET overflow")?;
            } else {
                frames[depth - 1].get += 4;
                if let Some((offset, increment)) = frames[depth - 1].insert {
                    self.method(state, offset, value)?;
                    frames[depth - 1].insert = Some((offset + u32::from(increment), increment));
                    continue;
                }
            }
            match std::mem::replace(&mut state.pending, Pending::None) {
                Pending::Sequence {
                    offset,
                    count,
                    increment,
                } => {
                    self.method(state, offset, value)?;
                    if count > 1 {
                        state.pending = Pending::Sequence {
                            offset: offset + u32::from(increment),
                            count: count - 1,
                            increment,
                        };
                    }
                    continue;
                }
                Pending::Mask { offset, mask } => {
                    let bit = mask.trailing_zeros();
                    self.method(state, offset + bit, value)?;
                    let mask = mask & (mask - 1);
                    if mask != 0 {
                        state.pending = Pending::Mask { offset, mask };
                    }
                    continue;
                }
                Pending::Gather {
                    offset,
                    count,
                    insert,
                    increment,
                } => {
                    if value & 3 != 0 {
                        return Err("unaligned gather address");
                    }
                    if count != 0 {
                        if depth == frames.len() {
                            return Err("nested gather depth exceeded");
                        }
                        let end = u64::from(value) + u64::from(count) * 4;
                        if end > 1u64 << 32 {
                            return Err("gather crosses 32-bit address space");
                        }
                        frames[depth] = Frame {
                            get: value.into(),
                            end,
                            insert: if insert {
                                Some((offset, increment))
                            } else {
                                None
                            },
                        };
                        depth += 1;
                    }
                    continue;
                }
                Pending::None => {}
            }
            match decode(value).ok_or("unsupported CDMA opcode")? {
                Opcode::SetClass {
                    class,
                    offset,
                    mask,
                } => {
                    state.class = class;
                    if mask != 0 {
                        state.pending = Pending::Mask { offset, mask };
                    }
                }
                Opcode::Incr { offset, count } | Opcode::NonIncr { offset, count } => {
                    if count != 0 {
                        state.pending = Pending::Sequence {
                            offset,
                            count,
                            increment: value >> 28 == 1,
                        };
                    }
                }
                Opcode::Mask { offset, mask } => {
                    if mask != 0 {
                        state.pending = Pending::Mask { offset, mask };
                    }
                }
                Opcode::Imm { offset, value } => self.method(state, offset, value)?,
                Opcode::Restart(target) => {
                    if depth != 0 {
                        // Restart changes the DMA ring GET, not the gather cursor.
                        // This combination is not used by the evidenced submit path.
                        return Err("RESTART inside gather is unsupported");
                    }
                    if target < state.start || target >= state.end {
                        return Err("restart target outside START/END");
                    }
                    state.get = target;
                }
                Opcode::Gather {
                    offset,
                    count,
                    insert,
                    increment,
                } => {
                    if !insert && (offset != 0 || increment) {
                        return Err("reserved gather header fields");
                    }
                    state.pending = Pending::Gather {
                        offset,
                        count,
                        insert,
                        increment,
                    };
                }
            }
        }
    }
}

impl MmioDevice for Channel {
    fn read(&self, _core: usize, offset: u64, size: u8) -> Option<u64> {
        if size != 4 || offset & 3 != 0 {
            return None;
        }
        let state = self.state.lock().unwrap();
        let value = match offset {
            0 => {
                u32::from(
                    !state.fault
                        && state.get == state.put
                        && matches!(state.pending, Pending::None),
                ) << 11
            }
            0x14 => state.start,
            0x18 => state.put,
            0x1c => state.get,
            0x20 => state.end,
            0x24 => state.ctrl,
            _ => return None,
        };
        Some(value.into())
    }

    fn write(&self, _core: usize, offset: u64, size: u8, value: u64) -> Option<()> {
        if size != 4 || offset & 3 != 0 {
            return None;
        }
        let value = value as u32;
        let mut state = self.state.lock().unwrap();
        match offset {
            0x14 | 0x18 | 0x20 if value & 3 != 0 => {
                self.notes.add(format!(
                    "host1x channel: unaligned DMA register {offset:#x}={value:#x}"
                ));
                return None;
            }
            0x14 => state.start = value,
            0x18 => state.put = value,
            0x20 => state.end = value,
            0x24 => {
                if value & !7 != 0 {
                    self.notes
                        .add(format!("host1x channel: unsupported DMACTRL {value:#x}"));
                    return None;
                }
                state.ctrl = value;
                if value & 2 != 0 {
                    state.get = if value & 4 != 0 {
                        state.put
                    } else {
                        state.start
                    };
                    state.pending = Pending::None;
                    state.fault = false;
                }
            }
            _ => {
                self.notes.add(format!(
                    "host1x channel: unsupported register write {offset:#x}={value:#x}"
                ));
                return None;
            }
        }
        if matches!(offset, 0x18 | 0x24)
            && state.ctrl & 1 == 0
            && let Err(reason) = self.execute(&mut state)
        {
            state.fault = true;
            self.notes.add(format!(
                "host1x channel: {reason}; GET={:#010x}, PUT={:#010x}, class={:#x}",
                state.get, state.put, state.class
            ));
            return None;
        }
        Some(())
    }
}

#[cfg(test)]
mod tests {
    use super::{Opcode, decode};

    #[test]
    fn class_and_mask_fields() {
        assert_eq!(
            decode((0x4e << 16) | (1 << 6) | 5),
            Some(Opcode::SetClass {
                class: 1,
                offset: 0x4e,
                mask: 5
            })
        );
        assert_eq!(
            decode(0x3abc8001),
            Some(Opcode::Mask {
                offset: 0xabc,
                mask: 0x8001
            })
        );
        assert_eq!(
            decode(0x0fffffff),
            Some(Opcode::SetClass {
                class: 0x3ff,
                offset: 0xfff,
                mask: 0x3f
            })
        );
    }

    #[test]
    fn sequential_and_immediate_packets() {
        assert_eq!(
            decode(0x11230002),
            Some(Opcode::Incr {
                offset: 0x123,
                count: 2
            })
        );
        assert_eq!(
            decode(0x2123ffff),
            Some(Opcode::NonIncr {
                offset: 0x123,
                count: 0xffff
            })
        );
        assert_eq!(
            decode(0x20000000),
            Some(Opcode::NonIncr {
                offset: 0,
                count: 0
            })
        );
        assert_eq!(
            decode(0x4123abcd),
            Some(Opcode::Imm {
                offset: 0x123,
                value: 0xabcd
            })
        );
    }

    #[test]
    fn restart_and_gather_fields() {
        assert_eq!(decode(0x51234567), Some(Opcode::Restart(0x12345670)));
        assert_eq!(
            decode(0x60003fff),
            Some(Opcode::Gather {
                offset: 0,
                count: 0x3fff,
                insert: false,
                increment: false
            })
        );
        assert_eq!(
            decode(0x61238005),
            Some(Opcode::Gather {
                offset: 0x123,
                count: 5,
                insert: true,
                increment: false
            })
        );
        assert_eq!(
            decode(0x6123c005),
            Some(Opcode::Gather {
                offset: 0x123,
                count: 5,
                insert: true,
                increment: true
            })
        );
    }

    #[test]
    fn unknown_opcodes_refuse() {
        for opcode in [7, 8, 9, 10, 11, 12, 13, 14, 15] {
            assert_eq!(decode(opcode << 28), None);
        }
    }
}
