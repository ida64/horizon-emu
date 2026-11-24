//! Which instructions a program can execute, and which of them control can
//! jump to.
//!
//! Control instructions are translated with the interpreter's own model
//! (a per-lane reconvergence stack, see [`super::super::exec`]); the
//! translation needs the set of instructions to translate and the program
//! counters a jump can land on (the `case` labels of the dispatch
//! `switch`): the entry, the targets of `BRA`, the targets `SSY`/`PBK`/
//! `PCNT` push (where `SYNC`/`BRK`/`CONT` land) and those of `BRX`. They
//! are found by walking the lanes' (program counter, stack) states, which
//! also gives every pop its exact destination.
//!
//! `BRX` jumps to `pc + 8 + imm24 + register`, where the compiler loads
//! the register from a jump table in constant buffer 1 (an `LDC` B32 of
//! slot 1, as NVN emits for `switch`). That every value the register can
//! hold at the branch comes
//! from such a load (or is the initial zero) is checked over all paths.
//! The words such a load can read are those of its table when its address
//! is provably the clamped index `min(x, K) << 2` the compiler emits
//! (`IMNMX` with an immediate `K`, `SHL` by 2), otherwise every word of
//! the bound constant buffer 1; plus 0 (what a read past the bound size
//! returns). The targets are exact for that buffer content, which the
//! translation therefore depends on.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use super::super::Program;
use super::super::decode::Op;
use super::super::exec::{bit, bits, branch_target, is_branch_to_self, sext};
use super::{PT, RZ, STACK_DEPTH};

/// `"{mnemonic} {what} {word} at {pc}"`, the interpreter's error format.
pub(super) fn describe(program: &Program, pc: u32, what: &str) -> String {
    match program.code.get((pc / 8) as usize).copied().flatten() {
        Some(inst) => format!(
            "{} {what} {:#018x} at {pc:#x}",
            inst.op.mnemonic(),
            inst.word
        ),
        None => format!("{what} at {pc:#x}"),
    }
}

pub(super) fn next_pc(pc: u32) -> u32 {
    let next = pc + 8;
    if next.is_multiple_of(32) {
        next + 8
    } else {
        next
    }
}

/// Whether the guard (bits 16-19) or condition-code test (bits 0-4) of a
/// control instruction can never hold: it never executes.
pub(super) fn inert(word: u64) -> bool {
    (bits(word, 16, 3) == PT && bit(word, 19)) || bits(word, 0, 5) == 0
}

/// Whether a control instruction always executes.
pub(super) fn always(word: u64) -> bool {
    bits(word, 16, 3) == PT && !bit(word, 19) && bits(word, 0, 5) == 15
}

pub(super) struct Flow {
    /// The instructions that can execute, in program order.
    pub pcs: Vec<u32>,
    /// The instructions control can jump to (the entry included).
    pub labels: BTreeSet<u32>,
    /// No jumps at all: only `EXIT`/`KIL` leave the straight line.
    pub linear: bool,
    /// Reachable `BRX`: the labels depend on constant buffer 1.
    pub jump_tables: bool,
}

/// Whether `inst` may write register `reg` (conservatively: any
/// destination field that covers it).
fn may_write(inst: super::super::exec::Inst, reg: u32) -> bool {
    let w = inst.word;
    let d = bits(w, 0, 8);
    let within = |base: u32, count: u32| base != RZ && (base..base + count).contains(&reg);
    match inst.op {
        Op::Bra
        | Op::Brx
        | Op::Sync
        | Op::Brk
        | Op::Cont
        | Op::Exit
        | Op::Kil
        | Op::Ssy
        | Op::Pbk
        | Op::Pcnt
        | Op::Nop
        | Op::Depbar
        | Op::Membar
        | Op::Bar
        | Op::Stg
        | Op::St
        | Op::Ast => false,
        Op::Texs | Op::Tlds | Op::Tld4s => within(d, 2) || within(bits(w, 28, 8), 2),
        Op::Tex
        | Op::TexB
        | Op::Tld
        | Op::TldB
        | Op::Tld4
        | Op::Tld4B
        | Op::Tmml
        | Op::TmmlB
        | Op::Txd
        | Op::TxdB
        | Op::Txq
        | Op::TxqB
        | Op::Ald
        | Op::Ldg
        | Op::Ldl
        | Op::Lds
        | Op::Ld => within(d, 4),
        Op::Ldc => within(d, 2),
        _ => d == reg,
    }
}

/// A program counter and an interned reconvergence stack content.
type State = (u32, usize);

/// More (program counter, stack) states than this: the control flow is
/// refused rather than walked further.
const STATE_LIMIT: usize = 1 << 16;

/// The lanes' reachable states and their edges.
struct Graph {
    successors: BTreeMap<State, Vec<State>>,
    predecessors: BTreeMap<State, Vec<State>>,
    labels: BTreeSet<u32>,
    /// The states of each reachable `BRX`.
    indirect: BTreeMap<u32, Vec<State>>,
}

/// Walks the (program counter, reconvergence stack) states from the entry
/// as the interpreter's lanes would (predicates and condition codes taken
/// both ways), so each pop goes exactly where the stack says. A state
/// where the interpreter stops with an error (stack overflow, nothing to
/// pop) has no successor: the translation ends the lane there. `BRX` at
/// `pc` goes to `tables[pc]` (nowhere yet if absent).
fn walk(program: &Program, tables: &BTreeMap<u32, Vec<u32>>) -> Result<Graph, String> {
    let refuse = |pc: u32, what: &str| describe(program, pc, what);
    let mut stacks: Vec<Vec<(u8, u32)>> = vec![Vec::new()];
    let mut stack_index: HashMap<Vec<(u8, u32)>, usize> = HashMap::from([(Vec::new(), 0)]);
    let mut labels = BTreeSet::from([8u32]);
    let mut successors: BTreeMap<State, Vec<State>> = BTreeMap::new();
    let mut indirect: BTreeMap<u32, Vec<State>> = BTreeMap::new();
    let mut work: Vec<State> = vec![(8, 0)];
    while let Some(state) = work.pop() {
        if successors.contains_key(&state) {
            continue;
        }
        if successors.len() == STATE_LIMIT {
            return Err(refuse(
                state.0,
                &format!("control flow with more than {STATE_LIMIT} reconvergence stack states"),
            ));
        }
        let (pc, stack_id) = state;
        let Some(inst) = program.code.get((pc / 8) as usize).copied().flatten() else {
            return Err(format!(
                "execution reaches {pc:#x}, outside the decoded program"
            ));
        };
        let w = inst.word;
        let target = || branch_target(w, pc).ok_or_else(|| refuse(pc, "target before the program"));
        let stack = stacks[stack_id].clone();
        let mut intern = |stack: Vec<(u8, u32)>| -> usize {
            *stack_index.entry(stack.clone()).or_insert_with(|| {
                stacks.push(stack);
                stacks.len() - 1
            })
        };
        let next = (next_pc(pc), stack_id);
        // (successor, whether it is a jump).
        let mut to: Vec<(State, bool)> = Vec::new();
        let mut falls_through = true;
        let kind = match inst.op {
            Op::Ssy | Op::Sync => 0u8,
            Op::Pbk | Op::Brk => 1,
            _ => 2,
        };
        match inst.op {
            Op::Ssy | Op::Pbk | Op::Pcnt => {
                if bit(w, 5) {
                    return Err(refuse(pc, "CA (constant-buffer target)"));
                }
                let target = target()?;
                // Pushed targets are where pops land: labels.
                labels.insert(target);
                falls_through = false;
                if stack.len() < STACK_DEPTH as usize {
                    let mut pushed = stack;
                    pushed.push((kind, target));
                    to.push(((next.0, intern(pushed)), false));
                }
            }
            Op::Bra | Op::Brx if !inert(w) => {
                if bit(w, 5) {
                    return Err(refuse(pc, "CA (constant-buffer target)"));
                }
                if bit(w, 6) {
                    return Err(refuse(pc, "LMT"));
                }
                if inst.op == Op::Bra {
                    // The terminating branch to itself ends the lane (the
                    // interpreter stops there).
                    if !is_branch_to_self(w, pc) {
                        to.push(((target()?, stack_id), true));
                    }
                } else {
                    indirect.entry(pc).or_default().push(state);
                    let targets = tables.get(&pc).into_iter().flatten();
                    to.extend(targets.map(|&t| ((t, stack_id), true)));
                }
                falls_through = !always(w);
            }
            Op::Sync | Op::Brk | Op::Cont if !inert(w) => {
                let mut popped = stack;
                let target = loop {
                    let Some((entry, target)) = popped.pop() else {
                        break None;
                    };
                    if entry == kind {
                        break Some(target);
                    }
                    // SYNC pops only its own SSY entry; BRK and CONT
                    // unwind inner SSY entries.
                    if kind == 0 || entry != 0 {
                        break None;
                    }
                };
                if let Some(target) = target {
                    to.push(((target, intern(popped)), true));
                }
                falls_through = !always(w);
            }
            Op::Exit | Op::Kil if !inert(w) => {
                if inst.op == Op::Exit && bit(w, 5) {
                    return Err(refuse(pc, "KEEPREFCOUNT"));
                }
                falls_through = !always(w);
            }
            _ => {}
        }
        if falls_through {
            to.push((next, false));
        }
        for &((target, _), jump) in &to {
            if jump {
                labels.insert(target);
            }
        }
        let to: Vec<State> = to.into_iter().map(|(s, _)| s).collect();
        work.extend(&to);
        successors.insert(state, to);
    }
    let mut predecessors: BTreeMap<State, Vec<State>> = BTreeMap::new();
    for (&from, to) in &successors {
        for &to in to {
            predecessors.entry(to).or_default().push(from);
        }
    }
    Ok(Graph {
        successors,
        predecessors,
        labels,
        indirect,
    })
}

/// The instructions whose write of `reg` can be the value `reg` holds
/// when execution reaches `starts`, with their states; `None` for the
/// initial zero. A guarded write may not execute, so earlier writes count
/// too.
fn reaching_writes(
    program: &Program,
    graph: &Graph,
    starts: &[State],
    reg: u32,
) -> BTreeMap<Option<u32>, Vec<State>> {
    let mut writes: BTreeMap<Option<u32>, Vec<State>> = BTreeMap::new();
    let mut seen = BTreeSet::new();
    let mut work: Vec<State> = starts
        .iter()
        .flat_map(|s| graph.predecessors.get(s).into_iter().flatten().copied())
        .collect();
    while let Some(state) = work.pop() {
        if !seen.insert(state) {
            continue;
        }
        let inst = program.code[(state.0 / 8) as usize].expect("reachable instruction");
        let guarded = !(bits(inst.word, 16, 3) == PT && !bit(inst.word, 19));
        if may_write(inst, reg) {
            writes.entry(Some(state.0)).or_default().push(state);
            if !guarded {
                continue;
            }
        }
        if state == (8, 0) {
            writes.entry(None).or_default().push(state);
        }
        work.extend(
            graph
                .predecessors
                .get(&state)
                .into_iter()
                .flatten()
                .copied(),
        );
    }
    if starts.contains(&(8, 0)) {
        writes.entry(None).or_default().push((8, 0));
    }
    writes
}

/// The byte addresses an `LDC` (at `states`) can read, when its address
/// register is provably `min(x, K) << 2` (an `IMNMX` with an immediate
/// bound and `SHL` by 2, the compiler's clamped `switch` index) or zero:
/// `(low, high)` inclusive, and whether an address can also wrap below 0
/// (a signed clamp of a negative index). `None` when that is not proven.
fn table_range(program: &Program, graph: &Graph, states: &[State]) -> Option<(i64, i64, bool)> {
    let w = program.code[(states[0].0 / 8) as usize]?.word;
    let (ra, offset) = (bits(w, 8, 8), i64::from(sext(bits(w, 20, 16), 16)));
    if ra == RZ {
        return Some((offset, offset, false));
    }
    let mut range: Option<(i64, i64, bool)> = None;
    let mut widen = |low: i64, high: i64, wraps: bool| {
        range = Some(range.map_or((low, high, wraps), |(l, h, w)| {
            (l.min(low), h.max(high), w || wraps)
        }));
    };
    for (write, shl_states) in reaching_writes(program, graph, states, ra) {
        let Some(shl) = write else {
            widen(offset, offset, false);
            continue;
        };
        let inst = program.code[(shl / 8) as usize]?;
        if inst.op != Op::ShlI
            || bits(inst.word, 0, 8) != ra
            || super::ops::imm20(inst.word) != 2
            || bit(inst.word, 43)
        {
            return None;
        }
        let x = bits(inst.word, 8, 8);
        for (write, _) in reaching_writes(program, graph, &shl_states, x) {
            let Some(min) = write else {
                widen(offset, offset, false);
                continue;
            };
            let clamp = program.code[(min / 8) as usize]?;
            let m = clamp.word;
            // IMNMX with PT selects min(a, K) (`exec/integer.rs`).
            if clamp.op != Op::ImnmxI
                || bits(m, 0, 8) != x
                || bits(m, 39, 4) != PT
                || bits(m, 43, 2) != 0
            {
                return None;
            }
            let bound = super::ops::imm20(m);
            if bit(m, 48) {
                // Signed: any index up to the bound; negative ones reach
                // down to address 0 (and wrap past any bound size).
                widen(0, offset + 4 * i64::from(bound), true);
            } else {
                widen(offset, offset + 4 * i64::from(bound as u32), false);
            }
        }
    }
    range
}

/// The words of constant buffer 1 the offset register of the `BRX` with
/// `states` can hold (0 where the initial value or a read past the bound
/// size can reach it); an error if a value can come from elsewhere.
fn table_words(
    program: &Program,
    graph: &Graph,
    states: &[State],
    constant_buffer_1: &[u8],
) -> Result<BTreeSet<u32>, String> {
    let brx = states[0].0;
    let reg = bits(program.code[(brx / 8) as usize].expect("BRX").word, 8, 8);
    let mut words = BTreeSet::new();
    if reg == RZ {
        return Ok(BTreeSet::from([0]));
    }
    let word_at = |address: i64| -> Option<u32> {
        let start = usize::try_from(address).ok()?;
        let bytes = constant_buffer_1.get(start..start + 4)?;
        Some(u32::from_le_bytes(bytes.try_into().expect("4 bytes")))
    };
    let end = constant_buffer_1.len() as i64;
    for (write, ldc_states) in reaching_writes(program, graph, states, reg) {
        let Some(at) = write else {
            // The initial zero.
            words.insert(0);
            continue;
        };
        let inst = program.code[(at / 8) as usize].expect("instruction");
        let w = inst.word;
        if inst.op != Op::Ldc
            || bits(w, 0, 8) != reg
            || bits(w, 36, 5) != 1
            || bits(w, 48, 3) != 4
            || bits(w, 44, 2) != 0
        {
            return Err(describe(
                program,
                brx,
                &format!(
                    "offset register written at {at:#x} (not a constant buffer 1 jump table load)"
                ),
            ));
        }
        let (low, high, wraps) = table_range(program, graph, &ldc_states).unwrap_or((0, end, true));
        if wraps || high + 4 > end {
            // A read past the bound size returns 0.
            words.insert(0);
        }
        let low = low.max(0) & !3;
        words.extend((low..=high.min(end)).step_by(4).filter_map(word_at));
    }
    Ok(words)
}

/// The targets of the `BRX` at `pc` for offsets `words`: `pc + 8 +
/// imm24 + v` where that is an instruction (a target on a scheduling
/// word continues at the next).
fn jump_targets(program: &Program, pc: u32, words: &BTreeSet<u32>) -> Vec<u32> {
    let w = program.code[(pc / 8) as usize].expect("BRX").word;
    let base = (pc + 8).wrapping_add(sext(bits(w, 20, 24), 24) as u32);
    let mut targets: Vec<u32> = words
        .iter()
        .map(|&v| base.wrapping_add(v))
        .map(|t| if t % 32 == 0 { t.wrapping_add(8) } else { t })
        .filter(|&t| {
            t % 8 == 0
                && program
                    .code
                    .get((t / 8) as usize)
                    .is_some_and(Option::is_some)
        })
        .collect();
    targets.sort_unstable();
    targets.dedup();
    targets
}

/// `constant_buffer_1`: the bytes `LDC` reads from constant buffer 1 (the
/// bound range), where `BRX` jump tables live.
///
/// The targets of each `BRX` depend on the paths reaching it, which depend
/// on the targets: the walk is repeated with the tables found until they
/// no longer change, and the last walk (complete for those tables) is the
/// one the tables were proven on.
pub(super) fn analyze(program: &Program, constant_buffer_1: &[u8]) -> Result<Flow, String> {
    let mut tables: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    let mut rounds = 0;
    let graph = loop {
        let graph = walk(program, &tables)?;
        let mut found = BTreeMap::new();
        for (&pc, states) in &graph.indirect {
            let words = table_words(program, &graph, states, constant_buffer_1)?;
            found.insert(pc, jump_targets(program, pc, &words));
        }
        if found == tables {
            break graph;
        }
        tables = found;
        rounds += 1;
        if rounds > 16 {
            return Err("BRX jump tables do not settle".into());
        }
    };
    let reachable: BTreeSet<u32> = graph.successors.keys().map(|&(pc, _)| pc).collect();
    let labels = graph.labels;
    let linear = labels.len() == 1
        && reachable.iter().all(|&pc| {
            let inst = program.code[(pc / 8) as usize].expect("reachable instruction");
            !matches!(
                inst.op,
                Op::Bra | Op::Brx | Op::Sync | Op::Brk | Op::Cont | Op::Ssy | Op::Pbk | Op::Pcnt
            ) || (inert(inst.word) && !matches!(inst.op, Op::Ssy | Op::Pbk | Op::Pcnt))
        });
    Ok(Flow {
        pcs: reachable.into_iter().collect(),
        labels,
        linear,
        jump_tables: !graph.indirect.is_empty(),
    })
}
