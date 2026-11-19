//! Guest time: how long the machine has run.
//!
//! Every clock the guest can see (the system counter, the display's frame
//! timing, the GPU's timer, the RTC, device deadlines) counts from this one.
//! It stands still while the machine is paused, so nothing a device does on
//! a timer happens while a snapshot is taken, and a restored machine sets
//! it to where the saved one was, so every deadline saved as guest time
//! means what it meant.
//!
//! Device threads that wait for a moment of guest time wait on a condition
//! variable of their own, with [`GuestClock::until`] as the timeout, so
//! they do not see time pass while the machine is paused.
//!
//! The cores read guest time for every trapped counter read, tens of
//! thousands of times a second, so reading it takes no lock: the whole
//! state is one atomic word. A change marks the word busy while it
//! reads host time, and a reader whose host time may have been read while
//! a change was under way reads again, so reads and changes stay in one
//! order, as under a lock: no reader sees time run on past the moment a
//! pause stopped it at.

use std::{
    sync::atomic::{self, AtomicU64, Ordering},
    thread,
    time::{Duration, Instant},
};

/// How long a waiting thread sleeps at most before it looks again, while
/// the clock stands still.
const PAUSED_POLL: Duration = Duration::from_millis(20);

/// Set in [`GuestClock`]'s state word while the clock stands still.
const PAUSED: u64 = 1 << 63;
/// The rest of the state word: guest time while paused, the offset from
/// host to guest time while running, in nanoseconds modulo 2^63.
const NANOS: u64 = PAUSED - 1;
/// The state word while a change is under way; as a paused guest time it
/// would be 292 years.
const BUSY: u64 = u64::MAX;

pub struct GuestClock {
    /// Host time is counted in nanoseconds from here.
    origin: Instant,
    /// Paused: [`PAUSED`] and guest time. Running: host time minus guest
    /// time, modulo 2^63, so that guest time is host time minus this.
    state: AtomicU64,
}

/// What [`GuestClock`]'s state word says.
#[derive(Clone, Copy)]
enum State {
    /// Standing still at this guest time, in nanoseconds.
    Paused(u64),
    /// Running: guest time is host time minus this, modulo 2^63.
    Running(u64),
}

impl State {
    fn decode(word: u64) -> Self {
        match word & PAUSED {
            0 => Self::Running(word),
            _ => Self::Paused(word & NANOS),
        }
    }

    fn encode(self) -> u64 {
        match self {
            Self::Paused(at) => PAUSED | at & NANOS,
            Self::Running(offset) => offset & NANOS,
        }
    }

    /// Guest time, in nanoseconds, with host time at `host`.
    fn now(self, host: u64) -> u64 {
        match self {
            Self::Paused(at) => at,
            Self::Running(offset) => host.wrapping_sub(offset) & NANOS,
        }
    }

    /// Running, from guest time `at` with host time at `host`.
    fn running(at: u64, host: u64) -> Self {
        Self::Running(host.wrapping_sub(at) & NANOS)
    }

    /// [`GuestClock::pause`] with host time at `host`.
    fn paused(self, host: u64) -> Self {
        Self::Paused(self.now(host))
    }

    /// [`GuestClock::resume`] with host time at `host`.
    fn resumed(self, host: u64) -> Self {
        match self {
            Self::Paused(at) => Self::running(at, host),
            running => running,
        }
    }

    /// [`GuestClock::set`] to `now` nanoseconds with host time at `host`.
    fn set(self, now: u64, host: u64) -> Self {
        match self {
            Self::Paused(_) => Self::Paused(now),
            Self::Running(_) => Self::running(now, host),
        }
    }
}

impl Default for GuestClock {
    fn default() -> Self {
        Self::new()
    }
}

impl GuestClock {
    /// A clock at zero, running.
    pub fn new() -> Self {
        Self {
            origin: Instant::now(),
            state: AtomicU64::new(State::Running(0).encode()),
        }
    }

    /// Host time, in nanoseconds since `origin`.
    fn host(&self, at: Instant) -> u64 {
        nanos(at.saturating_duration_since(self.origin))
    }

    /// The state word once no change is under way. Changes are rare and
    /// short; the thread making one may still have been preempted, so a
    /// reader that finds one gives up its turn rather than spin.
    fn settled(&self) -> u64 {
        loop {
            let word = self.state.load(Ordering::Acquire);
            if word != BUSY {
                return word;
            }
            thread::yield_now();
        }
    }

    /// The state and the host time read while it held.
    fn read(&self) -> (State, u64) {
        loop {
            let word = self.settled();
            let host = self.host(Instant::now());
            atomic::fence(Ordering::Acquire);
            if self.state.load(Ordering::Relaxed) == word {
                return (State::decode(word), host);
            }
        }
    }

    /// Changes the state with `change`, given the current one and host
    /// time, read while the word is [`BUSY`].
    fn update(&self, change: impl Fn(State, u64) -> State) {
        let word = loop {
            let word = self.settled();
            if self
                .state
                .compare_exchange_weak(word, BUSY, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
            {
                break word;
            }
        };
        let host = self.host(Instant::now());
        let changed = change(State::decode(word), host).encode();
        self.state.store(changed, Ordering::Release);
    }

    /// Guest time now.
    pub fn now(&self) -> Duration {
        let (state, host) = self.read();
        Duration::from_nanos(state.now(host))
    }

    /// Stops guest time where it is.
    pub fn pause(&self) {
        self.update(State::paused);
    }

    /// Lets guest time run on from where it stands.
    pub fn resume(&self) {
        self.update(State::resumed);
    }

    pub fn is_paused(&self) -> bool {
        matches!(State::decode(self.settled()), State::Paused(_))
    }

    /// Sets guest time; it stays paused or running as it was.
    pub fn set(&self, now: Duration) {
        let now = nanos(now);
        self.update(|state, host| state.set(now, host));
    }

    /// How long the host should wait for guest time to reach `deadline`:
    /// zero once it has, a short poll interval while the clock stands
    /// still (the caller then looks again).
    pub fn until(&self, deadline: Duration) -> Duration {
        let (state, host) = self.read();
        let now = Duration::from_nanos(state.now(host));
        match state {
            State::Running(_) => deadline.saturating_sub(now),
            State::Paused(_) if now >= deadline => Duration::ZERO,
            State::Paused(_) => PAUSED_POLL,
        }
    }
}

/// `duration` in nanoseconds; guest and host times stay far below 2^63 ns
/// (292 years).
fn nanos(duration: Duration) -> u64 {
    duration.as_nanos() as u64 & NANOS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_stands_still_while_paused_and_can_be_set() {
        let clock = GuestClock::new();
        clock.pause();
        let paused_at = clock.now();
        std::thread::sleep(Duration::from_millis(30));
        assert_eq!(clock.now(), paused_at);
        assert_eq!(clock.until(paused_at + Duration::from_secs(1)), PAUSED_POLL);
        assert_eq!(clock.until(paused_at), Duration::ZERO);
        clock.set(Duration::from_secs(100));
        assert_eq!(clock.now(), Duration::from_secs(100));
        clock.resume();
        std::thread::sleep(Duration::from_millis(10));
        assert!(clock.now() > Duration::from_secs(100));
        assert!(clock.now() < Duration::from_secs(101));
    }

    /// The clock as it was kept under a lock before its state became one
    /// word: guest time at a host instant, and that instant while running.
    #[derive(Clone, Copy)]
    struct Reference {
        at: Duration,
        since: Option<Instant>,
    }

    impl Reference {
        fn now(self, host: Instant) -> Duration {
            match self.since {
                Some(since) => self.at + host.duration_since(since),
                None => self.at,
            }
        }
    }

    /// The state word gives the guest time the locked state gave, to the
    /// nanosecond, at every host instant, across pauses, resumptions and
    /// settings while paused and while running.
    #[test]
    fn guest_time_is_what_the_locked_clock_gave() {
        let origin = Instant::now();
        let clock = GuestClock {
            origin,
            state: AtomicU64::new(State::Running(0).encode()),
        };
        let mut reference = Reference {
            at: Duration::ZERO,
            since: Some(origin),
        };
        let mut state = State::decode(clock.state.load(Ordering::Relaxed));
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut random = move |bound: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % bound
        };
        let mut host = origin;
        for _ in 0..100_000 {
            host += Duration::from_nanos(random(3_000_000_000));
            let at = clock.host(host);
            match random(6) {
                0 => {
                    reference = Reference {
                        at: reference.now(host),
                        since: None,
                    };
                    state = state.paused(at);
                }
                1 => {
                    reference.since = reference.since.or(Some(host));
                    state = state.resumed(at);
                }
                2 => {
                    let now = Duration::from_nanos(random(1 << 50));
                    reference.at = now;
                    reference.since = reference.since.map(|_| host);
                    state = state.set(nanos(now), at);
                }
                _ => {}
            }
            state = State::decode(state.encode());
            assert_eq!(
                Duration::from_nanos(state.now(at)),
                reference.now(host),
                "at host {at}"
            );
        }
    }

    /// Readers on other threads never see guest time go back, which they
    /// would if one read time on past a pause that another then saw stand
    /// still at an earlier moment.
    #[test]
    fn readers_never_see_time_go_back_across_pauses() {
        let clock = GuestClock::new();
        let stop = std::sync::atomic::AtomicBool::new(false);
        std::thread::scope(|scope| {
            for _ in 0..3 {
                scope.spawn(|| {
                    let mut last = Duration::ZERO;
                    while !stop.load(Ordering::SeqCst) {
                        let now = clock.now();
                        assert!(now >= last, "{now:?} after {last:?}");
                        last = now;
                    }
                });
            }
            for _ in 0..20 {
                std::thread::sleep(Duration::from_millis(2));
                clock.pause();
                let at = clock.now();
                std::thread::sleep(Duration::from_millis(2));
                assert_eq!(clock.now(), at);
                clock.resume();
            }
            stop.store(true, Ordering::SeqCst);
        });
    }
}
