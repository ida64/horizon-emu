//! Guest time: how long the machine has run.
//!
//! Every clock the guest can see (the system counter, the display's frame
//! timing, the GPU's timer, the RTC, device deadlines) counts from this one.
//! It stands still while the machine is paused, so nothing a device does on
//! a timer happens while a snapshot is taken, and a restored machine sets
//! it to where the saved one was, so every deadline saved as guest time
//! means what it meant.
//!
//! Device threads that wait for a moment of guest time do so through
//! [`GuestClock::sleep_until`], or, when they wait on a condition variable
//! of their own, with [`GuestClock::until`] as the timeout; either way
//! they do not see time pass while the machine is paused.

use std::{
    sync::{Condvar, Mutex},
    time::{Duration, Instant},
};

/// How long a waiting thread sleeps at most before it looks again, while
/// the clock stands still.
const PAUSED_POLL: Duration = Duration::from_millis(20);

pub struct GuestClock {
    state: Mutex<State>,
    changed: Condvar,
}

#[derive(Clone, Copy)]
struct State {
    /// Guest time at `since`.
    at: Duration,
    /// Host time from which guest time runs on; `None` while paused.
    since: Option<Instant>,
}

impl State {
    fn now(self) -> Duration {
        match self.since {
            Some(since) => self.at + since.elapsed(),
            None => self.at,
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
            state: Mutex::new(State {
                at: Duration::ZERO,
                since: Some(Instant::now()),
            }),
            changed: Condvar::new(),
        }
    }

    /// Guest time now.
    pub fn now(&self) -> Duration {
        self.state.lock().unwrap().now()
    }

    /// Stops guest time where it is.
    pub fn pause(&self) {
        let mut state = self.state.lock().unwrap();
        state.at = state.now();
        state.since = None;
        self.changed.notify_all();
    }

    /// Lets guest time run on from where it stands.
    pub fn resume(&self) {
        let mut state = self.state.lock().unwrap();
        if state.since.is_none() {
            state.since = Some(Instant::now());
        }
        self.changed.notify_all();
    }

    pub fn is_paused(&self) -> bool {
        self.state.lock().unwrap().since.is_none()
    }

    /// Sets guest time; it stays paused or running as it was.
    pub fn set(&self, now: Duration) {
        let mut state = self.state.lock().unwrap();
        state.at = now;
        if state.since.is_some() {
            state.since = Some(Instant::now());
        }
        self.changed.notify_all();
    }

    /// How long the host should wait for guest time to reach `deadline`:
    /// zero once it has, a short poll interval while the clock stands
    /// still (the caller then looks again).
    pub fn until(&self, deadline: Duration) -> Duration {
        let state = *self.state.lock().unwrap();
        match state.since {
            Some(_) => deadline.saturating_sub(state.now()),
            None if state.at >= deadline => Duration::ZERO,
            None => PAUSED_POLL,
        }
    }

    /// Blocks until guest time has reached `deadline`, however long the
    /// clock stands still on the way.
    pub fn sleep_until(&self, deadline: Duration) {
        let mut state = self.state.lock().unwrap();
        loop {
            let now = state.now();
            if now >= deadline {
                return;
            }
            let wait = match state.since {
                Some(_) => deadline - now,
                None => PAUSED_POLL,
            };
            state = self.changed.wait_timeout(state, wait).unwrap().0;
        }
    }

    /// Blocks for `span` of guest time.
    pub fn sleep(&self, span: Duration) {
        self.sleep_until(self.now() + span);
    }
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
}
