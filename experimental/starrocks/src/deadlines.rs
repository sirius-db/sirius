//! Per-query deadlines on this CN, from the FE's `TQueryOptions.query_timeout`.
//!
//! A query's deadline runs from its first fragment on this CN. Every wait of the query here
//! (receivers for their senders, `fetch_data` for rows, a receive allocation for a purge's frees)
//! ends by it, and a query still holding anything here once it passes is failed. The FE times the
//! query out too, and cancels it; whichever comes first ends it, and the other finds it ended.

use std::collections::hash_map::Entry;
use std::collections::{BTreeSet, HashMap};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::result_store::FragmentInstanceId;

/// The shortest and longest query timeout the CN accepts, in seconds.
const TIMEOUT_SECS: std::ops::RangeInclusive<i64> = 1..=24 * 3600;

/// A query's timeout on this CN, from its dispatch params; `None` when the FE sent none.
pub(crate) fn query_timeout(seconds: Option<i32>) -> Option<Duration> {
    let seconds = i64::from(seconds?);
    let clamped = seconds.clamp(*TIMEOUT_SECS.start(), *TIMEOUT_SECS.end());
    Some(Duration::from_secs(clamped as u64))
}

/// When one query's time is up on this CN.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Deadline {
    pub(crate) query: FragmentInstanceId,
    pub(crate) at: Instant,
    pub(crate) timeout: Duration,
}

#[derive(Debug, Default)]
struct State {
    by_query: HashMap<u64, Deadline>,
    /// The same deadlines in time order, so a check reads only those that passed.
    by_time: BTreeSet<(Instant, u64)>,
    /// A watcher thread is checking the deadlines.
    watched: bool,
}

/// The deadlines of the queries this CN started.
#[derive(Debug, Default)]
pub(crate) struct Deadlines {
    state: Mutex<State>,
}

impl Deadlines {
    /// Starts `query`'s deadline unless it already has one: the first fragment here sets it.
    /// Returns whether a watcher must be started to check it; if starting it fails, the caller
    /// says so with [`watcher_stopped`](Self::watcher_stopped).
    pub(crate) fn start(&self, query: FragmentInstanceId, timeout: Duration) -> bool {
        let mut state = self.lock();
        if let Entry::Vacant(entry) = state.by_query.entry(query.query_hi()) {
            let deadline = Deadline {
                query,
                at: Instant::now() + timeout,
                timeout,
            };
            entry.insert(deadline);
            state.by_time.insert((deadline.at, query.query_hi()));
        }
        !std::mem::replace(&mut state.watched, true)
    }

    /// No watcher checks the deadlines any more: the next [`start`](Self::start) starts one.
    pub(crate) fn watcher_stopped(&self) {
        self.lock().watched = false;
    }

    /// `query` (any instance id of it) ended here: its deadline no longer applies.
    pub(crate) fn end(&self, query: FragmentInstanceId) {
        let mut state = self.lock();
        if let Some(deadline) = state.by_query.remove(&query.query_hi()) {
            state.by_time.remove(&(deadline.at, query.query_hi()));
        }
    }

    /// How long `query` (any instance id of it) has left, if it has a deadline here.
    pub(crate) fn remaining(&self, query: FragmentInstanceId) -> Option<Duration> {
        self.lock()
            .by_query
            .get(&query.query_hi())
            .map(|deadline| deadline.at.saturating_duration_since(Instant::now()))
    }

    /// Takes the deadlines that passed by `now`, reading only those. Returns them, and whether
    /// any deadline is left to watch; when none is, the watcher stops and the next `start` starts
    /// another.
    pub(crate) fn take_passed(&self, now: Instant) -> (Vec<Deadline>, bool) {
        let mut state = self.lock();
        let mut passed = Vec::new();
        while let Some(&(at, query)) = state.by_time.first() {
            if at > now {
                break;
            }
            state.by_time.pop_first();
            passed.extend(state.by_query.remove(&query));
        }
        let left = !state.by_query.is_empty();
        state.watched = left;
        (passed, left)
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.lock().by_query.len()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_timeout_is_clamped_and_absent_means_none() {
        assert_eq!(query_timeout(None), None);
        assert_eq!(query_timeout(Some(0)), Some(Duration::from_secs(1)));
        assert_eq!(query_timeout(Some(-5)), Some(Duration::from_secs(1)));
        assert_eq!(query_timeout(Some(300)), Some(Duration::from_secs(300)));
        assert_eq!(
            query_timeout(Some(i32::MAX)),
            Some(Duration::from_secs(24 * 3600))
        );
    }

    #[test]
    fn the_first_fragment_sets_the_deadline_and_a_passed_one_is_taken_once() {
        let deadlines = Deadlines::default();
        let query = FragmentInstanceId::from_halves(4, 0);
        assert!(
            deadlines.start(query, Duration::from_millis(50)),
            "start a watcher"
        );
        assert!(
            !deadlines.start(
                FragmentInstanceId::from_halves(4, 7),
                Duration::from_secs(60)
            ),
            "one watcher at a time"
        );
        assert!(deadlines.remaining(query).unwrap() <= Duration::from_millis(50));
        let (passed, left) = deadlines.take_passed(Instant::now());
        assert!(passed.is_empty() && left);
        std::thread::sleep(Duration::from_millis(60));
        let (passed, left) = deadlines.take_passed(Instant::now());
        assert_eq!(passed.len(), 1);
        assert_eq!(passed[0].timeout, Duration::from_millis(50));
        assert!(!left);
        assert_eq!(deadlines.remaining(query), None);
        assert!(
            deadlines.start(query, Duration::from_secs(1)),
            "a new watcher once the last stopped"
        );
    }

    #[test]
    fn deadlines_are_taken_in_time_order_and_an_ended_query_drops_its_own() {
        let deadlines = Deadlines::default();
        for (hi, millis) in [(1, 30), (2, 10), (3, 20), (4, 10_000)] {
            deadlines.start(
                FragmentInstanceId::from_halves(hi, 0),
                Duration::from_millis(millis),
            );
        }
        deadlines.end(FragmentInstanceId::from_halves(3, 9));
        assert_eq!(deadlines.len(), 3);
        std::thread::sleep(Duration::from_millis(40));
        let (passed, left) = deadlines.take_passed(Instant::now());
        let order: Vec<u64> = passed.iter().map(|d| d.query.query_hi()).collect();
        assert_eq!(order, [2, 1], "the ended query 3 is not among them");
        assert!(left);
        assert_eq!(deadlines.len(), 1);
        deadlines.watcher_stopped();
        assert!(
            deadlines.start(
                FragmentInstanceId::from_halves(5, 0),
                Duration::from_secs(1)
            ),
            "a stopped watcher is replaced"
        );
    }
}
