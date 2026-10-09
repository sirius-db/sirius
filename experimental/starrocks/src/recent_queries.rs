//! What this CN remembers about queries that ended, for a while after they did.
//!
//! A cancel can overtake a dispatch, and frames of a failed query keep arriving after its purge,
//! so the CN remembers each ended query long enough to refuse whatever of it comes late. Memory is
//! bounded by time rather than by count: a count lets a burst of short queries push out one that
//! is still sending. A hard cap backs the window up.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tracing::warn;

use crate::result_store::FragmentInstanceId;

/// How long an ended query is remembered. Generous: until the query timeout bounds every wait, a
/// late fragment can only be later than the FE's own statement timeout by accident.
pub(crate) const REMEMBER_FOR: Duration = Duration::from_secs(3600);

/// At most this many ended queries are remembered; past it the oldest is forgotten early.
const MAX_ENDED_QUERIES: usize = 65_536;

/// How every cause of a normal end (QUERY_FINISHED, LIMIT_REACH) begins. A cause leads every error
/// it ends up in, so a sender whose fragment stopped because its query ended normally can tell,
/// from the error alone, that nothing failed.
pub(crate) const ENDED_NORMALLY: &str = "the query ended normally";

/// Whether `error` comes from a query that ended normally rather than from a failure.
pub(crate) fn is_normal_end(error: &str) -> bool {
    error.starts_with(ENDED_NORMALLY)
}

/// Values by query ([`FragmentInstanceId::query_hi`]), each kept for a fixed time after it was
/// first recorded, and at most `cap` of them.
#[derive(Debug)]
pub(crate) struct RecentQueries<V> {
    window: Duration,
    cap: usize,
    /// What it holds for each query, and when that was recorded.
    entries: HashMap<u64, (Instant, V)>,
    /// When each query was recorded, oldest first. An entry whose time no longer matches
    /// `entries` was removed or replaced, and is skipped.
    recorded: VecDeque<(Instant, u64)>,
}

impl<V> RecentQueries<V> {
    pub(crate) fn new(window: Duration, cap: usize) -> Self {
        Self {
            window,
            cap,
            entries: HashMap::new(),
            recorded: VecDeque::new(),
        }
    }

    /// `query`'s value, recorded with `value()` first if it has none.
    pub(crate) fn get_or_insert_with(&mut self, query: u64, value: impl FnOnce() -> V) -> &mut V {
        self.expire();
        if let std::collections::hash_map::Entry::Vacant(entry) = self.entries.entry(query) {
            let now = Instant::now();
            entry.insert((now, value()));
            self.recorded.push_back((now, query));
            self.cap_entries();
        }
        &mut self.entries.get_mut(&query).expect("recorded above").1
    }

    pub(crate) fn get(&mut self, query: u64) -> Option<&V> {
        self.expire();
        self.entries.get(&query).map(|(_, value)| value)
    }

    pub(crate) fn remove(&mut self, query: u64) -> Option<V> {
        self.entries.remove(&query).map(|(_, value)| value)
    }

    #[cfg(test)]
    pub(crate) fn len(&mut self) -> usize {
        self.expire();
        self.entries.len()
    }

    /// Forgets every query recorded longer ago than the window.
    fn expire(&mut self) {
        let now = Instant::now();
        while let Some(&(at, _)) = self.recorded.front() {
            if now.duration_since(at) < self.window {
                break;
            }
            self.forget_oldest();
        }
    }

    /// Forgets the oldest queries past the cap.
    fn cap_entries(&mut self) {
        while self.entries.len() > self.cap {
            if let Some(query) = self.forget_oldest() {
                warn!(
                    query_hi = query,
                    cap = self.cap,
                    "forgetting an ended query early: too many are remembered"
                );
            }
        }
    }

    /// Forgets the oldest recorded query, returning it unless its record was already gone.
    fn forget_oldest(&mut self) -> Option<u64> {
        let (at, query) = self.recorded.pop_front()?;
        match self.entries.get(&query) {
            Some((recorded, _)) if *recorded == at => {
                self.entries.remove(&query);
                Some(query)
            }
            _ => None,
        }
    }
}

/// How one query ended on this CN.
#[derive(Debug, Default)]
struct QueryEnd {
    /// Why: its first failure, or the FE's cancel. Its later fragments, frames, receivers and
    /// allocations are refused with it.
    cause: Option<Arc<str>>,
    /// The FE ended it: it cancelled the query, or this CN's result sink delivered the last row.
    /// A failure of its instances after that is only teardown.
    by_fe: bool,
    /// How many cancels the FE sent for it.
    cancels: u32,
}

/// The one record per ended query that the exchange, the FE reports and the cancel handler share.
#[derive(Debug)]
pub(crate) struct QueryEnds {
    ends: Mutex<RecentQueries<QueryEnd>>,
}

impl Default for QueryEnds {
    fn default() -> Self {
        Self::new(REMEMBER_FOR)
    }
}

impl QueryEnds {
    pub(crate) fn new(window: Duration) -> Self {
        Self {
            ends: Mutex::new(RecentQueries::new(window, MAX_ENDED_QUERIES)),
        }
    }

    /// Records that `query` (any instance id of it) failed or ended because of `cause`, unless it
    /// already had a cause, which is kept. Returns whether this was the first.
    pub(crate) fn fail(&self, query: FragmentInstanceId, cause: &str) -> bool {
        let mut ends = self.lock();
        let end = ends.get_or_insert_with(query.query_hi(), QueryEnd::default);
        if end.cause.is_some() {
            return false;
        }
        end.cause = Some(Arc::from(cause));
        true
    }

    /// Why `query` ended here, if it did.
    pub(crate) fn cause(&self, query: FragmentInstanceId) -> Option<Arc<str>> {
        self.lock()
            .get(query.query_hi())
            .and_then(|end| end.cause.clone())
    }

    /// The FE cancelled `query` because of `cause`. Returns how many cancels it sent for it so far.
    pub(crate) fn cancel(&self, query: FragmentInstanceId, cause: &str) -> u32 {
        let mut ends = self.lock();
        let end = ends.get_or_insert_with(query.query_hi(), QueryEnd::default);
        end.by_fe = true;
        end.cause.get_or_insert_with(|| Arc::from(cause));
        end.cancels += 1;
        end.cancels
    }

    /// `query` ended normally elsewhere, as a remote sender's frame said. Returns whether this is
    /// how it ended here first.
    pub(crate) fn end_normally(&self, query: FragmentInstanceId, cause: &str) -> bool {
        let mut ends = self.lock();
        let end = ends.get_or_insert_with(query.query_hi(), QueryEnd::default);
        end.by_fe = true;
        if end.cause.is_some() {
            return false;
        }
        end.cause = Some(Arc::from(cause));
        true
    }

    /// This CN's result sink of `query` delivered its last row.
    pub(crate) fn delivered(&self, query: FragmentInstanceId) {
        self.lock()
            .get_or_insert_with(query.query_hi(), QueryEnd::default)
            .by_fe = true;
    }

    /// Whether the FE already ended `query`, so a failure of it is only teardown.
    pub(crate) fn ended_by_fe(&self, query: FragmentInstanceId) -> bool {
        self.lock()
            .get(query.query_hi())
            .is_some_and(|end| end.by_fe)
    }

    /// How many cancels the FE sent for `query`.
    #[cfg(test)]
    pub(crate) fn cancels(&self, query: FragmentInstanceId) -> u32 {
        self.lock()
            .get(query.query_hi())
            .map_or(0, |end| end.cancels)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, RecentQueries<QueryEnd>> {
        self.ends
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_query_is_remembered_for_the_window_however_many_others_end() {
        let mut recent = RecentQueries::new(Duration::from_millis(200), usize::MAX);
        *recent.get_or_insert_with(1, || "first") = "first";
        assert_eq!(
            *recent.get_or_insert_with(1, || "second"),
            "first",
            "the first value is kept"
        );
        for query in 2..5000 {
            recent.get_or_insert_with(query, || "burst");
        }
        assert_eq!(
            recent.get(1),
            Some(&"first"),
            "a burst of queries pushes out none"
        );
        std::thread::sleep(Duration::from_millis(250));
        assert_eq!(recent.get(1), None);
        assert_eq!(recent.len(), 0);
    }

    #[test]
    fn past_the_cap_the_oldest_query_is_forgotten() {
        let mut recent = RecentQueries::new(REMEMBER_FOR, 3);
        for query in 1..=4 {
            recent.get_or_insert_with(query, || query);
        }
        assert_eq!(recent.len(), 3);
        assert_eq!(recent.get(1), None);
        assert_eq!(recent.get(4), Some(&4));
        // A removed and re-recorded query is not forgotten by its old record.
        recent.remove(2);
        recent.get_or_insert_with(2, || 20);
        recent.get_or_insert_with(5, || 5);
        assert_eq!(recent.get(2), Some(&20));
        assert_eq!(recent.get(3), None);
    }

    #[test]
    fn one_record_per_query_shared_by_failure_cancel_and_delivery() {
        let ends = QueryEnds::default();
        let query = FragmentInstanceId::from_halves(9, 0);
        let instance = FragmentInstanceId::from_halves(9, 4);
        assert!(ends.fail(instance, "scan exploded"));
        assert!(!ends.fail(query, "a later error"));
        assert!(!ends.ended_by_fe(query));
        assert_eq!(
            ends.cancel(query, "query cancelled: the user cancelled the query"),
            1
        );
        assert_eq!(ends.cancel(instance, "again"), 2);
        assert_eq!(
            ends.cause(query).as_deref(),
            Some("scan exploded"),
            "the first cause is kept"
        );
        assert!(ends.ended_by_fe(instance));

        let delivered = FragmentInstanceId::from_halves(10, 2);
        ends.delivered(delivered);
        assert!(ends.ended_by_fe(delivered));
        assert_eq!(
            ends.cause(delivered),
            None,
            "a delivered query is not refused"
        );
        assert!(is_normal_end(&format!(
            "{ENDED_NORMALLY}: the query finished (late)"
        )));
        assert!(!is_normal_end(
            "query cancelled: the user cancelled the query"
        ));
    }
}
