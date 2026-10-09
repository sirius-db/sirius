//! In-memory buffer of executed-fragment results awaiting FE `fetch_data` collection.
//!
//! StarRocks dispatches a fragment with `exec_plan_fragment`, then polls `fetch_data` with the
//! fragment instance id until end-of-stream. This store bridges those two RPCs: the exec handler
//! buffers a fragment's rows here, and each `fetch_data` poll drains them.

use std::fmt;
use std::{
    collections::{HashMap, VecDeque},
    sync::{Condvar, Mutex, PoisonError},
    time::{Duration, Instant},
};

use starrocks_thrift::data::TResultBatch;
use starrocks_thrift::types::TUniqueId;
use uuid::Uuid;

use crate::proto::starrocks::PUniqueId;

/// StarRocks `fragment_instance_id`, the key the FE passes to `fetch_data`.
///
/// StarRocks identifies a fragment instance by a 128-bit id split into `hi`/`lo`
/// 64-bit halves (thrift `TUniqueId` on dispatch, proto `PUniqueId` on
/// `fetch_data`). It is held here as a [`Uuid`] so the two wire forms compare
/// equal and logs render the canonical hyphenated form instead of `hi-lo`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct FragmentInstanceId(Uuid);

impl FragmentInstanceId {
    /// Packs the `hi`/`lo` 64-bit halves of a StarRocks unique id into a [`Uuid`].
    pub(crate) fn from_halves(hi: i64, lo: i64) -> Self {
        Self(Uuid::from_u64_pair(hi as u64, lo as u64))
    }

    /// The `hi` half, which a fragment instance shares with its query: the FE derives instance
    /// ids as `(query.hi, query.lo + n)` (`ExecutionDAG.setInstanceId`). An exchange frame
    /// carries only its receiver's instance id, so this is how it is matched to its query.
    pub(crate) fn query_hi(self) -> u64 {
        self.0.as_u64_pair().0
    }

    /// The proto form, for routing a frame to this instance.
    pub(crate) fn to_proto(self) -> PUniqueId {
        let (hi, lo) = self.0.as_u64_pair();
        PUniqueId {
            hi: hi as i64,
            lo: lo as i64,
        }
    }
}

impl From<&TUniqueId> for FragmentInstanceId {
    fn from(id: &TUniqueId) -> Self {
        Self::from_halves(id.hi, id.lo)
    }
}

impl From<&PUniqueId> for FragmentInstanceId {
    fn from(id: &PUniqueId) -> Self {
        Self::from_halves(id.hi, id.lo)
    }
}

impl fmt::Display for FragmentInstanceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// One buffered fragment result and where the FE `fetch_data` poll is in draining it.
#[derive(Debug)]
enum FragmentState {
    /// A result fragment accepted before its exchange inputs arrived. `query` lets a failure
    /// anywhere in the query fail this slot.
    Waiting { query: FragmentInstanceId },
    /// Rows produced, not yet delivered.
    Pending(TResultBatch),
    /// Rows delivered, at the given time; the next poll reports end-of-stream.
    Drained(Instant),
    /// The query failed, at the given time; every poll reports the cause.
    Failed(String, Instant),
}

impl FragmentState {
    /// When the slot ended, rows delivered or failed; `None` while it still has work.
    fn ended_at(&self) -> Option<Instant> {
        match self {
            Self::Drained(at) | Self::Failed(_, at) => Some(*at),
            Self::Waiting { .. } | Self::Pending(_) => None,
        }
    }
}

/// What a single `fetch_data` poll should return to the FE.
#[derive(Debug)]
pub(crate) struct FetchOutcome {
    /// Result rows to ship as the response attachment, when present.
    pub(crate) batch: Option<TResultBatch>,
    /// Monotonic packet sequence the FE uses to detect lost packets.
    pub(crate) packet_seq: i64,
    /// End-of-stream marker; the FE stops polling once true.
    pub(crate) eos: bool,
}

/// At most this many ended slots are kept; past it the oldest is dropped early.
const MAX_ENDED_SLOTS: usize = 65_536;

/// Process-wide store of fragment results keyed by fragment instance id.
///
/// Shared across all BRPC connections via an `Arc` inside the compute-node service, so a
/// `fetch_data` poll on one connection sees results buffered by an `exec_plan_fragment` on another.
///
/// A slot that ended (rows delivered, or failed) is kept for `window` after it ended, so a repeat
/// poll still reads EOS and the poll after a failure still reads its cause, then dropped; at most
/// [`MAX_ENDED_SLOTS`] ended slots are kept.
#[derive(Debug)]
pub(crate) struct ResultStore {
    /// Buffered results, and when each ended slot ended.
    inner: Mutex<Slots>,
    /// Wakes a `take_next` blocked on a waiting fragment.
    ready: Condvar,
    /// How long an ended slot is kept.
    window: Mutex<Duration>,
}

#[derive(Debug, Default)]
struct Slots {
    by_id: HashMap<FragmentInstanceId, FragmentState>,
    /// Ended slots in the order they ended. An entry whose slot no longer ended at that time was
    /// dropped or replaced since, and is skipped.
    ended: VecDeque<(Instant, FragmentInstanceId)>,
}

impl Slots {
    /// Puts `state` in `id`'s slot, noting when it ended if it did.
    fn set(&mut self, id: FragmentInstanceId, state: FragmentState) {
        if let Some(ended) = state.ended_at() {
            self.ended.push_back((ended, id));
        }
        self.by_id.insert(id, state);
    }

    /// Drops the slots that ended longer ago than `window`, and the oldest past the cap.
    fn prune(&mut self, window: Duration) {
        let now = Instant::now();
        while let Some(&(ended, id)) = self.ended.front() {
            if now.duration_since(ended) < window && self.ended.len() <= MAX_ENDED_SLOTS {
                break;
            }
            self.ended.pop_front();
            if self
                .by_id
                .get(&id)
                .and_then(FragmentState::ended_at)
                .is_some_and(|at| at == ended)
            {
                self.by_id.remove(&id);
            }
        }
    }
}

impl Default for ResultStore {
    fn default() -> Self {
        Self {
            inner: Mutex::default(),
            ready: Condvar::new(),
            window: Mutex::new(crate::recent_queries::REMEMBER_FOR),
        }
    }
}

impl ResultStore {
    /// Reserves the slot of a result fragment that waits on exchange inputs, so `fetch_data`
    /// waits for its rows instead of reporting an unknown fragment. A slot that already exists is
    /// kept, so a repeated dispatch cannot hide rows or a failure behind a fresh wait.
    pub(crate) fn reserve(&self, id: FragmentInstanceId, query: FragmentInstanceId) {
        self.lock()
            .by_id
            .entry(id)
            .or_insert(FragmentState::Waiting { query });
    }

    /// Buffers an executed fragment's result for later `fetch_data` collection.
    pub(crate) fn insert(&self, id: FragmentInstanceId, batch: TResultBatch) {
        self.lock().set(id, FragmentState::Pending(batch));
        self.ready.notify_all();
    }

    /// Keeps an ended slot for `window` after it ended.
    pub(crate) fn set_window(&self, window: Duration) {
        *self
            .window
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = window;
    }

    /// Drops `query`'s (any instance id of it) slots whose rows were all delivered: the FE
    /// cancelled the query, so it polls no more.
    pub(crate) fn forget_delivered(&self, query: FragmentInstanceId) {
        self.lock().by_id.retain(|id, state| {
            id.query_hi() != query.query_hi() || !matches!(state, FragmentState::Drained(_))
        });
    }

    /// Whether a result slot of `query` (any instance id of it) still waits for its rows.
    pub(crate) fn waits_for(&self, query: FragmentInstanceId) -> bool {
        self.lock().by_id.values().any(|state| {
            matches!(state, FragmentState::Waiting { query: waiting }
                if waiting.query_hi() == query.query_hi())
        })
    }

    /// How many slots the store holds, ended or not.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.lock().by_id.len()
    }

    /// Fails every result slot of `query` not yet delivered, so `fetch_data` reports `error` at
    /// once instead of waiting out its timeout, and rows nobody will fetch are freed. `query` may
    /// be any instance id of the query: like the exchange purge, this matches on the hi half they
    /// share.
    pub(crate) fn fail_query(&self, query: FragmentInstanceId, error: &str) {
        let mut slots = self.lock();
        let failing: Vec<FragmentInstanceId> = slots
            .by_id
            .iter()
            .filter(|(id, state)| match state {
                FragmentState::Waiting { query: waiting } => waiting.query_hi() == query.query_hi(),
                FragmentState::Pending(_) => id.query_hi() == query.query_hi(),
                FragmentState::Drained(_) | FragmentState::Failed(..) => false,
            })
            .map(|(id, _)| *id)
            .collect();
        for id in failing {
            slots.set(id, FragmentState::Failed(error.to_string(), Instant::now()));
        }
        drop(slots);
        self.ready.notify_all();
    }

    /// Advances the `fetch_data` state machine for one fragment: deliver rows once, then EOS.
    /// A fragment still waiting on its exchange inputs blocks the caller for up to `timeout`;
    /// replying not-ready instead would desync the FE packet counter. An id this CN never buffered
    /// is an error (StarRocks treats a missing result buffer as a failure, not an empty result),
    /// as are a failed query and a timed-out wait. A drained fragment stays in the map, for the
    /// window, so a repeat poll still reports EOS rather than reading as unknown.
    ///
    /// TODO(starrocks-execute): this is a single-batch, single-poller model. The real executor
    /// needs (a) chunked/streamed delivery of many batches and (b) safety against duplicate or
    /// concurrent polls (advance state only after the response is written; keep a per-fragment
    /// in-flight guard).
    pub(crate) fn take_next(
        &self,
        id: FragmentInstanceId,
        timeout: Duration,
    ) -> Result<FetchOutcome, String> {
        let deadline = Instant::now() + timeout;
        let mut guard = self.lock();
        loop {
            match guard.by_id.get_mut(&id) {
                None => return Err(format!("no buffered result for fragment instance {id}")),
                Some(FragmentState::Waiting { .. }) => {
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    if remaining.is_zero() {
                        return Err(format!(
                            "timed out after {timeout:?} waiting for fragment instance {id} to \
                             produce rows"
                        ));
                    }
                    guard = self
                        .ready
                        .wait_timeout(guard, remaining)
                        .unwrap_or_else(PoisonError::into_inner)
                        .0;
                }
                Some(FragmentState::Pending(_)) => {
                    let drained = FragmentState::Drained(Instant::now());
                    let Some(FragmentState::Pending(batch)) = guard.by_id.remove(&id) else {
                        unreachable!("state matched Pending")
                    };
                    guard.set(id, drained);
                    return Ok(FetchOutcome {
                        batch: Some(batch),
                        packet_seq: 0,
                        eos: false,
                    });
                }
                Some(FragmentState::Drained(_)) => {
                    return Ok(FetchOutcome {
                        batch: None,
                        packet_seq: 1,
                        eos: true,
                    });
                }
                Some(FragmentState::Failed(error, _)) => return Err(error.clone()),
            }
        }
    }

    /// Locks the slots, recovering from a poisoned mutex (state is disposable result data), and
    /// drops the ended ones past the window or the cap: only those, oldest first.
    fn lock(&self) -> std::sync::MutexGuard<'_, Slots> {
        let window = *self
            .window
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut slots = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        slots.prune(window);
        slots
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn batch(rows: &[&str]) -> TResultBatch {
        TResultBatch::new(
            rows.iter().map(|row| row.as_bytes().to_vec()).collect(),
            false,
            0,
            None,
        )
    }

    #[test]
    fn delivers_rows_once_then_reports_eos_on_repeat_polls() {
        let store = ResultStore::default();
        let id = FragmentInstanceId::from_halves(1, 2);
        store.insert(id, batch(&["a", "b"]));

        let first = store.take_next(id, Duration::ZERO).expect("known fragment");
        assert!(!first.eos);
        assert_eq!(first.batch.unwrap().rows.len(), 2);

        // A drained fragment keeps reporting EOS, never reverting to "unknown".
        let second = store
            .take_next(id, Duration::ZERO)
            .expect("drained fragment still known");
        assert!(second.eos);
        assert!(second.batch.is_none());

        let third = store
            .take_next(id, Duration::ZERO)
            .expect("drained fragment still known");
        assert!(third.eos);
    }

    #[test]
    fn ended_slots_age_out_after_the_window_and_delivered_ones_go_on_a_cancel() {
        let store = ResultStore::default();
        store.set_window(Duration::from_millis(200));
        let (drained, failed) = (
            FragmentInstanceId::from_halves(8, 1),
            FragmentInstanceId::from_halves(8, 2),
        );
        store.insert(drained, batch(&["a"]));
        store.take_next(drained, Duration::ZERO).unwrap();
        store.reserve(failed, FragmentInstanceId::from_halves(8, 0));
        store.fail_query(failed, "scan exploded");
        // Inside the window a repeat poll still reads EOS, and the failure its cause.
        assert!(store.take_next(drained, Duration::ZERO).unwrap().eos);
        assert_eq!(
            store.take_next(failed, Duration::ZERO).unwrap_err(),
            "scan exploded"
        );
        std::thread::sleep(Duration::from_millis(250));
        assert_eq!(store.len(), 0, "both aged out");

        // The FE's cancel drops a delivered slot at once, never a failed one.
        store.set_window(crate::recent_queries::REMEMBER_FOR);
        store.insert(drained, batch(&["a"]));
        store.take_next(drained, Duration::ZERO).unwrap();
        store.reserve(failed, FragmentInstanceId::from_halves(8, 0));
        store.fail_query(failed, "scan exploded");
        store.forget_delivered(FragmentInstanceId::from_halves(8, 0));
        assert!(store.take_next(drained, Duration::ZERO).is_err());
        assert!(store.take_next(failed, Duration::ZERO).is_err());
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn a_failure_frees_rows_nobody_fetched_and_ended_slots_are_capped() {
        let store = ResultStore::default();
        let pending = FragmentInstanceId::from_halves(11, 1);
        store.insert(pending, batch(&["a"]));
        store.fail_query(FragmentInstanceId::from_halves(11, 0), "scan exploded");
        assert_eq!(
            store.take_next(pending, Duration::ZERO).unwrap_err(),
            "scan exploded",
            "the rows are gone; the poll reads the cause"
        );

        // Past the cap the oldest ended slot goes first.
        for lo in 0..=MAX_ENDED_SLOTS as i64 {
            let id = FragmentInstanceId::from_halves(12, lo);
            store.reserve(id, FragmentInstanceId::from_halves(12, 0));
        }
        store.fail_query(FragmentInstanceId::from_halves(12, 0), "boom");
        assert!(store.len() <= MAX_ENDED_SLOTS + 1);
    }

    #[test]
    fn unknown_fragment_is_an_error() {
        let store = ResultStore::default();
        let err = store
            .take_next(FragmentInstanceId::from_halves(9, 9), Duration::ZERO)
            .unwrap_err();
        assert!(err.contains("no buffered result"), "{err}");
    }

    #[test]
    fn reserved_fragment_blocks_until_rows_arrive() {
        let store = std::sync::Arc::new(ResultStore::default());
        let id = FragmentInstanceId::from_halves(3, 4);
        store.reserve(id, FragmentInstanceId::from_halves(3, 0));
        let waiting = store.clone();
        let poll = std::thread::spawn(move || waiting.take_next(id, Duration::from_secs(5)));
        store.insert(id, batch(&["x"]));
        let outcome = poll.join().unwrap().expect("rows arrived");
        assert_eq!(outcome.batch.unwrap().rows.len(), 1);
    }

    #[test]
    fn reserve_keeps_an_existing_slot() {
        let store = ResultStore::default();
        let id = FragmentInstanceId::from_halves(7, 1);
        store.insert(id, batch(&["x"]));
        store.reserve(id, FragmentInstanceId::from_halves(7, 0));
        let outcome = store
            .take_next(id, Duration::ZERO)
            .expect("rows survive a repeated reserve");
        assert_eq!(outcome.batch.unwrap().rows.len(), 1);
    }

    #[test]
    fn failed_query_fails_only_its_waiting_slots() {
        let store = ResultStore::default();
        let (query, other) = (
            FragmentInstanceId::from_halves(5, 0),
            FragmentInstanceId::from_halves(6, 0),
        );
        let (id, other_id) = (
            FragmentInstanceId::from_halves(5, 1),
            FragmentInstanceId::from_halves(6, 1),
        );
        store.reserve(id, query);
        store.reserve(other_id, other);
        // By another instance of the query, as a remote sender's failure frame names it.
        store.fail_query(FragmentInstanceId::from_halves(5, 3), "merge exploded");
        let err = store.take_next(id, Duration::from_secs(5)).unwrap_err();
        assert!(err.contains("merge exploded"), "{err}");
        let err = store.take_next(other_id, Duration::ZERO).unwrap_err();
        assert!(err.contains("timed out"), "{err}");
    }
}
