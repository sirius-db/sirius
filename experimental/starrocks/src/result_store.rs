//! In-memory buffer of executed-fragment results awaiting FE `fetch_data` collection.
//!
//! StarRocks dispatches a fragment with `exec_plan_fragment`, then polls `fetch_data` with the
//! fragment instance id until end-of-stream. This store bridges those two RPCs: the exec handler
//! buffers a fragment's rows here, and each `fetch_data` poll drains them.

use std::fmt;
use std::{
    collections::HashMap,
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
    /// Rows delivered; the next poll reports end-of-stream.
    Drained,
    /// The query failed; every poll reports the cause.
    Failed(String),
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

/// Process-wide store of fragment results keyed by fragment instance id.
///
/// Shared across all BRPC connections via an `Arc` inside the compute-node service, so a
/// `fetch_data` poll on one connection sees results buffered by an `exec_plan_fragment` on another.
#[derive(Debug, Default)]
pub(crate) struct ResultStore {
    /// Buffered results keyed by fragment instance id.
    inner: Mutex<HashMap<FragmentInstanceId, FragmentState>>,
    /// Wakes a `take_next` blocked on a waiting fragment.
    ready: Condvar,
}

impl ResultStore {
    /// Reserves the slot of a result fragment that waits on exchange inputs, so `fetch_data`
    /// waits for its rows instead of reporting an unknown fragment. A slot that already exists is
    /// kept, so a repeated dispatch cannot hide rows or a failure behind a fresh wait.
    pub(crate) fn reserve(&self, id: FragmentInstanceId, query: FragmentInstanceId) {
        self.lock()
            .entry(id)
            .or_insert(FragmentState::Waiting { query });
    }

    /// Buffers an executed fragment's result for later `fetch_data` collection.
    pub(crate) fn insert(&self, id: FragmentInstanceId, batch: TResultBatch) {
        self.lock().insert(id, FragmentState::Pending(batch));
        self.ready.notify_all();
    }

    /// Fails every result slot of `query` still waiting, so `fetch_data` reports `error` at once
    /// instead of waiting out its timeout.
    pub(crate) fn fail_query(&self, query: FragmentInstanceId, error: &str) {
        for state in self.lock().values_mut() {
            if matches!(state, FragmentState::Waiting { query: waiting } if *waiting == query) {
                *state = FragmentState::Failed(error.to_string());
            }
        }
        self.ready.notify_all();
    }

    /// Advances the `fetch_data` state machine for one fragment: deliver rows once, then EOS.
    /// A fragment still waiting on its exchange inputs blocks the caller for up to `timeout`;
    /// replying not-ready instead would desync the FE packet counter. An id this CN never buffered
    /// is an error (StarRocks treats a missing result buffer as a failure, not an empty result),
    /// as are a failed query and a timed-out wait. A drained fragment stays in the map so a repeat
    /// poll still reports EOS rather than reading as unknown.
    ///
    /// TODO(starrocks-execute): this is a single-batch, single-poller model. The real executor
    /// needs (a) chunked/streamed delivery of many batches, (b) safety against duplicate or
    /// concurrent polls (advance state only after the response is written; keep a per-fragment
    /// in-flight guard), and (c) eviction of drained entries on `cancel_plan_fragment`/timeout so
    /// the map does not grow for the process lifetime.
    pub(crate) fn take_next(
        &self,
        id: FragmentInstanceId,
        timeout: Duration,
    ) -> Result<FetchOutcome, String> {
        let deadline = Instant::now() + timeout;
        let mut guard = self.lock();
        loop {
            match guard.get_mut(&id) {
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
                Some(state @ FragmentState::Pending(_)) => {
                    let FragmentState::Pending(batch) =
                        std::mem::replace(state, FragmentState::Drained)
                    else {
                        unreachable!("state matched Pending")
                    };
                    return Ok(FetchOutcome {
                        batch: Some(batch),
                        packet_seq: 0,
                        eos: false,
                    });
                }
                Some(FragmentState::Drained) => {
                    return Ok(FetchOutcome {
                        batch: None,
                        packet_seq: 1,
                        eos: true,
                    });
                }
                Some(FragmentState::Failed(error)) => return Err(error.clone()),
            }
        }
    }

    /// Locks the inner map, recovering from a poisoned mutex (state is disposable result data).
    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<FragmentInstanceId, FragmentState>> {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
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
        store.fail_query(query, "merge exploded");
        let err = store.take_next(id, Duration::from_secs(5)).unwrap_err();
        assert!(err.contains("merge exploded"), "{err}");
        let err = store.take_next(other_id, Duration::ZERO).unwrap_err();
        assert!(err.contains("timed out"), "{err}");
    }
}
