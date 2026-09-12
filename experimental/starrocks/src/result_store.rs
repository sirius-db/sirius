//! In-memory buffer of executed-fragment results awaiting FE `fetch_data` collection.
//!
//! StarRocks dispatches a fragment with `exec_plan_fragment`, then polls `fetch_data` with the
//! fragment instance id until end-of-stream. This store bridges those two RPCs: the exec handler
//! buffers a fragment's rows here, and each `fetch_data` poll drains them.

use std::fmt;
use std::{
    collections::HashMap,
    sync::{Condvar, Mutex},
    time::Duration,
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
pub(crate) struct FragmentInstanceId(Uuid);

impl FragmentInstanceId {
    /// Packs the `hi`/`lo` 64-bit halves of a StarRocks unique id into a [`Uuid`].
    pub(crate) fn from_halves(hi: i64, lo: i64) -> Self {
        Self(Uuid::from_u64_pair(hi as u64, lo as u64))
    }

    /// The proto `PUniqueId` form used by `fetch_data` and `transmit_chunk`.
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
    /// Result fragment accepted but waiting for its exchange input.
    Waiting,
    /// Rows produced, not yet delivered.
    Pending(TResultBatch),
    /// Rows delivered; the next poll reports end-of-stream.
    Drained,
    /// Execution failed; every poll re-reports the cause so the FE errors instead of waiting.
    Failed(String),
}

/// What a single `fetch_data` poll should return to the FE.
#[derive(Debug)]
pub(crate) enum FetchOutcome {
    /// Result-stream progress for a live fragment.
    Rows {
        /// Result rows to ship as the response attachment, when present.
        batch: Option<TResultBatch>,
        /// Monotonic packet sequence the FE uses to detect lost packets.
        packet_seq: i64,
        /// End-of-stream marker; the FE stops polling once true.
        eos: bool,
    },
    /// The fragment failed; the poll must surface this cause as an error.
    Failed(String),
}

/// Process-wide store of fragment results keyed by fragment instance id.
///
/// Shared across all BRPC connections via an `Arc` inside the compute-node service, so a
/// `fetch_data` poll on one connection sees results buffered by an `exec_plan_fragment` on another.
#[derive(Debug, Default)]
pub(crate) struct ResultStore {
    /// Buffered results keyed by fragment instance id.
    inner: Mutex<HashMap<FragmentInstanceId, FragmentState>>,
    /// Wakes a long-polling `fetch_data` when a waiting fragment gets rows or fails.
    ready: Condvar,
}

impl ResultStore {
    /// Marks an accepted result fragment whose execution is waiting on an exchange sender.
    pub(crate) fn reserve(&self, id: FragmentInstanceId) {
        self.lock().entry(id).or_insert(FragmentState::Waiting);
    }

    /// Buffers an executed fragment's result for later `fetch_data` collection. A recorded
    /// failure sticks: rows landing after a failure must not turn a loud error back into a
    /// silently incomplete result.
    pub(crate) fn insert(&self, id: FragmentInstanceId, batch: TResultBatch) {
        {
            let mut guard = self.lock();
            if !matches!(guard.get(&id), Some(FragmentState::Failed(_))) {
                guard.insert(id, FragmentState::Pending(batch));
            }
        }
        self.ready.notify_all();
    }

    /// Marks a fragment failed so `fetch_data` reports the cause instead of waiting forever.
    pub(crate) fn fail(&self, id: FragmentInstanceId, error: String) {
        self.lock().insert(id, FragmentState::Failed(error));
        self.ready.notify_all();
    }

    /// Blocks until fragment `id` has something to report, then advances the state machine.
    /// A timeout is a loud failure rather than an empty reply (the FE's packet counter desyncs
    /// on not-ready). Unknown ids stay `None`.
    pub(crate) fn wait_ready(
        &self,
        id: FragmentInstanceId,
        timeout: Duration,
    ) -> Option<FetchOutcome> {
        let deadline = std::time::Instant::now() + timeout;
        let mut guard = self.lock();
        while let Some(FragmentState::Waiting) = guard.get(&id) {
            let now = std::time::Instant::now();
            if now >= deadline {
                return Some(FetchOutcome::Failed(format!(
                    "timed out after {timeout:?} waiting for fragment instance {id} to produce \
                     rows (its exchange senders may have stalled)"
                )));
            }
            let (next, wait) = self
                .ready
                .wait_timeout(guard, deadline - now)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            guard = next;
            let _ = wait;
        }
        drop(guard);
        self.take_next(id)
    }

    /// Advances the `fetch_data` state machine for one fragment: deliver rows once, then EOS.
    /// Returns `None` for an id this CN never buffered, which the caller reports as an error
    /// (StarRocks treats a missing result buffer as a failure, not an empty result). A drained
    /// fragment stays in the map so a repeat poll still reports EOS rather than reading as unknown.
    pub(crate) fn take_next(&self, id: FragmentInstanceId) -> Option<FetchOutcome> {
        let mut guard = self.lock();
        match guard.get_mut(&id) {
            None => None,
            Some(FragmentState::Waiting) => Some(FetchOutcome::Rows {
                batch: None,
                packet_seq: 0,
                eos: false,
            }),
            Some(state @ FragmentState::Pending(_)) => {
                let FragmentState::Pending(batch) =
                    std::mem::replace(state, FragmentState::Drained)
                else {
                    unreachable!("state matched Pending")
                };
                Some(FetchOutcome::Rows {
                    batch: Some(batch),
                    packet_seq: 0,
                    eos: false,
                })
            }
            Some(FragmentState::Drained) => Some(FetchOutcome::Rows {
                batch: None,
                packet_seq: 1,
                eos: true,
            }),
            Some(FragmentState::Failed(error)) => Some(FetchOutcome::Failed(error.clone())),
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

        let first = store.take_next(id).expect("known fragment");
        let FetchOutcome::Rows { eos, batch, .. } = first else {
            panic!("expected rows");
        };
        assert!(!eos);
        assert_eq!(batch.unwrap().rows.len(), 2);

        // A drained fragment keeps reporting EOS, never reverting to "unknown".
        let second = store.take_next(id).expect("drained fragment still known");
        let FetchOutcome::Rows { eos, batch, .. } = second else {
            panic!("expected eos rows");
        };
        assert!(eos);
        assert!(batch.is_none());

        let third = store.take_next(id).expect("drained fragment still known");
        assert!(matches!(third, FetchOutcome::Rows { eos: true, .. }));
    }

    #[test]
    fn unknown_fragment_is_none() {
        let store = ResultStore::default();
        assert!(
            store
                .take_next(FragmentInstanceId::from_halves(9, 9))
                .is_none()
        );
    }

    #[test]
    fn reserved_fragment_blocks_until_rows_arrive() {
        let store = std::sync::Arc::new(ResultStore::default());
        let id = FragmentInstanceId::from_halves(3, 4);
        store.reserve(id);
        let waiting = store.clone();
        let thread = std::thread::spawn(move || waiting.wait_ready(id, Duration::from_secs(5)));
        store.insert(id, batch(&["x"]));
        match thread.join().unwrap() {
            Some(FetchOutcome::Rows {
                batch: Some(rows),
                eos: false,
                ..
            }) => assert_eq!(rows.rows.len(), 1),
            other => panic!("expected rows, got {other:?}"),
        }
    }

    #[test]
    fn failed_fragment_surfaces_through_wait_ready() {
        let store = ResultStore::default();
        let id = FragmentInstanceId::from_halves(5, 6);
        store.reserve(id);
        store.fail(id, "merge exploded".to_string());
        match store.wait_ready(id, Duration::from_secs(1)) {
            Some(FetchOutcome::Failed(cause)) => {
                assert!(cause.contains("merge exploded"), "{cause}")
            }
            other => panic!("expected failure, got {other:?}"),
        }
    }
}
