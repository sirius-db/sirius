//! In-memory buffer of executed-fragment results awaiting FE `fetch_data` collection.
//!
//! StarRocks dispatches a fragment with `exec_plan_fragment`, then polls `fetch_data` with the
//! fragment instance id until end-of-stream. This store bridges those two RPCs: the exec handler
//! buffers a fragment's rows here, and each `fetch_data` poll drains them.

use std::fmt;
use std::{collections::HashMap, sync::Mutex};

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

    /// The `hi`/`lo` halves back out, for proto messages that carry them (`PUniqueId`).
    #[cfg_attr(not(feature = "nixl-transport"), allow(dead_code))]
    pub(crate) fn as_halves(&self) -> (i64, i64) {
        let (hi, lo) = self.0.as_u64_pair();
        (hi as i64, lo as i64)
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
    /// Execution failed (e.g. on the dispatch worker); every poll re-reports the cause so the
    /// FE errors instead of polling a `Waiting` entry forever.
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
}

impl ResultStore {
    /// Marks an accepted result fragment whose execution is waiting on an exchange sender.
    pub(crate) fn reserve(&self, id: FragmentInstanceId) {
        self.lock().entry(id).or_insert(FragmentState::Waiting);
    }

    /// Buffers an executed fragment's result for later `fetch_data` collection.
    pub(crate) fn insert(&self, id: FragmentInstanceId, batch: TResultBatch) {
        self.lock().insert(id, FragmentState::Pending(batch));
    }

    /// Marks a fragment failed so `fetch_data` reports the cause instead of waiting forever.
    /// The failure sticks: like `Drained`, repeat polls keep re-reporting it rather than
    /// reverting to "unknown fragment".
    pub(crate) fn fail(&self, id: FragmentInstanceId, error: String) {
        self.lock().insert(id, FragmentState::Failed(error));
    }

    /// Advances the `fetch_data` state machine for one fragment: deliver rows once, then EOS.
    /// Returns `None` for an id this CN never buffered, which the caller reports as an error
    /// (StarRocks treats a missing result buffer as a failure, not an empty result). A drained
    /// fragment stays in the map so a repeat poll still reports EOS rather than reading as unknown.
    ///
    /// TODO(starrocks-execute): this is a single-batch, single-poller model. The real executor
    /// needs (a) chunked/streamed delivery of many batches, (b) safety against duplicate or
    /// concurrent polls (advance state only after the response is written; keep a per-fragment
    /// in-flight guard), and (c) eviction of drained entries on `cancel_plan_fragment`/timeout so
    /// the map does not grow for the process lifetime.
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

    /// Destructures a poll that must be live result-stream progress, not a failure.
    fn rows(outcome: FetchOutcome) -> (Option<TResultBatch>, bool) {
        match outcome {
            FetchOutcome::Rows { batch, eos, .. } => (batch, eos),
            FetchOutcome::Failed(error) => panic!("expected rows, got failure: {error}"),
        }
    }

    #[test]
    fn delivers_rows_once_then_reports_eos_on_repeat_polls() {
        let store = ResultStore::default();
        let id = FragmentInstanceId::from_halves(1, 2);
        store.insert(id, batch(&["a", "b"]));

        let (first_batch, first_eos) = rows(store.take_next(id).expect("known fragment"));
        assert!(!first_eos);
        assert_eq!(first_batch.unwrap().rows.len(), 2);

        // A drained fragment keeps reporting EOS, never reverting to "unknown".
        let (second_batch, second_eos) =
            rows(store.take_next(id).expect("drained fragment still known"));
        assert!(second_eos);
        assert!(second_batch.is_none());

        let (_, third_eos) = rows(store.take_next(id).expect("drained fragment still known"));
        assert!(third_eos);
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
    fn reserved_fragment_reports_not_ready_until_rows_arrive() {
        let store = ResultStore::default();
        let id = FragmentInstanceId::from_halves(4, 2);
        store.reserve(id);

        let (waiting_batch, waiting_eos) =
            rows(store.take_next(id).expect("reserved fragment is known"));
        assert!(!waiting_eos);
        assert!(waiting_batch.is_none());

        store.insert(id, batch(&["ready"]));
        let (ready_batch, ready_eos) =
            rows(store.take_next(id).expect("completed fragment is known"));
        assert!(!ready_eos);
        assert_eq!(ready_batch.unwrap().rows.len(), 1);
    }

    #[test]
    fn failed_fragment_reports_its_cause_on_every_poll() {
        let store = ResultStore::default();
        let id = FragmentInstanceId::from_halves(5, 5);
        store.reserve(id);
        store.fail(id, "receiver exploded".to_string());

        // The failure sticks across polls, mirroring Drained: the FE must never see the
        // fragment revert to "waiting" or "unknown" after its execution failed.
        for _ in 0..2 {
            match store.take_next(id).expect("failed fragment is known") {
                FetchOutcome::Failed(error) => assert_eq!(error, "receiver exploded"),
                other => panic!("expected failure, got {other:?}"),
            }
        }
    }
}
