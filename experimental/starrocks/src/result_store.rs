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
pub(crate) struct FragmentInstanceId(Uuid);

impl FragmentInstanceId {
    /// Packs the `hi`/`lo` 64-bit halves of a StarRocks unique id into a [`Uuid`].
    pub(crate) fn from_halves(hi: i64, lo: i64) -> Self {
        Self(Uuid::from_u64_pair(hi as u64, lo as u64))
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
    /// Rows produced, not yet delivered.
    Pending(TResultBatch),
    /// Rows delivered; the next poll reports end-of-stream.
    Drained,
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
}

impl ResultStore {
    /// Buffers an executed fragment's result for later `fetch_data` collection.
    pub(crate) fn insert(&self, id: FragmentInstanceId, batch: TResultBatch) {
        self.lock().insert(id, FragmentState::Pending(batch));
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
            Some(state @ FragmentState::Pending(_)) => {
                let FragmentState::Pending(batch) =
                    std::mem::replace(state, FragmentState::Drained)
                else {
                    unreachable!("state matched Pending")
                };
                Some(FetchOutcome {
                    batch: Some(batch),
                    packet_seq: 0,
                    eos: false,
                })
            }
            Some(FragmentState::Drained) => Some(FetchOutcome {
                batch: None,
                packet_seq: 1,
                eos: true,
            }),
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
        assert!(!first.eos);
        assert_eq!(first.batch.unwrap().rows.len(), 2);

        // A drained fragment keeps reporting EOS, never reverting to "unknown".
        let second = store.take_next(id).expect("drained fragment still known");
        assert!(second.eos);
        assert!(second.batch.is_none());

        let third = store.take_next(id).expect("drained fragment still known");
        assert!(third.eos);
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
}
