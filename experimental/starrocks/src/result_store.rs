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

/// Result-stream progress for a live fragment, in the shape dev's `fetch_data` handler reads
/// today (`take_next`). The dispatch layer switches that handler to [`ResultStore::wait_ready`]
/// and [`FetchOutcome`], which also carry a failure, and retires this struct with it.
#[derive(Debug)]
pub(crate) struct FetchProgress {
    /// Result rows to ship as the response attachment, when present.
    pub(crate) batch: Option<TResultBatch>,
    /// Monotonic packet sequence the FE uses to detect lost packets.
    pub(crate) packet_seq: i64,
    /// End-of-stream marker; the FE stops polling once true.
    pub(crate) eos: bool,
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
    // Read by the dispatch layer's fetch_data (stacked/cn-exchange-dispatch); dev's take_next drops it.
    #[allow(dead_code)]
    Failed(String),
}

/// Everything behind the store's one mutex: buffered fragment states plus the per-query
/// bookkeeping that lets any fragment failure reach the result instances the FE polls.
#[derive(Debug, Default)]
struct StoreState {
    /// Buffered results keyed by fragment instance id.
    fragments: HashMap<FragmentInstanceId, FragmentState>,
    /// Result-fragment instances reserved per query, so a failure anywhere in the query can
    /// fail every id the FE is polling on this CN.
    query_results: HashMap<FragmentInstanceId, Vec<FragmentInstanceId>>,
    /// First failure recorded per query. A result fragment that reserves *after* the failure
    /// landed fails immediately instead of waiting on senders that will never deliver.
    query_failures: HashMap<FragmentInstanceId, String>,
}

/// Process-wide store of fragment results keyed by fragment instance id.
///
/// Shared across all BRPC connections via an `Arc` inside the compute-node service, so a
/// `fetch_data` poll on one connection sees results buffered by an `exec_plan_fragment` on another.
#[derive(Debug, Default)]
pub(crate) struct ResultStore {
    /// Fragment states plus per-query failure bookkeeping (see [`StoreState`]).
    inner: Mutex<StoreState>,
    /// Signalled whenever a fragment leaves `Waiting` (rows buffered or failure recorded), so a
    /// long-polling `fetch_data` can block instead of replying not-ready. An empty reply is not
    /// harmless: the FE's ResultReceiver counts every packet, so a not-ready reply consumes a
    /// sequence number and the rows that follow arrive with a stale one ("expect=1, receive=0").
    ready: std::sync::Condvar,
}

impl ResultStore {
    /// Marks an accepted result fragment whose execution is waiting on an exchange sender,
    /// remembering which query it belongs to. If that query already failed (an intermediate
    /// fragment can fail before the FE dispatches the result fragment), the reservation lands
    /// as the failure so the very first `fetch_data` poll reports it.
    // Called by the dispatch worker (stacked/cn-exchange-dispatch); nothing on this base reserves.
    #[allow(dead_code)]
    pub(crate) fn reserve(&self, id: FragmentInstanceId, query_id: FragmentInstanceId) {
        let mut state = self.lock();
        let results = state.query_results.entry(query_id).or_default();
        if !results.contains(&id) {
            results.push(id);
        }
        if let Some(cause) = state.query_failures.get(&query_id).cloned() {
            state.fragments.insert(id, FragmentState::Failed(cause));
            drop(state);
            self.ready.notify_all();
            return;
        }
        state.fragments.entry(id).or_insert(FragmentState::Waiting);
    }

    /// Buffers an executed fragment's result for later `fetch_data` collection. A recorded
    /// failure sticks: rows landing after the query failed elsewhere must not turn a loud
    /// error back into a silently incomplete result.
    pub(crate) fn insert(&self, id: FragmentInstanceId, batch: TResultBatch) {
        let mut state = self.lock();
        if !matches!(state.fragments.get(&id), Some(FragmentState::Failed(_))) {
            state.fragments.insert(id, FragmentState::Pending(batch));
        }
        drop(state);
        self.ready.notify_all();
    }

    /// Marks a fragment failed so `fetch_data` reports the cause instead of waiting forever.
    /// The failure sticks: like `Drained`, repeat polls keep re-reporting it rather than
    /// reverting to "unknown fragment".
    // Called by the dispatch worker (stacked/cn-exchange-dispatch); nothing on this base fails.
    #[allow(dead_code)]
    pub(crate) fn fail(&self, id: FragmentInstanceId, error: String) {
        self.lock()
            .fragments
            .insert(id, FragmentState::Failed(error));
        self.ready.notify_all();
    }

    /// Fails fragment `failed_id` and every result-fragment instance reserved for `query_id`,
    /// waking any `fetch_data` long-poll. Also records the failure at query level so a result
    /// fragment that reserves later fails on arrival. Without this, an intermediate fragment
    /// failing before the result fragment registers would be lost and the FE would poll until
    /// its timeout. The propagated message carries the original fragment's error verbatim.
    // Called by the dispatch worker (stacked/cn-exchange-dispatch); nothing on this base fails.
    #[allow(dead_code)]
    pub(crate) fn fail_query(
        &self,
        query_id: FragmentInstanceId,
        failed_id: FragmentInstanceId,
        error: String,
    ) {
        let mut state = self.lock();
        let cause = format!("fragment instance {failed_id} failed: {error}");
        state
            .fragments
            .insert(failed_id, FragmentState::Failed(error));
        // First failure wins: later failures are usually downstream echoes of the first.
        let cause = state
            .query_failures
            .entry(query_id)
            .or_insert(cause)
            .clone();
        let results = state
            .query_results
            .get(&query_id)
            .cloned()
            .unwrap_or_default();
        for result_id in results {
            if result_id == failed_id
                || matches!(
                    state.fragments.get(&result_id),
                    Some(FragmentState::Failed(_))
                )
            {
                continue;
            }
            state
                .fragments
                .insert(result_id, FragmentState::Failed(cause.clone()));
        }
        drop(state);
        self.ready.notify_all();
    }

    /// Best-effort cancellation mark: a still-`Waiting` entry becomes a failure so a
    /// `fetch_data` long-poll returns instead of blocking out its timeout. Delivered,
    /// drained, or already-failed entries keep their state; unknown ids are ignored.
    // Called from cancel_plan_fragment in the dispatch layer (stacked/cn-exchange-dispatch).
    #[allow(dead_code)]
    pub(crate) fn cancel(&self, id: FragmentInstanceId, reason: String) {
        let mut state = self.lock();
        if let Some(entry @ FragmentState::Waiting) = state.fragments.get_mut(&id) {
            *entry = FragmentState::Failed(reason);
            drop(state);
            self.ready.notify_all();
        }
    }

    /// Blocks until fragment `id` has something to report, then advances the state machine.
    /// This is the long-poll `fetch_data` needs once receivers execute on the dispatch thread.
    /// The stock BE holds the rpc open until the sink produces; replying not-ready instead
    /// desyncs the FE's packet counter (see `ready`). A timeout is a *loud* failure rather than
    /// an empty reply, for the same reason.
    // The dispatch layer (stacked/cn-exchange-dispatch) moves fetch_data onto this long-poll.
    #[allow(dead_code)]
    pub(crate) fn wait_ready(
        &self,
        id: FragmentInstanceId,
        timeout: std::time::Duration,
    ) -> Option<FetchOutcome> {
        let deadline = std::time::Instant::now() + timeout;
        let mut guard = self.lock();
        while let Some(FragmentState::Waiting) = guard.fragments.get(&id) {
            let now = std::time::Instant::now();
            if now >= deadline {
                return Some(FetchOutcome::Failed(format!(
                    "timed out after {timeout:?} waiting for fragment instance {id} to produce \
                     rows (its exchange senders may have stalled)"
                )));
            }
            let (g, wait) = self
                .ready
                .wait_timeout(guard, deadline - now)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            guard = g;
            let _ = wait;
        }
        drop(guard);
        self.poll(id)
    }

    /// Advances the `fetch_data` state machine for one fragment without blocking: not-ready
    /// while `Waiting`, deliver rows once, then EOS, and the recorded cause once `Failed`.
    /// Returns `None` for an id this CN never buffered, which the caller reports as an error
    /// (StarRocks treats a missing result buffer as a failure, not an empty result). A drained
    /// fragment stays in the map so a repeat poll still reports EOS rather than reading as unknown.
    ///
    /// TODO(starrocks-execute): this is a single-batch, single-poller model. The real executor
    /// needs (a) chunked/streamed delivery of many batches, (b) safety against duplicate or
    /// concurrent polls (advance state only after the response is written; keep a per-fragment
    /// in-flight guard), and (c) eviction of drained entries (and the per-query bookkeeping) on
    /// `cancel_plan_fragment`/timeout so the maps do not grow for the process lifetime.
    pub(crate) fn poll(&self, id: FragmentInstanceId) -> Option<FetchOutcome> {
        let mut guard = self.lock();
        match guard.fragments.get_mut(&id) {
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

    /// Dev's `fetch_data` poll: [`Self::poll`] narrowed to the rows-only shape that handler
    /// reads. A `Failed` entry reads as `None` here, so the handler still answers with its
    /// "no buffered result" error rather than hanging; the cause text reaches the FE once the
    /// dispatch layer moves `fetch_data` onto [`Self::wait_ready`]. Nothing on this base records
    /// a failure, so that narrowing is unreachable outside tests until then.
    pub(crate) fn take_next(&self, id: FragmentInstanceId) -> Option<FetchProgress> {
        match self.poll(id)? {
            FetchOutcome::Rows {
                batch,
                packet_seq,
                eos,
            } => Some(FetchProgress {
                batch,
                packet_seq,
                eos,
            }),
            FetchOutcome::Failed(_) => None,
        }
    }

    /// Locks the inner state, recovering from a poisoned mutex (state is disposable result data).
    fn lock(&self) -> std::sync::MutexGuard<'_, StoreState> {
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

    /// Destructures a poll that must be a failure, returning its cause.
    fn failure(outcome: FetchOutcome) -> String {
        match outcome {
            FetchOutcome::Failed(error) => error,
            other => panic!("expected failure, got {other:?}"),
        }
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

    #[test]
    fn take_next_reads_a_failed_entry_as_none() {
        let store = ResultStore::default();
        let id = FragmentInstanceId::from_halves(3, 3);
        store.fail(id, "receiver exploded".to_string());

        // Dev's rows-only poll has no failure channel; it must not hang or fabricate rows.
        assert!(store.take_next(id).is_none());
        assert_eq!(failure(store.poll(id).expect("known")), "receiver exploded");
    }

    /// A query id for tests that only need the reserve association, not failure propagation.
    fn query(hi: i64) -> FragmentInstanceId {
        FragmentInstanceId::from_halves(hi, 0)
    }

    #[test]
    fn reserved_fragment_reports_not_ready_until_rows_arrive() {
        let store = ResultStore::default();
        let id = FragmentInstanceId::from_halves(4, 2);
        store.reserve(id, query(4));

        let (waiting_batch, waiting_eos) =
            rows(store.poll(id).expect("reserved fragment is known"));
        assert!(!waiting_eos);
        assert!(waiting_batch.is_none());

        store.insert(id, batch(&["ready"]));
        let (ready_batch, ready_eos) = rows(store.poll(id).expect("completed fragment is known"));
        assert!(!ready_eos);
        assert_eq!(ready_batch.unwrap().rows.len(), 1);
    }

    #[test]
    fn wait_ready_blocks_until_rows_arrive() {
        use std::sync::Arc;
        let store = Arc::new(ResultStore::default());
        let id = FragmentInstanceId::from_halves(9, 1);
        store.reserve(id, query(9));

        // Rows land from another thread while the poll is blocked, the dispatch-worker shape.
        let producer = {
            let store = Arc::clone(&store);
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(50));
                store.insert(id, batch(&["late"]));
            })
        };
        let (rows_batch, eos) = rows(
            store
                .wait_ready(id, std::time::Duration::from_secs(5))
                .expect("reserved fragment is known"),
        );
        producer.join().expect("producer thread");
        assert!(!eos);
        assert_eq!(rows_batch.expect("rows delivered").rows.len(), 1);
    }

    #[test]
    fn wait_ready_times_out_loudly_instead_of_replying_not_ready() {
        let store = ResultStore::default();
        let id = FragmentInstanceId::from_halves(9, 2);
        store.reserve(id, query(9));

        match store.wait_ready(id, std::time::Duration::from_millis(20)) {
            Some(FetchOutcome::Failed(cause)) => {
                assert!(cause.contains("timed out"), "cause: {cause}")
            }
            other => panic!("expected a loud timeout failure, got {other:?}"),
        }
    }

    #[test]
    fn failed_fragment_reports_its_cause_on_every_poll() {
        let store = ResultStore::default();
        let id = FragmentInstanceId::from_halves(5, 5);
        store.reserve(id, query(5));
        store.fail(id, "receiver exploded".to_string());

        // The failure sticks across polls, mirroring Drained: the FE must never see the
        // fragment revert to "waiting" or "unknown" after its execution failed.
        for _ in 0..2 {
            match store.poll(id).expect("failed fragment is known") {
                FetchOutcome::Failed(error) => assert_eq!(error, "receiver exploded"),
                other => panic!("expected failure, got {other:?}"),
            }
        }
    }

    #[test]
    fn fail_query_fails_every_reserved_result_instance() {
        let store = ResultStore::default();
        let result_id = FragmentInstanceId::from_halves(6, 1);
        let intermediate_id = FragmentInstanceId::from_halves(6, 2);
        store.reserve(result_id, query(6));

        store.fail_query(query(6), intermediate_id, "relay guard fired".to_string());

        // The failing instance carries its raw error; the result instance names the origin.
        let intermediate = failure(store.poll(intermediate_id).expect("known"));
        assert_eq!(intermediate, "relay guard fired");
        let result = failure(store.poll(result_id).expect("known"));
        assert!(
            result.contains(&intermediate_id.to_string()) && result.contains("relay guard fired"),
            "cause: {result}"
        );
    }

    #[test]
    fn query_failure_recorded_before_reserve_fails_the_result_instance_on_arrival() {
        let store = ResultStore::default();
        let result_id = FragmentInstanceId::from_halves(7, 1);
        let intermediate_id = FragmentInstanceId::from_halves(7, 2);

        // The intermediate fragment fails before the FE's result fragment ever registers.
        store.fail_query(
            query(7),
            intermediate_id,
            "sender never delivered".to_string(),
        );
        store.reserve(result_id, query(7));

        let cause = failure(
            store
                .wait_ready(result_id, std::time::Duration::from_secs(5))
                .expect("known"),
        );
        assert!(cause.contains("sender never delivered"), "cause: {cause}");
    }

    #[test]
    fn rows_arriving_after_a_query_failure_do_not_mask_it() {
        let store = ResultStore::default();
        let result_id = FragmentInstanceId::from_halves(8, 1);
        store.reserve(result_id, query(8));
        store.fail_query(
            query(8),
            FragmentInstanceId::from_halves(8, 2),
            "boom".to_string(),
        );

        store.insert(result_id, batch(&["late rows"]));

        let cause = failure(store.poll(result_id).expect("known"));
        assert!(cause.contains("boom"), "cause: {cause}");
    }

    #[test]
    fn cancel_fails_only_a_waiting_entry() {
        let store = ResultStore::default();
        let waiting = FragmentInstanceId::from_halves(11, 1);
        store.reserve(waiting, query(11));
        store.cancel(waiting, "cancelled by the FE".to_string());
        let cause = failure(store.poll(waiting).expect("known"));
        assert!(cause.contains("cancelled"), "cause: {cause}");

        // Delivered rows keep flowing; cancel must not clobber them.
        let delivered = FragmentInstanceId::from_halves(11, 2);
        store.insert(delivered, batch(&["row"]));
        store.cancel(delivered, "cancelled by the FE".to_string());
        let (rows_batch, eos) = rows(store.poll(delivered).expect("known"));
        assert!(!eos);
        assert_eq!(rows_batch.expect("rows survive cancel").rows.len(), 1);

        // Unknown ids stay unknown: cancel must not fabricate result entries.
        let unknown = FragmentInstanceId::from_halves(11, 3);
        store.cancel(unknown, "cancelled by the FE".to_string());
        assert!(store.poll(unknown).is_none());
    }
}
