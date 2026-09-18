//! Buffer of executed-query results awaiting FE `fetch_data` collection.
//!
//! Doris dispatches a query with `exec_plan_fragment`, then drives `fetch_data` on the
//! result fragment's backend until end-of-stream. The FE's `ResultReceiver` keys the poll by
//! `query_id` when `enable_parallel_result_sink` is on (the default) and by the result
//! fragment's instance id otherwise, so each result is registered under both. The receiver
//! expects `packet_seq` to start at 0 and increase by exactly one per response (data or the
//! final EOS), and a real BE **parks** a poll whose data is not ready yet instead of answering
//! empty (`ResultBlockBuffer::get_batch` → `_waiting_rpc`); this store mirrors that with an
//! async wait.

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex};

use doris_proto::PUniqueId;
use doris_thrift::data::TResultBatch;
use doris_thrift::types::TUniqueId;
use tokio::sync::Notify;
use uuid::Uuid;

/// A Doris 128-bit unique id (`TUniqueId` on dispatch, `PUniqueId` on `fetch_data`), held as
/// a [`Uuid`] so the two wire forms compare equal.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct UniqueId(Uuid);

impl UniqueId {
    /// Packs the `hi`/`lo` 64-bit halves of a Doris unique id into a [`Uuid`].
    pub(crate) fn from_halves(hi: i64, lo: i64) -> Self {
        Self(Uuid::from_u64_pair(hi as u64, lo as u64))
    }
}

impl From<&TUniqueId> for UniqueId {
    fn from(id: &TUniqueId) -> Self {
        Self::from_halves(id.hi, id.lo)
    }
}

impl From<&PUniqueId> for UniqueId {
    fn from(id: &PUniqueId) -> Self {
        Self::from_halves(id.hi, id.lo)
    }
}

impl fmt::Display for UniqueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (hi, lo) = self.0.as_u64_pair();
        write!(f, "{hi:x}-{lo:x}")
    }
}

/// What a single `fetch_data` poll should return to the FE.
#[derive(Debug, PartialEq)]
pub(crate) enum FetchOutcome {
    /// Result rows to ship in `row_batch`, tagged with their packet sequence.
    Data {
        batch: TResultBatch,
        packet_seq: i64,
    },
    /// End of stream; the FE stops polling once it sees this.
    Eos { packet_seq: i64, returned_rows: i64 },
    /// The query failed; the FE reports the message to the client.
    Failed(String),
}

/// Per-query result state: queued batches, the producer's close status, and the FE's packet
/// sequence.
#[derive(Debug, Default)]
struct SlotState {
    batches: VecDeque<TResultBatch>,
    /// `Some(Ok(rows))` once the producer finished, `Some(Err)` once it failed.
    closed: Option<Result<i64, String>>,
    packet_seq: i64,
    returned_rows: i64,
}

/// Producer/consumer handle for one query's results.
#[derive(Debug, Default)]
pub(crate) struct ResultSlot {
    state: Mutex<SlotState>,
    /// Wakes parked `fetch_data` polls when a batch arrives or the producer closes.
    notify: Notify,
}

impl ResultSlot {
    /// Queues one batch of rows for delivery.
    pub(crate) fn push(&self, batch: TResultBatch) {
        {
            let mut state = self.lock();
            state.returned_rows += batch.rows.len() as i64;
            state.batches.push_back(batch);
        }
        self.notify.notify_waiters();
    }

    /// Marks the producer finished; polls drain the remaining batches, then see EOS.
    pub(crate) fn close(&self) {
        {
            let mut state = self.lock();
            if state.closed.is_none() {
                state.closed = Some(Ok(state.returned_rows));
            }
        }
        self.notify.notify_waiters();
    }

    /// Marks the producer failed; the next poll (and every later one) reports the error.
    pub(crate) fn fail(&self, message: impl Into<String>) {
        {
            let mut state = self.lock();
            state.closed = Some(Err(message.into()));
        }
        self.notify.notify_waiters();
    }

    /// Advances the `fetch_data` state machine by one response, waiting while nothing is
    /// ready. Data is delivered in order; once the producer closed and the queue drained,
    /// every poll reports EOS with the next sequence number (the FE re-issues one poll after
    /// EOS in some paths, which must not read as an error).
    pub(crate) async fn fetch(&self) -> FetchOutcome {
        loop {
            // Register interest before checking state so a push/close between the check and
            // the await cannot be missed.
            let notified = self.notify.notified();
            {
                let mut state = self.lock();
                if let Some(Err(message)) = &state.closed {
                    return FetchOutcome::Failed(message.clone());
                }
                if let Some(mut batch) = state.batches.pop_front() {
                    let packet_seq = state.packet_seq;
                    state.packet_seq += 1;
                    batch.packet_seq = packet_seq;
                    return FetchOutcome::Data { batch, packet_seq };
                }
                if let Some(Ok(returned_rows)) = state.closed {
                    let packet_seq = state.packet_seq;
                    state.packet_seq += 1;
                    return FetchOutcome::Eos {
                        packet_seq,
                        returned_rows,
                    };
                }
            }
            notified.await;
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, SlotState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// Process-wide registry of result slots keyed by query id and result instance id.
///
/// Shared across gRPC connections via an `Arc` inside the backend service, so a `fetch_data`
/// on one connection sees what an `exec_plan_fragment` on another registered.
#[derive(Debug, Default)]
pub(crate) struct ResultStore {
    inner: Mutex<HashMap<UniqueId, Arc<ResultSlot>>>,
}

impl ResultStore {
    /// Registers a slot for `query_id` (and the result fragment's `instance_ids`) before
    /// execution starts, so a poll arriving early parks instead of failing as unknown.
    /// Re-registering a query replaces its previous slot.
    pub(crate) fn register(
        &self,
        query_id: UniqueId,
        instance_ids: impl IntoIterator<Item = UniqueId>,
    ) -> Arc<ResultSlot> {
        let slot = Arc::new(ResultSlot::default());
        let mut map = self.lock();
        map.insert(query_id, slot.clone());
        for instance_id in instance_ids {
            map.insert(instance_id, slot.clone());
        }
        slot
    }

    /// Looks up the slot a `fetch_data` poll refers to (query id or instance id).
    pub(crate) fn get(&self, id: UniqueId) -> Option<Arc<ResultSlot>> {
        self.lock().get(&id).cloned()
    }

    /// Drops every key pointing at the slot registered for `id` (query cancelled/finished).
    /// A poll parked on the slot is failed so the FE stops waiting.
    pub(crate) fn evict(&self, id: UniqueId) {
        let mut map = self.lock();
        let Some(slot) = map.remove(&id) else {
            return;
        };
        map.retain(|_, other| !Arc::ptr_eq(other, &slot));
        let mut state = slot.lock();
        if state.closed.is_none() {
            state.closed = Some(Err("query was cancelled".to_string()));
        }
        drop(state);
        slot.notify.notify_waiters();
    }

    /// Number of registered keys.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.lock().len()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<UniqueId, Arc<ResultSlot>>> {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    fn batch(rows: &[&str]) -> TResultBatch {
        TResultBatch::new(
            rows.iter().map(|row| row.as_bytes().to_vec()).collect(),
            false,
            0,
            None,
        )
    }

    #[tokio::test]
    async fn delivers_batches_in_order_then_eos_with_consecutive_packet_seqs() {
        let store = ResultStore::default();
        let query = UniqueId::from_halves(1, 2);
        let instance = UniqueId::from_halves(1, 3);
        let slot = store.register(query, [instance]);
        slot.push(batch(&["a", "b"]));
        slot.push(batch(&["c"]));
        slot.close();

        // Either key reaches the same slot.
        let by_instance = store.get(instance).unwrap();
        assert!(Arc::ptr_eq(&by_instance, &slot));

        match by_instance.fetch().await {
            FetchOutcome::Data { batch, packet_seq } => {
                assert_eq!(packet_seq, 0);
                assert_eq!(batch.packet_seq, 0);
                assert_eq!(batch.rows.len(), 2);
            }
            other => panic!("{other:?}"),
        }
        match slot.fetch().await {
            FetchOutcome::Data { batch, packet_seq } => {
                assert_eq!(packet_seq, 1);
                assert_eq!(batch.rows.len(), 1);
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(
            slot.fetch().await,
            FetchOutcome::Eos {
                packet_seq: 2,
                returned_rows: 3
            }
        );
        // A repeat poll after EOS keeps counting instead of reading as unknown.
        assert_eq!(
            slot.fetch().await,
            FetchOutcome::Eos {
                packet_seq: 3,
                returned_rows: 3
            }
        );
    }

    #[tokio::test]
    async fn poll_parks_until_data_or_close_arrives() {
        let store = ResultStore::default();
        let slot = store.register(UniqueId::from_halves(1, 2), []);
        let waiter = slot.clone();
        let poll = tokio::spawn(async move { waiter.fetch().await });

        // Nothing is ready: the poll must still be pending after a grace period.
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(!poll.is_finished());

        slot.push(batch(&["late"]));
        match tokio::time::timeout(Duration::from_secs(5), poll)
            .await
            .unwrap()
            .unwrap()
        {
            FetchOutcome::Data { batch, packet_seq } => {
                assert_eq!(packet_seq, 0);
                assert_eq!(batch.rows, vec![b"late".to_vec()]);
            }
            other => panic!("{other:?}"),
        }

        let waiter = slot.clone();
        let poll = tokio::spawn(async move { waiter.fetch().await });
        tokio::time::sleep(Duration::from_millis(50)).await;
        slot.close();
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(5), poll)
                .await
                .unwrap()
                .unwrap(),
            FetchOutcome::Eos {
                packet_seq: 1,
                returned_rows: 1
            }
        );
    }

    #[tokio::test]
    async fn failure_is_reported_to_parked_and_later_polls() {
        let store = ResultStore::default();
        let slot = store.register(UniqueId::from_halves(1, 2), []);
        let waiter = slot.clone();
        let poll = tokio::spawn(async move { waiter.fetch().await });
        tokio::time::sleep(Duration::from_millis(50)).await;
        slot.fail("boom");
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(5), poll)
                .await
                .unwrap()
                .unwrap(),
            FetchOutcome::Failed("boom".to_string())
        );
        assert_eq!(slot.fetch().await, FetchOutcome::Failed("boom".to_string()));
    }

    #[tokio::test]
    async fn evict_removes_all_keys_and_fails_parked_polls() {
        let store = ResultStore::default();
        let query = UniqueId::from_halves(1, 2);
        let slot = store.register(query, [UniqueId::from_halves(1, 3)]);
        assert_eq!(store.len(), 2);
        let waiter = slot.clone();
        let poll = tokio::spawn(async move { waiter.fetch().await });
        tokio::time::sleep(Duration::from_millis(50)).await;

        store.evict(UniqueId::from_halves(1, 3));

        assert_eq!(store.len(), 0);
        assert!(store.get(query).is_none());
        assert!(matches!(
            tokio::time::timeout(Duration::from_secs(5), poll)
                .await
                .unwrap()
                .unwrap(),
            FetchOutcome::Failed(_)
        ));
    }

    #[test]
    fn unique_id_display_matches_fe_format() {
        assert_eq!(UniqueId::from_halves(0x1a2b, 0x3c).to_string(), "1a2b-3c");
        assert_eq!(
            UniqueId::from(&TUniqueId::new(7, 8)),
            UniqueId::from(&PUniqueId { hi: 7, lo: 8 })
        );
    }
}
