//! Sequential exchange rendezvous.
//!
//! Matches StarRocks' receiver-first dispatch with senders that arrive later. A same-CN sender
//! parks native engine output as a [`SenderSlot`]; a remote sender's batches arrive as
//! [`RemoteBatch`]es and sit in [`SenderSource::Remote`] until eos. The receiver becomes ready
//! when every expected sender is complete. There is no fusion: every leaf runs, then hops, then
//! the root.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Mutex;

use starrocks_thrift::internal_service::TExecPlanFragmentParams;
use tracing::info;

use crate::fragment_executor::{KeySource, SenderSlot};
use crate::result_store::FragmentInstanceId;

/// Receiver identity used by both the stream sink and the exchange node.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ExchangeKey {
    pub(crate) fragment_instance_id: FragmentInstanceId,
    pub(crate) node_id: i32,
}

/// One batch a remote sender delivered into this CN's memory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RemoteBatch {
    /// Names the receive allocation holding the batch.
    pub token: u64,
    /// Exact row count, for the receiver's declared input cardinality.
    pub rows: u64,
}

/// One sender's output, and where it sits.
#[derive(Debug)]
pub(crate) enum SenderSource {
    /// A same-CN sender whose batches the engine parked on the GPU.
    LocalParked {
        /// Sender output names, which become the input stream's column names.
        names: Vec<String>,
        /// Where the engine parked this sender's batches.
        slot: SenderSlot,
    },
    /// A remote sender whose batches arrived over the exchange hop. Counts toward readiness only
    /// once `closed`.
    Remote {
        /// Sender output names carried on every frame (first frame wins, later frames must match).
        names: Vec<String>,
        /// Sender ordinal, needed to close the engine stream it feeds.
        sender_id: i32,
        /// Batches in arrival order.
        batches: Vec<RemoteBatch>,
        /// The sender announced eos; no more frames may follow.
        closed: bool,
    },
}

impl SenderSource {
    /// The sender's output column names, whichever side of the hop it sits on.
    pub(crate) fn names(&self) -> &[String] {
        match self {
            Self::LocalParked { names, .. } | Self::Remote { names, .. } => names,
        }
    }

    /// Where this sender's batches sit, for reading a key column in place.
    pub(crate) fn key_source(&self) -> KeySource {
        match self {
            Self::LocalParked { slot, .. } => KeySource::Parked(*slot),
            Self::Remote { batches, .. } => {
                KeySource::Received(batches.iter().map(|batch| batch.token).collect())
            }
        }
    }

    /// Whether this sender has finished producing (a parked local sender always has).
    fn is_complete(&self) -> bool {
        match self {
            Self::LocalParked { .. } => true,
            Self::Remote { closed, .. } => *closed,
        }
    }
}

/// One exchange input of a receiver fragment whose sender set is complete.
#[derive(Debug)]
pub(crate) struct ReadyExchangeInput {
    pub(crate) node_id: i32,
    pub(crate) sources: Vec<SenderSource>,
}

/// A receiver fragment whose exchange inputs are all ready for sequential execution.
#[derive(Debug)]
pub(crate) struct ReadyFragment {
    pub(crate) params: TExecPlanFragmentParams,
    pub(crate) inputs: Vec<ReadyExchangeInput>,
}

#[derive(Debug)]
struct PendingReceiver {
    params: TExecPlanFragmentParams,
    expected_senders: HashMap<i32, usize>,
}

/// How many purged queries are remembered. A frame for a query purged longer ago than this is
/// no longer refused, and the state it rebuilds waits for senders that never come.
const PURGED_QUERIES: usize = 1024;

/// What a purged query still held: the caller frees it outside the exchange lock.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Purged {
    /// Parked same-CN sender outputs, each released with `drop_parked`.
    pub(crate) slots: Vec<SenderSlot>,
    /// Direct-exchange receive buffers remote senders filled, each released by token.
    pub(crate) tokens: Vec<u64>,
}

impl Purged {
    fn add(&mut self, source: SenderSource) {
        match source {
            SenderSource::LocalParked { slot, .. } => self.slots.push(slot),
            SenderSource::Remote { batches, .. } => {
                self.tokens.extend(batches.iter().map(|batch| batch.token))
            }
        }
    }
}

/// What the rendezvous holds right now. All zero on an idle CN.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ExchangeCounts {
    /// Receivers waiting on at least one sender.
    pub(crate) receivers: usize,
    /// Parked same-CN sender outputs.
    pub(crate) parked_senders: usize,
    /// Remote batches holding receive buffers.
    pub(crate) remote_batches: usize,
}

#[derive(Debug, Default)]
struct ExchangeState {
    receivers: HashMap<FragmentInstanceId, PendingReceiver>,
    sources: HashMap<ExchangeKey, HashMap<i32, SenderSource>>,
    /// Next expected remote-frame sequence number per sender. A duplicate (below) is dropped
    /// idempotently; a gap (above) is a lost frame and fails the sender.
    remote_seq: HashMap<(ExchangeKey, i32), i64>,
    /// Queries purged after a failure or cancel, by [`FragmentInstanceId::query_hi`], with the
    /// error that purged them; `purged_order` is oldest first. A late receiver, sender or frame of
    /// one is refused, with that error, instead of rebuilding state that would never complete.
    purged: HashMap<u64, String>,
    purged_order: VecDeque<u64>,
    /// Runtime filter exchanges whose scan no longer reads them: a share arriving at one is not
    /// kept. Checked under the same lock as the push, so a share can't slip in after the close.
    closed_shares: HashSet<ExchangeKey>,
}

impl ExchangeState {
    fn refuse_purged(&self, fragment_instance_id: FragmentInstanceId) -> Result<(), String> {
        if let Some(error) = self.purged.get(&fragment_instance_id.query_hi()) {
            // The original error, so whichever refusal reaches the FE first still names the cause.
            return Err(format!(
                "the query of fragment instance {fragment_instance_id} already failed: {error}"
            ));
        }
        Ok(())
    }
}

/// Matches receiver-first StarRocks dispatch with later sender results.
#[derive(Debug, Default)]
pub(crate) struct LocalExchange {
    inner: Mutex<ExchangeState>,
}

impl LocalExchange {
    /// Registers a receiver fragment, returning it when every exchange input is already complete.
    pub(crate) fn register_receiver(
        &self,
        fragment_instance_id: FragmentInstanceId,
        expected_senders: Vec<(i32, usize)>,
        params: TExecPlanFragmentParams,
    ) -> Result<Option<ReadyFragment>, String> {
        if expected_senders.is_empty() {
            return Err("receiver fragment has no exchange inputs".to_string());
        }
        let mut expected_by_node = HashMap::with_capacity(expected_senders.len());
        for (node_id, expected) in expected_senders {
            if expected == 0 {
                return Err(format!("exchange node {node_id} expects no senders"));
            }
            if expected_by_node.insert(node_id, expected).is_some() {
                return Err(format!(
                    "duplicate exchange node {node_id} in receiver fragment"
                ));
            }
        }
        let mut state = self.lock();
        state.refuse_purged(fragment_instance_id)?;
        if state.receivers.contains_key(&fragment_instance_id) {
            return Err(format!(
                "duplicate receiver registration for fragment {fragment_instance_id}"
            ));
        }
        state.receivers.insert(
            fragment_instance_id,
            PendingReceiver {
                params,
                expected_senders: expected_by_node,
            },
        );
        Self::take_ready(&mut state, fragment_instance_id)
    }

    /// Records one sender as produced, returning the receiver when this completes its sender set.
    /// On error the source is not kept, and the caller still owns what it holds.
    pub(crate) fn push_sender(
        &self,
        key: ExchangeKey,
        sender_id: i32,
        source: SenderSource,
    ) -> Result<Option<ReadyFragment>, String> {
        let mut state = self.lock();
        Self::push_sender_locked(&mut state, key, sender_id, source)
    }

    fn push_sender_locked(
        state: &mut ExchangeState,
        key: ExchangeKey,
        sender_id: i32,
        source: SenderSource,
    ) -> Result<Option<ReadyFragment>, String> {
        state.refuse_purged(key.fragment_instance_id)?;
        let senders = state.sources.entry(key).or_default();
        if senders.contains_key(&sender_id) {
            return Err(format!("duplicate sender {sender_id} for exchange {key:?}"));
        }
        senders.insert(sender_id, source);
        Self::take_ready(state, key.fragment_instance_id)
    }

    /// Records one frame from a remote sender: a batch, eos, or both.
    pub(crate) fn push_remote_frame(
        &self,
        key: ExchangeKey,
        sender_id: i32,
        seq: i64,
        eos: bool,
        names: Vec<String>,
        batch: Option<RemoteBatch>,
    ) -> Result<Option<ReadyFragment>, String> {
        if names.is_empty() {
            return Err(format!(
                "remote sender {sender_id} for exchange {key:?} sent a frame without column \
                 names; the receiver cannot bind its input stream schema"
            ));
        }
        if batch.is_none() && !eos {
            return Err(format!(
                "remote sender {sender_id} for exchange {key:?} sent frame seq {seq} carrying \
                 neither a batch nor eos"
            ));
        }
        let mut state = self.lock();
        Self::push_frame_locked(&mut state, key, sender_id, seq, eos, names, batch)
    }

    #[allow(clippy::too_many_arguments)]
    fn push_frame_locked(
        state: &mut ExchangeState,
        key: ExchangeKey,
        sender_id: i32,
        seq: i64,
        eos: bool,
        names: Vec<String>,
        batch: Option<RemoteBatch>,
    ) -> Result<Option<ReadyFragment>, String> {
        state.refuse_purged(key.fragment_instance_id)?;
        let expected_seq = state.remote_seq.entry((key, sender_id)).or_insert(0);
        if seq < *expected_seq {
            info!(
                exchange = ?key,
                sender_id, seq, "dropping duplicate remote exchange frame"
            );
            return Ok(None);
        }
        if seq > *expected_seq {
            return Err(format!(
                "remote sender {sender_id} for exchange {key:?} skipped from frame seq \
                 {expected_seq} to {seq}; a frame was lost"
            ));
        }
        *expected_seq += 1;

        let senders = state.sources.entry(key).or_default();
        let source = senders
            .entry(sender_id)
            .or_insert_with(|| SenderSource::Remote {
                names: names.clone(),
                sender_id,
                batches: Vec::new(),
                closed: false,
            });
        let SenderSource::Remote {
            names: known_names,
            batches,
            closed,
            ..
        } = source
        else {
            return Err(format!(
                "remote frame for exchange {key:?} sender {sender_id} collides with a local \
                 parked sender of the same id"
            ));
        };
        if *closed {
            return Err(format!(
                "remote sender {sender_id} for exchange {key:?} sent frame seq {seq} after eos"
            ));
        }
        if known_names != &names {
            return Err(format!(
                "remote sender {sender_id} for exchange {key:?} changed its column names from \
                 {known_names:?} to {names:?}"
            ));
        }
        batches.extend(batch);
        *closed = eos;
        Self::take_ready(state, key.fragment_instance_id)
    }

    /// [`Self::push_sender`] for a runtime filter exchange. A closed one doesn't keep the share:
    /// it comes back for the caller to free.
    pub(crate) fn push_share_sender(
        &self,
        key: ExchangeKey,
        sender_id: i32,
        source: SenderSource,
    ) -> Result<Option<SenderSource>, String> {
        let mut state = self.lock();
        if state.closed_shares.contains(&key) {
            return Ok(Some(source));
        }
        Self::push_sender_locked(&mut state, key, sender_id, source)?;
        Ok(None)
    }

    /// [`Self::push_remote_frame`] for a runtime filter exchange. A closed one doesn't keep the
    /// frame (`false`): the caller frees its batch.
    pub(crate) fn push_share_frame(
        &self,
        key: ExchangeKey,
        sender_id: i32,
        seq: i64,
        eos: bool,
        names: Vec<String>,
        batch: Option<RemoteBatch>,
    ) -> Result<bool, String> {
        let mut state = self.lock();
        if state.closed_shares.contains(&key) {
            return Ok(false);
        }
        Self::push_frame_locked(&mut state, key, sender_id, seq, eos, names, batch)?;
        Ok(true)
    }

    /// Where exchange `key`'s batches sit, once every expected sender completed it, left in place
    /// for its receiver. `None` while a sender is still producing, and once the receiver took
    /// them (or was never registered here).
    pub(crate) fn complete_sources(&self, key: ExchangeKey) -> Option<Vec<KeySource>> {
        let state = self.lock();
        let expected = *state
            .receivers
            .get(&key.fragment_instance_id)?
            .expected_senders
            .get(&key.node_id)?;
        Self::completed(&state, key, expected)
    }

    /// Every share that reached runtime filter exchange `key` so far, in sender-id order: where
    /// its batches sit, its column names, and whether its sender finished. Left in place.
    pub(crate) fn shares(&self, key: ExchangeKey) -> Vec<(KeySource, Vec<String>, bool)> {
        let state = self.lock();
        let mut ordered = state
            .sources
            .get(&key)
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        ordered.sort_unstable_by_key(|(sender_id, _)| **sender_id);
        ordered
            .into_iter()
            .map(|(_, source)| {
                (
                    source.key_source(),
                    source.names().to_vec(),
                    source.is_complete(),
                )
            })
            .collect()
    }

    /// Closes runtime filter exchange `key`, and returns what its shares held for the caller to
    /// free. Later shares are not kept.
    pub(crate) fn close_shares(&self, key: ExchangeKey) -> Purged {
        let mut state = self.lock();
        state.closed_shares.insert(key);
        state.remote_seq.retain(|(seq_key, _), _| *seq_key != key);
        let mut taken = Purged::default();
        for source in state
            .sources
            .remove(&key)
            .into_iter()
            .flat_map(HashMap::into_values)
        {
            taken.add(source);
        }
        taken
    }

    /// Exchange `key`'s sources in sender-id order, once exactly `expected` senders completed it.
    fn completed(
        state: &ExchangeState,
        key: ExchangeKey,
        expected: usize,
    ) -> Option<Vec<KeySource>> {
        let senders = state.sources.get(&key)?;
        if senders.len() != expected || !senders.values().all(SenderSource::is_complete) {
            return None;
        }
        let mut ordered = senders.iter().collect::<Vec<_>>();
        ordered.sort_unstable_by_key(|(sender_id, _)| **sender_id);
        Some(
            ordered
                .into_iter()
                .map(|(_, source)| source.key_source())
                .collect(),
        )
    }

    /// Hands the receiver over, exactly once, when every expected sender of every exchange input
    /// is complete. Removing it under the lock is what makes the handoff happen once.
    fn take_ready(
        state: &mut ExchangeState,
        fragment_instance_id: FragmentInstanceId,
    ) -> Result<Option<ReadyFragment>, String> {
        let Some(receiver) = state.receivers.get(&fragment_instance_id) else {
            return Ok(None);
        };
        for (&node_id, &expected) in &receiver.expected_senders {
            let key = ExchangeKey {
                fragment_instance_id,
                node_id,
            };
            let sources = state.sources.get(&key);
            let total = sources.map(HashMap::len).unwrap_or(0);
            if total > expected {
                return Err(format!(
                    "exchange {key:?} received {total} senders but expected {expected}"
                ));
            }
            let complete = sources
                .map(|senders| {
                    senders
                        .values()
                        .filter(|source| source.is_complete())
                        .count()
                })
                .unwrap_or(0);
            if complete != expected {
                return Ok(None);
            }
        }

        let receiver = state
            .receivers
            .remove(&fragment_instance_id)
            .expect("receiver checked above");
        let mut node_ids = receiver.expected_senders.into_keys().collect::<Vec<_>>();
        node_ids.sort_unstable();
        let inputs = node_ids
            .into_iter()
            .map(|node_id| {
                let key = ExchangeKey {
                    fragment_instance_id,
                    node_id,
                };
                let mut senders = state
                    .sources
                    .remove(&key)
                    .unwrap_or_default()
                    .into_iter()
                    .collect::<Vec<_>>();
                senders.sort_unstable_by_key(|(sender_id, _)| *sender_id);
                let sources = senders.into_iter().map(|(_, source)| source).collect();
                ReadyExchangeInput { node_id, sources }
            })
            .collect();
        state
            .remote_seq
            .retain(|(key, _), _| key.fragment_instance_id != fragment_instance_id);
        Ok(Some(ReadyFragment {
            params: receiver.params,
            inputs,
        }))
    }

    /// Drops everything held for `query`'s receivers and returns what it still owned, then
    /// refuses any later receiver, sender or frame of the query with `error`. Idempotent; the
    /// first error is kept, being the cause. A receiver already handed over by `take_ready` is not
    /// here: it frees its own inputs.
    pub(crate) fn purge_query(&self, query: FragmentInstanceId, error: &str) -> Purged {
        let query_hi = query.query_hi();
        let ours = |id: &FragmentInstanceId| id.query_hi() == query_hi;
        let mut state = self.lock();
        if let Entry::Vacant(first) = state.purged.entry(query_hi) {
            first.insert(error.to_string());
            state.purged_order.push_back(query_hi);
            if state.purged_order.len() > PURGED_QUERIES {
                let oldest = state.purged_order.pop_front().expect("over capacity");
                state.purged.remove(&oldest);
            }
        }
        state.receivers.retain(|id, _| !ours(id));
        state
            .closed_shares
            .retain(|key| !ours(&key.fragment_instance_id));
        state
            .remote_seq
            .retain(|(key, _), _| !ours(&key.fragment_instance_id));
        let keys: Vec<ExchangeKey> = state
            .sources
            .keys()
            .filter(|key| ours(&key.fragment_instance_id))
            .copied()
            .collect();
        let mut purged = Purged::default();
        for key in keys {
            for source in state
                .sources
                .remove(&key)
                .into_iter()
                .flat_map(HashMap::into_values)
            {
                purged.add(source);
            }
        }
        purged
    }

    /// What the rendezvous holds right now.
    pub(crate) fn counts(&self) -> ExchangeCounts {
        let state = self.lock();
        let mut counts = ExchangeCounts {
            receivers: state.receivers.len(),
            ..ExchangeCounts::default()
        };
        for source in state.sources.values().flat_map(HashMap::values) {
            match source {
                SenderSource::LocalParked { .. } => counts.parked_senders += 1,
                SenderSource::Remote { batches, .. } => counts.remote_batches += batches.len(),
            }
        }
        counts
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, ExchangeState> {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use starrocks_thrift::internal_service::InternalServiceVersion;

    use super::*;

    fn key(instance: u64, node_id: i32) -> ExchangeKey {
        ExchangeKey {
            fragment_instance_id: FragmentInstanceId::from_halves(7, instance as i64),
            node_id,
        }
    }

    fn params() -> TExecPlanFragmentParams {
        TExecPlanFragmentParams {
            protocol_version: InternalServiceVersion::V1,
            fragment: None,
            desc_tbl: None,
            params: None,
            coord: None,
            backend_num: None,
            query_globals: None,
            query_options: None,
            enable_profile: None,
            resource_info: None,
            import_label: None,
            db_name: None,
            load_job_id: None,
            load_error_hub_info: None,
            is_pipeline: None,
            pipeline_dop: None,
            per_scan_node_dop: None,
            workgroup: None,
            enable_resource_group: None,
            func_version: None,
            enable_shared_scan: None,
            is_stream_pipeline: None,
            adaptive_dop_param: None,
            group_execution_scan_dop: None,
            pred_tree_params: None,
            exec_stats_node_ids: None,
            arrow_flight_sql_version: None,
        }
    }

    fn names() -> Vec<String> {
        vec!["id".to_string(), "name".to_string()]
    }

    fn local_slot(key: ExchangeKey, sender_id: i32) -> SenderSlot {
        SenderSlot {
            fragment_instance_id: key.fragment_instance_id,
            node_id: key.node_id,
            sender_id,
        }
    }

    fn local(key: ExchangeKey, sender_id: i32) -> SenderSource {
        SenderSource::LocalParked {
            names: names(),
            slot: local_slot(key, sender_id),
        }
    }

    fn remote(token: u64) -> RemoteBatch {
        RemoteBatch { token, rows: 1 }
    }

    /// Receiver-first dispatch: the receiver registers, its senders park later, and the fragment
    /// comes back on the push that completes the set, sources in sender-id order.
    #[test]
    fn receiver_registered_first_is_ready_on_its_last_sender() {
        let exchange = LocalExchange::default();
        let key = key(1, 7);
        assert!(
            exchange
                .register_receiver(key.fragment_instance_id, vec![(7, 2)], params())
                .unwrap()
                .is_none()
        );
        assert!(
            exchange
                .push_sender(key, 1, local(key, 1))
                .unwrap()
                .is_none()
        );
        let ready = exchange
            .push_sender(key, 0, local(key, 0))
            .unwrap()
            .expect("the second sender completes the set");
        assert_eq!(ready.params.protocol_version, InternalServiceVersion::V1);
        assert_eq!(ready.inputs.len(), 1);
        assert_eq!(ready.inputs[0].node_id, 7);
        assert_eq!(ready.inputs[0].sources[0].names(), names());
        let slots: Vec<SenderSlot> = ready.inputs[0]
            .sources
            .iter()
            .map(|source| match source {
                SenderSource::LocalParked { slot, .. } => *slot,
                other => panic!("expected a parked local source, got {other:?}"),
            })
            .collect();
        assert_eq!(slots, vec![local_slot(key, 0), local_slot(key, 1)]);
    }

    /// A runtime filter reads a build exchange's sources in place once they are complete, while
    /// the receiver still waits on another exchange; they are gone once it takes them.
    #[test]
    fn complete_sources_are_read_in_place_until_the_receiver_takes_them() {
        let exchange = LocalExchange::default();
        let build = key(3, 7);
        let probe = key(3, 8);
        assert_eq!(exchange.complete_sources(build), None, "no receiver yet");
        exchange
            .register_receiver(build.fragment_instance_id, vec![(7, 2), (8, 1)], params())
            .unwrap();
        exchange.push_sender(build, 0, local(build, 0)).unwrap();
        exchange
            .push_remote_frame(build, 1, 0, false, names(), Some(remote(41)))
            .unwrap();
        assert_eq!(
            exchange.complete_sources(build),
            None,
            "sender 1 is still open"
        );
        exchange
            .push_remote_frame(build, 1, 1, true, names(), Some(remote(42)))
            .unwrap();
        let expected = vec![
            KeySource::Parked(local_slot(build, 0)),
            KeySource::Received(vec![41, 42]),
        ];
        assert_eq!(exchange.complete_sources(build), Some(expected.clone()));
        assert_eq!(
            exchange.complete_sources(build),
            Some(expected),
            "reading takes nothing"
        );
        assert_eq!(exchange.complete_sources(probe), None);
        assert!(
            exchange
                .push_sender(probe, 0, local(probe, 0))
                .unwrap()
                .is_some()
        );
        assert_eq!(
            exchange.complete_sources(build),
            None,
            "the receiver took them"
        );
    }

    /// Senders that finish before their receiver is dispatched make it ready on registration.
    #[test]
    fn senders_parked_before_the_receiver_make_it_ready_on_registration() {
        let exchange = LocalExchange::default();
        let key = key(2, 7);
        assert!(
            exchange
                .push_sender(key, 0, local(key, 0))
                .unwrap()
                .is_none()
        );
        let ready = exchange
            .register_receiver(key.fragment_instance_id, vec![(7, 1)], params())
            .unwrap()
            .expect("the sender already parked");
        let SenderSource::LocalParked { names: got, slot } = &ready.inputs[0].sources[0] else {
            panic!("expected a parked local source");
        };
        assert_eq!(got, &names());
        assert_eq!(*slot, local_slot(key, 0));
    }

    /// A receiver with several exchange inputs waits for every one; ready inputs come out in
    /// node-id order.
    #[test]
    fn every_exchange_input_must_complete_and_inputs_come_out_in_node_order() {
        let exchange = LocalExchange::default();
        let (high, low) = (key(3, 9), key(3, 7));
        assert!(
            exchange
                .register_receiver(high.fragment_instance_id, vec![(9, 1), (7, 1)], params())
                .unwrap()
                .is_none()
        );
        assert!(
            exchange
                .push_sender(high, 0, local(high, 0))
                .unwrap()
                .is_none(),
            "one complete input of two is not ready"
        );
        let ready = exchange
            .push_sender(low, 0, local(low, 0))
            .unwrap()
            .expect("both inputs complete");
        let node_ids: Vec<i32> = ready.inputs.iter().map(|input| input.node_id).collect();
        assert_eq!(node_ids, vec![7, 9]);
    }

    #[test]
    fn receiver_registration_is_validated() {
        let exchange = LocalExchange::default();
        let instance = key(4, 7).fragment_instance_id;
        let err = exchange
            .register_receiver(instance, Vec::new(), params())
            .unwrap_err();
        assert!(err.contains("no exchange inputs"), "{err}");
        let err = exchange
            .register_receiver(instance, vec![(7, 0)], params())
            .unwrap_err();
        assert!(err.contains("expects no senders"), "{err}");
        let err = exchange
            .register_receiver(instance, vec![(7, 1), (7, 2)], params())
            .unwrap_err();
        assert!(err.contains("duplicate exchange node 7"), "{err}");
        exchange
            .register_receiver(instance, vec![(7, 1)], params())
            .unwrap();
        let err = exchange
            .register_receiver(instance, vec![(7, 1)], params())
            .unwrap_err();
        assert!(err.contains("duplicate receiver registration"), "{err}");
    }

    #[test]
    fn duplicate_and_surplus_senders_are_loud_errors() {
        let exchange = LocalExchange::default();
        let key = key(5, 7);
        exchange.push_sender(key, 0, local(key, 0)).unwrap();
        let err = exchange.push_sender(key, 0, local(key, 0)).unwrap_err();
        assert!(err.contains("duplicate sender 0"), "{err}");
        exchange.push_sender(key, 1, local(key, 1)).unwrap();
        let err = exchange
            .register_receiver(key.fragment_instance_id, vec![(7, 1)], params())
            .unwrap_err();
        assert!(err.contains("received 2 senders but expected 1"), "{err}");
    }

    #[test]
    fn remote_frames_accumulate_and_eos_completes_the_set() {
        let exchange = LocalExchange::default();
        let key = key(1, 7);

        assert!(
            exchange
                .push_remote_frame(key, 0, 0, false, names(), Some(remote(1)))
                .unwrap()
                .is_none()
        );
        assert!(
            exchange
                .push_remote_frame(key, 0, 1, false, names(), Some(remote(2)))
                .unwrap()
                .is_none()
        );
        assert!(
            exchange
                .register_receiver(key.fragment_instance_id, vec![(7, 1)], params())
                .unwrap()
                .is_none()
        );

        let ready = exchange
            .push_remote_frame(key, 0, 2, true, names(), None)
            .unwrap()
            .expect("eos completes the sender set");
        let SenderSource::Remote {
            names: got_names,
            sender_id,
            batches,
            closed,
        } = &ready.inputs[0].sources[0]
        else {
            panic!("expected a remote source");
        };
        assert_eq!(got_names, &names());
        assert_eq!(*sender_id, 0);
        assert!(*closed);
        assert_eq!(batches, &vec![remote(1), remote(2)]);
    }

    #[test]
    fn duplicate_remote_seq_is_dropped_idempotently() {
        let exchange = LocalExchange::default();
        let key = key(2, 7);
        exchange
            .push_remote_frame(key, 0, 0, false, names(), Some(remote(1)))
            .unwrap();
        assert!(
            exchange
                .push_remote_frame(key, 0, 0, false, names(), Some(remote(1)))
                .unwrap()
                .is_none(),
            "replayed data frame is dropped"
        );
        exchange
            .push_remote_frame(key, 0, 1, true, names(), None)
            .unwrap();
        assert!(
            exchange
                .push_remote_frame(key, 0, 1, true, names(), None)
                .unwrap()
                .is_none(),
            "replayed eos frame is dropped"
        );
        let ready = exchange
            .register_receiver(key.fragment_instance_id, vec![(7, 1)], params())
            .unwrap()
            .expect("sender already complete");
        let SenderSource::Remote { batches, .. } = &ready.inputs[0].sources[0] else {
            panic!("expected a remote source");
        };
        assert_eq!(batches.len(), 1);
    }

    #[test]
    fn remote_gap_is_a_lost_frame() {
        let exchange = LocalExchange::default();
        let key = key(6, 7);
        let err = exchange
            .push_remote_frame(key, 0, 1, true, names(), None)
            .unwrap_err();
        assert!(err.contains("skipped from frame seq 0 to 1"), "{err}");
    }

    #[test]
    fn an_open_remote_sender_does_not_complete_the_set() {
        let exchange = LocalExchange::default();
        let key = key(8, 7);
        exchange
            .register_receiver(key.fragment_instance_id, vec![(7, 1)], params())
            .unwrap();
        assert!(
            exchange
                .push_remote_frame(key, 0, 0, false, names(), Some(remote(1)))
                .unwrap()
                .is_none()
        );
    }

    fn query_key(hi: i64, instance: i64, node_id: i32) -> ExchangeKey {
        ExchangeKey {
            fragment_instance_id: FragmentInstanceId::from_halves(hi, instance),
            node_id,
        }
    }

    /// A failed query's waiting receiver, parked local sender and received remote batches are
    /// all returned for release; another query's state is untouched.
    #[test]
    fn purge_returns_what_a_query_holds_and_leaves_other_queries() {
        let exchange = LocalExchange::default();
        let (doomed, other) = (query_key(41, 2, 7), query_key(42, 2, 7));
        let doomed_late = query_key(41, 3, 9);
        for key in [doomed, other] {
            exchange
                .register_receiver(key.fragment_instance_id, vec![(7, 3)], params())
                .unwrap();
            exchange.push_sender(key, 0, local(key, 0)).unwrap();
            exchange
                .push_remote_frame(
                    key,
                    1,
                    0,
                    false,
                    names(),
                    Some(remote(key.fragment_instance_id.query_hi())),
                )
                .unwrap();
        }
        // A sender whose receiver was never registered here is purged too.
        exchange
            .push_remote_frame(doomed_late, 0, 0, false, names(), Some(remote(99)))
            .unwrap();

        let purged = exchange.purge_query(FragmentInstanceId::from_halves(41, 0), "boom");
        assert_eq!(purged.slots, vec![local_slot(doomed, 0)]);
        let mut tokens = purged.tokens;
        tokens.sort_unstable();
        assert_eq!(tokens, vec![41, 99]);
        assert_eq!(
            exchange.counts(),
            ExchangeCounts {
                receivers: 1,
                parked_senders: 1,
                remote_batches: 1,
            }
        );
        assert_eq!(
            exchange.purge_query(FragmentInstanceId::from_halves(41, 0), "again"),
            Purged::default(),
            "a second purge finds nothing"
        );
    }

    /// After a purge, nothing of the query can rebuild exchange state that would never complete.
    #[test]
    fn a_purged_query_refuses_late_receivers_senders_and_frames() {
        let exchange = LocalExchange::default();
        let key = query_key(43, 2, 7);
        exchange.purge_query(FragmentInstanceId::from_halves(43, 0), "GPU out of memory");
        exchange.purge_query(
            FragmentInstanceId::from_halves(43, 0),
            "cancelled by the FE",
        );
        let refused = |result: Result<Option<ReadyFragment>, String>| {
            let err = result.unwrap_err();
            assert!(err.contains("already failed: GPU out of memory"), "{err}");
        };
        refused(exchange.register_receiver(key.fragment_instance_id, vec![(7, 1)], params()));
        refused(exchange.push_sender(key, 0, local(key, 0)));
        refused(exchange.push_remote_frame(key, 0, 0, true, names(), Some(remote(5))));
        assert_eq!(exchange.counts(), ExchangeCounts::default());
        // Another query is unaffected.
        let live = query_key(44, 2, 7);
        assert!(
            exchange
                .register_receiver(live.fragment_instance_id, vec![(7, 1)], params())
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn purged_queries_are_remembered_up_to_a_bound() {
        let exchange = LocalExchange::default();
        for hi in 0..=PURGED_QUERIES as i64 {
            exchange.purge_query(FragmentInstanceId::from_halves(hi, 0), "boom");
        }
        let state = exchange.lock();
        assert_eq!(state.purged.len(), PURGED_QUERIES);
        assert!(
            !state.purged.contains_key(&0),
            "the oldest purge is forgotten first"
        );
        assert!(state.purged.contains_key(&(PURGED_QUERIES as u64)));
    }

    #[test]
    fn a_closed_filter_exchange_keeps_no_share() {
        let exchange = LocalExchange::default();
        let share = key(3, 1_000_000);
        // A share that arrived before the close comes back to be freed.
        exchange
            .push_share_frame(share, 0, 0, true, names(), Some(remote(5)))
            .unwrap();
        assert_eq!(exchange.shares(share).len(), 1);
        let closed = exchange.close_shares(share);
        assert_eq!(closed.tokens, vec![5]);
        // Later ones are not kept, whichever side they come from.
        assert!(
            !exchange
                .push_share_frame(share, 1, 0, true, names(), Some(remote(6)))
                .unwrap()
        );
        assert!(
            exchange
                .push_share_sender(share, 2, local(share, 2))
                .unwrap()
                .is_some()
        );
        assert!(exchange.shares(share).is_empty());
        assert_eq!(exchange.counts(), ExchangeCounts::default());
        // Another filter exchange is open.
        assert!(
            exchange
                .push_share_sender(key(4, 1_000_000), 2, local(key(4, 1_000_000), 2))
                .unwrap()
                .is_none()
        );
    }
}
