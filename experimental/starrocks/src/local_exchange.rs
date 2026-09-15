//! Sequential exchange rendezvous.
//!
//! Matches StarRocks' receiver-first dispatch with senders that arrive later. A same-CN sender
//! parks native engine output as a [`SenderSlot`]; a remote sender's packed GPU bytes arrive as
//! [`StagedBatch`] metadata (offset/length into the receiver's staging arena) and sit in
//! [`SenderSource::Remote`] until eos. The receiver becomes ready when every expected sender
//! is complete. There is no fusion: every leaf runs, then hops, then the root.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Mutex;

use starrocks_thrift::internal_service::TExecPlanFragmentParams;
use tracing::info;

use crate::fragment_executor::{SenderSlot, StagedBatch};
use crate::result_store::FragmentInstanceId;

/// Receiver identity used by both the stream sink and the exchange node.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ExchangeKey {
    pub(crate) fragment_instance_id: FragmentInstanceId,
    pub(crate) node_id: i32,
}

/// One sender's output, and where it sits.
#[derive(Clone, Debug)]
pub(crate) enum SenderSource {
    /// A same-CN sender whose batches the engine parked on the GPU.
    LocalParked {
        /// Sender output names, which become the input stream's column names.
        names: Vec<String>,
        /// Where the engine parked this sender's batches.
        slot: SenderSlot,
    },
    /// A remote sender whose packed batches arrived over the exchange hop. Counts toward
    /// readiness only once `closed`.
    Remote {
        /// Sender output names carried on every frame (first frame wins, later frames must match).
        names: Vec<String>,
        /// Sender ordinal, needed to `close_input` the engine stream it feeds.
        sender_id: i32,
        /// Packed batches in arrival order. Each non-empty `len` names a receiver-side arena
        /// lease the later engine path releases inside `push_packed`.
        batches: Vec<StagedBatch>,
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

    /// Whether this sender has finished producing (a parked local sender always has).
    fn is_complete(&self) -> bool {
        match self {
            Self::LocalParked { .. } => true,
            Self::Remote { closed, .. } => *closed,
        }
    }

    /// Receiver-side arena offsets that still hold a lease (`len != 0`). The caller of
    /// [`LocalExchange::retire_receiver`] releases these so a cancelled query cannot pin the
    /// arena.
    pub(crate) fn outstanding_lease_offsets(&self) -> Vec<u64> {
        match self {
            Self::LocalParked { .. } => Vec::new(),
            Self::Remote { batches, .. } => batches
                .iter()
                .filter(|batch| batch.len != 0)
                .map(|batch| batch.offset)
                .collect(),
        }
    }
}

/// One exchange input of a receiver fragment whose sender set is complete.
#[derive(Clone, Debug)]
pub(crate) struct ReadyExchangeInput {
    pub(crate) node_id: i32,
    pub(crate) sources: Vec<SenderSource>,
}

/// A receiver fragment whose exchange inputs are all ready for sequential execution.
#[derive(Debug)]
pub struct ReadyFragment {
    pub(crate) params: TExecPlanFragmentParams,
    pub(crate) inputs: Vec<ReadyExchangeInput>,
}

#[derive(Debug)]
struct PendingReceiver {
    params: TExecPlanFragmentParams,
    expected_senders: HashMap<i32, usize>,
}

/// How many cancelled receivers are remembered; the oldest is forgotten first.
const RETIRED_CAPACITY: usize = 1024;

#[derive(Debug, Default)]
struct ExchangeState {
    receivers: HashMap<FragmentInstanceId, PendingReceiver>,
    sources: HashMap<ExchangeKey, HashMap<i32, SenderSource>>,
    /// Next expected remote-frame sequence number per sender. A duplicate (below) is dropped
    /// idempotently; a gap (above) is a lost frame and fails the sender.
    remote_seq: HashMap<(ExchangeKey, i32), i64>,
    retired: HashSet<FragmentInstanceId>,
    retired_order: VecDeque<FragmentInstanceId>,
}

/// Matches receiver-first StarRocks dispatch with later sender results.
#[derive(Debug, Default)]
pub struct LocalExchange {
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
    pub(crate) fn push_sender(
        &self,
        key: ExchangeKey,
        sender_id: i32,
        source: SenderSource,
    ) -> Result<Option<ReadyFragment>, String> {
        let mut state = self.lock();
        if state.retired.contains(&key.fragment_instance_id) {
            return Ok(None);
        }
        let senders = state.sources.entry(key).or_default();
        if senders.contains_key(&sender_id) {
            return Err(format!("duplicate sender {sender_id} for exchange {key:?}"));
        }
        senders.insert(sender_id, source);
        Self::take_ready(&mut state, key.fragment_instance_id)
    }

    /// Records one packed hop frame from a remote sender: a staged batch, eos, or both.
    pub(crate) fn push_remote_frame(
        &self,
        key: ExchangeKey,
        sender_id: i32,
        seq: i64,
        eos: bool,
        names: Vec<String>,
        batch: Option<StagedBatch>,
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
        if state.retired.contains(&key.fragment_instance_id) {
            return Ok(None);
        }
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
        if let Some(batch) = batch {
            batches.push(batch);
        }
        if eos {
            *closed = true;
        }
        Self::take_ready(&mut state, key.fragment_instance_id)
    }

    /// Forgets a receiver the FE cancelled. Returns recorded sources so the caller can drop
    /// parked GPU output and release remote staging leases. Idempotent.
    pub(crate) fn retire_receiver(
        &self,
        fragment_instance_id: FragmentInstanceId,
    ) -> Vec<SenderSource> {
        let mut state = self.lock();
        state.receivers.remove(&fragment_instance_id);
        let mut keys = state
            .sources
            .keys()
            .filter(|key| key.fragment_instance_id == fragment_instance_id)
            .copied()
            .collect::<Vec<_>>();
        keys.sort_unstable_by_key(|key| key.node_id);
        let mut removed = Vec::new();
        for key in keys {
            let mut senders = state.sources.remove(&key).unwrap_or_default();
            let mut sender_ids = senders.keys().copied().collect::<Vec<_>>();
            sender_ids.sort_unstable();
            removed.extend(
                sender_ids
                    .into_iter()
                    .map(|sender_id| senders.remove(&sender_id).expect("sender id came from map")),
            );
        }
        state
            .remote_seq
            .retain(|(key, _), _| key.fragment_instance_id != fragment_instance_id);
        if state.retired.insert(fragment_instance_id) {
            state.retired_order.push_back(fragment_instance_id);
            while state.retired.len() > RETIRED_CAPACITY {
                if let Some(oldest) = state.retired_order.pop_front() {
                    state.retired.remove(&oldest);
                }
            }
        }
        removed
    }

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
                let mut senders = state.sources.remove(&key).unwrap_or_default();
                let mut sender_ids = senders.keys().copied().collect::<Vec<_>>();
                sender_ids.sort_unstable();
                let sources = sender_ids
                    .into_iter()
                    .map(|sender_id| senders.remove(&sender_id).expect("sender id came from map"))
                    .collect();
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

    fn staged(fill: u64) -> StagedBatch {
        StagedBatch {
            metadata: fill.to_le_bytes().to_vec(),
            offset: fill * 64,
            len: 8,
            rows: Some(1),
        }
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
                .push_remote_frame(key, 0, 0, false, names(), Some(staged(1)))
                .unwrap()
                .is_none()
        );
        assert!(
            exchange
                .push_remote_frame(key, 0, 1, false, names(), Some(staged(2)))
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
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].metadata, 1u64.to_le_bytes());
        assert_eq!(batches[1].offset, 128);
        assert_eq!(batches[1].len, 8);
    }

    #[test]
    fn duplicate_remote_seq_is_dropped_idempotently() {
        let exchange = LocalExchange::default();
        let key = key(2, 7);
        exchange
            .push_remote_frame(key, 0, 0, false, names(), Some(staged(1)))
            .unwrap();
        assert!(
            exchange
                .push_remote_frame(key, 0, 0, false, names(), Some(staged(1)))
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
                .push_remote_frame(key, 0, 0, false, names(), Some(staged(1)))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn retire_drops_a_pending_receiver_and_returns_leases_to_release() {
        let exchange = LocalExchange::default();
        let key = key(9, 7);
        exchange
            .register_receiver(key.fragment_instance_id, vec![(7, 1)], params())
            .unwrap();
        exchange
            .push_remote_frame(key, 0, 0, false, names(), Some(staged(3)))
            .unwrap();
        let removed = exchange.retire_receiver(key.fragment_instance_id);
        assert_eq!(removed.len(), 1);
        assert_eq!(removed[0].outstanding_lease_offsets(), vec![192]);
        assert!(
            exchange
                .push_remote_frame(key, 0, 1, true, names(), None)
                .unwrap()
                .is_none()
        );
        // Retired id is remembered: a late local park is ignored rather than resurrecting it.
        assert!(
            exchange
                .push_sender(key, 1, local(key, 1))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn zero_length_remote_batch_holds_no_lease() {
        let exchange = LocalExchange::default();
        let key = key(10, 7);
        let empty = StagedBatch {
            metadata: Vec::new(),
            offset: 0,
            len: 0,
            rows: Some(0),
        };
        exchange
            .push_remote_frame(key, 0, 0, true, names(), Some(empty))
            .unwrap();
        let ready = exchange
            .register_receiver(key.fragment_instance_id, vec![(7, 1)], params())
            .unwrap()
            .expect("eos with a zero-row batch completes the set");
        assert!(
            ready.inputs[0].sources[0]
                .outstanding_lease_offsets()
                .is_empty()
        );
    }
}
