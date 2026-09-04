//! Sequential exchange rendezvous.
//!
//! The exchange matches StarRocks' receiver-first dispatch with the senders that arrive later.
//! A same-node sender's rows stay on the GPU, parked in the engine as native batches; a remote
//! sender's batches arrive as staged packed bytes in this CN's arena (nixl tier) or as decoded
//! Arrow record batches in host memory (arrow tier).
//! Either way, what is tracked here is which senders have finished and where their output sits.
//!
//! A third kind of sender never runs at all: a same-node leaf whose plan was deferred into its
//! receiver ([`SenderSource::LocalPlan`], recorded by [`LocalExchange::offer_local_plan`]) is
//! complete the moment it is recorded, and the receiver splices the plan over its exchange node
//! before translating (`compute_node_service.rs`, `fold_deferred_plans`).

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::sync::Mutex;

use starrocks_plan_translator::FusionRefusal;
use starrocks_thrift::internal_service::TExecPlanFragmentParams;
use tracing::info;

use crate::fragment_executor::{SenderSlot, StagedBatch};
use crate::result_store::FragmentInstanceId;

/// Receiver identity used by both the stream sink and exchange node.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ExchangeKey {
    pub(crate) fragment_instance_id: FragmentInstanceId,
    pub(crate) node_id: i32,
}

/// One sender's output, and where it sits.
#[derive(Clone, Debug)]
pub(crate) enum SenderSource {
    /// A same-node sender whose batches the engine parked on the GPU.
    LocalParked {
        /// Sender output names, which become the input stream's column names.
        names: Vec<String>,
        /// Where the engine parked this sender's batches.
        slot: SenderSlot,
    },
    /// A remote sender whose batches arrived over `transmit_packed`: packed bytes nixl-WRITTEN
    /// into this CN's staging arena, or Arrow IPC frames decoded into host `RecordBatch`es (the
    /// arrow tier keeps a receiver's whole remote input in host RAM until dispatch; nothing but
    /// the host bounds it). Counts toward readiness only once `closed`.
    Remote {
        /// Sender output names carried on every `transmit_packed` frame (first frame wins,
        /// later frames must match).
        names: Vec<String>,
        /// Sender ordinal, needed to `close_input` the engine stream it feeds.
        sender_id: i32,
        /// Batches staged in arrival (sequence) order; the engine releases each lease after
        /// pushing it.
        batches: Vec<StagedBatch>,
        /// The sender announced eos; no more frames may follow.
        closed: bool,
    },
    /// A same-node sender whose plan was deferred into this receiver instead of running: the
    /// receiver splices it over this exchange node before translating. Complete by construction.
    LocalPlan(Box<LocalPlan>),
}

impl SenderSource {
    /// The sender's output column names, whichever side of the wire it sits on.
    pub(crate) fn names(&self) -> &[String] {
        match self {
            Self::LocalParked { names, .. } | Self::Remote { names, .. } => names,
            // A deferred plan feeds no stream: the fold in `execute_ready_fragment` consumes
            // every `LocalPlan` before `names()` is read for the inputs that still stream.
            Self::LocalPlan(_) => &[],
        }
    }

    /// Whether this sender has finished producing (a parked local sender always has, and a
    /// deferred plan has nothing to produce).
    fn is_complete(&self) -> bool {
        match self {
            Self::LocalParked { .. } | Self::LocalPlan(_) => true,
            Self::Remote { closed, .. } => *closed,
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
pub(crate) struct ReadyFragment {
    pub(crate) params: TExecPlanFragmentParams,
    pub(crate) inputs: Vec<ReadyExchangeInput>,
}

/// A same-node sender whose PLAN was deferred into its receiver instead of running. Holds thrift
/// only: no GPU memory and no staging lease, so cancellation and lease release have nothing to
/// do for it. `inputs` are the exchange inputs the deferred sender itself consumed (parked slots
/// or staged remote batches, never another `LocalPlan`: the fold consumes deferred plans before
/// anything is offered), relayed by the fused fragment under their own stream ids. Empty while
/// only leaves are deferred.
#[derive(Clone)]
pub(crate) struct LocalPlan {
    pub(crate) params: TExecPlanFragmentParams,
    pub(crate) inputs: Vec<ReadyExchangeInput>,
}

/// By hand: the derived `Debug` of `TExecPlanFragmentParams` is the whole fragment dump (tens of
/// thousands of lines), which must never reach a log line or an error string.
impl fmt::Debug for LocalPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let exec = self.params.params.as_ref();
        f.debug_struct("LocalPlan")
            .field(
                "query_id",
                &exec.map(|exec| FragmentInstanceId::from(&exec.query_id)),
            )
            .field(
                "fragment_instance_id",
                &exec.map(|exec| FragmentInstanceId::from(&exec.fragment_instance_id)),
            )
            .field(
                "plan_nodes",
                &self
                    .params
                    .fragment
                    .as_ref()
                    .and_then(|fragment| fragment.plan.as_ref())
                    .map_or(0, |plan| plan.nodes.len()),
            )
            .field("inputs", &self.inputs.len())
            .finish()
    }
}

/// Why an offered sender plan was not deferred. Logged by the caller, never an error: the sender
/// then runs and parks as it would without fusion.
#[derive(Debug)]
pub(crate) enum FuseSkip {
    /// No receiver of that fragment instance is pending: not registered yet, already released,
    /// or retired.
    NoPendingReceiver,
    /// The receiver is pending but its `per_exch_num_senders` lacks this exchange node id (a
    /// malformed dispatch).
    ExchangeNotDeclared,
    /// The exchange expects that many senders; only a single-sender exchange is the identity on
    /// rows.
    ReceiverExpectsMany(usize),
    /// A source is already recorded for the exchange. Cannot happen with one expected sender
    /// unless a frame was misrouted; kept as a guard.
    ExchangeAlreadySourced,
    /// The edge fails the translator's structural checks.
    Structural(FusionRefusal),
}

/// The variant name, so `fragment fusion skipped reason=` lines group by reason; the structural
/// variant carries the translator's one-line refusal.
impl fmt::Display for FuseSkip {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoPendingReceiver => f.write_str("NoPendingReceiver"),
            Self::ExchangeNotDeclared => f.write_str("ExchangeNotDeclared"),
            Self::ReceiverExpectsMany(expected) => write!(f, "ReceiverExpectsMany({expected})"),
            Self::ExchangeAlreadySourced => f.write_str("ExchangeAlreadySourced"),
            Self::Structural(refusal) => write!(f, "Structural({refusal})"),
        }
    }
}

/// The outcome of [`LocalExchange::offer_local_plan`].
pub(crate) enum FuseOffer {
    /// The plan is recorded as the exchange's only source; `Some` when that completed the
    /// receiver's sender set.
    Fused(Option<ReadyFragment>),
    /// Not deferred; the plan is handed back so the caller runs it. A leaf's caller still holds
    /// the params it offered and reads nothing here; the hand-back is for an offer that carries
    /// `inputs` (a middle fragment, the later `all` mode), which must run over them.
    Declined {
        #[cfg_attr(not(test), allow(dead_code))]
        plan: LocalPlan,
        skip: FuseSkip,
    },
}

#[derive(Debug)]
struct PendingReceiver {
    params: TExecPlanFragmentParams,
    expected_senders: HashMap<i32, usize>,
}

/// How many cancelled receivers are remembered; the oldest is forgotten first. A `const`, not a
/// knob: eviction only lets a very late frame re-create a source (today's behaviour), never
/// touches a live receiver, and StarRocks instance ids never recur.
const RETIRED_CAPACITY: usize = 1024;

#[derive(Debug, Default)]
struct ExchangeState {
    receivers: HashMap<FragmentInstanceId, PendingReceiver>,
    sources: HashMap<ExchangeKey, HashMap<i32, SenderSource>>,
    /// Next expected `transmit_packed` sequence number per remote sender: a duplicate (below) is
    /// dropped idempotently — brpc reconnect-retry can replay a frame — and a gap (above) is a
    /// lost frame, which must fail the query rather than silently drop rows.
    remote_seq: HashMap<(ExchangeKey, i32), i64>,
    /// Receivers the FE cancelled (`retire_receiver`), so a frame from a peer's still-draining
    /// sender is refused instead of re-creating the entry. `HashSet` twin of the FIFO so the
    /// per-frame check is O(1).
    retired: HashSet<FragmentInstanceId>,
    retired_order: VecDeque<FragmentInstanceId>,
}

/// Matches receiver-first StarRocks dispatch with later same-node sender results.
#[derive(Debug, Default)]
pub(crate) struct LocalExchange {
    inner: Mutex<ExchangeState>,
}

impl LocalExchange {
    /// Registers a receiver fragment, returning it when every exchange input is complete.
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
        let senders = state.sources.entry(key).or_default();
        if senders.contains_key(&sender_id) {
            return Err(format!("duplicate sender {sender_id} for exchange {key:?}"));
        }
        senders.insert(sender_id, source);
        Self::take_ready(&mut state, key.fragment_instance_id)
    }

    /// Offers a sender's plan for fusion into the pending receiver `key.fragment_instance_id` at
    /// exchange `key.node_id`, instead of running the sender.
    ///
    /// Under the lock: the receiver must be pending, declare the exchange, expect exactly one
    /// sender there, have no source for it yet, and `verdict(&receiver.params)` (the
    /// translator's structural checks against the registered, descriptor-resolved receiver) must
    /// pass. On success the plan is recorded as [`SenderSource::LocalPlan`] and `take_ready`
    /// decides whether the receiver is complete. A decline hands the plan back; nothing is
    /// cloned but what the verdict reads.
    ///
    /// # Errors
    /// A rendezvous inconsistency `take_ready` reports (more sources than expected).
    pub(crate) fn offer_local_plan(
        &self,
        key: ExchangeKey,
        sender_id: i32,
        plan: LocalPlan,
        verdict: impl FnOnce(&TExecPlanFragmentParams) -> Result<(), FusionRefusal>,
    ) -> Result<FuseOffer, String> {
        let mut state = self.lock();
        let skip = match state.receivers.get(&key.fragment_instance_id) {
            None => Some(FuseSkip::NoPendingReceiver),
            Some(receiver) => match receiver.expected_senders.get(&key.node_id) {
                None => Some(FuseSkip::ExchangeNotDeclared),
                Some(&expected) if expected != 1 => Some(FuseSkip::ReceiverExpectsMany(expected)),
                Some(_)
                    if state
                        .sources
                        .get(&key)
                        .is_some_and(|sources| !sources.is_empty()) =>
                {
                    Some(FuseSkip::ExchangeAlreadySourced)
                }
                Some(_) => verdict(&receiver.params).err().map(FuseSkip::Structural),
            },
        };
        if let Some(skip) = skip {
            return Ok(FuseOffer::Declined { plan, skip });
        }
        state
            .sources
            .entry(key)
            .or_default()
            .insert(sender_id, SenderSource::LocalPlan(Box::new(plan)));
        Self::take_ready(&mut state, key.fragment_instance_id).map(FuseOffer::Fused)
    }

    /// Records one `transmit_packed` frame from a remote sender: a staged batch, eos, or both.
    /// Returns the receiver when the eos completes its sender set.
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

        let expected_seq = state.remote_seq.entry((key, sender_id)).or_insert(0);
        if seq < *expected_seq {
            // brpc reconnect-retry can replay a frame the peer already processed; its batch (if
            // any) landed in the same lease the first delivery recorded, so dropping the replay
            // loses nothing and leaks nothing.
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
            // A frame that claims a batch must carry one: pack metadata for a packed frame,
            // at least one record batch for an Arrow frame.
            match &batch.arrow {
                None if batch.metadata.is_empty() => {
                    return Err(format!(
                        "remote sender {sender_id} for exchange {key:?} sent frame seq {seq} \
                         with empty pack metadata"
                    ));
                }
                Some(arrow) if arrow.is_empty() => {
                    return Err(format!(
                        "remote sender {sender_id} for exchange {key:?} sent Arrow frame seq \
                         {seq} with no record batches"
                    ));
                }
                _ => {}
            }
            batches.push(batch);
        }
        if eos {
            *closed = true;
        }
        Self::take_ready(&mut state, key.fragment_instance_id)
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
            // A remote sender counts only once its eos arrived; a local parked one is done by
            // construction.
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
        // The receiver is leaving the rendezvous; drop its remote-sender sequence tracking too.
        state
            .remote_seq
            .retain(|(key, _), _| key.fragment_instance_id != fragment_instance_id);
        Ok(Some(ReadyFragment {
            params: receiver.params,
            inputs,
        }))
    }

    /// Forgets a receiver the FE cancelled: its pending registration, every sender source recorded
    /// for it (all exchange nodes), and its remote sequence tracking. Returns the removed sources
    /// so the caller releases the staging leases of the remote ones. Remembers the id (bounded) so
    /// a frame from a peer's still-draining sender is refused instead of re-creating the entry.
    /// Idempotent; a receiver `take_ready` already released is a no-op plus the set insert.
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
                match state.retired_order.pop_front() {
                    Some(oldest) => {
                        state.retired.remove(&oldest);
                    }
                    None => break,
                }
            }
        }
        removed
    }

    /// Whether `retire_receiver` was called for this receiver (O(1)).
    pub(crate) fn is_retired(&self, fragment_instance_id: FragmentInstanceId) -> bool {
        self.lock().retired.contains(&fragment_instance_id)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, ExchangeState> {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use starrocks_thrift::internal_service::{InternalServiceVersion, TPlanFragmentExecParams};
    use starrocks_thrift::types::TUniqueId;

    use super::*;

    fn key(instance: u64, node_id: i32) -> ExchangeKey {
        ExchangeKey {
            fragment_instance_id: FragmentInstanceId::from_halves(7, instance as i64),
            node_id,
        }
    }

    /// The rendezvous only stores the params, so an empty shell is enough here.
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

    fn staged(fill: u8) -> StagedBatch {
        StagedBatch {
            metadata: vec![fill; 4],
            offset: u64::from(fill) * 1024,
            len: 512,
            rows: Some(u64::from(fill)),
            arrow: None,
            ipc_bytes: 0,
        }
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

    fn remote(sender_id: i32, closed: bool) -> SenderSource {
        SenderSource::Remote {
            names: names(),
            sender_id,
            batches: Vec::new(),
            closed,
        }
    }

    /// Receiver-first dispatch: the receiver registers, its senders park later in any order, and
    /// the fragment comes back on the push that completes the set, sources in sender-id order.
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
        assert_eq!(ready.inputs.len(), 1);
        assert_eq!(ready.inputs[0].node_id, 7);
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

    /// A receiver with several exchange inputs waits for every one of them; the ready inputs
    /// come out in node-id order, independent of completion order.
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

    /// Receiver registration refuses shapes that could never become ready or would be ambiguous.
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

    /// A sender that reports twice, or more senders than the receiver declared, is a bug in the
    /// sender set and is reported loudly rather than counted.
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

    /// Remote frames accumulate in sequence order; only the eos completes the sender, and the
    /// batches come out in arrival order.
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
        // The receiver registers mid-stream; still not ready, the sender has not closed.
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
        assert_eq!(ready.inputs.len(), 1);
        assert_eq!(ready.inputs[0].node_id, 7);
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
        assert_eq!(batches, &vec![staged(1), staged(2)]);
    }

    /// A replayed frame (brpc reconnect-retry) is dropped idempotently, even after eos.
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
        // The single staged batch survived, un-duplicated.
        let ready = exchange
            .register_receiver(key.fragment_instance_id, vec![(7, 1)], params())
            .unwrap()
            .expect("sender already complete");
        let SenderSource::Remote { batches, .. } = &ready.inputs[0].sources[0] else {
            panic!("expected a remote source");
        };
        assert_eq!(batches.len(), 1);
    }

    /// A sequence gap means a frame was lost; the query must fail rather than lose rows.
    #[test]
    fn remote_seq_gap_is_a_loud_error() {
        let exchange = LocalExchange::default();
        let key = key(3, 7);
        exchange
            .push_remote_frame(key, 0, 0, false, names(), Some(staged(1)))
            .unwrap();
        let err = exchange
            .push_remote_frame(key, 0, 2, false, names(), Some(staged(2)))
            .unwrap_err();
        assert!(err.contains("skipped from frame seq 1 to 2"), "{err}");
    }

    /// A new frame after eos is a protocol violation, never a silent append.
    #[test]
    fn remote_frame_after_eos_is_a_loud_error() {
        let exchange = LocalExchange::default();
        let key = key(4, 7);
        exchange
            .push_remote_frame(key, 0, 0, true, names(), None)
            .unwrap();
        let err = exchange
            .push_remote_frame(key, 0, 1, false, names(), Some(staged(1)))
            .unwrap_err();
        assert!(err.contains("after eos"), "{err}");
    }

    /// All frames of one sender must agree on the column names.
    #[test]
    fn remote_names_must_match_across_frames() {
        let exchange = LocalExchange::default();
        let key = key(5, 7);
        exchange
            .push_remote_frame(key, 0, 0, false, names(), Some(staged(1)))
            .unwrap();
        let err = exchange
            .push_remote_frame(key, 0, 1, false, vec!["other".to_string()], Some(staged(2)))
            .unwrap_err();
        assert!(err.contains("changed its column names"), "{err}");
    }

    /// Frames must carry names (the receiver binds its stream schema from them) and either a
    /// batch or eos.
    #[test]
    fn remote_frame_shape_is_validated() {
        let exchange = LocalExchange::default();
        let key = key(6, 7);
        let err = exchange
            .push_remote_frame(key, 0, 0, false, Vec::new(), Some(staged(1)))
            .unwrap_err();
        assert!(err.contains("without column names"), "{err}");
        let err = exchange
            .push_remote_frame(key, 0, 0, false, names(), None)
            .unwrap_err();
        assert!(err.contains("neither a batch nor eos"), "{err}");
    }

    /// A fan-in of one local parked sender and one remote sender becomes ready only when the
    /// local sender has parked AND the remote sender has closed.
    #[test]
    fn mixed_local_and_remote_sender_set() {
        let exchange = LocalExchange::default();
        let key = key(8, 7);
        assert!(
            exchange
                .register_receiver(key.fragment_instance_id, vec![(7, 2)], params())
                .unwrap()
                .is_none()
        );
        assert!(
            exchange
                .push_sender(key, 0, local(key, 0))
                .unwrap()
                .is_none()
        );
        assert!(
            exchange
                .push_remote_frame(key, 1, 0, false, names(), Some(staged(3)))
                .unwrap()
                .is_none(),
            "an open remote sender must not count as complete"
        );
        let ready = exchange
            .push_remote_frame(key, 1, 1, true, names(), None)
            .unwrap()
            .expect("both senders now complete");
        assert_eq!(ready.inputs[0].sources.len(), 2);
        assert!(matches!(
            ready.inputs[0].sources[0],
            SenderSource::LocalParked { .. }
        ));
        assert!(matches!(
            ready.inputs[0].sources[1],
            SenderSource::Remote { sender_id: 1, .. }
        ));
    }

    /// A remote frame colliding with an already-parked local sender of the same id is a bug in
    /// the sender set, reported loudly.
    #[test]
    fn remote_frame_colliding_with_local_sender_is_an_error() {
        let exchange = LocalExchange::default();
        let key = key(9, 7);
        exchange.push_sender(key, 0, local(key, 0)).unwrap();
        let err = exchange
            .push_remote_frame(key, 0, 0, false, names(), Some(staged(1)))
            .unwrap_err();
        assert!(err.contains("collides with a local parked sender"), "{err}");
    }

    /// A frame that claims a batch must carry one: a packed frame with empty pack metadata and
    /// an Arrow frame that decoded to no record batches (a schema-only IPC stream) are both
    /// refused, naming the frame.
    #[test]
    fn a_batch_frame_without_a_payload_is_refused() {
        let exchange = LocalExchange::default();
        let packed = StagedBatch {
            metadata: Vec::new(),
            ..staged(1)
        };
        let err = exchange
            .push_remote_frame(key(12, 7), 0, 0, false, names(), Some(packed))
            .unwrap_err();
        assert!(
            err.contains("frame seq 0 with empty pack metadata"),
            "{err}"
        );

        let arrow = StagedBatch::arrow(Vec::new(), 0);
        assert_eq!(arrow.rows, Some(0));
        let err = exchange
            .push_remote_frame(key(13, 7), 0, 0, false, names(), Some(arrow))
            .unwrap_err();
        assert!(
            err.contains("Arrow frame seq 0 with no record batches"),
            "{err}"
        );
    }

    /// An open remote sender is present but not finished; it must not count toward readiness.
    #[test]
    fn an_open_remote_sender_does_not_complete_the_set() {
        let exchange = LocalExchange::default();
        let key = key(9, 7);
        assert!(
            exchange
                .push_sender(key, 0, remote(0, false))
                .unwrap()
                .is_none()
        );
        assert!(
            exchange
                .register_receiver(key.fragment_instance_id, vec![(7, 1)], params())
                .unwrap()
                .is_none(),
            "an open remote sender must not count as complete"
        );
    }

    /// The receiver binds its input stream schema from the sender's names, whichever side of
    /// the wire the sender sits on.
    #[test]
    fn names_are_the_same_on_either_side_of_the_wire() {
        let key = key(10, 7);
        assert_eq!(local(key, 0).names(), names().as_slice());
        assert_eq!(remote(0, false).names(), names().as_slice());
    }

    /// A cancelled receiver leaves the rendezvous with every source recorded for it (so the
    /// caller can release the remote ones' leases), is remembered as retired, and its remote
    /// sequence tracking is gone; other receivers are unaffected.
    #[test]
    fn retire_receiver_returns_its_sources_and_refuses_later_frames() {
        let exchange = LocalExchange::default();
        let other = key(11, 7);
        let key = key(10, 7);
        assert!(
            exchange
                .register_receiver(key.fragment_instance_id, vec![(7, 2)], params())
                .unwrap()
                .is_none()
        );
        assert!(
            exchange
                .push_sender(
                    key,
                    0,
                    SenderSource::LocalParked {
                        names: names(),
                        slot: local_slot(key, 0),
                    },
                )
                .unwrap()
                .is_none()
        );
        assert!(
            exchange
                .push_remote_frame(key, 1, 0, false, names(), Some(staged(1)))
                .unwrap()
                .is_none()
        );

        let mut removed = exchange.retire_receiver(key.fragment_instance_id);
        assert_eq!(removed.len(), 2, "{removed:?}");
        assert!(matches!(removed[0], SenderSource::LocalParked { .. }));
        let SenderSource::Remote {
            batches, closed, ..
        } = removed.remove(1)
        else {
            panic!("expected a remote source");
        };
        assert_eq!(batches, vec![staged(1)]);
        assert!(!closed, "an open remote source is returned too");
        assert!(exchange.is_retired(key.fragment_instance_id));

        // Idempotent: nothing left to remove.
        assert!(
            exchange
                .retire_receiver(key.fragment_instance_id)
                .is_empty()
        );
        // The sequence tracking went with it: a frame for the retired receiver is judged from
        // seq 0 again (the service refuses such frames before they get here; this pins the state).
        let err = exchange
            .push_remote_frame(key, 1, 5, false, names(), Some(staged(2)))
            .unwrap_err();
        assert!(err.contains("skipped from frame seq 0 to 5"), "{err}");

        // Another receiver is unaffected.
        assert!(
            exchange
                .register_receiver(other.fragment_instance_id, vec![(7, 1)], params())
                .unwrap()
                .is_none()
        );
        assert!(
            exchange
                .push_sender(
                    other,
                    0,
                    SenderSource::LocalParked {
                        names: names(),
                        slot: local_slot(other, 0),
                    },
                )
                .unwrap()
                .is_some()
        );
        assert!(!exchange.is_retired(other.fragment_instance_id));
    }

    /// The retired set is bounded; the oldest entry is forgotten first.
    #[test]
    fn retired_receivers_are_bounded() {
        let exchange = LocalExchange::default();
        for i in 0..=RETIRED_CAPACITY {
            exchange.retire_receiver(FragmentInstanceId::from_halves(12, i as i64));
        }
        assert!(!exchange.is_retired(FragmentInstanceId::from_halves(12, 0)));
        assert!(exchange.is_retired(FragmentInstanceId::from_halves(12, 1)));
        assert!(exchange.is_retired(FragmentInstanceId::from_halves(12, RETIRED_CAPACITY as i64)));
    }

    // Fragment fusion: a sender's PLAN offered as the exchange's source instead of its rows.

    /// A deferred sender plan of instance `(7, instance)` in query `(7, 0)`, with no inputs.
    fn local_plan(instance: i64) -> LocalPlan {
        let mut params = params();
        params.params = Some(TPlanFragmentExecParams::new(
            TUniqueId::new(7, 0),
            TUniqueId::new(7, instance),
            BTreeMap::new(),
            BTreeMap::new(),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ));
        LocalPlan {
            params,
            inputs: Vec::new(),
        }
    }

    fn instance_of(plan: &LocalPlan) -> FragmentInstanceId {
        FragmentInstanceId::from(&plan.params.params.as_ref().unwrap().fragment_instance_id)
    }

    fn parked(key: ExchangeKey, sender_id: i32) -> SenderSource {
        SenderSource::LocalParked {
            names: names(),
            slot: local_slot(key, sender_id),
        }
    }

    fn declined(offer: FuseOffer) -> (LocalPlan, FuseSkip) {
        match offer {
            FuseOffer::Declined { plan, skip } => (plan, skip),
            FuseOffer::Fused(_) => panic!("expected the offer to be declined"),
        }
    }

    /// A receiver expecting one sender at exchange 7 becomes ready the moment that sender's plan
    /// is deferred: the plan is the exchange's only source and the receiver has left the
    /// rendezvous.
    #[test]
    fn offer_local_plan_completes_a_single_sender_receiver() {
        let exchange = LocalExchange::default();
        let key = key(20, 7);
        assert!(
            exchange
                .register_receiver(key.fragment_instance_id, vec![(7, 1)], params())
                .unwrap()
                .is_none()
        );
        let FuseOffer::Fused(ready) = exchange
            .offer_local_plan(key, 0, local_plan(21), |_| Ok(()))
            .unwrap()
        else {
            panic!("expected the offer to be accepted");
        };
        let ready = ready.expect("the deferred plan completes the sender set");
        assert_eq!(ready.inputs.len(), 1);
        assert_eq!(ready.inputs[0].node_id, 7);
        let [SenderSource::LocalPlan(plan)] = ready.inputs[0].sources.as_slice() else {
            panic!("expected exactly one deferred plan source");
        };
        assert_eq!(instance_of(plan), FragmentInstanceId::from_halves(7, 21));
        assert!(plan.inputs.is_empty());
        // The receiver left the rendezvous: a second offer finds nothing pending.
        let (_, skip) = declined(
            exchange
                .offer_local_plan(key, 0, local_plan(22), |_| Ok(()))
                .unwrap(),
        );
        assert!(matches!(skip, FuseSkip::NoPendingReceiver), "{skip}");
    }

    /// A leaf that arrives before its receiver is handed back to run and park as today, and
    /// leaves nothing behind in the rendezvous.
    #[test]
    fn offer_local_plan_declines_without_pending_receiver() {
        let exchange = LocalExchange::default();
        let key = key(23, 7);
        let (plan, skip) = declined(
            exchange
                .offer_local_plan(key, 0, local_plan(24), |_| Ok(()))
                .unwrap(),
        );
        assert!(matches!(skip, FuseSkip::NoPendingReceiver), "{skip}");
        assert_eq!(skip.to_string(), "NoPendingReceiver");
        assert_eq!(
            instance_of(&plan),
            FragmentInstanceId::from_halves(7, 24),
            "the plan offered is the plan handed back"
        );
        // Nothing was recorded: the receiver registering afterwards is not ready.
        assert!(
            exchange
                .register_receiver(key.fragment_instance_id, vec![(7, 1)], params())
                .unwrap()
                .is_none()
        );
    }

    /// Only a single-sender exchange is the identity on rows; a fan-in declines and the parked
    /// path still completes it.
    #[test]
    fn offer_local_plan_declines_when_receiver_expects_many() {
        let exchange = LocalExchange::default();
        let key = key(25, 7);
        exchange
            .register_receiver(key.fragment_instance_id, vec![(7, 2)], params())
            .unwrap();
        let (_, skip) = declined(
            exchange
                .offer_local_plan(key, 0, local_plan(26), |_| Ok(()))
                .unwrap(),
        );
        assert!(matches!(skip, FuseSkip::ReceiverExpectsMany(2)), "{skip}");
        assert_eq!(skip.to_string(), "ReceiverExpectsMany(2)");
        assert!(
            exchange
                .push_sender(key, 0, parked(key, 0))
                .unwrap()
                .is_none()
        );
        assert!(
            exchange
                .push_sender(key, 1, parked(key, 1))
                .unwrap()
                .is_some()
        );
    }

    /// The verdict sees the registered receiver's params; a refusal records nothing, so the
    /// parked path readies the receiver exactly as today.
    #[test]
    fn offer_local_plan_declines_on_structural_verdict_and_leaves_receiver_untouched() {
        let exchange = LocalExchange::default();
        let key = key(27, 7);
        let mut receiver = params();
        receiver.backend_num = Some(42);
        exchange
            .register_receiver(key.fragment_instance_id, vec![(7, 1)], receiver)
            .unwrap();
        let (_, skip) = declined(
            exchange
                .offer_local_plan(key, 0, local_plan(28), |receiver| {
                    assert_eq!(
                        receiver.backend_num,
                        Some(42),
                        "verdict over the registered params"
                    );
                    Err(FusionRefusal::SortedExchange { node_id: 7 })
                })
                .unwrap(),
        );
        assert!(
            matches!(
                skip,
                FuseSkip::Structural(FusionRefusal::SortedExchange { node_id: 7 })
            ),
            "{skip}"
        );
        assert_eq!(
            skip.to_string(),
            "Structural(exchange 7 merges sorted input)"
        );
        let ready = exchange
            .push_sender(key, 0, parked(key, 0))
            .unwrap()
            .expect("the parked sender completes the set");
        assert!(matches!(
            ready.inputs[0].sources.as_slice(),
            [SenderSource::LocalParked { .. }]
        ));
    }

    /// A pending receiver that does not declare the addressed exchange is a malformed dispatch;
    /// the offer declines rather than inventing an expectation.
    #[test]
    fn offer_local_plan_declines_an_undeclared_exchange() {
        let exchange = LocalExchange::default();
        let receiver = key(29, 7);
        exchange
            .register_receiver(receiver.fragment_instance_id, vec![(7, 1)], params())
            .unwrap();
        let (_, skip) = declined(
            exchange
                .offer_local_plan(key(29, 8), 0, local_plan(30), |_| Ok(()))
                .unwrap(),
        );
        assert!(matches!(skip, FuseSkip::ExchangeNotDeclared), "{skip}");
    }

    /// The guard for a source that reached the exchange before the offer (with one expected
    /// sender that is a misrouted frame, not a fan-in): the plan declines, the source stays.
    #[test]
    fn offer_local_plan_declines_an_already_sourced_exchange() {
        let exchange = LocalExchange::default();
        let key = key(31, 7);
        exchange
            .register_receiver(key.fragment_instance_id, vec![(7, 1)], params())
            .unwrap();
        assert!(
            exchange
                .push_remote_frame(key, 0, 0, false, names(), Some(staged(1)))
                .unwrap()
                .is_none()
        );
        let (_, skip) = declined(
            exchange
                .offer_local_plan(key, 1, local_plan(32), |_| Ok(()))
                .unwrap(),
        );
        assert!(matches!(skip, FuseSkip::ExchangeAlreadySourced), "{skip}");
    }

    /// A receiver over one deferred and one parked exchange becomes ready when the parked one
    /// lands, with both inputs in node-id order.
    #[test]
    fn mixed_deferred_and_parked_inputs_become_ready_together() {
        let exchange = LocalExchange::default();
        let deferred = key(33, 7);
        let parked_key = key(33, 8);
        exchange
            .register_receiver(
                deferred.fragment_instance_id,
                vec![(8, 1), (7, 1)],
                params(),
            )
            .unwrap();
        let FuseOffer::Fused(None) = exchange
            .offer_local_plan(deferred, 0, local_plan(34), |_| Ok(()))
            .unwrap()
        else {
            panic!("exchange 8 is still pending, so the receiver is not ready");
        };
        let ready = exchange
            .push_sender(parked_key, 0, parked(parked_key, 0))
            .unwrap()
            .expect("both inputs are complete");
        let node_ids: Vec<i32> = ready.inputs.iter().map(|input| input.node_id).collect();
        assert_eq!(node_ids, vec![7, 8]);
        assert!(matches!(
            ready.inputs[0].sources.as_slice(),
            [SenderSource::LocalPlan(_)]
        ));
        assert!(matches!(
            ready.inputs[1].sources.as_slice(),
            [SenderSource::LocalParked { .. }]
        ));
    }

    /// The cancel path: the cancelled receiver leaves with its deferred plan among the other
    /// sources (a plan holds no GPU memory, so the caller releases nothing for it), is remembered
    /// as retired, and a later offer for it declines instead of re-creating it.
    #[test]
    fn retire_receiver_returns_a_deferred_plan_with_the_other_sources() {
        let exchange = LocalExchange::default();
        let deferred = key(35, 7);
        let parked_key = key(35, 8);
        exchange
            .register_receiver(
                deferred.fragment_instance_id,
                vec![(7, 1), (8, 2)],
                params(),
            )
            .unwrap();
        let FuseOffer::Fused(None) = exchange
            .offer_local_plan(deferred, 0, local_plan(36), |_| Ok(()))
            .unwrap()
        else {
            panic!("exchange 8 is still pending, so the receiver is not ready");
        };
        assert!(
            exchange
                .push_sender(parked_key, 0, parked(parked_key, 0))
                .unwrap()
                .is_none()
        );

        let removed = exchange.retire_receiver(deferred.fragment_instance_id);
        assert_eq!(removed.len(), 2, "{removed:?}");
        let SenderSource::LocalPlan(plan) = &removed[0] else {
            panic!("exchange 7's source is the deferred plan: {removed:?}");
        };
        assert_eq!(instance_of(plan), FragmentInstanceId::from_halves(7, 36));
        assert!(matches!(removed[1], SenderSource::LocalParked { .. }));
        assert!(exchange.is_retired(deferred.fragment_instance_id));
        let (_, skip) = declined(
            exchange
                .offer_local_plan(deferred, 0, local_plan(37), |_| Ok(()))
                .unwrap(),
        );
        assert!(matches!(skip, FuseSkip::NoPendingReceiver), "{skip}");
    }

    /// The derived `Debug` of the params is the whole fragment dump; a plan's is a summary.
    #[test]
    fn local_plan_debug_is_a_summary_not_the_fragment_dump() {
        let plan = local_plan(38);
        let debug = format!("{plan:?}");
        assert!(
            debug.contains(&FragmentInstanceId::from_halves(7, 38).to_string()),
            "{debug}"
        );
        assert!(debug.contains("plan_nodes: 0"), "{debug}");
        assert!(debug.contains("inputs: 0"), "{debug}");
        assert!(!debug.contains("protocol_version"), "{debug}");
        assert!(debug.len() < 300, "{debug}");
    }
}
