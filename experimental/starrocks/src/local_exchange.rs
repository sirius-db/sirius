//! Sequential exchange rendezvous.
//!
//! The exchange matches StarRocks' receiver-first dispatch with the senders that arrive later.
//! A same-node sender's rows stay on the GPU, parked in the engine as native batches; a remote
//! sender's batches arrive as staged packed bytes in this CN's arena (nixl tier, PLAN-PATH-B B5).
//! Either way, what is tracked here is which senders have finished and where their output sits.

use std::collections::HashMap;
use std::sync::Mutex;

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
    /// A remote sender whose packed batches were nixl-WRITTEN into this CN's staging arena and
    /// announced over `transmit_packed`. Counts toward readiness only once `closed`.
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
}

impl SenderSource {
    /// The sender's output column names, whichever side of the wire it sits on.
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

#[derive(Debug, Default)]
struct ExchangeState {
    receivers: HashMap<FragmentInstanceId, PendingReceiver>,
    sources: HashMap<ExchangeKey, HashMap<i32, SenderSource>>,
    /// Next expected `transmit_packed` sequence number per remote sender: a duplicate (below) is
    /// dropped idempotently — brpc reconnect-retry can replay a frame — and a gap (above) is a
    /// lost frame, which must fail the query rather than silently drop rows.
    remote_seq: HashMap<(ExchangeKey, i32), i64>,
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
            if batch.metadata.is_empty() {
                return Err(format!(
                    "remote sender {sender_id} for exchange {key:?} sent frame seq {seq} with \
                     empty pack metadata"
                ));
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
        }
    }

    fn local_slot(key: ExchangeKey, sender_id: i32) -> SenderSlot {
        SenderSlot {
            fragment_instance_id: key.fragment_instance_id,
            node_id: key.node_id,
            sender_id,
        }
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
        exchange
            .push_sender(
                key,
                0,
                SenderSource::LocalParked {
                    names: names(),
                    slot: local_slot(key, 0),
                },
            )
            .unwrap();
        let err = exchange
            .push_remote_frame(key, 0, 0, false, names(), Some(staged(1)))
            .unwrap_err();
        assert!(err.contains("collides with a local parked sender"), "{err}");
    }
}
