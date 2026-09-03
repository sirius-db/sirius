//! Sequential exchange rendezvous.
//!
//! The exchange matches StarRocks' receiver-first dispatch with the senders that arrive later.
//! A same-node sender's rows stay on the GPU, parked in the engine as native batches; a remote
//! sender counts toward readiness only once it has announced eos. Either way, what is tracked
//! here is which senders have finished and where their output sits.

use std::collections::HashMap;
use std::sync::Mutex;

use starrocks_thrift::internal_service::TExecPlanFragmentParams;

use crate::fragment_executor::SenderSlot;
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
    /// A remote sender whose batches arrive over the cross-CN transport tier. Counts toward
    /// readiness only once `closed`.
    // Constructed by the cross-node transport tier (`push_remote_frame`); this base only matches
    // on it, in `names()` and `is_complete()`.
    #[allow(dead_code)]
    Remote {
        /// Sender output names carried on every transmitted frame (first frame wins, later
        /// frames must match).
        names: Vec<String>,
        /// Sender ordinal, needed to `close_input` the engine stream it feeds.
        sender_id: i32,
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
        let ready = exchange
            .push_sender(key, 1, remote(1, true))
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
}
