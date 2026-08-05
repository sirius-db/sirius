//! Sequential same-node exchange rendezvous.
//!
//! The exchange matches StarRocks' receiver-first dispatch with the senders that arrive later. It
//! carries no data: a sender's rows stay on the GPU, parked in the engine as native batches, and
//! what is tracked here is only which senders have produced and where their output sits.

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

/// One sender that has produced, and where its output sits.
///
/// An enum so the cross-node transport (PLAN-PATH-B B5) can add a `Remote` variant carrying
/// received batches next to the local GPU-parked variant.
#[derive(Clone, Debug)]
pub(crate) enum SenderSource {
    /// A same-node sender whose batches the engine parked on the GPU.
    LocalParked {
        /// Sender output names, which become the input stream's column names.
        names: Vec<String>,
        /// Where the engine parked this sender's batches.
        slot: SenderSlot,
    },
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
            let actual = state.sources.get(&key).map(HashMap::len).unwrap_or(0);
            if actual > expected {
                return Err(format!(
                    "exchange {key:?} received {actual} senders but expected {expected}"
                ));
            }
            if actual != expected {
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
