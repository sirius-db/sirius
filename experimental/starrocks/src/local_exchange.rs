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

/// One sender that has produced, and where the engine parked its output.
#[derive(Clone, Debug)]
pub(crate) struct SenderOutput {
    /// Sender output names, which become the input stream's column names.
    pub(crate) names: Vec<String>,
    /// Where the engine parked this sender's batches.
    pub(crate) slot: SenderSlot,
}

/// One exchange input of a receiver fragment whose sender set is complete.
#[derive(Debug)]
pub(crate) struct ReadyExchangeInput {
    pub(crate) node_id: i32,
    pub(crate) outputs: Vec<SenderOutput>,
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
    outputs: HashMap<ExchangeKey, HashMap<i32, SenderOutput>>,
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
        output: SenderOutput,
    ) -> Result<Option<ReadyFragment>, String> {
        let mut state = self.lock();
        let senders = state.outputs.entry(key).or_default();
        if senders.contains_key(&sender_id) {
            return Err(format!("duplicate sender {sender_id} for exchange {key:?}"));
        }
        senders.insert(sender_id, output);
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
            let actual = state.outputs.get(&key).map(HashMap::len).unwrap_or(0);
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
                let mut senders = state.outputs.remove(&key).unwrap_or_default();
                let mut sender_ids = senders.keys().copied().collect::<Vec<_>>();
                sender_ids.sort_unstable();
                let outputs = sender_ids
                    .into_iter()
                    .map(|sender_id| senders.remove(&sender_id).expect("sender id came from map"))
                    .collect();
                ReadyExchangeInput { node_id, outputs }
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

    fn fiid(n: i64) -> FragmentInstanceId {
        FragmentInstanceId::from_halves(0, n)
    }

    fn params() -> TExecPlanFragmentParams {
        TExecPlanFragmentParams::new(
            InternalServiceVersion::V1,
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
        )
    }

    fn output(receiver: FragmentInstanceId, node_id: i32, sender_id: i32) -> SenderOutput {
        SenderOutput {
            names: vec!["a".to_string()],
            slot: SenderSlot {
                fragment_instance_id: receiver,
                node_id,
                sender_id,
            },
        }
    }

    fn key(receiver: FragmentInstanceId, node_id: i32) -> ExchangeKey {
        ExchangeKey {
            fragment_instance_id: receiver,
            node_id,
        }
    }

    #[test]
    fn a_receiver_completes_when_its_last_sender_arrives() {
        let exchange = LocalExchange::default();
        let receiver = fiid(1);

        assert!(
            exchange
                .register_receiver(receiver, vec![(7, 2)], params())
                .unwrap()
                .is_none(),
            "no sender has produced yet"
        );
        assert!(
            exchange
                .push_sender(key(receiver, 7), 0, output(receiver, 7, 0))
                .unwrap()
                .is_none(),
            "one of two senders is not a complete set"
        );

        let ready = exchange
            .push_sender(key(receiver, 7), 1, output(receiver, 7, 1))
            .unwrap()
            .expect("second sender completes the set");
        assert_eq!(ready.inputs.len(), 1);
        assert_eq!(ready.inputs[0].node_id, 7);
        assert_eq!(ready.inputs[0].outputs.len(), 2);
    }

    #[test]
    fn senders_that_arrive_first_park_until_the_receiver_registers() {
        let exchange = LocalExchange::default();
        let receiver = fiid(2);

        // StarRocks dispatches receiver-first, but nothing guarantees the CN observes that
        // order; a sender completing before registration must park, not vanish.
        assert!(
            exchange
                .push_sender(key(receiver, 3), 0, output(receiver, 3, 0))
                .unwrap()
                .is_none()
        );

        let ready = exchange
            .register_receiver(receiver, vec![(3, 1)], params())
            .unwrap()
            .expect("registration finds the parked sender");
        assert_eq!(ready.inputs[0].outputs[0].slot.sender_id, 0);
    }

    #[test]
    fn a_duplicate_sender_id_is_rejected_not_counted() {
        let exchange = LocalExchange::default();
        let receiver = fiid(3);

        exchange
            .register_receiver(receiver, vec![(5, 2)], params())
            .unwrap();
        exchange
            .push_sender(key(receiver, 5), 0, output(receiver, 5, 0))
            .unwrap();

        // A retransmit counted as a second sender would mark the set complete and run the
        // receiver with half its input missing.
        let err = exchange
            .push_sender(key(receiver, 5), 0, output(receiver, 5, 0))
            .unwrap_err();
        assert!(err.contains("duplicate sender"), "{err}");
    }

    #[test]
    fn malformed_receiver_registrations_are_rejected() {
        let exchange = LocalExchange::default();

        let err = exchange
            .register_receiver(fiid(4), vec![], params())
            .unwrap_err();
        assert!(err.contains("no exchange inputs"), "{err}");

        let err = exchange
            .register_receiver(fiid(4), vec![(1, 0)], params())
            .unwrap_err();
        assert!(err.contains("expects no senders"), "{err}");

        let err = exchange
            .register_receiver(fiid(4), vec![(1, 1), (1, 2)], params())
            .unwrap_err();
        assert!(err.contains("duplicate exchange node"), "{err}");

        exchange
            .register_receiver(fiid(4), vec![(1, 1)], params())
            .unwrap();
        let err = exchange
            .register_receiver(fiid(4), vec![(2, 1)], params())
            .unwrap_err();
        assert!(err.contains("duplicate receiver"), "{err}");
    }

    #[test]
    fn more_senders_than_expected_is_an_error_not_a_ready_fragment() {
        let exchange = LocalExchange::default();
        let receiver = fiid(5);

        // A second exchange (node 8) keeps the receiver pending, so the surplus on node 9 is
        // observed rather than the exact match consuming the receiver first.
        exchange
            .register_receiver(receiver, vec![(9, 2), (8, 1)], params())
            .unwrap();
        exchange
            .push_sender(key(receiver, 9), 0, output(receiver, 9, 0))
            .unwrap();
        exchange
            .push_sender(key(receiver, 9), 1, output(receiver, 9, 1))
            .unwrap();
        // Readiness is exact-match: the surplus sender means the plan and the dispatch
        // disagree about the world, and running anyway would silently drop its output.
        let err = exchange
            .push_sender(key(receiver, 9), 2, output(receiver, 9, 2))
            .unwrap_err();
        assert!(err.contains("expected 2"), "{err}");
    }

    #[test]
    fn ready_inputs_and_outputs_come_back_sorted() {
        let exchange = LocalExchange::default();
        let receiver = fiid(6);

        exchange
            .register_receiver(receiver, vec![(11, 2), (4, 1)], params())
            .unwrap();
        // Arrival order scrambled on purpose: readiness must not depend on it, and the
        // returned order must be deterministic for the sequential executor.
        exchange
            .push_sender(key(receiver, 11), 1, output(receiver, 11, 1))
            .unwrap();
        exchange
            .push_sender(key(receiver, 4), 0, output(receiver, 4, 0))
            .unwrap();
        let ready = exchange
            .push_sender(key(receiver, 11), 0, output(receiver, 11, 0))
            .unwrap()
            .expect("all sender sets complete");

        let node_ids: Vec<i32> = ready.inputs.iter().map(|input| input.node_id).collect();
        assert_eq!(node_ids, vec![4, 11]);
        let sender_ids: Vec<i32> = ready.inputs[1]
            .outputs
            .iter()
            .map(|output| output.slot.sender_id)
            .collect();
        assert_eq!(sender_ids, vec![0, 1]);
    }
}
