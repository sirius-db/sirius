// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Task FSM analysis types.
//!
//! With `FsmEvents<T>` providing all generic trait impls (`Entity`, `Fsm`,
//! `FsmUsages`, `Using`, `FsmTypeDeclaration`), the task analyzer is just
//! type aliases plus application-specific helper methods.

use instrumentation_model::task::TaskTransition as ModelTaskTransition;
use quent_analyzer::{
    AnalyzerResult, Entity,
    fsm::{
        Transition,
        events::{FsmEvents, FsmEventsBuilder},
    },
};
use quent_query_engine_ui::OperatorFilter;
use quent_time::{TimeUnixNanoSec, Timestamp, span::SpanUnixNanoSec, to_secs_relative};
use quent_ui::{FiniteStateMachine, FsmTransition, FsmUsage};
use uuid::Uuid;

/// The reconstructed Task FSM.
pub type Task = FsmEvents<ModelTaskTransition>;

/// Builder for Task FSMs.
pub type TaskBuilder = FsmEventsBuilder<ModelTaskTransition>;

/// Application-specific methods on the Task FSM.
pub trait TaskExt {
    fn pipeline_uuid(&self) -> Option<Uuid>;
    fn executes_physical_operation(&self, physical_operator_id: u32) -> bool;
    fn matches_filter(&self, filter: &OperatorFilter) -> bool;
    fn active_span(&self) -> Option<SpanUnixNanoSec>;
    /// `(operator_uuid, active interval)` for each `Computing` run: from a `Computing`
    /// transition to the next transition of any kind. A task computes its operators
    /// sequentially, so this splits the task's activity into one span per operator.
    fn operator_active_spans(&self) -> Vec<(Uuid, SpanUnixNanoSec)>;
    /// The stable UUIDs of the physical operators this task computed, in transition order.
    fn computed_operator_uuids(&self) -> impl Iterator<Item = Uuid> + '_;
    fn try_to_ui_fsm(&self, epoch: TimeUnixNanoSec) -> AnalyzerResult<FiniteStateMachine>;
}

impl TaskExt for Task {
    fn pipeline_uuid(&self) -> Option<Uuid> {
        self.first_data().and_then(|t| match t {
            ModelTaskTransition::Created(data) => Some(data.pipeline_uuid),
            _ => None,
        })
    }

    fn executes_physical_operation(&self, physical_operator_id: u32) -> bool {
        self.transitions().iter().any(|transition| {
            matches!(
                &transition.data,
                ModelTaskTransition::Computing(data)
                    if data.current_operator_id == physical_operator_id
            )
        })
    }

    fn matches_filter(&self, filter: &OperatorFilter) -> bool {
        // Operator entities are declared per physical operator, so an operator filter matches
        // any task that computed that operator (`Computing.operator_uuid`).
        filter
            .operator_id
            .is_none_or(|operator_uuid| self.computed_operator_uuids().any(|u| u == operator_uuid))
    }

    fn active_span(&self) -> Option<SpanUnixNanoSec> {
        let start = self.transitions().get(1)?.timestamp();
        let end = self.transitions().last()?.timestamp();
        SpanUnixNanoSec::try_new(start, end).ok()
    }

    fn operator_active_spans(&self) -> Vec<(Uuid, SpanUnixNanoSec)> {
        let transitions = self.transitions();
        transitions
            .iter()
            .enumerate()
            .filter_map(|(i, transition)| {
                let ModelTaskTransition::Computing(data) = &transition.data else {
                    return None;
                };
                // The operator ran from this Computing transition until the task's next
                // transition (another Computing, or Finalizing). A trailing Computing with no
                // successor contributes no measurable span.
                let next = transitions.get(i + 1)?;
                let span =
                    SpanUnixNanoSec::try_new(transition.timestamp(), next.timestamp()).ok()?;
                Some((data.operator_uuid, span))
            })
            .collect()
    }

    fn computed_operator_uuids(&self) -> impl Iterator<Item = Uuid> + '_ {
        self.transitions()
            .iter()
            .filter_map(|transition| match &transition.data {
                ModelTaskTransition::Computing(data) => Some(data.operator_uuid),
                _ => None,
            })
    }

    fn try_to_ui_fsm(&self, epoch: TimeUnixNanoSec) -> AnalyzerResult<FiniteStateMachine> {
        let transitions = self
            .transitions()
            .iter()
            .map(|transition| {
                Ok(FsmTransition {
                    name: transition.name().to_string(),
                    usages: transition
                        .usages
                        .iter()
                        .map(|usage| FsmUsage {
                            resource: usage.resource_id,
                            capacities: usage
                                .capacities
                                .iter()
                                .map(|capacity| (capacity.name.to_string(), capacity.value))
                                .collect(),
                        })
                        .collect(),
                    timestamp: to_secs_relative(transition.timestamp(), epoch),
                })
            })
            .collect::<AnalyzerResult<Vec<_>>>()?;

        Ok(FiniteStateMachine {
            id: self.id(),
            type_name: self.type_name().to_string(),
            instance_name: self.instance_name().to_string(),
            transitions,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use instrumentation_model::task::{Computing, Finalizing, TaskTransition};
    use quent_events::Event;
    use quent_model::FsmEvent;

    fn computing(operator_uuid: Uuid, operator_id: u32) -> TaskTransition {
        TaskTransition::Computing(Computing {
            instance_name: format!("op-{operator_id}"),
            current_operator_id: operator_id,
            operator_uuid,
            input_bytes: 0,
            executor_thread: None,
        })
    }

    /// A task that computes two operators in sequence yields one span per operator, each running
    /// from its `Computing` transition to the next transition.
    #[test]
    fn operator_active_spans_splits_per_operator() {
        let task_id = Uuid::from_u128(1);
        let op_a = Uuid::from_u128(0xaa);
        let op_b = Uuid::from_u128(0xbb);

        let mut builder = TaskBuilder::try_new(task_id).unwrap();
        builder.push(Event::new(
            task_id,
            100,
            FsmEvent {
                seq: 0,
                state: computing(op_a, 0),
            },
        ));
        builder.push(Event::new(
            task_id,
            200,
            FsmEvent {
                seq: 1,
                state: computing(op_b, 1),
            },
        ));
        builder.push(Event::new(
            task_id,
            300,
            FsmEvent {
                seq: 2,
                state: TaskTransition::Finalizing(Finalizing {
                    instance_name: "done".to_string(),
                    success: true,
                }),
            },
        ));

        let task = builder.try_build().unwrap();
        let spans = task.operator_active_spans();

        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].0, op_a);
        assert_eq!(spans[0].1.start(), 100);
        assert_eq!(spans[0].1.end(), 200);
        assert_eq!(spans[1].0, op_b);
        assert_eq!(spans[1].1.start(), 200);
        assert_eq!(spans[1].1.end(), 300);

        assert!(task.computed_operator_uuids().eq([op_a, op_b]));
        assert!(task.matches_filter(&OperatorFilter {
            operator_id: Some(op_a)
        }));
        assert!(!task.matches_filter(&OperatorFilter {
            operator_id: Some(Uuid::from_u128(0xcc))
        }));
        assert!(task.matches_filter(&OperatorFilter { operator_id: None }));
    }

    /// A trailing `Computing` transition with no successor contributes no span (it has no
    /// measurable end), while earlier operators still produce spans.
    #[test]
    fn operator_active_spans_ignores_trailing_computing() {
        let task_id = Uuid::from_u128(2);
        let op_a = Uuid::from_u128(0xaa);

        let mut builder = TaskBuilder::try_new(task_id).unwrap();
        builder.push(Event::new(
            task_id,
            100,
            FsmEvent {
                seq: 0,
                state: computing(op_a, 0),
            },
        ));

        let task = builder.try_build().unwrap();
        assert!(task.operator_active_spans().is_empty());
    }
}
