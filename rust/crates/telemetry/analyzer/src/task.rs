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
use quent_attributes::Attribute;
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
    fn try_to_ui_fsm(
        &self,
        epoch: TimeUnixNanoSec,
        pipeline_name: Option<&str>,
    ) -> AnalyzerResult<FiniteStateMachine>;
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
        filter
            .operator_id
            .is_none_or(|pipeline_uuid| self.pipeline_uuid() == Some(pipeline_uuid))
    }

    fn active_span(&self) -> Option<SpanUnixNanoSec> {
        let start = self.transitions().get(1)?.timestamp();
        let end = self.transitions().last()?.timestamp();
        SpanUnixNanoSec::try_new(start, end).ok()
    }

    fn try_to_ui_fsm(
        &self,
        epoch: TimeUnixNanoSec,
        pipeline_name: Option<&str>,
    ) -> AnalyzerResult<FiniteStateMachine> {
        let raw = self.transitions();
        let transitions = raw
            .iter()
            .enumerate()
            .map(|(i, transition)| {
                // Derive the processing rate from input_bytes over the span.
                let mut derived_attributes = vec![];
                if let ModelTaskTransition::Computing(data) = &transition.data
                    && let Some(next) = raw.get(i + 1)
                {
                    let span_secs = (next.timestamp() - transition.timestamp()) as f64 / 1e9;
                    if span_secs > 0.0 {
                        derived_attributes.push(Attribute::f64(
                            "bytes_per_sec",
                            data.input_bytes as f64 / span_secs,
                        ));
                    }
                }
                // Stamped on every transition — created, which carries
                // pipeline_uuid, is filtered off resource lanes.
                if let Some(name) = pipeline_name {
                    derived_attributes.push(Attribute::string("pipeline", name));
                }
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
                    attributes: transition.attributes(),
                    derived_attributes,
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
