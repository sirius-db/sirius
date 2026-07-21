// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! BatchPlacement FSM analysis types.
//!
//! Mirrors `task.rs`: with `FsmEvents<T>` providing all generic trait impls
//! (`Entity`, `Fsm`, `FsmUsages`, `Using`, `FsmTypeDeclaration`), the batch
//! analyzer is just type aliases plus application-specific helper methods.

use instrumentation_model::batch::BatchPlacementTransition as ModelBatchPlacementTransition;
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

/// The reconstructed BatchPlacement FSM: one placement of a physical data batch on a
/// consuming pipeline's input port.
pub type BatchPlacement = FsmEvents<ModelBatchPlacementTransition>;

/// Builder for BatchPlacement FSMs.
pub type BatchPlacementBuilder = FsmEventsBuilder<ModelBatchPlacementTransition>;

/// Application-specific methods on the BatchPlacement FSM.
pub trait BatchPlacementExt {
    /// The engine's process-unique batch id, shared by all placements and
    /// re-packagings of one physical batch.
    fn batch_id(&self) -> Option<u64>;
    /// The consuming pipeline (== quent Operator id), from the registration.
    fn pipeline_uuid(&self) -> Option<Uuid>;
    /// The most recent task this batch was packaged into or processed by.
    fn task_uuid(&self) -> Option<Uuid>;
    fn matches_filter(&self, filter: &OperatorFilter) -> bool;
    fn active_span(&self) -> Option<SpanUnixNanoSec>;
    fn try_to_ui_fsm(&self, epoch: TimeUnixNanoSec) -> AnalyzerResult<FiniteStateMachine>;
}

impl BatchPlacementExt for BatchPlacement {
    fn batch_id(&self) -> Option<u64> {
        self.first_data().and_then(|t| match t {
            ModelBatchPlacementTransition::BatchRegistered(data) => Some(data.batch_id),
            _ => None,
        })
    }

    fn pipeline_uuid(&self) -> Option<Uuid> {
        self.first_data().and_then(|t| match t {
            ModelBatchPlacementTransition::BatchRegistered(data) => Some(data.pipeline_uuid),
            _ => None,
        })
    }

    fn task_uuid(&self) -> Option<Uuid> {
        self.transitions()
            .iter()
            .rev()
            .find_map(|transition| match &transition.data {
                ModelBatchPlacementTransition::BatchPackaged(data) => Some(data.task_uuid),
                ModelBatchPlacementTransition::BatchProcessing(data) => Some(data.task_uuid),
                _ => None,
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
                    // The transition's typed data (batch_id, pipeline_uuid,
                    // origin, task_uuid, reason, ...) rendered by the model.
                    attributes: transition.attributes(),
                    derived_attributes: vec![],
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
