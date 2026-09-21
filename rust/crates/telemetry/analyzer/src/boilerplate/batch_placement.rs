// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Batch-placement FSM analysis types.

use quent_analyzer::{
    AnalyzerResult, Entity, RefTreeEntity,
    fsm::{
        Fsm, FsmUsages, Transition,
        native::{AnalyzedFsm, AnalyzedFsmBuilder, AnalyzedTransition},
    },
    resource::{Usage, Using},
};
use quent_dynamic_attributes::DynamicAttribute;
use quent_query_engine_ui::OperatorFilter;
use quent_time::{TimeUnixNanoSec, Timestamp, span::SpanUnixNanoSec, to_secs_relative};
use quent_ui::{
    FiniteStateMachine, FsmTransition, FsmUsage,
    fsm::{FsmStateTypeDecl, FsmTransitionDecl, FsmTypeDecl, FsmTypeDeclaration},
};
use sirius_telemetry_store as schema;
use uuid::Uuid;

fn transition_attributes(event: &schema::BatchPlacementEvent) -> Vec<DynamicAttribute> {
    match event {
        schema::BatchPlacementEvent::BatchRegistered {
            batch_id,
            pipeline_uuid,
            port_uuid,
            origin,
            ..
        } => {
            let mut attributes = vec![
                DynamicAttribute::u64("batch_id", *batch_id),
                DynamicAttribute::string("pipeline_uuid", pipeline_uuid.target.to_string()),
            ];
            if let Some(port_uuid) = port_uuid {
                attributes.push(DynamicAttribute::string(
                    "port_uuid",
                    port_uuid.target.to_string(),
                ));
            }
            attributes.push(DynamicAttribute::string("origin", origin.clone()));
            attributes
        }
        schema::BatchPlacementEvent::BatchPackaged { task_uuid, .. }
        | schema::BatchPlacementEvent::BatchProcessing { task_uuid, .. } => {
            vec![DynamicAttribute::string("task_uuid", task_uuid.to_string())]
        }
        schema::BatchPlacementEvent::BatchConsumed { reason, .. } => {
            vec![DynamicAttribute::string("reason", reason.clone())]
        }
        schema::BatchPlacementEvent::BatchQueued { .. } => Vec::new(),
    }
}

fn declaration() -> FsmTypeDecl {
    let state = |name: &str, usages: &[&str]| FsmStateTypeDecl {
        name: name.to_owned(),
        usages: usages.iter().map(|usage| (*usage).to_owned()).collect(),
    };
    FsmTypeDecl {
        name: "batch_placement".to_owned(),
        states: vec![
            state("batch_registered", &["tier"]),
            state("batch_queued", &["tier"]),
            state("batch_packaged", &["tier"]),
            state("batch_processing", &["tier"]),
            state("batch_consumed", &[]),
        ],
        transitions: vec![
            FsmTransitionDecl::Entry("batch_registered".to_owned()),
            FsmTransitionDecl::Transition("batch_registered".to_owned(), "batch_queued".to_owned()),
            FsmTransitionDecl::Transition(
                "batch_registered".to_owned(),
                "batch_packaged".to_owned(),
            ),
            FsmTransitionDecl::Transition("batch_queued".to_owned(), "batch_queued".to_owned()),
            FsmTransitionDecl::Transition("batch_queued".to_owned(), "batch_packaged".to_owned()),
            FsmTransitionDecl::Transition("batch_packaged".to_owned(), "batch_packaged".to_owned()),
            FsmTransitionDecl::Transition(
                "batch_packaged".to_owned(),
                "batch_processing".to_owned(),
            ),
            FsmTransitionDecl::Transition(
                "batch_processing".to_owned(),
                "batch_packaged".to_owned(),
            ),
            FsmTransitionDecl::Transition(
                "batch_processing".to_owned(),
                "batch_processing".to_owned(),
            ),
            FsmTransitionDecl::Transition(
                "batch_processing".to_owned(),
                "batch_consumed".to_owned(),
            ),
            FsmTransitionDecl::Transition("batch_packaged".to_owned(), "batch_consumed".to_owned()),
            FsmTransitionDecl::Transition("batch_queued".to_owned(), "batch_consumed".to_owned()),
            FsmTransitionDecl::Exit("batch_consumed".to_owned()),
        ],
    }
}

/// One reconstructed placement of a physical batch on a consumer input port.
#[derive(Debug)]
pub struct BatchPlacement(AnalyzedFsm<schema::BatchPlacementEvent>);

impl BatchPlacement {
    pub(crate) fn from_builder(builder: BatchPlacementBuilder) -> AnalyzerResult<Self> {
        Ok(Self(builder.try_build()?))
    }

    pub fn transitions(&self) -> &[AnalyzedTransition<schema::BatchPlacementEvent>] {
        self.0.transitions()
    }

    fn first_data(&self) -> Option<&schema::BatchPlacementEvent> {
        self.0.transition(0).map(|transition| &transition.data)
    }

    fn instance_name(&self) -> &str {
        self.first_data()
            .and_then(|event| match event {
                schema::BatchPlacementEvent::BatchRegistered { instance_name, .. } => {
                    Some(instance_name.as_str())
                }
                _ => None,
            })
            .unwrap_or_default()
    }
}

/// Builder for batch-placement FSMs.
pub type BatchPlacementBuilder = AnalyzedFsmBuilder<schema::BatchPlacementEvent>;

impl Entity for BatchPlacement {
    fn id(&self) -> Uuid {
        self.0.id()
    }

    fn type_name(&self) -> &str {
        "batch_placement"
    }

    fn earliest_timestamp(&self) -> TimeUnixNanoSec {
        self.0.earliest_timestamp()
    }

    fn latest_timestamp(&self) -> TimeUnixNanoSec {
        self.0.latest_timestamp()
    }
}

impl RefTreeEntity for BatchPlacement {
    fn parent_id(&self) -> Option<Uuid> {
        self.pipeline_uuid()
    }
}

impl Fsm for BatchPlacement {
    type TransitionType = AnalyzedTransition<schema::BatchPlacementEvent>;

    fn len(&self) -> usize {
        self.0.len()
    }

    fn transition(&self, index: usize) -> Option<&Self::TransitionType> {
        self.0.transition(index)
    }
}

impl<'a> FsmUsages<'a> for BatchPlacement {
    fn usages_with_state_names(&'a self) -> impl Iterator<Item = (&'a str, impl Usage<'a>)> {
        self.0.usages_with_state_names()
    }
}

impl Using for BatchPlacement {
    fn usages(&self) -> impl Iterator<Item = impl Usage<'_>> {
        self.0.usages()
    }
}

impl FsmTypeDeclaration for BatchPlacement {
    fn fsm_type_declaration() -> FsmTypeDecl {
        declaration()
    }
}

/// Sirius-specific batch-placement queries and UI conversion.
pub trait BatchPlacementExt {
    fn batch_id(&self) -> Option<u64>;
    fn pipeline_uuid(&self) -> Option<Uuid>;
    fn last_task_uuid(&self) -> Option<Uuid>;
    fn matches_filter(&self, filter: &OperatorFilter) -> bool;
    fn active_span(&self) -> Option<SpanUnixNanoSec>;
    fn try_to_ui_fsm(&self, epoch: TimeUnixNanoSec) -> AnalyzerResult<FiniteStateMachine>;
}

impl BatchPlacementExt for BatchPlacement {
    fn batch_id(&self) -> Option<u64> {
        self.first_data().and_then(|transition| match transition {
            schema::BatchPlacementEvent::BatchRegistered { batch_id, .. } => Some(*batch_id),
            _ => None,
        })
    }

    fn pipeline_uuid(&self) -> Option<Uuid> {
        self.first_data().and_then(|transition| match transition {
            schema::BatchPlacementEvent::BatchRegistered { pipeline_uuid, .. } => {
                Some(pipeline_uuid.target)
            }
            _ => None,
        })
    }

    fn last_task_uuid(&self) -> Option<Uuid> {
        self.transitions()
            .iter()
            .rev()
            .find_map(|transition| match &transition.data {
                schema::BatchPlacementEvent::BatchPackaged { task_uuid, .. }
                | schema::BatchPlacementEvent::BatchProcessing { task_uuid, .. } => {
                    Some(*task_uuid)
                }
                _ => None,
            })
    }

    fn matches_filter(&self, filter: &OperatorFilter) -> bool {
        filter.operator_ids.is_empty()
            || self
                .pipeline_uuid()
                .is_some_and(|pipeline_uuid| filter.operator_ids.contains(&pipeline_uuid))
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
                    name: transition.name().to_owned(),
                    usages: transition
                        .usages()
                        .iter()
                        .map(|usage| FsmUsage {
                            resource: usage.resource_id,
                            capacities: usage
                                .capacities
                                .iter()
                                .map(|capacity| (capacity.name.to_owned(), capacity.value))
                                .collect(),
                        })
                        .collect(),
                    timestamp: to_secs_relative(transition.timestamp(), epoch),
                    attributes: transition_attributes(&transition.data),
                    derived_attributes: Vec::new(),
                })
            })
            .collect::<AnalyzerResult<Vec<_>>>()?;

        Ok(FiniteStateMachine {
            id: self.id(),
            type_name: self.type_name().to_owned(),
            instance_name: self.instance_name().to_owned(),
            transitions,
        })
    }
}
