// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Data-batch FSM analysis types.

use quent_analyzer::{
    AnalyzerResult, Entity,
    fsm::{
        Fsm, FsmStateTypeDecl, FsmTransitionDecl, FsmTypeDecl, FsmTypeDeclaration, FsmUsages,
        Transition,
        events::{AnalyzedTransition, FsmEvents, FsmEventsBuilder},
    },
    resource::{Usage, Using},
};
use quent_query_engine_ui::OperatorFilter;
use quent_time::{TimeUnixNanoSec, Timestamp, span::SpanUnixNanoSec, to_secs_relative};
use quent_ui::{FiniteStateMachine, FsmTransition, FsmUsage};
use sirius_telemetry_store as schema;
use uuid::Uuid;

fn declaration() -> FsmTypeDecl {
    let state = |name: &str, usages: &[&str]| FsmStateTypeDecl {
        name: name.to_owned(),
        usages: usages.iter().map(|usage| (*usage).to_owned()).collect(),
    };
    FsmTypeDecl {
        name: "data_batch".to_owned(),
        states: vec![
            state("constructed", &[]),
            state("stationary", &["memory"]),
            state("in_transit", &["source_memory", "dest_memory", "channel"]),
            state("destructed", &[]),
        ],
        transitions: vec![
            FsmTransitionDecl::Entry("constructed".to_owned()),
            FsmTransitionDecl::Transition("constructed".to_owned(), "stationary".to_owned()),
            FsmTransitionDecl::Transition("stationary".to_owned(), "in_transit".to_owned()),
            FsmTransitionDecl::Transition("in_transit".to_owned(), "stationary".to_owned()),
            FsmTransitionDecl::Transition("stationary".to_owned(), "stationary".to_owned()),
            FsmTransitionDecl::Transition("stationary".to_owned(), "destructed".to_owned()),
            FsmTransitionDecl::Exit("destructed".to_owned()),
        ],
    }
}

/// The reconstructed data-batch FSM.
#[derive(Debug)]
pub struct DataBatch(FsmEvents<schema::DataBatchEvent>);

impl DataBatch {
    pub(crate) fn from_builder(builder: DataBatchBuilder) -> AnalyzerResult<Self> {
        Ok(Self(builder.try_build()?))
    }

    pub fn transitions(&self) -> &[AnalyzedTransition<schema::DataBatchEvent>] {
        self.0.transitions()
    }

    fn first_data(&self) -> Option<&schema::DataBatchEvent> {
        self.0.first_data()
    }
}

/// Builder for data-batch FSMs.
pub type DataBatchBuilder = FsmEventsBuilder<schema::DataBatchEvent>;

impl Entity for DataBatch {
    fn id(&self) -> Uuid {
        self.0.id()
    }

    fn type_name(&self) -> &str {
        "data_batch"
    }

    fn instance_name(&self) -> &str {
        self.0.instance_name()
    }
}

impl Fsm for DataBatch {
    type TransitionType = AnalyzedTransition<schema::DataBatchEvent>;

    fn len(&self) -> usize {
        self.0.len()
    }

    fn transition(&self, index: usize) -> Option<&Self::TransitionType> {
        self.0.transition(index)
    }
}

impl<'a> FsmUsages<'a> for DataBatch {
    fn usages_with_state_names(&'a self) -> impl Iterator<Item = (&'a str, impl Usage<'a>)> {
        self.0.usages_with_state_names()
    }
}

impl Using for DataBatch {
    fn usages(&self) -> impl Iterator<Item = impl Usage<'_>> {
        self.0.usages()
    }
}

impl FsmTypeDeclaration for DataBatch {
    fn fsm_type_declaration() -> FsmTypeDecl {
        declaration()
    }
}

/// Sirius-specific data-batch queries and UI conversion.
pub trait DataBatchExt {
    fn producer_pipeline_uuid(&self) -> Option<Uuid>;
    fn matches_filter(&self, filter: &OperatorFilter) -> bool;
    fn active_span(&self) -> Option<SpanUnixNanoSec>;
    fn try_to_ui_fsm(&self, epoch: TimeUnixNanoSec) -> AnalyzerResult<FiniteStateMachine>;
}

impl DataBatchExt for DataBatch {
    fn producer_pipeline_uuid(&self) -> Option<Uuid> {
        self.first_data().and_then(|transition| match transition {
            schema::DataBatchEvent::Constructed {
                producer_pipeline_uuid,
                ..
            } => Some(producer_pipeline_uuid.target),
            _ => None,
        })
    }

    fn matches_filter(&self, filter: &OperatorFilter) -> bool {
        filter.operator_ids.is_empty()
            || self
                .producer_pipeline_uuid()
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
                    attributes: transition.attributes(),
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
