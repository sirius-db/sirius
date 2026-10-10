// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Data-batch FSM analysis types.

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
use rustc_hash::FxHashSet as HashSet;
use sirius_telemetry_store as schema;
use uuid::Uuid;

fn transition_attributes(event: &schema::DataBatchEvent) -> Vec<DynamicAttribute> {
    match event {
        schema::DataBatchEvent::Constructed {
            data_batch_id,
            producer_pipeline_id,
            ..
        } => vec![
            DynamicAttribute::u64("data_batch_id", *data_batch_id),
            DynamicAttribute::string(
                "producer_pipeline_id",
                producer_pipeline_id.target.to_string(),
            ),
        ],
        _ => Vec::new(),
    }
}

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
            state("exit", &[]),
        ],
        transitions: vec![
            FsmTransitionDecl::Entry("constructed".to_owned()),
            FsmTransitionDecl::Transition("constructed".to_owned(), "stationary".to_owned()),
            FsmTransitionDecl::Transition("stationary".to_owned(), "in_transit".to_owned()),
            FsmTransitionDecl::Transition("in_transit".to_owned(), "stationary".to_owned()),
            FsmTransitionDecl::Transition("stationary".to_owned(), "stationary".to_owned()),
            FsmTransitionDecl::Transition("stationary".to_owned(), "destructed".to_owned()),
            FsmTransitionDecl::Transition("destructed".to_owned(), "exit".to_owned()),
            FsmTransitionDecl::Exit("exit".to_owned()),
        ],
    }
}

/// The reconstructed data-batch FSM.
#[derive(Debug)]
pub struct DataBatch {
    fsm: AnalyzedFsm<schema::DataBatchEvent>,
    consumers: HashSet<Uuid>,
}

impl DataBatch {
    pub(crate) fn from_builder(builder: DataBatchBuilder) -> AnalyzerResult<Self> {
        Ok(Self {
            fsm: builder.try_build()?,
            consumers: HashSet::default(),
        })
    }

    pub(crate) fn add_consumer(&mut self, pipeline_id: Uuid) {
        self.consumers.insert(pipeline_id);
    }

    pub fn transitions(&self) -> &[AnalyzedTransition<schema::DataBatchEvent>] {
        self.fsm.transitions()
    }

    fn first_data(&self) -> Option<&schema::DataBatchEvent> {
        self.fsm.transition(0).map(|transition| &transition.data)
    }

    fn instance_name(&self) -> String {
        match self.first_data() {
            Some(schema::DataBatchEvent::Constructed { data_batch_id, .. }) => {
                format!("batch-{data_batch_id}")
            }
            _ => self.id().to_string(),
        }
    }
}

/// Builder for data-batch FSMs.
pub type DataBatchBuilder = AnalyzedFsmBuilder<schema::DataBatchEvent>;

impl Entity for DataBatch {
    fn id(&self) -> Uuid {
        self.fsm.id()
    }

    fn type_name(&self) -> &str {
        "data_batch"
    }

    fn earliest_timestamp(&self) -> TimeUnixNanoSec {
        self.fsm.earliest_timestamp()
    }

    fn latest_timestamp(&self) -> TimeUnixNanoSec {
        self.fsm.latest_timestamp()
    }
}

impl RefTreeEntity for DataBatch {
    fn parent_id(&self) -> Option<Uuid> {
        self.producer_pipeline_uuid()
    }
}

impl Fsm for DataBatch {
    type TransitionType = AnalyzedTransition<schema::DataBatchEvent>;

    fn len(&self) -> usize {
        self.fsm.len()
    }

    fn transition(&self, index: usize) -> Option<&Self::TransitionType> {
        self.fsm.transition(index)
    }
}

impl<'a> FsmUsages<'a> for DataBatch {
    fn usages_with_state_names(&'a self) -> impl Iterator<Item = (&'a str, impl Usage<'a>)> {
        self.fsm.usages_with_state_names()
    }
}

impl Using for DataBatch {
    fn usages(&self) -> impl Iterator<Item = impl Usage<'_>> {
        self.fsm.usages()
    }
}

impl FsmTypeDeclaration for DataBatch {
    fn fsm_type_declaration() -> FsmTypeDecl {
        declaration()
    }
}

/// Sirius-specific data-batch queries and UI conversion.
pub trait DataBatchExt {
    fn numeric_id(&self) -> Option<u64>;
    fn producer_pipeline_uuid(&self) -> Option<Uuid>;
    fn matches_filter(&self, filter: &OperatorFilter) -> bool;
    fn belongs_to(&self, pipelines: &HashSet<Uuid>) -> bool;
    fn operator_in(&self, pipelines: &HashSet<Uuid>) -> Option<Uuid>;
    fn active_span(&self) -> Option<SpanUnixNanoSec>;
    fn try_to_ui_fsm(&self, epoch: TimeUnixNanoSec) -> AnalyzerResult<FiniteStateMachine>;
}

impl DataBatchExt for DataBatch {
    fn numeric_id(&self) -> Option<u64> {
        match self.first_data()? {
            schema::DataBatchEvent::Constructed { data_batch_id, .. } => Some(*data_batch_id),
            _ => None,
        }
    }

    fn producer_pipeline_uuid(&self) -> Option<Uuid> {
        self.first_data().and_then(|transition| match transition {
            schema::DataBatchEvent::Constructed {
                producer_pipeline_id,
                ..
            } => Some(producer_pipeline_id.target),
            _ => None,
        })
    }

    fn matches_filter(&self, filter: &OperatorFilter) -> bool {
        filter.operator_ids.is_empty()
            || self
                .producer_pipeline_uuid()
                .is_some_and(|pipeline_uuid| filter.operator_ids.contains(&pipeline_uuid))
            || self
                .consumers
                .iter()
                .any(|id| filter.operator_ids.contains(id))
    }

    fn belongs_to(&self, pipelines: &HashSet<Uuid>) -> bool {
        self.operator_in(pipelines).is_some()
    }

    fn operator_in(&self, pipelines: &HashSet<Uuid>) -> Option<Uuid> {
        self.producer_pipeline_uuid()
            .filter(|id| pipelines.contains(id))
            .or_else(|| {
                self.consumers
                    .iter()
                    .filter(|id| pipelines.contains(id))
                    .copied()
                    .min()
            })
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
            instance_name: self.instance_name(),
            transitions,
        })
    }
}
