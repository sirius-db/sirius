// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Task FSM analysis types.

use quent_analyzer::{
    AnalyzerResult, Entity,
    fsm::{
        Fsm, FsmStateTypeDecl, FsmTransitionDecl, FsmTypeDecl, FsmTypeDeclaration, FsmUsages,
        Transition,
        events::{AnalyzedTransition, FsmEvents, FsmEventsBuilder},
    },
    resource::{Usage, Using},
};
use quent_dynamic_attributes::DynamicAttribute;
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
        name: "task".to_owned(),
        states: vec![
            state("created", &[]),
            state("queued", &["queue"]),
            state("routing", &["manager_thread"]),
            state("reserving", &["manager_thread"]),
            state("downgrading", &["manager_thread"]),
            state("preparing", &["executor_thread", "reservation"]),
            state("computing", &["executor_thread", "reservation"]),
            state("finalizing", &[]),
        ],
        transitions: vec![
            FsmTransitionDecl::Entry("created".to_owned()),
            FsmTransitionDecl::Transition("created".to_owned(), "queued".to_owned()),
            FsmTransitionDecl::Transition("created".to_owned(), "finalizing".to_owned()),
            FsmTransitionDecl::Transition("queued".to_owned(), "routing".to_owned()),
            FsmTransitionDecl::Transition("queued".to_owned(), "reserving".to_owned()),
            FsmTransitionDecl::Transition("queued".to_owned(), "finalizing".to_owned()),
            FsmTransitionDecl::Transition("routing".to_owned(), "queued".to_owned()),
            FsmTransitionDecl::Transition("routing".to_owned(), "reserving".to_owned()),
            FsmTransitionDecl::Transition("routing".to_owned(), "finalizing".to_owned()),
            FsmTransitionDecl::Transition("reserving".to_owned(), "downgrading".to_owned()),
            FsmTransitionDecl::Transition("reserving".to_owned(), "preparing".to_owned()),
            FsmTransitionDecl::Transition("reserving".to_owned(), "finalizing".to_owned()),
            FsmTransitionDecl::Transition("downgrading".to_owned(), "preparing".to_owned()),
            FsmTransitionDecl::Transition("downgrading".to_owned(), "finalizing".to_owned()),
            FsmTransitionDecl::Transition("preparing".to_owned(), "computing".to_owned()),
            FsmTransitionDecl::Transition("preparing".to_owned(), "finalizing".to_owned()),
            FsmTransitionDecl::Transition("computing".to_owned(), "computing".to_owned()),
            FsmTransitionDecl::Transition("computing".to_owned(), "finalizing".to_owned()),
            FsmTransitionDecl::Exit("finalizing".to_owned()),
        ],
    }
}

/// The reconstructed task FSM.
#[derive(Debug)]
pub struct Task(FsmEvents<schema::TaskEvent>);

impl Task {
    pub(crate) fn from_builder(builder: TaskBuilder) -> AnalyzerResult<Self> {
        Ok(Self(builder.try_build()?))
    }

    pub fn transitions(&self) -> &[AnalyzedTransition<schema::TaskEvent>] {
        self.0.transitions()
    }

    fn first_data(&self) -> Option<&schema::TaskEvent> {
        self.0.first_data()
    }
}

/// Builder for task FSMs.
pub type TaskBuilder = FsmEventsBuilder<schema::TaskEvent>;

impl Entity for Task {
    fn id(&self) -> Uuid {
        self.0.id()
    }

    fn type_name(&self) -> &str {
        "task"
    }

    fn instance_name(&self) -> &str {
        self.0.instance_name()
    }
}

impl Fsm for Task {
    type TransitionType = AnalyzedTransition<schema::TaskEvent>;

    fn len(&self) -> usize {
        self.0.len()
    }

    fn transition(&self, index: usize) -> Option<&Self::TransitionType> {
        self.0.transition(index)
    }
}

impl<'a> FsmUsages<'a> for Task {
    fn usages_with_state_names(&'a self) -> impl Iterator<Item = (&'a str, impl Usage<'a>)> {
        self.0.usages_with_state_names()
    }
}

impl Using for Task {
    fn usages(&self) -> impl Iterator<Item = impl Usage<'_>> {
        self.0.usages()
    }
}

impl FsmTypeDeclaration for Task {
    fn fsm_type_declaration() -> FsmTypeDecl {
        declaration()
    }
}

/// Sirius-specific task queries and UI conversion.
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
        self.first_data().and_then(|transition| match transition {
            schema::TaskEvent::Created { pipeline_uuid, .. } => Some(pipeline_uuid.target),
            _ => None,
        })
    }

    fn executes_physical_operation(&self, physical_operator_id: u32) -> bool {
        self.transitions().iter().any(|transition| {
            matches!(
                &transition.data,
                schema::TaskEvent::Computing {
                    current_operator_id,
                    ..
                } if *current_operator_id == physical_operator_id
            )
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

    fn try_to_ui_fsm(
        &self,
        epoch: TimeUnixNanoSec,
        pipeline_name: Option<&str>,
    ) -> AnalyzerResult<FiniteStateMachine> {
        let raw = self.transitions();
        let transitions = raw
            .iter()
            .enumerate()
            .map(|(index, transition)| {
                let mut derived_attributes = Vec::new();
                let input_bytes = match &transition.data {
                    schema::TaskEvent::Preparing { input_bytes, .. }
                    | schema::TaskEvent::Computing { input_bytes, .. } => Some(*input_bytes),
                    _ => None,
                };
                if let Some(input_bytes) = input_bytes
                    && let Some(next) = raw.get(index + 1)
                {
                    let span_secs = (next.timestamp() - transition.timestamp()) as f64 / 1e9;
                    if span_secs > 0.0 {
                        derived_attributes.push(DynamicAttribute::f64(
                            "bytes_per_sec",
                            input_bytes as f64 / span_secs,
                        ));
                    }
                }
                if let Some(name) = pipeline_name {
                    derived_attributes.push(DynamicAttribute::string("pipeline", name));
                }
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
                    derived_attributes,
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
