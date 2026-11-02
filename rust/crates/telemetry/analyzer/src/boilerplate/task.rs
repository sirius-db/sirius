// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Task FSM analysis types.

use quent_analyzer::{
    AnalyzerResult, Entity, RefTreeEntity,
    fsm::{
        Fsm, FsmUsages,
        native::{AnalyzedFsm, AnalyzedFsmBuilder, AnalyzedTransition},
    },
    resource::{Usage, Using},
};
use quent_dynamic_attributes::DynamicAttribute;
use quent_query_engine_ui::OperatorFilter;
use quent_time::{TimeUnixNanoSec, Timestamp, span::SpanUnixNanoSec};
use quent_ui::{
    FiniteStateMachine,
    fsm::{FsmTypeDecl, FsmTypeDeclaration},
};
use sirius_telemetry_store as schema;
use uuid::Uuid;

/// The reconstructed task FSM.
#[derive(Debug)]
pub struct Task(AnalyzedFsm<schema::TaskEvent>);

impl Task {
    pub(crate) fn from_builder(builder: TaskBuilder) -> AnalyzerResult<Self> {
        Ok(Self(builder.try_build()?))
    }

    pub fn transitions(&self) -> &[AnalyzedTransition<schema::TaskEvent>] {
        self.0.transitions()
    }

    fn first_data(&self) -> Option<&schema::TaskEvent> {
        self.0.transition(0).map(|transition| &transition.data)
    }

    fn instance_name(&self) -> String {
        format!("task-{}", self.id())
    }
}

/// Builder for task FSMs.
pub type TaskBuilder = AnalyzedFsmBuilder<schema::TaskEvent>;

super::native_fsm!(Task, schema::TaskEvent, 0);

impl RefTreeEntity for Task {
    fn parent_id(&self) -> Option<Uuid> {
        self.pipeline_uuid()
    }
}

impl FsmTypeDeclaration for Task {
    fn fsm_type_declaration() -> FsmTypeDecl {
        super::task_declaration()
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
                Ok(super::ui_transition(
                    transition,
                    epoch,
                    transition.data.attributes(),
                    derived_attributes,
                ))
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

use crate::{
    fsms::{QueryScope, ScopedFsm},
    model::SiriusModel,
};
use quent_query_engine_analyzer::QueryEngineModel;
impl ScopedFsm for Task {
    fn visible_in(&self, scope: &QueryScope<'_>) -> bool {
        self.pipeline_uuid()
            .is_some_and(|id| scope.operators.contains(&id))
    }
    fn in_reference_tree(&self, scope: &QueryScope<'_>) -> bool {
        self.visible_in(scope)
    }
    fn operator_in_scope(&self, _scope: &QueryScope<'_>) -> Option<Uuid> {
        self.pipeline_uuid()
    }
    fn matches_operator_filter(&self, filter: &quent_query_engine_ui::OperatorFilter) -> bool {
        self.matches_filter(filter)
    }
    fn to_ui(
        &self,
        model: &SiriusModel,
        epoch: TimeUnixNanoSec,
    ) -> AnalyzerResult<FiniteStateMachine> {
        let name = self
            .pipeline_uuid()
            .and_then(|id| model.operator(id).ok())
            .and_then(|op| op.data().instance_name.as_deref());
        self.try_to_ui_fsm(epoch, name)
    }
}
