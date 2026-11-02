// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Data-batch FSM analysis types.

use quent_analyzer::{
    AnalyzerResult, Entity, RefTreeEntity,
    fsm::{
        Fsm, FsmUsages,
        native::{AnalyzedFsm, AnalyzedFsmBuilder, AnalyzedTransition},
    },
    resource::{Usage, Using},
};
use quent_query_engine_ui::OperatorFilter;
use quent_time::{TimeUnixNanoSec, Timestamp, span::SpanUnixNanoSec};
use quent_ui::{
    FiniteStateMachine,
    fsm::{FsmTypeDecl, FsmTypeDeclaration},
};
use rustc_hash::FxHashSet as HashSet;
use sirius_telemetry_store as schema;
use uuid::Uuid;

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

native_fsm!(DataBatch, schema::DataBatchEvent, fsm);

impl RefTreeEntity for DataBatch {
    fn parent_id(&self) -> Option<Uuid> {
        self.producer_pipeline_uuid()
    }
}

impl FsmTypeDeclaration for DataBatch {
    fn fsm_type_declaration() -> FsmTypeDecl {
        super::data_batch_declaration()
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
                Ok(super::ui_transition(
                    transition,
                    epoch,
                    transition.data.attributes(),
                    Vec::new(),
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
impl ScopedFsm for DataBatch {
    fn visible_in(&self, scope: &QueryScope<'_>) -> bool {
        DataBatchExt::belongs_to(self, scope.operators)
    }
    fn in_reference_tree(&self, scope: &QueryScope<'_>) -> bool {
        self.producer_pipeline_uuid()
            .is_some_and(|id| scope.operators.contains(&id))
    }
    fn operator_in_scope(&self, scope: &QueryScope<'_>) -> Option<Uuid> {
        DataBatchExt::operator_in(self, scope.operators)
    }
    fn matches_operator_filter(&self, filter: &quent_query_engine_ui::OperatorFilter) -> bool {
        self.matches_filter(filter)
    }
    fn to_ui(
        &self,
        _model: &SiriusModel,
        epoch: TimeUnixNanoSec,
    ) -> AnalyzerResult<FiniteStateMachine> {
        self.try_to_ui_fsm(epoch)
    }
}
