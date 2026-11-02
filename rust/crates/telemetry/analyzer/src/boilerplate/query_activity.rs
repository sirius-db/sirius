// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use quent_ui::{
    FiniteStateMachine,
    fsm::{FsmTypeDecl, FsmTypeDeclaration},
};

pub(crate) type QueryActivityBuilder = AnalyzedFsmBuilder<schema::QueryActivityEvent>;
#[derive(Debug)]
pub(crate) struct QueryActivity(AnalyzedFsm<schema::QueryActivityEvent>);
impl QueryActivity {
    pub(crate) fn from_builder(builder: QueryActivityBuilder) -> AnalyzerResult<Self> {
        Ok(Self(builder.try_build()?))
    }
    pub(crate) fn track_id(&self) -> Uuid {
        match self.start() {
            schema::QueryActivityEvent::Started { track, .. } => track.target,
            _ => unreachable!(),
        }
    }
    fn start(&self) -> &schema::QueryActivityEvent {
        &self.0.transitions()[0].data
    }
    pub(crate) fn query_id(&self) -> Uuid {
        match self.start() {
            schema::QueryActivityEvent::Started { query_id, .. } => query_id.target,
            _ => unreachable!(),
        }
    }
    pub(crate) fn instance_name(&self) -> &str {
        match self.start() {
            schema::QueryActivityEvent::Started { phase, .. } => phase,
            _ => unreachable!(),
        }
    }
    pub(crate) fn matches_filter(&self, filter: &quent_query_engine_ui::OperatorFilter) -> bool {
        filter.operator_ids.is_empty()
    }
    pub(crate) fn try_to_ui_fsm(
        &self,
        epoch: TimeUnixNanoSec,
    ) -> AnalyzerResult<FiniteStateMachine> {
        let transitions = self
            .0
            .transitions()
            .iter()
            .map(|t| super::ui_transition(t, epoch, t.data.attributes(), Vec::new()))
            .collect();
        Ok(FiniteStateMachine {
            id: self.id(),
            type_name: self.type_name().into(),
            instance_name: self.instance_name().into(),
            transitions,
        })
    }
}
native_fsm!(QueryActivity, schema::QueryActivityEvent, 0);

impl RefTreeEntity for QueryActivity {
    fn parent_id(&self) -> Option<Uuid> {
        match self.start() {
            schema::QueryActivityEvent::Started {
                query_id,
                parent_activity_id,
                ..
            } => Some(
                parent_activity_id
                    .as_ref()
                    .map_or(query_id.target, |id| id.target),
            ),
            _ => None,
        }
    }
}
impl FsmTypeDeclaration for QueryActivity {
    fn fsm_type_declaration() -> FsmTypeDecl {
        super::query_activity_declaration()
    }
}

use crate::{
    fsms::{QueryScope, ScopedFsm},
    model::SiriusModel,
};
impl ScopedFsm for QueryActivity {
    fn visible_in(&self, scope: &QueryScope<'_>) -> bool {
        self.query_id() == scope.query_id
    }
    fn in_reference_tree(&self, _scope: &QueryScope<'_>) -> bool {
        false
    }
    fn operator_in_scope(&self, _scope: &QueryScope<'_>) -> Option<Uuid> {
        None
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
