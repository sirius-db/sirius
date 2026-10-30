// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use quent_analyzer::fsm::Transition;
use quent_dynamic_attributes::DynamicAttribute;
use quent_time::to_secs_relative;
use quent_ui::{
    FiniteStateMachine, FsmTransition, FsmUsage,
    fsm::{FsmStateTypeDecl, FsmTransitionDecl, FsmTypeDecl, FsmTypeDeclaration},
};

pub(crate) type QueryActivityBuilder = AnalyzedFsmBuilder<schema::QueryActivityEvent>;
#[derive(Debug)]
pub(crate) struct QueryActivity(AnalyzedFsm<schema::QueryActivityEvent>);
impl QueryActivity {
    pub(crate) fn from_builder(builder: QueryActivityBuilder) -> AnalyzerResult<Self> {
        Ok(Self(builder.try_build()?))
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
            .map(|t| {
                let mut attributes = Vec::new();
                match &t.data {
                    schema::QueryActivityEvent::Started {
                        phase,
                        kind,
                        parent_activity_id,
                        planning_generation,
                        execution_query_id,
                        ..
                    } => {
                        attributes.push(DynamicAttribute::string("phase", phase));
                        attributes.push(DynamicAttribute::string("kind", kind));
                        if let Some(id) = parent_activity_id {
                            attributes.push(DynamicAttribute::string(
                                "parent_activity_id",
                                id.target.to_string(),
                            ));
                        }
                        if let Some(n) = planning_generation {
                            attributes.push(DynamicAttribute::u64("planning_generation", *n));
                        }
                        if let Some(n) = execution_query_id {
                            attributes.push(DynamicAttribute::u64("execution_query_id", *n));
                        }
                    }
                    schema::QueryActivityEvent::Finished {
                        outcome, decision, ..
                    } => {
                        if let Some(decision) = decision {
                            attributes.push(DynamicAttribute::string("decision", decision));
                        }
                        if let Some(outcome) = outcome {
                            attributes.push(DynamicAttribute::string("outcome", outcome));
                        }
                    }
                }
                FsmTransition {
                    name: t.name().into(),
                    timestamp: to_secs_relative(t.timestamp(), epoch),
                    usages: t
                        .usages()
                        .iter()
                        .map(|u| FsmUsage {
                            resource: u.resource_id,
                            capacities: u
                                .capacities
                                .iter()
                                .map(|c| (c.name.into(), c.value))
                                .collect(),
                        })
                        .collect(),
                    attributes,
                    derived_attributes: vec![],
                }
            })
            .collect();
        Ok(FiniteStateMachine {
            id: self.id(),
            type_name: self.type_name().into(),
            instance_name: self.instance_name().into(),
            transitions,
        })
    }
}
impl Entity for QueryActivity {
    fn id(&self) -> Uuid {
        self.0.id()
    }
    fn type_name(&self) -> &str {
        "query_activity"
    }
    fn earliest_timestamp(&self) -> TimeUnixNanoSec {
        self.0.earliest_timestamp()
    }
    fn latest_timestamp(&self) -> TimeUnixNanoSec {
        self.0.latest_timestamp()
    }
}
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
impl Fsm for QueryActivity {
    type TransitionType = AnalyzedTransition<schema::QueryActivityEvent>;
    fn len(&self) -> usize {
        self.0.len()
    }
    fn transition(&self, index: usize) -> Option<&Self::TransitionType> {
        self.0.transition(index)
    }
}
impl<'a> FsmUsages<'a> for QueryActivity {
    fn usages_with_state_names(&'a self) -> impl Iterator<Item = (&'a str, impl Usage<'a>)> {
        self.0.usages_with_state_names()
    }
}
impl Using for QueryActivity {
    fn usages(&self) -> impl Iterator<Item = impl Usage<'_>> {
        self.0.usages()
    }
}
impl FsmTypeDeclaration for QueryActivity {
    fn fsm_type_declaration() -> FsmTypeDecl {
        FsmTypeDecl {
            name: "query_activity".into(),
            states: vec![
                FsmStateTypeDecl {
                    name: "working".into(),
                    usages: vec!["track".into()],
                },
                FsmStateTypeDecl {
                    name: "waiting".into(),
                    usages: vec!["track".into()],
                },
                FsmStateTypeDecl {
                    name: "finished".into(),
                    usages: vec![],
                },
            ],
            transitions: vec![
                FsmTransitionDecl::Entry("working".into()),
                FsmTransitionDecl::Entry("waiting".into()),
                FsmTransitionDecl::Transition("working".into(), "finished".into()),
                FsmTransitionDecl::Transition("waiting".into(), "finished".into()),
                FsmTransitionDecl::Exit("finished".into()),
            ],
        }
    }
}
