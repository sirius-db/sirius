// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use quent_dynamic_attributes::DynamicAttribute;

pub(crate) const QUERY_GROUP_ATTRIBUTE: &str = "query_group";

pub(crate) type QueryBuilder = AnalyzedFsmBuilder<schema::QueryEvent>;

#[derive(Debug)]
pub struct Query {
    fsm: AnalyzedFsm<schema::QueryEvent>,
    group_label: Option<String>,
}

impl Query {
    pub(crate) fn try_from_builder(builder: QueryBuilder) -> AnalyzerResult<Self> {
        Ok(Self {
            fsm: builder.try_build()?,
            group_label: None,
        })
    }

    pub(crate) fn set_group_label(&mut self, label: Option<String>) {
        self.group_label = label;
    }

    pub(crate) fn transitions(&self) -> &[AnalyzedTransition<schema::QueryEvent>] {
        self.fsm.transitions()
    }

    pub(crate) fn query_group_id(&self) -> Option<Uuid> {
        match self.fsm.transition(0).map(|transition| &transition.data)? {
            schema::QueryEvent::Init { query_group_id, .. } => Some(query_group_id.target),
            _ => None,
        }
    }
}

impl Entity for Query {
    fn id(&self) -> Uuid {
        self.fsm.id()
    }

    fn type_name(&self) -> &str {
        self.fsm.type_name()
    }

    fn earliest_timestamp(&self) -> TimeUnixNanoSec {
        self.fsm.earliest_timestamp()
    }

    fn latest_timestamp(&self) -> TimeUnixNanoSec {
        self.fsm.latest_timestamp()
    }
}

impl Fsm for Query {
    type TransitionType = AnalyzedTransition<schema::QueryEvent>;

    fn len(&self) -> usize {
        self.fsm.len()
    }

    fn transition(&self, index: usize) -> Option<&Self::TransitionType> {
        self.fsm.transition(index)
    }
}

impl<'a> FsmUsages<'a> for Query {
    fn usages_with_state_names(&'a self) -> impl Iterator<Item = (&'a str, impl Usage<'a>)> {
        self.fsm.usages_with_state_names()
    }
}

impl Using for Query {
    fn usages(&self) -> impl Iterator<Item = impl Usage<'_>> {
        self.fsm.usages()
    }
}

impl RefTreeEntity for Query {
    fn parent_id(&self) -> Option<Uuid> {
        self.query_group_id()
    }
}

impl QueryEntity for Query {
    fn to_ui(&self) -> AnalyzerResult<query_engine_ui::Query> {
        let transitions = self.transitions();
        let epoch = transitions.first().map(Timestamp::timestamp);
        let mut planning_s = None;
        let mut executing_s = None;
        let mut completed_s = None;

        if let Some(epoch) = epoch {
            for (index, transition) in transitions.iter().enumerate() {
                match transition.data {
                    schema::QueryEvent::Planning { .. } => {
                        planning_s = Some(try_to_secs_relative(transition.timestamp(), epoch)?);
                    }
                    schema::QueryEvent::Executing { .. } => {
                        executing_s = Some(try_to_secs_relative(transition.timestamp(), epoch)?);
                        if let Some(next) = transitions.get(index + 1) {
                            completed_s = Some(try_to_secs_relative(next.timestamp(), epoch)?);
                        }
                    }
                    _ => {}
                }
            }
        }

        Ok(query_engine_ui::Query {
            id: self.id(),
            // Preserve session grouping through Quent's generic query attributes.
            custom_attributes: self.group_label.as_ref().map_or_else(Vec::new, |label| {
                vec![DynamicAttribute::string(
                    QUERY_GROUP_ATTRIBUTE,
                    label.clone(),
                )]
            }),
            instance_name: transitions
                .first()
                .and_then(|transition| match &transition.data {
                    schema::QueryEvent::Init { instance_name, .. } => Some(instance_name.clone()),
                    _ => None,
                }),
            start_unix_ns: epoch,
            planning_s,
            executing_s,
            completed_s,
        })
    }
}
