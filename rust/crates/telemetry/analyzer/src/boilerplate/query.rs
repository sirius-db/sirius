// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

pub(crate) type QueryBuilder = AnalyzedFsmBuilder<schema::QueryEvent>;

#[derive(Debug)]
pub struct Query(AnalyzedFsm<schema::QueryEvent>);

impl Query {
    pub(crate) fn try_from_builder(builder: QueryBuilder) -> AnalyzerResult<Self> {
        Ok(Self(builder.try_build()?))
    }

    pub(crate) fn transitions(&self) -> &[AnalyzedTransition<schema::QueryEvent>] {
        self.0.transitions()
    }

    pub(crate) fn query_group_id(&self) -> Option<Uuid> {
        match self.0.transition(0).map(|transition| &transition.data)? {
            schema::QueryEvent::Init { query_group_id, .. } => Some(query_group_id.target),
            _ => None,
        }
    }
}

impl Entity for Query {
    fn id(&self) -> Uuid {
        self.0.id()
    }

    fn type_name(&self) -> &str {
        self.0.type_name()
    }

    fn earliest_timestamp(&self) -> TimeUnixNanoSec {
        self.0.earliest_timestamp()
    }

    fn latest_timestamp(&self) -> TimeUnixNanoSec {
        self.0.latest_timestamp()
    }
}

impl Fsm for Query {
    type TransitionType = AnalyzedTransition<schema::QueryEvent>;

    fn len(&self) -> usize {
        self.0.len()
    }

    fn transition(&self, index: usize) -> Option<&Self::TransitionType> {
        self.0.transition(index)
    }
}

impl<'a> FsmUsages<'a> for Query {
    fn usages_with_state_names(&'a self) -> impl Iterator<Item = (&'a str, impl Usage<'a>)> {
        self.0.usages_with_state_names()
    }
}

impl Using for Query {
    fn usages(&self) -> impl Iterator<Item = impl Usage<'_>> {
        self.0.usages()
    }
}

impl RefTreeEntity for Query {
    fn parent_id(&self) -> Option<Uuid> {
        self.query_group_id()
    }
}

impl QueryEntity for Query {
    fn query_group_id(&self) -> Option<Uuid> {
        self.query_group_id()
    }

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
            query_group_id: self.query_group_id().unwrap_or_default(),
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
