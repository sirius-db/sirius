// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[derive(Default)]
pub(crate) struct QueryGroupAccumulator {
    pub(crate) instance_name: Option<String>,
    pub(crate) engine_id: Option<Uuid>,
}

impl EntityEventAccumulator for QueryGroupAccumulator {
    type Event = schema::QueryGroupEvent;

    fn push(&mut self, event: Self::Event) {
        let schema::QueryGroupEvent::Declaration {
            instance_name,
            engine_id,
        } = event;
        self.instance_name = Some(instance_name);
        self.engine_id = Some(engine_id.target);
    }
}

#[derive(Debug)]
pub struct QueryGroup(AnalyzedEntity<QueryGroupAccumulator>);

impl QueryGroup {
    pub(crate) fn try_from_event(event: Event<schema::QueryGroupEvent>) -> AnalyzerResult<Self> {
        Ok(Self(AnalyzedEntity::try_from_event(event)?))
    }

    pub(crate) fn push(&mut self, event: Event<schema::QueryGroupEvent>) -> AnalyzerResult<()> {
        self.0.push(event)
    }

    pub(crate) fn data(&self) -> &QueryGroupAccumulator {
        self.0.accumulator()
    }
}

impl Entity for QueryGroup {
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

impl RefTreeEntity for QueryGroup {
    fn parent_id(&self) -> Option<Uuid> {
        self.0.accumulator().engine_id
    }
}

impl QueryGroupEntity for QueryGroup {
    fn to_ui(&self) -> query_engine_ui::QueryGroup {
        let data = self.data();
        query_engine_ui::QueryGroup {
            id: self.id(),
            instance_name: data.instance_name.clone(),
            engine_id: data.engine_id,
        }
    }
}
