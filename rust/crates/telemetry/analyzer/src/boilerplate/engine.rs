// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[derive(Default)]
pub(crate) struct EngineAccumulator {
    pub(crate) instance_name: Option<String>,
}

impl EntityEventAccumulator for EngineAccumulator {
    type Payload = schema::EngineEvent;

    fn push(&mut self, event: Self::Payload) {
        match event {
            schema::EngineEvent::Init { label } => self.instance_name = label,
        }
    }
}

#[derive(Debug)]
pub struct Engine(AnalyzedEntity<EngineAccumulator>);

impl Engine {
    pub(crate) fn try_from_event(event: Event<schema::EngineEvent>) -> AnalyzerResult<Self> {
        Ok(Self(AnalyzedEntity::try_from_event(event)?))
    }

    pub(crate) fn push(&mut self, event: Event<schema::EngineEvent>) -> AnalyzerResult<()> {
        self.0.push(event)
    }

    pub(crate) fn data(&self) -> &EngineAccumulator {
        self.0.accumulator()
    }
}

impl Entity for Engine {
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

impl RefTreeEntity for Engine {
    fn parent_id(&self) -> Option<Uuid> {
        None
    }
}

impl EngineEntity for Engine {
    fn to_ui(&self) -> AnalyzerResult<query_engine_ui::Engine> {
        let data = self.data();
        let start = self.earliest_timestamp();

        Ok(query_engine_ui::Engine {
            id: self.id(),
            start_time_unix_ns: Some(start),
            duration_s: None,
            instance_name: data.instance_name.clone(),
            implementation: None,
        })
    }
}
