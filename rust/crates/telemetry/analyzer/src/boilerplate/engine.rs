// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[derive(Default)]
pub(crate) struct EngineAccumulator {
    pub(crate) instance_name: Option<String>,
    pub(crate) implementation: Option<schema::EngineImplementationAttributes>,
    pub(crate) exited: bool,
}

impl quent_events::Entity for EngineAccumulator {
    type Event = schema::EngineEvent;
}

impl EntityEventAccumulator for EngineAccumulator {
    fn push(&mut self, event: Self::Event) {
        match event {
            schema::EngineEvent::Init {
                implementation,
                instance_name,
            } => {
                self.instance_name = instance_name;
                self.implementation = Some(implementation);
            }
            schema::EngineEvent::Exit => self.exited = true,
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
        let duration_s = data
            .exited
            .then(|| try_to_secs_relative(self.latest_timestamp(), start))
            .transpose()?;
        Ok(query_engine_ui::Engine {
            id: self.id(),
            start_time_unix_ns: Some(start),
            duration_s,
            instance_name: data.instance_name.clone(),
            implementation: data.implementation.as_ref().map(|implementation| {
                query_engine_ui::EngineImplementationAttributes {
                    name: implementation.name.clone(),
                    version: implementation.version.clone(),
                    custom_attributes: implementation.custom_attributes.0.clone(),
                }
            }),
        })
    }
}
