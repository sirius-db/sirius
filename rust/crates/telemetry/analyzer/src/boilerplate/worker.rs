// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[derive(Default)]
pub(crate) struct WorkerAccumulator {
    pub(crate) parent_engine_id: Option<Uuid>,
    pub(crate) instance_name: Option<String>,
    pub(crate) exited: bool,
}

impl EntityEventAccumulator for WorkerAccumulator {
    type Event = schema::WorkerEvent;

    fn push(&mut self, event: Self::Event) {
        match event {
            schema::WorkerEvent::Init {
                parent_engine_id,
                instance_name,
            } => {
                self.parent_engine_id = Some(parent_engine_id.target);
                self.instance_name = Some(instance_name);
            }
            schema::WorkerEvent::Exit => self.exited = true,
        }
    }
}

#[derive(Debug)]
pub struct Worker(AnalyzedEntity<WorkerAccumulator>);

impl Worker {
    pub(crate) fn try_from_event(event: Event<schema::WorkerEvent>) -> AnalyzerResult<Self> {
        Ok(Self(AnalyzedEntity::try_from_event(event)?))
    }

    pub(crate) fn push(&mut self, event: Event<schema::WorkerEvent>) -> AnalyzerResult<()> {
        self.0.push(event)
    }

    pub(crate) fn data(&self) -> &WorkerAccumulator {
        self.0.accumulator()
    }
}

impl Entity for Worker {
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

impl RefTreeEntity for Worker {
    fn parent_id(&self) -> Option<Uuid> {
        self.0.accumulator().parent_engine_id
    }
}

impl WorkerEntity for Worker {
    fn to_ui(&self, _epoch: TimeUnixNanoSec) -> query_engine_ui::Worker {
        let data = self.data();
        query_engine_ui::Worker {
            id: self.id(),
            parent_engine_id: data.parent_engine_id,
            instance_name: data.instance_name.clone(),
            start_unix_ns: Some(self.earliest_timestamp()),
            end_unix_ns: data.exited.then(|| self.latest_timestamp()),
        }
    }
}
