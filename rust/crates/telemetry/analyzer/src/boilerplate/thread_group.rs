// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[derive(Default)]
pub(crate) struct ThreadGroupAccumulator {
    instance_name: Option<String>,
    parent_group_id: Option<Uuid>,
}

impl EntityEventAccumulator for ThreadGroupAccumulator {
    type Payload = schema::ThreadGroupEvent;

    fn push(&mut self, event: Self::Payload) {
        let schema::ThreadGroupEvent::Declaration {
            label,
            worker_id,
            gpu_device_id,
        } = event
        else {
            return;
        };
        self.instance_name = Some(label);
        self.parent_group_id = Some(gpu_device_id.map_or(worker_id.target, |gpu| gpu.target));
    }
}

#[derive(Debug)]
pub(crate) struct ThreadGroup(AnalyzedEntity<ThreadGroupAccumulator>);

impl ThreadGroup {
    pub(crate) fn try_from_event(event: Event<schema::ThreadGroupEvent>) -> AnalyzerResult<Self> {
        Ok(Self(AnalyzedEntity::try_from_event(event)?))
    }

    pub(crate) fn push(&mut self, event: Event<schema::ThreadGroupEvent>) -> AnalyzerResult<()> {
        self.0.push(event)
    }

    pub(crate) fn instance_name(&self) -> &str {
        self.0
            .accumulator()
            .instance_name
            .as_deref()
            .expect("thread group must have a declaration event")
    }
}

impl Entity for ThreadGroup {
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

impl RefTreeEntity for ThreadGroup {
    fn parent_id(&self) -> Option<Uuid> {
        self.0.accumulator().parent_group_id
    }
}
