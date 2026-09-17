// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use quent_analyzer::resource::{CapacityDecl, Resource, ResourceTypeDecl};

use super::*;

#[derive(Default)]
pub(crate) struct TaskQueueAccumulator {
    instance_name: Option<String>,
    parent_group_id: Option<Uuid>,
    bounds: Option<schema::TaskQueueBounds>,
}

impl quent_events::Entity for TaskQueueAccumulator {
    type Event = schema::TaskQueueEvent;
}

impl EntityEventAccumulator for TaskQueueAccumulator {
    fn push(&mut self, event: Self::Event) {
        let schema::TaskQueueEvent::Declaration {
            instance_name,
            parent_group_id,
            bounds,
            ..
        } = event;
        self.instance_name = Some(instance_name);
        self.parent_group_id = Some(parent_group_id.target);
        self.bounds = Some(bounds);
    }
}

#[derive(Debug)]
pub(crate) struct TaskQueue(AnalyzedEntity<TaskQueueAccumulator>);

impl TaskQueue {
    pub(crate) fn try_from_event(event: Event<schema::TaskQueueEvent>) -> AnalyzerResult<Self> {
        Ok(Self(AnalyzedEntity::try_from_event(event)?))
    }

    pub(crate) fn push(&mut self, event: Event<schema::TaskQueueEvent>) -> AnalyzerResult<()> {
        self.0.push(event)
    }

    pub(crate) fn instance_name(&self) -> &str {
        self.0
            .accumulator()
            .instance_name
            .as_deref()
            .expect("task queue must have a declaration event")
    }

    pub(crate) fn resource_type_decl() -> ResourceTypeDecl {
        ResourceTypeDecl::new("task_queue", [CapacityDecl::new_occupancy("entries")])
    }
}

impl Entity for TaskQueue {
    fn id(&self) -> Uuid {
        self.0.id()
    }

    fn type_name(&self) -> &str {
        "task_queue"
    }

    fn earliest_timestamp(&self) -> TimeUnixNanoSec {
        self.0.earliest_timestamp()
    }

    fn latest_timestamp(&self) -> TimeUnixNanoSec {
        self.0.latest_timestamp()
    }
}

impl Resource for TaskQueue {}

impl RefTreeEntity for TaskQueue {
    fn parent_id(&self) -> Option<Uuid> {
        self.0.accumulator().parent_group_id
    }
}
