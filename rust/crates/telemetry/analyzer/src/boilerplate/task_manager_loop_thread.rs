// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use quent_analyzer::resource::{Resource, ResourceTypeDecl};

use super::*;

#[derive(Default)]
pub(crate) struct TaskManagerLoopThreadAccumulator {
    instance_name: Option<String>,
    parent_group_id: Option<Uuid>,
}

impl EntityEventAccumulator for TaskManagerLoopThreadAccumulator {
    type Payload = schema::TaskManagerLoopThreadEvent;

    fn push(&mut self, event: Self::Payload) {
        if let schema::TaskManagerLoopThreadEvent::Spawned { label, group_id } = event {
            self.instance_name = Some(label);
            self.parent_group_id = Some(group_id.target);
        }
    }
}

#[derive(Debug)]
pub(crate) struct TaskManagerLoopThread {
    entity: AnalyzedEntity<TaskManagerLoopThreadAccumulator>,
    legacy_parent_id: Option<Uuid>,
}

impl TaskManagerLoopThread {
    pub(crate) fn try_from_event(
        event: Event<schema::TaskManagerLoopThreadEvent>,
    ) -> AnalyzerResult<Self> {
        Ok(Self {
            entity: AnalyzedEntity::try_from_event(event)?,
            legacy_parent_id: None,
        })
    }

    pub(crate) fn push(
        &mut self,
        event: Event<schema::TaskManagerLoopThreadEvent>,
    ) -> AnalyzerResult<()> {
        self.entity.push(event)
    }

    pub(crate) fn set_legacy_parent(&mut self, group_id: Uuid) {
        self.legacy_parent_id = Some(group_id);
    }

    pub(crate) fn instance_name(&self) -> &str {
        self.entity
            .accumulator()
            .instance_name
            .as_deref()
            .expect("task manager loop thread must have a declaration event")
    }

    pub(crate) fn resource_type_decl() -> ResourceTypeDecl {
        ResourceTypeDecl::unit("task_manager_loop_thread")
    }
}

impl Entity for TaskManagerLoopThread {
    fn id(&self) -> Uuid {
        self.entity.id()
    }

    fn type_name(&self) -> &str {
        "task_manager_loop_thread"
    }

    fn earliest_timestamp(&self) -> TimeUnixNanoSec {
        self.entity.earliest_timestamp()
    }

    fn latest_timestamp(&self) -> TimeUnixNanoSec {
        self.entity.latest_timestamp()
    }
}

impl Resource for TaskManagerLoopThread {}

impl RefTreeEntity for TaskManagerLoopThread {
    fn parent_id(&self) -> Option<Uuid> {
        self.legacy_parent_id
            .or(self.entity.accumulator().parent_group_id)
    }
}
