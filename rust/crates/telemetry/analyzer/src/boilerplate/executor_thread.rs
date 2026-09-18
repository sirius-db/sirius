// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use quent_analyzer::resource::{Resource, ResourceTypeDecl};

use super::*;

#[derive(Default)]
pub(crate) struct ExecutorThreadAccumulator {
    instance_name: Option<String>,
    parent_group_id: Option<Uuid>,
}

impl EntityEventAccumulator for ExecutorThreadAccumulator {
    type Event = schema::ExecutorThreadEvent;

    fn push(&mut self, event: Self::Event) {
        let schema::ExecutorThreadEvent::Declaration {
            instance_name,
            parent_group_id,
            ..
        } = event;
        self.instance_name = Some(instance_name);
        self.parent_group_id = Some(parent_group_id.target);
    }
}

#[derive(Debug)]
pub(crate) struct ExecutorThread(AnalyzedEntity<ExecutorThreadAccumulator>);

impl ExecutorThread {
    pub(crate) fn try_from_event(
        event: Event<schema::ExecutorThreadEvent>,
    ) -> AnalyzerResult<Self> {
        Ok(Self(AnalyzedEntity::try_from_event(event)?))
    }

    pub(crate) fn push(&mut self, event: Event<schema::ExecutorThreadEvent>) -> AnalyzerResult<()> {
        self.0.push(event)
    }

    pub(crate) fn instance_name(&self) -> &str {
        self.0
            .accumulator()
            .instance_name
            .as_deref()
            .expect("executor thread must have a declaration event")
    }

    pub(crate) fn resource_type_decl() -> ResourceTypeDecl {
        ResourceTypeDecl::unit("executor_thread")
    }
}

impl Entity for ExecutorThread {
    fn id(&self) -> Uuid {
        self.0.id()
    }

    fn type_name(&self) -> &str {
        "executor_thread"
    }

    fn earliest_timestamp(&self) -> TimeUnixNanoSec {
        self.0.earliest_timestamp()
    }

    fn latest_timestamp(&self) -> TimeUnixNanoSec {
        self.0.latest_timestamp()
    }
}

impl Resource for ExecutorThread {}

impl RefTreeEntity for ExecutorThread {
    fn parent_id(&self) -> Option<Uuid> {
        self.0.accumulator().parent_group_id
    }
}
