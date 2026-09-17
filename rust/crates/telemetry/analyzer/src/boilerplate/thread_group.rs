// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[derive(Default)]
pub(crate) struct ThreadGroupAccumulator {
    instance_name: Option<String>,
    parent_group_id: Option<Uuid>,
}

impl quent_events::Entity for ThreadGroupAccumulator {
    type Event = schema::ThreadGroupEvent;
}

impl EntityEventAccumulator for ThreadGroupAccumulator {
    fn push(&mut self, event: Self::Event) {
        let schema::ThreadGroupEvent::Declaration {
            instance_name,
            parent_group_id,
            ..
        } = event;
        self.instance_name = Some(instance_name);
        self.parent_group_id = Some(parent_group_id.target);
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
