// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use quent_analyzer::resource::{CapacityDecl, Resource, ResourceTypeDecl};

use super::*;

#[derive(Default)]
pub(crate) struct ChannelAccumulator {
    instance_name: Option<String>,
    parent_group_id: Option<Uuid>,
    source_id: Option<Uuid>,
    target_id: Option<Uuid>,
    bounds: Option<schema::ChannelBounds>,
}

impl EntityEventAccumulator for ChannelAccumulator {
    type Event = schema::ChannelEvent;

    fn push(&mut self, event: Self::Event) {
        let schema::ChannelEvent::Declaration {
            instance_name,
            parent_group_id,
            source_id,
            target_id,
            bounds,
        } = event;
        self.instance_name = Some(instance_name);
        self.parent_group_id = Some(parent_group_id.target);
        self.source_id = Some(source_id.target);
        self.target_id = Some(target_id.target);
        self.bounds = Some(bounds);
    }
}

#[derive(Debug)]
pub(crate) struct Channel(AnalyzedEntity<ChannelAccumulator>);

impl Channel {
    pub(crate) fn try_from_event(event: Event<schema::ChannelEvent>) -> AnalyzerResult<Self> {
        Ok(Self(AnalyzedEntity::try_from_event(event)?))
    }

    pub(crate) fn push(&mut self, event: Event<schema::ChannelEvent>) -> AnalyzerResult<()> {
        self.0.push(event)
    }

    pub(crate) fn instance_name(&self) -> &str {
        self.0
            .accumulator()
            .instance_name
            .as_deref()
            .expect("channel must have a declaration event")
    }

    pub(crate) fn resource_type_decl() -> ResourceTypeDecl {
        ResourceTypeDecl::new("channel", [CapacityDecl::new_rate("bytes")])
    }
}

impl Entity for Channel {
    fn id(&self) -> Uuid {
        self.0.id()
    }

    fn type_name(&self) -> &str {
        "channel"
    }

    fn earliest_timestamp(&self) -> TimeUnixNanoSec {
        self.0.earliest_timestamp()
    }

    fn latest_timestamp(&self) -> TimeUnixNanoSec {
        self.0.latest_timestamp()
    }
}

impl Resource for Channel {}

impl RefTreeEntity for Channel {
    fn parent_id(&self) -> Option<Uuid> {
        self.0.accumulator().parent_group_id
    }
}
