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
}

impl EntityEventAccumulator for ChannelAccumulator {
    type Payload = schema::ChannelEvent;

    fn push(&mut self, event: Self::Payload) {
        if let schema::ChannelEvent::Declaration {
            source_tier,
            destination_tier,
            worker_id,
            gpu_id,
            label,
        } = event
        {
            self.instance_name = Some(label);
            self.parent_group_id = Some(gpu_id.map_or(worker_id.target, |gpu| gpu.target));
            self.source_id = Some(source_tier.target);
            self.target_id = Some(destination_tier.target);
        }
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
