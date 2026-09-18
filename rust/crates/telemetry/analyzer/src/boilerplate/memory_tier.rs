// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use quent_analyzer::resource::{CapacityDecl, Resource, ResourceTypeDecl};

use super::*;

#[derive(Default)]
pub(crate) struct MemoryTierAccumulator {
    instance_name: Option<String>,
    parent_group_id: Option<Uuid>,
    bounds: Option<schema::MemoryTierBounds>,
}

impl EntityEventAccumulator for MemoryTierAccumulator {
    type Event = schema::MemoryTierEvent;

    fn push(&mut self, event: Self::Event) {
        let schema::MemoryTierEvent::Declaration {
            instance_name,
            parent_group_id,
            bounds,
        } = event;
        self.instance_name = Some(instance_name);
        self.parent_group_id = Some(parent_group_id.target);
        self.bounds = Some(bounds);
    }
}

#[derive(Debug)]
pub(crate) struct MemoryTier(AnalyzedEntity<MemoryTierAccumulator>);

impl MemoryTier {
    pub(crate) fn try_from_event(event: Event<schema::MemoryTierEvent>) -> AnalyzerResult<Self> {
        Ok(Self(AnalyzedEntity::try_from_event(event)?))
    }

    pub(crate) fn push(&mut self, event: Event<schema::MemoryTierEvent>) -> AnalyzerResult<()> {
        self.0.push(event)
    }

    pub(crate) fn instance_name(&self) -> &str {
        self.0
            .accumulator()
            .instance_name
            .as_deref()
            .expect("memory tier must have a declaration event")
    }

    #[cfg(test)]
    pub(crate) fn bounds(&self) -> &schema::MemoryTierBounds {
        self.0
            .accumulator()
            .bounds
            .as_ref()
            .expect("memory tier must have a declaration event")
    }

    pub(crate) fn resource_type_decl() -> ResourceTypeDecl {
        ResourceTypeDecl::new("memory_tier", [CapacityDecl::new_occupancy("bytes")])
    }
}

impl Entity for MemoryTier {
    fn id(&self) -> Uuid {
        self.0.id()
    }

    fn type_name(&self) -> &str {
        "memory_tier"
    }

    fn earliest_timestamp(&self) -> TimeUnixNanoSec {
        self.0.earliest_timestamp()
    }

    fn latest_timestamp(&self) -> TimeUnixNanoSec {
        self.0.latest_timestamp()
    }
}

impl Resource for MemoryTier {}

impl RefTreeEntity for MemoryTier {
    fn parent_id(&self) -> Option<Uuid> {
        self.0.accumulator().parent_group_id
    }
}
