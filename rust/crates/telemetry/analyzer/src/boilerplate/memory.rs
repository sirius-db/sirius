// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use quent_analyzer::resource::{CapacityDecl, Resource, ResourceTypeDecl};

use super::*;

#[derive(Default)]
pub(crate) struct MemoryAccumulator {
    instance_name: Option<String>,
    parent_group_id: Option<Uuid>,
    bounds: Option<schema::MemorySpaceBounds>,
}

impl EntityEventAccumulator for MemoryAccumulator {
    type Payload = schema::MemorySpaceEvent;

    fn push(&mut self, event: Self::Payload) {
        if let schema::MemorySpaceEvent::Declaration {
            label,
            bounds,
            worker_id,
            gpu_id,
        } = event
        {
            self.instance_name = Some(label);
            self.parent_group_id = Some(gpu_id.map_or(worker_id.target, |gpu| gpu.target));
            self.bounds = Some(bounds);
        }
    }
}

#[derive(Debug)]
pub(crate) struct Memory(AnalyzedEntity<MemoryAccumulator>);

impl Memory {
    pub(crate) fn try_from_event(event: Event<schema::MemorySpaceEvent>) -> AnalyzerResult<Self> {
        Ok(Self(AnalyzedEntity::try_from_event(event)?))
    }

    pub(crate) fn push(&mut self, event: Event<schema::MemorySpaceEvent>) -> AnalyzerResult<()> {
        self.0.push(event)
    }

    pub(crate) fn instance_name(&self) -> &str {
        self.0
            .accumulator()
            .instance_name
            .as_deref()
            .expect("memory must have a declaration event")
    }

    pub(crate) fn resource_type_decl() -> ResourceTypeDecl {
        ResourceTypeDecl::new("memory_space", [CapacityDecl::new_occupancy("bytes")])
    }
}

impl Entity for Memory {
    fn id(&self) -> Uuid {
        self.0.id()
    }

    fn type_name(&self) -> &str {
        "memory_space"
    }

    fn earliest_timestamp(&self) -> TimeUnixNanoSec {
        self.0.earliest_timestamp()
    }

    fn latest_timestamp(&self) -> TimeUnixNanoSec {
        self.0.latest_timestamp()
    }
}

impl Resource for Memory {}

impl RefTreeEntity for Memory {
    fn parent_id(&self) -> Option<Uuid> {
        self.0.accumulator().parent_group_id
    }
}
