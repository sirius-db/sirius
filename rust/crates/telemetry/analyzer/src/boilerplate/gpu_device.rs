// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[derive(Default)]
pub(crate) struct GpuDeviceAccumulator {
    instance_name: Option<String>,
    parent_group_id: Option<Uuid>,
    ordinal: Option<u32>,
}

impl quent_events::Entity for GpuDeviceAccumulator {
    type Event = schema::GpuDeviceEvent;
}

impl EntityEventAccumulator for GpuDeviceAccumulator {
    fn push(&mut self, event: Self::Event) {
        let schema::GpuDeviceEvent::Declaration {
            instance_name,
            parent_group_id,
            ordinal,
        } = event;
        self.instance_name = Some(instance_name);
        self.parent_group_id = Some(parent_group_id.target);
        self.ordinal = Some(ordinal);
    }
}

#[derive(Debug)]
pub(crate) struct GpuDevice(AnalyzedEntity<GpuDeviceAccumulator>);

impl GpuDevice {
    pub(crate) fn try_from_event(event: Event<schema::GpuDeviceEvent>) -> AnalyzerResult<Self> {
        Ok(Self(AnalyzedEntity::try_from_event(event)?))
    }

    pub(crate) fn push(&mut self, event: Event<schema::GpuDeviceEvent>) -> AnalyzerResult<()> {
        self.0.push(event)
    }

    pub(crate) fn instance_name(&self) -> &str {
        self.0
            .accumulator()
            .instance_name
            .as_deref()
            .expect("GPU device must have a declaration event")
    }
}

impl Entity for GpuDevice {
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

impl RefTreeEntity for GpuDevice {
    fn parent_id(&self) -> Option<Uuid> {
        self.0.accumulator().parent_group_id
    }
}
