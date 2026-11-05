// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Independent copy operations; cloning does not change source residency.

use quent_analyzer::{
    AnalyzerError, AnalyzerResult, Entity, RefTreeEntity,
    resource::{CapacityValue, Usage, Using},
};
use quent_events::Event;
use quent_time::{TimeUnixNanoSec, span::SpanUnixNanoSec};
use sirius_telemetry_store as schema;
use uuid::Uuid;

pub(crate) const CHANNEL_BYTES: &str = "bytes";

/// One copy attempt, identified by its emitter and operation number.
#[derive(Debug)]
pub struct BatchClone {
    id: Uuid,
    clone_id: u64,
    parent_id: Uuid,
    start: Option<Event<schema::BatchCloneEvent>>,
    end: Option<Event<schema::BatchCloneEvent>>,
    channel_id: Option<Uuid>,
    destination_batch_uuid: Option<Uuid>,
    capacity: Option<CapacityValue>,
}

impl BatchClone {
    pub(crate) fn try_new(
        emitter_id: Uuid,
        clone_id: u64,
        parent_id: Uuid,
        start: Option<Event<schema::BatchCloneEvent>>,
        end: Option<Event<schema::BatchCloneEvent>>,
        channel_id: Option<Uuid>,
        destination_batch_uuid: Option<Uuid>,
    ) -> AnalyzerResult<Self> {
        if let (Some(start), Some(end)) = (&start, &end)
            && end.timestamp < start.timestamp
        {
            return Err(AnalyzerError::Validation(format!(
                "clone {clone_id} on {emitter_id} completes before it starts"
            )));
        }

        let mut copy = Self {
            id: Uuid::new_v5(&emitter_id, &clone_id.to_le_bytes()),
            clone_id,
            parent_id,
            start,
            end,
            channel_id,
            destination_batch_uuid,
            capacity: None,
        };

        // Count bytes only after successful completion on a declared channel.
        if copy.success() == Some(true)
            && copy.span().is_some_and(|span| span.duration() > 0)
            && channel_id.is_some()
        {
            copy.capacity = copy
                .bytes()
                .map(|bytes| CapacityValue::new(CHANNEL_BYTES, bytes));
        }
        Ok(copy)
    }

    pub fn clone_id(&self) -> u64 {
        self.clone_id
    }

    pub fn source_batch_id(&self) -> Option<Uuid> {
        match &self.start.as_ref()?.data {
            schema::BatchCloneEvent::Started {
                source_data_batch, ..
            } => Some(source_data_batch.target),
            _ => None,
        }
    }

    pub fn source_tier(&self) -> Option<Uuid> {
        match &self.start.as_ref()?.data {
            schema::BatchCloneEvent::Started { source_tier, .. } => Some(source_tier.target),
            _ => None,
        }
    }

    pub fn destination_tier(&self) -> Option<Uuid> {
        match &self.start.as_ref()?.data {
            schema::BatchCloneEvent::Started {
                destination_tier, ..
            } => Some(destination_tier.target),
            _ => None,
        }
    }

    pub fn destination_batch_id(&self) -> Option<u64> {
        match &self.start.as_ref()?.data {
            schema::BatchCloneEvent::Started {
                destination_data_batch_id,
                ..
            } => Some(*destination_data_batch_id),
            _ => None,
        }
    }

    pub fn destination_batch_uuid(&self) -> Option<Uuid> {
        self.destination_batch_uuid
    }

    pub fn bytes(&self) -> Option<u64> {
        match &self.start.as_ref()?.data {
            schema::BatchCloneEvent::Started { bytes, .. } => Some(*bytes),
            _ => None,
        }
    }

    pub fn success(&self) -> Option<bool> {
        match &self.end.as_ref()?.data {
            schema::BatchCloneEvent::Completed { success, .. } => Some(*success),
            _ => None,
        }
    }

    pub fn started_at(&self) -> Option<TimeUnixNanoSec> {
        self.start.as_ref().map(|event| event.timestamp)
    }

    pub fn completed_at(&self) -> Option<TimeUnixNanoSec> {
        self.end.as_ref().map(|event| event.timestamp)
    }

    pub fn span(&self) -> Option<SpanUnixNanoSec> {
        SpanUnixNanoSec::try_new(self.started_at()?, self.completed_at()?).ok()
    }

    pub fn channel_id(&self) -> Option<Uuid> {
        self.channel_id
    }
}

impl Entity for BatchClone {
    fn id(&self) -> Uuid {
        self.id
    }

    fn type_name(&self) -> &str {
        "batch_clone"
    }

    fn earliest_timestamp(&self) -> TimeUnixNanoSec {
        self.started_at()
            .or(self.completed_at())
            .expect("copy has an event")
    }

    fn latest_timestamp(&self) -> TimeUnixNanoSec {
        self.completed_at()
            .or(self.started_at())
            .expect("copy has an event")
    }
}

impl RefTreeEntity for BatchClone {
    fn parent_id(&self) -> Option<Uuid> {
        Some(self.parent_id)
    }
}

struct CloneUsage<'a> {
    copy: &'a BatchClone,
    resource_id: Uuid,
    span: SpanUnixNanoSec,
    capacity: &'a CapacityValue,
}

impl<'a> Usage<'a> for CloneUsage<'a> {
    fn entity_id(&self) -> Uuid {
        self.copy.id()
    }

    fn resource_id(&self) -> Uuid {
        self.resource_id
    }

    fn capacities(&self) -> impl Iterator<Item = &'a CapacityValue> {
        std::iter::once(self.capacity)
    }

    fn span(&self) -> SpanUnixNanoSec {
        self.span
    }
}

impl Using for BatchClone {
    fn usages(&self) -> impl Iterator<Item = impl Usage<'_>> {
        self.capacity.iter().filter_map(|capacity| {
            Some(CloneUsage {
                copy: self,
                resource_id: self.channel_id()?,
                span: self.span()?,
                capacity,
            })
        })
    }
}
