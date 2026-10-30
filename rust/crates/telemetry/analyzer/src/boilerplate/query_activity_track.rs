// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use quent_analyzer::{
    AnalyzerError,
    resource::{Resource, ResourceTypeDecl},
};
use schema::QueryActivityTrackEvent;

#[derive(Debug)]
pub(crate) struct QueryActivityTrack {
    id: Uuid,
    parent: Uuid,
    label: String,
    start: TimeUnixNanoSec,
    end: TimeUnixNanoSec,
}
impl QueryActivityTrack {
    pub(crate) fn try_from_event(event: Event<QueryActivityTrackEvent>) -> AnalyzerResult<Self> {
        if event.id.is_nil() {
            return Err(AnalyzerError::Validation(
                "activity track id cannot be nil".into(),
            ));
        }
        let QueryActivityTrackEvent::Declaration { label, activity_id } = event.data;
        Ok(Self {
            id: event.id,
            parent: activity_id.target,
            label,
            start: event.timestamp,
            end: event.timestamp,
        })
    }
    pub(crate) fn preparation(
        query_id: Uuid,
        start: TimeUnixNanoSec,
        end: TimeUnixNanoSec,
    ) -> Self {
        Self {
            id: QueryActivity::preparation_track_id(query_id),
            parent: query_id,
            label: "Preparation".into(),
            start,
            end,
        }
    }
    pub(crate) fn include(&mut self, start: TimeUnixNanoSec, end: TimeUnixNanoSec) {
        self.start = self.start.min(start);
        self.end = self.end.max(end);
    }
    pub(crate) fn instance_name(&self) -> &str {
        &self.label
    }
    pub(crate) fn resource_type_decl() -> ResourceTypeDecl {
        ResourceTypeDecl::unit("query_activity_track")
    }
}
impl Entity for QueryActivityTrack {
    fn id(&self) -> Uuid {
        self.id
    }
    fn type_name(&self) -> &str {
        "query_activity_track"
    }
    fn earliest_timestamp(&self) -> TimeUnixNanoSec {
        self.start
    }
    fn latest_timestamp(&self) -> TimeUnixNanoSec {
        self.end
    }
}
impl Resource for QueryActivityTrack {}
impl RefTreeEntity for QueryActivityTrack {
    fn parent_id(&self) -> Option<Uuid> {
        Some(self.parent)
    }
}
