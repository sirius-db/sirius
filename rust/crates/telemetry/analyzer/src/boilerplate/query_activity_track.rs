// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use quent_analyzer::resource::{Resource, ResourceTypeDecl};
use schema::QueryActivityTrackEvent;

use super::*;

#[derive(Default)]
pub(crate) struct QueryActivityTrackAccumulator {
    instance_name: Option<String>,
    parent_activity_id: Option<Uuid>,
}

impl EntityEventAccumulator for QueryActivityTrackAccumulator {
    type Payload = QueryActivityTrackEvent;

    fn push(&mut self, event: Self::Payload) {
        let QueryActivityTrackEvent::Declaration { label, activity_id } = event;

        self.instance_name = Some(label);
        self.parent_activity_id = Some(activity_id.target);
    }
}

#[derive(Debug)]
pub(crate) struct QueryActivityTrack(AnalyzedEntity<QueryActivityTrackAccumulator>);

impl QueryActivityTrack {
    pub(crate) fn try_from_event(event: Event<QueryActivityTrackEvent>) -> AnalyzerResult<Self> {
        Ok(Self(AnalyzedEntity::try_from_event(event)?))
    }

    pub(crate) fn instance_name(&self) -> &str {
        self.0
            .accumulator()
            .instance_name
            .as_deref()
            .expect("executor thread must have a declaration event")
    }

    pub(crate) fn resource_type_decl() -> ResourceTypeDecl {
        ResourceTypeDecl::unit("query_activity_track")
    }
}

impl Entity for QueryActivityTrack {
    fn id(&self) -> Uuid {
        self.0.id()
    }

    fn type_name(&self) -> &str {
        "query_activity_track"
    }

    fn earliest_timestamp(&self) -> TimeUnixNanoSec {
        self.0.earliest_timestamp()
    }

    fn latest_timestamp(&self) -> TimeUnixNanoSec {
        self.0.latest_timestamp()
    }
}

impl Resource for QueryActivityTrack {}

impl RefTreeEntity for QueryActivityTrack {
    fn parent_id(&self) -> Option<Uuid> {
        self.0.accumulator().parent_activity_id
    }
}
