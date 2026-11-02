// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Adapters for the pinned viewer's resource-row and root-plan contracts.
//! Lane membership is a query relationship, independent of resource occupancy.
//! Keep these projections until the viewer supports lanes and absent plans.
use quent_analyzer::{AnalyzerError, AnalyzerResult, Entity, RefTreeEntity};
use quent_query_engine_analyzer::QueryEngineModel;
use quent_query_engine_ui::{Plan, PlanTree, QueryEntities};
use sirius_telemetry_store::QueryActivityEvent;
use uuid::Uuid;

use crate::{boilerplate::QueryActivityTrack, model::SiriusModel, view::SiriusModelQueryView};

#[derive(Clone, Copy)]
pub(crate) struct PreparationLane(Uuid);

impl PreparationLane {
    pub(crate) fn new(query_id: Uuid) -> Self {
        Self(query_id)
    }

    pub(crate) fn id(self) -> Uuid {
        Uuid::new_v5(&self.0, b"sirius/query-preparation-track")
    }

    /// Legacy viewers select intervals by resource usage. Retain the original
    /// producer reference for validation before projecting it onto the lane.
    pub(crate) fn project_transition(event: &mut QueryActivityEvent) -> Option<Uuid> {
        if let QueryActivityEvent::Started {
            query_id, track, ..
        } = event
        {
            let source = track.target;
            track.target = Self::new(query_id.target).id();
            Some(source)
        } else {
            None
        }
    }

    /// Compatibility for untyped interval-row requests from the pinned viewer.
    pub(crate) fn default_entity_type(
        model: &SiriusModel,
        scope: Option<&std::collections::HashSet<Uuid>>,
    ) -> Option<&'static str> {
        scope
            .filter(|ids| {
                !ids.is_empty() && ids.iter().all(|id| model.activity_tracks.contains_key(id))
            })
            .map(|_| QueryActivityEvent::TYPE_NAME)
    }
}

pub(crate) fn project_preparation_lanes(
    model: &mut SiriusModel,
    source_tracks: &rustc_hash::FxHashMap<Uuid, QueryActivityTrack>,
) -> AnalyzerResult<()> {
    for activity in model.activities.values() {
        // Old traces must still validate the recorded resource reference. The
        // display projection cannot turn a malformed source into a valid trace.
        if source_tracks
            .get(&activity.source_track_id())
            .is_none_or(|track| track.parent_id() != Some(activity.id()))
        {
            return Err(AnalyzerError::Validation(
                "query activity has no matching track".into(),
            ));
        }
        let query_id = activity.query_id();
        model
            .activity_tracks
            .entry(PreparationLane::new(query_id).id())
            .and_modify(|track| {
                track.include(activity.earliest_timestamp(), activity.latest_timestamp())
            })
            .or_insert_with(|| {
                QueryActivityTrack::preparation(
                    query_id,
                    activity.earliest_timestamp(),
                    activity.latest_timestamp(),
                )
            });
    }
    model
        .resource_types
        .get_mut("query_activity_track")
        .unwrap()
        .used_by
        .insert(QueryActivityEvent::TYPE_NAME.into());
    Ok(())
}

pub(crate) fn plan_tree(
    view: &SiriusModelQueryView<'_>,
    entities: &mut QueryEntities,
    query_id: Uuid,
) -> AnalyzerResult<PlanTree> {
    if !entities.plans.is_empty() {
        return Ok(view.plan_tree(query_id)?.to_ui());
    }
    // Presentation only: no plan event, operator, worker, task or execution
    // window is invented. Remove only with a supported absent-plan contract.
    let id = Uuid::new_v5(&query_id, b"sirius/query-lifecycle-placeholder-plan");
    entities.plans.insert(
        id,
        Plan {
            id,
            instance_name: Some("No execution plan - query lifecycle only".into()),
            parent: Some(query_id),
            worker_id: None,
            edges: Vec::new(),
        },
    );
    entities
        .query
        .custom_attributes
        .push(quent_dynamic_attributes::DynamicAttribute::string(
            "plan_representation",
            "query_lifecycle_placeholder",
        ));
    Ok(PlanTree {
        id,
        worker: None,
        children: Vec::new(),
    })
}
