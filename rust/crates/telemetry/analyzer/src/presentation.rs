// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Adapt planless queries to the pinned viewer's required root plan.
use crate::view::SiriusModelQueryView;
use quent_analyzer::AnalyzerResult;
use quent_query_engine_analyzer::QueryEngineModel;
use quent_query_engine_ui::{Plan, PlanTree, QueryEntities};
use uuid::Uuid;

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
