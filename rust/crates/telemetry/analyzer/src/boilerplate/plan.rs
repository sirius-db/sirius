// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[derive(Default)]
pub(crate) struct PlanAccumulator {
    pub(crate) instance_name: Option<String>,
    pub(crate) parent_query_id: Option<Uuid>,
    pub(crate) parent_plan_id: Option<Uuid>,
    pub(crate) worker_id: Option<Uuid>,
    pub(crate) edges: Vec<(Uuid, Uuid)>,
}

impl quent_events::Entity for PlanAccumulator {
    type Event = schema::PlanEvent;
}

impl EntityEventAccumulator for PlanAccumulator {
    fn push(&mut self, event: Self::Event) {
        let schema::PlanEvent::Declaration {
            parent,
            instance_name,
            edges,
            worker_id,
        } = event;
        self.instance_name = Some(instance_name);
        self.parent_query_id = Some(parent.query_id.target);
        self.parent_plan_id = parent.plan_id.map(|plan| plan.target);
        self.worker_id = worker_id.map(|worker| worker.target);
        self.edges = edges
            .into_iter()
            .map(|edge| (edge.source.target, edge.target.target))
            .collect();
    }
}

#[derive(Debug)]
pub struct Plan(AnalyzedEntity<PlanAccumulator>);

impl Plan {
    pub(crate) fn try_from_event(event: Event<schema::PlanEvent>) -> AnalyzerResult<Self> {
        Ok(Self(AnalyzedEntity::try_from_event(event)?))
    }

    pub(crate) fn push(&mut self, event: Event<schema::PlanEvent>) -> AnalyzerResult<()> {
        self.0.push(event)
    }

    pub(crate) fn data(&self) -> &PlanAccumulator {
        self.0.accumulator()
    }
}

impl Entity for Plan {
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

impl RefTreeEntity for Plan {
    fn parent_id(&self) -> Option<Uuid> {
        self.0.accumulator().parent_query_id
    }
}

impl PlanEntity for Plan {
    fn parent_query_id(&self) -> Option<Uuid> {
        let data = self.data();
        data.parent_plan_id
            .is_none()
            .then_some(data.parent_query_id)
            .flatten()
    }

    fn parent_plan_id(&self) -> Option<Uuid> {
        self.data().parent_plan_id
    }

    fn worker_id(&self) -> Option<Uuid> {
        self.data().worker_id
    }

    fn edges(&self) -> impl Iterator<Item = (Uuid, Uuid)> + '_ {
        self.data().edges.iter().copied()
    }

    fn to_ui(&self) -> query_engine_ui::Plan {
        let data = self.data();
        query_engine_ui::Plan {
            id: self.id(),
            instance_name: data.instance_name.clone(),
            parent: data.parent_plan_id.or(data.parent_query_id),
            worker_id: data.worker_id,
            edges: data
                .edges
                .iter()
                .map(|&(source, target)| query_engine_ui::Edge { source, target })
                .collect(),
        }
    }
}
