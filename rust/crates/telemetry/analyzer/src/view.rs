// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use quent_analyzer::{
    AnalyzerError, AnalyzerResult, Entity, Model,
    resource::{
        Resource, ResourceGroup, ResourceGroupTypeDecl, ResourceTypeDecl,
        collection::ResourceCollection,
        runtime::{RtResource, RtResourceGroup},
    },
};
use quent_query_engine_analyzer::{
    QueryEngineModel,
    engine::Engine,
    model::QueryEngineEntityId as QeEntityRef,
    operator::Operator,
    plan::{Plan, tree::PlanTree},
    port::Port,
    query::Query,
    query_group::QueryGroup,
    view::InMemoryQueryEngineModelView,
    worker::Worker,
};
use quent_simulator_ui::EntityRef;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use uuid::Uuid;

use crate::{
    data_batch::{DataBatch, DataBatchExt},
    model::SiriusModel,
    task::{Task, TaskExt},
};

/// A view of the simulator model filtered to a specific query
// TODO(johanpel): figure out a better way to construct these views, or to
// filter the data on a per query basis. This is generally tricky because the
// state of resources of engines that are shared across query groups or across
// the entire engine could be modified by other queries.
pub(crate) struct SiriusModelQueryView<'a> {
    resource_types: HashMap<String, &'a ResourceTypeDecl>,
    resource_group_types: HashMap<String, &'a ResourceGroupTypeDecl>,
    query_engine: InMemoryQueryEngineModelView<'a>,
    resources: HashMap<Uuid, &'a RtResource>,
    resource_groups: HashMap<Uuid, &'a RtResourceGroup>,
    tasks: HashMap<Uuid, &'a Task>,
    data_batches: HashMap<Uuid, &'a DataBatch>,
}

impl<'a> SiriusModelQueryView<'a> {
    pub fn try_new(
        model: &'a SiriusModel,
        query_id: Uuid,
    ) -> AnalyzerResult<SiriusModelQueryView<'a>> {
        // QE scoped to single query
        let query_engine_view =
            InMemoryQueryEngineModelView::try_new(&model.query_engine, query_id)?;

        let mut resource_groups = HashMap::default();
        let mut resources = HashMap::default();

        for (resource_id, resource) in &model.arbitrary_resources.resources {
            if Self::collect_runtime_resource_ancestors(
                model,
                &query_engine_view,
                resource.parent_group_id(),
                &mut resource_groups,
            )? {
                resources.insert(*resource_id, resource);
            }
        }

        let resource_types = model
            .arbitrary_resources
            .resource_types
            .iter()
            .map(|(k, v)| (k.clone(), v))
            .collect();

        let mut result = SiriusModelQueryView {
            resource_types,
            resource_group_types: model
                .resource_group_types
                .iter()
                .map(|(name, resource_group_type)| (name.clone(), resource_group_type))
                .collect(),
            query_engine: query_engine_view,
            resource_groups,
            resources,
            tasks: HashMap::default(),
            data_batches: HashMap::default(),
        };

        // Operator entities are declared per physical operator, so a task belongs to this query's
        // view iff it computed one of the query's operators, and a data batch iff it was produced
        // by one of them.
        let operator_ids = result
            .operators()
            .map(|operator| operator.id())
            .collect::<HashSet<_>>();

        result.tasks = model
            .tasks
            .values()
            .filter(|task| {
                task.computed_operator_uuids()
                    .any(|operator_uuid| operator_ids.contains(&operator_uuid))
            })
            .map(|task| (task.id(), task))
            .collect();

        result.data_batches = model
            .data_batches
            .values()
            .filter(|data_batch| {
                data_batch
                    .producer_operator_uuid()
                    .is_some_and(|operator_uuid| operator_ids.contains(&operator_uuid))
            })
            .map(|data_batch| (data_batch.id(), data_batch))
            .collect();
        Ok(result)
    }

    fn collect_runtime_resource_ancestors(
        model: &'a SiriusModel,
        query_engine: &InMemoryQueryEngineModelView<'a>,
        mut parent_id: Uuid,
        groups: &mut HashMap<Uuid, &'a RtResourceGroup>,
    ) -> AnalyzerResult<bool> {
        // Accumulate runtime groups between the resource and the query-engine tree.
        // We only publish them if the chain eventually reaches a QE group visible in
        // this query view.
        let mut path = Vec::<&'a RtResourceGroup>::new();

        loop {
            // Found the bridge into the query-engine view. The resource structurally
            // belongs to this query view, so keep every runtime group on the path.
            if query_engine.resource_group(parent_id).is_ok() {
                for group in path {
                    groups.insert(group.id(), group);
                }
                return Ok(true);
            }

            if let Some(group) = model.arbitrary_resources.resource_groups.get(&parent_id) {
                path.push(group);
                // A runtime group without a parent cannot connect back to the QE tree, so
                // do not expose this resource or its partial ancestor path.
                let Some(next_parent_id) = group.parent_group_id() else {
                    return Ok(false);
                };
                parent_id = next_parent_id;
            } else {
                // The next parent is not a runtime group we know about, and it was not a
                // QE group above. This resource is outside the current query view.
                return Ok(false);
            };
        }
    }

    pub(crate) fn tasks(&self) -> impl Iterator<Item = &'a Task> + '_ {
        self.tasks.values().copied()
    }

    pub(crate) fn data_batches(&self) -> impl Iterator<Item = &'a DataBatch> + '_ {
        self.data_batches.values().copied()
    }

    pub(crate) fn runtime_resources(&self) -> impl Iterator<Item = &'a RtResource> + '_ {
        self.resources.values().copied()
    }

    pub(crate) fn runtime_resource_groups(&self) -> impl Iterator<Item = &'a RtResourceGroup> + '_ {
        self.resource_groups.values().copied()
    }

    pub(crate) fn runtime_resource_types(
        &self,
    ) -> impl Iterator<Item = (&str, &'a ResourceTypeDecl)> + '_ {
        self.resource_types
            .iter()
            .map(|(name, resource_type)| (name.as_str(), *resource_type))
    }

    pub(crate) fn runtime_resource_group_types(
        &self,
    ) -> impl Iterator<Item = (&str, &'a ResourceGroupTypeDecl)> + '_ {
        self.resource_group_types
            .iter()
            .map(|(name, group_type)| (name.as_str(), *group_type))
    }
}

impl<'a> QueryEngineModel for SiriusModelQueryView<'a> {
    fn engine(&self) -> AnalyzerResult<&Engine> {
        self.query_engine.engine()
    }
    fn query(&self, query_id: Uuid) -> AnalyzerResult<&Query> {
        self.query_engine.query(query_id)
    }
    fn query_group(&self, query_group_id: Uuid) -> AnalyzerResult<&QueryGroup> {
        self.query_engine.query_group(query_group_id)
    }
    fn worker(&self, worker_id: Uuid) -> AnalyzerResult<&Worker> {
        self.query_engine.worker(worker_id)
    }
    fn plan(&self, plan_id: Uuid) -> AnalyzerResult<&Plan> {
        self.query_engine.plan(plan_id)
    }
    fn operator(&self, operator_id: Uuid) -> AnalyzerResult<&Operator> {
        self.query_engine.operator(operator_id)
    }
    fn port(&self, port_id: Uuid) -> AnalyzerResult<&Port> {
        self.query_engine.port(port_id)
    }
    fn queries(&self) -> impl Iterator<Item = &Query> {
        self.query_engine.queries()
    }
    fn query_groups(&self) -> impl Iterator<Item = &QueryGroup> {
        self.query_engine.query_groups()
    }
    fn workers(&self) -> impl Iterator<Item = &Worker> {
        self.query_engine.workers()
    }
    fn plans(&self) -> impl Iterator<Item = &Plan> {
        self.query_engine.plans()
    }
    fn operators(&self) -> impl Iterator<Item = &Operator> {
        self.query_engine.operators()
    }
    fn ports(&self) -> impl Iterator<Item = &Port> {
        self.query_engine.ports()
    }
    fn plan_tree(&self, query_id: Uuid) -> AnalyzerResult<PlanTree> {
        self.query_engine.plan_tree(query_id)
    }
}

impl<'a> Model for SiriusModelQueryView<'a> {
    type EntityIdType = EntityRef;
    fn try_entity_ref(&self, entity_id: Uuid) -> AnalyzerResult<Self::EntityIdType> {
        if let Ok(qe_ref) = self.query_engine.try_entity_ref(entity_id) {
            Ok(match qe_ref {
                QeEntityRef::Engine(uuid) => EntityRef::Engine(uuid),
                QeEntityRef::Worker(uuid) => EntityRef::Worker(uuid),
                QeEntityRef::QueryGroup(uuid) => EntityRef::QueryGroup(uuid),
                QeEntityRef::Query(uuid) => EntityRef::Query(uuid),
                QeEntityRef::Plan(uuid) => EntityRef::Plan(uuid),
                QeEntityRef::Operator(uuid) => EntityRef::Operator(uuid),
                QeEntityRef::Port(uuid) => EntityRef::Port(uuid),
            })
        } else if self.resources.contains_key(&entity_id) {
            Ok(EntityRef::Resource(entity_id))
        } else if self.resource_groups.contains_key(&entity_id) {
            Ok(EntityRef::ResourceGroup(entity_id))
        } else {
            self.tasks
                .contains_key(&entity_id)
                .then_some(EntityRef::Task(entity_id))
                .ok_or(AnalyzerError::InvalidId(entity_id))
        }
    }
    fn root(&self) -> AnalyzerResult<&impl ResourceGroup> {
        self.query_engine.root()
    }
}

impl<'a> ResourceCollection for SiriusModelQueryView<'a> {
    fn resources(&self) -> impl Iterator<Item = &dyn Resource> {
        self.resources
            .values()
            .map(|resource| *resource as &dyn Resource)
    }
    fn resource_groups(&self) -> impl Iterator<Item = &dyn ResourceGroup> {
        let qe_groups = self.query_engine.resource_groups();
        let sirius_groups = self
            .resource_groups
            .values()
            .map(|r| *r as &dyn ResourceGroup);
        qe_groups.chain(sirius_groups)
    }
    fn resource(&self, resource_id: Uuid) -> AnalyzerResult<&dyn Resource> {
        // qe model has no leaf resources.
        self.resources
            .get(&resource_id)
            .map(|resource| *resource as &dyn Resource)
            .ok_or(AnalyzerError::InvalidId(resource_id))
    }
    fn resource_type(&self, resource_type_name: &str) -> AnalyzerResult<&ResourceTypeDecl> {
        self.resource_types
            .get(resource_type_name)
            .copied()
            .ok_or_else(|| AnalyzerError::InvalidTypeName(resource_type_name.to_owned()))
    }
    fn resource_group(&self, resource_group_id: Uuid) -> AnalyzerResult<&dyn ResourceGroup> {
        let qe_group = self.query_engine.resource_group(resource_group_id);
        if qe_group.is_ok() {
            qe_group
        } else {
            self.resource_groups
                .get(&resource_group_id)
                .map(|r| *r as &dyn ResourceGroup)
                .ok_or(AnalyzerError::InvalidId(resource_group_id))
        }
    }
    fn resource_group_child_groups(
        &self,
        resource_group_id: Uuid,
    ) -> AnalyzerResult<impl Iterator<Item = Uuid>> {
        self.resource_group(resource_group_id)?;
        Ok(self.resource_groups().filter_map(move |g| {
            g.parent_group_id()
                .is_some_and(|p| p == resource_group_id)
                .then_some(g.id())
        }))
    }
    fn resource_group_child_resources(
        &self,
        resource_group_id: Uuid,
    ) -> AnalyzerResult<impl Iterator<Item = Uuid>> {
        // Verify the resource group exists
        self.resource_group(resource_group_id)?;
        Ok(self
            .resources()
            .filter_map(move |r| (r.parent_group_id() == resource_group_id).then_some(r.id())))
    }
}
