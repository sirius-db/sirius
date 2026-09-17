// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use quent_analyzer::{
    AnalyzerError, AnalyzerResult, Entity, Model, RefTreeEntity,
    ref_tree::RefTreeCollection,
    resource::{Resource, ResourceTypeDecl, collection::ResourceCollection},
};
use quent_query_engine_analyzer::{QueryEngineModel, plan_tree::PlanTree};
use quent_query_engine_ui::EntityRef;
use quent_ui::ResourceGroupTypeDecl;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use uuid::Uuid;

use crate::{
    boilerplate::{
        BatchPlacement, BatchPlacementExt, DataBatch, DataBatchExt, Engine, Operator, Plan, Port,
        Query, QueryGroup, Task, TaskExt, Worker,
    },
    model::SiriusModel,
};

/// A view of the Sirius model filtered to a specific query.
// TODO(johanpel): figure out a better way to construct these views, or to
// filter the data on a per query basis. This is generally tricky because the
// state of resources of engines that are shared across query groups or across
// the entire engine could be modified by other queries.
pub(crate) struct SiriusModelQueryView<'a> {
    model: &'a SiriusModel,
    resource_types: HashMap<String, &'a ResourceTypeDecl>,
    resource_group_types: HashMap<String, &'a ResourceGroupTypeDecl>,
    engine: &'a Engine,
    query_group: &'a QueryGroup,
    query: &'a Query,
    workers: HashMap<Uuid, &'a Worker>,
    plans: HashMap<Uuid, &'a Plan>,
    operators: HashMap<Uuid, &'a Operator>,
    ports: HashMap<Uuid, &'a Port>,
    resources: HashMap<Uuid, &'a dyn Resource>,
    resource_groups: HashMap<Uuid, &'a dyn RefTreeEntity>,
    ref_tree_entities: HashMap<Uuid, &'a dyn RefTreeEntity>,
    tasks: HashMap<Uuid, &'a Task>,
    data_batches: HashMap<Uuid, &'a DataBatch>,
    batch_placements: HashMap<Uuid, &'a BatchPlacement>,
}

impl<'a> SiriusModelQueryView<'a> {
    pub fn try_new(model: &'a SiriusModel, query_id: Uuid) -> AnalyzerResult<Self> {
        let query = model.query(query_id)?;
        let query_group_id = query.query_group_id().ok_or_else(|| {
            AnalyzerError::IncompleteEntity(format!("query {query_id} has no query group"))
        })?;
        let query_group = model.query_group(query_group_id)?;
        let workers = model
            .query_workers(query_id)?
            .map(|worker| (worker.id(), worker))
            .collect::<HashMap<_, _>>();
        let plans = model
            .query_plans(query_id)?
            .map(|plan| (plan.id(), plan))
            .collect::<HashMap<_, _>>();
        let operators = model
            .plans_operators(plans.values().copied())?
            .map(|operator| (operator.id(), operator))
            .collect::<HashMap<_, _>>();
        let ports = model
            .operators_ports(operators.values().copied())?
            .map(|port| (port.id(), port))
            .collect::<HashMap<_, _>>();

        let query_engine_entity_ids = std::iter::once(model.engine.id())
            .chain(std::iter::once(query_group.id()))
            .chain(std::iter::once(query.id()))
            .chain(workers.keys().copied())
            .chain(plans.keys().copied())
            .chain(operators.keys().copied())
            .chain(ports.keys().copied())
            .collect::<HashSet<_>>();

        let mut resource_groups = HashMap::default();
        let mut resources = HashMap::default();
        for resource in model.resources() {
            if Self::collect_sirius_resource_ancestors(
                model,
                &query_engine_entity_ids,
                resource.id(),
                &mut resource_groups,
            )? {
                resources.insert(resource.id(), resource);
            }
        }

        let pipeline_ids = operators.keys().copied().collect::<HashSet<_>>();
        let tasks = model
            .tasks
            .values()
            .filter(|task| {
                task.pipeline_uuid()
                    .is_some_and(|pipeline_uuid| pipeline_ids.contains(&pipeline_uuid))
            })
            .map(|task| (task.id(), task))
            .collect::<HashMap<_, _>>();
        let data_batches = model
            .data_batches
            .values()
            .filter(|data_batch| {
                data_batch
                    .producer_pipeline_uuid()
                    .is_some_and(|pipeline_uuid| pipeline_ids.contains(&pipeline_uuid))
            })
            .map(|data_batch| (data_batch.id(), data_batch))
            .collect::<HashMap<_, _>>();
        let batch_placements = model
            .batch_placements
            .values()
            .filter(|batch| {
                batch
                    .pipeline_uuid()
                    .is_some_and(|pipeline_uuid| pipeline_ids.contains(&pipeline_uuid))
            })
            .map(|batch| (batch.id(), batch))
            .collect::<HashMap<_, _>>();

        let scoped_entity_ids = query_engine_entity_ids
            .iter()
            .copied()
            .chain(resource_groups.keys().copied())
            .chain(resources.keys().copied())
            .chain(tasks.keys().copied())
            .chain(data_batches.keys().copied())
            .chain(batch_placements.keys().copied());
        let ref_tree_entities = scoped_entity_ids
            .map(|id| model.ref_tree_entity(id).map(|entity| (id, entity)))
            .collect::<AnalyzerResult<HashMap<_, _>>>()?;

        Ok(Self {
            model,
            resource_types: model
                .resource_types
                .iter()
                .map(|(name, declaration)| (name.clone(), declaration))
                .collect(),
            resource_group_types: model
                .resource_group_types
                .iter()
                .map(|(name, declaration)| (name.clone(), declaration))
                .collect(),
            engine: &model.engine,
            query_group,
            query,
            workers,
            plans,
            operators,
            ports,
            resources,
            resource_groups,
            ref_tree_entities,
            tasks,
            data_batches,
            batch_placements,
        })
    }

    fn collect_sirius_resource_ancestors(
        model: &'a SiriusModel,
        query_engine_entity_ids: &HashSet<Uuid>,
        resource_id: Uuid,
        groups: &mut HashMap<Uuid, &'a dyn RefTreeEntity>,
    ) -> AnalyzerResult<bool> {
        let Some(mut parent_id) = model.ref_tree_entity(resource_id)?.parent_id() else {
            return Ok(false);
        };
        let mut path: Vec<&'a dyn RefTreeEntity> = Vec::new();
        let mut visited = HashSet::default();

        loop {
            if query_engine_entity_ids.contains(&parent_id) {
                for group in path {
                    groups.insert(group.id(), group);
                }
                return Ok(true);
            }
            if !visited.insert(parent_id) {
                return Err(AnalyzerError::Validation(format!(
                    "resource ancestor cycle at {parent_id}"
                )));
            }

            let group = model
                .gpu_devices
                .get(&parent_id)
                .map(|group| group as &dyn RefTreeEntity)
                .or_else(|| {
                    model
                        .thread_groups
                        .get(&parent_id)
                        .map(|group| group as &dyn RefTreeEntity)
                });
            let Some(group) = group else {
                return Ok(false);
            };
            path.push(group);
            let Some(next_parent_id) = group.parent_id() else {
                return Ok(false);
            };
            parent_id = next_parent_id;
        }
    }

    pub(crate) fn tasks(&self) -> impl Iterator<Item = &'a Task> + '_ {
        self.tasks.values().copied()
    }

    pub(crate) fn data_batches(&self) -> impl Iterator<Item = &'a DataBatch> + '_ {
        self.data_batches.values().copied()
    }

    pub(crate) fn batch_placements(&self) -> impl Iterator<Item = &'a BatchPlacement> + '_ {
        self.batch_placements.values().copied()
    }

    pub(crate) fn sirius_resources(&self) -> impl Iterator<Item = &'a dyn Resource> + '_ {
        self.resources.values().copied()
    }

    pub(crate) fn sirius_resource_groups(
        &self,
    ) -> impl Iterator<Item = &'a dyn RefTreeEntity> + '_ {
        self.resource_groups.values().copied()
    }

    pub(crate) fn resource_instance_name(&self, resource_id: Uuid) -> Option<&'a str> {
        self.resources.get(&resource_id)?;
        self.model.resource_instance_name(resource_id)
    }

    pub(crate) fn resource_group_instance_name(&self, group_id: Uuid) -> Option<&'a str> {
        self.resource_groups.get(&group_id)?;
        self.model.resource_scope_instance_name(group_id)
    }

    pub(crate) fn sirius_resource_types(
        &self,
    ) -> impl Iterator<Item = (&str, &'a ResourceTypeDecl)> + '_ {
        self.resource_types
            .iter()
            .map(|(name, resource_type)| (name.as_str(), *resource_type))
    }

    pub(crate) fn sirius_resource_group_types(
        &self,
    ) -> impl Iterator<Item = (&str, &'a ResourceGroupTypeDecl)> + '_ {
        self.resource_group_types
            .iter()
            .map(|(name, group_type)| (name.as_str(), *group_type))
    }
}

impl QueryEngineModel for SiriusModelQueryView<'_> {
    type Engine = Engine;
    type Query = Query;
    type QueryGroup = QueryGroup;
    type Worker = Worker;
    type Plan = Plan;
    type Operator = Operator;
    type Port = Port;

    fn engine(&self) -> AnalyzerResult<&Engine> {
        Ok(self.engine)
    }

    fn query(&self, query_id: Uuid) -> AnalyzerResult<&Query> {
        (self.query.id() == query_id)
            .then_some(self.query)
            .ok_or(AnalyzerError::InvalidId(query_id))
    }

    fn query_group(&self, query_group_id: Uuid) -> AnalyzerResult<&QueryGroup> {
        (self.query_group.id() == query_group_id)
            .then_some(self.query_group)
            .ok_or(AnalyzerError::InvalidId(query_group_id))
    }

    fn worker(&self, worker_id: Uuid) -> AnalyzerResult<&Worker> {
        self.workers
            .get(&worker_id)
            .copied()
            .ok_or(AnalyzerError::InvalidId(worker_id))
    }

    fn plan(&self, plan_id: Uuid) -> AnalyzerResult<&Plan> {
        self.plans
            .get(&plan_id)
            .copied()
            .ok_or(AnalyzerError::InvalidId(plan_id))
    }

    fn operator(&self, operator_id: Uuid) -> AnalyzerResult<&Operator> {
        self.operators
            .get(&operator_id)
            .copied()
            .ok_or(AnalyzerError::InvalidId(operator_id))
    }

    fn port(&self, port_id: Uuid) -> AnalyzerResult<&Port> {
        self.ports
            .get(&port_id)
            .copied()
            .ok_or(AnalyzerError::InvalidId(port_id))
    }

    fn queries(&self) -> impl Iterator<Item = &Query> {
        std::iter::once(self.query)
    }

    fn query_groups(&self) -> impl Iterator<Item = &QueryGroup> {
        std::iter::once(self.query_group)
    }

    fn workers(&self) -> impl Iterator<Item = &Worker> {
        self.workers.values().copied()
    }

    fn plans(&self) -> impl Iterator<Item = &Plan> {
        self.plans.values().copied()
    }

    fn operators(&self) -> impl Iterator<Item = &Operator> {
        self.operators.values().copied()
    }

    fn ports(&self) -> impl Iterator<Item = &Port> {
        self.ports.values().copied()
    }

    fn plan_tree(&self, query_id: Uuid) -> AnalyzerResult<PlanTree> {
        PlanTree::try_new(self.plans.values().copied(), query_id)
    }
}

impl Model for SiriusModelQueryView<'_> {
    type EntityIdType = EntityRef;

    fn try_entity_ref(&self, entity_id: Uuid) -> AnalyzerResult<Self::EntityIdType> {
        if self.engine.id() == entity_id {
            Ok(EntityRef::Engine(entity_id))
        } else if self.workers.contains_key(&entity_id) {
            Ok(EntityRef::Worker(entity_id))
        } else if self.query_group.id() == entity_id {
            Ok(EntityRef::QueryGroup(entity_id))
        } else if self.query.id() == entity_id {
            Ok(EntityRef::Query(entity_id))
        } else if self.plans.contains_key(&entity_id) {
            Ok(EntityRef::Plan(entity_id))
        } else if self.operators.contains_key(&entity_id) {
            Ok(EntityRef::Operator(entity_id))
        } else if self.ports.contains_key(&entity_id) {
            Ok(EntityRef::Port(entity_id))
        } else if self.resources.contains_key(&entity_id) {
            Ok(EntityRef::Resource(entity_id))
        } else if self.resource_groups.contains_key(&entity_id) {
            Ok(EntityRef::ResourceGroup(entity_id))
        } else {
            self.tasks
                .get(&entity_id)
                .map(|task| EntityRef::Application {
                    type_name: task.type_name().to_owned(),
                    id: entity_id,
                })
                .or_else(|| {
                    self.data_batches
                        .get(&entity_id)
                        .map(|data_batch| EntityRef::Application {
                            type_name: data_batch.type_name().to_owned(),
                            id: entity_id,
                        })
                })
                .or_else(|| {
                    self.batch_placements
                        .get(&entity_id)
                        .map(|batch| EntityRef::Application {
                            type_name: batch.type_name().to_owned(),
                            id: entity_id,
                        })
                })
                .ok_or(AnalyzerError::InvalidId(entity_id))
        }
    }
}

impl ResourceCollection for SiriusModelQueryView<'_> {
    fn resources(&self) -> impl Iterator<Item = &dyn Resource> {
        self.resources.values().copied()
    }

    fn resource(&self, resource_id: Uuid) -> AnalyzerResult<&dyn Resource> {
        self.resources
            .get(&resource_id)
            .copied()
            .ok_or(AnalyzerError::InvalidId(resource_id))
    }

    fn resource_type(&self, resource_type_name: &str) -> AnalyzerResult<&ResourceTypeDecl> {
        self.resource_types
            .get(resource_type_name)
            .copied()
            .ok_or_else(|| AnalyzerError::InvalidTypeName(resource_type_name.to_owned()))
    }
}

impl RefTreeCollection for SiriusModelQueryView<'_> {
    fn ref_tree_entities(&self) -> impl Iterator<Item = &dyn RefTreeEntity> {
        self.ref_tree_entities.values().copied()
    }

    fn ref_tree_entity(&self, entity_id: Uuid) -> AnalyzerResult<&dyn RefTreeEntity> {
        self.ref_tree_entities
            .get(&entity_id)
            .copied()
            .ok_or(AnalyzerError::InvalidId(entity_id))
    }
}
