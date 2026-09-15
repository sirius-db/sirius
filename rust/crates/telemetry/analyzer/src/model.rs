// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::hash_map::Entry;

use rustc_hash::FxHashMap as HashMap;
use sirius_telemetry_store::{self as schema, SiriusEvent};

use quent_analyzer::{
    AnalyzerError, AnalyzerResult, Entity, Model,
    fsm::collection::FsmCollection,
    resource::{
        CapacityDecl, CapacityValue, Resource, ResourceCapacities, ResourceGroup,
        ResourceGroupTypeDecl, ResourceTypeDecl, Usage, Using,
        collection::{ResourceCollection, derive_resource_group_types},
        runtime::RtResourceGroup,
    },
};
use quent_events::Event;
use quent_query_engine_analyzer::{
    OperatorEntityMut, QueryEngineEntityId, QueryEngineModel, QueryEngineModelMut,
    plan_tree::PlanTree,
};
use quent_query_engine_ui::EntityRef;
use tracing::warn;
use uuid::Uuid;

use crate::{
    batch_placement::{BatchPlacement, BatchPlacementBuilder},
    data_batch::{DataBatch, DataBatchBuilder, DataBatchExt},
    query_engine::{
        Engine, Operator, Plan, Port, Query, QueryEngine, QueryEngineBuilder, QueryGroup, Worker,
    },
    task::{Task, TaskBuilder, TaskExt},
    view::SiriusModelQueryView,
};

const GPU_DEVICE_GROUP_TYPE_NAME: &str = "gpu_device";
const THREAD_GROUP_TYPE_NAME: &str = "thread_group";
const TASK_QUEUE_TYPE_NAME: &str = "task_queue";
const TASK_MANAGER_LOOP_THREAD_TYPE_NAME: &str = "task_manager_loop_thread";
const EXECUTOR_THREAD_TYPE_NAME: &str = "executor_thread";
const QUEUE_ENTRIES_CAPACITY_NAME: &str = "entries";
const MEMORY_TYPE_NAME: &str = "memory";
const CHANNEL_TYPE_NAME: &str = "channel";
const MEMORY_BYTES_CAPACITY_NAME: &str = "bytes";
const CHANNEL_BYTES_CAPACITY_NAME: &str = "bytes";
/// Type name of the MemoryTier resources as recorded by the model.
pub(crate) const MEMORY_TIER_TYPE_NAME: &str = "memory_tier";
/// Capacity name of the MemoryTier `bytes` capacity as recorded by the model.
pub(crate) const MEMORY_TIER_BYTES_CAPACITY_NAME: &str = "bytes";

pub(crate) struct DeclaredResource {
    id: Uuid,
    instance_name: String,
    type_name: String,
    parent_group_id: Uuid,
    bounds: ResourceCapacities,
}

impl DeclaredResource {
    pub(crate) fn bounds(&self) -> &ResourceCapacities {
        &self.bounds
    }
}

impl Entity for DeclaredResource {
    fn id(&self) -> Uuid {
        self.id
    }

    fn type_name(&self) -> &str {
        &self.type_name
    }

    fn instance_name(&self) -> &str {
        &self.instance_name
    }
}

impl Resource for DeclaredResource {
    fn parent_group_id(&self) -> Uuid {
        self.parent_group_id
    }
}

#[derive(Default)]
pub(crate) struct SiriusResources {
    pub(crate) resource_types: HashMap<String, ResourceTypeDecl>,
    pub(crate) resources: HashMap<Uuid, DeclaredResource>,
    pub(crate) resource_groups: HashMap<Uuid, RtResourceGroup>,
}

impl SiriusResources {
    fn insert_resource_type(&mut self, declaration: ResourceTypeDecl) {
        self.resource_types
            .entry(declaration.name.clone())
            .or_insert(declaration);
    }

    fn insert_resource(
        &mut self,
        id: Uuid,
        type_name: &str,
        instance_name: String,
        parent_group_id: Uuid,
        bounds: ResourceCapacities,
    ) -> AnalyzerResult<()> {
        if id.is_nil() {
            return Err(AnalyzerError::InvalidId(id));
        }
        if self.resources.contains_key(&id) {
            return Err(AnalyzerError::Validation(format!(
                "resource {id} has multiple declarations"
            )));
        }

        self.resources.insert(
            id,
            DeclaredResource {
                id,
                instance_name,
                type_name: type_name.to_owned(),
                parent_group_id,
                bounds,
            },
        );
        Ok(())
    }

    fn push_group_raw(
        &mut self,
        id: Uuid,
        type_name: &str,
        instance_name: &str,
        parent_group_id: Option<Uuid>,
    ) {
        self.resource_groups.insert(
            id,
            RtResourceGroup {
                id,
                type_name: type_name.to_owned(),
                instance_name: instance_name.to_owned(),
                parent_group_id,
            },
        );
    }
}

impl ResourceCollection for SiriusResources {
    fn resources(&self) -> impl Iterator<Item = &dyn Resource> {
        self.resources
            .values()
            .map(|resource| resource as &dyn Resource)
    }

    fn resource_groups(&self) -> impl Iterator<Item = &dyn ResourceGroup> {
        self.resource_groups
            .values()
            .map(|group| group as &dyn ResourceGroup)
    }

    fn resource(&self, resource_id: Uuid) -> AnalyzerResult<&dyn Resource> {
        self.resources
            .get(&resource_id)
            .map(|resource| resource as &dyn Resource)
            .ok_or(AnalyzerError::InvalidId(resource_id))
    }

    fn resource_type(&self, resource_type_name: &str) -> AnalyzerResult<&ResourceTypeDecl> {
        self.resource_types
            .get(resource_type_name)
            .ok_or_else(|| AnalyzerError::InvalidTypeName(resource_type_name.to_owned()))
    }

    fn resource_group(&self, resource_group_id: Uuid) -> AnalyzerResult<&dyn ResourceGroup> {
        self.resource_groups
            .get(&resource_group_id)
            .map(|group| group as &dyn ResourceGroup)
            .ok_or(AnalyzerError::InvalidId(resource_group_id))
    }

    fn resource_group_child_groups(
        &self,
        resource_group_id: Uuid,
    ) -> AnalyzerResult<impl Iterator<Item = Uuid>> {
        self.resource_group(resource_group_id)?;
        Ok(self.resource_groups.values().filter_map(move |group| {
            group
                .parent_group_id
                .is_some_and(|parent| parent == resource_group_id)
                .then_some(group.id)
        }))
    }

    fn resource_group_child_resources(
        &self,
        resource_group_id: Uuid,
    ) -> AnalyzerResult<impl Iterator<Item = Uuid>> {
        self.resource_group(resource_group_id)?;
        Ok(self.resources.values().filter_map(move |resource| {
            (resource.parent_group_id == resource_group_id).then_some(resource.id)
        }))
    }
}

/// The analyzed Sirius engine model.
pub struct SiriusModel {
    pub(crate) query_engine: QueryEngine,
    pub(crate) sirius_resources: SiriusResources,
    pub(crate) tasks: HashMap<Uuid, Task>,
    pub(crate) data_batches: HashMap<Uuid, DataBatch>,
    pub(crate) batch_placements: HashMap<Uuid, BatchPlacement>,
    pub(crate) resource_group_types: HashMap<String, ResourceGroupTypeDecl>,
}

impl Model for SiriusModel {
    type EntityIdType = EntityRef;

    fn try_entity_ref(&self, entity_id: Uuid) -> AnalyzerResult<Self::EntityIdType> {
        if let Ok(qe_ref) = self.query_engine.try_entity_ref(entity_id) {
            Ok(match qe_ref {
                QueryEngineEntityId::Engine(uuid) => EntityRef::Engine(uuid),
                QueryEngineEntityId::Worker(uuid) => EntityRef::Worker(uuid),
                QueryEngineEntityId::QueryGroup(uuid) => EntityRef::QueryGroup(uuid),
                QueryEngineEntityId::Query(uuid) => EntityRef::Query(uuid),
                QueryEngineEntityId::Plan(uuid) => EntityRef::Plan(uuid),
                QueryEngineEntityId::Operator(uuid) => EntityRef::Operator(uuid),
                QueryEngineEntityId::Port(uuid) => EntityRef::Port(uuid),
            })
        } else if self.sirius_resources.resources.contains_key(&entity_id) {
            Ok(EntityRef::Resource(entity_id))
        } else if self
            .sirius_resources
            .resource_groups
            .contains_key(&entity_id)
        {
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

    fn root(&self) -> AnalyzerResult<&impl ResourceGroup> {
        self.query_engine.root()
    }
}

impl QueryEngineModel for SiriusModel {
    type Engine = Engine;
    type Query = Query;
    type QueryGroup = QueryGroup;
    type Worker = Worker;
    type Plan = Plan;
    type Operator = Operator;
    type Port = Port;

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

impl QueryEngineModelMut for SiriusModel {
    fn operator_mut(&mut self, operator_id: Uuid) -> AnalyzerResult<&mut Operator> {
        self.query_engine.operator_mut(operator_id)
    }
}

impl FsmCollection for SiriusModel {
    type Fsm = Task;

    fn fsms(&self) -> impl Iterator<Item = &Task> {
        self.tasks.values()
    }
}

impl SiriusModel {
    pub(crate) fn query_view(&self, query_id: Uuid) -> AnalyzerResult<SiriusModelQueryView<'_>> {
        SiriusModelQueryView::try_new(self, query_id)
    }
}

impl ResourceCollection for SiriusModel {
    fn resources(&self) -> impl Iterator<Item = &dyn Resource> {
        self.sirius_resources
            .resources()
            .chain(self.query_engine.resources())
    }
    fn resource_groups(&self) -> impl Iterator<Item = &dyn ResourceGroup> {
        self.sirius_resources
            .resource_groups()
            .chain(self.query_engine.resource_groups())
    }
    fn resource(&self, resource_id: Uuid) -> AnalyzerResult<&dyn Resource> {
        self.sirius_resources
            .resource(resource_id)
            .or_else(|_| self.query_engine.resource(resource_id))
    }
    fn resource_type(&self, resource_type_name: &str) -> AnalyzerResult<&ResourceTypeDecl> {
        self.query_engine
            .resource_type(resource_type_name)
            .or_else(|_| self.sirius_resources.resource_type(resource_type_name))
    }
    fn resource_group(&self, resource_group_id: Uuid) -> AnalyzerResult<&dyn ResourceGroup> {
        self.query_engine
            .resource_group(resource_group_id)
            .or_else(|_| self.sirius_resources.resource_group(resource_group_id))
    }

    fn resource_group_child_groups(
        &self,
        resource_group_id: Uuid,
    ) -> AnalyzerResult<impl Iterator<Item = Uuid>> {
        // Verify the resource group exists in at least one collection
        self.resource_group(resource_group_id)?;

        let engine = self
            .query_engine
            .resource_group_child_groups(resource_group_id)
            .ok();

        let sim = self
            .sirius_resources
            .resource_groups
            .values()
            .filter_map(move |group| {
                group
                    .parent_group_id
                    .and_then(|parent| (parent == resource_group_id).then_some(group.id))
            });

        Ok(engine.into_iter().flatten().chain(sim))
    }

    fn resource_group_child_resources(
        &self,
        resource_group_id: Uuid,
    ) -> AnalyzerResult<impl Iterator<Item = Uuid>> {
        // Verify the resource group exists in at least one collection
        self.resource_group(resource_group_id)?;

        let engine = self
            .query_engine
            .resource_group_child_resources(resource_group_id)
            .ok();

        let sim = self
            .sirius_resources
            .resources
            .values()
            .filter_map(move |resource| {
                (resource.parent_group_id() == resource_group_id).then_some(resource.id)
            });

        Ok(engine.into_iter().flatten().chain(sim))
    }
}

pub struct SiriusModelBuilder {
    query_engine: QueryEngineBuilder,
    sirius_resources: SiriusResources,
    tasks: HashMap<Uuid, TaskBuilder>,
    data_batches: HashMap<Uuid, DataBatchBuilder>,
    batch_placements: HashMap<Uuid, BatchPlacementBuilder>,
}

impl SiriusModelBuilder {
    pub(crate) fn try_new(engine_id: Uuid) -> AnalyzerResult<Self> {
        Ok(Self {
            query_engine: QueryEngineBuilder::try_new(engine_id)?,
            sirius_resources: SiriusResources::default(),
            tasks: HashMap::default(),
            data_batches: HashMap::default(),
            batch_placements: HashMap::default(),
        })
    }

    pub(crate) fn try_push(&mut self, event: Event<SiriusEvent>) -> AnalyzerResult<()> {
        let Event {
            id,
            timestamp,
            data,
        } = event;
        match data {
            SiriusEvent::Task(t) => {
                let task_builder = match self.tasks.entry(id) {
                    Entry::Occupied(entry) => entry.into_mut(),
                    Entry::Vacant(entry) => entry.insert(TaskBuilder::try_new(id)?),
                };
                task_builder.push_transition(Event::new(id, timestamp, t));
                Ok(())
            }
            SiriusEvent::Engine(event) => self
                .query_engine
                .push_engine(Event::new(id, timestamp, event)),
            SiriusEvent::Worker(event) => self
                .query_engine
                .push_worker(Event::new(id, timestamp, event)),
            SiriusEvent::QueryGroup(event) => self
                .query_engine
                .push_query_group(Event::new(id, timestamp, event)),
            SiriusEvent::Query(event) => self
                .query_engine
                .push_query(Event::new(id, timestamp, event)),
            SiriusEvent::Plan(event) => self
                .query_engine
                .push_plan(Event::new(id, timestamp, event)),
            SiriusEvent::Operator(event) => self
                .query_engine
                .push_operator(Event::new(id, timestamp, event)),
            SiriusEvent::Port(event) => self
                .query_engine
                .push_port(Event::new(id, timestamp, event)),
            SiriusEvent::GpuDevice(schema::GpuDeviceEvent::Declaration {
                instance_name,
                parent_group_id,
                ..
            }) => {
                self.sirius_resources.push_group_raw(
                    id,
                    GPU_DEVICE_GROUP_TYPE_NAME,
                    &instance_name,
                    Some(parent_group_id.target),
                );
                Ok(())
            }
            SiriusEvent::ThreadGroup(schema::ThreadGroupEvent::Declaration {
                instance_name,
                parent_group_id,
                ..
            }) => {
                self.sirius_resources.push_group_raw(
                    id,
                    THREAD_GROUP_TYPE_NAME,
                    &instance_name,
                    Some(parent_group_id.target),
                );
                Ok(())
            }
            SiriusEvent::TaskQueue(schema::TaskQueueEvent::Declaration {
                instance_name,
                parent_group_id,
                bounds,
                ..
            }) => self.declare_resource(
                id,
                TASK_QUEUE_TYPE_NAME,
                instance_name,
                parent_group_id.target,
                ResourceTypeDecl::new(
                    TASK_QUEUE_TYPE_NAME,
                    [CapacityDecl::new_occupancy(QUEUE_ENTRIES_CAPACITY_NAME)],
                ),
                ResourceCapacities(vec![CapacityValue::new(
                    QUEUE_ENTRIES_CAPACITY_NAME,
                    bounds.entries,
                )]),
            ),
            SiriusEvent::TaskManagerLoopThread(
                schema::TaskManagerLoopThreadEvent::Declaration {
                    instance_name,
                    parent_group_id,
                    ..
                },
            ) => self.declare_resource(
                id,
                TASK_MANAGER_LOOP_THREAD_TYPE_NAME,
                instance_name,
                parent_group_id.target,
                ResourceTypeDecl::unit(TASK_MANAGER_LOOP_THREAD_TYPE_NAME),
                ResourceCapacities(vec![CapacityValue::new("unit", 1)]),
            ),
            SiriusEvent::ExecutorThread(schema::ExecutorThreadEvent::Declaration {
                instance_name,
                parent_group_id,
                ..
            }) => self.declare_resource(
                id,
                EXECUTOR_THREAD_TYPE_NAME,
                instance_name,
                parent_group_id.target,
                ResourceTypeDecl::unit(EXECUTOR_THREAD_TYPE_NAME),
                ResourceCapacities(vec![CapacityValue::new("unit", 1)]),
            ),
            SiriusEvent::Memory(schema::MemoryEvent::Declaration {
                instance_name,
                parent_group_id,
                bounds,
            }) => self.declare_resource(
                id,
                MEMORY_TYPE_NAME,
                instance_name,
                parent_group_id.target,
                ResourceTypeDecl::new(
                    MEMORY_TYPE_NAME,
                    [CapacityDecl::new_occupancy(MEMORY_BYTES_CAPACITY_NAME)],
                ),
                ResourceCapacities(vec![CapacityValue::new(
                    MEMORY_BYTES_CAPACITY_NAME,
                    bounds.bytes,
                )]),
            ),
            SiriusEvent::Channel(schema::ChannelEvent::Declaration {
                instance_name,
                parent_group_id,
                bounds,
                ..
            }) => self.declare_resource(
                id,
                CHANNEL_TYPE_NAME,
                instance_name,
                parent_group_id.target,
                ResourceTypeDecl::new(
                    CHANNEL_TYPE_NAME,
                    [CapacityDecl::new_rate(CHANNEL_BYTES_CAPACITY_NAME)],
                ),
                ResourceCapacities(vec![CapacityValue::new(
                    CHANNEL_BYTES_CAPACITY_NAME,
                    bounds.bytes,
                )]),
            ),
            SiriusEvent::DataBatch(d) => {
                let data_batch_builder = match self.data_batches.entry(id) {
                    Entry::Occupied(entry) => entry.into_mut(),
                    Entry::Vacant(entry) => entry.insert(DataBatchBuilder::try_new(id)?),
                };
                data_batch_builder.push_transition(Event::new(id, timestamp, d));
                Ok(())
            }
            SiriusEvent::BatchPlacement(b) => {
                let batch_builder = match self.batch_placements.entry(id) {
                    Entry::Occupied(entry) => entry.into_mut(),
                    Entry::Vacant(entry) => entry.insert(BatchPlacementBuilder::try_new(id)?),
                };
                batch_builder.push_transition(Event::new(id, timestamp, b));
                Ok(())
            }
            SiriusEvent::MemoryTier(schema::MemoryTierEvent::Declaration {
                instance_name,
                parent_group_id,
                bounds,
            }) => self.declare_resource(
                id,
                MEMORY_TIER_TYPE_NAME,
                instance_name,
                parent_group_id.target,
                ResourceTypeDecl::new(
                    MEMORY_TIER_TYPE_NAME,
                    [CapacityDecl::new_occupancy(MEMORY_TIER_BYTES_CAPACITY_NAME)],
                ),
                ResourceCapacities(vec![CapacityValue::new(
                    MEMORY_TIER_BYTES_CAPACITY_NAME,
                    bounds.bytes,
                )]),
            ),
        }
    }

    fn declare_resource(
        &mut self,
        id: Uuid,
        type_name: &str,
        instance_name: String,
        parent_group_id: Uuid,
        declaration: ResourceTypeDecl,
        bounds: ResourceCapacities,
    ) -> AnalyzerResult<()> {
        self.sirius_resources.insert_resource_type(declaration);
        self.sirius_resources
            .insert_resource(id, type_name, instance_name, parent_group_id, bounds)
    }

    pub(crate) fn try_build(self) -> AnalyzerResult<SiriusModel> {
        // Build resources first. As we iterate over task builders and build all
        // tasks, we can populate the leaf resources used_by field.
        let mut resources = self.sirius_resources;
        for resource in resources.resources.values() {
            let resource_type = resources.resource_type(resource.type_name())?;
            for bound in &resource.bounds().0 {
                resource_type.try_capacity(bound.name)?;
            }
        }

        let mut query_engine = self.query_engine.try_build()?;

        let mut tasks = HashMap::default();
        for (task_id, task_builder) in self.tasks.into_iter() {
            let task = Task::from_builder(task_builder)?;
            for usage in task.usages() {
                let resource_type_name = resources
                    .resource(usage.resource_id())?
                    .type_name()
                    .to_owned();
                let set = &mut resources
                    .resource_types
                    .get_mut(&resource_type_name)
                    .unwrap()
                    .used_by;
                if !set.contains(task.type_name()) {
                    set.insert(task.type_name().to_owned());
                }
            }
            if let Some(operator_id) = task.pipeline_uuid() // Sirius Pipeline Uuid is Quent Operator Id
                && let Some(task_span) = task.active_span()
                && let Some(operator) = query_engine.operators.get_mut(&operator_id)
            {
                operator.extend_active_span(task_span);
            }

            tasks.insert(task_id, task);
        }

        let mut data_batches = HashMap::default();
        for (data_batch_id, data_batch_builder) in self.data_batches.into_iter() {
            match DataBatch::from_builder(data_batch_builder) {
                Ok(data_batch) => {
                    for usage in data_batch.usages() {
                        let resource_type_name = resources
                            .resource(usage.resource_id())?
                            .type_name()
                            .to_owned();
                        let set = &mut resources
                            .resource_types
                            .get_mut(&resource_type_name)
                            .unwrap()
                            .used_by;
                        if !set.contains(data_batch.type_name()) {
                            set.insert(data_batch.type_name().to_owned());
                        }
                    }
                    if let Some(operator_id) = data_batch.producer_pipeline_uuid() // Sirius Pipeline Uuid is Quent Operator Id
                        && let Some(data_batch_span) = data_batch.active_span()
                        && let Some(operator) = query_engine.operators.get_mut(&operator_id)
                    {
                        operator.extend_active_span(data_batch_span);
                    }

                    data_batches.insert(data_batch_id, data_batch);
                }
                Err(e) => warn!("Invalid data_batch encountered {e}"),
            }
        }

        let mut batch_placements = HashMap::default();
        for (batch_id, batch_builder) in self.batch_placements.into_iter() {
            let batch = BatchPlacement::from_builder(batch_builder)?;
            for usage in batch.usages() {
                let resource_type_name = resources
                    .resource(usage.resource_id())?
                    .type_name()
                    .to_owned();
                let set = &mut resources
                    .resource_types
                    .get_mut(&resource_type_name)
                    .unwrap()
                    .used_by;
                if !set.contains(batch.type_name()) {
                    set.insert(batch.type_name().to_owned());
                }
            }
            batch_placements.insert(batch_id, batch);
        }

        // Construct the model without group type decls being populated yet, we
        // will populate it based on the resource tree.
        let temp_model = SiriusModel {
            query_engine,
            sirius_resources: resources,
            tasks,
            data_batches,
            batch_placements,
            resource_group_types: HashMap::default(),
        };
        let mut resource_group_types = derive_resource_group_types(&temp_model)?;
        // Bubble up all the used_by_entity fields in the group type decls.
        for group_type_decl in resource_group_types.values_mut() {
            for contained_resource_type in &group_type_decl.contains_resource_types {
                if let Ok(resource_type) = temp_model
                    .sirius_resources
                    .resource_type(contained_resource_type)
                {
                    for entity_type in &resource_type.used_by {
                        group_type_decl
                            .used_by_entity_types
                            .insert(entity_type.clone());
                    }
                }
            }
        }

        Ok(SiriusModel {
            query_engine: temp_model.query_engine,
            sirius_resources: temp_model.sirius_resources,
            tasks: temp_model.tasks,
            data_batches: temp_model.data_batches,
            batch_placements: temp_model.batch_placements,
            resource_group_types,
        })
    }
}
