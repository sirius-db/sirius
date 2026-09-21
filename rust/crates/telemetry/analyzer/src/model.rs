// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeSet, hash_map::Entry};

use rustc_hash::FxHashMap as HashMap;
use sirius_telemetry_store::SiriusEvent;

use quent_analyzer::{
    AnalyzerError, AnalyzerResult, Entity, Model, RefTreeEntity,
    fsm::collection::FsmCollection,
    ref_tree::RefTreeCollection,
    resource::{Resource, ResourceTypeDecl, Usage, Using, collection::ResourceCollection},
};
use quent_events::Event;
use quent_query_engine_analyzer::{
    OperatorEntityMut, QueryEngineModel, QueryEngineModelMut, plan_tree::PlanTree,
};
use quent_query_engine_ui::EntityRef;
use quent_ui::ResourceGroupTypeDecl;
use tracing::warn;
use uuid::Uuid;

pub use crate::boilerplate::{Engine, Operator, Plan, Port, Query, QueryGroup, Worker};

use crate::{
    boilerplate::{
        BatchPlacement, BatchPlacementBuilder, Channel, DataBatch, DataBatchBuilder, DataBatchExt,
        ExecutorThread, GpuDevice, Memory, MemoryTier, QueryBuilder, Task, TaskBuilder, TaskExt,
        TaskManagerLoopThread, TaskQueue, ThreadGroup,
    },
    view::SiriusModelQueryView,
};

/// Type name of the MemoryTier resources as recorded by the model.
pub(crate) const MEMORY_TIER_TYPE_NAME: &str = "memory_tier";
/// Capacity name of the MemoryTier `bytes` capacity as recorded by the model.
pub(crate) const MEMORY_TIER_BYTES_CAPACITY_NAME: &str = "bytes";

fn derive_resource_scope_types(
    model: &SiriusModel,
) -> AnalyzerResult<HashMap<String, ResourceGroupTypeDecl>> {
    fn populate(
        node: &quent_analyzer::resource::tree::ResourceTreeNode,
        model: &SiriusModel,
        declarations: &mut HashMap<String, (BTreeSet<String>, BTreeSet<String>)>,
    ) -> AnalyzerResult<()> {
        if !node.is_resource {
            let mut contained_types = Vec::new();
            for resource_id in node.iter_resource_ids() {
                contained_types.push(model.resource_type_of(resource_id)?);
            }
            if !contained_types.is_empty() {
                let type_name = model
                    .ref_tree_entity(node.entity_id)?
                    .type_name()
                    .to_owned();
                let (used_by, contains) = declarations.entry(type_name).or_default();
                for resource_type in contained_types {
                    contains.insert(resource_type.name.clone());
                    used_by.extend(resource_type.used_by.iter().cloned());
                }
            }
        }
        for child in &node.children {
            populate(child, model, declarations)?;
        }
        Ok(())
    }

    let tree = quent_analyzer::resource::tree::ResourceTreeNode::try_new(model)?;
    let mut declarations = HashMap::default();
    populate(&tree, model, &mut declarations)?;
    Ok(declarations
        .into_iter()
        .map(|(name, (used_by_entity_types, contains_resource_types))| {
            (
                name.clone(),
                ResourceGroupTypeDecl {
                    name,
                    used_by_entity_types: used_by_entity_types.into_iter().collect(),
                    contains_resource_types: contains_resource_types.into_iter().collect(),
                },
            )
        })
        .collect())
}

/// The analyzed Sirius engine model.
pub struct SiriusModel {
    pub(crate) engine: Engine,
    pub(crate) workers: HashMap<Uuid, Worker>,
    pub(crate) query_groups: HashMap<Uuid, QueryGroup>,
    pub(crate) queries: HashMap<Uuid, Query>,
    pub(crate) plans: HashMap<Uuid, Plan>,
    pub(crate) operators: HashMap<Uuid, Operator>,
    pub(crate) ports: HashMap<Uuid, Port>,
    pub(crate) resource_types: HashMap<String, ResourceTypeDecl>,
    pub(crate) gpu_devices: HashMap<Uuid, GpuDevice>,
    pub(crate) thread_groups: HashMap<Uuid, ThreadGroup>,
    pub(crate) memories: HashMap<Uuid, Memory>,
    pub(crate) channels: HashMap<Uuid, Channel>,
    pub(crate) memory_tiers: HashMap<Uuid, MemoryTier>,
    pub(crate) task_queues: HashMap<Uuid, TaskQueue>,
    pub(crate) task_manager_loop_threads: HashMap<Uuid, TaskManagerLoopThread>,
    pub(crate) executor_threads: HashMap<Uuid, ExecutorThread>,
    pub(crate) tasks: HashMap<Uuid, Task>,
    pub(crate) data_batches: HashMap<Uuid, DataBatch>,
    pub(crate) batch_placements: HashMap<Uuid, BatchPlacement>,
    pub(crate) resource_group_types: HashMap<String, ResourceGroupTypeDecl>,
}

impl Model for SiriusModel {
    type EntityIdType = EntityRef;

    fn try_entity_ref(&self, entity_id: Uuid) -> AnalyzerResult<Self::EntityIdType> {
        if self.engine.id() == entity_id {
            Ok(EntityRef::Engine(entity_id))
        } else if self.workers.contains_key(&entity_id) {
            Ok(EntityRef::Worker(entity_id))
        } else if self.query_groups.contains_key(&entity_id) {
            Ok(EntityRef::QueryGroup(entity_id))
        } else if self.queries.contains_key(&entity_id) {
            Ok(EntityRef::Query(entity_id))
        } else if self.plans.contains_key(&entity_id) {
            Ok(EntityRef::Plan(entity_id))
        } else if self.operators.contains_key(&entity_id) {
            Ok(EntityRef::Operator(entity_id))
        } else if self.ports.contains_key(&entity_id) {
            Ok(EntityRef::Port(entity_id))
        } else if self.resource(entity_id).is_ok() {
            Ok(EntityRef::Resource(entity_id))
        } else if self.gpu_devices.contains_key(&entity_id)
            || self.thread_groups.contains_key(&entity_id)
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
        Ok(&self.engine)
    }

    fn query(&self, query_id: Uuid) -> AnalyzerResult<&Query> {
        self.queries
            .get(&query_id)
            .ok_or(AnalyzerError::InvalidId(query_id))
    }

    fn query_group(&self, query_group_id: Uuid) -> AnalyzerResult<&QueryGroup> {
        self.query_groups
            .get(&query_group_id)
            .ok_or(AnalyzerError::InvalidId(query_group_id))
    }

    fn worker(&self, worker_id: Uuid) -> AnalyzerResult<&Worker> {
        self.workers
            .get(&worker_id)
            .ok_or(AnalyzerError::InvalidId(worker_id))
    }

    fn plan(&self, plan_id: Uuid) -> AnalyzerResult<&Plan> {
        self.plans
            .get(&plan_id)
            .ok_or(AnalyzerError::InvalidId(plan_id))
    }

    fn operator(&self, operator_id: Uuid) -> AnalyzerResult<&Operator> {
        self.operators
            .get(&operator_id)
            .ok_or(AnalyzerError::InvalidId(operator_id))
    }

    fn port(&self, port_id: Uuid) -> AnalyzerResult<&Port> {
        self.ports
            .get(&port_id)
            .ok_or(AnalyzerError::InvalidId(port_id))
    }

    fn queries(&self) -> impl Iterator<Item = &Query> {
        self.queries.values()
    }

    fn query_groups(&self) -> impl Iterator<Item = &QueryGroup> {
        self.query_groups.values()
    }

    fn workers(&self) -> impl Iterator<Item = &Worker> {
        self.workers.values()
    }

    fn plans(&self) -> impl Iterator<Item = &Plan> {
        self.plans.values()
    }

    fn operators(&self) -> impl Iterator<Item = &Operator> {
        self.operators.values()
    }

    fn ports(&self) -> impl Iterator<Item = &Port> {
        self.ports.values()
    }

    fn plan_tree(&self, query_id: Uuid) -> AnalyzerResult<PlanTree> {
        PlanTree::try_new(self.plans.values(), query_id)
    }
}

impl QueryEngineModelMut for SiriusModel {
    fn operator_mut(&mut self, operator_id: Uuid) -> AnalyzerResult<&mut Operator> {
        self.operators
            .get_mut(&operator_id)
            .ok_or(AnalyzerError::InvalidId(operator_id))
    }
}

impl SiriusModel {
    pub(crate) fn query_view(&self, query_id: Uuid) -> AnalyzerResult<SiriusModelQueryView<'_>> {
        SiriusModelQueryView::try_new(self, query_id)
    }

    pub(crate) fn resource_instance_name(&self, resource_id: Uuid) -> Option<&str> {
        self.memories
            .get(&resource_id)
            .map(Memory::instance_name)
            .or_else(|| self.channels.get(&resource_id).map(Channel::instance_name))
            .or_else(|| {
                self.memory_tiers
                    .get(&resource_id)
                    .map(MemoryTier::instance_name)
            })
            .or_else(|| {
                self.task_queues
                    .get(&resource_id)
                    .map(TaskQueue::instance_name)
            })
            .or_else(|| {
                self.task_manager_loop_threads
                    .get(&resource_id)
                    .map(TaskManagerLoopThread::instance_name)
            })
            .or_else(|| {
                self.executor_threads
                    .get(&resource_id)
                    .map(ExecutorThread::instance_name)
            })
    }

    pub(crate) fn resource_scope_instance_name(&self, entity_id: Uuid) -> Option<&str> {
        self.gpu_devices
            .get(&entity_id)
            .map(GpuDevice::instance_name)
            .or_else(|| {
                self.thread_groups
                    .get(&entity_id)
                    .map(ThreadGroup::instance_name)
            })
    }

    fn sirius_resource(&self, resource_id: Uuid) -> Option<&dyn Resource> {
        self.memories
            .get(&resource_id)
            .map(|resource| resource as &dyn Resource)
            .or_else(|| {
                self.channels
                    .get(&resource_id)
                    .map(|resource| resource as &dyn Resource)
            })
            .or_else(|| {
                self.memory_tiers
                    .get(&resource_id)
                    .map(|resource| resource as &dyn Resource)
            })
            .or_else(|| {
                self.task_queues
                    .get(&resource_id)
                    .map(|resource| resource as &dyn Resource)
            })
            .or_else(|| {
                self.task_manager_loop_threads
                    .get(&resource_id)
                    .map(|resource| resource as &dyn Resource)
            })
            .or_else(|| {
                self.executor_threads
                    .get(&resource_id)
                    .map(|resource| resource as &dyn Resource)
            })
    }

    fn sirius_resources(&self) -> impl Iterator<Item = &dyn Resource> {
        self.memories
            .values()
            .map(|resource| resource as &dyn Resource)
            .chain(
                self.channels
                    .values()
                    .map(|resource| resource as &dyn Resource),
            )
            .chain(
                self.memory_tiers
                    .values()
                    .map(|resource| resource as &dyn Resource),
            )
            .chain(
                self.task_queues
                    .values()
                    .map(|resource| resource as &dyn Resource),
            )
            .chain(
                self.task_manager_loop_threads
                    .values()
                    .map(|resource| resource as &dyn Resource),
            )
            .chain(
                self.executor_threads
                    .values()
                    .map(|resource| resource as &dyn Resource),
            )
    }

    fn add_resource_users<'a>(
        &mut self,
        entity_type_name: &str,
        usages: impl Iterator<Item = impl Usage<'a>>,
    ) -> AnalyzerResult<()> {
        for usage in usages {
            let resource_type_name = self
                .resource(usage.resource_id())
                .map(Entity::type_name)?
                .to_owned();
            self.resource_types
                .get_mut(&resource_type_name)
                .ok_or_else(|| AnalyzerError::InvalidTypeName(resource_type_name.clone()))?
                .used_by
                .insert(entity_type_name.to_owned());
        }
        Ok(())
    }
}

impl FsmCollection for SiriusModel {
    type Fsm = Task;

    fn fsms(&self) -> impl Iterator<Item = &Task> {
        self.tasks.values()
    }
}

impl RefTreeCollection for SiriusModel {
    fn ref_tree_entities(&self) -> impl Iterator<Item = &dyn RefTreeEntity> {
        std::iter::once(&self.engine as &dyn RefTreeEntity)
            .chain(
                self.workers
                    .values()
                    .map(|entity| entity as &dyn RefTreeEntity),
            )
            .chain(
                self.query_groups
                    .values()
                    .map(|entity| entity as &dyn RefTreeEntity),
            )
            .chain(
                self.queries
                    .values()
                    .map(|entity| entity as &dyn RefTreeEntity),
            )
            .chain(
                self.plans
                    .values()
                    .map(|entity| entity as &dyn RefTreeEntity),
            )
            .chain(
                self.operators
                    .values()
                    .map(|entity| entity as &dyn RefTreeEntity),
            )
            .chain(
                self.ports
                    .values()
                    .map(|entity| entity as &dyn RefTreeEntity),
            )
            .chain(
                self.gpu_devices
                    .values()
                    .map(|entity| entity as &dyn RefTreeEntity),
            )
            .chain(
                self.thread_groups
                    .values()
                    .map(|entity| entity as &dyn RefTreeEntity),
            )
            .chain(
                self.memories
                    .values()
                    .map(|entity| entity as &dyn RefTreeEntity),
            )
            .chain(
                self.channels
                    .values()
                    .map(|entity| entity as &dyn RefTreeEntity),
            )
            .chain(
                self.memory_tiers
                    .values()
                    .map(|entity| entity as &dyn RefTreeEntity),
            )
            .chain(
                self.task_queues
                    .values()
                    .map(|entity| entity as &dyn RefTreeEntity),
            )
            .chain(
                self.task_manager_loop_threads
                    .values()
                    .map(|entity| entity as &dyn RefTreeEntity),
            )
            .chain(
                self.executor_threads
                    .values()
                    .map(|entity| entity as &dyn RefTreeEntity),
            )
            .chain(
                self.tasks
                    .values()
                    .map(|entity| entity as &dyn RefTreeEntity),
            )
            .chain(
                self.data_batches
                    .values()
                    .map(|entity| entity as &dyn RefTreeEntity),
            )
            .chain(
                self.batch_placements
                    .values()
                    .map(|entity| entity as &dyn RefTreeEntity),
            )
    }

    fn ref_tree_entity(&self, entity_id: Uuid) -> AnalyzerResult<&dyn RefTreeEntity> {
        if self.engine.id() == entity_id {
            Ok(&self.engine)
        } else if let Some(entity) = self.workers.get(&entity_id) {
            Ok(entity)
        } else if let Some(entity) = self.query_groups.get(&entity_id) {
            Ok(entity)
        } else if let Some(entity) = self.queries.get(&entity_id) {
            Ok(entity)
        } else if let Some(entity) = self.plans.get(&entity_id) {
            Ok(entity)
        } else if let Some(entity) = self.operators.get(&entity_id) {
            Ok(entity)
        } else if let Some(entity) = self.ports.get(&entity_id) {
            Ok(entity)
        } else if let Some(entity) = self.gpu_devices.get(&entity_id) {
            Ok(entity)
        } else if let Some(entity) = self.thread_groups.get(&entity_id) {
            Ok(entity)
        } else if let Some(entity) = self.memories.get(&entity_id) {
            Ok(entity)
        } else if let Some(entity) = self.channels.get(&entity_id) {
            Ok(entity)
        } else if let Some(entity) = self.memory_tiers.get(&entity_id) {
            Ok(entity)
        } else if let Some(entity) = self.task_queues.get(&entity_id) {
            Ok(entity)
        } else if let Some(entity) = self.task_manager_loop_threads.get(&entity_id) {
            Ok(entity)
        } else if let Some(entity) = self.executor_threads.get(&entity_id) {
            Ok(entity)
        } else if let Some(entity) = self.tasks.get(&entity_id) {
            Ok(entity)
        } else if let Some(entity) = self.data_batches.get(&entity_id) {
            Ok(entity)
        } else if let Some(entity) = self.batch_placements.get(&entity_id) {
            Ok(entity)
        } else {
            Err(AnalyzerError::InvalidId(entity_id))
        }
    }
}

impl ResourceCollection for SiriusModel {
    fn resources(&self) -> impl Iterator<Item = &dyn Resource> {
        self.sirius_resources()
    }

    fn resource(&self, resource_id: Uuid) -> AnalyzerResult<&dyn Resource> {
        self.sirius_resource(resource_id)
            .ok_or(AnalyzerError::InvalidId(resource_id))
    }

    fn resource_type(&self, resource_type_name: &str) -> AnalyzerResult<&ResourceTypeDecl> {
        self.resource_types
            .get(resource_type_name)
            .ok_or_else(|| AnalyzerError::InvalidTypeName(resource_type_name.to_owned()))
    }
}

pub struct SiriusModelBuilder {
    engine_id: Uuid,
    engine: Option<Engine>,
    workers: HashMap<Uuid, Worker>,
    query_groups: HashMap<Uuid, QueryGroup>,
    queries: HashMap<Uuid, QueryBuilder>,
    plans: HashMap<Uuid, Plan>,
    operators: HashMap<Uuid, Operator>,
    ports: HashMap<Uuid, Port>,
    gpu_devices: HashMap<Uuid, GpuDevice>,
    thread_groups: HashMap<Uuid, ThreadGroup>,
    memories: HashMap<Uuid, Memory>,
    channels: HashMap<Uuid, Channel>,
    memory_tiers: HashMap<Uuid, MemoryTier>,
    task_queues: HashMap<Uuid, TaskQueue>,
    task_manager_loop_threads: HashMap<Uuid, TaskManagerLoopThread>,
    executor_threads: HashMap<Uuid, ExecutorThread>,
    tasks: HashMap<Uuid, TaskBuilder>,
    data_batches: HashMap<Uuid, DataBatchBuilder>,
    batch_placements: HashMap<Uuid, BatchPlacementBuilder>,
}

impl SiriusModelBuilder {
    pub(crate) fn try_new(engine_id: Uuid) -> AnalyzerResult<Self> {
        if engine_id.is_nil() {
            return Err(AnalyzerError::Validation(
                "engine id cannot be nil".to_owned(),
            ));
        }
        Ok(Self {
            engine_id,
            engine: None,
            workers: HashMap::default(),
            query_groups: HashMap::default(),
            queries: HashMap::default(),
            plans: HashMap::default(),
            operators: HashMap::default(),
            ports: HashMap::default(),
            gpu_devices: HashMap::default(),
            thread_groups: HashMap::default(),
            memories: HashMap::default(),
            channels: HashMap::default(),
            memory_tiers: HashMap::default(),
            task_queues: HashMap::default(),
            task_manager_loop_threads: HashMap::default(),
            executor_threads: HashMap::default(),
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

        let is_resource_declaration = matches!(
            &data,
            SiriusEvent::Memory(_)
                | SiriusEvent::Channel(_)
                | SiriusEvent::MemoryTier(_)
                | SiriusEvent::TaskQueue(_)
                | SiriusEvent::TaskManagerLoopThread(_)
                | SiriusEvent::ExecutorThread(_)
        );
        if is_resource_declaration && self.contains_resource(id) {
            return Err(AnalyzerError::Validation(format!(
                "resource {id} has multiple declarations"
            )));
        }

        match data {
            SiriusEvent::Engine(event) => {
                if id != self.engine_id {
                    return Err(AnalyzerError::Validation(format!(
                        "multiple engine instances in one model: expected {}, found {id}",
                        self.engine_id
                    )));
                }
                let event = Event::new(id, timestamp, event);
                if let Some(engine) = &mut self.engine {
                    engine.push(event)
                } else {
                    self.engine = Some(Engine::try_from_event(event)?);
                    Ok(())
                }
            }
            SiriusEvent::Worker(event) => {
                let event = Event::new(id, timestamp, event);
                if let Some(worker) = self.workers.get_mut(&id) {
                    worker.push(event)
                } else {
                    self.workers.insert(id, Worker::try_from_event(event)?);
                    Ok(())
                }
            }
            SiriusEvent::QueryGroup(event) => {
                let event = Event::new(id, timestamp, event);
                if let Some(group) = self.query_groups.get_mut(&id) {
                    group.push(event)
                } else {
                    self.query_groups
                        .insert(id, QueryGroup::try_from_event(event)?);
                    Ok(())
                }
            }
            SiriusEvent::Query(event) => {
                match self.queries.entry(id) {
                    Entry::Occupied(entry) => {
                        entry
                            .into_mut()
                            .push_transition(Event::new(id, timestamp, event));
                    }
                    Entry::Vacant(entry) => {
                        let mut builder = QueryBuilder::try_new(id)?;
                        builder.push_transition(Event::new(id, timestamp, event));
                        entry.insert(builder);
                    }
                }
                Ok(())
            }
            SiriusEvent::Plan(event) => {
                let event = Event::new(id, timestamp, event);
                if let Some(plan) = self.plans.get_mut(&id) {
                    plan.push(event)
                } else {
                    self.plans.insert(id, Plan::try_from_event(event)?);
                    Ok(())
                }
            }
            SiriusEvent::Operator(event) => {
                let event = Event::new(id, timestamp, event);
                if let Some(operator) = self.operators.get_mut(&id) {
                    operator.push(event)
                } else {
                    self.operators.insert(id, Operator::try_from_event(event)?);
                    Ok(())
                }
            }
            SiriusEvent::Port(event) => {
                let event = Event::new(id, timestamp, event);
                if let Some(port) = self.ports.get_mut(&id) {
                    port.push(event)
                } else {
                    self.ports.insert(id, Port::try_from_event(event)?);
                    Ok(())
                }
            }
            SiriusEvent::GpuDevice(event) => {
                let event = Event::new(id, timestamp, event);
                if let Some(entity) = self.gpu_devices.get_mut(&id) {
                    entity.push(event)
                } else {
                    self.gpu_devices
                        .insert(id, GpuDevice::try_from_event(event)?);
                    Ok(())
                }
            }
            SiriusEvent::ThreadGroup(event) => {
                let event = Event::new(id, timestamp, event);
                if let Some(entity) = self.thread_groups.get_mut(&id) {
                    entity.push(event)
                } else {
                    self.thread_groups
                        .insert(id, ThreadGroup::try_from_event(event)?);
                    Ok(())
                }
            }
            SiriusEvent::Memory(event) => {
                let event = Event::new(id, timestamp, event);
                if let Some(resource) = self.memories.get_mut(&id) {
                    resource.push(event)
                } else {
                    self.memories.insert(id, Memory::try_from_event(event)?);
                    Ok(())
                }
            }
            SiriusEvent::Channel(event) => {
                let event = Event::new(id, timestamp, event);
                if let Some(resource) = self.channels.get_mut(&id) {
                    resource.push(event)
                } else {
                    self.channels.insert(id, Channel::try_from_event(event)?);
                    Ok(())
                }
            }
            SiriusEvent::MemoryTier(event) => {
                let event = Event::new(id, timestamp, event);
                if let Some(resource) = self.memory_tiers.get_mut(&id) {
                    resource.push(event)
                } else {
                    self.memory_tiers
                        .insert(id, MemoryTier::try_from_event(event)?);
                    Ok(())
                }
            }
            SiriusEvent::TaskQueue(event) => {
                let event = Event::new(id, timestamp, event);
                if let Some(resource) = self.task_queues.get_mut(&id) {
                    resource.push(event)
                } else {
                    self.task_queues
                        .insert(id, TaskQueue::try_from_event(event)?);
                    Ok(())
                }
            }
            SiriusEvent::TaskManagerLoopThread(event) => {
                let event = Event::new(id, timestamp, event);
                if let Some(resource) = self.task_manager_loop_threads.get_mut(&id) {
                    resource.push(event)
                } else {
                    self.task_manager_loop_threads
                        .insert(id, TaskManagerLoopThread::try_from_event(event)?);
                    Ok(())
                }
            }
            SiriusEvent::ExecutorThread(event) => {
                let event = Event::new(id, timestamp, event);
                if let Some(resource) = self.executor_threads.get_mut(&id) {
                    resource.push(event)
                } else {
                    self.executor_threads
                        .insert(id, ExecutorThread::try_from_event(event)?);
                    Ok(())
                }
            }
            SiriusEvent::Task(event) => {
                let task_builder = match self.tasks.entry(id) {
                    Entry::Occupied(entry) => entry.into_mut(),
                    Entry::Vacant(entry) => entry.insert(TaskBuilder::try_new(id)?),
                };
                task_builder.push_transition(Event::new(id, timestamp, event));
                Ok(())
            }
            SiriusEvent::DataBatch(event) => {
                let data_batch_builder = match self.data_batches.entry(id) {
                    Entry::Occupied(entry) => entry.into_mut(),
                    Entry::Vacant(entry) => entry.insert(DataBatchBuilder::try_new(id)?),
                };
                data_batch_builder.push_transition(Event::new(id, timestamp, event));
                Ok(())
            }
            SiriusEvent::BatchPlacement(event) => {
                let batch_builder = match self.batch_placements.entry(id) {
                    Entry::Occupied(entry) => entry.into_mut(),
                    Entry::Vacant(entry) => entry.insert(BatchPlacementBuilder::try_new(id)?),
                };
                batch_builder.push_transition(Event::new(id, timestamp, event));
                Ok(())
            }
        }
    }

    fn contains_resource(&self, id: Uuid) -> bool {
        self.memories.contains_key(&id)
            || self.channels.contains_key(&id)
            || self.memory_tiers.contains_key(&id)
            || self.task_queues.contains_key(&id)
            || self.task_manager_loop_threads.contains_key(&id)
            || self.executor_threads.contains_key(&id)
    }

    pub(crate) fn try_build(self) -> AnalyzerResult<SiriusModel> {
        let engine = self.engine.ok_or_else(|| {
            AnalyzerError::IncompleteEntity(format!("engine {} has no events", self.engine_id))
        })?;
        let queries = self
            .queries
            .into_iter()
            .map(|(id, builder)| Query::try_from_builder(builder).map(|query| (id, query)))
            .collect::<AnalyzerResult<HashMap<_, _>>>()?;
        let resource_types = [
            Memory::resource_type_decl(),
            Channel::resource_type_decl(),
            MemoryTier::resource_type_decl(),
            TaskQueue::resource_type_decl(),
            TaskManagerLoopThread::resource_type_decl(),
            ExecutorThread::resource_type_decl(),
        ]
        .into_iter()
        .map(|declaration| (declaration.name.clone(), declaration))
        .collect();

        let mut model = SiriusModel {
            engine,
            workers: self.workers,
            query_groups: self.query_groups,
            queries,
            plans: self.plans,
            operators: self.operators,
            ports: self.ports,
            resource_types,
            gpu_devices: self.gpu_devices,
            thread_groups: self.thread_groups,
            memories: self.memories,
            channels: self.channels,
            memory_tiers: self.memory_tiers,
            task_queues: self.task_queues,
            task_manager_loop_threads: self.task_manager_loop_threads,
            executor_threads: self.executor_threads,
            tasks: HashMap::default(),
            data_batches: HashMap::default(),
            batch_placements: HashMap::default(),
            resource_group_types: HashMap::default(),
        };

        for (task_id, task_builder) in self.tasks {
            let task = Task::from_builder(task_builder)?;
            model.add_resource_users(task.type_name(), task.usages())?;
            if let Some(operator_id) = task.pipeline_uuid()
                && let Some(task_span) = task.active_span()
                && let Ok(operator) = model.operator_mut(operator_id)
            {
                operator.extend_active_span(task_span);
            }
            model.tasks.insert(task_id, task);
        }

        for (data_batch_id, data_batch_builder) in self.data_batches {
            match DataBatch::from_builder(data_batch_builder) {
                Ok(data_batch) => {
                    model.add_resource_users(data_batch.type_name(), data_batch.usages())?;
                    if let Some(operator_id) = data_batch.producer_pipeline_uuid()
                        && let Some(data_batch_span) = data_batch.active_span()
                        && let Ok(operator) = model.operator_mut(operator_id)
                    {
                        operator.extend_active_span(data_batch_span);
                    }
                    model.data_batches.insert(data_batch_id, data_batch);
                }
                Err(error) => warn!("Invalid data_batch encountered {error}"),
            }
        }

        for (batch_id, batch_builder) in self.batch_placements {
            let batch = BatchPlacement::from_builder(batch_builder)?;
            model.add_resource_users(batch.type_name(), batch.usages())?;
            model.batch_placements.insert(batch_id, batch);
        }

        model.resource_group_types = derive_resource_scope_types(&model)?;
        Ok(model)
    }
}
