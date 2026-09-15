// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Schema-event-backed query-engine entities for Sirius.

use quent_analyzer::{
    AnalyzerError, AnalyzerResult, Entity, Model, Span,
    entity::{EntityData, EntityEvents},
    fsm::{
        Fsm, FsmUsages,
        events::{AnalyzedTransition, FsmEvents, FsmEventsBuilder},
    },
    resource::{
        Resource, ResourceGroup, ResourceTypeDecl, Usage, Using, collection::ResourceCollection,
    },
};
use quent_dynamic_attributes::DynamicAttributes;
use quent_events::Event;
use quent_query_engine_analyzer::{
    EngineEntity, OperatorEntity, OperatorEntityMut, PlanEntity, PortEntity, QueryEngineModel,
    QueryEngineModelMut, QueryEntity, QueryGroupEntity, WorkerEntity, plan_tree::PlanTree,
};
use quent_query_engine_ui as ui;
use quent_time::{TimeUnixNanoSec, Timestamp, span::SpanUnixNanoSec, try_to_secs_relative};
use rustc_hash::FxHashMap as HashMap;
use sirius_telemetry_store as schema;
use uuid::Uuid;

#[derive(Debug)]
pub struct Engine(EntityEvents<EngineStorage>);

struct EngineStorage;

#[derive(Default)]
struct EngineData {
    instance_name: Option<String>,
    implementation: Option<ui::EngineImplementationAttributes>,
}

impl quent_events::Entity for EngineStorage {
    type Event = schema::EngineEvent;
}

impl EntityData for EngineStorage {
    type Data = EngineData;

    fn push(data: &mut Self::Data, event: Self::Event) {
        if let schema::EngineEvent::Init {
            implementation,
            instance_name,
        } = event
        {
            data.instance_name = instance_name;
            data.implementation = Some(ui::EngineImplementationAttributes {
                name: implementation.name,
                version: implementation.version,
                custom_attributes: implementation.custom_attributes.0,
            });
        }
    }
}

impl Engine {
    fn try_new(id: Uuid) -> AnalyzerResult<Self> {
        Ok(Self(EntityEvents::new(id)?))
    }

    fn push(&mut self, event: Event<schema::EngineEvent>) {
        self.0.push(event);
    }
}

impl Entity for Engine {
    fn id(&self) -> Uuid {
        self.0.id()
    }
    fn type_name(&self) -> &str {
        "engine"
    }
    fn instance_name(&self) -> &str {
        self.0.data().instance_name.as_deref().unwrap_or_default()
    }
}

impl Span for Engine {
    fn span(&self) -> AnalyzerResult<SpanUnixNanoSec> {
        match (self.0.earliest_timestamp(), self.0.latest_timestamp()) {
            (Some(start), Some(end)) => Ok(SpanUnixNanoSec::try_new(start, end)?),
            _ => Err(AnalyzerError::IncompleteEntity(
                "engine does not have both init and exit events".to_owned(),
            )),
        }
    }
}

impl ResourceGroup for Engine {
    fn parent_group_id(&self) -> Option<Uuid> {
        None
    }
}

impl EngineEntity for Engine {
    fn to_ui(&self) -> AnalyzerResult<ui::Engine> {
        let start = self.0.earliest_timestamp();
        let end = self.0.latest_timestamp();
        let duration_s = match (start, end) {
            (Some(start), Some(end)) => Some(try_to_secs_relative(end, start)?),
            _ => None,
        };
        Ok(ui::Engine {
            id: self.0.id(),
            start_time_unix_ns: start,
            duration_s,
            instance_name: self.0.data().instance_name.clone(),
            implementation: self.0.data().implementation.as_ref().map(|implementation| {
                ui::EngineImplementationAttributes {
                    name: implementation.name.clone(),
                    version: implementation.version.clone(),
                    custom_attributes: implementation.custom_attributes.clone(),
                }
            }),
        })
    }
}

#[derive(Debug)]
pub struct Worker(EntityEvents<WorkerStorage>);

struct WorkerStorage;

#[derive(Default)]
struct WorkerData {
    parent_engine_id: Option<Uuid>,
    instance_name: Option<String>,
}

impl quent_events::Entity for WorkerStorage {
    type Event = schema::WorkerEvent;
}

impl EntityData for WorkerStorage {
    type Data = WorkerData;

    fn push(data: &mut Self::Data, event: Self::Event) {
        if let schema::WorkerEvent::Init {
            parent_engine_id,
            instance_name,
        } = event
        {
            data.parent_engine_id = Some(parent_engine_id.target);
            data.instance_name = Some(instance_name);
        }
    }
}

impl Worker {
    fn try_new(id: Uuid) -> AnalyzerResult<Self> {
        Ok(Self(EntityEvents::new(id)?))
    }

    fn push(&mut self, event: Event<schema::WorkerEvent>) {
        self.0.push(event);
    }
}

impl Entity for Worker {
    fn id(&self) -> Uuid {
        self.0.id()
    }
    fn type_name(&self) -> &str {
        "worker"
    }
    fn instance_name(&self) -> &str {
        self.0.data().instance_name.as_deref().unwrap_or_default()
    }
}

impl Span for Worker {
    fn span(&self) -> AnalyzerResult<SpanUnixNanoSec> {
        match (self.0.earliest_timestamp(), self.0.latest_timestamp()) {
            (Some(start), Some(end)) => Ok(SpanUnixNanoSec::try_new(start, end)?),
            _ => Err(AnalyzerError::IncompleteEntity(
                "worker does not have both init and exit events".to_owned(),
            )),
        }
    }
}

impl ResourceGroup for Worker {
    fn parent_group_id(&self) -> Option<Uuid> {
        self.0.data().parent_engine_id
    }
}

impl WorkerEntity for Worker {
    fn to_ui(&self, _epoch: TimeUnixNanoSec) -> ui::Worker {
        ui::Worker {
            id: self.0.id(),
            parent_engine_id: self.0.data().parent_engine_id,
            instance_name: self.0.data().instance_name.clone(),
            start_unix_ns: self.0.earliest_timestamp(),
            end_unix_ns: self.0.latest_timestamp(),
        }
    }
}

#[derive(Debug)]
pub struct QueryGroup(EntityEvents<QueryGroupStorage>);

struct QueryGroupStorage;

#[derive(Default)]
struct QueryGroupData {
    engine_id: Option<Uuid>,
    instance_name: Option<String>,
}

impl quent_events::Entity for QueryGroupStorage {
    type Event = schema::QueryGroupEvent;
}

impl EntityData for QueryGroupStorage {
    type Data = QueryGroupData;

    fn push(data: &mut Self::Data, event: Self::Event) {
        let schema::QueryGroupEvent::Declaration {
            instance_name,
            engine_id,
        } = event;
        data.instance_name = Some(instance_name);
        data.engine_id = Some(engine_id.target);
    }
}

impl QueryGroup {
    fn try_new(id: Uuid) -> AnalyzerResult<Self> {
        Ok(Self(EntityEvents::new(id)?))
    }

    fn push(&mut self, event: Event<schema::QueryGroupEvent>) {
        self.0.push(event);
    }
}

impl Entity for QueryGroup {
    fn id(&self) -> Uuid {
        self.0.id()
    }
    fn type_name(&self) -> &str {
        "query group"
    }
    fn instance_name(&self) -> &str {
        self.0.data().instance_name.as_deref().unwrap_or_default()
    }
}

impl ResourceGroup for QueryGroup {
    fn parent_group_id(&self) -> Option<Uuid> {
        self.0.data().engine_id
    }
}

impl QueryGroupEntity for QueryGroup {
    fn to_ui(&self) -> ui::QueryGroup {
        ui::QueryGroup {
            id: self.0.id(),
            instance_name: self.0.data().instance_name.clone(),
            engine_id: self.0.data().engine_id,
        }
    }
}

pub type QueryBuilder = FsmEventsBuilder<schema::QueryEvent>;

#[derive(Debug)]
pub struct Query(FsmEvents<schema::QueryEvent>);

impl Query {
    fn from_builder(builder: QueryBuilder) -> AnalyzerResult<Self> {
        Ok(Self(builder.try_build()?))
    }
}

impl QueryEntity for Query {
    fn query_group_id(&self) -> Option<Uuid> {
        match self.0.first_data()? {
            schema::QueryEvent::Init { query_group_id, .. } => Some(query_group_id.target),
            _ => None,
        }
    }

    fn to_ui(&self) -> AnalyzerResult<ui::Query> {
        let transitions = self.0.transitions();
        let epoch = transitions.first().map(Timestamp::timestamp);
        let mut planning_s = None;
        let mut executing_s = None;
        let mut completed_s = None;

        if let Some(epoch) = epoch {
            for (index, transition) in transitions.iter().enumerate() {
                match transition.data {
                    schema::QueryEvent::Planning { .. } => {
                        planning_s = Some(try_to_secs_relative(transition.timestamp(), epoch)?);
                    }
                    schema::QueryEvent::Executing { .. } => {
                        executing_s = Some(try_to_secs_relative(transition.timestamp(), epoch)?);
                        if let Some(next) = transitions.get(index + 1) {
                            completed_s = Some(try_to_secs_relative(next.timestamp(), epoch)?);
                        }
                    }
                    _ => {}
                }
            }
        }

        Ok(ui::Query {
            id: self.id(),
            query_group_id: self.query_group_id().unwrap_or_default(),
            instance_name: Some(self.instance_name().to_owned()).filter(|name| !name.is_empty()),
            start_unix_ns: epoch,
            planning_s,
            executing_s,
            completed_s,
        })
    }
}

impl Entity for Query {
    fn id(&self) -> Uuid {
        self.0.id()
    }
    fn type_name(&self) -> &str {
        self.0.type_name()
    }
    fn instance_name(&self) -> &str {
        self.0.instance_name()
    }
}

impl Fsm for Query {
    type TransitionType = AnalyzedTransition<schema::QueryEvent>;
    fn len(&self) -> usize {
        self.0.len()
    }
    fn transition(&self, index: usize) -> Option<&Self::TransitionType> {
        self.0.transition(index)
    }
}

impl<'a> FsmUsages<'a> for Query {
    fn usages_with_state_names(&'a self) -> impl Iterator<Item = (&'a str, impl Usage<'a>)> {
        self.0.usages_with_state_names()
    }
}

impl Using for Query {
    fn usages(&self) -> impl Iterator<Item = impl Usage<'_>> {
        self.0.usages()
    }
}

impl ResourceGroup for Query {
    fn parent_group_id(&self) -> Option<Uuid> {
        self.query_group_id()
    }
}

#[derive(Debug)]
pub struct Plan(EntityEvents<PlanStorage>);

struct PlanStorage;

#[derive(Default)]
struct PlanData {
    instance_name: Option<String>,
    parent_query_id: Option<Uuid>,
    parent_plan_id: Option<Uuid>,
    worker_id: Option<Uuid>,
    edges: Vec<(Uuid, Uuid)>,
}

impl quent_events::Entity for PlanStorage {
    type Event = schema::PlanEvent;
}

impl EntityData for PlanStorage {
    type Data = PlanData;

    fn push(data: &mut Self::Data, event: Self::Event) {
        let schema::PlanEvent::Declaration {
            parent,
            instance_name,
            edges,
            worker_id,
        } = event;
        data.instance_name = Some(instance_name);
        data.parent_query_id = Some(parent.query_id.target);
        data.parent_plan_id = parent.plan_id.map(|plan| plan.target);
        data.worker_id = worker_id.map(|worker| worker.target);
        data.edges = edges
            .into_iter()
            .map(|edge| (edge.source.target, edge.target.target))
            .collect();
    }
}

impl Plan {
    fn try_new(id: Uuid) -> AnalyzerResult<Self> {
        Ok(Self(EntityEvents::new(id)?))
    }

    fn push(&mut self, event: Event<schema::PlanEvent>) {
        self.0.push(event);
    }
}

impl Entity for Plan {
    fn id(&self) -> Uuid {
        self.0.id()
    }
    fn type_name(&self) -> &str {
        "plan"
    }
    fn instance_name(&self) -> &str {
        self.0.data().instance_name.as_deref().unwrap_or_default()
    }
}

impl ResourceGroup for Plan {
    fn parent_group_id(&self) -> Option<Uuid> {
        let data = self.0.data();
        data.worker_id
            .or(data.parent_plan_id)
            .or(data.parent_query_id)
    }
}

impl PlanEntity for Plan {
    fn parent_query_id(&self) -> Option<Uuid> {
        let data = self.0.data();
        data.parent_plan_id
            .is_none()
            .then_some(data.parent_query_id)
            .flatten()
    }
    fn parent_plan_id(&self) -> Option<Uuid> {
        self.0.data().parent_plan_id
    }
    fn worker_id(&self) -> Option<Uuid> {
        self.0.data().worker_id
    }
    fn edges(&self) -> impl Iterator<Item = (Uuid, Uuid)> + '_ {
        self.0.data().edges.iter().copied()
    }
    fn to_ui(&self) -> ui::Plan {
        ui::Plan {
            id: self.0.id(),
            instance_name: self.0.data().instance_name.clone(),
            parent: self
                .0
                .data()
                .parent_plan_id
                .or(self.0.data().parent_query_id),
            worker_id: self.0.data().worker_id,
            edges: self
                .0
                .data()
                .edges
                .iter()
                .map(|&(source, target)| ui::Edge { source, target })
                .collect(),
        }
    }
}

#[derive(Debug)]
pub struct Operator {
    inner: EntityEvents<OperatorStorage>,
    active_span: Option<SpanUnixNanoSec>,
}

struct OperatorStorage;

#[derive(Default)]
struct OperatorData {
    plan_id: Option<Uuid>,
    parent_operator_ids: Vec<Uuid>,
    instance_name: Option<String>,
    operator_type_name: Option<String>,
    custom_attributes: DynamicAttributes,
    statistics: Option<DynamicAttributes>,
}

impl quent_events::Entity for OperatorStorage {
    type Event = schema::OperatorEvent;
}

impl EntityData for OperatorStorage {
    type Data = OperatorData;

    fn push(data: &mut Self::Data, event: Self::Event) {
        match event {
            schema::OperatorEvent::Declaration {
                plan_id,
                parent_operator_ids,
                instance_name,
                type_name,
                custom_attributes,
            } => {
                data.plan_id = Some(plan_id.target);
                data.parent_operator_ids = parent_operator_ids
                    .into_iter()
                    .map(|operator| operator.target)
                    .collect();
                data.instance_name = Some(instance_name);
                data.operator_type_name = Some(type_name);
                data.custom_attributes = custom_attributes;
            }
            schema::OperatorEvent::Statistics { custom_attributes } => {
                data.statistics = Some(custom_attributes);
            }
        }
    }
}

impl Operator {
    fn try_new(id: Uuid) -> AnalyzerResult<Self> {
        Ok(Self {
            inner: EntityEvents::new(id)?,
            active_span: None,
        })
    }

    fn push(&mut self, event: Event<schema::OperatorEvent>) {
        self.inner.push(event);
    }
}

impl Entity for Operator {
    fn id(&self) -> Uuid {
        self.inner.id()
    }
    fn type_name(&self) -> &str {
        "operator"
    }
    fn instance_name(&self) -> &str {
        self.inner
            .data()
            .instance_name
            .as_deref()
            .unwrap_or_default()
    }
}

impl ResourceGroup for Operator {
    fn parent_group_id(&self) -> Option<Uuid> {
        self.inner.data().plan_id
    }
}

impl OperatorEntity for Operator {
    fn plan_id(&self) -> Option<Uuid> {
        self.inner.data().plan_id
    }
    fn parent_operator_ids(&self) -> impl ExactSizeIterator<Item = Uuid> + '_ {
        self.inner.data().parent_operator_ids.iter().copied()
    }
    fn active_span(&self) -> Option<SpanUnixNanoSec> {
        self.active_span
    }
    fn operator_type_name(&self) -> Option<&str> {
        self.inner.data().operator_type_name.as_deref()
    }
    fn to_ui(&self, epoch: TimeUnixNanoSec) -> ui::Operator {
        ui::Operator {
            id: self.inner.id(),
            plan_id: self.inner.data().plan_id,
            parent_operator_ids: self.inner.data().parent_operator_ids.clone(),
            instance_name: self.inner.data().instance_name.clone(),
            operator_type_name: self.inner.data().operator_type_name.clone(),
            custom_attributes: self
                .inner
                .data()
                .custom_attributes
                .iter()
                .map(|attribute| (attribute.key.clone(), attribute.value.clone()))
                .collect(),
            statistics: self.inner.data().statistics.as_ref().map(|statistics| {
                ui::OperatorStatistics {
                    custom_statistics: statistics
                        .iter()
                        .map(|attribute| {
                            (
                                attribute.key.clone(),
                                ui::OperatorStatistic {
                                    value: attribute.value.clone(),
                                    quantity: None,
                                },
                            )
                        })
                        .collect(),
                }
            }),
            active_span: self
                .active_span
                .and_then(|span| span.try_to_secs_relative(epoch).ok()),
        }
    }
}

impl OperatorEntityMut for Operator {
    fn extend_active_span(&mut self, span: SpanUnixNanoSec) {
        self.active_span = Some(match self.active_span {
            Some(existing) => existing.extend(&span),
            None => span,
        });
    }
}

#[derive(Debug)]
pub struct Port(EntityEvents<PortStorage>);

struct PortStorage;

#[derive(Default)]
struct PortData {
    operator_id: Option<Uuid>,
    instance_name: Option<String>,
    statistics: Option<DynamicAttributes>,
}

impl quent_events::Entity for PortStorage {
    type Event = schema::PortEvent;
}

impl EntityData for PortStorage {
    type Data = PortData;

    fn push(data: &mut Self::Data, event: Self::Event) {
        match event {
            schema::PortEvent::Declaration {
                operator_id,
                instance_name,
            } => {
                data.operator_id = Some(operator_id.target);
                data.instance_name = Some(instance_name);
            }
            schema::PortEvent::Statistics { custom_attributes } => {
                data.statistics = Some(custom_attributes);
            }
        }
    }
}

impl Port {
    fn try_new(id: Uuid) -> AnalyzerResult<Self> {
        Ok(Self(EntityEvents::new(id)?))
    }

    fn push(&mut self, event: Event<schema::PortEvent>) {
        self.0.push(event);
    }
}

impl Entity for Port {
    fn id(&self) -> Uuid {
        self.0.id()
    }
    fn type_name(&self) -> &str {
        "port"
    }
    fn instance_name(&self) -> &str {
        self.0.data().instance_name.as_deref().unwrap_or_default()
    }
}

impl ResourceGroup for Port {
    fn parent_group_id(&self) -> Option<Uuid> {
        self.0.data().operator_id
    }
}

impl PortEntity for Port {
    fn operator_id(&self) -> Option<Uuid> {
        self.0.data().operator_id
    }
    fn to_ui(&self, _epoch: TimeUnixNanoSec) -> ui::Port {
        ui::Port {
            id: self.0.id(),
            operator_id: self.0.data().operator_id,
            instance_name: self.0.data().instance_name.clone(),
            statistics: self
                .0
                .data()
                .statistics
                .as_ref()
                .map(|statistics| ui::PortStatistics {
                    custom_statistics: statistics
                        .iter()
                        .map(|attribute| (attribute.key.clone(), attribute.value.clone()))
                        .collect(),
                }),
        }
    }
}

#[derive(Debug)]
pub struct QueryEngine {
    pub engine: Engine,
    pub workers: HashMap<Uuid, Worker>,
    pub query_groups: HashMap<Uuid, QueryGroup>,
    pub queries: HashMap<Uuid, Query>,
    pub plans: HashMap<Uuid, Plan>,
    pub operators: HashMap<Uuid, Operator>,
    pub ports: HashMap<Uuid, Port>,
}

impl Model for QueryEngine {
    type EntityIdType = quent_query_engine_analyzer::QueryEngineEntityId;

    fn try_entity_ref(&self, id: Uuid) -> AnalyzerResult<Self::EntityIdType> {
        use quent_query_engine_analyzer::QueryEngineEntityId;

        if self.engine.id() == id {
            Ok(QueryEngineEntityId::Engine(id))
        } else if self.workers.contains_key(&id) {
            Ok(QueryEngineEntityId::Worker(id))
        } else if self.query_groups.contains_key(&id) {
            Ok(QueryEngineEntityId::QueryGroup(id))
        } else if self.queries.contains_key(&id) {
            Ok(QueryEngineEntityId::Query(id))
        } else if self.plans.contains_key(&id) {
            Ok(QueryEngineEntityId::Plan(id))
        } else if self.operators.contains_key(&id) {
            Ok(QueryEngineEntityId::Operator(id))
        } else if self.ports.contains_key(&id) {
            Ok(QueryEngineEntityId::Port(id))
        } else {
            Err(AnalyzerError::InvalidId(id))
        }
    }

    fn root(&self) -> AnalyzerResult<&impl ResourceGroup> {
        Ok(&self.engine)
    }
}

impl QueryEngineModel for QueryEngine {
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
    fn query(&self, id: Uuid) -> AnalyzerResult<&Query> {
        self.queries.get(&id).ok_or(AnalyzerError::InvalidId(id))
    }
    fn query_group(&self, id: Uuid) -> AnalyzerResult<&QueryGroup> {
        self.query_groups
            .get(&id)
            .ok_or(AnalyzerError::InvalidId(id))
    }
    fn worker(&self, id: Uuid) -> AnalyzerResult<&Worker> {
        self.workers.get(&id).ok_or(AnalyzerError::InvalidId(id))
    }
    fn plan(&self, id: Uuid) -> AnalyzerResult<&Plan> {
        self.plans.get(&id).ok_or(AnalyzerError::InvalidId(id))
    }
    fn operator(&self, id: Uuid) -> AnalyzerResult<&Operator> {
        self.operators.get(&id).ok_or(AnalyzerError::InvalidId(id))
    }
    fn port(&self, id: Uuid) -> AnalyzerResult<&Port> {
        self.ports.get(&id).ok_or(AnalyzerError::InvalidId(id))
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

impl QueryEngineModelMut for QueryEngine {
    fn operator_mut(&mut self, id: Uuid) -> AnalyzerResult<&mut Operator> {
        self.operators
            .get_mut(&id)
            .ok_or(AnalyzerError::InvalidId(id))
    }
}

impl ResourceCollection for QueryEngine {
    fn resources(&self) -> impl Iterator<Item = &dyn Resource> {
        std::iter::empty()
    }

    fn resource_groups(&self) -> impl Iterator<Item = &dyn ResourceGroup> {
        std::iter::once(&self.engine as &dyn ResourceGroup)
            .chain(
                self.workers
                    .values()
                    .map(|entity| entity as &dyn ResourceGroup),
            )
            .chain(
                self.query_groups
                    .values()
                    .map(|entity| entity as &dyn ResourceGroup),
            )
            .chain(
                self.queries
                    .values()
                    .map(|entity| entity as &dyn ResourceGroup),
            )
            .chain(
                self.plans
                    .values()
                    .map(|entity| entity as &dyn ResourceGroup),
            )
            .chain(
                self.operators
                    .values()
                    .map(|entity| entity as &dyn ResourceGroup),
            )
            .chain(
                self.ports
                    .values()
                    .map(|entity| entity as &dyn ResourceGroup),
            )
    }

    fn resource(&self, id: Uuid) -> AnalyzerResult<&dyn Resource> {
        Err(AnalyzerError::InvalidId(id))
    }

    fn resource_type(&self, name: &str) -> AnalyzerResult<&ResourceTypeDecl> {
        Err(AnalyzerError::InvalidTypeName(name.to_owned()))
    }

    fn resource_group(&self, id: Uuid) -> AnalyzerResult<&dyn ResourceGroup> {
        use quent_query_engine_analyzer::QueryEngineEntityId;
        match self.try_entity_ref(id)? {
            QueryEngineEntityId::Engine(_) => Ok(&self.engine),
            QueryEngineEntityId::Worker(_) => Ok(self.workers.get(&id).unwrap()),
            QueryEngineEntityId::QueryGroup(_) => Ok(self.query_groups.get(&id).unwrap()),
            QueryEngineEntityId::Query(_) => Ok(self.queries.get(&id).unwrap()),
            QueryEngineEntityId::Plan(_) => Ok(self.plans.get(&id).unwrap()),
            QueryEngineEntityId::Operator(_) => Ok(self.operators.get(&id).unwrap()),
            QueryEngineEntityId::Port(_) => Ok(self.ports.get(&id).unwrap()),
        }
    }

    fn resource_group_child_groups(&self, id: Uuid) -> AnalyzerResult<impl Iterator<Item = Uuid>> {
        self.resource_group(id)?;
        Ok(self
            .resource_groups()
            .filter_map(move |group| (group.parent_group_id() == Some(id)).then_some(group.id())))
    }

    fn resource_group_child_resources(
        &self,
        id: Uuid,
    ) -> AnalyzerResult<impl Iterator<Item = Uuid>> {
        self.resource_group(id)?;
        Ok(std::iter::empty())
    }
}

pub struct QueryEngineBuilder {
    engine: Engine,
    workers: HashMap<Uuid, Worker>,
    query_groups: HashMap<Uuid, QueryGroup>,
    queries: HashMap<Uuid, QueryBuilder>,
    plans: HashMap<Uuid, Plan>,
    operators: HashMap<Uuid, Operator>,
    ports: HashMap<Uuid, Port>,
}

impl QueryEngineBuilder {
    pub fn try_new(engine_id: Uuid) -> AnalyzerResult<Self> {
        Ok(Self {
            engine: Engine::try_new(engine_id)?,
            workers: HashMap::default(),
            query_groups: HashMap::default(),
            queries: HashMap::default(),
            plans: HashMap::default(),
            operators: HashMap::default(),
            ports: HashMap::default(),
        })
    }

    pub fn push_engine(&mut self, event: Event<schema::EngineEvent>) -> AnalyzerResult<()> {
        if event.id != self.engine.id() {
            return Err(AnalyzerError::Validation(format!(
                "multiple engine instances in one model: expected {}, found {}",
                self.engine.id(),
                event.id
            )));
        }
        self.engine.push(event);
        Ok(())
    }

    pub fn push_worker(&mut self, event: Event<schema::WorkerEvent>) -> AnalyzerResult<()> {
        let worker = match self.workers.entry(event.id) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(Worker::try_new(event.id)?)
            }
        };
        worker.push(event);
        Ok(())
    }

    pub fn push_query_group(
        &mut self,
        event: Event<schema::QueryGroupEvent>,
    ) -> AnalyzerResult<()> {
        let group = match self.query_groups.entry(event.id) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(QueryGroup::try_new(event.id)?)
            }
        };
        group.push(event);
        Ok(())
    }

    pub fn push_query(&mut self, event: Event<schema::QueryEvent>) -> AnalyzerResult<()> {
        let query = match self.queries.entry(event.id) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(QueryBuilder::try_new(event.id)?)
            }
        };
        query.push_transition(event);
        Ok(())
    }

    pub fn push_plan(&mut self, event: Event<schema::PlanEvent>) -> AnalyzerResult<()> {
        let plan = match self.plans.entry(event.id) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(Plan::try_new(event.id)?)
            }
        };
        plan.push(event);
        Ok(())
    }

    pub fn push_operator(&mut self, event: Event<schema::OperatorEvent>) -> AnalyzerResult<()> {
        let operator = match self.operators.entry(event.id) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(Operator::try_new(event.id)?)
            }
        };
        operator.push(event);
        Ok(())
    }

    pub fn push_port(&mut self, event: Event<schema::PortEvent>) -> AnalyzerResult<()> {
        let port = match self.ports.entry(event.id) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(Port::try_new(event.id)?)
            }
        };
        port.push(event);
        Ok(())
    }

    pub fn try_build(self) -> AnalyzerResult<QueryEngine> {
        let queries = self
            .queries
            .into_iter()
            .map(|(id, builder)| Ok((id, Query::from_builder(builder)?)))
            .collect::<AnalyzerResult<_>>()?;
        Ok(QueryEngine {
            engine: self.engine,
            workers: self.workers,
            query_groups: self.query_groups,
            queries,
            plans: self.plans,
            operators: self.operators,
            ports: self.ports,
        })
    }
}

pub struct QueryEngineView<'a> {
    engine: &'a Engine,
    query_group: &'a QueryGroup,
    query: &'a Query,
    workers: HashMap<Uuid, &'a Worker>,
    plans: HashMap<Uuid, &'a Plan>,
    operators: HashMap<Uuid, &'a Operator>,
    ports: HashMap<Uuid, &'a Port>,
}

impl<'a> QueryEngineView<'a> {
    pub fn try_new(model: &'a QueryEngine, query_id: Uuid) -> AnalyzerResult<Self> {
        let query = model.query(query_id)?;
        let query_group = model.query_group(query.query_group_id().unwrap_or_default())?;
        let workers: HashMap<Uuid, &Worker> = model
            .query_workers(query_id)?
            .map(|entity| (entity.id(), entity))
            .collect();
        let plans: HashMap<Uuid, &Plan> = model
            .query_plans(query_id)?
            .map(|entity| (entity.id(), entity))
            .collect();
        let operators: HashMap<Uuid, &Operator> = model
            .plans_operators(plans.values().copied())?
            .map(|entity| (entity.id(), entity))
            .collect();
        let ports: HashMap<Uuid, &Port> = model
            .operators_ports(operators.values().copied())?
            .map(|entity| (entity.id(), entity))
            .collect();
        Ok(Self {
            engine: &model.engine,
            query_group,
            query,
            workers,
            plans,
            operators,
            ports,
        })
    }
}

impl Model for QueryEngineView<'_> {
    type EntityIdType = quent_query_engine_analyzer::QueryEngineEntityId;

    fn try_entity_ref(&self, id: Uuid) -> AnalyzerResult<Self::EntityIdType> {
        use quent_query_engine_analyzer::QueryEngineEntityId;
        if self.engine.id() == id {
            Ok(QueryEngineEntityId::Engine(id))
        } else if self.workers.contains_key(&id) {
            Ok(QueryEngineEntityId::Worker(id))
        } else if self.query_group.id() == id {
            Ok(QueryEngineEntityId::QueryGroup(id))
        } else if self.query.id() == id {
            Ok(QueryEngineEntityId::Query(id))
        } else if self.plans.contains_key(&id) {
            Ok(QueryEngineEntityId::Plan(id))
        } else if self.operators.contains_key(&id) {
            Ok(QueryEngineEntityId::Operator(id))
        } else if self.ports.contains_key(&id) {
            Ok(QueryEngineEntityId::Port(id))
        } else {
            Err(AnalyzerError::InvalidId(id))
        }
    }

    fn root(&self) -> AnalyzerResult<&impl ResourceGroup> {
        Ok(self.engine)
    }
}

impl QueryEngineModel for QueryEngineView<'_> {
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
    fn query(&self, id: Uuid) -> AnalyzerResult<&Query> {
        (self.query.id() == id)
            .then_some(self.query)
            .ok_or(AnalyzerError::InvalidId(id))
    }
    fn query_group(&self, id: Uuid) -> AnalyzerResult<&QueryGroup> {
        (self.query_group.id() == id)
            .then_some(self.query_group)
            .ok_or(AnalyzerError::InvalidId(id))
    }
    fn worker(&self, id: Uuid) -> AnalyzerResult<&Worker> {
        self.workers
            .get(&id)
            .copied()
            .ok_or(AnalyzerError::InvalidId(id))
    }
    fn plan(&self, id: Uuid) -> AnalyzerResult<&Plan> {
        self.plans
            .get(&id)
            .copied()
            .ok_or(AnalyzerError::InvalidId(id))
    }
    fn operator(&self, id: Uuid) -> AnalyzerResult<&Operator> {
        self.operators
            .get(&id)
            .copied()
            .ok_or(AnalyzerError::InvalidId(id))
    }
    fn port(&self, id: Uuid) -> AnalyzerResult<&Port> {
        self.ports
            .get(&id)
            .copied()
            .ok_or(AnalyzerError::InvalidId(id))
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

impl ResourceCollection for QueryEngineView<'_> {
    fn resources(&self) -> impl Iterator<Item = &dyn Resource> {
        std::iter::empty()
    }
    fn resource_groups(&self) -> impl Iterator<Item = &dyn ResourceGroup> {
        std::iter::once(self.engine as &dyn ResourceGroup)
            .chain(std::iter::once(self.query_group as &dyn ResourceGroup))
            .chain(std::iter::once(self.query as &dyn ResourceGroup))
            .chain(
                self.workers
                    .values()
                    .map(|entity| *entity as &dyn ResourceGroup),
            )
            .chain(
                self.plans
                    .values()
                    .map(|entity| *entity as &dyn ResourceGroup),
            )
            .chain(
                self.operators
                    .values()
                    .map(|entity| *entity as &dyn ResourceGroup),
            )
            .chain(
                self.ports
                    .values()
                    .map(|entity| *entity as &dyn ResourceGroup),
            )
    }
    fn resource(&self, id: Uuid) -> AnalyzerResult<&dyn Resource> {
        Err(AnalyzerError::InvalidId(id))
    }
    fn resource_type(&self, name: &str) -> AnalyzerResult<&ResourceTypeDecl> {
        Err(AnalyzerError::InvalidTypeName(name.to_owned()))
    }
    fn resource_group(&self, id: Uuid) -> AnalyzerResult<&dyn ResourceGroup> {
        use quent_query_engine_analyzer::QueryEngineEntityId;
        match self.try_entity_ref(id)? {
            QueryEngineEntityId::Engine(_) => Ok(self.engine),
            QueryEngineEntityId::Worker(_) => Ok(*self.workers.get(&id).unwrap()),
            QueryEngineEntityId::QueryGroup(_) => Ok(self.query_group),
            QueryEngineEntityId::Query(_) => Ok(self.query),
            QueryEngineEntityId::Plan(_) => Ok(*self.plans.get(&id).unwrap()),
            QueryEngineEntityId::Operator(_) => Ok(*self.operators.get(&id).unwrap()),
            QueryEngineEntityId::Port(_) => Ok(*self.ports.get(&id).unwrap()),
        }
    }
    fn resource_group_child_groups(&self, id: Uuid) -> AnalyzerResult<impl Iterator<Item = Uuid>> {
        self.resource_group(id)?;
        Ok(self
            .resource_groups()
            .filter_map(move |group| (group.parent_group_id() == Some(id)).then_some(group.id())))
    }
    fn resource_group_child_resources(
        &self,
        id: Uuid,
    ) -> AnalyzerResult<impl Iterator<Item = Uuid>> {
        self.resource_group(id)?;
        Ok(std::iter::empty())
    }
}
