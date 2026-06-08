use quent_analyzer::{
    AnalyzerError, AnalyzerResult, Entity, Model,
    resource::{
        Resource, ResourceGroup, ResourceGroupTypeDecl, ResourceTypeDecl, Usage, Using,
        collection::ResourceCollection,
    },
};
use quent_query_engine_analyzer::{
    QueryEngineModel,
    engine::Engine,
    model::QueryEngineEntityId,
    operator::Operator,
    plan::{Plan, tree::PlanTree},
    port::Port,
    query::Query,
    query_group::QueryGroup,
    view::InMemoryQueryEngineModelView,
    worker::Worker,
};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use sirius_telemetry_ui::EntityRef;
use uuid::Uuid;

use crate::{
    model::SiriusModel,
    task::{Task, TaskExt},
};

pub(crate) struct SiriusModelQueryView<'a> {
    resource_types: HashMap<String, &'a ResourceTypeDecl>,
    resource_group_types: HashMap<String, &'a ResourceGroupTypeDecl>,
    query_engine: InMemoryQueryEngineModelView<'a>,
    resources: HashMap<Uuid, &'a quent_analyzer::resource::runtime::RtResource>,
    resource_groups: HashMap<Uuid, &'a quent_analyzer::resource::runtime::RtResourceGroup>,
    tasks: HashMap<Uuid, &'a Task>,
}

impl<'a> SiriusModelQueryView<'a> {
    pub(crate) fn try_new(
        model: &'a SiriusModel,
        query_id: Uuid,
    ) -> AnalyzerResult<SiriusModelQueryView<'a>> {
        let query_engine = InMemoryQueryEngineModelView::try_new(&model.query_engine, query_id)?;
        let pipeline_ids: HashSet<Uuid> = query_engine.operators.keys().copied().collect();

        // QE scoped to single query
        let query_engine_view =
            InMemoryQueryEngineModelView::try_new(&model.query_engine, query_id)?;

        // Only keep arbitrary groups that reference one of the QE model groups
        let resource_groups = model
            .arbitrary_resources
            .resource_groups
            .iter()
            .filter(|(_, v)| {
                v.parent_group_id
                    .and_then(|parent| query_engine_view.resource_group(parent).ok())
                    .is_some()
            })
            .map(|(k, v)| (*k, v))
            .collect::<HashMap<_, _>>();

        let resources = model
            .arbitrary_resources
            .resources
            .iter()
            .filter(|(_, resource)| {
                // This needs to reference a QE resource group:
                let in_qe = query_engine_view
                    .resource_group(resource.parent_group_id())
                    .is_ok();
                // Or an arbitrary resource group.
                let in_sirius = resource_groups.contains_key(&resource.parent_group_id());
                in_qe || in_sirius
            })
            .map(|(k, v)| (*k, v))
            .collect::<HashMap<_, _>>();

        let mut result = Self {
            resource_types: model
                .arbitrary_resources
                .resource_types
                .iter()
                .map(|(name, resource_type)| (name.clone(), resource_type))
                .collect(),
            resource_group_types: model
                .resource_group_types
                .iter()
                .map(|(name, resource_group_type)| (name.clone(), resource_group_type))
                .collect(),
            query_engine,
            resources,
            resource_groups,
            tasks: HashMap::default(),
        };

        result.tasks = model
            .tasks
            .values()
            .map(|task| (task.id(), task))
            .filter(|(_, task)| {
                task.usages()
                    .any(|usage| result.resource(usage.resource_id()).is_ok())
            })
            .collect();

        Ok(result)
    }

    pub(crate) fn tasks(&self) -> impl Iterator<Item = &'a Task> + '_ {
        self.tasks.values().copied()
    }

    pub(crate) fn resource_types(
        &self,
    ) -> impl Iterator<Item = (&String, &'a ResourceTypeDecl)> + '_ {
        self.resource_types
            .iter()
            .map(|(name, resource_type)| (name, *resource_type))
    }

    pub(crate) fn resource_group_types(
        &self,
    ) -> impl Iterator<Item = (&String, &'a ResourceGroupTypeDecl)> + '_ {
        self.resource_group_types
            .iter()
            .map(|(name, resource_group_type)| (name, *resource_group_type))
    }
}

impl<'a> Model for SiriusModelQueryView<'a> {
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

impl<'a> ResourceCollection for SiriusModelQueryView<'a> {
    fn resources(&self) -> impl Iterator<Item = &dyn Resource> {
        self.resources
            .values()
            .map(|resource| *resource as &dyn Resource)
    }

    fn resource_groups(&self) -> impl Iterator<Item = &dyn ResourceGroup> {
        self.query_engine.resource_groups().chain(
            self.resource_groups
                .values()
                .map(|group| *group as &dyn ResourceGroup),
        )
    }

    fn resource(&self, resource_id: Uuid) -> AnalyzerResult<&dyn Resource> {
        self.resources
            .get(&resource_id)
            .map(|resource| *resource as &dyn Resource)
            .ok_or(AnalyzerError::InvalidId(resource_id))
    }

    fn resource_type(&self, resource_type_name: &str) -> AnalyzerResult<&ResourceTypeDecl> {
        self.resource_types
            .get(resource_type_name)
            .copied()
            .ok_or_else(|| AnalyzerError::InvalidTypeName(resource_type_name.to_string()))
    }

    fn resource_group(&self, resource_group_id: Uuid) -> AnalyzerResult<&dyn ResourceGroup> {
        self.query_engine
            .resource_group(resource_group_id)
            .or_else(|_| {
                self.resource_groups
                    .get(&resource_group_id)
                    .map(|group| *group as &dyn ResourceGroup)
                    .ok_or(AnalyzerError::InvalidId(resource_group_id))
            })
    }

    fn resource_group_child_groups(
        &self,
        resource_group_id: Uuid,
    ) -> AnalyzerResult<impl Iterator<Item = Uuid>> {
        self.resource_group(resource_group_id)?;
        let query_engine = self
            .query_engine
            .resource_group_child_groups(resource_group_id)
            .ok();
        let task_resource_groups = self.resource_groups.values().filter_map(move |group| {
            group
                .parent_group_id
                .and_then(|parent| (parent == resource_group_id).then_some(group.id))
        });
        Ok(query_engine
            .into_iter()
            .flatten()
            .chain(task_resource_groups))
    }

    fn resource_group_child_resources(
        &self,
        resource_group_id: Uuid,
    ) -> AnalyzerResult<impl Iterator<Item = Uuid>> {
        self.resource_group(resource_group_id)?;
        Ok(self.resources.values().filter_map(move |resource| {
            (resource.parent_group_id() == resource_group_id).then_some(resource.id)
        }))
    }
}
