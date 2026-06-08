use instrumentation_model::SiriusEvent;
use quent_analyzer::{
    AnalyzerError, AnalyzerResult, Entity, Model, Span,
    fsm::FsmTypeDeclaration,
    resource::{ResourceTypeDecl, collection::ResourceCollection, tree::ResourceTreeNode},
    timeline::binned::resource::{
        ResourceTimeline, ResourceTimelineBuilder, ResourceTimelineByKey,
        ResourceTimelineByKeyBuilder,
    },
};
use quent_events::Event;
pub use quent_query_engine_analyzer::QueryEngineModel;
use quent_query_engine_analyzer::ui::UiAnalyzer;
use quent_query_engine_ui::{QueryBundle, QueryEntities};
use quent_time::bin::BinnedSpanSec;
use quent_time::{SpanNanoSec, TimeNanoSec, TimeUnixNanoSec, to_nanosecs, to_secs};
use quent_ui::{
    FiniteStateMachine, ResourceGroupNode, ResourceTree, convert_resource_tree,
    quantity::QuantitySpec,
    timeline::{
        request::{
            BulkChunkedTimelineRequest, BulkTimelineRequest, EntityFilter, SingleTimelineRequest,
            TimelineRequest,
        },
        response::{
            BulkChunkedTimelinesResponse, BulkTimelinesResponse, BulkTimelinesResponseEntry,
            ResourceTimeline as UiResourceTimeline, ResourceTimelineBinned,
            ResourceTimelineBinnedByState, SingleTimelineResponse,
        },
    },
};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use sirius_telemetry_ui::{EntityRef, QueryFilter, TaskFilter};
use std::{collections::HashMap as StdHashMap, sync::Arc};
use tracing::debug;
use uuid::Uuid;

use crate::{
    model::{SiriusModel, SiriusModelBuilder},
    task::{Task, TaskExt},
};

pub mod model;
pub mod task;
pub mod view;

pub struct SiriusUiAnalyzer {
    pub model: SiriusModel,
}

struct PlainBuilderSlot<'a> {
    entry_id: String,
    config_idx: usize,
    builder: ResourceTimelineBuilder<'a>,
    resource_id_filter: Arc<HashSet<Uuid>>,
    task_filter: TaskFilter,
    message: String,
}

struct PerStateBuilderSlot<'a> {
    entry_id: String,
    config_idx: usize,
    builder: ResourceTimelineByKeyBuilder<'a, &'a str>,
    resource_id_filter: Arc<HashSet<Uuid>>,
    task_filter: TaskFilter,
    message: String,
}

enum ResourceGroupTimelineResolution<'a> {
    Typed {
        resource_type: &'a ResourceTypeDecl,
        resource_type_name: String,
        message: String,
    },
    Empty {
        message: String,
    },
}

impl UiAnalyzer for SiriusUiAnalyzer {
    type Event = SiriusEvent;
    type EntityRef = EntityRef;
    type TimelineGlobalParams = QueryFilter;
    type TimelineParams = TaskFilter;

    fn extract_engine(
        engine_id: Uuid,
        events: impl Iterator<Item = Event<SiriusEvent>>,
    ) -> AnalyzerResult<quent_query_engine_ui::Engine> {
        use quent_query_engine_model::engine::EngineEvent;
        for event in events {
            if let SiriusEvent::Engine(EngineEvent::Init(init)) = event.data {
                return Ok(quent_query_engine_ui::Engine {
                    id: engine_id,
                    start_time_unix_ns: Some(event.timestamp),
                    duration_s: None,
                    instance_name: init.instance_name,
                    implementation: Some(
                        quent_query_engine_ui::EngineImplementationAttributes::from(
                            &init.implementation,
                        ),
                    ),
                });
            }
        }
        Ok(quent_query_engine_ui::Engine::new(engine_id))
    }

    fn try_new(
        engine_id: Uuid,
        events: impl Iterator<Item = Event<SiriusEvent>>,
    ) -> AnalyzerResult<Self> {
        let mut builder = SiriusModelBuilder::try_new(engine_id)?;
        {
            let _span = tracing::info_span!("ingest").entered();
            for event in events {
                builder.try_push(event)?;
            }
        }
        let model = {
            let _span = tracing::info_span!("build").entered();
            builder.try_build()?
        };

        let query_engine = &model.query_engine;
        tracing::info!(
            workers = query_engine.workers.len(),
            query_groups = query_engine.query_groups.len(),
            queries = query_engine.queries.len(),
            plans = query_engine.plans.len(),
            operators = query_engine.operators.len(),
            ports = query_engine.ports.len(),
            task_resources = model.arbitrary_resources.resources.len(),
            task_resource_groups = model.arbitrary_resources.resource_groups.len(),
            task_resource_types = model.arbitrary_resources.resource_types.len(),
            resource_group_types = model.resource_group_types.len(),
            tasks = model.tasks.len(),
        );

        Ok(Self { model })
    }

    fn query_bundle(&self, query_id: Uuid) -> AnalyzerResult<QueryBundle<EntityRef>> {
        debug!("constructing Sirius query view");
        let view = self.model.query_view(query_id)?;
        let query = self.model.query(query_id)?;
        let start_time_unix_ns = view.query_epoch(query_id)?;
        let duration_s = to_secs(query.span()?.duration());
        let epoch = view.query_epoch(query_id)?;

        let engine = view.engine()?.to_ui()?;
        let query_group_id = query.query_group_id().ok_or_else(|| {
            AnalyzerError::IncompleteEntity(format!("query {query_id} has no query_group_id"))
        })?;
        let query_group = view.query_group(query_group_id)?.to_ui();
        let query = query.to_ui()?;
        let workers = view
            .workers()
            .map(|worker| (worker.id(), worker.to_ui(epoch)))
            .collect();
        let plans = view.plans().map(|plan| (plan.id(), plan.to_ui())).collect();
        let operators = view
            .operators()
            .map(|operator| (operator.id(), operator.to_ui(epoch)))
            .collect();
        let ports = view
            .ports()
            .map(|port| (port.id(), port.to_ui(epoch)))
            .collect();
        let unique_operator_names = view
            .operators()
            .filter_map(|operator| operator.operator_type_name().map(str::to_owned))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        let resources = view
            .resources()
            .map(|resource| (resource.id(), resource.into()))
            .collect();
        let resource_types = view
            .resource_types()
            .map(|(name, resource_type)| (name.clone(), resource_type.into()))
            .collect();
        let resource_groups = self
            .model
            .arbitrary_resources
            .resource_groups()
            .map(|group| (group.id(), group.into()))
            .collect();
        let resource_group_types = view
            .resource_group_types()
            .map(|(name, resource_group_type)| (name.clone(), resource_group_type.into()))
            .collect();

        let task_decl = Task::fsm_type_declaration();
        let fsm_types = [(task_decl.name.clone(), task_decl)].into_iter().collect();

        let entities = QueryEntities {
            engine,
            query_group,
            query,
            workers,
            plans,
            operators,
            ports,
            resource_types,
            resource_group_types,
            resources,
            resource_groups,
            fsm_types,
        };

        let plan_tree = view.plan_tree(query_id)?.to_ui();
        let engine = view.engine()?;
        let resource_tree =
            convert_resource_tree(view.resource_tree()?, &view)?.unwrap_or_else(|| {
                ResourceTree::ResourceGroup(ResourceGroupNode {
                    id: EntityRef::Engine(engine.id()),
                    children: vec![],
                })
            });

        Ok(QueryBundle {
            query_id,
            entities,
            plan_tree,
            resource_tree,
            unique_operator_names,
            quantity_specs: [
                ("capacity_entries".to_string(), QuantitySpec::unit()),
                ("unit".to_string(), QuantitySpec::unit()),
            ]
            .into(),
            start_time_unix_ns,
            duration_s,
        })
    }

    fn query_engine_model(&self) -> &impl QueryEngineModel {
        &self.model
    }

    fn single_resource_timeline(
        &self,
        request: SingleTimelineRequest<Self::TimelineGlobalParams, Self::TimelineParams>,
    ) -> AnalyzerResult<SingleTimelineResponse> {
        let epoch = self
            .query_engine_model()
            .query_epoch(request.app_params.query_id)?;
        let view = self.model.query_view(request.app_params.query_id)?;
        let config = request.entry.config().try_into_binned_span(epoch)?;
        let config_secs = config.try_to_secs_relative(epoch)?;

        match request.entry {
            TimelineRequest::Resource(req) => {
                let resource_type = view.resource_type_of(req.resource_id)?;
                let long_entities_threshold = req.long_entities_threshold_s.map(to_nanosecs);
                let task_filter = req.application;
                let per_state = req.entity_filter.entity_type_name.is_some();
                let tasks =
                    self.filtered_tasks(&view, req.entity_filter, &task_filter, config.span)?;

                if per_state {
                    let mut builder = ResourceTimelineByKeyBuilder::try_new(
                        resource_type,
                        config,
                        long_entities_threshold,
                    )?;
                    self.populate_keyed_builder(&mut builder, tasks, |id| id == req.resource_id)?;
                    Ok(SingleTimelineResponse {
                        config: config_secs,
                        data: self.timeline_to_ui_keyed(builder.build(), epoch)?,
                    })
                } else {
                    let mut builder = ResourceTimelineBuilder::try_new(
                        resource_type,
                        config,
                        long_entities_threshold,
                    )?;
                    self.populate_plain_builder(&mut builder, tasks, |id| id == req.resource_id)?;
                    Ok(SingleTimelineResponse {
                        config: config_secs,
                        data: self.timeline_to_ui(builder.build(), epoch)?,
                    })
                }
            }
            TimelineRequest::ResourceGroup(req) => {
                let long_entities_threshold = req.long_entities_threshold_s.map(to_nanosecs);
                let tree = ResourceTreeNode::try_new(&view, req.resource_group_id)?;
                let resolution =
                    self.resolve_resource_group_timeline(&view, &tree, &req.resource_type_name)?;
                let resource_ids = tree
                    .iter_leaf_ids()
                    .filter(|&id| {
                        if let ResourceGroupTimelineResolution::Typed {
                            resource_type_name, ..
                        } = &resolution
                        {
                            view.resource(id)
                                .ok()
                                .is_some_and(|resource| resource.type_name() == resource_type_name)
                        } else {
                            false
                        }
                    })
                    .collect::<HashSet<_>>();
                let per_state = req.entity_filter.entity_type_name.is_some();
                let tasks =
                    self.filtered_tasks(&view, req.entity_filter, &req.app_params, config.span)?;

                let ResourceGroupTimelineResolution::Typed { resource_type, .. } = resolution
                else {
                    return Ok(SingleTimelineResponse {
                        config: config_secs,
                        data: empty_resource_timeline(per_state, config_secs),
                    });
                };

                if per_state {
                    let mut builder = ResourceTimelineByKeyBuilder::try_new(
                        resource_type,
                        config,
                        long_entities_threshold,
                    )?;
                    self.populate_keyed_builder(&mut builder, tasks, |id| {
                        resource_ids.contains(&id)
                    })?;
                    Ok(SingleTimelineResponse {
                        config: config_secs,
                        data: self.timeline_to_ui_keyed(builder.build(), epoch)?,
                    })
                } else {
                    let mut builder = ResourceTimelineBuilder::try_new(
                        resource_type,
                        config,
                        long_entities_threshold,
                    )?;
                    self.populate_plain_builder(&mut builder, tasks, |id| {
                        resource_ids.contains(&id)
                    })?;
                    Ok(SingleTimelineResponse {
                        config: config_secs,
                        data: self.timeline_to_ui(builder.build(), epoch)?,
                    })
                }
            }
        }
    }

    fn bulk_resource_timeline(
        &self,
        request: BulkTimelineRequest<Self::TimelineGlobalParams, Self::TimelineParams>,
    ) -> AnalyzerResult<BulkTimelinesResponse> {
        let epoch = self
            .query_engine_model()
            .query_epoch(request.app_params.query_id)?;
        let view = self.model.query_view(request.app_params.query_id)?;
        let resource_tree = view.resource_tree()?;

        let mut plain_builders: Vec<(
            String,
            ResourceTimelineBuilder<'_>,
            HashSet<Uuid>,
            TaskFilter,
            String,
        )> = Vec::new();
        let mut per_state_builders: Vec<(
            String,
            ResourceTimelineByKeyBuilder<'_, &str>,
            HashSet<Uuid>,
            TaskFilter,
            String,
        )> = Vec::new();

        let mut entries = std::collections::HashMap::new();
        for (entry_id, entry) in request.entries {
            let entry_config = match entry.config().try_into_binned_span(epoch) {
                Ok(entry_config) => entry_config,
                Err(error) => {
                    entries.insert(
                        entry_id,
                        BulkTimelinesResponseEntry::Error {
                            message: error.to_string(),
                        },
                    );
                    continue;
                }
            };
            let prep = match self.try_prepare_bulk_entry(&view, entry, &resource_tree) {
                Ok(prep) => prep,
                Err(error) => {
                    entries.insert(
                        entry_id,
                        BulkTimelinesResponseEntry::Error {
                            message: error.to_string(),
                        },
                    );
                    continue;
                }
            };
            let BulkEntryPrep {
                resource_type,
                resource_id_filter,
                entity_filter,
                task_filter,
                long_entities_threshold,
                message,
            } = prep;
            let per_state = entity_filter.entity_type_name.is_some();
            let Some(resource_type) = resource_type else {
                let config = entry_config.try_to_secs_relative(epoch)?;
                entries.insert(
                    entry_id,
                    BulkTimelinesResponseEntry::Ok {
                        message,
                        config,
                        data: empty_resource_timeline(per_state, config),
                    },
                );
                continue;
            };
            if per_state {
                per_state_builders.push((
                    entry_id,
                    ResourceTimelineByKeyBuilder::try_new(
                        resource_type,
                        entry_config,
                        long_entities_threshold,
                    )?,
                    resource_id_filter,
                    task_filter,
                    message,
                ));
            } else {
                plain_builders.push((
                    entry_id,
                    ResourceTimelineBuilder::try_new(
                        resource_type,
                        entry_config,
                        long_entities_threshold,
                    )?,
                    resource_id_filter,
                    task_filter,
                    message,
                ));
            }
        }

        let plain_index = build_resource_index(&plain_builders, |entry| &entry.2);
        let per_state_index = build_resource_index(&per_state_builders, |entry| &entry.2);
        let tasks = view.tasks().collect::<Vec<_>>();

        for task in tasks {
            for usage in task.normalized_usages() {
                let resource_id = usage.resource_id();
                if let Some(builder_indices) = plain_index.get(&resource_id) {
                    for &builder_idx in builder_indices {
                        if task.matches_filter(&plain_builders[builder_idx].3) {
                            plain_builders[builder_idx].1.try_push(&usage)?;
                        }
                    }
                }
            }
            for (state_name, usage) in task.normalized_usages_with_state_names() {
                let resource_id = usage.resource_id();
                if let Some(builder_indices) = per_state_index.get(&resource_id) {
                    for &builder_idx in builder_indices {
                        if task.matches_filter(&per_state_builders[builder_idx].3) {
                            per_state_builders[builder_idx]
                                .1
                                .try_push(state_name, &usage)?;
                        }
                    }
                }
            }
        }

        for (entry_id, builder, _, _, message) in plain_builders {
            let built = builder.build();
            let config = built.config.try_to_secs_relative(epoch)?;
            entries.insert(
                entry_id,
                BulkTimelinesResponseEntry::Ok {
                    message,
                    config,
                    data: self.timeline_to_ui(built, epoch)?,
                },
            );
        }
        for (entry_id, builder, _, _, message) in per_state_builders {
            let built = builder.build();
            let config = built.config.try_to_secs_relative(epoch)?;
            entries.insert(
                entry_id,
                BulkTimelinesResponseEntry::Ok {
                    message,
                    config,
                    data: self.timeline_to_ui_keyed(built, epoch)?,
                },
            );
        }

        Ok(BulkTimelinesResponse { entries })
    }

    fn bulk_chunked_resource_timeline(
        &self,
        request: BulkChunkedTimelineRequest<Self::TimelineGlobalParams, Self::TimelineParams>,
    ) -> AnalyzerResult<BulkChunkedTimelinesResponse> {
        let epoch = self
            .query_engine_model()
            .query_epoch(request.app_params.query_id)?;
        let view = self.model.query_view(request.app_params.query_id)?;
        let resource_tree = view.resource_tree()?;
        let n_configs = request.configs.len();

        let mut plain_builders = Vec::with_capacity(request.entries.len() * n_configs);
        let mut per_state_builders = Vec::with_capacity(request.entries.len() * n_configs);
        let mut error_entries = std::collections::HashMap::new();

        for (entry_id, entry) in &request.entries {
            let prep = match self.try_prepare_bulk_entry(&view, entry.clone(), &resource_tree) {
                Ok(prep) => prep,
                Err(error) => {
                    error_entries.insert(
                        entry_id.clone(),
                        (0..n_configs)
                            .map(|_| BulkTimelinesResponseEntry::Error {
                                message: error.to_string(),
                            })
                            .collect::<Vec<_>>(),
                    );
                    continue;
                }
            };
            let BulkEntryPrep {
                resource_type,
                resource_id_filter,
                entity_filter,
                task_filter,
                long_entities_threshold,
                message,
            } = prep;
            let per_state = entity_filter.entity_type_name.is_some();
            let Some(resource_type) = resource_type else {
                error_entries.insert(
                    entry_id.clone(),
                    request
                        .configs
                        .iter()
                        .map(|config| {
                            let config = config.clone().try_into_binned_span(epoch)?;
                            let config = config.try_to_secs_relative(epoch)?;
                            Ok(BulkTimelinesResponseEntry::Ok {
                                message: message.clone(),
                                config,
                                data: empty_resource_timeline(per_state, config),
                            })
                        })
                        .collect::<AnalyzerResult<Vec<_>>>()?,
                );
                continue;
            };
            let resource_id_filter = Arc::new(resource_id_filter);

            for (config_idx, config) in request.configs.iter().enumerate() {
                let entry_config = config.try_into_binned_span(epoch)?;
                if per_state {
                    per_state_builders.push(PerStateBuilderSlot {
                        entry_id: entry_id.clone(),
                        config_idx,
                        builder: ResourceTimelineByKeyBuilder::try_new(
                            resource_type,
                            entry_config,
                            long_entities_threshold,
                        )?,
                        resource_id_filter: Arc::clone(&resource_id_filter),
                        task_filter: task_filter.clone(),
                        message: message.clone(),
                    });
                } else {
                    plain_builders.push(PlainBuilderSlot {
                        entry_id: entry_id.clone(),
                        config_idx,
                        builder: ResourceTimelineBuilder::try_new(
                            resource_type,
                            entry_config,
                            long_entities_threshold,
                        )?,
                        resource_id_filter: Arc::clone(&resource_id_filter),
                        task_filter: task_filter.clone(),
                        message: message.clone(),
                    });
                }
            }
        }

        let plain_index =
            build_resource_index(&plain_builders, |slot| slot.resource_id_filter.as_ref());
        let per_state_index =
            build_resource_index(&per_state_builders, |slot| slot.resource_id_filter.as_ref());

        for task in view.tasks() {
            for usage in task.normalized_usages() {
                let resource_id = usage.resource_id();
                if let Some(builder_indices) = plain_index.get(&resource_id) {
                    for &builder_idx in builder_indices {
                        if task.matches_filter(&plain_builders[builder_idx].task_filter) {
                            plain_builders[builder_idx].builder.try_push(&usage)?;
                        }
                    }
                }
            }
            for (state_name, usage) in task.normalized_usages_with_state_names() {
                let resource_id = usage.resource_id();
                if let Some(builder_indices) = per_state_index.get(&resource_id) {
                    for &builder_idx in builder_indices {
                        if task.matches_filter(&per_state_builders[builder_idx].task_filter) {
                            per_state_builders[builder_idx]
                                .builder
                                .try_push(state_name, &usage)?;
                        }
                    }
                }
            }
        }

        let mut slots: HashMap<String, Vec<Option<BulkTimelinesResponseEntry>>> = request
            .entries
            .keys()
            .map(|key| (key.clone(), (0..n_configs).map(|_| None).collect()))
            .collect();
        for (entry_id, errors) in error_entries {
            slots.insert(entry_id, errors.into_iter().map(Some).collect());
        }

        for slot in plain_builders {
            let built = slot.builder.build();
            let config = built.config.try_to_secs_relative(epoch)?;
            slots.get_mut(&slot.entry_id).unwrap()[slot.config_idx] =
                Some(BulkTimelinesResponseEntry::Ok {
                    message: slot.message,
                    config,
                    data: self.timeline_to_ui(built, epoch)?,
                });
        }
        for slot in per_state_builders {
            let built = slot.builder.build();
            let config = built.config.try_to_secs_relative(epoch)?;
            slots.get_mut(&slot.entry_id).unwrap()[slot.config_idx] =
                Some(BulkTimelinesResponseEntry::Ok {
                    message: slot.message,
                    config,
                    data: self.timeline_to_ui_keyed(built, epoch)?,
                });
        }

        let entries = slots
            .into_iter()
            .map(|(key, values)| {
                let values = values
                    .into_iter()
                    .map(|value| {
                        value.ok_or(AnalyzerError::BrokenImpl(
                            "chunked bulk: missing builder slot",
                        ))
                    })
                    .collect::<AnalyzerResult<Vec<_>>>()?;
                Ok((key, values))
            })
            .collect::<AnalyzerResult<std::collections::HashMap<_, _>>>()?;

        Ok(BulkChunkedTimelinesResponse { entries })
    }
}

impl SiriusUiAnalyzer {
    fn filtered_tasks<'a>(
        &self,
        view: &'a crate::view::SiriusModelQueryView<'a>,
        entity_filter: EntityFilter,
        task_filter: &TaskFilter,
        time_window: SpanNanoSec,
    ) -> AnalyzerResult<Vec<&'a Task>> {
        if let Some(entity_type_name) = entity_filter.entity_type_name
            && entity_type_name != "task"
        {
            return Err(AnalyzerError::InvalidArgument(format!(
                "{entity_type_name} is not a known entity type in this model"
            )));
        }

        Ok(view
            .tasks()
            .filter(|task| {
                task.matches_filter(task_filter)
                    && task.span().is_ok_and(|span| span.intersects(&time_window))
            })
            .collect())
    }

    fn try_prepare_bulk_entry<'a>(
        &'a self,
        view: &'a crate::view::SiriusModelQueryView<'a>,
        request: TimelineRequest<TaskFilter>,
        tree: &ResourceTreeNode,
    ) -> AnalyzerResult<BulkEntryPrep<'a>> {
        Ok(match request {
            TimelineRequest::Resource(request) => BulkEntryPrep {
                resource_type: Some(view.resource_type_of(request.resource_id)?),
                resource_id_filter: [request.resource_id].into_iter().collect(),
                entity_filter: request.entity_filter,
                task_filter: request.application,
                long_entities_threshold: request.long_entities_threshold_s.map(to_nanosecs),
                message: String::new(),
            },
            TimelineRequest::ResourceGroup(request) => {
                let subtree = tree
                    .find(request.resource_group_id)
                    .ok_or(AnalyzerError::InvalidId(request.resource_group_id))?;
                let resolution = self.resolve_resource_group_timeline(
                    view,
                    subtree,
                    &request.resource_type_name,
                )?;
                let (resource_type, resource_type_name, message) = match resolution {
                    ResourceGroupTimelineResolution::Typed {
                        resource_type,
                        resource_type_name,
                        message,
                    } => (Some(resource_type), resource_type_name, message),
                    ResourceGroupTimelineResolution::Empty { message } => {
                        (None, String::new(), message)
                    }
                };
                let resource_id_filter = subtree
                    .iter_leaf_ids()
                    .filter(|&id| {
                        view.resource(id)
                            .ok()
                            .is_some_and(|resource| resource.type_name() == resource_type_name)
                    })
                    .collect();
                BulkEntryPrep {
                    resource_type,
                    resource_id_filter,
                    entity_filter: request.entity_filter,
                    task_filter: request.app_params,
                    long_entities_threshold: request.long_entities_threshold_s.map(to_nanosecs),
                    message,
                }
            }
        })
    }

    fn resolve_resource_group_timeline<'a>(
        &'a self,
        view: &'a crate::view::SiriusModelQueryView<'a>,
        subtree: &ResourceTreeNode,
        requested_resource_type_name: &str,
    ) -> AnalyzerResult<ResourceGroupTimelineResolution<'a>> {
        if !requested_resource_type_name.is_empty() {
            return Ok(ResourceGroupTimelineResolution::Typed {
                resource_type: view.resource_type(requested_resource_type_name)?,
                resource_type_name: requested_resource_type_name.to_string(),
                message: String::new(),
            });
        }

        let mut resource_type_names = subtree
            .iter_leaf_ids()
            .filter_map(|id| view.resource(id).ok())
            .map(|resource| resource.type_name().to_string())
            .filter(|resource_type_name| view.resource_type(resource_type_name).is_ok())
            .collect::<Vec<_>>();
        resource_type_names.sort();
        resource_type_names.dedup();

        let Some(resource_type_name) = resource_type_names.into_iter().next() else {
            return Ok(ResourceGroupTimelineResolution::Empty {
                message: "empty resource group".to_string(),
            });
        };

        Ok(ResourceGroupTimelineResolution::Typed {
            resource_type: view.resource_type(&resource_type_name)?,
            resource_type_name: resource_type_name.clone(),
            message: format!("inferred resource_type_name={resource_type_name}"),
        })
    }

    fn populate_plain_builder<'a>(
        &self,
        builder: &mut ResourceTimelineBuilder<'a>,
        tasks: Vec<&'a Task>,
        resource_filter: impl Fn(Uuid) -> bool,
    ) -> AnalyzerResult<()> {
        for task in tasks {
            for usage in task.normalized_usages() {
                if resource_filter(usage.resource_id()) {
                    builder.try_push(&usage)?;
                }
            }
        }
        Ok(())
    }

    fn populate_keyed_builder<'a>(
        &self,
        builder: &mut ResourceTimelineByKeyBuilder<'a, &'a str>,
        tasks: Vec<&'a Task>,
        resource_filter: impl Fn(Uuid) -> bool,
    ) -> AnalyzerResult<()> {
        for task in tasks {
            for (state_name, usage) in task.normalized_usages_with_state_names() {
                if resource_filter(usage.resource_id()) {
                    builder.try_push(state_name, &usage)?;
                }
            }
        }
        Ok(())
    }

    fn task_entities_to_ui_fsm(
        &self,
        entity_ids: &[Uuid],
        epoch: TimeUnixNanoSec,
    ) -> AnalyzerResult<Vec<FiniteStateMachine>> {
        entity_ids
            .iter()
            .filter_map(|&id| {
                self.model
                    .tasks
                    .get(&id)
                    .map(|task| task.try_to_ui_fsm(epoch))
            })
            .collect()
    }

    fn timeline_to_ui(
        &self,
        result: ResourceTimeline<'_>,
        epoch: TimeUnixNanoSec,
    ) -> AnalyzerResult<UiResourceTimeline> {
        let config = result.config.try_to_secs_relative(epoch)?;
        let capacities_values = result
            .data
            .into_iter()
            .map(|(capacity, values)| (capacity.to_string(), values))
            .collect();
        let long_fsms = self.task_entities_to_ui_fsm(&result.long_entities, epoch)?;
        Ok(UiResourceTimeline::Binned(ResourceTimelineBinned {
            config,
            capacities_values,
            long_fsms,
        }))
    }

    fn timeline_to_ui_keyed(
        &self,
        result: ResourceTimelineByKey<'_, &str>,
        epoch: TimeUnixNanoSec,
    ) -> AnalyzerResult<UiResourceTimeline> {
        let config = result.config.try_to_secs_relative(epoch)?;
        let mut capacities_states_values = StdHashMap::new();
        for ((state_name, capacity_name), values) in result.data {
            capacities_states_values
                .entry(capacity_name.to_string())
                .or_insert_with(StdHashMap::new)
                .insert(state_name.to_string(), values);
        }
        let long_fsms = self.task_entities_to_ui_fsm(&result.long_entities, epoch)?;
        Ok(UiResourceTimeline::BinnedByState(
            ResourceTimelineBinnedByState {
                config,
                capacities_states_values,
                long_fsms,
            },
        ))
    }
}

struct BulkEntryPrep<'a> {
    resource_type: Option<&'a ResourceTypeDecl>,
    resource_id_filter: HashSet<Uuid>,
    entity_filter: EntityFilter,
    task_filter: TaskFilter,
    long_entities_threshold: Option<TimeNanoSec>,
    message: String,
}

fn empty_resource_timeline(per_state: bool, config: BinnedSpanSec) -> UiResourceTimeline {
    if per_state {
        UiResourceTimeline::BinnedByState(ResourceTimelineBinnedByState {
            config,
            capacities_states_values: StdHashMap::new(),
            long_fsms: vec![],
        })
    } else {
        UiResourceTimeline::Binned(ResourceTimelineBinned {
            config,
            capacities_values: StdHashMap::new(),
            long_fsms: vec![],
        })
    }
}

fn build_resource_index<T>(
    entries: &[T],
    resource_ids: impl Fn(&T) -> &HashSet<Uuid>,
) -> HashMap<Uuid, Vec<usize>> {
    entries
        .iter()
        .enumerate()
        .flat_map(|(entry_index, entry)| {
            resource_ids(entry)
                .iter()
                .map(move |&resource_id| (resource_id, entry_index))
        })
        .fold(HashMap::default(), |mut acc, (resource_id, entry_index)| {
            acc.entry(resource_id).or_default().push(entry_index);
            acc
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use instrumentation_model::task;
    use quent_attributes::CustomAttributes;
    use quent_model::{Capacity, FsmEvent, Ref, Usage};
    use quent_query_engine_model::{engine, operator, plan, query, query_group, worker};
    use quent_ui::timeline::{
        request::{
            BulkTimelineRequest, EntityFilter, ResourceGroupTimelineRequest,
            ResourceTimelineRequest, SingleTimelineRequest, TimelineConfig, TimelineRequest,
        },
        response::{BulkTimelinesResponseEntry, ResourceTimeline as UiResourceTimeline},
    };

    struct Fixture {
        engine_id: Uuid,
        query_id: Uuid,
        queue_id: Uuid,
        manager_id: Uuid,
        executor_id: Uuid,
        events: Vec<Event<SiriusEvent>>,
    }

    fn id(n: u64) -> Uuid {
        Uuid::from_u64_pair(0, n)
    }

    fn fsm_event<T>(seq: u64, state: T) -> FsmEvent<T> {
        FsmEvent { seq, state }
    }

    fn fixture() -> Fixture {
        let engine_id = id(1);
        let worker_id = id(2);
        let query_group_id = id(3);
        let query_id = id(4);
        let plan_id = id(5);
        let pipeline_uuid = id(6);
        let queue_id = id(7);
        let manager_id = id(8);
        let executor_id = id(9);
        let task_id = id(10);

        let mut events = vec![
            Event::new(
                engine_id,
                0,
                SiriusEvent::Engine(engine::EngineEvent::Init(engine::Init {
                    implementation: engine::EngineImplementationAttributes {
                        name: Some("sirius".to_string()),
                        version: Some("test".to_string()),
                        custom_attributes: CustomAttributes::new(),
                    },
                    instance_name: Some("sirius-test".to_string()),
                })),
            ),
            Event::new(
                worker_id,
                1,
                SiriusEvent::Worker(worker::WorkerEvent::Init(worker::Init {
                    parent_engine_id: Ref::new(engine_id),
                    instance_name: "worker".to_string(),
                })),
            ),
            Event::new(
                query_group_id,
                2,
                SiriusEvent::QueryGroup(query_group::QueryGroupEvent::Declaration(
                    query_group::Declaration {
                        instance_name: "query-group".to_string(),
                        engine_id,
                    },
                )),
            ),
            Event::new(
                query_id,
                3,
                SiriusEvent::Query(fsm_event(
                    0,
                    query::QueryTransition::Init(query::Init {
                        instance_name: "query".to_string(),
                        query_group_id: Ref::new(query_group_id),
                    }),
                )),
            ),
            Event::new(
                query_id,
                4,
                SiriusEvent::Query(fsm_event(
                    1,
                    query::QueryTransition::Planning(query::Planning {}),
                )),
            ),
            Event::new(
                query_id,
                5,
                SiriusEvent::Query(fsm_event(
                    2,
                    query::QueryTransition::Executing(query::Executing {}),
                )),
            ),
            Event::new(
                plan_id,
                6,
                SiriusEvent::Plan(plan::PlanEvent::Declaration(plan::Declaration {
                    parent: plan::PlanParent {
                        query_id: Some(Ref::new(query_id)),
                        plan_id: None,
                    },
                    instance_name: "pipeline-plan".to_string(),
                    edges: vec![],
                    worker_id: Some(Ref::new(worker_id)),
                })),
            ),
            Event::new(
                pipeline_uuid,
                7,
                SiriusEvent::Operator(operator::OperatorEvent::Declaration(
                    operator::Declaration {
                        plan_id: Ref::new(plan_id),
                        parent_operator_ids: vec![],
                        instance_name: "pipeline".to_string(),
                        type_name: "Pipeline Id 1".to_string(),
                        custom_attributes: CustomAttributes::new(),
                    },
                )),
            ),
            Event::new(
                queue_id,
                8,
                SiriusEvent::TaskQueue(fsm_event(
                    0,
                    task::TaskQueueTransition::TaskQueueInitializing(task::TaskQueueInitializing {
                        instance_name: "task-queue".to_string(),
                        parent_group_id: engine_id,
                        resource_type_name: "task_queue".to_string(),
                    }),
                )),
            ),
            Event::new(
                queue_id,
                9,
                SiriusEvent::TaskQueue(fsm_event(
                    1,
                    task::TaskQueueTransition::TaskQueueOperating(task::TaskQueueOperating {
                        capacity_entries: Capacity::new(Some(1024)),
                    }),
                )),
            ),
            Event::new(
                manager_id,
                8,
                SiriusEvent::TaskManagerLoopThread(fsm_event(
                    0,
                    task::TaskManagerLoopThreadTransition::TaskManagerLoopThreadInitializing(
                        task::TaskManagerLoopThreadInitializing {
                            instance_name: "manager".to_string(),
                            parent_group_id: engine_id,
                            resource_type_name: "task_manager_loop_thread".to_string(),
                        },
                    ),
                )),
            ),
            Event::new(
                manager_id,
                9,
                SiriusEvent::TaskManagerLoopThread(fsm_event(
                    1,
                    task::TaskManagerLoopThreadTransition::TaskManagerLoopThreadOperating(
                        task::TaskManagerLoopThreadOperating,
                    ),
                )),
            ),
            Event::new(
                executor_id,
                8,
                SiriusEvent::ExecutorThread(fsm_event(
                    0,
                    task::ExecutorThreadTransition::ExecutorThreadInitializing(
                        task::ExecutorThreadInitializing {
                            instance_name: "executor".to_string(),
                            parent_group_id: engine_id,
                            resource_type_name: "executor_thread".to_string(),
                        },
                    ),
                )),
            ),
            Event::new(
                executor_id,
                9,
                SiriusEvent::ExecutorThread(fsm_event(
                    1,
                    task::ExecutorThreadTransition::ExecutorThreadOperating(
                        task::ExecutorThreadOperating,
                    ),
                )),
            ),
            Event::new(
                task_id,
                10,
                SiriusEvent::Task(fsm_event(
                    0,
                    task::TaskTransition::Created(task::Created {
                        instance_name: "task-10".to_string(),
                        pipeline_uuid,
                    }),
                )),
            ),
            Event::new(
                task_id,
                20,
                SiriusEvent::Task(fsm_event(
                    1,
                    task::TaskTransition::Queued(task::Queued {
                        queue: Some(Usage {
                            resource_id: Ref::new(queue_id),
                            capacity: Default::default(),
                        }),
                    }),
                )),
            ),
            Event::new(
                task_id,
                30,
                SiriusEvent::Task(fsm_event(
                    2,
                    task::TaskTransition::Routing(task::Routing {
                        instance_name: String::new(),
                        preferred_device_id: 0,
                        manager_thread: Some(Usage {
                            resource_id: Ref::new(manager_id),
                            capacity: Default::default(),
                        }),
                    }),
                )),
            ),
            Event::new(
                task_id,
                40,
                SiriusEvent::Task(fsm_event(
                    3,
                    task::TaskTransition::Reserving(task::Reserving {
                        instance_name: String::new(),
                        requested_bytes: 4096,
                        input_basis: 1024,
                        peak_estimate: 2048,
                        bytes_to_materialize: 1024,
                        manager_thread: Some(Usage {
                            resource_id: Ref::new(manager_id),
                            capacity: Default::default(),
                        }),
                    }),
                )),
            ),
            Event::new(
                task_id,
                50,
                SiriusEvent::Task(fsm_event(
                    4,
                    task::TaskTransition::Preparing(task::Preparing {
                        instance_name: String::new(),
                        target_tier: "device".to_string(),
                        executor_thread: Some(Usage {
                            resource_id: Ref::new(executor_id),
                            capacity: Default::default(),
                        }),
                    }),
                )),
            ),
            Event::new(
                task_id,
                60,
                SiriusEvent::Task(fsm_event(
                    5,
                    task::TaskTransition::Computing(task::Computing {
                        instance_name: String::new(),
                        current_operator_id: 42,
                        input_bytes: 2048,
                        executor_thread: Some(Usage {
                            resource_id: Ref::new(executor_id),
                            capacity: Default::default(),
                        }),
                    }),
                )),
            ),
            Event::new(
                task_id,
                90,
                SiriusEvent::Task(fsm_event(
                    6,
                    task::TaskTransition::Finalizing(task::Finalizing {
                        instance_name: String::new(),
                        success: true,
                    }),
                )),
            ),
            Event::new(
                task_id,
                100,
                SiriusEvent::Task(fsm_event(7, task::TaskTransition::Exit)),
            ),
            Event::new(
                query_id,
                110,
                SiriusEvent::Query(fsm_event(3, query::QueryTransition::Exit)),
            ),
            Event::new(
                queue_id,
                120,
                SiriusEvent::TaskQueue(fsm_event(
                    2,
                    task::TaskQueueTransition::TaskQueueFinalizing(task::TaskQueueFinalizing),
                )),
            ),
            Event::new(
                queue_id,
                121,
                SiriusEvent::TaskQueue(fsm_event(3, task::TaskQueueTransition::Exit)),
            ),
            Event::new(
                manager_id,
                120,
                SiriusEvent::TaskManagerLoopThread(fsm_event(
                    2,
                    task::TaskManagerLoopThreadTransition::TaskManagerLoopThreadFinalizing(
                        task::TaskManagerLoopThreadFinalizing,
                    ),
                )),
            ),
            Event::new(
                manager_id,
                121,
                SiriusEvent::TaskManagerLoopThread(fsm_event(
                    3,
                    task::TaskManagerLoopThreadTransition::Exit,
                )),
            ),
            Event::new(
                executor_id,
                120,
                SiriusEvent::ExecutorThread(fsm_event(
                    2,
                    task::ExecutorThreadTransition::ExecutorThreadFinalizing(
                        task::ExecutorThreadFinalizing,
                    ),
                )),
            ),
            Event::new(
                executor_id,
                121,
                SiriusEvent::ExecutorThread(fsm_event(3, task::ExecutorThreadTransition::Exit)),
            ),
            Event::new(
                worker_id,
                122,
                SiriusEvent::Worker(worker::WorkerEvent::Exit(worker::Exit)),
            ),
            Event::new(
                engine_id,
                123,
                SiriusEvent::Engine(engine::EngineEvent::Exit(engine::Exit)),
            ),
        ];
        events.sort_by_key(|event| event.timestamp);

        Fixture {
            engine_id,
            query_id,
            queue_id,
            manager_id,
            executor_id,
            events,
        }
    }

    fn analyzer(fixture: Fixture) -> SiriusUiAnalyzer {
        <SiriusUiAnalyzer as UiAnalyzer>::try_new(fixture.engine_id, fixture.events.into_iter())
            .unwrap()
    }

    fn timeline_config() -> TimelineConfig {
        TimelineConfig {
            num_bins: 8,
            start: 0.0,
            end: 0.0000002,
        }
    }

    fn task_filter() -> TaskFilter {
        TaskFilter {
            pipeline_uuid: None,
            current_operator_id: None,
        }
    }

    #[test]
    fn query_bundle_contains_task_timeline_entities() {
        let fixture = fixture();
        let query_id = fixture.query_id;
        let queue_id = fixture.queue_id;
        let manager_id = fixture.manager_id;
        let executor_id = fixture.executor_id;
        let analyzer = analyzer(fixture);

        let bundle = analyzer.query_bundle(query_id).unwrap();

        assert!(bundle.entities.fsm_types.contains_key("task"));
        assert!(bundle.entities.resources.contains_key(&queue_id));
        assert!(bundle.entities.resources.contains_key(&manager_id));
        assert!(bundle.entities.resources.contains_key(&executor_id));
        assert!(bundle.quantity_specs.contains_key("capacity_entries"));
    }

    #[test]
    fn queue_timeline_has_nonzero_queued_occupancy() {
        let fixture = fixture();
        let query_id = fixture.query_id;
        let queue_id = fixture.queue_id;
        let analyzer = analyzer(fixture);

        let response = analyzer
            .single_resource_timeline(SingleTimelineRequest {
                entry: TimelineRequest::Resource(ResourceTimelineRequest {
                    resource_id: queue_id,
                    long_entities_threshold_s: None,
                    entity_filter: EntityFilter {
                        entity_type_name: None,
                    },
                    application: task_filter(),
                    config: timeline_config(),
                }),
                app_params: QueryFilter { query_id },
            })
            .unwrap();

        let UiResourceTimeline::Binned(data) = response.data else {
            panic!("expected plain binned timeline");
        };
        let values = data.capacities_values.get("capacity_entries").unwrap();
        assert!(values.iter().any(|value| *value > 0.0));
    }

    #[test]
    fn per_state_timeline_contains_queued_state_and_long_fsm() {
        let fixture = fixture();
        let query_id = fixture.query_id;
        let queue_id = fixture.queue_id;
        let analyzer = analyzer(fixture);

        let response = analyzer
            .single_resource_timeline(SingleTimelineRequest {
                entry: TimelineRequest::Resource(ResourceTimelineRequest {
                    resource_id: queue_id,
                    long_entities_threshold_s: Some(0.0),
                    entity_filter: EntityFilter {
                        entity_type_name: Some("task".to_string()),
                    },
                    application: task_filter(),
                    config: timeline_config(),
                }),
                app_params: QueryFilter { query_id },
            })
            .unwrap();

        let UiResourceTimeline::BinnedByState(data) = response.data else {
            panic!("expected per-state binned timeline");
        };
        let states = data
            .capacities_states_values
            .get("capacity_entries")
            .unwrap();
        assert!(states.contains_key("queued"));
        assert!(states["queued"].iter().any(|value| *value > 0.0));
        assert_eq!(data.long_fsms.len(), 1);
    }

    #[test]
    fn empty_structural_group_single_timeline_returns_empty_plain_timeline() {
        let fixture = fixture();
        let query_id = fixture.query_id;
        let analyzer = analyzer(fixture);

        let response = analyzer
            .single_resource_timeline(SingleTimelineRequest {
                entry: TimelineRequest::ResourceGroup(ResourceGroupTimelineRequest {
                    resource_group_id: query_id,
                    resource_type_name: String::new(),
                    long_entities_threshold_s: None,
                    entity_filter: EntityFilter {
                        entity_type_name: None,
                    },
                    app_params: task_filter(),
                    config: timeline_config(),
                }),
                app_params: QueryFilter { query_id },
            })
            .unwrap();

        let UiResourceTimeline::Binned(data) = response.data else {
            panic!("expected empty plain binned timeline");
        };
        assert!(data.capacities_values.is_empty());
        assert!(data.long_fsms.is_empty());
    }

    #[test]
    fn empty_structural_group_single_timeline_returns_empty_per_state_timeline() {
        let fixture = fixture();
        let query_id = fixture.query_id;
        let analyzer = analyzer(fixture);

        let response = analyzer
            .single_resource_timeline(SingleTimelineRequest {
                entry: TimelineRequest::ResourceGroup(ResourceGroupTimelineRequest {
                    resource_group_id: query_id,
                    resource_type_name: String::new(),
                    long_entities_threshold_s: None,
                    entity_filter: EntityFilter {
                        entity_type_name: Some("task".to_string()),
                    },
                    app_params: task_filter(),
                    config: timeline_config(),
                }),
                app_params: QueryFilter { query_id },
            })
            .unwrap();

        let UiResourceTimeline::BinnedByState(data) = response.data else {
            panic!("expected empty per-state binned timeline");
        };
        assert!(data.capacities_states_values.is_empty());
        assert!(data.long_fsms.is_empty());
    }

    #[test]
    fn bulk_group_timeline_infers_empty_resource_type_name() {
        let fixture = fixture();
        let query_id = fixture.query_id;
        let engine_id = fixture.engine_id;
        let analyzer = analyzer(fixture);

        let response = analyzer
            .bulk_resource_timeline(BulkTimelineRequest {
                entries: [(
                    "root".to_string(),
                    TimelineRequest::ResourceGroup(ResourceGroupTimelineRequest {
                        resource_group_id: engine_id,
                        resource_type_name: String::new(),
                        long_entities_threshold_s: None,
                        entity_filter: EntityFilter {
                            entity_type_name: None,
                        },
                        app_params: task_filter(),
                        config: timeline_config(),
                    }),
                )]
                .into(),
                app_params: QueryFilter { query_id },
            })
            .unwrap();

        let Some(BulkTimelinesResponseEntry::Ok { message, data, .. }) =
            response.entries.get("root")
        else {
            panic!("expected inferred resource-group timeline");
        };
        assert!(message.starts_with("inferred resource_type_name="));
        let UiResourceTimeline::Binned(data) = data else {
            panic!("expected plain binned timeline");
        };
        assert!(
            data.capacities_values
                .values()
                .any(|values| values.iter().any(|value| *value > 0.0))
        );
    }

    #[test]
    fn bulk_timeline_reports_bad_entry_without_failing_valid_entries() {
        let fixture = fixture();
        let query_id = fixture.query_id;
        let engine_id = fixture.engine_id;
        let queue_id = fixture.queue_id;
        let analyzer = analyzer(fixture);

        let response = analyzer
            .bulk_resource_timeline(BulkTimelineRequest {
                entries: [
                    (
                        "valid".to_string(),
                        TimelineRequest::Resource(ResourceTimelineRequest {
                            resource_id: queue_id,
                            long_entities_threshold_s: None,
                            entity_filter: EntityFilter {
                                entity_type_name: None,
                            },
                            application: task_filter(),
                            config: timeline_config(),
                        }),
                    ),
                    (
                        "bad".to_string(),
                        TimelineRequest::ResourceGroup(ResourceGroupTimelineRequest {
                            resource_group_id: engine_id,
                            resource_type_name: "missing_resource_type".to_string(),
                            long_entities_threshold_s: None,
                            entity_filter: EntityFilter {
                                entity_type_name: None,
                            },
                            app_params: task_filter(),
                            config: timeline_config(),
                        }),
                    ),
                ]
                .into(),
                app_params: QueryFilter { query_id },
            })
            .unwrap();

        assert!(matches!(
            response.entries.get("valid"),
            Some(BulkTimelinesResponseEntry::Ok { .. })
        ));
        assert!(matches!(
            response.entries.get("bad"),
            Some(BulkTimelinesResponseEntry::Error { .. })
        ));
    }

    #[test]
    fn current_operator_filter_keeps_matching_tasks() {
        let fixture = fixture();
        let query_id = fixture.query_id;
        let executor_id = fixture.executor_id;
        let analyzer = analyzer(fixture);

        let matching = analyzer
            .single_resource_timeline(SingleTimelineRequest {
                entry: TimelineRequest::Resource(ResourceTimelineRequest {
                    resource_id: executor_id,
                    long_entities_threshold_s: None,
                    entity_filter: EntityFilter {
                        entity_type_name: None,
                    },
                    application: TaskFilter {
                        pipeline_uuid: None,
                        current_operator_id: Some(42),
                    },
                    config: timeline_config(),
                }),
                app_params: QueryFilter { query_id },
            })
            .unwrap();
        let non_matching = analyzer
            .single_resource_timeline(SingleTimelineRequest {
                entry: TimelineRequest::Resource(ResourceTimelineRequest {
                    resource_id: executor_id,
                    long_entities_threshold_s: None,
                    entity_filter: EntityFilter {
                        entity_type_name: None,
                    },
                    application: TaskFilter {
                        pipeline_uuid: None,
                        current_operator_id: Some(7),
                    },
                    config: timeline_config(),
                }),
                app_params: QueryFilter { query_id },
            })
            .unwrap();

        let UiResourceTimeline::Binned(matching_data) = matching.data else {
            panic!("expected plain binned timeline");
        };
        let UiResourceTimeline::Binned(non_matching_data) = non_matching.data else {
            panic!("expected plain binned timeline");
        };
        let matching_sum = matching_data.capacities_values["unit"].iter().sum::<f64>();
        let non_matching_sum = non_matching_data
            .capacities_values
            .get("unit")
            .map(|values| values.iter().sum::<f64>())
            .unwrap_or_default();
        assert!(matching_sum > 0.0);
        assert_eq!(non_matching_sum, 0.0);
    }
}
