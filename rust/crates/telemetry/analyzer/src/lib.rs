use instrumentation_model::{Sirius, SiriusEvent};
use quent_events::Event;
pub use quent_query_engine_analyzer::QueryEngineModel;
use quent_query_engine_analyzer::entities;
use quent_query_engine_analyzer::ui::{QuentViewer, UiAnalyzer, ViewerEventStream};
use quent_query_engine_ui::{OperatorFilter, QueryBundle, QueryEntities, QueryFilter};
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
use std::collections::HashMap as StdHashMap;
use std::sync::Arc;
use tracing::debug;

use quent_analyzer::{
    AnalyzerError, AnalyzerResult, Entity, Model, Span,
    fsm::{FsmTypeDeclaration, FsmUsages, collection::FsmCollection},
    resource::{
        ResourceGroup, ResourceTypeDecl, Usage, Using, collection::ResourceCollection,
        tree::ResourceTreeNode,
    },
    timeline::binned::resource::{
        ResourceTimeline, ResourceTimelineBuilder, ResourceTimelineByKey,
        ResourceTimelineByKeyBuilder,
    },
};
use quent_simulator_ui::EntityRef;
use quent_time::{SpanNanoSec, TimeNanoSec, TimeUnixNanoSec, to_nanosecs, to_secs};
use uuid::Uuid;

use crate::{
    data_batch::{DataBatch, DataBatchExt},
    model::{SiriusModel, SiriusModelBuilder},
    task::{Task, TaskExt},
    view::SiriusModelQueryView,
};

pub mod data_batch;
pub mod model;
pub mod task;
pub mod view;

const TASK_TYPE_NAME: &str = "task";
const DATA_BATCH_TYPE_NAME: &str = "data_batch";

/// `quent-open` viewer entry: renders Sirius events with [`SiriusUiAnalyzer`].
pub struct Viewer;

impl QuentViewer for Viewer {
    type Analyzer = SiriusUiAnalyzer;

    fn import_events(
        dir: &std::path::Path,
    ) -> quent_model::io::ImporterResult<ViewerEventStream<Self::Analyzer>> {
        Sirius::import_events(dir)
    }
}

pub struct SiriusUiAnalyzer {
    pub model: SiriusModel,
}

struct PlainBuilderSlot<'a> {
    entry_id: String,
    config_idx: usize,
    builder: ResourceTimelineBuilder<'a>,
    resource_id_filter: Arc<HashSet<Uuid>>,
    op_filter: OperatorFilter,
}

struct PerStateBuilderSlot<'a> {
    entry_id: String,
    config_idx: usize,
    builder: ResourceTimelineByKeyBuilder<'a, &'a str>,
    resource_id_filter: Arc<HashSet<Uuid>>,
    op_filter: OperatorFilter,
    entity_type_name: String,
}

/// Adapts the model's task map to the [`FsmCollection`] contract that
/// [`entities::list_entities`] ranks and pages over.
struct TaskCollection<'a>(&'a HashMap<Uuid, Task>);

impl FsmCollection for TaskCollection<'_> {
    type Fsm = Task;

    fn fsms(&self) -> impl Iterator<Item = &Task> {
        self.0.values()
    }
}

/// Adapts the model's data-batch map to the [`FsmCollection`] contract that
/// [`entities::list_entities`] ranks and pages over.
struct DataBatchCollection<'a>(&'a HashMap<Uuid, DataBatch>);

impl FsmCollection for DataBatchCollection<'_> {
    type Fsm = DataBatch;

    fn fsms(&self) -> impl Iterator<Item = &DataBatch> {
        self.0.values()
    }
}

impl UiAnalyzer for SiriusUiAnalyzer {
    type Event = SiriusEvent;
    type EntityRef = EntityRef;

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

        let qe = &model.query_engine;
        tracing::info!(
            workers = qe.workers.len(),
            query_groups = qe.query_groups.len(),
            queries = qe.queries.len(),
            plans = qe.plans.len(),
            operators = qe.operators.len(),
            ports = qe.ports.len(),
            resources = model.arbitrary_resources.resources.len(),
            resource_groups = model.arbitrary_resources.resource_groups.len(),
            resource_types = model.arbitrary_resources.resource_types.len(),
            resource_group_types = model.resource_group_types.len(),
            tasks = model.tasks.len(),
        );

        Ok(Self { model })
    }

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

    fn query_bundle(&self, query_id: Uuid) -> AnalyzerResult<QueryBundle<EntityRef>> {
        debug!("constructing view");
        // TODO(johanpel): A query view could be cached in an analyzer so
        // subsequent calls into the analyzer for that query could benefit from
        // it.
        let view = self.model.query_view(query_id)?;
        let query = self.model.query(query_id)?;
        let start_time_unix_ns = view.query_epoch(query_id)?;
        let duration_s = to_secs(query.span()?.duration());
        let epoch = view.query_epoch(query_id)?;

        debug!("converting query engine model entities");
        let engine = view.engine()?.to_ui()?;
        let query_group_id = query.query_group_id().ok_or_else(|| {
            quent_analyzer::AnalyzerError::IncompleteEntity(format!(
                "query {} has no query_group_id",
                query_id
            ))
        })?;
        let query_group = view.query_group(query_group_id)?.to_ui();
        let query = query.to_ui()?;
        let workers = view.workers().map(|w| (w.id(), w.to_ui(epoch))).collect();
        let plans = view.plans().map(|p| (p.id(), p.to_ui())).collect();
        let operators = view.operators().map(|o| (o.id(), o.to_ui(epoch))).collect();
        let ports = view.ports().map(|p| (p.id(), p.to_ui(epoch))).collect();
        let unique_operator_names = view
            .operators()
            .filter_map(|v| v.operator_type_name().map(|s| s.to_owned()))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        debug!("converting Sirius runtime resource entities");

        let resources = view
            .runtime_resources()
            .map(|resource| (resource.id(), resource.into()))
            .collect();

        let resource_groups = view
            .runtime_resource_groups()
            .map(|group| {
                let group: &dyn ResourceGroup = group;
                (group.id(), group.into())
            })
            .collect();

        let resource_types = view
            .runtime_resource_types()
            .map(|(name, resource_type)| (name.to_string(), resource_type.into()))
            .collect();

        let resource_group_types = view
            .runtime_resource_group_types()
            .map(|(name, group_type)| (name.to_string(), group_type.into()))
            .collect();

        let task_decl = Task::fsm_type_declaration();
        let data_batch_decl = DataBatch::fsm_type_declaration();
        let fsm_types = [
            (task_decl.name.clone(), task_decl),
            (data_batch_decl.name.clone(), data_batch_decl),
        ]
        .into_iter()
        .collect();

        let entities = QueryEntities {
            engine,
            query_group,
            query,
            workers,
            plans,
            operators,
            ports,
            resource_types,
            resources,
            resource_groups,
            resource_group_types,
            fsm_types,
        };

        debug!("deriving plan tree");
        let plan_tree = view.plan_tree(query_id)?.to_ui();

        debug!("deriving resource tree");
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
                ("capacity_bytes".into(), QuantitySpec::bytes()),
                ("capacity_entries".into(), QuantitySpec::unit()),
                ("unit".into(), QuantitySpec::unit()),
            ]
            .into(),
            start_time_unix_ns,
            duration_s,
        })
    }

    fn query_engine_model(&self) -> &impl QueryEngineModel {
        &self.model
    }

    fn list_entities(
        &self,
        request: quent_ui::entities::request::EntityListRequest<QueryFilter, OperatorFilter>,
    ) -> AnalyzerResult<quent_ui::entities::response::EntityListResponse> {
        let query_id = request.app_params.query_id;
        let epoch = self.query_engine_model().query_epoch(query_id)?;
        let entry = request.entry;
        let window = entry.window.try_into_span(epoch)?;
        let scope = entry
            .filter
            .scope
            .as_ref()
            .map(|s| s.resolve(&self.model))
            .transpose()?;
        let operator_filter = entry.application;

        // Restrict candidates to the requested query: an entity belongs to a
        // query iff its (producer) pipeline is one of that query's operators.
        // Without this, entities from a different query sharing a resource and
        // overlapping the window would leak in.
        let query_operators: HashSet<Uuid> = self
            .model
            .query_view(query_id)?
            .operators()
            .map(|op| op.id())
            .collect();

        let query = entities::ListQuery {
            scope: scope.as_ref(),
            window,
            filter: &entry.filter,
            sort: entry.sort,
            page: entry.page,
            epoch,
        };

        // The model holds two distinct FSM types (tasks and data batches), so
        // dispatch on the requested entity type rather than a single collection.
        // Absent an explicit type, default to tasks.
        match entry.filter.entity_type_name.as_deref() {
            Some(DATA_BATCH_TYPE_NAME) => entities::list_entities(
                &DataBatchCollection(&self.model.data_batches),
                |data_batch| {
                    data_batch
                        .producer_pipeline_uuid()
                        .is_some_and(|op| query_operators.contains(&op))
                        && data_batch.matches_filter(&operator_filter)
                },
                query,
            ),
            _ => entities::list_entities(
                &TaskCollection(&self.model.tasks),
                |task| {
                    task.pipeline_uuid()
                        .is_some_and(|op| query_operators.contains(&op))
                        && task.matches_filter(&operator_filter)
                },
                query,
            ),
        }
    }

    // TODO(johanpel): consider reusing the bulk request API with a single entry for requests like this.
    fn single_resource_timeline(
        &self,
        request: SingleTimelineRequest<QueryFilter, OperatorFilter>,
    ) -> AnalyzerResult<SingleTimelineResponse> {
        // TODO(johanpel): we may want to sanity-check whether the requested
        // resource/group is actually in the resource tree for a given query.

        // Calculate this ASAP to help fail quickly.
        let view = self.model.query_view(request.app_params.query_id)?;
        let epoch = view.query_epoch(request.app_params.query_id)?;
        let config = request.entry.config().try_into_binned_span(epoch)?;
        let config_secs = config.try_to_secs_relative(epoch)?;

        match request.entry {
            TimelineRequest::Resource(req) => {
                let resource_type = view.resource_type_of(req.resource_id)?;
                let long_entities_threshold = req.long_entities_threshold_s.map(to_nanosecs);
                let fsm_filter = req.application;

                if req.entity_filter.entity_type_name.is_some() {
                    let mut builder = ResourceTimelineByKeyBuilder::try_new(
                        resource_type,
                        config,
                        long_entities_threshold,
                    )?;

                    match req.entity_filter.entity_type_name.as_deref() {
                        Some(TASK_TYPE_NAME) => {
                            self.populate_keyed_builder(
                                &mut builder,
                                self.filtered_tasks(
                                    &view,
                                    req.entity_filter,
                                    &fsm_filter,
                                    config.span,
                                )?
                                .into_iter()
                                .filter(|task| {
                                    task.usages()
                                        .any(|usage| usage.resource_id() == req.resource_id)
                                }),
                                |id| id == req.resource_id,
                            )?;
                        }
                        Some(DATA_BATCH_TYPE_NAME) => {
                            self.populate_keyed_builder(
                                &mut builder,
                                self.filtered_data_batches(
                                    &view,
                                    req.entity_filter,
                                    &fsm_filter,
                                    config.span,
                                )?
                                .into_iter()
                                .filter(|db| {
                                    db.usages().any(|u| u.resource_id() == req.resource_id)
                                }),
                                |id| id == req.resource_id,
                            )?;
                        }
                        other => {
                            Err(AnalyzerError::InvalidArgument(format!(
                                "{:?} is not a known entity type in this model",
                                other
                            )))?;
                        }
                    }
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

                    builder.try_extend(
                        self.filtered_tasks(
                            &view,
                            req.entity_filter.clone(),
                            &fsm_filter,
                            config.span,
                        )?
                        .into_iter()
                        .flat_map(|task| task.usages())
                        .filter(|usage| usage.resource_id() == req.resource_id),
                    )?;
                    builder.try_extend(
                        self.filtered_data_batches(
                            &view,
                            req.entity_filter,
                            &fsm_filter,
                            config.span,
                        )?
                        .into_iter()
                        .flat_map(|db| db.usages())
                        .filter(|usage| usage.resource_id() == req.resource_id),
                    )?;
                    Ok(SingleTimelineResponse {
                        config: config_secs,
                        data: self.timeline_to_ui(builder.build(), epoch)?,
                    })
                }
            }
            TimelineRequest::ResourceGroup(req) => {
                let resource_type = view.resource_type(&req.resource_type_name)?;
                let long_entities_threshold = req.long_entities_threshold_s.map(to_nanosecs);
                let fsm_filter = req.app_params;

                // Build the resource tree for this group
                let tree = ResourceTreeNode::try_new(&view, req.resource_group_id)?;
                // Collect all leaf resource IDs of the requested type in the tree
                let resource_ids: HashSet<Uuid> = tree
                    .iter_leaf_ids()
                    .filter(|&id| {
                        view.resource(id)
                            .ok()
                            .map(|r| r.type_name() == resource_type.name.as_str())
                            .unwrap_or(false)
                    })
                    .collect();

                if req.entity_filter.entity_type_name.is_some() {
                    let mut builder = ResourceTimelineByKeyBuilder::try_new(
                        resource_type,
                        config,
                        long_entities_threshold,
                    )?;

                    match req.entity_filter.entity_type_name.as_deref() {
                        Some(TASK_TYPE_NAME) => {
                            self.populate_keyed_builder(
                                &mut builder,
                                self.filtered_tasks(
                                    &view,
                                    req.entity_filter,
                                    &fsm_filter,
                                    config.span,
                                )?
                                .into_iter()
                                .filter(|task| {
                                    task.usages()
                                        .any(|usage| resource_ids.contains(&usage.resource_id()))
                                }),
                                |id| resource_ids.contains(&id),
                            )?;
                        }
                        Some(DATA_BATCH_TYPE_NAME) => {
                            self.populate_keyed_builder(
                                &mut builder,
                                self.filtered_data_batches(
                                    &view,
                                    req.entity_filter,
                                    &fsm_filter,
                                    config.span,
                                )?
                                .into_iter()
                                .filter(|db| {
                                    db.usages()
                                        .any(|usage| resource_ids.contains(&usage.resource_id()))
                                }),
                                |id| resource_ids.contains(&id),
                            )?;
                        }
                        other => {
                            Err(AnalyzerError::InvalidArgument(format!(
                                "{:?} is not a known entity type in this model",
                                other
                            )))?;
                        }
                    }

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
                    builder.try_extend(
                        self.filtered_tasks(
                            &view,
                            req.entity_filter.clone(),
                            &fsm_filter,
                            config.span,
                        )?
                        .into_iter()
                        .flat_map(|task| task.usages())
                        .filter(|usage| resource_ids.contains(&usage.resource_id())),
                    )?;
                    builder.try_extend(
                        self.filtered_data_batches(
                            &view,
                            req.entity_filter,
                            &fsm_filter,
                            config.span,
                        )?
                        .into_iter()
                        .flat_map(|db| db.usages())
                        .filter(|usage| resource_ids.contains(&usage.resource_id())),
                    )?;
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
        request: BulkTimelineRequest<QueryFilter, OperatorFilter>,
    ) -> AnalyzerResult<BulkTimelinesResponse> {
        // Calculate this ASAP to help fail quickly.
        let epoch = self
            .query_engine_model()
            .query_epoch(request.app_params.query_id)?;

        // Construct a query view.
        let view = self.model.query_view(request.app_params.query_id)?;
        // Prepare resource tree, we'll reuse this as it is potentially
        // expensive to build for every entry.
        let resource_tree = view.resource_tree()?;

        // Prepare builders, resource id filters, and operator filters, one for
        // each bulk entry. After populating this, we'll build a reverse index,
        // that maps a resource_id to a list of indices in these vecs, for which
        // that resource's usages are relevant.
        let mut plain_builders: Vec<(
            String,
            ResourceTimelineBuilder,
            HashSet<Uuid>,
            OperatorFilter,
        )> = Vec::new();

        // Prepare them also for keyed builders (building by state).
        let mut per_state_builders: Vec<(
            String,
            ResourceTimelineByKeyBuilder<&str>,
            HashSet<Uuid>,
            OperatorFilter,
            String, // entity_type_name this slot breaks down by ("task" | "data_batch", etc.)
        )> = Vec::new();

        for (entry_id, entry) in request.entries {
            let entry_config = entry.config().try_into_binned_span(epoch)?;
            let BulkEntryPrep {
                resource_type,
                resource_id_filter,
                entity_filter,
                task_filter,
                long_entities_threshold,
            } = self.try_prepare_bulk_entry(&view, entry, &resource_tree)?;
            if let Some(entity_type_name) = entity_filter.entity_type_name {
                per_state_builders.push((
                    entry_id,
                    ResourceTimelineByKeyBuilder::try_new(
                        resource_type,
                        entry_config,
                        long_entities_threshold,
                    )?,
                    resource_id_filter,
                    task_filter,
                    entity_type_name,
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
                ));
            }
        }

        // Build reverse index so given the id of an entry in the request, we
        // can quickly look up all builders associated with the entry into which
        // we can push a usage.
        //
        // This is more efficient than going over all usages for each builder,
        // since the number of usages is typically going to be MUCH larger than
        // the number of builders.
        let plain_index: HashMap<Uuid, Vec<usize>> = plain_builders
            .iter()
            .enumerate()
            .flat_map(|(builders_index, builder)| {
                builder
                    .2
                    .iter()
                    .map(move |&resource_id| (resource_id, builders_index))
            })
            .fold(
                HashMap::default(),
                |mut acc, (resource_id, builders_index)| {
                    acc.entry(resource_id).or_default().push(builders_index);
                    acc
                },
            );
        let per_state_index: HashMap<Uuid, Vec<usize>> = per_state_builders
            .iter()
            .enumerate()
            .flat_map(|(builders_index, builder)| {
                builder
                    .2
                    .iter()
                    .map(move |&resource_id| (resource_id, builders_index))
            })
            .fold(
                HashMap::default(),
                |mut acc, (resource_id, builders_index)| {
                    acc.entry(resource_id).or_default().push(builders_index);
                    acc
                },
            );

        // Iterate over all usages once and push any usages of resources in our
        // lookup table to their respective builders. For now we only have
        // tasks.
        for task in view.tasks() {
            for usage in task.usages() {
                let resource_id = usage.resource_id();
                if let Some(builder_indices) = plain_index.get(&resource_id) {
                    for &builder_idx in builder_indices {
                        let builder = &mut plain_builders[builder_idx];
                        if task.matches_filter(&builder.3) {
                            builder.1.try_push(&usage)?;
                        }
                    }
                }
            }

            for (state_name, usage) in task.usages_with_state_names() {
                let resource_id = usage.resource_id();
                if let Some(builder_indices) = per_state_index.get(&resource_id) {
                    for &builder_idx in builder_indices {
                        let builder = &mut per_state_builders[builder_idx];
                        if builder.4 == TASK_TYPE_NAME && task.matches_filter(&builder.3) {
                            builder.1.try_push(state_name, &usage)?;
                        }
                    }
                }
            }
        }

        for data_batch in view.data_batches() {
            for usage in data_batch.usages() {
                let resource_id = usage.resource_id();
                if let Some(builder_indices) = plain_index.get(&resource_id) {
                    for &builder_idx in builder_indices {
                        let builder = &mut plain_builders[builder_idx];
                        if data_batch.matches_filter(&builder.3) {
                            builder.1.try_push(&usage)?;
                        }
                    }
                }
            }

            for (state_name, usage) in data_batch.usages_with_state_names() {
                let resource_id = usage.resource_id();
                if let Some(builder_indices) = per_state_index.get(&resource_id) {
                    for &builder_idx in builder_indices {
                        let builder = &mut per_state_builders[builder_idx];
                        if builder.4 == DATA_BATCH_TYPE_NAME
                            && data_batch.matches_filter(&builder.3)
                        {
                            builder.1.try_push(state_name, &usage)?;
                        }
                    }
                }
            }
        }

        // Collect results for all requests.
        let mut entries = std::collections::HashMap::default();
        for (entry_id, builder, _, _) in plain_builders {
            let built = builder.build();
            let config = built.config.try_to_secs_relative(epoch)?;
            entries.insert(
                entry_id,
                BulkTimelinesResponseEntry::Ok {
                    message: String::new(),
                    config,
                    data: self.timeline_to_ui(built, epoch)?,
                },
            );
        }
        for (key, builder, _, _, _) in per_state_builders {
            let built = builder.build();
            let config = built.config.try_to_secs_relative(epoch)?;
            entries.insert(
                key,
                BulkTimelinesResponseEntry::Ok {
                    message: String::new(),
                    config,
                    data: self.timeline_to_ui_keyed(built, epoch)?,
                },
            );
        }

        Ok(BulkTimelinesResponse { entries })
    }

    fn bulk_chunked_resource_timeline(
        &self,
        request: BulkChunkedTimelineRequest<QueryFilter, OperatorFilter>,
    ) -> AnalyzerResult<BulkChunkedTimelinesResponse> {
        let epoch = self
            .query_engine_model()
            .query_epoch(request.app_params.query_id)?;
        let view = self.model.query_view(request.app_params.query_id)?;
        let resource_tree = view.resource_tree()?;

        let n_configs = request.configs.len();

        let mut plain_builders: Vec<PlainBuilderSlot<'_>> =
            Vec::with_capacity(request.entries.len() * n_configs);
        let mut per_state_builders: Vec<PerStateBuilderSlot<'_>> =
            Vec::with_capacity(request.entries.len() * n_configs);

        // Per-entry prep runs once; the builders for that entry's N configs all share it.
        for (entry_id, entry) in &request.entries {
            let BulkEntryPrep {
                resource_type,
                resource_id_filter,
                entity_filter,
                task_filter,
                long_entities_threshold,
            } = self.try_prepare_bulk_entry(&view, entry.clone(), &resource_tree)?;

            // Wrap the filter once so per-config slots share one allocation.
            let resource_id_filter = Arc::new(resource_id_filter);

            for (config_idx, config) in request.configs.iter().enumerate() {
                let entry_config = config.try_into_binned_span(epoch)?;
                if let Some(type_name) = entity_filter.entity_type_name.clone() {
                    per_state_builders.push(PerStateBuilderSlot {
                        entry_id: entry_id.clone(),
                        config_idx,
                        builder: ResourceTimelineByKeyBuilder::try_new(
                            resource_type,
                            entry_config,
                            long_entities_threshold,
                        )?,
                        resource_id_filter: Arc::clone(&resource_id_filter),
                        op_filter: task_filter.clone(),
                        entity_type_name: type_name,
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
                        op_filter: task_filter.clone(),
                    });
                }
            }
        }

        let plain_index: HashMap<Uuid, Vec<usize>> = plain_builders
            .iter()
            .enumerate()
            .flat_map(|(builder_idx, slot)| {
                slot.resource_id_filter
                    .iter()
                    .map(move |&resource_id| (resource_id, builder_idx))
            })
            .fold(HashMap::default(), |mut acc, (resource_id, builder_idx)| {
                acc.entry(resource_id).or_default().push(builder_idx);
                acc
            });
        let per_state_index: HashMap<Uuid, Vec<usize>> = per_state_builders
            .iter()
            .enumerate()
            .flat_map(|(builder_idx, slot)| {
                slot.resource_id_filter
                    .iter()
                    .map(move |&resource_id| (resource_id, builder_idx))
            })
            .fold(HashMap::default(), |mut acc, (resource_id, builder_idx)| {
                acc.entry(resource_id).or_default().push(builder_idx);
                acc
            });

        // Single pass over all tasks/usages — the dominant cost — dispatched to
        // every matching (entry, config) builder. Builders filter by their own
        // span internally, so out-of-window usages are no-ops.
        for task in view.tasks() {
            for usage in task.usages() {
                let resource_id = usage.resource_id();
                if let Some(builder_indices) = plain_index.get(&resource_id) {
                    for &builder_idx in builder_indices {
                        let slot = &mut plain_builders[builder_idx];
                        if task.matches_filter(&slot.op_filter) {
                            slot.builder.try_push(&usage)?;
                        }
                    }
                }
            }
            for (state_name, usage) in task.usages_with_state_names() {
                let resource_id = usage.resource_id();
                if let Some(builder_indices) = per_state_index.get(&resource_id) {
                    for &builder_idx in builder_indices {
                        let slot = &mut per_state_builders[builder_idx];
                        if slot.entity_type_name == TASK_TYPE_NAME
                            && task.matches_filter(&slot.op_filter)
                        {
                            slot.builder.try_push(state_name, &usage)?;
                        }
                    }
                }
            }
        }

        for data_batch in view.data_batches() {
            for usage in data_batch.usages() {
                let resource_id = usage.resource_id();
                if let Some(builder_indices) = plain_index.get(&resource_id) {
                    for &builder_idx in builder_indices {
                        let slot = &mut plain_builders[builder_idx];
                        if data_batch.matches_filter(&slot.op_filter) {
                            slot.builder.try_push(&usage)?;
                        }
                    }
                }
            }
            for (state_name, usage) in data_batch.usages_with_state_names() {
                let resource_id = usage.resource_id();
                if let Some(builder_indices) = per_state_index.get(&resource_id) {
                    for &builder_idx in builder_indices {
                        let slot = &mut per_state_builders[builder_idx];
                        if slot.entity_type_name == DATA_BATCH_TYPE_NAME
                            && data_batch.matches_filter(&slot.op_filter)
                        {
                            slot.builder.try_push(state_name, &usage)?;
                        }
                    }
                }
            }
        }

        // Reassemble per-entry Vec aligned with `request.configs` order. Slots
        // start as `None` and must all be filled by the end — every (entry,
        // config_idx) had a builder, and every builder produces an `Ok`.
        let mut slots: HashMap<String, Vec<Option<BulkTimelinesResponseEntry>>> = request
            .entries
            .keys()
            .map(|k| (k.clone(), (0..n_configs).map(|_| None).collect()))
            .collect();

        for slot in plain_builders {
            let built = slot.builder.build();
            let config = built.config.try_to_secs_relative(epoch)?;
            let resp = BulkTimelinesResponseEntry::Ok {
                message: String::new(),
                config,
                data: self.timeline_to_ui(built, epoch)?,
            };
            slots.get_mut(&slot.entry_id).unwrap_or_else(|| {
                panic!("known key, instead found unknown key {}", slot.entry_id)
            })[slot.config_idx] = Some(resp);
        }
        for slot in per_state_builders {
            let built = slot.builder.build();
            let config = built.config.try_to_secs_relative(epoch)?;
            let resp = BulkTimelinesResponseEntry::Ok {
                message: String::new(),
                config,
                data: self.timeline_to_ui_keyed(built, epoch)?,
            };
            slots.get_mut(&slot.entry_id).unwrap_or_else(|| {
                panic!("known key, instead found unknown key {}", slot.entry_id)
            })[slot.config_idx] = Some(resp);
        }

        let entries = slots
            .into_iter()
            .map(|(k, v)| {
                let v = v
                    .into_iter()
                    .map(|opt| {
                        opt.ok_or(AnalyzerError::BrokenImpl(
                            "chunked bulk: missing builder slot",
                        ))
                    })
                    .collect::<AnalyzerResult<Vec<_>>>()?;
                Ok((k, v))
            })
            .collect::<AnalyzerResult<std::collections::HashMap<_, _>>>()?;

        Ok(BulkChunkedTimelinesResponse { entries })
    }
}

impl SiriusUiAnalyzer {
    /// Return an iterator over all tasks, filtered by time window and operator id.
    fn filtered_tasks<'a>(
        &self,
        view: &'a SiriusModelQueryView<'a>,
        entity_filter: EntityFilter,
        task_filter: &OperatorFilter,
        time_window: SpanNanoSec,
    ) -> AnalyzerResult<Vec<&'a Task>> {
        if let Some(entity_type_name) = entity_filter.entity_type_name
            && entity_type_name != TASK_TYPE_NAME
        {
            return Err(AnalyzerError::InvalidArgument(format!(
                "{entity_type_name} is not a known entity type in this model"
            )));
        }

        Ok(view
            .tasks()
            .filter(|task| task.span().is_ok_and(|s| s.intersects(&time_window)))
            .filter(|task| task.matches_filter(task_filter))
            .collect())
    }

    fn filtered_data_batches<'a>(
        &self,
        view: &'a SiriusModelQueryView<'a>,
        entity_filter: EntityFilter,
        filter: &OperatorFilter,
        time_window: SpanNanoSec,
    ) -> AnalyzerResult<Vec<&'a DataBatch>> {
        if let Some(entity_type_name) = entity_filter.entity_type_name
            && entity_type_name != DATA_BATCH_TYPE_NAME
        {
            return Err(AnalyzerError::InvalidArgument(format!(
                "{entity_type_name} is not a known entity type in this model"
            )));
        }
        Ok(view
            .data_batches()
            .filter(|db| db.span().is_ok_and(|s| s.intersects(&time_window)))
            .filter(|db| db.matches_filter(filter))
            .collect())
    }

    /// Given a TimelineRequest figure out what are:
    /// - The resource_type
    /// - For groups, the set of resources to aggregate for.
    /// - Whether this is a request to split out usage per state.
    /// - What operator ID filter to apply.
    /// - What the threshold is for long entities.
    fn try_prepare_bulk_entry<'a>(
        &self,
        view: &'a SiriusModelQueryView<'a>,
        request: TimelineRequest<OperatorFilter>,
        tree: &ResourceTreeNode,
    ) -> AnalyzerResult<BulkEntryPrep<'a>> {
        Ok(match request {
            TimelineRequest::Resource(r) => BulkEntryPrep {
                resource_type: view.resource_type_of(r.resource_id)?,
                resource_id_filter: [r.resource_id].into_iter().collect(),
                entity_filter: r.entity_filter,
                task_filter: r.application,
                long_entities_threshold: r.long_entities_threshold_s.map(to_nanosecs),
            },
            TimelineRequest::ResourceGroup(rg) => {
                let resource_type = view.resource_type(&rg.resource_type_name)?;
                let subtree = tree
                    .find(rg.resource_group_id)
                    .ok_or(AnalyzerError::InvalidId(rg.resource_group_id))?;
                let resource_ids: HashSet<Uuid> = subtree
                    .iter_leaf_ids()
                    .filter(|&id| {
                        view.resource(id)
                            .ok()
                            .is_some_and(|r| r.type_name() == resource_type.name.as_str())
                    })
                    .collect();
                BulkEntryPrep {
                    resource_type,
                    resource_id_filter: resource_ids,
                    entity_filter: rg.entity_filter,
                    task_filter: rg.app_params,
                    long_entities_threshold: rg.long_entities_threshold_s.map(to_nanosecs),
                }
            }
        })
    }

    /// Populate a keyed resource timeline builder with tasks.
    fn populate_keyed_builder<'a, F>(
        &self,
        builder: &mut ResourceTimelineByKeyBuilder<'a, &'a str>,
        fsms: impl Iterator<Item = &'a F>,
        resource_filter: impl Fn(Uuid) -> bool,
    ) -> AnalyzerResult<()>
    where
        F: FsmUsages<'a> + 'a,
    {
        for fsm in fsms {
            for (state_name, usage) in fsm.usages_with_state_names() {
                if resource_filter(usage.resource_id()) {
                    builder.try_push(state_name, &usage)?;
                }
            }
        }
        Ok(())
    }

    /// Turn a list of entity ids into UI-compatible FSM data.
    fn task_entities_to_ui_fsm(
        &self,
        entity_ids: &[Uuid],
        epoch: TimeUnixNanoSec,
    ) -> AnalyzerResult<Vec<FiniteStateMachine>> {
        entity_ids
            .iter()
            .filter_map(|&id| {
                if let Some(task) = self.model.tasks.get(&id) {
                    let pipeline_name = task
                        .pipeline_uuid()
                        .and_then(|id| self.model.query_engine.operators.get(&id))
                        .map(|operator| operator.instance_name());
                    Some(task.try_to_ui_fsm(epoch, pipeline_name))
                } else {
                    self.model
                        .data_batches
                        .get(&id)
                        .map(|db| db.try_to_ui_fsm(epoch))
                }
            })
            .collect()
    }

    /// Convert a timeline to a UI-compatible one.
    fn timeline_to_ui(
        &self,
        result: ResourceTimeline,
        epoch: TimeUnixNanoSec,
    ) -> AnalyzerResult<UiResourceTimeline> {
        let config = result.config.try_to_secs_relative(epoch)?;
        let capacities_values = result
            .data
            .into_iter()
            .map(|(k, v)| (k.to_owned(), v))
            .collect();
        let long_fsms = self.task_entities_to_ui_fsm(&result.long_entities, epoch)?;
        Ok(UiResourceTimeline::Binned(ResourceTimelineBinned {
            config,
            capacities_values,
            long_fsms,
        }))
    }

    /// Convert a keyed timeline to a UI-compatible one.
    fn timeline_to_ui_keyed(
        &self,
        result: ResourceTimelineByKey<&str>,
        epoch: TimeUnixNanoSec,
    ) -> AnalyzerResult<UiResourceTimeline> {
        let config = result.config.try_to_secs_relative(epoch)?;
        let mut capacities_states_values = StdHashMap::new();
        for ((state_name, capacity_name), values) in result.data {
            capacities_states_values
                .entry(capacity_name.to_owned())
                .or_insert_with(StdHashMap::new)
                .insert(state_name.to_owned(), values);
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

/// Helper struct to build bulk timeline responses.
struct BulkEntryPrep<'a> {
    resource_type: &'a ResourceTypeDecl,
    resource_id_filter: HashSet<Uuid>,
    entity_filter: EntityFilter,
    task_filter: OperatorFilter,
    long_entities_threshold: Option<TimeNanoSec>,
}
