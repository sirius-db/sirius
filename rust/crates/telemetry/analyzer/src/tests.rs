// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for Batch ingestion and the data-flow distribution timeline.

use quent_analyzer::{AnalyzerError, fsm::Fsm, resource::collection::ResourceCollection};
use quent_events::{EntityRef, Event};
use quent_io::{ExporterOptions, FileSystemExporterOptions, FileSystemFormat};
use quent_query_engine_analyzer::{
    QueryEngineModel, model::QueryEvent as NormalizedQueryEvent, ui::UiAnalyzer,
};
use quent_query_engine_ui::{OperatorFilter, QueryFilter};
use quent_store::event::{ModelEventStore, filesystem::Store};
use quent_ui::entities::request::{
    EntityListEntry, EntityListFilter, EntityListRequest, EntityScope, EntitySortKey, Sort,
    SortDir, TimeWindow,
};
use quent_ui::timeline::{
    categorical::CategoricalTimelineRequest,
    request::{
        EntityFilter, ResourceTimelineRequest, SingleTimelineRequest, TimelineConfig,
        TimelineRequest,
    },
    response::ResourceTimeline as UiResourceTimeline,
};
use sirius_telemetry_store::{self as schema, SiriusEvent};
use uuid::Uuid;

use crate::{SiriusUiAnalyzer, batch_placement::BatchPlacementExt, model::SiriusModelBuilder};

/// Nanoseconds; also the timestamp of the query's first transition, so the
/// query epoch. All batch timestamps below are relative to this.
const EPOCH: u64 = 1_000_000_000_000_000;

struct Fixture {
    engine_id: Uuid,
    query_id: Uuid,
    op1_id: Uuid,
    op2_id: Uuid,
    gpu_id: Uuid,
    host_id: Uuid,
    disk_id: Uuid,
    batch_a_id: Uuid,
    task_1_id: Uuid,
    events: Vec<Event<SiriusEvent>>,
}

fn tier_usage(
    resource_id: Uuid,
    bytes: u64,
) -> Option<EntityRef<schema::MemoryTier, schema::MemoryTierUsage>> {
    Some(EntityRef::new(
        resource_id,
        schema::MemoryTierUsage { bytes },
    ))
}

fn batch_event(id: Uuid, ts: u64, state: schema::BatchPlacementEvent) -> Event<SiriusEvent> {
    Event::new(id, ts, SiriusEvent::BatchPlacement(state))
}

fn memory_tier_events(id: Uuid, parent: Uuid, name: &str, bytes: u64) -> Vec<Event<SiriusEvent>> {
    vec![Event::new(
        id,
        EPOCH - 800,
        SiriusEvent::MemoryTier(schema::MemoryTierEvent::Declaration {
            instance_name: name.to_owned(),
            parent_group_id: EntityRef::new(parent, ()),
            bounds: schema::MemoryTierBounds { bytes },
        }),
    )]
}

/// A minimal engine with one query (two operators), the three memory tiers,
/// and optionally batch placements:
///
/// Batch A (pipeline op1, batch_id 7, 1000 bytes), timestamps relative to the
/// query epoch:
/// - t=0    registered on GPU
/// - t=100  queued on GPU
/// - t=300  queued on HOST (tier-change self-transition: spill)
/// - t=500  packaged (task 1) on HOST
/// - t=800  processing (task 1) on GPU
/// - t=1000 consumed ("processed")
///
/// Batch B lives on a pipeline that is *not* an operator of the query and must
/// not appear in the query's data-flow timeline.
fn fixture(with_batches: bool) -> Fixture {
    let engine_id = Uuid::from_u128(0x01);
    let query_group_id = Uuid::from_u128(0x02);
    let query_id = Uuid::from_u128(0x03);
    let plan_id = Uuid::from_u128(0x04);
    let op1_id = Uuid::from_u128(0x05);
    let op2_id = Uuid::from_u128(0x06);
    let gpu_id = Uuid::from_u128(0x07);
    let host_id = Uuid::from_u128(0x08);
    let disk_id = Uuid::from_u128(0x09);
    let batch_a_id = Uuid::from_u128(0x0a);
    let batch_b_id = Uuid::from_u128(0x0b);
    let task_1_id = Uuid::from_u128(0x0c);
    let worker_id = Uuid::from_u128(0x0f);

    let mut events = vec![
        Event::new(
            engine_id,
            EPOCH - 1000,
            SiriusEvent::Engine(schema::EngineEvent::Init {
                implementation: schema::EngineImplementationAttributes {
                    name: Some("Sirius".to_owned()),
                    version: None,
                    custom_attributes: Default::default(),
                },
                instance_name: None,
            }),
        ),
        Event::new(
            query_group_id,
            EPOCH - 900,
            SiriusEvent::QueryGroup(schema::QueryGroupEvent::Declaration {
                instance_name: "qg".to_owned(),
                engine_id: EntityRef::new(engine_id, ()),
            }),
        ),
    ];
    events.extend(memory_tier_events(gpu_id, engine_id, "GPU", 1 << 30));
    events.extend(memory_tier_events(host_id, engine_id, "HOST", 4 << 30));
    events.extend(memory_tier_events(disk_id, engine_id, "DISK", 16 << 30));

    // The query FSM: its first transition is the query epoch.
    events.extend([
        Event::new(
            query_id,
            EPOCH,
            SiriusEvent::Query(schema::QueryEvent::Init {
                seq: 0,
                instance_name: "q".to_owned(),
                query_group_id: EntityRef::new(query_group_id, ()),
            }),
        ),
        Event::new(
            query_id,
            EPOCH + 10,
            SiriusEvent::Query(schema::QueryEvent::Planning { seq: 1 }),
        ),
        Event::new(
            query_id,
            EPOCH + 20,
            SiriusEvent::Query(schema::QueryEvent::Executing { seq: 2 }),
        ),
        Event::new(
            query_id,
            EPOCH + 50_000,
            SiriusEvent::Query(schema::QueryEvent::Exit { seq: 3 }),
        ),
    ]);

    events.push(Event::new(
        worker_id,
        EPOCH - 950,
        SiriusEvent::Worker(schema::WorkerEvent::Init {
            parent_engine_id: EntityRef::new(engine_id, ()),
            instance_name: "worker".to_owned(),
        }),
    ));
    events.push(Event::new(
        plan_id,
        EPOCH + 5,
        SiriusEvent::Plan(schema::PlanEvent::Declaration {
            parent: schema::PlanParent {
                query_id: EntityRef::new(query_id, ()),
                plan_id: None,
            },
            instance_name: "plan".to_owned(),
            edges: vec![],
            worker_id: Some(EntityRef::new(worker_id, ())),
        }),
    ));
    for (op_id, name) in [(op1_id, "op1"), (op2_id, "op2")] {
        events.push(Event::new(
            op_id,
            EPOCH + 6,
            SiriusEvent::Operator(schema::OperatorEvent::Declaration {
                plan_id: EntityRef::new(plan_id, ()),
                parent_operator_ids: vec![],
                instance_name: name.to_owned(),
                type_name: "scan".to_owned(),
                custom_attributes: Default::default(),
            }),
        ));
    }

    if with_batches {
        let port_id = Uuid::from_u128(0x0d);
        events.extend([
            batch_event(
                batch_a_id,
                EPOCH,
                schema::BatchPlacementEvent::BatchRegistered {
                    seq: 0,
                    instance_name: "batch 7".to_owned(),
                    batch_id: 7,
                    pipeline_uuid: EntityRef::new(op1_id, ()),
                    port_uuid: Some(EntityRef::new(port_id, ())),
                    origin: "operator_output".to_owned(),
                    tier: tier_usage(gpu_id, 1000),
                },
            ),
            batch_event(
                batch_a_id,
                EPOCH + 100,
                schema::BatchPlacementEvent::BatchQueued {
                    seq: 1,
                    tier: tier_usage(gpu_id, 1000),
                },
            ),
            // Tier change while queued: spill from GPU to HOST.
            batch_event(
                batch_a_id,
                EPOCH + 300,
                schema::BatchPlacementEvent::BatchQueued {
                    seq: 2,
                    tier: tier_usage(host_id, 1000),
                },
            ),
            batch_event(
                batch_a_id,
                EPOCH + 500,
                schema::BatchPlacementEvent::BatchPackaged {
                    seq: 3,
                    task_uuid: task_1_id,
                    tier: tier_usage(host_id, 1000),
                },
            ),
            batch_event(
                batch_a_id,
                EPOCH + 800,
                schema::BatchPlacementEvent::BatchProcessing {
                    seq: 4,
                    task_uuid: task_1_id,
                    tier: tier_usage(gpu_id, 1000),
                },
            ),
            batch_event(
                batch_a_id,
                EPOCH + 1000,
                schema::BatchPlacementEvent::BatchConsumed {
                    seq: 5,
                    reason: "processed".to_owned(),
                },
            ),
        ]);

        // A batch on a pipeline outside the query.
        let foreign_pipeline = Uuid::from_u128(0x0e);
        events.extend([
            batch_event(
                batch_b_id,
                EPOCH,
                schema::BatchPlacementEvent::BatchRegistered {
                    seq: 0,
                    instance_name: "batch 8".to_owned(),
                    batch_id: 8,
                    pipeline_uuid: EntityRef::new(foreign_pipeline, ()),
                    port_uuid: Some(EntityRef::new(port_id, ())),
                    origin: "operator_output".to_owned(),
                    tier: tier_usage(disk_id, 500),
                },
            ),
            batch_event(
                batch_b_id,
                EPOCH,
                schema::BatchPlacementEvent::BatchQueued {
                    seq: 1,
                    tier: tier_usage(disk_id, 500),
                },
            ),
            batch_event(
                batch_b_id,
                EPOCH + 900,
                schema::BatchPlacementEvent::BatchConsumed {
                    seq: 2,
                    reason: "query_end".to_owned(),
                },
            ),
        ]);
    }

    Fixture {
        engine_id,
        query_id,
        op1_id,
        op2_id,
        gpu_id,
        host_id,
        disk_id,
        batch_a_id,
        task_1_id,
        events,
    }
}

/// Append one task on the op1 pipeline; only preparing/computing carry the
/// `reservation` tier usage (2048 bytes on GPU). Timestamps relative to the
/// query epoch:
/// - t=0    created
/// - t=100  queued (no reservation: contributes nothing to working space)
/// - t=200  reserving (no reservation usage yet)
/// - t=400  preparing, reservation 2048 B on GPU
/// - t=700  computing, reservation 2048 B on GPU
/// - t=900  finalizing (reservation released)
fn add_working_space_task(fixture: &mut Fixture) -> Uuid {
    let task_id = fixture.task_1_id;
    let executor_id = Uuid::from_u128(0x11);
    let task_event =
        |ts: u64, event: schema::TaskEvent| Event::new(task_id, ts, SiriusEvent::Task(event));
    fixture.events.push(Event::new(
        executor_id,
        EPOCH - 500,
        SiriusEvent::ExecutorThread(schema::ExecutorThreadEvent::Declaration {
            instance_name: "executor 1".to_owned(),
            parent_group_id: EntityRef::new(fixture.engine_id, ()),
            engine_id: EntityRef::new(fixture.engine_id, ()),
        }),
    ));
    fixture.events.extend([
        task_event(
            EPOCH,
            schema::TaskEvent::Created {
                seq: 0,
                instance_name: "task 1".to_owned(),
                pipeline_uuid: EntityRef::new(fixture.op1_id, ()),
            },
        ),
        task_event(
            EPOCH + 100,
            schema::TaskEvent::Queued {
                seq: 1,
                queue: None,
            },
        ),
        task_event(
            EPOCH + 200,
            schema::TaskEvent::Reserving {
                seq: 2,
                requested_bytes: 2048,
                input_basis: 1000,
                peak_estimate: 2048,
                bytes_to_materialize: 1000,
                manager_thread: None,
            },
        ),
        task_event(
            EPOCH + 400,
            schema::TaskEvent::Preparing {
                seq: 3,
                origin_tier: "GPU".to_owned(),
                target_tier: "GPU".to_owned(),
                input_bytes: 1000,
                executor_thread: Some(EntityRef::new(executor_id, schema::ExecutorThreadUsage)),
                reservation: tier_usage(fixture.gpu_id, 2048),
            },
        ),
        task_event(
            EPOCH + 700,
            schema::TaskEvent::Computing {
                seq: 4,
                current_operator_id: 0,
                input_bytes: 1000,
                peak_allocated_bytes: 1500,
                executor_thread: Some(EntityRef::new(executor_id, schema::ExecutorThreadUsage)),
                reservation: tier_usage(fixture.gpu_id, 2048),
            },
        ),
        task_event(
            EPOCH + 900,
            schema::TaskEvent::Finalizing {
                seq: 5,
                success: true,
            },
        ),
    ]);
    executor_id
}

fn analyzer(fixture: &mut Fixture) -> SiriusUiAnalyzer {
    let events = std::mem::take(&mut fixture.events);
    SiriusUiAnalyzer::try_new(fixture.engine_id, events.into_iter())
        .expect("analyzer builds from fixture events")
}

#[test]
fn generated_ndjson_round_trips_through_store_and_analyzer() {
    use sirius_telemetry_instrumentation as instrumentation;

    let output = tempfile::tempdir().expect("temporary output directory is created");
    let (context_id, engine_id, query_id) = {
        let context = instrumentation::Context::<instrumentation::Sirius>::try_new(
            ExporterOptions::FileSystem(FileSystemExporterOptions::new(
                FileSystemFormat::Ndjson,
                output.path().to_path_buf(),
            )),
        )
        .expect("NDJSON instrumentation context is created");
        let context_id = context.id();

        let mut engine = context.observer::<instrumentation::Engine>().handle();
        engine
            .init(
                instrumentation::EngineImplementationAttributes {
                    name: Some("Sirius".to_owned()),
                    version: None,
                    custom_attributes: instrumentation::DynamicAttributes::new(),
                },
                Some("round-trip engine".to_owned()),
            )
            .expect("engine init event is emitted");
        let engine_id = engine.uuid();

        let mut query_group = context.observer::<instrumentation::QueryGroup>().handle();
        query_group
            .declaration("round-trip group".to_owned(), engine.as_entity_ref())
            .expect("query-group declaration is emitted");

        let query = context
            .observer::<instrumentation::Query>()
            .handle()
            .init("SELECT 1".to_owned(), query_group.as_entity_ref())
            .planning()
            .executing()
            .exit();
        let query_id = query.uuid();

        engine.exit().expect("engine exit event is emitted");
        (context_id, engine_id, query_id)
    };

    let events = Store::<schema::Sirius>::new(output.path())
        .events(context_id)
        .expect("filesystem event streams are opened")
        .collect::<Result<Vec<_>, _>>()
        .expect("generated NDJSON events are decoded");
    assert_eq!(events.len(), 7);

    let analyzer = SiriusUiAnalyzer::try_new(engine_id, events.into_iter())
        .expect("filesystem events build a Sirius analyzer");
    assert!(analyzer.query_engine_model().engine().is_ok());
    let query = analyzer
        .model
        .query_engine
        .query(query_id)
        .expect("round-trip query is present");
    let terminal_transition = query
        .transition(query.len())
        .expect("round-trip query has a terminal transition");
    assert!(matches!(
        &terminal_transition.data,
        NormalizedQueryEvent::Done { seq: 3 }
    ));
}

/// A request for the full query window [0, 1000) ns in 10 bins of 100 ns.
fn request(query_id: Uuid, measures: &[&str]) -> CategoricalTimelineRequest<QueryFilter> {
    CategoricalTimelineRequest {
        measures: measures.iter().map(|m| m.to_string()).collect(),
        config: TimelineConfig {
            num_bins: 10,
            start: 0.0,
            end: 1e-6,
        },
        app_params: QueryFilter { query_id },
    }
}

#[test]
fn ingests_batches_and_memory_tiers() {
    let mut fixture = fixture(true);
    let analyzer = analyzer(&mut fixture);
    let model = &analyzer.model;

    assert_eq!(model.batch_placements.len(), 2);
    let batch_a = &model.batch_placements[&fixture.batch_a_id];
    assert_eq!(batch_a.batch_id(), Some(7));
    assert_eq!(batch_a.pipeline_uuid(), Some(fixture.op1_id));
    assert_eq!(batch_a.last_task_uuid(), Some(fixture.task_1_id));

    for (id, name, bytes) in [
        (fixture.gpu_id, "GPU", 1 << 30),
        (fixture.host_id, "HOST", 4 << 30),
        (fixture.disk_id, "DISK", 16 << 30),
    ] {
        let resource = model.resource(id).expect("tier resource exists");
        assert_eq!(resource.type_name(), "memory_tier");
        assert_eq!(resource.instance_name(), name);
        let bounds = &model.sirius_resources.resources[&id].bounds().0;
        assert_eq!(bounds.len(), 1);
        assert_eq!(bounds[0].name, "bytes");
        assert_eq!(bounds[0].value, Some(bytes));
    }
    assert!(
        model.sirius_resources.resource_types["memory_tier"]
            .used_by
            .contains("batch_placement")
    );
}

#[test]
fn data_batch_destructed_is_terminal() {
    let mut fixture = fixture(false);
    let data_batch_id = Uuid::from_u128(0x10);
    fixture.events.extend([
        Event::new(
            data_batch_id,
            EPOCH + 100,
            SiriusEvent::DataBatch(schema::DataBatchEvent::Constructed {
                seq: 0,
                instance_name: "batch 16".to_owned(),
                data_batch_id: 16,
                producer_pipeline_uuid: EntityRef::new(fixture.op1_id, ()),
            }),
        ),
        Event::new(
            data_batch_id,
            EPOCH + 200,
            SiriusEvent::DataBatch(schema::DataBatchEvent::Stationary {
                seq: 1,
                memory: None,
            }),
        ),
        Event::new(
            data_batch_id,
            EPOCH + 300,
            SiriusEvent::DataBatch(schema::DataBatchEvent::Destructed { seq: 2 }),
        ),
    ]);

    let analyzer = analyzer(&mut fixture);
    let transitions = analyzer.model.data_batches[&data_batch_id].transitions();
    assert!(matches!(
        transitions.last().map(|transition| &transition.data),
        Some(schema::DataBatchEvent::Destructed { .. })
    ));
}

#[test]
fn nil_application_fsm_ids_return_errors() {
    let engine_id = Uuid::from_u128(1);
    let pipeline_id = Uuid::from_u128(2);
    let events = [
        SiriusEvent::Task(schema::TaskEvent::Created {
            seq: 0,
            instance_name: "task".to_owned(),
            pipeline_uuid: EntityRef::new(pipeline_id, ()),
        }),
        SiriusEvent::DataBatch(schema::DataBatchEvent::Constructed {
            seq: 0,
            instance_name: "data batch".to_owned(),
            data_batch_id: 1,
            producer_pipeline_uuid: EntityRef::new(pipeline_id, ()),
        }),
        SiriusEvent::BatchPlacement(schema::BatchPlacementEvent::BatchRegistered {
            seq: 0,
            instance_name: "batch placement".to_owned(),
            batch_id: 1,
            pipeline_uuid: EntityRef::new(pipeline_id, ()),
            port_uuid: None,
            origin: "test".to_owned(),
            tier: None,
        }),
    ];

    for event in events {
        let mut builder = SiriusModelBuilder::try_new(engine_id).expect("model builder is created");
        let error = builder
            .try_push(Event::new(Uuid::nil(), EPOCH, event))
            .expect_err("nil application FSM ids are rejected");
        assert!(matches!(error, AnalyzerError::Validation(_)));
    }
}

#[test]
fn data_flow_timeline_bins() {
    let mut fixture = fixture(true);
    let analyzer = analyzer(&mut fixture);

    let binned = analyzer
        .data_flow_timeline(request(fixture.query_id, &[]))
        .expect("data flow timeline");

    assert_eq!(binned.decl.entity_type_name, "batch_placement");
    assert_eq!(binned.decl.dimension_name, "Memory Tier");
    let keys: Vec<&str> = binned
        .decl
        .dimension_keys
        .iter()
        .map(|k| k.key.as_str())
        .collect();
    assert_eq!(keys, ["GPU", "HOST", "DISK"]);
    let measures: Vec<&str> = binned
        .decl
        .measures
        .iter()
        .map(|m| m.name.as_str())
        .collect();
    assert_eq!(measures, ["count", "bytes"]);

    // Only op1 has placements in the query; absent series mean all-zero.
    assert_eq!(binned.operators.len(), 1);
    assert!(!binned.operators.contains_key(&fixture.op2_id));
    let series = &binned.operators[&fixture.op1_id];

    // Hand-computed spans over 10 bins of 100 ns; the queued state is split
    // across GPU and HOST by the tier-change self-transition at t=300.
    let count = &series.values["count"];
    assert_eq!(
        count["batch_queued"]["GPU"],
        [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    );
    assert_eq!(
        count["batch_queued"]["HOST"],
        [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    );
    assert_eq!(
        count["batch_packaged"]["HOST"],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]
    );
    assert_eq!(
        count["batch_processing"]["GPU"],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    );
    // batch_registered is omitted; batch_consumed holds no tier residency.
    assert!(!count.contains_key("batch_registered"));
    assert!(!count.contains_key("batch_consumed"));
    // No task holds a reservation, so the synthetic series is absent.
    assert!(!count.contains_key("task_working_space"));

    let bytes = &series.values["bytes"];
    assert_eq!(
        bytes["batch_queued"]["GPU"],
        [0.0, 1000.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    );
    assert_eq!(
        bytes["batch_processing"]["GPU"],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 1000.0]
    );
}

#[test]
fn data_flow_timeline_task_working_space() {
    let mut fixture = fixture(true);
    add_working_space_task(&mut fixture);
    let analyzer = analyzer(&mut fixture);

    let binned = analyzer
        .data_flow_timeline(request(fixture.query_id, &[]))
        .expect("data flow timeline");

    // The task lives on op1 like batch A: still a single operator series.
    assert_eq!(binned.operators.len(), 1);
    let series = &binned.operators[&fixture.op1_id];

    // Only the reservation-holding spans contribute: preparing [400, 700)
    // + computing [700, 900).
    let count = &series.values["count"];
    assert_eq!(
        count["task_working_space"]["GPU"],
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    );
    // The reservation stays on GPU: no other tier appears in the series.
    assert_eq!(count["task_working_space"].len(), 1);

    let bytes = &series.values["bytes"];
    assert_eq!(
        bytes["task_working_space"]["GPU"],
        [
            0.0, 0.0, 0.0, 0.0, 2048.0, 2048.0, 2048.0, 2048.0, 2048.0, 0.0
        ]
    );

    // The batch lifecycle series coexist, unchanged from the batch-only run.
    assert_eq!(
        count["batch_queued"]["GPU"],
        [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    );
    assert_eq!(
        count["batch_processing"]["GPU"],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    );
    assert_eq!(
        bytes["batch_packaged"]["HOST"],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 1000.0, 1000.0, 0.0, 0.0]
    );
    assert!(!count.contains_key("batch_registered"));
}

#[test]
fn data_flow_timeline_unsupported_without_memory_tiers() {
    // A recording made with batch telemetry disabled has no memory_tier
    // resources: the feature is unsupported (HTTP 501) and the UI hides it.
    let mut fixture = fixture(false);
    fixture
        .events
        .retain(|e| !matches!(e.data, SiriusEvent::MemoryTier(_)));
    let analyzer = analyzer(&mut fixture);

    let error = analyzer
        .data_flow_timeline(request(fixture.query_id, &[]))
        .expect_err("data flow is unsupported without batch telemetry");
    assert!(matches!(error, AnalyzerError::Unsupported));
}

#[test]
fn data_flow_timeline_empty_query_is_supported() {
    // Tier resources present but no placements (e.g. `select 1;`): the view
    // is supported and empty, not an error.
    let mut fixture = fixture(false);
    let analyzer = analyzer(&mut fixture);

    let binned = analyzer
        .data_flow_timeline(request(fixture.query_id, &[]))
        .expect("empty data flow is a valid response");
    assert!(binned.operators.is_empty());
    assert!(!binned.decl.dimension_keys.is_empty());
}

#[test]
fn data_flow_timeline_measures_filter() {
    let mut fixture = fixture(true);
    let analyzer = analyzer(&mut fixture);

    // Only "bytes": the count measure is neither declared nor computed.
    let binned = analyzer
        .data_flow_timeline(request(fixture.query_id, &["bytes"]))
        .expect("data flow timeline");
    let measures: Vec<&str> = binned
        .decl
        .measures
        .iter()
        .map(|m| m.name.as_str())
        .collect();
    assert_eq!(measures, ["bytes"]);
    let series = &binned.operators[&fixture.op1_id];
    assert!(series.values.contains_key("bytes"));
    assert!(!series.values.contains_key("count"));

    // Unknown-only measures are an error.
    let error = analyzer
        .data_flow_timeline(request(fixture.query_id, &["bogus"]))
        .expect_err("unknown measures are rejected");
    assert!(matches!(error, AnalyzerError::InvalidArgument(_)));

    // A typo next to a valid measure is an error too, not silently ignored.
    let error = analyzer
        .data_flow_timeline(request(fixture.query_id, &["count", "bogus"]))
        .expect_err("unknown measures are rejected even alongside valid ones");
    assert!(matches!(error, AnalyzerError::InvalidArgument(_)));
}

/// A single-timeline request for one resource over the query window
/// [0, 1000) ns in 10 bins of 100 ns.
fn single_timeline_request(
    query_id: Uuid,
    resource_id: Uuid,
    entity_type_name: Option<&str>,
) -> SingleTimelineRequest<QueryFilter, OperatorFilter> {
    SingleTimelineRequest {
        entry: TimelineRequest::Resource(ResourceTimelineRequest {
            resource_id,
            long_entities_threshold_s: None,
            entity_filter: EntityFilter {
                entity_type_name: entity_type_name.map(|name| name.to_string()),
            },
            application: OperatorFilter {
                operator_ids: vec![],
            },
            config: TimelineConfig {
                num_bins: 10,
                start: 0.0,
                end: 1e-6,
            },
        }),
        app_params: QueryFilter { query_id },
    }
}

#[test]
fn unit_thread_timeline_counts_usage() {
    let mut fixture = fixture(false);
    let executor_id = add_working_space_task(&mut fixture);
    let analyzer = analyzer(&mut fixture);

    let response = analyzer
        .single_resource_timeline(single_timeline_request(fixture.query_id, executor_id, None))
        .expect("plain timeline over an executor thread resource");
    let UiResourceTimeline::Binned(binned) = response.data else {
        panic!("expected a plain binned response");
    };

    assert_eq!(
        binned.capacities_values["unit"],
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    );
}

#[test]
fn batch_keyed_timeline_over_memory_tier_resource() {
    let mut fixture = fixture(true);
    let analyzer = analyzer(&mut fixture);

    // Per-state timeline of the GPU tier resource sliced by the batch FSM.
    let response = analyzer
        .single_resource_timeline(single_timeline_request(
            fixture.query_id,
            fixture.gpu_id,
            Some("batch_placement"),
        ))
        .expect("batch keyed timeline over a memory_tier resource");
    let UiResourceTimeline::BinnedByState(by_state) = response.data else {
        panic!("expected a per-state binned response");
    };

    let states = &by_state.capacities_states_values["bytes"];
    assert_eq!(
        states["batch_queued"],
        [0.0, 1000.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    );
    // The registered entry state occupies GPU for bin 0 but is explicitly
    // omitted from aggregated lanes.
    assert!(!states.contains_key("batch_registered"));
    assert_eq!(
        states["batch_processing"],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 1000.0]
    );
    // batch_packaged held the batch on HOST, not GPU.
    assert!(!states.contains_key("batch_packaged"));

    // The task path over the same resource keeps working (tasks never use
    // memory tiers, so it is simply empty).
    let response = analyzer
        .single_resource_timeline(single_timeline_request(
            fixture.query_id,
            fixture.gpu_id,
            Some("task"),
        ))
        .expect("task keyed timeline over a memory_tier resource");
    let UiResourceTimeline::BinnedByState(by_state) = response.data else {
        panic!("expected a per-state binned response");
    };
    assert!(by_state.capacities_states_values.is_empty());

    // Unknown entity types are still rejected.
    let error = analyzer
        .single_resource_timeline(single_timeline_request(
            fixture.query_id,
            fixture.gpu_id,
            Some("widget"),
        ))
        .expect_err("unknown entity types are rejected");
    assert!(matches!(error, AnalyzerError::InvalidArgument(_)));
}

#[test]
fn plain_timeline_over_memory_tier_resource_includes_batches() {
    let mut fixture = fixture(true);
    let analyzer = analyzer(&mut fixture);

    let response = analyzer
        .single_resource_timeline(single_timeline_request(
            fixture.query_id,
            fixture.gpu_id,
            None,
        ))
        .expect("plain timeline over a memory_tier resource");
    let UiResourceTimeline::Binned(binned) = response.data else {
        panic!("expected a plain binned response");
    };

    // The GPU residency of batch A: queued [0, 300) + processing [800, 1000).
    assert_eq!(
        binned.capacities_values["bytes"],
        [
            1000.0, 1000.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 1000.0
        ]
    );
}

#[test]
fn list_entities_scoped_to_memory_tier_resource_is_empty() {
    let mut fixture = fixture(true);
    let analyzer = analyzer(&mut fixture);

    // Only tasks are listable v1; a memory_tier scope must not error, it just
    // matches no tasks.
    let response = analyzer
        .list_entities(EntityListRequest {
            entry: EntityListEntry {
                window: TimeWindow {
                    start: 0.0,
                    end: 1e-6,
                },
                filter: EntityListFilter {
                    scope: Some(EntityScope::Resource {
                        resource_id: fixture.gpu_id,
                    }),
                    entity_type_name: None,
                    min_usage_s: None,
                },
                sort: Sort {
                    key: EntitySortKey::UsageDuration,
                    dir: SortDir::Desc,
                },
                page: None,
                application: OperatorFilter {
                    operator_ids: vec![],
                },
            },
            app_params: QueryFilter {
                query_id: fixture.query_id,
            },
        })
        .expect("list entities scoped to a memory_tier resource");
    assert_eq!(response.total, 0);
    assert!(response.items.is_empty());
}
