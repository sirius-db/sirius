//! Generated-schema ingestion and UI response tests.

use quent_analyzer::{Entity, Model};
use quent_events::{EntityRef, Event};
use quent_query_engine_analyzer::ui::UiAnalyzer;
use quent_query_engine_ui::{OperatorFilter, QueryFilter};
use quent_ui::timeline::{categorical::CategoricalTimelineRequest, request::TimelineConfig};
use sirius_telemetry_store::{self as s, SiriusEvent as E};
use uuid::Uuid;

use crate::{DataBatchExt, SiriusUiAnalyzer};

const SECOND: u64 = 1_000_000_000;
const EPOCH: u64 = 1_000_000_000_000;
const GPU_SPACE_LABEL: &str = "memory_space(tier=GPU, device_id=0, limit=1000)";
const HOST_SPACE_LABEL: &str = "memory_space(tier=HOST, device_id=0, limit=1000)";

fn id(n: u128) -> Uuid {
    Uuid::from_u128(n)
}
fn ev(n: u128, t: u64, data: E) -> Event<E> {
    Event::new(id(n), EPOCH + t * SECOND, data)
}

fn fixture() -> Vec<Event<E>> {
    let mut events = vec![
        ev(
            1,
            0,
            E::Engine(s::EngineEvent::Init {
                label: Some("Sirius".into()),
            }),
        ),
        ev(
            2,
            0,
            E::Worker(s::WorkerEvent::Init {
                parent_engine_id: EntityRef::new(id(1), ()),
                process_id: "42".into(),
                tag: "worker".into(),
            }),
        ),
        ev(
            3,
            0,
            E::QueryGroup(s::QueryGroupEvent::Declaration {
                label: "group".into(),
                engine_id: EntityRef::new(id(1), ()),
            }),
        ),
        ev(
            4,
            0,
            E::Query(s::QueryEvent::Init {
                seq: 0,
                instance_name: "select".into(),
                query_group_id: EntityRef::new(id(3), ()),
            }),
        ),
        ev(4, 1, E::Query(s::QueryEvent::Planning { seq: 1 })),
        ev(4, 2, E::Query(s::QueryEvent::Executing { seq: 2 })),
        ev(4, 10, E::Query(s::QueryEvent::Finished { seq: 3 })),
        ev(
            5,
            0,
            E::Plan(s::PlanEvent::Declaration {
                query_id: EntityRef::new(id(4), ()),
                label: "plan".into(),
                edges: vec![],
                worker_id: Some(EntityRef::new(id(2), ())),
            }),
        ),
        ev(
            6,
            0,
            E::Operator(s::OperatorEvent::Declaration {
                plan_id: EntityRef::new(id(5), ()),
                label: "scan".into(),
                type_name: "scan".into(),
                custom_attributes: Default::default(),
            }),
        ),
        ev(
            7,
            0,
            E::GpuDevice(s::GpuDeviceEvent::Declaration {
                label: "gpu-0".into(),
                worker_id: EntityRef::new(id(2), ()),
                ordinal: 0,
            }),
        ),
        ev(
            8,
            0,
            E::MemorySpace(s::MemorySpaceEvent::Declaration {
                label: GPU_SPACE_LABEL.into(),
                bounds: s::MemorySpaceBounds { bytes: 1000 },
                worker_id: EntityRef::new(id(2), ()),
                gpu_id: Some(EntityRef::new(id(7), ())),
            }),
        ),
        ev(
            9,
            0,
            E::TaskQueue(s::TaskQueueEvent::Created {
                worker_id: EntityRef::new(id(2), ()),
                gpu_device_id: Some(EntityRef::new(id(7), ())),
                label: "queue".into(),
            }),
        ),
        ev(
            10,
            0,
            E::TaskManagerLoopThread(s::TaskManagerLoopThreadEvent::Spawned {
                label: "manager".into(),
                group_id: EntityRef::new(id(11), ()),
            }),
        ),
        ev(
            11,
            0,
            E::ThreadGroup(s::ThreadGroupEvent::Declaration {
                label: "threads".into(),
                worker_id: EntityRef::new(id(2), ()),
                gpu_device_id: Some(EntityRef::new(id(7), ())),
            }),
        ),
        ev(
            12,
            0,
            E::ExecutorThread(s::ExecutorThreadEvent::Spawned {
                label: "executor".into(),
                group_id: EntityRef::new(id(11), ()),
            }),
        ),
        ev(
            13,
            2,
            E::DataBatch(s::DataBatchEvent::Constructed {
                seq: 0,
                data_batch_id: 77,
                producer_pipeline_id: EntityRef::new(id(6), ()),
            }),
        ),
        ev(
            13,
            3,
            E::DataBatch(s::DataBatchEvent::Stationary {
                seq: 1,
                memory: EntityRef::new(id(8), s::MemorySpaceUsage { bytes: 400 }),
            }),
        ),
        ev(
            13,
            8,
            E::DataBatch(s::DataBatchEvent::Destructed { seq: 2 }),
        ),
        ev(
            14,
            2,
            E::Task(s::TaskEvent::Created {
                seq: 0,
                pipeline_uuid: EntityRef::new(id(6), ()),
            }),
        ),
        ev(
            14,
            3,
            E::Task(s::TaskEvent::Queued {
                seq: 1,
                queue: EntityRef::new(id(9), s::TaskQueueUsage { entries: 1 }),
            }),
        ),
        ev(
            14,
            4,
            E::Task(s::TaskEvent::Reserving {
                seq: 2,
                requested_bytes: 400,
                input_basis: 400,
                peak_estimate: 400,
                bytes_to_materialize: 0,
                manager_thread: EntityRef::new(id(10), s::TaskManagerLoopThreadUsage),
            }),
        ),
        ev(
            14,
            5,
            E::Task(s::TaskEvent::Preparing {
                seq: 3,
                origin_tier: "GPU".into(),
                target_tier: "GPU".into(),
                input_bytes: 400,
                executor_thread: EntityRef::new(id(12), s::ExecutorThreadUsage),
                reservation: EntityRef::new(id(8), s::MemorySpaceUsage { bytes: 100 }),
            }),
        ),
        ev(
            14,
            6,
            E::Task(s::TaskEvent::Computing {
                seq: 4,
                current_operator_id: 0,
                input_bytes: 400,
                input_batch_ids: vec![77],
                peak_allocated_bytes: 100,
                executor_thread: EntityRef::new(id(12), s::ExecutorThreadUsage),
                reservation: EntityRef::new(id(8), s::MemorySpaceUsage { bytes: 100 }),
            }),
        ),
        ev(
            14,
            8,
            E::Task(s::TaskEvent::Finalizing {
                seq: 5,
                success: true,
            }),
        ),
    ];
    events.sort_by_key(|e| e.timestamp);
    events
}

#[test]
fn generated_events_feed_query_bundle() {
    let analyzer = SiriusUiAnalyzer::try_new(id(1), fixture().into_iter()).unwrap();
    assert_eq!(
        analyzer
            .query_bundle(id(4))
            .unwrap()
            .entities
            .operators
            .len(),
        1
    );
    assert_eq!(analyzer.model.data_batches[&id(13)].numeric_id(), Some(77));
    assert_eq!(
        analyzer
            .model
            .data_batch_by_number(id(6), 77)
            .map(|b| b.id()),
        Some(id(13))
    );
    assert!(analyzer.model.try_entity_ref(id(13)).is_ok());
}

#[test]
fn data_flow_has_rate_residency_and_working_space() {
    let analyzer = SiriusUiAnalyzer::try_new(id(1), fixture().into_iter()).unwrap();
    let response = analyzer
        .data_flow_timeline(CategoricalTimelineRequest {
            measures: vec![],
            config: TimelineConfig {
                num_bins: 10,
                start: 0.0,
                end: 10.0,
            },
            app_params: QueryFilter { query_id: id(4) },
        })
        .unwrap();
    let series = &response.operators[&id(6)].values;
    assert_eq!(
        series["input_bytes_per_sec"]["computing"][GPU_SPACE_LABEL][6],
        200.0
    );
    assert_eq!(
        series["input_bytes_per_sec"]["computing"][GPU_SPACE_LABEL][7],
        200.0
    );
    assert_eq!(series["bytes"]["stationary"][GPU_SPACE_LABEL][4], 400.0);
    assert_eq!(
        series["bytes"]["task_working_space"][GPU_SPACE_LABEL][6],
        100.0
    );
    assert_eq!(series["count"]["precompute"][GPU_SPACE_LABEL][4], 1.0);
    assert_eq!(series["count"]["computing"][GPU_SPACE_LABEL][6], 1.0);
    assert_eq!(series["bytes"]["computing"][GPU_SPACE_LABEL][6], 400.0);
    assert_eq!(
        response.decl.default_measure.as_deref(),
        Some("input_bytes_per_sec")
    );
}

#[test]
fn rate_respects_partial_windows_and_measure_filter() {
    let analyzer = SiriusUiAnalyzer::try_new(id(1), fixture().into_iter()).unwrap();
    let response = analyzer
        .data_flow_timeline(CategoricalTimelineRequest {
            measures: vec!["input_bytes_per_sec".into()],
            config: TimelineConfig {
                num_bins: 1,
                start: 5.5,
                end: 6.5,
            },
            app_params: QueryFilter { query_id: id(4) },
        })
        .unwrap();
    let series = &response.operators[&id(6)].values;
    assert_eq!(
        series["input_bytes_per_sec"]["computing"][GPU_SPACE_LABEL],
        vec![100.0]
    );
    assert_eq!(series.len(), 1);
    assert_eq!(response.decl.measures.len(), 1);

    assert!(
        analyzer
            .data_flow_timeline(CategoricalTimelineRequest {
                measures: vec!["bogus".into()],
                config: TimelineConfig {
                    num_bins: 1,
                    start: 0.0,
                    end: 10.0
                },
                app_params: QueryFilter { query_id: id(4) },
            })
            .is_err()
    );
}

#[test]
fn duplicate_ids_in_one_compute_stage_count_once() {
    let mut events = fixture();
    for event in &mut events {
        if let E::Task(s::TaskEvent::Computing {
            input_batch_ids, ..
        }) = &mut event.data
        {
            input_batch_ids.push(77);
        }
    }
    let analyzer = SiriusUiAnalyzer::try_new(id(1), events.into_iter()).unwrap();
    let response = analyzer
        .data_flow_timeline(CategoricalTimelineRequest {
            measures: vec!["count".into()],
            config: TimelineConfig {
                num_bins: 10,
                start: 0.0,
                end: 10.0,
            },
            app_params: QueryFilter { query_id: id(4) },
        })
        .unwrap();
    assert_eq!(
        response.operators[&id(6)].values["count"]["computing"][GPU_SPACE_LABEL][6],
        1.0
    );
    assert_eq!(response.decl.default_measure.as_deref(), Some("count"));
}

#[test]
fn transit_does_not_double_count_batch_bytes() {
    let mut events = fixture();
    events.retain(|event| {
        event.id != id(13)
            || !matches!(
                event.data,
                E::DataBatch(s::DataBatchEvent::Destructed { .. })
            )
    });
    events.extend([
        ev(
            16,
            0,
            E::MemorySpace(s::MemorySpaceEvent::Declaration {
                label: HOST_SPACE_LABEL.into(),
                bounds: s::MemorySpaceBounds { bytes: 1000 },
                worker_id: EntityRef::new(id(2), ()),
                gpu_id: None,
            }),
        ),
        ev(
            17,
            0,
            E::Channel(s::ChannelEvent::Declaration {
                source_tier: EntityRef::new(id(8), ()),
                destination_tier: EntityRef::new(id(16), ()),
                worker_id: EntityRef::new(id(2), ()),
                gpu_id: Some(EntityRef::new(id(7), ())),
                label: "copy".into(),
            }),
        ),
        ev(
            13,
            5,
            E::DataBatch(s::DataBatchEvent::InTransit {
                seq: 2,
                source_memory: EntityRef::new(id(8), s::MemorySpaceUsage { bytes: 400 }),
                dest_memory: EntityRef::new(id(16), s::MemorySpaceUsage { bytes: 400 }),
                channel: EntityRef::new(id(17), s::ChannelUsage { bytes: 400 }),
            }),
        ),
        ev(
            13,
            7,
            E::DataBatch(s::DataBatchEvent::Stationary {
                seq: 3,
                memory: EntityRef::new(id(16), s::MemorySpaceUsage { bytes: 400 }),
            }),
        ),
        ev(
            13,
            8,
            E::DataBatch(s::DataBatchEvent::Destructed { seq: 4 }),
        ),
    ]);
    let analyzer = SiriusUiAnalyzer::try_new(id(1), events.into_iter()).unwrap();
    let response = analyzer
        .data_flow_timeline(CategoricalTimelineRequest {
            measures: vec![],
            config: TimelineConfig {
                num_bins: 10,
                start: 0.0,
                end: 10.0,
            },
            app_params: QueryFilter { query_id: id(4) },
        })
        .unwrap();
    let series = &response.operators[&id(6)].values;
    assert_eq!(series["count"]["computing"]["IN_TRANSIT"][6], 1.0);
    assert_eq!(series["count"]["computing"][HOST_SPACE_LABEL][7], 1.0);
    assert_eq!(series["bytes"]["stationary"][HOST_SPACE_LABEL][7], 400.0);
    assert!(!series["bytes"]["computing"].contains_key("IN_TRANSIT"));
}

#[test]
fn empty_query_has_supported_empty_data_flow() {
    let events = fixture()
        .into_iter()
        .filter(|event| event.id != id(13) && event.id != id(14));
    let analyzer = SiriusUiAnalyzer::try_new(id(1), events).unwrap();
    let response = analyzer
        .data_flow_timeline(CategoricalTimelineRequest {
            measures: vec![],
            config: TimelineConfig {
                num_bins: 10,
                start: 0.0,
                end: 10.0,
            },
            app_params: QueryFilter { query_id: id(4) },
        })
        .unwrap();
    assert!(response.operators.is_empty());
    assert!(
        response
            .decl
            .measures
            .iter()
            .any(|measure| measure.name == "input_bytes_per_sec")
    );
}

#[test]
fn resource_timelines_and_entity_listing_use_generated_fsms() {
    use quent_ui::entities::request::{
        EntityListEntry, EntityListFilter, EntityListRequest, EntitySortKey, Sort, SortDir,
        TimeWindow,
    };
    use quent_ui::timeline::{
        request::{EntityFilter, ResourceTimelineRequest, SingleTimelineRequest, TimelineRequest},
        response::ResourceTimeline,
    };

    let analyzer = SiriusUiAnalyzer::try_new(id(1), fixture().into_iter()).unwrap();
    let config = TimelineConfig {
        num_bins: 10,
        start: 0.0,
        end: 10.0,
    };
    for (entity_type, state) in [("task", "computing"), ("data_batch", "stationary")] {
        let response = analyzer
            .single_resource_timeline(SingleTimelineRequest {
                entry: TimelineRequest::Resource(ResourceTimelineRequest {
                    resource_id: id(8),
                    long_entities_threshold_s: None,
                    entity_filter: EntityFilter {
                        entity_type_name: Some(entity_type.into()),
                    },
                    application: OperatorFilter {
                        operator_ids: vec![id(6)],
                    },
                    config,
                }),
                app_params: QueryFilter { query_id: id(4) },
            })
            .unwrap();
        let ResourceTimeline::BinnedByState(data) = response.data else {
            panic!("state timeline expected")
        };
        assert!(data.capacities_states_values["bytes"].contains_key(state));
    }

    let listed = analyzer
        .list_entities(EntityListRequest {
            entry: EntityListEntry {
                window: TimeWindow {
                    start: 0.0,
                    end: 10.0,
                },
                filter: EntityListFilter {
                    entity_type_name: Some("data_batch".into()),
                    ..Default::default()
                },
                sort: Sort {
                    key: EntitySortKey::UsageDuration,
                    dir: SortDir::Desc,
                },
                page: None,
                application: OperatorFilter {
                    operator_ids: vec![id(6)],
                },
            },
            app_params: QueryFilter { query_id: id(4) },
        })
        .unwrap();
    assert_eq!(listed.total, 1);
    assert_eq!(listed.items[0].entity.fsm.id, id(13));
}

#[test]
fn duplicate_numeric_batch_id_on_one_worker_is_left_unresolved() {
    let mut events = fixture();
    events.extend([
        ev(
            15,
            2,
            E::DataBatch(s::DataBatchEvent::Constructed {
                seq: 0,
                data_batch_id: 77,
                producer_pipeline_id: EntityRef::new(id(6), ()),
            }),
        ),
        ev(
            15,
            3,
            E::DataBatch(s::DataBatchEvent::Stationary {
                seq: 1,
                memory: EntityRef::new(id(8), s::MemorySpaceUsage { bytes: 1 }),
            }),
        ),
        ev(
            15,
            8,
            E::DataBatch(s::DataBatchEvent::Destructed { seq: 2 }),
        ),
    ]);
    let analyzer = SiriusUiAnalyzer::try_new(id(1), events.into_iter()).unwrap();
    // Both batches survive; only the ambiguous number lookup is withheld.
    assert!(analyzer.model.data_batches.contains_key(&id(13)));
    assert!(analyzer.model.data_batches.contains_key(&id(15)));
    assert!(analyzer.model.data_batch_by_number(id(6), 77).is_none());
}

#[test]
fn same_numeric_batch_id_on_two_workers_resolves_per_worker() {
    let mut events = fixture();
    events.extend([
        ev(
            40,
            0,
            E::Worker(s::WorkerEvent::Init {
                parent_engine_id: EntityRef::new(id(1), ()),
                process_id: "43".into(),
                tag: "worker".into(),
            }),
        ),
        ev(
            41,
            0,
            E::Plan(s::PlanEvent::Declaration {
                query_id: EntityRef::new(id(4), ()),
                label: "remote-plan".into(),
                edges: vec![],
                worker_id: Some(EntityRef::new(id(40), ())),
            }),
        ),
        ev(
            42,
            0,
            E::Operator(s::OperatorEvent::Declaration {
                plan_id: EntityRef::new(id(41), ()),
                label: "remote-scan".into(),
                type_name: "scan".into(),
                custom_attributes: Default::default(),
            }),
        ),
        // Same engine-native number (77) as batch 13, but produced on worker 40.
        ev(
            43,
            2,
            E::DataBatch(s::DataBatchEvent::Constructed {
                seq: 0,
                data_batch_id: 77,
                producer_pipeline_id: EntityRef::new(id(42), ()),
            }),
        ),
        ev(
            43,
            3,
            E::DataBatch(s::DataBatchEvent::Stationary {
                seq: 1,
                memory: EntityRef::new(id(8), s::MemorySpaceUsage { bytes: 1 }),
            }),
        ),
        ev(
            43,
            8,
            E::DataBatch(s::DataBatchEvent::Destructed { seq: 2 }),
        ),
    ]);
    let model = SiriusUiAnalyzer::try_new(id(1), events.into_iter())
        .unwrap()
        .model;
    let resolve = |pipeline, number| {
        model
            .data_batch_by_number(pipeline, number)
            .map(|batch| batch.id())
    };
    assert_eq!(resolve(id(6), 77), Some(id(13)));
    assert_eq!(resolve(id(42), 77), Some(id(43)));
}

#[test]
fn invalid_query_is_skipped_with_its_plan_subtree() {
    let mut events = fixture();
    events.extend([
        // Query 30 never reaches a final state, so its FSM is incomplete.
        ev(
            30,
            0,
            E::Query(s::QueryEvent::Init {
                seq: 0,
                instance_name: "aborted".into(),
                query_group_id: EntityRef::new(id(3), ()),
            }),
        ),
        ev(
            31,
            0,
            E::Plan(s::PlanEvent::Declaration {
                query_id: EntityRef::new(id(30), ()),
                label: "aborted-plan".into(),
                edges: vec![],
                worker_id: Some(EntityRef::new(id(2), ())),
            }),
        ),
        ev(
            32,
            0,
            E::Operator(s::OperatorEvent::Declaration {
                plan_id: EntityRef::new(id(31), ()),
                label: "aborted-scan".into(),
                type_name: "scan".into(),
                custom_attributes: Default::default(),
            }),
        ),
    ]);
    let model = SiriusUiAnalyzer::try_new(id(1), events.into_iter())
        .unwrap()
        .model;
    assert!(model.queries.contains_key(&id(4)));
    assert!(!model.queries.contains_key(&id(30)));
    assert!(!model.plans.contains_key(&id(31)));
    assert!(!model.operators.contains_key(&id(32)));
}

#[test]
fn invalid_data_batch_is_skipped() {
    let mut events = fixture();
    // Batch 16 is constructed but never reaches a final state.
    events.push(ev(
        16,
        2,
        E::DataBatch(s::DataBatchEvent::Constructed {
            seq: 0,
            data_batch_id: 99,
            producer_pipeline_id: EntityRef::new(id(6), ()),
        }),
    ));
    let model = SiriusUiAnalyzer::try_new(id(1), events.into_iter())
        .unwrap()
        .model;
    assert!(model.data_batches.contains_key(&id(13)));
    assert!(!model.data_batches.contains_key(&id(16)));
}

#[test]
fn duplicate_resource_group_declaration_is_rejected() {
    let mut events = fixture();
    events.push(ev(
        7,
        1,
        E::GpuDevice(s::GpuDeviceEvent::Declaration {
            label: "gpu-0-again".into(),
            worker_id: EntityRef::new(id(2), ()),
            ordinal: 0,
        }),
    ));
    assert!(SiriusUiAnalyzer::try_new(id(1), events.into_iter()).is_err());
}

#[test]
fn memory_space_dimensions_order_by_tier_then_device() {
    let mut labels = vec![
        crate::UNKNOWN_DIMENSION,
        "memory_space(tier=DISK, device_id=0, limit=1)",
        crate::IN_TRANSIT_DIMENSION,
        "memory_space(tier=HOST, device_id=0, limit=1)",
        "memory_space(tier=GPU, device_id=10, limit=1)",
        "memory_space(tier=GPU, device_id=2, limit=1)",
    ];
    labels.sort_unstable_by_key(|&label| crate::memory_space_rank(label));
    assert_eq!(
        labels,
        [
            "memory_space(tier=GPU, device_id=2, limit=1)",
            "memory_space(tier=GPU, device_id=10, limit=1)",
            "memory_space(tier=HOST, device_id=0, limit=1)",
            "memory_space(tier=DISK, device_id=0, limit=1)",
            crate::IN_TRANSIT_DIMENSION,
            crate::UNKNOWN_DIMENSION,
        ]
    );
}

#[test]
fn consumer_view_resolves_foreign_producer_batch() {
    let mut events = fixture();
    for event in &mut events {
        if let E::Task(s::TaskEvent::Computing {
            input_batch_ids, ..
        }) = &mut event.data
        {
            input_batch_ids.push(78);
        }
    }
    events.extend([
        ev(
            20,
            0,
            E::Query(s::QueryEvent::Init {
                seq: 0,
                instance_name: "other".into(),
                query_group_id: EntityRef::new(id(3), ()),
            }),
        ),
        ev(20, 1, E::Query(s::QueryEvent::Planning { seq: 1 })),
        ev(20, 2, E::Query(s::QueryEvent::Executing { seq: 2 })),
        ev(20, 10, E::Query(s::QueryEvent::Finished { seq: 3 })),
        ev(
            21,
            0,
            E::Plan(s::PlanEvent::Declaration {
                query_id: EntityRef::new(id(20), ()),
                label: "other plan".into(),
                edges: vec![],
                worker_id: Some(EntityRef::new(id(2), ())),
            }),
        ),
        ev(
            22,
            0,
            E::Operator(s::OperatorEvent::Declaration {
                plan_id: EntityRef::new(id(21), ()),
                label: "producer".into(),
                type_name: "scan".into(),
                custom_attributes: Default::default(),
            }),
        ),
        ev(
            23,
            2,
            E::DataBatch(s::DataBatchEvent::Constructed {
                seq: 0,
                data_batch_id: 78,
                producer_pipeline_id: EntityRef::new(id(22), ()),
            }),
        ),
        ev(
            23,
            3,
            E::DataBatch(s::DataBatchEvent::Stationary {
                seq: 1,
                memory: EntityRef::new(id(8), s::MemorySpaceUsage { bytes: 50 }),
            }),
        ),
        ev(
            23,
            8,
            E::DataBatch(s::DataBatchEvent::Destructed { seq: 2 }),
        ),
    ]);
    let analyzer = SiriusUiAnalyzer::try_new(id(1), events.into_iter()).unwrap();
    let view = analyzer.model.query_view(id(4)).unwrap();
    assert!(view.data_batches().any(|batch| batch.id() == id(23)));
    assert_eq!(
        analyzer
            .query_bundle(id(4))
            .unwrap()
            .entities
            .operators
            .len(),
        1
    );

    let response = analyzer
        .data_flow_timeline(CategoricalTimelineRequest {
            measures: vec!["count".into()],
            config: TimelineConfig {
                num_bins: 10,
                start: 0.0,
                end: 10.0,
            },
            app_params: QueryFilter { query_id: id(4) },
        })
        .unwrap();
    assert!(!response.operators.contains_key(&id(22)));
    assert_eq!(
        response.operators[&id(6)].values["count"]["computing"][GPU_SPACE_LABEL][6],
        2.0
    );
}

#[test]
fn generated_ndjson_imports_into_analyzer() {
    use quent_io::{
        ExporterOptions,
        filesystem::{self, Format},
    };
    use quent_query_engine_analyzer::ui::{QuentViewer, ViewerEventStream};
    use quent_store::{
        context::ContextSet,
        event::{CombinedEventLoader, filesystem::Loader},
    };
    use sirius_telemetry_instrumentation as instrumentation;

    let output = tempfile::tempdir().unwrap();
    let (context_id, engine_id, query_id) = {
        let context = instrumentation::Context::<instrumentation::Sirius>::try_new(
            ExporterOptions::FileSystem(filesystem::exporter::Options::new(
                Format::Ndjson,
                output.path().to_path_buf(),
            )),
        )
        .unwrap();
        let context_id = context.id();
        let mut engine = context.observer::<instrumentation::Engine>().handle();
        engine.init(Some("Sirius".into())).unwrap();
        let engine_id = engine.id();
        let mut group = context.observer::<instrumentation::QueryGroup>().handle();
        group
            .declaration("group".into(), engine.as_entity_ref())
            .unwrap();
        let query = context
            .observer::<instrumentation::Query>()
            .handle()
            .init("select".into(), group.as_entity_ref())
            .planning()
            .executing()
            .finished();
        let query_id = query.id();
        let mut worker = context.observer::<instrumentation::Worker>().handle();
        worker
            .init(engine.as_entity_ref(), "42".into(), "worker".into())
            .unwrap();
        let mut plan = context.observer::<instrumentation::Plan>().handle();
        plan.declaration(
            query.as_entity_ref(),
            "plan".into(),
            vec![],
            Some(worker.as_entity_ref()),
        )
        .unwrap();

        (context_id, engine_id, query_id)
    };

    let stored = Loader::<s::Sirius>::new(output.path(), ContextSet::one(context_id))
        .combined_events()
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(stored.len(), 8);

    let context_dir = output.path().join(context_id.to_string());
    let inventory = crate::Viewer::context_inventory(&context_dir).unwrap();
    assert!(inventory.analysis_target_ids.contains(&engine_id));
    let imported: ViewerEventStream<SiriusUiAnalyzer> =
        crate::Viewer::import_events(&context_dir).unwrap();
    let analyzer = SiriusUiAnalyzer::try_new(engine_id, imported).unwrap();
    assert_eq!(analyzer.query_bundle(query_id).unwrap().query_id, query_id);

    let worker_context_id = {
        let context = instrumentation::Context::<instrumentation::Sirius>::try_new(
            ExporterOptions::FileSystem(filesystem::exporter::Options::new(
                Format::Ndjson,
                output.path().to_path_buf(),
            )),
        )
        .unwrap();
        let mut worker = context.observer::<instrumentation::Worker>().handle();
        worker
            .init(
                instrumentation::EntityRef::new(engine_id, ()),
                "43".into(),
                "remote".into(),
            )
            .unwrap();
        context.id()
    };
    let worker_dir = output.path().join(worker_context_id.to_string());
    let inventory = crate::Viewer::context_inventory(&worker_dir).unwrap();
    assert!(inventory.analysis_target_ids.contains(&engine_id));
    // Keep imports within the selected context when the root contains others.
    assert_eq!(
        crate::Viewer::import_events(&worker_dir).unwrap().count(),
        1
    );
    assert_eq!(
        crate::Viewer::import_events(&context_dir).unwrap().count(),
        8
    );

    let engine_events = crate::Viewer::import_events(&context_dir).unwrap();
    let worker_events = crate::Viewer::import_events(&worker_dir).unwrap();
    let combined =
        SiriusUiAnalyzer::try_new(engine_id, engine_events.chain(worker_events)).unwrap();
    assert_eq!(combined.model.workers.len(), 2);
}

#[test]
fn fsm_diagrams_use_final_states() {
    use quent_ui::fsm::{FsmTransitionDecl, FsmTypeDeclaration};

    use crate::boilerplate::{DataBatch, Task};

    // UI diagrams must end at the same states as recorded events.
    for (declaration, terminal) in [
        (Task::fsm_type_declaration(), "finalizing"),
        (DataBatch::fsm_type_declaration(), "destructed"),
    ] {
        assert!(declaration.states.iter().all(|state| state.name != "exit"));
        assert!(declaration.transitions.iter().any(
            |transition| matches!(transition, FsmTransitionDecl::Exit(state) if state == terminal)
        ));
        assert!(declaration.transitions.iter().all(
            |transition| !matches!(transition, FsmTransitionDecl::Transition(source, _) if source == terminal)
        ));
    }
}

#[test]
fn query_groups_use_ui_attributes() {
    use quent_query_engine_analyzer::QueryEntity;

    let analyzer = SiriusUiAnalyzer::try_new(id(1), fixture().into_iter()).unwrap();
    let query = analyzer.model.queries[&id(4)].to_ui().unwrap();
    assert_eq!(query.custom_attributes.len(), 1);
    let attribute = &query.custom_attributes[0];
    assert_eq!(attribute.key, crate::boilerplate::QUERY_GROUP_ATTRIBUTE);
    assert!(matches!(
        &attribute.value,
        Some(quent_dynamic_attributes::DynamicValue::String(label)) if label == "group"
    ));
    assert_eq!(
        analyzer.query_initial_group_by_attribute().as_deref(),
        Some(crate::boilerplate::QUERY_GROUP_ATTRIBUTE)
    );
}

const COPY_CHANNEL: u128 = 17;
const COPY_HANDLE: u128 = 20;

fn clone_start(handle: u128, time: u64, clone_id: u64, bytes: u64) -> Event<E> {
    ev(
        handle,
        time,
        E::BatchClone(s::BatchCloneEvent::Started {
            clone_id,
            source_tier: EntityRef::new(id(8), ()),
            destination_tier: EntityRef::new(id(16), ()),
            source_data_batch: EntityRef::new(id(13), ()),
            destination_data_batch_id: 88,
            bytes,
        }),
    )
}

fn clone_fixture() -> Vec<Event<E>> {
    let mut events = fixture();
    events.extend([
        ev(
            16,
            0,
            E::MemorySpace(s::MemorySpaceEvent::Declaration {
                label: HOST_SPACE_LABEL.into(),
                bounds: s::MemorySpaceBounds { bytes: 1000 },
                worker_id: EntityRef::new(id(2), ()),
                gpu_id: None,
            }),
        ),
        ev(
            COPY_CHANNEL,
            0,
            E::Channel(s::ChannelEvent::Declaration {
                source_tier: EntityRef::new(id(8), ()),
                destination_tier: EntityRef::new(id(16), ()),
                worker_id: EntityRef::new(id(2), ()),
                gpu_id: Some(EntityRef::new(id(7), ())),
                label: "copy-only".into(),
            }),
        ),
        // Completion order differs from start order, including across handles.
        ev(
            COPY_HANDLE,
            6,
            E::BatchClone(s::BatchCloneEvent::Completed {
                clone_id: 2,
                success: true,
            }),
        ),
        ev(
            21,
            8,
            E::BatchClone(s::BatchCloneEvent::Completed {
                clone_id: 1,
                success: true,
            }),
        ),
        ev(
            COPY_HANDLE,
            7,
            E::BatchClone(s::BatchCloneEvent::Completed {
                clone_id: 1,
                success: true,
            }),
        ),
        clone_start(COPY_HANDLE, 4, 1, 600),
        clone_start(COPY_HANDLE, 5, 2, 400),
        clone_start(21, 6, 1, 200),
    ]);
    events
}

fn clone_timeline(
    analyzer: &SiriusUiAnalyzer,
    query: u128,
    operators: Vec<Uuid>,
    entity_type: Option<&str>,
) -> quent_ui::timeline::response::ResourceTimeline {
    use quent_ui::timeline::request::{
        EntityFilter, ResourceTimelineRequest, SingleTimelineRequest, TimelineRequest,
    };

    analyzer
        .single_resource_timeline(SingleTimelineRequest {
            entry: TimelineRequest::Resource(ResourceTimelineRequest {
                resource_id: id(COPY_CHANNEL),
                long_entities_threshold_s: None,
                entity_filter: EntityFilter {
                    entity_type_name: entity_type.map(str::to_owned),
                },
                application: OperatorFilter {
                    operator_ids: operators,
                },
                config: TimelineConfig {
                    num_bins: 10,
                    start: 0.0,
                    end: 10.0,
                },
            }),
            app_params: QueryFilter {
                query_id: id(query),
            },
        })
        .unwrap()
        .data
}

fn clone_list(
    analyzer: &SiriusUiAnalyzer,
    query: u128,
    operators: Vec<Uuid>,
    entity_type: &str,
) -> quent_query_engine_ui::EntityListResponse {
    clone_list_in_window(
        analyzer,
        query,
        operators,
        entity_type,
        quent_ui::entities::request::TimeWindow {
            start: 0.0,
            end: 10.0,
        },
    )
}

fn clone_list_in_window(
    analyzer: &SiriusUiAnalyzer,
    query: u128,
    operators: Vec<Uuid>,
    entity_type: &str,
    window: quent_ui::entities::request::TimeWindow,
) -> quent_query_engine_ui::EntityListResponse {
    use quent_ui::entities::request::{
        EntityListEntry, EntityListFilter, EntityListRequest, EntitySortKey, Sort, SortDir,
    };

    analyzer
        .list_entities(EntityListRequest {
            entry: EntityListEntry {
                window,
                filter: EntityListFilter {
                    entity_type_name: Some(entity_type.into()),
                    ..Default::default()
                },
                sort: Sort {
                    key: EntitySortKey::UsageDuration,
                    dir: SortDir::Desc,
                },
                page: None,
                application: OperatorFilter {
                    operator_ids: operators,
                },
            },
            app_params: QueryFilter {
                query_id: id(query),
            },
        })
        .unwrap()
}

#[test]
fn clone_pairs_ignore_order() {
    use std::collections::HashSet;

    let analyzer = SiriusUiAnalyzer::try_new(id(1), clone_fixture().into_iter()).unwrap();
    let reversed = SiriusUiAnalyzer::try_new(id(1), clone_fixture().into_iter().rev()).unwrap();
    assert_eq!(analyzer.model.batch_clones.len(), 3);
    assert_eq!(
        analyzer.model.batch_clones.keys().collect::<HashSet<_>>(),
        reversed.model.batch_clones.keys().collect::<HashSet<_>>(),
    );
    let mut intervals = analyzer
        .model
        .batch_clones
        .values()
        .map(|copy| {
            assert_eq!(copy.source_batch_id(), Some(id(13)));
            assert_eq!(copy.destination_batch_id(), Some(88));
            assert_eq!(copy.channel_id(), Some(id(COPY_CHANNEL)));
            assert_eq!(copy.success(), Some(true));
            assert!(analyzer.model.try_entity_ref(copy.id()).is_ok());
            (
                copy.clone_id(),
                copy.started_at(),
                copy.completed_at(),
                copy.bytes(),
            )
        })
        .collect::<Vec<_>>();
    intervals.sort();
    assert_eq!(
        intervals,
        vec![
            (
                1,
                Some(EPOCH + 4 * SECOND),
                Some(EPOCH + 7 * SECOND),
                Some(600)
            ),
            (
                1,
                Some(EPOCH + 6 * SECOND),
                Some(EPOCH + 8 * SECOND),
                Some(200)
            ),
            (
                2,
                Some(EPOCH + 5 * SECOND),
                Some(EPOCH + 6 * SECOND),
                Some(400)
            ),
        ]
    );
}

#[test]
fn clone_only_channels_show_rates() {
    use quent_ui::timeline::response::ResourceTimeline;

    let analyzer = SiriusUiAnalyzer::try_new(id(1), clone_fixture().into_iter()).unwrap();
    let bundle = analyzer.query_bundle(id(4)).unwrap();
    assert!(bundle.entities.resources.contains_key(&id(COPY_CHANNEL)));
    assert!(bundle.entities.fsm_types.contains_key("batch_clone"));
    let ResourceTimeline::Binned(plain) = clone_timeline(&analyzer, 4, vec![id(6)], None) else {
        panic!("plain timeline expected")
    };
    assert_eq!(
        plain.capacities_values["bytes"],
        vec![0.0, 0.0, 0.0, 0.0, 200.0, 600.0, 300.0, 100.0, 0.0, 0.0,]
    );
    let ResourceTimeline::BinnedByState(keyed) =
        clone_timeline(&analyzer, 4, vec![id(6)], Some("batch_clone"))
    else {
        panic!("clone state timeline expected")
    };
    assert_eq!(
        keyed.capacities_states_values["bytes"]["started"],
        plain.capacities_values["bytes"]
    );
    assert_eq!(
        clone_list(&analyzer, 4, vec![id(6)], "batch_clone").total,
        3
    );
    assert_eq!(clone_list(&analyzer, 4, vec![id(6)], "data_batch").total, 1);

    // A logical copy leaves the source batch's residency and byte count intact.
    let flow = analyzer
        .data_flow_timeline(CategoricalTimelineRequest {
            measures: vec!["bytes".into()],
            config: TimelineConfig {
                num_bins: 10,
                start: 0.0,
                end: 10.0,
            },
            app_params: QueryFilter { query_id: id(4) },
        })
        .unwrap();
    assert_eq!(
        flow.operators[&id(6)].values["bytes"]["stationary"][GPU_SPACE_LABEL][5],
        400.0
    );
}

#[test]
fn failed_clones_have_no_rate() {
    use quent_ui::timeline::response::ResourceTimeline;

    let mut events = clone_fixture();
    events.extend([
        clone_start(COPY_HANDLE, 4, 3, 900),
        ev(
            COPY_HANDLE,
            7,
            E::BatchClone(s::BatchCloneEvent::Completed {
                clone_id: 3,
                success: false,
            }),
        ),
        clone_start(COPY_HANDLE, 4, 4, 900),
        ev(
            COPY_HANDLE,
            6,
            E::BatchClone(s::BatchCloneEvent::Completed {
                clone_id: 5,
                success: true,
            }),
        ),
        clone_start(COPY_HANDLE, 5, 6, 900),
        ev(
            COPY_HANDLE,
            5,
            E::BatchClone(s::BatchCloneEvent::Completed {
                clone_id: 6,
                success: true,
            }),
        ),
    ]);
    let analyzer = SiriusUiAnalyzer::try_new(id(1), events.into_iter()).unwrap();
    assert_eq!(analyzer.model.batch_clones.len(), 7);
    let ResourceTimeline::Binned(timeline) = clone_timeline(&analyzer, 4, vec![id(6)], None) else {
        panic!("plain timeline expected")
    };
    assert_eq!(
        timeline.capacities_values["bytes"],
        vec![0.0, 0.0, 0.0, 0.0, 200.0, 600.0, 300.0, 100.0, 0.0, 0.0,]
    );
    let listed = clone_list(&analyzer, 4, vec![id(6)], "batch_clone");
    // Completion-only records have no source operator to attach to this query.
    assert_eq!(listed.total, 6);
}

#[test]
fn clone_filters_isolate_queries() {
    use quent_ui::timeline::response::ResourceTimeline;

    let mut events = clone_fixture();
    events.extend([
        ev(
            30,
            0,
            E::Query(s::QueryEvent::Init {
                seq: 0,
                instance_name: "other".into(),
                query_group_id: EntityRef::new(id(3), ()),
            }),
        ),
        ev(30, 10, E::Query(s::QueryEvent::Finished { seq: 1 })),
        ev(
            31,
            0,
            E::Plan(s::PlanEvent::Declaration {
                query_id: EntityRef::new(id(30), ()),
                label: "other".into(),
                edges: vec![],
                worker_id: Some(EntityRef::new(id(2), ())),
            }),
        ),
        ev(
            32,
            0,
            E::Operator(s::OperatorEvent::Declaration {
                plan_id: EntityRef::new(id(31), ()),
                label: "other-scan".into(),
                type_name: "scan".into(),
                custom_attributes: Default::default(),
            }),
        ),
    ]);
    let analyzer = SiriusUiAnalyzer::try_new(id(1), events.into_iter()).unwrap();
    assert_eq!(clone_list(&analyzer, 30, vec![], "batch_clone").total, 0);
    assert_eq!(
        clone_list(&analyzer, 4, vec![id(32)], "batch_clone").total,
        0
    );
    for (query, operators) in [(30, vec![]), (4, vec![id(32)])] {
        let ResourceTimeline::Binned(timeline) = clone_timeline(&analyzer, query, operators, None)
        else {
            panic!("plain timeline expected")
        };
        assert!(
            timeline
                .capacities_values
                .values()
                .flatten()
                .all(|value| *value == 0.0)
        );
    }
}

#[test]
fn same_space_clone_has_no_channel() {
    use quent_analyzer::resource::{Usage, Using};

    let mut events = clone_fixture();
    events.retain(|event| !matches!(event.data, E::BatchClone(_)));
    let mut started = clone_start(COPY_HANDLE, 4, 1, 600);
    let E::BatchClone(s::BatchCloneEvent::Started {
        destination_tier, ..
    }) = &mut started.data
    else {
        unreachable!()
    };
    *destination_tier = EntityRef::new(id(8), ());
    events.extend([
        started,
        ev(
            COPY_HANDLE,
            7,
            E::BatchClone(s::BatchCloneEvent::Completed {
                clone_id: 1,
                success: true,
            }),
        ),
    ]);
    let analyzer = SiriusUiAnalyzer::try_new(id(1), events.into_iter()).unwrap();
    let copy = analyzer.model.batch_clones.values().next().unwrap();
    assert_eq!(copy.channel_id(), None);
    assert!(
        copy.usages()
            .all(|usage| usage.resource_id() != id(COPY_CHANNEL))
    );
    assert_eq!(
        clone_list(&analyzer, 4, vec![id(6)], "batch_clone").total,
        1
    );
}

#[test]
fn clone_destinations_are_local() {
    let mut events = clone_fixture();
    events.extend([
        ev(
            40,
            0,
            E::Worker(s::WorkerEvent::Init {
                parent_engine_id: EntityRef::new(id(1), ()),
                process_id: "43".into(),
                tag: "remote".into(),
            }),
        ),
        ev(
            41,
            0,
            E::Plan(s::PlanEvent::Declaration {
                query_id: EntityRef::new(id(4), ()),
                label: "remote".into(),
                edges: vec![],
                worker_id: Some(EntityRef::new(id(40), ())),
            }),
        ),
        ev(
            42,
            0,
            E::Operator(s::OperatorEvent::Declaration {
                plan_id: EntityRef::new(id(41), ()),
                label: "remote".into(),
                type_name: "scan".into(),
                custom_attributes: Default::default(),
            }),
        ),
    ]);
    // Destination numbers repeat across workers, just like source batch numbers.
    for (batch, operator) in [(50, 6), (51, 42)] {
        events.extend([
            ev(
                batch,
                7,
                E::DataBatch(s::DataBatchEvent::Constructed {
                    seq: 0,
                    data_batch_id: 88,
                    producer_pipeline_id: EntityRef::new(id(operator), ()),
                }),
            ),
            ev(
                batch,
                7,
                E::DataBatch(s::DataBatchEvent::Stationary {
                    seq: 1,
                    memory: EntityRef::new(id(16), s::MemorySpaceUsage { bytes: 200 }),
                }),
            ),
            ev(
                batch,
                9,
                E::DataBatch(s::DataBatchEvent::Destructed { seq: 2 }),
            ),
        ]);
    }
    let analyzer = SiriusUiAnalyzer::try_new(id(1), events.into_iter()).unwrap();
    for copy in analyzer.model.batch_clones.values() {
        assert_eq!(copy.destination_batch_uuid(), Some(id(50)));
    }
}

#[test]
fn clone_bulk_keeps_details() {
    use quent_ui::timeline::{
        request::{
            BulkChunkedTimelineRequest, BulkTimelineRequest, EntityFilter,
            ResourceGroupTimelineRequest, SingleTimelineRequest, TimelineRequest,
        },
        response::{BulkTimelinesResponseEntry, ResourceTimeline},
    };

    let analyzer = SiriusUiAnalyzer::try_new(id(1), clone_fixture().into_iter()).unwrap();
    let config = TimelineConfig {
        num_bins: 10,
        start: 0.0,
        end: 10.0,
    };
    let entry = TimelineRequest::ResourceGroup(ResourceGroupTimelineRequest {
        resource_group_id: id(7),
        resource_type_name: "channel".into(),
        long_entities_threshold_s: Some(0.5),
        entity_filter: EntityFilter {
            entity_type_name: Some("batch_clone".into()),
        },
        app_params: OperatorFilter {
            operator_ids: vec![id(6)],
        },
        config,
    });
    let single = analyzer
        .single_resource_timeline(SingleTimelineRequest {
            entry: entry.clone(),
            app_params: QueryFilter { query_id: id(4) },
        })
        .unwrap();
    let bulk = analyzer
        .bulk_resource_timeline(BulkTimelineRequest {
            entries: [("copies".into(), entry.clone())].into(),
            app_params: QueryFilter { query_id: id(4) },
        })
        .unwrap();
    let chunked = analyzer
        .bulk_chunked_resource_timeline(BulkChunkedTimelineRequest {
            entries: [("copies".into(), entry)].into(),
            configs: vec![config],
            app_params: QueryFilter { query_id: id(4) },
        })
        .unwrap();
    let BulkTimelinesResponseEntry::Ok {
        data: bulk_data, ..
    } = &bulk.entries["copies"]
    else {
        panic!("bulk timeline expected")
    };
    let BulkTimelinesResponseEntry::Ok {
        data: chunk_data, ..
    } = &chunked.entries["copies"][0]
    else {
        panic!("chunked timeline expected")
    };
    let mut expected = None;
    for timeline in [&single.data, bulk_data, chunk_data] {
        let ResourceTimeline::BinnedByState(data) = timeline else {
            panic!("clone state timeline expected")
        };
        assert_eq!(data.long_fsms.len(), 3);
        assert!(
            data.long_fsms
                .iter()
                .all(|fsm| analyzer.model.batch_clones.contains_key(&fsm.id))
        );
        if let Some(values) = expected {
            assert_eq!(values, &data.capacities_states_values);
        }
        expected = Some(&data.capacities_states_values);
    }
}

#[test]
fn clone_windows_are_half_open() {
    use quent_ui::entities::request::TimeWindow;
    use std::collections::HashSet;

    let mut events = clone_fixture();
    events.retain(|event| !matches!(event.data, E::BatchClone(_)));
    // Complete spans touch either edge; only clone 2 overlaps [5, 6).
    for (clone_id, start, end) in [(1, 3, 5), (2, 4, 7), (3, 6, 8), (6, 5, 5), (7, 6, 6)] {
        events.extend([
            clone_start(COPY_HANDLE, start, clone_id, 200),
            ev(
                COPY_HANDLE,
                end,
                E::BatchClone(s::BatchCloneEvent::Completed {
                    clone_id,
                    success: true,
                }),
            ),
        ]);
    }
    events.extend([
        clone_start(COPY_HANDLE, 5, 4, 200),
        clone_start(COPY_HANDLE, 6, 5, 200),
    ]);
    let analyzer = SiriusUiAnalyzer::try_new(id(1), events.into_iter()).unwrap();
    let listed = clone_list_in_window(
        &analyzer,
        4,
        vec![id(6)],
        "batch_clone",
        TimeWindow {
            start: 5.0,
            end: 6.0,
        },
    );
    let expected = analyzer
        .model
        .batch_clones
        .values()
        .filter(|copy| [2, 4, 6].contains(&copy.clone_id()))
        .map(Entity::id)
        .collect::<HashSet<_>>();
    let actual = listed
        .items
        .iter()
        .map(|item| item.entity.fsm.id)
        .collect::<HashSet<_>>();
    assert_eq!(actual, expected);
}

#[test]
fn untyped_list_merges_entities() {
    use quent_ui::{
        entities::request::{
            EntityListEntry, EntityListFilter, EntityListRequest, EntityScope, EntitySortKey, Sort,
            SortDir, TimeWindow,
        },
        paginate::PageParams,
    };

    let mut events = clone_fixture();
    events.extend([
        ev(
            50,
            7,
            E::DataBatch(s::DataBatchEvent::Constructed {
                seq: 0,
                data_batch_id: 88,
                producer_pipeline_id: EntityRef::new(id(6), ()),
            }),
        ),
        ev(
            50,
            7,
            E::DataBatch(s::DataBatchEvent::Stationary {
                seq: 1,
                memory: EntityRef::new(id(16), s::MemorySpaceUsage { bytes: 200 }),
            }),
        ),
        ev(
            50,
            9,
            E::DataBatch(s::DataBatchEvent::Destructed { seq: 2 }),
        ),
    ]);
    let analyzer = SiriusUiAnalyzer::try_new(id(1), events.into_iter()).unwrap();
    let list = |filter, page| {
        analyzer
            .list_entities(EntityListRequest {
                entry: EntityListEntry {
                    window: TimeWindow {
                        start: 0.0,
                        end: 10.0,
                    },
                    filter,
                    sort: Sort {
                        key: EntitySortKey::UsageDuration,
                        dir: SortDir::Desc,
                    },
                    page,
                    application: OperatorFilter {
                        operator_ids: vec![id(6)],
                    },
                },
                app_params: QueryFilter { query_id: id(4) },
            })
            .unwrap()
    };
    let mut expected = ["task", "data_batch", "batch_clone"]
        .into_iter()
        .flat_map(|entity_type| {
            list(
                EntityListFilter {
                    entity_type_name: Some(entity_type.into()),
                    ..Default::default()
                },
                None,
            )
            .items
        })
        .collect::<Vec<_>>();
    expected.sort_by(|left, right| {
        right
            .usage_duration_s
            .total_cmp(&left.usage_duration_s)
            .then_with(|| left.entity.fsm.id.cmp(&right.entity.fsm.id))
    });
    let expected_ids = expected
        .iter()
        .map(|item| item.entity.fsm.id)
        .collect::<Vec<_>>();
    assert_eq!(expected_ids.len(), 6);
    let all = list(EntityListFilter::default(), None);
    assert_eq!(all.total, 6);
    assert_eq!(
        all.items
            .iter()
            .map(|item| item.entity.fsm.id)
            .collect::<Vec<_>>(),
        expected_ids
    );

    // Pagination follows the merged ranking, including ties between entity types.
    let page = list(
        EntityListFilter::default(),
        Some(PageParams { max: 2, page: 1 }),
    );
    assert_eq!(page.total, 6);
    assert_eq!(
        page.items
            .iter()
            .map(|item| item.entity.fsm.id)
            .collect::<Vec<_>>(),
        expected_ids[2..4]
    );

    // Host memory belongs to the destination batch; the channel belongs to copies.
    for (resource, expected_ids) in [
        (16, vec![id(50)]),
        (
            COPY_CHANNEL,
            list(
                EntityListFilter {
                    entity_type_name: Some("batch_clone".into()),
                    ..Default::default()
                },
                None,
            )
            .items
            .into_iter()
            .map(|item| item.entity.fsm.id)
            .collect(),
        ),
    ] {
        let scoped = list(
            EntityListFilter {
                scope: Some(EntityScope::Resource {
                    resource_id: id(resource),
                }),
                ..Default::default()
            },
            None,
        );
        assert_eq!(scoped.total as usize, expected_ids.len());
        assert_eq!(
            scoped
                .items
                .iter()
                .map(|item| item.entity.fsm.id)
                .collect::<Vec<_>>(),
            expected_ids
        );
    }
}
