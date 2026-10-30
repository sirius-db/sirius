//! Generated-schema ingestion and UI response tests.

use quent_analyzer::{Entity, Model};
use quent_events::{EntityRef, Event};
use quent_query_engine_analyzer::{QueryEngineModel, QueryEntity, ui::UiAnalyzer};
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

        let details = context.observer::<instrumentation::QueryDetails>().handle();
        details
            .update(
                query.as_entity_ref(),
                1,
                instrumentation::QueryMetadata {
                    instance_name: Some("ndjson early label".into()),
                    connection_id: Some(17),
                    query_ordinal: Some(42),
                    planning_generation: None,
                    execution_query_id: None,
                    worker_id: None,
                    disposition: None,
                    outcome: None,
                },
            )
            .unwrap();
        details
            .update(
                query.as_entity_ref(),
                2,
                instrumentation::QueryMetadata {
                    instance_name: Some("ndjson late label".into()),
                    connection_id: Some(17),
                    query_ordinal: Some(42),
                    planning_generation: Some(3),
                    execution_query_id: Some(123),
                    worker_id: Some(worker.as_entity_ref()),
                    disposition: Some("gpu".into()),
                    outcome: Some("success".into()),
                },
            )
            .unwrap();

        (context_id, engine_id, query_id)
    };

    let stored = Loader::<s::Sirius>::new(output.path(), ContextSet::one(context_id))
        .combined_events()
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(stored.len(), 10);

    let context_dir = output.path().join(context_id.to_string());
    let inventory = crate::Viewer::context_inventory(&context_dir).unwrap();
    assert!(inventory.analysis_target_ids.contains(&engine_id));
    let imported: ViewerEventStream<SiriusUiAnalyzer> =
        crate::Viewer::import_events(&context_dir).unwrap();
    let analyzer = SiriusUiAnalyzer::try_new(engine_id, imported).unwrap();
    let bundle = analyzer.query_bundle(query_id).unwrap();
    assert_eq!(bundle.query_id, query_id);
    assert_eq!(
        bundle.entities.query.instance_name.as_deref(),
        Some("ndjson late label")
    );

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
        10
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

fn query_metadata(name: Option<&str>, outcome: Option<&str>) -> s::QueryMetadata {
    s::QueryMetadata {
        instance_name: name.map(str::to_owned),
        connection_id: Some(17),
        query_ordinal: Some(42),
        planning_generation: Some(3),
        execution_query_id: None,
        worker_id: None,
        disposition: Some("cpu_only".into()),
        outcome: outcome.map(str::to_owned),
    }
}

fn metadata_event(revision: u64, metadata: s::QueryMetadata) -> Event<E> {
    ev(
        100,
        20,
        E::QueryDetails(s::QueryDetailsEvent::Update {
            query_id: EntityRef::new(id(4), ()),
            revision,
            metadata,
        }),
    )
}

fn planless_fixture() -> Vec<Event<E>> {
    // The engine may have GPU workers/resources from other queries. None of
    // those become associated with this query just because it lacks a plan.
    fixture()
        .into_iter()
        .filter(|event| match &event.data {
            E::Plan(_) | E::Operator(_) | E::Port(_) | E::Task(_) | E::DataBatch(_) => false,
            E::Query(data) => matches!(data, s::QueryEvent::Init { .. }),
            _ => true,
        })
        .collect()
}

#[test]
fn query_completion_uses_finished_and_preserves_incomplete_trace_policy() {
    for (planning, executing, finished) in [
        (false, false, true),
        (true, false, true),
        (true, true, true),
        (false, false, false),
        (true, false, false),
        (true, true, false),
    ] {
        let mut events = planless_fixture();
        if planning {
            events.push(ev(4, 1, E::Query(s::QueryEvent::Planning { seq: 1 })));
        }
        if executing {
            events.push(ev(4, 2, E::Query(s::QueryEvent::Executing { seq: 2 })));
        }
        if finished {
            events.push(ev(4, 4, E::Query(s::QueryEvent::Finished { seq: 3 })));
        }
        let analyzer = SiriusUiAnalyzer::try_new(id(1), events.into_iter()).unwrap();
        if !finished {
            // Preserve the pinned FSM analyzer's truncated-trace policy: omit
            // unfinished queries rather than inventing a terminal transition.
            assert!(analyzer.model.query(id(4)).is_err());
            assert!(analyzer.query_bundle(id(4)).is_err());
            continue;
        }
        let query = analyzer.model.query(id(4)).unwrap().to_ui().unwrap();
        assert_eq!(query.planning_s, planning.then_some(1.0));
        assert_eq!(query.executing_s, executing.then_some(2.0));
        assert_eq!(query.completed_s, finished.then_some(4.0));
        assert!(
            query
                .custom_attributes
                .iter()
                .all(|attr| attr.key != "outcome")
        );
        assert_eq!(analyzer.model.query_plans(id(4)).unwrap().count(), 0);
        assert_eq!(analyzer.model.query_workers(id(4)).unwrap().count(), 0);
        let view = analyzer.model.query_view(id(4)).unwrap();
        assert_eq!(view.query_plans(id(4)).unwrap().count(), 0);
        assert_eq!(view.query_workers(id(4)).unwrap().count(), 0);
        let bundle = analyzer.query_bundle(id(4)).unwrap();
        assert_eq!(bundle.duration_s, 4.0);
        assert_eq!(bundle.entities.plans.len(), 1);
        let placeholder = &bundle.entities.plans[&bundle.plan_tree.id];
        assert_eq!(
            placeholder.instance_name.as_deref(),
            Some("No execution plan - query lifecycle only")
        );
        assert!(placeholder.worker_id.is_none());
        assert!(placeholder.edges.is_empty());
        assert!(bundle.plan_tree.worker.is_none());
        assert!(bundle.entities.workers.is_empty());
        assert!(bundle.entities.operators.is_empty());
        assert!(bundle.entities.ports.is_empty());
        assert!(bundle.entities.resources.is_empty());
        assert!(analyzer.model.tasks.is_empty());
        assert!(analyzer.model.plans.is_empty());
        assert_eq!(
            bundle.plan_tree.id,
            analyzer.query_bundle(id(4)).unwrap().plan_tree.id
        );
    }
}

#[test]
fn late_metadata_uses_revision_without_changing_identity_group_or_completion() {
    let mut events = planless_fixture();
    events.push(ev(4, 1, E::Query(s::QueryEvent::Planning { seq: 1 })));
    events.push(ev(4, 4, E::Query(s::QueryEvent::Finished { seq: 2 })));
    // Metadata can arrive before Init, out of revision order, or after Finished.
    events.insert(
        0,
        metadata_event(2, query_metadata(Some("late label"), Some("failed"))),
    );
    events.push(metadata_event(1, query_metadata(Some("old label"), None)));
    let analyzer = SiriusUiAnalyzer::try_new(id(1), events.into_iter()).unwrap();
    let query = analyzer.model.query(id(4)).unwrap();
    assert_eq!(query.id(), id(4));
    assert_eq!(query.query_group_id(), Some(id(3)));
    let ui = query.to_ui().unwrap();
    assert_eq!(ui.instance_name.as_deref(), Some("late label"));
    assert_eq!(ui.completed_s, Some(4.0));
    assert_eq!(analyzer.query_bundle(id(4)).unwrap().duration_s, 4.0);
    let attrs = ui
        .custom_attributes
        .iter()
        .map(|attr| (attr.key.as_str(), &attr.value))
        .collect::<std::collections::HashMap<_, _>>();
    use quent_dynamic_attributes::DynamicValue;
    assert_eq!(
        attrs["query_group"],
        &Some(DynamicValue::String("group".into()))
    );
    assert_eq!(attrs["connection_id"], &Some(DynamicValue::U64(17)));
    assert_eq!(attrs["query_ordinal"], &Some(DynamicValue::U64(42)));
    assert_eq!(attrs["planning_generation"], &Some(DynamicValue::U64(3)));
    assert_eq!(
        attrs["outcome"],
        &Some(DynamicValue::String("failed".into()))
    );
    assert_eq!(
        attrs["disposition"],
        &Some(DynamicValue::String("cpu_only".into()))
    );
    assert!(!attrs.contains_key("execution_query_id"));
    // Keep a real bundle for manual/browser checks of the pinned viewer.
    if let Ok(path) = std::env::var("SIRIUS_PLANLESS_BUNDLE") {
        std::fs::write(
            path,
            serde_json::to_vec_pretty(&analyzer.query_bundle(id(4)).unwrap()).unwrap(),
        )
        .unwrap();
    }
}

#[test]
fn execution_metadata_does_not_turn_a_failed_gpu_attempt_into_a_failed_outer_query() {
    let mut events = fixture();
    for event in &mut events {
        if let E::Task(s::TaskEvent::Finalizing { success, .. }) = &mut event.data {
            *success = false;
        }
    }
    let mut metadata = query_metadata(Some("replayed"), Some("success"));
    metadata.execution_query_id = Some(123);
    metadata.worker_id = Some(EntityRef::new(id(2), ()));
    metadata.disposition = Some("gpu_to_cpu".into());
    events.push(metadata_event(1, metadata));
    let analyzer = SiriusUiAnalyzer::try_new(id(1), events.into_iter()).unwrap();
    let bundle = analyzer.query_bundle(id(4)).unwrap();
    assert_eq!(bundle.plan_tree.id, id(5));
    assert_eq!(bundle.entities.plans.len(), 1);
    let attrs = &bundle.entities.query.custom_attributes;
    assert!(attrs.iter().any(|attr| attr.key == "execution_query_id"
        && attr.value == Some(quent_dynamic_attributes::DynamicValue::U64(123))));
    assert!(attrs.iter().any(|attr| attr.key == "outcome"
        && attr.value
            == Some(quent_dynamic_attributes::DynamicValue::String(
                "success".into()
            ))));
    assert!(attrs.iter().any(|attr| attr.key == "disposition"
        && attr.value
            == Some(quent_dynamic_attributes::DynamicValue::String(
                "gpu_to_cpu".into()
            ))));
}

#[test]
fn query_metadata_rejects_invalid_revisions_and_multiple_owners() {
    let mut events = planless_fixture();
    events.push(metadata_event(0, query_metadata(None, None)));
    assert!(SiriusUiAnalyzer::try_new(id(1), events.into_iter()).is_err());
    let mut events = planless_fixture();
    events.push(metadata_event(1, query_metadata(None, None)));
    let mut other = metadata_event(2, query_metadata(None, None));
    other.id = id(101);
    events.push(other);
    assert!(SiriusUiAnalyzer::try_new(id(1), events.into_iter()).is_err());
}

#[test]
fn metadata_snapshots_replace_fields_and_keep_the_initial_name_as_fallback() {
    let mut events = planless_fixture();
    events.push(ev(4, 4, E::Query(s::QueryEvent::Finished { seq: 1 })));
    events.push(metadata_event(
        1,
        query_metadata(Some("previous label"), Some("failed")),
    ));
    events.push(metadata_event(
        2,
        s::QueryMetadata {
            instance_name: None,
            connection_id: None,
            query_ordinal: None,
            planning_generation: None,
            execution_query_id: None,
            worker_id: None,
            disposition: None,
            outcome: None,
        },
    ));
    let analyzer = SiriusUiAnalyzer::try_new(id(1), events.into_iter()).unwrap();
    let query = analyzer.model.query(id(4)).unwrap().to_ui().unwrap();
    assert_eq!(query.id, id(4));
    assert_eq!(query.instance_name.as_deref(), Some("select"));
    assert_eq!(query.custom_attributes.len(), 1);
    assert_eq!(query.custom_attributes[0].key, "query_group");
}

#[test]
fn a_malformed_real_plan_is_not_replaced_by_a_viewer_placeholder() {
    let mut events = fixture();
    for event in &mut events {
        if let E::Plan(s::PlanEvent::Declaration { worker_id, .. }) = &mut event.data {
            *worker_id = None;
        }
    }
    let analyzer = SiriusUiAnalyzer::try_new(id(1), events.into_iter()).unwrap();
    assert!(analyzer.query_bundle(id(4)).is_err());
    assert_eq!(analyzer.model.plans.len(), 1);
}

#[test]
fn query_list_hides_unnamed_observations_that_never_executed() {
    for (initial_name, late_name, executing, expected) in [
        ("unnamed_query", None, false, false),
        ("", None, false, false),
        ("  ", None, false, false),
        ("legacy label", None, false, true),
        ("unnamed_query", Some("late label"), false, true),
        ("unnamed_query", Some("unnamed_query"), false, true),
        ("unnamed_query", Some(""), false, false),
        ("unnamed_query", Some("  "), false, false),
        ("unnamed_query", None, true, true),
        ("unnamed_query", Some(""), true, true),
    ] {
        let mut events = planless_fixture();
        for event in &mut events {
            if let E::Query(s::QueryEvent::Init { instance_name, .. }) = &mut event.data {
                *instance_name = initial_name.into();
            }
        }
        events.push(ev(4, 1, E::Query(s::QueryEvent::Planning { seq: 1 })));
        if executing {
            events.push(ev(4, 2, E::Query(s::QueryEvent::Executing { seq: 2 })));
        }
        events.push(ev(
            4,
            4,
            E::Query(s::QueryEvent::Finished {
                seq: if executing { 3 } else { 2 },
            }),
        ));
        // A failed admitted attempt is not execution. A late explicit label still
        // keeps that planning-only failure visible, including after Finished.
        let mut metadata = query_metadata(late_name, Some("failed"));
        metadata.execution_query_id = Some(123);
        metadata.disposition = Some("gpu".into());
        events.push(metadata_event(1, metadata));
        let analyzer = SiriusUiAnalyzer::try_new(id(1), events.into_iter()).unwrap();
        let listed = analyzer
            .query_engine_model()
            .queries()
            .map(Entity::id)
            .collect::<Vec<_>>();
        assert_eq!(
            listed,
            if expected { vec![id(4)] } else { vec![] },
            "initial={initial_name:?}, late={late_name:?}, executing={executing}"
        );
        // Visibility never discards events/entities or prevents direct inspection.
        assert!(analyzer.model.query(id(4)).is_ok());
        assert!(analyzer.query_bundle(id(4)).is_ok());
    }
}

#[test]
fn query_list_visibility_uses_the_final_snapshot_not_any_previous_label() {
    let mut events = planless_fixture();
    for event in &mut events {
        if let E::Query(s::QueryEvent::Init { instance_name, .. }) = &mut event.data {
            *instance_name = "unnamed_query".into();
        }
    }
    events.push(ev(4, 1, E::Query(s::QueryEvent::Planning { seq: 1 })));
    events.push(ev(4, 4, E::Query(s::QueryEvent::Finished { seq: 2 })));
    // Highest revision clears the label. Out-of-order delivery of the older
    // label cannot make an unnamed planning-only query reappear in the catalog.
    events.push(metadata_event(2, query_metadata(None, Some("cancelled"))));
    events.push(metadata_event(1, query_metadata(Some("old"), None)));
    let analyzer = SiriusUiAnalyzer::try_new(id(1), events.into_iter()).unwrap();
    assert_eq!(analyzer.query_engine_model().queries().count(), 0);
    assert!(analyzer.query_bundle(id(4)).is_ok());
}
