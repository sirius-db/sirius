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
        ev(4, 10, E::Query(s::QueryEvent::Exit { seq: 3 })),
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
        ev(13, 8, E::DataBatch(s::DataBatchEvent::Exit { seq: 3 })),
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
        ev(14, 8, E::Task(s::TaskEvent::Exit { seq: 6 })),
    ];
    events.sort_by_key(|e| e.timestamp);
    events
}

#[test]
fn legacy_shared_manager_uses_group() {
    let mut events = fixture();
    events.push(ev(
        15,
        0,
        E::ThreadGroup(s::ThreadGroupEvent::Declaration {
            label: "shared-thread-group".into(),
            worker_id: EntityRef::new(id(2), ()),
            gpu_device_id: None,
        }),
    ));
    events.push(ev(
        16,
        0,
        E::TaskManagerLoopThread(s::TaskManagerLoopThreadEvent::Spawned {
            label: "task-scheduler-thread".into(),
            group_id: EntityRef::new(Uuid::nil(), ()),
        }),
    ));

    let analyzer = SiriusUiAnalyzer::try_new(id(1), events.into_iter()).unwrap();
    let thread = analyzer
        .model
        .task_manager_loop_threads
        .get(&id(16))
        .unwrap();
    assert_eq!(
        quent_analyzer::RefTreeEntity::parent_id(thread),
        Some(id(15))
    );
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
                E::DataBatch(s::DataBatchEvent::Destructed { .. } | s::DataBatchEvent::Exit { .. })
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
        ev(13, 8, E::DataBatch(s::DataBatchEvent::Exit { seq: 5 })),
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
        ev(15, 8, E::DataBatch(s::DataBatchEvent::Exit { seq: 3 })),
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
        ev(43, 8, E::DataBatch(s::DataBatchEvent::Exit { seq: 3 })),
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
fn exit_before_declaration_is_not_a_duplicate() {
    let mut events = fixture();
    events.insert(
        0,
        ev(12, 9, E::ExecutorThread(s::ExecutorThreadEvent::Exit)),
    );
    assert!(SiriusUiAnalyzer::try_new(id(1), events.into_iter()).is_ok());
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
fn resource_group_types_are_snake_case() {
    let model = SiriusUiAnalyzer::try_new(id(1), fixture().into_iter())
        .unwrap()
        .model;
    assert!(model.resource_group_types.contains_key("gpu_device"));
    assert!(model.resource_group_types.contains_key("thread_group"));
    assert!(!model.resource_group_types.contains_key("GpuDevice"));
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
        ev(20, 10, E::Query(s::QueryEvent::Exit { seq: 3 })),
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
        ev(23, 8, E::DataBatch(s::DataBatchEvent::Exit { seq: 3 })),
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
    use quent_store::event::{ModelEventStore, filesystem::Store};
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
            .exit();
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
        engine.exit().unwrap();
        (context_id, engine_id, query_id)
    };

    let stored = Store::<s::Sirius>::new(output.path())
        .events(context_id)
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(stored.len(), 9);

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
    let engine_events = crate::Viewer::import_events(&context_dir).unwrap();
    let worker_events = crate::Viewer::import_events(&worker_dir).unwrap();
    let combined =
        SiriusUiAnalyzer::try_new(engine_id, engine_events.chain(worker_events)).unwrap();
    assert_eq!(combined.model.workers.len(), 2);
}
