// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for Batch ingestion and the data-flow distribution timeline.

use instrumentation_model::SiriusEvent;
use instrumentation_model::batch::{
    BatchConsumed, BatchPackaged, BatchProcessing, BatchQueued, BatchRegistered, BatchTransition,
    MemoryTier, MemoryTierFinalizing, MemoryTierInitializing, MemoryTierOperating,
    MemoryTierTransition,
};
use quent_analyzer::{AnalyzerError, resource::collection::ResourceCollection};
use quent_events::Event;
use quent_model::{Capacity, FsmEvent, Ref, Usage};
use quent_query_engine_analyzer::ui::UiAnalyzer;
use quent_query_engine_model::{engine, operator, plan, query, query_group, worker};
use quent_query_engine_ui::{OperatorFilter, QueryFilter, data_flow::DataFlowTimelineResponse};
use quent_ui::entities::request::{
    EntityListEntry, EntityListFilter, EntityListRequest, EntityScope, EntitySortKey, Sort,
    SortDir, TimeWindow,
};
use quent_ui::timeline::{
    distribution::DistributionTimelineRequest,
    request::{
        EntityFilter, ResourceTimelineRequest, SingleTimelineRequest, TimelineConfig,
        TimelineRequest,
    },
    response::ResourceTimeline as UiResourceTimeline,
};
use uuid::Uuid;

use crate::{SiriusUiAnalyzer, batch::BatchExt};

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

fn tier_usage(resource_id: Uuid, bytes: u64) -> Option<Usage<MemoryTier>> {
    Some(Usage {
        resource_id: Ref::new(resource_id),
        capacity: MemoryTierOperating {
            capacity_bytes: Capacity::new(Some(bytes)),
        },
    })
}

fn batch_event(id: Uuid, ts: u64, seq: u64, state: BatchTransition) -> Event<SiriusEvent> {
    Event::new(id, ts, SiriusEvent::Batch(FsmEvent { seq, state }))
}

fn memory_tier_events(id: Uuid, parent: Uuid, name: &str, bytes: u64) -> Vec<Event<SiriusEvent>> {
    let event = |ts: u64, seq: u64, state: MemoryTierTransition| {
        Event::new(id, ts, SiriusEvent::MemoryTier(FsmEvent { seq, state }))
    };
    vec![
        event(
            EPOCH - 800,
            0,
            MemoryTierTransition::MemoryTierInitializing(MemoryTierInitializing {
                instance_name: name.to_string(),
                parent_group_id: parent,
                resource_type_name: "memory_tier".to_string(),
            }),
        ),
        event(
            EPOCH - 700,
            1,
            MemoryTierTransition::MemoryTierOperating(MemoryTierOperating {
                capacity_bytes: Capacity::new(Some(bytes)),
            }),
        ),
        event(
            EPOCH + 100_000,
            2,
            MemoryTierTransition::MemoryTierFinalizing(MemoryTierFinalizing),
        ),
        event(EPOCH + 100_001, 3, MemoryTierTransition::Exit),
    ]
}

/// A minimal engine with one query (two operators), the three memory tiers,
/// and optionally batch placements:
///
/// Batch A (pipeline op1, batch_id 7, 1000 bytes), timestamps relative to the
/// query epoch:
/// - t=0    registered on GPU (100ns span: pins the analyzer's explicit
///          omission of the bookkeeping entry state from aggregates)
/// - t=100  queued on GPU
/// - t=300  queued on HOST (tier-change self-transition: spill)
/// - t=500  packaged (task 1) on HOST
/// - t=800  processing (task 1) on GPU
/// - t=1000 consumed ("processed"), exit
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
            SiriusEvent::Engine(engine::EngineEvent::Init(engine::Init::default())),
        ),
        Event::new(
            query_group_id,
            EPOCH - 900,
            SiriusEvent::QueryGroup(query_group::QueryGroupEvent::Declaration(
                query_group::Declaration {
                    instance_name: "qg".to_string(),
                    engine_id,
                },
            )),
        ),
    ];
    events.extend(memory_tier_events(gpu_id, engine_id, "GPU", 1 << 30));
    events.extend(memory_tier_events(host_id, engine_id, "HOST", 4 << 30));
    events.extend(memory_tier_events(disk_id, engine_id, "DISK", 16 << 30));

    // The query FSM: its first transition is the query epoch.
    let query_event = |ts: u64, seq: u64, state: query::QueryTransition| {
        Event::new(query_id, ts, SiriusEvent::Query(FsmEvent { seq, state }))
    };
    events.extend([
        query_event(
            EPOCH,
            0,
            query::QueryTransition::Init(query::Init {
                instance_name: "q".to_string(),
                query_group_id: Ref::new(query_group_id),
            }),
        ),
        query_event(
            EPOCH + 10,
            1,
            query::QueryTransition::Planning(query::Planning {}),
        ),
        query_event(
            EPOCH + 20,
            2,
            query::QueryTransition::Executing(query::Executing {}),
        ),
        query_event(EPOCH + 50_000, 3, query::QueryTransition::Exit),
    ]);

    events.push(Event::new(
        worker_id,
        EPOCH - 950,
        SiriusEvent::Worker(worker::WorkerEvent::Init(worker::Init {
            parent_engine_id: Ref::new(engine_id),
            instance_name: "worker".to_string(),
        })),
    ));
    events.push(Event::new(
        plan_id,
        EPOCH + 5,
        SiriusEvent::Plan(plan::PlanEvent::Declaration(plan::Declaration {
            parent: plan::PlanParent {
                query_id: Some(Ref::new(query_id)),
                plan_id: None,
            },
            instance_name: "plan".to_string(),
            edges: vec![],
            worker_id: Some(Ref::new(worker_id)),
        })),
    ));
    for (op_id, name) in [(op1_id, "op1"), (op2_id, "op2")] {
        events.push(Event::new(
            op_id,
            EPOCH + 6,
            SiriusEvent::Operator(operator::OperatorEvent::Declaration(operator::Declaration {
                plan_id: Ref::new(plan_id),
                parent_operator_ids: vec![],
                instance_name: name.to_string(),
                type_name: "scan".to_string(),
                custom_attributes: Default::default(),
            })),
        ));
    }

    if with_batches {
        let port_id = Uuid::from_u128(0x0d);
        events.extend([
            batch_event(
                batch_a_id,
                EPOCH,
                0,
                BatchTransition::BatchRegistered(BatchRegistered {
                    instance_name: "batch 7".to_string(),
                    batch_id: 7,
                    pipeline_uuid: op1_id,
                    port_uuid: port_id,
                    origin: "operator_output".to_string(),
                    tier: tier_usage(gpu_id, 1000),
                }),
            ),
            batch_event(
                batch_a_id,
                EPOCH + 100,
                1,
                BatchTransition::BatchQueued(BatchQueued {
                    tier: tier_usage(gpu_id, 1000),
                }),
            ),
            // Tier change while queued: spill from GPU to HOST.
            batch_event(
                batch_a_id,
                EPOCH + 300,
                2,
                BatchTransition::BatchQueued(BatchQueued {
                    tier: tier_usage(host_id, 1000),
                }),
            ),
            batch_event(
                batch_a_id,
                EPOCH + 500,
                3,
                BatchTransition::BatchPackaged(BatchPackaged {
                    instance_name: "batch 7".to_string(),
                    task_uuid: task_1_id,
                    tier: tier_usage(host_id, 1000),
                }),
            ),
            batch_event(
                batch_a_id,
                EPOCH + 800,
                4,
                BatchTransition::BatchProcessing(BatchProcessing {
                    instance_name: "batch 7".to_string(),
                    task_uuid: task_1_id,
                    tier: tier_usage(gpu_id, 1000),
                }),
            ),
            batch_event(
                batch_a_id,
                EPOCH + 1000,
                5,
                BatchTransition::BatchConsumed(BatchConsumed {
                    instance_name: "batch 7".to_string(),
                    reason: "processed".to_string(),
                }),
            ),
            batch_event(batch_a_id, EPOCH + 1000, 6, BatchTransition::Exit),
        ]);

        // A batch on a pipeline outside the query.
        let foreign_pipeline = Uuid::from_u128(0x0e);
        events.extend([
            batch_event(
                batch_b_id,
                EPOCH,
                0,
                BatchTransition::BatchRegistered(BatchRegistered {
                    instance_name: "batch 8".to_string(),
                    batch_id: 8,
                    pipeline_uuid: foreign_pipeline,
                    port_uuid: port_id,
                    origin: "operator_output".to_string(),
                    tier: tier_usage(disk_id, 500),
                }),
            ),
            batch_event(
                batch_b_id,
                EPOCH,
                1,
                BatchTransition::BatchQueued(BatchQueued {
                    tier: tier_usage(disk_id, 500),
                }),
            ),
            batch_event(
                batch_b_id,
                EPOCH + 900,
                2,
                BatchTransition::BatchConsumed(BatchConsumed {
                    instance_name: "batch 8".to_string(),
                    reason: "query_end".to_string(),
                }),
            ),
            batch_event(batch_b_id, EPOCH + 900, 3, BatchTransition::Exit),
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

fn analyzer(fixture: &mut Fixture) -> SiriusUiAnalyzer {
    let events = std::mem::take(&mut fixture.events);
    SiriusUiAnalyzer::try_new(fixture.engine_id, events.into_iter())
        .expect("analyzer builds from fixture events")
}

/// A request for the full query window [0, 1000) ns in 10 bins of 100 ns.
fn request(query_id: Uuid, measures: &[&str]) -> DistributionTimelineRequest<QueryFilter> {
    DistributionTimelineRequest {
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

    assert_eq!(model.batches.len(), 2);
    let batch_a = &model.batches[&fixture.batch_a_id];
    assert_eq!(batch_a.batch_id(), Some(7));
    assert_eq!(batch_a.pipeline_uuid(), Some(fixture.op1_id));
    assert_eq!(batch_a.task_uuid(), Some(fixture.task_1_id));

    // The memory tiers are modeled as resources of type "memory_tier" with
    // their tier name as instance name, marked as used by batches.
    for (id, name) in [
        (fixture.gpu_id, "GPU"),
        (fixture.host_id, "HOST"),
        (fixture.disk_id, "DISK"),
    ] {
        let resource = model.resource(id).expect("tier resource exists");
        assert_eq!(resource.type_name(), "memory_tier");
        assert_eq!(resource.instance_name(), name);
    }
    assert!(
        model.arbitrary_resources.resource_types["memory_tier"]
            .used_by
            .contains("batch")
    );
}

#[test]
fn data_flow_timeline_bins() {
    let mut fixture = fixture(true);
    let analyzer = analyzer(&mut fixture);

    let response = analyzer
        .data_flow_timeline(request(fixture.query_id, &[]))
        .expect("data flow timeline");
    let DataFlowTimelineResponse::Binned(binned) = response else {
        panic!("expected a binned response");
    };

    assert_eq!(binned.decl.entity_type_name, "batch");
    assert_eq!(binned.decl.dimension_name, "Memory Tier");
    let keys: Vec<&str> = binned
        .decl
        .dimension_keys
        .iter()
        .map(|k| k.key.as_str())
        .collect();
    assert_eq!(keys, ["GPU", "HOST", "DISK"]);
    let measures: Vec<&str> = binned.decl.measures.iter().map(|m| m.name.as_str()).collect();
    assert_eq!(measures, ["count", "bytes"]);

    // Only op1 has batches within the query; the foreign-pipeline batch is
    // excluded, and op2 never appears since absent series mean all-zero.
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
    // batch_registered spans a real 100ns in this fixture but is explicitly
    // omitted by the analyzer (bookkeeping entry state); batch_consumed holds
    // no tier residency.
    assert!(!count.contains_key("batch_registered"));
    assert!(!count.contains_key("batch_consumed"));

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
fn data_flow_timeline_unsupported_without_batches() {
    let mut fixture = fixture(false);
    let analyzer = analyzer(&mut fixture);

    let response = analyzer
        .data_flow_timeline(request(fixture.query_id, &[]))
        .expect("data flow timeline");
    assert!(matches!(response, DataFlowTimelineResponse::Unsupported));
}

#[test]
fn data_flow_timeline_measures_filter() {
    let mut fixture = fixture(true);
    let analyzer = analyzer(&mut fixture);

    // Only "bytes": the count measure is neither declared nor computed.
    let response = analyzer
        .data_flow_timeline(request(fixture.query_id, &["bytes"]))
        .expect("data flow timeline");
    let DataFlowTimelineResponse::Binned(binned) = response else {
        panic!("expected a binned response");
    };
    let measures: Vec<&str> = binned.decl.measures.iter().map(|m| m.name.as_str()).collect();
    assert_eq!(measures, ["bytes"]);
    let series = &binned.operators[&fixture.op1_id];
    assert!(series.values.contains_key("bytes"));
    assert!(!series.values.contains_key("count"));

    // Unknown-only measures are an error.
    let error = analyzer
        .data_flow_timeline(request(fixture.query_id, &["bogus"]))
        .expect_err("unknown measures are rejected");
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
            application: OperatorFilter { operator_id: None },
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
fn batch_keyed_timeline_over_memory_tier_resource() {
    let mut fixture = fixture(true);
    let analyzer = analyzer(&mut fixture);

    // Per-state timeline of the GPU tier resource sliced by the batch FSM.
    let response = analyzer
        .single_resource_timeline(single_timeline_request(
            fixture.query_id,
            fixture.gpu_id,
            Some("batch"),
        ))
        .expect("batch keyed timeline over a memory_tier resource");
    let UiResourceTimeline::BinnedByState(by_state) = response.data else {
        panic!("expected a per-state binned response");
    };

    let states = &by_state.capacities_states_values["capacity_bytes"];
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
        binned.capacities_values["capacity_bytes"],
        [1000.0, 1000.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 1000.0]
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
                application: OperatorFilter { operator_id: None },
            },
            app_params: QueryFilter {
                query_id: fixture.query_id,
            },
        })
        .expect("list entities scoped to a memory_tier resource");
    assert_eq!(response.total, 0);
    assert!(response.items.is_empty());
}
