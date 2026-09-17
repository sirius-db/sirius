// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Temporary analysis adapters for generated schema entities.

// TODO(johanpel): Generate this module from schema metadata. See
// https://github.com/rapidsai/quent/issues/288.

use quent_analyzer::{
    AnalyzerResult, Entity, RefTreeEntity,
    entity::native::{AnalyzedEntity, EntityEventAccumulator},
    fsm::{
        Fsm, FsmUsages,
        native::{AnalyzedFsm, AnalyzedFsmBuilder, AnalyzedTransition},
    },
    resource::{Usage, Using},
};
use quent_events::Event;
use quent_query_engine_analyzer::{
    EngineEntity, OperatorEntity, OperatorEntityMut, PlanEntity, PortEntity, QueryEntity,
    QueryGroupEntity, WorkerEntity,
};
use quent_query_engine_ui as query_engine_ui;
use quent_time::{TimeUnixNanoSec, Timestamp, span::SpanUnixNanoSec, try_to_secs_relative};
use sirius_telemetry_store as schema;
use uuid::Uuid;

mod batch_placement;
mod channel;
mod data_batch;
mod engine;
mod executor_thread;
mod gpu_device;
mod memory;
mod memory_tier;
mod operator;
mod plan;
mod port;
mod query;
mod query_group;
mod task;
mod task_manager_loop_thread;
mod task_queue;
mod thread_group;
mod worker;

pub(crate) use batch_placement::BatchPlacementBuilder;
pub use batch_placement::{BatchPlacement, BatchPlacementExt};
pub(crate) use channel::Channel;
pub(crate) use data_batch::DataBatchBuilder;
pub use data_batch::{DataBatch, DataBatchExt};
pub use engine::Engine;
pub(crate) use executor_thread::ExecutorThread;
pub(crate) use gpu_device::GpuDevice;
pub(crate) use memory::Memory;
pub(crate) use memory_tier::MemoryTier;
pub use operator::Operator;
pub use plan::Plan;
pub use port::Port;
pub use query::Query;
pub(crate) use query::QueryBuilder;
pub use query_group::QueryGroup;
pub(crate) use task::TaskBuilder;
pub use task::{Task, TaskExt};
pub(crate) use task_manager_loop_thread::TaskManagerLoopThread;
pub(crate) use task_queue::TaskQueue;
pub(crate) use thread_group::ThreadGroup;
pub use worker::Worker;
