/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "duckdb/common/common.hpp"
#include "log/logging.hpp"
#include "query_id.hpp"
#include "telemetry-bridge/gen/context.rs.h"
#include "telemetry-bridge/gen/engine.rs.h"
#include "telemetry-bridge/gen/executor_thread.rs.h"
#include "telemetry-bridge/gen/gpu_device.rs.h"
#include "telemetry-bridge/gen/query_group.rs.h"
#include "telemetry-bridge/gen/task_manager_loop_thread.rs.h"
#include "telemetry-bridge/gen/task_queue.rs.h"
#include "telemetry-bridge/gen/thread_group.rs.h"
#include "telemetry-bridge/gen/uuid.rs.h"
#include "telemetry-bridge/gen/worker.rs.h"
#include "telemetry/memory_context.hpp"

#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace sirius::pipeline {
class sirius_pipeline;
}  // namespace sirius::pipeline

namespace sirius {
struct telemetry_config;
}  // namespace sirius

namespace sirius::telemetry {

/// Owns the top-level telemetry states for a single SiriusContext.
class telemetry_context {
 public:
  /// `gpu_device_ids` declares one per-GPU resource group (plus per-thread-type
  /// child buckets) under the engine, so thread telemetry can nest per device.
  [[nodiscard]] static std::shared_ptr<const telemetry_context> create(
    const sirius::telemetry_config& config,
    const cucascade::memory::memory_reservation_manager* manager = nullptr,
    const std::vector<int>& gpu_device_ids                       = {});

  ~telemetry_context();

  // Non-copyable, non-movable (owns opaque Rust boxes)
  telemetry_context(const telemetry_context&)            = delete;
  telemetry_context& operator=(const telemetry_context&) = delete;
  telemetry_context(telemetry_context&&)                 = delete;
  telemetry_context& operator=(telemetry_context&&)      = delete;

  [[nodiscard]] const uuid::UUID& engine_id() const { return engine_uuid_; }
  [[nodiscard]] const uuid::UUID& worker_id() const { return worker_uuid_; }
  /// The single, session-scoped query group that every query in this context is reported under.
  [[nodiscard]] const uuid::UUID& query_group_id() const { return query_group_uuid_; }
  /// The `{engine}-{session_label}` query group, declared on first use; empty or
  /// nullopt falls back to the default session-scoped group. Thread-safe.
  [[nodiscard]] uuid::UUID query_group_id_for(
    const std::optional<std::string>& session_label) const;
  /// The `gpu-N` device group for `device_id`; falls back to the engine group
  /// (with a warning) when the device was not declared at creation time.
  [[nodiscard]] const uuid::UUID& gpu_device_group_id(int device_id) const;
  /// The `executor_thread` bucket group under `gpu-N` (engine fallback as above).
  [[nodiscard]] const uuid::UUID& executor_thread_group_id(int device_id) const;
  /// The `task_manager_loop_thread` bucket group under `gpu-N` (engine fallback as above).
  [[nodiscard]] const uuid::UUID& manager_thread_group_id(int device_id) const;
  /// The `shared` group under the engine, for threads with no single GPU.
  [[nodiscard]] const uuid::UUID& shared_group_id() const { return shared_group_uuid_; }
  [[nodiscard]] const quent::Context& context() const { return *context_; }
  [[nodiscard]] const std::shared_ptr<const memory_context>& get_memory_context() const
  {
    return memory_context_;
  }

 private:
  telemetry_context(const sirius::telemetry_config& config,
                    const cucascade::memory::memory_reservation_manager* manager,
                    const std::vector<int>& gpu_device_ids);

  /// Telemetry group ids for one GPU: the `gpu-N` group and its per-thread-type
  /// child buckets.
  struct gpu_device_group_ids {
    uuid::UUID device;
    uuid::UUID executor_threads;
    uuid::UUID manager_threads;
  };

  uuid::UUID engine_uuid_;
  uuid::UUID worker_uuid_;
  uuid::UUID query_group_uuid_;
  uuid::UUID shared_group_uuid_;
  std::string engine_name_;
  mutable std::mutex labeled_groups_mutex_;
  mutable std::map<std::string, uuid::UUID> labeled_group_ids_;
  std::map<int, gpu_device_group_ids> gpu_group_ids_;
  rust::Box<quent::Context> context_;
  rust::Box<quent::engine::EngineObserver> engine_observer_;
  rust::Box<quent::worker::WorkerObserver> worker_observer_;
  rust::Box<quent::query_group::QueryGroupObserver> query_group_observer_;
  std::shared_ptr<const memory_context> memory_context_;
};

// A POD to hold common identifiers for useful telemetry.
struct query_telemetry_info {
  /// Quent's own UUID for this query (`QueryHandle::uuid()`), authoritative within telemetry.
  uuid::UUID telemetry_query_id;
  uuid::UUID worker_id;
  /// The engine-wide numeric query id (the execution window's id).
  sirius::query_id_t query_id;
};

/// Emit plan-level telemetry (operator declarations, port declarations, edges)
/// for the given set of pipelines. Called once during query construction.
void emit_plan_telemetry(
  const quent::Context& context,
  const duckdb::vector<duckdb::shared_ptr<pipeline::sirius_pipeline>>& pipelines,
  uuid::UUID plan_id,
  query_telemetry_info telemetry_info);

struct ExecutorThreadHandleWrapper {
  ExecutorThreadHandleWrapper(const telemetry_context& context,
                              const std::string& thread_name,
                              const uuid::UUID& parent_group_id)
    : handle(quent::executor_thread::create(context.context(),
                                            {
                                              .instance_name   = thread_name,
                                              .parent_group_id = parent_group_id,
                                            }))
  {
    handle->operating();
  }

  ExecutorThreadHandleWrapper(const ExecutorThreadHandleWrapper&)            = delete;
  ExecutorThreadHandleWrapper& operator=(const ExecutorThreadHandleWrapper&) = delete;
  ExecutorThreadHandleWrapper(ExecutorThreadHandleWrapper&&)                 = delete;
  ExecutorThreadHandleWrapper& operator=(ExecutorThreadHandleWrapper&&)      = delete;

  ~ExecutorThreadHandleWrapper()
  {
    handle->finalizing();
    handle->exit();
  }

  rust::Box<quent::executor_thread::ExecutorThreadHandle> handle;
};

struct TaskManagerLoopThreadHandleWrapper {
  TaskManagerLoopThreadHandleWrapper(const telemetry_context& context,
                                     const std::string& thread_name,
                                     const uuid::UUID& parent_group_id)
    : handle(quent::task_manager_loop_thread::create(context.context(),
                                                     {
                                                       .instance_name   = thread_name,
                                                       .parent_group_id = parent_group_id,
                                                     }))
  {
    handle->operating();
  }

  TaskManagerLoopThreadHandleWrapper(const TaskManagerLoopThreadHandleWrapper&)            = delete;
  TaskManagerLoopThreadHandleWrapper& operator=(const TaskManagerLoopThreadHandleWrapper&) = delete;
  TaskManagerLoopThreadHandleWrapper(TaskManagerLoopThreadHandleWrapper&&)                 = delete;
  TaskManagerLoopThreadHandleWrapper& operator=(TaskManagerLoopThreadHandleWrapper&&)      = delete;

  ~TaskManagerLoopThreadHandleWrapper()
  {
    handle->finalizing();
    handle->exit();
  }

  rust::Box<quent::task_manager_loop_thread::TaskManagerLoopThreadHandle> handle;
};

struct TaskQueueHandleWrapper {
  TaskQueueHandleWrapper(const telemetry_context& context,
                         const std::string& queue_name,
                         const uuid::UUID& parent_group_id)
    : handle(quent::task_queue::create(context.context(),
                                       {
                                         .instance_name   = queue_name,
                                         .parent_group_id = parent_group_id,
                                       }))
  {
    handle->operating({
      .capacity_entries = std::numeric_limits<uint64_t>::max(),
    });
  }

  TaskQueueHandleWrapper(const TaskQueueHandleWrapper&)            = delete;
  TaskQueueHandleWrapper& operator=(const TaskQueueHandleWrapper&) = delete;
  TaskQueueHandleWrapper(TaskQueueHandleWrapper&&)                 = delete;
  TaskQueueHandleWrapper& operator=(TaskQueueHandleWrapper&&)      = delete;

  ~TaskQueueHandleWrapper()
  {
    handle->finalizing();
    handle->exit();
  }

  rust::Box<quent::task_queue::TaskQueueHandle> handle;
};

// header-only shared thread-local storage handle: one per thread, shared across translation units
inline thread_local std::optional<ExecutorThreadHandleWrapper> executor_thread_telemetry_handle{
  std::nullopt};

// Initialize the thread local ExecutorThreadHandleWrapper for this worker thread.
inline void thread_local_executor_thread_telemtry_init(const telemetry_context& context,
                                                       const std::string& thread_name,
                                                       const uuid::UUID& parent_group_id)
{
  if (executor_thread_telemetry_handle.has_value()) {
    SIRIUS_LOG_WARN("ExecutorThreadHandleWrapper was already initialized; overriding.");
  }
  executor_thread_telemetry_handle.emplace(context, thread_name, parent_group_id);
}

}  // namespace sirius::telemetry
