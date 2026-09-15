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
#include "telemetry-bridge/gen/quent.hpp"
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

  ~telemetry_context() noexcept;

  // Non-copyable, non-movable (owns move-only Quent handles)
  telemetry_context(const telemetry_context&)            = delete;
  telemetry_context& operator=(const telemetry_context&) = delete;
  telemetry_context(telemetry_context&&)                 = delete;
  telemetry_context& operator=(telemetry_context&&)      = delete;

  [[nodiscard]] const quent::Uuid& engine_id() const { return engine_uuid_; }
  [[nodiscard]] const quent::Uuid& worker_id() const { return worker_uuid_; }
  /// The single, session-scoped query group that every query in this context is reported under.
  [[nodiscard]] const quent::Uuid& query_group_id() const { return query_group_uuid_; }
  /// The `{engine}-{session_label}` query group, declared on first use; empty or
  /// nullopt falls back to the default session-scoped group. Thread-safe.
  [[nodiscard]] quent::Uuid query_group_id_for(
    const std::optional<std::string>& session_label) const;
  /// The `gpu-N` device group for `device_id`; falls back to the engine group
  /// (with a warning) when the device was not declared at creation time.
  [[nodiscard]] const quent::Uuid& gpu_device_group_id(int device_id) const;
  /// The `executor_thread` bucket group under `gpu-N` (engine fallback as above).
  [[nodiscard]] const quent::Uuid& executor_thread_group_id(int device_id) const;
  /// The `task_manager_loop_thread` bucket group under `gpu-N` (engine fallback as above).
  [[nodiscard]] const quent::Uuid& manager_thread_group_id(int device_id) const;
  /// The `shared` group under the engine, for threads with no single GPU.
  [[nodiscard]] const quent::Uuid& shared_group_id() const { return shared_group_uuid_; }
  [[nodiscard]] const quent::Context& context() const { return context_; }
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
    quent::Uuid device;
    quent::Uuid executor_threads;
    quent::Uuid manager_threads;
  };

  quent::Uuid engine_uuid_;
  quent::Uuid worker_uuid_;
  quent::Uuid query_group_uuid_;
  quent::Uuid shared_group_uuid_;
  std::string engine_name_;
  mutable std::mutex labeled_groups_mutex_;
  mutable std::map<std::string, quent::Uuid> labeled_group_ids_;
  std::map<int, gpu_device_group_ids> gpu_group_ids_;
  quent::Context context_;
  std::shared_ptr<quent::engine::EngineObserver> engine_observer_;
  std::shared_ptr<quent::worker::WorkerObserver> worker_observer_;
  std::shared_ptr<quent::query_group::QueryGroupObserver> query_group_observer_;
  quent::Handle<quent::Engine> engine_handle_;
  quent::Handle<quent::Worker> worker_handle_;
  std::shared_ptr<const memory_context> memory_context_;
};

// A POD to hold common identifiers for useful telemetry.
struct query_telemetry_info {
  /// Quent's own UUID for this query (`FsmHandle::id()`), authoritative within telemetry.
  quent::Uuid telemetry_query_id;
  quent::Uuid worker_id;
  /// The engine-wide numeric query id (the execution window's id).
  sirius::query_id_t query_id;
};

/// Emit plan-level telemetry (operator declarations, port declarations, edges)
/// for the given set of pipelines. Called once during query construction.
void emit_plan_telemetry(
  const quent::Context& context,
  const duckdb::vector<duckdb::shared_ptr<pipeline::sirius_pipeline>>& pipelines,
  quent::Uuid plan_id,
  query_telemetry_info telemetry_info);

struct ExecutorThreadHandleWrapper {
  ExecutorThreadHandleWrapper(const telemetry_context& context,
                              const std::string& thread_name,
                              const quent::Uuid& parent_group_id)
    : handle(context.context().executor_thread_observer()->handle())
  {
    handle.declaration(quent::executor_thread::Declaration{
      .instance_name   = thread_name,
      .parent_group_id = parent_group_id,
      .engine_id       = quent::engine::EngineId(context.engine_id()),
    });
  }

  ExecutorThreadHandleWrapper(const ExecutorThreadHandleWrapper&)            = delete;
  ExecutorThreadHandleWrapper& operator=(const ExecutorThreadHandleWrapper&) = delete;
  ExecutorThreadHandleWrapper(ExecutorThreadHandleWrapper&&)                 = delete;
  ExecutorThreadHandleWrapper& operator=(ExecutorThreadHandleWrapper&&)      = delete;

  ~ExecutorThreadHandleWrapper() = default;

  quent::Handle<quent::ExecutorThread> handle;
};

struct TaskManagerLoopThreadHandleWrapper {
  TaskManagerLoopThreadHandleWrapper(const telemetry_context& context,
                                     const std::string& thread_name,
                                     const quent::Uuid& parent_group_id)
    : handle(context.context().task_manager_loop_thread_observer()->handle())
  {
    handle.declaration(quent::task_manager_loop_thread::Declaration{
      .instance_name   = thread_name,
      .parent_group_id = parent_group_id,
      .engine_id       = quent::engine::EngineId(context.engine_id()),
    });
  }

  TaskManagerLoopThreadHandleWrapper(const TaskManagerLoopThreadHandleWrapper&)            = delete;
  TaskManagerLoopThreadHandleWrapper& operator=(const TaskManagerLoopThreadHandleWrapper&) = delete;
  TaskManagerLoopThreadHandleWrapper(TaskManagerLoopThreadHandleWrapper&&)                 = delete;
  TaskManagerLoopThreadHandleWrapper& operator=(TaskManagerLoopThreadHandleWrapper&&)      = delete;

  ~TaskManagerLoopThreadHandleWrapper() = default;

  quent::Handle<quent::TaskManagerLoopThread> handle;
};

struct TaskQueueHandleWrapper {
  TaskQueueHandleWrapper(const telemetry_context& context,
                         const std::string& queue_name,
                         const quent::Uuid& parent_group_id)
    : handle(context.context().task_queue_observer()->handle())
  {
    handle.declaration(quent::task_queue::Declaration{
      .instance_name   = queue_name,
      .parent_group_id = parent_group_id,
      .engine_id       = quent::engine::EngineId(context.engine_id()),
      .bounds =
        quent::records::TaskQueueBounds{
          .entries = std::numeric_limits<uint64_t>::max(),
        },
    });
  }

  TaskQueueHandleWrapper(const TaskQueueHandleWrapper&)            = delete;
  TaskQueueHandleWrapper& operator=(const TaskQueueHandleWrapper&) = delete;
  TaskQueueHandleWrapper(TaskQueueHandleWrapper&&)                 = delete;
  TaskQueueHandleWrapper& operator=(TaskQueueHandleWrapper&&)      = delete;

  ~TaskQueueHandleWrapper() = default;

  quent::Handle<quent::TaskQueue> handle;
};

// header-only shared thread-local storage handle: one per thread, shared across translation units
inline thread_local std::optional<ExecutorThreadHandleWrapper> executor_thread_telemetry_handle{
  std::nullopt};

// Initialize the thread local ExecutorThreadHandleWrapper for this worker thread.
inline void thread_local_executor_thread_telemtry_init(const telemetry_context& context,
                                                       const std::string& thread_name,
                                                       const quent::Uuid& parent_group_id)
{
  if (executor_thread_telemetry_handle.has_value()) {
    SIRIUS_LOG_WARN("ExecutorThreadHandleWrapper was already initialized; overriding.");
  }
  executor_thread_telemetry_handle.emplace(context, thread_name, parent_group_id);
}

}  // namespace sirius::telemetry
