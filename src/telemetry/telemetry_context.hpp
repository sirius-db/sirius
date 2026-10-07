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

#include "log/logging.hpp"
#include "query_id.hpp"
#include "telemetry-bridge/gen/quent.hpp"
#include "telemetry/memory_context.hpp"

#include <cassert>
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

/// Configures NVTX discovery and creates the Quent context described by `config`.
/// This installs the process-global capture hook, so call it before topology
/// discovery or anything else emits NVTX. Throws `std::invalid_argument` for an
/// unrecognised `exporter` value.
[[nodiscard]] quent::Context make_quent_context(const sirius::telemetry_config& config);

/// Owns the top-level telemetry states for a single SiriusContext.
class telemetry_context {
 public:
  /// Telemetry handles for one GPU: the `gpu-N` group and its per-type
  /// child thread-groups.
  struct gpu_device_telemtry_handles {
    quent::Handle<quent::GpuDevice> device;
    quent::Handle<quent::ThreadGroup> manager_threads;
    quent::Handle<quent::ThreadGroup> executor_threads;
  };

  /// Takes ownership of an already-created Quent context. Building it is the
  /// caller's job: `quent::create_context` is what installs the process-global
  /// NVTX injection hook, and Quent drops every event dispatched before that hook
  /// exists, so the caller must create it before anything else emits NVTX.
  ///
  /// GPU memory spaces declare resource groups and thread buckets under the worker.
  [[nodiscard]] static std::shared_ptr<const telemetry_context> create(
    quent::Context&& context,
    const telemetry_config& config,
    const cucascade::memory::memory_reservation_manager* manager = nullptr);

  ~telemetry_context();

  // Non-copyable, non-movable (owns opaque Rust boxes)
  telemetry_context(const telemetry_context&)            = delete;
  telemetry_context& operator=(const telemetry_context&) = delete;
  telemetry_context(telemetry_context&&)                 = delete;
  telemetry_context& operator=(telemetry_context&&)      = delete;

  [[nodiscard]] const quent::engine::EngineId engine_id() const { return engine_handle_.id(); }
  [[nodiscard]] const quent::worker::WorkerId worker_id() const { return worker_handle_.id(); }
  /// The single, session-scoped query group that every query in this context is reported under.
  [[nodiscard]] const quent::query_group::QueryGroupId query_group_id() const
  {
    return default_query_group_handle_.id();
  }

  /// The `{engine}-{session_label}` query group, declared on first use; empty or
  /// nullopt falls back to the default session-scoped group. Thread-safe.
  [[nodiscard]] const quent::query_group::QueryGroupId query_group_id_for(
    const std::optional<std::string>& session_label) const;
  /// The `gpu-N` device group for `device_id`; falls back to the engine group
  /// (with a warning) when the device was not declared at creation time.
  [[nodiscard]] const gpu_device_telemtry_handles& gpu_device_telemetry_handles(
    int device_id) const;

  /// Group for threads serving more than one GPU.
  [[nodiscard]] quent::thread_group::ThreadGroupId shared_group_id() const
  {
    return shared_thread_group_handle_.id();
  }
  [[nodiscard]] const quent::Context& context() const { return context_; }
  [[nodiscard]] const std::shared_ptr<const memory_context>& get_memory_context() const
  {
    return memory_context_;
  }

 private:
  telemetry_context(quent::Context&& context,
                    const sirius::telemetry_config& config,
                    const cucascade::memory::memory_reservation_manager* manager);

  const gpu_device_telemtry_handles& fallback_gpu_device_telemtry_handles() const
  {
    std::call_once(fallback_flag_, [this] {
      auto device = context_.gpu_device_observer()->handle();
      device.declaration({
        .label     = "gpu-fallback",
        .worker_id = worker_handle_.id(),
        // Sentinel: the fallback stands in for any undeclared ordinal.
        .ordinal = std::numeric_limits<uint32_t>::max(),
      });
      auto manager_threads = context_.thread_group_observer()->handle();
      manager_threads.declaration({.label         = "gpu-fallback-manager-threads",
                                   .worker_id     = worker_handle_.id(),
                                   .gpu_device_id = device.id()});
      auto executor_threads = context_.thread_group_observer()->handle();
      executor_threads.declaration({.label         = "gpu-fallback-executor-threads",
                                    .worker_id     = worker_handle_.id(),
                                    .gpu_device_id = device.id()});
      fallback_gpu_device_handles_.emplace(gpu_device_telemtry_handles{
        .device           = std::move(device),
        .manager_threads  = std::move(manager_threads),
        .executor_threads = std::move(executor_threads),
      });
    });
    assert(fallback_gpu_device_handles_.has_value());
    return *fallback_gpu_device_handles_;
  }

  std::string engine_name_;
  quent::Context context_;

  mutable std::mutex labeled_groups_mutex_;
  mutable std::map<std::string, quent::query_group::QueryGroupId> labeled_group_ids_;

  std::map<int, gpu_device_telemtry_handles> gpu_group_ids_;
  mutable std::once_flag fallback_flag_;
  mutable std::optional<gpu_device_telemtry_handles> fallback_gpu_device_handles_;

  quent::Handle<quent::Engine> engine_handle_;
  quent::Handle<quent::Worker> worker_handle_;
  quent::Handle<quent::QueryGroup> default_query_group_handle_;
  quent::Handle<quent::ThreadGroup> shared_thread_group_handle_;

  std::shared_ptr<const memory_context> memory_context_;
};

// A POD to hold common identifiers for useful telemetry.
struct query_telemetry_info {
  /// Quent's own UUID for this query (`QueryHandle::uuid()`), authoritative within telemetry.
  quent::Uuid telemetry_query_id;
  quent::Uuid worker_id;
  /// The engine-wide numeric query id (the execution window's id).
  sirius::query_id_t query_id;
};

/// Emit plan-level telemetry (operator declarations, port declarations, edges)
/// for the given set of pipelines. Called once during query construction.
void emit_plan_telemetry(const quent::Context& context,
                         const std::vector<std::shared_ptr<pipeline::sirius_pipeline>>& pipelines,
                         quent::Uuid plan_id,
                         query_telemetry_info telemetry_info);

/// Emits `exit` from a destructor or scope guard. Those are noexcept, so a failure is logged
/// rather than propagated. Skips handles whose exit was already emitted.
template <typename Handle>
void exit_from_destructor(Handle& handle, std::string_view what) noexcept
{
  if (handle.exit_emitted()) { return; }
  try {
    handle.exit();
  } catch (std::exception const& e) {
    SIRIUS_LOG_ERROR("telemetry: {} exit failed: {}", what, e.what());
  } catch (...) {
    SIRIUS_LOG_ERROR("telemetry: {} exit failed", what);
  }
}

/// Per pool-thread ExecutorThread handle; emits `exit` when the thread ends.
struct executor_thread_telemetry {
  std::optional<quent::Handle<quent::ExecutorThread>> handle;
  ~executor_thread_telemetry()
  {
    if (handle) { exit_from_destructor(*handle, "executor thread"); }
  }
};
inline thread_local executor_thread_telemetry executor_thread_telemetry_state;

}  // namespace sirius::telemetry
