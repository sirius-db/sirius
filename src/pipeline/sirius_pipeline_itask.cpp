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

#include "pipeline/sirius_pipeline_itask.hpp"

#include "log/logging.hpp"
#include "telemetry/telemetry_context.hpp"

#include <exception>
#include <format>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace sirius::pipeline {
namespace {

[[noreturn]] void invalid_transition(std::string_view event)
{
  throw std::logic_error("invalid Quent Task transition to " + std::string(event));
}

void log_task_telemetry_finalization_failure(uint64_t task_id, const char* error) noexcept
{
  try {
    if (error != nullptr) {
      SIRIUS_LOG_WARN("Failed to finalize telemetry for task {}: {}", task_id, error);
    } else {
      SIRIUS_LOG_WARN("Failed to finalize telemetry for task {}: unknown exception", task_id);
    }
  } catch (...) {
  }
}

template <typename Ref, typename Id, typename Usage>
std::optional<Ref> usage_ref(quent::Uuid resource_id, Usage usage)
{
  if (resource_id == quent::nil_uuid()) { return std::nullopt; }
  return Ref{
    .target = Id(resource_id),
    .data   = std::move(usage),
  };
}

std::optional<quent::refs::TaskQueueUsageRef> task_queue_usage(quent::Uuid resource_id,
                                                               uint64_t entries)
{
  return usage_ref<quent::refs::TaskQueueUsageRef, quent::task_queue::TaskQueueId>(
    resource_id, quent::records::TaskQueueUsage{.entries = entries});
}

std::optional<quent::refs::TaskManagerLoopThreadUsageRef> manager_thread_usage(
  quent::Uuid resource_id)
{
  return usage_ref<quent::refs::TaskManagerLoopThreadUsageRef,
                   quent::task_manager_loop_thread::TaskManagerLoopThreadId>(
    resource_id, quent::records::TaskManagerLoopThreadUsage{});
}

std::optional<quent::refs::ExecutorThreadUsageRef> executor_thread_usage(quent::Uuid resource_id)
{
  return usage_ref<quent::refs::ExecutorThreadUsageRef, quent::executor_thread::ExecutorThreadId>(
    resource_id, quent::records::ExecutorThreadUsage{});
}

std::optional<quent::refs::MemoryTierUsageRef> memory_tier_usage(quent::Uuid resource_id,
                                                                 uint64_t bytes)
{
  return usage_ref<quent::refs::MemoryTierUsageRef, quent::memory_tier::MemoryTierId>(
    resource_id, quent::records::MemoryTierUsage{.bytes = bytes});
}

}  // namespace

sirius_pipeline_itask::sirius_pipeline_itask(
  uint64_t task_id,
  std::unique_ptr<sirius_pipeline_task_local_state> local_state,
  std::shared_ptr<sirius_pipeline_task_global_state> global_state)
  : itask(task_id, std::move(local_state), global_state),
    _telemetry_task_state(
      global_state->get_telemetry_context().context().task_observer()->handle().created(
        quent::task::Created{
          .instance_name = std::format("task-{}", task_id),
          .pipeline_uuid = quent::operator_::OperatorId(
            global_state->get_pipeline() ? global_state->get_pipeline()->pipeline_uuid()
                                         : quent::nil_uuid()),
        }))
{
}

sirius_pipeline_itask::~sirius_pipeline_itask() noexcept
{
  if (_telemetry_finalized) { return; }

  try {
    finalize_telemetry(false);
  } catch (const std::exception& error) {
    log_task_telemetry_finalization_failure(get_task_id(), error.what());
  } catch (...) {
    log_task_telemetry_finalization_failure(get_task_id(), nullptr);
  }
}

quent::Uuid sirius_pipeline_itask::telemetry_uuid() const { return _telemetry_task_state.uuid(); }

void sirius_pipeline_itask::telemetry_queued(telemetry_queued_data data)
{
  auto transition = [&data](auto&& current) {
    return std::move(current).queued(quent::task::Queued{
      .queue = task_queue_usage(data.queue_resource_id, data.queue_capacity_entries),
    });
  };
  if (_telemetry_task_state.transition<quent::task_state::Created, quent::task_state::Routing>(
        transition)) {
    return;
  }
  invalid_transition("queued");
}

void sirius_pipeline_itask::telemetry_routing(telemetry_routing_data data)
{
  if (!_telemetry_task_state.transition<quent::task_state::Queued>([&data](auto&& current) {
        return std::move(current).routing(quent::task::Routing{
          .preferred_device_id = data.preferred_device_id,
          .manager_thread      = manager_thread_usage(data.manager_thread_resource_id),
        });
      })) {
    invalid_transition("routing");
  }
}

void sirius_pipeline_itask::telemetry_reserving(telemetry_reserving_data data)
{
  auto transition = [&data](auto&& current) {
    return std::move(current).reserving(quent::task::Reserving{
      .requested_bytes      = data.requested_bytes,
      .input_basis          = data.input_basis,
      .peak_estimate        = data.peak_estimate,
      .bytes_to_materialize = data.bytes_to_materialize,
      .manager_thread       = manager_thread_usage(data.manager_thread_resource_id),
    });
  };
  if (_telemetry_task_state.transition<quent::task_state::Queued, quent::task_state::Routing>(
        transition)) {
    return;
  }
  invalid_transition("reserving");
}

void sirius_pipeline_itask::telemetry_downgrading(telemetry_downgrading_data data)
{
  if (!_telemetry_task_state.transition<quent::task_state::Reserving>([&data](auto&& current) {
        return std::move(current).downgrading(quent::task::Downgrading{
          .shortfall_bytes = data.shortfall_bytes,
          .partial_bytes   = data.partial_bytes,
          .manager_thread  = manager_thread_usage(data.manager_thread_resource_id),
        });
      })) {
    invalid_transition("downgrading");
  }
}

void sirius_pipeline_itask::telemetry_preparing(telemetry_preparing_data data)
{
  auto transition = [&data](auto&& current) {
    return std::move(current).preparing(quent::task::Preparing{
      .origin_tier     = std::move(data.origin_tier),
      .target_tier     = std::move(data.target_tier),
      .input_bytes     = data.input_bytes,
      .executor_thread = executor_thread_usage(data.executor_thread_resource_id),
      .reservation =
        memory_tier_usage(data.reservation_resource_id, data.reservation_capacity_bytes),
    });
  };
  if (_telemetry_task_state
        .transition<quent::task_state::Reserving, quent::task_state::Downgrading>(transition)) {
    return;
  }
  invalid_transition("preparing");
}

void sirius_pipeline_itask::telemetry_computing(telemetry_computing_data data)
{
  auto transition = [&data](auto&& current) {
    return std::move(current).computing(quent::task::Computing{
      .current_operator_id  = data.current_operator_id,
      .input_bytes          = data.input_bytes,
      .peak_allocated_bytes = data.peak_allocated_bytes,
      .executor_thread      = executor_thread_usage(data.executor_thread_resource_id),
      .reservation =
        memory_tier_usage(data.reservation_resource_id, data.reservation_capacity_bytes),
    });
  };
  if (_telemetry_task_state.transition<quent::task_state::Preparing, quent::task_state::Computing>(
        transition)) {
    return;
  }
  invalid_transition("computing");
}

void sirius_pipeline_itask::finalize_telemetry(bool success)
{
  if (_telemetry_finalized) { return; }

  auto transition = [success](auto&& current) {
    return std::move(current).finalizing(quent::task::Finalizing{.success = success});
  };
  if (!_telemetry_task_state.holds<quent::task_state::Finalizing>() &&
      !_telemetry_task_state.transition<quent::task_state::Created,
                                        quent::task_state::Queued,
                                        quent::task_state::Routing,
                                        quent::task_state::Reserving,
                                        quent::task_state::Downgrading,
                                        quent::task_state::Preparing,
                                        quent::task_state::Computing>(transition)) {
    invalid_transition("finalizing");
  }

  _telemetry_finalized = true;
}

}  // namespace sirius::pipeline
