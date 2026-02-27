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

#include "pipeline/gpu_pipeline_task.hpp"

#include "cudf/cudf_utils.hpp"
#include "log/logging.hpp"
#include "pipeline/oom_reschedule_exception.hpp"

#include <absl/cleanup/cleanup.h>
#include <absl/functional/any_invocable.h>
#include <cucascade/data/cpu_data_representation.hpp>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/data/data_repository_manager.hpp>
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/memory/error.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>
#include <data/data_batch_utils.hpp>
#include <data/sirius_converter_registry.hpp>

#include <iostream>
#include <optional>

namespace sirius {
namespace pipeline {

namespace {

std::optional<cucascade::data_batch_processing_handle> lock_or_prepare_batch(
  const std::shared_ptr<cucascade::data_batch>& batch,
  const cucascade::memory::memory_space* requested_memory_space,
  rmm::cuda_stream_view stream)
{
  const auto* target_space =
    requested_memory_space != nullptr ? requested_memory_space : batch->get_memory_space();
  if (target_space == nullptr) { return std::nullopt; }

  auto lock_result = batch->try_to_lock_for_processing(target_space->get_id());

  auto cancel_task_if_needed = []() {};

  const bool needs_conversion =
    requested_memory_space != nullptr &&
    lock_result.status == cucascade::lock_for_processing_status::memory_space_mismatch;

  if (!lock_result.success && needs_conversion) {
    try {
      auto& registry = sirius::converter_registry::get();
      switch (requested_memory_space->get_tier()) {
        case cucascade::memory::Tier::GPU: {
          auto prev_state = batch->get_state();
          if (!batch->try_to_lock_for_in_transit()) {
            cancel_task_if_needed();
            return std::nullopt;
          }
          try {
            batch->convert_to<cucascade::gpu_table_representation>(
              registry, requested_memory_space, stream);
          } catch (...) {
            batch->try_to_release_in_transit();
            throw;
          }
          batch->try_to_release_in_transit(std::optional<cucascade::batch_state>{prev_state});
          break;
        }
        case cucascade::memory::Tier::HOST: {
          auto prev_state = batch->get_state();
          if (!batch->try_to_lock_for_in_transit()) {
            cancel_task_if_needed();
            return std::nullopt;
          }
          try {
            batch->convert_to<cucascade::host_data_representation>(
              registry, requested_memory_space, stream);
          } catch (...) {
            batch->try_to_release_in_transit();
            throw;
          }
          batch->try_to_release_in_transit(std::optional<cucascade::batch_state>{prev_state});
          break;
        }
        default: cancel_task_if_needed(); return std::nullopt;
      }

      lock_result = batch->try_to_lock_for_processing(requested_memory_space->get_id());
    } catch (...) {
      cancel_task_if_needed();
      throw;
    }
  }

  if (!lock_result.success) {
    cancel_task_if_needed();
    return std::nullopt;
  }

  return std::move(lock_result.handle);
}

void validate_operator_output_types(const op::operator_data* data,
                                    const op::sirius_physical_operator& op)
{
  if (data == nullptr) { return; }
  const auto& expected_types = op.get_types();
  const auto& batches        = data->get_data_batches();
  for (size_t batch_index = 0; batch_index < batches.size(); batch_index++) {
    const auto& batch = batches[batch_index];
    if (!batch) { continue; }
    cudf::table_view tbl = get_cudf_table_view(*batch);
    if (static_cast<size_t>(tbl.num_columns()) != expected_types.size()) {
      SIRIUS_LOG_WARN(
        "gpu_pipeline_task: operator '{}' (id={}) output batch {} column count mismatch: got "
        "{}, expected {}",
        op.get_name(),
        op.get_operator_id(),
        batch_index,
        tbl.num_columns(),
        expected_types.size());
      return;
    }
    for (cudf::size_type c = 0; c < tbl.num_columns(); c++) {
      cudf::data_type expected_cudf = duckdb::GetCudfType(expected_types[c]);
      cudf::data_type actual        = tbl.column(c).type();
      if (actual != expected_cudf) {
        SIRIUS_LOG_WARN(
          "gpu_pipeline_task: operator '{}' (id={}) output batch {} column {} datatype "
          "mismatch: got {}, expected {}",
          op.get_name(),
          op.get_operator_id(),
          batch_index,
          c,
          cudf::type_to_name(actual),
          cudf::type_to_name(expected_cudf));
        return;
      }
    }
  }
}

}  // namespace

gpu_pipeline_task::gpu_pipeline_task(
  uint64_t task_id,
  std::vector<cucascade::shared_data_repository*> data_repos,
  std::unique_ptr<sirius_pipeline_task_local_state> local_state,
  std::shared_ptr<sirius_pipeline_task_global_state> global_state)
  : sirius_pipeline_itask(std::move(local_state), std::move(global_state)),
    _task_id(task_id),
    _data_repos(std::move(data_repos))
{
}

gpu_pipeline_task::~gpu_pipeline_task()
{
  if (_oom_rescheduled) { return; }
  if (_global_state == nullptr ||
      _global_state->cast<gpu_pipeline_task_global_state>().get_pipeline() == nullptr) {
    return;
  }
  _global_state->cast<gpu_pipeline_task_global_state>().get_pipeline()->mark_task_completed();
}

uint64_t gpu_pipeline_task::get_task_id() const { return _task_id; }

const sirius_pipeline* gpu_pipeline_task::get_pipeline() const
{
  return _global_state->cast<gpu_pipeline_task_global_state>().get_pipeline();
}

std::unique_ptr<op::operator_data> gpu_pipeline_task::compute_task(rmm::cuda_stream_view stream)
{
  auto pipeline     = _global_state->cast<gpu_pipeline_task_global_state>().get_pipeline();
  auto& local_state = _local_state->cast<gpu_pipeline_task_local_state>();
  auto operator_input_output_data = std::move(local_state._input_data);
  auto operators                  = pipeline->get_operators();
  auto start_index                = local_state._start_operator_index;

  if (start_index > 0) {
    SIRIUS_LOG_INFO("Pipeline {}: resuming task {} from operator index {} (of {})",
                    pipeline->get_pipeline_id(),
                    _task_id,
                    start_index,
                    operators.size());
  }

  std::string batch_sizes = "";
  for (auto& batch : operator_input_output_data->get_data_batches()) {
    auto view = get_cudf_table_view(*batch);
    batch_sizes += std::to_string(view.num_rows()) + "  ";
  }
  for (size_t i = start_index; i < operators.size(); i++) {
    auto& op = operators[i].get();
    SIRIUS_LOG_TRACE("Pipeline {}: operator {} (id={}) executing on {} batches with num row: {}",
                     pipeline->get_pipeline_id(),
                     op.get_name(),
                     op.get_operator_id(),
                     operator_input_output_data->get_data_batches().size(),
                     batch_sizes);
    try {
      auto start            = std::chrono::high_resolution_clock::now();
      auto temp_output_data = op.execute(*operator_input_output_data, stream);
      stream.synchronize();
      operator_input_output_data = std::move(temp_output_data);
      auto end                   = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
      batch_sizes   = "";
      for (auto& batch : operator_input_output_data->get_data_batches()) {
        auto view = get_cudf_table_view(*batch);
        batch_sizes += std::to_string(view.num_rows()) + "  ";
      }
      SIRIUS_LOG_TRACE(
        "Pipeline {}: operator {} (id={}) produced {} batches with num rows: {}, execution time: "
        "{:.2f} ms",
        pipeline->get_pipeline_id(),
        op.get_name(),
        op.get_operator_id(),
        operator_input_output_data ? operator_input_output_data->get_data_batches().size() : 0u,
        batch_sizes,
        duration.count() / 1000.0);
      validate_operator_output_types(operator_input_output_data.get(), op);
    } catch (const rmm::out_of_memory&) {
      SIRIUS_LOG_WARN("Pipeline {}: OOM at operator {} (id={}, index {}/{}), rescheduling task {}",
                      pipeline->get_pipeline_id(),
                      op.get_name(),
                      op.get_operator_id(),
                      i,
                      operators.size(),
                      _task_id);
      throw oom_reschedule_exception(
        std::move(operator_input_output_data),
        i,
        "OOM at operator " + op.get_name() + " (index " + std::to_string(i) + ")");
    }
  }
  return operator_input_output_data;
}

void gpu_pipeline_task::publish_output(op::operator_data& output_data, rmm::cuda_stream_view stream)
{
  auto sink_operators =
    _global_state->cast<gpu_pipeline_task_global_state>().get_pipeline()->get_sink();
  if (sink_operators) {
    sink_operators.get()->sink(output_data, stream);
  } else {
    throw std::runtime_error("Sink operator not found");
  }
}

void gpu_pipeline_task::execute(rmm::cuda_stream_view stream)
{
  auto& local_state = _local_state->cast<gpu_pipeline_task_local_state>();

  auto reservation = local_state.release_reservation();
  if (!reservation) { throw std::runtime_error("GPU pipeline task requires a memory reservation"); }
  const auto* requested_memory_space =
    reservation != nullptr ? &reservation->get_memory_space() : nullptr;
  auto* allocator = reservation->get_memory_resource_of<cucascade::memory::Tier::GPU>();
  allocator->attach_reservation_to_tracker(stream, std::move(reservation), nullptr, nullptr);
  absl::Cleanup source_closer = [allocator, stream]() {
    allocator->reset_stream_reservation(stream);
  };
  std::vector<cucascade::data_batch_processing_handle> processing_handles;
  if (!local_state._input_data) {
    throw std::runtime_error("gpu_pipeline_task::execute: input_data is null");
  }
  processing_handles.reserve(local_state._input_data->get_data_batches().size());

  for (const auto& batch : local_state._input_data->get_data_batches()) {
    auto handle = lock_or_prepare_batch(batch, requested_memory_space, stream);
    if (!handle) {
      // Failed to lock (or convert) one of the batches. Caller can retry later.
      return;
    }
    processing_handles.emplace_back(std::move(*handle));
  }

  // At this point, all input batches are locked for processing.
  // They will remain locked until the processing_handles go out of scope.

  // 2. Set reservation_aware_memory_resource_ref as the default cudf allocator
  // 3. Execute cudf operators on the pipeline
  auto output_data = compute_task(stream);
  stream.synchronize();
  if (output_data) { publish_output(*output_data, stream); }
  // 4. After each cudf operator, get peak total bytes to collect statistics
  // 5. Push output batches to the data repository

  // Processing handles are automatically released here when they go out of scope
}

std::size_t gpu_pipeline_task::get_input_size() const
{
  auto& local_state      = _local_state->cast<gpu_pipeline_task_local_state>();
  std::size_t input_size = 0;
  if (!local_state._input_data) { return 0; }
  for (const auto& batch : local_state._input_data->get_data_batches()) {
    if (!batch || !batch->get_data()) { continue; }
    input_size += batch->get_data()->get_size_in_bytes();
  }
  return input_size;
}

std::size_t gpu_pipeline_task::get_estimated_reservation_size() const
{
  // WSM TODO: this is a placeholder for the actual reservation size
  return get_input_size() * 1;
}

std::vector<op::sirius_physical_operator*> gpu_pipeline_task::get_output_consumers()
{
  std::vector<op::sirius_physical_operator*> output_consumers;
  if (_global_state == nullptr ||
      _global_state->cast<gpu_pipeline_task_global_state>().get_pipeline() == nullptr) {
    return output_consumers;
  }
  return _global_state->cast<gpu_pipeline_task_global_state>()
    .get_pipeline()
    ->get_output_consumers();
}

std::unique_ptr<gpu_pipeline_task> gpu_pipeline_task::create_rescheduled_task(
  uint64_t task_id, std::unique_ptr<sirius_pipeline_task_local_state> local_state)
{
  return std::make_unique<gpu_pipeline_task>(
    task_id, _data_repos, std::move(local_state), get_shared_global_state());
}

}  // namespace pipeline
}  // namespace sirius
