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
#include "memory/defragmenter_oom_policy.hpp"
#include "pipeline/oom_reschedule_exception.hpp"

#include <nvtx3/nvtx3.hpp>

#include <absl/cleanup/cleanup.h>
#include <cucascade/data/cpu_data_representation.hpp>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/memory/error.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>
#include <data/data_batch_utils.hpp>
#include <data/sirius_converter_registry.hpp>

#include <format>
#include <optional>

namespace sirius {
namespace pipeline {

namespace {

std::optional<cucascade::data_batch_processing_handle> lock_or_prepare_batch(
  const std::shared_ptr<cucascade::data_batch>& batch,
  const cucascade::memory::memory_space* requested_memory_space,
  rmm::cuda_stream_view stream)
{
  using status = cucascade::lock_for_processing_status;
  const auto* target_space =
    requested_memory_space != nullptr ? requested_memory_space : batch->get_memory_space();
  if (target_space == nullptr) { return std::nullopt; }

  // NOTE: only works in single gpu setup
  // wait for processing in case a shared batch is in transit in another thread.
  auto lock_result = batch->wait_to_lock_for_processing(target_space->get_id());

  auto cancel_task_if_needed = []() {
    SIRIUS_LOG_ERROR(
      "gpu_pipeline_task: failed to lock batch for processing and cannot prepare batch for "
      "processing. This likely means the batch is in transit and there is a bug in "
      "the in-transit locking logic. Cancelling task to avoid deadlock.");
  };

  while (!lock_result.success && lock_result.status == status::memory_space_mismatch) {
    try {
      auto& registry = sirius::converter_registry::get();
      switch (requested_memory_space->get_tier()) {
        case cucascade::memory::Tier::GPU: {
          auto prev_state = batch->get_state();
          if (!batch->try_to_lock_for_in_transit()) {
            auto current_state = batch->get_state();
            if (current_state == cucascade::batch_state::in_transit) {
              // If another thread has taken the in_transit lock, wait to acquire the processing
              // lock.
              lock_result = batch->wait_to_lock_for_processing(target_space->get_id());
              continue;
            }
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
      // bobbi (todo): delim join will return this warning for now, but there is no bug here, so we
      // can ignore it. we can do something about this after gtc
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

void log_operator_data(const op::sirius_physical_operator& op,
                       const op::operator_data& data,
                       const sirius_pipeline* pipeline,
                       const char* label,
                       const std::string& extra_info = "")
{
  std::string batch_rows = "";
  size_t total_bytes     = 0;
  for (auto& batch : data.get_data_batches()) {
    auto view = get_cudf_table_view(*batch);
    batch_rows += std::to_string(view.num_rows()) + "  ";
    total_bytes += batch->get_data()->get_size_in_bytes();
  }
  SIRIUS_LOG_TRACE(
    "Pipeline {}: operator {} (id={}) {} {} batches, num rows: {}, "
    "size: {} bytes ({:.2f} MB). {}",
    pipeline->get_pipeline_id(),
    op.get_name(),
    op.get_operator_id(),
    label,
    data.get_data_batches().size(),
    batch_rows,
    total_bytes,
    static_cast<double>(total_bytes) / (1024.0 * 1024.0),
    extra_info);
}

std::unique_ptr<op::operator_data> run_one_operator(
  op::sirius_physical_operator& op,
  const op::operator_data& operator_input_data,
  rmm::cuda_stream_view stream,
  const sirius_pipeline* pipeline,
  size_t op_index,
  size_t num_operators,
  cucascade::memory::reservation_aware_resource_adaptor* allocator)
{
  log_operator_data(op, operator_input_data, pipeline, "executing on");

  auto nvtx_label = std::format(
    "Pipeline {}: {} (id={})", pipeline->get_pipeline_id(), op.get_name(), op.get_operator_id());
  nvtx3::scoped_range nvtx_range{nvtx_label.c_str()};
  auto start                = std::chrono::high_resolution_clock::now();
  auto operator_output_data = op.execute(operator_input_data, stream);
  stream.synchronize();
  auto end      = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  auto peak_bytes        = allocator ? allocator->get_peak_allocated_bytes(stream) : 0;
  std::string extra_info = fmt::format(
    "execution time: {:.2f} ms, "
    "peak allocated: {} bytes ({:.2f} MB)",
    duration.count() / 1000.0,
    peak_bytes,
    static_cast<double>(peak_bytes) / (1024.0 * 1024.0));
  log_operator_data(op, *operator_output_data, pipeline, "produced", extra_info);

  validate_operator_output_types(operator_output_data.get(), op);
  return operator_output_data;
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

  for (size_t i = start_index; i < operators.size(); i++) {
    auto& op = operators[i].get();
    try {
      operator_input_output_data = run_one_operator(
        op, *operator_input_output_data, stream, pipeline, i, operators.size(), _allocator);
    } catch (const rmm::out_of_memory& oom) {
      auto peak_bytes = _allocator ? _allocator->get_peak_allocated_bytes(stream) : 0;
      // Subtract the peak allocated bytes to the input data to get the peak allocated bytes for the
      // operators, clamping to zero to avoid unsigned underflow.
      auto const bytes_to_materialize_input = local_state._bytes_to_materialize_input;
      if (peak_bytes > bytes_to_materialize_input) {
        peak_bytes -= bytes_to_materialize_input;
      } else {
        peak_bytes = 0;
      }
      size_t requested_bytes = 0;
      size_t global_usage    = 0;
      if (auto const* cc_oom =
            dynamic_cast<const cucascade::memory::cucascade_out_of_memory*>(&oom)) {
        requested_bytes = cc_oom->requested_bytes;
        global_usage    = cc_oom->global_usage;
      }
      size_t reservation_bytes =
        _local_state->cast<gpu_pipeline_task_local_state>().get_reservation_bytes();
      SIRIUS_LOG_WARN(
        "Pipeline {}: OOM at operator {} (id={}, index {}/{}), "
        "requested {} bytes ({:.2f} MB), global usage {} bytes ({:.2f} MB), "
        "peak allocated {} bytes ({:.2f} MB), "
        "bytes to materialize input {} bytes ({:.2f} MB), "
        "reservation {} bytes ({:.2f} MB), "
        "rescheduling task {}",
        pipeline->get_pipeline_id(),
        op.get_name(),
        op.get_operator_id(),
        i,
        operators.size(),
        requested_bytes,
        static_cast<double>(requested_bytes) / (1024.0 * 1024.0),
        global_usage,
        static_cast<double>(global_usage) / (1024.0 * 1024.0),
        peak_bytes,
        static_cast<double>(peak_bytes) / (1024.0 * 1024.0),
        local_state._bytes_to_materialize_input,
        static_cast<double>(local_state._bytes_to_materialize_input) / (1024.0 * 1024.0),
        reservation_bytes,
        static_cast<double>(reservation_bytes) / (1024.0 * 1024.0),
        _task_id);

      auto input_basis =
        _local_state->cast<gpu_pipeline_task_local_state>().get_task_consumption_basis();
      auto& global = _global_state->cast<gpu_pipeline_task_global_state>();
      global.get_memory_history().record_on_failure(input_basis, peak_bytes);

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
  auto pipeline       = _global_state->cast<gpu_pipeline_task_global_state>().get_pipeline();
  auto sink_operators = pipeline->get_sink();
  if (sink_operators) {
    auto nvtx_label = std::format("Pipeline {}: {} (id={}) sink",
                                  pipeline->get_pipeline_id(),
                                  sink_operators->get_name(),
                                  sink_operators->get_operator_id());
    nvtx3::scoped_range nvtx_range{nvtx_label.c_str()};
    auto const sink_start = std::chrono::high_resolution_clock::now();
    sink_operators.get()->sink(output_data, stream);
    auto const sink_end = std::chrono::high_resolution_clock::now();
    auto const sink_duration =
      std::chrono::duration_cast<std::chrono::microseconds>(sink_end - sink_start);
    SIRIUS_LOG_TRACE("Pipeline {}: operator {} (id={}) sink execution time: {:.2f} ms",
                     pipeline->get_pipeline_id(),
                     sink_operators->get_name(),
                     sink_operators->get_operator_id(),
                     sink_duration.count() / 1000.0);
  } else {
    throw std::runtime_error("Sink operator not found");
  }
}

void gpu_pipeline_task::execute(rmm::cuda_stream_view stream)
{
  auto& local_state = _local_state->cast<gpu_pipeline_task_local_state>();
  auto pipeline     = _global_state->cast<gpu_pipeline_task_global_state>().get_pipeline();
  auto operators    = pipeline->get_operators();
  auto& first_op    = operators[local_state._start_operator_index].get();

  std::string op_chain;
  auto source_op = pipeline->get_source();
  if (source_op) { op_chain += std::format("{} -> ", source_op->get_name()); }
  for (size_t i = 0; i < operators.size(); i++) {
    op_chain += operators[i].get().get_name();
    if (i + 1 < operators.size()) { op_chain += " -> "; }
  }
  auto sink_op = pipeline->get_sink();
  if (sink_op) { op_chain += std::format(" -> {}", sink_op->get_name()); }
  auto nvtx_label =
    std::format("Pipeline {} Task {} [{}]", pipeline->get_pipeline_id(), _task_id, op_chain);
  nvtx3::scoped_range nvtx_range{nvtx_label.c_str()};

  auto const prepare_start = std::chrono::high_resolution_clock::now();
  auto reservation         = local_state.release_reservation();
  if (!reservation) { throw std::runtime_error("GPU pipeline task requires a memory reservation"); }
  auto reservation_bytes = reservation->size();
  const auto* requested_memory_space =
    reservation != nullptr ? &reservation->get_memory_space() : nullptr;
  auto* allocator = reservation->get_memory_resource_of<cucascade::memory::Tier::GPU>();
  allocator->attach_reservation_to_tracker(
    stream, std::move(reservation), nullptr, std::make_unique<memory::defragmenter_oom_policy>());
  absl::Cleanup source_closer = [allocator, stream]() {
    allocator->reset_stream_reservation(stream);
  };

  std::vector<cucascade::data_batch_processing_handle> processing_handles;
  if (!local_state._input_data) {
    throw std::runtime_error("gpu_pipeline_task::execute: input_data is null");
  }
  processing_handles.reserve(local_state._input_data->get_data_batches().size());

  for (const auto& batch : local_state._input_data->get_data_batches()) {
    auto* resident_space = batch->get_memory_space();
    if (requested_memory_space && resident_space &&
        resident_space->get_id() != requested_memory_space->get_id()) {
      SIRIUS_LOG_TRACE(
        "Pipeline {}: fetching batch {} from tier {} to tier {} for operator {} (id={})",
        pipeline->get_pipeline_id(),
        batch->get_batch_id(),
        static_cast<int>(resident_space->get_tier()),
        static_cast<int>(requested_memory_space->get_tier()),
        first_op.get_name(),
        first_op.get_operator_id());
    }
    std::optional<cucascade::data_batch_processing_handle> handle;
    try {
      handle = lock_or_prepare_batch(batch, requested_memory_space, stream);
    } catch (const rmm::out_of_memory& oom) {
      auto peak_bytes  = allocator->get_peak_allocated_bytes(stream);
      auto input_basis = local_state.get_task_consumption_basis();
      auto& global     = _global_state->cast<gpu_pipeline_task_global_state>();
      global.get_memory_history().record_on_failure(input_basis, peak_bytes);

      SIRIUS_LOG_ERROR("Pipeline {}: OOM at batch {} preparing for processing, state: {}",
                       pipeline->get_pipeline_id(),
                       batch->get_batch_id(),
                       static_cast<int>(batch->get_state()));
      throw oom_reschedule_exception(std::move(local_state._input_data),
                                     0,
                                     "Failed to lock or prepare batch " +
                                       std::to_string(batch->get_batch_id()) + " for processing");
    } catch (const std::exception& e) {
      SIRIUS_LOG_ERROR(
        "Unknown error: {};  in lock or prepare for pipeline {}, batch {}, state: {}",
        e.what(),
        pipeline->get_pipeline_id(),
        batch->get_batch_id(),
        static_cast<int>(batch->get_state()));
      throw;
    }
    if (!handle) {
      // trick to retry the task
      throw oom_reschedule_exception(std::move(local_state._input_data),
                                     0,
                                     "Failed to lock or prepare batch " +
                                       std::to_string(batch->get_batch_id()) + " for processing");
      return;
    }
    processing_handles.emplace_back(std::move(*handle));
  }

  auto const prepare_end = std::chrono::high_resolution_clock::now();
  auto const prepare_duration =
    std::chrono::duration_cast<std::chrono::microseconds>(prepare_end - prepare_start);
  SIRIUS_LOG_TRACE("Pipeline {}: operator {} (id={}) prepare execution time: {:.2f} ms",
                   pipeline->get_pipeline_id(),
                   first_op.get_name(),
                   first_op.get_operator_id(),
                   prepare_duration.count() / 1000.0);

  // At this point, all input batches are locked for processing.
  // They will remain locked until the processing_handles go out of scope.

  // 2. Set reservation_aware_memory_resource_ref as the default cudf allocator
  // 3. Execute cudf operators on the pipeline
  _allocator                                     = allocator;
  auto input_basis                               = local_state.get_task_consumption_basis();
  std::unique_ptr<op::operator_data> output_data = compute_task(stream);

  // Record memory metrics for future reservation estimates
  if (output_data) {
    auto peak_bytes = _allocator ? _allocator->get_peak_allocated_bytes(stream) : 0;
    // Subtract the peak allocated bytes to the input data to get the peak allocated bytes for the
    // operators. Clamp at zero to avoid size_t underflow when estimates exceed the observed peak.
    if (peak_bytes > local_state._bytes_to_materialize_input) {
      peak_bytes -= local_state._bytes_to_materialize_input;
    } else {
      peak_bytes = 0;
    }
    std::size_t output_bytes = 0;
    for (const auto& batch : output_data->get_data_batches()) {
      if (batch && batch->get_data()) { output_bytes += batch->get_data()->get_size_in_bytes(); }
    }
    auto& global = _global_state->cast<gpu_pipeline_task_global_state>();
    global.get_memory_history().record({input_basis, peak_bytes, output_bytes});
    SIRIUS_LOG_TRACE(
      "Pipeline {}: memory history record - input_basis={}, output_bytes={}, reservation_bytes={}, "
      "peak_bytes={}, peak_bytes_to_materialize_input={}",
      pipeline->get_pipeline_id(),
      input_basis,
      output_bytes,
      reservation_bytes,
      peak_bytes,
      local_state._bytes_to_materialize_input);
  }

  if (output_data) { publish_output(*output_data, stream); }

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
  auto input_size =
    _local_state->cast<gpu_pipeline_task_local_state>().get_task_consumption_basis();

  auto bytes_to_materialize_input =
    _local_state->cast<gpu_pipeline_task_local_state>()._bytes_to_materialize_input;
  auto& global  = _global_state->cast<gpu_pipeline_task_global_state>();
  auto estimate = global.get_memory_history().estimate_peak_memory(input_size);
  if (estimate) { return *estimate + bytes_to_materialize_input; }
  // Fallback: no history yet (first batch in pipeline)
  return input_size * 2 + bytes_to_materialize_input;
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
