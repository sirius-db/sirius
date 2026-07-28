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
#include "op/scan/sirius_gpu_scan_operator_data.hpp"
#include "pipeline/oom_reschedule_exception.hpp"
#include "telemetry/batch_telemetry.hpp"
#include "telemetry/telemetry_context.hpp"

#include <nvtx3/nvtx3.hpp>
#include <thrust/system/system_error.h>

#include <absl/cleanup/cleanup.h>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/memory/error.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>
#include <data/data_batch_utils.hpp>

#include <cstdint>
#include <format>
#include <optional>
#include <string>
#include <unordered_set>

namespace sirius {
namespace pipeline {

namespace {

void validate_operator_output_types(const op::operator_data* data,
                                    const op::sirius_physical_operator& op)
{
  if (data == nullptr) { return; }
  auto* pipelineable_data = dynamic_cast<const op::pipelineable_operator_data*>(data);
  if (pipelineable_data == nullptr) { return; }
  const auto& expected_types = op.get_types();
  const auto& physical_types = op.get_physical_types();
  // A diagnostic must be robust against the invariant breach it detects: a malformed
  // (partial) sidecar would otherwise be indexed out of bounds below.
  if (!physical_types.empty() && physical_types.size() != expected_types.size()) {
    SIRIUS_LOG_WARN(
      "gpu_pipeline_task: operator '{}' (id={}) physical sidecar width {} != logical width {}",
      op.get_name(),
      op.get_operator_id(),
      physical_types.size(),
      expected_types.size());
    return;
  }
  const auto& batches = pipelineable_data->get_data_batches();
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
      cudf::data_type expected_cudf = physical_types.empty()
                                        ? sirius::get_cudf_type(expected_types[c])
                                        : physical_types[static_cast<std::size_t>(c)];
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

// Authoritative source for the GPU id used by per-task log lines: the
// executor's _per_thread_init runs cudaSetDevice(executor_gpu) on every
// worker thread, and compute_task wraps the per-task work in
// rmm::cuda_set_device_raii on the same id, so cudaGetDevice here reflects
// the executor that is running this task.
int current_gpu_id()
{
  int dev = -1;
  (void)::cudaGetDevice(&dev);
  return dev;
}

void log_operator_data(const op::sirius_physical_operator& op,
                       const op::operator_data& data,
                       const sirius_pipeline* pipeline,
                       uint64_t task_id,
                       const char* label,
                       const std::string& extra_info = "")
{
  std::string batch_rows = "";
  size_t total_bytes     = 0;
  size_t num_batches     = 0;

  if (auto* p_data = dynamic_cast<const op::pipelineable_operator_data*>(&data)) {
    const auto& batches = p_data->get_read_only_batches();
    num_batches         = batches.size();
    for (auto const& batch : batches) {
      if (batch.get_data()) {
        auto view = get_cudf_table_view(batch);
        batch_rows += std::to_string(view.num_rows()) + "  ";
        total_bytes += batch.get_data()->get_size_in_bytes();
      }
    }
  } else {
    SIRIUS_LOG_TRACE(
      "[GPU:{}] Pipeline {}: operator {} (id={}) task={} {} non-pipelineable data. {}",
      current_gpu_id(),
      pipeline->get_pipeline_id(),
      op.get_name(),
      op.get_operator_id(),
      task_id,
      label,
      extra_info);
    return;
  }

  SIRIUS_LOG_TRACE(
    "[GPU:{}] Pipeline {}: operator {} (id={}) task={} {} {} batches, num rows: {}, "
    "size: {} bytes ({:.2f} MB). {}",
    current_gpu_id(),
    pipeline->get_pipeline_id(),
    op.get_name(),
    op.get_operator_id(),
    task_id,
    label,
    num_batches,
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
  uint64_t task_id,
  size_t num_operators,
  cucascade::memory::reservation_aware_resource_adaptor* allocator)
{
  log_operator_data(op, operator_input_data, pipeline, task_id, "executing on");

  auto nvtx_label = std::format(
    "Pipeline {}: {} (id={})", pipeline->get_pipeline_id(), op.get_name(), op.get_operator_id());
  nvtx3::scoped_range nvtx_range{nvtx_label.c_str()};
  auto start = std::chrono::high_resolution_clock::now();
  std::unique_ptr<op::operator_data> operator_output_data;
  try {
    operator_output_data = op.execute(operator_input_data, stream);
  } catch (const std::exception& ex) {
    auto sticky_err = cudaGetLastError();
    if (sticky_err != cudaSuccess) {
      SIRIUS_LOG_WARN("Pipeline {}: {} (id={}) threw + left sticky CUDA error: [{}] {} — clearing",
                      pipeline->get_pipeline_id(),
                      op.get_name(),
                      op.get_operator_id(),
                      static_cast<int>(sticky_err),
                      cudaGetErrorString(sticky_err));
    }
    SIRIUS_LOG_WARN("Pipeline {}: {} (id={}) threw during execute: {}",
                    pipeline->get_pipeline_id(),
                    op.get_name(),
                    op.get_operator_id(),
                    ex.what());
    throw;
  }

  if (auto sticky_err = cudaGetLastError(); sticky_err != cudaSuccess) {
    SIRIUS_LOG_WARN(
      "Pipeline {}: {} (id={}) left a sticky CUDA error after execute: [{}] {} — clearing",
      pipeline->get_pipeline_id(),
      op.get_name(),
      op.get_operator_id(),
      static_cast<int>(sticky_err),
      cudaGetErrorString(sticky_err));
  }

  stream.synchronize();
  auto end      = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  auto peak_bytes        = allocator ? allocator->get_peak_allocated_bytes(stream) : 0;
  std::string extra_info = std::format(
    "execution time: {:.2f} ms, "
    "peak allocated: {} bytes ({:.2f} MB)",
    duration.count() / 1000.0,
    peak_bytes,
    static_cast<double>(peak_bytes) / (1024.0 * 1024.0));
  log_operator_data(op, *operator_output_data, pipeline, task_id, "produced", extra_info);

  validate_operator_output_types(operator_output_data.get(), op);
  return operator_output_data;
}

}  // namespace

std::size_t gpu_pipeline_task_local_state::get_estimated_bytes_to_materialize_input(
  const cucascade::memory::memory_space* target_space) const
{
  if (auto* scan_input = dynamic_cast<const op::scan::scan_operator_input*>(_input_data.get());
      scan_input && scan_input->is_resident()) {
    // Cached scan inputs can still reside in HOST and require an upload before execution.
    auto batch = scan_input->get_cached_batch();
    if (!batch) { return 0; }

    auto ro          = batch->to_read_only();
    auto const* data = ro.get_data();
    if (!data || ro.get_current_tier() == cucascade::memory::Tier::GPU) { return 0; }
    return data->get_uncompressed_data_size_in_bytes();
  }

  std::size_t input_size   = 0;
  auto* pipelineable_input = dynamic_cast<const op::pipelineable_operator_data*>(_input_data.get());
  if (pipelineable_input) {
    for (const auto& ro : pipelineable_input->get_read_only_batches(false)) {
      if (!ro.get_data()) { continue; }
      const bool non_gpu     = ro.get_current_tier() != cucascade::memory::Tier::GPU;
      const bool cross_space = target_space != nullptr && ro.get_memory_space() != nullptr &&
                               ro.get_memory_space()->get_id() != target_space->get_id();
      if (non_gpu || cross_space) {
        input_size += ro.get_data()->get_uncompressed_data_size_in_bytes();
      }
    }
  }
  return input_size;
}

gpu_pipeline_task::gpu_pipeline_task(
  uint64_t task_id,
  std::vector<cucascade::shared_data_repository*> data_repos,
  std::unique_ptr<sirius_pipeline_task_local_state> local_state,
  std::shared_ptr<sirius_pipeline_task_global_state> global_state)
  : sirius_pipeline_itask(task_id, std::move(local_state), std::move(global_state)),
    _data_repos(std::move(data_repos))
{
  // Subscribe to all input data_batches
  auto& ls = _local_state->cast<gpu_pipeline_task_local_state>();
  if (ls._input_data) {
    auto* pipelineable_input =
      dynamic_cast<const op::pipelineable_operator_data*>(ls._input_data.get());
    if (pipelineable_input) {
      for (const auto& batch : pipelineable_input->get_data_batches()) {
        if (batch) {
          batch->subscribe();
          _subscribed_batches.push_back(batch);
        }
      }
    }
  }
  if (auto* pipeline = _global_state->cast<gpu_pipeline_task_global_state>().get_pipeline()) {
    pipeline->mark_task_created();
    auto& registry = telemetry::batch_telemetry_registry::instance();
    for (const auto& weak_batch : _subscribed_batches) {
      if (auto batch = weak_batch.lock()) {
        registry.on_packaged(batch, pipeline->pipeline_uuid(), telemetry_handle().uuid());
        _claimed_batch_ids.push_back(batch->get_batch_id());
      }
    }
  }
}

gpu_pipeline_task::~gpu_pipeline_task()
{
  {
    auto& registry       = telemetry::batch_telemetry_registry::instance();
    const auto task_uuid = telemetry_handle().uuid();
    for (const auto batch_id : _claimed_batch_ids) {
      registry.on_consumed(batch_id, task_uuid);
    }
  }

  for (const auto& weak_batch : _subscribed_batches) {
    auto batch = weak_batch.lock();
    if (!batch) { continue; }
    try {
      batch->unsubscribe();
    } catch (...) {
      // The destructor must not throw; log if possible.
      SIRIUS_LOG_WARN("gpu_pipeline_task: unsubscribe failed for batch {}", batch->get_batch_id());
    }
  }
  _subscribed_batches.clear();

  if (_global_state == nullptr ||
      _global_state->cast<gpu_pipeline_task_global_state>().get_pipeline() == nullptr) {
    return;
  }
  _global_state->cast<gpu_pipeline_task_global_state>().get_pipeline()->mark_task_completed();
}

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
                    get_task_id(),
                    start_index,
                    operators.size());
  }

  auto executor_thread_resource_id = uuid::new_nil();
  if (telemetry::executor_thread_telemetry_handle.has_value()) {
    executor_thread_resource_id = telemetry::executor_thread_telemetry_handle->handle->uuid();
  } else {
    SIRIUS_LOG_ERROR(
      "gpu_pipeline_task::execute_operator: executor thread telemetry handle is not "
      "initialized");
  }

  for (size_t i = start_index; i < operators.size(); i++) {
    auto& op = operators[i].get();
    try {
      this->telemetry_handle().computing({
        .instance_name       = std::format("{}({})", op.get_name(), op.get_operator_id()),
        .current_operator_id = static_cast<uint32_t>(
          op.get_operator_id()),  // TODO(dhruv9vats): look into possible overflow
        .input_bytes          = operator_input_output_data->get_estimated_size_in_bytes(),
        .peak_allocated_bytes = _allocator ? _allocator->get_peak_allocated_bytes(stream) : 0,
        .executor_thread_resource_id = executor_thread_resource_id,
        .reservation_resource_id     = _reservation_tier_resource_id,
        .reservation_capacity_bytes  = _reservation_bytes,
      });
      operator_input_output_data = run_one_operator(
        op, *operator_input_output_data, stream, pipeline, _task_id, operators.size(), _allocator);
    } catch (const rmm::out_of_memory& oom) {
      auto peak_bytes = _allocator ? _allocator->get_peak_allocated_bytes(stream) : 0;
      // Subtract the peak allocated bytes to the input data to get the peak allocated bytes for the
      // operators, clamping to zero to avoid unsigned underflow.
      auto const bytes_to_materialize_input =
        local_state.get_reservation_size_info()->bytes_to_materialize_input;
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
        local_state.get_reservation_size_info()->bytes_to_materialize_input,
        static_cast<double>(local_state.get_reservation_size_info()->bytes_to_materialize_input) /
          (1024.0 * 1024.0),
        reservation_bytes,
        static_cast<double>(reservation_bytes) / (1024.0 * 1024.0),
        get_task_id());

      auto input_basis = _local_state->cast<gpu_pipeline_task_local_state>()
                           .get_reservation_size_info()
                           ->input_basis;
      auto& global = _global_state->cast<gpu_pipeline_task_global_state>();
      global.get_memory_history().record_on_failure(input_basis, peak_bytes);

      throw oom_reschedule_exception(
        std::move(operator_input_output_data),
        i,
        "OOM at operator " + op.get_name() + " (index " + std::to_string(i) + ")");
    } catch (const thrust::system_error& cuda_err) {
      auto err = static_cast<cudaError_t>(cuda_err.code().value());
      if (err == cudaErrorLaunchOutOfResources || err == cudaErrorInvalidValue) {
        SIRIUS_LOG_WARN(
          "Pipeline {}: CUDA launch error [{}] {} at operator {} (id={}, index {}/{}), "
          "rescheduling task {}",
          pipeline->get_pipeline_id(),
          static_cast<int>(err),
          cudaGetErrorString(err),
          op.get_name(),
          op.get_operator_id(),
          i,
          operators.size(),
          _task_id);
        throw cuda_launch_reschedule_exception(
          std::move(operator_input_output_data),
          i,
          static_cast<int>(err),
          std::format("CUDA launch error [{}] {} at operator {} (index {})",
                      static_cast<int>(err),
                      cudaGetErrorString(err),
                      op.get_name(),
                      i));
      }
      throw;
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
    std::format("Pipeline {} Task {} [{}]", pipeline->get_pipeline_id(), get_task_id(), op_chain);
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

  if (!local_state._input_data) {
    throw std::runtime_error("gpu_pipeline_task::execute: input_data is null");
  }

  auto executor_thread_resource_id = uuid::new_nil();
  if (telemetry::executor_thread_telemetry_handle.has_value()) {
    executor_thread_resource_id = telemetry::executor_thread_telemetry_handle->handle->uuid();
  } else {
    SIRIUS_LOG_ERROR(
      "gpu_pipeline_task::execute: executor thread telemetry handle is not initialized");
  }
  _reservation_bytes = reservation_bytes;
  if (requested_memory_space != nullptr) {
    _reservation_tier_resource_id = telemetry::batch_telemetry_registry::instance().tier_resource(
      requested_memory_space->get_tier(), requested_memory_space->get_id().device_id);
  }
  telemetry_handle().preparing({
    .instance_name               = "",
    .origin_tier                 = local_state._input_data->get_origin_tiers(),
    .target_tier                 = "GPU",
    .input_bytes                 = local_state._input_data->get_estimated_size_in_bytes(),
    .executor_thread_resource_id = executor_thread_resource_id,
    .reservation_resource_id     = _reservation_tier_resource_id,
    .reservation_capacity_bytes  = _reservation_bytes,
  });
  try {
    local_state._input_data->prepare_for_processing(requested_memory_space, stream);
    // synchronizing here to ensure the timing collected by Quent and logging for preparing the task
    // is accurate.
    stream.synchronize();
  } catch (const rmm::out_of_memory& oom) {
    auto peak_bytes           = allocator->get_peak_allocated_bytes(stream);
    const auto& res_info      = local_state.get_reservation_size_info();
    auto bytes_to_materialize = res_info->bytes_to_materialize_input;
    auto input_basis          = res_info->input_basis;
    // Keep the recorded peak clean of materialization overhead (host/disk upgrades and
    // cross-GPU clones — prepare's allocations are almost entirely those), consistent with
    // the success and compute-OOM record paths; the estimator re-adds
    // bytes_to_materialize_input on top of the history-based estimate.
    peak_bytes   = peak_bytes > bytes_to_materialize ? peak_bytes - bytes_to_materialize : 0;
    auto& global = _global_state->cast<gpu_pipeline_task_global_state>();
    global.get_memory_history().record_on_failure(input_basis, peak_bytes);

    SIRIUS_LOG_ERROR("Pipeline {}: OOM preparing batches for processing",
                     pipeline->get_pipeline_id());
    throw oom_reschedule_exception(
      std::move(local_state._input_data),
      0,
      std::string("OOM while preparing batches for processing: ") + oom.what());
  } catch (const std::exception& e) {
    SIRIUS_LOG_ERROR("Unknown error in prepare_for_processing for pipeline {}: {}",
                     pipeline->get_pipeline_id(),
                     e.what());
    throw;
  }

  auto const prepare_end = std::chrono::high_resolution_clock::now();
  auto const prepare_duration =
    std::chrono::duration_cast<std::chrono::microseconds>(prepare_end - prepare_start);
  SIRIUS_LOG_TRACE("Pipeline {}: operator {} (id={}) prepare execution time: {:.2f} ms",
                   pipeline->get_pipeline_id(),
                   first_op.get_name(),
                   first_op.get_operator_id(),
                   prepare_duration.count() / 1000.0);

  // All input batches are now locked for reading via _read_only_data_batches inside
  // local_state._input_data. The locks are released when the pipelineable_operator_data
  // is destroyed after the first operator's execute() consumes it.
  {
    auto& registry       = telemetry::batch_telemetry_registry::instance();
    const auto task_uuid = telemetry_handle().uuid();
    std::unordered_set<uint64_t> live_ids;
    for (const auto& weak_batch : _subscribed_batches) {
      if (auto batch = weak_batch.lock()) {
        live_ids.insert(batch->get_batch_id());
        registry.on_processing(batch, task_uuid);
      }
    }
    for (const auto batch_id : _claimed_batch_ids) {
      if (!live_ids.contains(batch_id)) { registry.on_processing_by_id(batch_id, task_uuid); }
    }
  }

  // 2. Set reservation_aware_memory_resource_ref as the default cudf allocator
  // 3. Execute cudf operators on the pipeline
  _allocator       = allocator;
  auto input_basis = local_state.get_reservation_size_info()->input_basis;
  std::unique_ptr<op::operator_data> output_data = compute_task(stream);

  // Record memory metrics for future reservation estimates
  if (output_data) {
    auto peak_bytes = _allocator ? _allocator->get_peak_allocated_bytes(stream) : 0;
    // Subtract the peak allocated bytes to the input data to get the peak allocated bytes for the
    // operators. Clamp at zero to avoid size_t underflow when estimates exceed the observed peak.
    if (peak_bytes > local_state.get_reservation_size_info()->bytes_to_materialize_input) {
      peak_bytes -= local_state.get_reservation_size_info()->bytes_to_materialize_input;
    } else {
      peak_bytes = 0;
    }
    std::size_t output_bytes = 0;
    auto* pipelineable_output =
      dynamic_cast<const op::pipelineable_operator_data*>(output_data.get());
    if (pipelineable_output) {
      for (const auto& batch : pipelineable_output->get_read_only_batches(false)) {
        output_bytes += batch.get_data()->get_size_in_bytes();
      }
    }
    auto& global = _global_state->cast<gpu_pipeline_task_global_state>();
    global.get_memory_history().record({input_basis, peak_bytes, output_bytes});
    SIRIUS_LOG_TRACE(
      "[GPU:{}] Pipeline {}: memory history record - task={}, input_basis={}, output_bytes={}, "
      "reservation_bytes={}, peak_bytes={}, peak_bytes_to_materialize_input={}",
      current_gpu_id(),
      pipeline->get_pipeline_id(),
      _task_id,
      input_basis,
      output_bytes,
      reservation_bytes,
      peak_bytes,
      local_state.get_reservation_size_info()->bytes_to_materialize_input);
  }

  if (output_data) { publish_output(*output_data, stream); }

  // The input pipelineable_operator_data (with its _read_only_data_batches) was destroyed
  // when compute_task replaced operator_input_output_data, releasing all shared locks.
}

std::size_t gpu_pipeline_task::get_input_size() const
{
  auto& local_state      = _local_state->cast<gpu_pipeline_task_local_state>();
  std::size_t input_size = 0;
  if (!local_state._input_data) { return 0; }
  auto* pipelineable_input =
    dynamic_cast<const op::pipelineable_operator_data*>(local_state._input_data.get());
  if (!pipelineable_input) { return 0; }
  for (const auto& batch : pipelineable_input->get_read_only_batches(false)) {
    input_size += batch.get_data()->get_size_in_bytes();
  }
  return input_size;
}

pipeline::reservation_size_info gpu_pipeline_task::get_estimated_reservation_size_info(
  const cucascade::memory::memory_space* target_space) const
{
  auto& ls                         = _local_state->cast<gpu_pipeline_task_local_state>();
  auto& gs                         = _global_state->cast<gpu_pipeline_task_global_state>();
  std::size_t input_basis          = ls.get_task_consumption_basis();
  std::size_t bytes_to_materialize = ls.get_estimated_bytes_to_materialize_input(target_space);
  auto peak_opt                    = gs.get_memory_history().estimate_peak_memory(input_basis);
  const auto input_type =
    ls._input_data ? ls._input_data->get_type() : op::operator_data_type::BASE;
  const bool input_resident = ls._input_data && ls._input_data->is_resident();
  auto const* scan_input =
    ls._input_data ? dynamic_cast<const op::scan::scan_operator_input*>(ls._input_data.get())
                   : nullptr;
  const bool input_needs_carrier_conversion =
    scan_input != nullptr && scan_input->needs_carrier_conversion;
  const std::size_t input_conversion_destination_bytes =
    scan_input != nullptr ? scan_input->conversion_destination_bytes : 0;
  auto working_set_bytes = input_basis;
  // Resident (cached) scan inputs report mask/filter copy peaks through their
  // working-set estimate too — it seeds the cold-start guess below via
  // input_stats, so do not gate this on residency.
  if (input_type == op::operator_data_type::GPU_SCAN && ls._input_data) {
    working_set_bytes = ls._input_data->get_estimated_working_set_size_in_bytes();
  }

  pipeline::reservation_size_info info;
  info.input_basis                = input_basis;
  info.bytes_to_materialize_input = bytes_to_materialize;
  info.had_history                = peak_opt.has_value();

  if (peak_opt.has_value()) {
    info.peak_memory_estimate = *peak_opt;
    // Non-resident scans keep the per-split decode floor even with history;
    // resident (cached) scans trust the learned peak — their mask/filter
    // model only seeds the cold start, and a warm undershoot is repaired by
    // record_on_failure + retry.
    if (input_type == op::operator_data_type::GPU_SCAN && !input_resident) {
      info.peak_memory_estimate = std::max(info.peak_memory_estimate, working_set_bytes);
    }
  } else {
    std::size_t num_batches = 0;
    if (auto* pd = dynamic_cast<const op::pipelineable_operator_data*>(ls._input_data.get())) {
      num_batches = pd->get_data_batches().size();
    }
    const op::input_stats stats{.num_batches                  = num_batches,
                                .bytes                        = input_basis,
                                .type                         = input_type,
                                .resident                     = input_resident,
                                .working_set_bytes            = working_set_bytes,
                                .needs_carrier_conversion     = input_needs_carrier_conversion,
                                .conversion_destination_bytes = input_conversion_destination_bytes};

    std::size_t max_estimate = 0;
    if (auto* pipeline = gs.get_pipeline()) {
      for (auto& op_ref : pipeline->get_operators()) {
        max_estimate = std::max(max_estimate, op_ref.get().no_history_peak_memory_estimate(stats));
      }
    }
    // If every operator returned 0 (all pass-throughs), fall back to the 2× default.
    info.peak_memory_estimate = (max_estimate > 0) ? max_estimate : (input_basis * 2);
  }

  info.reservation_size = info.peak_memory_estimate + bytes_to_materialize;
  return info;
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
