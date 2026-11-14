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

#include "op/sirius_physical_sort_sample.hpp"

#include "data/data_batch_utils.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "log/logging.hpp"
#include "op/cudf_sort_order.hpp"
#include "op/merge/gpu_merge_impl.hpp"
#include "pipeline/sirius_pipeline.hpp"  // IWYU pragma: keep
#include "sirius/exception.hpp"
#include "telemetry/nvtx.hpp"

#include <cudf/contiguous_split.hpp>
#include <cudf/copying.hpp>

#include <rmm/cuda_device.hpp>

#include <algorithm>

namespace sirius {
namespace op {

namespace {

uint64_t get_batch_bytes(const std::shared_ptr<::cucascade::data_batch>& batch)
{
  if (!batch) { return 0; }
  auto ro = batch->to_read_only();
  if (!ro.get_data()) { return 0; }
  return ro.get_data()->get_size_in_bytes();
}

bool repo_has_enough_sample_bytes(::cucascade::shared_data_repository* repo,
                                  uint64_t sort_sample_bytes,
                                  bool upstream_finished)
{
  if (!repo) { return false; }
  // Upstream done with no rows still needs a READY signal so the pipeline can drain.
  if (upstream_finished) { return true; }
  if (repo->total_size() == 0) { return false; }

  uint64_t accumulated = 0;
  for (size_t part = 0; part < repo->num_partitions(); ++part) {
    for (auto batch_id : repo->get_batch_ids(part)) {
      auto batch = repo->get_data_batch_by_id(batch_id, part);
      if (!batch) { continue; }
      accumulated += get_batch_bytes(batch);
      if (accumulated >= sort_sample_bytes) { return true; }
    }
  }
  return false;
}

std::size_t ceil_div(std::size_t numerator, std::size_t denominator)
{
  return denominator == 0 ? 1 : (numerator + denominator - 1) / denominator;
}

}  // namespace

sort_partition_sizing compute_sort_partition_sizing(const sort_partition_sizing_input& input)
{
  sort_partition_sizing result;
  auto const bytes = input.data_bytes;
  auto raise       = [&result](std::size_t partitions, const char* rule) {
    if (partitions > result.num_partitions) {
      result.num_partitions = partitions;
      result.rule           = rule;
    }
  };

  // An explicit partition size decides alone.
  if (input.max_partition_bytes_override > 0) {
    result.max_partition_bytes = input.max_partition_bytes_override;
    raise(ceil_div(bytes, result.max_partition_bytes), "max_sort_partition_bytes");
    return result;
  }

  if (!input.policy.auto_sizing) {
    // Previous rule: a fraction of the GPU memory free at this moment.
    result.max_partition_bytes = static_cast<std::size_t>(
      static_cast<double>(input.gpu_available_bytes) * input.memory_fraction);
    if (result.max_partition_bytes > 0) {
      raise(ceil_div(bytes, result.max_partition_bytes), "fraction_of_free");
    }
    return result;
  }

  // Memory: a fraction of the GPU's limit. Without a known limit, fall back to the free memory.
  auto const capacity =
    input.gpu_capacity_bytes > 0 ? input.gpu_capacity_bytes : input.gpu_available_bytes;
  auto const gpus = std::max<std::size_t>(input.policy.num_gpus, 1);
  result.max_partition_bytes =
    static_cast<std::size_t>(static_cast<double>(capacity) * input.memory_fraction);
  const char* memory_rule = "fraction_of_limit";
  // The sort holds the data and its sorted copy. When they do not fit the GPUs, runs are spilled
  // before the merge starts, and smaller partitions keep more of each merge on the GPU.
  if (capacity > 0 && bytes > (capacity / 2) * gpus) {
    auto const spill_bytes =
      static_cast<std::size_t>(static_cast<double>(capacity) * kSortPartitionSpillFraction);
    if (spill_bytes > 0 && spill_bytes < result.max_partition_bytes) {
      result.max_partition_bytes = spill_bytes;
      memory_rule                = "larger_than_gpu";
    }
  }
  if (result.max_partition_bytes > 0) {
    raise(ceil_div(bytes, result.max_partition_bytes), memory_rule);
  }

  // One-shot merges hold the partition, the merge temporaries and the output on one GPU:
  // 3 x bytes + 8 bytes per row, for kSortPartitionOneShotConcurrentMerges merges at once.
  if (input.policy.one_shot_only && capacity > 0 && input.data_rows > 0 && bytes > 0) {
    auto const row_width      = static_cast<double>(bytes) / static_cast<double>(input.data_rows);
    auto const one_shot_bytes = static_cast<std::size_t>(
      static_cast<double>(capacity) /
      ((3.0 + 8.0 / row_width) * static_cast<double>(kSortPartitionOneShotConcurrentMerges)));
    if (one_shot_bytes > 0 && one_shot_bytes < result.max_partition_bytes) {
      result.max_partition_bytes = one_shot_bytes;
    }
    if (one_shot_bytes > 0) { raise(ceil_div(bytes, one_shot_bytes), "one_shot_memory"); }
  }

  // Rows: a one-shot merge makes one cuDF table of its partition. Windowed merges are not
  // limited.
  auto const auto_route = !input.policy.one_shot_only && !input.policy.windowed_only &&
                          input.policy.auto_one_shot_partition_bytes > 0;
  auto const takes_one_shot = [&](std::size_t partitions) {
    return bytes / partitions <= input.policy.auto_one_shot_partition_bytes;
  };
  if ((input.policy.one_shot_only || (auto_route && takes_one_shot(result.num_partitions))) &&
      input.data_rows > kSortPartitionMaxRows * result.num_partitions) {
    raise(ceil_div(input.data_rows, kSortPartitionMaxRows), "row_limit");
  }

  // Concurrency: on several GPUs, one partition per merge task a GPU runs at once, while
  // partitions and the pieces each input batch is cut into stay large enough for that to pay.
  if (input.policy.windowed_only && gpus > 1 && input.policy.merge_concurrency > 1) {
    auto wanted = std::min(input.policy.merge_concurrency, kSortPartitionMaxConcurrency);
    wanted      = std::min(wanted, bytes / kSortPartitionMinConcurrencyBytes);
    if (input.avg_batch_bytes > 0) {
      wanted = std::min(wanted, input.avg_batch_bytes / kSortPartitionMinPieceBytes);
    }
    raise(wanted, "merge_concurrency");
  }

  // Auto mode: partitions at or below the one-shot threshold are merged whole, as many at once
  // as there are executor threads. When those merges do not fit the GPU together, keep the
  // partitions large enough to be merged in windows instead.
  if (auto_route && result.num_partitions > 1 && takes_one_shot(result.num_partitions)) {
    auto const partition_bytes = bytes / result.num_partitions;
    auto const row_bytes =
      input.data_rows > 0 ? 8 * (input.data_rows / result.num_partitions) : std::size_t{0};
    auto const concurrent =
      std::min(result.num_partitions, std::max<std::size_t>(input.policy.merge_concurrency, 1));
    if (concurrent * (3 * partition_bytes + row_bytes) > capacity) {
      // The largest count whose partitions stay clear of the threshold.
      auto const windowed_bytes =
        static_cast<std::size_t>(static_cast<double>(input.policy.auto_one_shot_partition_bytes) *
                                 kSortPartitionWindowedRouteMargin);
      result.num_partitions = std::max<std::size_t>(bytes / windowed_bytes, 1);
      result.rule           = "windowed_route";
    }
  }
  return result;
}

sirius_physical_sort_sample::~sirius_physical_sort_sample() { clear_partition_boundaries(); }

sirius_physical_sort_sample::sirius_physical_sort_sample(sirius_physical_order* order_by,
                                                         uint64_t sort_sample_bytes,
                                                         uint64_t max_partition_bytes,
                                                         double max_partition_memory_fraction)
  : sirius_physical_sort_sample(order_by->types,
                                copy_orders(order_by->orders),
                                order_by->estimated_cardinality,
                                sort_sample_bytes,
                                max_partition_bytes,
                                max_partition_memory_fraction)
{
}

sirius_physical_sort_sample::sirius_physical_sort_sample(
  duckdb::vector<sirius::logical_type> types,
  duckdb::vector<duckdb::BoundOrderByNode> orders,
  std::size_t estimated_cardinality,
  uint64_t sort_sample_bytes,
  uint64_t max_partition_bytes,
  double max_partition_memory_fraction)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::SORT_SAMPLE, std::move(types), estimated_cardinality),
    orders(std::move(orders)),
    _sort_sample_bytes(sort_sample_bytes),
    _max_partition_bytes_override(max_partition_bytes),
    _max_partition_memory_fraction(max_partition_memory_fraction)
{
}

std::optional<task_creation_hint> sirius_physical_sort_sample::get_next_task_hint()
{
  const auto state = _boundary_state.load(std::memory_order_acquire);

  // Boundaries already computed — process batches one at a time.
  if (state == BoundaryState::DONE) { return sirius_physical_operator::get_next_task_hint(); }

  // Boundary task already scheduled; wait for it to finish before creating more tasks.
  if (state == BoundaryState::SCHEDULED) { return std::nullopt; }

  // NOT_DONE: wait for enough sample bytes before scheduling the boundary task.
  auto port_ids = get_port_ids();
  if (port_ids.empty()) { return std::nullopt; }

  auto* p = get_port(port_ids[0]);
  if (!p || !p->repo) { return std::nullopt; }

  bool upstream_finished = p->src_pipeline && p->src_pipeline->is_pipeline_finished();
  if (repo_has_enough_sample_bytes(p->repo, _sort_sample_bytes, upstream_finished)) {
    return task_creation_hint{TaskCreationHint::READY, this};
  }

  if (p->src_pipeline && !upstream_finished) {
    auto* producer = &(p->src_pipeline->get_operators()[0].get());
    return task_creation_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA, producer};
  }

  return std::nullopt;
}

std::unique_ptr<operator_data> sirius_physical_sort_sample::get_next_task_input_data()
{
  const auto state = _boundary_state.load(std::memory_order_acquire);

  // After boundaries are computed, process one batch at a time (passthrough mode).
  if (state == BoundaryState::DONE) { return sirius_physical_operator::get_next_task_input_data(); }

  // Boundary input already claimed by the in-flight boundary task.
  if (state == BoundaryState::SCHEDULED) { return nullptr; }

  // NOT_DONE: accumulate batches up to the sample byte threshold so execute() can merge
  // pre-sorted runs and derive partition boundaries from a representative sample.
  auto port_ids = get_port_ids();
  if (port_ids.empty()) { return nullptr; }

  auto* p = get_port(port_ids[0]);
  if (!p || !p->repo) { return nullptr; }

  std::vector<std::shared_ptr<::cucascade::data_batch>> input_batch;
  uint64_t accumulated_bytes = 0;
  while (true) {
    auto batch = p->repo->pop_next_data_batch();
    if (!batch) { break; }
    accumulated_bytes += get_batch_bytes(batch);
    input_batch.push_back(std::move(batch));
    if (accumulated_bytes >= _sort_sample_bytes) { break; }
  }

  if (input_batch.empty()) { return nullptr; }

  _boundary_state.store(BoundaryState::SCHEDULED, std::memory_order_release);
  return std::make_unique<pipelineable_operator_data>(std::move(input_batch));
}

std::unique_ptr<operator_data> sirius_physical_sort_sample::execute(const operator_data& input_data,
                                                                    ::cuda::stream_ref stream)
{
  nvtx_scoped_range nvtx_range{"sirius_physical_sort_sample::execute"};
  auto& input               = dynamic_cast<const pipelineable_operator_data&>(input_data);
  const auto& input_batches = input.get_read_only_batches();

  // Fast path: boundaries already computed — just pass through.
  if (_boundary_state.load(std::memory_order_acquire) == BoundaryState::DONE) {
    SIRIUS_LOG_DEBUG("Sort sample: passthrough ({} batches)", input_batches.size());
    return std::make_unique<pipelineable_operator_data>(input.get_data_batches());
  }

  // do boundary computation
  SIRIUS_LOG_DEBUG("Sort sample: computing partition boundaries from {} batches ({} bytes target)",
                   input_batches.size(),
                   _sort_sample_bytes);
  auto start = std::chrono::high_resolution_clock::now();

  // 1. Collect valid batches and find memory space
  std::vector<::cucascade::read_only_data_batch> valid_batches;
  cucascade::memory::memory_space* space = nullptr;
  for (auto const& batch : input_batches) {
    if (!space) { space = batch.get_memory_space(); }
    valid_batches.push_back(batch);
  }

  if (valid_batches.empty() || !space) {
    _boundary_state.store(BoundaryState::DONE, std::memory_order_release);
    return std::make_unique<pipelineable_operator_data>(input.get_data_batches());
  }

  // Wrap GPU work in try/catch: if any allocation throws (e.g. rmm::out_of_memory), leave the state
  // at SCHEDULED and rethrow. The rescheduled task carries the same input and re-enters here to
  // retry, and keeping SCHEDULED prevents task_creator from spawning a duplicate boundary task.
  try {
    size_t total_sample_bytes = 0;
    for (auto const& batch : valid_batches) {
      total_sample_bytes += batch.get_data()->get_size_in_bytes();
    }

    // 2. Build cudf order vectors from BoundOrderByNode
    std::vector<int> order_key_idx;
    std::vector<cudf::order> column_order;
    std::vector<cudf::null_order> null_precedence;
    order_key_idx.reserve(orders.size());
    column_order.reserve(orders.size());
    null_precedence.reserve(orders.size());

    for (auto const& ord : orders) {
      if (ord.expression->GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) {
        throw not_implemented_exception("Sort sample only supports bound reference expressions");
      }
      auto idx = static_cast<int>(ord.expression->Cast<duckdb::BoundReferenceExpression>().Index());
      order_key_idx.push_back(idx);
      column_order.push_back(to_cudf_order(ord.type));
      null_precedence.push_back(to_cudf_null_order(ord.type, ord.null_order));
    }

    // 3. Merge pre-sorted sample batches (ORDER_BY sorts each batch locally).
    cudf::table_view merged_sample_view;
    std::shared_ptr<::cucascade::data_batch> merged_sample_batch;
    if (valid_batches.size() == 1) {
      merged_sample_view = get_cudf_table_view(valid_batches[0]);
    } else {
      merged_sample_batch = gpu_merge_impl::merge_order_by(valid_batches,
                                                           order_key_idx,
                                                           column_order,
                                                           null_precedence,
                                                           stream,
                                                           *space,
                                                           batch_telemetry());
      merged_sample_view  = get_cudf_table_view(merged_sample_batch->to_read_only());
    }

    // 4. Compute number of partitions
    bool complete_input = false;
    auto port_ids       = get_port_ids();
    if (!port_ids.empty()) {
      auto* port = get_port(port_ids[0]);
      // Check completion before repository emptiness: once completion is visible,
      // the upstream pipeline cannot publish another batch.
      complete_input = port && port->src_pipeline && port->src_pipeline->is_pipeline_finished() &&
                       port->repo && port->repo->all_empty();
    }

    size_t total_rows      = static_cast<size_t>(merged_sample_view.num_rows());
    size_t avg_batch_bytes = valid_batches.empty() ? 0 : total_sample_bytes / valid_batches.size();
    size_t avg_rows_per_batch = valid_batches.empty() ? 0 : total_rows / valid_batches.size();
    size_t num_parts          = 1;
    bool const can_size = complete_input || (estimated_cardinality > 0 && avg_rows_per_batch > 0);
    if (!can_size) {
      SIRIUS_LOG_WARN(
        "Sort sample: estimated_cardinality={} or avg_rows_per_batch={} is zero, "
        "defaulting to 1 partition",
        estimated_cardinality,
        avg_rows_per_batch);
    } else {
      size_t total_batch_count = valid_batches.size();
      size_t bytes_for_sizing  = total_sample_bytes;
      if (!complete_input) {
        total_batch_count = (estimated_cardinality + avg_rows_per_batch - 1) / avg_rows_per_batch;
        bytes_for_sizing  = avg_batch_bytes * total_batch_count;
      }
      sort_partition_sizing_input sizing_input;
      sizing_input.data_bytes          = bytes_for_sizing;
      sizing_input.data_rows           = complete_input ? total_rows : estimated_cardinality;
      sizing_input.avg_batch_bytes     = avg_batch_bytes;
      sizing_input.gpu_capacity_bytes  = space->get_max_memory();
      sizing_input.gpu_available_bytes = space->get_available_memory(stream);
      sizing_input.max_partition_bytes_override = _max_partition_bytes_override;
      sizing_input.memory_fraction              = _max_partition_memory_fraction;
      sizing_input.policy                       = _partition_policy;
      auto const sizing                         = compute_sort_partition_sizing(sizing_input);
      // One boundary row is needed per partition after the first.
      num_parts = std::min(sizing.num_partitions, std::max<size_t>(1, total_rows));

      SIRIUS_LOG_INFO(
        "[sort_sample][sizing] partitions={} rule={} max_partition_bytes={} data_bytes={} "
        "data_rows={} avg_batch_bytes={} gpu_limit_bytes={} gpu_free_bytes={} "
        "merge_concurrency={} merge_mode={} auto_sizing={} complete_input={}",
        num_parts,
        sizing.rule,
        sizing.max_partition_bytes,
        sizing_input.data_bytes,
        sizing_input.data_rows,
        sizing_input.avg_batch_bytes,
        sizing_input.gpu_capacity_bytes,
        sizing_input.gpu_available_bytes,
        _partition_policy.merge_concurrency,
        _partition_policy.one_shot_only ? "one_shot"
                                        : (_partition_policy.windowed_only ? "windowed" : "auto"),
        _partition_policy.auto_sizing ? 1 : 0,
        complete_input ? 1 : 0);
    }

    // 5. Pick P-1 evenly-spaced boundary rows from the merged sample (sort key columns only)
    if (num_parts <= 1 || total_rows == 0) {
      // Single partition — no boundaries needed
      _num_partitions = 1;
      clear_partition_boundaries();
    } else {
      // Compute boundary row indices: [total_rows/P, 2*total_rows/P, ..., (P-1)*total_rows/P]
      size_t num_boundaries = num_parts - 1;
      std::vector<int32_t> boundary_indices_host;
      boundary_indices_host.reserve(num_boundaries);
      for (size_t i = 1; i <= num_boundaries; i++) {
        auto idx = static_cast<int32_t>((i * total_rows) / num_parts);
        if (idx >= static_cast<int32_t>(total_rows)) { idx = static_cast<int32_t>(total_rows) - 1; }
        boundary_indices_host.push_back(idx);
      }

      // Create a device column with the boundary indices
      auto indices_col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                                   static_cast<cudf::size_type>(num_boundaries),
                                                   cudf::mask_state::UNALLOCATED,
                                                   stream,
                                                   space->get_default_allocator());
      CUDF_CUDA_TRY(cudaMemcpyAsync(indices_col->mutable_view().data<int32_t>(),
                                    boundary_indices_host.data(),
                                    num_boundaries * sizeof(int32_t),
                                    cudaMemcpyHostToDevice,
                                    stream.get()));

      // Extract only the sort key columns from merged sample for the boundaries
      std::vector<cudf::column_view> sort_key_cols;
      for (int idx : order_key_idx) {
        sort_key_cols.push_back(merged_sample_view.column(idx));
      }
      cudf::table_view sort_keys_view(sort_key_cols);

      // Gather boundary rows
      auto boundaries = cudf::gather(sort_keys_view,
                                     indices_col->view(),
                                     cudf::out_of_bounds_policy::DONT_CHECK,
                                     stream,
                                     space->get_default_allocator());

      // Bring the boundary rows to host memory. SORT_PARTITION tasks run on every GPU and on
      // streams other than this one as soon as DONE is published, so they must not read a device
      // table that this stream may still be writing (cudf::gather of fixed-width columns returns
      // before its kernel has run) or that lives on another GPU. The host copy is complete once
      // this stream is synchronised; each GPU uploads its own copy from it on first use.
      auto packed    = cudf::pack(boundaries->view(), stream, space->get_default_allocator());
      auto host      = std::make_unique<host_boundaries>();
      host->num_rows = num_boundaries;
      if (packed.metadata) { host->metadata = *packed.metadata; }
      if (packed.gpu_data && packed.gpu_data->size() > 0) {
        host->data.resize(packed.gpu_data->size());
        CUDF_CUDA_TRY(cudaMemcpyAsync(host->data.data(),
                                      packed.gpu_data->data(),
                                      host->data.size(),
                                      cudaMemcpyDeviceToHost,
                                      stream.get()));
      }
      CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
      {
        std::lock_guard<std::mutex> lock(_device_boundaries_mutex);
        _device_boundaries.clear();
        _host_boundaries = std::move(host);
      }
      _num_partitions = num_parts;
    }

    // SORT_PARTITION runs in this task right after execute() returns, and in tasks on every
    // other executor thread and GPU as soon as DONE is visible: the host copy of the boundaries
    // is complete at this point.
    _boundary_state.store(BoundaryState::DONE, std::memory_order_release);

  } catch (...) {
    // Leave the state at SCHEDULED so the rescheduled task retries boundary computation without
    // task_creator treating the operator as unscheduled and creating its own duplicate task.
    throw;
  }

  auto end      = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  SIRIUS_LOG_DEBUG("Sort sample: computed {} partitions with {} boundaries in {:.2f} ms",
                   _num_partitions,
                   num_boundary_rows(),
                   duration.count() / 1000.0);

  return std::make_unique<pipelineable_operator_data>(input.get_data_batches());
}

cudf::table_view sirius_physical_sort_sample::get_partition_boundaries(
  cucascade::memory::memory_space& space, ::cuda::stream_ref stream)
{
  const int device_id = space.get_device_id();
  std::lock_guard<std::mutex> lock(_device_boundaries_mutex);
  if (!_host_boundaries) {
    throw internal_exception("Sort sample: partition boundaries requested but not available");
  }
  auto it = _device_boundaries.find(device_id);
  if (it == _device_boundaries.end()) {
    auto copy = std::make_unique<device_boundaries>();
    copy->data =
      rmm::device_buffer(_host_boundaries->data.size(), stream, space.get_default_allocator());
    if (!_host_boundaries->data.empty()) {
      CUDF_CUDA_TRY(cudaMemcpyAsync(copy->data.data(),
                                    _host_boundaries->data.data(),
                                    _host_boundaries->data.size(),
                                    cudaMemcpyHostToDevice,
                                    stream.get()));
    }
    // Other tasks on this GPU read the copy from their own streams: finish the upload first.
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
    // The buffer outlives the task's stream; it is freed on the device's default stream.
    copy->data.set_stream(rmm::cuda_stream_default);
    copy->view = cudf::unpack(_host_boundaries->metadata.data(),
                              static_cast<const uint8_t*>(copy->data.data()));
    it         = _device_boundaries.emplace(device_id, std::move(copy)).first;
  }
  return it->second->view;
}

void sirius_physical_sort_sample::clear_partition_boundaries()
{
  std::lock_guard<std::mutex> lock(_device_boundaries_mutex);
  for (auto& [device_id, copy] : _device_boundaries) {
    // Free each copy with its own GPU current; this may run on a thread bound to another GPU.
    rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{device_id}};
    copy.reset();
  }
  _device_boundaries.clear();
  _host_boundaries.reset();
}

}  // namespace op
}  // namespace sirius
