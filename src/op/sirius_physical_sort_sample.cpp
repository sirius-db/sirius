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

#include "cudf/cudf_utils.hpp"
#include "data/data_batch_utils.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "log/logging.hpp"
#include "op/cudf_sort_order.hpp"
#include "op/merge/gpu_merge_impl.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "sirius/exception.hpp"

#include <nvtx3/nvtx3.hpp>

#include <functional>

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

}  // namespace

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
                                                                    rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_sort_sample::execute"};
  auto& input               = dynamic_cast<const pipelineable_operator_data&>(input_data);
  const auto& input_batches = input.get_read_only_batches();

  // Fast path: boundaries already computed — just pass through.
  if (_boundary_state.load(std::memory_order_acquire) == BoundaryState::DONE) {
    SIRIUS_LOG_DEBUG("Sort sample: passthrough ({} batches)", input_batches.size());
    return std::make_unique<pipelineable_operator_data>(input.get_read_only_batches());
  }

  const auto state = _boundary_state.load(std::memory_order_acquire);
  if (state == BoundaryState::SCHEDULED) {
    // Normal path: input was claimed by get_next_task_input_data().
  } else if (state == BoundaryState::NOT_DONE) {
    // OOM retry or direct unit-test invoke — claim boundary computation.
    auto expected = BoundaryState::NOT_DONE;
    if (!_boundary_state.compare_exchange_strong(
          expected, BoundaryState::SCHEDULED, std::memory_order_acq_rel)) {
      SIRIUS_LOG_DEBUG("Sort sample: passthrough ({} batches)", input_batches.size());
      return std::make_unique<pipelineable_operator_data>(input.get_read_only_batches());
    }
  } else {
    SIRIUS_LOG_DEBUG("Sort sample: passthrough ({} batches)", input_batches.size());
    return std::make_unique<pipelineable_operator_data>(input.get_read_only_batches());
  }

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
    return std::make_unique<pipelineable_operator_data>(input.get_read_only_batches());
  }

  // Wrap GPU work in try/catch: if any allocation throws (e.g. rmm::out_of_memory),
  // reset to NOT_DONE so the rescheduled task can retry boundary computation.
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
      if (ord.expression->expression_class != duckdb::ExpressionClass::BOUND_REF) {
        throw not_implemented_exception("Sort sample only supports bound reference expressions");
      }
      auto idx = static_cast<int>(ord.expression->Cast<duckdb::BoundReferenceExpression>().index);
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
      merged_sample_batch = gpu_merge_impl::merge_order_by(
        valid_batches, order_key_idx, column_order, null_precedence, stream, *space);
      merged_sample_view = get_cudf_table_view(merged_sample_batch->to_read_only());
    }

    // 4. Compute number of partitions
    size_t total_rows      = static_cast<size_t>(merged_sample_view.num_rows());
    size_t avg_batch_bytes = valid_batches.empty() ? 0 : total_sample_bytes / valid_batches.size();
    size_t avg_rows_per_batch = valid_batches.empty() ? 0 : total_rows / valid_batches.size();
    size_t num_parts          = 1;
    if (estimated_cardinality == 0 || avg_rows_per_batch == 0) {
      SIRIUS_LOG_WARN(
        "Sort sample: estimated_cardinality={} or avg_rows_per_batch={} is zero, "
        "defaulting to 1 partition",
        estimated_cardinality,
        avg_rows_per_batch);
    } else {
      size_t total_batch_count =
        (estimated_cardinality + avg_rows_per_batch - 1) / avg_rows_per_batch;
      size_t estimated_total_bytes = avg_batch_bytes * total_batch_count;
      size_t available_memory      = space->get_available_memory(stream);
      size_t max_partition_bytes   = _max_partition_bytes_override > 0
                                       ? _max_partition_bytes_override
                                       : static_cast<size_t>(static_cast<double>(available_memory) *
                                                           _max_partition_memory_fraction);

      if (max_partition_bytes > 0 && estimated_total_bytes > max_partition_bytes) {
        num_parts = (estimated_total_bytes + max_partition_bytes - 1) / max_partition_bytes;
      }

      SIRIUS_LOG_DEBUG(
        "Sort sample: estimated_cardinality={}, total_rows={}, avg_rows_per_batch={}, "
        "avg_batch_bytes={}, total_batch_count={}, "
        "estimated_total_bytes={}, available_memory={}, max_partition_bytes={}, num_partitions={}",
        estimated_cardinality,
        total_rows,
        avg_rows_per_batch,
        avg_batch_bytes,
        total_batch_count,
        estimated_total_bytes,
        available_memory,
        max_partition_bytes,
        num_parts);
    }

    // 5. Pick P-1 evenly-spaced boundary rows from the merged sample (sort key columns only)
    if (num_parts <= 1 || total_rows == 0) {
      // Single partition — no boundaries needed
      _num_partitions = 1;
      _partition_boundaries.reset();
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
                                    stream.value()));

      // Extract only the sort key columns from merged sample for the boundaries
      std::vector<cudf::column_view> sort_key_cols;
      for (int idx : order_key_idx) {
        sort_key_cols.push_back(merged_sample_view.column(idx));
      }
      cudf::table_view sort_keys_view(sort_key_cols);

      // Gather boundary rows
      _partition_boundaries = cudf::gather(sort_keys_view,
                                           indices_col->view(),
                                           cudf::out_of_bounds_policy::DONT_CHECK,
                                           stream,
                                           space->get_default_allocator());
      _num_partitions       = num_parts;
    }

    // sort_partition runs in the same gpu_pipeline_task immediately after this
    // execute() returns, so _partition_boundaries is visible before partition runs.
    _boundary_state.store(BoundaryState::DONE, std::memory_order_release);

  } catch (...) {
    // Reset to NOT_DONE so the rescheduled task can retry boundary computation.
    _boundary_state.store(BoundaryState::NOT_DONE, std::memory_order_release);
    throw;
  }

  auto end      = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  SIRIUS_LOG_DEBUG("Sort sample: computed {} partitions with {} boundaries in {:.2f} ms",
                   _num_partitions,
                   _partition_boundaries ? _partition_boundaries->num_rows() : 0,
                   duration.count() / 1000.0);

  return std::make_unique<pipelineable_operator_data>(input.get_read_only_batches());
}

}  // namespace op
}  // namespace sirius
