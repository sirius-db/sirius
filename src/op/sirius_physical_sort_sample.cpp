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
#include "pipeline/sirius_pipeline.hpp"

#include <cudf/concatenate.hpp>

namespace sirius {
namespace op {

sirius_physical_sort_sample::sirius_physical_sort_sample(sirius_physical_order* order_by)
  : sirius_physical_sort_sample(
      order_by->types, copy_orders(order_by->orders), order_by->estimated_cardinality)
{
}

sirius_physical_sort_sample::sirius_physical_sort_sample(
  duckdb::vector<duckdb::LogicalType> types,
  duckdb::vector<duckdb::BoundOrderByNode> orders,
  duckdb::idx_t estimated_cardinality,
  duckdb::idx_t num_sample_batches)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::SORT_SAMPLE, std::move(types), estimated_cardinality),
    orders(std::move(orders)),
    num_sample_batches(num_sample_batches)
{
}

std::optional<task_creation_hint> sirius_physical_sort_sample::get_next_task_hint()
{
  // If boundaries already computed, use default behavior (process batches as they arrive)
  if (_boundaries_computed.load()) { return sirius_physical_operator::get_next_task_hint(); }

  // Need to wait for N batches before computing boundaries
  auto port_ids = get_port_ids();
  if (port_ids.empty()) { return std::nullopt; }

  auto* p = get_port(port_ids[0]);
  if (!p) { return std::nullopt; }

  bool upstream_finished = p->src_pipeline && p->src_pipeline->is_pipeline_finished();
  bool has_enough        = p->repo->size() >= static_cast<size_t>(num_sample_batches);

  if (has_enough || (upstream_finished && p->repo->size() > 0)) {
    return task_creation_hint{TaskCreationHint::READY, this};
  }

  if (p->src_pipeline && !upstream_finished) {
    auto* producer = &(p->src_pipeline->get_operators()[0].get());
    return task_creation_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA, producer};
  }

  return std::nullopt;
}

std::vector<std::shared_ptr<cucascade::data_batch>> sirius_physical_sort_sample::execute(
  const std::vector<std::shared_ptr<cucascade::data_batch>>& input_batches,
  rmm::cuda_stream_view stream)
{
  // After boundaries are computed, just pass through
  if (_boundaries_computed.load()) {
    SIRIUS_LOG_DEBUG("Sort sample: passthrough ({} batches)", input_batches.size());
    return input_batches;
  }

  SIRIUS_LOG_DEBUG("Sort sample: computing partition boundaries from {} batches",
                   input_batches.size());
  auto start = std::chrono::high_resolution_clock::now();

  // 1. Collect valid batches and find memory space
  std::vector<std::shared_ptr<cucascade::data_batch>> valid_batches;
  cucascade::memory::memory_space* space = nullptr;
  for (auto const& batch : input_batches) {
    if (!batch) { continue; }
    if (!space) { space = batch->get_memory_space(); }
    valid_batches.push_back(batch);
  }

  if (valid_batches.empty() || !space) {
    _boundaries_computed.store(true);
    return input_batches;
  }

  // 2. Concatenate all sample batches into one table
  std::vector<cudf::table_view> sample_views;
  size_t total_sample_bytes = 0;
  sample_views.reserve(valid_batches.size());
  for (auto const& batch : valid_batches) {
    auto view = get_cudf_table_view(*batch);
    sample_views.push_back(view);
    total_sample_bytes += batch->get_data()->get_size_in_bytes();
  }

  auto concat_table = cudf::concatenate(sample_views, stream, space->get_default_allocator());

  // 3. Build cudf order vectors from BoundOrderByNode
  std::vector<int> order_key_idx;
  std::vector<cudf::order> column_order;
  std::vector<cudf::null_order> null_precedence;
  order_key_idx.reserve(orders.size());
  column_order.reserve(orders.size());
  null_precedence.reserve(orders.size());

  for (auto const& ord : orders) {
    if (ord.expression->expression_class != duckdb::ExpressionClass::BOUND_REF) {
      throw duckdb::NotImplementedException(
        "Sort sample only supports bound reference expressions");
    }
    auto idx = static_cast<int>(ord.expression->Cast<duckdb::BoundReferenceExpression>().index);
    order_key_idx.push_back(idx);
    column_order.push_back(ord.type == duckdb::OrderType::ASCENDING ? cudf::order::ASCENDING
                                                                    : cudf::order::DESCENDING);
    null_precedence.push_back(ord.null_order == duckdb::OrderByNullType::NULLS_FIRST
                                ? cudf::null_order::BEFORE
                                : cudf::null_order::AFTER);
  }

  // 4. Sort the concatenated sample by sort keys
  std::vector<cudf::column_view> sort_cols;
  for (int idx : order_key_idx) {
    sort_cols.push_back(concat_table->view().column(idx));
  }
  auto sorted_indices = cudf::sorted_order(cudf::table_view(sort_cols),
                                           column_order,
                                           null_precedence,
                                           stream,
                                           space->get_default_allocator());

  auto sorted_table = cudf::gather(concat_table->view(),
                                   sorted_indices->view(),
                                   cudf::out_of_bounds_policy::DONT_CHECK,
                                   stream,
                                   space->get_default_allocator());

  // 5. Compute number of partitions
  size_t total_rows         = static_cast<size_t>(sorted_table->num_rows());
  size_t avg_batch_bytes    = valid_batches.empty() ? 0 : total_sample_bytes / valid_batches.size();
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
                                                         MAX_PARTITION_MEMORY_FRACTION);

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

  // 6. Pick P-1 evenly-spaced boundary rows from the sorted sample (sort key columns only)
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

    // Extract only the sort key columns from sorted table for the boundaries
    std::vector<cudf::column_view> sort_key_cols;
    for (int idx : order_key_idx) {
      sort_key_cols.push_back(sorted_table->view().column(idx));
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

  _boundaries_computed.store(true);

  auto end      = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  SIRIUS_LOG_DEBUG("Sort sample: computed {} partitions with {} boundaries in {:.2f} ms",
                   _num_partitions,
                   _partition_boundaries ? _partition_boundaries->num_rows() : 0,
                   duration.count() / 1000.0);

  return input_batches;
}

}  // namespace op
}  // namespace sirius
