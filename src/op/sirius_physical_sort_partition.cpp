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

#include "op/sirius_physical_sort_partition.hpp"

#include "cudf/cudf_utils.hpp"
#include "data/data_batch_utils.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "log/logging.hpp"
#include "op/cudf_sort_order.hpp"
#include "op/sirius_physical_sort_sample.hpp"
#include "sirius/exception.hpp"

#include <cudf/search.hpp>

#include <nvtx3/nvtx3.hpp>

namespace sirius {
namespace op {

sirius_physical_sort_partition::sirius_physical_sort_partition(sirius_physical_order* order_by)
  : sirius_physical_sort_partition(order_by->types,
                                   copy_orders(order_by->orders),
                                   order_by->projections,
                                   order_by->estimated_cardinality)
{
}

sirius_physical_sort_partition::sirius_physical_sort_partition(
  duckdb::vector<sirius::logical_type> types,
  duckdb::vector<duckdb::BoundOrderByNode> orders,
  duckdb::vector<std::size_t> projections_p,
  std::size_t estimated_cardinality)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::SORT_PARTITION, std::move(types), estimated_cardinality),
    orders(std::move(orders)),
    projections(std::move(projections_p))
{
}

std::unique_ptr<operator_data> sirius_physical_sort_partition::execute(
  const operator_data& input_data, rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_sort_partition::execute"};
  auto& input               = dynamic_cast<const pipelineable_operator_data&>(input_data);
  const auto& input_batches = input.get_read_only_batches();

  // If no sample operator or only 1 partition, pass through
  if (!_sample_op || !_sample_op->boundaries_computed() || _sample_op->get_num_partitions() <= 1) {
    SIRIUS_LOG_DEBUG("Sort partition: passthrough ({} batches, {} partitions)",
                     input_batches.size(),
                     _sample_op ? _sample_op->get_num_partitions() : 1);
    return std::make_unique<pipelineable_operator_data>(input.get_read_only_batches());
  }

  auto start           = std::chrono::high_resolution_clock::now();
  size_t num_parts     = _sample_op->get_num_partitions();
  auto& boundaries     = _sample_op->get_partition_boundaries();
  auto boundaries_view = boundaries.view();

  // Build cudf order vectors from BoundOrderByNode
  std::vector<int> order_key_idx;
  std::vector<cudf::order> column_order;
  std::vector<cudf::null_order> null_precedence;
  order_key_idx.reserve(orders.size());
  column_order.reserve(orders.size());
  null_precedence.reserve(orders.size());

  for (auto const& ord : orders) {
    if (ord.expression->GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) {
      throw not_implemented_exception("Sort partition only supports bound reference expressions");
    }
    auto idx = static_cast<int>(ord.expression->Cast<duckdb::BoundReferenceExpression>().Index());
    order_key_idx.push_back(idx);
    column_order.push_back(to_cudf_order(ord.type));
    null_precedence.push_back(to_cudf_null_order(ord.type, ord.null_order));
  }

  std::vector<std::shared_ptr<cucascade::data_batch>> output_batches;

  for (auto const& batch : input_batches) {
    auto* space = batch.get_memory_space();
    if (!space) { continue; }
    auto input_table = get_cudf_table_view(batch);
    auto num_rows    = input_table.num_rows();

    if (num_rows == 0) { continue; }

    // Extract sort key columns from the input batch
    std::vector<cudf::column_view> sort_key_cols;
    for (int idx : order_key_idx) {
      sort_key_cols.push_back(input_table.column(idx));
    }
    cudf::table_view batch_sort_keys(sort_key_cols);

    // Use lower_bound: haystack=batch (sorted), needles=boundaries
    // Returns P-1 split positions in the batch
    auto split_positions_col = cudf::lower_bound(batch_sort_keys,
                                                 boundaries_view,
                                                 column_order,
                                                 null_precedence,
                                                 stream,
                                                 space->get_default_allocator());

    // Copy split positions to host (only P-1 values)
    size_t num_splits = static_cast<size_t>(split_positions_col->size());
    std::vector<int32_t> splits_host(num_splits);
    CUDF_CUDA_TRY(cudaMemcpyAsync(splits_host.data(),
                                  split_positions_col->view().data<int32_t>(),
                                  num_splits * sizeof(int32_t),
                                  cudaMemcpyDeviceToHost,
                                  stream.value()));
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream.value()));

    // Build slice indices: [0, split[0], split[0], split[1], ..., split[P-2], num_rows]
    std::vector<cudf::size_type> slice_indices;
    slice_indices.reserve(num_parts * 2);
    cudf::size_type prev = 0;
    for (size_t i = 0; i < num_splits; i++) {
      slice_indices.push_back(prev);
      slice_indices.push_back(splits_host[i]);
      prev = splits_host[i];
    }
    // Last partition: from last split to end
    slice_indices.push_back(prev);
    slice_indices.push_back(num_rows);

    // Slice into partitions
    auto partition_views = cudf::slice(input_table, slice_indices, stream);

    // Create data_batches for non-empty partitions
    for (size_t i = 0; i < partition_views.size(); i++) {
      if (partition_views[i].num_rows() == 0) { continue; }
      auto partition_table =
        std::make_unique<cudf::table>(partition_views[i], stream, space->get_default_allocator());
      output_batches.push_back(
        make_data_batch(std::move(partition_table), *space, stream, batch_telemetry()));
    }
  }

  auto end      = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  SIRIUS_LOG_DEBUG(
    "Sort partition: {} input batches → {} output batches ({} partitions) in {:.2f} ms",
    input_batches.size(),
    output_batches.size(),
    num_parts,
    duration.count() / 1000.0);

  return std::make_unique<pipelineable_operator_data>(output_batches);
}

void sirius_physical_sort_partition::on_finalize_operator()
{
  if (_sample_op) { _sample_op->clear_partition_boundaries(); }
}

}  // namespace op
}  // namespace sirius
