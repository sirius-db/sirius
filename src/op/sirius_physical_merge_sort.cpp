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

#include "op/sirius_physical_merge_sort.hpp"

#include "data/data_batch_utils.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "log/logging.hpp"
#include "op/merge/gpu_merge_impl.hpp"

#include <cudf/cudf_utils.hpp>

namespace sirius {
namespace op {

sirius_physical_merge_sort::sirius_physical_merge_sort(sirius_physical_order* order_by)
  : sirius_physical_merge_sort(
      order_by->types,                // copied by value
      copy_orders(order_by->orders),  // deep copy (contains unique_ptr<Expression>)
      order_by->projections,          // copied by value
      order_by->estimated_cardinality,
      order_by->is_index_sort)
{
  child_op = order_by;
}

sirius_physical_merge_sort::sirius_physical_merge_sort(
  duckdb::vector<duckdb::LogicalType> types,
  duckdb::vector<duckdb::BoundOrderByNode> orders,
  duckdb::vector<duckdb::idx_t> projections_p,
  duckdb::idx_t estimated_cardinality,
  bool is_index_sort_p)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::MERGE_SORT, std::move(types), estimated_cardinality),
    orders(std::move(orders)),
    projections(std::move(projections_p)),
    is_index_sort(is_index_sort_p)
{
}

std::vector<std::shared_ptr<cucascade::data_batch>> sirius_physical_merge_sort::execute(
  const std::vector<std::shared_ptr<cucascade::data_batch>>& input_batches,
  rmm::cuda_stream_view stream)
{
  SIRIUS_LOG_DEBUG("Executing merge sort");
  auto start = std::chrono::high_resolution_clock::now();

  // Collect valid batches and find memory space
  std::vector<std::shared_ptr<cucascade::data_batch>> valid_batches;
  cucascade::memory::memory_space* space = nullptr;
  for (auto const& batch : input_batches) {
    if (!batch) { continue; }
    if (!space) { space = batch->get_memory_space(); }
    valid_batches.push_back(batch);
  }

  if (valid_batches.empty() || !space) { return {}; }

  // Helper lambda to apply final projection to a batch (removes sort-key-only columns)
  auto apply_final_projection =
    [this, stream, space](
      std::shared_ptr<cucascade::data_batch> batch) -> std::shared_ptr<cucascade::data_batch> {
    if (_final_projections.empty() || !batch) { return batch; }
    auto table_view = sirius::get_cudf_table_view(*batch);
    std::vector<cudf::column_view> projected_cols;
    for (auto idx : _final_projections) {
      projected_cols.push_back(table_view.column(static_cast<cudf::size_type>(idx)));
    }
    auto projected_table = std::make_unique<cudf::table>(
      cudf::table_view(projected_cols), stream, space->get_default_allocator());
    return sirius::make_data_batch(std::move(projected_table), *space);
  };

  // Single batch: no merge needed
  if (valid_batches.size() == 1) {
    std::vector<std::shared_ptr<cucascade::data_batch>> outputs;
    outputs.push_back(apply_final_projection(valid_batches[0]));
    return outputs;
  }

  // Build cudf order vectors from BoundOrderByNode
  std::vector<int> order_key_idx;
  std::vector<cudf::order> column_order;
  std::vector<cudf::null_order> null_precedence;
  order_key_idx.reserve(orders.size());
  column_order.reserve(orders.size());
  null_precedence.reserve(orders.size());

  for (auto const& ord : orders) {
    if (ord.expression->expression_class != duckdb::ExpressionClass::BOUND_REF) {
      throw duckdb::NotImplementedException("Merge sort only supports bound reference expressions");
    }
    auto idx = static_cast<int>(ord.expression->Cast<duckdb::BoundReferenceExpression>().index);
    order_key_idx.push_back(idx);
    column_order.push_back(ord.type == duckdb::OrderType::ASCENDING ? cudf::order::ASCENDING
                                                                    : cudf::order::DESCENDING);
    null_precedence.push_back(ord.null_order == duckdb::OrderByNullType::NULLS_FIRST
                                ? cudf::null_order::BEFORE
                                : cudf::null_order::AFTER);
  }

  auto merged_batch = gpu_merge_impl::merge_order_by(
    valid_batches, order_key_idx, column_order, null_precedence, stream, *space);

  auto end      = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  SIRIUS_LOG_DEBUG("Merge sort time: {:.2f} ms", duration.count() / 1000.0);

  std::vector<std::shared_ptr<cucascade::data_batch>> outputs;
  if (merged_batch) { outputs.push_back(apply_final_projection(std::move(merged_batch))); }
  return outputs;
}

}  // namespace op
}  // namespace sirius
