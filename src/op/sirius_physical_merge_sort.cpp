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
#include "op/cudf_sort_order.hpp"
#include "op/merge/gpu_merge_impl.hpp"
#include "sirius/exception.hpp"

#include <nvtx3/nvtx3.hpp>

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
  duckdb::vector<sirius::logical_type> types,
  duckdb::vector<duckdb::BoundOrderByNode> orders,
  duckdb::vector<std::size_t> projections_p,
  std::size_t estimated_cardinality,
  bool is_index_sort_p)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::MERGE_SORT, std::move(types), estimated_cardinality),
    orders(std::move(orders)),
    projections(std::move(projections_p)),
    is_index_sort(is_index_sort_p)
{
}

std::unique_ptr<operator_data> sirius_physical_merge_sort::get_next_task_input_data()
{
  // Drain all batches from one partition per call (one merge task per partition).
  // Follows the same pattern as sirius_physical_grouped_aggregate_merge.
  std::lock_guard<std::mutex> lg(_drain_mutex);

  auto* repo = ports.begin()->second->repo;
  if (_current_partition_index < repo->num_partitions()) {
    std::vector<std::shared_ptr<cucascade::data_batch>> all_batches;
    while (true) {
      auto batch = repo->pop_next_data_batch(_current_partition_index);
      if (!batch) { break; }
      all_batches.push_back(std::move(batch));
    }
    _current_partition_index++;
    SIRIUS_LOG_DEBUG("merge_sort: drained {} batches for partition {}",
                     all_batches.size(),
                     _current_partition_index - 1);
    if (all_batches.empty()) { return nullptr; }
    // Tag with the source partition index so the scheduler pins this task to
    // partition_idx % num_gpus and spreads partitions across GPUs, independent of
    // where the upstream scan placed the bytes (mirrors grouped_aggregate_merge).
    return std::make_unique<partitioned_operator_data>(std::move(all_batches),
                                                       _current_partition_index - 1);
  }
  return nullptr;
}

std::unique_ptr<operator_data> sirius_physical_merge_sort::execute(const operator_data& input_data,
                                                                   rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_merge_sort::execute"};
  auto& input               = dynamic_cast<const pipelineable_operator_data&>(input_data);
  const auto& input_batches = input.get_read_only_batches();

  // Find memory space
  cucascade::memory::memory_space* space = nullptr;
  for (auto const& batch : input_batches) {
    if (!space) { space = batch.get_memory_space(); }
  }

  if (input_batches.empty() || !space) {
    return std::make_unique<pipelineable_operator_data>(
      std::vector<std::shared_ptr<cucascade::data_batch>>{});
  }

  // Helper lambda to apply final projection to a batch (removes sort-key-only columns)
  auto apply_final_projection =
    [this, stream, space](
      const cucascade::read_only_data_batch& ro) -> std::shared_ptr<cucascade::data_batch> {
    auto table_view = sirius::get_cudf_table_view(ro);
    std::vector<cudf::column_view> projected_cols;
    for (auto idx : _final_projections) {
      projected_cols.push_back(table_view.column(static_cast<cudf::size_type>(idx)));
    }
    auto projected_table = std::make_unique<cudf::table>(
      cudf::table_view(projected_cols), stream, space->get_default_allocator());
    return sirius::make_data_batch(std::move(projected_table), *space, stream);
  };

  // Single batch: no merge needed
  if (input_batches.size() == 1) {
    std::vector<std::shared_ptr<cucascade::data_batch>> outputs;
    if (_final_projections.empty()) {
      auto ro_vec = input.get_read_only_batches();
      outputs.push_back(cucascade::data_batch::to_idle(std::move(ro_vec[0])));
    } else {
      outputs.push_back(apply_final_projection(input_batches[0]));
    }
    return std::make_unique<pipelineable_operator_data>(outputs);
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
      throw not_implemented_exception("Merge sort only supports bound reference expressions");
    }
    auto idx = static_cast<int>(ord.expression->Cast<duckdb::BoundReferenceExpression>().index);
    order_key_idx.push_back(idx);
    column_order.push_back(to_cudf_order(ord.type));
    null_precedence.push_back(to_cudf_null_order(ord.type, ord.null_order));
  }

  auto merged_batch = gpu_merge_impl::merge_order_by(
    input_batches, order_key_idx, column_order, null_precedence, stream, *space);

  std::vector<std::shared_ptr<cucascade::data_batch>> outputs;
  if (merged_batch) {
    if (_final_projections.empty()) {
      outputs.push_back(std::move(merged_batch));
    } else {
      auto ro = merged_batch->to_read_only();
      outputs.push_back(apply_final_projection(ro));
    }
  }
  return std::make_unique<pipelineable_operator_data>(outputs);
}

}  // namespace op
}  // namespace sirius
