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

#include "op/sirius_physical_order.hpp"

#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "op/order/gpu_order_impl.hpp"
#include "sirius/exception.hpp"

#include <nvtx3/nvtx3.hpp>

namespace sirius {
namespace op {

sirius_physical_order::sirius_physical_order(duckdb::vector<sirius::logical_type> types,
                                             duckdb::vector<duckdb::BoundOrderByNode> orders,
                                             duckdb::vector<std::size_t> projections_p,
                                             std::size_t estimated_cardinality,
                                             bool is_index_sort_p)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::ORDER_BY, std::move(types), estimated_cardinality),
    orders(std::move(orders)),
    projections(std::move(projections_p)),
    is_index_sort(is_index_sort_p)
{
}

std::unique_ptr<operator_data> sirius_physical_order::execute(const operator_data& input_data,
                                                              rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_order::execute"};
  auto& input               = dynamic_cast<const pipelineable_operator_data&>(input_data);
  const auto& input_batches = input.get_data_batches();

  // Build cudf order vectors from BoundOrderByNode
  std::vector<int> order_key_idx;
  std::vector<cudf::order> column_order;
  std::vector<cudf::null_order> null_precedence;
  order_key_idx.reserve(orders.size());
  column_order.reserve(orders.size());
  null_precedence.reserve(orders.size());

  for (auto const& ord : orders) {
    if (ord.expression->expression_class != duckdb::ExpressionClass::BOUND_REF) {
      throw not_implemented_exception("Order by only supports bound reference expressions");
    }
    auto idx = static_cast<int>(ord.expression->Cast<duckdb::BoundReferenceExpression>().index);
    order_key_idx.push_back(idx);
    column_order.push_back(ord.type == duckdb::OrderType::ASCENDING ? cudf::order::ASCENDING
                                                                    : cudf::order::DESCENDING);
    null_precedence.push_back(ord.null_order == duckdb::OrderByNullType::NULLS_FIRST
                                ? cudf::null_order::BEFORE
                                : cudf::null_order::AFTER);
  }

  std::vector<int> proj_idx(projections.begin(), projections.end());

  std::vector<std::shared_ptr<cucascade::data_batch>> output_batches;
  output_batches.reserve(input_batches.size());

  for (auto const& batch : input_batches) {
    if (!batch) { continue; }
    // INVARIANT (SCHED-RR contract): all input batches arrive on target_space
    // via gpu_pipeline_task::execute_pipeline_task_round ->
    // pipelineable_operator_data::prepare_for_processing -> lock_or_prepare_batch.
    // batches[0]->get_memory_space() == target_space here.
    // See .planning/phases/15-mgpu-operator-colocation-audit/15-AUDIT-LOG.md.
    //
    // Phase 18 / DB-02 Recipe R1: scoped read-only accessor to resolve the
    // memory_space pointer through the public RAII surface (#117 makes
    // data_batch::get_memory_space private). The memory_space object is
    // owned by the long-lived reservation manager, so the pointer remains
    // valid after the accessor drops. We drop the accessor before calling
    // gpu_order_impl::local_order_by because that impl itself takes its
    // own scoped accessor on the same batch (P1 discipline).
    cucascade::memory::memory_space* space = nullptr;
    {
      auto ro = batch->to_read_only();
      space   = ro.get_memory_space();
    }
    if (!space) { continue; }

    auto sorted_batch = gpu_order_impl::local_order_by(
      batch, order_key_idx, column_order, null_precedence, proj_idx, stream, *space);
    if (sorted_batch) { output_batches.push_back(std::move(sorted_batch)); }
  }

  return std::make_unique<pipelineable_operator_data>(output_batches);
}

}  // namespace op
}  // namespace sirius
