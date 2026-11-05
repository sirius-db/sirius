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

#include "op/order/gpu_order_impl.hpp"
#include "sirius/exception.hpp"
#include "telemetry/nvtx.hpp"

namespace sirius {
namespace op {

sirius_physical_order::sirius_physical_order(duckdb::vector<sirius::logical_type> types,
                                             std::size_t input_width,
                                             duckdb::vector<duckdb::BoundOrderByNode> orders,
                                             duckdb::vector<std::size_t> projections_p,
                                             std::size_t estimated_cardinality,
                                             bool is_index_sort_p)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::ORDER_BY, std::move(types), estimated_cardinality),
    orders(std::move(orders)),
    comparison(std::make_shared<const sort_comparison_config>(this->orders, input_width)),
    projections(std::move(projections_p)),
    is_index_sort(is_index_sort_p)
{
}

std::unique_ptr<operator_data> sirius_physical_order::execute(const operator_data& input_data,
                                                              ::cuda::stream_ref stream)
{
  nvtx_scoped_range nvtx_range{"sirius_physical_order::execute"};
  auto& input               = dynamic_cast<const pipelineable_operator_data&>(input_data);
  const auto& input_batches = input.get_read_only_batches();

  auto const& order_key_idx   = comparison->key_columns;
  auto const& column_order    = comparison->directions;
  auto const& null_precedence = comparison->null_precedence;

  std::vector<int> proj_idx(projections.begin(), projections.end());

  std::vector<std::shared_ptr<cucascade::data_batch>> output_batches;
  output_batches.reserve(input_batches.size());

  for (auto const& batch : input_batches) {
    auto* space = batch.get_memory_space();
    if (!space) { continue; }

    auto sorted_batch = gpu_order_impl::local_order_by(batch,
                                                       order_key_idx,
                                                       column_order,
                                                       null_precedence,
                                                       proj_idx,
                                                       stream,
                                                       *space,
                                                       batch_telemetry());
    if (sorted_batch) { output_batches.push_back(std::move(sorted_batch)); }
  }

  return std::make_unique<pipelineable_operator_data>(output_batches);
}

}  // namespace op
}  // namespace sirius
