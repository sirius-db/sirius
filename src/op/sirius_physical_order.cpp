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

#include "data/data_batch_utils.hpp"
#include "log/logging.hpp"
#include "op/order/bounded_sort.hpp"
#include "sirius/exception.hpp"
#include "telemetry/nvtx.hpp"

#include <cudf/sorting.hpp>

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

    {
      nvtx_scoped_range local_range{"sort.local_runs"};
      auto table                = get_cudf_table_view(batch);
      std::size_t initial_bytes = 0, chunks = 0;
      for (auto slice : bounded_sort_slices(
             table, chunk_bytes, batch.get_data()->get_uncompressed_data_size_in_bytes(), stream)) {
        auto indices = cudf::sorted_order(slice.select(order_key_idx),
                                          column_order,
                                          null_precedence,
                                          stream,
                                          space->get_default_allocator());
        auto sorted  = cudf::gather(slice.select(proj_idx),
                                   indices->view(),
                                   cudf::out_of_bounds_policy::DONT_CHECK,
                                   stream,
                                   space->get_default_allocator());
        auto output  = make_data_batch(std::move(sorted), *space, stream, batch_telemetry());
        {
          auto metadata = output->to_read_only();
          initial_bytes = memory::saturating_add(
            initial_bytes, metadata.get_data()->get_uncompressed_data_size_in_bytes());
        }
        output_batches.push_back(std::move(output));
        ++chunks;
      }
      SIRIUS_LOG_DEBUG("bounded sort: initial_rows={}, initial_bytes={}, run_chunks={}",
                       table.num_rows(),
                       initial_bytes,
                       chunks);
    }
  }

  return std::make_unique<pipelineable_operator_data>(output_batches);
}

}  // namespace op
}  // namespace sirius
