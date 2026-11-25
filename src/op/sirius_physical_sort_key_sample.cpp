/*
 * Copyright 2026, Sirius Contributors.
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

#include "op/sirius_physical_sort_key_sample.hpp"

#include "data/data_batch_utils.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "sirius/exception.hpp"
#include "telemetry/nvtx.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>

#include <stdexcept>
#include <vector>

namespace sirius {
namespace op {

sirius_physical_sort_key_sample::sirius_physical_sort_key_sample(
  sirius_physical_order* order_by, std::shared_ptr<sort_key_sample> key_sample)
  : sirius_physical_sort_key_sample(order_by->types,
                                    copy_orders(order_by->orders),
                                    order_by->estimated_cardinality,
                                    std::move(key_sample))
{
}

sirius_physical_sort_key_sample::sirius_physical_sort_key_sample(
  duckdb::vector<sirius::logical_type> types,
  duckdb::vector<duckdb::BoundOrderByNode> orders,
  std::size_t estimated_cardinality,
  std::shared_ptr<sort_key_sample> key_sample)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::SORT_KEY_SAMPLE, std::move(types), estimated_cardinality),
    orders(std::move(orders)),
    _key_sample(std::move(key_sample))
{
  if (!_key_sample) {
    throw std::invalid_argument("[sirius_physical_sort_key_sample] the key-row store is null");
  }
}

std::unique_ptr<operator_data> sirius_physical_sort_key_sample::execute(
  const operator_data& input_data, ::cuda::stream_ref stream)
{
  nvtx_scoped_range nvtx_range{"sirius_physical_sort_key_sample::execute"};
  auto& input         = dynamic_cast<const pipelineable_operator_data&>(input_data);
  const auto& batches = input.get_read_only_batches();

  std::vector<int> key_indices;
  key_indices.reserve(orders.size());
  for (auto const& ord : orders) {
    if (ord.expression->GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) {
      throw not_implemented_exception("Sort key sample only supports bound reference expressions");
    }
    key_indices.push_back(
      static_cast<int>(ord.expression->Cast<duckdb::BoundReferenceExpression>().Index()));
  }

  // One per batch. A retry hands the same batches over again, as clones with new IDs if it moved
  // to another GPU, and their key rows must replace the earlier ones.
  auto const original_ids = input.original_batch_ids();
  if (original_ids.size() != batches.size()) {
    throw internal_exception("Sort key sample input has {} batches and {} original batch IDs",
                             batches.size(),
                             original_ids.size());
  }

  for (std::size_t position = 0; position < batches.size(); ++position) {
    auto const& batch = batches[position];
    auto* space       = batch.get_memory_space();
    if (!space || !batch.get_data()) { continue; }
    auto const table = get_cudf_table_view(batch);
    std::vector<cudf::column_view> key_columns;
    key_columns.reserve(key_indices.size());
    for (int idx : key_indices) {
      key_columns.push_back(table.column(idx));
    }
    _key_sample->add(original_ids[position],
                     cudf::table_view(key_columns),
                     batch.get_data()->get_size_in_bytes(),
                     *space,
                     stream);
  }

  return std::make_unique<pipelineable_operator_data>(input.get_data_batches());
}

}  // namespace op
}  // namespace sirius
