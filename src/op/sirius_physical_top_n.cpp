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

#include "op/sirius_physical_top_n.hpp"

#include "duckdb/common/assert.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/create_sort_key.hpp"
#include "duckdb/planner/filter/dynamic_filter.hpp"
#include "duckdb/storage/data_table.hpp"
#include "log/logging.hpp"
#include "op/sirius_physical_order.hpp"
#include "utils.hpp"

namespace sirius {
namespace op {

sirius_physical_top_n::sirius_physical_top_n(
  duckdb::vector<duckdb::LogicalType> types_p,
  duckdb::vector<duckdb::BoundOrderByNode> orders,
  duckdb::idx_t limit,
  duckdb::idx_t offset,
  duckdb::shared_ptr<duckdb::DynamicFilterData> dynamic_filter_p,
  duckdb::idx_t estimated_cardinality)
  : sirius_physical_operator(
      duckdb::PhysicalOperatorType::TOP_N, std::move(types_p), estimated_cardinality),
    orders(std::move(orders)),
    limit(limit),
    offset(offset),
    dynamic_filter(std::move(dynamic_filter_p))
{
  // sort_result = duckdb::make_shared_ptr<GPUIntermediateRelation>(types.size());
  // for (int col = 0; col < types.size(); col++) {
  //   sort_result->columns[col] = nullptr;
  // }
}

sirius_physical_top_n::~sirius_physical_top_n() {}

}  // namespace op
}  // namespace sirius
