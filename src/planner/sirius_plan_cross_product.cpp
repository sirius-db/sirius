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

#include "duckdb/planner/operator/logical_cross_product.hpp"
#include "op/sirius_physical_nested_loop_join.hpp"
#include "planner/sirius_physical_plan_generator.hpp"

#include <cudf/types.hpp>

#include <limits>
#include <string>

namespace sirius::planner {

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan(duckdb::LogicalCrossProduct& op)
{
  D_ASSERT(op.children.size() == 2);

  // A cross product runs as a nested loop join without conditions, which calls cudf::cross_join
  // on each pair of input batches and materializes every output row of the pair in one table.
  // Its estimate is the product of the child estimates. The generic create_plan() has already set
  // op.estimated_cardinality to the larger child when the join order optimizer set none.
  constexpr auto max_rows = static_cast<duckdb::idx_t>(std::numeric_limits<cudf::size_type>::max());
  auto const left_rows    = op.children[0]->EstimateCardinality(context);
  auto const right_rows   = op.children[1]->EstimateCardinality(context);
  if (left_rows != 0 && right_rows > max_rows / left_rows) {
    throw duckdb::NotImplementedException(
      "Cross product of an estimated " + std::to_string(left_rows) + " x " +
      std::to_string(right_rows) + " rows not supported in GPU");
  }
  op.estimated_cardinality = left_rows * right_rows;

  auto left  = create_plan(*op.children[0]);
  auto right = create_plan(*op.children[1]);
  return duckdb::make_uniq<sirius::op::sirius_physical_nested_loop_join>(
    op,
    std::move(left),
    std::move(right),
    duckdb::vector<sirius::join_condition>{},
    duckdb::JoinType::INNER,
    op.estimated_cardinality);
}

}  // namespace sirius::planner
