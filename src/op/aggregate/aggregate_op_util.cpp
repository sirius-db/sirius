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

#include "op/aggregate/aggregate_op_util.hpp"

#include "duckdb/common/assert.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"

#include <stdexcept>
#include <string>

namespace sirius {
namespace op {

CudfAggregateDefinitions convert_duckdb_aggregates_to_cudf(
  const duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>& groups_p,
  const duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>& expressions)
{
  CudfAggregateDefinitions result;

  // 1. Extract group_idx from groups_p
  for (const auto& group : groups_p) {
    D_ASSERT(group->type == duckdb::ExpressionType::BOUND_REF);
    auto& bound_ref = group->Cast<duckdb::BoundReferenceExpression>();
    result.group_idx.push_back(static_cast<int>(bound_ref.index));
  }

  // 2. Extract aggregates (cudf::aggregation::Kind) from expressions
  for (const auto& aggregate : expressions) {
    auto& aggr = aggregate->Cast<duckdb::BoundAggregateExpression>();

    // Convert DuckDB aggregate function name to cudf::aggregation::Kind
    cudf::aggregation::Kind agg_kind;
    if (aggr.function.name == "sum" || aggr.function.name == "sum_no_overflow") {
      agg_kind = cudf::aggregation::Kind::SUM;
    } else if (aggr.function.name == "count") {
      agg_kind = cudf::aggregation::Kind::COUNT_VALID;
    } else if (aggr.function.name == "count_star") {
      agg_kind = cudf::aggregation::Kind::COUNT_ALL;
    } else if (aggr.function.name == "min") {
      agg_kind = cudf::aggregation::Kind::MIN;
    } else if (aggr.function.name == "max") {
      agg_kind = cudf::aggregation::Kind::MAX;
    } else {
      throw std::runtime_error("Unsupported aggregate function: " + aggr.function.name);
    }
    result.cudf_aggregates.push_back(agg_kind);

    // 3. Extract aggregate_idx from the children of the aggregate expression
    if (aggr.children.empty()) {
      // COUNT(*) has no children - use 0 as a placeholder (will be handled by COUNT_ALL)
      if (aggr.function.name == "count_star") {
        result.cudf_aggregate_idx.push_back(0);
      } else {
        throw std::runtime_error("Unsupported aggregate function: " + aggr.function.name +
                                 " with no children");
      }
    } else {
      if (aggr.children.size() == 1) {
        // Extract the column index from the first child (most aggregates have one child)
        D_ASSERT(aggr.children[0]->type == duckdb::ExpressionType::BOUND_REF);
        auto& bound_ref = aggr.children[0]->Cast<duckdb::BoundReferenceExpression>();
        result.cudf_aggregate_idx.push_back(static_cast<int>(bound_ref.index));
      } else {
        throw std::runtime_error("Unsupported aggregate function: " + aggr.function.name +
                                 " with " + std::to_string(aggr.children.size()) + " children");
      }
    }
  }

  return result;
}

}  // namespace op
}  // namespace sirius
