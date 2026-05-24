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

#include "op/window/window_op_util.hpp"

#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression/bound_window_expression.hpp"

#include <limits>
#include <stdexcept>

namespace sirius {
namespace op {

namespace {

// Phase 1 keys are guaranteed to be BoundReferenceExpressions by create_plan's guard.
// Column indices are narrowed idx_t -> int (sirius_physical_partition::_partition_keys and the cuDF
// column accessors are int); guard the narrowing even though real schemas never reach INT_MAX.
int bound_ref_index(const duckdb::Expression& expr)
{
  if (expr.GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) {
    throw std::runtime_error("convert_duckdb_window_to_cudf: window key is not a BOUND_REF");
  }
  auto index = expr.Cast<duckdb::BoundReferenceExpression>().index;
  if (index > static_cast<duckdb::idx_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("convert_duckdb_window_to_cudf: window key index exceeds int range");
  }
  return static_cast<int>(index);
}

window_rank_kind rank_kind_for(duckdb::ExpressionType type)
{
  switch (type) {
    case duckdb::ExpressionType::WINDOW_ROW_NUMBER: return window_rank_kind::ROW_NUMBER;
    case duckdb::ExpressionType::WINDOW_RANK: return window_rank_kind::RANK;
    case duckdb::ExpressionType::WINDOW_RANK_DENSE: return window_rank_kind::DENSE_RANK;
    default:
      throw std::runtime_error(
        "convert_duckdb_window_to_cudf: unsupported window expression type " +
        duckdb::ExpressionTypeToString(type));
  }
}

}  // namespace

window_definitions convert_duckdb_window_to_cudf(
  const duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>& window_exprs)
{
  window_definitions defs;
  bool keys_recorded = false;

  for (const auto& expr : window_exprs) {
    auto& window = expr->Cast<duckdb::BoundWindowExpression>();

    // PARTITION BY / ORDER BY are identical across the LogicalWindow's expressions (Phase 1 guard),
    // so record them once from the first expression.
    if (!keys_recorded) {
      for (const auto& partition : window.partitions) {
        defs.partition_idx.push_back(bound_ref_index(*partition));
      }
      for (const auto& order : window.orders) {
        defs.order_idx.push_back(bound_ref_index(*order.expression));
        const bool descending = (order.type == duckdb::OrderType::DESCENDING);
        defs.order_dirs.push_back(descending ? cudf::order::DESCENDING : cudf::order::ASCENDING);

        // SQL NULLS FIRST/LAST is an absolute position in the result. cuDF, however, applies
        // null_order in the ascending frame and then reverses it for a DESCENDING column, so for a
        // descending key we must flip BEFORE<->AFTER to keep NULLS FIRST/LAST honored as written.
        const bool nulls_first = (order.null_order == duckdb::OrderByNullType::NULLS_FIRST);
        cudf::null_order cudf_null =
          nulls_first ? cudf::null_order::BEFORE : cudf::null_order::AFTER;
        if (descending) {
          cudf_null = (cudf_null == cudf::null_order::BEFORE) ? cudf::null_order::AFTER
                                                              : cudf::null_order::BEFORE;
        }
        defs.order_null.push_back(cudf_null);
      }
      keys_recorded = true;
    }

    defs.ranks.push_back(rank_kind_for(window.type));
  }

  return defs;
}

}  // namespace op
}  // namespace sirius
