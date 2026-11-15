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

#pragma once

#include "duckdb/planner/bound_query_node.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "op/cudf_sort_order.hpp"
#include "sirius/exception.hpp"

#include <limits>
#include <memory>
#include <vector>

namespace sirius::op {

/// Executable arguments validated against the input schema at construction and shared
/// by the stages established by ORDER_BY lowering.
struct sort_comparison_config {
  std::vector<int> key_columns;
  std::vector<cudf::order> directions;
  std::vector<cudf::null_order> null_precedence;

  sort_comparison_config(const duckdb::vector<duckdb::BoundOrderByNode>& orders,
                         std::size_t input_width)
  {
    for (auto const& order : orders) {
      if (order.expression->GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) {
        throw not_implemented_exception("Sort only supports bound reference expressions");
      }
      auto index = order.expression->Cast<duckdb::BoundReferenceExpression>().Index();
      if (index > static_cast<duckdb::idx_t>(std::numeric_limits<int>::max())) {
        throw internal_exception("Sort key index exceeds cuDF column limit");
      }
      if (index >= input_width) {
        throw invalid_input_exception("Sort key index exceeds input column count");
      }
      key_columns.push_back(static_cast<int>(index));
      directions.push_back(to_cudf_order(order.type));
      null_precedence.push_back(to_cudf_null_order(order.type, order.null_order));
    }
  }
};

}  // namespace sirius::op
