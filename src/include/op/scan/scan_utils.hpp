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

#pragma once

// duckdb
#include <duckdb/common/types.hpp>
#include <duckdb/planner/expression/bound_conjunction_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/table_filter.hpp>

namespace sirius::op::scan {
/**
 * @brief Convert a DuckDB TableFilterSet into a single bound DuckDB expression (conjunction of
 * all filters), suitable for passing to gpu_expression_translator::translate_expression().
 *
 * Returns nullptr if the filter set is empty or contains only unsupported filter types.
 */
inline duckdb::unique_ptr<duckdb::Expression> convert_table_filters_to_expression(
  const duckdb::TableFilterSet& filters,
  const duckdb::vector<duckdb::ColumnIndex>& column_ids,
  const duckdb::vector<duckdb::LogicalType>& returned_types,
  const duckdb::vector<duckdb::idx_t>& projection_ids)
{
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> filter_expressions;

  for (auto& [column_index, filter] : filters.filters) {
    // Skip optional and IS_NOT_NULL filters
    if (filter->filter_type == duckdb::TableFilterType::OPTIONAL_FILTER ||
        filter->filter_type == duckdb::TableFilterType::IS_NOT_NULL) {
      continue;
    }

    auto primary_idx = column_ids[column_index].GetPrimaryIndex();
    auto col_type    = returned_types[primary_idx];

    // The batch columns are produced by DUCKDB_SCAN in column_ids order.
    // So the batch column index is just the column_index itself (an index into column_ids).
    duckdb::idx_t batch_column_index = column_index;

    // Create column reference for this filter - uses the batch column index
    auto column_ref =
      duckdb::make_uniq<duckdb::BoundReferenceExpression>(col_type, batch_column_index);

    // Convert filter to expression
    auto expr = filter->ToExpression(*column_ref);
    filter_expressions.push_back(std::move(expr));
  }

  // No filters to apply
  if (filter_expressions.empty()) { return nullptr; }

  // Single filter - return directly without conjunction wrapper
  if (filter_expressions.size() == 1) { return std::move(filter_expressions[0]); }

  // Multiple filters - wrap in CONJUNCTION_AND
  auto conjunction =
    duckdb::make_uniq<duckdb::BoundConjunctionExpression>(duckdb::ExpressionType::CONJUNCTION_AND);
  for (auto& expr : filter_expressions) {
    conjunction->children.push_back(std::move(expr));
  }
  return conjunction;
}

}  // namespace sirius::op::scan
