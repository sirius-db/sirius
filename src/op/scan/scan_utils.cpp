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

// sirius
#include <duckdb/planner/expression/bound_conjunction_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <expression/ast/from_duckdb.hpp>
#include <helper/type_conversions.hpp>
#include <log/logging.hpp>
#include <op/scan/scan_utils.hpp>

// standard library
#include <format>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sirius::op {

std::vector<std::optional<std::size_t>> build_batch_column_map(
  const duckdb::vector<duckdb::idx_t>& projection_ids, std::size_t column_ids_count)
{
  std::vector<std::optional<std::size_t>> map(column_ids_count);  // default-constructs to nullopt

  if (projection_ids.empty()) {
    for (std::size_t i = 0; i < column_ids_count; i++) {
      map[i] = i;
    }
    return map;
  }

  // Sort projected indices — this matches the iteration order in
  // make_selected_column_indices which walks column_ids[0..N) and
  // includes only indices present in the projected set.
  std::vector<duckdb::idx_t> sorted(projection_ids.begin(), projection_ids.end());
  std::sort(sorted.begin(), sorted.end());

  for (std::size_t batch_pos = 0; batch_pos < sorted.size(); batch_pos++) {
    if (sorted[batch_pos] < column_ids_count) { map[sorted[batch_pos]] = batch_pos; }
  }
  return map;
}

duckdb::unique_ptr<duckdb::Expression> convert_table_filters_to_expression(
  const duckdb::TableFilterSet& filters,
  const duckdb::vector<duckdb::ColumnIndex>& column_ids,
  const duckdb::vector<sirius::logical_type>& returned_types,
  const std::vector<std::optional<std::size_t>>& batch_position_by_column_id,
  const std::unordered_set<std::size_t>& skip_primary_indices)
{
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> filter_expressions;

  for (auto& [column_index, filter] : filters.filters) {
    // Skip optional and IS_NOT_NULL filters
    if (filter->filter_type == duckdb::TableFilterType::OPTIONAL_FILTER ||
        filter->filter_type == duckdb::TableFilterType::IS_NOT_NULL) {
      continue;
    }

    auto primary_idx = column_ids.at(column_index).GetPrimaryIndex();
    if (skip_primary_indices.count(primary_idx)) {
      SIRIUS_LOG_DEBUG(
        "TABLE_SCAN filter: skipping filter on primary_idx={} (hive partition or equivalent)",
        primary_idx);
      continue;
    }
    auto const col_type = returned_types.at(primary_idx);

    SIRIUS_LOG_DEBUG("TABLE_SCAN filter: column_index={}, primary_idx={}, type={}, filter_type={}",
                     column_index,
                     primary_idx,
                     col_type.to_string(),
                     static_cast<int>(filter->filter_type));

    auto const& batch_pos = batch_position_by_column_id[column_index];
    if (!batch_pos.has_value()) {
      throw std::runtime_error(
        std::format("TABLE_SCAN filter: column_index ({}) not in projected batch", column_index));
    }
    auto const batch_column_index = static_cast<duckdb::idx_t>(*batch_pos);

    SIRIUS_LOG_DEBUG("TABLE_SCAN filter: batch_column_index={}", batch_column_index);

    auto column_ref = duckdb::make_uniq<duckdb::BoundReferenceExpression>(
      sirius::to_duckdb(col_type), batch_column_index);
    auto expr = filter->ToExpression(*column_ref);
    filter_expressions.push_back(std::move(expr));
  }

  if (filter_expressions.empty()) { return nullptr; }
  if (filter_expressions.size() == 1) { return std::move(filter_expressions[0]); }

  auto conjunction =
    duckdb::make_uniq<duckdb::BoundConjunctionExpression>(duckdb::ExpressionType::CONJUNCTION_AND);
  for (auto& expr : filter_expressions) {
    conjunction->children.push_back(std::move(expr));
  }
  return conjunction;
}

std::optional<gpu_expression_translator::translated_expression>
translate_duckdb_expression_with_names(gpu_expression_translator& translator,
                                       duckdb::Expression const& expr,
                                       gpu_expression_translator::column_name_resolver_fxn resolver)
{
  auto node = sirius::ast::from_duckdb(expr);
  if (!node) { return std::nullopt; }
  return translator.translate_expression_with_names(*node, std::move(resolver));
}

std::string table_filters_to_string(const duckdb::TableFilterSet& filters,
                                    const duckdb::vector<duckdb::ColumnIndex>& column_ids,
                                    std::span<const std::string> names)
{
  std::string result;
  for (const auto& [column_index, filter] : filters.filters) {
    if (!filter) { continue; }
    std::string column_name;
    if (column_index < column_ids.size()) {
      const auto col_id = column_ids[column_index].GetPrimaryIndex();
      column_name       = col_id < names.size() ? names[col_id] : "#" + std::to_string(col_id);
    } else {
      column_name = "#" + std::to_string(column_index);
    }
    if (!result.empty()) { result += " AND "; }
    result += filter->ToString(column_name);
  }
  return result;
}

}  // namespace sirius::op
