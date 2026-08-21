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
#include <duckdb/common/types/value.hpp>
#include <duckdb/planner/expression/bound_comparison_expression.hpp>
#include <duckdb/planner/expression/bound_conjunction_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/filter/conjunction_filter.hpp>
#include <duckdb/planner/filter/constant_filter.hpp>
#include <duckdb/planner/filter/in_filter.hpp>
#include <expression/ast/from_duckdb.hpp>
#include <helper/type_conversions.hpp>
#include <log/logging.hpp>
#include <op/scan/scan_utils.hpp>

// standard library
#include <cstdint>
#include <format>
#include <limits>
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

resolved_filter_column resolve_filtered_column(
  duckdb::idx_t column_index,
  const duckdb::vector<duckdb::ColumnIndex>& column_ids,
  const std::vector<std::optional<std::size_t>>& batch_position_by_column_id,
  const std::unordered_set<std::size_t>& skip_primary_indices)
{
  if (column_index >= column_ids.size()) { return {}; }
  auto const primary_index = static_cast<std::size_t>(column_ids[column_index].GetPrimaryIndex());
  if (skip_primary_indices.count(primary_index) != 0) {
    SIRIUS_LOG_DEBUG(
      "TABLE_SCAN filter: skipping filter on primary_idx={} (hive partition or equivalent)",
      primary_index);
    return {};
  }
  if (column_index >= batch_position_by_column_id.size() ||
      !batch_position_by_column_id[column_index].has_value()) {
    return {filter_column_status::not_in_batch, primary_index, 0};
  }
  return {filter_column_status::usable, primary_index, *batch_position_by_column_id[column_index]};
}

std::vector<table_filter_conjunct> decompose_table_filters(
  const duckdb::TableFilterSet& filters,
  const duckdb::vector<duckdb::ColumnIndex>& column_ids,
  const duckdb::vector<sirius::logical_type>& returned_types,
  const std::vector<std::optional<std::size_t>>& batch_position_by_column_id,
  const std::unordered_set<std::size_t>& skip_primary_indices)
{
  std::vector<table_filter_conjunct> conjuncts;

  for (auto& [column_index, filter] : filters.filters) {
    // Skip optional and IS_NOT_NULL filters
    if (filter->filter_type == duckdb::TableFilterType::OPTIONAL_FILTER ||
        filter->filter_type == duckdb::TableFilterType::IS_NOT_NULL) {
      continue;
    }

    auto const column = resolve_filtered_column(
      column_index, column_ids, batch_position_by_column_id, skip_primary_indices);
    if (column.status == filter_column_status::skipped) { continue; }
    if (column.status == filter_column_status::not_in_batch) {
      // A conjunct that has to be EVALUATED cannot reference a column that was
      // never materialized — unlike a filter used only for pruning, which may
      // be dropped. Loud, because it is a wiring bug rather than a shape we do
      // not support.
      throw std::runtime_error(
        std::format("TABLE_SCAN filter: column_index ({}) not in projected batch", column_index));
    }
    auto const col_type           = returned_types.at(column.primary_index);
    auto const batch_column_index = static_cast<duckdb::idx_t>(column.batch_position);

    SIRIUS_LOG_DEBUG(
      "TABLE_SCAN filter: column_index={}, primary_idx={}, type={}, filter_type={}, "
      "batch_column_index={}",
      column_index,
      column.primary_index,
      col_type.to_string(),
      static_cast<int>(filter->filter_type),
      batch_column_index);

    auto column_ref = duckdb::make_uniq<duckdb::BoundReferenceExpression>(
      sirius::to_duckdb(col_type), batch_column_index);
    conjuncts.push_back(
      {column.primary_index, column.batch_position, filter->ToExpression(*column_ref)});
  }

  return conjuncts;
}

duckdb::unique_ptr<duckdb::Expression> convert_table_filters_to_expression(
  const duckdb::TableFilterSet& filters,
  const duckdb::vector<duckdb::ColumnIndex>& column_ids,
  const duckdb::vector<sirius::logical_type>& returned_types,
  const std::vector<std::optional<std::size_t>>& batch_position_by_column_id,
  const std::unordered_set<std::size_t>& skip_primary_indices)
{
  auto conjuncts = decompose_table_filters(
    filters, column_ids, returned_types, batch_position_by_column_id, skip_primary_indices);

  if (conjuncts.empty()) { return nullptr; }
  if (conjuncts.size() == 1) { return std::move(conjuncts[0].expr); }

  auto conjunction =
    duckdb::make_uniq<duckdb::BoundConjunctionExpression>(duckdb::ExpressionType::CONJUNCTION_AND);
  for (auto& conjunct : conjuncts) {
    conjunction->children.push_back(std::move(conjunct.expr));
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

}  // namespace sirius::op
