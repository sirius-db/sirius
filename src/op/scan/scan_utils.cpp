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
#include <duckdb/planner/filter/conjunction_filter.hpp>
#include <duckdb/planner/filter/constant_filter.hpp>
#include <duckdb/planner/filter/in_filter.hpp>
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

namespace {

/// Append @p value's string payload to @p out. False (leaving @p out untouched)
/// for a null or non-VARCHAR constant: a null never equals anything, and a
/// non-string constant means the filter is not the shape we can push down.
bool append_string_constant(duckdb::Value const& value, std::vector<std::string>& out)
{
  if (value.IsNull() || value.type().id() != duckdb::LogicalTypeId::VARCHAR) { return false; }
  out.push_back(duckdb::StringValue::Get(value));
  return true;
}

/// Collect the value set @p filter tests for equality against, or false if it is
/// any other shape. Recursive so `x = 'a' OR x = 'b'` and an ANDed IS NOT NULL
/// both resolve.
bool collect_equality_values(duckdb::TableFilter const& filter, std::vector<std::string>& out)
{
  switch (filter.filter_type) {
    case duckdb::TableFilterType::CONSTANT_COMPARISON: {
      auto const& cmp = filter.Cast<duckdb::ConstantFilter>();
      if (cmp.comparison_type != duckdb::ExpressionType::COMPARE_EQUAL) { return false; }
      return append_string_constant(cmp.constant, out);
    }
    case duckdb::TableFilterType::IN_FILTER: {
      auto const& in = filter.Cast<duckdb::InFilter>();
      if (in.values.empty()) { return false; }
      for (auto const& value : in.values) {
        if (!append_string_constant(value, out)) { return false; }
      }
      return true;
    }
    case duckdb::TableFilterType::CONJUNCTION_OR: {
      // Every branch must contribute, or the union would under-approximate.
      auto const& disjunction = filter.Cast<duckdb::ConjunctionOrFilter>();
      if (disjunction.child_filters.empty()) { return false; }
      for (auto const& child : disjunction.child_filters) {
        if (!collect_equality_values(*child, out)) { return false; }
      }
      return true;
    }
    case duckdb::TableFilterType::CONJUNCTION_AND: {
      // Only the redundant IS NOT NULL may accompany the equality: an equality
      // against a non-null constant is already false/null for a null row, so
      // absorbing it does not change which rows survive. Any other conjunct
      // would narrow the result further than the value set describes.
      auto const& conjunction = filter.Cast<duckdb::ConjunctionAndFilter>();
      bool found              = false;
      for (auto const& child : conjunction.child_filters) {
        if (child->filter_type == duckdb::TableFilterType::IS_NOT_NULL) { continue; }
        if (found) { return false; }  // two value-bearing conjuncts: not a plain equality
        if (!collect_equality_values(*child, out)) { return false; }
        found = true;
      }
      return found;
    }
    default: return false;
  }
}

}  // namespace

std::unordered_map<std::size_t, std::vector<std::string>> extract_string_equality_pushdown(
  const duckdb::TableFilterSet& filters,
  const duckdb::vector<duckdb::ColumnIndex>& column_ids,
  const duckdb::vector<sirius::logical_type>& returned_types)
{
  std::unordered_map<std::size_t, std::vector<std::string>> result;
  for (auto const& [column_index, filter] : filters.filters) {
    if (!filter || column_index >= column_ids.size()) { continue; }
    auto const& column_id = column_ids[column_index];
    if (!column_id.HasPrimaryIndex() || column_id.IsRowIdColumn() || column_id.IsEmptyColumn() ||
        column_id.IsVirtualColumn()) {
      continue;
    }
    auto const primary_idx = static_cast<std::size_t>(column_id.GetPrimaryIndex());
    if (primary_idx >= returned_types.size()) { continue; }
    // Guard the column type as well as the constants: a non-VARCHAR column can
    // never be answered by a string key comparison.
    if (sirius::to_duckdb(returned_types[primary_idx]).id() != duckdb::LogicalTypeId::VARCHAR) {
      continue;
    }
    std::vector<std::string> values;
    if (!collect_equality_values(*filter, values) || values.empty()) { continue; }
    result.emplace(primary_idx, std::move(values));
  }
  return result;
}

duckdb::unique_ptr<duckdb::Expression> convert_table_filters_to_expression(
  const duckdb::TableFilterSet& filters,
  const duckdb::vector<duckdb::ColumnIndex>& column_ids,
  const duckdb::vector<sirius::logical_type>& returned_types,
  const std::vector<std::optional<std::size_t>>& batch_position_by_column_id,
  const std::unordered_set<std::size_t>& skip_primary_indices,
  const std::unordered_set<std::size_t>& boolean_substituted_primary_indices)
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

    // The column already carries this filter's answer as a BOOL8 mask (the
    // predicate was resolved during decompression), so the batch column IS the
    // conjunct — re-expressing the comparison would compare against a mask.
    if (boolean_substituted_primary_indices.count(primary_idx)) {
      SIRIUS_LOG_DEBUG(
        "TABLE_SCAN filter: primary_idx={} substituted by a decode-time BOOL8 mask at batch "
        "position {}",
        primary_idx,
        batch_column_index);
      filter_expressions.push_back(duckdb::make_uniq<duckdb::BoundReferenceExpression>(
        duckdb::LogicalType::BOOLEAN, batch_column_index));
      continue;
    }

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

}  // namespace sirius::op
