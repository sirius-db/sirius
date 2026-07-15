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

#include "cudf/cudf_utils.hpp"
#include "duckdb/common/assert.hpp"
#include "expression/aggregate_id.hpp"
#include "expression/ast/node.hpp"

#include <format>
#include <stdexcept>
#include <string>
#include <string_view>

namespace sirius {
namespace op {

namespace {

// Single place that builds the "Unsupported aggregate function: <name>" diagnostic so the
// message (and the aggregate_id -> name lookup) is not repeated at every rejection site.
[[noreturn]] void throw_unsupported_aggregate(sirius::aggregate_id fid,
                                              std::string_view detail = {})
{
  auto const name = sirius::to_duckdb_aggregate_name(fid);
  throw std::runtime_error(detail.empty()
                             ? std::format("Unsupported aggregate function: {}", name)
                             : std::format("Unsupported aggregate function: {} {}", name, detail));
}

}  // namespace

std::optional<cudf::aggregation::Kind> to_cudf_aggregation_kind(sirius::aggregate_id id)
{
  switch (id) {
    case sirius::aggregate_id::sum:
    case sirius::aggregate_id::sum_no_overflow: return cudf::aggregation::Kind::SUM;
    case sirius::aggregate_id::count: return cudf::aggregation::Kind::COUNT_VALID;
    case sirius::aggregate_id::count_star: return cudf::aggregation::Kind::COUNT_ALL;
    case sirius::aggregate_id::min: return cudf::aggregation::Kind::MIN;
    case sirius::aggregate_id::max: return cudf::aggregation::Kind::MAX;
    case sirius::aggregate_id::avg:
    case sirius::aggregate_id::first: return std::nullopt;
  }
  return std::nullopt;
}

CudfAggregateDefinitions convert_duckdb_aggregates_to_cudf(
  const duckdb::vector<std::unique_ptr<sirius::ast::node>>& groups_p,
  const duckdb::vector<std::unique_ptr<sirius::ast::node>>& expressions)
{
  CudfAggregateDefinitions result;

  // 1. Extract group_idx from groups_p
  for (const auto& group : groups_p) {
    auto const& ref =
      sirius::ast::require_reference(group.get(), "convert_duckdb_aggregates_to_cudf group");
    result.group_idx.push_back(static_cast<int>(ref.column_index));
  }

  // 2. Extract aggregates (cudf::aggregation::Kind) from expressions
  for (const auto& aggregate : expressions) {
    auto const& aggr = sirius::ast::require_aggregate(
      aggregate.get(), "convert_duckdb_aggregates_to_cudf aggregate");
    auto const fid       = aggr.function();
    auto const& children = aggr.arguments();

    // Handle AVG specially: it expands into SUM + COUNT_VALID
    if (fid == sirius::aggregate_id::avg) {
      D_ASSERT(children.size() == 1);
      D_ASSERT(children[0]->is_reference());
      auto col_idx = static_cast<int>(children[0]->as_reference().column_index);

      size_t sum_position = result.cudf_aggregates.size();
      result.cudf_aggregates.push_back(cudf::aggregation::Kind::SUM);
      result.cudf_aggregate_idx.push_back(col_idx);
      result.cudf_aggregate_struct_col_indices.push_back({});
      result.cudf_aggregates.push_back(cudf::aggregation::Kind::COUNT_VALID);
      result.cudf_aggregate_idx.push_back(col_idx);
      result.cudf_aggregate_struct_col_indices.push_back({});
      result.aggregate_slots.push_back(
        AggregateSlot{true, false, sum_position, sirius::get_cudf_type(aggr.return_type())});
      result.has_avg = true;
      continue;
    }

    // Handle COUNT(DISTINCT col) and COUNT(DISTINCT (col1, col2, ...)):
    // Use COLLECT_SET locally; merge via MERGE_SETS; then count list elements.
    // For multi-column, a struct column is synthesized from the component columns.
    if (aggr.distinct() && fid == sirius::aggregate_id::count) {
      D_ASSERT(children.size() == 1);
      auto const& child = *children[0];
      size_t position   = result.cudf_aggregates.size();
      result.cudf_aggregates.push_back(cudf::aggregation::Kind::COLLECT_SET);

      if (child.is_reference()) {
        // Single-column case: COUNT(DISTINCT col)
        result.cudf_aggregate_idx.push_back(static_cast<int>(child.as_reference().column_index));
        result.cudf_aggregate_struct_col_indices.push_back({});
      } else {
        // Multi-column case: COUNT(DISTINCT (col1, col2, ...)) — child is a struct_pack expression
        D_ASSERT(child.is_function_call());
        auto const& func_expr = child.as_function_call();
        std::vector<int> struct_indices;
        for (auto const& arg : func_expr.arguments()) {
          D_ASSERT(arg->is_reference());
          struct_indices.push_back(static_cast<int>(arg->as_reference().column_index));
        }
        D_ASSERT(!struct_indices.empty());
        result.cudf_aggregate_idx.push_back(-1);  // sentinel: struct column, see gpu_aggregate_impl
        result.cudf_aggregate_struct_col_indices.push_back(std::move(struct_indices));
      }

      result.aggregate_slots.push_back(AggregateSlot{false, true, position});
      result.has_count_distinct = true;
      continue;
    }

    auto const agg_kind = to_cudf_aggregation_kind(fid);
    if (!agg_kind) { throw_unsupported_aggregate(fid); }
    size_t current_position = result.cudf_aggregates.size();
    result.cudf_aggregates.push_back(*agg_kind);

    // 3. Extract aggregate_idx from the children of the aggregate expression
    if (children.empty()) {
      // COUNT(*) has no children - use 0 as a placeholder (will be handled by COUNT_ALL)
      if (fid == sirius::aggregate_id::count_star) {
        result.cudf_aggregate_idx.push_back(0);
      } else {
        throw_unsupported_aggregate(fid, "with no children");
      }
    } else {
      if (children.size() == 1) {
        // Extract the column index from the first child (most aggregates have one child)
        D_ASSERT(children[0]->is_reference());
        result.cudf_aggregate_idx.push_back(
          static_cast<int>(children[0]->as_reference().column_index));
      } else {
        throw_unsupported_aggregate(fid, "with " + std::to_string(children.size()) + " children");
      }
    }
    result.cudf_aggregate_struct_col_indices.push_back({});
    result.aggregate_slots.push_back(AggregateSlot{false, false, current_position});
  }

  return result;
}

namespace {

std::string_view cudf_aggregation_kind_to_sql_name(cudf::aggregation::Kind kind)
{
  switch (kind) {
    case cudf::aggregation::Kind::SUM: return "sum";
    case cudf::aggregation::Kind::MIN: return "min";
    case cudf::aggregation::Kind::MAX: return "max";
    case cudf::aggregation::Kind::MEAN: return "avg";
    case cudf::aggregation::Kind::COUNT_VALID: return "count";
    case cudf::aggregation::Kind::COUNT_ALL: return "count";
    case cudf::aggregation::Kind::NTH_ELEMENT: return "first";
    case cudf::aggregation::Kind::COLLECT_SET: return "collect_set";
    case cudf::aggregation::Kind::MERGE_SETS: return "merge_sets";
    default: return "agg";
  }
}

}  // namespace

std::string cudf_aggregate_definitions_to_string(
  const std::vector<int>& group_idx,
  const std::vector<cudf::aggregation::Kind>& cudf_aggregates,
  const std::vector<int>& cudf_aggregate_idx,
  const std::vector<std::vector<int>>& cudf_aggregate_struct_col_indices,
  const std::vector<AggregateSlot>& aggregate_slots)
{
  std::string keys;
  for (const int idx : group_idx) {
    if (!keys.empty()) { keys += ", "; }
    keys += "#" + std::to_string(idx);
  }

  std::string aggs;
  for (const AggregateSlot& slot : aggregate_slots) {
    if (slot.cudf_idx >= cudf_aggregates.size()) { continue; }  // display-only: never throw
    if (!aggs.empty()) { aggs += ", "; }
    const std::string input_col = slot.cudf_idx < cudf_aggregate_idx.size()
                                    ? "#" + std::to_string(cudf_aggregate_idx[slot.cudf_idx])
                                    : std::string{"?"};
    if (slot.is_avg) {
      aggs += "avg(" + input_col + ")";
    } else if (slot.is_count_distinct) {
      std::string cols;
      if (slot.cudf_idx < cudf_aggregate_struct_col_indices.size()) {
        for (const int col : cudf_aggregate_struct_col_indices[slot.cudf_idx]) {
          if (!cols.empty()) { cols += ", "; }
          cols += "#" + std::to_string(col);
        }
      }
      aggs += "count(DISTINCT " + (cols.empty() ? input_col : cols) + ")";
    } else if (const auto kind = cudf_aggregates[slot.cudf_idx];
               kind == cudf::aggregation::Kind::COUNT_ALL) {
      aggs += "count(*)";
    } else {
      aggs += std::string{cudf_aggregation_kind_to_sql_name(kind)} + "(" + input_col + ")";
    }
  }

  return "keys: " + (keys.empty() ? std::string{"(none)"} : keys) +
         "; aggs: " + (aggs.empty() ? std::string{"(none)"} : aggs);
}

}  // namespace op
}  // namespace sirius
