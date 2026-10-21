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

#include "planner/join_key_nullability.hpp"

#include <duckdb/catalog/catalog_entry/table_catalog_entry.hpp>
#include <duckdb/parser/constraints/not_null_constraint.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <duckdb/planner/operator/logical_filter.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <duckdb/planner/operator/logical_order.hpp>
#include <duckdb/planner/operator/logical_projection.hpp>
#include <duckdb/storage/statistics/base_statistics.hpp>

#include <optional>
#include <utility>
#include <vector>

namespace sirius::planner {
namespace {

// A certificate for a bare key whose native-table constraint survives every forwarding step.
class non_null_key {
 public:
  static std::optional<non_null_key> parse(duckdb::LogicalOperator const& subtree,
                                           duckdb::Expression const& expression)
  {
    if (expression.GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) {
      return std::nullopt;
    }
    auto ordinal     = expression.Cast<duckdb::BoundReferenceExpression>().Index();
    auto const& type = expression.GetReturnType();
    auto const* node = &subtree;
    while (true) {
      if (ordinal >= node->types.size() || node->types[ordinal] != type) { return std::nullopt; }
      if (node->type == duckdb::LogicalOperatorType::LOGICAL_GET) {
        return from_scan(node->Cast<duckdb::LogicalGet>(), ordinal, type);
      }
      if (node->children.size() != 1 || !node->children.front()) { return std::nullopt; }
      switch (node->type) {
        case duckdb::LogicalOperatorType::LOGICAL_PROJECTION: {
          auto const& projection = node->Cast<duckdb::LogicalProjection>();
          if (ordinal >= projection.expressions.size()) { return std::nullopt; }
          auto const& input = *projection.expressions[ordinal];
          if (input.GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF ||
              input.GetReturnType() != type) {
            return std::nullopt;
          }
          ordinal = input.Cast<duckdb::BoundReferenceExpression>().Index();
          break;
        }
        case duckdb::LogicalOperatorType::LOGICAL_FILTER: {
          auto const& map = node->Cast<duckdb::LogicalFilter>().projection_map;
          if (!map.empty()) {
            if (ordinal >= map.size()) { return std::nullopt; }
            ordinal = static_cast<duckdb::idx_t>(map[ordinal]);
          }
          break;
        }
        case duckdb::LogicalOperatorType::LOGICAL_ORDER_BY: {
          auto const& map = node->Cast<duckdb::LogicalOrder>().projection_map;
          if (!map.empty()) {
            if (ordinal >= map.size()) { return std::nullopt; }
            ordinal = static_cast<duckdb::idx_t>(map[ordinal]);
          }
          break;
        }
        case duckdb::LogicalOperatorType::LOGICAL_LIMIT:
        case duckdb::LogicalOperatorType::LOGICAL_TOP_N:
        case duckdb::LogicalOperatorType::LOGICAL_DISTINCT: break;
        default: return std::nullopt;
      }
      node = node->children.front().get();
    }
  }

 private:
  non_null_key() = default;

  static std::optional<non_null_key> from_scan(duckdb::LogicalGet const& scan,
                                               duckdb::idx_t ordinal,
                                               duckdb::LogicalType const& type)
  {
    if (!scan.children.empty() || !scan.projected_input.empty() || scan.has_pushed_projection ||
        scan.function.GetName().GetIdentifierName() != "seq_scan" || !scan.bind_data) {
      return std::nullopt;
    }
    auto const& columns = scan.GetColumnIds();
    if (!scan.projection_ids.empty() && ordinal >= scan.projection_ids.size()) {
      return std::nullopt;
    }
    auto const index = scan.projection_ids.empty()
                         ? ordinal
                         : static_cast<duckdb::idx_t>(scan.projection_ids[ordinal]);
    if (index >= columns.size()) { return std::nullopt; }
    auto const& column = columns[index];
    if (!column.HasPrimaryIndex() || column.IsRowIdColumn() || column.IsVirtualColumn() ||
        column.HasChildren()) {
      return std::nullopt;
    }
    auto table = scan.GetTable();
    if (!table || !table->TryGetDuckTableEntry() ||
        column.GetPrimaryIndex() >= table->GetColumns().LogicalColumnCount()) {
      return std::nullopt;
    }
    auto const logical_index = column.ToLogical();
    auto const& definition   = table->GetColumn(logical_index);
    if (definition.Generated() || definition.Type() != type) { return std::nullopt; }
    for (auto const& constraint : table->GetConstraints()) {
      if (constraint->type == duckdb::ConstraintType::NOT_NULL &&
          constraint->Cast<duckdb::NotNullConstraint>().index == logical_index) {
        return non_null_key{};
      }
    }
    return std::nullopt;
  }
};

}  // namespace

std::size_t normalize_non_null_join_keys(duckdb::LogicalComparisonJoin& join)
{
  // MARK observes FALSE versus UNKNOWN, even when their matching row sets agree.
  if (join.type != duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
      join.join_type == duckdb::JoinType::MARK || join.children.size() != 2 || !join.children[0] ||
      !join.children[1]) {
    return 0;
  }
  std::vector<std::size_t> normalized_keys;
  for (std::size_t index = 0; index < join.conditions.size(); ++index) {
    auto const& condition = join.conditions[index];
    if (!condition.IsComparison() ||
        condition.GetComparisonType() != duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
      continue;
    }
    auto const left  = non_null_key::parse(*join.children[0], condition.GetLHS());
    auto const right = non_null_key::parse(*join.children[1], condition.GetRHS());
    // Partial normalization can turn a hash equijoin into a mixed predicate join.
    if (!left && !right) { return 0; }
    normalized_keys.push_back(index);
  }
  for (auto index : normalized_keys) {
    auto& condition = join.conditions[index];
    duckdb::JoinCondition replacement(std::move(condition.LeftReference()),
                                      std::move(condition.RightReference()),
                                      duckdb::ExpressionType::COMPARE_EQUAL);
    if (condition.GetLeftStats()) {
      replacement.SetLeftStats(condition.GetLeftStats()->ToUnique());
    }
    if (condition.GetRightStats()) {
      replacement.SetRightStats(condition.GetRightStats()->ToUnique());
    }
    condition = std::move(replacement);
  }
  return normalized_keys.size();
}

}  // namespace sirius::planner
