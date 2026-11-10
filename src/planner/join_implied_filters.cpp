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

#include "planner/join_implied_filters.hpp"

#include "duckdb/common/multi_file/multi_file_states.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/filter/expression_filter.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/storage/statistics/numeric_stats.hpp"
#include "log/logging.hpp"
#include "planner/scan_column_origin.hpp"

#include <memory>
#include <optional>

namespace sirius::planner {
namespace {

constexpr double k_min_keep_ratio = 0.99;

struct key_column {
  duckdb::LogicalGet* get;
  duckdb::ProjectionIndex filter_index;
  duckdb::Identifier name;
  std::unique_ptr<duckdb::BaseStatistics> statistics;
};

/// The scan column join key @p ordinal of @p child reads, with the min and max its Parquet
/// footers give, or nullopt when the key does not reach a scan row by row or has no such bounds.
std::optional<key_column> footer_key_column(duckdb::LogicalOperator const& child,
                                            std::size_t ordinal)
{
  auto const origin = resolve_scan_column_origin(child, ordinal, origin_policy::row_wise);
  if (!origin) { return std::nullopt; }
  auto& get        = const_cast<duckdb::LogicalGet&>(*origin->get);
  auto const* bind = dynamic_cast<duckdb::MultiFileBindData const*>(get.bind_data.get());
  if (!bind) { return std::nullopt; }
  auto const& ids = get.GetColumnIds();
  auto const index =
    get.projection_ids.empty() ? origin->ordinal : get.projection_ids[origin->ordinal].GetIndex();
  if (index >= ids.size() || ids[index].IsVirtualColumn()) { return std::nullopt; }
  auto const column = ids[index].GetPrimaryIndex();
  if (column >= get.names.size() || column >= get.returned_types.size()) { return std::nullopt; }
  auto const& type = get.returned_types[column];
  if (!type.IsIntegral()) { return std::nullopt; }
  auto statistics = bind->file_options.GetColumnStatistics(get.names[column], type);
  if (!statistics || !duckdb::NumericStats::HasMinMax(*statistics)) { return std::nullopt; }
  return key_column{&get, duckdb::ProjectionIndex(index), get.names[column], std::move(statistics)};
}

double range_width(duckdb::BaseStatistics const& statistics)
{
  auto const min =
    duckdb::NumericStats::Min(statistics).DefaultCastAs(duckdb::LogicalType::HUGEINT);
  auto const max =
    duckdb::NumericStats::Max(statistics).DefaultCastAs(duckdb::LogicalType::HUGEINT);
  return duckdb::Hugeint::Cast<double>(max.GetValue<duckdb::hugeint_t>() -
                                       min.GetValue<duckdb::hugeint_t>()) +
         1.0;
}

/// Removes @p side's filter when it passes every value of @p other's key range and that range
/// spans at least `k_min_keep_ratio` of @p side's own.
void drop_implied_filter(key_column const& side, key_column const& other)
{
  if (side.statistics->GetType() != other.statistics->GetType()) { return; }
  auto const filter = side.get->table_filters.TryGetFilterByColumnIndex(side.filter_index);
  if (!filter || filter->filter_type != duckdb::TableFilterType::EXPRESSION_FILTER) { return; }
  auto const& expression_filter = filter->Cast<duckdb::ExpressionFilter>();
  if (!expression_filter.expr || !expression_filter.column_indexes.empty()) { return; }
  // NULL keys never match an equality, so only the other side's non-NULL keys must pass.
  auto matched_keys = other.statistics->Copy();
  matched_keys.Set(duckdb::StatsInfo::CANNOT_HAVE_NULL_VALUES);
  if (duckdb::ExpressionFilter::CheckExpressionStatistics(*expression_filter.expr, matched_keys) !=
      duckdb::FilterPropagateResult::FILTER_ALWAYS_TRUE) {
    return;
  }
  // An implied filter keeps at least the other side's range of the column. The share of the range
  // stands in for the share of rows, which holds for unique or evenly spread keys; rows bunched
  // on keys outside the other side's range can make the filter remove more.
  if (range_width(*other.statistics) < k_min_keep_ratio * range_width(*side.statistics)) { return; }
  SIRIUS_LOG_DEBUG("[join_implied_filters] Dropping {} on {}: the join implies it",
                   expression_filter.ToString(side.name.GetIdentifierName()),
                   side.name.GetIdentifierName());
  side.get->table_filters.RemoveFilterByColumnIndex(side.filter_index);
}

}  // namespace

void drop_join_implied_filters(duckdb::LogicalComparisonJoin& join)
{
  switch (join.join_type) {
    case duckdb::JoinType::INNER:
    case duckdb::JoinType::SEMI:
    case duckdb::JoinType::RIGHT_SEMI: break;
    default: return;
  }
  if (join.children.size() != 2) { return; }
  for (auto const& condition : join.conditions) {
    if (!condition.IsComparison() ||
        condition.GetComparisonType() != duckdb::ExpressionType::COMPARE_EQUAL ||
        condition.GetLHS().GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF ||
        condition.GetRHS().GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) {
      continue;
    }
    auto const left = footer_key_column(
      *join.children[0], condition.GetLHS().Cast<duckdb::BoundReferenceExpression>().Index());
    auto const right = footer_key_column(
      *join.children[1], condition.GetRHS().Cast<duckdb::BoundReferenceExpression>().Index());
    if (!left || !right) { continue; }
    drop_implied_filter(*left, *right);
    drop_implied_filter(*right, *left);
  }
}

}  // namespace sirius::planner
