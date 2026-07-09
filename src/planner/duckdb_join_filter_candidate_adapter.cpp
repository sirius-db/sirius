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

#include "planner/duckdb_join_filter_candidate_adapter.hpp"

#include <duckdb/common/assert.hpp>
#include <duckdb/common/helper.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <duckdb/planner/table_filter.hpp>

#include <algorithm>
#include <utility>

namespace sirius::planner::duckdb_join_filter_candidate_adapter {

static_assert(sizeof(std::size_t) >= sizeof(duckdb::idx_t),
              "DuckDB idx_t ordinals are stored as std::size_t without narrowing");

duckdb::unique_ptr<duckdb::JoinFilterPushdownInfo> detail::clone_filter_pushdown_info(
  duckdb::JoinFilterPushdownInfo const& src)
{
  auto dst                   = duckdb::make_uniq<duckdb::JoinFilterPushdownInfo>();
  dst->join_condition        = src.join_condition;
  dst->build_side_has_filter = src.build_side_has_filter;
  dst->probe_info.reserve(src.probe_info.size());
  for (auto const& pi : src.probe_info) {
    duckdb::JoinFilterPushdownFilter copy_pi;
    copy_pi.dynamic_filters = pi.dynamic_filters;
    copy_pi.columns         = pi.columns;
    dst->probe_info.push_back(std::move(copy_pi));
  }
  // min_max_aggregates intentionally omitted: Sirius does not read them.
  return dst;
}

namespace {

bool structurally_aligned(duckdb::LogicalOperator const& a, duckdb::LogicalOperator const& b)
{
  if (a.type != b.type || a.children.size() != b.children.size()) { return false; }
  for (std::size_t i = 0; i < a.children.size(); ++i) {
    if (!structurally_aligned(*a.children[i], *b.children[i])) { return false; }
  }
  return true;
}

void preserve_aligned(duckdb::LogicalOperator const& original,
                      duckdb::LogicalOperator& copy,
                      preserved_counts& counts)
{
  if (original.type == duckdb::LogicalOperatorType::LOGICAL_GET) {
    auto const& o = original.Cast<duckdb::LogicalGet>();
    auto& c       = copy.Cast<duckdb::LogicalGet>();
    if (o.dynamic_filters) {
      c.dynamic_filters = o.dynamic_filters;
      ++counts.gets;
    }
  } else if (original.type == duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
    auto const& o = original.Cast<duckdb::LogicalComparisonJoin>();
    auto& c       = copy.Cast<duckdb::LogicalComparisonJoin>();
    if (o.filter_pushdown) {
      c.filter_pushdown = detail::clone_filter_pushdown_info(*o.filter_pushdown);
      ++counts.joins;
    }
  }
  for (std::size_t i = 0; i < original.children.size(); ++i) {
    preserve_aligned(*original.children[i], *copy.children[i], counts);
  }
}

}  // namespace

void preserve_dynamic_filter_metadata(duckdb::LogicalOperator const& original,
                                      duckdb::LogicalOperator& copy,
                                      preserved_counts& counts)
{
  auto const aligned = structurally_aligned(original, copy);
  D_ASSERT(aligned);
  if (!aligned) { return; }
  preserve_aligned(original, copy, counts);
}

duckdb_join_filter_candidate extract(duckdb::LogicalComparisonJoin const& op)
{
  duckdb_join_filter_candidate out;
  if (!op.filter_pushdown) { return out; }
  auto const& info = *op.filter_pushdown;

  out.build_subtree_has_filter_hint = info.build_side_has_filter;

  if (info.probe_info.empty()) {
    out.kind = duckdb_candidate_kind::statistics_only;
    return out;
  }

  auto const malformed = []() {
    duckdb_join_filter_candidate m;
    m.kind = duckdb_candidate_kind::malformed;
    return m;
  };

  if (info.join_condition.empty()) { return malformed(); }

  // Compute and validate the join's condition ordinals and comparisons.
  out.condition_indexes.reserve(info.join_condition.size());
  out.condition_comparisons.reserve(info.join_condition.size());
  for (auto const cond_idx : info.join_condition) {
    if (cond_idx >= op.conditions.size()) { return malformed(); }
    if (std::find(out.condition_indexes.begin(), out.condition_indexes.end(), cond_idx) !=
        out.condition_indexes.end()) {
      return malformed();
    }
    out.condition_indexes.push_back(cond_idx);
    out.condition_comparisons.push_back(op.conditions[cond_idx].comparison);
  }

  // Collect probe targets with non-null channels.
  out.targets.reserve(info.probe_info.size());
  for (auto const& pi : info.probe_info) {
    if (pi.columns.size() != info.join_condition.size()) { return malformed(); }
    if (!pi.dynamic_filters) { continue; }

    duckdb_probe_target_candidate tgt;
    tgt.channel_identity = pi.dynamic_filters;
    tgt.columns.reserve(pi.columns.size());
    for (auto const& col : pi.columns) {
      duckdb_probe_target_candidate::probe_column pc;
      pc.column_index = col.probe_column_index.column_index;
      pc.storage_type = col.storage_type;
      tgt.columns.push_back(std::move(pc));
    }
    out.targets.push_back(std::move(tgt));
  }

  if (out.targets.empty()) {
    // Every recorded target had a null channel: nothing usable remains, so the candidate as a
    // whole fails closed.
    return malformed();
  }

  out.kind = duckdb_candidate_kind::admitted;
  return out;
}

duckdb::shared_ptr<duckdb::DynamicTableFilterSet> scan_channel_identity(
  duckdb::LogicalGet const& get)
{
  return get.dynamic_filters;
}

}  // namespace sirius::planner::duckdb_join_filter_candidate_adapter
