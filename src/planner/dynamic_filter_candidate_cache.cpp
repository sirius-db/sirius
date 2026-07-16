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

// sirius
#include <planner/dynamic_filter_candidate_cache.hpp>
#include <sirius/exception.hpp>

// duckdb
#include <duckdb/planner/operator/logical_comparison_join.hpp>

namespace sirius::planner {

namespace {

/// True when @p op plans through `plan_comparison_join`, i.e. is a comparison join or delim join.
bool is_comparison_join(duckdb::LogicalOperator const& op)
{
  return op.type == duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
         op.type == duckdb::LogicalOperatorType::LOGICAL_DELIM_JOIN;
}

template <class Fn>
void for_each_comparison_join(duckdb::LogicalOperator const& op, Fn&& fn)
{
  if (is_comparison_join(op)) { fn(op.Cast<duckdb::LogicalComparisonJoin>()); }
  for (auto const& child : op.children) {
    for_each_comparison_join(*child, fn);
  }
}

}  // namespace

void dynamic_filter_candidate_cache::capture_pre_resolver(duckdb::LogicalOperator const& root)
{
  if (_captured) {
    throw sirius::internal_exception(
      "[dynamic_filter_candidate_cache] capture_pre_resolver called twice on one cache");
  }

  logical_join_set captured_joins;
  for_each_comparison_join(root, [&captured_joins](duckdb::LogicalComparisonJoin const& join) {
    captured_joins.emplace(&join);
  });
  _captured_joins.swap(captured_joins);
  _captured = true;
}

void dynamic_filter_candidate_cache::extract_post_resolver(duckdb::LogicalOperator const& root)
{
  if (!_captured) {
    throw sirius::internal_exception(
      "[dynamic_filter_candidate_cache] extract_post_resolver called before capture_pre_resolver");
  }
  if (_extracted) {
    throw sirius::internal_exception(
      "[dynamic_filter_candidate_cache] extract_post_resolver called twice on one cache");
  }

  candidate_map extracted_entries;
  extracted_entries.reserve(_captured_joins.size());
  for_each_comparison_join(
    root, [this, &extracted_entries](duckdb::LogicalComparisonJoin const& join) {
      if (_captured_joins.find(&join) == _captured_joins.end()) {
        throw sirius::internal_exception(
          "[dynamic_filter_candidate_cache] post-resolver tree contains an uncaptured join");
      }
      auto const [it, inserted] =
        extracted_entries.emplace(&join, duckdb_join_filter_candidate_adapter::extract(join));
      static_cast<void>(it);
      if (!inserted) {
        throw sirius::internal_exception(
          "[dynamic_filter_candidate_cache] post-resolver tree visits one join more than once");
      }
    });
  if (extracted_entries.size() != _captured_joins.size()) {
    throw sirius::internal_exception(
      "[dynamic_filter_candidate_cache] post-resolver tree is missing a captured join");
  }
  _entries.swap(extracted_entries);
  _extracted = true;
}

duckdb_join_filter_candidate const& dynamic_filter_candidate_cache::candidate_for(
  duckdb::LogicalComparisonJoin const& join) const
{
  if (!_extracted) {
    throw sirius::internal_exception(
      "[dynamic_filter_candidate_cache] candidate requested before successful extraction");
  }
  auto it = _entries.find(&join);
  if (it == _entries.end()) {
    throw sirius::internal_exception(
      "[dynamic_filter_candidate_cache] candidate requested for an uncaptured join");
  }
  return it->second;
}

}  // namespace sirius::planner
