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
#include <duckdb/common/exception.hpp>
#include <duckdb/planner/expression_iterator.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>

// standard library
#include <utility>

namespace sirius::planner {

namespace {

/// True when @p op plans through `plan_comparison_join`, i.e. is a comparison join or delim join.
bool is_comparison_join(duckdb::LogicalOperator const& op)
{
  return op.type == duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
         op.type == duckdb::LogicalOperatorType::LOGICAL_DELIM_JOIN;
}

/// Scan an expression tree for resolver-state evidence: `BOUND_COLUMN_REF` exists only before the
/// ColumnBindingResolver pass, `BOUND_REF` only after it.
void classify_resolver_state(duckdb::Expression const& expr,
                             bool& has_bound_ref,
                             bool& has_column_ref)
{
  if (expr.GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF) {
    has_bound_ref = true;
  } else if (expr.GetExpressionClass() == duckdb::ExpressionClass::BOUND_COLUMN_REF) {
    has_column_ref = true;
  }
  duckdb::ExpressionIterator::EnumerateChildren(expr, [&](duckdb::Expression const& child) {
    classify_resolver_state(child, has_bound_ref, has_column_ref);
  });
}

/// A pushdown-bearing join is "already resolved" when its conditions carry positional references
/// and no column references anywhere — the unambiguous post-resolver shape.
bool join_is_already_resolved(duckdb::LogicalComparisonJoin const& join)
{
  bool has_bound_ref  = false;
  bool has_column_ref = false;
  for (auto const& cond : join.conditions) {
    if (cond.left) { classify_resolver_state(*cond.left, has_bound_ref, has_column_ref); }
    if (cond.right) { classify_resolver_state(*cond.right, has_bound_ref, has_column_ref); }
  }
  return has_bound_ref && !has_column_ref;
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

void dynamic_filter_candidate_cache::capture_pre_resolver(
  duckdb::LogicalOperator const& root,
  [[maybe_unused]] duckdb::ClientContext& context)  // reserved for C1b's domain snapshot
{
  if (_captured) {
    throw sirius::internal_exception(
      "[dynamic_filter_candidate_cache] capture_pre_resolver called twice on one cache");
  }

  for_each_comparison_join(root, [this](duckdb::LogicalComparisonJoin const& join) {
    // Only pushdown-bearing joins matter for the fence: they are where the pre-resolver snapshot
    // (C1b) must be taken, and the only entries the consumers route on.
    if (join.filter_pushdown && join_is_already_resolved(join)) { _saw_resolved_plan = true; }
    _entries.try_emplace(&join, nullptr);  // node identity; candidate filled post-resolver
  });
  _captured = true;

#ifdef DEBUG
  // Debug fence: an already-positionally-resolved plan reached the generator without a completed
  // snapshot — an upstream path resolved before create_plan(unique_ptr). Supported entry paths
  // leave resolution to the generator (the redundant pre-resolutions were removed in C1a-2).
  if (_saw_resolved_plan) {
    throw sirius::internal_exception(
      "[dynamic_filter_candidate_cache] plan was column-binding-resolved before the generator; "
      "the pre-resolver snapshot is impossible on this path");
  }
#endif
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

  for_each_comparison_join(root, [this](duckdb::LogicalComparisonJoin const& join) {
    auto candidate = std::make_shared<duckdb_join_filter_candidate const>(
      duckdb_join_filter_candidate_adapter::extract(join));
    _entries[&join] = std::move(candidate);
  });
  _extracted = true;
}

std::shared_ptr<duckdb_join_filter_candidate const> dynamic_filter_candidate_cache::find(
  duckdb::LogicalComparisonJoin const& join) const
{
  auto it = _entries.find(&join);
  if (it == _entries.end()) { return nullptr; }
  return it->second;
}

}  // namespace sirius::planner
