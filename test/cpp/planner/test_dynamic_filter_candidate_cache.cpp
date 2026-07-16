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

/**
 * @file test_dynamic_filter_candidate_cache.cpp
 * @brief Contract tests for sirius::planner::dynamic_filter_candidate_cache (C1a-2a): the
 *        single-use capture/extract protocol and exact pre/post node correlation.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/execution/operator/join/join_filter_pushdown.hpp>
#include <duckdb/function/table_function.hpp>
#include <duckdb/planner/expression/bound_columnref_expression.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <planner/dynamic_filter_candidate_cache.hpp>
#include <sirius/exception.hpp>

#include <type_traits>
#include <utility>
#include <vector>

using sirius::planner::duckdb_candidate_kind;
using sirius::planner::dynamic_filter_candidate_cache;

namespace {

duckdb::unique_ptr<duckdb::Expression> make_side([[maybe_unused]] duckdb::ClientContext& context)
{
  return duckdb::make_uniq<duckdb::BoundColumnRefExpression>(duckdb::LogicalType::INTEGER,
                                                             duckdb::ColumnBinding{1, 0});
}

duckdb::JoinCondition make_condition(duckdb::ClientContext& context)
{
  duckdb::JoinCondition cond;
  cond.left       = make_side(context);
  cond.right      = make_side(context);
  cond.comparison = duckdb::ExpressionType::COMPARE_EQUAL;
  return cond;
}

/// One-condition pushdown metadata targeting one live channel, mirroring the adapter test's
/// minimal admitted shape.
duckdb::unique_ptr<duckdb::JoinFilterPushdownInfo> make_admitted_pushdown()
{
  auto info                   = duckdb::make_uniq<duckdb::JoinFilterPushdownInfo>();
  info->join_condition        = duckdb::vector<duckdb::idx_t>{0};
  info->build_side_has_filter = true;
  duckdb::JoinFilterPushdownFilter target;
  target.dynamic_filters = duckdb::make_shared_ptr<duckdb::DynamicTableFilterSet>();
  duckdb::JoinFilterPushdownColumn col;
  col.probe_column_index = duckdb::ColumnBinding{1, 0};
  col.storage_type       = duckdb::LogicalType::INTEGER;
  target.columns.push_back(col);
  info->probe_info.push_back(target);
  return info;
}

duckdb::unique_ptr<duckdb::LogicalComparisonJoin> make_join(
  duckdb::ClientContext& context,
  bool with_pushdown,
  duckdb::LogicalOperatorType logical_type = duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN)
{
  auto join =
    duckdb::make_uniq<duckdb::LogicalComparisonJoin>(duckdb::JoinType::INNER, logical_type);
  join->conditions.push_back(make_condition(context));
  if (with_pushdown) { join->filter_pushdown = make_admitted_pushdown(); }
  return join;
}

/// A minimal constructible LogicalGet for tree shape (never executed).
duckdb::unique_ptr<duckdb::LogicalGet> make_get()
{
  return duckdb::make_uniq<duckdb::LogicalGet>(
    /*table_index=*/0,
    duckdb::TableFunction(),
    /*bind_data=*/nullptr,
    duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER},
    duckdb::vector<duckdb::string>{"a"});
}

/// Shared in-memory database used to construct bound DuckDB expressions.
struct test_context {
  duckdb::DuckDB db{nullptr};
  duckdb::Connection con{db};
  duckdb::ClientContext& context() { return *con.context; }
};

}  // namespace

static_assert(!std::is_copy_constructible_v<dynamic_filter_candidate_cache>);
static_assert(!std::is_copy_assignable_v<dynamic_filter_candidate_cache>);
static_assert(!std::is_move_constructible_v<dynamic_filter_candidate_cache>);
static_assert(!std::is_move_assignable_v<dynamic_filter_candidate_cache>);

//===-----------------------------------------------------------------------------------------===//
// Single-use protocol
//===-----------------------------------------------------------------------------------------===//

TEST_CASE("cache rejects a second capture_pre_resolver", "[dynamic_filter][candidate_cache]")
{
  test_context tc;
  auto join = make_join(tc.context(), /*with_pushdown=*/false);

  dynamic_filter_candidate_cache cache;
  cache.capture_pre_resolver(*join);
  REQUIRE_THROWS_AS(cache.capture_pre_resolver(*join), sirius::internal_exception);
}

TEST_CASE("cache rejects extract_post_resolver before capture_pre_resolver",
          "[dynamic_filter][candidate_cache]")
{
  test_context tc;
  auto join = make_join(tc.context(), /*with_pushdown=*/false);

  dynamic_filter_candidate_cache cache;
  REQUIRE_THROWS_AS(cache.extract_post_resolver(*join), sirius::internal_exception);
  REQUIRE_THROWS_AS(cache.candidate_for(*join), sirius::internal_exception);
}

TEST_CASE("cache rejects a second extract_post_resolver", "[dynamic_filter][candidate_cache]")
{
  test_context tc;
  auto join = make_join(tc.context(), /*with_pushdown=*/false);

  dynamic_filter_candidate_cache cache;
  cache.capture_pre_resolver(*join);
  cache.extract_post_resolver(*join);
  REQUIRE_THROWS_AS(cache.extract_post_resolver(*join), sirius::internal_exception);
}

//===-----------------------------------------------------------------------------------------===//
// Candidate lookup
//===-----------------------------------------------------------------------------------------===//

TEST_CASE("candidate lookup rejects a join outside the extracted tree",
          "[dynamic_filter][candidate_cache]")
{
  test_context tc;
  auto captured = make_join(tc.context(), /*with_pushdown=*/false);
  auto stranger = make_join(tc.context(), /*with_pushdown=*/false);

  dynamic_filter_candidate_cache cache;
  REQUIRE_THROWS_AS(cache.candidate_for(*captured), sirius::internal_exception);

  cache.capture_pre_resolver(*captured);
  cache.extract_post_resolver(*captured);
  REQUIRE_THROWS_AS(cache.candidate_for(*stranger), sirius::internal_exception);
}

TEST_CASE("candidate lookup rejects access before extraction", "[dynamic_filter][candidate_cache]")
{
  test_context tc;
  auto join = make_join(tc.context(), /*with_pushdown=*/true);

  dynamic_filter_candidate_cache cache;
  cache.capture_pre_resolver(*join);
  REQUIRE_THROWS_AS(cache.candidate_for(*join), sirius::internal_exception);
}

TEST_CASE("candidate lookup returns an explicit absent value for a pushdown-free join",
          "[dynamic_filter][candidate_cache]")
{
  test_context tc;
  auto join = make_join(tc.context(), /*with_pushdown=*/false);

  dynamic_filter_candidate_cache cache;
  cache.capture_pre_resolver(*join);
  cache.extract_post_resolver(*join);

  REQUIRE(cache.candidate_for(*join).kind() == duckdb_candidate_kind::absent);
}

TEST_CASE("candidate lookup returns the immutable extracted values",
          "[dynamic_filter][candidate_cache]")
{
  test_context tc;
  auto join = make_join(tc.context(), /*with_pushdown=*/true);

  dynamic_filter_candidate_cache cache;
  cache.capture_pre_resolver(*join);
  cache.extract_post_resolver(*join);

  auto const& candidate = cache.candidate_for(*join);
  REQUIRE(candidate.kind() == duckdb_candidate_kind::admitted);
  REQUIRE(candidate.condition_indexes() == std::vector<std::size_t>{0});
}

TEST_CASE("DELIM joins share the plan_comparison_join entry space",
          "[dynamic_filter][candidate_cache]")
{
  test_context tc;
  // A DELIM join carrying pushdown, with a plain comparison join below it: both plan through
  // plan_comparison_join, so both must be captured and extractable.
  auto delim      = make_join(tc.context(),
                         /*with_pushdown=*/true,
                         duckdb::LogicalOperatorType::LOGICAL_DELIM_JOIN);
  auto inner      = make_join(tc.context(), /*with_pushdown=*/false);
  auto* inner_raw = inner.get();
  delim->children.push_back(std::move(inner));
  delim->children.push_back(make_get());

  dynamic_filter_candidate_cache cache;
  cache.capture_pre_resolver(*delim);
  cache.extract_post_resolver(*delim);

  REQUIRE(cache.candidate_for(*delim).kind() == duckdb_candidate_kind::admitted);
  REQUIRE(cache.candidate_for(*inner_raw).kind() == duckdb_candidate_kind::absent);
}

//===-----------------------------------------------------------------------------------------===//
// Exact pre/post correlation and transactional publication
//===-----------------------------------------------------------------------------------------===//

TEST_CASE("cache rejects an uncaptured post-resolver join without publishing partial entries",
          "[dynamic_filter][candidate_cache]")
{
  test_context tc;
  auto root = make_join(tc.context(), /*with_pushdown=*/true);

  dynamic_filter_candidate_cache cache;
  cache.capture_pre_resolver(*root);

  auto extra      = make_join(tc.context(), /*with_pushdown=*/true);
  auto* extra_raw = extra.get();
  root->children.push_back(std::move(extra));
  REQUIRE_THROWS_AS(cache.extract_post_resolver(*root), sirius::internal_exception);
  REQUIRE_THROWS_AS(cache.candidate_for(*root), sirius::internal_exception);
  REQUIRE_THROWS_AS(cache.candidate_for(*extra_raw), sirius::internal_exception);

  root->children.clear();
  cache.extract_post_resolver(*root);
  REQUIRE(cache.candidate_for(*root).kind() == duckdb_candidate_kind::admitted);
}

TEST_CASE("cache rejects a missing captured join and remains retryable",
          "[dynamic_filter][candidate_cache]")
{
  test_context tc;
  auto root       = make_join(tc.context(), /*with_pushdown=*/false);
  auto child      = make_join(tc.context(), /*with_pushdown=*/true);
  auto* child_raw = child.get();
  root->children.push_back(std::move(child));

  dynamic_filter_candidate_cache cache;
  cache.capture_pre_resolver(*root);

  auto detached_child = std::move(root->children.front());
  root->children.clear();
  REQUIRE_THROWS_AS(cache.extract_post_resolver(*root), sirius::internal_exception);
  REQUIRE_THROWS_AS(cache.candidate_for(*root), sirius::internal_exception);
  REQUIRE_THROWS_AS(cache.candidate_for(*child_raw), sirius::internal_exception);

  root->children.push_back(std::move(detached_child));
  cache.extract_post_resolver(*root);
  REQUIRE(cache.candidate_for(*root).kind() == duckdb_candidate_kind::absent);
  REQUIRE(cache.candidate_for(*child_raw).kind() == duckdb_candidate_kind::admitted);
}
