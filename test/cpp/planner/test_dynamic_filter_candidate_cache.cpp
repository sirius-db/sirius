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
 * @brief Contract tests for sirius::planner::dynamic_filter_candidate_cache (C1a-2): the
 *        single-use capture/extract protocol, the resolver fence, and find()'s tri-state.
 *
 * The fence cases are build-flavor-split on purpose: a pushdown-bearing plan already in
 * post-resolver form throws in DEBUG builds and records saw_resolved_plan() in release builds
 * (where extraction must still be positionally correct). Each flavor asserts its own branch and
 * WARNs about the other, so a pass in one flavor never silently vacates the other's coverage —
 * the C1a-2 gate runs both.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/execution/operator/join/join_filter_pushdown.hpp>
#include <duckdb/function/table_function.hpp>
#include <duckdb/planner/expression/bound_cast_expression.hpp>
#include <duckdb/planner/expression/bound_columnref_expression.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <planner/dynamic_filter_candidate_cache.hpp>
#include <planner/sirius_physical_plan_generator.hpp>
#include <sirius/exception.hpp>

#include <memory>
#include <utility>

using sirius::planner::duckdb_candidate_kind;
using sirius::planner::dynamic_filter_candidate_cache;

namespace {

/// The expression shape of one side of a join condition, chosen per fence case.
enum class cond_shape {
  pre_resolver,        ///< BOUND_COLUMN_REF: the only shape the generator may receive (C1a-2).
  post_resolver,       ///< BOUND_REF: the unambiguous post-ColumnBindingResolver shape.
  constant,            ///< No reference at all: the documented undetectable corner.
  cast_of_column_ref,  ///< BOUND_CAST wrapping BOUND_COLUMN_REF: pre-resolver, must not fence.
};

duckdb::unique_ptr<duckdb::Expression> make_side(cond_shape shape,
                                                 [[maybe_unused]] duckdb::ClientContext& context)
{
  switch (shape) {
    case cond_shape::pre_resolver:
      return duckdb::make_uniq<duckdb::BoundColumnRefExpression>(duckdb::LogicalType::INTEGER,
                                                                 duckdb::ColumnBinding{1, 0});
    case cond_shape::post_resolver:
      return duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::INTEGER,
                                                                 0ULL);
    case cond_shape::constant:
      return duckdb::make_uniq<duckdb::BoundConstantExpression>(duckdb::Value::INTEGER(42));
    case cond_shape::cast_of_column_ref: {
      auto child = duckdb::make_uniq<duckdb::BoundColumnRefExpression>(duckdb::LogicalType::INTEGER,
                                                                       duckdb::ColumnBinding{1, 0});
      return duckdb::BoundCastExpression::AddDefaultCastToType(std::move(child),
                                                               duckdb::LogicalType::BIGINT);
    }
  }
  return nullptr;  // unreachable
}

duckdb::JoinCondition make_condition(cond_shape shape, duckdb::ClientContext& context)
{
  duckdb::JoinCondition cond;
  cond.left       = make_side(shape, context);
  cond.right      = make_side(shape, context);
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
  cond_shape shape,
  duckdb::ClientContext& context,
  bool with_pushdown,
  duckdb::LogicalOperatorType logical_type = duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN)
{
  auto join =
    duckdb::make_uniq<duckdb::LogicalComparisonJoin>(duckdb::JoinType::INNER, logical_type);
  join->conditions.push_back(make_condition(shape, context));
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

/// Shared in-memory database: the cache takes a ClientContext& (reserved for C1b's domain
/// snapshot) and the generator-level fence case constructs a real generator from one.
struct test_context {
  duckdb::DuckDB db{nullptr};
  duckdb::Connection con{db};
  duckdb::ClientContext& context() { return *con.context; }
};

}  // namespace

//===-----------------------------------------------------------------------------------------===//
// Single-use protocol
//===-----------------------------------------------------------------------------------------===//

TEST_CASE("cache rejects a second capture_pre_resolver", "[dynamic_filter][candidate_cache]")
{
  test_context tc;
  auto join = make_join(cond_shape::pre_resolver, tc.context(), /*with_pushdown=*/false);

  dynamic_filter_candidate_cache cache;
  cache.capture_pre_resolver(*join, tc.context());
  REQUIRE(cache.capture_completed());
  REQUIRE_THROWS_AS(cache.capture_pre_resolver(*join, tc.context()), sirius::internal_exception);
}

TEST_CASE("cache rejects extract_post_resolver before capture_pre_resolver",
          "[dynamic_filter][candidate_cache]")
{
  test_context tc;
  auto join = make_join(cond_shape::pre_resolver, tc.context(), /*with_pushdown=*/false);

  dynamic_filter_candidate_cache cache;
  REQUIRE_THROWS_AS(cache.extract_post_resolver(*join), sirius::internal_exception);
  REQUIRE_FALSE(cache.extraction_completed());
}

TEST_CASE("cache rejects a second extract_post_resolver", "[dynamic_filter][candidate_cache]")
{
  test_context tc;
  auto join = make_join(cond_shape::pre_resolver, tc.context(), /*with_pushdown=*/false);

  dynamic_filter_candidate_cache cache;
  cache.capture_pre_resolver(*join, tc.context());
  cache.extract_post_resolver(*join);
  REQUIRE(cache.extraction_completed());
  REQUIRE_THROWS_AS(cache.extract_post_resolver(*join), sirius::internal_exception);
}

//===-----------------------------------------------------------------------------------------===//
// find() tri-state and entry sharing
//===-----------------------------------------------------------------------------------------===//

TEST_CASE("find is null for a join the cache never saw", "[dynamic_filter][candidate_cache]")
{
  test_context tc;
  auto captured = make_join(cond_shape::pre_resolver, tc.context(), /*with_pushdown=*/false);
  auto stranger = make_join(cond_shape::pre_resolver, tc.context(), /*with_pushdown=*/false);

  dynamic_filter_candidate_cache cache;
  REQUIRE(cache.find(*captured) == nullptr);  // nothing captured at all

  cache.capture_pre_resolver(*captured, tc.context());
  cache.extract_post_resolver(*captured);
  REQUIRE(cache.find(*stranger) == nullptr);  // not in the extracted tree
}

TEST_CASE("find is null after capture but before extraction", "[dynamic_filter][candidate_cache]")
{
  test_context tc;
  auto join = make_join(cond_shape::pre_resolver, tc.context(), /*with_pushdown=*/true);

  dynamic_filter_candidate_cache cache;
  cache.capture_pre_resolver(*join, tc.context());
  REQUIRE(cache.find(*join) == nullptr);  // node identity recorded, candidate not yet extracted
}

TEST_CASE("find returns a non-null absent candidate for a pushdown-free join",
          "[dynamic_filter][candidate_cache]")
{
  test_context tc;
  auto join = make_join(cond_shape::pre_resolver, tc.context(), /*with_pushdown=*/false);

  dynamic_filter_candidate_cache cache;
  cache.capture_pre_resolver(*join, tc.context());
  cache.extract_post_resolver(*join);

  auto candidate = cache.find(*join);
  REQUIRE(candidate != nullptr);  // "no metadata" is a cached answer, not a cache miss
  REQUIRE(candidate->kind == duckdb_candidate_kind::absent);
}

TEST_CASE("repeated find returns the one shared immutable entry",
          "[dynamic_filter][candidate_cache]")
{
  test_context tc;
  auto join = make_join(cond_shape::pre_resolver, tc.context(), /*with_pushdown=*/true);

  dynamic_filter_candidate_cache cache;
  cache.capture_pre_resolver(*join, tc.context());
  cache.extract_post_resolver(*join);

  auto first  = cache.find(*join);
  auto second = cache.find(*join);
  REQUIRE(first != nullptr);
  // Same object, not an equal copy: extraction ran exactly once and every consumer (C3 discovery,
  // plan_comparison_join) shares it non-destructively.
  REQUIRE(first.get() == second.get());
  REQUIRE(first->kind == duckdb_candidate_kind::admitted);
  REQUIRE(first->condition_indexes == std::vector<std::size_t>{0});
}

TEST_CASE("DELIM joins share the plan_comparison_join entry space",
          "[dynamic_filter][candidate_cache]")
{
  test_context tc;
  // A DELIM join carrying pushdown, with a plain comparison join below it: both plan through
  // plan_comparison_join, so both must be captured and extractable.
  auto delim      = make_join(cond_shape::pre_resolver,
                         tc.context(),
                         /*with_pushdown=*/true,
                         duckdb::LogicalOperatorType::LOGICAL_DELIM_JOIN);
  auto inner      = make_join(cond_shape::pre_resolver, tc.context(), /*with_pushdown=*/false);
  auto* inner_raw = inner.get();
  delim->children.push_back(std::move(inner));
  delim->children.push_back(make_get());

  dynamic_filter_candidate_cache cache;
  cache.capture_pre_resolver(*delim, tc.context());
  cache.extract_post_resolver(*delim);

  auto delim_candidate = cache.find(*delim);
  REQUIRE(delim_candidate != nullptr);
  REQUIRE(delim_candidate->kind == duckdb_candidate_kind::admitted);

  auto inner_candidate = cache.find(*inner_raw);
  REQUIRE(inner_candidate != nullptr);
  REQUIRE(inner_candidate->kind == duckdb_candidate_kind::absent);
}

//===-----------------------------------------------------------------------------------------===//
// The resolver fence
//===-----------------------------------------------------------------------------------------===//

TEST_CASE("an already-resolved pushdown-bearing plan trips the fence",
          "[dynamic_filter][candidate_cache]")
{
  test_context tc;
  auto join = make_join(cond_shape::post_resolver, tc.context(), /*with_pushdown=*/true);

  dynamic_filter_candidate_cache cache;
#ifdef DEBUG
  // Debug flavor: fail loudly — some upstream path resolved before the generator.
  REQUIRE_THROWS_AS(cache.capture_pre_resolver(*join, tc.context()), sirius::internal_exception);
  WARN("release-flavor fence semantics not covered in this build; the C1a-2 gate runs both");
#else
  // Release flavor: record the signal and proceed — extraction stays positionally correct.
  cache.capture_pre_resolver(*join, tc.context());
  REQUIRE(cache.saw_resolved_plan());
  cache.extract_post_resolver(*join);

  auto candidate = cache.find(*join);
  REQUIRE(candidate != nullptr);
  REQUIRE(candidate->kind == duckdb_candidate_kind::admitted);
  REQUIRE(candidate->condition_indexes == std::vector<std::size_t>{0});
  REQUIRE(candidate->targets.size() == 1);
  WARN("debug-flavor fence throw not covered in this build; the C1a-2 gate runs both");
#endif
}

TEST_CASE("no fence for a resolved plan without pushdown", "[dynamic_filter][candidate_cache]")
{
  test_context tc;
  auto join = make_join(cond_shape::post_resolver, tc.context(), /*with_pushdown=*/false);

  dynamic_filter_candidate_cache cache;
  cache.capture_pre_resolver(*join, tc.context());  // must not throw in any build flavor
  REQUIRE_FALSE(cache.saw_resolved_plan());
}

TEST_CASE("no fence for BOUND_CAST-wrapped pre-resolver keys", "[dynamic_filter][candidate_cache]")
{
  test_context tc;
  auto join = make_join(cond_shape::cast_of_column_ref, tc.context(), /*with_pushdown=*/true);

  dynamic_filter_candidate_cache cache;
  // CAST(BOUND_COLUMN_REF) is still pre-resolver form: the column ref inside the cast must be
  // found by the recursive classifier, so this plan is NOT "already resolved".
  cache.capture_pre_resolver(*join, tc.context());
  REQUIRE_FALSE(cache.saw_resolved_plan());
}

TEST_CASE("no fence for constants-only conditions", "[dynamic_filter][candidate_cache]")
{
  test_context tc;
  auto join = make_join(cond_shape::constant, tc.context(), /*with_pushdown=*/true);

  dynamic_filter_candidate_cache cache;
  // Neither BOUND_REF nor BOUND_COLUMN_REF exists, so resolver state is undetectable and the
  // fence must stay quiet (the documented corner: saw_resolved_plan()==false is not a guarantee
  // of pre-resolver form here — such joins have no key columns to snapshot anyway).
  cache.capture_pre_resolver(*join, tc.context());
  REQUIRE_FALSE(cache.saw_resolved_plan());
}

TEST_CASE("mixed tree fences on the one resolved pushdown-bearing join",
          "[dynamic_filter][candidate_cache]")
{
  test_context tc;
  // Root join is clean pre-resolver; its child join carries pushdown and is already resolved.
  // The fence signal must come from the child — proving the walk inspects every join, not just
  // the root.
  auto root  = make_join(cond_shape::pre_resolver, tc.context(), /*with_pushdown=*/false);
  auto child = make_join(cond_shape::post_resolver, tc.context(), /*with_pushdown=*/true);
  root->children.push_back(std::move(child));
  root->children.push_back(make_get());

  dynamic_filter_candidate_cache cache;
#ifdef DEBUG
  REQUIRE_THROWS_AS(cache.capture_pre_resolver(*root, tc.context()), sirius::internal_exception);
#else
  cache.capture_pre_resolver(*root, tc.context());
  REQUIRE(cache.saw_resolved_plan());
#endif
}

//===-----------------------------------------------------------------------------------------===//
// The generator-level fence (the silently-vacated-coverage regression class)
//===-----------------------------------------------------------------------------------------===//

TEST_CASE("a pre-resolving harness fails loudly at create_plan in debug builds",
          "[dynamic_filter][candidate_cache]")
{
#ifdef DEBUG
  // The regression class this guards: a test helper (or new entry path) that resolves column
  // bindings before create_plan(unique_ptr) would make the pre-resolver snapshot impossible and
  // silently vacate coverage if the failure were swallowed. In debug builds the capture fence
  // must surface it at the generator boundary.
  test_context tc;
  duckdb::unique_ptr<duckdb::LogicalOperator> plan =
    make_join(cond_shape::post_resolver, tc.context(), /*with_pushdown=*/true);
  plan->children.push_back(make_get());
  plan->children.push_back(make_get());

  sirius::planner::sirius_physical_plan_generator generator(tc.context());
  REQUIRE_THROWS_AS(generator.create_plan(std::move(plan)), sirius::internal_exception);
#else
  WARN(
    "Skipped: the generator-level fence exists only in debug builds (release records "
    "saw_resolved_plan instead); the C1a-2 gate runs the debug flavor separately");
#endif
}
