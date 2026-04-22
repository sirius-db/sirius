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

// Tests for sirius::join_condition and the duckdb::ExpressionType <-> sirius::comparison_type
// mapping helpers.

#include "catch.hpp"
#include "expression/expression_internal.hpp"
#include "expression/join_condition.hpp"

#include <duckdb/common/enums/expression_type.hpp>
#include <duckdb/common/types/value.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/joinside.hpp>

#include <memory>
#include <utility>
#include <vector>

using sirius::comparison_type;
using sirius::join_condition;

// ============================================================================
// Helpers
// ============================================================================

namespace {

constexpr comparison_type k_all_comparisons[] = {
  comparison_type::equal,
  comparison_type::not_equal,
  comparison_type::lt,
  comparison_type::le,
  comparison_type::gt,
  comparison_type::ge,
  comparison_type::distinct_from,
  comparison_type::not_distinct_from,
};

duckdb::unique_ptr<duckdb::Expression> make_const(int32_t v)
{
  return duckdb::make_uniq<duckdb::BoundConstantExpression>(duckdb::Value::INTEGER(v));
}

duckdb::JoinCondition make_cond(duckdb::unique_ptr<duckdb::Expression> left,
                                duckdb::unique_ptr<duckdb::Expression> right,
                                duckdb::ExpressionType comparison)
{
  duckdb::JoinCondition c;
  c.left       = std::move(left);
  c.right      = std::move(right);
  c.comparison = comparison;
  return c;
}

}  // namespace

// ============================================================================
// from_duckdb / to_duckdb round-trip
// ============================================================================

TEST_CASE("join_condition - to_duckdb then from_duckdb round-trips every comparison_type",
          "[join_condition]")
{
  for (auto c : k_all_comparisons) {
    REQUIRE(sirius::from_duckdb(sirius::to_duckdb(c)) == c);
  }
}

TEST_CASE("join_condition - from_duckdb throws on an unsupported ExpressionType",
          "[join_condition]")
{
  REQUIRE_THROWS_AS(sirius::from_duckdb(duckdb::ExpressionType::COMPARE_IN), std::runtime_error);
}

// ============================================================================
// wrap_join_conditions
// ============================================================================

TEST_CASE("join_condition - wrap_join_conditions transfers expressions and maps comparisons",
          "[join_condition]")
{
  std::vector<duckdb::JoinCondition> input;
  std::vector<duckdb::Expression const*> left_ptrs;
  std::vector<duckdb::Expression const*> right_ptrs;

  struct entry {
    duckdb::ExpressionType in;
    comparison_type expected_out;
  };
  constexpr entry entries[] = {
    {duckdb::ExpressionType::COMPARE_EQUAL, comparison_type::equal},
    {duckdb::ExpressionType::COMPARE_LESSTHAN, comparison_type::lt},
    {duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM, comparison_type::not_distinct_from},
  };

  for (auto const& [duckdb_type, _] : entries) {
    auto l = make_const(10);
    auto r = make_const(20);
    left_ptrs.push_back(l.get());
    right_ptrs.push_back(r.get());
    input.push_back(make_cond(std::move(l), std::move(r), duckdb_type));
  }

  auto wrapped = sirius::wrap_join_conditions(std::move(input));

  REQUIRE(wrapped.size() == std::size(entries));
  for (std::size_t i = 0; i < wrapped.size(); ++i) {
    REQUIRE(sirius::unwrap(wrapped[i].left) == left_ptrs[i]);
    REQUIRE(sirius::unwrap(wrapped[i].right) == right_ptrs[i]);
    REQUIRE(wrapped[i].comparison == entries[i].expected_out);
  }
}

TEST_CASE("join_condition - wrap_join_conditions on empty input returns empty vector",
          "[join_condition]")
{
  auto out = sirius::wrap_join_conditions({});
  REQUIRE(out.empty());
}

TEST_CASE("join_condition - wrap_join_conditions propagates from_duckdb throws", "[join_condition]")
{
  std::vector<duckdb::JoinCondition> input;
  input.push_back(make_cond(make_const(1), make_const(2), duckdb::ExpressionType::COMPARE_IN));

  REQUIRE_THROWS_AS(sirius::wrap_join_conditions(std::move(input)), std::runtime_error);
}
