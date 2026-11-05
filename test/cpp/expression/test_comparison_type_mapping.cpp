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

// Tests for the surviving sirius::join_condition surface: the
// duckdb::ExpressionType <-> sirius::comparison_type mapping helpers and the
// wrap_join_conditions boundary builder (which now produces AST-node sides).

#include "catch.hpp"
#include "expression/ast/constant.hpp"
#include "expression/ast/node.hpp"
#include "expression/join_condition.hpp"

#include <duckdb/common/enums/expression_type.hpp>
#include <duckdb/common/types/value.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/joinside.hpp>

#include <memory>
#include <utility>
#include <variant>
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
  return duckdb::JoinCondition(std::move(left), std::move(right), comparison);
}

// Assert the AST node is an INTEGER constant equal to `expected`.
void require_int_constant(sirius::ast::node const* n, int32_t expected)
{
  REQUIRE(n != nullptr);
  REQUIRE(n->holds<sirius::ast::constant>());
  auto const& c = n->get<sirius::ast::constant>();
  REQUIRE(std::holds_alternative<int32_t>(c.payload));
  REQUIRE(std::get<int32_t>(c.payload) == expected);
}

}  // namespace

// ============================================================================
// from_duckdb / to_duckdb round-trip
// ============================================================================

TEST_CASE("comparison_type - to_duckdb then from_duckdb round-trips every comparison_type",
          "[join_condition]")
{
  for (auto c : k_all_comparisons) {
    REQUIRE(sirius::from_duckdb(sirius::to_duckdb(c)) == c);
  }
}

TEST_CASE("comparison_type - from_duckdb throws on an unsupported ExpressionType",
          "[join_condition]")
{
  REQUIRE_THROWS_AS(sirius::from_duckdb(duckdb::ExpressionType::COMPARE_IN), std::runtime_error);
}

// ============================================================================
// wrap_join_conditions
// ============================================================================

TEST_CASE("wrap_join_conditions transfers expressions and maps comparisons", "[join_condition]")
{
  std::vector<duckdb::JoinCondition> input;

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
    input.push_back(make_cond(make_const(10), make_const(20), duckdb_type));
  }

  auto wrapped = sirius::wrap_join_conditions(std::move(input));

  REQUIRE(wrapped.size() == std::size(entries));
  for (std::size_t i = 0; i < wrapped.size(); ++i) {
    require_int_constant(wrapped[i].left.get(), 10);
    require_int_constant(wrapped[i].right.get(), 20);
    REQUIRE(wrapped[i].comparison == entries[i].expected_out);
  }
}

TEST_CASE("wrap_join_conditions on empty input returns empty vector", "[join_condition]")
{
  auto out = sirius::wrap_join_conditions(std::vector<duckdb::JoinCondition>{});
  REQUIRE(out.empty());
}

TEST_CASE("wrap_join_conditions propagates from_duckdb throws", "[join_condition]")
{
  std::vector<duckdb::JoinCondition> input;
  input.push_back(make_cond(make_const(1), make_const(2), duckdb::ExpressionType::COMPARE_IN));

  REQUIRE_THROWS_AS(sirius::wrap_join_conditions(std::move(input)), std::runtime_error);
}
