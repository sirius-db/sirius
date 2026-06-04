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

// Tests for sirius::expression — the opaque wrapper that now holds a Sirius AST
// node. wrap() is the DuckDB->Sirius translation boundary (via ast::from_duckdb),
// so these tests assert AST node structure/values rather than duckdb pointer
// identity (which the backing-type flip intentionally destroys). See
// https://github.com/sirius-db/sirius/issues/701.

#include "catch.hpp"
#include "expression/ast/constant.hpp"
#include "expression/ast/node.hpp"
#include "expression/expression.hpp"
#include "expression/expression_internal.hpp"

#include <duckdb/common/types/value.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>

#include <memory>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

using sirius::expression;

// ============================================================================
// Compile-time invariants
// ============================================================================

static_assert(!std::is_copy_constructible_v<expression>, "sirius::expression must be move-only");
static_assert(!std::is_copy_assignable_v<expression>, "sirius::expression must be move-only");
static_assert(std::is_nothrow_move_constructible_v<expression>,
              "sirius::expression move-constructor must be noexcept");
static_assert(std::is_nothrow_move_assignable_v<expression>,
              "sirius::expression move-assignment must be noexcept");

// ============================================================================
// Helpers
// ============================================================================

namespace {

std::unique_ptr<duckdb::Expression> make_const(int32_t v)
{
  return std::make_unique<duckdb::BoundConstantExpression>(duckdb::Value::INTEGER(v));
}

// Assert the wrapped node is an INTEGER constant equal to `expected`.
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
// Construction and null state
// ============================================================================

TEST_CASE("expression - default-constructed is null", "[expression]")
{
  expression e;
  REQUIRE(e.is_null());
  REQUIRE_FALSE(static_cast<bool>(e));
  REQUIRE(sirius::unwrap(e) == nullptr);
}

TEST_CASE("expression - wrap(nullptr) yields a null expression", "[expression]")
{
  expression e = sirius::wrap(nullptr);
  REQUIRE(e.is_null());
  REQUIRE(sirius::unwrap(e) == nullptr);
}

// ============================================================================
// wrap / unwrap round-trip
// ============================================================================

TEST_CASE("expression - wrap translates to a Sirius AST node", "[expression]")
{
  expression e = sirius::wrap(make_const(42));
  REQUIRE_FALSE(e.is_null());
  REQUIRE(static_cast<bool>(e));
  require_int_constant(sirius::unwrap(e), 42);
}

TEST_CASE("expression - wrap preserves the underlying constant value", "[expression]")
{
  expression e = sirius::wrap(make_const(42));
  require_int_constant(sirius::unwrap(e), 42);
}

TEST_CASE("expression - const unwrap overload returns pointer-to-const", "[expression]")
{
  expression mutable_e = sirius::wrap(make_const(5));
  expression const& ce = mutable_e;

  sirius::ast::node const* p = sirius::unwrap(ce);  // picks const overload
  require_int_constant(p, 5);
}

// ============================================================================
// Move semantics
// ============================================================================

TEST_CASE("expression - move-construction nulls the source", "[expression]")
{
  expression a = sirius::wrap(make_const(7));
  expression b{std::move(a)};

  REQUIRE(a.is_null());
  REQUIRE_FALSE(b.is_null());
  require_int_constant(sirius::unwrap(b), 7);
}

TEST_CASE("expression - move-assignment nulls the source", "[expression]")
{
  expression a = sirius::wrap(make_const(11));
  expression b;
  b = std::move(a);

  REQUIRE(a.is_null());
  REQUIRE_FALSE(b.is_null());
  require_int_constant(sirius::unwrap(b), 11);
}

// ============================================================================
// release()
// ============================================================================

TEST_CASE("expression - release transfers ownership and nulls the source", "[expression]")
{
  expression e                            = sirius::wrap(make_const(99));
  std::unique_ptr<sirius::ast::node> back = sirius::release(e);

  REQUIRE(e.is_null());
  require_int_constant(back.get(), 99);
}

TEST_CASE("expression - release on a null expression returns nullptr", "[expression]")
{
  expression e;
  REQUIRE(sirius::release(e) == nullptr);
  REQUIRE(e.is_null());
}

TEST_CASE("expression - double release is safe", "[expression]")
{
  expression e = sirius::wrap(make_const(3));

  auto first = sirius::release(e);
  REQUIRE(first != nullptr);
  REQUIRE(e.is_null());

  auto second = sirius::release(e);
  REQUIRE(second == nullptr);
  REQUIRE(e.is_null());
}

// ============================================================================
// wrap_many
// ============================================================================

TEST_CASE("expression - wrap_many preserves size and order", "[expression]")
{
  std::vector<std::unique_ptr<duckdb::Expression>> input;
  for (int32_t i = 0; i < 4; ++i) {
    input.push_back(make_const(i));
  }

  auto wrapped = sirius::wrap_many(std::move(input));

  REQUIRE(wrapped.size() == 4);
  for (int32_t i = 0; i < 4; ++i) {
    require_int_constant(sirius::unwrap(wrapped[static_cast<std::size_t>(i)]), i);
  }
}

TEST_CASE("expression - wrap_many on empty input returns empty vector", "[expression]")
{
  auto wrapped = sirius::wrap_many(std::vector<std::unique_ptr<duckdb::Expression>>{});
  REQUIRE(wrapped.empty());
}

TEST_CASE("expression - wrap_many handles mixed null and non-null slots", "[expression]")
{
  std::vector<std::unique_ptr<duckdb::Expression>> input;
  input.push_back(make_const(1));
  input.push_back(nullptr);
  input.push_back(make_const(3));

  auto wrapped = sirius::wrap_many(std::move(input));

  REQUIRE(wrapped.size() == 3);
  require_int_constant(sirius::unwrap(wrapped[0]), 1);
  REQUIRE(wrapped[1].is_null());
  require_int_constant(sirius::unwrap(wrapped[2]), 3);
}
