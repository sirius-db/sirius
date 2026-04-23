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

// Tests for sirius::expression — the opaque wrapper around duckdb::Expression
// that lets Super Sirius operator headers stay free of duckdb/planner/expression/ includes.

#include "catch.hpp"
#include "expression/expression.hpp"
#include "expression/expression_internal.hpp"

#include <duckdb/common/types/value.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>

#include <memory>
#include <type_traits>
#include <utility>
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

TEST_CASE("expression - wrap preserves pointer identity via unwrap", "[expression]")
{
  auto raw      = make_const(42);
  auto* raw_ptr = raw.get();

  expression e = sirius::wrap(std::move(raw));
  REQUIRE_FALSE(e.is_null());
  REQUIRE(static_cast<bool>(e));
  REQUIRE(sirius::unwrap(e) == raw_ptr);
}

TEST_CASE("expression - wrap preserves underlying expression state", "[expression]")
{
  expression e = sirius::wrap(make_const(42));

  auto const* casted = dynamic_cast<duckdb::BoundConstantExpression const*>(sirius::unwrap(e));
  REQUIRE(casted != nullptr);
  REQUIRE(casted->value == duckdb::Value::INTEGER(42));
}

TEST_CASE("expression - const unwrap overload returns pointer-to-const", "[expression]")
{
  auto raw      = make_const(5);
  auto* raw_ptr = raw.get();

  expression mutable_e = sirius::wrap(std::move(raw));
  expression const& ce = mutable_e;

  duckdb::Expression const* p = sirius::unwrap(ce);  // picks const overload
  REQUIRE(p == raw_ptr);
}

// ============================================================================
// Move semantics
// ============================================================================

TEST_CASE("expression - move-construction nulls the source", "[expression]")
{
  auto raw      = make_const(7);
  auto* raw_ptr = raw.get();

  expression a = sirius::wrap(std::move(raw));
  expression b{std::move(a)};

  REQUIRE(a.is_null());
  REQUIRE_FALSE(b.is_null());
  REQUIRE(sirius::unwrap(b) == raw_ptr);
}

TEST_CASE("expression - move-assignment nulls the source", "[expression]")
{
  auto raw      = make_const(11);
  auto* raw_ptr = raw.get();

  expression a = sirius::wrap(std::move(raw));
  expression b;
  b = std::move(a);

  REQUIRE(a.is_null());
  REQUIRE_FALSE(b.is_null());
  REQUIRE(sirius::unwrap(b) == raw_ptr);
}

// ============================================================================
// release()
// ============================================================================

TEST_CASE("expression - release transfers ownership and nulls the source", "[expression]")
{
  auto raw      = make_const(99);
  auto* raw_ptr = raw.get();

  expression e                             = sirius::wrap(std::move(raw));
  std::unique_ptr<duckdb::Expression> back = sirius::release(e);

  REQUIRE(e.is_null());
  REQUIRE(back.get() == raw_ptr);
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

TEST_CASE("expression - wrap_many preserves size, order, and per-element identity", "[expression]")
{
  std::vector<std::unique_ptr<duckdb::Expression>> input;
  std::vector<duckdb::Expression const*> originals;
  for (int32_t i = 0; i < 4; ++i) {
    auto e = make_const(i);
    originals.push_back(e.get());
    input.push_back(std::move(e));
  }

  auto wrapped = sirius::wrap_many(std::move(input));

  REQUIRE(wrapped.size() == originals.size());
  for (std::size_t i = 0; i < wrapped.size(); ++i) {
    REQUIRE(sirius::unwrap(wrapped[i]) == originals[i]);
  }
}

TEST_CASE("expression - wrap_many on empty input returns empty vector", "[expression]")
{
  auto wrapped = sirius::wrap_many(std::vector<std::unique_ptr<duckdb::Expression>>{});
  REQUIRE(wrapped.empty());
}

TEST_CASE("expression - wrap_many handles mixed null and non-null slots", "[expression]")
{
  auto a      = make_const(1);
  auto c      = make_const(3);
  auto* a_ptr = a.get();
  auto* c_ptr = c.get();

  std::vector<std::unique_ptr<duckdb::Expression>> input;
  input.push_back(std::move(a));
  input.push_back(nullptr);
  input.push_back(std::move(c));

  auto wrapped = sirius::wrap_many(std::move(input));

  REQUIRE(wrapped.size() == 3);
  REQUIRE(sirius::unwrap(wrapped[0]) == a_ptr);
  REQUIRE(wrapped[1].is_null());
  REQUIRE(sirius::unwrap(wrapped[2]) == c_ptr);
}
