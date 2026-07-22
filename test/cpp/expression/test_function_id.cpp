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

// Tests for sirius::function_id — the closed scalar-function enum + bidirectional
// DuckDB-name mappers.
//
// Round-trips every supported function_id through to_duckdb_function_name /
// from_duckdb_function_name, verifies the substring/substr alias collapse
// (D-SUB-1), the std::nullopt return on unknown names, and locks the 27-entry
// ABI cardinality at compile time.

#include "catch.hpp"
#include "expression/function_id.hpp"

#include <cstdint>
#include <optional>
#include <string_view>
#include <type_traits>

using sirius::from_duckdb_function_name;
using sirius::function_id;
using sirius::to_duckdb_function_name;

// ============================================================================
// Compile-time invariants (D-01 — locked ABI)
// ============================================================================

static_assert(std::is_enum_v<function_id>, "sirius::function_id must be an enum class.");
static_assert(sizeof(function_id) == 2,
              "sirius::function_id is uint16_t-backed (D-01 — locked ABI).");
static_assert(static_cast<uint16_t>(function_id::error) + 1 == 28,
              "sirius::function_id has exactly 28 entries (D-01 — locked ABI).");

// ============================================================================
// Round-trip every function_id entry through the name mappers
// ============================================================================

TEST_CASE("ast_function_id - add round-trips through name mappers", "[ast_function_id]")
{
  auto const name = to_duckdb_function_name(function_id::add);
  REQUIRE(name == "+");
  auto const id = from_duckdb_function_name(name);
  REQUIRE(id.has_value());
  REQUIRE(*id == function_id::add);
}

TEST_CASE("ast_function_id - sub round-trips through name mappers", "[ast_function_id]")
{
  auto const name = to_duckdb_function_name(function_id::sub);
  REQUIRE(name == "-");
  auto const id = from_duckdb_function_name(name);
  REQUIRE(id.has_value());
  REQUIRE(*id == function_id::sub);
}

TEST_CASE("ast_function_id - mul round-trips through name mappers", "[ast_function_id]")
{
  auto const name = to_duckdb_function_name(function_id::mul);
  REQUIRE(name == "*");
  auto const id = from_duckdb_function_name(name);
  REQUIRE(id.has_value());
  REQUIRE(*id == function_id::mul);
}

TEST_CASE("ast_function_id - div round-trips through name mappers", "[ast_function_id]")
{
  auto const name = to_duckdb_function_name(function_id::div);
  REQUIRE(name == "/");
  auto const id = from_duckdb_function_name(name);
  REQUIRE(id.has_value());
  REQUIRE(*id == function_id::div);
}

TEST_CASE("ast_function_id - int_div round-trips through name mappers", "[ast_function_id]")
{
  auto const name = to_duckdb_function_name(function_id::int_div);
  REQUIRE(name == "//");
  auto const id = from_duckdb_function_name(name);
  REQUIRE(id.has_value());
  REQUIRE(*id == function_id::int_div);
}

TEST_CASE("ast_function_id - mod round-trips through name mappers", "[ast_function_id]")
{
  auto const name = to_duckdb_function_name(function_id::mod);
  REQUIRE(name == "%");
  auto const id = from_duckdb_function_name(name);
  REQUIRE(id.has_value());
  REQUIRE(*id == function_id::mod);
}

TEST_CASE("ast_function_id - Substrait arithmetic names resolve to arithmetic ids",
          "[ast_function_id]")
{
  REQUIRE(from_duckdb_function_name("add") == function_id::add);
  REQUIRE(from_duckdb_function_name("subtract") == function_id::sub);
  REQUIRE(from_duckdb_function_name("multiply") == function_id::mul);
  REQUIRE(from_duckdb_function_name("divide") == function_id::div);
  REQUIRE(from_duckdb_function_name("modulus") == function_id::mod);
}

TEST_CASE("ast_function_id - substring round-trips through name mappers", "[ast_function_id]")
{
  auto const name = to_duckdb_function_name(function_id::substring);
  REQUIRE(name == "substring");
  auto const id = from_duckdb_function_name(name);
  REQUIRE(id.has_value());
  REQUIRE(*id == function_id::substring);
}

TEST_CASE("ast_function_id - like round-trips through name mappers", "[ast_function_id]")
{
  auto const name = to_duckdb_function_name(function_id::like);
  REQUIRE(name == "~~");
  auto const id = from_duckdb_function_name(name);
  REQUIRE(id.has_value());
  REQUIRE(*id == function_id::like);
}

TEST_CASE("ast_function_id - not_like round-trips through name mappers", "[ast_function_id]")
{
  auto const name = to_duckdb_function_name(function_id::not_like);
  REQUIRE(name == "!~~");
  auto const id = from_duckdb_function_name(name);
  REQUIRE(id.has_value());
  REQUIRE(*id == function_id::not_like);
}

TEST_CASE("ast_function_id - contains round-trips through name mappers", "[ast_function_id]")
{
  auto const name = to_duckdb_function_name(function_id::contains);
  REQUIRE(name == "contains");
  auto const id = from_duckdb_function_name(name);
  REQUIRE(id.has_value());
  REQUIRE(*id == function_id::contains);
}

TEST_CASE("ast_function_id - prefix round-trips through name mappers", "[ast_function_id]")
{
  auto const name = to_duckdb_function_name(function_id::prefix);
  REQUIRE(name == "prefix");
  auto const id = from_duckdb_function_name(name);
  REQUIRE(id.has_value());
  REQUIRE(*id == function_id::prefix);
}

TEST_CASE("ast_function_id - suffix round-trips through name mappers", "[ast_function_id]")
{
  auto const name = to_duckdb_function_name(function_id::suffix);
  REQUIRE(name == "suffix");
  auto const id = from_duckdb_function_name(name);
  REQUIRE(id.has_value());
  REQUIRE(*id == function_id::suffix);
}

TEST_CASE("ast_function_id - strlen round-trips through name mappers", "[ast_function_id]")
{
  auto const name = to_duckdb_function_name(function_id::strlen);
  REQUIRE(name == "strlen");
  auto const id = from_duckdb_function_name(name);
  REQUIRE(id.has_value());
  REQUIRE(*id == function_id::strlen);
}

TEST_CASE("ast_function_id - length round-trips through name mappers", "[ast_function_id]")
{
  auto const name = to_duckdb_function_name(function_id::length);
  REQUIRE(name == "length");
  auto const id = from_duckdb_function_name(name);
  REQUIRE(id.has_value());
  REQUIRE(*id == function_id::length);
}

TEST_CASE("ast_function_id - regexp_replace round-trips through name mappers", "[ast_function_id]")
{
  auto const name = to_duckdb_function_name(function_id::regexp_replace);
  REQUIRE(name == "regexp_replace");
  auto const id = from_duckdb_function_name(name);
  REQUIRE(id.has_value());
  REQUIRE(*id == function_id::regexp_replace);
}

TEST_CASE("ast_function_id - concat round-trips through name mappers", "[ast_function_id]")
{
  auto const name = to_duckdb_function_name(function_id::concat);
  REQUIRE(name == "concat");
  auto const id = from_duckdb_function_name(name);
  REQUIRE(id.has_value());
  REQUIRE(*id == function_id::concat);
}

TEST_CASE("ast_function_id - concat and || both resolve to function_id::concat",
          "[ast_function_id]")
{
  auto const id_fn = from_duckdb_function_name("concat");
  auto const id_op = from_duckdb_function_name("||");
  REQUIRE(id_fn.has_value());
  REQUIRE(id_op.has_value());
  REQUIRE(*id_fn == function_id::concat);
  REQUIRE(*id_op == function_id::concat);
}

TEST_CASE("ast_function_id - year round-trips through name mappers", "[ast_function_id]")
{
  auto const name = to_duckdb_function_name(function_id::year);
  REQUIRE(name == "year");
  auto const id = from_duckdb_function_name(name);
  REQUIRE(id.has_value());
  REQUIRE(*id == function_id::year);
}

TEST_CASE("ast_function_id - month round-trips through name mappers", "[ast_function_id]")
{
  auto const name = to_duckdb_function_name(function_id::month);
  REQUIRE(name == "month");
  auto const id = from_duckdb_function_name(name);
  REQUIRE(id.has_value());
  REQUIRE(*id == function_id::month);
}

TEST_CASE("ast_function_id - day round-trips through name mappers", "[ast_function_id]")
{
  auto const name = to_duckdb_function_name(function_id::day);
  REQUIRE(name == "day");
  auto const id = from_duckdb_function_name(name);
  REQUIRE(id.has_value());
  REQUIRE(*id == function_id::day);
}

TEST_CASE("ast_function_id - hour round-trips through name mappers", "[ast_function_id]")
{
  auto const name = to_duckdb_function_name(function_id::hour);
  REQUIRE(name == "hour");
  auto const id = from_duckdb_function_name(name);
  REQUIRE(id.has_value());
  REQUIRE(*id == function_id::hour);
}

TEST_CASE("ast_function_id - minute round-trips through name mappers", "[ast_function_id]")
{
  auto const name = to_duckdb_function_name(function_id::minute);
  REQUIRE(name == "minute");
  auto const id = from_duckdb_function_name(name);
  REQUIRE(id.has_value());
  REQUIRE(*id == function_id::minute);
}

TEST_CASE("ast_function_id - second round-trips through name mappers", "[ast_function_id]")
{
  auto const name = to_duckdb_function_name(function_id::second);
  REQUIRE(name == "second");
  auto const id = from_duckdb_function_name(name);
  REQUIRE(id.has_value());
  REQUIRE(*id == function_id::second);
}

TEST_CASE("ast_function_id - millisecond round-trips through name mappers", "[ast_function_id]")
{
  auto const name = to_duckdb_function_name(function_id::millisecond);
  REQUIRE(name == "millisecond");
  auto const id = from_duckdb_function_name(name);
  REQUIRE(id.has_value());
  REQUIRE(*id == function_id::millisecond);
}

TEST_CASE("ast_function_id - microsecond round-trips through name mappers", "[ast_function_id]")
{
  auto const name = to_duckdb_function_name(function_id::microsecond);
  REQUIRE(name == "microsecond");
  auto const id = from_duckdb_function_name(name);
  REQUIRE(id.has_value());
  REQUIRE(*id == function_id::microsecond);
}

TEST_CASE("ast_function_id - date_trunc round-trips through name mappers", "[ast_function_id]")
{
  auto const name = to_duckdb_function_name(function_id::date_trunc);
  REQUIRE(name == "date_trunc");
  auto const id = from_duckdb_function_name(name);
  REQUIRE(id.has_value());
  REQUIRE(*id == function_id::date_trunc);
}

TEST_CASE("ast_function_id - row round-trips through name mappers", "[ast_function_id]")
{
  auto const name = to_duckdb_function_name(function_id::row);
  REQUIRE(name == "row");
  auto const id = from_duckdb_function_name(name);
  REQUIRE(id.has_value());
  REQUIRE(*id == function_id::row);
}

TEST_CASE("ast_function_id - struct_pack round-trips through name mappers", "[ast_function_id]")
{
  auto const name = to_duckdb_function_name(function_id::struct_pack);
  REQUIRE(name == "struct_pack");
  auto const id = from_duckdb_function_name(name);
  REQUIRE(id.has_value());
  REQUIRE(*id == function_id::struct_pack);
}

TEST_CASE("ast_function_id - error round-trips through name mappers", "[ast_function_id]")
{
  auto const name = to_duckdb_function_name(function_id::error);
  REQUIRE(name == "error");
  auto const id = from_duckdb_function_name(name);
  REQUIRE(id.has_value());
  REQUIRE(*id == function_id::error);
}

// ============================================================================
// Substring/substr alias collapse (D-SUB-1)
// ============================================================================

TEST_CASE("ast_function_id - substring and substr both resolve to function_id::substring",
          "[ast_function_id]")
{
  auto const id_long  = from_duckdb_function_name("substring");
  auto const id_short = from_duckdb_function_name("substr");
  REQUIRE(id_long.has_value());
  REQUIRE(id_short.has_value());
  REQUIRE(*id_long == function_id::substring);
  REQUIRE(*id_short == function_id::substring);
}

TEST_CASE(
  "ast_function_id - to_duckdb_function_name(substring) returns the canonical \"substring\"",
  "[ast_function_id]")
{
  REQUIRE(to_duckdb_function_name(function_id::substring) == "substring");
}

// ============================================================================
// Unknown name → std::nullopt (D-01 — no sentinel enum value)
// ============================================================================

TEST_CASE("ast_function_id - unknown function name returns std::nullopt", "[ast_function_id]")
{
  REQUIRE_FALSE(from_duckdb_function_name("nonexistent_fn").has_value());
}

TEST_CASE("ast_function_id - empty function name returns std::nullopt", "[ast_function_id]")
{
  REQUIRE_FALSE(from_duckdb_function_name("").has_value());
}
