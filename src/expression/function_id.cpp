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

#include "expression/function_id.hpp"

// standard library
#include <array>
#include <cstddef>
#include <optional>
#include <string_view>
#include <utility>

namespace sirius {

namespace {

// Forward table: DuckDB function name -> Sirius function id.
// Symbolic SQL operators and their Substrait spellings resolve to the same ids.
// Linear scan; called once per BoundFunctionExpression at executor entry.
constexpr std::array<std::pair<std::string_view, function_id>, 35> kForwardTable = {{
  {"+", function_id::add},
  {"add", function_id::add},
  {"-", function_id::sub},
  {"subtract", function_id::sub},
  {"*", function_id::mul},
  {"multiply", function_id::mul},
  {"/", function_id::div},
  {"divide", function_id::div},
  {"//", function_id::int_div},
  {"%", function_id::mod},
  {"modulus", function_id::mod},
  {"substring", function_id::substring},  // canonical name
  {"substr", function_id::substring},     // alias
  {"~~", function_id::like},
  {"!~~", function_id::not_like},
  {"contains", function_id::contains},
  {"prefix", function_id::prefix},
  {"suffix", function_id::suffix},
  {"strlen", function_id::strlen},
  {"length", function_id::length},
  {"regexp_replace", function_id::regexp_replace},
  {"concat", function_id::concat},       // concat() call — ignores NULL args
  {"||", function_id::concat_operator},  // || operator (ConcatOperatorFun) — propagates NULL
  {"year", function_id::year},
  {"month", function_id::month},
  {"day", function_id::day},
  {"hour", function_id::hour},
  {"minute", function_id::minute},
  {"second", function_id::second},
  {"millisecond", function_id::millisecond},
  {"microsecond", function_id::microsecond},
  {"date_trunc", function_id::date_trunc},
  {"row", function_id::row},
  {"struct_pack", function_id::struct_pack},
  {"error", function_id::error},
}};

// Reverse table: Sirius function id -> canonical DuckDB function name.
// Indexed directly by enum value; never searched.
constexpr std::array<std::string_view, 29> kReverseTable = {
  "+",          "-",         "*",           "/",           "//",
  "%",          "substring", "~~",          "!~~",         "contains",
  "prefix",     "suffix",    "strlen",      "length",      "regexp_replace",
  "concat",     "||",        "year",        "month",       "day",
  "hour",       "minute",    "second",      "millisecond", "microsecond",
  "date_trunc", "row",       "struct_pack", "error",
};

static_assert(static_cast<std::size_t>(function_id::error) + 1 == 29,
              "function_id::error must be the last entry; cardinality locked at 29.");
static_assert(kReverseTable.size() == 29,
              "kReverseTable must have one slot per function_id value.");
static_assert(kForwardTable.size() == 35,
              "kForwardTable includes SQL and Substrait aliases for supported function ids.");

// Walks both tables to ensure every enum value has exactly one canonical
// forward entry whose name matches the reverse table at the same index.
// Catches table drift (missing ids, duplicated canonicals, name mismatches)
// at compile time.
consteval bool function_id_tables_are_consistent()
{
  std::array<int, kReverseTable.size()> canonical_counts{};
  for (auto const& [name, id] : kForwardTable) {
    auto const idx = static_cast<std::size_t>(id);
    if (idx >= kReverseTable.size()) { return false; }
    if (name == kReverseTable[idx]) { ++canonical_counts[idx]; }
  }
  for (auto const count : canonical_counts) {
    if (count != 1) { return false; }
  }
  return true;
}

static_assert(function_id_tables_are_consistent(),
              "function_id forward/reverse tables must agree on one canonical name per id.");

}  // namespace

std::optional<function_id> from_duckdb_function_name(std::string_view name)
{
  for (auto const& [entry_name, id] : kForwardTable) {
    if (entry_name == name) { return id; }
  }
  return std::nullopt;
}

std::string_view to_duckdb_function_name(function_id id)
{
  return kReverseTable[static_cast<std::size_t>(id)];
}

}  // namespace sirius
