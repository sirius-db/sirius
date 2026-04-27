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

// D-02: forward table — 28 entries (substring has TWO DuckDB-side aliases).
// Linear scan; called once per BoundFunctionExpression at executor entry.
inline constexpr std::array<std::pair<std::string_view, function_id>, 28> kForwardTable = {{
  {"+",              function_id::add},
  {"-",              function_id::sub},
  {"*",              function_id::mul},
  {"/",              function_id::div},
  {"//",             function_id::int_div},
  {"%",              function_id::mod},
  {"substring",      function_id::substring},   // canonical (D-SUB-1)
  {"substr",         function_id::substring},   // alias    (D-SUB-1)
  {"~~",             function_id::like},
  {"!~~",            function_id::not_like},
  {"contains",       function_id::contains},
  {"prefix",         function_id::prefix},
  {"suffix",         function_id::suffix},
  {"strlen",         function_id::strlen},
  {"length",         function_id::length},
  {"regexp_replace", function_id::regexp_replace},
  {"year",           function_id::year},
  {"month",          function_id::month},
  {"day",            function_id::day},
  {"hour",           function_id::hour},
  {"minute",         function_id::minute},
  {"second",         function_id::second},
  {"millisecond",    function_id::millisecond},
  {"microsecond",    function_id::microsecond},
  {"date_trunc",     function_id::date_trunc},
  {"row",            function_id::row},
  {"struct_pack",    function_id::struct_pack},
  {"error",          function_id::error},
}};

// D-02: reverse table — 27 entries, indexed by enum value (NOT searched).
inline constexpr std::array<std::string_view, 27> kReverseTable = {
  "+", "-", "*", "/", "//", "%",
  "substring", "~~", "!~~", "contains", "prefix", "suffix",
  "strlen", "length", "regexp_replace",
  "year", "month", "day", "hour", "minute", "second",
  "millisecond", "microsecond", "date_trunc",
  "row", "struct_pack", "error",
};

static_assert(static_cast<std::size_t>(function_id::error) + 1 == 27,
              "function_id::error must be the last entry; cardinality locked at 27.");
static_assert(kReverseTable.size() == 27,
              "kReverseTable must have one slot per function_id value.");
static_assert(kForwardTable.size() == 28,
              "kForwardTable has one extra entry for the substring/substr alias.");

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
