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

// sirius
#include "sirius/exception.hpp"  // sirius::internal_exception

// standard library
#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <optional>
#include <string_view>

namespace sirius {

namespace {

// DuckDB name + argument count -> function id. A name may repeat with disjoint arities: a
// one-argument "-" is negation. Linear scan; called once per BoundFunctionExpression.
struct function_signature {
  std::string_view name;
  function_id id;
  std::size_t min_arguments;
  std::size_t max_arguments;
};

constexpr std::size_t kVariadic = std::numeric_limits<std::size_t>::max();

constexpr std::array<function_signature, 44> kSignatures = {{
  {"+", function_id::add, 2, 2},
  {"add", function_id::add, 2, 2},
  {"-", function_id::sub, 2, 2},
  {"subtract", function_id::sub, 2, 2},
  {"*", function_id::mul, 2, 2},
  {"multiply", function_id::mul, 2, 2},
  {"/", function_id::div, 2, 2},
  {"divide", function_id::div, 2, 2},
  {"//", function_id::int_div, 2, 2},
  {"%", function_id::mod, 2, 2},
  {"modulus", function_id::mod, 2, 2},
  {"substring", function_id::substring, 2, 3},  // canonical name
  {"substr", function_id::substring, 2, 3},     // alias
  {"~~", function_id::like, 2, 2},
  {"!~~", function_id::not_like, 2, 2},
  {"contains", function_id::contains, 2, 2},
  {"prefix", function_id::prefix, 2, 2},
  {"suffix", function_id::suffix, 2, 2},
  {"strlen", function_id::strlen, 1, 1},
  {"length", function_id::length, 1, 1},
  // The GPU handler does not apply the optional fourth (options) argument.
  {"regexp_replace", function_id::regexp_replace, 3, 3},
  {"concat", function_id::concat, 1, kVariadic},  // concat() call — ignores NULL args
  {"||", function_id::concat_operator, 2, 2},  // || operator (ConcatOperatorFun) — propagates NULL
  {"year", function_id::year, 1, 1},
  {"month", function_id::month, 1, 1},
  {"day", function_id::day, 1, 1},
  {"hour", function_id::hour, 1, 1},
  {"minute", function_id::minute, 1, 1},
  {"second", function_id::second, 1, 1},
  {"millisecond", function_id::millisecond, 1, 1},
  {"microsecond", function_id::microsecond, 1, 1},
  {"date_trunc", function_id::date_trunc, 2, 2},
  {"row", function_id::row, 1, kVariadic},
  {"struct_pack", function_id::struct_pack, 1, kVariadic},
  {"error", function_id::error, 1, 1},
  {"constant_or_null", function_id::constant_or_null, 2, kVariadic},
  {"upper", function_id::upper, 1, 1},
  {"lower", function_id::lower, 1, 1},
  {"round", function_id::round, 1, 2},
  {"abs", function_id::abs, 1, 1},
  // dividend, divisor and the constant result scale.
  {"sirius_decimal_divide", function_id::decimal_divide, 3, 3},
  {"-", function_id::negate, 1, 1},
  {"subtract", function_id::negate, 1, 1},
  {"negate", function_id::negate, 1, 1},  // Substrait spelling
}};

// Reverse table: Sirius function id -> canonical DuckDB function name.
// Indexed directly by enum value; never searched.
constexpr std::array<std::string_view, 36> kReverseTable = {
  "+",          "-",         "*",           "/",           "//",
  "%",          "substring", "~~",          "!~~",         "contains",
  "prefix",     "suffix",    "strlen",      "length",      "regexp_replace",
  "concat",     "||",        "year",        "month",       "day",
  "hour",       "minute",    "second",      "millisecond", "microsecond",
  "date_trunc", "row",       "struct_pack", "error",       "constant_or_null",
  "upper",      "lower",     "round",       "abs",         "sirius_decimal_divide",
  "-"};

static_assert(static_cast<std::size_t>(function_id::negate) + 1 == 36,
              "function_id::negate must be the last entry; cardinality locked at 36.");
static_assert(kReverseTable.size() == 36,
              "kReverseTable must have one slot per function_id value.");

// Each id has exactly one row named as in kReverseTable, and rows sharing a name have disjoint
// arities, so a (name, argument count) pair resolves to at most one id.
consteval bool function_id_tables_are_consistent()
{
  std::array<int, kReverseTable.size()> canonical_counts{};
  for (auto const& signature : kSignatures) {
    auto const idx = static_cast<std::size_t>(signature.id);
    if (idx >= kReverseTable.size()) { return false; }
    if (signature.min_arguments > signature.max_arguments) { return false; }
    if (signature.name == kReverseTable[idx]) { ++canonical_counts[idx]; }
  }
  for (auto const count : canonical_counts) {
    if (count != 1) { return false; }
  }
  for (std::size_t i = 0; i < kSignatures.size(); ++i) {
    for (std::size_t j = i + 1; j < kSignatures.size(); ++j) {
      auto const& a = kSignatures[i];
      auto const& b = kSignatures[j];
      if (a.name == b.name && a.min_arguments <= b.max_arguments &&
          b.min_arguments <= a.max_arguments) {
        return false;
      }
    }
  }
  return true;
}

static_assert(function_id_tables_are_consistent(),
              "function_id tables must agree on one canonical name per id and resolve each "
              "(name, argument count) pair to at most one id.");

}  // namespace

std::optional<function_id> from_duckdb_function_name(std::string_view name,
                                                     std::size_t argument_count)
{
  for (auto const& signature : kSignatures) {
    if (signature.name == name && signature.min_arguments <= argument_count &&
        argument_count <= signature.max_arguments) {
      return signature.id;
    }
  }
  return std::nullopt;
}

std::string_view to_duckdb_function_name(function_id id)
{
  auto const idx = static_cast<std::size_t>(id);
  if (idx >= kReverseTable.size()) {
    throw internal_exception("[function_id] function id {} is out of range", idx);
  }
  return kReverseTable[idx];
}

bool accepts_argument_count(function_id id, std::size_t argument_count)
{
  return std::ranges::any_of(kSignatures, [&](function_signature const& signature) {
    return signature.id == id && signature.min_arguments <= argument_count &&
           argument_count <= signature.max_arguments;
  });
}

}  // namespace sirius
