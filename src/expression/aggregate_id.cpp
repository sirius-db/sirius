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

#include "expression/aggregate_id.hpp"

// standard library
#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <string_view>
#include <utility>

namespace sirius {

namespace {

// Forward table: DuckDB aggregate name -> Sirius aggregate id.
// Linear scan; called once per BoundAggregateExpression at translator entry.
constexpr std::array<std::pair<std::string_view, aggregate_id>, 8> kForwardTable = {{
  {"sum", aggregate_id::sum},
  {"sum_no_overflow", aggregate_id::sum_no_overflow},
  {"count", aggregate_id::count},
  {"count_star", aggregate_id::count_star},
  {"min", aggregate_id::min},
  {"max", aggregate_id::max},
  {"avg", aggregate_id::avg},
  {"first", aggregate_id::first},
}};

// Reverse table: Sirius aggregate id -> canonical DuckDB aggregate name.
// Indexed directly by enum value; never searched.
constexpr std::array<std::string_view, 8> kReverseTable = {
  "sum",
  "sum_no_overflow",
  "count",
  "count_star",
  "min",
  "max",
  "avg",
  "first",
};

static_assert(static_cast<std::size_t>(aggregate_id::first) + 1 == 8,
              "aggregate_id::first must be the last entry; cardinality locked at 8.");
static_assert(kReverseTable.size() == 8,
              "kReverseTable must have one slot per aggregate_id value.");
static_assert(kForwardTable.size() == 8,
              "kForwardTable must have one entry per aggregate_id value.");

// Walks both tables to ensure every enum value has exactly one canonical
// forward entry whose name matches the reverse table at the same index.
// Catches table drift (missing ids, duplicated canonicals, name mismatches)
// at compile time.
consteval bool aggregate_id_tables_are_consistent()
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

static_assert(aggregate_id_tables_are_consistent(),
              "aggregate_id forward/reverse tables must agree on one canonical name per id.");

}  // namespace

std::optional<aggregate_id> from_duckdb_aggregate_name(std::string_view name)
{
  auto const it =
    std::ranges::find(kForwardTable, name, &decltype(kForwardTable)::value_type::first);
  if (it == kForwardTable.end()) { return std::nullopt; }
  return it->second;
}

std::string_view to_duckdb_aggregate_name(aggregate_id id)
{
  return kReverseTable[static_cast<std::size_t>(id)];
}

}  // namespace sirius
