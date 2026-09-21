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

#pragma once

// standard library
#include <cstdint>
#include <optional>
#include <string_view>

namespace sirius {

/**
 * @brief Closed identifier for every named aggregate function the Sirius GPU
 *        executor recognises today.
 *
 * Backing type uint16_t pinned as ABI (locked at compile time in
 * test_ast_aggregate.cpp). The integer values are part of the public ABI —
 * new entries go at the end, never in the middle. Cardinality is exactly 8.
 *
 * Mirrors the closed sirius::function_id enum precedent. See
 * https://github.com/sirius-db/sirius/issues/863.
 */
enum class aggregate_id : uint16_t {
  sum = 0,
  sum_no_overflow,
  count,
  count_star,
  min,
  max,
  avg,
  first,
};

/**
 * @brief Resolve a DuckDB aggregate-function name to the matching aggregate_id.
 *
 * Returns std::nullopt when the name does not correspond to any aggregate the
 * Sirius executor handles (a fallback signal). The DuckDB-side names equal the
 * enum spellings.
 */
std::optional<aggregate_id> from_duckdb_aggregate_name(std::string_view name);

/**
 * @brief Recover the canonical DuckDB aggregate name for an aggregate_id.
 *
 * Returns the DuckDB-side spelling that round-trips back through
 * from_duckdb_aggregate_name.
 */
std::string_view to_duckdb_aggregate_name(aggregate_id id);

}  // namespace sirius
