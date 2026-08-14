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
 * @brief Closed identifier for every named scalar function the Sirius GPU
 *        executor recognises today.
 *
 * Backing type uint16_t pinned as ABI (locked at compile time in
 * test_function_id.cpp). The order is grouped by category for human
 * readability; integer values are part of the public ABI — new entries go
 * at the end of their category, never in the middle. Cardinality is
 * exactly 29 (D-01).
 */
enum class function_id : uint16_t {
  // Arithmetic — 6 entries (also the contents of supported_ast_functions)
  add = 0,
  sub,
  mul,
  div,
  int_div,
  mod,

  // String — 11 entries
  substring,
  like,
  not_like,
  contains,
  prefix,
  suffix,
  strlen,
  length,
  regexp_replace,
  concat,           // concat() function — ignores NULL arguments
  concat_operator,  // DuckDB's || operator — propagates NULL (distinct from concat)

  // Temporal — 9 entries
  year,
  month,
  day,
  hour,
  minute,
  second,
  millisecond,
  microsecond,
  date_trunc,

  // Struct / control — 3 entries
  row,
  struct_pack,
  error,
};

/**
 * @brief Resolve a DuckDB scalar-function name to the matching function_id.
 *
 * Returns std::nullopt when the name does not correspond to any function
 * the Sirius executor handles. Both "substring" and "substr" resolve to
 * function_id::substring (D-SUB-1).
 */
std::optional<function_id> from_duckdb_function_name(std::string_view name);

/**
 * @brief Recover the canonical DuckDB function name for a function_id.
 *
 * Returns the DuckDB-side spelling that round-trips back through
 * from_duckdb_function_name. For function_id::substring, the canonical
 * spelling is "substring" (D-SUB-1).
 */
std::string_view to_duckdb_function_name(function_id id);

}  // namespace sirius
