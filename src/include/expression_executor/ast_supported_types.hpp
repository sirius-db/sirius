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

// duckdb
#include <duckdb/common/types.hpp>

// standard library
#include <array>
#include <string_view>

// Internal header shared by expression-executor .cpp files that need to consult the static
// allow-lists of AST-compatible CAST target types and BOUND_FUNCTION names. Keeps DuckDB
// includes out of the public gpu_expression_executor.hpp surface.

namespace sirius {

/// CAST return types that are currently safe to lower into a cuDF AST.
inline constexpr std::array<duckdb::LogicalTypeId, 3> supported_ast_cast_types{
  {duckdb::LogicalTypeId::UBIGINT, duckdb::LogicalTypeId::BIGINT, duckdb::LogicalTypeId::DOUBLE}};

/// BOUND_FUNCTION names that are currently safe to lower into a cuDF AST.
inline constexpr std::array<std::string_view, 6> supported_ast_functions{
  "+", "-", "*", "/", "//", "%"};

}  // namespace sirius
