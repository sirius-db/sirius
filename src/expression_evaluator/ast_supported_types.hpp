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

// sirius
#include "expression/function_id.hpp"
#include "helper/logical_type.hpp"  // sirius::type_id

// duckdb
#include <duckdb/common/types.hpp>

// standard library
#include <array>

// Internal AST capability lists shared by the evaluator, translator, and planning code.
// Keeping them here avoids exposing DuckDB types through expression_evaluator.hpp.

namespace sirius {

/// Whether a semantic CAST from @p source to @p target may lower to a cuDF AST CAST_TO_* op. These
/// ops convert like static_cast: they truncate floats toward zero and wrap integers, where DuckDB
/// rounds and range-checks. An integer target therefore takes only BOOLEAN and integer sources
/// whose values all fit; HUGEINT and UHUGEINT run as INT64 and UINT64 on the GPU.
constexpr bool cast_lowers_to_cudf_ast(sirius::type_id source, sirius::type_id target)
{
  switch (target) {
    case sirius::type_id::DOUBLE: return true;
    case sirius::type_id::BIGINT:
      switch (source) {
        case sirius::type_id::BOOLEAN:
        case sirius::type_id::TINYINT:
        case sirius::type_id::SMALLINT:
        case sirius::type_id::INTEGER:
        case sirius::type_id::BIGINT:
        case sirius::type_id::HUGEINT:
        case sirius::type_id::UTINYINT:
        case sirius::type_id::USMALLINT:
        case sirius::type_id::UINTEGER: return true;
        default: return false;
      }
    case sirius::type_id::UBIGINT:
      switch (source) {
        case sirius::type_id::BOOLEAN:
        case sirius::type_id::UTINYINT:
        case sirius::type_id::USMALLINT:
        case sirius::type_id::UINTEGER:
        case sirius::type_id::UBIGINT:
        case sirius::type_id::UHUGEINT: return true;
        default: return false;
      }
    default: return false;
  }
}

/// BOUND_FUNCTION names that are currently safe to lower into a cuDF AST.
inline constexpr std::array<function_id, 6> supported_ast_functions{function_id::add,
                                                                    function_id::sub,
                                                                    function_id::mul,
                                                                    function_id::div,
                                                                    function_id::int_div,
                                                                    function_id::mod};

}  // namespace sirius
