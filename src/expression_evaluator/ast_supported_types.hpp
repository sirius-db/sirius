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
#include "expression/ast/node.hpp"
#include "expression/function_id.hpp"
#include "helper/logical_type.hpp"

// duckdb
#include <duckdb/common/types.hpp>

// standard library
#include <algorithm>
#include <array>

// Internal AST capability lists shared by the evaluator, translator, and planning code.
// Keeping them here avoids exposing DuckDB types through expression_evaluator.hpp.

namespace sirius {

/// CAST return types that are currently safe to lower into a cuDF AST.
inline constexpr std::array<duckdb::LogicalTypeId, 3> supported_ast_cast_types{
  {duckdb::LogicalTypeId::UBIGINT, duckdb::LogicalTypeId::BIGINT, duckdb::LogicalTypeId::DOUBLE}};

/// Sirius-typed mirror of supported_ast_cast_types — same set of CAST target
/// types as above, expressed via sirius::type_id for native AST consumers.
inline constexpr std::array<sirius::type_id, 3> supported_ast_cast_types_native{
  {sirius::type_id::UBIGINT, sirius::type_id::BIGINT, sirius::type_id::DOUBLE}};

/// Whether a semantic CAST of @p source to @p target lowers to one cuDF AST op. The AST's integer
/// casts wrap, and DECIMAL-to-integer casts also require rounding. Fixed-point sources therefore
/// lower only to DOUBLE; integer targets go through the checked materialized conversion.
[[nodiscard]] inline bool ast_supports_cast(logical_type const& source, logical_type const& target)
{
  auto const supported = std::find(supported_ast_cast_types_native.begin(),
                                   supported_ast_cast_types_native.end(),
                                   target.id()) != supported_ast_cast_types_native.end();
  return supported && (!source.has_fixed_point_carrier() || target.id() == type_id::DOUBLE);
}

/// Whether @p call's result or any argument is DECIMAL or HUGEINT. cuDF AST does not enforce
/// DuckDB's result width and scale for these, so such a function is materialized.
[[nodiscard]] inline bool has_fixed_point_operand(ast::function_call const& call)
{
  return call.return_type().has_fixed_point_carrier() ||
         std::ranges::any_of(call.arguments(), [](auto const& arg) {
           return arg->return_type().has_fixed_point_carrier();
         });
}

/// BOUND_FUNCTION names that are currently safe to lower into a cuDF AST.
inline constexpr std::array<function_id, 6> supported_ast_functions{function_id::add,
                                                                    function_id::sub,
                                                                    function_id::mul,
                                                                    function_id::div,
                                                                    function_id::int_div,
                                                                    function_id::mod};

}  // namespace sirius
