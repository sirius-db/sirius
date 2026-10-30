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
#include <type_traits>

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
/// casts wrap, so a HUGEINT source, which a narrowing cast must range-check, lowers only to DOUBLE.
[[nodiscard]] inline bool ast_supports_cast(logical_type const& source, logical_type const& target)
{
  auto const supported = std::find(supported_ast_cast_types_native.begin(),
                                   supported_ast_cast_types_native.end(),
                                   target.id()) != supported_ast_cast_types_native.end();
  return supported && (source.id() != type_id::HUGEINT || target.id() == type_id::DOUBLE);
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

/// Whether // or % needs row checks before cuDF can execute it. A nonzero constant other than
/// -1 cannot divide by zero or overflow a signed minimum. Fixed-point constants are conservatively
/// admitted only when positive; their arithmetic still materializes to preserve SQL width/scale.
[[nodiscard]] inline bool division_requires_check(ast::function_call const& call)
{
  if (call.function() != function_id::int_div && call.function() != function_id::mod) {
    return false;
  }
  auto const& divisor = *call.arguments()[1];
  if (!divisor.holds<ast::constant>()) { return true; }
  auto const safe = std::visit(
    []<typename T>(T const& v) {
      if constexpr (std::is_arithmetic_v<T>) {
        return v != 0 && (!std::is_signed_v<T> || v != static_cast<T>(-1));
      } else if constexpr (std::is_same_v<T, decimal32> || std::is_same_v<T, decimal64> ||
                           std::is_same_v<T, decimal128>) {
        return v.value > 0;
      } else {
        return false;
      }
    },
    divisor.get<ast::constant>().payload);
  return !safe;
}

/// BOUND_FUNCTION names that are currently safe to lower into a cuDF AST.
inline constexpr std::array<function_id, 6> supported_ast_functions{function_id::add,
                                                                    function_id::sub,
                                                                    function_id::mul,
                                                                    function_id::div,
                                                                    function_id::int_div,
                                                                    function_id::mod};

/// Shared by the evaluator, operation counter and standalone translator. Checked operations
/// must materialize so a join or pushed-down predicate cannot bypass their runtime guards.
[[nodiscard]] inline bool ast_supports_function(ast::function_call const& call)
{
  return !call.date_operation().has_value() && !has_fixed_point_operand(call) &&
         !division_requires_check(call) &&
         std::find(supported_ast_functions.begin(),
                   supported_ast_functions.end(),
                   call.function()) != supported_ast_functions.end();
}

}  // namespace sirius
