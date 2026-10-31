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

/// Whether an integer conversion can lose values through narrowing or a signedness change.
/// UHUGEINT uses a UINT64 carrier, as reflected by fixed_width_byte_size().
[[nodiscard]] inline bool integer_cast_requires_range_check(logical_type const& source,
                                                            logical_type const& target)
{
  if (!source.is_integer() || !target.is_integer()) { return false; }
  auto const is_unsigned = [](type_id id) {
    return id == type_id::UTINYINT || id == type_id::USMALLINT || id == type_id::UINTEGER ||
           id == type_id::UBIGINT || id == type_id::UHUGEINT;
  };
  auto const source_unsigned = is_unsigned(source.id());
  auto const target_unsigned = is_unsigned(target.id());
  auto const source_width    = source.fixed_width_byte_size();
  auto const target_width    = target.fixed_width_byte_size();
  if (source_unsigned == target_unsigned) { return source_width > target_width; }
  // A signed source may be negative; an unsigned source needs an extra sign bit.
  return !source_unsigned || source_width >= target_width;
}

/// Whether a semantic CAST lowers to one cuDF AST op. Integer conversions that can overflow
/// must materialize for range checks. Fixed-point sources lower only to DOUBLE because
/// DECIMAL-to-integer casts also require rounding and checked materialized conversion.
[[nodiscard]] inline bool ast_supports_cast(logical_type const& source, logical_type const& target)
{
  auto const supported = std::find(supported_ast_cast_types_native.begin(),
                                   supported_ast_cast_types_native.end(),
                                   target.id()) != supported_ast_cast_types_native.end();
  return supported && !integer_cast_requires_range_check(source, target) &&
         (!source.has_fixed_point_carrier() || target.id() == type_id::DOUBLE);
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
