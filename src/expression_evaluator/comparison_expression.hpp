/*
 * Copyright 2026, Sirius Contributors.
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

#include "expression/join_condition.hpp"
#include "helper/logical_type.hpp"
#include "sirius/exception.hpp"

#include <cudf/ast/expressions.hpp>

namespace sirius {
inline bool floating_comparison(logical_type const& left, logical_type const& right)
{
  auto floating = [](logical_type const& type) {
    return type.id() == type_id::FLOAT || type.id() == type_id::DOUBLE;
  };
  return floating(left) || floating(right);
}

inline cudf::ast::expression const& comparison_expression(cudf::ast::tree& tree,
                                                          comparison_type comparison,
                                                          cudf::ast::expression const& left,
                                                          cudf::ast::expression const& right,
                                                          bool floating)
{
  using op    = cudf::ast::ast_operator;
  auto binary = [&](op kind, auto const& a, auto const& b) -> cudf::ast::expression const& {
    return tree.emplace<cudf::ast::operation>(kind, a, b);
  };
  auto negate = [&](auto const& value) -> cudf::ast::expression const& {
    return tree.emplace<cudf::ast::operation>(op::NOT, value);
  };
  op kind;
  switch (comparison) {
    case comparison_type::equal: kind = op::EQUAL; break;
    case comparison_type::not_equal: kind = floating ? op::EQUAL : op::NOT_EQUAL; break;
    case comparison_type::gt: kind = op::GREATER; break;
    case comparison_type::ge: kind = op::GREATER_EQUAL; break;
    case comparison_type::lt: kind = op::LESS; break;
    case comparison_type::le: kind = op::LESS_EQUAL; break;
    case comparison_type::distinct_from:
    case comparison_type::not_distinct_from: kind = op::NULL_EQUAL; break;
    default: throw invalid_input_exception("Unsupported comparison operator");
  }
  auto const* result = &binary(kind, left, right);
  if (floating) {
    // DuckDB orders NaN above all numbers and treats all NaNs as equal.
    // NULL_EQUAL(x, x) is false only for NaN, so these flags are never NULL.
    auto const& left_nan  = negate(binary(op::NULL_EQUAL, left, left));
    auto const& right_nan = negate(binary(op::NULL_EQUAL, right, right));
    cudf::ast::expression const* special;
    switch (comparison) {
      case comparison_type::gt:
        special = &binary(op::LOGICAL_AND, left_nan, negate(right_nan));
        break;
      case comparison_type::ge: special = &left_nan; break;
      case comparison_type::lt:
        special = &binary(op::LOGICAL_AND, negate(left_nan), right_nan);
        break;
      case comparison_type::le: special = &right_nan; break;
      default: special = &binary(op::LOGICAL_AND, left_nan, right_nan); break;
    }
    // Ordinary OR propagates a NULL comparison; null-safe equality is non-nullable.
    result = &binary(op::LOGICAL_OR, *result, *special);
  }
  if ((floating && comparison == comparison_type::not_equal) ||
      comparison == comparison_type::distinct_from)
    return negate(*result);
  return *result;
}
}  // namespace sirius
