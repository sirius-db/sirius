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

// sirius
#include <duckdb/common/enums/expression_type.hpp>
#include <expression_executor/gpu_expression_executor.hpp>
#include <operator/empty_str_check.cuh>
#include <sirius/exception.hpp>

// duckdb
#include <duckdb/common/exception.hpp>
#include <duckdb/planner/expression/bound_comparison_expression.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>

// cudf
#include <cudf/ast/ast_operator.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/cudf_utils.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/transform.hpp>

// standard library
#include <memory>

namespace sirius {
using execute_result = gpu_expression_executor::execute_result;

execute_result gpu_expression_executor::execute(duckdb::BoundComparisonExpression const& expr,
                                                execution_mode mode)
{
  if (_strategy != expression_executor_strategy::MATERIALIZE &&
      (mode == execution_mode::AST || count_ast_ops(expr) >= _min_ast_size)) {
    auto comparison_type_switch_ast =
      [](duckdb::BoundComparisonExpression const& expr) -> cudf::ast::ast_operator {
      using enum duckdb::ExpressionType;
      switch (expr.GetExpressionType()) {
        case COMPARE_EQUAL: return cudf::ast::ast_operator::EQUAL;
        case COMPARE_GREATERTHAN: return cudf::ast::ast_operator::GREATER;
        case COMPARE_GREATERTHANOREQUALTO: return cudf::ast::ast_operator::GREATER_EQUAL;
        case COMPARE_LESSTHAN: return cudf::ast::ast_operator::LESS;
        case COMPARE_LESSTHANOREQUALTO: return cudf::ast::ast_operator::LESS_EQUAL;
        case COMPARE_NOTEQUAL: return cudf::ast::ast_operator::NOT_EQUAL;
        case COMPARE_DISTINCT_FROM:  // Fallthrough: special handling below
        case COMPARE_NOT_DISTINCT_FROM: return cudf::ast::ast_operator::NULL_EQUAL;
        default:
          throw invalid_input_exception(
            "[expression_executor:comparison] Unrecognized comparison type : {}",
            static_cast<int>(expr.GetExpressionType()));
      }
    };

    auto left             = execute(*expr.left, execution_mode::AST);
    auto right            = execute(*expr.right, execution_mode::AST);
    auto const& comp_expr = _ast_tree.emplace<cudf::ast::operation>(
      comparison_type_switch_ast(expr), left.get_expr(), right.get_expr());
    // COMPARE_DISTINCT_FROM is semantically equivalent to NOT(NULL_EQUAL())
    auto const& final_comp_expr =
      expr.GetExpressionType() == duckdb::ExpressionType::COMPARE_DISTINCT_FROM
        ? _ast_tree.emplace<cudf::ast::operation>(cudf::ast::ast_operator::NOT, comp_expr)
        : comp_expr;

    //===----------1: AST Mode----------===//
    if (mode == execution_mode::AST) {
      return execute_result(
        ast_result(final_comp_expr,
                   {left.get_temp_scalar_indices(), right.get_temp_scalar_indices()},
                   {left.get_temp_column_indices(), right.get_temp_column_indices()}));
    }

    //===----------2: MATERIALIZE Mode, evaluate node with AST----------===//
    // Evaluate the AST subtree
    auto result_column = execute_ast(final_comp_expr);

    // Release consumed temporaries
    release_temporaries({left.get_temp_scalar_indices(), right.get_temp_scalar_indices()},
                        {left.get_temp_column_indices(), right.get_temp_column_indices()});
    return execute_result(std::move(result_column));
  }

  //===----------3: MATERIALIZE Mode, evaluate node with unary/binary ops----------===//
  if (mode == execution_mode::AST) {
    auto result = execute(expr, execution_mode::MATERIALIZE);
    return materialize_as_ast_column(result.release_column());
  }
  auto comparison_type_switch =
    [](duckdb::BoundComparisonExpression const& expr) -> cudf::binary_operator {
    using enum duckdb::ExpressionType;
    switch (expr.GetExpressionType()) {
      case COMPARE_EQUAL: return cudf::binary_operator::EQUAL;
      case COMPARE_GREATERTHAN: return cudf::binary_operator::GREATER;
      case COMPARE_GREATERTHANOREQUALTO: return cudf::binary_operator::GREATER_EQUAL;
      case COMPARE_LESSTHAN: return cudf::binary_operator::LESS;
      case COMPARE_LESSTHANOREQUALTO: return cudf::binary_operator::LESS_EQUAL;
      case COMPARE_NOTEQUAL: return cudf::binary_operator::NOT_EQUAL;
      case COMPARE_DISTINCT_FROM: return cudf::binary_operator::NULL_NOT_EQUALS;
      case COMPARE_NOT_DISTINCT_FROM: return cudf::binary_operator::NULL_EQUALS;
      default:
        throw invalid_input_exception(
          "[expression_executor:comparison] Unrecognized comparison type : {}",
          static_cast<int>(expr.GetExpressionType()));
    }
  };

  auto left              = execute(*expr.left, execution_mode::MATERIALIZE);
  auto right             = execute(*expr.right, execution_mode::MATERIALIZE);
  auto const output_type = GetCudfType(expr.return_type);

  std::unique_ptr<cudf::column> result_column;
  if (left.is_scalar()) {
    result_column = cudf::binary_operation(left.get_scalar(),
                                           right.get_column_view(),
                                           comparison_type_switch(expr),
                                           output_type,
                                           _stream,
                                           _mr);
  } else if (right.is_scalar()) {
    result_column = cudf::binary_operation(left.get_column_view(),
                                           right.get_scalar(),
                                           comparison_type_switch(expr),
                                           output_type,
                                           _stream,
                                           _mr);
  } else {
    result_column = cudf::binary_operation(left.get_column_view(),
                                           right.get_column_view(),
                                           comparison_type_switch(expr),
                                           output_type,
                                           _stream,
                                           _mr);
  }
  return execute_result(std::move(result_column));
}
}  // namespace sirius
