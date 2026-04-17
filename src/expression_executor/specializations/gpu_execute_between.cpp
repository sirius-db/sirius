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
#include <expression_executor/gpu_expression_executor.hpp>

// duckdb
#include <duckdb/planner/expression/bound_between_expression.hpp>

// cudf
#include <cudf/binaryop.hpp>
#include <cudf/cudf_utils.hpp>
#include <cudf/transform.hpp>

namespace sirius {
using execute_result = gpu_expression_executor::execute_result;

execute_result gpu_expression_executor::execute(duckdb::BoundBetweenExpression const& expr,
                                                execution_mode mode)
{
  if (_strategy != expression_executor_strategy::MATERIALIZE &&
      (mode == execution_mode::AST || count_ast_ops(expr) >= _min_ast_size)) {
    auto comparison_type_switch_ast = [](duckdb::BoundBetweenExpression const& expr)
      -> std::pair<cudf::ast::ast_operator, cudf::ast::ast_operator> {
      cudf::ast::ast_operator lower_op;
      cudf::ast::ast_operator upper_op;
      if (expr.lower_inclusive) {
        lower_op = cudf::ast::ast_operator::GREATER_EQUAL;
      } else {
        lower_op = cudf::ast::ast_operator::GREATER;
      }
      if (expr.upper_inclusive) {
        upper_op = cudf::ast::ast_operator::LESS_EQUAL;
      } else {
        upper_op = cudf::ast::ast_operator::LESS;
      }
      return {lower_op, upper_op};
    };

    auto input = execute(*expr.input, execution_mode::AST);
    D_ASSERT(!input.is_scalar());
    auto lower                = execute(*expr.lower, execution_mode::AST);
    auto upper                = execute(*expr.upper, execution_mode::AST);
    auto [lower_op, upper_op] = comparison_type_switch_ast(expr);
    auto const& lower_expr =
      _ast_tree.emplace<cudf::ast::operation>(lower_op, input.get_expr(), lower.get_expr());
    auto const& upper_expr =
      _ast_tree.emplace<cudf::ast::operation>(upper_op, input.get_expr(), upper.get_expr());
    auto const& between_expr = _ast_tree.emplace<cudf::ast::operation>(
      cudf::ast::ast_operator::LOGICAL_AND, lower_expr, upper_expr);

    //===----------1: AST Mode----------===//
    if (mode == execution_mode::AST) {
      return execute_result(ast_result(between_expr,
                                       {input.get_temp_scalar_indices(),
                                        lower.get_temp_scalar_indices(),
                                        upper.get_temp_scalar_indices()},
                                       {input.get_temp_column_indices(),
                                        lower.get_temp_column_indices(),
                                        upper.get_temp_column_indices()}));
    }

    //===----------2: MATERIALIZE Mode, evaluate node with AST----------===//
    // Evaluate the AST subtree
    auto result_column = execute_ast(between_expr);
    release_temporaries({input.get_temp_scalar_indices(),
                         lower.get_temp_scalar_indices(),
                         upper.get_temp_scalar_indices()},
                        {input.get_temp_column_indices(),
                         lower.get_temp_column_indices(),
                         upper.get_temp_column_indices()});
    return execute_result(std::move(result_column));
  }

  //===----------3: MATERIALIZE Mode, evaluate node with unary/binary ops----------===//
  auto comparison_type_switch = [](duckdb::BoundBetweenExpression const& expr)
    -> std::pair<cudf::binary_operator, cudf::binary_operator> {
    cudf::binary_operator lower_op;
    cudf::binary_operator upper_op;
    if (expr.lower_inclusive) {
      lower_op = cudf::binary_operator::GREATER_EQUAL;
    } else {
      lower_op = cudf::binary_operator::GREATER;
    }
    if (expr.upper_inclusive) {
      upper_op = cudf::binary_operator::LESS_EQUAL;
    } else {
      upper_op = cudf::binary_operator::LESS;
    }
    return {lower_op, upper_op};
  };

  auto input                = execute(*expr.input, execution_mode::MATERIALIZE);
  auto lower                = execute(*expr.lower, execution_mode::MATERIALIZE);
  auto upper                = execute(*expr.upper, execution_mode::MATERIALIZE);
  auto [lower_op, upper_op] = comparison_type_switch(expr);
  auto const output_type    = GetCudfType(expr.return_type);

  std::unique_ptr<cudf::column> lower_cmp;
  std::unique_ptr<cudf::column> upper_cmp;
  if (lower.is_scalar()) {
    lower_cmp = cudf::binary_operation(
      input.get_column_view(), lower.get_scalar(), lower_op, output_type, _stream, _mr);
  } else {
    lower_cmp = cudf::binary_operation(
      input.get_column_view(), lower.get_column_view(), lower_op, output_type, _stream, _mr);
  }
  if (upper.is_scalar()) {
    upper_cmp = cudf::binary_operation(
      input.get_column_view(), upper.get_scalar(), upper_op, output_type, _stream, _mr);
  } else {
    upper_cmp = cudf::binary_operation(
      input.get_column_view(), upper.get_column_view(), upper_op, output_type, _stream, _mr);
  }
  auto result_column = cudf::binary_operation(lower_cmp->view(),
                                              upper_cmp->view(),
                                              cudf::binary_operator::LOGICAL_AND,
                                              output_type,
                                              _stream,
                                              _mr);
  return execute_result(std::move(result_column));
}

}  // namespace sirius
