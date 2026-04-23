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
#include <sirius/exception.hpp>

// duckdb
#include <duckdb/common/exception.hpp>
#include <duckdb/common/typedefs.hpp>
#include <duckdb/planner/expression/bound_conjunction_expression.hpp>

// cudf
#include <cudf/binaryop.hpp>
#include <cudf/cudf_utils.hpp>

namespace sirius {
using execute_result = gpu_expression_executor::execute_result;

execute_result gpu_expression_executor::execute(duckdb::BoundConjunctionExpression const& expr,
                                                execution_mode mode)
{
  if (_strategy != expression_executor_strategy::MATERIALIZE &&
      (mode == execution_mode::AST || count_ast_ops(expr) >= _min_ast_size)) {
    auto conjunction_type_switch_ast =
      [](duckdb::BoundConjunctionExpression const& expr) -> cudf::ast::ast_operator {
      using enum duckdb::ExpressionType;
      switch (expr.GetExpressionType()) {
        case CONJUNCTION_AND: return cudf::ast::ast_operator::LOGICAL_AND;
        case CONJUNCTION_OR: return cudf::ast::ast_operator::LOGICAL_OR;
        default:
          throw invalid_input_exception(
            "[gpu_expression_executor:conjunction] unrecognized conjunction type {}",
            static_cast<int>(expr.GetExpressionType()));
      }
    };

    auto output = execute(*expr.children[0], execution_mode::AST);

    for (duckdb::idx_t i = 1; i < expr.children.size(); i++) {
      auto child = execute(*expr.children[i], execution_mode::AST);

      auto const& output_expr = _ast_tree.emplace<cudf::ast::operation>(
        conjunction_type_switch_ast(expr), output.get_expr(), child.get_expr());
      output = execute_result(
        ast_result(output_expr,
                   {output.get_temp_scalar_indices(), child.get_temp_scalar_indices()},
                   {output.get_temp_column_indices(), child.get_temp_column_indices()}));
    }

    if (mode == execution_mode::AST) {
      //===----------1: AST Mode----------===//
      return output;
    }
    //===----------2: MATERIALIZE Mode, evaluate node with AST----------===//
    // Evaluate the AST subtree
    auto result_column = execute_ast(output.get_expr());

    // Release consumed temporaries
    release_temporaries(output.get_temp_scalar_indices(), output.get_temp_column_indices());
    return execute_result(std::move(result_column));
  }

  //===----------3: MATERIALIZE Mode, evaluate node with unary/binary ops----------===//
  auto conjunction_type_switch =
    [](duckdb::BoundConjunctionExpression const& expr) -> cudf::binary_operator {
    using enum duckdb::ExpressionType;
    switch (expr.GetExpressionType()) {
      case CONJUNCTION_AND: return cudf::binary_operator::LOGICAL_AND;
      case CONJUNCTION_OR: return cudf::binary_operator::LOGICAL_OR;
      default:
        throw invalid_input_exception(
          "[gpu_expression_executor:conjunction] unrecognized conjunction type {}",
          static_cast<int>(expr.GetExpressionType()));
    }
  };

  auto const output_type = GetCudfType(expr.return_type);

  // Resolve the children incrementally into the output
  auto output = execute(*expr.children[0], execution_mode::MATERIALIZE);
  // DuckDB should prune all scalar conjuncts away
  D_ASSERT(!output.is_scalar());
  for (duckdb::idx_t i = 1; i < expr.children.size(); i++) {
    auto child = execute(*expr.children[i], execution_mode::MATERIALIZE);
    D_ASSERT(!child.is_scalar());
    auto output_column = cudf::binary_operation(output.get_column_view(),
                                                child.get_column_view(),
                                                conjunction_type_switch(expr),
                                                output_type,
                                                _stream,
                                                _mr);
    output             = execute_result(std::move(output_column));
  }
  return output;
}

}  // namespace sirius
