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
#include <duckdb/common/exception.hpp>
#include <duckdb/planner/expression/bound_case_expression.hpp>
#include <duckdb/planner/expression/bound_function_expression.hpp>

// cudf
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/cudf_utils.hpp>
#include <cudf/reduction.hpp>

// We need to handle implicit error checks inserted as CASE statements by DuckDB
#define ERROR_FUNC_STR "error"

namespace sirius {
using execute_result = gpu_expression_executor::execute_result;
execute_result gpu_expression_executor::execute(duckdb::BoundCaseExpression const& expr,
                                                execution_mode mode)
{
  //===----------MATERIALIZE (AST breaker)----------===//
  // CASE cannot be represented as a cudf AST operation, so we always materialize it. If the caller
  // requested AST mode, we materialize the result and wrap it as a temporary column that the
  // parent's AST tree can reference.
  std::unique_ptr<cudf::column> output;

  // First, execute the ELSE
  auto current_result = execute(*expr.else_expr, execution_mode::MATERIALIZE);

  // Loop backwards, so that the THEN of the first true WHEN is copied to the output column
  auto num_checks = static_cast<int32_t>(
    expr.case_checks.size());  // This is sane, and needed for the descending loop index
  for (int32_t i = num_checks - 1; i >= 0; --i) {
    auto& case_check = expr.case_checks[i];

    // Fist, execute the WHEN expression to get boolean array intermediate
    auto current_mask = execute(*case_check.when_expr, execution_mode::MATERIALIZE);

    // Check for error functions
    if (case_check.then_expr->GetExpressionClass() == duckdb::ExpressionClass::BOUND_FUNCTION &&
        case_check.then_expr->Cast<duckdb::BoundFunctionExpression>().function.name ==
          ERROR_FUNC_STR) {
      // If the THEN is true anywhere, throw error()
      bool throw_error = false;
      if (current_mask.is_scalar()) {
        auto const& bool_scalar =
          static_cast<cudf::scalar_type_t<bool> const&>(current_mask.get_scalar());
        if (bool_scalar.is_valid(_stream)) { throw_error = bool_scalar.value(_stream); }
      } else {
        auto any_result = cudf::reduce(current_mask.get_column_view(),
                                       *cudf::make_any_aggregation<cudf::reduce_aggregation>(),
                                       cudf::data_type(cudf::type_id::BOOL8),
                                       _stream,
                                       _mr);
        throw_error     = static_cast<cudf::scalar_type_t<bool>*>(any_result.get())->value(_stream);
      }
      if (throw_error) {
        // Assume that this arises for the stated error
        throw duckdb::InternalException(
          "[gpu_expression_executor:case]: More than one row returned by a subquery used as an "
          "expression.");
      }
      continue;
    }

    // Otherwise, execute the THEN and selectively copy to the output
    auto current_then = execute(*case_check.then_expr, execution_mode::MATERIALIZE);
    if (current_result.is_scalar()) {
      // This can only possibly happen when i = num_checks - 1
      if (current_then.is_scalar()) {
        output = cudf::copy_if_else(current_then.get_scalar(),
                                    current_result.get_scalar(),
                                    current_mask.get_column_view(),
                                    _stream,
                                    _mr);
      } else {
        output = cudf::copy_if_else(current_then.get_column_view(),
                                    current_result.get_scalar(),
                                    current_mask.get_column_view(),
                                    _stream,
                                    _mr);
      }
    } else if (current_then.is_scalar()) {
      output = cudf::copy_if_else(current_then.get_scalar(),
                                  current_result.get_column_view(),
                                  current_mask.get_column_view(),
                                  _stream,
                                  _mr);
    } else {
      output = cudf::copy_if_else(current_then.get_column_view(),
                                  current_result.get_column_view(),
                                  current_mask.get_column_view(),
                                  _stream,
                                  _mr);
    }
    current_result = execute_result(std::move(output));
  }
  if (mode == execution_mode::AST) {
    // The caller wants an AST node. Materialize the CASE result into a temp column and return an
    // ast_result with a column_reference to it.
    std::unique_ptr<cudf::column> result_column;
    if (current_result.is_scalar()) {
      result_column = cudf::make_column_from_scalar(
        current_result.get_scalar(), _input_table.num_rows(), _stream, _mr);
    } else {
      result_column =
        std::make_unique<cudf::column>(current_result.get_column_view(), _stream, _mr);
    }
    return materialize_as_ast_column(std::move(result_column));
  }
  return current_result;
}
}  // namespace sirius
