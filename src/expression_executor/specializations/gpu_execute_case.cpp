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
#include <expression_executor/gpu_expression_executor_state.hpp>

// duckdb
#include <duckdb/common/exception.hpp>
#include <duckdb/planner/expression/bound_case_expression.hpp>
#include <duckdb/planner/expression/bound_function_expression.hpp>

// cudf
#include <cudf/copying.hpp>
#include <cudf/reduction.hpp>

// We need to handle implicit error checks inserted as CASE statements by DuckDB
#define ERROR_FUNC_STR "error"

namespace sirius::experimental {
using execute_result = gpu_expression_executor::execute_result;
execute_result gpu_expression_executor::execute(duckdb::BoundCaseExpression const& expr,
                                                execution_mode mode)
{
  //===----------MATERIALIZE mode only (AST breaker)----------===//
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
  return current_result;
}
}  // namespace sirius::experimental

namespace duckdb {
namespace sirius {

std::unique_ptr<GpuExpressionState> GpuExpressionExecutor::InitializeState(
  const BoundCaseExpression& expr, GpuExpressionExecutorState& root)
{
  // auto result = make_uniq<GpuCaseExpressionState>(expr, root);
  auto result = std::make_unique<GpuExpressionState>(expr, root);
  for (auto& case_check : expr.case_checks) {
    result->AddChild(*case_check.when_expr);
    result->AddChild(*case_check.then_expr);
  }
  result->AddChild(*expr.else_expr);
  return result;
}

/**
 * Executing CASE expression is tricky, especially in device code. I do not follow DuckDB here,
 * which emits row ids when evaluating the WHEN expressions, selectively executes the THEN
 * expressions with the given row ids, scatters the results to the output, and then continues
 * evaluating the next WHEN with the leftover rowids. This has the effect of not doing wasted
 * computation for the ELSE expressions and succeeding WHEN expressions. However, compacting,
 * gathering, and scattering is more expensive on GPU, and CuDF does not provide conditional
 * execution APIs, which leaves me with executing the WHEN and THEN expressions on all input data.
 * Moreover, following CuDF in using unique_ptr semantics forces me to emit a new output column
 * for every case. However, if there are few CASE statements (as is the case in TPC-H), this should
 * be fine, if not optimal.
 */
std::unique_ptr<cudf::column> GpuExpressionExecutor::Execute(const BoundCaseExpression& expr,
                                                             GpuExpressionState* state)
{
  // First, execute the ELSE
  auto else_state     = state->child_states.back().get();
  auto current_output = Execute(*expr.else_expr, else_state);

  // Loop backwards, so that the THEN of the first true WHEN is copied to the output column
  auto num_checks = static_cast<int32_t>(
    expr.case_checks.size());  // This is sane, and needed for the descending loop index
  for (int32_t i = num_checks - 1; i >= 0; --i) {
    auto& case_check  = expr.case_checks[i];
    auto* check_state = state->child_states[2 * i].get();
    auto* then_state  = state->child_states[2 * i + 1].get();

    // Fist, execute the WHEN expression to get boolean array intermediate
    auto current_mask = Execute(*case_check.when_expr, check_state);

    // Check for error functions
    if (case_check.then_expr->GetExpressionClass() == ExpressionClass::BOUND_FUNCTION &&
        case_check.then_expr->Cast<BoundFunctionExpression>().function.name == ERROR_FUNC_STR) {
      // If the THEN is true anywhere, throw error()
      auto any_result = cudf::reduce(current_mask->view(),
                                     *cudf::make_any_aggregation<cudf::reduce_aggregation>(),
                                     cudf::data_type(cudf::type_id::BOOL8),
                                     execution_stream,
                                     resource_ref);
      if (static_cast<cudf::scalar_type_t<bool>*>(any_result.get())->value()) {
        // Assume that this arises for the stated error
        throw InternalException(
          "Execute[Case]: More than one row returned by a subquery used as "
          "an expression.");
      }
      continue;
    }

    // Otherwise, execute the THEN and selectively copy to the output
    auto current_then = Execute(*case_check.then_expr, then_state);
    current_output    = cudf::copy_if_else(current_then->view(),
                                        current_output->view(),
                                        current_mask->view(),
                                        execution_stream,
                                        resource_ref);
  }
  return current_output;
}

}  // namespace sirius
}  // namespace duckdb
