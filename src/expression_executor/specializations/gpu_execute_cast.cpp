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
#include <duckdb/planner/expression/bound_cast_expression.hpp>

// cudf
#include <cudf/unary.hpp>

namespace sirius::experimental {
using execute_result = gpu_expression_executor::execute_result;

execute_result gpu_expression_executor::execute(duckdb::BoundCastExpression const& expr,
                                                execution_mode mode)
{
  auto const ast_supported = std::find(supported_ast_cast_types.begin(),
                                       supported_ast_cast_types.end(),
                                       expr.return_type.id()) != supported_ast_cast_types.end();

  if (ast_supported && (mode == execution_mode::AST || count_ast_ops(expr) >= _min_ast_size)) {
    auto cast_type_switch = [](duckdb::LogicalTypeId type_id) -> cudf::ast::ast_operator {
      using enum duckdb::LogicalTypeId;
      switch (type_id) {
        case UBIGINT: return cudf::ast::ast_operator::CAST_TO_UINT64;
        case BIGINT: return cudf::ast::ast_operator::CAST_TO_INT64;
        case DOUBLE: return cudf::ast::ast_operator::CAST_TO_FLOAT64;
        default:
          throw duckdb::InternalException(
            "[gpu_expression_executor] execute called on a CAST expression with unsupported return "
            "type for AST execution: {}",
            duckdb::LogicalTypeIdToString(type_id));
      }
    };

    auto child     = execute(*expr.child, execution_mode::AST);
    auto cast_expr = _ast_tree.emplace<cudf::ast::operation>(
      cast_type_switch(expr.return_type.id()), child.get_expr());

    if (mode == execution_mode::AST) {
      //===----------1: AST Mode----------===//
      return execute_result(
        ast_result(cast_expr, child.get_temp_scalar_indices(), child.get_temp_column_indices()));
    }
    //===----------2: MATERIALIZE Mode, evaluate node with AST----------===//
    // JIT compile the AST Subtree
    auto result_column = execute_ast(cast_expr);

    // Release consumed temporaries
    release_temporaries(child.get_temp_scalar_indices(), child.get_temp_column_indices());
    return execute_result(std::move(result_column));
  }

  //===----------3: MATERIALIZE Mode, evaluate node with unary/binary ops----------===//
  auto const return_type = GetCudfType(expr.return_type);
  auto child             = execute(*expr.child, execution_mode::MATERIALIZE);
  auto result_column     = cudf::cast(child.get_column_view(), return_type, _stream, _mr);
  return execute_result(std::move(result_column));
}

}  // namespace sirius::experimental

namespace duckdb {
namespace sirius {

std::unique_ptr<GpuExpressionState> GpuExpressionExecutor::InitializeState(
  const BoundCastExpression& expr, GpuExpressionExecutorState& root)
{
  auto result = make_uniq<GpuExpressionState>(expr, root);
  result->AddChild(*expr.child);
  return std::move(result);
}

// Note that constants are CASTed before they reach the execution engine
std::unique_ptr<cudf::column> GpuExpressionExecutor::Execute(const BoundCastExpression& expr,
                                                             GpuExpressionState* state)
{
  auto return_type = GetCudfType(expr.return_type);

  // Resolve the child
  auto* child_state = state->child_states[0].get();
  auto child        = Execute(*expr.child, child_state);

  // Execute the cast
  return cudf::cast(child->view(), return_type, execution_stream, resource_ref);
}

}  // namespace sirius
}  // namespace duckdb
