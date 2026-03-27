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
#include <duckdb/common/typedefs.hpp>
#include <duckdb/planner/expression/bound_conjunction_expression.hpp>

// cudf
#include <cudf/binaryop.hpp>

namespace sirius::experimental {
using execute_result = gpu_expression_executor::execute_result;

execute_result gpu_expression_executor::execute(duckdb::BoundConjunctionExpression const& expr,
                                                execution_mode mode)
{
  if (mode == execution_mode::AST || count_ast_ops(expr) >= _min_ast_size) {
    auto conjunction_type_switch_ast =
      [](duckdb::BoundConjunctionExpression const& expr) -> cudf::ast::ast_operator {
      using enum duckdb::ExpressionType;
      switch (expr.GetExpressionType()) {
        case CONJUNCTION_AND: return cudf::ast::ast_operator::LOGICAL_AND;
        case CONJUNCTION_OR: return cudf::ast::ast_operator::LOGICAL_OR;
        default:
          throw duckdb::InternalException(
            "[gpu_expression_executor:conjunction] unrecognized conjunction type {}",
            static_cast<int>(expr.GetExpressionType()));
      }
    };

    auto output = execute(*expr.children[0], execution_mode::AST);

    for (idx_t i = 1; i < expr.children.size(); i++) {
      auto child = execute(*expr.children[i], execution_mode::AST);

      auto output_expr = _ast_tree.emplace<cudf::ast::operation>(
        conjunction_type_switch_ast(expr), output.get_expr(), child.get_expr());
      output = execute_result(
        ast_result(output_expr,
                   {output.get_temp_scalar_indices(), child.get_temp_scalar_indices()},
                   {output.get_temp_column_indices(), child.get_temp_column_indices()}));
    }

    if (mode == execution_mode::AST) {
      //===----------1: AST Mode----------===//
      return std::move(output);
    }
    //===----------2: MATERIALIZE Mode, evaluate node with AST----------===//
    // JIT compile the AST Subtree
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
        throw duckdb::InternalException(
          "[gpu_expression_executor:conjunction] unrecognized conjunction type {}",
          static_cast<int>(expr.GetExpressionType()));
    }
  };

  auto const output_type = GetCudfType(expr.return_type);

  // Resolve the children incrementally into the output
  auto output = execute(*expr.children[0], execution_mode::MATERIALIZE);
  // DuckDB should prune all scalar conjuncts away
  D_ASSERT(!output.is_scalar());
  for (idx_t i = 1; i < expr.children.size(); i++) {
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
  return std::move(output);
}

}  // namespace sirius::experimental

namespace duckdb {
namespace sirius {

std::unique_ptr<GpuExpressionState> GpuExpressionExecutor::InitializeState(
  const BoundConjunctionExpression& expr, GpuExpressionExecutorState& root)
{
  auto result = make_uniq<GpuExpressionState>(expr, root);
  for (auto& child : expr.children) {
    result->AddChild(*child);
  }
  return std::move(result);
}

std::unique_ptr<cudf::column> GpuExpressionExecutor::Execute(const BoundConjunctionExpression& expr,
                                                             GpuExpressionState* state)
{
  auto return_type = GetCudfType(expr.return_type);

  // Resolve the children incrementally into the output
  std::unique_ptr<cudf::column> output_column;
  for (idx_t i = 0; i < expr.children.size(); i++) {
    D_ASSERT(state->child_states[i]->expr.return_type == LogicalType::BOOLEAN);

    auto current_result = Execute(*expr.children[i], state->child_states[i].get());

    if (i == 0) {
      // Nothing to compare
      output_column = std::move(current_result);
    } else {
      // AND/OR current result with output collecte so far
      switch (expr.GetExpressionType()) {
        case ExpressionType::CONJUNCTION_AND:
          output_column = cudf::binary_operation(current_result->view(),
                                                 output_column->view(),
                                                 cudf::binary_operator::LOGICAL_AND,
                                                 return_type,
                                                 execution_stream,
                                                 resource_ref);
          break;
        case ExpressionType::CONJUNCTION_OR:
          output_column = cudf::binary_operation(current_result->view(),
                                                 output_column->view(),
                                                 cudf::binary_operator::LOGICAL_OR,
                                                 return_type,
                                                 execution_stream,
                                                 resource_ref);
          break;
        default: throw InternalException("Execute[Conjunction]: Unknown conjunction type!");
      }
    }
  }

  return std::move(output_column);
}

}  // namespace sirius
}  // namespace duckdb
