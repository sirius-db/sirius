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

#include "duckdb/common/exception.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "expression_executor/gpu_expression_executor.hpp"
#include "expression_executor/gpu_expression_executor_state.hpp"
#include <cudf/binaryop.hpp>

namespace duckdb
{
namespace sirius
{

std::unique_ptr<GpuExpressionState>
GpuExpressionExecutor::InitializeState(const BoundConjunctionExpression& expr,
                                       GpuExpressionExecutorState& root)
{
  auto result = make_uniq<GpuExpressionState>(expr, root);
  for (auto& child : expr.children)
  {
    result->AddChild(*child);
  }
  return std::move(result);
}

std::unique_ptr<cudf::column> GpuExpressionExecutor::Execute(const BoundConjunctionExpression& expr,
                                                             GpuExpressionState* state)
{
  auto return_type = GpuExpressionState::GetCudfType(expr.return_type);

  // Resolve the children incrementally into the output
  std::unique_ptr<cudf::column> output_column;
  for (idx_t i = 0; i < expr.children.size(); i++)
  {
    D_ASSERT(state->child_states[i]->expr.return_type == LogicalType::BOOLEAN;);

    auto current_result = Execute(*expr.children[i], state->child_states[i].get());

    if (i == 0)
    {
      // Nothing to compare
      output_column = std::move(current_result);
    }
    else
    {
      // AND/OR current result with output collecte so far
      switch (expr.GetExpressionType())
      {
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
        default:
          throw InternalException("Execute[Conjunction]: Unknown conjunction type!");
      }
    }
  }

  return std::move(output_column);
}

} // namespace sirius
} // namespace duckdb
