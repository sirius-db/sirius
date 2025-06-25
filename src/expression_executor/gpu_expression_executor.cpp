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

#include "expression_executor/gpu_expression_executor.hpp"
#include "cuda_stream_view.hpp"
#include "duckdb/common/exception.hpp"
#include "expression_executor/gpu_dispatcher.hpp"
#include "expression_executor/gpu_expression_executor_state.hpp"
#include "gpu_buffer_manager.hpp"
#include "gpu_columns.hpp"
#include "operator/gpu_materialize.hpp"

namespace duckdb
{
namespace sirius
{

GpuExpressionExecutor::GpuExpressionExecutor(const Expression& expr)
{
  AddExpression(expr);
}

GpuExpressionExecutor::GpuExpressionExecutor(const vector<unique_ptr<Expression>>& expressions)
{
  D_ASSERT(expressions.size() > 0);

  for (const auto& expr : expressions)
  {
    AddExpression(*expr);
  }
}

void GpuExpressionExecutor::AddExpression(const Expression& expr)
{
  // Add the given expression to the list of expressions this executor is responsible for
  expressions.push_back(&expr);

  // Initialize the executor state of the expression and add to the list of executor states
  auto state = std::make_unique<GpuExpressionExecutorState>();
  Initialize(expr, *state);
  states.push_back(std::move(state));
}

void GpuExpressionExecutor::ClearExpressions()
{
  states.clear();
  expressions.clear();
}

void GpuExpressionExecutor::Initialize(const Expression& expr, GpuExpressionExecutorState& state)
{
  // Set the executor of the executor state to this GpuExpressionExecutor
  state.executor = this;

  // Initialize the state of the expression
  state.root_state = InitializeState(expr, state);
}

void GpuExpressionExecutor::SetInputColumns(const GPUIntermediateRelation& input_relation)
{
  input_count           = 0;
  has_null_input_column = false;

  // Shallow copy the columns
  input_columns = input_relation.columns;

  // Set the input count
  if (input_columns.empty())
  {
    input_count = 1;
  }
  else
  {
    // All columns that are not null should have the same count
    for (const auto& col : input_columns)
    {
      const auto temp_count = col == nullptr ? 0
                              : col->row_ids == nullptr
                                ? static_cast<cudf::size_type>(col->column_length)
                                : static_cast<cudf::size_type>(col->row_id_count);
      if (temp_count > 0)
      {
        input_count = temp_count;
      }
      else
      {
        has_null_input_column = true;
      }
    }
  }
}

// Helper template function for HasNullLeaf()
template <typename ExpressionT>
bool GpuExpressionExecutor::HasNullLeafLoop(const ExpressionT& expr) const
{
  for (const auto& child : expr.children)
  {
    if (HasNullLeaf(*child))
    {
      return true;
    }
  }
  return false;
}

bool GpuExpressionExecutor::HasNullLeaf(const Expression& expr) const
{
  // Check if the expression is a null reference
  switch (expr.GetExpressionClass())
  {
    case ExpressionClass::BOUND_BETWEEN: {
      const auto& between_expr = expr.Cast<BoundBetweenExpression>();
      return HasNullLeaf(*between_expr.input) || HasNullLeaf(*between_expr.lower) ||
             HasNullLeaf(*between_expr.upper);
    }
    case ExpressionClass::BOUND_CASE: {
      const auto& case_expr = expr.Cast<BoundCaseExpression>();
      for (const auto& case_check : case_expr.case_checks)
      {
        if (HasNullLeaf(*case_check.when_expr) || HasNullLeaf(*case_check.then_expr))
        {
          return true;
        }
      }
      return HasNullLeaf(*case_expr.else_expr);
    }
    case ExpressionClass::BOUND_CAST: {
      const auto& cast_expr = expr.Cast<BoundCastExpression>();
      return HasNullLeaf(*cast_expr.child);
    }
    case ExpressionClass::BOUND_COMPARISON: {
      const auto& comp_expr = expr.Cast<BoundComparisonExpression>();
      return HasNullLeaf(*comp_expr.left) || HasNullLeaf(*comp_expr.right);
    }
    case ExpressionClass::BOUND_CONJUNCTION: {
      return HasNullLeafLoop(expr.Cast<BoundConjunctionExpression>());
    }
    case ExpressionClass::BOUND_CONSTANT: {
      // Base case
      return false;
    }
    case ExpressionClass::BOUND_FUNCTION: {
      return HasNullLeafLoop(expr.Cast<BoundFunctionExpression>());
    }
    case ExpressionClass::BOUND_OPERATOR: {
      return HasNullLeafLoop(expr.Cast<BoundOperatorExpression>());
    }
    case ExpressionClass::BOUND_REF: {
      // Base case
      const auto& ref_expr = expr.Cast<BoundReferenceExpression>();
      const auto& col      = input_columns[ref_expr.index];
      return col == nullptr || col->data_wrapper.data == nullptr;
    }
    default:
      throw InternalException("HasNullLeaf called on an expression [" + expr.ToString() +
                              "] with unsupported expression class!");
  }
  return false;
}

void GpuExpressionExecutor::Execute(const GPUIntermediateRelation& input_relation,
                                    GPUIntermediateRelation& output_relation,
                                    rmm::cuda_stream_view stream)
{
  D_ASSERT(expressions.size() == output_relation.columns.size());
  D_ASSERT(!expressions.empty());

  execution_stream = stream;
  SetInputColumns(input_relation);

  // Loop over expressions to execute
  for (idx_t i = 0; i < expressions.size(); ++i)
  {
    const auto& expr = *expressions[i];

    // If the expression is a reference, just pass it through
    if (expr.expression_class == ExpressionClass::BOUND_REF)
    {
      auto input_idx             = expr.Cast<BoundReferenceExpression>().index;
      output_relation.columns[i] = input_relation.columns[input_idx];
      continue;
    }

    // Make placeholder output column
    output_relation.columns[i] =
      make_shared_ptr<GPUColumn>(0, convertLogicalTypeToColumnType(expr.return_type), nullptr);

    // Skip execution if the input count is zero or if there is a null leaf
    if (input_count == 0 || (has_null_input_column && HasNullLeaf(expr)))
    {
      continue;
    }

    // Otherwise, execute the expression
    auto result = ExecuteExpression(i);

    // Cast the `result` from libcudf to `return_type` if `result` has different types.
    // E.g., `extract(year from col)` from libcudf returns int16_t but duckdb requires int64_t
    auto cudf_return_type = GpuExpressionState::GetCudfType(expressions[i]->return_type);
    if (result->type().id() != cudf_return_type.id())
    {
      result =
        cudf::cast(result->view(), cudf_return_type, execution_stream, resource_ref);
    }

    // Transfer to output relation (zero copy)
    output_relation.columns[i]->setFromCudfColumn(*result,
                                                  false, // How to know?
                                                  nullptr,
                                                  0,
                                                  &GPUBufferManager::GetInstance());
  }
}

void GpuExpressionExecutor::Select(GPUIntermediateRelation& input_relation,
                                   GPUIntermediateRelation& output_relation,
                                  rmm::cuda_stream_view stream)
{
  D_ASSERT(expressions.size() == 1);
  D_ASSERT(expressions[0]->return_type == LogicalType::BOOLEAN);

  execution_stream = stream;
  SetInputColumns(input_relation);

  // If the input count is zero or if there is a null leaf, just materialize
  if (input_count == 0 || (has_null_input_column && HasNullLeaf(*expressions[0])))
  {
    HandleMaterializeRowIDs(input_relation,
                            output_relation,
                            0,
                            nullptr,
                            &GPUBufferManager::GetInstance(),
                            true);
    return;
  }

  // Execute the boolean expression
  auto bitmap = ExecuteExpression(0);

  // Generate the selection vector
  auto [row_ids, count] = GpuDispatcher::DispatchSelect(bitmap->view(), resource_ref);

  // Compact
  HandleMaterializeRowIDs(input_relation,
                          output_relation,
                          count,
                          row_ids,
                          &GPUBufferManager::GetInstance(),
                          true);
}

std::unique_ptr<cudf::column> GpuExpressionExecutor::ExecuteExpression(idx_t expr_idx)
{
  D_ASSERT(expr_idx < expressions.size());

  return Execute(*expressions[expr_idx], states[expr_idx]->root_state.get());
}

std::unique_ptr<cudf::column> GpuExpressionExecutor::Execute(const Expression& expr,
                                                             GpuExpressionState* state)
{
  switch (expr.GetExpressionClass())
  {
    case ExpressionClass::BOUND_BETWEEN:
      return Execute(expr.Cast<BoundBetweenExpression>(), state);
    case ExpressionClass::BOUND_CASE:
      return Execute(expr.Cast<BoundCaseExpression>(), state);
    case ExpressionClass::BOUND_CAST:
      return Execute(expr.Cast<BoundCastExpression>(), state);
    case ExpressionClass::BOUND_COMPARISON:
      return Execute(expr.Cast<BoundComparisonExpression>(), state);
    case ExpressionClass::BOUND_CONJUNCTION:
      return Execute(expr.Cast<BoundConjunctionExpression>(), state);
    case ExpressionClass::BOUND_CONSTANT:
      return Execute(expr.Cast<BoundConstantExpression>(), state);
    case ExpressionClass::BOUND_FUNCTION:
      return Execute(expr.Cast<BoundFunctionExpression>(), state);
    case ExpressionClass::BOUND_OPERATOR:
      return Execute(expr.Cast<BoundOperatorExpression>(), state);
    case ExpressionClass::BOUND_PARAMETER:
      throw NotImplementedException("Execute[BOUND_PARAMETER]: Not yet implemented!");
    case ExpressionClass::BOUND_REF:
      return Execute(expr.Cast<BoundReferenceExpression>(), state);
    default:
      throw InternalException("Execute called on an expression [" + expr.ToString() +
                              "] with unsupported expression class!");
  }
}

std::unique_ptr<GpuExpressionState>
GpuExpressionExecutor::InitializeState(const Expression& expr, GpuExpressionExecutorState& state)
{
  switch (expr.GetExpressionClass())
  {
    case ExpressionClass::BOUND_BETWEEN:
      return InitializeState(expr.Cast<BoundBetweenExpression>(), state);
    case ExpressionClass::BOUND_CASE:
      return InitializeState(expr.Cast<BoundCaseExpression>(), state);
    case ExpressionClass::BOUND_CAST:
      return InitializeState(expr.Cast<BoundCastExpression>(), state);
    case ExpressionClass::BOUND_COMPARISON:
      return InitializeState(expr.Cast<BoundComparisonExpression>(), state);
    case ExpressionClass::BOUND_CONJUNCTION:
      return InitializeState(expr.Cast<BoundConjunctionExpression>(), state);
    case ExpressionClass::BOUND_CONSTANT:
      return InitializeState(expr.Cast<BoundConstantExpression>(), state);
    case ExpressionClass::BOUND_FUNCTION:
      return InitializeState(expr.Cast<BoundFunctionExpression>(), state);
    case ExpressionClass::BOUND_OPERATOR:
      return InitializeState(expr.Cast<BoundOperatorExpression>(), state);
    case ExpressionClass::BOUND_PARAMETER:
      throw NotImplementedException("InitializeState[BOUND_PARAMETER]: Not yet implemented!");
    case ExpressionClass::BOUND_REF:
      return InitializeState(expr.Cast<BoundReferenceExpression>(), state);
    default:
      throw InternalException("InitializeState: Unknown ExpressionClass!");
  }
}

} // namespace sirius
} // namespace duckdb