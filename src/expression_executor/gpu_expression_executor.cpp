#include "expression_executor/gpu_expression_executor.hpp"
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
  // Shallow copy the columns
  input_columns = input_relation.columns;

  // Set the input count
  if (input_columns.empty())
  {
    input_count = 1;
  }
  else
  {
    // All columns should have the same count
    const auto col = input_columns[0];
    // The input column may be null, in which case the expression evaluation should be a no-op
    input_count = col == nullptr            ? 0
                  : col->row_ids == nullptr ? static_cast<cudf::size_type>(col->column_length)
                                            : static_cast<cudf::size_type>(col->row_id_count);
  }
}

void GpuExpressionExecutor::Execute(const GPUIntermediateRelation& input_relation,
                                    GPUIntermediateRelation& output_relation)
{
  D_ASSERT(expressions.size() == output_relation.columns);
  D_ASSERT(!expressions.empty());
  SetInputColumns(input_relation);

  // Loop over expressions to execute
  for (idx_t i = 0; i < expressions.size(); ++i)
  {
    // If the expression is a reference, just pass it through
    if (expressions[i]->expression_class == ExpressionClass::BOUND_REF)
    {
      auto input_idx             = expressions[i]->Cast<BoundReferenceExpression>().index;
      output_relation.columns[i] = input_relation.columns[input_idx];
      continue;
    }

    // Make placeholder column
    output_relation.columns[i] =
      make_shared_ptr<GPUColumn>(0,
                                 convertLogicalTypeToColumnType(expressions[i]->return_type),
                                 nullptr);

    // If input count is zero, no-op
    if (input_count == 0)
    {
      continue;
    }

    // Execute the expression
    auto result = ExecuteExpression(i);

    // Cast the `result` from libcudf to `return_type` if `result` has different types.
    // E.g., `extract(year from col)` from libcudf returns int16_t but duckdb requires int64_t
    auto cudf_return_type = GpuExpressionState::GetCudfType(expressions[i]->return_type);
    if (result->type().id() != cudf_return_type.id()) {
      result = cudf::cast(result->view(),
                          cudf_return_type,
                          cudf::get_default_stream(),
                          resource_ref);
    }

    // Transfer to output relation (zero copy)
    output_relation.columns[i]->setFromCudfColumn(*result,
                                                  false,
                                                  nullptr,
                                                  0,
                                                  &GPUBufferManager::GetInstance());
  }
}

void GpuExpressionExecutor::Select(GPUIntermediateRelation& input_relation,
                                   GPUIntermediateRelation& output_relation)
{
  D_ASSERT(expressions.size() == 1);
  D_ASSERT(expressions[0]->return_type == LogicalType::BOOLEAN);

  SetInputColumns(input_relation);

  // If input count is zero, no-op
  if (input_count == 0)
  {
    HandleMaterializeRowIDs(input_relation,
                            output_relation,
                            input_count,
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