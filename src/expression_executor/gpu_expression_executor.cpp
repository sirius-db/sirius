#include "expression_executor/gpu_expression_executor.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/helper.hpp"
#include "duckdb/common/types.hpp"
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

  /// DEBUG ///
  std::cout << "GpuExpressionExecutor received " << expressions.size() << " expressions\n";
  for (size_t i = 0; i < expressions.size(); ++i)
  {
    std::cout << "Expr[" << i << "] is " << (expressions[i] ? "valid" : "nullptr") << "\n";
  }

  for (const auto& expr : expressions)
  {
    AddExpression(*expr);
  }
}

void GpuExpressionExecutor::AddExpression(const Expression& expr)
{
  // Add the given expression to the list of expressions this executor is responsible for
  std::cout << "Adding expression: " << expr.ToString() << "\n";
  expressions.push_back(&expr);

  // Initialize the executor state of the expression and add to the list of executor states
  auto state = std::make_unique<GpuExpressionExecutorState>();
  Initialize(expr, *state);
  std::cout << "Done initializing state...\n";
  states.push_back(std::move(state));
  std::cout << "Expression added...\n";
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
  std::cout << "\n\nEXECUTING EXPRESSION\n";
  std::cout << "SETTING INPUT COLUMNS...\n";
  SetInputColumns(input_relation);
  std::cout << "DONE SETTING INPUT COLUMNS...\n";

  // Loop over expressions to execute
  for (idx_t i = 0; i < expressions.size(); ++i)
  {
    // If the expression is a reference, just pass it through
    if (expressions[i]->expression_class == ExpressionClass::BOUND_REF)
    {
      std::cout << "PASSING REFERENCE THROUGH...\n";
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

    // Transfer to output relation (zero copy)
    // auto result_view = result->view();
    // output_relation.columns[i] =
    //   GPUBufferManager::GetInstance().copyDataFromcuDFColumn(result_view, 0);
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
  if (expressions.size() != 1)
  {
    // Throw this away later...
    throw InvalidInputException("Select: only single expressions are supported!");
  }
  if (expressions[0]->return_type != LogicalType::BOOLEAN)
  {
    // Throw this away later
    throw InvalidInputException("Select: only logical expressions are allowed!");
  }
  std::cout << "\n\tEXECUTING FILTER\n\n";
  std::cout << "SETTING INPUT COLUMNS...\n";
  SetInputColumns(input_relation);
  std::cout << "DONE SETTING INPUT COLUMNS...\n";

  // If input count is zero, no-op
  if (input_count == 0)
  {
    // for (idx_t i = 0; i < input_columns.size(); ++i)
    // {
    //   output_relation.columns[i] = input_relation.columns[i];
    // }
    // output_relation.columns = input_relation.columns; // This should be fine
    std::cout << "FILTER DOING NOTHING\n";
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
  std::cout << "\tFILTER COUNT: " << count << "\n";

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
  D_ASSERT(output_column->data_wrapper.type ==
           convertLogicalTypeToColumnType(expressions[expr_idx]->return_type));

  std::cout << "Executing root expression " << expressions[expr_idx]->ToString() << "...\n";

  return Execute(*expressions[expr_idx], states[expr_idx]->root_state.get());
}

std::unique_ptr<cudf::column> GpuExpressionExecutor::Execute(const Expression& expr,
                                                             GpuExpressionState* state)
{
  std::cout << "Executing expression " << expr.ToString() << "...\n";

  switch (expr.GetExpressionClass())
  {
    case ExpressionClass::BOUND_BETWEEN:
      std::cout << "Executing between...\n";
      return Execute(expr.Cast<BoundBetweenExpression>(), state);
    case ExpressionClass::BOUND_CASE:
      std::cout << "Executing case...\n";
      return Execute(expr.Cast<BoundCaseExpression>(), state);
    case ExpressionClass::BOUND_CAST:
      std::cout << "Executing cast...\n";
      return Execute(expr.Cast<BoundCastExpression>(), state);
    case ExpressionClass::BOUND_COMPARISON:
      std::cout << "Executing comparison...\n";
      return Execute(expr.Cast<BoundComparisonExpression>(), state);
    case ExpressionClass::BOUND_CONJUNCTION:
      std::cout << "Executing conjunction...\n";
      return Execute(expr.Cast<BoundConjunctionExpression>(), state);
    case ExpressionClass::BOUND_CONSTANT:
      std::cout << "Executing constant...\n";
      return Execute(expr.Cast<BoundConstantExpression>(), state);
    case ExpressionClass::BOUND_FUNCTION:
      std::cout << "Executing function...\n";
      return Execute(expr.Cast<BoundFunctionExpression>(), state);
    case ExpressionClass::BOUND_OPERATOR:
      std::cout << "Executing operator...\n";
      return Execute(expr.Cast<BoundOperatorExpression>(), state);
    case ExpressionClass::BOUND_PARAMETER:
      throw NotImplementedException("Execute[BOUND_PARAMETER]: Not yet implemented!");
    case ExpressionClass::BOUND_REF:
      std::cout << "Executing reference...\n";
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