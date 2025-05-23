#include "duckdb/common/exception.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "expression_executor/gpu_expression_executor.hpp"
#include "expression_executor/gpu_expression_executor_state.hpp"
#include <cudf/binaryop.hpp>

namespace duckdb
{
namespace sirius
{

std::unique_ptr<GpuExpressionState>
GpuExpressionExecutor::InitializeState(const BoundComparisonExpression& expr,
                                       GpuExpressionExecutorState& root)
{
  auto result = std::make_unique<GpuExpressionState>(expr, root);
  result->AddChild(*expr.left);
  result->AddChild(*expr.right);
  return std::move(result);
}

std::unique_ptr<cudf::column> GpuExpressionExecutor::Execute(const BoundComparisonExpression& expr,
                                                             GpuExpressionState* state)
{
  auto return_type = GpuExpressionState::GetCudfType(expr.return_type);

  // Resolve the children
  auto left  = Execute(*expr.left, state->child_states[0].get());
  auto right = Execute(*expr.right, state->child_states[1].get());

  // Execute the comparison
  auto comparison_function =
    [&left, &right, &return_type, this](
      cudf::binary_operator comparison_op) -> std::unique_ptr<cudf::column> {
    return cudf::binary_operation(left->view(),
                                  right->view(),
                                  comparison_op,
                                  return_type,
                                  cudf::get_default_stream(),
                                  resource_ref);
  };
  switch (expr.GetExpressionType())
  {
    case ExpressionType::COMPARE_EQUAL:
      return comparison_function(cudf::binary_operator::EQUAL);
    case ExpressionType::COMPARE_NOTEQUAL:
      return comparison_function(cudf::binary_operator::NOT_EQUAL);
    case ExpressionType::COMPARE_LESSTHAN:
      return comparison_function(cudf::binary_operator::LESS);
    case ExpressionType::COMPARE_GREATERTHAN:
      return comparison_function(cudf::binary_operator::GREATER);
    case ExpressionType::COMPARE_LESSTHANOREQUALTO:
      return comparison_function(cudf::binary_operator::LESS_EQUAL);
    case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
      return comparison_function(cudf::binary_operator::GREATER_EQUAL);
    case ExpressionType::COMPARE_DISTINCT_FROM:
    case ExpressionType::COMPARE_NOT_DISTINCT_FROM:
      throw NotImplementedException("Execute[Comparison]: DISTINCT comparisons not yet "
                                    "implemented!");
    default:
      throw InternalException("Execute[Comparison]: Unknown comparison type!");
  }
}

} // namespace sirius
} // namespace duckdb
