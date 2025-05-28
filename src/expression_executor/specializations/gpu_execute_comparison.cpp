#include "duckdb/common/exception.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "expression_executor/gpu_expression_executor.hpp"
#include "expression_executor/gpu_expression_executor_state.hpp"
#include <cudf/binaryop.hpp>
#include <cudf/scalar/scalar.hpp>
#include <memory>
#include <type_traits>

namespace duckdb
{
namespace sirius
{
//----------InitializeState----------//
std::unique_ptr<GpuExpressionState>
GpuExpressionExecutor::InitializeState(const BoundComparisonExpression& expr,
                                       GpuExpressionExecutorState& root)
{
  auto result = std::make_unique<GpuExpressionState>(expr, root);
  result->AddChild(*expr.left);
  result->AddChild(*expr.right);
  return std::move(result);
}

// Helper object to reduce bloat in Execute()
template <cudf::binary_operator ComparisonOp>
struct ComparisonDispatcher
{
  // The executor
  GpuExpressionExecutor& executor;

  // Constructor
  explicit ComparisonDispatcher(GpuExpressionExecutor& exec)
      : executor(exec)
  {}

  // Scalar comparison operator
  template <typename T>
  std::unique_ptr<cudf::column> DoScalarComparison(const cudf::column_view& left,
                                                   const T& right_value,
                                                   const cudf::data_type& return_type)
  {
    if constexpr (std::is_same_v<T, std::string>)
    {
      // Create a string scalar from the constant value
      auto string_scalar =
        cudf::string_scalar(right_value, true, cudf::get_default_stream(), executor.resource_ref);
      return cudf::binary_operation(left,
                                    string_scalar,
                                    ComparisonOp,
                                    return_type,
                                    cudf::get_default_stream(),
                                    executor.resource_ref);
    }
    else
    {
      // Create a numeric scalar from the constant value
      auto numeric_scalar =
        cudf::numeric_scalar(right_value, true, cudf::get_default_stream(), executor.resource_ref);
      return cudf::binary_operation(left,
                                    numeric_scalar,
                                    ComparisonOp,
                                    return_type,
                                    cudf::get_default_stream(),
                                    executor.resource_ref);
    }
  }

  // Dispatch operator
  std::unique_ptr<cudf::column> operator()(const BoundComparisonExpression& expr,
                                           GpuExpressionState* state)
  {
    D_ASSERT(expr.children.size() == 2);
    auto return_type = GpuExpressionState::GetCudfType(expr.return_type);

    // Resolve the children (DuckDB moves constants to the right comparator)
    auto left = executor.Execute(*expr.left, state->child_states[0].get());

    // If the right side is a constant, do not materialize in a column
    if (expr.right->GetExpressionClass() == ExpressionClass::BOUND_CONSTANT)
    {
      auto right_value = expr.right->Cast<BoundConstantExpression>().value;

      switch (GpuExpressionState::GetCudfType(expr.right->return_type).id())
      {
        case cudf::type_id::INT32:
          return DoScalarComparison<int32_t>(left->view(),
                                             right_value.GetValue<int32_t>(),
                                             return_type);
        case cudf::type_id::UINT64:
          return DoScalarComparison<uint64_t>(left->view(),
                                              right_value.GetValue<uint64_t>(),
                                              return_type);
        case cudf::type_id::FLOAT32:
          return DoScalarComparison<float_t>(left->view(),
                                             right_value.GetValue<float>(),
                                             return_type);
        case cudf::type_id::FLOAT64:
          return DoScalarComparison<double_t>(left->view(),
                                              right_value.GetValue<double>(),
                                              return_type);
        case cudf::type_id::BOOL8:
          return DoScalarComparison<bool>(left->view(), right_value.GetValue<bool>(), return_type);
        case cudf::type_id::STRING:
          return DoScalarComparison<std::string>(left->view(),
                                                 right_value.GetValue<std::string>(),
                                                 return_type);
        default:
          throw InternalException("Execute[Comparison]: Unsupported constant type for comparison!");
      }
    }

    // The right side is NOT a constant, so we need to execute it
    auto right = executor.Execute(*expr.right, state->child_states[1].get());

    // Execute the comparison
    return cudf::binary_operation(left->view(),
                                  right->view(),
                                  ComparisonOp,
                                  return_type,
                                  cudf::get_default_stream(),
                                  executor.resource_ref);
  }
};

//----------Execute[Comparison]----------//
std::unique_ptr<cudf::column> GpuExpressionExecutor::Execute(const BoundComparisonExpression& expr,
                                                             GpuExpressionState* state)
{
  auto return_type = GpuExpressionState::GetCudfType(expr.return_type);

  // Execute the comparison
  switch (expr.GetExpressionType())
  {
    case ExpressionType::COMPARE_EQUAL: {
      ComparisonDispatcher<cudf::binary_operator::EQUAL> dispatcher(*this);
      return dispatcher(expr, state);
    }
    case ExpressionType::COMPARE_NOTEQUAL: {
      ComparisonDispatcher<cudf::binary_operator::NOT_EQUAL> dispatcher(*this);
      return dispatcher(expr, state);
    }
    case ExpressionType::COMPARE_LESSTHAN: {
      ComparisonDispatcher<cudf::binary_operator::LESS> dispatcher(*this);
      return dispatcher(expr, state);
    }
    case ExpressionType::COMPARE_GREATERTHAN: {
      ComparisonDispatcher<cudf::binary_operator::GREATER> dispatcher(*this);
      return dispatcher(expr, state);
    }
    case ExpressionType::COMPARE_LESSTHANOREQUALTO: {
      ComparisonDispatcher<cudf::binary_operator::LESS_EQUAL> dispatcher(*this);
      return dispatcher(expr, state);
    }
    case ExpressionType::COMPARE_GREATERTHANOREQUALTO: {
      ComparisonDispatcher<cudf::binary_operator::GREATER_EQUAL> dispatcher(*this);
      return dispatcher(expr, state);
    }
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
