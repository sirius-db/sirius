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
        cudf::string_scalar(right_value, true, executor.execution_stream, executor.resource_ref);
      return cudf::binary_operation(left,
                                    string_scalar,
                                    ComparisonOp,
                                    return_type,
                                    executor.execution_stream,
                                    executor.resource_ref);
    }
    else
    {
      // Create a numeric scalar from the constant value
      auto numeric_scalar =
        cudf::numeric_scalar(right_value, true, executor.execution_stream, executor.resource_ref);
      return cudf::binary_operation(left,
                                    numeric_scalar,
                                    ComparisonOp,
                                    return_type,
                                    executor.execution_stream,
                                    executor.resource_ref);
    }
  }

  // Scalar comparison operator for decimal types
  template <typename T>
  std::unique_ptr<cudf::column> DoScalarComparison(const cudf::column_view& left,
                                                   typename T::rep right_value,
                                                   numeric::scale_type scale,
                                                   const cudf::data_type& return_type) {
    std::unique_ptr<cudf::scalar> right_decimal_scalar;
    if (left.type().id() == cudf::type_to_id<T>()) {
      right_decimal_scalar = std::make_unique<cudf::fixed_point_scalar<T>>(
        right_value, scale, true, executor.execution_stream, executor.resource_ref);
    } else {
      // If types are different, need to construct `right_decimal_scalar` using `left.type()`
      switch (left.type().id()) {
        case cudf::type_id::DECIMAL32: {
          if (right_value > std::numeric_limits<int32_t>::max()) {
            throw InternalException("Cannot cast right decimal scalar to decimal32, value greater than INT32_MAX");
          }
          right_decimal_scalar = std::make_unique<cudf::fixed_point_scalar<numeric::decimal32>>(
            static_cast<int32_t>(right_value), scale, true, executor.execution_stream, executor.resource_ref);
          break;
        }
        case cudf::type_id::DECIMAL64: {
          if (right_value > std::numeric_limits<int64_t>::max()) {
            throw InternalException("Cannot cast right decimal scalar to decimal64, value greater than INT64_MAX");
          }
          right_decimal_scalar = std::make_unique<cudf::fixed_point_scalar<numeric::decimal64>>(
            static_cast<int64_t>(right_value), scale, true, executor.execution_stream, executor.resource_ref);
          break;
        }
        case cudf::type_id::DECIMAL128: {
          right_decimal_scalar = std::make_unique<cudf::fixed_point_scalar<numeric::decimal128>>(
            static_cast<__int128_t>(right_value), scale, true, executor.execution_stream, executor.resource_ref);
          break;
        }
        default:
          throw InternalException("Left column is not decimal with right decimal constant in `DoRightScalarBinaryOp`: %d",
                                  static_cast<int>(left.type().id()));
      }
    }
    return cudf::binary_operation(left,
                                  *right_decimal_scalar,
                                  ComparisonOp,
                                  return_type,
                                  executor.execution_stream,
                                  executor.resource_ref);
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
        case cudf::type_id::INT16:
          return DoScalarComparison<int16_t>(left->view(),
                                             right_value.GetValue<int16_t>(),
                                             return_type);
        case cudf::type_id::INT32:
          return DoScalarComparison<int32_t>(left->view(),
                                             right_value.GetValue<int32_t>(),
                                             return_type);
        case cudf::type_id::INT64:
          return DoScalarComparison<int64_t>(left->view(),
                                             right_value.GetValue<int64_t>(),
                                             return_type);
        case cudf::type_id::FLOAT32:
          return DoScalarComparison<float_t>(left->view(),
                                             right_value.GetValue<float_t>(),
                                             return_type);
        case cudf::type_id::FLOAT64:
          return DoScalarComparison<double_t>(left->view(),
                                              right_value.GetValue<double_t>(),
                                              return_type);
        case cudf::type_id::BOOL8:
          return DoScalarComparison<bool>(left->view(), right_value.GetValue<bool>(), return_type);
        case cudf::type_id::STRING:
          return DoScalarComparison<std::string>(left->view(),
                                                 right_value.GetValue<std::string>(),
                                                 return_type);
        case cudf::type_id::DECIMAL32:
          // cudf decimal type uses negative scale, same for below
          return DoScalarComparison<numeric::decimal32>(
            left->view(), right_value.GetValueUnsafe<int32_t>(),
            numeric::scale_type{-duckdb::DecimalType::GetScale(right_value.type())},
            return_type);
        case cudf::type_id::DECIMAL64:
          return DoScalarComparison<numeric::decimal64>(
            left->view(), right_value.GetValueUnsafe<int64_t>(),
            numeric::scale_type{-duckdb::DecimalType::GetScale(right_value.type())},
            return_type);
        case cudf::type_id::DECIMAL128: {
          duckdb::hugeint_t hugeint_value = right_value.GetValueUnsafe<duckdb::hugeint_t>();
          return DoScalarComparison<numeric::decimal128>(
            left->view(), (__int128_t(hugeint_value.upper) << 64) | hugeint_value.lower,
            numeric::scale_type{-duckdb::DecimalType::GetScale(right_value.type())},
            return_type);
        }
        default:
          throw InternalException("Execute[Comparison]: Unsupported constant type for comparison: %d!",
            static_cast<int>(GpuExpressionState::GetCudfType(expr.right->return_type).id()));
      }
    }

    // The right side is NOT a constant, so we need to execute it
    auto right = executor.Execute(*expr.right, state->child_states[1].get());

    // Execute the comparison
    return cudf::binary_operation(left->view(),
                                  right->view(),
                                  ComparisonOp,
                                  return_type,
                                  executor.execution_stream,
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
