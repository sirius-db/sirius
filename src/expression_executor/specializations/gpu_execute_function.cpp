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

#include "duckdb/common/assert.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "expression_executor/gpu_dispatcher.hpp"
#include "expression_executor/gpu_expression_executor.hpp"
#include "expression_executor/gpu_expression_executor_state.hpp"
#include "gpu_physical_strings_matching.hpp"
#include "log/logging.hpp"
#include <cudf/binaryop.hpp>
#include <cudf/datetime.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/slice.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/unary.hpp>
#include <string>

namespace duckdb
{
namespace sirius
{

// There has to be a better way to extract the function semantics than string comparison!
//----------Function Strings----------//
#define ADD_FUNC_STR "+"
#define SUB_FUNC_STR "-"
#define MUL_FUNC_STR "*"
#define DIV_FUNC_STR "/"
#define INT_DIV_FUNC_STR "//"
#define MOD_FUNC_STR "%"
#define SUBSTRING_FUNC_STR_1 "substring"
#define SUBSTRING_FUNC_STR_2 "substr"
#define LIKE_FUNC_STR "~~"
#define NOT_LIKE_FUNC_STR "!~~"
#define CONTAINS_FUNC_STR "contains"
#define PREFIX_FUNC_STR "prefix"
#define SUFFIX_FUNC_STR "suffix"
#define YEAR_FUNC_STR "year"
#define MONTH_FUNC_STR "month"
#define DAY_FUNC_STR "day"
#define ERROR_FUNC_STR "error"

#define SPLIT_DELIMITER "%"

//----------InitializeState----------//
std::unique_ptr<GpuExpressionState>
GpuExpressionExecutor::InitializeState(const BoundFunctionExpression& expr,
                                       GpuExpressionExecutorState& root)
{
  auto result = std::make_unique<GpuExpressionState>(expr, root);
  for (auto& child : expr.children)
  {
    result->AddChild(*child);
  }
  return std::move(result);
}

//----------StringMatchingDispatcher----------//
// Helper template functor for string matching operations to reduce bloat in Execute()
template <StringMatchingType MatchType, bool UseCudf>
struct StringMatchingDispatcher
{
  // The executor
  GpuExpressionExecutor& executor;

  // Constructor
  explicit StringMatchingDispatcher(GpuExpressionExecutor& exec)
      : executor(exec)
  {}

  // Dispatch operator
  std::unique_ptr<cudf::column> operator()(const BoundFunctionExpression& expr,
                                           GpuExpressionState* state)
  {
    D_ASSERT(expr.children.size() == 2);
    D_ASSERT(expr.children[1]->GetExpressionClass() == ExpressionClass::BOUND_CONSTANT);

    auto input                 = executor.Execute(*expr.children[0], state->child_states[0].get());
    const auto& match_str_expr = expr.children[1]->Cast<BoundConstantExpression>();
    const auto& match_str      = match_str_expr.value.GetValue<std::string>();

    // For the following `MatchType`, we only support using cudf
    if constexpr (MatchType == StringMatchingType::SUFFIX)
    {
      const auto match_str_scalar =
        cudf::string_scalar(match_str, true, executor.execution_stream, executor.resource_ref);
      return cudf::strings::ends_with(input->view(),
                                      match_str_scalar,
                                      executor.execution_stream,
                                      executor.resource_ref);
    }

    // For the following `MatchType`, we support using both cudf or the one implmented by Sirius
    if constexpr (UseCudf)
    {
      //----------Using CuDF----------//
      cudf::strings_column_view input_view(input->view());
      if constexpr (MatchType == StringMatchingType::LIKE ||
                    MatchType == StringMatchingType::NOT_LIKE)
      {
        std::vector<std::string> match_terms = string_split(match_str, SPLIT_DELIMITER);

        auto like = cudf::strings::like(cudf::strings_column_view(input_view),
                                        cudf::string_scalar(match_str),
                                        cudf::string_scalar(""),
                                        executor.execution_stream,
                                        executor.resource_ref);

        // LIKE or NOT LIKE?
        if constexpr (MatchType == StringMatchingType::LIKE)
        {
          return std::move(like);
        }
        else
        {
          // Negate the match result
          return cudf::unary_operation(like->view(),
                                       cudf::unary_operator::NOT,
                                       executor.execution_stream,
                                       executor.resource_ref);
        }
      }
      else if constexpr (MatchType == StringMatchingType::CONTAINS)
      {
        const auto match_str_scalar =
          cudf::string_scalar(match_str, true, executor.execution_stream, executor.resource_ref);
        return cudf::strings::contains(input_view,
                                       match_str_scalar,
                                       executor.execution_stream,
                                       executor.resource_ref);
      }
      else if constexpr (MatchType == StringMatchingType::PREFIX)
      {
        const auto match_str_scalar =
          cudf::string_scalar(match_str, true, executor.execution_stream, executor.resource_ref);
        return cudf::strings::starts_with(input_view,
                                          match_str_scalar,
                                          executor.execution_stream,
                                          executor.resource_ref);
      }
      else {
        throw NotImplementedException("Unsupported StringMatchingType when using cudf: %d",
                                      static_cast<int>(MatchType));
      }
    }
    else
    {
      //----------Using Sirius----------//
      return GpuDispatcher::DispatchStringMatching<MatchType>(input->view(),
                                                              match_str,
                                                              executor.resource_ref);
    }
  }
};

//----------NumericBinaryFunctionDispatcher----------//
template <cudf::binary_operator BinOp>
struct NumericBinaryFunctionDispatcher
{
  // The executor
  GpuExpressionExecutor& executor;

  // Constructor
  explicit NumericBinaryFunctionDispatcher(GpuExpressionExecutor& exec)
      : executor(exec)
  {}

  // Left scalar binary operator for numeric types
  template <typename T>
  std::unique_ptr<cudf::column> DoLeftScalarBinaryOp(const T& left_value,
                                                     const cudf::column_view& right,
                                                     const cudf::data_type& return_type)
  {
    auto left_numeric_scalar =
      cudf::numeric_scalar(left_value, true, executor.execution_stream, executor.resource_ref);
    return cudf::binary_operation(left_numeric_scalar,
                                  right,
                                  BinOp,
                                  return_type,
                                  executor.execution_stream,
                                  executor.resource_ref);
  }

  // Left scalar binary operator for decimal types
  template <typename T>
  std::unique_ptr<cudf::column> DoLeftScalarBinaryOp(typename T::rep left_value,
                                                     numeric::scale_type scale,
                                                     const cudf::column_view& right,
                                                     const cudf::data_type& return_type)
  {
    std::unique_ptr<cudf::scalar> left_decimal_scalar;
    if (right.type().id() == cudf::type_to_id<T>()) {
      left_decimal_scalar = std::make_unique<cudf::fixed_point_scalar<T>>(
        left_value, scale, true, executor.execution_stream, executor.resource_ref);
    } else {
      // If types are different, need to construct `left_decimal_scalar` using `right.type()`
      switch (right.type().id()) {
        case cudf::type_id::DECIMAL32: {
          if (left_value > std::numeric_limits<int32_t>::max()) {
            throw InternalException("Cannot cast left decimal scalar to decimal32, value greater than INT32_MAX");
          }
          left_decimal_scalar = std::make_unique<cudf::fixed_point_scalar<numeric::decimal32>>(
            static_cast<int32_t>(left_value), scale, true, executor.execution_stream, executor.resource_ref);
          break;
        }
        case cudf::type_id::DECIMAL64: {
          if (left_value > std::numeric_limits<int64_t>::max()) {
            throw InternalException("Cannot cast left decimal scalar to decimal64, value greater than INT64_MAX");
          }
          left_decimal_scalar = std::make_unique<cudf::fixed_point_scalar<numeric::decimal64>>(
            static_cast<int64_t>(left_value), scale, true, executor.execution_stream, executor.resource_ref);
          break;
        }
        case cudf::type_id::DECIMAL128: {
          left_decimal_scalar = std::make_unique<cudf::fixed_point_scalar<numeric::decimal128>>(
            static_cast<__int128_t>(left_value), scale, true, executor.execution_stream, executor.resource_ref);
          break;
        }
        default:
          throw InternalException("Right column is not decimal with left decimal constant in `DoLeftScalarBinaryOp`: %d",
                                  static_cast<int>(right.type().id()));
      }
    }
    return cudf::binary_operation(*left_decimal_scalar,
                                  right,
                                  BinOp,
                                  return_type,
                                  executor.execution_stream,
                                  executor.resource_ref);
  }

  // Right scalar binary operator for numeric types
  template <typename T>
  std::unique_ptr<cudf::column> DoRightScalarBinaryOp(const cudf::column_view& left,
                                                      const T& right_value,
                                                      const cudf::data_type& return_type)
  {
    auto right_numeric_scalar =
      cudf::numeric_scalar(right_value, true, executor.execution_stream, executor.resource_ref);
    return cudf::binary_operation(left,
                                  right_numeric_scalar,
                                  BinOp,
                                  return_type,
                                  executor.execution_stream,
                                  executor.resource_ref);
  }

  // Right scalar binary operator for decimal types
  template <typename T>
  std::unique_ptr<cudf::column> DoRightScalarBinaryOp(const cudf::column_view& left,
                                                      typename T::rep right_value,
                                                      numeric::scale_type scale,
                                                      const cudf::data_type& return_type)
  {
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
                                  BinOp,
                                  return_type,
                                  executor.execution_stream,
                                  executor.resource_ref);
  }

  // Dispatch operator
  std::unique_ptr<cudf::column> operator()(const BoundFunctionExpression& expr,
                                           GpuExpressionState* state)
  {
    D_ASSERT(expr.children.size() == 2);
    const auto& return_type = GpuExpressionState::GetCudfType(expr.return_type);

    // Resolve children
    if (expr.children[0]->GetExpressionClass() == ExpressionClass::BOUND_CONSTANT)
    {
      // LHS is a constant, so skip its column materialization
      const auto& left_value = expr.children[0]->Cast<BoundConstantExpression>().value;
      const auto& right      = executor.Execute(*expr.children[1], state->child_states[1].get());

      auto cudf_type = GpuExpressionState::GetCudfType(expr.children[0]->return_type);
      switch (cudf_type.id())
      {
        case cudf::type_id::INT32:
          return DoLeftScalarBinaryOp(left_value.GetValue<int32_t>(), right->view(), return_type);
        case cudf::type_id::UINT64:
          return DoLeftScalarBinaryOp(left_value.GetValue<uint64_t>(), right->view(), return_type);
        case cudf::type_id::FLOAT32:
          return DoLeftScalarBinaryOp(left_value.GetValue<float_t>(), right->view(), return_type);
        case cudf::type_id::FLOAT64:
          return DoLeftScalarBinaryOp(left_value.GetValue<double_t>(), right->view(), return_type);
        case cudf::type_id::DECIMAL32:
          // cudf decimal type uses negative scale, same for below
          return DoLeftScalarBinaryOp<numeric::decimal32>(
            left_value.GetValueUnsafe<int32_t>(),
            numeric::scale_type{-duckdb::DecimalType::GetScale(left_value.type())},
            right->view(), return_type);
        case cudf::type_id::DECIMAL64:
          return DoLeftScalarBinaryOp<numeric::decimal64>(
            left_value.GetValueUnsafe<int64_t>(),
            numeric::scale_type{-duckdb::DecimalType::GetScale(left_value.type())},
            right->view(), return_type);
        case cudf::type_id::DECIMAL128: {
          duckdb::hugeint_t hugeint_value = left_value.GetValueUnsafe<duckdb::hugeint_t>();
          return DoLeftScalarBinaryOp<numeric::decimal128>(
            (__int128_t(hugeint_value.upper) << 64) | hugeint_value.lower,
            numeric::scale_type{-duckdb::DecimalType::GetScale(left_value.type())},
            right->view(), return_type);
        }
        case cudf::type_id::BOOL8:
          throw NotImplementedException("Execute[Function]: Boolean types not supported for "
                                        "numeric binary operations!");
        default:
          throw InternalException("Execute[Function]: Unknown cudf type of left constant for binary operation: %d",
                                  static_cast<int>(cudf_type.id()));
      }
    }
    else if (expr.children[1]->GetExpressionClass() == ExpressionClass::BOUND_CONSTANT)
    {
      // RHS is a constant, so skip its column materialization
      const auto& right_value = expr.children[1]->Cast<BoundConstantExpression>().value;
      const auto& left        = executor.Execute(*expr.children[0], state->child_states[0].get());

      auto cudf_type = GpuExpressionState::GetCudfType(expr.children[1]->return_type);
      switch (cudf_type.id())
      {
        case cudf::type_id::INT32:
          return DoRightScalarBinaryOp(left->view(), right_value.GetValue<int32_t>(), return_type);
        case cudf::type_id::UINT64:
          return DoRightScalarBinaryOp(left->view(), right_value.GetValue<uint64_t>(), return_type);
        case cudf::type_id::FLOAT32:
          return DoRightScalarBinaryOp(left->view(), right_value.GetValue<float_t>(), return_type);
        case cudf::type_id::FLOAT64:
          return DoRightScalarBinaryOp(left->view(), right_value.GetValue<double_t>(), return_type);
        case cudf::type_id::DECIMAL32:
          // cudf decimal type uses negative scale, same for below
          return DoRightScalarBinaryOp<numeric::decimal32>(
            left->view(), right_value.GetValueUnsafe<int32_t>(),
            numeric::scale_type{-duckdb::DecimalType::GetScale(right_value.type())},
            return_type);
        case cudf::type_id::DECIMAL64:
          return DoRightScalarBinaryOp<numeric::decimal64>(
            left->view(), right_value.GetValueUnsafe<int64_t>(),
            numeric::scale_type{-duckdb::DecimalType::GetScale(right_value.type())},
            return_type);
        case cudf::type_id::DECIMAL128: {
          duckdb::hugeint_t hugeint_value = right_value.GetValueUnsafe<duckdb::hugeint_t>();
          return DoRightScalarBinaryOp<numeric::decimal128>(
            left->view(), (__int128_t(hugeint_value.upper) << 64) | hugeint_value.lower,
            numeric::scale_type{-duckdb::DecimalType::GetScale(right_value.type())},
            return_type);
        }
        case cudf::type_id::BOOL8:
          throw NotImplementedException("Execute[Function]: Boolean types not supported for "
                                        "numeric binary operations!");
        default:
          throw InternalException("Execute[Function]: Unknown cudf type of right constant for binary operation: %d",
                                  static_cast<int>(cudf_type.id()));
      }
    }

    // NEITHER side is a constant, so we need to execute both children
    auto left  = executor.Execute(*expr.children[0], state->child_states[0].get());
    auto right = executor.Execute(*expr.children[1], state->child_states[1].get());

    // Execute the binary operation
    return cudf::binary_operation(left->view(),
                                  right->view(),
                                  BinOp,
                                  GpuExpressionState::GetCudfType(expr.return_type),
                                  executor.execution_stream,
                                  executor.resource_ref);
  }
};

//----------DatetimeExtractFunctionDispatcher----------//
template <cudf::datetime::datetime_component COMP>
struct DatetimeExtractFunctionDispatcher
{
  // The executor
  GpuExpressionExecutor& executor;

  // Constructor
  explicit DatetimeExtractFunctionDispatcher(GpuExpressionExecutor& exec)
      : executor(exec)
  {}

  // Dispatch operator
  std::unique_ptr<cudf::column> operator()(const BoundFunctionExpression& expr,
                                           GpuExpressionState* state)
  {
    D_ASSERT(expr.children.size() == 1);
    auto input = executor.Execute(*expr.children[0], state->child_states[0].get());
    return cudf::datetime::extract_datetime_component(input->view(),
                                                      COMP,
                                                      executor.execution_stream,
                                                      executor.resource_ref);
  }
};

//----------Execute----------//
std::unique_ptr<cudf::column> GpuExpressionExecutor::Execute(const BoundFunctionExpression& expr,
                                                             GpuExpressionState* state)
{
  const auto& function_expression_state = state->Cast<GpuExpressionState>();
  const auto& func_str                  = expr.function.name;

  //----------Numeric Binary Functions----------//
  if (func_str == ADD_FUNC_STR)
  {
    NumericBinaryFunctionDispatcher<cudf::binary_operator::ADD> binary_function(*this);
    return binary_function(expr, state);
  }
  else if (func_str == SUB_FUNC_STR)
  {
    NumericBinaryFunctionDispatcher<cudf::binary_operator::SUB> binary_function(*this);
    return binary_function(expr, state);
  }
  else if (func_str == MUL_FUNC_STR)
  {
    NumericBinaryFunctionDispatcher<cudf::binary_operator::MUL> binary_function(*this);
    return binary_function(expr, state);
  }
  else if (func_str == DIV_FUNC_STR || func_str == INT_DIV_FUNC_STR)
  {
    // For non-integer division on integer types, DuckDB inserts a CAST
    NumericBinaryFunctionDispatcher<cudf::binary_operator::DIV> binary_function(*this);
    return binary_function(expr, state);
  }
  else if (func_str == MOD_FUNC_STR)
  {
    NumericBinaryFunctionDispatcher<cudf::binary_operator::MOD> binary_function(*this);
    return binary_function(expr, state);
  }
  else if (func_str == ERROR_FUNC_STR)
  {
    throw InternalException("Execute[Function]: error() should be handled by Execute[Case]!");
  }

  //----------String Functions----------//
  if (func_str == SUBSTRING_FUNC_STR_1 || func_str == SUBSTRING_FUNC_STR_2)
  {
    // We assume the start and len arguments are constants (seems to be the case in DuckDB)
    D_ASSERT(expr.children.size() == 3);
    D_ASSERT(expr.children[1]->GetExpressionClass() == ExpressionClass::BOUND_CONSTANT);
    D_ASSERT(expr.children[2]->GetExpressionClass() == ExpressionClass::BOUND_CONSTANT);

    const auto& start_expr = expr.children[1]->Cast<BoundConstantExpression>();
    const auto& len_expr   = expr.children[2]->Cast<BoundConstantExpression>();

    auto input = Execute(*expr.children[0], state->child_states[0].get());

    if constexpr (use_cudf)
    {
      cudf::strings_column_view input_view(input->view());
      const auto cudf_start = start_expr.value.GetValue<cudf::size_type>() - 1;
      const auto cudf_end   = len_expr.value.GetValue<cudf::size_type>() + cudf_start;

      return cudf::strings::slice_strings(input_view,
                                          cudf_start,
                                          cudf_end,
                                          1, // Step
                                          execution_stream,
                                          resource_ref);
    }
    else
    {
      const auto sirius_start = start_expr.value.GetValue<uint64_t>() - 1;
      const auto sirius_len   = len_expr.value.GetValue<uint64_t>();

      return GpuDispatcher::DispatchSubstring(input->view(),
                                              sirius_start,
                                              sirius_len,
                                              resource_ref,
                                              execution_stream);
    }
  }
  else if (func_str == LIKE_FUNC_STR)
  {
    StringMatchingDispatcher<StringMatchingType::LIKE, use_cudf> dispatcher(*this);
    return dispatcher(expr, state);
  }
  else if (func_str == NOT_LIKE_FUNC_STR)
  {
    StringMatchingDispatcher<StringMatchingType::NOT_LIKE, use_cudf> dispatcher(*this);
    return dispatcher(expr, state);
  }
  else if (func_str == CONTAINS_FUNC_STR)
  {
    StringMatchingDispatcher<StringMatchingType::CONTAINS, use_cudf> dispatcher(*this);
    return dispatcher(expr, state);
  }
  else if (func_str == PREFIX_FUNC_STR)
  {
    StringMatchingDispatcher<StringMatchingType::PREFIX, use_cudf> dispatcher(*this);
    return dispatcher(expr, state);
  }
  else if (func_str == SUFFIX_FUNC_STR)
  {
    StringMatchingDispatcher<StringMatchingType::SUFFIX, use_cudf> dispatcher(*this);
    return dispatcher(expr, state);
  }

  //----------Datetime Extract Functions----------//
  else if (func_str == YEAR_FUNC_STR)
  {
    DatetimeExtractFunctionDispatcher<cudf::datetime::datetime_component::YEAR> dispatcher(*this);
    return dispatcher(expr, state);
  }
  else if (func_str == MONTH_FUNC_STR)
  {
    DatetimeExtractFunctionDispatcher<cudf::datetime::datetime_component::MONTH> dispatcher(*this);
    return dispatcher(expr, state);
  }
  else if (func_str == DAY_FUNC_STR)
  {
    DatetimeExtractFunctionDispatcher<cudf::datetime::datetime_component::DAY> dispatcher(*this);
    return dispatcher(expr, state);
  }

  // If we've gotten this far, we've encountered a unimplemented function type
  throw InternalException("Execute[Function]: Unknown function type: %s", func_str);
}

} // namespace sirius
} // namespace duckdb
