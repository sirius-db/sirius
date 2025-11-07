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

#include "cudf/cudf_utils.hpp"
#include "duckdb/common/assert.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "expression_executor/gpu_dispatcher.hpp"
#include "expression_executor/gpu_expression_executor.hpp"
#include "expression_executor/gpu_expression_executor_state.hpp"
#include "expression_executor/regex/regex_playground.hpp"
#include "gpu_physical_strings_matching.hpp"
#include "log/logging.hpp"
#include <cudf/binaryop.hpp>
#include <cudf/datetime.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/attributes.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/replace_re.hpp>
#include <cudf/strings/slice.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/unary.hpp>
#include <string>
#include <regex>

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
#define HOUR_FUNC_STR "hour"
#define MINUTE_FUNC_STR "minute"
#define SECOND_FUNC_STR "second"
#define MILLISECOND_FUNC_STR "millisecond"
#define MICROSECOND_FUNC_STR "microsecond"
#define DATE_TRUNC_FUNC_STR "date_trunc"
#define STRLEN_FUNC_STR "strlen"
#define LENGTH_FUNC_STR "length"
#define REGEXP_REPLACE_FUNC_STR "regexp_replace"
#define ERROR_FUNC_STR "error"

#define SPLIT_DELIMITER "%"
#define WARP_SIZE 32

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
template <StringMatchingType MatchType>
struct StringMatchingDispatcher
{
  // The executor
  GpuExpressionExecutor& executor;
  const bool UseCudf;

  // Constructor
  explicit StringMatchingDispatcher(GpuExpressionExecutor& exec, bool use_cudf)
      : executor(exec), UseCudf(use_cudf)
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
    if (UseCudf)
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
        // There is an int32 overflow bug in `contains()` before cudf-25.10, where we have to use `like()`
        // if the input is too large
#if CUDF_VERSION_NUM < 2510
        bool can_use_contains = input->size() <= INT32_MAX / WARP_SIZE;
        if (can_use_contains) {
          return cudf::strings::contains(input->view(),
                                         cudf::string_scalar(match_str,
                                                             true,
                                                             executor.execution_stream,
                                                             executor.resource_ref),
                                         executor.execution_stream,
                                         executor.resource_ref);
        }
        return cudf::strings::like(cudf::strings_column_view(input_view),
                                   cudf::string_scalar("%" + match_str + "%",
                                                       true,
                                                       executor.execution_stream,
                                                       executor.resource_ref),
                                   cudf::string_scalar(""),
                                   executor.execution_stream,
                                   executor.resource_ref);
#else
        return cudf::strings::contains(input->view(),
                                       cudf::string_scalar(match_str,
                                                           true,
                                                           executor.execution_stream,
                                                           executor.resource_ref),
                                       executor.execution_stream,
                                       executor.resource_ref);
#endif
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
        case cudf::type_id::INT16:
          return DoLeftScalarBinaryOp(left_value.GetValue<int16_t>(), right->view(), return_type);
        case cudf::type_id::INT32:
          return DoLeftScalarBinaryOp(left_value.GetValue<int32_t>(), right->view(), return_type);
        case cudf::type_id::INT64:
          return DoLeftScalarBinaryOp(left_value.GetValue<int64_t>(), right->view(), return_type);
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
        case cudf::type_id::INT16:
          return DoRightScalarBinaryOp(left->view(), right_value.GetValue<int16_t>(), return_type);
        case cudf::type_id::INT32:
          return DoRightScalarBinaryOp(left->view(), right_value.GetValue<int32_t>(), return_type);
        case cudf::type_id::INT64:
          return DoRightScalarBinaryOp(left->view(), right_value.GetValue<int64_t>(), return_type);
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

//----------DatetimeTruncateFunctionDispatcher----------//
struct DatetimeTruncateFunctionDispatcher {
  // The executor
  GpuExpressionExecutor& executor;

  // Constructor
  explicit DatetimeTruncateFunctionDispatcher(GpuExpressionExecutor& exec)
      : executor(exec) {}

  // Dispatch operator
  std::unique_ptr<cudf::column> operator()(const BoundFunctionExpression& expr,
                                           GpuExpressionState* state)
  {
    D_ASSERT(expr.children.size() == 2);
    std::string freq_str = expr.children[0]->Cast<BoundConstantExpression>().value.GetValue<std::string>();
    auto input = executor.Execute(*expr.children[1], state->child_states[1].get());
    if (freq_str == "day") {
      return cudf::datetime::floor_datetimes(input->view(),
                                             cudf::datetime::rounding_frequency::DAY,
                                             executor.execution_stream,
                                             executor.resource_ref);
    } else if (freq_str == "hour") {
      return cudf::datetime::floor_datetimes(input->view(),
                                             cudf::datetime::rounding_frequency::HOUR,
                                             executor.execution_stream,
                                             executor.resource_ref);
    } else if (freq_str == "minute") {
      return cudf::datetime::floor_datetimes(input->view(),
                                             cudf::datetime::rounding_frequency::MINUTE,
                                             executor.execution_stream,
                                             executor.resource_ref);
    } else if (freq_str == "second") {
      return cudf::datetime::floor_datetimes(input->view(),
                                             cudf::datetime::rounding_frequency::SECOND,
                                             executor.execution_stream,
                                             executor.resource_ref);
    } else if (freq_str == "millisecond") {
      return cudf::datetime::floor_datetimes(input->view(),
                                             cudf::datetime::rounding_frequency::MILLISECOND,
                                             executor.execution_stream,
                                             executor.resource_ref);
    } else if (freq_str == "microsecond") {
      return cudf::datetime::floor_datetimes(input->view(),
                                             cudf::datetime::rounding_frequency::MICROSECOND,
                                             executor.execution_stream,
                                             executor.resource_ref);
    } else {
      throw InvalidInputException("Execute[Function]: Unknown extract type for date_trunc(): %s", freq_str);
    }
  }
};

//----------UnaryFunctionDispatcher----------//
template <UnaryFunctionType FuncType>
struct UnaryFunctionDispatcher {
  // The executor
  GpuExpressionExecutor& executor;

  // Constructor
  explicit UnaryFunctionDispatcher(GpuExpressionExecutor& exec)
      : executor(exec) {}

  // Dispatch operator
  std::unique_ptr<cudf::column> operator()(const BoundFunctionExpression& expr,
                                           GpuExpressionState* state) {
    D_ASSERT(expr.children.size() == 1);
    auto input = executor.Execute(*expr.children[0], state->child_states[0].get());

    switch (FuncType) {
      case UnaryFunctionType::STRLEN:
        return cudf::strings::count_bytes(input->view(),
                                          executor.execution_stream,
                                          executor.resource_ref);
      case UnaryFunctionType::LENGTH:
        return cudf::strings::count_characters(input->view(),
                                               executor.execution_stream,
                                               executor.resource_ref);
      default:
        throw NotImplementedException("Unsupported UnaryFunctionType: %d", static_cast<int>(FuncType));
    }
  }
};

//----------RegexFunctionDispatcher----------//
struct RegexFunctionDispatcher {
  // The executor
  GpuExpressionExecutor& executor;

  // Constructor
  explicit RegexFunctionDispatcher(GpuExpressionExecutor& exec)
      : executor(exec) {}

  // Dispatch operator
  std::unique_ptr<cudf::column> operator()(const BoundFunctionExpression& expr,
                                           GpuExpressionState* state) {
    auto input_cudf_column = executor.Execute(*expr.children[0], state->child_states[0].get());
    std::string pattern_str = expr.children[1]->Cast<BoundConstantExpression>().value.GetValue<std::string>();
    std::string replace_str = expr.children[2]->Cast<BoundConstantExpression>().value.GetValue<std::string>();
    bool has_backrefs = std::regex_search(replace_str, std::regex(R"(\\[0-9])"));
    if (has_backrefs) {
      if (Config::ENABLE_REGEX_JIT_IMPL) {
        if (pattern_str == R"(^https?://(?:www\.)?([^/]+)/.*$)" && replace_str == R"(\1)") {
          return ::sirius::expression::regex_playground::jit_transform_clickbench_q28_regex(input_cudf_column->view());
        }
      }
      auto regex_prog = cudf::strings::regex_program::create(std::string_view(pattern_str));
      return cudf::strings::replace_with_backrefs(cudf::strings_column_view(input_cudf_column->view()),
                                                  *regex_prog,
                                                  std::string_view(replace_str),
                                                  executor.execution_stream,
                                                  executor.resource_ref);
    } else {
      auto replace_cudf_column = cudf::make_column_from_scalar(cudf::string_scalar(replace_str,
                                                                                  true,
                                                                                  executor.execution_stream,
                                                                                  executor.resource_ref),
                                                               1,
                                                               executor.execution_stream,
                                                               executor.resource_ref);
      return cudf::strings::replace_re(cudf::strings_column_view(input_cudf_column->view()),
                                       {pattern_str},
                                       cudf::strings_column_view(replace_cudf_column->view()),
                                       cudf::strings::regex_flags::DEFAULT,
                                       executor.execution_stream,
                                       executor.resource_ref);
    }
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

    if (Config::USE_CUDF_EXPR)
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
    StringMatchingDispatcher<StringMatchingType::LIKE> dispatcher(*this, Config::USE_CUDF_EXPR);
    return dispatcher(expr, state);
  }
  else if (func_str == NOT_LIKE_FUNC_STR)
  {
    StringMatchingDispatcher<StringMatchingType::NOT_LIKE> dispatcher(*this, Config::USE_CUDF_EXPR);
    return dispatcher(expr, state);
  }
  else if (func_str == CONTAINS_FUNC_STR)
  {
    StringMatchingDispatcher<StringMatchingType::CONTAINS> dispatcher(*this, Config::USE_CUDF_EXPR);
    return dispatcher(expr, state);
  }
  else if (func_str == PREFIX_FUNC_STR)
  {
    StringMatchingDispatcher<StringMatchingType::PREFIX> dispatcher(*this, Config::USE_CUDF_EXPR);
    return dispatcher(expr, state);
  }
  else if (func_str == SUFFIX_FUNC_STR)
  {
    StringMatchingDispatcher<StringMatchingType::SUFFIX> dispatcher(*this, Config::USE_CUDF_EXPR);
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
  else if (func_str == HOUR_FUNC_STR)
  {
    DatetimeExtractFunctionDispatcher<cudf::datetime::datetime_component::HOUR> dispatcher(*this);
    return dispatcher(expr, state);
  }
  else if (func_str == MINUTE_FUNC_STR)
  {
    DatetimeExtractFunctionDispatcher<cudf::datetime::datetime_component::MINUTE> dispatcher(*this);
    return dispatcher(expr, state);
  }
  else if (func_str == SECOND_FUNC_STR)
  {
    DatetimeExtractFunctionDispatcher<cudf::datetime::datetime_component::SECOND> dispatcher(*this);
    return dispatcher(expr, state);
  }
  else if (func_str == MILLISECOND_FUNC_STR)
  {
    DatetimeExtractFunctionDispatcher<cudf::datetime::datetime_component::MILLISECOND> dispatcher(*this);
    return dispatcher(expr, state);
  }
  else if (func_str == MICROSECOND_FUNC_STR)
  {
    DatetimeExtractFunctionDispatcher<cudf::datetime::datetime_component::MICROSECOND> dispatcher(*this);
    return dispatcher(expr, state);
  }

  //----------DateTime Truncate Functions----------//
  else if (func_str == DATE_TRUNC_FUNC_STR) {
    DatetimeTruncateFunctionDispatcher dispatcher(*this);
    return dispatcher(expr, state);
  }

  //----------Unary Functions----------//
  else if (func_str == STRLEN_FUNC_STR) {
    UnaryFunctionDispatcher<UnaryFunctionType::STRLEN> dispatcher(*this);
    return dispatcher(expr, state);
  }
  else if (func_str == LENGTH_FUNC_STR) {
    UnaryFunctionDispatcher<UnaryFunctionType::LENGTH> dispatcher(*this);
    return dispatcher(expr, state);
  }

  //----------Regex Functions----------//
  else if (func_str == REGEXP_REPLACE_FUNC_STR) {
    RegexFunctionDispatcher dispatcher(*this);
    return dispatcher(expr, state);
  }

  // If we've gotten this far, we've encountered a unimplemented function type
  throw InternalException("Execute[Function]: Unknown function type: %s", func_str);
}

} // namespace sirius
} // namespace duckdb
