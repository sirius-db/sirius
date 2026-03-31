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
#include <operator/empty_str_check.cuh>

// duckdb
#include <duckdb/common/exception.hpp>
#include <duckdb/planner/expression/bound_comparison_expression.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>

// cudf
#include <cudf/binaryop.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/transform.hpp>

// standard library
#include <memory>
#include <type_traits>

namespace sirius::experimental {
using execute_result = gpu_expression_executor::execute_result;

execute_result gpu_expression_executor::execute(duckdb::BoundComparisonExpression const& expr,
                                                execution_mode mode)
{
  if (_strategy != expression_executor_strategy::MATERIALIZE &&
      (mode == execution_mode::AST || count_ast_ops(expr) >= _min_ast_size)) {
    auto comparison_type_switch_ast =
      [](duckdb::BoundComparisonExpression const& expr) -> cudf::ast::ast_operator {
      using enum duckdb::ExpressionType;
      switch (expr.GetExpressionType()) {
        case COMPARE_EQUAL: return cudf::ast::ast_operator::EQUAL;
        case COMPARE_GREATERTHAN: return cudf::ast::ast_operator::GREATER;
        case COMPARE_GREATERTHANOREQUALTO: return cudf::ast::ast_operator::GREATER_EQUAL;
        case COMPARE_LESSTHAN: return cudf::ast::ast_operator::LESS;
        case COMPARE_LESSTHANOREQUALTO: return cudf::ast::ast_operator::LESS_EQUAL;
        case COMPARE_NOTEQUAL: return cudf::ast::ast_operator::NOT_EQUAL;
        case COMPARE_DISTINCT_FROM:
        case COMPARE_NOT_DISTINCT_FROM:
          throw duckdb::NotImplementedException(
            "[expression_executor:comparison] DISTINCT comparisons not supported in expression "
            "executor.");
        default:
          throw duckdb::InternalException(
            "[expression_executor:comparison] Unrecognized comparison type : {}",
            static_cast<int>(expr.GetExpressionType()));
      }
    };

    auto left      = execute(*expr.left, mode);
    auto right     = execute(*expr.right, mode);
    auto comp_expr = _ast_tree.emplace<cudf::ast::operation>(
      comparison_type_switch_ast(expr), left.get_expr(), right.get_expr());

    //===----------1: AST Mode----------===//
    if (mode == execution_mode::AST) {
      return execute_result(
        ast_result(comp_expr,
                   {left.get_temp_scalar_indices(), right.get_temp_scalar_indices()},
                   {left.get_temp_column_indices(), right.get_temp_column_indices()}));
    }

    //===----------2: MATERIALIZE Mode, evaluate node with AST----------===//
    // JIT compile the AST Subtree
    auto result_column = execute_ast(comp_expr);

    // Release consumed temporaries
    release_temporaries({left.get_temp_scalar_indices(), right.get_temp_scalar_indices()},
                        {left.get_temp_column_indices(), right.get_temp_column_indices()});
    return execute_result(std::move(result_column));
  }

  //===----------3: MATERIALIZE Mode, evaluate node with unary/binary ops----------===//
  auto comparison_type_switch =
    [](duckdb::BoundComparisonExpression const& expr) -> cudf::binary_operator {
    using enum duckdb::ExpressionType;
    switch (expr.GetExpressionType()) {
      case COMPARE_EQUAL: return cudf::binary_operator::EQUAL;
      case COMPARE_GREATERTHAN: return cudf::binary_operator::GREATER;
      case COMPARE_GREATERTHANOREQUALTO: return cudf::binary_operator::GREATER_EQUAL;
      case COMPARE_LESSTHAN: return cudf::binary_operator::LESS;
      case COMPARE_LESSTHANOREQUALTO: return cudf::binary_operator::LESS_EQUAL;
      case COMPARE_NOTEQUAL: return cudf::binary_operator::NOT_EQUAL;
      case COMPARE_DISTINCT_FROM:
      case COMPARE_NOT_DISTINCT_FROM:
        throw duckdb::NotImplementedException(
          "[expression_executor:comparison] DISTINCT comparisons not supported in expression "
          "executor.");
      default:
        throw duckdb::InternalException(
          "[expression_executor:comparison] Unrecognized comparison type : {}",
          static_cast<int>(expr.GetExpressionType()));
    }
  };

  auto left              = execute(*expr.left, execution_mode::MATERIALIZE);
  auto right             = execute(*expr.right, execution_mode::MATERIALIZE);
  auto const output_type = GetCudfType(expr.return_type);

  std::unique_ptr<cudf::column> result_column;
  if (left.is_scalar()) {
    result_column = cudf::binary_operation(left.get_scalar(),
                                           right.get_column_view(),
                                           comparison_type_switch(expr),
                                           output_type,
                                           _stream,
                                           _mr);
  } else if (right.is_scalar()) {
    result_column = cudf::binary_operation(left.get_column_view(),
                                           right.get_scalar(),
                                           comparison_type_switch(expr),
                                           output_type,
                                           _stream,
                                           _mr);
  } else {
    result_column = cudf::binary_operation(left.get_column_view(),
                                           right.get_column_view(),
                                           comparison_type_switch(expr),
                                           output_type,
                                           _stream,
                                           _mr);
  }
  return execute_result(std::move(result_column));
}
}  // namespace sirius::experimental

namespace duckdb {
namespace sirius {
//----------InitializeState----------//
std::unique_ptr<GpuExpressionState> GpuExpressionExecutor::InitializeState(
  const BoundComparisonExpression& expr, GpuExpressionExecutorState& root)
{
  auto result = std::make_unique<GpuExpressionState>(expr, root);
  result->AddChild(*expr.left);
  result->AddChild(*expr.right);
  return std::move(result);
}

// Helper object to reduce bloat in Execute()
template <cudf::binary_operator ComparisonOp>
struct ComparisonDispatcher {
  // The executor
  GpuExpressionExecutor& executor;

  // Constructor
  explicit ComparisonDispatcher(GpuExpressionExecutor& exec) : executor(exec) {}

  // Scalar comparison operator
  template <typename T>
  std::unique_ptr<cudf::column> DoScalarComparison(const cudf::column_view& left,
                                                   const T& right_value,
                                                   bool right_is_null,
                                                   const cudf::data_type& return_type)
  {
    if constexpr (std::is_same_v<T, std::string>) {
      // Create a string scalar from the constant value
      auto string_scalar = cudf::string_scalar(
        right_value, !right_is_null, executor.execution_stream, executor.resource_ref);

      return cudf::binary_operation(left,
                                    string_scalar,
                                    ComparisonOp,
                                    return_type,
                                    executor.execution_stream,
                                    executor.resource_ref);
    } else if constexpr (std::is_same_v<T, int32_t>) {
      // For int32_t, check at runtime if this is a TIMESTAMP_DAYS comparison
      if (left.type().id() == cudf::type_id::TIMESTAMP_DAYS) {
        auto date_scalar = cudf::timestamp_scalar<cudf::timestamp_D>(cudf::duration_D{right_value},
                                                                     !right_is_null,
                                                                     executor.execution_stream,
                                                                     executor.resource_ref);
        auto result      = cudf::binary_operation(left,
                                             date_scalar,
                                             ComparisonOp,
                                             return_type,
                                             executor.execution_stream,
                                             executor.resource_ref);
        return result;
      } else {
        // Regular int32_t comparison
        auto numeric_scalar = cudf::numeric_scalar(
          right_value, !right_is_null, executor.execution_stream, executor.resource_ref);
        return cudf::binary_operation(left,
                                      numeric_scalar,
                                      ComparisonOp,
                                      return_type,
                                      executor.execution_stream,
                                      executor.resource_ref);
      }
    } else if constexpr (std::is_same_v<T, int64_t>) {
      // For int64_t, check at runtime if this is a timestamp comparison
      switch (left.type().id()) {
        case cudf::type_id::TIMESTAMP_SECONDS: {
          auto ts_scalar = cudf::timestamp_scalar<cudf::timestamp_s>(cudf::duration_s{right_value},
                                                                     !right_is_null,
                                                                     executor.execution_stream,
                                                                     executor.resource_ref);
          return cudf::binary_operation(left,
                                        ts_scalar,
                                        ComparisonOp,
                                        return_type,
                                        executor.execution_stream,
                                        executor.resource_ref);
        }
        case cudf::type_id::TIMESTAMP_MILLISECONDS: {
          auto ts_scalar =
            cudf::timestamp_scalar<cudf::timestamp_ms>(cudf::duration_ms{right_value},
                                                       !right_is_null,
                                                       executor.execution_stream,
                                                       executor.resource_ref);
          return cudf::binary_operation(left,
                                        ts_scalar,
                                        ComparisonOp,
                                        return_type,
                                        executor.execution_stream,
                                        executor.resource_ref);
        }
        case cudf::type_id::TIMESTAMP_MICROSECONDS: {
          auto ts_scalar =
            cudf::timestamp_scalar<cudf::timestamp_us>(cudf::duration_us{right_value},
                                                       !right_is_null,
                                                       executor.execution_stream,
                                                       executor.resource_ref);
          return cudf::binary_operation(left,
                                        ts_scalar,
                                        ComparisonOp,
                                        return_type,
                                        executor.execution_stream,
                                        executor.resource_ref);
        }
        case cudf::type_id::TIMESTAMP_NANOSECONDS: {
          auto ts_scalar =
            cudf::timestamp_scalar<cudf::timestamp_ns>(cudf::duration_ns{right_value},
                                                       !right_is_null,
                                                       executor.execution_stream,
                                                       executor.resource_ref);
          return cudf::binary_operation(left,
                                        ts_scalar,
                                        ComparisonOp,
                                        return_type,
                                        executor.execution_stream,
                                        executor.resource_ref);
        }
        default: {
          // Regular int64_t comparison
          auto numeric_scalar = cudf::numeric_scalar(
            right_value, !right_is_null, executor.execution_stream, executor.resource_ref);
          return cudf::binary_operation(left,
                                        numeric_scalar,
                                        ComparisonOp,
                                        return_type,
                                        executor.execution_stream,
                                        executor.resource_ref);
        }
      }
    } else {
      // Create a numeric scalar from the constant value
      auto numeric_scalar = cudf::numeric_scalar(
        right_value, !right_is_null, executor.execution_stream, executor.resource_ref);
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
                                                   bool right_is_null,
                                                   numeric::scale_type scale,
                                                   const cudf::data_type& return_type)
  {
    std::unique_ptr<cudf::scalar> right_decimal_scalar;
    if (left.type().id() == cudf::type_to_id<T>()) {
      right_decimal_scalar = std::make_unique<cudf::fixed_point_scalar<T>>(
        right_value, scale, !right_is_null, executor.execution_stream, executor.resource_ref);
    } else {
      // If types are different, need to construct `right_decimal_scalar` using `left.type()`
      switch (left.type().id()) {
        case cudf::type_id::DECIMAL32: {
          if (right_value > std::numeric_limits<int32_t>::max()) {
            throw InternalException(
              "Cannot cast right decimal scalar to decimal32, value greater than INT32_MAX");
          }
          right_decimal_scalar = std::make_unique<cudf::fixed_point_scalar<numeric::decimal32>>(
            static_cast<int32_t>(right_value),
            scale,
            !right_is_null,
            executor.execution_stream,
            executor.resource_ref);
          break;
        }
        case cudf::type_id::DECIMAL64: {
          if (right_value > std::numeric_limits<int64_t>::max()) {
            throw InternalException(
              "Cannot cast right decimal scalar to decimal64, value greater than INT64_MAX");
          }
          right_decimal_scalar = std::make_unique<cudf::fixed_point_scalar<numeric::decimal64>>(
            static_cast<int64_t>(right_value),
            scale,
            !right_is_null,
            executor.execution_stream,
            executor.resource_ref);
          break;
        }
        case cudf::type_id::DECIMAL128: {
          right_decimal_scalar = std::make_unique<cudf::fixed_point_scalar<numeric::decimal128>>(
            static_cast<__int128_t>(right_value),
            scale,
            !right_is_null,
            executor.execution_stream,
            executor.resource_ref);
          break;
        }
        default:
          throw InternalException(
            "Left column is not decimal with right decimal constant in `DoRightScalarBinaryOp`: %d",
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
    auto return_type = GetCudfType(expr.return_type);

    // Resolve the children
    // DuckDB sometimes moves constants to the right comparator.
    // Even though this does not always happen, this file is based on the principle that DuckDB
    // always puts the constant on the right side. Therefore, if the left side is a constant, we
    // swap the children.

    const Expression* left_expr  = expr.left.get();
    const Expression* right_expr = expr.right.get();
    if (expr.left->GetExpressionClass() == ExpressionClass::BOUND_CONSTANT) {
      left_expr  = expr.right.get();
      right_expr = expr.left.get();
    }

    auto left = executor.Execute(*left_expr, state->child_states[0].get());

    // If the right side is a constant, do not materialize in a column
    if (right_expr->GetExpressionClass() == ExpressionClass::BOUND_CONSTANT) {
      auto right_value = right_expr->Cast<BoundConstantExpression>().value;

      switch (GetCudfType(right_expr->return_type).id()) {
        case cudf::type_id::INT16:
          return DoScalarComparison<int16_t>(
            left->view(),
            !right_value.IsNull() ? right_value.GetValue<int16_t>() : 0,
            right_value.IsNull(),
            return_type);
        case cudf::type_id::INT32:
          return DoScalarComparison<int32_t>(
            left->view(),
            !right_value.IsNull() ? right_value.GetValue<int32_t>() : 0,
            right_value.IsNull(),
            return_type);
        case cudf::type_id::INT64:
          return DoScalarComparison<int64_t>(
            left->view(),
            !right_value.IsNull() ? right_value.GetValue<int64_t>() : 0,
            right_value.IsNull(),
            return_type);
        case cudf::type_id::FLOAT32:
          return DoScalarComparison<float_t>(
            left->view(),
            !right_value.IsNull() ? right_value.GetValue<float_t>() : 0,
            right_value.IsNull(),
            return_type);
        case cudf::type_id::FLOAT64:
          return DoScalarComparison<double_t>(
            left->view(),
            !right_value.IsNull() ? right_value.GetValue<double_t>() : 0,
            right_value.IsNull(),
            return_type);
        case cudf::type_id::BOOL8:
          return DoScalarComparison<bool>(left->view(),
                                          !right_value.IsNull() ? right_value.GetValue<bool>() : 0,
                                          right_value.IsNull(),
                                          return_type);
        case cudf::type_id::STRING:
          return DoScalarComparison<std::string>(
            left->view(),
            !right_value.IsNull() ? right_value.GetValue<std::string>() : "",
            right_value.IsNull(),
            return_type);
        case cudf::type_id::TIMESTAMP_DAYS:
          // DuckDB DATE is int32_t (days since epoch), same as cuDF TIMESTAMP_DAYS
          return DoScalarComparison<int32_t>(

            left->view(),
            !right_value.IsNull() ? right_value.GetValue<int32_t>() : 0,
            right_value.IsNull(),
            return_type);

        case cudf::type_id::TIMESTAMP_SECONDS:
        case cudf::type_id::TIMESTAMP_MILLISECONDS:
        case cudf::type_id::TIMESTAMP_MICROSECONDS:
        case cudf::type_id::TIMESTAMP_NANOSECONDS:
          // DuckDB timestamps are int64_t internally
          return DoScalarComparison<int64_t>(
            left->view(),
            !right_value.IsNull() ? right_value.GetValue<int64_t>() : 0,
            right_value.IsNull(),
            return_type);

        case cudf::type_id::DECIMAL32:
          // cudf decimal type uses negative scale, same for below
          return DoScalarComparison<numeric::decimal32>(
            left->view(),
            !right_value.IsNull() ? right_value.GetValueUnsafe<int32_t>() : 0,
            right_value.IsNull(),
            numeric::scale_type{-duckdb::DecimalType::GetScale(right_value.type())},
            return_type);
        case cudf::type_id::DECIMAL64:
          return DoScalarComparison<numeric::decimal64>(
            left->view(),
            !right_value.IsNull() ? right_value.GetValueUnsafe<int64_t>() : 0,
            right_value.IsNull(),
            numeric::scale_type{-duckdb::DecimalType::GetScale(right_value.type())},
            return_type);
        case cudf::type_id::DECIMAL128: {
          duckdb::hugeint_t hugeint_value =
            !right_value.IsNull() ? right_value.GetValueUnsafe<duckdb::hugeint_t>() : 0;
          return DoScalarComparison<numeric::decimal128>(
            left->view(),
            (__int128_t(hugeint_value.upper) << 64) | hugeint_value.lower,
            right_value.IsNull(),
            numeric::scale_type{-duckdb::DecimalType::GetScale(right_value.type())},
            return_type);
        }
        default:
          throw InternalException(
            "Execute[Comparison]: Unsupported constant type for comparison: %d!",
            static_cast<int>(GetCudfType(expr.right->return_type).id()));
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
  auto return_type = GetCudfType(expr.return_type);

  // P5: empty-string check via offsets on the old gpu_processing path.
  // Rewrites (varchar_col <> '') to offset-based non-empty check, avoiding
  // string char buffer materialization entirely.
  if (!use_data_batch_apis && expr.GetExpressionType() == ExpressionType::COMPARE_NOTEQUAL &&
      expr.left->type == ExpressionType::BOUND_REF &&
      expr.right->GetExpressionClass() == ExpressionClass::BOUND_CONSTANT) {
    auto& right_val = expr.right->Cast<BoundConstantExpression>().value;
    if (right_val.type().id() == LogicalTypeId::VARCHAR && !right_val.IsNull() &&
        right_val.GetValue<std::string>().empty()) {
      auto& ref = expr.left->Cast<BoundReferenceExpression>();
      auto& col = input_columns[ref.index];
      if (col->data_wrapper.type.id() == GPUColumnTypeId::VARCHAR &&
          col->data_wrapper.offset != nullptr) {
        size_t num_rows = col->row_ids != nullptr ? col->row_id_count : col->column_length;
        return sirius::EmptyStrCheck(
          col->data_wrapper.offset, col->row_ids, num_rows, execution_stream, resource_ref);
      }
    }
  }

  // Execute the comparison
  switch (expr.GetExpressionType()) {
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
      throw NotImplementedException(
        "Execute[Comparison]: DISTINCT comparison not yet implemented!");
    case ExpressionType::COMPARE_NOT_DISTINCT_FROM: {
      ComparisonDispatcher<cudf::binary_operator::NULL_EQUALS> dispatcher(*this);
      return dispatcher(expr, state);
    }

    default: throw InternalException("Execute[Comparison]: Unknown comparison type!");
  }
}

}  // namespace sirius
}  // namespace duckdb
