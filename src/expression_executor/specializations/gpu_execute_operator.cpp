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
#include <log/logging.hpp>

// duckdb
#include <duckdb/common/assert.hpp>
#include <duckdb/common/exception.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/expression/bound_operator_expression.hpp>

// cudf
#include <cudf/binaryop.hpp>
#include <cudf/cudf_utils.hpp>
#include <cudf/search.hpp>
#include <cudf/unary.hpp>

// rmm
#include <rmm/device_uvector.hpp>

// standard library
#include <algorithm>

namespace {
template <typename T>
std::unique_ptr<cudf::column> execute_numeric_in(const duckdb::BoundOperatorExpression& expr,
                                                 const cudf::column_view& input_view,
                                                 rmm::device_async_resource_ref mr,
                                                 rmm::cuda_stream_view stream)
{
  std::vector<T> children_vals;
  for (std::size_t child = 1; child < expr.children.size(); ++child) {
    const auto& child_expression = expr.children[child]->Cast<duckdb::BoundConstantExpression>();
    children_vals.push_back(child_expression.value.GetValue<T>());
  }
  rmm::device_uvector<T> children_vals_d(children_vals.size(), stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(children_vals_d.data(),
                                children_vals.data(),
                                children_vals.size() * sizeof(T),
                                cudaMemcpyHostToDevice,
                                stream));
  cudf::column_view children_view(input_view.type(),
                                  static_cast<cudf::size_type>(children_vals.size()),
                                  children_vals_d.data(),
                                  nullptr,
                                  0,
                                  0);
  return cudf::contains(children_view, input_view, stream, mr);
}

std::unique_ptr<cudf::column> execute_string_in(const duckdb::BoundOperatorExpression& expr,
                                                const cudf::column_view& input_view,
                                                rmm::device_async_resource_ref mr,
                                                rmm::cuda_stream_view stream)
{
  auto const num_strings = static_cast<cudf::size_type>(expr.children.size() - 1);
  auto const num_offsets = num_strings + 1;

  // We need to convert to cudf/arrow format...
  std::vector<char> chars;
  std::vector<cudf::size_type> offsets;
  cudf::size_type offset = 0;
  for (std::size_t child = 1; child < expr.children.size(); ++child) {
    const auto& child_expression = expr.children[child]->Cast<duckdb::BoundConstantExpression>();
    const auto& child_string     = child_expression.value.GetValue<std::string>();
    chars.insert(chars.end(), child_string.begin(), child_string.end());
    offsets.push_back(offset);
    offset += static_cast<cudf::size_type>(child_string.size());
  }
  offsets.push_back(offset);

  // Allocate buffers and copy to device
  rmm::device_uvector<char> chars_buffer(offset, stream, mr);
  rmm::device_uvector<cudf::size_type> offsets_buffer(num_offsets, stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(chars_buffer.data(),
                                chars.data(),
                                chars.size() * sizeof(char),
                                cudaMemcpyHostToDevice,
                                stream));
  CUDF_CUDA_TRY(cudaMemcpyAsync(offsets_buffer.data(),
                                offsets.data(),
                                offsets.size() * sizeof(cudf::size_type),
                                cudaMemcpyHostToDevice,
                                stream));

  // Make CuDF things
  auto offsets_col = std::make_unique<cudf::column>(cudf::data_type(cudf::type_id::INT32),
                                                    num_offsets,
                                                    std::move(offsets_buffer).release(),
                                                    rmm::device_buffer{},
                                                    0);
  std::vector<std::unique_ptr<cudf::column>> children;
  children.push_back(std::move(offsets_col));
  auto in_strings_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::STRING},
                                                       num_strings,
                                                       std::move(chars_buffer).release(),
                                                       rmm::device_buffer{},
                                                       0,
                                                       std::move(children));

  // Execute the search
  return cudf::contains(in_strings_col->view(), input_view, stream, mr);
}

}  // namespace

namespace sirius {
using execute_result = gpu_expression_executor::execute_result;

execute_result gpu_expression_executor::execute(duckdb::BoundOperatorExpression const& expr,
                                                execution_mode mode)
{
  if (_strategy != expression_executor_strategy::MATERIALIZE &&
      (mode == execution_mode::AST || count_ast_ops(expr) >= _min_ast_size)) {
    auto operator_type_switch_ast =
      [&](duckdb::BoundOperatorExpression const& expr) -> execute_result {
      switch (expr.type) {
        case duckdb::ExpressionType::COMPARE_IN:  // Fallthrough
        case duckdb::ExpressionType::COMPARE_NOT_IN: {
          D_ASSERT(expr.children.size() > 1);
          auto test                = execute(*expr.children[0], execution_mode::AST);
          auto comparator          = execute(*expr.children[1], execution_mode::AST);
          expr_ref comparison_expr = _ast_tree.emplace<cudf::ast::operation>(
            cudf::ast::ast_operator::EQUAL, test.get_expr(), comparator.get_expr());
          auto output = execute_result(
            ast_result(comparison_expr,
                       {test.get_temp_scalar_indices(), comparator.get_temp_scalar_indices()},
                       {test.get_temp_column_indices(), comparator.get_temp_column_indices()}));

          // Build an OR tree of comparisons
          for (std::size_t child_idx = 2; child_idx < expr.children.size(); ++child_idx) {
            auto comparator               = execute(*expr.children[child_idx], execution_mode::AST);
            expr_ref next_comparison_expr = _ast_tree.emplace<cudf::ast::operation>(
              cudf::ast::ast_operator::EQUAL, test.get_expr(), comparator.get_expr());
            comparison_expr = _ast_tree.emplace<cudf::ast::operation>(
              cudf::ast::ast_operator::LOGICAL_OR, comparison_expr, next_comparison_expr);
            output = execute_result(
              ast_result(comparison_expr,
                         {output.get_temp_scalar_indices(), comparator.get_temp_scalar_indices()},
                         {output.get_temp_column_indices(), comparator.get_temp_column_indices()}));
          }

          if (expr.type == duckdb::ExpressionType::COMPARE_IN) { return output; }
          auto const& not_expr =
            _ast_tree.emplace<cudf::ast::operation>(cudf::ast::ast_operator::NOT, comparison_expr);
          return execute_result(ast_result(
            not_expr, output.get_temp_scalar_indices(), output.get_temp_column_indices()));
        }
        case duckdb::ExpressionType::OPERATOR_COALESCE:
          /// KEVIN: TODO
          throw duckdb::NotImplementedException(
            "[gpu_expression_executor] execute called on an unsupported COALESCE operator "
            "expression.");
        case duckdb::ExpressionType::OPERATOR_TRY:
          throw duckdb::NotImplementedException(
            "[gpu_expression_executor] execute called on an unsupported TRY operator expression.");
        case duckdb::ExpressionType::OPERATOR_NOT: {
          D_ASSERT(expr.children.size() == 1);
          auto child = execute(*expr.children[0], execution_mode::AST);
          auto const& not_expr =
            _ast_tree.emplace<cudf::ast::operation>(cudf::ast::ast_operator::NOT, child.get_expr());
          return execute_result(
            ast_result(not_expr, child.get_temp_scalar_indices(), child.get_temp_column_indices()));
        }
        case duckdb::ExpressionType::OPERATOR_IS_NULL:  // Fallthrough
        case duckdb::ExpressionType::OPERATOR_IS_NOT_NULL: {
          D_ASSERT(expr.children.size() == 1);
          auto child               = execute(*expr.children[0], execution_mode::AST);
          auto const& is_null_expr = _ast_tree.emplace<cudf::ast::operation>(
            cudf::ast::ast_operator::IS_NULL, child.get_expr());
          if (expr.type == duckdb::ExpressionType::OPERATOR_IS_NULL) {
            return execute_result(ast_result(
              is_null_expr, child.get_temp_scalar_indices(), child.get_temp_column_indices()));
          } else {
            auto const& not_expr =
              _ast_tree.emplace<cudf::ast::operation>(cudf::ast::ast_operator::NOT, is_null_expr);
            return execute_result(ast_result(
              not_expr, child.get_temp_scalar_indices(), child.get_temp_column_indices()));
          }
        }
        default:
          throw duckdb::InternalException(
            "[gpu_expression_executor] execute called on an operator expression [{}] with "
            "unknown/unsupported operator type: {}",
            expr.ToString(),
            static_cast<int>(expr.type));
      }
    };

    auto output = operator_type_switch_ast(expr);

    if (mode == execution_mode::AST) {
      //===----------1: AST Mode----------===//
      return output;
    }

    //===----------2: MATERIALIZE Mode, evaluate node with AST----------===//
    auto result_column = execute_ast(output.get_expr());

    // Release consumed temporaries
    release_temporaries(output.get_temp_scalar_indices(), output.get_temp_column_indices());
    return execute_result(std::move(result_column));
  }

  //===----------3: MATERIALIZE Mode, evaluate node with unary/binary ops----------===//
  if (mode == execution_mode::AST) {
    auto result = execute(expr, execution_mode::MATERIALIZE);
    return materialize_as_ast_column(result.release_column());
  }
  switch (expr.type) {
    case duckdb::ExpressionType::COMPARE_IN:  // Fallthrough
    case duckdb::ExpressionType::COMPARE_NOT_IN: {
      D_ASSERT(expr.children.size() > 1);
      auto test = execute(*expr.children[0], execution_mode::MATERIALIZE);
      D_ASSERT(!test.is_scalar());  // IN with scalar LHS should have been already resolved

      // Optimization: special handling for case where RHS are all constants
      if (std::all_of(expr.children.begin() + 1, expr.children.end(), [](const auto& child) {
            return child->GetExpressionClass() == duckdb::ExpressionClass::BOUND_CONSTANT;
          })) {
        // All types should be the same
        /// KEVIN: TODO: Support more types here
        std::unique_ptr<cudf::column> contains_column;
        switch (test.get_column_view().type().id()) {
          case cudf::type_id::INT16:
            contains_column =
              execute_numeric_in<int16_t>(expr, test.get_column_view(), _mr, _stream);
            break;
          case cudf::type_id::INT32:
            contains_column =
              execute_numeric_in<int32_t>(expr, test.get_column_view(), _mr, _stream);
            break;
          case cudf::type_id::INT64:
            contains_column =
              execute_numeric_in<int64_t>(expr, test.get_column_view(), _mr, _stream);
            break;
          case cudf::type_id::FLOAT32:
            contains_column =
              execute_numeric_in<float_t>(expr, test.get_column_view(), _mr, _stream);
            break;
          case cudf::type_id::FLOAT64:
            contains_column =
              execute_numeric_in<double_t>(expr, test.get_column_view(), _mr, _stream);
            break;
          case cudf::type_id::BOOL8:
            contains_column =
              execute_numeric_in<uint8_t>(expr, test.get_column_view(), _mr, _stream);
            break;
          case cudf::type_id::STRING:
            contains_column = execute_string_in(expr, test.get_column_view(), _mr, _stream);
            break;
          default:
            throw duckdb::NotImplementedException(
              "[gpu_expression_executor] execute IN called with unsupported scalar haystack type "
              "{}",
              static_cast<int>(test.get_column_view().type().id()));
        }
        if (expr.type == duckdb::ExpressionType::COMPARE_NOT_IN) {
          contains_column =
            cudf::unary_operation(contains_column->view(), cudf::unary_operator::NOT, _stream, _mr);
        }
        return execute_result(std::move(contains_column));
      }

      // Some hastack referent is not a scalar
      auto const output_type = GetCudfType(expr.return_type);
      auto comparator        = execute(*expr.children[1], execution_mode::MATERIALIZE);
      auto comparison_column = cudf::binary_operation(test.get_column_view(),
                                                      comparator.get_column_view(),
                                                      cudf::binary_operator::EQUAL,
                                                      output_type,
                                                      _stream,
                                                      _mr);
      auto output            = execute_result(std::move(comparison_column));

      // For every child, OR the result of the comparison with the left to get the overall result.
      for (std::size_t child = 2; child < expr.children.size(); ++child) {
        // Resolve the child
        auto comparator        = execute(*expr.children[child], execution_mode::MATERIALIZE);
        auto comparison_column = cudf::binary_operation(test.get_column_view(),
                                                        comparator.get_column_view(),
                                                        cudf::binary_operator::EQUAL,
                                                        output_type,
                                                        _stream,
                                                        _mr);

        auto combined_comparison_column = cudf::binary_operation(output.get_column_view(),
                                                                 comparison_column->view(),
                                                                 cudf::binary_operator::LOGICAL_OR,
                                                                 output_type,
                                                                 _stream,
                                                                 _mr);
        output                          = execute_result(std::move(combined_comparison_column));
      }

      if (expr.type == duckdb::ExpressionType::COMPARE_IN) { return output; }
      auto not_comparison_column =
        cudf::unary_operation(output.get_column_view(), cudf::unary_operator::NOT, _stream, _mr);
      return execute_result(std::move(not_comparison_column));
    }
    case duckdb::ExpressionType::OPERATOR_COALESCE:
      /// KEVIN: TODO
      throw duckdb::NotImplementedException(
        "[gpu_expression_executor] execute called on an unsupported COALESCE operator expression.");
    case duckdb::ExpressionType::OPERATOR_TRY:
      throw duckdb::NotImplementedException(
        "[gpu_expression_executor] execute called on an unsupported TRY operator expression.");
    case duckdb::ExpressionType::OPERATOR_NOT: {
      D_ASSERT(expr.children.size() == 1);
      auto child = execute(*expr.children[0], execution_mode::MATERIALIZE);
      return execute_result(
        cudf::unary_operation(child.get_column_view(), cudf::unary_operator::NOT, _stream, _mr));
    }
    case duckdb::ExpressionType::OPERATOR_IS_NULL:  // Fallthrough
    case duckdb::ExpressionType::OPERATOR_IS_NOT_NULL: {
      D_ASSERT(expr.children.size() == 1);
      auto child          = execute(*expr.children[0], execution_mode::MATERIALIZE);
      auto is_null_result = cudf::is_null(child.get_column_view(), _stream, _mr);
      if (expr.type == duckdb::ExpressionType::OPERATOR_IS_NULL) {
        return execute_result(std::move(is_null_result));
      }
      return execute_result(
        cudf::unary_operation(is_null_result->view(), cudf::unary_operator::NOT, _stream, _mr));
    }
    default:
      throw duckdb::InternalException(
        "[gpu_expression_executor] execute called on an operator expression [{}] with "
        "unknown/unsupported "
        "operator type: {}",
        expr.ToString(),
        static_cast<int>(expr.type));
  }
}

}  // namespace sirius
