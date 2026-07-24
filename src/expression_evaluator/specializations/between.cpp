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
#include <expression/ast/node.hpp>
#include <expression_evaluator/expression_evaluator.hpp>
#include <sirius/exception.hpp>

// cudf
#include <cudf/binaryop.hpp>
#include <cudf/cudf_utils.hpp>
#include <cudf/transform.hpp>

namespace sirius {
using evaluate_result = expression_evaluator::evaluate_result;

evaluate_result expression_evaluator::evaluate(sirius::ast::between const& alt,
                                               evaluation_mode mode)
{
  // Compressed materialization: BETWEEN over a narrowed reference with both bounds representable
  // in its carrier compares directly at the narrow width, skipping the restoration cast (see the
  // comparison specialization for the value-preservation argument).
  auto const narrow_carrier = narrow_domain_carrier(*alt.input, {alt.lower.get(), alt.upper.get()});
  if (narrow_carrier) { ++_narrow_domain_comparison_count; }

  auto const ast_op_count = alt.cudf_ast_op_count();

  if (_strategy != expression_evaluator_strategy::MATERIALIZE &&
      (mode == evaluation_mode::AST || ast_op_count >= _min_ast_size)) {
    auto const lower_ast_op = alt.lower_inclusive ? cudf::ast::ast_operator::GREATER_EQUAL
                                                  : cudf::ast::ast_operator::GREATER;
    auto const upper_ast_op =
      alt.upper_inclusive ? cudf::ast::ast_operator::LESS_EQUAL : cudf::ast::ast_operator::LESS;

    auto input = evaluate_narrow_domain_operand(*alt.input, narrow_carrier, evaluation_mode::AST);
    D_ASSERT(!input.is_scalar());
    auto lower = evaluate_narrow_domain_operand(*alt.lower, narrow_carrier, evaluation_mode::AST);
    auto upper = evaluate_narrow_domain_operand(*alt.upper, narrow_carrier, evaluation_mode::AST);

    auto const& lower_expr =
      _ast_tree.emplace<cudf::ast::operation>(lower_ast_op, input.get_expr(), lower.get_expr());
    auto const& upper_expr =
      _ast_tree.emplace<cudf::ast::operation>(upper_ast_op, input.get_expr(), upper.get_expr());
    auto const& between_expr = _ast_tree.emplace<cudf::ast::operation>(
      cudf::ast::ast_operator::LOGICAL_AND, lower_expr, upper_expr);

    //===----------1: AST Mode----------===//
    if (mode == evaluation_mode::AST) {
      return evaluate_result(ast_result(between_expr,
                                        {input.get_temp_scalar_indices(),
                                         lower.get_temp_scalar_indices(),
                                         upper.get_temp_scalar_indices()},
                                        {input.get_temp_column_indices(),
                                         lower.get_temp_column_indices(),
                                         upper.get_temp_column_indices()}));
    }

    //===----------2: MATERIALIZE Mode, evaluate node with AST----------===//
    auto result_column = evaluate_ast(between_expr);
    release_temporaries({input.get_temp_scalar_indices(),
                         lower.get_temp_scalar_indices(),
                         upper.get_temp_scalar_indices()},
                        {input.get_temp_column_indices(),
                         lower.get_temp_column_indices(),
                         upper.get_temp_column_indices()});
    return evaluate_result(std::move(result_column));
  }

  //===----------3: MATERIALIZE Mode, evaluate node with unary/binary ops----------===//
  auto const lower_bin_op =
    alt.lower_inclusive ? cudf::binary_operator::GREATER_EQUAL : cudf::binary_operator::GREATER;
  auto const upper_bin_op =
    alt.upper_inclusive ? cudf::binary_operator::LESS_EQUAL : cudf::binary_operator::LESS;

  auto input =
    evaluate_narrow_domain_operand(*alt.input, narrow_carrier, evaluation_mode::MATERIALIZE);
  auto lower =
    evaluate_narrow_domain_operand(*alt.lower, narrow_carrier, evaluation_mode::MATERIALIZE);
  auto upper =
    evaluate_narrow_domain_operand(*alt.upper, narrow_carrier, evaluation_mode::MATERIALIZE);
  // BETWEEN always returns BOOLEAN.
  auto const output_type = cudf::data_type{cudf::type_id::BOOL8};

  std::unique_ptr<cudf::column> lower_cmp;
  std::unique_ptr<cudf::column> upper_cmp;
  if (lower.is_scalar()) {
    lower_cmp = cudf::binary_operation(
      input.get_column_view(), lower.get_scalar(), lower_bin_op, output_type, _stream, _mr);
  } else {
    lower_cmp = cudf::binary_operation(
      input.get_column_view(), lower.get_column_view(), lower_bin_op, output_type, _stream, _mr);
  }
  if (upper.is_scalar()) {
    upper_cmp = cudf::binary_operation(
      input.get_column_view(), upper.get_scalar(), upper_bin_op, output_type, _stream, _mr);
  } else {
    upper_cmp = cudf::binary_operation(
      input.get_column_view(), upper.get_column_view(), upper_bin_op, output_type, _stream, _mr);
  }
  auto result_column = cudf::binary_operation(lower_cmp->view(),
                                              upper_cmp->view(),
                                              cudf::binary_operator::LOGICAL_AND,
                                              output_type,
                                              _stream,
                                              _mr);
  return evaluate_result(std::move(result_column));
}

}  // namespace sirius
