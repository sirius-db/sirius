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
#include <expression/join_condition.hpp>
#include <expression_evaluator/expression_evaluator.hpp>
#include <sirius/exception.hpp>

// cudf
#include <cudf/ast/ast_operator.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/scalar/scalar.hpp>

// standard library
#include <memory>

namespace sirius {
using evaluate_result = expression_evaluator::evaluate_result;

namespace {

cudf::ast::ast_operator comparison_op_to_ast(sirius::comparison_type op)
{
  switch (op) {
    case sirius::comparison_type::equal: return cudf::ast::ast_operator::EQUAL;
    case sirius::comparison_type::gt: return cudf::ast::ast_operator::GREATER;
    case sirius::comparison_type::ge: return cudf::ast::ast_operator::GREATER_EQUAL;
    case sirius::comparison_type::lt: return cudf::ast::ast_operator::LESS;
    case sirius::comparison_type::le: return cudf::ast::ast_operator::LESS_EQUAL;
    case sirius::comparison_type::not_equal: return cudf::ast::ast_operator::NOT_EQUAL;
    case sirius::comparison_type::distinct_from:  // fallthrough — wrapped in NOT below
    case sirius::comparison_type::not_distinct_from: return cudf::ast::ast_operator::NULL_EQUAL;
    default:
      throw invalid_input_exception(
        "[expression_evaluator:comparison] Unrecognized comparison type : {}",
        static_cast<int>(op));
  }
}

cudf::binary_operator comparison_op_to_binary(sirius::comparison_type op)
{
  switch (op) {
    case sirius::comparison_type::equal: return cudf::binary_operator::EQUAL;
    case sirius::comparison_type::gt: return cudf::binary_operator::GREATER;
    case sirius::comparison_type::ge: return cudf::binary_operator::GREATER_EQUAL;
    case sirius::comparison_type::lt: return cudf::binary_operator::LESS;
    case sirius::comparison_type::le: return cudf::binary_operator::LESS_EQUAL;
    case sirius::comparison_type::not_equal: return cudf::binary_operator::NOT_EQUAL;
    case sirius::comparison_type::distinct_from: return cudf::binary_operator::NULL_NOT_EQUALS;
    case sirius::comparison_type::not_distinct_from: return cudf::binary_operator::NULL_EQUALS;
    default:
      throw invalid_input_exception(
        "[expression_evaluator:comparison] Unrecognized comparison type : {}",
        static_cast<int>(op));
  }
}

}  // namespace

evaluate_result expression_evaluator::evaluate(sirius::ast::comparison const& alt,
                                               evaluation_mode mode)
{
  // Compressed materialization: a comparison between a narrowed reference and a constant
  // representable in its carrier is evaluated directly at the narrow width — same values, same
  // family, same DECIMAL scale, so every comparison outcome (including NULL handling) is
  // identical, and the full-column restoration cast is skipped.
  auto narrow_carrier = narrow_domain_carrier(*alt.left, {alt.right.get()});
  if (!narrow_carrier) { narrow_carrier = narrow_domain_carrier(*alt.right, {alt.left.get()}); }
  // Column-vs-column at one shared carrier: both operands pass through as their raw narrow
  // columns, so neither is widened. Without this a pair like `l_commitdate < l_receiptdate`
  // restores both sides and the narrowing is pure cost.
  if (!narrow_carrier) {
    narrow_carrier = narrow_domain_reference_pair_carrier(*alt.left, *alt.right);
  }
  if (narrow_carrier) { ++_narrow_domain_comparison_count; }

  auto const ast_op_count = alt.cudf_ast_op_count();
  if (_strategy != expression_evaluator_strategy::MATERIALIZE &&
      (mode == evaluation_mode::AST || ast_op_count >= _min_ast_size)) {
    auto left  = evaluate_narrow_domain_operand(*alt.left, narrow_carrier, evaluation_mode::AST);
    auto right = evaluate_narrow_domain_operand(*alt.right, narrow_carrier, evaluation_mode::AST);
    auto const& comp_expr = _ast_tree.emplace<cudf::ast::operation>(
      comparison_op_to_ast(alt.op), left.get_expr(), right.get_expr());
    // DISTINCT_FROM is semantically equivalent to NOT(NULL_EQUAL())
    auto const& final_comp_expr =
      alt.op == sirius::comparison_type::distinct_from
        ? _ast_tree.emplace<cudf::ast::operation>(cudf::ast::ast_operator::NOT, comp_expr)
        : comp_expr;

    //===----------1: AST Mode----------===//
    if (mode == evaluation_mode::AST) {
      return evaluate_result(compose(final_comp_expr, {&left, &right}));
    }

    //===----------2: MATERIALIZE Mode, evaluate node with AST----------===//
    auto result_column = evaluate_ast(final_comp_expr);

    release_temporaries({&left, &right});
    return evaluate_result(std::move(result_column));
  }

  //===----------3: MATERIALIZE Mode, evaluate node with unary/binary ops----------===//
  auto left =
    evaluate_narrow_domain_operand(*alt.left, narrow_carrier, evaluation_mode::MATERIALIZE);
  auto right =
    evaluate_narrow_domain_operand(*alt.right, narrow_carrier, evaluation_mode::MATERIALIZE);
  // Comparison ops always return BOOLEAN — no logical_type on the AST node, so use BOOL8 directly.
  auto const output_type = cudf::data_type{cudf::type_id::BOOL8};
  auto const binary_op   = comparison_op_to_binary(alt.op);

  std::unique_ptr<cudf::column> result_column;
  if (left.is_scalar()) {
    result_column = cudf::binary_operation(
      left.get_scalar(), right.get_column_view(), binary_op, output_type, _stream, _mr);
  } else if (right.is_scalar()) {
    result_column = cudf::binary_operation(
      left.get_column_view(), right.get_scalar(), binary_op, output_type, _stream, _mr);
  } else {
    result_column = cudf::binary_operation(
      left.get_column_view(), right.get_column_view(), binary_op, output_type, _stream, _mr);
  }
  if (mode == evaluation_mode::AST) { return materialize_as_ast_column(std::move(result_column)); }
  return evaluate_result(std::move(result_column));
}
}  // namespace sirius
