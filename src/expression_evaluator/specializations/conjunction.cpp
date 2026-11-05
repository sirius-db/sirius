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
#include <expression_evaluator/expression_evaluator.hpp>
#include <sirius/exception.hpp>

// duckdb

// cudf
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/replace.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/unary.hpp>

// standard library
#include <optional>

namespace sirius {
using evaluate_result = expression_evaluator::evaluate_result;

namespace {

cudf::ast::ast_operator conjunction_op_to_ast(sirius::ast::conjunction::kind op)
{
  // SQL AND/OR use Kleene three-valued logic (e.g. TRUE OR NULL = TRUE), so use
  // cuDF's NULL_LOGICAL_* operators rather than the null-propagating LOGICAL_*.
  switch (op) {
    case sirius::ast::conjunction::kind::op_and: return cudf::ast::ast_operator::NULL_LOGICAL_AND;
    case sirius::ast::conjunction::kind::op_or: return cudf::ast::ast_operator::NULL_LOGICAL_OR;
    default:
      throw invalid_input_exception(
        "[expression_evaluator:conjunction] unrecognized conjunction type {}",
        static_cast<int>(op));
  }
}

cudf::binary_operator conjunction_op_to_binary(sirius::ast::conjunction::kind op)
{
  // Kleene three-valued logic (SQL semantics); see conjunction_op_to_ast.
  switch (op) {
    case sirius::ast::conjunction::kind::op_and: return cudf::binary_operator::NULL_LOGICAL_AND;
    case sirius::ast::conjunction::kind::op_or: return cudf::binary_operator::NULL_LOGICAL_OR;
    default:
      throw invalid_input_exception(
        "[expression_evaluator:conjunction] unrecognized conjunction type {}",
        static_cast<int>(op));
  }
}

}  // namespace

evaluate_result expression_evaluator::evaluate(sirius::ast::conjunction const& alt,
                                               evaluation_mode mode)
{
  if (alt.children.empty()) {
    throw invalid_input_exception(
      "[expression_evaluator:conjunction] conjunction has no children — malformed AST.");
  }
  auto const ast_op_count = alt.cudf_ast_op_count();

  if (_policy.strategy != expression_evaluator_strategy::MATERIALIZE &&
      (mode == evaluation_mode::AST || ast_op_count >= _min_ast_size)) {
    auto const ast_op = conjunction_op_to_ast(alt.op);

    auto output = evaluate(*alt.children[0], evaluation_mode::AST);

    for (std::size_t i = 1; i < alt.children.size(); ++i) {
      auto child = evaluate(*alt.children[i], evaluation_mode::AST);
      auto const& output_expr =
        _ast_tree.emplace<cudf::ast::operation>(ast_op, output.get_expr(), child.get_expr());
      output = evaluate_result(compose(output_expr, {&output, &child}));
    }

    if (mode == evaluation_mode::AST) {
      //===----------1: AST Mode----------===//
      return output;
    }
    //===----------2: MATERIALIZE Mode, evaluate node with AST----------===//
    auto result_column = evaluate_ast(output.get_expr());

    release_temporaries({&output});
    return evaluate_result(std::move(result_column));
  }

  //===----------3: MATERIALIZE Mode, evaluate node with unary/binary ops----------===//
  auto const binary_op = conjunction_op_to_binary(alt.op);
  // Conjunction ops always return BOOLEAN.
  auto const output_type = cudf::data_type{cudf::type_id::BOOL8};

  // Resolve the children incrementally into the output. Scalar children survive constant folding
  // when they can't decide the result, e.g. `x OR NULL`.
  auto output = evaluate(*alt.children[0], evaluation_mode::MATERIALIZE);
  for (std::size_t i = 1; i < alt.children.size(); ++i) {
    auto child = evaluate(*alt.children[i], evaluation_mode::MATERIALIZE);
    if (output.is_scalar() && child.is_scalar()) {
      output = evaluate_result(
        cudf::make_column_from_scalar(output.get_scalar(), _input_table.num_rows(), _stream, _mr));
    }
    std::unique_ptr<cudf::column> output_column;
    if (output.is_scalar()) {
      output_column = cudf::binary_operation(
        output.get_scalar(), child.get_column_view(), binary_op, output_type, _stream, _mr);
    } else if (child.is_scalar()) {
      output_column = cudf::binary_operation(
        output.get_column_view(), child.get_scalar(), binary_op, output_type, _stream, _mr);
    } else {
      output_column = cudf::binary_operation(
        output.get_column_view(), child.get_column_view(), binary_op, output_type, _stream, _mr);
    }
    output = evaluate_result(std::move(output_column));
  }
  return output;
}

evaluate_result expression_evaluator::evaluate_predicate(sirius::ast::node const& expr)
{
  if (!expr.holds<sirius::ast::conjunction>()) {
    return evaluate(expr, evaluation_mode::MATERIALIZE);
  }
  auto const& alt = expr.get<sirius::ast::conjunction>();
  return with_selective_retry([&] { return evaluate(alt, evaluation_mode::MATERIALIZE); },
                              [&] { return evaluate_conjunction_selectively(alt); });
}

std::unique_ptr<cudf::column> expression_evaluator::evaluate_conjunction_selectively(
  sirius::ast::conjunction const& alt)
{
  // DuckDB's Select: AND evaluates each operand on the rows that every earlier operand passed, and
  // OR on the rows that none passed, where NULL does not pass. Every mask below is a null-free
  // BOOL8 column over all input rows.
  auto const num_rows  = _input_table.num_rows();
  auto const condition = cudf::data_type{cudf::type_id::BOOL8};
  if (num_rows == 0) { return cudf::make_empty_column(condition); }
  auto const no     = cudf::numeric_scalar<bool>(false, true, _stream, _mr);
  bool const is_and = alt.op == sirius::ast::conjunction::kind::op_and;

  // The rows the next operand evaluates; empty while that is every row.
  std::unique_ptr<cudf::column> pending;
  // OR: the rows an operand passed.
  std::unique_ptr<cudf::column> passed;
  for (auto const& child : alt.children) {
    auto const candidates =
      pending ? std::optional<cudf::column_view>{pending->view()} : std::nullopt;
    auto outcome = evaluate_on_selected_rows(
      *child, candidates ? &*candidates : nullptr, condition, /*predicate=*/true);
    // Every row is decided, so no later operand is evaluated.
    if (!outcome) { break; }
    if (outcome->has_nulls()) { outcome = cudf::replace_nulls(outcome->view(), no, _stream, _mr); }
    if (candidates) {
      auto const none = cudf::make_column_from_scalar(no, num_rows, _stream, _mr);
      outcome         = scatter_selected(outcome->view(), none->view(), *candidates);
    }

    if (is_and) {
      pending = std::move(outcome);
      continue;
    }
    pending = candidates
                ? cudf::copy_if_else(no, *candidates, outcome->view(), _stream, _mr)
                : cudf::unary_operation(outcome->view(), cudf::unary_operator::NOT, _stream, _mr);
    passed  = passed ? cudf::binary_operation(passed->view(),
                                             outcome->view(),
                                             cudf::binary_operator::LOGICAL_OR,
                                             condition,
                                             _stream,
                                             _mr)
                     : std::move(outcome);
  }
  auto result = is_and ? std::move(pending) : std::move(passed);
  if (!result) {
    throw invalid_input_exception(
      "[expression_evaluator:conjunction] conjunction has no children — malformed AST.");
  }
  return result;
}

}  // namespace sirius
