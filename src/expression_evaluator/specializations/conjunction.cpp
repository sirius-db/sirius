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

// duckdb
#include <duckdb/common/exception.hpp>

// cudf
#include <cudf/binaryop.hpp>
#include <cudf/cudf_utils.hpp>

namespace sirius {
using evaluate_result = expression_evaluator::evaluate_result;

namespace {

cudf::ast::ast_operator conjunction_op_to_ast(sirius::ast::conjunction::kind op)
{
  switch (op) {
    case sirius::ast::conjunction::kind::op_and: return cudf::ast::ast_operator::LOGICAL_AND;
    case sirius::ast::conjunction::kind::op_or: return cudf::ast::ast_operator::LOGICAL_OR;
    default:
      throw invalid_input_exception(
        "[expression_evaluator:conjunction] unrecognized conjunction type {}",
        static_cast<int>(op));
  }
}

cudf::binary_operator conjunction_op_to_binary(sirius::ast::conjunction::kind op)
{
  switch (op) {
    case sirius::ast::conjunction::kind::op_and: return cudf::binary_operator::LOGICAL_AND;
    case sirius::ast::conjunction::kind::op_or: return cudf::binary_operator::LOGICAL_OR;
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

  if (_strategy != expression_evaluator_strategy::MATERIALIZE &&
      (mode == evaluation_mode::AST || ast_op_count >= _min_ast_size)) {
    auto const ast_op = conjunction_op_to_ast(alt.op);

    auto output = evaluate(*alt.children[0], evaluation_mode::AST);

    for (std::size_t i = 1; i < alt.children.size(); ++i) {
      auto child = evaluate(*alt.children[i], evaluation_mode::AST);
      auto const& output_expr =
        _ast_tree.emplace<cudf::ast::operation>(ast_op, output.get_expr(), child.get_expr());
      output = evaluate_result(
        ast_result(output_expr,
                   {output.get_temp_scalar_indices(), child.get_temp_scalar_indices()},
                   {output.get_temp_column_indices(), child.get_temp_column_indices()}));
    }

    if (mode == evaluation_mode::AST) {
      //===----------1: AST Mode----------===//
      return output;
    }
    //===----------2: MATERIALIZE Mode, evaluate node with AST----------===//
    auto result_column = evaluate_ast(output.get_expr());

    release_temporaries(output.get_temp_scalar_indices(), output.get_temp_column_indices());
    return evaluate_result(std::move(result_column));
  }

  //===----------3: MATERIALIZE Mode, evaluate node with unary/binary ops----------===//
  auto const binary_op = conjunction_op_to_binary(alt.op);
  // Conjunction ops always return BOOLEAN.
  auto const output_type = cudf::data_type{cudf::type_id::BOOL8};

  // Resolve the children incrementally into the output
  auto output = evaluate(*alt.children[0], evaluation_mode::MATERIALIZE);
  // DuckDB should prune all scalar conjuncts away
  D_ASSERT(!output.is_scalar());
  for (std::size_t i = 1; i < alt.children.size(); ++i) {
    auto child = evaluate(*alt.children[i], evaluation_mode::MATERIALIZE);
    D_ASSERT(!child.is_scalar());
    auto output_column = cudf::binary_operation(
      output.get_column_view(), child.get_column_view(), binary_op, output_type, _stream, _mr);
    output = evaluate_result(std::move(output_column));
  }
  return output;
}

}  // namespace sirius
