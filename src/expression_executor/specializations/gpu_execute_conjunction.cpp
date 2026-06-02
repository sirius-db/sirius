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
#include <expression/ast/from_duckdb.hpp>
#include <expression/ast/node.hpp>
#include <expression_executor/gpu_expression_executor.hpp>
#include <sirius/exception.hpp>

// duckdb
#include <duckdb/common/exception.hpp>
#include <duckdb/planner/expression/bound_conjunction_expression.hpp>

// cudf
#include <cudf/binaryop.hpp>
#include <cudf/cudf_utils.hpp>

namespace sirius {
using execute_result = gpu_expression_executor::execute_result;

namespace {

cudf::ast::ast_operator conjunction_op_to_ast(sirius::ast::conjunction::kind op)
{
  switch (op) {
    case sirius::ast::conjunction::kind::op_and: return cudf::ast::ast_operator::LOGICAL_AND;
    case sirius::ast::conjunction::kind::op_or: return cudf::ast::ast_operator::LOGICAL_OR;
    default:
      throw invalid_input_exception(
        "[gpu_expression_executor:conjunction] unrecognized conjunction type {}",
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
        "[gpu_expression_executor:conjunction] unrecognized conjunction type {}",
        static_cast<int>(op));
  }
}

}  // namespace

execute_result gpu_expression_executor::execute(sirius::ast::conjunction const& alt,
                                                execution_mode mode)
{
  if (alt.children.empty()) {
    throw invalid_input_exception(
      "[gpu_expression_executor:conjunction] conjunction has no children — malformed AST.");
  }
  auto const ast_op_count = alt.cudf_ast_op_count();

  if (_strategy != expression_executor_strategy::MATERIALIZE &&
      (mode == execution_mode::AST || ast_op_count >= _min_ast_size)) {
    auto const ast_op = conjunction_op_to_ast(alt.op);

    auto output = execute(*alt.children[0], execution_mode::AST);

    for (std::size_t i = 1; i < alt.children.size(); ++i) {
      auto child = execute(*alt.children[i], execution_mode::AST);
      auto const& output_expr =
        _ast_tree.emplace<cudf::ast::operation>(ast_op, output.get_expr(), child.get_expr());
      output = execute_result(
        ast_result(output_expr,
                   {output.get_temp_scalar_indices(), child.get_temp_scalar_indices()},
                   {output.get_temp_column_indices(), child.get_temp_column_indices()}));
    }

    if (mode == execution_mode::AST) {
      //===----------1: AST Mode----------===//
      return output;
    }
    //===----------2: MATERIALIZE Mode, evaluate node with AST----------===//
    auto result_column = execute_ast(output.get_expr());

    release_temporaries(output.get_temp_scalar_indices(), output.get_temp_column_indices());
    return execute_result(std::move(result_column));
  }

  //===----------3: MATERIALIZE Mode, evaluate node with unary/binary ops----------===//
  auto const binary_op = conjunction_op_to_binary(alt.op);
  // Conjunction ops always return BOOLEAN.
  auto const output_type = cudf::data_type{cudf::type_id::BOOL8};

  // Resolve the children incrementally into the output
  auto output = execute(*alt.children[0], execution_mode::MATERIALIZE);
  // DuckDB should prune all scalar conjuncts away
  D_ASSERT(!output.is_scalar());
  for (std::size_t i = 1; i < alt.children.size(); ++i) {
    auto child = execute(*alt.children[i], execution_mode::MATERIALIZE);
    D_ASSERT(!child.is_scalar());
    auto output_column = cudf::binary_operation(
      output.get_column_view(), child.get_column_view(), binary_op, output_type, _stream, _mr);
    output = execute_result(std::move(output_column));
  }
  return output;
}

// DuckDB-typed entrypoint. Bridges callers that still pass duckdb::Expression
// directly into the executor; the eventual home for this from_duckdb step is
// the planning stage so the executor sees only native sirius::ast types, but
// until upstream call sites are migrated this overload (and the duckdb
// includes it requires) must stay.
execute_result gpu_expression_executor::execute(duckdb::BoundConjunctionExpression const& expr,
                                                execution_mode mode)
{
  auto node = sirius::ast::from_duckdb(expr);
  if (!node) {
    throw not_implemented_exception(
      "[gpu_expression_executor:conjunction] BoundConjunctionExpression with conjunction type {} "
      "could not be lowered to a Sirius AST node.",
      static_cast<int>(expr.GetExpressionType()));
  }
  return execute(*node, mode);
}

}  // namespace sirius
