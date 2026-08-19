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
#include <expression_evaluator/ast_supported_types.hpp>
#include <expression_evaluator/expression_evaluator.hpp>
#include <helper/logical_type.hpp>
#include <helper/numeric_narrowing.hpp>
#include <sirius/exception.hpp>

// cudf
#include <cudf/cudf_utils.hpp>
#include <cudf/unary.hpp>

// standard library
#include <algorithm>

namespace sirius {
using evaluate_result = expression_evaluator::evaluate_result;

namespace {

cudf::ast::ast_operator cast_op_to_ast(sirius::type_id id)
{
  switch (id) {
    case sirius::type_id::UBIGINT: return cudf::ast::ast_operator::CAST_TO_UINT64;
    case sirius::type_id::BIGINT: return cudf::ast::ast_operator::CAST_TO_INT64;
    case sirius::type_id::DOUBLE: return cudf::ast::ast_operator::CAST_TO_FLOAT64;
    default:
      throw invalid_input_exception(
        "[cast_op_to_ast] unsupported CAST target type id={}; cuDF AST supports "
        "UBIGINT, BIGINT, DOUBLE.",
        static_cast<int>(id));
  }
}

}  // namespace

evaluate_result expression_evaluator::evaluate(sirius::ast::cast const& alt, evaluation_mode mode)
{
  auto const ast_supported =
    std::find(supported_ast_cast_types_native.begin(),
              supported_ast_cast_types_native.end(),
              alt.target_type.id()) != supported_ast_cast_types_native.end();

  auto const ast_op_count = alt.cudf_ast_op_count();

  // Carrier restores must reach the materialized branch, the only path authorized to use the
  // physical representation tunnel. Semantic casts may use the cuDF AST path.
  if (ast_supported && alt.kind == sirius::ast::cast_kind::semantic &&
      _strategy != expression_evaluator_strategy::MATERIALIZE &&
      (mode == evaluation_mode::AST || ast_op_count >= _min_ast_size)) {
    auto child            = evaluate(*alt.child, evaluation_mode::AST);
    auto const& cast_expr = _ast_tree.emplace<cudf::ast::operation>(
      cast_op_to_ast(alt.target_type.id()), child.get_expr());

    if (mode == evaluation_mode::AST) {
      //===----------1: AST Mode----------===//
      return evaluate_result(compose(cast_expr, {&child}));
    }
    //===----------2: MATERIALIZE Mode, evaluate node with AST----------===//
    auto result_column = evaluate_ast(cast_expr);

    release_temporaries({&child});
    return evaluate_result(std::move(result_column));
  }

  //===----------3: MATERIALIZE Mode, evaluate node with unary/binary ops----------===//
  auto const return_type = sirius::get_cudf_type(alt.target_type);
  auto child             = evaluate(*alt.child, evaluation_mode::MATERIALIZE);
  D_ASSERT(!child.is_scalar());  // CAST should never be called on a scalar
  // Only planner-certified carrier restoration may tunnel through the narrowed representation.
  // A semantic cast delegates to cuDF and is never reinterpreted as a physical DATE restore.
  auto result_column =
    alt.kind == sirius::ast::cast_kind::carrier_restore
      ? sirius::cast_through_rep(child.get_column_view(), return_type, _stream, _mr)
      : cudf::cast(child.get_column_view(), return_type, _stream, _mr);
  if (mode == evaluation_mode::AST) {
    // The parent is executing in AST mode, so add the materialized result to the AST tree.
    return materialize_as_ast_column(std::move(result_column));
  }
  return evaluate_result(std::move(result_column));
}

}  // namespace sirius
