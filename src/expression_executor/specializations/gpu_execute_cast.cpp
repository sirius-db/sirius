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
#include <expression_executor/ast_supported_types.hpp>
#include <expression_executor/gpu_expression_executor.hpp>
#include <helper/logical_type.hpp>
#include <sirius/exception.hpp>

// duckdb
#include <duckdb/planner/expression/bound_cast_expression.hpp>

// cudf
#include <cudf/cudf_utils.hpp>
#include <cudf/unary.hpp>

// standard library
#include <algorithm>

namespace sirius {
using execute_result = gpu_expression_executor::execute_result;

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

execute_result gpu_expression_executor::execute(sirius::ast::cast const& alt, execution_mode mode)
{
  auto const ast_supported =
    std::find(supported_ast_cast_types_native.begin(),
              supported_ast_cast_types_native.end(),
              alt.target_type.id()) != supported_ast_cast_types_native.end();

  auto const ast_op_count = alt.cudf_ast_op_count();

  if (ast_supported && _strategy != expression_executor_strategy::MATERIALIZE &&
      (mode == execution_mode::AST || ast_op_count >= _min_ast_size)) {
    auto child            = execute(*alt.child, execution_mode::AST);
    auto const& cast_expr = _ast_tree.emplace<cudf::ast::operation>(
      cast_op_to_ast(alt.target_type.id()), child.get_expr());

    if (mode == execution_mode::AST) {
      //===----------1: AST Mode----------===//
      return execute_result(
        ast_result(cast_expr, child.get_temp_scalar_indices(), child.get_temp_column_indices()));
    }
    //===----------2: MATERIALIZE Mode, evaluate node with AST----------===//
    auto result_column = execute_ast(cast_expr);

    release_temporaries(child.get_temp_scalar_indices(), child.get_temp_column_indices());
    return execute_result(std::move(result_column));
  }

  //===----------3: MATERIALIZE Mode, evaluate node with unary/binary ops----------===//
  auto const return_type = sirius::get_cudf_type(alt.target_type);
  auto child             = execute(*alt.child, execution_mode::MATERIALIZE);
  D_ASSERT(!child.is_scalar());  // CAST should never be called on a scalar
  auto result_column = cudf::cast(child.get_column_view(), return_type, _stream, _mr);
  if (mode == execution_mode::AST) {
    // The parent is executing in AST mode, so add the materialized result to the AST tree.
    return materialize_as_ast_column(std::move(result_column));
  }
  return execute_result(std::move(result_column));
}

// DuckDB-typed entrypoint. Bridges callers that still pass duckdb::Expression
// directly into the executor; the eventual home for this from_duckdb step is
// the planning stage so the executor sees only native sirius::ast types, but
// until upstream call sites are migrated this overload (and the duckdb
// includes it requires) must stay.
execute_result gpu_expression_executor::execute(duckdb::BoundCastExpression const& expr,
                                                execution_mode mode)
{
  auto node = sirius::ast::from_duckdb(expr);
  if (!node) {
    throw not_implemented_exception(
      "[gpu_expression_executor:cast] BoundCastExpression could not be lowered to a Sirius AST "
      "node (the cast child is unsupported).");
  }
  return execute(*node, mode);
}

}  // namespace sirius
