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

// duckdb
#include <duckdb/planner/expression/bound_reference_expression.hpp>

namespace sirius {
using ast_result     = gpu_expression_executor::ast_result;
using execute_result = gpu_expression_executor::execute_result;

execute_result gpu_expression_executor::execute(sirius::ast::reference const& alt,
                                                execution_mode mode)
{
  if (mode == execution_mode::AST) {
    auto const& col_expr = _ast_tree.emplace<cudf::ast::column_reference>(alt.column_index);
    return execute_result(ast_result(col_expr));
  }
  return execute_result(_input_table.column(alt.column_index));
}

execute_result gpu_expression_executor::execute(duckdb::BoundReferenceExpression const& expr,
                                                execution_mode mode)
{
  auto node = sirius::ast::from_duckdb(expr);
  return execute(*node, mode);
}
}  // namespace sirius
