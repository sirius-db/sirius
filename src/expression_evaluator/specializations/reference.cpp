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

namespace sirius {
using ast_result      = expression_evaluator::ast_result;
using evaluate_result = expression_evaluator::evaluate_result;

evaluate_result expression_evaluator::evaluate(sirius::ast::reference const& alt,
                                               evaluation_mode mode)
{
  if (mode == evaluation_mode::AST) {
    auto const& col_expr = _ast_tree.emplace<cudf::ast::column_reference>(alt.column_index);
    return evaluate_result(ast_result(col_expr));
  }
  return evaluate_result(_input_table.column(alt.column_index));
}
}  // namespace sirius
