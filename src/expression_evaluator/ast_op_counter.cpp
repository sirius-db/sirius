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
#include <expression/ast/aggregate.hpp>
#include <expression/ast/between.hpp>
#include <expression/ast/case_expr.hpp>
#include <expression/ast/cast.hpp>
#include <expression/ast/coalesce.hpp>
#include <expression/ast/comparison.hpp>
#include <expression/ast/conjunction.hpp>
#include <expression/ast/constant.hpp>
#include <expression/ast/function_call.hpp>
#include <expression/ast/in_list.hpp>
#include <expression/ast/node.hpp>
#include <expression/ast/reference.hpp>
#include <expression/ast/unary_op.hpp>
#include <expression_evaluator/ast_supported_types.hpp>
#include <sirius/exception.hpp>

// standard library
#include <algorithm>
#include <cstddef>
#include <variant>

// Single source of truth for "how many cuDF AST operators does this Sirius AST
// alternative lower to, recursively." Drives the AST-vs-MATERIALIZE strategy
// gate in each expression_evaluator::evaluate(alt, mode) body via the
// `ast_op_count >= _min_ast_size` check.
//
// The method name carries the cuDF coupling: counts are tied to cuDF's
// ast_operator set, supported_ast_cast_types, supported_ast_functions, and
// known cuDF AST limitations (no CASE, no COALESCE, decimal-intermediate bug,
// etc.). A new AST backend would need its own equivalent method, not a
// generic rename.

namespace sirius::ast {

// Leaves — never contribute ops themselves.
std::size_t reference::cudf_ast_op_count() const { return 0; }
std::size_t constant::cudf_ast_op_count() const { return 0; }

// cuDF AST does not support CASE or COALESCE — force materialize.
std::size_t case_expr::cudf_ast_op_count() const { return 0; }
std::size_t coalesce::cudf_ast_op_count() const { return 0; }

std::size_t comparison::cudf_ast_op_count() const
{
  // DISTINCT_FROM lowers to NOT(NULL_EQUAL(...)) — two cuDF AST ops, not one.
  // NOT_DISTINCT_FROM is a single NULL_EQUAL and stays at one.
  std::size_t const self = (op == sirius::comparison_type::distinct_from) ? 2U : 1U;
  return self + left->cudf_ast_op_count() + right->cudf_ast_op_count();
}

std::size_t conjunction::cudf_ast_op_count() const
{
  if (children.empty()) {
    throw invalid_input_exception(
      "[ast::conjunction::cudf_ast_op_count] no children — malformed AST.");
  }
  // Number of AND/OR cuDF ops is one less than number of children.
  std::size_t count = children.size() - 1;
  for (auto const& c : children) {
    count += c->cudf_ast_op_count();
  }
  return count;
}

std::size_t between::cudf_ast_op_count() const
{
  // Lowers to (input >= lower) AND (input <= upper) — 2 comparison ops + 1 AND.
  return 3 + input->cudf_ast_op_count() + lower->cudf_ast_op_count() + upper->cudf_ast_op_count();
}

std::size_t cast::cudf_ast_op_count() const
{
  // Only target types in supported_ast_cast_types_native lower to a single
  // CAST_TO_* cuDF AST op; others force materialize.
  bool const supported = std::find(supported_ast_cast_types_native.begin(),
                                   supported_ast_cast_types_native.end(),
                                   target_type.id()) != supported_ast_cast_types_native.end();
  return supported ? 1 + child->cudf_ast_op_count() : 0;
}

std::size_t unary_op::cudf_ast_op_count() const
{
  switch (op) {
    case kind::op_not: return 1 + child->cudf_ast_op_count();
    case kind::op_is_null: return 1 + child->cudf_ast_op_count();
    case kind::op_is_not_null: return 2 + child->cudf_ast_op_count();  // NOT(IS_NULL(...))
    case kind::op_try:
      throw not_implemented_exception(
        "[ast::unary_op::cudf_ast_op_count] TRY operator is not implemented.");
    default:
      throw internal_exception("[ast::unary_op::cudf_ast_op_count] unknown kind: {}",
                               static_cast<int>(op));
  }
}

std::size_t in_list::cudf_ast_op_count() const
{
  if (values.empty()) {
    throw invalid_input_exception("[ast::in_list::cudf_ast_op_count] no values — malformed AST.");
  }
  // children = [probe] + values; with N = values.size():
  //   num_or = N - 1 ; num_eq = N ; +1 if negated.
  auto const num_or = values.size() - 1;
  auto const num_eq = values.size();
  std::size_t count = num_or + num_eq + (negated ? 1U : 0U);
  count += probe->cudf_ast_op_count();
  for (auto const& v : values) {
    count += v->cudf_ast_op_count();
  }
  return count;
}

std::size_t function_call::cudf_ast_op_count() const
{
  // DECIMAL output -> 0 (cuDF AST chokes on intermediate decimal results
  // until https://github.com/rapidsai/cudf/pull/21996 lands).
  if (return_type().id() == sirius::type_id::DECIMAL) { return 0; }
  bool const supported =
    std::find(supported_ast_functions.begin(), supported_ast_functions.end(), function()) !=
    supported_ast_functions.end();
  if (!supported) { return 0; }
  std::size_t count = 1;
  for (auto const& a : arguments()) {
    count += a->cudf_ast_op_count();
  }
  return count;
}

std::size_t aggregate::cudf_ast_op_count() const
{
  // Aggregate nodes never lower to cuDF AST ops — the method exists solely to
  // keep the node variant exhaustive (node.hpp's has_cudf_ast_op_count concept).
  // Dead at runtime: nothing constructs or visits an aggregate node in the
  // expression-executor path. See https://github.com/sirius-db/sirius/issues/863.
  throw not_implemented_exception(
    "[ast::aggregate::cudf_ast_op_count] aggregate nodes are not lowered to cuDF AST ops (#863).");
}

}  // namespace sirius::ast
