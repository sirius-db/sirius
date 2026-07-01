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

#include "expression/ast/to_duckdb.hpp"

// sirius
#include "expression/ast/aggregate.hpp"
#include "expression/ast/between.hpp"
#include "expression/ast/case_expr.hpp"
#include "expression/ast/cast.hpp"
#include "expression/ast/coalesce.hpp"
#include "expression/ast/comparison.hpp"
#include "expression/ast/conjunction.hpp"
#include "expression/ast/constant.hpp"
#include "expression/ast/function_call.hpp"
#include "expression/ast/in_list.hpp"
#include "expression/ast/node.hpp"
#include "expression/ast/reference.hpp"
#include "expression/ast/unary_op.hpp"
#include "expression/function_id.hpp"
#include "expression/join_condition.hpp"  // sirius::to_duckdb(comparison_type)
#include "expression/value.hpp"           // sirius::to_duckdb(value, logical_type)
#include "helper/type_conversions.hpp"    // sirius::to_duckdb(logical_type)

#include <sirius/exception.hpp>

// duckdb
#include <duckdb/common/enums/expression_type.hpp>
#include <duckdb/common/exception.hpp>
#include <duckdb/function/scalar_function.hpp>
#include <duckdb/planner/expression.hpp>
#include <duckdb/planner/expression/bound_between_expression.hpp>
#include <duckdb/planner/expression/bound_case_expression.hpp>
#include <duckdb/planner/expression/bound_cast_expression.hpp>
#include <duckdb/planner/expression/bound_comparison_expression.hpp>
#include <duckdb/planner/expression/bound_conjunction_expression.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/expression/bound_function_expression.hpp>
#include <duckdb/planner/expression/bound_operator_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>

// standard library
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace sirius::ast {

namespace {

// DuckDB has its own unique_ptr typedef (duckdb::unique_ptr<T,...>) distinct from
// std::unique_ptr; the DuckDB Bound*Expression ctors only accept the DuckDB form.
// These small adapters convert between the two without copying — both share the
// std::default_delete<T> deleter so a release/wrap round-trip is well-defined.
duckdb::unique_ptr<duckdb::Expression> to_duck_ptr(std::unique_ptr<duckdb::Expression> p)
{
  return duckdb::unique_ptr<duckdb::Expression>(p.release());
}

std::unique_ptr<duckdb::Expression> from_duck_ptr(duckdb::unique_ptr<duckdb::Expression> p)
{
  return std::unique_ptr<duckdb::Expression>(p.release());
}

template <class T>
std::unique_ptr<duckdb::Expression> from_duck_derived_ptr(duckdb::unique_ptr<T> p)
{
  // T derives from duckdb::Expression; static_cast widens the pointer.
  return std::unique_ptr<duckdb::Expression>(p.release());
}

}  // namespace

std::unique_ptr<duckdb::Expression> to_duckdb(reference const& alt)
{
  // Reconstruct the reference with its recorded return_type. Executor
  // specializations resolve the column type from the input table_view and
  // ignore this field, but join-key resolution (nested-loop / hash join) reads
  // BoundReferenceExpression::return_type directly to decide whether a cast is
  // needed, so the real type must be preserved. Fall back to INTEGER only when
  // the node carries no type (default-constructed SQLNULL sentinel).
  auto return_type = alt.return_type.id() == sirius::type_id::SQLNULL
                       ? duckdb::LogicalType{duckdb::LogicalTypeId::INTEGER}
                       : sirius::to_duckdb(alt.return_type);
  return from_duck_derived_ptr(duckdb::make_uniq<duckdb::BoundReferenceExpression>(
    std::move(return_type), static_cast<duckdb::idx_t>(alt.column_index)));
}

std::unique_ptr<duckdb::Expression> to_duckdb(constant const& alt)
{
  auto duck_value = sirius::to_duckdb(alt.payload, alt.return_type);
  return from_duck_derived_ptr(
    duckdb::make_uniq<duckdb::BoundConstantExpression>(std::move(duck_value)));
}

std::unique_ptr<duckdb::Expression> to_duckdb(comparison const& alt)
{
  auto left  = to_duck_ptr(to_duckdb(*alt.left));
  auto right = to_duck_ptr(to_duckdb(*alt.right));
  return from_duck_derived_ptr(duckdb::make_uniq<duckdb::BoundComparisonExpression>(
    sirius::to_duckdb(alt.op), std::move(left), std::move(right)));
}

std::unique_ptr<duckdb::Expression> to_duckdb(conjunction const& alt)
{
  duckdb::ExpressionType etype;
  switch (alt.op) {
    case conjunction::kind::op_and: etype = duckdb::ExpressionType::CONJUNCTION_AND; break;
    case conjunction::kind::op_or: etype = duckdb::ExpressionType::CONJUNCTION_OR; break;
    case conjunction::kind::invalid:
    default:
      throw duckdb::InternalException(
        "[sirius::ast::to_duckdb] conjunction with kind::invalid is unreachable; the "
        "Sirius AST translator filters this case before construction");
  }
  auto out = duckdb::make_uniq<duckdb::BoundConjunctionExpression>(etype);
  for (auto const& child : alt.children) {
    out->children.push_back(to_duck_ptr(to_duckdb(*child)));
  }
  return from_duck_derived_ptr(std::move(out));
}

std::unique_ptr<duckdb::Expression> to_duckdb(between const& alt)
{
  return from_duck_derived_ptr(
    duckdb::make_uniq<duckdb::BoundBetweenExpression>(to_duck_ptr(to_duckdb(*alt.input)),
                                                      to_duck_ptr(to_duckdb(*alt.lower)),
                                                      to_duck_ptr(to_duckdb(*alt.upper)),
                                                      alt.lower_inclusive,
                                                      alt.upper_inclusive));
}

std::unique_ptr<duckdb::Expression> to_duckdb(case_expr const& alt)
{
  // Reconstruct the case expression with its recorded return_type. Executor
  // specializations derive the result type from the THEN branches at evaluation
  // time and ignore this field, but the AST-path post_process recovers each
  // expression's return_type via to_duckdb, so the real type must be preserved
  // here to avoid spurious down-casts of the materialized result.
  auto out = duckdb::make_uniq<duckdb::BoundCaseExpression>(sirius::to_duckdb(alt.return_type()));
  out->case_checks.reserve(alt.cases.size());
  for (auto const& wt : alt.cases) {
    duckdb::BoundCaseCheck check;
    check.when_expr = to_duck_ptr(to_duckdb(*wt.when_));
    check.then_expr = to_duck_ptr(to_duckdb(*wt.then_));
    out->case_checks.push_back(std::move(check));
  }
  out->else_expr = to_duck_ptr(to_duckdb(*alt.else_));
  return from_duck_derived_ptr(std::move(out));
}

std::unique_ptr<duckdb::Expression> to_duckdb(cast const& alt)
{
  auto child       = to_duck_ptr(to_duckdb(*alt.child));
  auto target_type = sirius::to_duckdb(alt.target_type);
  if (alt.try_cast) {
    // Try-cast surface uses the full ctor with is_try_cast=true.
    // (BoundCastExpression::AddDefaultCastToType always sets is_try_cast=false.)
    return from_duck_derived_ptr(
      duckdb::make_uniq<duckdb::BoundCastExpression>(std::move(child),
                                                     std::move(target_type),
                                                     duckdb::BoundCastInfo{nullptr},
                                                     /*is_try_cast=*/true));
  }
  return from_duck_ptr(
    duckdb::BoundCastExpression::AddDefaultCastToType(std::move(child), std::move(target_type)));
}

std::unique_ptr<duckdb::Expression> to_duckdb(unary_op const& alt)
{
  duckdb::ExpressionType etype;
  switch (alt.op) {
    case unary_op::kind::op_not: etype = duckdb::ExpressionType::OPERATOR_NOT; break;
    case unary_op::kind::op_is_null: etype = duckdb::ExpressionType::OPERATOR_IS_NULL; break;
    case unary_op::kind::op_is_not_null:
      etype = duckdb::ExpressionType::OPERATOR_IS_NOT_NULL;
      break;
    case unary_op::kind::op_try: etype = duckdb::ExpressionType::OPERATOR_TRY; break;
    case unary_op::kind::invalid:
    default:
      throw duckdb::InternalException(
        "[sirius::ast::to_duckdb] unary_op with kind::invalid is unreachable; the "
        "Sirius AST translator filters this case before construction");
  }
  auto out =
    duckdb::make_uniq<duckdb::BoundOperatorExpression>(etype, duckdb::LogicalType::BOOLEAN);
  out->children.push_back(to_duck_ptr(to_duckdb(*alt.child)));
  return from_duck_derived_ptr(std::move(out));
}

std::unique_ptr<duckdb::Expression> to_duckdb(coalesce const& alt)
{
  // COALESCE result type is the common type of its children. The downstream
  // executor specialization re-derives this from the children at evaluation
  // time and ignores this field, but the AST-path post_process recovers each
  // expression's return_type via to_duckdb, so the real type must be preserved
  // here to avoid spurious down-casts of the materialized result.
  auto out = duckdb::make_uniq<duckdb::BoundOperatorExpression>(
    duckdb::ExpressionType::OPERATOR_COALESCE, sirius::to_duckdb(alt.return_type()));
  for (auto const& child : alt.children) {
    out->children.push_back(to_duck_ptr(to_duckdb(*child)));
  }
  return from_duck_derived_ptr(std::move(out));
}

std::unique_ptr<duckdb::Expression> to_duckdb(in_list const& alt)
{
  auto out = duckdb::make_uniq<duckdb::BoundOperatorExpression>(
    alt.negated ? duckdb::ExpressionType::COMPARE_NOT_IN : duckdb::ExpressionType::COMPARE_IN,
    duckdb::LogicalType::BOOLEAN);
  out->children.push_back(to_duck_ptr(to_duckdb(*alt.probe)));
  for (auto const& value : alt.values) {
    out->children.push_back(to_duck_ptr(to_duckdb(*value)));
  }
  return from_duck_derived_ptr(std::move(out));
}

std::unique_ptr<duckdb::Expression> to_duckdb(function_call const& alt)
{
  auto return_type = sirius::to_duckdb(alt.return_type());

  // Minimal ScalarFunction stub: name + empty argument-types + return_type +
  // null function pointer. The downstream executor specialization
  // (function.cpp) dispatches via the function-name lookup, never
  // invokes function_ptr_t or inspects arguments().
  auto name = std::string{sirius::to_duckdb_function_name(alt.function())};
  duckdb::ScalarFunction stub_fn(
    std::move(name), duckdb::vector<duckdb::LogicalType>{}, return_type, /*function=*/nullptr);

  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> children;
  children.reserve(alt.arguments().size());
  for (auto const& arg : alt.arguments()) {
    children.push_back(to_duck_ptr(to_duckdb(*arg)));
  }

  return from_duck_derived_ptr(
    duckdb::make_uniq<duckdb::BoundFunctionExpression>(std::move(return_type),
                                                       std::move(stub_fn),
                                                       std::move(children),
                                                       /*bind_info=*/nullptr));
}

std::unique_ptr<duckdb::Expression> to_duckdb(aggregate const& /*alt*/)
{
  throw not_implemented_exception(
    "[sirius::ast::to_duckdb] aggregate reconstruction is not implemented; consumers read the "
    "aggregate node natively (#863).");
}

std::unique_ptr<duckdb::Expression> to_duckdb(node const& expr)
{
  return std::visit([](auto const& alt) { return to_duckdb(alt); }, expr.v);
}

}  // namespace sirius::ast
