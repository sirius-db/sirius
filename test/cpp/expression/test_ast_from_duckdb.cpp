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

// Tests for sirius::ast::from_duckdb — the DuckDB-to-Sirius expression
// translator.
//
// Covers the full BoundExpression dispatch surface via direct construction of
// duckdb::BoundXxxExpression instances, plus the per-class subtype fallback
// (unsupported-but-well-formed → nullptr) and the BOUND_OPERATOR demultiplex
// table. One additional case exercises the translator on a real DuckDB Binder
// output, end-to-end from SQL string to Sirius AST tree.

#include "ast_test_builders.hpp"
#include "catch.hpp"
#include "duckdb/main/settings.hpp"
#include "expression/ast/from_duckdb.hpp"

// sirius — node accessors and per-node struct types
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
#include "expression/join_condition.hpp"  // sirius::comparison_type

// duckdb — direct-ctor construction surface
#include <duckdb/common/exception.hpp>
#include <duckdb/common/types/value.hpp>
#include <duckdb/function/scalar_function.hpp>
#include <duckdb/main/client_config.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/main/connection.hpp>
#include <duckdb/main/database.hpp>
#include <duckdb/planner/expression/bound_between_expression.hpp>
#include <duckdb/planner/expression/bound_case_expression.hpp>
#include <duckdb/planner/expression/bound_cast_expression.hpp>
#include <duckdb/planner/expression/bound_comparison_expression.hpp>
#include <duckdb/planner/expression/bound_conjunction_expression.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/expression/bound_function_expression.hpp>
#include <duckdb/planner/expression/bound_operator_expression.hpp>
#include <duckdb/planner/expression/bound_parameter_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/logical_operator.hpp>

// standard library
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

using duckdb::BoundBetweenExpression;
using duckdb::BoundCaseCheck;
using duckdb::BoundCaseExpression;
using duckdb::BoundCastExpression;
using duckdb::BoundComparisonExpression;
using duckdb::BoundConjunctionExpression;
using duckdb::BoundConstantExpression;
using duckdb::BoundFunctionExpression;
using duckdb::BoundOperatorExpression;
using duckdb::BoundParameterExpression;
using duckdb::BoundReferenceExpression;
using duckdb::ExpressionType;
using duckdb::LogicalType;
using duckdb::LogicalTypeId;
using duckdb::ScalarFunction;
using duckdb::Value;

using sirius::ast::between;
using sirius::ast::case_expr;
using sirius::ast::cast;
using sirius::ast::coalesce;
using sirius::ast::comparison;
using sirius::ast::conjunction;
using sirius::ast::constant;
using sirius::ast::function_call;
using sirius::ast::in_list;
using sirius::ast::node;
using sirius::ast::reference;
using sirius::ast::unary_op;
using sirius::ast::test::make_bound_int_const;
using sirius::ast::test::make_bound_ref;

// ============================================================================
// BOUND_REFERENCE
// ============================================================================

TEST_CASE("ast_from_duckdb - BOUND_REF translates to reference node (column 0)",
          "[ast_from_duckdb]")
{
  auto expr = make_bound_ref(0);
  auto out  = sirius::ast::from_duckdb(*expr);
  REQUIRE(out);
  REQUIRE(out->holds<reference>());
  REQUIRE(out->get<reference>().column_index == 0);
}

TEST_CASE("ast_from_duckdb - BOUND_REF translates to reference node (column 5)",
          "[ast_from_duckdb]")
{
  auto expr = make_bound_ref(5);
  auto out  = sirius::ast::from_duckdb(*expr);
  REQUIRE(out);
  REQUIRE(out->holds<reference>());
  REQUIRE(out->get<reference>().column_index == 5);
}

// ============================================================================
// BOUND_CONSTANT
// ============================================================================

TEST_CASE("ast_from_duckdb - BOUND_CONSTANT INTEGER translates to constant node",
          "[ast_from_duckdb]")
{
  auto expr = make_bound_int_const(42);
  auto out  = sirius::ast::from_duckdb(*expr);
  REQUIRE(out);
  REQUIRE(out->holds<constant>());
  auto const& c = out->get<constant>();
  REQUIRE(c.return_type().id() == sirius::type_id::INTEGER);
  REQUIRE(std::holds_alternative<int32_t>(c.payload));
  REQUIRE(std::get<int32_t>(c.payload) == 42);
}

TEST_CASE("ast_from_duckdb - BOUND_CONSTANT VARCHAR translates to constant node",
          "[ast_from_duckdb]")
{
  auto expr = duckdb::make_uniq<BoundConstantExpression>(Value("hi"));
  auto out  = sirius::ast::from_duckdb(*expr);
  REQUIRE(out);
  REQUIRE(out->holds<constant>());
  auto const& c = out->get<constant>();
  REQUIRE(c.return_type().id() == sirius::type_id::VARCHAR);
  REQUIRE(std::holds_alternative<std::string>(c.payload));
  REQUIRE(std::get<std::string>(c.payload) == "hi");
}

TEST_CASE("ast_from_duckdb - BOUND_CONSTANT NULL of INTEGER preserves type", "[ast_from_duckdb]")
{
  auto expr =
    duckdb::make_uniq<BoundConstantExpression>(Value(LogicalType{LogicalTypeId::INTEGER}));
  auto out = sirius::ast::from_duckdb(*expr);
  REQUIRE(out);
  REQUIRE(out->holds<constant>());
  auto const& c = out->get<constant>();
  REQUIRE(c.return_type().id() == sirius::type_id::INTEGER);
  REQUIRE(std::holds_alternative<sirius::null_value>(c.payload));
}

// ============================================================================
// BOUND_COMPARISON — one TEST_CASE per supported comparison subtype + the
// distinct-from fallback.
// ============================================================================

TEST_CASE("ast_from_duckdb - BOUND_COMPARISON EQUAL translates to comparison node",
          "[ast_from_duckdb]")
{
  auto left  = make_bound_ref(0);
  auto right = make_bound_int_const(3);
  auto expr  = BoundComparisonExpression::Create(
    ExpressionType::COMPARE_EQUAL, std::move(left), std::move(right));
  auto out = sirius::ast::from_duckdb(*expr);
  REQUIRE(out);
  REQUIRE(out->holds<comparison>());
  auto const& cmp = out->get<comparison>();
  REQUIRE(cmp.op == sirius::comparison_type::equal);
  REQUIRE(cmp.left);
  REQUIRE(cmp.right);
  REQUIRE(cmp.left->holds<reference>());
  REQUIRE(cmp.right->holds<constant>());
}

TEST_CASE("ast_from_duckdb - BOUND_COMPARISON NOTEQUAL translates to comparison node",
          "[ast_from_duckdb]")
{
  auto left  = make_bound_ref(0);
  auto right = make_bound_int_const(3);
  auto expr  = BoundComparisonExpression::Create(
    ExpressionType::COMPARE_NOTEQUAL, std::move(left), std::move(right));
  auto out = sirius::ast::from_duckdb(*expr);
  REQUIRE(out);
  REQUIRE(out->holds<comparison>());
  REQUIRE(out->get<comparison>().op == sirius::comparison_type::not_equal);
}

TEST_CASE("ast_from_duckdb - BOUND_COMPARISON LESSTHAN translates to comparison node",
          "[ast_from_duckdb]")
{
  auto left  = make_bound_ref(0);
  auto right = make_bound_int_const(3);
  auto expr  = BoundComparisonExpression::Create(
    ExpressionType::COMPARE_LESSTHAN, std::move(left), std::move(right));
  auto out = sirius::ast::from_duckdb(*expr);
  REQUIRE(out);
  REQUIRE(out->holds<comparison>());
  REQUIRE(out->get<comparison>().op == sirius::comparison_type::lt);
}

TEST_CASE("ast_from_duckdb - BOUND_COMPARISON LESSTHANOREQUALTO translates to comparison node",
          "[ast_from_duckdb]")
{
  auto left  = make_bound_ref(0);
  auto right = make_bound_int_const(3);
  auto expr  = BoundComparisonExpression::Create(
    ExpressionType::COMPARE_LESSTHANOREQUALTO, std::move(left), std::move(right));
  auto out = sirius::ast::from_duckdb(*expr);
  REQUIRE(out);
  REQUIRE(out->holds<comparison>());
  REQUIRE(out->get<comparison>().op == sirius::comparison_type::le);
}

TEST_CASE("ast_from_duckdb - BOUND_COMPARISON GREATERTHAN translates to comparison node",
          "[ast_from_duckdb]")
{
  auto left  = make_bound_ref(0);
  auto right = make_bound_int_const(3);
  auto expr  = BoundComparisonExpression::Create(
    ExpressionType::COMPARE_GREATERTHAN, std::move(left), std::move(right));
  auto out = sirius::ast::from_duckdb(*expr);
  REQUIRE(out);
  REQUIRE(out->holds<comparison>());
  REQUIRE(out->get<comparison>().op == sirius::comparison_type::gt);
}

TEST_CASE("ast_from_duckdb - BOUND_COMPARISON GREATERTHANOREQUALTO translates to comparison node",
          "[ast_from_duckdb]")
{
  auto left  = make_bound_ref(0);
  auto right = make_bound_int_const(3);
  auto expr  = BoundComparisonExpression::Create(
    ExpressionType::COMPARE_GREATERTHANOREQUALTO, std::move(left), std::move(right));
  auto out = sirius::ast::from_duckdb(*expr);
  REQUIRE(out);
  REQUIRE(out->holds<comparison>());
  REQUIRE(out->get<comparison>().op == sirius::comparison_type::ge);
}

TEST_CASE("ast_from_duckdb - BOUND_COMPARISON DISTINCT_FROM translates to comparison node",
          "[ast_from_duckdb]")
{
  auto left  = make_bound_ref(0);
  auto right = make_bound_int_const(3);
  auto expr  = BoundComparisonExpression::Create(
    ExpressionType::COMPARE_DISTINCT_FROM, std::move(left), std::move(right));
  auto out = sirius::ast::from_duckdb(*expr);
  REQUIRE(out);
  REQUIRE(out->holds<comparison>());
  REQUIRE(out->get<comparison>().op == sirius::comparison_type::distinct_from);
}

TEST_CASE("ast_from_duckdb - BOUND_COMPARISON NOT_DISTINCT_FROM translates to comparison node",
          "[ast_from_duckdb]")
{
  auto left  = make_bound_ref(0);
  auto right = make_bound_int_const(3);
  auto expr  = BoundComparisonExpression::Create(
    ExpressionType::COMPARE_NOT_DISTINCT_FROM, std::move(left), std::move(right));
  auto out = sirius::ast::from_duckdb(*expr);
  REQUIRE(out);
  REQUIRE(out->holds<comparison>());
  REQUIRE(out->get<comparison>().op == sirius::comparison_type::not_distinct_from);
}

// ============================================================================
// BOUND_CONJUNCTION
// ============================================================================

TEST_CASE("ast_from_duckdb - BOUND_CONJUNCTION AND translates to conjunction(op_and)",
          "[ast_from_duckdb]")
{
  auto and_expr = duckdb::make_uniq<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_AND);
  and_expr->GetChildrenMutable().push_back(make_bound_ref(0, LogicalTypeId::BOOLEAN));
  and_expr->GetChildrenMutable().push_back(make_bound_ref(1, LogicalTypeId::BOOLEAN));
  auto out = sirius::ast::from_duckdb(*and_expr);
  REQUIRE(out);
  REQUIRE(out->holds<conjunction>());
  auto const& conj = out->get<conjunction>();
  REQUIRE(conj.op == conjunction::kind::op_and);
  REQUIRE(conj.children.size() == 2);
  REQUIRE(conj.children[0]);
  REQUIRE(conj.children[1]);
}

TEST_CASE("ast_from_duckdb - BOUND_CONJUNCTION OR translates to conjunction(op_or)",
          "[ast_from_duckdb]")
{
  auto or_expr = duckdb::make_uniq<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_OR);
  or_expr->GetChildrenMutable().push_back(make_bound_ref(0, LogicalTypeId::BOOLEAN));
  or_expr->GetChildrenMutable().push_back(make_bound_ref(1, LogicalTypeId::BOOLEAN));
  auto out = sirius::ast::from_duckdb(*or_expr);
  REQUIRE(out);
  REQUIRE(out->holds<conjunction>());
  REQUIRE(out->get<conjunction>().op == conjunction::kind::op_or);
}

// ============================================================================
// BOUND_BETWEEN
// ============================================================================

TEST_CASE("ast_from_duckdb - BOUND_BETWEEN translates to between node with inclusive bounds",
          "[ast_from_duckdb]")
{
  auto bt  = BoundBetweenExpression::Create(make_bound_ref(0),
                                           make_bound_int_const(1),
                                           make_bound_int_const(10),
                                           /*lower_inclusive=*/true,
                                           /*upper_inclusive=*/true);
  auto out = sirius::ast::from_duckdb(*bt);
  REQUIRE(out);
  REQUIRE(out->holds<between>());
  auto const& bw = out->get<between>();
  REQUIRE(bw.input);
  REQUIRE(bw.lower);
  REQUIRE(bw.upper);
  REQUIRE(bw.lower_inclusive == true);
  REQUIRE(bw.upper_inclusive == true);
  REQUIRE(bw.input->holds<reference>());
  REQUIRE(bw.lower->holds<constant>());
  REQUIRE(bw.upper->holds<constant>());
}

// ============================================================================
// BOUND_CASE
// ============================================================================

TEST_CASE("ast_from_duckdb - BOUND_CASE WHEN/THEN/ELSE translates to case_expr",
          "[ast_from_duckdb]")
{
  // CASE WHEN col(0) = 1 THEN 10 ELSE 0 END
  auto check_expr = BoundComparisonExpression::Create(
    ExpressionType::COMPARE_EQUAL, make_bound_ref(0), make_bound_int_const(1));
  auto then_expr = make_bound_int_const(10);

  BoundCaseCheck case_check;
  case_check.when_expr = std::move(check_expr);
  case_check.then_expr = std::move(then_expr);

  auto else_expr = make_bound_int_const(0);
  auto case_node = duckdb::make_uniq<BoundCaseExpression>(LogicalType{LogicalTypeId::INTEGER});
  case_node->ElseMutable() = std::move(else_expr);
  case_node->CaseChecksMutable().push_back(std::move(case_check));

  auto out = sirius::ast::from_duckdb(*case_node);
  REQUIRE(out);
  REQUIRE(out->holds<case_expr>());
  auto const& ce = out->get<case_expr>();
  REQUIRE(ce.cases.size() == 1);
  REQUIRE(ce.cases[0].when_);
  REQUIRE(ce.cases[0].then_);
  REQUIRE(ce.else_);
}

TEST_CASE("ast_from_duckdb - BOUND_CASE with unsupported WHEN propagates nullptr",
          "[ast_from_duckdb]")
{
  // The WHEN subexpression is a BoundParameterExpression (unsupported -> nullptr).
  // The whole CASE must collapse to nullptr.
  auto bad_when  = duckdb::make_uniq<BoundParameterExpression>(duckdb::Identifier{"p_when"});
  auto then_expr = make_bound_int_const(10);

  BoundCaseCheck case_check;
  case_check.when_expr = std::move(bad_when);
  case_check.then_expr = std::move(then_expr);

  auto else_expr = make_bound_int_const(0);
  auto case_node = duckdb::make_uniq<BoundCaseExpression>(LogicalType{LogicalTypeId::INTEGER});
  case_node->ElseMutable() = std::move(else_expr);
  case_node->CaseChecksMutable().push_back(std::move(case_check));

  REQUIRE(sirius::ast::from_duckdb(*case_node) == nullptr);
}

// ============================================================================
// BOUND_CAST
// ============================================================================

TEST_CASE("ast_from_duckdb - BOUND_CAST INTEGER -> BIGINT translates to cast node",
          "[ast_from_duckdb]")
{
  auto child = make_bound_ref(0);
  auto cast_expr =
    BoundCastExpression::AddDefaultCastToType(std::move(child), LogicalType{LogicalTypeId::BIGINT});
  auto out = sirius::ast::from_duckdb(*cast_expr);
  REQUIRE(out);
  REQUIRE(out->holds<cast>());
  auto const& c = out->get<cast>();
  REQUIRE(c.child);
  REQUIRE(c.child->holds<reference>());
  REQUIRE(c.target_type.id() == sirius::type_id::BIGINT);
  REQUIRE(c.try_cast == false);
}

TEST_CASE("ast_from_duckdb - BOUND_CAST honors try_cast = true", "[ast_from_duckdb]")
{
  auto child     = make_bound_ref(0);
  auto cast_expr = BoundCastExpression::AddDefaultCastToType(
    std::move(child), LogicalType{LogicalTypeId::BIGINT}, /*try_cast=*/true);
  auto out = sirius::ast::from_duckdb(*cast_expr);
  REQUIRE(out);
  REQUIRE(out->holds<cast>());
  auto const& c = out->get<cast>();
  REQUIRE(c.target_type.id() == sirius::type_id::BIGINT);
  REQUIRE(c.try_cast == true);
}

// ============================================================================
// BOUND_FUNCTION
// ============================================================================

TEST_CASE("ast_from_duckdb - BOUND_FUNCTION '+' resolves to function_id::add", "[ast_from_duckdb]")
{
  auto add_expr = duckdb::make_uniq<BoundFunctionExpression>(
    duckdb::BoundScalarFunction(ScalarFunction(
      "+", {LogicalType::INTEGER, LogicalType::INTEGER}, LogicalType::INTEGER, nullptr)),
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>{},
    nullptr);
  add_expr->GetChildrenMutable().push_back(make_bound_ref(0));
  add_expr->GetChildrenMutable().push_back(make_bound_int_const(3));

  auto out = sirius::ast::from_duckdb(*add_expr);
  REQUIRE(out);
  REQUIRE(out->holds<function_call>());
  auto const& fc = out->get<function_call>();
  REQUIRE(fc.function() == sirius::function_id::add);
  REQUIRE(fc.arguments().size() == 2);
  REQUIRE(fc.return_type().id() == sirius::type_id::INTEGER);
}

TEST_CASE("ast_from_duckdb - BOUND_FUNCTION 'substring' resolves to function_id::substring",
          "[ast_from_duckdb]")
{
  auto fn_expr = duckdb::make_uniq<BoundFunctionExpression>(
    duckdb::BoundScalarFunction(
      ScalarFunction("substring",
                     {LogicalType::VARCHAR, LogicalType::INTEGER, LogicalType::INTEGER},
                     LogicalType::VARCHAR,
                     nullptr)),
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>{},
    nullptr);
  fn_expr->GetChildrenMutable().push_back(make_bound_ref(0, LogicalTypeId::VARCHAR));
  fn_expr->GetChildrenMutable().push_back(make_bound_int_const(1));
  fn_expr->GetChildrenMutable().push_back(make_bound_int_const(3));

  auto out = sirius::ast::from_duckdb(*fn_expr);
  REQUIRE(out);
  REQUIRE(out->holds<function_call>());
  REQUIRE(out->get<function_call>().function() == sirius::function_id::substring);
}

TEST_CASE("ast_from_duckdb - BOUND_FUNCTION 'substr' alias resolves to function_id::substring",
          "[ast_from_duckdb]")
{
  auto fn_expr = duckdb::make_uniq<BoundFunctionExpression>(
    duckdb::BoundScalarFunction(
      ScalarFunction("substr",
                     {LogicalType::VARCHAR, LogicalType::INTEGER, LogicalType::INTEGER},
                     LogicalType::VARCHAR,
                     nullptr)),
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>{},
    nullptr);
  fn_expr->GetChildrenMutable().push_back(make_bound_ref(0, LogicalTypeId::VARCHAR));
  fn_expr->GetChildrenMutable().push_back(make_bound_int_const(1));
  fn_expr->GetChildrenMutable().push_back(make_bound_int_const(3));

  auto out = sirius::ast::from_duckdb(*fn_expr);
  REQUIRE(out);
  REQUIRE(out->holds<function_call>());
  REQUIRE(out->get<function_call>().function() == sirius::function_id::substring);
}

TEST_CASE("ast_from_duckdb - BOUND_FUNCTION unknown name returns nullptr", "[ast_from_duckdb]")
{
  auto fn_expr = duckdb::make_uniq<BoundFunctionExpression>(
    duckdb::BoundScalarFunction(
      ScalarFunction("nonexistent_fn", {LogicalType::INTEGER}, LogicalType::INTEGER, nullptr)),
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>{},
    nullptr);
  fn_expr->GetChildrenMutable().push_back(make_bound_ref(0));

  REQUIRE(sirius::ast::from_duckdb(*fn_expr) == nullptr);
}

// ============================================================================
// BOUND_OPERATOR — full demultiplex table coverage (NOT / IS_NULL /
// IS_NOT_NULL / TRY / COALESCE / COMPARE_IN / COMPARE_NOT_IN), plus one
// unsupported ExpressionType that signals fallback via nullptr.
// ============================================================================

TEST_CASE("ast_from_duckdb - BOUND_OPERATOR NOT translates to unary_op(op_not)",
          "[ast_from_duckdb]")
{
  auto not_expr = duckdb::make_uniq<BoundOperatorExpression>(ExpressionType::OPERATOR_NOT,
                                                             LogicalType{LogicalTypeId::BOOLEAN});
  not_expr->GetChildrenMutable().push_back(make_bound_ref(0, LogicalTypeId::BOOLEAN));
  auto out = sirius::ast::from_duckdb(*not_expr);
  REQUIRE(out);
  REQUIRE(out->holds<unary_op>());
  auto const& uo = out->get<unary_op>();
  REQUIRE(uo.op == unary_op::kind::op_not);
  REQUIRE(uo.child);
}

TEST_CASE("ast_from_duckdb - BOUND_OPERATOR IS_NULL translates to unary_op(op_is_null)",
          "[ast_from_duckdb]")
{
  auto is_null_expr = duckdb::make_uniq<BoundOperatorExpression>(
    ExpressionType::OPERATOR_IS_NULL, LogicalType{LogicalTypeId::BOOLEAN});
  is_null_expr->GetChildrenMutable().push_back(make_bound_ref(0));
  auto out = sirius::ast::from_duckdb(*is_null_expr);
  REQUIRE(out);
  REQUIRE(out->holds<unary_op>());
  REQUIRE(out->get<unary_op>().op == unary_op::kind::op_is_null);
}

TEST_CASE("ast_from_duckdb - BOUND_OPERATOR IS_NOT_NULL translates to unary_op(op_is_not_null)",
          "[ast_from_duckdb]")
{
  auto is_not_null_expr = duckdb::make_uniq<BoundOperatorExpression>(
    ExpressionType::OPERATOR_IS_NOT_NULL, LogicalType{LogicalTypeId::BOOLEAN});
  is_not_null_expr->GetChildrenMutable().push_back(make_bound_ref(0));
  auto out = sirius::ast::from_duckdb(*is_not_null_expr);
  REQUIRE(out);
  REQUIRE(out->holds<unary_op>());
  REQUIRE(out->get<unary_op>().op == unary_op::kind::op_is_not_null);
}

TEST_CASE("ast_from_duckdb - BOUND_OPERATOR TRY translates to unary_op(op_try)",
          "[ast_from_duckdb]")
{
  auto try_expr = duckdb::make_uniq<BoundOperatorExpression>(ExpressionType::OPERATOR_TRY,
                                                             LogicalType{LogicalTypeId::INTEGER});
  try_expr->GetChildrenMutable().push_back(make_bound_ref(0));
  auto out = sirius::ast::from_duckdb(*try_expr);
  REQUIRE(out);
  REQUIRE(out->holds<unary_op>());
  REQUIRE(out->get<unary_op>().op == unary_op::kind::op_try);
}

TEST_CASE("ast_from_duckdb - BOUND_OPERATOR COALESCE translates to coalesce(N children)",
          "[ast_from_duckdb]")
{
  auto coalesce_expr = duckdb::make_uniq<BoundOperatorExpression>(
    ExpressionType::OPERATOR_COALESCE, LogicalType{LogicalTypeId::INTEGER});
  coalesce_expr->GetChildrenMutable().push_back(make_bound_ref(0));
  coalesce_expr->GetChildrenMutable().push_back(make_bound_ref(1));
  coalesce_expr->GetChildrenMutable().push_back(make_bound_int_const(0));

  auto out = sirius::ast::from_duckdb(*coalesce_expr);
  REQUIRE(out);
  REQUIRE(out->holds<coalesce>());
  REQUIRE(out->get<coalesce>().children.size() == 3);
}

TEST_CASE("ast_from_duckdb - BOUND_OPERATOR COMPARE_IN translates to in_list(negated=false)",
          "[ast_from_duckdb]")
{
  auto in_expr = duckdb::make_uniq<BoundOperatorExpression>(ExpressionType::COMPARE_IN,
                                                            LogicalType{LogicalTypeId::BOOLEAN});
  in_expr->GetChildrenMutable().push_back(make_bound_ref(0));
  in_expr->GetChildrenMutable().push_back(make_bound_int_const(2));
  in_expr->GetChildrenMutable().push_back(make_bound_int_const(5));
  in_expr->GetChildrenMutable().push_back(make_bound_int_const(8));

  auto out = sirius::ast::from_duckdb(*in_expr);
  REQUIRE(out);
  REQUIRE(out->holds<in_list>());
  auto const& il = out->get<in_list>();
  REQUIRE(il.negated == false);
  REQUIRE(il.values.size() == 3);
  REQUIRE(il.probe);
  REQUIRE(il.probe->holds<reference>());
}

TEST_CASE("ast_from_duckdb - BOUND_OPERATOR COMPARE_NOT_IN translates to in_list(negated=true)",
          "[ast_from_duckdb]")
{
  auto in_expr = duckdb::make_uniq<BoundOperatorExpression>(ExpressionType::COMPARE_NOT_IN,
                                                            LogicalType{LogicalTypeId::BOOLEAN});
  in_expr->GetChildrenMutable().push_back(make_bound_ref(0));
  in_expr->GetChildrenMutable().push_back(make_bound_int_const(2));
  in_expr->GetChildrenMutable().push_back(make_bound_int_const(4));

  auto out = sirius::ast::from_duckdb(*in_expr);
  REQUIRE(out);
  REQUIRE(out->holds<in_list>());
  auto const& il = out->get<in_list>();
  REQUIRE(il.negated == true);
  REQUIRE(il.values.size() == 2);
}

TEST_CASE("ast_from_duckdb - BOUND_OPERATOR unsupported ExpressionType returns nullptr",
          "[ast_from_duckdb]")
{
  // OPERATOR_NULLIF is a well-formed BoundOperatorExpression kind that is not in
  // the demultiplex table; signal fallback via nullptr.
  auto nullif_expr = duckdb::make_uniq<BoundOperatorExpression>(
    ExpressionType::OPERATOR_NULLIF, LogicalType{LogicalTypeId::INTEGER});
  nullif_expr->GetChildrenMutable().push_back(make_bound_ref(0));
  nullif_expr->GetChildrenMutable().push_back(make_bound_int_const(0));

  REQUIRE(sirius::ast::from_duckdb(*nullif_expr) == nullptr);
}

TEST_CASE("ast_from_duckdb - BOUND_OPERATOR NOT with unsupported child propagates nullptr",
          "[ast_from_duckdb]")
{
  // BoundParameterExpression translates to nullptr (BOUND_PARAMETER class).
  // Wrapping it in NOT must propagate the nullptr up.
  auto bad_child = duckdb::make_uniq<BoundParameterExpression>(duckdb::Identifier{"p_not"});
  auto not_expr  = duckdb::make_uniq<BoundOperatorExpression>(ExpressionType::OPERATOR_NOT,
                                                             LogicalType{LogicalTypeId::BOOLEAN});
  not_expr->GetChildrenMutable().push_back(std::move(bad_child));

  REQUIRE(sirius::ast::from_duckdb(*not_expr) == nullptr);
}

TEST_CASE("ast_from_duckdb - BOUND_OPERATOR COMPARE_IN with unsupported probe propagates nullptr",
          "[ast_from_duckdb]")
{
  auto bad_probe = duckdb::make_uniq<BoundParameterExpression>(duckdb::Identifier{"p_in"});
  auto in_expr   = duckdb::make_uniq<BoundOperatorExpression>(ExpressionType::COMPARE_IN,
                                                            LogicalType{LogicalTypeId::BOOLEAN});
  in_expr->GetChildrenMutable().push_back(std::move(bad_probe));
  in_expr->GetChildrenMutable().push_back(make_bound_int_const(2));
  in_expr->GetChildrenMutable().push_back(make_bound_int_const(3));

  REQUIRE(sirius::ast::from_duckdb(*in_expr) == nullptr);
}

TEST_CASE("ast_from_duckdb - BOUND_OPERATOR COALESCE with unsupported child propagates nullptr",
          "[ast_from_duckdb]")
{
  auto bad_child = duckdb::make_uniq<BoundParameterExpression>(duckdb::Identifier{"p_coalesce"});
  auto coalesce_expr = duckdb::make_uniq<BoundOperatorExpression>(
    ExpressionType::OPERATOR_COALESCE, LogicalType{LogicalTypeId::INTEGER});
  coalesce_expr->GetChildrenMutable().push_back(make_bound_ref(0));
  coalesce_expr->GetChildrenMutable().push_back(std::move(bad_child));
  coalesce_expr->GetChildrenMutable().push_back(make_bound_int_const(7));

  REQUIRE(sirius::ast::from_duckdb(*coalesce_expr) == nullptr);
}

// ============================================================================
// BOUND_PARAMETER
// ============================================================================

TEST_CASE("ast_from_duckdb - BOUND_PARAMETER returns nullptr", "[ast_from_duckdb]")
{
  auto param_expr = duckdb::make_uniq<BoundParameterExpression>(duckdb::Identifier{"p1"});
  REQUIRE(sirius::ast::from_duckdb(*param_expr) == nullptr);
}

// ============================================================================
// Binder end-to-end — one case parses a SQL statement through DuckDB's real
// Binder and confirms each bound Expression translates to a non-null tree.
// Tagged separately so it can be skipped in fast iteration loops. The tag is
// deliberately kept off the shared GPU-init environment so this case does not
// trigger a multi-second SiriusContext setup it does not need.
// ============================================================================

// Why optimizer-off rather than populate-the-table-and-leave-optimizer-on:
// populating `t` with rows would let the optimizer run, which would fold the
// binder's BETWEEN-as-AND rewrite back into a single BoundBetween — but it
// would also pull in two unrelated rewrites that obscure what we're testing.
// FilterPushdown can move `a BETWEEN 1 AND 10` into LogicalGet::table_filters
// (a TableFilter tree, not BoundExpressions) and elide the LogicalFilter
// entirely, so the BETWEEN disappears from op.expressions. LikeOptimizationRule
// rewrites `b LIKE 'x%'` into a `prefix(b, 'x')` function call, so the LIKE
// landmark disappears too. Both rewrites are optimizer-version sensitive and
// would couple this test to optimizer pass behavior. We keep the optimizer
// off and assert against the binder's actual output — the rewrites the binder
// itself performs (documented below) are stable and part of the binder's
// public contract.
TEST_CASE("ast_from_duckdb - real Binder output translates to non-null trees",
          "[ast_from_duckdb][ast_from_duckdb_binder]")
{
  duckdb::DBConfig config;
  duckdb::DuckDB db(nullptr, &config);
  duckdb::Connection conn(db);
  conn.Query("CREATE TABLE t(a INTEGER, b VARCHAR, c BIGINT);");

  // The optimizer would constant-fold a query against an empty table down to
  // EMPTY_RESULT, stripping out every BoundExpression. Run the planner without
  // the optimizer so the test actually exercises from_duckdb on real Binder
  // output.
  duckdb::Settings::Set<duckdb::EnableOptimizerSetting>(
    *conn.context, duckdb::SetScope::SESSION, duckdb::Value::BOOLEAN(false));

  auto plan =
    conn.ExtractPlan("SELECT a + 3, b LIKE 'x%', c IS NOT NULL FROM t WHERE a BETWEEN 1 AND 10");
  REQUIRE(plan);

  // Recursive descent over the Sirius AST. We collect the presence of the
  // landmark kinds we expect from the four SQL fragments in the query,
  // rather than asserting an exact tree shape (the Binder may wrap children
  // in implicit casts or rearrange unrelated structure).
  //
  // BETWEEN landmark note: two binder/planner rewrites combine to strip
  // away both `between` and `conjunction` nodes for this query:
  //   1. The binder rewrites a BETWEEN whose input is non-volatile /
  //      non-parameter / non-subquery into a copy-input AND of two
  //      comparisons — see
  //      duckdb/src/planner/binder/expression/bind_between_expression.cpp
  //      lines 54-63. The optimizer would normally fold that pair back into
  //      a single BoundBetweenExpression, but this test disables the
  //      optimizer.
  //   2. LogicalFilter::SplitPredicates (filter construction, not the
  //      optimizer — see duckdb/src/planner/operator/logical_filter.cpp
  //      line 25) then flattens the top-level AND into two separate filter
  //      expressions on the LogicalFilter, so no BoundConjunction survives
  //      in the planned tree either.
  // Net effect: the filter for `a BETWEEN 1 AND 10` arrives here as two
  // sibling BoundComparisonExpressions (`a >= 1` and `a <= 10`). Coverage
  // for the actual translate_between arm lives in the direct-construction
  // case "BOUND_BETWEEN translates to between node with inclusive bounds"
  // earlier in this file; here we assert the rewritten form instead.
  bool saw_compare_ge  = false;
  bool saw_compare_le  = false;
  bool saw_add         = false;
  bool saw_like        = false;
  bool saw_is_not_null = false;
  // The query's expected node set after the binder + filter rewrites is
  // {comparison, function_call, unary_op, reference, constant}. The
  // between/conjunction arms are retained defensively so a future query
  // tweak that legitimately produces either kind (e.g. volatile BETWEEN
  // input, or an AND that survives because it's nested inside a
  // projection rather than a filter) still descends through them.
  std::function<void(node const&)> visit_node = [&](node const& n) {
    if (n.holds<between>()) {
      auto const& bw = n.get<between>();
      if (bw.input) visit_node(*bw.input);
      if (bw.lower) visit_node(*bw.lower);
      if (bw.upper) visit_node(*bw.upper);
    } else if (n.holds<conjunction>()) {
      // Defensive descent. See the BETWEEN landmark note above for why no
      // conjunction node is expected for this specific query — but if a
      // future tweak places an AND somewhere LogicalFilter::SplitPredicates
      // can't flatten (e.g. inside a projection), we still want to recurse
      // through it so leaf landmarks remain reachable.
      auto const& cj = n.get<conjunction>();
      for (auto const& c : cj.children) {
        if (c) visit_node(*c);
      }
    } else if (n.holds<comparison>()) {
      // The two comparisons produced by the BETWEEN rewrite land here as
      // sibling expressions on the LogicalFilter: `a >= 1` (ge) and
      // `a <= 10` (le). Asserting both ops appear is what pins down that
      // the rewritten form really came from the original BETWEEN rather
      // than from some other binder path.
      auto const& cmp = n.get<comparison>();
      if (cmp.op == sirius::comparison_type::ge) saw_compare_ge = true;
      if (cmp.op == sirius::comparison_type::le) saw_compare_le = true;
      if (cmp.left) visit_node(*cmp.left);
      if (cmp.right) visit_node(*cmp.right);
    } else if (n.holds<function_call>()) {
      auto const& fc = n.get<function_call>();
      if (fc.function() == sirius::function_id::add) saw_add = true;
      if (fc.function() == sirius::function_id::like) saw_like = true;
      for (auto const& a : fc.arguments()) {
        if (a) visit_node(*a);
      }
    } else if (n.holds<unary_op>()) {
      auto const& uo = n.get<unary_op>();
      if (uo.op == unary_op::kind::op_is_not_null) saw_is_not_null = true;
      if (uo.child) visit_node(*uo.child);
    }
    // reference and constant are leaves; nothing to descend into.
  };

  // DFS walk the LogicalOperator tree, collecting every Expression on every
  // operator. The structural assertion is: every Expression translates to a
  // non-null Sirius AST tree, AND the union of all produced trees covers the
  // landmark kinds named above.
  std::size_t expression_count = 0;
  std::function<void(duckdb::LogicalOperator const&)> walk =
    [&](duckdb::LogicalOperator const& op) {
      for (auto const& e : op.expressions) {
        REQUIRE(e);
        auto out = sirius::ast::from_duckdb(*e);
        REQUIRE(out);
        visit_node(*out);
        ++expression_count;
      }
      for (auto const& child : op.children) {
        walk(*child);
      }
    };
  walk(*plan);

  // 3 projection expressions (a+3, b LIKE 'x%', c IS NOT NULL) + 2 filter
  // expressions (the BETWEEN rewrite produces `a >= 1` and `a <= 10`, and
  // LogicalFilter::SplitPredicates flattens them into separate sibling
  // expressions — see the BETWEEN landmark note above). The optimizer is
  // disabled, so the binder output should not collapse any of these.
  REQUIRE(expression_count >= 5);
  REQUIRE(saw_compare_ge);
  REQUIRE(saw_compare_le);
  REQUIRE(saw_add);
  REQUIRE(saw_like);
  REQUIRE(saw_is_not_null);
}
