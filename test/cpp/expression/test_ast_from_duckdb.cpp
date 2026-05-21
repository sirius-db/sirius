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

#include "catch.hpp"
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

// ============================================================================
// BOUND_REFERENCE
// ============================================================================

TEST_CASE("ast_from_duckdb - BOUND_REF translates to reference node (column 0)",
          "[ast_from_duckdb]")
{
  auto expr = duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0);
  auto out  = sirius::ast::from_duckdb(*expr);
  REQUIRE(out);
  REQUIRE(out->holds<reference>());
  REQUIRE(out->get<reference>().column_index == 0);
}

TEST_CASE("ast_from_duckdb - BOUND_REF translates to reference node (column 5)",
          "[ast_from_duckdb]")
{
  auto expr = duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 5);
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
  auto expr = duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(42));
  auto out  = sirius::ast::from_duckdb(*expr);
  REQUIRE(out);
  REQUIRE(out->holds<constant>());
  auto const& c = out->get<constant>();
  REQUIRE(c.return_type.id() == sirius::type_id::INTEGER);
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
  REQUIRE(c.return_type.id() == sirius::type_id::VARCHAR);
  REQUIRE(std::holds_alternative<std::string>(c.payload));
  REQUIRE(std::get<std::string>(c.payload) == "hi");
}

TEST_CASE("ast_from_duckdb - BOUND_CONSTANT NULL of INTEGER preserves type",
          "[ast_from_duckdb]")
{
  auto expr =
    duckdb::make_uniq<BoundConstantExpression>(Value(LogicalType{LogicalTypeId::INTEGER}));
  auto out = sirius::ast::from_duckdb(*expr);
  REQUIRE(out);
  REQUIRE(out->holds<constant>());
  auto const& c = out->get<constant>();
  REQUIRE(c.return_type.id() == sirius::type_id::INTEGER);
  REQUIRE(std::holds_alternative<sirius::null_value>(c.payload));
}

// ============================================================================
// BOUND_COMPARISON — one TEST_CASE per supported comparison subtype + the
// distinct-from fallback.
// ============================================================================

TEST_CASE("ast_from_duckdb - BOUND_COMPARISON EQUAL translates to comparison node",
          "[ast_from_duckdb]")
{
  auto left  = duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0);
  auto right = duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(3));
  auto expr  = duckdb::make_uniq<BoundComparisonExpression>(
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
  auto left  = duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0);
  auto right = duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(3));
  auto expr  = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_NOTEQUAL, std::move(left), std::move(right));
  auto out = sirius::ast::from_duckdb(*expr);
  REQUIRE(out);
  REQUIRE(out->holds<comparison>());
  REQUIRE(out->get<comparison>().op == sirius::comparison_type::not_equal);
}

TEST_CASE("ast_from_duckdb - BOUND_COMPARISON LESSTHAN translates to comparison node",
          "[ast_from_duckdb]")
{
  auto left  = duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0);
  auto right = duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(3));
  auto expr  = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_LESSTHAN, std::move(left), std::move(right));
  auto out = sirius::ast::from_duckdb(*expr);
  REQUIRE(out);
  REQUIRE(out->holds<comparison>());
  REQUIRE(out->get<comparison>().op == sirius::comparison_type::lt);
}

TEST_CASE("ast_from_duckdb - BOUND_COMPARISON LESSTHANOREQUALTO translates to comparison node",
          "[ast_from_duckdb]")
{
  auto left  = duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0);
  auto right = duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(3));
  auto expr  = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_LESSTHANOREQUALTO, std::move(left), std::move(right));
  auto out = sirius::ast::from_duckdb(*expr);
  REQUIRE(out);
  REQUIRE(out->holds<comparison>());
  REQUIRE(out->get<comparison>().op == sirius::comparison_type::le);
}

TEST_CASE("ast_from_duckdb - BOUND_COMPARISON GREATERTHAN translates to comparison node",
          "[ast_from_duckdb]")
{
  auto left  = duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0);
  auto right = duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(3));
  auto expr  = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_GREATERTHAN, std::move(left), std::move(right));
  auto out = sirius::ast::from_duckdb(*expr);
  REQUIRE(out);
  REQUIRE(out->holds<comparison>());
  REQUIRE(out->get<comparison>().op == sirius::comparison_type::gt);
}

TEST_CASE("ast_from_duckdb - BOUND_COMPARISON GREATERTHANOREQUALTO translates to comparison node",
          "[ast_from_duckdb]")
{
  auto left  = duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0);
  auto right = duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(3));
  auto expr  = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_GREATERTHANOREQUALTO, std::move(left), std::move(right));
  auto out = sirius::ast::from_duckdb(*expr);
  REQUIRE(out);
  REQUIRE(out->holds<comparison>());
  REQUIRE(out->get<comparison>().op == sirius::comparison_type::ge);
}

TEST_CASE("ast_from_duckdb - BOUND_COMPARISON DISTINCT_FROM returns nullptr",
          "[ast_from_duckdb]")
{
  auto left  = duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0);
  auto right = duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(3));
  auto expr  = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_DISTINCT_FROM, std::move(left), std::move(right));
  REQUIRE(sirius::ast::from_duckdb(*expr) == nullptr);
}

// ============================================================================
// BOUND_CONJUNCTION
// ============================================================================

TEST_CASE("ast_from_duckdb - BOUND_CONJUNCTION AND translates to conjunction(op_and)",
          "[ast_from_duckdb]")
{
  auto and_expr = duckdb::make_uniq<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_AND);
  and_expr->children.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::BOOLEAN}, 0));
  and_expr->children.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::BOOLEAN}, 1));
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
  or_expr->children.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::BOOLEAN}, 0));
  or_expr->children.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::BOOLEAN}, 1));
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
  auto bt = duckdb::make_uniq<BoundBetweenExpression>(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0),
    duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(1)),
    duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(10)),
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
  auto check_expr = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_EQUAL,
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0),
    duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(1)));
  auto then_expr = duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(10));

  BoundCaseCheck case_check;
  case_check.when_expr = std::move(check_expr);
  case_check.then_expr = std::move(then_expr);

  auto else_expr  = duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(0));
  auto case_node  = duckdb::make_uniq<BoundCaseExpression>(LogicalType{LogicalTypeId::INTEGER});
  case_node->else_expr = std::move(else_expr);
  case_node->case_checks.push_back(std::move(case_check));

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
  // The WHEN subexpression is COMPARE_DISTINCT_FROM (unsupported -> nullptr).
  // The whole CASE must collapse to nullptr.
  auto bad_when = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_DISTINCT_FROM,
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0),
    duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(1)));
  auto then_expr = duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(10));

  BoundCaseCheck case_check;
  case_check.when_expr = std::move(bad_when);
  case_check.then_expr = std::move(then_expr);

  auto else_expr  = duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(0));
  auto case_node  = duckdb::make_uniq<BoundCaseExpression>(LogicalType{LogicalTypeId::INTEGER});
  case_node->else_expr = std::move(else_expr);
  case_node->case_checks.push_back(std::move(case_check));

  REQUIRE(sirius::ast::from_duckdb(*case_node) == nullptr);
}

// ============================================================================
// BOUND_CAST
// ============================================================================

TEST_CASE("ast_from_duckdb - BOUND_CAST INTEGER -> BIGINT translates to cast node",
          "[ast_from_duckdb]")
{
  auto child = duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0);
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

TEST_CASE("ast_from_duckdb - BOUND_CAST honors try_cast = true",
          "[ast_from_duckdb]")
{
  auto child = duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0);
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

TEST_CASE("ast_from_duckdb - BOUND_FUNCTION '+' resolves to function_id::add",
          "[ast_from_duckdb]")
{
  auto add_expr = duckdb::make_uniq<BoundFunctionExpression>(
    LogicalType{LogicalTypeId::INTEGER},
    ScalarFunction(
      "+", {LogicalType::INTEGER, LogicalType::INTEGER}, LogicalType::INTEGER, nullptr),
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>{},
    nullptr);
  add_expr->children.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0));
  add_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(3)));

  auto out = sirius::ast::from_duckdb(*add_expr);
  REQUIRE(out);
  REQUIRE(out->holds<function_call>());
  auto const& fc = out->get<function_call>();
  REQUIRE(fc.function == sirius::function_id::add);
  REQUIRE(fc.arguments.size() == 2);
  REQUIRE(fc.return_type.id() == sirius::type_id::INTEGER);
}

TEST_CASE("ast_from_duckdb - BOUND_FUNCTION 'substring' resolves to function_id::substring",
          "[ast_from_duckdb]")
{
  auto fn_expr = duckdb::make_uniq<BoundFunctionExpression>(
    LogicalType{LogicalTypeId::VARCHAR},
    ScalarFunction("substring",
                   {LogicalType::VARCHAR, LogicalType::INTEGER, LogicalType::INTEGER},
                   LogicalType::VARCHAR,
                   nullptr),
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>{},
    nullptr);
  fn_expr->children.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::VARCHAR}, 0));
  fn_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(1)));
  fn_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(3)));

  auto out = sirius::ast::from_duckdb(*fn_expr);
  REQUIRE(out);
  REQUIRE(out->holds<function_call>());
  REQUIRE(out->get<function_call>().function == sirius::function_id::substring);
}

TEST_CASE("ast_from_duckdb - BOUND_FUNCTION 'substr' alias resolves to function_id::substring",
          "[ast_from_duckdb]")
{
  auto fn_expr = duckdb::make_uniq<BoundFunctionExpression>(
    LogicalType{LogicalTypeId::VARCHAR},
    ScalarFunction("substr",
                   {LogicalType::VARCHAR, LogicalType::INTEGER, LogicalType::INTEGER},
                   LogicalType::VARCHAR,
                   nullptr),
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>{},
    nullptr);
  fn_expr->children.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::VARCHAR}, 0));
  fn_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(1)));
  fn_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(3)));

  auto out = sirius::ast::from_duckdb(*fn_expr);
  REQUIRE(out);
  REQUIRE(out->holds<function_call>());
  REQUIRE(out->get<function_call>().function == sirius::function_id::substring);
}

TEST_CASE("ast_from_duckdb - BOUND_FUNCTION unknown name returns nullptr",
          "[ast_from_duckdb]")
{
  auto fn_expr = duckdb::make_uniq<BoundFunctionExpression>(
    LogicalType{LogicalTypeId::INTEGER},
    ScalarFunction(
      "nonexistent_fn", {LogicalType::INTEGER}, LogicalType::INTEGER, nullptr),
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>{},
    nullptr);
  fn_expr->children.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0));

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
  auto not_expr = duckdb::make_uniq<BoundOperatorExpression>(
    ExpressionType::OPERATOR_NOT, LogicalType{LogicalTypeId::BOOLEAN});
  not_expr->children.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::BOOLEAN}, 0));
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
  is_null_expr->children.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0));
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
  is_not_null_expr->children.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0));
  auto out = sirius::ast::from_duckdb(*is_not_null_expr);
  REQUIRE(out);
  REQUIRE(out->holds<unary_op>());
  REQUIRE(out->get<unary_op>().op == unary_op::kind::op_is_not_null);
}

TEST_CASE("ast_from_duckdb - BOUND_OPERATOR TRY translates to unary_op(op_try)",
          "[ast_from_duckdb]")
{
  auto try_expr = duckdb::make_uniq<BoundOperatorExpression>(
    ExpressionType::OPERATOR_TRY, LogicalType{LogicalTypeId::INTEGER});
  try_expr->children.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0));
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
  coalesce_expr->children.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0));
  coalesce_expr->children.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 1));
  coalesce_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(0)));

  auto out = sirius::ast::from_duckdb(*coalesce_expr);
  REQUIRE(out);
  REQUIRE(out->holds<coalesce>());
  REQUIRE(out->get<coalesce>().children.size() == 3);
}

TEST_CASE("ast_from_duckdb - BOUND_OPERATOR COMPARE_IN translates to in_list(negated=false)",
          "[ast_from_duckdb]")
{
  auto in_expr = duckdb::make_uniq<BoundOperatorExpression>(
    ExpressionType::COMPARE_IN, LogicalType{LogicalTypeId::BOOLEAN});
  in_expr->children.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0));
  in_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(2)));
  in_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(5)));
  in_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(8)));

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
  auto in_expr = duckdb::make_uniq<BoundOperatorExpression>(
    ExpressionType::COMPARE_NOT_IN, LogicalType{LogicalTypeId::BOOLEAN});
  in_expr->children.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0));
  in_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(2)));
  in_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(4)));

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
  nullif_expr->children.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0));
  nullif_expr->children.push_back(
    duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(0)));

  REQUIRE(sirius::ast::from_duckdb(*nullif_expr) == nullptr);
}

TEST_CASE("ast_from_duckdb - BOUND_OPERATOR NOT with unsupported child propagates nullptr",
          "[ast_from_duckdb]")
{
  // COMPARE_DISTINCT_FROM is the canonical unsupported-comparison subtype that
  // routes to nullptr; wrapping it in NOT must propagate the nullptr up.
  auto bad_child = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_DISTINCT_FROM,
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0),
    duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(1)));
  auto not_expr = duckdb::make_uniq<BoundOperatorExpression>(
    ExpressionType::OPERATOR_NOT, LogicalType{LogicalTypeId::BOOLEAN});
  not_expr->children.push_back(std::move(bad_child));

  REQUIRE(sirius::ast::from_duckdb(*not_expr) == nullptr);
}

TEST_CASE("ast_from_duckdb - BOUND_OPERATOR COMPARE_IN with unsupported probe propagates nullptr",
          "[ast_from_duckdb]")
{
  auto bad_probe = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_DISTINCT_FROM,
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0),
    duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(1)));
  auto in_expr = duckdb::make_uniq<BoundOperatorExpression>(
    ExpressionType::COMPARE_IN, LogicalType{LogicalTypeId::BOOLEAN});
  in_expr->children.push_back(std::move(bad_probe));
  in_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(2)));
  in_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(3)));

  REQUIRE(sirius::ast::from_duckdb(*in_expr) == nullptr);
}

TEST_CASE("ast_from_duckdb - BOUND_OPERATOR COALESCE with unsupported child propagates nullptr",
          "[ast_from_duckdb]")
{
  auto bad_child = duckdb::make_uniq<BoundComparisonExpression>(
    ExpressionType::COMPARE_DISTINCT_FROM,
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0),
    duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(1)));
  auto coalesce_expr = duckdb::make_uniq<BoundOperatorExpression>(
    ExpressionType::OPERATOR_COALESCE, LogicalType{LogicalTypeId::INTEGER});
  coalesce_expr->children.push_back(
    duckdb::make_uniq<BoundReferenceExpression>(LogicalType{LogicalTypeId::INTEGER}, 0));
  coalesce_expr->children.push_back(std::move(bad_child));
  coalesce_expr->children.push_back(duckdb::make_uniq<BoundConstantExpression>(Value::INTEGER(7)));

  REQUIRE(sirius::ast::from_duckdb(*coalesce_expr) == nullptr);
}

// ============================================================================
// BOUND_PARAMETER
// ============================================================================

TEST_CASE("ast_from_duckdb - BOUND_PARAMETER returns nullptr",
          "[ast_from_duckdb]")
{
  auto param_expr = duckdb::make_uniq<BoundParameterExpression>(std::string{"p1"});
  REQUIRE(sirius::ast::from_duckdb(*param_expr) == nullptr);
}

// ============================================================================
// Binder end-to-end — one case parses a SQL statement through DuckDB's real
// Binder and confirms each bound Expression translates to a non-null tree.
// Tagged separately so it can be skipped in fast iteration loops. The tag is
// deliberately kept off the shared GPU-init environment so this case does not
// trigger a multi-second SiriusContext setup it does not need.
// ============================================================================

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
  duckdb::ClientConfig::GetConfig(*conn.context).enable_optimizer = false;

  auto plan = conn.ExtractPlan(
    "SELECT a + 3, b LIKE 'x%', c IS NOT NULL FROM t WHERE a BETWEEN 1 AND 10");
  REQUIRE(plan);

  // DFS walk the LogicalOperator tree, collecting every Expression on every
  // operator. The structural assertion is: every Expression translates to a
  // non-null Sirius AST tree. We do NOT assert the exact tree shape because
  // the Binder may insert implicit casts or simplifications.
  std::size_t expression_count = 0;
  std::function<void(duckdb::LogicalOperator const&)> walk =
    [&](duckdb::LogicalOperator const& op) {
      for (auto const& e : op.expressions) {
        REQUIRE(e);
        auto out = sirius::ast::from_duckdb(*e);
        REQUIRE(out);
        ++expression_count;
      }
      for (auto const& child : op.children) {
        walk(*child);
      }
    };
  walk(*plan);

  // The query carries multiple projection + filter expressions; ensure the
  // traversal actually visited something.
  REQUIRE(expression_count >= 1);
}
