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

// Tests for sirius::ast::to_duckdb — the inverse-direction Sirius->DuckDB
// expression translator.
//
// Covers the full sirius::ast::node dispatch surface via direct construction
// of each variant alternative, mapped to its matching duckdb::BoundXxx
// expression. Each section also validates the structural shape of the
// reconstructed DuckDB tree (expression class, operator type, child counts,
// payload fidelity). A self-equivalence sweep then runs every alternative
// through from_duckdb(*to_duckdb(node)) and confirms structural equivalence
// with the original Sirius AST.

#include "ast_test_builders.hpp"
#include "catch.hpp"
#include "expression/ast/from_duckdb.hpp"
#include "expression/ast/to_duckdb.hpp"

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
#include "expression/value.hpp"           // sirius::value, sirius::null_value
#include "helper/logical_type.hpp"        // sirius::logical_type

// duckdb — direct-ctor + downstream introspection surface
#include <duckdb/common/enums/expression_type.hpp>
#include <duckdb/common/types/value.hpp>
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
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

using duckdb::BoundBetweenExpression;
using duckdb::BoundCaseExpression;
using duckdb::BoundCastExpression;
using duckdb::BoundComparisonExpression;
using duckdb::BoundConjunctionExpression;
using duckdb::BoundConstantExpression;
using duckdb::BoundFunctionExpression;
using duckdb::BoundOperatorExpression;
using duckdb::BoundReferenceExpression;
using duckdb::ExpressionClass;
using duckdb::ExpressionType;
using duckdb::LogicalType;
using duckdb::LogicalTypeId;
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

namespace {

using sirius::ast::test::make_int_const;
using sirius::ast::test::make_ref;
using sirius::ast::test::make_str_const;

// ----------------------------------------------------------------------------
// Structural-equality helpers — used by the self-equivalence sweep below to
// compare the original node tree against from_duckdb(*to_duckdb(orig)).
// ----------------------------------------------------------------------------

bool same_node(node const& a, node const& b);  // forward declaration

bool same_reference(node const& a, node const& b)
{
  return a.holds<reference>() && b.holds<reference>() &&
         a.get<reference>().column_index == b.get<reference>().column_index;
}

bool same_constant(node const& a, node const& b)
{
  if (!a.holds<constant>() || !b.holds<constant>()) return false;
  auto const& ca = a.get<constant>();
  auto const& cb = b.get<constant>();
  if (ca.return_type().id() != cb.return_type().id()) return false;
  // sirius::value alternatives do not all expose operator==; compare via the
  // variant index plus a targeted check for the primitive alternatives the
  // self-equivalence sweep exercises (int32_t covers the BoundConstantExpression
  // round-trip used here; other alternatives are compared structurally only).
  if (ca.payload.index() != cb.payload.index()) return false;
  if (std::holds_alternative<int32_t>(ca.payload)) {
    return std::get<int32_t>(ca.payload) == std::get<int32_t>(cb.payload);
  }
  if (std::holds_alternative<std::string>(ca.payload)) {
    return std::get<std::string>(ca.payload) == std::get<std::string>(cb.payload);
  }
  if (std::holds_alternative<bool>(ca.payload)) {
    return std::get<bool>(ca.payload) == std::get<bool>(cb.payload);
  }
  if (std::holds_alternative<int64_t>(ca.payload)) {
    return std::get<int64_t>(ca.payload) == std::get<int64_t>(cb.payload);
  }
  if (std::holds_alternative<double>(ca.payload)) {
    return std::get<double>(ca.payload) == std::get<double>(cb.payload);
  }
  // Other alternatives (null_value, date/timestamp, decimal*) — variant index
  // match is sufficient evidence of structural equivalence for the
  // self-equivalence sweep; payload fidelity is already covered by Phase 2's
  // value round-trip tests.
  return true;
}

bool same_comparison(node const& a, node const& b)
{
  if (!a.holds<comparison>() || !b.holds<comparison>()) return false;
  auto const& ca = a.get<comparison>();
  auto const& cb = b.get<comparison>();
  if (ca.op != cb.op) return false;
  if (!ca.left || !cb.left || !ca.right || !cb.right) return false;
  return same_node(*ca.left, *cb.left) && same_node(*ca.right, *cb.right);
}

bool same_conjunction(node const& a, node const& b)
{
  if (!a.holds<conjunction>() || !b.holds<conjunction>()) return false;
  auto const& ca = a.get<conjunction>();
  auto const& cb = b.get<conjunction>();
  if (ca.op != cb.op) return false;
  if (ca.children.size() != cb.children.size()) return false;
  for (size_t i = 0; i < ca.children.size(); ++i) {
    if (!ca.children[i] || !cb.children[i]) return false;
    if (!same_node(*ca.children[i], *cb.children[i])) return false;
  }
  return true;
}

bool same_between(node const& a, node const& b)
{
  if (!a.holds<between>() || !b.holds<between>()) return false;
  auto const& ba = a.get<between>();
  auto const& bb = b.get<between>();
  if (ba.lower_inclusive != bb.lower_inclusive) return false;
  if (ba.upper_inclusive != bb.upper_inclusive) return false;
  if (!ba.input || !bb.input || !ba.lower || !bb.lower || !ba.upper || !bb.upper) return false;
  return same_node(*ba.input, *bb.input) && same_node(*ba.lower, *bb.lower) &&
         same_node(*ba.upper, *bb.upper);
}

bool same_case_expr(node const& a, node const& b)
{
  if (!a.holds<case_expr>() || !b.holds<case_expr>()) return false;
  auto const& ca = a.get<case_expr>();
  auto const& cb = b.get<case_expr>();
  if (ca.cases.size() != cb.cases.size()) return false;
  for (size_t i = 0; i < ca.cases.size(); ++i) {
    if (!ca.cases[i].when_ || !cb.cases[i].when_) return false;
    if (!ca.cases[i].then_ || !cb.cases[i].then_) return false;
    if (!same_node(*ca.cases[i].when_, *cb.cases[i].when_)) return false;
    if (!same_node(*ca.cases[i].then_, *cb.cases[i].then_)) return false;
  }
  if (!ca.else_ || !cb.else_) return false;
  return same_node(*ca.else_, *cb.else_);
}

bool same_cast(node const& a, node const& b)
{
  if (!a.holds<cast>() || !b.holds<cast>()) return false;
  auto const& ca = a.get<cast>();
  auto const& cb = b.get<cast>();
  if (ca.try_cast != cb.try_cast) return false;
  if (ca.target_type.id() != cb.target_type.id()) return false;
  if (!ca.child || !cb.child) return false;
  return same_node(*ca.child, *cb.child);
}

bool same_unary_op(node const& a, node const& b)
{
  if (!a.holds<unary_op>() || !b.holds<unary_op>()) return false;
  auto const& ua = a.get<unary_op>();
  auto const& ub = b.get<unary_op>();
  if (ua.op != ub.op) return false;
  if (!ua.child || !ub.child) return false;
  return same_node(*ua.child, *ub.child);
}

bool same_coalesce(node const& a, node const& b)
{
  if (!a.holds<coalesce>() || !b.holds<coalesce>()) return false;
  auto const& ca = a.get<coalesce>();
  auto const& cb = b.get<coalesce>();
  if (ca.children.size() != cb.children.size()) return false;
  for (size_t i = 0; i < ca.children.size(); ++i) {
    if (!ca.children[i] || !cb.children[i]) return false;
    if (!same_node(*ca.children[i], *cb.children[i])) return false;
  }
  return true;
}

bool same_in_list(node const& a, node const& b)
{
  if (!a.holds<in_list>() || !b.holds<in_list>()) return false;
  auto const& ia = a.get<in_list>();
  auto const& ib = b.get<in_list>();
  if (ia.negated != ib.negated) return false;
  if (ia.values.size() != ib.values.size()) return false;
  if (!ia.probe || !ib.probe) return false;
  if (!same_node(*ia.probe, *ib.probe)) return false;
  for (size_t i = 0; i < ia.values.size(); ++i) {
    if (!ia.values[i] || !ib.values[i]) return false;
    if (!same_node(*ia.values[i], *ib.values[i])) return false;
  }
  return true;
}

bool same_function_call(node const& a, node const& b)
{
  if (!a.holds<function_call>() || !b.holds<function_call>()) return false;
  auto const& fa = a.get<function_call>();
  auto const& fb = b.get<function_call>();
  if (fa.function() != fb.function()) return false;
  if (fa.return_type().id() != fb.return_type().id()) return false;
  if (fa.arguments().size() != fb.arguments().size()) return false;
  for (size_t i = 0; i < fa.arguments().size(); ++i) {
    if (!fa.arguments()[i] || !fb.arguments()[i]) return false;
    if (!same_node(*fa.arguments()[i], *fb.arguments()[i])) return false;
  }
  return true;
}

bool same_node(node const& a, node const& b)
{
  if (a.v.index() != b.v.index()) return false;
  if (a.holds<reference>()) return same_reference(a, b);
  if (a.holds<constant>()) return same_constant(a, b);
  if (a.holds<comparison>()) return same_comparison(a, b);
  if (a.holds<conjunction>()) return same_conjunction(a, b);
  if (a.holds<between>()) return same_between(a, b);
  if (a.holds<case_expr>()) return same_case_expr(a, b);
  if (a.holds<cast>()) return same_cast(a, b);
  if (a.holds<unary_op>()) return same_unary_op(a, b);
  if (a.holds<coalesce>()) return same_coalesce(a, b);
  if (a.holds<in_list>()) return same_in_list(a, b);
  if (a.holds<function_call>()) return same_function_call(a, b);
  return false;
}

}  // namespace

// ============================================================================
// reference
// ============================================================================

TEST_CASE("ast_to_duckdb - reference translates to BoundReferenceExpression (column 3)",
          "[ast_to_duckdb]")
{
  node orig{reference{/*column_index=*/3}};
  auto out = sirius::ast::to_duckdb(orig);
  REQUIRE(out);
  REQUIRE(out->GetExpressionClass() == ExpressionClass::BOUND_REF);
  auto const& ref = out->Cast<BoundReferenceExpression>();
  REQUIRE(ref.index == 3);
}

// ============================================================================
// constant
// ============================================================================

TEST_CASE("ast_to_duckdb - constant INTEGER round-trips to BoundConstantExpression",
          "[ast_to_duckdb]")
{
  node orig{constant{
    /*payload=*/sirius::value{int32_t{42}},
    /*return_type=*/sirius::logical_type::make(sirius::type_id::INTEGER),
  }};
  auto out = sirius::ast::to_duckdb(orig);
  REQUIRE(out);
  REQUIRE(out->GetExpressionClass() == ExpressionClass::BOUND_CONSTANT);
  auto const& bc = out->Cast<BoundConstantExpression>();
  REQUIRE(bc.value == Value::INTEGER(42));
}

TEST_CASE("ast_to_duckdb - constant VARCHAR round-trips to BoundConstantExpression",
          "[ast_to_duckdb]")
{
  node orig{constant{
    /*payload=*/sirius::value{std::string{"hello"}},
    /*return_type=*/sirius::logical_type::make(sirius::type_id::VARCHAR),
  }};
  auto out = sirius::ast::to_duckdb(orig);
  REQUIRE(out);
  REQUIRE(out->GetExpressionClass() == ExpressionClass::BOUND_CONSTANT);
  auto const& bc = out->Cast<BoundConstantExpression>();
  REQUIRE(bc.value == Value("hello"));
}

TEST_CASE("ast_to_duckdb - constant typed NULL (INTEGER) round-trips to BoundConstantExpression",
          "[ast_to_duckdb]")
{
  node orig{constant{
    /*payload=*/sirius::value{},  // null_value default
    /*return_type=*/sirius::logical_type::make(sirius::type_id::INTEGER),
  }};
  auto out = sirius::ast::to_duckdb(orig);
  REQUIRE(out);
  REQUIRE(out->GetExpressionClass() == ExpressionClass::BOUND_CONSTANT);
  auto const& bc = out->Cast<BoundConstantExpression>();
  REQUIRE(bc.value.IsNull());
  REQUIRE(bc.value.type().id() == LogicalTypeId::INTEGER);
}

TEST_CASE("ast_to_duckdb - constant DECIMAL64 round-trips to BoundConstantExpression",
          "[ast_to_duckdb]")
{
  node orig{constant{
    /*payload=*/sirius::value{sirius::decimal64{/*value=*/12345, /*scale=*/2}},
    /*return_type=*/sirius::logical_type::make_decimal(/*precision=*/18, /*scale=*/2),
  }};
  auto out = sirius::ast::to_duckdb(orig);
  REQUIRE(out);
  REQUIRE(out->GetExpressionClass() == ExpressionClass::BOUND_CONSTANT);
  auto const& bc = out->Cast<BoundConstantExpression>();
  REQUIRE(bc.value.type().id() == LogicalTypeId::DECIMAL);
}

// ============================================================================
// comparison — one TEST_CASE per supported subtype.
// ============================================================================

TEST_CASE("ast_to_duckdb - comparison EQUAL translates to COMPARE_EQUAL", "[ast_to_duckdb]")
{
  node orig{comparison{sirius::comparison_type::equal, make_ref(0), make_int_const(3)}};
  auto out = sirius::ast::to_duckdb(orig);
  REQUIRE(out);
  REQUIRE(out->GetExpressionType() == ExpressionType::COMPARE_EQUAL);
  auto const& cmp = out->Cast<BoundComparisonExpression>();
  REQUIRE(cmp.left);
  REQUIRE(cmp.right);
  REQUIRE(cmp.left->GetExpressionClass() == ExpressionClass::BOUND_REF);
  REQUIRE(cmp.right->GetExpressionClass() == ExpressionClass::BOUND_CONSTANT);
}

TEST_CASE("ast_to_duckdb - comparison NOT_EQUAL translates to COMPARE_NOTEQUAL", "[ast_to_duckdb]")
{
  node orig{comparison{sirius::comparison_type::not_equal, make_ref(0), make_int_const(3)}};
  auto out = sirius::ast::to_duckdb(orig);
  REQUIRE(out);
  REQUIRE(out->GetExpressionType() == ExpressionType::COMPARE_NOTEQUAL);
}

TEST_CASE("ast_to_duckdb - comparison LT translates to COMPARE_LESSTHAN", "[ast_to_duckdb]")
{
  node orig{comparison{sirius::comparison_type::lt, make_ref(0), make_int_const(3)}};
  auto out = sirius::ast::to_duckdb(orig);
  REQUIRE(out);
  REQUIRE(out->GetExpressionType() == ExpressionType::COMPARE_LESSTHAN);
}

TEST_CASE("ast_to_duckdb - comparison LE translates to COMPARE_LESSTHANOREQUALTO",
          "[ast_to_duckdb]")
{
  node orig{comparison{sirius::comparison_type::le, make_ref(0), make_int_const(3)}};
  auto out = sirius::ast::to_duckdb(orig);
  REQUIRE(out);
  REQUIRE(out->GetExpressionType() == ExpressionType::COMPARE_LESSTHANOREQUALTO);
}

TEST_CASE("ast_to_duckdb - comparison GT translates to COMPARE_GREATERTHAN", "[ast_to_duckdb]")
{
  node orig{comparison{sirius::comparison_type::gt, make_ref(0), make_int_const(3)}};
  auto out = sirius::ast::to_duckdb(orig);
  REQUIRE(out);
  REQUIRE(out->GetExpressionType() == ExpressionType::COMPARE_GREATERTHAN);
}

TEST_CASE("ast_to_duckdb - comparison GE translates to COMPARE_GREATERTHANOREQUALTO",
          "[ast_to_duckdb]")
{
  node orig{comparison{sirius::comparison_type::ge, make_ref(0), make_int_const(3)}};
  auto out = sirius::ast::to_duckdb(orig);
  REQUIRE(out);
  REQUIRE(out->GetExpressionType() == ExpressionType::COMPARE_GREATERTHANOREQUALTO);
}

// ============================================================================
// conjunction
// ============================================================================

TEST_CASE("ast_to_duckdb - conjunction AND translates to BoundConjunctionExpression (AND)",
          "[ast_to_duckdb]")
{
  std::vector<std::unique_ptr<node>> children;
  children.push_back(make_ref(0));
  children.push_back(make_ref(1));
  node orig{conjunction{conjunction::kind::op_and, std::move(children)}};
  auto out = sirius::ast::to_duckdb(orig);
  REQUIRE(out);
  REQUIRE(out->GetExpressionType() == ExpressionType::CONJUNCTION_AND);
  REQUIRE(out->GetExpressionClass() == ExpressionClass::BOUND_CONJUNCTION);
  auto const& conj = out->Cast<BoundConjunctionExpression>();
  REQUIRE(conj.children.size() == 2);
}

TEST_CASE("ast_to_duckdb - conjunction OR (3 children) translates to BoundConjunctionExpression",
          "[ast_to_duckdb]")
{
  std::vector<std::unique_ptr<node>> children;
  children.push_back(make_ref(0));
  children.push_back(make_ref(1));
  children.push_back(make_ref(2));
  node orig{conjunction{conjunction::kind::op_or, std::move(children)}};
  auto out = sirius::ast::to_duckdb(orig);
  REQUIRE(out);
  REQUIRE(out->GetExpressionType() == ExpressionType::CONJUNCTION_OR);
  auto const& conj = out->Cast<BoundConjunctionExpression>();
  REQUIRE(conj.children.size() == 3);
}

// ============================================================================
// between
// ============================================================================

TEST_CASE("ast_to_duckdb - between translates to BoundBetweenExpression (inclusive bounds)",
          "[ast_to_duckdb]")
{
  node orig{between{
    /*input=*/make_ref(0),
    /*lower=*/make_int_const(1),
    /*upper=*/make_int_const(10),
    /*lower_inclusive=*/true,
    /*upper_inclusive=*/true,
  }};
  auto out = sirius::ast::to_duckdb(orig);
  REQUIRE(out);
  REQUIRE(out->GetExpressionClass() == ExpressionClass::BOUND_BETWEEN);
  auto const& bt = out->Cast<BoundBetweenExpression>();
  REQUIRE(bt.lower_inclusive == true);
  REQUIRE(bt.upper_inclusive == true);
  REQUIRE(bt.input);
  REQUIRE(bt.lower);
  REQUIRE(bt.upper);
}

// ============================================================================
// case_expr
// ============================================================================

TEST_CASE("ast_to_duckdb - case_expr (single WHEN/THEN + ELSE) translates to BoundCaseExpression",
          "[ast_to_duckdb]")
{
  std::vector<case_expr::when_then> cases;
  cases.push_back(case_expr::when_then{
    /*when_=*/std::make_unique<node>(
      comparison{sirius::comparison_type::equal, make_ref(0), make_int_const(1)}),
    /*then_=*/make_int_const(10),
  });
  node orig{case_expr{std::move(cases),
                      /*else_=*/make_int_const(0),
                      sirius::logical_type::make(sirius::type_id::INTEGER)}};
  auto out = sirius::ast::to_duckdb(orig);
  REQUIRE(out);
  REQUIRE(out->GetExpressionClass() == ExpressionClass::BOUND_CASE);
  auto const& ce = out->Cast<BoundCaseExpression>();
  REQUIRE(ce.return_type.id() == LogicalTypeId::INTEGER);
  REQUIRE(ce.case_checks.size() == 1);
  REQUIRE(ce.case_checks[0].when_expr);
  REQUIRE(ce.case_checks[0].then_expr);
  REQUIRE(ce.else_expr);
}

TEST_CASE("ast_to_duckdb - case_expr (two WHEN/THEN + ELSE) translates to BoundCaseExpression",
          "[ast_to_duckdb]")
{
  std::vector<case_expr::when_then> cases;
  cases.push_back(case_expr::when_then{
    std::make_unique<node>(
      comparison{sirius::comparison_type::equal, make_ref(0), make_int_const(1)}),
    make_int_const(10),
  });
  cases.push_back(case_expr::when_then{
    std::make_unique<node>(
      comparison{sirius::comparison_type::equal, make_ref(0), make_int_const(2)}),
    make_int_const(20),
  });
  node orig{case_expr{
    std::move(cases), make_int_const(0), sirius::logical_type::make(sirius::type_id::INTEGER)}};
  auto out = sirius::ast::to_duckdb(orig);
  REQUIRE(out);
  REQUIRE(out->GetExpressionClass() == ExpressionClass::BOUND_CASE);
  auto const& ce = out->Cast<BoundCaseExpression>();
  REQUIRE(ce.case_checks.size() == 2);
}

// ============================================================================
// cast
// ============================================================================

TEST_CASE("ast_to_duckdb - cast INTEGER->BIGINT (default) translates to BoundCastExpression",
          "[ast_to_duckdb]")
{
  node orig{cast{
    /*child=*/make_ref(0),
    /*target_type=*/sirius::logical_type::make(sirius::type_id::BIGINT),
    /*try_cast=*/false,
  }};
  auto out = sirius::ast::to_duckdb(orig);
  REQUIRE(out);
  REQUIRE(out->GetExpressionClass() == ExpressionClass::BOUND_CAST);
  auto const& ct = out->Cast<BoundCastExpression>();
  REQUIRE(ct.try_cast == false);
  REQUIRE(ct.return_type.id() == LogicalTypeId::BIGINT);
}

TEST_CASE("ast_to_duckdb - cast INTEGER->BIGINT (try_cast=true) translates to BoundCastExpression",
          "[ast_to_duckdb]")
{
  node orig{cast{
    /*child=*/make_ref(0),
    /*target_type=*/sirius::logical_type::make(sirius::type_id::BIGINT),
    /*try_cast=*/true,
  }};
  auto out = sirius::ast::to_duckdb(orig);
  REQUIRE(out);
  REQUIRE(out->GetExpressionClass() == ExpressionClass::BOUND_CAST);
  auto const& ct = out->Cast<BoundCastExpression>();
  REQUIRE(ct.try_cast == true);
  REQUIRE(ct.return_type.id() == LogicalTypeId::BIGINT);
}

// ============================================================================
// unary_op — one TEST_CASE per kind.
// ============================================================================

TEST_CASE("ast_to_duckdb - unary_op op_not translates to OPERATOR_NOT", "[ast_to_duckdb]")
{
  node orig{unary_op{unary_op::kind::op_not, make_ref(0)}};
  auto out = sirius::ast::to_duckdb(orig);
  REQUIRE(out);
  REQUIRE(out->GetExpressionType() == ExpressionType::OPERATOR_NOT);
  REQUIRE(out->Cast<BoundOperatorExpression>().children.size() == 1);
}

TEST_CASE("ast_to_duckdb - unary_op op_is_null translates to OPERATOR_IS_NULL", "[ast_to_duckdb]")
{
  node orig{unary_op{unary_op::kind::op_is_null, make_ref(0)}};
  auto out = sirius::ast::to_duckdb(orig);
  REQUIRE(out);
  REQUIRE(out->GetExpressionType() == ExpressionType::OPERATOR_IS_NULL);
}

TEST_CASE("ast_to_duckdb - unary_op op_is_not_null translates to OPERATOR_IS_NOT_NULL",
          "[ast_to_duckdb]")
{
  node orig{unary_op{unary_op::kind::op_is_not_null, make_ref(0)}};
  auto out = sirius::ast::to_duckdb(orig);
  REQUIRE(out);
  REQUIRE(out->GetExpressionType() == ExpressionType::OPERATOR_IS_NOT_NULL);
}

TEST_CASE("ast_to_duckdb - unary_op op_try translates to OPERATOR_TRY", "[ast_to_duckdb]")
{
  node orig{unary_op{unary_op::kind::op_try, make_ref(0)}};
  auto out = sirius::ast::to_duckdb(orig);
  REQUIRE(out);
  REQUIRE(out->GetExpressionType() == ExpressionType::OPERATOR_TRY);
}

// ============================================================================
// coalesce
// ============================================================================

TEST_CASE("ast_to_duckdb - coalesce (3 args) translates to OPERATOR_COALESCE", "[ast_to_duckdb]")
{
  std::vector<std::unique_ptr<node>> children;
  children.push_back(make_ref(0));
  children.push_back(make_ref(1));
  children.push_back(make_int_const(0));
  node orig{coalesce{std::move(children), sirius::logical_type::make(sirius::type_id::INTEGER)}};
  auto out = sirius::ast::to_duckdb(orig);
  REQUIRE(out);
  REQUIRE(out->GetExpressionType() == ExpressionType::OPERATOR_COALESCE);
  REQUIRE(out->Cast<BoundOperatorExpression>().return_type.id() == LogicalTypeId::INTEGER);
  REQUIRE(out->Cast<BoundOperatorExpression>().children.size() == 3);
}

// ============================================================================
// in_list
// ============================================================================

TEST_CASE("ast_to_duckdb - in_list (negated=false) translates to COMPARE_IN", "[ast_to_duckdb]")
{
  std::vector<std::unique_ptr<node>> values;
  values.push_back(make_int_const(2));
  values.push_back(make_int_const(5));
  values.push_back(make_int_const(8));
  node orig{in_list{
    /*probe=*/make_ref(0),
    /*values=*/std::move(values),
    /*negated=*/false,
  }};
  auto out = sirius::ast::to_duckdb(orig);
  REQUIRE(out);
  REQUIRE(out->GetExpressionType() == ExpressionType::COMPARE_IN);
  // probe + 3 values = 4 children.
  REQUIRE(out->Cast<BoundOperatorExpression>().children.size() == 4);
}

TEST_CASE("ast_to_duckdb - in_list (negated=true) translates to COMPARE_NOT_IN", "[ast_to_duckdb]")
{
  std::vector<std::unique_ptr<node>> values;
  values.push_back(make_int_const(2));
  values.push_back(make_int_const(4));
  node orig{in_list{
    /*probe=*/make_ref(0),
    /*values=*/std::move(values),
    /*negated=*/true,
  }};
  auto out = sirius::ast::to_duckdb(orig);
  REQUIRE(out);
  REQUIRE(out->GetExpressionType() == ExpressionType::COMPARE_NOT_IN);
}

// ============================================================================
// function_call — sweep across a representative function in each category.
// ============================================================================

TEST_CASE("ast_to_duckdb - function_call add translates to BoundFunctionExpression",
          "[ast_to_duckdb]")
{
  std::vector<std::unique_ptr<node>> args;
  args.push_back(make_ref(0));
  args.push_back(make_int_const(3));
  node orig{function_call{sirius::function_id::add,
                          std::move(args),
                          sirius::logical_type::make(sirius::type_id::INTEGER)}};
  auto out = sirius::ast::to_duckdb(orig);
  REQUIRE(out);
  REQUIRE(out->GetExpressionClass() == ExpressionClass::BOUND_FUNCTION);
  auto const& fn = out->Cast<BoundFunctionExpression>();
  REQUIRE(fn.function.name == "+");
  REQUIRE(fn.children.size() == 2);
  REQUIRE(fn.return_type.id() == LogicalTypeId::INTEGER);
}

TEST_CASE("ast_to_duckdb - function_call substring translates to BoundFunctionExpression",
          "[ast_to_duckdb]")
{
  std::vector<std::unique_ptr<node>> args;
  args.push_back(make_ref(0));
  args.push_back(make_int_const(1));
  args.push_back(make_int_const(3));
  node orig{function_call{sirius::function_id::substring,
                          std::move(args),
                          sirius::logical_type::make(sirius::type_id::VARCHAR)}};
  auto out = sirius::ast::to_duckdb(orig);
  REQUIRE(out);
  REQUIRE(out->GetExpressionClass() == ExpressionClass::BOUND_FUNCTION);
  auto const& fn = out->Cast<BoundFunctionExpression>();
  REQUIRE(fn.function.name == "substring");
  REQUIRE(fn.children.size() == 3);
}

TEST_CASE("ast_to_duckdb - function_call like translates to BoundFunctionExpression",
          "[ast_to_duckdb]")
{
  std::vector<std::unique_ptr<node>> args;
  args.push_back(make_ref(0));
  args.push_back(make_str_const("x%"));
  node orig{function_call{sirius::function_id::like,
                          std::move(args),
                          sirius::logical_type::make(sirius::type_id::BOOLEAN)}};
  auto out = sirius::ast::to_duckdb(orig);
  REQUIRE(out);
  REQUIRE(out->GetExpressionClass() == ExpressionClass::BOUND_FUNCTION);
  auto const& fn = out->Cast<BoundFunctionExpression>();
  REQUIRE(fn.function.name == "~~");
  REQUIRE(fn.children.size() == 2);
}

TEST_CASE("ast_to_duckdb - function_call regexp_replace translates to BoundFunctionExpression",
          "[ast_to_duckdb]")
{
  std::vector<std::unique_ptr<node>> args;
  args.push_back(make_ref(0));
  args.push_back(make_str_const("a"));
  args.push_back(make_str_const("b"));
  node orig{function_call{sirius::function_id::regexp_replace,
                          std::move(args),
                          sirius::logical_type::make(sirius::type_id::VARCHAR)}};
  auto out = sirius::ast::to_duckdb(orig);
  REQUIRE(out);
  REQUIRE(out->GetExpressionClass() == ExpressionClass::BOUND_FUNCTION);
  auto const& fn = out->Cast<BoundFunctionExpression>();
  REQUIRE(fn.function.name == "regexp_replace");
  REQUIRE(fn.children.size() == 3);
}

TEST_CASE("ast_to_duckdb - function_call year translates to BoundFunctionExpression",
          "[ast_to_duckdb]")
{
  std::vector<std::unique_ptr<node>> args;
  args.push_back(make_ref(0));
  node orig{function_call{sirius::function_id::year,
                          std::move(args),
                          sirius::logical_type::make(sirius::type_id::BIGINT)}};
  auto out = sirius::ast::to_duckdb(orig);
  REQUIRE(out);
  REQUIRE(out->GetExpressionClass() == ExpressionClass::BOUND_FUNCTION);
  auto const& fn = out->Cast<BoundFunctionExpression>();
  REQUIRE(fn.function.name == "year");
  REQUIRE(fn.children.size() == 1);
}

// ============================================================================
// case_expr / coalesce return_type fidelity — regression guard (#701)
//
// The AST-path executor recovers each expression's return_type by round-tripping
// the node through to_duckdb, so case_expr and coalesce MUST emit their recorded
// return_type rather than a placeholder. The earlier INTEGER-typed cases above
// cannot catch a placeholder regression (the placeholder itself was INTEGER), so
// these assert a non-INTEGER (DECIMAL) return type survives the Sirius->DuckDB
// translation. A DECIMAL silently narrowed to INTEGER corrupts the materialized
// result and trips a downstream join type-mismatch.
// ============================================================================

TEST_CASE("ast_to_duckdb - case_expr preserves non-INTEGER (DECIMAL) return_type",
          "[ast_to_duckdb]")
{
  std::vector<case_expr::when_then> cases;
  cases.push_back(case_expr::when_then{
    std::make_unique<node>(comparison{sirius::comparison_type::gt, make_ref(0), make_int_const(1)}),
    make_int_const(10),
  });
  node orig{case_expr{std::move(cases),
                      /*else_=*/make_int_const(0),
                      sirius::logical_type::make_decimal(/*precision=*/38, /*scale=*/12)}};
  auto out = sirius::ast::to_duckdb(orig);
  REQUIRE(out);
  REQUIRE(out->GetExpressionClass() == ExpressionClass::BOUND_CASE);
  // Must be the recorded DECIMAL type, NOT an INTEGER placeholder.
  REQUIRE(out->Cast<BoundCaseExpression>().return_type.id() == LogicalTypeId::DECIMAL);
}

TEST_CASE("ast_to_duckdb - coalesce preserves non-INTEGER (DECIMAL) return_type", "[ast_to_duckdb]")
{
  std::vector<std::unique_ptr<node>> children;
  children.push_back(make_ref(0));
  children.push_back(make_ref(1));
  node orig{coalesce{std::move(children),
                     sirius::logical_type::make_decimal(/*precision=*/38, /*scale=*/12)}};
  auto out = sirius::ast::to_duckdb(orig);
  REQUIRE(out);
  REQUIRE(out->GetExpressionType() == ExpressionType::OPERATOR_COALESCE);
  // Must be the recorded DECIMAL type, NOT an INTEGER placeholder.
  REQUIRE(out->Cast<BoundOperatorExpression>().return_type.id() == LogicalTypeId::DECIMAL);
}

// ============================================================================
// Self-equivalence sweep — for every alternative, round-trip through
// from_duckdb(*to_duckdb(node)) and check structural equivalence with the
// original node tree.
// ============================================================================

TEST_CASE("ast_to_duckdb - reference self-equivalence via from_duckdb", "[ast_to_duckdb]")
{
  node orig{reference{/*column_index=*/7}};
  auto duck_expr = sirius::ast::to_duckdb(orig);
  REQUIRE(duck_expr);
  auto round = sirius::ast::from_duckdb(*duck_expr);
  REQUIRE(round);
  REQUIRE(same_reference(orig, *round));
}

TEST_CASE("ast_to_duckdb - constant self-equivalence via from_duckdb", "[ast_to_duckdb]")
{
  node orig{constant{
    /*payload=*/sirius::value{int32_t{99}},
    /*return_type=*/sirius::logical_type::make(sirius::type_id::INTEGER),
  }};
  auto duck_expr = sirius::ast::to_duckdb(orig);
  REQUIRE(duck_expr);
  auto round = sirius::ast::from_duckdb(*duck_expr);
  REQUIRE(round);
  REQUIRE(same_constant(orig, *round));
}

TEST_CASE("ast_to_duckdb - comparison self-equivalence via from_duckdb", "[ast_to_duckdb]")
{
  node orig{comparison{sirius::comparison_type::lt, make_ref(0), make_int_const(5)}};
  auto duck_expr = sirius::ast::to_duckdb(orig);
  REQUIRE(duck_expr);
  auto round = sirius::ast::from_duckdb(*duck_expr);
  REQUIRE(round);
  REQUIRE(same_comparison(orig, *round));
}

TEST_CASE("ast_to_duckdb - conjunction self-equivalence via from_duckdb", "[ast_to_duckdb]")
{
  std::vector<std::unique_ptr<node>> children;
  children.push_back(std::make_unique<node>(
    comparison{sirius::comparison_type::equal, make_ref(0), make_int_const(1)}));
  children.push_back(std::make_unique<node>(
    comparison{sirius::comparison_type::equal, make_ref(1), make_int_const(2)}));
  node orig{conjunction{conjunction::kind::op_and, std::move(children)}};
  auto duck_expr = sirius::ast::to_duckdb(orig);
  REQUIRE(duck_expr);
  auto round = sirius::ast::from_duckdb(*duck_expr);
  REQUIRE(round);
  REQUIRE(same_conjunction(orig, *round));
}

TEST_CASE("ast_to_duckdb - between self-equivalence via from_duckdb", "[ast_to_duckdb]")
{
  node orig{between{
    make_ref(0),
    make_int_const(1),
    make_int_const(10),
    /*lower_inclusive=*/true,
    /*upper_inclusive=*/true,
  }};
  auto duck_expr = sirius::ast::to_duckdb(orig);
  REQUIRE(duck_expr);
  auto round = sirius::ast::from_duckdb(*duck_expr);
  REQUIRE(round);
  REQUIRE(same_between(orig, *round));
}

TEST_CASE("ast_to_duckdb - case_expr self-equivalence via from_duckdb", "[ast_to_duckdb]")
{
  std::vector<case_expr::when_then> cases;
  cases.push_back(case_expr::when_then{
    std::make_unique<node>(
      comparison{sirius::comparison_type::equal, make_ref(0), make_int_const(1)}),
    make_int_const(10),
  });
  node orig{case_expr{
    std::move(cases), make_int_const(0), sirius::logical_type::make(sirius::type_id::INTEGER)}};
  auto duck_expr = sirius::ast::to_duckdb(orig);
  REQUIRE(duck_expr);
  auto round = sirius::ast::from_duckdb(*duck_expr);
  REQUIRE(round);
  REQUIRE(same_case_expr(orig, *round));
}

TEST_CASE("ast_to_duckdb - cast self-equivalence via from_duckdb", "[ast_to_duckdb]")
{
  node orig{cast{
    make_ref(0),
    sirius::logical_type::make(sirius::type_id::BIGINT),
    /*try_cast=*/false,
  }};
  auto duck_expr = sirius::ast::to_duckdb(orig);
  REQUIRE(duck_expr);
  auto round = sirius::ast::from_duckdb(*duck_expr);
  REQUIRE(round);
  REQUIRE(same_cast(orig, *round));
}

TEST_CASE("ast_to_duckdb - unary_op self-equivalence via from_duckdb", "[ast_to_duckdb]")
{
  node orig{unary_op{unary_op::kind::op_is_null, make_ref(0)}};
  auto duck_expr = sirius::ast::to_duckdb(orig);
  REQUIRE(duck_expr);
  auto round = sirius::ast::from_duckdb(*duck_expr);
  REQUIRE(round);
  REQUIRE(same_unary_op(orig, *round));
}

TEST_CASE("ast_to_duckdb - coalesce self-equivalence via from_duckdb", "[ast_to_duckdb]")
{
  std::vector<std::unique_ptr<node>> children;
  children.push_back(make_ref(0));
  children.push_back(make_ref(1));
  children.push_back(make_int_const(0));
  node orig{coalesce{std::move(children), sirius::logical_type::make(sirius::type_id::INTEGER)}};
  auto duck_expr = sirius::ast::to_duckdb(orig);
  REQUIRE(duck_expr);
  auto round = sirius::ast::from_duckdb(*duck_expr);
  REQUIRE(round);
  REQUIRE(same_coalesce(orig, *round));
}

TEST_CASE("ast_to_duckdb - in_list self-equivalence via from_duckdb", "[ast_to_duckdb]")
{
  std::vector<std::unique_ptr<node>> values;
  values.push_back(make_int_const(2));
  values.push_back(make_int_const(5));
  node orig{in_list{make_ref(0), std::move(values), /*negated=*/true}};
  auto duck_expr = sirius::ast::to_duckdb(orig);
  REQUIRE(duck_expr);
  auto round = sirius::ast::from_duckdb(*duck_expr);
  REQUIRE(round);
  REQUIRE(same_in_list(orig, *round));
}

TEST_CASE("ast_to_duckdb - function_call self-equivalence via from_duckdb", "[ast_to_duckdb]")
{
  std::vector<std::unique_ptr<node>> args;
  args.push_back(make_ref(0));
  args.push_back(make_int_const(3));
  node orig{function_call{sirius::function_id::add,
                          std::move(args),
                          sirius::logical_type::make(sirius::type_id::INTEGER)}};
  auto duck_expr = sirius::ast::to_duckdb(orig);
  REQUIRE(duck_expr);
  auto round = sirius::ast::from_duckdb(*duck_expr);
  REQUIRE(round);
  REQUIRE(same_function_call(orig, *round));
}
