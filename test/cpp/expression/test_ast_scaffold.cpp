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

// Tests for sirius::ast — the Sirius-native expression AST scaffold.
//
// These tests verify that every node struct can be instantiated, that the
// sirius::ast::node variant holds every node kind, that recursive trees build
// and destruct cleanly, and that node is move-only.
//
// Pure type-only test: no SiriusContext, no GPU memory, no DuckDB types.

#include "catch.hpp"
#include "expression/ast/node.hpp"
#include "expression/join_condition.hpp"
#include "helper/logical_type.hpp"

#include <memory>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

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
// Compile-time invariants
// ============================================================================

static_assert(!std::is_copy_constructible_v<node>,
              "sirius::ast::node must be move-only (variant over unique_ptr-holding structs).");
static_assert(!std::is_copy_assignable_v<node>, "sirius::ast::node must be move-only.");
static_assert(std::is_move_constructible_v<node>, "sirius::ast::node must be move-constructible.");
static_assert(std::is_move_assignable_v<node>, "sirius::ast::node must be move-assignable.");
static_assert(std::variant_size_v<node::variant_t> == 12,
              "sirius::ast::node has exactly 12 alternatives.");

// ============================================================================
// Per-node instantiation
// ============================================================================

TEST_CASE("ast_scaffold - reference holds a column index", "[ast_scaffold]")
{
  reference r{42};
  REQUIRE(r.column_index == 42);
  node n{std::move(r)};
  REQUIRE(n.holds<reference>());
  REQUIRE(n.get<reference>().column_index == 42);
}

TEST_CASE("ast_scaffold - constant default-constructs", "[ast_scaffold]")
{
  constant c{};
  node n{std::move(c)};
  REQUIRE(n.holds<constant>());
}

TEST_CASE("ast_scaffold - comparison holds recursive children", "[ast_scaffold]")
{
  comparison c;
  c.op    = sirius::comparison_type::equal;
  c.left  = std::make_unique<node>(reference{0});
  c.right = std::make_unique<node>(reference{1});
  REQUIRE(c.op == sirius::comparison_type::equal);
  REQUIRE(c.left);
  REQUIRE(c.right);
  node n{std::move(c)};
  REQUIRE(n.holds<comparison>());
}

TEST_CASE("ast_scaffold - conjunction holds N-ary children", "[ast_scaffold]")
{
  conjunction conj;
  conj.op = conjunction::kind::op_and;
  conj.children.push_back(std::make_unique<node>(reference{0}));
  conj.children.push_back(std::make_unique<node>(reference{1}));
  conj.children.push_back(std::make_unique<node>(reference{2}));
  REQUIRE(conj.children.size() == 3);
  node n{std::move(conj)};
  REQUIRE(n.holds<conjunction>());
}

TEST_CASE("ast_scaffold - conjunction op defaults to invalid sentinel", "[ast_scaffold]")
{
  conjunction conj;
  REQUIRE(conj.op == conjunction::kind::invalid);
}

TEST_CASE("ast_scaffold - between holds three children and inclusivity flags", "[ast_scaffold]")
{
  between b;
  b.input           = std::make_unique<node>(reference{0});
  b.lower           = std::make_unique<node>(reference{1});
  b.upper           = std::make_unique<node>(reference{2});
  b.lower_inclusive = true;
  b.upper_inclusive = false;
  REQUIRE(b.input);
  REQUIRE(b.lower);
  REQUIRE(b.upper);
  REQUIRE(b.lower_inclusive);
  REQUIRE_FALSE(b.upper_inclusive);
  node n{std::move(b)};
  REQUIRE(n.holds<between>());
}

TEST_CASE("ast_scaffold - case_expr holds when/then pairs and an else clause", "[ast_scaffold]")
{
  case_expr c;
  case_expr::when_then wt;
  wt.when_ = std::make_unique<node>(reference{0});
  wt.then_ = std::make_unique<node>(reference{1});
  c.cases.push_back(std::move(wt));
  c.else_ = std::make_unique<node>(reference{2});
  REQUIRE(c.cases.size() == 1);
  REQUIRE(c.cases[0].when_);
  REQUIRE(c.cases[0].then_);
  REQUIRE(c.else_);
  node n{std::move(c)};
  REQUIRE(n.holds<case_expr>());
}

TEST_CASE("ast_scaffold - cast holds child, target type, and try flag", "[ast_scaffold]")
{
  cast cst;
  cst.child       = std::make_unique<node>(reference{0});
  cst.try_cast    = true;
  cst.target_type = sirius::logical_type::make(sirius::type_id::BIGINT);
  REQUIRE(cst.child);
  REQUIRE(cst.try_cast);
  REQUIRE(cst.target_type.id() == sirius::type_id::BIGINT);
  node n{std::move(cst)};
  REQUIRE(n.holds<cast>());
  REQUIRE(n.get<cast>().target_type.id() == sirius::type_id::BIGINT);
}

TEST_CASE("ast_scaffold - cast target_type defaults to SQLNULL placeholder", "[ast_scaffold]")
{
  cast cst;
  REQUIRE(cst.target_type.id() == sirius::type_id::SQLNULL);
  REQUIRE_FALSE(cst.try_cast);
}

TEST_CASE("ast_scaffold - unary_op wraps a single child", "[ast_scaffold]")
{
  unary_op u;
  u.op    = unary_op::kind::op_is_null;
  u.child = std::make_unique<node>(reference{0});
  REQUIRE(u.op == unary_op::kind::op_is_null);
  REQUIRE(u.child);
  node n{std::move(u)};
  REQUIRE(n.holds<unary_op>());
}

TEST_CASE("ast_scaffold - unary_op op defaults to invalid sentinel", "[ast_scaffold]")
{
  unary_op u;
  REQUIRE(u.op == unary_op::kind::invalid);
}

TEST_CASE("ast_scaffold - coalesce holds N-ary children", "[ast_scaffold]")
{
  coalesce co;
  co.children.push_back(std::make_unique<node>(reference{0}));
  co.children.push_back(std::make_unique<node>(reference{1}));
  co.children.push_back(std::make_unique<node>(reference{2}));
  REQUIRE(co.children.size() == 3);
  node n{std::move(co)};
  REQUIRE(n.holds<coalesce>());
}

TEST_CASE("ast_scaffold - in_list holds a probe, values, and a negation flag", "[ast_scaffold]")
{
  in_list il;
  il.probe = std::make_unique<node>(reference{0});
  il.values.push_back(std::make_unique<node>(reference{1}));
  il.values.push_back(std::make_unique<node>(reference{2}));
  il.negated = true;
  REQUIRE(il.probe);
  REQUIRE(il.values.size() == 2);
  REQUIRE(il.negated);
  node n{std::move(il)};
  REQUIRE(n.holds<in_list>());
}

TEST_CASE("ast_scaffold - function_call holds arguments and a return type", "[ast_scaffold]")
{
  std::vector<std::unique_ptr<node>> args;
  args.push_back(std::make_unique<node>(reference{0}));
  args.push_back(std::make_unique<node>(reference{1}));
  function_call fc{sirius::function_id::add,
                   std::move(args),
                   sirius::logical_type::make(sirius::type_id::INTEGER)};
  REQUIRE(fc.arguments().size() == 2);
  node n{std::move(fc)};
  REQUIRE(n.holds<function_call>());
}

// ============================================================================
// Recursive tree construction (proves unique_ptr<node> composition works).
// ============================================================================

TEST_CASE("ast_scaffold - comparison tree with reference children round-trips via variant",
          "[ast_scaffold]")
{
  comparison c;
  c.op    = sirius::comparison_type::lt;
  c.left  = std::make_unique<node>(reference{7});
  c.right = std::make_unique<node>(reference{11});

  node root{std::move(c)};
  REQUIRE(root.holds<comparison>());

  auto& c_ref = root.get<comparison>();
  REQUIRE(c_ref.op == sirius::comparison_type::lt);
  REQUIRE(c_ref.left->holds<reference>());
  REQUIRE(c_ref.right->holds<reference>());
  REQUIRE(c_ref.left->get<reference>().column_index == 7);
  REQUIRE(c_ref.right->get<reference>().column_index == 11);
}

TEST_CASE("ast_scaffold - deeper tree (conjunction of comparisons) destructs cleanly",
          "[ast_scaffold]")
{
  conjunction conj;
  conj.op = conjunction::kind::op_or;
  for (uint32_t i = 0; i < 3; ++i) {
    comparison c;
    c.op    = sirius::comparison_type::equal;
    c.left  = std::make_unique<node>(reference{i});
    c.right = std::make_unique<node>(reference{i + 100});
    conj.children.push_back(std::make_unique<node>(std::move(c)));
  }
  REQUIRE(conj.children.size() == 3);

  node root{std::move(conj)};
  REQUIRE(root.holds<conjunction>());
  REQUIRE(root.get<conjunction>().children.size() == 3);
  // Implicit: destructor chain on leaving scope proves unique_ptr<node> cleans up recursively.
}
