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

// Tests for sirius::ast::to_string, the single-line display renderer behind the
// operator params_to_string overrides (telemetry operator metadata).

#include "ast_test_builders.hpp"
#include "catch.hpp"
#include "expression/aggregate_id.hpp"
#include "expression/ast/node.hpp"
#include "expression/ast/to_string.hpp"
#include "expression/function_id.hpp"
#include "expression/join_condition.hpp"

#include <memory>
#include <utility>
#include <vector>

using namespace sirius::ast;
using sirius::ast::test::make_int_const;
using sirius::ast::test::make_ref;
using sirius::ast::test::make_str_const;

TEST_CASE("ast to_string renders leaves", "[ast][to_string]")
{
  CHECK(to_string(*make_ref(3)) == "#3");
  CHECK(to_string(*make_int_const(42)) == "42");
  CHECK(to_string(*make_str_const("abc")) == "'abc'");
}

TEST_CASE("ast to_string renders comparisons and conjunctions", "[ast][to_string]")
{
  comparison cmp;
  cmp.op    = sirius::comparison_type::le;
  cmp.left  = make_ref(0);
  cmp.right = make_int_const(5);

  comparison cmp2;
  cmp2.op    = sirius::comparison_type::not_equal;
  cmp2.left  = make_ref(1);
  cmp2.right = make_str_const("x");

  conjunction conj;
  conj.op = conjunction::kind::op_and;
  conj.children.push_back(std::make_unique<node>(std::move(cmp)));
  conj.children.push_back(std::make_unique<node>(std::move(cmp2)));

  CHECK(to_string(node{std::move(conj)}) == "#0 <= 5 AND #1 != 'x'");
}

TEST_CASE("ast to_string parenthesizes nested conjunctions", "[ast][to_string]")
{
  comparison a;
  a.op    = sirius::comparison_type::equal;
  a.left  = make_ref(0);
  a.right = make_int_const(1);
  comparison b;
  b.op    = sirius::comparison_type::equal;
  b.left  = make_ref(1);
  b.right = make_int_const(2);

  conjunction inner;
  inner.op = conjunction::kind::op_or;
  inner.children.push_back(std::make_unique<node>(std::move(a)));
  inner.children.push_back(std::make_unique<node>(std::move(b)));

  comparison c;
  c.op    = sirius::comparison_type::gt;
  c.left  = make_ref(2);
  c.right = make_int_const(3);

  conjunction outer;
  outer.op = conjunction::kind::op_and;
  outer.children.push_back(std::make_unique<node>(std::move(inner)));
  outer.children.push_back(std::make_unique<node>(std::move(c)));

  CHECK(to_string(node{std::move(outer)}) == "(#0 = 1 OR #1 = 2) AND #2 > 3");
}

TEST_CASE("ast to_string renders between, cast, unary, in_list, coalesce", "[ast][to_string]")
{
  between btw;
  btw.input = make_ref(0);
  btw.lower = make_int_const(1);
  btw.upper = make_int_const(9);
  CHECK(to_string(node{std::move(btw)}) == "#0 BETWEEN 1 AND 9");

  between half_open;
  half_open.input           = make_ref(0);
  half_open.lower           = make_int_const(1);
  half_open.upper           = make_int_const(9);
  half_open.upper_inclusive = false;
  CHECK(to_string(node{std::move(half_open)}) == "(#0 >= 1 AND #0 < 9)");

  cast c;
  c.child       = make_ref(2);
  c.target_type = sirius::logical_type::make(sirius::type_id::DOUBLE);
  CHECK(to_string(node{std::move(c)}) == "CAST(#2 AS DOUBLE)");

  unary_op is_null;
  is_null.op    = unary_op::kind::op_is_null;
  is_null.child = make_ref(1);
  CHECK(to_string(node{std::move(is_null)}) == "#1 IS NULL");

  in_list in;
  in.probe = make_ref(0);
  in.values.push_back(make_int_const(1));
  in.values.push_back(make_int_const(2));
  CHECK(to_string(node{std::move(in)}) == "#0 IN (1, 2)");

  std::vector<std::unique_ptr<node>> coalesce_children;
  coalesce_children.push_back(make_ref(0));
  coalesce_children.push_back(make_int_const(0));
  coalesce co{std::move(coalesce_children), sirius::logical_type::make(sirius::type_id::INTEGER)};
  CHECK(to_string(node{std::move(co)}) == "COALESCE(#0, 0)");
}

TEST_CASE("ast to_string renders aggregates and function calls", "[ast][to_string]")
{
  std::vector<std::unique_ptr<node>> sum_args;
  sum_args.push_back(make_ref(2));
  aggregate sum_agg{sirius::aggregate_id::sum,
                    std::move(sum_args),
                    sirius::logical_type::make(sirius::type_id::BIGINT),
                    /*distinct=*/false};
  CHECK(to_string(node{std::move(sum_agg)}) == "sum(#2)");

  aggregate count_star{sirius::aggregate_id::count_star,
                       {},
                       sirius::logical_type::make(sirius::type_id::BIGINT),
                       /*distinct=*/false};
  CHECK(to_string(node{std::move(count_star)}) == "count_star(*)");

  std::vector<std::unique_ptr<node>> distinct_args;
  distinct_args.push_back(make_ref(1));
  aggregate count_distinct{sirius::aggregate_id::count,
                           std::move(distinct_args),
                           sirius::logical_type::make(sirius::type_id::BIGINT),
                           /*distinct=*/true};
  CHECK(to_string(node{std::move(count_distinct)}) == "count(DISTINCT #1)");
}

TEST_CASE("join condition list renders as single line", "[ast][to_string]")
{
  duckdb::vector<sirius::join_condition> conditions;
  conditions.push_back(
    sirius::join_condition{make_ref(0), make_ref(1), sirius::comparison_type::equal});
  conditions.push_back(
    sirius::join_condition{make_ref(2), make_ref(3), sirius::comparison_type::lt});
  CHECK(sirius::to_string(conditions) == "#0 = #1 AND #2 < #3");
}

TEST_CASE("ast to_string tolerates null children", "[ast][to_string]")
{
  comparison cmp;
  cmp.op   = sirius::comparison_type::equal;
  cmp.left = make_ref(0);
  // right stays null
  CHECK(to_string(node{std::move(cmp)}) == "#0 = ?");
}
