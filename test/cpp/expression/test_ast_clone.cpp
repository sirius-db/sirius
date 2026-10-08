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

// Tests for sirius::ast::clone — the deep-clone helper used by the aggregate
// copy path. node is move-only, so clone must reconstruct each alternative and
// recursively duplicate every child unique_ptr into an independent allocation.

#include "ast_test_builders.hpp"
#include "catch.hpp"
#include "expression/ast/utils.hpp"

// sirius — node accessors and per-node struct types
#include "expression/aggregate_id.hpp"
#include "expression/ast/aggregate.hpp"
#include "expression/ast/function_call.hpp"
#include "expression/ast/node.hpp"
#include "expression/ast/reference.hpp"
#include "expression/function_id.hpp"
#include "helper/logical_type.hpp"

// standard library
#include <memory>
#include <utility>
#include <vector>

using sirius::ast::aggregate;
using sirius::ast::clone;
using sirius::ast::function_call;
using sirius::ast::node;
using sirius::ast::reference;

// ============================================================================
// Test 1: reference
// ============================================================================

TEST_CASE("ast_clone - cloning a reference yields an independent node with equal fields",
          "[ast_clone]")
{
  auto src = std::make_unique<node>(
    reference{/*column_index=*/7, sirius::logical_type::make(sirius::type_id::INTEGER)});

  auto cloned = clone(*src);

  REQUIRE(cloned != nullptr);
  REQUIRE(cloned.get() != src.get());  // independent allocation
  REQUIRE(cloned->holds<reference>());
  auto const& ref = cloned->get<reference>();
  REQUIRE(ref.column_index == 7);
  REQUIRE(ref.return_type().id() == sirius::type_id::INTEGER);
}

// ============================================================================
// Test 2: aggregate with one reference child
// ============================================================================

TEST_CASE("ast_clone - cloning an aggregate deep-clones its reference child", "[ast_clone]")
{
  std::vector<std::unique_ptr<node>> args;
  args.push_back(std::make_unique<node>(
    reference{/*column_index=*/3, sirius::logical_type::make(sirius::type_id::BIGINT)}));

  auto src = std::make_unique<node>(aggregate{sirius::aggregate_id::sum,
                                              std::move(args),
                                              sirius::logical_type::make(sirius::type_id::BIGINT),
                                              /*distinct=*/true});

  auto cloned = clone(*src);

  REQUIRE(cloned != nullptr);
  REQUIRE(cloned.get() != src.get());
  REQUIRE(cloned->holds<aggregate>());
  auto const& agg = cloned->get<aggregate>();
  REQUIRE(agg.function() == sirius::aggregate_id::sum);
  REQUIRE(agg.distinct() == true);
  REQUIRE(agg.return_type().id() == sirius::type_id::BIGINT);
  REQUIRE(agg.arguments().size() == 1);

  // The child is an independent allocation holding an equal reference.
  auto const& src_child = src->get<aggregate>().arguments()[0];
  REQUIRE(agg.arguments()[0].get() != src_child.get());
  REQUIRE(agg.arguments()[0]->holds<reference>());
  REQUIRE(agg.arguments()[0]->get<reference>().column_index == 3);
}

// ============================================================================
// Test 3: function_call (struct_pack shape) with two reference children
// ============================================================================

TEST_CASE("ast_clone - cloning a struct_pack function_call deep-clones every child", "[ast_clone]")
{
  std::vector<std::unique_ptr<node>> args;
  args.push_back(std::make_unique<node>(
    reference{/*column_index=*/1, sirius::logical_type::make(sirius::type_id::INTEGER)}));
  args.push_back(std::make_unique<node>(
    reference{/*column_index=*/2, sirius::logical_type::make(sirius::type_id::INTEGER)}));

  auto src =
    std::make_unique<node>(function_call{sirius::function_id::struct_pack,
                                         std::move(args),
                                         sirius::logical_type::make(sirius::type_id::INTEGER)});

  auto cloned = clone(*src);

  REQUIRE(cloned != nullptr);
  REQUIRE(cloned->holds<function_call>());
  auto const& fc = cloned->get<function_call>();
  REQUIRE(fc.function() == sirius::function_id::struct_pack);
  REQUIRE(fc.arguments().size() == 2);

  auto const& src_args = src->get<function_call>().arguments();
  for (std::size_t i = 0; i < 2; ++i) {
    REQUIRE(fc.arguments()[i].get() != src_args[i].get());
    REQUIRE(fc.arguments()[i]->holds<reference>());
    REQUIRE(fc.arguments()[i]->get<reference>().column_index ==
            src_args[i]->get<reference>().column_index);
  }
}

TEST_CASE("native reference traversal covers every child shape without changing indices",
          "[ast_references]")
{
  using namespace sirius::ast;
  using namespace sirius::ast::test;
  auto const type = sirius::logical_type::make(sirius::type_id::INTEGER);
  std::vector<node> roots;
  roots.emplace_back(reference{9, type});
  roots.emplace_back(constant{sirius::value{int32_t{1}}, type});
  roots.emplace_back(comparison{sirius::comparison_type::equal, make_ref(1), make_ref(2)});
  conjunction conjunction_node;
  conjunction_node.children.push_back(make_ref(3));
  conjunction_node.children.push_back(make_ref(3));
  roots.emplace_back(std::move(conjunction_node));
  roots.emplace_back(between{make_ref(4), make_ref(5), make_ref(6), true, true});
  std::vector<case_expr::when_then> cases;
  cases.push_back({make_ref(7), make_ref(8)});
  roots.emplace_back(case_expr{std::move(cases), make_ref(9), type});
  roots.emplace_back(cast{make_ref(10), type, true, cast_kind::semantic});
  roots.emplace_back(unary_op{unary_op::kind::op_is_null, make_ref(11)});
  std::vector<std::unique_ptr<node>> children;
  children.push_back(make_ref(12));
  children.push_back(make_int_const(0));
  roots.emplace_back(coalesce{std::move(children), type});
  in_list list;
  list.probe = make_ref(13);
  list.values.push_back(make_ref(14));
  list.values.push_back(make_int_const(0));
  roots.emplace_back(std::move(list));
  std::vector<std::unique_ptr<node>> args;
  args.push_back(make_ref(15));
  args.push_back(make_ref(16));
  roots.emplace_back(function_call{sirius::function_id::add, std::move(args), type});
  args.clear();
  args.push_back(make_ref(17));
  roots.emplace_back(aggregate{sirius::aggregate_id::sum, std::move(args), type, false});

  std::vector<uint32_t> indices;
  for (auto const& root : roots) {
    visit_references(root, [&](reference const& ref) { indices.push_back(ref.column_index); });
  }
  CHECK(indices ==
        std::vector<uint32_t>{9, 1, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17});
}
