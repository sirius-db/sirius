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

#include "ast_test_builders.hpp"
#include "catch.hpp"
#include "expression/ast/cast.hpp"
#include "expression/ast/function_call.hpp"
#include "expression/ast/node.hpp"
#include "expression/ast/reference.hpp"
#include "expression/ast/utils.hpp"
#include "expression/function_id.hpp"
#include "helper/logical_type.hpp"

#include <memory>
#include <utility>
#include <vector>

using sirius::ast::cast;
using sirius::ast::cast_kind;
using sirius::ast::clone;
using sirius::ast::function_call;
using sirius::ast::node;
using sirius::ast::reference;
using sirius::ast::substitute_references;
using sirius::ast::test::make_ref;

TEST_CASE("ast_substitute - remaps a reference through a reordering inner projection",
          "[ast_substitute]")
{
  std::vector<std::unique_ptr<node>> inner_select_list;
  inner_select_list.push_back(make_ref(2));
  inner_select_list.push_back(make_ref(0));

  auto substituted = substitute_references(*make_ref(0), inner_select_list);

  REQUIRE(substituted->holds<reference>());
  REQUIRE(substituted->get<reference>().column_index == 2);
}

TEST_CASE("ast_substitute - substitutes references inside a function call", "[ast_substitute]")
{
  std::vector<std::unique_ptr<node>> inner_select_list;
  inner_select_list.push_back(make_ref(4));

  std::vector<std::unique_ptr<node>> args;
  args.push_back(make_ref(0));
  args.push_back(make_ref(0));
  auto outer =
    std::make_unique<node>(function_call{sirius::function_id::add,
                                         std::move(args),
                                         sirius::logical_type::make(sirius::type_id::INTEGER)});

  auto substituted = substitute_references(*outer, inner_select_list);

  REQUIRE(substituted->holds<function_call>());
  auto const& fc = substituted->get<function_call>();
  REQUIRE(fc.arguments().size() == 2);
  REQUIRE(fc.arguments()[0]->get<reference>().column_index == 4);
  REQUIRE(fc.arguments()[1]->get<reference>().column_index == 4);
}

TEST_CASE("ast_substitute - clone and substitution preserve cast kind and try_cast",
          "[ast_substitute]")
{
  // rebuild() must carry a cast's provenance tag and TRY semantics through both transforms: a
  // carrier_restore flattened back to the default-constructed semantic kind would turn a
  // planner-emitted representation restore into a value-converting SQL cast (and vice versa), and
  // a dropped try_cast would turn null-on-failure into throw-on-failure.
  auto const check = [](cast_kind kind, bool try_cast) {
    node const original{
      cast{make_ref(0), sirius::logical_type::make(sirius::type_id::DATE), try_cast, kind}};

    auto cloned = clone(original);
    REQUIRE(cloned->holds<cast>());
    REQUIRE(cloned->get<cast>().kind == kind);
    REQUIRE(cloned->get<cast>().try_cast == try_cast);
    REQUIRE(cloned->get<cast>().target_type.id() == sirius::type_id::DATE);
    REQUIRE(cloned->get<cast>().child->get<reference>().column_index == 0);

    std::vector<std::unique_ptr<node>> inner_select_list;
    inner_select_list.push_back(make_ref(3));
    auto substituted = substitute_references(original, inner_select_list);
    REQUIRE(substituted->holds<cast>());
    REQUIRE(substituted->get<cast>().kind == kind);
    REQUIRE(substituted->get<cast>().try_cast == try_cast);
    REQUIRE(substituted->get<cast>().target_type.id() == sirius::type_id::DATE);
    REQUIRE(substituted->get<cast>().child->get<reference>().column_index == 3);
  };

  check(cast_kind::carrier_restore, /*try_cast=*/false);
  check(cast_kind::semantic, /*try_cast=*/true);
}
