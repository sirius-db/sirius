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

#include "catch.hpp"
#include "expression/ast/function_call.hpp"
#include "expression/ast/node.hpp"
#include "expression/ast/reference.hpp"
#include "expression/ast/substitute.hpp"
#include "expression/function_id.hpp"
#include "helper/logical_type.hpp"

#include <memory>
#include <utility>
#include <vector>

using sirius::ast::function_call;
using sirius::ast::node;
using sirius::ast::reference;
using sirius::ast::substitute_references;

namespace {

std::unique_ptr<node> make_reference(uint32_t column_index)
{
  return std::make_unique<node>(
    reference{column_index, sirius::logical_type::make(sirius::type_id::INTEGER)});
}

}  // namespace

TEST_CASE("ast_substitute - remaps a reference through a reordering inner projection",
          "[ast_substitute]")
{
  std::vector<std::unique_ptr<node>> inner_select_list;
  inner_select_list.push_back(make_reference(2));
  inner_select_list.push_back(make_reference(0));

  auto substituted = substitute_references(*make_reference(0), inner_select_list);

  REQUIRE(substituted->holds<reference>());
  REQUIRE(substituted->get<reference>().column_index == 2);
}

TEST_CASE("ast_substitute - substitutes references inside a function call", "[ast_substitute]")
{
  std::vector<std::unique_ptr<node>> inner_select_list;
  inner_select_list.push_back(make_reference(4));

  std::vector<std::unique_ptr<node>> args;
  args.push_back(make_reference(0));
  args.push_back(make_reference(0));
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
