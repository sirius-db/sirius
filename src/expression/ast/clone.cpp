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

#include "expression/ast/utils.hpp"

// sirius
#include "expression/ast/node.hpp"  // complete node: needed to construct/destroy child unique_ptr vectors

// standard library
#include <algorithm>
#include <iterator>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

namespace sirius::ast {

namespace {

// Deep-clone a single optional child node owner.
std::unique_ptr<node> clone_child(std::unique_ptr<node> const& child)
{
  return child ? clone(*child) : nullptr;
}

// Deep-clone a vector of child node owners.
std::vector<std::unique_ptr<node>> clone_children(
  std::vector<std::unique_ptr<node>> const& children)
{
  std::vector<std::unique_ptr<node>> out;
  out.reserve(children.size());
  std::ranges::transform(children, std::back_inserter(out), clone_child);
  return out;
}

}  // namespace

std::unique_ptr<node> clone(node const& src)
{
  return std::visit(
    [](auto const& alt) -> std::unique_ptr<node> {
      using T = std::decay_t<decltype(alt)>;

      if constexpr (std::is_same_v<T, reference>) {
        return std::make_unique<node>(reference{alt.column_index, alt.return_type});
      } else if constexpr (std::is_same_v<T, constant>) {
        return std::make_unique<node>(constant{alt.payload, alt.return_type});
      } else if constexpr (std::is_same_v<T, comparison>) {
        return std::make_unique<node>(
          comparison{alt.op, clone_child(alt.left), clone_child(alt.right)});
      } else if constexpr (std::is_same_v<T, conjunction>) {
        return std::make_unique<node>(conjunction{alt.op, clone_children(alt.children)});
      } else if constexpr (std::is_same_v<T, between>) {
        return std::make_unique<node>(between{clone_child(alt.input),
                                              clone_child(alt.lower),
                                              clone_child(alt.upper),
                                              alt.lower_inclusive,
                                              alt.upper_inclusive});
      } else if constexpr (std::is_same_v<T, case_expr>) {
        std::vector<case_expr::when_then> cases;
        cases.reserve(alt.cases.size());
        for (auto const& wt : alt.cases) {
          cases.push_back(case_expr::when_then{clone_child(wt.when_), clone_child(wt.then_)});
        }
        return std::make_unique<node>(
          case_expr{std::move(cases), clone_child(alt.else_), alt.return_type()});
      } else if constexpr (std::is_same_v<T, cast>) {
        return std::make_unique<node>(cast{clone_child(alt.child), alt.target_type, alt.try_cast});
      } else if constexpr (std::is_same_v<T, unary_op>) {
        return std::make_unique<node>(unary_op{alt.op, clone_child(alt.child)});
      } else if constexpr (std::is_same_v<T, coalesce>) {
        return std::make_unique<node>(coalesce{clone_children(alt.children), alt.return_type()});
      } else if constexpr (std::is_same_v<T, in_list>) {
        return std::make_unique<node>(
          in_list{clone_child(alt.probe), clone_children(alt.values), alt.negated});
      } else if constexpr (std::is_same_v<T, function_call>) {
        return std::make_unique<node>(
          function_call{alt.function(), clone_children(alt.arguments()), alt.return_type()});
      } else if constexpr (std::is_same_v<T, aggregate>) {
        return std::make_unique<node>(aggregate{
          alt.function(), clone_children(alt.arguments()), alt.return_type(), alt.distinct()});
      } else {
        static_assert(sizeof(T) == 0,
                      "Unhandled sirius::ast alternative in clone(node) — add a clone arm "
                      "for the new variant member");
      }
    },
    src.v);
}

}  // namespace sirius::ast
