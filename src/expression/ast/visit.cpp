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

#include "expression/ast/visit.hpp"

// sirius
#include "expression/ast/node.hpp"

// standard library
#include <functional>
#include <memory>
#include <type_traits>
#include <variant>

namespace sirius::ast {

namespace {

void visit_child(std::unique_ptr<node> const& child,
                 std::function<void(reference const&)> const& fn)
{
  if (child) { visit_references(*child, fn); }
}

}  // namespace

void visit_references(node const& root, std::function<void(reference const&)> const& fn)
{
  std::visit(
    [&](auto const& alt) {
      using T = std::decay_t<decltype(alt)>;

      if constexpr (std::is_same_v<T, reference>) {
        fn(alt);
      } else if constexpr (std::is_same_v<T, constant>) {
        // leaf, no references
      } else if constexpr (std::is_same_v<T, comparison>) {
        visit_child(alt.left, fn);
        visit_child(alt.right, fn);
      } else if constexpr (std::is_same_v<T, conjunction>) {
        for (auto const& child : alt.children) {
          visit_child(child, fn);
        }
      } else if constexpr (std::is_same_v<T, between>) {
        visit_child(alt.input, fn);
        visit_child(alt.lower, fn);
        visit_child(alt.upper, fn);
      } else if constexpr (std::is_same_v<T, case_expr>) {
        for (auto const& wt : alt.cases) {
          visit_child(wt.when_, fn);
          visit_child(wt.then_, fn);
        }
        visit_child(alt.else_, fn);
      } else if constexpr (std::is_same_v<T, cast>) {
        visit_child(alt.child, fn);
      } else if constexpr (std::is_same_v<T, unary_op>) {
        visit_child(alt.child, fn);
      } else if constexpr (std::is_same_v<T, coalesce>) {
        for (auto const& child : alt.children) {
          visit_child(child, fn);
        }
      } else if constexpr (std::is_same_v<T, in_list>) {
        visit_child(alt.probe, fn);
        for (auto const& child : alt.values) {
          visit_child(child, fn);
        }
      } else if constexpr (std::is_same_v<T, function_call>) {
        for (auto const& child : alt.arguments()) {
          visit_child(child, fn);
        }
      } else if constexpr (std::is_same_v<T, aggregate>) {
        for (auto const& child : alt.arguments()) {
          visit_child(child, fn);
        }
      } else {
        static_assert(sizeof(T) == 0,
                      "Unhandled sirius::ast alternative in visit_references — add a "
                      "traversal arm for the new variant member");
      }
    },
    root.v);
}

}  // namespace sirius::ast
