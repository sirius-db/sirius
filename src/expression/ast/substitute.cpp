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

#include "expression/ast/substitute.hpp"

// sirius
#include "expression/ast/clone.hpp"
#include "expression/ast/node.hpp"

// standard library
#include <algorithm>
#include <iterator>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

namespace sirius::ast {

namespace {

// Recurse into an optional child; null slots stay null.
std::unique_ptr<node> substitute_child(std::unique_ptr<node> const& child,
                                       std::vector<std::unique_ptr<node>> const& inner_select_list)
{
  return child ? substitute_references(*child, inner_select_list) : nullptr;
}

// Apply substitute_child to every element of a child vector.
std::vector<std::unique_ptr<node>> substitute_children(
  std::vector<std::unique_ptr<node>> const& children,
  std::vector<std::unique_ptr<node>> const& inner_select_list)
{
  std::vector<std::unique_ptr<node>> out;
  out.reserve(children.size());
  std::ranges::transform(children, std::back_inserter(out), [&](auto const& child) {
    return substitute_child(child, inner_select_list);
  });
  return out;
}

}  // namespace

// Deep-copy @p src, remapping every reference #i to a clone of inner_select_list[i].
// Used when folding adjacent projections: outer references index the inner
// projection's output columns, so each #i is spliced with the inner expression
// that produced column i (whose own references already point at the inner input).
std::unique_ptr<node> substitute_references(
  node const& src, std::vector<std::unique_ptr<node>> const& inner_select_list)
{
  return std::visit(
    [&](auto const& alt) -> std::unique_ptr<node> {
      using T = std::decay_t<decltype(alt)>;

      if constexpr (std::is_same_v<T, reference>) {
        // Out-of-range or null inner slots are left unchanged — a null slot is the
        // from_duckdb signal for an unsupported expression and must not be folded through.
        if (alt.column_index >= inner_select_list.size() || !inner_select_list[alt.column_index]) {
          return clone(src);
        }
        // Each outer reference site gets its own clone; duplicate sites duplicate work.
        return clone(*inner_select_list[alt.column_index]);
      } else if constexpr (std::is_same_v<T, constant>) {
        return std::make_unique<node>(constant{alt.payload, alt.return_type});
      } else if constexpr (std::is_same_v<T, comparison>) {
        return std::make_unique<node>(comparison{alt.op,
                                                 substitute_child(alt.left, inner_select_list),
                                                 substitute_child(alt.right, inner_select_list)});
      } else if constexpr (std::is_same_v<T, conjunction>) {
        return std::make_unique<node>(
          conjunction{alt.op, substitute_children(alt.children, inner_select_list)});
      } else if constexpr (std::is_same_v<T, between>) {
        return std::make_unique<node>(between{substitute_child(alt.input, inner_select_list),
                                              substitute_child(alt.lower, inner_select_list),
                                              substitute_child(alt.upper, inner_select_list),
                                              alt.lower_inclusive,
                                              alt.upper_inclusive});
      } else if constexpr (std::is_same_v<T, case_expr>) {
        std::vector<case_expr::when_then> cases;
        cases.reserve(alt.cases.size());
        for (auto const& wt : alt.cases) {
          cases.push_back(case_expr::when_then{substitute_child(wt.when_, inner_select_list),
                                               substitute_child(wt.then_, inner_select_list)});
        }
        return std::make_unique<node>(case_expr{
          std::move(cases), substitute_child(alt.else_, inner_select_list), alt.return_type()});
      } else if constexpr (std::is_same_v<T, cast>) {
        return std::make_unique<node>(
          cast{substitute_child(alt.child, inner_select_list), alt.target_type, alt.try_cast});
      } else if constexpr (std::is_same_v<T, unary_op>) {
        return std::make_unique<node>(
          unary_op{alt.op, substitute_child(alt.child, inner_select_list)});
      } else if constexpr (std::is_same_v<T, coalesce>) {
        return std::make_unique<node>(
          coalesce{substitute_children(alt.children, inner_select_list), alt.return_type()});
      } else if constexpr (std::is_same_v<T, in_list>) {
        return std::make_unique<node>(in_list{substitute_child(alt.probe, inner_select_list),
                                              substitute_children(alt.values, inner_select_list),
                                              alt.negated});
      } else if constexpr (std::is_same_v<T, function_call>) {
        return std::make_unique<node>(
          function_call{alt.function(),
                        substitute_children(alt.arguments(), inner_select_list),
                        alt.return_type()});
      } else if constexpr (std::is_same_v<T, aggregate>) {
        return std::make_unique<node>(
          aggregate{alt.function(),
                    substitute_children(alt.arguments(), inner_select_list),
                    alt.return_type(),
                    alt.distinct()});
      } else {
        static_assert(sizeof(T) == 0,
                      "Unhandled sirius::ast alternative in substitute_references — add a "
                      "substitute arm for the new variant member");
      }
    },
    src.v);
}

}  // namespace sirius::ast
