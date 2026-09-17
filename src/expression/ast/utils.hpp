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

#pragma once

// sirius
#include "expression/ast/node.hpp"  // sirius::ast::node, sirius::ast::reference

// standard library
#include <memory>
#include <type_traits>
#include <vector>

namespace sirius::ast {

template <typename Fn>
void visit_references(node const& root, Fn&& fn);

namespace detail {

template <typename Fn>
void visit_child(std::unique_ptr<node> const& child, Fn& fn)
{
  if (child) { visit_references(*child, fn); }
}

}  // namespace detail

/**
 * @brief Invoke @p fn on every reference node in the tree rooted at @p root
 * (pre-order, read-only).
 *
 * A single reusable traversal primitive so callers (e.g. projection folding's
 * use-count analysis) do not each re-enumerate the variant alternatives.
 */
template <typename Fn>
void visit_references(node const& root, Fn&& fn)
{
  std::visit(
    [&](auto const& alt) {
      using T = std::decay_t<decltype(alt)>;

      if constexpr (std::is_same_v<T, reference>) {
        fn(alt);
      } else if constexpr (std::is_same_v<T, constant>) {
        // leaf, no references
      } else if constexpr (std::is_same_v<T, comparison>) {
        detail::visit_child(alt.left, fn);
        detail::visit_child(alt.right, fn);
      } else if constexpr (std::is_same_v<T, conjunction>) {
        for (auto const& child : alt.children) {
          detail::visit_child(child, fn);
        }
      } else if constexpr (std::is_same_v<T, between>) {
        detail::visit_child(alt.input, fn);
        detail::visit_child(alt.lower, fn);
        detail::visit_child(alt.upper, fn);
      } else if constexpr (std::is_same_v<T, case_expr>) {
        for (auto const& wt : alt.cases) {
          detail::visit_child(wt.when_, fn);
          detail::visit_child(wt.then_, fn);
        }
        detail::visit_child(alt.else_, fn);
      } else if constexpr (std::is_same_v<T, cast>) {
        detail::visit_child(alt.child, fn);
      } else if constexpr (std::is_same_v<T, unary_op>) {
        detail::visit_child(alt.child, fn);
      } else if constexpr (std::is_same_v<T, coalesce>) {
        for (auto const& child : alt.children) {
          detail::visit_child(child, fn);
        }
      } else if constexpr (std::is_same_v<T, in_list>) {
        detail::visit_child(alt.probe, fn);
        for (auto const& child : alt.values) {
          detail::visit_child(child, fn);
        }
      } else if constexpr (std::is_same_v<T, function_call>) {
        for (auto const& child : alt.arguments()) {
          detail::visit_child(child, fn);
        }
      } else if constexpr (std::is_same_v<T, aggregate>) {
        for (auto const& child : alt.arguments()) {
          detail::visit_child(child, fn);
        }
      } else {
        static_assert(sizeof(T) == 0,
                      "Unhandled sirius::ast alternative in visit_references — add a "
                      "traversal arm for the new variant member");
      }
    },
    root.v);
}

/**
 * @brief Deep-clone a Sirius AST node tree.
 *
 * `node` is move-only (children are owned via std::unique_ptr<node>), so it has
 * no copy constructor. clone walks @p src via std::visit, reconstructs each
 * alternative from its public fields/accessors, and recursively deep-clones
 * every child unique_ptr. The returned tree is an independent allocation that
 * shares no ownership with @p src.
 *
 * Used by the aggregate copy path (sirius_physical_ungrouped_aggregate's
 * copy_expressions), where an aggregate node must be duplicated and a
 * to_duckdb round-trip is impossible (to_duckdb(aggregate) throws by design).
 * See https://github.com/sirius-db/sirius/issues/701.
 */
std::unique_ptr<node> clone(node const& src);

/**
 * @brief Replace every column reference in @p expr with a deep clone of the
 * corresponding expression from @p inner_select_list.
 *
 * Used when folding adjacent physical projections: the outer projection's
 * references index into the inner projection's output columns, so composition
 * substitutes each reference with the inner expression tree.
 */
std::unique_ptr<node> substitute_references(
  node const& expr, std::vector<std::unique_ptr<node>> const& inner_select_list);

}  // namespace sirius::ast
