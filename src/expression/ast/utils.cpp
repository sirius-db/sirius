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
#include "expression/ast/node.hpp"

// standard library
#include <algorithm>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace sirius::ast {

namespace {

// Map a child-owner transform over a vector of child node owners.
template <class Xform>
std::vector<std::unique_ptr<node>> rebuild_children(
  std::vector<std::unique_ptr<node>> const& children, Xform const& xform)
{
  std::vector<std::unique_ptr<node>> out;
  out.reserve(children.size());
  std::ranges::transform(children, std::back_inserter(out), xform);
  return out;
}

// Reconstruct @p src by rebuilding each held alternative, routing every owned
// child through @p xform (which maps a `unique_ptr<node> const&` to a freshly
// built child owner; null slots stay null). Both clone() and
// substitute_references() share this single set of per-alternative arms; they
// differ only in the child transform and in how the `reference` arm is handled
// (clone copies it; substitute_references remaps it before delegating here).
template <class Xform>
std::unique_ptr<node> rebuild(node const& src, Xform const& xform)
{
  return std::visit(
    [&](auto const& alt) -> std::unique_ptr<node> {
      using T = std::decay_t<decltype(alt)>;

      if constexpr (std::is_same_v<T, reference>) {
        return std::make_unique<node>(reference{alt.column_index, alt.return_type});
      } else if constexpr (std::is_same_v<T, constant>) {
        return std::make_unique<node>(constant{alt.payload, alt.return_type});
      } else if constexpr (std::is_same_v<T, comparison>) {
        return std::make_unique<node>(comparison{alt.op, xform(alt.left), xform(alt.right)});
      } else if constexpr (std::is_same_v<T, conjunction>) {
        return std::make_unique<node>(conjunction{alt.op, rebuild_children(alt.children, xform)});
      } else if constexpr (std::is_same_v<T, between>) {
        return std::make_unique<node>(between{xform(alt.input),
                                              xform(alt.lower),
                                              xform(alt.upper),
                                              alt.lower_inclusive,
                                              alt.upper_inclusive});
      } else if constexpr (std::is_same_v<T, case_expr>) {
        std::vector<case_expr::when_then> cases;
        cases.reserve(alt.cases.size());
        for (auto const& wt : alt.cases) {
          cases.push_back(case_expr::when_then{xform(wt.when_), xform(wt.then_)});
        }
        return std::make_unique<node>(
          case_expr{std::move(cases), xform(alt.else_), alt.return_type()});
      } else if constexpr (std::is_same_v<T, cast>) {
        return std::make_unique<node>(cast{xform(alt.child), alt.target_type, alt.try_cast});
      } else if constexpr (std::is_same_v<T, unary_op>) {
        return std::make_unique<node>(unary_op{alt.op, xform(alt.child)});
      } else if constexpr (std::is_same_v<T, coalesce>) {
        return std::make_unique<node>(
          coalesce{rebuild_children(alt.children, xform), alt.return_type()});
      } else if constexpr (std::is_same_v<T, in_list>) {
        return std::make_unique<node>(
          in_list{xform(alt.probe), rebuild_children(alt.values, xform), alt.negated});
      } else if constexpr (std::is_same_v<T, function_call>) {
        return std::make_unique<node>(function_call{
          alt.function(), rebuild_children(alt.arguments(), xform), alt.return_type()});
      } else if constexpr (std::is_same_v<T, aggregate>) {
        return std::make_unique<node>(aggregate{alt.function(),
                                                rebuild_children(alt.arguments(), xform),
                                                alt.return_type(),
                                                alt.distinct()});
      } else {
        static_assert(sizeof(T) == 0,
                      "Unhandled sirius::ast alternative in rebuild(node) — add an arm "
                      "for the new variant member");
      }
    },
    src.v);
}

}  // namespace

std::unique_ptr<node> clone(node const& src)
{
  return rebuild(src, [](std::unique_ptr<node> const& child) -> std::unique_ptr<node> {
    return child ? clone(*child) : nullptr;
  });
}

// Deep-copy @p src, remapping every reference #i to a clone of inner_select_list[i].
// Used when folding adjacent projections: outer references index the inner
// projection's output columns, so each #i is spliced with the inner expression
// that produced column i (whose own references already point at the inner input).
std::unique_ptr<node> substitute_references(
  node const& src, std::vector<std::unique_ptr<node>> const& inner_select_list)
{
  if (src.holds<reference>()) {
    auto const& ref = src.get<reference>();
    // Out-of-range or null inner slots are left unchanged — a null slot is the
    // from_duckdb signal for an unsupported expression and must not be folded through.
    if (ref.column_index >= inner_select_list.size() || !inner_select_list[ref.column_index]) {
      return clone(src);
    }
    // Each outer reference site gets its own clone; duplicate sites duplicate work.
    return clone(*inner_select_list[ref.column_index]);
  }
  return rebuild(src, [&](std::unique_ptr<node> const& child) -> std::unique_ptr<node> {
    return child ? substitute_references(*child, inner_select_list) : nullptr;
  });
}

}  // namespace sirius::ast
