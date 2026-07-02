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
#include "helper/logical_type.hpp"  // sirius::logical_type

// standard library
#include <concepts>
#include <cstddef>
#include <memory>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>

namespace sirius::ast {

// Forward declaration of node as a struct (not a type alias).
// std::variant cannot contain an incomplete type, so `node` must wrap the
// variant inside a struct that CAN be forward-declared. The per-node headers
// store std::unique_ptr<node> with node still incomplete, which is permitted
// for unique_ptr.
struct node;

}  // namespace sirius::ast

// Per-node headers complete each alternative type below. Each header carries
// its own `struct node;` forward declaration so it can hold
// std::unique_ptr<node> children without depending on this file's ordering.
#include "expression/ast/aggregate.hpp"
#include "expression/ast/between.hpp"
#include "expression/ast/case_expr.hpp"
#include "expression/ast/cast.hpp"
#include "expression/ast/coalesce.hpp"
#include "expression/ast/comparison.hpp"
#include "expression/ast/conjunction.hpp"
#include "expression/ast/constant.hpp"
#include "expression/ast/function_call.hpp"
#include "expression/ast/in_list.hpp"
#include "expression/ast/reference.hpp"
#include "expression/ast/unary_op.hpp"

namespace sirius::ast {

/// Compile-time contract: every alternative held in node::variant_t must
/// implement `std::size_t cudf_ast_op_count() const`. Without this concept,
/// a missing method only surfaces as a deep template-instantiation error
/// when node::cudf_ast_op_count()'s std::visit fans out. The static_assert
/// below uses this concept to produce a one-line, named error at the variant
/// declaration site instead.
template <class T>
concept has_cudf_ast_op_count = requires(T const& t) {
  { t.cudf_ast_op_count() } -> std::convertible_to<std::size_t>;
};

/// Compile-time contract: every alternative must expose `return_type()` so the
/// std::visit dispatch in node::return_type() type-checks for every kind.
template <class T>
concept has_return_type = requires(T const& t) {
  { t.return_type() } -> std::convertible_to<sirius::logical_type>;
};

namespace detail {

template <class Variant>
struct all_have_cudf_ast_op_count;

template <class... Ts>
struct all_have_cudf_ast_op_count<std::variant<Ts...>> {
  static constexpr bool value = (has_cudf_ast_op_count<Ts> && ...);
};

template <class Variant>
struct all_have_return_type;

template <class... Ts>
struct all_have_return_type<std::variant<Ts...>> {
  static constexpr bool value = (has_return_type<Ts> && ...);
};

}  // namespace detail

/**
 * @brief Sum type over every Sirius AST node kind.
 *
 * `node` is a struct wrapping `std::variant<...>` rather than a plain type
 * alias. A struct is required because:
 *   1. Recursive children in per-node headers store `std::unique_ptr<node>`,
 *      which needs `node` to be forward-declarable. Type aliases cannot be
 *      forward-declared; structs can.
 *   2. `std::variant<Ts...>` requires every `T` to be complete at the point
 *      the variant is instantiated. Since the variant is defined here — after
 *      all per-node headers are included — every alternative is complete.
 *
 * The alternative order is part of the public ABI: std::variant indexes its
 * alternatives by position, and downstream std::visit dispatch (added by the
 * dual-path executor work, sirius-db/sirius#698) depends on this ordering.
 * Inserting a new alternative MUST happen at the end to preserve the index.
 */
struct node {
  using variant_t = std::variant<reference,
                                 constant,
                                 comparison,
                                 conjunction,
                                 between,
                                 case_expr,
                                 cast,
                                 unary_op,
                                 coalesce,
                                 in_list,
                                 function_call,
                                 aggregate>;

  static_assert(detail::all_have_cudf_ast_op_count<variant_t>::value,
                "Every alternative in sirius::ast::node::variant_t must implement "
                "std::size_t cudf_ast_op_count() const. Add the method to the "
                "missing alternative struct.");

  static_assert(detail::all_have_return_type<variant_t>::value,
                "Every alternative in sirius::ast::node::variant_t must implement "
                "sirius::logical_type return_type() const. Add the method to the "
                "missing alternative struct.");

  variant_t v;

  node() = default;

  /// Construct a node from any alternative by forwarding into the variant.
  /// Excluded from overload resolution when T is `node` itself so the copy /
  /// move special members below are preferred.
  template <class T, std::enable_if_t<!std::is_same_v<std::decay_t<T>, node>, int> = 0>
  node(T&& alt) : v(std::forward<T>(alt))
  {
  }

  // Move-only. Expression trees own their children via std::unique_ptr, so
  // copying is neither supported nor intended.
  node(const node&)                = delete;
  node& operator=(const node&)     = delete;
  node(node&&) noexcept            = default;
  node& operator=(node&&) noexcept = default;

  ~node() = default;

  /// Returns true if this node currently holds an alternative of type T.
  template <class T>
  [[nodiscard]] bool holds() const noexcept
  {
    return std::holds_alternative<T>(v);
  }

  /// Returns a reference to the held alternative of type T.
  /// Throws std::bad_variant_access if the held alternative is a different type.
  template <class T>
  [[nodiscard]] T& get()
  {
    return std::get<T>(v);
  }

  template <class T>
  [[nodiscard]] T const& get() const
  {
    return std::get<T>(v);
  }

  [[nodiscard]] bool is_reference() const noexcept;
  [[nodiscard]] bool is_aggregate() const noexcept;
  [[nodiscard]] bool is_function_call() const noexcept;

  /// Returns the held alternative; throws std::bad_variant_access if the type does not match.
  [[nodiscard]] reference const& as_reference() const;
  [[nodiscard]] aggregate const& as_aggregate() const;
  [[nodiscard]] function_call const& as_function_call() const;

  [[nodiscard]] std::size_t cudf_ast_op_count() const;

  /// The SQL result type of this expression node, recovered natively (no DuckDB
  /// round-trip). Dispatches via std::visit to each alternative's return_type().
  [[nodiscard]] sirius::logical_type return_type() const;
};

/// Strict accessors for operator boundaries: return the held alternative or throw
/// not_implemented_exception (with `context`) when `n` is null or holds a different type.
[[nodiscard]] reference const& require_reference(node const* n, std::string_view context);
[[nodiscard]] aggregate const& require_aggregate(node const* n, std::string_view context);

}  // namespace sirius::ast
