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

#include <duckdb/common/unique_ptr.hpp>
#include <duckdb/common/vector.hpp>

#include <memory>
#include <vector>

namespace duckdb {
class Expression;
}

namespace sirius {

/**
 * @brief Opaque, move-only wrapper around a Sirius AST node.
 *
 * Lets Super Sirius operator headers, the expression-executor public header, and the
 * physical-plan-generator header avoid including any duckdb/planner/expression/... header.
 *
 * Internally stores a std::unique_ptr<impl>, where `impl` is forward-declared here and
 * defined in expression_internal.hpp. wrap() translates an incoming duckdb::Expression
 * into a sirius::ast::node at the boundary; code that needs to reach the underlying node
 * includes expression_internal.hpp and uses the free `unwrap` / `release` helpers; no
 * friend declarations are required. See https://github.com/sirius-db/sirius/issues/701.
 *
 * Semantics mirror std::unique_ptr<sirius::ast::node>: unique ownership, move-only.
 */
class expression {
 public:
  struct impl;

  /// Constructs a null expression.
  expression() noexcept;

  /// Takes ownership of an impl instance. Usable only at translation units that can form a
  /// complete `impl` (i.e., those including expression_internal.hpp).
  explicit expression(std::unique_ptr<impl> p) noexcept;

  ~expression();

  expression(expression&&) noexcept;
  expression& operator=(expression&&) noexcept;

  expression(const expression&)            = delete;
  expression& operator=(const expression&) = delete;

  /// Returns true if this wrapper does not hold a Sirius AST node.
  [[nodiscard]] bool is_null() const noexcept { return !_impl; }

  /// Returns true if this wrapper holds a Sirius AST node.
  explicit operator bool() const noexcept { return static_cast<bool>(_impl); }

  /**
   * @brief Opaque access to the internal impl.
   *
   * The returned pointer is only useful in translation units that also include
   * expression_internal.hpp (which completes the `impl` type). Everywhere else, the
   * returned pointer is inert.
   */
  [[nodiscard]] impl* get_impl() noexcept { return _impl.get(); }
  [[nodiscard]] impl const* get_impl() const noexcept { return _impl.get(); }

 private:
  std::unique_ptr<impl> _impl;
};

//===----------------------------------------------------------------------===//
// Boundary factories
//===----------------------------------------------------------------------===//

/**
 * @brief Translates a duckdb::Expression unique_ptr into a sirius::expression.
 *
 * This is the DuckDB->Sirius translation boundary: the incoming expression is lowered
 * to a sirius::ast::node via sirius::ast::from_duckdb and the resulting node is stored.
 *
 * @param expr  The DuckDB expression to translate. May be null; a null input — or an
 *              unsupported shape that from_duckdb declines to translate — produces a null
 *              sirius::expression.
 * @return The wrapped sirius::expression.
 */
expression wrap(std::unique_ptr<duckdb::Expression> expr);

/**
 * @brief Wraps each element of a DuckDB expression vector.
 *
 * @param exprs  The DuckDB expressions to wrap. The input vector is drained.
 * @return A vector of sirius::expression, preserving size and order.
 */
std::vector<expression> wrap_many(std::vector<std::unique_ptr<duckdb::Expression>> exprs);

/**
 * @brief duckdb::vector overload of wrap_many.
 *
 * Matches the container convention used by Super Sirius operator headers, which hold
 * `duckdb::vector<sirius::expression>` (not `std::vector<...>`). Semantics identical to the
 * std::vector overload.
 */
duckdb::vector<expression> wrap_many(duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> exprs);

}  // namespace sirius
