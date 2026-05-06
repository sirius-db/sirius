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

// standard library
#include <cstdint>
#include <memory>

namespace sirius::ast {

struct node;

/**
 * @brief Sirius-native representation of single-operand SQL operators.
 *
 * Covers the unary members of duckdb::BoundOperatorExpression's kind set:
 * boolean NOT, the null predicates IS NULL / IS NOT NULL, and the error-catch
 * TRY wrapper. All four share the same shape (one child) and only differ in
 * the operator tag, so they collapse into a single AST node.
 *
 * COALESCE and IN / NOT IN are not unary; they live in their own node types
 * (`coalesce`, `in_list`) because their operand structure differs.
 */
struct unary_op {
  /// `invalid` is the default-constructed value so a unary_op whose `op` was
  /// never set is detectable rather than silently behaving as IS NULL.
  /// Real operators carry an `op_` prefix; this avoids C++ keyword collisions
  /// (`not`, `try` are alternative tokens) without resorting to a trailing
  /// underscore on only some values, and keeps every operator visually uniform.
  enum class kind : uint8_t {
    invalid,
    op_not,
    op_is_null,
    op_is_not_null,
    op_try,
  };

  kind op{kind::invalid};
  std::unique_ptr<node> child;
};

}  // namespace sirius::ast
