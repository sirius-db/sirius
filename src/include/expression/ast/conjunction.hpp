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
#include <cstdint>
#include <memory>
#include <vector>

namespace sirius::ast {

struct node;

/**
 * @brief Sirius-native mirror of duckdb::BoundConjunctionExpression.
 *
 * Matches DuckDB's N-ary structure — do not collapse to a binary tree.
 */
struct conjunction {
  /// `invalid` is the default-constructed value so a conjunction whose `op`
  /// was never set is detectable rather than silently behaving as AND.
  /// Real operators carry an `op_` prefix; this avoids C++ keyword collisions
  /// (`and`, `or` are alternative tokens) without resorting to a trailing
  /// underscore on only some values, and keeps every operator visually uniform.
  enum class kind : uint8_t {
    invalid,
    op_and,
    op_or,
  };

  kind op{kind::invalid};
  std::vector<std::unique_ptr<node>> children;

  [[nodiscard]] sirius::logical_type return_type() const noexcept
  {
    return sirius::logical_type::make(sirius::type_id::BOOLEAN);
  }

  std::size_t cudf_ast_op_count() const;
};

}  // namespace sirius::ast
