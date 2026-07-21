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
#include <memory>

namespace sirius::ast {

struct node;

/**
 * @brief Sirius-native mirror of duckdb::BoundCastExpression.
 *
 * `try_cast == true` corresponds to DuckDB's TRY_CAST (null on overflow/parse
 * failure instead of throwing).
 */
struct cast {
  std::unique_ptr<node> child;
  sirius::logical_type target_type;
  bool try_cast{false};

  /// A cast's result type is its target type.
  [[nodiscard]] sirius::logical_type const& return_type() const noexcept { return target_type; }

  std::size_t cudf_ast_op_count() const;
};

}  // namespace sirius::ast
