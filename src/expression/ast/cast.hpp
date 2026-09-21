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

namespace sirius::ast {

struct node;

/**
 * @brief Distinguishes value-converting casts from physical carrier restores.
 *
 * `semantic` denotes a logical value conversion, including casts translated from DuckDB.
 * `carrier_restore` denotes a compressed-schema planner operation that restores a narrowed
 * physical carrier to its native type. Only `carrier_restore` may use
 * `sirius::cast_through_rep` during expression evaluation.
 */
enum class cast_kind : uint8_t {
  semantic,        ///< Logical value-converting cast
  carrier_restore  ///< Planner-inserted restoration of a narrowed physical carrier
};

/**
 * @brief Sirius AST cast with explicit conversion provenance.
 *
 * `try_cast` mirrors DuckDB's flag for translated semantic casts. `kind` distinguishes logical
 * value conversion from planner-inserted carrier restoration.
 */
struct cast {
  std::unique_ptr<node> child;
  sirius::logical_type target_type;
  bool try_cast{false};
  cast_kind kind{cast_kind::semantic};

  /// A cast's result type is its target type.
  [[nodiscard]] sirius::logical_type const& return_type() const noexcept { return target_type; }

  std::size_t cudf_ast_op_count() const;
};

}  // namespace sirius::ast
