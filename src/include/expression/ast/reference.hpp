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
#include <utility>

namespace sirius::ast {

/**
 * @brief Sirius-native mirror of duckdb::BoundReferenceExpression.
 *
 * Carries an index into the input data batch's column vector and the SQL type
 * of the referenced column (mirroring duckdb::BoundReferenceExpression's
 * return_type). The executor resolves column types from the input table_view at
 * runtime and ignores return_type(); it exists so the cuDF-AST translator's
 * decimal-propagation guard can see the type of a referenced decimal column.
 */
struct reference {
  uint32_t column_index{0};

  reference() = default;
  explicit reference(uint32_t column_index, sirius::logical_type return_type = {})
    : column_index(column_index), return_type_(std::move(return_type))
  {
  }

  [[nodiscard]] sirius::logical_type const& return_type() const noexcept { return return_type_; }

  std::size_t cudf_ast_op_count() const;

 private:
  sirius::logical_type return_type_{};
};

}  // namespace sirius::ast
