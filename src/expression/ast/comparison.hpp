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
#include "expression/join_condition.hpp"  // sirius::comparison_type
#include "helper/logical_type.hpp"        // sirius::logical_type

// standard library
#include <memory>

namespace sirius::ast {

struct node;  // defined in node.hpp; forward declaration is sufficient here.

/**
 * @brief Sirius-native mirror of duckdb::BoundComparisonExpression.
 *
 * Binary comparison with recursive children. `op` reuses sirius::comparison_type
 * already defined for join conditions.
 */
struct comparison {
  sirius::comparison_type op{sirius::comparison_type::equal};
  std::unique_ptr<node> left;
  std::unique_ptr<node> right;

  [[nodiscard]] sirius::logical_type return_type() const noexcept
  {
    return sirius::logical_type::make(sirius::type_id::BOOLEAN);
  }

  std::size_t cudf_ast_op_count() const;
};

}  // namespace sirius::ast
