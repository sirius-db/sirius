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
#include <vector>

namespace sirius::ast {

struct node;

/**
 * @brief Sirius-native representation of SQL IN / NOT IN.
 *
 * Mirrors the COMPARE_IN / COMPARE_NOT_IN kinds of
 * duckdb::BoundOperatorExpression. The probe is evaluated once and checked
 * against every value in the list; `negated` flips the result for NOT IN.
 * Split out of the broader `unary_op` node because the operand structure
 * (one probe + N values) is distinct from the other SQL-operator cases.
 */
struct in_list {
  std::unique_ptr<node> probe;
  std::vector<std::unique_ptr<node>> values;
  bool negated{false};

  [[nodiscard]] sirius::logical_type return_type() const noexcept
  {
    return sirius::logical_type::make(sirius::type_id::BOOLEAN);
  }

  std::size_t cudf_ast_op_count() const;
};

}  // namespace sirius::ast
