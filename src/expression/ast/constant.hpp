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
#include "expression/value.hpp"     // sirius::value
#include "helper/logical_type.hpp"  // sirius::logical_type

// standard library
#include <utility>

namespace sirius::ast {

/**
 * @brief Sirius-native mirror of duckdb::BoundConstantExpression.
 *
 * Carries a typed literal. `payload` is the typed sum-type holding the SQL
 * value; `return_type()` carries DECIMAL precision and the SQL type of typed
 * NULLs. Mirrors the shape of cast::target_type and function_call::return_type().
 */
struct constant {
  sirius::value payload;

  constant() = default;
  constant(sirius::value payload, sirius::logical_type return_type)
    : payload(std::move(payload)), return_type_(std::move(return_type))
  {
  }

  [[nodiscard]] sirius::logical_type const& return_type() const noexcept { return return_type_; }

  std::size_t cudf_ast_op_count() const;

 private:
  sirius::logical_type return_type_;
};

}  // namespace sirius::ast
