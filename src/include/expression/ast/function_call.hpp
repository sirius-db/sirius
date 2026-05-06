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

namespace detail {
/// Placeholder function identifier.
/// Phase 3 replaces this with sirius::function_id.
enum class function_placeholder : uint16_t {
  unknown = 0,
};
}  // namespace detail

/**
 * @brief Sirius-native mirror of duckdb::BoundFunctionExpression.
 *
 * Arguments are variadic; `return_type` is set by the translator at
 * construction time so the executor does not need to re-derive it.
 */
struct function_call {
  detail::function_placeholder function{detail::function_placeholder::unknown};
  std::vector<std::unique_ptr<node>> arguments;
  sirius::logical_type return_type;
};

}  // namespace sirius::ast
