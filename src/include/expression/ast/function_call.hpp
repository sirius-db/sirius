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
#include "expression/function_id.hpp"  // sirius::function_id
#include "helper/logical_type.hpp"     // sirius::logical_type

// standard library
#include <memory>
#include <vector>

namespace sirius::ast {

struct node;

/**
 * @brief Sirius-native mirror of duckdb::BoundFunctionExpression.
 *
 * Arguments are variadic; `return_type` is set by the translator at
 * construction time so the executor does not need to re-derive it.
 */
struct function_call {
  sirius::function_id function{sirius::function_id::add};
  std::vector<std::unique_ptr<node>> arguments;
  sirius::logical_type return_type;
};

}  // namespace sirius::ast
