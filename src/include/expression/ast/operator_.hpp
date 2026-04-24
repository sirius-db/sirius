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
#include <vector>

namespace sirius::ast {

struct node;

/**
 * @brief Sirius-native mirror of duckdb::BoundOperatorExpression.
 *
 * Mixed-arity: is_null / is_not_null / not_ are unary, coalesce is N-ary,
 * in / not_in take a value plus a list. Arity is implied by the enum tag;
 * `children` carries all operands.
 *
 * Trailing underscore on `operator_` because `operator` is a C++ keyword.
 */
struct operator_ {
  enum class kind : uint8_t {
    is_null,
    is_not_null,
    not_,
    coalesce,
    try_,
    in,
    not_in,
  };

  kind op{kind::is_null};
  std::vector<std::unique_ptr<node>> children;
};

}  // namespace sirius::ast
