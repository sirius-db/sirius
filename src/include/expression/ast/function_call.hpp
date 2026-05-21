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
 *
 * No default state: the only way to construct a `function_call` is via the
 * three-argument constructor, so every instance was built with an explicit
 * function id, an arguments vector, and a return type. Move-only (the
 * arguments vector holds `unique_ptr<node>`); private fields with const
 * accessors prevent post-construction mutation. Moved-from instances are
 * left in the standard valid-but-unspecified state and must not be read.
 */
class function_call {
 public:
  function_call(sirius::function_id id,
                std::vector<std::unique_ptr<node>> arguments,
                sirius::logical_type return_type)
      : function_(id),
        arguments_(std::move(arguments)),
        return_type_(std::move(return_type))
  {
  }

  function_call()                                    = delete;
  function_call(function_call const&)                = delete;
  function_call& operator=(function_call const&)     = delete;
  function_call(function_call&&) noexcept            = default;
  function_call& operator=(function_call&&) noexcept = default;
  ~function_call()                                   = default;

  [[nodiscard]] sirius::function_id function() const noexcept { return function_; }
  [[nodiscard]] std::vector<std::unique_ptr<node>> const& arguments() const noexcept
  {
    return arguments_;
  }
  [[nodiscard]] sirius::logical_type const& return_type() const noexcept { return return_type_; }

 private:
  sirius::function_id function_;
  std::vector<std::unique_ptr<node>> arguments_;
  sirius::logical_type return_type_;
};

}  // namespace sirius::ast
