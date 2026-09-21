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
 * @brief Sirius-native representation of SQL COALESCE.
 *
 * N-ary expression that returns the first non-null argument. Mirrors the
 * OPERATOR_COALESCE kind of duckdb::BoundOperatorExpression. Split out of
 * the broader `unary_op` node because coalesce is the only non-unary member
 * of that family.
 *
 * `return_type` is set by the translator at construction time so the executor
 * does not need to re-derive it. Move-only (children hold `unique_ptr<node>`);
 * the return type accessor exposes the recorded type. Moved-from instances are
 * left in the standard valid-but-unspecified state.
 */
class coalesce {
 public:
  coalesce(std::vector<std::unique_ptr<node>> children, sirius::logical_type return_type)
    : children(std::move(children)), return_type_(std::move(return_type))
  {
  }

  coalesce()                               = delete;
  coalesce(coalesce const&)                = delete;
  coalesce& operator=(coalesce const&)     = delete;
  coalesce(coalesce&&) noexcept            = default;
  coalesce& operator=(coalesce&&) noexcept = default;
  ~coalesce()                              = default;

  [[nodiscard]] sirius::logical_type const& return_type() const noexcept { return return_type_; }

  std::vector<std::unique_ptr<node>> children;

  std::size_t cudf_ast_op_count() const;

 private:
  sirius::logical_type return_type_;
};

}  // namespace sirius::ast
