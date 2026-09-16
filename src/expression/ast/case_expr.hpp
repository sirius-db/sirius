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
 * @brief Sirius-native mirror of duckdb::BoundCaseExpression.
 *
 * Named case_expr (not `case`) because `case` is a C++ keyword. Inner member
 * names that alias keywords (when/then/else) keep a trailing underscore since
 * those are stuck with the keyword-collision constraint.
 *
 * `return_type` is set by the translator at construction time so the executor
 * does not need to re-derive it. Move-only (the when/then/else members hold
 * `unique_ptr<node>`); the return type accessor exposes the recorded type.
 * Moved-from instances are left in the standard valid-but-unspecified state.
 */
class case_expr {
 public:
  struct when_then {
    std::unique_ptr<node> when_;
    std::unique_ptr<node> then_;
  };

  case_expr(std::vector<when_then> cases,
            std::unique_ptr<node> else_clause,
            sirius::logical_type return_type)
    : cases(std::move(cases)), else_(std::move(else_clause)), return_type_(std::move(return_type))
  {
  }

  case_expr()                                = delete;
  case_expr(case_expr const&)                = delete;
  case_expr& operator=(case_expr const&)     = delete;
  case_expr(case_expr&&) noexcept            = default;
  case_expr& operator=(case_expr&&) noexcept = default;
  ~case_expr()                               = default;

  [[nodiscard]] sirius::logical_type const& return_type() const noexcept { return return_type_; }

  std::vector<when_then> cases;
  std::unique_ptr<node> else_;

  std::size_t cudf_ast_op_count() const;

 private:
  sirius::logical_type return_type_;
};

}  // namespace sirius::ast
