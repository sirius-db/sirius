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
 */
struct case_expr {
  struct when_then {
    std::unique_ptr<node> when_;
    std::unique_ptr<node> then_;
  };

  std::vector<when_then> cases;
  std::unique_ptr<node> else_;
};

}  // namespace sirius::ast
