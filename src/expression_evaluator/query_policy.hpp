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
#include "config.hpp"

#include <optional>
namespace sirius {
struct expression_query_policy {
  expression_evaluator_strategy strategy{expression_evaluator_strategy::AST_INTERPRET};
  bool regex_jit{true};
};
// GPU workers install the immutable query policy for the complete task, including nested
// evaluators. Scoped restoration is required because a worker serves different queries.
inline thread_local std::optional<expression_query_policy> active_expression_policy;
class scoped_expression_policy {
 public:
  explicit scoped_expression_policy(expression_query_policy policy)
    : previous(active_expression_policy)
  {
    active_expression_policy = policy;
  }
  ~scoped_expression_policy() { active_expression_policy = previous; }
  scoped_expression_policy(const scoped_expression_policy&) = delete;

 private:
  std::optional<expression_query_policy> previous;
};
inline bool query_regex_jit_enabled()
{
  return active_expression_policy ? active_expression_policy->regex_jit
                                  : duckdb::Config::ENABLE_REGEX_JIT_IMPL;
}
}  // namespace sirius
