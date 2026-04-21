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

// sirius
#include <expression_executor/expression_executor_strategy.hpp>

// standard library
#include <cassert>

namespace sirius {

bool string_to_strategy(std::string_view sv, expression_executor_strategy& out)
{
  if (sv == "materialize") {
    out = expression_executor_strategy::MATERIALIZE;
    return true;
  }
  if (sv == "ast_interpret") {
    out = expression_executor_strategy::AST_INTERPRET;
    return true;
  }
  if (sv == "ast_jit") {
    out = expression_executor_strategy::AST_JIT;
    return true;
  }
  return false;
}

std::string_view strategy_to_string(expression_executor_strategy s)
{
  switch (s) {
    case expression_executor_strategy::MATERIALIZE: return "materialize";
    case expression_executor_strategy::AST_INTERPRET: return "ast_interpret";
    case expression_executor_strategy::AST_JIT: return "ast_jit";
  }
  assert(false && "Unrecognized expression_executor_strategy");
  __builtin_unreachable();
}

}  // namespace sirius
