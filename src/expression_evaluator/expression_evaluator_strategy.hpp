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

#include <array>
#include <cstddef>
#include <string_view>

namespace sirius {

/**
 * @brief The expression_evaluator_strategy defines how the expression_evaluator executes.
 * MATERIALIZE: Materializes intermediate results as cudf::columns. Floating comparisons use
 *              interpreted ASTs for NaN semantics; this strategy never selects AST JIT.
 * AST_INTERPRET: Executes by constructing a cudf::ast::tree and interpreting it with cudf::compute
 *                (monolithic kernel with large switch statement).
 * AST_JIT: Executes by constructing a cudf::ast::tree and compiling it with cudf::compute_jit.
 * NVRTC_JIT: Executes by compiling Sirius expression trees directly into standalone CUDA
 *            __global__ kernels via NVRTC.
 *
 * @note Not all expression nodes have cuDF AST equivalents ('AST breakers'), so the AST-based
 * strategies build ASTs for as many nodes as they can before encountering AST breakers. The result
 * is a tree whose nodes are AST trees and whose edges are AST breakers.
 */
enum class expression_evaluator_strategy {
  MATERIALIZE,
  AST_INTERPRET,
  AST_JIT,
  NVRTC_JIT,
};

/**
 * @brief Returns the canonical string representation of an expression_evaluator_strategy, or an
 * empty view for a value that names no enumerator.
 */
// The switch is exhaustive by construction: leaving out an enumerator is a compile error, so every
// enumerator has a spelling, which `all_expression_evaluator_strategies` relies on below.
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
[[nodiscard]] constexpr std::string_view strategy_to_string(
  expression_evaluator_strategy strategy) noexcept
{
  switch (strategy) {
    case expression_evaluator_strategy::MATERIALIZE: return "materialize";
    case expression_evaluator_strategy::AST_INTERPRET: return "ast_interpret";
    case expression_evaluator_strategy::AST_JIT: return "ast_jit";
    case expression_evaluator_strategy::NVRTC_JIT: return "nvrtc_jit";
  }
  return {};
}
#pragma GCC diagnostic pop

/**
 * @brief Every expression_evaluator_strategy, in declaration order.
 *
 * Code that must cover every strategy, such as `string_to_strategy` and the integration tests,
 * iterates this list. A new enumerator that is not appended here fails to compile:
 * `strategy_to_string` must spell it, and the list must end at the first value it cannot spell.
 */
inline constexpr std::array all_expression_evaluator_strategies{
  expression_evaluator_strategy::MATERIALIZE,
  expression_evaluator_strategy::AST_INTERPRET,
  expression_evaluator_strategy::AST_JIT,
  expression_evaluator_strategy::NVRTC_JIT,
};

static_assert(
  [] {
    for (std::size_t i = 0; i < all_expression_evaluator_strategies.size(); ++i) {
      if (all_expression_evaluator_strategies[i] != static_cast<expression_evaluator_strategy>(i)) {
        return false;
      }
    }
    return strategy_to_string(
             static_cast<expression_evaluator_strategy>(all_expression_evaluator_strategies.size()))
      .empty();
  }(),
  "all_expression_evaluator_strategies must list every enumerator once, in declaration order");

[[nodiscard]] constexpr bool is_cudf_ast_strategy(expression_evaluator_strategy strategy) noexcept
{
  return strategy == expression_evaluator_strategy::AST_INTERPRET ||
         strategy == expression_evaluator_strategy::AST_JIT;
}

/**
 * @brief Parse a string into an expression_evaluator_strategy.
 * Accepts the `strategy_to_string` spelling of each entry of `all_expression_evaluator_strategies`.
 * @return true on success; false on unrecognized input.
 */
bool string_to_strategy(std::string_view sv, expression_evaluator_strategy& out);

}  // namespace sirius
