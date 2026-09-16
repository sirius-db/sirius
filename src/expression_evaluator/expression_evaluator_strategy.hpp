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

#include <string_view>

namespace sirius {

/**
 * @brief The expression_evaluator_strategy defines how the expression_evaluator executes.
 * MATERIALIZE: Executes by materializing intermediate results as cudf::columns (every expression
 *              node is a single kernel).
 * AST_INTERPRET: Executes by constructing a cudf::ast::tree and interpreting it with cudf::compute
 *                (monolithic kernel with large switch statement).
 * AST_JIT: Executes by constructing a cudf::ast::tree and compiling it with cudf::compute_jit.
 *
 * @note Not all expression nodes have cuDF AST equivalents ('AST breakers'), so the AST-based
 * strategies build ASTs for as many nodes as they can before encountering AST breakers. The result
 * is a tree whose nodes are AST trees and whose edges are AST breakers.
 */
enum class expression_evaluator_strategy {
  MATERIALIZE,
  AST_INTERPRET,
  AST_JIT,
};

/**
 * @brief Parse a string into an expression_evaluator_strategy.
 * Accepts "materialize", "ast_interpret", "ast_jit".
 * @return true on success; false on unrecognized input.
 */
bool string_to_strategy(std::string_view sv, expression_evaluator_strategy& out);

/**
 * @brief Returns the canonical string representation of an expression_evaluator_strategy.
 */
std::string_view strategy_to_string(expression_evaluator_strategy s);

}  // namespace sirius
