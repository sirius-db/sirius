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
#include "expression/ast/node.hpp"  // sirius::ast::node

// standard library
#include <memory>

namespace duckdb {
class Expression;
}  // namespace duckdb

namespace sirius::ast {

/**
 * @brief Translate a DuckDB bound expression tree into a Sirius AST tree.
 *
 * Recursively walks @p expr, dispatching on its ExpressionClass to the matching
 * Sirius node constructor. Returns nullptr when the expression (or any of its
 * children) carries an unsupported subtype the executor cannot lower; the
 * caller is expected to interpret nullptr as a fallback signal.
 *
 * @throws duckdb::InternalException on an ExpressionClass that the Sirius
 *         translator has never seen.
 * @throws sirius::not_implemented_exception if any embedded duckdb::Value or
 *         duckdb::LogicalType carries a type unsupported by the upstream
 *         sirius::from_duckdb(...) helpers. This can fire from
 *         BoundConstantExpression (Value translation), BoundCastExpression
 *         (target type), or BoundFunctionExpression (return type).
 */
std::unique_ptr<node> from_duckdb(duckdb::Expression const& expr);

}  // namespace sirius::ast
