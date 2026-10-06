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
#include "expression/ast/node.hpp"  // sirius::ast::node + 11 alternatives

// standard library
#include <memory>

namespace duckdb {
class Expression;
}  // namespace duckdb

namespace sirius::ast {

/**
 * @brief Reconstruct a DuckDB bound expression tree from a Sirius AST tree.
 *
 * Recursively walks @p expr, dispatching on its variant alternative via
 * std::visit to the matching per-alternative overload. Returns a freshly-
 * allocated duckdb::Expression tree that the caller owns. The output is
 * structurally equivalent to the input that sirius::ast::from_duckdb would
 * have produced the AST from; round-tripping a tree through from_duckdb ->
 * to_duckdb -> executor yields outputs identical to executing the original
 * DuckDB expression.
 *
 * @throws duckdb::InternalException if a function_call carries a function_id
 *         enum value with no DuckDB name mapping (should be impossible — every
 *         enum value is registered in sirius::to_duckdb_function_name).
 */
std::unique_ptr<duckdb::Expression> to_duckdb(node const& expr);

// Per-alternative overloads — called directly by expression_evaluator shims
// to sidestep node's move-only constraint. The top-level to_duckdb(node) above
// dispatches to these via std::visit.
std::unique_ptr<duckdb::Expression> to_duckdb(aggregate const& alt);
std::unique_ptr<duckdb::Expression> to_duckdb(between const& alt);
std::unique_ptr<duckdb::Expression> to_duckdb(case_expr const& alt);
std::unique_ptr<duckdb::Expression> to_duckdb(cast const& alt);
std::unique_ptr<duckdb::Expression> to_duckdb(coalesce const& alt);
std::unique_ptr<duckdb::Expression> to_duckdb(comparison const& alt);
std::unique_ptr<duckdb::Expression> to_duckdb(conjunction const& alt);
std::unique_ptr<duckdb::Expression> to_duckdb(constant const& alt);
std::unique_ptr<duckdb::Expression> to_duckdb(function_call const& alt);
std::unique_ptr<duckdb::Expression> to_duckdb(in_list const& alt);
std::unique_ptr<duckdb::Expression> to_duckdb(reference const& alt);
std::unique_ptr<duckdb::Expression> to_duckdb(unary_op const& alt);

}  // namespace sirius::ast
