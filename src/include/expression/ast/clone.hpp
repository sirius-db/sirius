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

namespace sirius::ast {

/**
 * @brief Deep-clone a Sirius AST node tree.
 *
 * `node` is move-only (children are owned via std::unique_ptr<node>), so it has
 * no copy constructor. clone walks @p src via std::visit, reconstructs each
 * alternative from its public fields/accessors, and recursively deep-clones
 * every child unique_ptr. The returned tree is an independent allocation that
 * shares no ownership with @p src.
 *
 * Used by the aggregate copy path (sirius_physical_ungrouped_aggregate's
 * copy_expressions), where an aggregate node must be duplicated and a
 * to_duckdb round-trip is impossible (to_duckdb(aggregate) throws by design).
 * See https://github.com/sirius-db/sirius/issues/701.
 */
std::unique_ptr<node> clone(node const& src);

}  // namespace sirius::ast
