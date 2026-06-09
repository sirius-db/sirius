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

#include "expression/ast/node.hpp"
#include "helper/logical_type.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_projection.hpp"

#include <cstddef>
#include <memory>

namespace sirius::planner {

//! Returns true when every select-list slot is a non-null reference to column i.
bool is_identity_ast_projection(
  duckdb::vector<std::unique_ptr<sirius::ast::node>> const& select_list);

//! Returns true when both projections have only non-null select-list entries.
bool can_combine_projections(sirius::op::sirius_physical_projection const& outer,
                             sirius::op::sirius_physical_projection const& inner);

//! Compose @p outer over @p inner, re-homing the combined operator on inner's child.
duckdb::unique_ptr<sirius::op::sirius_physical_operator> combine_projections(
  sirius::op::sirius_physical_projection const& outer,
  duckdb::unique_ptr<sirius::op::sirius_physical_projection> inner);

//! Push a projection on @p child, folding with an adjacent child projection when possible.
duckdb::unique_ptr<sirius::op::sirius_physical_operator> push_projection(
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> child,
  duckdb::vector<sirius::logical_type> types,
  duckdb::vector<std::unique_ptr<sirius::ast::node>> select_list,
  std::size_t estimated_cardinality);

//! Recursively fold PROJECTION → PROJECTION chains anywhere in the plan tree.
duckdb::unique_ptr<sirius::op::sirius_physical_operator> fold_adjacent_projections(
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> plan);

}  // namespace sirius::planner
