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

#include <cstddef>
#include <memory>

namespace sirius::planner {

/**
 * @brief Utilities for creating and folding adjacent physical projections.
 *
 * When two projections stack as PROJECTION(outer) → PROJECTION(inner) → …, folding
 * composes their select lists into a single projection over inner's child. In that
 * layout, **outer** is the parent (closer to the query root) and **inner** is its
 * immediate child. Outer select-list references (#i) index into inner's output
 * columns; folding substitutes each reference with a clone of inner.select_list[i].
 */

/**
 * @brief Wrap @p child in a projection and fold with an adjacent child projection when safe.
 *
 * @param child Subtree that becomes the new projection's (or folded projection's) input.
 * @param types Output column types of the projection.
 * @param select_list Expressions evaluated to produce @p types; may reference @p child's columns.
 * @param estimated_cardinality Cardinality estimate carried on the projection operator.
 *
 * If @p select_list is an identity passthrough ([#0, #1, …, #n-1]) and @p types matches
 * @p child->types, no projection is created and @p child is returned unchanged.
 *
 * Otherwise a projection is appended on @p child. If that child is already a projection
 * and folding is safe, the two are merged into one projection whose select list is the
 * composition of outer over inner.
 *
 * Folding refuses when either select list contains a null slot (unsupported-expression
 * fallback from from_duckdb) or when an inner expression would be duplicated at multiple
 * outer reference sites without being trivial (bare reference or constant).
 */
duckdb::unique_ptr<sirius::op::sirius_physical_operator> push_projection(
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> child,
  duckdb::vector<sirius::logical_type> types,
  duckdb::vector<std::unique_ptr<sirius::ast::node>> select_list,
  std::size_t estimated_cardinality);

/**
 * @brief Recursively fold PROJECTION → PROJECTION chains anywhere in a finished plan.
 *
 * @param plan Root of the physical operator tree to optimize in place.
 * @return The same tree with adjacent projection stacks collapsed where safe.
 *
 * Visits children first, then attempts to fold a projection at the current node with
 * its child projection. This post-pass catches stacks that push_projection missed —
 * for example projections sitting under joins or built in separate plan-builder steps.
 *
 * Does not fold across non-projection operators; only immediate parent/child projection
 * pairs are candidates. Identity passthrough projections whose child already exposes the
 * same output schema are elided.
 */
duckdb::unique_ptr<sirius::op::sirius_physical_operator> fold_adjacent_projections(
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> plan);

}  // namespace sirius::planner
