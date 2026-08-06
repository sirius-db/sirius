/*
 * Copyright 2026, Sirius Contributors.
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

/**
 * @file dynamic_filter_target_discovery.hpp
 * @brief Discovers one producing join's dynamic-filter targets in its built probe subtree
 *
 * `trace_probe_key()` follows one admitted probe key through the physical probe subtree.
 * `TABLE_SCAN` terminals are scan-binding sites; other terminals can be wrapped by
 * `place_endpoint()`. Both operations use @ref descent_steps. Each accepted step remaps the key
 * ordinal into its child's output space.
 */

#pragma once

// duckdb
#include <duckdb/common/enums/join_type.hpp>
#include <duckdb/common/unique_ptr.hpp>

// cudf
#include <cudf/types.hpp>

// stdlib
#include <cstddef>
#include <functional>
#include <optional>
#include <vector>

namespace sirius::ast {
struct node;
}  // namespace sirius::ast

namespace sirius::op {
class sirius_physical_operator;
}  // namespace sirius::op

namespace sirius::planner {

/**
 * @brief Controls whether a dynamic-filter trace may enter join build blocks
 */
struct descent_policy {
  /// Whether the trace may enter an intervening join's build block
  bool descend_build_blocks = false;
};

/**
 * @brief One accepted trace step into an operator child
 */
struct descent_step {
  std::size_t child_index   = 0;  ///< Index of the child to enter
  std::size_t child_ordinal = 0;  ///< Traced key ordinal in the child's output space
};

/**
 * @brief Where one branch of a key's trace bottomed out
 */
struct route_terminal {
  sirius::op::sirius_physical_operator* node = nullptr;  ///< Borrowed from the walked subtree
  std::size_t ordinal                        = 0;        ///< In @c node's output space
};

/**
 * @brief Result of splicing endpoints into a subtree
 *
 * `subtree` owns the complete rewrapped subtree. `site_ordinals` holds one entry per spliced site,
 * in the same deterministic pre-order (ascending child index) the walk visits terminals, so a
 * caller minting one channel per `endpoint_factory` invocation can zip its mints with these
 * ordinals. Every plan without a physical set operation has exactly one site.
 */
struct endpoint_placement {
  duckdb::unique_ptr<sirius::op::sirius_physical_operator>
    subtree;  ///< Rewrapped subtree containing the endpoint(s)
  std::vector<std::size_t>
    site_ordinals;  ///< Traced ordinal per site, in the sited operator's output space
};

/**
 * @brief Resolve the input column forwarded by a projection expression
 *
 * @param[in] expression Projection expression for the traced output column
 * @return The referenced input column, or `std::nullopt` when `expression` is not a plain reference
 */
[[nodiscard]] std::optional<std::size_t> projection_reference_input(
  sirius::ast::node const& expression);

/**
 * @brief Resolve the input column forwarded by a group-by output
 *
 * @param[in] group_idx Input column for each grouping-key output
 * @param[in] grouping_set_count Number of grouping sets
 * @param[in] output_ordinal The traced ordinal in the aggregate's output space
 * @return The input column, or `std::nullopt` when the ordinal is not a key of a single grouping
 * set
 */
[[nodiscard]] std::optional<std::size_t> group_by_key_input(std::vector<int> const& group_idx,
                                                            std::size_t grouping_set_count,
                                                            std::size_t output_ordinal);

/**
 * @brief Resolve the join child that supplies one output column
 *
 * The probe block is pure mechanism; the build block is additionally gated on
 * `descent_policy::descend_build_blocks`, because a build-block hop is the SIP capability rather
 * than a property of the join's output layout.
 *
 * @param[in] join_type Join type whose output is being traced
 * @param[in] probe_block_output_columns Probe-child columns emitted by the join
 * @param[in] build_block_output_columns Build-child columns emitted by the join
 * @param[in] output_ordinal The traced ordinal in the join's output space
 * @param[in] policy Which hops the walking producer may take
 * @return The child and its corresponding ordinal, or `std::nullopt` when the join output cannot be
 * traced safely
 */
[[nodiscard]] std::optional<descent_step> join_block_descent(
  duckdb::JoinType join_type,
  std::vector<cudf::size_type> const& probe_block_output_columns,
  std::vector<cudf::size_type> const& build_block_output_columns,
  std::size_t output_ordinal,
  descent_policy policy);

/**
 * @brief Resolve the trace steps through one physical operator
 *
 * Zero steps mean @p node is a terminal for the traced key; one step is a pass-through; more than
 * one is a set-operation fan-out (a physical UNION's output is positionally aligned with every
 * child, so each step carries the same ordinal). Steps are ordered by ascending child index.
 *
 * @param[in] node The operator to descend through
 * @param[in] output_ordinal The traced ordinal in @p node's output space
 * @param[in] policy Which hops the walking producer may take
 * @return The accepted child steps; empty when the trace terminates at @p node
 */
[[nodiscard]] std::vector<descent_step> descent_steps(
  sirius::op::sirius_physical_operator const& node,
  std::size_t output_ordinal,
  descent_policy policy);

/**
 * @brief Pure classification of where one key's trace bottoms out
 *
 * Follows `descent_steps()` pre-order with ascending child index and returns one terminal per
 * reached branch; no plan without a physical set operation produces more than one. The walk
 * mutates nothing and transfers no ownership; terminal node pointers are borrowed from the walked
 * subtree.
 *
 * @param[in] root The probe subtree to trace
 * @param[in] a0 The traced key ordinal in @p root's output space (the producing join's probe-child
 * entry ordinal)
 * @param[in] policy Which hops this producer may take
 * @return The terminal node and exit ordinal of every reached branch
 */
[[nodiscard]] std::vector<route_terminal> trace_probe_key(
  sirius::op::sirius_physical_operator& root, std::size_t a0, descent_policy policy);

/**
 * @brief Whether a producing join of type @p t may bind a key INTO a probe-side scan
 *
 * Mirrors the early-return gate of DuckDB's `JoinFilterPushdownOptimizer::GenerateJoinFilters`,
 * which generates scan filters only for INNER, RIGHT, and SEMI producers. The other types preserve
 * or negate unmatched probe rows, so pre-filtering the probe by build-key membership would change
 * results. RIGHT is sound even though it is not direct-route admissible: a RIGHT join keeps every
 * BUILD row and drops probe rows that match nothing, so removing probe rows whose key is outside
 * the build-key set removes only rows the join was going to drop -- the NULL-padded rows a RIGHT
 * join fabricates come from unmatched BUILD rows, which a probe-side filter cannot touch.
 */
[[nodiscard]] constexpr bool scan_route_join_type_admissible(duckdb::JoinType t) noexcept
{
  return t == duckdb::JoinType::INNER || t == duckdb::JoinType::RIGHT ||
         t == duckdb::JoinType::SEMI;
}

/**
 * @brief Factory for a childless endpoint that matches the operator it will wrap
 *
 * `place_endpoint()` passes the selected operator to the factory, then installs that operator as
 * the returned endpoint's child. The factory is invoked once per spliced site, in the same
 * deterministic order `trace_probe_key()` reports terminals.
 */
using endpoint_factory = std::function<duckdb::unique_ptr<sirius::op::sirius_physical_operator>(
  sirius::op::sirius_physical_operator const& site_node)>;

/**
 * @brief Place an endpoint at the deepest safe site of every reached branch for one probe key
 *
 * Starting with @p a0 in @p subtree's output space, the function follows `descent_steps()` until
 * an operator refuses the trace, then inserts an endpoint immediately above that operator. If the
 * root refuses, the endpoint wraps the root. A fan-out step splices one endpoint per reached
 * branch. The factory has no scan knowledge, so a caller that scan-binds any branch of a key must
 * not call this for that key. On mixed branches, unbound branches therefore receive no direct
 * endpoint.
 *
 * Existing `sirius_physical_dynamic_filter` operators are transparent to the trace, so placing one
 * key does not prevent later keys from reaching the same depth.
 *
 * @pre The producing join compares the key with `sirius::comparison_type::equal`;
 * `direct_route_admissible()` enforces this before placement.
 * @pre @p subtree is not null.
 *
 * @param[in] subtree Probe subtree whose ownership transfers to the result
 * @param[in] a0 The traced key ordinal in @p subtree's output space
 * @param[in] policy Which hops this producer may take
 * @param[in] make_endpoint Factory for the endpoint operator(s)
 * @return The rewrapped subtree and the traced ordinal at each spliced site
 */
[[nodiscard]] endpoint_placement place_endpoint(
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> subtree,
  std::size_t a0,
  descent_policy policy,
  endpoint_factory const& make_endpoint);

}  // namespace sirius::planner
