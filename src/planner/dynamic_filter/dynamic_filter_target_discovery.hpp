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
 * @brief Traces admitted keys to scan or join-edge targets
 */

#pragma once

#include <cudf/types.hpp>

#include <duckdb/common/enums/join_type.hpp>
#include <duckdb/common/unique_ptr.hpp>

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

struct descent_policy {
  bool descend_build_blocks = false;
};

struct descent_step {
  std::size_t child_index   = 0;
  std::size_t child_ordinal = 0;
};

struct route_terminal {
  sirius::op::sirius_physical_operator* node = nullptr;  // Borrowed from the walked subtree.
  std::size_t ordinal                        = 0;
};

/**
 * @brief Result of endpoint splicing; site ordinals follow terminal pre-order
 *
 * Each site ordinal is in the wrapped operator's output schema.
 */
struct endpoint_placement {
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> subtree;
  std::vector<std::size_t> site_ordinals;
};

/**
 * @brief Returns the input column of a plain projection reference
 */
[[nodiscard]] std::optional<std::size_t> projection_reference_input(
  sirius::ast::node const& expression);

/**
 * @brief Returns the input column of a key in a single grouping set
 */
[[nodiscard]] std::optional<std::size_t> group_by_key_input(std::vector<int> const& group_idx,
                                                            std::size_t grouping_set_count,
                                                            std::size_t output_ordinal);

/**
 * @brief Returns the join child and ordinal that safely supplies an output column
 *
 * LEFT/OUTER build-block and RIGHT/OUTER probe-block descent rely on plain-equality admission.
 */
[[nodiscard]] std::optional<descent_step> join_block_descent(
  duckdb::JoinType join_type,
  std::vector<cudf::size_type> const& probe_block_output_columns,
  std::vector<cudf::size_type> const& build_block_output_columns,
  std::size_t output_ordinal,
  descent_policy policy);

/**
 * @brief Returns accepted children in index order; empty means terminal and multiple means fan-out
 */
[[nodiscard]] std::vector<descent_step> descent_steps(
  sirius::op::sirius_physical_operator const& node,
  std::size_t output_ordinal,
  descent_policy policy);

/**
 * @brief Traces terminals in ascending-child pre-order without modifying the tree
 */
[[nodiscard]] std::vector<route_terminal> trace_probe_key(
  sirius::op::sirius_physical_operator& root, std::size_t a0, descent_policy policy);

/**
 * @brief Matches DuckDB's INNER/RIGHT/SEMI producer allowlist
 */
[[nodiscard]] constexpr bool scan_route_join_type_admissible(duckdb::JoinType t) noexcept
{
  return t == duckdb::JoinType::INNER || t == duckdb::JoinType::RIGHT ||
         t == duckdb::JoinType::SEMI;
}

/**
 * @brief Returns a childless endpoint; `place_endpoint()` installs the selected node as its child
 */
using endpoint_factory = std::function<duckdb::unique_ptr<sirius::op::sirius_physical_operator>(
  sirius::op::sirius_physical_operator const& site_node)>;

/**
 * @brief Inserts endpoints above the deepest terminals
 *
 * Existing dynamic-filter endpoints remain transparent to later placements.
 *
 * @pre @p subtree is not null.
 * @pre The key passed direct-route admission and was not scan-bound.
 */
[[nodiscard]] endpoint_placement place_endpoint(
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> subtree,
  std::size_t a0,
  descent_policy policy,
  endpoint_factory const& make_endpoint);

}  // namespace sirius::planner
