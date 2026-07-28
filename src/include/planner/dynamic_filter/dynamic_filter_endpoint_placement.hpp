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
 * @file dynamic_filter_endpoint_placement.hpp
 * @brief Pure trace that locates where a join-edge (direct-route) dynamic filter is applied
 */

#pragma once

// duckdb
#include <duckdb/common/enums/join_type.hpp>

// cudf
#include <cudf/types.hpp>

// stdlib
#include <cstddef>
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
 * @brief The operator and output ordinal at which an endpoint is applied
 */
struct endpoint_site {
  sirius::op::sirius_physical_operator* node =
    nullptr;                ///< The deepest operator the filter can wrap
  std::size_t ordinal = 0;  ///< Traced key ordinal in @ref node's output space
};

/**
 * @brief One accepted descent hop: which child to enter and the traced ordinal in that child's
 * space
 */
struct descent_step {
  std::size_t child_index   = 0;  ///< Index of the child to enter
  std::size_t child_ordinal = 0;  ///< Traced key ordinal in the child's output space
};

/**
 * @brief The PROJECTION rule: the input column a pass-through output expression forwards, if any
 *
 *@param[in] expression The projection's output expression for the traced ordinal
 * @return The input column ordinal, or nullopt when the expression is not a plain reference
 */
[[nodiscard]] std::optional<std::size_t> projection_reference_input(
  sirius::ast::node const& expression);

/**
 * @brief The GROUP_BY rule: the input column a grouping-key output forwards, if any
 *
 * @param[in] group_idx Per grouping-key output ordinal, its input column ordinal
 * @param[in] grouping_set_count The aggregate's `grouping_sets.size()`, i.e., the number of
 * grouping sets
 * @param[in] output_ordinal The traced ordinal in the aggregate's output space
 * @return The input column ordinal, or nullopt when the ordinal is not a single-grouping-set key
 */
[[nodiscard]] std::optional<std::size_t> group_by_key_input(std::vector<int> const& group_idx,
                                                            std::size_t grouping_set_count,
                                                            std::size_t output_ordinal);

/**
 * @brief the HASH_JOIN rule: which child and child ordinal a join output forwards, if any
 *
 * @param[in] join_type The producing join's type
 * @param[in] probe_block_output_columns The join's `lhs_output_columns.col_idxs`
 * @param[in] build_block_output_columns The join's `rhs_output_columns.col_idxs`
 * @param[in] output_ordinal The traced ordinal in the join's output space
 * @return The descent hop, or nullopt when this block/type combination is not traceable
 */
[[nodiscard]] std::optional<descent_step> join_block_descent(
  duckdb::JoinType join_type,
  std::vector<cudf::size_type> const& probe_block_output_columns,
  std::vector<cudf::size_type> const& build_block_output_columns,
  std::size_t output_ordinal);

/**
 * @brief One descent hop over a physical operator, dispatching on its kind
 *
 * @param[in] node The operator to descend through
 * @param[in] output_ordinal The traced ordinal in @p node's output space
 * @return The descent hop, or nullopt when @p node refuses (the trace stops)
 */
[[nodiscard]] std::optional<descent_step> pass_through_step(
  sirius::op::sirius_physical_operator const& node, std::size_t output_ordinal);

/**
 * @brief Trace the deepest safe endpoint site for one probe key [`P` is the current join]
 *
 * @param[in] probe_subtree_root P's immediate probe child (P's `children[0]`)
 * @param[in] a0 P's probe-side key ordinal in @p probe_subtree_root's output space
 * @return The endpoint site; never empty
 */
[[nodiscard]] endpoint_site resolve_endpoint_site(
  sirius::op::sirius_physical_operator* probe_subtree_root, std::size_t a0);

}  // namespace sirius::planner