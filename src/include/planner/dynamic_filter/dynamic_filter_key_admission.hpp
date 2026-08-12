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
 * @file dynamic_filter_key_admission.hpp
 * @brief Converts join conditions into statically admitted dynamic-filter keys
 *
 * `admit_dynamic_filter_keys()` decides key legality only; where each admitted key lands is the
 * discovery walk's output (`dynamic_filter_target_discovery.hpp`), not admission's. The admitted
 * keys carry three coordinate spaces that are not interchangeable: the original join-condition
 * index (provenance), the runtime build-table ordinal, and the producing join's probe-child
 * ordinal -- the entry ordinal every use of an admitted key starts a trace from. The dense
 * admitted-key index is the position in the returned vector.
 */

#pragma once

#include "duckdb/common/enums/join_type.hpp"
#include "expression/join_condition.hpp"
#include "op/dynamic_filter/dynamic_filter_publish_plan.hpp"

#include <cudf/types.hpp>

#include <cstddef>
#include <optional>
#include <vector>

namespace duckdb {
class Expression;
}  // namespace duckdb

namespace sirius::planner {

/**
 * @brief Classify one join-condition side as a direct reference, a cast of one, or computed
 *
 * The trichotomy matches the physical hash join's key extraction: `BOUND_REF` is direct,
 * `BOUND_CAST` wrapping a `BOUND_REF` is cast, and anything else -- including a cast of a computed
 * expression -- is computed.
 *
 * @param[in] key_side One side of a join condition, before computed-key materialization
 */
[[nodiscard]] op::dynamic_filter_key_shape classify_key_side(duckdb::Expression const& key_side);

/**
 * @brief Classify both sides of every join condition before computed-key materialization
 *
 * Must be called before `materialize_expression_join_keys` rewrites computed equality sides into
 * plain bound references -- afterwards the shapes are unrecoverable from the conditions alone.
 *
 * @param[in] conditions The join's conditions in original planner order
 * @return One shape per condition, index-aligned with `conditions`
 */
[[nodiscard]] std::vector<op::dynamic_filter_condition_shape> classify_join_key_shapes(
  duckdb::vector<duckdb::JoinCondition> const& conditions);

/**
 * @brief Admit the statically legal build keys of one producing hash join
 *
 * A key is admitted when its condition uses `sirius::comparison_type::equal`, neither carried side
 * shape is a cast, both materialized sides are bound references, and cuDF can represent the build
 * type. Computed keys remain eligible after materialization.
 *
 * @throw std::invalid_argument if `condition_shapes` or a non-empty
 * `condition_domain_cardinalities` is not aligned one-to-one with `conditions`
 *
 * @param[in] conditions The wrapped join conditions in original planner order, post-materialization
 * @param[in] condition_shapes Carried pre-materialization classification, aligned with `conditions`
 * @param[in] condition_domain_cardinalities Per condition index, the base-table row bound used as
 * the coverage denominator (0 = unknown); empty when no evidence exists
 * @param[in] build_side_unique_column The build child's sole proven-unique output ordinal; empty
 * unless exactly one column is proven unique -- a composite uniqueness proof bounds distinct
 * tuples, not one column's values, and must not arm any key's coverage gate. An admitted key whose
 * build ordinal equals this value is marked `build_key_proven_unique`.
 * @return The admitted keys, in admitted (planner) order; a key's position is the admitted-key
 * index `key_binding::admitted_key_index` refers to
 */
[[nodiscard]] std::vector<op::dynamic_filter_publish_plan::admitted_key> admit_dynamic_filter_keys(
  duckdb::vector<sirius::join_condition> const& conditions,
  std::vector<op::dynamic_filter_condition_shape> const& condition_shapes,
  std::vector<std::size_t> const& condition_domain_cardinalities,
  std::optional<std::size_t> build_side_unique_column = std::nullopt);

/**
 * @brief Static legality of one admitted key for a join-edge (direct) endpoint
 *
 * A direct route requires an INNER or SEMI join, equality comparison, direct references on both
 * sides, and identical INT32 or INT64 storage types.
 *
 * @param[in] shape The condition's carried pre-materialization side shapes
 * @return True when `place_endpoint()` may place the key in the producing join's probe subtree
 */
[[nodiscard]] bool direct_route_admissible(duckdb::JoinType join_type,
                                           sirius::comparison_type comparison,
                                           op::dynamic_filter_condition_shape shape,
                                           cudf::data_type probe_storage_type,
                                           cudf::data_type build_storage_type) noexcept;

}  // namespace sirius::planner
