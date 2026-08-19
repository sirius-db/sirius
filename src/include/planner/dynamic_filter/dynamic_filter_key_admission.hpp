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
 * @brief Classifies a pre-materialization side as a reference, cast reference, or computed key
 *
 * The direct/cast/computed trichotomy must match the key-extraction switch in
 * sirius_physical_hash_join.cpp: a shape admitted here is materialized there.
 */
[[nodiscard]] op::dynamic_filter_key_shape classify_key_side(duckdb::Expression const& key_side);

/**
 * @brief Captures both sides before `materialize_expression_join_keys` rewrites them
 */
[[nodiscard]] std::vector<op::dynamic_filter_condition_shape> classify_join_key_shapes(
  duckdb::vector<duckdb::JoinCondition> const& conditions);

/**
 * @brief Admits statically legal dynamic-filter keys
 *
 * Admission requires equality, non-cast shapes, bound materialized sides, and a supported build
 * type. A composite uniqueness proof does not prove an individual column unique.
 *
 * @throw std::invalid_argument if `condition_shapes` or a non-empty domain vector does not align
 * with @p conditions
 */
[[nodiscard]] std::vector<op::dynamic_filter_publish_plan::admitted_key> admit_dynamic_filter_keys(
  duckdb::vector<sirius::join_condition> const& conditions,
  std::vector<op::dynamic_filter_condition_shape> const& condition_shapes,
  std::vector<std::size_t> const& condition_domain_cardinalities,
  std::optional<std::size_t> build_side_unique_column = std::nullopt);

/**
 * @brief Tests for INNER or SEMI equality over direct, identical INT32 or INT64 storage
 */
[[nodiscard]] bool direct_route_admissible(duckdb::JoinType join_type,
                                           sirius::comparison_type comparison,
                                           op::dynamic_filter_condition_shape shape,
                                           cudf::data_type probe_storage_type,
                                           cudf::data_type build_storage_type) noexcept;

}  // namespace sirius::planner
