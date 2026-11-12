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

#include "op/sirius_physical_hash_join.hpp"
#include "sirius_config.hpp"

namespace sirius::planner {

/**
 * @brief Optional metadata supplied by a hash join's planner caller
 *
 * Empty defaults preserve all input columns, add no delimiter columns, and publish no dynamic
 * filters or statistics. Admission and join-specific policy remain with the caller.
 */
struct hash_join_options {
  duckdb::vector<std::size_t> left_projection_map;
  duckdb::vector<std::size_t> right_projection_map;
  duckdb::vector<sirius::logical_type> delim_types;
  sirius::op::dynamic_filter_publish_plan dynamic_filter_plan;
  sirius::op::dynamic_filter_stats* dynamic_filter_stats_sink = nullptr;
};

/**
 * @brief Construct a hash join with the common settings from one operator-parameter snapshot
 *
 * Takes ownership of both inputs, conditions, and optional publication metadata. Comparison-join
 * optimization and scalar-subquery error policy are applied separately by the comparison-join
 * planner.
 */
duckdb::unique_ptr<sirius::op::sirius_physical_hash_join> make_hash_join(
  duckdb::LogicalOperator& op,
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> left,
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> right,
  duckdb::vector<sirius::join_condition> conditions,
  duckdb::JoinType join_type,
  sirius::operator_params const& params,
  hash_join_options options = {});

}  // namespace sirius::planner
