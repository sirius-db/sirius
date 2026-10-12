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

#include <duckdb/common/enums/optimizer_type.hpp>

#include <array>

namespace sirius::planner {

// Rewrites that produce aggregate or scan shapes unsupported by GPU execution.
inline constexpr std::array unsupported_optimizers{
  duckdb::OptimizerType::PARTIAL_AGGREGATE_PUSHDOWN,
  duckdb::OptimizerType::GROUPING_SETS,
  duckdb::OptimizerType::AGGREGATE_REUSE,
  duckdb::OptimizerType::DISTINCT_AGGREGATE_REWRITE,
  // Evaluate scalar expressions above scans instead of inside DuckDB readers.
  duckdb::OptimizerType::SCALAR_FN_PUSHDOWN};

}  // namespace sirius::planner
