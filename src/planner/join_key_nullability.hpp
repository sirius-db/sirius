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

#include <cstddef>

namespace duckdb {
class LogicalComparisonJoin;
}

namespace sirius::planner {

/**
 * @brief Replace null-safe equality with equality when a catalog NOT NULL constraint proves
 *        that NULL cannot match NULL. Returns the number of normalized conditions.
 *
 * Follows only direct column references through projections, filters, ordering, limits and
 * DISTINCT to a native table scan. MARK joins retain their three-valued result semantics.
 * No data statistics are used. Unknown shapes retain the original comparison.
 * Every null-safe equality key must be proven before any condition changes, preserving a
 * uniform null policy for hash equijoins.
 *
 * @pre Types and column bindings have been resolved, and logical children are still owned.
 */
std::size_t normalize_non_null_join_keys(duckdb::LogicalComparisonJoin& join);

}  // namespace sirius::planner
