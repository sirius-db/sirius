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

namespace sirius::planner {

/// @brief How many joins and `LogicalGet` nodes received preserved dynamic-filter metadata during
/// a plan copy (surfaced for diagnostic logging and unit-test assertions).
///
/// Produced by `duckdb_join_filter_candidate_adapter::preserve_dynamic_filter_metadata` and
/// returned through `transparent::copy_logical_plan`.
struct preserved_counts {
  std::size_t joins = 0;
  std::size_t gets  = 0;
};

}  // namespace sirius::planner
