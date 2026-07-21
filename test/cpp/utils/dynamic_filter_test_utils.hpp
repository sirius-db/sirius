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

#include "op/dynamic_filter/dynamic_filter_stats.hpp"
#include "transparent_execution_test_utils.hpp"

#include <duckdb.hpp>

namespace sirius::test {

/// Snapshot of the connection's `SiriusContext`-owned dynamic-filter counters. Tests assert
/// deltas around a query, and only the deterministic-policy family as equalities -- the
/// opportunistic-delivery family races with probe-side draining and supports directional
/// assertions only (see `op/dynamic_filter/dynamic_filter_stats.hpp`).
inline sirius::op::dynamic_filter_stats_snapshot get_dynamic_filter_stats_snapshot(
  duckdb::Connection& con)
{
  return get_registered_sirius_context(con)->get_dynamic_filter_stats_snapshot();
}

}  // namespace sirius::test
