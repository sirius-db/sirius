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

#include <cudf/aggregation.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>
#include <vector>

namespace sirius::op {

/**
 * @brief Grouped aggregation for batches with very few groups, such as TPC-H Q1.
 *
 * cuDF's hash group-by makes every thread update the same few shared-memory slots when a batch has
 * only a handful of groups, and its DECIMAL128 sums are compare-and-swap loops. This path instead
 * maps each row's keys to a small group number, accumulates per-warp 64-bit partial sums with
 * shared-memory atomics, and combines them into 128-bit results.
 *
 * It applies only when every key column is a non-null STRING whose values are all exactly one
 * character, there are at most two key columns and at most 64 distinct groups, every aggregate is
 * SUM over a non-null DECIMAL64 or INT64 column, COUNT_VALID over a non-null column, or COUNT_ALL,
 * and the input values are small enough for the 64-bit partial sums. Otherwise it returns nullptr
 * and the caller uses the general path.
 *
 * @return The table `[keys..., aggregates...]` with the same column types the general path
 * produces (DECIMAL64 sums widened to DECIMAL128, counts as INT64), or nullptr when not applicable.
 */
std::unique_ptr<cudf::table> try_low_cardinality_groupby(
  cudf::table_view const& input,
  std::vector<int> const& group_idx,
  std::vector<cudf::aggregation::Kind> const& aggregates,
  std::vector<int> const& aggregate_idx,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

}  // namespace sirius::op
