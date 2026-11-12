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

#include "duckdb/common/unique_ptr.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_order.hpp"
#include "sirius_config.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace sirius::planner {

/// Sizing and cut options of the distributed sort built by `build_sort_exchange`.
struct sort_exchange_options {
  /// Bytes SORT_SAMPLE accumulates before it computes the range boundaries.
  uint64_t sort_sample_bytes = config::DEFAULT_SORT_SAMPLE_BYTES;
  /// Target bytes per range. 0 sizes ranges from `max_partition_memory_fraction`.
  uint64_t max_partition_bytes = 0;
  /// Fraction of available GPU memory per range when `max_partition_bytes` is 0.
  double max_partition_memory_fraction = config::DEFAULT_MAX_SORT_PARTITION_MEMORY_FRACTION;
  /// Number of leading sort keys the range boundaries are sampled and compared on. Unset means
  /// all keys. Whatever the count, rows equal on these keys always land in the same range.
  std::optional<std::size_t> boundary_key_count;

  /// The options configured in @p params, with every sort key used for boundaries.
  static sort_exchange_options from(operator_params const& params);
};

/**
 * @brief Build the distributed sort `MERGE_SORT → SORT_PARTITION → SORT_SAMPLE → ORDER_BY`.
 *
 * ORDER_BY sorts each batch locally, SORT_SAMPLE picks range boundaries from a sample of the
 * sorted batches, SORT_PARTITION cuts every batch at those boundaries, and MERGE_SORT merges each
 * range after a FULL barrier. SORT_SAMPLE is not a sink, so it runs inside the SORT_PARTITION
 * pipeline.
 *
 * @p order is modified: its projections and types become the identity over its input so every
 * column reaches SORT_SAMPLE and SORT_PARTITION. A non-identity projection it held is applied by
 * MERGE_SORT instead.
 *
 * @param order The local sort. It must already hold its input as its only child.
 * @param options Range sizing and the boundary key prefix.
 * @return The MERGE_SORT at the root of the chain.
 * @throws std::invalid_argument if @p order has no child, or if `options.boundary_key_count` is 0
 * or larger than the number of sort keys.
 */
duckdb::unique_ptr<op::sirius_physical_operator> build_sort_exchange(
  duckdb::unique_ptr<op::sirius_physical_order> order, sort_exchange_options const& options);

}  // namespace sirius::planner
