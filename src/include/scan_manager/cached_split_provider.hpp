/*
 * Copyright 2025, Sirius Contributors.
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

#include "op/scan/scan_plan.hpp"
#include "scan_manager/split_provider.hpp"

#include <cudf/column/column.hpp>

#include <cucascade/data/cpu_data_representation.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <duckdb/planner/expression.hpp>

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <unordered_map>
#include <variant>
#include <vector>

namespace sirius::scan_manager {
/**
 * @brief Split provider backed by pre-pinned columns from a pinned_entry.
 *
 * The scan_manager builds either the per-column GPU chunk vectors (in scan_plan
 * D-order) or the per-batch host_data_representation chunks (with a D-order
 * column-index map). Internally both are normalised into a per-batch variant so
 * @ref produce_split() can dispatch with @c std::visit and emit one
 * @ref op::scan::scan_cached_operator_data per chunk. Each emitted batch carries
 * either a zero-copy view-backed gpu_table_representation or a sliced
 * host_data_representation (downstream conversion happens in
 * @c scan_cached_operator_data::prepare_for_processing), plus the filter
 * expression and a shared scan_plan; emissions are pushed into the connector by
 * @ref split_provider::run.
 *
 * @par Inputs
 *   - GPU constructor: @p columns_per_request[d] is the chunk vector for
 *     D-position @p d. All inner vectors must have the same size — that size is
 *     the number of emitted batches.
 *   - HOST constructor: @p host_chunks holds one host_data_representation per
 *     emitted batch (each covering all pinned columns), and @p column_indices
 *     selects the columns required by the scan in scan_plan D-order.
 *   - @p chunk_memory_spaces is parallel to the inner vectors of
 *     @p columns_per_request: chunk_memory_spaces[i] is the memory_space*
 *     for chunk i across all D-positions. Each emitted data_batch carries
 *     the memory_space its data lives on so data-locality scheduling fans
 *     tasks correctly across GPUs.
 *   - @p filter_expression and @p plan are forwarded unchanged on every
 *     emitted batch, mirroring the parquet path's per-split contract. The scan
 *     operator queries @c needs_output_assembly(*plan) to decide whether to
 *     reshape the cached batch — when false, the cached batch is forwarded
 *     straight through (no permute, no prune -> no copy).
 */
class cached_split_provider : public split_provider {
 public:
  /// One emitted batch's worth of GPU-resident columns, already arranged in
  /// scan_plan D-order. produce_split() turns this into a zero-copy
  /// gpu_table_representation.
  using gpu_chunk = std::vector<std::shared_ptr<cudf::column>>;
  /// One emitted batch's worth of host-resident data; produce_split() slices it
  /// down to the requested column indices before wrapping it as a data_batch.
  using host_chunk = std::shared_ptr<cucascade::host_data_representation>;
  /// Per-batch storage — exactly one of the two alternatives is populated per
  /// emitted chunk. The same provider only ever holds chunks of one tier (set
  /// at construction by which ctor was called).
  using chunk_variant = std::variant<gpu_chunk, host_chunk>;

  /// GPU-tier constructor: the pinned columns are GPU-resident cudf columns.
  /// Each callable returned by @ref next_split_provider() emits one zero-copy
  /// gpu_table_representation per batch.
  cached_split_provider(std::vector<std::vector<std::shared_ptr<cudf::column>>> columns_per_request,
                        std::vector<cucascade::memory::memory_space*> chunk_memory_spaces,
                        std::shared_ptr<duckdb::Expression> filter_expression,
                        std::shared_ptr<op::scan::scan_plan const> plan);

  /// HOST-tier constructor: each chunk in @p host_chunks is a host_data_representation
  /// holding all pinned columns. @p column_indices selects the columns required by the
  /// scan in scan_plan D-order. Each callable returned by @ref next_split_provider()
  /// slices the claimed chunk by these indices, converts the slice to a
  /// gpu_table_representation on the current device's GPU memory_space (looked
  /// up via @p gpu_memory_spaces keyed by device_id), and emits a GPU-resident
  /// data_batch. The conversion happens at produce_split time so downstream
  /// consumers (sirius_gpu_parquet_scan_operator::execute) see GPU batches and
  /// do not need a separate host-aware code path.
  ///
  /// @param gpu_memory_spaces device_id -> memory_space lookup; cached_split_provider
  ///                          calls cudaGetDevice() to identify the executing GPU
  ///                          and converts host chunks onto the matching space.
  ///                          Empty map throws at conversion time.
  cached_split_provider(
    std::vector<std::shared_ptr<cucascade::host_data_representation>> host_chunks,
    std::vector<std::size_t> column_indices,
    cucascade::memory::memory_space& memory_space,
    std::unordered_map<int, cucascade::memory::memory_space*> gpu_memory_spaces,
    std::shared_ptr<duckdb::Expression> filter_expression,
    std::shared_ptr<op::scan::scan_plan const> plan);

  [[nodiscard]] bool has_more_splits() const override;

  /// \brief Atomically claim the next chunk index and return a callable that
  ///        wraps the cached batch in a single-element vector. After every
  ///        chunk has been claimed, returns a null callable.
  std::function<std::vector<std::unique_ptr<op::operator_data>>()> next_split_provider() override;

 private:
  std::vector<std::unique_ptr<op::operator_data>> produce_split(std::size_t batch_idx);

  std::vector<chunk_variant> _batches;
  // Per-chunk memory_space lookup — each chunk carries the memory_space
  // its data lives on so data-locality scheduling fans tasks correctly.
  // Populated in GPU mode (per-chunk); empty in HOST mode (uses _memory_space).
  std::vector<cucascade::memory::memory_space*> _chunk_memory_spaces;
  std::vector<std::size_t> _column_indices;
  cucascade::memory::memory_space* _memory_space;
  // HOST-mode only: device_id -> GPU memory_space lookup used at produce_split
  // time to materialize host chunks onto the executing GPU.
  std::unordered_map<int, cucascade::memory::memory_space*> _gpu_memory_spaces;
  std::shared_ptr<duckdb::Expression> _filter_expression;
  std::shared_ptr<op::scan::scan_plan const> _plan;
  std::atomic<std::size_t> _next_batch_idx{0};
};

}  // namespace sirius::scan_manager
