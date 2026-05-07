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

#include <cucascade/memory/memory_space.hpp>
#include <duckdb/planner/expression.hpp>

#include <memory>
#include <vector>

namespace sirius::scan_manager {

/**
 * @brief Split provider backed by pre-pinned columns from a pinned_entry.
 *
 * The scan_manager builds the per-column chunk vectors in scan_plan D-order
 * (one entry per @c data_columns slot, looked up by name in the pinned entry).
 * start() then assembles one @ref op::scan::scan_cached_operator_data per
 * chunk: each carries a zero-copy view-backed data_batch over the pinned
 * columns plus the filter expression and a shared scan_plan, and is pushed
 * into the connector.
 *
 * @par Inputs
 *   - @p columns_per_request[d] is the chunk vector for D-position @p d. All
 *     inner vectors must have the same size — that size is the number of
 *     emitted batches.
 *   - @p chunk_memory_spaces is parallel to the inner vectors of
 *     @p columns_per_request: chunk_memory_spaces[i] is the memory_space*
 *     for chunk i across all D-positions. Per Phase 22 D-04, each emitted
 *     data_batch carries the memory_space its data lives on so SCHED-01
 *     routing fans tasks correctly across GPUs.
 *   - @p filter_expression and @p plan are forwarded unchanged on every
 *     emitted batch, mirroring the parquet path's per-split contract. The scan
 *     operator queries @c needs_output_assembly(*plan) to decide whether to
 *     reshape the cached batch — when false, the cached batch is forwarded
 *     straight through (no permute, no prune -> no copy).
 */
class cached_split_provider : public split_provider {
 public:
  cached_split_provider(std::vector<std::vector<std::shared_ptr<cudf::column>>> columns_per_request,
                        std::vector<cucascade::memory::memory_space*> chunk_memory_spaces,
                        std::shared_ptr<duckdb::Expression> filter_expression,
                        std::shared_ptr<op::scan::scan_plan const> plan);

  std::future<void> start(exec::thread_pool& pool, split_connector& connector) override;

 private:
  std::vector<std::vector<std::shared_ptr<cudf::column>>> _columns_per_request;
  // Phase 22 D-04: per-chunk memory_space lookup. Replaces entry-level
  // _memory_space (now gone post-PIN-MGPU-01); each chunk carries the
  // memory_space its data lives on so SCHED-01 routing fans tasks correctly.
  std::vector<cucascade::memory::memory_space*> _chunk_memory_spaces;
  std::shared_ptr<duckdb::Expression> _filter_expression;
  std::shared_ptr<op::scan::scan_plan const> _plan;
};

}  // namespace sirius::scan_manager
