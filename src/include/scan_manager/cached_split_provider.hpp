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

#include "op/scan/hive_partition.hpp"  // partition_inject_fn_t
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
 * columns plus the filter expression and the inject closure derived from the
 * scan_plan, and is pushed into the connector.
 *
 * @par Inputs
 *   - @p columns_per_request[d] is the chunk vector for D-position @p d. All
 *     inner vectors must have the same size — that size is the number of
 *     emitted batches.
 *   - @p memory_space is captured into each emitted data_batch so memory
 *     accounting matches where the cached columns reside.
 *   - @p filter_expression and @p inject_fn are forwarded unchanged on every
 *     emitted batch, mirroring the parquet path's per-split contract.
 *     @p inject_fn may be a null function — in that case the cached batch is
 *     forwarded straight through (no permute, no prune -> no copy).
 */
class cached_split_provider : public split_provider {
 public:
  cached_split_provider(std::vector<std::vector<std::shared_ptr<cudf::column>>> columns_per_request,
                        cucascade::memory::memory_space& memory_space,
                        std::shared_ptr<duckdb::Expression> filter_expression,
                        op::scan::partition_inject_fn_t inject_fn);

  std::future<void> start(exec::thread_pool& pool, split_connector& connector) override;

 private:
  std::vector<std::vector<std::shared_ptr<cudf::column>>> _columns_per_request;
  cucascade::memory::memory_space* _memory_space;
  std::shared_ptr<duckdb::Expression> _filter_expression;
  op::scan::partition_inject_fn_t _inject_fn;
};

}  // namespace sirius::scan_manager
