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

#include "expression_executor/gpu_expression_translator_internal.hpp"
#include "scan_manager/split_provider.hpp"

#include <cudf/column/column.hpp>

#include <cucascade/memory/memory_space.hpp>

#include <duckdb/planner/expression.hpp>

#include <memory>
#include <variant>
#include <vector>

namespace sirius::scan_manager {

/**
 * @brief Split provider backed by pre-pinned columns from a pinned_entry.
 *
 * The scan_manager builds the per-column chunk vectors by looking up the
 * columns required by the incoming scan operator (in column_ids order) in the
 * pinned_entry's @c data_batches_by_column map. start() then assembles one
 * @ref op::scan::scan_cached_operator_data per chunk: each carries a
 * zero-copy view-backed data_batch over the pinned columns plus the
 * filter / post-filter-projection metadata derived from the scan_info, and is
 * pushed into the connector.
 *
 * @par Inputs
 *   - @p columns_per_request[c] is the chunk vector for the c-th requested
 *     column. All inner vectors must have the same size — that size is the
 *     number of emitted batches.
 *   - @p memory_space is captured into each emitted data_batch so memory
 *     accounting matches where the cached columns reside.
 *   - @p filter_expression and @p post_filter_projection_ids are forwarded
 *     unchanged on every emitted batch, mirroring the parquet path.
 */
class cached_split_provider : public split_provider {
 public:
  using translated_expression =
    sirius::gpu_expression_translator::translated_expression;

  cached_split_provider(
    std::vector<std::vector<std::shared_ptr<cudf::column>>> columns_per_request,
    cucascade::memory::memory_space& memory_space,
    std::variant<std::shared_ptr<translated_expression>, std::shared_ptr<duckdb::Expression>>
      filter_expression,
    std::vector<std::size_t> post_filter_projection_ids);

  std::future<void> start(exec::thread_pool& pool, split_connector& connector) override;

 private:
  std::vector<std::vector<std::shared_ptr<cudf::column>>> _columns_per_request;
  cucascade::memory::memory_space* _memory_space;
  std::variant<std::shared_ptr<translated_expression>, std::shared_ptr<duckdb::Expression>>
    _filter_expression;
  std::vector<std::size_t> _post_filter_projection_ids;
};

}  // namespace sirius::scan_manager
