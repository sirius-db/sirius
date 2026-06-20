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

#include "vss/vss_pattern.hpp"

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cstddef>
#include <memory>

namespace sirius {
namespace op {

/**
 * @brief Per-batch VSS top-k.
 *
 * Finds the `offset + limit` rows of @p input nearest to `pattern.query` under
 * `pattern.metric`, then assembles the fused projection's output columns
 * (passthrough columns gathered from @p input, plus the cuVS distance column).
 * The result is the per-batch candidate set handed to @ref merge_vss_top_k; it
 * is ordered nearest-first and is not yet offset-sliced (the global merge owns
 * the final offset). Returns the empty output schema when `limit == 0` or
 * @p input is empty.
 */
std::unique_ptr<cudf::table> compute_vss_top_k(cudf::table_view input,
                                               sirius::vss::vss_top_k_pattern const& pattern,
                                               std::size_t limit,
                                               std::size_t offset,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref memory_resource);

/**
 * @brief Consolidate per-batch VSS candidates into the global nearest rows.
 *
 * Each input row already carries its cuVS distance in column @p distance_index,
 * so this is a top-k (ascending) on that column over the concatenated candidate
 * rows, returning `min(num_rows, offset + limit)` rows sorted nearest-first. As
 * with @ref compute_vss_top_k, the final OFFSET slice is applied by the caller.
 */
std::unique_ptr<cudf::table> merge_vss_top_k(cudf::table_view input,
                                             cudf::size_type distance_index,
                                             std::size_t limit,
                                             std::size_t offset,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref memory_resource);

}  // namespace op
}  // namespace sirius
