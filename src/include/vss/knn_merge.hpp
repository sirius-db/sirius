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

#include "vss/brute_force_search.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <raft/core/device_resources.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdint>

namespace sirius::vss {

/**
 * @brief Merge per-part top-k results (one part per right batch) into a single
 *        per-query top-k, via cuVS knn_merge_parts.
 *
 * The join's selection stage searches each right batch separately, producing for
 * every left row a sorted top-k within that batch. This reduces those parts to
 * the global top-k per left row.
 *
 * Input layout is part-major and flat: @p stacked_distances / @p stacked_neighbors
 * are [n_parts * n_samples * k], part p's [n_samples, k] block at offset
 * p * n_samples * k, which is exactly what concatenating the per-part partials in
 * order yields. Neighbor ids are expected to be already global (the selection stage
 * shifts them), so no per-part translation is applied here.
 *
 * Both parts must share @p n_samples and @p k (uniform k across right batches).
 * Results are nearest-first per query, flattened [n_samples * k], matching
 * @ref brute_force_knn. Runs async on @p res's stream.
 *
 * @param res               Caller-owned RAFT resources; runs on its stream.
 * @param stacked_distances FLOAT32 [n_parts*n_samples*k], part-major.
 * @param stacked_neighbors INT64 [n_parts*n_samples*k], part-major, global ids.
 * @param n_samples         Rows per part (the left batch's row count).
 * @param n_parts           Number of parts (right batches merged).
 * @param k                 Neighbors per query.
 * @param stream            Stream for scratch allocations.
 * @param mr                Device resource for the output columns.
 * @return Merged neighbor-index and distance columns [n_samples*k], plus
 *         n_samples and k.
 */
knn_result knn_merge_parts_topk(
  raft::device_resources const& res,
  cudf::column_view const& stacked_distances,
  cudf::column_view const& stacked_neighbors,
  int64_t n_samples,
  int64_t n_parts,
  int64_t k,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace sirius::vss
