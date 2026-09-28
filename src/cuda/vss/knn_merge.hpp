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

#include "cuda/vss/brute_force_search.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <raft/core/device_resources.hpp>

#include <rmm/resource_ref.hpp>

#include <cstdint>

namespace sirius::vss {

//! cuVS knn_merge_parts only supports up to this many neighbors per query.
inline constexpr int64_t KNN_MERGE_MAX_K = 1024;

/**
 * @brief Merge per-part top-k results (one part per right batch) into a single
 *        per-query top-k, via cuVS knn_merge_parts.
 *
 * Each right batch is searched separately, giving every left row a sorted top-k within that
 * batch. This reduces those parts to the top-k over all of them.
 *
 * Input layout is part-major and flat: @p stacked_distances / @p stacked_neighbors are
 * [n_parts * n_samples * k], part p's [n_samples, k] block at offset p * n_samples * k, which is
 * what concatenating the per-part results in order yields. Neighbor ids are returned as given
 * (no per-part translation), so they must already be unique across parts.
 *
 * Results are nearest-first per query, flattened [n_samples * k], matching @ref brute_force_knn.
 * Runs async on @p res's stream.
 *
 * @param res               Caller-owned RAFT resources; runs on its stream.
 * @param stacked_distances FLOAT32 [n_parts*n_samples*k], part-major.
 * @param stacked_neighbors INT64 [n_parts*n_samples*k], part-major, unique across parts.
 * @param n_samples         Rows per part (the left batch's row count).
 * @param n_parts           Number of parts (right batches merged).
 * @param k                 Neighbors per query, at most @ref KNN_MERGE_MAX_K.
 * @param mr                Device resource for the output columns.
 * @return Merged neighbor-index and distance columns [n_samples*k].
 */
knn_result knn_merge_parts_topk(
  raft::device_resources const& res,
  cudf::column_view const& stacked_distances,
  cudf::column_view const& stacked_neighbors,
  int64_t n_samples,
  int64_t n_parts,
  int64_t k,
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace sirius::vss
