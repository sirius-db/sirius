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

#include "vss/cudf_raft_interop.hpp"

#include <cudf/column/column_view.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuvs/distance/distance.hpp>

#include <cstdint>

namespace sirius::vss {

/**
 * @brief Recompute the exact distance for each selected neighbor, in place,
 *        directly from the vectors (no norm decomposition → no cancellation).
 *
 * The selection pass ranks with the Expanded (GEMM) metric, which loses
 * precision for distances near zero — exactly the dedup regime. This refines the
 * *values* of the chosen top-k without changing the selection: for L2 it computes
 * `sqrt(sum (q-d)^2)`; for cosine it computes `1 - (q·d)/(|q||d|)` from the raw
 * dot product and norms.
 *
 * Runs per selected pair (`n_queries * k` of them), using the query and dataset
 * vectors already on hand for this (left batch, right batch) pair, so nothing is
 * gathered or concatenated. Must be called BEFORE the neighbor ids are shifted to
 * global space — @p neighbors are local ids into @p dataset.
 *
 * @param queries    Row-major `[n_queries, dim]` query vectors.
 * @param dataset    Row-major `[n_dataset, dim]` vectors of this right batch.
 * @param neighbors  INT64 `[n_queries * k]` local ids into @p dataset.
 * @param distances  FLOAT32 `[n_queries * k]` distances, overwritten in place.
 * @param k          Neighbors per query.
 * @param metric     Selection metric; L2Sqrt gets a euclidean refine, Cosine a cosine refine.
 * @param stream     Stream the kernel runs on.
 */
void refine_topk_distances(dataset_matrix_view queries,
                           dataset_matrix_view dataset,
                           cudf::column_view const& neighbors,
                           cudf::mutable_column_view const& distances,
                           int64_t k,
                           cuvs::distance::DistanceType metric,
                           rmm::cuda_stream_view stream);

}  // namespace sirius::vss
