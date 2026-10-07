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

#include "cuda/vss/cudf_raft_interop.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>

#include <raft/core/device_resources.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cuvs/distance/distance.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace sirius::vss {

/**
 * @brief Knobs that describe a clustering, i.e. the centroids themselves.
 *
 * Separated from @ref assignment_spec because these are amortizable: one trained set of
 * centroids can serve many joins, while an assignment is per-query. Nothing today reuses a
 * clustering across queries, but keeping the two apart is what makes that possible without
 * moving every field.
 */
struct clustering_spec {
  /// Number of centroids. 0 selects @ref resolve_n_clusters's sqrt(n_rows) default.
  std::int64_t n_clusters{0};
  /// Rows to train on. 0 selects @ref resolve_train_rows's default. Fewer rows means a
  /// faster fit against looser centroids; the whole dataset is the other extreme.
  std::int64_t train_rows{0};
  /// Balanced k-means iterations.
  std::int32_t n_iters{20};
  /// Offsets the training sample so a re-fit under the same spec is reproducible while
  /// different seeds draw different rows.
  std::uint64_t seed{0};
  /// Must be a metric balanced k-means supports, and must match the metric the join that
  /// consumes these centroids uses. Both values @ref ann_distance_type_from_metric returns
  /// qualify, and both are true-distance rather than squared, which is what lets
  /// @ref assignment_spec::radius_factor be read as a plain distance ratio.
  cuvs::distance::DistanceType metric{cuvs::distance::DistanceType::L2SqrtExpanded};
};

/**
 * @brief Knobs that describe how rows attach to an existing set of centroids.
 *
 * Applies unchanged to either side of a join: on the build side more than one cluster per row
 * is replication (a row is findable from more entry points), on the probe side it is a wider
 * search. The mechanism is identical, which is why there is one spec and not two.
 */
struct assignment_spec {
  /// Clusters per row when @ref radius_factor is 0.
  std::int64_t n_probes{1};
  /// When > 0, a row takes every candidate centroid within (1 + radius_factor) times its
  /// distance to the nearest one, instead of a fixed count. Relative rather than absolute so
  /// one value means the same thing across datasets, dimensionalities and metrics.
  double radius_factor{0.0};
  /// Upper bound on clusters per row under @ref radius_factor, which also fixes how deep the
  /// candidate search runs. 0 falls back to @ref n_probes. Ignored when radius_factor is 0.
  std::int64_t max_probes{0};
};

/**
 * @brief Which clusters each vector belongs to, as a ragged edge list.
 *
 * One row per (vector, cluster) pair, so a vector appears once under a 1:1 assignment and
 * several times when it takes multiple clusters. Deliberately not a per-row cluster id
 * column: the edge-list shape is the same whether the assignment came from a fixed count or
 * a radius, and it is what the join, the partitioner and the SQL surface all consume.
 *
 * All three columns have equal length and are positionally aligned. Ordered by @c row_ids,
 * and within a row by increasing @c distances.
 */
struct cluster_assignment {
  std::unique_ptr<cudf::column> row_ids;      ///< INT64 row id in the source table's row space.
  std::unique_ptr<cudf::column> cluster_ids;  ///< INT32 centroid index, in centroid row order.
  std::unique_ptr<cudf::column> distances;  ///< FLOAT32 distance from the vector to that centroid.
};

/// Requested count, or sqrt(n_rows) when @p requested is 0. Clamped to [1, n_rows].
[[nodiscard]] std::int64_t resolve_n_clusters(std::int64_t requested, std::int64_t n_rows);

/// Requested count, or 256 rows per cluster when @p requested is 0. Clamped to [n_clusters,
/// n_rows], since balanced k-means cannot place more centroids than it has training rows.
[[nodiscard]] std::int64_t resolve_train_rows(std::int64_t requested,
                                              std::int64_t n_rows,
                                              std::int64_t n_clusters);

/**
 * @brief A @p take-row sample of a fixed-width FLOAT32 LIST vector column.
 *
 * A strided walk rather than a uniform random draw: @p seed only shifts the starting offset
 * within the first stride. That keeps a fit reproducible and costs no shuffle, but on a column
 * already ordered by something correlated with the vectors the sample is biased, and the
 * remedy is more rows rather than a different seed.
 *
 * Returns a copy of the whole column when @p take covers it. Exposed so a caller streaming a
 * corpus chunk by chunk can shrink each chunk to its share of the training sample and drop the
 * staged chunk before taking the next.
 */
[[nodiscard]] std::unique_ptr<cudf::column> sample_vector_rows(cudf::column_view const& vectors,
                                                               std::int64_t take,
                                                               std::uint64_t seed,
                                                               rmm::cuda_stream_view stream,
                                                               rmm::device_async_resource_ref mr);

/**
 * @brief Train balanced k-means centroids over @p chunks.
 *
 * @p chunks are the dataset's vectors held as separate fixed-width FLOAT32 LIST columns (each
 * unsliced, gap-free, width @p dim), in row order -- the same shape the pinned-table and
 * out-of-core paths already hand around, so no caller has to concatenate a whole dataset to
 * cluster it.
 *
 * Only the training sample is resident at once, not the dataset: rows are drawn from every
 * chunk in proportion to its size, so the fit's device footprint is set by
 * @ref clustering_spec::train_rows. The draw is a strided walk, not a uniform random one; on
 * a dataset already ordered by something correlated with the vectors that is a biased sample,
 * and the fix is to raise train_rows rather than to trust the stride.
 *
 * Returns the centroids as a @p dim-wide FLOAT32 LIST column of `n_clusters` rows -- the same
 * representation as a vector column, so the result round-trips through
 * @ref list_column_as_dataset_view and can be handed straight back as a table.
 *
 * The work is enqueued on @p stream and synchronized before returning.
 */
[[nodiscard]] std::unique_ptr<cudf::column> train_centroids(
  std::vector<cudf::column_view> const& chunks,
  std::int64_t dim,
  clustering_spec const& spec,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Attach every vector in @p vectors to its nearest centroids.
 *
 * This is a brute-force k-NN of the vectors against the centroid matrix, which is cheap
 * because a centroid set is small: the cost is `n_rows * n_clusters * dim`, and n_clusters is
 * the square root of a table, not the table.
 *
 * @p row_id_base is added to every emitted row id, so a caller walking a dataset chunk by
 * chunk passes the rows already consumed and gets dataset-global ids back.
 *
 * @p vectors and @p centroids must share @p dim. @p metric must be the one the centroids were
 * trained under; a mismatch silently assigns rows to the wrong clusters rather than failing.
 *
 * Enqueued on @p res's stream and not synchronized: the returned columns are only readable on
 * the host after the caller syncs. @p res is the caller's because it must outlive that pending
 * work -- a handle destroyed here would free its cuVS workspace while kernels still read it --
 * and because reusing one handle across batches pays its workspace setup once.
 */
[[nodiscard]] cluster_assignment assign_to_centroids(raft::device_resources const& res,
                                                     cudf::column_view const& vectors,
                                                     cudf::column_view const& centroids,
                                                     std::int64_t dim,
                                                     assignment_spec const& spec,
                                                     std::int64_t row_id_base,
                                                     cuvs::distance::DistanceType metric,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr);

/**
 * @brief Each row's single nearest centroid, as INT32 labels written to @p labels.
 *
 * The one-probe case of @ref assign_to_centroids without its edge list: cuVS's k-means predict
 * fuses the distance and the argmin into one pass, where a top-1 search materializes every
 * row-centroid distance and then selects from it. Same enqueue-only contract.
 */
void nearest_centroid_labels(raft::device_resources const& res,
                             cudf::column_view const& vectors,
                             cudf::column_view const& centroids,
                             std::int64_t dim,
                             cuvs::distance::DistanceType metric,
                             std::int32_t* labels,
                             rmm::device_async_resource_ref mr);

}  // namespace sirius::vss
