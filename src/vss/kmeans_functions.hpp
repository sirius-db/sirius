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

#include "vss/vector_clustering.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace cucascade {
class host_data_representation;
}  // namespace cucascade
namespace duckdb {
class SiriusContext;
}  // namespace duckdb

namespace sirius::vss {

/**
 * @brief `sirius_kmeans_fit(table, column, name => ...)`: train centroids over a pinned column
 *        and keep them GPU-resident under @c name.
 *
 * The centroids stay in the index cache rather than being returned as rows. A clustering is
 * consumed by later GPU work (assignment, and the join's cluster pruning), so round-tripping
 * it through a DuckDB table would only mean uploading it again, and a name is what lets both
 * sides of a join share one clustering -- which is the whole point of co-clustering.
 */
struct kmeans_fit_request {
  std::string name;     ///< Cache key the clustering is stored under; replaces any existing one.
  std::string catalog;  ///< Resolved catalog of the pinned table
  std::string schema;   ///< Resolved schema
  std::string table;    ///< Base table holding the vectors to cluster
  std::string column;   ///< Vector column
  std::int64_t dim{0};  ///< Vector dimensionality, from the column's ARRAY type
  std::string metric{"l2"};
  clustering_spec spec;
};

/**
 * @brief `sirius_kmeans_assign(table, column, clustering)`: attach a pinned column's rows to a
 *        named clustering's centroids.
 *
 * Emits the assignment edge list as rows, so the caller can join it back onto the table and
 * materialize a cluster-ordered copy in ordinary SQL. Applies to either side of a join: the
 * build side to partition the corpus, the probe side to co-cluster the queries.
 */
struct kmeans_assign_request {
  std::string clustering;  ///< Name a previous @ref kmeans_fit_request stored
  std::string catalog;
  std::string schema;
  std::string table;
  std::string column;
  std::int64_t dim{0};
  assignment_spec spec;
};

/// What a fit resolved its auto-valued knobs to, for the caller to report.
struct kmeans_fit_result {
  std::int64_t n_clusters{0};
  std::int64_t dim{0};
  std::int64_t train_rows{0};  ///< Rows actually sampled, which the auto default derives from
  std::int64_t n_rows{0};      ///< Rows in the clustered column
};

kmeans_fit_result run_kmeans_fit(duckdb::SiriusContext& ctx, const kmeans_fit_request& req);

/// Result columns: `row_id INT64`, `cluster_id INT32`, `distance FLOAT32`.
std::unique_ptr<cucascade::host_data_representation> run_kmeans_assign(
  duckdb::SiriusContext& ctx, const kmeans_assign_request& req);

/**
 * @brief `sirius_kmeans_centroids(clustering)`: read a named clustering's centroids back as rows.
 *
 * Emitted in long form -- one row per (centroid, component) rather than one row per centroid --
 * so the result needs no ARRAY column and can be pivoted in plain SQL.
 *
 * This exists to make the clustering checkable against something other than itself. Without it
 * a test can only confirm that an assignment is internally consistent; with it, a query can
 * recompute each row's nearest centroid on the CPU and compare, which is what distinguishes
 * "the code agrees with itself" from "the answer is right".
 *
 * Result columns: `cluster_id INT32`, `dim_index INT32`, `value FLOAT32`.
 */
std::unique_ptr<cucascade::host_data_representation> run_kmeans_centroids(
  duckdb::SiriusContext& ctx, const std::string& clustering);

/// The centroid column a named clustering holds, or nullptr when @p name is absent or names an
/// entry that is not a clustering. Owned by the cache; valid until that entry is dropped.
[[nodiscard]] const cudf::column* find_clustering_centroids(duckdb::SiriusContext& ctx,
                                                            const std::string& name);

}  // namespace sirius::vss
