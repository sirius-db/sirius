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

#include "cuda/vss/brute_force_search.hpp"
#include "cuda/vss/cudf_raft_interop.hpp"
#include "vss/scoped_device_resource.hpp"
#include "vss/vector_clustering.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include <cuvs/cluster/kmeans.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sirius::vss {

namespace {

/// Rows a fixed-width LIST vector column holds.
[[nodiscard]] std::int64_t chunk_row_count(cudf::column_view const& chunk)
{
  return static_cast<std::int64_t>(chunk.size());
}

}  // namespace

std::unique_ptr<cudf::column> sample_vector_rows(cudf::column_view const& vectors,
                                                 std::int64_t take,
                                                 std::uint64_t seed,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  auto const rows = chunk_row_count(vectors);
  if (take <= 0) { throw std::invalid_argument("sample_vector_rows: take must be >= 1"); }
  if (take >= rows) { return std::make_unique<cudf::column>(vectors, stream, mr); }

  // The stride walk is what keeps this allocation-light: no shuffle, no random index buffer,
  // just an arithmetic sequence used as a gather map.
  auto const stride = rows / take;
  auto const start  = static_cast<std::int32_t>(seed % static_cast<std::uint64_t>(stride));
  cudf::numeric_scalar<std::int32_t> const init(start, true, stream);
  cudf::numeric_scalar<std::int32_t> const step(static_cast<std::int32_t>(stride), true, stream);
  auto const map = cudf::sequence(static_cast<cudf::size_type>(take), init, step, stream, mr);

  auto gathered = cudf::gather(
    cudf::table_view{{vectors}}, map->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
  return std::move(gathered->release()[0]);
}

std::int64_t resolve_n_clusters(std::int64_t requested, std::int64_t n_rows)
{
  if (n_rows <= 0) { return 0; }
  auto const resolved =
    requested > 0 ? requested : static_cast<std::int64_t>(std::sqrt(static_cast<double>(n_rows)));
  return std::clamp<std::int64_t>(resolved, 1, n_rows);
}

std::int64_t resolve_train_rows(std::int64_t requested,
                                std::int64_t n_rows,
                                std::int64_t n_clusters)
{
  if (n_rows <= 0 || n_clusters <= 0) { return 0; }
  auto const resolved = requested > 0 ? requested : n_clusters * 256;
  return std::clamp<std::int64_t>(resolved, n_clusters, n_rows);
}

std::unique_ptr<cudf::column> train_centroids(std::vector<cudf::column_view> const& chunks,
                                              std::int64_t dim,
                                              clustering_spec const& spec,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  if (dim <= 0) { throw std::invalid_argument("train_centroids: dim must be >= 1"); }

  std::int64_t n_rows = 0;
  for (auto const& chunk : chunks) {
    n_rows += chunk_row_count(chunk);
  }
  if (n_rows == 0) { throw std::invalid_argument("train_centroids: no rows to cluster"); }

  auto const n_clusters = resolve_n_clusters(spec.n_clusters, n_rows);
  auto const train_rows = resolve_train_rows(spec.train_rows, n_rows, n_clusters);

  // Each chunk contributes in proportion to its size, rounded up so a chunk small relative to
  // train_rows still lands at least one row rather than dropping out of the sample entirely.
  std::vector<std::unique_ptr<cudf::column>> samples;
  std::int64_t taken = 0;
  for (auto const& chunk : chunks) {
    auto const rows = chunk_row_count(chunk);
    if (rows == 0) { continue; }
    auto take = std::min<std::int64_t>(rows, ((train_rows * rows) + n_rows - 1) / n_rows);
    take      = std::min<std::int64_t>(take, train_rows - taken);
    if (take <= 0) { break; }
    samples.push_back(sample_vector_rows(chunk, take, spec.seed, stream, mr));
    taken += take;
  }
  if (samples.empty()) { throw std::invalid_argument("train_centroids: sample is empty"); }

  std::unique_ptr<cudf::column> merged;
  if (samples.size() > 1) {
    std::vector<cudf::column_view> views;
    views.reserve(samples.size());
    for (auto const& s : samples) {
      views.push_back(s->view());
    }
    merged = cudf::concatenate(views, stream, mr);
  } else {
    merged = std::move(samples.front());
  }

  auto const sample_view = list_column_as_dataset_view(merged->view(), dim);
  auto centroid_values   = cudf::make_numeric_column(cudf::data_type{cudf::type_id::FLOAT32},
                                                   static_cast<cudf::size_type>(n_clusters * dim),
                                                   cudf::mask_state::UNALLOCATED,
                                                   stream,
                                                   mr);

  {
    // cuVS takes no allocator, so its k-means scratch is routed by the ambient resource.
    scoped_current_device_resource const route{mr};
    raft::device_resources res{stream};
    auto centroids_view = raft::make_device_matrix_view<float, std::int64_t, raft::row_major>(
      centroid_values->mutable_view().data<float>(), n_clusters, dim);

    cuvs::cluster::kmeans::balanced_params params;
    params.metric  = spec.metric;
    params.n_iters = static_cast<std::uint32_t>(std::max(spec.n_iters, 1));
    cuvs::cluster::kmeans::fit(res, params, sample_view, centroids_view);
    // The sample borrows `merged`, which is dropped on return, so the fit must be complete.
    raft::resource::sync_stream(res);
  }

  cudf::numeric_scalar<std::int32_t> const zero(0, true, stream);
  cudf::numeric_scalar<std::int32_t> const width(static_cast<std::int32_t>(dim), true, stream);
  auto offsets =
    cudf::sequence(static_cast<cudf::size_type>(n_clusters + 1), zero, width, stream, mr);

  return cudf::make_lists_column(static_cast<cudf::size_type>(n_clusters),
                                 std::move(offsets),
                                 std::move(centroid_values),
                                 0,
                                 rmm::device_buffer{});
}

cluster_assignment assign_to_centroids(raft::device_resources const& res,
                                       cudf::column_view const& vectors,
                                       cudf::column_view const& centroids,
                                       std::int64_t dim,
                                       assignment_spec const& spec,
                                       std::int64_t row_id_base,
                                       cuvs::distance::DistanceType metric,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  auto const queries    = list_column_as_dataset_view(vectors, dim);
  auto const centers    = list_column_as_dataset_view(centroids, dim);
  auto const n_rows     = static_cast<std::int64_t>(queries.extent(0));
  auto const n_clusters = static_cast<std::int64_t>(centers.extent(0));
  if (n_clusters == 0) { throw std::invalid_argument("assign_to_centroids: no centroids"); }

  auto const radius_mode = spec.radius_factor > 0.0;
  // Under a radius the search still has to be bounded, and max_probes is that bound: it is
  // both the cap on clusters per row and how deep the candidate search runs.
  auto const requested = radius_mode && spec.max_probes > 0 ? spec.max_probes : spec.n_probes;
  auto const depth     = std::clamp<std::int64_t>(requested, 1, n_clusters);

  // A cudf column cannot hold more than 2^31 rows, and the edge list is n_rows * depth long, so
  // a large enough table times a deep enough probe overflows the column rather than the maths.
  // Caught here because the truncation is silent: the cast would produce a short, wrong result.
  auto const edges = n_rows * depth;
  if (edges > static_cast<std::int64_t>(std::numeric_limits<cudf::size_type>::max())) {
    throw std::invalid_argument(
      "assign_to_centroids: " + std::to_string(n_rows) + " rows x " + std::to_string(depth) +
      " clusters is " + std::to_string(edges) +
      " edges, which exceeds what a cudf column can hold; assign in smaller chunks or lower "
      "n_probes/max_probes");
  }

  // One GEMM and one selection, like the join's own searches: cuVS brute force at so small a
  // depth runs a fused SIMT kernel instead, which took ~0.26 s to route 180k rows to 2 of 1024
  // centroids (SIRIUS_VSS_GEMM_TOPK=0 keeps cuVS, as for the join).
  auto const use_gemm = [&] {
    auto const* v = std::getenv("SIRIUS_VSS_GEMM_TOPK");
    return (v == nullptr || std::string_view{v} != "0") && gemm_search_supports(metric);
  }();
  auto knn = use_gemm ? gemm_topk(res, centers, queries, depth, metric, mr)
                      : brute_force_knn(res, centers, queries, depth, metric, mr);

  auto const total = static_cast<cudf::size_type>(edges);
  cudf::numeric_scalar<std::int32_t> const zero(0, true, stream);
  cudf::numeric_scalar<std::int32_t> const one(1, true, stream);
  cudf::numeric_scalar<std::int32_t> const depth32(static_cast<std::int32_t>(depth), true, stream);

  // Position within the flattened [n_rows, depth] result; integer-divided by depth it is the
  // row each entry belongs to, and multiplied back it addresses that row's nearest centroid.
  auto const positions = cudf::sequence(total, zero, one, stream, mr);
  auto const local_row = cudf::binary_operation(positions->view(),
                                                depth32,
                                                cudf::binary_operator::DIV,
                                                cudf::data_type{cudf::type_id::INT32},
                                                stream,
                                                mr);

  cudf::numeric_scalar<std::int64_t> const base(row_id_base, true, stream);
  auto const row_ids_local =
    cudf::cast(local_row->view(), cudf::data_type{cudf::type_id::INT64}, stream, mr);
  auto row_ids = cudf::binary_operation(row_ids_local->view(),
                                        base,
                                        cudf::binary_operator::ADD,
                                        cudf::data_type{cudf::type_id::INT64},
                                        stream,
                                        mr);
  auto cluster_ids =
    cudf::cast(knn.neighbors->view(), cudf::data_type{cudf::type_id::INT32}, stream, mr);
  auto distances = std::move(knn.distances);

  if (!radius_mode) {
    return cluster_assignment{std::move(row_ids), std::move(cluster_ids), std::move(distances)};
  }

  auto const nearest_pos = cudf::binary_operation(local_row->view(),
                                                  depth32,
                                                  cudf::binary_operator::MUL,
                                                  cudf::data_type{cudf::type_id::INT32},
                                                  stream,
                                                  mr);
  auto const nearest     = cudf::gather(cudf::table_view{{distances->view()}},
                                    nearest_pos->view(),
                                    cudf::out_of_bounds_policy::DONT_CHECK,
                                    stream,
                                    mr);
  cudf::numeric_scalar<float> const factor(
    static_cast<float>(1.0 + spec.radius_factor), true, stream);
  auto const limit = cudf::binary_operation(nearest->get_column(0).view(),
                                            factor,
                                            cudf::binary_operator::MUL,
                                            cudf::data_type{cudf::type_id::FLOAT32},
                                            stream,
                                            mr);
  // The nearest centroid is at distance d0 and the limit is (1 + factor) * d0, so every row
  // keeps at least its own nearest and no row can drop out of the assignment.
  auto const keep = cudf::binary_operation(distances->view(),
                                           limit->view(),
                                           cudf::binary_operator::LESS_EQUAL,
                                           cudf::data_type{cudf::type_id::BOOL8},
                                           stream,
                                           mr);

  auto filtered = cudf::apply_boolean_mask(
    cudf::table_view{{row_ids->view(), cluster_ids->view(), distances->view()}},
    keep->view(),
    stream,
    mr);
  auto columns = filtered->release();
  return cluster_assignment{std::move(columns[0]), std::move(columns[1]), std::move(columns[2])};
}

void nearest_centroid_labels(raft::device_resources const& res,
                             cudf::column_view const& vectors,
                             cudf::column_view const& centroids,
                             std::int64_t dim,
                             cuvs::distance::DistanceType metric,
                             std::int32_t* labels,
                             rmm::device_async_resource_ref mr)
{
  auto const queries = list_column_as_dataset_view(vectors, dim);
  auto const centers = list_column_as_dataset_view(centroids, dim);
  if (centers.extent(0) == 0) {
    throw std::invalid_argument("nearest_centroid_labels: no centroids");
  }
  if (queries.extent(0) == 0) { return; }
  // cuVS takes no allocator, so its scratch is routed by the ambient resource.
  scoped_current_device_resource const route{mr};
  cuvs::cluster::kmeans::balanced_params params;
  params.metric = metric;
  cuvs::cluster::kmeans::predict(
    res,
    params,
    queries,
    centers,
    raft::make_device_vector_view<int, std::int64_t>(labels, queries.extent(0)));
}

}  // namespace sirius::vss
