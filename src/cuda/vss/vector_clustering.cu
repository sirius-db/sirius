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
#include "vss/bound_gemm.hpp"
#include "vss/cluster_fold.hpp"
#include "vss/cluster_lists.hpp"
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
#include <cudf/utilities/error.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include <rmm/device_uvector.hpp>

#include <cuvs/cluster/kmeans.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
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

namespace {

constexpr int kLabelBlock = 256;
/// Rows ranked per launch: 1,024 probe tiles, and what sizes the candidate buffer.
constexpr std::int64_t kLabelBatch = std::int64_t{1} << 17;
/// One centroid in this many seeds a row's bound, so about this many centroids pass its filter.
constexpr std::int64_t kSeedStride = 16;
/// Candidates the buffer holds per row: four times what the seed leaves on average.
constexpr std::int64_t kCandidatesPerRow = 4 * kSeedStride + 16;
/// With fewer centroids than this there is no sample worth a second pass.
constexpr std::int64_t kMinFilteredCentroids = 256;

int label_grid(std::int64_t n)
{
  return static_cast<int>(std::clamp<std::int64_t>((n + kLabelBlock - 1) / kLabelBlock, 1, 65535));
}

// One fixed summation order for every FP32 distance the labels are decided by.
__device__ inline float direct_sq_distance(float const* a, float const* b, int64_t d)
{
  float acc = 0.f;
  for (int64_t j = 0; j < d; ++j) {
    float const e = a[j] - b[j];
    acc           = fmaf(e, e, acc);
  }
  return acc;
}

// bound[r] = row r's FP32 distance to its seed centroid, raised by the sum's rounding so it is
// at least the true distance of every centroid whose computed distance is no larger.
__global__ void seed_bound_kernel(float const* x,
                                  int64_t m,
                                  int64_t d,
                                  float const* seeds,
                                  int const* seed,
                                  float inflate,
                                  float* bound)
{
  for (int64_t r = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; r < m;
       r += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    bound[r] = direct_sq_distance(x + r * d, seeds + seed[r] * d, d) * inflate;
  }
}

__global__ void reset_rows_kernel(unsigned* nearest_bits, unsigned long long* best, int64_t m)
{
  for (int64_t r = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; r < m;
       r += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    nearest_bits[r] = 0x7f800000u;  // +inf
    best[r]         = ~0ull;
  }
}

// nearest_bits[r] = the smallest FP16-computed distance among row r's candidates. A distance
// that is not finite (a value past half's range) is counted in *unusable.
__global__ void nearest_filtered_kernel(int32_t const* rows,
                                        float const* dist,
                                        unsigned long long const* count,
                                        unsigned long long capacity,
                                        unsigned* nearest_bits,
                                        unsigned long long* unusable)
{
  auto const n = static_cast<int64_t>(min(*count, capacity));
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    float const f = dist[i];
    if (!(f < INFINITY)) {
      atomicAdd(unusable, 1ull);
      continue;
    }
    atomicMin(nearest_bits + rows[i], __float_as_uint(fmaxf(f, 0.f)));
  }
}

// A candidate's FP16-computed distance is within limit - bound of its true one, so only those
// within twice that of the row's smallest can be its nearest centroid; each of those is scored
// in FP32 and the smallest (distance, centroid) kept.
__global__ void rescore_kernel(float const* x,
                               int64_t d,
                               float const* centers,
                               int32_t const* rows,
                               int64_t const* ids,
                               float const* dist,
                               unsigned long long const* count,
                               unsigned long long capacity,
                               unsigned const* nearest_bits,
                               float const* bound,
                               float const* limit,
                               float widen,
                               unsigned long long* best)
{
  auto const n = static_cast<int64_t>(min(*count, capacity));
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    auto const r    = static_cast<int64_t>(rows[i]);
    float const f   = fmaxf(dist[i], 0.f);
    float const cut = (__uint_as_float(nearest_bits[r]) + 2.f * (limit[r] - bound[r])) * widen;
    if (!(f <= cut)) { continue; }
    float const t = direct_sq_distance(x + r * d, centers + ids[i] * d, d);
    atomicMin(best + r,
              (static_cast<unsigned long long>(__float_as_uint(t)) << 32) |
                static_cast<unsigned long long>(static_cast<unsigned>(ids[i])));
  }
}

__global__ void take_labels_kernel(unsigned long long const* best,
                                   int64_t m,
                                   int32_t* labels,
                                   unsigned long long* unusable)
{
  for (int64_t r = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; r < m;
       r += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    if (best[r] == ~0ull) {
      atomicAdd(unusable, 1ull);
      labels[r] = 0;
    } else {
      labels[r] = static_cast<int32_t>(best[r] & 0xffffffffull);
    }
  }
}

bool is_l2(cuvs::distance::DistanceType metric)
{
  return metric == cuvs::distance::DistanceType::L2Expanded ||
         metric == cuvs::distance::DistanceType::L2SqrtExpanded;
}

}  // namespace

struct centroid_labeler::impl {
  impl(raft::device_resources const& res_,
       cudf::column_view const& centroids_,
       std::int64_t dim_,
       cuvs::distance::DistanceType metric_,
       rmm::device_async_resource_ref mr_)
    : res{res_}, centroids{centroids_}, dim{dim_}, metric{metric_}, mr{mr_}
  {
  }

  raft::device_resources const& res;
  cudf::column_view centroids;
  std::int64_t dim;
  cuvs::distance::DistanceType metric;
  rmm::device_async_resource_ref mr;

  std::int64_t n_centers{0};
  float const* centers{nullptr};
  // The filtered path's view of the centroids; empty when every batch goes to cuVS.
  std::optional<rmm::device_uvector<float>> seeds;
  std::optional<rmm::device_uvector<std::uint16_t>> half;
  std::optional<rmm::device_uvector<float>> half_sq;
  std::optional<rmm::device_uvector<std::int64_t>> identity;
  std::int64_t n_seeds{0};
  float half_error{0}, half_norm{0};

  void predict(float const* x, std::int64_t m, float const* to, std::int64_t n_to, int* labels)
  {
    // cuVS takes no allocator, so its scratch is routed by the ambient resource.
    scoped_current_device_resource const route{mr};
    cuvs::cluster::kmeans::balanced_params params;
    params.metric = metric;
    cuvs::cluster::kmeans::predict(
      res,
      params,
      raft::make_device_matrix_view<float const, std::int64_t>(x, m, dim),
      raft::make_device_matrix_view<float const, std::int64_t>(to, n_to, dim),
      raft::make_device_vector_view<int, std::int64_t>(labels, m));
  }
};

centroid_labeler::centroid_labeler(raft::device_resources const& res,
                                   cudf::column_view const& centroids,
                                   std::int64_t dim,
                                   cuvs::distance::DistanceType metric,
                                   rmm::device_async_resource_ref mr)
  : _impl{std::make_unique<impl>(res, centroids, dim, metric, mr)}
{
  auto& s            = *_impl;
  auto const centers = list_column_as_dataset_view(centroids, dim);
  s.n_centers        = centers.extent(0);
  s.centers          = centers.data_handle();
  if (s.n_centers == 0) { throw std::invalid_argument("centroid_labeler: no centroids"); }
  auto const* off = std::getenv("SIRIUS_VSS_BOUND_GEMM");
  if ((off != nullptr && std::strcmp(off, "0") == 0) || !is_l2(metric) ||
      !bound_filter_f16_supports(dim) || s.n_centers < kMinFilteredCentroids) {
    return;
  }
  auto const stream = raft::resource::get_cuda_stream(res);
  auto const n      = s.n_centers;
  rmm::device_uvector<std::uint16_t> half(static_cast<std::size_t>(n * dim), stream, mr);
  rmm::device_uvector<float> half_sq(static_cast<std::size_t>(n), stream, mr);
  rmm::device_uvector<unsigned int> extremes(2, stream, mr);
  CUDF_CUDA_TRY(cudaMemsetAsync(extremes.data(), 0, 2 * sizeof(unsigned int), stream.value()));
  narrow_to_float16(s.centers, n * dim, half.data(), stream);
  half_rows_norms(s.centers,
                  half.data(),
                  n,
                  dim,
                  half_sq.data(),
                  nullptr,
                  extremes.data(),
                  extremes.data() + 1,
                  stream);
  s.n_seeds = (n + kSeedStride - 1) / kSeedStride;
  std::vector<std::int64_t> picked(static_cast<std::size_t>(s.n_seeds));
  for (std::int64_t i = 0; i < s.n_seeds; ++i) {
    picked[static_cast<std::size_t>(i)] = i * kSeedStride;
  }
  rmm::device_uvector<std::int64_t> picked_d(picked.size(), stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(picked_d.data(),
                                picked.data(),
                                picked.size() * sizeof(std::int64_t),
                                cudaMemcpyHostToDevice,
                                stream.value()));
  rmm::device_uvector<float> seeds(static_cast<std::size_t>(s.n_seeds * dim), stream, mr);
  gather_rows(s.centers, dim, picked_d.data(), s.n_seeds, seeds.data(), stream);
  unsigned int bits[2] = {0, 0};
  CUDF_CUDA_TRY(
    cudaMemcpyAsync(bits, extremes.data(), sizeof(bits), cudaMemcpyDeviceToHost, stream.value()));
  stream.synchronize();
  float max_sq = 0.f;
  std::memcpy(&s.half_error, &bits[0], sizeof(float));
  std::memcpy(&max_sq, &bits[1], sizeof(float));
  s.half_norm = std::sqrt(max_sq);
  // A centroid component past half's range rounds to infinity, and nothing can be bounded.
  if (!(s.half_norm < 65504.f)) { return; }
  s.seeds.emplace(std::move(seeds));
  s.half.emplace(std::move(half));
  s.half_sq.emplace(std::move(half_sq));
}

centroid_labeler::~centroid_labeler() = default;

void centroid_labeler::label(cudf::column_view const& vectors, std::int32_t* labels)
{
  auto& s            = *_impl;
  auto const queries = list_column_as_dataset_view(vectors, s.dim);
  auto const total   = static_cast<std::int64_t>(queries.extent(0));
  if (total == 0) { return; }
  auto const stream = raft::resource::get_cuda_stream(s.res);
  auto const dim    = s.dim;
  if (!s.seeds) {
    s.predict(queries.data_handle(), total, s.centers, s.n_centers, labels);
    stream.synchronize();
    return;
  }
  auto const mr    = s.mr;
  auto const batch = std::min(total, kLabelBatch);
  if (!s.identity || static_cast<std::int64_t>(s.identity->size()) < batch) {
    std::vector<std::int64_t> rows(static_cast<std::size_t>(batch));
    for (std::int64_t i = 0; i < batch; ++i) {
      rows[static_cast<std::size_t>(i)] = i;
    }
    s.identity.emplace(rows.size(), stream, mr);
    CUDF_CUDA_TRY(cudaMemcpyAsync(s.identity->data(),
                                  rows.data(),
                                  rows.size() * sizeof(std::int64_t),
                                  cudaMemcpyHostToDevice,
                                  stream.value()));
    stream.synchronize();
  }
  auto const rows_sz = static_cast<std::size_t>(batch);
  rmm::device_uvector<int> seed(rows_sz, stream, mr);
  rmm::device_uvector<float> bound(rows_sz, stream, mr), limit(rows_sz, stream, mr);
  rmm::device_uvector<float> x_sq(rows_sz, stream, mr), x_error(rows_sz, stream, mr);
  rmm::device_uvector<std::uint16_t> x_half(static_cast<std::size_t>(batch * dim), stream, mr);
  rmm::device_uvector<unsigned> nearest_bits(rows_sz, stream, mr);
  rmm::device_uvector<unsigned long long> best(rows_sz, stream, mr);
  rmm::device_uvector<unsigned long long> unusable(1, stream, mr);
  bound_candidates candidates(batch * kCandidatesPerRow, stream, mr);
  auto const capacity = static_cast<unsigned long long>(candidates.capacity());
  // The FP32 sum of d squared differences is within this of the true distance, relatively.
  float const gamma = 2.f * static_cast<float>(dim) * 5.9604645e-8f;

  for (std::int64_t at = 0; at < total; at += batch) {
    auto const m  = std::min(batch, total - at);
    auto const* x = queries.data_handle() + at * dim;
    s.predict(x, m, s.seeds->data(), s.n_seeds, seed.data());
    seed_bound_kernel<<<label_grid(m), kLabelBlock, 0, stream.value()>>>(
      x, m, dim, s.seeds->data(), seed.data(), 1.f + 2.f * gamma, bound.data());
    CUDF_CHECK_CUDA(stream.value());
    narrow_to_float16(x, m * dim, x_half.data(), stream);
    half_rows_norms(
      x, x_half.data(), m, dim, x_sq.data(), x_error.data(), nullptr, nullptr, stream);
    float16_bound_limit(bound.data(),
                        x_sq.data(),
                        x_error.data(),
                        s.half_error,
                        s.half_norm,
                        dim,
                        m,
                        limit.data(),
                        stream);
    CUDF_CUDA_TRY(
      cudaMemsetAsync(candidates.count.data(), 0, sizeof(unsigned long long), stream.value()));
    CUDF_CUDA_TRY(cudaMemsetAsync(unusable.data(), 0, sizeof(unsigned long long), stream.value()));
    reset_rows_kernel<<<label_grid(m), kLabelBlock, 0, stream.value()>>>(
      nearest_bits.data(), best.data(), m);
    bound_filter_f16(s.half->data(),
                     s.half_sq->data(),
                     s.n_centers,
                     0,
                     x_half.data(),
                     x_sq.data(),
                     s.identity->data(),
                     m,
                     dim,
                     limit.data(),
                     float16_slack{},
                     candidates,
                     stream);
    auto const grid = label_grid(static_cast<std::int64_t>(capacity));
    nearest_filtered_kernel<<<grid, kLabelBlock, 0, stream.value()>>>(candidates.rows.data(),
                                                                      candidates.distances.data(),
                                                                      candidates.count.data(),
                                                                      capacity,
                                                                      nearest_bits.data(),
                                                                      unusable.data());
    rescore_kernel<<<grid, kLabelBlock, 0, stream.value()>>>(x,
                                                             dim,
                                                             s.centers,
                                                             candidates.rows.data(),
                                                             candidates.ids.data(),
                                                             candidates.distances.data(),
                                                             candidates.count.data(),
                                                             capacity,
                                                             nearest_bits.data(),
                                                             bound.data(),
                                                             limit.data(),
                                                             1.f + 4.f * gamma + 1e-6f,
                                                             best.data());
    take_labels_kernel<<<label_grid(m), kLabelBlock, 0, stream.value()>>>(
      best.data(), m, labels + at, unusable.data());
    CUDF_CHECK_CUDA(stream.value());
    unsigned long long passed = 0, bad = 0;
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      &passed, candidates.count.data(), sizeof(passed), cudaMemcpyDeviceToHost, stream.value()));
    CUDF_CUDA_TRY(
      cudaMemcpyAsync(&bad, unusable.data(), sizeof(bad), cudaMemcpyDeviceToHost, stream.value()));
    stream.synchronize();
    // More candidates than the buffer holds, a value past half's range, or a device without
    // the tensor-core kernel: this batch is ranked in FP32.
    if (passed > capacity || bad != 0) {
      s.predict(x, m, s.centers, s.n_centers, labels + at);
      stream.synchronize();
    }
  }
}

}  // namespace sirius::vss
