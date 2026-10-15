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

// test
#include <catch.hpp>

// sirius
#include <vss/cluster_lists.hpp>
#include <vss/vector_clustering.hpp>

// raft
#include <raft/core/device_resources.hpp>

// cudf
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

// rmm
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

using sirius::vss::assign_to_centroids;
using sirius::vss::assignment_spec;
using sirius::vss::clustering_spec;
using sirius::vss::resolve_n_clusters;
using sirius::vss::resolve_train_rows;
using sirius::vss::train_centroids;
using Metric = cuvs::distance::DistanceType;

// A Sirius ARRAY<FLOAT>[dim] column: cudf LIST over a contiguous, uniform FLOAT32 child.
std::unique_ptr<cudf::column> make_vectors(std::vector<float> const& values, cudf::size_type dim)
{
  auto const n_rows = static_cast<cudf::size_type>(values.size()) / dim;
  auto child        = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::FLOAT32}, n_rows * dim, cudf::mask_state::UNALLOCATED);
  cudaMemcpy(child->mutable_view().data<float>(),
             values.data(),
             sizeof(float) * values.size(),
             cudaMemcpyHostToDevice);

  std::vector<std::int32_t> offsets(n_rows + 1);
  for (cudf::size_type i = 0; i <= n_rows; ++i) {
    offsets[i] = i * dim;
  }
  auto offsets_col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, n_rows + 1, cudf::mask_state::UNALLOCATED);
  cudaMemcpy(offsets_col->mutable_view().data<std::int32_t>(),
             offsets.data(),
             sizeof(std::int32_t) * offsets.size(),
             cudaMemcpyHostToDevice);

  return cudf::make_lists_column(
    n_rows, std::move(offsets_col), std::move(child), 0, rmm::device_buffer{});
}

template <typename T>
std::vector<T> to_host(cudf::column_view const& col)
{
  std::vector<T> host(col.size());
  cudaMemcpy(host.data(), col.data<T>(), sizeof(T) * host.size(), cudaMemcpyDeviceToHost);
  return host;
}

// Two tight, well-separated groups in 2-D: rows 0-3 near the origin, rows 4-7 near (10,10).
std::vector<float> two_group_dataset()
{
  return {0.0f,
          0.0f,
          1.0f,
          0.0f,
          0.0f,
          1.0f,
          1.0f,
          1.0f,
          10.0f,
          10.0f,
          11.0f,
          10.0f,
          10.0f,
          11.0f,
          11.0f,
          11.0f};
}

}  // namespace

TEST_CASE("resolve_n_clusters defaults to sqrt(n_rows) and clamps", "[vss]")
{
  CHECK(resolve_n_clusters(0, 10000) == 100);
  CHECK(resolve_n_clusters(7, 10000) == 7);
  CHECK(resolve_n_clusters(0, 0) == 0);
  // A request larger than the table cannot be honoured: k-means needs a row per centroid.
  CHECK(resolve_n_clusters(50, 8) == 8);
  CHECK(resolve_n_clusters(0, 1) == 1);
}

TEST_CASE("resolve_train_rows never trains on fewer rows than centroids", "[vss]")
{
  CHECK(resolve_train_rows(0, 1000000, 100) == 25600);
  CHECK(resolve_train_rows(5000, 1000000, 100) == 5000);
  // Defaults that overshoot the table fall back to the whole table, not to an error.
  CHECK(resolve_train_rows(0, 1000, 100) == 1000);
  // A request below the centroid count would leave centroids untrainable.
  CHECK(resolve_train_rows(10, 1000, 100) == 100);
}

TEST_CASE("train_centroids places one centroid per separated group", "[vss]")
{
  constexpr cudf::size_type dim = 2;
  auto const stream             = cudf::get_default_stream();
  auto const mr                 = cudf::get_current_device_resource_ref();
  raft::device_resources res{stream};

  auto const vectors = make_vectors(two_group_dataset(), dim);

  clustering_spec spec;
  spec.n_clusters = 2;
  spec.metric     = Metric::L2SqrtExpanded;
  auto centroids  = train_centroids({vectors->view()}, dim, spec, stream, mr);

  REQUIRE(centroids->size() == 2);
  auto const values = to_host<float>(cudf::lists_column_view(centroids->view()).child());
  REQUIRE(values.size() == 4);

  // Which centroid lands on which group is not fixed, so check the set rather than the order.
  auto const near_origin   = [](float x, float y) { return x < 5.0f && y < 5.0f; };
  auto const near_ten      = [](float x, float y) { return x > 5.0f && y > 5.0f; };
  bool const first_origin  = near_origin(values[0], values[1]) && near_ten(values[2], values[3]);
  bool const second_origin = near_origin(values[2], values[3]) && near_ten(values[0], values[1]);
  CHECK((first_origin || second_origin));
}

TEST_CASE("train_centroids samples across every chunk", "[vss]")
{
  constexpr cudf::size_type dim = 2;
  auto const stream             = cudf::get_default_stream();
  auto const mr                 = cudf::get_current_device_resource_ref();
  raft::device_resources res{stream};

  // One group per chunk, so a fit that read only the first chunk could not place a centroid
  // on the second group.
  auto const chunk_a = make_vectors({0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f}, dim);
  auto const chunk_b = make_vectors({10.0f, 10.0f, 11.0f, 10.0f, 10.0f, 11.0f, 11.0f, 11.0f}, dim);

  clustering_spec spec;
  spec.n_clusters = 2;
  spec.train_rows = 4;
  spec.metric     = Metric::L2SqrtExpanded;
  auto centroids  = train_centroids({chunk_a->view(), chunk_b->view()}, dim, spec, stream, mr);

  REQUIRE(centroids->size() == 2);
  auto const values = to_host<float>(cudf::lists_column_view(centroids->view()).child());
  auto const max_x  = std::max(values[0], values[2]);
  CHECK(max_x > 5.0f);
}

TEST_CASE("train_centroids rejects an empty dataset", "[vss]")
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  clustering_spec spec;
  spec.n_clusters = 2;
  CHECK_THROWS_AS(train_centroids({}, 2, spec, stream, mr), std::invalid_argument);
}

TEST_CASE("assign_to_centroids emits one edge per row at n_probes 1", "[vss]")
{
  constexpr cudf::size_type dim = 2;
  auto const stream             = cudf::get_default_stream();
  auto const mr                 = cudf::get_current_device_resource_ref();
  raft::device_resources res{stream};

  auto const vectors   = make_vectors(two_group_dataset(), dim);
  auto const centroids = make_vectors({0.5f, 0.5f, 10.5f, 10.5f}, dim);

  assignment_spec spec;
  spec.n_probes         = 1;
  auto const assignment = assign_to_centroids(
    res, vectors->view(), centroids->view(), dim, spec, 0, Metric::L2SqrtExpanded, stream, mr);
  stream.synchronize();

  REQUIRE(assignment.row_ids->size() == 8);
  auto const rows     = to_host<std::int64_t>(assignment.row_ids->view());
  auto const clusters = to_host<std::int32_t>(assignment.cluster_ids->view());
  for (std::int64_t i = 0; i < 8; ++i) {
    CHECK(rows[i] == i);
    CHECK(clusters[i] == (i < 4 ? 0 : 1));
  }
}

TEST_CASE("assign_to_centroids offsets row ids by the chunk base", "[vss]")
{
  constexpr cudf::size_type dim = 2;
  auto const stream             = cudf::get_default_stream();
  auto const mr                 = cudf::get_current_device_resource_ref();
  raft::device_resources res{stream};

  auto const vectors   = make_vectors({0.0f, 0.0f, 10.0f, 10.0f}, dim);
  auto const centroids = make_vectors({0.5f, 0.5f, 10.5f, 10.5f}, dim);

  assignment_spec spec;
  spec.n_probes         = 1;
  auto const assignment = assign_to_centroids(
    res, vectors->view(), centroids->view(), dim, spec, 1000, Metric::L2SqrtExpanded, stream, mr);
  stream.synchronize();

  auto const rows = to_host<std::int64_t>(assignment.row_ids->view());
  REQUIRE(rows.size() == 2);
  CHECK(rows[0] == 1000);
  CHECK(rows[1] == 1001);
}

TEST_CASE("assign_to_centroids repeats a row once per probe and orders by distance", "[vss]")
{
  constexpr cudf::size_type dim = 2;
  auto const stream             = cudf::get_default_stream();
  auto const mr                 = cudf::get_current_device_resource_ref();
  raft::device_resources res{stream};

  auto const vectors   = make_vectors({0.0f, 0.0f}, dim);
  auto const centroids = make_vectors({0.5f, 0.5f, 10.5f, 10.5f, 100.5f, 100.5f}, dim);

  assignment_spec spec;
  spec.n_probes         = 3;
  auto const assignment = assign_to_centroids(
    res, vectors->view(), centroids->view(), dim, spec, 0, Metric::L2SqrtExpanded, stream, mr);
  stream.synchronize();

  REQUIRE(assignment.row_ids->size() == 3);
  auto const rows      = to_host<std::int64_t>(assignment.row_ids->view());
  auto const clusters  = to_host<std::int32_t>(assignment.cluster_ids->view());
  auto const distances = to_host<float>(assignment.distances->view());
  CHECK(rows == std::vector<std::int64_t>{0, 0, 0});
  CHECK(clusters == std::vector<std::int32_t>{0, 1, 2});
  CHECK(distances[0] < distances[1]);
  CHECK(distances[1] < distances[2]);
}

TEST_CASE("assign_to_centroids caps probes at the centroid count", "[vss]")
{
  constexpr cudf::size_type dim = 2;
  auto const stream             = cudf::get_default_stream();
  auto const mr                 = cudf::get_current_device_resource_ref();
  raft::device_resources res{stream};

  auto const vectors   = make_vectors({0.0f, 0.0f}, dim);
  auto const centroids = make_vectors({0.5f, 0.5f, 10.5f, 10.5f}, dim);

  assignment_spec spec;
  spec.n_probes         = 10;
  auto const assignment = assign_to_centroids(
    res, vectors->view(), centroids->view(), dim, spec, 0, Metric::L2SqrtExpanded, stream, mr);
  stream.synchronize();

  CHECK(assignment.row_ids->size() == 2);
}

TEST_CASE("assign_to_centroids radius mode keeps only near-tied centroids", "[vss]")
{
  constexpr cudf::size_type dim = 2;
  auto const stream             = cudf::get_default_stream();
  auto const mr                 = cudf::get_current_device_resource_ref();
  raft::device_resources res{stream};

  // The query sits at (0,0): centroid 0 is at distance 1, centroid 1 at 1.1, centroid 2 at 50.
  auto const vectors   = make_vectors({0.0f, 0.0f}, dim);
  auto const centroids = make_vectors({1.0f, 0.0f, 1.1f, 0.0f, 50.0f, 0.0f}, dim);

  assignment_spec spec;
  spec.radius_factor    = 0.2;  // keeps everything within 1.2x of the nearest
  spec.max_probes       = 3;
  auto const assignment = assign_to_centroids(
    res, vectors->view(), centroids->view(), dim, spec, 0, Metric::L2SqrtExpanded, stream, mr);
  stream.synchronize();

  REQUIRE(assignment.row_ids->size() == 2);
  auto const clusters = to_host<std::int32_t>(assignment.cluster_ids->view());
  CHECK(clusters == std::vector<std::int32_t>{0, 1});
}

TEST_CASE("assign_to_centroids radius mode always keeps the nearest centroid", "[vss]")
{
  constexpr cudf::size_type dim = 2;
  auto const stream             = cudf::get_default_stream();
  auto const mr                 = cudf::get_current_device_resource_ref();
  raft::device_resources res{stream};

  auto const vectors   = make_vectors({0.0f, 0.0f, 100.0f, 100.0f}, dim);
  auto const centroids = make_vectors({1.0f, 0.0f, 50.0f, 0.0f, 90.0f, 90.0f}, dim);

  // A zero radius admits nothing beyond the nearest, which is the tightest setting that must
  // still leave every row present in the assignment.
  assignment_spec spec;
  spec.radius_factor    = 0.000001;
  spec.max_probes       = 3;
  auto const assignment = assign_to_centroids(
    res, vectors->view(), centroids->view(), dim, spec, 0, Metric::L2SqrtExpanded, stream, mr);
  stream.synchronize();

  REQUIRE(assignment.row_ids->size() == 2);
  auto const rows     = to_host<std::int64_t>(assignment.row_ids->view());
  auto const clusters = to_host<std::int32_t>(assignment.cluster_ids->view());
  CHECK(rows == std::vector<std::int64_t>{0, 1});
  CHECK(clusters == std::vector<std::int32_t>{0, 2});
}

TEST_CASE("assign_to_centroids round-trips centroids trained by train_centroids", "[vss]")
{
  constexpr cudf::size_type dim = 2;
  auto const stream             = cudf::get_default_stream();
  auto const mr                 = cudf::get_current_device_resource_ref();
  raft::device_resources res{stream};

  auto const vectors = make_vectors(two_group_dataset(), dim);

  clustering_spec spec;
  spec.n_clusters      = 2;
  spec.metric          = Metric::L2SqrtExpanded;
  auto const centroids = train_centroids({vectors->view()}, dim, spec, stream, mr);

  assignment_spec assign;
  assign.n_probes       = 1;
  auto const assignment = assign_to_centroids(
    res, vectors->view(), centroids->view(), dim, assign, 0, spec.metric, stream, mr);
  stream.synchronize();

  REQUIRE(assignment.cluster_ids->size() == 8);
  auto const clusters = to_host<std::int32_t>(assignment.cluster_ids->view());
  // Each separated group must land wholly in one cluster, and the two groups in different ones.
  CHECK(std::all_of(
    clusters.begin(), clusters.begin() + 4, [&](std::int32_t c) { return c == clusters[0]; }));
  CHECK(std::all_of(
    clusters.begin() + 4, clusters.end(), [&](std::int32_t c) { return c == clusters[4]; }));
  CHECK(clusters[0] != clusters[4]);
}

// The list build square-roots whichever maximum these report into max_row_norm, which bounds the
// FP16 filter; both must report the squared norm, or rows of norm above 1 get too tight a bound.
TEST_CASE("half and FP32 row norms report the same largest squared norm", "[vss]")
{
  constexpr std::int64_t dim  = 16;
  constexpr std::int64_t rows = 2;
  auto const stream           = cudf::get_default_stream();
  // Integer components round to half exactly, so both norms are exactly 3^2 and 40^2.
  std::vector<float> host(rows * dim, 0.f);
  host[0]       = 3.f;
  host[dim + 1] = 40.f;

  rmm::device_uvector<float> x(host.size(), stream);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    x.data(), host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice, stream.value()));
  rmm::device_uvector<std::uint16_t> h(host.size(), stream);
  sirius::vss::narrow_to_float16(x.data(), rows * dim, h.data(), stream);
  rmm::device_uvector<float> sq(rows, stream);
  rmm::device_uvector<unsigned int> bits(2, stream);
  CUDF_CUDA_TRY(cudaMemsetAsync(bits.data(), 0, 2 * sizeof(unsigned int), stream.value()));

  sirius::vss::half_rows_norms(
    x.data(), h.data(), rows, dim, sq.data(), nullptr, nullptr, bits.data(), stream);
  sirius::vss::float_row_sq_norms(x.data(), rows, dim, sq.data(), bits.data() + 1, stream);

  std::vector<unsigned int> out(2);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    out.data(), bits.data(), 2 * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream.value()));
  stream.synchronize();
  float half_max = 0.f, fp32_max = 0.f;
  std::memcpy(&half_max, &out[0], sizeof(float));
  std::memcpy(&fp32_max, &out[1], sizeof(float));
  CHECK(half_max == 1600.f);
  CHECK(fp32_max == 1600.f);
}

namespace {

// The label the list build promises: the centroid at the smallest FP32 squared distance, summed
// in component order with fused multiply-adds, the lowest-numbered of equals.
std::vector<std::int32_t> nearest_by_hand(std::vector<float> const& rows,
                                          std::vector<float> const& centers,
                                          std::int64_t dim)
{
  auto const n = static_cast<std::int64_t>(rows.size()) / dim;
  auto const k = static_cast<std::int64_t>(centers.size()) / dim;
  std::vector<std::int32_t> labels(static_cast<std::size_t>(n));
  for (std::int64_t r = 0; r < n; ++r) {
    float best = INFINITY;
    for (std::int64_t c = 0; c < k; ++c) {
      float acc = 0.f;
      for (std::int64_t j = 0; j < dim; ++j) {
        float const e = rows[static_cast<std::size_t>(r * dim + j)] -
                        centers[static_cast<std::size_t>(c * dim + j)];
        acc = std::fmaf(e, e, acc);
      }
      if (acc < best) {
        best                                = acc;
        labels[static_cast<std::size_t>(r)] = static_cast<std::int32_t>(c);
      }
    }
  }
  return labels;
}

std::vector<std::int32_t> labels_of(std::vector<float> const& rows,
                                    std::vector<float> const& centers,
                                    cudf::size_type dim)
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  raft::device_resources res{stream};
  auto const vectors   = make_vectors(rows, dim);
  auto const centroids = make_vectors(centers, dim);
  sirius::vss::centroid_labeler labeler{res, centroids->view(), dim, Metric::L2Expanded, mr};
  rmm::device_uvector<std::int32_t> labels(static_cast<std::size_t>(vectors->size()), stream);
  labeler.label(vectors->view(), labels.data());
  std::vector<std::int32_t> host(labels.size());
  CUDF_CUDA_TRY(cudaMemcpyAsync(host.data(),
                                labels.data(),
                                host.size() * sizeof(std::int32_t),
                                cudaMemcpyDeviceToHost,
                                stream.value()));
  stream.synchronize();
  return host;
}

}  // namespace

// More rows than one launch ranks, enough centroids for the filtered path, and the cases a
// rounded filter could get wrong: rows that are a centroid, and two centroids that are equal.
TEST_CASE("centroid_labeler returns the centroid at the smallest FP32 distance", "[vss][kmeans]")
{
  constexpr cudf::size_type dim  = 16;
  constexpr std::int64_t n_rows  = 140'000;
  constexpr std::int64_t n_cents = 512;
  std::mt19937 gen(7);
  std::uniform_real_distribution<float> uniform(-1.f, 1.f);
  std::vector<float> centers(static_cast<std::size_t>(n_cents * dim));
  for (auto& v : centers) {
    v = uniform(gen);
  }
  std::copy_n(centers.begin() + 7 * dim, dim, centers.begin() + 300 * dim);
  std::vector<float> rows(static_cast<std::size_t>(n_rows * dim));
  std::normal_distribution<float> noise(0.f, 0.05f);
  for (std::int64_t r = 0; r < n_rows; ++r) {
    auto const c = static_cast<std::int64_t>(gen() % n_cents);
    for (std::int64_t j = 0; j < dim; ++j) {
      // Every fifth row sits near a centroid, the rest anywhere; every thousandth is a centroid.
      rows[static_cast<std::size_t>(r * dim + j)] =
        r % 1000 == 0 ? centers[static_cast<std::size_t>(c * dim + j)]
        : r % 5 == 0  ? centers[static_cast<std::size_t>(c * dim + j)] + noise(gen)
                      : uniform(gen);
    }
  }
  auto const expected = nearest_by_hand(rows, centers, dim);
  auto const labels   = labels_of(rows, centers, dim);
  REQUIRE(labels.size() == expected.size());
  std::int64_t different = 0;
  for (std::size_t r = 0; r < labels.size(); ++r) {
    different += labels[r] != expected[r];
  }
  CHECK(different == 0);
  CHECK(std::count(labels.begin(), labels.end(), 300) == 0);
}

// A row past half's range cannot be bounded on the tensor cores; its batch is ranked in FP32.
TEST_CASE("centroid_labeler ranks a batch with a value outside half's range", "[vss][kmeans]")
{
  constexpr cudf::size_type dim  = 16;
  constexpr std::int64_t n_cents = 256;
  constexpr std::int64_t n_rows  = 1'000;
  std::vector<float> centers(static_cast<std::size_t>(n_cents * dim), 0.f);
  for (std::int64_t c = 0; c < n_cents; ++c) {
    centers[static_cast<std::size_t>(c * dim)] = 100.f * static_cast<float>(c);
  }
  std::vector<float> rows(static_cast<std::size_t>(n_rows * dim), 0.f);
  std::vector<std::int32_t> expected(static_cast<std::size_t>(n_rows));
  for (std::int64_t r = 0; r < n_rows; ++r) {
    auto const c                            = (r * 37) % n_cents;
    rows[static_cast<std::size_t>(r * dim)] = 100.f * static_cast<float>(c) + 20.f;
    expected[static_cast<std::size_t>(r)]   = static_cast<std::int32_t>(c);
  }
  rows[static_cast<std::size_t>(500 * dim)] = 1e6f;
  expected[500]                             = n_cents - 1;
  CHECK(labels_of(rows, centers, dim) == expected);
}

TEST_CASE("scatter_rows moves whole rows of any width", "[vss][kmeans]")
{
  auto const stream = cudf::get_default_stream();
  for (std::int64_t const width : {3, 8, 96}) {
    constexpr std::int64_t n = 1'000;
    std::vector<std::uint8_t> src(static_cast<std::size_t>(n * width));
    for (std::size_t i = 0; i < src.size(); ++i) {
      src[i] = static_cast<std::uint8_t>(i * 31 + 7);
    }
    std::vector<std::int64_t> dest(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i) {
      dest[static_cast<std::size_t>(i)] = (i * 387) % n;  // 387 and 1,000 are coprime
    }
    rmm::device_uvector<std::uint8_t> src_d(src.size(), stream), out_d(src.size(), stream);
    rmm::device_uvector<std::int64_t> dest_d(dest.size(), stream);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      src_d.data(), src.data(), src.size(), cudaMemcpyHostToDevice, stream.value()));
    CUDF_CUDA_TRY(cudaMemcpyAsync(dest_d.data(),
                                  dest.data(),
                                  dest.size() * sizeof(std::int64_t),
                                  cudaMemcpyHostToDevice,
                                  stream.value()));
    sirius::vss::scatter_rows(reinterpret_cast<std::byte const*>(src_d.data()),
                              width,
                              dest_d.data(),
                              n,
                              reinterpret_cast<std::byte*>(out_d.data()),
                              stream);
    std::vector<std::uint8_t> out(src.size());
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      out.data(), out_d.data(), out.size(), cudaMemcpyDeviceToHost, stream.value()));
    stream.synchronize();
    std::int64_t wrong = 0;
    for (std::int64_t i = 0; i < n; ++i) {
      wrong += std::memcmp(out.data() + dest[static_cast<std::size_t>(i)] * width,
                           src.data() + i * width,
                           static_cast<std::size_t>(width)) != 0;
    }
    CAPTURE(width);
    CHECK(wrong == 0);
  }
}
